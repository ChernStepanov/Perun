from dataclasses import dataclass

import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
from patsy import dmatrix
from scipy import stats

from . import events
from .config import LAGS

HAC = {"cov_type": "HAC", "cov_kwds": {"maxlags": 4}}


def calendar(index: pd.DatetimeIndex, spline_df: int | None = None) -> pd.DataFrame:
    n = len(index)
    t = np.arange(n)
    X = pd.DataFrame(index=index)
    if spline_df is None:
        spline_df = max(4, round(n / 26))
    if spline_df:
        S = dmatrix(f"cr(t, df={spline_df}) - 1", {"t": t}, return_type="dataframe")
        S.index = index
        S.columns = [f"s{i}" for i in range(S.shape[1])]
        X = X.join(S)
    else:
        X["const"] = 1.0
        X["t"] = t / 52.0
    woy = index.isocalendar().week.to_numpy().astype(float)
    for k in (1, 2):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * woy / 52.18)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * woy / 52.18)
    end = index + pd.Timedelta(days=6)
    xmas = ((index.month == 12) & (end.day >= 25)) | ((index.month == 12) & (index.day >= 25)) | \
           ((index.month == 1) & (index.day <= 1)) | ((end.month == 1) & (end.year > index.year))
    X["xmas"] = xmas.astype(float)
    return X


def confounder_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for i, c in enumerate(events.confounders().itertuples()):
        s = c.start - pd.Timedelta(days=c.start.day_of_week)
        out[f"c{i}"] = ((index >= s) & (index <= c.end)).astype(float)
    return out.loc[:, out.sum() > 0]


def design(index: pd.DatetimeIndex, spline_df: int | None = None, confounders: bool = True,
           extra: pd.DataFrame | None = None) -> pd.DataFrame:
    X = calendar(index, spline_df)
    if confounders:
        X = X.join(confounder_dummies(index))
    if extra is not None:
        X = X.join(extra)
    return X


def fit(y: pd.Series, X: pd.DataFrame):
    ok = y.notna()
    return sm.GLM(y[ok], X[ok], family=sm.families.Poisson()).fit(**HAC)


def pct(beta, se=None):
    if se is None:
        return 100 * (np.exp(beta) - 1)
    return 100 * (np.exp(beta) - 1), 100 * (np.exp(beta - 1.96 * se) - 1), 100 * (np.exp(beta + 1.96 * se) - 1)


def effect_table(res, cols: list[str]) -> pd.DataFrame:
    b, se, p = res.params[cols], res.bse[cols], res.pvalues[cols]
    eff, lo, hi = pct(b, se)
    return pd.DataFrame({"beta": b, "se": se, "p": p, "effect_pct": eff, "ci_lo": lo, "ci_hi": hi})


def lag_exposure(index: pd.DatetimeIndex, event_weeks: pd.Series, bins=None,
                 prefix: str = "") -> pd.DataFrame:
    exp = events.exposure(index, event_weeks, LAGS)
    if bins is None:
        bins = events.LAG_BINS
    out = events.binned(exp, bins)
    out.columns = [prefix + c for c in out.columns]
    return out


POST_BINS = {"pre": range(-4, 0), "post": range(0, 13)}


@dataclass
class LagResult:
    profile: pd.DataFrame
    bins: pd.DataFrame
    placebo: np.ndarray
    observed: float
    p_placebo: float


def lag_study(y: pd.Series, event_weeks: pd.Series, n_placebo: int = 500, seed: int = 7,
              base: pd.DataFrame | None = None) -> LagResult:
    index = y.index
    base = design(index) if base is None else base

    full = events.exposure(index, event_weeks, LAGS)
    res = fit(y, base.join(full))
    profile = effect_table(res, list(full.columns))
    profile.index = [int(c[1:]) for c in profile.index]

    binned = lag_exposure(index, event_weeks)
    bins = effect_table(fit(y, base.join(binned)), list(binned.columns))

    post = lag_exposure(index, event_weeks, POST_BINS)
    observed = fit(y, base.join(post)).params["post"]
    placebo, placebo_bins = circular_placebo(y, base, event_weeks, n_placebo, seed)
    p = _two_sided(placebo, observed)
    bins["p_placebo"] = [_two_sided(placebo_bins[:, j], bins["beta"].iloc[j]) for j in range(len(bins))]
    return LagResult(profile, bins, placebo, observed, p)


def _two_sided(null: np.ndarray, obs: float) -> float:
    return float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)))


def shift_weeks(event_weeks: pd.Series, index: pd.DatetimeIndex, offset: int) -> pd.Series:
    pos = index.get_indexer(event_weeks)
    pos = pos[pos >= 0]
    return pd.Series(index[(pos + offset) % len(index)])


def circular_placebo(y: pd.Series, base: pd.DataFrame, event_weeks: pd.Series, n: int,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    index = y.index
    rng = np.random.default_rng(seed)
    margin = max(LAGS) + 1
    offsets = rng.integers(margin, len(index) - margin, size=n)
    post_null = np.empty(n)
    bins_null = np.empty((n, len(events.LAG_BINS)))
    for i, off in enumerate(offsets):
        fake = shift_weeks(event_weeks, index, int(off))
        post = lag_exposure(index, fake, POST_BINS)
        post_null[i] = fit(y, base.join(post)).params["post"]
        binned = lag_exposure(index, fake)
        bins_null[i] = fit(y, base.join(binned)).params[list(binned.columns)].to_numpy()
    return post_null, bins_null


def open_vs_closed(y: pd.Series, rel: pd.DataFrame, base: pd.DataFrame | None = None) -> dict:
    index = y.index
    base = design(index) if base is None else base
    eo = lag_exposure(index, rel.loc[rel["access"] == "open", "week"], POST_BINS, "open_")
    ec = lag_exposure(index, rel.loc[rel["access"] == "closed", "week"], POST_BINS, "closed_")
    res = fit(y, base.join(eo).join(ec))
    tab = effect_table(res, ["open_pre", "open_post", "closed_pre", "closed_post"])
    w = res.wald_test("open_post = closed_post", scalar=True)
    return {"table": tab, "wald_stat": float(w.statistic), "wald_p": float(w.pvalue)}


def its(y: pd.Series, break_date: str, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    index = y.index
    X = design(index, spline_df=0, extra=extra)
    b = pd.Timestamp(break_date)
    post = (index >= b).astype(float)
    X["level"] = post
    X["slope"] = post * (np.arange(len(index)) - index.get_indexer([b], method="bfill")[0]) / 52.0
    res = fit(y, X)
    return effect_table(res, ["t", "level", "slope"])


def abnormal(y: pd.Series, base: pd.DataFrame | None = None) -> pd.Series:
    base = design(y.index) if base is None else base
    res = fit(y, base)
    mu = res.predict(base)
    return np.log((y + 0.5) / (mu + 0.5))


def event_car(ar: pd.Series, event_weeks: pd.Series, pre=(-4, -1), post=(0, 4)) -> pd.DataFrame:
    rows = []
    idx = ar.index
    for w in event_weeks:
        i = idx.get_indexer([w])[0]
        if i < 0 or i + pre[0] < 0 or i + post[1] >= len(idx):
            rows.append((w, np.nan, np.nan))
            continue
        before = ar.iloc[i + pre[0]: i + pre[1] + 1].mean()
        after = ar.iloc[i + post[0]: i + post[1] + 1].mean()
        rows.append((w, before, after))
    out = pd.DataFrame(rows, columns=["week", "ar_pre", "ar_post"])
    out["diff_pct"] = pct(out["ar_post"] - out["ar_pre"])
    return out


def its_scan(y: pd.Series, start: str, end: str) -> pd.DataFrame:
    rows = []
    for d in y.index[(y.index >= start) & (y.index <= end)]:
        tab = its(y, str(d.date()))
        rows.append((d, tab.loc["level", "effect_pct"], tab.loc["level", "beta"] / tab.loc["level", "se"]))
    return pd.DataFrame(rows, columns=["week", "level_pct", "z"]).set_index("week")


def changepoints(y: pd.Series, min_size: int = 8, pen: float = 1.0) -> list[pd.Timestamp]:
    X = calendar(y.index, spline_df=0)
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    signal = np.log1p(y.to_numpy()) - np.log1p(res.predict(X).to_numpy()) + np.log1p(res.predict(X).to_numpy()).mean()
    sigma = np.median(np.abs(np.diff(signal))) / 0.6745 / np.sqrt(2)
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal.reshape(-1, 1))
    bk = algo.predict(pen=pen * np.log(len(signal)) * sigma ** 2)
    return [y.index[b] for b in bk[:-1]]


def near_share(dates: list[pd.Timestamp], event_weeks: pd.Series, index: pd.DatetimeIndex,
               radius: int = 4) -> tuple[float, float]:
    ev = index.get_indexer(event_weeks)
    ev = ev[ev >= 0]
    covered = np.zeros(len(index), bool)
    for e in ev:
        covered[max(0, e - radius): e + radius + 1] = True
    pos = index.get_indexer(dates)
    hit = covered[pos].mean() if len(pos) else np.nan
    return float(hit), float(covered.mean())


def ttest_shift(a: pd.Series, b: pd.Series) -> dict:
    u = stats.mannwhitneyu(a.dropna(), b.dropna())
    return {"median_a": float(a.median()), "median_b": float(b.median()),
            "n_a": int(a.notna().sum()), "n_b": int(b.notna().sum()), "p": float(u.pvalue)}


def share_shift(counts_pre: pd.Series, counts_post: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"pre": counts_pre, "post": counts_post}).fillna(0)
    chi = stats.chi2_contingency(df.to_numpy().T)
    df["share_pre"] = df["pre"] / df["pre"].sum() * 100
    df["share_post"] = df["post"] / df["post"].sum() * 100
    df["delta_pp"] = df["share_post"] - df["share_pre"]
    df.attrs["chi2_p"] = float(chi.pvalue)
    return df.sort_values("delta_pp")
