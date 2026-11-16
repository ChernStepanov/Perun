import numpy as np
import pandas as pd

from .config import EVENTS, week_start
from .sources import epoch


def releases() -> pd.DataFrame:
    df = pd.read_csv(EVENTS / "llm_releases.csv", parse_dates=["date"])
    df["week"] = week_start(df["date"])
    return df


def offensive() -> pd.DataFrame:
    df = pd.read_csv(EVENTS / "ai_offensive.csv", parse_dates=["date"])
    df["week"] = week_start(df["date"])
    return df


def confounders() -> pd.DataFrame:
    return pd.read_csv(EVENTS / "confounders.csv", parse_dates=["start", "end"])


def check_against_epoch(tolerance_days: int = 3) -> pd.DataFrame:
    rel = releases()
    reg = epoch.load()
    out = []
    for r in rel.itertuples():
        stem = r.model.split(" (")[0]
        hit = reg[reg["model"].str.startswith(stem, na=False)]
        if hit.empty:
            out.append((r.model, r.date, pd.NaT, np.nan))
            continue
        nearest = hit.iloc[(hit["date"] - r.date).abs().argsort()[:1]]
        d = nearest["date"].iloc[0]
        out.append((r.model, r.date, d, (r.date - d).days))
    res = pd.DataFrame(out, columns=["model", "date", "epoch_date", "diff_days"])
    res["ok"] = res["diff_days"].abs() <= tolerance_days
    return res


def exposure(weeks: pd.DatetimeIndex, event_weeks: pd.Series, lags=range(-4, 13)) -> pd.DataFrame:
    counts = event_weeks.value_counts().reindex(weeks, fill_value=0)
    cols = {}
    for k in lags:
        cols[f"L{k}"] = counts.shift(k, fill_value=0).to_numpy()
    return pd.DataFrame(cols, index=weeks)


def binned(exp: pd.DataFrame, bins: dict[str, range]) -> pd.DataFrame:
    return pd.DataFrame({name: exp[[f"L{k}" for k in ks]].sum(axis=1) for name, ks in bins.items()},
                        index=exp.index)


LAG_BINS = {
    "pre": range(-4, 0),
    "w0_3": range(0, 4),
    "w4_7": range(4, 8),
    "w8_12": range(8, 13),
}
