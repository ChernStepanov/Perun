import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from . import events, models, panel
from .config import REPORTS, STUDY_START

OUT = REPORTS / "results"

ITS_BREAKS = {
    "ChatGPT": "2022-11-28",
    "LLaMA leak": "2023-02-27",
    "DeepSeek-R1": "2025-01-20",
}

REGION_FROM = "2024-01-01"

TECH_OUTCOMES = ["kev_fresh", "kev_ransomware", "exploits", "exploits_cve"]


def _row(name, r: models.LagResult) -> dict:
    b = r.bins
    return {
        "outcome": name,
        **{f"{k}_pct": b.loc[k, "effect_pct"] for k in b.index},
        **{f"{k}_p_hac": b.loc[k, "p"] for k in b.index},
        **{f"{k}_p_placebo": b.loc[k, "p_placebo"] for k in b.index},
        "post_pct": models.pct(r.observed), "p_placebo": r.p_placebo,
    }


def h1_h2(p: pd.DataFrame, rel: pd.DataFrame, n_placebo: int) -> tuple[pd.DataFrame, dict]:
    rows, profiles = [], {}
    for col in panel.outcome_columns(p) + ["victims_by_attack"]:
        y = p[col]
        if col == "victims_by_attack":
            y = y[y.index >= "2023-01-02"]
        if col.startswith("region_"):
            y = y[y.index >= REGION_FROM]
        r = models.lag_study(y, rel["week"], n_placebo=n_placebo)
        rows.append(_row(col, r))
        profiles[col] = r
    tab = pd.DataFrame(rows).set_index("outcome")
    tab["q_bh"] = multipletests(tab["p_placebo"], method="fdr_bh")[1]
    return tab, profiles


def post_vs_attack(p: pd.DataFrame, rel: pd.DataFrame, n_placebo: int) -> pd.DataFrame:
    window = (p.index >= "2023-01-02") & (p.index <= p.index[-1] - pd.Timedelta(weeks=6))
    rows = []
    for col in ("victims", "victims_by_attack"):
        r = models.lag_study(p.loc[window, col], rel["week"], n_placebo=n_placebo)
        b = r.bins[["effect_pct", "p", "p_placebo"]].copy()
        b.insert(0, "series", col)
        rows.append(b)
    return pd.concat(rows).rename_axis("bin").reset_index()


def h3(p: pd.DataFrame, rel: pd.DataFrame, n_placebo: int) -> tuple[pd.DataFrame, dict]:
    y = p["victims"]
    joint = models.open_vs_closed(y, rel)
    rows = []
    for access, other in (("open", "closed"), ("closed", "open")):
        ctrl = models.lag_exposure(y.index, rel.loc[rel["access"] == other, "week"],
                                   models.POST_BINS, f"{other}_")
        base = models.design(y.index).join(ctrl)
        r = models.lag_study(y, rel.loc[rel["access"] == access, "week"], n_placebo, base=base)
        rows.append({"access": access, "n_events": int((rel["access"] == access).sum()),
                     "post_pct": models.pct(r.observed), "p_placebo": r.p_placebo})
    intensity = {}
    for access in ("open", "closed"):
        ctrl = p[[f"llm_releases_{'closed' if access == 'open' else 'open'}"]]
        lagged = pd.DataFrame({f"rel_{access}_l{k}": p[f"llm_releases_{access}"].shift(k, fill_value=0)
                               for k in range(0, 9)})
        X = models.design(y.index).join(ctrl).join(lagged)
        res = models.fit(y, X)
        cols = list(lagged.columns)
        cum = res.params[cols].sum()
        se = np.sqrt(res.cov_params().loc[cols, cols].to_numpy().sum())
        eff, lo, hi = models.pct(cum, se)
        intensity[access] = {"cum_pct_per_release": eff, "ci_lo": lo, "ci_hi": hi}
    return pd.DataFrame(rows).set_index("access"), {"joint": joint, "intensity": intensity}


def h4(p: pd.DataFrame, rel: pd.DataFrame, off: pd.DataFrame, n_placebo: int) -> dict:
    out = {}
    rows = []
    agents = off.loc[off["kind"].isin(["agent", "malicious_llm", "malware", "cyber_model"]), "week"]
    for col in TECH_OUTCOMES:
        y = p[col]
        if col.startswith("kev"):
            y = y[y.index >= "2022-07-04"]
        for label, ev in (("releases", rel["week"]), ("offensive_ai", agents)):
            r = models.lag_study(y, ev, n_placebo=n_placebo)
            rows.append({"outcome": col, "events": label, "post_pct": models.pct(r.observed),
                         "p_placebo": r.p_placebo})
    out["weekly"] = pd.DataFrame(rows)

    k = panel.kev_lags()
    k["period"] = np.where(k["date_added"] < "2022-11-30", "до ChatGPT", "после ChatGPT")
    k["half"] = k["date_added"].dt.year.astype(str) + np.where(k["date_added"].dt.month <= 6, "-H1", "-H2")
    by_half = k.groupby("half").agg(
        n=("cve", "size"),
        median_days=("days_to_kev", "median"),
        within_7d=("days_to_kev", lambda d: (d <= 7).mean() * 100),
        ransomware_share=("ransomware", lambda r: r.mean() * 100))
    out["kev_by_half"] = by_half
    pre = k.loc[k["period"] == "до ChatGPT", "days_to_kev"]
    post = k.loc[k["period"] == "после ChatGPT", "days_to_kev"]
    out["kev_shift"] = models.ttest_shift(pre, post)
    early = k.loc[k["date_added"] < "2025-01-01", "days_to_kev"]
    late = k.loc[k["date_added"] >= "2025-01-01", "days_to_kev"]
    out["kev_shift_2025"] = models.ttest_shift(early, late)
    return out


def structure(v: pd.DataFrame) -> dict:
    v = v[v["discovered"] >= STUDY_START]
    pre = v[v["discovered"] < "2022-11-30"]
    post = v[v["discovered"] >= "2022-11-30"]
    late = v[v["discovered"] >= "2025-01-01"]
    mid = v[(v["discovered"] >= "2022-11-30") & (v["discovered"] < "2025-01-01")]
    y2024 = v[(v["discovered"] >= REGION_FROM) & (v["discovered"] < "2025-01-01")]
    known = lambda d, col: d.loc[~d[col].isin(["Unknown", "Не указан"]), col].value_counts()
    return {
        "sector_chatgpt": models.share_shift(known(pre, "sector"), known(post, "sector")),
        "sector_2025": models.share_shift(known(mid, "sector"), known(late, "sector")),
        "region_2025": models.share_shift(known(y2024, "region"), known(late, "region")),
    }


def h0(p: pd.DataFrame) -> pd.DataFrame:
    y = p["victims"]
    X = models.design(y.index)
    res = models.fit(y, X)
    names = events.confounders()["name"].tolist()
    cols = [c for c in X.columns if c.startswith("c") and c[1:].isdigit()]
    tab = models.effect_table(res, cols)
    tab.index = [names[int(c[1:])] for c in cols]
    no_conf = models.fit(y, models.design(y.index, confounders=False))
    tab.attrs["deviance_with"] = float(res.deviance)
    tab.attrs["deviance_without"] = float(no_conf.deviance)
    xmas = models.effect_table(res, ["xmas"])
    tab.attrs["xmas_pct"] = float(xmas["effect_pct"].iloc[0])
    tab.attrs["xmas_p"] = float(xmas["p"].iloc[0])
    return tab


def its_all(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, date in ITS_BREAKS.items():
        for col in ("victims", "ransomwatch"):
            y = p[col].dropna()
            if col == "ransomwatch":
                y = y[y.index < "2025-03-03"]
            if pd.Timestamp(date) > y.index[-26]:
                continue
            tab = models.its(y, date)
            sens = models.its(y, date, extra=np.log(p.loc[y.index, ["active_groups"]].clip(lower=1)))
            rows.append({"break": name, "series": col,
                         "level_pct": tab.loc["level", "effect_pct"], "level_p": tab.loc["level", "p"],
                         "slope_pct_year": tab.loc["slope", "effect_pct"], "slope_p": tab.loc["slope", "p"],
                         "level_pct_groups_ctrl": sens.loc["level", "effect_pct"],
                         "level_p_groups_ctrl": sens.loc["level", "p"]})
    return pd.DataFrame(rows)


def lag_publication(v: pd.DataFrame) -> pd.DataFrame:
    d = v.dropna(subset=["lag_days"])
    d = d[d["lag_days"] >= 0]
    q = d.groupby(d["discovered"].dt.year)["lag_days"].describe(percentiles=[.25, .5, .75, .9])
    q["share_same_day"] = d.groupby(d["discovered"].dt.year)["lag_days"].apply(lambda s: (s == 0).mean() * 100)
    return q


def run(n_placebo: int = 500) -> dict:
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    p = panel.load()
    rel = events.releases()
    off = events.offensive()
    v = panel.victims()

    res = {}
    res["h1_h2"], res["profiles"] = h1_h2(p, rel, n_placebo)
    res["post_vs_attack"] = post_vs_attack(p, rel, n_placebo)
    res["h3"], res["h3_extra"] = h3(p, rel, n_placebo)
    res["h4"] = h4(p, rel, off, n_placebo)
    res["h0"] = h0(p)
    res["its"] = its_all(p)
    res["its_scan"] = models.its_scan(p["victims"], "2022-03-07", "2024-06-24")
    res["structure"] = structure(v)
    res["pub_lag"] = lag_publication(v)
    res["changepoints"] = models.changepoints(p["victims"])
    res["cp_near"] = models.near_share(res["changepoints"], rel["week"], p.index)
    res["epoch_check"] = events.check_against_epoch()
    both = p[["victims", "ransomwatch"]].dropna().loc[:"2025-03-01"]
    res["validation"] = {
        "corr_ransomwatch_weekly": float(both.corr().iloc[0, 1]),
        "corr_ransomwatch_monthly": float(both.resample("MS").sum().corr().iloc[0, 1]),
        "weeks": len(p), "start": str(p.index[0].date()), "end": str(p.index[-1].date()),
        "victims_total": int(p["victims"].sum()),
    }
    _dump(res)
    return res


def _dump(res: dict) -> None:
    res["h1_h2"].to_csv(OUT / "h1_h2_lag_bins.csv", float_format="%.4f")
    prof = pd.DataFrame({k: r.profile["effect_pct"] for k, r in res["profiles"].items()})
    prof.to_csv(OUT / "h2_lag_profile_pct.csv", float_format="%.3f")
    res["post_vs_attack"].to_csv(OUT / "h2_post_vs_attack_date.csv", float_format="%.4f", index=False)
    res["h3"].to_csv(OUT / "h3_open_closed.csv", float_format="%.4f")
    res["h3_extra"]["joint"]["table"].to_csv(OUT / "h3_joint.csv", float_format="%.4f")
    res["h4"]["weekly"].to_csv(OUT / "h4_technical.csv", float_format="%.4f", index=False)
    res["h4"]["kev_by_half"].to_csv(OUT / "h4_kev_time_to_exploit.csv", float_format="%.2f")
    res["h0"].to_csv(OUT / "h0_confounders.csv", float_format="%.4f")
    res["its"].to_csv(OUT / "its.csv", float_format="%.4f", index=False)
    for k, t in res["structure"].items():
        t.to_csv(OUT / f"structure_{k}.csv", float_format="%.3f")
    res["pub_lag"].to_csv(OUT / "publication_lag.csv", float_format="%.2f")
    res["epoch_check"].to_csv(OUT / "events_vs_epoch.csv", index=False)
    res["its_scan"].to_csv(OUT / "its_scan.csv", float_format="%.3f")
    summary = {
        "validation": res["validation"],
        "h3_wald": {k: res["h3_extra"]["joint"][k] for k in ("wald_stat", "wald_p")},
        "h3_intensity": res["h3_extra"]["intensity"],
        "kev_shift": res["h4"]["kev_shift"],
        "kev_shift_2025": res["h4"]["kev_shift_2025"],
        "h0_deviance": {"with": res["h0"].attrs["deviance_with"],
                        "without": res["h0"].attrs["deviance_without"]},
        "xmas": {"pct": res["h0"].attrs["xmas_pct"], "p": res["h0"].attrs["xmas_p"]},
        "changepoints": [str(d.date()) for d in res["changepoints"]],
        "its_scan_chatgpt_rank": int((res["its_scan"]["z"] >= res["its_scan"].loc["2022-11-28", "z"]).sum()),
        "its_scan_n": len(res["its_scan"]),
        "changepoints_near_release": res["cp_near"],
        "structure_chi2_p": {k: t.attrs["chi2_p"] for k, t in res["structure"].items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
