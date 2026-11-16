import numpy as np
import pandas as pd

from . import events
from .config import MIN_WEEKLY_MEAN, PROCESSED, STUDY_START, week_start
from .regions import region_of
from .sources import epoch, exploitdb, kev, ransomware_live, ransomwatch

SECTOR_SLUG = {
    "Professional Services": "professional",
    "Manufacturing": "manufacturing",
    "Technology": "technology",
    "Healthcare": "healthcare",
    "Retail & E-Commerce": "retail",
    "Financial Services": "finance",
    "Transportation": "transport",
    "Education": "education",
    "Agriculture and Food Production": "agrifood",
    "Government & Defense": "government",
    "Energy & Utilities": "energy",
    "Hospitality": "hospitality",
    "Unknown": "unknown",
}
REGION_SLUG = {
    "Северная Америка": "north_america",
    "Европа": "europe",
    "Азия и Океания": "asia_pacific",
    "Латинская Америка": "latam",
    "Ближний Восток и Африка": "mea",
    "СНГ": "cis",
    "Не указан": "unknown",
}


def _weekly(dates: pd.Series, weeks: pd.DatetimeIndex) -> pd.Series:
    return week_start(dates).value_counts().reindex(weeks, fill_value=0)


def _weeks(victims: pd.DataFrame) -> pd.DatetimeIndex:
    last = week_start(victims["discovered"]).max()
    return pd.date_range(STUDY_START, last - pd.Timedelta(weeks=1), freq="W-MON")


def victims() -> pd.DataFrame:
    df = ransomware_live.load()
    df["region"] = df["country"].map(region_of)
    df["week"] = week_start(df["discovered"])
    return df


def build() -> pd.DataFrame:
    v = victims()
    weeks = _weeks(v)
    out = pd.DataFrame(index=weeks)
    out.index.name = "week"

    out["victims"] = _weekly(v["discovered"], weeks)
    out["victims_by_attack"] = _weekly(v["attackdate"].dropna(), weeks)
    out["active_groups"] = v.groupby("week")["group_name"].nunique().reindex(weeks, fill_value=0)
    top_share = v.groupby("week")["group_name"].agg(lambda g: g.value_counts(normalize=True).iloc[0])
    out["top_group_share"] = top_share.reindex(weeks)

    for name, slug in REGION_SLUG.items():
        out[f"region_{slug}"] = _weekly(v.loc[v["region"] == name, "discovered"], weeks)
    for name, slug in SECTOR_SLUG.items():
        out[f"sector_{slug}"] = _weekly(v.loc[v["sector"] == name, "discovered"], weeks)

    rw = ransomwatch.load()
    rw_weeks = _weekly(rw["discovered"], weeks)
    rw_last = week_start(rw["discovered"]).max()
    out["ransomwatch"] = rw_weeks.where(weeks < rw_last)

    k = kev.load()
    out["kev"] = _weekly(k["date_added"], weeks)
    out["kev_ransomware"] = _weekly(k.loc[k["ransomware"], "date_added"], weeks)
    out["kev_fresh"] = _weekly(k.loc[k["fresh"], "date_added"], weeks)

    x = exploitdb.load()
    out["exploits"] = _weekly(x.loc[~x["metasploit"], "date_published"], weeks)
    out["exploits_cve"] = _weekly(x.loc[~x["metasploit"] & x["has_cve"], "date_published"], weeks)

    m = epoch.load()
    out["llm_releases_open"] = _weekly(m.loc[m["access"] == "open", "date"], weeks)
    out["llm_releases_closed"] = _weekly(m.loc[m["access"] == "closed", "date"], weeks)

    rel = events.releases()
    out["flagship_open"] = rel.loc[rel["access"] == "open", "week"].value_counts().reindex(weeks, fill_value=0)
    out["flagship_closed"] = rel.loc[rel["access"] == "closed", "week"].value_counts().reindex(weeks, fill_value=0)
    return out


def outcome_columns(panel: pd.DataFrame) -> list[str]:
    cols = ["victims"]
    for prefix in ("region_", "sector_"):
        for c in panel.columns:
            if c.startswith(prefix) and not c.endswith("unknown") and panel[c].mean() >= MIN_WEEKLY_MEAN:
                cols.append(c)
    return cols


def save(panel: pd.DataFrame) -> None:
    panel.to_csv(PROCESSED / "weekly.csv", float_format="%.4g")


def load() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "weekly.csv", index_col="week", parse_dates=["week"]).asfreq("W-MON")


def kev_lags() -> pd.DataFrame:
    k = kev.load()
    k = k[k["fresh"] & k["days_to_kev"].notna()].copy()
    k["log_days"] = np.log1p(k["days_to_kev"].clip(lower=0))
    return k
