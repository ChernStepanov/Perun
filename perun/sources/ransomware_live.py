import asyncio
import json

import pandas as pd

from ..config import RAW
from ..http import fetch_many

API = "https://api.ransomware.live/v2/victims/{y}/{m:02d}"
DIR = RAW / "ransomware_live"
FIRST_MONTH = pd.Period("2021-01", "M")

UNKNOWN_SECTOR = {"", "Not Found", "Unknown", "Other", None}


def _months(today: pd.Timestamp) -> list[pd.Period]:
    return list(pd.period_range(FIRST_MONTH, today.to_period("M"), freq="M"))


def collect(force: bool = False) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today()
    months = _months(today)
    fresh = set(months[-2:])
    todo = [p for p in months if force or p in fresh or not (DIR / f"{p}.json").exists()]
    if not todo:
        return
    urls = {API.format(y=p.year, m=p.month): p for p in todo}
    bodies = asyncio.run(fetch_many(list(urls), concurrency=3, timeout=120))
    for url, body in bodies.items():
        if body is None:
            continue
        (DIR / f"{urls[url]}.json").write_bytes(body)


def _norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    s = s.str.replace(r"\(.*?\)", "", regex=True)
    s = s.str.replace(r"^(www\.)", "", regex=True)
    return s.str.replace(r"[^a-z0-9а-я]+", "", regex=True)


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(DIR.glob("*.json")):
        rows.extend(json.loads(f.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    df = df.rename(columns={"group": "group_name", "activity": "sector"})
    df["discovered"] = pd.to_datetime(df["discovered"], utc=True, format="mixed", errors="coerce")
    df["attackdate"] = pd.to_datetime(df["attackdate"], utc=True, format="mixed", errors="coerce")
    df["discovered"] = df["discovered"].dt.tz_localize(None)
    df["attackdate"] = df["attackdate"].dt.tz_localize(None)

    bad = (df["attackdate"] > df["discovered"] + pd.Timedelta(days=1)) | \
          (df["attackdate"] < pd.Timestamp("2019-01-01"))
    df.loc[bad, "attackdate"] = pd.NaT
    same = (df["discovered"] - df["attackdate"]).abs() < pd.Timedelta(minutes=1)
    df.loc[same, "attackdate"] = pd.NaT

    df["country"] = df["country"].fillna("").str.upper().replace({"UK": "GB"})
    df["sector"] = df["sector"].where(~df["sector"].isin(UNKNOWN_SECTOR), "Unknown")
    df["key"] = df["group_name"].str.lower() + "|" + _norm_name(
        df["domain"].where(df["domain"].fillna("") != "", df["victim"]))
    df = df.sort_values("discovered").drop_duplicates("key", keep="first")
    df["lag_days"] = (df["discovered"] - df["attackdate"]).dt.days
    cols = ["victim", "domain", "group_name", "country", "sector",
            "attackdate", "discovered", "lag_days"]
    return df[cols].reset_index(drop=True)
