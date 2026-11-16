from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EVENTS = DATA / "events"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

WEEK_FREQ = "W-MON"
STUDY_START = pd.Timestamp("2021-09-13")

EVENT_WINDOW = (-4, 4)
LAGS = range(-4, 13)

MIN_WEEKLY_MEAN = 3.0

USER_AGENT = "perun-research/0.3 (+https://github.com/ChernStepanov/Perun)"

for _p in (RAW, PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


def week_start(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, utc=True, errors="coerce", format="mixed").dt.tz_localize(None).dt.normalize()
    return s - pd.to_timedelta(s.dt.day_of_week, unit="D")
