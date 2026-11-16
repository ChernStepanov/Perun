import pandas as pd

from ..config import RAW
from ..http import fetch_to, load_json

URL = "https://raw.githubusercontent.com/joshhighet/ransomwatch/main/posts.json"
FILE = RAW / "ransomwatch_posts.json"


def collect(force: bool = False) -> None:
    fetch_to(URL, FILE, force=force)


def load() -> pd.DataFrame:
    df = pd.DataFrame(load_json(FILE))
    df["discovered"] = pd.to_datetime(df["discovered"], errors="coerce", format="mixed")
    df["key"] = df["group_name"].str.lower() + "|" + \
        df["post_title"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    return df.sort_values("discovered").drop_duplicates("key")[["group_name", "discovered"]]
