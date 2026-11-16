import pandas as pd

from ..config import RAW
from ..http import fetch_to

URL = "https://epoch.ai/data/all_ai_models.csv"
FILE = RAW / "epoch_all_ai_models.csv"

OPEN = {"Open weights (unrestricted)", "Open weights (restricted use)",
        "Open weights (non-commercial)"}
CLOSED = {"API access", "Hosted access (no API)"}


def collect(force: bool = False) -> None:
    fetch_to(URL, FILE, force=True)


def load() -> pd.DataFrame:
    df = pd.read_csv(FILE, low_memory=False, usecols=[
        "Model", "Domain", "Organization", "Publication date", "Model accessibility",
        "Parameters", "Training compute (FLOP)", "Country (of organization)"])
    df = df.rename(columns={
        "Model": "model", "Domain": "domain", "Organization": "organization",
        "Publication date": "date", "Model accessibility": "accessibility",
        "Parameters": "parameters", "Training compute (FLOP)": "compute",
        "Country (of organization)": "country"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["domain"].fillna("").str.contains("Language")]
    df["access"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[df["accessibility"].isin(OPEN), "access"] = "open"
    df.loc[df["accessibility"].isin(CLOSED), "access"] = "closed"
    return df.dropna(subset=["date", "access"]).reset_index(drop=True)
