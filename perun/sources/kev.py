import asyncio
import json

import pandas as pd

from ..config import RAW
from ..http import fetch_many, fetch_to, load_json

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
CVELIST = "https://raw.githubusercontent.com/CVEProject/cvelistV5/main/cves/{year}/{bucket}xxx/{cve}.json"
KEV_FILE = RAW / "kev.json"
CVE_DATES = RAW / "cve_published.csv"


def _cvelist_url(cve: str) -> str:
    _, year, num = cve.split("-")
    return CVELIST.format(year=year, bucket=int(num) // 1000, cve=cve)


def collect(force: bool = False) -> None:
    fetch_to(KEV_URL, KEV_FILE, force=True)
    ids = [v["cveID"] for v in load_json(KEV_FILE)["vulnerabilities"]]
    known = pd.read_csv(CVE_DATES) if CVE_DATES.exists() and not force else \
        pd.DataFrame(columns=["cve", "cve_published"])
    todo = sorted(set(ids) - set(known["cve"]))
    if not todo:
        return
    urls = {_cvelist_url(c): c for c in todo}
    bodies = asyncio.run(fetch_many(list(urls), concurrency=16))
    rows = []
    for url, body in bodies.items():
        if body is None:
            rows.append((urls[url], None))
            continue
        meta = json.loads(body)["cveMetadata"]
        rows.append((urls[url], meta.get("datePublished")))
    new = pd.DataFrame(rows, columns=["cve", "cve_published"])
    pd.concat([known, new]).to_csv(CVE_DATES, index=False)


def load() -> pd.DataFrame:
    kev = pd.DataFrame(load_json(KEV_FILE)["vulnerabilities"])
    kev = kev.rename(columns={"cveID": "cve", "dateAdded": "date_added",
                              "vendorProject": "vendor"})
    kev["date_added"] = pd.to_datetime(kev["date_added"])
    kev["ransomware"] = kev["knownRansomwareCampaignUse"].eq("Known")
    dates = pd.read_csv(CVE_DATES)
    dates["cve_published"] = pd.to_datetime(dates["cve_published"], utc=True, format="mixed",
                                            errors="coerce").dt.tz_localize(None)
    kev = kev.merge(dates, on="cve", how="left")
    kev["days_to_kev"] = (kev["date_added"] - kev["cve_published"].dt.normalize()).dt.days
    kev["fresh"] = kev["cve_published"] >= pd.Timestamp("2021-11-03")
    return kev[["cve", "vendor", "product", "date_added", "cve_published",
                "days_to_kev", "fresh", "ransomware"]]
