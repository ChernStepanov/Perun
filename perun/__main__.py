import argparse
import time

from . import analysis, panel, plots
from .sources import epoch, exploitdb, kev, ransomware_live, ransomwatch

SOURCES = {
    "ransomware_live": ransomware_live,
    "ransomwatch": ransomwatch,
    "kev": kev,
    "exploitdb": exploitdb,
    "epoch": epoch,
}


def collect(force: bool) -> None:
    for name, mod in SOURCES.items():
        t = time.time()
        mod.collect(force=force)
        print(f"{name:16s} {time.time() - t:6.1f} c")


def build() -> None:
    p = panel.build()
    panel.save(p)
    print(f"weekly.csv: {len(p)} недель, {p.index[0].date()} — {p.index[-1].date()}")


def analyze(n_placebo: int) -> None:
    res = analysis.run(n_placebo)
    plots.render(res)
    print(res["h1_h2"][["post_pct", "p_placebo", "q_bh"]].round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(prog="perun")
    ap.add_argument("step", choices=["collect", "build", "analyze", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--placebo", type=int, default=500)
    a = ap.parse_args()
    if a.step in ("collect", "all"):
        collect(a.force)
    if a.step in ("build", "all"):
        build()
    if a.step in ("analyze", "all"):
        analyze(a.placebo)


if __name__ == "__main__":
    main()
