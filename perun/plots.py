import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import events, models, panel
from .config import FIGURES, ROOT

INK = "#111111"
GREY = "#9a9a9a"
OPEN = "#c0392b"
CLOSED = "#1f4e79"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
})


def _releases(ax, rel):
    for r in rel.itertuples():
        ax.axvline(r.date, color=OPEN if r.access == "open" else CLOSED, lw=0.7, alpha=0.55, zorder=1)


def _confounders(ax):
    for c in events.confounders().itertuples():
        ax.axvspan(c.start, c.end, color=GREY, alpha=0.18, lw=0, zorder=0)


def weekly(p: pd.DataFrame, rel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    _confounders(ax)
    _releases(ax, rel)
    ax.plot(p.index, p["victims"], color=GREY, lw=0.8, label="ransomware.live, в неделю")
    ax.plot(p.index, p["victims"].rolling(8, center=True).mean(), color=INK, lw=1.8, label="скользящее среднее, 8 нед.")
    rw = p["ransomwatch"].loc[:"2025-03-01"]
    ax.plot(rw.index, rw.rolling(8, center=True).mean(), color=INK, lw=1, ls="--", label="ransomwatch, 8 нед.")
    ax.plot([], [], color=OPEN, lw=1, label="релиз открытой модели")
    ax.plot([], [], color=CLOSED, lw=1, label="релиз закрытой модели")
    ax.fill_between([], [], color=GREY, alpha=0.3, label="контрольные события")
    ax.set_ylabel("жертв на лик-сайтах")
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(FIGURES / "weekly_victims.png")
    plt.close(fig)


def lag_profile(r: models.LagResult, name: str):
    pr = r.profile
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvspan(-4.5, -0.5, color=GREY, alpha=0.15, lw=0)
    ax.errorbar(pr.index, pr["effect_pct"], yerr=[pr["effect_pct"] - pr["ci_lo"], pr["ci_hi"] - pr["effect_pct"]],
                fmt="o", color=INK, ms=4, capsize=2, lw=0.8)
    ax.set_xlabel("недель от релиза")
    ax.set_ylabel("отклонение от базы, %")
    ax.set_xticks(pr.index)
    fig.savefig(FIGURES / f"lag_profile_{name}.png")
    plt.close(fig)


def placebo(r: models.LagResult, name: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(models.pct(r.placebo), bins=40, color=GREY, edgecolor="white")
    ax.axvline(models.pct(r.observed), color=OPEN, lw=2)
    ax.set_xlabel("эффект недель 0–12 после релиза, %")
    ax.set_ylabel("сдвигов календаря")
    ax.set_title(f"факт {models.pct(r.observed):+.1f}%, плацебо p = {r.p_placebo:.2f}", loc="left", fontsize=9)
    ax.grid(False)
    fig.savefig(FIGURES / f"placebo_{name}.png")
    plt.close(fig)


def its_scan(scan: pd.DataFrame, rel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(scan.index, scan["z"], color=INK, lw=1.2)
    ax.axhline(1.96, color=GREY, ls="--", lw=0.8)
    _releases(ax, rel[rel["date"].between(scan.index[0], scan.index[-1])])
    ax.axvline(pd.Timestamp("2022-11-30"), color=CLOSED, lw=2)
    ax.set_ylabel("z скачка уровня")
    ax.set_xlabel("предполагаемая дата разрыва")
    fig.savefig(FIGURES / "its_scan.png")
    plt.close(fig)


def kev(by_half: pd.DataFrame):
    by_half = by_half[by_half["n"] >= 20]
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(by_half))
    ax.bar(x, by_half["within_7d"], color=GREY, label="доля за ≤ 7 дней, %")
    ax.plot(x, by_half["median_days"], color=INK, marker="o", lw=1.2, label="медиана, дней")
    ax.set_xticks(x, by_half.index, rotation=45)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("от публикации CVE до внесения в CISA KEV", loc="left", fontsize=9)
    fig.savefig(FIGURES / "kev_time_to_exploit.png")
    plt.close(fig)


def shares(tab: pd.DataFrame, name: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(tab) + 0.8))
    ax.barh(tab.index, tab["delta_pp"], color=[OPEN if d > 0 else CLOSED for d in tab["delta_pp"]])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("изменение доли, п.п.")
    ax.set_title(title, loc="left", fontsize=9)
    fig.savefig(FIGURES / f"shares_{name}.png")
    plt.close(fig)


def effects(tab: pd.DataFrame):
    t = tab.sort_values("post_pct")
    fig, ax = plt.subplots(figsize=(6, 0.26 * len(t) + 0.8))
    ax.barh(t.index, t["post_pct"], color=[INK if p < 0.05 else GREY for p in t["p_placebo"]])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("недели 0–12 после релиза, % к базе (чёрное: плацебо p < 0.05)")
    fig.savefig(FIGURES / "effects_by_outcome.png")
    plt.close(fig)


def dashboard(p: pd.DataFrame, rel: pd.DataFrame):
    cols = ["victims"] + [c for c in p.columns if c.startswith(("region_", "sector_"))] + \
           ["kev_fresh", "kev_ransomware", "exploits", "llm_releases_open", "llm_releases_closed"]
    fig = go.Figure()
    for i, c in enumerate(cols):
        fig.add_trace(go.Scatter(x=p.index, y=p[c].rolling(4).mean(), name=c, visible=i == 0,
                                 line=dict(color=INK, width=1.6)))
    shapes = [dict(type="line", xref="x", yref="paper", x0=r.date, x1=r.date, y0=0, y1=1,
                   line=dict(color=OPEN if r.access == "open" else CLOSED, width=0.7))
              for r in rel.itertuples()]
    shapes += [dict(type="rect", xref="x", yref="paper", x0=c.start, x1=c.end, y0=0, y1=1,
                    fillcolor=GREY, opacity=0.2, line_width=0) for c in events.confounders().itertuples()]
    buttons = [dict(label=c, method="update", args=[{"visible": [j == i for j in range(len(cols))]}])
               for i, c in enumerate(cols)]
    fig.add_trace(go.Scatter(x=rel["date"], y=[0] * len(rel), mode="markers", name="релизы",
                             text=rel["model"] + " · " + rel["organization"], hoverinfo="text+x",
                             marker=dict(color=[OPEN if a == "open" else CLOSED for a in rel["access"]],
                                         size=7, symbol="triangle-up")))
    fig.update_layout(template="simple_white", shapes=shapes, height=560,
                      title="Око Перуна · недельная динамика, скользящее среднее 4 недели",
                      updatemenus=[dict(buttons=buttons, x=0, y=1.12, xanchor="left")],
                      margin=dict(t=110, l=50, r=20, b=40), showlegend=False)
    (ROOT / "dashboard").mkdir(exist_ok=True)
    fig.write_html(ROOT / "dashboard" / "index.html", include_plotlyjs="cdn")


def render(res: dict):
    p = panel.load()
    rel = events.releases()
    weekly(p, rel)
    lag_profile(res["profiles"]["victims"], "victims")
    placebo(res["profiles"]["victims"], "victims")
    its_scan(res["its_scan"], rel)
    kev(res["h4"]["kev_by_half"])
    shares(res["structure"]["sector_chatgpt"], "sector_chatgpt", "отрасли: до и после ChatGPT")
    shares(res["structure"]["sector_2025"], "sector_2025", "отрасли: 2023–2024 против 2025–2026")
    shares(res["structure"]["region_2025"], "region_2025", "регионы: 2024 против 2025–2026")
    effects(res["h1_h2"])
    dashboard(p, rel)
