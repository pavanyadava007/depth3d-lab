"""Generate the matplotlib figures of the paper from results/*.json (PDF output in paper/figs/).

  figs/scale_per_camera.pdf   median scale factor GT/pred per camera and model, p10/p90 whiskers
  figs/absrel_bins.pdf        AbsRel per depth bin, raw (left panel) and median-scaled (right panel)

Models that were not measured (no json file) are simply absent from the plots.
Run: .venv/bin/python paper/make_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figs"

ORDER = ["depth_pro", "depth_pro_gtfocal", "dav2_metric_outdoor_large", "moge2", "moge2_gtfocal"]
LABEL = {
    "depth_pro": "Depth Pro",
    "depth_pro_gtfocal": "Depth Pro (true f)",
    "dav2_metric_outdoor_large": "DA-V2 outdoor L",
    "moge2": "MoGe-2",
    "moge2_gtfocal": "MoGe-2 (true f)",
}
# categorical palette in fixed slot order; colour follows the model, not its rank, so a missing model
# does not repaint the others. Adjacent-pair CVD separation of this order was checked with the palette validator.
COLOR = {
    "depth_pro": "#2a78d6",
    "depth_pro_gtfocal": "#eb6834",
    "dav2_metric_outdoor_large": "#1baf7a",
    "moge2": "#eda100",
    "moge2_gtfocal": "#e87ba4",
}
MARKER = {"depth_pro": "o", "depth_pro_gtfocal": "s", "dav2_metric_outdoor_large": "^", "moge2": "D", "moge2_gtfocal": "v"}
CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAM_LONG = {"CAM_FRONT": "F", "CAM_FRONT_LEFT": "FL", "CAM_FRONT_RIGHT": "FR", "CAM_BACK": "B",
            "CAM_BACK_LEFT": "BL", "CAM_BACK_RIGHT": "BR"}  # fmt: skip
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load_results() -> list[dict]:
    files = sorted(RES.glob("mono_depth_*.json"))
    rs = [json.loads(p.read_text()) for p in files]
    rs.sort(key=lambda r: ORDER.index(r["variant"]) if r["variant"] in ORDER else 99)
    return rs


def style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def top_legend(fig, handles, labels, ncol: int) -> None:
    fig.legend(handles, labels, frameon=False, fontsize=6.5, ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.2)  # fmt: skip


def scale_per_camera(rs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    n = len(rs)
    width = 0.84 / n
    x = np.arange(len(CAMS))
    for i, r in enumerate(rs):
        sc = r["scale_factor_by_camera"]
        med = np.array([sc[c]["median"] for c in CAMS])
        lo = med - np.array([sc[c]["p10"] for c in CAMS])
        hi = np.array([sc[c]["p90"] for c in CAMS]) - med
        pos = x + (i - (n - 1) / 2) * width
        ax.bar(pos, med, width=width * 0.86, color=COLOR.get(r["variant"], MUTED), label=LABEL.get(r["variant"], r["variant"]),
               linewidth=0)  # fmt: skip
        ax.errorbar(pos, med, yerr=[lo, hi], fmt="none", ecolor=MUTED, elinewidth=0.7, capsize=1.5)
    ax.axhline(1.0, color=TEXT, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([CAM_LONG[c] for c in CAMS])
    ax.set_ylabel("scale factor GT/pred")
    style_axes(ax)
    h, lab = ax.get_legend_handles_labels()
    top_legend(fig, h, lab, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIG / "scale_per_camera.pdf")
    plt.close(fig)


def absrel_bins(rs: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.5), sharey=True)
    bins = list(rs[0]["per_bin_raw"].keys())
    x = np.arange(len(bins))
    for ax, key, title in ((axes[0], "per_bin_raw", "raw"), (axes[1], "per_bin_scaled", "median-scaled")):
        for r in rs:
            v = r["variant"]
            y = [r[key][b]["abs_rel"] for b in bins]
            ax.plot(x, y, color=COLOR.get(v, MUTED), linewidth=1.6, marker=MARKER.get(v, "o"), markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6, label=LABEL.get(v, v))  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("m", "") for b in bins])
        ax.set_xlabel("LiDAR depth bin [m]")
        ax.set_title(title, fontsize=7, color=TEXT)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.25, len(bins) - 0.75)
        style_axes(ax)
    axes[0].set_ylabel("AbsRel")
    h, lab = axes[0].get_legend_handles_labels()
    top_legend(fig, h, lab, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIG / "absrel_bins.pdf")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rs = load_results()
    if not rs:
        raise SystemExit("no results/mono_depth_*.json; nothing to plot")
    scale_per_camera(rs)
    absrel_bins(rs)
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
