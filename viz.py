"""Shared plotting helpers for the visualization scripts (PNG output, no window)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"

INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e5e4df"
COLORS = {  # categorical slots 1-5 of a colorblind-checked palette
    "attention": "#2a78d6",
    "ffn": "#eb6834",
    "embedding": "#1baf7a",
    "norm": "#eda100",
    "output": "#e87ba4",
}


def heatmap(ax, m, title, xlabels=None, ylabels=None, cmap="Blues", vmin=None, vmax=None):
    """Matrix as an image. Sequential 'Blues' for weights/masks, diverging 'RdBu' for signed values."""
    im = ax.imshow(m, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right", fontsize=8)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120, facecolor="white")
    plt.close(fig)
    print(f"   -> saved figures/{name}")
