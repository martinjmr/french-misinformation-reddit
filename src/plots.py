"""Horizontal bar charts in a light and a dark version, for the README."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

INK = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", accent="#2a78d6", baseline="#898781"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", accent="#3987e5", baseline="#898781"),
}


def hbar(rows: list[tuple[str, float, float, bool]], title: str, path: Path, mode: str,
         ref: tuple[float, str] | None = None, xmax: float = 100) -> None:
    """rows: (label, value in %, std in %, emphasised). ref: (x, label) for a vertical reference line."""
    ink = INK[mode]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.4), dpi=200)
    fig.patch.set_facecolor(ink["surface"])
    ax.set_facecolor(ink["surface"])
    ax.set_xlim(0, xmax * 1.08)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    fig.subplots_adjust(left=0.38, right=0.97, top=1 - 0.75 / (0.55 * len(rows) + 1.4),
                        bottom=0.45 / (0.55 * len(rows) + 1.4))
    fig.canvas.draw()

    bbox = ax.get_window_extent()  # keep the 4px rounding circular on screen
    x_per_px, y_per_px = xmax * 1.08 / bbox.width, len(rows) / bbox.height
    r, height = 7 * x_per_px, 0.42
    for i, (_, value, std, emphasised) in enumerate(rows):
        color = ink["accent"] if emphasised else ink["baseline"]
        ax.add_patch(FancyBboxPatch((0, i - height / 2), value, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                    mutation_aspect=y_per_px / x_per_px, linewidth=0, facecolor=color))
        ax.add_patch(Rectangle((0, i - height / 2), max(value - r, 0), height, linewidth=0, facecolor=color))
        label = f"{value:.1f}%" + (f" ± {std:.1f}" if std else "")
        ax.text(value + xmax * 0.015, i, label, va="center", ha="left", fontsize=9.5,
                color=ink["primary"] if emphasised else ink["secondary"])
    if ref:
        ax.axvline(ref[0], color=ink["muted"], linewidth=1, zorder=0)
        ax.text(ref[0] + xmax * 0.01, -0.62, ref[1], va="bottom", ha="left", fontsize=8.5, color=ink["muted"])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in rows], fontsize=9, color=ink["secondary"])
    ticks = [t for t in (0, 25, 50, 75, 100) if t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks], fontsize=8.5, color=ink["muted"])
    ax.tick_params(length=0)
    ax.grid(axis="x", color=ink["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(ink["grid"])
    fig.text(0.02, 1 - 0.2 / (0.55 * len(rows) + 1.4), title, fontsize=10.5, color=ink["primary"],
             ha="left", va="top", fontweight="bold")
    fig.savefig(path, facecolor=ink["surface"])
    plt.close(fig)
