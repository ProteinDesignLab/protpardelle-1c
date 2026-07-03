"""Paper-quality architecture comparison: U-ViT vs DiT denoiser blocks.

Reads no data — purely a schematic. Emits both PDF (vector, paper-ready) and
PNG (preview) into the same directory as this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ----- Paper-style defaults -----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.0,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ----- Color palette (color-blind friendly, muted) -----
C_NORM = "#E6E6E6"
C_ADA  = "#F2C14E"
C_ATTN = "#7DA7D9"
C_MLP  = "#7DC0A6"
C_GATE = "#E58A8A"
C_RES  = "#A8A8A8"
C_TIME = "#C9A2D8"
C_TEXT = "#222222"


def box(ax, xy, w, h, label, color, fontsize=8.5, italic=False, bold=False, lw=0.6):
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=lw, edgecolor="#444", facecolor=color, zorder=3,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=C_TEXT, weight=weight, zorder=4)
    return y + h / 2, y - h / 2  # top, bottom


def arrow(ax, p_from, p_to, color="#333", lw=0.85, style="-|>", mut=9, ls="-", zorder=2):
    a = FancyArrowPatch(
        p_from, p_to, arrowstyle=style, mutation_scale=mut,
        color=color, lw=lw, linestyle=ls, zorder=zorder,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(a)


def vlink(ax, x, y_from_bottom, y_to_top, color="#333", lw=0.85):
    """Short vertical arrow from one block's bottom to another's top."""
    arrow(ax, (x, y_from_bottom), (x, y_to_top), color=color, lw=lw)


def plus_node(ax, xy, r=0.11):
    x, y = xy
    circ = plt.Circle((x, y), r, facecolor="white", edgecolor="#333", lw=0.8, zorder=4)
    ax.add_patch(circ)
    ax.text(x, y, "+", ha="center", va="center", fontsize=9, color=C_TEXT, zorder=5)


def time_chip(ax, xy, label="time emb. t  (1024-d)"):
    box(ax, xy, 1.95, 0.42, label, C_TIME, fontsize=7.6, lw=0.6)


def time_rail(ax, x_rail, y_top, y_bottom, color="#9966B0"):
    ax.plot([x_rail, x_rail], [y_top, y_bottom], color=color,
            lw=1.0, ls=(0, (1.5, 1.5)), zorder=1)


def adaln_tap(ax, x_rail, y, x_target, label, color=C_ADA, ls="-"):
    arrow(ax, (x_rail, y), (x_target, y),
          color=color, lw=1.0, style="-|>", mut=9, ls=ls, zorder=2)
    ax.text(x_rail + 0.08, y + 0.13, label,
            ha="left", va="bottom", fontsize=7.2,
            color="#5a4200", zorder=4)


# ============================================================
# Geometry constants (compact)
# ============================================================
# Block widths — sized to fit their labels at fontsize 8.5
W_LN   = 2.20  # LayerNorm, Self-Attention
W_LIN  = 2.10  # Linear 256→1024 etc.
W_SMLU = 1.30  # SiLU
W_ADA  = 1.95  # AdaLN (γ, β)
W_ADAX = 2.75  # AdaLN with size annotation (wide enough for ✓/⚠)
W_GATE = 1.50  # ⊙ (α + 1)

H_BLK  = 0.42  # uniform block height
GAP    = 0.18  # gap between consecutive box edges (= arrow length)
STEP   = H_BLK + GAP  # 0.60 = step between block centers


def step_down(y):
    return y - STEP


# ============================================================
# Build figure
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.6, 6.8))
fig.suptitle(
    "Protpardelle denoiser block: U-ViT (original) vs DiT (new)",
    fontsize=12.0, weight="bold", y=0.985,
)


def draw_panel(ax, *, title, title_color, mode, total_label, footnote, footnote_color,
               y_start_offset=0.0):
    """mode ∈ {'uvit', 'dit'}."""
    ax.set_xlim(-3.4, 3.7)
    ax.set_ylim(0.3, 9.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.5, weight="bold", pad=4, color=title_color)

    x_main = -0.4
    x_rail = 2.65  # time-embedding rail x

    # Top of diagram (varies per-panel so DiT and U-ViT bottoms align)
    y_top = 9.10 - y_start_offset

    # Time embedding chip + rail
    time_chip(ax, (x_rail, y_top))

    # Top: input label and residual takeoff arrow into LayerNorm
    y_top_in = y_top - 0.32
    ax.text(x_main, y_top, "input  x", ha="center", fontsize=8.6)

    # ---------------- Attention sub-block ----------------
    y_ln1 = y_top - 0.70
    box(ax, (x_main, y_ln1), W_LN, H_BLK, "LayerNorm", C_NORM)
    y_mod_a = step_down(y_ln1)
    box(ax, (x_main, y_mod_a), W_ADA, H_BLK, "scale, shift  (γ, β)",
        C_ADA)
    y_attn = step_down(y_mod_a)
    box(ax, (x_main, y_attn), W_LN, H_BLK, "Self-Attention", C_ATTN)

    # arrow: input → LN
    arrow(ax, (x_main, y_top_in), (x_main, y_ln1 + H_BLK / 2))
    vlink(ax, x_main, y_ln1 - H_BLK/2, y_mod_a + H_BLK/2)
    vlink(ax, x_main, y_mod_a - H_BLK/2, y_attn + H_BLK/2)

    # AdaLN tap into attn modulation
    if mode == "uvit":
        attn_label = "Linear 1024 → 512  (2 · dim, 0.52 M)"
    else:
        attn_label = "Linear 1024 → 768  (3 · dim, 0.79 M)"
    adaln_tap(ax, x_rail, y_mod_a, x_main + W_ADA / 2 + 0.02, attn_label)

    # Optional DiT post-attention gate
    if mode == "dit":
        y_g1 = step_down(y_attn)
        box(ax, (x_main, y_g1), W_GATE, H_BLK, "⊙ (α₁ + 1)",
            C_GATE, bold=True, fontsize=8.5)
        vlink(ax, x_main, y_attn - H_BLK/2, y_g1 + H_BLK/2)
        adaln_tap(ax, x_rail, y_g1, x_main + W_GATE / 2 + 0.02,
                  "α₁ gate (same proj.)", color=C_GATE, ls=":")
        y_after_attn = y_g1 - H_BLK / 2
    else:
        y_after_attn = y_attn - H_BLK / 2

    # ⊕ for attention residual
    y_plus_a = y_after_attn - 0.32
    plus_node(ax, (x_main, y_plus_a))
    arrow(ax, (x_main, y_after_attn), (x_main, y_plus_a + 0.11))

    # residual bypass: input → ⊕_a (a single arc on the left)
    x_bypass = x_main - 1.45
    ax.plot([x_main, x_bypass], [y_top_in + 0.10, y_top_in + 0.10],
            color=C_RES, lw=0.85, zorder=1)
    ax.plot([x_bypass, x_bypass], [y_top_in + 0.10, y_plus_a],
            color=C_RES, lw=0.85, ls=(0, (3, 2)), zorder=1)
    arrow(ax, (x_bypass, y_plus_a), (x_main - 0.11, y_plus_a),
          color=C_RES, lw=0.85)

    # ---------------- FFN sub-block ----------------
    y_ln2 = y_plus_a - 0.32 - H_BLK / 2
    box(ax, (x_main, y_ln2), W_LN, H_BLK, "LayerNorm", C_NORM)
    arrow(ax, (x_main, y_plus_a - 0.11), (x_main, y_ln2 + H_BLK / 2))

    if mode == "uvit":
        # ORDER: LN → Linear in → SiLU → AdaLN(1024) → Linear out
        y_lin1 = step_down(y_ln2)
        box(ax, (x_main, y_lin1), W_LIN, H_BLK, "Linear  256 → 1024", C_MLP)
        y_silu = step_down(y_lin1)
        box(ax, (x_main, y_silu), W_SMLU, H_BLK, "SiLU", C_MLP, fontsize=8.2)
        y_mod_b = step_down(y_silu)
        box(ax, (x_main, y_mod_b), W_ADAX, H_BLK,
            "scale, shift   on 1024-d", C_ADA, bold=True, fontsize=8.5)
        y_lin2 = step_down(y_mod_b)
        box(ax, (x_main, y_lin2), W_LIN, H_BLK, "Linear  1024 → 256", C_MLP)

        for y_a, y_b in [(y_ln2, y_lin1), (y_lin1, y_silu),
                         (y_silu, y_mod_b), (y_mod_b, y_lin2)]:
            vlink(ax, x_main, y_a - H_BLK / 2, y_b + H_BLK / 2)

        adaln_tap(ax, x_rail, y_mod_b, x_main + W_ADAX / 2 + 0.02,
                  "Linear 1024 → 2048\n(2 · inner_dim, 2.10 M)  ← bottleneck")
        y_after_ffn = y_lin2 - H_BLK / 2

    else:  # dit
        # ORDER: LN → AdaLN(256) → Linear in → SiLU → Linear out → α₂ gate
        y_mod_b = step_down(y_ln2)
        box(ax, (x_main, y_mod_b), W_ADAX, H_BLK,
            "scale, shift   on 256-d", C_ADA, bold=True, fontsize=8.5)
        y_lin1 = step_down(y_mod_b)
        box(ax, (x_main, y_lin1), W_LIN, H_BLK, "Linear  256 → 1024", C_MLP)
        y_silu = step_down(y_lin1)
        box(ax, (x_main, y_silu), W_SMLU, H_BLK, "SiLU", C_MLP, fontsize=8.2)
        y_lin2 = step_down(y_silu)
        box(ax, (x_main, y_lin2), W_LIN, H_BLK, "Linear  1024 → 256", C_MLP)
        y_g2 = step_down(y_lin2)
        box(ax, (x_main, y_g2), W_GATE, H_BLK, "⊙ (α₂ + 1)",
            C_GATE, bold=True, fontsize=8.5)

        for y_a, y_b in [(y_ln2, y_mod_b), (y_mod_b, y_lin1), (y_lin1, y_silu),
                         (y_silu, y_lin2), (y_lin2, y_g2)]:
            vlink(ax, x_main, y_a - H_BLK / 2, y_b + H_BLK / 2)

        adaln_tap(ax, x_rail, y_mod_b, x_main + W_ADAX / 2 + 0.02,
                  "Linear 1024 → 768\n(3 · dim, 0.79 M)  ← compact")
        adaln_tap(ax, x_rail, y_g2, x_main + W_GATE / 2 + 0.02,
                  "α₂ gate (same proj.)", color=C_GATE, ls=":")
        y_after_ffn = y_g2 - H_BLK / 2

    # ⊕ for FFN residual
    y_plus_b = y_after_ffn - 0.32
    plus_node(ax, (x_main, y_plus_b))
    arrow(ax, (x_main, y_after_ffn), (x_main, y_plus_b + 0.11))

    # second residual bypass: ⊕_a → ⊕_b
    x_bypass2 = x_main - 1.45
    arrow(ax, (x_main - 0.11, y_plus_a), (x_bypass2, y_plus_a),
          color=C_RES, lw=0.85, style="-")
    ax.plot([x_bypass2, x_bypass2], [y_plus_a, y_plus_b],
            color=C_RES, lw=0.85, ls=(0, (3, 2)), zorder=1)
    arrow(ax, (x_bypass2, y_plus_b), (x_main - 0.11, y_plus_b),
          color=C_RES, lw=0.85)

    # rail extends only as far as needed
    time_rail(ax, x_rail, y_top - 0.21, y_plus_b)

    # output arrow + label
    y_out = y_plus_b - 0.32
    arrow(ax, (x_main, y_plus_b - 0.11), (x_main, y_out + 0.05))
    ax.text(x_main, y_out - 0.08, "→ next block", ha="center",
            fontsize=8.5, va="top")

    # Total banner under title
    ax.text(0.0, 9.55, total_label, ha="center", va="top",
            fontsize=10.0, weight="bold", color=title_color)

    # Footnote — placed just below each panel's diagram (so U-ViT, which has
    # fewer boxes, doesn't get a huge gap above its footnote).
    y_footnote = y_out - 0.45
    ax.text(-3.35, y_footnote, footnote, fontsize=8.0, color=C_TEXT,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=footnote_color["fc"],
                      edgecolor=footnote_color["ec"], lw=0.6))


draw_panel(
    axes[0],
    title="(a)  U-ViT block",
    title_color="#222222",
    mode="uvit",
    total_label="35.5 M params total",
    footnote=(
        "AdaLN modulates on inner_dim = 1024  (post-expansion)\n"
        "Per-block:  attn 0.79 M  +  FFN 2.62 M  =  3.41 M\n"
        "10 blocks × 3.41 M  +  I/O 1.4 M  ≈  35.5 M"
    ),
    footnote_color={"fc": "#FFF8E0", "ec": "#D9B84C"},
)

draw_panel(
    axes[1],
    title="(b)  DiT block      −29.5 % vs U-ViT",
    title_color="#1a6b30",
    mode="dit",
    total_label="25.0 M params total",
    footnote=(
        "AdaLN modulates on dim = 256  (pre-expansion)\n"
        "Per-block:  attn 1.05 M  +  FFN 1.31 M  =  2.36 M\n"
        "10 blocks × 2.36 M  +  I/O 1.4 M  ≈  25.0 M\n"
        "FFN AdaLN savings:  2.10 M → 0.79 M / block"
    ),
    footnote_color={"fc": "#E8F5EE", "ec": "#5BAE7C"},
)

# Shared legend
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=C_ADA, edgecolor="#444", lw=0.5,
                  label="AdaLN  (scale, shift)"),
    plt.Rectangle((0, 0), 1, 1, facecolor=C_GATE, edgecolor="#444", lw=0.5,
                  label="DiT-only gate  ⊙ (α + 1)"),
    plt.Rectangle((0, 0), 1, 1, facecolor=C_ATTN, edgecolor="#444", lw=0.5,
                  label="Self-Attention"),
    plt.Rectangle((0, 0), 1, 1, facecolor=C_MLP, edgecolor="#444", lw=0.5,
                  label="MLP / FFN"),
    plt.Rectangle((0, 0), 1, 1, facecolor=C_NORM, edgecolor="#444", lw=0.5,
                  label="LayerNorm"),
    plt.Rectangle((0, 0), 1, 1, facecolor=C_TIME, edgecolor="#444", lw=0.5,
                  label="Time embedding (sinusoidal → MLP, shared)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=6,
           frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, 0.02))

plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.05, wspace=0.05)

out_dir = Path(__file__).parent
fig.savefig(out_dir / "uvit_vs_dit.pdf")
fig.savefig(out_dir / "uvit_vs_dit.png", dpi=200)
print(f"Wrote: {out_dir/'uvit_vs_dit.pdf'}")
print(f"Wrote: {out_dir/'uvit_vs_dit.png'}")
