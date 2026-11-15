import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
RED="#D62839"; GREY="#6B7280"; ORANGE="#E09F3E"; DARK="#111827"; BLUE="#2F6FB0"
fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=250)
ax.set_xlim(0, 10); ax.set_ylim(0, 8.7); ax.axis("off"); ax.set_aspect("equal")
pos = {"A": (2.0, 3.6), "B": (8.0, 3.6), "H1-01": (8.0, 7.4)}
def arrow(p, q, color, rad, lw, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=18, lw=lw, color=color, linestyle=ls, shrinkA=30, shrinkB=30, zorder=1))
arrow(pos["A"], pos["B"], GREY, -0.62, 2.0)
arrow(pos["B"], pos["A"], ORANGE, -0.62, 2.6)
arrow(pos["A"], pos["B"], RED, 0.0, 4.0)
arrow(pos["H1-01"], pos["B"], GREY, 0.0, 1.6, "--")
for n, (x, y) in pos.items():
    big = n in ("A", "B")
    ax.add_patch(Circle((x, y), 0.75 if big else 0.6, color=BLUE if big else "#9CA3AF", zorder=2))
    ax.text(x, y, n, ha="center", va="center", color="white", fontsize=15 if big else 9, fontweight="bold", zorder=3)
ax.text(5.0, 6.25, "A paid B 15 times before", ha="center", fontsize=10, color=GREY)
ax.text(5.0, 3.95, "FLAGGED: 8,197 USD, ACH", ha="center", fontsize=10.5, color=RED, fontweight="bold")
ax.text(5.0, 0.35, "B paid A 13 times before,\nlast one 4.0 h ago (about 14,859 USD)", ha="center", fontsize=10, color="#B7791F")
ax.text(8.45, 5.55, "one of B's\nother neighbours", ha="left", fontsize=9, color=GREY)
plt.tight_layout(pad=0.1)
plt.savefig("img/e4_007_graph.png", transparent=False, facecolor="white")
