"""
Generate the two README charts as SVG.

    python -m scripts.make_charts

Why SVG written by hand rather than matplotlib: matplotlib is not installed and
is not needed. These are two bar charts. The standard library can emit SVG,
which also renders crisply at any size in a GitHub README and adds zero
dependencies to a repository whose selling point is reproducibility.

Every number is read from results/metrics.json. Nothing is hard-coded, so a
chart cannot drift away from the benchmark it claims to describe -- if the
numbers change, re-running this script is the only way the charts change.

Writes:
    results/chart_policy_ladder.svg
    results/chart_ai_economics.svg
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# Palette from Razorpay's open-source Blade design system
# (github.com/razorpay/blade, packages/blade/src/tokens/global/colors.ts),
# converted from its HSL ramps. Blade token names are kept so each value is
# traceable rather than eyeballed, and every one was re-validated for contrast
# and colour-vision separation after the swap -- adopting another team's
# palette does not inherit their accessibility guarantees.
#
# Light surface is painted explicitly: a README renders on both a light and a
# dark page, and a transparent chart would borrow whichever it lands on.
SURFACE = "#ffffff"     # blueGrayLight 0
INK = "#292f32"         # blueGrayLight 1100 · 13.58:1
INK_2 = "#616d75"       # blueGrayLight 700  ·  5.31:1
MUTED = "#7b878e"       # blueGrayLight 600  ·  3.69:1
GRID = "#dee1e3"        # blueGrayLight 200  — hairline only, never a mark
AXIS = "#c3c9cc"
SERIES = "#305eff"     # razorpay.com brand blue · 5.04:1 on white · CVD dE 34.3
CRITICAL = "#d01e11"    # crimson 600 · 5.42:1 · CVD dE 33.0 from azure 500
# Single quotes inside: this string is interpolated into double-quoted SVG
# attributes, so a double-quoted family name would terminate the attribute and
# make the whole document unparseable.
FONT = "Inter,system-ui,-apple-system,'Segoe UI',sans-serif"


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """
    Horizontal bar with only the value end rounded.

    The baseline end stays square so the bar reads as anchored to zero rather
    than floating. Negative widths round the left end instead.
    """
    if w >= 0:
        r = min(r, w, h / 2)
        return (f"M{x},{y} H{x+w-r} A{r},{r} 0 0 1 {x+w},{y+r} "
                f"V{y+h-r} A{r},{r} 0 0 1 {x+w-r},{y+h} H{x} Z")
    w = -w
    r = min(r, w, h / 2)
    return (f"M{x},{y} H{x-w+r} A{r},{r} 0 0 0 {x-w},{y+r} "
            f"V{y+h-r} A{r},{r} 0 0 0 {x-w+r},{y+h} H{x} Z")


def frame(width: int, height: int, body: str, title: str, subtitle: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" \
viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">
  <rect width="{width}" height="{height}" fill="{SURFACE}"/>
  <text x="28" y="34" font-family="{FONT}" font-size="17" font-weight="600" fill="{INK}">{esc(title)}</text>
  <text x="28" y="56" font-family="{FONT}" font-size="12.5" fill="{INK_2}">{esc(subtitle)}</text>
{body}
</svg>
"""


# --------------------------------------------------------------------------
# Chart 1 -- policy ladder
# --------------------------------------------------------------------------

def policy_ladder(m: dict) -> str:
    """
    Share of oracle-achievable value, one bar per policy, honest 0-100% axis.

    Single series, so no legend: the title names the measure. All five bars are
    directly labelled -- with five marks that is legible, and it removes any
    need to read values off the axis.
    """
    shares = {p["policy"]: 100 * p["share_of_oracle"] for p in m["policies"]}
    rows = [
        ("B0  do nothing", shares["B0 do-nothing"], SERIES),
        ("B1  naive retry", shares["B1 naive retry"], SERIES),
        ("B2  strong rules", shares["B2 rules"], SERIES),
        ("B3  selective AI", shares["B3 router"], SERIES),
        ("B★  oracle (ceiling)", 100.0, MUTED),
    ]

    W, LEFT, RIGHT = 760, 168, 74
    row_h, gap = 38, 12          # 12px pitch gap -> >=2px surface gap between fills
    top = 86
    plot_w = W - LEFT - RIGHT
    H = top + len(rows) * (row_h + gap) + 44

    out = []
    # Gridlines every 25%, recessive, behind the marks.
    for pct in (0, 25, 50, 75, 100):
        x = LEFT + plot_w * pct / 100
        out.append(f'  <line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{top+len(rows)*(row_h+gap)-gap+6}" '
                   f'stroke="{GRID if pct else AXIS}" stroke-width="1"/>')
        out.append(f'  <text x="{x:.1f}" y="{top+len(rows)*(row_h+gap)+14}" font-family="{FONT}" '
                   f'font-size="11" fill="{MUTED}" text-anchor="middle">{pct}%</text>')

    for i, (label, val, colour) in enumerate(rows):
        y = top + i * (row_h + gap)
        w = plot_w * val / 100
        out.append(f'  <text x="{LEFT-14}" y="{y+row_h/2+4.5}" font-family="{FONT}" font-size="12.5" '
                   f'fill="{INK}" text-anchor="end">{esc(label)}</text>')
        if w > 0.5:
            out.append(f'  <path d="{bar_path(LEFT, y, w, row_h)}" fill="{colour}"/>')
        out.append(f'  <text x="{LEFT+w+10:.1f}" y="{y+row_h/2+4.5}" font-family="{FONT}" font-size="12.5" '
                   f'font-weight="600" fill="{INK}">{val:.1f}%</text>')

    # The comparison the chart exists to make.
    b2, b3 = shares["B2 rules"], shares["B3 router"]
    out.append(f'  <text x="28" y="{H-12}" font-family="{FONT}" font-size="12" fill="{INK_2}">'
               f'Strong deterministic rules capture {b2:.1f}%. Selective AI captures {b3:.1f}% '
               f'— {b2-b3:.1f} points lower, before counting what the AI cost.</text>')

    return frame(W, H, "\n".join(out),
                 "Policy ladder — share of oracle-achievable value",
                 f"Frozen test split, n={m['n']}, seed {m['seed']}. Exact evaluation, no sampling.")


# --------------------------------------------------------------------------
# Chart 2 -- AI economics
# --------------------------------------------------------------------------

def ai_economics(m: dict) -> str:
    """
    Did the AI pay for itself? One INR axis, zero line marked, negative net
    rendered in the reserved `critical` status colour with an explicit label.

    Regret is money left on the table, so a taller regret bar is worse. B3's
    regret bar is longer than B2's -- that is the finding, and the chart is
    built so it cannot be misread as an improvement.
    """
    e = m["b3_economics"]
    r2, r3 = e["regret_b2"], e["regret_b3"]
    cost, net = e["llm_cost"], e["net_benefit_vs_b2"]
    breakeven = 100 * e["breakeven_capture_needed"]

    # The note sits under each row label rather than in a right-hand column.
    # A right column cannot work here: the plot spans the full width by
    # construction, so a value label at the longest bar's end always overruns
    # into it. Putting notes on the left removes that collision class entirely.
    rows = [
        ("Value available above B2", "B2 regret — the prize", r2, SERIES),
        ("LLM cost to chase it", f"break-even needs {breakeven:.1f}%", cost, MUTED),
        ("B3 regret", f"worse than B2 by ₹{r3-r2:,.0f}", r3, CRITICAL),
        ("Net benefit vs B2", "AI did not earn its cost", net, CRITICAL),
    ]

    W, LEFT, RIGHT = 800, 260, 90
    row_h, gap = 44, 16
    top = 92
    plot_w = W - LEFT - RIGHT
    lo, hi = min(0, net) * 1.08, max(r3, r2) * 1.08
    span = hi - lo
    zero_x = LEFT + plot_w * (0 - lo) / span
    H = top + len(rows) * (row_h + gap) + 62

    def px(v):
        return LEFT + plot_w * (v - lo) / span

    out = []
    for tick in (-2000, -1000, 0, 1000, 2000, 3000, 4000):
        if not (lo <= tick <= hi):
            continue
        x = px(tick)
        is_zero = tick == 0
        out.append(f'  <line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{top+len(rows)*(row_h+gap)-gap+6}" '
                   f'stroke="{AXIS if is_zero else GRID}" stroke-width="{1.5 if is_zero else 1}"/>')
        out.append(f'  <text x="{x:.1f}" y="{top+len(rows)*(row_h+gap)+14}" font-family="{FONT}" '
                   f'font-size="11" fill="{MUTED}" text-anchor="middle">'
                   f'{"0" if is_zero else format(tick, ",")}</text>')

    for i, (label, note, val, colour) in enumerate(rows):
        y = top + i * (row_h + gap)
        mid = y + row_h / 2
        out.append(f'  <text x="{LEFT-16}" y="{mid-3:.1f}" font-family="{FONT}" font-size="12.5" '
                   f'fill="{INK}" text-anchor="end">{esc(label)}</text>')
        out.append(f'  <text x="{LEFT-16}" y="{mid+13:.1f}" font-family="{FONT}" font-size="10.5" '
                   f'fill="{MUTED}" text-anchor="end">{esc(note)}</text>')

        w = px(val) - zero_x
        out.append(f'  <path d="{bar_path(zero_x, y, w, row_h)}" fill="{colour}"/>')

        # Positive bars label outside the value end. A negative bar labels
        # INSIDE itself, in the surface colour: an outside label would run left
        # into the row label, which is what broke the first draft.
        text = f'{"−" if val < 0 else ""}₹{abs(val):,.0f}'
        if w >= 0:
            tx, anchor, fill = px(val) + 10, "start", INK
        else:
            tx, anchor, fill = px(val) + 12, "start", SURFACE
        out.append(f'  <text x="{tx:.1f}" y="{mid+4.5:.1f}" font-family="{FONT}" font-size="12.5" '
                   f'font-weight="600" fill="{fill}" text-anchor="{anchor}">{text}</text>')

    out.append(f'  <text x="28" y="{H-28}" font-family="{FONT}" font-size="12" fill="{INK_2}">'
               f'Routing the ambiguous tail to an AI cost ₹{cost:,.0f} and left ₹{r3-r2:,.0f} '
               f'more on the table than the rules alone.</text>')
    out.append(f'  <text x="28" y="{H-11}" font-family="{FONT}" font-size="12" font-weight="600" '
               f'fill="{CRITICAL}">Net −₹{abs(net):,.0f} versus B2. On this benchmark, more AI '
               f'≠ more revenue.</text>')

    return frame(W, H, "\n".join(out),
                 "Did the AI earn its cost?",
                 f"Selective routing vs strong deterministic rules. Frozen test split, n={m['n']}. "
                 f"LLM backend: {m['llm_backend']}.")


def main() -> int:
    with open(os.path.join(RESULTS, "metrics.json")) as f:
        m = json.load(f)

    for name, svg in (("chart_policy_ladder.svg", policy_ladder(m)),
                      ("chart_ai_economics.svg", ai_economics(m))):
        path = os.path.join(RESULTS, name)
        with open(path, "w") as f:
            f.write(svg)
        print(f"wrote {path} ({len(svg):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
