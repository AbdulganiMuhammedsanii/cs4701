#!/usr/bin/env python3
"""Generate 5 SVG figures for the report from results/*.json (no matplotlib)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 720, 440
ML, MR, MT, MB = 70, 30, 50, 70
PLOT_W = W - ML - MR
PLOT_H = H - MT - MB

PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def header(w: int = W, h: int = H, title: str = "") -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Helvetica, Arial, sans-serif" font-size="13">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-size="16" font-weight="bold">{title}</text>',
    ]


def y_axis(ymin: float, ymax: float, ticks: int = 5, label: str = "") -> list[str]:
    out = [f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PLOT_H}" stroke="black"/>']
    for i in range(ticks + 1):
        v = ymin + (ymax - ymin) * i / ticks
        y = MT + PLOT_H - (PLOT_H * i / ticks)
        out.append(f'<line x1="{ML-4}" y1="{y}" x2="{ML}" y2="{y}" stroke="black"/>')
        out.append(f'<text x="{ML-8}" y="{y+4}" text-anchor="end">{v:.2f}</text>')
        out.append(f'<line x1="{ML}" y1="{y}" x2="{ML+PLOT_W}" y2="{y}" stroke="#eee"/>')
    if label:
        out.append(
            f'<text x="20" y="{MT+PLOT_H/2}" text-anchor="middle" '
            f'transform="rotate(-90 20 {MT+PLOT_H/2})">{label}</text>'
        )
    return out


def x_axis_categorical(cats: list[str], label: str = "") -> list[tuple[str, float]]:
    """Return list of (label, center_x) and emit x-axis ticks/labels via side effect dropped."""
    n = len(cats)
    out_pos = []
    for i, c in enumerate(cats):
        x = ML + PLOT_W * (i + 0.5) / n
        out_pos.append((c, x))
    return out_pos


def x_axis_lines(cats: list[str], label: str = "") -> list[str]:
    out = [f'<line x1="{ML}" y1="{MT+PLOT_H}" x2="{ML+PLOT_W}" y2="{MT+PLOT_H}" stroke="black"/>']
    n = len(cats)
    for i, c in enumerate(cats):
        x = ML + PLOT_W * (i + 0.5) / n
        out.append(f'<line x1="{x}" y1="{MT+PLOT_H}" x2="{x}" y2="{MT+PLOT_H+4}" stroke="black"/>')
        out.append(f'<text x="{x}" y="{MT+PLOT_H+20}" text-anchor="middle">{c}</text>')
    if label:
        out.append(f'<text x="{ML+PLOT_W/2}" y="{H-15}" text-anchor="middle">{label}</text>')
    return out


def x_axis_numeric(xmin: float, xmax: float, ticks: list[float], label: str = "") -> list[str]:
    out = [f'<line x1="{ML}" y1="{MT+PLOT_H}" x2="{ML+PLOT_W}" y2="{MT+PLOT_H}" stroke="black"/>']
    for v in ticks:
        x = ML + PLOT_W * (v - xmin) / (xmax - xmin)
        out.append(f'<line x1="{x}" y1="{MT+PLOT_H}" x2="{x}" y2="{MT+PLOT_H+4}" stroke="black"/>')
        out.append(f'<text x="{x}" y="{MT+PLOT_H+20}" text-anchor="middle">{v:g}</text>')
    if label:
        out.append(f'<text x="{ML+PLOT_W/2}" y="{H-15}" text-anchor="middle">{label}</text>')
    return out


def yval_to_pix(v: float, ymin: float, ymax: float) -> float:
    return MT + PLOT_H - PLOT_H * (v - ymin) / (ymax - ymin)


def legend(items: list[tuple[str, str]], x: float, y: float) -> list[str]:
    out = []
    for i, (lbl, color) in enumerate(items):
        out.append(f'<rect x="{x}" y="{y + i*18}" width="14" height="12" fill="{color}"/>')
        out.append(f'<text x="{x+20}" y="{y + i*18 + 11}">{lbl}</text>')
    return out


def write_svg(name: str, parts: list[str]) -> Path:
    p = OUT / name
    p.write_text("\n".join(parts) + "\n</svg>\n")
    return p


# ------------- FIG 1: Headline grouped bar -------------
def fig1() -> Path:
    m = json.loads((RES / "metrics.json").read_text())
    verifiers = ["embedding", "nli", "finetuned"]
    series = [
        ("Acc (3-way)", [m[v]["accuracy"] for v in verifiers], PAL[0]),
        ("F1 macro (3-way)", [m[v]["f1_macro"] for v in verifiers], PAL[1]),
        ("Acc (binary)", [m[v + "_binary"]["accuracy"] for v in verifiers], PAL[2]),
        ("F1 macro (binary)", [m[v + "_binary"]["f1_macro"] for v in verifiers], PAL[3]),
    ]
    parts = header(title="Figure 1 — Headline accuracy & macro-F1 across verifiers")
    parts += y_axis(0.0, 1.0, 5, "Score")
    pos = x_axis_categorical(["Embedding (τ=0.35)", "Pretrained NLI", "Fine-tuned RoBERTa"])
    parts += x_axis_lines(["Embedding (τ=0.35)", "Pretrained NLI", "Fine-tuned RoBERTa"])
    n_groups = len(verifiers)
    n_series = len(series)
    group_w = PLOT_W / n_groups * 0.78
    bar_w = group_w / n_series
    for gi, v in enumerate(verifiers):
        cx = pos[gi][1]
        start_x = cx - group_w / 2
        for si, (lbl, vals, color) in enumerate(series):
            x = start_x + si * bar_w
            y = yval_to_pix(vals[gi], 0, 1)
            h = MT + PLOT_H - y
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w-1}" height="{h}" fill="{color}"/>')
            parts.append(
                f'<text x="{x+bar_w/2}" y="{y-3}" text-anchor="middle" font-size="10">{vals[gi]:.2f}</text>'
            )
    parts += legend([(s[0], s[2]) for s in series], ML + 10, MT + 6)
    return write_svg("fig1_headline.svg", parts)


# ------------- FIG 2: Threshold sweep -------------
def fig2() -> Path:
    d = json.loads((RES / "initial_experiments.json").read_text())
    pts = d["embedding_threshold_sweep"]
    xs = [p["threshold"] for p in pts]
    y3 = [p["accuracy"] for p in pts]
    yb = [p["binary"]["accuracy"] for p in pts]
    parts = header(title="Figure 2 — Embedding baseline accuracy vs cosine threshold τ")
    parts += y_axis(0.4, 0.8, 4, "Accuracy")
    parts += x_axis_numeric(min(xs), max(xs), xs, "Threshold τ")

    def to_xy(x, y):
        px = ML + PLOT_W * (x - min(xs)) / (max(xs) - min(xs))
        py = yval_to_pix(y, 0.4, 0.8)
        return px, py

    for ys, color, name in [(y3, PAL[0], "3-way"), (yb, PAL[1], "Binary collapse")]:
        path = "M " + " L ".join(f"{a},{b}" for a, b in (to_xy(x, y) for x, y in zip(xs, ys)))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in zip(xs, ys):
            px, py = to_xy(x, y)
            parts.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"/>')
            parts.append(
                f'<text x="{px}" y="{py-8}" text-anchor="middle" font-size="10" fill="{color}">{y:.3f}</text>'
            )
    parts += legend([("3-way", PAL[0]), ("Binary collapse", PAL[1])], W - 180, MT + 6)
    return write_svg("fig2_threshold.svg", parts)


# ------------- FIG 3: Evidence length ablation -------------
def fig3() -> Path:
    d = json.loads((RES / "evidence_length_ablation.json").read_text())
    rows = d["lengths"]
    cats = []
    emb = []
    nli = []
    for r in rows:
        c = r["max_evidence_chars"]
        cats.append("full" if c == "full" else str(c))
        emb.append(r["embedding"]["accuracy"])
        nli.append(r["nli"]["accuracy"])
    order = sorted(range(len(cats)), key=lambda i: (cats[i] == "full", int(cats[i]) if cats[i] != "full" else 9999))
    cats = [cats[i] for i in order]
    emb = [emb[i] for i in order]
    nli = [nli[i] for i in order]
    parts = header(title="Figure 3 — Three-way accuracy vs max evidence chars")
    parts += y_axis(0.4, 0.65, 5, "Accuracy (3-way)")
    parts += x_axis_lines(cats, "max_evidence_chars (prefix truncation)")
    n = len(cats)

    def to_xy(i, y):
        px = ML + PLOT_W * (i + 0.5) / n
        py = yval_to_pix(y, 0.4, 0.65)
        return px, py

    for ys, color, name in [(emb, PAL[0], "Embedding"), (nli, PAL[1], "Pretrained NLI")]:
        path = "M " + " L ".join(f"{a},{b}" for a, b in (to_xy(i, y) for i, y in enumerate(ys)))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i, y in enumerate(ys):
            px, py = to_xy(i, y)
            parts.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"/>')
            parts.append(
                f'<text x="{px}" y="{py-8}" text-anchor="middle" font-size="10" fill="{color}">{y:.3f}</text>'
            )
    parts += legend([("Embedding", PAL[0]), ("Pretrained NLI", PAL[1])], W - 180, MT + 6)
    return write_svg("fig3_length.svg", parts)


# ------------- FIG 4: Claim vs response -------------
def fig4() -> Path:
    d = json.loads((RES / "claim_vs_response_ablation.json").read_text())
    verifiers = ["embedding", "nli"]
    claim_acc = [d["claim_level"][v]["accuracy"] for v in verifiers]
    resp_acc = [d["response_level"][v]["accuracy"] for v in verifiers]
    claim_f1 = [d["claim_level"][v]["f1_macro"] for v in verifiers]
    resp_f1 = [d["response_level"][v]["f1_macro"] for v in verifiers]
    series = [
        ("Acc — claim-level", claim_acc, PAL[0]),
        ("Acc — response-level", resp_acc, PAL[1]),
        ("F1 macro — claim-level", claim_f1, PAL[2]),
        ("F1 macro — response-level", resp_f1, PAL[3]),
    ]
    parts = header(title="Figure 4 — Claim-level vs response-level verification (multi-claim fixture)")
    parts += y_axis(0.0, 1.0, 5, "Score")
    pos = x_axis_categorical(["Embedding", "Pretrained NLI"])
    parts += x_axis_lines(["Embedding", "Pretrained NLI"])
    n_groups = len(verifiers)
    n_series = len(series)
    group_w = PLOT_W / n_groups * 0.78
    bar_w = group_w / n_series
    for gi in range(n_groups):
        cx = pos[gi][1]
        start_x = cx - group_w / 2
        for si, (lbl, vals, color) in enumerate(series):
            x = start_x + si * bar_w
            y = yval_to_pix(vals[gi], 0, 1)
            h = MT + PLOT_H - y
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w-1}" height="{h}" fill="{color}"/>')
            parts.append(
                f'<text x="{x+bar_w/2}" y="{y-3}" text-anchor="middle" font-size="10">{vals[gi]:.2f}</text>'
            )
    parts += legend([(s[0], s[2]) for s in series], ML + 10, MT + 6)
    parts.append(
        f'<text x="{W/2}" y="{H-32}" text-anchor="middle" font-size="11" fill="#444">'
        f'Claim-level decomposition catches hallucinated spans that response-level aggregation hides.</text>'
    )
    return write_svg("fig4_claim_vs_response.svg", parts)


# ------------- FIG 5: Bootstrap CIs -------------
def fig5() -> Path:
    d = json.loads((RES / "bootstrap_accuracy.json").read_text())
    pairs = d["bootstrap"]["pairs"]
    order = ["embedding_minus_finetuned", "nli_minus_finetuned", "embedding_minus_nli"]
    labels = ["embedding − finetuned", "nli − finetuned", "embedding − nli"]
    means = [pairs[k]["mean_diff"] for k in order]
    los = [pairs[k]["ci95_low"] for k in order]
    his = [pairs[k]["ci95_high"] for k in order]
    xmin, xmax = -0.5, 0.1
    parts = header(title="Figure 5 — Paired bootstrap 95% CI for accuracy differences (n=10,000)")
    # horizontal layout
    parts.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PLOT_H}" stroke="black"/>')
    parts.append(f'<line x1="{ML}" y1="{MT+PLOT_H}" x2="{ML+PLOT_W}" y2="{MT+PLOT_H}" stroke="black"/>')
    # x ticks
    for v in [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1]:
        x = ML + PLOT_W * (v - xmin) / (xmax - xmin)
        parts.append(f'<line x1="{x}" y1="{MT+PLOT_H}" x2="{x}" y2="{MT+PLOT_H+4}" stroke="black"/>')
        parts.append(f'<text x="{x}" y="{MT+PLOT_H+20}" text-anchor="middle">{v:+.1f}</text>')
        parts.append(f'<line x1="{x}" y1="{MT}" x2="{x}" y2="{MT+PLOT_H}" stroke="#eee"/>')
    # zero line
    x0 = ML + PLOT_W * (0 - xmin) / (xmax - xmin)
    parts.append(f'<line x1="{x0}" y1="{MT}" x2="{x0}" y2="{MT+PLOT_H}" stroke="#888" stroke-dasharray="4,3"/>')
    # bars
    n = len(order)
    row_h = PLOT_H / n
    for i, lbl in enumerate(labels):
        cy = MT + row_h * (i + 0.5)
        mlx = ML + PLOT_W * (means[i] - xmin) / (xmax - xmin)
        lox = ML + PLOT_W * (los[i] - xmin) / (xmax - xmin)
        hix = ML + PLOT_W * (his[i] - xmin) / (xmax - xmin)
        parts.append(f'<line x1="{lox}" y1="{cy}" x2="{hix}" y2="{cy}" stroke="{PAL[0]}" stroke-width="3"/>')
        parts.append(f'<line x1="{lox}" y1="{cy-8}" x2="{lox}" y2="{cy+8}" stroke="{PAL[0]}" stroke-width="2"/>')
        parts.append(f'<line x1="{hix}" y1="{cy-8}" x2="{hix}" y2="{cy+8}" stroke="{PAL[0]}" stroke-width="2"/>')
        parts.append(f'<circle cx="{mlx}" cy="{cy}" r="5" fill="{PAL[3]}"/>')
        parts.append(f'<text x="{ML-10}" y="{cy+4}" text-anchor="end">{lbl}</text>')
        parts.append(
            f'<text x="{hix+8}" y="{cy+4}" font-size="11" fill="#444">'
            f'Δ={means[i]:+.3f} [{los[i]:+.3f}, {his[i]:+.3f}]</text>'
        )
    parts.append(
        f'<text x="{ML+PLOT_W/2}" y="{H-15}" text-anchor="middle">Accuracy difference (A − B)</text>'
    )
    parts.append(
        f'<text x="{x0+6}" y="{MT+12}" font-size="10" fill="#666">zero</text>'
    )
    return write_svg("fig5_bootstrap.svg", parts)


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5):
        p = fn()
        print("wrote", p)
