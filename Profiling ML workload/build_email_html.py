#!/usr/bin/env python3
"""Builds an email-safe HTML version of the walkthrough for the Gmail draft.
Deliberately NOT the same markup as walkthrough.html -- email clients don't
support CSS custom properties, grid, or flexbox reliably, so everything here
is inline-styled tables and divs, the standard email-HTML convention."""
import base64
import csv
import html as htmllib

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def fnum(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return s
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.3g}"

FONT = "font-family:Arial,Helvetica,sans-serif;"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
BORDER = "#e1e0d9"
ACCENT = "#2a78d6"
ACCENT2 = "#eb6834"
SURFACE = "#fcfcfb"
GOOD = "#0ca30c"

def th(text, extra=""):
    return (f'<th style="{FONT}text-align:left;padding:7px 10px;background:{SURFACE};'
           f'border-bottom:2px solid {BORDER};color:{INK2};font-size:11.5px;'
           f'text-transform:uppercase;letter-spacing:.04em;{extra}">{text}</th>')

def td(text, extra=""):
    return (f'<td style="{FONT}padding:7px 10px;border-bottom:1px solid {BORDER};'
           f'font-size:13.5px;color:{INK};{extra}">{text}</td>')

def pill(text, color):
    return (f'<span style="{FONT}display:inline-block;padding:2px 8px;border-radius:999px;'
           f'font-size:11px;font-weight:bold;color:#fff;background:{color};white-space:nowrap;">{text}</span>')

# ---------------------------------------------------------------------------
# Architecture comparison table
# ---------------------------------------------------------------------------
arch_rows = [
    ("Type", "Dense causal transformer", "Mixture-of-Experts"),
    ("Total / activated params", "27.8B / 27.8B (all)", "284B / 13B"),
    ("Layers", "52 (39 local + 13 global)", "43 (2 SWA + 21 CSA + 20 HCA)"),
    ("Hidden size", "6,656", "4,096"),
    ("Attention", "GQA 32Q/2KV, window 2,048", "Hybrid CSA/HCA: compress+index+select+attend"),
    ("FFN", "SwiGLU, 19,968", "1 shared + 6-of-256 routed, width 2,048"),
    ("Speculative module", "5-layer draft, GQA 32Q/8KV, block 16", "MTP depth 1 (\"DSpark\"), block 7"),
    ("Stochastic surface", "1 scalar (&alpha;, accept rate)", "2 independent: routing (40/43 layers) + accept rate"),
]
arch_table = [f'<table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;margin:10px 0 4px;">',
             '<tr>' + th("") + th("Muse Glimmer-30B") + th("DeepSeek-V4-Flash") + '</tr>']
for label, a, b in arch_rows:
    arch_table.append('<tr>' + td(label, f"color:{MUTED};white-space:nowrap;") + td(a) + td(b) + '</tr>')
arch_table.append('</table>')
arch_table_html = "".join(arch_table)

# ---------------------------------------------------------------------------
# Bottleneck highlights (short_chat_L1500 regime, both models)
# ---------------------------------------------------------------------------
g_rows = [r for r in read_csv("bottleneck_summary.csv") if r["context_regime"] == "short_chat_L1500"]
d_rows = [r for r in read_csv("deepseek_bottleneck_summary.csv") if r["context_regime"] == "short_chat_L1500"]

bt_table = ['<table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;margin:10px 0 4px;">',
           '<tr>' + th("Model") + th("Task") + th("Latency") + th("Dominant kernel") + th("Bound") + '</tr>']
for model, rows in [("Glimmer-30B", g_rows), ("DeepSeek-V4-Flash", d_rows)]:
    for r in rows:
        bound = r["dominant_bound_type"]
        color = GOOD if bound == "compute" else ACCENT2
        bt_table.append('<tr>' + td(model, f"color:{MUTED};white-space:nowrap;") +
                        td(f'<span style="font-family:monospace;">{r["task"]}</span>') +
                        td(f'{r["task_expected_latency_ms"]} ms', "text-align:right;white-space:nowrap;") +
                        td(f'<span style="font-family:monospace;font-size:12px;">{htmllib.escape(r["dominant_kernel"])}</span>') +
                        td(pill(bound, color)) + '</tr>')
bt_table.append('</table>')
bt_table_html = "".join(bt_table)

# ---------------------------------------------------------------------------
# Read + base64 the two headline images for inline (cid) attachment
# ---------------------------------------------------------------------------
def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

glimmer_cid = "glimmer_kernel_dag.png"
deepseek_cid = "deepseek_kernel_dag.png"

with open("email_images.txt", "w") as f:
    pass  # placeholder, images written separately below by the caller script

print("ARCH_TABLE_HTML_LEN", len(arch_table_html))
print("BT_TABLE_HTML_LEN", len(bt_table_html))

with open("_arch_table.html", "w") as f:
    f.write(arch_table_html)
with open("_bt_table.html", "w") as f:
    f.write(bt_table_html)
print("wrote _arch_table.html, _bt_table.html")
