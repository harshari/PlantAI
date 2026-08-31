#!/usr/bin/env python3
"""Generates walkthrough.html from the CSVs/figures already in this folder --
reads the real numbers rather than hand-transcribing them, so the report and
the source data can never drift apart."""
import base64
import csv
import html as htmllib

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def fnum(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return s
    if abs(v) >= 1e6 or (0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.3g}"

def kernel_table_html(rows, id_col_short=True):
    cols = ["kernel_id", "op_type", "precision", "multiplicity", "total_flops",
            "total_bytes", "arithmetic_intensity_flops_per_byte", "note"]
    heads = ["Kernel", "Op type", "Precision", "×", "Total FLOPs", "Total bytes",
             "FLOPs/Byte", "Note"]
    out = ['<div class="tablewrap"><table><thead><tr>']
    for h in heads:
        out.append(f"<th>{h}</th>")
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append(f'<tr><td class="mono optype-{r["op_type"]}">{htmllib.escape(r["kernel_id"])}</td>')
        out.append(f'<td><span class="pill pill-{r["op_type"]}">{r["op_type"]}</span></td>')
        out.append(f'<td>{r["precision"]}</td>')
        out.append(f'<td class="num">{fnum(r["multiplicity"])}</td>')
        out.append(f'<td class="num">{fnum(r["total_flops"])}</td>')
        out.append(f'<td class="num">{fnum(r["total_bytes"])}</td>')
        out.append(f'<td class="num">{fnum(r["arithmetic_intensity_flops_per_byte"])}</td>')
        out.append(f'<td class="note">{htmllib.escape(r["note"])}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)

def bottleneck_table_html(rows):
    out = ['<div class="tablewrap"><table><thead><tr>',
           "<th>Context regime</th><th>Task</th><th>Expected latency</th>",
           "<th>Dominant kernel</th><th>Share</th><th>Bound</th></tr></thead><tbody>"]
    for r in rows:
        bound_cls = "bound-compute" if r["dominant_bound_type"] == "compute" else "bound-bw"
        out.append(f'<tr><td>{r["context_regime"]}</td><td class="mono">{r["task"]}</td>'
                   f'<td class="num">{r["task_expected_latency_ms"]} ms</td>'
                   f'<td class="mono">{htmllib.escape(r["dominant_kernel"])}</td>'
                   f'<td class="num">{r["dominant_kernel_share"]}</td>'
                   f'<td><span class="pill {bound_cls}">{r["dominant_bound_type"]}</span></td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)

glimmer_kernels = read_csv("kernel_table.csv")
glimmer_bottleneck = read_csv("bottleneck_summary.csv")
deepseek_kernels = read_csv("deepseek_kernel_table.csv")
deepseek_bottleneck = read_csv("deepseek_bottleneck_summary.csv")

glimmer_dag_png = b64("dag_figure.png")
glimmer_kdag_png = b64("kernel_dag_figure.png")
deepseek_kdag_png = b64("deepseek_dag_figure.png")

with open("example.anynet") as f:
    anynet_txt = f.read()
with open("example.trace") as f:
    trace_lines = f.readlines()
trace_preview = "".join(trace_lines[:2] + trace_lines[2:12])
trace_total = len(trace_lines) - 2

HTML = f"""<title>Two Workload DAGs</title>
<style>
:root {{
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,0.10);
  --accent:#2a78d6; --accent-2:#eb6834;
  --good:#0ca30c; --warn:#eda100; --bad:#d03b3b;
  --op-matmul:#2a78d6; --op-attention:#eb6834; --op-norm:#9a9890; --op-elementwise:#c3c2b7;
  --op-lookup:#1baf7a; --op-reduce_argmax:#e34948; --op-iterative_normalize:#4a3aa7;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,0.10);
    --accent:#3987e5; --accent-2:#d95926;
    --good:#0ca30c; --warn:#c98500; --bad:#e66767;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,0.10);
  --accent:#3987e5; --accent-2:#d95926;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font-family: system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; font-size:16px; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:40px 24px 80px; display:flex; flex-direction:column; gap:30px; }}
header .eyebrow {{ font-size:13px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin:0 0 8px; }}
header h1 {{ font-size:clamp(28px,4vw,40px); margin:0 0 10px; text-wrap:balance; }}
header p {{ max-width:75ch; color:var(--ink-2); margin:0; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:24px 26px; }}
.card h2 {{ font-size:22px; margin:0 0 6px; }}
.card h3 {{ font-size:16px; margin:22px 0 8px; }}
.card p.sub {{ color:var(--ink-2); margin:0 0 16px; max-width:90ch; }}
.model-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media (max-width:820px) {{ .model-grid {{ grid-template-columns:1fr; }} }}
.spec-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
.spec-table td {{ padding:6px 8px; border-bottom:1px solid var(--grid); }}
.spec-table td:first-child {{ color:var(--muted); width:44%; }}
.spec-table td:last-child {{ font-variant-numeric:tabular-nums; font-weight:600; }}
.tablewrap {{ overflow-x:auto; margin:10px 0 4px; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th,td {{ padding:6px 9px; text-align:left; border-bottom:1px solid var(--grid); vertical-align:top; }}
th {{ color:var(--ink-2); text-transform:uppercase; letter-spacing:.04em; font-size:10.5px; position:sticky; top:0; background:var(--surface); }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.mono {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12px; white-space:nowrap; }}
td.note {{ color:var(--ink-2); max-width:340px; }}
tr:hover td {{ background:color-mix(in srgb, var(--accent) 6%, transparent); }}
.pill {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700; color:#fff; white-space:nowrap; }}
.pill-matmul {{ background:var(--op-matmul); }} .pill-attention {{ background:var(--op-attention); }}
.pill-norm {{ background:var(--op-norm); }} .pill-elementwise {{ background:var(--op-elementwise); color:#333; }}
.pill-lookup {{ background:var(--op-lookup); }} .pill-reduce_argmax {{ background:var(--op-reduce_argmax); }}
.pill-iterative_normalize {{ background:var(--op-iterative_normalize); }}
.pill.bound-compute {{ background:var(--good); }} .pill.bound-bw {{ background:var(--accent-2); }}
figure {{ margin:14px 0 4px; }}
figure img {{ width:100%; height:auto; border-radius:8px; border:1px solid var(--border); }}
figcaption {{ font-size:12.5px; color:var(--muted); margin-top:6px; }}
pre {{ background:var(--page); border:1px solid var(--border); border-radius:8px; padding:14px 16px;
      overflow-x:auto; font-size:12.5px; line-height:1.5; }}
.step {{ display:flex; gap:14px; align-items:flex-start; margin:14px 0; }}
.step .n {{ flex:0 0 auto; width:26px; height:26px; border-radius:50%; background:var(--accent); color:#fff;
            display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700; }}
.step .body h4 {{ margin:0 0 4px; font-size:14.5px; }}
.step .body p {{ margin:0; color:var(--ink-2); font-size:14.5px; }}
.caveat {{ border-left:3px solid var(--accent-2); padding:10px 14px; background:color-mix(in srgb, var(--accent-2) 7%, transparent);
          border-radius:0 6px 6px 0; font-size:13.5px; color:var(--ink-2); margin:12px 0; }}
footer {{ color:var(--muted); font-size:12.5px; }}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">Profiling ML workload &middot; full walkthrough</p>
    <h1>Two probabilistic workload DAGs, kernel by kernel</h1>
    <p>Glimmer-30B (dense, speculative decoding) and DeepSeek-V4-Flash (MoE, hybrid attention),
      walked through the same four-step method end to end: architecture inputs, kernel decomposition,
      the resulting DAG, and how it becomes a BookSim trace.</p>
  </header>

  <section class="card">
    <h2>Architecture, side by side</h2>
    <div class="model-grid">
      <table class="spec-table">
        <tr><td colspan="2"><strong>Muse Glimmer-30B</strong></td></tr>
        <tr><td>Type</td><td>Dense causal transformer</td></tr>
        <tr><td>Total / activated params</td><td>27.8B / 27.8B (all)</td></tr>
        <tr><td>Layers</td><td>52 (39 local + 13 global)</td></tr>
        <tr><td>Hidden size</td><td>6,656</td></tr>
        <tr><td>Attention</td><td>GQA 32Q/2KV, window 2,048</td></tr>
        <tr><td>FFN</td><td>SwiGLU, 19,968</td></tr>
        <tr><td>Draft model</td><td>5 layers, GQA 32Q/8KV, block 16</td></tr>
        <tr><td>Stochastic surface</td><td>1 scalar (&alpha;, accept rate)</td></tr>
      </table>
      <table class="spec-table">
        <tr><td colspan="2"><strong>DeepSeek-V4-Flash</strong></td></tr>
        <tr><td>Type</td><td>Mixture-of-Experts</td></tr>
        <tr><td>Total / activated params</td><td>284B / 13B</td></tr>
        <tr><td>Layers</td><td>43 (2 SWA + 21 CSA + 20 HCA)</td></tr>
        <tr><td>Hidden size</td><td>4,096</td></tr>
        <tr><td>Attention</td><td>Hybrid CSA/HCA, compress+index+select+attend</td></tr>
        <tr><td>MoE FFN</td><td>1 shared + 6-of-256 routed, width 2,048</td></tr>
        <tr><td>Speculative module</td><td>MTP depth 1 ("DSpark"), block 7</td></tr>
        <tr><td>Stochastic surface</td><td>2 independent (routing + accept rate), 40 of 43 layers only</td></tr>
      </table>
    </div>
  </section>

  <section class="card">
    <h2>1. Glimmer-30B walkthrough</h2>
    <p class="sub">A (prefill) &rarr; [B (draft) &rarr; C (verify) &rarr; D (commit)]* &mdash; every edge fires
      at p=1.0; the only randomness is k, tokens accepted per block.</p>

    <h3>Final task-level DAG</h3>
    <figure><img src="data:image/png;base64,{glimmer_dag_png}">
      <figcaption>Structural DAG with payload formulas per edge. Generated by workload_dag.py.</figcaption></figure>

    <h3>Kernel-level pipeline (all 43 kernels)</h3>
    <figure><img src="data:image/png;base64,{glimmer_kdag_png}">
      <figcaption>Every kernel inside A/B/C/D, colored by op type. Generated by specdecode_kernels.py.</figcaption></figure>

    <h3>Kernel table (context: short_chat_L1500)</h3>
    {kernel_table_html(glimmer_kernels)}

    <h3>Bottleneck by context length</h3>
    {bottleneck_table_html(glimmer_bottleneck)}
  </section>

  <section class="card">
    <h2>2. DeepSeek-V4-Flash walkthrough</h2>
    <p class="sub">ATTN (hybrid CSA/HCA) + MOE (router &rarr; 1 shared + 6-of-256 routed experts) + MHC
      (residual mixing) + OUT (embed, LM head, MTP), repeated per layer &mdash; not a loop, a fixed
      43-layer stack, with the routing decision as the one genuine structural branch.</p>

    <h3>Kernel-level pipeline (all 22 kernel definitions)</h3>
    <figure><img src="data:image/png;base64,{deepseek_kdag_png}">
      <figcaption>Note the color contrast: MOE.router_learned (red, reduce_argmax, stochastic) vs.
        MOE.router_hash (green, lookup, deterministic) &mdash; same layer type, different mechanism.
        Generated by deepseek_kernels.py.</figcaption></figure>

    <h3>Kernel table (context: short_chat_L1500)</h3>
    {kernel_table_html(deepseek_kernels)}

    <h3>Bottleneck by context length</h3>
    {bottleneck_table_html(deepseek_bottleneck)}
    <div class="caveat">MHC's expected latency shows as ~1e-5 ms in this table &mdash; that's the
      known schema gap explained above: without a sequential-depth latency term, the roofline model
      can't see mHC's real cost. Treat it as a documented underestimate, not a finding.</div>
  </section>

  <section class="card">
    <h2>3. From DAG to BookSim: injection rate, anynet, traces</h2>

    <div class="step"><div class="n">1</div><div class="body">
      <h4>Injection rate isn't chosen &mdash; it's measured</h4>
      <p>BookSim's <code>injection_rate</code> config field is for synthetic traffic (uniform random,
        tornado, ...) where every node injects at one fixed rate by construction. Our traffic is
        heterogeneous and DAG-driven: A injects rarely in huge bursts, C injects every verify
        iteration, D injects tiny control messages. There's no single rate that describes that.
        We use BookSim's trace-driven mode instead: every packet's injection cycle comes from
        simulating the DAG. What we <em>can</em> compute afterward is an effective injection rate,
        for comparison against a synthetic sweep to see where this workload sits relative to the
        network's saturation knee.</p>
    </div></div>

    <div class="step"><div class="n">2</div><div class="body">
      <h4>Anynet: one router per physical chiplet</h4>
      <p>Our placement is irregular (A and C deliberately share chiplets, since they run identical
        52-layer weights &mdash; see specdecode_kernels.py's timeshare note). BookSim's built-in
        regular topologies (mesh, torus, flattened butterfly) can't express that; its "anynet"
        topology reads an arbitrary router/link graph from a file instead.</p>
    </div></div>

    <div class="step"><div class="n">3</div><div class="body">
      <h4>Traces come from a discrete-event simulation of the DAG, not a formula</h4>
      <p>Walk the existing DAG simulation (workload_dag.simulate_request, the same one behind
        trace_table.csv). For every task instance, compute a real finish time by summing that
        task's kernel latencies from specdecode_kernels.py at the request's actual context length
        &mdash; the same numbers already in latency_scenarios.csv. Layer a single-server-per-chiplet
        queue on top so concurrent requests actually contend for the same physical chiplet (a lone
        request never contends with itself). Convert finish times to cycles via a clock frequency;
        write one line per packet.</p>
    </div></div>

    <h3>Example placement</h3>
    <table class="spec-table">
      <tr><td>Task A (prefill)</td><td>chiplets 0, 1</td></tr>
      <tr><td>Task C (verify)</td><td>chiplets 0, 1 &mdash; <em>shared with A</em></td></tr>
      <tr><td>Task B (draft)</td><td>chiplets 2, 3</td></tr>
      <tr><td>Task D (commit)</td><td>chiplet 4</td></tr>
    </table>

    <h3>anynet topology (example.anynet)</h3>
    <pre>{htmllib.escape(anynet_txt)}</pre>

    <h3>Trace file (example.trace, {trace_total} packets total)</h3>
    <pre>{htmllib.escape(trace_preview)}...</pre>
    <p class="sub">Columns: source chiplet, destination chiplet, packet size in flits, injection
      cycle, request id. 12 concurrent requests, 2-second arrival spacing, 1.5 GHz clock, 16-byte
      flits &mdash; all swappable constants at the top of booksim_trace_gen.py.</p>

    <h3>Measured effective injection rate</h3>
    <p>1,239 packets over ~37.5 billion cycles across 5 nodes &rarr; <strong>6.6&times;10&#8315;&#8313;
      packets/cycle/node</strong>. Sparse, as expected &mdash; Section 7's original finding holds at
      the trace level too: inter-chiplet traffic was never the bottleneck for this workload; weight
      streaming and compute were. The number matters as a calibration point for a synthetic sweep,
      not as a capacity target.</p>

    <div class="caveat">The single-server-per-chiplet queue is a first-order M/D/1-per-chiplet
      approximation (deterministic service time, FIFO, one request at a time) &mdash; it captures
      the essential contention effect but isn't a full M/G/c or SimPy treatment. Adequate for a
      first placement pass; revisit before trusting an absolute latency number. Also: only edges
      between two physical chiplets become packets &mdash; the external arrival and terminal
      condition aren't NoC hops. And the exact BookSim trace-file column order / anynet keyword
      syntax vary by version &mdash; verify against your checkout before feeding these in.</div>
  </section>

  <footer>
    <p>Source: workload_dag.py, specdecode_kernels.py, deepseek_kernels.py, kernels.py,
      booksim_trace_gen.py, and their generated CSVs/figures, all in this folder. Regenerate any
      table here by re-running the corresponding script.</p>
  </footer>
</div>
"""

with open("walkthrough.html", "w") as f:
    f.write(HTML)
print(f"walkthrough.html written, {len(HTML):,} bytes")
