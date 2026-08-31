#!/usr/bin/env python3
"""Assembles the final email-safe HTML body (inline styles only) for the
Gmail draft, using the table fragments from build_email_html.py."""

with open("_arch_table.html") as f:
    ARCH_TABLE = f.read()
with open("_bt_table.html") as f:
    BT_TABLE = f.read()

FONT = "font-family:Arial,Helvetica,sans-serif;"
INK = "#0b0b0b"
INK2 = "#3a3a38"
MUTED = "#6b6a66"
ACCENT = "#2a78d6"
BORDER = "#e1e0d9"
CARD = "#fcfcfb"

def p(text, extra=""):
    return f'<p style="{FONT}font-size:14.5px;line-height:1.6;color:{INK2};margin:0 0 14px;{extra}">{text}</p>'

def h2(text):
    return (f'<h2 style="{FONT}font-size:19px;color:{INK};margin:28px 0 6px;'
           f'border-bottom:2px solid {BORDER};padding-bottom:6px;">{text}</h2>')

def h3(text):
    return f'<h3 style="{FONT}font-size:15px;color:{INK};margin:18px 0 6px;">{text}</h3>'

def callout(text, color=ACCENT):
    return (f'<table cellpadding="0" cellspacing="0" style="width:100%;margin:14px 0;">'
           f'<tr><td style="border-left:3px solid {color};background:#f2f6fb;padding:10px 14px;'
           f'{FONT}font-size:13.5px;color:{INK2};line-height:1.55;">{text}</td></tr></table>')

BODY = f"""
<div style="{FONT}max-width:680px;margin:0 auto;padding:8px 4px;">

  <p style="{FONT}font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:{MUTED};margin:0 0 6px;">
    Profiling ML workload &middot; DAG-to-hardware-mapping method</p>
  <h1 style="{FONT}font-size:24px;color:{INK};margin:0 0 14px;line-height:1.25;">
    From a HuggingFace model card to a probabilistic hardware-mapping DAG &mdash; two worked examples</h1>

  {p("This is a walkthrough of a method for turning any HuggingFace model into a kernel-level, "
     "probabilistic workload DAG &mdash; the input format our floorplan / chiplet-placement search "
     "needs. The method is general, not tuned to one model: two contrasting worked examples follow, "
     "plus the four-step recipe both of them follow, so you can run this on a model of your own choosing.")}

  {callout("One correction up front, because it's easy to get wrong: a DAG is not &ldquo;one DAG per "
           "probability.&rdquo; A workload's DAG <b>template</b> (its nodes and edges) can have zero, one, "
           "or many independent stochastic decision points &mdash; and even multiple instances of what "
           "looks like &ldquo;the same layer type&rdquo; inside one model can differ in whether they're "
           "stochastic at all (see DeepSeek's hash-routed vs. learned-routed MoE layers below). "
           "Sweeping the <i>parameter</i> of one distribution gives you multiple operating points of "
           "the same template, not multiple templates.")}

  {h2("The general method, in four steps")}
  {p("<b>1. Pull static architecture inputs</b> from config.json &mdash; hidden size, layer count, "
     "attention/KV heads, FFN width, activation function, attention pattern, vocab size, precision. "
     "All deterministic, same every forward pass.")}
  {p("<b>2. Decompose each repeating block into kernels</b> &mdash; a mechanical recipe once step 1 "
     "is known: norm &rarr; qkv_proj[matmul] &rarr; attention &rarr; out_proj[matmul] &rarr; norm "
     "&rarr; ffn[matmul(s)+activation] &rarr; ffn_down[matmul]. Each kernel gets an op type, a FLOPs "
     "formula, and a bytes formula.")}
  {p("<b>3. Separate communication into three kinds</b> &mdash; weight bytes (read from local HBM, "
     "dominates at low batch), local I/O bytes (on-chip activation/KV), and inter-node bytes (the "
     "only kind that crosses the interconnect and feeds a NoC simulator). Conflating these is the "
     "classic mistake.")}
  {p("<b>4. Find the stochastic decision points</b> &mdash; NOT in config.json. This comes from the "
     "inference algorithm, and needs either a published closed form or actual profiling.")}

  {h2("Architecture, side by side")}
  {ARCH_TABLE}

  {h2("Example 1 &mdash; Muse Glimmer-30B (speculative decoding)")}
  {p("<b>What it is:</b> a dense causal transformer with an attached perception encoder, 27.8B "
     "parameters, <i>all</i> activated every forward pass. 52 layers, hidden size 6,656, GQA "
     "attention (32Q/2KV) with a repeating [local,local,local,global] pattern &mdash; 39 layers use "
     "a 2,048-token sliding window, 13 use full-context attention. Ships a small 5-layer draft model "
     "(32Q/8KV) for speculative decoding, proposing 16-token blocks.")}
  {p("<b>Why it's probabilistic:</b> the draft model guesses up to 16 tokens, the full model verifies "
     "in parallel and accepts a prefix. Tokens accepted per block, k, follows "
     "P(k=j) = &alpha;<sup>j</sup>(1&minus;&alpha;), where &alpha; is the per-token acceptance "
     "probability &mdash; the <i>entire</i> stochastic surface of this workload. Every DAG edge fires "
     "with probability 1; only the payload size is random.")}
  {p("<b>Kernel-level correction:</b> task B (draft) isn't one parallel pass over 16 positions &mdash; "
     "it's 16 <i>sequential</i> micro-steps, since the draft model can't know token 2 before "
     "generating token 1. Matters for placement: B's steps are serially dependent (limited "
     "pipelining value), while C's 52 layers over 16 positions run in true parallel.")}
  {p("<b>What the analysis found:</b> at short context (L=1,500), task A is compute-bound (its FFN "
     "matmul dominates); B and C are bandwidth-bound (streaming a full weight matrix for one token is "
     "the classic batch=1 memory-bound case). At long context (L=100,000), A's <i>dominant kernel "
     "itself flips</i> from FFN-compute to attention-bandwidth &mdash; reading back the growing KV "
     "cache becomes the bigger transfer than any weight matrix.")}

  {callout('<b>Diagram:</b> the full kernel-level pipeline for A/B/C/D (colored by op type '
           '&mdash; matmul blue, attention orange, norm gray, lookup green) is viewable at full '
           'resolution in the walkthrough link below, and sent to you separately as '
           '<code>kernel_dag_figure.png</code> to drag into this draft if you want it inline.', ACCENT)}

  {h2("Example 2 &mdash; DeepSeek-V4-Flash (MoE + hybrid attention + speculative decoding)")}
  {p("<b>What it is:</b> a Mixture-of-Experts model, 284B total parameters, only 13B "
     "<i>activated</i> per token &mdash; that gap is the tell for MoE vs. Glimmer's dense 27.8B "
     "where everything fires every time. 43 layers, hidden size 4,096. Every layer: 1 shared expert "
     "(always fires) + 256 routed experts (6 chosen per token). Supports 1M-token context via a "
     "hybrid attention scheme. Ships a multi-token-prediction module inference engines attach as a "
     "speculative-decoding drafter (branded &ldquo;DSpark&rdquo;), proposing 7-token blocks.")}
  {p("<b>Why it's probabilistic &mdash; two independent layers, a different kind of randomness "
     "than example 1:</b>")}
  {p("&nbsp;&nbsp;(a) <b>Expert routing</b> &mdash; &ldquo;which 6 of 256 experts&rdquo; is a genuine "
     "categorical distribution, a real structural branch (which edge fires), not just a payload-size "
     "scalar like &alpha;. This is exactly the traffic-equation machinery from the original toy "
     "example, which Glimmer's workload never needed.")}
  {p("&nbsp;&nbsp;(b) <b>But not every MoE layer is stochastic.</b> The first 3 layers use "
     "<b>Hash routing</b> &mdash; a deterministic function of token ID, a lookup, zero probability "
     "involved. Only the other 40 layers use learned, data-dependent routing that needs a profiled "
     "histogram to characterize. Two instances of &ldquo;the same layer type&rdquo; differing in whether "
     "they're random at all.")}
  {p("<b>Two structural surprises this model forced into the kernel vocabulary:</b> its hybrid "
     "attention is a 6-kernel <i>pipeline</i> (compress &rarr; index (FP4) &rarr; top-k select "
     "&rarr; attend &rarr; grouped output projection), not Glimmer's one fused attention op. And "
     "its residual-connection replacement (mHC) is 20 sequential Sinkhorn-Knopp steps on a tiny "
     "4&times;4 matrix &mdash; utterly negligible in FLOPs/bytes, but bound by "
     "<i>sequential dependency depth</i>, a resource axis neither compute-bound nor bandwidth-bound "
     "roofline math covers. Needed a new op type: <code>iterative_normalize</code>.")}

  {callout('<b>Diagram:</b> sent to you separately as <code>deepseek_dag_figure.png</code> '
           '&mdash; note the contrast between MOE.router_learned (red, reduce_argmax, stochastic) '
           'and MOE.router_hash (green, lookup, deterministic): same layer type, different '
           'mechanism. MHC.sinkhorn_iterate (purple) is the new op type.', ACCENT)}

  {h2("Bottleneck highlights (short-context regime, both models)")}
  {BT_TABLE}

  {h2("From DAG to BookSim: injection rate, anynet, traces")}
  {p("<b>Injection rate isn't chosen &mdash; it's measured.</b> BookSim's <code>injection_rate</code> "
     "field is for synthetic traffic (uniform random, tornado) where every node injects at one fixed "
     "rate. Our traffic is heterogeneous and DAG-driven, so we use trace-driven mode instead: every "
     "packet's injection cycle comes from simulating the DAG. The effective rate is then measured "
     "afterward (in our worked example: 6.6&times;10<sup>&minus;9</sup> packets/cycle/node &mdash; "
     "sparse, as expected) for comparison against a synthetic sweep.")}
  {p("<b>Anynet</b> is BookSim's arbitrary-topology mechanism &mdash; needed because our placement is "
     "irregular (A and C deliberately share physical chiplets, since they run identical 52-layer "
     "weights). <b>Traces</b> come from a real discrete-event simulation: walk the DAG, sum each "
     "task's kernel latencies at the request's actual context length, layer a per-chiplet queue on "
     "top so concurrent requests actually contend, convert to cycles, write one packet per line.")}

  {h2("Takeaways for your own model")}
  {p("&bull; Config.json gives you the deterministic half for free; the stochastic half is never in "
     "the config &mdash; it's a property of the inference algorithm.<br>"
     "&bull; Count the independent stochastic decision points before coding: zero (a vision encoder), "
     "one (Glimmer's &alpha;), or several at once (DeepSeek's routing + MTP).<br>"
     "&bull; Don't assume every instance of a layer <i>type</i> shares the same probabilistic "
     "behavior &mdash; DeepSeek's hash-routed vs. learned-routed MoE layers are the cleanest "
     "counterexample.<br>"
     "&bull; When a kernel's FLOPs and bytes both look negligible, check whether it's a sequential "
     "dependency chain instead (mHC) &mdash; a resource axis roofline math doesn't cover.")}

  {p(f'Full interactive walkthrough (all kernel tables in full, both complete DAG figures, the '
     f'anynet/trace files, and every caveat): '
     f'<a href="https://claude.ai/code/artifact/1015be86-702e-41fc-9a9e-7350db2e344f" '
     f'style="color:{ACCENT};">claude.ai/code/artifact/1015be86-702e-41fc-9a9e-7350db2e344f</a>. '
     f'All source code and CSVs are in the repo under <code>Profiling ML workload/</code>.')}

  {p("Happy to walk through building either of these out fully in code, or do the same pass live on "
     "a model one of you picks &mdash; MoE, dense, vision, or otherwise. The method doesn't change; "
     "only the numbers in step 1 and the shape of step 4 do.", f"margin-top:18px;")}

</div>
"""

with open("email_body.html", "w") as f:
    f.write(BODY)
print(f"email_body.html written, {len(BODY):,} bytes")
