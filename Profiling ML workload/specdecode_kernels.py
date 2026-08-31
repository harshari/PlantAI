#!/usr/bin/env python3
"""Kernel-level decomposition of the speculative-decoding workload (workload_dag.py).

workload_dag.py answers "which task fires next, with what payload" (Level 1 --
the task graph: A=prefill, B=draft, C=verify, D=commit). This file answers the
question one level down: "what actually runs INSIDE each task" (Level 2 -- the
kernel graph). Every task node is a short pipeline of named kernels, each with
a concrete op type, a FLOPs formula, a bytes formula, and a precision -- that's
what a hardware-mapping search actually needs to place work onto compute units
vs. cache vs. memory-bandwidth-provisioned tiles.

Cross-check performed below (see reconcile(), printed at import time): summing
this file's kernel-level FLOPs for tasks A and C should land close to the
Section-4 aggregate formulas from the original notes (5.0e10*L + 346112*L^2
for A, 16*5.0e10 + 5537792*L for C), since those were supplied as fixed
reference numbers without a shown derivation. It does NOT match exactly --
Section 4's quadratic term turns out to have been built from 4*d_model*L^2
per global layer (no causal-mask halving, using d_model=6656 as a stand-in for
attention width), whereas this file derives the quadratic term from the
model's actual q_dim = n_q_heads*head_dim = 4096 (not equal to d_model here,
because this GQA config projects down before splitting into heads) WITH causal
masking applied. The smaller q_dim and the missing causal halving partially
cancel, so the two derivations track closely at short context (ratio
~0.95-1.00x at L<=1,500, where the linear, non-attention term dominates
anyway) and diverge as L grows and the quadratic term takes over (ratio
~0.77x at L=65,536, ~0.70x at L=100,000 -- run reconcile(L) for exact numbers
at any L). Treat the kernel-level numbers below as the more architecturally
faithful ones going forward.

Model constants below are Muse Glimmer-30B; several draft-model dimensions are
still unresolved / assumed (matching the "Open items" already flagged in
explanation.txt) -- each is called out inline, not silently guessed.
"""

from __future__ import annotations

from kernels import (Kernel, BYTES_PER_ELEM, GENERIC_CHIPLET,
                     write_kernel_table, write_latency_table)

# ---------------------------------------------------------------------------
# Model constants (Muse Glimmer-30B) -- swap for your target model
# ---------------------------------------------------------------------------
D_MODEL = 6656
N_LOCAL, N_GLOBAL, WINDOW = 39, 13, 2048
HEAD_DIM = 128
N_Q_HEADS_MAIN, N_KV_HEADS_MAIN = 32, 2
N_Q_HEADS_DRAFT, N_KV_HEADS_DRAFT = 32, 8         # head_dim=128 for draft: ASSUMED shared
D_FFN_MAIN = 19968                                # given (SwiGLU intermediate)
D_FFN_DRAFT = None                                # UNRESOLVED -- not published (Open items)
D_MODEL_DRAFT = D_MODEL                           # ASSUMED: draft shares main's d_model
                                                   # (self-distilled-drafter pattern); not
                                                   # published either way
DRAFT_LAYERS = 5
BLOCK = 16
VOCAB_SIZE = 128_256                              # ASSUMED (Llama-3-style vocab).
                                                   # Not published for Glimmer-30B --
                                                   # placeholder until confirmed.
PRECISION = "bf16"
B = BYTES_PER_ELEM[PRECISION]

Q_DIM_MAIN = N_Q_HEADS_MAIN * HEAD_DIM              # 4096
KV_DIM_MAIN = N_KV_HEADS_MAIN * HEAD_DIM            # 256
Q_DIM_DRAFT = N_Q_HEADS_DRAFT * HEAD_DIM            # 4096
KV_DIM_DRAFT = N_KV_HEADS_DRAFT * HEAD_DIM          # 1024

# ---------------------------------------------------------------------------
# Context regimes: how "how long the iteration runs" / "what we're asking
# for" changes which kernel dominates. block_L is fixed at BLOCK per the spec;
# L is context position, the thing that actually moves the bottleneck.
# ---------------------------------------------------------------------------
CONTEXT_REGIMES = {
    "short_chat_L1500":     {"L": 1500,   "block_L": BLOCK},   # Azure coding-trace median
    "long_agent_L32768":    {"L": 32768,  "block_L": BLOCK},   # extended agent session
    "long_agent_L100000":   {"L": 100000, "block_L": BLOCK},   # long-context tail
}

# ---------------------------------------------------------------------------
# FLOPs helpers
# ---------------------------------------------------------------------------

def _matmul_flops(tokens, d_in, d_out):
    return 2.0 * tokens * d_in * d_out


def _norm_flops(tokens, d):
    return 5.0 * tokens * d                              # standard small-constant estimate


def _attn_flops_prefill(qdim, Lq, window=None):
    """Causal self-attention over Lq positions, ONE layer. window=None -> global
    (full causal L^2/2); window=W -> local (each query sees <=W prior tokens,
    already capped so no further causal halving needed once Lq>W)."""
    if window is None:
        return 4.0 * qdim * Lq * Lq * 0.5                 # causal halving
    if Lq <= window:
        return 4.0 * qdim * Lq * Lq * 0.5
    return 4.0 * qdim * Lq * window                       # capped, ~flat beyond window


def _attn_flops_decode(qdim, Lq, Lkv, window=None):
    """Lq NEW query positions attending back to Lkv cached positions (verify /
    draft micro-step). window=None -> global (sees full Lkv); window=W ->
    local (capped at W regardless of Lkv). No intra-block causal halving
    applied (Lq<<Lkv makes this a small approximation, noted here)."""
    Lkv_eff = Lkv if window is None else min(Lkv, window)
    return 4.0 * qdim * Lq * Lkv_eff

# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------
KERNELS: list[Kernel] = []


def _add(**kw):
    KERNELS.append(Kernel(**kw))


# ---- Task A: prefill (main model, one parallel pass over L tokens) --------
_add(id="A.embed_lookup", task="A", layer_class="embedding", op_type="lookup",
     multiplicity_fn=lambda c: 1,
     flops_fn=lambda c: 0.0,
     weight_bytes_fn=lambda c: c["L"] * D_MODEL * B,     # rows touched in the embed table
     io_bytes_fn=lambda c: c["L"] * D_MODEL * B,
     precision=PRECISION,
     note="Gather, not compute: touches only the L rows of the embedding "
          "table that are actually used, not the full table.")

for name, window in [("local_attn_layer", WINDOW), ("global_attn_layer", None)]:
    mult = N_LOCAL if window is not None else N_GLOBAL
    tag = "local" if window is not None else "global"
    _add(id=f"A.{name}.rmsnorm_attn", task="A", layer_class=name, op_type="norm",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _norm_flops(c["L"], D_MODEL),
         weight_bytes_fn=lambda c: D_MODEL * B,
         io_bytes_fn=lambda c: 2 * c["L"] * D_MODEL * B,
         precision=PRECISION, note="Pre-attention RMSNorm.")
    _add(id=f"A.{name}.qkv_proj", task="A", layer_class=name, op_type="matmul",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _matmul_flops(c["L"], D_MODEL, Q_DIM_MAIN + 2 * KV_DIM_MAIN),
         weight_bytes_fn=lambda c: D_MODEL * (Q_DIM_MAIN + 2 * KV_DIM_MAIN) * B,
         io_bytes_fn=lambda c: c["L"] * (D_MODEL + Q_DIM_MAIN + 2 * KV_DIM_MAIN) * B,
         precision=PRECISION, note=f"{tag} layer, GQA {N_Q_HEADS_MAIN}Q/{N_KV_HEADS_MAIN}KV.")
    _add(id=f"A.{name}.attention", task="A", layer_class=name, op_type="attention",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=(lambda c, w=window: _attn_flops_prefill(Q_DIM_MAIN, c["L"], w)),
         weight_bytes_fn=lambda c: 0.0,
         io_bytes_fn=(lambda c, w=window: c["L"] * (min(c["L"], w) if w else c["L"])
                     * HEAD_DIM * 2 * B),   # K+V read for the attended window
         precision=PRECISION,
         note=("Windowed causal self-attention, w=2048." if window else
               "Full-context causal self-attention -- the quadratic-in-L term."))
    _add(id=f"A.{name}.out_proj", task="A", layer_class=name, op_type="matmul",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _matmul_flops(c["L"], Q_DIM_MAIN, D_MODEL),
         weight_bytes_fn=lambda c: Q_DIM_MAIN * D_MODEL * B,
         io_bytes_fn=lambda c: c["L"] * (Q_DIM_MAIN + D_MODEL) * B,
         precision=PRECISION, note="Attention output projection.")
    _add(id=f"A.{name}.rmsnorm_ffn", task="A", layer_class=name, op_type="norm",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _norm_flops(c["L"], D_MODEL),
         weight_bytes_fn=lambda c: D_MODEL * B,
         io_bytes_fn=lambda c: 2 * c["L"] * D_MODEL * B,
         precision=PRECISION, note="Pre-FFN RMSNorm.")
    _add(id=f"A.{name}.ffn_gate_up", task="A", layer_class=name, op_type="matmul",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _matmul_flops(c["L"], D_MODEL, 2 * D_FFN_MAIN),
         weight_bytes_fn=lambda c: D_MODEL * 2 * D_FFN_MAIN * B,
         io_bytes_fn=lambda c: c["L"] * (D_MODEL + 2 * D_FFN_MAIN) * B,
         precision=PRECISION, note="SwiGLU gate+up projections, d_ffn=19,968.")
    _add(id=f"A.{name}.swiglu_act", task="A", layer_class=name, op_type="elementwise",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: 3.0 * c["L"] * D_FFN_MAIN,   # silu(gate)*up: ~3 ops/elem
         weight_bytes_fn=lambda c: 0.0,
         io_bytes_fn=lambda c: 3 * c["L"] * D_FFN_MAIN * B,
         precision=PRECISION, note="SwiGLU activation, elementwise.")
    _add(id=f"A.{name}.ffn_down", task="A", layer_class=name, op_type="matmul",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _matmul_flops(c["L"], D_FFN_MAIN, D_MODEL),
         weight_bytes_fn=lambda c: D_FFN_MAIN * D_MODEL * B,
         io_bytes_fn=lambda c: c["L"] * (D_FFN_MAIN + D_MODEL) * B,
         precision=PRECISION, note="FFN down-projection back to d_model.")

_add(id="A.final_norm", task="A", layer_class="output", op_type="norm",
     multiplicity_fn=lambda c: 1,
     flops_fn=lambda c: _norm_flops(c["L"], D_MODEL),
     weight_bytes_fn=lambda c: D_MODEL * B,
     io_bytes_fn=lambda c: 2 * c["L"] * D_MODEL * B,
     precision=PRECISION,
     note="No LM head in A -- A hands off a hidden state to B, not a token "
          "(see A->B edge). Logits are only needed in B (draft proposals) "
          "and C (verify + bonus token).")

# ---- Task B: draft, 16 sequential micro-steps of a 5-layer model ----------
# The "16*" factor in the original Section-4 formula for B ("16 *
# (unresolved)") already implied this: the draft model is autoregressive
# within itself and cannot propose the whole block in one parallel pass --
# it proposes token 1, feeds it back, proposes token 2, ... x16.
for name in ["draft_layer"]:
    mult = lambda c: DRAFT_LAYERS * c["block_L"]           # 5 layers x 16 micro-steps
    _add(id=f"B.{name}.qkv_proj", task="B", layer_class=name, op_type="matmul",
         multiplicity_fn=mult,
         flops_fn=lambda c: _matmul_flops(1, D_MODEL_DRAFT, Q_DIM_DRAFT + 2 * KV_DIM_DRAFT),
         weight_bytes_fn=lambda c: D_MODEL_DRAFT * (Q_DIM_DRAFT + 2 * KV_DIM_DRAFT) * B,
         io_bytes_fn=lambda c: (D_MODEL_DRAFT + Q_DIM_DRAFT + 2 * KV_DIM_DRAFT) * B,
         precision=PRECISION,
         note=f"GQA {N_Q_HEADS_DRAFT}Q/{N_KV_HEADS_DRAFT}KV. d_model_draft ASSUMED "
              f"= main's ({D_MODEL_DRAFT}) -- not published.")
    _add(id=f"B.{name}.attention", task="B", layer_class=name, op_type="attention",
         multiplicity_fn=mult,
         flops_fn=lambda c: _attn_flops_decode(Q_DIM_DRAFT, 1, c["L"], window=WINDOW),
         weight_bytes_fn=lambda c: 0.0,
         io_bytes_fn=lambda c: min(c["L"], WINDOW) * HEAD_DIM * 2 * B,
         precision=PRECISION,
         note="ASSUMED windowed (local-only) attention throughout the draft "
              "model; not stated explicitly for the draft in the model card.")
    _add(id=f"B.{name}.out_proj", task="B", layer_class=name, op_type="matmul",
         multiplicity_fn=mult,
         flops_fn=lambda c: _matmul_flops(1, Q_DIM_DRAFT, D_MODEL_DRAFT),
         weight_bytes_fn=lambda c: Q_DIM_DRAFT * D_MODEL_DRAFT * B,
         io_bytes_fn=lambda c: (Q_DIM_DRAFT + D_MODEL_DRAFT) * B,
         precision=PRECISION, note="Draft attention output projection.")
    _add(id=f"B.{name}.ffn", task="B", layer_class=name, op_type="matmul",
         multiplicity_fn=mult,
         flops_fn=lambda c: None,       # UNRESOLVED: d_ffn_draft not published
         weight_bytes_fn=lambda c: 0.0,  # unresolved -> excluded from byte totals too
         io_bytes_fn=lambda c: 0.0,
         precision=PRECISION,
         note="UNRESOLVED: draft FFN intermediate width not published. "
              "Inspect GGUF tensor shapes to fill in (see Open items).")

_add(id="B.embed_lookup", task="B", layer_class="embedding", op_type="lookup",
     multiplicity_fn=lambda c: c["block_L"],
     flops_fn=lambda c: 0.0,
     weight_bytes_fn=lambda c: D_MODEL_DRAFT * B,
     io_bytes_fn=lambda c: D_MODEL_DRAFT * B,
     precision=PRECISION,
     note="One gather per micro-step, for the just-committed/proposed token.")
_add(id="B.lm_head", task="B", layer_class="output", op_type="matmul",
     multiplicity_fn=lambda c: c["block_L"],
     flops_fn=lambda c: _matmul_flops(1, D_MODEL_DRAFT, VOCAB_SIZE),
     weight_bytes_fn=lambda c: D_MODEL_DRAFT * VOCAB_SIZE * B,
     io_bytes_fn=lambda c: (D_MODEL_DRAFT + VOCAB_SIZE) * B,
     precision=PRECISION,
     note="Draft logits, one micro-step at a time. VOCAB_SIZE=128,256 ASSUMED "
          "(Llama-3-style) -- not published.")

# ---- Task C: verify (main model, ONE parallel pass over 16 new positions) -
for name, window in [("local_attn_layer", WINDOW), ("global_attn_layer", None)]:
    mult = N_LOCAL if window is not None else N_GLOBAL
    tag = "local" if window is not None else "global"
    _add(id=f"C.{name}.rmsnorm_attn", task="C", layer_class=name, op_type="norm",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _norm_flops(c["block_L"], D_MODEL),
         weight_bytes_fn=lambda c: D_MODEL * B,
         io_bytes_fn=lambda c: 2 * c["block_L"] * D_MODEL * B,
         precision=PRECISION, note="Pre-attention RMSNorm, 16 new positions.")
    _add(id=f"C.{name}.qkv_proj", task="C", layer_class=name, op_type="matmul",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _matmul_flops(c["block_L"], D_MODEL, Q_DIM_MAIN + 2 * KV_DIM_MAIN),
         weight_bytes_fn=lambda c: D_MODEL * (Q_DIM_MAIN + 2 * KV_DIM_MAIN) * B,
         io_bytes_fn=lambda c: c["block_L"] * (D_MODEL + Q_DIM_MAIN + 2 * KV_DIM_MAIN) * B,
         precision=PRECISION,
         note=f"Same weights as A.{name}.qkv_proj -- A and C are excellent "
              f"candidates to timeshare one chiplet class (see explanation.txt).")
    _add(id=f"C.{name}.attention", task="C", layer_class=name, op_type="attention",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=(lambda c, w=window: _attn_flops_decode(Q_DIM_MAIN, c["block_L"], c["L"], w)),
         weight_bytes_fn=lambda c: 0.0,
         io_bytes_fn=(lambda c, w=window: (min(c["L"], w) if w else c["L"])
                     * HEAD_DIM * 2 * B),
         precision=PRECISION,
         note=("Windowed -- roughly flat in L, capped at w=2048." if window else
               "16 new positions attend to the FULL context L -- linear-in-L, "
               "not quadratic (only 16 new queries, not L of them)."))
    _add(id=f"C.{name}.out_proj", task="C", layer_class=name, op_type="matmul",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _matmul_flops(c["block_L"], Q_DIM_MAIN, D_MODEL),
         weight_bytes_fn=lambda c: Q_DIM_MAIN * D_MODEL * B,
         io_bytes_fn=lambda c: c["block_L"] * (Q_DIM_MAIN + D_MODEL) * B,
         precision=PRECISION, note="Attention output projection.")
    _add(id=f"C.{name}.rmsnorm_ffn", task="C", layer_class=name, op_type="norm",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _norm_flops(c["block_L"], D_MODEL),
         weight_bytes_fn=lambda c: D_MODEL * B,
         io_bytes_fn=lambda c: 2 * c["block_L"] * D_MODEL * B,
         precision=PRECISION, note="Pre-FFN RMSNorm.")
    _add(id=f"C.{name}.ffn_gate_up", task="C", layer_class=name, op_type="matmul",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _matmul_flops(c["block_L"], D_MODEL, 2 * D_FFN_MAIN),
         weight_bytes_fn=lambda c: D_MODEL * 2 * D_FFN_MAIN * B,
         io_bytes_fn=lambda c: c["block_L"] * (D_MODEL + 2 * D_FFN_MAIN) * B,
         precision=PRECISION, note="Same weights as A's FFN -- see timeshare note above.")
    _add(id=f"C.{name}.swiglu_act", task="C", layer_class=name, op_type="elementwise",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: 3.0 * c["block_L"] * D_FFN_MAIN,
         weight_bytes_fn=lambda c: 0.0,
         io_bytes_fn=lambda c: 3 * c["block_L"] * D_FFN_MAIN * B,
         precision=PRECISION, note="SwiGLU activation, elementwise.")
    _add(id=f"C.{name}.ffn_down", task="C", layer_class=name, op_type="matmul",
         multiplicity_fn=lambda c, m=mult: m,
         flops_fn=lambda c: _matmul_flops(c["block_L"], D_FFN_MAIN, D_MODEL),
         weight_bytes_fn=lambda c: D_FFN_MAIN * D_MODEL * B,
         io_bytes_fn=lambda c: c["block_L"] * (D_FFN_MAIN + D_MODEL) * B,
         precision=PRECISION, note="FFN down-projection.")

_add(id="C.embed_lookup", task="C", layer_class="embedding", op_type="lookup",
     multiplicity_fn=lambda c: 1,
     flops_fn=lambda c: 0.0,
     weight_bytes_fn=lambda c: c["block_L"] * D_MODEL * B,
     io_bytes_fn=lambda c: c["block_L"] * D_MODEL * B,
     precision=PRECISION, note="Gather for the 16 proposed-block token IDs.")
_add(id="C.lm_head", task="C", layer_class="output", op_type="matmul",
     multiplicity_fn=lambda c: 1,
     flops_fn=lambda c: _matmul_flops(c["block_L"], D_MODEL, VOCAB_SIZE),
     weight_bytes_fn=lambda c: D_MODEL * VOCAB_SIZE * B,
     io_bytes_fn=lambda c: c["block_L"] * (D_MODEL + VOCAB_SIZE) * B,
     precision=PRECISION,
     note="Target-model logits at all 16 positions in one matmul -- needed "
          "for the rejection-sampling ratio AND the bonus token. VOCAB_SIZE "
          "ASSUMED (see B.lm_head).")

# ---- Task D: commit -- comparison only, no forward pass -------------------
_add(id="D.reject_sample_compare", task="D", layer_class="control", op_type="reduce_argmax",
     multiplicity_fn=lambda c: 1,
     flops_fn=lambda c: c["block_L"] * VOCAB_SIZE * 1.0,   # compare, ~1 FLOP/elem
     weight_bytes_fn=lambda c: 0.0,
     io_bytes_fn=lambda c: c["block_L"] * VOCAB_SIZE * 4 * 2,   # draft+target probs, fp32
     precision="fp32",
     note="Rejection-sampling ratio test per position: draft vs. target "
          "probability. Tiny FLOPs; the real cost of D is the KV-cache write "
          "already modeled as the C->D inter-node edge in workload_dag.py.")

# ---------------------------------------------------------------------------
# Reconciliation against Section-4's task-level aggregate formulas
# ---------------------------------------------------------------------------

def reconcile(L: int = 16384) -> None:
    ctx = {"L": L, "block_L": BLOCK}
    a_kernel_total = sum(k.flops_total(ctx) or 0 for k in KERNELS if k.task == "A")
    c_kernel_total = sum(k.flops_total(ctx) or 0 for k in KERNELS if k.task == "C")
    a_notes = 5.0e10 * L + 346112 * L * L
    c_notes = 16 * 5.0e10 + 5537792 * L
    print(f"[reconcile] L={L}")
    print(f"  Task A: kernel-level={a_kernel_total:.4e}  Section-4 aggregate={a_notes:.4e}  "
          f"ratio={a_kernel_total/a_notes:.2f}x")
    print(f"  Task C: kernel-level={c_kernel_total:.4e}  Section-4 aggregate={c_notes:.4e}  "
          f"ratio={c_kernel_total/c_notes:.2f}x")
    print("  (Section 4's quadratic/linear terms were supplied without a shown "
          "derivation; this kernel-level build derives from q_dim=4096 with "
          "causal masking, vs. Section 4's apparent 4*d_model*L^2 without it -- "
          "see this file's module docstring.)")

# ---------------------------------------------------------------------------
# Bottleneck-by-regime rollup: which kernel dominates each task's latency,
# and is it compute- or bandwidth-bound, per context regime.
# ---------------------------------------------------------------------------

def bottleneck_summary(hw=GENERIC_CHIPLET) -> list[dict]:
    rows = []
    for regime, ctx in CONTEXT_REGIMES.items():
        for task in ["A", "B", "C", "D"]:
            tks = [k for k in KERNELS if k.task == task]
            totals = [(k, k.expected_latency_ms(ctx, hw) * k.multiplicity(ctx)) for k in tks]
            totals.sort(key=lambda t: -t[1])
            task_total = sum(t for _, t in totals)
            top_k, top_ms = totals[0]
            bound = top_k.latency_scenarios(ctx, hw)[0][2]
            rows.append({
                "context_regime": regime, "task": task,
                "task_expected_latency_ms": f"{task_total:.4g}",
                "dominant_kernel": top_k.id,
                "dominant_kernel_share": f"{top_ms/task_total:.1%}" if task_total else "n/a",
                "dominant_bound_type": bound,
            })
    return rows


def write_bottleneck_csv(path: str = "bottleneck_summary.csv") -> None:
    import csv
    rows = bottleneck_summary()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# ---------------------------------------------------------------------------
# Figure: kernel pipeline strip per task
# ---------------------------------------------------------------------------

OP_COLORS = {
    "matmul": "#2a78d6", "attention": "#eb6834", "norm": "#9a9890",
    "elementwise": "#c3c2b7", "lookup": "#1baf7a", "reduce_argmax": "#e34948",
    "memory": "#4a3aa7",
}


def draw_kernel_figure(basename: str = "kernel_dag_figure") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch

    ink, muted = "#0b0b0b", "#52514e"
    task_labels = {
        "A": "A -- prefill (52-layer fwd over L tokens, one parallel pass)",
        "B": "B -- draft (5-layer fwd, 16 sequential micro-steps)",
        "C": "C -- verify (52-layer fwd over 16 new positions, one parallel pass)",
        "D": "D -- commit (rejection-sample compare; KV write is an inter-node edge)",
    }
    # one representative chain per task: local-layer group, global-layer
    # group (if distinct), then any task-level-only kernels, collapsed by
    # layer_class so 52 layers don't become 52 boxes.
    chains = {}
    for task in ["A", "B", "C", "D"]:
        seen_lc, chain = set(), []
        for k in KERNELS:
            if k.task != task:
                continue
            if k.layer_class in seen_lc and k.layer_class not in ("embedding", "output", "control"):
                continue
            seen_lc.add(k.layer_class)
            lc_kernels = [x for x in KERNELS if x.task == task and x.layer_class == k.layer_class]
            chain.append((k.layer_class, lc_kernels))
        chains[task] = chain

    fig, axes = plt.subplots(4, 1, figsize=(16, 11.5))
    ctx_ref = CONTEXT_REGIMES["short_chat_L1500"]
    for ax, task in zip(axes, ["A", "B", "C", "D"]):
        ax.set_xlim(0, 16); ax.set_ylim(0, 1.6); ax.axis("off")
        ax.set_title(task_labels[task], fontsize=11.5, color=ink, loc="left", pad=4)
        x = 0.3
        for layer_class, lc_kernels in chains[task]:
            mult = lc_kernels[0].multiplicity(ctx_ref)
            group_w = 0.05 + 1.55 * len(lc_kernels)
            gx0 = x
            for k in lc_kernels:
                w = 1.5
                box = FancyBboxPatch((x, 0.35), w, 0.85, boxstyle="round,pad=0.04",
                                     fc=OP_COLORS.get(k.op_type, "#ccc"), ec="white", lw=1.0)
                ax.add_patch(box)
                short = k.id.split(".")[-1].replace("_", "\n")
                ax.text(x + w / 2, 0.775, short, ha="center", va="center",
                        fontsize=6.7, color="white", fontweight="bold")
                x += w + 0.12
            group_w = x - gx0 - 0.12
            if mult > 1:
                ax.add_patch(plt.Rectangle((gx0 - 0.08, 0.28), group_w + 0.16, 1.0,
                                           fill=False, ec=muted, lw=1.1, linestyle=(0, (3, 2))))
                ax.text(gx0 + group_w / 2, 1.36, f"x {mult}  ({layer_class})",
                        ha="center", fontsize=8.3, color=muted, fontweight="bold")
            x += 0.28
        ax.set_xlim(0, max(x + 0.3, 4))

    handles = [Patch(fc=c, label=op) for op, c in OP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9.5, frameon=False,
              bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Kernel-level pipeline per task -- Muse Glimmer-30B speculative decoding",
                fontsize=14, color=ink, y=0.995)
    fig.text(0.5, 0.965,
             "Each box is one kernel: op type (color), FLOPs/bytes/precision in kernel_table.csv, "
             "latency scenarios in latency_scenarios.csv. Dashed groups repeat per layer-class.",
             ha="center", fontsize=9, color=muted)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(f"{basename}.png", dpi=200, facecolor="white")
    fig.savefig(f"{basename}.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    reconcile(16384)
    write_kernel_table(KERNELS, CONTEXT_REGIMES["short_chat_L1500"],
                       "kernel_table.csv", context_label="short_chat_L1500")
    write_latency_table(KERNELS, CONTEXT_REGIMES, "latency_scenarios.csv")
    write_bottleneck_csv("bottleneck_summary.csv")
    draw_kernel_figure("kernel_dag_figure")
    print(f"[kernels]  {len(KERNELS)} kernels across A/B/C/D")
    print("[files]    kernel_table.csv, latency_scenarios.csv, "
          "bottleneck_summary.csv, kernel_dag_figure.png/svg")
