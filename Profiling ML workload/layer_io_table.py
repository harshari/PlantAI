#!/usr/bin/env python3
"""Per-layer table for Glimmer-30B: weight (layer) size, compute, and OUTPUT
activation size -- distinct from kernel_table.csv's io_bytes column, which
conflates input reads and output writes together. Here, output_bytes is
specifically what that kernel WRITES (the activation tensor handed to the
next kernel), computed from its actual output shape, not a combined estimate.

Reuses specdecode_kernels.KERNELS for id/task/op_type/precision/multiplicity/
FLOPs/weight-bytes (unchanged, already correct); adds an output-shape map
here rather than touching the shared Kernel schema, since "output activation
bytes alone" is a narrower question than the general io_bytes framework
answers.
"""
import csv
from specdecode_kernels import (KERNELS, CONTEXT_REGIMES, D_MODEL, Q_DIM_MAIN,
                                KV_DIM_MAIN, D_FFN_MAIN, D_MODEL_DRAFT, Q_DIM_DRAFT,
                                KV_DIM_DRAFT, VOCAB_SIZE, BLOCK)
from kernels import BYTES_PER_ELEM

CTX = CONTEXT_REGIMES["short_chat_L1500"]  # L=1500, block_L=16
B_BF16 = BYTES_PER_ELEM["bf16"]

# output activation SHAPE per kernel: (tokens_expr, out_dim, note)
# tokens_expr: "L" (task A, one pass over context), "block_L" (task C, one
# pass over the 16-token block), "1" (task B, per micro-step)
def out_shape(kernel_id, ctx):
    L, block_L = ctx["L"], ctx["block_L"]
    tok = {"A": L, "C": block_L, "B": 1, "D": 1}[kernel_id.split(".")[0]]

    if kernel_id.endswith("embed_lookup"):
        d = D_MODEL_DRAFT if kernel_id.startswith("B.") else D_MODEL
        return tok, d, "embedding vector"
    if ".rmsnorm_" in kernel_id or kernel_id.endswith("final_norm"):
        return tok, D_MODEL, "normalized hidden state"
    if kernel_id.endswith("qkv_proj"):
        if kernel_id.startswith("B."):
            return tok, Q_DIM_DRAFT + 2 * KV_DIM_DRAFT, "Q+K+V, concatenated"
        return tok, Q_DIM_MAIN + 2 * KV_DIM_MAIN, "Q+K+V, concatenated"
    if kernel_id.endswith(".attention"):
        qdim = Q_DIM_DRAFT if kernel_id.startswith("B.") else Q_DIM_MAIN
        return tok, qdim, "attention output, pre-out-proj"
    if kernel_id.endswith("out_proj"):
        d = D_MODEL_DRAFT if kernel_id.startswith("B.") else D_MODEL
        return tok, d, "attention output, post-projection"
    if kernel_id.endswith("ffn_gate_up"):
        return tok, 2 * D_FFN_MAIN, "gate+up, pre-activation"
    if kernel_id.endswith("swiglu_act"):
        return tok, D_FFN_MAIN, "gated FFN hidden state"
    if kernel_id.endswith("ffn_down"):
        return tok, D_MODEL, "FFN output, back to d_model"
    if kernel_id == "B.draft_layer.ffn":
        return tok, None, "UNRESOLVED -- d_ffn_draft not published"
    if kernel_id.endswith("lm_head"):
        return tok, VOCAB_SIZE, "logits"
    if kernel_id == "D.reject_sample_compare":
        return block_L, 1, "accept/reject decision per position (int, not a full tensor)"
    return tok, None, "?"


def fnum(x):
    if x is None:
        return "?"
    if x >= 1e6:
        return f"{x:.3e}"
    return f"{x:,.0f}" if x == int(x) else f"{x:,.3g}"


rows = []
for k in KERNELS:
    mult = k.multiplicity(CTX)
    flops = k.flops_fn(CTX)
    wbytes = k.weight_bytes_fn(CTX)
    tok, out_dim, note = out_shape(k.id, CTX)
    if out_dim is None:
        out_bytes = None
    else:
        prec_bytes = BYTES_PER_ELEM.get(k.precision, B_BF16)
        out_bytes = tok * out_dim * prec_bytes
    rows.append({
        "kernel_id": k.id, "task": k.task, "op_type": k.op_type,
        "datatype": k.precision, "multiplicity": mult,
        "weight_shape_elems": None if wbytes == 0 else round(wbytes / B_BF16 if k.precision == "bf16" else wbytes / BYTES_PER_ELEM.get(k.precision, B_BF16)),
        "layer_size_bytes_per_instance": wbytes,
        "compute_flops_per_instance": flops,
        "output_tokens": tok, "output_dim": out_dim,
        "output_activation_bytes_per_instance": out_bytes,
        "output_note": note,
    })

with open("layer_io_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# console table, grouped by task
print(f"{'kernel':38} {'dtype':6} {'x':>4} {'layer size (B)':>16} {'compute (FLOPs)':>17} {'out shape':>14} {'out bytes':>12}  note")
for task in ["A", "B", "C", "D"]:
    print(f"--- Task {task} ---")
    for r in rows:
        if r["task"] != task:
            continue
        shape = f'{r["output_tokens"]}x{r["output_dim"]}' if r["output_dim"] else "?"
        print(f'{r["kernel_id"]:38} {r["datatype"]:6} x{r["multiplicity"]:<3} '
              f'{fnum(r["layer_size_bytes_per_instance"]):>16} {fnum(r["compute_flops_per_instance"]):>17} '
              f'{shape:>14} {fnum(r["output_activation_bytes_per_instance"]):>12}  {r["output_note"]}')
print(f"\nwrote layer_io_table.csv, {len(rows)} rows, context={CTX}")
