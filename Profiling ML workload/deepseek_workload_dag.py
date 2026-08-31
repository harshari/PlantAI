#!/usr/bin/env python3
"""DSpark speculative-decoding loop for DeepSeek-V4-Flash, mirroring
workload_dag.py's simulate_request() but built on deepseek_kernels.py's
KERNELS instead of specdecode_kernels.py's. Answers "all combinations" of
context length x accept-rate alpha for DeepSeek, at the same trace-file
granularity already built for Glimmer (payload_bytes, L_context, k_accepted,
cum_output_tokens, src_task_compute_flops).

DAG (parallel to Glimmer's A -> [B->C->D]*):
  PREFILL (once)  -> [MTP_DRAFT (7 sequential micro-steps) -> VERIFY (one
  parallel pass over 7 positions) -> COMMIT (accept k of 7)]* -> exit

Three things are NOT reused from Glimmer and had to be derived fresh:

1. KV bytes/token. Glimmer's 53,248 B/token doesn't apply -- DeepSeek's KV
   cache is compressed (CSA compresses every 4 raw tokens into 1 entry, HCA
   every 128), so the marginal, AMORTIZED cost per new raw token is:
     CSA layer: (c*1B_fp8 + c_I*0.5B_fp4) / m   = (512 + 64) / 4   = 144 B
     HCA layer: (c*1B_fp8) / m'                 = 512 / 128        = 4 B
   Summed across 21 CSA + 20 HCA layers = 21*144 + 20*4 = 3,104 B/token.
   The 2 SWA-only layers and the sliding-window branch on every hybrid layer
   are EXCLUDED here -- they're a fixed-size window (128 tokens), not
   growing with L, so they don't contribute to marginal per-token KV growth
   (same logic Glimmer used to exclude local-layer KV beyond the window).
   3,104 B/token vs Glimmer's 53,248 is a ~17x reduction -- which is the
   right order of magnitude for the paper's own headline claim ("~10% of
   the KV cache of DeepSeek-V3.2 at 1M context"); this wasn't tuned to hit
   that number, it falls out of the compression ratios DeepSeek gave.

2. MTP draft-step cost. Real MTP modules (per the DeepSeek-V3 design this
   paper says it inherits unchanged) are ONE lightweight extra transformer
   block reusing the shared embedding/output head -- NOT a full 43-layer
   pass, and structurally different from Glimmer's separate small draft
   model. Exact MTP module width isn't published. APPROXIMATED here as
   1/43rd of a full single-token forward pass (i.e. "one layer's worth"),
   run DSPARK_BLOCK=7 times sequentially per iteration, matching the "16*"
   -> "16 sequential micro-steps" correction already applied to Glimmer's
   task B. Flagged clearly: this is an order-of-magnitude stand-in, not a
   sourced number -- replace once the MTP module's real width is known.

3. Routing during the loop. Per the earlier decision: uniform routing
   assumed (each of 256 experts equally likely) -- MOE.router_learned's
   FLOPs/bytes already reflect "compute the affinity score", which doesn't
   change under this assumption; what the assumption actually affects is
   downstream expert-placement load-balance, out of scope for this file.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, asdict

from deepseek_kernels import (KERNELS, D_MODEL, N_LAYERS, N_CSA, N_HCA,
                              N_SWA_ONLY, M_CSA, M_HCA, C, C_I, VOCAB_SIZE)
from kernels import BYTES_PER_ELEM

DSPARK_BLOCK = 7          # model card: num_speculative_tokens
B_BF16 = BYTES_PER_ELEM["bf16"]
B_FP8 = BYTES_PER_ELEM["fp8"]
B_FP4 = BYTES_PER_ELEM["fp4"]

# --- derived formulas (see module docstring for the math) ------------------
KV_BYTES_PER_TOKEN = N_CSA * (C * B_FP8 + C_I * B_FP4) / M_CSA + N_HCA * (C * B_FP8) / M_HCA
HIDDEN_STATE_BYTES = D_MODEL * B_BF16
DRAFT_BLOCK_MSG_BYTES = 64      # scaled down from Glimmer's 128 for block=7; illustrative
LOOP_CTRL_BYTES = 64
PROMPT_ID_BYTES = 4


def task_forward_flops(tokens: int, L: int) -> float:
    """Total FLOPs for one full 43-layer forward pass (ATTN+MOE+MHC+OUT),
    processing `tokens` new positions against context length L."""
    ctx = {"L": L, "tokens": tokens}
    return sum((k.flops_total(ctx) or 0) for k in KERNELS)


def prefill_kv_publish_bytes(L: int) -> float:
    return KV_BYTES_PER_TOKEN * L


def mtp_draft_step_flops(L: int) -> float:
    """APPROXIMATION: 1/N_LAYERS share of a full single-token forward pass.
    See module docstring, point 2."""
    return task_forward_flops(1, L) / N_LAYERS

# --- acceptance distribution, block=7 (not 16) ------------------------------

def sample_k(alpha: float, rng: random.Random) -> int:
    for j in range(DSPARK_BLOCK):
        if rng.random() >= alpha:
            return j
    return DSPARK_BLOCK


def expected_k(alpha: float) -> float:
    return (sum(j * alpha**j * (1 - alpha) for j in range(DSPARK_BLOCK))
           + DSPARK_BLOCK * alpha**DSPARK_BLOCK)

# --- trace event ------------------------------------------------------------

@dataclass
class TraceEvent:
    event_id: int
    request_id: int
    iteration: int
    src_task: str
    dst_task: str
    payload_bytes: int
    L_context: int
    k_accepted: str
    cum_output_tokens: int
    src_task_compute_flops: str
    note: str


def simulate_request(request_id: int, prompt_len: int, out_tokens: int,
                     alpha: float, rng: random.Random, start_event_id: int = 0):
    ev: list[TraceEvent] = []
    eid = start_event_id

    def emit(iteration, src, dst, nbytes, L, k, cum, flops, note):
        nonlocal eid
        ev.append(TraceEvent(eid, request_id, iteration, src, dst, int(round(nbytes)),
                             L, k, cum, f"{flops:.6g}" if flops else 0, note))
        eid += 1

    L = prompt_len
    emit(0, "EXT", "PREFILL", prompt_len * PROMPT_ID_BYTES, L, "", 0, 0,
         "external arrival: prompt token IDs")
    prefill_flops = task_forward_flops(L, L)
    emit(0, "PREFILL", "COMMIT", prefill_kv_publish_bytes(L), L, "", 0, prefill_flops,
         "one-time prefill KV publish (compressed CSA/HCA, grows linearly with L)")
    emit(0, "PREFILL", "MTP_DRAFT", HIDDEN_STATE_BYTES, L, "", 0, prefill_flops,
         "prefill->draft handoff: last-token hidden state")

    produced = 0
    iteration = 0
    while produced < out_tokens:
        iteration += 1
        k = sample_k(alpha, rng)
        draft_flops = mtp_draft_step_flops(L) * DSPARK_BLOCK
        emit(iteration, "MTP_DRAFT", "VERIFY", DRAFT_BLOCK_MSG_BYTES, L, "", produced,
             draft_flops, f"proposed {DSPARK_BLOCK}-token block via MTP head")
        verify_flops = task_forward_flops(DSPARK_BLOCK, L)
        emit(iteration, "VERIFY", "COMMIT", k * KV_BYTES_PER_TOKEN + HIDDEN_STATE_BYTES,
             L, k, produced, verify_flops, "verify result: accepted-token KV + context carry")
        produced += k
        L += k
        if produced < out_tokens:
            emit(iteration, "COMMIT", "MTP_DRAFT", LOOP_CTRL_BYTES, L, k, produced, 0,
                 "loop continue")
        else:
            emit(iteration, "COMMIT", "EXIT", 0, L, k, produced, 0,
                 "terminal condition (EOS / max length)")

    return ev


def write_trace(events, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(events[0]).keys()))
        w.writeheader()
        for e in events:
            w.writerow(asdict(e))


if __name__ == "__main__":
    # paper's Figure 9 checkpoint sequence, binary K (8192, not 8000)
    CONTEXT_SWEEP = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    ALPHA_SWEEP = [0.65, 0.75, 0.85]
    OUT_TOKENS = 100

    summary = []
    for L in CONTEXT_SWEEP:
        for alpha in ALPHA_SWEEP:
            rng = random.Random(7)
            events = simulate_request(0, L, OUT_TOKENS, alpha, rng)
            path = f"trace_deepseek_alpha{str(alpha).replace('0.', '')}_L{L}.csv"
            write_trace(events, path)
            iters = max(e.iteration for e in events)
            total_bytes = sum(e.payload_bytes for e in events)
            prefill_kv = next(e.payload_bytes for e in events if e.note.startswith("one-time"))
            total_verify_commit = sum(e.payload_bytes for e in events if e.src_task == "VERIFY")
            summary.append((path, alpha, L, iters, prefill_kv, total_verify_commit, total_bytes))

    with open("deepseek_sweep_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "alpha", "L", "iterations", "prefill_kv_bytes",
                   "total_verify_to_commit_bytes", "total_bytes"])
        w.writerows(summary)

    print(f"{'file':42} {'alpha':6} {'L':>9} {'iters':6} {'prefill_KV_B':>13} {'verify->commit_B':>17} {'total_B':>13}")
    for row in summary:
        print(f"{row[0]:42} {row[1]:<6} {row[2]:>9,} {row[3]:<6} {row[4]:>13,} {row[5]:>17,} {row[6]:>13,}")
    print(f"\n{len(summary)} files written, plus deepseek_sweep_summary.csv")
    print(f"KV_BYTES_PER_TOKEN = {KV_BYTES_PER_TOKEN:.1f} (vs Glimmer's 53,248 -- "
         f"{53248/KV_BYTES_PER_TOKEN:.1f}x smaller)")
