#!/usr/bin/env python3
"""Probabilistic workload DAG for block-verification speculative decoding.

Encodes the task graph, payload formulas, and stochastic unrolling rule from
the working notes ("Probabilistic Workload DAG -> Hardware Mapping"), and
generates the artifacts the hardware-mapping pipeline consumes:

  1. a LOGICAL trace table (task-ID based, no physical chiplet IDs -- the
     placement layer remaps logical->physical per floorplan candidate, so the
     trace is generated once and reused across the whole floorplan search);
  2. the expected-DAG figure (dag_figure.png / .svg);
  3. traffic-equation solves (visit counts V_i), including the Section 2.1
     toy example as a self-test;
  4. an ensemble mode that samples many stochastic unrollings to produce
     iteration-count / burst-size distributions instead of a single trace.

Reference model: Muse Glimmer-30B (dense causal transformer, d_model=6656,
52 layers = 39 local (window 2048) + 13 global, GQA 32Q/2KV main, 32Q/8KV
draft, 5 draft layers, speculative block size 16, ~27.8B LM params).
Swap the constants below for your target model where marked.

Usage:
  python3 workload_dag.py                          # trace + figure, defaults
  python3 workload_dag.py --alpha 0.85 --prompt 16384 --out-tokens 256
  python3 workload_dag.py --ensemble 1000          # distribution summary
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Model constants (Muse Glimmer-30B) -- swap for your target model
# ---------------------------------------------------------------------------
D_MODEL = 6656
N_LAYERS = 52
N_LOCAL = 39          # sliding-window layers, window capped
N_GLOBAL = 13         # full-context attention layers
WINDOW = 2048
HEAD_DIM = 128
KV_HEADS_MAIN = 2
KV_HEADS_DRAFT = 8
DRAFT_LAYERS = 5
BLOCK = 16            # speculative block size (tokens proposed per draft pass)
PARAMS_LM = 27.8e9
BYTES_PER_ELEM = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5}

# ---------------------------------------------------------------------------
# Per-kernel compute / communication formulas (Table 2 of the notes)
# ---------------------------------------------------------------------------

def kv_bytes_per_token(precision: str = "bf16", model: str = "main") -> float:
    """KV_bytes/token = 2 * n_kv_heads * head_dim * n_layers * B_prec.
    main/bf16 -> 53,248 B; draft/bf16 -> 20,480 B."""
    if model == "main":
        heads, layers = KV_HEADS_MAIN, N_LAYERS
    elif model == "draft":
        heads, layers = KV_HEADS_DRAFT, DRAFT_LAYERS
    else:
        raise ValueError(model)
    return 2 * heads * HEAD_DIM * layers * BYTES_PER_ELEM[precision]


def kv_bytes_per_token_per_layer(precision: str = "bf16") -> float:
    """Main-model KV bytes per token per layer (1,024 B at BF16)."""
    return 2 * KV_HEADS_MAIN * HEAD_DIM * BYTES_PER_ELEM[precision]


def prefill_flops(L: int) -> float:
    """Task A. Linear term = 52-layer forward (~2*N_params/token);
    quadratic term = the 13 global-attention layers."""
    return 5.0e10 * L + 346112 * L * L


def verify_flops(L: int) -> float:
    """Task C: 16-token block through 52 layers + global attention over full L."""
    return 16 * 5.0e10 + 5537792 * L


def draft_flops(L: int):
    """Task B. UNRESOLVED: draft FFN intermediate width is not published.
    QKV + out-proj are known; FFN term missing -- see 'Open items' in the
    notes (inspect GGUF tensor shapes to fill this in). Returns None."""
    return None


def prefill_kv_writeout_bytes(L: int, precision: str = "bf16") -> float:
    """Task A communication: (39*min(L,2048) + 13*L) * per-layer KV bytes.
    Local layers cap at the sliding window; global layers grow with L."""
    per_layer = kv_bytes_per_token_per_layer(precision)
    return (N_LOCAL * min(L, WINDOW) + N_GLOBAL * L) * per_layer


HIDDEN_STATE_BYTES = D_MODEL * BYTES_PER_ELEM["bf16"]   # 13,312 B: one token's
                                                        # hidden state (context carry)
DRAFT_BLOCK_MSG_BYTES = 128    # B->C: proposed token IDs + probabilities
LOOP_CTRL_BYTES = 64           # D->B loop-continue control message (assumption)
PROMPT_ID_BYTES = 4            # external arrival: ~4 B per prompt token ID

# ---------------------------------------------------------------------------
# DAG encoding (Table 1 of the notes)
# ---------------------------------------------------------------------------
# The loop C->D->B is NOT a probabilistic branch: every edge below fires with
# p=1.0 on each iteration. What is stochastic is k (tokens accepted per
# verify/commit), drawn per iteration; the loop exits on an external terminal
# condition (EOS / max length), not on an edge probability.

NODES = {
    "A": "Prefill: 52-layer forward over prompt (length L)",
    "B": "Draft: 5-layer forward, proposes a 16-token block",
    "C": "Verify: 52-layer forward, proposed block vs. full context",
    "D": "KV-commit: accept k of 16 tokens, k ~ TruncGeom(alpha)",
}

EDGES = [
    # (src, dst, probability, payload description)
    ("EXT", "A", 1.0, "external arrival: prompt token IDs (~4 B/token)"),
    ("A",   "B", 1.0, "handoff: last-token hidden state (13,312 B @ BF16)"),
    ("A",   "D", 1.0, "one-time prefill KV publish: (39*min(L,2048)+13*L)*1024 B @ BF16"),
    ("B",   "C", 1.0, "proposed block: token IDs + probs (~128 B)"),
    ("C",   "D", 1.0, "accepted-token KV: k*53,248 B + 13,312 B context carry @ BF16"),
    ("D",   "B", 1.0, "loop continue (until EOS / max length): control (~64 B)"),
    ("D",   "EXIT", 1.0, "terminal condition reached (EOS / max length)"),
]

# ---------------------------------------------------------------------------
# Acceptance distribution (Section 2.3): truncated geometric
# ---------------------------------------------------------------------------

def acceptance_pmf(alpha: float) -> list:
    """Full distribution: P(k=j) = alpha^j*(1-alpha), j=0..15; P(k=16)=alpha^16."""
    return [alpha**j * (1 - alpha) for j in range(BLOCK)] + [alpha**BLOCK]


def sample_k(alpha: float, rng: random.Random) -> int:
    """Draw from acceptance_pmf by sequential accept/reject."""
    for j in range(BLOCK):
        if rng.random() >= alpha:
            return j
    return BLOCK


def expected_k(alpha: float) -> float:
    return sum(j * alpha**j * (1 - alpha) for j in range(BLOCK)) + BLOCK * alpha**BLOCK

# ---------------------------------------------------------------------------
# Traffic equations (Section 2): V_i = ext_i + sum_j V_j * p(j->i)
# Fixed-point iteration; feedback/self loops are fine (Jackson network).
# ---------------------------------------------------------------------------

def solve_traffic(routing: dict, external: dict, tol: float = 1e-12,
                  max_iter: int = 100000) -> dict:
    """routing: {src: [(dst, p), ...]};  external: {node: external visits}."""
    nodes = set(external) | set(routing)
    for dsts in routing.values():
        nodes |= {d for d, _ in dsts}
    nodes.discard("EXIT")
    V = {n: float(external.get(n, 0.0)) for n in nodes}
    for _ in range(max_iter):
        V_new = {n: float(external.get(n, 0.0)) for n in nodes}
        for src, dsts in routing.items():
            for dst, p in dsts:
                if dst in V_new:
                    V_new[dst] += V.get(src, 0.0) * p
        delta = max(abs(V_new[n] - V[n]) for n in nodes)
        V = V_new
        if delta < tol:
            break
    return V


def toy_traffic_selftest() -> dict:
    """Section 2.1 toy example: A->B->C->D with C->D=0.3, C->C=0.2, C->exit=0.5.
    Expected: V_C = 1.25, V_D = 0.375 (the 20% retry inflates C's load 25%)."""
    V = solve_traffic(
        routing={"A": [("B", 1.0)], "B": [("C", 1.0)],
                 "C": [("C", 0.2), ("D", 0.3), ("EXIT", 0.5)]},
        external={"A": 1.0},
    )
    assert abs(V["C"] - 1.25) < 1e-9, V
    assert abs(V["D"] - 0.375) < 1e-9, V
    return V


def loop_visit_counts(alpha: float, out_tokens: int) -> dict:
    """Expected visit counts for the real speculative loop. Not a branch
    solve: iterations are set by the terminal condition, E[iters] ~
    out_tokens / E[k] (per notes: ~34 iters / 100 tokens at alpha=0.75)."""
    it = out_tokens / expected_k(alpha)
    return {"A": 1.0, "B": it, "C": it, "D": it}

# ---------------------------------------------------------------------------
# Stochastic unrolling -> logical trace (Section 2.2)
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    event_id: int
    request_id: int
    iteration: int          # 0 = prefill phase
    src_task: str           # LOGICAL task IDs -- placement maps these to
    dst_task: str           # physical chiplet IDs per floorplan candidate
    payload_bytes: int
    L_context: int          # context position when the edge fires
    k_accepted: str         # '' where not applicable
    cum_output_tokens: int
    note: str


def simulate_request(request_id: int, prompt_len: int, out_tokens: int,
                     alpha: float, precision: str, rng: random.Random,
                     start_event_id: int = 0):
    """One sampled unrolling of the template graph. Returns (events, stats)."""
    ev: list[TraceEvent] = []
    eid = start_event_id
    kv_tok = kv_bytes_per_token(precision, "main")

    def emit(iteration, src, dst, nbytes, L, k, cum, note):
        nonlocal eid
        ev.append(TraceEvent(eid, request_id, iteration, src, dst,
                             int(round(nbytes)), L, k, cum, note))
        eid += 1

    L = prompt_len
    emit(0, "EXT", "A", prompt_len * PROMPT_ID_BYTES, L, "", 0,
         "external arrival: prompt token IDs")
    emit(0, "A", "D", prefill_kv_writeout_bytes(L, precision), L, "", 0,
         "one-time prefill KV publish (grows linearly with L)")
    emit(0, "A", "B", HIDDEN_STATE_BYTES, L, "", 0,
         "prefill->draft handoff: last-token hidden state")

    produced = 0
    iteration = 0
    ks: list[int] = []
    while produced < out_tokens:
        iteration += 1
        k = sample_k(alpha, rng)
        ks.append(k)
        emit(iteration, "B", "C", DRAFT_BLOCK_MSG_BYTES, L, "", produced,
             "proposed 16-token block: IDs + probs")
        # C->D always fires: k accepted tokens' KV plus the context carry,
        # matching Table 2's  E[k]*53248 + 13312  expectation.
        emit(iteration, "C", "D", k * kv_tok + HIDDEN_STATE_BYTES, L, k, produced,
             "verify result: accepted-token KV + context carry")
        produced += k
        L += k
        if produced < out_tokens:
            emit(iteration, "D", "B", LOOP_CTRL_BYTES, L, k, produced,
                 "loop continue")
        else:
            emit(iteration, "D", "EXIT", 0, L, k, produced,
                 "terminal condition (EOS / max length)")

    stats = {
        "iterations": iteration,
        "mean_k": sum(ks) / len(ks),
        "final_L": L,
        "total_CD_bytes": sum(e.payload_bytes for e in ev
                              if e.src_task == "C" and e.dst_task == "D"),
        "prefill_kv_bytes": int(prefill_kv_writeout_bytes(prompt_len, precision)),
        "prefill_flops": prefill_flops(prompt_len),
        "last_verify_flops": verify_flops(L),
    }
    return ev, stats

# ---------------------------------------------------------------------------
# Prompt-length distribution (Section 6, agent-heavy anchor)
# ---------------------------------------------------------------------------
PROMPT_BUCKETS = [   # (share, lo, hi) -- log-uniform inside each bucket
    (0.40,    512,   2048),
    (0.20,   2048,   4096),
    (0.22,   4096,  16384),
    (0.10,  16384,  32768),
    (0.08,  32768, 100000),
]


def sample_prompt_length(rng: random.Random) -> int:
    r, acc = rng.random(), 0.0
    for share, lo, hi in PROMPT_BUCKETS:
        acc += share
        if r <= acc:
            return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
    return PROMPT_BUCKETS[-1][2]

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_trace(events: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(events[0]).keys()))
        w.writeheader()
        for e in events:
            w.writerow(asdict(e))


def draw_figure(basename: str = "dag_figure") -> None:
    """Expected-DAG figure, drawn from the EDGES table above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ink, muted, accent, loopc = "#0b0b0b", "#52514e", "#2a78d6", "#b0432f"
    fig, ax = plt.subplots(figsize=(15.5, 7.0))
    ax.set_xlim(0, 15.5); ax.set_ylim(0, 7.0); ax.axis("off")

    y0 = 3.3                       # main flow centerline
    HW, HH = 1.02, 0.62            # box half-width / half-height
    pos = {"A": (3.6, y0), "B": (6.7, y0), "C": (9.8, y0), "D": (12.9, y0)}
    sub = {
        "A": "prefill, 52-layer fwd\nFLOPs: 5.0e10*L + 346112*L^2",
        "B": "draft, 5-layer fwd\n16-token block (FLOPs: TBD FFN width)",
        "C": "verify, 52-layer fwd\nFLOPs: 16*5.0e10 + 5537792*L",
        "D": "KV commit, accept k of 16\nk ~ TruncGeom(alpha), FLOPs ~ 0",
    }
    for n, (x, y) in pos.items():
        box = FancyBboxPatch((x - HW, y - HH), 2 * HW, 2 * HH,
                             boxstyle="round,pad=0.10", fc="#f2f5fa",
                             ec=accent, lw=1.6)
        ax.add_patch(box)
        ax.text(x, y + 0.30, n, ha="center", va="center",
                fontsize=15, fontweight="bold", color=ink)
        ax.text(x, y - 0.22, sub[n], ha="center", va="center",
                fontsize=8.4, color=muted)

    ax.text(0.85, y0, "arrival", ha="center", va="center",
            fontsize=11, color=muted, style="italic")
    ax.text(14.75, y0, "exit\n(EOS /\nmax len)", ha="center", va="center",
            fontsize=10.5, color=muted, style="italic")

    def arrow(p1, p2, color=ink, rad=0.0, ls="-", shrink=6):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=16, color=color, lw=1.5,
            linestyle=ls, connectionstyle=f"arc3,rad={rad}",
            shrinkA=shrink, shrinkB=shrink))

    def elabel(x, y, text, color=ink):
        ax.text(x, y, text, ha="center", va="bottom", fontsize=8.6, color=color)

    # straight edges along the centerline; labels sit ABOVE the box tops
    ylab = 4.30                    # box tops end at ~4.02
    arrow((1.30, y0), (pos["A"][0] - HW - 0.12, y0))
    elabel(2.05, ylab, "p=1.0\nprompt IDs\n~4 B/tok")
    arrow((pos["A"][0] + HW + 0.12, y0), (pos["B"][0] - HW - 0.12, y0))
    elabel(5.15, ylab, "p=1.0\nhidden-state handoff\n13,312 B")
    arrow((pos["B"][0] + HW + 0.12, y0), (pos["C"][0] - HW - 0.12, y0))
    elabel(8.25, ylab, "p=1.0\nproposed block\n~128 B")
    arrow((pos["C"][0] + HW + 0.12, y0), (pos["D"][0] - HW - 0.12, y0))
    elabel(11.35, ylab, "p=1.0\nk*53,248 B\n+ 13,312 B carry")
    arrow((pos["D"][0] + HW + 0.12, y0), (14.30, y0))
    elabel(14.15, ylab, "terminal\ncondition")

    # one-time prefill KV publish: A -> D arcing over the top
    arrow((pos["A"][0], y0 + HH + 0.20), (pos["D"][0], y0 + HH + 0.20),
          color=accent, rad=-0.42, ls="--", shrink=2)
    ax.text(8.25, 6.35, "one-time prefill KV publish:  (39*min(L,2048) + 13*L) * 1,024 B",
            ha="center", va="center", fontsize=9.2, color=accent)

    # loop back: D -> B arcing under the bottom
    arrow((pos["D"][0], y0 - HH - 0.20), (pos["B"][0], y0 - HH - 0.20),
          color=loopc, rad=-0.32, ls="--", shrink=2)
    ax.text(9.8, 1.00, "loop until EOS / max length  (p=1.0 per iteration; ~64 B control)\n"
                       "the stochastic quantity is k, not the edge choice",
            ha="center", va="center", fontsize=9.2, color=loopc)

    ax.text(0.15, 0.42, "Template graph + stochastic unrolling: each request samples its own "
                        "k sequence  (P(k=j) = alpha^j (1-alpha), j=0..15;  P(16) = alpha^16).",
            fontsize=8.4, color=muted, ha="left")
    ax.text(0.15, 0.16, "Payloads @ BF16, main-model KV = 53,248 B/token. The trace stays "
                        "logical (task IDs); the placement layer maps tasks to chiplets per "
                        "floorplan candidate.",
            fontsize=8.4, color=muted, ha="left")
    ax.set_title("Speculative-decoding workload DAG  --  A (prefill) -> [ B (draft) -> C (verify) -> D (commit) ]*",
                 fontsize=13, color=ink, pad=10)
    fig.tight_layout()
    fig.savefig(f"{basename}.png", dpi=200, facecolor="white")
    fig.savefig(f"{basename}.svg", facecolor="white")
    plt.close(fig)

def write_acceptance_pmf_csv(path: str = "acceptance_pmf.csv",
                             alphas=(0.65, 0.75, 0.85)) -> None:
    """The acceptance distribution as data: P(k=j) per alpha bound."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k"] + [f"P(k)_alpha_{a}" for a in alphas])
        pmfs = [acceptance_pmf(a) for a in alphas]
        for j in range(BLOCK + 1):
            w.writerow([j] + [f"{p[j]:.6g}" for p in pmfs])
        w.writerow(["E[k]"] + [f"{expected_k(a):.4g}" for a in alphas])


def draw_distributions(basename: str = "distributions", out_tokens: int = 100,
                       ensemble_n: int = 2000, seed: int = 7) -> None:
    """The three probability distributions that drive the stochastic unrolling:
    (1) acceptance PMF P(k) at the assumed alpha bounds (Section 2.3);
    (2) prompt-length distribution (Section 6, agent-heavy anchor);
    (3) resulting iteration-count distribution per request (sampled)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    alphas = [0.65, 0.75, 0.85]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3

    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.42, wspace=0.25)

    # -- top row: acceptance PMF, one small multiple per alpha (shared y) ----
    for i, (a, c) in enumerate(zip(alphas, colors)):
        ax = fig.add_subplot(gs[0, i])
        pmf = acceptance_pmf(a)
        ax.bar(range(BLOCK + 1), pmf, color=c, width=0.72)
        ax.set_ylim(0, 0.37)
        ax.set_title(f"alpha = {a}   (E[k] = {expected_k(a):.2f})",
                     fontsize=11, color=ink)
        ax.set_xlabel("k accepted per block", fontsize=9, color=muted)
        if i == 0:
            ax.set_ylabel("P(k)", fontsize=9, color=muted)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.tick_params(labelsize=8.5, colors=muted)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(f"P(k=16) = {pmf[BLOCK]:.1%}\n(whole block accepted)",
                    xy=(16, pmf[BLOCK]), xytext=(9.2, 0.30), fontsize=8.2,
                    color=muted, arrowprops=dict(arrowstyle="->", color=muted, lw=0.8))
    fig.text(0.5, 0.965, "Acceptance distribution  P(k=j) = alpha^j (1-alpha),  "
             "j=0..15;   P(k=16) = alpha^16   (assumed bounds - profiling pending)",
             ha="center", fontsize=12, color=ink)

    # -- bottom left: prompt-length distribution (Section 6) -----------------
    axp = fig.add_subplot(gs[1, 0])
    labels = ["512-2K", "2K-4K", "4K-16K", "16K-32K", "32K-100K"]
    shares = [s * 100 for s, _, _ in PROMPT_BUCKETS]
    axp.bar(range(len(shares)), shares, color="#2a78d6", width=0.66)
    for i, s in enumerate(shares):
        axp.text(i, s + 1.1, f"{s:.0f}%", ha="center", fontsize=9, color=ink)
    axp.set_xticks(range(len(labels)))
    axp.set_xticklabels(labels, fontsize=8.2, rotation=20)
    axp.set_ylim(0, 48)
    axp.set_ylabel("share of requests (%)", fontsize=9, color=muted)
    axp.set_title("Prompt-length distribution\n(Section 6, agent-heavy anchor; "
                  "log-uniform in bucket)", fontsize=10, color=ink)
    axp.tick_params(labelsize=8.5, colors=muted)
    axp.spines[["top", "right"]].set_visible(False)

    # -- bottom middle+right: iteration-count distribution (sampled) ---------
    axh = fig.add_subplot(gs[1, 1:])
    rng = random.Random(seed)
    for a, c in zip(alphas, colors):
        counts = {}
        for _ in range(ensemble_n):
            produced, iters = 0, 0
            while produced < out_tokens:
                iters += 1
                produced += sample_k(a, rng)
            counts[iters] = counts.get(iters, 0) + 1
        xs = sorted(counts)
        ys = [counts[x] / ensemble_n for x in xs]
        axh.plot(xs, ys, drawstyle="steps-mid", color=c, lw=1.8,
                 label=f"alpha={a}  (mean {sum(x*counts[x] for x in xs)/ensemble_n:.1f})")
    axh.set_xlabel(f"loop iterations to produce {out_tokens} output tokens",
                   fontsize=9, color=muted)
    axh.set_ylabel("probability", fontsize=9, color=muted)
    axh.set_title(f"Resulting iteration-count distribution per request "
                  f"({ensemble_n} sampled unrollings)", fontsize=10, color=ink)
    axh.legend(fontsize=9, frameon=False)
    axh.tick_params(labelsize=8.5, colors=muted)
    axh.spines[["top", "right"]].set_visible(False)

    fig.savefig(f"{basename}.png", dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(f"{basename}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="per-token acceptance probability (bounds: 0.65/0.75/0.85)")
    ap.add_argument("--prompt", type=int, default=1500,
                    help="prompt length L0 (default: Azure coding-trace median)")
    ap.add_argument("--out-tokens", type=int, default=100)
    ap.add_argument("--precision", choices=BYTES_PER_ELEM, default="bf16",
                    help="KV/activation precision (weight quant != KV quant!)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trace", default="trace_table.csv")
    ap.add_argument("--figure", default="dag_figure", help="basename; '' to skip")
    ap.add_argument("--ensemble", type=int, default=0,
                    help="N>0: sample N requests (prompt lengths from the "
                         "Section-6 distribution) and print summary stats")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    V = toy_traffic_selftest()
    print(f"[selftest] Section 2.1 toy traffic equations OK: "
          f"V_A={V['A']:.3f} V_B={V['B']:.3f} V_C={V['C']:.3f} V_D={V['D']:.3f}")

    Vloop = loop_visit_counts(args.alpha, args.out_tokens)
    print(f"[loop]     alpha={args.alpha}: E[k]={expected_k(args.alpha):.2f}, "
          f"expected visits per request for {args.out_tokens} output tokens: "
          f"A={Vloop['A']:.0f}, B=C=D={Vloop['B']:.1f}")

    if args.ensemble > 0:
        iters, cd_bytes, prompts = [], [], []
        for rid in range(args.ensemble):
            L0 = sample_prompt_length(rng)
            _, s = simulate_request(rid, L0, args.out_tokens, args.alpha,
                                    args.precision, rng)
            iters.append(s["iterations"]); cd_bytes.append(s["total_CD_bytes"])
            prompts.append(L0)
        iters.sort(); prompts.sort()
        n = args.ensemble
        print(f"[ensemble] N={n}, alpha={args.alpha}, {args.out_tokens} output tokens/request")
        print(f"           iterations: mean={sum(iters)/n:.1f}  "
              f"p50={iters[n//2]}  p95={iters[int(n*0.95)]}  max={iters[-1]}")
        print(f"           prompt len: p50={prompts[n//2]}  p95={prompts[int(n*0.95)]}")
        print(f"           total C->D bytes/request: mean={sum(cd_bytes)/n:,.0f} "
              f"(invariant: ~= out_tokens*KV/token + iters*13,312 carry)")
        return

    events, stats = simulate_request(0, args.prompt, args.out_tokens,
                                     args.alpha, args.precision, rng)
    write_trace(events, args.trace)
    print(f"[trace]    {len(events)} events -> {args.trace}")
    print(f"           iterations={stats['iterations']}, mean k={stats['mean_k']:.2f}, "
          f"final L={stats['final_L']}")
    print(f"           prefill: {stats['prefill_flops']:.3e} FLOPs, "
          f"KV publish {stats['prefill_kv_bytes']:,} B")
    print(f"           total C->D bytes={stats['total_CD_bytes']:,} "
          f"(invariant: independent of alpha up to the 13,312 B/iter carry)")
    if args.figure:
        draw_figure(args.figure)
        print(f"[figure]   {args.figure}.png / {args.figure}.svg")
        draw_distributions(out_tokens=args.out_tokens, seed=args.seed)
        write_acceptance_pmf_csv()
        print("[dists]    distributions.png / distributions.svg / acceptance_pmf.csv")


if __name__ == "__main__":
    main()
