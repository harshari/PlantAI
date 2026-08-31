#!/usr/bin/env python3
"""Second workload, on the SAME Kernel/HardwareProfile schema (kernels.py):
Glimmer-30B's perception encoder ("dense causal transformer + perception
encoder" per the model card). This exists to prove the kernel framework
generalizes across workload SHAPE, not just workload identity -- the
speculative-decoding workload (specdecode_kernels.py) has a loop, branching
payload sizes, and causal/windowed attention; this one has none of that:

  EXT(image) -> patch_embed -> [vit_block]*32 -> final_norm -> proj_head -> EXIT

Single pass, no loop, fixed-size input, FULL bidirectional (non-causal,
non-windowed) self-attention across all patches. If a workload can be
described as "a sequence of kernels with known op types, FLOPs formulas, and
byte formulas," it plugs into this same machinery -- kernel_table /
latency_scenarios / the figure generator all come straight from kernels.py
unchanged. See explanation.txt, "Generalizing to more workloads," for how a
third shape (MoE routing -- a workload with REAL probabilistic branching,
unlike either workload here) would plug in too.

IMPORTANT: none of the dimensions below are published for Glimmer-30B's
perception encoder -- the model card gives text-decoder specs only. Every
constant here is an ASSUMED placeholder (CLIP-ViT-H/14-style, a common public
reference scale), clearly separate from the text-decoder constants in
specdecode_kernels.py, which ARE given. Treat this file as a template to
re-parametrize once real perception-encoder specs are available, not as a
sourced result.
"""

from __future__ import annotations

from kernels import (Kernel, BYTES_PER_ELEM, GENERIC_CHIPLET,
                     write_kernel_table, write_latency_table)

# ---------------------------------------------------------------------------
# ASSUMED constants (CLIP-ViT-H/14-style placeholder -- NOT published)
# ---------------------------------------------------------------------------
PATCH = 14
CHANNELS = 3
D_VIS = 1280
N_HEADS_VIS = 16
HEAD_DIM_VIS = D_VIS // N_HEADS_VIS         # 80
DEPTH_VIS = 32
D_FFN_VIS = 4 * D_VIS                       # 5120, standard 4x GELU-MLP ratio
D_MODEL_LM = 6656                           # main LM's d_model (given), fusion target
PRECISION = "bf16"
B = BYTES_PER_ELEM[PRECISION]

IMAGE_REGIMES = {
    "res_448": {"image_res": 448},   # 32x32 = 1,024 patches + 1 CLS = 1,025 tokens
    "res_896": {"image_res": 896},   # 64x64 = 4,096 patches + 1 CLS = 4,097 tokens
}


def num_tokens(ctx) -> int:
    side = ctx["image_res"] // PATCH
    return side * side + 1   # + CLS token


def _matmul_flops(tokens, d_in, d_out):
    return 2.0 * tokens * d_in * d_out


KERNELS: list[Kernel] = []


def _add(**kw):
    KERNELS.append(Kernel(**kw))


_add(id="V.patch_embed", task="V", layer_class="embedding", op_type="matmul",
     multiplicity_fn=lambda c: 1,
     flops_fn=lambda c: _matmul_flops(num_tokens(c) - 1, PATCH * PATCH * CHANNELS, D_VIS),
     weight_bytes_fn=lambda c: PATCH * PATCH * CHANNELS * D_VIS * B,
     io_bytes_fn=lambda c: num_tokens(c) * D_VIS * B,
     precision=PRECISION,
     note="Patchify + linear project, treated as one strided-conv-as-matmul. "
          "Fixed-size single pass -- no loop, unlike the spec-decode workload.")

for _ in [0]:  # single layer_class, kept as a loop for structural symmetry with specdecode
    lc = "vit_block"
    _add(id="V.vit_block.norm_attn", task="V", layer_class=lc, op_type="norm",
         multiplicity_fn=lambda c: DEPTH_VIS,
         flops_fn=lambda c: 5.0 * num_tokens(c) * D_VIS,
         weight_bytes_fn=lambda c: D_VIS * B,
         io_bytes_fn=lambda c: 2 * num_tokens(c) * D_VIS * B,
         precision=PRECISION, note="LayerNorm, pre-attention.")
    _add(id="V.vit_block.qkv_proj", task="V", layer_class=lc, op_type="matmul",
         multiplicity_fn=lambda c: DEPTH_VIS,
         flops_fn=lambda c: _matmul_flops(num_tokens(c), D_VIS, 3 * D_VIS),
         weight_bytes_fn=lambda c: D_VIS * 3 * D_VIS * B,
         io_bytes_fn=lambda c: num_tokens(c) * 4 * D_VIS * B,
         precision=PRECISION,
         note=f"Standard MHA (ASSUMED, no GQA) -- {N_HEADS_VIS} heads x {HEAD_DIM_VIS} dim.")
    _add(id="V.vit_block.attention", task="V", layer_class=lc, op_type="attention",
         multiplicity_fn=lambda c: DEPTH_VIS,
         flops_fn=lambda c: 4.0 * D_VIS * num_tokens(c) * num_tokens(c),   # NO causal halving
         weight_bytes_fn=lambda c: 0.0,
         io_bytes_fn=lambda c: num_tokens(c) * HEAD_DIM_VIS * 2 * B * N_HEADS_VIS,
         precision=PRECISION,
         note="FULL bidirectional self-attention -- every patch attends to every "
              "other patch, no causal mask, no window. Quadratic in num_tokens, "
              "which itself is quadratic in image resolution -- a steeper wall "
              "than the text model's windowed-by-default local layers.")
    _add(id="V.vit_block.out_proj", task="V", layer_class=lc, op_type="matmul",
         multiplicity_fn=lambda c: DEPTH_VIS,
         flops_fn=lambda c: _matmul_flops(num_tokens(c), D_VIS, D_VIS),
         weight_bytes_fn=lambda c: D_VIS * D_VIS * B,
         io_bytes_fn=lambda c: num_tokens(c) * 2 * D_VIS * B,
         precision=PRECISION, note="Attention output projection.")
    _add(id="V.vit_block.norm_ffn", task="V", layer_class=lc, op_type="norm",
         multiplicity_fn=lambda c: DEPTH_VIS,
         flops_fn=lambda c: 5.0 * num_tokens(c) * D_VIS,
         weight_bytes_fn=lambda c: D_VIS * B,
         io_bytes_fn=lambda c: 2 * num_tokens(c) * D_VIS * B,
         precision=PRECISION, note="LayerNorm, pre-FFN.")
    _add(id="V.vit_block.ffn_up", task="V", layer_class=lc, op_type="matmul",
         multiplicity_fn=lambda c: DEPTH_VIS,
         flops_fn=lambda c: _matmul_flops(num_tokens(c), D_VIS, D_FFN_VIS),
         weight_bytes_fn=lambda c: D_VIS * D_FFN_VIS * B,
         io_bytes_fn=lambda c: num_tokens(c) * (D_VIS + D_FFN_VIS) * B,
         precision=PRECISION, note="MLP up-projection, single matmul (GELU, not SwiGLU -- no gate).")
    _add(id="V.vit_block.gelu_act", task="V", layer_class=lc, op_type="elementwise",
         multiplicity_fn=lambda c: DEPTH_VIS,
         flops_fn=lambda c: 2.0 * num_tokens(c) * D_FFN_VIS,
         weight_bytes_fn=lambda c: 0.0,
         io_bytes_fn=lambda c: 2 * num_tokens(c) * D_FFN_VIS * B,
         precision=PRECISION, note="GELU activation, elementwise.")
    _add(id="V.vit_block.ffn_down", task="V", layer_class=lc, op_type="matmul",
         multiplicity_fn=lambda c: DEPTH_VIS,
         flops_fn=lambda c: _matmul_flops(num_tokens(c), D_FFN_VIS, D_VIS),
         weight_bytes_fn=lambda c: D_FFN_VIS * D_VIS * B,
         io_bytes_fn=lambda c: num_tokens(c) * (D_FFN_VIS + D_VIS) * B,
         precision=PRECISION, note="MLP down-projection.")

_add(id="V.final_norm", task="V", layer_class="output", op_type="norm",
     multiplicity_fn=lambda c: 1,
     flops_fn=lambda c: 5.0 * num_tokens(c) * D_VIS,
     weight_bytes_fn=lambda c: D_VIS * B,
     io_bytes_fn=lambda c: 2 * num_tokens(c) * D_VIS * B,
     precision=PRECISION, note="Final LayerNorm.")
_add(id="V.proj_head", task="V", layer_class="output", op_type="matmul",
     multiplicity_fn=lambda c: 1,
     flops_fn=lambda c: _matmul_flops(num_tokens(c), D_VIS, D_MODEL_LM),
     weight_bytes_fn=lambda c: D_VIS * D_MODEL_LM * B,
     io_bytes_fn=lambda c: num_tokens(c) * (D_VIS + D_MODEL_LM) * B,
     precision=PRECISION,
     note="Cross-modal projection: d_vis=1,280 -> d_model=6,656 so image "
          "tokens can join the main LM's context (feeds into task A's "
          "prefill in the text workload).")

# ---------------------------------------------------------------------------
# Figure: single pipeline strip (no loop, no branch -- one straight chain)
# ---------------------------------------------------------------------------

OP_COLORS = {
    "matmul": "#2a78d6", "attention": "#eb6834", "norm": "#9a9890",
    "elementwise": "#c3c2b7", "lookup": "#1baf7a", "reduce_argmax": "#e34948",
}


def draw_vision_figure(basename: str = "vision_dag_figure") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch, FancyArrowPatch

    ink, muted = "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(16, 4.3))
    ax.set_xlim(0, 16); ax.set_ylim(0, 2.1); ax.axis("off")
    ax.set_title("V -- perception encoder: EXT(image) -> patch_embed -> "
                 "[vit_block]x32 -> final_norm -> proj_head -> EXIT  "
                 "(single pass, no loop, no branch)",
                 fontsize=12.5, color=ink, loc="left", pad=6)

    groups = [("patch_embed", [k for k in KERNELS if k.id == "V.patch_embed"], 1),
              ("vit_block", [k for k in KERNELS if k.layer_class == "vit_block"], DEPTH_VIS),
              ("output", [k for k in KERNELS if k.layer_class == "output"], 1)]
    x = 0.5
    for label, ks, mult in groups:
        gx0 = x
        for k in ks:
            w = 1.55
            box = FancyBboxPatch((x, 0.55), w, 0.95, boxstyle="round,pad=0.045",
                                 fc=OP_COLORS.get(k.op_type, "#ccc"), ec="white", lw=1.0)
            ax.add_patch(box)
            short = k.id.split(".")[-1].replace("_", "\n")
            ax.text(x + w / 2, 1.02, short, ha="center", va="center",
                    fontsize=7.2, color="white", fontweight="bold")
            x += w + 0.14
        gw = x - gx0 - 0.14
        if mult > 1:
            ax.add_patch(plt.Rectangle((gx0 - 0.09, 0.47), gw + 0.18, 1.14,
                                       fill=False, ec=muted, lw=1.1, linestyle=(0, (3, 2))))
            ax.text(gx0 + gw / 2, 1.72, f"x {mult}  ({label})",
                    ha="center", fontsize=9, color=muted, fontweight="bold")
        x += 0.3
    ax.set_xlim(0, x + 0.3)

    fig.text(0.5, 0.135,
             "No loop, no probabilistic branch, no causal mask, no sliding window -- "
             "fixed-size single pass through every patch. Contrast with the spec-decode "
             "workload's A->[B->C->D]* loop (workload_dag.py / dag_figure.png).",
             ha="center", fontsize=8.7, color=muted)
    handles = [Patch(fc=c, label=op) for op, c in OP_COLORS.items() if op != "reduce_argmax"]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9.5, frameon=False,
              bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.savefig(f"{basename}.png", dpi=200, facecolor="white")
    fig.savefig(f"{basename}.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    write_kernel_table(KERNELS, IMAGE_REGIMES["res_448"],
                       "vision_kernel_table.csv", context_label="res_448")
    write_latency_table(KERNELS, IMAGE_REGIMES, "vision_latency_scenarios.csv")
    draw_vision_figure("vision_dag_figure")

    for label, ctx in IMAGE_REGIMES.items():
        total_flops = sum(k.flops_total(ctx) or 0 for k in KERNELS)
        total_ms = sum(k.expected_latency_ms(ctx) * k.multiplicity(ctx) for k in KERNELS)
        print(f"[{label}] tokens={num_tokens(ctx)}  total FLOPs={total_flops:.3e}  "
              f"expected latency={total_ms:.1f} ms")
    print(f"[kernels]  {len(KERNELS)} kernels, task V")
    print("[files]    vision_kernel_table.csv, vision_latency_scenarios.csv, "
          "vision_dag_figure.png/svg")
