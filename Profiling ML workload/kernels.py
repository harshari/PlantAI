#!/usr/bin/env python3
"""Generic kernel-level framework, shared by every workload in this folder.

This is the "step one" layer: before anything gets mapped onto hardware, each
task node in a workload DAG (A/B/C/D, or a ViT block, or an MoE expert) has to
be decomposed into its actual KERNELS -- individual ops with a concrete type
(matmul / attention / elementwise / lookup / memory), a FLOPs formula, a bytes
formula (weights streamed from local HBM + local activation/KV traffic), and a
precision. That decomposition is workload-specific and lives in
specdecode_kernels.py / vision_kernels.py; this file only defines the shared
vocabulary and the latency-scenario mechanism both of them use.

Two separate probabilistic layers exist in this project, and they answer
different questions -- don't conflate them:

  1. STRUCTURAL probability (workload_dag.py): which edge of the task DAG
     fires next, and with what payload. For the spec-decode workload this
     turned out to be trivial (every edge fires at p=1.0; the only random
     variable is the payload size k). For an MoE workload this is NOT
     trivial -- "which expert does this token route to" is a real branch,
     and is exactly the case the traffic-equation solver in workload_dag.py
     was built for (see explanation.txt, "Generalizing to MoE workloads").

  2. LATENCY probability (this file): given a fixed kernel with fixed FLOPs
     and bytes, how long does it actually take on hardware. Even a
     deterministic matmul doesn't have a deterministic wall-clock latency in
     a shared chiplet system -- HBM channel contention, NoC link contention
     with sibling chiplets, and batching effects make the SAME kernel
     invocation faster or slower run to run. That's the "30% of the time
     step 3 takes 10ms, 70% of the time it takes 30ms" you're asking for.
     We don't have profiled contention data yet (open item, same as
     everything else under "Open items" in explanation.txt), so this file
     computes a roofline latency (compute-bound vs bandwidth-bound, via
     max(flops/peak_flops, bytes/peak_bw)) and then applies an ASSUMED,
     clearly-labeled contention distribution on top of the bandwidth term.
     This is a placeholder prior, not a measurement -- the intended use is
     exactly the feedback arrow in the pipeline diagram: once a placement is
     fixed and run through BookSim/ns-3, replace CONTENTION_SCENARIOS with
     the simulator's observed contention distribution for that placement.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Hardware profile (placeholder -- see "Open items" in explanation.txt)
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    name: str
    peak_flops_per_sec: float        # dense compute throughput at target precision
    hbm_bw_bytes_per_sec: float      # per-chiplet local HBM/DRAM bandwidth
    noc_bw_bytes_per_sec: float      # inter-chiplet link bandwidth (for context; not
                                     # used directly here -- that's BookSim/ns-3's job)
    note: str = ""


GENERIC_CHIPLET = HardwareProfile(
    name="generic_chiplet_v1",
    peak_flops_per_sec=200e12,        # 200 TFLOP/s dense BF16 -- ILLUSTRATIVE
    hbm_bw_bytes_per_sec=1.6e12,      # 1.6 TB/s -- ILLUSTRATIVE (HBM3-class)
    noc_bw_bytes_per_sec=256e9,       # 256 GB/s inter-chiplet link -- ILLUSTRATIVE
    note="NOT a real datasheet. Open item: replace all three fields with the "
         "actual target chiplet's spec before trusting the compute-vs-bandwidth "
         "verdicts below.",
)

# Assumed contention distribution applied to the bandwidth-bound term only --
# compute (ALU) contention across chiplets is a much smaller effect than
# shared-memory-channel contention in practice, so only the bytes side gets a
# multiplier. ASSUMED, not profiled -- replace with BookSim/ns-3 output once a
# placement is fixed (see module docstring).
CONTENTION_SCENARIOS = [
    # (probability, bandwidth_multiplier, label)
    (0.70, 1.00, "uncontended: exclusive HBM channel / NoC link access"),
    (0.30, 2.60, "contended: sibling chiplet sharing the same HBM channel or NoC link"),
]

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

Ctx = dict  # runtime context: {"L": int, "tokens": int, "batch": int, ...}


@dataclass
class Kernel:
    id: str                # e.g. "C.global_attn.attention"
    task: str               # which task node this belongs to (A/B/C/D, or vision task id)
    layer_class: str        # repeated structural unit, e.g. "global_attn_layer"
    op_type: str             # "matmul" | "attention" | "elementwise" | "norm" | "lookup" | "reduce_argmax" | "memory"
    multiplicity_fn: Callable[[Ctx], int]      # how many times this fires per task invocation
    flops_fn: Callable[[Ctx], Optional[float]]  # FLOPs for ONE instance; None if unresolved
    weight_bytes_fn: Callable[[Ctx], float]     # weight bytes read from local HBM, ONE instance
    io_bytes_fn: Callable[[Ctx], float]         # local activation/KV bytes touched, ONE instance
    precision: str
    note: str = ""

    def multiplicity(self, ctx: Ctx) -> int:
        return self.multiplicity_fn(ctx)

    def flops_total(self, ctx: Ctx) -> Optional[float]:
        f = self.flops_fn(ctx)
        return None if f is None else f * self.multiplicity(ctx)

    def weight_bytes_total(self, ctx: Ctx) -> float:
        return self.weight_bytes_fn(ctx) * self.multiplicity(ctx)

    def io_bytes_total(self, ctx: Ctx) -> float:
        return self.io_bytes_fn(ctx) * self.multiplicity(ctx)

    def arithmetic_intensity(self, ctx: Ctx) -> Optional[float]:
        f = self.flops_total(ctx)
        if f is None:
            return None
        b = self.weight_bytes_total(ctx) + self.io_bytes_total(ctx)
        return None if b == 0 else f / b

    def roofline_ms(self, ctx: Ctx, hw: HardwareProfile = GENERIC_CHIPLET):
        """Returns (compute_ms, bw_ms) for ONE instance (not multiplicity-scaled --
        multiplicity mostly represents independent layer instances that pipeline,
        so we roofline per-instance and let the caller decide how to aggregate)."""
        f = self.flops_fn(ctx)
        wb = self.weight_bytes_fn(ctx)
        ib = self.io_bytes_fn(ctx)
        compute_ms = (f / hw.peak_flops_per_sec * 1000) if f is not None else None
        bw_ms = (wb + ib) / hw.hbm_bw_bytes_per_sec * 1000
        return compute_ms, bw_ms

    def latency_scenarios(self, ctx: Ctx, hw: HardwareProfile = GENERIC_CHIPLET):
        """List of (probability, latency_ms, bound_type, label) for ONE instance.
        bound_type is 'compute' if the compute term dominates even under the
        worst-case contended bandwidth scenario, else 'bandwidth'."""
        compute_ms, bw_ms = self.roofline_ms(ctx, hw)
        out = []
        for prob, mult, label in CONTENTION_SCENARIOS:
            bw_scenario_ms = bw_ms * mult
            if compute_ms is None:
                lat, bound = bw_scenario_ms, "bandwidth (compute FLOPs unresolved)"
            elif compute_ms >= bw_scenario_ms:
                lat, bound = compute_ms, "compute"
            else:
                lat, bound = bw_scenario_ms, "bandwidth"
            out.append((prob, lat, bound, label))
        return out

    def expected_latency_ms(self, ctx: Ctx, hw: HardwareProfile = GENERIC_CHIPLET) -> float:
        return sum(p * lat for p, lat, _, _ in self.latency_scenarios(ctx, hw))


BYTES_PER_ELEM = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5, "fp32": 4.0}

# ---------------------------------------------------------------------------
# Generic report writers -- work for any list[Kernel], any workload
# ---------------------------------------------------------------------------

def write_kernel_table(kernels: list, ctx: Ctx, path: str,
                       context_label: str = "") -> None:
    """The 'what is the kernel, how much compute, how much data, what
    precision, matmul-or-lookup' table."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel_id", "task", "layer_class", "op_type", "precision",
                    "multiplicity", "flops_per_instance", "total_flops",
                    "weight_bytes_per_instance", "io_bytes_per_instance",
                    "total_bytes", "arithmetic_intensity_flops_per_byte",
                    "context", "note"])
        for k in kernels:
            mult = k.multiplicity(ctx)
            fpi = k.flops_fn(ctx)
            tf = k.flops_total(ctx)
            wb = k.weight_bytes_fn(ctx)
            ib = k.io_bytes_fn(ctx)
            tb = k.weight_bytes_total(ctx) + k.io_bytes_total(ctx)
            ai = k.arithmetic_intensity(ctx)
            w.writerow([
                k.id, k.task, k.layer_class, k.op_type, k.precision, mult,
                "UNRESOLVED" if fpi is None else f"{fpi:.6g}",
                "UNRESOLVED" if tf is None else f"{tf:.6g}",
                f"{wb:.6g}", f"{ib:.6g}", f"{tb:.6g}",
                "n/a" if ai is None else f"{ai:.4g}",
                context_label, k.note,
            ])


def write_latency_table(kernels: list, contexts: dict, path: str,
                        hw: HardwareProfile = GENERIC_CHIPLET) -> None:
    """The '30% it takes 10ms, 70% it takes 30ms' table, per kernel per
    named context regime. contexts: {label: ctx_dict}."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel_id", "task", "context_regime", "scenario",
                    "probability", "latency_ms_per_instance", "bound_type",
                    "expected_latency_ms_per_instance",
                    "expected_total_latency_ms_all_instances"])
        for label, ctx in contexts.items():
            for k in kernels:
                exp = k.expected_latency_ms(ctx, hw)
                mult = k.multiplicity(ctx)
                for prob, lat, bound, scen_label in k.latency_scenarios(ctx, hw):
                    w.writerow([k.id, k.task, label, scen_label, f"{prob:.2f}",
                               f"{lat:.5g}", bound, f"{exp:.5g}", f"{exp*mult:.5g}"])
