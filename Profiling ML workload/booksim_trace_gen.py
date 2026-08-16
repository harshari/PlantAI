#!/usr/bin/env python3
"""Bridges this folder's DAG/kernel model to BookSim: the missing
"[Task placement] -> [Trace generation]" box from the original pipeline
diagram. Answers three concrete questions:

  1. "What injection rate?" -- there isn't one, as a chosen input. BookSim's
     `injection_rate` config field is for SYNTHETIC traffic (uniform random,
     tornado, ...), where every node injects at the same fixed rate by
     construction. Our traffic is heterogeneous and DAG-driven: A injects
     rarely but in huge bursts (one KV publish per request), C injects
     every verify iteration, D injects tiny control messages. There is no
     single rate that describes that. Instead we use BookSim's TRACE-DRIVEN
     mode: every packet's injection cycle is computed from this file's
     discrete-event simulation of the DAG, not chosen up front. What we CAN
     do is measure an EFFECTIVE injection rate from the resulting trace
     (packets / cycles / node) after the fact -- useful for sanity-checking
     against a synthetic injection-rate sweep to see whether the workload's
     bursty real traffic sits below or above the network's saturation knee.
     print_effective_injection_rate() below does exactly that.

  2. "What anynet will we create?" -- BookSim ships regular topologies
     (mesh, torus, flattened butterfly, fat tree...) parametrically, but our
     chiplet placement is irregular: different task types get different
     chiplet counts, and some tasks deliberately SHARE chiplets (see the A/C
     weight-sharing placement below). BookSim's "anynet" topology reads an
     arbitrary router/link graph from a file instead of generating one from
     a formula -- exactly what an irregular placement needs. write_anynet()
     emits one such file from a placement dict.

  3. "How do we get the traces?" -- walk this folder's existing DAG
     simulation (workload_dag.simulate_request, the same one that produced
     trace_table.csv), and for every task instance compute a real finish
     time by summing that task's kernel latencies from specdecode_kernels.py
     at the request's actual context length L -- the same expected-latency
     numbers already in latency_scenarios.csv, not new invented ones. A
     single-server-per-chiplet queue is layered on top so that CONCURRENT
     requests contend for the same physical chiplet, which is the entire
     point of running this through BookSim at all -- a single isolated
     request never contends with itself. Injection times in ms are then
     converted to cycles via an assumed clock frequency and written out.

CAVEATS, stated once here rather than buried in comments below:
  - The single-server-per-chiplet queue is a first-order M/D/1-per-chiplet
    approximation (deterministic service time = the kernel-latency sum, one
    request served at a time, FIFO). It captures the ESSENTIAL contention
    effect but is not a full M/G/c or SimPy treatment (see Section 8 of
    explanation.txt for that caveat already on record) -- adequate for a
    first placement pass, not for a publication-grade latency number.
  - Only EDGES BETWEEN TWO PHYSICAL CHIPLETS become BookSim packets. The
    external arrival (EXT->A) and terminal condition (D->EXIT) are excluded
    from the trace file -- they're not NoC hops, they're where the DAG
    touches the outside world. EXT's timing still sets each request's
    baseline arrival time.
  - The exact BookSim trace-file column order and the exact anynet file
    keyword syntax vary by BookSim version/fork (vanilla booksim2 vs.
    academic forks with custom trace-traffic managers). What's below is a
    clearly-specified, simple format -- verify it against your specific
    checkout's trace-traffic-manager parser (or its `src/examples/anynet/`
    directory, if present) before feeding it in, and adapt the writer
    functions' formatting, not the underlying event computation.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass

from workload_dag import simulate_request, kv_bytes_per_token
from specdecode_kernels import KERNELS as SPEC_KERNELS

CLOCK_HZ = 1.5e9          # ASSUMED chiplet clock -- swap for your target's real frequency
FLIT_BYTES = 16            # ASSUMED flit size (128 bits) -- match your BookSim config's flit width

# ---------------------------------------------------------------------------
# Example placement: logical task ID -> list of physical chiplet IDs.
# A and C intentionally share physical chiplets 0/1 -- this encodes the
# weight-sharing placement insight from specdecode_kernels.py directly (same
# 52-layer main-model weights, so timesharing one chiplet class is valid).
# Swap this dict for any other placement candidate; nothing else changes.
# ---------------------------------------------------------------------------
PLACEMENT = {
    "A": [0, 1],
    "C": [0, 1],
    "B": [2, 3],
    "D": [4],
}
ALL_CHIPLETS = sorted({cid for ids in PLACEMENT.values() for cid in ids})


def task_compute_ms(task: str, ctx: dict) -> float:
    """Sum of expected kernel latency, across every kernel belonging to
    `task`, at the given context. Reuses specdecode_kernels.py's per-kernel
    roofline+contention model unchanged -- these are the same numbers in
    latency_scenarios.csv, not new estimates."""
    return sum(k.expected_latency_ms(ctx) * k.multiplicity(ctx)
              for k in SPEC_KERNELS if k.task == task)


@dataclass
class Packet:
    request_id: int
    src_chiplet: int
    dst_chiplet: int
    size_bytes: int
    injection_ms: float


def _group_events(events):
    """Collapse consecutive events sharing (iteration, src_task) -- they're
    the SAME task instance's outbound edges, fired together at one finish
    time (e.g. task A's KV-publish and hidden-state-handoff edges both fire
    right after ONE prefill pass, not two)."""
    groups = []
    i = 0
    while i < len(events):
        src_task, iteration = events[i].src_task, events[i].iteration
        group = []
        while i < len(events) and events[i].src_task == src_task and events[i].iteration == iteration:
            group.append(events[i]); i += 1
        groups.append(group)
    return groups


def simulate_placement(n_requests: int = 12, arrival_interval_ms: float = 2000.0,
                       alpha: float = 0.75, out_tokens: int = 100,
                       precision: str = "bf16", seed: int = 7):
    """Runs n_requests through the DAG (same simulate_request() that made
    trace_table.csv), staggered by arrival_interval_ms, and layers a
    single-server-per-chiplet queue on top to get REAL (contended) injection
    times. This is a genuine (if simplified) discrete-event simulation: a
    priority queue orders every request's next-ready task-instance by its
    earliest-possible start time, so concurrent requests actually interleave
    -- processing one request's full chain to completion before starting the
    next would artificially serialize everything (an earlier version of this
    script did exactly that; the giveaway was that widening arrival spacing
    from 8ms to 2000ms didn't change the result at all, since the real
    bottleneck was the buggy ordering, not contention)."""
    rng = random.Random(seed)
    chiplet_free_ms = {c: 0.0 for c in ALL_CHIPLETS}
    rr_counter = {task: 0 for task in PLACEMENT}
    packets: list[Packet] = []
    eid = 0

    request_groups: dict[int, list] = {}
    for rid in range(n_requests):
        prompt_len = 1500   # Azure coding-trace median, matches trace_table.csv's default
        events, _ = simulate_request(rid, prompt_len, out_tokens, alpha, precision, rng, eid)
        eid += len(events)
        request_groups[rid] = _group_events(events)

    # heap entries: (ready_time_ms, request_id, group_index)
    heap = [(rid * arrival_interval_ms, rid, 0) for rid in range(n_requests)]
    heapq.heapify(heap)

    while heap:
        ready_ms, rid, gidx = heapq.heappop(heap)
        group = request_groups[rid][gidx]
        src_task = group[0].src_task

        if src_task == "EXT":
            finish_ms = ready_ms
        else:
            ctx = {"L": group[0].L_context, "block_L": 16}
            chiplets = PLACEMENT[src_task]
            chosen = chiplets[rr_counter[src_task] % len(chiplets)]
            rr_counter[src_task] += 1
            start_ms = max(ready_ms, chiplet_free_ms[chosen])
            compute_ms = task_compute_ms(src_task, ctx)
            finish_ms = start_ms + compute_ms
            chiplet_free_ms[chosen] = finish_ms
            src_chiplet = chosen

            for ev in group:
                if ev.src_task == "EXT" or ev.dst_task == "EXIT":
                    continue  # not NoC traffic -- see module docstring
                dst_chiplets = PLACEMENT[ev.dst_task]
                dst_chiplet = dst_chiplets[rr_counter.get(ev.dst_task, 0) % len(dst_chiplets)]
                packets.append(Packet(rid, src_chiplet, dst_chiplet, ev.payload_bytes, finish_ms))

        if gidx + 1 < len(request_groups[rid]):
            heapq.heappush(heap, (finish_ms, rid, gidx + 1))

    return packets, chiplet_free_ms


# ---------------------------------------------------------------------------
# Question 1: effective injection rate (measured, not chosen)
# ---------------------------------------------------------------------------

def print_effective_injection_rate(packets: list[Packet]) -> None:
    if not packets:
        print("[injection-rate] no packets"); return
    t_end_ms = max(p.injection_ms for p in packets)
    cycles = t_end_ms * 1e-3 * CLOCK_HZ
    n_nodes = len(ALL_CHIPLETS)
    rate = len(packets) / cycles / n_nodes if cycles else 0.0
    print(f"[injection-rate] {len(packets)} packets over {cycles:,.0f} cycles, "
          f"{n_nodes} nodes -> effective injection rate = {rate:.3e} packets/cycle/node")
    print("                 (measured from the trace, not a chosen input -- "
          "compare this against a synthetic injection-rate sweep on the same "
          "topology to see how close this workload's bursty real traffic "
          "sits to the network's saturation knee)")
    per_chiplet = {}
    for p in packets:
        per_chiplet[p.src_chiplet] = per_chiplet.get(p.src_chiplet, 0) + 1
    for c in ALL_CHIPLETS:
        n = per_chiplet.get(c, 0)
        print(f"                 chiplet {c}: {n} packets injected "
              f"({n/cycles:.3e} packets/cycle)" if cycles else "")


# ---------------------------------------------------------------------------
# Question 2: anynet topology file
# ---------------------------------------------------------------------------

def write_anynet(path: str = "example.anynet") -> None:
    """One router per physical chiplet (router id == chiplet id), each
    star-connected through one shared hub router. Chiplets that are shared
    across task types (A and C both sit on chiplets 0/1 here) still get
    exactly ONE router each -- routing is a property of the physical
    chiplet, not of which logical task is currently using it. A star
    through a hub is a reasonable starting topology because every inter-task
    edge in this DAG is fairly low-volume (kernel_table.csv /
    latency_scenarios.csv: inter-node bytes never exceeded a few hundred KB
    even at L=100k) -- it doesn't need all-to-all bandwidth, just low hop
    count. VERIFY this file's exact keyword syntax against your BookSim
    build's anynet parser (src/networks/anynet.cpp) before use -- the
    structure (routers, their node attachments, their router-to-router
    links) is right; the literal keywords may need adjusting per version.
    """
    lines = [f"router {c} node {c}" for c in ALL_CHIPLETS]
    hub = max(ALL_CHIPLETS) + 1
    lines.append(f"router {hub} node")   # hub: no directly-attached compute nodes
    lines += [f"link router {c} router {hub}" for c in ALL_CHIPLETS]

    with open(path, "w") as f:
        f.write("# anynet topology -- one router per physical chiplet, star-connected\n")
        f.write("# through a shared hub router. router <id> node <chiplet id attached>\n")
        f.write("# link router <a> router <b>  -- bidirectional link between two routers.\n")
        f.write("# CHECK exact keyword syntax against your BookSim version before use.\n")
        f.write("\n".join(lines) + "\n")
    print(f"[anynet]   {path}  ({len(ALL_CHIPLETS)} chiplet routers + 1 hub)")


# ---------------------------------------------------------------------------
# Question 3: the trace file itself
# ---------------------------------------------------------------------------

def write_trace(packets: list[Packet], path: str = "example.trace") -> None:
    """One line per packet: src dst size_flits injection_cycle. Sorted by
    injection cycle, as most trace-driven traffic managers expect."""
    rows = []
    for p in packets:
        cycle = int(round(p.injection_ms * 1e-3 * CLOCK_HZ))
        size_flits = max(1, -(-p.size_bytes // FLIT_BYTES))   # ceil division
        rows.append((cycle, p.src_chiplet, p.dst_chiplet, size_flits, p.request_id))
    rows.sort()
    with open(path, "w") as f:
        f.write("# src dst size_flits injection_cycle request_id\n")
        f.write("# CHECK column order against your BookSim trace-traffic-manager parser.\n")
        for cycle, src, dst, size_flits, rid in rows:
            f.write(f"{src} {dst} {size_flits} {cycle} {rid}\n")
    print(f"[trace]    {path}  ({len(rows)} packets, "
          f"cycles {rows[0][0]}-{rows[-1][0]})" if rows else f"[trace] {path} (empty)")


if __name__ == "__main__":
    packets, chiplet_free_ms = simulate_placement()
    print(f"[sim]      {len(packets)} inter-chiplet packets from "
          f"{len(set(p.request_id for p in packets))} concurrent requests")
    for c in ALL_CHIPLETS:
        owner = [t for t, ids in PLACEMENT.items() if c in ids]
        print(f"           chiplet {c} ({','.join(owner)}): free at {chiplet_free_ms[c]:.2f} ms "
              f"-- {'CONTENDED (queued past its own compute)' if chiplet_free_ms[c] > 0 else ''}")
    print_effective_injection_rate(packets)
    write_anynet()
    write_trace(packets)
