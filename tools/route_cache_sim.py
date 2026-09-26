#!/usr/bin/env python3
"""Simulate cache-aware routing on a decode routing trace (GLM53F_ROUTE_TOP16).

    python tools/route_cache_sim.py trace16.txt [--slots 2684] [--warm 20]

Each trace line is one decode MoE layer: the layer, then the 16 best candidates in the
router's selection order as id:score (sigmoid, before normalisation); the first 8 are the
experts the model chose. The simulation replays the lines through an LRU expert cache of
--slots slots (2684 = the 38 GB int4 cache) and compares policies:

  exact      the model: every chosen expert is used; a missing one is read from the SSD.
  swap T     a chosen expert missing from the cache whose normalised weight is below T is
             replaced by the best-ranked candidate 9-16 that is resident (and not already
             chosen); only if there is none, or the weight is >= T, it is read.
  drop T     chosen experts whose normalised weight is below T are skipped (the rest are
             renormalised); the others behave as in exact.
  drop+swap  both, with their own thresholds.

It reports SSD reads per token, experts computed per token, and the share of the router's
weight that was replaced or dropped (the proxy for how far the output moves: 0 = exact).

Open loop: the trace is the exact model's routing. A policy that changes the output would
also change later hidden states and therefore later routing, which this cannot see, so
quality has to be measured in the engine; this only says how much SSD traffic is at stake.
"""
import argparse
from collections import OrderedDict

K = 8
N_EXP = 288


def load(path):
    steps = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 1 + K:
                continue
            layer = int(p[0])
            cand = [(int(a), float(b)) for a, b in (t.split(":") for t in p[1:])]
            steps.append((layer, cand))
    return steps


def split_tokens(steps):
    """A new token starts when the layer number does not increase."""
    tokens, cur, last = [], [], None
    for layer, cand in steps:
        if last is not None and layer <= last:
            tokens.append(cur)
            cur = []
        cur.append((layer, cand))
        last = layer
    if cur:
        tokens.append(cur)
    return tokens


def run(tokens, slots, warm, swap_t=None, drop_t=None):
    cache = OrderedDict()
    reads = computed = 0
    moved = 0.0            # router weight replaced or dropped, summed over layer-steps
    changed = nsteps = 0
    for ti, tok in enumerate(tokens):
        live = ti >= warm
        for layer, cand in tok:
            chosen = cand[:K]
            tot = sum(s for _, s in chosen) or 1.0
            w = [s / tot for _, s in chosen]
            chosen_ids = {e for e, _ in chosen}
            use = []
            step_moved = 0.0
            for (e, _), wj in zip(chosen, w):
                if drop_t is not None and wj < drop_t:
                    step_moved += wj
                    continue
                key = layer * N_EXP + e
                if key in cache or swap_t is None or wj >= swap_t:
                    use.append(key)
                    continue
                sub = next((layer * N_EXP + c for c, _ in cand[K:]
                            if c not in chosen_ids
                            and layer * N_EXP + c in cache
                            and layer * N_EXP + c not in use),
                           None)
                if sub is None:
                    use.append(key)
                else:
                    use.append(sub)
                    step_moved += wj
            for key in use:
                if key in cache:
                    cache.move_to_end(key)
                else:
                    if live:
                        reads += 1
                    cache[key] = 1
                    if len(cache) > slots:
                        cache.popitem(last=False)
            if live:
                computed += len(use)
                moved += step_moved
                changed += step_moved > 0
                nsteps += 1
    n = max(len(tokens) - warm, 1)
    return reads / n, computed / n, moved / max(nsteps, 1), 100.0 * changed / max(nsteps, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, default=2684)
    ap.add_argument("--warm", type=int, default=20)
    a = ap.parse_args()
    tokens = split_tokens(load(a.trace))
    print(f"{len(tokens)} tokens ({len(tokens) - a.warm} measured), {a.slots} cache slots")
    print(f"{'policy':18s} {'reads/tok':>9s} {'vs exact':>9s} {'experts/tok':>11s} "
          f"{'weight moved':>12s} {'layers changed':>14s}")
    base = None
    rows = [("exact", None, None)]
    rows += [(f"swap {t:.2f}", t, None) for t in (0.05, 0.10, 0.15, 0.20, 1.01)]
    rows += [(f"drop {t:.2f}", None, t) for t in (0.03, 0.05, 0.08)]
    rows += [(f"drop .05+swap .15", 0.15, 0.05)]
    for name, st, dt in rows:
        r, c, m, ch = run(tokens, a.slots, a.warm, st, dt)
        base = base if base is not None else r
        print(f"{name:18s} {r:9.1f} {100 * r / base:8.0f}% {c:11.1f} {100 * m:11.1f}% {ch:13.0f}%")
    print("\n'swap 1.01' swaps every missing expert that has a resident substitute (the ceiling).")
    print("weight moved: average share of a layer's routing weight replaced or dropped.")


if __name__ == "__main__":
    main()
