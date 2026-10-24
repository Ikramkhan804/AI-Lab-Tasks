"""
Lab 5: Adversarial Search - Minimax & Alpha-Beta Pruning
CAI601400 - Artificial Intelligence Lab Manual

Lab Tasks implemented here
  Task 1: Trace the supplied tree by hand using Minimax   -> trace_minimax()
  Task 2: Add a leaf-evaluation counter to both algorithms -> minimax(), alphabeta()
  Task 3: Reorder branches and observe pruning changes     -> reorder_experiment()
Plus: edge-case / verification tests                       -> run_edge_cases()

Trees are complete binary trees stored as a flat list of leaf utilities
(left-to-right). Root is a MAX node; levels alternate MAX, MIN, MAX ...
"""
import itertools
import math
import random

NEG_INF = float("-inf")
POS_INF = float("inf")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def pad_scores(scores):
    """Pad with 0 to the next power of two (as in the manual's driver code).
    Returns (padded_scores, depth)."""
    scores = list(scores)
    depth = math.ceil(math.log2(len(scores))) if len(scores) > 1 else 0
    scores += [0] * (2 ** depth - len(scores))
    return scores, depth


# --------------------------------------------------------------------------
# Task 2a: Minimax with leaf counter
# --------------------------------------------------------------------------
def minimax(depth, index, max_turn, scores, target, counter):
    """Plain Minimax. counter is a one-element list used as a mutable counter."""
    if depth == target:
        counter[0] += 1                      # leaf evaluated
        return scores[index]
    left = minimax(depth + 1, index * 2, not max_turn, scores, target, counter)
    right = minimax(depth + 1, index * 2 + 1, not max_turn, scores, target, counter)
    return max(left, right) if max_turn else min(left, right)


# --------------------------------------------------------------------------
# Task 2b: Alpha-Beta with leaf counter
# --------------------------------------------------------------------------
def alphabeta(depth, index, max_turn, scores, target, alpha, beta, counter):
    if depth == target:
        counter[0] += 1                      # leaf evaluated
        return scores[index]
    if max_turn:
        value = NEG_INF
        for i in range(2):
            value = max(value, alphabeta(depth + 1, index * 2 + i, False,
                                         scores, target, alpha, beta, counter))
            alpha = max(alpha, value)
            if alpha >= beta:                # beta cut-off
                break
        return value
    value = POS_INF
    for i in range(2):
        value = min(value, alphabeta(depth + 1, index * 2 + i, True,
                                     scores, target, alpha, beta, counter))
        beta = min(beta, value)
        if alpha >= beta:                    # alpha cut-off
            break
    return value


def run_minimax(scores):
    scores, depth = pad_scores(scores)
    c = [0]
    return minimax(0, 0, True, scores, depth, c), c[0]


def run_alphabeta(scores):
    scores, depth = pad_scores(scores)
    c = [0]
    return alphabeta(0, 0, True, scores, depth, NEG_INF, POS_INF, c), c[0]


# --------------------------------------------------------------------------
# Task 1: bottom-up hand-trace table for Minimax
# --------------------------------------------------------------------------
def trace_minimax(scores):
    scores, depth = pad_scores(scores)
    print(f"Leaves (depth {depth}): {scores}")
    level_vals = scores
    for d in range(depth - 1, -1, -1):
        is_max = (d % 2 == 0)
        kind = "MAX" if is_max else "MIN"
        new_vals = []
        for i in range(0, len(level_vals), 2):
            a, b = level_vals[i], level_vals[i + 1]
            v = max(a, b) if is_max else min(a, b)
            new_vals.append(v)
            print(f"  depth {d} node {i // 2} ({kind}): "
                  f"{kind.lower()}({a}, {b}) = {v}")
        level_vals = new_vals
    print(f"  => Root value = {level_vals[0]}")
    return level_vals[0]


def trace_alphabeta(scores):
    """Alpha-beta with a printed log of every visit and every cut-off."""
    scores, depth = pad_scores(scores)
    log = []

    def ab(d, idx, is_max, alpha, beta):
        pad = "  " * d
        if d == depth:
            log.append(f"{pad}leaf[{idx}] = {scores[idx]}  (evaluated)")
            return scores[idx]
        kind = "MAX" if is_max else "MIN"
        log.append(f"{pad}{kind} d{d}n{idx}  alpha={alpha} beta={beta}")
        value = NEG_INF if is_max else POS_INF
        for i in range(2):
            child = ab(d + 1, idx * 2 + i, not is_max, alpha, beta)
            if is_max:
                value = max(value, child)
                alpha = max(alpha, value)
            else:
                value = min(value, child)
                beta = min(beta, value)
            if alpha >= beta:
                if i == 0:
                    log.append(f"{pad}  PRUNE right child of d{d}n{idx} "
                               f"(alpha={alpha} >= beta={beta})")
                break
        log.append(f"{pad}{kind} d{d}n{idx} returns {value}")
        return value

    root = ab(0, 0, True, NEG_INF, POS_INF)
    print("\n".join(log))
    print(f"  => Root value = {root}")
    return root


# --------------------------------------------------------------------------
# Task 3: reorder branches (swap children at internal nodes)
# --------------------------------------------------------------------------
def build_tree(scores):
    """Flat leaves -> nested [left, right] lists."""
    nodes = list(scores)
    while len(nodes) > 1:
        nodes = [[nodes[i], nodes[i + 1]] for i in range(0, len(nodes), 2)]
    return nodes[0]


def flatten(tree):
    if not isinstance(tree, list):
        return [tree]
    return flatten(tree[0]) + flatten(tree[1])


def apply_swaps(tree, swap_bits):
    """Swap children at internal nodes. swap_bits is consumed in pre-order."""
    it = iter(swap_bits)

    def rec(node):
        if not isinstance(node, list):
            return node
        l, r = rec(node[0]), rec(node[1])
        return [r, l] if next(it) else [l, r]

    return rec(tree)


def reorder_experiment(scores):
    scores, depth = pad_scores(scores)
    tree = build_tree(scores)
    n_internal = 2 ** depth - 1
    results = []
    for bits in itertools.product([0, 1], repeat=n_internal):
        leaves = flatten(apply_swaps(tree, bits))
        v, cnt = run_alphabeta(leaves)
        results.append((cnt, v, bits, leaves))
    values = {r[1] for r in results}
    counts = [r[0] for r in results]
    best = min(results, key=lambda r: r[0])
    worst = max(results, key=lambda r: r[0])
    return results, values, counts, best, worst


# --------------------------------------------------------------------------
# Edge cases / verification
# --------------------------------------------------------------------------
def run_edge_cases():
    print("Edge case 1: single leaf (depth 0)")
    print("  minimax:", run_minimax([7]), " alphabeta:", run_alphabeta([7]))

    print("Edge case 2: all leaves equal")
    print("  minimax:", run_minimax([4] * 8), " alphabeta:", run_alphabeta([4] * 8))

    print("Edge case 3: negative utilities")
    s = [-3, -5, -2, -9, -1, -7, -4, -8]
    print("  minimax:", run_minimax(s), " alphabeta:", run_alphabeta(s))

    print("Edge case 4: leaf count not a power of two (zero padding, as in 5.1)")
    s = [3, 5, 2, 9, 0, 7]
    print("  padded:", pad_scores(s)[0])
    print("  minimax:", run_minimax(s), " alphabeta:", run_alphabeta(s))

    print("Edge case 5: 2000 random trees - alpha-beta must equal minimax and "
          "never evaluate more leaves")
    random.seed(42)
    bad = 0
    saved = 0
    for _ in range(2000):
        d = random.choice([2, 3, 4, 5])
        s = [random.randint(-20, 20) for _ in range(2 ** d)]
        mv, mc = run_minimax(s)
        av, ac = run_alphabeta(s)
        if mv != av or ac > mc:
            bad += 1
        saved += mc - ac
    print(f"  mismatches/violations: {bad}   total leaves saved: {saved}")
    assert bad == 0


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
if __name__ == "__main__":
    tree_a = [3, 5, 2, 9, 0, 7]                 # Section 5.1 (padded to 8)
    tree_b = [3, 5, 2, 9, 3, 5, 2, 9]           # Section 5.2

    print("=" * 70)
    print("TASK 1: Minimax hand-trace (Section 5.1 tree)")
    print("=" * 70)
    trace_minimax(tree_a)

    print("\nTASK 1 (extra): Minimax hand-trace (Section 5.2 tree)")
    trace_minimax(tree_b)

    print("\n" + "=" * 70)
    print("TASK 2: Leaf-evaluation counters")
    print("=" * 70)
    for name, t in (("Tree A [3,5,2,9,0,7]+pad", tree_a),
                    ("Tree B [3,5,2,9,3,5,2,9]", tree_b)):
        mv, mc = run_minimax(t)
        av, ac = run_alphabeta(t)
        print(f"{name}\n  Minimax   : value={mv}, leaves evaluated={mc}"
              f"\n  Alpha-Beta: value={av}, leaves evaluated={ac}"
              f"\n  Same value: {mv == av}")

    print("\nAlpha-Beta trace for Tree A:")
    trace_alphabeta(tree_a)
    print("\nAlpha-Beta trace for Tree B:")
    trace_alphabeta(tree_b)

    print("\n" + "=" * 70)
    print("TASK 3: Branch reordering (all 2^7 = 128 child-swap configurations)")
    print("=" * 70)
    for name, t in (("Tree A", tree_a), ("Tree B", tree_b)):
        results, values, counts, best, worst = reorder_experiment(t)
        dist = {c: counts.count(c) for c in sorted(set(counts))}
        print(f"{name}: root values seen = {values}  (must be a single value)")
        print(f"  leaves evaluated: min={min(counts)}, max={max(counts)}, "
              f"distribution={dist}")
        print(f"  best ordering : {best[3]} -> {best[0]} leaves")
        print(f"  worst ordering: {worst[3]} -> {worst[0]} leaves")

    print("\n" + "=" * 70)
    print("EDGE CASES / COMPARISON TESTS")
    print("=" * 70)
    run_edge_cases()