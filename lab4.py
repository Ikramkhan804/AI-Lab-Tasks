"""Lab 4: A* heuristic audit, UCS control, and overestimate experiment.

Inputs transcribed from the supplied screenshots:
  S -> A (1), B (4)
  A -> C (2)
  B -> G (5)
  C -> G (3)
  h = {S: 5, A: 4, B: 4, C: 2, G: 0}

Priority queue tuple is (f, g, node, path), so ties use Python tuple order:
f first, then g, then node label, then path. Successors are considered in the
listed order. Valid goal processing is included in the processed trace but
excluded from expanded nodes/count. Stale entries are ignored before goal
checking. Standard-library Python only.
"""

from heapq import heappop, heappush
from itertools import count

GRAPH = {
    "S": [("A", 1), ("B", 4)],
    "A": [("C", 2)],
    "B": [("G", 5)],
    "C": [("G", 3)],
    "G": [],
}
HEURISTIC = {"S": 5, "A": 4, "B": 4, "C": 2, "G": 0}
START = "S"
GOAL = "G"


def path_cost(graph, path):
    """Independently sum the weights along a returned path."""
    total = 0
    for source, target in zip(path, path[1:]):
        total += next(weight for neighbor, weight in graph[source]
                      if neighbor == target)
    return total


def best_first_search(graph, heuristic, start, goal):
    """Run A* (or UCS when heuristic is all zero).

    Return path, reported cost, processed trace (goal included), expanded
    nodes (goal excluded), and stale entries skipped.
    """
    sequence = count()
    frontier = [(heuristic[start], 0, start, [start])]
    best_g = {start: 0}
    processed = []
    expanded = []
    stale_skipped = []

    while frontier:
        f, g, node, path = heappop(frontier)

        # Ignore outdated queue entries before processing a node or its goal.
        if g > best_g.get(node, float("inf")):
            stale_skipped.append((node, g, f))
            continue

        processed.append((node, g, heuristic[node], f))
        if node == goal:
            return {
                "path": path,
                "cost": g,
                "processed": processed,
                "expanded": expanded,
                "stale_skipped": stale_skipped,
            }

        expanded.append(node)
        for neighbor, weight in graph.get(node, []):
            new_g = g + weight
            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = new_g
                new_f = new_g + heuristic[neighbor]
                heappush(frontier,
                         (new_f, new_g, neighbor, path + [neighbor]))

    return {
        "path": [], "cost": float("inf"), "processed": processed,
        "expanded": expanded, "stale_skipped": stale_skipped,
    }


def manual_first_three(graph, heuristic, start):
    """Print first three non-goal expansions with arithmetic/frontier."""
    sequence = count()
    frontier = [(heuristic[start], 0, start, [start])]
    best_g = {start: 0}
    expansion_number = 0

    while frontier and expansion_number < 3:
        f, g, node, path = heappop(frontier)
        if g > best_g.get(node, float("inf")):
            continue
        if node == GOAL:
            break

        expansion_number += 1
        print(f"Expansion {expansion_number}: select {node}")
        print(f"  g={g}; h={heuristic[node]}; f=g+h={g}+{heuristic[node]}={f}")
        print("  Generate successors in listed order:")
        for neighbor, weight in graph.get(node, []):
            new_g = g + weight
            old = best_g.get(neighbor, float("inf"))
            if new_g < old:
                best_g[neighbor] = new_g
                new_f = new_g + heuristic[neighbor]
                heappush(frontier,
                         (new_f, new_g, neighbor, path + [neighbor]))
                print(f"    {node}->{neighbor}: g={g}+{weight}={new_g}; "
                      f"h={heuristic[neighbor]}; f={new_g}+"
                      f"{heuristic[neighbor]}={new_f}")

        remaining = sorted(frontier)
        print("  Remaining frontier (node: g, h, f):")
        for nf, ng, nnode, _ in remaining:
            print(f"    {nnode}: g={ng}, h={heuristic[nnode]}, f={nf}")
        if remaining:
            nf, ng, nnode, _ = remaining[0]
            print(f"  Next selected: {nnode}, lowest priority tuple "
                  f"(f={nf}, g={ng}, node={nnode})")
        print()


def print_run(label, graph, heuristic):
    result = best_first_search(graph, heuristic, START, GOAL)
    independently_summed = path_cost(graph, result["path"])
    print(label)
    print("  processed (goal included):", result["processed"])
    print("  expanded (goal excluded):", result["expanded"])
    print("  expansion count:", len(result["expanded"]))
    print("  returned path:", " -> ".join(result["path"]))
    print("  returned cost:", result["cost"])
    print("  independent path-cost sum:", independently_summed)
    print("  stale entries skipped:", result["stale_skipped"])
    print()
    assert result["cost"] == independently_summed
    return result


def main():
    print("First three A* expansions by hand, verified by the queue run")
    manual_first_three(GRAPH, HEURISTIC, START)

    print_run("A* with supplied heuristic", GRAPH, HEURISTIC)
    zero_heuristic = {node: 0 for node in GRAPH}
    print_run("UCS control (zero heuristic)", GRAPH, zero_heuristic)

    # Change only h(C); preserve every edge, weight, endpoint, and queue rule.
    overestimate = dict(HEURISTIC)
    overestimate["C"] = 10
    print_run("A* overestimate experiment (only h(C)=10)",
              GRAPH, overestimate)

    print("Heuristic audit")
    true_remaining = {"S": 6, "A": 5, "B": 5, "C": 3, "G": 0}
    for node in GRAPH:
        print(f"  {node}: supplied h={HEURISTIC[node]}, "
              f"true remaining cost={true_remaining[node]}, "
              f"admissible={HEURISTIC[node] <= true_remaining[node]}")
    print("  Consistency edge checks:")
    for node, edges in GRAPH.items():
        for neighbor, weight in edges:
            lhs = HEURISTIC[node]
            rhs = weight + HEURISTIC[neighbor]
            print(f"    {node}->{neighbor}: {lhs} <= {weight}+"
                  f"{HEURISTIC[neighbor]}={rhs}: {lhs <= rhs}")
    print("  Overestimate check: h(C)=10 > true remaining cost 3; "
          "on C->G, 10 <= 3+0 is false.")


if __name__ == "__main__":
    main()
