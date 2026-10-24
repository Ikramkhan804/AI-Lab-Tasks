from pathlib import Path
import heapq
from itertools import count

# Explicit substitute graph because the referenced Section 3.2 source was not in the workspace.
GRAPH = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],  # B -> A creates a cycle; search must avoid looping.
    "C": ["F"],
    "D": ["G"],
    "E": ["G"],
    "F": ["G"],
    "G": [],
}
START, GOAL = "A", "G"


def bfs_trace(graph, start, goal):
    queue = [start]
    discovered = {start}
    parent = {start: None}
    rows = []
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for successor in graph[node]:
            if successor not in discovered:
                discovered.add(successor)
                parent[successor] = node
                queue.append(successor)
        rows.append((node, list(queue)))
        if node == goal:  # stop when the goal is processed
            break
    path = []
    if goal in parent:
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
    return rows, order, path


def dfs_trace(graph, start, goal):
    # Stack holds full paths; the rightmost path is the top of stack.
    stack = [[start]]
    rows = []
    order = []
    found_path = []
    while stack:
        path = stack.pop()
        node = path[-1]
        order.append(node)
        if node == goal:  # stop when the goal is processed
            found_path = path
            rows.append((node, [list(p) for p in stack]))
            break
        # Reverse push preserves the stated left-to-right successor order.
        for successor in reversed(graph[node]):
            if successor not in path:  # path-local cycle check
                stack.append(path + [successor])
        rows.append((node, [list(p) for p in stack]))
    return rows, order, found_path


FOUND, CUTOFF, FAILURE = "FOUND", "CUTOFF", "FAILURE"


def dls(graph, start, goal, limit):
    """Depth-limited search; root depth is zero, with a path-local cycle check."""
    def visit(node, depth, path):
        if node == goal:
            return FOUND, path
        if depth == limit:
            # Cutoff only if a non-cycle successor exists beyond this limit.
            has_deeper_candidate = any(child not in path for child in graph[node])
            return (CUTOFF if has_deeper_candidate else FAILURE), []
        saw_cutoff = False
        for child in graph[node]:
            if child in path:
                continue
            status, result_path = visit(child, depth + 1, path + [child])
            if status == FOUND:
                return FOUND, result_path
            if status == CUTOFF:
                saw_cutoff = True
        return (CUTOFF if saw_cutoff else FAILURE), []

    return visit(start, 0, [start])


def iddfs(graph, start, goal, maximum_depth):
    tried = []
    for limit in range(1, maximum_depth + 1):
        tried.append(limit)
        status, path = dls(graph, start, goal, limit)
        if status == FOUND:
            return status, path, tried
        if status == FAILURE:
            return status, [], tried
    return CUTOFF, [], tried


# Weighted extension of the same directed graph plus an explicitly added A -> G corridor.
WEIGHTED_GRAPH = {
    "A": [("B", 2), ("C", 3), ("G", 20)],
    "B": [("A", 2), ("D", 2), ("E", 5)],
    "C": [("F", 2)],
    "D": [("G", 8)],
    "E": [("G", 2)],
    "F": [("G", 2)],
    "G": [],
}


def ucs(graph, start, goal):
    serial = count()
    frontier = [(0, next(serial), start, [start])]
    best_cost = {start: 0}
    settled = set()
    processed = []
    while frontier:
        cost, _, node, path = heapq.heappop(frontier)
        if node in settled or cost != best_cost.get(node):
            continue
        settled.add(node)
        processed.append((node, cost))
        if node == goal:
            return path, cost, processed
        for child, weight in graph[node]:
            new_cost = cost + weight
            if child not in settled and new_cost < best_cost.get(child, float("inf")):
                best_cost[child] = new_cost
                heapq.heappush(frontier, (new_cost, next(serial), child, path + [child]))
    return [], float("inf"), processed


def path_cost(graph, path):
    total = 0
    for left, right in zip(path, path[1:]):
        total += next(weight for node, weight in graph[left] if node == right)
    return total


def main():
    print("GRAPH successor order:", GRAPH)
    for label, search in (("BFS", bfs_trace), ("DFS", dfs_trace)):
        rows, order, path = search(GRAPH, START, GOAL)
        print()
        print(f"{label} trace (frontier shown after processing; stack top is rightmost)")
        for node, frontier in rows:
            print(f"processed={node}; frontier={frontier}")
        print("visit order:", order)
        print("final path:", path)

    print()
    print("DLS at limits 1, 2, 3, 4")
    for limit in (1, 2, 3, 4):
        status, path = dls(GRAPH, START, GOAL, limit)
        print(f"limit={limit}; status={status}; path={path or 'none'}")

    print()
    print("IDDFS maximum depths 1, 2, 3, 4")
    for maximum in (1, 2, 3, 4):
        status, path, tried = iddfs(GRAPH, START, GOAL, maximum)
        print(f"maximum={maximum}; status={status}; path={path or 'none'}; limits_attempted={tried}")

    path, cost, processed = ucs(WEIGHTED_GRAPH, START, GOAL)
    print()
    print("UCS weighted graph")
    print("edge weights:", WEIGHTED_GRAPH)
    print("processed node/cost order:", processed)
    print("UCS path:", path)
    print("returned cost:", cost)
    print("independent path-cost sum:", path_cost(WEIGHTED_GRAPH, path))
    bfs_rows, bfs_order, bfs_path = bfs_trace(
        {node: [child for child, _ in children] for node, children in WEIGHTED_GRAPH.items()},
        START, GOAL
    )
    print("BFS on weighted topology visit order:", bfs_order)
    print("BFS fewest-hop path:", bfs_path)
    print("BFS route cost:", path_cost(WEIGHTED_GRAPH, bfs_path))


if __name__ == "__main__":
    main()
