from collections import deque


def get_action(curr, next_node):
    """Determine the action required to move from current node to next node."""
    curr_x, curr_y = curr
    next_x, next_y = next_node

    if next_x > curr_x:
        return 'East'
    elif next_x < curr_x:
        return 'West'
    elif next_y > curr_y:
        return 'North'
    elif next_y < curr_y:
        return 'South'
    else:
        return None  # Same position


def bfs_distance_and_first_action(graph, source):
    """
    Find shortest path from source to all other nodes.
    Returns a dictionary mapping each destination to a tuple of (distance, first_action).
    """
    visited = set([source])
    queue = deque([(source, None, 0)])  # (node, first_action, distance) tuples
    results = {}

    while queue:
        current, action, distance = queue.popleft()

        # If this is not the source, record the result
        if current != source:
            results[current] = (distance, action)

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)

                # Determine the action to take
                next_action = get_action(current, neighbor)

                # If this is a direct neighbor of the source, this is the first action
                # Otherwise, propagate the first action that was taken from the source
                first_action = next_action if current == source else action

                queue.append((neighbor, first_action, distance + 1))

    return results


def all_pairs_first_actions(graph):
    """
    Find the distance and first action in the shortest path between all pairs of nodes.
    Returns a dictionary mapping (source, destination) to (distance, first_action) tuple.
    """
    all_results = {}

    for source in graph:
        results = bfs_distance_and_first_action(graph, source)
        all_results[(source, source)] = (0, 'Stop')
        for destination, (distance, action) in results.items():
            all_results[(source, destination)] = (distance, action)

    return all_results


def create_graph(game_state):
    """
    Creates a graph of the game state, nodes being all valid positions and edges between adjacent nodes.
    """
    width, height = game_state.data.layout.width, game_state.data.layout.height
    walls = set(game_state.get_walls().as_list())

    graph = {}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for x in range(width):
        for y in range(height):
            if (x, y) in walls:
                continue

            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls):
                    neighbors.append((nx, ny))

            if neighbors:
                graph[(x, y)] = neighbors
            else:
                graph[(x, y)] = []

    return graph
