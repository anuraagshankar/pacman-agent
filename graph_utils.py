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
    Find the shortest path from source to all other nodes.
    Returns a dictionary mapping each destination to a tuple of 
    (distance, list_of_first_actions). For the source, the action is ['Stop'].
    """
    # best: maps node -> (distance, set_of_first_actions)
    best = {source: (0, set())}
    # Queue holds (node, set_of_first_actions) for that node at the best distance
    queue = deque([(source, set())])

    while queue:
        current, actions = queue.popleft()
        current_distance, _ = best[current]

        for neighbor in graph[current]:
            new_distance = current_distance + 1
            # Determine the first action:
            # If we are at the source, compute it directly;
            # otherwise, inherit the first action(s) from the current node.
            if current == source:
                new_actions = {get_action(current, neighbor)}
            else:
                new_actions = actions

            if neighbor not in best:
                best[neighbor] = (new_distance, new_actions)
                queue.append((neighbor, new_actions))
            else:
                recorded_distance, recorded_actions = best[neighbor]
                if new_distance < recorded_distance:
                    best[neighbor] = (new_distance, new_actions)
                    queue.append((neighbor, new_actions))
                elif new_distance == recorded_distance:
                    # Union the new actions with the recorded ones
                    union_actions = recorded_actions.union(new_actions)
                    if union_actions != recorded_actions:
                        best[neighbor] = (recorded_distance, union_actions)
                        queue.append((neighbor, union_actions))

    # Convert the sets of actions to lists.
    results = {}
    for node, (dist, actions) in best.items():
        if node == source:
            results[node] = (dist, ['Stop'])
        else:
            results[node] = (dist, list(actions))
    return results


def all_pairs_first_actions(graph):
    """
    Find the distance and first action in the shortest path between all pairs of nodes.
    Returns a dictionary mapping (source, destination) to (distance, first_action) tuple.
    """
    all_results = {}

    for source in graph:
        results = bfs_distance_and_first_action(graph, source)
        all_results[(source, source)] = (0, ['Stop'])
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


def find_entry_points(graph, root, my_area_nodes):
    # Convert my_area_nodes to a set for O(1) lookups
    my_area = set(my_area_nodes)

    # Check if root is in enemy territory
    if root not in my_area:
        return set()  # Root is already in enemy territory

    visited = set([root])
    queue = deque([root])
    entry_points = set()

    while queue and not entry_points:
        current = queue.popleft()

        # If this is my area node, check if it connects to any enemy nodes
        if current in my_area:
            has_enemy_neighbor = False

            for neighbor in graph.get(current, []):
                if neighbor not in my_area:
                    # This is an entry point - it connects to enemy territory
                    entry_points.add(current)
                    has_enemy_neighbor = True

                # Continue BFS, but only through my area
                if neighbor in my_area and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return entry_points
