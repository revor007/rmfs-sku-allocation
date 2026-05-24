import networkx as nx
from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function
import heapq
from lib.types.netlogo_coordinate import NetLogoCoordinate
from lib.math import unidirectionRMFSHeuristic, calculateUnidirectionRMFSHeuristic
from lib.constant import *
import pandas as pd
class DirectedGraph:
    key = ''

    def __init__(self):
        """Initialize an instance with a directed graph."""
        self.graph = nx.DiGraph()

    @staticmethod
    def nodeValid(node):
        """Check if a node is valid based on custom logic.

        Args:
            node (str): The node in format 'x,y'.

        Returns:
            bool: True if the node is valid, False otherwise.
        """
        x, y = map(int, node.split(","))
        return x >= 2 and y >= 0

    def addNode(self, node):
        """Add a node to the graph if it's valid.

        Args:
            node (str): The node to add.
        """
        if self.nodeValid(node):
            self.graph.add_node(node)

    def addEdge(self, start, end, weight):
        """Add an edge between two nodes with a weight if both nodes are valid.

        Args:
            start (str): The start node.
            end (str): The end node.
            weight (float): The weight of the edge.
        """
        if self.nodeValid(start) and self.nodeValid(end):
            self.graph.add_edge(start, end, weight=weight)
    
    def add_all_direction_paths(self, obj_key, weight):
        x, y = map(int, obj_key.split(','))
        directions = {
            'left': (x - 1, y),
            'right': (x + 1, y),
            'up': (x, y + 1),
            'down': (x, y - 1)
        }

        for (key, tuple) in directions.items():
            (nx, ny) = tuple
            neighbor_key = f"{nx},{ny}"
            self.addEdge(obj_key, neighbor_key, weight=weight)

    @staticmethod
    def getHeading(p1: NetLogoCoordinate, p2: NetLogoCoordinate):
        if p1.x == p2.x:
            if p1.y > p2.y:
                return 180
            else:
                return 0
        elif p1.y == p2.y:
            if p1.x > p2.x:
                return 270
            else:
                return 90

    def dijkstraModified(self, start, end, penalties, zone_boundary, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Create a copy of the graph so we can modify it without affecting the original
        G = self.graph.copy()

        # Increase the weight of the edges leading to and from the nodes to avoid
        if avoid:
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 10000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 10000

        # Increase the weight of edges in every zone based on the penalty
        for index, zone in enumerate(zone_boundary):
            for row in range(zone[1][0], zone[0][0]):
                for col in range(zone[0][1], zone[1][1]):
                    coordinate_str = f"{row},{col}"
                    for neighbor in list(G.neighbors(coordinate_str)) + list(G.predecessors(coordinate_str)):
                        if G.has_edge(neighbor, coordinate_str):
                            G[neighbor][coordinate_str]['weight'] = penalties[index]
                        if G.has_edge(coordinate_str, neighbor):
                            G[coordinate_str][neighbor]['weight'] = penalties[index]

        try:
            # Use Dijkstra's algorithm to find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
            return path
        except nx.NetworkXNoPath:
            return None

    def dijkstra(self, start, end, avoid=False):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Increase the weight of the edges leading to and from the nodes to avoid
        if len(avoid) > 0:
            G = self.graph.copy()
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 1000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 1000
            try:
                # Use Dijkstra's algorithm to find the shortest path
                path = nx.shortest_path(G, source=start, target=end, weight='weight', method='dijkstra')
                return path
            except nx.NetworkXNoPath:
                return None
        else:
            try:
                # Use Dijkstra's algorithm to find the shortest path
                path = nx.shortest_path(self.graph, source=start, target=end, weight='weight', method='dijkstra')
                return path
            except nx.NetworkXNoPath:
                return None
        
    def astarWaypoint(self, start, W1, W2, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """

        self.pod_path = pd.read_csv(PARENT_DIRECTORY + '/data/output/generated_pod.csv', header=None
                                     ).to_dict(orient='records')

        # Increase the weight of the edges leading to and from the nodes to avoid
        if len(avoid) > 0:
            G = self.graph.copy()
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 1000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 1000

            try:
                # Use Dijkstra's algorithm to find the shortest path
                path1 = nx.astar_path(G, source=start, target=W1, heuristic=calculateUnidirectionRMFSHeuristic, weight='weight')
                path2 = nx.astar_path(G, source=W1, target=W2, weight='weight')
                return path1 + path2[1:]
            except nx.NetworkXNoPath:
                return None
            
        else:
            if W1 == W2:
                path = nx.astar_path(self.graph, source=start, target=W2, heuristic=calculateUnidirectionRMFSHeuristic, weight='weight')
                return path
            else:
                try:
                    # Use Dijkstra's algorithm to find the shortest path
                    path1 = nx.astar_path(self.graph, source=start, target=W1, heuristic=calculateUnidirectionRMFSHeuristic, weight='weight')
                    path2 = nx.astar_path(self.graph, source=W1, target=W2, heuristic=calculateUnidirectionRMFSHeuristic, weight='weight')
                    return path1 + path2[1:]
                except nx.NetworkXNoPath:
                    return None
                
    def RRAWaypoint(self, start, W1, W2, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        path1 = None
        path2 = None
        state = {}

        # Increase the weight of the edges leading to and from the nodes to avoid
        if len(avoid) > 0:
            G = self.graph.copy()
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 1000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 1000

            try:
                while path1 is None:
                    path1, state = self.reverse_resumable_astar(G, source=start, target=W1, state=state)
                state = {}
                while path2 is None:
                    path2, state = self.reverse_resumable_astar(G, source=W1, target=W2,  state=state)
                return path1 + path2[1:]
            except nx.NetworkXNoPath:
                return None
            
        else:
            if W1 == W2:
                while path1 is None:
                    path1, state = self.reverse_resumable_astar(self.graph, source=start, target=W1, state=state)
                return path1
            
            else:
                try:
                    while path1 is None:
                        path1, state = self.reverse_resumable_astar(self.graph, source=start, target=W1, state=state)
                    state = {}
                    while path2 is None:
                        path2, state = self.reverse_resumable_astar(self.graph, source=W1, target=W2,  state=state)
                    
                    return path1 + path2[1:]
                except nx.NetworkXNoPath:
                    return None
    
    def reverse_resumable_astar(self, G, source, target, state, heuristic="direction-constrained-manhattan", weight="weight"):
        """
        One step of the reverse resumable A* algorithm.
        Call this repeatedly with updated state to resume the search.
        
        Parameters:
            G        : A NetworkX DiGraph
            source   : Source node
            target   : Target node (where reverse search begins)
            state    : Dictionary storing search state
            heuristic: Heuristic name
            weight   : Edge weight key

        Returns:
            path     : List of nodes if goal found, else None
            state    : Updated search state (for resumption)
        """
        # Init state on first call
        if not state:
            state['counter'] = count()
            state['open_set'] = [(0, next(state['counter']), target, 0, None)]  # (f, tie, node, g, parent)
            state['open_dict'] = {}
            state['closed_set'] = {}

        def manhattan(u, v):
            return abs(u[0] - v[0]) + abs(u[1] - v[1])

        def direction_constrained(u, v):
            if isinstance(u, str):
                u = tuple(map(float, u.split(",")))
            if isinstance(v, str):
                v = tuple(map(float, v.split(",")))
            return unidirectionRMFSHeuristic(pod_path=self.pod_path, start_node=u, end_node=v)

        heuristic_fn = {
            "manhattan": manhattan,
            "direction-constrained-manhattan": direction_constrained
        }.get(heuristic, None)

        if heuristic_fn is None:
            raise ValueError("Unsupported heuristic")

        weight = _weight_function(G, weight)

        G_pred = G._pred  # reversed graph traversal

        if not state['open_set']:
            return None, state  # nothing to expand

        _, __, curr, g, parent = heapq.heappop(state['open_set'])

        if curr in state['closed_set']:
            return None, state

        state['closed_set'][curr] = parent

        if curr == source:
            # Reconstruct path
            path = [curr]
            while parent is not None:
                path.append(parent)
                parent = state['closed_set'].get(parent)
            # path.reverse()
            return path, state

        for neighbor, edge_data in G_pred[curr].items():
            cost = weight(curr, neighbor, edge_data)
            if cost is None:
                continue

            h = heuristic_fn(neighbor, source)
            f = g + cost + h

            if neighbor in state['open_dict'] and state['open_dict'][neighbor] <= g + cost:
                continue

            state['open_dict'][neighbor] = g + cost
            heapq.heappush(state['open_set'], (f, next(state['counter']), neighbor, g + cost, curr))

        return None, state
    
    def whca_star_path(G, source, target, heuristic="manhattan", weight="weight", *,
                    current_time=0, window_size=10, reservation_table=None):
        """
        WHCA* A*-based pathfinding with a time-window and reservation table.
        """

        if heuristic == "manhattan":
            def heuristic(u, v):
                return abs(u[0] - v[0]) + abs(u[1] - v[1])
        if heuristic == "euclidean":
            def heuristic(u, v):
                return ((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2) ** 0.5
        # if heuristic is "modified_manhattan":
        #     # Handles unidirectional nodes

        weight = _weight_function(G, weight)
        G_succ = G._adj
        c = count()
        
        # Priority queue stores: (f_score, tie_breaker, node, g_score, time_step, parent)
        queue = [(0, next(c), source, 0, current_time, None)]
        enqueued = {}
        explored = {}

        while queue:
            _, __, curr, g, t, parent = heappop(queue)

            if t - current_time >= window_size or t > 1000:
                continue  # Exceeded window size or safety limit

            if (curr, t) in explored:
                continue
            explored[(curr, t)] = parent

            if curr == target:
                path = [(curr, t)]
                while parent is not None:
                    curr, t = parent
                    path.append((curr, t))
                    parent = explored.get((curr, t))
                return list(reversed(path))

            for neighbor, edge_data in G_succ[curr].items():
                cost = weight(curr, neighbor, edge_data)
                if cost is None:
                    continue

                next_time = t + 1
                if reservation_table and (neighbor, next_time) in reservation_table:
                    continue  # Another agent is occupying this node at that time

                h = heuristic(neighbor, target)
                f = g + cost + h

                if (neighbor, next_time) in enqueued and enqueued[(neighbor, next_time)] <= g + cost:
                    continue

                enqueued[(neighbor, next_time)] = g + cost
                heappush(queue, (f, next(c), neighbor, g + cost, next_time, (curr, t)))

        return None  # No path found within this window
    
