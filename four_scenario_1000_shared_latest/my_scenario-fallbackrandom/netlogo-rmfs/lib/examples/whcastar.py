import networkx as nx
from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function



class Agent:
    def __init__(self, id, start, goal, priority=0):
        self.id = id
        self.start = start
        self.goal = goal
        self.current_pos = start
        self.path = []
        self.priority = priority

    def set_path(self, path):
        self.path = path
        if path:
            self.current_pos = path[-1][0]  # Last node in window

    def advance(self):
        if self.path:
            self.current_pos = self.path.pop(0)[0]


agents = [Agent(1, (0, 0), (5, 5)), Agent(2, (2, 2), (0, 0))]

max_time = 1000
WINDOW_SIZE = 10
G = nx.grid_2d_graph(10, 10)  # Example grid graph

for current_time in range(0, max_time, WINDOW_SIZE):
    reservation_table = set()  # Clear or keep persistent depending on logic

    for agent in agents:
        path = whca_star_path(G, agent.current_pos, agent.goal,
                              current_time=current_time,
                              window_size=WINDOW_SIZE,
                              reservation_table=reservation_table)
        if path:
            agent.set_path(path)
            for node, t in path:
                reservation_table.add((node, t))
    
    for step in range(WINDOW_SIZE):
        for agent in agents:
            agent.advance()
        # (Optional) visualize, log, or store agent positions
