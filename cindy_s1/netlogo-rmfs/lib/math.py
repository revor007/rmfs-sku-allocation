import pandas as pd
from lib.constant import *

global pod_path
pod_path = None

def calculate_distance(x0, y0, x1, y1):
    return abs(((x0 - x1) * (x0 - x1)) + ((y0 - y1) * (y0 - y1)))

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
     
    return intersection / union  

def compute_jaccard_similarity(data):
    similarity_dict = {}
    grouped = data.groupby('order_id')['item_id'].apply(set)
    for order_dum, items in grouped.items():
        similarities = []
        for other_order_dum, other_items in grouped.items():
            if order_dum == other_order_dum:
                similarities.append(1.0)  # similarity with itself is 1
            else:
                similarity = jaccard_similarity(items, other_items)
                similarities.append(similarity)
        similarity_dict[order_dum] = similarities
    return grouped, similarity_dict

def calculateManhattanDistance(u, v):
    return abs(u[0] - v[0]) + abs(u[1] - v[1])

def calculateEuclideanDistance(u, v):
    return ((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2) ** 0.5

def calculateUnidirectionRMFSHeuristic(u, v):
    if isinstance(u, str):
        u = tuple(map(float, u.split(",")))
    if isinstance(v, str):
        v = tuple(map(float, v.split(",")))
    return unidirectionRMFSHeuristic(start_node=u, end_node=v)

def directionConstrainedManhattanDistance(pod_path_dictionary=None, start_node=None, end_node=None, robot=False):

    global pod_path

    if pod_path is None:
        pod_path = pd.read_csv(PARENT_DIRECTORY + '/data/output/generated_pod.csv', header=None
                                     ).to_dict(orient='records')

    try:
        direction_constraint_above_dest = pod_path[end_node[1] + 1][end_node[0]]
    except:
        direction_constraint_above_dest = 3
    try:
        direction_constraint_below_dest = pod_path[end_node[1] - 1][end_node[0]]
    except:
        direction_constraint_below_dest = 3

    if direction_constraint_above_dest > direction_constraint_below_dest:
        y_end_node = end_node[1] + 1
    else:
        y_end_node = end_node[1] - 1
    direction_constraint_nearest_dest = max(direction_constraint_above_dest, direction_constraint_below_dest)

    manhattan_distance = calculateManhattanDistance(
        (start_node[0], start_node[1]), (end_node[0], end_node[1]))
    
    if robot == False:
        return manhattan_distance, 0

    direction_constrained_manhattan_distance = manhattan_distance
    turn_amount = 2
    
    # U turn calculation for Direction-constrained Manhattan Distance
    if start_node[0] < end_node[0]: # Destination is to the right of the robot's position
        if direction_constraint_nearest_dest == 4:  # Direction Constraint -> Left
            # print(start_node[0], start_node[1])
            # print(end_node[0], end_node[1], direction_constraint_nearest_dest)
            # X-axis u-turn
            check_right = True
            delta_x = 0
            while check_right:
                direction_constraint_nearest_right = pod_path[y_end_node][end_node[0] + delta_x]
                if direction_constraint_nearest_right != 4:
                    check_right = False
                delta_x += 1 
            direction_constrained_manhattan_distance += 2 * delta_x # U-turn distance
            turn_amount += 1

            # Y-axis u-turn
            delta_y = 0
            if start_node[1] < end_node[1]: # Destination is above the robot's position
                if direction_constraint_nearest_right == 6: # Direction Constraint -> Down
                    delta_y = 3
                    direction_constrained_manhattan_distance += 2 * delta_y
                    turn_amount += 1
            else:
                if direction_constraint_nearest_right == 7: # Direction Constraint -> Up
                    delta_y = 3
                    direction_constrained_manhattan_distance += 2 * delta_y
                    turn_amount += 1
            # print(manhattan_distance, direction_constrained_manhattan_distance, delta_x, delta_y, turn_amount)
        
    elif start_node[0] > end_node[0]: # Destination is to the left of the robot's position
        if direction_constraint_nearest_dest == 5: # Direction Constraint -> Right
            # X-axis u-turn
            check_left = True
            delta_x = 0
            while check_left:
                direction_constraint_nearest_left = pod_path[y_end_node][end_node[0] - delta_x]
                if direction_constraint_nearest_left != 5:
                    check_left = False
                delta_x += 1
            direction_constrained_manhattan_distance += 2 * delta_x
            turn_amount += 1
        if pod_path[start_node[1]][start_node[0]] == 5: # Direction Constraint -> Right
            check_right = True
            delta_x = 0
            while check_right:
                direction_constraint_nearest_right = pod_path[y_end_node][end_node[0] + delta_x]
                if direction_constraint_nearest_right != 4:
                    check_right = False
                delta_x += 1 
            direction_constrained_manhattan_distance += 2 * delta_x # U-turn distance
            turn_amount += 1

    return direction_constrained_manhattan_distance, turn_amount

def calculateEnergy(velocity, acceleration):
        
    tick_unit = TICK_TO_SECOND
    average_speed = velocity + (0.5 * acceleration * tick_unit)
    MAXIMUM_SPEED = 1.5
    # energy consumption related, based on Leo
    MASS = 300
    LOAD_MASS = 0
    GRAVITY = 9.8
    FRICTION = 0.02
    INERTIA = 0.15
    LIFT_COEF = 0.2
    ROBOT_WIDTH = 0.6 # m
    ROBOT_RADIUS = ROBOT_WIDTH / 2
    LENGTH = 0.75 # m
        
    if acceleration > 0 and velocity != 0:
        return (MASS + LOAD_MASS) * ((GRAVITY * FRICTION) + (
                acceleration * INERTIA)) * average_speed * tick_unit
    elif acceleration < 0 and velocity != 0:
        return (MASS + LOAD_MASS) * abs(-(GRAVITY * FRICTION) + (
                acceleration * INERTIA)) * average_speed * tick_unit
    elif velocity != 0:
        return (MASS + LOAD_MASS) * GRAVITY * FRICTION * velocity * tick_unit
    return 0

def unidirectionRMFSHeuristic(pod_path_dict=None, start_node=None, end_node=None, robot=False):
    """
    Unidirectional RMFS Heuristic
    """
    global pod_path

    if pod_path is None:
        pod_path = pd.read_csv(PARENT_DIRECTORY + '/data/output/generated_pod.csv', header=None
                                     ).to_dict(orient='records')
    
    if robot == False:
        d_m = calculateManhattanDistance(start_node, end_node)
        turn_amount = 2
    else:
        d_m, turn_amount = directionConstrainedManhattanDistance(pod_path,
            start_node=start_node, end_node=end_node, robot=robot)
    
    A_acc = 1.0
    A_dec = 1.0
    V_bar = (d_m/2) * (3/5) if d_m < 5 else 1.5
    t_acc = turn_amount * (V_bar / A_acc)
    t_dec = turn_amount * (V_bar / A_dec)
    d_acc_dec = (V_bar**2 / 2) * ((1/A_acc) + (1/A_dec))
    t_const = (d_m - (turn_amount * d_acc_dec)) / V_bar if V_bar != 0 else 0
    
    e_const = max( 0, t_const * calculateEnergy(V_bar, 0))
    e_acc = t_acc * calculateEnergy(V_bar/2, A_acc)
    e_dec = t_dec * calculateEnergy(V_bar/2, -A_dec)

    e_total = e_const + e_acc + e_dec # convert to kJ
    return e_total
