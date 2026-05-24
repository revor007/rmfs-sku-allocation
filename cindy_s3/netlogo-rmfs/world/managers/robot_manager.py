from __future__ import annotations
from typing import List, TYPE_CHECKING
from world.entities.robot import Robot
import json
import numpy as np
import datetime
from lib.math import *
import requests
import math

if TYPE_CHECKING:
    from world.warehouse import Warehouse

model_init_url = "http://127.0.0.1:8100/initialize"
model_predict_url = "http://127.0.0.1:8100/predict"

class RobotManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.robots: List[Robot] = []
        self.robot_counter = 0
        
        self.heuristic_rl = False # Ryan's HG-MDRL
        self.deterministic_rl = False # Deterministic = False
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_buffer = []
        self.log_buffer_size = 100
        self.current_rl_state = {}
        self.gamma = 0
        self.gamma_n = self.gamma
        self.frameskip = 4
        self.num_steps = 100000
        
        self.tick_for_finishing_task = []
        self.rl_done = 0
        self.memory = None
    
    def initRobotManager(self):
        for robot in self.robots:
            robot.setRobotManager(self)
            
        if self.heuristic_rl:
            hyperparameters = TEST_HYPERPARAMETERS
            self.num_agents = self.robot_counter
            self.previous_rl_state = np.zeros((self.robot_counter, 18), dtype=np.float32)
            self.previous_rl_map_state = np.zeros((49, 31))
            self.previous_actions = [0 for _ in range(self.robot_counter)]
            self.action_masking = [[1 for _ in range(len(self.get_action_space()))] for _ in range(self.num_agents)]
            self.previous_action_masking = [[1 for _ in range(len(self.get_action_space()))] for _ in range(self.num_agents)]
            post_response = requests.post(model_init_url, json={
                'num_agents':self.num_agents,
                'state_dim':self.previous_rl_state.shape[1],
                'map_state_dim':(self.previous_rl_map_state.shape[0], self.previous_rl_map_state.shape[1]),
                'action_dim':len(self.get_action_space()),
                'lambd_0':hyperparameters['lambd_0'],
                'lambd_scheduler_alpha':hyperparameters['lambd_scheduler_alpha'],
                'actor_lr':hyperparameters['actor_lr'],
                'critic_lr':hyperparameters['critic_lr'],
                'gamma':hyperparameters['gamma'],
            }
            )
            print("Model Initialized!")
            self.gamma = hyperparameters['gamma']
            self.gamma_n = self.gamma
            self.model_path = os.path.join(
                os.getcwd(),
                'ai',
                'mappo_routing',
                f"alr-{hyperparameters['actor_lr']}_clr-{hyperparameters['critic_lr']}_lambd0-{hyperparameters['lambd_0']}_lambdaplha-{hyperparameters['lambd_scheduler_alpha']}_gamma-{hyperparameters['gamma']}",
                'model.pt'
            )
            self.detour_budget = 3
            self.detour_budget_list = [self.detour_budget for _ in range(self.robot_counter)]
            self.previous_valid_waypoints = [[1 for _ in range(len(self.get_action_space()))] for _ in range(self.robot_counter)]

    def getAllRobots(self):
        return self.robots
    
    def getRobotByName(self, robot_name):
        for o in self.getAllRobots():
            if o.object_type == "robot" and o.robotName() == robot_name:
                return o
    
    def findNearestAvailableRobot(self, coordinate):
        """
        Find the nearest available robot to a given coordinate.
        Returns None if no robots are available.
        """
        from lib.math import calculateManhattanDistance
        
        available_robots = []
        for robot in self.robots:
            if (robot.job is None or robot.job.is_finished) and robot.current_state == 'idle':
                distance = calculateManhattanDistance([robot.pos_x, robot.pos_y], [coordinate.x, coordinate.y])
                available_robots.append((robot, distance))
        
        if not available_robots:
            return None
            
        # Sort by distance and return the nearest
        available_robots.sort(key=lambda x: x[1])
        return available_robots[0][0]

    def getRobotByCoordinate(self, x, y):
        for o in self.getAllRobots():
            if o.object_type == "robot" and o.pos_x == x and o.pos_y == y:
                return o
                    
    def getRobotsByCoordinate(self, coords):
        robots = []
        for coord in coords:
            robot = self.getRobotByCoordinate(coord[0], coord[1])
            if robot:
                robots.append(robot)
        return robots
    
    def createRobot(self, x, y):
        robot = Robot(self.robot_counter, x, y)
        self.robots.append(robot)
        self.tick_for_finishing_task.append(0)
        self.robot_counter += 1
        robot._id = self.warehouse.total_pod + 1
        self.warehouse.total_pod += 1
        return robot
    

    # MDRL Functions

    def getCurrentWaypoints(self):
        waypoints = []
        for robot in self.robots:
            waypoints.append((robot.w1))

        print(f"Current Waypoints: {waypoints}")
        return waypoints

    def updateWaypoint(self):
        self.pod_path = pd.read_csv(PARENT_DIRECTORY + '/data/output/generated_pod.csv', header=None
                                    ).to_dict(orient='records')
        self.action_masking = []
        default_action_masking = [1 for _ in range(len(self.get_action_space()))]
        
        for robot in self.robots:
            try:
                if robot.approaching_w1 == True:
                    valid_waypoints = self.getValidWaypoints(robot)
                    self.previous_valid_waypoints[robot._id_num] = valid_waypoints
                    self.action_masking.append(valid_waypoints)
                else:
                    self.action_masking.append(self.previous_valid_waypoints[robot._id_num])
            except:
                valid_waypoints = self.getValidWaypoints(robot)
                self.previous_valid_waypoints[robot._id_num] = valid_waypoints
                self.action_masking.append(valid_waypoints)

        payload = {
            'previous_rl_state': self.previous_rl_state,
            'action_masking': self.action_masking,
            'previous_rl_map_state': self.previous_rl_map_state,
            'deterministic': self.deterministic_rl,
            }
        
        response = requests.post(model_predict_url, json=self.to_serializable(payload)).json()

        actions = response["actions"][0]
        log_pis = response["log_pis"]
        value = response["value"]
  
        total_rewards = 0
        action_arr = []
        new_waypoints = []
        rewards_arr = []

        for i, robot in enumerate(self.robots):
            if robot.initial_w1:
                if robot.approaching_w1:
                    w1 = self.actionToWaypoint(robot, actions[robot._id - 1])
                    robot.last_action = actions[robot._id - 1]
                    robot.w1 = w1
                else:
                    robot.last_action = actions[robot._id - 1]
                    w1 = robot.w1
            else:
                w1 = robot.destination
                robot.last_action = 8
                robot.w1 = w1
            new_waypoints.append(w1)
            
            action_arr.append(robot.last_action)
            reward = self.calculateReward(w1, robot)
            # reward = self.reshapeReward(reward, robot)
            rewards_arr.append(reward)
            total_rewards += reward
        return new_waypoints, action_arr, rewards_arr, log_pis, value

    def actionToWaypoint(self, robot: Robot, action):
        pos_key = (round(robot.pos_x), round(robot.pos_y))
        fallback_key = (round(robot.pos_x - 1), round(robot.pos_y - 1))

        if pos_key in self.action_mapping_inv:
            coor_x, coor_y = self.action_mapping_inv[pos_key]
        elif fallback_key in self.action_mapping_inv:
            coor_x, coor_y = self.action_mapping_inv[fallback_key]
        else:
            # Fallback: use robot's current position directly (or raise an error/log warning)
            coor_x, coor_y = round(robot.pos_x), round(robot.pos_y)

        action_dict = {
            0: (coor_x, coor_y + 1),  # Up
            1: (coor_x + 1, coor_y),  # Right
            2: (coor_x + 2, coor_y),  # Down
            3: (coor_x + 1, coor_y - 1),  # Left
            4: (coor_x, coor_y - 1),
            5: (coor_x - 1, coor_y - 1),
            6: (coor_x - 2, coor_y),
            7: (coor_x - 1, coor_y),
            8: (robot.destination.x, robot.destination.y)  # Destination
        }

        new_action = action_dict.get(action, (robot.destination.x, robot.destination.y))
        
        if action == 8:
            return (robot.destination.x, robot.destination.y)

        return self.action_mapping.get(new_action, (robot.destination.x, robot.destination.y))


    def getValidWaypoints(self, robot: Robot):
        pos_key = (round(robot.pos_x), round(robot.pos_y))
        des_key = (round(robot.destination.x), round(robot.destination.y))
        fallback_key = (round(robot.pos_x - 1), round(robot.pos_y - 1))

        if pos_key in self.action_mapping_inv:
            coor_x, coor_y = self.action_mapping_inv[pos_key]
        elif fallback_key in self.action_mapping_inv:
            coor_x, coor_y = self.action_mapping_inv[fallback_key]
        else:
            # Fallback: use robot's current position directly (or raise/log)
            coor_x, coor_y = round(robot.pos_x), round(robot.pos_y)

        action_dict = {
            0: (coor_x, coor_y + 1),
            1: (coor_x + 1, coor_y),
            2: (coor_x + 2, coor_y),
            3: (coor_x + 1, coor_y - 1),
            4: (coor_x, coor_y - 1),
            5: (coor_x - 1, coor_y - 1),
            6: (coor_x - 2, coor_y),
            7: (coor_x - 1, coor_y),
            8: (robot.destination.x, robot.destination.y)  # Destination
        }

        current_detour_budget = self.detour_budget_list[robot._id_num]
        action_masking = []
        for key in action_dict.values():
            action_masking.append(1 if key in self.action_mapping else 0)
        action_masking[-1] = 0

        dist = np.linalg.norm(np.array(pos_key) - np.array(des_key))

        if dist <= 15:
            action_masking[-1] = 1
        
        # action_masking = [0 for _ in range(len(action_dict.keys()))]
        # action_masking[-1] = 1
        # if current_detour_budget != 0:
        #     action_masking = []
        #     for key in action_dict.values():
        #         action_masking.append(1 if key in self.action_mapping else 0)

        #     action_masking[-1] = 1
        #     if robot.last_action != 8:
        #         self.detour_budget_list[robot._id_num] -= 1
        #     robot.detour_budget = self.detour_budget_list[robot._id_num]
        
        return action_masking


    def calculateReward(self, w1, robot: Robot, c1=0.03, c2=0.02, c3=5.0, c4=0.01, c5=0.001):
        """
        Calculate shaped reward considering:
        - Task completion (sparse reward)
        - Energy efficiency (continuous penalty)
        - Detour-avoidable traffic congestion (continuous penalty)
        - Time efficiency (continuous penalty)
        
        Args:
            w1: Weight parameter (if needed for specific calculations)
            robot: Robot object
            c1: Energy penalty weight
            c2: Traffic congestion penalty weight  
            c3: Task completion reward weight
            c4: Time penalty weight
        """
        reward = 0.0

        try:
            current_state = self.current_state[robot._id_num]
            previous_state = self.previous_state[robot._id_num]
            if current_state[7] not in ['idle', 'station_processing']:
                ticks_waiting = self.tick_for_finishing_task[robot._id_num]

                if current_state[7] != previous_state[7]:
                    self.tick_for_finishing_task[robot._id_num] = 0
                    self.handleDetourBudget(robot)
                    task_finished = 1.0
                else:
                    self.tick_for_finishing_task[robot._id_num] += 1
                    task_finished = 0.0

                # === SHAPED REWARD COMPONENTS ===
            
                # 1. Task Completion Reward (sparse)
                completion_reward = c3 * task_finished

                # 2. Energy Efficiency penalty
                energy_penalty = 0.0
                congestion_penalty = 0.0
                time_penalty = 0.0
                robot_position = (current_state[3], current_state[4])
                if 7 < current_state[3] * 49 < 42 and current_state[3] != current_state[5]:  # If not in station area
                    speed = current_state[1] * 1.5
                    acceleration = (current_state[2] * 2) - 1
                    energy = calculateEnergy(speed, acceleration)
                    energy_cost = (energy / 1000) * (ticks_waiting / 1000.0)
                    energy_penalty = min(c1, c1 * (energy_cost))

                    # 3. Detour-Avoidable Congestion Penalty (simplified)
                    congestion_penalty = c2 * self.getDetourAvoidableCongestion(robot_position, robot._id_num)
                    
                    # 4. Time Efficiency Penalty (continuous)
                    time_penalty = min(c4, c4 * (ticks_waiting / 1000.0))
                
                # 5. Movement Efficiency Reward (continuous)
                movement_reward = 0.0
                
                # Reward any movement (encourages action over inaction)
                current_pos = np.array([current_state[3], current_state[4]])
                previous_pos = np.array([previous_state[3], previous_state[4]])
                dest_pos = np.array([current_state[5], current_state[6]])

                movement_distance = np.linalg.norm(dest_pos - current_pos)
                previous_distance = np.linalg.norm(dest_pos - previous_pos)
                
                # Small reward for movement
                if movement_distance < previous_distance:
                    movement_reward = c5
                
                # === COMBINE ALL COMPONENTS ===
                reward = (completion_reward + movement_reward - 
                        energy_penalty - congestion_penalty - time_penalty)
                
                max_possible_reward = c3 + c5  # completion + max movement
                reward = reward / max_possible_reward

                return reward
            else:
                return reward
        
        except Exception as e:
            # print(f"Error in calculateReward: {e}")
            return 0.0

    def getDetourAvoidableCongestion(self, position, current_robot_id, detection_radius=3):
        """
        Calculate congestion penalty focused on situations where detours can help.
        Simple approach: penalize when multiple robots are clustered in a small area,
        as this indicates avoidable bottlenecks.
        
        Args:
            position: Current robot position (x, y)
            current_robot_id: ID of current robot
            detection_radius: Radius to check for nearby robots
            
        Returns:
            float: Congestion penalty (0 to 1)
        """
        x, y = position
        nearby_robots = 0
        
        # Count active robots within detection radius
        for robot_id, state in enumerate(self.current_state):
            if robot_id == current_robot_id or state[7] == 0:  # Skip self and inactive robots
                continue
                
            robot_x, robot_y = state[3], state[4]
            distance = math.sqrt((x - robot_x)**2 + (y - robot_y)**2)
            
            if distance <= detection_radius:
                nearby_robots += 1
        
        # Simple congestion penalty: more robots nearby = higher penalty
        # This captures situations where detours could spread out the traffic
        if nearby_robots == 0:
            return 0.0
        elif nearby_robots <= 2:
            return 0.1  # Light congestion
        elif nearby_robots <= 4:
            return 0.3  # Moderate congestion
        else:
            return 0.6  # Heavy congestion (definitely detour-worthy)
        
    def calculateSparseReward(self, w1, robot: Robot, c1=0.1, c2=0.3, c3=1):
            reward = 0.0
            try:
                current_state = self.current_state[robot._id_num]
                previous_state = self.previous_state[robot._id_num]

                ticks_waiting = self.tick_for_finishing_task[robot._id_num]

                if current_state[7] != previous_state[7]:
                    self.tick_for_finishing_task[robot._id_num] = 0
                    self.handleDetourBudget(robot)
                    task_finished = 1.0
                else:
                    self.tick_for_finishing_task[robot._id_num] += 1
                    task_finished = 0.0

                reward += c3 * task_finished

                reward = reward / (c3)

                if current_state[7] == 0:
                    reward = 0
            except Exception as e:
                pass

            return reward


    def reshapeReward(self, reward, robot: Robot, reward_shaping_mode = "pbrs"):
        try:
            current_state = self.current_state[robot._id_num]
            previous_state = self.previous_state[robot._id_num]

            previous_heuristic_value = unidirectionRMFSHeuristic(self.pod_path,
                (previous_state[3] * 49, previous_state[4] * 31), (previous_state[5] * 49, previous_state[6] * 31), robot=True)
            current_heuristic_value = unidirectionRMFSHeuristic(self.pod_path,
                (current_state[3] * 49, current_state[4] * 31), (current_state[5] * 49, current_state[6] * 31), robot=True)
            
            discount0 = self.gamma
            discount = self.gamma_n

            if reward_shaping_mode == "hurl":
                reward = reward + (discount0-discount)*(1/max(current_heuristic_value, 1))
            if reward_shaping_mode == "pbrs":
                reward = reward + discount0*(1/max(current_heuristic_value, 1)) - (1/max(previous_heuristic_value, 1))

        except:
            pass

        return reward
    
    def handleDetourBudget(self, robot: Robot, detour_budget_k = 3):
        self.detour_budget_list[robot._id_num] = detour_budget_k
        robot.detour_budget = detour_budget_k

    def get_action_space(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def log_reward(self, log_entry):
        log_file = os.path.join(os.path.dirname(self.model_path), ('log_'+self.current_time +'.jsonl'))

        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Append to in-memory buffer
        self.log_buffer.append(log_entry)

        # Flush to disk in batches
        if len(self.log_buffer) >= self.log_buffer_size:
            with open(log_file, 'a') as f:
                for entry in self.log_buffer:
                    f.write(json.dumps(entry) + '\n')
            self.log_buffer = []

    def read_log(self, train_log_path):
        with open(train_log_path, 'r') as f:
            logs = [json.loads(line) for line in f]
        return logs

    def to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()  # Convert NumPy scalar to native Python type
        elif isinstance(obj, dict):
            return {k: self.to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_serializable(v) for v in obj]
        else:
            return obj
