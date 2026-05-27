from __future__ import annotations
import os
from collections import Counter
from typing import Optional, List, TYPE_CHECKING, Dict
import time

import pandas as pd
import random
from lib.types.directed_graph import DirectedGraph
from world.layout import Layout
from world.landscape import Landscape
from lib.math import *
from lib.types.netlogo_coordinate import NetLogoCoordinate
from world.managers.intersection_manager import IntersectionManager
from world.managers.order_manager import OrderManager
from world.managers.zone_manager import ZoneManager
from world.managers.job_manager import JobManager
from world.managers.robot_manager import RobotManager
from world.managers.pod_manager import PodManager
from world.managers.area_path_manager import AreaPathManager
from world.managers.station_manager import StationManager
from world.managers.storage_manager import StorageManager
from world.entities.order import Order
from world.entities.pod import Pod
from world.entities.robot import Robot
from world.entities.job import Job
from lib.generator.order_generator import *
from lib.constant import *
from lib.math import *
from world.managers.adaptive_replenishment import *
if TYPE_CHECKING:
    from world.entities.object import Object
import requests

model_remember_url = "http://127.0.0.1:8100/remember"
model_getmemory_url = "http://127.0.0.1:8100/getmemory"
model_train_url = "http://127.0.0.1:8100/train"

class Warehouse:
    DIMENSION = 60
    def __init__(self):
        self._tick = 0
        self._step = 0
        self.empty_job_steps = 0
        self.ignored_types = ["station", "area_path", "intersection"]
        self.job_queue = []
        self.stop_and_go = 0
        self.total_energy = 0
        self.total_pod = 0
        self.total_turning = 0
        self.warehouse_size = []
        self.total_fixed_load_energy = 0  # Track total energy with fixed load mass
        self.total_pod = 0 # ga perlu sebetulnya?
        self.total_turning = 0
        self.replenishment_count = 0  # Counter for total SKUs replenished
        self.replenishment_trips = 0  # Counter for total pod visits for replenishment
        self.replenished_pods = {}  # Track pod visits: {pod_id: visit_count}
        self.pod_visit_to_station = 0  # Track pod visits count to picking station
        self.orders_fulfilled = 0  # Track order fulfillment count
        self.delivered_order_lines = 0  # Track fully satisfied order-SKU lines
        self.total_picked_units = 0  # Track total picked unit quantity
        self.average_inventory_level = 0  # Track average inventory level
        self.average_pod_inventory_level = 0  # Track average pod inventory level
        self.average_weighted_pod_utilization = 0  # Track average weighted pod utilization
        self.layout = Layout()

        landscape_dimension = self.DIMENSION
        generated_pod_path = PARENT_DIRECTORY + '/data/output/generated_pod.csv'
        try:
            generated_pod_df = pd.read_csv(generated_pod_path, header=None)
            rows, cols = generated_pod_df.shape
            landscape_dimension = max(self.DIMENSION, rows, cols) + 5
            self.pod_path = generated_pod_df.to_dict(orient='records')
            print("Warehouse: Loaded existing pod layout")
        except FileNotFoundError:
            print("Warehouse: Pod layout not found yet, will load after layout generation")
            self.pod_path = None

        self.landscape = Landscape(landscape_dimension)
        self.order_manager = OrderManager(self)
        self.zone_manager = ZoneManager(self)
        self.job_manager = JobManager(self)
        self.intersection_manager = IntersectionManager(self, self.landscape.current_date_string)
        self.area_path_manager = AreaPathManager(self)
        self.robot_manager = RobotManager(self)
        self.pod_manager = PodManager(self)
        self.storage_manager = StorageManager(self)
        self.station_manager = StationManager(self)
        self.next_process_tick = 0
        self.update_intersection_using_RL = False
        self.zoning = True
        self.assign_order_df = None

        self.station_manager = StationManager(self)
            
        self.next_process_tick = 0
        self.update_intersection_using_RL = False
        # self.zoning = False  # Flag to control whether to use zoning or not

        self.robot_using_RL = False
        self.rl_state = {}
        self.old_w1 = []
        self.new_w1 = []
        
        self.graph = DirectedGraph()
        self.graph_pod = DirectedGraph()
        self.updated_assigned_order = True
        
        self.adaptive_policy = AdaptiveReplenishmentPolicy()
        self.use_adaptive_replenishment = False  # Flag to control which policy to use
        self.global_critical_skus = set()
        self.global_watchlist_interval = int(os.getenv("RMFS_GLOBAL_WATCHLIST_TICKS", "300"))
        self.global_watchlist_threshold = float(os.getenv("RMFS_GLOBAL_WATCHLIST_THRESHOLD", "0.9"))
        self.pod_replenishment_threshold = float(os.getenv("RMFS_POD_REPLENISHMENT_THRESHOLD", "0.4"))
        self.last_watchlist_refresh_tick = -1
        self.last_hold_recheck_tick = -1
        
        self.sku_picking_queue = []  # Queue for SKUs that need to be picked
        self.queued_qty_by_order: Dict[int, Dict] = {}
        self.active_job_qty_by_order: Dict[int, Dict] = {}
        self.pending_replenishment_dispatches: List[Dict] = []
        self.replenishment_dispatch_aging_ticks = int(
            os.getenv("RMFS_REPLENISHMENT_AGING_TICKS", "300")
        )
        self.health_check_interval = int(os.getenv("RMFS_HEALTH_CHECK_INTERVAL", "0"))
        self.health_stall_ticks = int(os.getenv("RMFS_HEALTH_STALL_TICKS", "600"))
        self.persist_assign_order_csv = os.getenv(
            "RMFS_PERSIST_ASSIGN_ORDER_CSV",
            "0",
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self.last_health_check_tick = -1
        self.health_status = "healthy"
        self.last_health_status = "healthy"
        self.health_consistency_violations = 0
        self.health_zombie_order_count = 0
        self.health_pending_replenishment_count = 0
        self.health_aged_pending_replenishment_count = 0
        self.health_oldest_pending_replenishment_age = 0
        self.health_last_progress_tick = 0
        self.health_last_fulfillment_tick = 0
        self.health_last_pod_visit_tick = 0
        self.health_last_replenishment_tick = 0
        self.health_last_queue_change_tick = 0
        self.health_last_unfinished_decrease_tick = 0
        self.health_stalled_tick_count = 0
        self._health_prev_fulfilled_orders = 0
        self._health_prev_pod_visits = 0
        self._health_prev_replenishment_trips = 0
        self._health_prev_sku_queue_len = 0
        self._health_prev_unfinished_orders = 0
    
    def _increase_nested_quantity(self, store: Dict, order_id, sku, qty):
        if order_id is None or sku is None or qty <= 0:
            return

        order_bucket = store.get(order_id)
        if order_bucket is None:
            order_bucket = {}
            store[order_id] = order_bucket

        order_bucket[sku] = order_bucket.get(sku, 0) + qty

    def _decrease_nested_quantity(self, store: Dict, order_id, sku, qty):
        if order_id is None or sku is None or qty <= 0:
            return

        order_bucket = store.get(order_id)
        if not order_bucket:
            return

        remaining_qty = order_bucket.get(sku, 0) - qty
        if remaining_qty > 0:
            order_bucket[sku] = remaining_qty
        else:
            order_bucket.pop(sku, None)

        if not order_bucket:
            store.pop(order_id, None)

    def _enqueue_sku_request(self, order_id, sku, qty):
        if qty is None or qty <= 0:
            return None

        request = {"order_id": order_id, "sku": sku, "qty": qty}
        self.sku_picking_queue.append(request)
        self._increase_nested_quantity(self.queued_qty_by_order, order_id, sku, qty)
        return request

    def _rebuild_sku_picking_queue_index(self):
        rebuilt_index = {}
        for request in self.sku_picking_queue:
            self._increase_nested_quantity(
                rebuilt_index,
                request.get("order_id"),
                request.get("sku"),
                request.get("qty", 0),
            )
        self.queued_qty_by_order = rebuilt_index

    def _replace_sku_picking_queue(self, new_queue):
        self.sku_picking_queue = new_queue
        self._rebuild_sku_picking_queue_index()

    def _register_active_job_quantity(self, order_id, sku, qty):
        self._increase_nested_quantity(self.active_job_qty_by_order, order_id, sku, qty)

    def _release_active_job_quantity(self, order_id, sku, qty):
        self._decrease_nested_quantity(self.active_job_qty_by_order, order_id, sku, qty)

    def _add_picking_task_to_job(self, job: Job, order_id, sku, qty):
        job.addPickingTask(order_id, sku, qty)
        self._register_active_job_quantity(order_id, sku, qty)

    def _release_job_active_quantities(self, job: Job):
        for order_id, sku, qty in job.orders:
            self._release_active_job_quantity(order_id, sku, qty)

    
    def setAssignOrderData(self):
        file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        self.assign_order_df = pd.read_csv(
            file_path,
            dtype={
                "sequence_id": "int32",
                "order_id": "int32",
                "order_type": "int8",
                "item_id": "int32",
                "item_quantity": "int32",
                "order_arrival": "int64",
                "assigned_station": "object",
                "assigned_pod": "float32",
                "status": "int8"
            }
        )
        self.total_orders_expected = int(self.assign_order_df["order_id"].nunique())
        self.last_order_arrival = int(self.assign_order_df["order_arrival"].max())
        
    def initWarehouse(self):
        self.robot_manager.initRobotManager()
        self.station_manager.initStationManager()
        self.pod_manager.initPodManager()
        # self.pod_manager.loadPodsFromCSV() # CALLING NEW FUNCTION
        
        # area path and intersection entity don't need connection back to the managers
        # self.area_path_manager.initAreaPathManager()
        # self.intersection_manager.initIntersectionManager()
        
        # Robot RL
        self.intersection_coords = self.intersection_manager.getAllIntersectionCoordinates()
        self.action_coords = sorted(
            [(x - 3, y) for x, y in self.intersection_coords if x > 13] + 
            [(x, y - 1) for x, y in self.intersection_coords if y > 2 and x in [15, 27, 39]] + 
            [(x, y - 2) for x, y in self.intersection_coords if y > 2 and x in [21, 33]],
            key=lambda coord: coord[0]  # Sort by x-value
        )
        self.action_coor_dict = {coord : idx for idx, coord in enumerate(self.action_coords)}
        
        self.robot_action_map = {}
        for coords in self.action_coords:
            if coords[0] in [9, 15, 21, 27, 33]:
                x, y = coords
                self.robot_action_map[(x // 3, y // 3)] = coords
            else:
                x, y = coords
                self.robot_action_map[(x // 3, y // 3)] = coords

        self.robot_manager.action_mapping = self.robot_action_map
        self.robot_manager.action_mapping_inv = {v: k for k, v in self.robot_action_map.items()}
        print(self.robot_manager.action_mapping)

    def setWarehouseSize(self, size):
        self.warehouse_size = size

    def allRobotsIdle(self):
        return all(
            (robot.job is None or robot.job.is_finished) and robot.current_state == "idle"
            for robot in self.robot_manager.getAllRobots()
        )

    def isSimulationComplete(self):
        after_last_arrival = int(self._tick) >= int(getattr(self, "last_order_arrival", 0))
        all_orders_fulfilled = self.orders_fulfilled >= int(getattr(self, "total_orders_expected", 0))
        no_unfinished_orders = len(self.order_manager.unfinished_orders) == 0
        no_pending_work = (
            len(self.job_queue) == 0
            and len(self.sku_picking_queue) == 0
            and len(self.pending_replenishment_dispatches) == 0
        )

        return (
            after_last_arrival
            and all_orders_fulfilled
            and no_unfinished_orders
            and no_pending_work
            and self.allRobotsIdle()
        )

    def loadPodPath(self):
        """Load pod path after layout generation is complete"""
        if self.pod_path is None:
            try:
                self.pod_path = pd.read_csv(PARENT_DIRECTORY + '/data/output/generated_pod.csv', header=None
                                            ).to_dict(orient='records')
                print("      ✅ Warehouse: Pod layout loaded successfully")
            except FileNotFoundError:
                print("      ❌ Warehouse: Pod layout still not available")
                self.pod_path = []

    def getWarehouseSize(self):
        return self.warehouse_size

    def getObjects(self):
        result = []
        result.extend(self.area_path_manager.getAllAreaPaths())
        result.extend(self.intersection_manager.getAllIntersections())
        result.extend(self.pod_manager.getAllPods())
        result.extend(self.robot_manager.getAllRobots())
        result.extend(self.station_manager.getAllStations())
        return result
    
    def getMovableObjects(self):
        result = []
        for o in self.getObjects():
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)

        return result


    def tick(self):
        if int(self._tick) == self.next_process_tick:
            start_time = time.time()
            self.findNewOrders()
            self.processOrders() # hasilnya berupa  job queue
            if self.update_intersection_using_RL:
                self.intersection_manager.updateDirectionUsingDQN(int(self._tick))
            # print("Process orders:", time.time() - start_time)

        # Refresh the global watchlist and enqueue idle-pod replenishment
        # candidates before dispatching robots.
        current_tick = int(self._tick)

        if (
            current_tick > 0
            and current_tick % self.global_watchlist_interval == 0
            and current_tick != self.last_watchlist_refresh_tick
        ):
            self.update_global_sku_watchlist()
            self.checkAndTriggerProactiveReplenishment()
            self.last_watchlist_refresh_tick = current_tick
             
        if (
            current_tick > 0
            and current_tick % 500 == 0
            and current_tick != self.last_hold_recheck_tick
        ):
            self.recheck_on_hold_orders()
            self.last_hold_recheck_tick = current_tick

        # Give aged replenishment requests a chance to dispatch before
        # regular picking assignment. Non-aged requests wait until after
        # picking dispatch and use any remaining idle robots.
        self.refreshMandatoryReplenishmentPods()
        self.dispatchPendingReplenishmentRequests(prioritize_aged_only=True)

        # Baseline job assignment 

        # if len(self.job_queue) > 0:
        #     current_distance = 1000000
        #     nearest_robot: Optional[Robot] = None

        #     # Assign job to robot
        #     for o in self.getMovableObjects():
        #         if len(self.job_queue) > 0:
        #             job: Job = self.job_queue[0]

        #             if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
        #                 dist = calculateManhattanDistance([o.pos_x, o.pos_y], [job.pod_coordinate.x, job.pod_coordinate.y])
        #                 if dist < current_distance:
        #                     nearest_robot = o
        #                     current_distance = dist

        #     if nearest_robot is not None:
        #         # print(len(self.job_queue))
        #         job: Job = self.job_queue.pop(0)
        #         nearest_robot.assignJobAndSetToTakePod(job)
        #         job: Job = self.job_queue.pop(0)
            # /Assign job to robot

        # Rika's batch job assignment
        # self.assignJobsInBatches()
        self.assignJobsInBatches_MODIFIED()
        self.dispatchPendingReplenishmentRequests(prioritize_aged_only=False)
        
        if len(self.job_queue) == 0:
            self.empty_job_steps += 1
        else:
            self.empty_job_steps = 0

        # Ngitung energy + replenishment
        total_energy = 0
        total_fixed_load_energy = 0
        total_turning = 0
            
        robot_info  = []
        global_robot_positions = []
        robot_velocity = []
        idle_time = []
        robot_state = []
    
        # Process robot menyelesaikan job dari assign robot 

        # Generate RL State - Robot Information
        for o in self.getMovableObjects():
            if isinstance(o, Robot):
                global_robot_positions.append(f'{int(o.pos_x)},{int(o.pos_y)}')
                robot_velocity.append(o.velocity)
                
                robot_info.append([
                    # o.id,
                    o.heading / 360,
                    # o.shape,
                    o.velocity / 1.5,
                    (o.acceleration + 1 ) / 2,
                    int(o.pos_x) / 49,
                    int(o.pos_y) / 31,
                    o.destination.x / 49,
                    o.destination.y / 31,
                    o.color / 94,
                    o.w1.x / 49,
                    o.w1.y / 31,
                    o.approaching_w1,
                    o.idle_time/1200,
                    min(1, self.robot_manager.tick_for_finishing_task[o._id_num] / 400),
                    # o.detour_budget / 3,
                ])
                idle_time.append(o.idle_time)
                robot_state.append(o.current_state)
            
        if self.robot_manager.heuristic_rl:
            start_time = time.time()
            if self._step % self.robot_manager.frameskip == 0 and self._step > 0:
                self.robot_state, self.map_state = self.getRobotsState(robot_info, global_robot_positions, robot_velocity)

                new_waypoints, actions, rewards, log_pis, value = self.robot_manager.updateWaypoint()

                dones = 0
                if self._step == self.robot_manager.num_steps:
                    dones = 1
                    print(f"Episode Done for {self.robot_manager.num_steps} steps")
                    
                if self.empty_job_steps > 400:
                    dones = 1
                    print(f"Empty job for {self.empty_job_steps} steps")

                should_terminate = self.check_congestion_termination(idle_time, robot_state, rewards, robot_info)
                if should_terminate:
                    dones = 1
                
                if self._step < 4*50:
                    rewards = [0] * self.robot_manager.robot_counter

                payload = {
                    'previous_rl_state': self.robot_manager.previous_rl_state,
                    'actions': actions,
                    'rewards': rewards,
                    'rl_state': self.robot_state,
                    'dones': dones,
                    'log_pis': log_pis,
                    'value': value,
                    'previous_rl_map_state': self.robot_manager.previous_rl_map_state,
                    'rl_map_state': self.map_state,
                    'previous_action_masks': self.robot_manager.previous_action_masking,
                    'action_masks': self.robot_manager.action_masking,
                }

                
                remember_response = requests.post(model_remember_url, json=self.to_serializable(payload))

                # self.robot_manager.memory = requests.post(model_getmemory_url).json()

                if not self.robot_manager.deterministic_rl:
                    train_response = requests.post(model_train_url, json={"step":self._step})
                
                self.robot_manager.rl_done = dones

                self.robot_manager.previous_state = self.robot_manager.previous_rl_state
                self.robot_manager.current_state = self.robot_state

                self.robot_manager.previous_rl_state = self.robot_state
                self.robot_manager.previous_rl_map_state = self.map_state
                self.robot_manager.previous_action_masking = self.robot_manager.action_masking
            # RL timing debug intentionally disabled during experiment runs.

        for o in self.getMovableObjects():
            initial_velocity = o.velocity
            if isinstance(o, Robot):
                o.move()
                total_energy += o.energy_consumption
                total_fixed_load_energy += o.fixed_load_energy_consumption
                total_turning += o.turning
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1

                if o.job is not None and o.job.picking_delay == 0 and not o.job.is_finished:
                    need_replenish_pod = self.finishTaskInJob(o.job) # main function nyelesaiin job
                    self.trigger_bundled_proactive_replenishment(pod=o.job.pod, robot=o)
                    # if need_replenish_pod:
                    #     print(f"cihuy masuk")
                    #     pod: Pod = o.job.pod
                    #     station_replenish = self.station_manager.findAvailableReplenishmentStation()
                    #     new_job = self.job_manager.createJob(pod.coordinate, station_id=station_replenish.id, pod=pod)
                    #     new_job.addReplenishmentTask(pod)
                    #     o.assignJobAndSetToStation(new_job)
                        

                if o.current_state == 'idle' and o.job is not None:
                    released_pod = o.job.pod
                    if not self.tryGuaranteeReplenishmentOnRelease(released_pod, o):
                        self.pod_manager.setPodAvailable(released_pod)
                        o.job = None

        self.total_energy = total_energy
        self.total_fixed_load_energy = total_fixed_load_energy
        self.total_turning = total_turning
        # /Ngitung energy + replenishment

        if int(self._tick) == self.next_process_tick:
            self.next_process_tick += 1
            if self.update_intersection_using_RL:
                self.intersection_manager.updateModelAfterExecution(self._tick)

        current_tick = int(self._tick)
        if (
            self.health_check_interval > 0
            and current_tick % self.health_check_interval == 0
            and current_tick != self.last_health_check_tick
        ):
            self.refreshSimulationHealth()

        self._tick += TICK_TO_SECOND
        self._step += 1

    def finishTaskInJob(self, job: Job):
        job_station = self.station_manager.getStationById(job.station_id)
        if job_station.isPickerStation():
            return self.finishPickingTask(job) # concrete implementation cara menyelesaikan task nya
        elif job_station.isReplenishmentStation():
            return self.finishReplenishmentTask(job)
    
    def finishPickingTask(self, job: Job):
        pod: Pod = job.pod
        sku_need_replenished = []
        self._release_job_active_quantities(job)
        file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        for order_id, sku, quantity in job.orders:
            order: Order = self.order_manager.getOrderById(order_id)
            actual_picked = pod.pickSKU(sku, quantity)
            if actual_picked > 0:
                self.total_picked_units += int(actual_picked)
                line_was_complete = (
                    order.skus[sku]["quantity_delivered"] >= order.skus[sku]["total_quantity"]
                )
                order.deliverQuantity(sku, actual_picked)
                self.pod_manager.reduceSKUData(sku, actual_picked)
                line_is_complete = (
                    order.skus[sku]["quantity_delivered"] >= order.skus[sku]["total_quantity"]
                )
                if not line_was_complete and line_is_complete:
                    self.delivered_order_lines += 1

            remaining_qty = quantity - actual_picked
            if remaining_qty > 0:
                order.releaseCommittedQuantity(sku, remaining_qty)
                self._enqueue_sku_request(order_id, sku, remaining_qty)
                order.is_in_queue = True

            sku, replenished_status = self.pod_manager.isSKUNeedReplenishment(sku)

            # SKU Replenished Triggered
            if(replenished_status == True): sku_need_replenished.append(sku)

            sku_remaining = order.getQuantityLeftForSKU(sku)
            line_status = 1 if sku_remaining <= 0 else 0
            self.assign_order_df.loc[
                ((self.assign_order_df['order_id'] == order.id) & (self.assign_order_df['item_id'] == sku)),
                'status'
            ] = line_status
            if self.persist_assign_order_csv:
                self.assign_order_df.to_csv(file_path, index=False)
            self.updated_assigned_order = True

            if order.isOrderCompleted():
                self.order_manager.finishOrder(order_id, int(self._tick))
                station = self.station_manager.getStationById(order.station_id)
                if station is not None:
                    station.removeOrder(order_id, order)
                self.insertFinishedOrderToCSV(order)
                self.orders_fulfilled += 1

        if sku_need_replenished:
            self.global_critical_skus.update(sku_need_replenished)
    
    #     return None

    # def finishPickingTask(self, job: Job):
    #     pod: Pod = self.pod_manager.getPodsByCoordinate(job.pod_coordinate.x, job.pod_coordinate.y)
    #     UL = 0.6 #CHANGING PARAMETER
    #     KL = 0.6 #CHANGING PARAMETER

    #     # Increment pod visit to station
    #     self.pod_visit_to_station += 1

    #     # Process picking tasks
    #     for order_id, sku, quantity in job.orders:
    #         order: Order = self.order_manager.getOrderById(order_id)
    #         # order.deliverQuantity(sku, quantity)
    #         print("before tick", self._tick)
    #         print("order, sku, quantity :" ,order_id, sku, quantity)

    #         # pod.pickSKU(sku, quantity)

    #         # Check for SKU Replenishment
    #         # sku is sku_id (String)
    #         # self.pod_manager.reduceSKUData(sku, quantity)

    #         # Only pick up to available quantity in the pod
    #         actual_picked = pod.pickSKU(sku, quantity)  # Let pod handle its own inventory
    #         self.pod_manager.updateGlobalInventory(sku, actual_picked)  # Update global tracking
    #         order.deliverQuantity(sku, actual_picked)  # Only deliver what was actually picked

        # # TRACY
        # # Get pod that have SKU that need to be replenished
        # unique_sku_need_replenished = list(set(sku_need_replenished))
        # replenished_pod_needed_by_sku = self.pod_manager.getPodNeedReplenishment(unique_sku_need_replenished)
        # # Determine which pod will be Replenished
        # pod_id_will_be_replenished = self.pod_manager.determinePodWillBeReplenished(replenished_pod_needed_by_sku)
        # # Get the pod that will be Replenished
        # pod_will_be_replenished = self.pod_manager.getPodByNumber(pod_id_will_be_replenished)

        # Replenishment baseline
        job.is_finished = True
        if len(sku_need_replenished) > 0:
            return True
        need_replenish_pod = pod.isNeedReplenishment()
        return need_replenish_pod
    
    def finishReplenishmentTask(self, job: Job):
        pod: Pod = job.pod
        
        # Langsung baca 'daftar tugas' dari job. Gak perlu mikir/nebak lagi.
        skus_to_replenish = job.skus_for_replenishment
        
        skus_replenished = 0
        restored_quantities = {}
        if skus_to_replenish:
            before_qty = {
                sku_id: pod.skus[sku_id]['current_qty']
                for sku_id in skus_to_replenish
                if sku_id in pod.skus
            }
            # Suruh pod untuk mengisi HANYA SKU yang ada di daftar.
            skus_replenished = pod.replenishFlaggedSKUs(skus_to_replenish)
            for sku_id, old_qty in before_qty.items():
                new_qty = pod.skus[sku_id]['current_qty']
                restored_qty = max(0, new_qty - old_qty)
                if restored_qty > 0:
                    restored_quantities[sku_id] = restored_qty
                    self.pod_manager.increaseSKUData(sku_id, restored_qty)
                    _, still_below_reorder = self.pod_manager.isSKUNeedReplenishment(sku_id)
                    if still_below_reorder:
                        self.global_critical_skus.add(sku_id)
                    else:
                        self.global_critical_skus.discard(sku_id)

        # Update metrik
        self.replenishment_count += skus_replenished
        # Pastikan ini satu-satunya tempat counter trip di-update!
        self.replenishment_trips += 1
        pod.is_awaiting_replenishment = False
        pod.has_pending_replenishment_dispatch = False
        pod.must_replenish_before_pick = False
        if restored_quantities:
            self.recheck_on_hold_orders_for_skus(restored_quantities.keys())
        
        pod_number = pod.pod_number
        if pod_number in self.replenished_pods:
            self.replenished_pods[pod_number] += 1
        else:
            self.replenished_pods[pod_number] = 1
        
        # Panggil fungsi logging (pastikan ini versi upgrade yang bisa terima 'job')
        self.insertReplenishmentDataToCSV(job, skus_replenished)

        job.is_finished = True
        return False

    def insertFinishedOrderToCSV(self, order: Order):
        header = ["order_id", "order_arrival", "process_start_time", "order_complete_time", "station_id"]
        data = [order.id, order.order_arrival, order.process_start_time, order.order_complete_time,
                order.station_id]

        write_to_csv("order-finished.csv", header, data, self.landscape.current_date_string)

    def insertReplenishmentDataToCSV(self, job: Job, skus_replenished: int):
        """
        Menyimpan data operasi replenishment ke CSV.
        Sekarang menerima objek 'job' untuk data yang lebih lengkap.
        """
        # 1. "Bongkar" semua info yang kita butuhkan dari object 'job'
        pod_id = job.pod.pod_number
        replenishment_time = getattr(job, "replenishment_delay_total", job.replenishment_delay)
        station_id = job.station_id
        sku_list_str = str(job.skus_for_replenishment)

        # 2. Siapkan header CSV yang lengkap
        header = ["pod_id", "skus_replenished", "replenishment_time", "station_id", 
                "replenished_sku_list", "total_replenishment_count", "total_replenishment_trips", 
                "pod_visit_count", "current_tick"]
        
        pod_visit_count = self.replenished_pods.get(pod_id, 0)
        
        # 3. Susun datanya
        data = [
            pod_id,
            skus_replenished,
            replenishment_time,
            station_id,
            sku_list_str,
            self.replenishment_count,
            self.replenishment_trips,
            pod_visit_count,
            self._tick
        ]

        # 4. Tulis ke file
        # Asumsi lo punya fungsi helper global bernama write_to_csv
        write_to_csv("replenishment-operations.csv", header, data, self.landscape.current_date_string)

    def findNewOrders(self):
        if self.assign_order_df.empty:
            return
                  
        current_second = self.next_process_tick
        previous_second = (self.next_process_tick - 1)

        # new_orders = new_file_df

        # Filter orders that have arrived by the current second and have not been processed before
        new_orders = self.assign_order_df[(self.assign_order_df['order_arrival']<= current_second) & 
                               (self.assign_order_df['order_arrival'] > previous_second) &
                               (self.assign_order_df['status'] == -3)]
        
        grouped_orders = new_orders.groupby('order_id')

        # print(f"New job queue found: {len(new_orders)}")

        for order_id, group in grouped_orders:
            order_items = group[['item_id', 'item_quantity']].to_dict('records')
            order = self.order_manager.createOrder(order_id, current_second)

            # Add each item in the group to the order
            for item in order_items:
                order.addSKU(item['item_id'], item['item_quantity'])
        return new_orders

    def getActiveJobQuantitiesForOrder(self, order_id):
        return dict(self.active_job_qty_by_order.get(order_id, {}))

    def getQueuedQuantitiesForOrder(self, order_id):
        return dict(self.queued_qty_by_order.get(order_id, {}))

    def repairOrderDemandState(self, order: Order):
        if order is None or order.isOrderCompleted():
            return False

        if order.station_id is None:
            order.is_in_queue = False

        active_job_quantities = self.getActiveJobQuantitiesForOrder(order.id)
        queued_quantities = self.getQueuedQuantitiesForOrder(order.id)
        repaired = False

        for sku, details in order.skus.items():
            delivered_qty = details["quantity_delivered"]
            committed_qty = details["quantity_committed"]
            total_qty = details["total_quantity"]

            active_committed_qty = active_job_quantities.get(sku, 0)
            stranded_committed_qty = max(0, committed_qty - active_committed_qty)
            if stranded_committed_qty > 0:
                order.releaseCommittedQuantity(sku, stranded_committed_qty)
                committed_qty -= stranded_committed_qty
                repaired = True

            true_remaining_qty = max(0, total_qty - delivered_qty)
            queued_qty = queued_quantities.get(sku, 0)
            accounted_qty = active_committed_qty + queued_qty
            missing_qty = true_remaining_qty - accounted_qty

            if missing_qty > 0 and order.station_id is not None:
                self._enqueue_sku_request(order.id, sku, missing_qty)
                queued_quantities[sku] = queued_qty + missing_qty
                order.is_in_queue = True
                repaired = True

        return repaired

    def updateHealthProgressMarkers(self):
        current_tick = int(self._tick)
        progress_happened = False

        current_fulfilled_orders = int(self.orders_fulfilled)
        if current_fulfilled_orders > self._health_prev_fulfilled_orders:
            self.health_last_fulfillment_tick = current_tick
            progress_happened = True
        self._health_prev_fulfilled_orders = current_fulfilled_orders

        current_pod_visits = int(self.pod_visit_to_station)
        if current_pod_visits > self._health_prev_pod_visits:
            self.health_last_pod_visit_tick = current_tick
            progress_happened = True
        self._health_prev_pod_visits = current_pod_visits

        current_replenishment_trips = int(self.replenishment_trips)
        if current_replenishment_trips > self._health_prev_replenishment_trips:
            self.health_last_replenishment_tick = current_tick
            progress_happened = True
        self._health_prev_replenishment_trips = current_replenishment_trips

        current_sku_queue_len = int(len(self.sku_picking_queue))
        if current_sku_queue_len != self._health_prev_sku_queue_len:
            self.health_last_queue_change_tick = current_tick
            progress_happened = True
        self._health_prev_sku_queue_len = current_sku_queue_len

        current_unfinished_orders = int(len(self.order_manager.unfinished_orders))
        if current_unfinished_orders < self._health_prev_unfinished_orders:
            self.health_last_unfinished_decrease_tick = current_tick
            progress_happened = True
        self._health_prev_unfinished_orders = current_unfinished_orders

        if progress_happened:
            self.health_last_progress_tick = current_tick

    def countHealthConsistencyViolations(self):
        violations = 0
        zombie_orders = 0

        for order in self.order_manager.orders:
            true_remaining_total = 0
            for details in order.skus.values():
                total_qty = details["total_quantity"]
                delivered_qty = details["quantity_delivered"]
                committed_qty = details["quantity_committed"]

                if delivered_qty < 0 or committed_qty < 0:
                    violations += 1
                if delivered_qty > total_qty or committed_qty > total_qty:
                    violations += 1
                if delivered_qty + committed_qty > total_qty:
                    violations += 1

                true_remaining_total += max(0, total_qty - delivered_qty)

            if (
                order in self.order_manager.unfinished_orders
                and not order.isOrderCompleted()
                and not order.getRemainingSKU()
                and true_remaining_total > 0
            ):
                active_job_qty = sum(self.getActiveJobQuantitiesForOrder(order.id).values())
                queued_qty = sum(self.getQueuedQuantitiesForOrder(order.id).values())
                if active_job_qty + queued_qty == 0:
                    zombie_orders += 1

        return violations, zombie_orders

    def getPendingReplenishmentHealth(self):
        current_tick = int(self._tick)
        pending_count = len(self.pending_replenishment_dispatches)
        if pending_count == 0:
            return 0, 0, 0

        oldest_age = 0
        aged_count = 0
        for request in self.pending_replenishment_dispatches:
            age = max(0, current_tick - int(request.get("created_tick", current_tick)))
            if age > oldest_age:
                oldest_age = age
            if age >= self.replenishment_dispatch_aging_ticks:
                aged_count += 1

        return pending_count, aged_count, oldest_age

    @staticmethod
    def getHealthSeverity(status: str) -> int:
        severity = {
            "healthy": 0,
            "warning": 1,
            "stalled": 2,
        }
        return severity.get(status, 0)

    def buildHealthSnapshot(self):
        current_tick = int(self._tick)
        progress_gap = max(0, current_tick - self.health_last_progress_tick)

        return {
            "tick": current_tick,
            "fulfilled_orders": int(self.orders_fulfilled),
            "unfinished_orders": int(len(self.order_manager.unfinished_orders)),
            "job_queue_length": int(len(self.job_queue)),
            "sku_queue_length": int(len(self.sku_picking_queue)),
            "pod_visits": int(self.pod_visit_to_station),
            "replenishment_trips": int(self.replenishment_trips),
            "pending_replenishment_count": int(self.health_pending_replenishment_count),
            "aged_pending_replenishment_count": int(self.health_aged_pending_replenishment_count),
            "oldest_pending_replenishment_age": int(self.health_oldest_pending_replenishment_age),
            "consistency_violations": int(self.health_consistency_violations),
            "zombie_orders": int(self.health_zombie_order_count),
            "last_progress_tick": int(self.health_last_progress_tick),
            "progress_gap": int(progress_gap),
            "status": self.health_status,
        }

    def writeHealthSnapshot(self, snapshot):
        header = [
            "tick",
            "fulfilled_orders",
            "unfinished_orders",
            "job_queue_length",
            "sku_queue_length",
            "pod_visits",
            "replenishment_trips",
            "pending_replenishment_count",
            "aged_pending_replenishment_count",
            "oldest_pending_replenishment_age",
            "consistency_violations",
            "zombie_orders",
            "last_progress_tick",
            "progress_gap",
            "status",
        ]
        data = [snapshot[column] for column in header]
        write_to_csv(
            "simulation-health.csv",
            header,
            data,
            self.landscape.current_date_string,
        )

    def emitHealthStatusIfNeeded(self, snapshot):
        current_severity = self.getHealthSeverity(snapshot["status"])
        previous_severity = self.getHealthSeverity(self.last_health_status)

        if (
            current_severity > previous_severity
            or (
                snapshot["status"] != self.last_health_status
                and snapshot["status"] != "healthy"
            )
        ):
            print(
                "SIM_HEALTH "
                f"tick={snapshot['tick']} "
                f"status={snapshot['status']} "
                f"fulfilled={snapshot['fulfilled_orders']} "
                f"unfinished={snapshot['unfinished_orders']} "
                f"sku_queue={snapshot['sku_queue_length']} "
                f"pending_repl={snapshot['pending_replenishment_count']} "
                f"oldest_repl_age={snapshot['oldest_pending_replenishment_age']} "
                f"zombies={snapshot['zombie_orders']} "
                f"violations={snapshot['consistency_violations']}"
            )

        self.last_health_status = snapshot["status"]

    def refreshSimulationHealth(self, force_log: bool = False):
        self.updateHealthProgressMarkers()

        violations, zombie_orders = self.countHealthConsistencyViolations()
        pending_count, aged_count, oldest_age = self.getPendingReplenishmentHealth()

        self.health_consistency_violations = violations
        self.health_zombie_order_count = zombie_orders
        self.health_pending_replenishment_count = pending_count
        self.health_aged_pending_replenishment_count = aged_count
        self.health_oldest_pending_replenishment_age = oldest_age

        current_tick = int(self._tick)
        progress_gap = max(0, current_tick - self.health_last_progress_tick)
        unfinished_orders = len(self.order_manager.unfinished_orders)
        after_last_arrival = current_tick >= int(getattr(self, "last_order_arrival", 0))

        status = "healthy"
        if violations > 0:
            status = "stalled"
        elif zombie_orders > 0:
            status = "warning"

        if (
            status != "stalled"
            and unfinished_orders > 0
            and progress_gap >= self.health_stall_ticks
            and (
                after_last_arrival
                or len(self.sku_picking_queue) > 0
                or len(self.pending_replenishment_dispatches) > 0
            )
        ):
            status = "stalled"

        if (
            status == "healthy"
            and oldest_age >= max(self.replenishment_dispatch_aging_ticks * 2, self.health_check_interval)
        ):
            status = "warning"

        self.health_status = status
        if status == "stalled":
            self.health_stalled_tick_count += 1

        snapshot = self.buildHealthSnapshot()
        if force_log or snapshot["tick"] != self.last_health_check_tick:
            self.writeHealthSnapshot(snapshot)
            self.emitHealthStatusIfNeeded(snapshot)
            self.last_health_check_tick = snapshot["tick"]

        return snapshot

    def processOrders(self):
        # Loop pada setiap order yang belum selesai
        orders_to_process = [o for o in self.order_manager.unfinished_orders if not o.on_hold]
        for order in orders_to_process:
            self.repairOrderDemandState(order)
            can_be_fulfilled, insufficient_skus = self.canFulfillOrder(order)
        
            if not can_be_fulfilled:
                # Jika tidak bisa, tandai sebagai on_hold dan skip
                # print(f"WARNING: Order {order.id} put ON HOLD due to insufficient SKUs: {insufficient_skus}")
                order.on_hold = True
                continue

            # 1. Tetap assign stasiun jika belum ada
            if order.station_id is None:
                available_station = self.station_manager.findHighestSimilarityStation(order.skus, self.pod_manager)
                if available_station:
                    order.assignStation(available_station.id)
                    available_station.addOrder(order.id, order)
                    # Update status di DataFrame pusat
                    self.assign_order_df.loc[self.assign_order_df['order_id'] == order.id, 'assigned_station'] = available_station.id
                    self.assign_order_df.loc[self.assign_order_df['order_id'] == order.id, 'status'] = -1
                else:
                    # Jika tidak ada stasiun tersedia, skip order ini untuk sementara
                    continue
            
            # 2. Saring order yang mustahil dikerjakan (stok=0)
            # can_be_fulfilled, insufficient_skus = self.canFulfillOrder(order)
            # if not can_be_fulfilled:
            #     # print(f"WARNING: Order {order.id} on hold, SKU {insufficient_skus} out of stock.")
            #     continue # Lanjut ke order berikutnya

            # 3. Masukkan semua item dari order yang valid ke 'Papan Pesanan'
            # Hanya jika order ini belum pernah dimasukkan ke antrian sebelumnya
            if not order.is_in_queue:
                for sku, details in order.getRemainingSKU().items():
                    if details > 0: # Pastikan hanya minta item yang masih dibutuhkan
                        self._enqueue_sku_request(order.id, sku, details)
                
                # Tandai order ini agar tidak dimasukkan ke antrian lagi
                order.is_in_queue = True 

        # TIDAK ADA LAGI LOGIKA getAvailablePod ATAU createJob DI SINI. TUGASNYA SELESAI.

    # =========================================== OLD CODE BROW ANJAY ============================================
    # def processOrders(self):
    #     file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
    #     robots_location = []
    #     unfulfillable_orders = []  # Track orders that cannot be fulfilled

    #     for o in self.getMovableObjects():
    #         if len(self.job_queue) > 0:
    #             job: Job = self.job_queue[0]

    #             if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
    #                 robots_location.append([o.pos_x, o.pos_y])

    #     for order in self.order_manager.unfinished_orders[:]:  # Copy to avoid modification during iteration

    #         # if self._tick >= 4318: # Kita cuma peduli deket-deket titik crash
    #         #     print(f"--- Tick: {self._tick}, Processing Order ID: {order.id}, Station ID: {order.station_id} ---")
    #         if order.station_id is None: # assign order ke stasiun kalau dalam tick tersebut belom di-assign samsek ke stasiun.
    #             # available_station = self.station_manager.findAvailablePickingStation()
    #             available_station = self.station_manager.findHighestSimilarityStation(order.skus, self.pod_manager)              
                 
    #             if available_station is not None:
    #                 order.assignStation(available_station.id)
    #                 available_station.addOrder(order.id, order)

    #                 try:
    #                     self.assign_order_df.loc[self.assign_order_df['order_id'] == order.id, 'assigned_station'] = available_station.id
    #                     self.assign_order_df.loc[self.assign_order_df['order_id'] == order.id, 'status'] = -1

    #                     self.assign_order_df.to_csv(file_path, index=False)
    #                     self.updated_assigned_order = True
    #                 except Exception as e:
    #                     print(f"Error updating assign_order_df for order {order.id}: {e}")
    #                     # Continue without updating CSV if there's an error
    #                     pass
    #             else:
    #                 continue

    #         if order.process_start_time <= 0:
    #             order.startProcessing(int(self._tick))
            
    #         # Get the station assigned to this order and orders in that station
    #         order_station = self.station_manager.getStationById(order.station_id)
    #         orders_in_station = order_station.getOrdersInStation()

    #         # For Emily {A:10, B:5, C:12}
    #         skus_in_station = order_station.getSKUsInStation()

    #         # For Jhen {A:[5,5], B:[5], C:[3,4,5]}
    #         skus_in_station_dict = order_station.getSKUsInStationDict()
            
    #         station_coordinate = order_station.coordinate

    #         # print("order sku :", order.getRemainingSKU())

    #         for sku in order.getRemainingSKU():
    #             # print("masuk remaining sku bosq")
    #              # This is the baseline
    #             available_pod: Pod = self.pod_manager.getAvailablePod(sku) 
                
    #             if available_pod is None:
    #                 # print(f"No available pod for SKU {sku} in order {order.id}")
    #                 continue

    #             quantity_to_take = order.getQuantityLeftForSKU(sku)
    #             order.commitQuantity(sku, quantity_to_take)

    #             # Commiting every order that has the sku in the pod chosen
    #             available_pod.pickSKU(sku, quantity_to_take)
                
    #              # Append pod to station
    #             order_station.addPod(available_pod.pod_number)
    #             available_pod.station = order_station

    #             self.assign_order_df.loc[((self.assign_order_df['order_id'] == order.id) & (self.assign_order_df['item_id'] == sku)), 'assigned_pod'] = int(available_pod.pod_number)
                
    #             self.assign_order_df.loc[((self.assign_order_df['order_id'] == order.id) & (self.assign_order_df['item_id'] == sku)), 'status'] = 0

    #             try:
    #                 self.assign_order_df.to_csv(file_path, index=False)
    #                 self.updated_assigned_order = True
    #             except Exception as e:
    #                 print(f"Error saving assign_order_df for order {order.id}, sku {sku}: {e}")
    #                 pass

    #             job = self.job_manager.createJob(available_pod.coordinate, station_id=order.station_id, pod=available_pod)
    #             self.pod_manager.setPodNotAvailable(available_pod)                # print(f"sku {sku} quantity {quantity_to_take}")

    #             job.addPickingTask(order.id, sku, quantity_to_take) # Simple kan disini ya beb
    #             pod_skus = [i for i in available_pod.skus]               


    #             self.job_queue.append(job) #ini yang penting
    
    
    def recheck_on_hold_orders(self):
        # print("INFO: Re-checking on-hold orders...")
        on_hold_orders = [o for o in self.order_manager.unfinished_orders if o.on_hold]
        for order in on_hold_orders:
            can_be_fulfilled, _ = self.canFulfillOrder(order)
            if can_be_fulfilled:
                print(f"Order {order.id} is now fulfillable! Removing from hold.")
                order.on_hold = False

    def recheck_on_hold_orders_for_skus(self, replenished_skus):
        target_skus = {sku for sku in replenished_skus if sku}
        if not target_skus:
            return 0

        released_orders = 0
        on_hold_orders = [o for o in self.order_manager.unfinished_orders if o.on_hold]
        for order in on_hold_orders:
            remaining_skus = order.getRemainingSKU()
            if not remaining_skus:
                continue
            if not any(sku in remaining_skus for sku in target_skus):
                continue

            can_be_fulfilled, _ = self.canFulfillOrder(order)
            if can_be_fulfilled:
                print(
                    f"Order {order.id} is now fulfillable after replenishment! Removing from hold."
                )
                order.on_hold = False
                released_orders += 1

        return released_orders

    def canFulfillOrder(self, order):
        """
        Check if order can be fulfilled with current global inventory.
        Returns (can_fulfill: bool, insufficient_skus: list)
        
        Edge cases handled:
        - Empty order
        - SKUs not in any pod
        - Zero quantities
        - Pod availability issues
        """
        if not order or not order.getRemainingSKU():
            return True, []  # Empty order is fulfillable
            
        insufficient_skus = []
        
        for sku, required_qty in order.getRemainingSKU().items():
            if required_qty <= 0:
                print(f"Order {order.id}: SKU {sku} has zero or negative quantity requirement, skipping")
                continue  # Skip zero or negative quantity requirements
                
            # Get all pods containing this SKU
            pods_with_sku = self.pod_manager.getPodsBySKU(sku)
            if not pods_with_sku:
                insufficient_skus.append(sku)
                print(f"Order {order.id}: SKU {sku} not found in any pod")
                continue
                
            # Calculate total available quantity across all pods
            total_available = 0
            for pod in pods_with_sku:
                if pod and sku in pod.skus:
                    current_qty = pod.skus[sku].get('current_qty', 0)
                    if current_qty > 0:
                        total_available += current_qty
            
            # Check if total available meets requirement
            if total_available < required_qty:
                insufficient_skus.append(sku)
                # print(f"Order {order.id}: Insufficient {sku} - need {required_qty}, have {total_available}")
        
        can_fulfill = len(insufficient_skus) == 0
        return can_fulfill, insufficient_skus


    def checkAndTriggerProactiveReplenishment(self):
        """
        Periodically sweep globally critical SKUs and assign at most one
        replenishment request to the single most depleted eligible pod for
        each SKU.
        """
        if not self.global_critical_skus:
            return

        for sku_id in sorted(self.global_critical_skus):
            self.enqueueBestReplenishmentDispatchForSKU(sku_id)

    def enqueuePendingReplenishmentDispatch(
        self,
        pod: Pod,
        skus_to_replenish: List[str],
    ):
        if pod is None or not skus_to_replenish:
            return False
        if pod.is_awaiting_replenishment:
            return False

        urgency_level = self.getReplenishmentUrgencyLevel(skus_to_replenish)
        existing_request = self.getPendingReplenishmentDispatch(pod.pod_number)
        if existing_request is not None:
            merged_skus = sorted(
                set(existing_request.get("skus_to_replenish", []))
                .union(skus_to_replenish)
            )
            merged_urgency = max(
                int(existing_request.get("urgency_level", 0)),
                self.getReplenishmentUrgencyLevel(merged_skus),
            )
            existing_request["skus_to_replenish"] = merged_skus
            existing_request["urgency_level"] = merged_urgency
            existing_request["guaranteed_on_release"] = (
                bool(existing_request.get("guaranteed_on_release", False))
                or merged_urgency > 0
            )
            pod.has_pending_replenishment_dispatch = True
            if bool(existing_request.get("guaranteed_on_release", False)):
                pod.must_replenish_before_pick = True
            return False

        self.pending_replenishment_dispatches.append(
            {
                "pod_number": int(pod.pod_number),
                "skus_to_replenish": list(skus_to_replenish),
                "created_tick": int(self._tick),
                "urgency_level": int(urgency_level),
                "guaranteed_on_release": bool(urgency_level > 0),
            }
        )
        pod.has_pending_replenishment_dispatch = True
        if urgency_level > 0:
            pod.must_replenish_before_pick = True
        return True

    def removePendingReplenishmentDispatch(self, pod_number: int):
        removed = False
        remaining_requests = []
        for request in self.pending_replenishment_dispatches:
            if int(request["pod_number"]) == int(pod_number):
                removed = True
                continue
            remaining_requests.append(request)

        self.pending_replenishment_dispatches = remaining_requests
        try:
            pod = self.pod_manager.getPodByNumber(int(pod_number))
        except (IndexError, TypeError):
            pod = None
        if pod is not None:
            pod.has_pending_replenishment_dispatch = False
            pod.must_replenish_before_pick = False
        return removed

    def dispatchPendingReplenishmentRequests(self, prioritize_aged_only: bool = False):
        if not self.pending_replenishment_dispatches:
            return 0

        dispatched_count = 0
        current_tick = int(self._tick)
        pending_requests = sorted(
            list(self.pending_replenishment_dispatches),
            key=lambda request: (
                0 if self.shouldGuaranteeReplenishmentRequest(request, current_tick) else 1,
                -int(request.get("urgency_level", 0)),
                -(current_tick - int(request.get("created_tick", current_tick))),
                int(request.get("pod_number", 0)),
            ),
        )

        for request in pending_requests:
            if (
                prioritize_aged_only
                and not self.shouldGuaranteeReplenishmentRequest(request, current_tick)
            ):
                continue

            station = self.station_manager.findAvailableReplenishmentStation()
            if station is None:
                break

            try:
                pod = self.pod_manager.getPodByNumber(int(request["pod_number"]))
            except (IndexError, TypeError):
                pod = None
            if pod is None:
                self.removePendingReplenishmentDispatch(int(request["pod_number"]))
                continue

            if pod.is_awaiting_replenishment:
                self.removePendingReplenishmentDispatch(pod.pod_number)
                continue

            if not pod.is_idle:
                continue

            skus_to_replenish, _ = self.get_replenishment_skus_for_pod(pod)
            if not skus_to_replenish:
                self.removePendingReplenishmentDispatch(pod.pod_number)
                continue

            robot = self.robot_manager.findNearestAvailableRobot(pod.coordinate)
            if robot is None:
                break

            success = self.sendPodForReplenishment(
                pod,
                station,
                skus_to_replenish,
                robot,
            )
            if success:
                dispatched_count += 1

        return dispatched_count

    def sendPodForReplenishment(
        self,
        pod,
        station,
        skus_to_replenish: Optional[List[str]] = None,
        robot: Optional[Robot] = None,
    ):
        """
        Send a specific pod to replenishment station.
        
        Edge cases handled:
        - Pod already assigned to job
        - Pod not available
        - Station capacity issues
        - Robot availability
        """
        if pod is None or station is None:
            return False
        if pod.is_awaiting_replenishment:
            return False
            
        # Check if pod is already busy
        # if not pod.is_idle:
        #     print(f"Pod {pod.pod_number} is already busy, cannot send for replenishment")
        #     return False
            
        # Check if pod is at a valid location
        if not hasattr(pod, 'coordinate') or pod.coordinate is None:
            print(f"Pod {pod.pod_number} has invalid coordinate")
            return False

        if skus_to_replenish is None:
            skus_to_replenish = []

        if robot is None:
            robot = self.robot_manager.findNearestAvailableRobot(pod.coordinate)
            if robot is None:
                print(f"No available robot found for replenishment job on pod {pod.pod_number}")
                return False
            
        try:
            # Create replenishment job
            new_job = self.job_manager.createJob(
                pod.coordinate,
                station_id=station.id,
                pod=pod,
                skus_for_replenishment=skus_to_replenish,
            )
            new_job.addReplenishmentTask(pod, skus_to_replenish)
            
            # Find available robot to handle the job
            # nearest_robot = self.robot_manager.findNearestAvailableRobot(pod.coordinate)
            # if nearest_robot is None:
            #     # print(f"No available robot found for replenishment job")
            #     return False
                
            # Assign job to robot
            robot.assignJobAndSetToStation(new_job)
            self.pod_manager.setPodNotAvailable(pod)
            self.removePendingReplenishmentDispatch(pod.pod_number)
            pod.is_awaiting_replenishment = True
            pod.has_pending_replenishment_dispatch = False
            pod.must_replenish_before_pick = False
            
            # Track replenishment metrics
            # self.replenishment_trips += 1
            
            print(f"Successfully created replenishment job for pod {pod.pod_number}")
            return True
            
        except Exception as e:
            print(f"Error creating replenishment job for pod {pod.pod_number}: {e}")
            return False

    def generateResult(self):
        result = []
        for o in self.getMovableObjects():
            result.append({
                'id': o.id,
                'heading': o.heading,
                'shape': o.shape,
                'velocity': o.velocity,
                'acceleration': o.acceleration,
                'pos_x': o.pos_x,
                'pos_y': o.pos_y,
                'color': o.color,
            })

        return result
    
    def getRobotsState(self, robot_info, global_robot_positions, robot_velocity):
        rl_state = {}

        global pod_path
        global pod_path_array

        if pod_path is None:
            pod_path = pd.read_csv(PARENT_DIRECTORY + '/data/output/generated_pod.csv', header=None
                                        ).to_dict(orient='records')
            pod_path_array = np.array(pd.read_csv(PARENT_DIRECTORY + '/data/output/generated_pod.csv', header=None))

        # Generate RL State - Path Information
        global_congestion_map = np.zeros((49, 31))
        robot_num = 0
        for o in self.getMovableObjects():
            if isinstance(o, Robot):
                norm_distance = 1
                destination_flag = 0
                if o.current_state in ["delivering_pod", "returning_pod"]:
                    current_pos = (round(o.pos_x), round(o.pos_y))
                    current_dest = (round(o.destination.x), round(o.destination.y))
                    norm_distance = np.linalg.norm(np.array(current_pos) - np.array(current_dest))

                    # If no initial waypoint assigned, assign w1 if robot is on a valid action point
                    if not o.initial_w1:
                        if self.robot_manager.action_mapping_inv.get(current_pos) is not None:
                            o.approaching_w1 = True
                            o.initial_w1 = True

                    # If w1 is assigned and robot reaches it, prepare to reroute to destination
                    if o.initial_w1:
                        w1_pos = (round(o.w1.x), round(o.w1.y))
                        dest_pos = (round(o.destination.x), round(o.destination.y))
                        
                        if self.robot_manager.action_mapping_inv.get(current_pos) is not None and current_pos != o.previous_w1:
                            o.previous_w1 = current_pos
                            o.approaching_w1 = True
                            
                num_of_robots = 0
                collective_velocity = 0
                ideal_velocity = 0

                robot_positions_dict = {tuple(pos): idx for idx, pos in enumerate(global_robot_positions)}
                for coordinate_sequence, coordinate in enumerate(o.path):
                    robot_id = robot_positions_dict.get(tuple(coordinate))
                    if robot_id is not None:
                        collective_velocity += ((robot_velocity[robot_id] - 1.5) / (coordinate_sequence + 1))
                        ideal_velocity += (1.5 / (coordinate_sequence + 1))
                        num_of_robots += 1
                path_velocity = (ideal_velocity + collective_velocity) / (ideal_velocity + 1e-3)
                manhattan_distance, _ = directionConstrainedManhattanDistance(pod_path,
                    [int(o.pos_x), int(o.pos_y)], [int(o.w1.x), int(o.w1.y)])
                estimated_time = manhattan_distance / max(0.1, path_velocity)

                # Local path information
                robot_info[robot_num].append(num_of_robots / 30)
                robot_info[robot_num].append(max(0.1, path_velocity) / 1.5)
                robot_info[robot_num].append(estimated_time * TICK_TO_SECOND / 60)
                robot_info[robot_num].append(norm_distance / 58) # Norm 49 & 31
                robot_info[robot_num].append(1 if norm_distance <= 15 else 0)
                robot_num += 1
                for path_coordinate in o.path:
                    coordinate = [int(i) for i in path_coordinate.split(',')]
                    global_congestion_map[coordinate[0], coordinate[1]] += 1/(o.path.index(path_coordinate) + 1)
                    
        rl_state['robot_info'] = np.array(robot_info)
        rl_state['global_congestion_map'] = global_congestion_map / 30 # Normalize the congestion map
        rl_state['warehouse_layout'] = np.transpose(pod_path_array / 99)

        # Add warehouse layout to the congestion map for state representation
        map_state = rl_state['warehouse_layout'] + np.array(rl_state['global_congestion_map'])

        return rl_state['robot_info'], map_state
    
    def check_congestion_termination(self, idle_time, robot_state, rewards, robot_info):
        """
        Check if episode should terminate due to system-wide congestion.
        Terminates when either:
        1. 15+ total robots are severely congested (>1200 idle steps), OR
        2. 8+ pod-carrying robots are severely congested (>1200 idle steps)
        
        Args:
            idle_time: List of idle times for each robot
            robot_state: List of current states for each robot  
            rewards: List of current rewards (will be modified for penalized robots)
        
        Returns:
            bool: True if episode should terminate due to congestion
        """
        
        # Count severely congested robots (>1200 idle steps = ~2 minutes)
        total_congested = 0
        carrying_congested = 0
        congested_robot_ids = []
        
        for i, idle in enumerate(idle_time):
            if idle > 1200:  # Severely congested threshold
                total_congested += 1
                congested_robot_ids.append(i)
                
                # Check if this robot is carrying a pod
                if robot_state[i] in ['delivering_pod', 'returning_pod']:
                    carrying_congested += 1
        
        # Check termination conditions
        terminate_general = total_congested >= 15  # 15+ total robots stuck
        terminate_critical = carrying_congested >= 8  # 8+ pod-carrying robots stuck
        
        should_terminate = terminate_general or terminate_critical
        
        if should_terminate:
            # Apply graduated penalties to congested robots
            for robot_id in congested_robot_ids:
                print(robot_id, "Robot state:", robot_state[robot_id], "Heading:", robot_info[robot_id][0]*360, "\n Velocity:", robot_info[robot_id][1]*1.5, 
                      "\n Positions x,y:", f"{robot_info[robot_id][3]*49} , {robot_info[robot_id][4]*31}", f"\n Destinations x,y: {robot_info[robot_id][5]*49} , {robot_info[robot_id][6]*31}",
                      f"\n W1 x,y: {robot_info[robot_id][7]*49} , {robot_info[robot_id][8]*31}")
                idle_steps = idle_time[robot_id]
                
                # Graduated penalty based on severity
                if robot_state[robot_id] in ['delivering_pod', 'returning_pod']:
                    # Harsher penalty for pod-carrying robots
                    if idle_steps > 1800:  # 3+ minutes
                        rewards[robot_id] = -1.0
                    elif idle_steps > 1500:  # 2.5+ minutes  
                        rewards[robot_id] = -0.8
                    else:  # 2+ minutes
                        rewards[robot_id] = -0.6
                else:
                    # Standard penalty for non-carrying robots
                    if idle_steps > 1800:  # 3+ minutes
                        rewards[robot_id] = -0.8
                    elif idle_steps > 1500:  # 2.5+ minutes
                        rewards[robot_id] = -0.6  
                    else:  # 2+ minutes
                        rewards[robot_id] = -0.4
            
            # Log the termination reason
            if terminate_critical:
                print(f"Episode terminated: {carrying_congested} pod-carrying robots severely congested (threshold: 8)")
            if terminate_general:
                print(f"Episode terminated: {total_congested} total robots severely congested (threshold: 15)")
                
            print(f"Penalized {len(congested_robot_ids)} congested robots")
        
        return should_terminate
    
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

    def getReplenishmentStats(self):
        """Return statistics about replenishment operations."""
        return {
            'total_skus_replenished': self.replenishment_count,
            'total_pod_visits': self.replenishment_trips,
            'unique_pods_replenished': len(self.replenished_pods),
            'pod_visit_counts': self.replenished_pods,
            'max_visits_per_pod': max(self.replenished_pods.values()) if self.replenished_pods else 0,
            'avg_visits_per_pod': sum(self.replenished_pods.values()) / len(self.replenished_pods) if self.replenished_pods else 0,
            'avg_skus_per_visit': self.replenishment_count / self.replenishment_trips if self.replenishment_trips > 0 else 0,
            'efficiency': self.replenishment_count / (self.replenishment_trips * len(self.replenished_pods)) if self.replenished_pods else 0
        }

    def resetReplenishmentCounters(self):
        """Reset all replenishment counters and tracking dictionary."""
        self.replenishment_count = 0
        self.replenishment_trips = 0
        self.replenished_pods.clear()

    def buildInventoryDataFrame(self):
        """Build a DataFrame with warehouse inventory data, ensuring all item_ids from items.csv are present, always using item_id column."""
        import pandas as pd
        import os
        
        # Load all item_ids from items.csv (cache in self if not already loaded)
        if not hasattr(self, '_all_item_ids'):
            items_path = os.path.join(PARENT_DIRECTORY, 'data/output/items.csv')
            items_df = pd.read_csv(items_path)
            # Always use item_id column
            self._all_item_ids = set(items_df['item_id'])
        
        data = []
        # Prepare fallback pods.csv DataFrame (cache in self for efficiency)
        if not hasattr(self, '_pods_csv_df'):
            pods_csv_path = os.path.join(PARENT_DIRECTORY, 'data/output/pods.csv')
            print("jadi ada ga ?", os.path.exists(pods_csv_path))
            if os.path.exists(pods_csv_path):
                self._pods_csv_df = pd.read_csv(pods_csv_path)
            else:
                self._pods_csv_df = pd.DataFrame()

        for item_id in self._all_item_ids:
            sku_data = self.pod_manager.skus_data.get(item_id, None)
            if sku_data:
                current_qty = sku_data['current_global_qty']
                max_qty = sku_data['max_global_qty']
                inventory_ratio = current_qty / max_qty if max_qty > 0 else 0
            else:
                # Fallback: sum qty and max_qty from pods.csv for this item_id
                if not self._pods_csv_df.empty:
                    mask = self._pods_csv_df['item'].astype(str) == str(item_id)
                    current_qty = self._pods_csv_df.loc[mask, 'qty'].sum()
                    max_qty = self._pods_csv_df.loc[mask, 'max_qty'].sum()
                    inventory_ratio = current_qty / max_qty if max_qty > 0 else 0
                else:
                    current_qty = 0
                    max_qty = 0
                    inventory_ratio = 0
            data.append({
                'timestamp': self._tick,
                'item_id': item_id,
                'current_quantity': current_qty,
                'max_quantity': max_qty,
                'inventory_ratio': inventory_ratio
            })
        # Create DataFrame
        df = pd.DataFrame(data)
        return df

    def calculateAndSaveInventoryRatio(self):
        """Calculate inventory ratio for each SKU and save to CSV."""
        # Build DataFrame
        df = self.buildInventoryDataFrame()
        
        if df.empty:
            print("No inventory data to save")
            return 0
        
        # Calculate average inventory level
        total_ratio = df['inventory_ratio'].sum()
        self.average_inventory_level = (total_ratio / len(df)) * 100 if len(df) > 0 else 0  # Convert to percentage
        
        # Save DataFrame to CSV directly
        csv_filename = f"inventory-ratios-{self.landscape.current_date_string}.csv"
        csv_path = os.path.join("result", self.landscape.current_date_string, csv_filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Append to CSV (create if doesn't exist)
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        
        # Calculate and save pod inventory metrics
        self.calculateAndSavePodInventoryMetrics()
        
        # Calculate and save weighted pod utilization
        self.calculateAndSaveWeightedPodUtilization()
        
        print(f"Inventory data saved: {len(df)} records")
        print(f"Average inventory level: {self.average_inventory_level:.2f}%")
        
        return self.average_inventory_level

    def buildPodInventoryDataFrame(self):
        """Build a DataFrame with pod inventory data, ensuring all item_id/slot in every pod_id from pods.csv are included at every checkpoint."""
        import pandas as pd
        data = []
        # Prepare fallback pods.csv DataFrame (cache in self for efficiency)
        if not hasattr(self, '_pods_csv_df'):
            pods_csv_path = os.path.join(PARENT_DIRECTORY, 'data/output/pods.csv')
            if os.path.exists(pods_csv_path):
                self._pods_csv_df = pd.read_csv(pods_csv_path)
            else:
                self._pods_csv_df = pd.DataFrame()
        # Get all unique pod_ids from both pod_manager and pods.csv
        pod_ids_from_manager = set(pod.pod_number for pod in self.pod_manager.getAllPods())
        pod_ids_from_csv = set(self._pods_csv_df['pod_id'].unique()) if not self._pods_csv_df.empty else set()
        all_pod_ids = pod_ids_from_manager.union(pod_ids_from_csv)
        for pod_id in all_pod_ids:
            # Get all slot/item_id for this pod from pods.csv
            pod_rows = self._pods_csv_df[self._pods_csv_df['pod_id'] == pod_id] if not self._pods_csv_df.empty else pd.DataFrame()
            # Try to get pod from pod_manager
            pod = next((p for p in self.pod_manager.getAllPods() if p.pod_number == pod_id), None)
            # Build a dict of live pod SKUs if available
            live_skus = pod.getAllSKUInPod() if pod is not None else {}
            # For every item_id/slot in pods.csv for this pod, use live data if available, else pods.csv
            for _, row in pod_rows.iterrows():
                sku_id = row['item']
                max_qty = row['max_qty']
                # Use live pod data if available, else pods.csv
                if sku_id in live_skus:
                    current_qty = live_skus[sku_id]['current_qty']
                    max_qty_live = live_skus[sku_id]['limit_qty']
                    # Use max_qty from live if it differs (shouldn't at t=1)
                    max_qty = max_qty_live
                else:
                    current_qty = row['qty']
                inventory_ratio = current_qty / max_qty if max_qty > 0 else 0
                data.append({
                    'timestamp': self._tick,
                    'pod_id': pod_id,
                    'item_id': sku_id,
                    'current_quantity': current_qty,
                    'max_quantity': max_qty,
                    'inventory_ratio': inventory_ratio
                })
        # Create DataFrame
        df = pd.DataFrame(data)
        return df

    def calculateAndSavePodInventoryMetrics(self):
        """Calculate pod inventory metrics for each SKU and save to CSV."""
        # Build DataFrame
        df = self.buildPodInventoryDataFrame()
        
        if df.empty:
            print("No pod inventory data to save")
            return 0
        
        # Calculate average pod inventory level
        total_pod_ratio = df['inventory_ratio'].sum()
        total_pod_skus = len(df)
        self.average_pod_inventory_level = (total_pod_ratio / total_pod_skus) * 100 if total_pod_skus > 0 else 0
        
        # Save DataFrame to CSV directly
        csv_filename = f"pod-inventory-ratios-{self.landscape.current_date_string}.csv"
        csv_path = os.path.join("result", self.landscape.current_date_string, csv_filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Append to CSV (create if doesn't exist)
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        
        print(f"Pod inventory data saved: {len(df)} records")
        print(f"Average pod inventory level: {self.average_pod_inventory_level:.2f}%")
        
        return self.average_pod_inventory_level

    def calculateAndSaveWeightedPodUtilization(self):
        """Calculate average weighted pod utilization and save to CSV."""
        MAX_POD_WEIGHT = 1300  # Maximum pod weight capacity
        data = []
        total_weighted_utilization = 0
        active_pods_count = 0

        # Prepare fallback pods.csv DataFrame (cache in self for efficiency)
        if not hasattr(self, '_pods_csv_df'):
            pods_csv_path = os.path.join(PARENT_DIRECTORY, 'data/output/pods.csv')
            if os.path.exists(pods_csv_path):
                self._pods_csv_df = pd.read_csv(pods_csv_path)
            else:
                self._pods_csv_df = pd.DataFrame()

        # Get all unique pod_ids from both pod_manager and pods.csv
        pod_ids_from_manager = set(pod.pod_number for pod in self.pod_manager.getAllPods())
        pod_ids_from_csv = set(self._pods_csv_df['pod_id'].unique()) if not self._pods_csv_df.empty else set()
        all_pod_ids = pod_ids_from_manager.union(pod_ids_from_csv)

        for pod_id in all_pod_ids:
            pod = next((p for p in self.pod_manager.getAllPods() if p.pod_number == pod_id), None)
            if pod is not None:
                # Use live pod data (dynamic weight)
                current_weight = pod.getTotalMass()
                # For max_weight, use pods.csv (static max possible weight)

    # def assignJobsInBatches(self):
    #     if not self.job_queue:
    #         return

    #     BATCH_SIZE = 7
    #     current_batch_size = min(BATCH_SIZE, len(self.job_queue))


    #     current_batch = [(i, self.job_queue[i]) for i in range(current_batch_size)]
    #     unfinished_jobs = [(i, job) for i, job in current_batch if not job.is_finished]

    #     if not unfinished_jobs:
    #         for _ in range(current_batch_size):
    #             self.job_queue.pop(0)
    #         return

    #     movable_robots = [
    #         o for o in self.getMovableObjects()
    #         if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle'
    #     ]

    #     job_assignments = []

    #     for robot in movable_robots:
    #         best_job_info = None
    #         best_distance = float('inf')

    #         for i, job in unfinished_jobs:
    #             if getattr(job, 'skip_count', 0) >= BATCH_SIZE-1:
    #                 best_job_info = (i, job)
    #                 break

    #         if best_job_info is None:
    #             for i, job in unfinished_jobs:
    #                 dist = calculateManhattanDistance(
    #                     [robot.pos_x, robot.pos_y],
    #                     [job.pod_coordinate.x, job.pod_coordinate.y]
    #                 )
    #                 if dist < best_distance:
    #                     best_distance = dist
    #                     best_job_info = (i, job)

    #         if best_job_info:
    #             job_assignments.append((robot, best_job_info[0], best_job_info[1]))

    #     for robot, job_index, job in sorted(job_assignments, key=lambda x: -x[1]):
    #         if 0 <= job_index < len(self.job_queue) and self.job_queue[job_index] == job:
    #             self.job_queue.pop(job_index)
    #             robot.assignJobAndSetToTakePod(job)

    #     assigned_jobs = {job for _, _, job in job_assignments}
    #     for _, job in unfinished_jobs:
    #         if job not in assigned_jobs:
    #             if not hasattr(job, 'skip_count'):
    #                 job.skip_count = 1
    #             else:
    #                 job.skip_count += 1

    def assignJobsInBatches(self):
        """
        Versi baru yang menjadi 'Dispatcher Cerdas'.
        Membaca antrian permintaan SKU dan mengubahnya menjadi Job jika memungkinkan,
        dengan logika bundling untuk efisiensi.
        """
        # 1. Cek cepat jika tidak ada permintaan atau tidak ada robot yang siap
        if not self.sku_picking_queue:
            return
            
        available_robots = [r for r in self.getMovableObjects() if isinstance(r, Robot) and r.current_state == 'idle']
        if not available_robots:
            return

        # 2. Siapkan 'wadah' untuk job yang dibuat di tick ini agar bisa di-bundle
        jobs_created_this_tick = {}  # Format: {pod_id: job_object}
        processed_requests = [] # Untuk mencatat permintaan yang berhasil

        # 3. Loop pada setiap permintaan di antrian
        for request in self.sku_picking_queue:
            # Jika robot nganggur sudah habis, berhenti memproses untuk tick ini
            if not available_robots:
                break

            sku_id = request['sku']
            order_id = request['order_id']
            
            # 4. Coba cari pod yang tersedia untuk SKU ini
            available_pod = self.pod_manager.getAvailablePod(sku_id)
            
            # 5. Jika pod TIDAK ditemukan (sibuk/stok habis), ABAIKAN.
            # Permintaan ini akan tetap di antrian untuk dicoba lagi di tick berikutnya.
            if available_pod is None:
                continue

            if available_pod.is_awaiting_replenishment:
                continue

            # --- Jika pod DITEMUKAN, jalankan logika bundling ---
            pod_id = available_pod.pod_number
            
            if pod_id in jobs_created_this_tick:
                # KASUS A: SUDAH ADA JOB UNTUK POD INI -> 'Titip' tugas baru
                job = jobs_created_this_tick[pod_id]
                self._add_picking_task_to_job(job, order_id, sku_id, request['qty'])
                print(f"DEBUG: Hitchhiked SKU {sku_id} for Order {order_id} onto existing Job {job.id}")
            else:
                # KASUS B: BELUM ADA JOB UNTUK POD INI -> Buat job baru
                order = self.order_manager.getOrderById(order_id)
                job = self.job_manager.createJob(
                    available_pod.coordinate,
                    station_id=order.station_id,
                    pod=available_pod
                )
                self._add_picking_task_to_job(job, order_id, sku_id, request['qty'])
                
                # Cari robot terdekat dan berikan job
                robot_to_assign = self.robot_manager.findNearestAvailableRobot(available_pod.coordinate)
                if robot_to_assign:
                    robot_to_assign.assignJobAndSetToTakePod(job)
                    self.pod_manager.setPodNotAvailable(available_pod)
                    available_robots.remove(robot_to_assign) # Hapus dari daftar robot nganggur
                    jobs_created_this_tick[pod_id] = job # Catat job baru ini untuk bundling
                else:
                    # Seharusnya tidak terjadi, tapi ini safety net
                    self._release_active_job_quantity(order_id, sku_id, request['qty'])
                    continue
            
                self.assign_order_df.loc[((self.assign_order_df['order_id'] == order.id) & (self.assign_order_df['item_id'] == sku_id)), 'assigned_pod'] = int(available_pod.pod_number)
                
                self.assign_order_df.loc[((self.assign_order_df['order_id'] == order.id) & (self.assign_order_df['item_id'] == sku_id)), 'status'] = 0

            # Tandai permintaan ini sebagai berhasil diproses
            processed_requests.append(request)

        # 6. Setelah loop selesai, bersihkan semua permintaan yang sudah jadi Job
        if processed_requests:
            processed_counter = Counter(
                (req.get("order_id"), req.get("sku"), req.get("qty", 0))
                for req in processed_requests
            )
            rebuilt_queue = []
            for req in self.sku_picking_queue:
                key = (req.get("order_id"), req.get("sku"), req.get("qty", 0))
                if processed_counter.get(key, 0) > 0:
                    processed_counter[key] -= 1
                    continue
                rebuilt_queue.append(req)
            self._replace_sku_picking_queue(rebuilt_queue)
            # print(f"✅ Batch assigned {len(jobs_created_this_tick)} jobs for {len(processed_requests)} SKU requests.")

    def assignJobsInBatches_MODIFIED(self):
        """
        Versi 'Dispatcher Super Cerdas'.
        Mengelompokkan permintaan berdasarkan Pod untuk bundling yang maksimal.
        Satu Pod -> Satu Robot -> Borong semua tugas.
        """
        # 1. Cek cepat jika tidak ada permintaan atau tidak ada robot yang siap
        if not self.sku_picking_queue:
            return
            
        available_robots = [r for r in self.getMovableObjects() if isinstance(r, Robot) and r.current_state == 'idle']
        if not available_robots:
            return

        # ================= LOGIKA BARU START =================
        
        # 2. Kumpulin dulu semua request berdasarkan kombinasi (station, pod)
        # Ini mencegah satu job membawa task dari station yang berbeda.
        # Format: {(station_id, pod_id): {'pod_object': Pod, 'station_id': str, 'requests': [req1, req2, ...]}}
        requests_by_group = {}
        available_robot_coords = [[robot.pos_x, robot.pos_y] for robot in available_robots]
        
        # Simpan request yang pod-nya nggak ketemu/sibuk untuk ditaruh lagi di antrian
        unprocessed_requests = [] 
        dropped_requests = []

        for request in self.sku_picking_queue:
            sku_id = request['sku']
            order = self.order_manager.getOrderById(request['order_id'])
            if order is None:
                dropped_requests.append(request)
                continue

            if order.station_id is None:
                order.is_in_queue = False
                dropped_requests.append(request)
                continue

            order_station = self.station_manager.getStationById(order.station_id)
            skus_in_station_dict = order_station.getSKUsInStationDict()

            # Gunakan pemilihan pod yang mempertimbangkan kebutuhan station agar pile-on lebih baik.
            available_pod = self.pod_manager.getAvailablePodInventory(
                sku_id,
                skus_in_station_dict,
                order_station.coordinate,
                available_robot_coords,
            )
            if available_pod is None:
                available_pod = self.pod_manager.getAvailablePod(sku_id)
            
            # Jika pod TIDAK ditemukan atau sedang nunggu restock, skip dulu
            # Request ini akan diproses di tick berikutnya
            if available_pod is None or available_pod.is_awaiting_replenishment:
                unprocessed_requests.append(request)
                continue

            pod_id = available_pod.pod_number
            group_key = (order.station_id, pod_id)
            
            # Masukkan request ke 'rombongan' pod dan station-nya
            if group_key not in requests_by_group:
                requests_by_group[group_key] = {
                    'pod_object': available_pod,
                    'station_id': order.station_id,
                    'requests': [],
                }
            
            requests_by_group[group_key]['requests'].append(request)

        # 3. Sekarang, proses setiap 'rombongan' per Pod/Station.
        # Prioritaskan grup terbesar supaya satu kunjungan pod mengerjakan lebih banyak task.
        processed_requests = [] # Untuk mencatat semua request yang berhasil jadi Job
        
        grouped_assignments = sorted(
            requests_by_group.items(),
            key=lambda item: len(item[1]['requests']),
            reverse=True,
        )

        # Kita iterasi pada dictionary rombongan yang sudah dibuat
        for (_, _), data in grouped_assignments:
            # Cek lagi, robotnya masih ada nggak?
            if not available_robots:
                # Jika robot habis, sisa rombongan ini akan dicoba lagi di tick selanjutnya
                unprocessed_requests.extend(data['requests'])
                continue

            pod_object = data['pod_object']
            station_id = data['station_id']
            pod_requests = data['requests']

            # Pod yang sama bisa muncul di grup station lain dari snapshot awal queue.
            # Begitu satu grup sudah mengambil pod ini, grup lain harus ditunda.
            if not pod_object.is_idle or pod_object.is_awaiting_replenishment:
                unprocessed_requests.extend(pod_requests)
                continue

            # Cari robot terdekat untuk ngambil pod ini
            robot_to_assign = self.robot_manager.findNearestAvailableRobot(pod_object.coordinate)

            if robot_to_assign:
                # Hanya buat SATU job untuk satu pod ini
                job = self.job_manager.createJob(
                    pod_object.coordinate,
                    station_id=station_id,
                    pod=pod_object
                )
                
                # BORONG SEMUA! Masukkan semua task dari rombongan ini ke job yang sama
                for req in pod_requests:
                    order = self.order_manager.getOrderById(req['order_id'])
                    if order is None:
                        continue

                    order.commitQuantity(req['sku'], req['qty'])
                    self._add_picking_task_to_job(job, req['order_id'], req['sku'], req['qty'])

                    # Update DataFrame atau status tracking lo
                    self.assign_order_df.loc[((self.assign_order_df['order_id'] == req['order_id']) & (self.assign_order_df['item_id'] == req['sku'])), 'assigned_pod'] = int(pod_object.pod_number)
                    self.assign_order_df.loc[((self.assign_order_df['order_id'] == req['order_id']) & (self.assign_order_df['item_id'] == req['sku'])), 'status'] = 0 # Status: Assigned

                # Assign robot dan kunci resource
                self.station_manager.getStationById(station_id).addPod(pod_object.pod_number)
                robot_to_assign.assignJobAndSetToTakePod(job)
                self.pod_manager.setPodNotAvailable(pod_object)
                available_robots.remove(robot_to_assign)
                self.pod_visit_to_station += 1
                
                # Tandai semua request di rombongan ini sebagai berhasil diproses
                processed_requests.extend(pod_requests)
            else:
                # Jika nggak ada robot (semua sibuk), balikin request rombongan ini ke antrian
                unprocessed_requests.extend(pod_requests)

        # 4. Update antrian utama: buang yang sudah diproses, sisakan yang belum
        processed_counter = Counter(
            (req.get("order_id"), req.get("sku"), req.get("qty", 0))
            for req in processed_requests
        )
        unprocessed_counter = Counter(
            (req.get("order_id"), req.get("sku"), req.get("qty", 0))
            for req in unprocessed_requests
        )
        dropped_counter = Counter(
            (req.get("order_id"), req.get("sku"), req.get("qty", 0))
            for req in dropped_requests
        )
        remaining_requests = []
        for req in self.sku_picking_queue:
            key = (req.get("order_id"), req.get("sku"), req.get("qty", 0))
            if processed_counter.get(key, 0) > 0:
                processed_counter[key] -= 1
                continue
            if unprocessed_counter.get(key, 0) > 0:
                unprocessed_counter[key] -= 1
                continue
            if dropped_counter.get(key, 0) > 0:
                dropped_counter[key] -= 1
                continue
            remaining_requests.append(req)

        self._replace_sku_picking_queue(unprocessed_requests + remaining_requests)

    # ================= LOGIKA BARU END ===================


# ========================================================== OLD CODE BROW SED ========================================================== 
    # def assignJobsInBatches(self):
    #     """
    #     Efficiently assign multiple jobs to available robots in batches.
    #     This improves robot utilization by assigning jobs to ALL available robots,
    #     not just the nearest one.
    #     """
    #     if len(self.job_queue) == 0:
    #         return
        
    #     # Get all available robots
    #     available_robots = []
    #     for robot in self.getMovableObjects():
    #         if (robot.object_type == "robot" and 
    #             (robot.job is None or robot.job.is_finished) and 
    #             robot.current_state == 'idle'):
    #             available_robots.append(robot)
        
    #     if not available_robots:
    #         return
        
    #     jobs_assigned = 0
    #     max_assignments = min(len(self.job_queue), len(available_robots))
        
    #     # Assign jobs to robots based on distance
    #     for _ in range(max_assignments):
    #         if len(self.job_queue) == 0:
    #             break
                
    #         job = self.job_queue[0]
    #         best_robot = None
    #         min_distance = float('inf')
            
    #         # Find the closest available robot to this job
    #         for robot in available_robots:
    #             distance = calculateManhattanDistance(
    #                 [robot.pos_x, robot.pos_y], 
    #                 [job.pod_coordinate.x, job.pod_coordinate.y]
    #             )
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 best_robot = robot
            
    #         if best_robot is not None:
    #             # Assign job to robot
    #             self.job_queue.pop(0)
    #             best_robot.assignJobAndSetToTakePod(job)
    #             available_robots.remove(best_robot)  # Remove from available list
    #             jobs_assigned += 1
        
    #     if jobs_assigned > 0:
    #         print(f"✅ Batch assigned {jobs_assigned} jobs to robots")
            
    def update_global_sku_watchlist(self):
        """
        Rebuild the global watchlist from warehouse inventory ratio metadata.
        """
        self.global_critical_skus = set()
        all_skus_data = self.pod_manager.getAllSKUData()

        for sku_id, data in all_skus_data.items():
            threshold_ratio = float(data.get("global_threshold_inv_level", 0))
            if threshold_ratio <= 0:
                continue

            current_global_ratio = float(data.get("global_inv_level", 0))
            if current_global_ratio <= threshold_ratio:
                self.global_critical_skus.add(sku_id)

    def get_pod_average_fill_score(self, pod: Pod) -> float:
        if not pod.skus:
            return 1.0

        total_inventory_level = 0.0
        counted_skus = 0
        for details in pod.skus.values():
            limit_qty = details.get('limit_qty', 0)
            if limit_qty <= 0:
                continue

            total_inventory_level += details.get('current_qty', 0) / limit_qty
            counted_skus += 1

        if counted_skus == 0:
            return 1.0

        return total_inventory_level / counted_skus

    def get_below_reorder_skus_for_pod(self, pod: Pod) -> List[int]:
        if pod is None or not pod.skus:
            return []

        critical_skus = []
        for sku_id in pod.skus.keys():
            _, needs_replenishment = self.pod_manager.isSKUNeedReplenishment(sku_id)
            if needs_replenishment:
                self.global_critical_skus.add(sku_id)
                critical_skus.append(sku_id)

        return sorted(critical_skus)

    def get_pod_critical_fill_score(
        self,
        pod: Pod,
        critical_skus: Optional[List[int]] = None,
    ) -> float:
        if pod is None or not pod.skus:
            return 1.0

        if critical_skus is None:
            critical_skus = self.get_below_reorder_skus_for_pod(pod)

        if not critical_skus:
            return 1.0

        fill_ratios = []
        for sku_id in critical_skus:
            details = pod.skus.get(sku_id)
            if not details:
                continue

            limit_qty = details.get("limit_qty", 0)
            if limit_qty <= 0:
                continue

            current_qty = details.get("current_qty", 0)
            fill_ratios.append(max(0.0, min(1.0, current_qty / limit_qty)))

        if not fill_ratios:
            return 1.0

        return float(sum(fill_ratios) / len(fill_ratios))

    def get_replenishment_skus_for_pod(self, pod: Pod) -> tuple[list, float]:
        """
        Returns the below-reorder SKUs that make this pod eligible for
        replenishment, using Qj_critical as the binary pod-level gate.
        """
        if pod is None or not pod.skus:
            return [], 1.0

        critical_skus = self.get_below_reorder_skus_for_pod(pod)
        qj_score = self.get_pod_critical_fill_score(pod, critical_skus)
        if qj_score >= self.pod_replenishment_threshold:
            return [], qj_score

        return critical_skus, qj_score

    def getPendingReplenishmentDispatch(self, pod_number: int) -> Optional[Dict]:
        for request in self.pending_replenishment_dispatches:
            if int(request["pod_number"]) == int(pod_number):
                return request
        return None

    def getPendingReplenishmentRequestForSKU(self, sku_id: int) -> Optional[Dict]:
        for request in self.pending_replenishment_dispatches:
            if sku_id in request.get("skus_to_replenish", []):
                return request
        return None

    def hasOnHoldDemandForSKUs(self, skus_to_check) -> bool:
        target_skus = {sku for sku in skus_to_check if sku}
        if not target_skus:
            return False

        for order in self.order_manager.unfinished_orders:
            if not getattr(order, "on_hold", False):
                continue

            remaining_skus = order.getRemainingSKU()
            if any(sku in remaining_skus for sku in target_skus):
                return True

        return False

    def hasActiveQueueDemandForSKUs(self, skus_to_check) -> bool:
        target_skus = {sku for sku in skus_to_check if sku}
        if not target_skus:
            return False

        return any(
            request.get("sku") in target_skus and request.get("qty", 0) > 0
            for request in self.sku_picking_queue
        )

    def getReplenishmentUrgencyLevel(self, skus_to_check) -> int:
        if self.hasOnHoldDemandForSKUs(skus_to_check):
            return 2
        if self.hasActiveQueueDemandForSKUs(skus_to_check):
            return 1
        return 0

    def shouldGuaranteeReplenishmentRequest(
        self,
        request: Optional[Dict],
        current_tick: Optional[int] = None,
    ) -> bool:
        if request is None:
            return False

        if current_tick is None:
            current_tick = int(self._tick)

        if bool(request.get("guaranteed_on_release", False)):
            return True

        wait_time = current_tick - int(request.get("created_tick", current_tick))
        return wait_time >= self.replenishment_dispatch_aging_ticks

    def getMostDepletedEligiblePodForSKU(self, sku_id: int):
        pods_with_sku = self.pod_manager.getPodsBySKU(sku_id) or []
        best_candidate = None
        best_key = None

        for pod in pods_with_sku:
            if pod is None or pod.is_awaiting_replenishment:
                continue

            skus_to_replenish, qj_score = self.get_replenishment_skus_for_pod(pod)
            if not skus_to_replenish or sku_id not in skus_to_replenish:
                continue

            sku_details = pod.skus.get(sku_id, {})
            sku_limit_qty = sku_details.get("limit_qty", 0)
            sku_current_qty = sku_details.get("current_qty", 0)
            sku_fill_ratio = (
                sku_current_qty / sku_limit_qty
                if sku_limit_qty > 0
                else 1.0
            )
            idle_penalty = 0 if pod.is_idle else 1
            candidate_key = (
                qj_score,
                sku_fill_ratio,
                idle_penalty,
                int(pod.pod_number),
            )

            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_candidate = (pod, skus_to_replenish, qj_score)

        return best_candidate

    def enqueueBestReplenishmentDispatchForSKU(self, sku_id: int) -> bool:
        existing_request = self.getPendingReplenishmentRequestForSKU(sku_id)
        if existing_request is not None:
            try:
                existing_pod = self.pod_manager.getPodByNumber(
                    int(existing_request["pod_number"])
                )
            except (IndexError, TypeError):
                existing_pod = None

            if existing_pod is not None:
                return self.enqueuePendingReplenishmentDispatch(
                    existing_pod,
                    existing_request.get("skus_to_replenish", []),
                )
            return False

        candidate = self.getMostDepletedEligiblePodForSKU(sku_id)
        if candidate is None:
            return False

        pod, skus_to_replenish, _ = candidate
        return self.enqueuePendingReplenishmentDispatch(pod, skus_to_replenish)

    def refreshMandatoryReplenishmentPods(self):
        current_tick = int(self._tick)

        for pod in self.pod_manager.getAllPods():
            if pod is None or pod.is_awaiting_replenishment:
                continue
            pod.must_replenish_before_pick = False

        for request in self.pending_replenishment_dispatches:
            if not self.shouldGuaranteeReplenishmentRequest(request, current_tick):
                continue

            try:
                pod = self.pod_manager.getPodByNumber(int(request["pod_number"]))
            except (IndexError, TypeError):
                pod = None

            if pod is not None and not pod.is_awaiting_replenishment:
                pod.must_replenish_before_pick = True

    def tryGuaranteeReplenishmentOnRelease(self, pod: Pod, robot: Robot) -> bool:
        if pod is None or robot is None:
            return False

        request = self.getPendingReplenishmentDispatch(pod.pod_number)
        if request is None:
            return False

        if not self.shouldGuaranteeReplenishmentRequest(request, int(self._tick)):
            return False

        skus_to_replenish, _ = self.get_replenishment_skus_for_pod(pod)
        if not skus_to_replenish:
            self.removePendingReplenishmentDispatch(pod.pod_number)
            return False

        station = self.station_manager.findAvailableReplenishmentStation()
        if station is None:
            pod.must_replenish_before_pick = True
            return False

        success = self.sendPodForReplenishment(
            pod,
            station,
            skus_to_replenish,
            robot,
        )
        if not success:
            pod.must_replenish_before_pick = True
        return success
    
    def trigger_bundled_proactive_replenishment(self, pod: Pod, robot: Robot):
        """
        After a picking job finishes on a pod, revisit any SKUs in that pod
        that have crossed their reorder point and enqueue one replenishment
        request per critical SKU using the single-most-depleted eligible pod.
        """
        if pod is None:
            return

        critical_skus = self.get_below_reorder_skus_for_pod(pod)
        if not critical_skus:
            return

        for sku_id in critical_skus:
            self.enqueueBestReplenishmentDispatchForSKU(sku_id)
