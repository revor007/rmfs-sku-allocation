from __future__ import annotations
from typing import List, TYPE_CHECKING
from world.entities.pod import Pod
from lib.types.netlogo_coordinate import NetLogoCoordinate
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances
from lib.constant import PARENT_DIRECTORY
import os
if TYPE_CHECKING:
    from world.warehouse import Warehouse
    
class PodManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.pods: List[Pod] = []
        self.pod_counter = 0
        self.sku_to_pods = {}
        self.coordinate_to_pods = {}
        self.skus_data = {}
        self.item_weights = {}  # Dictionary to store item weights from pods.csv
        self.use_adaptive_replenishment = False  # Flag to control which policy to use
        self.pod_return_logic = "fixed" # "fixed" or "nearest" up 2 u

    def initPodManager(self):
        # Load item weights from pods.csv
        self._loadItemWeights()
        
        for pod in self.pods:
            pod.setPodManager(self)

    def _loadItemWeights(self):
        """Load item weights from pods.csv file."""
        try:
            pods_df = pd.read_csv(PARENT_DIRECTORY + '/data/output/pods.csv')
            item_column = 'item_id' if 'item_id' in pods_df.columns else 'item'
            if item_column not in pods_df.columns or 'item_weight' not in pods_df.columns:
                raise KeyError("pods.csv must contain item/item_id and item_weight columns")

            # Create a dictionary mapping item id to unit weight.
            self.item_weights = dict(zip(pods_df[item_column], pods_df['item_weight']))
        except Exception as e:
            print(f"Warning: Could not load item weights from pods.csv: {e}")
            self.item_weights = {}

    def getAllPods(self):
        return self.pods
    
    def getAllSKUData(self):
        return self.skus_data
    
    def getPodsBySKU(self, sku):
        return self.sku_to_pods.get(sku, None)

    def getPodsByCoordinate(self, x, y):
        return self.coordinate_to_pods.get((x, y), None)

    def getPodByNumber(self, pod_number):
        return self.pods[pod_number]
    
    def getAvailablePod(self, sku: str):
        if sku in self.sku_to_pods:
            seen_pods = set()
            for pod in self.sku_to_pods[sku]:
                if pod.pod_number in seen_pods:
                    continue
                seen_pods.add(pod.pod_number)
                if pod.is_idle and not pod.is_awaiting_replenishment and not pod.must_replenish_before_pick:
                    qty = pod.getQuantity(sku)
                    if qty > 0:
                        return pod
        return None

    def getAvailablePodSimilarity(self, sku: str, skus_in_station, station_coordinate, robots_coordinate):
        # If SKU is available
        sku_in_station_list = [i for i in skus_in_station]
        pod_available_for_multiple_items = pd.DataFrame(columns=["pod_id", "similarity_score", "distance_to_station", "distance_to_robot"])
        
        station_coordinate = [station_coordinate.x, station_coordinate.y]

        if sku in self.sku_to_pods:
            for pod in self.sku_to_pods[sku]:
                similarity_score = 1
                if (
                    pod.is_idle is True
                    and not pod.is_awaiting_replenishment
                    and not pod.must_replenish_before_pick
                ):
                    pod_skus = [i for i in pod.skus]
                    pod_skus_in_station_skus_mask = np.isin(sku_in_station_list, pod_skus)
                    pod_skus_in_station_skus = np.array(sku_in_station_list)[pod_skus_in_station_skus_mask]
                    
                    if len(pod_skus_in_station_skus) > 0:
                        for skus in pod_skus_in_station_skus:
                            skus_qty_in_pod = pod.getQuantity(skus)
                            if skus_qty_in_pod > 0:
                                similarity_score += 1
                    
                    pod_coordinate = [pod.coordinate.x, pod.coordinate.y]
                    distance = manhattan_distances([pod_coordinate],[station_coordinate])[0][0]

                    distance_to_robot = self._distancePodToRobot(pod_coordinate, robots_coordinate)
                    
                    pod_available_for_multiple_items = pd.concat([pod_available_for_multiple_items, 
                                                                pd.DataFrame([[pod.pod_number, similarity_score, distance, distance_to_robot]], 
                                                                                                            columns=["pod_id", 
                                                                                                                    "similarity_score", 
                                                                                                                    "distance_to_station", "distance_to_robot"])], ignore_index=True) 
            pod_available_for_multiple_items["distance_score"] = pod_available_for_multiple_items["distance_to_station"].max() - pod_available_for_multiple_items["distance_to_station"] + pod_available_for_multiple_items["distance_to_robot"].max() - pod_available_for_multiple_items["distance_to_robot"]
            pod_available_for_multiple_items.sort_values(by=["similarity_score", "distance_score"], ascending=[False, False], inplace=True)
            pod_available_for_multiple_items.reset_index(drop=True, inplace=True)
            pod_available_for_multiple_items = pod_available_for_multiple_items[pod_available_for_multiple_items["similarity_score"] > 0]

            assigned_pod = None
            if len(pod_available_for_multiple_items) > 0:
                assigned_pod_id = pod_available_for_multiple_items.iloc[0]["pod_id"]
           
                assigned_pod = self.getPodByNumber(assigned_pod_id)
        
            return assigned_pod
    
    def getAvailablePodInventory(self, sku: str, skus_in_station_dict, station_coordinate, robots_coordinate):
        sku_in_station_list = list(skus_in_station_dict)
        station_coordinate = [station_coordinate.x, station_coordinate.y]
        if sku not in self.sku_to_pods:
            return

        candidates = []
        max_distance_to_station = None
        for pod in self.sku_to_pods[sku]:
            if (
                pod.is_idle is not True
                or pod.is_awaiting_replenishment
                or pod.must_replenish_before_pick
            ):
                continue

            similarity_score = 0
            for station_sku in sku_in_station_list:
                if station_sku in pod.skus and pod.getQuantity(station_sku) > 0:
                    similarity_score += 1

            pod_coordinate = [pod.coordinate.x, pod.coordinate.y]
            distance_to_station = abs(pod_coordinate[0] - station_coordinate[0]) + abs(
                pod_coordinate[1] - station_coordinate[1]
            )
            distance_to_robot = self._distancePodToRobot(pod_coordinate, robots_coordinate)
            inventory_score = self._countFulfillment(skus_in_station_dict, pod.skus)

            if max_distance_to_station is None or distance_to_station > max_distance_to_station:
                max_distance_to_station = distance_to_station

            candidates.append(
                (
                    pod,
                    similarity_score,
                    inventory_score,
                    distance_to_station,
                    distance_to_robot,
                )
            )

        if not candidates:
            return

        if max_distance_to_station is None:
            max_distance_to_station = 0

        best_pod = None
        best_cost = None
        station_sku_count = len(sku_in_station_list)
        for pod, similarity_score, inventory_score, distance_to_station, distance_to_robot in candidates:
            if similarity_score <= 0 or inventory_score <= 0:
                continue

            station_distance_score = max_distance_to_station - distance_to_station
            cost = (station_distance_score + distance_to_robot) * similarity_score * (
                station_sku_count / inventory_score
            )
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pod = pod

        return best_pod
    
    def getPodNeedReplenishment(self, list_of_sku):
        replenished_pod_needed_every_sku = {}

        for sku in list_of_sku:
            check_pod: List[Pod] = self.sku_to_pods[sku]
            replenished_pod_needed_every_sku[sku] = check_pod
        
        return replenished_pod_needed_every_sku
    
    def setPodNotAvailable(self, pod: Pod):
        pod.is_idle = False

    def setPodAvailable(self, pod: Pod):
        pod.is_idle = True

    def createPod(self, x: int, y: int):
        pod = Pod(self.pod_counter, x, y)
        self.pod_counter += 1
        self.pods.append(pod)
        self.coordinate_to_pods[(pod.pos_x, pod.pos_y)] = pod

        for sku in pod.skus:
            if sku not in self.sku_to_pods:
                self.sku_to_pods[sku] = []
            self.sku_to_pods[sku].append(pod)
        
        return pod

    def addSKUToPod(self, sku: int, pod: Pod):
        if sku not in self.sku_to_pods:
            self.sku_to_pods[sku] = []
        if pod not in self.sku_to_pods[sku]:
            self.sku_to_pods[sku].append(pod)
        item_weight = self.item_weights.get(sku, 0.0)

        # pod.addSKU(sku, limit_qty, current_qty, item_weight)

    @staticmethod
    def _parse_threshold_value(raw_value):
        try:
            return float(str(raw_value).replace(",", "."))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _resolve_global_reorder_point_qty(threshold_value: float, max_global_qty: int) -> int:
        if threshold_value <= 1.0:
            return int(max(0, np.ceil(max_global_qty * threshold_value)))
        return int(max(0, np.ceil(threshold_value)))

    def addSKUData(self,sku,current_qty,max_qty,global_threshold_inv_level):
        sku_id = sku
        threshold_value = self._parse_threshold_value(global_threshold_inv_level)

        if sku_id not in self.skus_data:
            reorder_point_qty = self._resolve_global_reorder_point_qty(threshold_value, max_qty)
            self.skus_data[sku_id] = {
                'current_global_qty': current_qty,
                'max_global_qty': max_qty,
                'global_inv_level': (current_qty / max_qty) if max_qty > 0 else 0,
                'global_threshold_inv_level': threshold_value,
                'global_reorder_point_qty': reorder_point_qty,
            }
        else:
            self.skus_data[sku_id]['current_global_qty'] += current_qty
            self.skus_data[sku_id]['max_global_qty'] += max_qty
            if self.skus_data[sku_id]['max_global_qty'] > 0:
                self.skus_data[sku_id]['global_inv_level'] = (
                    self.skus_data[sku_id]['current_global_qty']
                    / self.skus_data[sku_id]['max_global_qty']
                )
            else:
                self.skus_data[sku_id]['global_inv_level'] = 0

            if threshold_value <= 1.0:
                self.skus_data[sku_id]['global_reorder_point_qty'] = (
                    self._resolve_global_reorder_point_qty(
                        threshold_value,
                        self.skus_data[sku_id]['max_global_qty'],
                    )
                )
            else:
                self.skus_data[sku_id]['global_reorder_point_qty'] = max(
                    int(self.skus_data[sku_id].get('global_reorder_point_qty', 0)),
                    self._resolve_global_reorder_point_qty(threshold_value, max_qty),
                )
            self.skus_data[sku_id]['global_threshold_inv_level'] = threshold_value

    def reduceSKUDataForPod(self, sku, quantity, pod_id):
        """Reduce the quantity of a SKU for a specific pod and update global skus_data accordingly, never going below zero. Returns the actual reduced amount."""
        # Find the pod by pod_id
        pod = next((p for p in self.pods if p.pod_number == pod_id), None)
        actual_reduced = 0
        if pod is not None and sku in pod.skus:
            available_qty = pod.skus[sku]['current_qty']
            reduce_qty = min(quantity, available_qty)
            pod.skus[sku]['current_qty'] -= reduce_qty
            actual_reduced = reduce_qty
            # Update pod mass if needed
            if hasattr(pod, 'sku_weights') and sku in pod.sku_weights:
                pod.current_load_mass -= pod.sku_weights[sku] * reduce_qty
        # Update global skus_data
        if sku in self.skus_data:
            self.skus_data[sku]['current_global_qty'] -= actual_reduced
            if self.skus_data[sku]['max_global_qty'] > 0:
                self.skus_data[sku]['global_inv_level'] = self.skus_data[sku]['current_global_qty'] / \
                                                          self.skus_data[sku]['max_global_qty']
            else:
                self.skus_data[sku]['global_inv_level'] = 0
        return actual_reduced

    def reduceSKUData(self,sku,quantity): #NOT USED AS IT DIDNT TAKEN INTO ACCOUNT THE POD_ID
         if sku in self.skus_data:
            self.skus_data[sku]['current_global_qty'] -= quantity # current_global_qty -> ngecek warehouse threshold
            self.skus_data[sku]['global_inv_level'] = self.skus_data[sku]['current_global_qty'] / self.skus_data[sku]['max_global_qty']

    def increaseSKUData(self, sku, quantity):
        if quantity <= 0 or sku not in self.skus_data:
            return

        sku_data = self.skus_data[sku]
        sku_data['current_global_qty'] += quantity
        if sku_data['current_global_qty'] > sku_data['max_global_qty']:
            sku_data['current_global_qty'] = sku_data['max_global_qty']

        if sku_data['max_global_qty'] > 0:
            sku_data['global_inv_level'] = (
                sku_data['current_global_qty'] / sku_data['max_global_qty']
            )
        else:
            sku_data['global_inv_level'] = 0

    def getSKUReorderPointQty(self, sku_id):
        sku_data = self.skus_data.get(sku_id)
        if sku_data is None:
            return 0
        return int(sku_data.get('global_reorder_point_qty', 0))
    
    def isSKUNeedReplenishment(self, sku_id):
        sku_data = self.skus_data.get(sku_id)
        if sku_data is None:
            return sku_id, False

        global_threshold_inv_level = float(sku_data.get('global_threshold_inv_level', 0))
        current_global_ratio = float(sku_data.get('global_inv_level', 0))
        return sku_id, current_global_ratio <= global_threshold_inv_level

    def updateGlobalInventory(self, sku, reduced_quantity):
        """Update global inventory tracking after pod-level reduction"""
        if sku in self.skus_data and self.skus_data[sku]['max_global_qty'] > 0:
            self.skus_data[sku]['current_global_qty'] -= reduced_quantity
            # Prevent negative global inventory
            if self.skus_data[sku]['current_global_qty'] < 0:
                self.skus_data[sku]['current_global_qty'] = 0
            self.skus_data[sku]['global_inv_level'] = self.skus_data[sku]['current_global_qty'] / self.skus_data[sku][
                'max_global_qty']

    def getBestReplenishmentCandidate(self, sku):
        """
        Find the best pod candidate for replenishing a specific SKU.
        Prioritizes pods with:
        1. Contains the SKU
        2. Currently idle (not assigned to jobs)
        3. Low current inventory for the SKU
        4. High total inventory utilization (more efficient replenishment)
        5. Scarcity factor - prioritize pods with last instances of SKU
        
        Edge cases handled:
        - SKU not found in any pod
        - All pods busy
        - Pods with zero capacity
        - Invalid pod data
        """
        if not sku or sku not in self.sku_to_pods:
            print(f"SKU {sku} not found in any pod")
            return None
            
        pods_with_sku = self.sku_to_pods[sku]
        
        # Count pods with available quantity and track zero-quantity pods for scarcity factor
        pods_with_available_qty = 0
        pods_with_zero_qty = []
        
        for pod in pods_with_sku:
            if pod is not None and hasattr(pod, 'skus') and sku in pod.skus:
                current_qty = pod.skus[sku].get('current_qty', 0)
                if current_qty > 0:
                    pods_with_available_qty += 1
                elif pod.skus[sku].get('limit_qty', 0) > 0:
                    # Track pods that previously had this SKU (have max_qty but current_qty=0)
                    pods_with_zero_qty.append(pod)
        
        # Calculate scarcity multiplier (higher when fewer pods have the SKU)
        total_pods_with_sku = len(pods_with_sku) if pods_with_sku else 1
        scarcity_multiplier = max(1.0, 3.0 * (1 - pods_with_available_qty / total_pods_with_sku))
        
        # Enhanced scarcity for completely depleted SKUs
        if pods_with_available_qty == 0 and pods_with_zero_qty:
            scarcity_multiplier = 5.0  # Critical depletion multiplier
            print(f"SKU {sku} is completely depleted! Found {len(pods_with_zero_qty)} pods that previously had this SKU")
        
        candidate_pods = []
        
        for pod in pods_with_sku:
            if pod is None or not hasattr(pod, 'skus') or sku not in pod.skus:
                continue
                
            # Skip if pod is already busy
            if not pod.is_idle:
                continue
                
            # Get SKU data
            sku_data = pod.skus[sku]
            limit_qty = sku_data.get('limit_qty', 0)
            current_qty = sku_data.get('current_qty', 0)
            
            # Skip if pod has no capacity for this SKU (never had it)
            if limit_qty <= 0:
                continue
                
            # Calculate replenishment potential
            sku_utilization = current_qty / limit_qty if limit_qty > 0 else 0
            
            # Calculate overall pod utilization
            total_current = sum(item.get('current_qty', 0) for item in pod.skus.values())
            total_limit = sum(item.get('limit_qty', 0) for item in pod.skus.values())
            pod_utilization = total_current / total_limit if total_limit > 0 else 0
            
            # Base priority score: lower SKU utilization = higher priority
            base_priority = (1 - sku_utilization) + (pod_utilization * 0.3)
            
            # Special handling for zero-quantity pods (depleted SKUs)
            if current_qty == 0 and pods_with_available_qty == 0:
                # For completely depleted SKUs, prioritize by max_qty capacity
                max_qty_all_pods = max(p.skus[sku].get('limit_qty', 0) 
                                     for p in pods_with_sku 
                                     if p is not None and hasattr(p, 'skus') and sku in p.skus)
                max_qty_ratio = limit_qty / max_qty_all_pods if max_qty_all_pods > 0 else 0
                base_priority = 2.0 + max_qty_ratio  # High base priority for depleted SKUs
                print(f"Zero-qty pod {pod.pod_number}: max_qty={limit_qty}, ratio={max_qty_ratio:.2f}")
            
            # Apply scarcity multiplier for last instance prioritization
            priority_score = base_priority * scarcity_multiplier
            
            # Extra boost if this pod has the last available quantity
            if current_qty > 0 and pods_with_available_qty <= 2:
                priority_score += 2.0  # Critical last instance boost
            
            # Extra boost for zero-quantity pods that had high capacity
            elif current_qty == 0 and limit_qty > 0:
                priority_score += 1.0  # Boost for pods that previously had the SKU
            
            candidate_pods.append({
                'pod': pod,
                'sku_utilization': sku_utilization,
                'pod_utilization': pod_utilization,
                'priority_score': priority_score,
                'scarcity_multiplier': scarcity_multiplier,
                'replenishment_need': limit_qty - current_qty
            })
        
        if not candidate_pods:
            print(f"No suitable candidate pods found for SKU {sku}")
            return None
            # # Fallback: try to find any pod with this SKU (even if busy) for critical situations
            # print(f"No idle pods found for SKU {sku}, checking busy pods as fallback...")
            #
            # for pod in pods_with_sku:
            #     if pod is None or not hasattr(pod, 'skus') or sku not in pod.skus:
            #         continue
            #
            #     sku_data = pod.skus[sku]
            #     limit_qty = sku_data.get('limit_qty', 0)
            #     current_qty = sku_data.get('current_qty', 0)
            #
            #     if limit_qty <= 0:
            #         continue
            #
            #     # For busy pods, still prioritize by need and capacity
            #     sku_utilization = current_qty / limit_qty if limit_qty > 0 else 0
            #     priority_score = (1 - sku_utilization) * scarcity_multiplier
            #
            #     candidate_pods.append({
            #         'pod': pod,
            #         'sku_utilization': sku_utilization,
            #         'pod_utilization': 0.0,  # Unknown for busy pods
            #         'priority_score': priority_score,
            #         'scarcity_multiplier': scarcity_multiplier,
            #         'replenishment_need': limit_qty - current_qty,
            #         'is_busy': True
            #     })
            #
            # if not candidate_pods:
            #     print(f"No suitable candidate pods found for SKU {sku} (checked both idle and busy pods)")
            #     return None
            # else:
            #     print(f"Found {len(candidate_pods)} busy pods as fallback candidates for SKU {sku}")
            
        # Sort by priority score (higher is better)
        candidate_pods.sort(key=lambda x: x['priority_score'], reverse=True)
        
        best_candidate = candidate_pods[0]
        print(f"Selected pod {best_candidate['pod'].pod_number} for SKU {sku} replenishment "
              f"(SKU util: {best_candidate['sku_utilization']:.2f}, "
              f"Pod util: {best_candidate['pod_utilization']:.2f}, "
              f"Scarcity: {best_candidate['scarcity_multiplier']:.2f}, "
              f"Priority: {best_candidate['priority_score']:.2f}, "
              f"Pods w/ qty: {pods_with_available_qty})")
        
        return best_candidate['pod']
    
    def determinePodWillBeReplenished(self, replenished_pod_needed_by_sku):
        stock_out_probability_of_each_pod = {}

        all_pods = sum(replenished_pod_needed_by_sku.values(), [])
        unique_pods = set(all_pods)
        unique_pods_list = list(unique_pods)

        for pod in unique_pods_list: 
            skus_in_pod = pod.skus
            pod_stock_out_probability = 0
            for sku in skus_in_pod:
                sku_current_qty = sku['current_qty']
                # Get the max amount of the SKU that have been ordered
                # Sum the probability from the probability of sku_current_qty until probability of max qty in the sku
            # Put in the result of the sum probability of each pod to the stock_out_probability_of_each_pod
        # Return the pod.pod_number with the highest value of stock_out_probability_of_each_pod

    def checkWarehouseSKUPodPolicy(self, pod: Pod, UL: float, KL: float):
        """
        New Policy: Warehouse Inventory SKU Pod Policy
        Phase 1: Analyze the warehouse inventory level of each SKU
        Phase 2: Calculate pod replenishment index for flagged SKUs
        """
        # Phase 1: Analyze the warehouse inventory level of each SKU
        flagged_skus = self.getFlaggedSKUs(UL, pod) # Pass pod to getFlaggedSKUs to check SKUs in the pod
        if len(flagged_skus) == 0:
            print("No flagged SKUs found")
            return False
        
        # Phase 2: Calculate pod replenishment index for flagged SKUs
        pod_replenishment_index = pod.calculatePodReplenishmentIndex(flagged_skus)
        print(f"Flagged_skus: {len(flagged_skus)}")
        print(f"Pod replenishment index (Qp): {pod_replenishment_index:.3f}")
        print(f"Pod threshold (KL): {KL}")

        return pod_replenishment_index > KL
    
    def getFlaggedSKUs(self, UL: float, pod: Pod = None):
        """
        Get the flagged SKUs from the pod
        Phase 1: Identify SKUs that need replenishment based on warehouse inventory level
        Return: List of flagged SKUs where Ui < UL (Oi=1, True)
        """
        flagged_skus = []

        # If pod is specified, only check SKUs in the pod
        skus_to_check = pod.skus.keys() if pod else self.skus_data.keys()
        
        for sku_id in skus_to_check:
            if sku_id in self.skus_data:
                sku_data = self.skus_data[sku_id]
                # Calculate the utilzation ratio: Ui =Si/Xi
                current_qty = int(sku_data['current_global_qty'])
                max_qty = int(sku_data['max_global_qty'])
            
                if max_qty > 0: # Avoid division by zero
                    utilization_ratio = current_qty / max_qty
                    if utilization_ratio < UL:
                        flagged_skus.append(sku_id)
                #         print(f"SKU {sku_id} is FLAGGED because Ui < UL (Utilization ratio: {utilization_ratio:.3f} < {UL:.3f})")
                # else:
                #     print(f"Warning: Max quantity for SKU {sku_id} is 0")
                
        return flagged_skus
    
                
    def _distancePodToRobot(self, pod_coordinate, robots_coordinate):
        pod_coordinate = np.array(pod_coordinate).reshape(1, -1)
        distance_to_robot_score = 1000
        robots_coordinate = np.array(robots_coordinate)
        if len(robots_coordinate) == 0:
            return distance_to_robot_score

        distances = manhattan_distances(pod_coordinate, robots_coordinate)
        distance_to_robot_score = float(np.min(distances))
        
        return distance_to_robot_score
    
    def _countFulfillment(self, skus_in_station_dict, pod_skus):
        total_fulfillment = 1
        pod_skus_copy = {
            sku: details.get("current_qty", 0)
            for sku, details in pod_skus.items()
        }
        for sku in skus_in_station_dict:
            for order_qty in skus_in_station_dict[sku]:
                if sku in pod_skus_copy and pod_skus_copy[sku] >= order_qty:
                    pod_skus_copy[sku] -= order_qty
                    total_fulfillment += 1
                else: 
                    continue

        return total_fulfillment
    
    def checkAdaptiveReplenishmentPolicy(self, pod: Pod, UL: float = 0.6) -> bool:
        """
        Adaptive Replenishment Policy
        Phase 1: Analyze the warehouse inventory level of each SKU
        Phase 2: Calculate pod replenishment index using class-based thresholds
        Args:
            pod: The pod to check for replenishment
            UL: Upper limit threshold for warehouse inventory (default: 0.6)
        """
        if not hasattr(self.warehouse, 'adaptive_policy'):
            print("Warning: Adaptive policy not initialized, falling back to baseline policy")
            return self.checkWarehouseSKUPodPolicy(pod, UL, UL)
        
        # Phase 1: Analyze the warehouse inventory level of each SKU
        flagged_skus = self.getFlaggedSKUs(UL)
        if len(flagged_skus) == 0:
            print("No flagged SKUs found")
            return False
        
        # Phase 2: Calculate pod replenishment using adaptive policy
        try:
            assessment = self.warehouse.adaptive_policy.process_pod(pod, self.skus_data)
            
            print(f"Pod {pod.pod_number} Adaptive Assessment:")
            print(f"  Threshold: {assessment.threshold:.3f}")
            print(f"  Priority: {assessment.priority}")
            print(f"  Current Fill: {assessment.current_fill_level:.3f}")
            print(f"  Needs Replenishment: {assessment.needs_replenishment}")
            print(f"  SKUs to replenish: {len(assessment.skus_to_replenish)}")
            
            if assessment.skus_to_replenish:
                print("  Priority SKUs:")
                for sku_assess in assessment.skus_to_replenish[:5]:  # Show top 5
                    print(f"    SKU {sku_assess.sku_id}: Priority {sku_assess.priority}, Ratio {sku_assess.inv_ratio:.3f}")
            
            return assessment.needs_replenishment
            
        except Exception as e:
            print(f"Error in adaptive policy assessment: {e}")
            return self.checkWarehouseSKUPodPolicy(pod, UL, UL)  # Fallback to baseline
    
    def loadPodsFromCSV(self): # NEW FUNCTION TO CONNECT THE PODS.CSV WITH THE COORDINATE
        # Step 1: Build pod_id -> (x, y) mapping from generated_pod.csv, where value 1 marks a pod location
        pod_id_to_coord = {}
        generated_pod_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_pod.csv')
        if os.path.exists(generated_pod_path):
            pod_grid = pd.read_csv(generated_pod_path, header=None).values
            pod_id_counter = 0
            for y, row in enumerate(pod_grid):
                for x, val in enumerate(row):
                    try:
                        if int(val) == 1:
                            pod_id_to_coord[pod_id_counter] = (x, y)
                            pod_id_counter += 1
                    except Exception:
                        continue
        else:
            # Fallback: use in-memory pods if already created
            for pod in self.pods:
                pod_id_to_coord[pod.pod_number] = (pod.pos_x, pod.pos_y)

        # Step 2: Load pods.csv and assign SKUs/slots to pods
        pods_csv_path = os.path.join(PARENT_DIRECTORY, 'data/output/pods.csv')
        if os.path.exists(pods_csv_path):
            pods_df = pd.read_csv(pods_csv_path)
            for _, row in pods_df.iterrows():
                pod_id = int(row['pod_id'])
                sku = int(row['item_id'])
                qty = int(row['qty'])
                max_qty = int(row['max_qty'])
                item_weight = float(row['item_weight'])
                # Get pod coordinates
                x, y = pod_id_to_coord.get(pod_id, (0, 0))
                # Find or create pod
                pod = next((p for p in self.pods if p.pod_number == pod_id), None)
                if pod is None:
                    pod = Pod(pod_id, x, y)
                    self.pods.append(pod)
                    self.coordinate_to_pods[(x, y)] = pod
                # Add SKU to pod
                pod.addSKU(sku, max_qty, qty, item_weight)
                # Register pod in sku_to_pods
                if sku not in self.sku_to_pods:
                    self.sku_to_pods[sku] = []
                if pod not in self.sku_to_pods[sku]:
                    self.sku_to_pods[sku].append(pod)
