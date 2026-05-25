from typing import TYPE_CHECKING, List
from world.entities.object import Object
from lib.types.netlogo_coordinate import NetLogoCoordinate
if TYPE_CHECKING:
    from world.managers.pod_manager import PodManager

class Pod(Object):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, 'pod', x, y)
        self.pod_manager: PodManager = None
        self.original_pod_coordinate = NetLogoCoordinate(0, 0)
        self.pod_number = id
        self.shape = 'full square'
        self.skus = {}
        self.is_idle = True
        self.being_carried = False
        self.station = None
        self.need_replenishment = False
        self.velocity = 0
        self.acceleration = 0
        self.mass = 0  # Base mass of empty pod
        self.sku_weights = {}  # Dictionary to store weight per unit for each SKU
        self.current_load_mass = 0  # Current total mass of items in pod
        self.is_awaiting_replenishment = False
        self.has_pending_replenishment_dispatch = False
        self.must_replenish_before_pick = False
        

    def __eq__(self, other):
        if isinstance(other, Pod):
            return self.pod_number == other.pod_number
        return False

    def __hash__(self):
        return hash(self.pod_number)

    def __repr__(self):
        return f"Pod({self.pod_number})"

    def setPodManager(self, pod_manager):
        self.pod_manager = pod_manager

    # def addSKU(self, sku, limit_qty, current_qty, threshold, item_weight=None):
    def addSKU(self, sku, limit_qty, current_qty, threshold, item_weight=None):
        """Add a new SKU with its limit, current quantity, threshold, and weight per unit."""
        if sku in self.skus:
            self.skus[sku]['limit_qty'] += limit_qty
            self.skus[sku]['current_qty'] += current_qty
            self.skus[sku]['threshold'] = threshold
        else:
            self.skus[sku] = {
                'limit_qty': limit_qty,
                'current_qty': current_qty,
                'threshold': threshold
            }
        if item_weight is not None:
            self.sku_weights[sku] = item_weight
            self.current_load_mass += item_weight * current_qty

    def calculateCurrentLoadMass(self):
        """Calculate the current total mass of all items in the pod."""
        total_mass = 0
        for sku, details in self.skus.items():
            if sku in self.sku_weights:
                total_mass += self.sku_weights[sku] * details['current_qty']
        return total_mass

    def recalculateCurrentLoadMass(self):
        """Recalculate and update the current load mass based on current quantities."""
        self.current_load_mass = self.calculateCurrentLoadMass()

    def getTotalMass(self):
        """Get the total mass of the pod including its base mass and current load."""
        # Recalculate to ensure accuracy
        self.recalculateCurrentLoadMass()
        return self.mass + self.current_load_mass

    def isNeedReplenishment(self):
        """Check if 50% or more SKUs are below their threshold to determine if the pod needs to move to a
        replenishment station."""
        count_below_threshold = 0
        total_skus = len(self.skus)
        alpha = total_skus / 2
        for details in self.skus.values():
            # print(f"crt {details['current_qty']} limit {details['limit_qty']} th {details['threshold']}")
            if float(details['current_qty'])/float(details['limit_qty']) <= float(details['threshold']):
                count_below_threshold += 1

        if count_below_threshold >= alpha:
            return True
        return False

    def replenishAllSKU(self):
        """Replenish all SKUs by setting each SKU's current quantity to its limit quantity."""
        for sku in self.skus:
            self.skus[sku]['current_qty'] = self.skus[sku]['limit_qty']

    def pickSKU(self, sku, qty):
        available_qty = self.skus[sku]['current_qty']
        picked_qty = min(qty, available_qty)
        self.skus[sku]['current_qty'] -= picked_qty
        return picked_qty
    #         old_qty = self.skus[sku]['current_qty']
    #         self.skus[sku]['current_qty'] = self.skus[sku]['limit_qty']
    #         # Update load mass
    #         if sku in self.sku_weights:
    #             self.current_load_mass += self.sku_weights[sku] * (self.skus[sku]['limit_qty'] - old_qty)
    
    def replenishFlaggedSKUs(self, flagged_skus):
        """
        Replenish flagged SKUs by setting each flagged SKU's current quantity to its limit quantity.
        """
        replenishment_count = 0
        for sku_id in flagged_skus:
            if sku_id in self.skus:
                old_qty = self.skus[sku_id]['current_qty']
                self.skus[sku_id]['current_qty'] = self.skus[sku_id]['limit_qty']
                # Update load mass
                # if sku_id in self.sku_weights:
                #     self.current_load_mass += self.sku_weights[sku_id] * (self.skus[sku_id]['limit_qty'] - old_qty)
                replenishment_count += 1
        # print(f"Total SKUs replenished: {replenishment_count} flagged SKUs")
        return replenishment_count

    # def pickSKU(self, sku, qty):
    #     """Pick items from a SKU and update the pod's load mass. Never allow negative inventory."""
    #     if sku in self.skus and sku in self.sku_weights:
    #         # self.skus[sku]['current_qty'] -= qty
    #         available_qty = self.skus[sku]['current_qty']
    #         reduce_qty = min(qty, available_qty)
    #         self.skus[sku]['current_qty'] -= reduce_qty
    #         # Update load mass
    #         # self.current_load_mass -= self.sku_weights[sku] * qty
    #         self.current_load_mass -= self.sku_weights[sku] * reduce_qty
    #         return reduce_qty
    #     return 0

    def getQuantity(self, sku):
        return self.skus[sku]['current_qty']

    def getUnassignedSKUs(self):
        """Return a list of SKUs that have not yet been assigned a pod."""
        unassigned_skus = [sku for sku, details in self.skus.items() if details['pod'] is None]
        return unassigned_skus

    def setPodStation(self, station):
        self.station = station
        return
    
    def removePodStation(self):
        self.station = None
        return

    def getAllSKUInPod(self):
        return self.skus
    
    def calculatePodReplenishmentIndex(self, flagged_skus: List):
        """
        Phase 2: Calculate pod replenishment index for flagged SKUs
        Qp = sum(Oi*Uip) /len total skus in pod
        """
        total_skus = len(self.skus)
        if total_skus == 0:
            return 0
        
        sum_utilization_ratios = 0
        flagged_skus_in_pod = 0

        for sku_id in flagged_skus:
            if sku_id in self.skus:
                # Calculate pod-level utilization ratio: Uip = Sip/Xip
                current_qty = int(self.skus[sku_id]['current_qty'])
                limit_qty = int(self.skus[sku_id]['limit_qty'])

                if limit_qty > 0 and current_qty > 0:
                    utilization_ratio_pod = current_qty / limit_qty
                    sum_utilization_ratios += utilization_ratio_pod
                    flagged_skus_in_pod += 1
        # Calculate Qp = 1/n * sum(Oi*Uip)
        pod_replenishment_index = sum_utilization_ratios / (total_skus/2) #CHANGED: Half of Item_id within pod to have the lesser denominator 50% empty or stocked out/flagged

        return pod_replenishment_index

    def replenishSpecificSKUs(self, sku_ids: List[str]) -> int:
        """
        Replenish specific SKUs, limited to half of the total SKUs in the pod.
        Only the highest priority SKUs will be replenished if the number of SKUs to replenish
        exceeds the limit. Only replenishes SKUs that are properly associated with this pod.
        Args:
            sku_ids: List of SKU IDs to replenish (should be pre-sorted by priority)
        Returns:
            Number of SKUs actually replenished
        """
        # Calculate maximum number of SKUs that can be replenished
        max_skus_to_replenish = len(self.skus) // 2
        
        # If no SKUs to replenish or empty pod, return 0
        if not sku_ids or not self.skus:
            return 0
            
        # Filter SKUs to only those associated with this pod
        associated_skus = [sku_id for sku_id in sku_ids if sku_id in self.skus]
        
        # Limit the number of SKUs to replenish
        skus_to_replenish = associated_skus[:max_skus_to_replenish]
        
        replenished_count = 0
        for sku_id in skus_to_replenish:
            # Verify SKU is still in pod's SKUs (double check)
            if sku_id in self.skus:
                old_qty = self.skus[sku_id]['current_qty']
                self.skus[sku_id]['current_qty'] = self.skus[sku_id]['limit_qty']
                # Update load mass
                if sku_id in self.sku_weights:
                    self.current_load_mass += self.sku_weights[sku_id] * (self.skus[sku_id]['limit_qty'] - old_qty)
                replenished_count += 1
        
        print(f"Pod {self.pod_number} replenishment:")
        print(f"  - Total SKUs in pod: {len(self.skus)}")
        print(f"  - SKUs requested for replenishment: {len(sku_ids)}")
        print(f"  - SKUs associated with pod: {len(associated_skus)}")
        print(f"  - SKUs actually replenished: {replenished_count}/{len(skus_to_replenish)} (max: {max_skus_to_replenish})")
        
        return replenished_count
