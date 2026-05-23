from lib.types.netlogo_coordinate import NetLogoCoordinate
import csv
import os
from lib.constant import *
from typing import Optional

class Job:
    def __init__(self, id: int, pod_coordinate: NetLogoCoordinate, station_id, pod, skus_for_replenishment: Optional[list] = None):
        self.id = id
        self.pod_coordinate = pod_coordinate
        self.pod_return_coordinate = NetLogoCoordinate(0,0)
        self.pod_pickup_coordinate = NetLogoCoordinate(0,0)
        self.original_robot_coordinate = NetLogoCoordinate(0,0)
        self.pickup_distance = 0
        self.pod = pod
        self.station_id = station_id
        self.orders = []  # This will hold tuples of (order_id, sku, quantity)
        # Delay values are in simulation steps, not replay-time ticks.
        self.picking_delay_per_sku = float(os.getenv("RMFS_PICKING_DELAY_PER_SKU", "3.5"))
        self.picking_delay = 0
        # 3.5 steps ~= 0.525 replay minutes because each step advances _tick by 0.15.
        self.replenishment_delay_per_sku = float(os.getenv("RMFS_REPLENISHMENT_DELAY_PER_SKU", "3.5"))
        self.replenishment_delay = 0 # Ganti
        self.replenishment_delay_total = 0
        self.is_finished = False
        self.skip_count = 0
        self.skus_for_replenishment = skus_for_replenishment

    def __str__(self):
        return f"Job: {self.id}, {self.pod_coordinate}, {self.station_id}, {self.orders}"

    def __repr__(self):
        return self.__str__()
    
    def addPickingTask(self, order_id, sku, quantity):
        """Add an order with the specific SKU and quantity to be picked."""
        self.orders.append((order_id, sku, quantity))
        self.picking_delay += self.picking_delay_per_sku
    
    def addReplenishmentTask(self, pod, sku_ids: Optional[list] = None):
        if sku_ids is None:
            total_skus = len(pod.skus)
        else:
            total_skus = len(sku_ids)

        added_delay = total_skus * self.replenishment_delay_per_sku
        self.replenishment_delay += added_delay
        self.replenishment_delay_total += added_delay

    def isBeingProcessed(self):
        """Check if the job is being processed based on delays."""
        return self.picking_delay > 0 or self.replenishment_delay > 0

    def decrementDelay(self):
        """Decrement the picking or replenishment delay."""
        if self.picking_delay > 0:
            self.picking_delay = max(0, self.picking_delay - 1)
        elif self.replenishment_delay > 0:
            self.replenishment_delay = max(0, self.replenishment_delay - 1)

    def popOrder(self):
        return self.orders.pop(0)
    
    def writePodReturnReport(self, drop_off_distance):
        log_file = os.path.join(PARENT_DIRECTORY, "data", "output", "pod_return_log.csv")
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(["Job ID", "Pod", "Original Robot (Before Pickup) Coordinate", "Pod Pickup Coordinate", "Pod Return Coordinate", "Station ID", "Pickup Distance based on Path Planning", "Dropoff Distance based on Path Planning"])
            
            # Write the data
            writer.writerow([
                self.id,
                self.pod,
                self.original_robot_coordinate,
                self.pod.original_pod_coordinate,
                self.pod_return_coordinate,
                self.station_id,
                self.pickup_distance,
                drop_off_distance
            ])
