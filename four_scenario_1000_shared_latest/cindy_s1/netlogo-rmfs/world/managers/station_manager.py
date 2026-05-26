from __future__ import annotations
from typing import List, Optional, Dict, TYPE_CHECKING
from world.entities.station import Station
from world.entities.picker import Picker
from world.entities.replenishment import Replenishment
from .pod_manager import PodManager
import pandas as pd
import numpy as np
import random

if TYPE_CHECKING:
    from world.warehouse import Warehouse

class StationManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.picker_counter = 0
        self.picking_stations: List[Station] = []
        self.replenishment_counter = 0
        self.replenishment_stations: List[Station] = []
        self.stations_by_id: Dict[int, Station] = {}

    def initStationManager(self):
        for station in self.getAllStations():
            station.setStationManager(self)

    def getAllStations(self):
        return self.picking_stations + self.replenishment_stations
    
    def getStationById(self, station_id):
        return self.stations_by_id[station_id]
    
    def addStation(self, station: Station):
        self.stations_by_id[station.station_id] = station

        if station.isPickerStation():
            self.picking_stations.append(station)
        elif station.isReplenishmentStation():
            self.replenishment_stations.append(station)

    def createPickerStation(self, x: int, y: int, data: pd.DataFrame):
        obj = Picker(self.picker_counter, x, y, data)
        self.picker_counter += 1
        self.addStation(obj)
    
    def createReplenishmentStation(self, x: int, y: int, data: pd.DataFrame):
        obj = Replenishment(self.replenishment_counter, x, y, data)
        self.replenishment_counter += 1
        self.addStation(obj)
    
    def findAvailablePickingStation(self) -> Optional[Station]:
        # Filter stations that have capacity
        candidate_stations = [station for station in self.picking_stations if len(station.order_ids) < station.max_orders]

        if not candidate_stations:
            return None

        # Calculate pod counts for each station (replace with actual pod tracking logic)
        pod_counts = {station: len(station.incoming_pod) for station in candidate_stations}

        # Find the minimum number of pods assigned among candidates
        min_pods = min(pod_counts.values())

        # Select all stations that have the minimum number of pods
        least_loaded_stations = [
            station for station in candidate_stations 
            if pod_counts[station] == min_pods
        ]

        # Randomly pick one to evenly distribute pods
        return random.choice(least_loaded_stations)
    
    def findAvailableReplenishmentStation(self) -> Optional[Station]:
        # Initialize the available station variable as None
        available_station = None
        # Initialize the minimum number of orders to a high value to find the station with the least orders
        min_orders = float('inf')

        # Iterate through each station to check the number of orders
        for station in self.replenishment_stations:
            if len(station.robot_ids) < station.max_orders:
                # Check if this station has fewer orders than the current minimum
                if len(station.robot_ids) < min_orders:
                    min_orders = len(station.robot_ids)
                    available_station = station

        return available_station

    def findHighestSimilarityStation(self, skus_in_order, pod_manager: PodManager) -> Optional[Station]:
        sku_in_order_list = list(skus_in_order)
        available_station = [
            station
            for station in self.picking_stations
            if len(station.order_ids) < station.max_orders
        ]

        if len(available_station) == 1:
            return available_station[0]

        if len(available_station) <= 0:
            return None

        station_rankings = []
        for station in available_station:
            station_pod_skus_set = set()
            for pod_id in station.incoming_pod:
                pod = pod_manager.getPodByNumber(pod_id)
                if pod is None:
                    continue
                for item, details in pod.skus.items():
                    if details['current_qty'] > 0:
                        station_pod_skus_set.add(item)

            similarity_score = sum(1 for sku in sku_in_order_list if sku in station_pod_skus_set)
            station_rankings.append(
                (
                    station,
                    similarity_score,
                    len(station.order_ids),
                    len(station.incoming_pod),
                )
            )

        if not station_rankings:
            return None

        max_similarity = max(ranking[1] for ranking in station_rankings)
        if max_similarity <= 0:
            return self.findAvailablePickingStation()

        best_rankings = [
            ranking for ranking in station_rankings if ranking[1] == max_similarity
        ]
        min_order_count = min(ranking[2] for ranking in best_rankings)
        best_rankings = [
            ranking for ranking in best_rankings if ranking[2] == min_order_count
        ]
        min_incoming_pods = min(ranking[3] for ranking in best_rankings)
        best_rankings = [
            ranking for ranking in best_rankings if ranking[3] == min_incoming_pods
        ]

        return random.choice(best_rankings)[0]

    
