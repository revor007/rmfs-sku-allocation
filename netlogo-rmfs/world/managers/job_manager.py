from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
from world.entities.job import Job
from lib.types.netlogo_coordinate import NetLogoCoordinate
if TYPE_CHECKING:
    from world.warehouse import Warehouse

class JobManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.jobs: List[Job] = []
        self.job_counter = 0

    def createJob(self, pod_coordinate: NetLogoCoordinate, station_id, pod, skus_for_replenishment: Optional[list] = None):
        obj = Job(
            self.job_counter,
            pod_coordinate,
            station_id,
            pod,
            skus_for_replenishment=skus_for_replenishment,
        )
        self.jobs.append(obj)
        self.job_counter += 1
        return obj
