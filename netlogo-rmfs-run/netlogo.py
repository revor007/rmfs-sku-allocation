import pickle
import os
import traceback
from typing import List
import time

import cProfile
import pstats

from lib.generator.warehouse_generator import *
from pip._internal import main as pipmain
from lib.file import *
from world.warehouse import Warehouse


def setup():
    global warehouse
    try:
        # Initialize the simulation warehouse
        assignment_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        if os.path.exists(assignment_path):
            os.remove(assignment_path)
        warehouse = Warehouse()

        # if(os.path.exists(PARENT_DIRECTORY + "/data/output/generated_pod.csv")):
        #     # delete
        #     os.remove(PARENT_DIRECTORY + "/data/output/generated_pod.csv")
        # if(os.path.exists(PARENT_DIRECTORY + "/data/output/pods.csv")):
        #     # delete
        #     os.remove(PARENT_DIRECTORY + "/data/output/pods.csv")
        # if(os.path.exists(PARENT_DIRECTORY + "/data/output/items.csv")):
        #     # delete
        #     os.remove(PARENT_DIRECTORY + "/data/output/items.csv")
        # if(os.path.exists(PARENT_DIRECTORY + "/data/output/items_slots_configuration.csv")):
        #     # delete
        #     os.remove(PARENT_DIRECTORY + "/data/output/items_slots_configuration.csv")
        if(os.path.exists(PARENT_DIRECTORY + "/data/output/generated_database_order.csv")):
            # delete
            os.remove(PARENT_DIRECTORY + "/data/output/generated_database_order.csv")
        if(os.path.exists(PARENT_DIRECTORY + "/data/output/generated_order.csv")):
            # delete
            os.remove(PARENT_DIRECTORY + "/data/output/generated_order.csv")


        print("AKU HADIRRRRR")
        
        # Populate the warehouse with objects and connections
        draw_layout(warehouse)
        # print(warehouse.intersection_manager.intersections[0].intersection_coordinate)
        
        # Generate initial results
        next_result = warehouse.generateResult()
        
        warehouse.initWarehouse()
        warehouse.robot_manager.deterministic_rl = True

        return next_result

    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."

def tick():
    try:
        global warehouse
        # print("========tick========")

        print("before tick", warehouse._tick)
        # Perform a simulation tick
        warehouse.tick()

        # Stop when the actual-order replay has fully drained.
        if warehouse.isSimulationComplete():
            return "STOP"
        
        # Generate results after the tick
        next_result = warehouse.generateResult()
            
        # Calculate energy efficiency (energy per order)
        energy_efficiency = warehouse.total_energy / warehouse.orders_fulfilled if warehouse.orders_fulfilled > 0 else 0
        energy_efficiency_fixed = warehouse.total_fixed_load_energy / warehouse.orders_fulfilled if warehouse.orders_fulfilled > 0 else 0
        
        print("visiting pod:", warehouse.pod_visit_to_station)
        
        return [next_result, warehouse.total_energy, len(warehouse.job_queue), warehouse.stop_and_go, 
                warehouse.total_turning, warehouse.replenishment_count, warehouse.replenishment_trips,
                warehouse.pod_visit_to_station, warehouse.orders_fulfilled, warehouse.average_inventory_level,
                energy_efficiency, warehouse.average_pod_inventory_level, warehouse.average_weighted_pod_utilization, warehouse.total_fixed_load_energy,
                energy_efficiency_fixed]
    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."

    except Exception as e:
        traceback.print_exc()
        return "An error occurred. See the details above."

def setup_py():
    def install_package(package_name):
        """Install a Python package using pip."""
        pipmain(['install', package_name])

    # List of packages to install
    packages = ["networkx", "matplotlib"]

    # Install each package
    for package in packages:
        install_package(package)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    result = setup()
    try:
        for _ in range(192000):
            result = tick()
            if result == "STOP":
                break
    except Exception as e:
        traceback.print_exc()
        print("An error occurred during the tick process. See the details above.")

    profiler.disable()  # Stop profiling

    # Print the profiling results, sorted by time taken
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumtime").print_stats(10)  
    # Top 10 functions

    # with open('result.txt', 'w') as result_file:
    #     result_file.write(str(result))
