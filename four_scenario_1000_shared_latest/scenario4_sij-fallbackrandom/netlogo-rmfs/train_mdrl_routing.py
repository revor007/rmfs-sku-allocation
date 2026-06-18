import pickle
import os
import traceback

import requests

from lib.generator.warehouse_generator import *
from pip._internal import main as pipmain
from lib.file import *
from world.warehouse import Warehouse

# 243 combinations of hyperparameters
actor_lr = [1e-6, 5e-5, 25e-5]
critic_lr = [1e-6, 5e-5, 25e-5]
lambd_0 = [0.5, 0.9, 0.99]
lambd_scheduler_alpha = [1e-5, 1.0, 1e5]
gamma = [0.99916, 0.999583, 0.9997917] # 180s 360s

def setup(hyperparameters=None):
    global warehouse
    try:
        assignment_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        if os.path.exists(assignment_path):
            os.remove(assignment_path)
        warehouse = Warehouse()

        draw_layout(warehouse)
        next_result = warehouse.generateResult()
        
        warehouse.initWarehouse(type="train", hyperparameters=hyperparameters);
        return next_result

    except Exception as e:
        traceback.print_exc()
        return "An error occurred. See the details above."
    
def tick(hyperparameters):
    global warehouse
    import time
    try:            
        start_time = time.time()
        warehouse.tick()
        next_result = warehouse.generateResult()
        print("Warehouse tick:", time.time()-start_time)
        return [next_result, warehouse.total_energy, len(warehouse.job_queue), warehouse.stop_and_go,
                warehouse.total_turning]

    except Exception as e:
        traceback.print_exc()
        raise
    
def train(hyperparameters: dict):
    timestep = 0
    while timestep < 5000000:
        training_buffer = None
        for i in range(100):
            setup(hyperparameters=hyperparameters)
            print(f"Starting episode {i}")
            try:
                warehouse.robot_manager.memory = training_buffer
            except:
                pass
            for j in range(100000):
                try:
                    tick(hyperparameters=hyperparameters)
                    timestep += 1
                except Exception as e:
                    print(f"Error during tick at episode {i}, step {j}. Training aborted.")
                    break  # or use: raise if you want full traceback
                training_buffer = warehouse.robot_manager.memory

                if warehouse.robot_manager.rl_done == 1:
                    break
    
if __name__ == "__main__":
    # # Hyperparameter grid search 
    # for actor_lr_value in actor_lr:
    #     for critic_lr_value in critic_lr:
    #         for lambd_0_value in lambd_0:
    #             for lambd_scheduler_alpha_value in lambd_scheduler_alpha:
    #                 for gamma_value in gamma:
    #                     hyperparameters = {
    #                         "actor_lr": actor_lr_value,
    #                         "critic_lr": critic_lr_value,
    #                         "lambd_0": lambd_0_value,
    #                         "lambd_scheduler_alpha": lambd_scheduler_alpha_value,
    #                         "gamma": gamma_value
    #                     }
    #                     print(f"Training with hyperparameters: {hyperparameters}")
    #                     train(hyperparameters)
    hyperparameters = {
        "actor_lr": actor_lr[0],
        "critic_lr": critic_lr[0],
        "lambd_0": lambd_0[1],
        "lambd_scheduler_alpha": lambd_scheduler_alpha[1],
        "gamma": gamma[0]
    }
    print(f"Training with hyperparameters: {hyperparameters}")
    train(hyperparameters)