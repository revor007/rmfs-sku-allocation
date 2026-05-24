import os

TICK_TO_SECOND = 0.15
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PARENT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)

actor_lr = [1e-6, 5e-5, 25e-5]
critic_lr = [1e-6, 5e-5, 25e-5]
lambd_0 = [0.5, 0.9, 0.99]
lambd_scheduler_alpha = [1e-5, 1.0, 1e5]
gamma = [0.99916, 0.999583, 0.9997917] # 180s 360s

TEST_HYPERPARAMETERS = {
        "actor_lr": actor_lr[0],
        "critic_lr": critic_lr[0],
        "lambd_0": lambd_0[1],
        "lambd_scheduler_alpha": lambd_scheduler_alpha[1],
        "gamma": gamma[0]
    }

PARENT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
