from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List, Any
import torch
import torch.nn as nn
import gc
from contextlib import asynccontextmanager
import numpy as np
from mdrl_routing import MultiAgentPPO
import random
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelState:
    def __init__(self):
        self.model = None
        self.step = 1

    async def initialize_workflow(self):
        if self.model:
            await self.cleanup_workflow()
        # self.model = MultiAgentA2C().to(device)

    async def cleanup_workflow(self):
        self.clear_gpu_memory()

    @staticmethod
    def clear_gpu_memory():
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

model_state = ModelState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_state.initialize_workflow()
    yield
    await model_state.cleanup_workflow()

app = FastAPI(lifespan=lifespan)

# -------------------------------
# Request Models
# -------------------------------

class InitializeInput(BaseModel):
    num_agents:int
    state_dim:int 
    map_state_dim:Any
    action_dim:int
    lambd_0:float
    lambd_scheduler_alpha:float
    actor_lr:float
    critic_lr:float
    gamma:float

class PredictInput(BaseModel):
    previous_rl_state: Any
    action_masking: Any
    previous_rl_map_state: Any
    deterministic: bool

class ReplayBuffer(BaseModel):
    previous_rl_state: Any
    actions: Any
    rewards: Any
    rl_state: Any
    dones: Any
    log_pis: Any
    value: Any
    previous_rl_map_state: Any
    rl_map_state: Any
    previous_action_masks: Any
    action_masks: Any

class TrainInput(BaseModel):
    step: int

# -------------------------------
# Initialization Endpoint
# -------------------------------

@app.post("/initialize")
def CreateNewModel(input: InitializeInput):
    if not model_state.model:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        model = MultiAgentPPO(
            input.num_agents, 
            input.state_dim, 
            input.map_state_dim,
            input.action_dim,
            hidden_dim=64,
            lambd_0=input.lambd_0,
            lambd_scheduler_alpha=input.lambd_scheduler_alpha,
            actor_lr=input.actor_lr,
            critic_lr=input.critic_lr,
            gamma=input.gamma
        )

        model.load_model(model.model_path)
        model_state.model = model
        return {"status": "Model initialized"}
    
    else:
        return {"status": "Model initialized"}

# -------------------------------
# Predict Endpoint
# -------------------------------

@app.post("/predict")
def predict(input: PredictInput):
    if not model_state.model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Convert inputs to correct types
        obs = np.array(input.previous_rl_state, dtype=np.float32)
        mask = input.action_masking
        map_state = np.array(input.previous_rl_map_state, dtype=np.float32)

        actions, log_pis = model_state.model.getAction(obs, mask, map_state, input.deterministic)

        obs = torch.tensor(obs, dtype=torch.float32, device=model_state.model.device).unsqueeze(0)
        actions = torch.tensor(actions, dtype=torch.long, device=model_state.model.device).unsqueeze(0)
        log_pis = torch.tensor(log_pis, dtype=torch.float32, device=model_state.model.device)

        flat_obs = torch.tensor(obs, dtype=torch.float32, device=model_state.model.device).view(-1).unsqueeze(0)
        map_state_tensor = torch.tensor(map_state, dtype=torch.float32, device=model_state.model.device).unsqueeze(0)
        
        # === Compute value estimates per agent ===
        B, num_agents, state_dim = obs.shape

        own_states = obs.reshape(-1, state_dim)                    # (B * num_agents, state_dim)
        actions_flat = actions.reshape(-1)                         # (B * num_agents,)
        agent_ids = torch.arange(num_agents, device=obs.device).repeat(B)  # (B * num_agents,)
        flat_obs_expanded = flat_obs.repeat_interleave(num_agents, dim=0)         # (B * num_agents, num_agents * state_dim)
        map_state_expanded = map_state_tensor.unsqueeze(1).repeat(1, num_agents, 1, 1).reshape(-1, 49, 31)  # (B * num_agents, 49, 31)

        values_flat = model_state.model.critic(
            own_state=own_states,
            collective_state=flat_obs_expanded,
            action=actions_flat,
            map_state=map_state_expanded,
            agent_id=agent_ids
        ).squeeze(-1)  # (B * num_agents,)

        values = values_flat.view(B, num_agents)

        # Convert tensors to Python native types for JSON serialization
        actions_list = actions.cpu().tolist()        # shape: (B, num_agents)
        log_pis_list = log_pis.cpu().tolist()        # shape: (B, num_agents)
        values_list = values.cpu().tolist()          # shape: (B, num_agents)

        return {
            "actions": actions_list,
            "log_pis": log_pis_list,
            "value": values_list
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Predict error: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Remember Endpoint
# -------------------------------

@app.post("/remember")
def remember(input: ReplayBuffer):
    try:
        model_state.model.store_step(
            input.previous_rl_state,
            input.actions,
            input.rewards,
            input.rl_state,
            input.dones,
            input.log_pis,
            input.value,
            input.previous_rl_map_state,
            input.rl_map_state,
            input.previous_action_masks,
            input.action_masks
        )

        return {"status": "memory stored"}

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Train Endpoint
# -------------------------------

@app.post("/train")
def train(input: TrainInput):
    try:
        if model_state.model.store_step_iteration % model_state.model.rollout_length == 0 and model_state.model.store_step_iteration > 0:
            print("Starting Learning")
            try:
                model_state.model.train_on_rollout()
            except Exception as replay_error:
                print("Replay error:", traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Replay error: {replay_error}")
            print("Finished Learning")
            model_state.model.save_model()
        return {"status": "training step complete"}
    except Exception as e:
        print("Outer error:", traceback.format_exc())  # <--- this will print full traceback
        raise HTTPException(status_code=500, detail=str(e))



# -------------------------------
# Get Memory Endpoint
# -------------------------------
@app.post("/getmemory")
def get_memory():
    if not model_state.model or len(model_state.model.rollout["states"]) < 2:
        raise HTTPException(status_code=400, detail="Not enough memory stored")

    current_state = model_state.model.rollout["states"][-1]
    previous_state = model_state.model.rollout["states"][-2]
    return {
        "status": "Memory extracted",
        "current_state": current_state,
        "previous_state": previous_state
    }

# -------------------------------
# Get Property Endpoint
# -------------------------------
@app.get("/getproperty")
def get_property():
    return {
        "status" : "Memory extracted",
        "gamma" : model_state.model.gamma,
        "gamma_n" : model_state.model.gamma_n,
        "num_steps": model_state.model.num_steps
    }
    
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "workflow_initialized": model_state.model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)