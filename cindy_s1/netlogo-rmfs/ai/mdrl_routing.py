import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import itertools

import numpy as np
import random
from collections import deque
import os

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, map_state_dim, action_dim, num_agents, hidden_dim=64, unit_embed_dim=64, action_embed_dim=64):
        super(ActorNetwork, self).__init__()

        self.unit_embed_dim = unit_embed_dim
        self.num_agents = num_agents
        self.unit_embedding = nn.Embedding(self.num_agents, self.unit_embed_dim)

        # Original state processing layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Process collective observations from all agents
        self.collective_fc1 = nn.Linear(state_dim * num_agents, 2*hidden_dim)
        self.collective_fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        
        # Convolutional layers for map_state processing
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)  # -> (batch_size, 16, 25, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # -> (batch_size, 32, 13, 8)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> (batch_size, 64, 7, 4)
        
        # Calculate the flattened size after conv layers: 64 * 7 * 4 = 1792
        conv_output_size = 64 * 7 * 4
        self.fcmap1 = nn.Linear(conv_output_size, 896)
        self.fcmap2 = nn.Linear(896, 448)
        self.fcmap3 = nn.Linear(448, 224)
        self.fcmap4 = nn.Linear(224, 112)
        self.fcmap5 = nn.Linear(112, hidden_dim)
        
        # Final layers combining both features
        self.fc3 = nn.Linear(3*hidden_dim, 2*hidden_dim)
        self.fc4 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, action_dim)

    def forward(self, own_state, collective_state, map_state, agent_id):
        # Get unit embedding for this agent
        agent_id_tensor = torch.tensor(agent_id, dtype=torch.long, device=own_state.device)
        if own_state.dim() > 1:  # Batch processing
            agent_id_tensor = agent_id_tensor.expand(own_state.shape[0])
        unit_embed = self.unit_embedding(agent_id_tensor)
        
        x_own = F.relu(self.fc1(own_state))
        x_own = F.relu(self.fc2(x_own))

        x_collective = F.relu(self.collective_fc1(collective_state))
        x_collective = F.relu(self.collective_fc2(x_collective))

        attn_logits = torch.matmul(unit_embed, x_own.unsqueeze(-1)).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=-1)
        x_own = torch.sum(attn_weights.unsqueeze(-1) * unit_embed, dim=1)
        
        # Process map_state with conv layers
        # Add channel dimension if not present: (batch_size, 49, 31) -> (batch_size, 1, 49, 31)
        if len(map_state.shape) == 3:
            map_state = map_state.unsqueeze(1)
        
        x_map = F.relu(self.conv1(map_state))
        x_map = F.relu(self.conv2(x_map))
        x_map = F.relu(self.conv3(x_map))
        
        x_map = x_map.flatten(start_dim=1)
        x_map = F.relu(self.fcmap1(x_map))
        x_map = F.relu(self.fcmap2(x_map))
        x_map = F.relu(self.fcmap3(x_map))
        x_map = F.relu(self.fcmap4(x_map))
        x_map = F.relu(self.fcmap5(x_map))

        x_all = torch.cat((x_own, x_collective, x_map), dim=-1)
        x_all = F.relu(self.fc3(x_all))
        x_all = F.relu(self.fc4(x_all))
        return self.fc5(x_all)
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, map_state_dim, action_dim, num_agents,
                 hidden_dim=64, unit_embed_dim=64, action_embed_dim=64):
        super(CriticNetwork, self).__init__()

        self.unit_embedding = nn.Embedding(num_agents, unit_embed_dim)
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)

        # Own state + action + unit ID
        self.fc1 = nn.Linear(state_dim + action_embed_dim + unit_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Collective state MLP
        self.collective_fc1 = nn.Linear(state_dim * num_agents, 2 * hidden_dim)
        self.collective_fc2 = nn.Linear(2 * hidden_dim, hidden_dim)

        # Map CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        conv_output_size = 64 * 7 * 4
        self.fcmap1 = nn.Linear(conv_output_size, 896)
        self.fcmap2 = nn.Linear(896, 448)
        self.fcmap3 = nn.Linear(448, 224)
        self.fcmap4 = nn.Linear(224, 112)
        self.fcmap5 = nn.Linear(112, hidden_dim)

        # Final combination
        self.fc3 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, own_state, collective_state, action, map_state, agent_id):
        # Embeddings
        action_embed = self.action_embedding(action.long())
        agent_id_tensor = torch.tensor(agent_id, dtype=torch.long, device=own_state.device)
        if own_state.dim() > 1:  # Batch processing
            agent_id_tensor = agent_id_tensor.expand(own_state.shape[0])
        unit_embed = self.unit_embedding(agent_id_tensor)

        # Own-state branch
        x_self = torch.cat([own_state, action_embed, unit_embed], dim=-1)
        x_self = F.relu(self.fc1(x_self))
        x_self = F.relu(self.fc2(x_self))

        # Collective-state branch
        x_coll = F.relu(self.collective_fc1(collective_state))
        x_coll = F.relu(self.collective_fc2(x_coll))

        # Map branch
        if map_state.dim() == 3:
            map_state = map_state.unsqueeze(1)
        x_map = F.relu(self.conv1(map_state))
        x_map = F.relu(self.conv2(x_map))
        x_map = F.relu(self.conv3(x_map))
        x_map = x_map.flatten(start_dim=1)
        x_map = F.relu(self.fcmap1(x_map))
        x_map = F.relu(self.fcmap2(x_map))
        x_map = F.relu(self.fcmap3(x_map))
        x_map = F.relu(self.fcmap4(x_map))
        x_map = F.relu(self.fcmap5(x_map))

        # Combine all
        x_all = torch.cat([x_self, x_coll, x_map], dim=-1)
        x_all = F.relu(self.fc3(x_all))
        return self.fc4(x_all)


class CriticNetworkMAPPO(nn.Module):
    def __init__(self, state_dim, map_state_dim, action_dim, num_agents, hidden_dim=64, action_embed_dim=64):
        super(CriticNetwork, self).__init__()
        self.num_agents = num_agents
        self.action_embed_dim = action_embed_dim

        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)

        # Original state and action processing layers
        self.fc1 = nn.Linear(state_dim * num_agents, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim + action_embed_dim * num_agents, 2*hidden_dim)
        
        # Convolutional layers for map_state processing (same as Actor)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)  # -> (batch_size, 16, 25, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # -> (batch_size, 32, 13, 8)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> (batch_size, 64, 7, 4)
        
        # Calculate the flattened size after conv layers: 64 * 7 * 4 = 1792
        conv_output_size = 64 * 7 * 4
        self.fcmap1 = nn.Linear(conv_output_size, 896)
        self.fcmap2 = nn.Linear(896, 448)
        self.fcmap3 = nn.Linear(448, 224)
        self.fcmap4 = nn.Linear(224, 112)
        self.fcmap5 = nn.Linear(112, hidden_dim)
        
        # Final layers combining all features
        self.fc3 = nn.Linear(3*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, collective_state, collective_actions, map_state):
        # Process state and actions
        x = F.relu(self.fc1(collective_state))

        action_embeds = self.action_embedding(collective_actions.long())  # (batch_size, num_agents, action_embed_dim)
        action_embeds = action_embeds.view(collective_actions.shape[0], -1)
        x = F.relu(self.fc2(torch.cat([x, action_embeds], dim=-1)))
        
        # Process map_state with conv layers
        # Add channel dimension if not present: (batch_size, 49, 31) -> (batch_size, 1, 49, 31)
        if len(map_state.shape) == 3:
            map_state = map_state.unsqueeze(1)
        
        x_map = F.relu(self.conv1(map_state))
        x_map = F.relu(self.conv2(x_map))
        x_map = F.relu(self.conv3(x_map))
        
        # Flatten and process through FC layer
        x_map = x_map.flatten(start_dim=1)
        x_map = F.relu(self.fcmap1(x_map))
        x_map = F.relu(self.fcmap2(x_map))
        x_map = F.relu(self.fcmap3(x_map))
        x_map = F.relu(self.fcmap4(x_map))
        x_map = F.relu(self.fcmap5(x_map))
        
        # Combine all features
        x_all = torch.cat((x, x_map), dim=-1)
        x_all = F.relu(self.fc3(x_all))
        return self.fc4(x_all)

class GAE:
    def __init__(self, gamma: float, lam: float):
        self.gamma = gamma
        self.lam = lam

    def __call__(self, rewards, values, dones):
        """
        Args:
            rewards: (T, num_agents)
            values: (T + 1, num_agents)
            dones: (T, num_agents)
        Returns:
            advantages: (T, num_agents)
            returns: (T, num_agents)
        """
        T, num_agents = rewards.shape
        advantages = np.zeros((T, num_agents), dtype=np.float32)
        last_adv = np.zeros(num_agents)

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_adv = delta + self.gamma * self.lam * mask * last_adv
            advantages[t] = last_adv

        returns = advantages + values[:-1]
        return advantages, returns

class ClippedPPOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_pi: torch.Tensor, sampled_log_pi: torch.Tensor,
                advantage: torch.Tensor, clip: float) -> torch.Tensor:
        ratio = torch.exp(log_pi - sampled_log_pi)
        clipped_ratio = ratio.clamp(min=1.0-clip,
                                    max=1.0+clip)
        policy_reward = torch.min(ratio * advantage,
                                  clipped_ratio * advantage)
        self.clip_fraction = (abs((ratio-1.0)) > clip).to(torch.float).mean()
        return -policy_reward.sum()
    
class ClippedValueFunctionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, value: torch.Tensor, sampled_value: torch.Tensor, 
                sampled_return: torch.Tensor, clip: float):
        clipped_value = sampled_value + (value-sampled_value).clamp(min=-clip,max=clip)
        vf_loss = torch.max((value-sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        return 0.5 * vf_loss.sum()
        
class MultiAgentPPO():
    def __init__(self, 
                 num_agents, 
                 state_dim, 
                 map_state_dim,
                 action_dim, 
                 hidden_dim=64, 
                 lambd_0=0.95, 
                 gamma = 0.95, 
                 actor_lr=1e-3, 
                 critic_lr=1e-3, 
                 lambd_scheduler_alpha=1,
                 actor_parameter_sharing=False,
                 log_dir=None,
                 unit_embed_dim=64):
        super(MultiAgentPPO, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.map_state_dim = map_state_dim
        self.action_dim = action_dim
        self.batch_size = 2048
        self.gae = GAE(0.99, 0.95)
        self.ppo_loss = ClippedPPOLoss()
        self.value_loss = ClippedValueFunctionLoss()


        self.rollout = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_pis": [],
            "values": [],
            "map_states": [],
            "next_map_states": [],
            "action_masks": [],
            "next_action_masks": []
        }

        self.epsilon = 0.95
        self.epsilon_min = 0
        self.epsilon_decay = 0.995
        self.actor_learning_rate = actor_lr
        self.critic_learning_rate = critic_lr
        self.value_loss_coef = 1.0
        self.entropy_bonus_coef = 0.01
        self.clip_range = 0.2
        self.max_grad_norm = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_episodes = 50
        self.num_steps = 10000
        self.rollout_length = self.batch_size
        self.frameskip = 4
        self.lstm_unroll_length = 16
        self.current_step = 0
        self.store_step_iteration = 0
        self.save_every = 512
        self.episode_lengths = []  # Store lengths of completed episodes
        self.current_episode_length = 0  # Track current episode step count
        self.total_episodes = 0


        self.actor_parameter_sharing = actor_parameter_sharing
        self.shared_actor = ActorNetwork(state_dim, map_state_dim, action_dim, self.num_agents, 
                                       hidden_dim, unit_embed_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, map_state_dim, 
                                    action_dim, self.num_agents, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.shared_actor.parameters(), lr=3e-4, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, eps=1e-5)

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.num_episodes*self.num_steps)


        self.lambd0 = lambd_0
        self.lambd_scheduler_alpha = lambd_scheduler_alpha
        self._lambd = self.lambd0 if isinstance(self.lambd0, TanhLS) else TanhLS(self.lambd0, self.lambd_scheduler_alpha)

        self.gamma = gamma
        self.gamma_n = self.guidance_discount
        
        self.model_path = os.path.join(
            os.getcwd(), 'mappo_routing',f'alr-{self.actor_learning_rate}_clr-{self.critic_learning_rate}_lambd0-{self.lambd0}_lambdaplha-{self.lambd_scheduler_alpha}_gamma-{self.gamma}', 'model.pt'
                )
        
        # TensorBoard logging setup
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), 'mappo_routing',f'alr-{self.actor_learning_rate}_clr-{self.critic_learning_rate}_lambd0-{self.lambd0}_lambdaplha-{self.lambd_scheduler_alpha}_gamma-{self.gamma}', 'runs', 'mappo')
        
        self.writer = SummaryWriter(log_dir)
        self.training_step = 0  # Global training step counter
        print(f"TensorBoard logging to: {log_dir}")
        
    def store_step(self, state, action, reward, next_state, done, log_pi, value, map_state, next_map_state, action_masks, next_action_masks):
        self.rollout["states"].append(state)
        self.rollout["actions"].append(action)
        self.rollout["rewards"].append(reward)
        self.rollout["next_states"].append(next_state)
        self.rollout["dones"].append(done)
        self.rollout["log_pis"].append(log_pi)
        self.rollout["values"].append(value[0])
        self.rollout["map_states"].append(map_state)
        self.rollout["next_map_states"].append(next_map_state)
        self.rollout["action_masks"].append(action_masks)
        self.rollout["next_action_masks"].append(next_action_masks)

        self.store_step_iteration += 1
        
        # ALWAYS increment episode length, even before checking done
        self.current_episode_length += 1
        
        # Debug: Print every 100 steps to see what's happening
        if self.store_step_iteration % 100 == 0:
            print(f"Step {self.store_step_iteration}, Episode length: {self.current_episode_length}, Done: {done}")
        
        # Check if any agent is done (episode ended)
        episode_ended = False
        
        if isinstance(done, (list, tuple)):
            episode_ended = any(done)
        elif isinstance(done, np.ndarray):
            episode_ended = np.any(done)
        elif isinstance(done, (bool, int, float)):
            episode_ended = bool(done)
        else:
            # Try to convert to numpy array and check
            try:
                done_array = np.array(done)
                episode_ended = np.any(done_array)
            except:
                print(f"Warning: Could not process 'done' with type {type(done)} and value {done}")
                episode_ended = False
        
        if episode_ended:
            self.episode_lengths.append(self.current_episode_length)
            self.total_episodes += 1
            print(f"Episode {self.total_episodes} completed with length {self.current_episode_length}")
            
            # Log episode length immediately with proper step counter
            self.writer.add_scalar('Episode/Length', self.current_episode_length, self.total_episodes)
            
            # Reset episode length AFTER logging
            self.current_episode_length = 0
            
            # Log running average of last 100 episodes
            if len(self.episode_lengths) >= 100:
                avg_length = np.mean(self.episode_lengths[-100:])
                self.writer.add_scalar('Episode/Length_Avg_100', avg_length, self.total_episodes)
            
            # Also log some recent stats
            if len(self.episode_lengths) >= 10:
                recent_avg = np.mean(self.episode_lengths[-10:])
                self.writer.add_scalar('Episode/Length_Avg_10', recent_avg, self.total_episodes)



    def _calc_loss(self, samples):
        """
        IPPO implementation following equation (7) from the paper:
        L(θ,φ) = Σ(a=1 to N) [L^a(θ) + λ_critic L^a(φ)] + λ_entropy H(π^a)
        """
        
        total_policy_loss = 0
        total_value_loss = 0  
        total_entropy = 0
        log_probs = []
        
        for agent_id in range(self.num_agents):
            agent_state = samples['obs'][:, agent_id, :]
            logits = self.shared_actor(agent_state, samples['collective_obs'], 
                                    samples['map_state'], agent_id)
            dist = torch.distributions.Categorical(logits=logits)
            agent_actions = samples['actions'][:, agent_id]
            agent_log_pi = dist.log_prob(agent_actions)
            log_probs.append(agent_log_pi)
            
            # Agent's advantage (should be normalized per agent)
            agent_advantage = samples['advantages'][:, agent_id]
            agent_advantage = self._normalize(agent_advantage)

            agent_sampled_log_pi = samples['log_pis'][:, agent_id]
            
            # L^a(θ) - Policy loss for agent a (Equation 5)
            agent_policy_loss = self.ppo_loss(
                agent_log_pi.unsqueeze(1), 
                agent_sampled_log_pi.unsqueeze(1), 
                agent_advantage.unsqueeze(1), 
                self.clip_range
            )
            
            # L^a(φ) - Value loss for agent a (Equation 6) 
            agent_value = self.critic(agent_state, samples['collective_obs'], agent_actions,
                                    samples['map_state'], agent_id)
            agent_sampled_value = samples['value'][:, agent_id].unsqueeze(1)  
            agent_returns = samples['returns'][:, agent_id].unsqueeze(1)
            
            agent_value_loss = self.value_loss(
                agent_value,
                agent_sampled_value,
                agent_returns, 
                self.clip_range
            )
            
            # H(π^a) - Entropy for agent a
            agent_entropy = dist.entropy().mean()
            
            # Sum across agents (Equation 7)
            total_policy_loss += agent_policy_loss
            total_value_loss += agent_value_loss  
            total_entropy += agent_entropy

        log_pi = torch.stack(log_probs).transpose(0,1)
        
        # Final IPPO loss following Equation (7)
        # L(θ,φ) = Σ L^a(θ) + λ_critic Σ L^a(φ) + λ_entropy Σ H(π^a)
        total_loss = (total_policy_loss +  # Σ L^a(θ)
                    self.value_loss_coef * total_value_loss +  # λ_critic Σ L^a(φ)
                    self.entropy_bonus_coef * total_entropy)   # λ_entropy Σ H(π^a)

        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        
        return {
            'total_loss': total_loss,
            'policy_loss': total_policy_loss,
            'value_loss': total_value_loss,
            'entropy_bonus': total_entropy,
            'kl_divergence': approx_kl_divergence
        }
    
    def _calc_loss_mappo(self, samples):
        # Normalize advantages
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Policy loss with action masking
        log_probs = []
        entropies = []
        
        for agent_id in range(self.num_agents):
            agent_state = samples['obs'][:, agent_id, :]
            logits = self.shared_actor(agent_state, samples['collective_obs'], 
                                    samples['map_state'], agent_id)
            
            agent_masks = samples['action_masks'][:, agent_id, :]
            # Set invalid action logits to very negative values
            masked_logits = logits.clone()
            masked_logits[agent_masks == 0] = -1e8
            
            dist = torch.distributions.Categorical(logits=masked_logits)
            agent_actions = samples['actions'][:, agent_id].long()
            agent_log_pi = dist.log_prob(agent_actions)
            log_probs.append(agent_log_pi)
            entropies.append(dist.entropy())

        log_pi = torch.stack(log_probs).transpose(0, 1)
        entropy_bonus = torch.stack(entropies).mean()
        
        value = self.critic(samples['collective_obs'], samples['collective_actions'], samples['map_state'])

        policy_loss = self.ppo_loss(log_pi, samples['log_pis'], sampled_normalized_advantage, self.clip_range)
        value_loss = self.value_loss(value, samples['value'], samples['returns'], self.clip_range)

        total_loss = (policy_loss 
                    + self.value_loss_coef * value_loss 
                    - self.entropy_bonus_coef * entropy_bonus)

        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_bonus': entropy_bonus,
            'kl_divergence': approx_kl_divergence
        }

    def train_on_rollout(self, epochs=8, minibatch_size=128):
        print("Training on rollout")
        states = torch.tensor(np.stack(self.rollout["states"]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.stack(self.rollout["actions"]), dtype=torch.float32).to(self.device)
        original_rewards = np.stack(self.rollout["rewards"])
        dones = np.stack(self.rollout["dones"])
        values = np.stack(self.rollout["values"])
        log_pis = torch.tensor(np.stack(self.rollout["log_pis"]), dtype=torch.float32).to(self.device)
        map_states = torch.tensor(np.stack(self.rollout["map_states"]), dtype=torch.float32).to(self.device)
        next_map_states = torch.tensor(np.stack(self.rollout["next_map_states"]), dtype=torch.float32).to(self.device)
        
        # Include action masks if they exist in rollout
        action_masks = torch.tensor(np.stack(self.rollout["action_masks"]), dtype=torch.float32).to(self.device)

        # Apply team spirit to rewards
        team_spirit = 0.0  # You can make this a class parameter
        if team_spirit > 0.0:
            team_rewards = original_rewards.mean(axis=1, keepdims=True)  # (T, 1)
            team_rewards = np.broadcast_to(team_rewards, original_rewards.shape)  # (T, num_agents)
            
            blended_rewards = (1.0 - team_spirit) * original_rewards + team_spirit * team_rewards
            
            # Log reward statistics
            print(f"Team Spirit: {team_spirit}")
            print(f"Original rewards - Mean: {original_rewards.mean():.4f}, Std: {original_rewards.std():.4f}")
            print(f"Team rewards - Mean: {team_rewards.mean():.4f}, Std: {team_rewards.std():.4f}")
            print(f"Blended rewards - Mean: {blended_rewards.mean():.4f}, Std: {blended_rewards.std():.4f}")
        else:
            blended_rewards = original_rewards

        with torch.no_grad():
            print("Sampling for advantage")

            next_state = torch.tensor(self.rollout["next_states"][-1], dtype=torch.float32).to(self.device)  # (num_agents, state_dim)
            next_map = torch.tensor(self.rollout["next_map_states"][-1], dtype=torch.float32).to(self.device)  # (49, 31)
            next_map = next_map.unsqueeze(0)  # (1, 49, 31)
            next_collective_state = next_state.flatten().unsqueeze(0)  # (1, num_agents * state_dim)
            next_action_masks = torch.tensor(self.rollout["next_action_masks"][-1], dtype=torch.float32).to(self.device)

            # Repeat shared states per agent
            collective_state = next_collective_state.expand(self.num_agents, -1) # (num_agents, dim)
            map_state = next_map.expand(self.num_agents, -1, -1)  # avoids unnecessary memory copies
            agent_ids = torch.arange(self.num_agents, device=next_state.device)              # (num_agents,)

            # Actor forward pass: (num_agents, action_dim)
            logits = self.shared_actor(
                own_state=next_state,                       # (num_agents, state_dim)
                collective_state=collective_state,          # (num_agents, state_dim)
                map_state=map_state,                        # (num_agents, state_dim)
                agent_id=agent_ids                          # (num_agents,)
            )

            # Apply action mask
            logits = logits.clone()
            logits[next_action_masks == 0] = -1e8

            # Sample actions
            dists = torch.distributions.Categorical(logits=logits)
            sampled_actions = dists.sample()  # (num_agents,)

            # Critic value estimation
            values = self.critic(
                own_state=next_state,                       # (num_agents, state_dim)
                collective_state=collective_state,          # (num_agents, state_dim)
                action=sampled_actions,                     # (num_agents,)
                map_state=map_state,                        # (num_agents, state_dim)
                agent_id=agent_ids                          # (num_agents,)
            ).squeeze(-1).cpu().numpy()                     # shape: (num_agents,)

            # Append value to rollout
            last_values = values.reshape(1, -1)               # (1, num_agents)
            values = np.append(self.rollout["values"], last_values, axis=0)  # (T+1, num_agents)


        advantages, returns = self.gae(blended_rewards, values, dones)  # each: (T, num_agents)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        print(blended_rewards.shape, values.shape, advantages.shape, returns.shape)
        # Log advantage statistics before training
        self._log_advantage_stats(advantages)

        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)

        # Track losses across epochs
        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_bonuses = []
        epoch_kl_divergences = []

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_kl = 0
            num_batches = 0
            np.random.shuffle(indices)

            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Slice minibatch
                mb_states = states[mb_indices]              # shape: (B, num_agents, state_dim)
                mb_actions = actions[mb_indices]            # shape: (B, num_agents)
                mb_log_pis = log_pis[mb_indices]            # shape: (B, num_agents)
                mb_advantages = advantages[mb_indices]      # shape: (B, num_agents)
                mb_returns = returns[mb_indices]            # shape: (B, num_agents)
                mb_map_states = map_states[mb_indices]      # shape: (B, 49, 31)
                mb_action_masks = action_masks[mb_indices]  # shape: (B, num_agents, action_dim)

                batch_size = mb_states.shape[0]
                num_agents = mb_states.shape[1]
                state_dim = mb_states.shape[2]

                # Flatten for collective input
                collective_obs = mb_states.view(batch_size, -1)       # (B, num_agents * state_dim)
                collective_actions = mb_actions.view(batch_size, -1)  # (B, num_agents)
                collective_map_state = mb_map_states                  # (B, 49, 31)

                # === Compute value estimates per agent ===
                B, num_agents, state_dim = mb_states.shape

                # Flatten agent dimension into batch
                own_states_flat = mb_states.reshape(B * num_agents, state_dim)      # (B*num_agents, state_dim)
                actions_flat = mb_actions.reshape(B * num_agents)                   # (B*num_agents,)
                map_states_flat = collective_map_state.repeat_interleave(num_agents, dim=0)  # (B*num_agents, ...)
                collective_obs_flat = collective_obs.repeat_interleave(num_agents, dim=0)  # (B*num_agents, ...)

                # Create agent_id tensor
                agent_ids = torch.arange(num_agents, device=mb_states.device).repeat(B)  # (B*num_agents,)

                # Pass through critic
                values_flat = self.critic(
                    own_state=own_states_flat,
                    collective_state=collective_obs_flat,
                    action=actions_flat,
                    map_state=map_states_flat,
                    agent_id=agent_ids
                ).squeeze(-1)  # (B*num_agents,)

                # Reshape back to (B, num_agents)
                value = values_flat.view(B, num_agents)

                # === Prepare samples for loss calculation ===
                samples = {
                    'obs': mb_states,                          # (B, num_agents, state_dim)
                    'map_state': mb_map_states,                # (B, 49, 31)
                    'actions': mb_actions,                     # (B, num_agents)
                    'log_pis': mb_log_pis,                     # (B, num_agents)
                    'advantages': mb_advantages,               # (B, num_agents)
                    'returns': mb_returns,                     # (B, num_agents)
                    'collective_obs': collective_obs,          # (B, num_agents * state_dim)
                    'collective_actions': collective_actions,  # (B, num_agents)
                    'value': value.detach(),                   # (B, num_agents)
                    'action_masks': mb_action_masks            # (B, num_agents, action_dim)
                }

                loss_dict = self._calc_loss(samples)
                loss = loss_dict['total_loss']

                # Accumulate losses for logging
                epoch_loss += loss.item()
                epoch_policy_loss += loss_dict['policy_loss'].item()
                epoch_value_loss += loss_dict['value_loss'].item()
                epoch_entropy += loss_dict['entropy_bonus'].item()
                epoch_kl += loss_dict['kl_divergence'].item()
                num_batches += 1

                # Backprop minibatch
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.shared_actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self.training_step += 1
            
            # Average losses for this epoch
            epoch_losses.append(epoch_loss / num_batches)
            epoch_policy_losses.append(epoch_policy_loss / num_batches)
            epoch_value_losses.append(epoch_value_loss / num_batches)
            epoch_entropy_bonuses.append(epoch_entropy / num_batches)
            epoch_kl_divergences.append(epoch_kl / num_batches)

        # Log average losses across all epochs
        # Pass both original and blended rewards for comprehensive logging
        reward_info = {
            'original_rewards': original_rewards,
            'blended_rewards': blended_rewards,
            'team_spirit': team_spirit
        }
        
        self._log_training_metrics(
            np.mean(epoch_losses),
            np.mean(epoch_policy_losses),
            np.mean(epoch_value_losses),
            np.mean(epoch_entropy_bonuses),
            np.mean(epoch_kl_divergences),
            reward_info  # Pass reward info instead of just rewards
        )

        self.update_guidance_discount()
        self.reset_rollout_trajectory()

    def reset_rollout_trajectory(self):
        for key in self.rollout:
            self.rollout[key] = []
    
    def _log_advantage_stats(self, advantages):
        """Log advantage statistics to TensorBoard"""
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        adv_min = advantages.min().item()
        adv_max = advantages.max().item()
        
        self.writer.add_scalar('Advantages/Mean', adv_mean, self.training_step)
        self.writer.add_scalar('Advantages/Std', adv_std, self.training_step)
        self.writer.add_scalar('Advantages/Min', adv_min, self.training_step)
        self.writer.add_scalar('Advantages/Max', adv_max, self.training_step)
        self.writer.add_histogram('Advantages/Distribution', advantages, self.training_step)

    def _log_training_metrics(self, total_loss, policy_loss, value_loss, entropy_bonus, kl_divergence, reward_info):
        """Log training metrics to TensorBoard"""
        # Loss metrics
        self.writer.add_scalar('Loss/Total', total_loss, self.training_step)
        self.writer.add_scalar('Loss/Policy', policy_loss, self.training_step)
        self.writer.add_scalar('Loss/Value', value_loss, self.training_step)
        self.writer.add_scalar('Loss/Entropy_Bonus', entropy_bonus, self.training_step)
        self.writer.add_scalar('Loss/KL_Divergence', kl_divergence, self.training_step)
        
        # Handle both old format (just rewards) and new format (reward_info dict)
        if isinstance(reward_info, dict):
            # New format with team spirit
            original_rewards = reward_info['original_rewards']
            blended_rewards = reward_info['blended_rewards']
            team_spirit = reward_info['team_spirit']
            
            # Original individual reward metrics
            orig_reward_mean = np.mean(original_rewards)
            orig_reward_std = np.std(original_rewards)
            orig_reward_sum = np.sum(original_rewards)
            
            self.writer.add_scalar('Rewards/Original/Mean', orig_reward_mean, self.training_step)
            self.writer.add_scalar('Rewards/Original/Std', orig_reward_std, self.training_step)
            self.writer.add_scalar('Rewards/Original/Sum', orig_reward_sum, self.training_step)
            
            # Blended reward metrics (what the agents actually train on)
            blended_reward_mean = np.mean(blended_rewards)
            blended_reward_std = np.std(blended_rewards)
            blended_reward_sum = np.sum(blended_rewards)
            
            self.writer.add_scalar('Rewards/Blended/Mean', blended_reward_mean, self.training_step)
            self.writer.add_scalar('Rewards/Blended/Std', blended_reward_std, self.training_step)
            self.writer.add_scalar('Rewards/Blended/Sum', blended_reward_sum, self.training_step)
            
            # Team reward metrics (collective component)
            team_rewards = original_rewards.mean(axis=1, keepdims=True)
            team_rewards = np.broadcast_to(team_rewards, original_rewards.shape)
            team_reward_mean = np.mean(team_rewards)
            team_reward_std = np.std(team_rewards)
            
            self.writer.add_scalar('Rewards/Team/Mean', team_reward_mean, self.training_step)
            self.writer.add_scalar('Rewards/Team/Std', team_reward_std, self.training_step)
            
            # Per-agent reward analysis
            num_agents = original_rewards.shape[1]
            for agent_id in range(num_agents):
                agent_orig_mean = np.mean(original_rewards[:, agent_id])
                agent_blended_mean = np.mean(blended_rewards[:, agent_id])
                
                self.writer.add_scalar(f'Rewards/Agent_{agent_id}/Original_Mean', agent_orig_mean, self.training_step)
                self.writer.add_scalar(f'Rewards/Agent_{agent_id}/Blended_Mean', agent_blended_mean, self.training_step)
            
            # Team spirit parameter
            self.writer.add_scalar('Parameters/Team_Spirit', team_spirit, self.training_step)
            
            # Reward variance analysis (how much individual rewards differ from team reward)
            reward_variance_from_team = np.mean((original_rewards - team_rewards) ** 2)
            self.writer.add_scalar('Rewards/Variance_From_Team', reward_variance_from_team, self.training_step)
            
        else:
            # Backward compatibility: old format where reward_info is just the rewards array
            rewards = reward_info
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)
            reward_sum = np.sum(rewards)
            
            self.writer.add_scalar('Rewards/Mean', reward_mean, self.training_step)
            self.writer.add_scalar('Rewards/Std', reward_std, self.training_step)
            self.writer.add_scalar('Rewards/Sum', reward_sum, self.training_step)
        
        # Episode length metrics (if we have completed episodes)
        if self.episode_lengths:
            recent_episodes = self.episode_lengths[-10:]  # Last 10 episodes
            self.writer.add_scalar('Episode/Length_Mean_Recent', np.mean(recent_episodes), self.training_step)
            self.writer.add_scalar('Episode/Length_Std_Recent', np.std(recent_episodes), self.training_step)
            self.writer.add_scalar('Episode/Total_Episodes', self.total_episodes, self.training_step)
        
        # Training parameters
        self.writer.add_scalar('Parameters/Epsilon', self.epsilon, self.training_step)
        self.writer.add_scalar('Parameters/Guidance_Discount', self.guidance_discount, self.training_step)
        self.writer.add_scalar('Parameters/Lambda', self._lambd(), self.training_step)

    def get_episode_stats(self):
        """Return episode length statistics"""
        if not self.episode_lengths:
            return None
        
        return {
            'total_episodes': self.total_episodes,
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            'min_length': np.min(self.episode_lengths),
            'max_length': np.max(self.episode_lengths),
            'recent_mean': np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths)
        }

    @staticmethod
    def _normalize(adv: torch.Tensor):
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def getActionProbability(self, state, map_state, action_masking=None):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        collective_state = state.flatten().unsqueeze(0)
        map_state = torch.tensor(map_state, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        if action_masking is not None:
            action_masking = torch.tensor(action_masking, dtype=torch.float32).to(self.device)
        
        softmax_actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        log_probs = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)

        with torch.no_grad():
            num_agents = state.shape[0]

            # Prepare agent ids tensor
            agent_ids = torch.arange(num_agents, device=state.device)  # (num_agents,)

            # Expand collective_state and map_state to batch size = num_agents
            collective_state_expanded = collective_state.expand(num_agents, -1)   # (num_agents, collective_dim)
            map_state_expanded = map_state.expand(num_agents, -1, -1, -1)        # (num_agents, C, H, W)

            # Forward pass for all agents in one call
            logits = self.shared_actor(state, collective_state_expanded, map_state_expanded, agent_ids)  # (num_agents, action_dim)

            if action_masking is not None:
                mask = action_masking.bool()   # (num_agents, action_dim)
                logits = logits.clone()
                logits[~mask] = -1e8

            probs = torch.softmax(logits, dim=-1)  # (num_agents, action_dim)
            log_probs = torch.log(probs + 1e-10)   # (num_agents, action_dim)

            # If you want numpy arrays:
            softmax_actions = probs.cpu().numpy()   # (num_agents, action_dim)
            log_probs_np = log_probs.cpu().numpy()  # (num_agents, action_dim)
        
        return softmax_actions, log_probs_np

    def getAction(self, state, action_masking, map_state, deterministic=False):
        softmax_actions, log_probs = self.getActionProbability(state, map_state, action_masking)
        
        actions = np.zeros((self.num_agents,), dtype=np.int64)
        log_pis = np.zeros((self.num_agents,), dtype=np.float32)

        decay_rate = (self.epsilon - self.epsilon_min) / (self.num_steps * self.num_episodes)
        self.epsilon = max(self.epsilon - decay_rate, self.epsilon_min)

        for agent_id in range(self.num_agents):
            valid_actions = np.where(np.atleast_1d(action_masking[agent_id]) == 1)[0]
            
            if len(valid_actions) == 0:
                # If no valid actions, this shouldn't happen but handle gracefully
                action = 0
                print(f"Warning: No valid actions for agent {agent_id}")
            elif deterministic or (self.epsilon < np.random.rand()):
                # Choose best valid action
                valid_probs = softmax_actions[agent_id][valid_actions]
                best_valid_idx = np.argmax(valid_probs)
                action = valid_actions[best_valid_idx]
            else:
                # Random selection from valid actions
                probs = softmax_actions[agent_id][valid_actions]
                probs /= probs.sum()  # Normalize (should already be normalized due to masking)
                chosen_idx = np.random.choice(len(valid_actions), p=probs)
                action = valid_actions[chosen_idx]
                
                # Apply special logic for destination proximity
                pos_x = state[agent_id, 3] * 49
                pos_y = state[agent_id, 4] * 31
                dest_x = state[agent_id, 5] * 49
                dest_y = state[agent_id, 6] * 31
                
                if np.linalg.norm(np.array((pos_x, pos_y)) - np.array((dest_x, dest_y))) <= 15:
                    if action_masking[agent_id][8] == 1:  # Check if action 8 is valid
                        action = 8

            actions[agent_id] = action
            log_pis[agent_id] = log_probs[agent_id][action]

        return actions, log_pis

    
    # Actor Parameter Sharing
    
    def save_model(self, path=None):
        """Save model to file."""
        model_path=self.model_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.shared_actor.state_dict(), model_path)
        torch.save(self.critic.state_dict(), model_path.replace('.pt', '_critic.pt'))
        print(f"Model saved to {model_path} and {model_path.replace('.pt', '_critic.pt')}")

    def load_model(self, path=None):
        """Load model from file."""
        model_path=self.model_path
        if os.path.exists(model_path):
            self.shared_actor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(model_path.replace('.pt', '_critic.pt'), map_location=self.device))
            self.critic.to(self.device)  # Move critic to device

            print(f"Model loaded from {model_path} and {model_path.replace('.pt', '_critic.pt')}")
            # for p in self.shared_actor.parameters():
            #     print(p.data.norm())
        else:
            print(f"Model file {model_path} not found.")
        return self.shared_actor, self.critic
    
    @property
    def guidance_discount(self):
        return self._lambd()*self.gamma
    
    def update_guidance_discount(self):
        self._lambd.update()
        self.gamma_n = self.guidance_discount

    def close_logger(self):
        """Close TensorBoard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()
            print("TensorBoard writer closed")

    def __del__(self):
        """Cleanup TensorBoard writer on deletion"""
        self.close_logger()

class LambdaScheduler:
    def __init__(self, init_lambd, lambd_scheduler_alpha=1, n_epochs=100000):
        self._n_epochs = n_epochs
        self._itr = 0
        self._init_lambd = init_lambd
        self._lambd_scheduler_alpha = lambd_scheduler_alpha
        self._lambd = init_lambd

    def update(self):
        self._itr +=1

    def __call__(self):
        return min(1.0, max(0.0, self._lambd))
    
class LinearLS(LambdaScheduler):
    def update(self):
        super().update()
        self._lambd = self._init_lambd + (1.0-self._init_lambd)*self._delta

    @property
    def _delta(self):  # a value in [0,1]
        return min(1.0,self._itr/self._n_epochs)

class TanhLS(LinearLS):
    @property
    def _delta(self):  # a value in [0,1]
        limit = 0.99
        delta = np.tanh(self._itr/(self._lambd_scheduler_alpha * max(1,self._n_epochs-1))*np.arctanh(limit)) / limit
        return min(1.0, delta)