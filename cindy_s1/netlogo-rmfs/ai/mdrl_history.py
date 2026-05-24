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
    def __init__(self, state_dim, map_state_dim, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        # Original state processing layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Convolutional layers for map_state processing
        # Input: (batch_size, 1, 49, 31) - treating as single channel 2D map
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)  # -> (batch_size, 16, 25, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # -> (batch_size, 32, 13, 8)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> (batch_size, 64, 7, 4)
        
        # Calculate the flattened size after conv layers: 64 * 7 * 4 = 1792
        conv_output_size = 64 * 7 * 4
        self.fcmap = nn.Linear(conv_output_size, hidden_dim)
        
        # Final layers combining both features
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, map_state):
        # Process regular state
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Process map_state with conv layers
        # Add channel dimension if not present: (batch_size, 49, 31) -> (batch_size, 1, 49, 31)
        if len(map_state.shape) == 3:
            map_state = map_state.unsqueeze(1)
        
        x_map = F.relu(self.conv1(map_state))
        x_map = F.relu(self.conv2(x_map))
        x_map = F.relu(self.conv3(x_map))
        
        # Flatten and process through FC layer
        x_map = x_map.flatten(start_dim=1)
        x_map = F.relu(self.fcmap(x_map))
        
        # Combine features
        x_all = torch.cat((x, x_map.squeeze(0)), dim=-1)
        x_all = F.relu(self.fc3(x_all))
        return self.fc4(x_all)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, map_state_dim, action_dim, num_agents, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        # Original state and action processing layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, 2*hidden_dim)
        
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

    def forward(self, state, actions, map_state):
        # Process state and actions
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat([x, actions], dim=-1)))
        
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
        x_all = torch.cat((x, x_map.squeeze(0)), dim=-1)
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

        returns = advantages + values[:-1, None]
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
        return -policy_reward.mean()
    
class ClippedValueFunctionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, value: torch.Tensor, sampled_value: torch.Tensor, 
                sampled_return: torch.Tensor, clip: float):
        clipped_value = sampled_value + (value-sampled_value).clamp(min=-clip,max=clip)
        vf_loss = torch.max((value-sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        return 0.5 * vf_loss.mean()
        
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
                 log_dir=None):
        super(MultiAgentPPO, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.map_state_dim = map_state_dim
        self.action_dim = action_dim
        self.batch_size = 256
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
            "next_map_states": []
        }

        self.epsilon = 0.95
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.actor_learning_rate = actor_lr
        self.critic_learning_rate = critic_lr
        self.value_loss_coef = 1.0
        self.entropy_bonus_coef = 0.01
        self.clip_range = 0.2
        self.max_grad_norm = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_episodes = 10
        self.num_steps = 20000
        self.rollout_length = 256
        self.frameskip = 4
        self.lstm_unroll_length = 16
        self.current_step = 0
        self.store_step_iteration = 0
        self.save_every = 1024
        self.episode_lengths = []  # Store lengths of completed episodes
        self.current_episode_length = 0  # Track current episode step count
        self.total_episodes = 0


        self.actor_parameter_sharing = actor_parameter_sharing
        self.shared_actor = ActorNetwork(state_dim, map_state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.shared_actor.parameters(), lr=self.actor_learning_rate, eps=1e-2)

        self.critic = CriticNetwork(state_dim * self.num_agents, map_state_dim, action_dim * self.num_agents, self.num_agents, hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate, eps = 1e-2)

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
        
    def store_step(self, state, action, reward, next_state, done, log_pi, value, map_state, next_map_state):
        self.rollout["states"].append(state)
        self.rollout["actions"].append(action)
        self.rollout["rewards"].append(reward)
        self.rollout["next_states"].append(next_state)
        self.rollout["dones"].append(done)
        self.rollout["log_pis"].append(log_pi)
        self.rollout["values"].append(value)
        self.rollout["map_states"].append(map_state)
        self.rollout["next_map_states"].append(next_map_state)

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



    def train_on_rollout(self, epochs=8, minibatch_size=64):
        print("Training on rollout")
        states = torch.tensor(np.stack(self.rollout["states"]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.stack(self.rollout["actions"]), dtype=torch.float32).to(self.device)
        rewards = np.stack(self.rollout["rewards"])
        dones = np.stack(self.rollout["dones"])
        values = np.stack(self.rollout["values"])
        log_pis = torch.tensor(np.stack(self.rollout["log_pis"]), dtype=torch.float32).to(self.device)
        map_states = torch.tensor(np.stack(self.rollout["map_states"]), dtype=torch.float32).to(self.device)

        next_map_states = torch.tensor(np.stack(self.rollout["next_map_states"]), dtype=torch.float32).to(self.device)


        with torch.no_grad():
            print("Sampling for advantage")
            next_state = torch.tensor(self.rollout["next_states"][-1], dtype=torch.float32).to(self.device)
            next_map = torch.tensor(self.rollout["next_map_states"][-1], dtype=torch.float32).to(self.device).unsqueeze(0)
            one_hot_actions = []
            for agent_id in range(self.num_agents):
                logits = self.shared_actor(next_state[agent_id], next_map)
                dist = torch.distributions.Categorical(logits=logits)
                sampled_action = dist.sample()
                one_hot = torch.nn.functional.one_hot(sampled_action, num_classes=self.action_dim).float()
                one_hot_actions.append(one_hot)

            next_joint_action = torch.cat(one_hot_actions, dim=0).view(-1)
            next_joint_state = next_state.flatten()
            last_value = self.critic(next_joint_state, next_joint_action, next_map).squeeze().cpu().numpy()

            # Add to value array for bootstrapping GAE
            values = np.append(values, last_value[None], axis=0)

        # Compute GAE advantages and returns
        advantages, returns = self.gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

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
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_log_pis = log_pis[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_map_states = map_states[mb_indices]

                collective_obs = mb_states.view(minibatch_size, -1)
                collective_map_state = mb_map_states
                collective_actions = mb_actions.view(minibatch_size, -1)

                print(collective_map_state.shape)

                value = self.critic(collective_obs, collective_actions, collective_map_state)

                # Build samples dict for loss calc
                samples = {
                    'obs': mb_states,
                    'map_state': mb_map_states,
                    'actions': mb_actions,
                    'log_pis': mb_log_pis,
                    'advantages': mb_advantages,
                    'returns': mb_returns,
                    'collective_obs': collective_obs,
                    'collective_actions': collective_actions,
                    'value': value.detach(),  # old_values replaced by current critic eval here
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
        self._log_training_metrics(
            np.mean(epoch_losses),
            np.mean(epoch_policy_losses),
            np.mean(epoch_value_losses),
            np.mean(epoch_entropy_bonuses),
            np.mean(epoch_kl_divergences),
            rewards
        )

        self.update_guidance_discount()

        self.reset_rollout_trajectory()

    def _calc_loss(self, samples):
        # Normalize advantages
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Policy loss
        log_probs = []
        for agent_id in range(self.num_agents):
            agent_state = samples['obs'][:, agent_id, :]
            logits = self.shared_actor(agent_state, samples['map_state'])
            dist = torch.distributions.Categorical(logits=logits)
            agent_actions = samples['actions'][:, agent_id, :].argmax(dim=-1)
            agent_log_pi = dist.log_prob(agent_actions)
            log_probs.append(agent_log_pi)

        log_pi = torch.stack(log_probs).transpose(0,1)
        value = self.critic(samples['collective_obs'], samples['collective_actions'], samples['map_state'])

        policy_loss = self.ppo_loss(log_pi, samples['log_pis'], sampled_normalized_advantage, self.clip_range)
        entropy_bonus = dist.entropy().mean()
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

    def _log_training_metrics(self, total_loss, policy_loss, value_loss, entropy_bonus, kl_divergence, rewards):
        """Log training metrics to TensorBoard"""
        # Loss metrics
        self.writer.add_scalar('Loss/Total', total_loss, self.training_step)
        self.writer.add_scalar('Loss/Policy', policy_loss, self.training_step)
        self.writer.add_scalar('Loss/Value', value_loss, self.training_step)
        self.writer.add_scalar('Loss/Entropy_Bonus', entropy_bonus, self.training_step)
        self.writer.add_scalar('Loss/KL_Divergence', kl_divergence, self.training_step)
        
        # Reward metrics
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

    def getActionProbability(self, state, map_state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        map_state = torch.tensor(map_state, dtype=torch.float32).to(self.device).unsqueeze(0)
        softmax_actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        log_probs = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)

        with torch.no_grad():
            for agent_id in range(self.num_agents):
                logits = self.shared_actor(state[agent_id, :], map_state)
                probs = torch.softmax(logits, dim=-1)
                softmax_actions[agent_id] = probs.cpu().numpy()
                log_probs[agent_id] = torch.log(probs + 1e-10).cpu().numpy()
        
        return softmax_actions, log_probs
    
    def getAction(self, state, action_masking, map_state, deterministic=False):
        softmax_actions, log_probs = self.getActionProbability(state, map_state)
        
        filtered_actions = softmax_actions * action_masking
        actions = np.zeros((self.num_agents,), dtype=np.int64)
        log_pis = np.zeros((self.num_agents,), dtype=np.float32)

        decay_rate = (self.epsilon - self.epsilon_min) / (self.num_steps * self.num_episodes)
        self.epsilon = max(self.epsilon - decay_rate, self.epsilon_min)

        for agent_id in range(self.num_agents):
            if deterministic or (self.epsilon < np.random.rand()):
                action = np.argmax(filtered_actions[agent_id])
            else:
                probs = filtered_actions[agent_id]
                probs /= probs.sum()  # normalize after masking
                action = np.random.choice(self.action_dim, p=probs)

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

class MultiAgentA2C():
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
                 actor_parameter_sharing=False):

        super(MultiAgentA2C, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.map_state_dim = map_state_dim
        self.action_dim = action_dim
        self.batch_size = 64

        self.memory = deque(maxlen=20000)
        self.epsilon = 0.95
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.actor_learning_rate = actor_lr
        self.critic_learning_rate = critic_lr
        self.entropy_coeff = 0.01
        self.max_grad_norm = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_episodes = 1
        self.num_steps = 5000
        self.current_iteration = 0
        self.learn_after = 100
        self.save_every = 250
        self.learn_every = 1

        self.actor_parameter_sharing = actor_parameter_sharing
        self.actors = nn.ModuleList([ActorNetwork(state_dim, map_state_dim, action_dim, hidden_dim).to(self.device) for _ in range(num_agents)])
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.actor_learning_rate, eps=1e-2) for actor in self.actors]
        self.shared_actor = ActorNetwork(state_dim, map_state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.shared_actor.parameters(), lr=self.actor_learning_rate, eps=1e-2)

        self.critic = CriticNetwork(state_dim * self.num_agents, map_state_dim, action_dim * self.num_agents, self.num_agents, hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate, eps = 1e-2)

        self.lambd0 = lambd_0
        self.lambd_scheduler_alpha = lambd_scheduler_alpha
        self._lambd = self.lambd0 if isinstance(self.lambd0, TanhLS) else TanhLS(self.lambd0, self.lambd_scheduler_alpha)

        self.gamma = gamma
        self.gamma_n = self.guidance_discount
        
        self.model_path = os.path.join(
            os.getcwd(), 'maa2c_routing',f'alr-{self.actor_learning_rate}_clr-{self.critic_learning_rate}_lambd0-{self.lambd0}_lambdaplha-{self.lambd_scheduler_alpha}_gamma-{self.gamma}', 'model.pt'
                )
    def remember(self, state, action, reward, next_state, done, at_waypoint, priority, map_state, next_map_state):
        self.memory.append((state, action, reward, next_state, done, at_waypoint, priority, map_state, next_map_state))

    def replay(self):
        priorities = [m[6] for m in self.memory]
        minibatch = random.choices(self.memory, weights=priorities, k=self.batch_size)

        states = torch.from_numpy(np.stack([m[0] for m in minibatch])).float().to(self.device).view(-1, self.num_agents, self.state_dim)
        actions = torch.from_numpy(np.stack([m[1] for m in minibatch])).float().to(self.device).view(-1, self.num_agents, self.action_dim)
        rewards = torch.from_numpy(np.stack([m[2] for m in minibatch])).float().to(self.device).view(-1, self.num_agents)
        next_states = torch.from_numpy(np.stack([m[3] for m in minibatch])).float().to(self.device).view(-1, self.num_agents, self.state_dim)
        dones = torch.from_numpy(np.stack([m[4] for m in minibatch])).float().to(self.device).repeat(self.num_agents).view(-1, self.num_agents)
        at_waypoint = torch.from_numpy(np.stack([m[5] for m in minibatch])).float().to(self.device).view(-1, self.num_agents)
        map_state = torch.from_numpy(np.stack([m[7] for m in minibatch])).float().to(self.device).view(-1, self.map_state_dim)
        next_map_state = torch.from_numpy(np.stack([m[8] for m in minibatch])).float().to(self.device).view(-1, self.map_state_dim)

        collective_states = states.view(-1, self.num_agents * self.state_dim)
        collective_actions = actions.view(-1, self.num_agents * self.action_dim)

        # Sample next actions using the shared actor for all agents
        next_states_reshaped = next_states.view(-1, self.state_dim)
        repeated_map_state = torch.repeat_interleave(next_map_state, self.num_agents, dim=0)
        logits = self.shared_actor(next_states_reshaped, repeated_map_state)
        dist = torch.distributions.Categorical(logits=logits)
        sampled_actions = dist.sample()
        one_hot_actions = F.one_hot(sampled_actions, num_classes=self.action_dim).float()
        next_actions = one_hot_actions.view(-1, self.num_agents * self.action_dim)

        next_collective_states = next_states.view(-1, self.num_agents * self.state_dim)
        
        values = self.critic(collective_states, collective_actions, map_state)
        next_values = self.critic(next_collective_states, next_actions, next_map_state).detach()

        global_rewards = rewards.mean(dim=1, keepdim=True)  # or use weighted variant
        global_dones = dones[:, 0].unsqueeze(1)  # assume first agent or use torch.any(dones, dim=1)

        # Centralized TD targets
        targets = global_rewards + self.gamma * next_values * (1 - global_dones)  # shape: [batch_size, 1]

        # Zero gradients
        self.actor_optimizer.zero_grad()

        actor_loss_total = 0
        entropy_loss_total = 0
        active_waypoints_total = 0

        states_reshaped = states.view(-1, self.state_dim)
        map_state_expanded = torch.repeat_interleave(map_state, self.num_agents, dim=0)
        logits = self.shared_actor(states_reshaped, map_state_expanded)
        action_probs = F.softmax(logits, dim=-1)

        entropy = -torch.sum(action_probs * torch.log2(action_probs + 1e-10), dim=1)
        entropy = entropy.view(-1, self.num_agents)
        entropy_loss = torch.sum(entropy * at_waypoint)

        # Log probs
        dist = torch.distributions.Categorical(logits=logits)
        taken_actions = actions.argmax(dim=-1).view(-1)
        log_probs = dist.log_prob(taken_actions).view(-1, self.num_agents)

        td_error = (targets - values.detach()).squeeze(1).unsqueeze(1)  # shape [batch, 1]
        pg_loss = -torch.sum(log_probs * td_error * at_waypoint)

        active_waypoints_total = torch.sum(at_waypoint)
        actor_loss_total = pg_loss / (active_waypoints_total + 1e-8)
        entropy_loss_total = entropy_loss / (active_waypoints_total + 1e-8)


        # Total loss
        actor_loss = actor_loss_total - self.entropy_coeff * entropy_loss_total

        # Backward and step
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.shared_actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()


        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss = F.smooth_l1_loss(values, targets.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.update_guidance_discount()

        self.current_iteration += 1

    def getActionProbability(self, state, map_state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).view(self.num_agents, self.state_dim)
        map_state = torch.tensor(map_state, dtype=torch.float32).to(self.device)
        softmax_actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        
        if deterministic:
            with torch.no_grad():
                for agent_id in range(self.num_agents):
                    if self.actor_parameter_sharing:
                        action_probs = self.shared_actor(state[agent_id, :], map_state)
                    else:
                        action_probs = self.actors[agent_id](state[agent_id, :], map_state)
                    softmax_actions[agent_id] = torch.softmax(action_probs, dim=-1).cpu().numpy()
        else:
            for agent_id in range(self.num_agents):
                if self.actor_parameter_sharing:
                    action_probs = self.shared_actor(state[agent_id, :], map_state)
                    # print(action_probs)
                else:
                    action_probs = self.actors[agent_id](state[agent_id, :], map_state)
                softmax_actions[agent_id] = torch.softmax(action_probs, dim=-1).cpu().detach().numpy()
        return softmax_actions
    
    def getAction(self, state, action_masking, map_state, deterministic=False):
        softmax_actions = self.getActionProbability(state, map_state, deterministic)
        for inner_list in action_masking:
            inner_list[-1] = 1
        filtered_actions = softmax_actions * action_masking
        actions = np.zeros((self.num_agents,), dtype=np.int64)

        decay_rate = (self.epsilon - self.epsilon_min) / (self.num_steps * self.num_episodes)
        self.epsilon = max(self.epsilon - decay_rate, self.epsilon_min)

        for agent_id in range(self.num_agents):
            if deterministic:
                actions[agent_id] = np.argmax(filtered_actions[agent_id])
            else:
                if self.epsilon < np.random.rand():
                    actions[agent_id] = np.argmax(filtered_actions[agent_id])
                else:
                    actions[agent_id] = np.random.choice(self.action_dim, p=action_masking[agent_id]/np.sum(action_masking[agent_id]))
        return actions
    
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

    

class LambdaScheduler:
    def __init__(self, init_lambd, lambd_scheduler_alpha=1, n_epochs=20000):
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