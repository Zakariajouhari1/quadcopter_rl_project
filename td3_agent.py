import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Any
import copy

class ReplayBuffer:
    """
    Experience replay buffer for TD3 off-policy learning.
    Stores transitions and provides random sampling for training.
    """
    
    def __init__(self, max_size: int = int(1e6)):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.FloatTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.BoolTensor([t[4] for t in batch]).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def size(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

class Actor(nn.Module):
    """
    Actor network for TD3 - outputs deterministic continuous actions.
    Maps states to actions in the continuous action space.
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, 
                 hidden_dim: int = 400):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        # Network architecture (as specified in document)
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 300)  # Second hidden layer with 300 neurons
        self.layer3 = nn.Linear(300, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in [self.layer1, self.layer2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Final layer with smaller initialization for stable initial policy
        nn.init.uniform_(self.layer3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.layer3.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        action = torch.tanh(self.layer3(x)) * self.max_action
        return action

class Critic(nn.Module):
    """
    Critic network for TD3 - estimates Q-values for state-action pairs.
    TD3 uses two critic networks to reduce overestimation bias.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 400):
        super(Critic, self).__init__()
        
        # Q1 network
        self.layer1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2_q1 = nn.Linear(hidden_dim, 300)
        self.layer3_q1 = nn.Linear(300, 1)
        
        # Q2 network (twin network)
        self.layer1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2_q2 = nn.Linear(hidden_dim, 300)
        self.layer3_q2 = nn.Linear(300, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in [self.layer1_q1, self.layer2_q1, self.layer1_q2, self.layer2_q2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Final layers
        for layer in [self.layer3_q1, self.layer3_q2]:
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(layer.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both critic networks."""
        state_action = torch.cat([state, action], dim=1)
        
        # Q1 network
        q1 = F.relu(self.layer1_q1(state_action))
        q1 = F.relu(self.layer2_q1(q1))
        q1 = self.layer3_q1(q1)
        
        # Q2 network
        q2 = F.relu(self.layer1_q2(state_action))
        q2 = F.relu(self.layer2_q2(q2))
        q2 = self.layer3_q2(q2)
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q1 network only (for actor updates)."""
        state_action = torch.cat([state, action], dim=1)
        q1 = F.relu(self.layer1_q1(state_action))
        q1 = F.relu(self.layer2_q1(q1))
        q1 = self.layer3_q1(q1)
        return q1

class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise.
    Provides temporally correlated noise suitable for momentum-based tasks.
    """
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, 
                 sigma: float = 0.2, dt: float = 1e-2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        """Reset the noise process."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Sample noise from the OU process."""
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state += dx
        return self.state

class TD3Agent:
    """
    Twin-Delayed Deep Deterministic Policy Gradient (TD3) Agent.
    Implements the three key improvements over DDPG:
    1. Clipped Double-Q Learning
    2. Target Policy Smoothing  
    3. Delayed Policy Updates
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 config: Dict[str, Any] = None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Load configuration
        self.config = config or self._load_default_config()
        
        # TD3 hyperparameters
        self.lr_actor = self.config.get('lr_actor', 1e-3)
        self.lr_critic = self.config.get('lr_critic', 1e-3)
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)  # Soft update parameter
        self.policy_delay = self.config.get('policy_delay', 2)  # Delayed policy updates
        self.target_noise = self.config.get('target_noise', 0.2)  # Target policy smoothing noise
        self.noise_clip = self.config.get('noise_clip', 0.5)  # Target noise clipping
        self.batch_size = self.config.get('batch_size', 256)
        
        # Exploration noise parameters
        self.exploration_noise = self.config.get('exploration_noise', 0.1)
        self.noise_decay = self.config.get('noise_decay', 0.995)
        self.min_noise = self.config.get('min_noise', 0.01)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TD3 Agent using device: {self.device}")
        
        # Initialize networks
        hidden_dim = self.config.get('hidden_dim', 400)
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Replay buffer
        buffer_size = self.config.get('buffer_size', int(1e6))
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(
            action_dim, 
            sigma=self.exploration_noise,
            theta=self.config.get('ou_theta', 0.15),
            dt=self.config.get('ou_dt', 1e-2)
        )
        
        # Training statistics
        self.total_iterations = 0
        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
        
        # Learning state
        self.learning_started = False
        self.warmup_steps = self.config.get('warmup_steps', 1000)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default TD3 configuration."""
        return {
            'lr_actor': 1e-3,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_delay': 2,
            'target_noise': 0.2,
            'noise_clip': 0.5,
            'batch_size': 256,
            'buffer_size': int(1e6),
            'hidden_dim': 400,
            'exploration_noise': 0.1,
            'noise_decay': 0.995,
            'min_noise': 0.01,
            'warmup_steps': 1000,
            'ou_theta': 0.15,
            'ou_dt': 1e-2
        }
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action using the current policy.
        
        Args:
            state: Current environment state
            add_noise: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        # Add exploration noise during training
        if add_noise:
            if self.total_iterations < self.warmup_steps:
                # Random actions during warmup
                action = np.random.uniform(-self.max_action, self.max_action, self.action_dim)
            else:
                # Add OU noise for exploration
                noise = self.noise.sample() * self.exploration_noise
                action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                      next_state: np.ndarray, done: bool):
        """Add experience to replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self) -> Dict[str, float]:
        """
        Train the TD3 agent using a batch of experiences.
        
        Returns:
            Dictionary containing training metrics
        """
        if self.replay_buffer.size() < self.batch_size:
            return {}
        
        self.learning_started = True
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Train critics
        critic_loss = self._train_critics(states, actions, rewards, next_states, dones)
        
        # Train actor (delayed policy updates)
        actor_loss = None
        if self.total_iterations % self.policy_delay == 0:
            actor_loss = self._train_actor(states)
            self._soft_update_targets()
        
        self.total_iterations += 1
        
        # Decay exploration noise
        self.exploration_noise = max(self.min_noise, self.exploration_noise * self.noise_decay)
        
        # Prepare training metrics
        metrics = {
            'critic_loss': critic_loss,
            'total_iterations': self.total_iterations,
            'exploration_noise': self.exploration_noise,
            'buffer_size': self.replay_buffer.size()
        }
        
        if actor_loss is not None:
            metrics['actor_loss'] = actor_loss
        
        return metrics
    
    def _train_critics(self, states: torch.Tensor, actions: torch.Tensor,
                      rewards: torch.Tensor, next_states: torch.Tensor,
                      dones: torch.Tensor) -> float:
        """Train critic networks using TD3's clipped double-Q learning."""
        
        with torch.no_grad():
            # Target policy smoothing: add noise to target actions
            target_actions = self.actor_target(next_states)
            noise = torch.clamp(
                torch.randn_like(target_actions) * self.target_noise,
                -self.noise_clip, self.noise_clip
            )
            target_actions = torch.clamp(target_actions + noise, -self.max_action, self.max_action)
            
            # Compute target Q-values using minimum of two critics (clipped double-Q learning)
            target_q1, target_q2 = self.critic_target(next_states, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic losses
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        self.critic_optimizer.step()
        
        # Store metrics
        self.critic_losses.append(critic_loss.item())
        self.q_values.append(current_q1.mean().item())
        
        return critic_loss.item()
    
    def _train_actor(self, states: torch.Tensor) -> float:
        """Train actor network using policy gradient."""
        
        # Compute actor loss (negative Q-value)
        actor_actions = self.actor(states)
        actor_loss = -self.critic.q1_forward(states, actor_actions).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        
        self.actor_optimizer.step()
        
        # Store metrics
        self.actor_losses.append(actor_loss.item())
        
        return actor_loss.item()
    
    def _soft_update_targets(self):
        """Soft update of target networks using polyak averaging."""
        
        # Update actor target
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_models(self, filepath_prefix: str):
        """Save actor and critic models."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_iterations': self.total_iterations,
            'exploration_noise': self.exploration_noise,
            'config': self.config
        }, f"{filepath_prefix}_td3_checkpoint.pth")
        
        # Save individual networks for deployment
        torch.save(self.actor.state_dict(), f"{filepath_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filepath_prefix}_critic.pth")
        
        print(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str, load_optimizers: bool = True):
        """Load actor and critic models."""
        checkpoint = torch.load(f"{filepath_prefix}_td3_checkpoint.pth", map_location=self.device)
        
        # Load networks
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target = copy.deepcopy(self.critic)
        
        # Load optimizers
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load training state
        self.total_iterations = checkpoint.get('total_iterations', 0)
        self.exploration_noise = checkpoint.get('exploration_noise', self.config['exploration_noise'])
        
        print(f"Models loaded from: {filepath_prefix}")
    
    def freeze_policy(self):
        """Freeze the actor network for deployment."""
        self.actor.eval()
        for param in self.actor.parameters():
            param.requires_grad = False
        print("Policy network frozen for deployment")
    
    def unfreeze_policy(self):
        """Unfreeze the actor network for continued training."""
        self.actor.train()
        for param in self.actor.parameters():
            param.requires_grad = True
        print("Policy network unfrozen for training")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'total_iterations': self.total_iterations,
            'buffer_size': self.replay_buffer.size(),
            'exploration_noise': self.exploration_noise,
            'learning_started': self.learning_started
        }
        
        if self.actor_losses:
            stats.update({
                'avg_actor_loss': np.mean(self.actor_losses[-100:]),  # Last 100 episodes
                'actor_loss_trend': self.actor_losses[-10:] if len(self.actor_losses) >= 10 else self.actor_losses
            })
        
        if self.critic_losses:
            stats.update({
                'avg_critic_loss': np.mean(self.critic_losses[-100:]),
                'critic_loss_trend': self.critic_losses[-10:] if len(self.critic_losses) >= 10 else self.critic_losses
            })
        
        if self.q_values:
            stats.update({
                'avg_q_value': np.mean(self.q_values[-100:]),
                'q_value_trend': self.q_values[-10:] if len(self.q_values) >= 10 else self.q_values
            })
        
        return stats
    
    def reset_noise(self):
        """Reset exploration noise process."""
        self.noise.reset()
    
    def evaluate_policy(self, env, num_episodes: int = 10, max_steps: int = 1000) -> Dict[str, float]:
        """
        Evaluate the current policy without exploration noise.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Evaluation metrics
        """
        
        self.actor.eval()  # Set to evaluation mode
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Select action without noise
                action = self.select_action(state, add_noise=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Check for mission success (could be defined in info)
                if info.get('mission_completed', False):
                    success_count += 1
                
                if terminated or truncated:
                    break
                
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        self.actor.train()  # Set back to training mode
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'success_rate': success_count / num_episodes,
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }

class CurriculumLearning:
    """
    Curriculum learning helper for progressive difficulty increase.
    Gradually increases task complexity during training.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Curriculum stages
        self.stages = [
            {
                'name': 'stabilization',
                'duration': self.config.get('stage1_duration', 50000),
                'max_distance': 2.0,
                'fixed_start': True,
                'no_wind': True,
                'single_waypoint': True
            },
            {
                'name': 'simple_navigation',
                'duration': self.config.get('stage2_duration', 100000),
                'max_distance': 5.0,
                'fixed_start': False,
                'no_wind': True,
                'single_waypoint': False
            },
            {
                'name': 'full_mission',
                'duration': float('inf'),
                'max_distance': float('inf'),
                'fixed_start': False,
                'no_wind': False,
                'single_waypoint': False
            }
        ]
        
        self.current_stage = 0
        self.stage_progress = 0
    
    def update(self, training_step: int) -> Dict[str, Any]:
        """Update curriculum based on training progress."""
        
        # Check if should advance to next stage
        if (self.current_stage < len(self.stages) - 1 and 
            self.stage_progress >= self.stages[self.current_stage]['duration']):
            self.current_stage += 1
            self.stage_progress = 0
            print(f"Advanced to curriculum stage: {self.stages[self.current_stage]['name']}")
        
        self.stage_progress += 1
        
        return self.get_current_config()
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current curriculum configuration."""
        return self.stages[self.current_stage].copy()
    
    def is_final_stage(self) -> bool:
        """Check if in final curriculum stage."""
        return self.current_stage >= len(self.stages) - 1
            