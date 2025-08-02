#!/usr/bin/env python3
"""
Main training script for Autonomous Quadcopter Navigation with TD3 and STL.
This script orchestrates the entire training process with comprehensive logging,
evaluation, and model checkpointing.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
import multiprocessing as mp
from pathlib import Path
import logging
from collections import deque

# Import our custom modules
from quadcopter_env import QuadcopterEnv
from td3_agent import TD3Agent, CurriculumLearning
from reward_shaping import STLRewardShaper
from stl_parser import (create_quadcopter_mission_formula, STLRobustnessCalculator, 
                       STLMonitor, AdaptiveSTLReward, trajectory_to_state_dicts)
from data_logger import DataLogger
from networks import NetworkFactory

class TrainingConfig:
    """Configuration manager for training parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        
        default_config = {
            # Environment parameters
            'env': {
                'max_episode_steps': 1000,
                'simulation_dt': 1/240,
                'control_freq': 48,
                'waypoint_A': [0.0, 0.0, 1.0],
                'waypoint_B': [5.0, 5.0, 1.5],
                'waypoint_charging': [2.5, -2.5, 0.5],
                'waypoint_tolerance': 0.3,
                'domain_randomization': True,
                'wind_disturbance_max': 0.5
            },
            
            # TD3 Agent parameters
            'agent': {
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
                'warmup_steps': 1000
            },
            
            # Network architecture
            'network': {
                'architecture': 'standard',  # 'standard', 'ensemble'
                'use_attention': False,
                'use_residual': False,
                'use_layer_norm': True,
                'dropout': 0.1,
                'use_spectral_norm': False,
                'hidden_dims': [400, 300]
            },
            
            # Training parameters
            'training': {
                'total_episodes': 5000,
                'max_steps_per_episode': 1000,
                'eval_episodes': 10,
                'eval_frequency': 100,
                'save_frequency': 500,
                'log_frequency': 10,
                'parallel_envs': 4,
                'use_curriculum': True,
                'early_stopping_patience': 1000
            },
            
            # STL parameters
            'stl': {
                'use_stl_reward': True,
                'stl_weight': 1.0,
                'adaptive_stl': True,
                'mission_type': 'surveillance'
            },
            
            # Logging and output
            'output': {
                'log_dir': './logs',
                'model_dir': './models',
                'plot_dir': './plots',
                'experiment_name': 'quadcopter_td3_stl',
                'save_trajectories': True,
                'plot_frequency': 200
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Deep merge configurations
            self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, section: str) -> Dict[str, Any]:
        """Get configuration section."""
        return self.config.get(section, {})
    
    def save(self, filepath: str):
        """Save current configuration to file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

class ExperimentManager:
    """Manages experiment setup, logging, and results tracking."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.output_config = config.get('output')
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.output_config['experiment_name']}_{timestamp}"
        self.experiment_dir = Path(self.output_config['log_dir']) / self.experiment_name
        
        # Create subdirectories
        self.log_dir = self.experiment_dir / 'logs'
        self.model_dir = self.experiment_dir / 'models'
        self.plot_dir = self.experiment_dir / 'plots'
        self.trajectory_dir = self.experiment_dir / 'trajectories'
        
        for dir_path in [self.log_dir, self.model_dir, self.plot_dir, self.trajectory_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Save configuration
        config.save(self.experiment_dir / 'config.yaml')
        
        # Initialize data logger
        self.data_logger = DataLogger(self.log_dir)
        
        logging.info(f"Experiment initialized: {self.experiment_name}")
        logging.info(f"Experiment directory: {self.experiment_dir}")
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.log_dir / 'training.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

class ParallelEnvironment:
    """Manages parallel environments for efficient data collection."""
    
    def __init__(self, env_config: Dict[str, Any], num_envs: int = 4):
        self.env_config = env_config
        self.num_envs = num_envs
        self.envs = []
        
        # Create environments
        for i in range(num_envs):
            env = QuadcopterEnv(env_config)
            self.envs.append(env)
    
    def reset_all(self) -> List[np.ndarray]:
        """Reset all environments."""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        return observations
    
    def step_all(self, actions: List[np.ndarray]) -> List[tuple]:
        """Step all environments."""
        results = []
        for env, action in zip(self.envs, actions):
            result = env.step(action)
            results.append(result)
        return results
    
    def close_all(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

class TrainingManager:
    """Main training manager that orchestrates the entire training process."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = TrainingConfig(config_path)
        
        # Setup experiment
        self.experiment = ExperimentManager(self.config)
        
        # Initialize components
        self.setup_environment()
        self.setup_agent()
        self.setup_stl_components()
        self.setup_curriculum()
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_eval_score = -float('inf')
        self.no_improvement_count = 0
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.stl_robustness = deque(maxlen=100)
        
        logging.info("Training manager initialized successfully")
    
    def setup_environment(self):
        """Setup training and evaluation environments."""
        env_config = self.config.get('env')
        training_config = self.config.get('training')
        
        # Training environment (parallel)
        if training_config['parallel_envs'] > 1:
            self.train_env = ParallelEnvironment(env_config, training_config['parallel_envs'])
        else:
            self.train_env = QuadcopterEnv(env_config)
        
        # Evaluation environment (single)
        self.eval_env = QuadcopterEnv(env_config)
        
        # Get environment dimensions
        if hasattr(self.train_env, 'envs'):
            sample_env = self.train_env.envs[0]
        else:
            sample_env = self.train_env
        
        self.state_dim = sample_env.observation_space.shape[0]
        self.action_dim = sample_env.action_space.shape[0]
        self.max_action = float(sample_env.action_space.high[0])
        
        logging.info(f"Environment setup - State dim: {self.state_dim}, Action dim: {self.action_dim}")
    
    def setup_agent(self):
        """Setup TD3 agent with network configuration."""
        agent_config = self.config.get('agent')
        network_config = self.config.get('network')
        
        # Merge network config into agent config
        agent_config.update({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_action': self.max_action
        })
        
        self.agent = TD3Agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            config=agent_config
        )
        
        logging.info("TD3 agent initialized")
        logging.info(f"Agent configuration: {agent_config}")
    
    def setup_stl_components(self):
        """Setup STL-related components."""
        stl_config = self.config.get('stl')
        env_config = self.config.get('env')
        
        if not stl_config['use_stl_reward']:
            self.stl_reward_shaper = None
            self.stl_monitor = None
            return
        
        # Create waypoints dictionary
        waypoints = {
            'A': np.array(env_config['waypoint_A']),
            'B': np.array(env_config['waypoint_B']),
            'charging': np.array(env_config['waypoint_charging'])
        }
        
        # Create STL formula
        self.stl_formula = create_quadcopter_mission_formula(
            waypoints,
            mission_type=stl_config['mission_type'],
            tolerance=env_config['waypoint_tolerance']
        )
        
        # Setup reward shaper
        if stl_config['adaptive_stl']:
            self.stl_reward_shaper = AdaptiveSTLReward(self.stl_formula, stl_config['stl_weight'])
        else:
            self.stl_reward_shaper = STLRewardShaper(self.config.get('env'))
        
        # Setup monitor
        self.stl_monitor = STLMonitor(self.stl_formula)
        
        logging.info("STL components initialized")
        logging.info(f"STL formula: {self.stl_formula}")
    
    def setup_curriculum(self):
        """Setup curriculum learning if enabled."""
        training_config = self.config.get('training')
        
        if training_config['use_curriculum']:
            self.curriculum = CurriculumLearning(training_config)
            logging.info("Curriculum learning enabled")
        else:
            self.curriculum = None
    
    def train(self):
        """Main training loop."""
        training_config = self.config.get('training')
        total_episodes = training_config['total_episodes']
        
        logging.info(f"Starting training for {total_episodes} episodes")
        
        try:
            for episode in range(total_episodes):
                self.episode = episode
                
                # Update curriculum if enabled
                if self.curriculum:
                    curriculum_config = self.curriculum.update(self.total_steps)
                    # Apply curriculum modifications to environment
                
                # Train one episode
                episode_metrics = self.train_episode()
                
                # Log episode results
                self.log_episode_results(episode_metrics)
                
                # Evaluate periodically
                if episode % training_config['eval_frequency'] == 0:
                    eval_metrics = self.evaluate()
                    self.log_evaluation_results(eval_metrics)
                    
                    # Check for improvement
                    if eval_metrics['avg_reward'] > self.best_eval_score:
                        self.best_eval_score = eval_metrics['avg_reward']
                        self.no_improvement_count = 0
                        self.save_best_model()
                    else:
                        self.no_improvement_count += training_config['eval_frequency']
                
                # Save model periodically
                if episode % training_config['save_frequency'] == 0:
                    self.save_checkpoint(episode)
                
                # Plot results periodically
                if episode % self.config.get('output')['plot_frequency'] == 0:
                    self.plot_training_progress()
                
                # Early stopping check
                if (self.no_improvement_count >= training_config['early_stopping_patience'] and 
                    episode > training_config['early_stopping_patience']):
                    logging.info(f"Early stopping triggered after {episode} episodes")
                    break
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training failed with error: {e}")
            raise e
        finally:
            self.cleanup()
        
        logging.info("Training completed successfully")
    
    def train_episode(self) -> Dict[str, Any]:
        """Train a single episode."""
        # Reset environment
        if hasattr(self.train_env, 'envs'):
            observations = self.train_env.reset_all()
        else:
            obs, _ = self.train_env.reset()
            observations = [obs]
        
        episode_reward = 0
        episode_length = 0
        episode_trajectory = []
        episode_stl_robustness = []
        
        done_flags = [False] * len(observations)
        
        while not all(done_flags) and episode_length < self.config.get('training')['max_steps_per_episode']:
            # Select actions
            actions = []
            for obs in observations:
                action = self.agent.select_action(obs, add_noise=True)
                actions.append(action)
            
            # Step environments
            if hasattr(self.train_env, 'envs'):
                results = self.train_env.step_all(actions)
            else:
                result = self.train_env.step(actions[0])
                results = [result]
            
            # Process results
            for i, (obs, action, result) in enumerate(zip(observations, actions, results)):
                if done_flags[i]:
                    continue
                
                next_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                
                # Store trajectory for STL evaluation
                if self.config.get('output')['save_trajectories']:
                    state_dict = {
                        'position': next_obs[:3],
                        'orientation': next_obs[3:6],
                        'linear_velocity': next_obs[6:9],
                        'angular_velocity': next_obs[9:12],
                        'time': episode_length
                    }
                    episode_trajectory.append(state_dict)
                
                # Compute STL-enhanced reward
                if self.stl_reward_shaper and hasattr(self.stl_reward_shaper, 'calculate_reward'):
                    stl_reward, reward_breakdown = self.stl_reward_shaper.calculate_reward(
                        obs, action, next_obs, info
                    )
                    reward = stl_reward
                elif self.stl_reward_shaper and hasattr(self.stl_reward_shaper, 'compute_adaptive_reward'):
                    # For AdaptiveSTLReward
                    success_rate = np.mean(self.success_rates) if self.success_rates else 0.0
                    trajectory_states = trajectory_to_state_dicts([next_obs], {})
                    stl_reward = self.stl_reward_shaper.compute_adaptive_reward(
                        trajectory_states, self.episode, success_rate
                    )
                    reward += stl_reward
                
                # Update STL monitor
                if self.stl_monitor:
                    state_dict = {
                        'position': next_obs[:3],
                        'orientation': next_obs[3:6],
                        'linear_velocity': next_obs[6:9],
                        'angular_velocity': next_obs[9:12]
                    }
                    robustness, satisfied = self.stl_monitor.update(state_dict)
                    episode_stl_robustness.append(robustness)
                
                # Store experience in replay buffer
                self.agent.add_experience(obs, action, reward, next_obs, done)
                
                # Train agent
                if self.agent.replay_buffer.size() >= self.agent.batch_size:
                    training_metrics = self.agent.train()
                    
                # Update state
                observations[i] = next_obs
                episode_reward += reward
                done_flags[i] = done
            
            episode_length += 1
            self.total_steps += 1
        
        # Episode finished - compute final metrics
        episode_metrics = {
            'episode': self.episode,
            'reward': episode_reward / len(observations),  # Average across parallel envs
            'length': episode_length,
            'total_steps': self.total_steps,
            'stl_robustness': np.mean(episode_stl_robustness) if episode_stl_robustness else 0.0,
            'agent_stats': self.agent.get_training_stats()
        }
        
        # Store trajectory if enabled
        if (self.config.get('output')['save_trajectories'] and 
            self.episode % 50 == 0):  # Save every 50 episodes
            trajectory_path = self.experiment.trajectory_dir / f'episode_{self.episode}_trajectory.npy'
            np.save(trajectory_path, episode_trajectory)
        
        # Update performance tracking
        self.episode_rewards.append(episode_metrics['reward'])
        self.episode_lengths.append(episode_metrics['length'])
        self.stl_robustness.append(episode_metrics['stl_robustness'])
        
        # Determine success (can be customized based on mission requirements)
        success = episode_metrics['stl_robustness'] > 0 or episode_metrics['reward'] > 200
        self.success_rates.append(float(success))
        episode_metrics['success'] = success
        
        return episode_metrics
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the current policy."""
        training_config = self.config.get('training')
        eval_episodes = training_config['eval_episodes']
        
        logging.info(f"Starting evaluation for {eval_episodes} episodes")
        
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        eval_stl_robustness = []
        
        for eval_ep in range(eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_trajectory = []
            done = False
            
            while not done and episode_length < training_config['max_steps_per_episode']:
                # Select action without noise
                action = self.agent.select_action(obs, add_noise=False)
                next_obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                # Store trajectory for STL evaluation
                state_dict = {
                    'position': next_obs[:3],
                    'orientation': next_obs[3:6],
                    'linear_velocity': next_obs[6:9],
                    'angular_velocity': next_obs[9:12],
                    'time': episode_length
                }
                episode_trajectory.append(state_dict)
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            # Evaluate STL satisfaction
            if self.stl_formula:
                calculator = STLRobustnessCalculator(self.stl_formula)
                stl_robustness = calculator.compute_overall_robustness(episode_trajectory)
                eval_stl_robustness.append(stl_robustness)
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            # Determine success
            success = (len(eval_stl_robustness) > 0 and eval_stl_robustness[-1] > 0) or episode_reward > 200
            eval_successes.append(success)
        
        eval_metrics = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_successes),
            'avg_stl_robustness': np.mean(eval_stl_robustness) if eval_stl_robustness else 0.0,
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards)
        }
        
        logging.info(f"Evaluation completed - Avg reward: {eval_metrics['avg_reward']:.2f}, "
                    f"Success rate: {eval_metrics['success_rate']:.2f}")
        
        return eval_metrics
    
    def log_episode_results(self, metrics: Dict[str, Any]):
        """Log episode results to various outputs."""
        if self.episode % self.config.get('training')['log_frequency'] == 0:
            # Console logging
            logging.info(
                f"Episode {metrics['episode']}: "
                f"Reward={metrics['reward']:.2f}, "
                f"Length={metrics['length']}, "
                f"STL Robustness={metrics['stl_robustness']:.3f}, "
                f"Success={metrics['success']}"
            )
            
            # Data logger
            self.experiment.data_logger.log_scalar('train/episode_reward', metrics['reward'], metrics['episode'])
            self.experiment.data_logger.log_scalar('train/episode_length', metrics['length'], metrics['episode'])
            self.experiment.data_logger.log_scalar('train/stl_robustness', metrics['stl_robustness'], metrics['episode'])
            self.experiment.data_logger.log_scalar('train/success_rate', np.mean(self.success_rates), metrics['episode'])
            
            # Agent statistics
            agent_stats = metrics['agent_stats']
            if 'avg_actor_loss' in agent_stats:
                self.experiment.data_logger.log_scalar('train/actor_loss', agent_stats['avg_actor_loss'], metrics['episode'])
            if 'avg_critic_loss' in agent_stats:
                self.experiment.data_logger.log_scalar('train/critic_loss', agent_stats['avg_critic_loss'], metrics['episode'])
            if 'avg_q_value' in agent_stats:
                self.experiment.data_logger.log_scalar('train/q_value', agent_stats['avg_q_value'], metrics['episode'])
            
            self.experiment.data_logger.log_scalar('train/exploration_noise', agent_stats['exploration_noise'], metrics['episode'])
            self.experiment.data_logger.log_scalar('train/buffer_size', agent_stats['buffer_size'], metrics['episode'])
    
    def log_evaluation_results(self, metrics: Dict[str, Any]):
        """Log evaluation results."""
        logging.info(
            f"Evaluation at episode {self.episode}: "
            f"Avg Reward={metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}, "
            f"Success Rate={metrics['success_rate']:.2f}, "
            f"STL Robustness={metrics['avg_stl_robustness']:.3f}"
        )
        
        # Data logger
        self.experiment.data_logger.log_scalar('eval/avg_reward', metrics['avg_reward'], self.episode)
        self.experiment.data_logger.log_scalar('eval/success_rate', metrics['success_rate'], self.episode)
        self.experiment.data_logger.log_scalar('eval/avg_stl_robustness', metrics['avg_stl_robustness'], self.episode)
        self.experiment.data_logger.log_scalar('eval/avg_length', metrics['avg_length'], self.episode)
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_path = self.experiment.model_dir / f'checkpoint_episode_{episode}'
        self.agent.save_models(str(checkpoint_path))
        
        # Save additional training state
        training_state = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_eval_score': self.best_eval_score,
            'no_improvement_count': self.no_improvement_count,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'success_rates': list(self.success_rates),
            'stl_robustness': list(self.stl_robustness)
        }
        
        import pickle
        with open(checkpoint_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(training_state, f)
        
        logging.info(f"Checkpoint saved at episode {episode}")
    
    def save_best_model(self):
        """Save the best performing model."""
        best_model_path = self.experiment.model_dir / 'best_model'
        self.agent.save_models(str(best_model_path))
        logging.info(f"Best model saved with score: {self.best_eval_score:.2f}")
    
    def plot_training_progress(self):
        """Plot training progress and save figures."""
        if len(self.episode_rewards) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Episode {self.episode}', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(list(self.episode_rewards))
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Success rate
        if len(self.success_rates) > 10:
            # Moving average success rate
            window_size = min(50, len(self.success_rates) // 4)
            success_ma = np.convolve(self.success_rates, np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(success_ma)
        axes[0, 1].set_title('Success Rate (Moving Average)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(list(self.episode_lengths))
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # STL robustness
        axes[1, 1].plot(list(self.stl_robustness))
        axes[1, 1].set_title('STL Robustness')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Robustness Score')
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Satisfaction Threshold')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.experiment.plot_dir / f'training_progress_episode_{self.episode}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training progress plot saved: {plot_path}")
    
    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        
        if hasattr(self.train_env, 'close_all'):
            self.train_env.close_all()
        else:
            self.train_env.close()
        
        self.eval_env.close()
        
        # Final save
        self.save_checkpoint(self.episode)
        self.plot_training_progress()
        
        # Close data logger
        self.experiment.data_logger.close()
        
        logging.info("Cleanup completed")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train TD3 agent for quadcopter navigation with STL')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation, no training')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    
    args = parser.parse_args()
    
    # Initialize training manager
    trainer = TrainingManager(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        # Implementation for resuming training
        logging.info(f"Resuming training from checkpoint: {args.resume}")
        # Load checkpoint and restore training state
        pass
    
    if args.eval_only:
        # Only run evaluation
        logging.info("Running evaluation only")
        eval_metrics = trainer.evaluate()
        print("Evaluation Results:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value}")
    else:
        # Run training
        trainer.train()

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    main()
                