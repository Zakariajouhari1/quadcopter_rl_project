import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
import math
import yaml
from typing import Dict, List, Tuple, Optional, Any
import time

class QuadcopterEnv(gym.Env):
    """
    Custom OpenAI Gym environment for quadcopter navigation with STL objectives.
    Wraps PyBullet physics simulation with realistic dynamics, sensor noise, and delays.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super(QuadcopterEnv, self).__init__()
        
        # Load configuration
        self.config = config or self._load_default_config()
        
        # Environment parameters
        self.max_episode_steps = self.config.get('max_episode_steps', 1000)
        self.dt = self.config.get('simulation_dt', 1/240)  # 240 Hz simulation
        self.control_freq = self.config.get('control_freq', 48)  # 48 Hz control
        self.control_dt = 1 / self.control_freq
        self.steps_per_control = int(self.control_dt / self.dt)
        
        # Mission waypoints (A -> B -> Charging Station)
        self.waypoints = {
            'A': np.array(self.config.get('waypoint_A', [0.0, 0.0, 1.0])),
            'B': np.array(self.config.get('waypoint_B', [5.0, 5.0, 1.5])),
            'charging': np.array(self.config.get('waypoint_charging', [2.5, -2.5, 0.5]))
        }
        self.waypoint_sequence = ['B', 'charging', 'A']  # Starting from A, go to B, then charging, then back to A
        self.current_target_idx = 0
        self.waypoint_tolerance = self.config.get('waypoint_tolerance', 0.3)
        
        # Physics simulation setup
        self.physics_client = None
        self.drone_id = None
        self.plane_id = None
        
        # Quadcopter physical parameters
        self.mass = self.config.get('drone_mass', 0.5)  # kg
        self.arm_length = self.config.get('arm_length', 0.23)  # m
        self.motor_constant = self.config.get('motor_constant', 8.54858e-06)
        self.moment_constant = self.config.get('moment_constant', 0.016)
        
        # State and action spaces (following document specifications)
        # State: [pos(3), orientation(3), linear_vel(3), angular_vel(3)] = 12D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        # Action: [rotor1_cmd, rotor2_cmd, rotor3_cmd, rotor4_cmd] = 4D
        # Normalized to [-1, 1], then mapped to motor commands
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Sensor noise parameters (for realistic simulation)
        self.position_noise_std = self.config.get('position_noise_std', 0.01)
        self.velocity_noise_std = self.config.get('velocity_noise_std', 0.02)
        self.attitude_noise_std = self.config.get('attitude_noise_std', 0.005)
        self.angular_vel_noise_std = self.config.get('angular_vel_noise_std', 0.01)
        self.sensor_delay_steps = self.config.get('sensor_delay_steps', 2)
        
        # State estimation (sensor fusion simulation)
        self.state_history = []
        self.max_history_length = max(10, self.sensor_delay_steps + 5)
        
        # Domain randomization parameters
        self.domain_randomization = self.config.get('domain_randomization', True)
        self.mass_variation = self.config.get('mass_variation', 0.1)  # ±10%
        self.wind_disturbance_max = self.config.get('wind_disturbance_max', 0.5)  # m/s
        
        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        self.mission_progress = {'A_visited': False, 'B_visited': False, 'charging_visited': False}
        
        # Reward tracking for analysis
        self.cumulative_reward = 0.0
        self.last_distance_to_target = 0.0
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration parameters."""
        return {
            'max_episode_steps': 1000,
            'simulation_dt': 1/240,
            'control_freq': 48,
            'waypoint_A': [0.0, 0.0, 1.0],
            'waypoint_B': [5.0, 5.0, 1.5],
            'waypoint_charging': [2.5, -2.5, 0.5],
            'waypoint_tolerance': 0.3,
            'drone_mass': 0.5,
            'arm_length': 0.23,
            'motor_constant': 8.54858e-06,
            'moment_constant': 0.016,
            'position_noise_std': 0.01,
            'velocity_noise_std': 0.02,
            'attitude_noise_std': 0.005,
            'angular_vel_noise_std': 0.01,
            'sensor_delay_steps': 2,
            'domain_randomization': True,
            'mass_variation': 0.1,
            'wind_disturbance_max': 0.5
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize or reconnect PyBullet
        if self.physics_client is None:
            self.physics_client = bullet_client.BulletClient(connection_mode=p.DIRECT)
            self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        else:
            self.physics_client.resetSimulation()
        
        # Set up physics
        self.physics_client.setGravity(0, 0, -9.81)
        self.physics_client.setTimeStep(self.dt)
        self.physics_client.setRealTimeSimulation(0)
        
        # Load ground plane
        self.plane_id = self.physics_client.loadURDF("plane.urdf")
        
        # Domain randomization - vary drone mass
        if self.domain_randomization:
            mass_multiplier = 1.0 + np.random.uniform(-self.mass_variation, self.mass_variation)
            self.current_mass = self.mass * mass_multiplier
        else:
            self.current_mass = self.mass
        
        # Create quadcopter (simplified as a box with 4 sphere propellers)
        self._create_quadcopter()
        
        # Reset mission state
        self.current_target_idx = 0
        self.mission_progress = {'A_visited': True, 'B_visited': False, 'charging_visited': False}
        self.step_count = 0
        self.cumulative_reward = 0.0
        
        # Initialize state history for sensor delays
        self.state_history = []
        
        # Get initial observation
        obs = self._get_observation()
        self.last_distance_to_target = self._get_distance_to_current_target()
        
        info = {
            'current_target': self.waypoint_sequence[self.current_target_idx],
            'mission_progress': self.mission_progress.copy(),
            'episode': self.episode_count
        }
        
        self.episode_count += 1
        return obs, info
    
    def _create_quadcopter(self):
        """Create quadcopter model in PyBullet simulation."""
        # Main body (center of mass)
        body_collision = self.physics_client.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05]
        )
        body_visual = self.physics_client.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], rgbaColor=[0.2, 0.2, 0.8, 1.0]
        )
        
        # Starting position (Point A)
        start_pos = self.waypoints['A'] + np.random.uniform(-0.1, 0.1, 3)  # Small randomization
        start_orientation = self.physics_client.getQuaternionFromEuler([0, 0, np.random.uniform(-0.1, 0.1)])
        
        self.drone_id = self.physics_client.createMultiBody(
            baseMass=self.current_mass,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=start_pos,
            baseOrientation=start_orientation
        )
        
        # Set dynamics properties for realistic flight
        self.physics_client.changeDynamics(
            self.drone_id, -1,
            linearDamping=0.1,
            angularDamping=0.1,
            restitution=0.1,
            lateralFriction=0.3
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action for multiple simulation steps (control frequency vs simulation frequency)
        for _ in range(self.steps_per_control):
            self._apply_action(action)
            
            # Add wind disturbances (domain randomization)
            if self.domain_randomization:
                wind_force = np.random.uniform(
                    -self.wind_disturbance_max, 
                    self.wind_disturbance_max, 
                    3
                )
                self.physics_client.applyExternalForce(
                    self.drone_id, -1, wind_force, [0, 0, 0], p.WORLD_FRAME
                )
            
            self.physics_client.stepSimulation()
        
        # Get observation with sensor noise and delays
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated, truncated = self._check_termination()
        
        # Update mission progress
        self._update_mission_progress()
        
        # Prepare info
        info = {
            'current_target': self.waypoint_sequence[self.current_target_idx],
            'distance_to_target': self._get_distance_to_current_target(),
            'mission_progress': self.mission_progress.copy(),
            'cumulative_reward': self.cumulative_reward,
            'step': self.step_count
        }
        
        self.step_count += 1
        self.cumulative_reward += reward
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Convert normalized actions to motor forces and apply to quadcopter."""
        # Map actions from [-1, 1] to motor thrust range
        # Assuming hover thrust per motor is approximately mg/4
        hover_thrust_per_motor = (self.current_mass * 9.81) / 4
        max_thrust_per_motor = hover_thrust_per_motor * 2  # Allow 2x hover thrust
        
        # Convert normalized actions to actual thrusts
        motor_thrusts = (action + 1) / 2 * max_thrust_per_motor  # Map [-1,1] to [0, max_thrust]
        
        # Apply forces at motor positions (simplified - forces applied at body center with torques)
        total_thrust = np.sum(motor_thrusts)
        
        # Calculate torques based on motor layout (X configuration)
        # Motor layout: 1(front-right), 2(back-left), 3(front-left), 4(back-right)
        # Rotation directions: 1&2 CCW, 3&4 CW
        
        # Roll torque (around x-axis)
        roll_torque = self.arm_length * (motor_thrusts[0] + motor_thrusts[3] - motor_thrusts[1] - motor_thrusts[2])
        
        # Pitch torque (around y-axis)  
        pitch_torque = self.arm_length * (motor_thrusts[0] + motor_thrusts[2] - motor_thrusts[1] - motor_thrusts[3])
        
        # Yaw torque (around z-axis) - from motor rotation resistance
        yaw_torque = self.moment_constant * (motor_thrusts[1] + motor_thrusts[2] - motor_thrusts[0] - motor_thrusts[3])
        
        # Apply vertical thrust
        self.physics_client.applyExternalForce(
            self.drone_id, -1, [0, 0, total_thrust], [0, 0, 0], p.LINK_FRAME
        )
        
        # Apply torques
        self.physics_client.applyExternalTorque(
            self.drone_id, -1, [roll_torque, pitch_torque, yaw_torque], p.LINK_FRAME
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with sensor noise and delays."""
        # Get true state from simulation
        pos, quat = self.physics_client.getBasePositionAndOrientation(self.drone_id)
        linear_vel, angular_vel = self.physics_client.getBaseVelocity(self.drone_id)
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        euler = self.physics_client.getEulerFromQuaternion(quat)
        
        # Create clean state vector
        true_state = np.array([
            pos[0], pos[1], pos[2],                    # position
            euler[0], euler[1], euler[2],              # orientation (roll, pitch, yaw)
            linear_vel[0], linear_vel[1], linear_vel[2], # linear velocity
            angular_vel[0], angular_vel[1], angular_vel[2] # angular velocity
        ], dtype=np.float32)
        
        # Add sensor noise (simulating IMU, barometer, etc.)
        noisy_state = true_state.copy()
        
        # Position noise (GPS/visual positioning)
        noisy_state[0:3] += np.random.normal(0, self.position_noise_std, 3)
        
        # Attitude noise (IMU)
        noisy_state[3:6] += np.random.normal(0, self.attitude_noise_std, 3)
        
        # Linear velocity noise
        noisy_state[6:9] += np.random.normal(0, self.velocity_noise_std, 3)
        
        # Angular velocity noise (gyroscope)
        noisy_state[9:12] += np.random.normal(0, self.angular_vel_noise_std, 3)
        
        # Store in history for sensor delays
        self.state_history.append(noisy_state)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
        
        # Apply sensor delay
        if len(self.state_history) > self.sensor_delay_steps:
            delayed_state = self.state_history[-(self.sensor_delay_steps + 1)]
        else:
            delayed_state = noisy_state
        
        return delayed_state
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on mission progress and distance to target."""
        # This is a basic reward function - will be enhanced with STL robustness
        current_pos = np.array(self.physics_client.getBasePositionAndOrientation(self.drone_id)[0])
        current_target = self.waypoints[self.waypoint_sequence[self.current_target_idx]]
        
        # Distance-based reward
        distance_to_target = np.linalg.norm(current_pos - current_target)
        distance_reward = -distance_to_target  # Negative distance encourages getting closer
        
        # Progress reward (improvement in distance)
        progress_reward = (self.last_distance_to_target - distance_to_target) * 10
        self.last_distance_to_target = distance_to_target
        
        # Waypoint reached bonus
        waypoint_bonus = 0.0
        if distance_to_target < self.waypoint_tolerance:
            waypoint_bonus = 100.0  # Large bonus for reaching waypoint
        
        # Penalties
        altitude_penalty = 0.0
        if current_pos[2] < 0.1:  # Too close to ground
            altitude_penalty = -50.0
        elif current_pos[2] > 10.0:  # Too high
            altitude_penalty = -10.0
        
        # Energy penalty (penalize excessive control effort)
        # This would require storing the last action, simplified for now
        energy_penalty = 0.0
        
        # Stability penalty (penalize excessive angular velocities)
        _, angular_vel = self.physics_client.getBaseVelocity(self.drone_id)
        angular_vel_magnitude = np.linalg.norm(angular_vel)
        stability_penalty = -angular_vel_magnitude * 0.1
        
        total_reward = (distance_reward + progress_reward + waypoint_bonus + 
                       altitude_penalty + energy_penalty + stability_penalty)
        
        return total_reward
    
    def _update_mission_progress(self):
        """Update mission progress and target waypoint."""
        current_pos = np.array(self.physics_client.getBasePositionAndOrientation(self.drone_id)[0])
        current_target = self.waypoints[self.waypoint_sequence[self.current_target_idx]]
        
        # Check if current target is reached
        if np.linalg.norm(current_pos - current_target) < self.waypoint_tolerance:
            target_name = self.waypoint_sequence[self.current_target_idx]
            
            # Update mission progress
            if target_name == 'B':
                self.mission_progress['B_visited'] = True
            elif target_name == 'charging':
                self.mission_progress['charging_visited'] = True
            elif target_name == 'A':
                self.mission_progress['A_visited'] = True
                # Reset for next cycle
                self.mission_progress = {'A_visited': True, 'B_visited': False, 'charging_visited': False}
            
            # Move to next target
            self.current_target_idx = (self.current_target_idx + 1) % len(self.waypoint_sequence)
            self.last_distance_to_target = self._get_distance_to_current_target()
    
    def _get_distance_to_current_target(self) -> float:
        """Get distance to current target waypoint."""
        current_pos = np.array(self.physics_client.getBasePositionAndOrientation(self.drone_id)[0])
        current_target = self.waypoints[self.waypoint_sequence[self.current_target_idx]]
        return np.linalg.norm(current_pos - current_target)
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
        terminated = False
        truncated = False
        
        # Check for crashes (too close to ground with high velocity or tilted)
        pos, quat = self.physics_client.getBasePositionAndOrientation(self.drone_id)
        linear_vel, angular_vel = self.physics_client.getBaseVelocity(self.drone_id)
        euler = self.physics_client.getEulerFromQuaternion(quat)
        
        # Crash conditions
        if pos[2] < 0.05:  # Hit ground
            terminated = True
        elif abs(euler[0]) > math.pi/3 or abs(euler[1]) > math.pi/3:  # Too tilted
            terminated = True
        elif np.linalg.norm(linear_vel) > 15.0:  # Too fast
            terminated = True
        elif np.linalg.norm(pos[:2]) > 20.0:  # Too far from origin
            terminated = True
        
        # Episode length limit
        if self.step_count >= self.max_episode_steps:
            truncated = True
        
        return terminated, truncated
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get detailed current state for analysis."""
        pos, quat = self.physics_client.getBasePositionAndOrientation(self.drone_id)
        linear_vel, angular_vel = self.physics_client.getBaseVelocity(self.drone_id)
        euler = self.physics_client.getEulerFromQuaternion(quat)
        
        return {
            'position': np.array(pos),
            'orientation_quat': np.array(quat),
            'orientation_euler': np.array(euler),
            'linear_velocity': np.array(linear_vel),
            'angular_velocity': np.array(angular_vel),
            'current_target': self.waypoint_sequence[self.current_target_idx],
            'distance_to_target': self._get_distance_to_current_target(),
            'mission_progress': self.mission_progress.copy()
        }
    
    def close(self):
        """Clean up resources."""
        if self.physics_client is not None:
            self.physics_client.disconnect()
            self.physics_client = None
    
    def render(self, mode='human'):
        """Render the environment (basic implementation)."""
        if mode == 'human':
            # For visualization, you could add GUI mode to PyBullet
            # This is a placeholder for rendering functionality
            state = self.get_current_state()
            print(f"Step {self.step_count}: Pos={state['position']:.2f}, "
                  f"Target={state['current_target']}, "
                  f"Distance={state['distance_to_target']:.2f}")
