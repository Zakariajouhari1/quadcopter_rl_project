import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import yaml
import math
from enum import Enum

class MissionState(Enum):
    """Mission state enumeration for STL specification."""
    AT_A = "at_A"
    AT_B = "at_B" 
    AT_CHARGING = "at_charging"
    EN_ROUTE_A_TO_B = "en_route_A_to_B"
    EN_ROUTE_B_TO_CHARGING = "en_route_B_to_charging"
    EN_ROUTE_CHARGING_TO_A = "en_route_charging_to_A"

class STLRewardShaper:
    """
    Advanced reward shaping using Signal Temporal Logic (STL) robustness scores.
    Implements the mission specification: A → B → Charging → A (infinite loop)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()
        
        # Mission waypoints
        self.waypoints = {
            'A': np.array(self.config.get('waypoint_A', [0.0, 0.0, 1.0])),
            'B': np.array(self.config.get('waypoint_B', [5.0, 5.0, 1.5])), 
            'charging': np.array(self.config.get('waypoint_charging', [2.5, -2.5, 0.5]))
        }
        
        # STL parameters
        self.waypoint_tolerance = self.config.get('waypoint_tolerance', 0.3)
        self.time_horizon = self.config.get('stl_time_horizon', 50)  # Steps
        self.max_time_at_waypoint = self.config.get('max_time_at_waypoint', 20)  # Steps
        self.min_time_between_waypoints = self.config.get('min_time_between_waypoints', 10)  # Steps
        
        # Reward weights
        self.stl_weight = self.config.get('stl_weight', 1.0)
        self.distance_weight = self.config.get('distance_weight', 0.1)
        self.progress_weight = self.config.get('progress_weight', 0.5)
        self.safety_weight = self.config.get('safety_weight', 2.0)
        self.energy_weight = self.config.get('energy_weight', 0.01)
        
        # Safety constraints
        self.min_altitude = self.config.get('min_altitude', 0.2)
        self.max_altitude = self.config.get('max_altitude', 8.0)
        self.max_velocity = self.config.get('max_velocity', 10.0)
        self.max_angular_velocity = self.config.get('max_angular_velocity', 3.0)
        self.max_tilt_angle = self.config.get('max_tilt_angle', math.pi/4)  # 45 degrees
        
        # Trajectory history for STL evaluation
        self.trajectory_history = []
        self.max_history_length = self.time_horizon + 10
        
        # Mission timing
        self.mission_timer = 0
        self.last_waypoint_time = 0
        self.waypoint_visit_times = {'A': [], 'B': [], 'charging': []}
        
        # STL atomic propositions tracking
        self.current_mission_state = MissionState.AT_A
        self.mission_cycle_count = 0
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for reward shaping."""
        return {
            'waypoint_A': [0.0, 0.0, 1.0],
            'waypoint_B': [5.0, 5.0, 1.5],
            'waypoint_charging': [2.5, -2.5, 0.5],
            'waypoint_tolerance': 0.3,
            'stl_time_horizon': 50,
            'max_time_at_waypoint': 20,
            'min_time_between_waypoints': 10,
            'stl_weight': 1.0,
            'distance_weight': 0.1,
            'progress_weight': 0.5, 
            'safety_weight': 2.0,
            'energy_weight': 0.01,
            'min_altitude': 0.2,
            'max_altitude': 8.0,
            'max_velocity': 10.0,
            'max_angular_velocity': 3.0,
            'max_tilt_angle': math.pi/4
        }
    
    def reset(self):
        """Reset reward shaper for new episode."""
        self.trajectory_history = []
        self.mission_timer = 0
        self.last_waypoint_time = 0
        self.waypoint_visit_times = {'A': [], 'B': [], 'charging': []}
        self.current_mission_state = MissionState.AT_A
        self.mission_cycle_count = 0
    
    def calculate_reward(self, 
                        state: np.ndarray, 
                        action: np.ndarray,
                        next_state: np.ndarray,
                        info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive reward using STL robustness and auxiliary terms.
        
        Args:
            state: Current state [pos(3), orient(3), lin_vel(3), ang_vel(3)]
            action: Action taken [rotor_cmd(4)]
            next_state: Next state after action
            info: Additional environment information
            
        Returns:
            total_reward: Combined reward value
            reward_breakdown: Dictionary with individual reward components
        """
        
        # Extract state information
        position = next_state[:3]
        orientation = next_state[3:6]  # [roll, pitch, yaw]
        linear_velocity = next_state[6:9]
        angular_velocity = next_state[9:12]
        
        # Update trajectory history
        self._update_trajectory_history(next_state, info)
        
        # Calculate individual reward components
        reward_components = {}
        
        # 1. STL Robustness Score (main mission objective)
        stl_reward, stl_breakdown = self._calculate_stl_reward()
        reward_components['stl_robustness'] = stl_reward * self.stl_weight
        reward_components.update(stl_breakdown)
        
        # 2. Distance-based reward (immediate guidance)
        distance_reward = self._calculate_distance_reward(position, info)
        reward_components['distance'] = distance_reward * self.distance_weight
        
        # 3. Progress reward (improvement over time)
        progress_reward = self._calculate_progress_reward(info)
        reward_components['progress'] = progress_reward * self.progress_weight
        
        # 4. Safety constraints (critical for flight)
        safety_reward = self._calculate_safety_reward(position, orientation, 
                                                    linear_velocity, angular_velocity)
        reward_components['safety'] = safety_reward * self.safety_weight
        
        # 5. Energy efficiency (penalize excessive control)
        energy_reward = self._calculate_energy_reward(action)
        reward_components['energy'] = energy_reward * self.energy_weight
        
        # 6. Mission timing rewards
        timing_reward = self._calculate_timing_reward()
        reward_components['timing'] = timing_reward
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        self.mission_timer += 1
        
        return total_reward, reward_components
    
    def _update_trajectory_history(self, state: np.ndarray, info: Dict[str, Any]):
        """Update trajectory history for STL evaluation."""
        trajectory_point = {
            'time': self.mission_timer,
            'position': state[:3].copy(),
            'orientation': state[3:6].copy(),
            'linear_velocity': state[6:9].copy(),
            'angular_velocity': state[9:12].copy(),
            'current_target': info.get('current_target', 'B'),
            'mission_progress': info.get('mission_progress', {})
        }
        
        self.trajectory_history.append(trajectory_point)
        
        # Keep history within bounds
        if len(self.trajectory_history) > self.max_history_length:
            self.trajectory_history.pop(0)
        
        # Update mission state based on current position
        self._update_mission_state(state[:3], info)
    
    def _update_mission_state(self, position: np.ndarray, info: Dict[str, Any]):
        """Update current mission state based on position and progress."""
        # Check if at any waypoint
        at_A = np.linalg.norm(position - self.waypoints['A']) < self.waypoint_tolerance
        at_B = np.linalg.norm(position - self.waypoints['B']) < self.waypoint_tolerance
        at_charging = np.linalg.norm(position - self.waypoints['charging']) < self.waypoint_tolerance
        
        # Update waypoint visit times
        if at_A and self.current_mission_state != MissionState.AT_A:
            self.waypoint_visit_times['A'].append(self.mission_timer)
            self.last_waypoint_time = self.mission_timer
            if len(self.waypoint_visit_times['A']) > 1:  # Completed a cycle
                self.mission_cycle_count += 1
        elif at_B and self.current_mission_state != MissionState.AT_B:
            self.waypoint_visit_times['B'].append(self.mission_timer)
            self.last_waypoint_time = self.mission_timer
        elif at_charging and self.current_mission_state != MissionState.AT_CHARGING:
            self.waypoint_visit_times['charging'].append(self.mission_timer)
            self.last_waypoint_time = self.mission_timer
        
        # Update mission state
        if at_A:
            self.current_mission_state = MissionState.AT_A
        elif at_B:
            self.current_mission_state = MissionState.AT_B
        elif at_charging:
            self.current_mission_state = MissionState.AT_CHARGING
        else:
            # Determine en-route state based on target
            current_target = info.get('current_target', 'B')
            if current_target == 'B':
                self.current_mission_state = MissionState.EN_ROUTE_A_TO_B
            elif current_target == 'charging':
                self.current_mission_state = MissionState.EN_ROUTE_B_TO_CHARGING
            elif current_target == 'A':
                self.current_mission_state = MissionState.EN_ROUTE_CHARGING_TO_A
    
    def _calculate_stl_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate STL robustness score for the mission specification.
        
        STL Formula (simplified): G(F(at_B) → F(at_charging) → F(at_A))
        "Globally, eventually visit B, then eventually visit charging, then eventually return to A"
        """
        
        if len(self.trajectory_history) < 2:
            return 0.0, {'stl_sequence': 0.0, 'stl_timing': 0.0, 'stl_safety': 0.0}
        
        stl_breakdown = {}
        
        # 1. Sequence Robustness: Correct order A→B→Charging→A
        sequence_robustness = self._evaluate_sequence_robustness()
        stl_breakdown['stl_sequence'] = sequence_robustness
        
        # 2. Timing Robustness: Reasonable timing between waypoints
        timing_robustness = self._evaluate_timing_robustness()
        stl_breakdown['stl_timing'] = timing_robustness
        
        # 3. Safety Robustness: Always maintain safety constraints
        safety_robustness = self._evaluate_safety_robustness()
        stl_breakdown['stl_safety'] = safety_robustness
        
        # 4. Liveness Robustness: Progress toward next waypoint
        liveness_robustness = self._evaluate_liveness_robustness()
        stl_breakdown['stl_liveness'] = liveness_robustness
        
        # Combined STL robustness (weighted sum)
        total_stl_robustness = (
            0.4 * sequence_robustness +
            0.2 * timing_robustness +
            0.3 * safety_robustness +
            0.1 * liveness_robustness
        )
        
        return total_stl_robustness, stl_breakdown
    
    def _evaluate_sequence_robustness(self) -> float:
        """Evaluate robustness of mission sequence compliance."""
        if not self.waypoint_visit_times['B'] or not self.waypoint_visit_times['charging']:
            # Haven't completed basic sequence yet - give partial credit for progress
            if len(self.waypoint_visit_times['B']) > 0:
                return 0.3  # Reached B
            return 0.0
        
        # Check if sequence is correct: times should be increasing A < B < Charging < A
        robustness = 1.0
        
        # Get latest visits
        latest_A = self.waypoint_visit_times['A'][-1] if self.waypoint_visit_times['A'] else 0
        latest_B = self.waypoint_visit_times['B'][-1] if self.waypoint_visit_times['B'] else -1
        latest_charging = self.waypoint_visit_times['charging'][-1] if self.waypoint_visit_times['charging'] else -1
        
        # Check temporal ordering
        if latest_B > 0 and latest_charging > latest_B:
            if len(self.waypoint_visit_times['A']) > 1:  # Completed full cycle
                if latest_A > latest_charging:
                    robustness = 1.0  # Perfect sequence
                else:
                    robustness = 0.7  # Reached charging but not back to A
            else:
                robustness = 0.5  # Reached charging
        elif latest_B > 0:
            robustness = 0.3  # Only reached B
        else:
            robustness = 0.0  # No progress
        
        return robustness
    
    def _evaluate_timing_robustness(self) -> float:
        """Evaluate timing constraints robustness."""
        if len(self.trajectory_history) < self.min_time_between_waypoints:
            return 0.0
        
        # Penalize staying too long at waypoints or taking too long between them
        time_since_last_waypoint = self.mission_timer - self.last_waypoint_time
        
        if time_since_last_waypoint > self.max_time_at_waypoint * 3:
            # Taking too long to reach next waypoint
            penalty = min(1.0, (time_since_last_waypoint - self.max_time_at_waypoint * 3) / 50.0)
            return -penalty
        elif time_since_last_waypoint < self.min_time_between_waypoints:
            # Moving too fast (might be unrealistic)
            return 0.5
        else:
            # Good timing
            return 1.0
    
    def _evaluate_safety_robustness(self) -> float:
        """Evaluate safety constraints over trajectory history."""
        if not self.trajectory_history:
            return 0.0
        
        safety_violations = 0
        total_points = len(self.trajectory_history)
        
        for point in self.trajectory_history[-min(20, len(self.trajectory_history)):]:  # Check recent history
            pos = point['position']
            orient = point['orientation']
            lin_vel = point['linear_velocity']
            ang_vel = point['angular_velocity']
            
            # Check altitude constraints
            if pos[2] < self.min_altitude or pos[2] > self.max_altitude:
                safety_violations += 1
            
            # Check velocity constraints
            if np.linalg.norm(lin_vel) > self.max_velocity:
                safety_violations += 1
            
            # Check angular velocity constraints
            if np.linalg.norm(ang_vel) > self.max_angular_velocity:
                safety_violations += 1
            
            # Check tilt constraints
            if abs(orient[0]) > self.max_tilt_angle or abs(orient[1]) > self.max_tilt_angle:
                safety_violations += 1
        
        # Calculate safety robustness as 1 - violation_rate
        violation_rate = safety_violations / (total_points * 4)  # 4 safety checks per point
        return 1.0 - min(1.0, violation_rate)
    
    def _evaluate_liveness_robustness(self) -> float:
        """Evaluate progress toward mission objectives."""
        if len(self.trajectory_history) < 2:
            return 0.0
        
        # Check if making progress toward current target
        current_pos = self.trajectory_history[-1]['position']
        prev_pos = self.trajectory_history[-2]['position']
        
        current_target = self.trajectory_history[-1]['current_target']
        target_pos = self.waypoints[current_target]
        
        # Calculate distances
        current_dist = np.linalg.norm(current_pos - target_pos)
        prev_dist = np.linalg.norm(prev_pos - target_pos)
        
        # Positive robustness if getting closer, negative if moving away
        progress = prev_dist - current_dist
        
        # Normalize progress
        normalized_progress = np.tanh(progress * 5.0)  # Scale and bound to [-1, 1]
        
        return normalized_progress
    
    def _calculate_distance_reward(self, position: np.ndarray, info: Dict[str, Any]) -> float:
        """Calculate distance-based reward for immediate guidance."""
        current_target = info.get('current_target', 'B')
        target_pos = self.waypoints[current_target]
        distance = np.linalg.norm(position - target_pos)
        
        # Exponential decay reward based on distance
        distance_reward = -distance  # Linear penalty for distance
        
        # Bonus for being very close
        if distance < self.waypoint_tolerance:
            distance_reward += 10.0  # Arrival bonus
        elif distance < self.waypoint_tolerance * 2:
            distance_reward += 5.0   # Close approach bonus
        
        return distance_reward
    
    def _calculate_progress_reward(self, info: Dict[str, Any]) -> float:
        """Calculate reward for making progress toward mission completion."""
        progress = info.get('mission_progress', {})
        
        # Reward based on mission completion state
        progress_reward = 0.0
        
        if progress.get('B_visited', False):
            progress_reward += 20.0
        
        if progress.get('charging_visited', False):
            progress_reward += 20.0
        
        # Bonus for completing full cycles
        progress_reward += self.mission_cycle_count * 50.0
        
        return progress_reward
    
    def _calculate_safety_reward(self, position: np.ndarray, orientation: np.ndarray,
                                linear_velocity: np.ndarray, angular_velocity: np.ndarray) -> float:
        """Calculate safety-based rewards and penalties."""
        safety_reward = 0.0
        
        # Altitude safety
        if position[2] < self.min_altitude:
            safety_reward -= 100.0 * (self.min_altitude - position[2])  # Severe penalty for low altitude
        elif position[2] > self.max_altitude:
            safety_reward -= 10.0 * (position[2] - self.max_altitude)  # Penalty for high altitude
        else:
            safety_reward += 1.0  # Small bonus for safe altitude
        
        # Velocity safety
        speed = np.linalg.norm(linear_velocity)
        if speed > self.max_velocity:
            safety_reward -= 20.0 * (speed - self.max_velocity)
        
        # Angular velocity safety
        angular_speed = np.linalg.norm(angular_velocity)
        if angular_speed > self.max_angular_velocity:
            safety_reward -= 10.0 * (angular_speed - self.max_angular_velocity)
        
        # Orientation safety (tilt angles)
        roll, pitch = orientation[0], orientation[1]
        if abs(roll) > self.max_tilt_angle:
            safety_reward -= 50.0 * (abs(roll) - self.max_tilt_angle)
        if abs(pitch) > self.max_tilt_angle:
            safety_reward -= 50.0 * (abs(pitch) - self.max_tilt_angle)
        
        return safety_reward
    
    def _calculate_energy_reward(self, action: np.ndarray) -> float:
        """Calculate energy efficiency reward (penalize excessive control effort)."""
        # Penalize large control actions (energy consumption)
        control_effort = np.sum(np.square(action))
        energy_penalty = -control_effort
        
        # Penalize rapid changes in control (would require action history)
        # Simplified: penalize actions far from hover (action = 0 is hover)
        hover_penalty = -np.sum(np.square(action))
        
        return energy_penalty + hover_penalty
    
    def _calculate_timing_reward(self) -> float:
        """Calculate timing-based rewards for mission efficiency."""
        timing_reward = 0.0
        
        # Reward for reasonable mission timing
        time_since_last_waypoint = self.mission_timer - self.last_waypoint_time
        
        if time_since_last_waypoint > self.max_time_at_waypoint * 2:
            # Taking too long - penalty increases over time
            timing_reward -= (time_since_last_waypoint - self.max_time_at_waypoint * 2) * 0.1
        
        # Reward for completing mission cycles efficiently
        if self.mission_cycle_count > 0:
            avg_cycle_time = self.mission_timer / self.mission_cycle_count
            if avg_cycle_time < 200:  # Efficient cycle time
                timing_reward += 10.0
        
        return timing_reward
    
    def get_stl_robustness_metrics(self) -> Dict[str, float]:
        """Get detailed STL robustness metrics for analysis."""
        if not self.trajectory_history:
            return {}
        
        _, stl_breakdown = self._calculate_stl_reward()
        
        metrics = {
            'sequence_robustness': stl_breakdown.get('stl_sequence', 0.0),
            'timing_robustness': stl_breakdown.get('stl_timing', 0.0),
            'safety_robustness': stl_breakdown.get('stl_safety', 0.0),
            'liveness_robustness': stl_breakdown.get('stl_liveness', 0.0),
            'mission_cycles_completed': self.mission_cycle_count,
            'current_mission_state': self.current_mission_state.value,
            'waypoint_visits': {
                'A': len(self.waypoint_visit_times['A']),
                'B': len(self.waypoint_visit_times['B']),
                'charging': len(self.waypoint_visit_times['charging'])
            }
        }
        
        return metrics
    
    def evaluate_trajectory_stl_satisfaction(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Evaluate STL satisfaction for a complete trajectory (for post-episode analysis).
        
        Args:
            trajectory: List of trajectory points with position, time, etc.
            
        Returns:
            STL robustness score for the entire trajectory
        """
        
        # Simplified STL evaluation for complete trajectory
        # In practice, this would use a proper STL library like STLRom or STLCG++
        
        waypoint_visits = {'A': [], 'B': [], 'charging': []}
        
        # Extract waypoint visits from trajectory
        for i, point in enumerate(trajectory):
            pos = point['position']
            
            if np.linalg.norm(pos - self.waypoints['A']) < self.waypoint_tolerance:
                waypoint_visits['A'].append(i)
            elif np.linalg.norm(pos - self.waypoints['B']) < self.waypoint_tolerance:
                waypoint_visits['B'].append(i)
            elif np.linalg.norm(pos - self.waypoints['charging']) < self.waypoint_tolerance:
                waypoint_visits['charging'].append(i)
        
        # Check if sequence A→B→Charging→A is satisfied
        robustness = 0.0
        
        if waypoint_visits['B'] and waypoint_visits['charging']:
            # Basic sequence achieved
            robustness = 0.5
            
            # Check temporal ordering
            latest_B = max(waypoint_visits['B'])
            latest_charging = max(waypoint_visits['charging'])
            
            if latest_charging > latest_B:
                robustness = 0.7
                
                # Check if returned to A after charging
                A_after_charging = [t for t in waypoint_visits['A'] if t > latest_charging]
                if A_after_charging:
                    robustness = 1.0  # Full mission cycle completed
        
        return robustness