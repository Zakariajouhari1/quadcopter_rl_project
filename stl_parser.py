import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from abc import ABC, abstractmethod
import math
from enum import Enum
from dataclasses import dataclass
import re

class STLOperator(Enum):
    """STL temporal and logical operators."""
    # Temporal operators
    ALWAYS = "G"          # Globally (Always)
    EVENTUALLY = "F"      # Finally (Eventually)
    UNTIL = "U"          # Until
    NEXT = "X"           # Next
    
    # Logical operators
    AND = "∧"            # Conjunction
    OR = "∨"             # Disjunction
    NOT = "¬"            # Negation
    IMPLIES = "→"        # Implication
    
    # Atomic predicates
    PREDICATE = "p"      # Atomic predicate

@dataclass
class TimeInterval:
    """Time interval for temporal operators."""
    lower: float
    upper: float
    
    def __post_init__(self):
        if self.lower > self.upper:
            raise ValueError(f"Invalid interval: [{self.lower}, {self.upper}]")
    
    def contains(self, time: float) -> bool:
        """Check if time is within interval."""
        return self.lower <= time <= self.upper
    
    def __str__(self) -> str:
        return f"[{self.lower}, {self.upper}]"

class STLFormula(ABC):
    """Abstract base class for STL formulas."""
    
    @abstractmethod
    def evaluate_robustness(self, trajectory: List[Dict[str, Any]], time: int) -> float:
        """Evaluate robustness score at given time."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the formula."""
        pass

class AtomicPredicate(STLFormula):
    """Atomic predicate in STL formula."""
    
    def __init__(self, name: str, predicate_func: Callable[[Dict[str, Any]], float]):
        self.name = name
        self.predicate_func = predicate_func
    
    def evaluate_robustness(self, trajectory: List[Dict[str, Any]], time: int) -> float:
        """Evaluate atomic predicate robustness."""
        if time >= len(trajectory):
            return -float('inf')  # Outside trajectory bounds
        
        return self.predicate_func(trajectory[time])
    
    def __str__(self) -> str:
        return self.name

class Negation(STLFormula):
    """Negation operator."""
    
    def __init__(self, formula: STLFormula):
        self.formula = formula
    
    def evaluate_robustness(self, trajectory: List[Dict[str, Any]], time: int) -> float:
        """Evaluate negation robustness."""
        return -self.formula.evaluate_robustness(trajectory, time)
    
    def __str__(self) -> str:
        return f"¬({self.formula})"

class Conjunction(STLFormula):
    """Conjunction (AND) operator."""
    
    def __init__(self, left: STLFormula, right: STLFormula):
        self.left = left
        self.right = right
    
    def evaluate_robustness(self, trajectory: List[Dict[str, Any]], time: int) -> float:
        """Evaluate conjunction robustness (minimum)."""
        left_rob = self.left.evaluate_robustness(trajectory, time)
        right_rob = self.right.evaluate_robustness(trajectory, time)
        return min(left_rob, right_rob)
    
    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"

class Disjunction(STLFormula):
    """Disjunction (OR) operator."""
    
    def __init__(self, left: STLFormula, right: STLFormula):
        self.left = left
        self.right = right
    
    def evaluate_robustness(self, trajectory: List[Dict[str, Any]], time: int) -> float:
        """Evaluate disjunction robustness (maximum)."""
        left_rob = self.left.evaluate_robustness(trajectory, time)
        right_rob = self.right.evaluate_robustness(trajectory, time)
        return max(left_rob, right_rob)
    
    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"

class Implication(STLFormula):
    """Implication operator."""
    
    def __init__(self, antecedent: STLFormula, consequent: STLFormula):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluate_robustness(self, trajectory: List[Dict[str, Any]], time: int) -> float:
        """Evaluate implication robustness (¬A ∨ B)."""
        ant_rob = self.antecedent.evaluate_robustness(trajectory, time)
        cons_rob = self.consequent.evaluate_robustness(trajectory, time)
        return max(-ant_rob, cons_rob)
    
    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"

class Always(STLFormula):
    """Always (Globally) temporal operator."""
    
    def __init__(self, formula: STLFormula, interval: Optional[TimeInterval] = None):
        self.formula = formula
        self.interval = interval or TimeInterval(0, float('inf'))
    
    def evaluate_robustness(self, trajectory: List[Dict[str, Any]], time: int) -> float:
        """Evaluate always robustness (minimum over interval)."""
        min_robustness = float('inf')
        
        # Determine time bounds
        start_time = max(0, time + int(self.interval.lower))
        end_time = min(len(trajectory) - 1, time + int(self.interval.upper))
        
        if start_time > end_time:
            return -float('inf')  # Invalid interval
        
        for t in range(start_time, end_time + 1):
            robustness = self.formula.evaluate_robustness(trajectory, t)
            min_robustness = min(min_robustness, robustness)
        
        return min_robustness
    
    def __str__(self) -> str:
        return f"G{self.interval}({self.formula})"

class Eventually(STLFormula):
    """Eventually (Finally) temporal operator."""
    
    def __init__(self, formula: STLFormula, interval: Optional[TimeInterval] = None):
        self.formula = formula
        self.interval = interval or TimeInterval(0, float('inf'))
    
    def evaluate_robustness(self, trajectory: List[Dict[str, Any]], time: int) -> float:
        """Evaluate eventually robustness (maximum over interval)."""
        max_robustness = -float('inf')
        
        # Determine time bounds
        start_time = max(0, time + int(self.interval.lower))
        end_time = min(len(trajectory) - 1, time + int(self.interval.upper))
        
        if start_time > end_time:
            return -float('inf')  # Invalid interval
        
        for t in range(start_time, end_time + 1):
            robustness = self.formula.evaluate_robustness(trajectory, t)
            max_robustness = max(max_robustness, robustness)
        
        return max_robustness
    
    def __str__(self) -> str:
        return f"F{self.interval}({self.formula})"

class Until(STLFormula):
    """Until temporal operator."""
    
    def __init__(self, left: STLFormula, right: STLFormula, 
                 interval: Optional[TimeInterval] = None):
        self.left = left
        self.right = right
        self.interval = interval or TimeInterval(0, float('inf'))
    
    def evaluate_robustness(self, trajectory: List[Dict[str, Any]], time: int) -> float:
        """Evaluate until robustness."""
        max_robustness = -float('inf')
        
        start_time = max(0, time + int(self.interval.lower))
        end_time = min(len(trajectory) - 1, time + int(self.interval.upper))
        
        for t2 in range(start_time, end_time + 1):
            # Right formula must be true at t2
            right_rob = self.right.evaluate_robustness(trajectory, t2)
            
            # Left formula must be true from time to t2-1
            min_left_rob = float('inf')
            for t1 in range(time, t2):
                if t1 < len(trajectory):
                    left_rob = self.left.evaluate_robustness(trajectory, t1)
                    min_left_rob = min(min_left_rob, left_rob)
                else:
                    min_left_rob = -float('inf')
                    break
            
            # Until robustness is min of both conditions
            until_rob = min(right_rob, min_left_rob)
            max_robustness = max(max_robustness, until_rob)
        
        return max_robustness
    
    def __str__(self) -> str:
        return f"({self.left} U{self.interval} {self.right})"

class QuadcopterSTLPredicates:
    """Predefined STL predicates for quadcopter missions."""
    
    def __init__(self, waypoints: Dict[str, np.ndarray], tolerance: float = 0.3):
        self.waypoints = waypoints
        self.tolerance = tolerance
    
    def at_waypoint(self, waypoint_name: str) -> Callable[[Dict[str, Any]], float]:
        """Predicate: drone is at specified waypoint."""
        def predicate(state_dict: Dict[str, Any]) -> float:
            position = state_dict['position']
            target = self.waypoints[waypoint_name]
            distance = np.linalg.norm(position - target)
            # Robustness: positive when inside tolerance, negative outside
            return self.tolerance - distance
        
        return predicate
    
    def safe_altitude(self, min_alt: float = 0.2, max_alt: float = 8.0) -> Callable[[Dict[str, Any]], float]:
        """Predicate: drone maintains safe altitude."""
        def predicate(state_dict: Dict[str, Any]) -> float:
            altitude = state_dict['position'][2]
            # Distance from safe altitude bounds
            if altitude < min_alt:
                return altitude - min_alt  # Negative (unsafe)
            elif altitude > max_alt:
                return max_alt - altitude  # Negative (unsafe)
            else:
                return min(altitude - min_alt, max_alt - altitude)  # Positive (safe)
        
        return predicate
    
    def stable_orientation(self, max_tilt: float = math.pi/4) -> Callable[[Dict[str, Any]], float]:
        """Predicate: drone maintains stable orientation."""
        def predicate(state_dict: Dict[str, Any]) -> float:
            orientation = state_dict['orientation']
            roll, pitch = orientation[0], orientation[1]
            max_tilt_current = max(abs(roll), abs(pitch))
            return max_tilt - max_tilt_current
        
        return predicate
    
    def reasonable_velocity(self, max_vel: float = 5.0) -> Callable[[Dict[str, Any]], float]:
        """Predicate: drone maintains reasonable velocity."""
        def predicate(state_dict: Dict[str, Any]) -> float:
            velocity = state_dict['linear_velocity']
            speed = np.linalg.norm(velocity)
            return max_vel - speed
        
        return predicate

class MissionSTLSpecs:
    """Predefined STL specifications for common quadcopter missions."""
    
    def __init__(self, waypoints: Dict[str, np.ndarray], tolerance: float = 0.3):
        self.predicates = QuadcopterSTLPredicates(waypoints, tolerance)
        self.waypoints = waypoints
    
    def create_surveillance_mission_spec(self) -> STLFormula:
        """
        Create STL specification for A→B→Charging→A surveillance mission.
        Formula: G(F[0,50](at_B) ∧ F[0,100](at_charging) ∧ F[0,150](at_A))
        "Always, eventually visit B within 50 steps, then charging within 100, then A within 150"
        """
        
        # Atomic predicates
        at_A = AtomicPredicate("at_A", self.predicates.at_waypoint('A'))
        at_B = AtomicPredicate("at_B", self.predicates.at_waypoint('B'))
        at_charging = AtomicPredicate("at_charging", self.predicates.at_waypoint('charging'))
        
        # Safety predicates
        safe_alt = AtomicPredicate("safe_altitude", self.predicates.safe_altitude())
        stable_orient = AtomicPredicate("stable_orientation", self.predicates.stable_orientation())
        reasonable_vel = AtomicPredicate("reasonable_velocity", self.predicates.reasonable_velocity())
        
        # Mission sequence: Eventually visit B, then charging, then A
        visit_B = Eventually(at_B, TimeInterval(0, 50))
        visit_charging = Eventually(at_charging, TimeInterval(0, 100))
        visit_A = Eventually(at_A, TimeInterval(0, 150))
        
        # Sequential mission
        mission_sequence = Conjunction(
            visit_B,
            Conjunction(visit_charging, visit_A)
        )
        
        # Safety constraints (always maintain)
        safety_constraints = Conjunction(
            safe_alt,
            Conjunction(stable_orient, reasonable_vel)
        )
        
        # Combined specification: Always maintain safety AND accomplish mission
        full_spec = Always(
            Conjunction(safety_constraints, mission_sequence),
            TimeInterval(0, 200)  # Over entire episode
        )
        
        return full_spec
    
    def create_simple_navigation_spec(self, target: str) -> STLFormula:
        """Create simple navigation specification to a single target."""
        
        at_target = AtomicPredicate(f"at_{target}", self.predicates.at_waypoint(target))
        safe_alt = AtomicPredicate("safe_altitude", self.predicates.safe_altitude())
        
        # Eventually reach target while maintaining safety
        reach_target = Eventually(at_target, TimeInterval(0, 100))
        maintain_safety = Always(safe_alt, TimeInterval(0, 100))
        
        return Conjunction(reach_target, maintain_safety)
    
    def create_periodic_patrol_spec(self, period: int = 100) -> STLFormula:
        """Create specification for periodic patrol between waypoints."""
        
        at_A = AtomicPredicate("at_A", self.predicates.at_waypoint('A'))
        at_B = AtomicPredicate("at_B", self.predicates.at_waypoint('B'))
        
        # Visit A and B periodically
        periodic_A = Always(Eventually(at_A, TimeInterval(0, period)), TimeInterval(0, float('inf')))
        periodic_B = Always(Eventually(at_B, TimeInterval(0, period)), TimeInterval(0, float('inf')))
        
        return Conjunction(periodic_A, periodic_B)

class STLRobustnessCalculator:
    """Calculator for STL robustness scores with optimization for RL rewards."""
    
    def __init__(self, formula: STLFormula):
        self.formula = formula
        self.robustness_cache = {}  # Cache for performance
    
    def compute_trajectory_robustness(self, trajectory: List[Dict[str, Any]]) -> List[float]:
        """Compute robustness score for each time step in trajectory."""
        robustness_scores = []
        
        for t in range(len(trajectory)):
            robustness = self.formula.evaluate_robustness(trajectory, t)
            robustness_scores.append(robustness)
        
        return robustness_scores
    
    def compute_overall_robustness(self, trajectory: List[Dict[str, Any]]) -> float:
        """Compute overall robustness score for entire trajectory."""
        if not trajectory:
            return -float('inf')
        
        # Use robustness at time 0 (which considers the entire future)
        return self.formula.evaluate_robustness(trajectory, 0)
    
    def compute_windowed_robustness(self, trajectory: List[Dict[str, Any]], 
                                   window_size: int = 10) -> List[float]:
        """Compute robustness over sliding windows for online evaluation."""
        windowed_scores = []
        
        for t in range(len(trajectory)):
            window_start = max(0, t - window_size + 1)
            window_trajectory = trajectory[window_start:t + 1]
            
            if window_trajectory:
                # Evaluate formula at the end of the window
                robustness = self.formula.evaluate_robustness(window_trajectory, len(window_trajectory) - 1)
                windowed_scores.append(robustness)
            else:
                windowed_scores.append(-float('inf'))
        
        return windowed_scores
    
    def normalize_robustness(self, robustness: float, min_val: float = -10.0, max_val: float = 10.0) -> float:
        """Normalize robustness score to [0, 1] range for RL rewards."""
        # Clip to bounds
        clipped = max(min_val, min(max_val, robustness))
        
        # Normalize to [0, 1]
        normalized = (clipped - min_val) / (max_val - min_val)
        
        return normalized
    
    def get_satisfaction_probability(self, robustness: float) -> float:
        """Convert robustness to satisfaction probability using sigmoid."""
        return 1.0 / (1.0 + np.exp(-robustness))

class STLParser:
    """Parser for STL formulas from string representations."""
    
    def __init__(self, predicates: Dict[str, Callable[[Dict[str, Any]], float]]):
        self.predicates = predicates
        self.operators = {
            'G': self._parse_always,
            'F': self._parse_eventually,
            'U': self._parse_until,
            '&': self._parse_conjunction,
            '|': self._parse_disjunction,
            '!': self._parse_negation,
            '->': self._parse_implication
        }
    
    def parse(self, formula_string: str) -> STLFormula:
        """Parse STL formula from string."""
        # This is a simplified parser - in practice, you'd want a more robust implementation
        tokens = self._tokenize(formula_string)
        return self._parse_expression(tokens)
    
    def _tokenize(self, formula_string: str) -> List[str]:
        """Tokenize formula string."""
        # Simple tokenization - would need enhancement for complex formulas
        tokens = []
        i = 0
        while i < len(formula_string):
            if formula_string[i].isspace():
                i += 1
                continue
            
            # Check for multi-character operators
            if i < len(formula_string) - 1:
                two_char = formula_string[i:i+2]
                if two_char in ['G[', 'F[', '->', '&&', '||']:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Single character tokens
            tokens.append(formula_string[i])
            i += 1
        
        return tokens
    
    def _parse_expression(self, tokens: List[str]) -> STLFormula:
        """Parse expression from tokens."""
        # Simplified parsing - would need a proper parser for complex formulas
        if not tokens:
            raise ValueError("Empty formula")
        
        # Handle atomic predicates
        if len(tokens) == 1 and tokens[0] in self.predicates:
            return AtomicPredicate(tokens[0], self.predicates[tokens[0]])
        
        # This is a placeholder - a full implementation would need proper precedence handling
        raise NotImplementedError("Complex formula parsing not fully implemented")
    
    def _parse_always(self, tokens: List[str], index: int) -> Tuple[STLFormula, int]:
        """Parse Always operator."""
        # Implementation would extract interval and subformula
        pass
    
    def _parse_eventually(self, tokens: List[str], index: int) -> Tuple[STLFormula, int]:
        """Parse Eventually operator."""
        pass
    
    def _parse_until(self, tokens: List[str], index: int) -> Tuple[STLFormula, int]:
        """Parse Until operator."""
        pass
    
    def _parse_conjunction(self, tokens: List[str], index: int) -> Tuple[STLFormula, int]:
        """Parse Conjunction operator."""
        pass
    
    def _parse_disjunction(self, tokens: List[str], index: int) -> Tuple[STLFormula, int]:
        """Parse Disjunction operator."""
        pass
    
    def _parse_negation(self, tokens: List[str], index: int) -> Tuple[STLFormula, int]:
        """Parse Negation operator."""
        pass
    
    def _parse_implication(self, tokens: List[str], index: int) -> Tuple[STLFormula, int]:
        """Parse Implication operator."""
        pass

class AdaptiveSTLReward:
    """Adaptive STL reward calculator that adjusts weights based on learning progress."""
    
    def __init__(self, formula: STLFormula, initial_weight: float = 1.0):
        self.formula = formula
        self.robustness_calculator = STLRobustnessCalculator(formula)
        self.weight = initial_weight
        
        # Adaptive parameters
        self.success_history = []
        self.difficulty_level = 1.0
        self.adaptation_rate = 0.01
        
        # Performance tracking
        self.robustness_history = []
        self.reward_history = []
    
    def compute_adaptive_reward(self, trajectory: List[Dict[str, Any]], 
                               episode: int, success_rate: float = 0.0) -> float:
        """Compute adaptive STL reward based on learning progress."""
        
        # Basic robustness score
        robustness = self.robustness_calculator.compute_overall_robustness(trajectory)
        
        # Adapt weight based on success rate
        self._adapt_weight(success_rate, episode)
        
        # Scale robustness by current weight
        adaptive_reward = robustness * self.weight
        
        # Add shaping based on difficulty
        shaped_reward = self._apply_difficulty_shaping(adaptive_reward, trajectory)
        
        # Track performance
        self.robustness_history.append(robustness)
        self.reward_history.append(shaped_reward)
        
        return shaped_reward
    
    def _adapt_weight(self, success_rate: float, episode: int):
        """Adapt STL weight based on learning progress."""
        
        # Increase weight if success rate is low (need more STL guidance)
        if success_rate < 0.3:
            self.weight = min(3.0, self.weight + self.adaptation_rate)
        # Decrease weight if success rate is high (learned the task)
        elif success_rate > 0.7:
            self.weight = max(0.5, self.weight - self.adaptation_rate)
        
        # Adjust difficulty based on performance
        if episode > 100 and episode % 100 == 0:
            recent_success = np.mean(self.success_history[-100:]) if len(self.success_history) >= 100 else success_rate
            if recent_success > 0.8:
                self.difficulty_level = min(2.0, self.difficulty_level + 0.1)
            elif recent_success < 0.3:
                self.difficulty_level = max(0.5, self.difficulty_level - 0.1)
    
    def _apply_difficulty_shaping(self, reward: float, trajectory: List[Dict[str, Any]]) -> float:
        """Apply difficulty-based reward shaping."""
        
        # Add bonus for completing tasks quickly (higher difficulty)
        if self.difficulty_level > 1.0 and reward > 0:
            time_bonus = max(0, (200 - len(trajectory)) / 200) * self.difficulty_level
            reward += time_bonus
        
        # Add penalty for inefficient paths (higher difficulty)
        if self.difficulty_level > 1.5 and len(trajectory) > 2:
            path_efficiency = self._compute_path_efficiency(trajectory)
            if path_efficiency < 0.5:
                reward -= (1 - path_efficiency) * self.difficulty_level
        
        return reward
    
    def _compute_path_efficiency(self, trajectory: List[Dict[str, Any]]) -> float:
        """Compute path efficiency (straight-line distance vs actual path)."""
        if len(trajectory) < 2:
            return 1.0
        
        start_pos = trajectory[0]['position']
        end_pos = trajectory[-1]['position']
        straight_line_distance = np.linalg.norm(end_pos - start_pos)
        
        # Compute actual path length
        actual_distance = 0
        for i in range(1, len(trajectory)):
            actual_distance += np.linalg.norm(
                trajectory[i]['position'] - trajectory[i-1]['position']
            )
        
        if actual_distance == 0:
            return 1.0
        
        efficiency = straight_line_distance / actual_distance
        return min(1.0, efficiency)

class STLMonitor:
    """Online STL monitor for real-time robustness evaluation."""
    
    def __init__(self, formula: STLFormula, window_size: int = 50):
        self.formula = formula
        self.window_size = window_size
        self.trajectory_buffer = []
        self.robustness_calculator = STLRobustnessCalculator(formula)
        
        # Monitoring statistics
        self.violation_count = 0
        self.satisfaction_count = 0
        self.robustness_trend = []
    
    def update(self, state_dict: Dict[str, Any]) -> Tuple[float, bool]:
        """Update monitor with new state and return current robustness."""
        
        # Add to trajectory buffer
        self.trajectory_buffer.append(state_dict)
        
        # Maintain window size
        if len(self.trajectory_buffer) > self.window_size:
            self.trajectory_buffer.pop(0)
        
        # Compute current robustness
        current_robustness = self.formula.evaluate_robustness(self.trajectory_buffer, len(self.trajectory_buffer) - 1)
        
        # Update statistics
        is_satisfied = current_robustness > 0
        if is_satisfied:
            self.satisfaction_count += 1
        else:
            self.violation_count += 1
        
        self.robustness_trend.append(current_robustness)
        
        # Keep trend history bounded
        if len(self.robustness_trend) > 1000:
            self.robustness_trend.pop(0)
        
        return current_robustness, is_satisfied
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        total_updates = self.satisfaction_count + self.violation_count
        
        return {
            'total_updates': total_updates,
            'satisfaction_count': self.satisfaction_count,
            'violation_count': self.violation_count,
            'satisfaction_rate': self.satisfaction_count / max(1, total_updates),
            'current_robustness': self.robustness_trend[-1] if self.robustness_trend else 0,
            'avg_robustness': np.mean(self.robustness_trend) if self.robustness_trend else 0,
            'robustness_std': np.std(self.robustness_trend) if self.robustness_trend else 0,
            'min_robustness': np.min(self.robustness_trend) if self.robustness_trend else 0,
            'max_robustness': np.max(self.robustness_trend) if self.robustness_trend else 0
        }
    
    def reset(self):
        """Reset monitor state."""
        self.trajectory_buffer = []
        self.violation_count = 0
        self.satisfaction_count = 0
        self.robustness_trend = []

# Utility functions for STL integration

def create_quadcopter_mission_formula(waypoints: Dict[str, np.ndarray], 
                                     mission_type: str = "surveillance",
                                     tolerance: float = 0.3) -> STLFormula:
    """Create STL formula for common quadcopter missions."""
    
    mission_specs = MissionSTLSpecs(waypoints, tolerance)
    
    if mission_type == "surveillance":
        return mission_specs.create_surveillance_mission_spec()
    elif mission_type == "navigation":
        return mission_specs.create_simple_navigation_spec('B')
    elif mission_type == "patrol":
        return mission_specs.create_periodic_patrol_spec()
    else:
        raise ValueError(f"Unknown mission type: {mission_type}")

def trajectory_to_state_dicts(trajectory_data: List[np.ndarray], 
                             waypoints: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Convert raw trajectory data to state dictionaries for STL evaluation."""
    
    state_dicts = []
    for state in trajectory_data:
        if len(state) >= 12:  # Standard quadcopter state
            state_dict = {
                'position': state[:3],
                'orientation': state[3:6],
                'linear_velocity': state[6:9],
                'angular_velocity': state[9:12],
                'time': len(state_dicts)
            }
            state_dicts.append(state_dict)
    
    return state_dicts

def evaluate_mission_success(trajectory: List[Dict[str, Any]], 
                           waypoints: Dict[str, np.ndarray],
                           tolerance: float = 0.3) -> Dict[str, Any]:
    """Evaluate mission success using STL robustness."""
    
    # Create surveillance mission formula
    formula = create_quadcopter_mission_formula(waypoints, "surveillance", tolerance)
    calculator = STLRobustnessCalculator(formula)
    
    # Compute robustness
    overall_robustness = calculator.compute_overall_robustness(trajectory)
    trajectory_robustness = calculator.compute_trajectory_robustness(trajectory)
    
    # Determine success
    is_successful = overall_robustness > 0
    satisfaction_probability = calculator.get_satisfaction_probability(overall_robustness)
    
    return {
        'success': is_successful,
        'overall_robustness': overall_robustness,
        'satisfaction_probability': satisfaction_probability,
        'trajectory_robustness': trajectory_robustness,
        'min_robustness': np.min(trajectory_robustness),
        'max_robustness': np.max(trajectory_robustness),
        'avg_robustness': np.mean(trajectory_robustness)
    }
            