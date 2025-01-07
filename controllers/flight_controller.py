# controllers/flight_controller.py

"""
Main flight control system for paraglider navigation.
Integrates state estimation, navigation, and safety systems
to generate and validate control commands.

The control hierarchy consists of:
1. Safety monitoring and emergency response
2. Navigation and path planning
3. State estimation and sensor fusion
4. Control output processing and validation
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from config.flight_config import control, safety, sensors
from config.vehicle_config import vehicle as vehicle_config
from config.general_config import sim
from utils.time_manager import time_manager
from .navigation import NavigationController
from .state_estimator import StateEstimator

class FlightController:
    """
    Master control system coordinating all flight subsystems.
    """
    
    def __init__(self):
        """Initialize flight control system and all subsystems."""
        # Initialize critical subsystems
        self.state_estimator = StateEstimator()
        self.navigation = NavigationController()
        
        # Control state
        self.left_brake = 0.0
        self.right_brake = 0.0
        self.control_mode = "normal"  # normal, safety, emergency
        self.previous_mode = "normal"
        self.mode_entry_time = time_manager.get_time()
        
        # Initialize safety monitoring system
        self._init_safety_system()
        
        # Initialize performance tracking
        self._init_performance_tracking()
        
        # Timing and update management
        self._init_timing()
        
        # Debug and logging flags
        self.debug_mode = sim.DEBUG_MODE
    
    def _init_safety_system(self):
        """Initialize safety monitoring system."""
        # Safety condition monitors
        self.safety_triggers = {
            'low_altitude': False,
            'high_speed': False,
            'excessive_descent': False,
            'excessive_bank': False,
            'control_saturation': False,
            'navigation_error': False,
            'sensor_failure': False
        }
        
        # Safety thresholds
        self.safety_thresholds = {
            'min_altitude': safety.MIN_ALTITUDE,
            'max_speed': safety.MAX_SPEED,
            'max_descent_rate': control.EMERGENCY_DESCENT_RATE,
            'max_bank_angle': safety.MAX_BANK_ANGLE,
            'max_control_error': 0.8,
            'sensor_timeout': 1.0  # seconds
        }
        
        # Safety system state
        self.safety_state = {
            'warning_active': False,
            'emergency_active': False,
            'last_warning_time': 0.0,
            'cumulative_warnings': 0,
            'recovery_attempts': 0
        }
    
    def _init_performance_tracking(self):
        """Initialize performance tracking systems."""
        # Control performance metrics
        self.performance_metrics = {
            'control_outputs': [],
            'safety_interventions': 0,
            'mode_transitions': 0,
            'control_saturations': 0,
            'navigation_errors': [],
            'response_times': []
        }
        
        # Historical data tracking
        self.control_history: List[Dict] = []
        self.safety_history: List[Dict] = []
        self.history_length = 100  # Number of states to track
        
        # Statistical tracking
        self.statistics = {
            'average_response_time': 0.0,
            'max_control_deviation': 0.0,
            'stability_metric': 1.0,
            'safety_score': 1.0
        }
    
    def _init_timing(self):
        """Initialize timing and update management."""
        current_time = time_manager.get_time()
        
        # Update timing
        self.last_update = current_time
        self.last_safety_check = current_time
        self.last_metrics_update = current_time
        
        # Update intervals
        self.safety_check_interval = 0.1  # 10 Hz safety checks
        self.metrics_update_interval = 1.0  # 1 Hz metrics updates
        self.control_update_interval = 1.0 / control.CONTROL_UPDATE_RATE
        
        # Timing tracking
        self.update_times: List[float] = []
        self.computation_times: List[float] = []
    
    def update(self, vehicle, environment_conditions: Dict) -> Tuple[float, float]:
        """
        Update flight control system and generate control outputs.
        
        Args:
            vehicle: Current vehicle state
            environment_conditions: Current environmental conditions
            
        Returns:
            tuple: (left_brake, right_brake) control inputs
        """
        current_time = time_manager.get_time()
        dt = current_time - self.last_update
        
        try:
            # Start performance timing
            update_start_time = current_time
            
            # Update state estimation
            self._update_state_estimation(vehicle, dt)
            
            # Perform safety checks if interval elapsed
            if current_time - self.last_safety_check >= self.safety_check_interval:
                self._check_safety_conditions(vehicle)
                self.last_safety_check = current_time
            
            # Generate control outputs based on current mode
            left_brake, right_brake = self._generate_mode_controls(vehicle)
            
            # Process and validate control outputs
            self.left_brake, self.right_brake = self._process_control_outputs(
                left_brake, right_brake, vehicle
            )
            
            # Update performance metrics
            self._update_performance_metrics(current_time)
            
            # Track computation time
            self.computation_times.append(time_manager.get_time() - update_start_time)
            if len(self.computation_times) > self.history_length:
                self.computation_times.pop(0)
            
            self.last_update = current_time
            return self.left_brake, self.right_brake
            
        except Exception as e:
            print(f"Flight controller error: {str(e)}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return self._safe_default_controls()
    
    def _update_state_estimation(self, vehicle, dt: float):
        """
        Update state estimation with current sensor data.
        
        Args:
            vehicle: Vehicle object with current state
            dt: Time step in seconds
        """
        # Update state estimation with robust error handling
        try:
            # Predict step
            self.state_estimator.predict(dt)
            
            # GPS update if available
            if hasattr(vehicle, 'gps_position') and hasattr(vehicle, 'gps_velocity'):
                self.state_estimator.update_gps(
                    vehicle.gps_position,
                    vehicle.gps_velocity
                )
            
            # IMU update if available
            if hasattr(vehicle, 'acceleration') and hasattr(vehicle, 'angular_velocity'):
                self.state_estimator.update_imu(
                    vehicle.acceleration,
                    vehicle.angular_velocity
                )
            
            # Barometer update if available
            if hasattr(vehicle, 'barometric_altitude'):
                self.state_estimator.update_baro(vehicle.barometric_altitude)
            
            # Check sensor health
            self._check_sensor_health()
            
        except Exception as e:
            print(f"State estimation error: {str(e)}")
            self.safety_triggers['sensor_failure'] = True
    
    def _check_sensor_health(self):
        """Check health and validity of sensor updates."""
        estimator_state = self.state_estimator.get_state()
        
        # Check measurement validity flags
        sensor_valid = all([
            estimator_state['measurement_valid']['gps'],
            estimator_state['measurement_valid']['imu'],
            estimator_state['measurement_valid']['baro']
        ])
        
        # Update sensor failure trigger
        self.safety_triggers['sensor_failure'] = not sensor_valid
        
        if not sensor_valid and self.debug_mode:
            print("Warning: Sensor failure detected")
    
    def _check_safety_conditions(self, vehicle):
        """
        Perform comprehensive safety checks.
        
        Args:
            vehicle: Current vehicle state
        """
        # Store previous trigger state
        previous_triggers = self.safety_triggers.copy()
        
        # Reset triggers
        self.safety_triggers = {key: False for key in self.safety_triggers}
        
        try:
            # Altitude check
            altitude = -vehicle.position[2]
            self.safety_triggers['low_altitude'] = altitude < self.safety_thresholds['min_altitude']
            
            # Speed check
            airspeed = np.linalg.norm(vehicle.velocity_air)
            self.safety_triggers['high_speed'] = airspeed > self.safety_thresholds['max_speed']
            
            # Descent rate check
            descent_rate = -vehicle.velocity[2]
            self.safety_triggers['excessive_descent'] = (
                descent_rate > self.safety_thresholds['max_descent_rate']
            )
            
            # Bank angle check
            bank_angle = self._calculate_bank_angle(vehicle.orientation)
            self.safety_triggers['excessive_bank'] = (
                abs(bank_angle) > self.safety_thresholds['max_bank_angle']
            )
            
            # Control saturation check
            self.safety_triggers['control_saturation'] = (
                max(self.left_brake, self.right_brake) > 0.9 or
                min(self.left_brake, self.right_brake) < 0.1
            )
            
            # Navigation error check
            nav_state = self.navigation.get_state()
            self.safety_triggers['navigation_error'] = (
                nav_state['errors']['cross_track'] > self.safety_thresholds['max_control_error']
            )
            
            # Update control mode based on triggers
            self._update_control_mode(previous_triggers)
            
            # Log safety state if in debug mode
            if self.debug_mode and any(self.safety_triggers.values()):
                self._log_safety_state()
                
        except Exception as e:
            print(f"Safety check error: {str(e)}")
            self.control_mode = "emergency"  # Default to emergency mode on error
    
    def _calculate_bank_angle(self, orientation: np.ndarray) -> float:
        """
        Calculate bank angle from orientation matrix.
        
        Args:
            orientation: Vehicle orientation matrix
            
        Returns:
            float: Bank angle in radians
        """
        return np.arctan2(
            np.sqrt(orientation[0,2]**2 + orientation[1,2]**2),
            orientation[2,2]
        )
    
    def _update_control_mode(self, previous_triggers: Dict):
        """
        Update control mode based on safety triggers.
        
        Args:
            previous_triggers: Previous safety trigger states
        """
        previous_mode = self.control_mode
        
        # Check for emergency conditions
        emergency_conditions = [
            self.safety_triggers['low_altitude'],
            self.safety_triggers['high_speed'],
            self.safety_triggers['excessive_descent'],
            self.safety_triggers['sensor_failure']
        ]
        
        # Check for safety warning conditions
        safety_conditions = [
            self.safety_triggers['excessive_bank'],
            self.safety_triggers['control_saturation'],
            self.safety_triggers['navigation_error']
        ]
        
        # Determine new mode
        if any(emergency_conditions):
            new_mode = "emergency"
        elif any(safety_conditions):
            new_mode = "safety"
        else:
            new_mode = "normal"
        
        # Handle mode transition
        if new_mode != self.control_mode:
            self._handle_mode_transition(new_mode)
    
    def _handle_mode_transition(self, new_mode: str):
        """
        Handle control mode transition.
        
        Args:
            new_mode: New control mode to transition to
        """
        # Store previous mode data
        self.previous_mode = self.control_mode
        mode_duration = time_manager.get_time() - self.mode_entry_time
        
        # Update mode
        self.control_mode = new_mode
        self.mode_entry_time = time_manager.get_time()
        
        # Update metrics
        self.performance_metrics['mode_transitions'] += 1
        
        # Log transition if in debug mode
        if self.debug_mode:
            print(f"Control mode transition: {self.previous_mode} -> {new_mode}")
            print(f"Previous mode duration: {mode_duration:.1f}s")
            print(f"Active triggers: {[k for k, v in self.safety_triggers.items() if v]}")
    
    def _generate_mode_controls(self, vehicle) -> Tuple[float, float]:
        """
        Generate control outputs based on current mode.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            tuple: (left_brake, right_brake) control inputs
        """
        if self.control_mode == "emergency":
            return self._emergency_controls(vehicle)
        elif self.control_mode == "safety":
            return self._safety_controls(vehicle)
        else:
            return self._normal_controls(vehicle)
    
    def _emergency_controls(self, vehicle) -> Tuple[float, float]:
        """
        Generate emergency control inputs.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            tuple: Emergency control inputs
        """
        # Initialize with safe defaults
        left_brake = right_brake = 0.4
        
        if self.safety_triggers['high_speed']:
            # Strong symmetric braking to reduce speed
            brake_value = min(0.8, vehicle_config.MAX_BRAKE_DEFLECTION)
            left_brake = right_brake = brake_value
            
        elif self.safety_triggers['excessive_descent']:
            # Maximum safe braking to arrest descent
            brake_value = min(0.9, vehicle_config.MAX_BRAKE_DEFLECTION)
            left_brake = right_brake = brake_value
            
        elif self.safety_triggers['low_altitude']:
            # Prepare for emergency landing
            left_brake = right_brake = control.FLARE_BRAKE_VALUE
            
        elif self.safety_triggers['sensor_failure']:
            # Stable configuration with moderate braking
            left_brake = right_brake = 0.5
        
        return left_brake, right_brake
    
    def _safety_controls(self, vehicle) -> Tuple[float, float]:
        """
        Generate safety-oriented control inputs.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            tuple: Safety-oriented control inputs
        """
        # Get base control inputs from navigation
        left_brake, right_brake = self.navigation.update(vehicle)
        
        # Modify controls based on safety conditions
        if self.safety_triggers['excessive_bank']:
            # Level controls to reduce bank angle
            base_brake = 0.4
            left_brake = right_brake = base_brake
            
        elif self.safety_triggers['control_saturation']:
            # Limit control range
            left_brake = np.clip(left_brake, 0.2, 0.8)
            right_brake = np.clip(right_brake, 0.2, 0.8)
            
        elif self.safety_triggers['navigation_error']:
            # Add correction bias
            nav_state = self.navigation.get_state()
            correction = 0.1 * np.sign(nav_state['errors']['cross_track'])
            left_brake += correction
            right_brake -= correction
        
        return left_brake, right_brake
    
    def _normal_controls(self, vehicle) -> Tuple[float, float]:
        """
        Generate normal flight control inputs using navigation system.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            tuple: Normal control inputs
        """
        # Get base navigation controls
        left_brake, right_brake = self.navigation.update(vehicle)
        
        # Apply control smoothing
        left_brake = self._smooth_control(left_brake, self.left_brake)
        right_brake = self._smooth_control(right_brake, self.right_brake)
        
        # Apply performance optimizations based on flight conditions
        left_brake, right_brake = self._optimize_controls(
            left_brake,
            right_brake,
            vehicle
        )
        
        return left_brake, right_brake
    
    def _smooth_control(self, target: float, current: float) -> float:
        """
        Apply smoothing to control transitions.
        
        Args:
            target: Target control value
            current: Current control value
            
        Returns:
            float: Smoothed control value
        """
        # Calculate smoothing factor based on control error
        error = abs(target - current)
        alpha = min(0.3, error)  # Adaptive smoothing
        
        # Apply exponential smoothing
        return current + alpha * (target - current)
    
    def _optimize_controls(self, left_brake: float, right_brake: float,
                         vehicle) -> Tuple[float, float]:
        """
        Optimize control inputs for current flight conditions.
        
        Args:
            left_brake: Base left brake input
            right_brake: Base right brake input
            vehicle: Current vehicle state
            
        Returns:
            tuple: Optimized control inputs
        """
        # Get current flight parameters
        airspeed = np.linalg.norm(vehicle.velocity_air)
        altitude = -vehicle.position[2]
        
        # Optimize for airspeed
        if airspeed < vehicle_config.BEST_GLIDE_SPEED:
            # Reduce braking to increase speed
            brake_reduction = 0.1 * (1 - airspeed/vehicle_config.BEST_GLIDE_SPEED)
            left_brake = max(0.1, left_brake - brake_reduction)
            right_brake = max(0.1, right_brake - brake_reduction)
            
        elif airspeed > vehicle_config.BEST_GLIDE_SPEED:
            # Increase braking to reduce speed
            brake_increase = 0.1 * (airspeed/vehicle_config.BEST_GLIDE_SPEED - 1)
            left_brake = min(0.8, left_brake + brake_increase)
            right_brake = min(0.8, right_brake + brake_increase)
        
        # Adjust for altitude
        if altitude < safety.LOW_ALTITUDE_WARNING * 1.5:
            # More conservative controls at low altitude
            left_brake = np.clip(left_brake, 0.2, 0.7)
            right_brake = np.clip(right_brake, 0.2, 0.7)
        
        return left_brake, right_brake
    
    def _process_control_outputs(self, left_brake: float, right_brake: float,
                               vehicle) -> Tuple[float, float]:
        """
        Process and validate final control outputs.
        
        Args:
            left_brake: Raw left brake input
            right_brake: Raw right brake input
            vehicle: Current vehicle state
            
        Returns:
            tuple: Processed and validated control inputs
        """
        # Apply safety bounds
        left_brake = np.clip(
            left_brake,
            0.0,
            vehicle_config.MAX_BRAKE_DEFLECTION
        )
        right_brake = np.clip(
            right_brake,
            0.0,
            vehicle_config.MAX_BRAKE_DEFLECTION
        )
        
        # Check for control saturation
        if (left_brake >= vehicle_config.MAX_BRAKE_DEFLECTION or
            right_brake >= vehicle_config.MAX_BRAKE_DEFLECTION):
            self.performance_metrics['control_saturations'] += 1
            if self.debug_mode:
                print("Warning: Control saturation detected")
        
        # Validate control differential
        max_differential = 0.5 * vehicle_config.MAX_BRAKE_DEFLECTION
        current_differential = abs(left_brake - right_brake)
        
        if current_differential > max_differential:
            # Reduce differential while maintaining average brake position
            average_brake = (left_brake + right_brake) / 2
            half_diff = max_differential / 2
            left_brake = average_brake + half_diff * np.sign(left_brake - right_brake)
            right_brake = average_brake - half_diff * np.sign(left_brake - right_brake)
        
        return left_brake, right_brake
    
    def _update_performance_metrics(self, current_time: float):
        """
        Update controller performance metrics.
        
        Args:
            current_time: Current simulation time
        """
        # Update control history
        self.control_history.append({
            'time': current_time,
            'left_brake': self.left_brake,
            'right_brake': self.right_brake,
            'mode': self.control_mode,
            'safety_triggers': self.safety_triggers.copy()
        })
        
        # Maintain history length
        if len(self.control_history) > self.history_length:
            self.control_history.pop(0)
        
        # Calculate response time
        if len(self.computation_times) > 0:
            avg_response = np.mean(self.computation_times)
            self.performance_metrics['response_times'].append(avg_response)
            self.statistics['average_response_time'] = avg_response
        
        # Calculate control stability
        if len(self.control_history) > 1:
            control_changes = []
            for i in range(1, len(self.control_history)):
                prev = self.control_history[i-1]
                curr = self.control_history[i]
                change = np.sqrt(
                    (curr['left_brake'] - prev['left_brake'])**2 +
                    (curr['right_brake'] - prev['right_brake'])**2
                )
                control_changes.append(change)
            
            self.statistics['stability_metric'] = 1.0 - min(1.0, np.mean(control_changes) * 5)
        
        # Update safety score
        self.statistics['safety_score'] = self._calculate_safety_score()
    
    def _calculate_safety_score(self) -> float:
        """
        Calculate overall safety score based on performance history.
        
        Returns:
            float: Safety score between 0 and 1
        """
        # Initialize base score
        score = 1.0
        
        # Penalize for active safety triggers
        active_triggers = sum(self.safety_triggers.values())
        score -= 0.1 * active_triggers
        
        # Penalize for recent mode transitions
        recent_transitions = sum(1 for state in self.control_history[-10:]
                               if state['mode'] != self.control_mode)
        score -= 0.05 * recent_transitions
        
        # Penalize for control saturations
        score -= 0.01 * self.performance_metrics['control_saturations']
        
        # Add bonus for stable control
        score += 0.1 * self.statistics['stability_metric']
        
        return np.clip(score, 0.0, 1.0)
    
    def _safe_default_controls(self) -> Tuple[float, float]:
        """
        Generate safe default control values for error cases.
        
        Returns:
            tuple: Safe default control inputs
        """
        return 0.3, 0.3  # Moderate symmetric braking
    
    def _log_safety_state(self):
        """Log current safety state for debugging."""
        print("\nSafety State Update:")
        print(f"Mode: {self.control_mode}")
        print("Active Triggers:", [k for k, v in self.safety_triggers.items() if v])
        print(f"Safety Score: {self.statistics['safety_score']:.2f}")
        print(f"Stability Metric: {self.statistics['stability_metric']:.2f}")
    
    def get_state(self) -> Dict:
        """
        Get comprehensive controller state.
        
        Returns:
            Dictionary containing current controller state
        """
        return {
            'control': {
                'left_brake': self.left_brake,
                'right_brake': self.right_brake,
                'mode': self.control_mode,
                'mode_duration': time_manager.get_time() - self.mode_entry_time
            },
            'safety': {
                'triggers': self.safety_triggers.copy(),
                'safety_score': self.statistics['safety_score'],
                'stability_metric': self.statistics['stability_metric']
            },
            'performance': {
                'control_saturations': self.performance_metrics['control_saturations'],
                'mode_transitions': self.performance_metrics['mode_transitions'],
                'average_response_time': self.statistics['average_response_time'],
                'computation_load': len(self.computation_times) > 0 and
                                  np.mean(self.computation_times)
            },
            'subsystems': {
                'navigation': self.navigation.get_state(),
                'state_estimation': self.state_estimator.get_state()
            },
            'history': {
                'control_history_length': len(self.control_history),
                'last_update': self.last_update
            }
        }