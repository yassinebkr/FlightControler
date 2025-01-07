# controllers/navigation.py

"""
Navigation controller for autonomous paraglider flight.
Implements sophisticated path planning and control strategies
for waypoint navigation and landing approaches.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from config.flight_config import control, WAYPOINTS
from config.general_config import safety
from utils.time_manager import time_manager

class NavigationController:
    """Advanced navigation control system for paraglider flight."""
    
    def __init__(self):
        """Initialize navigation controller with comprehensive state tracking."""
        # Navigation state
        self.current_waypoint_idx = 0
        self.waypoints = WAYPOINTS
        self.last_waypoint = None
        self.next_waypoint = self._get_next_waypoint()
        
        # Control state
        self.left_brake = 0.0
        self.right_brake = 0.0
        
        # Flight phase management
        self.phase = "cruise"  # cruise, approach, landing, flare
        self.last_update = 0.0
        self.phase_entry_time = 0.0
        self.waypoint_dwell_time = 0.0
        
        # Performance tracking
        self.navigation_metrics = {
            'cross_track_error': 0.0,
            'heading_error': 0.0,
            'altitude_error': 0.0,
            'waypoints_reached': 0,
            'phase_transitions': 0,
            'average_tracking_error': 0.0
        }
        
        # Control history for derivative calculations
        self.error_history = {
            'heading': [],
            'track': [],
            'altitude': [],
            'timestamps': []
        }
        self.history_length = 50
        
        # Wind estimation
        self.wind_estimate = np.zeros(3)
        self.wind_samples = []
        self.last_wind_update = 0.0
    
    def update(self, vehicle) -> Tuple[float, float]:
        """
        Update navigation controls based on vehicle state.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            tuple: (left_brake, right_brake) control inputs
        """
        current_time = time_manager.get_time()
        
        # Rate limiting for control updates
        dt = current_time - self.last_update
        if dt < 1.0 / control.CONTROL_UPDATE_RATE:
            return self.left_brake, self.right_brake
        
        try:
            # Update navigation state
            self._update_navigation_state(vehicle, current_time)
            
            # Update wind estimation
            if vehicle.airspeed > control.MIN_AIRSPEED_FOR_ESTIMATION:
                self._update_wind_estimate(vehicle, current_time)
            
            # Generate control inputs based on current phase
            self.left_brake, self.right_brake = self._generate_control_inputs(vehicle)
            
            # Check waypoint progression
            self._check_waypoint_progression(vehicle, current_time)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            self.last_update = current_time
            return self.left_brake, self.right_brake
            
        except Exception as e:
            print(f"Navigation controller error: {str(e)}")
            return 0.0, 0.0
    
    def _update_navigation_state(self, vehicle, current_time: float):
        """
        Update internal navigation state and flight phase.
        
        Args:
            vehicle: Current vehicle state
            current_time: Current simulation time
        """
        if not self.next_waypoint:
            return
        
        # Calculate navigation errors
        position_error = self.next_waypoint[0] - vehicle.position
        self.navigation_metrics['cross_track_error'] = self._calculate_cross_track_error(
            vehicle.position,
            self.last_waypoint[0] if self.last_waypoint else vehicle.position,
            self.next_waypoint[0]
        )
        
        # Calculate heading error with wind correction
        desired_track = self._calculate_desired_track(vehicle, position_error)
        current_heading = self._get_vehicle_heading(vehicle)
        
        # Normalize heading error to [-pi, pi]
        self.navigation_metrics['heading_error'] = np.remainder(
            desired_track - current_heading + np.pi,
            2 * np.pi
        ) - np.pi
        
        # Calculate altitude error
        target_altitude = self._calculate_target_altitude()
        self.navigation_metrics['altitude_error'] = target_altitude - (-vehicle.position[2])
        
        # Update flight phase based on conditions
        self._update_flight_phase(vehicle, current_time)
    
    def _calculate_cross_track_error(self, position: np.ndarray,
                                   path_start: np.ndarray,
                                   path_end: np.ndarray) -> float:
        """
        Calculate cross-track error from desired path.
        
        Args:
            position: Current position
            path_start: Path start point
            path_end: Path end point
            
        Returns:
            float: Cross-track error in meters
        """
        if np.all(path_start == path_end):
            return np.linalg.norm(position - path_start)
            
        path_vector = path_end - path_start
        path_length = np.linalg.norm(path_vector)
        
        if path_length < 1e-6:
            return 0.0
            
        path_direction = path_vector / path_length
        position_vector = position - path_start
        
        # Project position onto path
        projection = np.dot(position_vector, path_direction)
        projected_point = path_start + projection * path_direction
        
        # Calculate perpendicular distance
        return np.linalg.norm(position - projected_point)
    
    def _calculate_desired_track(self, vehicle, position_error: np.ndarray) -> float:
        """
        Calculate desired track angle with wind correction.
        
        Args:
            vehicle: Current vehicle state
            position_error: Position error vector
            
        Returns:
            float: Desired track angle in radians
        """
        # Calculate basic desired track
        desired_track = np.arctan2(position_error[1], position_error[0])
        
        # Apply wind correction if estimation available
        if np.linalg.norm(self.wind_estimate) > 0.1:
            wind_correction = self._calculate_wind_correction(
                vehicle.airspeed,
                self.wind_estimate,
                desired_track
            )
            desired_track += wind_correction
        
        return desired_track
    
    def _calculate_wind_correction(self, airspeed: float,
                                 wind: np.ndarray,
                                 desired_track: float) -> float:
        """
        Calculate wind correction angle.
        
        Args:
            airspeed: Current airspeed
            wind: Wind vector
            desired_track: Desired track angle
            
        Returns:
            float: Wind correction angle in radians
        """
        wind_speed = np.linalg.norm(wind)
        if wind_speed < 0.1 or airspeed < safety.MIN_SPEED:
            return 0.0
        
        wind_angle = np.arctan2(wind[1], wind[0])
        relative_wind_angle = wind_angle - desired_track
        
        # Calculate correction angle using wind triangle
        correction = np.arcsin(
            np.clip(wind_speed * np.sin(relative_wind_angle) / airspeed, -1, 1)
        )
        
        return correction
    
    def _update_flight_phase(self, vehicle, current_time: float):
        """
        Update flight phase based on conditions.
        
        Args:
            vehicle: Current vehicle state
            current_time: Current simulation time
        """
        previous_phase = self.phase
        waypoint_mode = self.next_waypoint[1]
        altitude = -vehicle.position[2]
        
        # Determine appropriate flight phase
        if waypoint_mode == "flare" and altitude < control.FLARE_HEIGHT:
            new_phase = "flare"
        elif waypoint_mode == "land" and altitude < control.LANDING_PATTERN_ALTITUDE:
            new_phase = "landing"
        else:
            new_phase = "cruise"
        
        # Handle phase transition
        if new_phase != self.phase:
            self.phase = new_phase
            self.phase_entry_time = current_time
            self.navigation_metrics['phase_transitions'] += 1
            print(f"Transitioning to {new_phase} phase at {altitude:.1f}m")
    
    def _generate_control_inputs(self, vehicle) -> Tuple[float, float]:
        """
        Generate control inputs based on current phase.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            tuple: (left_brake, right_brake) control inputs
        """
        if self.phase == "flare":
            return self._flare_controls()
        elif self.phase == "landing":
            return self._landing_controls(vehicle)
        else:
            return self._cruise_controls(vehicle)
    
    def _cruise_controls(self, vehicle) -> Tuple[float, float]:
        """
        Generate controls for cruise flight with improved tracking.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            tuple: (left_brake, right_brake) control inputs
        """
        # PID control for heading
        heading_p = control.HEADING_P_GAIN * self.navigation_metrics['heading_error']
        
        # Calculate derivative term using error history
        if len(self.error_history['heading']) > 1:
            dt = self.error_history['timestamps'][-1] - self.error_history['timestamps'][-2]
            if dt > 0:
                heading_d = control.HEADING_D_GAIN * (
                    self.navigation_metrics['heading_error'] -
                    self.error_history['heading'][-1]
                ) / dt
            else:
                heading_d = 0.0
        else:
            heading_d = 0.0
        
        # Combine control terms
        turn_command = np.clip(heading_p + heading_d, -1.0, 1.0)
        
        # Convert to brake commands with improved control mapping
        base_brake = 0.3  # Base brake value for better control
        if turn_command > 0:  # Right turn
            left_brake = base_brake
            right_brake = base_brake + turn_command * 0.7
        else:  # Left turn
            left_brake = base_brake - turn_command * 0.7
            right_brake = base_brake
        
        # Apply altitude correction
        altitude_correction = self._calculate_altitude_correction(vehicle)
        left_brake += altitude_correction
        right_brake += altitude_correction
        
        return (
            np.clip(left_brake, 0.0, 1.0),
            np.clip(right_brake, 0.0, 1.0)
        )
    
    def _landing_controls(self, vehicle) -> Tuple[float, float]:
        """
        Generate controls for landing approach with precise tracking.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            tuple: (left_brake, right_brake) control inputs
        """
        # Increase base brake value for better speed control
        base_brake = 0.4
        
        # Calculate turn command with reduced gains for stability
        turn_command = 0.7 * control.HEADING_P_GAIN * self.navigation_metrics['heading_error']
        turn_command = np.clip(turn_command, -0.4, 0.4)
        
        # Apply turn command
        if turn_command > 0:
            left_brake = base_brake
            right_brake = base_brake + turn_command
        else:
            left_brake = base_brake - turn_command
            right_brake = base_brake
        
        # Adjust speed based on glidepath
        if vehicle.airspeed > control.APPROACH_SPEED:
            speed_correction = 0.1
            left_brake += speed_correction
            right_brake += speed_correction
        
        return (
            np.clip(left_brake, 0.0, 1.0),
            np.clip(right_brake, 0.0, 1.0)
        )
    
    def _flare_controls(self) -> Tuple[float, float]:
        """
        Generate controls for landing flare.
        
        Returns:
            tuple: (left_brake, right_brake) control inputs
        """
        return control.FLARE_BRAKE_VALUE, control.FLARE_BRAKE_VALUE
    
    def _calculate_altitude_correction(self, vehicle) -> float:
        """
        Calculate altitude correction for brake inputs.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            float: Altitude correction value
        """
        altitude_error = self.navigation_metrics['altitude_error']
        correction = control.ALTITUDE_P_GAIN * altitude_error
        
        # Apply stronger correction if significantly below glidepath
        if altitude_error < -50:
            correction *= 1.5
        
        return np.clip(correction, -0.2, 0.2)
    
    def _check_waypoint_progression(self, vehicle, current_time: float):
        """
        Check and handle waypoint progression.
        
        Args:
            vehicle: Current vehicle state
            current_time: Current simulation time
        """
        if not self.next_waypoint:
            return
            
        # Calculate distance to waypoint
        distance = np.linalg.norm(self.next_waypoint[0] - vehicle.position)
        
        # Check if waypoint is reached
        if ((self.phase != "flare" and distance <= control.WAYPOINT_RADIUS) or
            (self.phase == "flare" and vehicle.ground_contact)):
            self._advance_waypoint()
            self.navigation_metrics['waypoints_reached'] += 1
            
        # Check for timeout
        elif current_time - self.phase_entry_time > control.MAX_WAYPOINT_PROGRESSION_TIME:
            print(f"Waypoint {self.current_waypoint_idx} timed out")
            self._advance_waypoint()
    
    def _advance_waypoint(self):
        """Advance to next waypoint with proper state updates."""
        self.last_waypoint = self.next_waypoint
        self.current_waypoint_idx += 1
        self.next_waypoint = self._get_next_waypoint()
        self.waypoint_dwell_time = 0.0
    
    def _get_next_waypoint(self) -> Optional[Tuple[np.ndarray, str]]:
        """
        Get next waypoint if available.
        
        Returns:
            Optional[Tuple]: (position, mode) or None if no more waypoints
        """
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None
    
    def _get_vehicle_heading(self, vehicle) -> float:
        """
        Get vehicle heading from orientation matrix.
        
        Args:
            vehicle: Current vehicle state
            
        Returns:
            float: Vehicle heading in radians
        """
        return np.arctan2(
            vehicle.orientation[1,0],
            vehicle.orientation[0,0]
        )
    
    def _update_wind_estimate(self, vehicle, current_time: float):
        """
        Update wind estimation using vehicle state.
        
        Args:
            vehicle: Current vehicle state
            current_time: Current simulation time
        """
        # Update wind estimation periodically
        if current_time - self.last_wind_update < control.WIND_ESTIMATION_WINDOW:
            return
            
        # Add new wind sample
        wind_sample = vehicle.velocity - vehicle.velocity_air
        self.wind_samples.append(wind_sample)
        
        # Maintain window of recent samples
        window_size = int(control.WIND_ESTIMATION_WINDOW * control.CONTROL_UPDATE_RATE)
        if len(self.wind_samples) > window_size:
            self.wind_samples.pop(0)
        
        # Calculate weighted average of wind samples
        weights = np.linspace(0.5, 1.0, len(self.wind_samples))
        weighted_sum = np.zeros(3)
        for sample, weight in zip(self.wind_samples, weights):
            weighted_sum += sample * weight
        
        self.wind_estimate = weighted_sum / np.sum(weights)
        self.last_wind_update = current_time
    
    def _update_error_history(self):
        """Update error history with current navigation errors."""
        current_time = time_manager.get_time()
        
        # Add current errors to history
        self.error_history['heading'].append(self.navigation_metrics['heading_error'])
        self.error_history['track'].append(self.navigation_metrics['cross_track_error'])
        self.error_history['altitude'].append(self.navigation_metrics['altitude_error'])
        self.error_history['timestamps'].append(current_time)
        
        # Maintain fixed history length
        if len(self.error_history['timestamps']) > self.history_length:
            for key in self.error_history:
                self.error_history[key].pop(0)
    
    def _update_performance_metrics(self):
        """Update navigation performance metrics."""
        # Update error history
        self._update_error_history()
        
        # Calculate average tracking error
        if len(self.error_history['track']) > 0:
            self.navigation_metrics['average_tracking_error'] = np.mean(
                self.error_history['track']
            )
    
    def _calculate_target_altitude(self) -> float:
        """
        Calculate target altitude based on current phase and next waypoint.
        
        Returns:
            float: Target altitude in meters
        """
        if not self.next_waypoint:
            return 0.0
            
        if self.phase == "landing":
            # Calculate glide slope for landing
            distance_to_waypoint = np.linalg.norm(
                self.next_waypoint[0][:2] - self.last_waypoint[0][:2]
            )
            if distance_to_waypoint > 0:
                current_distance = np.linalg.norm(
                    self.next_waypoint[0][:2] - self.position[:2]
                )
                return (current_distance / distance_to_waypoint) * (-self.last_waypoint[0][2])
        
        # Default to next waypoint altitude
        return -self.next_waypoint[0][2]
    
    def get_state(self) -> Dict:
        """
        Get current controller state.
        
        Returns:
            Dictionary containing current navigation state
        """
        return {
            'navigation': {
                'current_waypoint': self.current_waypoint_idx,
                'phase': self.phase,
                'phase_time': time_manager.get_time() - self.phase_entry_time
            },
            'control': {
                'left_brake': self.left_brake,
                'right_brake': self.right_brake
            },
            'errors': {
                'cross_track': self.navigation_metrics['cross_track_error'],
                'heading': self.navigation_metrics['heading_error'],
                'altitude': self.navigation_metrics['altitude_error']
            },
            'performance': {
                'waypoints_reached': self.navigation_metrics['waypoints_reached'],
                'phase_transitions': self.navigation_metrics['phase_transitions'],
                'average_tracking_error': self.navigation_metrics['average_tracking_error']
            },
            'wind_estimate': self.wind_estimate.copy()
        }