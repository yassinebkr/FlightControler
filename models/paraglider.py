# models/paraglider.py

"""
Paraglider physics model with aerodynamics and control surfaces.
Implements complex flight dynamics with wind interaction.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .vehicle import Vehicle
from config.constants import GRAVITY, GRAVITY_VECTOR
from config.vehicle_config import vehicle as vehicle_config
from config.general_config import safety
from utils.time_manager import time_manager

class Paraglider(Vehicle):
    """Physics model for paraglider flight dynamics."""
    
    def __init__(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None
    ):
        """
        Initialize paraglider state.
        
        Args:
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz]
            orientation: Initial orientation matrix (3x3)
        """
        # Initialize base vehicle
        super().__init__(position, velocity, orientation)
        
        # Wing configuration
        self.wing_area = vehicle_config.WING_AREA
        self.wing_span = vehicle_config.WING_SPAN
        self.aspect_ratio = vehicle_config.ASPECT_RATIO
        
        # Control state
        self.left_brake = 0.0
        self.right_brake = 0.0
        self.brake_positions = np.zeros(2, dtype=np.float64)
        self.prev_brake_positions = np.zeros(2, dtype=np.float64)
        self.max_brake_change = vehicle_config.BRAKE_REACTION_SPEED * self.fixed_dt
        
        # Flight state
        self.angle_of_attack = 0.0
        self.sideslip_angle = 0.0
        self.airspeed = 0.0
        self.load_factor = 1.0
        
        # Wind interaction
        self.velocity_air = np.zeros(3, dtype=np.float64)
        self.wind_velocity = np.zeros(3, dtype=np.float64)
        self.prev_wind = np.zeros(3, dtype=np.float64)
        self.wind_acceleration = np.zeros(3, dtype=np.float64)
        
        # Initialize flight statistics
        self._init_flight_statistics()
        
        # State history
        self._init_state_history()
    
    def _init_flight_statistics(self):
        """Initialize flight performance tracking."""
        self.flight_statistics = {
            'max_load_factor': 0.0,
            'max_sink_rate': 0.0,
            'total_distance': 0.0,
            'average_glide_ratio': 0.0,
            'ground_contacts': 0,
            'min_airspeed': float('inf'),
            'max_airspeed': 0.0,
            'max_bank_angle': 0.0,
            'stability_violations': 0
        }
    
    def _init_state_history(self):
        """Initialize state history tracking."""
        self.state_history = {
            'positions': [],
            'velocities': [],
            'orientations': [],
            'brake_positions': [],
            'airspeeds': [],
            'load_factors': [],
            'timestamps': []
        }
        self.history_length = 100  # Number of states to track
    
    def set_control_inputs(self, left_brake: float, right_brake: float):
        """
        Set brake control inputs with rate limiting and smoothing.
        
        Args:
            left_brake: Left brake input [0-1]
            right_brake: Right brake input [0-1]
        """
        # Store previous brake positions
        self.prev_brake_positions = self.brake_positions.copy()
        
        # Validate and clip inputs
        left_brake = np.clip(left_brake, 0.0, vehicle_config.MAX_BRAKE_DEFLECTION)
        right_brake = np.clip(right_brake, 0.0, vehicle_config.MAX_BRAKE_DEFLECTION)
        
        # Apply rate limiting
        for i, (target, current) in enumerate([(left_brake, self.brake_positions[0]),
                                             (right_brake, self.brake_positions[1])]):
            delta = target - current
            delta = np.clip(delta, -self.max_brake_change, self.max_brake_change)
            self.brake_positions[i] = current + delta
        
        # Update brake values
        self.left_brake = self.brake_positions[0]
        self.right_brake = self.brake_positions[1]
    
    def update(self, dt: float, environment_conditions: Optional[Dict] = None) -> bool:
        """
        Update paraglider physics with improved stability.
        
        Args:
            dt: Time step in seconds
            environment_conditions: Current environmental conditions
            
        Returns:
            bool: True if update successful
        """
        try:
            # Update environmental interactions
            if environment_conditions:
                if not self._update_environmental_state(environment_conditions):
                    return False
            
            # Calculate and apply forces
            self._calculate_forces()
            
            # Update base physics
            if not super().update(dt):
                return False
            
            # Handle ground interaction
            if not self._handle_ground_interaction(dt):
                return False
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Update state history
            self._update_state_history()
            
            return True
            
        except Exception as e:
            print(f"Paraglider physics update error: {str(e)}")
            return False
    
    def _update_environmental_state(self, conditions: Dict) -> bool:
        """
        Update state based on environmental conditions.
        
        Args:
            conditions: Environmental conditions dictionary
            
        Returns:
            bool: True if update successful
        """
        try:
            # Store previous wind for acceleration calculation
            self.prev_wind = self.wind_velocity.copy()
            
            # Update wind state
            self.wind_velocity = np.array(conditions['wind'], dtype=np.float64)
            
            # Calculate wind acceleration for gust response
            dt = time_manager.get_time() - self.last_update
            if dt > 0:
                self.wind_acceleration = (self.wind_velocity - self.prev_wind) / dt
            
            # Calculate airspeed vector and magnitude
            self.velocity_air = self.velocity - self.wind_velocity
            self.airspeed = np.linalg.norm(self.velocity_air)
            
            # Update air density
            self.air_density = conditions['air_density']
            
            # Add thermal effects if present
            if 'thermal' in conditions:
                thermal_velocity = np.array([0, 0, conditions['thermal']['vertical_velocity']])
                self.add_force(thermal_velocity * self.mass)
            
            return True
            
        except Exception as e:
            print(f"Environmental state update error: {str(e)}")
            return False
    
    def _calculate_forces(self):
        """Calculate and apply aerodynamic forces with improved model."""
        # Apply gravity
        self.add_force(GRAVITY_VECTOR * self.mass)
        
        # Only calculate aero forces if airspeed is sufficient
        if self.airspeed < 1e-3:
            return
        
        # Calculate control inputs
        brake_avg = (self.left_brake + self.right_brake) / 2
        brake_diff = self.right_brake - self.left_brake
        
        # Transform airspeed to body frame
        airspeed_body = self.orientation.T @ self.velocity_air
        
        # Calculate aerodynamic angles
        self.angle_of_attack = np.arctan2(-airspeed_body[2], airspeed_body[0])
        self.sideslip_angle = np.arctan2(airspeed_body[1], airspeed_body[0])
        
        # Calculate effective angle of attack with brake effect
        brake_aoa_effect = brake_avg * vehicle_config.BRAKE_EFFECTIVENESS
        aoa_effective = self.angle_of_attack + brake_aoa_effect
        
        # Calculate lift coefficient with improved stall model
        cl = self._calculate_lift_coefficient(aoa_effective)
        
        # Calculate drag coefficient
        cd = self._calculate_drag_coefficient(cl, brake_avg)
        
        # Calculate side force coefficient
        cy = self._calculate_side_force_coefficient(self.sideslip_angle)
        
        # Calculate dynamic pressure
        dynamic_pressure = 0.5 * self.air_density * self.airspeed**2
        
        # Calculate forces
        lift = cl * dynamic_pressure * self.wing_area
        drag = cd * dynamic_pressure * self.wing_area
        side_force = cy * dynamic_pressure * self.wing_area
        
        # Transform forces to world frame
        force_body = np.array([-drag, side_force, -lift])
        force_world = self.orientation @ force_body
        
        # Apply forces
        self.add_force(force_world)
        
        # Calculate and apply moments
        self._calculate_moments(brake_diff, dynamic_pressure)
    
    def _calculate_lift_coefficient(self, aoa: float) -> float:
        """
        Calculate lift coefficient with improved stall model.
        
        Args:
            aoa: Effective angle of attack
            
        Returns:
            float: Lift coefficient
        """
        # Linear region
        cl = vehicle_config.LIFT_SLOPE * aoa
        
        # Stall modeling
        if abs(aoa) > vehicle_config.STALL_ANGLE:
            # Smooth stall transition
            stall_factor = 1.0 - np.tanh(
                (abs(aoa) - vehicle_config.STALL_ANGLE) / 
                (vehicle_config.STALL_ANGLE * 0.2)
            )
            
            # Add post-stall lift
            post_stall = 0.5 * np.sign(aoa) * np.sin(2 * aoa)
            
            cl = cl * stall_factor + post_stall * (1 - stall_factor)
        
        # Limit maximum lift
        return np.clip(cl, -vehicle_config.LIFT_MAX, vehicle_config.LIFT_MAX)
    
    def _calculate_drag_coefficient(self, cl: float, brake_setting: float) -> float:
        """
        Calculate total drag coefficient.
        
        Args:
            cl: Current lift coefficient
            brake_setting: Average brake setting
            
        Returns:
            float: Drag coefficient
        """
        # Parasitic drag
        cd0 = vehicle_config.ZERO_LIFT_DRAG
        
        # Induced drag
        induced_drag = (vehicle_config.INDUCED_DRAG_FACTOR * cl**2 / 
                       (np.pi * self.aspect_ratio))
        
        # Brake drag
        brake_drag = brake_setting**2 * 0.1
        
        # Add small drag offset for stability
        min_drag = 0.01
        
        return cd0 + induced_drag + brake_drag + min_drag
    
    def _calculate_side_force_coefficient(self, sideslip: float) -> float:
        """
        Calculate side force coefficient.
        
        Args:
            sideslip: Sideslip angle
            
        Returns:
            float: Side force coefficient
        """
        # Linear side force with saturation
        cy_slope = -1.0  # Side force slope
        cy = cy_slope * np.sin(sideslip)
        
        # Limit maximum side force
        return np.clip(cy, -0.5, 0.5)
    
    def _calculate_moments(self, brake_diff: float, dynamic_pressure: float):
        """
        Calculate and apply aerodynamic moments.
        
        Args:
            brake_diff: Brake differential
            dynamic_pressure: Dynamic pressure
        """
        # Roll moment from brake differential
        roll_coeff = -brake_diff * vehicle_config.BRAKE_EFFECTIVENESS
        roll_moment = (roll_coeff * dynamic_pressure * 
                      self.wing_area * self.wing_span)
        
        # Pitch moment for stability
        pitch_coeff = -0.1 * self.angle_of_attack  # Pitch stability
        pitch_moment = (pitch_coeff * dynamic_pressure * 
                       self.wing_area * self.wing_span)
        
        # Yaw moment from sideslip and brake differential
        yaw_coeff = 0.2 * self.sideslip_angle  # Weather vane stability
        yaw_coeff += brake_diff * 0.1  # Differential brake effect
        yaw_moment = yaw_coeff * dynamic_pressure * self.wing_area * self.wing_span
        
        # Combine moments
        moment = np.array([roll_moment, pitch_moment, yaw_moment])
        
        # Add damping moments
        damping = -self.angular_velocity * 0.1
        moment += damping
        
        self.add_moment(moment)
    
    def _handle_ground_interaction(self, dt: float) -> bool:
        """
        Handle collision with ground with improved physics.
        
        Args:
            dt: Time step
            
        Returns:
            bool: True if handling successful
        """
        try:
            if self.position[2] <= 0:
                # Record ground contact
                if not self.ground_contact:
                    self.ground_contact = True
                    self.flight_statistics['ground_contacts'] += 1
                    
                    # Check landing speed
                    if np.linalg.norm(self.velocity) > safety.MAX_VELOCITY_LIMIT * 0.5:
                        print("Warning: Hard landing detected")
                        return False
                
                # Set altitude to ground level
                self.position[2] = 0
                
                # Handle ground reaction
                if self.velocity[2] < 0:
                    # Normal force
                    normal_velocity = self.velocity[2]
                    self.velocity[2] = abs(normal_velocity * vehicle_config.BOUNCE_DAMPING)
                    
                    # Ground friction
                    horizontal_velocity = self.velocity[:2]
                    horizontal_speed = np.linalg.norm(horizontal_velocity)
                    
                    if horizontal_speed > 0:
                        friction_decel = (vehicle_config.GROUND_FRICTION * GRAVITY * 
                                        vehicle_config.SLIDE_DAMPING)
                        friction_factor = max(0, 1 - friction_decel * dt / horizontal_speed)
                        self.velocity[:2] *= friction_factor
                
                # Prevent penetration
                self.velocity[2] = max(0, self.velocity[2])
            else:
                self.ground_contact = False
            
            return True
            
        except Exception as e:
            print(f"Ground interaction error: {str(e)}")
            return False
    
    def _update_performance_metrics(self):
        """Update flight performance tracking."""
        # Update load factor
        self.load_factor = np.linalg.norm(self.acceleration) / GRAVITY
        self.flight_statistics['max_load_factor'] = max(
            self.flight_statistics['max_load_factor'],
            self.load_factor
        )
        
        # Track maximum sink rate
        sink_rate = max(0, -self.velocity[2])
        self.flight_statistics['max_sink_rate'] = max(
            self.flight_statistics['max_sink_rate'],
            sink_rate
        )
        
        # Update airspeed extremes
        self.flight_statistics['min_airspeed'] = min(
            self.flight_statistics['min_airspeed'],
            self.airspeed
        )
        self.flight_statistics['max_airspeed'] = max(
            self.flight_statistics['max_airspeed'],
            self.airspeed
        )
        
        # Calculate bank angle
        bank_angle = np.arccos(np.clip(self.orientation[2,2], -1.0, 1.0))
        self.flight_statistics['max_bank_angle'] = max(
            self.flight_statistics['max_bank_angle'],
            abs(bank_angle)
        )
        
        # Update distance tracking
        if len(self.state_history['positions']) > 1:
            distance = np.linalg.norm(
                self.position - self.state_history['positions'][-1]
            )
            self.flight_statistics['total_distance'] += distance
        
        # Calculate glide ratio when descending
        if self.velocity[2] < -0.1:  # Only when descending
            horizontal_speed = np.linalg.norm(self.velocity[:2])
            current_glide_ratio = horizontal_speed / abs(self.velocity[2])
            
            # Update moving average of glide ratio
            alpha = 0.05  # Smoothing factor
            if self.flight_statistics['average_glide_ratio'] == 0:
                self.flight_statistics['average_glide_ratio'] = current_glide_ratio
            else:
                self.flight_statistics['average_glide_ratio'] = (
                    (1 - alpha) * self.flight_statistics['average_glide_ratio'] +
                    alpha * current_glide_ratio
                )
    
    def _update_state_history(self):
        """Update state history with memory management."""
        current_time = time_manager.get_time()
        
        # Add current state to history
        self.state_history['positions'].append(self.position.copy())
        self.state_history['velocities'].append(self.velocity.copy())
        self.state_history['orientations'].append(self.orientation.copy())
        self.state_history['brake_positions'].append(self.brake_positions.copy())
        self.state_history['airspeeds'].append(self.airspeed)
        self.state_history['load_factors'].append(self.load_factor)
        self.state_history['timestamps'].append(current_time)
        
        # Maintain fixed history length
        if len(self.state_history['timestamps']) > self.history_length:
            for key in self.state_history:
                self.state_history[key].pop(0)
    
    def get_state(self) -> Dict:
        """
        Get complete paraglider state.
        
        Returns:
            Dictionary containing current paraglider state
        """
        # Get base vehicle state
        state = super().get_state()
        
        # Add paraglider-specific state
        state.update({
            'control': {
                'left_brake': self.left_brake,
                'right_brake': self.right_brake,
                'brake_positions': self.brake_positions.copy()
            },
            'aero': {
                'airspeed': self.airspeed,
                'angle_of_attack': self.angle_of_attack,
                'sideslip_angle': self.sideslip_angle,
                'load_factor': self.load_factor,
                'air_density': getattr(self, 'air_density', 1.225)
            },
            'wind': {
                'velocity': self.wind_velocity.copy(),
                'relative_velocity': self.velocity_air.copy(),
                'acceleration': self.wind_acceleration.copy()
            },
            'performance': self.flight_statistics.copy(),
            'history': {
                'length': len(self.state_history['timestamps']),
                'last_timestamp': self.state_history['timestamps'][-1] if self.state_history['timestamps'] else 0
            }
        })
        
        return state