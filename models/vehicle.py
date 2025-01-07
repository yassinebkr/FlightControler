# models/vehicle.py

"""
Base vehicle class implementing core physics and state management.
Provides foundation for specific vehicle types like paraglider.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from config.general_config import safety
from config.vehicle_config import vehicle
from utils.time_manager import time_manager

class Vehicle:
    def __init__(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None
    ):
        """
        Initialize vehicle state.
        
        Args:
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz]
            orientation: Initial orientation matrix (3x3)
        """
        # Core state vectors with explicit typing and validation
        self.position = np.array(position if position is not None else 
                               vehicle.INITIAL_POSITION, dtype=np.float64)
        self.velocity = np.array(velocity if velocity is not None else 
                               vehicle.INITIAL_VELOCITY, dtype=np.float64)
        self.orientation = np.array(orientation if orientation is not None else 
                                  vehicle.INITIAL_ORIENTATION, dtype=np.float64)
        
        # Ensure orientation is orthonormal
        if orientation is not None:
            u, _, vh = np.linalg.svd(self.orientation)
            self.orientation = u @ vh
        
        # Timing management
        self.last_update = time_manager.get_time()
        self.dt_accumulator = 0.0
        self.fixed_dt = 0.01  # Fixed physics timestep
        
        # Physical properties
        self.mass = vehicle.MASS
        self.inertia = self._calculate_inertia()
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        # Motion state
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.angular_acceleration = np.zeros(3, dtype=np.float64)
        
        # Ground interaction flag
        self.ground_contact = False
        
        # Force and moment accumulators with explicit typing
        self.forces: List[Tuple[np.ndarray, np.ndarray]] = []
        self.moments: List[np.ndarray] = []
        
        # Stability tracking
        self._prev_position = self.position.copy()
        self._prev_velocity = self.velocity.copy()
        self.max_position_change = 0.0
        self.max_velocity_change = 0.0
        self.max_acceleration = 0.0
        self.instability_count = 0
    
    def _calculate_inertia(self) -> np.ndarray:
        """
        Calculate basic inertia tensor.
        
        Returns:
            3x3 inertia tensor matrix
        """
        # Improved inertia calculation for stability
        length = self.mass ** (1/3)  # Approximate size from mass
        width = length * 0.8
        height = length * 1.2
        
        # Principal moments with minimum values for stability
        min_inertia = self.mass * 0.01  # Minimum inertia value
        Ixx = max(min_inertia, (1/12) * self.mass * (height**2 + width**2))
        Iyy = max(min_inertia, (1/12) * self.mass * (length**2 + height**2))
        Izz = max(min_inertia, (1/12) * self.mass * (length**2 + width**2))
        
        return np.diag([Ixx, Iyy, Izz])
    
    def add_force(self, force: np.ndarray, point: Optional[np.ndarray] = None):
        """
        Add force at specified point (defaults to CG).
        
        Args:
            force: Force vector [Fx, Fy, Fz]
            point: Point of force application
        """
        # Input validation and conversion
        force = np.asarray(force, dtype=np.float64)
        
        # Force magnitude limiting
        max_force = safety.MAX_ACCELERATION_LIMIT * self.mass
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > max_force:
            force = force * (max_force / force_magnitude)
        
        # Handle application point
        if point is None:
            point = np.zeros(3, dtype=np.float64)
        else:
            point = np.asarray(point, dtype=np.float64)
            # Limit moment arm for stability
            max_arm = 10.0  # meters
            arm_length = np.linalg.norm(point)
            if arm_length > max_arm:
                point = point * (max_arm / arm_length)
        
        self.forces.append((force, point))
        
        # Add moment if force not at CG
        if not np.allclose(point, 0):
            moment = np.cross(point, force)
            self.add_moment(moment)
    
    def add_moment(self, moment: np.ndarray):
        """
        Add moment/torque with improved stability.
        
        Args:
            moment: Moment vector [Mx, My, Mz]
        """
        moment = np.asarray(moment, dtype=np.float64)
        
        # Limit maximum moment for stability
        max_moment = safety.MAX_ACCELERATION_LIMIT * np.trace(self.inertia) * 0.1
        moment_magnitude = np.linalg.norm(moment)
        if moment_magnitude > max_moment:
            moment = moment * (max_moment / moment_magnitude)
        
        self.moments.append(moment)
    
    def update(self, dt: float) -> bool:
        """
        Update vehicle state using semi-implicit Euler integration.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            bool: True if update successful
        """
        try:
            # Accumulate time step
            self.dt_accumulator += dt
            
            # Run fixed time step updates
            while self.dt_accumulator >= self.fixed_dt:
                if not self._fixed_step_update(self.fixed_dt):
                    return False
                self.dt_accumulator -= self.fixed_dt
            
            # Update timing
            self.last_update = time_manager.get_time()
            return True
            
        except Exception as e:
            print(f"Physics update error: {str(e)}")
            return False
    
    def _fixed_step_update(self, dt: float) -> bool:
        """
        Perform single fixed time step update.
        
        Args:
            dt: Fixed time step
            
        Returns:
            bool: True if update successful
        """
        # Store previous state
        self._prev_position = self.position.copy()
        self._prev_velocity = self.velocity.copy()
        
        # Calculate net force and moment
        net_force = np.sum([f for f, _ in self.forces], axis=0)
        net_moment = np.sum(self.moments, axis=0)
        
        # Clear accumulators
        self.forces.clear()
        self.moments.clear()
        
        # Update linear motion (semi-implicit Euler)
        self.acceleration = net_force / self.mass
        new_velocity = self.velocity + self.acceleration * dt
        new_position = self.position + new_velocity * dt
        
        # Update angular motion
        self.angular_acceleration = self.inertia_inv @ net_moment
        new_angular_velocity = self.angular_velocity + self.angular_acceleration * dt
        
        # Update orientation using quaternion integration
        orientation_valid = self._update_orientation(dt, new_angular_velocity)
        if not orientation_valid:
            return False
        
        # Validate and apply new state
        if self._validate_state_update(new_position, new_velocity, new_angular_velocity):
            self.position = new_position
            self.velocity = new_velocity
            self.angular_velocity = new_angular_velocity
            return True
        
        return False
    
    def _update_orientation(self, dt: float, angular_velocity: np.ndarray) -> bool:
        """
        Update orientation using quaternion integration.
        
        Args:
            dt: Time step
            angular_velocity: Angular velocity vector
            
        Returns:
            bool: True if update successful
        """
        try:
            # Convert current orientation to quaternion
            q = self._matrix_to_quaternion(self.orientation)
            
            # Calculate quaternion derivative
            w_magnitude = np.linalg.norm(angular_velocity)
            if w_magnitude > 1e-10:
                axis = angular_velocity / w_magnitude
                angle = w_magnitude * dt
                dq = np.array([
                    np.cos(angle/2),
                    axis[0] * np.sin(angle/2),
                    axis[1] * np.sin(angle/2),
                    axis[2] * np.sin(angle/2)
                ])
                
                # Apply quaternion multiplication
                q = self._quaternion_multiply(q, dq)
                
                # Normalize quaternion
                q = q / np.linalg.norm(q)
                
                # Convert back to rotation matrix
                self.orientation = self._quaternion_to_matrix(q)
                
                # Ensure orthonormality
                u, _, vh = np.linalg.svd(self.orientation)
                self.orientation = u @ vh
            
            return True
            
        except Exception as e:
            print(f"Orientation update error: {str(e)}")
            return False
    
    def _validate_state_update(self, 
                             new_position: np.ndarray,
                             new_velocity: np.ndarray,
                             new_angular_velocity: np.ndarray) -> bool:
        """
        Validate state update values.
        
        Args:
            new_position: New position vector
            new_velocity: New velocity vector
            new_angular_velocity: New angular velocity vector
            
        Returns:
            bool: True if state update is valid
        """
        # Check for NaN or infinite values
        if (np.any(np.isnan(new_position)) or np.any(np.isnan(new_velocity)) or
            np.any(np.isnan(new_angular_velocity)) or
            np.any(np.isinf(new_position)) or np.any(np.isinf(new_velocity)) or
            np.any(np.isinf(new_angular_velocity))):
            return False
        
        # Check position limits
        if np.any(np.abs(new_position) > safety.MAX_POSITION_LIMIT):
            return False
        
        # Check velocity limits
        if np.linalg.norm(new_velocity) > safety.MAX_VELOCITY_LIMIT:
            return False
        
        # Check acceleration
        velocity_change = new_velocity - self.velocity
        acceleration_magnitude = np.linalg.norm(velocity_change) / self.fixed_dt
        if acceleration_magnitude > safety.MAX_ACCELERATION_LIMIT:
            return False
        
        # Check angular velocity limits
        max_angular_velocity = 10.0  # rad/s
        if np.linalg.norm(new_angular_velocity) > max_angular_velocity:
            return False
        
        return True
    
    def _matrix_to_quaternion(self, matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion."""
        trace = np.trace(matrix)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (matrix[2,1] - matrix[1,2]) / S
            y = (matrix[0,2] - matrix[2,0]) / S
            z = (matrix[1,0] - matrix[0,1]) / S
        elif matrix[0,0] > matrix[1,1] and matrix[0,0] > matrix[2,2]:
            S = np.sqrt(1.0 + matrix[0,0] - matrix[1,1] - matrix[2,2]) * 2
            w = (matrix[2,1] - matrix[1,2]) / S
            x = 0.25 * S
            y = (matrix[0,1] + matrix[1,0]) / S
            z = (matrix[0,2] + matrix[2,0]) / S
        elif matrix[1,1] > matrix[2,2]:
            S = np.sqrt(1.0 + matrix[1,1] - matrix[0,0] - matrix[2,2]) * 2
            w = (matrix[0,2] - matrix[2,0]) / S
            x = (matrix[0,1] + matrix[1,0]) / S
            y = 0.25 * S
            z = (matrix[1,2] + matrix[2,1]) / S
        else:
            S = np.sqrt(1.0 + matrix[2,2] - matrix[0,0] - matrix[1,1]) * 2
            w = (matrix[1,0] - matrix[0,1]) / S
            x = (matrix[0,2] + matrix[2,0]) / S
            y = (matrix[1,2] + matrix[2,1]) / S
            z = 0.25 * S
        return np.array([w, x, y, z])
    
    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])
    
    def get_state(self) -> Dict:
        """
        Get complete vehicle state.
        
        Returns:
            Dictionary containing current vehicle state
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'angular_acceleration': self.angular_acceleration.copy(),
            'stability': {
                'max_position_change': self.max_position_change,
                'max_velocity_change': self.max_velocity_change,
                'max_acceleration': self.max_acceleration,
                'instability_count': self.instability_count
            },
            'timing': {
                'last_update': self.last_update,
                'dt_accumulator': self.dt_accumulator,
                'fixed_dt': self.fixed_dt
            },
            'ground_contact': self.ground_contact
        }