# controllers/state_estimator.py

"""
Extended Kalman Filter (EKF) based state estimator for paraglider navigation.
Performs sensor fusion from multiple sources to maintain accurate state estimates.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from config.constants import GRAVITY_VECTOR
from config.flight_config import sensors
from utils.time_manager import time_manager

class StateEstimator:
    """Extended Kalman Filter implementation for state estimation."""
    
    def __init__(self):
        """Initialize state estimator with proper covariance handling."""
        # State vector dimensions
        # [position(3), velocity(3), orientation(3), angular_rates(3), wind(3)]
        self.nx = 15
        
        # Initialize state vector
        self.state = np.zeros(self.nx, dtype=np.float64)
        
        # Initialize covariance with realistic uncertainties
        self.covariance = np.diag([
            25.0, 25.0, 25.0,     # Position uncertainty (5m std)
            4.0, 4.0, 4.0,        # Velocity uncertainty (2m/s std)
            0.04, 0.04, 0.04,     # Orientation uncertainty (0.2rad std)
            0.01, 0.01, 0.01,     # Angular rate uncertainty (0.1rad/s std)
            16.0, 16.0, 16.0      # Wind uncertainty (4m/s std)
        ])
        
        # Process noise based on expected dynamics
        self.Q = np.diag([
            0.1, 0.1, 0.1,    # Position process noise
            0.5, 0.5, 0.5,    # Velocity process noise
            0.01, 0.01, 0.01, # Orientation process noise
            0.05, 0.05, 0.05, # Angular rate process noise
            0.2, 0.2, 0.2     # Wind process noise
        ])
        
        # Measurement noise matrices
        self._init_measurement_noise()
        
        # Initialize timing
        self.last_update = time_manager.get_time()
        self.dt = 0.02  # Default 50Hz
        
        # Measurement validity flags
        self.measurement_valid = {
            'gps': False,
            'imu': False,
            'baro': False
        }
        
        # Create state history buffer for smoothing
        self.state_history = []
        self.history_length = 50
    
    def _init_measurement_noise(self):
        """Initialize measurement noise matrices with sensor specifications."""
        # GPS noise matrix [position(3), velocity(3)]
        self.R_gps = np.diag([
            sensors.gps.POSITION_NOISE**2,
            sensors.gps.POSITION_NOISE**2,
            sensors.gps.POSITION_NOISE**2,
            sensors.gps.VELOCITY_NOISE**2,
            sensors.gps.VELOCITY_NOISE**2,
            sensors.gps.VELOCITY_NOISE**2
        ])
        
        # IMU noise matrix [acceleration(3), angular_rates(3)]
        self.R_imu = np.diag([
            sensors.imu.ACCELEROMETER_NOISE**2,
            sensors.imu.ACCELEROMETER_NOISE**2,
            sensors.imu.ACCELEROMETER_NOISE**2,
            sensors.imu.GYROSCOPE_NOISE**2,
            sensors.imu.GYROSCOPE_NOISE**2,
            sensors.imu.GYROSCOPE_NOISE**2
        ])
        
        # Barometer noise (scalar)
        self.R_baro = np.array([[sensors.baro.ALTITUDE_NOISE**2]])
    
    def predict(self, dt: Optional[float] = None):
        """
        Perform EKF prediction step with improved dynamics model.
        
        Args:
            dt: Time step (uses default if None)
        """
        if dt is not None:
            self.dt = dt
        
        # Extract current state components
        pos = self.state[0:3]
        vel = self.state[3:6]
        orientation = self.state[6:9]
        rates = self.state[9:12]
        wind = self.state[12:15]
        
        # Predict state using nonlinear dynamics
        new_pos = pos + vel * self.dt
        new_vel = vel  # Assume constant velocity
        new_orientation = self._predict_orientation(orientation, rates)
        new_rates = rates  # Assume constant angular rates
        new_wind = wind  # Assume constant wind
        
        # Update state vector
        self.state = np.concatenate([
            new_pos, new_vel, new_orientation, new_rates, new_wind
        ])
        
        # Calculate state transition Jacobian
        F = self._compute_state_transition_jacobian()
        
        # Update covariance
        self.covariance = F @ self.covariance @ F.T + self.Q * self.dt
        
        # Ensure covariance stays symmetric
        self.covariance = (self.covariance + self.covariance.T) / 2
    
    def _predict_orientation(self, orientation: np.ndarray,
                           rates: np.ndarray) -> np.ndarray:
        """
        Predict orientation using quaternion integration.
        
        Args:
            orientation: Current orientation angles
            rates: Angular rates
            
        Returns:
            Updated orientation angles
        """
        # Convert euler angles to quaternion
        q = self._euler_to_quaternion(orientation)
        
        # Create angular velocity quaternion
        w_quat = np.array([0, rates[0], rates[1], rates[2]])
        
        # Quaternion derivative
        q_dot = 0.5 * self._quaternion_multiply(q, w_quat)
        
        # Integrate quaternion
        q_new = q + q_dot * self.dt
        
        # Normalize quaternion
        q_new = q_new / np.linalg.norm(q_new)
        
        # Convert back to euler angles
        return self._quaternion_to_euler(q_new)
    
    def _compute_state_transition_jacobian(self) -> np.ndarray:
        """
        Compute Jacobian of state transition function.
        
        Returns:
            State transition Jacobian matrix
        """
        F = np.eye(self.nx)
        
        # Position from velocity
        F[0:3, 3:6] = np.eye(3) * self.dt
        
        # Orientation from angular rates (linearized)
        F[6:9, 9:12] = np.eye(3) * self.dt
        
        return F
    
    def update_gps(self, pos_meas: np.ndarray, vel_meas: np.ndarray):
        """
        Update state with GPS measurements.
        
        Args:
            pos_meas: Position measurement [x, y, z]
            vel_meas: Velocity measurement [vx, vy, vz]
        """
        if pos_meas is None or vel_meas is None:
            self.measurement_valid['gps'] = False
            return
        
        self.measurement_valid['gps'] = True
        
        # Combine measurements
        z = np.concatenate([pos_meas, vel_meas])
        
        # Create measurement matrix
        H = np.zeros((6, self.nx))
        H[0:3, 0:3] = np.eye(3)  # Position measurement
        H[3:6, 3:6] = np.eye(3)  # Velocity measurement
        
        # Compute innovation
        innovation = z - np.concatenate([self.state[0:3], self.state[3:6]])
        
        # Perform update
        self._update(innovation, H, self.R_gps)
    
    def update_imu(self, accel: np.ndarray, gyro: np.ndarray):
        """
        Update state with IMU measurements.
        
        Args:
            accel: Acceleration measurement [ax, ay, az]
            gyro: Angular rate measurement [wx, wy, wz]
        """
        if accel is None or gyro is None:
            self.measurement_valid['imu'] = False
            return
            
        self.measurement_valid['imu'] = True
        
        # Calculate expected acceleration
        orientation = self.state[6:9]
        expected_accel = self._rotate_vector(GRAVITY_VECTOR, orientation)
        
        # Get expected angular rates
        expected_rates = self.state[9:12]
        
        # Combine measurements
        z = np.concatenate([accel, gyro])
        z_expected = np.concatenate([expected_accel, expected_rates])
        
        # Create measurement matrix
        H = np.zeros((6, self.nx))
        H[0:3, 6:9] = self._gravity_jacobian(orientation)
        H[3:6, 9:12] = np.eye(3)
        
        # Perform update
        self._update(z - z_expected, H, self.R_imu)
    
    def update_baro(self, altitude: float):
        """
        Update state with barometer measurement.
        
        Args:
            altitude: Altitude measurement
        """
        if altitude is None:
            self.measurement_valid['baro'] = False
            return
            
        self.measurement_valid['baro'] = True
        
        z = np.array([altitude])
        H = np.zeros((1, self.nx))
        H[0, 2] = -1.0  # Negative z position
        
        self._update(z - (-self.state[2]), H, self.R_baro)
    
    def _update(self, innovation: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Perform Kalman filter update step with numerical stability checks.
        
        Args:
            innovation: Measurement innovation
            H: Measurement matrix
            R: Measurement noise covariance
        """
        # Compute innovation covariance
        S = H @ self.covariance @ H.T + R
        
        try:
            # Use Cholesky decomposition for numerical stability
            L = np.linalg.cholesky(S)
            K = self.covariance @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.state = self.state + K @ innovation
            
            # Update covariance using Joseph form for numerical stability
            I = np.eye(self.nx)
            temp = I - K @ H
            self.covariance = (temp @ self.covariance @ temp.T + 
                             K @ R @ K.T)
            
            # Ensure covariance stays symmetric
            self.covariance = (self.covariance + self.covariance.T) / 2
            
        except np.linalg.LinAlgError as e:
            print(f"Numerical error in Kalman update: {e}")
            
        # Update state history
        self._update_state_history()
    
    def _update_state_history(self):
        """Maintain state history for smoothing operations."""
        self.state_history.append({
            'state': self.state.copy(),
            'covariance': np.diag(self.covariance).copy(),
            'time': time_manager.get_time()
        })
        
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)
    
    def _rotate_vector(self, vector: np.ndarray, euler: np.ndarray) -> np.ndarray:
        """
        Rotate vector using euler angles.
        
        Args:
            vector: Vector to rotate
            euler: Euler angles [roll, pitch, yaw]
            
        Returns:
            Rotated vector
        """
        # Create rotation matrices
        roll, pitch, yaw = euler
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        return R @ vector
    
    def _gravity_jacobian(self, orientation: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of gravity vector with respect to orientation.
        
        Args:
            orientation: Euler angles [roll, pitch, yaw]
            
        Returns:
            Jacobian matrix
        """
        roll, pitch, yaw = orientation
        g = GRAVITY_VECTOR[2]
        
        # Partial derivatives with respect to roll
        J_roll = np.array([
            [0],
            [g * np.cos(roll) * np.cos(pitch)],
            [-g * np.sin(roll) * np.cos(pitch)]
        ])
        
        # Partial derivatives with respect to pitch
        J_pitch = np.array([
            [-g * np.cos(pitch)],
            [g * np.sin(roll) * np.sin(pitch)],
            [g * np.cos(roll) * np.sin(pitch)]
        ])
        
        # Partial derivatives with respect to yaw (gravity independent)
        J_yaw = np.zeros((3, 1))
        
        return np.hstack([J_roll, J_pitch, J_yaw])
    
    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """
        Convert euler angles to quaternion.
        
        Args:
            euler: Euler angles [roll, pitch, yaw]
            
        Returns:
            Quaternion [w, x, y, z]
        """
        roll, pitch, yaw = euler
        
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def _quaternion_to_euler(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to euler angles.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Euler angles [roll, pitch, yaw]
        """
        # Extract quaternion components
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions.
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            
        Returns:
            Resulting quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
    
    def get_state(self) -> Dict:
        """
        Get current state estimate with uncertainty information.
        
        Returns:
            Dictionary containing state estimates and uncertainties
        """
        state_dict = {
            'position': {
                'value': self.state[0:3],
                'covariance': self.covariance[0:3, 0:3].diagonal()
            },
            'velocity': {
                'value': self.state[3:6],
                'covariance': self.covariance[3:6, 3:6].diagonal()
            },
            'orientation': {
                'value': self.state[6:9],
                'covariance': self.covariance[6:9, 6:9].diagonal()
            },
            'angular_rates': {
                'value': self.state[9:12],
                'covariance': self.covariance[9:12, 9:12].diagonal()
            },
            'wind': {
                'value': self.state[12:15],
                'covariance': self.covariance[12:15, 12:15].diagonal()
            },
            'measurement_valid': self.measurement_valid.copy()
        }
        
        # Add smoothed estimates if available
        if len(self.state_history) > 1:
            smoothed_state = self._get_smoothed_state()
            state_dict['smoothed'] = smoothed_state
        
        return state_dict
    
    def _get_smoothed_state(self) -> Dict:
        """
        Get smoothed state estimate using recent history.
        
        Returns:
            Dictionary containing smoothed state estimates
        """
        # Calculate weighted average of recent states
        weights = np.linspace(0.5, 1.0, len(self.state_history))
        weights /= np.sum(weights)
        
        smoothed_state = np.zeros_like(self.state)
        smoothed_covariance = np.zeros_like(self.covariance.diagonal())
        
        for hist, weight in zip(self.state_history, weights):
            smoothed_state += hist['state'] * weight
            smoothed_covariance += hist['covariance'] * weight
        
        return {
            'state': smoothed_state,
            'covariance': smoothed_covariance
        }