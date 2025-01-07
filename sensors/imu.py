# sensors/imu.py
import numpy as np
from .sensor_base import VectorSensor
from config.flight_config import SensorConfig
from config.constants import GRAVITY
from utils.time_manager import time_manager

class IMU:
    """Combined IMU with accelerometer and gyroscope."""
    
    def __init__(self):
        self.accelerometer = Accelerometer()
        self.gyroscope = Gyroscope()
        
        # Temperature effects
        self.temperature = 20.0  # °C
        self.temp_coefficient = 0.001  # Noise increase per °C deviation
    
    def measure(self, vehicle):
        """Get IMU measurements."""
        accel = self.accelerometer.measure(vehicle)
        gyro = self.gyroscope.measure(vehicle)
        return accel, gyro
    
    def update_temperature(self, temp):
        """Update temperature compensation."""
        self.temperature = temp
        temp_factor = 1 + abs(temp - 20) * self.temp_coefficient
        
        self.accelerometer.adjust_noise(temp_factor)
        self.gyroscope.adjust_noise(temp_factor)

class Accelerometer(VectorSensor):
    def __init__(self):
        super().__init__(
            update_rate=SensorConfig.IMU.UPDATE_RATE,
            noise_std=np.array([
                SensorConfig.IMU.ACCELEROMETER_NOISE,
                SensorConfig.IMU.ACCELEROMETER_NOISE,
                SensorConfig.IMU.ACCELEROMETER_NOISE
            ])
        )
        self.base_noise = self.noise_std.copy()
    
    def _get_measurement(self, vehicle):
        """Get acceleration in body frame."""
        # Transform gravity to body frame
        gravity_body = vehicle.orientation.T @ np.array([0, 0, -GRAVITY])
        
        # Get acceleration from vehicle
        if hasattr(vehicle, 'acceleration'):
            accel_body = vehicle.orientation.T @ vehicle.acceleration
        else:
            accel_body = np.zeros(3)
        
        return accel_body - gravity_body
    
    def adjust_noise(self, factor):
        """Adjust noise levels."""
        self.noise_std = self.base_noise * factor
        self.covariance = np.diag(self.noise_std**2)

class Gyroscope(VectorSensor):
    def __init__(self):
        super().__init__(
            update_rate=SensorConfig.IMU.UPDATE_RATE,
            noise_std=np.array([
                SensorConfig.IMU.GYROSCOPE_NOISE,
                SensorConfig.IMU.GYROSCOPE_NOISE,
                SensorConfig.IMU.GYROSCOPE_NOISE
            ])
        )
        self.base_noise = self.noise_std.copy()
        
        # Drift characteristics
        self.drift_rate = np.zeros(3)
        self.drift_walk_sigma = 0.0001  # rad/s^2
    
    def _get_measurement(self, vehicle):
        """Get angular velocity measurement."""
        # Update drift
        self.drift_rate += np.random.normal(0, self.drift_walk_sigma, 3)
        
        # Get true angular velocity and add drift
        return vehicle.angular_velocity + self.drift_rate
    
    def adjust_noise(self, factor):
        """Adjust noise levels."""
        self.noise_std = self.base_noise * factor
        self.covariance = np.diag(self.noise_std**2)