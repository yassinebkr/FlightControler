# sensors/gps.py
import numpy as np
from .sensor_base import VectorSensor
from config.flight_config import SensorConfig
from utils.time_manager import time_manager

class GPS(VectorSensor):
    def __init__(self):
        """Initialize GPS sensor with configuration parameters."""
        super().__init__(
            update_rate=SensorConfig.GPS.UPDATE_RATE,
            noise_std=np.array([
                SensorConfig.GPS.POSITION_NOISE,
                SensorConfig.GPS.POSITION_NOISE,
                SensorConfig.GPS.POSITION_NOISE * 1.5
            ])
        )
        
        self.velocity_noise = np.array([
            SensorConfig.GPS.VELOCITY_NOISE,
            SensorConfig.GPS.VELOCITY_NOISE,
            SensorConfig.GPS.VELOCITY_NOISE * 1.5
        ])
        
        self.last_velocity = None
    
    def measure(self, vehicle):
        """
        Get GPS measurement from vehicle.
        
        Args:
            vehicle: Vehicle object with current state
            
        Returns:
            tuple: (position, velocity) measurements with noise
        """
        current_time = time_manager.get_time()
        
        if current_time - self.last_update < 1.0 / self.update_rate:
            return self.last_value, self.last_velocity
        
        # Generate position measurement
        position = self._apply_noise(vehicle.position)
        
        # Generate velocity measurement
        velocity = vehicle.velocity + np.random.normal(0, self.velocity_noise)
        
        self.last_update = current_time
        self.last_value = position
        self.last_velocity = velocity
        
        return position, velocity
    
    def get_position_accuracy(self):
        """Get current position accuracy estimate."""
        return self.noise_std
    
    def get_velocity_accuracy(self):
        """Get current velocity accuracy estimate."""
        return self.velocity_noise