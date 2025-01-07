# sensors/sensor_base.py
import numpy as np
from utils.time_manager import time_manager

class Sensor:
    """Base class for all sensors."""
    
    def __init__(self, update_rate, noise_std=0.0, bias=0.0):
        self.update_rate = update_rate
        self.noise_std = noise_std
        self.bias = bias
        self.last_update = -np.inf
        self.last_value = None
    
    def measure(self, vehicle):
        """Get sensor measurement with proper timing."""
        current_time = time_manager.get_time()
        
        if current_time - self.last_update < 1.0 / self.update_rate:
            return self.last_value
        
        measured = self._get_measurement(vehicle)
        measured = self._apply_noise(measured)
        
        self.last_update = current_time
        self.last_value = measured
        return measured
    
    def _get_measurement(self, vehicle):
        """Get raw measurement from vehicle state."""
        raise NotImplementedError
    
    def _apply_noise(self, true_value):
        """Apply noise and bias to true value."""
        if np.isscalar(true_value):
            return true_value + self.bias + np.random.normal(0, self.noise_std)
        else:
            return true_value + self.bias + np.random.normal(0, self.noise_std, size=true_value.shape)

class VectorSensor(Sensor):
    """Base class for vector sensors."""
    
    def __init__(self, update_rate, noise_std, bias=None):
        super().__init__(update_rate, noise_std,
                        bias if bias is not None else np.zeros_like(noise_std))
        self.covariance = np.diag(noise_std**2)