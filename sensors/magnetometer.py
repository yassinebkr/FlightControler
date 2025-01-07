# sensors/magnetometer.py
import numpy as np
from .sensor_base import VectorSensor
from config.flight_config import SensorConfig
from config.constants import NORTH_VECTOR, DEG_TO_RAD
from utils.time_manager import time_manager

class Magnetometer(VectorSensor):
    def __init__(self):
        """Initialize magnetometer with configuration parameters."""
        super().__init__(
            update_rate=SensorConfig.Magnetometer.UPDATE_RATE,
            noise_std=np.array([
                SensorConfig.Magnetometer.HEADING_NOISE,
                SensorConfig.Magnetometer.HEADING_NOISE,
                SensorConfig.Magnetometer.HEADING_NOISE
            ])
        )
        
        # Magnetic field parameters
        self.field_strength = 1.0
        self.declination = 0.0  # Local magnetic declination
        self.inclination = 60.0 * DEG_TO_RAD  # Magnetic inclination
        
        # Create magnetic field vector
        self.mag_field = self._create_magnetic_field()
    
    def _create_magnetic_field(self):
        """Create local magnetic field vector."""
        # Start with true north
        field = NORTH_VECTOR.copy()
        
        # Apply declination (rotation around vertical)
        cos_dec = np.cos(self.declination)
        sin_dec = np.sin(self.declination)
        field = np.array([
            field[0] * cos_dec - field[1] * sin_dec,
            field[0] * sin_dec + field[1] * cos_dec,
            field[2]
        ])
        
        # Apply inclination (dip angle)
        cos_inc = np.cos(self.inclination)
        sin_inc = np.sin(self.inclination)
        field = np.array([
            field[0] * cos_inc,
            field[1] * cos_inc,
            -sin_inc
        ])
        
        # Normalize and scale
        field = field / np.linalg.norm(field) * self.field_strength
        return field
    
    def _get_measurement(self, vehicle):
        """Get magnetic field measurement in body frame."""
        # Transform magnetic field to body frame
        mag_body = vehicle.orientation.T @ self.mag_field
        return mag_body
    
    def get_heading(self, vehicle):
        """
        Calculate heading from magnetometer reading.
        
        Args:
            vehicle: Vehicle object with current state
            
        Returns:
            float: Heading angle in radians
        """
        # Get magnetic measurement
        mag = self.measure(vehicle)
        
        # Project to horizontal plane
        mag_horizontal = mag[:2]
        
        # Calculate heading
        heading = np.arctan2(mag_horizontal[0], mag_horizontal[1])
        
        # Correct for magnetic declination
        true_heading = heading - self.declination
        
        # Normalize to [0, 2π]
        true_heading = true_heading % (2 * np.pi)
        
        return true_heading