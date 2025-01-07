# sensors/barometer.py
import numpy as np
from .sensor_base import Sensor
from config.flight_config import SensorConfig
from config.constants import (STANDARD_PRESSURE, STANDARD_TEMPERATURE,
                            TEMPERATURE_LAPSE_RATE, GRAVITY, AIR_GAS_CONSTANT)
from utils.time_manager import time_manager

class Barometer(Sensor):
    def __init__(self):
        """Initialize barometer with configuration parameters."""
        super().__init__(
            update_rate=SensorConfig.Barometer.UPDATE_RATE,
            noise_std=SensorConfig.Barometer.ALTITUDE_NOISE
        )
        self.pressure_noise = 2.0  # Pascal
        
    def _get_measurement(self, vehicle):
        """Get altitude measurement from barometric pressure."""
        # Calculate true pressure at altitude
        true_pressure = self._altitude_to_pressure(vehicle.position[2])
        
        # Add noise to pressure measurement
        measured_pressure = true_pressure + np.random.normal(0, self.pressure_noise)
        
        # Convert back to altitude
        measured_altitude = self._pressure_to_altitude(measured_pressure)
        
        return measured_altitude
        
    def _altitude_to_pressure(self, altitude):
        """Convert altitude to pressure using standard atmosphere model."""
        temperature = STANDARD_TEMPERATURE + TEMPERATURE_LAPSE_RATE * altitude
        power = GRAVITY / (AIR_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
        pressure = STANDARD_PRESSURE * (temperature / STANDARD_TEMPERATURE) ** power
        return pressure
        
    def _pressure_to_altitude(self, pressure):
        """Convert pressure to altitude using standard atmosphere model."""
        power = AIR_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE / GRAVITY
        temperature_ratio = (pressure / STANDARD_PRESSURE) ** power
        altitude = (temperature_ratio * STANDARD_TEMPERATURE - STANDARD_TEMPERATURE) / TEMPERATURE_LAPSE_RATE
        return altitude