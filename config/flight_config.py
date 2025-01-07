# config/flight_config.py

"""
Flight control and navigation configuration.
Includes controller parameters, safety limits, and waypoint definitions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Literal
from .constants import DEG_TO_RAD

# Type definitions
WaypointMode = Literal["goto", "probe", "land", "flare"]
Waypoint = Tuple[np.ndarray, WaypointMode]

@dataclass
class FlightControlSettings:
    """Comprehensive flight control system parameters."""
    # Update frequencies
    CONTROL_UPDATE_RATE: float = 5.0  # Hz
    SENSOR_UPDATE_RATE: float = 50.0  # Hz
    
    # Navigation parameters
    MIN_TURN_RADIUS: float = 75.0  # meters
    SAFETY_MARGIN: float = 1.3  # safety factor
    LANDING_PATTERN_ALTITUDE: float = 250.0  # meters
    FINAL_LEG_DISTANCE: float = 150.0  # meters
    
    # Control gains
    HEADING_P_GAIN: float = 0.2
    HEADING_D_GAIN: float = 0.05
    ALTITUDE_P_GAIN: float = 0.15
    
    # Landing parameters
    FLARE_HEIGHT: float = 5.0  # meters
    FLARE_BRAKE_VALUE: float = 0.6  # [0-1]
    APPROACH_SPEED: float = 8.5  # m/s
    
    # Navigation parameters
    WAYPOINT_RADIUS: float = 75.0  # meters
    FINAL_WAYPOINT_RADIUS: float = 25.0  # meters
    
    # Wind estimation
    WIND_ESTIMATION_WINDOW: float = 3.0  # seconds
    MIN_AIRSPEED_FOR_ESTIMATION: float = 6.0  # m/s
    
    # Safety parameters
    MAX_WAYPOINT_PROGRESSION_TIME: float = 180.0  # seconds
    EMERGENCY_DESCENT_RATE: float = 2.0  # m/s
    SAFE_GROUND_CONTACT_VELOCITY: float = 3.0  # m/s

@dataclass
class SafetySettings:
    """Comprehensive safety limits and thresholds."""
    # Altitude limits
    MIN_ALTITUDE: float = 100.0  # meters
    MAX_ALTITUDE: float = 3000.0  # meters
    
    # Speed limits
    MIN_SPEED: float = 6.0  # m/s
    MAX_SPEED: float = 15.0  # m/s
    MAX_BANK_ANGLE: float = 40 * DEG_TO_RAD  # radians
    MAX_PITCH_ANGLE: float = 25 * DEG_TO_RAD  # radians
    
    # Emergency thresholds
    COLLISION_DISTANCE: float = 30.0  # meters
    LOW_ALTITUDE_WARNING: float = 150.0  # meters
    
    # Flight envelope limits
    MAX_LOAD_FACTOR: float = 2.5  # g
    MIN_LOAD_FACTOR: float = 0.0  # g
    MAX_DESCENT_RATE: float = 8.0  # m/s
    
    # Control limits
    MAX_CONTROL_RATE: float = 1.0  # units/second
    MAX_CONTROL_DEFLECTION: float = 1.0  # normalized

@dataclass
class SensorSettings:
    """Comprehensive sensor configuration parameters."""
    
    @dataclass
    class GPS:
        UPDATE_RATE: float = 5.0  # Hz
        POSITION_NOISE: float = 2.5  # meters
        VELOCITY_NOISE: float = 0.15  # m/s
    
    @dataclass
    class IMU:
        UPDATE_RATE: float = 50.0  # Hz
        ACCELEROMETER_NOISE: float = 0.15  # m/s²
        GYROSCOPE_NOISE: float = 0.015  # rad/s
    
    @dataclass
    class Barometer:
        UPDATE_RATE: float = 25.0  # Hz
        ALTITUDE_NOISE: float = 0.7  # meters
    
    @dataclass
    class Magnetometer:
        UPDATE_RATE: float = 25.0  # Hz
        HEADING_NOISE: float = 2.5 * DEG_TO_RAD  # radians
    
    gps = GPS()
    imu = IMU()
    baro = Barometer()
    mag = Magnetometer()

# Flight path waypoints
# Format: (position, mode)
WAYPOINTS: List[Waypoint] = [
    (np.array([0.0, -800.0, 1200.0], dtype=np.float64), "goto"),   # Initial approach
    (np.array([0.0, -400.0, 900.0], dtype=np.float64), "goto"),    # Intermediate point
    (np.array([0.0, -200.0, 600.0], dtype=np.float64), "goto"),    # Descending point
    (np.array([0.0, -50.0, 250.0], dtype=np.float64), "land"),     # Landing approach
    (np.array([0.0, 0.0, 1.0], dtype=np.float64), "flare"),        # Final flare
]

# Create global instances
control = FlightControlSettings()
safety = SafetySettings()
sensors = SensorSettings()