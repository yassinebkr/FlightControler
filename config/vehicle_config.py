# config/vehicle_config.py

"""
Paraglider vehicle configuration parameters.
Defines physical characteristics and flight performance settings.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Final
from .constants import DEG_TO_RAD

@dataclass
class VehicleSettings:
    """Physical and performance parameters for the paraglider."""
    
    # Physical parameters
    MASS: float = 100.0  # kg (includes pilot mass)
    WING_AREA: float = 25.0  # m²
    WING_SPAN: float = 10.0  # m
    
    # Initial conditions
    INITIAL_POSITION: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -1000.0, 1200.0], dtype=np.float64)
    )
    INITIAL_VELOCITY: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 0.0, -1.5], dtype=np.float64)
    )
    INITIAL_ORIENTATION: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    
    # Aerodynamic parameters
    ZERO_LIFT_DRAG: float = 0.02  # Parasitic drag coefficient
    LIFT_SLOPE: float = 5.5  # Lift curve slope
    LIFT_MAX: float = 1.3  # Maximum lift coefficient
    STALL_ANGLE: float = 12 * DEG_TO_RAD  # Stall angle in radians
    INDUCED_DRAG_FACTOR: float = 1.05  # Induced drag factor
    
    # Control parameters
    MAX_BRAKE_DEFLECTION: float = 0.8  # Maximum brake deflection [0-1]
    BRAKE_EFFECTIVENESS: float = 0.7  # Brake effectiveness factor [0-1]
    BRAKE_REACTION_SPEED: float = 1.5  # Brake actuation speed (1/s)
    
    # Flight envelope
    MIN_SPEED: float = 7.0  # Minimum airspeed (m/s)
    MAX_SPEED: float = 14.0  # Maximum airspeed (m/s)
    BEST_GLIDE_SPEED: float = 10.0  # Best glide speed (m/s)
    MIN_SINK_RATE: float = 1.3  # Minimum sink rate (m/s)
    MAX_TURN_RATE: float = 15 * DEG_TO_RAD  # Maximum turn rate (rad/s)
    
    # Performance characteristics
    BEST_GLIDE_RATIO: float = 8.5  # Best glide ratio
    MIN_SINK_SPEED: float = 8.5  # Speed for minimum sink (m/s)
    TRIM_SPEED: float = 9.5  # Natural trim speed (m/s)
    
    # Ground handling
    BOUNCE_DAMPING: float = 0.25  # Ground bounce damping [0-1]
    GROUND_FRICTION: float = 0.25  # Ground friction coefficient [0-1]
    SLIDE_DAMPING: float = 0.7  # Ground slide damping [0-1]
    
    @property
    def ASPECT_RATIO(self) -> float:
        """Calculate wing aspect ratio."""
        return self.WING_SPAN**2 / self.WING_AREA

def validate_vehicle_settings(settings: VehicleSettings) -> bool:
    """
    Validate vehicle configuration parameters.
    
    Args:
        settings: VehicleSettings instance to validate
        
    Returns:
        bool: True if all validations pass
        
    Raises:
        AssertionError: If any validation fails
    """
    # Physical parameters validation
    assert settings.MASS > 0, "Mass must be positive"
    assert settings.WING_AREA > 0, "Wing area must be positive"
    assert settings.WING_SPAN > 0, "Wing span must be positive"
    assert settings.ASPECT_RATIO > 0, "Invalid wing dimensions"
    
    # Initial state validation
    assert settings.INITIAL_POSITION.shape == (3,), "Invalid initial position shape"
    assert settings.INITIAL_VELOCITY.shape == (3,), "Invalid initial velocity shape"
    assert settings.INITIAL_ORIENTATION.shape == (3, 3), "Invalid initial orientation shape"
    assert np.allclose(
        settings.INITIAL_ORIENTATION @ settings.INITIAL_ORIENTATION.T,
        np.eye(3)
    ), "Initial orientation must be orthogonal"
    
    # Aerodynamic parameters validation
    assert settings.ZERO_LIFT_DRAG >= 0, "Invalid drag coefficient"
    assert settings.LIFT_SLOPE > 0, "Invalid lift slope"
    assert settings.LIFT_MAX > 0, "Invalid maximum lift"
    assert settings.STALL_ANGLE > 0, "Invalid stall angle"
    
    # Control parameters validation
    assert 0 <= settings.MAX_BRAKE_DEFLECTION <= 1, "Invalid brake deflection range"
    assert 0 <= settings.BRAKE_EFFECTIVENESS <= 1, "Invalid brake effectiveness range"
    assert settings.BRAKE_REACTION_SPEED > 0, "Invalid brake reaction speed"
    
    # Flight envelope validation
    assert settings.MIN_SPEED > 0, "Invalid minimum speed"
    assert settings.MAX_SPEED > settings.MIN_SPEED, "Invalid speed range"
    assert settings.MIN_SPEED <= settings.BEST_GLIDE_SPEED <= settings.MAX_SPEED, \
        "Best glide speed outside valid range"
    assert settings.MIN_SINK_RATE > 0, "Invalid sink rate"
    assert settings.MAX_TURN_RATE > 0, "Invalid turn rate"
    
    # Ground handling validation
    assert 0 <= settings.BOUNCE_DAMPING <= 1, "Invalid bounce damping"
    assert 0 <= settings.GROUND_FRICTION <= 1, "Invalid ground friction"
    assert 0 <= settings.SLIDE_DAMPING <= 1, "Invalid slide damping"
    
    return True

# Create global instance
vehicle = VehicleSettings()

# Validate on import
validate_vehicle_settings(vehicle)