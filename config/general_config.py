# config/general_config.py

"""
Consolidated simulation settings and parameters.
All core configuration settings are defined here.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .constants import DEG_TO_RAD

@dataclass
class SimSettings:
    """Core simulation settings."""
    # Time settings
    DT: float = 0.05  # Simulation time step (seconds)
    SIMULATION_TIME: float = 600.0  # Total simulation time (seconds)
    TIME_SCALING: float = 1.0  # Real-time scaling factor
    
    # Feature toggles
    ENABLE_VISUALIZATION: bool = True
    MULTITHREADED_DISPLAY: bool = False
    ENABLE_LOGGING: bool = True
    DEBUG_MODE: bool = False
    
    # Update rates
    DISPLAY_UPDATE_RATE: float = 20.0  # Hz
    LOG_RATE: float = 10.0  # Hz
    
    # Paths
    LOG_DIR: str = "logs"

@dataclass
class SafetySettings:
    """Safety limits and thresholds."""
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
    
    # Simulation bounds
    MAX_POSITION_LIMIT: float = 50000.0  # meters
    MAX_VELOCITY_LIMIT: float = 50.0  # m/s
    MAX_ACCELERATION_LIMIT: float = 20.0  # m/s²

@dataclass
class VisSettings:
    """Visualization settings combining traditional and PyVista settings."""
    # Window settings
    WINDOW_SIZE: tuple = (1024, 768)
    DPI: int = 80  # Screen DPI
    DISPLAY_UPDATE_RATE: float = 20.0  # Hz
    
    # View settings
    GRID_SIZE: float = 2000.0  # meters
    GRID_SPACING: float = 500.0  # meters
    TRAIL_LENGTH: int = 100  # points
    
    # Colors
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'background': 'black',
        'ground': 'darkgreen',
        'vehicle': 'white',
        'trail': 'yellow',
        'waypoint': 'red',
        'grid': 'gray'
    })
    
    # Camera settings
    CAMERA_POSITION: np.ndarray = field(
        default_factory=lambda: np.array([-1000, -750, 750], dtype=np.float64)
    )
    CAMERA_OFFSET: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 150], dtype=np.float64)
    )
    CAMERA_ELEVATION: float = 30.0  # degrees
    CAMERA_AZIMUTH: float = 45.0  # degrees
    
    # PyVista specific settings
    ENABLE_LIGHTING: bool = True
    SHOW_AXES: bool = True
    SHOW_EDGES: bool = True
    LINE_WIDTH: float = 1.0
    POINT_SIZE: float = 10.0
    SMOOTH_SHADING: bool = True
    RENDER_POINTS_AS_SPHERES: bool = True
    
    # Camera focus settings
    CAMERA_FOCUS_POINT: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0])
    )
    CAMERA_UP_DIRECTION: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 1])
    )

# Create global instances
sim = SimSettings()
safety = SafetySettings()
vis = VisSettings()

def validate_configs() -> bool:
    """
    Validate all configuration parameters.
    
    Returns:
        bool: True if all validations pass
        
    Raises:
        AssertionError: If any validation fails
    """
    # Simulation settings validation
    assert sim.DT > 0, "Time step must be positive"
    assert sim.SIMULATION_TIME > 0, "Simulation time must be positive"
    assert sim.DISPLAY_UPDATE_RATE > 0, "Display rate must be positive"
    assert sim.LOG_RATE > 0, "Log rate must be positive"
    
    # Safety limits validation
    assert safety.MAX_ALTITUDE > safety.MIN_ALTITUDE, "Invalid altitude range"
    assert safety.MAX_SPEED > safety.MIN_SPEED, "Invalid speed range"
    assert safety.MAX_POSITION_LIMIT > 0, "Position limit must be positive"
    assert safety.MAX_VELOCITY_LIMIT > 0, "Velocity limit must be positive"
    
    # Visualization settings validation
    assert vis.GRID_SIZE > 0, "Grid size must be positive"
    assert vis.GRID_SPACING > 0, "Grid spacing must be positive"
    assert vis.TRAIL_LENGTH > 0, "Trail length must be positive"
    assert all(isinstance(color, str) for color in vis.COLORS.values()), "Invalid color format"
    assert len(vis.WINDOW_SIZE) == 2, "Invalid window size format"
    assert all(x > 0 for x in vis.WINDOW_SIZE), "Window dimensions must be positive"
    
    return True

# Validate configurations on import
validate_configs()