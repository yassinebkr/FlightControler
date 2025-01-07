# config/wind_config.py

"""Wind configuration parameters."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from .constants import DEG_TO_RAD

@dataclass
class WindConfig:
    """Wind model configuration parameters."""
    # Layer definitions [height_start, height_end, min_speed, max_speed]
    LAYERS: List[List[float]] = field(default_factory=lambda: [
        [0, 500, 1, 3],      # Surface layer
        [500, 5000, 2, 5],   # Lower atmosphere
        [5000, 10000, 3, 6]  # Upper atmosphere
    ])
    
    # Wind characteristics
    BASE_DIRECTION: np.ndarray = field(
        default_factory=lambda: np.array([-0.7, -0.7, 0], dtype=np.float64)
    )
    
    DIRECTION_CHANGE_RATE: float = 0.003 * DEG_TO_RAD  # rad/s
    
    INTENSITY_VARIATION: Tuple[float, float] = field(
        default_factory=lambda: (0.7, 1.1)  # Intensity multipliers
    )
    
    VARIATION_FREQUENCY: float = 0.005  # Hz
    
    # Thermal settings
    THERMAL_STRENGTH: float = 1.5  # m/s
    THERMAL_RADIUS: float = 75.0  # m
    THERMAL_HEIGHT: float = 1500.0  # m
    THERMAL_FREQUENCY: float = 0.05  # Hz

# Create a global instance for easy access
wind_config = WindConfig()