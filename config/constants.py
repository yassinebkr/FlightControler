# config/constants.py

"""
Physical constants and unit conversions for paraglider simulation.
All values use SI units unless otherwise specified.
"""

from typing import Final
import numpy as np
from numpy import pi

# Unit conversions
M_TO_KM: Final[float] = 1/1000  # meters to kilometers
KM_TO_M: Final[float] = 1000    # kilometers to meters
DEG_TO_RAD: Final[float] = pi/180  # degrees to radians
RAD_TO_DEG: Final[float] = 180/pi  # radians to degrees

# Physical constants
GRAVITY: Final[float] = 9.81  # m/s²
AIR_GAS_CONSTANT: Final[float] = 287.05  # J/(kg·K)
STANDARD_PRESSURE: Final[float] = 101325  # Pa at sea level
STANDARD_TEMPERATURE: Final[float] = 288.15  # K at sea level
TEMPERATURE_LAPSE_RATE: Final[float] = -0.0065  # K/m
DENSITY_SEA_LEVEL: Final[float] = 1.225  # kg/m³

# Common vectors as numpy arrays
GRAVITY_VECTOR: Final[np.ndarray] = np.array([0, 0, -GRAVITY], dtype=np.float64)
NORTH_VECTOR: Final[np.ndarray] = np.array([0, 1, 0], dtype=np.float64)
EAST_VECTOR: Final[np.ndarray] = np.array([1, 0, 0], dtype=np.float64)
UP_VECTOR: Final[np.ndarray] = np.array([0, 0, 1], dtype=np.float64)

def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """
    Create rotation matrix around Z axis.
    
    Args:
        angle_rad: Rotation angle in radians
    
    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float64)

def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """
    Create rotation matrix around Y axis.
    
    Args:
        angle_rad: Rotation angle in radians
    
    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float64)

def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """
    Create rotation matrix around X axis.
    
    Args:
        angle_rad: Rotation angle in radians
    
    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=np.float64)