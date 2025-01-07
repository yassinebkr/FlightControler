# config/__init__.py

"""
Configuration package initialization.

This module ensures proper importing and initialization 
of configuration modules for the paraglider simulation system.
"""

from . import constants
from . import general_config
from . import vehicle_config
from . import flight_config

# Expose key configuration instances for easy access
from .general_config import sim, safety, vis
from .vehicle_config import vehicle
from .flight_config import control, sensors, WAYPOINTS

# Validate configurations on import
try:
    from .general_config import validate_configs
    validate_configs()
except Exception as e:
    print(f"Configuration Validation Warning: {str(e)}")

__all__ = [
    'constants',
    'general_config',
    'vehicle_config',
    'flight_config',
    'sim',
    'safety',
    'vis',
    'vehicle',
    'control',
    'sensors',
    'WAYPOINTS'
]