# config/general_config.py

"""
General simulation settings and parameters.
"""

from .constants import *

# Simulation parameters
class SimulationConfig:
    # Time settings
    DT = 0.05  # Simulation time step (seconds)
    SIMULATION_TIME = 600  # Total simulation time (seconds)
    TIME_SCALING = 1.0  # Real-time factor (1.0 = real-time, 0 = as fast as possible)
    
    # Visualization settings
    ENABLE_VISUALIZATION = True
    MULTITHREADED_DISPLAY = True
    DISPLAY_UPDATE_RATE = 50  # Hz
    
    # Logging settings
    ENABLE_LOGGING = True
    LOG_DIR = "logs"
    LOG_RATE = 10  # Hz
    OVERRIDE_LOGS = True
    
    # Debug settings
    DEBUG_MODE = False
    SHOW_WIND = True
    SHOW_FORCES = False
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters"""
        assert cls.DT > 0, "Time step must be positive"
        assert cls.SIMULATION_TIME > 0, "Simulation time must be positive"
        assert cls.DISPLAY_UPDATE_RATE > 0, "Display update rate must be positive"
        assert cls.LOG_RATE > 0, "Log rate must be positive"
        return True

# Wind model parameters
class config:
    # Layer definitions [height_start, height_end, min_speed, max_speed]
    LAYERS = [
        [0, 1000, 2, 5],     # Surface layer: 2-5 m/s
        [1000, 12000, 3, 8],  # Troposphere: 3-8 m/s
        [12000, 20000, 4, 12] # Stratosphere: 4-12 m/s
    ]
    
    # Wind characteristics
    BASE_DIRECTION = np.array([-1, -1, 0])
    DIRECTION_CHANGE_RATE = 0.005 * DEG_TO_RAD  # rad/s
    INTENSITY_VARIATION = [0.8, 1.2]  # Intensity multipliers
    VARIATION_FREQUENCY = 0.01  # Hz
    
    # Thermal/updraft settings
    THERMAL_STRENGTH = 2.0  # m/s
    THERMAL_RADIUS = 100.0  # m
    THERMAL_HEIGHT = 2000.0  # m
    THERMAL_FREQUENCY = 0.1  # Hz

# Visualization settings
class VisualizationConfig:
    # Colors
    COLORS = {
        'background': 'black',
        'background_top': 'navy',
        'ground': 'darkgreen',
        'grid': 'gray',
        'vehicle': 'white',
        'trail': 'yellow',
        'wind': 'cyan',
        'waypoints': 'magenta'
    }
    
    # Display settings
    GRID_SIZE = 10000  # meters
    GRID_SPACING = 500  # meters
    TRAIL_LENGTH = 1000  # points
    
    # Camera settings
    INITIAL_CAMERA_POSITION = np.array([0, -1000, 500])  # meters
    CAMERA_FOCUS_OFFSET = np.array([0, 100, 0])  # meters from vehicle
    
    # Update rates
    MIN_UPDATE_INTERVAL = 0.02  # seconds (50 Hz max)

# Initialize simulation variables
T = 0  # Current simulation time
DT = SimulationConfig.DT
DT_SAVE = DT  # For pause functionality

# Validate configuration
SimulationConfig.validate()