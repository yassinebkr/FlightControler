# # models/environment.py

# """
# Environmental model for atmospheric conditions and wind.
# Handles wind layers, thermals, and atmospheric properties.
# """

# import numpy as np
# from typing import Dict, List, Optional
# from config.constants import (
#     STANDARD_PRESSURE,
#     STANDARD_TEMPERATURE,
#     TEMPERATURE_LAPSE_RATE,
#     GRAVITY,
#     AIR_GAS_CONSTANT,
#     DENSITY_SEA_LEVEL
# )
# from config.general_config import sim, vis
# from utils.time_manager import time_manager

# class Environment:
#     """Environmental model managing atmospheric conditions."""
    
#     def __init__(self):
#         """Initialize environment model with comprehensive atmospheric conditions."""
#         # Define base wind configuration
#         self.base_direction = np.array([-0.7, -0.7, 0], dtype=np.float64)
#         self.direction_change_rate = 0.003  # rad/s
#         self.intensity_variation = (0.7, 1.1)  # Intensity multipliers
#         self.variation_frequency = 0.005  # Hz
        
#         # Define thermal parameters
#         self.thermal_strength = 1.5  # m/s
#         self.thermal_radius = 75.0  # m
#         self.thermal_height = 1500.0  # m
#         self.thermal_frequency = 0.05  # Hz
        
#         # Define wind layers [height_start, height_end, min_speed, max_speed]
#         self.wind_layers = [
#             [0, 500, 1, 3],      # Surface layer
#             [500, 5000, 2, 5],   # Lower atmosphere
#             [5000, 10000, 3, 6]  # Upper atmosphere
#         ]
        
#         # Wind state initialization with proper typing
#         self.wind_state: Dict[str, np.ndarray] = {
#             'direction': self._normalize_vector(self.base_direction.copy()),
#             'speed_multiplier': np.array(1.0),
#             'turbulence': np.zeros(3, dtype=np.float64),
#             'rotation_angle': np.array(0.0)
#         }
        
#         # Initialize thermal sources with proper structure
#         self.thermals: List[Dict[str, np.ndarray]] = []
#         self._initialize_thermals()
        
#         # State tracking with proper time initialization
#         self.last_update = time_manager.get_time()
        
#         # Performance optimization: precalculate constants
#         self._precalc_constants()
    
#     def _precalc_constants(self):
#         """Precalculate commonly used atmospheric constants."""
#         self._gravity_term = -GRAVITY / (AIR_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
#         self._density_factor = GRAVITY / (AIR_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
#         self._pressure_factor = (STANDARD_PRESSURE / 
#                                (STANDARD_TEMPERATURE ** self._density_factor))
    
#     def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
#         """
#         Normalize vector with numerical stability enhancements.
        
#         Args:
#             vector: Vector to normalize
            
#         Returns:
#             Normalized vector
#         """
#         norm = np.linalg.norm(vector)
#         if norm > 1e-10:  # Improved numerical stability threshold
#             return vector / norm
#         return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    
#     def _initialize_thermals(self):
#         """Initialize thermal distribution with improved randomization."""
#         num_thermals = np.random.randint(2, 5)
        
#         for _ in range(num_thermals):
#             # Create thermal with vectorized operations
#             position = np.array([
#                 np.random.uniform(-2000, 2000),  # x
#                 np.random.uniform(-2000, 2000),  # y
#                 np.random.uniform(500, 1500)     # z
#             ], dtype=np.float64)
            
#             thermal = {
#                 'position': position,
#                 'strength': np.random.uniform(1.0, self.thermal_strength),
#                 'radius': np.random.uniform(50, self.thermal_radius),
#                 'lifetime': np.random.uniform(60, 120),
#                 'drift_velocity': np.random.normal(0, 0.5, 3)  # Added drift
#             }
            
#             self.thermals.append(thermal)
    
#     def update(self, current_time: float):
#         """
#         Update environmental conditions with time-based changes.
        
#         Args:
#             current_time: Current simulation time
#         """
#         dt = current_time - self.last_update
#         if dt < 0.1:  # Rate limiting
#             return
        
#         # Update wind conditions
#         self._update_wind(current_time, dt)
        
#         # Update thermal dynamics
#         self._update_thermals(dt)
        
#         self.last_update = current_time
    
#     def _update_wind(self, t: float, dt: float):
#         """
#         Update wind conditions with improved physics model.
        
#         Args:
#             t: Current time
#             dt: Time step
#         """
#         # Update rotation with smoother transitions
#         self.wind_state['rotation_angle'] += self.direction_change_rate * dt
        
#         # Create rotation matrix
#         angle = self.wind_state['rotation_angle']
#         cos_angle = np.cos(angle)
#         sin_angle = np.sin(angle)
#         rotation = np.array([
#             [cos_angle, -sin_angle, 0],
#             [sin_angle, cos_angle, 0],
#             [0, 0, 1]
#         ], dtype=np.float64)
        
#         # Update wind direction with smooth interpolation
#         new_direction = rotation @ self.base_direction
#         self.wind_state['direction'] = self._normalize_vector(
#             0.95 * self.wind_state['direction'] + 
#             0.05 * new_direction
#         )
        
#         # Update intensity with smoother variation
#         time_factor = 2 * np.pi * self.variation_frequency * t
#         intensity_range = self.intensity_variation[1] - self.intensity_variation[0]
#         self.wind_state['speed_multiplier'] = (
#             self.intensity_variation[0] +
#             0.5 * intensity_range * (1 + np.sin(time_factor))
#         )
        
#         # Update turbulence with improved model
#         turbulence_scale = 0.3 * self.wind_state['speed_multiplier']
#         new_turbulence = np.random.normal(0, turbulence_scale, 3)
#         self.wind_state['turbulence'] = (
#             0.9 * self.wind_state['turbulence'] +
#             0.1 * new_turbulence
#         )
    
#     def _update_thermals(self, dt: float):
#         """
#         Update thermal sources with improved physics model.
        
#         Args:
#             dt: Time step
#         """
#         # Remove expired thermals
#         self.thermals = [thermal for thermal in self.thermals 
#                         if thermal['lifetime'] > 0]
        
#         # Update existing thermals
#         for thermal in self.thermals:
#             # Reduce lifetime
#             thermal['lifetime'] -= dt
            
#             # Apply drift with boundary conditions
#             new_position = thermal['position'] + thermal['drift_velocity'] * dt
            
#             # Ensure thermals stay within reasonable bounds
#             bounds = np.array([2000, 2000, 1500])
#             new_position = np.clip(new_position, -bounds, bounds)
            
#             # Update position and potentially reflect drift
#             thermal['position'] = new_position
            
#             # Randomly modify drift for natural movement
#             thermal['drift_velocity'] += np.random.normal(0, 0.1, 3) * dt
#             thermal['drift_velocity'] *= 0.98  # Damping
        
#         # Add new thermals with probability based on current count
#         if len(self.thermals) < 5 and np.random.random() < 0.02:
#             self._add_thermal()
    
#     def _add_thermal(self):
#         """Add a new thermal with realistic initialization."""
#         new_thermal = {
#             'position': np.array([
#                 np.random.uniform(-2000, 2000),
#                 np.random.uniform(-2000, 2000),
#                 np.random.uniform(500, 1500)
#             ], dtype=np.float64),
#             'strength': np.random.uniform(1.0, self.thermal_strength),
#             'radius': np.random.uniform(50, self.thermal_radius),
#             'lifetime': np.random.uniform(60, 120),
#             'drift_velocity': np.random.normal(0, 0.5, 3)
#         }
        
#         self.thermals.append(new_thermal)
    
#     def get_conditions(self, position: np.ndarray, t: float) -> Dict:
#         """
#         Get environmental conditions at specific position.
        
#         Args:
#             position: Position vector [x, y, z]
#             t: Current time
            
#         Returns:
#             Environmental conditions including wind, density, etc.
#         """
#         return {
#             'wind': self._get_wind(position),
#             'air_density': self._get_air_density(position[2]),
#             'temperature': self._get_temperature(position[2]),
#             'thermal': self._get_thermal_effect(position)
#         }
    
#     def _get_wind(self, position: np.ndarray) -> np.ndarray:
#         """
#         Calculate wind vector at given position with improved altitude effects.
        
#         Args:
#             position: Position vector
            
#         Returns:
#             Wind vector
#         """
#         altitude = position[2]
        
#         # Get base wind for current layer
#         wind_speed = 0.0
#         for layer_start, layer_end, min_speed, max_speed in self.wind_layers:
#             if layer_start <= altitude < layer_end:
#                 # Smooth interpolation within layer
#                 layer_fraction = (altitude - layer_start) / (layer_end - layer_start)
#                 wind_speed = min_speed + layer_fraction * (max_speed - min_speed)
#                 break
        
#         # Apply direction and intensity with improved combination
#         base_wind = (self.wind_state['direction'] * 
#                     wind_speed * 
#                     self.wind_state['speed_multiplier'])
        
#         # Add turbulence with altitude scaling
#         turbulence_factor = np.exp(-altitude / 5000)  # Reduce turbulence with altitude
#         final_wind = base_wind + self.wind_state['turbulence'] * turbulence_factor
        
#         return final_wind
    
#     def _get_air_density(self, altitude: float) -> float:
#         """
#         Calculate air density using improved atmospheric model.
        
#         Args:
#             altitude: Altitude in meters
            
#         Returns:
#             Air density in kg/m³
#         """
#         # Limit altitude to reasonable range
#         altitude = np.clip(altitude, -500, 20000)
        
#         # Calculate temperature
#         temperature = self._get_temperature(altitude)
        
#         # Calculate pressure with precalculated constants
#         pressure = self._pressure_factor * (temperature ** self._density_factor)
        
#         # Calculate density
#         density = pressure / (AIR_GAS_CONSTANT * temperature)
        
#         return np.clip(density, 0.05, DENSITY_SEA_LEVEL * 1.2)
    
#     def _get_temperature(self, altitude: float) -> float:
#         """
#         Calculate temperature using standard atmosphere model.
        
#         Args:
#             altitude: Altitude in meters
            
#         Returns:
#             Temperature in Kelvin
#         """
#         return STANDARD_TEMPERATURE + TEMPERATURE_LAPSE_RATE * altitude
    
#     def _get_thermal_effect(self, position: np.ndarray) -> Dict:
#         """
#         Calculate thermal effects with improved model.
        
#         Args:
#             position: Position vector
            
#         Returns:
#             Dictionary with thermal effects
#         """
#         total_lift = 0.0
#         contributing_thermals = 0
        
#         for thermal in self.thermals:
#             # Calculate distance to thermal core
#             distance = np.linalg.norm(position - thermal['position'])
            
#             # Apply gaussian thermal model if within radius
#             if distance < thermal['radius']:
#                 # Normalized distance from center (0 to 1)
#                 d = distance / thermal['radius']
                
#                 # Gaussian profile with vertical scaling
#                 vertical_scale = np.exp(-(position[2] - thermal['position'][2])**2 / 
#                                      (2 * (thermal['radius']/2)**2))
                
#                 strength = (thermal['strength'] * 
#                           np.exp(-4 * d * d) * 
#                           vertical_scale)
                
#                 total_lift += strength
#                 contributing_thermals += 1
        
#         return {
#             'vertical_velocity': total_lift,
#             'thermal_count': contributing_thermals,
#             'total_thermals': len(self.thermals)
#         }
# models/environment.py

"""
Environmental model for atmospheric conditions and wind.
Handles wind layers, thermals, and atmospheric properties.
"""

import numpy as np
from typing import Dict, List, Optional
from config.constants import (
    STANDARD_PRESSURE,
    STANDARD_TEMPERATURE,
    TEMPERATURE_LAPSE_RATE,
    GRAVITY,
    AIR_GAS_CONSTANT,
    DENSITY_SEA_LEVEL
)
from config.general_config import sim, vis
from utils.time_manager import time_manager

class Environment:
    """Environmental model managing atmospheric conditions."""
    
    def __init__(self):
        """Initialize environment model with comprehensive atmospheric conditions."""
        # Define base wind configuration with explicit typing
        self.base_direction = np.array([-0.7, -0.7, 0], dtype=np.float64)
        self.direction_change_rate = 0.003  # rad/s
        self.intensity_variation = (0.7, 1.1)  # Intensity multipliers
        self.variation_frequency = 0.005  # Hz
        
        # Define thermal parameters
        self.thermal_strength = 1.5  # m/s
        self.thermal_radius = 75.0  # m
        self.thermal_height = 1500.0  # m
        self.thermal_frequency = 0.05  # Hz
        
        # Define wind layers [height_start, height_end, min_speed, max_speed]
        self.wind_layers = np.array([
            [0, 500, 1, 3],      # Surface layer
            [500, 5000, 2, 5],   # Lower atmosphere
            [5000, 10000, 3, 6]  # Upper atmosphere
        ], dtype=np.float64)
        
        # Wind state initialization with proper typing
        self.wind_state: Dict[str, np.ndarray] = {
            'direction': self._normalize_vector(self.base_direction.copy()),
            'speed_multiplier': np.array(1.0, dtype=np.float64),
            'turbulence': np.zeros(3, dtype=np.float64),
            'rotation_angle': np.array(0.0, dtype=np.float64)
        }
        
        # Initialize thermal sources
        self.thermals: List[Dict[str, np.ndarray]] = []
        self._initialize_thermals()
        
        # State tracking
        self.last_update = time_manager.get_time()
        
        # Precalculate constants for atmospheric model
        self._precalc_constants()
    
    def _precalc_constants(self):
        """Precalculate commonly used atmospheric constants."""
        self._gravity_term = -GRAVITY / (AIR_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
        self._density_factor = GRAVITY / (AIR_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
        self._pressure_factor = (STANDARD_PRESSURE / 
                               (STANDARD_TEMPERATURE ** self._density_factor))
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize vector with numerical stability enhancements.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 1e-10:  # Improved numerical stability threshold
            return vector / norm
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    
    def _initialize_thermals(self):
        """Initialize thermal distribution with improved randomization."""
        num_thermals = np.random.randint(2, 5)
        
        for _ in range(num_thermals):
            position = np.array([
                np.random.uniform(-2000, 2000),  # x
                np.random.uniform(-2000, 2000),  # y
                np.random.uniform(500, 1500)     # z
            ], dtype=np.float64)
            
            thermal = {
                'position': position,
                'strength': np.random.uniform(1.0, self.thermal_strength),
                'radius': np.random.uniform(50, self.thermal_radius),
                'lifetime': np.random.uniform(60, 120),
                'drift_velocity': np.random.normal(0, 0.5, 3)
            }
            
            self.thermals.append(thermal)
    
    def update(self, current_time: float):
        """
        Update environmental conditions with time-based changes.
        
        Args:
            current_time: Current simulation time
        """
        try:
            dt = current_time - self.last_update
            if dt < 0.1:  # Rate limiting
                return
            
            # Update wind conditions
            self._update_wind(current_time, dt)
            
            # Update thermal dynamics
            self._update_thermals(dt)
            
            self.last_update = current_time
            
        except Exception as e:
            print(f"Environment update error: {str(e)}")
    
    def _update_wind(self, t: float, dt: float):
        """
        Update wind conditions with improved physics model.
        
        Args:
            t: Current time
            dt: Time step
        """
        # Update rotation with smoother transitions
        self.wind_state['rotation_angle'] += self.direction_change_rate * dt
        
        # Create rotation matrix
        angle = float(self.wind_state['rotation_angle'])  # Ensure scalar
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Update wind direction with smooth interpolation
        new_direction = rotation @ self.base_direction
        self.wind_state['direction'] = self._normalize_vector(
            0.95 * self.wind_state['direction'] + 
            0.05 * new_direction
        )
        
        # Update intensity with smoother variation
        time_factor = 2 * np.pi * self.variation_frequency * t
        intensity_range = self.intensity_variation[1] - self.intensity_variation[0]
        self.wind_state['speed_multiplier'] = np.array(
            self.intensity_variation[0] +
            0.5 * intensity_range * (1 + np.sin(time_factor)),
            dtype=np.float64
        )
        
        # Update turbulence with improved model and stability
        turbulence_scale = 0.3 * float(self.wind_state['speed_multiplier'])
        new_turbulence = np.random.normal(0, turbulence_scale, 3)
        self.wind_state['turbulence'] = (
            0.9 * self.wind_state['turbulence'] +
            0.1 * new_turbulence
        ).astype(np.float64)
    
    def _update_thermals(self, dt: float):
        """
        Update thermal sources with improved physics model.
        
        Args:
            dt: Time step
        """
        # Remove expired thermals
        self.thermals = [thermal for thermal in self.thermals 
                        if thermal['lifetime'] > 0]
        
        # Update existing thermals
        for thermal in self.thermals:
            # Reduce lifetime
            thermal['lifetime'] -= dt
            
            # Apply drift with boundary conditions
            new_position = thermal['position'] + thermal['drift_velocity'] * dt
            
            # Ensure thermals stay within reasonable bounds
            bounds = np.array([2000, 2000, 1500], dtype=np.float64)
            new_position = np.clip(new_position, -bounds, bounds)
            
            # Update position and potentially reflect drift
            thermal['position'] = new_position
            
            # Randomly modify drift for natural movement
            drift_modification = np.random.normal(0, 0.1, 3) * dt
            thermal['drift_velocity'] = (thermal['drift_velocity'] + drift_modification) * 0.98
        
        # Add new thermals with probability based on current count
        if len(self.thermals) < 5 and np.random.random() < 0.02:
            self._add_thermal()
    
    def _add_thermal(self):
        """Add a new thermal with realistic initialization."""
        new_thermal = {
            'position': np.array([
                np.random.uniform(-2000, 2000),
                np.random.uniform(-2000, 2000),
                np.random.uniform(500, 1500)
            ], dtype=np.float64),
            'strength': np.random.uniform(1.0, self.thermal_strength),
            'radius': np.random.uniform(50, self.thermal_radius),
            'lifetime': np.random.uniform(60, 120),
            'drift_velocity': np.random.normal(0, 0.5, 3)
        }
        
        self.thermals.append(new_thermal)
    
    def get_conditions(self, position: np.ndarray, t: float) -> Dict:
        """
        Get environmental conditions at specific position.
        
        Args:
            position: Position vector [x, y, z]
            t: Current time
            
        Returns:
            Environmental conditions including wind, density, etc.
        """
        try:
            return {
                'wind': self._get_wind(position),
                'air_density': self._get_air_density(position[2]),
                'temperature': self._get_temperature(position[2]),
                'thermal': self._get_thermal_effect(position)
            }
        except Exception as e:
            print(f"Error getting conditions: {str(e)}")
            # Return safe default values
            return {
                'wind': np.zeros(3, dtype=np.float64),
                'air_density': DENSITY_SEA_LEVEL,
                'temperature': STANDARD_TEMPERATURE,
                'thermal': {
                    'vertical_velocity': 0.0,
                    'thermal_count': 0,
                    'total_thermals': 0
                }
            }
    
    def _get_wind(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate wind vector at given position with improved altitude effects.
        
        Args:
            position: Position vector
            
        Returns:
            Wind vector
        """
        altitude = float(position[2])  # Ensure scalar
        
        # Get base wind for current layer
        wind_speed = 0.0
        for layer_start, layer_end, min_speed, max_speed in self.wind_layers:
            if layer_start <= altitude < layer_end:
                # Smooth interpolation within layer
                layer_fraction = (altitude - layer_start) / (layer_end - layer_start)
                wind_speed = min_speed + layer_fraction * (max_speed - min_speed)
                break
        
        # Apply direction and intensity with improved combination
        base_wind = (self.wind_state['direction'] * 
                    wind_speed * 
                    float(self.wind_state['speed_multiplier']))
        
        # Add turbulence with altitude scaling
        turbulence_factor = np.exp(-altitude / 5000)  # Reduce turbulence with altitude
        final_wind = base_wind + self.wind_state['turbulence'] * turbulence_factor
        
        # Ensure output is float64
        return final_wind.astype(np.float64)
    
    def _get_air_density(self, altitude: float) -> float:
        """
        Calculate air density using improved atmospheric model.
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Air density in kg/m³
        """
        # Limit altitude to reasonable range
        altitude = np.clip(altitude, -500, 20000)
        
        # Calculate temperature
        temperature = self._get_temperature(altitude)
        
        # Calculate pressure with precalculated constants
        pressure = self._pressure_factor * (temperature ** self._density_factor)
        
        # Calculate density
        density = pressure / (AIR_GAS_CONSTANT * temperature)
        
        return float(np.clip(density, 0.05, DENSITY_SEA_LEVEL * 1.2))
    
    def _get_temperature(self, altitude: float) -> float:
        """
        Calculate temperature using standard atmosphere model.
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Temperature in Kelvin
        """
        return float(STANDARD_TEMPERATURE + TEMPERATURE_LAPSE_RATE * altitude)
    
    def _get_thermal_effect(self, position: np.ndarray) -> Dict:
        """
        Calculate thermal effects with improved model.
        
        Args:
            position: Position vector
            
        Returns:
            Dictionary with thermal effects
        """
        total_lift = 0.0
        contributing_thermals = 0
        
        for thermal in self.thermals:
            # Calculate distance to thermal core
            distance = np.linalg.norm(position - thermal['position'])
            
            # Apply gaussian thermal model if within radius
            if distance < thermal['radius']:
                # Normalized distance from center (0 to 1)
                d = distance / thermal['radius']
                
                # Gaussian profile with vertical scaling
                vertical_scale = np.exp(-(position[2] - thermal['position'][2])**2 / 
                                     (2 * (thermal['radius']/2)**2))
                
                strength = (thermal['strength'] * 
                          np.exp(-4 * d * d) * 
                          vertical_scale)
                
                total_lift += float(strength)
                contributing_thermals += 1
        
        return {
            'vertical_velocity': total_lift,
            'thermal_count': contributing_thermals,
            'total_thermals': len(self.thermals)
        }