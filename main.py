
    
    # main.py

"""
Paraglider Flight Simulation - Main Execution Script

Manages the entire simulation lifecycle, coordinating initialization, 
execution, and graceful shutdown of the paraglider flight simulation system.
"""

import sys
import os
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# Ensure project root is in Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Core simulation components
from models.paraglider import Paraglider
from models.environment import Environment
from controllers.flight_controller import FlightController
from visualization.visualizer import FlightVisualizer
from utils.logger import FlightLogger
from utils.time_manager import time_manager

# Configuration imports
from config.general_config import sim, safety, vis
from config.vehicle_config import vehicle
from config.flight_config import WAYPOINTS

class SimulationError(Exception):
    """Custom exception for simulation-specific critical errors."""
    pass

class ParagliderSimulation:
    """
    Comprehensive simulation manager responsible for coordinating
    all aspects of the paraglider flight simulation.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize simulation environment with robust configuration management.
        
        Args:
            log_dir: Base directory for logging simulation data
        """
        # Session and logging configuration
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, self.session_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize core simulation state
        self._reset_simulation_state()
        
        # Error tracking
        self.initialization_errors: List[str] = []
        self.runtime_errors: List[str] = []
    
    def _reset_simulation_state(self):
        """Reset all simulation state variables to initial conditions."""
        # Simulation control flags
        self.initialized = False
        self.running = False
        self.paused = False
        
        # Simulation components
        self.vehicle: Optional[Paraglider] = None
        self.environment: Optional[Environment] = None
        self.controller: Optional[FlightController] = None
        self.visualizer: Optional[FlightVisualizer] = None
        self.logger: Optional[FlightLogger] = None
        
        # Performance and tracking
        self.frame_count = 0
        self.start_time = 0.0
        self.last_viz_update = 0.0
        
        # Performance statistics
        self.statistics = {
            'max_speed': 0.0,
            'min_altitude': float('inf'),
            'max_altitude': 0.0,
            'ground_contacts': 0,
            'waypoints_reached': 0,
            'runtime_errors': 0
        }
    
    def initialize(self) -> bool:
        """
        Comprehensive initialization of all simulation components.
        
        Returns:
            bool: Indicates successful initialization
        """
        print("\nInitializing Paraglider Flight Simulation...")
        
        try:
            # Validate configuration parameters
            self._validate_configuration()
            
            # Reset time management
            time_manager.reset()
            
            # Initialize critical simulation components
            self._initialize_components()
            
            # Validate initial simulation state
            if not self._validate_initial_state():
                raise SimulationError("Invalid initial simulation configuration")
            
            self.initialized = True
            self.start_time = time_manager.get_time()
            
            print("✓ Simulation Initialized Successfully")
            return True
        
        except Exception as e:
            self._log_error(f"Initialization Failed: {str(e)}")
            return False
    
    def _validate_configuration(self):
        """Validate and set default configuration parameters."""
        configuration_defaults = [
            ('MAX_POSITION_LIMIT', 50000.0, safety),
            ('MAX_VELOCITY_LIMIT', 50.0, safety),
            ('MAX_ACCELERATION_LIMIT', 20.0, safety),
            ('DEBUG_MODE', False, sim)
        ]
        
        for attr, default_value, config_module in configuration_defaults:
            if not hasattr(config_module, attr):
                setattr(config_module, attr, default_value)
                print(f"Set default {attr}: {default_value}")
    
    def _initialize_components(self):
        """Initialize all critical simulation components."""
        component_initializers = [
            ("Environment", self._initialize_environment),
            ("Vehicle", self._initialize_vehicle),
            ("Flight Controller", self._initialize_flight_controller),
            ("Visualization", self._initialize_visualization),
            ("Logger", self._initialize_logger)
        ]
        
        for name, initializer in component_initializers:
            try:
                print(f"Initializing {name}...")
                initializer()
            except Exception as e:
                error_msg = f"Failed to initialize {name}: {str(e)}"
                self._log_error(error_msg)
                raise SimulationError(error_msg)
    
    def _initialize_environment(self):
        """Initialize atmospheric and environmental model."""
        self.environment = Environment()
    
    def _initialize_vehicle(self):
        """Initialize paraglider vehicle with initial conditions."""
        self.vehicle = Paraglider(
            position=vehicle.INITIAL_POSITION.copy(),
            velocity=vehicle.INITIAL_VELOCITY.copy(),
            orientation=vehicle.INITIAL_ORIENTATION.copy()
        )
    
    def _initialize_flight_controller(self):
        """Initialize autonomous flight control system."""
        self.controller = FlightController()
    
    def _initialize_visualization(self):
        """Initialize visualization if enabled in configuration."""
        if sim.ENABLE_VISUALIZATION:
            self.visualizer = FlightVisualizer()
    
    def _initialize_logger(self):
        """Initialize logging system and record initial state."""
        if sim.ENABLE_LOGGING:
            self.logger = FlightLogger(self.log_dir)
            self._log_initial_state()
    
    def _log_initial_state(self):
        """Log comprehensive initial simulation metadata."""
        if not self.logger:
            return
        
        initial_state = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'simulation': {k: v for k, v in vars(sim).items() if not k.startswith('__')},
                'vehicle': {k: v for k, v in vars(vehicle).items() if not k.startswith('__')},
                'visualization': {k: v for k, v in vars(vis).items() if not k.startswith('__')}
            },
            'initial_conditions': {
                'position': vehicle.INITIAL_POSITION.tolist(),
                'velocity': vehicle.INITIAL_VELOCITY.tolist(),
                'orientation': vehicle.INITIAL_ORIENTATION.tolist()
            },
            'waypoints': [
                {'position': pos.tolist(), 'mode': mode}
                for pos, mode in WAYPOINTS
            ]
        }
        
        self.logger.log_metadata(initial_state)
    
    def _validate_initial_state(self) -> bool:
        """
        Validate critical simulation parameters and initial vehicle state.
        
        Returns:
            bool: Indicates if initial state is valid
        """
        validation_checks = [
            (self.environment is not None, "Environment not initialized"),
            (self.vehicle is not None, "Vehicle not initialized"),
            (self.controller is not None, "Flight controller not initialized"),
            (self._validate_initial_position(), "Invalid initial vehicle position"),
            (self._validate_initial_velocity(), "Invalid initial vehicle velocity"),
            (self._validate_waypoints(), "Invalid waypoint configuration")
        ]
        
        for condition, error_message in validation_checks:
            if not condition:
                self._log_error(error_message)
                return False
        
        return True
    
    def _validate_initial_position(self) -> bool:
        """Validate initial vehicle position."""
        position = self.vehicle.position
        return (isinstance(position, np.ndarray) and 
                position.shape == (3,) and 
                np.all(np.abs(position) < safety.MAX_POSITION_LIMIT))
    
    def _validate_initial_velocity(self) -> bool:
        """Validate initial vehicle velocity."""
        velocity = self.vehicle.velocity
        speed = np.linalg.norm(velocity)
        return (isinstance(velocity, np.ndarray) and 
                velocity.shape == (3,) and 
                speed <= safety.MAX_VELOCITY_LIMIT)
    
    def _validate_waypoints(self) -> bool:
        """Validate waypoint configuration."""
        return len(WAYPOINTS) > 0 and all(
            len(wp) == 2 and 
            isinstance(wp[0], np.ndarray) and 
            wp[0].shape == (3,)
            for wp in WAYPOINTS
        )
    
    def run(self):
        """
        Execute main simulation loop with comprehensive monitoring.
        
        Manages time progression, component updates, 
        performance tracking, and error handling.
        """
        if not self.initialized:
            print("Error: Simulation not initialized. Call initialize() first.")
            return

        print("\nStarting Paraglider Simulation...")
        self.running = True

        try:
            while self.running:
                current_time = time_manager.get_time()
                
                # Check simulation termination conditions
                if current_time >= sim.SIMULATION_TIME:
                    print("\nSimulation time limit reached")
                    break

                # Update simulation state
                if not self._update_simulation():
                    print("\nSimulation update failed")
                    break

                # Update visualization
                if (self.visualizer and 
                    current_time - self.last_viz_update >= 1.0/vis.DISPLAY_UPDATE_RATE):
                    self._update_visualization()

                # Update performance metrics
                self._update_performance_metrics()

                # Increment frame counter
                self.frame_count += 1

                # Manage time scaling
                if sim.TIME_SCALING > 0:
                    self._manage_time_scaling(current_time)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            self._handle_runtime_error(e)
        finally:
            self._finalize_simulation()
    
    def _update_simulation(self) -> bool:
        """Update single simulation time step."""
        try:
            current_time = time_manager.get_time()

            # Update environment
            self.environment.update(current_time)

            # Get environmental conditions
            conditions = self.environment.get_conditions(
                self.vehicle.position,
                current_time
            )

            # Update flight controller
            left_brake, right_brake = self.controller.update(
                self.vehicle,
                conditions
            )

            # Update vehicle controls
            self.vehicle.set_control_inputs(left_brake, right_brake)

            # Update vehicle physics
            if not self.vehicle.update(sim.DT, conditions):
                print("❌ Vehicle physics update failed")
                return False

            # Log data if logging is enabled
            if self.logger:
                self._log_frame_data(current_time, conditions)

            # Update simulation time
            time_manager.update(sim.DT)

            return True

        except Exception as e:
            self._handle_runtime_error(f"Simulation update error: {str(e)}")
            return False
    
    def _update_visualization(self):
        """Update visualization with error handling."""
        try:
            self.visualizer.update(self.vehicle, self.controller)
            self.last_viz_update = time_manager.get_time()
        except Exception as e:
            print(f"Visualization update error: {str(e)}")
            self.visualizer = None
    
    def _update_performance_metrics(self):
        """Track and update simulation performance statistics."""
        speed = np.linalg.norm(self.vehicle.velocity)
        self.statistics['max_speed'] = max(
            self.statistics['max_speed'], speed
        )

        altitude = self.vehicle.position[2]
        self.statistics['min_altitude'] = min(
            self.statistics['min_altitude'], altitude
        )
        self.statistics['max_altitude'] = max(
            self.statistics['max_altitude'], altitude
        )

        if getattr(self.vehicle, 'ground_contact', False):
            self.statistics['ground_contacts'] += 1
    
    def _manage_time_scaling(self, current_time: float):
        """Manage real-time simulation scaling and performance."""
        target_frame_time = sim.DT / sim.TIME_SCALING
        elapsed = time_manager.get_time() - current_time

        if elapsed < target_frame_time:
            time_manager.pause(target_frame_time - elapsed)
    
    def _handle_runtime_error(self, error):
        """Comprehensive runtime error handling and logging."""
        print(f"\nRuntime Error: {str(error)}")
        if sim.DEBUG_MODE:
            traceback.print_exc()

        self.statistics['runtime_errors'] += 1
        self.runtime_errors.append(str(error))
    
    def _finalize_simulation(self):
        """Perform comprehensive simulation shutdown and reporting."""
        print("\nFinalizing Simulation...")

        # Stop visualization
        if self.visualizer:
            self.visualizer.quit()

        # Close logger
        if self.logger:
            self.logger.close()

        # Print final simulation summary
        self._print_simulation_summary()
    
    def _print_simulation_summary(self):
        """Generate and print comprehensive simulation report."""
        print("\nSimulation Summary:")
        print(f"Total Simulation Time: {time_manager.get_time():.2f} seconds")
        print(f"Total Frames: {self.frame_count}")
        print(f"Maximum Speed: {self.statistics['max_speed']:.2f} m/s")
        print(f"Altitude Range: {self.statistics['min_altitude']:.2f} to {self.statistics['max_altitude']:.2f} m")
        print(f"Ground Contacts: {self.statistics['ground_contacts']}")
        print(f"Runtime Errors: {self.statistics['runtime_errors']}")
    
    def _log_frame_data(self, current_time: float, conditions: Dict):
        """Log comprehensive frame data for analysis."""
        if not self.logger:
            return

        # Log vehicle state
        self.logger.log_vehicle_state(
            self.vehicle.get_state(),
            current_time
        )

        # Log environmental conditions
        self.logger.log_environment(
            conditions,
            current_time
        )

        # Log control state
        if self.controller:
            self.logger.log_control(
                self.controller.get_state(),
                current_time
            )
    
    def _log_error(self, message: str):
        """Log simulation errors with timestamp."""
        print(f"❌ ERROR: {message}")
        self.initialization_errors.append(message)

def main():
    """Main entry point for paraglider flight simulation."""
    try:
        simulation = ParagliderSimulation()
        
        if not simulation.initialize():
            print("Simulation initialization failed. Exiting.")
            sys.exit(1)
        
        simulation.run()
    
    except Exception as e:
        print(f"Unhandled simulation error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()