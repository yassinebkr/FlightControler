# test_config.py

import numpy as np
from pathlib import Path
import sys
import importlib

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_configuration():
    """Test and validate all configuration parameters."""
    tests_passed = 0
    tests_failed = 0
    
    print("Testing configuration...")
    
    try:
        # Test general configuration
        from config.general_config import SimulationConfig, WindConfig, VisualizationConfig
        
        print("\nTesting General Configuration:")
        
        # Test simulation parameters
        assert SimulationConfig.DT > 0, "Invalid time step"
        assert SimulationConfig.SIMULATION_TIME > 0, "Invalid simulation time"
        assert SimulationConfig.TIME_SCALING >= 0, "Invalid time scaling"
        print("✓ Simulation parameters valid")
        tests_passed += 1
        
        # Test wind configuration
        assert len(WindConfig.LAYERS) > 0, "No wind layers defined"
        for layer in WindConfig.LAYERS:
            assert len(layer) == 4, "Invalid wind layer format"
            assert layer[0] < layer[1], "Invalid layer heights"
            assert layer[2] <= layer[3], "Invalid wind speed range"
        print("✓ Wind configuration valid")
        tests_passed += 1
        
        # Test visualization configuration
        assert VisualizationConfig.GRID_SIZE > 0, "Invalid grid size"
        assert VisualizationConfig.GRID_SPACING > 0, "Invalid grid spacing"
        assert VisualizationConfig.TRAIL_LENGTH > 0, "Invalid trail length"
        print("✓ Visualization configuration valid")
        tests_passed += 1
        
        # Test flight configuration
        from config.flight_config import FlightControlConfig, WAYPOINTS
        
        print("\nTesting Flight Configuration:")
        
        # Test control parameters
        assert FlightControlConfig.CONTROL_UPDATE_RATE > 0, "Invalid control rate"
        assert FlightControlConfig.MIN_TURN_RADIUS > 0, "Invalid turn radius"
        print("✓ Control parameters valid")
        tests_passed += 1
        
        # Test waypoints
        assert len(WAYPOINTS) > 0, "No waypoints defined"
        for wp in WAYPOINTS:
            assert len(wp) == 2, "Invalid waypoint format"
            pos, mode = wp
            assert isinstance(pos, np.ndarray) and pos.shape == (3,), "Invalid waypoint position"
            assert mode in ["goto", "probe", "land", "flare"], "Invalid waypoint mode"
        print("✓ Waypoint configuration valid")
        tests_passed += 1
        
        # Test vehicle configuration
        from config.vehicle_config import VehicleConfig
        
        print("\nTesting Vehicle Configuration:")
        
        # Test physical parameters
        assert VehicleConfig.MASS > 0, "Invalid mass"
        assert VehicleConfig.WING_AREA > 0, "Invalid wing area"
        assert VehicleConfig.WING_SPAN > 0, "Invalid wing span"
        print("✓ Physical parameters valid")
        tests_passed += 1
        
        # Test performance parameters
        assert VehicleConfig.MIN_SPEED < VehicleConfig.MAX_SPEED, "Invalid speed range"
        assert VehicleConfig.BEST_GLIDE_SPEED > VehicleConfig.MIN_SPEED, "Invalid best glide speed"
        assert VehicleConfig.BEST_GLIDE_RATIO > 0, "Invalid glide ratio"
        print("✓ Performance parameters valid")
        tests_passed += 1
        
        # Test initial conditions
        assert isinstance(VehicleConfig.INITIAL_POSITION, np.ndarray), "Invalid initial position"
        assert isinstance(VehicleConfig.INITIAL_VELOCITY, np.ndarray), "Invalid initial velocity"
        assert isinstance(VehicleConfig.INITIAL_ORIENTATION, np.ndarray), "Invalid initial orientation"
        print("✓ Initial conditions valid")
        tests_passed += 1
        
        print("\nTesting Dependencies:")
        
        # Test required packages
        required_packages = [
            'numpy',
            'scipy',
            'pyvista',
            'matplotlib'
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                print(f"✓ {package} installed")
                tests_passed += 1
            except ImportError:
                print(f"✗ {package} not found")
                tests_failed += 1
        
    except AssertionError as e:
        print(f"✗ Configuration error: {str(e)}")
        tests_failed += 1
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        tests_failed += 1
    
    print(f"\nConfiguration tests complete:")
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    
    return tests_failed == 0

if __name__ == "__main__":
    if test_configuration():
        print("\nAll configuration tests passed - ready to run simulation")
        sys.exit(0)
    else:
        print("\nConfiguration tests failed - please fix errors before running simulation")
        sys.exit(1)
