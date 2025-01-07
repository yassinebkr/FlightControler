# test_simulation.py

import numpy as np
import time
import matplotlib.pyplot as plt
from models.paraglider import Paraglider
from models.environment import Environment
from controllers.navigation import NavigationController
from visualization.visualizer import FlightVisualizer
from config.vehicle_config import VehicleConfig
from config.flight_config import FlightControlConfig, WAYPOINTS
from utils.time_manager import time_manager

class SimulationTester:
    def __init__(self):
        """Initialize simulation components for testing."""
        # Reset time manager
        time_manager.reset()
        
        # Initialize components
        self.environment = Environment()
        self.vehicle = Paraglider(
            position=VehicleConfig.INITIAL_POSITION.copy(),
            velocity=VehicleConfig.INITIAL_VELOCITY.copy(),
            orientation=VehicleConfig.INITIAL_ORIENTATION.copy()
        )
        self.controller = NavigationController()
        self.visualizer = FlightVisualizer()
        
        # Test parameters
        self.dt = 0.05  # 20Hz simulation step
        self.test_duration = 300  # seconds (increased for more comprehensive testing)
        
        # Recording arrays
        self.reset_recording_arrays()
    
    def reset_recording_arrays(self):
        """Reset all recording arrays for a new test."""
        self.positions = []
        self.velocities = []
        self.orientations = []
        self.controls = []
        self.times = []
        self.debug_info = []
        self.ground_contacts = 0
        self.unstable_frames = 0
    
    def run_basic_flight_test(self):
        """Test basic flight dynamics without aggressive control inputs."""
        print("Running basic flight test...")
        
        # Reset recording arrays
        self.reset_recording_arrays()
        
        t = 0
        while t < self.test_duration:
            # Update environment
            self.environment.update(t)
            
            # Record state
            self._record_state(t)
            
            # Update physics
            if not self.vehicle.update(self.dt):
                print("Simulation became unstable. Terminating test.")
                self.unstable_frames += 1
                break
            
            t += self.dt
        
        self._analyze_results("Basic Flight")
    
    def run_controlled_flight_test(self):
        """Test flight with navigation controller active."""
        print("Running controlled flight test...")
        
        # Reset recording arrays
        self.reset_recording_arrays()
        
        # Start visualization
        self.visualizer.start()
        
        t = 0
        try:
            while t < self.test_duration and self.visualizer.active:
                loop_start = time.time()
                
                # Update environment
                self.environment.update(t)
                
                # Get control inputs
                left_brake, right_brake = self.controller.update(self.vehicle)
                self.vehicle.set_control_inputs(left_brake, right_brake)
                
                # Record state
                self._record_state(t, [left_brake, right_brake])
                
                # Update vehicle physics
                if not self.vehicle.update(self.dt):
                    print("Simulation became unstable. Terminating test.")
                    self.unstable_frames += 1
                    break
                
                # Update visualization
                self.visualizer.update(self.vehicle, self.controller)
                
                # Time management
                elapsed = time.time() - loop_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
                
                t += self.dt
                
        except KeyboardInterrupt:
            print("Test interrupted by user")
        except Exception as e:
            print(f"Test error: {e}")
        finally:
            self.visualizer.quit()
        
        self._analyze_results("Controlled Flight")
    
    def _record_state(self, t, controls=None):
        """
        Record current simulation state.
        
        Args:
            t (float): Current simulation time
            controls (list, optional): Control input values
        """
        # Record basic state information
        self.times.append(t)
        self.positions.append(self.vehicle.position.copy())
        self.velocities.append(self.vehicle.velocity.copy())
        self.orientations.append(self.vehicle.orientation.copy())
        
        # Record control inputs if provided
        if controls is not None:
            self.controls.append(controls)
        
        # Record debug information
        state = self.vehicle.get_state()
        self.debug_info.append(state.get('debug', {}))
        
        # Track ground contacts
        if state.get('ground_contact', False):
            self.ground_contacts += 1
    
    def _analyze_results(self, test_name):
        """Analyze and plot test results."""
        if len(self.positions) == 0:
            print("No data collected during simulation.")
            return
        
        # Convert to numpy arrays
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        times = np.array(self.times)
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'{test_name} - Simulation Analysis', fontsize=16)
        
        # 3D Flight Path
        ax1 = fig.add_subplot(221, projection='3d')
        scatter = ax1.scatter(
            positions[:, 0], 
            positions[:, 1], 
            positions[:, 2],
            c=times, 
            cmap='viridis'
        )
        ax1.set_title('Flight Path')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Altitude (m)')
        plt.colorbar(scatter, ax=ax1, label='Time (s)')
        
        # Speed Analysis
        ax2 = fig.add_subplot(222)
        speed = np.linalg.norm(velocities, axis=1)
        ax2.plot(times, speed)
        ax2.set_title('Airspeed')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.grid(True)
        
        # Altitude Tracking
        ax3 = fig.add_subplot(223)
        ax3.plot(times, positions[:, 2])
        ax3.set_title('Altitude Progression')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Altitude (m)')
        ax3.grid(True)
        
        # Control Inputs
        ax4 = fig.add_subplot(224)
        if len(self.controls) > 0:
            controls = np.array(self.controls)
            ax4.plot(times[:len(controls)], controls[:, 0], label='Left Brake')
            ax4.plot(times[:len(controls)], controls[:, 1], label='Right Brake')
        ax4.set_title('Control Inputs')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Brake Position')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{test_name.lower().replace(" ", "_")}_results.png')
        plt.close()
        
        # Comprehensive Performance Analysis
        self._print_performance_summary(test_name, positions, velocities, times)
    
    def _print_performance_summary(self, test_name, positions, velocities, times):
        """
        Generate and print comprehensive performance summary.
        
        Args:
            test_name (str): Name of the test
            positions (np.ndarray): Position data
            velocities (np.ndarray): Velocity data
            times (np.ndarray): Time data
        """
        # Calculate performance metrics
        speed = np.linalg.norm(velocities, axis=1)
        
        # Prepare summary dictionary
        summary = {
            'Test Name': test_name,
            'Total Simulation Time': f"{times[-1]:.2f} s",
            'Flight Duration': f"{times[-1]:.2f} s",
            
            # Position Analysis
            'Start Position': f"{positions[0]}",
            'End Position': f"{positions[-1]}",
            'Total Horizontal Distance': np.sum(np.linalg.norm(np.diff(positions[:, :2]), axis=1)),
            'Total Vertical Distance': np.abs(positions[-1, 2] - positions[0, 2]),
            
            # Speed Analysis
            'Average Speed': f"{np.mean(speed):.2f} m/s",
            'Maximum Speed': f"{np.max(speed):.2f} m/s",
            'Minimum Speed': f"{np.min(speed):.2f} m/s",
            
            # Altitude Analysis
            'Initial Altitude': f"{positions[0, 2]:.2f} m",
            'Final Altitude': f"{positions[-1, 2]:.2f} m",
            'Maximum Altitude': f"{np.max(positions[:, 2]):.2f} m",
            'Minimum Altitude': f"{np.min(positions[:, 2]):.2f} m",
            
            # Stability Metrics
            'Ground Contacts': self.ground_contacts,
            'Unstable Frames': self.unstable_frames
        }
        
        # Print summary
        print("\n--- Performance Summary ---")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Optional: Log summary to file
        self._log_summary(summary)
    
    def _log_summary(self, summary):
        """
        Log performance summary to a text file.
        
        Args:
            summary (dict): Performance summary dictionary
        """
        import os
        from datetime import datetime
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/simulation_summary_{timestamp}.txt"
        
        # Write summary to file
        with open(filename, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Summary logged to {filename}")

def main():
    """Run simulation tests."""
    # Create tester instance
    tester = SimulationTester()
    
    print("\n=== Running Basic Flight Test ===")
    tester.run_basic_flight_test()
    
    print("\n=== Running Controlled Flight Test ===")
    tester = SimulationTester()  # Reset for new test
    tester.run_controlled_flight_test()

if __name__ == "__main__":
    main()