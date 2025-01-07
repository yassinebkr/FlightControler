# # utils/data_analyzer.py

# """
# Analysis and visualization tools for flight data logs.
# """

# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime

# class FlightDataAnalyzer:
#     """Tools for analyzing and visualizing flight data."""
    
#     def __init__(self, session_dir):
#         """
#         Initialize analyzer with a session directory.
        
#         Args:
#             session_dir (str): Path to session directory containing logs
#         """
#         self.session_dir = session_dir
#         self.metadata = self._load_metadata()
#         self.data = {
#             'vehicle_state': None,
#             'sensor_data': None,
#             'control_inputs': None,
#             'environment': None,
#             'events': None
#         }
        
#     def _load_metadata(self):
#         """Load session metadata."""
#         metadata_file = os.path.join(self.session_dir, "metadata.json")
#         with open(metadata_file, 'r') as f:
#             return json.load(f)
            
#     def load_data(self, data_types=None):
#         """
#         Load specified data types from log files.
        
#         Args:
#             data_types (list, optional): List of data types to load. Loads all if None.
#         """
#         if data_types is None:
#             data_types = self.data.keys()
            
#         for data_type in data_types:
#             # Load and concatenate CSV chunks
#             chunks = []
#             for i in range(self.metadata['chunks']):
#                 filename = os.path.join(
#                     self.session_dir, 
#                     "csv", 
#                     f"{data_type}_chunk_{i}.csv"
#                 )
#                 if os.path.exists(filename):
#                     chunk = pd.read_csv(filename)
#                     chunks.append(chunk)
                    
#             if chunks:
#                 self.data[data_type] = pd.concat(chunks, ignore_index=True)
                
#     def analyze_trajectory(self):
#         """Analyze flight trajectory and generate statistics."""
#         if self.data['vehicle_state'] is None:
#             self.load_data(['vehicle_state'])
            
#         df = self.data['vehicle_state']
        
#         # Calculate basic statistics
#         stats = {
#             'flight_duration': df['timestamp'].max() - df['timestamp'].min(),
#             'max_altitude': df['z'].max(),
#             'max_speed': np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2).max(),
#             'total_distance': np.sum(np.sqrt(
#                 np.diff(df['x'])**2 + 
#                 np.diff(df['y'])**2 + 
#                 np.diff(df['z'])**2
#             )),
#             'average_glide_ratio': abs(
#                 np.sum(np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2)) /
#                 np.sum(np.diff(df['z']))
#             )
#         }
        
#         return stats
        
#     def analyze_control_performance(self):
#         """Analyze control system performance."""
#         if self.data['control_inputs'] is None:
#             self.load_data(['control_inputs'])
            
#         df = self.data['control_inputs']
        
#         # Calculate control statistics
#         stats = {
#             'average_left_brake': df['left_brake'].mean(),
#             'average_right_brake': df['right_brake'].mean(),
#             'brake_activity': {
#                 'left': df['left_brake'].std(),
#                 # utils/data_analyzer.py (continued)

#                 'left': df['left_brake'].std(),
#                 'right': df['right_brake'].std()
#             },
#             'mode_transitions': df['mode'].value_counts().to_dict(),
#             'waypoints_reached': df['waypoint_index'].max()
#         }
        
#         # Calculate control response times
#         control_changes = df[abs(df['left_brake'].diff()) + 
#                            abs(df['right_brake'].diff()) > 0.1]
#         if not control_changes.empty:
#             stats['average_response_time'] = np.mean(
#                 np.diff(control_changes['timestamp']))
        
#         return stats
        
#     def analyze_wind_effects(self):
#         """Analyze wind effects on flight performance."""
#         if self.data['environment'] is None:
#             self.load_data(['environment'])
            
#         df = self.data['environment']
        
#         # Calculate wind statistics
#         wind_speed = np.sqrt(df['wind_x']**2 + df['wind_y']**2 + df['wind_z']**2)
#         stats = {
#             'average_wind_speed': wind_speed.mean(),
#             'max_wind_speed': wind_speed.max(),
#             'wind_direction_variability': np.std(
#                 np.arctan2(df['wind_y'], df['wind_x'])),
#             'wind_conditions': {
#                 'calm': np.mean(wind_speed < 3),
#                 'moderate': np.mean((wind_speed >= 3) & (wind_speed < 7)),
#                 'strong': np.mean(wind_speed >= 7)
#             }
#         }
        
#         return stats
        
#     def plot_trajectory_3d(self):
#         """Create 3D plot of flight trajectory."""
#         if self.data['vehicle_state'] is None:
#             self.load_data(['vehicle_state'])
            
#         df = self.data['vehicle_state']
        
#         fig = plt.figure(figsize=(12, 8))
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Plot trajectory
#         scatter = ax.scatter(df['x'], df['y'], df['z'],
#                            c=df['timestamp'] - df['timestamp'].min(),
#                            cmap='viridis',
#                            s=10)
        
#         # Add waypoints if available
#         if self.data['control_inputs'] is not None:
#             waypoints = self.data['control_inputs'].groupby('waypoint_index').first()
#             ax.scatter(waypoints['estimated_x'],
#                       waypoints['estimated_y'],
#                       waypoints['estimated_z'],
#                       color='red',
#                       s=100,
#                       marker='*',
#                       label='Waypoints')
        
#         # Customize plot
#         ax.set_xlabel('X (m)')
#         ax.set_ylabel('Y (m)')
#         ax.set_zlabel('Altitude (m)')
#         ax.set_title('Flight Trajectory')
        
#         # Add colorbar
#         cbar = plt.colorbar(scatter)
#         cbar.set_label('Time (s)')
        
#         plt.legend()
#         return fig
        
#     def plot_control_inputs(self):
#         """Plot control inputs over time."""
#         if self.data['control_inputs'] is None:
#             self.load_data(['control_inputs'])
            
#         df = self.data['control_inputs']
        
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
#         time = df['timestamp'] - df['timestamp'].min()
        
#         # Plot brake commands
#         ax1.plot(time, df['left_brake'], label='Left Brake')
#         ax1.plot(time, df['right_brake'], label='Right Brake')
#         ax1.set_ylabel('Brake Position')
#         ax1.set_title('Control Inputs')
#         ax1.legend()
#         ax1.grid(True)
        
#         # Plot flight mode
#         mode_numeric = pd.Categorical(df['mode']).codes
#         ax2.plot(time, mode_numeric, 'k-')
#         ax2.set_ylabel('Flight Mode')
#         ax2.set_xlabel('Time (s)')
        
#         # Add mode labels
#         unique_modes = df['mode'].unique()
#         ax2.set_yticks(range(len(unique_modes)))
#         ax2.set_yticklabels(unique_modes)
#         ax2.grid(True)
        
#         plt.tight_layout()
#         return fig
        
#     def plot_wind_analysis(self):
#         """Plot wind conditions during flight."""
#         if self.data['environment'] is None:
#             self.load_data(['environment'])
            
#         df = self.data['environment']
        
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
#         time = df['timestamp'] - df['timestamp'].min()
        
#         # Plot wind speed
#         wind_speed = np.sqrt(df['wind_x']**2 + df['wind_y']**2 + df['wind_z']**2)
#         ax1.plot(time, wind_speed, label='Wind Speed')
#         ax1.set_ylabel('Wind Speed (m/s)')
#         ax1.set_title('Wind Conditions')
#         ax1.legend()
#         ax1.grid(True)
        
#         # Plot wind direction
#         wind_dir = np.arctan2(df['wind_y'], df['wind_x']) * 180 / np.pi
#         ax2.plot(time, wind_dir, label='Wind Direction')
#         ax2.set_ylabel('Wind Direction (deg)')
#         ax2.set_xlabel('Time (s)')
#         ax2.legend()
#         ax2.grid(True)
        
#         plt.tight_layout()
#         return fig
        
#     def generate_flight_report(self, output_dir):
#         """Generate comprehensive flight report with analysis and plots."""
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Gather all analysis
#         trajectory_stats = self.analyze_trajectory()
#         control_stats = self.analyze_control_performance()
#         wind_stats = self.analyze_wind_effects()
        
#         # Generate plots and save them
#         plots = {
#             'trajectory': self.plot_trajectory_3d(),
#             'control': self.plot_control_inputs(),
#             'wind': self.plot_wind_analysis()
#         }
        
#         for name, fig in plots.items():
#             fig.savefig(os.path.join(output_dir, f'{name}.png'))
#             plt.close(fig)
        
#         # Create report dictionary
#         report = {
#             'session_id': self.metadata['session_id'],
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'flight_statistics': trajectory_stats,
#             'control_analysis': control_stats,
#             'wind_analysis': wind_stats
#         }
        
#         # Save report as JSON
#         report_file = os.path.join(output_dir, 'flight_report.json')
#         with open(report_file, 'w') as f:
#             json.dump(report, f, indent=4)
        
#         return report

#     def export_to_csv(self, output_dir):
#         """Export all data to CSV files."""
#         os.makedirs(output_dir, exist_ok=True)
        
#         for data_type, df in self.data.items():
#             if df is not None:
#                 output_file = os.path.join(output_dir, f'{data_type}.csv')
#                 df.to_csv(output_file, index=False)
# utils/data_analyzer.py

"""
Advanced Data Analysis and Visualization Toolkit for Flight Simulation

Provides comprehensive tools for extracting insights, generating 
performance reports, and visualizing simulation data.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional, List

class FlightDataAnalyzer:
    """
    Comprehensive flight data analysis system with multi-dimensional 
    processing and visualization capabilities.
    
    Key Capabilities:
    - Metadata extraction
    - Performance metrics calculation
    - Advanced data visualization
    - Detailed flight report generation
    """
    
    def __init__(self, session_dir: str):
        """
        Initialize analyzer for specific simulation session.
        
        Args:
            session_dir: Path to simulation session log directory
        """
        self.session_dir = session_dir
        self.metadata = self._load_metadata()
        
        # Initialize data storage
        self.data_sources = {
            'vehicle_state': None,
            'environment': None,
            'control': None,
            'sensor': None,
            'events': None
        }
    
    def _load_metadata(self) -> Dict:
        """
        Load comprehensive session metadata.
        
        Returns:
            Dict containing session configuration and parameters
        """
        metadata_path = os.path.join(self.session_dir, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Metadata loading error: {e}")
            return {}
    
    def load_data(self, data_types: Optional[List[str]] = None):
        """
        Load simulation data from CSV files.
        
        Args:
            data_types: List of data types to load (default: all)
        """
        if data_types is None:
            data_types = list(self.data_sources.keys())
        
        for data_type in data_types:
            csv_path = os.path.join(self.session_dir, 'csv', f"{data_type}.csv")
            try:
                self.data_sources[data_type] = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error loading {data_type} data: {e}")
    
    def analyze_trajectory(self) -> Dict:
        """
        Perform comprehensive trajectory analysis.
        
        Returns:
            Dict with detailed flight trajectory metrics
        """
        if self.data_sources['vehicle_state'] is None:
            self.load_data(['vehicle_state'])
        
        df = self.data_sources['vehicle_state']
        
        # Calculate trajectory metrics
        trajectory_metrics = {
            'flight_duration': df['timestamp'].max() - df['timestamp'].min(),
            'max_altitude': df['z'].max(),
            'min_altitude': df['z'].min(),
            'max_speed': np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2).max(),
            'total_distance': np.sum(np.sqrt(
                np.diff(df['x'])**2 + 
                np.diff(df['y'])**2 + 
                np.diff(df['z'])**2
            )),
            'average_glide_ratio': abs(
                np.sum(np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2)) /
                np.sum(np.diff(df['z']))
            )
        }
        
        return trajectory_metrics
    
    def generate_flight_report(self, output_dir: Optional[str] = None) -> Dict:
        """
        Generate comprehensive flight performance report.
        
        Args:
            output_dir: Directory to save report and visualizations
        
        Returns:
            Dict containing detailed flight analysis
        """
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(self.session_dir, 'report')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all data sources
        self.load_data()
        
        # Compute comprehensive report
        report = {
            'metadata': self.metadata,
            'trajectory': self.analyze_trajectory(),
            'visualizations': {}
        }
        
        # Generate and save visualizations
        visualization_methods = [
            self._plot_trajectory_3d,
            self._plot_altitude_profile,
            self._plot_speed_profile
        ]
        
        for plot_func in visualization_methods:
            try:
                fig = plot_func()
                plot_name = plot_func.__name__.replace('_plot_', '')
                fig.savefig(os.path.join(output_dir, f"{plot_name}_plot.png"))
                plt.close(fig)
            except Exception as e:
                print(f"Error generating {plot_name} plot: {e}")
        
        return report
    
    def _plot_trajectory_3d(self) -> plt.Figure:
        """Generate 3D trajectory plot."""
        df = self.data_sources['vehicle_state']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            df['x'], df['y'], df['z'], 
            c=df['timestamp'], 
            cmap='viridis'
        )
        
        ax.set_title('Flight Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Altitude (m)')
        
        plt.colorbar(scatter, label='Time (s)')
        return fig
    
    def _plot_altitude_profile(self) -> plt.Figure:
        """Generate altitude profile plot."""
        df = self.data_sources['vehicle_state']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['timestamp'], df['z'])
        
        ax.set_title('Altitude Profile')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        
        return fig
    
    def _plot_speed_profile(self) -> plt.Figure:
        """Generate speed profile plot."""
        df = self.data_sources['vehicle_state']
        
        # Calculate total speed
        df['total_speed'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['timestamp'], df['total_speed'])
        
        ax.set_title('Speed Profile')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        
        return fig

# Optional: Add a main execution block for standalone testing
def main():
    """
    Standalone execution for testing data analysis capabilities.
    """
    try:
        # Specify a recent simulation session directory
        session_dirs = sorted([
            os.path.join('logs', d) 
            for d in os.listdir('logs') 
            if os.path.isdir(os.path.join('logs', d))
        ])
        
        if not session_dirs:
            print("No simulation sessions found.")
            return
        
        latest_session = session_dirs[-1]
        analyzer = FlightDataAnalyzer(latest_session)
        
        # Generate comprehensive report
        report = analyzer.generate_flight_report()
        
        # Print key metrics
        print("\n--- Flight Performance Summary ---")
        print(json.dumps(report['trajectory'], indent=2))
    
    except Exception as e:
        print(f"Analysis error: {e}")

if __name__ == "__main__":
    main()