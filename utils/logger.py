# utils/logger.py

"""
Flight Data Logging System

Manages organized recording of simulation data with advanced 
buffered writing and comprehensive metadata tracking.
"""

import os
import json
import csv
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

class FlightLogger:
    """
    Advanced logging system for capturing and managing 
    comprehensive flight simulation data.
    
    Provides robust mechanisms for:
    - Buffered data writing
    - Metadata logging
    - Performance tracking
    - Flexible data storage
    """
    def log_control(self, control_state: Dict, timestamp: float):
        """
        Log control system state and actions.
        
        Args:
            control_state: Dictionary containing control system state
            timestamp: Current simulation time
        """
        if not control_state:
            return
            
        # Extract control mode and state information
        control_info = control_state.get('control', {})
        safety_info = control_state.get('safety', {})
        performance = control_state.get('performance', {})
        
        # Create record with essential control parameters
        record = [
            timestamp,
            control_info.get('mode', 'normal'),
            control_info.get('left_brake', 0.0),
            control_info.get('right_brake', 0.0),
            safety_info.get('safety_score', 1.0),
            safety_info.get('stability_metric', 1.0),
            performance.get('control_saturations', 0),
            performance.get('mode_transitions', 0)
        ]
        
        # Add record to control buffer
        self.buffers['control'].append(record)
        self._check_buffer('control')
        
        # Log significant events if present
        self._log_control_events(control_state, timestamp)
    
    def _log_control_events(self, control_state: Dict, timestamp: float):
        """
        Log significant control events for analysis.
        
        Args:
            control_state: Dictionary containing control system state
            timestamp: Current simulation time
        """
        # Check for active safety triggers
        safety_info = control_state.get('safety', {})
        triggers = safety_info.get('triggers', {})
        
        for trigger_name, active in triggers.items():
            if active:
                self._log_event(
                    timestamp,
                    'safety_trigger',
                    f"Safety trigger activated: {trigger_name}"
                )
        
        # Log mode transitions
        control_info = control_state.get('control', {})
        current_mode = control_info.get('mode')
        
        if hasattr(self, '_last_mode') and self._last_mode != current_mode:
            self._log_event(
                timestamp,
                'mode_transition',
                f"Control mode changed: {self._last_mode} -> {current_mode}"
            )
        
        self._last_mode = current_mode
    
    def _log_event(self, timestamp: float, event_type: str, description: str):
        """
        Log a specific event with timestamp and description.
        
        Args:
            timestamp: Event timestamp
            event_type: Type of event
            description: Event description
        """
        record = [timestamp, event_type, description]
        self.buffers['events'].append(record)
        self._check_buffer('events')


    def log_environment(self, conditions: Dict, timestamp: float):
        """
        Log environmental conditions.
        
        Args:
            conditions: Dictionary containing environmental conditions
            timestamp: Current simulation time
        """
        if not conditions:
            return
            
        record = [
            timestamp,
            conditions.get('wind', [0, 0, 0])[0],  # wind_x
            conditions.get('wind', [0, 0, 0])[1],  # wind_y
            conditions.get('wind', [0, 0, 0])[2],  # wind_z
            conditions.get('air_density', 0.0),
            conditions.get('temperature', 0.0)
        ]
        
        self.buffers['environment'].append(record)
        self._check_buffer('environment')
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize logging system with configurable output directory.
        
        Args:
            log_dir: Base directory for storing log files
        """
        # Create unique session identifier
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir
        self.session_dir = os.path.join(log_dir, self.session_id)
        self.csv_dir = os.path.join(self.session_dir, "csv")
        
        # Ensure directory structure exists
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Initialize data buffers
        self.buffers = {
            'vehicle_state': [],
            'environment': [],
            'control': [],
            'sensor': [],
            'events': []
        }
        
        # Prepare CSV file handles and writers
        self.csv_files = {}
        self.csv_writers = {}
        self._initialize_csv_files()
        
        # Logging configuration
        self.buffer_size = 1000
        self.last_flush = time.time()
        self.flush_interval = 5.0
        
        # Performance tracking
        self.stats = {
            'total_records': 0,
            'events': 0,
            'last_write': time.time()
        }
    
    def _initialize_csv_files(self):
        """
        Initialize CSV files with structured headers for each data type.
        Ensures consistent and comprehensive data logging.
        """
        headers = {
            'vehicle_state': [
                'timestamp', 'x', 'y', 'z', 
                'vx', 'vy', 'vz', 
                'roll', 'pitch', 'yaw', 
                'left_brake', 'right_brake'
            ],
            'environment': [
                'timestamp', 'wind_x', 'wind_y', 'wind_z',
                'air_density', 'temperature'
            ],
            'control': [
                'timestamp', 'mode', 'heading_error', 
                'track_error', 'altitude_error', 'waypoint_index'
            ],
            'sensor': [
                'timestamp', 'gps_x', 'gps_y', 'gps_z',
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'baro_altitude'
            ],
            'events': [
                'timestamp', 'event_type', 'description', 'value'
            ]
        }
        
        for data_type, header in headers.items():
            file_path = os.path.join(self.csv_dir, f"{data_type}.csv")
            self.csv_files[data_type] = open(file_path, 'w', newline='')
            self.csv_writers[data_type] = csv.writer(self.csv_files[data_type])
            self.csv_writers[data_type].writerow(header)
    
    def log_metadata(self, metadata: Dict):
        """
        Log comprehensive session metadata with robust serialization.
        
        Args:
            metadata: Dictionary containing session metadata
        """
        def convert_numpy(obj):
            """Recursively convert NumPy objects to JSON-serializable types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_metadata = convert_numpy(metadata)
        
        metadata_path = os.path.join(self.session_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=4)
    
    def log_vehicle_state(self, state: Dict, timestamp: float):
        """
        Record comprehensive vehicle state information.
        
        Args:
            state: Vehicle state dictionary
            timestamp: Current simulation timestamp
        """
        record = [
            timestamp,
            state['position'][0], state['position'][1], state['position'][2],
            state['velocity'][0], state['velocity'][1], state['velocity'][2],
            state['orientation'][0], state['orientation'][1], state['orientation'][2],
            state['control']['left_brake'], state['control']['right_brake']
        ]
        
        self.buffers['vehicle_state'].append(record)
        self._check_buffer('vehicle_state')
    
    def close(self):
        """
        Perform comprehensive logging system shutdown.
        Flushes remaining data and closes file handles.
        """
        # Flush any remaining buffered data
        self._flush_all_buffers()
        
        # Close all CSV file handles
        for file_handle in self.csv_files.values():
            file_handle.close()
        
        # Log final session metadata
        final_metadata = {
            'session_id': self.session_id,
            'end_time': datetime.now().isoformat(),
            'statistics': self.get_stats()
        }
        self.log_metadata(final_metadata)
    
    def _check_buffer(self, buffer_type: str):
        """
        Manage data buffer, triggering flush when buffer is full.
        
        Args:
            buffer_type: Type of data buffer to check
        """
        if len(self.buffers[buffer_type]) >= self.buffer_size:
            self._flush_buffer(buffer_type)
        
        current_time = time.time()
        if current_time - self.last_flush >= self.flush_interval:
            self._flush_all_buffers()
            self.last_flush = current_time
    
    def _flush_buffer(self, buffer_type: str):
        """
        Write buffered data to corresponding CSV file.
        
        Args:
            buffer_type: Type of data buffer to flush
        """
        if not self.buffers[buffer_type]:
            return
        
        try:
            self.csv_writers[buffer_type].writerows(self.buffers[buffer_type])
            self.csv_files[buffer_type].flush()
            
            self.stats['total_records'] += len(self.buffers[buffer_type])
            self.stats['last_write'] = time.time()
            
            self.buffers[buffer_type].clear()
            
        except Exception as e:
            print(f"Error flushing {buffer_type} buffer: {str(e)}")
    
    def _flush_all_buffers(self):
        """Comprehensively flush all data buffers to disk."""
        for buffer_type in self.buffers:
            self._flush_buffer(buffer_type)
    
    def get_stats(self) -> Dict:
        """
        Retrieve current logging system performance statistics.
        
        Returns:
            Dictionary of logging performance metrics
        """
        return {
            'session_id': self.session_id,
            'total_records': self.stats['total_records'],
            'events': self.stats['events'],
            'last_write': self.stats['last_write'],
            'buffer_sizes': {k: len(v) for k, v in self.buffers.items()}
        }