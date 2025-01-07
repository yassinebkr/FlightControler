# utils/time_manager.py

"""
Advanced Time Management Utility for Paraglider Flight Simulation

Provides comprehensive time tracking and manipulation capabilities 
to support precise simulation timing and performance monitoring.
"""

import time as system_time
import numpy as np

class TimeManager:
    """
    Sophisticated time management system for simulation control.
    
    Responsibilities:
    - Track simulation time
    - Manage time scaling
    - Monitor performance metrics
    - Provide precise time control mechanisms
    """
    
    def __init__(self):
        """
        Initialize time management system with comprehensive state tracking.
        
        Configures internal state for accurate and flexible time management
        across simulation environments.
        """
        # Reset to initial state
        self.reset()
        
        # Performance and stability parameters
        self.time_dilation_factor = 1.0
        self.max_time_step = 0.1  # Maximum allowed time step
        self.min_time_step = 0.001  # Minimum allowed time step
        
        # Detailed performance metrics
        self.performance_metrics = {
            'total_updates': 0,
            'cumulative_elapsed_time': 0.0,
            'maximum_time_step': 0.0,
            'paused_duration': 0.0
        }
    
    def reset(self):
        """
        Comprehensively reset time manager to initial state.
        
        Ensures clean initialization for new simulation scenarios
        with precise time tracking mechanisms.
        """
        # Simulation time tracking
        self.sim_time = 0.0
        self.start_wall_time = system_time.time()
        self.last_update_time = system_time.time()
        
        # State management flags
        self.paused = False
        self.frozen = False
        
        # Default time step configuration
        self.default_time_step = 0.05
    
    def update(self, dt=None):
        """
        Update simulation time with enhanced stability controls.
        
        Args:
            dt (float, optional): Specific time step for update
        
        Returns:
            float: Updated simulation time
        """
        # Use default time step if not specified
        time_step = dt if dt is not None else self.default_time_step
        
        # Implement robust time step validation
        validated_time_step = float(np.clip(
            time_step, 
            self.min_time_step, 
            self.max_time_step
        ))
        
        # Check simulation state constraints
        if self.paused or self.frozen:
            return self.sim_time
        
        # Apply time dilation mechanism
        adjusted_time_step = validated_time_step * self.time_dilation_factor
        
        # Update simulation time
        self.sim_time += adjusted_time_step
        
        # Update performance tracking
        self.performance_metrics['total_updates'] += 1
        self.performance_metrics['cumulative_elapsed_time'] += adjusted_time_step
        self.performance_metrics['maximum_time_step'] = max(
            self.performance_metrics['maximum_time_step'], 
            adjusted_time_step
        )
        
        return self.sim_time
    
    def get_time(self):
        """
        Retrieve current simulation time.
        
        Returns:
            float: Current simulation time
        """
        return self.sim_time
    
    def pause(self, duration=None):
        """
        Pause simulation or apply controlled time suspension.
        
        Args:
            duration (float, optional): Specific pause duration
        """
        pause_start = system_time.time()
        
        if duration is not None:
            # Precise sleep mechanism
            system_time.sleep(duration)
            self.performance_metrics['paused_duration'] += duration
        else:
            # Set paused state
            self.paused = True
    
    def resume(self):
        """
        Resume simulation from paused state.
        """
        self.paused = False
        self.last_update_time = system_time.time()
    
    def set_time_dilation(self, factor):
        """
        Configure time scaling factor.
        
        Args:
            factor (float): Time dilation multiplier
        """
        # Constrain time dilation to reasonable range
        self.time_dilation_factor = float(np.clip(factor, 0.1, 10.0))
    
    def get_performance_summary(self):
        """
        Generate comprehensive time management performance report.
        
        Returns:
            dict: Detailed performance metrics
        """
        return {
            'total_simulation_time': self.sim_time,
            'total_updates': self.performance_metrics['total_updates'],
            'average_time_step': (
                self.performance_metrics['cumulative_elapsed_time'] / 
                max(self.performance_metrics['total_updates'], 1)
            ),
            'maximum_time_step': self.performance_metrics['maximum_time_step'],
            'total_paused_time': self.performance_metrics['paused_duration']
        }

# Create global time management instance
time_manager = TimeManager()