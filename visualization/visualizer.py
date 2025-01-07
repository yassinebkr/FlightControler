# # visualization/visualizer.py

# """
# Flight visualization using PyVista.
# Provides real-time 3D visualization of the paraglider simulation.
# """

# import numpy as np
# import pyvista as pv
# from typing import List, Optional, Dict, Tuple
# import threading
# import time
# from pathlib import Path
# from config.general_config import vis, sim
# from config.flight_config import WAYPOINTS
# from utils.time_manager import time_manager

# class FlightVisualizer:
#     """Real-time 3D visualization system using PyVista."""
    
#     def __init__(self):
#         """Initialize visualization system with PyVista setup."""
#         # Core state
#         self.active = True
#         self.last_update = time_manager.get_time()
#         self.update_rate = getattr(vis, 'DISPLAY_UPDATE_RATE', 20.0)
        
#         # Initialize visualization components
#         self._init_state()
#         self._configure_pyvista()
#         self._init_plotter()
#         self._setup_scene()
        
#         # Start visualization
#         if getattr(sim, 'MULTITHREADED_DISPLAY', False):
#             self._start_render_thread()
#         else:
#             self.plotter.show(interactive_update=True)
    
#     def _init_state(self):
#         """Initialize visualization state tracking."""
#         self.trail_points: List[np.ndarray] = []
#         self.max_trail_points = getattr(vis, 'TRAIL_LENGTH', 100)
#         self.waypoint_actors: List[pv.Actor] = []
#         self.render_times: List[float] = []
#         self.max_render_times = 100
        
#         # Scene actors
#         self.vehicle_actor = None
#         self.trail_actor = None
#         self.wind_actor = None
#         self.ground_actor = None
#         self.axes_actor = None
        
#         # Thread control
#         self.render_thread = None
#         self.render_lock = threading.Lock()
    
#     def _configure_pyvista(self):
#         """Configure PyVista global settings."""
#         bg_color = getattr(vis, 'BACKGROUND_COLOR', 'black')
#         pv.global_theme.background = bg_color
#         pv.global_theme.show_edges = True
#         pv.global_theme.edge_color = 'white'
#         pv.global_theme.font.size = 12
#         pv.global_theme.font.family = 'arial'
#         pv.global_theme.show_scalar_bar = False
    
#     def _init_plotter(self):
#         """Initialize PyVista plotter."""
#         window_size = getattr(vis, 'WINDOW_SIZE', (1024, 768))
#         self.plotter = pv.Plotter(
#             window_size=window_size,
#             off_screen=getattr(sim, 'MULTITHREADED_DISPLAY', False)
#         )
        
#         # Set up camera
#         camera_pos = getattr(vis, 'CAMERA_POSITION', [-1000, -750, 750])
#         focus_point = getattr(vis, 'CAMERA_FOCUS_POINT', [0, 0, 0])
#         up_direction = getattr(vis, 'CAMERA_UP_DIRECTION', [0, 0, 1])
        
#         self.plotter.camera_position = [
#             camera_pos,
#             focus_point,
#             up_direction
#         ]
        
#         self._setup_interactions()
#         self._configure_rendering()
    
#     def _configure_rendering(self):
#         """Configure rendering features."""
#         # Basic rendering features
#         self.plotter.enable_anti_aliasing()
        
#         # Configure lighting using universally supported methods
#         if getattr(vis, 'ENABLE_LIGHTING', True):
#             try:
#                 # Modern PyVista uses lighting manager
#                 if hasattr(self.plotter, 'enable_lightkit'):
#                     self.plotter.enable_lightkit()
#                 # Fallback to basic lighting setup
#                 elif hasattr(self.plotter, 'enable_eye_dome_lighting'):
#                     self.plotter.enable_eye_dome_lighting()
#             except Exception:
#                 pass  # Silently handle unsupported lighting features
        
#         # Enable depth features if available
#         try:
#             self.plotter.enable_depth_peeling()
#         except Exception:
#             pass  # Silently handle if depth peeling is unsupported
#     def _setup_interactions(self):
#         """Configure interaction handlers."""
#         self.plotter.add_key_event('q', self.quit)
#         self.plotter.add_key_event('r', self._reset_camera)
#         self.plotter.add_key_event('g', self._toggle_grid)
#         self.plotter.add_key_event('c', self._cycle_camera)
#         self.plotter.enable_trackball_style()
    
#     def _setup_scene(self):
#         """Set up initial visualization scene."""
#         with self.render_lock:
#             self._add_ground_grid()
#             if getattr(vis, 'SHOW_AXES', True):
#                 self._add_coordinate_axes()
#             self._add_waypoints()
    
#     def _add_ground_grid(self):
#         """Add ground grid to scene."""
#         grid_size = getattr(vis, 'GRID_SIZE', 2000.0)
#         grid_spacing = getattr(vis, 'GRID_SPACING', 500.0)
        
#         grid = pv.Plane(
#             i_size=grid_size,
#             j_size=grid_size,
#             i_resolution=max(2, int(grid_size/grid_spacing)),
#             j_resolution=max(2, int(grid_size/grid_spacing))
#         )
        
#         self.ground_actor = self.plotter.add_mesh(
#             grid,
#             color=getattr(vis, 'GRID_COLOR', 'gray'),
#             style='wireframe',
#             line_width=getattr(vis, 'LINE_WIDTH', 1),
#             opacity=0.5,
#             pickable=False,
#             culling=True
#         )
    
#     def _add_coordinate_axes(self):
#         """Add coordinate axes to scene."""
#         # Use plotter's built-in axes instead of pv.Axes
#         self.plotter.add_axes(
#             xlabel='X',
#             ylabel='Y',
#             zlabel='Z',
#             line_width=2,
#             labels_off=False
#         )
    
#     def _add_waypoints(self):
#         """Add waypoint visualization."""
#         waypoint_color = getattr(vis, 'WAYPOINT_COLOR', 'red')
#         for pos, mode in WAYPOINTS:
#             # Create waypoint sphere
#             sphere = pv.Sphere(
#                 radius=20.0,
#                 center=pos,
#                 phi_resolution=10,
#                 theta_resolution=10
#             )
            
#             actor = self.plotter.add_mesh(
#                 sphere,
#                 color=waypoint_color,
#                 opacity=0.7,
#                 smooth_shading=True,
#                 pickable=True,
#                 render_lines_as_tubes=True
#             )
            
#             self.waypoint_actors.append(actor)
            
#             # Add vertical reference line for landing waypoints
#             if mode in ['land', 'flare']:
#                 line = pv.Line(pointa=[pos[0], pos[1], 0], pointb=pos)
#                 actor = self.plotter.add_mesh(
#                     line,
#                     color=waypoint_color,
#                     opacity=0.3,
#                     line_width=2,
#                     pickable=False
#                 )
#                 self.waypoint_actors.append(actor)

    
#     def update(self, vehicle, controller: Optional = None) -> bool:
#         """Update visualization state."""
#         if not self.active:
#             return False
            
#         current_time = time_manager.get_time()
#         if current_time - self.last_update < 1.0 / self.update_rate:
#             return True
            
#         try:
#             with self.render_lock:
#                 update_start = time.time()
                
#                 self._update_vehicle(vehicle)
#                 self._update_trail(vehicle.position)
                
#                 if hasattr(vehicle, 'wind_velocity'):
#                     self._update_wind(vehicle)
                
#                 self._update_camera(vehicle.position)
#                 self.plotter.render()
                
#                 self._update_performance_metrics(update_start)
                
#                 self.last_update = current_time
#                 return True
                
#         except Exception as e:
#             print(f"Visualization update error: {str(e)}")
#             if sim.DEBUG_MODE:
#                 import traceback
#                 traceback.print_exc()
#             return False
    
#     def _update_vehicle(self, vehicle):
#         """Update vehicle visualization."""
#         heading = np.arctan2(vehicle.velocity[1], vehicle.velocity[0])
#         pitch = np.arctan2(-vehicle.velocity[2],
#                           np.sqrt(vehicle.velocity[0]**2 + vehicle.velocity[1]**2))
        
#         cone = pv.Cone(
#             center=vehicle.position,
#             direction=[np.cos(heading)*np.cos(pitch),
#                       np.sin(heading)*np.cos(pitch),
#                       -np.sin(pitch)],
#             height=50.0,
#             radius=10.0,
#             resolution=20,
#             capping=True
#         )
        
#         if self.vehicle_actor:
#             self.plotter.remove_actor(self.vehicle_actor)
        
#         self.vehicle_actor = self.plotter.add_mesh(
#             cone,
#             color='white',
#             smooth_shading=True,
#             culling=True,
#             pickable=False
#         )
    
#     def _update_trail(self, position: np.ndarray):
#         """Update trail visualization."""
#         self.trail_points.append(position.copy())
#         if len(self.trail_points) > self.max_trail_points:
#             self.trail_points.pop(0)
        
#         if len(self.trail_points) > 1:
#             points = np.array(self.trail_points)
#             trail = pv.PolyData(points)
            
#             n_points = len(points) - 1
#             lines = np.empty((n_points, 3), dtype=np.int_)
#             lines[:, 0] = 2
#             lines[:, 1] = np.arange(n_points)
#             lines[:, 2] = np.arange(1, n_points + 1)
#             trail.lines = lines.ravel()
            
#             if self.trail_actor:
#                 self.plotter.remove_actor(self.trail_actor)
            
#             self.trail_actor = self.plotter.add_mesh(
#                 trail,
#                 color='yellow',
#                 line_width=2,
#                 opacity=0.7,
#                 pickable=False,
#                 render_lines_as_tubes=True
#             )
    
#     def _update_wind(self, vehicle):
#         """Update wind visualization."""
#         direction = vehicle.wind_velocity / max(np.linalg.norm(vehicle.wind_velocity), 1e-6)
#         scale = 50.0
        
#         arrow = pv.Arrow(
#             start=vehicle.position,
#             direction=direction * scale,
#             tip_length=0.2,
#             tip_radius=0.1,
#             shaft_radius=0.05,
#             shaft_resolution=20,
#             tip_resolution=20
#         )
        
#         if self.wind_actor:
#             self.plotter.remove_actor(self.wind_actor)
        
#         self.wind_actor = self.plotter.add_mesh(
#             arrow,
#             color='cyan',
#             opacity=0.7,
#             smooth_shading=True,
#             pickable=False
#         )
    
#     def _update_camera(self, target_position: np.ndarray):
#         """Update camera position."""
#         camera_offset = getattr(vis, 'CAMERA_OFFSET', np.array([100, 75, 75]))
#         camera_pos = target_position + camera_offset
        
#         self.plotter.camera_position = [
#             camera_pos.tolist(),
#             target_position.tolist(),
#             [0, 0, 1]  # Up direction
#         ]
    
#     def _update_performance_metrics(self, update_start: float):
#         """Update performance metrics."""
#         render_time = time.time() - update_start
#         self.render_times.append(render_time)
#         if len(self.render_times) > self.max_render_times:
#             self.render_times.pop(0)
    
#     def _start_render_thread(self):
#         """Start render thread."""
#         self.render_thread = threading.Thread(target=self._render_loop)
#         self.render_thread.daemon = True
#         self.render_thread.start()
    
#     def _render_loop(self):
#         """Main render loop."""
#         try:
#             self.plotter.show(interactive_update=True)
#         except Exception as e:
#             print(f"Render thread error: {str(e)}")
#             if sim.DEBUG_MODE:
#                 import traceback
#                 traceback.print_exc()
    
#     def _reset_camera(self):
#         """Reset camera position."""
#         default_camera_pos = [-1000, -750, 750]
#         default_focus_point = [0, 0, 0]
#         default_up_direction = [0, 0, 1]
        
#         self.plotter.camera_position = [
#             default_camera_pos,
#             default_focus_point,
#             default_up_direction
#         ]
#         self.plotter.reset_camera()
    
#     def _cycle_camera(self):
#         """Cycle through camera views."""
#         views = [
#             ('xy', [0, 0, 1]),    # Top view
#             ('xz', [0, -1, 0]),   # Front view
#             ('yz', [1, 0, 0])     # Side view
#         ]
#         current_up = tuple(self.plotter.camera.GetViewUp())
        
#         for i, (view, up) in enumerate(views):
#             if np.allclose(up, current_up):
#                 next_view = views[(i + 1) % len(views)]
#                 self.plotter.view_vector(next_view[1])
#                 break
    
#     def _toggle_grid(self):
#         """Toggle grid visibility."""
#         if self.ground_actor:
#             self.ground_actor.SetVisibility(
#                 not self.ground_actor.GetVisibility()
#             )
    
#     def quit(self):
#         """Clean shutdown."""
#         self.active = False
#         if hasattr(self, 'plotter'):
#             self.plotter.close()
    
#     def is_active(self) -> bool:
#         """Check if visualization is active."""
#         return (self.active and 
#                 hasattr(self, 'plotter') and 
#                 not self.plotter.closed)




# visualization/visualizer.py

# visualization/visualizer.py

"""
Flight visualization using PyVista.
Provides real-time 3D visualization of the paraglider simulation.
"""

import numpy as np
import pyvista as pv
from typing import List, Optional, Dict, Tuple
import threading
import time
from pathlib import Path
from config.general_config import vis, sim
from config.flight_config import WAYPOINTS
from utils.time_manager import time_manager

class FlightVisualizer:
    """Real-time 3D visualization system using PyVista."""
    
    def __init__(self):
        """Initialize visualization system with PyVista setup."""
        # Core state initialization
        self.active = True
        self.last_update = time_manager.get_time()
        self.update_rate = getattr(vis, 'DISPLAY_UPDATE_RATE', 20.0)
        
        # Initialize required attributes
        self.trail_points: List[np.ndarray] = []
        self.max_trail_points = getattr(vis, 'TRAIL_LENGTH', 100)
        self.waypoint_actors: List[pv.Actor] = []
        self.render_times: List[float] = []
        self.max_render_times = 100
        
        # Scene actors
        self.vehicle_actor = None
        self.trail_actor = None
        self.wind_actor = None
        self.ground_actor = None
        self.axes_actor = None
        
        # Thread control
        self.render_thread = None
        self.render_lock = threading.Lock()
        
        # Camera state tracking
        self.last_camera_update = 0.0
        self.camera_update_rate = 30.0  # Hz
        self.last_camera_position = None
        
        try:
            # Initialize visualization system
            self._configure_pyvista()
            self._init_plotter()
            self._setup_scene()
            
            # Start visualization
            if getattr(sim, 'MULTITHREADED_DISPLAY', False):
                self._start_render_thread()
            else:
                self.plotter.show(interactive_update=True)
        except Exception as e:
            raise RuntimeError(f"Visualization initialization failed: {str(e)}")
    
    def _configure_pyvista(self):
        """Configure PyVista global settings."""
        bg_color = getattr(vis, 'BACKGROUND_COLOR', 'black')
        pv.global_theme.background = bg_color
        pv.global_theme.show_edges = True
        pv.global_theme.edge_color = 'white'
        pv.global_theme.font.size = 12
        pv.global_theme.font.family = 'arial'
        pv.global_theme.show_scalar_bar = False
        pv.global_theme.smooth_shading = True
    
    def _init_plotter(self):
        """Initialize PyVista plotter with optimized settings."""
        window_size = getattr(vis, 'WINDOW_SIZE', (1024, 768))
        self.plotter = pv.Plotter(
            window_size=window_size,
            off_screen=getattr(sim, 'MULTITHREADED_DISPLAY', False),
            notebook=False
        )
        
        # Set up camera with stable parameters
        camera_pos = getattr(vis, 'CAMERA_POSITION', [-1000, -750, 750])
        focus_point = getattr(vis, 'CAMERA_FOCUS_POINT', [0, 0, 0])
        up_direction = getattr(vis, 'CAMERA_UP_DIRECTION', [0, 0, 1])
        
        self.plotter.camera_position = [camera_pos, focus_point, up_direction]
        self.plotter.camera.SetViewAngle(60.0)
        self.plotter.camera.SetClippingRange(1.0, 5000.0)
        
        # Initialize key bindings
        self.plotter.add_key_event('q', self.quit)
        self.plotter.add_key_event('r', self.reset_camera)
        self.plotter.add_key_event('g', self.toggle_grid)
        self.plotter.add_key_event('c', self.cycle_camera)
        self.plotter.enable_trackball_style()
        
        # Configure rendering features
        self._configure_rendering()
    
    def _configure_rendering(self):
        """Configure rendering features with fallbacks."""
        try:
            self.plotter.enable_anti_aliasing()
            
            if getattr(vis, 'ENABLE_LIGHTING', True):
                if hasattr(self.plotter, 'enable_lightkit'):
                    self.plotter.enable_lightkit()
                elif hasattr(self.plotter, 'enable_eye_dome_lighting'):
                    self.plotter.enable_eye_dome_lighting()
            
            if hasattr(self.plotter, 'enable_depth_peeling'):
                self.plotter.enable_depth_peeling()
        except Exception:
            pass  # Silently handle unsupported rendering features
    
    def _setup_scene(self):
        """Set up initial visualization scene."""
        with self.render_lock:
            self._add_ground_grid()
            if getattr(vis, 'SHOW_AXES', True):
                self._add_coordinate_axes()
            self._add_waypoints()
    
    def _add_ground_grid(self):
        """Add ground grid to scene."""
        grid_size = getattr(vis, 'GRID_SIZE', 2000.0)
        grid_spacing = getattr(vis, 'GRID_SPACING', 500.0)
        
        grid = pv.Plane(
            i_size=grid_size,
            j_size=grid_size,
            i_resolution=max(2, int(grid_size/grid_spacing)),
            j_resolution=max(2, int(grid_size/grid_spacing))
        )
        
        self.ground_actor = self.plotter.add_mesh(
            grid,
            color=getattr(vis, 'GRID_COLOR', 'gray'),
            style='wireframe',
            line_width=getattr(vis, 'LINE_WIDTH', 1),
            opacity=0.5,
            pickable=False,
            culling=True
        )
    
    def _add_coordinate_axes(self):
        """Add coordinate axes to scene."""
        self.plotter.add_axes(
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
            line_width=2,
            labels_off=False
        )
    
    def _add_waypoints(self):
        """Add waypoint visualization."""
        waypoint_color = getattr(vis, 'WAYPOINT_COLOR', 'red')
        for pos, mode in WAYPOINTS:
            sphere = pv.Sphere(
                radius=20.0,
                center=pos,
                phi_resolution=10,
                theta_resolution=10
            )
            
            actor = self.plotter.add_mesh(
                sphere,
                color=waypoint_color,
                opacity=0.7,
                smooth_shading=True,
                pickable=True,
                render_lines_as_tubes=True
            )
            
            self.waypoint_actors.append(actor)
            
            if mode in ['land', 'flare']:
                line = pv.Line(pointa=[pos[0], pos[1], 0], pointb=pos)
                actor = self.plotter.add_mesh(
                    line,
                    color=waypoint_color,
                    opacity=0.3,
                    line_width=2,
                    pickable=False
                )
                self.waypoint_actors.append(actor)
    
    def update(self, vehicle, controller: Optional = None) -> bool:
        """Update visualization state with improved physics synchronization."""
        if not self.active:
            return False
        
        current_time = time_manager.get_time()
        if current_time - self.last_update < 1.0 / self.update_rate:
            return True
        
        try:
            with self.render_lock:
                update_start = time.time()
                
                # Create defensive copies of physics state
                position = np.array(vehicle.position, dtype=np.float64, copy=True)
                velocity = np.array(vehicle.velocity, dtype=np.float64, copy=True)
                
                # Validate physics state
                if not self._validate_physics_state(position, velocity):
                    if sim.DEBUG_MODE:
                        print("Physics state validation failed")
                    return True  # Continue simulation despite visualization issues
                
                # Update visualization with validated state
                success = True
                success &= self._update_vehicle_safe(position, velocity)
                success &= self._update_trail(position)
                
                if hasattr(vehicle, 'wind_velocity'):
                    wind_vel = np.array(vehicle.wind_velocity, dtype=np.float64, copy=True)
                    success &= self._update_wind_safe(position, wind_vel)
                
                # Rate-limited camera update
                current_time = time.time()
                if current_time - self.last_camera_update >= 1.0 / self.camera_update_rate:
                    success &= self._update_camera_stable(position)
                    if success:
                        self.last_camera_update = current_time
                
                if success:
                    self.plotter.render()
                    self._update_performance_metrics(update_start)
                    self.last_update = current_time
                
                return True  # Always return True to prevent simulation failure
                
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Visualization update warning: {str(e)}")
            return True  # Continue simulation despite visualization issues
    
    def _validate_physics_state(self, position: np.ndarray, velocity: np.ndarray) -> bool:
        """Validate physics state before visualization update."""
        try:
            # Ensure arrays are proper numpy arrays
            if not (isinstance(position, np.ndarray) and isinstance(velocity, np.ndarray)):
                return False
            
            # Check array shapes
            if position.shape != (3,) or velocity.shape != (3,):
                return False
            
            # Check for NaN or infinite values
            if not (np.all(np.isfinite(position)) and np.all(np.isfinite(velocity))):
                return False
            
            # Check reasonable bounds
            pos_limit = 1e5
            vel_limit = 1e3
            if (np.any(np.abs(position) > pos_limit) or 
                np.any(np.abs(velocity) > vel_limit)):
                return False
            
            return True
        except Exception:
            return False
    
    def _update_vehicle_safe(self, position: np.ndarray, velocity: np.ndarray) -> bool:
        """Update vehicle visualization with safety checks."""
        try:
            # Calculate orientation with safety limits
            speed = np.linalg.norm(velocity)
            if speed > 1e-6:
                heading = np.arctan2(velocity[1], velocity[0])
                raw_pitch = np.arctan2(-velocity[2], speed)
                # Limit pitch angle
                pitch = np.clip(raw_pitch, -np.pi/2.5, np.pi/2.5)
            else:
                heading = 0.0
                pitch = 0.0
            
            # Create vehicle geometry
            cone = pv.Cone(
                center=position,
                direction=[np.cos(heading)*np.cos(pitch),
                          np.sin(heading)*np.cos(pitch),
                          -np.sin(pitch)],
                height=50.0,
                radius=10.0,
                resolution=20,
                capping=True
            )
            
            if self.vehicle_actor:
                self.plotter.remove_actor(self.vehicle_actor)
            
            self.vehicle_actor = self.plotter.add_mesh(
                cone,
                color=getattr(vis, 'VEHICLE_COLOR', 'white'),
                smooth_shading=True,
                culling=True,
                pickable=False
            )
            
            return True
            
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Vehicle update warning: {str(e)}")
            return False
    
    def _update_wind_safe(self, position: np.ndarray, wind_velocity: np.ndarray) -> bool:
        """Update wind visualization with safety checks."""
        try:
            magnitude = np.linalg.norm(wind_velocity)
            if magnitude > 1e-6:
                direction = wind_velocity / magnitude
                arrow = pv.Arrow(
                    start=position,
                    direction=direction * min(50.0, magnitude),  # Limit max length
                    tip_length=0.2,
                    tip_radius=0.1,
                    shaft_radius=0.05,
                    shaft_resolution=20,
                    tip_resolution=20
                )
                
                if self.wind_actor:
                    self.plotter.remove_actor(self.wind_actor)
                
                self.wind_actor = self.plotter.add_mesh(
                    arrow,
                    color='cyan',
                    opacity=0.7,
                    smooth_shading=True,
                    pickable=False
                )
            return True
            
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Wind update warning: {str(e)}")
            return False
    
    def _update_trail(self, position: np.ndarray) -> bool:
        """Update trail visualization with safety checks."""
        try:
            self.trail_points.append(position.copy())
            if len(self.trail_points) > self.max_trail_points:
                self.trail_points.pop(0)
            
            if len(self.trail_points) > 1:
                points = np.array(self.trail_points)
                trail = pv.PolyData(points)
                
                n_points = len(points) - 1
                lines = np.empty((n_points, 3), dtype=np.int_)
                lines[:, 0] = 2
                lines[:, 1] = np.arange(n_points)
                lines[:, 2] = np.arange(1, n_points + 1)
                trail.lines = lines.ravel()
                
                if self.trail_actor:
                    self.plotter.remove_actor(self.trail_actor)
                
                self.trail_actor = self.plotter.add_mesh(
                    trail,
                    color=getattr(vis, 'TRAIL_COLOR', 'yellow'),
                    line_width=getattr(vis, 'LINE_WIDTH', 2),
                    opacity=0.7,
                    pickable=False,
                    render_lines_as_tubes=True
                )
            return True
            
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Trail update warning: {str(e)}")
            return False
    
    def _update_camera_stable(self, target_position: np.ndarray) -> bool:
        """Update camera position with improved stability."""
        try:
            # Calculate stable camera parameters
            base_distance = 150.0
            height_offset = max(75.0, target_position[2] * 0.1)
            
            # Calculate desired camera position with smooth angle
            desired_pos = target_position + np.array([
                -base_distance * 0.866,  # cos(60) for stable view
                -base_distance * 0.5,    # sin(60) for stable view
                height_offset
            ])
            
            # Apply smooth transition
            if self.last_camera_position is not None:
                # Calculate smooth transition factor based on distance
                delta = desired_pos - self.last_camera_position
                distance = np.linalg.norm(delta)
                smooth_factor = min(0.15, 1.0 / (1.0 + distance * 0.01))
                
                # Apply smoothing with speed limit
                max_speed = 50.0
                movement = delta * smooth_factor
                if np.linalg.norm(movement) > max_speed:
                    movement = movement * (max_speed / np.linalg.norm(movement))
                
                camera_pos = self.last_camera_position + movement
            else:
                camera_pos = desired_pos
            
            # Store for next update
            self.last_camera_position = camera_pos.copy()
            
            # Update camera with stable parameters
            self.plotter.camera_position = [
                camera_pos.tolist(),
                target_position.tolist(),
                [0, 0, 1]  # Maintain vertical up vector
            ]
            
            # Ensure stable view parameters
            self.plotter.camera.SetViewAngle(60.0)
            self.plotter.camera.SetClippingRange(1.0, 5000.0)
            
            return True
        
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Camera update warning: {str(e)}")
            return False
    
    def _update_performance_metrics(self, update_start: float):
        """Update visualization performance metrics."""
        try:
            render_time = time.time() - update_start
            self.render_times.append(render_time)
            while len(self.render_times) > self.max_render_times:
                self.render_times.pop(0)
        except Exception:
            pass  # Silently handle performance metric errors
    
    def _start_render_thread(self):
        """Start visualization in separate thread."""
        try:
            self.render_thread = threading.Thread(target=self._render_loop)
            self.render_thread.daemon = True
            self.render_thread.start()
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Render thread start error: {str(e)}")
    
    def _render_loop(self):
        """Main render loop with error recovery."""
        try:
            while self.active:
                with self.render_lock:
                    self.plotter.show(interactive_update=True)
                time.sleep(0.001)  # Prevent thread hogging
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Render loop error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def quit(self):
        """Perform clean shutdown of visualization."""
        try:
            self.active = False
            
            # Clean up render thread if active
            if self.render_thread and self.render_thread.is_alive():
                self.render_thread.join(timeout=1.0)
            
            # Close plotter if it exists
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.close()
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Visualization shutdown error: {str(e)}")
    
    def reset_camera(self):
        """Reset camera position with stability."""
        try:
            default_camera_pos = [-1000, -750, 750]
            default_focus_point = [0, 0, 0]
            default_up_direction = [0, 0, 1]
            
            # Reset camera position with stable parameters
            self.plotter.camera_position = [
                default_camera_pos,
                default_focus_point,
                default_up_direction
            ]
            
            # Set stable view parameters
            self.plotter.camera.SetViewAngle(60.0)
            self.plotter.camera.SetClippingRange(1.0, 5000.0)
            self.plotter.reset_camera()
            
            # Clear camera history
            self.last_camera_position = None
            
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Camera reset error: {str(e)}")
    
    def toggle_grid(self):
        """Toggle ground grid visibility."""
        if self.ground_actor:
            self.ground_actor.SetVisibility(
                not self.ground_actor.GetVisibility()
            )
    
    def cycle_camera(self):
        """Cycle through camera views with stability."""
        try:
            views = [
                ('xy', [0, 0, 1]),    # Top view
                ('xz', [0, -1, 0]),   # Front view
                ('yz', [1, 0, 0])     # Side view
            ]
            current_up = tuple(self.plotter.camera.GetViewUp())
            
            for i, (view, up) in enumerate(views):
                if np.allclose(up, current_up):
                    next_view = views[(i + 1) % len(views)]
                    self.plotter.view_vector(next_view[1])
                    
                    # Reset view parameters for stability
                    self.plotter.camera.SetViewAngle(60.0)
                    self.plotter.camera.SetClippingRange(1.0, 5000.0)
                    self.last_camera_position = None  # Reset smooth tracking
                    break
        
        except Exception as e:
            if sim.DEBUG_MODE:
                print(f"Camera cycle error: {str(e)}")
    
    def is_active(self) -> bool:
        """Check if visualization is active and ready."""
        return (self.active and 
                hasattr(self, 'plotter') and 
                not self.plotter.closed)