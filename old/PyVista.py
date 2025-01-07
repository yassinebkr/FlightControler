import pyvista as pv
import numpy as np
from threading import Thread

MULTITHREADED_DISPLAY = True

class PyVista:

    #def __init__(self):

    def start(self):
        self.run = True

        self.plotter = pv.Plotter()


        grid = pv.Plane(i_size=100, j_size=100)
        self.plotter.set_background("black")
        self.plotter.add_mesh(grid, color='lightgrey', style='wireframe', line_width=1)

        self.plotter.show(interactive_update=True)
        self.plotter.add_key_event("Escape", self.stop)
        self.plotter.add_key_event("p", pause)
        
        self.plotter.add_key_event("q", self.stop)

        self.estimates = FLIGHT_SW.p/1000
        self.simulated = SIMULATED_VEHICULE.p/1000

        # base
        colors = ['red', 'green', 'blue']
        for i in range(3):
            axis = zeros(3)
            axis[i] = 10
            self.plotter.add_lines(Matrix([zeros(3), zeros(3) + axis]), color=colors[i])

        # waypoints
        if WAYPOINTS:
            waypoints = Matrix([w if type(w) in (int, float) else w[0] for w in WAYPOINTS]) /1000
            self.plotter.add_points(waypoints, color='magenta', point_size=10)     

        #for multi threading
        if MULTITHREADED_DISPLAY:
            while self.run:
                if T==0:
                    self.clear()
                try:
                    self.update()
                except:
                    self.run = False
                    traceback.print_exc()

                time.sleep(0.02) # 50 fps
        
    def update(self):
        # wind
        position = FLIGHT_SW.p/1000

        self.wind = self.plotter.add_lines(Vector([0,0,FLIGHT_SW.z/1000], Vector([0,0,FLIGHT_SW.z/1000]) + WIND(position[2]*1000)/10 ), color="cyan", name="wind")

        self.windE = self.plotter.add_lines(Vector(position, position + FLIGHT_SW.windEstimate), color="yellow", name="windE")

        #print(WIND(position[2]*1000), FLIGHT_SW.windEstimate)

        self.estimates = vstack((self.estimates, position))
        self.simulated = vstack((self.simulated, SIMULATED_VEHICULE.p/1000))

        self.plotter.add_points(self.simulated, color='white', point_size=3, name="simulated")
        self.plotter.add_points(self.estimates, color='yellow', point_size=3, name="estimates")

        # position/orientation
        colors = ['red', 'green', 'blue']
        for i in range(3):
            self.plotter.add_lines(Matrix([position + FLIGHT_SW.r[:,i], position + FLIGHT_SW.r[:,i]*3]), color=colors[i], name=f"axis_{i}")
        
        #debug
        # position[2] -= 5
        # try: self.plotter.add_lines(Matrix([position, position + FLIGHT_SW.deviation/10 ]), color="grey", name="deviation")
        # except: pass

        # self.plotter.add_lines(Matrix([position, position + FLIGHT_SW.v/1000 ]), color="white", name="speed")
        
        self.plotter.update()


    def stop(self):
        global SIMU_END
        SIMU_END = -1
    
    def clear(self):
        self.estimates = FLIGHT_SW.p/1000
        self.simulated = SIMULATED_VEHICULE.p/1000

        self.plotter.clear()
        self.plotter.update()

    def kill(self):
        self.plotter.close()
        self.run = False
