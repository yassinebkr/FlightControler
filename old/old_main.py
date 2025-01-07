import traceback
import code, sys, os
import time

###### file loading #######

def load(directory):
    """ load every file in <target> (and its subdirectories) in the main scope (replace classic <import>) """
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and not file.endswith('.test.py'):
                # Construct full file path
                path = os.path.join(root, file)
                exc(path)



def exc(path):
    """ execute a python file """
    
    with open(path) as f:
        txt = f.read()
        print("loading", path)
        try:
            exec(compile(txt, path, "exec"), globals(), globals())
        except:
            traceback.print_exc()
            input("error, retry...")
            exit()


####### runtime #######

def pause():
    global DT, DT_SAVE
    """ pause/resume the simulation """
    global DT, DT_SAVE
    if DT:
        DT_SAVE = DT
        DT = 0
        print("paused")
    else:
        DT = DT_SAVE
        print("resumed")

def stop():
    global loop

    if visualisor:
        visualisor.stop()
        visualisor.update()
        visualisor.kill()
        renderThread.join()
    loop = False

    exit()


def cmd():
    """ debug console """
    try:
        code.interact(banner="\n r() to restart, q() to quit, in live mode p() to pause/resume the simulation\ninteresting variables: SIMULATED_VEHICULE, FLIGHT_SW\n", local=globals())
        print("")
    except :

        # try:
        #     visualisor.stop()
        #     renderThread.join()
        # except: pass

        sys.stdin = os.fdopen(0)    

#for cmd

def r():
    """ restart """
    if visualisor: visualisor.stop()
    exit()
q = stop
p = pause

loop = True
visualisor = None


    
while loop:
    print("\n\n\n\n\n")
    # safe execution
    try:
        ####### intialize the project by loading all files #######
        load("utils")
        exc("conf/generalConf.py")
        load("shared")
        load("flightSW")
        load("simulator")
        #exc("conf.py")
        load("conf")

        
        ####### Runtime #######

        # intialize visualisation
        if VISUALISATION and not visualisor: # run a 3D visualisor
            visualisor = VISUALISATION()

            if MULTITHREADED_DISPLAY:
                renderThread = Thread(target=visualisor.start)
                renderThread.start()
            else:
                visualisor.start()

        if INTERACTIVE: # realtime console
            consoleThread = Thread(target=cmd)
            consoleThread.start()

        t0 = time.time()
        while T < SIMU_END:
            SIMULATED_VEHICULE.compute()

            #if VISUALISATION: VISUALISATION.update()

            dt = time.time()-t0
            t0 = time.time()

            if TIME_EQUIVALANCE: # reduce speed for visualisation prupose
                toSleep = DT/TIME_EQUIVALANCE - dt
                if toSleep>0:
                    time.sleep(toSleep) # synchronise simulation time

            # if T%TIME_EQUIVALANCE<DT: # debu log # TODO remove
            #     #FLIGHT_SW.predict(save=True)
            #     print("T=", T,"step computation:", dt*1000,"ms", dt)
            #     print("real position:", SIMULATED_VEHICULE.p, "estimated:", FLIGHT_SW.p)
            #     print("real velocity:", SIMULATED_VEHICULE.v, "estimated:", FLIGHT_SW.v)
            #     print("real yaw:", SIMULATED_VEHICULE.yaw, "estimated:", FLIGHT_SW.yaw,FLIGHT_SW.Y)

            if not MULTITHREADED_DISPLAY: visualisor.update()

            T += DT

            if LOGGER_DUMP_F and T and not T%LOGGER_DUMP_F:
                print("log dumped!", T)
                LOGGER.dumpAll(LOG_PATH)

            if SIMULATED_VEHICULE.z<=0: #imperfect if target != 0
                print("landed! end of the simulation")
                break

            
    except SystemExit: pass
    except:
        traceback.print_exc()

        # try:
        #     visualisor.stop()
        #     renderThread.join()
        # except: pass

    if LOGGER:
        try: LOGGER.dumpAll(LOG_PATH) # save data
        except: traceback.print_exc()

    if INTERACTIVE: consoleThread.join() # wait for the user to quit
    else: cmd() # run the console
  