import nimblephysics as nimble
import os


world: nimble.simulation.World = nimble.simulation.World()
arm: nimble.dynamics.Skeleton = world.loadSkeleton(os.path.join(
    os.path.dirname(__file__), "./panda.urdf"))

# Your code here
# arm.setPositions([0, 0, 0,0,0,0,-90*(3.1415/180),-90*(3.1415/180),-90*(3.1415/180), -90*(3.1415/180), -90*(3.1415/180), -90*(3.1415/180),-90*(3.1415/180),0,0])
# help(world.loadSkeleton)
arm.setPositions([0, 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

arm_pos = arm.getPositions()
print(arm_pos)

gui = nimble.NimbleGUI(world)
gui.serve(8080)
gui.blockWhileServing()
