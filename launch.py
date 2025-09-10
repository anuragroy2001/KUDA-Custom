import time
import openai

from envs.real_env import RealEnv
from planner.planner import KUDAPlanner
from utils import get_config_real
import time

openai.api_key = None



config = get_config_real('configs/real_config.yaml')
env_config = config.env
start_time = time.time()
env = RealEnv(env_config)
env.start(exposure_time=3)
end_time = time.time()
print("Environment initialized in {:.4f} seconds".format(end_time - start_time))
time.sleep(3)
start_time = time.time()
planner_config = config.planner
planner = KUDAPlanner(env, planner_config)
end_time = time.time()
print("Planner initialized in {:.4f} seconds".format(end_time - start_time))
# object = material = 'rope'
# object = material = 'cube'
# object = material = 'T_shape'
# object = 'red cube'
# object = 'coffee_beans'
# object = 'candy'
# material = 'granular'

object = 'white cable'
material = 'rope'
# instruction = 'straighten the white cable'
# instruction = 'move the red cube to the left'
# instruction = 'Put the ends of the white cable together'
# instruction = 'make the rope into a "V" shape'
# instruction = 'put two ends of the rope together'
# instruction = 'move all the cubes to the pink cross'
# instruction = 'move the yellow cube to the red cross'
# instruction = 'move all the coffee beans to the red cross'
# instruction = "move the orange T into the pink square"
instruction = "manipulate the white cable to make a smile face"

planner(object, material, instruction)
