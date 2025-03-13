import time
import openai

from envs.real_env import RealEnv
from planner.planner import KUDAPlanner
from utils import get_config_real

openai.api_key = None


config = get_config_real('configs/real_config.yaml')
env_config = config.env
env = RealEnv(env_config)
env.start(exposure_time=3)
time.sleep(3)

planner_config = config.planner
planner = KUDAPlanner(env, planner_config)
object = material = 'rope'
# object = material = 'cube'
# object = material = 'T_shape'

# object = 'coffee_beans'
# object = 'candy'
# material = 'granular'

instruction = 'straighten the rope'
# instruction = 'make the rope into a "V" shape'
# instruction = 'put two ends of the rope together'
# instruction = 'move all the cubes to the pink cross'
# instruction = 'move the yellow cube to the red cross'
# instruction = 'move all the coffee beans to the red cross'
# instruction = "move the orange T into the pink square"

planner(object, material, instruction)
