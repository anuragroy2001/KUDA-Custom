from PIL import Image
import numpy as np
import openai

from planner.planner import KUDAPlanner
from utils import get_config_real

openai.api_key = None


config = get_config_real('configs/real_config.yaml')
planner_config = config.planner
env = None
planner = KUDAPlanner(env, planner_config)
img = Image.open('prompts/random_0.jpg')
img = np.array(img)[:, :, ::-1] # in BGR format
instruction = "divide the chessmen into two groups based on their color"
# instruction = "gather the purple earplugs together"
planner.demo_call(img, instruction)
