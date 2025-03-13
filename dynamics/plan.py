from .plan_pyflex import closed_loop_plan as pyflex_closed_loop_plan
from .plan_mlp import closed_loop_plan as mlp_closed_loop_plan

def closed_loop_plan(*args, **kwargs):
    if args[3] == 'rope' or args[3] == 'cube' or args[3] == 'granular':
        return pyflex_closed_loop_plan(*args, **kwargs)
    elif args[3] == 'T_shape':
        return mlp_closed_loop_plan(*args, **kwargs)
