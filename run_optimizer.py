from simulation_utils import create_env, compute_best, play
import sys

task = 'Tosser'
w = [0.39885564, 0.05660242, -0.15491889, 0.70144939]
iter_count = 5  # the optimization is nonconvex, so you can specify the number of random starting points

##### YOU DO NOT NEED TO MODIFY THE CODE BELOW THIS LINE #####

simulation_object = create_env(task.lower())
optimal_ctrl = compute_best(simulation_object, w, iter_count)
play(simulation_object, optimal_ctrl)
