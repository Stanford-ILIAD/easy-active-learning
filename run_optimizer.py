from simulation_utils import create_env, compute_best, play
import sys

task = sys.argv[1].lower()
iter_count = int(sys.argv[2]) # the optimization is nonconvex, so you can specify the number of random starting points
w = [float(x) for x in sys.argv[3:]]

##### YOU DO NOT NEED TO MODIFY THE CODE BELOW THIS LINE #####

simulation_object = create_env(task.lower())
optimal_ctrl = compute_best(simulation_object, w, iter_count)
play(simulation_object, optimal_ctrl)
