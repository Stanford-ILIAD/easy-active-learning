from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo
import sys

def nonbatch(task, criterion, M, N):
    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    for i in range(N):
        w_samples, delta_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        input_A, input_B = run_algo(criterion, simulation_object, w_samples, delta_samples)
        phi_A, phi_B, s = get_feedback(simulation_object, input_A, input_B)
        w_sampler.feed(phi_A, phi_B, s)
    w_samples, delta_samples = w_sampler.sample(M)
    print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))


