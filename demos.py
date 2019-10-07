from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo
import sys

def nonbatch(task, criterion, query_type, epsilon, M):
	simulation_object = create_env(task)		
	d = simulation_object.num_of_features

	true_delta = 1 # make this None if you will also learn delta, and change the samplers below from sample_given_delta to sample (and of course remove the true_delta argument)

	lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
	upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

	w_sampler = Sampler(d)
	i = 0
	score = np.inf
	while score >= epsilon:
		w_samples, delta_samples = w_sampler.sample_given_delta(M, query_type, true_delta)
		mean_w_samples = np.mean(w_samples,axis=0)
		print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
		input_A, input_B, score = run_algo(criterion, simulation_object, w_samples, delta_samples)
		if criterion == 'information':
			print('Expected info gain = {}'.format(score))
		elif criterion == 'volume':
			print('Expected volume removal (meaningless scale) = {}'.format(score/M))
		if score > epsilon:
			phi_A, phi_B, s = get_feedback(simulation_object, input_A, input_B, query_type)
			w_sampler.feed(phi_A, phi_B, [s])
			i += 1
	w_samples, delta_samples = w_sampler.sample_given_delta(M, query_type, true_delta)
	mean_w_samples = np.mean(w_samples,axis=0)
	print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
