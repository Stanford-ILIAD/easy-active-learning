from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo
import sys

def nonbatch(task, criterion, query_type, c, M, N, simulated_user=False):
	simulation_object = create_env(task)		
	d = simulation_object.num_of_features
	
	true_w = None
	true_delta = None
	if simulated_user:
		true_w = np.random.rand(d)
		#true_w = np.array([0.32869792, -0.13381809, 0.34833876, -0.67377869])
		true_w = true_w / np.linalg.norm(true_w)
		true_delta = 10*np.random.rand()
		print(true_delta)

	lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
	upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

	w_sampler = Sampler(d)
	for i in range(N):
		w_samples, delta_samples = w_sampler.sample(M, query_type)
		mean_w_samples = np.mean(w_samples,axis=0)
		print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
		if true_w is not None:
			print('m = {}'.format(np.mean(w_samples.dot(true_w)/np.linalg.norm(w_samples, axis=1))))
		input_A, input_B = run_algo(criterion, simulation_object, w_samples, delta_samples)
		phi_A, phi_B, s = get_feedback(simulation_object, input_A, input_B, query_type, true_w, true_delta)
		w_sampler.feed(phi_A, phi_B, [s])
	w_samples, delta_samples = w_sampler.sample(M, query_type)
	print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
	if true_w is not None:
			print('m = {}'.format(np.mean(w_samples.dot(true_w)/np.linalg.norm(w_samples, axis=1))))


