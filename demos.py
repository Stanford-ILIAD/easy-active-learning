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
		#true_w = np.array([0.57152483, -0.31678367, 0.08319346, -0.75238709]) # driver
		#true_w = np.array([0.39885564, 0.05660242, -0.15491889, 0.70144939]) # tosser
		#true_w = np.array([-0.49809642, -0.55274919, 0.20785098, -0.63495375]) # fetch
		true_w = true_w / np.linalg.norm(true_w)
		true_delta = 1
		print(true_delta)

	lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
	upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

	w_sampler = Sampler(d)
	i = 0
	score = np.inf
	while score >= c and (c != 0 or i < N):
		w_samples, delta_samples = w_sampler.sample_given_delta(M, query_type, true_delta)
		mean_w_samples = np.mean(w_samples,axis=0)
		print('w-estimate = {}'.format(mean_w_samples))
		if true_w is not None:
			print('m = {}'.format(np.mean(w_samples.dot(true_w)/np.linalg.norm(w_samples, axis=1))))
		input_A, input_B, score = run_algo(criterion, simulation_object, w_samples, delta_samples)
		if score > c:
			phi_A, phi_B, s = get_feedback(simulation_object, input_A, input_B, query_type, true_w, true_delta)
			print('psi = ' + str(phi_A-phi_B))
			w_sampler.feed(phi_A, phi_B, [s])
			i += 1
	w_samples, delta_samples = w_sampler.sample_given_delta(M, query_type, true_delta)
	print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
	if true_w is not None:
			print('m = {}'.format(np.mean(w_samples.dot(true_w)/np.linalg.norm(w_samples, axis=1))))


