import numpy as np
import scipy.optimize as opt
import algos
from models import Driver, LunarLander, MountainCar, Swimmer, Tosser, LDS, Fetch

# w_driver = [0.32869792, -0.13381809, 0.34833876, -0.67377869]
def simulate_human(phi_A, phi_B, w, delta, query_type):
	p = np.random.rand()
	if query_type == 'strong':
		delta = 0
	pA = 1. / (1 + np.exp(delta - w.dot(phi_A - phi_B)))
	pB = 1. / (1 + np.exp(delta + w.dot(phi_A - phi_B)))
	print(w.dot(phi_A - phi_B))
	print(1-pA-pB)
	if p < pA:
		return -1
	elif p < pA + pB:
		return 1
	assert query_type == 'weak', 'Probably a numerical error in human simulation'
	return 0
	

def get_feedback(simulation_object, input_A, input_B, query_type, true_w=None, true_delta=None):
	simulation_object.feed(input_A)
	phi_A = np.array(simulation_object.get_features())
	simulation_object.feed(input_B)
	phi_B = np.array(simulation_object.get_features())
	psi = phi_A - phi_B
	s = -2
	if true_w is None:
		while s==-2:
			if query_type == 'weak':
				selection = input('A/B to watch, 1/2 to vote, 0 for IDK: ').lower()
			elif query_type == 'strong':
				selection = input('A/B to watch, 1/2 to vote: ').lower()
			else:
				print('There is no query type called ' + query_type)
				exit(0)
			if selection == 'a':
				simulation_object.feed(input_A)
				simulation_object.watch(1)
			elif selection == 'b':
				simulation_object.feed(input_B)
				simulation_object.watch(1)
			elif selection == '0' and query_type == 'weak':
				s = 0
			elif selection == '1':
				s = -1
			elif selection == '2':
				s = 1
	else:
		s = simulate_human(phi_A, phi_B, true_w, true_delta, query_type)
	print(s)
	return phi_A, phi_B, s


def create_env(task):
	if task == 'lds':
		return LDS()
	elif task == 'driver':
		return Driver()
	elif task == 'lunarlander':
		return LunarLander()
	elif task == 'mountaincar':
		return MountainCar()
	elif task == 'swimmer':
		return Swimmer()
	elif task == 'tosser':
		return Tosser()
	elif task == 'fetch':
		return Fetch()
	else:
		print('There is no task called ' + task)
		exit(0)


def run_algo(criterion, simulation_object, w_samples, delta_samples):
	if criterion == 'information':
		return algos.information(simulation_object, w_samples, delta_samples)
	if criterion == 'volume':
		return algos.volume(simulation_object, w_samples, delta_samples)
	elif criterion == 'random':
		return algos.random(simulation_object)
	else:
		print('There is no criterion called ' + criterion)
		exit(0)


def func(ctrl_array, *args):
	simulation_object = args[0]
	w = np.array(args[1])
	simulation_object.set_ctrl(ctrl_array)
	features = simulation_object.get_features()
	return -np.mean(np.array(features).dot(w))

def perform_best(simulation_object, w, iter_count=10):
	u = simulation_object.ctrl_size
	lower_ctrl_bound = [x[0] for x in simulation_object.ctrl_bounds]
	upper_ctrl_bound = [x[1] for x in simulation_object.ctrl_bounds]
	opt_val = np.inf
	for _ in range(iter_count):
		temp_res = opt.fmin_l_bfgs_b(func, x0=np.random.uniform(low=lower_ctrl_bound, high=upper_ctrl_bound, size=(u)), args=(simulation_object, w), bounds=simulation_object.ctrl_bounds, approx_grad=True)
		if temp_res[1] < opt_val:
			optimal_ctrl = temp_res[0]
			opt_val = temp_res[1]
	simulation_object.set_ctrl(optimal_ctrl)
	keep_playing = 'y'
	while keep_playing == 'y':
		keep_playing = 'u'
		simulation_object.watch(1)
		while keep_playing != 'n' and keep_playing != 'y':
			keep_playing = input('Again? [y/n]: ').lower()
	return -opt_val
