import numpy as np
import scipy.optimize as opt
import algos
from models import Driver, LunarLander, MountainCar, Swimmer, Tosser


def get_feedback(simulation_object, input_A, input_B):
    simulation_object.feed(input_A)
    phi_A = simulation_object.get_features()
    simulation_object.feed(input_B)
    phi_B = simulation_object.get_features()
    psi = np.array(phi_A) - np.array(phi_B)
    s = 0
    while s==0:
        selection = input('A/B to watch, 1/2 to vote: ').lower()
        if selection == 'a':
            simulation_object.feed(input_A)
            simulation_object.watch(1)
        elif selection == 'b':
            simulation_object.feed(input_B)
            simulation_object.watch(1)
        elif selection == '1':
            s = 1
        elif selection == '2':
            s = -1
    return psi, s


def create_env(task):
    if task == 'driver':
        return Driver()
    elif task == 'lunarlander':
        return LunarLander()
    elif task == 'mountaincar':
        return MountainCar()
    elif task == 'swimmer':
        return Swimmer()
    elif task == 'tosser':
        return Tosser()
    else:
        print('There is no task called ' + task)
        exit(0)


def run_algo(criterion, simulation_object, w_samples):
    if criterion == 'information':
        return algos.information(simulation_object, w_samples)
    if criterion == 'volume':
        return algos.volume(simulation_object, w_samples)
    elif criterion == 'random':
        return algos.random(simulation_object, w_samples)
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
