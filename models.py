from simulator import LDSSimulation, DrivingSimulation, MujocoSimulation, FetchSimulation
import numpy as np


class LDS(LDSSimulation):
    def __init__(self, total_time=25, recording_time=[0,25]):
        super(LDS, self).__init__(name='lds', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 5
        self.state_size = 0
        self.feed_size = self.ctrl_size*self.input_size + self.state_size
        self.ctrl_bounds = [(-0.1,0.1),(-0.2,0.2),(-0.1,0.1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 6

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # speed (lower is better)
        speed1 = 3*np.mean(np.abs(recording[:,1])) / 0.3805254
        speed2 = 3*np.mean(np.abs(recording[:,3])) / 0.11415762
        speed3 = 3*np.mean(np.abs(recording[:,5])) / 0.5707881

        # distance to the desired position (lower is better)
        distance1 = 3*np.mean(np.abs(recording[:,0]-1)) / 4.072655
        distance2 = 3*np.mean(np.abs(recording[:,2]-1)) / 0.94199475
        distance3 = 3*np.mean(np.abs(recording[:,4]-1)) / 7.10111927

        return [speed1, distance1, speed2, distance2, speed3, distance3]

    @property
    def state(self):
        return [self._state[i] for i in range(6)]
    @state.setter
    def state(self, value):
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = value[i]
        self.ctrl = list(arr)

    def feed(self, value):
        ctrl_value = value[:]
        self.set_ctrl(ctrl_value)



class Driver(DrivingSimulation):
    def __init__(self, total_time=50, recording_time=[0,50]):
        super(Driver ,self).__init__(name='driver', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # staying in lane (higher is better)
        staying_in_lane = np.mean(np.exp(-30*np.min([np.square(recording[:,0,0]-0.17), np.square(recording[:,0,0]), np.square(recording[:,0,0]+0.17)], axis=0))) / 0.15343634

        # keeping speed (lower is better)
        keeping_speed = np.mean(np.square(recording[:,0,3]-1)) / 0.42202643

        # heading (higher is better)
        heading = np.mean(np.sin(recording[:,0,2])) / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance = np.mean(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1])))) / 0.15258019

        return [staying_in_lane, keeping_speed, heading, collision_avoidance]

    @property
    def state(self):
        return [self.robot.x, self.human.x]
    @state.setter
    def state(self, value):
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j], value[j+1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value):
        ctrl_value = value[:]
        self.set_ctrl(ctrl_value)



class Tosser(MujocoSimulation):
    def __init__(self, total_time=1000, recording_time=[200,1000]):
        super(Tosser ,self).__init__(name='tosser', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 4
        self.state_size = 5
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = [(-0.2,0.2),(-0.785,0.785),(-0.1,0.1),(-0.1,-0.07),(-1.5,1.5)]
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # horizontal range
        horizontal_range = -np.min([x[3] for x in recording]) / 0.25019166

        # maximum altitude
        maximum_altitude = np.max([x[2] for x in recording]) / 0.18554402

        # number of flips
        num_of_flips = np.sum(np.abs([recording[i][4] - recording[i-1][4] for i in range(1,len(recording))]))/(np.pi*2) / 0.33866545
        
        # distance to closest basket (gaussian fit)
        dist_to_basket = np.exp(-3*np.linalg.norm([np.minimum(np.abs(recording[len(recording)-1][3] + 0.9), np.abs(recording[len(recording)-1][3] + 1.4)), recording[len(recording)-1][2]+0.85])) / 0.17801466

        return [horizontal_range, maximum_altitude, num_of_flips, dist_to_basket]

    @property
    def state(self):
        return self.sim.get_state()
    @state.setter
    def state(self, value):
        self.reset()
        temp_state = self.initial_state
        temp_state.qpos[:] = value[:]
        self.initial_state = temp_state

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        arr[150:175] = [value[:self.input_size]]*25
        arr[175:200] = [value[self.input_size:2*self.input_size]]*25
        self.ctrl = arr

    def feed(self, value):
        initial_state = value[:self.state_size]
        ctrl_value = value[self.state_size:self.feed_size]
        self.initial_state.qpos[:] = initial_state
        self.set_ctrl(ctrl_value)



class Fetch(FetchSimulation):
    def __init__(self, total_time=152, recording_time=[0,152]):
        super(Fetch ,self).__init__(total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 3*19
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = [(-np.pi/2,np.pi/2)]*self.state_size
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)
        f1 = np.mean(recording[-1,:]) / 1.71217351
        f2 = np.mean(recording[-1,:]) / 1.8090672
        f3 = np.mean(recording[-1,:]) / 2.40721058
        f4 = np.mean(recording[-1,:]) / 0.2506069
        return [f1, f2, f3, f4]

    @property
    def state(self):
        return self.sim.sim.get_state()
    @state.setter
    def state(self, value):
        self.sim.sim.set_state(value)

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j+i] for i in range(3)]
            j += 3
        self.ctrl = list(arr)

    def feed(self, value):
        initial_state = value[:self.state_size]
        ctrl_value = value[self.state_size:self.feed_size]
        self.sim.reset()
        self.set_ctrl(ctrl_value)
