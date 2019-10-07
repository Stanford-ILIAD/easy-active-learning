import numpy as np

from fetch_gym.envs import robot_env, rotations, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        if model_path == 'fetch/reach.xml':
            self.type = 'reach'
        elif model_path == 'fetch/reach_test.xml':
            self.type = 'reach_test'
        else:
            self.type = None

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=3,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        try:
            w = [-0.54, -0.10, 0.83]

            table_pos = self.sim.data.get_body_xpos('table')
            obstacle_pos = self.sim.data.get_body_xpos('boxobstacle')
            goal_pos = self.sim.data.get_body_xpos('goal')

            d = goal_distance(achieved_goal, goal)
            dist_threshold = 0.35  # ensure that this is the same as in domain.py

            x = achieved_goal  # TODO: Gleb, this is end_effector position, correct?

            goal_dist = 25 * -np.exp(
                -np.sqrt((x[0] - goal_pos[0]) ** 2 + (x[1] - goal_pos[1]) ** 2 + (x[2] - goal_pos[2]) ** 2))
            table_dist = 5 * -np.exp(-np.sqrt((x[2] - table_pos[2]) ** 2))
            obstacle_dist = 40 * (1 - np.exp(
                -np.sqrt((x[0] - obstacle_pos[0]) ** 2 + (x[1] - obstacle_pos[1]) ** 2 + (x[2] - obstacle_pos[2]) ** 2)))
            final_goal_dist = 250 * np.exp(-d) if d < dist_threshold else 0

            return np.array([goal_dist, table_dist, obstacle_dist]).dot(w)
        except:
            d = goal_distance(achieved_goal, goal)
            if self.reward_type == 'sparse':
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope

        pos_ctrl, gripper_ctrl = action[:3], 0

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        # rot_ctrl = action[3:]
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)


    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:torso_lift_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        ### COMMENT OUT WHEN RUNNING SIMULATION EXPERIMENTS -- ONLY REQUIRED FOR HUMAN EXP
        # Setting initial state for Fetch Move
        if self.type:
            names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint",
             "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
            names = ["robot0:" + n for n in names]
            values = [1.364, -0.294, -2.948, 0.906, -0.275, -1.206, 3.086]
            if self.type == 'reach_test':
                values[0] = -values[0]
            for i in range(len(names)):
                self.sim.data.set_joint_qpos(names[i], values[i])

        # Randomize start position of object.
        # if self.has_object:
            # object_xpos = self.initial_gripper_xpos[:2]
            # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            #     object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            # object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            # assert object_qpos.shape == (7,)
            # object_qpos[:2] = object_xpos
            # self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = np.array([1.45, 0.75, 0.42])
        return goal
        # if self.has_object:
        #     goal = self.initial_gripper_xpos[:3]
        #     goal[0] += 0.06
        #     goal += self.target_offset
        #     goal[2] = self.height_offset
        # else:
        #     goal = self.initial_gripper_xpos[:3]
        # return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.7, -0.3, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
