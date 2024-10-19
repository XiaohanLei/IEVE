import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat
import os
import torch
import cv2
from torchvision import transforms
from envs.utils.fmm_planner import FMMPlanner
from constants import coco_categories
import envs.utils.pose as pu
from utils.model import Classifier1
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import envs.utils.depth_utils as du
import copy



class ObjectGoal_Env(habitat.RLEnv):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        super().__init__(config_env, dataset)
        self.args = args
        self.rank = rank
        self._task_config = config_env

        # Loading dataset info file
        self.split = config_env.habitat.dataset.split
        self.device = torch.device("cuda",  \
            int(config_env.habitat.simulator.habitat_sim_v0.gpu_device_id))
        # self.device = torch.device("cuda", 0)
        # self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(
        #     split=self.split)
        self.episodes_dir = os.path.join("data/datasets/instance_imagenav/hm3d/v3", self.split)
        # dataset_info_file = self.episodes_dir + \
        #     "{split}_info.pbz2".format(split=self.split)
        # with bz2.BZ2File(dataset_info_file, 'rb') as f:
        #     self.dataset_info = cPickle.load(f)

        # Specifying action and observation space
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                 args.frame_width),
                                                dtype='uint8')

        # Initializations
        self.episode_no = 0
        self.dump_location = "tmp"
        self.exp_name = "exp1"

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # Episode Dataset info
        self.eps_data = None
        self.eps_data_idx = None
        self.gt_planner = None
        self.object_boundary = None
        self.goal_idx = None
        self.goal_name = None
        self.map_obj_origin = None
        self.starting_loc = None
        self.starting_distance = None

        # Episode tracking info
        self.curr_distance = None
        self.prev_distance = None
        self.timestep = None
        self.stopped = None
        self.path_length = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None

        self.classifier1 = Classifier1().to(self.device)
        state_dict = torch.load("pretrained_models/cla3_model.pth",
                        map_location=lambda storage, loc: storage)
        self.classifier1.load_state_dict(state_dict)
        self.classifier1.eval()

        self.rgb_frames = []
        self.depth_frames = []
        self.transform_frames = []

        camera_matrix = du.get_camera_matrix(
            args.env_frame_width, args.env_frame_height, args.hfov)
        self.data_base = {
            "camera_model": "OPENCV_FISHEYE",
            "fl_x": camera_matrix.f,
            "fl_y": camera_matrix.f,
            "cx": camera_matrix.xc,
            "cy": camera_matrix.zc,
            "w": args.env_frame_width,
            "h": args.env_frame_height,
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,
            "p2": 0.0 
        }


    def load_new_episode(self):
        """The function loads a fixed episode from the episode dataset. This
        function is used for evaluating a trained model on the val split.
        """

        args = self.args
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        if self.scene_path != self.last_scene_path:
            episodes_file = self.episodes_dir + \
                "/content/{}.json.gz".format(scene_name)

            print("Loading episodes from: {}".format(episodes_file))
            with gzip.open(episodes_file, 'r') as f:
                self.eps_data = json.loads(
                    f.read().decode('utf-8'))["episodes"]

            self.eps_data_idx = 0
            self.last_scene_path = self.scene_path

        # Load episode info
        episode = self.eps_data[self.eps_data_idx]
        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)
        pos = episode["start_position"]
        rot = quaternion.from_float_array(episode["start_rotation"])

        self.episode_geo_distance = episode["info"]["geodesic_distance"]
        self.episode_euc_distance = episode["info"]["euclidean_distance"]

        goal_name = episode["object_category"]
        goal_idx = episode["goal_object_id"]

        self.goal_idx = 0
        self.gt_goal_idx = goal_idx
        self.goal_name = goal_name

        self._env.sim.set_agent_state(pos, rot)

        obs = self._env.sim.get_observations_at(pos, rot)

        return obs


    def update_after_reset(self):
        name2index = {
            "chair": 0,
            "sofa": 1,
            "plant": 2,
            "bed": 3,
            "toilet": 4,
            "tv_monitor": 5,
        }
        args = self.args

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        if self.scene_path != self.last_scene_path:
            episodes_file = self.episodes_dir + \
                "/content/{}.json.gz".format(scene_name)

            print("Loading episodes from: {}".format(episodes_file))
            with gzip.open(episodes_file, 'r') as f:
                self.eps_data = json.loads(
                    f.read().decode('utf-8'))["episodes"]

            self.eps_data_idx = 0
            self.last_scene_path = self.scene_path

        # Load episode info
        episode = self.eps_data[self.eps_data_idx]
        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)

        self.episode_geo_distance = episode["info"]["geodesic_distance"]
        self.episode_euc_distance = episode["info"]["euclidean_distance"]

        goal_name = episode["object_category"]
        goal_idx = episode["goal_object_id"]

        self.goal_idx = 0
        self.goal_name = goal_name
        self.gt_goal_idx = name2index[goal_name]
        self.goal_object_id = int(self._env.current_episode.goal_object_id)
        


    def sim_map_to_sim_continuous(self, coords):
        """Converts ground-truth 2D Map coordinates to absolute Habitat
        simulator position and rotation.
        """
        agent_state = self._env.sim.get_agent_state(0)
        y, x = coords
        min_x, min_y = self.map_obj_origin / 100.0

        cont_x = x / 20. + min_x
        cont_y = y / 20. + min_y
        agent_state.position[0] = cont_y
        agent_state.position[2] = cont_x

        rotation = agent_state.rotation
        rvec = quaternion.as_rotation_vector(rotation)

        if self.args.train_single_eps:
            rvec[1] = 0.0
        else:
            rvec[1] = np.random.rand() * 2 * np.pi
        rot = quaternion.from_rotation_vector(rvec)

        return agent_state.position, rot

    def sim_continuous_to_sim_map(self, sim_loc):
        """Converts absolute Habitat simulator pose to ground-truth 2D Map
        coordinates.
        """
        x, y, o = sim_loc
        min_x, min_y = self.map_obj_origin / 100.0
        x, y = int((-x - min_x) * 20.), int((-y - min_y) * 20.)

        o = np.rad2deg(o) + 180.0
        return y, x, o

    def get_goal_id(self):

        # call child function
        get_sem_pred = getattr(self, '_get_sem_pred')
        # get_sem_pred = getattr(self, '_get_sem_pred_internimg')
        instance_whwh = get_sem_pred(self.instance_imagegoal.astype(np.uint8), None, pred_bbox=True)
        ins_whwh = [instance_whwh[i] for i in range(len(instance_whwh)) \
            if (instance_whwh[i][2][3]-instance_whwh[i][2][1])>1/6*self.instance_imagegoal.shape[0] or \
                (instance_whwh[i][2][2]-instance_whwh[i][2][0])>1/6*self.instance_imagegoal.shape[1]]
        if ins_whwh != []:
            ins_whwh = sorted(ins_whwh,  \
                key=lambda s: ((s[2][0]+s[2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                    +((s[2][1]+s[2][3]-self.instance_imagegoal.shape[0])/2)**2 \
                )
            if ((ins_whwh[0][2][0]+ins_whwh[0][2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                    +((ins_whwh[0][2][1]+ins_whwh[0][2][3]-self.instance_imagegoal.shape[0])/2)**2 < \
                        ((self.instance_imagegoal.shape[1] / 6)**2 )*2:
                return int(ins_whwh[0][0])
                # self.goal_idx = int(ins_whwh[0][0])
        return None


    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args
        self.global_step = 0
        new_scene = self.episode_no % args.num_train_episodes == 0



        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []
       
        obs = super().reset()
        self.update_after_reset()
        if 'semantic' in obs:
            semantic_obs = obs['semantic']
            sem = np.where(semantic_obs == self.goal_object_id, 1, 0)
            self.semantic_obs = sem
            self.sign = np.any(sem > 0)

        if new_scene:
            self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id
            print("Changing scene: {}/{}".format(self.rank, self.scene_name))

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id

        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()
        # upstair or downstair check
        # self.start_height = self._env.current_episode.start_position[1]
        agent_state = self._env.sim.get_agent_state(0).position
        self.start_height = agent_state[1]
        self.agent_height = self.args.camera_height

        self.start_position = self._env.sim.get_agent_state(0).position
        self.start_rotation = self._env.sim.get_agent_state(0).rotation
        self.transform_matrix = self.get_transformation_matrix()

            
        torch.set_grad_enabled(False)


        self.info['goal_cat_id'] = self.gt_goal_idx
        self.info['instance_imagegoal'] = obs['instance_imagegoal']
        self.instance_imagegoal = obs['instance_imagegoal']


        idx = self.get_goal_id()
        if idx is not None:
            self.gt_goal_idx = idx

        print(f"rank:{self.rank}, episode:{self.episode_no}, cat_id:{self.gt_goal_idx}, cat_name:{self.goal_name}")
        torch.set_grad_enabled(True)

        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['goal_cat_id'] = self.gt_goal_idx
        self.info['goal_name'] = self.goal_name
        self.info['agent_height'] = self.agent_height
        

        return state, self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        

        if action['action_args']['velocity_stop'] > 0:
            self.stopped = True

        obs, rew, done, _ = super().step(action)
        self.transform_matrix = self.get_transformation_matrix()
    
        if 'semantic' in obs:
            semantic_obs = obs['semantic']
            sem = np.where(semantic_obs == self.goal_object_id, 1, 0)
            self.semantic_obs = sem
            self.sign = np.any(sem > 0)

        agent_state = self._env.sim.get_agent_state(0).position
        self.agent_height = self.args.camera_height + agent_state[1] - self.start_height
        self.info['agent_height'] = self.agent_height

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        spl, success, dist = 0., 0., 0.
        if done:
            spl, success, dist, soft_spl = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success
            self.info['soft_spl'] = soft_spl
            self.info['geo_distance'] = self.episode_geo_distance
            self.info['euc_distance'] = self.episode_euc_distance

        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info['time'] = self.timestep

        return state, rew, done, self.info

    def _visualize_goal(self, input):
        dump_dir = "{}/dump/{}/".format(self.dump_location,
                                        self.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        fn = '{}/episodes/thread_{}/eps_{}/{}-Vis-goal.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.episode_no)
        cv2.imwrite(fn, cv2.cvtColor(input, cv2.COLOR_RGB2BGR))

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 10.0)

    def get_reward(self, observations):
        _, s, d, _ = self.get_metrics()
        if d > 6. :
            d = 6.
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        reward =  10. * s
        
        return reward

    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        metrics = self.habitat_env.get_metrics()
        spl, success, dist = metrics['spl'], metrics['success'], metrics['distance_to_goal']
        soft_spl = metrics['soft_spl']
        return spl, success, dist, soft_spl

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

    def get_spaces(self):
        """Returns observation and action spaces for the ObjectGoal task."""
        return self.observation_space, self.action_space

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def save_data_nerf(self, state):
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3]
        depth = state[:, :, 3]
        depth = (depth * 30000.).astype(np.uint16)


        ep_dir_rgb = '{}/episodes/thread_{}/eps_{}/images/'.format(
            'nerf', self.rank, self.episode_no)
        ep_dir_depth = '{}/episodes/thread_{}/eps_{}/depth/'.format(
            'nerf', self.rank, self.episode_no)
        if not os.path.exists(ep_dir_rgb):
            os.makedirs(ep_dir_rgb)
        if not os.path.exists(ep_dir_depth):
            os.makedirs(ep_dir_depth)
        fn_rgb = '{}frame_{:05d}.jpeg'.format(
                ep_dir_rgb,
                self.timestep)
        fn_depth = '{}{:04d}.png'.format(
                ep_dir_depth,
                self.timestep)

        cv2.imwrite(fn_rgb, rgb[:, :, ::-1])
        cv2.imwrite(fn_depth, depth)

        self.rgb_frames.append('images/frame_{:05d}.jpeg'.format(self.timestep))
        self.depth_frames.append('depth/{:04d}.png'.format(self.timestep))
        self.transform_frames.append(self.transform_matrix)


    def get_transformation_matrix(self):
        initial_position = self.start_position
        rotation_quaternion = self.start_rotation
        current_position = self._env.sim.get_agent_state(0).position
        current_rotation = self._env.sim.get_agent_state(0).rotation

        # Convert the rotation quaternion to a rotation matrix
        initial_rotation_matrix = quaternion.as_rotation_matrix(rotation_quaternion)
        current_rotation_matrix = quaternion.as_rotation_matrix(current_rotation)

        # Create a 4x4 transformation matrix from the rotation matrix and translation vector
        initial_transform = np.eye(4)
        initial_transform[:3, :3] = initial_rotation_matrix
        initial_transform[:3, 3] = initial_position

        current_transform = np.eye(4)
        current_transform[:3, :3] = current_rotation_matrix
        current_transform[:3, 3] = current_position

        # Compute the inverse of the initial transformation matrix
        initial_transform_inverse = np.linalg.inv(initial_transform)

        # Compute the relative transformation matrix
        relative_transform = np.dot(initial_transform_inverse, current_transform)

        return relative_transform


    
