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
import sys
sys.path.append("/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR")
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


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

        config_file = '/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR/projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py'
        checkpoint_file = '/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR/checkpoints/co_dino_5scale_swin_large_16e_o365tococo.pth'
        self.codetr = init_detector(config_file, checkpoint_file, device=self.device)   

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

    def get_box_with_codetr(self, image, thresh=0.5):
        result = inference_detector(self.codetr, image)
        re1 = []
        re1.append(result[56])
        re1.append(result[57])
        re1.append(result[58])
        re1.append(result[59])
        re1.append(result[61])
        re1.append(result[62])
        final = []
        for i in range(6):
            for j in range(re1[i].shape[0]):
                if re1[i][j, 4] > thresh:
                    final.append([i, re1[i][j, 4], re1[i][j, :4]])
        return final

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
        if new_scene:
            self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id
            print("Changing scene: {}/{}".format(self.rank, self.scene_name))

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id

        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()

        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['goal_cat_id'] = self.gt_goal_idx
        self.info['goal_name'] = self.goal_name

        torch.set_grad_enabled(False)
        _, goal_cat = torch.max(\
            self.classifier1(\
                transforms.ToTensor()(obs['instance_imagegoal']).to(self.device).unsqueeze(0)),
                    dim=1)
        self.goal_idx = goal_cat.item()
        self.info['goal_cat_id'] = self.gt_goal_idx
        self.info['instance_imagegoal'] = obs['instance_imagegoal']
        self.instance_imagegoal = obs['instance_imagegoal']
        print(f"rank:{self.rank}, episode:{self.episode_no}, cat_id:{self.gt_goal_idx}, cat_name:{self.goal_name}")

        instance_whwh = self.get_box_with_codetr(self.instance_imagegoal.astype(np.uint8), \
            0.1)
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
                self.goal_idx = int(ins_whwh[0][0])
        
        # else:
        #     self.goal_idx = 7


        self.global_step = 1
        torch.set_grad_enabled(True)
        

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
        

        action = action["action"]
        # action = 1
        if action == 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            # action = 3

        obs, rew, done, _ = super().step( \
            list(self._task_config.habitat.task.actions)[action])

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
        return (0., 1.0)

    def get_reward(self, observations):
        return 0
        curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
        self.curr_distance = self.gt_planner.fmm_dist[curr_loc[0],
                                                      curr_loc[1]] / 20.0

        reward = (self.prev_distance - self.curr_distance) * \
            self.args.reward_coeff

        self.prev_distance = self.curr_distance
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
        if self.goal_idx == self.gt_goal_idx :
            success = 1
        elif (self.gt_goal_idx == 0 or self.gt_goal_idx == 1) and (self.goal_idx == 0 or self.goal_idx == 1):
            success = 1
        else:
            success = 0



        # print(self.habitat_env.get_metrics())
        # return 0, 0, 0
        # curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
        # dist = self.gt_planner.fmm_dist[curr_loc[0], curr_loc[1]] / 20.0
        # if dist == 0.0:
        #     success = 1
        # else:
        #     success = 0
        # spl = min(success * self.starting_distance / self.path_length, 1)
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

    
