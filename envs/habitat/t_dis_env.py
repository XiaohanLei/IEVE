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
from scipy import ndimage
import torch
import cv2
from torchvision import transforms
from envs.utils.fmm_planner import FMMPlanner
from constants import coco_categories
import envs.utils.pose as pu
from utils.model import Classifier1
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image
from magnum import Quaternion, Vector3
import magnum as mn
import math
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch
import sys
sys.path.append("/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR")
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from sklearn.cluster import DBSCAN


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

        # config_file = '/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR/projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py'
        # checkpoint_file = '/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR/checkpoints/co_dino_5scale_swin_large_16e_o365tococo.pth'
        # self.codetr = init_detector(config_file, checkpoint_file, device=self.device)   
        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='disk').eval().to(self.device)

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

    def get_box_with_codetr(self, image, thresh=0.1):
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
        self.instance_imagegoal = obs['instance_imagegoal']
        self.update_after_reset()
        if new_scene:
            self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id
            print("Changing scene: {}/{}".format(self.rank, self.scene_name))

        agent_state = self._env.sim.get_agent_state(0).position
        self.start_height = agent_state[1]
        self.agent_height = self.args.camera_height
        self.info['agent_height'] = self.agent_height

        ###############################################
        # before start: make sure the instance class id
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
                transforms.ToTensor()(self.instance_imagegoal).to(self.device).unsqueeze(0)),
                    dim=1)
        self.goal_idx = goal_cat.item()
        self.info['goal_cat_id'] = self.gt_goal_idx
        self.info['instance_imagegoal'] = self.instance_imagegoal   

        sem_pred = getattr(self, '_get_sem_pred')     

        # instance_whwh = self.get_box_with_codetr(self.instance_imagegoal.astype(np.uint8), \
        #     0.1)
        instance_whwh = sem_pred(self.instance_imagegoal.astype(np.uint8), use_seg=True, pred_bbox=True)

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

        if self.gt_goal_idx == 0 or self.gt_goal_idx == 1:
            self.gt_goal_idx = self.goal_idx
        
        # else:
        #     self.goal_idx = 7

        #######################################

        # now during an episode, get multiple positve and negative observations of the instance goal

        #######################################

        # define some hyper parameters
        match_ratio = 1/500
        bbox_thersh = 1/6
        min_d = 2
        max_d = 4
        

        #######################################

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        # self._visualize_goal(obs['instance_imagegoal'].astype(np.uint8))

        scene = self._env.sim.semantic_annotations()
        self.goal_object_id = int(self._env.current_episode.goal_object_id)
        
        self.goal_pos = self._env.current_episode.goals[0].position
        start_height = self._env.current_episode.start_position[1]
        positive_pose = []
        positive_obs = []
        negative_pose = []
        negative_obs = []

        for i in range(200):
            temp_pos = self._env.sim.pathfinder.get_random_navigable_point()
            if ((((temp_pos[0]-self.goal_pos[0])**2+\
                (temp_pos[2]-self.goal_pos[2])**2)<max_d**2) and (((temp_pos[0]-self.goal_pos[0])**2+\
                (temp_pos[2]-self.goal_pos[2])**2)>min_d**2)) and abs(temp_pos[1]-start_height) < 0.3 :
                positive_pose.append(temp_pos)
        if positive_pose != []:
            # add a angle noise to the egocentric view
            angle_noise = np.random.randint(60, size=len(positive_pose)) - 30
            angle_noise = np.deg2rad(angle_noise)
            # memorize the index to del
            index_del = []

            for i in range(len(positive_pose)):
                source_position = np.array(positive_pose[i], dtype=np.float32)
                source_rotation = pu.quaternion_from_coeff(self._env.current_episode.start_rotation)
                goal_position = np.array(self.goal_pos)
                direction_vector = goal_position - source_position
                direction_vector_agent = pu.quaternion_rotate_vector(
                    source_rotation.inverse(), direction_vector
                )
                rho, phi = pu.cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                dist, angle = rho, -phi
                # add a noise
                angle += angle_noise[i]

                up_axis = mn.Vector3(0, 1, 0)
                st_rot = self._env.current_episode.start_rotation
                rot = pu.list2qua(st_rot) * \
                    mn.Quaternion.rotation(mn.Rad(angle), up_axis)
                self._env.sim.set_agent_state(positive_pose[i], pu.qua2list(rot))
                sim_obs = self._env.sim.get_sensor_observations()
                obs = self._env.sim._sensor_suite.get_observations(sim_obs)
                # check if agent can see the instance (pixel ratio > 0.01)
                semantic_obs = obs['semantic']
                counts = np.bincount(semantic_obs.flatten())
                total_count = np.sum(counts)
                sem_obj = scene.objects[self.goal_object_id]
                if self.goal_object_id < counts.shape[0]:
                    count = counts[self.goal_object_id]
                else:
                    count = 0
                pixel_ratio = count / total_count

                if pixel_ratio > 0.01:
                    # visulization:                
                    # self._visualize_rgb(obs['rgb'].astype(np.uint8))
                    # preprocess semantic obs
                    # fuck !!!!! should know that uint8 contains element from 0 to 255
                    # while uint16 or int 32 contains a larger range 
                    # 唯有屎才可以咏志

                    
                    # sem_seg_pred = sem_pred(obs['rgb'].astype(np.uint8))
                    # sem = sem_seg_pred[:, :, self.gt_goal_idx]
                    # mask_depth = sem * obs['depth'][:, :, 0]
                    # index = self.local_feature_match_lightglue(obs['rgb'].astype(np.uint8))
                    # self._visualize_mask(mask_depth, index.shape[0], True)

                    # sem = np.where(semantic_obs == self.goal_object_id, 1, 0)
                    
                    # self._visualize_semantic(semantic_obs)
                    

                    positive_obs.append(obs)
                    self.timestep += 1
                else:
                    index_del.append(i)

            positive_pose = [n for i, n in enumerate(positive_pose) if i not in index_del]


        self.timestep = 0

        ##########
        # for i in range(int(10*len(positive_pose))):
        #     temp_pos = self._env.sim.pathfinder.get_random_navigable_point()
        #     if  abs(temp_pos[1]-start_height) < 0.3 :
        #         negative_pose.append(temp_pos)
        # if negative_pose != []:
        #     # add a angle noise to the egocentric view
        #     angle_noise = np.random.randint(360, size=len(negative_pose)) - 180
        #     angle_noise = np.deg2rad(angle_noise)
        #     # memorize the index to del
        #     index_del = []
        #     for i in range(len(negative_pose)):
        #         source_position = np.array(negative_pose[i], dtype=np.float32)
        #         source_rotation = pu.quaternion_from_coeff(self._env.current_episode.start_rotation)
        #         goal_position = np.array(self.goal_pos)
        #         direction_vector = goal_position - source_position
        #         direction_vector_agent = pu.quaternion_rotate_vector(
        #             source_rotation.inverse(), direction_vector
        #         )
        #         rho, phi = pu.cartesian_to_polar(
        #             -direction_vector_agent[2], direction_vector_agent[0]
        #         )
        #         dist, angle = rho, -phi
        #         # add a noise
        #         angle += angle_noise[i]

        #         up_axis = mn.Vector3(0, 1, 0)
        #         st_rot = self._env.current_episode.start_rotation
        #         rot = pu.list2qua(st_rot) * \
        #             mn.Quaternion.rotation(mn.Rad(angle), up_axis)
        #         self._env.sim.set_agent_state(negative_pose[i], pu.qua2list(rot))
        #         sim_obs = self._env.sim.get_sensor_observations()
        #         obs = self._env.sim._sensor_suite.get_observations(sim_obs)
        #         # check if agent can see the instance (pixel ratio > 0.01)
        #         semantic_obs = obs['semantic']
        #         counts = np.bincount(semantic_obs.flatten())
        #         total_count = np.sum(counts)
        #         sem_obj = scene.objects[self.goal_object_id]
        #         if self.goal_object_id < counts.shape[0]:
        #             count = counts[self.goal_object_id]
        #         else:
        #             count = 0
        #         pixel_ratio = count / total_count

        #         if pixel_ratio < 0.001:


        #             # sem = np.where(semantic_obs == self.goal_object_id, self.goal_object_id, 0)
        #             # self._visualize_semantic(semantic_obs)

        #             # sem_seg_pred = sem_pred(obs['rgb'].astype(np.uint8))
        #             # sem = sem_seg_pred[:, :, self.gt_goal_idx]
        #             # mask_depth = sem * obs['depth'][:, :, 0]
        #             # index = self.local_feature_match_lightglue(obs['rgb'].astype(np.uint8))
        #             # self._visualize_mask(mask_depth, index.shape[0], False)

        #             negative_obs.append(obs)
        #             self.timestep += 1
        #         else:
        #             index_del.append(i)

        #     negative_pose = [n for i, n in enumerate(positive_pose) if i not in index_del]
        ##########



        for i in range(int(1.5*len(positive_pose))):
            temp_pos = self._env.sim.pathfinder.get_random_navigable_point()
            if  abs(temp_pos[1]-start_height) < 0.3 :
                negative_pose.append(temp_pos)
        if negative_pose != []:
            # add a angle noise to the egocentric view
            angle_noise = np.random.randint(360, size=len(negative_pose)) - 180
            angle_noise = np.deg2rad(angle_noise)
            # memorize the index to del
            index_del = []
            for i in range(len(negative_pose)):
                source_position = np.array(negative_pose[i], dtype=np.float32)
                source_rotation = pu.quaternion_from_coeff(self._env.current_episode.start_rotation)
                goal_position = np.array(self.goal_pos)
                direction_vector = goal_position - source_position
                direction_vector_agent = pu.quaternion_rotate_vector(
                    source_rotation.inverse(), direction_vector
                )
                rho, phi = pu.cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                dist, angle = rho, -phi
                # add a noise
                angle += angle_noise[i]

                up_axis = mn.Vector3(0, 1, 0)
                st_rot = self._env.current_episode.start_rotation
                rot = pu.list2qua(st_rot) * \
                    mn.Quaternion.rotation(mn.Rad(angle), up_axis)
                self._env.sim.set_agent_state(negative_pose[i], pu.qua2list(rot))
                sim_obs = self._env.sim.get_sensor_observations()
                obs = self._env.sim._sensor_suite.get_observations(sim_obs)
                # check if agent can see the instance (pixel ratio > 0.01)
                semantic_obs = obs['semantic']
                counts = np.bincount(semantic_obs.flatten())
                total_count = np.sum(counts)
                sem_obj = scene.objects[self.goal_object_id]
                if self.goal_object_id < counts.shape[0]:
                    count = counts[self.goal_object_id]
                else:
                    count = 0
                pixel_ratio = count / total_count

                if pixel_ratio < 0.001:
                    # visulization:                
                    # self._visualize_rgb(obs['rgb'].astype(np.uint8))
                    # preprocess semantic obs
                    # fuck !!!!! should know that uint8 contains element from 0 to 255
                    # while uint16 or int 32 contains a larger range 
                    # 唯有屎才可以咏志
                    # sem = np.where(semantic_obs == self.goal_object_id, self.goal_object_id, 0)
                    # self._visualize_semantic(semantic_obs)

                    # sem_seg_pred = sem_pred(obs['rgb'].astype(np.uint8))
                    # sem = sem_seg_pred[:, :, self.gt_goal_idx]
                    # mask_depth = sem * obs['depth'][:, :, 0]
                    # index = self.local_feature_match_lightglue(obs['rgb'].astype(np.uint8))
                    # self._visualize_mask(mask_depth, index.shape[0], False)

                    negative_obs.append(obs)
                    self.timestep += 1
                else:
                    index_del.append(i)

            negative_pose = [n for i, n in enumerate(positive_pose) if i not in index_del]

        #######################################

        # now, get the positive and negative observation, and predicted goal id, perform instance discriminator
        # first test our method, give three metrics(measured with times): except TP, FP, TN, FN, add two MP, MN

        ######################################

        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.MP = 0
        self.MN = 0

        for i in range(len(positive_obs)):
            # sign = self.instance_discriminator(positive_obs[i], bbox_thersh=bbox_thersh, match_ratio=match_ratio)
            sign = self.ins_dis_v4(positive_obs[i])
            if sign == 0:
                self.TP += 1
            elif sign == 1:
                self.MP += 1
            else:
                self.FP += 1
        for i in range(len(negative_obs)):
            # sign = self.instance_discriminator(negative_obs[i], bbox_thersh=bbox_thersh, match_ratio=match_ratio)
            sign = self.ins_dis_v4(negative_obs[i])
            if sign == 0:
                self.TN += 1
            elif sign == 1:
                self.MN += 1
            else:
                self.FN += 1
        


        self.global_step = 1
        torch.set_grad_enabled(True)
        

        return state, self.info

    def ins_dis_v4(self, obs):
        index = self.local_feature_match_lightglue(obs['rgb'].astype(np.uint8))
        if index.shape[0] > 40:
            return 0
        else:
            return 2

    def ins_dis_v3(self, obs):

        sem_pred = getattr(self, '_get_sem_pred')  
        seg_mask = sem_pred(obs['rgb'].astype(np.uint8), use_seg=True)
        index = self.local_feature_match_lightglue(obs['rgb'].astype(np.uint8))
        if np.any(seg_mask[:, :, self.goal_idx] > 0):
            process_depth = np.where(seg_mask[:, :, self.goal_idx] > 0, obs['depth'][:, :, 0], 1) * 30
            min_depth = process_depth.min()
            if min_depth < 3:
                if index.shape[0] > 60:
                    return 0
                return 2
            else:
                if index.shape[0] > 100:
                    return 0
                else: 
                    return 1
        else:
            if index.shape[0] > 100:
                return 0

            return 2



    def ins_dis_v2(self, obs):
        '''
        when the object is close to the agent, directly perform local feature matching,
        if the object is far from the agent e.g. 3m, if exits the same class of object, 
        give a maybe, if not, 
        remember if agent stays closer to the target, it may not see the 
        '''
        id_lo_whwh = self.get_box_with_codetr(obs['rgb'].astype(np.uint8), 0.1)
        id_lo_whwh_speci = [id_lo_whwh[i] for i in range(len(id_lo_whwh)) \
                if id_lo_whwh[i][0] == self.goal_idx]
        if id_lo_whwh_speci != []:
            # this means there at least exists maybe object
            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2]).astype(int)
            goal_mask = np.zeros((obs['rgb'].shape[0], obs['rgb'].shape[1]))
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.

            rgb_center = self.get_mask_center(goal_mask)
            # goal_dis = obs['depth'][int(rgb_center[0]), int(rgb_center[1]), 0] * 30. #m
            goal_dis = self.compute_ins_dis_v1(obs['depth'][:, :, 0]*30., whwh)

            if goal_dis < 3:
                index = self.local_feature_match_lightglue(obs['rgb'].astype(np.uint8))
                if index.shape[0] > 25:
                    return 0
                else:
                    return 2
            else:
                return 1

        else:
            # don't get the object's bounding box, directly give false
            return 2


    def ins_dis_v1(self, obs):
        '''
        only performs local feature matching when close
        e.g. dis = 2m
        '''
        index = self.local_feature_match_lightglue(obs['rgb'].astype(np.uint8))
        if index.shape[0] > 60:
            return 0
        else:
            return 2

    def compute_ins_dis_v1(self, depth, whwh):
        '''
        analyze the maxium depth points's pos
        make sure the object is within the range of 10m
        '''
        hist, bins = np.histogram(depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].flatten(), \
            bins=100,range=(0,10))
        max_index = np.argmax(hist)
        return bins[max_index]

    def compute_ins_dis(self, depth, whwh):
        '''
        params: depth image: (h, w), with value of meters
        whwh: boundingbox
        '''
        points = depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].reshape(-1, 1)

        # Create a DBSCAN object and fit it to your data
        dbscan = DBSCAN(eps=0.3, min_samples=100)
        dbscan.fit(points)

        # Retrieve the cluster labels
        cluster_labels = dbscan.labels_

        # Find the unique cluster labels (excluding noise points labeled as -1)
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]

        # Calculate the mean depth for each cluster
        cluster_means = []
        for label in unique_labels:
            cluster_points = points[cluster_labels == label]
            cluster_mean = np.mean(cluster_points)
            cluster_means.append(cluster_mean)

        return cluster_means[0]

    def instance_discriminator(self, obs, bbox_thersh=1/6, match_ratio=1/25):
        '''
        0:true, 1:maybe, 2:false
        logic: 
        if exits the same object:
            if big enough and matched: true return 0
            else: false return 2
            if not big enough : maybe return 1
        '''
        id_lo_whwh = self.get_box_with_codetr(obs['rgb'].astype(np.uint8), 0.1)
        id_lo_whwh_speci = [id_lo_whwh[i] for i in range(len(id_lo_whwh)) \
                if id_lo_whwh[i][0] == self.goal_idx]
        if id_lo_whwh_speci != []:
            # this means there at least exists maybe object
            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2]).astype(int)
            w, h = whwh[2]-whwh[0], whwh[3]-whwh[1]
            whwh = np.array([whwh[0]-w/2, whwh[1]-h/2, whwh[2]+w/2, whwh[3]+h/2])
            whwh = np.clip(whwh, 0, 1000)
            whwh = whwh.astype(int)
            # redefine goal mask
            goal_mask = np.zeros((obs['rgb'].shape[0], obs['rgb'].shape[1]))
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.
            if (whwh[3]-whwh[1])/goal_mask.shape[0] > bbox_thersh or (whwh[2]-whwh[0])/goal_mask.shape[1] > bbox_thersh:
                index = self.local_feature_match_lightglue(obs['rgb'].astype(np.uint8))
                match_points = self.count_overlap(goal_mask, index)
                if match_points > goal_mask.sum() * (match_ratio):
                    return 0
                else:
                    return 2
            else:
                return 1
        else:
            return 2

    def local_feature_match_lightglue(self, raw_rgb):
        with torch.set_grad_enabled(False):
            ob = numpy_image_to_torch(raw_rgb).to(self.device)
            gi = numpy_image_to_torch(self.instance_imagegoal).to(self.device)
            try:
                feats0, _, matches01  = match_pair(self.extractor, self.matcher, ob, gi
                    )
                # indices with shape (K, 2)
                matches = matches01['matches']
                # in case that the matches collapse make a check
                b = torch.nonzero(matches[..., 0] < 2047, as_tuple=False)
                c = torch.index_select(matches[..., 0], dim=0, index=b.squeeze())
                points0 = feats0['keypoints'][c]
                return points0.numpy()
            except:
                return np.zeros((1, 2))

    
    def get_mask_center(self, image_mask):
        '''
        image_mask: (N, N)
        '''
        center = ndimage.center_of_mass(image_mask)
        if np.isnan(center[0]) or np.isnan(center[1]):
            return None
        center = [int(center[0]), int(center[1])]
        return center



    def count_overlap(self, ma1, index):
        '''
        count index in mask
        '''
        ma2 = np.zeros_like(ma1,dtype=int)
        ma1[ma1 != 0] = 1
        index = index.astype(np.int16)
        ma2[index[:, 1], index[:, 0]] = 1
        ma = (ma1 + ma2) // 2

        # self.save_image(ma1, "t1")
        # self.save_image(ma2, "t2")

        return ma.sum()


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

        action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": np.array([0]),
                        "linear_velocity": np.array([0]),
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([1]),
                    },
                }
        obs, rew, done, _ = super().step(action)

        # obs, rew, done, _ = super().step( \
        #     list(self._task_config.habitat.task.actions)[action])

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        agent_state = self._env.sim.get_agent_state(0).position
        self.agent_height = self.args.camera_height + agent_state[1] - self.start_height
        self.info['agent_height'] = self.agent_height

        spl, success, dist = 0., 0., 0.
        if done:
            spl, success, dist, soft_spl = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success
            self.info['soft_spl'] = soft_spl
            self.info['geo_distance'] = self.episode_geo_distance
            self.info['euc_distance'] = self.episode_euc_distance
            self.info['TP'] = self.TP
            self.info['TN'] = self.TN
            self.info['MP'] = self.MP
            self.info['MN'] = self.MN
            self.info['FP'] = self.FP
            self.info['FN'] = self.FN

        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info['time'] = self.timestep

        return state, rew, done, self.info

    def _visualize_rgb(self, input):
        dump_dir = "{}/dump/{}/".format(self.dump_location,
                                        self.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        fn = '{}/episodes/thread_{}/eps_{}/{}-Vis-{}-rgb.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.episode_no, self.timestep)
        cv2.imwrite(fn, cv2.cvtColor(input, cv2.COLOR_RGB2BGR))

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

    def _visualize_semantic(self, semantic_obs):
        dump_dir = "{}/dump/{}/".format(self.dump_location,
                                        self.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
    
        fn = '{}/episodes/thread_{}/eps_{}/{}-Vis-{}-sem.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.episode_no, self.timestep)
        semantic_img.save(fn)

    def _visualize_mask(self, mask, points, sign):
        # mask should be 0, 1
        dump_dir = "{}/dump/{}/".format(self.dump_location,
                                        self.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        mask = (mask * 255).astype(int)
        if sign:
            fn = '{}/episodes/thread_{}/eps_{}/positive-{}-{}.png'.format(
                    dump_dir, self.rank, self.episode_no,
                     self.timestep, str(int(points )))
            cv2.imwrite(fn, mask)
        else:
            fn = '{}/episodes/thread_{}/eps_{}/negative-{}-{}.png'.format(
                    dump_dir, self.rank, self.episode_no,
                     self.timestep, str(int(points)))
            cv2.imwrite(fn, mask)

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

    
