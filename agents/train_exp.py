import warnings
warnings.filterwarnings('ignore')
import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
import math
import torch
from torchvision import transforms
import torch.nn.functional as F
from scipy import ndimage
from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.imagegoal_env import ObjectGoal_Env
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from agents.utils.count import counter, MaskDINO
from constants import color_palette
import envs.utils.pose as pu
import agents.utils.visualization as vu
from mcc_utils.sam_utils import load_model_hf, find_the_centered_box
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, predict
from segment_anything import build_sam, SamPredictor 
from huggingface_hub import hf_hub_download
from kornia.feature import LoFTR
from collections import Counter
import logging
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib
from groundingdino.util.inference import load_model, load_image, predict, annotate
import sys
sys.path.append("/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR")
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from projects import *
import wandb
from skimage.draw import line

class Instance_Exp_Env_Agent(ObjectGoal_Env):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):
        # import os
        # os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
        self.args = args
        super().__init__(args, rank, config_env, dataset)
        

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.habitat.simulator.habitat_sim_v0.gpu_device_id

        self.sem_pred = SemanticPredMaskRCNN(args)

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.instance_imagegoal = None

        # self.loftr = LoFTR('indoor').to(self.device)
        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='disk').eval().to(self.device)

        map_size = self.args.map_size_cm // self.args.map_resolution
        full_w, full_h = map_size, map_size
        self.local_w = int(full_w / self.args.global_downscaling)
        self.local_h = int(full_h / self.args.global_downscaling)
        self.global_goal = None
        # define a temporal goal with a living time
        self.temp_goal = None
        self.last_temp_goal = None # avoid choose one goal twice
        self.forbidden_temp_goal = []
        self.flag = 0
        self.goal_instance_whwh = None
        # define untraversible area of the goal: 0 means area can be goals, 1 means cannot be
        self.goal_map_mask = np.zeros((full_w, full_h))
        self.pred_box = []
        torch.set_grad_enabled(False)

        # config_file = '/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR/projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py'
        # checkpoint_file = '/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR/checkpoints/co_dino_5scale_swin_large_16e_o365tococo.pth'
        # self.codetr = init_detector(config_file, checkpoint_file, device=self.device)   
 

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None

    def reset(self):
        args = self.args

        obs, info = super().reset()
        self.raw_obs = obs[:3, :, :].transpose(1, 2, 0)
        obs = self._preprocess_obs(obs)
        self.obs = obs

        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.global_goal = None
        self.temp_goal = None
        self.last_temp_goal = None
        self.forbidden_temp_goal = []
        self.goal_map_mask = np.zeros(map_shape)
        self.goal_instance_whwh = None
        self.pred_box = []

        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        return obs, info

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

    def local_feature_match_lightglue(self):
        with torch.set_grad_enabled(False):
            ob = numpy_image_to_torch(self.raw_obs).to(self.device)
            gi = numpy_image_to_torch(self.instance_imagegoal).to(self.device)
            try:
                feats0, _, matches01  = match_pair(self.extractor, self.matcher, ob, gi
                    )
                # indices with shape (K, 2)
                matches = matches01['matches']
                # in case that the matches collapse make a check
                b = torch.nonzero(matches[..., 0] < 2048, as_tuple=False)
                c = torch.index_select(matches[..., 0], dim=0, index=b.squeeze())
                points0 = feats0['keypoints'][c].numpy()  
                points0[:, 0] = np.clip(points0[:, 0], 0, self.raw_obs.shape[1]-1)
                points0[:, 1] = np.clip(points0[:, 1], 0, self.raw_obs.shape[0]-1)
                return points0 
            except:
                return np.zeros((1, 2))


    def local_feature_match(self):
        # with torch.set_grad_enabled(False):
        ob = numpy_image_to_torch(self.raw_obs).to(self.device)
        gi = numpy_image_to_torch(self.instance_imagegoal).to(self.device)
        input1 = transforms.functional.rgb_to_grayscale(ob)
        input2 = transforms.functional.rgb_to_grayscale(gi)
        inputs = {
            "image0": input1.unsqueeze(0),
            "image1": input2.unsqueeze(0)
        }
        out = self.loftr(inputs)
        points0 = out['keypoints0'].cpu()
        # torch.set_grad_enabled(True)
        return points0.numpy()

    def get_mask_center(self, image_mask):
        '''
        image_mask: (N, N)
        '''
        center = ndimage.center_of_mass(image_mask)
        if np.isnan(center[0]) or np.isnan(center[1]):
            return None
        center = [int(center[0]), int(center[1])]
        return center

    def save_image(self, image, name):
        '''
        only supports save grayscale
        '''
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        fn1 = '{}/episodes/thread_{}/eps_{}/{}_{}.png'.format(
                dump_dir, self.rank, self.episode_no, name, self.timestep)
        cv2.imwrite(fn1, image*255)
    
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

    def check_if_mask_big_enough(self, goal_mask, f):
        '''
        params : goal_mask: (H, W)
                f : thresh ratio, should less than 1 and greater than 0
        '''
        return goal_mask.sum() > goal_mask.shape[0] * goal_mask.shape[1] * ((f)**2)

    def compute_ins_dis_v1(self, depth, whwh):
        '''
        analyze the maxium depth points's pos
        make sure the object is within the range of 10m
        '''
        hist, bins = np.histogram(depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].flatten(), \
            bins=100,range=(0,1000))
        max_index = np.argmax(hist)
        return bins[max_index]


    def compute_object_distance(self, goal_mask, index, rgb_center):
        ma2 = np.zeros_like(goal_mask,dtype=int)
        goal_mask[goal_mask != 0] = 1
        index = index.astype(np.int16)
        ma2[index[:, 1], index[:, 0]] = 1
        maa = (goal_mask.astype(int) + ma2) // 2
        # if maa.sum() != 0:
        if False:
            masked_depth = ma.masked_array(self.obs[3, :, :], maa == 1.) / self.args.map_resolution
            goal_dis = np.clip(masked_depth, 0, 180).mean()
        else: 
            goal_dis = self.obs[3:4, :, :][0, int(rgb_center[0]), int(rgb_center[1])] / self.args.map_resolution
            # hist, bins = np.histogram(depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].flatten(), \
            #     bins=100,range=(0,10))
            # max_index = np.argmax(hist)
            # return bins[max_index]
        return goal_dis

    def instance_discriminator(self, planner_inputs, id_lo_whwh_speci):
        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        map_pred = np.rint(planner_inputs['map_pred'])
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        goal_mask = self.obs[4+self.gt_goal_idx, :, :]
        # ds = 4
        # goal_mask = goal_mask[:, :, self.gt_goal_idx][ds // 2::ds, ds // 2::ds]

        # try:
        # now the logic here should be termed as get closer enough to check

        if self.instance_imagegoal is None:
            # not initialized
            return planner_inputs
        elif self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal = [self.global_goal[0] - gx1, self.global_goal[1] - gy1]
            goal = pu.threshold_poses(goal, map_pred.shape)
            planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
            planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
            return planner_inputs
        elif planner_inputs['found_goal'] == 1:

            # preprocess foundation model output
            # this order is not correct, we should follow the order of from near to further
            # id_lo_whwh_speci = sorted(id_lo_whwh_speci, key=lambda s: s[1], reverse=True)
            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2] / 4).astype(int)
            w, h = whwh[2]-whwh[0], whwh[3]-whwh[1]
            goal_mask = np.zeros_like(goal_mask)
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.

            # index = self.local_feature_match()
            index = self.local_feature_match_lightglue()
            match_points = index.shape[0]
            # if self.rank == 3:
            #     print(f"Rank: {self.rank}, timestep: {self.timestep},  matched points {match_points} ")

            if self.temp_goal is not None:
                goal_dis = pu.get_l2_distance(self.temp_goal[0]-gx1, start[0], self.temp_goal[1]-gy1, start[1])
                goal = np.array([self.temp_goal[0]-gx1, self.temp_goal[1]-gy1])
                goal = pu.threshold_poses(goal, map_pred.shape)
                planner_inputs['found_goal'] = 0
            else:
                goal_dis = self.compute_ins_dis_v1(self.obs[3, :, :], whwh) / self.args.map_resolution
                rgb_center = np.array([whwh[3]+whwh[1], whwh[2]+whwh[0]])//2
                goal_angle = -self.args.hfov / 2 * (rgb_center[1] - self.obs.shape[2]/2) \
                / (self.obs.shape[2]/2)
                goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                    start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
                goal = pu.threshold_poses(goal, map_pred.shape)
                planner_inputs['found_goal'] = 0
            
            if goal_dis < 200 / self.args.map_resolution:

                if match_points > 60:

                    planner_inputs['found_goal'] = 1
                    planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
                    planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
                    self.global_goal = [int(goal[0])+gx1, int(goal[1])+gy1]
                    self.temp_goal = None
                    # print(f"Rank: {self.rank}, timestep: {self.timestep},  0 !")

                else:
                    planner_inputs['goal'] = planner_inputs['exp_goal']
                    self.forbidden_temp_goal.append([int(goal[0])+gx1, int(goal[1])+gy1])

                    # print(f"Rank: {self.rank}, timestep: {self.timestep},  1 !")
            else:

                ttt = [self.forbidden_temp_goal[i] for i in range(len(self.forbidden_temp_goal)) if \
                    pu.get_l2_distance(int(goal[0])+gx1, self.forbidden_temp_goal[i][0], \
                    int(goal[1])+gy1, self.forbidden_temp_goal[i][1]) < 5]  
                if ttt != []:
                    self.temp_goal = None
                    planner_inputs['goal'] = planner_inputs['exp_goal']
                else:
                    planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
                    planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
                    self.temp_goal = [int(goal[0])+gx1, int(goal[1])+gy1]

                # print(f"Rank: {self.rank}, timestep: {self.timestep},  2 !")
            return planner_inputs
        else:
            if self.temp_goal is not None:  
                goal = np.array([self.temp_goal[0]-gx1, self.temp_goal[1]-gy1])
                goal = pu.threshold_poses(goal, map_pred.shape)
                planner_inputs['found_goal'] = 0
                ttt = [self.forbidden_temp_goal[i] for i in range(len(self.forbidden_temp_goal)) if \
                    pu.get_l2_distance(int(goal[0])+gx1, self.forbidden_temp_goal[i][0], \
                    int(goal[1])+gy1, self.forbidden_temp_goal[i][1]) < 5]  
                if ttt != []:
                    self.temp_goal = None
                    planner_inputs['goal'] = planner_inputs['exp_goal']
                else:               
                    goal = [self.temp_goal[0] - gx1, self.temp_goal[1] - gy1]
                    goal = pu.threshold_poses(goal, map_pred.shape)
                    planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
                    planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1

            # print(f"Rank: {self.rank}, timestep: {self.timestep},  3 !")
            return planner_inputs


    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs.shape), 0., False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0

        # recompute found_goal 
        # planner_inputs["found_goal"] = np.any(self.obs[4+self.gt_goal_idx, :, :])

        # id_lo_whwh = self.get_box_with_codetr(self.raw_obs.astype(np.uint8), 0.6)
        id_lo_whwh = self.pred_box
        if self.args.print_images and id_lo_whwh != []:
            for i in range(len(id_lo_whwh)):
                id_lo_whwh[i][2] = id_lo_whwh[i][2].astype(int)
                self.rgb_vis = self.rgb_vis.astype(np.uint8)
                cv2.rectangle(self.rgb_vis, (id_lo_whwh[i][2][0], id_lo_whwh[i][2][1]), \
                    (id_lo_whwh[i][2][2], id_lo_whwh[i][2][3]), (0, 255, 0), 2)  
            self.rgb_vis.astype(np.uint8)

        id_lo_whwh_speci = [id_lo_whwh[i] for i in range(len(id_lo_whwh)) \
                    if id_lo_whwh[i][0] == self.gt_goal_idx]

        planner_inputs["found_goal"] = (id_lo_whwh_speci != [])

        self.instance_discriminator(planner_inputs, id_lo_whwh_speci)
        action = self._plan(planner_inputs)

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)

        # due to bad annotation by the dataset, auto reset the gt_goal_cat
        if self.timestep == 1:
            # instance_whwh = self.get_box_with_codetr(self.instance_imagegoal.astype(np.uint8), \
            #     0.1)
            if self.gt_goal_idx == 0 or self.gt_goal_idx == 1:
                _, _, instance_whwh = self.sem_pred.get_prediction(self.instance_imagegoal.astype(np.uint8))
                ins_whwh = [instance_whwh[i] for i in range(len(instance_whwh)) \
                    if (instance_whwh[i][2][3]-instance_whwh[i][2][1])>1/5*self.instance_imagegoal.shape[0] or \
                        (instance_whwh[i][2][2]-instance_whwh[i][2][0])>1/5*self.instance_imagegoal.shape[1]]
                if ins_whwh != []:
                    ins_whwh = sorted(ins_whwh,  \
                        key=lambda s: ((s[2][0]+s[2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                            +((s[2][1]+s[2][3]-self.instance_imagegoal.shape[0])/2)**2 \
                        )
                    self.gt_goal_idx = int(ins_whwh[0][0])
                # self.goal_instance_whwh = ins_whwh[0][2]

        if action == 0 and planner_inputs["found_goal"] == 0:
            print(f"Rank: {self.rank}, timestep: {self.timestep},  stoped early, FATAL ERROR !")

        if action >= 0:

            # act
            action = {'action': action}
            obs, rew, done, info = super().step(action)
            self.raw_obs = obs[:3, :, :].transpose(1, 2, 0)

            # preprocess obs
            obs = self._preprocess_obs(obs) 
            self.last_action = action['action']
            self.obs = obs
            self.info = info

            info['g_reward'] += rew

            return obs, rew, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs_shape), 0., False, self.info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)\
        
        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        # self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
        #                                start[1] - 0:start[1] + 1] = 1
        rr, cc = line(last_start[0], last_start[1], start[0], start[1])
        self.visited[gx1:gx2, gy1:gy2][rr, cc] = 1

        if args.visualize or args.print_images:            
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 9:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 7)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)
        # if self.global_goal is not None:
        #     stg = (self.global_goal[0]-gx1, self.global_goal[1]-gy1)

        # Deterministic Local Policy
        if stop and planner_inputs['found_goal'] == 1:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        # # dilate traversible area to prevent stuck
        # selem = skimage.morphology.disk(1)
        # traversible = skimage.morphology.binary_dilation(
        #     traversible,
        #     selem)

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        if self.global_goal is None:
            selem = skimage.morphology.disk(10)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        else:
            selem = skimage.morphology.disk(14)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def get_frontier_map(self, planner_inputs):
        """Function responsible for computing frontiers in the input map

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obs_map' (ndarray): (M, M) map of obstacle locations
                    'exp_map' (ndarray): (M, M) map of explored locations

        Returns:
            frontier_map (ndarray): (M, M) binary map of frontier locations
        """
        args = self.args

        obs_map = np.rint(planner_inputs["obs_map"])
        exp_map = np.rint(planner_inputs["exp_map"])
        # compute free and unexplored maps
        free_map = (1 - obs_map) * exp_map
        unk_map = 1 - exp_map
        # Clean maps
        kernel = np.ones((5, 5), dtype=np.uint8)
        free_map = cv2.morphologyEx(free_map, cv2.MORPH_CLOSE, kernel)
        unk_map[free_map == 1] = 0
        unk_map_shiftup = np.pad(
            unk_map, ((0, 1), (0, 0)), mode="constant", constant_values=0
        )[1:, :]
        unk_map_shiftdown = np.pad(
            unk_map, ((1, 0), (0, 0)), mode="constant", constant_values=0
        )[:-1, :]
        unk_map_shiftleft = np.pad(
            unk_map, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )[:, 1:]
        unk_map_shiftright = np.pad(
            unk_map, ((0, 0), (1, 0)), mode="constant", constant_values=0
        )[:, :-1]
        frontiers = (
            (free_map == unk_map_shiftup)
            | (free_map == unk_map_shiftdown)
            | (free_map == unk_map_shiftleft)
            | (free_map == unk_map_shiftright)
        ) & (
            free_map == 1
        )  # (H, W)
        frontiers = frontiers.astype(np.uint8)
        # Select only large-enough frontiers
        contours, _ = cv2.findContours(
            frontiers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) > 0:
            contours = [c[:, 0].tolist() for c in contours]  # Clean format
            new_frontiers = np.zeros_like(frontiers)  
            # Only pick largest 1 frontiers
            contours = sorted(contours, key=lambda x: len(x), reverse=True)
            for contour in contours[:1]:
                contour = np.array(contour)
                # Select only the central point of the contour
                lc = len(contour)
                if lc > 0:
                    new_frontiers[contour[lc // 2, 1], contour[lc // 2, 0]] = 1
            frontiers = new_frontiers
        frontiers = frontiers > 0
        # Mask out frontiers very close to the agent
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        ## Convert current location to map coordinates
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, frontiers.shape)
        ## Mask out a 100.0 x 100.0 cm region center on the agent
        ncells = int(100.0 / args.map_resolution)
        frontiers[
            (start[0] - ncells) : (start[0] + ncells + 1),
            (start[1] - ncells) : (start[1] + ncells + 1),
        ] = False
        # Handle edge case where frontier becomes zero
        if not np.any(frontiers):
            # Set a random location to True
            rand_y = np.random.randint(start[0] - ncells, start[0] + ncells + 1)
            rand_x = np.random.randint(start[1] - ncells, start[1] + ncells + 1)
            frontiers[rand_y, rand_x] = True

        return frontiers

    def get_fmm_dists(self, planner_inputs):
        """Function responsible for planning, and identifying reachable locations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'pred_map'   (ndarray): (N, H, W) map with 0 as floor, 1 - N as categories
                    'map_resolution' (int): size of grid-cell in pred_map

        Returns:
            fmm_dists (ndarray): (N, H, W) map of FMM dists per category
        """
        pred_map = planner_inputs["pred_map"]
        fmm_dists = np.zeros_like(pred_map)
        fmm_dists.fill(np.inf)
        map_resolution = planner_inputs["map_resolution"]
        orig_map_resolution = self.args.map_resolution
        assert orig_map_resolution == map_resolution

        # Setup planner
        traversible = pred_map[0]
        planner = FMMPlanner(traversible)
        # Get FMM dists to each category
        selem = skimage.morphology.disk(
            int(self.object_boundary / 4.0 * 100.0 / self.args.map_resolution)
        )
        for i in range(1, fmm_dists.shape[0]):
            if np.count_nonzero(pred_map[i]) == 0:
                continue
            goal_map = cv2.dilate(pred_map[i], selem)
            planner.set_multi_goal(goal_map)
            fmm_dist = planner.fmm_dist * map_resolution / 100.0
            fmm_dists[i] = fmm_dist

        return fmm_dists

    def _get_reachability(self, grid, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        # Note: Unlike _get_stg, no boundary is added here since we only want
        # to determine reachability.
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(3)
        goal = cv2.dilate(goal, selem)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist * self.args.map_resolution / 100.0

        reachability = fmm_dist < fmm_dist.max()

        return reachability.astype(np.float32), fmm_dist.astype(np.float32)

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        sem_seg_pred = self._get_sem_pred(
            rgb.astype(np.uint8), use_seg=use_seg)
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)
        index = self.local_feature_match_lightglue()
        index = index.astype(np.int16)
        sem_seg_pred[index[:, 1], index[:, 0], -1] = 1

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def _get_sem_pred(self, rgb, use_seg=True):
        if use_seg:
            semantic_pred, self.rgb_vis, self.pred_box = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return semantic_pred

    def _visualize(self, inputs):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        # add a check with collision map
        map_pred[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 1

        sem_map += 5

        no_cat_mask = sem_map == 11
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        tmp = cv2.resize(self.rgb_vis, (360, 480),
                                interpolation=cv2.INTER_NEAREST)
        if self.instance_imagegoal is not None:
            tmp_goal = cv2.resize(self.instance_imagegoal, (480, 480),
                                    interpolation=cv2.INTER_NEAREST)
            tmp_goal = cv2.cvtColor(tmp_goal, cv2.COLOR_RGB2BGR)
            self.vis_image[50:50+480, 15:495] = tmp_goal
        self.vis_image[50:50+480, 510:870] = tmp
        self.vis_image[50:530, 885:1365] = sem_map_vis

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(885, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            cv2.imwrite(fn, self.vis_image)
