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
from mayavi import mlab
from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.imagegoal_env import ObjectGoal_Env
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from agents.utils.count import counter, MaskDINO
from constants import color_palette
import envs.utils.pose as pu
import agents.utils.visualization as vu
from collections import Counter
import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch
import numpy.ma as ma
import matplotlib.pyplot as plt
# from projects import *
from skimage.draw import line_aa, line
from argparse import ArgumentParser
import os.path as osp



class Instance_Exp_Env_Agent(ObjectGoal_Env):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):
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

        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='disk').eval().to(self.device)

        map_size = self.args.map_size_cm // self.args.map_resolution
        full_w, full_h = map_size, map_size
        self.full_w = full_w
        self.full_h = full_h
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
        self.goal_map_mask = np.ones((full_w, full_h))
        self.pred_box = []
        torch.set_grad_enabled(False)

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None

    def reset(self):
        args = self.args

        obs, info = super().reset()
        self.raw_obs = obs[:3, :, :].transpose(1, 2, 0)
        self.raw_depth = obs[3:4, :, :]

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
        self.goal_map_mask = np.ones(map_shape)
        self.goal_instance_whwh = None
        self.pred_box = []
        self.been_stuck = False
        self.stuck_goal = None
        self.frontier_vis = None

        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        return obs, info

    def local_feature_match_lightglue(self, re_key2=False):
        with torch.set_grad_enabled(False):
            ob = numpy_image_to_torch(self.raw_obs[:, :, :3]).to(self.device)
            gi = numpy_image_to_torch(self.instance_imagegoal).to(self.device)
            try:
                feats0, feats1, matches01  = match_pair(self.extractor, self.matcher, ob, gi
                    )
                # indices with shape (K, 2)
                matches = matches01['matches']
                # in case that the matches collapse make a check
                b = torch.nonzero(matches[..., 0] < 2048, as_tuple=False)
                c = torch.index_select(matches[..., 0], dim=0, index=b.squeeze())
                points0 = feats0['keypoints'][c]
                if re_key2:
                    return (points0.numpy(), feats1['keypoints'][c].numpy())
                else:
                    return points0.numpy()  
            except:
                if re_key2:
                    # print(f'{self.rank}  {self.timestep}  h')
                    return (np.zeros((1, 2)), np.zeros((1, 2)))
                else:
                    # print(f'{self.rank}  {self.timestep}  h')
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

    def compute_ins_dis_v1(self, depth, whwh, k=3):
        '''
        analyze the maxium depth points's pos
        make sure the object is within the range of 10m
        '''
        hist, bins = np.histogram(depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].flatten(), \
            bins=200,range=(0,2000))
        peak_indices = np.argsort(hist)[-k:]  # Get the indices of the top k peaks
        peak_values = hist[peak_indices] + hist[np.clip(peak_indices-1, 0, len(hist)-1)]  + \
            hist[np.clip(peak_indices+1, 0, len(hist)-1)]
        max_area_index = np.argmax(peak_values)  # Find the index of the peak with the largest area
        max_index = peak_indices[max_area_index]
        # max_index = np.argmax(hist)
        return bins[max_index]

    def compute_ins_goal_map(self, whwh, start, start_o):
        goal_mask = np.zeros_like(self.obs[3, :, :])
        goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1
        semantic_mask = (self.obs[4+self.gt_goal_idx, :, :] > 0) & (goal_mask > 0)

        depth_h, depth_w = np.where(semantic_mask > 0)
        goal_dis = self.obs[3, :, :][depth_h, depth_w] / self.args.map_resolution

        goal_angle = -self.args.hfov / 2 * (depth_w - self.obs.shape[2]/2) \
        / (self.obs.shape[2]/2)
        goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
            start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
        goal_map = np.zeros((self.local_w, self.local_h))
        goal[0] = np.clip(goal[0], 0, 240-1).astype(int)
        goal[1] = np.clip(goal[1], 0, 240-1).astype(int)
        goal_map[goal[0], goal[1]] = 1
        return goal_map

    def compute_ins_goal_map_mask(self, mask, start, start_o):

        semantic_mask = mask[:, :, 0] * 1
        selem = skimage.morphology.disk(3)
        semantic_mask = skimage.morphology.erosion(semantic_mask, selem)

        depth_h, depth_w = np.where(semantic_mask > 0)
        depth_h, depth_w = depth_h//4, depth_w//4
        goal_dis = self.obs[3, :, :][depth_h, depth_w] / self.args.map_resolution

        goal_angle = -self.args.hfov / 2 * (depth_w - self.obs.shape[2]/2) \
        / (self.obs.shape[2]/2)
        goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
            start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
        goal_map = np.zeros((self.local_w, self.local_h))
        goal[0] = np.clip(goal[0], 0, 240-1).astype(int)
        goal[1] = np.clip(goal[1], 0, 240-1).astype(int)
        goal_map[goal[0], goal[1]] = 1
        
        
        return goal_map


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


    def instance_discriminator_v2(self, planner_inputs, id_lo_whwh_speci):
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        map_pred = np.rint(planner_inputs['map_pred'])
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        goal_mask = self.semantic_obs

        if self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
            return planner_inputs

        if np.any(goal_mask > 0):
            
            global_goal = np.zeros((self.full_w, self.full_h))
            goal_map = self.compute_ins_goal_map_mask(goal_mask, start, start_o)

            global_goal[gx1:gx2, gy1:gy2] = goal_map
            if np.any(goal_map > 0):
                planner_inputs['found_goal'] = 1
                self.global_goal = global_goal
                planner_inputs['goal'] = goal_map
            else:
                planner_inputs['found_goal'] = 0
                planner_inputs['goal'] = planner_inputs['exp_goal']
            
        else:
            planner_inputs['found_goal'] = 0
            planner_inputs['goal'] = planner_inputs['exp_goal']
        return planner_inputs



    def instance_discriminator_v1(self, planner_inputs, id_lo_whwh_speci):
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
        if self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
        elif planner_inputs['found_goal'] == 1:

            # preprocess foundation model output
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
            

            if match_points > 100:
                planner_inputs['found_goal'] = 1
                global_goal = np.zeros((self.full_w, self.full_h))
                goal_map = self.compute_ins_goal_map(whwh, start, start_o)

                global_goal[gx1:gx2, gy1:gy2] = goal_map
                self.global_goal = global_goal
                planner_inputs['goal'] = goal_map
            else:
                planner_inputs['found_goal'] = 0
                planner_inputs['goal'] = planner_inputs['exp_goal']
        else:
            planner_inputs['found_goal'] = 0
            planner_inputs['goal'] = planner_inputs['exp_goal']


    def instance_discriminator_auto(self, planner_inputs, id_lo_whwh_speci):
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

        if self.instance_imagegoal is None:
            # not initialized
            return planner_inputs
        elif self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
            return planner_inputs
        elif self.been_stuck:
            
            planner_inputs['found_goal'] = 0
            if self.stuck_goal is None:

                navigable_indices = np.argwhere(self.visited[gx1:gx2, gy1:gy2] > 0)
                goal = np.array([0, 0])
                for _ in range(100):
                    random_index = np.random.choice(len(navigable_indices))
                    goal = navigable_indices[random_index]
                    if pu.get_l2_distance(goal[0], start[0], goal[1], start[1]) > 16:
                        break

                goal = pu.threshold_poses(goal, map_pred.shape)                
                self.stuck_goal = [int(goal[0])+gx1, int(goal[1])+gy1]
            else:
                goal = np.array([self.stuck_goal[0]-gx1, self.stuck_goal[1]-gy1])
                goal = pu.threshold_poses(goal, map_pred.shape)
            planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
            planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
        elif planner_inputs['found_goal'] == 1:

            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2] / 4).astype(int)
            w, h = whwh[2]-whwh[0], whwh[3]-whwh[1]
            goal_mask = np.zeros_like(goal_mask)
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.

            # index = self.local_feature_match()
            index = self.local_feature_match_lightglue()
            match_points = index.shape[0]
            planner_inputs['found_goal'] = 0
            if self.temp_goal is not None:
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
            else:
                goal_map = self.compute_ins_goal_map(whwh, start, start_o)
                if not np.any(goal_map>0) :
                    tgoal_dis = self.compute_ins_dis_v1(self.obs[3, :, :], whwh) / self.args.map_resolution
                    rgb_center = np.array([whwh[3]+whwh[1], whwh[2]+whwh[0]])//2
                    goal_angle = -self.args.hfov / 2 * (rgb_center[1] - self.obs.shape[2]/2) \
                    / (self.obs.shape[2]/2)
                    goal = [start[0]+tgoal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                        start[1]+tgoal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
                    goal = pu.threshold_poses(goal, map_pred.shape)
                    rr,cc = skimage.draw.ellipse(goal[0], goal[1], 10, 10, shape=goal_map.shape)
                    goal_map[rr, cc] = 1
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)

            if goal_dis is None:
                self.temp_goal = None
                planner_inputs['goal'] = planner_inputs['exp_goal']
                selem = skimage.morphology.disk(3)
                goal_map = skimage.morphology.dilation(goal_map, selem)
                self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                print(f"Rank: {self.rank}, timestep: {self.timestep},  temp goal unavigable !")
            else:
                if match_points > 100:
                    planner_inputs['found_goal'] = 1
                    global_goal = np.zeros((self.full_w, self.full_h))
                    global_goal[gx1:gx2, gy1:gy2] = goal_map
                    self.global_goal = global_goal
                    planner_inputs['goal'] = goal_map
                    self.temp_goal = None
                else:
                    with torch.no_grad():
                        dd = transforms.ToTensor()(self.raw_depth[0, :, :])
                        dd = transforms.Resize((160, 120))(dd)
                        depth_tensor = dd.unsqueeze(0).to(self.device)
                        extras_tensor = torch.tensor([match_points / 300.]).unsqueeze(0).to(self.device)
                        sign = self.switch_policy(depth_tensor, extras_tensor).item()
                        # print(f'{self.rank}  {self.timestep}  {sign}')
                        if sign > 0.9:
                            planner_inputs['found_goal'] = 1
                            global_goal = np.zeros((self.full_w, self.full_h))
                            global_goal[gx1:gx2, gy1:gy2] = goal_map
                            self.global_goal = global_goal
                            planner_inputs['goal'] = goal_map
                            self.temp_goal = None
                        else:
                            if goal_dis < 150 :
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                self.temp_goal = None
                                selem = skimage.morphology.disk(1)
                                goal_map = skimage.morphology.dilation(goal_map, selem)
                                self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                            else:
                                new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                                if np.any(new_goal_map > 0):
                                    planner_inputs['goal'] = new_goal_map
                                    temp_goal = np.zeros((self.full_w, self.full_h))
                                    temp_goal[gx1:gx2, gy1:gy2] = new_goal_map
                                    self.temp_goal = temp_goal
                                else:
                                    planner_inputs['goal'] = planner_inputs['exp_goal']
                                    self.temp_goal = None

            return planner_inputs
        else:
            planner_inputs['goal'] = planner_inputs['exp_goal']
            if self.temp_goal is not None:  
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
                planner_inputs['found_goal'] = 0
                new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                if np.any(new_goal_map > 0):
                    if goal_dis is not None:
                        planner_inputs['goal'] = new_goal_map
                        if goal_dis < 100:
                            index = self.local_feature_match_lightglue()
                            match_points = index.shape[0]
                            if match_points < 60:
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                selem = skimage.morphology.disk(3)
                                new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                                self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                                self.temp_goal = None
                    else:
                        selem = skimage.morphology.disk(3)
                        new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                        self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                        self.temp_goal = None
                        print(f"Rank: {self.rank}, timestep: {self.timestep},  temp goal unavigable !")
                else:
                    self.temp_goal = None
                    
                
            return planner_inputs

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
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
            # goal = [self.global_goal[0] - gx1, self.global_goal[1] - gy1]
            # goal = pu.threshold_poses(goal, map_pred.shape)
            # planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
            # planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
            return planner_inputs
        elif self.been_stuck:
            
            planner_inputs['found_goal'] = 0
            if self.stuck_goal is None:
                # goal_dis = -30 / self.args.map_resolution
                # goal_angle = 0
                # goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                #     start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]

                navigable_indices = np.argwhere(self.visited[gx1:gx2, gy1:gy2] > 0)
                goal = np.array([0, 0])
                for _ in range(100):
                    random_index = np.random.choice(len(navigable_indices))
                    goal = navigable_indices[random_index]
                    if pu.get_l2_distance(goal[0], start[0], goal[1], start[1]) > 16:
                        break

                goal = pu.threshold_poses(goal, map_pred.shape)                
                self.stuck_goal = [int(goal[0])+gx1, int(goal[1])+gy1]
            else:
                goal = np.array([self.stuck_goal[0]-gx1, self.stuck_goal[1]-gy1])
                goal = pu.threshold_poses(goal, map_pred.shape)
            planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
            planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
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

            index = self.local_feature_match_lightglue()
            match_points = index.shape[0]
            planner_inputs['found_goal'] = 0

            if self.temp_goal is not None:
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
            else:
                goal_map = self.compute_ins_goal_map(whwh, start, start_o)
                if not np.any(goal_map>0) :
                    tgoal_dis = self.compute_ins_dis_v1(self.obs[3, :, :], whwh) / self.args.map_resolution
                    rgb_center = np.array([whwh[3]+whwh[1], whwh[2]+whwh[0]])//2
                    goal_angle = -self.args.hfov / 2 * (rgb_center[1] - self.obs.shape[2]/2) \
                    / (self.obs.shape[2]/2)
                    goal = [start[0]+tgoal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                        start[1]+tgoal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
                    goal = pu.threshold_poses(goal, map_pred.shape)
                    rr,cc = skimage.draw.ellipse(goal[0], goal[1], 10, 10, shape=goal_map.shape)
                    goal_map[rr, cc] = 1


                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)

            if goal_dis is None:
                self.temp_goal = None
                planner_inputs['goal'] = planner_inputs['exp_goal']
                selem = skimage.morphology.disk(3)
                goal_map = skimage.morphology.dilation(goal_map, selem)
                self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                print(f"Rank: {self.rank}, timestep: {self.timestep},  temp goal unavigable !")
            else:
                if match_points > 100:
                    planner_inputs['found_goal'] = 1
                    global_goal = np.zeros((self.full_w, self.full_h))
                    global_goal[gx1:gx2, gy1:gy2] = goal_map
                    self.global_goal = global_goal
                    planner_inputs['goal'] = goal_map
                    self.temp_goal = None
                else:
                    if goal_dis < 150 :
                        if match_points > 60:
                            planner_inputs['found_goal'] = 1
                            global_goal = np.zeros((self.full_w, self.full_h))
                            global_goal[gx1:gx2, gy1:gy2] = goal_map
                            self.global_goal = global_goal
                            planner_inputs['goal'] = goal_map
                            self.temp_goal = None
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
                            selem = skimage.morphology.disk(1)
                            goal_map = skimage.morphology.dilation(goal_map, selem)
                            self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                    else:
                        new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                        if np.any(new_goal_map > 0):
                            planner_inputs['goal'] = new_goal_map
                            temp_goal = np.zeros((self.full_w, self.full_h))
                            temp_goal[gx1:gx2, gy1:gy2] = new_goal_map
                            self.temp_goal = temp_goal
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
            return planner_inputs

        else:
            planner_inputs['goal'] = planner_inputs['exp_goal']
            if self.temp_goal is not None:  
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
                planner_inputs['found_goal'] = 0
                new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                if np.any(new_goal_map > 0):
                    if goal_dis is not None:
                        planner_inputs['goal'] = new_goal_map
                        if goal_dis < 100:
                            index = self.local_feature_match_lightglue()
                            # index = self.local_feature_match_gluestick()
                            match_points = index.shape[0]
                            if match_points < 60:
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                selem = skimage.morphology.disk(3)
                                new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                                self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                                self.temp_goal = None
                    else:
                        selem = skimage.morphology.disk(3)
                        new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                        self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                        self.temp_goal = None
                        print(f"Rank: {self.rank}, timestep: {self.timestep},  temp goal unavigable !")
                else:
                    self.temp_goal = None
                    
                    
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


        id_lo_whwh = self.pred_box


        id_lo_whwh_speci = [id_lo_whwh[i] for i in range(len(id_lo_whwh)) \
                    if id_lo_whwh[i][0] == self.gt_goal_idx]


        planner_inputs["found_goal"] = (id_lo_whwh_speci != [])

        self.instance_discriminator(planner_inputs, id_lo_whwh_speci)

        action = self._plan(planner_inputs)

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)
            if 'sem_map_pred_3d' in planner_inputs:
                self._render_3d(planner_inputs)

        if action['action_args']['velocity_stop'] >= 0:

            obs, rew, done, info = super().step(action)
            self.raw_obs = obs[:3, :, :].transpose(1, 2, 0)
            self.raw_depth = obs[3:4, :, :]

            # preprocess obs
            obs = self._preprocess_obs(obs) 
            self.last_action = (action['action_args']['linear_velocity'] > 0)
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
        rr, cc, _ = line_aa(last_start[0], last_start[1], start[0], start[1])
        self.visited[gx1:gx2, gy1:gy2][rr, cc] += 1

        if args.visualize or args.print_images:            
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # relieve the stuck goal
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        if abs(x1 - x2) >= 0.05 or abs(y1 - y2) >= 0.05:
            self.been_stuck = False
            self.stuck_goal = None

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                    self.been_stuck = True
                self.col_width = min(self.col_width, 5)
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


        # modify the code with velocity control
        # Deterministic Local Policy
        if stop and planner_inputs['found_goal'] == 1:
            action = {
                "action": ("velocity_control", "velocity_stop"),
                "action_args": {
                    "angular_velocity": np.array([0]),
                    "linear_velocity": np.array([0]),
                    "camera_pitch_velocity": np.array([0]),
                    "velocity_stop": np.array([1]),
                },
            }
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

            # if relative_angle > self.args.turn_angle / 2.:
            if relative_angle > 15.:
                ang_vel = np.array([abs(relative_angle) / 60.])
                ang_vel = np.clip(ang_vel, 0., 1.)
                action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": -ang_vel,
                        "linear_velocity": np.array([-1]),
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([0]),
                    },
                }
            # elif relative_angle < -self.args.turn_angle / 2.:
            elif relative_angle < -15.:
                ang_vel = np.array([abs(relative_angle) / 60.])
                ang_vel = np.clip(ang_vel, 0., 1.)
                action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": ang_vel,
                        "linear_velocity": np.array([-1]),
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([0]),
                    },
                }
            else:
                lin_vel = np.array([pu.get_l2_distance(stg_x, start[0], stg_y, start[1]) * 5 / 35.])
                lin_vel = np.clip(lin_vel, 0., 1.)
                action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": np.array([0]),
                        "linear_velocity": lin_vel,
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([0]),
                    },
                }

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        ########################
        # selem = skimage.morphology.disk(3)
        # grid = skimage.morphology.dilation(grid, selem)
        # selem = skimage.morphology.disk(1)
        # grid = skimage.morphology.erosion(grid, selem)
        ########################

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1


        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)
        visited = add_boundary(self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2], value=0)

        planner = FMMPlanner(traversible)
        if self.global_goal is not None or self.temp_goal is not None:
            selem = skimage.morphology.disk(10)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        elif self.stuck_goal is not None:
            selem = skimage.morphology.disk(1)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        else:
            selem = skimage.morphology.disk(3)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]


        if self.global_goal is not None:
            st_dis = pu.get_l2_dis_point_map(state, goal) * self.args.map_resolution
            fmm_dist = planner.fmm_dist * self.args.map_resolution 
            dis = fmm_dist[start[0]+1, start[1]+1]
            if st_dis < 100 and dis/st_dis > 2:
                return (0, 0), True

        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
        if replan:
            stg_x, stg_y, _, stop = planner.get_short_term_goal(state, 2)
        # if self.rank == 0 and (self.timestep >= 50):
        #     np.set_printoptions(precision=3, suppress=True, linewidth=300)
        #     print(f"{self.timestep}    \n   {subset}")

        ########################
        # stg_x, stg_y, _, stop = planner.get_short_term_goal_v1(state, 48)
        # top_k_goal = np.zeros((48, 2))
        # top_k_goal[:, 0], top_k_goal[:, 1] = stg_x, stg_y
        # stg_goal = pu.ca_short_term_goal(np.rint(1-traversible), state, top_k_goal, 1)
        # if stg_goal is not None:
        #     stg_x = stg_goal[0]
        #     stg_y = stg_goal[1]
        # else:
        #     stg_x = top_k_goal[0, 0]
        #     stg_y = top_k_goal[0, 1]
        #     print(f"{self.rank}, {self.timestep}, error happens ! fall back to original")
        ########################

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
        # selem = skimage.morphology.disk(3)
        # obs_map = skimage.morphology.dilation(
        #     obs_map,
        #     selem)
        # exp_map = skimage.morphology.dilation(
        #     exp_map,
        #     selem)
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

        # self.frontier_vis = np.zeros((240, 240, 3), np.uint8)
        # contours_vis, _ = cv2.findContours(
        #     frontiers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        # self.frontier_vis = cv2.drawContours(self.frontier_vis,contours_vis,-1,(122,122,255),3) 

        if len(contours) > 0:
            contours = [c[:, 0].tolist() for c in contours]  # Clean format
            new_frontiers = np.zeros_like(frontiers)  
            # Only pick largest 1 frontiers
            contours = sorted(contours, key=lambda x: len(x), reverse=True)
            for contour in contours[:5]:
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
            for _ in range(5):
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

    def check_temp_goal_reachability(self, grid, temp_goal, start, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape
        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        

        # Note: Unlike _get_stg, no boundary is added here since we only want
        # to determine reachability.
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(14)
        goal = np.zeros((self.local_w, self.local_h))
        goal[int(temp_goal[0] - gx1), int(temp_goal[1] - gy1)] = 1
        goal = cv2.dilate(goal, selem)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist * self.args.map_resolution / 100.0

        navigable_indices = np.argwhere(self.visited[gx1:gx2, gy1:gy2] > 0)
        navigable_point = np.array([0, 0])
        for _ in range(100):
            random_index = np.random.choice(len(navigable_indices))
            navigable_point = navigable_indices[random_index]
            if pu.get_l2_distance(navigable_point[0], start[0], navigable_point[1], start[1]) < 20:
                break

        return fmm_dist[navigable_point[0], navigable_point[1]] < fmm_dist.max()

    # cm
    def compute_temp_goal_distance(self, grid, goal_map, start, planning_window):

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape
        goal = goal_map * 1
        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        st_dis = pu.get_l2_dis_point_map(start, goal) * self.args.map_resolution  # cm

        traversible = add_boundary(traversible)
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        
        goal = cv2.dilate(goal, selem)
        
        goal = add_boundary(goal, value=0)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist * self.args.map_resolution 
        dis = fmm_dist[start[0]+1, start[1]+1]

        
        if dis < fmm_dist.max() and dis/st_dis < 2:
            return dis
        else:
            return None




    def _get_reachability(self, grid, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        

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

        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), depth=depth[:, :, 0], use_seg=use_seg)

        sem_seg_pred = self._get_sem_pred(
            rgb.astype(np.uint8), use_seg=use_seg)

        # sem_seg_pred =self._get_sem_pred_internimg(
        #     rgb.astype(np.uint8), depth[:, :, 0], use_seg=use_seg
        # )
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

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

        # lj dataset
        for i in range(depth.shape[0]):
            depth[i, :][depth[i, :] == 0.] = depth[i, :].max() + 0.01

        

        # up or down floor check
        # agent_state = self._env.sim.get_agent_state(0).position
        # if abs(self.start_height - agent_state[1]) > 0.2:
        #     depth[:, :] = 0.
        #     print(f"Rank: {self.rank}, timestep: {self.timestep},  Agent Height Change !")

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def _get_sem_pred_internimg(self, rgb, depth, use_seg=True, pred_bbox=False):
        coco_categories_mapping_reduce = {
            56: 0,  # chair
            57: 1,  # couch
            58: 2,  # potted plant
            59: 3,  # bed
            61: 4,  # toilet
            62: 5,  # tv
        }
        conf = 0.3
        from constants import coco_categories_mapping
        if pred_bbox:
            bbox, mask = inference_detector(self.internimg, rgb)
            final = []
            for key in coco_categories_mapping_reduce:
                tt = bbox[key]
                select_index = np.where(tt[:, 4] > conf)
                select_tt = tt[select_index, :]
                select_tt = select_tt.squeeze(axis=0)
                for index in range(select_tt.shape[0]):
                    final.append([coco_categories_mapping_reduce[key], select_tt[index, 4], select_tt[index, :4]])
            return final
        else:
            if use_seg:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                bbox, mask = inference_detector(self.internimg, rgb)
                bbox_vis, mask_vis = bbox.copy(), mask.copy()
                final = []
                for key in coco_categories_mapping:
                    select_index = np.where(bbox[key][:, 4] > conf)
                    tt_mask = mask[key]
                    tt_mask = [tt_mask[i] for i in range(len(tt_mask)) if bbox[key][i, 4] > conf]
                    semantic_pred[:, :, coco_categories_mapping[key]] = np.logical_or.reduce(tt_mask, axis=0)
                    if key in coco_categories_mapping_reduce:
                        tt_box = bbox[key][select_index, :]
                        tt_box = tt_box.squeeze(axis=0)
                        for index in range(tt_box.shape[0]):
                            final.append([coco_categories_mapping[key], tt_box[index, 4], tt_box[index, :4]])

                self.pred_box = final
                # self.rgb_vis = self.internimg.show_result(
                #                     rgb,
                #                     (bbox_vis, mask_vis),
                #                     score_thr=0.3,
                #                     show=False,
                #                     bbox_color='coco',
                #                     text_color=(200, 200, 200),
                #                     mask_color='coco',
                #                     out_file=None
                #                 )[:, :, ::-1]
                # self.rgb_vis = rgb[:, :, ::-1]
                if depth is not None:
                    normalize_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.rgb_vis = cv2.cvtColor(normalize_depth, cv2.COLOR_GRAY2BGR)

                semantic_pred = semantic_pred.astype(np.float32)
            else:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                self.rgb_vis = rgb[:, :, ::-1]
            return semantic_pred

    def _get_sem_pred(self, rgb, depth=None, use_seg=True, pred_bbox=False):
        if pred_bbox:
            semantic_pred, self.rgb_vis, self.pred_box = self.sem_pred.get_prediction(rgb)
            return self.pred_box
        else:
            if use_seg:
                semantic_pred, self.rgb_vis, self.pred_box = self.sem_pred.get_prediction(rgb)
                semantic_pred = semantic_pred.astype(np.float32)
                if depth is not None:
                    normalize_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.rgb_vis = cv2.cvtColor(normalize_depth, cv2.COLOR_GRAY2BGR)
            else:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                self.rgb_vis = rgb[:, :, ::-1]
            return semantic_pred

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

    def render_3d_semantic_occupancy(self, occupancy_map, file_name):

        mlab.options.offscreen = True

        color_palette = np.array([
            [1.0, 1.0, 1.0],
            [0.6, 0.6, 0.6],
            [0.95, 0.95, 0.95],
            [0.96, 0.36, 0.26],
            [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
            [0.9400000000000001, 0.7818, 0.66],
            [0.8882000000000001, 0.9400000000000001, 0.66],
            [0.66, 0.9400000000000001, 0.8518000000000001],
            [0.7117999999999999, 0.66, 0.9400000000000001],
            [0.9218, 0.66, 0.9400000000000001],
            [0.9400000000000001, 0.66, 0.748199999999999]]) * 255
        
        color_palette = np.concatenate([color_palette.astype(int), np.ones([color_palette.shape[0], 1]) * 255], axis=1)

        xx, yy, zz, color = None, None, None, None
        for i in range(occupancy_map.shape[0]):
            x, y, z = np.where(occupancy_map[i, ...] == 1)
            if xx is not None:
                xx = np.concatenate((x, xx))
                yy = np.concatenate((y, yy))
                zz = np.concatenate((z, zz))
                color = np.concatenate((np.ones_like(x) * (i+1), color))
            else:
                xx = x
                yy = y
                zz = z
                color = np.ones_like(x) * (i+1)

        fig = mlab.figure(bgcolor=(1, 1, 1), size=(480, 480))
        plot = mlab.points3d(xx, yy, zz, color, mode="cube", figure=fig, scale_factor=1.0)

        # magic to modify lookup table
        plot.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, color_palette.shape[0])
        plot.module_manager.scalar_lut_manager.lut.number_of_colors = color_palette.shape[0]
        plot.module_manager.scalar_lut_manager.lut.table = color_palette

        mlab.view(azimuth=230, distance=200)
        mlab.savefig(filename=file_name)
        mlab.points3d

    def render_pyplot(self, occupancy_map, file_name):
        color_palette = [
            [1.0, 1.0, 1.0],
            [0.6, 0.6, 0.6],
            [0.95, 0.95, 0.95],
            [0.96, 0.36, 0.26],
            [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
            [0.9400000000000001, 0.7818, 0.66],
            [0.8882000000000001, 0.9400000000000001, 0.66],
            [0.66, 0.9400000000000001, 0.8518000000000001],
            [0.7117999999999999, 0.66, 0.9400000000000001],
            [0.9218, 0.66, 0.9400000000000001],
            [0.9400000000000001, 0.66, 0.748199999999999]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # for i in range(occupancy_map.shape[0]):
        ax.voxels(occupancy_map[0, ...], facecolors=color_palette[1])
        ax.voxels(occupancy_map[1, ...], facecolors=color_palette[3])
        ax.voxels(occupancy_map[2, ...], facecolors=color_palette[4])
        ax.voxels(occupancy_map[3, ...], facecolors=color_palette[5])
        ax.voxels(occupancy_map[4, ...], facecolors=color_palette[6])
        ax.voxels(occupancy_map[5, ...], facecolors=color_palette[7])
        ax.voxels(occupancy_map[6, ...], facecolors=color_palette[8])
        ax.set_axis_off()
        plt.savefig(file_name)

    def render_voxels(self, voxels, output_path):
        # Get the coordinates of occupied voxels
        voxels = voxels.transpose(0,1,3,2)

        color_palette = [
            [0.95, 0.95, 0.95],
            [0.96, 0.36, 0.26],
            [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
            [0.9400000000000001, 0.7818, 0.66],
            [0.8882000000000001, 0.9400000000000001, 0.66],
            [0.66, 0.9400000000000001, 0.8518000000000001],
            [0.7117999999999999, 0.66, 0.9400000000000001],
            [0.9218, 0.66, 0.9400000000000001],
            [0.9400000000000001, 0.66, 0.748199999999999]]
        for i in range(voxels.shape[0]):
            if i == 0:
                occupied_voxels = np.argwhere(voxels[i] > 0.5)
                # colors = np.ones([len(occupied_voxels), 3]) * color_palette[i]
                max_height = np.max(occupied_voxels[:, 1])
                min_height = np.min(occupied_voxels[:, 1])
                scale_factor =  (occupied_voxels[:, 1] - min_height)/(max_height - min_height)
                base_color = np.tile([[0.6, 0.6, 0.6]], (len(occupied_voxels), 1))
                colors = base_color - [0.6, 0.6, 0.6] * scale_factor[:, None]
            elif i < 7:
                temp_occupied_voxels = np.argwhere(voxels[i] > 0.5)
                if temp_occupied_voxels.shape[0] != 0:
                    occupied_voxels = np.concatenate((occupied_voxels, temp_occupied_voxels))
                    colors = np.concatenate((colors, np.ones([len(temp_occupied_voxels), 3]) * color_palette[i]))



        # occupied_voxels = np.argwhere(voxels > 0.5)

        # Create a point cloud from the occupied voxel coordinates
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(occupied_voxels)
        # pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.9, 0.9, 0.9]], (len(occupied_voxels), 1)))  # Set color for occupied voxels
        pcd.colors = o3d.utility.Vector3dVector(colors)  # Set color for occupied voxels

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.8)

        # Create a visualizer and set the render options
        vis = o3d.visualization.Visualizer()
        vis.create_window()  # Set the window size to match the desired image resolution
        vis.add_geometry(voxel_grid)

        ctr = vis.get_view_control()
        ctr.translate(0, 200)  # Move the camera up
        ctr.rotate(0, 200)  # Rotate the camera pitch
        # ctr.set_constant_z_near(20)

        img = vis.capture_screen_float_buffer(True)
        # Save the rendered images
        img = np.asarray(img) * 255  # Convert the image to a numpy array
        resize_img = cv2.resize(img,(960, 540),
                                interpolation=cv2.INTER_NEAREST)

        saved_image = np.zeros((540, 960+405, 3), np.uint8)
        tmp = cv2.resize(self.rgb_vis, (405, 540),interpolation=cv2.INTER_NEAREST)
        saved_image[:, :405, :] = tmp
        saved_image[:, 405:960+405, :] = resize_img
        

        plt.imsave(output_path,
                        saved_image,
                        dpi=1)

        vis.destroy_window()

    def _render_3d(self, inputs):

        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs['sem_map_pred_3d']
        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}-3D.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            # self.render_pyplot(map_pred, fn)
            # self.render_3d_semantic_occupancy(map_pred, fn)
            self.render_voxels(map_pred, fn)
        


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
        # vis_mask = self.visited[gx1:gx2, gy1:gy2] == 1

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

        if self.frontier_vis is not None:
            tt_vis = cv2.cvtColor(self.frontier_vis, cv2.COLOR_RGB2GRAY)
            sem_map[tt_vis > 0] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        # if self.frontier_vis is not None:
        #     sem_map_vis = np.where(self.frontier_vis > 0, self.frontier_vis, sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        tmp = cv2.resize(self.rgb_vis, (360, 480),
                                interpolation=cv2.INTER_NEAREST)

        tmp_goal = cv2.resize(self.instance_imagegoal, (480, 480),
                                interpolation=cv2.INTER_NEAREST)
        tmp_goal = cv2.cvtColor(tmp_goal, cv2.COLOR_RGB2BGR)
        self.vis_image[50:50+480, 15:495] = tmp_goal
        self.vis_image[50:50+480, 510:870] = tmp
        self.vis_image[50:530, 885:1365] = sem_map_vis
        self.vis_image[50:530, 495:510] = [255,255,255]

        if self.global_goal is not None:
            vu.update_text(self.vis_image, 'Exploitation')
        elif self.temp_goal is not None:
            vu.update_text(self.vis_image, 'Verificaiton')
        else:
            vu.update_text(self.vis_image, 'Exploration')

        # match0, match1 = self.local_feature_match_lightglue(re_key2=True)
        # try:
        #     for i in range(len(match0)):
        #         pt1 = match1[i]*480/512 + np.array([15, 50])
        #         pt2 = match0[i]*480/640 + np.array([510, 50])
        #         pt1 = pt1.astype(int)
        #         pt2 = pt2.astype(int)
        #         cv2.circle(self.vis_image, pt1, 3, (0, 255, 0), -1)  # Draw circle on image1
        #         cv2.circle(self.vis_image, pt2, 3, (0, 255, 0), -1)  # Draw circle on image2
        #         random_color = np.random.randint(0, 256, (3), np.uint8).tolist()
        #         random_color = tuple(random_color)
        #         cv2.line(self.vis_image, pt1, pt2, random_color, 2)
        # except:
        #     print("h")

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
            
            # try:
            #     import habitat
            #     top_down_map = habitat.utils.visualizations.maps.colorize_topdown_map(self.info['top_down_map']['map']) 
            #     top_fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Top-{}.png'.format(
            #     dump_dir, self.rank, self.episode_no,
            #     self.rank, self.episode_no, self.timestep)
            #     cv2.imwrite(top_fn, top_down_map)
            # except:
            #     import habitat
            #     from typing import cast
            #     from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
            #     top_down_map = habitat.utils.visualizations.maps.get_topdown_map_from_sim(
            #         cast("HabitatSim", self.habitat_env.sim), map_resolution=1024
            #     )
            #     top_fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Top-{}.png'.format(
            #     dump_dir, self.rank, self.episode_no,
            #     self.rank, self.episode_no, self.timestep)
            #     cv2.imwrite(top_fn, top_down_map)

            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            
            cv2.imwrite(fn, self.vis_image)
            
