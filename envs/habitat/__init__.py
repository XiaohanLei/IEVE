# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
import os
import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat import Env, RLEnv, VectorEnv, make_dataset

from agents.sem_exp import Sem_Exp_Env_Agent
from agents.instance_exp import Instance_Exp_Env_Agent
# from agents.train_exp import Instance_Exp_Env_Agent
# from agents.t_dis import Instance_Exp_Env_Agent
from .objectgoal_env import ObjectGoal_Env
from omegaconf import OmegaConf
from .utils.vector_env import VectorEnv
from tqdm import tqdm
# from train_agents import MCCAgent

def make_env_fn(args, config_env, rank):
    dataset = make_dataset(config_env.habitat.dataset.type, config=config_env.habitat.dataset)
    OmegaConf.set_readonly(config_env, False)
    config_env.habitat.simulator.scene = dataset.episodes[0].scene_id
    OmegaConf.set_readonly(config_env, True)

    if args.agent == "sem_exp":
        env = Instance_Exp_Env_Agent(args=args, rank=rank,
                                config_env=config_env,
                                dataset=dataset
                                )
    else:
        env = ObjectGoal_Env(args=args, rank=rank,
                             config_env=config_env,
                             dataset=dataset
                             )

    env.seed(rank)
    return env


def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            # scene = filename[: -len(scene_dataset_ext) + 4]
            scene = filename[: -len(scene_dataset_ext) ]
            scenes.append(scene)
    scenes.sort()
    return scenes


def construct_envs(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_path="envs/habitat/configs/"
                                         + args.task_config)
    OmegaConf.set_readonly(basic_config, False)
    basic_config.habitat.dataset.split = args.split

    OmegaConf.set_readonly(basic_config, True)

    dataset = make_dataset(basic_config.habitat.dataset.type)
    scenes = basic_config.habitat.dataset.content_scenes
    if "*" in basic_config.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(basic_config.habitat.dataset)



    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")
    for i in tqdm(range(args.num_processes)):
        config_env = cfg_env(config_path="envs/habitat/configs/"
                                           + args.task_config)
        OmegaConf.set_readonly(config_env, False)

        if len(scenes) > 0:
            contentss = scenes[
                sum(scene_split_sizes[:i]):
                sum(scene_split_sizes[:i + 1])
            ]

            bad_scense = []
            good_scense = [contentss[i] for i in range(len(contentss)) if contentss[i] not in bad_scense]

            config_env.habitat.dataset.content_scenes = good_scense
            print("Thread {}: {}".format(i, config_env.habitat.dataset.content_scenes))

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

        config_env.habitat.environment.iterator_options.shuffle = False

        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = args.env_frame_width
        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = args.env_frame_height
        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = args.hfov
        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0, args.camera_height, 0]

        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = args.env_frame_width
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = args.env_frame_height
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = args.hfov
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = args.min_depth
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = args.max_depth
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0, args.camera_height, 0]

        config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = args.env_frame_width
        config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = args.env_frame_height
        config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.hfov = args.hfov
        config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.position = [0, args.camera_height, 0]

        config_env.habitat.simulator.agents.main_agent.height = args.camera_height

        config_env.habitat.dataset.split = args.split

        # config_env.freeze()
        OmegaConf.set_readonly(config_env, True)
        env_configs.append(config_env)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, range(args.num_processes))
            )
        ),
    )

    return envs
