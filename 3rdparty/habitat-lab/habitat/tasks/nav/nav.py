#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO, lots of typing errors in here

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import attr
import numpy as np
import quaternion
from gym import spaces

from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.spaces import EmptySpace
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations import fog_of_war, maps

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    from habitat_sim import RigidState
    from habitat_sim.physics import VelocityControl
except ImportError:
    pass

try:
    import magnum as mn
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig


cv2 = try_cv2_import()


MAP_THICKNESS_SCALAR: int = 128


@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius."""

    room_id: str = attr.ib(default=None, validator=not_none_validator)
    room_name: Optional[str] = None


@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal] = attr.ib(
        default=None,
        validator=not_none_validator,
        on_setattr=Episode._reset_shortest_path_cache_hook,
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None


@registry.register_sensor
class PointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in PointGoal Navigation.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            `goal_format` which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a `dimensionality` field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._goal_format = getattr(config, "goal_format", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        self._dimensionality = getattr(config, "dimensionality", 2)
        assert self._dimensionality in [2, 3]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_pointgoal(
        self, source_position, source_rotation, goal_position
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1]
                    / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return direction_vector_agent

    def get_observation(
        self,
        observations,
        episode: NavigationEpisode,
        *args: Any,
        **kwargs: Any,
    ):
        source_position = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            source_position, rotation_world_start, goal_position
        )


@registry.register_sensor
class ImageGoalSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoal"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        # to be sure that the rotation is the same for the same episode_id
        # since the task is currently using pointnav Dataset.
        seed = abs(hash(episode.episode_id)) % (2**32)
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        return goal_observation[self._rgb_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor(name="PointGoalWithGPSCompassSensor")
class IntegratedPointGoalGPSAndCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            `goal_format` which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a `dimensionality` field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal_with_gps_compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "heading"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        if isinstance(rotation_world_agent, quaternion.quaternion):
            return self._quat_to_xy_heading(rotation_world_agent.inverse())
        else:
            raise ValueError("Agent's rotation was not a quaternion")


@registry.register_sensor(name="CompassSensor")
class EpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the episode,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        if isinstance(rotation_world_agent, quaternion.quaternion):
            return self._quat_to_xy_heading(
                rotation_world_agent.inverse() * rotation_world_start
            )
        else:
            raise ValueError("Agent's rotation was not a quaternion")


@registry.register_sensor(name="GPSSensor")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the `dimensionality` field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "gps"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._dimensionality = getattr(config, "dimensionality", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position = agent_state.position

        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array(
                [-agent_position[2], agent_position[0]], dtype=np.float32
            )
        else:
            return agent_position.astype(np.float32)


@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "proximity"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "max_detection_radius", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        return np.array(
            [
                self._sim.distance_to_closest_obstacle(
                    current_position, self._max_detection_radius
                )
            ],
            dtype=np.float32,
        )


@registry.register_measure
class Success(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._success_distance = self._config.success_distance

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and distance_to_target < self._success_distance
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class SoftSPL(SPL):
    r"""Soft SPL

    Similar to spl with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "soft_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class Collisions(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "collisions"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = {"count": 0, "is_collision": False}

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._metric["is_collision"] = False
        if self._sim.previous_step_collided:
            self._metric["count"] += 1
            self._metric["is_collision"] = True


@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure"""

    def __init__(
        self,
        sim: "HabitatSim",
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.map_padding
        self._step_count: Optional[int] = None
        self._map_resolution = config.map_resolution
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: List[Optional[Tuple[int, int]]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.draw_border,
        )

        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.draw_view_points:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        if goal.view_points is not None:
                            for view_point in goal.view_points:
                                self._draw_point(
                                    view_point.agent_state.position,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                    except AttributeError:
                        pass

    def _draw_goals_positions(self, episode):
        if self._config.draw_goal_positions:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        self._draw_point(
                            goal.position, maps.MAP_TARGET_POINT_INDICATOR
                        )
                    except AttributeError:
                        pass

    def _draw_goals_aabb(self, episode):
        if self._config.draw_goal_aabbs:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                        if self._is_on_same_floor(center[1])
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            (
                                self._top_down_map.shape[0],
                                self._top_down_map.shape[1],
                            ),
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: NavigationEpisode, agent_position: AgentState
    ):
        if self._config.draw_shortest_path:
            _shortest_path_points = (
                self._sim.get_straight_shortest_path_points(
                    agent_position, episode.goals[0].position
                )
            )
            self._shortest_path_points = [
                maps.to_grid(
                    p[2],
                    p[0],
                    (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                    sim=self._sim,
                )
                for p in _shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def _is_on_same_floor(
        self, height, ref_floor_height=None, ceiling_height=2.0
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height <= height < ref_floor_height + ceiling_height

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._top_down_map = self.get_original_map()
        self._step_count = 0
        agent_position = self._sim.get_agent_state().position
        self._previous_xy_location = [
            None for _ in range(len(self._sim.habitat_config.agents))
        ]

        if hasattr(episode, "goals"):
            # draw source and target parts last to avoid overlap
            self._draw_goals_view_points(episode)
            self._draw_goals_aabb(episode)
            self._draw_goals_positions(episode)
            self._draw_shortest_path(episode, agent_position)

        if self._config.draw_source:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

        self.update_metric(episode, None)
        self._step_count = 0

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        map_positions: List[Tuple[float]] = []
        map_angles = []
        for agent_index in range(len(self._sim.habitat_config.agents)):
            agent_state = self._sim.get_agent_state(agent_index)
            map_positions.append(self.update_map(agent_state, agent_index))
            map_angles.append(TopDownMap.get_polar_angle(agent_state))
        self._metric = {
            "map": self._top_down_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": map_positions,
            "agent_angle": map_angles,
        }

    @staticmethod
    def get_polar_angle(agent_state):
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation
        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]
        return np.array(phi)

    def update_map(self, agent_state: AgentState, agent_index: int):
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.max_episode_steps, 245
            )

            thickness = self.line_thickness
            if self._previous_xy_location[agent_index] is not None:
                cv2.line(
                    self._top_down_map,
                    self._previous_xy_location[agent_index],
                    (a_y, a_x),
                    color,
                    thickness=thickness,
                )
        angle = TopDownMap.get_polar_angle(agent_state)
        self.update_fog_of_war_mask(np.array([a_x, a_y]), angle)

        self._previous_xy_location[agent_index] = (a_y, a_x)
        return a_x, a_y

    def update_fog_of_war_mask(self, agent_position, angle):
        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                angle,
                fov=self._config.fog_of_war.fov,
                max_line_len=self._config.fog_of_war.visibility_dist
                / maps.calculate_meters_per_pixel(
                    self._map_resolution, sim=self._sim
                ),
            )


@registry.register_measure
class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None
        self._distance_to = self._config.distance_to

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        if self._distance_to == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._distance_to == "POINT":
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    [goal.position for goal in episode.goals],
                    episode,
                )
            elif self._distance_to == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            else:
                logger.error(
                    f"Non valid distance_to parameter was provided: {self._distance_to }"
                )

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = distance_to_target


@registry.register_measure
class DistanceToGoalReward(Measure):
    """
    The measure calculates a reward based on the distance towards the goal.
    The reward is `- (new_distance - previous_distance)` i.e. the
    decrease of distance to the goal.
    """

    cls_uuid: str = "distance_to_goal_reward"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._previous_distance: Optional[float] = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._previous_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = -(distance_to_target - self._previous_distance)
        self._previous_distance = distance_to_target


@registry.register_task_action
class MoveForwardAction(SimulatorTaskAction):
    name: str = "move_forward"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.move_forward)


@registry.register_task_action
class TurnLeftAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.turn_left)


@registry.register_task_action
class TurnRightAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.turn_right)


@registry.register_task_action
class StopAction(SimulatorTaskAction):
    name: str = "stop"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_stop_called = True  # type: ignore
        return self._sim.get_observations_at()  # type: ignore


@registry.register_task_action
class VelocityStopAction(SimulatorTaskAction):
    name: str = "velocity_stop"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore

    def step(
        self, velocity_stop, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if velocity_stop[0] > 0.0:
            task.is_stop_called = True  # type: ignore

        return self._sim.get_observations_at()  # type: ignore

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "velocity_stop": spaces.Box(
                    low=np.array([-1]),
                    high=np.array([1]),
                    dtype=np.float32,
                )
            }
        )


@registry.register_task_action
class LookUpAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.look_up)


@registry.register_task_action
class LookDownAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.look_down)


@registry.register_task_action
class TeleportAction(SimulatorTaskAction):
    # TODO @maksymets: Propagate through Simulator class
    COORDINATE_EPSILON = 1e-6
    COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
    COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "teleport"

    def step(
        self,
        *args: Any,
        position: List[float],
        rotation: Sequence[float],
        **kwargs: Any,
    ):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            return self._sim.get_observations_at()  # type: ignore

        return self._sim.get_observations_at(
            position=position, rotation=rotation, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([self.COORDINATE_MIN] * 3),
                    high=np.array([self.COORDINATE_MAX] * 3),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
                ),
            }
        )


@registry.register_task_action
class VelocityAction(SimulatorTaskAction):
    name: str = "velocity_control"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Initialize Habitat-Sim velocity control interface
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        # Cache config
        self._lin_vel_range = self._config.lin_vel_range
        self._ang_vel_range = self._config.ang_vel_range
        self._ang_vel_range_camera_pitch = (
            self._config.ang_vel_range_camera_pitch
        )
        self._ang_range_camera_pitch = self._config.ang_range_camera_pitch
        self._enable_scale_convert = self._config.enable_scale_convert
        self._time_step = self._config.time_step

    @property
    def action_space(self):
        if self._enable_scale_convert:
            return spaces.Dict(
                {
                    "linear_velocity": spaces.Box(
                        low=np.array([-1]),
                        high=np.array([1]),
                        dtype=np.float32,
                    ),
                    "angular_velocity": spaces.Box(
                        low=np.array([-1]),
                        high=np.array([1]),
                        dtype=np.float32,
                    ),
                    "camera_pitch_velocity": spaces.Box(
                        low=np.array([-1]),
                        high=np.array([1]),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            return spaces.Dict(
                {
                    "linear_velocity": spaces.Box(
                        low=np.array([self._lin_vel_range[0]]),
                        high=np.array([self._lin_vel_range[1]]),
                        dtype=np.float32,
                    ),
                    "angular_velocity": spaces.Box(
                        low=np.array([self._ang_vel_range[0]]),
                        high=np.array([self._ang_vel_range[1]]),
                        dtype=np.float32,
                    ),
                    "camera_pitch_velocity": spaces.Box(
                        low=np.array([self._ang_vel_range_camera_pitch[0]]),
                        high=np.array([self._ang_vel_range_camera_pitch[1]]),
                        dtype=np.float32,
                    ),
                }
            )

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore

    def step(
        self,
        *args: Any,
        task: EmbodiedTask,
        linear_velocity: float,
        angular_velocity: float,
        camera_pitch_angular_velocity: float = 0.0,
        time_step: Optional[float] = None,
        **kwargs: Any,
    ):
        r"""Moves the agent with a provided linear and angular velocity for the
        provided amount of time

        Args:
            linear_velocity: between [-1,1], scaled according to
                             config.lin_vel_range
            angular_velocity: between [-1,1], scaled according to
                             config.ang_vel_range
            camera_pitch_angular_velocity: between [-1,1], scaled according to
                             config.ang_vel_range
            time_step: amount of time to move the agent for
        """
        # Preprocess velocity input
        (
            lin_vel_processed,
            ang_vel_processed,
            camera_pitch_ang_vel_processed,
        ) = self._preprocess_action(
            linear_velocity, angular_velocity, camera_pitch_angular_velocity
        )

        # Apply camera action and get next observation
        agent_state_result = self._apply_camera_pitch_velocity_action(
            camera_pitch_ang_vel_processed,
            time_step=time_step,
        )

        # Apply action and get next observation
        agent_state_result = self._apply_velocity_action(
            lin_vel_processed,
            ang_vel_processed,
            time_step=time_step,
        )

        return self._get_agent_observation(agent_state_result)

    def _preprocess_action(
        self, linear_velocity, angular_velocity, camera_pitch_angular_velocity
    ):
        """Perform scaling and clamping of input"""
        if self._enable_scale_convert:
            linear_velocity = self._scale_inputs(
                linear_velocity,
                [-1, 1],
                self._lin_vel_range,
            )
            angular_velocity = self._scale_inputs(
                angular_velocity,
                [-1, 1],
                self._ang_vel_range,
            )
            camera_pitch_angular_velocity = self._scale_inputs(
                camera_pitch_angular_velocity,
                [-1, 1],
                self._ang_vel_range,
            )

        linear_velocity_clamped = np.clip(
            linear_velocity,
            self._lin_vel_range[0],
            self._lin_vel_range[1],
        )
        angular_velocity_clamped = np.clip(
            angular_velocity,
            self._ang_vel_range[0],
            self._ang_vel_range[1],
        )
        camera_pitch_angular_velocity_clamped = np.clip(
            camera_pitch_angular_velocity,
            self._ang_vel_range_camera_pitch[0],
            self._ang_vel_range_camera_pitch[1],
        )

        return (
            linear_velocity_clamped,
            angular_velocity_clamped,
            camera_pitch_angular_velocity_clamped,
        )

    def _apply_velocity_action(
        self,
        linear_velocity: float,
        angular_velocity: float,
        time_step: Optional[float] = None,
    ):
        """
        Apply velocity command to simulation, step simulation, and return agent observation
        """
        # Parse inputs
        if time_step is None:
            time_step = self._time_step

        # Map velocity actions
        self.vel_control.linear_velocity = mn.Vector3(
            [0.0, 0.0, -linear_velocity]
        )
        self.vel_control.angular_velocity = mn.Vector3(
            [0.0, angular_velocity, 0.0]
        )
        agent_state = self._sim.get_agent_state()

        # Convert from np.quaternion (quaternion.quaternion) to mn.Quaternion
        normalized_quaternion = agent_state.rotation
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        current_rigid_state = RigidState(
            agent_mn_quat,
            agent_state.position,
        )

        # manually integrate the rigid state
        goal_rigid_state = self.vel_control.integrate_transform(
            time_step, current_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        step_fn = self._sim.pathfinder.try_step_no_sliding  # type: ignore
        final_position = step_fn(
            agent_state.position, goal_rigid_state.translation
        )
        final_rotation = [
            *goal_rigid_state.rotation.vector,
            goal_rigid_state.rotation.scalar,
        ]

        # Check if a collision occured
        dist_moved_before_filter = (
            goal_rigid_state.translation - agent_state.position
        ).dot()
        dist_moved_after_filter = (final_position - agent_state.position).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the
        # filter is _less_ than the amount moved before the application of the
        # filter.
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

        # TODO: Make a better way to flag collisions
        self._sim._prev_sim_obs["collided"] = collided  # type: ignore

        # Update the state of the agent
        self._sim.set_agent_state(  # type: ignore
            final_position, final_rotation, reset_sensors=False
        )

        final_agent_state = self._sim.get_agent_state()
        final_agent_state.position = final_position
        final_agent_state.rotation = goal_rigid_state.rotation

        return final_agent_state

    def _apply_camera_pitch_velocity_action(
        self,
        camera_pitch_angular_velocity: float,
        time_step: Optional[float] = None,
    ):
        """
        Apply velocity command to the camera tilt angle for looking up and down
        """
        if time_step is None:
            time_step = self._time_step

        # Map velocity actions
        self.vel_control.linear_velocity = mn.Vector3([0.0, 0.0, 0.0])
        self.vel_control.angular_velocity = mn.Vector3(
            [camera_pitch_angular_velocity, 0.0, 0.0]
        )
        sensor_names = list(self._sim.agents[0]._sensors.keys())  # type: ignore
        sensor_state = self._sim.agents[0]._sensors[sensor_names[0]].node  # type: ignore

        # Construct the sensor rigid state
        agent_mn_quat = sensor_state.rotation
        current_rigid_state = RigidState(
            agent_mn_quat,
            sensor_state.translation,
        )

        # manually integrate the rigid state
        goal_rigid_state = self.vel_control.integrate_transform(
            time_step, current_rigid_state
        )

        # Compute the delta increase
        delta = np.sign(camera_pitch_angular_velocity) * abs(
            float(goal_rigid_state.rotation.angle())
            - float(current_rigid_state.rotation.angle())
        )

        # Get the camera pitch angle
        camera_pitch_ang = self._get_camera_pitch_angle()

        # Handle the min and max pitch angles
        if camera_pitch_ang + delta > self._ang_range_camera_pitch[1]:
            next_camera_pitch_ang = self._ang_range_camera_pitch[1]
            goal_rigid_state.rotation = mn.Quaternion.rotation(
                mn.Rad(next_camera_pitch_ang), mn.Vector3(1, 0, 0)
            )
        elif camera_pitch_ang + delta < self._ang_range_camera_pitch[0]:
            next_camera_pitch_ang = self._ang_range_camera_pitch[0]
            goal_rigid_state.rotation = mn.Quaternion.rotation(
                mn.Rad(next_camera_pitch_ang), mn.Vector3(1, 0, 0)
            )

        # Update all the sensors
        for sensor in sensor_names:
            self._sim.agents[0]._sensors[  # type: ignore
                sensor
            ].node.rotation = goal_rigid_state.rotation

        return self._sim.get_agent_state()

    def _get_agent_observation(self, agent_state=None):
        position = None
        rotation = None

        if agent_state is not None:
            position = agent_state.position
            rotation = [
                *agent_state.rotation.vector,
                agent_state.rotation.scalar,
            ]

        return self._sim.get_observations_at(
            position=position,
            rotation=rotation,
            keep_agent_at_new_pose=True,
        )

    def _get_camera_pitch_angle(self):
        """
        Get the camera pitch angle
        """
        # Get the sensor node
        sensor_name = list(self._sim.agents[0]._sensors.keys())[0]  # type: ignore
        sensor_state = self._sim.agents[0]._sensors[sensor_name].node  # type: ignore

        # Construct the sensor rigid state
        camera_mn_quat = sensor_state.rotation

        # Get the current camera pitch angle
        camera_rotvec = quaternion.from_float_array(
            np.array([camera_mn_quat.scalar] + list(camera_mn_quat.vector))
        )
        camera_rotvec = quaternion.as_rotation_vector(camera_rotvec)
        camera_angle = camera_rotvec[0]
        return camera_angle

    @staticmethod
    def _scale_inputs(
        input_val: float, input_range: List[float], output_range: List[float]
    ) -> float:
        """
        Transform input from input range to output range
        (ex: from normalized input in range [-1, 1] to [y_min, y_max])
        TODO: This function should go into utils of some sort
        """
        w_input = input_range[1] - input_range[0]
        w_output = output_range[1] - output_range[0]
        return (
            output_range[0] + (input_val - input_range[0]) * w_output / w_input
        )


@registry.register_task_action
class WaypointAction(VelocityAction):
    name: str = "waypoint_control"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Init goto velocity controller
        try:
            from habitat.utils.controller_wrapper import (
                DiffDriveVelocityController,
            )

            self.w2v_controller = DiffDriveVelocityController(self._config)
        except ModuleNotFoundError as exc:
            additional_error_message = """
            Missing dependencies for waypoint type actions.
            Install habitat-lab with the 'home_robot' option to enable this feature.
            pip install -e "habitat-lab[home_robot]
            """
            raise ModuleNotFoundError(additional_error_message) from exc

        # Cache hydra configs
        self._waypoint_lin_range = self._config.waypoint_lin_range
        self._waypoint_ang_range = self._config.waypoint_ang_range
        self._delta_ang_range_camera_pitch = (
            self._config.delta_ang_range_camera_pitch
        )
        self._wait_duration_range = self._config.wait_duration_range
        self._yaw_input_in_degrees = self._config.yaw_input_in_degrees
        self._min_abs_lin_speed = self._config.min_abs_lin_speed
        self._min_abs_ang_speed = self._config.min_abs_ang_speed
        self._min_abs_ang_speed_camera_pitch = (
            self._config.min_abs_ang_speed_camera_pitch
        )
        self._w_max_camera_pitch = self._config.w_max_camera_pitch
        self._acc_ang_camera_pitch = self._config.acc_ang_camera_pitch

        if self._yaw_input_in_degrees:
            self._waypoint_ang_range = [
                np.deg2rad(d) for d in self._waypoint_ang_range
            ]

    @property
    def action_space(self):
        if self._enable_scale_convert:
            return spaces.Dict(
                {
                    "xyt_waypoint": spaces.Box(
                        low=-np.ones(3),
                        high=np.ones(3),
                        dtype=np.float32,
                    ),
                    "delta_camera_pitch_angle": spaces.Box(
                        low=np.array([-1]),
                        high=np.array([1]),
                        dtype=np.float32,
                    ),
                    "max_duration": spaces.Box(
                        low=np.array([0]),
                        high=np.array([1]),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            lo = [
                self._waypoint_lin_range[0],
                self._waypoint_lin_range[0],
                self._waypoint_ang_range[0],
            ]
            hi = [
                self._waypoint_lin_range[1],
                self._waypoint_lin_range[1],
                self._waypoint_ang_range[1],
            ]
            return spaces.Dict(
                {
                    "xyt_waypoint": spaces.Box(
                        low=np.array(lo),
                        high=np.array(hi),
                        dtype=np.float32,
                    ),
                    "delta_camera_pitch_angle": spaces.Box(
                        low=np.array([self._delta_ang_range_camera_pitch[0]]),
                        high=np.array([self._delta_ang_range_camera_pitch[1]]),
                        dtype=np.float32,
                    ),
                    "max_duration": spaces.Box(
                        low=np.array([self._wait_duration_range[0]]),
                        high=np.array([self._wait_duration_range[1]]),
                        dtype=np.float32,
                    ),
                }
            )

    def step(
        self,
        task: EmbodiedTask,
        xyt_waypoint: List[float],
        delta_camera_pitch_angle: float,
        max_duration: float,
        *args,
        **kwargs,
    ):
        # Preprocess waypoint input
        assert len(xyt_waypoint) == 3, "Waypoint vector must be of length 3."
        (
            xyt_waypoint_processed,
            delta_camera_pitch_angle_processed,
            max_duration_processed,
        ) = self._preprocess_action(
            xyt_waypoint, delta_camera_pitch_angle, max_duration
        )

        # Execute waypoint
        return self._step_rel_waypoint(
            xyt_waypoint_processed,
            delta_camera_pitch_angle_processed,
            max_duration_processed,
            *args,
            **kwargs,
        )

    def _preprocess_action(
        self, xyt_waypoint, delta_camera_pitch_angle, max_duration
    ):
        """Perform scaling and clamping of input"""
        # Scale
        if self._enable_scale_convert:
            xyt_waypoint[0] = self._scale_inputs(
                xyt_waypoint[0],
                [-1, 1],
                [
                    self._waypoint_lin_range[0],
                    self._waypoint_lin_range[1],
                ],
            )
            xyt_waypoint[1] = self._scale_inputs(
                xyt_waypoint[1],
                [-1, 1],
                [
                    self._waypoint_lin_range[0],
                    self._waypoint_lin_range[1],
                ],
            )
            xyt_waypoint[2] = self._scale_inputs(
                xyt_waypoint[2],
                [-1, 1],
                [
                    self._waypoint_ang_range[0],
                    self._waypoint_ang_range[1],
                ],
            )
            delta_camera_pitch_angle = self._scale_inputs(
                delta_camera_pitch_angle,
                [-1, 1],
                [
                    self._delta_ang_range_camera_pitch[0],
                    self._delta_ang_range_camera_pitch[1],
                ],
            )
            max_duration = self._scale_inputs(
                max_duration,
                [0, 1],
                [
                    self._wait_duration_range[0],
                    self._wait_duration_range[1],
                ],
            )

        # Clamp
        xyt_waypoint_clamped = np.array(
            [
                np.clip(xyt_waypoint[0], *self._waypoint_lin_range),
                np.clip(xyt_waypoint[1], *self._waypoint_lin_range),
                np.clip(xyt_waypoint[2], *self._waypoint_ang_range),
            ]
        )
        delta_camera_pitch_angle_clamped = np.clip(
            delta_camera_pitch_angle, *self._delta_ang_range_camera_pitch
        )
        max_duration_clamped = np.clip(
            max_duration, *self._wait_duration_range
        )

        # Convert deg to rad
        if self._yaw_input_in_degrees:
            xyt_waypoint_clamped[2] = np.deg2rad(xyt_waypoint_clamped[2])

        return (
            xyt_waypoint_clamped,
            delta_camera_pitch_angle_clamped,
            max_duration_clamped,
        )

    def _step_rel_waypoint(
        self,
        xyt_waypoint,
        max_wait_duration,
        delta_camera_pitch_angle=0.0,
        *args,
        **kwargs,
    ):
        """Use the waypoint-to-velocity controller to navigate to the waypoint"""

        # Initialize control loop
        xyt_init = self._agent_state_to_xyt(self._sim.get_agent_state())
        self.w2v_controller.set_goal(
            xyt_waypoint, start=xyt_init, relative=True
        )

        xyt = xyt_init.copy()

        # Get the goal camera angle
        camera_pitch_ang = self._get_camera_pitch_angle()
        goal_camera_pitch_ang = camera_pitch_ang + delta_camera_pitch_angle

        # Forward simulate
        max_duration = max(
            max_wait_duration, self._time_step
        )  # always run for 1 step
        for _t in np.arange(0.0, max_duration, self._time_step):
            # Query velocity controller for control input
            linear_velocity, angular_velocity = self.w2v_controller.forward(
                xyt
            )

            # Query velocity controller for control input of pitch of camera
            camera_pitch_ang = self._get_camera_pitch_angle()
            camera_pitch_angular_err = goal_camera_pitch_ang - camera_pitch_ang
            camera_pitch_angular_velocity = (
                self.w2v_controller.velocity_feedback_control(
                    camera_pitch_angular_err,
                    self._acc_ang_camera_pitch,
                    self._w_max_camera_pitch,
                )
            )

            # Apply action and step simulation
            next_agent_state = self._apply_camera_pitch_velocity_action(
                camera_pitch_angular_velocity,
                time_step=self._time_step,
            )
            next_agent_state = self._apply_velocity_action(
                linear_velocity, angular_velocity, time_step=self._time_step
            )
            xyt = self._agent_state_to_xyt(next_agent_state)

            # Complete action early if commanded speed is low
            if (
                abs(linear_velocity) < self._min_abs_lin_speed
                and abs(angular_velocity) < self._min_abs_ang_speed
                and abs(camera_pitch_angular_velocity)
                < self._min_abs_ang_speed_camera_pitch
            ):
                break

        return self._get_agent_observation(next_agent_state)

    @staticmethod
    def _agent_state_to_xyt(agent_state):
        """
        Home robot coordinate system: +z = up
        Habitat coordinate system: +y = up

        We map the coordinate systems by assuming the +x axis is shared
        x -> x
        y -> -z
        rz -> -ry
        """
        # Convert rotation to rotation vector
        q = agent_state.rotation
        if type(q) is mn.Quaternion:
            q = quaternion.from_float_array(
                np.array([q.scalar] + list(q.vector))
            )
        elif type(q) is np.ndarray:
            q = quaternion.from_float_array(q)
        elif type(q) is not quaternion.quaternion:
            raise TypeError("Invalid agent_state.rotation type.")

        agent_rotvec = quaternion.as_rotation_vector(q)

        return np.array(
            [
                agent_state.position[0],
                -agent_state.position[2],
                agent_rotvec[1] + np.pi / 2.0,
            ]
        )


@registry.register_task_action
class MoveForwardWaypointAction(WaypointAction):
    name: str = "move_forward_waypoint"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._forward_step_size = self._config.forward_step_size
        self._max_wait_duration = self._config.max_wait_duration

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        xyt_waypoint = np.array([self._forward_step_size, 0.0, 0.0])
        return self._step_rel_waypoint(
            xyt_waypoint, self._config.max_wait_duration, *args, **kwargs
        )

    @property
    def action_space(self):
        return EmptySpace()


@registry.register_task_action
class TurnLeftWaypointAction(WaypointAction):
    name: str = "turn_left_waypoint"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._turn_angle = self._config.turn_angle
        self._max_wait_duration = self._config.max_wait_duration

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        xyt_waypoint = np.array([0.0, 0.0, self._turn_angle])
        return self._step_rel_waypoint(
            xyt_waypoint, self._config.max_wait_duration, *args, **kwargs
        )

    @property
    def action_space(self):
        return EmptySpace()


@registry.register_task_action
class TurnRightWaypointAction(WaypointAction):
    name: str = "turn_right_waypoint"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._turn_angle = self._config.turn_angle
        self._max_wait_duration = self._config.max_wait_duration

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        xyt_waypoint = np.array([0.0, 0.0, -self._config.turn_angle])
        return self._step_rel_waypoint(
            xyt_waypoint, self._config.max_wait_duration, *args, **kwargs
        )

    @property
    def action_space(self):
        return EmptySpace()


@registry.register_task_action
class LookUpDiscreteToVelocityAction(WaypointAction):
    name: str = "look_up_discrete_to_velocity"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._turn_angle = self._config.turn_angle
        self._max_wait_duration = self._config.max_wait_duration

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        xyt_waypoint = np.array([0.0, 0.0, 0.0])
        delta_camera_pitch_angle = self._turn_angle
        return self._step_rel_waypoint(
            xyt_waypoint,
            self._config.max_wait_duration,
            delta_camera_pitch_angle,
            *args,
            **kwargs,
        )

    @property
    def action_space(self):
        return EmptySpace()


@registry.register_task_action
class LookDownDiscreteToVelocityAction(WaypointAction):
    name: str = "look_down_discrete_to_velocity"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._turn_angle = self._config.turn_angle
        self._max_wait_duration = self._config.max_wait_duration

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        xyt_waypoint = np.array([0.0, 0.0, 0.0])
        delta_camera_pitch_angle = -self._turn_angle
        return self._step_rel_waypoint(
            xyt_waypoint,
            self._config.max_wait_duration,
            delta_camera_pitch_angle,
            *args,
            **kwargs,
        )

    @property
    def action_space(self):
        return EmptySpace()


@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        with read_write(config):
            config.simulator.scene = episode.scene_id
            if (
                episode.start_position is not None
                and episode.start_rotation is not None
            ):
                agent_config = get_agent_config(config.simulator)
                agent_config.start_position = episode.start_position
                agent_config.start_rotation = [
                    float(k) for k in episode.start_rotation
                ]
                agent_config.is_set_start_state = True
        return config

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
