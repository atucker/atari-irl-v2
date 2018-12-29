from typing import NamedTuple, Optional, List, Dict, Any, Tuple
import numpy as np
import gym

class TimeShape(NamedTuple):
    T: Optional[int] = None
    num_envs: Optional[int] = None


class Observations(NamedTuple):
    time_shape: TimeShape
    observations: np.ndarray


class Actions(NamedTuple):
    time_shape: TimeShape
    actions: np.ndarray


class Rewards(NamedTuple):
    time_shape: TimeShape
    rewards: np.ndarray


class EnvInfo(NamedTuple):
    time_shape: TimeShape

    # These come from the gym Environment interface
    obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    epinfos: List[Dict[str, Any]]


class PolicyInfo(NamedTuple):
    time_shape: TimeShape
    actions: np.ndarray


class Buffer(NamedTuple):
    time_shape: TimeShape
    obs: np.ndarray
    acts: np.ndarray
    rewards: np.ndarray


class Batch(NamedTuple):
    time_shape: TimeShape
    env_info: EnvInfo
    policy_info: PolicyInfo

    @property
    def obs(self):
        return self.env_info.obs

    @property
    def acts(self):
        return self.policy_info.actions

    @property
    def rewards(self):
        return self.env_info.rewards


class PolicyTrainer:
    def __init__(self, obs_space: Tuple[int], act_space: gym.Space) -> None:
        self.obs_space = obs_space
        self.act_space = act_space

    def get_actions(self, obs_batch: Observations) -> PolicyInfo:
        raise NotImplemented

    def train(self, buffer: Buffer) -> None:
        raise NotImplemented


class RewardModelTrainer:
    def __init__(self, obs_space: Tuple[int], act_space: Tuple[int]) -> None:
        raise NotImplemented

    def get_rewards(self, batch: Batch) -> np.ndarray:
        raise NotImplemented

    def train(self, buffer: Buffer) -> None:
        raise NotImplemented
