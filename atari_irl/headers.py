from typing import NamedTuple, Optional, Tuple, Generic, TypeVar, Dict, List, Any, TYPE_CHECKING
import numpy as np
import gym
from baselines.common.vec_env import VecEnv


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
    next_obs: np.ndarray
    next_dones: np.ndarray

    epinfobuf: 'List[Dict[str, Any]]'


class PolicyInfo:
    _fields = ('time_shape', 'actions')
    def __init__(self, *, time_shape: TimeShape, actions: np.ndarray) -> None:
        self.time_shape = time_shape
        self.actions = actions
    

class SamplerState(NamedTuple):
    obs: np.ndarray
    dones: np.ndarray
        

T = TypeVar('T')


class Buffer(Generic[T]):
    def __init__(
        self, *, 
        time_shape: Optional[TimeShape],
        policy_info: Optional[T],
        env_info: Optional[EnvInfo],
        sampler_state: Optional[SamplerState]
    ) -> None:
        self.time_shape = time_shape
        self.policy_info = policy_info
        self.env_info = env_info
        self.sampler_state = sampler_state

    @property
    def obs(self):
        return self.env_info.obs

    @property
    def next_obs(self):
        return self.env_info.next_obs

    @property
    def acts(self):
        return self.policy_info.actions

    @property
    def rewards(self):
        return self.env_info.rewards

    @property
    def dones(self):
        return self.env_info.dones
    
    @property
    def next_dones(self):
        return self.env_info.dones


class Batch(NamedTuple):
    time_shape: TimeShape
    env_info: EnvInfo
    policy_info: PolicyInfo
    sampler_state: SamplerState

    @property
    def obs(self):
        return self.env_info.obs

    @property
    def next_obs(self):
        return self.env_info.next_obs

    @property
    def acts(self):
        return self.policy_info.actions

    @property
    def rewards(self):
        return self.env_info.rewards

    @property
    def dones(self):
        return self.env_info.dones


class PolicyTrainer:
    def __init__(self, env: VecEnv) -> None:
        self.obs_space = env.observation_space
        self.act_space = env.action_space

    def get_actions(self, obs_batch: Observations) -> PolicyInfo:
        raise NotImplemented

    def train(self, buffer: Buffer, itr: int) -> None:
        raise NotImplemented


class RewardModelTrainer:
    def __init__(self, obs_space: Tuple[int], act_space: Tuple[int]) -> None:
        raise NotImplemented

    def get_rewards(self, batch: Batch) -> np.ndarray:
        raise NotImplemented

    def train(self, buffer: Buffer) -> None:
        raise NotImplemented
