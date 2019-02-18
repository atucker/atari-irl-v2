from typing import NamedTuple, Optional, Tuple, Generic, TypeVar, Dict, List, \
    Any, TYPE_CHECKING, Type
from typing import NamedTuple, Optional, Tuple, Generic, TypeVar, Dict, List, Any, Callable, Iterator
from collections import namedtuple, OrderedDict
import numpy as np
import gym
from baselines.common.vec_env import VecEnv
import functools

from .utils import one_hot

class TimeShape(NamedTuple):
    T: Optional[int] = None
    num_envs: Optional[int] = None
    batches: Optional[int] = None
        
    def check_shape(self, arr: np.ndarray) -> None:
        if self.batches is None:
            if self.T is not None and self.num_envs is not None:
                assert arr.shape[0] == self.num_envs
                assert arr.shape[1] == self.T
            else:
                N = self.T or self.num_envs
                assert arr.shape[0] == N
        else:
            raise NotImplemented
            
    def reshape(self, from_time_shape: 'TimeShape', data: np.ndarray) -> None:
        from_time_shape.check_shape(data)
        if self.T is not None and self.num_envs is None and self.batches is None:
            assert self.T == from_time_shape.T * from_time_shape.num_envs
            ans = data.reshape((self.T, *data.shape[2:]))
            self.check_shape(ans)
            return ans
        else:
            raise NotImplemented
            
    @property
    def size(self):
        values = [v for v in (self.T, self.num_envs, self.batches) if v is not None]
        return functools.reduce(lambda a, b: a * b, values, 1)


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
        discriminator: Any,
        time_shape: Optional[TimeShape],
        policy_info: Optional[T],
        env_info: Optional[EnvInfo],
        sampler_state: Optional[SamplerState]
    ) -> None:
        self.discriminator = discriminator
        self.time_shape = time_shape
        self.policy_info = policy_info
        self.env_info = env_info
        self.sampler_state = sampler_state

    def reshuffle(self):            
        np.random.shuffle(self.shuffle)

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
        return self.env_info.next_dones
    
    def iter_items(self, *keys, start_at=0) -> Iterator:
        TupClass = namedtuple('TupClass', keys)
        
        if self.time_shape.batches is None:
            if self.time_shape.num_envs is None or self.time_shape.T is None:
                for i in range(self.obs.shape[0]):
                    if i >= start_at:
                        yield TupClass(**dict(
                            (key, getattr(self, key)[i]) for key in keys
                        ))
            else:
                for i in range(self.obs.shape[0]):
                    for j in range(self.obs.shape[1]):
                        if i * self.obs.shape[1] + j >= start_at:
                            yield TupClass(**dict(
                                (key, getattr(self, key)[i, j]) for key in keys
                            ))
        else:
            for b in range(self.time_shape.batches):
                if self.time_shape.num_envs is None or self.time_shape.T is None:
                    for i in range(self.obs[b].shape[0]):
                        if i >= start_at:
                            yield TupClass(**dict(
                                (key, getattr(self, key)[b][i]) for key in keys
                            ))
                else:
                    for i in range(self.obs[b].shape[0]):
                        for j in range(self.obs[b].shape[1]):
                            if i * self.obs[b].shape[1] + j >= start_at:
                                yield TupClass(**dict(
                                    (key, getattr(self, key)[b][i, j]) for key in keys
                                ))

    def sample_batch(
        self,
        *keys: Tuple[str],
        batch_size: int,
        modify_obs: Callable[[np.ndarray], np.ndarray] = lambda obs: obs,
        one_hot_acts_to_dim: Optional[int] = None,
        debug=False
    ) -> Tuple[np.ndarray]:
        if not hasattr(self, 'sample_idx'):
            self.sample_idx = 0
            self.shuffle = np.arange(self.time_shape.size)
            self.reshuffle()
        
        # If we'd run past the end, then reshuffle
        # It's fine to miss the last few because we're reshuffling, and so any index
        # is equally likely to miss out
        if self.sample_idx + batch_size >= self.time_shape.size:
            self.sample_idx = 0
            self.shuffle = np.arange(self.time_shape.size)
            self.reshuffle()
            
        if self.sample_idx + batch_size >= len(self.shuffle):
            self.shuffle = np.arange(self.time_shape.size)
            self.reshuffle()
            
        #assert self.shuffle.shape[0] >= batch_size, f"batch size {batch_size} > amount of data {self.time_shape.size}"
        batch_slice = slice(self.sample_idx, self.sample_idx+batch_size)
        sampled_keys = {}

        def get_key(key):
            if key in sampled_keys:
                return sampled_keys[key]

            ans = getattr(self, key)[self.shuffle[batch_slice]]
            if 'obs' in key:
                ans = modify_obs(ans)
            if 'act' in key and one_hot_acts_to_dim is not None and len(ans.shape) == 1:
                ans = one_hot(ans, one_hot_acts_to_dim)
            if debug:
                print(f"{key}: {ans.shape}")
            assert ans.shape[0] > 0

            sampled_keys[key] = ans
            return ans

        if self.discriminator is not None:
            for key in keys:
                if key != 'rewards':
                    get_key(key)

            if 'rewards' in keys:
                sampled_keys['rewards'] = self.discriminator.eval(**sampled_keys)

        ans = tuple(get_key(key) for key in keys)
        
        # increment the read index
        self.sample_idx += batch_size
        
        return ans


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
    
    @property
    def next_dones(self):
        return self.env_info.next_dones


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


class Stacker:
    def __init__(self, other_cls: Type) -> None:
        self.data_cls = other_cls
        self.data = OrderedDict((f, []) for f in self.data_cls._fields)

    def append(self, tup: NamedTuple) -> None:
        assert isinstance(tup, self.data_cls)
        for f in tup._fields:
            self.data[f].append(getattr(tup, f))

    def __getattr__(self, item) -> Any:
        return self.data[item]

    def reset(self) -> None:
        for f in self.data.keys():
            self.data[f] = []
