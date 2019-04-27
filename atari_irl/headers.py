from typing import NamedTuple, Optional, Tuple, Generic, TypeVar, Dict, List, \
    Any, TYPE_CHECKING, Type
from typing import NamedTuple, Optional, Tuple, Generic, TypeVar, Dict, List, Any, Callable, Iterator
from collections import namedtuple, OrderedDict
import numpy as np
import gym
from baselines.common.vec_env import VecEnv
from baselines.ppo2.runner import sf01
import functools

from .utils import one_hot, inv_sf01


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
        time_shape: Optional[TimeShape],

        # maybe this should be replace_ instead?
        # we don't do inplace modifications
        overwrite_rewards: bool,
        overwrite_logprobs: bool,
        discriminator: Optional[Any] = None,
        policy: Optional[Any] = None,

        policy_info: Optional[T],
        env_info: Optional[EnvInfo],
        sampler_state: Optional[SamplerState]
    ) -> None:
        self.time_shape = time_shape

        if overwrite_rewards:
            assert discriminator is not None
        self.overwrite_rewards = overwrite_rewards
        self.discriminator = discriminator

        if overwrite_logprobs:
            assert policy is not None
        self.overwrite_logprobs = overwrite_logprobs
        self.policy = policy

        self.policy_info = policy_info
        self.env_info = env_info
        self.sampler_state = sampler_state

        self._latest_batch = None

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
    
    @property
    def lprobs(buffer):
        assert buffer.policy_info.lprobs is not None, "Log probabilities not provided by policy info"
        return buffer.policy_info.lprobs
    
    def add_batch(self, samples):
        self._latest_batch = samples
    
    @property
    def latest_batch(self):
        def handle_env_info(env_info):
            if self.overwrite_rewards:
                assert np.isclose(
                    self._latest_batch.rewards,
                    inv_sf01(
                        sf01(self._latest_batch.rewards),
                        self._latest_batch.rewards.shape
                    )
                ).all()
                # need to reconstruct the whole thing because it's a namedtuple
                return EnvInfo(
                    time_shape=env_info.time_shape,
                    obs=env_info.obs,
                    next_obs=env_info.next_obs,
                    rewards=inv_sf01(
                        self.discriminator.eval(
                            obs=sf01(self._latest_batch.obs),
                            next_obs=sf01(self._latest_batch.next_obs),
                            acts=one_hot(
                                sf01(self._latest_batch.acts),
                                self.discriminator.dU
                            ),
                            log_probs=sf01(self._latest_batch.policy_info.lprobs)
                        ),
                        self._latest_batch.rewards.shape
                    ),
                    dones=env_info.dones,
                    next_dones=env_info.next_dones,
                    epinfobuf=[]
                )
            else:
                return env_info

        return Batch(
            time_shape=self._latest_batch.time_shape,
            sampler_state=self._latest_batch.sampler_state,
            env_info=handle_env_info(
                self._latest_batch.env_info,

            ),
            policy_info=self._latest_batch.policy_info
        )

    def reshuffle(self):
        np.random.shuffle(self.shuffle)

    def _reset_shuffle(self):
        self.sample_idx = 0
        self.shuffle = np.arange(self.time_shape.size)
        self.reshuffle()

    def _handle_shuffle_edge_cases(self, batch_size):
        # Initialize shuffle logic if we haven't yet
        # sometimes we don't know our time_shape until later, so wait to do this
        # until now
        if not hasattr(self, 'sample_idx'):
            self._reset_shuffle()

        # If we'd run past the end, then reshuffle
        # It's fine to miss the last few because we're reshuffling, and so any index
        # is equally likely to miss out
        if self.sample_idx + batch_size >= self.time_shape.size:
            self._reset_shuffle()

        # If our shuffle list is too small for our current sample, extend it
        if self.sample_idx + batch_size >= len(self.shuffle):
            self.shuffle = np.arange(self.time_shape.size)
            self.reshuffle()

    def sample_batch(
        self,
        *keys: Tuple[str],
        batch_size: int,
        modify_obs: Callable[[np.ndarray], np.ndarray] = lambda obs: obs,
        one_hot_acts_to_dim: Optional[int] = None,
        debug=False
    ) -> Tuple[np.ndarray]:
        self._handle_shuffle_edge_cases(batch_size)

        batch_slice = slice(self.sample_idx, self.sample_idx+batch_size)
        sampled_keys = {}

        def get_key(key):
            if key in sampled_keys:
                return sampled_keys[key]

            ans = getattr(self, key)[self.shuffle[batch_slice]]
            if 'obs' in key:
                ans = modify_obs(ans)
            if 'act' in key and one_hot_acts_to_dim is not None and len(ans[0].shape) == 0:
                ans = one_hot(ans, one_hot_acts_to_dim)
            if debug:
                print(f"{key}: {ans.shape}")
            assert len(ans) > 0

            if 'act' in key:
                ans = np.array(ans)
            sampled_keys[key] = ans
            return ans

        def compute_logprobs():
            get_key('obs')
            get_key('acts')
            sampled_keys['lprobs'] = self.policy.get_a_logprobs(
                obs=sampled_keys['obs'],
                acts=sampled_keys['acts']
            )

        
        for key in keys:
            if key not in ['lprobs', 'rewards']:
                get_key(key)
        
        if self.overwrite_logprobs:
            if 'lprobs' in keys:
                compute_logprobs()

        if self.overwrite_rewards:
            if 'rewards' in keys:
                compute_logprobs()
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

    @property
    def lprobs(self):
        return self.policy_info.lprobs

class PolicyTrainer:
    info_class = None

    def __init__(self, env: VecEnv) -> None:
        self.obs_space = env.observation_space
        self.act_space = env.action_space

    def get_actions(self, obs_batch: Observations) -> PolicyInfo:
        raise NotImplemented

    def train_step(self, buffer: Buffer, itr: int, log_freq: int, logger: Any) -> None:
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
