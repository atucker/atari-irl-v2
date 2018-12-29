from typing import NamedTuple, Tuple, Any, Dict, List, Type, Optional
from collections import OrderedDict
import numpy as np
import gym

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from . import environments


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


class Buffer(NamedTuple):
    time_shape: TimeShape
    obs: np.ndarray
    acts: np.ndarray
    rewards: np.ndarray


class Policy:
    def __init__(self, obs_space: Tuple[int], act_space: gym.Space) -> None:
        self.obs_space = obs_space
        self.act_space = act_space

    def get_actions(self, obs_batch: Observations) -> PolicyInfo:
        raise NotImplemented

    def train(self, buffer: Buffer) -> None:
        raise NotImplemented


class RandomPolicy(Policy):
    def get_actions(self, obs: Observations) -> PolicyInfo:
        assert obs.time_shape.T is None
        assert obs.time_shape.num_envs is not None
        return PolicyInfo(
            time_shape=obs.time_shape,
            actions=np.array([
                self.act_space.sample() for _ in range(obs.time_shape.num_envs)
            ])
        )

    def train(self, buffer: Buffer) -> None:
        pass


class RewardModel:
    def __init__(self, obs_space: Tuple[int], act_space: Tuple[int]) -> None:
        raise NotImplemented

    def get_rewards(self, batch: Batch) -> np.ndarray:
        raise NotImplemented

    def train(self, buffer: Buffer) -> None:
        raise NotImplemented


class Sampler:
    def __init__(self, env: gym.Env, policy: Policy) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.policy = policy
        self.obs = None
        self.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()

    def sample_batch(self, rollout_t: int, show=False) -> Batch:
        assert self.obs is not None, "Need to call reset"

        time_step  = TimeShape(num_envs=self.num_envs)
        time_shape = TimeShape(num_envs=self.num_envs, T=rollout_t)

        env_info_stacker = Stacker(EnvInfo)
        policy_info_stacker = Stacker(PolicyInfo)

        for _ in range(rollout_t):
            policy_step = self.policy.get_actions(Observations(
                time_shape=time_step,
                observations=self.obs
            ))
            policy_info_stacker.append(policy_step)
            actions = policy_step.actions

            self.obs[:], rewards, self.dones, epinfos = self.env.step(actions)
            env_info_stacker.append(EnvInfo(
                time_shape=time_step,
                obs=self.obs.copy(),
                rewards=rewards,
                dones=self.dones,
                epinfos=epinfos
            ))

        return Batch(
            time_shape=time_shape,
            env_info=EnvInfo(
                time_shape=time_shape,
                obs=np.array(env_info_stacker.obs),
                rewards=np.array(env_info_stacker.rewards),
                dones=np.array(env_info_stacker.dones),
                epinfos=[_ for _ in env_info_stacker.epinfos if _]
            ),
            policy_info=PolicyInfo(
                time_shape=time_shape,
                actions=np.array(policy_info_stacker.actions)
            )
        )


class DummyBuffer:
    def __init__(self):
        self.batch = None

    def add_batch(self, samples):
        self.batch = samples


class IRL:
    def __init__(self, args):
        self.env = environments.make_vec_env(
            env_name='PongNoFrameskip-v4',
            seed=0,
            one_hot_code=False,
            num_envs=8
        )

        self.policy = RandomPolicy(
            obs_space=self.env.observation_space,
            act_space=self.env.action_space
        )
        self.sampler = Sampler(
            env=self.env,
            policy=self.policy
        )

    def obtain_samples(self):
        return self.sampler.sample_batch(128)

    def update_buffer(self):
        pass

    def update_policy(self):
        pass

    def update_discriminator(self):
        pass

    def train(self):
        samples = self.obtain_samples()
        import pdb; pdb.set_trace()
        #buffer  = self.update_buffer(samples)
        #buffer  = self.update_discriminator(buffer)
        #self.update_policy(buffer)
