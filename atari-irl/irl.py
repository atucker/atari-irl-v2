from typing import NamedTuple, Tuple, Any, Dict, List
from collections import OrderedDict
import numpy as np
import gym


class TimeShape(Tuple[int]):
    @property
    def T(self):
        return self[-1]
    @property
    def num_envs(self):
        return self[-2]


class Observations(NamedTuple):
    time_shape: TimeShape
    obs: np.ndarray


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
    def __init__(self, other_cls: NamedTuple) -> None:
        self.data_cls = other_cls
        self.data = OrderedDict((f, []) for f in self.data_cls._fields)

    def add(self, tup: NamedTuple) -> None:
        assert isinstance(tup, self.data_cls)
        for f in tup._fields:
            self.data[f].append(getattr(tup, f))


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

    def get_actions(self, obs_batch: Observations) -> Tuple[Actions, Any]:
        raise NotImplemented

    def train(self, buffer: Buffer) -> None:
        raise NotImplemented


class RandomPolicy(Policy):
    def get_actions(self, obs: Observations) -> Tuple[Actions, Any]:
        if len(obs.time_shape) == 1:
            return Actions(
                time_shape=obs.time_shape,
                actions=np.array([
                    self.act_space.sample() for _ in range(obs.time_shape.T)
                ])
            ), None
        elif len(obs.time_shape) == 2:
            return Actions(
                time_shape=obs.time_shape,
                actions=np.array([
                    [self.act_space.sample() for _ in range(obs.time_shape.num_envs)]
                    for _ in range(obs.time_shape.T)
                ])
            ), None
        else:
            raise NotImplemented

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

        time_shape = TimeShape((self.num_envs, rollout_t))

        env_info_stacker = Stacker[EnvInfo]()
        policy_info_stacker = Stacker[PolicyInfo]()

        for _ in range(rollout_t):
            policy_step = self.policy.get_actions(self.obs)
            policy_info_stacker.append(policy_step)

            self.obs[:], rewards, self.dones, epinfos = self.env.step(policy_step.actions)
            env_info_stacker.append(EnvInfo(
                time_shape=TimeShape((self.num_envs, 1)),
                obs=self.obs.copy(),
                rewards=rewards,
                dones=self.dones,
                epinfos=epinfos
            ))

        return Batch(
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
        self.env = gym.make('PongNoFrameskip-v4')
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


IRL(None).train()
