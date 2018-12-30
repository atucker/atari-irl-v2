from typing import NamedTuple, Any, Type, TypeVar, Generic, TypeVar
from collections import OrderedDict
import numpy as np
import gym
from baselines.common.vec_env import VecEnv

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import environments
import policies
from headers import TimeShape, EnvInfo, PolicyInfo, Observations, PolicyTrainer, Batch, Buffer

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


class RandomPolicy(PolicyTrainer):
    def get_actions(self, obs: Observations) -> PolicyInfo:
        assert obs.time_shape.T is None
        assert obs.time_shape.num_envs is not None
        return PolicyInfo(
            time_shape=obs.time_shape,
            actions=np.array([
                self.act_space.sample() for _ in range(obs.time_shape.num_envs)
            ])
        )

    def train(self, buffer: Buffer, i: int) -> None:
        pass

class Sampler:
    def __init__(self, env: VecEnv, policy: PolicyTrainer) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.policy = policy
        self.obs = None
        self.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()

    def sample_batch(self, rollout_t: int, show=False) -> Batch:
        assert self.obs is not None, "Need to call reset"
        assert issubclass(self.policy.info_class, PolicyInfo)
        
        env_info_stacker = Stacker(EnvInfo)
        policy_info_stacker = Stacker(self.policy.info_class)

        time_step  = TimeShape(num_envs=self.num_envs)
        time_shape = TimeShape(num_envs=self.num_envs, T=rollout_t)

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
                epinfobuf=[
                    info.get('episode') for info in epinfos if info.get('episode')
                ]
            ))

        return Batch(
            time_shape=time_shape,
            env_info=EnvInfo(
                time_shape=time_shape,
                obs=np.array(env_info_stacker.obs),
                rewards=np.array(env_info_stacker.rewards),
                dones=np.array(env_info_stacker.dones),
                epinfobuf=[_ for l in env_info_stacker.epinfobuf for _ in l]
            ),
            # TODO(Aaron): Make this a cleaner method, probably of stacker
            # where each field can define a lambda for how to process it
            # instead of assuming that we just use np.array
            policy_info=self.policy.info_class(
                time_shape=time_shape,
                **dict(
                    (field, np.array(getattr(policy_info_stacker, field)))
                    for field in self.policy.info_class._fields
                    if field != 'time_shape'
                )
            )
        )


T = TypeVar('T')


class DummyBuffer(Buffer[T]):
    def __init__(self):
        super().__init__(
            time_shape=None,
            policy_info=None,
            env_info=None
        )
        self.batch = None

    def add_batch(self, samples: Batch) -> None:
        self.batch = samples

        self.time_shape = samples.time_shape
        self.env_info = samples.env_info
        self.policy_info = samples.policy_info


class IRL:
    def __init__(self, args):
        self.env = environments.make_vec_env(
            env_name='PongNoFrameskip-v4',
            seed=0,
            one_hot_code=False,
            num_envs=8
        )

        self.buffer = DummyBuffer[policies.PPO2Info]()
        self.policy = policies.PPO2Trainer(
            env=self.env,
            network='cnn'
        )
        self.sampler = Sampler(
            env=self.env,
            policy=self.policy
        )

    def obtain_samples(self):
        return self.sampler.sample_batch(128)

    def train(self):
        for i in range(10000):
            samples = self.obtain_samples()
            self.buffer.add_batch(samples)
            #self.update_discriminator(self.buffer)
            self.policy.train(
                buffer=self.buffer,
                itr=i
            )

IRL(None).train()