from typing import NamedTuple, Any, Type, TypeVar, Generic, TypeVar
from collections import OrderedDict, deque
import numpy as np
import gym

from baselines.common.vec_env import VecEnv
from baselines import logger
from baselines.ppo2.ppo2 import safemean

from . import environments, policies, buffers, discriminators, experts
from .headers import TimeShape, EnvInfo, PolicyInfo, Observations, PolicyTrainer, Batch, Buffer, SamplerState
from .utils import Stacker

import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        self.dones = None
        self.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()
        self.dones = np.zeros(self.num_envs).astype(np.bool)

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
            
            obs_copy = self.obs.copy()
            dones_copy = self.dones.copy()

            self.obs[:], rewards, self.dones, epinfos = self.env.step(actions)
            if show:
                self.env.render()
                
            env_info_stacker.append(EnvInfo(
                time_shape=time_step,
                obs=obs_copy,
                next_obs=self.obs.copy(),
                rewards=rewards,
                dones=dones_copy,
                next_dones=self.dones.copy(),
                epinfobuf=[
                    info.get('episode') for info in epinfos if info.get('episode')
                ]
            ))

        return Batch(
            time_shape=time_shape,
            sampler_state=SamplerState(
                obs=self.obs.copy(),
                dones=self.dones.copy()
            ),
            env_info=EnvInfo(
                time_shape=time_shape,
                obs=np.array(env_info_stacker.obs),
                next_obs=np.array(env_info_stacker.next_obs),
                rewards=np.array(env_info_stacker.rewards),
                dones=np.array(env_info_stacker.dones),
                next_dones=np.array(env_info_stacker.next_dones),
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


class IRL:
    def __init__(self, args, fname=None):
        self.env = environments.make_vec_env(
            env_name='PLECatcher-v0',
            seed=0,
            one_hot_code=False,
            num_envs=1
        )

        self.mask_rewards = True

        self.discriminator = discriminators.AtariAIRL(
            env=self.env,
            expert_buffer=experts.ExpertBuffer.from_trajectories(pickle.load(open(fname, 'rb')))
        )

        self.buffer = buffers.ViewBuffer[policies.QInfo](
            self.discriminator, policies.QInfo
        )
        self.policy = policies.QTrainer(
            env=self.env,
            network='conv_only'
        )
        self.sampler = Sampler(
            env=self.env,
            policy=self.policy
        )

        self.eval_epinfobuf = deque(maxlen=100)
        self.total_episodes = 0
        self.batch_t = 1
        
    def obtain_samples(self):
        batch = self.sampler.sample_batch(self.batch_t)
        self.total_episodes += len(batch.env_info.epinfobuf)
        self.eval_epinfobuf.extend(batch.env_info.epinfobuf)

        if self.mask_rewards:
            rewards = np.zeros(batch.env_info.rewards.shape)
        else:
            rewards = batch.env_info.rewards

        return Batch(
            time_shape=batch.time_shape,
            sampler_state=batch.sampler_state,
            env_info=EnvInfo(
                time_shape=batch.env_info.time_shape,
                obs=batch.env_info.obs,
                next_obs=batch.env_info.next_obs,
                rewards=rewards,
                dones=batch.env_info.dones,
                next_dones=batch.env_info.next_dones,
                epinfobuf=[]
            ),
            policy_info=batch.policy_info
        )
        
    def log_performance(self, i):
        logger.logkv('itr', i)
        logger.logkv('cumulative episodes', self.total_episodes)
        logger.logkv('timesteps covered', i * self.env.num_envs * self.batch_t)
        logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in self.eval_epinfobuf]))
        logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in self.eval_epinfobuf]))
        logger.logkv('buffer size', self.buffer.time_shape.size)
        logger.dumpkvs()

    def train(self):
        log_freq = 100
        logger.configure()
        
        for i in range(int(100000)):
            samples = self.obtain_samples()
            self.buffer.add_batch(samples)
            if self.mask_rewards: assert np.isclose(samples.rewards.sum(), 0.0)
            self.policy.train(
                buffer=self.buffer,
                discriminator=self.discriminator,
                itr=i,
                log_freq=log_freq
            )
            if i % 128 == 0:
                self.discriminator.fit(
                    buffer=self.buffer,
                    policy=self.policy,
                    itr=i
                )
            if i % log_freq == 0:
                self.log_performance(i)


def main():
    # train an expert
    # run the expert to generate trajectories
    # train an encoder
    # run IRL
    pass