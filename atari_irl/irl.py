from typing import NamedTuple, Any, Type, TypeVar, Generic, TypeVar
from collections import OrderedDict, deque
import numpy as np
import gym

from baselines.common.vec_env import VecEnv
from baselines import logger
from baselines.ppo2.ppo2 import safemean

from . import environments, policies, buffers, discriminators, experts, experiments
from .headers import TimeShape, EnvInfo, PolicyInfo, Observations, PolicyTrainer, Batch, Buffer, SamplerState


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


class IRL:
    def __init__(
            self,
            *,
            cache=None,
            expert_trajectory_fname=None,
            ablation=None
    ):
        self.env = environments.make_vec_env(
            env_name='PLECatcher-v0',
            seed=0,
            one_hot_code=False,
            num_envs=1
        )

        self.mask_rewards = True
        self.train_discriminator = True
        build_discriminator = True

        if ablation == 'train_rl':
            self.mask_rewards = False
            self.train_discriminator = False
            build_discriminator = False

        self.discriminator = None if not build_discriminator else discriminators.AtariAIRL(
            env=self.env,
            expert_buffer=experts.ExpertBuffer.from_trajectories(
                pickle.load(open(expert_trajectory_fname, 'rb'))
            )
        )

        self.buffer = buffers.ViewBuffer[policies.QInfo](
            self.discriminator,
            policies.QInfo
        )
        self.policy = policies.QTrainer(
            env=self.env,
            network='conv_only'
        )
        self.sampler = policies.Sampler(
            env=self.env,
            policy=self.policy
        )

        self.cache = cache if cache is not None else experiments.FilesystemCache('test_cache')
        if self.policy.key in self.cache:
            print(f"Restoring policy {self.policy.key} from cache!")
            self.policy.restore_values_from_cache(self.cache)

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
        
        for i in range(int(50000)):
            samples = self.obtain_samples()
            self.buffer.add_batch(samples)
            if self.mask_rewards: assert np.isclose(samples.rewards.sum(), 0.0)
            self.policy.train_step(
                buffer=self.buffer,
                itr=i,
                log_freq=log_freq
            )
            if i % 1024 == 0 and self.train_discriminator:
                self.discriminator.train_step(
                    buffer=self.buffer,
                    policy=self.policy,
                    itr=i
                )
            if i % log_freq == 0:
                self.log_performance(i)

            if i % 4096 == 0:
                print("Doing a cache roundtrip...")
                self.policy.store_in_cache(self.cache, key_mod='_training')
                self.policy.restore_values_from_cache(self.cache, key_mod='_training')


def main():

    # train an expert
    # run the expert to generate trajectories
    # train an encoder
    # run IRL
    pass
