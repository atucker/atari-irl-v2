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

import tensorflow as tf


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
            env,
            cache=None,
            trajectories,
            policy_args,
            ablation=None,
            score_discrim=True,
            fixed_buffer_ratio=32,
            buffer_size=None
    ):
        self.env = env

        self.mask_rewards = True
        self.train_discriminator = True
        build_discriminator = True

        if ablation == 'train_rl':
            self.mask_rewards = False
            self.train_discriminator = False
            build_discriminator = False

        self.discriminator = None if not build_discriminator else discriminators.AtariAIRL(
            env=self.env,
            expert_buffer=experts.ExpertBuffer.from_trajectories(trajectories),
            score_discrim=score_discrim,
            max_itrs=100
        )

        self.T = 5000000000

        policy_type = policy_args.pop('policy_type')
        make_policy_fn = {
            'Q': policies.easy_init_Q,
            'PPO2': policies.easy_init_PPO
        }[policy_type]
        policy_class = {
            'Q': policies.QTrainer,
            'PPO2': policies.PPO2Trainer
        }[policy_type]

        self.batch_t = 128 if policy_type == 'PPO2' else 1
        self.fixed_buffer_ratio = fixed_buffer_ratio
        if policy_type == 'Q':
            self.fixed_buffer_ratio *= 128
            policy_args['learning_starts'] = self.fixed_buffer_ratio

        self.policy = make_policy_fn(
            env=self.env,
            **policy_args
        )

        self.buffer = buffers.ViewBuffer[policy_class.info_class](
            discriminator=self.discriminator,
            policy=self.policy,
            policy_info_class=policy_class.info_class,
            maxlen=int(buffer_size / env.num_envs) if buffer_size else self.fixed_buffer_ratio
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
        log_freq = 1
        discriminator_train_freq = self.fixed_buffer_ratio
        logger.configure()
        
        for i in range(int(self.T)):
            samples = self.obtain_samples()
            self.buffer.add_batch(samples)
            if self.mask_rewards: assert np.isclose(samples.rewards.sum(), 0.0)
            self.policy.train_step(
                buffer=self.buffer,
                itr=i,
                log_freq=log_freq,
                logger=logger
            )
            if i % discriminator_train_freq == 0 and self.train_discriminator:
                self.discriminator.train_step(
                    buffer=self.buffer,
                    policy=self.policy,
                    itr=i,
                    logger=logger
                )

            if (
                self.train_discriminator and i % discriminator_train_freq == 0 or
                not self.train_discriminator and i % log_freq == 0
            ):
                self.log_performance(i)


def main(
        *,
        env_name='PLECatcher-v0',
        expert_total_timesteps=10e6,
        imitator_total_timesteps=10e6,
        num_trajectories=10,
        use_trajectories_file='',
        use_expert_file='',
        score_discrim=True,
        update_ratio=32,
        buffer_size=None,
        seed=0,
        do_irl=True,
        expert_type='PPO',
        imitator_policy_type='PPO'
):
    print(f"Running process {os.getpid()}")
    cache = experiments.FilesystemCache('test_cache')
    env = environments.make_vec_env(
        env_name=env_name,
        seed=seed,
        one_hot_code=False,
        num_envs=8
    )
    ncpu=32
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu,
        device_count={'GPU': 2},
    )
    config.gpu_options.allow_growth=True

    if use_trajectories_file:
        import joblib
        trajectories = joblib.load(use_trajectories_file)
    else:
        with tf.Graph().as_default():
            with tf.Session(config=config) as sess:
                tf.random.set_random_seed(seed)
                if use_expert_file:
                    expert = experiments.TfObject.create_from_file(
                        env=env,
                        fname=use_expert_file
                    )
                else:
                    with cache.context('expert'):
                        if expert_type == 'Q':
                            expert = policies.easy_init_Q(
                                env=env,
                                network='conv_only',
                                total_timesteps=int(expert_total_timesteps)
                            )
                        else:
                            expert = policies.easy_init_PPO(
                                env=env,
                                network='cnn',
                                total_timesteps=int(expert_total_timesteps)
                            )
                            expert.cached_train(cache)

                with cache.context('trajectories'):
                    with cache.context(cache.hash_key(expert.key)):
                        sampler = policies.Sampler(
                            env=env,
                            policy=expert
                        )
                        trajectories = sampler.cached_sample_trajectories(
                            cache,
                            num_trajectories=num_trajectories,
                            one_hot_code=True
                        )

    # TODO(Aaron): Train an encoder

    policy_args = {}
    if imitator_policy_type == 'Q':
        policy_args = {
            'policy_type': 'Q',
            'network': 'conv_only',
            'total_timesteps': int(imitator_total_timesteps)
        }
    else:
        policy_args = {
            'policy_type': 'PPO2',
            'network': 'cnn',
            'total_timesteps': int(imitator_total_timesteps)
        }

    if do_irl:
        with tf.Graph().as_default():
            with tf.Session(config=config) as sess:
                tf.random.set_random_seed(seed)
                with cache.context('irl'):
                    irl_runner = IRL(
                        env=env,
                        cache=cache,
                        trajectories=trajectories,
                        policy_args=policy_args,
                        score_discrim=score_discrim,
                        fixed_buffer_ratio=update_ratio,
                        buffer_size=buffer_size
                    )
                    irl_runner.train()

    env.reset()
    env.close()
