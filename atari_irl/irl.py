from typing import NamedTuple, Any, Type, TypeVar, Generic, TypeVar
from collections import OrderedDict, deque
import numpy as np

from baselines import logger
from baselines.ppo2.ppo2 import safemean

from . import environments, policies, buffers, discriminators, experts, experiments, utils
from .headers import EnvInfo, PolicyInfo, Observations, Batch, Buffer


import os
import psutil
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


class RandomPolicy(policies.Policy):
    def __init__(self, config: policies.EnvConfiguration) -> None:
        super().__init__(config)

    def initialize_graph(self):
        pass

    def get_actions(self, obs: Observations) -> PolicyInfo:
        assert obs.time_shape.T is None
        assert obs.time_shape.num_envs is not None
        return PolicyInfo(
            time_shape=obs.time_shape,
            actions=np.array([
                self.config.action_space.sample()
                for _ in range(obs.time_shape.num_envs)
            ])
        )

    def get_a_logprobs(self, obs: np.ndarray, acts: np.ndarray) -> np.ndarray:
        return -np.log(self.config.action_space.n) * np.ones(len(obs))

    def train(self, buffer: Buffer, i: int) -> None:
        pass


def make_random_policy(env, **kwargs) -> RandomPolicy:
    return RandomPolicy(policies.EnvConfiguration.from_env(env))


class IRL:
    def __init__(
            self,
            *,
            env,
            cache=None,
            trajectories,
            policy_args,
            ablation=None,
            fixed_buffer_ratio=32,
            buffer_size=None,
            state_only=False,
            information_bottleneck_nats=None,
            reward_change_penalty=None,
            reward_change_constraint=None,
            transfer_function='positive',
            mean_type=None,
            rescale_type=None,
            args=None
    ):
        self.env = env

        self.mask_rewards = True
        self.train_policy = True
        self.train_discriminator = True
        overwrite_rewards = True
        build_discriminator = True

        def skip_discriminator():
            self.mask_rewards = False
            self.train_discriminator = False
            nonlocal overwrite_rewards
            overwrite_rewards = False
            nonlocal build_discriminator
            build_discriminator = False

        if ablation == 'train_rl':
            skip_discriminator()
        elif ablation == 'train_discriminator':
            self.train_policy = False
        elif ablation == 'sampler':
            skip_discriminator()
            self.train_policy = False

        policy_type = policy_args.pop('policy_type')
        make_policy_fn = {
            'Q': policies.easy_init_Q,
            'PPO2': policies.easy_init_PPO,
            'Random': make_random_policy
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

        discrim_config = discriminators.DiscriminatorConfiguration(
            max_itrs=100,
            state_only=state_only,
            score_config=discriminators.ScoreConfiguration(
                transfer_function=transfer_function,
                mean_type=mean_type,
                rescale_type=rescale_type
            ),
            bottleneck_config=discriminators.InfoBottleneckConfig(
                enabled=information_bottleneck_nats is not None,
                information_bottleneck_nats=information_bottleneck_nats
            ),
            reward_change_config=discriminators.RewardChangeConfig(
                enabled=(
                    reward_change_penalty is not None or
                    reward_change_constraint is not None
                ),
                reward_change_penalty=reward_change_penalty,
                reward_change_constraint=reward_change_constraint,
            ),
            gradient_penalty_config=discriminators.GradientPenaltyConfig(
                enabled=args.gradient_penalty is not None,
                gradient_penalty=args.gradient_penalty
            )
        )
        random_buffer = None
        if discrim_config.score_config.prepare_random_batch:
            random_sampler = policies.Sampler(
                env=self.env,
                policy=make_random_policy(env=env)
            )
            random_buffer = buffers.ViewBuffer[self.policy.InfoClass](
                discriminator=None,
                policy=random_sampler.policy,
                policy_info_class=RandomPolicy.InfoClass,
                maxlen=32,
                overwrite_rewards=False
            )
            self.env.reset()
            for i in range(32):
                random_buffer.add_batch(
                    random_sampler.sample_batch(self.batch_t)
                )
            self.env.reset()

        self.discriminator = None if not build_discriminator else discriminators.AtariAIRL(
            env=self.env,
            random_buffer=random_buffer,
            expert_buffer=experts.ExpertBuffer.from_trajectories(trajectories),
            config=discrim_config
        )

        self.T = 50000000

        self.buffer = buffers.ViewBuffer[self.policy.InfoClass](
            discriminator=self.discriminator,
            policy=self.policy,
            policy_info_class=self.policy.InfoClass,
            maxlen=int(buffer_size / env.num_envs) if buffer_size else self.fixed_buffer_ratio,
            overwrite_rewards=overwrite_rewards
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

        samples = Batch(
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
        if self.mask_rewards: assert np.isclose(samples.rewards.sum(), 0.0)
        return samples
        
    def log_performance(self, i):
        logger.logkv('itr', i)
        logger.logkv('cumulative episodes', self.total_episodes)
        logger.logkv('timesteps covered', i * self.env.num_envs * self.batch_t)
        logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in self.eval_epinfobuf]))
        logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in self.eval_epinfobuf]))
        logger.logkv('buffer size', self.buffer.time_shape.size)
        logger.logkv('memory used (GB)', psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024))
        logger.dumpkvs()

    def train_step(
            self,
            i,
            logger=None,
            log_memory=False,
            train_discriminator_now=False,
            log_now=False
    ):
        with utils.light_log_mem("sample step", log_memory):
            samples = self.obtain_samples()
            self.buffer.add_batch(samples)

        if self.train_policy and i > self.fixed_buffer_ratio:
            with utils.light_log_mem("policy train step", log_memory):
                self.policy.train_step(
                    buffer=self.buffer,
                    itr=i,
                    logger=logger if log_now else None,
                    log_freq=1
                )

        if self.train_discriminator and train_discriminator_now:
            with utils.light_log_mem("discriminator train step", log_memory):
                if self.discriminator.config.score_config.prepare_random_batch:
                    self.discriminator.train_step(
                        buffer=self.discriminator.random_buffer,
                        policy=self.discriminator.random_buffer.policy,
                        itr=i,
                        logger=None
                    )
                self.discriminator.train_step(
                    buffer=self.buffer,
                    policy=self.policy,
                    itr=i,
                    logger=logger if log_now else None
                )

        if log_now:
            self.log_performance(i)
            with utils.light_log_mem("garbage collection", log_memory):
                gc.collect()

    def save(self, i=None):
        def _save():
            with self.cache.context('irl_runner'):
                with self.cache.context('policy'):
                    self.policy.store_in_cache(self.cache)
                with self.cache.context('discriminator'):
                    self.discriminator.store_in_cache(self.cache)

        if i is not None:
            with self.cache.context(str(i)):
                _save()
        else:
            _save()

    def train(self):
        log_freq = 1
        discriminator_train_freq = self.fixed_buffer_ratio
        logger.configure()
        
        for i in range(1, int(self.T) + 1):
            self.train_step(
                i,
                logger=logger,
                log_memory=False,
                train_discriminator_now=i % discriminator_train_freq == 0,
                log_now=i % log_freq == 0
            )

            #if i % self.fixed_buffer_ratio * 10 == 0:
            #    self.save(i)

def main(
        *,
        env_name='PLECatcher-v0',
        expert_total_timesteps=10e6,
        imitator_total_timesteps=10e6,
        num_trajectories=10,
        use_trajectories_file='',
        use_expert_file='',
        update_ratio=32,
        buffer_size=None,
        seed=0,
        do_irl=True,
        expert_type='PPO',
        imitator_policy_type='PPO',
        state_only=False,
        num_envs=8,
        load_policy_initialization=None,
        information_bottleneck_nats=None,
        reward_change_penalty=None,
        reward_change_constraint=None,
        transfer_function='positive',
        mean_type=None,
        rescale_type=None,
        args=None
):
    print(f"Running process {os.getpid()}")
    cache = experiments.FilesystemCache('test_cache')
    env = environments.make_vec_env(
        env_name=env_name,
        seed=seed,
        one_hot_code=False,
        num_envs=num_envs
    )
    ncpu=64
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu,
        device_count={'GPU': 4},
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
                        # train it if we have to
                        expert = policies.PolicyTrainer(
                            policy=expert,
                            env=env
                        ).cached_train(cache)

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
            'total_timesteps': int(imitator_total_timesteps),
            'load_initialization': load_policy_initialization
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
                        fixed_buffer_ratio=update_ratio,
                        buffer_size=buffer_size,
                        state_only=state_only,
                        information_bottleneck_nats=information_bottleneck_nats,
                        reward_change_penalty=reward_change_penalty,
                        reward_change_constraint=reward_change_constraint,
                        transfer_function=transfer_function,
                        mean_type=mean_type,
                        rescale_type=rescale_type,
                        args=args,
                    )
                    irl_runner.train()

    env.reset()
    env.close()

    return
