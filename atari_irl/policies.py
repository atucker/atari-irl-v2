from typing import NamedTuple, Dict, Any
import gym
import baselines
import numpy as np
import time
import os
import os.path as osp
from collections import deque

from baselines import logger
from baselines.common.vec_env import VecEnv
import baselines.common.policies

import baselines.ppo2.model
from baselines.ppo2.ppo2 import safemean, explained_variance
from baselines.ppo2.runner import sf01

import baselines.deepq
from baselines.deepq import deepq
import baselines.common.tf_util as U
from baselines.common.vec_env import VecEnv

import tensorflow as tf

from .headers import PolicyTrainer, PolicyInfo, Observations, Buffer, TimeShape
from .discriminators import AtariAIRL
from .utils import one_hot

from .utils import Stacker
from .headers import Batch, EnvInfo, SamplerState

from .experiments import TfObject, Configuration, FilesystemCache
from .buffers import ViewBuffer, DummyBuffer


class EnvSpec(NamedTuple):
    """
    The baselines code wants a full environment, but only uses the
    action and observation space definitions. Let's hope that that stays true,
    because otherwise it makes the interface less clean...
    """
    observation_space: gym.Space
    action_space: gym.Space


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

        time_step = TimeShape(num_envs=self.num_envs)
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
                time_shape=time_shape,
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

    def sample_trajectories(self, num_trajectories=10, render=False, one_hot_code=False):
        completed_trajectories = []
        observations = [[] for _ in range(self.env.num_envs)]
        actions = [[] for _ in range(self.env.num_envs)]

        time_step = TimeShape(num_envs=self.num_envs)

        obs = self.env.reset()
        while len(completed_trajectories) < num_trajectories:
            policy_step = self.policy.get_actions(Observations(
                time_shape=time_step,
                observations=self.obs
            ))
            acts = policy_step.actions

            # We append observation, actions tuples here, since they're defined now
            for i, (o, a) in enumerate(zip(obs, acts)):
                observations[i].append(o)
                actions[i].append(a)

            # Figure out our consequences
            obs, _, dones, _ = self.env.step(acts)
            if render:
                self.env.render()

            # If we're done, then append that trajectory and restart
            for i, done in enumerate(dones):
                if done:
                    completed_trajectories.append({
                        'observations': np.array(observations[i]).astype(np.float16),
                        'actions': one_hot(actions[i], self.env.action_space.n) if one_hot_code else np.vstack(actions[i]),
                    })
                    observations[i] = []
                    actions[i] = []

        np.random.shuffle(completed_trajectories)
        return completed_trajectories[:num_trajectories]

    def cached_sample_trajectories(self, cache, *, num_trajectories=10, render=False, one_hot_code=False):
        key = self.policy.key + f';trajectories:num_trajectories={num_trajectories},one_hot={one_hot_code}'

        if key not in cache:
            print("Sampling Trajectories!")
            cache[key] = self.sample_trajectories(
                num_trajectories=num_trajectories,
                render=render,
                one_hot_code=one_hot_code
            )
            print(f"Stored trajectories in {key}")
        else:
            print(f"Restoring trajectories from {key}")

        return cache[key]


class EnvConfiguration(Configuration):
    default_values = dict(
        name='ERROR: This has to be defined'
    )


class NetworkKwargsConfiguration(Configuration):
    default_values = dict(
        network='conv_only',
        # Our life is much easier if we often just use the default arguments
        # for baselines function creation, and so we'll keep this here
        # TODO(Aaron): serialize the baselines version as part of the cache
        network_kwargs={},
        serialization_scheme='overrides_baselines_default_kwargs'
    )


class PPO2TrainingConfiguration(Configuration):
    default_values = dict(
        total_timesteps=10e6,
        gamma=0.99,
        lam=0.95,
        lr=2.5e-4,
        cliprange=0.1,
        nsteps=128,
        nminibatches=4,
        noptepochs=4,
        nenvs=8
    )


class PPO2Config(Configuration):
    default_values = dict(
        training=PPO2TrainingConfiguration(),
        network=NetworkKwargsConfiguration(),
        env=EnvConfiguration()
    )


class PPO2Info(PolicyInfo):
    _fields = ('time_shape', 'actions', 'values', 'neglogpacs')

    def __init__(
        self, *,
        time_shape: TimeShape,
        actions: np.ndarray,
        values: np.ndarray,
        neglogpacs: np.ndarray
    ) -> None:
        super().__init__(time_shape=time_shape, actions=actions)
        self.values = values
        self.neglogpacs = neglogpacs
        
    @property
    def lprobs(self):
        return self.neglogpacs * -1


class PPO2Trainer(PolicyTrainer, TfObject):
    info_class = PPO2Info
    class_registration_name = 'PPO2Network'
    
    def __init__(
            self,
            env: VecEnv,
            network: str,
            total_timesteps: int=10e6,
            **network_kwargs
    ) -> None:
        super().__init__(env)

        self.env = env

        self.log_interval = 1
        self.save_interval = 0
        
        training_config = PPO2TrainingConfiguration(
            total_timesteps=total_timesteps,
            nenvs=env.num_envs
        )
        self.nbatch = training_config.nenvs * training_config.nsteps
        self.nbatch_train = self.nbatch // training_config.nminibatches
        self.nupdates = training_config.total_timesteps // self.nbatch

        self.tfirststart = None
        self.model = None
        
        TfObject.__init__(self, PPO2Config(
            training=training_config,
            network=NetworkKwargsConfiguration(
                network=network,
                network_kwargs=network_kwargs
            ),
            env=EnvConfiguration(
                name=env.unwrapped.specs[0].id
            )
        ))

    def initialize_graph(self):
        print('initializing!')
        self.model = baselines.ppo2.model.Model(
            policy=baselines.common.policies.build_policy(
                self.env,
                self.config.network.network,
                **self.config.network.network_kwargs
            ),
            ob_space=self.env.observation_space,
            ac_space=self.env.action_space,
            nbatch_act=self.env.num_envs,
            nbatch_train=self.nbatch_train,
            nsteps=self.config.training.nsteps,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5
        )

    def get_actions(self, obs_batch: Observations) -> PPO2Info:
        actions, values, _, neglogpacs = self.model.step(
            obs_batch.observations,
            # We're not going to support recurrent policies for now
            # Hopefully that doesn't break things...
            S=None,
            M=None
        )
        return PPO2Info(
            time_shape=obs_batch.time_shape,
            actions=actions,
            values=values,
            neglogpacs=neglogpacs
        )

    def get_probabilities_for_obs(self, obs: np.ndarray) -> np.ndarray:
        tm = self.model.train_model
        return tf.get_default_session().run(
            tf.nn.softmax(tm.pd.logits),
            {tm.X: obs}
        )

    def get_a_logprobs(self, obs: np.ndarray, acts: np.ndarray) -> np.ndarray:
        probs = self.get_probabilities_for_obs(obs)
        return np.log((probs * acts).sum(axis=1))

    def train_step(self, buffer: Buffer[PPO2Info], itr: int, log_freq=1000, logger=None, cache=None) -> None:
        tstart = time.time()
        frac = 1.0 - (itr - 1.0) / self.nupdates
        if itr == 0:
            self.tfirststart=tstart

        # Calculate the learning rate
        lrnow = self.config.training.lr * frac
        # Calculate the cliprange
        cliprangenow = self.config.training.cliprange * frac
        
        batch = buffer.latest_batch
        
        # discount/bootstrap off value fn
        last_values = self.model.value(
            batch.sampler_state.obs,
            S=None,
            M=batch.sampler_state.dones
        )
        mb_returns = np.zeros_like(batch.rewards)
        mb_advs = np.zeros_like(batch.rewards)
        lastgaelam = 0
        for t in reversed(range(self.config.training.nsteps)):
            if t == self.config.training.nsteps - 1:
                nextnonterminal = 1.0 - batch.sampler_state.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - batch.dones[t+1]
                nextvalues = batch.policy_info.values[t+1]
            delta = batch.rewards[t] + self.config.training.gamma * nextvalues * nextnonterminal - batch.policy_info.values[t]
            mb_advs[t] = lastgaelam = delta + self.config.training.gamma * self.config.training.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + batch.policy_info.values

        obs = sf01(batch.obs)
        returns = sf01(mb_returns)
        masks = sf01(batch.dones)
        actions = sf01(batch.acts)
        values = sf01(batch.policy_info.values)
        neglogpacs = sf01(batch.policy_info.neglogpacs)
        
        # Index of each element of batch_size
        # Create the indices array
        inds = np.arange(self.nbatch)
        mblossvals = []

        for _ in range(self.config.training.noptepochs):
            assert self.nbatch % self.config.training.nminibatches == 0
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.nbatch, self.nbatch_train):
                end = start + self.nbatch_train
                mbinds = inds[start:end]
                slices = (
                    arr[mbinds] for arr in
                    (obs, returns, masks, actions, values, neglogpacs)
                )
                loss = self.model.train(lrnow, cliprangenow, *slices)
                mblossvals.append(loss)

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(self.nbatch / (tnow - tstart))

        if itr % log_freq == 0 or itr == 1 or logger:
            # Calculates if value function is a good predictor of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", itr * self.config.training.nsteps)
            logger.logkv("nupdates", itr)
            logger.logkv("total_timesteps", itr * self.nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('time_elapsed', tnow - self.tfirststart)
            logger.logkv("mean reward", np.mean(batch.rewards))
            logger.logkv("mean return", np.mean(mb_returns))
            for (lossval, lossname) in zip(lossvals, self.model.loss_names):
                logger.logkv(lossname, lossval)

        if self.save_interval and (itr % self.save_interval == 0 or itr == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % itr)
            print('Saving to', savepath)
            self.model.save(savepath)
            
    def train(self, cache):
        print(f"Training PPO2 policy with key {self.key}")
        log_freq = self.log_interval
        logger.configure()

        sampler = Sampler(env=self.env, policy=self)
        buffer = DummyBuffer[PPO2Info](
            overwrite_rewards=False,
            overwrite_logprobs=False
        )

        total_episodes = 0
        total_timesteps = 0
        i = 0
        eval_epinfobuf = []
        while total_timesteps < self.config.training.total_timesteps:
            batch = sampler.sample_batch(self.config.training.nsteps)
            eval_epinfobuf.extend(batch.env_info.epinfobuf)
            buffer.add_batch(batch)

            self.train_step(
                buffer=buffer,
                itr=i,
                log_freq=log_freq,
                logger=logger,
                cache=cache
            )

            if i % log_freq == 0:
                logger.logkv('itr', i)
                logger.logkv('cumulative episodes', total_episodes)
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]))
                logger.logkv('buffer size', buffer.time_shape.size)
                logger.dumpkvs()

            if i % int(self.config.training.total_timesteps / 10) == 0:
                print("Doing a cache roundtrip...")
                with cache.context('training'):
                    with cache.context(str(i)):
                        self.store_in_cache(cache)
                        self.restore_values_from_cache(cache)
                        
            i += 1
            total_episodes += len(batch.env_info.epinfobuf)
            total_timesteps += self.config.training.nsteps * self.config.training.nenvs


TfObject.register_cachable_class('PPO2Network', PPO2Trainer)


class QInfo(PolicyInfo):
    _fields = ('time_shape', 'actions', 'explore_frac')

    def __init__(
            self, *,
            time_shape: TimeShape,
            actions: np.ndarray,
            explore_frac: float
    ) -> None:
        super().__init__(time_shape=time_shape, actions=actions)
        self.explore_frac = explore_frac


class QTrainingConfiguration(Configuration):
    default_values = dict(
        lr=5e-4,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        total_timesteps=100000,
        param_noise=False,
        learning_starts=1000,
        train_freq=1,
        batch_size=32,
        target_network_update_freq=10000,
        prioritized_replay=False
    )


class QConfig(Configuration):
    default_values = dict(
        training=QTrainingConfiguration(),
        network=NetworkKwargsConfiguration(),
        env=EnvConfiguration()
    )


class QTrainer(PolicyTrainer, TfObject):
    info_class = QInfo
    class_registration_name = 'QNetwork'

    def __init__(
            self,
            env: VecEnv,
            network: str,
            total_timesteps=100000,
            learning_starts=1000,
            **network_kwargs
    ) -> None:
        PolicyTrainer.__init__(self, env)

        self.train_model = None
        self.update_target = None
        self.debug = None
        self.act = None

        self.env = env
        TfObject.__init__(self, QConfig(
            training=QTrainingConfiguration(
                total_timesteps=total_timesteps,
                learning_starts=learning_starts
            ),
            network=NetworkKwargsConfiguration(
                network=network,
                network_kwargs=network_kwargs
            ),
            env=EnvConfiguration(
                name=env.unwrapped.specs[0].id
            )
        ))

        # Create the replay buffer
        self.beta_schedule = None

        # Create the schedule for exploration starting from 1.
        self.exploration = deepq.LinearSchedule(
            schedule_timesteps=int(self.config.training.exploration_fraction * self.config.training.total_timesteps),
            initial_p=1.0,
            final_p=self.config.training.exploration_final_eps
        )

        U.initialize()
        self.update_target()

        self.action_space = env.action_space
        self.t = 0
        self.env = env

    def initialize_graph(self):
        env = self.env
        q_func = deepq.build_q_func(
            self.config.network.network,
            self.config.network.network_kwargs
        )
        observation_space = env.observation_space

        def make_obs_ph(name):
            return deepq.ObservationInput(observation_space, name=name)

        act, self.train_model, self.update_target, self.debug = baselines.deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.config.training.lr),
            gamma=self.config.training.gamma,
            grad_norm_clipping=10,
            param_noise=self.config.training.param_noise
        )
        assert 'q_values' in self.debug

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': env.action_space.n,
        }

        self.act = deepq.ActWrapper(act, act_params)

    def get_actions(self, obs_batch: Observations) -> QInfo:
        # Take action and update exploration to the newest value
        kwargs = {}
        update_eps = self.exploration.value(self.t)
        update_param_noise_threshold = 0.

        self.last_explore_frac = update_eps
        
        return QInfo(
            time_shape=obs_batch.time_shape,
            actions=self.act(
                np.array(obs_batch.observations)[None],
                update_eps=update_eps,
                **kwargs
            ),
            explore_frac = [update_eps]
        )
    
    def get_a_logprobs(self, obs: np.ndarray, acts: np.ndarray) -> np.ndarray:
        if isinstance(acts, list) or len(acts.shape) == 1:
            acts = one_hot(acts, self.act_space.n)
        qs = self.debug['q_values'](obs)
        random_p = 1.0 / self.action_space.n
        logp_argmax = np.log((1.0 - self.last_explore_frac) * 1.0 + self.last_explore_frac * random_p)
        logp_other = np.log(self.last_explore_frac) - np.log(self.action_space.n)
        
        #a_logprobs = acts.copy()
        #a_logprobs[~a_logprobs.astype(np.bool)] = logp_other
        #a_logprobs[a_logprobs.astype(np.bool)] = logp_argmax
        
        a_logprobs = (acts.argmax(axis=1) == qs.argmax(axis=1))
        a_logprobs[~a_logprobs.astype(np.bool)] = logp_other
        a_logprobs[a_logprobs.astype(np.bool)] = logp_argmax
        return a_logprobs

    def train_step(self, buffer: Buffer[QInfo], itr: int, log_freq=1000, logger=None) -> None:
        assert itr == self.t
        t = itr

        if t > self.config.training.learning_starts and t % self.config.training.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            obses_t, actions, rewards, obses_tp1, dones = buffer.sample_batch(
                'obs', 'acts', 'rewards', 'next_obs', 'next_dones',
                batch_size=self.config.training.batch_size
            )
            weights, batch_idxes = np.ones_like(rewards), None
            td_errors = self.train_model(obses_t, actions, rewards, obses_tp1, dones, weights)

        if t > self.config.training.learning_starts and t % self.config.training.target_network_update_freq == 0:
            # Update target network periodically.
            self.update_target()
            
        if itr % log_freq == 0:
            logger.logkv('"% time spent exploring"', int(100 * buffer.policy_info.explore_frac[-1]))
            if t > self.config.training.learning_starts and t % self.config.training.train_freq == 0:
                logger.logkv("mean reward", np.mean(rewards))
        self.t += 1

    def train(self, cache):
        print(f"Training Q Learning policy with key {self.key}")
        log_freq = 100
        logger.configure()

        sampler = Sampler(env=self.env, policy=self)
        buffer = ViewBuffer[QInfo](None, QInfo)

        total_episodes = 0
        eval_epinfobuf = []
        for i in range(int(self.config.training.total_timesteps)):
            batch = sampler.sample_batch(1)
            total_episodes += len(batch.env_info.epinfobuf)
            eval_epinfobuf.extend(batch.env_info.epinfobuf)
            buffer.add_batch(batch)

            if i % self.config.training.train_freq == 0:
                self.train_step(
                    buffer=buffer,
                    itr=i,
                    log_freq=log_freq
                )

            if i % log_freq == 0:
                logger.logkv('itr', i)
                logger.logkv('cumulative episodes', total_episodes)
                logger.logkv('timesteps covered', i * self.env.num_envs)
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]))
                logger.logkv('buffer size', buffer.time_shape.size)
                logger.dumpkvs()

            if i % int(self.config.training.total_timesteps / 10) == 0:
                print("Doing a cache roundtrip...")
                with cache.context('training'):
                    with cache.context(str(i)):
                        self.store_in_cache(cache)
                        self.restore_values_from_cache(cache)


TfObject.register_cachable_class('QNetwork', QTrainer)
