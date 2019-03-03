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

from .experiments import TfObject, Configuration
from .buffers import ViewBuffer


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


class PPO2Trainer(PolicyTrainer, TfObject):
    info_class = PPO2Info
    
    def __init__(
            self,
            env: VecEnv,
            network: str,
            **network_kwargs
    ) -> None:
        super().__init__(env)
        nenvs = env.num_envs
        total_timesteps = 10e6

        self.gamma = 0.99
        self.lam = 0.95
        self.lr = 2.5e-4
        self.cliprange = 0.1
        self.nsteps = 128
        self.nminibatches = 4
        self.noptepochs = 4
        self.nbatch = nenvs * self.nsteps
        self.nbatch_train = self.nbatch // self.nminibatches
        self.nupdates = total_timesteps // self.nbatch

        self.log_interval = 1
        self.save_interval = 0

        self.tfirststart = None

        self.model = baselines.ppo2.model.Model(
            policy=baselines.common.policies.build_policy(
                env,
                network,
                **network_kwargs
            ),
            ob_space=env.observation_space,
            ac_space=env.action_space,
            nbatch_act=env.num_envs,
            nbatch_train=self.nbatch_train,
            nsteps=self.nsteps,
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
    
    """
    def get_probabilities_for_obs(self, obs: np.ndarray) -> np.ndarray:
        tm = self.model.train_model
        return tf.get_default_session().run(
            tf.nn.softmax(tm.pd.logits),
            {tm.X: obs}
        )

    def get_a_logprobs(self, obs: np.ndarray, acts: np.ndarray) -> np.ndarray:
        probs = self.get_probabilities_for_obs(obs)
        ""
        utils.batched_call(
            # needs to be a tuple for the batched call to work
            lambda obs: (self.get_probabilities_for_obs(obs),),
            self.model.train_model.X.shape[0].value,
            (obs,),
            check_safety=False
        )[0]
        ""
        return np.log((probs * acts).sum(axis=1))
    """

    def train_step(self, buffer: Buffer[PPO2Info], itr: int, log_freq=1000) -> None:
        tstart = time.time()
        frac = 1.0 - (itr - 1.0) / self.nupdates
        if itr == 0:
            self.tfirststart=tstart

        # Calculate the learning rate
        lrnow = self.lr * frac
        # Calculate the cliprange
        cliprangenow = self.cliprange * frac

        
        # discount/bootstrap off value fn
        last_values = self.model.value(
            buffer.sampler_state.obs,
            S=None,
            M=buffer.sampler_state.dones
        )
        mb_returns = np.zeros_like(buffer.rewards)
        mb_advs = np.zeros_like(buffer.rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - buffer.sampler_state.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - buffer.dones[t+1]
                nextvalues = buffer.policy_info.values[t+1]
            delta = buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - buffer.policy_info.values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + buffer.policy_info.values

        obs = sf01(buffer.obs)
        returns = sf01(mb_returns)
        masks = sf01(buffer.dones)
        actions = sf01(buffer.acts)
        values = sf01(buffer.policy_info.values)
        neglogpacs = sf01(buffer.policy_info.neglogpacs)
        
        # Index of each element of batch_size
        # Create the indices array
        inds = np.arange(self.nbatch)
        mblossvals = []

        for _ in range(self.noptepochs):
            assert self.nbatch % self.nminibatches == 0
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

        if itr % log_freq == 0 or itr == 1:
            # Calculates if value function is a good predictor of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", itr * self.nsteps)
            logger.logkv("nupdates", itr)
            logger.logkv("total_timesteps", itr * self.nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('time_elapsed', tnow - self.tfirststart)
            for (lossval, lossname) in zip(lossvals, self.model.loss_names):
                logger.logkv(lossname, lossval)

        if self.save_interval and (itr % self.save_interval == 0 or itr == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % itr)
            print('Saving to', savepath)
            self.model.save(savepath)


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
        gamma=1.0,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        total_timesteps=100000,
        param_noise=False,
        learning_starts=1000,
        train_freq=1,
        batch_size=32,
        target_network_update_freq=500,
        prioritized_replay=False
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


class QConfig(Configuration):
    default_values = dict(
        training=QTrainingConfiguration(),
        network=NetworkKwargsConfiguration()
    )


class QTrainer(PolicyTrainer, TfObject):
    info_class = QInfo
    class_registration_name = 'QNetwork'

    def __init__(
            self,
            env: VecEnv,
            network: str,
            **network_kwargs
    ) -> None:
        PolicyTrainer.__init__(self, env)

        self.train_model = None
        self.update_target = None
        self.debug = None
        self.act = None

        self.env = env
        TfObject.__init__(self, QConfig(
            network_cfg=NetworkKwargsConfiguration(
                network=network,
                network_kwargs=network_kwargs
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
            optimizer=tf.train.AdamOptimizer(learning_rate=self.config.training.trn_cfg.lr),
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
        qs = self.debug['q_values'](obs)
        random_p = 1.0 / self.action_space.n
        logp_argmax = np.log((1.0 - self.last_explore_frac) * 1.0 + self.last_explore_frac * random_p)
        logp_other = np.log(self.last_explore_frac) - np.log(self.action_space.n)
        
        #a_logprobs = acts.copy()
        #a_logprobs[~a_logprobs.astype(np.bool)] = logp_other
        #a_logprobs[a_logprobs.astype(np.bool)] = logp_argmax
        
        a_logprobs = (acts.argmax(axis=1) == qs.argmax(axis=1))
        a_logprobs[~a_logprobs.astype(np.bool)] == logp_other
        a_logprobs[a_logprobs.astype(np.bool)] == logp_argmax
        return a_logprobs

    def train_step(self, buffer: Buffer[QInfo], itr: int, log_freq=1000) -> None:
        assert itr == self.t
        t = itr
        
        if t > self.config.training.learning_starts and t % self.config.training.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            obses_t, actions, rewards, obses_tp1, dones = buffer.sample_batch(
                'obs', 'acts', 'rewards', 'next_obs', 'next_dones',
                batch_size=self.trn_cfg.batch_size
            )
            weights, batch_idxes = np.ones_like(rewards), None
            td_errors = self.train_model(obses_t, actions, rewards, obses_tp1, dones, weights)

        if t > self.config.training.learning_starts and t % self.config.training.target_network_update_freq == 0:
            # Update target network periodically.
            self.update_target()
            
        if itr % log_freq == 0:
            logger.logkv('"% time spent exploring"', int(100 * buffer.policy_info.explore_frac[-1]))
            
        self.t += 1

    def train(self):
        log_freq = 100
        logger.configure()

        sampler = Sampler(env=self.env, policy=self)
        buffer = ViewBuffer[QInfo](None, QInfo)

        for i in range(int(self.config.training.total_timesteps / self.config.training.batch_size)):
            batch = sampler.sample_batch(self.config.training.batch_size)
            buffer.add_batch(batch)

            if i % self.config.training.train_freq == 0:
                self.train_step(
                    buffer=buffer,
                    itr=i,
                    log_freq=log_freq
                )

            if i % 4096 == 0:
                print("Doing a cache roundtrip...")
                self.store_in_cache(self.cache, key_mod='_training')
                self.restore_values_from_cache(self.cache, key_mod='_training')


TfObject.register_cachable_class('QNetwork', QTrainer)
