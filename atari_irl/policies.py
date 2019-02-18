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

import tensorflow as tf

from .headers import PolicyTrainer, PolicyInfo, Observations, Buffer, TimeShape
from .discriminators import AtariAIRL
from .utils import one_hot

from .experiments import TfObject, Configuration


class EnvSpec(NamedTuple):
    """
    The baselines code wants a full environment, but only uses the
    action and observation space definitions. Let's hope that that stays true,
    because otherwise it makes the interface less clean...
    """
    observation_space: gym.Space
    action_space: gym.Space


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
        network_kwargs={},
        env=None
    )
    attrs_exclude_from_key={'env',}


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
        self.trn_cfg = QTrainingConfiguration()

        self.train_model = None
        self.update_target = None
        self.debug = None
        self.act = None

        TfObject.__init__(
            self,
            NetworkKwargsConfiguration(
                network=network,
                network_kwargs=network_kwargs
            )
        )

        # Create the replay buffer
        self.beta_schedule = None

        # Create the schedule for exploration starting from 1.
        self.exploration = deepq.LinearSchedule(
            schedule_timesteps=int(self.trn_cfg.exploration_fraction * self.trn_cfg.total_timesteps),
            initial_p=1.0,
            final_p=self.trn_cfg.exploration_final_eps
        )

        U.initialize()
        self.update_target()

        self.action_space = env.action_space
        self.t = 0
        self.env = env

    def initialize_graph(self):
        env = self.tf_obj_config.env
        q_func = deepq.build_q_func(
            self.tf_obj_config.network,
            self.tf_obj_config.network_kwargs
        )
        observation_space = env.observation_space

        def make_obs_ph(name):
            return deepq.ObservationInput(observation_space, name=name)

        act, self.train_model, self.update_target, self.debug = baselines.deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.trn_cfg.lr),
            gamma=self.trn_cfg.gamma,
            grad_norm_clipping=10,
            param_noise=self.trn_cfg.param_noise
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
        
        if t > self.trn_cfg.learning_starts and t % self.trn_cfg.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            obses_t, actions, rewards, obses_tp1, dones = buffer.sample_batch(
                'obs', 'acts', 'rewards', 'next_obs', 'next_dones',
                batch_size=self.batch_size
            )
            weights, batch_idxes = np.ones_like(rewards), None
            td_errors = self.train_model(obses_t, actions, rewards, obses_tp1, dones, weights)

        if t > self.trn_cfg.learning_starts and t % self.trn_cfg.target_network_update_freq == 0:
            # Update target network periodically.
            self.update_target()
            
        if itr % log_freq == 0:
            logger.logkv('"% time spent exploring"', int(100 * buffer.policy_info.explore_frac[-1]))
            
        self.t += 1


TfObject.register_cachable_class('QNetwork', QTrainer)
