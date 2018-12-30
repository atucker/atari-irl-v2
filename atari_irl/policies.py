from typing import NamedTuple
import gym
import baselines
import numpy as np
import time
import os
import os.path as osp

from baselines import logger
from baselines.common.vec_env import VecEnv
import baselines.ppo2.model
import baselines.common.policies
from baselines.ppo2.ppo2 import safemean, explained_variance

from headers import PolicyTrainer, PolicyInfo, Observations, Buffer, TimeShape


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


class PPO2Trainer(PolicyTrainer):
    def __init__(
            self,
            env: VecEnv,
            network: str,
            **network_kwargs
    ) -> None:
        super().__init__(env)
        nenvs = env.num_envs
        total_timesteps = 10e6

        self.lr = 2.5e-4
        self.cliprange = 0.1
        self.nsteps = 128
        self.nminibatches = 4
        self.noptepochs = 4
        self.nbatch = nenvs * self.nsteps
        self.nbatch_train = self.nbatch // self.nminibatches
        self.nupdates = total_timesteps // self.nbatch

        self.log_interval = 10
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

    def train(self, buffer: Buffer[PPO2Info], itr: int) -> None:
        tstart = time.time()
        frac = 1.0 - (itr - 1.0) / self.nupdates

        # Calculate the learning rate
        lrnow = self.lr * frac
        # Calculate the cliprange
        cliprangenow = self.cliprange * frac

        # Index of each element of batch_size
        # Create the indices array
        inds = np.arange(self.nbatch)
        mblossvals = []

        obs = buffer.obs
        returns = buffer.rewards
        masks = buffer.dones
        actions = buffer.actions
        values = buffer.policy_info.values
        neglogpacs = buffer.policy_info.neglogpacs
        epinfobuf = buffer.env_info.epinfobuf

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

        if itr % self.log_interval == 0 or itr == 1:
            # Calculates if value function is a good predictor of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", itr * self.nsteps)
            logger.logkv("nupdates", itr)
            logger.logkv("total_timesteps", itr * self.nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))

            logger.logkv('time_elapsed', tnow - self.tfirststart)
            for (lossval, lossname) in zip(lossvals, self.model.loss_names):
                logger.logkv(lossname, lossval)

        if self.save_interval and (itr % self.save_interval == 0 or itr == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % itr)
            print('Saving to', savepath)
            self.model.save(savepath)
