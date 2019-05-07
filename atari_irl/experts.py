from atari_irl import headers
from atari_irl.headers import Buffer, PolicyInfo, EnvInfo, TimeShape
from atari_irl.utils import Stacker
from typing import NamedTuple, Tuple, Callable
import numpy as np


class Trajectory(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    dones: np.ndarray

        
class ExpertLookupPolicyInfo(PolicyInfo):
    _fields = ('time_shape', 'actions', 'next_actions')
    def __init__(
        self, *,
        time_shape: TimeShape,
        actions: np.ndarray,
        next_actions: np.ndarray
    ) -> None:
        super().__init__(time_shape=time_shape, actions=actions)
        self.next_actions = next_actions

        
class ExpertBuffer(Buffer[PolicyInfo]):
    def __init__(
        self, *,
        obs,
        next_obs,
        acts,
        next_acts,
        dones,
        next_dones
    ):
        T = obs.shape[0]
        assert next_obs.shape[0] == T
        assert acts.shape[0] == T
        assert next_acts.shape[0] == T
        assert dones.shape[0] == T
        assert next_dones.shape[0] == T
        time_shape = TimeShape(T=T, num_envs=None)
        
        super().__init__(
            overwrite_rewards=False,
            overwrite_logprobs=False,
            discriminator=None,
            policy=None,
            time_shape=time_shape,
            env_info=EnvInfo(
                obs=obs,
                next_obs=obs,
                time_shape=time_shape,
                dones=dones,
                next_dones=next_dones,
                rewards=None,
                epinfobuf=[]
            ),
            policy_info=ExpertLookupPolicyInfo(
                time_shape=time_shape,
                actions=acts,
                next_actions=next_acts
            ),
            sampler_state=None
        )
        
    @staticmethod
    def from_trajectories(trajectories):
        stacker = Stacker(Trajectory)
        
        for traj in trajectories:     
            T = len(traj['observations'])
            dones = np.zeros((T, 1)).astype(np.bool)
            dones[-1, 0] = True
            stacker.append(Trajectory(
                observations = traj['observations'],
                actions = traj['actions'],
                dones = dones.reshape((T, 1)).copy()
            ))
            
        obs = np.vstack(stacker.observations)
        acts = np.vstack(stacker.actions)
        dones = np.vstack([d.reshape((len(d), 1)) for d in stacker.dones])
        
        next_obs = np.vstack((obs[1:], np.zeros((1,) + obs[0].shape)))
        next_acts = np.vstack((acts[1:], np.zeros(((1,) + acts[0].shape))))
        next_dones = np.vstack((dones[1:], np.array([[False]])))
                               
        next_obs[dones[:, 0]] *= 0
        next_acts[dones[:, 0]] *= 0

        check = ((obs[1:] - next_obs[:-1]) ** 2).sum(axis=(1, 2, 3))
        for t in range(len(check)):
            if dones[t]:
                assert check[t] != 0
            else:
                assert check[t] == 0
                
        return ExpertBuffer(
            obs=obs, next_obs=next_obs,
            acts=acts, next_acts=next_acts,
            dones=dones, next_dones=next_dones
        )

    @property
    def next_acts(self):
        return self.policy_info.next_actions
