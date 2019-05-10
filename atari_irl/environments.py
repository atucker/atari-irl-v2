from typing import NamedTuple
import pickle
import numpy as np
import tensorflow as tf

from baselines.bench import Monitor
from baselines.common.vec_env import VecEnvWrapper, VecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, wrap_deepmind
from baselines.common import set_global_seeds
from baselines import logger

from gym.spaces.discrete import Discrete
from gym import spaces
import gym

import ple
import os

from .utils import one_hot

def vec_normalize(env):
    return VecNormalize(env)


mujoco_modifiers = {
    'env_modifiers': [],
    'vec_env_modifiers': [vec_normalize]
}


# from baselines.common.cmd_util.make_atari_env
def wrap_env_with_args(Wrapper, **kwargs):
    return lambda env: Wrapper(env, **kwargs)


def noop_reset(noop_max):
    def _thunk(env):
        assert 'NoFrameskip' in env.spec.id
        return NoopResetEnv(env, noop_max=noop_max)
    return _thunk


def atari_setup(env):
    # from baselines.common.atari_wrappers
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


class TimeLimitEnv(gym.Wrapper):
    def __init__(self, env, time_limit=500):
        gym.Wrapper.__init__(self, env)
        self.steps = 0
        self.time_limit = time_limit

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        f1, f2, done, f3 = self.env.step(actions)
        self.steps += 1
        if self.steps > self.time_limit:
            done = True

        return f1, f2, done, f3


class VecRewardZeroingEnv(VecEnvWrapper):
    def step(self, actions):
        _1, reward, _2, _3 = self.venv.step(actions)
        return _1, np.zeros((_1.shape[0],)), _2, _3

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.venv.step_wait()


class VecIRLRewardEnv(VecEnvWrapper):
    def __init__(self, env, *, reward_network):
        VecEnvWrapper.__init__(self, env)
        self.reward_network = reward_network
        self.prev_obs = None

    def step(self, acts):
        obs, _, done, info = self.venv.step(acts)

        assert np.sum(_) == 0

        if self.prev_obs is None:
            rewards = np.zeros(obs.shape[0])
        else:
            assert not self.reward_network.score_discrim
            rewards = tf.get_default_session(
            ).run(self.reward_network.reward, feed_dict={
                self.reward_network.act_t: acts,
                self.reward_network.obs_t: self.prev_obs
            })[:, 0]

        if self.reward_network.drop_framestack:
            self.prev_obs = obs[:, :, :, -1:]
        else:
            self.prev_obs = obs

        assert len(rewards) == len(obs)
        return obs, rewards, done, info

    def reset(self):
        self.prev_obs = None
        return self.venv.reset()

    def step_wait(self):
        self.venv.step_wait()
        

class EncoderWrappedEnv(VecEnvWrapper):
    def __init__(self, env, *, encoder):
        VecEnvWrapper.__init__(self, env)
        self.encoder = encoder
        self.observation_space = spaces.Box(
            shape=(self.encoder.d_embedding,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max
        )
        print("Wrapping with encoder")
        
    def step(self, acts):
        obs, rewards, done, info = self.venv.step(acts)
        obs = self.encoder.base_vector(obs)
        return obs, rewards, done, info
    
    def reset(self):
        return self.encoder.base_vector(self.venv.reset())
    
    def step_wait(self):
        self.venv.step_wait()


class OneHotDecodingEnv(gym.Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, one_hot_actions):
        return self.env.step(np.argmax(one_hot_actions, axis=0))


class VecOneHotEncodingEnv(VecEnvWrapper):
    def __init__(self, venv, dim=6):
        VecEnvWrapper.__init__(self, venv)
        self.dim = self.action_space.n

    def step(self, actions):
        return self.venv.step(one_hot(actions, self.dim))

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.venv.step_wait()


class DummyVecEnvWrapper(VecEnvWrapper):
    def step(self, actions):
        return self.venv.step(actions)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()


def wrap_with_monitor(env):
    logger_path = logger.get_dir()
    return Monitor(env, logger_path, allow_early_resets=True)


easy_env_modifiers = {
    'env_modifiers': [
        lambda env: wrap_deepmind(env, frame_stack=False, clip_rewards=False),
        wrap_env_with_args(TimeLimitEnv, time_limit=1000000)
    ],
    'vec_env_modifiers': [
        wrap_env_with_args(DummyVecEnvWrapper)
    ]
}

import functools
# Episode Life causes us to not actually reset the environments, which means
# that interleaving, and even running the normal sampler a bunch of times
# will give us super short trajectories. Right now we set it to false, but
# that's not an obviously correct way to handle the problem
atari_modifiers = {
    'env_modifiers': [
        wrap_env_with_args(NoopResetEnv, noop_max=30),
        wrap_env_with_args(MaxAndSkipEnv, skip=4),
        wrap_with_monitor,
        functools.partial(wrap_deepmind, episode_life=False, frame_stack=False),
    ],
    'vec_env_modifiers': [
        wrap_env_with_args(VecFrameStack, nstack=4)
    ]
}


def one_hot_wrap_modifiers(modifiers):
    return {
        'env_modifiers': modifiers['env_modifiers'] + [
            wrap_env_with_args(OneHotDecodingEnv)
        ],
        'vec_env_modifiers': modifiers['vec_env_modifiers']
    }


class ConstantStatistics(object):
    def __init__(self, running_mean):
        self.mean = running_mean.mean
        self.var = running_mean.var
        self.count = running_mean.count

    def update(self, x):
        pass

    def update_from_moments(self, _batch_mean, _batch_var, _batch_count):
        pass


def serialize_env_wrapper(env_wrapper):
    venv = env_wrapper.venv
    env_wrapper.venv = None
    serialized = pickle.dumps(env_wrapper)
    env_wrapper.venv = venv
    return serialized


def restore_serialized_env_wrapper(env_wrapper, venv):
    env_wrapper.venv = venv
    env_wrapper.num_envs = venv.num_envs
    if hasattr(env_wrapper, 'ret'):
        env_wrapper.ret = np.zeros(env_wrapper.num_envs)
    return env_wrapper


def make_const(norm):
    '''Monkey patch classes such as VecNormalize that use a
       RunningMeanStd (or compatible class) to keep track of statistics.'''
    for k, v in norm.__dict__.items():
        if hasattr(v, 'update_from_moments'):
            setattr(norm, k, ConstantStatistics(v))


class JustPress1Environment(gym.Env):
    def __init__(self):
        super().__init__()
        self.reward_range = (0, 1)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3))

        self.black = np.zeros(self.observation_space.shape).astype(np.uint8)
        self.white = np.ones(self.observation_space.shape).astype(np.uint8) * 255
        
        self.random_seed = 0
        self.np_random = np.random.RandomState(0)

        class Ale:
            def lives(self):
                return 1
        self.ale = Ale()

    def seed(self, seed=None):
        if seed is None:
            seed = 0
        self.random_seed = seed
        self.np_random.seed(seed)
        
    def is_done(self):
        return self.np_random.random_sample() > .99

    def step(self, action):
        if action == 0:
            return self.black, 0.0, self.is_done(), {}
        else:
            return self.white, 1.0, self.is_done(), {}

    def reset(self):
        return self.black

    def render(self):
        raise NotImplementedError
        
    def get_action_meanings(self):
        return ['NOOP', 'OP', 'USELESS1', 'USELESS2', 'USELESS3', 'USELESS4']


class SimonSaysEnvironment(JustPress1Environment):
    def __init__(self):
        super().__init__()

        self.correct = np.zeros(self.observation_space.shape).astype(np.uint8)
        self.incorrect = np.zeros(self.observation_space.shape).astype(np.uint8)
        boundary = self.correct.shape[1] // 2
        self.correct[:,:boundary] = 255
        self.incorrect[:,boundary:] = 255
        
        self.next_move = self.np_random.randint(2)
        self.obs_map = {
            0: self.black,
            1: self.white
        }
        self.turns = 0

    @staticmethod
    def isint(n):
        return isinstance(n, np.int64) or isinstance(n, int)

    def set_next_move_get_obs(self):
        assert self.next_move is None
        self.next_move = self.np_random.randint(2)
        return self.obs_map[self.next_move]

    def step(self, action):
        reward = 0.0
        self.turns += 1

        if self.next_move is not None:
            if self.isint(action) and action == self.next_move:
                reward = 2.0
                obs = self.correct
            else:
                obs = self.incorrect
            self.next_move = None
        else:
            obs = self.set_next_move_get_obs()

        return obs, reward, self.turns >= 100, {'next_move': self.next_move}

    def reset(self):
        self.turns = 0
        self.next_move = None
        return self.set_next_move_get_obs()


class VisionSaysEnvironment(SimonSaysEnvironment):
    def __init__(self):
        super().__init__()
        self.zero = np.zeros(self.observation_space.shape).astype(np.uint8)
        self.one = np.zeros(self.observation_space.shape).astype(np.uint8)

        self.one[50:150, 120:128, :] = 255

        self.zero[50:150, 100:108, :] = 255
        self.zero[50:150, 140:148, :] = 255
        self.zero[50:58, 100:148, :] = 255
        self.zero[142:150, 100:148, :] = 255

        self.obs_map = {
            0: self.one,
            1: self.zero
        }


def state_preprocessor(d):
    return np.array([d[key] for key in sorted(d.keys())])


def make_ple_game(game_class, obs_type, **kwargs):
    class PLEGame(gym.Env):
        def __init__(self):
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            super().__init__()
            self.ple = ple.PLE(
                game_class(**kwargs),
                state_preprocessor=state_preprocessor,
                display_screen=False
            )

            self.ple.init()

            self.reward_range = (
                min(self.ple.game.rewards.values()),
                max(self.ple.game.rewards.values())
            )

            self.obs_type = obs_type
            if self.obs_type == 'rgb':
                self.get_obs = self.ple.getScreenRGB
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(*self.ple.getScreenDims(), 3)
                )
            elif self.obs_type == 'state_vector':
                self.get_obs = self.ple.getGameState
                self.observation_space = gym.spaces.Box(
                    low=-1000, high=1000, shape=self.get_obs().shape, dtype=np.float64
                )
            else:
                assert False, "obs_type must be rgb or state_vector"

            self.action_space = gym.spaces.Discrete(6)
            assert len(self.ple.getActionSet()) < 6
            self._actions = self.ple.getActionSet()
            self._actions += [
                None for _ in range(6 - len(self._actions))
            ]
            self._action_mapping = self.ple.game.actions
            self._action_mapping['NOOP'] = None

            self.ale = self.ple
            self.np_random = np.random.RandomState(0)

        def seed(self, seed=None):
            self.ple.rng.seed(seed)
            self.np_random.seed(seed)

        def is_done(self):
            return self.ple.game_over()

        def step(self, action):
            reward = self.ple.act( self._actions[action])

            return self.get_obs(), reward, self.is_done(), self.ple.game.getGameState()

        def reset(self):
            self.ple.reset_game()
            return self.get_obs()

        def render(self, *args):
            return self.ple.getScreenRGB()

        def get_action_meanings(self):
            reverse_dict = dict(zip(
                self._action_mapping.values(),
                self._action_mapping.keys()
            ))
            ans = [reverse_dict[a] for a in self._actions]
            ans[0] = 'NOOP'
            return ans

    return PLEGame

gym.envs.register(
    id='VisionSays-v0',
    entry_point='atari_irl.environments:VisionSaysEnvironment'
)

gym.envs.register(
    id='SimonSays-v0',
    entry_point='atari_irl.environments:SimonSaysEnvironment'
)

no_modifiers = {
    'env_modifiers': [],
    'vec_env_modifiers': []
}

PLEPong = make_ple_game(ple.games.pong.Pong, 'rgb')
PLEPongState = make_ple_game(ple.games.pong.Pong, 'state_vector')
PLECatcher = make_ple_game(ple.games.catcher.Catcher, 'rgb', init_lives=10000)
PLECatcherState = make_ple_game(ple.games.catcher.Catcher, 'state_vector', init_lives=10000)

gym.envs.register(
    id='PLEPong-v0',
    max_episode_steps=100000,
    entry_point='atari_irl.environments:PLEPong'
)

gym.envs.register(
    id='PLEPongState-v0',
    max_episode_steps=100000,
    entry_point='atari_irl.environments:PLEPongState'
)

gym.envs.register(
    id='PLECatcher-v0',
    max_episode_steps=1000,
    entry_point='atari_irl.environments:PLECatcher'
)

gym.envs.register(
    id='PLECatcherState-v0',
    max_episode_steps=1000,
    entry_point='atari_irl.environments:PLECatcherState'
)


env_mapping = {
    'PongNoFrameskip-v4': atari_modifiers,
    'EnduroNoFrameskip-v4': atari_modifiers,
    'BreakoutNoFrameskip-v4': atari_modifiers,
    'CartPole-v1': mujoco_modifiers,
    'VisionSays-v0': easy_env_modifiers,
    'SimonSays-v0': easy_env_modifiers,
    'PLEPong-v0': atari_modifiers,
    'PLEPongState-v0': no_modifiers,
    'PLECatcher-v0': atari_modifiers,
    'PLECatcherState-v0': no_modifiers,
}


def make_vec_env(*, env_name: str, seed=0, one_hot_code=False, num_envs=8) -> VecEnv:
    set_global_seeds(seed)

    env_modifiers = env_mapping[env_name]['env_modifiers']
    if one_hot_code:
        env_modifiers = one_hot_wrap_modifiers(env_modifiers)

    def make_env(i):
        def _thunk():
            env = gym.make(env_name)
            #env = Monitor(env, logger_path, allow_early_resets=True)
            env.seed(seed + i)
            for fn in env_modifiers:
                env = fn(env)
            return env

        return _thunk

    base_vec_env = SubprocVecEnv([make_env(i + seed) for i in range(num_envs)])

    vec_env_modifiers = env_mapping[env_name]['vec_env_modifiers']
    env = base_vec_env
    for fn in vec_env_modifiers:
        env = fn(env)

    return env
