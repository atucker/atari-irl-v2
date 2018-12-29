from typing import NamedTuple, Tuple
import numpy as np
import gym


class Observations(NamedTuple):
    shape: Tuple[int]
    obs: np.ndarray


class Batch(NamedTuple):
    shape: Tuple[int]
    obs: np.ndarray
    acts: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


class Buffer(NamedTuple):
    shape: Tuple[int]
    obs: np.ndarray
    acts: np.ndarray
    rewards: np.ndarray


class Policy:
    def __init__(self, obs_space: Tuple[int], act_space: Tuple[int]) -> None:
        raise NotImplemented

    def get_actions(self, obs_batch: Observations) -> Batch:
        raise NotImplemented

    def train(self, buffer: Buffer) -> None:
        raise NotImplemented


class RewardModel:
    def __init__(self, obs_space: Tuple[int], act_space: Tuple[int]) -> None:
        raise NotImplemented

    def get_rewards(self, batch: Batch) -> np.ndarray:
        raise NotImplemented

    def train(self, buffer: Buffer) -> None:
        raise NotImplemented


class Sampler:
    def __init__(self, env: gym.Env, policy: Policy, rollout_t: int) -> None:
        self.env = env
        self.policy = policy
        self.rollout_t = rollout_t

    def sample_batch(self, rollout_t: int) -> Batch:
        pass

    def reset(self) -> None:
        self.env.reset()


class DummyBuffer:
    def __init__(self):
        self.batch = None

    def add_batch(self, samples):
        self.batch = samples


class IRL:
    def __init__(self, args):
        pass

    def obtain_samples(self):
        pass

    def update_buffer(self):
        pass

    def update_policy(self):
        pass

    def update_discriminator(self):
        pass

    def train(self):
        samples = self.obtain_samples()
        buffer  = self.update_buffer(samples)
        buffer  = self.update_discriminator(buffer)
        self.update_policy(buffer)

