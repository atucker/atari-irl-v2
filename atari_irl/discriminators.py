import tensorflow as tf
import numpy as np
from baselines.a2c.utils import conv, fc, conv_to_fc
import baselines.common.tf_util as U
from .experiments import TfObject, Configuration
from .headers import Stacker, Buffer, EnvInfo, Batch
from .utils import one_hot, set_seed
from .policies import Policy
from typing import NamedTuple, Optional
from baselines import logger
import functools


def batch_norm(x, name):
    shape = (1, *x.shape[1:])
    with tf.variable_scope(name):
        mean = tf.get_variable('mean', shape, initializer=tf.constant_initializer(0.0))
        variance = tf.get_variable('variance', shape, initializer=tf.constant_initializer(1.0))
        offset = tf.get_variable('offset', shape, initializer=tf.constant_initializer(0.0))
        scale = tf.get_variable('scale', shape, initializer=tf.constant_initializer(1.0))
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, 0.001, name)


def dcgan_cnn(unscaled_images, **conv_kwargs):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = lambda name, inpt: tf.nn.leaky_relu(batch_norm(inpt, name))
    h = activ('l1', conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
    h2 = activ('l2', conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ('l3', conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    return conv_to_fc(h3)


def cnn_net(x, actions=None, dout=1, **conv_kwargs):
    h = dcgan_cnn(x, **conv_kwargs)
    activ = lambda name, inpt: tf.nn.leaky_relu(batch_norm(inpt, name))

    if actions is not None:
        assert dout == 1
        h = tf.concat([actions, h], axis=1)

    h_final = activ('h_final', fc(h, 'fc1', nh=512, init_scale=np.sqrt(2)))
    return fc(h_final, 'output', nh=dout, init_scale=np.sqrt(2))


class ArchConfiguration(Configuration):
    default_values = dict(
        network='cnn_net',
        arch_args={}
    )

    def create(self, inpt, **kwargs):
        create_fn = {'cnn_net': cnn_net}[self.network]
        return create_fn(inpt, **kwargs, **self.arch_args)

    @property
    def obs_dtype(self):
        if self.network == 'cnn_net':
            return tf.int8
        else:
            return tf.float32


class DiscriminatorConfiguration(Configuration):
    default_values = dict(
        name='discriminator',
        reward_arch=ArchConfiguration(),
        value_arch=ArchConfiguration(),
        score_discrim=False,
        discount=0.99,
        state_only=False,
        max_itrs=100,
        drop_framestack=False,
        seed=0
    )


class ItrData(NamedTuple):
    loss: float
    accuracy: float
    score: float


class AtariAIRL(TfObject):
    def __init__(
         self, *,
         env,
         expert_buffer,
         config
    ):
        self.expert_buffer = expert_buffer
        self.config = config

        self.action_space = env.action_space
        self.dO = functools.reduce(
            lambda a, b: a * b,
            env.observation_space.shape,
            1
        )
        self.dOshape = env.observation_space.shape
        if self.config.drop_framestack:
            assert len(self.dOshape) == 3
            self.dOshape = (*self.dOshape[:-1], 1)
        self.dU = env.action_space.n

        self.score_discrim = self.config.score_discrim
        self.state_only = self.config.state_only
        self.drop_framestack = self.config.drop_framestack
        self.modify_obs = lambda obs: obs
        self.gamma = self.config.discount
        self.max_itrs = self.config.max_itrs

        self.obs_t = None
        self.nobs_t = None
        self.act_t = None
        self.nact_t = None
        self.labels = None
        self.lprobs = None
        self.lr = None
        self.reward = None
        self.value_fn = None
        self.qfn = None
        self.discrim_output = None
        self.accuracy = None
        self.update_accuracy = None
        self.loss = None
        self.step = None
        self.grad_reward = None
        self.score_mean = 0
        self.score_std = 1

        TfObject.__init__(self, config)

    def initialize_graph(self):
        set_seed(self.config.seed)
        with tf.variable_scope(self.config.name) as _vs:
            # Should be batch_size x T x dO/dU
            obs_dtype = self.config.reward_arch.obs_dtype
            self.obs_t = tf.placeholder(obs_dtype, list((None,) + self.dOshape), name='obs')
            self.nobs_t = tf.placeholder(obs_dtype, list((None,) + self.dOshape), name='nobs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.nact_t = tf.placeholder(tf.float32, [None, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            with tf.variable_scope('discrim') as dvs:
                rew_input = self.obs_t
                with tf.variable_scope('reward'):
                    if self.state_only:
                        self.reward = self.config.reward_arch.create(
                            rew_input,
                            dout=1
                        )
                    else:
                        self.reward = self.config.reward_arch.create(
                            rew_input,
                            actions=self.act_t,
                            dout=1
                        )
                        
                # value function shaping
                with tf.variable_scope('vfn'):
                    fitted_value_fn_next = self.config.value_arch.create(
                        self.nobs_t,
                        dout=1
                    )
                with tf.variable_scope('vfn', reuse=True):
                    self.value_fn = fitted_value_fn = self.config.value_arch.create(
                        self.obs_t,
                        dout=1
                    )

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma * fitted_value_fn_next
                log_p_tau = self.reward + self.gamma * fitted_value_fn_next - fitted_value_fn

            log_q_tau = self.lprobs

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau - log_pq)
            self.accuracy, self.update_accuracy = tf.metrics.accuracy(
                labels=self.labels,
                predictions=self.discrim_output > 0.5
            )
            self.loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            self.grad_reward = tf.gradients(self.reward, [self.obs_t, self.act_t])
            
            U.initialize()
            tf.get_default_session().run(tf.local_variables_initializer())

    def _process_discrim_output(self, score):
        score = np.clip(score, 1e-7, 1 - 1e-7)

        score = np.log(score) - np.log(1 - score)
        score = score[:, 0]
        return score, score
        #return np.clip((score - self.score_mean) / self.score_std, -3, 3), score

    def train_step(self, buffer: Buffer, policy: Policy, batch_size=256, lr=1e-3, verbose=False, itr=0, **kwargs):
        if batch_size > buffer.time_shape.size:
            return
        
        raw_discrim_scores = []
        stacker = Stacker(ItrData)
        
        # Train discriminator
        for it in range(self.max_itrs):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                buffer.sample_batch(
                    'next_obs',
                    'obs',
                    'next_acts',
                    'acts',
                    'lprobs',
                    batch_size=batch_size,
                    modify_obs=self.modify_obs,
                    one_hot_acts_to_dim=self.dU
                )

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch = \
                self.expert_buffer.sample_batch(
                    'next_obs',
                    'obs',
                    'next_acts',
                    'acts',
                    batch_size=batch_size,
                    modify_obs=self.modify_obs
                )
            

            # TODO(Aaron): put this graph directly into the reward network
            expert_lprobs_batch = policy.get_a_logprobs(
                obs=expert_obs_batch,
                acts=expert_act_batch
            )

            # Build feed dict
            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            nact_batch = np.concatenate([nact_batch, nexpert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(
                np.concatenate([
                    lprobs_batch,
                    expert_lprobs_batch
                ],
                axis=0),
            axis=1).astype(np.float32)

            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.nact_t: nact_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr
            }

            loss, _, acc, scores = tf.get_default_session().run(
                [self.loss, self.step, self.update_accuracy, self.discrim_output],
                feed_dict=feed_dict
            )
            # we only want the average score for the non-expert demos
            non_expert_slice = slice(0, batch_size)
            score, raw_score = self._process_discrim_output(scores[non_expert_slice])
            assert len(score) == batch_size
            assert np.sum(labels[non_expert_slice]) == 0
            raw_discrim_scores.append(raw_score)

            stacker.append(ItrData(
                loss=loss,
                accuracy=acc,
                score=np.mean(score)
            ))


        mean_loss = np.mean(stacker.loss)
        mean_acc = np.mean(stacker.accuracy)
        mean_score = np.mean(score)
        if verbose and (it % int(self.max_itrs / 10) == 0 or it == self.max_itrs - 1):
            print(f'\t{it}/{self.max_itrs}')
            print('\tLoss:%f' % mean_loss)
            print('\tAccuracy:%f' % mean_acc)


        if logger:
            logger.logkv('GCLDiscrimLoss', mean_loss)
            logger.logkv('GCLDiscrimAccuracy', mean_acc)
            logger.logkv('GCLMeanScore', mean_score)

        # set the center for our normal distribution
        scores = np.hstack(raw_discrim_scores)
        self.score_std = np.std(scores)
        self.score_mean = np.mean(scores)

    def eval(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        next_obs: Optional[np.ndarray]=None,
        lprobs: Optional[np.ndarray]=None,
        **kwargs
    ) -> np.ndarray:
        if isinstance(acts, list):
            acts = np.array(acts)
        if len(acts.shape) == 1:
            acts = one_hot(acts, self.dU)
        if self.score_discrim:
            obs, obs_next, acts, path_probs = (
                self.modify_obs(obs),
                self.modify_obs(next_obs),
                acts,
                lprobs
            )
            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(
                self.discrim_output,
                feed_dict={
                    self.act_t: acts,
                    self.obs_t: obs,
                    self.nobs_t: obs_next,
                    self.lprobs: path_probs
                }
            )
            score, _ = self._process_discrim_output(scores)

        else:
            reward = tf.get_default_session().run(
                self.reward, feed_dict={
                    self.act_t: acts,
                    self.obs_t: self.modify_obs(obs)
                }
            )
            score = reward[:, 0]

        if np.isnan(np.mean(score)):
            import pdb; pdb.set_trace()

        return score
