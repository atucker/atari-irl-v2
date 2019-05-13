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


def leaky_relu(name, inpt):
    return tf.nn.leaky_relu(inpt, name=name)


def leaky_relu_batch_norm(name, inpt):
    return tf.nn.leaky_relu(batch_norm(inpt, name))


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
    activ = leaky_relu
    h = activ('l1', conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
    h2 = activ('l2', conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ('l3', conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    return conv_to_fc(h3)


def last_linear_hidden_layer(x, actions=None, d=512, **conv_kwargs):
    h = dcgan_cnn(x, **conv_kwargs)
    activ = leaky_relu

    if actions is not None:
        h = tf.concat([actions, h], axis=1)

    return activ('h_final', fc(h, 'fc1', nh=d, init_scale=np.sqrt(2)))


def cnn_net(x, actions=None, dout=1, **conv_kwargs):
    h_final = last_linear_hidden_layer(x=x, actions=actions, **conv_kwargs)
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


class ScoreConfiguration(Configuration):
    default_values = dict(
        score_type='discrim',
        transfer_function='positive',
        mean_type=None,
        rescale_type=None
    )


class DiscriminatorConfiguration(Configuration):
    default_values = dict(
        name='discriminator',
        reward_arch=ArchConfiguration(),
        value_arch=ArchConfiguration(),
        fuse_archs=True,
        information_bottleneck_bits=.05,
        kl_constraint_alpha=1e-4,
        gradient_penalty=None,
        reward_change_penalty=None,
        discount=0.99,
        state_only=False,
        max_itrs=100,
        drop_framestack=False,
        seed=0,
        score_config=ScoreConfiguration()
    )


class ScoreProcessor:
    def __init__(self, config: ScoreConfiguration, discriminator: 'AtariAIRL') -> None:
        self.config = config
        self.random_buffer = None
        self.discriminator = discriminator

    @property
    def score_tensor(self) -> tf.Tensor:
        return {
            'reward': self.discriminator.reward,
            'discrim': self.discriminator.discrim_output,
            'q': self.discriminator.qfn,
            'value': self.discriminator.value_fn
        }[self.config.score_type]

    def initialize(self, random_buffer: Buffer) -> None:
        self.random_buffer = random_buffer

    def _process_scores(self, scores) -> np.ndarray:
        if self.config.score_type == 'discrim':
            scores = np.clip(scores, 1e-7, 1 - 1e-7)

        scores = {
            'positive': lambda s: -np.log(1 - s),
            'negative': lambda s: np.log(s),
            'both': lambda s: np.log(s) - np.log(1 - s),
            'identity': lambda s: s
        }[self.config.transfer_function](scores)

        return scores

    def train_itr(self, *, expert_scores=None, policy_scores=None) -> None:
        pass

    def finalize_train_step(self, logger=None) -> None:
        if logger:
            pass

    def eval(
            self, *,
            obs: np.ndarray,
            acts: np.ndarray,
            next_obs: Optional[np.ndarray]=None,
            lprobs: Optional[np.ndarray]=None
    ) -> np.ndarray:
        modify_obs = self.discriminator.modify_obs
        scores = self._process_scores(
            tf.get_default_session().run(
                self.score_tensor,
                feed_dict={
                    self.discriminator.act_t: acts,
                    self.discriminator.obs_t: modify_obs(obs),
                    self.discriminator.nobs_t: modify_obs(next_obs),
                    self.discriminator.lprobs: lprobs,
                    self.discriminator.train_time: False
                }
            )
        )
        return scores


class ItrData(NamedTuple):
    loss: float
    accuracy: float
    score: float
    mean_kl: float


class AtariAIRL(TfObject):
    class_registration_name = 'IRLDiscriminator'

    def __init__(
            self, *,
            env,
            expert_buffer,
            random_buffer=None,
            config
    ):
        self.expert_buffer = expert_buffer
        self.random_buffer = random_buffer
        self.config = config
        self.score_manager = ScoreProcessor(
            config=config.score_config,
            discriminator=self
        )

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
        self.beta = None
        self.beta_value = None
        self.next_beta_value = None
        self.train_time = None
        self.z = None
        self.z_dist = None
        self.z_dist_next = None
        self.OLDLOGPTAU = None
        self.mean_kl = None
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

        TfObject.__init__(self, config, scope_name=self.config.name)

    def _build_z(self, *, obs_t, nobs_t, train_time):
        if self.config.information_bottleneck_bits is None:
            with tf.variable_scope('z'):
                h_final_next = last_linear_hidden_layer(nobs_t)
            with tf.variable_scope('z', reuse=True):
                h_final = last_linear_hidden_layer(obs_t)
        else:
            with tf.variable_scope('z'):
                with tf.variable_scope('mu'):
                    mu_next = last_linear_hidden_layer(nobs_t)
                with tf.variable_scope('sigma'):
                    sigmasq_next = last_linear_hidden_layer(nobs_t)

            with tf.variable_scope('z', reuse=True):
                with tf.variable_scope('mu', reuse=True):
                    mu = last_linear_hidden_layer(obs_t)
                with tf.variable_scope('sigma', reuse=True):
                    sigmasq = last_linear_hidden_layer(obs_t)

            self.z_dist = tf.distributions.Normal(loc=mu, scale=sigmasq)
            self.z_dist_next = tf.distributions.Normal(loc=mu_next, scale=sigmasq_next)
            h_final = tf.cond(
                train_time,
                lambda: self.z_dist.sample(),
                lambda: mu
            )
            h_final_next = tf.cond(
                train_time,
                lambda: self.z_dist_next.sample(),
                lambda: mu_next
            )

        return h_final, h_final_next

    def _setup_reward_and_shaping_functions(self):
        if not self.config.fuse_archs:
            with tf.variable_scope('reward'):
                if self.state_only:
                    reward = self.config.reward_arch.create(
                        self.obs_t, dout=1
                    )
                else:
                    reward = self.config.reward_arch.create(
                        self.obs_t, actions=self.act_t, dout=1
                    )

            # value function shaping
            with tf.variable_scope('vfn'):
                fitted_value_fn_next = self.config.value_arch.create(
                    self.nobs_t,
                    dout=1
                )
            with tf.variable_scope('vfn', reuse=True):
                fitted_value_fn = self.config.value_arch.create(
                    self.obs_t,
                    dout=1
                )
        else:
            assert self.state_only
            z, z_next = self._build_z(
                obs_t=self.obs_t,
                nobs_t=self.nobs_t,
                train_time=self.train_time
            )
            self.z = z

            with tf.variable_scope('vfn'):
                fitted_value_fn_next = fc(
                    z_next, 'output', nh=1, init_scale=np.sqrt(2)
                )
            with tf.variable_scope('vfn', reuse=True):
                fitted_value_fn = fc(
                    z, 'output', nh=1, init_scale=np.sqrt(2)
                )
            with tf.variable_scope('reward'):
                reward = fc(
                    z, 'output', nh=1, init_scale=np.sqrt(2)
                )

        return reward, fitted_value_fn, fitted_value_fn_next

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
            self.OLDLOGPTAU = tf.placeholder(tf.float32, [None, 1], name='OLDLOGPTAU')
            self.lr = tf.placeholder(tf.float32, (), name='lr')
            self.train_time = tf.placeholder(tf.bool, (), name='train_time')

            with tf.variable_scope('discrim') as dvs:
                (
                    self.reward, self.value_fn, fitted_value_fn_next
                ) = self._setup_reward_and_shaping_functions()
                fitted_value_fn = self.value_fn

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma * fitted_value_fn_next
                self.log_p_tau = self.reward + self.gamma * fitted_value_fn_next - fitted_value_fn

            log_q_tau = self.lprobs
            log_pq = tf.reduce_logsumexp([self.log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(self.log_p_tau - log_pq)
            self.accuracy, self.update_accuracy = tf.metrics.accuracy(
                labels=self.labels,
                predictions=self.discrim_output > 0.5
            )
            expert_loss = self.labels * (self.log_p_tau - log_pq)
            policy_loss = (1 - self.labels) * (log_q_tau - log_pq)
            classification_loss = -tf.reduce_mean(expert_loss + policy_loss)

            self.loss = classification_loss
            if self.config.information_bottleneck_bits is not None:
                with tf.variable_scope('bottleneck') as _bn:
                    self.beta = tf.placeholder(tf.float32, (), name='beta')
                    self.beta_value = 1
                    self.mean_kl = tf.reduce_mean(
                        self.z_dist.kl_divergence(
                            tf.distributions.Normal(loc=0.0, scale=1.0)
                        )
                    )
                    Ic = self.config.information_bottleneck_bits
                    info_loss = self.beta * (self.mean_kl - Ic)
                    self.next_beta_value = tf.math.maximum(
                        0.0,
                        self.beta + self.config.kl_constraint_alpha * (self.mean_kl - Ic)
                    )
                    self.loss = self.loss + info_loss

            if self.config.gradient_penalty is not None:
                # For some reason I can't compute this out to obs_t, so I don't
                # really think that this works...
                policy_loss_grads = tf.gradients(
                    -tf.reduce_mean(policy_loss), self.z
                )
                grad_loss = tf.reduce_mean(policy_loss_grads)
                self.loss = self.loss + self.config.gradient_penalty * grad_loss

            if self.config.reward_change_penalty is not None:
                approxkl = tf.reduce_mean((self.log_p_tau - self.OLDLOGPTAU) ** 2)
                self.loss = self.loss + self.config.reward_change_penalty * approxkl

            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.grad_reward = tf.gradients(self.reward, [self.obs_t, self.act_t])

            U.initialize()
            tf.get_default_session().run(tf.local_variables_initializer())

    def train_step(self, buffer: Buffer, policy: Policy, batch_size=256, lr=1e-3, verbose=False, itr=0, **kwargs):
        if batch_size > buffer.time_shape.size:
            return

        stacker = Stacker(ItrData)
        
        # Train discriminator
        for it in range(self.max_itrs):
            if it % 20 == 0:
                print(f"Discriminator training itr {it}...")
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
                self.lr: lr,
                self.train_time: True
            }
            if self.config.reward_change_penalty is not None:
                feed_dict[self.OLDLOGPTAU] = tf.get_default_session().run(
                    self.log_p_tau,
                    feed_dict=feed_dict
                )

            # not an elif
            if self.config.information_bottleneck_bits is None:
                loss, _, acc, scores = tf.get_default_session().run(
                    [
                        self.loss,
                        self.step,
                        self.update_accuracy,
                        self.score_manager.score_tensor
                    ],
                    feed_dict=feed_dict
                )
                mean_kl = 0.0
            else:
                feed_dict[self.beta] = self.beta_value
                loss, _, acc, scores, mean_kl, next_beta = tf.get_default_session().run(
                    [
                        self.loss,
                        self.step,
                        self.update_accuracy,
                        self.score_manager.score_tensor,
                        self.mean_kl,
                        self.next_beta_value
                    ],
                    feed_dict=feed_dict
                )
                self.beta_value = next_beta

            policy_slice = slice(0, batch_size)
            expert_slice = slice(batch_size, -1)
            self.score_manager.train_itr(
                policy_scores=scores[policy_slice],
                expert_scores=scores[expert_slice]
            )

            stacker.append(ItrData(
                loss=loss,
                accuracy=acc,
                score=np.mean(scores[policy_slice]), # type: ignore
                mean_kl=mean_kl
            ))

        mean_loss = np.mean(stacker.loss)
        mean_acc = np.mean(stacker.accuracy)
        mean_score = np.mean(stacker.score)

        if logger:
            logger.logkv('GCLDiscrimLoss', mean_loss)
            logger.logkv('GCLDiscrimAccuracy', mean_acc)
            logger.logkv('GCLMeanScore', mean_score)
            if self.config.information_bottleneck_bits is not None:
                logger.logkv('BottleneckKL', np.mean(stacker.mean_kl))
                logger.logkv('Ic', self.config.information_bottleneck_bits)
                logger.logkv('BottleneckBeta', self.beta_value)

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
        if len(lprobs.shape) == 1:
            lprobs = lprobs.reshape((lprobs.shape[0], 1))

        score = self.score_manager.eval(
            obs=obs,
            acts=acts,
            next_obs=next_obs,
            lprobs=lprobs
        )

        if np.isnan(np.mean(score)):
            import pdb; pdb.set_trace()

        return score
