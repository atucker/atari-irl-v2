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
    scaled_images = unscaled_images / 255.0
    activ = leaky_relu
    h = activ('l1', conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
    h2 = activ('l2', conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ('l3', conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    ans = conv_to_fc(h3)
    return ans


def last_linear_hidden_layer(x, config, actions=None, d=512, **conv_kwargs):
    h = config.create(x, **conv_kwargs)

    activ = leaky_relu

    if actions is not None:
        h = tf.concat([actions, h], axis=1)

    return activ('h_final', fc(h, 'fc1', nh=d, init_scale=np.sqrt(2)))


class ArchConfiguration(Configuration):
    default_values = dict(
        network='dcgan_cnn',
        arch_args={}
    )

    def create(self, inpt, **kwargs):
        create_fn = {
            'dcgan_cnn': dcgan_cnn,
            'mlp': lambda x: x
        }[self.network]
        return create_fn(inpt, **kwargs, **self.arch_args)

    @property
    def obs_dtype(self):
        return tf.float32


class ScoreConfiguration(Configuration):
    default_values = dict(
        score_type='discrim',
        transfer_function='positive',
        mean_type=None,
        rescale_type=None
    )

    @property
    def prepare_random_batch(self):
        return self.mean_type is not None and 'random' in self.mean_type


class ScoreProcessor:
    def __init__(self, config: ScoreConfiguration, discriminator: 'AtariAIRL') -> None:
        self.config = config
        self.random_buffer = None
        self.discriminator = discriminator
        self.random_mean = None
        self.expert_mean = None
        self.policy_mean = None
        self.policy_std = None

        self.expert_mean_buffer = []
        self.policy_score_buffer = []

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
        if np.isnan(np.mean(scores)):
            scores = np.nan_to_num(scores, copy=False)
            logger.info("Warning, encountered NaN and replaced with 0")

        if self.config.score_type == 'discrim':
            scores = np.clip(scores, 1e-7, 1 - 1e-7)

        scores = {
            'positive': lambda s: -np.log(1 - s),
            'negative': lambda s: np.log(s),
            'both': lambda s: np.log(s) - np.log(1 - s),
            'identity': lambda s: s
        }[self.config.transfer_function](scores)

        return scores

    def train_itr(self, *, expert_scores, policy_scores) -> None:
        self.expert_mean_buffer.append(
            np.mean(self._process_scores(expert_scores))
        )
        self.policy_score_buffer.extend(
            self._process_scores(policy_scores)
        )

    def finalize_train_step(self, logger=None) -> None:
        if self.config.prepare_random_batch:
            batch_size = 1024
            nobs_batch, obs_batch, act_batch = \
                self.discriminator.random_buffer.sample_batch(
                    'next_obs',
                    'obs',
                    'acts',
                    batch_size=batch_size,
                    modify_obs=self.discriminator.modify_obs,
                    one_hot_acts_to_dim=self.discriminator.dU
                )

            lprobs_batch = self.discriminator.random_buffer.policy.get_a_logprobs(
                obs=obs_batch,
                acts=act_batch
            ).reshape((batch_size, 1))

            random_scores = self.eval(
                obs=obs_batch,
                next_obs=nobs_batch,
                acts=act_batch,
                lprobs=lprobs_batch,
                rescale=False
            )
            self.random_mean = np.mean(random_scores)

        self.expert_mean = np.mean(self.expert_mean_buffer)
        self.policy_mean = np.mean(self.policy_score_buffer)
        self.policy_std = np.std(self.policy_score_buffer)

        self.expert_mean_buffer = []
        self.policy_score_buffer = []
        if logger:
            logger.logkv("Random Mean", self.random_mean)
            logger.logkv("Expert Mean", self.expert_mean)
            logger.logkv("Policy Mean", self.policy_mean)
            logger.logkv("Policy std", self.policy_std)

    def eval(
            self, *,
            obs: np.ndarray,
            acts: np.ndarray,
            next_obs: Optional[np.ndarray]=None,
            lprobs: Optional[np.ndarray]=None,
            rescale=True
    ) -> np.ndarray:
        modify_obs = self.discriminator.modify_obs
        feed_dict = {
            self.discriminator.act_t: acts,
            self.discriminator.obs_t: modify_obs(obs),
            self.discriminator.train_time: False
        }
        if self.config.score_type != 'reward':
            feed_dict[self.discriminator.nobs_t] = modify_obs(next_obs)
            feed_dict[self.discriminator.lprobs] = lprobs

        scores = self._process_scores(
            tf.get_default_session().run(
                self.score_tensor,
                feed_dict=feed_dict
            )
        )
        if rescale and self.config.mean_type is not None:
            mean = min(self.random_mean, self.policy_mean)
            scores = scores - mean

        return scores


class InfoBottleneckConfig(Configuration):
    default_values = dict(
        enabled=True,
        information_bottleneck_nats=1,
        kl_constraint_alpha=1e-4
    )


class InfoBottleneckPenalty:
    def __init__(self, *, config, discriminator) -> None:
        logger.info(f"Creating info bottleneck with config {config}")
        self.config = config
        self.discriminator = discriminator

        class ItrInfo(NamedTuple):
            mean_kl: float

        self.ItrInfo = ItrInfo
        self.stacker = None

        z = discriminator.z
        z_dist = discriminator.z_dist

        with tf.variable_scope('bottleneck') as _bn:
            self.batch_size = tf.placeholder(tf.int32, (), name='batch_size')
            self.beta = tf.placeholder(tf.float32, (), name='beta')
            self.beta_value = 0
            r_normal = tf.distributions.Normal(
                loc=np.zeros(z.shape[1], dtype=np.float32),
                scale=np.ones(z.shape[1], dtype=np.float32)
            )

            def expert(var):
                return tf.slice(var, [self.batch_size], [-1])

            def policy(var):
                return tf.slice(var, [0], [self.batch_size])

            kls = tf.reduce_sum(
                r_normal.kl_divergence(z_dist),
                axis=1
            )

            with tf.variable_scope('expert'):
                self.expert_kl = tf.reduce_mean(expert(kls), axis=0)
            with tf.variable_scope('policy'):
                self.policy_kl = tf.reduce_mean(policy(kls), axis=0)

            self.mean_kl = 0.5 * (self.expert_kl + self.policy_kl)
            Ic = self.config.information_bottleneck_nats
            self.next_beta_value = tf.math.maximum(
                0.0,
                self.beta + self.config.kl_constraint_alpha * (self.mean_kl - Ic)
            )
            self.info_loss = self.beta * (self.mean_kl - Ic)

    def add_to_feed_and_request_dicts(self, *, feed_dict, request_dict, batch_size) -> None:
        if self.stacker is None:
            self.stacker = Stacker(self.ItrInfo)

        feed_dict[self.batch_size] = batch_size
        feed_dict[self.beta] = self.beta_value

        request_dict['mean_kl'] = self.mean_kl
        #request_dict['policy_kl'] = self.policy_kl
        #request_dict['expert_kl'] = self.expert_kl
        request_dict['next_beta'] = self.next_beta_value

    def update_from_result_dict(self, result_dict) -> None:
        self.beta_value = result_dict['next_beta']
        self.stacker.append(self.ItrInfo(mean_kl=result_dict['mean_kl']))

    def log(self, *, logger) -> None:
        logger.logkv('BottleneckKL', np.mean(self.stacker.mean_kl))
        logger.logkv('Ic', self.config.information_bottleneck_nats)
        logger.logkv('BottleneckBeta', self.beta_value)
        self.stacker = None


class RewardChangeConfig(Configuration):
    default_values = dict(
        enabled=True,
        reward_change_penalty=None,
        reward_change_constraint=None,
        reward_constraint_alpha=1e-5,
    )


class RewardChangePenalty:
    def __init__(self, *, config, discriminator) -> None:
        logger.info(f"Creating reward change penalty with config {config}")
        self.config = config
        self.discriminator = discriminator

        class ItrInfo(NamedTuple):
            approx_reward_kl: float

        self.ItrInfo = ItrInfo
        self.stacker = None

        with tf.variable_scope('reward_penalty') as scope:
            self.OLDLOGPTAU = tf.placeholder(tf.float32, [None, 1], name='OLDLOGPTAU')

            self.approx_reward_kl = tf.reduce_mean(
                (self.discriminator.log_p_tau - self.OLDLOGPTAU) ** 2
            )

            if self.config.reward_change_penalty is not None:
                self.reward_change_loss = self.config.reward_change_penalty * self.approx_reward_kl

            elif self.config.reward_change_constraint is not None:
                assert self.config.reward_change_penalty is None
                self.reward_beta = tf.placeholder(tf.float32, (), name='reward_beta')
                self.reward_beta_value = 1
                self.next_reward_beta = tf.math.maximum(
                    0.0,
                    self.reward_beta + self.config.reward_constraint_alpha * (
                        self.approx_reward_kl - self.config.reward_change_constraint
                    )
                )
                self.reward_change_loss = self.reward_beta * self.approx_reward_kl

    def add_to_feed_and_request_dicts(self, *, feed_dict, request_dict) -> None:
        if self.stacker is None:
            self.stacker = Stacker(self.ItrInfo)

        feed_dict[self.OLDLOGPTAU] = tf.get_default_session().run(
            self.discriminator.log_p_tau,
            feed_dict=feed_dict
        )

        if self.config.reward_change_constraint is not None:
            feed_dict[self.reward_beta] = self.reward_beta_value
            request_dict['approx_reward_kl'] = self.approx_reward_kl
            request_dict['next_reward_beta'] = self.next_reward_beta

    def update_from_result_dict(self, result_dict) -> None:
        self.reward_beta_value = result_dict['next_reward_beta']
        self.stacker.append(
            self.ItrInfo(approx_reward_kl=result_dict['approx_reward_kl'])
        )

    def log(self, *, logger) -> None:
        logger.logkv('RewardKL', np.mean(self.stacker.approx_reward_kl))
        logger.logkv('Reward Change Constraint', self.config.reward_change_constraint)
        logger.logkv('RewardBeta', self.reward_beta_value)
        self.stacker = None


class GradientPenaltyConfig(Configuration):
    default_values = dict(
        enabled=True,
        gradient_penalty=10
    )


class GradientPenalty:
    def __init__(self, *, config, discriminator):
        logger.info(f"Creating reward change penalty with config {config}")
        self.config = config
        self.discriminator = discriminator

        class ItrInfo(NamedTuple):
            grad_loss: float

        self.ItrInfo = ItrInfo
        self.stacker = None

        with tf.variable_scope('gradient_penalty') as scope:
            # For some reason I can't compute this out to obs_t, so I don't
            # really think that this works...
            policy_loss_grads = tf.gradients(
                discriminator.z,
                [
                    discriminator.obs_t,
                    discriminator.nobs_t,
                    discriminator.act_t
                ]
            )
            obs_gradsize = tf.reduce_mean(policy_loss_grads[0] ** 2)
            nobs_gradsize = tf.reduce_mean(policy_loss_grads[1] ** 2)
            grad_loss = obs_gradsize + nobs_gradsize
            if policy_loss_grads[2] is not None:
                # made up number
                act_t_gradsize = tf.reduce_mean(policy_loss_grads[2] ** 2) / 5
                grad_loss = grad_loss + act_t_gradsize
            self.gradient_loss = self.config.gradient_penalty * grad_loss

    def add_to_feed_and_request_dicts(self, *, feed_dict, request_dict) -> None:
        if self.stacker is None:
            self.stacker = Stacker(self.ItrInfo)

        request_dict['grad_loss'] = self.gradient_loss

    def update_from_result_dict(self, result_dict) -> None:
        self.stacker.append(
            self.ItrInfo(grad_loss=result_dict['grad_loss'])
        )

    def log(self, *, logger) -> None:
        logger.logkv('Gradient Loss', np.mean(self.stacker.grad_loss))
        self.stacker = None


class DiscriminatorConfiguration(Configuration):
    default_values = dict(
        name='discriminator',
        reward_arch=ArchConfiguration(),
        value_arch=ArchConfiguration(),
        fuse_archs=False,
        gradient_penalty=None,
        discount=0.99,
        state_only=True,
        max_itrs=100,
        drop_framestack=False,
        seed=0,
        score_config=ScoreConfiguration(),
        bottleneck_config=InfoBottleneckConfig(),
        reward_change_config=RewardChangeConfig(),
        gradient_penalty_config=GradientPenaltyConfig(),
        hidden_d=512
    )


class ItrData(NamedTuple):
    loss: float
    accuracy: float
    score: float


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

        self.drop_framestack = self.config.drop_framestack
        self.modify_obs = lambda obs: obs
        self.gamma = self.config.discount
        self.max_itrs = self.config.max_itrs

        # inputs
        self.obs_t = None
        self.nobs_t = None
        self.act_t = None
        self.nact_t = None
        self.labels = None
        self.lprobs = None
        self.lr = None
        self.train_time = None

        # encoding
        self.z = None
        self.z_dist = None
        self.z_dist_next = None

        # penalties
        self.bottleneck = None
        self.reward_change_modification = None
        self.gradient_penalty = None

        # outputs
        self.reward = None
        self.value_fn = None
        self.qfn = None
        self.discrim_output = None
        self.accuracy = None
        self.update_accuracy = None

        # training
        self.loss = None
        self.step = None
        self.grad_reward = None

        TfObject.__init__(self, config, scope_name=self.config.name)

    def _build_mu_sigma(self, x, config: ArchConfiguration, actions=None, reuse=False):
        with tf.variable_scope('mu', reuse=reuse):
            mu = last_linear_hidden_layer(x, config=config, actions=actions, d=self.config.hidden_d)
        with tf.variable_scope('sigma', reuse=reuse):
            sigma = last_linear_hidden_layer(x, config=config, actions=actions, d = self.config.hidden_d)
        return mu, sigma

    def _setup_reward_and_shaping_functions(self, *, obs_t, act_t, nobs_t, train_time):
        with tf.variable_scope('h'):
            h_mu, h_sigma = self._build_mu_sigma(
                obs_t,
                config=self.config.value_arch
            )
        with tf.variable_scope('h', reuse=True):
            h_next_mu, h_next_sigma = self._build_mu_sigma(
                nobs_t,
                config=self.config.value_arch,
                reuse=True
            )

        if not self.config.fuse_archs:
            with tf.variable_scope('g'):
                actions = None if self.config.state_only else act_t
                g_mu, g_sigma = self._build_mu_sigma(
                    obs_t,
                    config=self.config.reward_arch,
                    actions=actions
                )

            mu = tf.concat([g_mu, h_mu, h_next_mu], axis=1)
            sigma = tf.concat([g_sigma, h_sigma, h_next_sigma], axis=1)

            d_g = g_mu.shape[1].value
            d_h = h_mu.shape[1].value

            # these indices are offset + length
            g_idxs = (0, d_g)
            h_idxs = (d_g, d_h)
            h_next_idxs = (d_g + d_h, d_h)

        else:
            mu = tf.concat([h_mu, h_next_mu], axis=1)
            sigma = tf.concat([h_sigma, h_next_sigma], axis=1)

            d_h = h_mu.shape[1].value

            # these indices are offset + length
            g_idxs = (0, d_h)
            h_idxs = (0, d_h)
            h_next_idxs = (d_h, -1)

        noise = tf.random_normal(tf.shape(mu))
        self.z_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        self.z = tf.cond(
            train_time,
            lambda: mu + noise * sigma,
            lambda: mu
        )

        assert len(self.z.shape) == 2

        def slice_z(idxs):
            # Unconcatenate z
            return tf.slice(self.z, [0, idxs[0]], [-1, idxs[1]])

        with tf.variable_scope('vfn'):
            z_value_next = slice_z(h_next_idxs)
            fitted_value_fn_next = fc(
                z_value_next, 'output', nh=1, init_scale=np.sqrt(2)
            )
        with tf.variable_scope('vfn', reuse=True):
            z_value = slice_z(h_idxs)
            fitted_value_fn = fc(
                z_value, 'output', nh=1, init_scale=np.sqrt(2)
            )
        with tf.variable_scope('reward'):
            z_reward = slice_z(g_idxs)
            reward = fc(
                z_reward, 'output', nh=1, init_scale=np.sqrt(2)
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
            self.lr = tf.placeholder(tf.float32, (), name='lr')
            self.train_time = tf.placeholder(tf.bool, (), name='train_time')

            with tf.variable_scope('discrim') as dvs:
                (
                    self.reward, self.value_fn, fitted_value_fn_next
                ) = self._setup_reward_and_shaping_functions(
                    obs_t=self.obs_t,
                    nobs_t=self.nobs_t,
                    act_t=self.act_t,
                    train_time=self.train_time
                )
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
            if self.config.bottleneck_config.enabled:
                self.bottleneck = InfoBottleneckPenalty(
                    config=self.config.bottleneck_config,
                    discriminator=self
                )
                self.loss = self.loss + self.bottleneck.info_loss

            if self.config.gradient_penalty_config.enabled:
                self.gradient_penalty = GradientPenalty(
                    config=self.config.gradient_penalty_config,
                    discriminator=self
                )
                self.loss = self.loss + self.gradient_penalty.gradient_loss

            if self.config.reward_change_config.enabled:
                self.reward_change_modification = RewardChangePenalty(
                    config=self.config.reward_change_config,
                    discriminator=self
                )
                self.loss = self.loss + self.reward_change_modification.reward_change_loss

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
            request_dict = {
                'loss': self.loss,
                '_': self.step,
                'acc': self.update_accuracy,
                'scores': self.score_manager.score_tensor
            }

            if self.bottleneck is not None:
                self.bottleneck.add_to_feed_and_request_dicts(
                    feed_dict=feed_dict,
                    request_dict=request_dict,
                    batch_size=batch_size
                )

            if self.gradient_penalty is not None:
                self.gradient_penalty.add_to_feed_and_request_dicts(
                    feed_dict=feed_dict,
                    request_dict=request_dict
                )

            if self.reward_change_modification is not None:
                self.reward_change_modification.add_to_feed_and_request_dicts(
                    feed_dict=feed_dict,
                    request_dict=request_dict
                )

            result_dict = tf.get_default_session().run(
                request_dict,
                feed_dict=feed_dict
            )

            if self.bottleneck is not None:
                self.bottleneck.update_from_result_dict(result_dict)

            if self.gradient_penalty is not None:
                self.gradient_penalty.update_from_result_dict(result_dict)

            if self.reward_change_modification is not None:
                self.reward_change_modification.update_from_result_dict(result_dict)

            loss  = result_dict['loss']
            acc   = result_dict['acc']
            scores = result_dict['scores']

            policy_slice = slice(0, batch_size)
            expert_slice = slice(batch_size, -1)

            self.score_manager.train_itr(
                policy_scores=scores[policy_slice],
                expert_scores=scores[expert_slice]
            )

            stacker.append(ItrData(
                loss=loss,
                accuracy=acc,
                score=float(np.mean(scores[policy_slice]))
            ))

        mean_loss = np.mean(stacker.loss)
        mean_acc = np.mean(stacker.accuracy)
        mean_score = np.mean(stacker.score)

        self.score_manager.finalize_train_step(logger=logger)

        if logger:
            logger.logkv('GCLDiscrimLoss', mean_loss)
            logger.logkv('GCLDiscrimAccuracy', mean_acc)
            logger.logkv('GCLMeanScore', mean_score)
            if self.bottleneck is not None:
                self.bottleneck.log(logger=logger)
            if self.gradient_penalty is not None:
                self.gradient_penalty.log(logger=logger)
            if self.reward_change_modification is not None:
                self.reward_change_modification.log(logger=logger)

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
        if lprobs is not None and len(lprobs.shape) == 1:
            lprobs = lprobs.reshape((lprobs.shape[0], 1))

        score = self.score_manager.eval(
            obs=obs,
            acts=acts,
            next_obs=next_obs,
            lprobs=lprobs
        )

        if np.isnan(np.mean(score)):
            assert False, "NaN Score"

        return score
