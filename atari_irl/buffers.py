from typing import TypeVar, Tuple, Callable, Iterator, Optional, Any
from .headers import Buffer, TimeShape, Batch, EnvInfo
import numpy as np
from collections import namedtuple

T = TypeVar('T')


class DummyBuffer(Buffer[T]):
    def __init__(
        self, *,
        overwrite_rewards: bool,
        overwrite_logprobs: bool,
        discriminator: Optional[Any] = None,
        policy: Optional[Any] = None,
    ):
        super().__init__(
            overwrite_rewards=overwrite_rewards,
            overwrite_logprobs=overwrite_logprobs,
            discriminator=discriminator,
            policy=policy,
            time_shape=None,
            policy_info=None,
            env_info=None,
            sampler_state=None
        )
        self.batch = None

    def add_batch(self, samples: Batch) -> None:
        super().add_batch(samples)
        self.batch = samples

        self.time_shape = samples.time_shape
        self.env_info = samples.env_info
        self.policy_info = samples.policy_info
        self.sampler_state = samples.sampler_state


class IterableBufferMixin:
    def iter_items(self, *keys, start_at=0) -> Iterator:
        assert self.time_shape, "Tried to use IterableBufferMixin without time_shape"
        TupClass = namedtuple('TupClass', keys)

        if self.time_shape.batches is None:
            if self.time_shape.num_envs is None or self.time_shape.T is None:
                for i in range(self.obs.shape[0]):
                    if i >= start_at:
                        yield TupClass(**dict(
                            (key, getattr(self, key)[i]) for key in keys
                        ))
            else:
                for i in range(self.obs.shape[0]):
                    for j in range(self.obs.shape[1]):
                        if i * self.obs.shape[1] + j >= start_at:
                            yield TupClass(**dict(
                                (key, getattr(self, key)[i, j]) for key in keys
                            ))
        else:
            for b in range(self.time_shape.batches):
                if self.time_shape.num_envs is None or self.time_shape.T is None:
                    for i in range(self.obs[b].shape[0]):
                        if i >= start_at:
                            yield TupClass(**dict(
                                (key, getattr(self, key)[b][i]) for key in keys
                            ))
                else:
                    for i in range(self.obs[b].shape[0]):
                        for j in range(self.obs[b].shape[1]):
                            if i * self.obs[b].shape[1] + j >= start_at:
                                yield TupClass(**dict(
                                    (key, getattr(self, key)[b][i, j]) for key in keys
                                ))


class FlatBuffer(Buffer[T]):
    def __init__(self, discriminator, policy_info_class):
        super().__init__(
            discriminator=discriminator,
            time_shape=None,
            policy_info=None,
            env_info=None,
            sampler_state=None
        )
        self.policy_info_class = policy_info_class
        
    def add_batch(self, samples: Batch, debug=False) -> None:
        super().add_batch(samples)
        assert samples.time_shape.num_envs is not None
        assert samples.time_shape.T is not None
        
        N = samples.time_shape.T * samples.time_shape.num_envs
        
        flattened_batch_time_shape = TimeShape(T=N, num_envs=None)
        def flatten_field_data(data):
            return flattened_batch_time_shape.reshape(
                from_time_shape=samples.time_shape,
                data=data
            )
        flattened_env_info = EnvInfo(
            time_shape=flattened_batch_time_shape,
            obs=flatten_field_data(samples.obs),
            rewards=flatten_field_data(samples.rewards),
            dones=flatten_field_data(samples.dones),
            next_obs=flatten_field_data(samples.next_obs),
            next_dones=flatten_field_data(samples.next_dones),
            epinfobuf=samples.env_info.epinfobuf
        )
        
        policy_info_dict = {'time_shape': samples.policy_info.time_shape}
        for field in samples.policy_info._fields:
            if field != 'time_shape':
                policy_info_dict[field] = flatten_field_data(getattr(samples.policy_info, field))
        flattened_policy_info = self.policy_info_class(**policy_info_dict)
        
        
        if self.time_shape is None:
            assert self.policy_info is None
            assert self.env_info is None
            assert self.sampler_state is None
            
            self.time_shape = flattened_batch_time_shape
            self.env_info = flattened_env_info
            self.policy_info = flattened_policy_info
        else:
            self.time_shape = TimeShape(T=self.time_shape.T + N, num_envs=None)
            self.env_info = EnvInfo(
                time_shape=self.time_shape,
                obs=np.vstack((self.env_info.obs, flattened_env_info.obs)),
                rewards=np.vstack((self.env_info.rewards, flattened_env_info.rewards)),
                dones=np.vstack((self.env_info.dones, flattened_env_info.dones)),
                next_obs=np.vstack((self.env_info.next_obs, flattened_env_info.next_obs)),
                next_dones=np.vstack((self.env_info.next_dones, flattened_env_info.next_dones)),
                epinfobuf=self.env_info.epinfobuf + flattened_env_info.epinfobuf
            )
            self.policy_info.time_shape = self.time_shape
            for field in self.policy_info._fields:
                if field != 'time_shape':
                    setattr(
                        self.policy_info,
                        field,
                        np.vstack((
                            getattr(self.policy_info, field),
                            getattr(flattened_policy_info, field)
                        ))
                    )
                
        if debug:
            print(self.env_info.time_shape)
            for field in self.env_info._fields:
                if field not in {'time_shape', 'epinfobuf'}:
                    print(f"{field}: {getattr(self.env_info, field).shape}")
            print(self.policy_info.time_shape)
            for field in self.policy_info._fields:
                if field not in {'time_shape', 'epinfobuf'}:
                    print(f"{field}: {getattr(self.policy_info, field).shape}")
            print()

        self.sampler_state = samples.sampler_state

    
class BatchedList:
    """
    Holds a list of numpy arrays, and manages access to them through indices
    """
    def __init__(self, maxlen=None):
        self.time_shape = None
        self._data = []
        self.maxlen=maxlen
        
    def append(self, batch):
        assert self.maxlen is None or self._data <= self.maxlen
        if self.time_shape is None:
            assert len(self._data) == 0
            self.time_shape = TimeShape(
                num_envs=batch.shape[0],
                T=batch.shape[1]
            )
            
        self._data.append(batch)
        if self.maxlen is not None and len(self._data) > self.maxlen:
            self._data.pop(0)
        
    def get_idxs(self, key):
        if isinstance(key, tuple):
            if len(key) == 3:
                return key
            elif len(key) == 2:
                t = key[1] % self.time_shape.T
                b = int(key[1] / self.time_shape.T)
                n = key[0]
                return (b, n, t)
            else:
                assert False
        elif isinstance(key, int) or isinstance(key, np.int64):
            T = self.time_shape.T * len(self._data)
            n = int(key / T)
            remainder = key - n * T
            t = remainder % self.time_shape.T
            b = int(remainder / self.time_shape.T)
            return (b, n, t)
        else:
            assert False
        
    def __getitem__(self, key):
        assert self.time_shape is not None
        if isinstance(key, tuple) or isinstance(key, int) or isinstance(key, np.int64):
            b, n, t = self.get_idxs(key)
            return self._data[b][n, t]
        else:
            return np.array([self[k] for k in key])
        
    def tplusone(self, key):
        assert self.time_shape is not None
        assert not isinstance(key, tuple)
        if isinstance(key, int) or isinstance(key, np.int64):
            _b, _n, _t = self.get_idxs(key)
            b, n, t = self.get_idxs(key + 1)
            if n != _n:
                return self._data[b][n, t] * 0
            else:
                return self._data[b][n, t]
        else:
            return np.array([self[k] for k in key])
        
    def __mul__(self, coefficient):
        for d in self._data:
            d *= coefficient
        return self
     
    
class ViewBuffer(Buffer[T], IterableBufferMixin):
    def __init__(self, *, discriminator, policy, policy_info_class, maxlen=None):
        super().__init__(
            overwrite_rewards=True,
            overwrite_logprobs=True,
            discriminator=discriminator,
            policy=policy,
            time_shape=None,
            policy_info=None,
            env_info=EnvInfo(
                time_shape=None,
                obs=BatchedList(maxlen=maxlen),
                rewards=BatchedList(maxlen=maxlen),
                dones=BatchedList(maxlen=maxlen),
                next_obs=BatchedList(maxlen=maxlen),
                next_dones=BatchedList(maxlen=maxlen),
                epinfobuf=[]
            ),
            sampler_state=None
        )
        self.policy_info_class = policy_info_class
        
        policy_info_dict = {'time_shape': self.time_shape}
        for field in policy_info_class._fields:
            if field != 'time_shape':
                policy_info_dict[field] = BatchedList()

        self.discriminator = discriminator
        self.policy_info = self.policy_info_class(**policy_info_dict)
        
    def add_batch(self, samples: Batch, debug=False) -> None:
        super().add_batch(samples)
        assert samples.time_shape.num_envs is not None
        assert samples.time_shape.T is not None
        assert samples.time_shape.batches is None
        
        if self.time_shape is None:
            assert self.sampler_state is None
            self.time_shape = TimeShape(
                batches=1,
                T=samples.time_shape.T,
                num_envs=samples.time_shape.num_envs
            )
        else:
            assert self.time_shape.T == samples.time_shape.T
            assert self.time_shape.num_envs == samples.time_shape.num_envs
            
            self.time_shape = TimeShape(
                batches=self.time_shape.batches + 1,
                T=samples.time_shape.T,
                num_envs=samples.time_shape.num_envs
            )

        for field in self.env_info._fields:
            if field not in ('time_shape', 'epinfobuf'):
                getattr(self.env_info, field).append(
                    getattr(samples.env_info, field)
                )
        self.env_info.epinfobuf.extend(samples.env_info.epinfobuf)
                
        for field in self.policy_info._fields:
            if field != 'time_shape':
                getattr(self.policy_info, field).append(
                    getattr(samples.policy_info, field)
                )
                
        #self.env_info.time_shape = self.time_shape
        self.policy_info.time_shape = self.time_shape
        self.sampler_state = samples.sampler_state
    
    @property
    def next_acts(buffer):
        class NextActions:
            def __getitem__(self, key):
                return buffer.acts.tplusone(key)
        return NextActions()
