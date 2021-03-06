import inspect
from typing import Dict, Any, Type, List, NamedTuple, Optional, Set, Tuple, Union, Generic, TypeVar
import argparse
import tensorflow as tf
import numpy as np
import joblib
import os.path
import hashlib
import re

class CacheHash(NamedTuple):
    value: str

class Cache:
    def __init__(self):
        self.context_stack = []

    @staticmethod
    def hash_key(key: Union[str, CacheHash]) -> str:
        if isinstance(key, CacheHash):
            hash = key.value
        else:
            m = hashlib.md5()
            m.update(key.encode('utf-8'))
            hash = m.hexdigest()[:128]
        return hash

    def full_key(self, key):
        return key + '_' + '.'.join(self.context_stack)

    def __getitem__(self, key: str) -> Any:
        raise NotImplemented()

    def __contains__(self, key: str) -> bool:
        raise NotImplemented()

    def __setitem__(self, key: str, value: Any) -> None:
        raise NotImplemented()

    def context(cache, name):
        class CacheContext:
            def __init__(self):
                self.name = name

            def __enter__(self):
                cache.context_stack.append(self.name)

            def __exit__(self, type, value, traceback):
                cache.context_stack.pop()

        return CacheContext()

    def context_item_keys(self) -> List[str]:
        """
        Get the item keys that are under our context
        """
        raise NotImplementedError


class DictCache(Cache):
    def __init__(self):
        super().__init__()
        self.data = {}

    def __getitem__(self, key: str) -> Any:
        return self.data[self.full_key(key)]

    def __contains__(self, key: str) -> bool:
        return self.full_key(key) in self.data

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[self.full_key(key)] = value

    def context_items(self) -> List[str]:
        raise NotImplementedError


class FilesystemCache(Cache):
    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

    def full_key(self, key: Union[str, CacheHash]):
        hash = self.hash_key(key)
        return os.path.join(self.base_dir, '/'.join(self.context_stack + [hash]))

    def filename(self, key: Union[str, CacheHash]):
        return f"{self.full_key(key)}.pkl"

    def __getitem__(self, key: Union[str, CacheHash]) -> Any:
        return joblib.load(self.filename(key))

    def __contains__(self, key: Union[str, CacheHash]) -> bool:
        return os.path.exists(self.filename(key))

    def __setitem__(self, key: Union[str, CacheHash], value: Any) -> None:
        curdir = self.base_dir
        for dir in self.context_stack:
            curdir = os.path.join(curdir, dir)
            if not os.path.exists(curdir):
                os.mkdir(curdir)

        joblib.dump(value, self.filename(key))

    def context_item_keys(self) -> List[str]:
        return os.listdir(os.path.join(self.base_dir, '/'.join(self.context_stack)))


class Configuration:
    #TODO(Aaron): Actually make the default values immutable
    default_values: Dict[str, Any] = {}
    attrs_exclude_from_key: Set[str] = set()

    def __init__(self, **overrides: Any) -> None:
        self.items = dict(**self.default_values)
        for key, val in overrides.items():
            assert key in self.default_values, f"Unrecognized attribute {key}"
            self.items[key] = val

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        pass
    
    @classmethod
    def get_args_from_parser(cls, parser: argparse.ArgumentParser) -> Dict[str, Any]:
        return dict(*[
            (key, getattr(parser, key)) for key in cls.default_values.keys()
        ])

    def __getattr__(self, key: str) -> Any:
        try:
            return super(Configuration, self).__getattribute__('items')[key]
        except KeyError:
            return super(Configuration, self).__getattr__(key)

    @property
    def key(self) -> str:
        return ','.join([
            f"{key}={str(self.items[key])}"
            for key in self.items.keys()
            if key not in self.attrs_exclude_from_key
        ])

    def __str__(self) -> str:
        return self.key

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Configuration):
            return all(
                other.items[key] == value for key, value in self.items.items()
            )
        else:
            return False


class Context(NamedTuple):
    config: Configuration
    cache: Cache


class TfObject:
    _cachable_classes = {}
    class_registration_name = None
    version = 1.0

    def __init__(self, config, scope_name='', initialize=True):
        print(f"Initializing TfObject {self.class_registration_name} with config {config}")
        self.config = config

        with tf.variable_scope(scope_name) as scope:
            self.initialize_graph()
            self._scope = scope
            self.scope_name = f"{tf.get_variable_scope().name}/{scope_name}"

    def initialize_graph(self):
        raise NotImplementedError

    # Methods to deal with saving/restoring parameters at all
    @property
    def tensors(self) -> List[tf.Tensor]:
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            self.scope_name
        )

    @property
    def values(self) -> List[np.ndarray]:
        return tf.get_default_session().run(self.tensors)

    def restore(self, values: List[np.ndarray]):
        restores = []
        for tf_tensor, np_array in zip(self.tensors, values):
            restores.append(tf_tensor.assign(np_array))
        tf.get_default_session().run(restores)

    # Methods to save/restore from a cache
    @property
    def key(self):
        return f"tf_obj-{self.class_registration_name}-v{self.version};config-{self.config.key}"

    def store_in_cache(self, cache: Cache) -> None:
        print(f"Saving tfobject {self.key} in {cache.filename(self.key)}")
        cache[self.key] = (self.class_registration_name, self.config, self.values)
        config_fname = cache.full_key(self.key) + '_description.txt'
        with open(config_fname, 'w') as f:
            for part in re.split('[,;-]', self.key):
                f.write(part + '\n')

    def restore_values_from_cache(self, cache: Cache) -> None:
        (class_name, config_from_cache, values) = cache[self.key]
        assert class_name == self.class_registration_name
        assert config_from_cache == self.config
        self.restore(values)

    # Methods for reconstructing an object from purely the cache
    @classmethod
    def register_cachable_class(cls, class_name, obj_class: Type['TfObject']) -> None:
        assert class_name not in cls._cachable_classes
        cls._cachable_classes[class_name] = obj_class

    @classmethod
    def _create_from_tuple(cls, tuple):
        (class_name, config_from_cache, values) = tuple
        assert class_name in cls._cachable_classes
        initialized_object = cls._cachable_classes[class_name](config_from_cache)
        initialized_object.restore(values)
        return initialized_object

    @classmethod
    def create_from_file(cls, fname: str) -> 'TfObject':
        return cls._create_from_tuple(joblib.load(fname))

    @classmethod
    def create_from_cache(cls, cache: Cache, key: str) -> 'TfObject':
        return cls._create_from_tuple(cache[key])


T = TypeVar('T')


class TfObjectTrainer(Generic[T]):
    def __init__(self, trainee: T) -> None:
        assert isinstance(trainee, TfObject)
        self.trainee = trainee

    # Method for automatically training a model or retrieving it from the cache
    def cached_train(self, cache: Cache) -> T:
        if self.trainee.key in cache:
            self.trainee.restore_values_from_cache(cache)
        else:
            self.train(cache)
            self.trainee.store_in_cache(cache)
        return self.trainee

    def train(self, cache):
        raise NotImplemented()

    def store_training_checkpoint(self, cache: Cache, itr: int, extra_data={}):
        with cache.context('training'):
            with cache.context(cache.hash_key(self.trainee.key)):
                with cache.context(str(itr)):
                    self.trainee.store_in_cache(cache)
                    cache['extra_data'] = extra_data

    def restore_training_checkpoint(self, cache: Cache, itr=None):
        with cache.context('training'):
            with cache.context(cache.hash_key(self.trainee.key)):
                available_itrs = cache.context_item_keys()
                if not available_itrs:
                    print("Didn't find anything to resume, starting from scratch")
                    return 1, {}
                itr = itr or max([int(itr) for itr in available_itrs])
                with cache.context(str(itr)):
                    self.trainee.restore_values_from_cache(cache)
                    if 'extra_data' in cache:
                        extra_data = cache['extra_data']
                    else:
                        extra_data = {}

        print(f"resuming training from iteration {itr} using {cache.filename(self.trainee.key)}")
        return itr, extra_data

