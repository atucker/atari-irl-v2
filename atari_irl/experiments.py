import inspect
from typing import Dict, Any, Type, List, NamedTuple, Optional, Set, Tuple
import argparse
import tensorflow as tf
import numpy as np
import joblib
import os.path
import hashlib
import re


class Cache:
    def __init__(self):
        self.context_stack = []

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

    def full_key(self, key):
        m = hashlib.md5()
        m.update(key.encode('utf-8'))
        hash = m.hexdigest()[:128]
        return os.path.join(self.base_dir, '/'.join(self.context_stack + [hash]))

    def __getitem__(self, key: str) -> Any:
        return joblib.load(self.full_key(key) + '.pkl')

    def __contains__(self, key: str) -> bool:
        return os.path.exists(self.full_key(key))

    def __setitem__(self, key: str, value: Any) -> None:
        curdir = self.base_dir
        for dir in self.context_stack:
            curdir = os.path.join(curdir, dir)
            if not os.path.exists(curdir):
                os.mkdir(curdir)

        joblib.dump(value, self.full_key(key) + '.pkl')

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
    version = 0

    def __init__(self, config, scope_name='', initialize=True):
        self.config = config
        with tf.variable_scope(scope_name) as scope:
            self.initialize_graph()
            self.scope = scope

    # Methods to deal with saving/restoring parameters at all
    @property
    def tensors(self) -> List[tf.Tensor]:
        return self.scope.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

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
    def create_from_cache(cls, cache: Cache, key: str) -> 'TfObject':
        (class_name, config_from_cache, values) = cache[key]
        assert class_name in cls._cachable_classes
        initialized_object = cls._cachable_classes[class_name](config_from_cache)
        initialized_object.restore(values)
        return initialized_object

    # Method for automatically training a model or retrieving it from the cache
    def cached_train(self, cache: Cache) -> 'TfObject':
        if self.key in cache:
            self.restore_values_from_cache(cache)
        else:
            self.train(cache)
            self.store_in_cache(cache)
        return self

    def initialize_graph(self):
        pass

    def train(self, cache):
        raise NotImplemented()

    def store_training_checkpoint(self, cache: Cache, itr: int, extra_data={}):
        with cache.context('training'):
            with cache.context(cache.full_key(self.key)):
                with cache.context(str(itr)):
                    self.store_in_cache(cache)
                    for key, value in extra_data.items():
                        cache[key] = value

    def restore_training_checkpoint(self, cache: Cache):
        with cache.context('training'):
            with cache.context(cache.full_key(self.key)):
                available_itrs = cache.context_item_keys()
                if not available_itrs:
                    assert False, "restore initialized from empty context, cannot resume training"
                itr = int(max(available_itrs))
                with cache.context(str(itr)):
                    self.restore_values_from_cache(cache)
                    extra_data = {}
                    for key in cache.context_item_keys():
                        extra_data[key] = cache[key]
        return itr, extra_data

