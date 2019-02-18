import inspect
from typing import Dict, Any, Type, List, NamedTuple, Optional
import argparse
import tensorflow as tf
import numpy as np


class Cache:
    def __getitem__(self, key: str) -> Any:
        raise NotImplemented()

    def __contains__(self, key: str) -> bool:
        raise NotImplemented()

    def __setitem__(self, key: str, value: Any) -> None:
        raise NotImplemented()


class DictCache(Cache):
    def __init__(self):
        self.data = {}

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value


class Configuration:
    default_values = {}
    attrs_exclude_from_key = set()

    def __init__(self, **overrides: Dict[str, Any]) -> None:
        self.items = dict(**self.default_values)
        for key, val in overrides.items():
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
        return self.items[key]

    @property
    def key(self) -> str:
        return ','.join([
            f"{key}={getattr(self, key)}"
            for key in self.items.keys()
            if key not in self.attrs_exclude_from_key
        ])


class Context(NamedTuple):
    config: Configuration
    cache: Cache


class Stage:
    name = ""
    version = 0
    config_class = Configuration

    def __init__(self, args):
        self.args = args

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        cls.config_class.add_args(parser)

    def run(self, context: Context) -> Any:
        assert isinstance(context.config, self.config_class)
        key = f"stage-{self.name};config-{context.config.key};version-{self.version}"

        if key not in context.cache:
            ans = self.execute(context.config)
            context.cache[key] = ans
            assert key in context.cache

        return context.cache[key]

    def execute(self, context: Context) -> Any:
        raise NotImplemented()


class TfObject:
    _cachable_classes = {}
    class_registration_name = None
    version = 0

    def __init__(self, config, scope_name=''):
        self.tf_obj_config = config
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
        return f"tf_obj-{self.class_registration_name}-v{self.version};config-{self.tf_obj_config.key}"

    def store_in_cache(self, cache: Cache) -> None:
        cache[self.key] = (self.class_registration_name, self.tf_obj_config, self.values)

    def restore_values_from_cache(self, cache: Cache) -> None:
        (class_name, config_from_cache, values) = cache[self.key]
        assert class_name == self.class_registration_name
        assert config_from_cache == self.tf_obj_config
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
            self.train()
            self.store_in_cache(cache)
        return self

    def initialize_graph(self):
        pass

    def train(self):
        pass
