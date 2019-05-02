import numpy as np
from typing import Type, NamedTuple, Any, Union
from collections import OrderedDict
import psutil
from contextlib import contextmanager

class Stacker:
    def __init__(self, other_cls: Type) -> None:
        self.data_cls = other_cls
        self.data = OrderedDict((f, []) for f in self.data_cls._fields)

    def append(self, tup: Any) -> None:
        assert isinstance(tup, self.data_cls)
        for f in tup._fields:
            self.data[f].append(getattr(tup, f))

    def __getattr__(self, item) -> Any:
        return self.data[item]

    
def one_hot(x, dim):
    assert isinstance(x, list) or len(x.shape) == 1
    ans = np.zeros((len(x), dim))
    for n, i in enumerate(x):
        ans[n, i] = 1
    return ans


def inv_sf01(arr, s):
    return arr.reshape(s[1], s[0], *s[2:]).swapaxes(0, 1)


@contextmanager
def light_log_mem(name):
    before_mem = psutil.virtual_memory().used / 1024
    yield
    after_mem = psutil.virtual_memory().used / 1024
    print(f"{before_mem} used before {name}, increased by {after_mem - before_mem}")
