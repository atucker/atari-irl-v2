import numpy as np
from typing import Type, NamedTuple, Any
from collections import OrderedDict


class Stacker:
    def __init__(self, other_cls: Type) -> None:
        self.data_cls = other_cls
        self.data = OrderedDict((f, []) for f in self.data_cls._fields)

    def append(self, tup: NamedTuple) -> None:
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
