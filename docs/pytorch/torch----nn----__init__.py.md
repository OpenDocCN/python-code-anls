# `.\pytorch\torch\nn\__init__.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类别名，包括参数(Parameter)，未初始化的缓冲区(UninitializedBuffer)，未初始化的参数(UninitializedParameter)
from torch.nn.parameter import (
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)
# 导入 torch.nn.modules 下的所有模块（但不建议这样做，因为 F403 号警告）
from torch.nn.modules import *  # noqa: F403
# 导入 torch.nn 下的模块和函数别名
from torch.nn import (
    attention as attention,
    functional as functional,
    init as init,
    modules as modules,
    parallel as parallel,
    parameter as parameter,
    utils as utils,
)
# 导入 torch.nn.parallel 中的 DataParallel 类别
from torch.nn.parallel import DataParallel as DataParallel


def factory_kwargs(kwargs):
    r"""Return a canonicalized dict of factory kwargs.

    Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed
    to factory functions like torch.empty, or errors if unrecognized kwargs are present.

    This function makes it simple to write code like this::

        class MyModule(nn.Module):
            def __init__(self, **kwargs):
                factory_kwargs = torch.nn.factory_kwargs(kwargs)
                self.weight = Parameter(torch.empty(10, **factory_kwargs))

    Why should you use this function instead of just passing `kwargs` along directly?

    1. This function does error validation, so if there are unexpected kwargs we will
    immediately report an error, instead of deferring it to the factory call
    2. This function supports a special `factory_kwargs` argument, which can be used to
    explicitly specify a kwarg to be used for factory functions, in the event one of the
    factory kwargs conflicts with an already existing argument in the signature (e.g.
    in the signature ``def f(dtype, **kwargs)``, you can specify ``dtype`` for factory
    functions, as distinct from the dtype argument, by saying
    ``f(dtype1, factory_kwargs={"dtype": dtype2})``)
    """
    # 如果 kwargs 为 None，则返回空字典
    if kwargs is None:
        return {}
    # 简单的关键字集合，这些关键字可以直接使用
    simple_keys = {"device", "dtype", "memory_format"}
    # 预期的关键字集合，包括简单关键字以及特殊的 factory_kwargs
    expected_keys = simple_keys | {"factory_kwargs"}
    # 如果 kwargs 中的关键字不在预期的集合中，抛出 TypeError 异常
    if not kwargs.keys() <= expected_keys:
        raise TypeError(f"unexpected kwargs {kwargs.keys() - expected_keys}")

    # 创建结果字典 r，初始化为 factory_kwargs 中的内容（如果存在）
    r = dict(kwargs.get("factory_kwargs", {}))
    # 遍历简单关键字集合
    for k in simple_keys:
        # 如果 kwargs 中包含该关键字
        if k in kwargs:
            # 如果 r 中已经包含该关键字，抛出 TypeError 异常（因为重复指定了）
            if k in r:
                raise TypeError(
                    f"{k} specified twice, in **kwargs and in factory_kwargs"
                )
            # 将 kwargs 中的关键字值赋给 r 中的对应关键字
            r[k] = kwargs[k]

    # 返回规范化后的结果字典 r
    return r
```