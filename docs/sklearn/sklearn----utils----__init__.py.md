# `D:\src\scipysrc\scikit-learn\sklearn\utils\__init__.py`

```
"""Various utilities to help with development."""

# 导入平台相关模块和警告模块
import platform
import warnings
# 导入集合抽象类中的序列类
from collections.abc import Sequence

# 导入 NumPy 库并使用 np 别名
import numpy as np

# 从内部模块导入自定义异常
from ..exceptions import DataConversionWarning
# 导入内部模块中的特定功能模块
from . import _joblib, metadata_routing
# 导入自定义的数据结构 Bunch
from ._bunch import Bunch
# 导入数据处理相关功能
from ._chunking import gen_batches, gen_even_slices
# 导入估计器 HTML 表示相关功能
from ._estimator_html_repr import estimator_html_repr

# 使 _safe_indexing 从当前模块导入，用于向后兼容性，即使这个特定的辅助函数被视为半私有，并且通常对希望遵循 scikit-learn 估计器 API 的第三方库非常有用。
# 特别地，尽管其名称以 `_` 开头，_safe_indexing 已包含在我们的公共 API 文档中。
from ._indexing import (
    _safe_indexing,  # noqa
    resample,
    shuffle,
)
# 导入掩码相关功能
from ._mask import safe_mask
# 导入类权重计算相关功能
from .class_weight import compute_class_weight, compute_sample_weight
# 导入弃用相关功能
from .deprecation import deprecated
# 导入发现所有估计器的功能
from .discovery import all_estimators
# 导入扩展数学功能
from .extmath import safe_sqr
# 导入 MurmurHash 哈希算法
from .murmurhash import murmurhash3_32
# 导入验证相关功能
from .validation import (
    as_float_array,
    assert_all_finite,
    check_array,
    check_consistent_length,
    check_random_state,
    check_scalar,
    check_symmetric,
    check_X_y,
    column_or_1d,
    indexable,
)

# TODO(1.7): 删除 parallel_backend 和 register_parallel_backend
msg = "在1.5版本中弃用，在1.7版本中将移除。请使用 joblib.{}。"
# 使用 deprecated 装饰器标记 register_parallel_backend 函数
register_parallel_backend = deprecated(msg)(_joblib.register_parallel_backend)


# 如果是一个类，则 deprecated 将更改 _joblib 模块中的对象，因此我们需要子类化它
@deprecated(msg)
class parallel_backend(_joblib.parallel_backend):
    pass


__all__ = [
    "murmurhash3_32",
    "as_float_array",
    "assert_all_finite",
    "check_array",
    "check_random_state",
    "compute_class_weight",
    "compute_sample_weight",
    "column_or_1d",
    "check_consistent_length",
    "check_X_y",
    "check_scalar",
    "indexable",
    "check_symmetric",
    "deprecated",
    "parallel_backend",
    "register_parallel_backend",
    "resample",
    "shuffle",
    "all_estimators",
    "DataConversionWarning",
    "estimator_html_repr",
    "Bunch",
    "metadata_routing",
    "safe_sqr",
    "safe_mask",
    "gen_batches",
    "gen_even_slices",
]


# TODO(1.7): 删除
# 定义一个函数 __getattr__，用于获取指定名称的属性
def __getattr__(name):
    if name == "IS_PYPY":
        # 如果请求获取的属性为 "IS_PYPY"，则发出警告
        warnings.warn(
            "IS_PYPY 已弃用，并将在1.7版本中移除。",
            FutureWarning,
        )
        # 返回当前 Python 实现是否为 PyPy
        return platform.python_implementation() == "PyPy"
    # 如果请求获取的属性不存在，则引发 AttributeError 异常
    raise AttributeError(f"模块 {__name__} 中没有属性 {name}")


# TODO(1.7): 删除 tosequence
# 使用 deprecated 装饰器标记 tosequence 函数
@deprecated("在1.5版本中弃用，在1.7版本中将移除")
# 定义函数 tosequence，将可迭代对象 x 转换为序列，尽可能避免复制
def tosequence(x):
    """将可迭代对象 x 转换为序列，尽可能避免复制。

    Parameters
    ----------
    x : iterable
        要转换的可迭代对象。

    Returns
    -------
    """
    # 检查输入参数 x 是否为 NumPy 数组（ndarray）
    if isinstance(x, np.ndarray):
        # 如果是 NumPy 数组，则将其转换为 ndarray 类型并返回
        return np.asarray(x)
    # 如果输入参数 x 是 Python 内置的 Sequence 类型（如列表、元组等），则直接返回 x
    elif isinstance(x, Sequence):
        return x
    # 如果输入参数 x 不是 NumPy 数组也不是 Sequence 类型，将其转换为列表类型并返回
    else:
        return list(x)
```