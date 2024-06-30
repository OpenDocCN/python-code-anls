# `D:\src\scipysrc\scikit-learn\sklearn\__init__.py`

```
# 配置全局设置并获取工作环境信息。

# 引入日志记录模块
import logging
# 引入操作系统相关功能模块
import os
# 引入随机数生成模块
import random
# 引入系统相关模块
import sys

# 从内部模块中导入配置相关函数和对象
from ._config import config_context, get_config, set_config

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


# PEP0440 兼容的格式化版本号，详情请见:
# https://www.python.org/dev/peps/pep-0440/
#
# 通用发布标记:
#   X.Y.0   # Y增量后的第一个发布版本
#   X.Y.Z   # Bug修复版本
#
# 可接受的预发布标记:
#   X.Y.ZaN   # Alpha版本
#   X.Y.ZbN   # Beta版本
#   X.Y.ZrcN  # 候选发布版本
#   X.Y.Z     # 最终发布版本
#
# 开发分支标记为: 'X.Y.dev' 或 'X.Y.devN'，其中N为整数。
# 'X.Y.dev0' 是 'X.Y.dev' 的规范版本。
#
__version__ = "1.6.dev0"


# 在OSX上，由于同时加载多个OpenMP库，可能导致运行时错误。
# 这可能发生在调用BLAS内部的prange时。
# 设置以下环境变量允许同时加载多个OpenMP库。
# 这不应降低性能，因为我们在代码的特定部分手动处理潜在的超订阅性能问题，
# 在可能出现嵌套OpenMP循环的代码段中，通过动态重新配置内部OpenMP运行时来临时禁用它。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# 解决intel-openmp 2019.5中发现的问题:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

try:
    # 此变量由构建过程注入到__builtins__中。
    # 在二进制未构建时，用于启用导入sklearn子包。
    # mypy错误: 无法确定'__SKLEARN_SETUP__'的类型
    __SKLEARN_SETUP__  # type: ignore
except NameError:
    __SKLEARN_SETUP__ = False

if __SKLEARN_SETUP__:
    sys.stderr.write("在构建过程中部分导入sklearn。\n")
    # 在构建过程中不导入scikit-learn的其余部分，
    # 因为可能尚未编译完成。
else:
    # `_distributor_init` 允许分发商运行自定义的初始化代码。
    # 例如，对于Windows wheel，这用于预加载
    # 存储在sklearn/.libs子文件夹中的OpenMP中的vcomp共享库运行时。
    # 必须在导入show_versions之前执行此操作，
    # 因为这样做有助于确保正确的共享库加载。
    pass
    # 导入 __check_build、_distributor_init 函数，这些函数与 OpenMP 运行时相关联，
    # 先导入它们是为了能够检查和使用 OpenMP dll，如果找不到 OpenMP dll，则导入会失败。
    from . import (
        __check_build,  # noqa: F401
        _distributor_init,  # noqa: F401
    )
    # 导入 clone 函数，位于当前包的 base 模块中
    from .base import clone
    # 导入 show_versions 函数，位于当前包的 utils 模块的 _show_versions 子模块中
    from .utils._show_versions import show_versions

    # __all__ 列表定义了当前模块中所有公开的名称
    __all__ = [
        "calibration",
        "cluster",
        "covariance",
        "cross_decomposition",
        "datasets",
        "decomposition",
        "dummy",
        "ensemble",
        "exceptions",
        "experimental",
        "externals",
        "feature_extraction",
        "feature_selection",
        "gaussian_process",
        "inspection",
        "isotonic",
        "kernel_approximation",
        "kernel_ridge",
        "linear_model",
        "manifold",
        "metrics",
        "mixture",
        "model_selection",
        "multiclass",
        "multioutput",
        "naive_bayes",
        "neighbors",
        "neural_network",
        "pipeline",
        "preprocessing",
        "random_projection",
        "semi_supervised",
        "svm",
        "tree",
        "discriminant_analysis",
        "impute",
        "compose",
        # Non-modules:
        "clone",
        "get_config",
        "set_config",
        "config_context",
        "show_versions",
    ]

    # _BUILT_WITH_MESON 变量用于标记是否使用 Meson 构建工具构建的 sklearn
    _BUILT_WITH_MESON = False
    try:
        # 尝试导入 sklearn._built_with_meson 模块，如果成功则说明是用 Meson 构建的
        import sklearn._built_with_meson  # noqa: F401
        # 设置 _BUILT_WITH_MESON 为 True
        _BUILT_WITH_MESON = True
    except ModuleNotFoundError:
        # 如果模块未找到，则将 _BUILT_WITH_MESON 保持为 False
        pass
# 定义模块设置函数，用于确保全局可控的随机数生成器种子
def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    
    import numpy as np  # 导入 NumPy 库

    # 检查环境变量中是否存在随机种子，如果不存在则创建一个
    _random_seed = os.environ.get("SKLEARN_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max  # 生成一个随机种子
    _random_seed = int(_random_seed)  # 将随机种子转换为整数类型
    print("I: Seeding RNGs with %r" % _random_seed)  # 打印种子信息
    np.random.seed(_random_seed)  # 使用 NumPy 设置随机数生成器种子
    random.seed(_random_seed)  # 使用 Python 内置模块设置随机数生成器种子
```