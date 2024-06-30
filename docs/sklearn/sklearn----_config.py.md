# `D:\src\scipysrc\scikit-learn\sklearn\_config.py`

```
"""Global configuration state and functions for management"""

import os  # 导入操作系统相关模块
import threading  # 导入线程相关模块
from contextlib import contextmanager as contextmanager  # 导入上下文管理器模块

# 全局配置字典，包含各种默认配置项
_global_config = {
    "assume_finite": bool(os.environ.get("SKLEARN_ASSUME_FINITE", False)),  # 从环境变量获取或默认为 False
    "working_memory": int(os.environ.get("SKLEARN_WORKING_MEMORY", 1024)),  # 从环境变量获取或默认为 1024
    "print_changed_only": True,  # 默认为 True
    "display": "diagram",  # 默认为 "diagram"
    "pairwise_dist_chunk_size": int(
        os.environ.get("SKLEARN_PAIRWISE_DIST_CHUNK_SIZE", 256)
    ),  # 从环境变量获取或默认为 256
    "enable_cython_pairwise_dist": True,  # 默认为 True
    "array_api_dispatch": False,  # 默认为 False
    "transform_output": "default",  # 默认为 "default"
    "enable_metadata_routing": False,  # 默认为 False
    "skip_parameter_validation": False,  # 默认为 False
}

_threadlocal = threading.local()  # 创建线程本地存储对象


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, "global_config"):  # 如果线程本地存储中没有 global_config 属性
        _threadlocal.global_config = _global_config.copy()  # 则复制全局配置作为默认配置
    return _threadlocal.global_config  # 返回线程本地的全局配置


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global scikit-learn configuration.
    set_config : Set global scikit-learn configuration.

    Examples
    --------
    >>> import sklearn
    >>> config = sklearn.get_config()
    >>> config.keys()
    dict_keys([...])
    """
    # 返回线程本地配置的副本，以防止用户修改返回的字典而影响全局配置
    return _get_threadlocal_config().copy()


def set_config(
    assume_finite=None,
    working_memory=None,
    print_changed_only=None,
    display=None,
    pairwise_dist_chunk_size=None,
    enable_cython_pairwise_dist=None,
    array_api_dispatch=None,
    transform_output=None,
    enable_metadata_routing=None,
    skip_parameter_validation=None,
):
    """Set global scikit-learn configuration.

    .. versionadded:: 0.19

    Parameters
    ----------
    assume_finite : bool, default=None
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error.  Global default: False.

        .. versionadded:: 0.19

    working_memory : int, default=None
        If set, scikit-learn will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. Global default: 1024.

        .. versionadded:: 0.20

    """
    # 函数体未完整，未包含完整的参数说明
    print_changed_only : bool, default=None
        # 控制是否只打印设置为非默认值的参数，例如在打印估计器时
        # 当 True 时，只打印 'SVC()'，而默认行为是打印包括所有非更改参数的完整字符串 'SVC(C=1.0, cache_size=200, ...)'
        .. versionadded:: 0.21

    display : {'text', 'diagram'}, default=None
        # 控制在 Jupyter lab 或 notebook 中以何种形式显示估计器的展示方式
        # 如果为 'diagram'，则以图表形式显示；如果为 'text'，则以文本形式显示，默认为 'diagram'
        .. versionadded:: 0.23

    pairwise_dist_chunk_size : int, default=None
        # 加速的成对距离计算后端的每个块中的行向量数量
        # 默认为 256，适用于大多数现代笔记本电脑的缓存和架构
        # 用于更轻松地对 scikit-learn 内部进行基准测试和测试
        # 不推荐最终用户自定义此配置设置
        .. versionadded:: 1.1

    enable_cython_pairwise_dist : bool, default=None
        # 在可能时使用加速的成对距离计算后端
        # 全局默认为 True
        # 用于更轻松地对 scikit-learn 内部进行基准测试和测试
        # 不推荐最终用户自定义此配置设置
        .. versionadded:: 1.1

    array_api_dispatch : bool, default=None
        # 当输入遵循 Array API 标准时，是否使用 Array API 调度
        # 默认为 False
        # 更多细节请参阅用户指南中的 Array API 部分
        .. versionadded:: 1.2

    transform_output : str, default=None
        # 配置 `transform` 和 `fit_transform` 的输出格式
        # 查看示例以了解如何使用 API 的详细信息
        # - `"default"`: 转换器的默认输出格式
        # - `"pandas"`: 返回 DataFrame 输出
        # - `"polars"`: 返回 Polars 输出
        # - `None`: Transform 配置保持不变
        .. versionadded:: 1.2
        .. versionadded:: 1.4
            添加了 `"polars"` 选项。

    enable_metadata_routing : bool, default=None
        # 启用元数据路由功能，默认情况下此功能已禁用
        # 更多细节请参阅元数据路由用户指南
        # - `True`: 启用元数据路由
        # - `False`: 禁用元数据路由，使用旧语法
        # - `None`: 配置保持不变
        .. versionadded:: 1.3
    """
    skip_parameter_validation : bool, default=None
        如果为 `True`，则在估计器的 `fit` 方法以及传递给公共辅助函数的参数中，禁用超参数类型和值的验证。
        这在某些情况下可以节省时间，但可能会导致底层崩溃和带有令人困惑的错误消息的异常。

        注意，对于数据参数（如 `X` 和 `y`），仅跳过类型验证，但将继续使用 `check_array` 进行验证。

        .. versionadded:: 1.3

    See Also
    --------
    config_context : 全局 scikit-learn 配置的上下文管理器。
    get_config : 检索全局配置的当前值。

    Examples
    --------
    >>> from sklearn import set_config
    >>> set_config(display='diagram')  # doctest: +SKIP
    """
    local_config = _get_threadlocal_config()  # 获取当前线程的本地配置

    if assume_finite is not None:
        local_config["assume_finite"] = assume_finite  # 设置 assume_finite 参数
    if working_memory is not None:
        local_config["working_memory"] = working_memory  # 设置 working_memory 参数
    if print_changed_only is not None:
        local_config["print_changed_only"] = print_changed_only  # 设置 print_changed_only 参数
    if display is not None:
        local_config["display"] = display  # 设置 display 参数
    if pairwise_dist_chunk_size is not None:
        local_config["pairwise_dist_chunk_size"] = pairwise_dist_chunk_size  # 设置 pairwise_dist_chunk_size 参数
    if enable_cython_pairwise_dist is not None:
        local_config["enable_cython_pairwise_dist"] = enable_cython_pairwise_dist  # 设置 enable_cython_pairwise_dist 参数
    if array_api_dispatch is not None:
        from .utils._array_api import _check_array_api_dispatch

        _check_array_api_dispatch(array_api_dispatch)  # 检查并设置 array_api_dispatch 参数
        local_config["array_api_dispatch"] = array_api_dispatch  # 设置 array_api_dispatch 参数
    if transform_output is not None:
        local_config["transform_output"] = transform_output  # 设置 transform_output 参数
    if enable_metadata_routing is not None:
        local_config["enable_metadata_routing"] = enable_metadata_routing  # 设置 enable_metadata_routing 参数
    if skip_parameter_validation is not None:
        local_config["skip_parameter_validation"] = skip_parameter_validation  # 设置 skip_parameter_validation 参数
# 定义一个上下文管理器，用于配置全局的 scikit-learn 设置
@contextmanager
def config_context(
    *,
    assume_finite=None,  # 控制是否跳过有限性验证，默认为 False
    working_memory=None,  # 控制临时数组的最大尺寸（单位为 MiB），默认为 1024
    print_changed_only=None,  # 控制是否只打印非默认参数的值，默认为 True
    display=None,  # 控制在 Jupyter 环境中如何显示估算器，默认为 'diagram'
    pairwise_dist_chunk_size=None,  # 控制加速的成对距离计算的行向量数，默认为 256
    enable_cython_pairwise_dist=None,  # 控制是否使用加速的成对距离计算后端，默认为 True
    array_api_dispatch=None,  # 未使用的参数，保留了 API 调度功能
    transform_output=None,  # 未使用的参数，保留了转换输出功能
    enable_metadata_routing=None,  # 未使用的参数，保留了元数据路由功能
    skip_parameter_validation=None,  # 未使用的参数，保留了跳过参数验证功能
):
    """Context manager for global scikit-learn configuration.

    Parameters
    ----------
    assume_finite : bool, default=None
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error. If None, the existing value won't change.
        The default value is False.

    working_memory : int, default=None
        If set, scikit-learn will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. If None, the existing value won't change.
        The default value is 1024.

    print_changed_only : bool, default=None
        If True, only the parameters that were set to non-default
        values will be printed when printing an estimator. For example,
        ``print(SVC())`` while True will only print 'SVC()', but would print
        'SVC(C=1.0, cache_size=200, ...)' with all the non-changed parameters
        when False. If None, the existing value won't change.
        The default value is True.

        .. versionchanged:: 0.23
           Default changed from False to True.

    display : {'text', 'diagram'}, default=None
        If 'diagram', estimators will be displayed as a diagram in a Jupyter
        lab or notebook context. If 'text', estimators will be displayed as
        text. If None, the existing value won't change.
        The default value is 'diagram'.

        .. versionadded:: 0.23

    pairwise_dist_chunk_size : int, default=None
        The number of row vectors per chunk for the accelerated pairwise-
        distances reduction backend. Default is 256 (suitable for most of
        modern laptops' caches and architectures).

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    enable_cython_pairwise_dist : bool, default=None
        Use the accelerated pairwise-distances reduction backend when
        possible. Global default: True.

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1
    """
    # 是否启用 Array API 分派，当输入遵循 Array API 标准时使用。默认为 False。
    # 详细信息请参阅用户指南中的 Array API 部分。
    # 版本 1.2 中新增功能。
    array_api_dispatch : bool, default=None

    # 配置 `transform` 和 `fit_transform` 的输出格式。
    # 查看示例以了解如何使用 API 的详细信息。
    # - `"default"`: 转换器的默认输出格式
    # - `"pandas"`: DataFrame 输出
    # - `"polars"`: Polars 输出
    # - `None`: 转换配置不变
    # 版本 1.2 和版本 1.4 中新增功能，添加了 `"polars"` 选项。
    transform_output : str, default=None

    # 启用元数据路由。默认情况下此功能已禁用。
    # 查阅元数据路由用户指南以获取更多详细信息。
    # - `True`: 启用元数据路由
    # - `False`: 禁用元数据路由，使用旧语法
    # - `None`: 配置不变
    # 版本 1.3 中新增功能。
    enable_metadata_routing : bool, default=None

    # 如果为 `True`，则禁用评估器的 `fit` 方法中超参数类型和值的验证，
    # 以及传递给公共辅助函数的参数验证。在某些情况下可以节省时间，
    # 但可能会导致低级崩溃和带有混淆错误消息的异常。
    # 注意，对于数据参数（如 `X` 和 `y`），仅跳过类型验证，
    # 但将继续运行 `check_array` 进行验证。
    # 版本 1.3 中新增功能。
    skip_parameter_validation : bool, default=None

    # 生成器函数，没有返回值。
    Yields
    ------
    None.

    See Also
    --------
    set_config : 设置全局 scikit-learn 配置。
    get_config : 获取当前全局配置的值。

    Notes
    -----
    所有设置，包括当前修改的设置，将在退出上下文管理器时恢复为其先前值。

    Examples
    --------
    >>> import sklearn
    >>> from sklearn.utils.validation import assert_all_finite
    >>> with sklearn.config_context(assume_finite=True):
    ...     assert_all_finite([float('nan')])
    >>> with sklearn.config_context(assume_finite=True):
    ...     with sklearn.config_context(assume_finite=False):
    ...         assert_all_finite([float('nan')])
    Traceback (most recent call last):
    ...
    ValueError: Input contains NaN...
    """
    # 获取当前配置的旧值，用于在退出时恢复配置。
    old_config = get_config()
    # 设置配置参数，使用给定的参数值来更新配置
    set_config(
        assume_finite=assume_finite,  # 控制是否假定输入数组不包含无穷大或 NaN 值
        working_memory=working_memory,  # 控制用于计算的内存限制
        print_changed_only=print_changed_only,  # 控制是否仅打印变更的数据
        display=display,  # 控制结果的显示方式
        pairwise_dist_chunk_size=pairwise_dist_chunk_size,  # 控制成对距离计算的块大小
        enable_cython_pairwise_dist=enable_cython_pairwise_dist,  # 控制是否启用 Cython 加速的成对距离计算
        array_api_dispatch=array_api_dispatch,  # 控制是否启用数组 API 的分派
        transform_output=transform_output,  # 控制输出是否进行转换
        enable_metadata_routing=enable_metadata_routing,  # 控制是否启用元数据路由
        skip_parameter_validation=skip_parameter_validation,  # 控制是否跳过参数验证
    )

    # 尝试执行 yield 语句块
    try:
        yield
    # 最终执行的步骤，恢复到旧的配置状态
    finally:
        # 恢复到之前保存的旧配置状态
        set_config(**old_config)
```