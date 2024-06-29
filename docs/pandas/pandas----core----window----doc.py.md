# `D:\src\scipysrc\pandas\pandas\core\window\doc.py`

```
# 导入未来的注解功能
from __future__ import annotations

# 导入文本格式化工具
from textwrap import dedent

# 从 pandas 核心共享文档中导入共享文档
from pandas.core.shared_docs import _shared_docs

# 创建 _shared_docs 的副本
_shared_docs = dict(**_shared_docs)

# 定义一个函数，创建 numpydoc 部分的标题
def create_section_header(header: str) -> str:
    """Create numpydoc section header"""
    return f"{header}\n{'-' * len(header)}\n"

# 模板化的标题，用于描述窗口方法和聚合方法的计算
template_header = "\nCalculate the {window_method} {aggregation_description}.\n\n"

# 模板化的返回值部分说明
template_returns = dedent(
    """
    Series or DataFrame
        Return type is the same as the original object with ``np.float64`` dtype.\n
    """
).replace("\n", "", 1)

# 模板化的参见部分说明
template_see_also = dedent(
    """
    Series.{window_method} : Calling {window_method} with Series data.
    DataFrame.{window_method} : Calling {window_method} with DataFrames.
    Series.{agg_method} : Aggregating {agg_method} for Series.
    DataFrame.{agg_method} : Aggregating {agg_method} for DataFrame.\n
    """
).replace("\n", "", 1)

# 模板化的 numeric_only 参数说明
kwargs_numeric_only = dedent(
    """
    numeric_only : bool, default False
        Include only float, int, boolean columns.

        .. versionadded:: 1.5.0\n
    """
).replace("\n", "", 1)

# 模板化的 scipy 参数说明
kwargs_scipy = dedent(
    """
    **kwargs
        Keyword arguments to configure the ``SciPy`` weighted window type.\n
    """
).replace("\n", "", 1)

# 模板化的窗口应用参数说明
window_apply_parameters = dedent(
    """
    func : function
        Must produce a single value from an ndarray input if ``raw=True``
        or a single value from a Series if ``raw=False``. Can also accept a
        Numba JIT function with ``engine='numba'`` specified.

    raw : bool, default False
        * ``False`` : passes each row or column as a Series to the
          function.
        * ``True`` : the passed function will receive ndarray
          objects instead.
          If you are just applying a NumPy reduction function this will
          achieve much better performance.

    engine : str, default None
        * ``'cython'`` : Runs rolling apply through C-extensions from cython.
        * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
          Only available when ``raw`` is set to ``True``.
        * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

    engine_kwargs : dict, default None
        * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
        * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
          and ``parallel`` dictionary keys. The values must either be ``True`` or
          ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
          ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
          applied to both the ``func`` and the ``apply`` rolling aggregation.

    args : tuple, default None
        Positional arguments to be passed into func.

    kwargs : dict, default None
        Keyword arguments to be passed into func.\n
    """
).replace("\n", "", 1)

# Numba 的注意事项说明
numba_notes = (
    # 创建一个字符串，包含链接到文档中两个特定部分的引用
    "See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for "
    # 提供额外文档和性能注意事项，主要针对 Numba 引擎
    "extended documentation and performance considerations for the Numba engine.\n\n"
# 定义一个函数 `window_agg_numba_parameters`，接受一个字符串类型参数 `version`，返回一个格式化后的字符串
def window_agg_numba_parameters(version: str = "1.3") -> str:
    # 返回一个格式化的多行字符串，描述了函数支持的引擎选择和参数设置
    return (
        dedent(
            """
    engine : str, default None
        * ``'cython'`` : Runs the operation through C-extensions from cython.
        * ``'numba'`` : Runs the operation through JIT compiled code from numba.
        * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

          .. versionadded:: {version}.0

    engine_kwargs : dict, default None
        * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
        * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
          and ``parallel`` dictionary keys. The values must either be ``True`` or
          ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
          ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

          .. versionadded:: {version}.0\n
    """
        )
        .replace("\n", "", 1)  # 替换第一个换行符为空字符串，保持字符串的格式化
        .replace("{version}", version)  # 将字符串中的 `{version}` 替换为传入的版本号参数
    )
```