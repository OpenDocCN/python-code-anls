# `D:\src\scipysrc\scikit-learn\sklearn\utils\_missing.py`

```
# 导入数学库
import math
# 导入数字类型模块
import numbers
# 导入上下文管理工具中的异常抑制功能
from contextlib import suppress

# 定义函数，用于检测是否为 NaN
def is_scalar_nan(x):
    """Test if x is NaN.

    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not float('nan').

    Parameters
    ----------
    x : any type
        Any scalar value.

    Returns
    -------
    bool
        Returns true if x is NaN, and false otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils._missing import is_scalar_nan
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    # 如果 x 不是整数且是实数，并且是 NaN，则返回 True
    return (
        not isinstance(x, numbers.Integral)
        and isinstance(x, numbers.Real)
        and math.isnan(x)
    )

# 定义函数，用于检测是否为 Pandas 的 NA 值
def is_pandas_na(x):
    """Test if x is pandas.NA.

    We intentionally do not use this function to return `True` for `pd.NA` in
    `is_scalar_nan`, because estimators that support `pd.NA` are the exception
    rather than the rule at the moment. When `pd.NA` is more universally
    supported, we may reconsider this decision.

    Parameters
    ----------
    x : any type

    Returns
    -------
    boolean
    """
    # 尝试导入 Pandas 的 NA 值
    with suppress(ImportError):
        from pandas import NA
        # 如果 x 和 NA 相同，则返回 True
        return x is NA
    
    # 如果未成功导入 Pandas 或 x 不是 NA，则返回 False
    return False
```