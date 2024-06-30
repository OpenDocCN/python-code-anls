# `D:\src\scipysrc\sympy\sympy\polys\matrices\dfm.py`

```
"""
sympy.polys.matrices.dfm

Provides the :class:`DFM` class if ``GROUND_TYPES=flint'``. Otherwise, ``DFM``
is a placeholder class that raises NotImplementedError when instantiated.
"""

# 从 sympy.external.gmpy 导入 GROUND_TYPES
from sympy.external.gmpy import GROUND_TYPES

# 检查 GROUND_TYPES 是否为 "flint"
if GROUND_TYPES == "flint":  # pragma: no cover
    # 当 python-flint 安装时，尝试使用它来处理稠密矩阵
    # 如果域由 python-flint 支持的话。
    from ._dfm import DFM

else: # pragma: no cover
    # 其他代码应该能导入此处，并且它应该表现为一个不支持任何域的版本的 DFM。
    class DFM_dummy:
        """
        Placeholder class for DFM when python-flint is not installed.
        """
        # 初始化函数，抛出 NotImplementedError 异常
        def __init__(*args, **kwargs):
            raise NotImplementedError("DFM requires GROUND_TYPES=flint.")

        # 类方法：检查是否支持指定域
        @classmethod
        def _supports_domain(cls, domain):
            return False

        # 类方法：获取 flint 函数，抛出 NotImplementedError 异常
        @classmethod
        def _get_flint_func(cls, domain):
            raise NotImplementedError("DFM requires GROUND_TYPES=flint.")

    # mypy 对这种条件类型分配的处理确实有些困难。
    # 或许有更好的方式来注释此处，而不是使用 type: ignore。
    # 将 DFM_dummy 赋值给 DFM，用作类型注释
    DFM = DFM_dummy # type: ignore
```