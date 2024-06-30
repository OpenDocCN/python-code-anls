# `D:\src\scipysrc\sympy\sympy\matrices\utilities.py`

```
# 导入必要的模块
from contextlib import contextmanager
from threading import local
# 导入 sympy 的函数 expand_mul
from sympy.core.function import expand_mul

# 定义 DotProdSimpState 类，继承自 threading.local
class DotProdSimpState(local):
    def __init__(self):
        self.state = None

# 创建 DotProdSimpState 类的实例 _dotprodsimp_state
_dotprodsimp_state = DotProdSimpState()

# 定义上下文管理器 dotprodsimp，用于管理 _dotprodsimp_state 的状态
@contextmanager
def dotprodsimp(x):
    # 保存当前的 _dotprodsimp_state.state 状态
    old = _dotprodsimp_state.state
    
    try:
        # 将 _dotprodsimp_state.state 设置为 x
        _dotprodsimp_state.state = x
        # 执行 yield 语句块
        yield
    finally:
        # 恢复 _dotprodsimp_state.state 到旧的状态
        _dotprodsimp_state.state = old

# 定义 _dotprodsimp 函数，作为 simplify.dotprodsimp 的包装，避免循环导入问题
def _dotprodsimp(expr, withsimp=False):
    """Wrapper for simplify.dotprodsimp to avoid circular imports."""
    # 导入 dotprodsimp 函数
    from sympy.simplify.simplify import dotprodsimp as dps
    # 调用 dotprodsimp 函数进行简化操作
    return dps(expr, withsimp=withsimp)

# 定义 _get_intermediate_simp 函数，控制中间简化过程的支持函数
def _get_intermediate_simp(deffunc=lambda x: x, offfunc=lambda x: x,
        onfunc=_dotprodsimp, dotprodsimp=None):
    """Support function for controlling intermediate simplification. Returns a
    simplification function according to the global setting of dotprodsimp
    operation.

    ``deffunc``     - Function to be used by default.
    ``offfunc``     - Function to be used if dotprodsimp has been turned off.
    ``onfunc``      - Function to be used if dotprodsimp has been turned on.
    ``dotprodsimp`` - True, False or None. Will be overridden by global
                      _dotprodsimp_state.state if that is not None.
    """
    # 如果 dotprodsimp 显式为 False 或者 _dotprodsimp_state.state 为 False，则返回 offfunc 函数
    if dotprodsimp is False or _dotprodsimp_state.state is False:
        return offfunc
    # 如果 dotprodsimp 显式为 True 或者 _dotprodsimp_state.state 为 True，则返回 onfunc 函数
    if dotprodsimp is True or _dotprodsimp_state.state is True:
        return onfunc
    
    # 否则返回默认的 deffunc 函数
    return deffunc

# 定义 _get_intermediate_simp_bool 函数，与 _get_intermediate_simp 类似，但返回布尔值而非函数
def _get_intermediate_simp_bool(default=False, dotprodsimp=None):
    """Same as ``_get_intermediate_simp`` but returns bools instead of functions
    by default."""
    # 调用 _get_intermediate_simp 函数，返回相应的布尔值
    return _get_intermediate_simp(default, False, True, dotprodsimp)

# 定义 _iszero 函数，检查 x 是否为零
def _iszero(x):
    """Returns True if x is zero."""
    return getattr(x, 'is_zero', None)

# 定义 _is_zero_after_expand_mul 函数，通过 expand_mul 测试是否为零，适用于多项式和有理函数
def _is_zero_after_expand_mul(x):
    """Tests by expand_mul only, suitable for polynomials and rational
    functions."""
    # 判断经过 expand_mul 处理后的 x 是否等于零
    return expand_mul(x) == 0

# 定义 _simplify 函数，作为 simplify 函数的包装，避免循环导入问题
def _simplify(expr):
    """ Wrapper to avoid circular imports. """
    # 导入 simplify 函数
    from sympy.simplify.simplify import simplify
    # 调用 simplify 函数对表达式进行简化
    return simplify(expr)
```