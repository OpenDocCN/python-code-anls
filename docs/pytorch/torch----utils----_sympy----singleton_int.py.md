# `.\pytorch\torch\utils\_sympy\singleton_int.py`

```py
# mypy: allow-untyped-defs
# 导入 sympy 库
import sympy
# 从 sympy.multipledispatch 模块中导入 dispatch 函数
from sympy.multipledispatch import dispatch

# 定义仅公开 SingletonInt 类的列表
__all__ = ["SingletonInt"]

# 定义 SingletonInt 类，继承自 sympy.AtomicExpr
class SingletonInt(sympy.AtomicExpr):
    # 运算优先级，用于表达式计算的优先级设定
    _op_priority = 99999

    # 创建新的 SingletonInt 实例
    def __new__(cls, *args, coeff=None, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    # 初始化方法，用于设定实例的初始值和系数
    def __init__(self, val, *, coeff=1):
        self._val = val
        self._coeff = coeff
        super().__init__()

    # 重载相等比较方法，用于判断两个 SingletonInt 实例是否相等
    def _eval_Eq(self, other):
        if (
            isinstance(other, SingletonInt)
            and other._val == self._val
            and self._coeff == other._coeff
        ):
            return sympy.true
        else:
            return sympy.false

    # 返回空集合，用于在表达式中包含该 SingletonInt 实例时避免错误
    @property
    def free_symbols(self):
        return set()

    # 重载乘法操作符，用于实现 SingletonInt 与数值的乘法
    def __mul__(self, other):
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    # 右侧乘法操作的重载，与 __mul__ 方法类似
    def __rmul__(self, other):
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    # 以下运算符的重载方法，暂时抛出 NotImplementedError
    def __add__(self, other):
        raise NotImplementedError("NYI")

    def __sub__(self, other):
        raise NotImplementedError("NYI")

    def __truediv__(self, other):
        raise NotImplementedError("NYI")

    def __floordiv__(self, other):
        raise NotImplementedError("NYI")

    def __mod__(self, other):
        raise NotImplementedError("NYI")


# 用于处理 SingletonInt 和 sympy.Integer 之间的比较关系，根据具体情况返回 true 或抛出错误
@dispatch(sympy.Integer, SingletonInt)
def _eval_is_ge(a, b):
    if a < 2:
        return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")

# 用于处理 SingletonInt 和 sympy.Integer 之间的比较关系，根据具体情况返回 true 或抛出错误
@dispatch(SingletonInt, sympy.Integer)  # type: ignore[no-redef]
def _eval_is_ge(a, b):  # noqa: F811
    if b <= 2:
        return sympy.true
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")

# 用于处理两个 SingletonInt 实例之间的比较关系，根据具体情况返回 true 或抛出错误
@dispatch(SingletonInt, SingletonInt)  # type: ignore[no-redef]
def _eval_is_ge(a, b):  # noqa: F811
    if a._val == b._val:
        if a._coeff >= b._coeff:
            return sympy.true
        else:
            return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")
```