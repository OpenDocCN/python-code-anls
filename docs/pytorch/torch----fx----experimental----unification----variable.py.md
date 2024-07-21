# `.\pytorch\torch\fx\experimental\unification\variable.py`

```py
# mypy: allow-untyped-defs
# 导入上下文管理器模块
from contextlib import contextmanager
# 从本地模块导入 hashable 函数
from .utils import hashable
# 从本地模块导入 dispatch 函数
from .dispatch import dispatch

# 全局逻辑变量集合，初始化为空集合
_global_logic_variables = set()  # type: ignore[var-annotated]
# 简写 _glv 作为 _global_logic_variables 的别名
_glv = _global_logic_variables


class Var:
    """ Logic Variable """

    _id = 1

    # 定义 Var 类的构造函数
    def __new__(cls, *token):
        # 如果参数列表为空，则使用默认命名规则
        if len(token) == 0:
            token = f"_{Var._id}"  # type: ignore[assignment]
            Var._id += 1
        # 如果参数列表长度为1，则直接使用传入的 token
        elif len(token) == 1:
            token = token[0]

        # 调用父类的构造函数创建对象
        obj = object.__new__(cls)
        # 设置对象的 token 属性
        obj.token = token  # type: ignore[attr-defined]
        return obj

    # 定义对象的字符串表示形式
    def __str__(self):
        return "~" + str(self.token)  # type: ignore[attr-defined]
    # 将 __str__ 方法作为 __repr__ 方法的别名
    __repr__ = __str__

    # 定义对象的相等比较方法
    def __eq__(self, other):
        # 类型和 token 值相同时返回 True
        return type(self) == type(other) and self.token == other.token  # type: ignore[attr-defined]

    # 定义对象的哈希方法
    def __hash__(self):
        # 返回类型和 token 值的哈希值
        return hash((type(self), self.token))  # type: ignore[attr-defined]


# 定义 var 函数，返回一个 lambda 函数用于创建 Var 对象
def var():
    return lambda *args: Var(*args)


# 定义 vars 函数，返回一个 lambda 函数用于创建多个 Var 对象的列表
def vars():
    return lambda n: [var() for i in range(n)]


# 根据 dispatch 装饰器，定义 isvar 函数处理 Var 类型参数的情况
@dispatch(Var)
def isvar(v):
    return True

# isvar 函数本身


# 根据 dispatch 装饰器，定义 isvar 函数处理一般对象的情况
@dispatch(object)  # type: ignore[no-redef]
def isvar(o):
    # 返回条件判断结果，判断是否为逻辑变量
    return not not _glv and hashable(o) and o in _glv


# 定义上下文管理器 variables，用于管理逻辑变量
@contextmanager
def variables(*variables):
    """
    Context manager for logic variables

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> from __future__ import with_statement
        >>> with variables(1):
        ...     print(isvar(1))
        True
        >>> print(isvar(1))
        False
        >>> # Normal approach
        >>> from unification import unify
        >>> x = var('x')
        >>> unify(x, 1)
        {~x: 1}
        >>> # Context Manager approach
        >>> with variables('x'):
        ...     print(unify('x', 1))
        {'x': 1}
    """
    # 备份全局逻辑变量集合
    old_global_logic_variables = _global_logic_variables.copy()
    # 更新全局逻辑变量集合，添加新的变量
    _global_logic_variables.update(set(variables))
    try:
        yield
    finally:
        # 清空当前逻辑变量集合
        _global_logic_variables.clear()
        # 恢复全局逻辑变量集合为原始状态
        _global_logic_variables.update(old_global_logic_variables)
```