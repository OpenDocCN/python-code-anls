# `D:\src\scipysrc\pandas\pandas\core\computation\ops.py`

```
"""
Operator classes for eval.
"""

from __future__ import annotations  # 允许在类型提示中使用当前类的名称作为返回类型

from datetime import datetime  # 导入 datetime 模块中的 datetime 类
from functools import partial  # 导入 functools 模块中的 partial 函数
import operator  # 导入 operator 模块，用于操作符函数的集合
from typing import (  # 导入 typing 模块中的类型提示工具
    TYPE_CHECKING,  # 类型检查标志，用于检查循环导入
    Literal,  # 用于定义字面值类型的类型提示
)

import numpy as np  # 导入 NumPy 库并命名为 np

from pandas._libs.tslibs import Timestamp  # 导入 pandas 内部的时间戳对象

from pandas.core.dtypes.common import (  # 导入 pandas 核心模块中的通用数据类型检查函数
    is_list_like,  # 检查对象是否类列表
    is_numeric_dtype,  # 检查对象是否数值类型
    is_scalar,  # 检查对象是否标量
)

import pandas.core.common as com  # 导入 pandas 核心共用模块并命名为 com
from pandas.core.computation.common import (  # 导入 pandas 核心计算模块中的常用计算函数
    ensure_decoded,  # 确保对象解码
    result_type_many,  # 计算多个对象的结果类型
)
from pandas.core.computation.scope import DEFAULT_GLOBALS  # 导入 pandas 核心计算模块中的默认全局变量

from pandas.io.formats.printing import (  # 导入 pandas IO 模块中的打印格式化工具
    pprint_thing,  # 格式化打印对象
    pprint_thing_encoded,  # 编码后格式化打印对象
)

if TYPE_CHECKING:
    from collections.abc import (  # 导入 collections.abc 模块中的抽象集合类
        Callable,  # 可调用对象类型提示
        Iterable,  # 可迭代对象类型提示
        Iterator,  # 迭代器对象类型提示
    )

REDUCTIONS = ("sum", "prod", "min", "max")  # 定义一个包含聚合操作名的元组

_unary_math_ops = (  # 定义包含一元数学运算操作名的元组
    "sin", "cos", "tan", "exp", "log", "expm1", "log1p",
    "sqrt", "sinh", "cosh", "tanh", "arcsin", "arccos",
    "arctan", "arccosh", "arcsinh", "arctanh", "abs",
    "log10", "floor", "ceil",
)
_binary_math_ops = ("arctan2",)  # 定义包含二元数学运算操作名的元组

MATHOPS = _unary_math_ops + _binary_math_ops  # 将一元和二元数学运算操作名合并为一个列表

LOCAL_TAG = "__pd_eval_local_"  # 定义本地变量标签

class Term:  # 定义 Term 类
    def __new__(cls, name, env, side=None, encoding=None):
        klass = Constant if not isinstance(name, str) else cls  # 如果 name 不是 str 类型，则使用 Constant 类
        supr_new = super(Term, klass).__new__  # type: ignore[misc]
        return supr_new(klass)

    is_local: bool  # 声明 is_local 属性为布尔类型

    def __init__(self, name, env, side=None, encoding=None) -> None:
        self._name = name  # 设置实例变量 _name 为传入的 name 参数
        self.env = env  # 设置实例变量 env 为传入的 env 参数
        self.side = side  # 设置实例变量 side 为传入的 side 参数
        tname = str(name)  # 将 name 参数转换为字符串类型
        self.is_local = tname.startswith(LOCAL_TAG) or tname in DEFAULT_GLOBALS  # 判断是否是本地变量
        self._value = self._resolve_name()  # 解析变量名并设置实例变量 _value
        self.encoding = encoding  # 设置实例变量 encoding 为传入的 encoding 参数

    @property
    def local_name(self) -> str:
        return self.name.replace(LOCAL_TAG, "")  # 返回去除本地标签后的名称字符串

    def __repr__(self) -> str:
        return pprint_thing(self.name)  # 返回格式化后的名称字符串

    def __call__(self, *args, **kwargs):
        return self.value  # 调用实例返回其值

    def evaluate(self, *args, **kwargs) -> Term:
        return self  # 对象自身作为结果返回

    def _resolve_name(self):
        local_name = str(self.local_name)  # 获取去除本地标签后的名称字符串
        is_local = self.is_local  # 获取本地变量标志
        if local_name in self.env.scope and isinstance(
            self.env.scope[local_name], type
        ):
            is_local = False  # 如果在环境的作用域中且为类型对象，则不是本地变量

        res = self.env.resolve(local_name, is_local=is_local)  # 解析变量名
        self.update(res)  # 更新结果

        if hasattr(res, "ndim") and isinstance(res.ndim, int) and res.ndim > 2:
            raise NotImplementedError(
                "N-dimensional objects, where N > 2, are not supported with eval"
            )  # 如果结果是超过2维的对象，则抛出 NotImplementedError

        return res  # 返回解析后的结果对象
    def update(self, value) -> None:
        """
        Update method for assigning a new value to the object.

        search order for local (i.e., @variable) variables:

        scope, key_variable
        [('locals', 'local_name'),
         ('globals', 'local_name'),
         ('locals', 'key'),
         ('globals', 'key')]
        """
        # 获取对象的名称作为键值
        key = self.name

        # 如果键是一个字符串，则将新值更新到环境中
        if isinstance(key, str):
            self.env.swapkey(self.local_name, key, new_value=value)

        # 更新对象的值
        self.value = value

    @property
    def is_scalar(self) -> bool:
        """
        Property that checks if the value is a scalar.

        Returns:
            bool: True if the value is scalar, False otherwise.
        """
        return is_scalar(self._value)

    @property
    def type(self):
        """
        Property that retrieves the type of the stored value.

        Returns:
            type: Type of the stored value.
        """
        try:
            # 检索值的数据类型（可能对于大型、混合数据类型的帧非常慢）
            return self._value.values.dtype
        except AttributeError:
            try:
                # ndarray 的数据类型
                return self._value.dtype
            except AttributeError:
                # 标量的类型
                return type(self._value)

    return_type = type

    @property
    def raw(self) -> str:
        """
        Property that returns a string representation of the object.

        Returns:
            str: String representation containing object's name and type.
        """
        return f"{type(self).__name__}(name={self.name!r}, type={self.type})"

    @property
    def is_datetime(self) -> bool:
        """
        Property that checks if the stored value type is a datetime.

        Returns:
            bool: True if the stored value is a datetime, False otherwise.
        """
        try:
            t = self.type.type
        except AttributeError:
            t = self.type

        return issubclass(t, (datetime, np.datetime64))

    @property
    def value(self):
        """
        Property that retrieves the stored value.

        Returns:
            Any: Stored value.
        """
        return self._value

    @value.setter
    def value(self, new_value) -> None:
        """
        Setter method for assigning a new value to the object.

        Args:
            new_value (Any): New value to be assigned.
        """
        self._value = new_value

    @property
    def name(self):
        """
        Property that retrieves the name of the object.

        Returns:
            str: Name of the object.
        """
        return self._name

    @property
    def ndim(self) -> int:
        """
        Property that retrieves the number of dimensions of the stored value.

        Returns:
            int: Number of dimensions.
        """
        return self._value.ndim
class Constant(Term):
    # 继承自 Term 类的常量类
    def _resolve_name(self):
        # 返回常量的名称
        return self._name

    @property
    def name(self):
        # 返回常量的值作为名称
        return self.value

    def __repr__(self) -> str:
        # 返回常量的值的规范表示形式
        # 在 Python 2 中，float 的 str() 可能会比 repr() 更短
        return repr(self.name)


_bool_op_map = {"not": "~", "and": "&", "or": "|"}
# 布尔操作符到其对应 Python 运算符的映射字典


class Op:
    """
    Hold an operator of arbitrary arity.
    """
    # 表示一个任意元数操作符

    op: str

    def __init__(self, op: str, operands: Iterable[Term | Op], encoding=None) -> None:
        # 初始化操作符实例
        self.op = _bool_op_map.get(op, op)
        # 根据操作符从映射字典中获取对应的 Python 运算符，如果不存在则使用原操作符
        self.operands = operands
        # 设置操作符的操作数
        self.encoding = encoding
        # 设置操作符的编码方式

    def __iter__(self) -> Iterator:
        # 返回操作符的迭代器
        return iter(self.operands)

    def __repr__(self) -> str:
        """
        Print a generic n-ary operator and its operands using infix notation.
        """
        # 返回操作符及其操作数的通用 n-元操作符的中缀表示形式
        # 递归处理操作数
        parened = (f"({pprint_thing(opr)})" for opr in self.operands)
        return pprint_thing(f" {self.op} ".join(parened))

    @property
    def return_type(self):
        # 返回操作符的返回类型
        # 如果操作符是布尔运算符，则将类型强制为 np.bool_
        if self.op in (CMP_OPS_SYMS + BOOL_OPS_SYMS):
            return np.bool_
        return result_type_many(*(term.type for term in com.flatten(self)))

    @property
    def has_invalid_return_type(self) -> bool:
        # 检查操作符的返回类型是否无效
        types = self.operand_types
        obj_dtype_set = frozenset([np.dtype("object")])
        return self.return_type == object and types - obj_dtype_set

    @property
    def operand_types(self):
        # 返回操作符的操作数的类型集合
        return frozenset(term.type for term in com.flatten(self))

    @property
    def is_scalar(self) -> bool:
        # 检查操作符的所有操作数是否为标量
        return all(operand.is_scalar for operand in self.operands)

    @property
    def is_datetime(self) -> bool:
        # 检查操作符的返回类型是否为日期时间类型
        try:
            t = self.return_type.type
        except AttributeError:
            t = self.return_type

        return issubclass(t, (datetime, np.datetime64))


def _in(x, y):
    """
    Compute the vectorized membership of ``x in y`` if possible, otherwise
    use Python.
    """
    # 计算 ``x in y`` 的向量化成员关系，如果不可行则使用 Python 的方法
    try:
        return x.isin(y)
    except AttributeError:
        if is_list_like(x):
            try:
                return y.isin(x)
            except AttributeError:
                pass
        return x in y


def _not_in(x, y):
    """
    Compute the vectorized membership of ``x not in y`` if possible,
    otherwise use Python.
    """
    # 计算 ``x not in y`` 的向量化成员关系，如果不可行则使用 Python 的方法
    try:
        return ~x.isin(y)
    except AttributeError:
        if is_list_like(x):
            try:
                return ~y.isin(x)
            except AttributeError:
                pass
        return x not in y


CMP_OPS_SYMS = (">", "<", ">=", "<=", "==", "!=", "in", "not in")
# 比较操作符的符号集合
_cmp_ops_funcs = (
    operator.gt,
    operator.lt,
    operator.ge,
    operator.le,
    operator.eq,
    operator.ne,
    _in,
    _not_in,
)
# 比较操作符对应的函数集合
_cmp_ops_dict = dict(zip(CMP_OPS_SYMS, _cmp_ops_funcs))
# 比较操作符与其对应函数的映射字典

BOOL_OPS_SYMS = ("&", "|", "and", "or")
# 布尔操作符的符号集合
_bool_ops_funcs = (operator.and_, operator.or_, operator.and_, operator.or_)
# 布尔操作符对应的函数集合
# 创建一个从布尔运算符到相应函数的字典
_bool_ops_dict = dict(zip(BOOL_OPS_SYMS, _bool_ops_funcs))

# 定义包含所有算术运算符的元组
ARITH_OPS_SYMS = ("+", "-", "*", "/", "**", "//", "%")
# 创建一个从算术运算符到相应函数的字典
_arith_ops_funcs = (
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.pow,
    operator.floordiv,
    operator.mod,
)
_arith_ops_dict = dict(zip(ARITH_OPS_SYMS, _arith_ops_funcs))

# 创建一个空字典，用于存储所有二元运算符的函数映射
_binary_ops_dict = {}

# 将比较运算符、布尔运算符和算术运算符的字典依次合并到_binary_ops_dict中
for d in (_cmp_ops_dict, _bool_ops_dict, _arith_ops_dict):
    _binary_ops_dict.update(d)


def _cast_inplace(terms, acceptable_dtypes, dtype) -> None:
    """
    在原地对表达式进行类型转换。

    Parameters
    ----------
    terms : Op
        需要进行类型转换的表达式。
    acceptable_dtypes : list of acceptable numpy.dtype
        若term的dtype在此列表中，则不进行类型转换。
    dtype : str or numpy.dtype
        要转换的目标dtype。
    """
    dt = np.dtype(dtype)
    for term in terms:
        if term.type in acceptable_dtypes:
            continue

        try:
            new_value = term.value.astype(dt)
        except AttributeError:
            new_value = dt.type(term.value)
        term.update(new_value)


def is_term(obj) -> bool:
    """
    判断对象是否为Term的实例。

    Parameters
    ----------
    obj : any
        要检查的对象。

    Returns
    -------
    bool
        如果obj是Term的实例则返回True，否则返回False。
    """
    return isinstance(obj, Term)


class BinOp(Op):
    """
    表示一个二元运算符及其操作数。

    Parameters
    ----------
    op : str
        运算符字符串。
    lhs : Term or Op
        左操作数。
    rhs : Term or Op
        右操作数。
    """

    def __init__(self, op: str, lhs, rhs) -> None:
        super().__init__(op, (lhs, rhs))
        self.lhs = lhs
        self.rhs = rhs

        # 检查是否禁止使用仅标量的布尔运算符
        self._disallow_scalar_only_bool_ops()

        # 将操作数的值转换为合适的类型
        self.convert_values()

        try:
            # 根据运算符获取相应的函数
            self.func = _binary_ops_dict[op]
        except KeyError as err:
            # 如果运算符不在字典中，抛出错误并指出有效的运算符列表
            keys = list(_binary_ops_dict.keys())
            raise ValueError(
                f"Invalid binary operator {op!r}, valid operators are {keys}"
            ) from err

    def __call__(self, env):
        """
        递归地在Python环境中评估表达式。

        Parameters
        ----------
        env : Scope
            表示表达式的环境。

        Returns
        -------
        object
            表达式评估的结果。
        """
        # 递归地评估左右节点
        left = self.lhs(env)
        right = self.rhs(env)

        # 调用相应的函数来执行二元操作
        return self.func(left, right)
    def evaluate(self, env, engine: str, parser, term_type, eval_in_python):
        """
        Evaluate a binary operation *before* being passed to the engine.

        Parameters
        ----------
        env : Scope
            Scope object containing variables and their values
        engine : str
            Name of the engine to use for evaluation
        parser : str
            Parser to use for parsing expressions
        term_type : type
            Type of the term being evaluated
        eval_in_python : list
            List of operations to evaluate in Python

        Returns
        -------
        term_type
            The "pre-evaluated" expression as an instance of ``term_type``
        """
        if engine == "python":
            # If the engine is Python, directly evaluate the expression
            res = self(env)
        else:
            # If the engine is not Python, recursively evaluate the left and right nodes

            left = self.lhs.evaluate(
                env,
                engine=engine,
                parser=parser,
                term_type=term_type,
                eval_in_python=eval_in_python,
            )

            right = self.rhs.evaluate(
                env,
                engine=engine,
                parser=parser,
                term_type=term_type,
                eval_in_python=eval_in_python,
            )

            # Handle base cases for evaluation
            if self.op in eval_in_python:
                res = self.func(left.value, right.value)
            else:
                from pandas.core.computation.eval import eval

                res = eval(self, local_dict=env, engine=engine, parser=parser)

        # Add the result to the environment and return the term
        name = env.add_tmp(res)
        return term_type(name, env=env)

    def convert_values(self) -> None:
        """
        Convert datetimes to a comparable value in an expression.
        """

        def stringify(value):
            encoder: Callable
            if self.encoding is not None:
                encoder = partial(pprint_thing_encoded, encoding=self.encoding)
            else:
                encoder = pprint_thing
            return encoder(value)

        lhs, rhs = self.lhs, self.rhs

        if is_term(lhs) and lhs.is_datetime and is_term(rhs) and rhs.is_scalar:
            v = rhs.value
            if isinstance(v, (int, float)):
                v = stringify(v)
            v = Timestamp(ensure_decoded(v))
            if v.tz is not None:
                v = v.tz_convert("UTC")
            self.rhs.update(v)

        if is_term(rhs) and rhs.is_datetime and is_term(lhs) and lhs.is_scalar:
            v = lhs.value
            if isinstance(v, (int, float)):
                v = stringify(v)
            v = Timestamp(ensure_decoded(v))
            if v.tz is not None:
                v = v.tz_convert("UTC")
            self.lhs.update(v)
    def _disallow_scalar_only_bool_ops(self) -> None:
        # 将右操作数和左操作数分别赋值给rhs和lhs变量
        rhs = self.rhs
        lhs = self.lhs

        # GH#24883 如果需要，解封rhs的dtype以确保得到一个类型对象
        rhs_rt = rhs.return_type
        rhs_rt = getattr(rhs_rt, "type", rhs_rt)
        lhs_rt = lhs.return_type
        lhs_rt = getattr(lhs_rt, "type", lhs_rt)
        
        # 如果左操作数或右操作数是标量，并且操作符存在于布尔运算字典中，并且
        # rhs_rt和lhs_rt不是bool或np.bool_的子类时，抛出NotImplementedError异常
        if (
            (lhs.is_scalar or rhs.is_scalar)
            and self.op in _bool_ops_dict
            and (
                not (
                    issubclass(rhs_rt, (bool, np.bool_))
                    and issubclass(lhs_rt, (bool, np.bool_))
                )
            )
        ):
            raise NotImplementedError("cannot evaluate scalar only bool ops")
class Div(BinOp):
    """
    Div operator to special case casting.

    Parameters
    ----------
    lhs, rhs : Term or Op
        The Terms or Ops in the ``/`` expression.
    """

    def __init__(self, lhs, rhs) -> None:
        super().__init__("/", lhs, rhs)

        # 检查操作数类型是否为数值类型，如果不是则引发类型错误
        if not is_numeric_dtype(lhs.return_type) or not is_numeric_dtype(
            rhs.return_type
        ):
            raise TypeError(
                f"unsupported operand type(s) for {self.op}: "
                f"'{lhs.return_type}' and '{rhs.return_type}'"
            )

        # 避免将 float32 强制转换为不必要的 float64
        acceptable_dtypes = [np.float32, np.float64]
        _cast_inplace(com.flatten(self), acceptable_dtypes, np.float64)


UNARY_OPS_SYMS = ("+", "-", "~", "not")
_unary_ops_funcs = (operator.pos, operator.neg, operator.invert, operator.invert)
_unary_ops_dict = dict(zip(UNARY_OPS_SYMS, _unary_ops_funcs))


class UnaryOp(Op):
    """
    Hold a unary operator and its operands.

    Parameters
    ----------
    op : str
        The token used to represent the operator.
    operand : Term or Op
        The Term or Op operand to the operator.

    Raises
    ------
    ValueError
        * If no function associated with the passed operator token is found.
    """

    def __init__(self, op: Literal["+", "-", "~", "not"], operand) -> None:
        super().__init__(op, (operand,))
        self.operand = operand

        try:
            # 查找与操作符对应的函数
            self.func = _unary_ops_dict[op]
        except KeyError as err:
            raise ValueError(
                f"Invalid unary operator {op!r}, "
                f"valid operators are {UNARY_OPS_SYMS}"
            ) from err

    def __call__(self, env) -> MathCall:
        # 获取操作数的值
        operand = self.operand(env)
        # 返回应用操作符函数后的结果
        return self.func(operand)  # type: ignore[operator]

    def __repr__(self) -> str:
        # 返回表示对象的字符串，包括操作符和操作数
        return pprint_thing(f"{self.op}({self.operand})")

    @property
    def return_type(self) -> np.dtype:
        operand = self.operand
        # 如果操作数的返回类型是布尔型，则返回布尔型
        if operand.return_type == np.dtype("bool"):
            return np.dtype("bool")
        # 如果操作数是 Op 类型且其操作符在比较或布尔操作符字典中，则返回布尔型
        if isinstance(operand, Op) and (
            operand.op in _cmp_ops_dict or operand.op in _bool_ops_dict
        ):
            return np.dtype("bool")
        # 默认返回整型
        return np.dtype("int")


class MathCall(Op):
    def __init__(self, func, args) -> None:
        super().__init__(func.name, args)
        self.func = func

    def __call__(self, env):
        # 获取每个操作数在当前环境下的值
        operands = [op(env) for op in self.operands]  # type: ignore[operator]
        # 调用函数并返回结果
        return self.func.func(*operands)

    def __repr__(self) -> str:
        # 返回表示对象的字符串，包括操作符和操作数
        operands = map(str, self.operands)
        return pprint_thing(f"{self.op}({','.join(operands)})")


class FuncNode:
    # FuncNode 类暂无具体实现，待添加
    pass
    # 初始化方法，创建一个 MathFunc 对象
    def __init__(self, name: str) -> None:
        # 检查给定的函数名是否在 MATHOPS 中
        if name not in MATHOPS:
            # 如果不在支持的函数列表中，则抛出值错误异常
            raise ValueError(f'"{name}" is not a supported function')
        # 将函数名保存到对象的属性中
        self.name = name
        # 使用 getattr 函数从 numpy 模块中获取相应的函数对象，并保存到对象的属性中
        self.func = getattr(np, name)

    # 对象调用方法，用于生成 MathCall 对象
    def __call__(self, *args) -> MathCall:
        # 返回一个 MathCall 对象，传递当前对象及其参数
        return MathCall(self, args)
```