# `D:\src\scipysrc\scikit-learn\sklearn\utils\_param_validation.py`

```
# 导入 functools 模块，用于高阶函数操作
# 导入 math 模块，提供数学运算函数
# 导入 operator 模块，提供内置运算符的函数形式
# 导入 re 模块，提供正则表达式匹配操作
# 导入 ABC 和 abstractmethod 类，用于定义抽象基类和抽象方法
# 导入 Iterable 类，用于检查对象是否可迭代
# 导入 signature 函数，用于获取函数的参数签名
# 导入 Integral 和 Real 类，用于检查数值类型

import functools
import math
import operator
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from inspect import signature
from numbers import Integral, Real

# 导入 numpy 库，重命名为 np
# 导入 scipy.sparse 中的 csr_matrix 和 issparse 函数
# 导入上层目录中的 _config 模块的 config_context 和 get_config 函数
# 导入当前目录中的 validation 模块的 _is_arraylike_not_scalar 函数

import numpy as np
from scipy.sparse import csr_matrix, issparse
from .._config import config_context, get_config
from .validation import _is_arraylike_not_scalar


class InvalidParameterError(ValueError, TypeError):
    """自定义异常类，用于在类/方法/函数的参数类型或值无效时引发异常。"""
    # 继承自 ValueError 和 TypeError，以保持向后兼容性。


def validate_parameter_constraints(parameter_constraints, params, caller_name):
    """验证给定参数的类型和值是否符合预期。

    Parameters
    ----------
    parameter_constraints : dict or {"no_validation"}
        如果为 "no_validation"，则跳过对此参数的验证。

        如果为 dict，则必须是 `param_name: constraints_list` 的形式。
        参数将满足列表中任一约束条件才算有效。
        约束条件可以是：
        - 一个 Interval 对象，表示连续或离散的数字范围
        - 字符串 "array-like"
        - 字符串 "sparse matrix"
        - 字符串 "random_state"
        - 可调用对象
        - None，表示参数值可以是 None
        - 任何类型，表示任何该类型的实例都是有效的
        - 一个 Options 对象，表示给定类型的一组元素
        - 一个 StrOptions 对象，表示一组字符串
        - 字符串 "boolean"
        - 字符串 "verbose"
        - 字符串 "cv_object"
        - 字符串 "nan"
        - 一个 MissingValues 对象，表示缺失值的标记
        - 一个 HasMethods 对象，表示对象必须具有的方法
        - 一个 Hidden 对象，表示不向用户公开的约束条件

    params : dict
        字典，键为参数名，值为参数值。用于对约束条件进行验证。

    caller_name : str
        调用此函数的估计器、函数或方法的名称。
    """
    # 遍历传入的参数字典，param_name 是参数名，param_val 是参数值
    for param_name, param_val in params.items():
        # 如果参数没有定义约束，允许跳过验证，以便第三方评估器继承 sklearn 评估器时不受验证工具的约束
        if param_name not in parameter_constraints:
            continue
        
        # 获取参数名对应的约束条件
        constraints = parameter_constraints[param_name]

        # 如果约束条件为 "no_validation"，则跳过验证
        if constraints == "no_validation":
            continue
        
        # 将约束条件转换为约束对象列表，每个约束通过 make_constraint 函数创建
        constraints = [make_constraint(constraint) for constraint in constraints]

        # 遍历该参数的所有约束条件
        for constraint in constraints:
            # 如果当前约束条件满足参数值，则无需继续检查其他约束条件
            if constraint.is_satisfied_by(param_val):
                # 这个约束条件已满足，可以中断循环
                break
        else:
            # 如果没有任何约束条件满足，抛出 InvalidParameterError 异常，并提供详细的错误消息

            # 过滤掉不希望在错误消息中显示的约束条件，例如内部选项或非官方支持的选项
            constraints = [
                constraint for constraint in constraints if not constraint.hidden
            ]

            # 根据约束条件的数量构建错误消息中的约束描述字符串
            if len(constraints) == 1:
                constraints_str = f"{constraints[0]}"
            else:
                constraints_str = (
                    f"{', '.join([str(c) for c in constraints[:-1]])} or"
                    f" {constraints[-1]}"
                )

            # 抛出参数错误异常，指示哪个参数在哪个函数中出错，并提供详细的错误信息
            raise InvalidParameterError(
                f"The {param_name!r} parameter of {caller_name} must be"
                f" {constraints_str}. Got {param_val!r} instead."
            )
def make_constraint(constraint):
    """Convert the constraint into the appropriate Constraint object.

    Parameters
    ----------
    constraint : object
        The constraint to convert.

    Returns
    -------
    constraint : instance of _Constraint
        The converted constraint.
    """
    if isinstance(constraint, str) and constraint == "array-like":
        return _ArrayLikes()  # 返回一个针对数组类似对象的约束对象
    if isinstance(constraint, str) and constraint == "sparse matrix":
        return _SparseMatrices()  # 返回一个针对稀疏矩阵的约束对象
    if isinstance(constraint, str) and constraint == "random_state":
        return _RandomStates()  # 返回一个针对随机状态的约束对象
    if constraint is callable:
        return _Callables()  # 返回一个针对可调用对象的约束对象
    if constraint is None:
        return _NoneConstraint()  # 返回一个空约束对象
    if isinstance(constraint, type):
        return _InstancesOf(constraint)  # 返回一个针对特定类型实例的约束对象
    if isinstance(
        constraint, (Interval, StrOptions, Options, HasMethods, MissingValues)
    ):
        return constraint  # 直接返回某些特定类型的约束对象本身
    if isinstance(constraint, str) and constraint == "boolean":
        return _Booleans()  # 返回一个针对布尔值的约束对象
    if isinstance(constraint, str) and constraint == "verbose":
        return _VerboseHelper()  # 返回一个用于详细信息辅助的约束对象
    if isinstance(constraint, str) and constraint == "cv_object":
        return _CVObjects()  # 返回一个针对交叉验证对象的约束对象
    if isinstance(constraint, Hidden):
        constraint = make_constraint(constraint.constraint)
        constraint.hidden = True  # 对于隐藏约束，递归处理并设置其隐藏属性
        return constraint
    if isinstance(constraint, str) and constraint == "nan":
        return _NanConstraint()  # 返回一个针对NaN值的约束对象
    raise ValueError(f"Unknown constraint type: {constraint}")  # 抛出未知约束类型的异常


def validate_params(parameter_constraints, *, prefer_skip_nested_validation):
    """Decorator to validate types and values of functions and methods.

    Parameters
    ----------
    parameter_constraints : dict
        A dictionary `param_name: list of constraints`. See the docstring of
        `validate_parameter_constraints` for a description of the accepted constraints.

        Note that the *args and **kwargs parameters are not validated and must not be
        present in the parameter_constraints dictionary.

    prefer_skip_nested_validation : bool
        If True, the validation of parameters of inner estimators or functions
        called by the decorated function will be skipped.

        This is useful to avoid validating many times the parameters passed by the
        user from the public facing API. It's also useful to avoid validating
        parameters that we pass internally to inner functions that are guaranteed to
        be valid by the test suite.

        It should be set to True for most functions, except for those that receive
        non-validated objects as parameters or that are just wrappers around classes
        because they only perform a partial validation.

    Returns
    -------
    decorated_function : function or method
        The decorated function.
    """
    def decorator(func):
        # 将参数约束字典作为函数的属性设置，以便动态检查约束条件以进行自动化测试。
        setattr(func, "_skl_parameter_constraints", parameter_constraints)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取全局配置中的参数验证跳过标志
            global_skip_validation = get_config()["skip_parameter_validation"]
            if global_skip_validation:
                return func(*args, **kwargs)

            # 获取函数的签名信息
            func_sig = signature(func)

            # 将 *args/**kwargs 映射到函数签名中
            params = func_sig.bind(*args, **kwargs)
            params.apply_defaults()

            # 忽略 self/cls 和位置参数/关键字标记
            to_ignore = [
                p.name
                for p in func_sig.parameters.values()
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            to_ignore += ["self", "cls"]
            params = {k: v for k, v in params.arguments.items() if k not in to_ignore}

            # 验证参数约束条件
            validate_parameter_constraints(
                parameter_constraints, params, caller_name=func.__qualname__
            )

            try:
                # 使用配置上下文管理器，设置参数验证跳过标志
                with config_context(
                    skip_parameter_validation=(
                        prefer_skip_nested_validation or global_skip_validation
                    )
                ):
                    return func(*args, **kwargs)
            except InvalidParameterError as e:
                # 当函数只是一个估计器的包装器时，允许函数委托验证给估计器，
                # 但在错误消息中用函数的名称替换估计器的名称，以避免混淆。
                msg = re.sub(
                    r"parameter of \w+ must be",
                    f"parameter of {func.__qualname__} must be",
                    str(e),
                )
                raise InvalidParameterError(msg) from e

        return wrapper

    return decorator
class RealNotInt(Real):
    """A type that represents reals that are not instances of int.

    Behaves like float, but also works with values extracted from numpy arrays.
    isintance(1, RealNotInt) -> False
    isinstance(1.0, RealNotInt) -> True
    """

RealNotInt.register(float)
# 将 float 类型注册为 RealNotInt 的子类


def _type_name(t):
    """Convert type into human readable string."""
    module = t.__module__
    qualname = t.__qualname__
    if module == "builtins":
        return qualname
    elif t == Real:
        return "float"
    elif t == Integral:
        return "int"
    return f"{module}.{qualname}"
# 将给定类型 t 转换成可读的字符串形式


class _Constraint(ABC):
    """Base class for the constraint objects."""

    def __init__(self):
        self.hidden = False
        # 初始化 hidden 属性为 False

    @abstractmethod
    def is_satisfied_by(self, val):
        """Whether or not a value satisfies the constraint.

        Parameters
        ----------
        val : object
            The value to check.

        Returns
        -------
        is_satisfied : bool
            Whether or not the constraint is satisfied by this value.
        """
        # 抽象方法：检查值是否满足约束条件

    @abstractmethod
    def __str__(self):
        """A human readable representational string of the constraint."""
        # 抽象方法：返回约束条件的人类可读字符串表示


class _InstancesOf(_Constraint):
    """Constraint representing instances of a given type.

    Parameters
    ----------
    type : type
        The valid type.
    """

    def __init__(self, type):
        super().__init__()
        self.type = type
        # 初始化约束条件类型

    def is_satisfied_by(self, val):
        return isinstance(val, self.type)
        # 检查值是否是给定类型的实例

    def __str__(self):
        return f"an instance of {_type_name(self.type)!r}"
        # 返回约束条件的字符串表示，指定类型的实例


class _NoneConstraint(_Constraint):
    """Constraint representing the None singleton."""

    def is_satisfied_by(self, val):
        return val is None
        # 检查值是否为 None

    def __str__(self):
        return "None"
        # 返回字符串 "None"


class _NanConstraint(_Constraint):
    """Constraint representing the indicator `np.nan`."""

    def is_satisfied_by(self, val):
        return (
            not isinstance(val, Integral) and isinstance(val, Real) and math.isnan(val)
        )
        # 检查值是否为 `np.nan`

    def __str__(self):
        return "numpy.nan"
        # 返回字符串 "numpy.nan"


class _PandasNAConstraint(_Constraint):
    """Constraint representing the indicator `pd.NA`."""

    def is_satisfied_by(self, val):
        try:
            import pandas as pd

            return isinstance(val, type(pd.NA)) and pd.isna(val)
            # 检查值是否为 `pd.NA`
        except ImportError:
            return False

    def __str__(self):
        return "pandas.NA"
        # 返回字符串 "pandas.NA"


class Options(_Constraint):
    """Constraint representing a finite set of instances of a given type.

    Parameters
    ----------
    type : type

    options : set
        The set of valid scalars.

    deprecated : set or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """
    # 初始化方法，用于设置对象的类型、选项和可选的弃用选项集合
    def __init__(self, type, options, *, deprecated=None):
        # 调用父类的初始化方法
        super().__init__()
        # 设置对象的类型
        self.type = type
        # 设置对象的选项
        self.options = options
        # 如果传入了弃用选项集合，则使用该集合，否则使用空集合
        self.deprecated = deprecated or set()

        # 检查弃用选项集合是否是选项集合的子集，如果不是则抛出 ValueError 异常
        if self.deprecated - self.options:
            raise ValueError("The deprecated options must be a subset of the options.")

    # 判断给定的值是否符合对象要求
    def is_satisfied_by(self, val):
        # 返回值是否是指定类型的实例，并且在选项列表中
        return isinstance(val, self.type) and val in self.options

    # 给选项添加弃用标记（如果需要的话）
    def _mark_if_deprecated(self, option):
        """Add a deprecated mark to an option if needed."""
        # 将选项转换成字符串形式
        option_str = f"{option!r}"
        # 如果选项在弃用选项集合中，则在选项字符串后面添加 "(deprecated)" 标记
        if option in self.deprecated:
            option_str = f"{option_str} (deprecated)"
        return option_str

    # 返回对象的字符串表示形式
    def __str__(self):
        # 使用 _mark_if_deprecated 方法给所有选项添加弃用标记（如果需要的话），并用逗号分隔连接成字符串
        options_str = (
            f"{', '.join([self._mark_if_deprecated(o) for o in self.options])}"
        )
        # 返回格式化后的对象描述字符串
        return f"a {_type_name(self.type)} among {{{options_str}}}"
class StrOptions(Options):
    """Constraint representing a finite set of strings.
    
    Parameters
    ----------
    options : set of str
        The set of valid strings.
    
    deprecated : set of str or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """

    def __init__(self, options, *, deprecated=None):
        # 调用父类 Options 的构造函数，传入字符串类型和有效选项集合
        super().__init__(type=str, options=options, deprecated=deprecated)


class Interval(_Constraint):
    """Constraint representing a typed interval.
    
    Parameters
    ----------
    type : {numbers.Integral, numbers.Real, RealNotInt}
        The set of numbers in which to set the interval.
        
        If RealNotInt, only reals that don't have the integer type
        are allowed. For example 1.0 is allowed but 1 is not.
    
    left : float or int or None
        The left bound of the interval. None means left bound is -∞.
    
    right : float, int or None
        The right bound of the interval. None means right bound is +∞.
    
    closed : {"left", "right", "both", "neither"}
        Whether the interval is open or closed. Possible choices are:
        
        - `"left"`: the interval is closed on the left and open on the right.
          It is equivalent to the interval `[ left, right )`.
        - `"right"`: the interval is closed on the right and open on the left.
          It is equivalent to the interval `( left, right ]`.
        - `"both"`: the interval is closed.
          It is equivalent to the interval `[ left, right ]`.
        - `"neither"`: the interval is open.
          It is equivalent to the interval `( left, right )`.
    
    Notes
    -----
    Setting a bound to `None` and setting the interval closed is valid. For instance,
    strictly speaking, `Interval(Real, 0, None, closed="both")` corresponds to
    `[0, +∞) U {+∞}`.
    """

    def __init__(self, type, left, right, *, closed):
        # 调用父类 _Constraint 的构造函数
        super().__init__()
        # 设置属性：约束类型、左边界、右边界、闭合方式
        self.type = type
        self.left = left
        self.right = right
        self.closed = closed
        
        # 检查参数的有效性
        self._check_params()
    def _check_params(self):
        # 检查参数 self.type 是否为 Integral、Real 或 RealNotInt 中的一种，否则抛出 ValueError 异常
        if self.type not in (Integral, Real, RealNotInt):
            raise ValueError(
                "type must be either numbers.Integral, numbers.Real or RealNotInt."
                f" Got {self.type} instead."
            )

        # 检查参数 self.closed 是否为 'left'、'right'、'both' 或 'neither' 中的一种，否则抛出 ValueError 异常
        if self.closed not in ("left", "right", "both", "neither"):
            raise ValueError(
                "closed must be either 'left', 'right', 'both' or 'neither'. "
                f"Got {self.closed} instead."
            )

        # 如果 self.type 是 Integral
        if self.type is Integral:
            suffix = "for an interval over the integers."
            # 如果 self.left 不为 None 且不是 Integral 类型，则抛出 TypeError 异常
            if self.left is not None and not isinstance(self.left, Integral):
                raise TypeError(f"Expecting left to be an int {suffix}")
            # 如果 self.right 不为 None 且不是 Integral 类型，则抛出 TypeError 异常
            if self.right is not None and not isinstance(self.right, Integral):
                raise TypeError(f"Expecting right to be an int {suffix}")
            # 如果 self.left 为 None 且 self.closed 为 'left' 或 'both'，则抛出 ValueError 异常
            if self.left is None and self.closed in ("left", "both"):
                raise ValueError(
                    f"left can't be None when closed == {self.closed} {suffix}"
                )
            # 如果 self.right 为 None 且 self.closed 为 'right' 或 'both'，则抛出 ValueError 异常
            if self.right is None and self.closed in ("right", "both"):
                raise ValueError(
                    f"right can't be None when closed == {self.closed} {suffix}"
                )
        else:
            # 如果 self.left 不为 None 且不是 Real 类型，则抛出 TypeError 异常
            if self.left is not None and not isinstance(self.left, Real):
                raise TypeError("Expecting left to be a real number.")
            # 如果 self.right 不为 None 且不是 Real 类型，则抛出 TypeError 异常
            if self.right is not None and not isinstance(self.right, Real):
                raise TypeError("Expecting right to be a real number.")

        # 如果 self.right 和 self.left 都不为 None 且 self.right <= self.left，则抛出 ValueError 异常
        if self.right is not None and self.left is not None and self.right <= self.left:
            raise ValueError(
                f"right can't be less than left. Got left={self.left} and "
                f"right={self.right}"
            )

    def __contains__(self, val):
        # 如果 val 不是 Integral 类型且是 NaN，则返回 False
        if not isinstance(val, Integral) and np.isnan(val):
            return False

        # 根据 self.closed 的值选择相应的比较操作符
        left_cmp = operator.lt if self.closed in ("left", "both") else operator.le
        right_cmp = operator.gt if self.closed in ("right", "both") else operator.ge

        # 设置 left 和 right 的默认值
        left = -np.inf if self.left is None else self.left
        right = np.inf if self.right is None else self.right

        # 检查 val 是否在指定的区间内，如果不在则返回 False
        if left_cmp(val, left):
            return False
        if right_cmp(val, right):
            return False
        return True

    def is_satisfied_by(self, val):
        # 如果 val 的类型不是 self.type，则返回 False
        if not isinstance(val, self.type):
            return False

        # 调用 __contains__ 方法检查 val 是否在当前对象定义的区间内，返回结果
        return val in self
    # 返回对象的字符串表示形式
    def __str__(self):
        # 根据对象的类型确定描述类型字符串
        type_str = "an int" if self.type is Integral else "a float"
        
        # 确定左边界的方括号或圆括号
        left_bracket = "[" if self.closed in ("left", "both") else "("
        
        # 确定左边界的值，如果没有指定则使用负无穷大
        left_bound = "-inf" if self.left is None else self.left
        
        # 确定右边界的值，如果没有指定则使用正无穷大
        right_bound = "inf" if self.right is None else self.right
        
        # 确定右边界的方括号或圆括号
        right_bracket = "]" if self.closed in ("right", "both") else ")"
        
        # 如果左边界或右边界是实数但类型不是整数，则将其转换为浮点数以获得更好的表示
        if not self.type == Integral and isinstance(self.left, Real):
            left_bound = float(left_bound)
        if not self.type == Integral and isinstance(self.right, Real):
            right_bound = float(right_bound)
        
        # 构建并返回对象的字符串表示形式
        return (
            f"{type_str} in the range "
            f"{left_bracket}{left_bound}, {right_bound}{right_bracket}"
        )
# 表示约束条件，用于表示类似数组的对象
class _ArrayLikes(_Constraint):
    """Constraint representing array-likes"""

    # 检查给定的值是否符合数组样式的约束条件
    def is_satisfied_by(self, val):
        return _is_arraylike_not_scalar(val)

    # 返回描述字符串，表示该约束条件是一个类似数组的对象
    def __str__(self):
        return "an array-like"


# 表示约束条件，用于表示稀疏矩阵
class _SparseMatrices(_Constraint):
    """Constraint representing sparse matrices."""

    # 检查给定的值是否符合稀疏矩阵的约束条件
    def is_satisfied_by(self, val):
        return issparse(val)

    # 返回描述字符串，表示该约束条件是一个稀疏矩阵
    def __str__(self):
        return "a sparse matrix"


# 表示约束条件，用于表示可调用对象
class _Callables(_Constraint):
    """Constraint representing callables."""

    # 检查给定的值是否为可调用对象
    def is_satisfied_by(self, val):
        return callable(val)

    # 返回描述字符串，表示该约束条件是一个可调用对象
    def __str__(self):
        return "a callable"


# 表示约束条件，用于表示随机状态
class _RandomStates(_Constraint):
    """Constraint representing random states.

    Convenience class for
    [Interval(Integral, 0, 2**32 - 1, closed="both"), np.random.RandomState, None]
    """

    # 初始化函数，设定特定的约束条件
    def __init__(self):
        super().__init__()
        self._constraints = [
            Interval(Integral, 0, 2**32 - 1, closed="both"),
            _InstancesOf(np.random.RandomState),
            _NoneConstraint(),
        ]

    # 检查给定的值是否符合任何一个约束条件
    def is_satisfied_by(self, val):
        return any(c.is_satisfied_by(val) for c in self._constraints)

    # 返回描述字符串，表示该约束条件是一个随机状态对象
    def __str__(self):
        return (
            f"{', '.join([str(c) for c in self._constraints[:-1]])} or"
            f" {self._constraints[-1]}"
        )


# 表示约束条件，用于表示类似布尔值的对象
class _Booleans(_Constraint):
    """Constraint representing boolean likes.

    Convenience class for
    [bool, np.bool_]
    """

    # 初始化函数，设定特定的约束条件
    def __init__(self):
        super().__init__()
        self._constraints = [
            _InstancesOf(bool),
            _InstancesOf(np.bool_),
        ]

    # 检查给定的值是否符合任何一个约束条件
    def is_satisfied_by(self, val):
        return any(c.is_satisfied_by(val) for c in self._constraints)

    # 返回描述字符串，表示该约束条件是一个类似布尔值的对象
    def __str__(self):
        return (
            f"{', '.join([str(c) for c in self._constraints[:-1]])} or"
            f" {self._constraints[-1]}"
        )


# 表示约束条件，用于辅助参数中的 verbose 参数
class _VerboseHelper(_Constraint):
    """Helper constraint for the verbose parameter.

    Convenience class for
    [Interval(Integral, 0, None, closed="left"), bool, numpy.bool_]
    """

    # 初始化函数，设定特定的约束条件
    def __init__(self):
        super().__init__()
        self._constraints = [
            Interval(Integral, 0, None, closed="left"),
            _InstancesOf(bool),
            _InstancesOf(np.bool_),
        ]

    # 检查给定的值是否符合任何一个约束条件
    def is_satisfied_by(self, val):
        return any(c.is_satisfied_by(val) for c in self._constraints)

    # 返回描述字符串，表示该约束条件是一个辅助参数中的 verbose 参数
    def __str__(self):
        return (
            f"{', '.join([str(c) for c in self._constraints[:-1]])} or"
            f" {self._constraints[-1]}"
        )


# 表示约束条件，用于辅助参数中的 missing_values 参数
class MissingValues(_Constraint):
    """Helper constraint for the `missing_values` parameters.

    Convenience for
    [
        Integral,
        Interval(Real, None, None, closed="both"),
        str,   # when numeric_only is False
        None,  # when numeric_only is False
        _NanConstraint(),
        _PandasNAConstraint(),
    ]

    Parameters
    ----------

    """

    # 初始化函数，设定特定的约束条件
    def __init__(self):
        super().__init__()
        self._constraints = [
            Integral,
            Interval(Real, None, None, closed="both"),
            str,   # when numeric_only is False
            None,  # when numeric_only is False
            _NanConstraint(),
            _PandasNAConstraint(),
        ]
    # numeric_only : bool, default=False
    # 是否仅考虑数值类型的缺失值标记。

    """
    # 初始化一个缺失值约束对象。
    """
    def __init__(self, numeric_only=False):
        # 调用父类的初始化方法
        super().__init__()

        # 设置实例变量 numeric_only，指示是否仅考虑数值类型的缺失值
        self.numeric_only = numeric_only

        # 定义缺失值约束的列表
        self._constraints = [
            _InstancesOf(Integral),  # 要求是整数类型的实例
            # 使用 Real 类型的区间来忽略 np.nan，它有自己的约束
            Interval(Real, None, None, closed="both"),
            _NanConstraint(),         # NaN 约束
            _PandasNAConstraint(),    # Pandas 的 NA 约束
        ]
        # 如果 numeric_only 为 False，则扩展约束列表以包含字符串实例约束和 None 约束
        if not self.numeric_only:
            self._constraints.extend([_InstancesOf(str), _NoneConstraint()])

    # 检查给定值是否满足任何一个约束条件
    def is_satisfied_by(self, val):
        return any(c.is_satisfied_by(val) for c in self._constraints)

    # 返回该对象的字符串表示形式，描述其包含的所有约束条件
    def __str__(self):
        return (
            f"{', '.join([str(c) for c in self._constraints[:-1]])} or"
            f" {self._constraints[-1]}"
        )
class HasMethods(_Constraint):
    """Constraint representing objects that expose specific methods.
    
    It is useful for parameters following a protocol and where we don't want to impose
    an affiliation to a specific module or class.
    
    Parameters
    ----------
    methods : str or list of str
        The method(s) that the object is expected to expose.
    """

    @validate_params(
        {"methods": [str, list]},
        prefer_skip_nested_validation=True,
    )
    def __init__(self, methods):
        # 调用父类的初始化方法
        super().__init__()
        # 如果传入的 methods 是字符串，则转为单元素列表
        if isinstance(methods, str):
            methods = [methods]
        # 设置对象的 methods 属性
        self.methods = methods

    def is_satisfied_by(self, val):
        # 检查对象是否实现了所有指定的方法
        return all(callable(getattr(val, method, None)) for method in self.methods)

    def __str__(self):
        # 根据 methods 属性生成描述字符串
        if len(self.methods) == 1:
            methods = f"{self.methods[0]!r}"
        else:
            methods = (
                f"{', '.join([repr(m) for m in self.methods[:-1]])} and"
                f" {self.methods[-1]!r}"
            )
        return f"an object implementing {methods}"


class _IterablesNotString(_Constraint):
    """Constraint representing iterables that are not strings."""

    def is_satisfied_by(self, val):
        # 检查 val 是否是 iterable 类型但不是字符串类型
        return isinstance(val, Iterable) and not isinstance(val, str)

    def __str__(self):
        # 返回描述字符串
        return "an iterable"


class _CVObjects(_Constraint):
    """Constraint representing cv objects.
    
    Convenient class for
    [
        Interval(Integral, 2, None, closed="left"),
        HasMethods(["split", "get_n_splits"]),
        _IterablesNotString(),
        None,
    ]
    """

    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化约束条件列表
        self._constraints = [
            Interval(Integral, 2, None, closed="left"),  # 要求是大于等于2的整数区间（左闭右开）
            HasMethods(["split", "get_n_splits"]),       # 要求具有 'split' 和 'get_n_splits' 方法
            _IterablesNotString(),                      # 要求是非字符串的可迭代对象
            _NoneConstraint(),                          # 要求是 None 类型
        ]

    def is_satisfied_by(self, val):
        # 检查 val 是否满足任一约束条件
        return any(c.is_satisfied_by(val) for c in self._constraints)

    def __str__(self):
        # 根据 _constraints 属性生成描述字符串
        return (
            f"{', '.join([str(c) for c in self._constraints[:-1]])} or"
            f" {self._constraints[-1]}"
        )


class Hidden:
    """Class encapsulating a constraint not meant to be exposed to the user.
    
    Parameters
    ----------
    constraint : str or _Constraint instance
        The constraint to be used internally.
    """

    def __init__(self, constraint):
        # 设置内部约束条件
        self.constraint = constraint


def generate_invalid_param_val(constraint):
    """Return a value that does not satisfy the constraint.
    
    Raises a NotImplementedError if there exists no invalid value for this constraint.
    This is only useful for testing purpose.
    
    Parameters
    ----------
    constraint : _Constraint instance
        The constraint to generate a value for.
    
    Returns
    -------
    val : object
        A value that does not satisfy the constraint.
    """
    # 检查 constraint 是否是 StrOptions 类型的实例
    if isinstance(constraint, StrOptions):
        # 返回一个字符串，内容为 "not " 后跟 constraint.options 中所有选项的逻辑或连接结果
        return f"not {' or '.join(constraint.options)}"

    # 检查 constraint 是否是 MissingValues 类型的实例
    if isinstance(constraint, MissingValues):
        # 返回一个包含整数 1, 2, 3 的 NumPy 数组
        return np.array([1, 2, 3])

    # 检查 constraint 是否是 _VerboseHelper 类型的实例
    if isinstance(constraint, _VerboseHelper):
        # 返回整数 -1
        return -1

    # 检查 constraint 是否是 HasMethods 类型的实例
    if isinstance(constraint, HasMethods):
        # 返回一个新创建的没有方法的类的实例
        return type("HasNotMethods", (), {})()

    # 检查 constraint 是否是 _IterablesNotString 类型的实例
    if isinstance(constraint, _IterablesNotString):
        # 返回字符串 "a string"
        return "a string"

    # 检查 constraint 是否是 _CVObjects 类型的实例
    if isinstance(constraint, _CVObjects):
        # 返回字符串 "not a cv object"
        return "not a cv object"

    # 检查 constraint 是否是 Interval 类型的实例，并且其 type 属性是 Integral 类型
    if isinstance(constraint, Interval) and constraint.type is Integral:
        # 如果左边界不为 None，则返回左边界减 1 的值
        if constraint.left is not None:
            return constraint.left - 1
        # 如果右边界不为 None，则返回右边界加 1 的值
        if constraint.right is not None:
            return constraint.right + 1

        # 如果既没有左边界也没有右边界，则抛出 NotImplementedError 异常
        raise NotImplementedError

    # 检查 constraint 是否是 Interval 类型的实例，并且其 type 属性是 Real 或 RealNotInt 之一
    if isinstance(constraint, Interval) and constraint.type in (Real, RealNotInt):
        # 如果左边界不为 None，则返回左边界减 1e-6 的值
        if constraint.left is not None:
            return constraint.left - 1e-6
        # 如果右边界不为 None，则返回右边界加 1e-6 的值
        if constraint.right is not None:
            return constraint.right + 1e-6

        # 如果边界是 -inf, +inf
        if constraint.closed in ("right", "neither"):
            return -np.inf
        if constraint.closed in ("left", "neither"):
            return np.inf

        # 如果区间是 [-inf, +inf]
        return np.nan

    # 如果以上所有情况都不符合，则抛出 NotImplementedError 异常
    raise NotImplementedError
# 生成满足特定约束条件的值，仅用于测试目的

def generate_valid_param(constraint):
    """Return a value that does satisfy a constraint.

    This is only useful for testing purpose.

    Parameters
    ----------
    constraint : Constraint instance
        The constraint to generate a value for.

    Returns
    -------
    val : object
        A value that does satisfy the constraint.
    """

    if isinstance(constraint, _ArrayLikes):
        # 如果约束是 _ArrayLikes 类型，则返回一个示例数组
        return np.array([1, 2, 3])

    if isinstance(constraint, _SparseMatrices):
        # 如果约束是 _SparseMatrices 类型，则返回一个稀疏矩阵示例
        return csr_matrix([[0, 1], [1, 0]])

    if isinstance(constraint, _RandomStates):
        # 如果约束是 _RandomStates 类型，则返回一个特定种子的随机状态生成器
        return np.random.RandomState(42)

    if isinstance(constraint, _Callables):
        # 如果约束是 _Callables 类型，则返回一个简单的匿名函数
        return lambda x: x

    if isinstance(constraint, _NoneConstraint):
        # 如果约束是 _NoneConstraint 类型，则返回 None
        return None

    if isinstance(constraint, _InstancesOf):
        if constraint.type is np.ndarray:
            # 对于 _InstancesOf 约束，如果是 np.ndarray 类型，则返回一个示例数组
            return np.array([1, 2, 3])

        if constraint.type in (Integral, Real):
            # 对于 _InstancesOf 约束，如果是 Integral 或 Real 抽象类，则返回整数 1
            return 1

        # 对于其他类型，则实例化该类型并返回
        return constraint.type()

    if isinstance(constraint, _Booleans):
        # 如果约束是 _Booleans 类型，则返回 True
        return True

    if isinstance(constraint, _VerboseHelper):
        # 如果约束是 _VerboseHelper 类型，则返回整数 1
        return 1

    if isinstance(constraint, MissingValues) and constraint.numeric_only:
        # 如果约束是 MissingValues 类型且仅限于数值，返回 np.nan
        return np.nan

    if isinstance(constraint, MissingValues) and not constraint.numeric_only:
        # 如果约束是 MissingValues 类型但不仅限于数值，返回字符串 "missing"
        return "missing"

    if isinstance(constraint, HasMethods):
        # 如果约束是 HasMethods 类型，则创建一个匿名类，具有约束指定的方法（方法体为空）
        return type(
            "ValidHasMethods", (), {m: lambda self: None for m in constraint.methods}
        )()

    if isinstance(constraint, _IterablesNotString):
        # 如果约束是 _IterablesNotString 类型，则返回一个示例列表
        return [1, 2, 3]

    if isinstance(constraint, _CVObjects):
        # 如果约束是 _CVObjects 类型，则返回整数 5
        return 5

    if isinstance(constraint, Options):  # includes StrOptions
        # 如果约束是 Options 类型，则返回第一个选项
        for option in constraint.options:
            return option

    if isinstance(constraint, Interval):
        # 如果约束是 Interval 类型
        interval = constraint
        if interval.left is None and interval.right is None:
            return 0
        elif interval.left is None:
            return interval.right - 1
        elif interval.right is None:
            return interval.left + 1
        else:
            if interval.type is Real:
                # 如果是实数区间，则返回区间中点
                return (interval.left + interval.right) / 2
            else:
                # 对于其他类型的区间，返回左端点加 1
                return interval.left + 1

    # 如果未识别的约束类型，则引发 ValueError 异常
    raise ValueError(f"Unknown constraint type: {constraint}")
```