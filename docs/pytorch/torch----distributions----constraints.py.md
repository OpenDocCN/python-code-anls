# `.\pytorch\torch\distributions\constraints.py`

```py
# mypy: allow-untyped-defs
r"""
The following constraints are implemented:

- ``constraints.boolean``
- ``constraints.cat``
- ``constraints.corr_cholesky``
- ``constraints.dependent``
- ``constraints.greater_than(lower_bound)``
- ``constraints.greater_than_eq(lower_bound)``
- ``constraints.independent(constraint, reinterpreted_batch_ndims)``
- ``constraints.integer_interval(lower_bound, upper_bound)``
- ``constraints.interval(lower_bound, upper_bound)``
- ``constraints.less_than(upper_bound)``
- ``constraints.lower_cholesky``
- ``constraints.lower_triangular``
- ``constraints.multinomial``
- ``constraints.nonnegative``
- ``constraints.nonnegative_integer``
- ``constraints.one_hot``
- ``constraints.positive_integer``
- ``constraints.positive``
- ``constraints.positive_semidefinite``
- ``constraints.positive_definite``
- ``constraints.real_vector``
- ``constraints.real``
- ``constraints.simplex``
- ``constraints.symmetric``
- ``constraints.stack``
- ``constraints.square``
- ``constraints.symmetric``
- ``constraints.unit_interval``
"""

import torch

__all__ = [
    "Constraint",                   # 导出的类名列表
    "boolean",                      # 布尔约束
    "cat",                          # 类别约束
    "corr_cholesky",                # 相关 Cholesky 约束
    "dependent",                    # 依赖约束
    "dependent_property",           # 依赖属性
    "greater_than",                 # 大于约束
    "greater_than_eq",              # 大于等于约束
    "independent",                  # 独立约束
    "integer_interval",             # 整数区间约束
    "interval",                     # 区间约束
    "half_open_interval",           # 半开区间约束
    "is_dependent",                 # 是否依赖
    "less_than",                    # 小于约束
    "lower_cholesky",               # 下三角 Cholesky 约束
    "lower_triangular",             # 下三角形约束
    "multinomial",                  # 多项式约束
    "nonnegative",                  # 非负约束
    "nonnegative_integer",          # 非负整数约束
    "one_hot",                      # 独热编码约束
    "positive",                     # 正数约束
    "positive_semidefinite",        # 正半定约束
    "positive_definite",            # 正定约束
    "positive_integer",             # 正整数约束
    "real",                         # 实数约束
    "real_vector",                  # 实向量约束
    "simplex",                      # 单纯形约束
    "square",                       # 方阵约束
    "stack",                        # 堆叠约束
    "symmetric",                    # 对称约束
    "unit_interval",                # 单位区间约束
]


class Constraint:
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.

    Attributes:
        is_discrete (bool): Whether constrained space is discrete.
            Defaults to False.
        event_dim (int): Number of rightmost dimensions that together define
            an event. The :meth:`check` method will remove this many dimensions
            when computing validity.
    """
    is_discrete = False  # 默认为连续约束
    event_dim = 0  # 默认为单变量约束

    def check(self, value):
        """
        Returns a byte tensor of ``sample_shape + batch_shape`` indicating
        whether each event in value satisfies this constraint.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__[1:] + "()"


class _Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    These variables obey no simple coordinate-wise constraints.
    """
    Args:
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    ```
    
    # 定义类的初始化方法，用于设定静态属性值
    def __init__(self, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        self._is_discrete = is_discrete  # 设置对象的_is_discrete属性
        self._event_dim = event_dim  # 设置对象的_event_dim属性
        super().__init__()  # 调用父类的初始化方法

    @property
    # 属性装饰器，用于获取对象的is_discrete属性
    def is_discrete(self):
        if self._is_discrete is NotImplemented:
            raise NotImplementedError(".is_discrete cannot be determined statically")  # 如果_is_discrete属性为NotImplemented，则抛出异常
        return self._is_discrete  # 返回对象的_is_discrete属性值

    @property
    # 属性装饰器，用于获取对象的event_dim属性
    def event_dim(self):
        if self._event_dim is NotImplemented:
            raise NotImplementedError(".event_dim cannot be determined statically")  # 如果_event_dim属性为NotImplemented，则抛出异常
        return self._event_dim  # 返回对象的_event_dim属性值

    def __call__(self, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        """
        Support for syntax to customize static attributes::

            constraints.dependent(is_discrete=True, event_dim=1)
        """
        # 根据给定的参数或者对象的属性，返回_Dependent对象
        if is_discrete is NotImplemented:
            is_discrete = self._is_discrete
        if event_dim is NotImplemented:
            event_dim = self._event_dim
        return _Dependent(is_discrete=is_discrete, event_dim=event_dim)

    def check(self, x):
        raise ValueError("Cannot determine validity of dependent constraint")  # 抛出数值错误异常，表示无法确定相关约束的有效性
# 检查给定的约束是否为 `_Dependent` 类型的对象
def is_dependent(constraint):
    """
    Checks if ``constraint`` is a ``_Dependent`` object.

    Args:
        constraint : A ``Constraint`` object.

    Returns:
        ``bool``: True if ``constraint`` can be refined to the type ``_Dependent``, False otherwise.

    Examples:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch.distributions.constraints import is_dependent

        >>> dist = Bernoulli(probs = torch.tensor([0.6], requires_grad=True))
        >>> constraint1 = dist.arg_constraints["probs"]
        >>> constraint2 = dist.arg_constraints["logits"]

        >>> for constraint in [constraint1, constraint2]:
        >>>     if is_dependent(constraint):
        >>>         continue
    """
    return isinstance(constraint, _Dependent)


class _DependentProperty(property, _Dependent):
    """
    Decorator that extends @property to act like a `Dependent` constraint when
    called on a class and act like a property when called on an object.

    Example::

        class Uniform(Distribution):
            def __init__(self, low, high):
                self.low = low
                self.high = high
            @constraints.dependent_property(is_discrete=False, event_dim=0)
            def support(self):
                return constraints.interval(self.low, self.high)

    Args:
        fn (Callable): The function to be decorated.
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    """

    def __init__(
        self, fn=None, *, is_discrete=NotImplemented, event_dim=NotImplemented
    ):
        super().__init__(fn)
        self._is_discrete = is_discrete
        self._event_dim = event_dim

    def __call__(self, fn):
        """
        Support for syntax to customize static attributes::

            @constraints.dependent_property(is_discrete=True, event_dim=1)
            def support(self):
                ...
        """
        return _DependentProperty(
            fn, is_discrete=self._is_discrete, event_dim=self._event_dim
        )


class _IndependentConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """
    # 初始化方法，接受基础约束和重新解释的批次维度作为参数
    def __init__(self, base_constraint, reinterpreted_batch_ndims):
        # 断言base_constraint是Constraint类型的实例
        assert isinstance(base_constraint, Constraint)
        # 断言reinterpreted_batch_ndims是非负整数
        assert isinstance(reinterpreted_batch_ndims, int)
        assert reinterpreted_batch_ndims >= 0
        # 将参数赋值给对象的属性
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        # 调用父类的初始化方法
        super().__init__()

    @property
    # 返回基础约束是否是离散的
    def is_discrete(self):
        return self.base_constraint.is_discrete

    @property
    # 返回事件维度，是基础约束的事件维度加上重新解释的批次维度
    def event_dim(self):
        return self.base_constraint.event_dim + self.reinterpreted_batch_ndims

    # 检查给定值是否符合约束
    def check(self, value):
        # 调用基础约束的check方法进行检查
        result = self.base_constraint.check(value)
        # 如果结果的维度小于重新解释的批次维度，抛出值错误异常
        if result.dim() < self.reinterpreted_batch_ndims:
            expected = self.base_constraint.event_dim + self.reinterpreted_batch_ndims
            raise ValueError(
                f"Expected value.dim() >= {expected} but got {value.dim()}"
            )
        # 将结果重塑为期望的形状，以便按行检查所有元素
        result = result.reshape(
            result.shape[: result.dim() - self.reinterpreted_batch_ndims] + (-1,)
        )
        result = result.all(-1)
        return result

    # 返回对象的字符串表示形式，包括基础约束和重新解释的批次维度
    def __repr__(self):
        return f"{self.__class__.__name__[1:]}({repr(self.base_constraint)}, {self.reinterpreted_batch_ndims})"
class _Boolean(Constraint):
    """
    Constrain to the two values `{0, 1}`.
    """

    is_discrete = True  # 表示这个约束是离散的

    def check(self, value):
        return (value == 0) | (value == 1)  # 检查传入的值是否为 0 或 1


class _OneHot(Constraint):
    """
    Constrain to one-hot vectors.
    """

    is_discrete = True  # 表示这个约束是离散的
    event_dim = 1  # 事件维度为 1

    def check(self, value):
        is_boolean = (value == 0) | (value == 1)  # 检查值是否为 0 或 1
        is_normalized = value.sum(-1).eq(1)  # 检查值的和是否为 1
        return is_boolean.all(-1) & is_normalized  # 返回是否所有元素都是 0 或 1，并且和为 1


class _IntegerInterval(Constraint):
    """
    Constrain to an integer interval `[lower_bound, upper_bound]`.
    """

    is_discrete = True  # 表示这个约束是离散的

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound  # 设置下界
        self.upper_bound = upper_bound  # 设置上界
        super().__init__()

    def check(self, value):
        return (
            (value % 1 == 0) & (self.lower_bound <= value) & (value <= self.upper_bound)
        )  # 检查值是否为整数，且在指定的闭区间内

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]  # 获取类名，去掉第一个字符 '_'
        fmt_string += (
            f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        )  # 格式化字符串，显示约束的下界和上界
        return fmt_string  # 返回格式化后的字符串


class _IntegerLessThan(Constraint):
    """
    Constrain to an integer interval `(-inf, upper_bound]`.
    """

    is_discrete = True  # 表示这个约束是离散的

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound  # 设置上界
        super().__init__()

    def check(self, value):
        return (value % 1 == 0) & (value <= self.upper_bound)  # 检查值是否为整数，且在指定的开区间内

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]  # 获取类名，去掉第一个字符 '_'
        fmt_string += f"(upper_bound={self.upper_bound})"  # 格式化字符串，显示约束的上界
        return fmt_string  # 返回格式化后的字符串


class _IntegerGreaterThan(Constraint):
    """
    Constrain to an integer interval `[lower_bound, inf)`.
    """

    is_discrete = True  # 表示这个约束是离散的

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound  # 设置下界
        super().__init__()

    def check(self, value):
        return (value % 1 == 0) & (value >= self.lower_bound)  # 检查值是否为整数，且在指定的开区间内

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]  # 获取类名，去掉第一个字符 '_'
        fmt_string += f"(lower_bound={self.lower_bound})"  # 格式化字符串，显示约束的下界
        return fmt_string  # 返回格式化后的字符串


class _Real(Constraint):
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """

    def check(self, value):
        return value == value  # 检查值是否为实数，不包括 NaN


class _GreaterThan(Constraint):
    """
    Constrain to a real half line `(lower_bound, inf]`.
    """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound  # 设置下界
        super().__init__()

    def check(self, value):
        return self.lower_bound < value  # 检查值是否大于指定的下界

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]  # 获取类名，去掉第一个字符 '_'
        fmt_string += f"(lower_bound={self.lower_bound})"  # 格式化字符串，显示约束的下界
        return fmt_string  # 返回格式化后的字符串


class _GreaterThanEq(Constraint):
    """
    Constrain to a real half line `[lower_bound, inf)`.
    """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound  # 设置下界
        super().__init__()
    # 检查给定的值是否大于等于对象的下界
    def check(self, value):
        return self.lower_bound <= value

    # 返回对象的字符串表示形式，包括类名和下界信息
    def __repr__(self):
        # 构造格式化字符串，类名去掉首字母'_'，加上下界信息
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound})"
        return fmt_string
# 定义一个约束类 `_LessThan`，用于表示一个半开区间 `[-inf, upper_bound)`
class _LessThan(Constraint):
    """
    Constrain to a real half line `[-inf, upper_bound)`.
    """

    # 初始化方法，接收上界 `upper_bound` 参数
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        super().__init__()  # 调用父类的初始化方法

    # 检查方法，检查给定的值是否小于 `upper_bound`
    def check(self, value):
        return value < self.upper_bound

    # 返回对象的字符串表示形式，包括类名和上界信息
    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]  # 获取类名，去除第一个字符 `_`
        fmt_string += f"(upper_bound={self.upper_bound})"  # 格式化字符串表示上界
        return fmt_string


# 定义一个约束类 `_Interval`，用于表示一个闭区间 `[lower_bound, upper_bound]`
class _Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`.
    """

    # 初始化方法，接收下界 `lower_bound` 和上界 `upper_bound` 参数
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()  # 调用父类的初始化方法

    # 检查方法，检查给定的值是否在闭区间 `[lower_bound, upper_bound]` 内
    def check(self, value):
        return (self.lower_bound <= value) & (value <= self.upper_bound)

    # 返回对象的字符串表示形式，包括类名和下界、上界信息
    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]  # 获取类名，去除第一个字符 `_`
        fmt_string += (
            f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        )  # 格式化字符串表示下界和上界
        return fmt_string


# 定义一个约束类 `_HalfOpenInterval`，用于表示一个半开区间 `[lower_bound, upper_bound)`
class _HalfOpenInterval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound)`.
    """

    # 初始化方法，接收下界 `lower_bound` 和上界 `upper_bound` 参数
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()  # 调用父类的初始化方法

    # 检查方法，检查给定的值是否在半开区间 `[lower_bound, upper_bound)` 内
    def check(self, value):
        return (self.lower_bound <= value) & (value < self.upper_bound)

    # 返回对象的字符串表示形式，包括类名和下界、上界信息
    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]  # 获取类名，去除第一个字符 `_`
        fmt_string += (
            f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        )  # 格式化字符串表示下界和上界
        return fmt_string


# 定义一个约束类 `_Simplex`，用于表示单位单纯形（最内层维度的单纯形）
class _Simplex(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """

    event_dim = 1  # 定义事件维度为 1

    # 检查方法，检查给定的值是否为单位单纯形
    def check(self, value):
        return torch.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)


# 定义一个约束类 `_Multinomial`，用于表示非负整数值的多项式分布，总和最多不超过上界
class _Multinomial(Constraint):
    """
    Constrain to nonnegative integer values summing to at most an upper bound.

    Note due to limitations of the Multinomial distribution, this currently
    checks the weaker condition ``value.sum(-1) <= upper_bound``. In the future
    this may be strengthened to ``value.sum(-1) == upper_bound``.
    """

    is_discrete = True  # 声明为离散分布
    event_dim = 1  # 定义事件维度为 1

    # 初始化方法，接收上界 `upper_bound` 参数
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    # 检查方法，检查给定的值是否满足多项式分布的约束条件
    def check(self, x):
        return (x >= 0).all(dim=-1) & (x.sum(dim=-1) <= self.upper_bound)


# 定义一个约束类 `_LowerTriangular`，用于表示下三角方阵
class _LowerTriangular(Constraint):
    """
    Constrain to lower-triangular square matrices.
    """

    event_dim = 2  # 定义事件维度为 2

    # 检查方法，检查给定的值是否为下三角矩阵
    def check(self, value):
        value_tril = value.tril()  # 提取输入值的下三角部分
        return (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]


# 定义一个约束类 `_LowerCholesky`，用于表示带有正对角线的下三角方阵
class _LowerCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals.
    """

    event_dim = 2  # 定义事件维度为 2
    # 定义一个方法用于检查矩阵是否符合特定条件
    def check(self, value):
        # 计算输入矩阵的下三角部分
        value_tril = value.tril()
        # 检查是否所有元素都在其对应位置上与原矩阵相等，返回布尔值
        lower_triangular = (
            (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]
        )

        # 检查矩阵对角线上的元素是否全部大于零，返回布尔值
        positive_diagonal = (value.diagonal(dim1=-2, dim2=-1) > 0).min(-1)[0]
        
        # 返回两个条件的逻辑与结果
        return lower_triangular & positive_diagonal
class _CorrCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals and each
    row vector being of unit length.
    """

    event_dim = 2  # 设置事件维度为2

    def check(self, value):
        tol = (
            torch.finfo(value.dtype).eps * value.size(-1) * 10
        )  # 计算容差，eps是value数据类型的最小精度，10是可调整的修正因子
        row_norm = torch.linalg.norm(value.detach(), dim=-1)  # 计算每行向量的范数
        unit_row_norm = (row_norm - 1.0).abs().le(tol).all(dim=-1)  # 检查每行向量是否单位长度
        return _LowerCholesky().check(value) & unit_row_norm  # 检查是否为下三角Cholesky矩阵


class _Square(Constraint):
    """
    Constrain to square matrices.
    """

    event_dim = 2  # 设置事件维度为2

    def check(self, value):
        return torch.full(
            size=value.shape[:-2],
            fill_value=(value.shape[-2] == value.shape[-1]),  # 检查是否为方阵
            dtype=torch.bool,
            device=value.device,
        )


class _Symmetric(_Square):
    """
    Constrain to Symmetric square matrices.
    """

    def check(self, value):
        square_check = super().check(value)  # 检查是否为方阵
        if not square_check.all():
            return square_check
        return torch.isclose(value, value.mT, atol=1e-6).all(-2).all(-1)  # 检查是否对称


class _PositiveSemidefinite(_Symmetric):
    """
    Constrain to positive-semidefinite matrices.
    """

    def check(self, value):
        sym_check = super().check(value)  # 检查是否对称
        if not sym_check.all():
            return sym_check
        return torch.linalg.eigvalsh(value).ge(0).all(-1)  # 检查是否半正定


class _PositiveDefinite(_Symmetric):
    """
    Constrain to positive-definite matrices.
    """

    def check(self, value):
        sym_check = super().check(value)  # 检查是否对称
        if not sym_check.all():
            return sym_check
        return torch.linalg.cholesky_ex(value).info.eq(0)  # 检查是否正定


class _Cat(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    each of size `lengths[dim]`, in a way compatible with :func:`torch.cat`.
    """

    def __init__(self, cseq, dim=0, lengths=None):
        assert all(isinstance(c, Constraint) for c in cseq)  # 确保所有约束条件为Constraint类的实例
        self.cseq = list(cseq)  # 存储约束条件列表
        if lengths is None:
            lengths = [1] * len(self.cseq)
        self.lengths = list(lengths)  # 存储子矩阵的长度列表
        assert len(self.lengths) == len(self.cseq)  # 确保长度列表与约束条件列表长度一致
        self.dim = dim  # 存储维度参数
        super().__init__()

    @property
    def is_discrete(self):
        return any(c.is_discrete for c in self.cseq)  # 检查是否为离散约束

    @property
    def event_dim(self):
        return max(c.event_dim for c in self.cseq)  # 返回约束条件中的最大事件维度

    def check(self, value):
        assert -value.dim() <= self.dim < value.dim()  # 确保维度参数在有效范围内
        checks = []
        start = 0
        for constr, length in zip(self.cseq, self.lengths):
            v = value.narrow(self.dim, start, length)  # 从value中切片出子矩阵
            checks.append(constr.check(v))  # 对子矩阵应用约束条件
            start = start + length  # 更新切片起始位置，避免使用+=以兼容JIT
        return torch.cat(checks, self.dim)  # 合并所有约束条件检查结果
    """
    `cseq` at the submatrices at dimension `dim`,
    in a way compatible with :func:`torch.stack`.
    """

    # 初始化函数，接受约束序列 `cseq` 和维度参数 `dim`
    def __init__(self, cseq, dim=0):
        # 断言确保 `cseq` 中的每个元素都是 Constraint 类的实例
        assert all(isinstance(c, Constraint) for c in cseq)
        # 将 `cseq` 转换为列表并存储在对象属性中
        self.cseq = list(cseq)
        # 存储维度参数 `dim` 到对象属性中
        self.dim = dim
        # 调用父类的初始化函数
        super().__init__()

    # 返回是否存在离散约束的属性
    @property
    def is_discrete(self):
        # 如果 `cseq` 中的任意约束对象具有 is_discrete 属性为真，则返回真
        return any(c.is_discrete for c in self.cseq)

    # 返回事件维度的属性
    @property
    def event_dim(self):
        # 计算 `cseq` 中约束对象的最大事件维度
        dim = max(c.event_dim for c in self.cseq)
        # 如果 `self.dim + dim` 小于 0，则将 dim 值增加 1
        if self.dim + dim < 0:
            dim += 1
        return dim

    # 检查给定值的方法
    def check(self, value):
        # 断言确保维度参数 `self.dim` 在合理范围内
        assert -value.dim() <= self.dim < value.dim()
        # 通过选择操作获取沿 `self.dim` 维度的子张量列表
        vs = [value.select(self.dim, i) for i in range(value.size(self.dim))]
        # 使用 torch.stack 将每个约束 `constr.check(v)` 的结果堆叠起来，沿 `self.dim` 维度
        return torch.stack(
            [constr.check(v) for v, constr in zip(vs, self.cseq)], self.dim
        )
# 定义公共接口变量和函数
dependent = _Dependent()  # 创建一个 _Dependent 类的实例对象，并赋值给 dependent
dependent_property = _DependentProperty  # 将 _DependentProperty 类的引用赋给 dependent_property
independent = _IndependentConstraint  # 将 _IndependentConstraint 类的引用赋给 independent
boolean = _Boolean()  # 创建一个 _Boolean 类的实例对象，并赋值给 boolean
one_hot = _OneHot()  # 创建一个 _OneHot 类的实例对象，并赋值给 one_hot
nonnegative_integer = _IntegerGreaterThan(0)  # 创建一个 _IntegerGreaterThan 类的实例对象，限定大于零的整数，并赋值给 nonnegative_integer
positive_integer = _IntegerGreaterThan(1)  # 创建一个 _IntegerGreaterThan 类的实例对象，限定大于一的整数，并赋值给 positive_integer
integer_interval = _IntegerInterval  # 将 _IntegerInterval 类的引用赋给 integer_interval
real = _Real()  # 创建一个 _Real 类的实例对象，并赋值给 real
real_vector = independent(real, 1)  # 调用 independent 函数创建一个实数向量约束对象，并赋值给 real_vector
positive = _GreaterThan(0.0)  # 创建一个 _GreaterThan 类的实例对象，限定大于零的数，并赋值给 positive
nonnegative = _GreaterThanEq(0.0)  # 创建一个 _GreaterThanEq 类的实例对象，限定大于等于零的数，并赋值给 nonnegative
greater_than = _GreaterThan  # 将 _GreaterThan 类的引用赋给 greater_than
greater_than_eq = _GreaterThanEq  # 将 _GreaterThanEq 类的引用赋给 greater_than_eq
less_than = _LessThan  # 将 _LessThan 类的引用赋给 less_than
multinomial = _Multinomial  # 将 _Multinomial 类的引用赋给 multinomial
unit_interval = _Interval(0.0, 1.0)  # 创建一个指定区间为 [0.0, 1.0] 的 _Interval 类的实例对象，并赋值给 unit_interval
interval = _Interval  # 将 _Interval 类的引用赋给 interval
half_open_interval = _HalfOpenInterval  # 将 _HalfOpenInterval 类的引用赋给 half_open_interval
simplex = _Simplex()  # 创建一个 _Simplex 类的实例对象，并赋值给 simplex
lower_triangular = _LowerTriangular()  # 创建一个 _LowerTriangular 类的实例对象，并赋值给 lower_triangular
lower_cholesky = _LowerCholesky()  # 创建一个 _LowerCholesky 类的实例对象，并赋值给 lower_cholesky
corr_cholesky = _CorrCholesky()  # 创建一个 _CorrCholesky 类的实例对象，并赋值给 corr_cholesky
square = _Square()  # 创建一个 _Square 类的实例对象，并赋值给 square
symmetric = _Symmetric()  # 创建一个 _Symmetric 类的实例对象，并赋值给 symmetric
positive_semidefinite = _PositiveSemidefinite()  # 创建一个 _PositiveSemidefinite 类的实例对象，并赋值给 positive_semidefinite
positive_definite = _PositiveDefinite()  # 创建一个 _PositiveDefinite 类的实例对象，并赋值给 positive_definite
cat = _Cat  # 将 _Cat 类的引用赋给 cat
stack = _Stack  # 将 _Stack 类的引用赋给 stack
```