# `.\pytorch\torch\export\dynamic_shapes.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import dataclasses  # 用于数据类的装饰器
import inspect  # 用于获取对象信息
import sys  # 提供对解释器的访问和控制
import weakref  # 提供对弱引用对象的支持
from collections import defaultdict  # 提供默认字典的实现
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union  # 引入类型提示

import torch  # 引入 PyTorch 模块
from torch.utils._pytree import (  # 引入 PyTorch 内部的 _pytree 模块中的函数
    _get_node_type,
    BUILTIN_TYPES,
    SUPPORTED_NODES,
    tree_flatten,
    tree_map,
)

from .exported_program import ExportedProgram  # 从当前包中导入 ExportedProgram 类

if TYPE_CHECKING:
    from sympy import Symbol  # 如果是类型检查，导入符号类型

    from torch._guards import Source  # 如果是类型检查，导入 Source 类

    from ..fx.experimental.symbolic_shapes import ShapeEnv, StrictMinMaxConstraint  # 如果是类型检查，导入额外的符号形状模块

__all__ = [  # 声明模块中公开的所有符号
    "Constraint",
    "Dim",
    "dims",
    "dynamic_dim",
    "refine_dynamic_shapes_from_suggested_fixes",
]


class _Dim(type):
    """
    Metaclass for :func:`Dim` types.
    """

    @staticmethod
    def readable(name, min_, max_):
        from torch.utils._sympy.numbers import int_oo  # 导入 PyTorch 内部符号计算库中的 int_oo

        if min_ == 2:
            min_ = None  # 如果最小值为 2，设为 None
        if max_ == int_oo:
            max_ = None  # 如果最大值为正无穷，设为 None
        if min_ is None and max_ is None:
            return f"Dim('{name}')"  # 如果最小值和最大值都为 None，返回简单格式的字符串
        if min_ is None:
            return f"Dim('{name}', max={max_})"  # 如果最小值为 None，返回带最大值的格式化字符串
        if max_ is None:
            return f"Dim('{name}', min={min_})"  # 如果最大值为 None，返回带最小值的格式化字符串
        return f"Dim('{name}', min={min_}, max={max_})"  # 否则返回带最小值和最大值的格式化字符串

    def __add__(cls, other):
        # 实现 Dim 类型对象与整数的加法
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to add {other} to {cls.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return cls._derive(lambda x: x + other)  # 返回加法操作的衍生 Dim 类型对象

    def __radd__(cls, other):
        return cls + other  # 实现反向加法操作

    def __sub__(cls, other):
        # 实现 Dim 类型对象与整数的减法
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to subtract {other} from {cls.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return cls._derive(lambda x: x - other)  # 返回减法操作的衍生 Dim 类型对象

    def __rsub__(cls, other):
        raise NotImplementedError(
            f"Attempted to negate {cls.__name__}. "
            "(Only increasing linear operations with integer coefficients are supported.)"
        )  # 不支持反向减法操作，抛出异常

    def __mul__(cls, other):
        # 实现 Dim 类型对象与整数的乘法
        if type(other) is not int or other <= 0:
            raise NotImplementedError(
                f"Attempted to multiply {other} with {cls.__name__}, where a positive integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return cls._derive(lambda x: x * other)  # 返回乘法操作的衍生 Dim 类型对象

    def __rmul__(cls, other):
        return cls * other  # 实现反向乘法操作

    def _derived_name(cls, fn):
        from sympy import sympify  # 导入 sympy 中的 sympify 函数

        return str(fn(sympify(cls.__name__)))  # 返回衍生名称的字符串表示

    def _derive(cls, fn):
        return _DerivedDim(cls._derived_name(fn), (int,), {"root": cls, "fn": fn})  # 返回衍生 Dim 类型对象
class _StaticDim(_Dim):
    """
    Meta class for static :func:`Dim` types.

    This class is only for setting and checking static dim constraints,
    and the user should never interact with it.
    """

    @property
    def min(self):
        # 返回维度的最小值，这里假设 self.value 是维度的值
        return self.value  # type: ignore[attr-defined]

    @property
    def max(self):
        # 返回维度的最大值，这里假设 self.value 是维度的值
        return self.value  # type: ignore[attr-defined]


class _DerivedDim(_Dim):
    """
    Metaclass for derived :func:`Dim` types.

    Currently we only support increasing linear expressions with integer coefficients.
    In other words, a derived Dim can always be written in the form Ax + B, where
    x is a regular Dim (i.e., non-derived Dim), A and B are integers, and A is positive.
    (In particular, the latter ensures that x < y => Ax + B < Ay + B.)
    These restrictions on the form of derived Dims makes the metatheory simpler: e.g.,
    it simplifies computing ranges for derived Dims, solving for underlying regular Dims,
    deciding equalities between derived Dims, and so on.

    The function lambda x: Ax + B is expressed by `fn`, where x is a normal Dim, `root`.
    The range of a derived Dim is computed by mapping `fn` over the range of its `root`.
    """

    @property
    def min(self):
        # 假设 self.fn 是一个递增函数
        # TODO(avik): 使用 sympy 进行值范围分析是否更好？
        from sympy import Integer

        from torch.utils._sympy.numbers import int_oo

        if self.root.min is -int_oo:  # type: ignore[attr-defined]
            return -int_oo  # 因为是递增函数，所以不需要 fn

        _min_symint = self.fn(Integer(self.root.min))  # type: ignore[attr-defined]
        root = self.root  # type: ignore[attr-defined]
        assert _min_symint >= 0, (
            f"Expected derived min value of {self.__name__} to be >= 0. "
            f"Please specify an appropriate min value for {root.__name__} "
            f"(currently {root.min})."
        )
        return int(_min_symint)

    @property
    def max(self):
        # 假设 self.fn 是一个递增函数
        # TODO(avik): 使用 sympy 进行值范围分析是否更好？
        from sympy import Integer

        from torch.utils._sympy.numbers import int_oo

        if self.root.max is int_oo:  # type: ignore[attr-defined]
            return int_oo  # 因为是递增函数，所以不需要 fn

        _max_symint = self.fn(Integer(self.root.max))  # type: ignore[attr-defined]
        root = self.root  # type: ignore[attr-defined]
        assert _max_symint <= sys.maxsize - 1, (
            f"Expected derived max value of {self.__name__} to be <= {sys.maxsize - 1}. "
            f"Please specify an appropriate max value for {root.__name__} "
            f"(currently {root.max})."
        )
        return int(_max_symint)
    def _derive(self, fn):
        # 定义一个内部方法 _derive，接受一个函数 fn 作为参数
        # 我们支持嵌套，例如，2*dim + 1。
        # 这是通过在相同根上组合操作来实现的。
        # 因此，根始终是常规的维度（即非衍生的维度）。
        return _DerivedDim(
            self._derived_name(fn),  # 调用对象的 _derived_name 方法，传入 fn 函数作为参数
            (int,),  # 返回的 _DerivedDim 实例的第二个参数，这里是一个包含 int 类型的元组
            {"root": self.root, "fn": lambda x: fn(self.fn(x))},  # type: ignore[attr-defined]
            # 返回的 _DerivedDim 实例的第三个参数，一个包含 "root" 和 "fn" 键的字典，
            # "root" 对应当前对象的根属性，"fn" 对应一个 lambda 函数，它对输入进行两次函数调用：首先调用 self.fn(x)，然后将结果传递给 fn 函数
        )
def Dim(name: str, *, min: Optional[int] = None, max: Optional[int] = None):
    """
    :func:`Dim` constructs a type analogous to a named symbolic integer with a range.
    It can be used to describe multiple possible values of a dynamic tensor dimension.
    Note that different dynamic dimensions of the same tensor, or of different tensors,
    can be described by the same type.

    Args:
        name (str): Human-readable name for debugging.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        A type that can be used in dynamic shape specifications for tensors.
    """
    # Importing infinity value from torch utilities
    from torch.utils._sympy.numbers import int_oo

    # Assigning default values if min or max are None
    _min = 0 if min is None else min
    _max = int_oo if max is None else max
    # Asserting that max is greater than min, raising an error if not
    assert _max > _min, f"Cannot create Dim with inconsistent min={min}, max={max}"
    # Creating a new Dim object with given parameters
    dim = _Dim(name, (int,), {"min": _min, "max": _max})
    # Setting the module name for the created Dim object
    dim.__module__ = getattr(
        inspect.getmodule(inspect.stack()[1][0]), "__name__", "__main__"
    )
    # Returning the created Dim object
    return dim


def dims(*names: str, min: Optional[int] = None, max: Optional[int] = None):
    """
    Util to create multiple :func:`Dim` types.
    """
    # Creating a tuple of Dim objects for each name provided
    return tuple(Dim(name, min=min, max=max) for name in names)


@dataclasses.dataclass
class _ConstraintTarget:
    """
    This represents input tensor dimensions.  Don't create this
    class directly; instead, use :func:`dynamic_dim`.
    """

    w_tensor: Any  # weakref to torch.Tensor
    # TODO: We don't need t_id; we can get it off of w_tensor
    t_id: int
    dim: int


class _ConstraintFactory(type):
    """
    Metaclass that ensures a private constructor for :class:`_Constraint`
    """

    def __call__(cls, *args, **kwargs):
        # Preventing direct instantiation of _Constraint
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor. "
            f"Please use torch.export.dynamic_dim() to create one"
        )

    @classmethod
    def _create(
        cls, w_tensor, t_id, dim, constraint_range, shared=None, debug_name=None
    ):
        # Creating a _Constraint object using private constructor
        return super().__call__(
            w_tensor, t_id, dim, constraint_range, shared, debug_name
        )


def _create_constraint(
    w_tensor, t_id, dim, constraint_range, shared=None, debug_name=None
):
    # Creating a _Constraint object using _ConstraintFactory's _create method
    return _Constraint._create(
        w_tensor, t_id, dim, constraint_range, shared, debug_name
    )


@dataclasses.dataclass
class _Constraint(_ConstraintTarget, metaclass=_ConstraintFactory):
    """

    .. warning::
        Do not construct :class:`_Constraint` directly, use :func:`dynamic_dim` instead.

    This represents constraints on input tensor dimensions, e.g., requiring
    them to be fully polymorphic or within some range.

    """

    # NOTE(avik): In the future, this could be Union[StrictMinMaxConstraint, <other kinds>]
    constraint_range: "StrictMinMaxConstraint"
    # Represent that `constraint_range` is shared with another _ConstraintTarget, which
    # 表示一个可选的约束目标，通常用于与另一个动态维度指定相等性的情况
    shared: Optional[_ConstraintTarget] = None
    # 表示一个可选的调试名称，通常用于调试目的
    debug_name: Optional[str] = None

    def _clone_with_range(self, lower=0, upper=None):
        # 从 torch.fx.experimental.symbolic_shapes 模块中局部导入 StrictMinMaxConstraint
        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
        # 从 torch.utils._sympy.numbers 模块中局部导入 int_oo
        from torch.utils._sympy.numbers import int_oo
        # 从 torch.utils._sympy.value_ranges 模块中局部导入 ValueRanges

        # 如果未提供上限值，则使用 int_oo 作为默认的无穷大上限
        if upper is None:
            upper = int_oo

        # 创建 StrictMinMaxConstraint 对象，用于表示约束的范围
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & ValueRanges(lower=lower, upper=upper),
            warn_only=False,
        )
        # 调用 _create_constraint 方法创建一个新的约束对象，基于当前约束的各个属性
        return _create_constraint(
            self.w_tensor,
            self.t_id,
            self.dim,
            constraint_range,
            self.shared,
            self.debug_name,
        )

    def __ge__(self, lower):
        # 实现大于等于运算符的重载，返回一个基于新下限的克隆约束对象
        return self._clone_with_range(lower=lower)

    def __gt__(self, lower):
        # 实现大于运算符的重载，返回一个基于新下限加一的克隆约束对象
        return self._clone_with_range(lower=lower + 1)

    def __le__(self, upper):
        # 实现小于等于运算符的重载，返回一个基于新上限的克隆约束对象
        return self._clone_with_range(upper=upper)

    def __lt__(self, upper):
        # 实现小于运算符的重载，返回一个基于新上限减一的克隆约束对象
        return self._clone_with_range(upper=upper - 1)

    def __bool__(self):
        # 抛出 TypeError 异常，因为不支持复合表达式如 a <= x <= b
        raise TypeError(
            "Cannot determine truth value of _Constraint. "
            "If you are trying to combine _Constraint's with logical connectives, "
            "you can specify them separately instead."
        )

    @property
    def serializable_spec(self):
        # 返回一个序列化兼容格式的约束规范，以便在图模块中保存而不破坏模块的序列化
        # 保存的约束将直接用于后期导出的过程，将约束转换为运行时断言
        # 保存的约束不会出现在序列化的模块中
        # TODO: 需要一种更好的方式，目前我们使用 't_id' 来映射约束，这不是很可靠
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
        }
    def __eq__(self, other):
        # 如果 `other` 不是 `_Constraint` 的实例，则抛出类型错误
        if not isinstance(other, _Constraint):
            raise TypeError(
                "A dynamic dim can be specified equal only to another dynamic dim. "
                f"Equality with {type(other)} is not supported."
            )

        # 从本地导入 `StrictMinMaxConstraint` 类
        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint

        # 创建一个新的约束范围对象 `constraint_range`
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & other.constraint_range.vr,  # 计算两个约束范围的交集
            warn_only=False,  # 设置警告标志为 `False`
        )

        # 确定用于调试的名称 `debug_name`
        if self.debug_name is None:
            debug_name = other.debug_name  # 如果当前对象的调试名称为空，则使用 `other` 的调试名称
        else:
            assert other.debug_name is None or self.debug_name == other.debug_name
            debug_name = self.debug_name  # 否则确保 `other` 的调试名称为空或与当前对象相同，并使用当前对象的调试名称

        # 创建并返回一个新的约束对象
        return _create_constraint(
            self.w_tensor,  # 当前对象的权重张量
            self.t_id,  # 当前对象的 ID
            self.dim,  # 当前对象的维度
            constraint_range,  # 约束范围对象
            shared=_ConstraintTarget(other.w_tensor, other.t_id, other.dim),  # 共享的约束目标对象
            debug_name=debug_name,  # 调试名称
        )
@dataclasses.dataclass
class _PhantomRoot:
    """
    This represents the root of a derived Dim where the root does not directly
    specify the shape of any input dimension, but the derived Dim does.

    e.g., the input shapes 2*dim and dim + 1 are related via a "phantom" dim.

    The fields `name`, `constraint_range`, and `val` carried by a phantom root
    help create a symbol for it. Any derived dims with this phantom root are
    backed by expressions over this symbol.
    """

    name: str                          # 字符串字段，表示幻影根的名称
    constraint_range: "StrictMinMaxConstraint"  # 严格最小最大约束对象，表示幻影根的约束范围
    val: int                           # 整数字段，表示幻影根的值


@dataclasses.dataclass
class _DerivedConstraint(_ConstraintTarget):
    """
    This represents a derived Dim, whose root is either a regular constraint target
    (which directly specifies the shape of some input dimension) or a phantom root
    (which does so indirectly).
    """

    # NOTE: This is not currently a subclass of _Constraint because we do not support
    # `shared` for derived `Dim`s. Indeed, sharing is a necessary concept only for
    # legacy constraints based on `dynamic_dim`: equality can be expressed simply by
    # reusing the same (derived or normal) `Dim`.
    root: Union[_ConstraintTarget, _PhantomRoot]  # 根对象，可以是约束目标或幻影根
    fn: Callable                        # 可调用对象，描述约束的函数
    constraint_range: "StrictMinMaxConstraint"  # 严格最小最大约束对象，描述约束的范围
    debug_name: Optional[str] = None    # 可选的调试名称字符串

    @property
    def shared(self):
        # Some code paths expect a union of _Constraint and _DerivedConstraint.
        # Thus we expose a `shared` field that is always None.
        # TODO(avik): clean this up
        return None                     # 返回空值，表示共享字段为None

    @property
    def serializable_spec(self):
        # same as _Constraint.serializable_spec
        return {
            "t_id": self.t_id,         # 张量ID
            "dim": self.dim,           # 维度
            "min": self.constraint_range.vr.lower,   # 约束范围的最小值
            "max": self.constraint_range.vr.upper,   # 约束范围的最大值
        }


Constraint = Union[_Constraint, _DerivedConstraint]  # 定义约束类型为约束目标或派生约束


def dynamic_dim(t: torch.Tensor, index: int, debug_name: Optional[str] = None):
    """
    .. warning::
        (This feature is DEPRECATED. See :func:`Dim` instead.)

    :func:`dynamic_dim` constructs a :class:`_Constraint` object that describes the dynamism of
    a dimension ``index`` of tensor ``t``. :class:`_Constraint` objects should be passed to
    ``constraints`` argument of :func:`export`.

    Args:
        t (torch.Tensor): Example input tensor that have dynamic dimension size(s)
        index (int): Index of dynamic dimension

    Returns:
        A :class:`_Constraint` object that describes shape dynamism. It can be passed to :func:`export` so
        that :func:`export` does not assume static size of specified tensor, i.e. keeping it dynamic
        as a symbolic size rather than specializing according to size of example tracing input.

    Specifically :func:`dynamic_dim` can be used to express following types of dynamism.
    """
    # 引入 torch 库中的随机数生成功能
    t0 = torch.rand(2, 3)
    t1 = torch.rand(3, 4)

    # 指定 t0 的第一维可以是动态大小，而不总是静态大小为 2
    constraints = [dynamic_dim(t0, 0)]
    # 调用 export 函数导出模型，传入 t0 和 t1 作为参数，并附带约束条件
    ep = export(fn, (t0, t1), constraints=constraints)

    # 引入 torch 库中的随机数生成功能
    t0 = torch.rand(10, 3)
    t1 = torch.rand(3, 4)

    # 指定 t0 的第一维可以是动态大小，且具有下限为 5（包含）
    # 指定 t1 的第二维可以是动态大小，且具有下限为 2（不包含）
    constraints = [
        dynamic_dim(t0, 0) >= 5,
        dynamic_dim(t1, 1) > 2,
    ]
    # 调用 export 函数导出模型，传入 t0 和 t1 作为参数，并附带约束条件
    ep = export(fn, (t0, t1), constraints=constraints)

    # 引入 torch 库中的随机数生成功能
    t0 = torch.rand(10, 3)
    t1 = torch.rand(3, 4)

    # 指定 t0 的第一维可以是动态大小，且具有上限为 16（包含）
    # 指定 t1 的第二维可以是动态大小，且具有上限为 8（不包含）
    constraints = [
        dynamic_dim(t0, 0) <= 16,
        dynamic_dim(t1, 1) < 8,
    ]
    # 调用 export 函数导出模型，传入 t0 和 t1 作为参数，并附带约束条件
    ep = export(fn, (t0, t1), constraints=constraints)

    # 引入 torch 库中的随机数生成功能
    t0 = torch.rand(10, 3)
    t1 = torch.rand(3, 4)

    # 指定 t0 的第二维大小始终等于 t1 的第一维大小
    constraints = [
        dynamic_dim(t0, 1) == dynamic_dim(t1, 0),
    ]
    # 调用 export 函数导出模型，传入 t0 和 t1 作为参数，并附带约束条件
    ep = export(fn, (t0, t1), constraints=constraints)

    # 混合匹配上述所有类型的约束，只要它们没有表达冲突的要求

    # 引入 torch 库中的动态维度相关异常及其类型
    from torch._dynamo.exc import UserError, UserErrorType

    # 如果输入不是 torch.Tensor 类型，则抛出用户异常
    if not isinstance(t, torch.Tensor):
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            f"Expected tensor as input to dynamic_dim but got {type(t)}",
        )

    # 如果 tensor 的维度小于 1，则抛出用户异常
    if t.dim() < 1:
        raise UserError(
            UserErrorType.DYNAMIC_DIM, "Cannot mark 0-dimension tensors to be dynamic"
        )

    # 如果传入的索引超出了 tensor 的维度范围，则抛出用户异常
    if index >= t.dim():
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            f"Expected the dimension passed to dynamic_dim to be in the range [0:{t.dim()-1}]"
            f" but got {index}, which is out of bounds for the given tensor.",
        )

    # 在本地导入 sympy 库的相关模块

    # 从 torch.fx.experimental.symbolic_shapes 中引入 StrictMinMaxConstraint
    from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
    # 从 torch.utils._sympy.numbers 中引入 int_oo（正无穷大）
    from torch.utils._sympy.numbers import int_oo
    # 从 torch.utils._sympy.value_ranges 中引入 ValueRanges
    from torch.utils._sympy.value_ranges import ValueRanges

    # 调用 _create_constraint 函数创建约束条件并返回结果

    return _create_constraint(
        weakref.ref(t),
        id(t),
        index,
        StrictMinMaxConstraint(vr=ValueRanges(lower=0, upper=int_oo), warn_only=False),
        debug_name=debug_name,
    )
def _process_equalities(
    constraint: Constraint,
    get_sources: Callable[[int, int], List["Source"]],
    shape_env: "ShapeEnv",
    source_pairs: List[Tuple["Source", "Source"]],
    derived_equalities: List[Tuple["Source", Union["Source", "Symbol"], Callable]],
    phantom_symbols: Dict[str, "Symbol"],
):
    """
    Updates `source_pairs`, `derived_equalities`, and `phantom_symbols` (which become
    fields of `EqualityConstraint`) based on a given input `constraint`.
    """

    # 获取与约束条件相关的源和其他源列表
    source, *other_sources = get_sources(constraint.t_id, constraint.dim)
    
    # 当 t.size()[dim] 映射到 src0, src1, ..., srcN 时，我们添加约束条件使得 src0 等于 src1, ..., srcN
    source_pairs.extend((source, other_source) for other_source in other_sources)
    
    if not isinstance(constraint, _DerivedConstraint):
        if constraint.shared is not None:
            # 此外，当 t.size()[dim] 被指定等于 t'.size()[dim']，且 t'.size()[dim'] 映射到 src1', ..., srcN' 时，
            # 我们还添加约束条件使得 src0 等于 src1', ..., srcN'
            other_sources = get_sources(constraint.shared.t_id, constraint.shared.dim)
            source_pairs.extend(
                (source, other_source) for other_source in other_sources
            )
    else:
        # 根据 _DerivedConstraint 的根进行分支
        if not isinstance(constraint.root, _PhantomRoot):
            # 根可能指向输入源
            root = get_sources(constraint.root.t_id, constraint.root.dim)[0]  # type: ignore[assignment]
        else:
            # 或者根可能指向一个幻象符号
            if constraint.root.name in phantom_symbols:
                root = phantom_symbols[constraint.root.name]  # type: ignore[assignment]
            else:
                # 根据 _PhantomRoot 在形状环境中创建一个幻象符号
                root = shape_env.create_symbol(
                    val=constraint.root.val,
                    source=torch._dynamo.source.ConstantSource(constraint.root.name),
                    dynamic_dim=torch.fx.experimental.symbolic_shapes.DimDynamic.DYNAMIC,
                    constraint_dim=constraint.root.constraint_range,
                )
                phantom_symbols[constraint.root.name] = root  # type: ignore[assignment]

        fn = constraint.fn
        # 衍生的等式 (source, root, fn) 非正式地对应于 source = fn(root)
        # 这里 source 描述一个输入，root 可能描述另一个输入或者一个幻象符号
        derived_equalities.append((source, root, fn))


def _tree_map(
    func: Callable[..., Any],
    tree: Any,
    *dynamic_shapes: Any,
) -> Any:
    """
    Customized tree_map for mapping pytrees to dynamic_shapes.

    For built-in types (e.g., standard collections) this behaves exactly like tree_map.

    OTOH for a user-defined class C registered with pytree, we cannot assume that a C
    """
    def is_leaf(t):
        # 判断节点 t 是否为叶子节点
        # BUILTIN_TYPES 是 SUPPORTED_NODES 的子集，后者包含所有在 pytree 中注册的类型。
        # 不在 BUILTIN_TYPES 中的类型包括原始类型 (int, float, str, bool, None, torch.Tensor)，
        # 这些类型不在 SUPPORTED_NODES 中；而在 pytree 中注册的用户定义类则在其中。
        return _get_node_type(t) not in BUILTIN_TYPES
    
    def f(t, *dynamic_shapes):
        typ = _get_node_type(t)
        # 如果 typ 不在 BUILTIN_TYPES 中
        if typ in SUPPORTED_NODES:
            # 表明 typ 是在 pytree 中注册的用户定义类，
            # 因此需要展开并递归处理
            return tree_map(
                f,
                SUPPORTED_NODES[typ].flatten_fn(t)[0],
                *dynamic_shapes,
                is_leaf=is_leaf,
            )
        else:
            # typ 是原始类型之一，直接应用 func 处理
            return func(t, *dynamic_shapes)
    
    # 对输入的 pytree 应用 func 函数
    # 以及可能的动态形状 dynamic_shapes 进行匹配
    # 返回一个输出 pytree，将 func 应用到每个 (int, float, str, bool, None, torch.Tensor) 类型的值上
    return tree_map(f, tree, *dynamic_shapes, is_leaf=is_leaf)
def _combine_args(f, args, kwargs, _is_torch_jit_trace=False):
    # 将参数 args 和 kwargs 按照函数 f 的签名结构组合起来，这是在调用 f 时发生的情况
    if isinstance(f, ExportedProgram):
        # 如果 f 是 ExportedProgram 类型，则获取其 module
        f = f.module()
    if not _is_torch_jit_trace:
        # 如果不是 Torch JIT 跟踪模式
        signature = (
            inspect.signature(f.forward)
            if isinstance(f, torch.nn.Module)
            else inspect.signature(f)
        )
        kwargs = kwargs if kwargs is not None else {}
        # 使用函数签名绑定传入的 args 和 kwargs，获取参数字典
        return signature.bind(*args, **kwargs).arguments
    # 返回 args
    return args


class ShapesCollection:
    """
    Builder for dynamic_shapes.
    Used to assign dynamic shape specifications to tensors that appear in inputs.
    
    Example::
        args = ({"x": tensor_x, "others": [tensor_y, tensor_z]})

        dim = torch.export.Dim(...)
        dynamic_shapes = torch.export.ShapesCollection()
        dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
        dynamic_shapes[tensor_y] = {0: dim * 2}
        # This is equivalent to the following (now auto-generated):
        # dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [{0: dim * 2}, None]}

        torch.export(..., args, dynamic_shapes=dynamic_shapes)
    """

    def __init__(self):
        # 初始化一个空字典用于存储 tensor 对象的形状信息
        self._shapes = {}

    def __setitem__(self, t, shape):
        assert isinstance(
            t, torch.Tensor
        ), f"Cannot assign shape to non-tensor type {type(t)}"
        # 断言 t 是 torch.Tensor 类型，否则抛出异常
        # TODO(avik): check that shape is indeed a Shape
        t_id = id(t)
        if t_id in self._shapes:
            _shape = self._shapes[t_id]
            assert (
                shape == _shape
            ), f"Shapes assigned to tensor do not match: expected {_shape}, got {shape}"
        else:
            self._shapes[id(t)] = shape

    def __getitem__(self, t):
        # 根据 tensor 对象 t 的 id 获取其对应的形状信息
        t_id = id(t)
        if t_id in self._shapes:
            return self._shapes[t_id]
        else:
            return None

    def __len__(self):
        # 返回当前存储的 tensor 对象的形状信息的数量
        return len(self._shapes)

    def dynamic_shapes(self, m, args, kwargs=None):
        """
        Generate dynamic_shapes.
        """
        # 存储找到形状信息的 tensor 对象的 id
        t_ids = set()

        def find_shape(t):
            # 根据 tensor 对象 t 的 id 查找其形状信息，如果找不到返回 None
            t_id = id(t)
            if t_id in self._shapes:
                t_ids.add(t_id)
                return self._shapes[t_id]
            else:
                return None

        # 组合函数 m 的参数 args 和 kwargs
        combined_args = _combine_args(m, args, kwargs)
        # 使用 find_shape 函数遍历 combined_args 中的所有 tensor 对象，获取其形状信息
        dynamic_shapes = _tree_map(find_shape, combined_args)
        # 如果存在部分被分配形状信息但未在参数中找到的 tensor 对象，则抛出异常
        if any(t_id not in t_ids for t_id in self._shapes):
            raise ValueError(
                "Some tensors that were assigned shapes were not found in args. "
                "Maybe such tensors were copied when passing them as args? "
                "Maybe such tensors are contained in classes that were not registered with pytree?"
            )
        # 返回动态形状信息
        return dynamic_shapes


def _process_dynamic_shapes(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    # dynamic_shapes 是一个可选类型的变量，可以是字典、元组或列表中的任意一种，或者是 None
    _is_torch_jit_trace=False,
    # _is_torch_jit_trace 是一个布尔类型的变量，默认值为 False
    # 返回类型为 Optional[List[Constraint]] 的函数定义，表示返回一个约束条件列表或空值
    ) -> Optional[List[Constraint]]:
    # 从 torch._dynamo.exc 导入 UserError 和 UserErrorType
    from torch._dynamo.exc import UserError, UserErrorType

    # 如果 dynamic_shapes 为 None 或空列表，则返回 None
    if dynamic_shapes is None or len(dynamic_shapes) == 0:
        return None

    # symbols 是一个字典，用于存储表示输入形状维度的 Dim 名称到约束条件列表的映射
    symbols: Dict[str, List[Constraint]] = defaultdict(list)
    
    # phantom_roots 是一个字典，用于跟踪不直接表示输入形状维度的根节点
    phantom_roots: Dict[str, _PhantomRoot] = {}
    
    # derived_constraints_with_phantom_root 是一个列表，用于存储带有幻影根的衍生约束
    derived_constraints_with_phantom_root: List[_DerivedConstraint] = []

    # bounds 是一个字典，用于存储维度名称到其最小值和最大值元组的映射
    bounds: Dict[str, Tuple[int, int]] = {}

    # 定义函数 check_same_bounds，用于检查同一符号维度的不同定义是否一致
    def check_same_bounds(dim):
        # 如果 dim.__name__ 在 symbols 字典中已存在
        if dim.__name__ in symbols:
            # 获取该维度名称的最小值和最大值
            min_, max_ = bounds[dim.__name__]
            # 检查当前维度的最小值和最大值是否与已存储的值不同
            if dim.min != min_ or dim.max != max_:
                # 准备异常信息并抛出 UserError
                this_ = _Dim.readable(dim.__name__, min_, max_)
                that_ = _Dim.readable(dim.__name__, dim.min, dim.max)
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    # 错误信息提示有关同一符号维度不同定义的问题
                    f"Found different definitions {this_} and {that_} "
                    f"for the same symbolic dimension {dim}!",
                )
        else:
            # 如果维度名称不在 symbols 字典中，将其最小值和最大值添加到 bounds 字典中
            bounds[dim.__name__] = (dim.min, dim.max)
    # 定义一个函数用于更新张量的符号约束
    def update_symbols(tensor, shape):
        # 定义一个内部函数，用于创建静态维度对象
        def _create_static_dim(tensor, i, value):
            return _StaticDim(str(value), (int,), {"value": value})

        # 如果 shape 是字典类型，则遍历其键值对
        if isinstance(shape, dict):
            for i, dim in shape.items():
                # 如果维度是整数或者 _Dim 类型的对象
                if isinstance(dim, (int, _Dim)):
                    # 如果维度是整数，则创建静态维度对象
                    if isinstance(dim, int):
                        dim = _create_static_dim(tensor, i, dim)
                    # 检查维度是否具有相同的边界条件
                    check_same_bounds(dim)
                    # 将维度转换为约束，并添加到对应的符号约束列表中
                    constraint = to_constraint(dim, tensor, i)
                    symbols[dim.__name__].append(constraint)
                else:
                    # 如果维度不是整数或 _Dim 对象，并且不为 None，则抛出用户错误异常
                    if dim is not None:
                        raise UserError(
                            UserErrorType.INVALID_INPUT,
                            f"Unexpected item #{i} ({dim}) in dynamic_shape {shape} of Tensor, "
                            "try None instead",
                        )
        # 如果 shape 是 tuple 或者 list 类型，则遍历其中的元素
        elif isinstance(shape, (tuple, list)):
            for i, dim in enumerate(shape):
                # 如果维度是整数或者 _Dim 类型的对象
                if isinstance(dim, (int, _Dim)):
                    # 如果维度是整数，则创建静态维度对象
                    if isinstance(dim, int):
                        dim = _create_static_dim(tensor, i, dim)
                    # 检查维度是否具有相同的边界条件
                    check_same_bounds(dim)
                    # 将维度转换为约束，并添加到对应的符号约束列表中
                    constraint = to_constraint(dim, tensor, i)
                    symbols[dim.__name__].append(constraint)
                else:
                    # 如果维度不是整数或 _Dim 对象，并且不为 None，则抛出用户错误异常
                    if dim is not None:
                        raise UserError(
                            UserErrorType.INVALID_INPUT,
                            f"Unexpected item #{i} ({dim}) in dynamic_shape {shape} of Tensor, "
                            "try None instead",
                        )
        # 如果 shape 不是字典、tuple 或 list 类型，并且不为 None，则抛出用户错误异常
        else:
            if shape is not None:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Unexpected dynamic_shape {shape} of Tensor, " "try None instead",
                )

    # 定义一个函数，用于关联参数的形状和动态形状
    def assoc_shapes(combined_args, dynamic_shapes):
        # 定义一个内部函数，用于关联单个参数的形状和动态形状
        def assoc_shape(t, dynamic_shape):
            # 如果 t 是 torch.Tensor 类型，则更新其符号约束
            if isinstance(t, torch.Tensor):
                update_symbols(t, dynamic_shape)
            else:
                # 如果 t 不是 torch.Tensor 类型，并且 dynamic_shape 不为 None，则抛出用户错误异常
                if dynamic_shape is not None:
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"Cannot associate shape {dynamic_shape} to non-tensor type {type(t)}, "
                        f"expected None",
                    )

        # 使用 _tree_map 函数将 assoc_shape 应用于 combined_args 和 dynamic_shapes 的元素上
        _tree_map(assoc_shape, combined_args, dynamic_shapes)

    # 将函数 f、args 和 kwargs 以及 _is_torch_jit_trace 作为参数，组合成 combined_args
    combined_args = _combine_args(
        f, args, kwargs, _is_torch_jit_trace=_is_torch_jit_trace
    )
    # 如果 dynamic_shapes 不是字典类型，则断言其为 tuple 或 list 类型，并将 combined_args 中的值组成相同类型的对象
    if not isinstance(dynamic_shapes, dict):
        assert isinstance(dynamic_shapes, (tuple, list))
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]
    # 将 combined_args 和 dynamic_shapes 关联起来，更新其符号约束
    assoc_shapes(combined_args, dynamic_shapes)

    # 初始化约束列表
    constraints = []
    # 对 derived_constraints_with_phantom_root 列表中的每个元素进行迭代
    for derived_constraint_with_phantom_root in derived_constraints_with_phantom_root:
        # 获取 phantom_root_name，这是 derived_constraint_with_phantom_root 的根节点的名称
        phantom_root_name = derived_constraint_with_phantom_root.root.name  # type: ignore[union-attr]
        # 检查 phantom_root_name 是否在 symbols 字典中
        if phantom_root_name in symbols:
            # 如果存在对应的输入形状维度与此名称相对应，表示不需要使用虚拟符号
            # 注意(avik)：总体上，我们希望保持根节点作为虚拟符号的不变性，
            # 即它们实际上是“虚拟的”，即它们不能由任何输入源表示。
            # 这在我们决定派生相等性时很重要，因为我们可以将注意力专注于输入源：
            # 决定涉及虚拟符号的派生相等性相比之下是微不足道的。
            derived_constraint_with_phantom_root.root = symbols[phantom_root_name][0]

    # 遍历 symbols 字典中的值
    for dynamic_dims in symbols.values():
        # 检查 dynamic_dims 中的所有元素是否都是 _DerivedConstraint 类型
        if all(
            isinstance(dynamic_dim, _DerivedConstraint) for dynamic_dim in dynamic_dims
        ):
            # 如果是，则将所有约束添加到 constraints 列表中
            constraints.extend(dynamic_dims)
        else:
            # 否则，将 dynamic_dims 中的第一个元素视为主要约束
            primary, *others = dynamic_dims
            # 如果还有其他元素，将它们与主要约束之间的相等性添加到 constraints 中
            if others:
                for other in others:
                    constraints.append(primary == other)  # type: ignore[arg-type]
            else:
                # 如果没有其他元素，只将主要约束添加到 constraints 中
                constraints.append(primary)

    # 返回生成的约束列表 constraints
    return constraints  # type: ignore[return-value]
def _get_dim_name_mapping(
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None]
):
    # 创建空字典，用于存储名称到维度对象的映射关系
    name_to_dim = {}
    # 使用 tree_flatten 函数展开 dynamic_shapes，获取其中的维度对象列表
    for dim in tree_flatten(
        dynamic_shapes,
        is_leaf=lambda x: isinstance(x, _Dim),
    )[0]:
        # 如果维度对象为空或者为整数，则跳过
        if dim is None or isinstance(dim, int):
            continue
        # 将维度对象的名称映射到其本身
        name_to_dim[dim.__name__] = dim
        # 如果维度对象是 DerivedDim 类型，则将其根维度的名称映射到根维度对象
        if isinstance(dim, _DerivedDim):
            name_to_dim[dim.root.__name__] = dim.root  # type: ignore[attr-defined]
    # 返回名称到维度对象的映射字典
    return name_to_dim


def refine_dynamic_shapes_from_suggested_fixes(
    msg: str,
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any]],
) -> Union[Dict[str, Any], Tuple[Any], List[Any]]:
    """
    处理导出动态形状的建议修复，以及/或自动动态形状。

    根据 ConstraintViolation 错误消息和原始动态形状，优化给定的动态形状规范。

    大多数情况下行为是直接的 - 即对于专门化或精化 Dim 范围的建议修复，或者建议修复表明派生关系的情况下，
    新的动态形状规范将会相应地更新。

    例如：
    建议修复：
        dim = Dim('dim', min=3, max=6) -> 这只是精化了 dim 的范围
        dim = 4 -> 这将专门化为一个常数
        dy = dx + 1 -> dy 被指定为独立维度，但实际上与 dx 有关联

    然而，与派生维度相关的建议修复可能会更复杂。
    例如，如果为根维度提供了一个建议修复，则基于根计算新的派生维度值。

    例如：
    dx = Dim('dx')
    dy = dx + 2
    dynamic_shapes = {"x": (dx,), "y": (dy,)}

    建议修复：
        dx = 4  # 专门化将导致 dy 也专门化为 6
        dx = Dim('dx', max=6)  # dy 现在的最大值为 8

    派生维度的建议修复还可以用于表达可整除性约束。
    这涉及创建新的根维度，这些维度不与特定的输入形状相关联。
    在这种情况下，根维度不会直接出现在新规范中，但作为一个维度的根。

    例如：
    建议修复：
        _dx = Dim('_dx', max=1024)  # 这不会出现在返回结果中，但 dx 将会
        dx = 4*_dx  # dx 现在可以被 4 整除，最大值为 4096
    """

    import re  # 导入正则表达式模块

    import sympy  # 导入符号计算模块

    from torch._dynamo.exc import UserError, UserErrorType  # 导入自定义异常和异常类型
    from torch.fx.experimental.symbolic_shapes import _is_supported_equivalence  # 导入符号形状模块中的函数

    try:
        # 提取出错误消息中的建议修复部分
        shape_fixes_msg = msg.split("Suggested fixes:")[1].strip()
    except Exception as exc:
        # 如果无法提取建议修复部分，则抛出用户错误异常
        raise UserError(
            UserErrorType.INVALID_INPUT,
            "Suggested fixes not found in error message given to refine_dynamic_shapes_from_suggested_fixes()",
        ) from exc

    # 构建 shape_fixes 字典，用于存储建议修复
    shape_fixes = {}
    # 将 shape_fixes_msg 按行拆分，并逐行处理
    for fix in shape_fixes_msg.split("\n"):
        # 去除每行两侧的空白字符
        fix = fix.strip()
        
        # 使用正则表达式匹配形如 "name = Dim('dim_name'.*)" 的字符串
        if match := re.match(r"(.*) = Dim\('(.*)'.*\)", fix):
            name = match.group(1)  # 获取变量名
            _min, _max = None, None
            
            # 使用正则表达式匹配形如 ".* = Dim('.*', min=number.*" 的字符串，并提取最小值
            if match_min := re.match(r".* = Dim\('.*', min\=([0-9]+).*\)", fix):
                _min = int(match_min.group(1))  # 获取最小值
            
            # 使用正则表达式匹配形如 ".* = Dim('.*'.*max=number)" 的字符串，并提取最大值
            if match_max := re.match(r".* = Dim\('.*'.*max\=([0-9]+)\)", fix):
                _max = int(match_max.group(1))  # 获取最大值
            
            # 创建 Dim 对象并存储到 shape_fixes 字典中
            shape_fixes[name] = Dim(name, min=_min, max=_max)
        else:
            # 若不符合 Dim 对象的定义，则假设其为 "name = expr" 形式的表达式
            name, expr = fix.split(" = ")
            expr = sympy.sympify(expr)  # 将表达式转换为 SymPy 的表示形式
            
            # 若表达式是整数，则存储为静态整数
            if isinstance(expr, sympy.Number):
                shape_fixes[name] = int(expr)
            else:
                shape_fixes[name] = expr  # 否则存储为关系或派生维度

    # 获取动态形状的名称映射
    name_to_dim = _get_dim_name_mapping(dynamic_shapes)

    # 跟踪派生维度的根
    roots: Set[str] = set()
    for k, c in shape_fixes.items():
        # 断言 c 是 int、_Dim、_DerivedDim 或 sympy.Expr 类型之一
        assert isinstance(c, (int, _Dim, _DerivedDim, sympy.Expr))
        
        # 若 c 是 SymPy 表达式，则检查其是否是支持的等价关系
        if isinstance(c, sympy.Expr):
            assert _is_supported_equivalence(c)  # 断言表达式是支持的等价关系
            shape_fixes[k] = c
            roots.add(str(next(iter(c.free_symbols))))  # 将表达式的第一个自由符号添加到 roots 中
        
        # 若 c 是 _DerivedDim 类型，则将其根名称添加到 roots 中
        if isinstance(c, _DerivedDim):
            roots.add(c.root.__name__)  # 将 _DerivedDim 对象的根名称添加到 roots 中，忽略类型检查

    # 断言 shape_fixes 字典中的所有键都存在于 name_to_dim 中或者是新的根
    for k, c in shape_fixes.items():
        assert k in name_to_dim or k in roots

    # 缓存以避免生成多个派生维度对象
    derived_dim_cache: Dict[str, _DerivedDim] = {}
    # 定义一个函数 apply_fixes，用于处理维度修复
    def apply_fixes(dim, dummy):
        # 如果 dim 为 None 或者是整数，表示不是动态维度，直接返回 dim
        if dim is None or isinstance(dim, int):  # not dynamic
            return dim
        # 如果 dim 是一个函数，并且其名称在 shape_fixes 字典中，直接修复
        elif dim.__name__ in shape_fixes:  # directly fix
            # 获取修复后的维度
            fix = shape_fixes[dim.__name__]
            # 如果 fix 是 sympy.Expr 类型，表示是导出或相关的
            if isinstance(fix, sympy.Expr):  # now derived or related
                # 如果 fix 在 derived_dim_cache 中已经存在，则直接返回缓存的值
                if str(fix) in derived_dim_cache:
                    return derived_dim_cache[str(fix)]
                else:
                    # 获取 fix 中的自由符号
                    symbol = next(iter(fix.free_symbols))
                    # 尝试定位符号所代表的修复值
                    if symbol.name in shape_fixes:  # type: ignore[attr-defined]
                        root = shape_fixes[symbol.name]  # type: ignore[attr-defined]
                    else:
                        assert symbol.name in name_to_dim  # type: ignore[attr-defined]
                        root = name_to_dim[symbol.name]  # type: ignore[attr-defined]
                    # 计算修复值
                    modulus, remainder = sympy.polys.polytools.div(fix, symbol)
                    dim = root
                    if modulus != 1:
                        dim = int(modulus) * dim
                    if remainder != 0:
                        dim = dim + int(remainder)
                    # 将计算后的修复值存入缓存
                    derived_dim_cache[str(fix)] = dim
                    return dim
            else:
                return fix
        # 如果 dim 是 _DerivedDim 类型，并且其 root.__name__ 在 shape_fixes 中
        elif isinstance(dim, _DerivedDim) and dim.root.__name__ in shape_fixes:  # type: ignore[attr-defined]
            # 如果 dim 在 derived_dim_cache 中已经存在，则直接返回缓存的值
            if dim.__name__ in derived_dim_cache:
                return derived_dim_cache[dim.__name__]
            else:  # 根据 root 计算新的导出值
                _dim = dim.fn(shape_fixes[dim.root.__name__])  # type: ignore[attr-defined]
                derived_dim_cache[dim.__name__] = _dim
                return _dim
        # 如果以上情况均不满足，则返回未改变的 dim
        return dim  # unchanged dim

    # 调用 _tree_map 函数，将 apply_fixes 应用到 dynamic_shapes 列表中的每个元素上
    return _tree_map(apply_fixes, dynamic_shapes, dynamic_shapes)
```