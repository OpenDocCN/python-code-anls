# `.\pytorch\torch\_inductor\index_propagation.py`

```py
# mypy: allow-untyped-defs
"""This file implements the IndexPropagation ops handler, which wraps an
underlying handler to add a limited form of constant propagation, as well as
propagation of sympy expressions downstream of ops.index_expr calls.

For example, say we have the IR:

   tmp0 = ops.index_expr(x, torch.int32)
   tmp1 = ops.constant(2, torch.int32)
   tmp2 = ops.mul(tmp0, tmp1)
   tmp3 = ops.indirect_indexing(tmp2, x_size)
   tmp4 = ops.load("buf0", tmp3)

The underlying handler would just see:

   ops.load("buf0", x * 2)

This is limited by the set of operators handled in the sympy expression
printers. So simple operations like minimum and maximum cannot be translated to
SymPy expressions yet, despite sympy.Min and sympy.Max existing.

"""

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload, Tuple, Union
from typing_extensions import TypeAlias

import sympy  # Importing SymPy library

import torch  # Importing Torch library
from torch._prims_common import dtype_to_type, is_integer_dtype  # Importing specific functions from Torch
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where  # Importing functions from Torch Sympy utilities
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges  # Importing specific functions from Torch Sympy utilities
from .utils import generate_assert  # Importing a specific function from local utilities
from .virtualized import V  # Importing V from local virtualized module

_ExprType = Union[sympy.Expr, float, int, bool]  # Defining a type alias for SymPy expression or basic types


def _is_constant(val: _ExprType):
    """Check if the given value is a constant.

    Args:
        val: The value to check.

    Returns:
        bool: True if the value is a constant, False otherwise.
    """
    if isinstance(val, sympy.Basic):
        return val.is_number
    return isinstance(val, (int, float, bool))


def upper_bound(val: _ExprType):
    """Calculate the upper bound of the given value.

    Args:
        val: The value for which to calculate the upper bound.

    Returns:
        Union[sympy.Expr, float, int, bool]: The upper bound of the value.
    """
    return bound_sympy(val).upper if isinstance(val, sympy.Expr) else val


@dataclass
class TypedExpr:
    """A SymPy expression with associated type.

    Attributes:
        expr (_ExprType): The SymPy expression or basic type value.
        dtype (torch.dtype): The Torch data type associated with the expression.
    """

    expr: _ExprType
    dtype: torch.dtype

    def is_constant(self):
        """Check if the expression is a constant.

        Returns:
            bool: True if the expression is a constant, False otherwise.
        """
        return _is_constant(self.expr)

    def __post_init__(self):
        """Initialize the object after other attributes have been set.

        Converts the expression to the appropriate data type if it is a constant.
        """
        if _is_constant(self.expr):
            self.expr = dtype_to_type(self.dtype)(self.expr)


class SymPyOps:
    """An ops handler where all IR values are SymPy expressions.

    When a value cannot be represented as a SymPy expression, the method is
    either not defined, or returns NotImplemented.
    """

    @staticmethod
    def identity(value: Any) -> Any:
        """Return the input value as it is.

        Args:
            value: The input value.

        Returns:
            Any: The same input value.
        """
        return value

    @staticmethod
    def constant(value: Union[int, float, bool], dtype: torch.dtype) -> TypedExpr:
        """Create a constant SymPy expression with a specified data type.

        Args:
            value: The constant value.
            dtype: The Torch data type.

        Returns:
            TypedExpr: A TypedExpr object representing the constant value.
        """
        return TypedExpr(value, dtype)

    @staticmethod
    def index_expr(value: Union[sympy.Expr, int], dtype: torch.dtype) -> TypedExpr:
        """Create a SymPy expression representing an indexed value.

        Args:
            value: The SymPy expression or integer index value.
            dtype: The Torch data type.

        Returns:
            TypedExpr: A TypedExpr object representing the indexed value.
        """
        return TypedExpr(value, dtype)

    @staticmethod
    def to_dtype(
        value: TypedExpr, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None
    ) -> TypedExpr:
        """Convert a TypedExpr object to a different Torch data type.

        Args:
            value: The TypedExpr object to convert.
            dtype: The target Torch data type.
            src_dtype: Optional source Torch data type if available.

        Returns:
            TypedExpr: A new TypedExpr object with the converted data type.
        """
        return TypedExpr(value.expr, dtype)

    @staticmethod
    def abs(x: TypedExpr) -> TypedExpr:
        """Compute the absolute value of a TypedExpr.

        Args:
            x: The TypedExpr object to compute the absolute value of.

        Returns:
            TypedExpr: A new TypedExpr object representing the absolute value.
        """
        return TypedExpr(abs(x.expr), x.dtype)  # type: ignore[arg-type]

    @staticmethod
    def square(x: TypedExpr) -> TypedExpr:
        """Compute the square of a TypedExpr.

        Args:
            x: The TypedExpr object to compute the square of.

        Returns:
            TypedExpr: A new TypedExpr object representing the square.
        """
        return TypedExpr(x.expr * x.expr, x.dtype)

    @staticmethod
    # 定义一个静态方法，用于执行两个 TypedExpr 类型对象的加法，并返回一个新的 TypedExpr 对象
    def add(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # 根据输入的两个表达式的数据类型，推断出结果的数据类型
        result_type = torch.promote_types(x.dtype, y.dtype)
        # 返回一个新的 TypedExpr 对象，包含两个表达式相加的结果以及推断出的数据类型
        return TypedExpr(x.expr + y.expr, result_type)

    # 定义一个静态方法，用于执行两个 TypedExpr 类型对象的减法，并返回一个新的 TypedExpr 对象
    @staticmethod
    def sub(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # 根据输入的两个表达式的数据类型，推断出结果的数据类型
        result_type = torch.promote_types(x.dtype, y.dtype)
        # 返回一个新的 TypedExpr 对象，包含两个表达式相减的结果以及推断出的数据类型
        return TypedExpr(x.expr - y.expr, result_type)

    # 定义一个静态方法，用于执行两个 TypedExpr 类型对象的乘法，并返回一个新的 TypedExpr 对象
    @staticmethod
    def mul(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # 根据输入的两个表达式的数据类型，推断出结果的数据类型
        result_type = torch.promote_types(x.dtype, y.dtype)
        # 返回一个新的 TypedExpr 对象，包含两个表达式相乘的结果以及推断出的数据类型
        return TypedExpr(x.expr * y.expr, result_type)

    # 定义一个静态方法，用于对一个 TypedExpr 类型对象进行取负操作，并返回一个新的 TypedExpr 对象
    @staticmethod
    def neg(x: TypedExpr) -> TypedExpr:
        # 返回一个新的 TypedExpr 对象，包含原始表达式取负的结果以及相同的数据类型
        return TypedExpr(-x.expr, x.dtype)

    # 定义一个静态方法，用于执行两个 TypedExpr 类型对象的整数除法，并返回一个新的 TypedExpr 对象
    @staticmethod
    def floordiv(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # 根据输入的两个表达式的数据类型，推断出结果的数据类型
        result_type = torch.promote_types(x.dtype, y.dtype)
        # 如果结果数据类型不是整数类型，则返回 NotImplemented
        if not is_integer_dtype(result_type):
            return NotImplemented
        
        # 返回一个新的 TypedExpr 对象，包含整数除法 FloorDiv(x.expr, y.expr) 的结果以及推断出的数据类型
        return TypedExpr(FloorDiv(x.expr, y.expr), result_type)

    # 定义一个静态方法，用于执行两个 TypedExpr 类型对象的取模运算，并返回一个新的 TypedExpr 对象或 NotImplemented
    @staticmethod
    def mod(x: TypedExpr, y: TypedExpr) -> Optional[TypedExpr]:
        # 根据输入的两个表达式的数据类型，推断出结果的数据类型
        result_type = torch.promote_types(x.dtype, y.dtype)
        # 如果结果数据类型不是整数类型，则返回 NotImplemented
        if not is_integer_dtype(result_type):
            return NotImplemented
        
        # 使用 ModularIndexing 对象执行取模运算，返回一个新的 TypedExpr 对象，包含取模运算的结果以及推断出的数据类型
        result_expr = ModularIndexing(x.expr, sympy.Integer(1), y.expr)
        return TypedExpr(result_expr, result_type)

    # 定义一个静态方法，用于执行两个 TypedExpr 类型对象的求余运算，并返回一个新的 TypedExpr 对象或 NotImplemented
    @staticmethod
    def remainder(x: TypedExpr, y: TypedExpr) -> Optional[TypedExpr]:
        # 根据输入的两个表达式的数据类型，推断出结果的数据类型
        result_type = torch.promote_types(x.dtype, y.dtype)
        # 如果结果数据类型不是整数类型，则返回 NotImplemented
        if not is_integer_dtype(result_type):
            return NotImplemented

        # 将表达式 x.expr 和 y.expr 转换为 sympy 表达式
        x_expr = sympy.sympify(x.expr)
        y_expr = sympy.sympify(y.expr)
        
        # 如果 x_expr 是非负数并且与 y_expr 的正负性相反，则执行取模运算，并返回结果
        if (
            x_expr.is_nonnegative is not None
            and x_expr.is_nonnegative == y_expr.is_positive
        ):
            result_expr = ModularIndexing(x.expr, sympy.Integer(1), y.expr)
            return TypedExpr(result_expr, result_type)
        
        # 否则，返回 NotImplemented
        return NotImplemented

    # 定义一个静态方法，用于执行两个 TypedExpr 类型对象的取最小值操作，并返回一个新的 TypedExpr 对象
    @staticmethod
    def minimum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # 根据输入的两个表达式的数据类型，推断出结果的数据类型
        result_type = torch.promote_types(x.dtype, y.dtype)
        # 返回一个新的 TypedExpr 对象，包含取两个表达式最小值的结果以及推断出的数据类型
        return TypedExpr(sympy.Min(x.expr, y.expr), result_type)

    # 定义一个静态方法，用于执行两个 TypedExpr 类型对象的取最大值操作，并返回一个新的 TypedExpr 对象
    @staticmethod
    def maximum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # 根据输入的两个表达式的数据类型，推断出结果的数据类型
        result_type = torch.promote_types(x.dtype, y.dtype)
        # 返回一个新的 TypedExpr 对象，包含取两个表达式最大值的结果以及推断出的数据类型
        return TypedExpr(sympy.Max(x.expr, y.expr), result_type)
@dataclass
class IndexPropVar:
    value: Any  # 可以是 IR 值，或者如果 is_symbolic 为 True 则是 TypedExpr 类型的表达式
    is_symbolic: bool = False

    @staticmethod
    def new_symbolic(expr: TypedExpr) -> "IndexPropVar":
        return IndexPropVar(expr, is_symbolic=True)

    def __post_init__(self):
        assert not self.is_symbolic or isinstance(
            self.value, TypedExpr
        ), "Symbolic IndexPropVar must contain a TypedExpr"


IndexPropResult: TypeAlias = Union[IndexPropVar, Tuple["IndexPropResult", ...]]


class IndexPropagation:
    """Ops wrapper that tries to propagate constant and index_expr values through the computation.

    This aims to maximize the compile time simplification possible, and convert
    indirect indexing from arange into normal static indexing.

    """

    def __init__(self, inner: Any, iter_ranges: Dict[sympy.Symbol, sympy.Expr]):
        self._inner = inner
        self.shape_env = V.graph.sizevars.shape_env

        # 初始化变量到范围的映射，包括图形大小变量环境中的和迭代范围中的
        var_to_range = {
            k: ValueRanges(0, upper_bound(v) - 1) for k, v in iter_ranges.items()
        }
        self.var_to_range = tuple(
            itertools.chain(self.shape_env.var_to_range.items(), var_to_range.items())
        )

        # 构建关于迭代范围的公理
        axioms = []
        for x, s in iter_ranges.items():
            axioms.append(0 <= x)
            axioms.append(x < s)
        self.axioms = tuple(axioms) + self.shape_env.get_axioms()

    def materialize_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Any:
        # 从 SymPy 表达式构造新的常数或者索引表达式
        if _is_constant(expr):
            val = dtype_to_type(dtype)(expr)
            return self._inner.constant(val, dtype)
        return self._inner.index_expr(expr, dtype)

    def unwrap(self, a: Union[Any, IndexPropVar]) -> Any:
        if isinstance(a, (list, tuple)):
            return tuple(self.unwrap(v) for v in a)

        if not isinstance(a, IndexPropVar):
            return a

        # 如果可能，优先使用 SymPy 表示
        if a.is_symbolic:
            return self.materialize_expr(a.value.expr, a.value.dtype)

        return a.value

    def wrap(self, a) -> IndexPropResult:
        if isinstance(a, (list, tuple)):
            return tuple(self.wrap(v) for v in a)
        return IndexPropVar(a)

    @overload
    def fallback(
        self,
        name: Literal["indirect_indexing"],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> IndexPropVar:
        ...

    @overload
    def fallback(
        self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> IndexPropResult:
        ...

    def fallback(
        self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> IndexPropResult:
        # 回退到包装处理程序
        new_args = [self.unwrap(a) for a in args]
        new_kwargs = {k: self.unwrap(v) for k, v in kwargs.items()}
        return self.wrap(getattr(self._inner, name)(*new_args, **new_kwargs))
    def propagate_sympy(
        self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> IndexPropResult:
        # 从当前操作调用构建一个新的 SymPy 表达式
        def unwrap(a: Union[Any, IndexPropVar]) -> Any:
            # 如果参数不是 IndexPropVar 类型，则直接返回
            if not isinstance(a, IndexPropVar):
                return a
            # 如果是 IndexPropVar 类型，则返回其值
            return a.value

        # 对参数进行解包，确保不包含 IndexPropVar 类型
        new_args = [unwrap(a) for a in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        # 调用 SymPyOps 中的相应函数，生成新的 SymPy 表达式
        new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
        # 检查新表达式是否有效，不允许浮点数出现在 SymPy 表达式中，但允许浮点数常量的传播
        is_valid_expr = new_expr is not NotImplemented and (
            new_expr.is_constant()
            or new_expr.expr.is_integer
        )
        # 如果表达式无效，则回退到默认处理
        if not is_valid_expr:
            return self.fallback(name, args, kwargs)
        # 创建一个新的符号变量并返回
        return IndexPropVar.new_symbolic(new_expr)

    def __getattr__(self, name: str) -> Callable[..., IndexPropResult]:
        # 动态获取属性的内部函数
        def inner(*args: Any, **kwargs: Any) -> IndexPropResult:
            # 如果 SymPyOps 中没有对应的属性，则回退到默认处理
            if not hasattr(SymPyOps, name):
                return self.fallback(name, args, kwargs)

            # 获取所有参数中的 IndexPropVar 变量
            var_arguments = [
                a
                for a in itertools.chain(args, kwargs.values())
                if isinstance(a, IndexPropVar)
            ]
            # 如果参数中存在非符号变量，则回退到默认处理
            if not all(v.is_symbolic for v in var_arguments):
                return self.fallback(name, args, kwargs)

            # 使用 propagate_sympy 处理符号计算
            return self.propagate_sympy(name, args, kwargs)

        return inner

    def statically_true(self, e):
        """
        给定一些迭代范围，返回一个函数，根据值范围、保护知识和运行时断言确定给定表达式的真假。

        FIXME 我认为这可能不完全正确，因为我们可能无法使用所有的运行时断言
              如果有问题，只需在 `self.axioms` 中使用保护

              处理这个问题的正确方法是有一个全局的 shape_env，在代码中发生运行时断言时将其添加进去。
              然后，在 SimplifyIndexing 中使用它来执行 wrap_expr，并在 CSEProxy.check_bounds 中省略上/下界，
              也适用于间接索引
        """
        # 使用 shape_env 的静态评估方法评估表达式的静态真值
        evaluated = self.shape_env._maybe_evaluate_static(
            e,
            axioms=self.axioms,
            var_to_range=self.var_to_range,
        )
        # 返回表达式的布尔值
        return bool(evaluated)

    def indirect_indexing(
        self, index: Union[Any, IndexPropVar], size: Any, check: bool = True
    ):
        # 间接索引处理函数
        ) -> Any:
        # 如果索引是 IndexPropVar 类型并且是符号的，执行以下逻辑
        if isinstance(index, IndexPropVar) and index.is_symbolic:
            # 如果找到可以转换为直接索引的内容，则执行此操作
            # 仍然需要（可能）包装表达式并添加边界检查
            # 我们希望进行常量折叠，因为我们不允许将内核融合到间接索引中

            # 将索引的表达式转换为 sympy 的表达式对象
            expr = sympy.sympify(index.value.expr)

            # TODO 或许将这个逻辑移到简化索引传递中
            def wrap_expr(expr):
                # 处理正数、负数和混合情况
                if self.statically_true(0 <= expr):
                    return expr
                elif self.statically_true(expr < 0):
                    return expr + size
                else:
                    return Where(expr < 0, expr + size, expr)

            # 有时证明 0 <= expr 比证明更弱的 -size <= expr 更容易
            can_prove_lower = self.statically_true(0 <= expr) or self.statically_true(
                -size <= expr
            )
            can_prove_upper = self.statically_true(expr < size)
            # 包装表达式以处理边界情况
            expr = wrap_expr(expr)
            # 如果生成检查语句，则执行
            if generate_assert(check):
                # 触发回退逻辑来检查边界
                self.fallback(
                    "check_bounds",
                    (expr, size),
                    dict(lower=not can_prove_lower, upper=not can_prove_upper),
                )
            # 返回处理后的表达式
            return expr

        # 对于间接索引的情况
        indirect_var = self.fallback(
            "indirect_indexing", (index, size, check), {}
        ).value
        # 确保间接变量不在 var_to_range 中
        assert (
            indirect_var not in self.var_to_range
        ), f"{indirect_var} should've been created in the fallback."
        # 定义间接索引的范围并添加到 var_to_range 中
        indirect_range = (indirect_var, ValueRanges(0, upper_bound(size) - 1))
        self.var_to_range = self.var_to_range + (indirect_range,)
        # 返回间接索引变量
        return indirect_var
```