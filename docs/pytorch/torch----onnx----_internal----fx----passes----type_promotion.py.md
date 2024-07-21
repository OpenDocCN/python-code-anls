# `.\pytorch\torch\onnx\_internal\fx\passes\type_promotion.py`

```py
# 设置 mypy 选项，允许未标记类型的函数定义
# 所有权：["模块：onnx"]
from __future__ import annotations

# 导入必要的模块
import abc
import dataclasses
import inspect
import logging
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Set, Union

import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback

# 导入 torch._prims_common 模块和 torch._refs 模块
from torch import _prims_common, _refs

# 从 torch._prims_common 模块中导入 ELEMENTWISE_TYPE_PROMOTION_KIND 和 wrappers
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    wrappers as _prims_common_wrappers,
)

# 从 torch._refs 模块中导入 linalg、nn 和 special
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
# 从 torch._refs.nn 模块中导入 functional
from torch._refs.nn import functional as _functional_refs
# 从 torch._subclasses 模块中导入 fake_tensor
from torch._subclasses import fake_tensor
# 从 torch.fx.experimental 模块中导入 proxy_tensor
from torch.fx.experimental import proxy_tensor

# 导入解决 beartype 问题的 torch.fx.node.Node
from torch.fx.node import Node  # noqa: F401

# 从 torch.onnx._internal 模块中导入 _beartype
from torch.onnx._internal import _beartype
# 从 torch.onnx._internal.fx 模块中导入 _pass、diagnostics 和 fx_type_utils
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
# 从 torch.utils 模块中导入 _python_dispatch 和 _pytree

# 获取 logger 对象
logger = logging.getLogger(__name__)

# TODO：将此部分移动到类型工具中
# 标量类型到张量 dtype 的映射
_SCALAR_TYPE_TENSOR_DTYPE_MAP: Mapping[type, torch.dtype] = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float32,
    complex: torch.complex32,
}

# 尝试获取函数的闭包变量
def _try_getclosurevars(func):
    try:
        return inspect.getclosurevars(func)
    except TypeError as e:
        return None

# 数据类，用于存储节点和其输入的类型提升快照
@dataclasses.dataclass
class TypePromotionSnapshot:
    """Type promotion snapshot for a fx node and its inputs.

    Contains the promoted dtype for args and kwargs that needs promoting.
    Contains the expected node output dtype.
    """

    args_dtypes: Mapping[int, torch.dtype]
    """Mapping from arg position to dtype to promote to."""

    kwargs_dtypes: Mapping[str, torch.dtype]
    """Mapping from kwarg name to dtype to promote to."""

    out_dtype: torch.dtype
    """Expected output dtype of the node."""

# 类型提升规则的基类，针对 'torch.ops.{namespace}.{op_name}'
class TypePromotionRule(abc.ABC):
    """Base class for type promotion rule per 'torch.ops.{namespace}.{op_name}'."""

    def __init__(self, namespace: str, op_name: str):
        self.namespace = namespace
        self.op_name = op_name

    # 使此方法也成为抽象方法，因为子类需要重写 __eq__()
    # 覆盖 __eq__() 且未定义 __hash__() 的类将其 __hash__() 隐式设置为 None
    # 参考：https://docs.python.org/3/reference/datamodel.html#object.__hash__
    @abc.abstractmethod
    def __hash__(self) -> int:
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        ...
    def is_valid(self) -> bool:
        """Check if the rule is valid."""
        # 获取指定命名空间下的模块，若不存在则创建
        module = getattr(torch.ops, self.namespace)
        # 获取指定操作名称对应的 Python 函数或操作
        py_op = getattr(module, self.op_name, None)
        # 如果未找到对应的操作，则记录警告并返回 False
        if py_op is None:
            logger.warning(
                "Cannot find op: %s in module: %s", self.op_name, self.namespace
            )
            return False
        # 如果找到的操作不是 OpOverloadPacket 类型，则记录警告并返回 False
        if not isinstance(py_op, torch._ops.OpOverloadPacket):
            logger.warning(
                "Op: torch.ops.%s.%s is not an OpOverloadPacket, got: %s",
                self.namespace,
                self.op_name,
                type(py_op),
            )
            return False

        # 如果以上条件都满足，则返回 True，表示规则有效
        return True

    @abc.abstractmethod
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        """Preview type promotion results for provided set of args and kwargs.

        Returns a TypePromotionSnapshot object that contains the promoted dtypes for
        the arguments and the expected output dtype.
        """
        # 这是一个抽象方法，在子类中实现时，用于预览给定参数和关键字参数的类型提升结果。
        # 返回一个 TypePromotionSnapshot 对象，其中包含参数的提升后的数据类型和预期的输出数据类型。
        ...
class ElementwiseTypePromotionRule(TypePromotionRule):
    """Defines how to perform elementwise type promotion for 'torch.ops.{namespace}.{op_name}'."""

    _USE_OPMATH: bool = False
    """Whether to use opmath to compute the promoted input dtype.
    If used, upcasts will be inserted everywhere for lower precision models.
    Set to False and have torchlib handle upcasts in op implementation internally.
    """

    def __init__(
        self,
        namespace: str,
        op_name: str,
        promote_args_positions: Sequence[int],
        promote_kwargs_names: Sequence[str],
        promotion_kind: _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND,
    ):
        """Constructs a TypePromotionRule for elementwise operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.add'.
            op_name: Name of the op. E.g. 'add' in 'torch.ops.aten.add'.
            promote_args_positions: Positions of args to promote.
            promote_kwargs_names: Names of kwargs to promote.
            promotion_kind: Type promotion kind. Refer to [_prims_common.elementwise_dtypes](https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promote_args_positions = promote_args_positions
        self.promote_kwargs_names = promote_kwargs_names
        self.promotion_kind = promotion_kind

    def __repr__(self):
        return (
            f"ElementwiseTypePromotionRule('{self.namespace}', '{self.op_name}', "
            f"{self.promote_args_positions}, {self.promote_kwargs_names}, {self.promotion_kind})"
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ElementwiseTypePromotionRule):
            return False
        return (
            self.namespace == __value.namespace
            and self.op_name == __value.op_name
            and self.promote_args_positions == __value.promote_args_positions
            and self.promote_kwargs_names == __value.promote_kwargs_names
            and self.promotion_kind == __value.promotion_kind
        )

    def __hash__(self) -> int:
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def _consolidate_input_dtype(
        self, computed_dtype: torch.dtype, result_dtype: torch.dtype
    ):
        """Consolidates input and result dtypes based on computed and result dtypes.

        Args:
            computed_dtype: The computed dtype.
            result_dtype: The result dtype.

        This method is responsible for consolidating input and result dtypes based on the computed and result dtypes provided.
        """
    ) -> torch.dtype:
        """
        指定函数返回值类型为 torch.dtype，并且以下是函数的文档字符串，描述了函数的作用和背景信息。
        虽然 opmath 是保持精度一致的正确方法，但它会在计算图中插入强制类型转换。
        对于后端来说，这很难优化，因为无法区分插入的强制类型转换和模型代码中的类型转换。
        因此，我们将输入的 dtype 统一为结果的 dtype，以避免这种情况发生。
        """
        if not self._USE_OPMATH and self.promotion_kind in (
            _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        ):
            # 如果不使用 opmath 并且是默认的或者整数到浮点数的类型提升方式，则返回结果的 dtype
            return result_dtype
        # 否则返回计算得到的 dtype
        return computed_dtype

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        """
        预览类型提升的效果，基于输入的参数和关键字参数。
        """
        # 从参数中选择候选参数，其位置在 promote_args_positions 中，并且不为空值
        candidate_args = {
            i: args[i]
            for i in self.promote_args_positions
            if i < len(args) and args[i] is not None
        }
        # 从关键字参数中选择候选参数，其名称在 promote_kwargs_names 中，并且不为空值
        candidate_kwargs = {
            name: kwargs[name]
            for name in self.promote_kwargs_names
            if name in kwargs and kwargs[name] is not None
        }

        # 计算元素操作的输入 dtype 和结果 dtype
        computed_dtype, result_dtype = _prims_common.elementwise_dtypes(
            *_pytree.arg_tree_leaves(*candidate_args.values(), **candidate_kwargs),
            type_promotion_kind=self.promotion_kind,
        )

        # 统一输入的 dtype 和结果的 dtype
        consolidated_input_dtype = self._consolidate_input_dtype(
            computed_dtype, result_dtype
        )

        # 返回一个 TypePromotionSnapshot 对象，包含候选参数和关键字参数的统一 dtype，以及结果的 dtype
        return TypePromotionSnapshot(
            dict.fromkeys(candidate_args.keys(), consolidated_input_dtype),
            dict.fromkeys(candidate_kwargs.keys(), consolidated_input_dtype),
            result_dtype,
        )
    # 定义一个类，继承自ElementwiseTypePromotionRule，用于处理按元素操作的类型提升规则
class DivElementwiseTypePromotionRule(ElementwiseTypePromotionRule):
    """Reference type promotion rule from torch._refs.div.

    Rule depends on the value of the `rounding_mode` argument.
    """

    def __init__(self):
        # 调用父类构造函数初始化
        super().__init__(
            "aten",
            "div",
            promote_args_positions=(0, 1),  # 指定要提升类型的参数位置
            promote_kwargs_names=(),  # 指定要提升类型的关键字参数名
            promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,  # 默认的类型提升种类
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # 从kwargs中获取rounding_mode参数的值
        rounding_mode = kwargs.get("rounding_mode", None)
        if rounding_mode is None:
            # 如果rounding_mode为None，使用整数除法，将类型提升种类设置为INT_TO_FLOAT
            self.promotion_kind = (
                _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
            )
            # 调用父类的方法预览类型提升结果
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "trunc":
            # 如果rounding_mode为'trunc'，使用截断除法，将类型提升种类设置为DEFAULT
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            # 调用父类的方法预览类型提升结果
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "floor":
            # 如果rounding_mode为'floor'，使用向下取整除法，将类型提升种类设置为DEFAULT
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            # 调用父类的方法预览类型提升结果
            return super().preview_type_promotion(args, kwargs)
        # 如果rounding_mode不是上述预期的值，抛出异常
        raise ValueError(f"Unknown rounding_mode: {rounding_mode}")


class ReductionTypePromotionRule(TypePromotionRule):
    def __init__(
        self,
        namespace: str,
        op_name: str,
        promotion_kind: _prims_common.REDUCTION_OUTPUT_TYPE_KIND,
    ):
        """Constructs a TypePromotionRule for reduction operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.sum'.
            op_name: Name of the op. E.g. 'sum' in 'torch.ops.aten.sum'.
            promotion_kind: Type promotion kind. Refer to [_prims_common.reduction_dtypes]((https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py)) for detail.  # noqa: B950
        """
        # 调用父类构造函数初始化
        super().__init__(namespace, op_name)
        self.promotion_kind = promotion_kind

    def __repr__(self):
        # 返回该对象的字符串表示形式，用于调试和输出
        return f"ReductionTypePromotionRule('{self.namespace}', '{self.op_name}', {self.promotion_kind})"

    def __eq__(self, __value: object) -> bool:
        # 判断当前对象是否等于另一个对象
        if not isinstance(__value, ElementwiseTypePromotionRule):
            return False
        # 比较命名空间、操作名称和类型提升种类是否相等
        return (
            self.namespace == __value.namespace
            and self.op_name == __value.op_name
            and self.promotion_kind == __value.promotion_kind
        )

    def __hash__(self) -> int:
        # 计算对象的哈希值，用于在集合中快速查找
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # 确保至少有一个参数传递给 reduction 操作函数
        assert (
            len(args) >= 1
        ), f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
        # 取第一个参数作为操作对象，必须是 torch.Tensor 类型
        arg = args[0]
        assert isinstance(arg, torch.Tensor), f"{type(arg)=} is not torch.Tensor"
        # 从关键字参数中获取 dtype，如果未指定则为 None
        dtype: Optional[torch.dtype] = kwargs.get("dtype", None)

        # 获取计算过程中使用的数据类型和最终结果的数据类型
        computation_dtype, result_dtype = _prims_common.reduction_dtypes(
            arg, self.promotion_kind, dtype
        )
        # 如果最终结果的数据类型为 None，则根据 `promotion_kind` 设置为与计算数据类型相同
        if result_dtype is None:
            result_dtype = computation_dtype

        # 返回一个 TypePromotionSnapshot 对象，包含计算数据类型、空的参数映射、以及结果数据类型
        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )
class AllOrAnyReductionTypePromotionRule(ReductionTypePromotionRule):
    """Reference type promotion rule from torch.ops.aten.all or torch.ops.aten.any.

    This is a special case where computation dtype is always torch.bool.
    The result dtype is always uint8 if `dtype` kwarg is uint8, otherwise torch.bool.
    """

    def __init__(self, op_name: str):
        # 调用父类的构造函数，设置命名空间 "aten"，操作名字由参数 op_name 给定
        super().__init__(
            "aten",
            op_name,
            _prims_common.REDUCTION_OUTPUT_TYPE_KIND.ALWAYS_BOOL,
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # 断言至少有一个参数传递给降维操作
        assert (
            len(args) >= 1
        ), f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
        # 获取第一个参数
        arg = args[0]
        # 断言第一个参数是 torch.Tensor 类型
        assert isinstance(arg, torch.Tensor), f"{type(arg)=} is not torch.Tensor"
        # 计算时使用的数据类型始终为 torch.bool
        computation_dtype = torch.bool
        # 如果参数的数据类型是 torch.uint8，则结果数据类型为 torch.uint8，否则为 torch.bool
        result_dtype = torch.uint8 if arg.dtype == torch.uint8 else torch.bool
        # 返回类型升级快照对象，包括计算时的数据类型和结果数据类型
        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )


class SumLikeReductionTypePromotionRule(ReductionTypePromotionRule):
    """Reference type promotion rule from torch.ops.aten.sum.

    This is a special case where computation dtype is always torch.int64 for integral arg,
    unless overridden by `dtype` kwarg.
    """

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # 断言至少有一个参数传递给降维操作
        assert (
            len(args) >= 1
        ), f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
        # 获取第一个参数
        arg = args[0]
        # 断言第一个参数是 torch.Tensor 类型
        assert isinstance(arg, torch.Tensor), f"{type(arg)=} is not torch.Tensor"
        # 获取关键字参数 `dtype`，默认为 None
        dtype: Optional[torch.dtype] = kwargs.get("dtype", None)
        # 如果未指定 dtype，则根据参数的数据类型来决定
        if dtype is None:
            if _prims_common.is_boolean_dtype(
                arg.dtype
            ) or _prims_common.is_integer_dtype(arg.dtype):
                # 如果参数的数据类型是布尔型或整数型，则结果数据类型为 torch.int64
                dtype = torch.int64
            else:
                # 否则结果数据类型与参数数据类型相同
                dtype = arg.dtype
        # 调用父类的 preview_type_promotion 方法，传递更新后的 dtype 参数
        return super().preview_type_promotion(args, {"dtype": dtype})


# NOTE: [Update type promotion rule]
# BELOW TABLE IS GENERATED FROM `TypePromotionRuleSetGenerator.generate_from_torch_refs`.
# DO NOT EDIT MANUALLY !!!
# For missing rules or discrepancies, please
# 1. Run `pytest test/onnx/test_fx_type_promotion.py` to validate if the generated rule set is current.
#    If it is not, update with new generated set.
# 2. If discrepancies still exist, consider debugging torch._refs or report a bug.
# 3. If rules are still missing, add them to `_EXTRA_TYPE_PROMOTION_RULE_SET` or report a bug.
# Check `TypePromotionRule` class for how each rule is defined and used.
_GENERATED_ATEN_TYPE_PROMOTION_RULE_SET = {
    ElementwiseTypePromotionRule(
        "aten", "abs", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),  # 对于 abs 函数，将复数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "abs_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),  # 对于 abs_ 函数（原地操作），将复数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "acos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 acos 函数，将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "acos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 acos_ 函数（原地操作），将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "acosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 acosh 函数，将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "acosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 acosh_ 函数（原地操作），将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "add", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 对于 add 函数，按照默认规则进行类型提升
    ElementwiseTypePromotionRule(
        "aten", "add_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 对于 add_ 函数（原地操作），按照默认规则进行类型提升
    ElementwiseTypePromotionRule(
        "aten", "addcdiv", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 addcdiv 函数，将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "addcdiv_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 addcdiv_ 函数（原地操作），将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "addcmul", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 对于 addcmul 函数，按照默认规则进行类型提升
    ElementwiseTypePromotionRule(
        "aten", "addcmul_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 对于 addcmul_ 函数（原地操作），按照默认规则进行类型提升
    ElementwiseTypePromotionRule(
        "aten", "addr", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 对于 addr 函数，按照默认规则进行类型提升
    ElementwiseTypePromotionRule(
        "aten", "asin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 asin 函数，将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "asin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 asin_ 函数（原地操作），将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "asinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 asinh 函数，将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "asinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 asinh_ 函数（原地操作），将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "atan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 atan 函数，将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "atan2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 atan2 函数，将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "atan2_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 atan2_ 函数（原地操作），将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "atan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 atan_ 函数（原地操作），将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "atanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 atanh 函数，将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "atanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 对于 atanh_ 函数（原地操作），将整数提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 对于 bitwise_and 函数，按照默认规则进行类型提升
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行按位与操作（in-place），操作数索引为0和1，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    # 创建一个元素级的类型提升规则，用于执行按位左移操作，操作数索引为0和1，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    # 创建一个元素级的类型提升规则，用于执行按位左移操作（in-place），操作数索引为0和1，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "bitwise_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行按位取反操作，操作数索引为0，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "bitwise_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行按位取反操作（in-place），操作数索引为0，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "bitwise_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行按位或操作，操作数索引为0和1，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "bitwise_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行按位或操作（in-place），操作数索引为0和1，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    # 创建一个元素级的类型提升规则，用于执行按位右移操作，操作数索引为0和1，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    # 创建一个元素级的类型提升规则，用于执行按位右移操作（in-place），操作数索引为0和1，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行按位异或操作，操作数索引为0和1，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行按位异或操作（in-place），操作数索引为0和1，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "cat", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    # 创建一个元素级的类型提升规则，用于执行张量连接操作，操作数索引为0，无需额外操作，使用无操作数学类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "cauchy", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于生成柯西分布张量，操作数索引为0，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "cauchy_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于生成柯西分布张量（in-place），操作数索引为0，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "ceil", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于向上取整操作，操作数索引为0，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "ceil_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于向上取整操作（in-place），操作数索引为0，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "celu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行CELU激活函数操作，操作数索引为0，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "celu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行CELU激活函数操作（in-place），操作数索引为0，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "clamp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行张量夹紧操作，操作数索引为0、1和2，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "clamp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级的类型提升规则，用于执行张量夹紧操作（in-place），操作数索引为0、1和2，无需额外操作，使用默认类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "copysign", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级的类型提升规则，用于执行符号复制操作，操作数索引为0和1，无需额外操作，使用整数到浮点数类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "copysign_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级的类型提升规则，用于执行符号复制操作（in-place），操作数索引为0和1，无需额外操作，使用整数到浮点数类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "cos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级的类型提升规则，用于执行余弦函数操作，操作数索引为0，无需额外操作，使用整数到浮点数类型提升策略
    ElementwiseTypePromotionRule(
        "aten", "cos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 定义元素级别的类型提升规则，将整数类型提升为浮点类型，对应 torch 中的 cos_ 函数

    ElementwiseTypePromotionRule(
        "aten", "cosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 类似上面，将整数类型提升为浮点类型，对应 torch 中的 cosh 函数

    ElementwiseTypePromotionRule(
        "aten", "cosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 cosh_ 函数

    ElementwiseTypePromotionRule(
        "aten", "deg2rad", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 deg2rad 函数

    ElementwiseTypePromotionRule(
        "aten", "deg2rad_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 deg2rad_ 函数

    ElementwiseTypePromotionRule(
        "aten", "digamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 digamma 函数

    ElementwiseTypePromotionRule(
        "aten", "digamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 digamma_ 函数

    ElementwiseTypePromotionRule(
        "aten", "elu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 默认的类型提升规则，对应 torch 中的 elu 函数

    ElementwiseTypePromotionRule(
        "aten", "elu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 默认的类型提升规则，对应 torch 中的 elu_ 函数

    ElementwiseTypePromotionRule(
        "aten", "eq", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),  # 总是将操作数提升为布尔类型的规则，对应 torch 中的 eq 函数

    ElementwiseTypePromotionRule(
        "aten", "eq_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),  # 总是将操作数提升为布尔类型的规则，对应 torch 中的 eq_ 函数

    ElementwiseTypePromotionRule(
        "aten", "erf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 erf 函数

    ElementwiseTypePromotionRule(
        "aten", "erf_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 erf_ 函数

    ElementwiseTypePromotionRule(
        "aten", "erfc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 erfc 函数

    ElementwiseTypePromotionRule(
        "aten", "erfc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 erfc_ 函数

    ElementwiseTypePromotionRule(
        "aten", "erfinv", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 erfinv 函数

    ElementwiseTypePromotionRule(
        "aten", "erfinv_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 erfinv_ 函数

    ElementwiseTypePromotionRule(
        "aten", "exp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 exp 函数

    ElementwiseTypePromotionRule(
        "aten", "exp2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 exp2 函数

    ElementwiseTypePromotionRule(
        "aten", "exp2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 exp2_ 函数

    ElementwiseTypePromotionRule(
        "aten", "exp_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 exp_ 函数

    ElementwiseTypePromotionRule(
        "aten", "expm1", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 expm1 函数

    ElementwiseTypePromotionRule(
        "aten", "expm1_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),  # 将整数类型提升为浮点类型，对应 torch 中的 expm1_ 函数

    ElementwiseTypePromotionRule(
        "aten", "exponential", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 默认的类型提升规则，对应 torch 中的 exponential 函数

    ElementwiseTypePromotionRule(
        "aten", "exponential_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),  # 默认的类型提升规则，对应 torch 中的 exponential_ 函数
    ElementwiseTypePromotionRule(
        "aten", "fill", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    # 定义元素级类型提升规则：操作符为 "fill"，输入参数为 [0]，无输出参数，类型提升种类为 NO_OPMATH
    
    ElementwiseTypePromotionRule(
        "aten", "floor", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "floor"，输入参数为 [0]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "floor_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "floor_"，输入参数为 [0]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "floor_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "floor_divide"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "floor_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "floor_divide_"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "fmax", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "fmax"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "fmin", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "fmin"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "fmod", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "fmod"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "fmod_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "fmod_"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "frac", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "frac"，输入参数为 [0]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "frac_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "frac_"，输入参数为 [0]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "gcd", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "gcd"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "gcd_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "gcd_"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "ge", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 定义元素级类型提升规则：操作符为 "ge"，输入参数为 [0, 1]，无输出参数，类型提升种类为 ALWAYS_BOOL
    
    ElementwiseTypePromotionRule(
        "aten", "ge_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 定义元素级类型提升规则：操作符为 "ge_"，输入参数为 [0, 1]，无输出参数，类型提升种类为 ALWAYS_BOOL
    
    ElementwiseTypePromotionRule(
        "aten", "gelu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "gelu"，输入参数为 [0]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "geometric", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "geometric"，输入参数为 [0]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "geometric_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "geometric_"，输入参数为 [0]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "glu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "glu"，输入参数为 [0]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "gt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 定义元素级类型提升规则：操作符为 "gt"，输入参数为 [0, 1]，无输出参数，类型提升种类为 ALWAYS_BOOL
    
    ElementwiseTypePromotionRule(
        "aten", "gt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 定义元素级类型提升规则：操作符为 "gt_"，输入参数为 [0, 1]，无输出参数，类型提升种类为 ALWAYS_BOOL
    
    ElementwiseTypePromotionRule(
        "aten", "hardtanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "hardtanh"，输入参数为 [0]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "heaviside", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "heaviside"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "heaviside_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "heaviside_"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    
    ElementwiseTypePromotionRule(
        "aten", "huber_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则：操作符为 "huber_loss"，输入参数为 [0, 1]，无输出参数，类型提升种类为 DEFAULT
    ElementwiseTypePromotionRule(
        "aten", "hypot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级类型提升规则，用于函数 "hypot"，将第0和第1个参数的类型提升为相同的类型
    ElementwiseTypePromotionRule(
        "aten", "hypot_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级类型提升规则，用于原地操作函数 "hypot_"，将第0和第1个参数的类型提升为相同的类型
    ElementwiseTypePromotionRule(
        "aten", "i0", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级类型提升规则，用于函数 "i0"，将第0个参数的整数类型提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "i0_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级类型提升规则，用于原地操作函数 "i0_"，将第0个参数的整数类型提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "igamma", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级类型提升规则，用于函数 "igamma"，将第0和第1个参数的整数类型提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "igamma_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级类型提升规则，用于原地操作函数 "igamma_"，将第0和第1个参数的整数类型提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "igammac", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级类型提升规则，用于函数 "igammac"，将第0和第1个参数的整数类型提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "igammac_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级类型提升规则，用于原地操作函数 "igammac_"，将第0和第1个参数的整数类型提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "isfinite", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个元素级类型提升规则，用于函数 "isfinite"，将第0个参数的类型提升为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "isinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个元素级类型提升规则，用于函数 "isinf"，将第0个参数的类型提升为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "isnan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个元素级类型提升规则，用于函数 "isnan"，将第0个参数的类型提升为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "isneginf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个元素级类型提升规则，用于函数 "isneginf"，将第0个参数的类型提升为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "isposinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个元素级类型提升规则，用于函数 "isposinf"，将第0个参数的类型提升为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "isreal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个元素级类型提升规则，用于函数 "isreal"，将第0个参数的类型提升为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "l1_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    # 创建一个元素级类型提升规则，用于函数 "l1_loss"，将第0和第1个参数的复数类型提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "lcm", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级类型提升规则，用于函数 "lcm"，将第0和第1个参数的类型提升为相同的类型
    ElementwiseTypePromotionRule(
        "aten", "lcm_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级类型提升规则，用于原地操作函数 "lcm_"，将第0和第1个参数的类型提升为相同的类型
    ElementwiseTypePromotionRule(
        "aten", "le", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个元素级类型提升规则，用于函数 "le"，将第0和第1个参数的类型提升为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "le_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个元素级类型提升规则，用于原地操作函数 "le_"，将第0和第1个参数的类型提升为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "leaky_relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级类型提升规则，用于函数 "leaky_relu"，使用默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "lerp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级类型提升规则，用于函数 "lerp"，使用默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "lerp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个元素级类型提升规则，用于原地操作函数 "lerp_"，使用默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "lgamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级类型提升规则，用于函数 "lgamma"，将第0个参数的整数类型提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "lgamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个元素级类型提升规则，用于原地操作函数 "lgamma_"，将第0个参数的整数类型提升为浮点数类型
    ElementwiseTypePromotionRule(
        "aten", "log", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个类型提升规则，将整数参数的 'log' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "log10", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 类似地，将整数参数的 'log10' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "log10_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 类似地，将整数参数的 'log10_' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "log1p", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 类似地，将整数参数的 'log1p' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "log1p_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 类似地，将整数参数的 'log1p_' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "log2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 类似地，将整数参数的 'log2' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "log2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 类似地，将整数参数的 'log2_' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "log_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 类似地，将整数参数的 'log_' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "log_normal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 创建一个类型提升规则，对于 'log_normal' 操作，使用默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "log_normal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 类似地，对于 'log_normal_' 操作，使用默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "logaddexp", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 类似地，对于 'logaddexp' 操作，使用默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "logaddexp2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 类似地，对于 'logaddexp2' 操作，使用默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "logical_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个类型提升规则，将 'logical_and' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "logical_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 类似地，将 'logical_and_' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "logical_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个类型提升规则，将 'logical_not' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "logical_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 类似地，将 'logical_not_' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "logical_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个类型提升规则，将 'logical_or' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "logical_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 类似地，将 'logical_or_' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "logical_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个类型提升规则，将 'logical_xor' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "logical_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 类似地，将 'logical_xor_' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "logit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 创建一个类型提升规则，将整数参数的 'logit' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "logsumexp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 类似地，将整数参数的 'logsumexp' 操作转换为浮点数
    ElementwiseTypePromotionRule(
        "aten", "lt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 创建一个类型提升规则，将 'lt' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "lt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 类似地，将 'lt_' 操作的参数始终转换为布尔类型
    ElementwiseTypePromotionRule(
        "aten", "maximum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "maximum"，接受两个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "minimum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "minimum"，接受两个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "mish", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "mish"，接受一个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "mish_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "mish_"，接受一个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "mse_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    # 定义元素级操作类型提升规则：函数名为 "mse_loss"，接受两个输入张量，无额外参数，使用复杂类型到浮点类型的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "mul", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "mul"，接受两个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "mul_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "mul_"，接受两个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "ne", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 定义元素级操作类型提升规则：函数名为 "ne"，接受两个输入张量，无额外参数，始终返回布尔类型的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "ne_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 定义元素级操作类型提升规则：函数名为 "ne_"，接受两个输入张量，无额外参数，始终返回布尔类型的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "neg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "neg"，接受一个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "neg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "neg_"，接受一个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "nextafter", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    # 定义元素级操作类型提升规则：函数名为 "nextafter"，接受两个输入张量，无额外参数，不执行数学操作的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "nextafter_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    # 定义元素级操作类型提升规则：函数名为 "nextafter_"，接受两个输入张量，无额外参数，不执行数学操作的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "nll_loss", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "nll_loss"，接受一个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "normal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "normal"，接受一个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "normal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "normal_"，接受一个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "pdist", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "pdist"，接受一个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten",
        "poisson_nll_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    ),
    # 定义元素级操作类型提升规则：函数名为 "poisson_nll_loss"，接受两个输入张量，无额外参数，整数类型到浮点类型的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "pow", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    # 定义元素级操作类型提升规则：函数名为 "pow"，接受两个输入张量，无额外参数，布尔类型到长整型的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "pow_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    # 定义元素级操作类型提升规则：函数名为 "pow_"，接受两个输入张量，无额外参数，布尔类型到长整型的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "prelu", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级操作类型提升规则：函数名为 "prelu"，接受两个输入张量，无额外参数，使用默认的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "rad2deg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级操作类型提升规则：函数名为 "rad2deg"，接受一个输入张量，无额外参数，整数类型到浮点类型的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "rad2deg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级操作类型提升规则：函数名为 "rad2deg_"，接受一个输入张量，无额外参数，整数类型到浮点类型的类型提升策略

    ElementwiseTypePromotionRule(
        "aten", "reciprocal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级操作类型提升规则：函数名为 "reciprocal"，接受一个输入张量，无额外参数，整数类型到浮点类型的类型提升策略
    ElementwiseTypePromotionRule(
        "aten", "reciprocal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 reciprocal_，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 relu，使用默认的类型提升规则
    ElementwiseTypePromotionRule(
        "aten", "remainder", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 remainder，使用默认的类型提升规则，涉及两个参数
    ElementwiseTypePromotionRule(
        "aten", "remainder_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 remainder_，使用默认的类型提升规则，涉及两个参数
    ElementwiseTypePromotionRule(
        "aten", "round", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 round，使用默认的类型提升规则
    ElementwiseTypePromotionRule(
        "aten", "rsqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 rsqrt，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "rsqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 rsqrt_，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "rsub", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 rsub，使用默认的类型提升规则，涉及两个参数
    ElementwiseTypePromotionRule(
        "aten", "selu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 selu，使用默认的类型提升规则
    ElementwiseTypePromotionRule(
        "aten", "selu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 selu_，使用默认的类型提升规则
    ElementwiseTypePromotionRule(
        "aten", "sgn", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sgn，使用默认的类型提升规则
    ElementwiseTypePromotionRule(
        "aten", "sgn_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sgn_，使用默认的类型提升规则
    ElementwiseTypePromotionRule(
        "aten", "sigmoid", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sigmoid，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "sigmoid_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sigmoid_，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "sign", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sign，使用默认的类型提升规则
    ElementwiseTypePromotionRule(
        "aten", "sign_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sign_，使用默认的类型提升规则
    ElementwiseTypePromotionRule(
        "aten", "signbit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    # 设置 Elementwise 类型提升规则：操作为 signbit，结果总是布尔类型
    ElementwiseTypePromotionRule(
        "aten", "sin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sin，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "sin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sin_，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "sinc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sinc，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "sinc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sinc_，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "sinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sinh，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten", "sinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 设置 Elementwise 类型提升规则：操作为 sinh_，第一个参数类型从整数到浮点数的提升
    ElementwiseTypePromotionRule(
        "aten",
        "smooth_l1_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    # 设置 Elementwise 类型提升规则：操作为 smooth_l1_loss，使用复杂到浮点数的类型提升规则，涉及两个参数
    ElementwiseTypePromotionRule(
        "aten", "softplus", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.softplus 操作
    ElementwiseTypePromotionRule(
        "aten", "sqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.sqrt 操作，将整数提升为浮点数
    ElementwiseTypePromotionRule(
        "aten", "sqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.sqrt_ 操作，将整数提升为浮点数
    ElementwiseTypePromotionRule(
        "aten", "square", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.square 操作，将布尔值提升为长整型
    ElementwiseTypePromotionRule(
        "aten", "square_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.square_ 操作，将布尔值提升为长整型
    ElementwiseTypePromotionRule(
        "aten", "sub", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.sub 操作，默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "sub_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.sub_ 操作，默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "tan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.tan 操作，将整数提升为浮点数
    ElementwiseTypePromotionRule(
        "aten", "tan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.tan_ 操作，将整数提升为浮点数
    ElementwiseTypePromotionRule(
        "aten", "tanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.tanh 操作，将整数提升为浮点数
    ElementwiseTypePromotionRule(
        "aten", "tanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.tanh_ 操作，将整数提升为浮点数
    ElementwiseTypePromotionRule(
        "aten", "threshold", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.threshold 操作，默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "threshold_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.threshold_ 操作，默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "true_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.true_divide 操作，将整数提升为浮点数
    ElementwiseTypePromotionRule(
        "aten", "true_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.true_divide_ 操作，将整数提升为浮点数
    ElementwiseTypePromotionRule(
        "aten", "trunc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.trunc 操作，默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "trunc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.trunc_ 操作，默认的类型提升方式
    ElementwiseTypePromotionRule(
        "aten", "where", [1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.where 操作，无运算数学操作
    ElementwiseTypePromotionRule(
        "aten", "xlogy", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.xlogy 操作，将整数提升为浮点数
    ElementwiseTypePromotionRule(
        "aten", "xlogy_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    # 定义元素级类型提升规则，针对 torch 中 aten.xlogy_ 操作，将整数提升为浮点数
# 手动编辑的额外类型提升规则集。在添加新规则之前，请查看注释[更新类型提升规则]。
_EXTRA_TYPE_PROMOTION_RULE_SET = {
    # torch._refs跳过类型提升装饰对`clamp_min`和`clamp_max`的应用，因为调用被路由到装饰后的`aten.clamp`操作。
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_max",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_min",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    # torch.ops.aten.div.Tensor_mode应用不同的类型提升规则，具体取决于`mode`参数的值。
    DivElementwiseTypePromotionRule(),
    # 由于逻辑已经写入操作参考实现内部，手动策划缩减操作。
    AllOrAnyReductionTypePromotionRule("all"),
    AllOrAnyReductionTypePromotionRule("any"),
    # ReductionTypePromotionRule定义了规则，以指定在`amax`操作中输出类型保持不变。
    ReductionTypePromotionRule(
        "aten",
        "amax",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    ReductionTypePromotionRule(
        "aten",
        "amin",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    # 对于torch.ops.aten.mean来说，它是一个特殊情况，不需要类型提升。
    ReductionTypePromotionRule(
        "aten",
        "std",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "std_mean",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "var",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    # SumLikeReductionTypePromotionRule定义了规则，以指定在`cumprod`操作中输出类型保持不变。
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumprod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumsum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "prod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "sum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
}


class ElementwiseTypePromotionRuleSetGenerator:
    """从装饰有元素类型提升规则的参考操作中提取信息的方法。

    目标是检索装饰器：

    ```python
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a", "b"),
            type_promotion_kind=type_promotion_kind,
        )
    ```py

    从参考操作中提供信息，指出哪些参数会被提升。
    """
    @classmethod
    def generate_from_torch_refs(cls) -> Set[ElementwiseTypePromotionRule]:
        """从torch._C._refs下的引用操作中生成类型提升规则集合。"""
        rule_set = set()
        rule_set.update(cls._parse_torch_refs(_refs))
        rule_set.update(cls._parse_torch_refs(_nn_refs))
        rule_set.update(cls._parse_torch_refs(_linalg_refs))
        rule_set.update(cls._parse_torch_refs(_special_refs))
        rule_set.update(cls._parse_torch_refs(_functional_refs))
        return rule_set

    @classmethod
    def _parse_torch_refs(
        cls, ref_module: ModuleType
    ) -> Set[ElementwiseTypePromotionRule]:
        """从给定的模块中解析类型提升规则集合。

        Args:
            ref_module (ModuleType): 要解析的模块对象。

        Returns:
            Set[ElementwiseTypePromotionRule]: 包含解析的类型提升规则对象的集合。
        """
        logger.info("处理模块：%s", ref_module.__name__)
        rule_set = set()
        for name in ref_module.__all__:
            decorated_op = getattr(ref_module, name)
            rule = cls._parse_type_promotion_rule_from_refs_op(decorated_op)
            if rule is not None and rule.is_valid():
                rule_set.add(rule)

        return rule_set

    @classmethod
    def _parse_type_promotion_rule_from_refs_op(
        cls,
        decorated_op: Callable,
    ) -> Optional[ElementwiseTypePromotionRule]:
        """从装饰过的操作中解析类型提升规则。

        Args:
            decorated_op (Callable): 装饰过的操作对象。

        Returns:
            Optional[ElementwiseTypePromotionRule]: 解析出的类型提升规则对象，如果解析失败返回None。
        """
    ) -> Optional[ElementwiseTypePromotionRule]:
        """从 torch._refs 下的装饰器中检索和解析类型提升规则。"""
        fn = decorated_op
        type_promo_wrapper = None
        while fn_closure_vars := _try_getclosurevars(fn):
            # 检查是否能获取到闭包变量
            if "fn" not in fn_closure_vars.nonlocals:
                break
            # 如果在非局部变量中找不到 "fn"，则退出循环
            if "self" in fn_closure_vars.nonlocals and isinstance(
                fn_closure_vars.nonlocals["self"],
                _prims_common_wrappers.elementwise_type_promotion_wrapper,
            ):
                # 如果找到了 self，并且它是 elementwise_type_promotion_wrapper 的实例
                type_promo_wrapper = fn_closure_vars.nonlocals["self"]
                break
            # 继续向上追溯函数的闭包
            fn = fn_closure_vars.nonlocals["fn"]

        if type_promo_wrapper is not None:
            # 如果找到了类型提升的包装器
            signature = inspect.signature(decorated_op)

            pos = 0
            promote_args_positions = []
            promote_kwargs_names = []

            if type_promo_wrapper.type_promoting_arg_names is not None:
                # 如果存在类型提升的参数名列表
                for name, param in signature.parameters.items():
                    if name in type_promo_wrapper.type_promoting_arg_names:
                        # 如果参数名在类型提升参数名列表中
                        if param.kind in (
                            param.POSITIONAL_OR_KEYWORD,
                            param.POSITIONAL_ONLY,
                        ):
                            # 如果参数是位置参数或者位置或关键字参数
                            promote_args_positions.append(pos)
                        elif param.kind == param.KEYWORD_ONLY:
                            # 如果参数是仅限关键字参数
                            promote_kwargs_names.append(name)
                    pos += 1

            return ElementwiseTypePromotionRule(
                "aten",
                decorated_op.__name__,
                promote_args_positions=promote_args_positions,
                promote_kwargs_names=promote_kwargs_names,
                promotion_kind=type_promo_wrapper.type_promotion_kind,
            )

        logger.warning(
            "无法找到类型提升规则：%s.%s",
            decorated_op.__module__,
            decorated_op.__name__,
        )
        # 如果未找到类型提升规则，则记录警告并返回 None
        return None
class TypePromotionTable:
    """Type promotion table for torch.ops."""

    def __init__(self):
        # 初始化空的规则表
        self._rule_table = {}
        # 添加生成的 ATen 类型提升规则集中的规则
        for rule in _GENERATED_ATEN_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)
        # 添加额外的类型提升规则集中的规则
        for rule in _EXTRA_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)

    @_beartype.beartype
    def add_rule(self, rule: TypePromotionRule) -> None:
        """Add a type promotion rule for a python op in a torch.ops module.

        Args:
            rule: Type promotion rule.
            module: Module containing the op. E.g. torch.ops.aten.

        Raises:
            ValueError: If the rule is invalid.
        """
        # 如果规则无效，抛出 ValueError 异常
        if not rule.is_valid():
            raise ValueError(f"Invalid type promotion rule: {rule}")
        # 将规则存储在规则表中，键为命名空间和操作名的组合
        self._rule_table[f"{rule.namespace}.{rule.op_name}"] = rule

    @_beartype.beartype
    def get_rule(
        self, py_op: torch._ops.OpOverloadPacket
    ) -> Optional[TypePromotionRule]:
        """Get type promotion rule for a python op under 'torch.ops.<namespace>'."""
        # 返回给定 Python 操作的类型提升规则，如果找不到则返回 None
        return self._rule_table.get(str(py_op), None)


@_beartype.beartype
def get_type_promotion_rule(
    diagnostic: diagnostics.Diagnostic,
    node: torch.fx.Node,
    type_promotion_table: TypePromotionTable,
) -> Optional[TypePromotionRule]:
    """Get type promotion rule for a node.

    Args:
        diagnostic: Diagnostic object.
        node: Node to get type promotion rule for.
        type_promotion_table: Type promotion table.

    Returns:
        Type promotion rule for the node. None if no rule is found or if the node is not
        representing a torch operator.
    """
    # 获取节点的目标操作
    op = node.target
    # 如果目标操作不是 OpOverload 类型
    if not isinstance(op, torch._ops.OpOverload):
        # 设置诊断消息，指出为什么跳过该节点处理
        diagnostic.message = (
            f"Skipped for {diagnostics.format_argument(node)}: "
            f"node.target is not OpOverload. Got type: {type(op)}"
        )
        return None
    # 获取目标操作的类型提升规则
    if (rule := type_promotion_table.get_rule(op.overloadpacket)) is None:
        # 设置诊断消息，指出为什么找不到类型提升规则
        diagnostic.message = (
            f"Skipped for {diagnostics.format_argument(node)}: "
            f"Cannot find type promotion rule for op: {op}"
        )
        return None

    # 记录诊断信息，表明找到了类型提升规则
    diagnostic.info("Found type promotion rule: %s", rule)
    return rule


class _OpTraceDispatchMode(_python_dispatch.TorchDispatchMode):
    """Trace ops that were dispatched.

    Utilize the dispatch mechanism in [`__torch_dispatch__`](https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557)
    to trace op overloads that were dispatched to. This is used to find the compatible
    op overload for a given op overload packet for different set of args and kwargs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化跟踪操作的空列表
        self.traced_ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 将调度到的操作函数添加到跟踪操作列表中
        self.traced_ops.append(func)
        # 调用操作函数并返回其结果
        return func(*args, **kwargs)
# 使用装饰器确保该函数被 beartype 包装，用于类型检查
@_beartype.beartype
# 查找兼容的 OpOverload，基于提供的参数和关键字参数
def find_compatible_op_overload(
    op: torch._ops.OpOverloadPacket, args: tuple, kwargs: dict
) -> torch._ops.OpOverload:
    """Find compatible OpOverload for an OpOverloadPacket using provided args and kwargs.

    Each "call_function" fx.Node in the fx.GraphModule has a target that represents a torch._ops.OpOverload.
    The OpOverload contains an OpOverloadPacket that holds all the available overloads for the operation.

    During the type promotion pass, there are cases where the types of the args and kwargs may change,
    such as promoting Python numbers to tensors. Consequently, the original OpOverload might not be
    compatible with the updated args and kwargs. This function is used to identify the compatible
    OpOverload for the given args and kwargs.

    Args:
        op: OpOverloadPacket to find compatible OpOverload for.
        args: The positional arguments to consider for compatibility.
        kwargs: The keyword arguments to consider for compatibility.

    Returns:
        torch._ops.OpOverload: The compatible OpOverload found for the given args and kwargs.

    Raises:
        RuntimeError: If no compatible op overload is found.

    Examples:
        >>> import torch
        >>> packet = torch.ops.aten.pow
        >>> args = (torch.tensor([1.0, 2.0]), 2)
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Scalar'
        >>> args = (torch.tensor([1.0, 2.0]), torch.tensor(2.0))
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Tensor'
    """
    # 利用调度机制寻找兼容的 op 过载
    op_trace_dispatch_mode = _OpTraceDispatchMode()
    # 使用上下文管理器启动 op 的跟踪调度模式
    with op_trace_dispatch_mode:
        # 调用 op，并传入 args 和 kwargs
        op(*args, **kwargs)
    # 断言至少有一个被跟踪的 op
    assert (
        len(op_trace_dispatch_mode.traced_ops) >= 1
    ), "Expected at least 1 traced op, got 0"

    # 获取第一个被跟踪的 op
    new_op_overload = op_trace_dispatch_mode.traced_ops[0]
    # 断言新的 op 是 OpOverload 类型
    assert isinstance(
        new_op_overload, torch._ops.OpOverload
    ), f"Expected OpOverload, got {type(new_op_overload)}"
    # 断言新的 op 使用了相同的 OpOverload packet
    assert (
        new_op_overload.overloadpacket == op
    ), f"Expected same OpOverload packet, got {new_op_overload.overloadpacket} != {op}"

    # 返回找到的新的兼容 OpOverload
    return new_op_overload


class _TypePromotionInterpreter(torch.fx.Interpreter):
    """Interpreter that inserts type promotion for each node."""

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        type_promotion_table: TypePromotionTable,
    ):
        # 调用父类构造函数初始化解释器
        super().__init__(module)
        # 存储诊断上下文和类型提升表
        self.diagnostic_context = diagnostic_context
        self.type_promotion_table = type_promotion_table
    # 运行节点并根据 `fx_traceback.get_current_meta()` 设置元数据

    def _run_node_and_set_meta(self, node) -> Any:
        """
        Run node and set meta according to `fx_traceback.get_current_meta()`.

        This should be used on new nodes or nodes that have been modified.
        By default `Interpreter.run_node` does not update `node.meta`.
        Set `node.meta` to the current meta, except for `node.meta["val"]`, which is
        recomputed.
        """
        # 调用父类的方法来运行节点
        out = super().run_node(node)
        # 将新的输出值更新到解释器环境状态中
        self.env[node] = out
        # 根据当前的元数据更新节点的元数据，但不更新已存在的 `node.meta["val"]`
        node.meta.update(
            (k, v)
            for k, v in fx_traceback.get_current_meta().items()
            if k not in node.meta
        )
        # 使用 proxy_tensor.extract_val(out) 方法重新计算 `node.meta["val"]`
        node.meta["val"] = proxy_tensor.extract_val(out)
        # 返回节点的输出
        return out

    @_beartype.beartype
    def _create_node(
        self,
        graph: torch.fx.Graph,
        op_type: str,
        target: torch.fx.node.Target,
        args: tuple,
        kwargs: dict,
    ) -> torch.fx.Node:
        """
        Create a node and set its metadata.

        Ensure `op_type` is one of the expected types: 'call_function', 'call_method',
        'get_attr', 'call_module', 'placeholder', 'output'.
        """
        # 断言确保 op_type 在预期的操作类型之中
        assert op_type in (
            "call_function",
            "call_method",
            "get_attr",
            "call_module",
            "placeholder",
            "output",
        ), f"Unexpected op_type: {op_type}"
        # 在图中创建一个节点，使用指定的目标、参数和关键字参数
        node = getattr(graph, op_type)(target, args, kwargs)
        # 运行节点并设置节点的元数据
        self._run_node_and_set_meta(node)
        # 返回创建的节点
        return node

    @_beartype.beartype
    def _rerun_node_after_type_promotion(
        self,
        diagnostic: diagnostics.Diagnostic,
        node: torch.fx.Node,
        expected_out_dtype: torch.dtype,
    ) -> None:
        """重新运行经过类型提升后的节点，并更新 node.meta["val"] 的输出值。"""
        # 获取节点的当前值
        node_val = node.meta.get("val", None)
        # 确保节点的值不为空
        assert node_val is not None, f"Node {node} node.meta['val'] is not set."
        # 从环境中获取节点的参数和关键字参数
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        # 获取节点的目标操作
        target = node.target
        # 确保目标操作是 OpOverload 类型
        assert isinstance(
            target, torch._ops.OpOverload
        ), f"Expected OpOverload, got {type(target)}"
        # 找到兼容的 OpOverload，并设置为节点的新目标操作
        node.target = find_compatible_op_overload(target.overloadpacket, args, kwargs)

        # 运行节点并更新其元数据
        new_node_val = self._run_node_and_set_meta(node)
        # 确保运行后的节点值类型与之前的节点值类型一致
        assert isinstance(new_node_val, type(node_val)), (
            f"run_node output type should not change between runs. "
            f"Got {type(new_node_val)}, expect {type(node_val)}."
        )

        # 如果节点值是 torch.Tensor 类型
        if isinstance(node_val, torch.Tensor):
            prev_node_dtype = node_val.dtype

            # 确保节点值的数据类型与预期的输出数据类型一致
            assert prev_node_dtype == expected_out_dtype, (
                f"node.meta['val'].dtype({prev_node_dtype}) does not agree with "
                f"type promotion rule({expected_out_dtype})."
            )

            # 如果新节点值的数据类型与预期的输出数据类型不一致
            if new_node_val.dtype != expected_out_dtype:
                # 由于显式类型提升，预期的结果数据类型可能与计算数据类型不同。
                # 需要将输出显式转换为预期的数据类型。
                # 关于 "op math" 主题可以参考 `_prims_common.elementwise_dtypes`。
                graph = node.graph
                with graph.inserting_after(node):
                    # 创建一个节点，将输出显式转换为预期的数据类型
                    output_cast_node = self._create_node(
                        graph,
                        "call_function",
                        torch.ops.prims.convert_element_type.default,
                        (node,),
                        {"dtype": expected_out_dtype},
                    )
                    # 替换节点的所有使用
                    node.replace_all_uses_with(output_cast_node)
                    output_cast_node.args = (node,)
                    # 提示信息
                    diagnostic.info(
                        "Node '%s' output dtype becomes %s due to op math. "
                        "Cast back to %s.",
                        node,
                        new_node_val.dtype,
                        expected_out_dtype,
                    )

        # 如果节点值是 torch 的符号类型
        elif fx_type_utils.is_torch_symbolic_type(node_val):
            raise NotImplementedError(
                "Type promotion does not support node output of sym types."
            )
        # 如果节点值是列表或元组
        elif isinstance(node_val, (list, tuple)):
            raise NotImplementedError(
                "Type promotion does not support node output of list or tuple."
            )
        else:
            # 如果节点值类型未知，则引发运行时错误
            raise RuntimeError(f"Unexpected node output type: {type(node_val)}.")

    @_beartype.beartype
    @_beartype.beartype
    def _maybe_promote_node(
        self,
        diagnostic: diagnostics.Diagnostic,
        node: torch.fx.Node,
        rule: TypePromotionRule,
    ) -> torch.fx.Node:
        """Promote node inputs and outputs according to type promotion rule."""
        # 从当前环境中获取参数和关键字参数
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        # 预览类型提升的信息
        type_promotion_info = rule.preview_type_promotion(args, kwargs)
        new_args = []
        new_kwargs = {}
        # 处理每个位置参数
        for i, arg in enumerate(node.args):
            # 对参数进行可能的提升
            new_args.append(
                self._maybe_promote_arg(
                    diagnostic, node, arg, type_promotion_info.args_dtypes.get(i, None)
                )
            )

        # 处理关键字参数
        for name, arg in node.kwargs.items():
            # 对参数进行可能的提升
            new_kwargs[name] = self._maybe_promote_arg(
                diagnostic, node, arg, type_promotion_info.kwargs_dtypes.get(name, None)
            )
        new_args = tuple(new_args)

        # 如果参数或关键字参数有变化，则应用类型提升
        if node.args != new_args or node.kwargs != new_kwargs:
            diagnostic.message = f"Applied type promotion for {node}. "
            node.args = new_args
            node.kwargs = new_kwargs
            # 在类型提升后重新运行节点
            self._rerun_node_after_type_promotion(
                diagnostic, node, type_promotion_info.out_dtype
            )
        else:
            diagnostic.message = f"Type promotion not needed for {node}. "

        return node

    @diagnostics.diagnose_call(
        rule=diagnostics.rules.fx_node_insert_type_promotion,
        level=diagnostics.levels.NONE,
    )
    def run_node(self, node: torch.fx.Node) -> Any:
        """This method is an override which inserts type promotion nodes as needed.

        For each `call_function` node, an initial check is conducted to determine if a type
        promotion rule is applicable. If a relevant rule exists, type casting nodes are
        introduced for the corresponding arguments. The OpOverload of the node is updated
        to one that accommodates the promoted types. Should the output type be different,
        type casting node is inserted for this output.

        The call `super().run_node(node)` is guaranteed to be invoked for each node.
        In the case of new or modified nodes, the result of `super().run_node(node)` is
        used to update its `node.meta["val"]` value.
        """
        # 获取当前正在处理的节点的诊断上下文
        diagnostic = self.diagnostic_context.inflight_diagnostic()
        # 将当前节点设置为处理中的节点
        with self._set_current_node(node):
            # 如果节点不是 `call_function` 类型，则跳过处理
            if node.op != "call_function":
                diagnostic.message = f"Skipped {node}: not a call_function."
            # 获取适用的类型提升规则
            elif rule := get_type_promotion_rule(
                diagnostic, node, self.type_promotion_table
            ):
                # 可能进行节点的类型提升
                self._maybe_promote_node(diagnostic, node, rule)

        # 调用父类方法处理节点，并返回结果
        return super().run_node(node)
class InsertTypePromotion(_pass.Transform):
    """Explicitly insert type promotion ops to the graph.

    This class subclasses `_pass.Transform` to provide graph level diagnostic tracking.
    Underneath, the main pass is driven by `_TypePromotionInterpreter`, which is a subclass
    of `torch.fx.Interpreter` to interpret the fx.Graph and perform the insertion of type
    promotion operations.

    The interpreter is extended with ability to track diagnostic information for each node.

    By re-running the new and modified nodes using the interpreter, we can update the
    metadata, specifically the fake tensor stored under node.meta["val"], and ensure it
    reflects the latest changes.

    See [FXE0015: fx_node_insert_type_promotion](https://pytorch.org/docs/main/generated/onnx_dynamo_diagnostics_rules/FXE0015%3Afx-node-insert-type-promotion.html) for more details.  # noqa: B950
    """

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        type_promotion_table: Optional[TypePromotionTable] = None,
    ):
        super().__init__(diagnostic_context, module)
        # 初始化 InsertTypePromotion 对象时，创建 _TypePromotionInterpreter 实例
        self.interpreter = _TypePromotionInterpreter(
            diagnostic_context, module, type_promotion_table or TypePromotionTable()
        )

    def _fetch_fake_args(
        self,
    ) -> Sequence[
        Optional[
            Union[
                fake_tensor.FakeTensor,
                float,
                int,
                bool,
                torch.SymInt,
                torch.SymFloat,
                torch.SymBool,
            ]
        ]
    ]:
        """Fetch fake args from fx graph.

        For each argument, try to fetch fake tensor from the matching placeholder node.
        """
        fake_args = []
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                try:
                    # 尝试获取节点 meta 数据中的 "val" 字段作为 meta_value
                    meta_value = _val = node.meta.get("val", None)
                except RuntimeError as e:
                    if not node.users:
                        # 如果占位符节点没有被使用，安全地将 meta_value 设为 None
                        meta_value = None
                    else:
                        # 如果无法获取符号化的假参数，且占位符节点有用户，抛出异常
                        raise RuntimeError(
                            "Cannot fetch symbolic fake args from fx graph. "
                            "InsertTypePromotion pass needs to run with pre-existing fake args, "
                            "Otherwise the pass will produce inaccurate dynamic shape. "
                        ) from e

                fake_args.append(meta_value)
        return fake_args

    @_beartype.beartype
    # 定义一个方法 `_run`，返回类型为 `torch.fx.GraphModule`
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        # 断言确保没有传入位置参数，因为 `InsertTypePromotion` 需要从图中推断符号化的假参数
        assert not args, (
            "`InsertTypePromotion` deduces symbolic fake arguments from the graph. "
            "It does not accept concrete arguments as input because this pass requires "
            "re-running the graph. When executed with newly faked concrete arguments, "
            "the pass loses the symbolic dynamic shape information."
        )
        # 断言确保没有传入关键字参数 `kwargs`，因为此方法不支持关键字参数
        assert not kwargs, "`kwargs` is not supported"

        # 获取假参数
        fake_args = self._fetch_fake_args()
        # 获取假模式
        fake_mode = self.fake_mode
        # 断言确保假模式不为 `None`
        assert fake_mode is not None, "Cannot detect fake_mode."

        # 使用上下文管理器，可能禁用假张量模式
        # 设置当前的假模式
        # 保留节点元数据的执行上下文
        with proxy_tensor.maybe_disable_fake_tensor_mode(), (
            fake_mode
        ), fx_traceback.preserve_node_meta():
            # 运行解释器，传入假参数
            self.interpreter.run(*fake_args)

        # 返回当前模块
        return self.module
```