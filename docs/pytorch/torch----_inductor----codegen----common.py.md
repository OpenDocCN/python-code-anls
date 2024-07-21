# `.\pytorch\torch\_inductor\codegen\common.py`

```py
# 允许未定义的类型定义（mypy配置）
# 导入上下文管理器模块
import contextlib
# 导入数据类模块
import dataclasses
# 导入函数工具模块
import functools
# 导入迭代工具模块
import itertools
# 导入日志模块
import logging
# 导入数学模块
import math
# 导入操作符模块
import operator
# 导入正则表达式模块
import re
# 导入枚举模块中的auto和Enum类
from enum import auto, Enum
# 从迭代工具模块中导入chain函数
from itertools import chain
# 导入类型提示相关模块
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

# 导入符号计算模块
import sympy
# 从符号计算打印模块中导入打印机类
from sympy.printing.printer import Printer

# 导入PyTorch模块
import torch
# 导入Torch的FX模块
import torch.fx
# 从Torch的私有模块中导入元素级别类型提升的类型
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
# 导入Torch的私有模块_pytree
from torch.utils import _pytree as pytree
# 导入Torch的私有模块_sympy.symbol
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT
# 导入Torch的私有模块_sympy.value_ranges
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges

# 从上层模块导入config和metrics
from .. import config, metrics
# 从工具模块导入若干实用函数和类
from ..utils import (
    DeferredLineBase,
    generate_assert,
    IndentedBuffer,
    sympy_dot,
    sympy_subs,
    unique,
)
# 从虚拟化模块导入操作、操作处理器、操作数值、缩减类型、存储模式和V类
from ..virtualized import ops, OpsHandler, OpsValue, ReductionType, StoreMode, V

# 获取调度日志记录器
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")


def data_type_logger(msg):
    # 检查调度日志记录器是否启用DEBUG级别日志，如果是则记录数据类型传播信息
    if schedule_log.isEnabledFor(logging.DEBUG):
        schedule_log.debug("Data type propagation: %s", msg)


@dataclasses.dataclass
class WorkspaceArg:
    """A temporary buffer used for a single kernel, then discarded.

    Not registered as a traditional buffer since there are no users,
    so it would be dead code eliminated.
    """
    # 缓冲区大小，使用符号表达式表示
    nbytes: sympy.Expr
    # 是否进行零填充
    zero_fill: bool


@dataclasses.dataclass
class TensorArg:
    """Represents a tensor argument for kernel operations."""
    # 张量名称
    name: str
    # 缓冲区名称
    buffer: str
    # 张量数据类型
    dtype: torch.dtype
    # 偏移量，仅在C++环境下有效
    offset: sympy.Expr = sympy.Integer(0)
    # 别名，仅在Halide环境下有效
    alias_of: Optional[str] = None


@dataclasses.dataclass
class SizeArg:
    """Represents a size argument used in kernel computations."""
    # 大小参数名称
    name: str
    # 大小参数的表达式
    expr: sympy.Expr

    @property
    def alias_of(self):
        return None


@dataclasses.dataclass
class DeviceCodegen:
    """Represents code generation specifics for a device."""
    # 调度方式
    scheduling: Any
    # 包装代码生成类型
    wrapper_codegen: type
    # C++包装代码生成类型，默认为None
    cpp_wrapper_codegen: type = type(None)


# 定义用于内核覆盖操作的类
class DeviceOpOverrides:
    # 导入原始流作为函数未实现的方法
    def import_get_raw_stream_as(self, name):
        raise NotImplementedError

    # 设置设备的方法，未实现
    def set_device(self, device_idx):
        raise NotImplementedError

    # 同步方法，未实现
    def synchronize(self):
        raise NotImplementedError

    # 设备保护方法，未实现
    def device_guard(self, device_idx):
        raise NotImplementedError


# 设备操作覆盖字典
device_op_overrides_dict: Dict[str, DeviceOpOverrides] = {}


# Inductor生成的代码包括两个主要部分：内核代码和包装器代码。
# 对于任何希望与Inductor集成的新后端，必须定制这两个主要部分以生成其特定代码。
#
# 内核代码生成由不同的调度决定。因此，新后端需要为其独特的内核代码生成提供自定义调度。
# 目前，CppScheduling和TritonScheduling分别用于C++/OpenMP和Triton后端。
#
# 将后端注册到指定设备的函数
def register_backend_for_device(
    device: str,
    device_scheduling: Any,
    device_wrapper_codegen: type,
    device_cpp_wrapper_codegen: type = type(None),
):
    # 将给定设备的调度器和包装器代码生成器注册到全局设备代码生成器字典中
    device_codegens[device] = DeviceCodegen(
        device_scheduling, device_wrapper_codegen, device_cpp_wrapper_codegen
    )


# 枚举后端特性
class BackendFeature(Enum):
    FOREACH = auto()  # 自动分配特性
    BUCKETIZE = auto()  # 桶装特性
    INPLACE_BUFFERS = auto()  # 原地缓冲特性
    MASKED_SCATTER_WITH_INDEX = auto()  # 掩码散列与索引特性
    SCAN = auto()  # 扫描特性
    SORT = auto()  # 排序特性
    TUPLE_REDUCTION = auto()  # 元组缩减特性
    PREFER_STORE_LOOP_ORDER = auto()  # 首选存储循环顺序特性
    TRITON_TEMPLATES = auto()  # Triton 模板特性
    REDUCE_TO_SINGLE_ELEMENT = auto()  # 缩减至单元素特性


# 获取给定设备的后端特性
def get_backend_features(device: Union[torch.device, str]):
    init_backend_registration()
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        assert isinstance(device, str)
        device_type = device
        device = torch.device(device_type)
    # 获取设备类型对应的调度器实例，并返回其后端特性
    scheduling = get_scheduling_for_device(device_type)
    return scheduling(None).get_backend_features(device)


# 检查给定设备是否具有特定后端特性
def has_backend_feature(device, feature):
    """See also V.graph.has_feature"""
    assert isinstance(feature, BackendFeature)
    # 判断给定设备是否具有指定的后端特性
    return feature in get_backend_features(device)


# 获取给定设备的调度器实例
def get_scheduling_for_device(device: str):
    # 如果设备在设备代码生成器中注册，返回其调度器；否则返回 None
    return device_codegens[device].scheduling if device in device_codegens else None


# 获取给定设备的包装器代码生成器实例
def get_wrapper_codegen_for_device(device: str, cpp_wrapper: bool = False):
    if device in device_codegens:
        wrapper_codegen_obj: DeviceCodegen = device_codegens[device]
        # 如果请求 C++ 包装器，则返回 C++ 包装器代码生成器；否则返回 Python 包装器代码生成器
        return (
            wrapper_codegen_obj.cpp_wrapper_codegen
            if cpp_wrapper
            else wrapper_codegen_obj.wrapper_codegen
        )
    else:
        return None


# 使用 functools 缓存的方式初始化后端注册
@functools.lru_cache(None)
def init_backend_registration():
    # 导入必要的后端调度器和包装器类
    from .cpp import CppScheduling
    from .cpp_wrapper_cpu import CppWrapperCpu
    from .cpp_wrapper_cuda import CppWrapperCuda
    # 导入 CUDACombinedScheduling 类，用于处理 CUDA 设备的组合调度
    from .cuda_combined_scheduling import CUDACombinedScheduling
    # 导入 HalideScheduling 类，用于处理 Halide 调度
    from .halide import HalideScheduling
    # 导入 TritonScheduling 类，用于处理 Triton 调度
    from .triton import TritonScheduling
    # 导入 WrapperCodeGen 类，用于生成包装代码
    from .wrapper import WrapperCodeGen

    # 如果当前环境中没有为 CPU 设备注册调度
    if get_scheduling_for_device("cpu") is None:
        # 定义 CPU 设备的后端调度字典，支持 cpp 和 halide 调度
        cpu_backends = {"cpp": CppScheduling, "halide": HalideScheduling}
        # 为 CPU 设备注册后端调度
        register_backend_for_device(
            "cpu",
            # 根据配置选择合适的后端调度类来创建实例
            lambda *args, **kwargs: cpu_backends[config.cpu_backend](*args, **kwargs),
            WrapperCodeGen,   # 使用 WrapperCodeGen 生成代码包装
            CppWrapperCpu     # 使用 CppWrapperCpu 包装 CPU 代码
        )

    # 如果当前环境中没有为 CUDA 设备注册调度
    if get_scheduling_for_device("cuda") is None:
        # 定义 CUDA 设备的后端调度字典，支持 triton 和 halide 调度
        cuda_backends = {"triton": CUDACombinedScheduling, "halide": HalideScheduling}
        # 为 CUDA 设备注册后端调度
        register_backend_for_device(
            "cuda",
            # 根据配置选择合适的后端调度类来创建实例
            lambda *args, **kwargs: cuda_backends[config.cuda_backend](*args, **kwargs),
            WrapperCodeGen,   # 使用 WrapperCodeGen 生成代码包装
            CppWrapperCuda    # 使用 CppWrapperCuda 包装 CUDA 代码
        )

    # 如果当前环境中没有为 XPU 设备注册调度
    if get_scheduling_for_device("xpu") is None:
        # 为 XPU 设备注册 Triton 调度
        register_backend_for_device("xpu", TritonScheduling, WrapperCodeGen)
# 将索引添加到列表末尾，防止重排序
def index_prevent_reordering(index: List[sympy.Expr], index_vars, sizes):
    from ..ir import FlexibleLayout

    return [*index, sympy_dot(index_vars, FlexibleLayout.contiguous_strides(sizes))]


# 注册特定设备的操作覆盖
def register_device_op_overrides(device: str, device_op_overrides: DeviceOpOverrides):
    # 将设备操作覆盖存储到全局字典中
    device_op_overrides_dict[device] = device_op_overrides


# 获取特定设备的操作覆盖
def get_device_op_overrides(device: str):
    assert isinstance(device, str)

    # 如果设备操作覆盖字典为空，则导入相关模块的默认操作覆盖
    if not device_op_overrides_dict.keys():
        from .cuda import device_op_overrides  # noqa: F401
        from .xpu import device_op_overrides as xpu_op_overrides  # noqa: F401

    # 如果设备在设备操作覆盖字典中，则返回对应的操作覆盖
    if device in device_op_overrides_dict.keys():
        return device_op_overrides_dict[device]


# 使用缓存装饰器，返回一组布尔运算字符串
@functools.lru_cache(None)
def boolean_ops():
    return (
        "is_inf",
        "is_nan",
        "logical_not",
        "signbit",
        "le",
        "lt",
        "ge",
        "gt",
        "eq",
        "ne",
    )


# 不同数据类型到计算数据类型的映射字典
DTYPE_TO_COMPUTATION_DTYPE = {
    torch.bfloat16: torch.float,
    torch.float16: torch.float,
    **{
        dtype: dtype
        for dtype in [
            torch.bool,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        ]
    },
}


# 数据类型传播类
class DataTypePropagation:
    def __init__(self, body) -> None:
        self.body = body
        # 初始化包含根块和子块图的字典
        self.graphs: Dict[Union[Callable[..., Any], str], Any] = {
            "root": body.root_block.graph
        }
        for k, v in body.subblocks.items():
            self.graphs[k] = v.graph

    # 通过输入推断节点数据类型
    def deduce_node_dtype_by_inputs(self, node: torch.fx.Node):
        inputs = node.all_input_nodes
        # 获取所有非占位符节点的输入节点
        input_nodes = [
            n for n in inputs if isinstance(n, torch.fx.Node) and n.op != "placeholder"
        ]
        if len(input_nodes) == 0:
            return None

        # 检查所有输入节点是否已传播数据类型
        all_input_nodes_propagated = all(
            OptimizationContext.key in n.meta
            and n.meta[OptimizationContext.key].dtype is not None
            for n in input_nodes
        )
        if not all_input_nodes_propagated:
            return None

        # 通过促进所有输入节点的数据类型来推断节点数据类型
        return functools.reduce(
            torch.promote_types,
            [n.meta[OptimizationContext.key].dtype for n in input_nodes],
        )

    # 通过子图推断节点数据类型
    def deduce_node_dtype_by_subgraph(self, node: torch.fx.Node):
        sub_graph = self.graphs[node.target]
        # 传播子图并断言结果非空
        dtype = self.propagate_graph(sub_graph)
        assert dtype
        return dtype
    def deduce_node_dtype(self, node: torch.fx.Node):
        # 如果节点的目标在布尔运算列表中，则返回 torch.bool 类型
        if node.target in boolean_ops():
            return torch.bool

        # 如果节点的操作是占位符，则返回 None
        if node.op == "placeholder":
            return None

        # 如果节点的目标是 "output"
        if node.target == "output":
            # 如果输出节点只有一个参数，则可以推断为输出节点
            if len(node.args) != 1:
                return None

        # 如果节点的目标是以下之一，则返回其最后一个参数的数据类型
        if node.target in (
            "to_dtype",
            "index_expr",
        ):
            return node.args[-1]

        # 如果节点的目标是以下之一，则返回 torch.float 类型
        if node.target in (
            "rand",
            "randn",
        ):
            return torch.float

        # 如果节点的目标是以下之一，则返回 torch.int64 类型
        if node.target in (
            "get_index",
            "index_expr",
            "randint64",
        ):
            return torch.int64

        # 如果节点的目标是以下之一，则返回与缓冲区名相关的数据类型
        if node.target in (
            "load",
            "store",
            "store_reduction",
        ):
            buf_name = node.args[1]
            return V.graph.get_dtype(buf_name)  # type: ignore[arg-type]

        # 如果节点的目标是 operator.getitem，则递归调用 deduce_node_dtype 获取参数的数据类型
        if node.target == operator.getitem:
            return self.deduce_node_dtype(node.args[0])  # type: ignore[arg-type]

        # 如果节点的目标是 "reduction"，则返回第二个参数的数据类型
        if node.target == "reduction":
            return node.args[1]

        # 如果节点的目标是 "constant"，则返回与 DTYPE_TO_COMPUTATION_DTYPE 中最后一个参数相关的数据类型
        if node.target == "constant":
            return DTYPE_TO_COMPUTATION_DTYPE[node.args[-1]]  # type: ignore[index]

        # 如果节点的目标以 "masked_subblock" 开头，则调用 deduce_node_dtype_by_subgraph 来获取数据类型
        if node.target.startswith("masked_subblock"):
            return self.deduce_node_dtype_by_subgraph(node)

        # 否则调用 deduce_node_dtype_by_inputs 来推断节点的数据类型
        return self.deduce_node_dtype_by_inputs(node)

    def propagate_graph(self, graph: torch.fx.Graph):
        # 断言图中有节点
        assert graph.nodes
        graph_dtype = None
        # 对于图中的每个节点
        for node in graph.nodes:
            # 如果节点的元数据中包含 OptimizationContext.key，则使用现有的优化上下文，否则创建一个新的
            if OptimizationContext.key in node.meta:
                opt_ctx = node.meta[OptimizationContext.key]
            else:
                opt_ctx = OptimizationContext()

            # 推断节点的数据类型并将其存储在优化上下文中
            opt_ctx.dtype = self.deduce_node_dtype(node)
            node.meta[OptimizationContext.key] = opt_ctx

            # 如果节点的目标是 "output"，则将图的数据类型设置为此节点的数据类型
            if node.target == "output":
                graph_dtype = opt_ctx.dtype

        # 返回图的数据类型
        return graph_dtype

    def propagate(self):
        # 对根图进行数据类型传播
        self.propagate_graph(self.graphs["root"])

    @classmethod
    def propagate_loopbody(cls, body):
        # 创建类的实例并传播数据类型
        return cls(body).propagate()

    @classmethod
    def propagate_scheduler_node(cls, node):
        # 导入所需的模块
        from ..ir import LoopBody
        from ..scheduler import SchedulerNode

        # 断言节点是 SchedulerNode 类的实例，并且其 _body 属性是 LoopBody 类的实例
        assert isinstance(node, SchedulerNode)
        assert isinstance(node._body, LoopBody)

        # 传播节点的循环体的数据类型
        DataTypePropagation.propagate_loopbody(node._body)
# 包含用于打印表达式的规则，这些规则适用于 C/C++ 和 Python
class ExprPrinter(Printer):

    # 静态方法：在字符串周围添加括号，如果已经在括号中，则不添加
    @staticmethod
    def paren(string):
        # 检查字符串是否完全包含在括号中
        def all_in_parens(string):
            if string[0] != "(" or len(string) < 2:
                return False
            count = 1
            for i, char in enumerate(string[1:]):
                if char == "(":
                    count += 1
                elif char == ")":
                    count -= 1
                if count == 0 and i != len(string) - 2:
                    return False
            assert count == 0
            return True

        # 如果字符串是 CSEVariable 的实例，或者匹配字母、数字、下划线和点号的字符串，或者已经在括号中，则直接返回原始字符串
        if (
            isinstance(string, CSEVariable)
            or re.match(r"^[a-z0-9_.]+$", string, re.I)
            or re.match(r"^\([^)]*\)$", string, re.I)
            or string == ""
        ):
            return string
        # 如果字符串不是已经在括号中，则在字符串周围添加括号
        if all_in_parens(string):
            return string
        return f"({string})"

    # 打印 Relational 表达式
    def _print_Relational(self, expr):
        return f" {expr.rel_op} ".join(map(self.paren, map(self._print, expr.args)))

    # 打印 Mul 表达式
    def _print_Mul(self, expr):
        return "*".join(map(self.paren, map(self._print, expr.args)))

    # 打印 Add 表达式
    def _print_Add(self, expr):
        return " + ".join(map(self.paren, map(self._print, expr.args)))

    # 打印 Mod 表达式
    # 注意：因为 Mod 仅对正数定义，因此在 C/Python 中其行为是一致的
    def _print_Mod(self, expr):
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    # 打印 FloatTrueDiv 表达式
    def _print_FloatTrueDiv(self, expr):
        lhs, rhs = expr.args
        return f"{self.paren(self._print(lhs))} / {self.paren(self._print(rhs))}"

    # 打印 CleanDiv 表达式，实际上调用 _print_FloorDiv 方法
    def _print_CleanDiv(self, expr):
        return self._print_FloorDiv(expr)

    # 打印 Identity 表达式
    def _print_Identity(self, expr):
        return self._print(expr.args[0])

    # 打印 GreaterThan 表达式
    def _print_GreaterThan(self, expr):
        # GreaterThan:          >=
        # StrictlyGreaterThan:  >
        # 随便怎么想...
        return " >= ".join(map(self.paren, map(self._print, expr.args)))

    # 打印 align 表达式，确保其参数只有一个
    # 注意：C 实现被注入到 torch/_inductor/codegen/wrapper.py 中
    def _print_align(self, expr):
        assert len(expr.args) == 1
        return f"align({self._print(expr.args[0])})"

    # 这个方法必须实现，因为 sympy 会将 x * x 合并成 Pow(x, 2)，我们希望打印出 x * x 的形式，
    # 特别是，我们从不生成带有浮点数的 sympy.Pow。
    #
    # 注意：这里的 pow 是自然的，你永远不应该使用内置的 sympy.pow
    # 对于 FloatPow，符号指数应该是 PowByNatural。这些意味着 exp 保证是整数。
    # 定义一个方法来打印指数表达式
    def _print_Pow(self, expr):
        # 将表达式分解为基数和指数
        base, exp = expr.args
        # 打印基数部分
        base = self._print(base)
        # 确保指数为整数
        assert exp == int(exp), exp
        exp = int(exp)
        # 确保指数为非负数
        assert exp >= 0
        if exp > 0:
            # 如果指数大于0，返回基数的exp次幂的字符串形式
            return "*".join([self.paren(base)] * exp)
        else:  # exp == 0
            # 如果指数为0，返回字符串"1"
            return "1"

    # 显式地声明未实现的函数，以防止默认的 sympy 打印行为
    # 默认行为会将 ToFloat(...) 直接输出到你的 IR 中。这里的错误
    # 消息更好，因为它告诉你应该将其放入哪个打印类中。
    def _print_ToFloat(self, expr):
        raise NotImplementedError(f"_print_ToFloat not implemented for {type(self)}")

    def _print_Infinity(self, expr):
        raise NotImplementedError(f"_print_Infinity not implemented for {type(self)}")

    def _print_NegativeInfinity(self, expr):
        raise NotImplementedError(
            f"_print_NegativeInfinity not implemented for {type(self)}"
        )

    def _print_FloorDiv(self, expr):
        raise NotImplementedError(f"_print_FloorDiv not implemented for {type(self)}")

    def _print_PythonMod(self, expr):
        raise NotImplementedError(f"_print_PythonMod not implemented for {type(self)}")

    def _print_IntTrueDiv(self, expr):
        raise NotImplementedError(f"_print_IntTrueDiv not implemented for {type(self)}")

    def _print_PowByNatural(self, expr):
        raise NotImplementedError(
            f"_print_PowByNatural not implemented for {type(self)}"
        )

    def _print_FloatPow(self, expr):
        raise NotImplementedError(f"_print_FloatPow not implemented for {type(self)}")

    def _print_TruncToInt(self, expr):
        raise NotImplementedError(f"_print_TruncToInt not implemented for {type(self)}")

    def _print_RoundToInt(self, expr):
        raise NotImplementedError(f"_print_RoundToInt not implemented for {type(self)}")

    def _print_RoundDecimal(self, expr):
        raise NotImplementedError(
            f"_print_RoundDecimal not implemented for {type(self)}"
        )

    # 注意：某些浮点数操作故意未实现打印功能。
    # 你可以实现它们来快速解除阻塞，但最好问问自己为什么我们没有在张量
    # 宇宙中进行这些计算。

    def _print_TruncToFloat(self, expr):
        raise NotImplementedError(
            f"_print_TruncToFloat not implemented for {type(self)}"
        )

    def doprint(self, expr, *, simplify: bool = True):
        # TODO: 为什么人们会在这里将字符串传递给打印机 :think:
        # 如果启用简化并且表达式是 sympy.Expr 类型且 V.graph 具有 "sizevars" 属性，
        # 则对表达式进行简化处理
        if simplify and isinstance(expr, sympy.Expr) and hasattr(V.graph, "sizevars"):
            expr = V.graph.sizevars.simplify(expr)
        # 调用父类的 doprint 方法来打印表达式
        return super().doprint(expr)
class PythonPrinter(ExprPrinter):
    # 继承自 ExprPrinter 类的 PythonPrinter 类，用于打印特定表达式对象的字符串表示

    def _print_ToFloat(self, expr):
        # 打印将表达式转换为浮点数的字符串表示
        assert len(expr.args) == 1
        return f"float({self._print(expr.args[0])})"

    def _print_ModularIndexing(self, expr):
        # 打印模运算索引的字符串表示
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} // {div})"
        return f"{x} % {mod}"

    def _print_Infinity(self, expr):
        # 打印正无穷的字符串表示
        return "math.inf"

    def _print_NegativeInfinity(self, expr):
        # 打印负无穷的字符串表示
        return "-math.inf"

    # WARNING: this is dangerous for Triton, which has C-style modulus
    def _print_PythonMod(self, expr):
        # 打印 Python 风格的模运算的字符串表示
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    # WARNING: this is dangerous for Triton, which has C-style modulus
    def _print_FloorDiv(self, expr):
        # 打印向下取整除法的字符串表示
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} // {div})"

    # WARNING: this is dangerous for Triton, when lhs, rhs > 2**53, Python
    # does a special algorithm
    def _print_IntTrueDiv(self, expr):
        # 打印整数真除的字符串表示
        lhs, rhs = expr.args
        return f"{self.paren(self._print(lhs))} / {self.paren(self._print(rhs))}"

    def _helper_sqrt(self, expr):
        # 辅助函数，用于打印平方根函数的字符串表示
        return f"math.sqrt({self._print(expr)})"

    def _print_OpaqueUnaryFn_sqrt(self, expr):
        # 打印平方根的字符串表示
        return self._helper_sqrt(expr.args[0])

    def _print_FloatPow(self, expr):
        # 打印浮点数幂运算的字符串表示
        base, exp = expr.args
        return f"{self.paren(self._print(base))} ** {self.paren(self._print(exp))}"

    # TODO: Not sure this works with Triton, even when base/exp are integral
    def _print_PowByNatural(self, expr):
        # 打印自然数幂运算的字符串表示
        base, exp = expr.args
        return f"{self.paren(self._print(base))} ** {self.paren(self._print(exp))}"

    def _print_floor(self, expr):
        # 打印向下取整函数的字符串表示
        assert len(expr.args) == 1
        return f"math.floor({self._print(expr.args[0])})"

    def _print_FloorToInt(self, expr):
        # 打印向下取整到整数的字符串表示
        assert len(expr.args) == 1
        return f"math.floor({self._print(expr.args[0])})"

    def _print_TruncToInt(self, expr):
        # 打印截断到整数的字符串表示
        assert len(expr.args) == 1
        # This also could have been int(), they'll do the same thing for float
        return f"math.trunc({self._print(expr.args[0])})"

    def _print_ceiling(self, expr):
        # 打印向上取整函数的字符串表示
        assert len(expr.args) == 1
        return f"math.ceil({self._print(expr.args[0])})"

    def _print_CeilToInt(self, expr):
        # 打印向上取整到整数的字符串表示
        assert len(expr.args) == 1
        return f"math.ceil({self._print(expr.args[0])})"

    def _print_Abs(self, expr):
        # 打印取绝对值函数的字符串表示
        assert len(expr.args) == 1
        return f"abs({self._print(expr.args[0])})"

    # NB: It's expected that we've made explicit any promotion in the sympy
    # expression, so it doesn't matter that Python max/min doesn't perform
    # promotion
    def _print_Max(self, expr):
        # 打印最大值函数的字符串表示
        assert len(expr.args) >= 2
        return f"max({', '.join(map(self._print, expr.args))})"
    # 定义一个方法用于打印最小值表达式
    def _print_Min(self, expr):
        # 断言表达式参数至少有两个
        assert len(expr.args) >= 2
        # 返回格式化后的最小值函数调用表达式
        return f"min({', '.join(map(self._print, expr.args))})"

    # 定义一个方法用于打印余弦函数表达式
    def _print_OpaqueUnaryFn_cos(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的余弦函数调用表达式
        return f"math.cos({self._print(expr.args[0])})"

    # 定义一个方法用于打印双曲余弦函数表达式
    def _print_OpaqueUnaryFn_cosh(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的双曲余弦函数调用表达式
        return f"math.cosh({self._print(expr.args[0])})"

    # 定义一个方法用于打印反余弦函数表达式
    def _print_OpaqueUnaryFn_acos(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的反余弦函数调用表达式
        return f"math.acos({self._print(expr.args[0])})"

    # 定义一个方法用于打印正弦函数表达式
    def _print_OpaqueUnaryFn_sin(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的正弦函数调用表达式
        return f"math.sin({self._print(expr.args[0])})"

    # 定义一个方法用于打印双曲正弦函数表达式
    def _print_OpaqueUnaryFn_sinh(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的双曲正弦函数调用表达式
        return f"math.sinh({self._print(expr.args[0])})"

    # 定义一个方法用于打印反正弦函数表达式
    def _print_OpaqueUnaryFn_asin(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的反正弦函数调用表达式
        return f"math.asin({self._print(expr.args[0])})"

    # 定义一个方法用于打印正切函数表达式
    def _print_OpaqueUnaryFn_tan(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的正切函数调用表达式
        return f"math.tan({self._print(expr.args[0])})"

    # 定义一个方法用于打印双曲正切函数表达式
    def _print_OpaqueUnaryFn_tanh(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的双曲正切函数调用表达式
        return f"math.tanh({self._print(expr.args[0])})"

    # 定义一个方法用于打印反正切函数表达式
    def _print_OpaqueUnaryFn_atan(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的反正切函数调用表达式
        return f"math.atan({self._print(expr.args[0])})"

    # 定义一个方法用于打印四舍五入到整数表达式
    def _print_RoundToInt(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回格式化后的四舍五入到整数调用表达式
        return f"round({self._print(expr.args[0])})"

    # 定义一个方法用于打印四舍五入到指定小数位数表达式
    def _print_RoundDecimal(self, expr):
        # 断言表达式参数恰好有两个
        assert len(expr.args) == 2
        # 将表达式参数解构为 number 和 ndigits
        number, ndigits = expr.args
        # 断言 ndigits 是 sympy.Integer 类型
        assert isinstance(ndigits, sympy.Integer)
        # 返回格式化后的四舍五入到指定小数位数调用表达式
        return f"round({self._print(number)}, {ndigits})"
class OpOverrides:
    # 定义一个操作重载类，用于扩展和重写操作符和函数
    def __init__(self, parent):
        super().__init__()
        self._parent = parent
        # 初始化函数，接受一个父对象作为参数，并将其保存在实例变量中

    def __getattr__(self, item):
        # 当尝试访问当前对象中不存在的属性时，委托给父对象进行处理
        return getattr(self._parent, item)

    @staticmethod
    def identity(value):
        # 返回输入值本身，用于触发公共子表达式消除（CSE）
        return value

    @staticmethod
    def constant(value, dtype):
        # 返回给定值的字符串表示形式，用于表示常量
        return repr(value)

    @staticmethod
    def reciprocal(x):
        # 返回 x 的倒数
        return ops.truediv(ops.constant(1, torch.int32), x)

    @staticmethod
    def square(x):
        # 返回 x 的平方
        return ops.mul(x, x)

    @staticmethod
    def erfc(x):
        # 返回 1 减去 x 的误差函数
        return ops.sub(ops.constant(1, torch.float32), ops.erf(x))

    @staticmethod
    def erfcx(x):
        # 返回 exp(x^2) 与 erf(x) 乘积的结果
        return ops.mul(ops.exp(ops.square(x)), ops.erfc(x))

    @staticmethod
    def expm1(x):
        # 返回 exp(x) 减去 1 的结果
        return ops.sub(ops.exp(x), ops.constant(1, torch.float32))

    @staticmethod
    def log10(x):
        # 返回以 10 为底 x 的对数
        return ops.mul(ops.log(x), ops.constant(1 / math.log(10), torch.float32))

    @staticmethod
    def log2(x):
        # 返回以 2 为底 x 的对数
        return ops.mul(ops.log(x), ops.constant(1 / math.log(2), torch.float32))

    @staticmethod
    def exp2(x):
        # 返回以 2 为底 x 的指数
        return ops.exp(ops.mul(x, ops.constant(math.log(2), torch.float32)))

    @staticmethod
    def log1p(x):
        # 返回 log(1 + x) 的结果
        return ops.log(ops.add(x, ops.constant(1, torch.int32)))

    @staticmethod
    def sigmoid(x):
        # 返回 sigmoid 函数的结果，用 ops.exp 和 ops.truediv 实现
        one = ops.constant(1, torch.int32)
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(x))))

    @staticmethod
    def libdevice_sigmoid(x):
        # 返回 libdevice sigmoid 函数的结果，用 ops.exp 和 ops.truediv 实现
        one = ops.constant(1, torch.int32)
        return ops.truediv(one, ops.add(one, ops.libdevice_exp(ops.neg(x))))

    @staticmethod
    def relu(x):
        # 返回 ReLU 函数的结果，即 x 与 0 的最大值
        return ops.maximum(x, ops.constant(0, torch.int32))

    @staticmethod
    def libdevice_abs(x):
        # 返回 libdevice abs 函数的结果，即 x 的绝对值
        return ops.abs(x)

    @staticmethod
    def libdevice_sqrt(x):
        # 返回 libdevice sqrt 函数的结果，即 x 的平方根
        return ops.sqrt(x)

    @staticmethod
    def libdevice_cos(x):
        # 返回 libdevice cos 函数的结果，即 x 的余弦值
        return ops.cos(x)

    @staticmethod
    def libdevice_sin(x):
        # 返回 libdevice sin 函数的结果，即 x 的正弦值
        return ops.sin(x)

    @staticmethod
    def libdevice_log(x):
        # 返回 libdevice log 函数的结果，即 x 的自然对数
        return ops.log(x)

    @staticmethod
    def libdevice_exp(x):
        # 返回 libdevice exp 函数的结果，即 e 的 x 次方
        return ops.exp(x)

    @staticmethod
    def bitwise_not(x):
        # 返回 x 的按位取反结果，使用 ExprPrinter.paren 函数打印表达式
        return f"~{ExprPrinter.paren(x)}"

    @staticmethod
    def logical_not(a):
        # 返回逻辑非操作的结果，使用 ExprPrinter.paren 函数打印表达式
        return f"{ExprPrinter.paren(a)} == 0"

    @staticmethod
    def bitwise_and(x, y):
        # 返回 x 和 y 的按位与结果，使用 ExprPrinter.paren 函数打印表达式
        return f"{ExprPrinter.paren(x)} & {ExprPrinter.paren(y)}"

    @staticmethod
    def bitwise_or(x, y):
        # 返回 x 和 y 的按位或结果，使用 ExprPrinter.paren 函数打印表达式
        return f"{ExprPrinter.paren(x)} | {ExprPrinter.paren(y)}"

    @staticmethod
    def bitwise_xor(x, y):
        # 返回 x 和 y 的按位异或结果，使用 ExprPrinter.paren 函数打印表达式
        return f"{ExprPrinter.paren(x)} ^ {ExprPrinter.paren(y)}"

    @staticmethod
    def bitwise_left_shift(x, y):
        # 返回 x 向左移位 y 次的结果，使用 ExprPrinter.paren 函数打印表达式
        return f"{ExprPrinter.paren(x)} << {ExprPrinter.paren(y)}"

    @staticmethod
    def bitwise_right_shift(x, y):
        # 返回 x 向右移位 y 次的结果，使用 ExprPrinter.paren 函数打印表达式
        return f"{ExprPrinter.paren(x)} >> {ExprPrinter.paren(y)}"
    # 定义一个静态方法remainder，计算a除以b的余数，并根据余数的符号进行调整
    def remainder(a, b):
        # 计算a除以b的余数
        r = ops.mod(a, b)
        # 构建条件：余数不等于0且其符号与b的符号不同
        cond = ops.and_(
            ops.ne(r, ops.constant(0, torch.int32)),
            ops.ne(ops.signbit(r), ops.signbit(b)),
        )
        # 根据条件选择返回值：若条件成立，则返回r加b的结果，否则返回r本身
        return ops.where(cond, ops.add(r, b), r)

    # 定义一个静态方法trunc_to_int，将浮点数a截断为整数，并转换为指定的dtype类型
    @staticmethod
    def trunc_to_int(a, dtype):
        return ops.to_dtype(ops.trunc(a), dtype)

    # 定义一个静态方法floor_to_int，将浮点数a向下取整为整数，并转换为指定的dtype类型
    @staticmethod
    def floor_to_int(a, dtype):
        return ops.to_dtype(ops.floor(a), dtype)

    # 定义一个静态方法ceil_to_int，将浮点数a向上取整为整数，并转换为指定的dtype类型
    @staticmethod
    def ceil_to_int(a, dtype):
        return ops.to_dtype(ops.ceil(a), dtype)

    # 定义一个静态方法round_to_int，将浮点数a四舍五入为整数，并转换为指定的dtype类型
    @staticmethod
    def round_to_int(a, dtype):
        return ops.to_dtype(ops.round(a), dtype)

    # 定义一个静态方法int_truediv，计算a除以b的浮点数结果
    @staticmethod
    def int_truediv(a, b):
        # TODO: this is wrong
        # TODO: an easy bandaid is to generate runtime asserts that it's
        # <= 2**53, which is when this equation is correct
        # 执行整数除法，返回浮点数结果
        return ops.truediv(a, b)

    # 定义一个类方法load_seed，加载名为name的种子数据，使用整数offset作为偏移量
    @classmethod
    def load_seed(name, offset):
        return ops.load(name, sympy.Integer(offset))

    # 定义一个类方法_initialize_pointwise_overrides，初始化逐点操作的覆盖实现
    @classmethod
    def _initialize_pointwise_overrides(cls, target):
        # 断言目标target必须是{"triton", "cpp", "cppvec"}之一
        assert target in {"triton", "cpp", "cppvec"}, target

        # 遍历pointwise_overrides_data字典中的每个函数名funcname及其数据data
        for funcname, data in pointwise_overrides_data.items():
            # 获取目标target对应的实现impl
            impl = getattr(data, target)
            # 如果实现为None，则跳过当前函数名funcname的设置
            if impl is None:
                continue
            # 将实现impl作为静态方法绑定到当前类(cls)的函数名funcname上
            setattr(cls, funcname, staticmethod(impl))
# 定义一个数据类 OverridesData，用于存储特定函数的相关信息
@dataclasses.dataclass
class OverridesData:
    name: str  # 函数名称
    cpp: Callable[..., str]  # 生成 C++ 代码的回调函数
    triton: Optional[Callable[..., str]] = None  # Triton 实现的回调函数（如果有）
    cppvec: Optional[Callable[..., str]] = None  # 生成 C++ 向量化代码的回调函数（如果有）
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND = (
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )  # 元素级类型提升方式，默认为默认方式

# 定义一个字典 pointwise_overrides_data，存储不同函数名称对应的 OverridesData 对象
pointwise_overrides_data: Dict[str, OverridesData] = dict(
    # airy_ai 函数的配置
    airy_ai=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"airy_ai_forward({x})",  # 生成 airy_ai 的 C++ 前向函数调用代码
        name="special_airy_ai",
    ),
    # bessel_j0 函数的配置
    bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j0_forward({x})",  # 生成 bessel_j0 的 C++ 前向函数调用代码
        triton=lambda x: f"libdevice.j0({x})",  # Triton 实现的 bessel_j0 函数调用代码
        name="special_bessel_j0",
    ),
    # bessel_j1 函数的配置
    bessel_j1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j1_forward({x})",  # 生成 bessel_j1 的 C++ 前向函数调用代码
        triton=lambda x: f"libdevice.j1({x})",  # Triton 实现的 bessel_j1 函数调用代码
        name="special_bessel_j1",
    ),
    # bessel_y0 函数的配置
    bessel_y0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y0_forward({x})",  # 生成 bessel_y0 的 C++ 前向函数调用代码
        triton=lambda x: f"libdevice.y0({x})",  # Triton 实现的 bessel_y0 函数调用代码
        name="special_bessel_y0",
    ),
    # bessel_y1 函数的配置
    bessel_y1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y1_forward({x})",  # 生成 bessel_y1 的 C++ 前向函数调用代码
        triton=lambda x: f"libdevice.y1({x})",  # Triton 实现的 bessel_y1 函数调用代码
        name="special_bessel_y1",
    ),
    # digamma 函数的配置
    digamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_digamma({x})",  # 生成 digamma 的 C++ 计算函数调用代码
        cppvec=lambda x: f"{x}.digamma()",  # 生成 digamma 的 C++ 向量化代码调用
        name="digamma",
    ),
    # erfcx 函数的配置
    erfcx=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_erfcx({x})",  # 生成 erfcx 的 C++ 计算函数调用代码
        triton=lambda x: f"libdevice.erfcx({x})",  # Triton 实现的 erfcx 函数调用代码
        name="special_erfcx",
    ),
    # fma 函数的配置
    fma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y, z: f"std::fma({x}, {y}, {z})",  # 生成 fma 的 C++ 标准库函数调用代码
        cppvec=lambda x, y, z: f"fmadd({x}, {y}, {z})",  # 生成 fma 的 C++ 向量化代码调用
        triton=lambda x, y, z: f"libdevice.fma({x}, {y}, {z})",  # Triton 实现的 fma 函数调用代码
        name="fma",
    ),
    # igamma 函数的配置
    igamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",  # 生成 igamma 的 C++ 计算函数调用代码
        name="igamma",
    ),
    # igammac 函数的配置
    igammac=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",  # 生成 igammac 的 C++ 计算函数调用代码
        name="igammac",
    ),
    # entr 函数的配置，没有 cpp 和 triton 实现，作为分解定义
    # erf, erfc 函数的配置
)
    gammainc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",
        name="special_gammainc",
    ),
    # 定义特殊函数gammainc，将整数转换为浮点数进行运算，使用C++函数calc_igamma进行计算

    gammaincc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",
        name="special_gammaincc",
    ),
    # 定义特殊函数gammaincc，将整数转换为浮点数进行运算，使用C++函数calc_igammac进行计算

    i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i0({x})",
        cppvec=lambda x: f"{x}.i0()",
        name="i0",
    ),
    # 定义特殊函数i0，将整数转换为浮点数进行运算，使用C++函数calc_i0进行计算，同时支持Triton和cppvec格式

    i0e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0e({x})",
        cppvec=lambda x: f"{x}.i0e()",
        name="special_i0e",
    ),
    # 定义特殊函数i0e，将整数转换为浮点数进行运算，使用C++函数calc_i0e进行计算，同时支持cppvec格式

    i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i1({x})",
        name="special_i1",
    ),
    # 定义特殊函数i1，将整数转换为浮点数进行运算，使用C++函数calc_i1进行计算，同时支持Triton格式

    i1e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1e({x})",
        name="special_i1e",
    ),
    # 定义特殊函数i1e，将整数转换为浮点数进行运算，使用C++函数calc_i1e进行计算

    log_ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_log_ndtr({x})",
        name="special_log_ndtr",
    ),
    # 定义特殊函数log_ndtr，将整数转换为浮点数进行运算，使用C++函数calc_log_ndtr进行计算

    modified_bessel_i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i0_forward({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i0({x})",
        name="special_modified_bessel_i0",
    ),
    # 定义特殊函数modified_bessel_i0，将整数转换为浮点数进行运算，使用C++函数modified_bessel_i0_forward进行计算，同时支持Triton格式

    modified_bessel_i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i1_forward({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i1({x})",
        name="special_modified_bessel_i1",
    ),
    # 定义特殊函数modified_bessel_i1，将整数转换为浮点数进行运算，使用C++函数modified_bessel_i1_forward进行计算，同时支持Triton格式

    modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k0_forward({x})",
        name="special_modified_bessel_k0",
    ),
    # 定义特殊函数modified_bessel_k0，将整数转换为浮点数进行运算，使用C++函数modified_bessel_k0_forward进行计算

    modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k1_forward({x})",
        name="special_modified_bessel_k1",
    ),
    # 定义特殊函数modified_bessel_k1，将整数转换为浮点数进行运算，使用C++函数modified_bessel_k1_forward进行计算

    ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtr({x})",
        name="special_ndtr",
    ),
    # 定义特殊函数ndtr，将整数转换为浮点数进行运算，使用C++函数calc_ndtr进行计算

    ndtri=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtri({x})",
        name="special_ndtri",
    ),
    # 定义特殊函数ndtri，将整数转换为浮点数进行运算，使用C++函数calc_ndtri进行计算
    polygamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_polygamma({y}, {x})",
        name="polygamma",
    ),
    # psi - alias to digamma
    # round
    scaled_modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k0_forward({x})",
        name="special_scaled_modified_bessel_k0",
    ),
    scaled_modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k1_forward({x})",
        name="special_scaled_modified_bessel_k1",
    ),
    # sinc
    spherical_bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"spherical_bessel_j0_forward({x})",
        name="special_spherical_bessel_j0",
    ),
    zeta=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"zeta({x}, {y})",
        name="special_zeta",
    ),
    chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_t_forward({x}, {y})",
        name="special_chebyshev_polynomial_t",
    ),
    chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_u_forward({x}, {y})",
        name="special_chebyshev_polynomial_u",
    ),
    chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_v_forward({x}, {y})",
        name="special_chebyshev_polynomial_v",
    ),
    chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_w_forward({x}, {y})",
        name="special_chebyshev_polynomial_w",
    ),
    legendre_polynomial_p=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"legendre_polynomial_p_forward({x}, {y})",
        name="special_legendre_polynomial_p",
    ),
    shifted_chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_t_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_t",
    ),
    shifted_chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_u_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_u",
    ),
    shifted_chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_v_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_v",
    ),
    # 定义一个名为 shifted_chebyshev_polynomial_v 的 OverridesData 对象，
    # 指定类型提升方式为整数到浮点数，
    # 定义了一个用于 C++ 的 lambda 函数，用于生成特定的前向推导式，
    # 设置对象名称为 "special_shifted_chebyshev_polynomial_v"
    
    shifted_chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_w_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_w",
    ),
    # 定义一个名为 shifted_chebyshev_polynomial_w 的 OverridesData 对象，
    # 指定类型提升方式为整数到浮点数，
    # 定义了一个用于 C++ 的 lambda 函数，用于生成特定的前向推导式，
    # 设置对象名称为 "special_shifted_chebyshev_polynomial_w"
    
    hermite_polynomial_h=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_h_forward({x}, {y})",
        name="special_hermite_polynomial_h",
    ),
    # 定义一个名为 hermite_polynomial_h 的 OverridesData 对象，
    # 指定类型提升方式为整数到浮点数，
    # 定义了一个用于 C++ 的 lambda 函数，用于生成特定的前向推导式，
    # 设置对象名称为 "special_hermite_polynomial_h"
    
    hermite_polynomial_he=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_he_forward({x}, {y})",
        name="special_hermite_polynomial_he",
    ),
    # 定义一个名为 hermite_polynomial_he 的 OverridesData 对象，
    # 指定类型提升方式为整数到浮点数，
    # 定义了一个用于 C++ 的 lambda 函数，用于生成特定的前向推导式，
    # 设置对象名称为 "special_hermite_polynomial_he"
    
    laguerre_polynomial_l=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"laguerre_polynomial_l_forward({x}, {y})",
        name="special_laguerre_polynomial_l",
    ),
    # 定义一个名为 laguerre_polynomial_l 的 OverridesData 对象，
    # 指定类型提升方式为整数到浮点数，
    # 定义了一个用于 C++ 的 lambda 函数，用于生成特定的前向推导式，
    # 设置对象名称为 "special_laguerre_polynomial_l"
# 使用 mypy 检查 OpOverrides 协议是否被正确实现
def _typecheck_OpOverrides(h: OpOverrides) -> OpsHandler[str]:
    # 返回参数 h，表明 OpOverrides 协议已正确实现
    return h


class DeferredLine(DeferredLineBase):
    """表示可通过将名称添加到 V.graph.removed_buffers 来“取消写入”的行"""

    def __init__(self, name, line):
        super().__init__(line)
        self.name = name
        # 确保 line 不是 DeferredLineBase 的实例，避免错误状态

    def __call__(self):
        # 检查 self.name 在以下所有数据结构中是否都不存在
        if all(
            self.name not in x
            for x in (
                V.graph.removed_buffers,
                V.kernel.removed_buffers,
                V.graph.inplaced_to_remove,
                V.kernel.inplaced_to_remove,
            )
        ):
            return self.line  # 如果不存在于任何数据结构中，则返回行内容
        return None  # 否则返回 None

    def _new_line(self, line):
        # 返回一个新的 DeferredLine 实例，用于处理新的行内容
        return DeferredLine(self.name, line)


class BracesBuffer(IndentedBuffer):
    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            # 根据指定的偏移量增加或减少缩进，并生成相应的大括号
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("}")
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                self._indent -= 1
                self.writeline("}")

        return ctx()


class InplacedBuffer(NamedTuple):
    """用于表示具有内部名称和其他名称列表的命名元组"""

    inner_name: str  # 内部名称
    other_names: List[str]  # 其他名称列表


class KernelArgs:
    @staticmethod
    def _lookup(prefix, odict, name):
        # 如果名称不在 odict 中，则将其添加到 odict 中，并返回相应的名称
        assert isinstance(name, (str, sympy.Symbol))
        if name not in odict:
            odict[name] = f"{prefix}{len(odict)}"
        return odict[name]

    def __init__(self, sizevars=None):
        # 初始化 KernelArgs 类的实例
        self.input_buffers = dict()     # 输入缓冲区字典
        self.output_buffers = dict()    # 输出缓冲区字典
        self.inplace_buffers = dict()   # 就地缓冲区字典
        self.sizevars = sizevars or dict()  # 尺寸变量字典，如果未提供则为空字典
        self.workspace_arg = None      # 工作空间参数，默认为 None

    def __repr__(self):
        # 返回 KernelArgs 实例的字符串表示形式
        return "KernelArgs({})".format(
            ", ".join(
                map(
                    repr,
                    [
                        self.input_buffers,
                        self.output_buffers,
                        self.inplace_buffers,
                        self.sizevars,
                    ],
                )
            )
        )

    def _buffer_is_marked_removed(self, name):
        # 检查给定的名称是否标记为已移除的缓冲区
        return isinstance(name, str) and name.startswith("REMOVED")
    # 根据给定的名称从对象 V 中查找调度器对象，并获取可能的突变后名称
    def input(self, name):
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        # 断言确保名称不在已移除的缓冲区列表中
        assert name not in V.graph.removed_buffers, name
        # 如果名称在输出缓冲区中，则返回对应的数据
        if name in self.output_buffers:
            return self.output_buffers[name]
        # 如果名称在就地缓冲区中，则返回其内部名称
        if name in self.inplace_buffers:
            return self.inplace_buffers[name].inner_name
        # 如果名称以 "seed" 开头，则调用 _lookup 方法查找输入缓冲区中对应的值
        if name.startswith("seed"):
            return self._lookup("seed", self.input_buffers, name)
        # 否则调用 _lookup 方法查找输入缓冲区中对应的值
        return self._lookup("in_ptr", self.input_buffers, name)

    # 根据给定的名称从对象 V 中查找调度器对象，并获取可能的突变后名称
    def output(self, name):
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        # 断言确保名称不在已移除的缓冲区列表中
        assert name not in V.graph.removed_buffers, name
        # 如果名称在就地缓冲区中，则返回其内部名称
        if name in self.inplace_buffers:
            return self.inplace_buffers[name].inner_name
        # 否则调用 _lookup 方法查找输出缓冲区中对应的值
        return self._lookup("out_ptr", self.output_buffers, name)

    # 将输入名称和输出名称的关系存储为就地缓冲区
    def make_inplace(self, input_name, output_name):
        # 断言确保输出名称不在就地缓冲区中
        assert output_name not in self.inplace_buffers
        # 如果输入名称在就地缓冲区中已存在，则更新其关联的输出名称
        if input_name in self.inplace_buffers:
            buf = self.inplace_buffers[input_name]
            buf.other_names.append(output_name)
            self.inplace_buffers[output_name] = buf
        # 否则创建一个新的就地缓冲区对象，并存储到输入和输出名称对应的位置
        else:
            buf = InplacedBuffer(
                f"in_out_ptr{len(unique(self.inplace_buffers.values()))}",
                [input_name, output_name],
            )
            self.inplace_buffers[input_name] = buf
            self.inplace_buffers[output_name] = buf

    # 设置工作空间参数并返回相应的指针名称和偏移量
    def workspace(self, nbytes: sympy.Expr, zero_fill: bool):
        # 如果工作空间参数尚未设置，则创建新的 WorkspaceArg 对象
        if self.workspace_arg is None:
            self.workspace_arg = WorkspaceArg(nbytes, zero_fill)
            return "ws_ptr", 0

        # 计算新的偏移量，并更新工作空间参数对象
        offset = self.workspace_arg.nbytes
        zero_fill = zero_fill or self.workspace_arg.zero_fill
        self.workspace_arg = WorkspaceArg(offset + nbytes, zero_fill)
        return "ws_ptr", offset

    # 根据给定的名称和值获取对应的偏移量
    def seed_offset(self, name, value):
        # 如果值已经在 sizevars 中存在，则直接返回其对应的偏移量
        if value in self.sizevars:
            return self.sizevars[value]
        # 如果名称已经在 sizevars 的值中存在，则为名称添加序号后返回
        if name in self.sizevars.values():
            name = (
                f"{name}{sum(1 for v in self.sizevars.values() if v.startswith(name))}"
            )
        # 将值和名称存储到 sizevars 中，并返回名称
        self.sizevars[value] = name
        return name

    # 根据给定的名称获取对应的大小
    def size(self, name):
        # 如果名称是字符串 "seed"，则直接设置 sizevars 中的 "seed" 为 "seed" 并返回
        if str(name) == "seed":
            self.sizevars["seed"] = "seed"
            return "seed"
        # 否则调用 _lookup 方法查找 sizevars 中对应的值
        return self._lookup("ks", self.sizevars, name)

    # 返回输入缓冲区、输出缓冲区和 sizevars 中的所有键的链式迭代器
    def call_names(self):
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    # 根据给定的 buf 和数据类型 dtype 返回对应的 buf
    def wrap_ptr_arg(self, buf, dtype):
        return buf

    # 根据给定的 size 返回其字符串表示形式
    def wrap_size_arg(self, size):
        return str(size)
    # 定义一个方法，用于生成用于 C++ 调用的参数定义、调用参数和参数类型列表
    def cpp_argdefs(self):
        # 导入必要的模块和变量
        from .cpp_utils import DTYPE_TO_CPP, INDEX_TYPE
        
        # 初始化空列表，用于存放调用参数、参数定义和参数类型
        call_args = []
        arg_defs = []
        arg_types = []
        
        # 遍历所有输入缓冲的唯一值
        for inplaced in unique(self.inplace_buffers.values()):
            # 如果当前缓冲被标记为已移除，则跳过
            if self._buffer_is_marked_removed(inplaced):
                continue
            # 获取外部名称和内部名称
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            # 获取外部名称对应的数据类型
            dtype = V.graph.get_dtype(outer)
            # 获取对应的 C++ 数据类型
            cpp_dtype = DTYPE_TO_CPP[dtype]
            # 添加参数定义，形如 "数据类型* 内部名称"
            arg_defs.append(f"{cpp_dtype}* {inner}")
            # 调用包装函数，将外部名称和数据类型包装为指针参数，并添加到调用参数列表中
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            # 添加参数类型，形如 "数据类型*"
            arg_types.append(f"{cpp_dtype}*")
        
        # 遍历所有输入缓冲字典中的键值对
        for outer, inner in self.input_buffers.items():
            # 如果外部名称已在原地操作缓冲中，则跳过
            if outer in self.inplace_buffers:
                continue
            # 获取外部名称对应的数据类型
            dtype = V.graph.get_dtype(outer)
            # 获取对应的 C++ 数据类型
            cpp_dtype = DTYPE_TO_CPP[dtype]
            # 添加参数定义，形如 "const 数据类型* 内部名称"
            arg_defs.append(f"const {cpp_dtype}* {inner}")
            # 调用包装函数，将外部名称和数据类型包装为常量指针参数，并添加到调用参数列表中
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            # 添加参数类型，形如 "const 数据类型*"
            arg_types.append(f"const {cpp_dtype}*")
        
        # 遍历所有输出缓冲字典中的键值对
        for outer, inner in self.output_buffers.items():
            # 如果外部名称已在原地操作缓冲中或内部名称已被标记为已移除，则跳过
            if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
                continue
            # 获取外部名称对应的数据类型
            dtype = V.graph.get_dtype(outer)
            # 获取对应的 C++ 数据类型
            cpp_dtype = DTYPE_TO_CPP[dtype]
            # 添加参数定义，形如 "数据类型* 内部名称"
            arg_defs.append(f"{cpp_dtype}* {inner}")
            # 调用包装函数，将外部名称和数据类型包装为指针参数，并添加到调用参数列表中
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            # 添加参数类型，形如 "数据类型*"
            arg_types.append(f"{cpp_dtype}*")
        
        # 遍历所有尺寸变量字典中的键值对
        for outer, inner in self.sizevars.items():
            # 添加参数定义，形如 "const INDEX_TYPE 内部名称"
            arg_defs.append(f"const {INDEX_TYPE} {inner}")
            # 调用包装函数，将外部名称包装为尺寸参数，并添加到调用参数列表中
            call_args.append(self.wrap_size_arg(outer))
            # 添加参数类型，形如 "const INDEX_TYPE"
            arg_types.append(f"const {INDEX_TYPE}")
            # 如果存在包装代码，则确保计算尺寸已完成
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        
        # 断言工作空间参数为空，因为在 CPU 上不支持工作空间
        assert self.workspace_arg is None, "Workspace not supported on CPU"
        
        # 返回参数定义列表
        return arg_defs, call_args, arg_types
    # 获取 Python 函数参数的定义列表
    def python_argdefs(self):
        arg_defs = []  # 存储参数定义的列表
        call_args = []  # 存储调用参数的列表
        arg_types = []  # 存储参数类型的列表
        precompile_args: List[Union[TensorArg, SizeArg, WorkspaceArg]] = []  # 存储预编译参数的列表，可以是 TensorArg, SizeArg 或 WorkspaceArg 类型的对象

        # 遍历所有唯一的 inplace 缓冲区值
        for inplaced in unique(self.inplace_buffers.values()):
            # 如果缓冲区被标记为已移除，则跳过
            if self._buffer_is_marked_removed(inplaced):
                continue
            # 将缓冲区的内部名称添加到参数定义列表
            arg_defs.append(inplaced.inner_name)
            # 将缓冲区的最后一个别名添加到调用参数列表
            call_args.append(inplaced.other_names[-1])
            # 获取缓冲区最后一个别名的数据类型，并添加到参数类型列表
            arg_types.append(V.graph.get_dtype(inplaced.other_names[-1]))
            # 创建对应的 TensorArg 对象，并添加到预编译参数列表
            precompile_args.append(
                TensorArg(
                    name=inplaced.inner_name,
                    buffer=inplaced.other_names[-1],
                    dtype=V.graph.get_dtype(inplaced.other_names[-1]),
                )
            )

        # 遍历输入缓冲区和输出缓冲区的键值对
        for outer, inner in chain(
            self.input_buffers.items(), self.output_buffers.items()
        ):
            # 如果键在 inplace 缓冲区中或者值被标记为已移除，则跳过
            if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
                continue
            # 将值（缓冲区内部名称）添加到参数定义列表
            arg_defs.append(inner)
            # 将键（缓冲区外部名称）添加到调用参数列表
            call_args.append(outer)
            # 获取键的数据类型，并添加到参数类型列表
            arg_types.append(V.graph.get_dtype(outer))
            # 创建对应的 TensorArg 对象，并添加到预编译参数列表
            precompile_args.append(
                TensorArg(
                    name=inner,
                    buffer=outer,
                    dtype=V.graph.get_dtype(outer),
                )
            )

        # 遍历大小变量字典的键值对
        for outer, inner in self.sizevars.items():
            # 将内部名称（大小变量）添加到参数定义列表
            arg_defs.append(inner)
            # 将外部名称（大小变量的键）添加到调用参数列表
            call_args.append(outer)
            # 获取外部名称的类型，并添加到参数类型列表
            arg_types.append(type(outer))
            # 创建 SizeArg 对象，并添加到预编译参数列表
            precompile_args.append(SizeArg(inner, outer))
            # 如果存在图形包装代码，则确保计算外部名称的大小
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)

        # 如果存在工作空间参数，则添加到参数定义列表、调用参数列表和预编译参数列表中
        if self.workspace_arg is not None:
            arg_defs.append("ws_ptr")
            call_args.append("workspace")
            precompile_args.append(self.workspace_arg)

        # 返回参数定义列表、调用参数列表、预编译参数列表和参数类型列表
        return arg_defs, call_args, precompile_args, arg_types

    # 返回具有别名的 inplace 缓冲区对
    def aliases(self):
        for inplaced in unique(self.inplace_buffers.values()):
            # 如果缓冲区被标记为已移除，则跳过
            if self._buffer_is_marked_removed(inplaced):
                continue
            # 遍历缓冲区的所有别名
            for other in inplaced.other_names:
                # 如果别名在要移除的 inplace 列表中，则跳过
                if (
                    other in V.graph.inplaced_to_remove
                    or other in V.kernel.inplaced_to_remove
                ):
                    continue
                # 如果别名在输入缓冲区中，则生成对应的输入缓冲区和缓冲区内部名称的别名对
                if other in self.input_buffers:
                    yield self.input_buffers[other], inplaced.inner_name
                # 如果别名在输出缓冲区中，则生成对应的输出缓冲区和缓冲区内部名称的别名对
                if other in self.output_buffers:
                    yield self.output_buffers[other], inplaced.inner_name

    # 检查给定名称的缓冲区是否已移除
    def is_removed(self, name):
        # 定义内部函数检查缓冲区是否已移除
        def _is_removed(name, buffers):
            return name not in buffers or self._buffer_is_marked_removed(buffers[name])

        # 返回输出缓冲区和 inplace 缓冲区中指定名称的移除状态
        return _is_removed(name, self.output_buffers) and _is_removed(
            name, self.inplace_buffers
        )

    # 包含 inplace 缓冲区，但排除已移除的缓冲区。基本上，在对该内核进行调用之后，哪些缓冲区实际上包含
    # 更新的数据？模仿 python_argdefs.
    def live_output_buffers(self):
        # 创建一个空集合，用于存储活跃的输出变量名
        live_outs = set()
        
        # 遍历所有独特的 inplace_buffers 值
        for inplaced in unique(self.inplace_buffers.values()):
            # 如果当前变量标记为已移除，则跳过
            if self._buffer_is_marked_removed(inplaced):
                continue
            # 将该变量的最后一个其他名称添加到 live_outs 集合中
            live_outs.add(inplaced.other_names[-1])
        
        # 遍历所有的 output_buffers 字典键值对
        for outer, inner in self.output_buffers.items():
            # 如果外部键在 inplace_buffers 中或内部值标记为已移除，则跳过
            if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
                continue
            # 将外部键添加到 live_outs 集合中
            live_outs.add(outer)
        
        # 返回存储活跃输出变量名的集合
        return live_outs
class CSEVariable:
    """A CSEVariable is just a name for an expression but it is useful to be able to annotate them on a backend dependent basis.
    To do so, the backends can simply overload `Kernel.create_cse_var`
    The "CSEVariable.update_on_args" method gives you a hook for annotations
    See example of TritonCSEVariable in triton.py
    """

    def __init__(self, name, bounds: ValueRanges[Any]):
        # 确保 bounds 是 ValueRanges 类的实例
        assert isinstance(bounds, ValueRanges)
        # 设置变量名和取值范围
        self.name = name
        self.bounds = bounds
        self.use_count = 1  # 记录表达式被使用的次数

    def __str__(self):
        # 返回变量名的字符串表示
        return self.name

    def __hash__(self) -> int:
        # 返回变量名的哈希值
        return hash(self.name)

    def __eq__(self, other) -> bool:
        # 比较两个 CSEVariable 对象是否相等
        return type(other) == type(self) and other.name == self.name

    def update_on_args(self, name, args, kwargs):
        # 在参数更新时提供一个钩子函数的占位
        pass

    def __repr__(self):
        # 返回 CSEVariable 对象的字符串表示
        return f"{self.__class__.__name__}({self.name!r})"


class CppWrapperKernelArgs(KernelArgs):
    def wrap_ptr_arg(self, buf, dtype):
        from .cpp_utils import DTYPE_TO_CPP

        if config.abi_compatible:
            # 在 abi_compatible 模式下，直接返回 buf
            # 后续在 wrapper.generate_kernel_all 中会形成正确的调用参数
            return buf
        else:
            # 返回格式化后的指针参数字符串
            return f"({DTYPE_TO_CPP[dtype]}*)({buf}.data_ptr())"

    def wrap_size_arg(self, size):
        # 返回格式化后的尺寸参数字符串
        return f"{size}"


class CSE:
    """Common subexpression elimination"""

    def __init__(
        self,
        prefix="",
        suffix="",
        name_prefix="tmp",
        iter_buffers=None,
        store_cache=None,
        reduction_cache=None,
        varname_map=None,
    ):
        # 初始化 CSE 对象的各个属性
        self.prefix = prefix
        self.suffix = suffix
        self.cache = {}
        self.name_prefix = name_prefix
        self.store_cache = store_cache or {}
        self.reduction_cache = reduction_cache or {}
        self.iter_buffer_ids = iter_buffers or itertools.count()
        self.invalidated_stores = set()
        self.varname_map = varname_map or {}

    def invalidate(self, keep_vars: Set[str]):
        # 清除无效的缓存和存储
        for name, tmp in list(self.store_cache.items()):
            if tmp not in keep_vars:
                del self.store_cache[name]
                self.invalidated_stores.add(name)
        # 清除无效的缓存项
        self.cache = {k: v for k, v in self.cache.items() if v in keep_vars}

    def clone(self):
        # 克隆一个 CSE 对象，注意 reduction_cache 没有被克隆，这可能是有意的
        return CSE(
            prefix=self.prefix,
            suffix=self.suffix,
            name_prefix=self.name_prefix,
            iter_buffers=self.iter_buffer_ids,
            store_cache=self.store_cache,
            varname_map=self.varname_map,
        )
    # 定义一个生成函数，用于生成代码片段并写入到指定的缓冲区
    def generate(
        self,
        buffer: IndentedBuffer,  # 缓冲区对象，用于存储生成的代码
        expr: Union[str, CSEVariable, OpsValue, IndentedBuffer],  # 表达式，可以是字符串、CSE变量、操作值对象或缩进缓冲区
        *,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),  # 可选参数，表达式的值范围，默认为未知
        write=True,  # 是否执行写入操作，默认为True
        assignment=True,  # 是否生成赋值语句，默认为True
    ) -> CSEVariable:
        if isinstance(expr, OpsValue):
            expr = expr.value  # 如果表达式是OpsValue对象，则取其值

        assert isinstance(expr, (str, CSEVariable, IndentedBuffer)), type(expr)  # 断言表达式的类型为字符串、CSE变量或缩进缓冲区

        assert write or assignment  # 断言write或assignment至少有一个为True

        if isinstance(expr, CSEVariable):
            # 如果表达式是CSE变量
            # 如果表达式的值范围与给定的bounds不同，调用tighten方法使其范围更加严格
            expr.bounds = expr.bounds.tighten(bounds)
            expr.use_count += 1  # 增加表达式的使用计数
            return expr  # 返回表达式本身作为结果

        cache_key = expr.getvalue() if isinstance(expr, IndentedBuffer) else expr
        var = self.cache.get(cache_key, None)  # 从缓存中获取与cache_key对应的变量

        if not var:
            var = self.newvar(bounds)  # 如果变量不存在于缓存中，则创建一个新的变量
            self.cache[cache_key] = var  # 将新创建的变量存入缓存
            if write:
                if V.kernel.current_node:
                    # 如果当前节点存在，调用codegen_originating_info方法将代码生成信息传递给缓冲区
                    V.kernel.current_node.codegen_originating_info(
                        buffer, only_once=True
                    )
                if isinstance(expr, IndentedBuffer):
                    if assignment:
                        buffer.writeline(f"{self.prefix}{var} =")  # 在缓冲区中写入赋值语句的前缀部分
                    buffer.splice(expr)  # 将缩进缓冲区的内容插入到当前缓冲区中
                    buffer.writeline(self.suffix)  # 在缓冲区中写入赋值语句的后缀部分
                else:
                    if assignment:
                        line = f"{self.prefix}{var} = {expr}{self.suffix}"  # 构造赋值语句
                    else:
                        line = f"{expr}{self.suffix}"  # 直接生成表达式语句
                    buffer.writeline(line)  # 在缓冲区中写入生成的代码行
        else:
            var.bounds = var.bounds.tighten(bounds)  # 如果变量已存在，则调用tighten方法更新其值范围
            var.use_count += 1  # 增加变量的使用计数

        return var  # 返回生成或获取的变量作为结果

    # 创建一个新的变量，并将其添加到内部变量映射表中
    def newvar(self, bounds: ValueRanges[Any] = ValueRanges.unknown()) -> CSEVariable:
        var_name = f"{self.name_prefix}{next(self.iter_buffer_ids)}"  # 生成新变量的名称
        var = V.kernel.create_cse_var(var_name, bounds)  # 调用内核方法创建一个新的CSE变量
        self.varname_map[var_name] = var  # 将变量名和变量对象映射存入varname_map中
        return var  # 返回新创建的变量作为结果
    @contextlib.contextmanager
    def set_current_node(self, node):
        # 保存当前节点，并获取其边界信息作为节点到值范围的字典
        prior = self.current_node
        self.current_node = node
        self.node_to_bounds = node._body.bounds().get_bounds()
        try:
            yield
        finally:
            # 恢复之前的节点
            self.current_node = prior
    # 定义一个方法用于交换缓冲区
    def swap_buffers(self, lb, cb=None, sb=None):
        # 定义一个局部函数，用于复制当前状态
        def scope_cse(cse):
            # 克隆传入的参数并创建新的对象，包含缓存、减少缓存和存储缓存
            new_cse = cse.clone()
            new_cse.cache = ScopedDict(cse.cache)
            new_cse.reduction_cache = ScopedDict(cse.reduction_cache)
            new_cse.store_cache = ScopedDict(cse.store_cache)
            return new_cse

        # 如果 cb 为空，则使用 lb
        if cb is None:
            cb = lb
        # 备份当前对象的加载、计算、存储和公共子表达式消除状态
        loads = self.loads
        compute = self.compute
        stores = self.stores
        cse = self.cse
        # 设置对象的加载、计算、存储和公共子表达式消除状态为指定值
        self.loads = lb
        self.compute = cb
        self.stores = sb
        self.cse = scope_cse(cse)
        try:
            # 返回生成器，允许在此处插入自定义代码
            yield
        finally:
            # 恢复加载、计算、存储和公共子表达式消除状态到先前备份的值
            self.loads = loads
            self.compute = compute
            self.stores = stores
            self.cse = cse

    # 加载指定名称和索引的符号表达式变量
    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        raise NotImplementedError

    # 间接加载指定名称和索引的符号表达式变量
    def indirect_load(self, name: str, index: sympy.Expr):
        """A load the depends on an index we have read"""
        # 备份当前加载状态
        prior = self.loads
        try:
            # 将加载状态设置为计算状态，因为它可能有依赖项
            self.loads = self.compute
            return self.load(name, index)
        finally:
            # 恢复先前的加载状态
            self.loads = prior

    # 存储指定名称、索引和值的归约结果
    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
        raise NotImplementedError

    # 存储指定名称、索引、值和存储模式的结果
    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        raise NotImplementedError

    # 执行指定数据类型、源数据类型和归约类型的归约操作
    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        raise NotImplementedError

    # 执行扫描操作，结合给定函数和值进行计算
    def scan(
        self,
        dtypes: Tuple[torch.dtype, ...],
        combine_fn: Callable[
            [Tuple[CSEVariable, ...], Tuple[CSEVariable, ...]], Tuple[CSEVariable, ...]
        ],
        values: Tuple[CSEVariable, ...],
    ) -> Tuple[CSEVariable, ...]:
        raise NotImplementedError

    # 执行排序操作，根据指定数据类型、值、稳定性和降序标志
    def sort(
        self,
        dtypes: Tuple[torch.dtype, ...],
        values: Tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> Tuple[CSEVariable, ...]:
        raise NotImplementedError

    # 返回变量的范围
    def var_ranges(self):
        raise NotImplementedError

    # 执行分桶操作，根据给定值、偏移名称、偏移量大小、索引数据类型和右侧标志
    def bucketize(
        self,
        values: CSEVariable,
        offsets_name: str,
        offsets_size: sympy.Expr,
        indexing_dtype: torch.dtype,
        right: bool,
    ) -> CSEVariable:
        """
        See [Note: Inductor bucketize op]
        """
        raise NotImplementedError

    # 返回断言函数的字符串表示
    @property
    def assert_function(self) -> str:
        raise NotImplementedError

    # 执行间接断言，使用变量、上限、下限和可选掩码
    def indirect_assert(
        self,
        var: Union[CSEVariable, str],
        lower: Optional[str],
        upper: Optional[str],
        mask: Optional[str] = None,
    ) -> str:
        # 如果变量是CSEVariable类型，则将其转换为字符串表示
        if isinstance(var, CSEVariable):
            var = str(var)
        # 断言变量是字符串类型
        assert isinstance(var, str)
        # 断言lower参数为None或者字符串类型
        assert lower is None or isinstance(lower, str)
        # 断言upper参数为None或者字符串类型
        assert upper is None or isinstance(upper, str)
        # 如果lower和upper都存在
        if lower and upper:
            # 由于Python运算符优先级，需要用括号括起来
            # 在triton中使用and/or/not操作符更不容易出错
            cond = f"({lower} <= {var}) & ({var} < {upper})"
            # 打印的条件字符串
            cond_print = f"{lower} <= {var} < {upper}"
        elif lower:
            # 如果只有lower存在
            cond = f"{lower} <= {var}"
            cond_print = cond
        else:
            # 否则，只有upper存在
            assert upper
            cond = f"{var} < {upper}"
            cond_print = cond

        # 如果存在mask参数，则将条件cond加上掩码操作
        if mask:
            cond = f"({cond}) | ~({mask})"

        # 返回断言的字符串表示，用于输出错误信息
        return f'{self.assert_function}({cond}, "index out of bounds: {cond_print}")'

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ):
        # 抛出未实现错误，该方法未被实现
        raise NotImplementedError

    def index_to_str(self, index: sympy.Expr) -> str:
        # 抛出未实现错误，该方法未被实现
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        注意，在生成triton模板内核时，V.graph.scheduler可能为None。
        """
        # 如果V.graph.scheduler不为None，则移除本地缓冲区的内核
        if V.graph.scheduler:
            V.graph.scheduler.remove_kernel_local_buffers()
        # 调用父类的__exit__方法
        super().__exit__(exc_type, exc_val, exc_tb)

    def rename_indexing(self, index) -> sympy.Expr:
        # 为索引表达式添加必要的内核参数，并将索引表达式中的变量重命名为内核参数名
        if isinstance(index, (list, tuple)):
            # 如果索引是列表或元组，则递归处理每个元素
            return [self.rename_indexing(x) for x in index]  # type: ignore[return-value]
        # 简化索引表达式中的V.graph.sizevars
        index = V.graph.sizevars.simplify(index)
        # 按名称排序索引表达式中的自由符号
        sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
        # 准备替换字典，用于将符号替换为对应的内核参数大小
        replacements = {
            x: self.args.size(x)
            for x in sorted_symbols
            if symbol_is_type(
                x,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                ),
            )
        }
        # 使用替换字典替换索引表达式中的符号
        return sympy_subs(index, replacements)

    def create_cse_var(self, *args, **kwargs):
        # 创建并返回一个CSEVariable对象
        return CSEVariable(*args, **kwargs)
@dataclasses.dataclass
class OptimizationContext:
    key: ClassVar[str] = "opt_ctx"  # 类属性：用于标识优化上下文的键名

    dtype: Optional[torch.dtype] = None  # 可选的数据类型，通常用于指定张量的数据类型
    ops_name: str = ""  # 操作名称，用于描述优化上下文中的操作的名称


@functools.lru_cache(None)
def jinja2_env():
    try:
        import jinja2

        # 尝试导入 Jinja2 库并返回一个 Jinja2 环境对象，如果导入失败则返回 None
        return jinja2.Environment(
            undefined=jinja2.StrictUndefined,  # 使用严格未定义模式
        )
    except ImportError:
        return None  # 如果导入失败则返回 None


class KernelTemplate:
    """
    Base class for defining kernel templates.

    Children classes: TritonTemplate, CUDATemplate
    """

    @staticmethod
    def indent_except_first(source: str, num_indents: int, indents_spacing=4):
        # 在除第一行之外的每一行前添加指定数量的缩进空格
        lines = source.splitlines(True)
        if len(lines) > 1:
            lines[1:] = [
                (" " * indents_spacing * num_indents) + line for line in lines[1:]
            ]
        return "".join(lines)

    @staticmethod
    def _template_from_string(source):
        # 从字符串创建模板对象
        env = jinja2_env()
        if env is not None:
            env.filters["indent_except_first"] = KernelTemplate.indent_except_first
            return env.from_string(source)  # 使用 Jinja2 环境从字符串创建模板对象
        return None  # 如果导入 Jinja2 失败则返回 None

    @staticmethod
    def _fake_get_dtype(fake_out):
        _get_dtype_real = V.graph.get_dtype  # 获取真实的数据类型获取函数

        def get_dtype(name):
            if name == fake_out.get_name():  # 如果名称与 fake_out 对象的名称匹配
                return fake_out.get_dtype()  # 返回 fake_out 对象的数据类型
            return _get_dtype_real(name)  # 否则调用真实的数据类型获取函数获取数据类型

        return get_dtype

    def __init__(self, name: str):
        self.name = name  # 初始化模板名称

    def maybe_append_choice(self, choices, **kwargs):
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """
        try:
            choices.append(self.generate(**kwargs))  # 尝试生成一个新的 ChoiceCaller 并将其添加到 choices 列表中
        except NotImplementedError:
            pass  # 如果生成未实现异常，则忽略

    def generate(self, **kwargs) -> "torch._inductor.ir.ChoiceCaller":
        """
        Generates a ChoiceCaller instance from the given arguments.
        """
        raise NotImplementedError  # 生成一个 ChoiceCaller 实例，由子类实现具体逻辑
```