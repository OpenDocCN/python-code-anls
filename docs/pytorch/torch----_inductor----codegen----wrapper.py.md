# `.\pytorch\torch\_inductor\codegen\wrapper.py`

```
# mypy: allow-untyped-defs
# 引入未类型化的函数定义允许声明
from __future__ import annotations

# 导入必要的标准库模块
import collections
import contextlib
import dataclasses
import dis
import functools
import inspect
import logging
import operator
import re

# 导入临时文件模块
import tempfile
# 导入 itertools 中的 count 函数
from itertools import count
# 导入类型提示相关模块
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

# 导入 sympy 符号计算库
import sympy
# 从 sympy 中导入 Expr 类型
from sympy import Expr

# 导入 PyTorch 库及其子模块
import torch
import torch._ops
# 导入 torch 的 dtype 类型定义
from torch import dtype as torch_dtype
# 导入 torch._dynamo.utils 中的 counters 和 dynamo_timed 函数
from torch._dynamo.utils import counters, dynamo_timed
# 导入 torch._inductor.codegen.multi_kernel 中的 MultiKernelState 类
from torch._inductor.codegen.multi_kernel import MultiKernelState
# 导入 torch._inductor.runtime.runtime_utils 中的 cache_dir 函数
from torch._inductor.runtime.runtime_utils import cache_dir
# 导入 torch.fx.experimental.symbolic_shapes 中的 ConvertIntKey, DivideByKey, SymTypes 类
from torch.fx.experimental.symbolic_shapes import ConvertIntKey, DivideByKey, SymTypes
# 导入 torch.fx.node 中的 _get_qualified_name 函数
from torch.fx.node import _get_qualified_name
# 导入 torch.utils._sympy.singleton_int 中的 SingletonInt 类
from torch.utils._sympy.singleton_int import SingletonInt
# 导入 torch.utils._sympy.symbol 中的 symbol_is_type, SymT 函数和类
from torch.utils._sympy.symbol import symbol_is_type, SymT

# 导入上级目录中的模块
from .. import async_compile, config, ir

# 从上级目录中的 codecache 模块导入 output_code_log 函数
from ..codecache import output_code_log
# 从上级目录中的 ir 模块导入 ReinterpretView 类
from ..ir import ReinterpretView
# 从上级目录中的 runtime 模块导入 triton_heuristics 模块
from ..runtime import triton_heuristics
# 从上级目录中的 runtime.hints 模块导入 DeviceProperties 类
from ..runtime.hints import DeviceProperties
# 从上级目录中的 utils 模块导入 cache_on_self, get_benchmark_name, LineContext, sympy_product, sympy_str 函数
from ..utils import (
    cache_on_self,
    get_benchmark_name,
    LineContext,
    sympy_product,
    sympy_str,
)
# 从上级目录中的 virtualized 模块导入 V 类
from ..virtualized import V
# 从当前目录中的 aoti_hipify_utils 模块导入 maybe_hipify_code_wrapper 函数
from .aoti_hipify_utils import maybe_hipify_code_wrapper
# 从当前目录中的 common 模块导入 CodeGen, DeferredLine, IndentedBuffer, PythonPrinter 类
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
# 从当前目录中的 triton_utils 模块导入 config_of, signature_to_meta 函数
from .triton_utils import config_of, signature_to_meta

# 如果是类型检查阶段，导入 triton 和 GraphLowering 类
if TYPE_CHECKING:
    import triton
    from ..graph import GraphLowering

# 定义一个函数 pexpr，用于将 Python 对象打印为字符串
pexpr = PythonPrinter().doprint

# 定义一个类型别名 ReuseKey，表示缓冲区重用的键
ReuseKey = Tuple[torch.device, torch.dtype, str]


# 定义一个函数 buffer_reuse_key，用于生成缓冲区重用的键
def buffer_reuse_key(node: ir.Buffer) -> ReuseKey:
    return (
        node.get_device(),  # 获取节点的设备信息
        node.get_dtype(),   # 获取节点的数据类型信息
        # 生成节点布局的存储大小的符号表示，以避免因为大小提示相同而尝试重用缓冲区
        sympy_str(V.graph.sizevars.simplify(node.layout.storage_size())),
    )


# 定义一个函数 convert_arg_type，用于将 Torch 参数类型转换为对应的 C++ 类型字符串
def convert_arg_type(arg: torch.Argument) -> str:
    from .cpp import CONTAINER_PYTHON_TO_CPP, PYTHON_TO_CPP

    # 使用 arg.real_type 替代 arg.type，以获取 ScalarType 而不是 int 类型
    python_type = repr(arg.real_type)  # type: ignore[attr-defined]

    if python_type == "Tensor":
        # 根据条件转换为 Tensor 的 C++ 引用类型
        if arg.alias_info is not None and arg.alias_info.is_write:
            return f"at::{python_type}&"
        else:
            return f"at::{python_type} const&"

    if python_type in PYTHON_TO_CPP:
        # 如果在映射表中找到对应的 Python 类型，则返回对应的 C++ 类型
        cpp_type = PYTHON_TO_CPP[python_type]
        return cpp_type

    # 对于容器类型的参数（如 Optional[*]），返回其字符串表示
    # （该部分代码未完整，已省略）
    # 遍历PYTHON到C++容器映射字典中的每一对键值对
    for py_container, cpp_container in CONTAINER_PYTHON_TO_CPP.items():
        # 在python_type中查找与当前py_container匹配的子容器类型
        container_match = re.findall(py_container + r"\[([a-zA-Z_]+)]", python_type)
        # 如果找到了一个匹配的子容器类型
        if len(container_match) == 1:
            # 获取匹配到的子容器类型
            contained_type = container_match[0]
            # 检查子容器类型是否在PYTHON到C++类型映射中
            assert (
                contained_type in PYTHON_TO_CPP
            ), f"unsupported {py_container} type in convert_arg_type: {contained_type}"
            # 获取对应的C++类型
            cpp_contained_type = PYTHON_TO_CPP[contained_type]
            # 返回包含C++模板参数的字符串表示形式
            return f"{cpp_container}<{cpp_contained_type}>"

    # 如果未找到匹配的容器类型，抛出异常
    raise AssertionError(f"unsupport python_type: {python_type}")
# 将返回类型从 torch.Argument 转换为对应的 C++ 类型字符串
def convert_return_type(ret: torch.Argument) -> str:
    # 使用 ret.real_type 而不是 ret.type，以获取 ScalarType 而不是 int
    python_type = repr(ret.real_type)  # type: ignore[attr-defined]
    # Python 到 C++ 类型的映射字典
    python_to_cpp = {
        "Tensor": "at::Tensor",
        "List[Tensor]": "std::vector<at::Tensor>",
    }

    # 根据 Python 类型获取对应的 C++ 类型字符串
    cpp_type = python_to_cpp.get(python_type, None)
    # 断言确保找到对应的 C++ 类型
    assert cpp_type is not None, f"NYI return type: {python_type}"

    # 当返回类型是 Tensor 并且具有 alias_info 时，返回类型是引用
    if python_type == "Tensor" and ret.alias_info is not None:
        cpp_type += "&"
    
    # 返回最终的 C++ 返回类型字符串
    return cpp_type


# 根据给定的 torch._ops.OpOverload 获取 C++ 操作的签名字符串
def get_cpp_op_schema(kernel: torch._ops.OpOverload) -> str:
    # 获取操作的参数列表和返回值列表
    args = kernel._schema.arguments
    returns = kernel._schema.returns

    # 确保至少有一个返回值
    num_returns = len(returns)
    assert num_returns > 0, "must have at least one return value"

    # 根据返回值的数量确定返回类型字符串
    if num_returns == 1:
        cpp_return_value = convert_return_type(returns[0])
    elif num_returns > 1:
        tuple_returns = ", ".join([convert_return_type(r) for r in returns])
        cpp_return_value = f"std::tuple<{tuple_returns}>"

    # 构建 C++ 函数签名的参数部分
    cpp_arg_type = [f"{convert_arg_type(arg)} {arg.name}" for arg in args]
    # 返回完整的 C++ 函数签名字符串
    return f"{cpp_return_value}({', '.join(cpp_arg_type)})"  # type: ignore[possibly-undefined]


# TODO: Move to a well known place
# TritonMetaParams 是一个字典，键为字符串，值为整数
# TritonGrid 是一个元组或者一个函数，元组中的元素可以是整数或 sympy.Expr 对象
def user_defined_kernel_grid_fn_code(
    name: str,
    configs: List[triton.Config],
    grids: List[TritonGrid],
    wrapper: Optional[WrapperCodeGen] = None,
) -> Tuple[str, str]:
    # 创建一个缩进缓冲区对象
    output = IndentedBuffer()

    # 将整数转换为 sympy.Expr 对象的辅助函数
    def _convert_to_sympy_expr(item: Union[int, sympy.Expr]) -> sympy.Expr:
        return item if isinstance(item, sympy.Expr) else sympy.Integer(item)

    # 确定网格的实际使用值和在编译时自动调优时的示例网格
    def determine_grid(
        grid: TritonGrid,
    ):
        """
        This function return a tuple of two values: the first one is for the real grid
        which is used in the generated code; the second one is an example grid with
        concreate values which is used in the autotune block to run the generated
        kernels at compile time.
        """
        if wrapper is None or callable(grid):
            # 在 eager 模式或者 grid 是 callable 时，直接返回原始 grid
            return grid, grid
        
        # 当 grid 包含 ints/Expr 时，利用 wrapper 的表达式打印功能进行代码生成
        sympy_grid = tuple(_convert_to_sympy_expr(g) for g in grid)
        return (
            wrapper.codegen_shape_tuple(sympy_grid),
            wrapper.codegen_shape_tuple(
                tuple(wrapper.generate_example_arg_value(g) for g in sympy_grid)
            )
            if config.triton.autotune_at_compile_time
            else None,
        )
    def writeline(line: str, example_grid: Optional[str] = None):
        # 调用 output 对象的 writeline 方法，输出一行文本
        output.writeline(line)
        # 如果存在 wrapper 对象并且配置中开启了编译时自动调整，则调用 wrapper 的 kernel_autotune_calls 的 writeline 方法，输出示例网格或者当前行
        if wrapper and config.triton.autotune_at_compile_time:
            wrapper.kernel_autotune_calls.writeline(example_grid or line)

    # 构造函数名，格式化字符串，创建函数名
    fn_name = f"grid_wrapper_for_{name}"
    # 调用 writeline 方法，输出定义函数的文本
    writeline(f"def {fn_name}(meta):")
    # 设置 kernel_autotune_calls 的缩进，如果存在 wrapper 对象并且配置中开启了编译时自动调整，则缩进，否则使用空上下文
    kernel_autotune_calls_indent = (
        wrapper.kernel_autotune_calls.indent()
        if wrapper and config.triton.autotune_at_compile_time
        else contextlib.nullcontext()
    )
    # 使用 output 对象进行缩进
    with output.indent(), kernel_autotune_calls_indent:
        # 如果 grids 列表长度为 1
        if len(grids) == 1:
            # 确定网格和示例网格
            grid, example_grid = determine_grid(grids[0])
            # 调用 writeline 方法，输出返回网格的文本，如果提供了示例网格，则同时输出示例网格的文本
            writeline(f"return {grid}", f"return {example_grid}")
        else:
            # 断言 grids 列表长度大于 1
            assert len(grids) > 1
            # 断言 grids 列表长度与 configs 列表长度相等
            assert len(grids) == len(configs)
            # 创建一个空集合 seen，用于存储已经处理过的条件语句
            seen = set()
            # 遍历 grids 和 configs 列表
            for grid, c in zip(grids, configs):
                # 构造守卫条件列表
                guards = [f"meta['{name}'] == {val}" for name, val in c.kwargs.items()]
                guards = " and ".join(guards)
                # 确定网格和示例网格
                grid, example_grid = determine_grid(grid)
                # 构造条件语句
                statement = f"if {guards}: return {grid}"
                # 如果条件语句已经在 seen 中出现过，则跳过当前循环
                if statement in seen:
                    continue
                # 将条件语句添加到 seen 中
                seen.add(statement)
                # 调用 writeline 方法，输出条件语句的文本，如果提供了示例网格，则同时输出示例网格的文本
                writeline(statement, f"if {guards}: return {example_grid}")

    # 返回函数名和 output 对象当前的所有输出内容
    return fn_name, output.getvalue()
# 使用 dataclass 装饰器定义 SymbolicCallArg 类，用于表示符号调用的参数
@dataclasses.dataclass
class SymbolicCallArg:
    inner: str
    # inner 表示的原始符号表达式
    inner_expr: sympy.Expr

    def __str__(self):
        return str(self.inner)


# 默认线程堆栈大小因平台而异：
# - Linux: 8 MB
# - macOS: 512 KB
# - Windows: 1 MB
# 在此选择比最小值小得多的值作为最大堆栈分配大小
MAX_STACK_ALLOCATION_SIZE = 1024 * 100


class MemoryPlanningState:
    def __init__(self):
        super().__init__()
        # reuse_pool 是一个字典，key 是 ReuseKey 类型，value 是 FreeIfNotReusedLine 类型的列表
        self.reuse_pool: Dict[
            ReuseKey, List[FreeIfNotReusedLine]
        ] = collections.defaultdict(list)
        # total_allocated_buffer_size 记录总分配的缓冲区大小
        self.total_allocated_buffer_size: int = 0

    def __contains__(self, key: ReuseKey) -> bool:
        # 检查 reuse_pool 中是否存在指定的 key
        return bool(self.reuse_pool.get(key, None))

    def pop(self, key: ReuseKey) -> FreeIfNotReusedLine:
        # 从 reuse_pool 中弹出指定 key 对应的项，要求项必须是 FreeIfNotReusedLine 类型
        item = self.reuse_pool[key].pop()
        assert not item.is_reused  # 断言该项未被重用
        return item

    def push(self, key: ReuseKey, item: FreeIfNotReusedLine) -> None:
        # 将未被重用的项 item 推入 reuse_pool 中的指定 key 对应的列表中
        assert not item.is_reused  # 断言该项未被重用
        self.reuse_pool[key].append(item)


class WrapperLine:
    pass


# 使用 dataclass 装饰器定义 EnterSubgraphLine 类，表示进入子图的包装行
@dataclasses.dataclass
class EnterSubgraphLine(WrapperLine):
    wrapper: WrapperCodeGen
    graph: GraphLowering

    def codegen(self, code: IndentedBuffer) -> None:
        # 将 codegened_graph 推入 wrapper 对象中
        self.wrapper.push_codegened_graph(self.graph)
        # 对代码进行缩进处理
        code.do_indent()


# 使用 dataclass 装饰器定义 ExitSubgraphLine 类，表示退出子图的包装行
@dataclasses.dataclass
class ExitSubgraphLine(WrapperLine):
    wrapper: WrapperCodeGen

    def codegen(self, code: IndentedBuffer) -> None:
        # 从 wrapper 对象中弹出 codegened_graph
        self.wrapper.pop_codegened_graph()
        # 对代码进行取消缩进处理
        code.do_unindent()


# 使用 dataclass 装饰器定义 EnterDeviceContextManagerLine 类，表示进入设备上下文管理器的包装行
@dataclasses.dataclass
class EnterDeviceContextManagerLine(WrapperLine):
    device_idx: int
    last_seen_device_guard_index: Optional[int]
    # 如果图形对象的cpp_wrapper属性为真，则执行以下代码块
    def codegen(self, code: IndentedBuffer) -> None:
        # 在代码缓冲区中写入一个空行
        if V.graph.cpp_wrapper:
            code.writeline("\n")
            # 如果处于AOT模式下
            if V.graph.aot_mode:
                # 在AOT模式下，有一个作为参数提供的流。流与设备关联，因此我们不希望设备改变。
                # CUDAStreamGuard设置流和设备。
                if self.last_seen_device_guard_index is None:
                    # 如果之前没有看到设备保护索引
                    if config.abi_compatible:
                        # 如果ABI兼容，创建AOTICudaStreamGuard对象
                        code.writeline(
                            "AOTICudaStreamGuard stream_guard(stream, this->device_idx_);"
                        )
                    else:
                        # 如果不是ABI兼容，可能需要进行HIP代码包装
                        code.writeline(
                            maybe_hipify_code_wrapper(
                                "at::cuda::CUDAStreamGuard stream_guard("
                                + "at::cuda::getStreamFromExternal(stream, this->device_idx_));"
                            )
                        )
                else:
                    # 如果之前看到了设备保护索引，则确保索引和设备索引相同
                    assert (
                        self.last_seen_device_guard_index == self.device_idx
                    ), "AOTInductor only supports running on one CUDA device"
            else:
                # 如果不是AOT模式
                if self.last_seen_device_guard_index is None:
                    # 如果之前没有看到设备保护索引
                    code.writeline(
                        f"AOTICudaGuard device_guard({self.device_idx});"
                        if config.abi_compatible
                        else maybe_hipify_code_wrapper(
                            f"at::cuda::CUDAGuard device_guard({self.device_idx});"
                        )
                    )
                else:
                    # 如果之前看到了设备保护索引，则设置设备索引
                    code.writeline(f"device_guard.set_index({self.device_idx});")
        else:
            # 如果图形对象的cpp_wrapper属性为假，则执行以下代码块
            # _DeviceGuard比device有更少的开销，但只接受整数作为参数
            code.writeline(f"with {V.graph.device_ops.device_guard(self.device_idx)}:")
            code.do_indent()
            code.writeline(V.graph.device_ops.set_device(self.device_idx))
# ExitDeviceContextManagerLine 类，继承自 WrapperLine 类
class ExitDeviceContextManagerLine(WrapperLine):
    
    # 重写 codegen 方法，用于生成代码
    def codegen(self, code: IndentedBuffer) -> None:
        # 如果 V.graph.cpp_wrapper 为假，则执行减少缩进操作
        if not V.graph.cpp_wrapper:
            code.do_unindent()


# MemoryPlanningLine 数据类，继承自 WrapperLine 类
@dataclasses.dataclass
class MemoryPlanningLine(WrapperLine):
    # 包装器实例
    wrapper: WrapperCodeGen

    # plan 方法，用于内存规划，返回 MemoryPlanningLine 实例
    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        """First pass to find reuse"""
        return self

    # codegen 方法，用于生成代码，在第二次遍历中输出代码
    def codegen(self, code: IndentedBuffer) -> None:
        """Second pass to output code"""
        pass

    # __str__ 方法，返回适合一行显示的字符串表示形式
    def __str__(self) -> str:
        """
        Emits a string representation that fits on one line.
        """
        args: List[str] = []
        # 遍历数据类的字段
        for field in dataclasses.fields(self):
            # 如果字段名为 "wrapper"，则跳过
            if field.name == "wrapper":
                continue
            # 获取字段值
            val = getattr(self, field.name)
            # 根据字段类型选择性输出字段值
            args.append(
                f"{field.name}={val.get_name() if field.type is ir.Buffer else val}"
            )
        # 返回数据类的类型名和参数的字符串表示形式
        return f"{type(self).__name__}({', '.join(args)})"


# AllocateLine 数据类，继承自 MemoryPlanningLine 类
@dataclasses.dataclass
class AllocateLine(MemoryPlanningLine):
    # 缓冲区节点
    node: ir.Buffer

    # plan 方法，用于内存规划，返回 MemoryPlanningLine 实例
    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        # 如果缓冲区节点的名称在 V.graph.removed_buffers 中，返回 NullLine 对象
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        # 尝试重用最近释放的缓冲区
        key = buffer_reuse_key(self.node)
        if config.allow_buffer_reuse and key in state:
            free_line = state.pop(key)
            free_line.is_reused = True
            # 返回 ReuseLine 对象，表示重用缓冲区
            return ReuseLine(self.wrapper, free_line.node, self.node)

        # 如果缓冲区节点的设备类型为 "cpu"
        if self.node.get_device().type == "cpu":
            # 获取缓冲区的静态形状
            static_shape = self.wrapper.static_shape_for_buffer_or_none(self.node)
            if static_shape is not None:
                # 计算静态形状的总大小，并添加到 state 的总分配缓冲区大小中
                state.total_allocated_buffer_size += int(
                    functools.reduce(operator.mul, static_shape, 1)
                )

        # 返回当前实例，表示没有执行缓冲区的重用
        return self

    # codegen 方法，用于生成代码
    def codegen(self, code: IndentedBuffer) -> None:
        # 断言缓冲区节点的名称不在 V.graph.removed_buffers 中
        assert self.node.get_name() not in V.graph.removed_buffers
        # 生成缓冲区分配的代码行，并添加到 IndentedBuffer 对象中
        line = self.wrapper.make_buffer_allocation(self.node)
        code.writeline(line)


# FreeIfNotReusedLine 数据类，继承自 MemoryPlanningLine 类
@dataclasses.dataclass
class FreeIfNotReusedLine(MemoryPlanningLine):
    # 缓冲区节点
    node: ir.Buffer
    # 是否被重用的标志
    is_reused: bool = False

    # plan 方法，用于内存规划，返回 MemoryPlanningLine 实例
    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        # 如果存在别名输出的输入，则返回当前实例
        if len(self.node.get_inputs_that_alias_output()) > 0:
            return self
        # 如果节点布局是多输出布局，则返回当前实例
        if isinstance(self.node.layout, ir.MultiOutputLayout):
            return self
        # 断言没有被重用
        assert not self.is_reused
        # 如果缓冲区节点的名称在 V.graph.removed_buffers 中，返回 NullLine 对象
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
        # 如果允许缓冲区重用，则将当前实例添加到 state 中，并返回当前实例
        if config.allow_buffer_reuse:
            state.push(buffer_reuse_key(self.node), self)
        return self

    # codegen 方法，用于生成代码
    def codegen(self, code: IndentedBuffer) -> None:
        # 断言缓冲区节点的名称不在 V.graph.removed_buffers 中
        assert self.node.get_name() not in V.graph.removed_buffers
        # 如果没有被重用，则生成释放缓冲区的代码行，并添加到 IndentedBuffer 对象中
        if not self.is_reused:
            code.writeline(self.wrapper.make_buffer_free(self.node))


# ReuseLine 数据类，继承自 MemoryPlanningLine 类
@dataclasses.dataclass
class ReuseLine(MemoryPlanningLine):
    # 包装器实例
    wrapper: WrapperCodeGen
    # 被重用的缓冲区节点
    from_node: ir.Buffer
    # 重用的目标缓冲区节点
    to_node: ir.Buffer

    # 由于没有实现额外的方法或属性，这里不需要额外的注释
    pass
    # 表示类成员变量 `node`，类型为 `ir.Buffer`
    node: ir.Buffer
    # 表示类成员变量 `reused_as`，类型为 `ir.Buffer`
    reused_as: ir.Buffer
    # 布尔类型变量 `delete_old`，默认为 True

    # 定义方法 `plan`，接受 `state` 参数，返回 `MemoryPlanningLine` 对象
    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        # 如果 `node` 的名称在全局变量 `V.graph.removed_buffers` 中
        if self.node.get_name() in V.graph.removed_buffers:
            # 断言 `reused_as` 的名称也在 `V.graph.removed_buffers` 中
            assert self.reused_as.get_name() in V.graph.removed_buffers
            # 如果条件满足，返回一个空行对象 `NullLine`
            return NullLine(self.wrapper)
        # 如果 `reused_as` 的名称不在 `V.graph.removed_buffers` 中
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        # 返回当前对象 `self`
        return self

    # 定义方法 `codegen`，接受 `code` 参数，无返回值
    def codegen(self, code: IndentedBuffer) -> None:
        # 断言 `node` 的名称不在 `V.graph.removed_buffers` 中
        assert self.node.get_name() not in V.graph.removed_buffers
        # 断言 `reused_as` 的名称不在 `V.graph.removed_buffers` 中
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        # 在代码缓冲区中写入一行代码，调用 `wrapper` 对象的 `make_buffer_reuse` 方法，
        # 参数为 `self.node`、`self.reused_as` 和 `self.delete_old`
        code.writeline(
            self.wrapper.make_buffer_reuse(self.node, self.reused_as, self.delete_old)
        )
class NullLine(MemoryPlanningLine):
    pass


# 创建一个名为 NullLine 的类，继承自 MemoryPlanningLine 类
class NullLine(MemoryPlanningLine):
    # pass 表示该类目前没有额外的实现或属性
    pass



BufferName = str


# 定义一个别名 BufferName，表示它是一个字符串类型
BufferName = str



class WrapperCodeGen(CodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """


# 创建一个名为 WrapperCodeGen 的类，继承自 CodeGen 类
class WrapperCodeGen(CodeGen):
    """
    生成调用内核的 Python 外部包装器。
    """
    # 这是一个类的文档字符串，描述了该类的作用
    # 初始化函数，继承父类的初始化方法
    def __init__(self):
        super().__init__()
        # 生成唯一递增整数的迭代器，用于生成唯一的名称
        self._names_iter: Iterator[int] = count()
        # 以下是用于存储生成代码的缓冲区对象
        self.header = IndentedBuffer()  # 存储代码头部内容的缓冲区
        self.prefix = IndentedBuffer()  # 存储代码前缀内容的缓冲区
        self.suffix = IndentedBuffer()  # 存储代码后缀内容的缓冲区
        self.wrapper_call = IndentedBuffer()  # 存储代码包装调用内容的缓冲区
        self.kernel_autotune_defs = IndentedBuffer()  # 存储内核自动调优定义内容的缓冲区
        self.kernel_autotune_calls = IndentedBuffer()  # 存储内核自动调优调用内容的缓冲区
        self.kernel_autun_names: Set[str] = set()  # 存储内核自动调优名称的集合
        # 如果生成的源代码完全相同，则重用预存在的内核
        self.src_to_kernel: Dict[str, str] = {}  # 存储源代码到内核名称的映射字典
        self.kernel_numel_expr: Set[Tuple[str, GraphLowering]] = set()  # 存储内核元素数量表达式的集合
        self.lines: List[Union[MemoryPlanningLine, LineContext]] = []  # 存储行对象的列表
        self.declare = ""  # 存储声明内容的字符串
        self.declare_maybe_reference = ""  # 存储可能是引用的声明内容的字符串
        self.ending = ""  # 存储结尾内容的字符串
        self.open_bracket = "["  # 存储开放括号的字符串
        self.closed_bracket = "]"  # 存储闭合括号的字符串
        self.comment = "#"  # 存储注释符号的字符串
        self.namespace = ""  # 存储命名空间的字符串
        self.none_str = "None"  # 存储表示空值的字符串
        self.size = "size()"  # 存储表示尺寸的字符串
        self.stride = "stride()"  # 存储表示步幅的字符串
        self.last_seen_device_guard_index: Optional[int] = None  # 存储最后见到的设备保护索引的可选整数
        self.supports_intermediate_hooks = True  # 标记是否支持中间挂钩
        self.expr_printer: Callable[[Any], str] = pexpr  # 表达式打印器的函数类型
        self.user_defined_kernel_cache: Dict[Tuple[Any, ...], Tuple[str, Any]] = {}  # 用户定义内核缓存的字典
        self.unbacked_symbol_decls: Set[str] = set()  # 存储无后备符号声明的集合，每个元素是 sympy.Symbol 类型的字符串表示
        self.allow_stack_allocation: Optional[bool] = None  # 是否允许堆栈分配的可选布尔值
        self.stack_allocated_buffers: Dict[BufferName, ir.Buffer] = {}  # 存储堆栈分配的缓冲区字典
        self.computed_sizes: Set[sympy.Symbol] = set()  # 存储计算大小的 sympy.Symbol 类型的集合

        # 用于跟踪当前正在生成代码的 GraphLowering 实例，主要用途是将图实例包含到缓存键中，
        # 避免在降低嵌套子图时进行跨图缓存
        self.codegened_graph_stack = []

        # 写入初始代码头部
        self.write_header()
        # 写入初始代码前缀
        self.write_prefix()
        # 写入内核自动调优定义的初始头部
        self.write_kernel_autotune_defs_header()

        # 如果不是AOT模式，则遍历常量表示字典，将每个常量写入代码中
        if not V.graph.aot_mode:
            for name, hashed in V.graph.constant_reprs.items():
                # 将带有哈希值的常量写入代码，确保不同的常量被放入不同的文件中
                self.write_constant(name, hashed)

        self.allocated: Set[BufferName] = set()  # 存储已分配缓冲区的集合
        self.freed: Set[BufferName] = set()  # 存储已释放缓冲区的集合

        # 用于将重用缓冲区映射到被重用的缓冲区
        self.reuses: Dict[BufferName, BufferName] = dict()

        # 使用 functools.lru_cache 对 write_get_raw_stream 方法进行缓存，以提高性能
        self.write_get_raw_stream = functools.lru_cache(None)(
            self.write_get_raw_stream
        )

        # 使用 functools.lru_cache 对 add_import_once 函数进行缓存，以避免重复导入
        @functools.lru_cache(None)
        def add_import_once(line: str) -> None:
            # 将指定的行添加到头部缓冲区中
            self.header.writeline(line)
            # 如果配置允许在编译时进行自动调优，则将指定的行添加到内核自动调优调用缓冲区中
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(line)

        self.add_import_once = add_import_once  # 将 add_import_once 函数赋值给实例属性

        self._metas: Dict[str, str] = {}  # 存储元数据的字典
        self._meta_vars: Set[str] = set()  # 存储元数据变量名的集合
        self.multi_kernel_state = MultiKernelState()  # 多内核状态对象的初始化
    def write_constant(self, name: str, hashed: str) -> None:
        # 将常量写入头文件，格式为 {name} = None  # {hashed}
        self.header.writeline(f"{name} = None  # {hashed}")

    def write_header(self) -> None:
        # 尝试获取 Torch 的追踪上下文
        context = torch._guards.TracingContext.try_get()
        aot_config_comment = ""
        # 如果上下文存在并且有 AOT 图形名称，则设置注释
        if context is not None and context.aot_graph_name is not None:
            aot_config_comment = f"# AOT ID: {context.aot_graph_name}"
        # 在头文件中插入导入声明和可能的 AOT 配置注释
        self.header.splice(
            f"""
                {aot_config_comment}
                from ctypes import c_void_p, c_long
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align

                from torch import device, empty_strided
                from {async_compile.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels
                from torch._inductor.codegen.multi_kernel import MultiKernelCall

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                _quantized = torch.ops._quantized
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                async_compile = AsyncCompile()

            """
        )

    def write_kernel_autotune_defs_header(self) -> None:
        # 在内核自动调优定义头文件中插入导入声明和 AsyncCompile 实例化
        self.kernel_autotune_defs.splice(
            f"""
                import torch
                from torch._dynamo.testing import rand_strided
                from torch._dynamo.utils import preserve_rng_state
                from torch._inductor.select_algorithm import AlgorithmSelectorCache
                from {async_compile.__name__} import AsyncCompile

                async_compile = AsyncCompile()
                generate_example_value = AlgorithmSelectorCache.generate_example_value
            """
        )

    @cache_on_self
    def write_triton_header_once(self) -> None:
        # 插入 Triton 相关的导入声明和 Grid 相关函数导入
        import_str = f"""
            import triton
            import triton.language as tl
            from {triton_heuristics.__name__} import grid, split_scan_grid, start_graph, end_graph
            """
        self.header.splice(import_str)
        # 如果配置要求在编译时自动调优，则在内核自动调优调用部分也插入相同的导入声明
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.splice(import_str)
        # 确保写入获取原始流的头文件部分
        self.write_get_raw_stream_header_once()

    @cache_on_self


这些注释详细解释了每个函数和代码块的作用，符合要求的格式和内容。
    # 写入一次获取原始流的头部信息
    def write_get_raw_stream_header_once(self) -> None:
        # 将获取原始流函数导入到头部信息中
        self.header.writeline(
            V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
        )
        # 如果在编译时进行自动调优
        if config.triton.autotune_at_compile_time:
            # 向核心自动调优调用中写入获取原始流函数的导入
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )

    # 添加元数据，仅添加一次
    def add_meta_once(self, meta: TritonMetaParams) -> str:
        meta = repr(meta)
        # 如果元数据不在已有元数据中
        if meta not in self._metas:
            var = f"meta{len(self._metas)}"
            self._metas[meta] = var
            # 向头部信息中写入变量和其对应的元数据
            self.header.writeline(f"{var} = {meta}")
            # 如果在编译时进行自动调优
            if config.triton.autotune_at_compile_time:
                # 向核心自动调优调用中写入变量和其对应的元数据
                self.kernel_autotune_calls.writeline(f"{var} = {meta}")
                self._meta_vars.add(var)
        return self._metas[meta]

    # 在自身缓存上添加缓存装饰器
    @cache_on_self
    def get_output_refs(self) -> List[str]:
        # 返回所有图输出的引用列表
        return [x.codegen_reference(self.wrapper_call) for x in V.graph.graph_outputs]

    # 标记输出类型，无操作
    def mark_output_type(self) -> None:
        return

    # 生成输入尺寸断言
    def codegen_input_size_asserts(self) -> None:
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
                continue

            # 对于尺寸为零的张量，跳过比较步长是棘手的，现在忽略它们
            if sympy_product(buf.get_size()) == 0:
                continue
            size = self.codegen_shape_tuple(buf.get_size())
            stride = self.codegen_shape_tuple(buf.get_stride())
            # 向前缀代码中写入尺寸和步长断言
            self.prefix.writeline(f"assert_size_stride({name}, {size}, {stride})")

    # 生成输入NaN断言
    def codegen_input_nan_asserts(self) -> None:
        # 向前缀代码中写入注释，确保图输入不是NaN/inf
        self.prefix.writeline("# make sure graph inputs are not nan/inf")
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
                continue

            line = f"assert not {name}.isnan().any().item()"
            # 向前缀代码中写入NaN断言
            self.prefix.writeline(line)
            line = f"assert not {name}.isinf().any().item()"
            # 向前缀代码中写入inf断言
            self.prefix.writeline(line)

    # 写入前缀代码
    def write_prefix(self) -> None:
        # 将代码块插入前缀
        self.prefix.splice(
            """

            async_compile.wait(globals())
            del async_compile

            def call(args):
            """
        )
        with self.prefix.indent():
            # 如果是调试同步图形
            if config.triton.debug_sync_graph:
                # 向前缀代码中写入设备操作的同步调用
                self.prefix.writeline(V.graph.device_ops.synchronize())
            # 如果图输入非空
            if V.graph.graph_inputs:
                lhs = ", ".join(V.graph.graph_input_names)
                if len(V.graph.graph_input_names) == 1:
                    lhs += ","
                # 向前缀代码中写入参数赋值
                self.prefix.writeline(f"{lhs} = args")
                self.prefix.writeline("args.clear()")

            # 生成图输入
            self.codegen_inputs(self.prefix, V.graph.graph_inputs)
            # 如果需要尺寸断言
            if config.size_asserts:
                self.codegen_input_size_asserts()
            # 如果需要NaN断言
            if config.nan_asserts:
                self.codegen_input_nan_asserts()
    # 定义一个方法，生成并返回一个原始流名称，用于设备索引和图形
    # 如果图形未提供，则默认为空
    def write_get_raw_stream(self, device_idx: int, graph=None) -> str:
        # 确保仅在每个图实例中进行流缓存，这对于嵌套子图的代码生成至关重要
        self.write_get_raw_stream_header_once()
        # 构建流名称，格式为"stream{device_idx}"
        name = f"stream{device_idx}"
        # 调用self.writeline方法，将获取原始流的调用写入代码
        self.writeline(f"{name} = get_raw_stream({device_idx})")
        # 返回生成的流名称
        return name

    # 获取当前已生成的图形对象
    def get_codegened_graph(self):
        return self.codegened_graph_stack[-1]

    # 将生成的图形对象压入堆栈
    def push_codegened_graph(self, graph):
        self.codegened_graph_stack.append(graph)

    # 弹出并返回堆栈顶部的生成图形对象
    def pop_codegened_graph(self):
        return self.codegened_graph_stack.pop()

    # 返回下一个内核后缀字符串，使用_names_iter的下一个元素
    def next_kernel_suffix(self) -> str:
        return f"{next(self._names_iter)}"

    # 生成进入设备保护的代码块，以确保代码在特定设备上运行
    def codegen_device_guard_enter(self, device_idx: int) -> None:
        # 将进入设备上下文管理器的代码行写入缓冲区
        self.writeline(
            EnterDeviceContextManagerLine(device_idx, self.last_seen_device_guard_index)
        )
        # 如果配置为在编译时自动调优，则生成相应的代码块
        if config.triton.autotune_at_compile_time:
            # 写入Trition头信息的代码只有一次
            self.write_triton_header_once()
            # 写入自动调优调用代码块的开始
            self.kernel_autotune_calls.writeline(
                f"with {V.graph.device_ops.device_guard(device_idx)}:"
            )
            self.kernel_autotune_calls.do_indent()
            # 设置当前设备并获取原始流
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.set_device(device_idx)
            )
            self.kernel_autotune_calls.writeline(
                f"stream{device_idx} = get_raw_stream({device_idx})"
            )
        # 更新最后一次见到的设备保护索引
        self.last_seen_device_guard_index = device_idx

    # 生成退出设备保护的代码行
    def codegen_device_guard_exit(self) -> None:
        # 将退出设备上下文管理器的代码行写入缓冲区
        self.writeline(ExitDeviceContextManagerLine())

    # 生成返回语句，返回指定的输出引用列表
    def generate_return(self, output_refs: List[str]) -> None:
        if output_refs:
            # 生成返回指定输出引用列表的代码行
            self.wrapper_call.writeline("return (" + ", ".join(output_refs) + ", )")
        else:
            # 生成返回空元组的代码行
            self.wrapper_call.writeline("return ()")

    # 在生成过程中生成前缀的代码，但此方法当前没有实现任何操作
    def generate_before_suffix(self, result: IndentedBuffer) -> None:
        return

    # 在生成过程结束时生成代码，但此方法当前没有实现任何操作
    def generate_end(self, result: IndentedBuffer) -> None:
        return

    # 生成回退内核的代码，用于生成外部内核分配
    def generate_fallback_kernel(self, fallback_kernel, args):
        self.generate_extern_kernel_alloc(fallback_kernel, args)
    # 生成外部内核分配代码，处理外部内核返回情况和相关信息
    def generate_extern_kernel_alloc(self, extern_kernel, args):
        # 如果 extern_kernel 的布局是 NoneLayout，则相当于它没有返回任何内容
        no_return = isinstance(extern_kernel.layout, ir.NoneLayout)
        # 获取外部内核的输出名称、原始节点和内核名称
        output_name = extern_kernel.get_name()
        origin_node = extern_kernel.get_origin_node()
        kernel_name = extern_kernel.get_kernel_name()
        ending = self.ending
        # 如果启用了内存规划并且内核名称中包含 "view_as_complex"，则添加 ".clone()" 以解决视图操作回退问题
        if config.memory_planning and "view_as_complex" in kernel_name:
            ending = f".clone(){ending}"

        # 根据是否有返回值来生成相应的调用代码
        if no_return:
            self.writeline(f"{self.declare}{kernel_name}({', '.join(args)}){ending}")
        else:
            self.writeline(
                f"{self.declare}{output_name} = {kernel_name}({', '.join(args)}){ending}"
            )
            # 如果支持中间挂钩，生成中间挂钩调用
            if (
                self.supports_intermediate_hooks
                and config.generate_intermediate_hooks
                and origin_node is not None
            ):
                counters["inductor"]["intermediate_hooks"] += 1
                self.writeline(
                    f"run_intermediate_hooks({origin_node.name!r}, {output_name})"
                )

    # 生成外部内核输出代码，包括输出参数设置
    def generate_extern_kernel_out(
        self, kernel: str, out: str, out_view: Optional[str], args: List[str]
    ):
        args.append(f"out={out_view if out_view else out}")
        self.writeline(f"{kernel}({', '.join(args)})")

    # 生成用户定义的 Triton 内核代码，包括网格函数和内核调用
    def generate_user_defined_triton_kernel(
        self, kernel_name, grid, configs, args, triton_meta, raw_args
    ):
        # 获取用户定义内核的网格函数和代码
        grid_fn, code = user_defined_kernel_grid_fn_code(
            kernel_name, configs, grid, wrapper=self
        )
        # 在调用之前生成网格包装函数的代码
        for line in code.split("\n"):
            self.writeline(line)

        # 获取原始参数的数据类型并生成内核调用
        arg_types = [
            arg.get_dtype() if hasattr(arg, "get_dtype") else type(arg)
            for arg in raw_args
        ]
        self.generate_kernel_call(
            kernel_name, args, grid_fn=grid_fn, arg_types=arg_types, raw_args=raw_args
        )

    # 生成 scatter 操作的回退代码
    def generate_scatter_fallback(
        self,
        output,
        inputs,
        cpp_kernel_name,
        python_kernel_name,
        src_is_tensor,
        reduce,
        kwargs,
    ):
        line = f"{python_kernel_name}({','.join(map(str, inputs))}"
        if python_kernel_name.startswith("aten.scatter_reduce"):
            line += ", ".join([""] + kwargs)
        else:
            if reduce:
                line += f", reduce={repr(reduce)}"
        line += ")"
        self.writeline(line)
    # 生成索引放置的回退函数，调用内核函数来执行操作
    def generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        # 将索引列表转换为字符串表示，用于生成函数调用的参数
        indices_str = f"{self.open_bracket}{', '.join(indices)}{self.closed_bracket}"
        # 构建参数列表
        args = [x, indices_str, values, accumulate]
        # 调用封装了内核调用的方法，并写入相应的代码行
        self.writeline(self.wrap_kernel_call(kernel, args))

    # 生成外部内核分配和查找模式（如果需要的话）的函数
    def generate_extern_kernel_alloc_and_find_schema_if_needed(
        self,
        buf_name: str,
        python_kernel_name: str,
        cpp_kernel_name: str,
        codegen_args: List[str],
        cpp_op_schema: str,
        cpp_kernel_key: str,
        cpp_kernel_overload_name: str = "",
        op_overload: Optional[torch._ops.OpOverload] = None,
        raw_args=None,
        outputs=None,
    ):
        # 生成 Python 内核的调用语句，并分配给指定的缓冲区名称
        self.writeline(f"{buf_name} = {python_kernel_name}({', '.join(codegen_args)})")

    # 应用装饰器 dynamo_timed 到下面的函数
    @dynamo_timed
    # 定义生成函数，接受一个布尔值参数 is_inference，表示是否进行推断过程
    def generate(self, is_inference):
        # 如果配置中开启了 profile_bandwidth 选项，则执行写入 Triton 头部信息的操作
        if config.profile_bandwidth:
            self.write_triton_header_once()
        
        # 创建一个 IndentedBuffer 对象作为结果
        result = IndentedBuffer()
        # 将对象自身的头部内容添加到结果中
        result.splice(self.header)
        
        # 如果满足条件：使用 Triton，有 cpp_wrapper，且是常量图模式，则重新初始化结果为空的 IndentedBuffer
        if V.graph.aot_mode and V.graph.cpp_wrapper and V.graph.is_const_graph:
            result = IndentedBuffer()
        
        # 使用 contextlib.ExitStack 管理多个上下文，进入 self.wrapper_call 的缩进上下文
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            
            # 如果配置中开启了 profiler_mark_wrapper_call 选项，则生成包装调用的性能分析标记
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            
            # 如果配置中开启了 profile_bandwidth 选项，则生成开始图的性能分析标记
            if config.profile_bandwidth:
                self.generate_start_graph()

            # 如果是推断过程并且配置中开启了 memory_planning，则执行内存规划
            if is_inference and config.memory_planning:
                self.memory_plan()
                # TODO: 整合内存规划和堆栈分配？
                self.allow_stack_allocation = False
            else:
                # 否则重用已有的内存规划
                self.memory_plan_reuse()

            # 如果配置中开启了 triton.store_cubin 选项，则生成重置内核保存标志位的操作
            if config.triton.store_cubin:
                self.generate_reset_kernel_saved_flags()

            # 遍历 self.lines 中的每一行
            for line in self.lines:
                # 如果该行是 WrapperLine 类型的对象，则调用其 codegen 方法生成代码
                if isinstance(line, WrapperLine):
                    line.codegen(self.wrapper_call)
                else:
                    # 否则将该行直接写入 wrapper_call
                    self.wrapper_call.writeline(line)

            # 获取输出引用列表
            output_refs = self.get_output_refs()
            # 标记输出类型
            self.mark_output_type()
            
            # 如果配置中开启了 triton.debug_sync_graph 选项，则同步图操作
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline(V.graph.device_ops.synchronize())

            # 如果配置中开启了 profile_bandwidth 选项，则生成结束图的性能分析标记
            if config.profile_bandwidth:
                self.generate_end_graph()

            # 如果配置中开启了 triton.store_cubin 选项，则保存未编译内核的操作
            if config.triton.store_cubin:
                self.generate_save_uncompiled_kernels()

            # 如果配置中开启了 triton.autotune_at_compile_time 选项，则生成并运行编译时自动调整块
            if config.triton.autotune_at_compile_time:
                self.generate_and_run_autotune_block()

            # 生成返回输出引用的操作
            self.generate_return(output_refs)

        # 最终化前缀内容
        self.finalize_prefix()
        # 将前缀内容添加到结果中
        result.splice(self.prefix)

        # 在结果的缩进下文中，将 self.wrapper_call 的内容添加到结果中
        with result.indent():
            result.splice(self.wrapper_call)

        # 在结果中生成后缀内容
        self.generate_before_suffix(result)
        # 将后缀内容添加到结果中
        result.splice(self.suffix)

        # 结束生成过程
        self.generate_end(result)

        # 添加基准测试的支持代码到结果中
        self.add_benchmark_harness(result)

        # 返回结果对象的字符串表示，包括行映射信息
        return result.getvaluewithlinemap()
    def generate_and_run_autotune_block(self):
        """
        Compose self.kernel_autotune_defs and self.kernel_autotune_calls into a single block of
        code and execute it to trigger Triton kernel compilation and auto-tuning
        """
        # 将自动生成的内核定义和调用代码组合成一个代码块，用于触发 Triton 内核编译和自动调优
        self.kernel_autotune_defs.splice(
            """
            async_compile.wait(globals())
            del async_compile
        """
        )
        # 创建一个空的作用域字典，用于执行动态生成的调优代码块
        scope = dict()  # type: ignore[var-annotated]
        # 将内核定义和调用代码合并为一个调优代码块
        tuning_code = (
            self.kernel_autotune_defs.getvalue() + self.kernel_autotune_calls.getvalue()
        )
        if output_code_log.level == logging.DEBUG:
            # 如果日志级别为 DEBUG，则将自动调优代码块保存到文件中
            # 创建一个临时文件
            with tempfile.NamedTemporaryFile(
                dir=cache_dir(), suffix=".py", delete=False
            ) as f:
                # 将调优代码块写入临时文件
                f.write(tuning_code.encode("utf-8"))
                file_path = f.name
            # 记录调优代码块的信息到日志中
            output_code_log.debug(
                "\nCompile-time auto-tuning code: \n%s\nAuto-tuning code written to %s",
                tuning_code,
                file_path,
            )
        # 执行生成的调优代码块，实现内核的自动调优
        exec(tuning_code, scope)

    def memory_plan(self):
        from .memory_planning import MemoryPlanner

        # 使用 MemoryPlanner 对象规划内存分配方案
        self.lines = MemoryPlanner(self).plan(self.lines)

    def memory_plan_reuse(self):
        out_names = V.graph.get_output_names()

        while (
            self.lines
            and isinstance(self.lines[-1], MemoryPlanningLine)
            # TODO: this seems legit, NullLine has no node
            and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
        ):
            # 移除最后几行无用的内存规划信息
            self.lines.pop()

        # 两次遍历生成代码中的分配计划
        planning_states = [MemoryPlanningState()]
        past_planning_states = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            if isinstance(line, MemoryPlanningLine):
                # 执行当前行的内存规划
                self.lines[i] = line.plan(planning_states[-1])
            elif isinstance(line, EnterSubgraphLine):
                # 进入子图时，创建新的内存规划状态
                planning_states.append(MemoryPlanningState())
            elif isinstance(line, ExitSubgraphLine):
                # 退出子图时，保存当前的内存规划状态
                past_planning_states.append(planning_states.pop())
        past_planning_states.append(planning_states.pop())
        # 确保所有内存规划状态已经处理完毕
        assert len(planning_states) == 0

        # 将所有嵌套作用域中分配的缓冲区大小求和，作为总分配大小
        total_allocated_buffer_size = sum(
            s.total_allocated_buffer_size for s in past_planning_states
        )

        # 根据总分配大小及配置决定是否允许栈上分配
        self.allow_stack_allocation = (
            self.allow_stack_allocation is not False
            and config.allow_stack_allocation
            and total_allocated_buffer_size <= MAX_STACK_ALLOCATION_SIZE
        )
    # 生成声明输入变量大小的代码，将其添加到指定的代码缓冲区中
    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"{self.declare}{name}_size = {name}.{self.size}{self.ending}")

    # 生成声明输入变量步长的代码，将其添加到指定的代码缓冲区中
    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        code.writeline(
            f"{self.declare}{name}_stride = {name}.{self.stride}{self.ending}"
        )

    # 为所有符号形状分配本地变量
    def codegen_inputs(
        self, code: IndentedBuffer, graph_inputs: Dict[str, ir.TensorBox]
    ):
        """Assign all symbolic shapes to locals"""

        @functools.lru_cache(None)
        def sizeof(name):
            # 调用输入大小变量声明函数，并返回生成的本地变量名
            self.codegen_input_size_var_decl(code, name)
            return f"{name}_size"

        @functools.lru_cache(None)
        def strideof(name):
            # 调用输入步长变量声明函数，并返回生成的本地变量名
            self.codegen_input_stride_var_decl(code, name)
            return f"{name}_stride"

        # 分配所有需要的符号形状到本地变量
        bound_vars: Set[sympy.Symbol] = set()

        # 检查是否为表达式
        def is_expr(x):
            return isinstance(x[1], sympy.Expr)

        # 过滤出表达式形式的图输入
        graph_inputs_expr = list(filter(is_expr, graph_inputs.items()))
        # 过滤出非表达式形式的图输入
        graph_inputs_tensors = list(
            filter(lambda x: not is_expr(x), graph_inputs.items())
        )

        # 处理表达式形式的输入
        for name, shape in graph_inputs_expr:
            if isinstance(shape, sympy.Symbol) and shape not in bound_vars:
                # 生成声明符号形状的代码，并将其添加到代码缓冲区
                code.writeline(f"{self.declare}{shape} = {name}{self.ending}")
                bound_vars.add(shape)

        # 处理非表达式形式的输入
        for name, value in graph_inputs_tensors:
            # 获取张量的大小
            shapes = value.get_size()
            for dim, shape in enumerate(shapes):
                if isinstance(shape, sympy.Symbol) and shape not in bound_vars:
                    # 生成声明符号形状的代码，并将其添加到代码缓冲区
                    code.writeline(
                        f"{self.declare}{shape} = {sizeof(name)}[{dim}]{self.ending}"
                    )
                    bound_vars.add(shape)

        # 处理非表达式形式的输入
        for name, value in graph_inputs_tensors:
            # 获取张量的步长
            shapes = value.get_stride()
            for dim, shape in enumerate(shapes):
                if isinstance(shape, sympy.Symbol) and shape not in bound_vars:
                    # 生成声明符号形状的代码，并将其添加到代码缓冲区
                    code.writeline(
                        f"{self.declare}{shape} = {strideof(name)}[{dim}]{self.ending}"
                    )
                    bound_vars.add(shape)

    # 确保计算符号大小已完成
    def ensure_size_computed(self, sym: sympy.Symbol):
        if isinstance(sym, sympy.Symbol) and symbol_is_type(sym, SymT.PRECOMPUTED_SIZE):
            if sym in self.computed_sizes:
                return
            self.computed_sizes.add(sym)
            expr = V.graph.sizevars.inv_precomputed_replacements[sym]
            self.writeline(
                f"{self.declare}{sym} = {self.expr_printer(expr)}{self.ending}"
            )

    # 最终化前缀
    def finalize_prefix(self):
        pass

    # 生成 Python 大小变量的表达式
    def codegen_python_sizevar(self, x: Expr, *, simplify: bool = True) -> str:
        return pexpr(x, simplify=simplify)

    # 生成大小变量的表达式
    def codegen_sizevar(self, x: Expr) -> str:
        return self.codegen_python_sizevar(x)
    # 生成元组访问的代码，返回形如 "basename[index]" 的字符串
    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        return f"{basename}[{index}]"

    # 生成 Python 形式的元组形状代码，将元组形状转换为字符串表示
    def codegen_python_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        # 将元组中的每个元素映射为字符串表示的大小变量，并存入列表 parts 中
        parts = list(map(self.codegen_python_sizevar, shape))
        # 如果 parts 为空，则返回空元组的表示 "()" 
        if len(parts) == 0:
            return "()"
        # 如果 parts 只有一个元素，则返回单元素元组的表示 "(parts[0], )"
        if len(parts) == 1:
            return f"({parts[0]}, )"
        # 否则，返回多元素元组的表示 "(part1, part2, ...)"
        return f"({', '.join(parts)})"

    # 生成元组形状代码的入口函数，调用内部的 Python 形式的元组生成函数
    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        return self.codegen_python_shape_tuple(shape)

    # 生成从池中分配内存的代码，返回相应的函数调用字符串
    def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
        return "alloc_from_pool({})".format(
            ", ".join(
                [
                    name,
                    pexpr(offset),  # bytes not numel
                    str(dtype),
                    self.codegen_shape_tuple(shape),
                    self.codegen_shape_tuple(stride),
                ]
            )
        )

    # 生成重新解释视图的代码，返回对应的函数调用字符串
    def codegen_reinterpret_view(self, data, size, stride, offset, writer) -> str:
        # 将大小和步长转换为字符串表示的元组形状
        size = self.codegen_shape_tuple(size)
        stride = self.codegen_shape_tuple(stride)
        offset = self.codegen_sizevar(offset)  # 转换偏移量为字符串表示
        return f"reinterpret_tensor({data.get_name()}, {size}, {stride}, {offset})"

    # 生成设备之间的张量复制代码，调用相应的写入方法
    def codegen_device_copy(self, src, dst):
        self.writeline(f"{dst}.copy_({src})")  # 调用 dst 对象的 copy_ 方法复制 src

    # 生成多输出的代码，声明名字和对应的值
    def codegen_multi_output(self, name, value):
        self.writeline(f"{self.declare}{name} = {value}{self.ending}")

    # 生成动态标量的代码，根据节点生成对应的赋值语句
    def codegen_dynamic_scalar(self, node):
        # 从节点的输入中获取数据的引用
        (data,) = (t.codegen_reference() for t in node.inputs)
        # 根据节点的 keypath 数量和类型生成不同的赋值语句
        if len(node.keypath) == 0:
            self.writeline(f"{node.sym} = {data}.item()")  # 直接赋值节点的符号为数据的单个值
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], ConvertIntKey):
            self.writeline(f"{node.sym} = 1 if {data}.item() else 0")  # 转换为整数键的情况下，赋值为条件表达式
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], DivideByKey):
            # 按除法键分割的情况下，分别计算未除数，进行断言，再计算商
            self.writeline(f"{node.sym}_undivided = {data}.item()")
            self.writeline(
                f"assert {node.sym}_undivided % {node.keypath[0].divisor} == 0, "
                f"f'{{{node.sym}_undivided}} not divisible by {node.keypath[0].divisor}'"
            )
            self.writeline(
                f"{node.sym} = {node.sym}_undivided // {node.keypath[0].divisor}"
            )
        else:
            raise AssertionError(f"unrecognized keypath {node.keypath}")  # 未识别的 keypath 抛出异常
        # 为了统一性，定义变量并赋值为 None，但不应该使用这个缓冲区
        self.writeline(f"{node.get_name()} = None")  # 定义节点的名称并赋值为 None
    # 在生成的代码中添加性能基准测试框架，用于调试
    def add_benchmark_harness(self, output):
        # 如果配置中未启用性能基准测试框架，则直接返回
        if not config.benchmark_harness:
            return

        # 编译生成模块用于性能测试
        self.benchmark_compiled_module(output)

        # 写入代码行，判断当前脚本是否作为主程序运行
        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            # 导入编译模块的性能测试函数，并调用之
            output.writelines(
                [
                    "from torch._inductor.wrapper_benchmark import compiled_module_main",
                    f"compiled_module_main('{get_benchmark_name()}', benchmark_compiled_module)",
                ]
            )

    # 定义一个内核函数
    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=True
    ):
        # 如果存在元数据，将其作为注释添加到内核函数体中
        metadata_comment = f"{metadata}\n" if metadata else ""
        body = f"\n\n{metadata_comment}{name} = {kernel}"
        # 将定义的内核函数体插入到头部
        self.header.splice(body)
        # 如果配置中指定了在编译时进行自动调优，则将内核自动调优定义插入到相应位置
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_defs.splice(body)

    # 生成指定内核名称和树结构的元素数量表达式
    def generate_numel_expr(self, kernel_name: str, tree):
        expr = f"{kernel_name}_{tree.prefix}numel"
        # 如果表达式及其对应的图形不在内核元素数量表达式集合中，则添加
        if (expr, V.graph) not in self.kernel_numel_expr:
            # 声明表达式在每个图形（作用域）中只出现一次
            self.kernel_numel_expr.add((expr, V.graph))
            # 写入表达式声明语句，例如 s0*64
            self.writeline(
                f"{self.declare}{expr} = {self.expr_printer(tree.numel)}{self.ending}"
            )
        else:
            # 如果表达式已经存在，则直接赋值
            self.writeline(f"{expr} = {self.expr_printer(tree.numel)}{self.ending}")
        # 返回符号调用参数对象，表示表达式及其元素数量
        return SymbolicCallArg(expr, tree.numel)

    # 生成工作空间分配语句
    def generate_workspace_allocation(self, nbytes, device, zero_fill):
        # 生成工作空间的分配语句，包括设备、数据类型、形状和步长等信息
        line = self.make_allocation(
            "workspace", device, torch.uint8, shape=(nbytes,), stride=(1,)
        )
        # 写入工作空间分配语句
        self.writeline(line)
        # 如果需要进行零填充，则在分配后立即进行
        if zero_fill:
            self.writeline(f"workspace.zero_(){self.ending}")

    # 包装内核调用语句
    def wrap_kernel_call(self, name, call_args):
        # 返回格式化后的内核调用语句
        return f"{name}({', '.join(call_args)}){self.ending}"

    # 生成性能分析器标记包装器调用语句
    def generate_profiler_mark_wrapper_call(self, stack):
        # 导入性能分析器记录函数
        self.wrapper_call.writeline("from torch.profiler import record_function")
        # 在包装器调用上下文中，记录函数执行时间并标记图形ID
        self.wrapper_call.writeline(
            f"with record_function('graph_{V.graph.graph_id}_inductor_wrapper_call'):"
        )
        # 进入代码块的缩进上下文
        stack.enter_context(self.wrapper_call.indent())

    # 生成开始图形的调用语句
    def generate_start_graph(self):
        # 写入开始图形的调用语句
        self.wrapper_call.writeline("start_graph()")
    # 调用生成结束图形的方法，向包装器写入
    # 根据参数生成示例的参数值，并返回生成的变量名或字符串表示
    def generate_example_arg_value(self, arg, arg_type=None, raw_arg=None, index=None):
        # 如果参数类型是 torch_dtype
        if isinstance(arg_type, torch_dtype):
            # 如果 V.graph 中存在缓冲区 arg，则使用它
            if V.graph.get_buffer(arg) is not None:
                buf_name = arg
                buf = V.graph.get_buffer(arg)
            else:
                # 否则确保 raw_arg 不为空，否则抛出异常
                assert (
                    raw_arg is not None
                ), "V.graph.get_buffer(arg) and raw_arg can't be None at the same time"
                # 使用临时变量名创建缓冲区
                buf_name = f"tmp_arg_{index}"
                buf = raw_arg

            # 获取缓冲区的大小，并进行大小提示处理
            size = V.graph.sizevars.size_hints(
                buf.get_size(),
                fallback=config.unbacked_symint_fallback,
            )
            # 获取缓冲区的步长，并进行大小提示处理
            stride = V.graph.sizevars.size_hints(
                buf.get_stride(),
                fallback=config.unbacked_symint_fallback,
            )
            # 获取缓冲区的设备信息
            device = buf.get_device()
            # 获取缓冲区的数据类型
            dtype = buf.get_dtype()
            # 获取缓冲区的偏移量，并进行大小提示处理
            offset = V.graph.sizevars.size_hint(
                buf.layout.offset,
                fallback=config.unbacked_symint_fallback,
            )
            # 生成示例值的字符串表示
            value = f"generate_example_value({size}, {stride}, '{device}', {dtype}, {offset})"
            # 将生成的赋值语句写入到 kernel_autotune_calls 中
            self.kernel_autotune_calls.writeline(f"{buf_name} = {value}")
            # 返回生成的变量名
            return buf_name
        # 如果参数是 int、float 或 bool 类型，则直接返回其字符串表示
        elif isinstance(arg, (int, float, bool)):
            return str(arg)
        else:
            # 否则认为 arg 是一个符号或符号表达式
            if isinstance(arg, str):
                # 如果 arg 在 _meta_vars 中，则直接返回
                if arg in self._meta_vars:
                    return arg
                # 如果 raw_arg 为空，则返回字符串 "None"
                if raw_arg is None:
                    return "None"
                # 否则将 arg 设置为 raw_arg
                arg = raw_arg
            # 如果 arg 是 SymbolicCallArg 类型，则将其转换为 inner_expr
            if isinstance(arg, SymbolicCallArg):
                arg = arg.inner_expr
            # 如果 arg 在预计算替换的反向映射中，则使用其替换值
            if arg in V.graph.sizevars.inv_precomputed_replacements:
                arg = V.graph.sizevars.inv_precomputed_replacements[arg]
            # 返回 arg 的大小提示处理后的字符串表示
            return str(
                V.graph.sizevars.size_hint(
                    arg,
                    fallback=config.unbacked_symint_fallback,
                )
            )

    # 根据给定的参数生成内核调用
    def generate_kernel_call(
        self,
        kernel_name,
        call_args,
        grid=None,
        device_index=None,
        cuda=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
    ):
        # 实现省略，未提供详细的代码内容

    # 将一行文本添加到 self.lines 列表中
    def writeline(self, line):
        self.lines.append(line)

    # 将多行文本逐行添加到 self.lines 列表中
    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    # 将一个 LineContext 对象添加到 self.lines 列表中
    def enter_context(self, ctx):
        self.lines.append(LineContext(ctx))
    # 将值转换为字符串表示，用于生成参数列表
    def val_to_arg_str(self, s, type_=None):
        # 导入相关函数和检查 Triton 包是否可用
        from torch.utils._triton import dtype_to_string, has_triton_package
        
        # 如果 Triton 包可用，则导入 Triton 模块
        if has_triton_package():
            import triton
        
        # 如果 s 是 SymTypes 类型，则返回其节点表达式的字符串表示
        if isinstance(s, SymTypes):
            return pexpr(s.node.expr)
        # 如果 s 是 sympy.Expr 类型，则返回其表达式的字符串表示
        elif isinstance(s, sympy.Expr):
            return pexpr(s)
        # 如果 s 是 tuple 或 list 类型
        elif isinstance(s, (tuple, list)):

            @dataclasses.dataclass
            class Shim:
                ref: Any

                def __repr__(self):
                    return self.ref

            # 返回元组或列表的字符串表示，其中每个元素通过递归调用 val_to_arg_str 处理
            return repr(type(s)(Shim(self.val_to_arg_str(a)) for a in s))
        # 如果 s 是 torch._ops.OpOverload 类型，则返回其限定名称的字符串表示
        elif isinstance(s, torch._ops.OpOverload):
            return _get_qualified_name(s)
        # 如果 s 是 ir.Buffer 或 ReinterpretView 类型，则返回其代码生成引用的字符串表示
        elif isinstance(s, (ir.Buffer, ReinterpretView)):
            return s.codegen_reference()
        # 如果 Triton 包可用且 s 是 triton.language.dtype 类型，则返回其字符串表示
        elif has_triton_package() and isinstance(s, triton.language.dtype):  # type: ignore[possibly-undefined]
            return dtype_to_string(s)
        # 否则，返回 s 的字符串表示
        else:
            return repr(s)

    # 以下方法用于内存管理

    # 根据 buffer 生成内存分配语句
    def make_buffer_allocation(self, buffer):
        # 获取 buffer 的设备信息、数据类型、形状和步长
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        # 调用 make_allocation 生成分配语句并返回
        return self.make_allocation(buffer.get_name(), device, dtype, shape, stride)

    # 根据名称、设备、数据类型、形状和步长生成内存分配语句
    def make_allocation(self, name, device, dtype, shape, stride):
        # 如果设备类型为 "cpu" 或 "cuda"，使用优化路径生成分配语句
        if device.type in ("cpu", "cuda"):
            return (
                f"{name} = empty_strided_{device.type}("
                f"{self.codegen_shape_tuple(shape)}, "
                f"{self.codegen_shape_tuple(stride)}, "
                f"{dtype})"
            )
        # 对于其他设备，生成通用的分配语句
        return (
            f"{name} = empty_strided("
            f"{self.codegen_shape_tuple(shape)}, "
            f"{self.codegen_shape_tuple(stride)}, "
            f"device='{device.type}', dtype={dtype})"
        )

    # 生成张量别名语句，将旧名称的张量重命名为新名称
    def make_tensor_alias(self, new_name, old_name, comment=""):
        return f"{self.declare}{new_name} = {old_name}{self.ending}  {self.comment} {comment}"

    # 根据 buffer 生成内存释放语句
    def make_buffer_free(self, buffer):
        return f"del {buffer.get_name()}"

    # 根据名称列表生成多个内存释放语句
    def make_free_by_names(self, names_to_del: List[str]):
        return f"del {', '.join(name for name in names_to_del)}"

    # 生成精确的缓冲区重用代码生成语句，包括声明、赋值、删除和注释
    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        return f"{self.declare_maybe_reference}{new_name} = {old_name}{del_line}{self.ending}  {self.comment} reuse"
    # 定义方法用于处理旧缓冲区到新缓冲区的重用，包括可能的释放操作
    def make_buffer_reuse(self, old, new, delete_old: bool):
        # 断言旧缓冲区和新缓冲区的数据类型相同
        assert old.get_dtype() == new.get_dtype()
        # 获取旧缓冲区的名称和新缓冲区的名称
        old_name = old.get_name()
        new_name = new.get_name()
        # 初始化删除行，默认为分号
        del_line = ";"
        # 如果旧缓冲区名称不在图的输出名称列表中且需要删除旧缓冲区，则进行释放操作
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        # 如果旧缓冲区的大小、步长与新缓冲区相同
        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            # 如果旧缓冲区名称在栈分配的缓冲区中，将新缓冲区添加到栈分配的缓冲区中
            if old_name in self.stack_allocated_buffers:
                self.stack_allocated_buffers[new_name] = new
            # 返回确切的缓冲区重用代码生成结果
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        # 否则，生成重新解释视图的代码
        reinterpret_view = self.codegen_reinterpret_view(
            old, new.get_size(), new.get_stride(), 0, self.wrapper_call
        )
        # 如果重新解释视图已经在栈分配的缓冲区中，将新缓冲区添加到栈分配的缓冲区中
        if reinterpret_view in self.stack_allocated_buffers:
            self.stack_allocated_buffers[new_name] = new
        # 返回分配代码，包括删除行和重用的注释
        return f"{self.declare_maybe_reference}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"

    # 生成延迟分配的代码
    def codegen_deferred_allocation(self, name, layout):
        self.writeline(
            DeferredLine(
                name,
                # 生成声明可能引用的语句和视图代码生成的引用代码，包括别名的注释
                f"{self.declare_maybe_reference}{name} = {layout.view.codegen_reference()}{self.ending}  "
                f"{self.comment} alias",
            )
        )

    # 生成分配缓冲区的代码
    def codegen_allocation(self, buffer):
        name = buffer.get_name()

        # 如果缓冲区已经被移除或者已经分配，则直接返回
        if name in V.graph.removed_buffers or name in self.allocated:
            return
        # 将缓冲区名称添加到已分配集合中
        self.allocated.add(name)
        # 如果缓冲区是外部内核分配或者多重输出类型，则直接返回
        if isinstance(
            buffer,
            (ir.ExternKernelAlloc, ir.MultiOutput),
        ):
            return

        # 获取缓冲区的布局
        layout = buffer.get_layout()
        # 如果布局是变异布局，直接返回
        if isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            return
        # 如果布局是NoneLayout，直接返回
        if isinstance(layout, ir.NoneLayout):
            return
        # 如果布局是非拥有布局
        if isinstance(layout, ir.NonOwningLayout):
            # 断言视图类型是重新解释视图，否则抛出异常
            assert isinstance(
                layout.view, ir.ReinterpretView
            ), f"unexpected {type(layout.view)}: {layout.view}"
            # 递归调用分配代码生成方法和延迟分配代码生成方法
            self.codegen_allocation(layout.view.data)
            self.codegen_deferred_allocation(name, layout)
            return

        # 否则，写入分配行代码
        self.writeline(AllocateLine(self, buffer))

    # 生成释放缓冲区的代码
    def codegen_free(self, buffer):
        # 断言工作空间大小为0，目前只支持零工作空间大小
        assert (
            buffer.get_workspace_size() == 0
        ), "Only support zero workspace size for now!"

        # 获取缓冲区名称
        name = buffer.get_name()

        # 如果是输入缓冲区类型，则生成释放缓冲区的代码并返回
        if isinstance(buffer, ir.InputBuffer):
            self.writeline(self.make_buffer_free(buffer))
            return

        # 如果不能重用缓冲区，则直接返回
        if not self.can_reuse(buffer):
            return
        # 将缓冲区名称添加到已释放集合中
        self.freed.add(name)

        # 写入如果未重用则释放行代码
        self.writeline(FreeIfNotReusedLine(self, buffer))
    # 检查给定的输入缓冲是否可以被重新使用
    def can_reuse(self, input_buffer, output_buffer=None):
        # 获取输入缓冲的名称
        name = input_buffer.get_name()
        # 如果输入缓冲的名称出现在被移除的缓冲区、图输入、常量、torchbind 常量、永不重用缓冲区或已释放的缓冲区集合中，返回 False
        if (
            name in V.graph.removed_buffers
            or name in V.graph.graph_inputs
            or name in V.graph.constants
            or name in V.graph.torchbind_constants
            or name in V.graph.never_reuse_buffers
            or name in self.freed
        ):
            return False

        # 允许重新使用输入缓冲
        return True

    # 检查给定的缓冲是否被重新使用，通过比较缓冲的名称
    def did_reuse(self, buffer, reused_buffer):
        # 检查给定的缓冲是否在 self.reuses 字典中，并且被重新使用的缓冲的名称匹配
        return (
            buffer.get_name() in self.reuses
            and self.reuses[buffer.get_name()] == reused_buffer.get_name()
        )

    # 生成支持就地重用的代码，确保输入缓冲和输出缓冲的重用键相同
    def codegen_inplace_reuse(self, input_buffer, output_buffer):
        assert buffer_reuse_key(input_buffer) == buffer_reuse_key(output_buffer)
        # 生成输入缓冲的分配代码
        self.codegen_allocation(input_buffer)
        # 将输入缓冲的名称添加到已释放集合中
        self.freed.add(input_buffer.get_name())
        # 将输出缓冲的名称添加到已分配集合中
        self.allocated.add(output_buffer.get_name())
        # 记录输出缓冲重新使用了输入缓冲的信息
        self.reuses[output_buffer.get_name()] = input_buffer.get_name()
        # 在生成代码中写入支持重用的行
        self.writeline(ReuseLine(self, input_buffer, output_buffer))

    # 生成未支持的符号声明代码，确保在 CppWrapperCpu 中只生成一次声明
    def codegen_unbacked_symbol_decl(self, symbol):
        name = str(symbol)
        if name in self.unbacked_symbol_decls:
            return name
        else:
            # 当在 CppWrapperCpu 中时，只生成一次声明
            self.unbacked_symbol_decls.add(name)
            return self.declare + name

    # 生成子图的前缀代码，用外部输入初始化内部输入
    def codegen_subgraph_prefix(self, subgraph, outer_inputs, outer_outputs):
        for inner_input, outer_input in zip(subgraph.graph.graph_inputs, outer_inputs):
            self.writeline(f"{self.declare}{inner_input} = {outer_input}{self.ending}")

    # 生成子图的后缀代码，将内部输出赋给外部输出
    def codegen_subgraph_suffix(self, subgraph, outer_inputs, outer_outputs):
        for inner_output, outer_output in zip(
            subgraph.graph.graph_outputs, outer_outputs
        ):
            self.writeline(
                f"{outer_output} = {inner_output.codegen_reference()}{self.ending}"
            )

    # 生成子图的代码，包括前缀、子图本身、后缀的生成
    def codegen_subgraph(self, subgraph, outer_inputs, outer_outputs):
        try:
            # 将当前子图推入已生成子图堆栈
            self.push_codegened_graph(subgraph.graph)
            # 在生成代码中写入子图名称的注释
            self.writeline(f"{self.comment} subgraph: {subgraph.name}")
            # 生成子图的前缀代码
            self.codegen_subgraph_prefix(subgraph, outer_inputs, outer_outputs)
            # 保存父图，并设置当前图为子图的图处理程序
            parent_graph = V.graph
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.codegen_subgraph(
                    parent_graph=parent_graph,
                )
            # 生成子图的后缀代码
            self.codegen_subgraph_suffix(subgraph, outer_inputs, outer_outputs)
        finally:
            # 从已生成子图堆栈中弹出当前子图
            self.pop_codegened_graph()
    def codegen_conditional(self, conditional):
        # 获取条件结构的名称
        name = conditional.get_name()

        # 生成一个空列表，长度为条件结构输出的数量
        self.writeline(f"{name} = [None] * {len(conditional.outputs)}")

        # 生成外部输入引用列表，每个元素是操作数的代码生成引用
        outer_inputs = [buf.codegen_reference() for buf in conditional.operands]
        
        # 生成外部输出引用列表，每个元素是条件结构输出的索引表达式
        outer_outputs = [f"{name}[{i}]" for i in range(len(conditional.outputs))]

        # 获取条件表达式的代码生成引用
        predicate = conditional.predicate.codegen_reference()

        # 如果条件不是常量形状缓冲区，则将 Tensor 条件移到主机端
        if not isinstance(conditional.predicate, ir.ShapeAsConstantBuffer):
            predicate = f"{predicate}.item()"  # 将条件转换为其值

        # 生成一个空列表，长度为条件结构输出的数量（重复的代码，可能为错误）
        self.writeline(f"{name} = [None] * {len(conditional.outputs)}")

        # 生成条件语句的代码行
        self.writeline(f"if {predicate}:")
        self.writeline(EnterSubgraphLine(self, conditional.true_subgraph.graph))
        self.codegen_subgraph(conditional.true_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))

        # 生成 else 分支的代码行
        self.writeline("else:")
        self.writeline(EnterSubgraphLine(self, conditional.false_subgraph.graph))
        self.codegen_subgraph(conditional.false_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))


    def codegen_while_loop(self, while_loop):
        # 获取 while 循环的名称
        name = while_loop.get_name()

        # 生成外部携带输入的引用列表，每个元素是携带输入缓冲区的代码生成引用
        outer_carried_inputs = [
            buf.codegen_reference() for buf in while_loop.carried_inputs
        ]

        # 生成外部附加输入的引用列表，每个元素是附加输入缓冲区的代码生成引用
        outer_additional_inputs = [
            buf.codegen_reference() for buf in while_loop.additional_inputs
        ]

        # 生成一个空列表，长度为外部携带输入的数量
        self.writeline(f"{name} = [None] * {len(outer_carried_inputs)}")

        # 为每个外部携带输入设置初始状态，在循环之前
        for i, inp in enumerate(outer_carried_inputs):
            self.writeline(f"{name}[{i}] = {inp}")

        # 生成条件外部输入引用列表，包括外部携带输入和外部附加输入
        cond_outer_inputs = [
            *[f"{name}[{i}]" for i in range(len(outer_carried_inputs))],
            *outer_additional_inputs,
        ]

        # 生成条件外部输出引用列表，包含条件结果的名称
        cond_outer_outputs = [f"{name}_cond_result"]

        # 生成主体外部输入引用列表，与 cond_fn 和 body_fn 使用相同的输入
        body_outer_inputs = list(cond_outer_inputs)

        # 主体外部输出引用列表，只传递携带输入的部分，附加输入按原样传递
        body_outer_outputs = body_outer_inputs[: len(outer_carried_inputs)]

        # 生成 while 循环的无限循环开始
        self.writeline("while True:")

        # 进入条件子图代码生成
        self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))
        self.codegen_subgraph(
            while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
        )

        # 生成条件不成立时跳出循环的代码行
        self.writeline(f"if not {cond_outer_outputs[0]}.item(): break")

        # 退出条件子图代码生成
        self.writeline(ExitSubgraphLine(self))

        # 进入主体子图代码生成
        self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
        self.codegen_subgraph(
            while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
        )

        # 退出主体子图代码生成
        self.writeline(ExitSubgraphLine(self))
    # 定义一个静态方法，用于尝试从参数 x 中获取静态已知的整数或者返回 None
    def statically_known_int_or_none(x):
        try:
            # 如果 x 具有属性 "free_symbols"，则返回 None
            if getattr(x, "free_symbols", None):
                # _maybe_evaluate_static 将返回 (s0 // (2 // s0)) 作为 2，
                # 但是实际的代码生成仍将在这里生成完整的表达式。
                return None
            # 从图的图形属性中的形状环境中尝试静态评估 x，并转换为整数返回
            val = V.graph._shape_env._maybe_evaluate_static(x)
            return int(val)
        except Exception:
            # 发生任何异常则返回 None
            return None

    @staticmethod
    # 定义一个静态方法，用于尝试从列表 lst 中获取静态已知的整数列表或者返回 None
    def statically_known_list_of_ints_or_none(lst):
        result = []
        # 遍历列表 lst 中的每个元素 x
        for x in lst:
            # 调用 statically_known_int_or_none 方法尝试获取 x 的静态整数值
            num = WrapperCodeGen.statically_known_int_or_none(x)
            # 如果 num 是 None，则直接返回 None
            if num is None:
                return None
            # 否则将 num 添加到结果列表中
            result.append(num)
        # 返回整数列表 result
        return result

    @staticmethod
    # 定义一个静态方法，用于检查列表 lst 是否能够静态地确定其包含整数列表
    def is_statically_known_list_of_ints(lst):
        # 调用 statically_known_list_of_ints_or_none 方法，如果返回结果不是 None，则返回 True，否则返回 False
        return WrapperCodeGen.statically_known_list_of_ints_or_none(lst) is not None

    @staticmethod
    # 定义一个静态方法，用于尝试从 buffer 的大小信息中获取静态已知的整数列表或者返回 None
    def static_shape_for_buffer_or_none(buffer):
        # 调用 statically_known_list_of_ints_or_none 方法，传入 buffer 的大小信息，并返回结果
        return WrapperCodeGen.statically_known_list_of_ints_or_none(buffer.get_size())

    @staticmethod
    # 定义一个静态方法，用于检查 buffer 是否能够静态地确定其具有静态形状
    def can_prove_buffer_has_static_shape(buffer):
        # 调用 static_shape_for_buffer_or_none 方法，如果返回结果不是 None，则返回 True，否则返回 False
        return WrapperCodeGen.static_shape_for_buffer_or_none(buffer) is not None
```