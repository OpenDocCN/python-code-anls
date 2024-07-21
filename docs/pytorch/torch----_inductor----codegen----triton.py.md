# `.\pytorch\torch\_inductor\codegen\triton.py`

```py
# 在使用 mypy 时允许未标注的函数定义
# 从 __future__ 模块导入注释作为类型提示的特性
from __future__ import annotations

# 导入用于数据类的装饰器
import dataclasses
# 导入 functools 模块
import functools
# 导入 itertools 模块
import itertools
# 导入 logging 模块
import logging
# 导入 os 模块
import os
# 导入 textwrap 模块
import textwrap
# 导入 functools 模块中的 lru_cache 装饰器
from functools import lru_cache
# 导入类型提示相关的模块
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

# 导入 sympy 符号计算库
import sympy

# 导入 torch 深度学习框架
import torch
# 导入 torch 内部日志模块
import torch._logging
# 从 torch._dynamo.utils 模块导入保留随机数生成器状态的函数
from torch._dynamo.utils import preserve_rng_state

# 导入 torch._inductor.runtime.hints 模块中的 AutotuneHint 和 DeviceProperties 类
from torch._inductor.runtime.hints import AutotuneHint, DeviceProperties
# 导入 torch._prims_common 模块中的 is_integer_dtype 函数
from torch._prims_common import is_integer_dtype
# 导入 torch.utils._sympy.functions 模块中的 CeilDiv, FloorDiv 和 ModularIndexing 类
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
# 导入 torch.utils._triton 模块中的 has_triton_package 函数
from torch.utils._triton import has_triton_package
# 导入 ...utils._sympy.symbol 模块中的符号相关函数和类
from ...utils._sympy.symbol import free_symbol_is_type, prefix_str, symbol_is_type, SymT
# 导入 ...utils._sympy.value_ranges 模块中的 ValueRanges 类
from ...utils._sympy.value_ranges import ValueRanges

# 导入当前包的相关模块和类
from .. import config, ir
# 从 ..codecache 模块导入代码哈希、路径获取和 Python 代码缓存相关函数和类
from ..codecache import code_hash, get_path, PyCodeCache
# 从 ..metrics 模块导入是否启用度量表和内核元数据日志函数
from ..metrics import is_metric_table_enabled, log_kernel_metadata
# 从 ..runtime.hints 模块导入减少提示相关类和常量
from ..runtime.hints import ReductionHint, TRITON_MAX_BLOCK
# 从 ..runtime.runtime_utils 模块导入 GPU 性能评估相关函数
from ..runtime.runtime_utils import do_bench_gpu, get_max_y_grid, next_power_of_2
# 从 ..utils 模块导入各种实用函数和类
from ..utils import (
    cache_on_self,
    get_bounds_index_expr,
    get_fused_kernel_name,
    get_kernel_metadata,
    is_welford_reduction,
    Placeholder,
    sympy_dot,
    sympy_subs,
)
# 从 ..virtualized 模块导入操作相关类和函数
from ..virtualized import _ops as ops, OpsHandler, ReductionType, StoreMode, V
# 从 ..wrapper_benchmark 模块导入通过源代码获取内核类别的函数
from ..wrapper_benchmark import get_kernel_category_by_source_code
# 从 .common 模块导入后端特性、常量折叠、变量折叠、延迟行、缩进缓冲区、操作重写、Python 打印机、大小参数、张量参数等
from .common import (
    BackendFeature,
    CSE,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
    SizeArg,
    TensorArg,
)
# 从 .simd 模块导入常量表示、迭代范围入口、根迭代范围、符号表达式、SIMD 内核、SIMD 调度等
from .simd import (
    constant_repr,
    IterationRangesEntry,
    IterationRangesRoot,
    pexpr,
    SIMDKernel,
    SIMDScheduling,
)
# 从 .triton_utils 模块导入配置、签名和元信息相关函数
from .triton_utils import config_of, signature_of, signature_to_meta

# 如果是类型检查阶段，导入 IRNode 类
if TYPE_CHECKING:
    from ..ir import IRNode

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)
# 获取性能提示的日志记录器
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
# 获取调度的日志记录器
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
# 获取融合的日志记录器
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")

# 使用 lru_cache 装饰器，缓存 gen_attr_descriptor_import 函数的结果
@lru_cache(None)
def gen_attr_descriptor_import():
    """
    如果 triton 包的版本足够新，导入 AttrsDescriptor 类。
    """
    # 如果没有安装 triton 包，则返回空字符串
    if not has_triton_package():
        return ""

    # 导入 triton.compiler.compiler 模块
    import triton.compiler.compiler

    # 如果 triton.compiler.compiler 模块中有 AttrsDescriptor 类定义，则导入该类
    if hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        return "from triton.compiler.compiler import AttrsDescriptor"
    else:
        return ""


# 使用 lru_cache 装饰器，缓存 gen_common_triton_imports 函数的结果
@lru_cache(None)
def gen_common_triton_imports():
    # 创建一个缩进缓冲区对象
    imports = IndentedBuffer()
    # 向缓冲区添加字符串，导入 triton 和 triton.language 模块
    imports.splice(
        """
        import triton
        import triton.language as tl
        """
    )
    # 如果有需要，导入 AttrsDescriptor 类
    if attr_desc := gen_attr_descriptor_import():
        imports.writeline(attr_desc)
    # 导入需要的模块到当前命名空间中
    imports.splice(
        """
        from torch._inductor.runtime import triton_helpers, triton_heuristics
        from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
        from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
        """
    )
    # 返回导入的模块集合的字符串表示形式
    return imports.getvalue()
# 创建一个字典，将SymT.XBLOCK、SymT.YBLOCK和SymT.RINDEX映射为对应的符号表达式，用于表示块的偏移量
block_offsets = {
    symt: sympy.Symbol(f"{prefix_str[symt]}offset", integer=True)
    for symt in [SymT.XBLOCK, SymT.YBLOCK, SymT.RINDEX]
}

# 创建一个字典，将SymT.XBLOCK、SymT.YBLOCK和SymT.RINDEX映射为对应的符号表达式，用于表示块的大小
block_sizes = {
    symt: sympy.Symbol(f"{prefix_str[symt].upper()}BLOCK", integer=True, nonzero=True)
    for symt in [SymT.XBLOCK, SymT.YBLOCK, SymT.RINDEX]
}


@dataclasses.dataclass
class IndexingOptions:
    index_str: str  # 索引选项的字符串表示
    mask_vars: Set[str]  # 用于掩码的变量集合
    mask_str: str  # 掩码选项的字符串表示
    expand_str: Optional[str]  # 可选的扩展字符串表示
    _has_rindex: bool  # 是否具有相对索引标志
    index: sympy.Expr  # 索引表达式

    def has_mask(self):
        return bool(self.mask_vars)  # 判断是否存在掩码变量

    def has_indirect(self):
        return free_symbol_is_type(self.index, SymT.TMP)  # 判断索引是否具有间接性质

    def has_rindex(self):
        return self._has_rindex  # 判断是否具有相对索引标志

    def has_tmpmask(self):
        return "tmp" in self.mask_str  # 判断是否具有临时掩码

    def has_rmask(self):
        return "rmask" in self.mask_str  # 判断是否具有相对掩码


@dataclasses.dataclass
class BlockPtrOptions:
    params: BlockParameters  # 块参数对象
    constant_offset: sympy.Expr  # 常量偏移表达式
    order: List[int]  # 顺序列表，指示块指针选项的顺序
    mask_vars: Set[str]  # 用于掩码的变量集合
    reshape_suffix: List[str]  # 重塑后缀列表，用于表示块指针选项的重塑形状

    @property
    def shape(self) -> List[sympy.Expr]:
        return self.params.shape  # 返回块的形状列表

    @property
    def block_shape(self) -> List[sympy.Expr]:
        return self.params.block_shape  # 返回块的块形状列表

    @property
    def strides(self) -> List[sympy.Expr]:
        return self.params.strides  # 返回块的步幅列表

    @property
    def offsets(self) -> List[sympy.Expr]:
        return self.params.offsets  # 返回块的偏移量列表

    @staticmethod
    def create(
        *,
        params: BlockParameters,  # 块参数对象
        constant_offset: sympy.Expr,  # 常量偏移表达式
        range_trees: List[IterationRangesEntry],  # 迭代范围树列表
        mask_vars: Set[str],  # 用于掩码的变量集合
        reshape_suffix: List[str],  # 重塑后缀列表，用于表示块指针选项的重塑形状
    ) -> BlockPtrOptions:
        """Helper to create a BlockPtrOptions instance"""
        # 创建一个 BlockPtrOptions 实例的辅助函数

        reshape_suffix = [f"{t.prefix.upper()}BLOCK" for t in range_trees]

        # 只有在输出的秩与块的秩相同时才丢弃广播维度，否则会导致形状错误。
        drop_broadcasts = len(reshape_suffix) == len(params.strides)

        broadcasting_dim = [s == 0 for s in params.strides]
        for i, is_broadcasting in enumerate(broadcasting_dim):
            if is_broadcasting and drop_broadcasts:
                # 为了性能，丢弃任何步长为0的维度
                reshape_suffix[i] = "1"

        if V.kernel.no_x_dim:
            assert range_trees[0].prefix == "x"
            reshape_suffix.pop(0)

        if (
            not V.kernel.inside_reduction
            and len(params.strides) == len(V.kernel.numels) - 1
            and V.kernel.numels[-1] != 1
        ):
            # 需要扩展秩以匹配当 self.inside_reduction=True 时的秩
            reshape_suffix.append("1")

        def filter(it):
            """Removes any broadcasting dims from a given sequence"""
            # 从给定序列中删除任何广播维度
            assert len(it) == len(broadcasting_dim)
            return [
                item
                for item, is_broadcasting in zip(it, broadcasting_dim)
                if not is_broadcasting or not drop_broadcasts
            ]

        # 从输入中去除广播维度
        params = BlockParameters(
            **{key: filter(val) for key, val in dataclasses.asdict(params).items()}
        )

        def lookup_size(exprs: Iterable[sympy.Expr]) -> List[sympy.Expr]:
            # 查找预计算大小
            return [V.graph.sizevars.lookup_precomputed_size(expr) for expr in exprs]

        # 查找预计算的大小
        params.shape = lookup_size(params.shape)
        params.strides = lookup_size(params.strides)

        return BlockPtrOptions(
            params=params,
            constant_offset=V.graph.sizevars.lookup_precomputed_size(constant_offset),
            order=list(reversed(range(len(params.shape)))),
            mask_vars=mask_vars,
            reshape_suffix=reshape_suffix,
        )

    def replace_roffset(self, expr: sympy.Expr, replacement: sympy.Expr) -> sympy.Expr:
        """
        Replaces instances of roffset with the new expression.
        """
        # 用新表达式替换 roffset 的实例。
        roffset = block_offsets[SymT.RINDEX]
        return sympy_subs(expr, {roffset: replacement})
    def format(self, name: str, roffset=True) -> str:
        """
        Codegen a call to tl.make_block_ptr()

        Args:
            name: variable name for pointer
            roffset: should roffset be included in offsets=..., for use with tl.advance()

        Returns:
            "tl.make_block_ptr(...)"
        """
        # 获取索引到字符串转换函数
        f = V.kernel.index_to_str
        # 复制偏移量列表
        offsets = [*self.offsets]
        # 如果不包括 roffset，则将偏移量中的 roffset 替换为 0
        if not roffset:
            offsets = [
                self.replace_roffset(offset, sympy.Integer(0)) for offset in offsets
            ]
        # 构造参数列表
        args = [
            f"{name} + ({f(self.constant_offset)})"
            if self.constant_offset != 0
            else name,
            f"shape={f(self.shape)}",
            f"strides={f(self.strides)}",
            f"block_shape={f(self.block_shape)}",
            f"order={f(self.order)}",
            f"offsets={f(offsets)}",
        ]
        # 返回调用 tl.make_block_ptr() 的代码字符串
        return f"tl.make_block_ptr({', '.join(args)})"

    @cache_on_self
    def boundary_check(self) -> List[int]:
        """List of indices to pass to tl.load(boundary_check=...)"""
        # 获取图中的尺寸变量
        sizevars = V.graph.sizevars

        # 替换形状表达式中的最大块大小
        # 这在多个检查中起作用，因为块大小是2的幂次方
        block_to_max: Dict[sympy.Expr, Any] = {
            block_size: TRITON_MAX_BLOCK[prefix_str[symt].upper()]
            for symt, block_size in block_sizes.items()
        }

        # 返回需要传递给 tl.load(boundary_check=...) 的索引列表
        return [
            idx
            for idx in range(len(self.shape))
            if (
                not sizevars.statically_known_equals(
                    self.strides[idx], sympy.Integer(0)
                )
                and not sizevars.statically_known_multiple_of(
                    self.shape[idx], self.block_shape[idx]
                )
                and not sizevars.statically_known_multiple_of(
                    self.shape[idx], sympy_subs(self.block_shape[idx], block_to_max)
                )
                and not (
                    V.kernel.no_x_dim
                    and self.block_shape[idx] == block_sizes[SymT.XBLOCK]
                )
            )
        ]

    def advance_roffset(self):
        """
        Codegen string to pass to tl.advance(name, ...).

        Advance is the difference between offsets in each loop iteration.
        To compute it, we replace roffset with multiples of RBLOCK.
        Since we expect roffset to vary in range(0, rnumel, RBLOCK), the first
        iteration has roffset=0, while the second has roffset=RBLOCK.
        """
        # 获取 RINDEX 对应的块大小
        rblock = block_sizes[SymT.RINDEX]
        # 计算每个偏移量的进步
        advance = [
            (
                self.replace_roffset(offset, rblock)
                - self.replace_roffset(offset, sympy.Integer(0))
            )
            for offset in self.offsets
        ]
        # 返回进步字符串，用于传递给 tl.advance(name, ...)
        return V.kernel.index_to_str(advance)

    def has_indirect(self):
        # block_ptr 无法进行间接索引，因此始终返回 False
        return False  # block_ptr can't do indirect indexing
    # 检查在 self.block_shape 中是否存在任何表达式的自由符号类型为 SymT.RINDEX
    def has_rindex(self) -> bool:
        return any(free_symbol_is_type(expr, SymT.RINDEX) for expr in self.block_shape)
    
    # 检查是否存在间接索引（rindex），委托给 has_rindex 方法
    def has_rmask(self):
        return self.has_rindex()
    
    # 检查是否有临时掩码（tmpmask），始终返回 False，因为 block_ptr 无法进行间接索引
    def has_tmpmask(self):
        return False  # block_ptr can't do indirect indexing
    
    # 检查是否存在掩码（mask），依据是否有边界检查的结果来返回布尔值
    def has_mask(self):
        return bool(self.boundary_check())
# 根据传递的 value、old_shape 和 new_shape 执行形状重塑操作，修复了 Triton 库的一个问题
def triton_reshape(value: str, old_shape: List[str], new_shape: List[str]):
    """Workaround https://github.com/openai/triton/issues/2836"""
    # 断言 old_shape 和 new_shape 是列表类型
    assert isinstance(old_shape, list) and isinstance(new_shape, list)
    # 如果旧形状与新形状相同，则直接返回 value
    if old_shape == new_shape:
        return value
    # 如果 new_shape 中不是所有维度都是 "1"，并且不等于 old_shape，则执行下面的逻辑
    if [s for s in new_shape if s != "1"] != old_shape:
        return f"tl.reshape({value}, [{', '.join(new_shape)}])"
    # 使用 [:, None] 语法进行重写，这种语法比较稳定
    idx = 0
    expand = []
    # 遍历 new_shape，根据 old_shape 来扩展维度
    for size in new_shape:
        if idx < len(old_shape) and size == old_shape[idx]:
            expand.append(":")
            idx += 1
        else:
            assert size == "1"
            expand.append("None")
    # 断言扩展的维度数量等于 old_shape 的长度
    assert idx == len(old_shape)
    # 返回重塑后的表达式
    return f"{value}[{', '.join(expand)}]"


# 注意：继承自 PythonPrinter 可能存在潜在的风险，因为 Triton 对一些操作符的实现方式
# 与 Python 语义不一致（与 C 语义一致）。我们必须重写所有这些操作符，否则可能导致潜在的正确性问题。
class TritonPrinter(PythonPrinter):
    # 重写 _print_TruncToInt 方法，将其转换为 Triton 特定的格式
    def _print_TruncToInt(self, expr):
        assert len(expr.args) == 1
        return (
            f"libdevice.trunc({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    # 重写 _print_ToFloat 方法，将其转换为 Triton 特定的格式
    def _print_ToFloat(self, expr):
        assert len(expr.args) == 1
        return f"{self.paren(self._print(expr.args[0]))}.to(tl.float64)"

    # TODO: 下面的几个方法重写存在问题，具体可以参考对应的 issue 进行修复
    # 但是对于 Sympy 表达式，大多数情况下应该能正常工作，因为我们通常不处理负数除法和较大的整数除法。

    # 重写 _print_PythonMod 方法，将其转换为 Triton 特定的格式
    def _print_PythonMod(self, expr):
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    # 重写 _print_FloorDiv 方法，将其转换为 Triton 特定的格式
    def _print_FloorDiv(self, expr):
        assert expr.is_integer
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} // {div})"

    # 重写 _print_IntTrueDiv 方法，将其转换为 Triton 特定的格式
    def _print_IntTrueDiv(self, expr):
        lhs, rhs = expr.args
        return f"{self.paren(self._print(lhs))} / {self.paren(self._print(rhs))}"

    # 重写 _print_floor 方法，将其转换为 Triton 特定的格式
    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return (
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )
    # 打印一个表达式的 floor 函数调用，将其结果转换为指定的索引数据类型
    def _print_FloorToInt(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 floor 函数
        return (
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    # 打印一个表达式的 ceiling 函数调用，将其结果转换为指定的索引数据类型
    def _print_ceiling(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 ceil 函数
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    # 打印一个表达式的 CeilToInt 函数调用，将其结果转换为指定的索引数据类型
    def _print_CeilToInt(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 ceil 函数
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    # 辅助函数，打印一个表达式的 sqrt 函数调用，将其结果转换为 tl.float32 类型
    def _helper_sqrt(self, expr):
        # 构造函数调用字符串，使用 libdevice 库的 sqrt 函数
        return f"libdevice.sqrt({self._print(expr)}.to(tl.float32))"

    # 打印一个表达式的 Where 函数调用，参数包括条件 c、真值 p 和假值 q
    def _print_Where(self, expr):
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        # 构造函数调用字符串，使用 tl.where 函数
        return f"tl.where({c}, {p}, {q})"

    # 打印一个表达式的最大值或最小值函数调用，根据 cmp 参数指定大于或小于比较
    def _print_min_max_helper(self, expr: sympy.Expr, cmp: str) -> str:
        """
        Helper for max/min code genereration.
        cmp: > or <
        """
        # 确定表达式参数个数
        nargs = len(expr.args)
        # 如果表达式只有一个参数，直接打印该参数
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        # 分割参数列表
        mid = len(expr.args) // 2
        cls = type(expr)
        a = self._print(cls(*expr.args[:mid]))
        b = self._print(cls(*expr.args[mid:]))

        # 使用宏来传播 constexprs
        # 构造条件表达式字符串，用于求解最大值或最小值
        a, b = tuple(f"({x})" for x in (a, b))
        assert cmp in {">", "<"}, f"Unexpected comparator: '{cmp}'"
        return f"({a} * ({a} {cmp}= {b}) + {b} * ({b} {cmp} {a}))"

    # 打印一个表达式的最小值函数调用
    def _print_Min(self, expr):
        # 使用 _print_min_max_helper 辅助函数，指定比较操作为 "<"
        return self._print_min_max_helper(expr, "<")

    # 打印一个表达式的最大值函数调用
    def _print_Max(self, expr):
        # 使用 _print_min_max_helper 辅助函数，指定比较操作为 ">"
        return self._print_min_max_helper(expr, ">")

    # 打印一个表达式的绝对值函数调用
    def _print_Abs(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 tl_math 库的 abs 函数
        return f"tl_math.abs({self._print(expr.args[0])})"

    # 打印一个表达式的 cos 函数调用，参数类型转换为 tl.float32
    def _print_OpaqueUnaryFn_cos(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 cos 函数
        return f"libdevice.cos(({self._print(expr.args[0])}).to(tl.float32))"

    # 打印一个表达式的 cosh 函数调用，参数类型转换为 tl.float32
    def _print_OpaqueUnaryFn_cosh(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 cosh 函数
        return f"libdevice.cosh(({self._print(expr.args[0])}).to(tl.float32))"

    # 打印一个表达式的 acos 函数调用，参数类型转换为 tl.float32
    def _print_OpaqueUnaryFn_acos(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 acos 函数
        return f"libdevice.acos(({self._print(expr.args[0])}).to(tl.float32))"

    # 打印一个表达式的 sin 函数调用，参数类型转换为 tl.float32
    def _print_OpaqueUnaryFn_sin(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 sin 函数
        return f"libdevice.sin(({self._print(expr.args[0])}).to(tl.float32))"

    # 打印一个表达式的 sinh 函数调用，参数类型转换为 tl.float32
    def _print_OpaqueUnaryFn_sinh(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 sinh 函数
        return f"libdevice.sinh(({self._print(expr.args[0])}).to(tl.float32))"

    # 打印一个表达式的 asin 函数调用，参数类型转换为 tl.float32
    def _print_OpaqueUnaryFn_asin(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 asin 函数
        return f"libdevice.asin(({self._print(expr.args[0])}).to(tl.float32))"

    # 打印一个表达式的 tan 函数调用，参数类型转换为 tl.float32
    def _print_OpaqueUnaryFn_tan(self, expr):
        # 确保表达式只有一个参数
        assert len(expr.args) == 1
        # 构造函数调用字符串，使用 libdevice 库的 tan 函数
        return f"libdevice.tan(({self._print(expr.args[0])}).to(tl.float32))"
    # 定义一个方法，用于打印表达式中的 tanh 函数
    def _print_OpaqueUnaryFn_tanh(self, expr):
        # 断言表达式参数数量为1
        assert len(expr.args) == 1
        # 返回 tanh 函数的字符串表示，要求参数转换为 float32 类型
        return f"libdevice.tanh(({self._print(expr.args[0])}).to(tl.float32))"

    # 定义一个方法，用于打印表达式中的 atan 函数
    def _print_OpaqueUnaryFn_atan(self, expr):
        # 断言表达式参数数量为1
        assert len(expr.args) == 1
        # 返回 atan 函数的字符串表示，要求参数转换为 float32 类型
        return f"libdevice.atan(({self._print(expr.args[0])}).to(tl.float32))"

    # 定义一个方法，用于打印表达式中的 RoundToInt 函数
    def _print_RoundToInt(self, expr):
        # 断言表达式参数数量为1
        assert len(expr.args) == 1
        # 返回 llrint 函数的字符串表示
        return f"libdevice.llrint({self._print(expr.args[0])})"

    # 定义一个方法，用于打印表达式中的 RoundDecimal 函数
    def _print_RoundDecimal(self, expr):
        # 断言表达式参数数量为2
        assert len(expr.args) == 2
        # 解包参数 number 和 ndigits
        number, ndigits = expr.args
        # 如果 number 是整数，则 ndigits 必须为非负数，否则抛出 ValueError 异常
        if number.is_integer:
            assert ndigits < 0
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        # 返回 nearbyint 函数的字符串表示，处理 number 乘以 10 的 ndigits 次方，再除以 10 的 -ndigits 次方
        return f"libdevice.nearbyint(1e{ndigits} * {self.paren(self._print(number))}) * 1e{-ndigits}"
texpr = TritonPrinter().doprint
# 将 TritonPrinter 实例化后的 doprint 方法赋值给 texpr，用于表达式打印

def triton_compute_type(dtype):
    # 从 dtype 中获取 Triton 类型名称
    triton_type_name = str(dtype).split(".")[-1]
    # 根据 Triton 类型名称映射到 Triton 的数据类型
    if triton_type_name == "bool":
        triton_type_name = "int1"
    elif triton_type_name in ("float16", "bfloat16"):
        # 对于 float16 和 bfloat16 类型，使用 float32 进行计算
        triton_type_name = "float32"
    elif triton_type_name == "float8_e4m3fn":
        triton_type_name = "float8e4nv"
    elif triton_type_name == "float8_e5m2":
        triton_type_name = "float8e5"
    elif triton_type_name == "float8_e4m3fnuz":
        triton_type_name = "float8e4b8"
    elif triton_type_name == "float8_e5m2fnuz":
        triton_type_name = "float8e5b16"
    # 返回 Triton 的类型名称
    return f"tl.{triton_type_name}"

def triton_store_type(dtype):
    # 从 dtype 中获取 Triton 类型名称
    triton_type_name = str(dtype).split(".")[-1]
    # 根据 Triton 类型名称映射到 Triton 的存储类型
    if triton_type_name == "bool":
        triton_type_name = "int8"
    elif triton_type_name == "float8_e4m3fn":
        triton_type_name = "float8e4nv"
    elif triton_type_name == "float8_e5m2":
        triton_type_name = "float8e5"
    # 返回 Triton 的存储类型名称
    return f"tl.{triton_type_name}"

def triton_acc_type(dtype):
    # 如果 dtype 是有符号整数类型，则根据其位数返回相应的 Triton 类型
    if is_integer_dtype(dtype) and dtype.is_signed:
        nbits = 64 if dtype == torch.int64 else 32
        return f"tl.int{nbits}"
    # 否则调用 triton_compute_type 函数获取 Triton 类型
    return triton_compute_type(dtype)

class TritonCSEVariable(CSEVariable):
    def __init__(self, name, bounds: ValueRanges[Any]):
        super().__init__(name, bounds)
        # 用于跟踪变量在间接索引时需要的掩码
        self.mask_vars: Set[str] = set()

    def update_on_args(self, name, args, kwargs):
        # 遍历参数列表，更新变量的掩码变量集合
        for arg in args:
            if isinstance(arg, TritonCSEVariable):
                self.mask_vars.update(arg.mask_vars)
            elif isinstance(arg, sympy.Symbol) and arg.name[0] in "xyr":
                # 当索引变量用于计算间接读取索引时，关联的读取操作需要使用掩码
                self.mask_vars.update({f"{arg.name[0]}mask"})

class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    # 将输入张量 x 转换为指定的数据类型 dtype
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None):
        # 获取每个线程的最小元素数量，用于确定数据类型转换时的要求
        def _get_min_elements_per_thread(
            src_dtype: torch.dtype, dst_dtype: torch.dtype
        ) -> int:
            if src_dtype == dst_dtype:
                # 不需要数据类型转换，不需要最小元素数量要求
                return 0

            # fp8 数据类型转换需要最小元素数量要求
            # 参考 Triton 实现：https://github.com/openai/triton/blob/10f59d8ce04052521c1bc0cb3a3f8b98918fc7e3/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L10
            fp8_dtypes = {
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            }
            # Triton 不支持 fp8_e4m3 和 fp8_e5m2 之间的类型转换
            assert not (
                src_dtype in fp8_dtypes
                and dst_dtype in fp8_dtypes
                and src_dtype != dst_dtype
            ), "Conversions between float8_e5m2 and float8_e4m3fn is not supported!"
            if src_dtype == torch.float8_e5m2 or dst_dtype == torch.float8_e5m2:
                return 4
            if src_dtype == torch.float8_e4m3fn or dst_dtype == torch.float8_e4m3fn:
                return 2
            # 不需要最小元素数量要求
            return 0

        if src_dtype is not None:
            # 如果设置了 dtype 和 src_dtype，则用于 torch to(dtype=dtype) 操作
            # 如果在同一个内核中有多个 fp8 转换，则取最大的最小元素数量
            V.kernel.min_elem_per_thread = max(
                _get_min_elements_per_thread(src_dtype, dtype),
                V.kernel.min_elem_per_thread,
            )

        if dtype == torch.bool:
            return f"({x} != 0)"
        elif dtype == torch.uint8:
            # 解决 llvm 无符号整数转换语义问题，将负值转换为 0
            return f"{x}.to(tl.int8).to(tl.uint8)"
        return f"{x}.to({triton_compute_type(dtype)})"

    @staticmethod
    # 将输入张量 x 按位转换为指定的数据类型 dtype
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        triton_dtype = triton_compute_type(dtype)
        # 可能将 float16 或 bfloat16 提升为 float32，导致 dtype 的位宽与输入张量不同
        # 在这种情况下，需要将输入张量转换为其 src_type，执行位转换，然后将位转换后的张量转换回 float 以确保使用正确精度的值
        if src_dtype in (torch.float16, torch.bfloat16):
            triton_src_dtype = str(src_dtype).split(".")[-1]
            cast_x = f"{x}.to(tl.{triton_src_dtype})"
            cast_x = f"{cast_x}.to({triton_dtype}, bitcast=True)"
            return f"{cast_x}.to(tl.float32)"
        else:
            return f"{x}.to({triton_dtype}, bitcast=True)"
    @staticmethod
    def _shaped_constant(value, dtype, shape):
        # 将Python类型的值转换为Triton类型的常量表示
        type_ = torch._prims_common.dtype_to_type(dtype)
        # 获取Triton计算类型
        triton_val = constant_repr(type_(value))
        triton_type = triton_compute_type(dtype)

        if triton_type == "tl.float32":
            # 在Triton中，浮点常量始终为f32类型
            return triton_val

        # 注意：使用张量以获取预期的类型。
        # 否则，例如，float64常量会被截断为float32。
        return f"tl.full({shape}, {triton_val}, {triton_type})"

    @classmethod
    def constant(cls, value, dtype):
        # 返回形状为空列表的常量表达式
        return cls._shaped_constant(value, dtype, shape=[])

    @staticmethod
    def abs(x):
        # 返回x的绝对值的Triton库函数表示
        return f"tl_math.abs({x})"

    @staticmethod
    def libdevice_abs(x):
        # 返回x的绝对值的libdevice库函数表示
        return f"libdevice.abs({x})"

    @staticmethod
    def exp(x):
        # 返回x的指数函数的Triton库函数表示
        return f"tl_math.exp({x})"

    @staticmethod
    def libdevice_exp(x):
        # 返回x的指数函数的libdevice库函数表示
        return f"libdevice.exp({x})"

    @staticmethod
    def exp2(x):
        # 返回x的2次幂函数的libdevice库函数表示
        return f"libdevice.exp2({x})"

    @staticmethod
    def expm1(x):
        # 返回x的exp(x)-1函数的libdevice库函数表示
        return f"libdevice.expm1({x})"

    @staticmethod
    def sqrt(x):
        # 返回x的平方根函数的libdevice库函数表示
        return f"libdevice.sqrt({x})"

    @staticmethod
    def libdevice_sqrt(x):
        # 返回x的平方根函数的libdevice库函数表示
        return f"libdevice.sqrt({x})"

    @staticmethod
    def relu(x):
        bug = config.triton.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            # 如果配置中设置为"compile_error"，则返回编译错误信息
            return "compile error!"
        elif bug == "runtime_error":
            # 注意：只要输入不全为零，就会触发运行时错误
            return f'triton_helpers.device_assert_then({x} == 0, "injected assert fail", {x})'
        elif bug == "accuracy":
            # 返回x加1，用于测试精度
            return f"{x} + 1"
        elif bug is None:
            # 如果没有指定错误，返回0和x中的较大值
            return ops.maximum(ops.constant(0, torch.int32), x)
        else:
            # 抛出未识别的配置错误
            raise AssertionError(
                f"unrecognized config triton.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def minimum(a, b):
        # 返回a和b的最小值的Triton帮助函数表示
        return f"triton_helpers.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        # 返回a和b的最大值的Triton帮助函数表示
        return f"triton_helpers.maximum({a}, {b})"

    @staticmethod
    def where(a, b, c):
        # 返回根据条件a选择的值的Triton库函数表示
        return f"tl.where({a}, {b}, {c})"

    @staticmethod
    def cos(x):
        # 返回x的余弦函数的Triton库函数表示
        return f"tl_math.cos({x})"

    @staticmethod
    def libdevice_cos(x):
        # 返回x的余弦函数的libdevice库函数表示
        return f"libdevice.cos({x})"

    @staticmethod
    def sin(x):
        # 返回x的正弦函数的Triton库函数表示
        return f"tl_math.sin({x})"

    @staticmethod
    def libdevice_sin(x):
        # 返回x的正弦函数的libdevice库函数表示
        return f"libdevice.sin({x})"

    @classmethod
    def index_expr(cls, expr, dtype):
        # 抛出未实现错误，因为index_expr函数未在内核外实现
        raise NotImplementedError("ops.index_expr not implemented outside a kernel")

    @staticmethod
    def masked(mask, body, other):
        # 抛出未实现错误，因为masked函数未在内核外实现
        raise NotImplementedError("ops.masked not implemented outside a kernel")

    @staticmethod
    def lgamma(x):
        # 返回x的gamma函数的libdevice库函数表示
        return f"libdevice.lgamma({x})"

    @staticmethod
    def erf(x):
        # 返回x的误差函数的libdevice库函数表示
        return f"libdevice.erf({x})"
    @staticmethod
    def cosh(x):
        # 返回 x 的双曲余弦值，使用 libdevice 库函数
        return f"libdevice.cosh({x})"

    @staticmethod
    def sinh(x):
        # 返回 x 的双曲正弦值，使用 libdevice 库函数
        return f"libdevice.sinh({x})"

    @staticmethod
    def acos(x):
        # 返回 x 的反余弦值，使用 libdevice 库函数
        return f"libdevice.acos({x})"

    @staticmethod
    def acosh(x):
        # 返回 x 的反双曲余弦值，使用 libdevice 库函数
        return f"libdevice.acosh({x})"

    @staticmethod
    def asin(x):
        # 返回 x 的反正弦值，使用 libdevice 库函数
        return f"libdevice.asin({x})"

    @staticmethod
    def asinh(x):
        # 返回 x 的反双曲正弦值，使用 libdevice 库函数
        return f"libdevice.asinh({x})"

    @staticmethod
    def atan2(x, y):
        # 返回 arctan(x/y) 的值，使用 libdevice 库函数
        return f"libdevice.atan2({x}, {y})"

    @staticmethod
    def atan(x):
        # 返回 x 的反正切值，使用 libdevice 库函数
        return f"libdevice.atan({x})"

    @staticmethod
    def atanh(x):
        # 返回 x 的反双曲正切值，使用 libdevice 库函数
        return f"libdevice.atanh({x})"

    @staticmethod
    def copysign(x, y):
        # 返回 x 的符号与 y 相同的值，使用 libdevice 库函数
        return f"libdevice.copysign({x}, {y})"

    @staticmethod
    def erfc(x):
        # 返回 x 的互补误差函数值，使用 libdevice 库函数
        return f"libdevice.erfc({x})"

    @staticmethod
    def erfinv(x):
        # 返回 x 的逆误差函数值，使用 libdevice 库函数
        return f"libdevice.erfinv({x})"

    @staticmethod
    def hypot(x, y):
        # 返回 sqrt(x*x + y*y) 的值，使用 libdevice 库函数
        return f"libdevice.hypot({x}, {y})"

    @staticmethod
    def log10(x):
        # 返回 x 的以 10 为底的对数值，使用 libdevice 库函数
        return f"libdevice.log10({x})"

    @staticmethod
    def log2(x):
        # 返回 x 的以 2 为底的对数值，使用 libdevice 库函数
        return f"libdevice.log2({x})"

    @staticmethod
    def nextafter(x, y):
        # 返回 x 与 y 之间的下一个浮点数，使用 libdevice 库函数
        return f"libdevice.nextafter({x}, {y})"

    @staticmethod
    def logical_and(a, b):
        # 返回 a 与 b 的逻辑与操作结果
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        # 返回 a 的逻辑非操作结果
        return f"{a} == 0"

    @staticmethod
    def logical_or(a, b):
        # 返回 a 与 b 的逻辑或操作结果
        return f"{a} | {b}"

    @staticmethod
    def logical_xor(a, b):
        # 返回 a 与 b 的逻辑异或操作结果
        return f"({a} ^ {b})"

    @staticmethod
    def bitwise_and(a, b):
        # 返回 a 与 b 的按位与操作结果
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        # 返回 a 的按位取反操作结果
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        # 返回 a 与 b 的按位或操作结果
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        # 返回 a 与 b 的按位异或操作结果
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        # 返回 a 向左位移 b 位的结果
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        # 返回 a 向右位移 b 位的结果
        return f"{a} >> {b}"

    @staticmethod
    def rand(seed, offset):
        # 使用 seed 和 offset 生成随机数，使用 tl.rand 函数
        offset = f"({offset}).to(tl.uint32)"
        return f"tl.rand({seed}, {offset})"

    @staticmethod
    def randn(seed, offset):
        # 使用 seed 和 offset 生成正态分布的随机数，使用 tl.randn 函数
        offset = f"({offset}).to(tl.uint32)"
        return f"tl.randn({seed}, {offset})"

    @staticmethod
    def randint64(seed, offset, low, high):
        # 使用 seed 和 offset 生成指定范围内的 64 位随机整数，使用 triton_helpers.randint64 函数
        offset = f"({offset}).to(tl.uint32)"
        return f"triton_helpers.randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    def load_seed(name, offset):
        # 抛出未实现错误，ops.load_seed 在内核外部未实现
        raise NotImplementedError("ops.load_seed not implemented outside a kernel")

    @staticmethod
    def rsqrt(x):
        # 返回 x 的平方根的倒数，使用 libdevice 库函数
        return f"libdevice.rsqrt({x})"

    @staticmethod
    def log1p(x):
        # 返回 log(1 + x) 的值，使用 libdevice 库函数
        return f"libdevice.log1p({x})"

    @staticmethod
    def tan(x):
        # 返回 x 的正切值，使用 libdevice 库函数
        return f"libdevice.tan({x})"

    @staticmethod
    def tanh(x):
        # 返回 x 的双曲正切值，使用 libdevice 库函数
        return f"libdevice.tanh({x})"

    @staticmethod
    def sigmoid(x):
        # 返回 x 的 sigmoid 函数值，使用 tl.sigmoid 函数
        return f"tl.sigmoid({x})"
    @staticmethod
    def signbit(x):
        # 返回一个字符串，表示是否为负数（注意：这里对于浮点数 -0.0 的处理是错误的）
        return f"libdevice.signbit({x}) if ({x}).dtype is tl.float32 else {x} < 0"

    @staticmethod
    def fmod(a, b):
        # 返回一个字符串，表示调用 libdevice.fmod 函数的结果
        return f"libdevice.fmod({a}, {b})"

    @staticmethod
    def pow(a, b):
        # 返回一个字符串，表示调用 libdevice.pow 函数的结果
        return f"libdevice.pow({a}, {b})"

    @staticmethod
    def log(x):
        # 返回一个字符串，表示调用 tl_math.log 函数的结果
        return f"tl_math.log({x})"

    @staticmethod
    def libdevice_log(x):
        # 返回一个字符串，表示调用 libdevice.log 函数的结果
        return f"libdevice.log({x})"

    @staticmethod
    def isinf(x):
        # 返回一个字符串，表示调用 libdevice.isinf 函数的结果，转换为 tl.int1 类型
        return f"libdevice.isinf({x}).to(tl.int1)"

    @staticmethod
    def isnan(x):
        # 返回一个字符串，表示调用 libdevice.isnan 函数的结果，转换为 tl.int1 类型
        return f"libdevice.isnan({x}).to(tl.int1)"

    @staticmethod
    def round(x):
        # 返回一个字符串，表示调用 libdevice.nearbyint 函数的结果
        return f"libdevice.nearbyint({x})"

    @staticmethod
    def floor(x):
        # 返回一个字符串，表示调用 libdevice.floor 函数的结果
        return f"libdevice.floor({x})"

    @staticmethod
    def floordiv(a, b):
        # 返回一个字符串，表示根据特定规则进行整数除法（向下取整），考虑符号和余数情况
        # 注意：这里的 // 在 triton 中表现为 truncdiv 而不是 floordiv
        quot = f"{a} // {b}"
        rem = f"{a} % {b}"
        return f"tl.where(({a} < 0) != ({b} < 0), tl.where({rem} != 0, {quot} - 1, {quot}), {quot})"

    @staticmethod
    def sign(x):
        # 返回一个字符串，表示对 x 的符号进行处理后的结果
        z = ops.constant(0, torch.int32)
        left = ops.to_dtype((ops.lt(z, x)), torch.int8)
        right = ops.to_dtype((ops.lt(x, z)), torch.int8)
        sub = ops.sub(left, right)
        return f"{sub}.to({x}.dtype)"

    @staticmethod
    def trunc(x):
        # 返回一个字符串，表示调用 libdevice.trunc 函数的结果
        return f"libdevice.trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # 返回一个字符串，表示根据特定规则进行整数除法（截断除法），考虑符号情况
        # 注意：这里的 // 在 triton 中表现为 truncdiv 而不是 floordiv
        return f"{a} // {b}"

    @staticmethod
    def ceil(x):
        # 返回一个字符串，表示调用 libdevice.ceil 函数的结果
        return f"libdevice.ceil({x})"
# 初始化 TritonOverrides 中的点操作重写，使用 "triton" 作为参数
TritonOverrides._initialize_pointwise_overrides("triton")

# 用于使用 mypy 检查协议是否正确实现
def _typecheck_TritonOverrides(h: TritonOverrides) -> OpsHandler[str]:
    return h

class TritonKernelOverrides(TritonOverrides):
    """在 TritonKernel 中将元素操作映射到 Triton

    与 TritonOverrides 不同，这些假设代码将插入到主 Triton 内核的主体中，
    因此可能使用已在当前作用域中定义的索引和掩码变量。
    """

    @classmethod
    def constant(cls, value, dtype):
        # 注意：不能使用 shape=[]，因为 triton-rocm 不支持
        # 可以使用 shape=[1] 代替，但正确的 ndim 可以避免 triton IR 中出现额外的 `tt.expand_dim` 操作
        ndim = V.kernel.triton_tensor_ndim()
        shape = [1] * ndim
        return cls._shaped_constant(value, dtype, shape=shape)

    @classmethod
    def index_expr(cls, expr, dtype):
        # 获取索引表达式的索引选项
        indexing = V.kernel.indexing(expr, block_ptr=False)
        assert isinstance(indexing, IndexingOptions)
        # 生成变量，并确保其是正确的数据类型
        var = V.kernel.cse.generate(
            V.kernel.compute, indexing.index_str, bounds=get_bounds_index_expr(expr)
        )

        if dtype not in {torch.int32, torch.int64}:
            var = V.kernel.cse.generate(V.kernel.compute, cls.to_dtype(var, dtype))
        # 设置掩码变量
        var.mask_vars = indexing.mask_vars
        return var

    @staticmethod
    def masked(mask, body, other):
        if mask is not None and torch.version.hip is not None:
            # 生成掩码加载
            mask = V.kernel.cse.generate(
                V.kernel.compute,
                f"{mask}.to(tl.int1)",
            )

        # 查找 body 图中的输出节点
        nodes = body.graph.find_nodes(op="output")
        assert nodes, "graph for body does not contain an output"

        need_where = False
        for node in nodes:
            for arg in node.args:
                if arg.target != "load" or V.graph.is_unspec_arg(arg.args[0]):
                    need_where = True

        value = None if need_where else other
        # 使用新的掩码加载
        with V.kernel.mask_loads(mask, value=value) as new_mask:
            result = body()

        if need_where:
            # 如果需要 where 操作，根据结果和其他值生成新的 CSE 变量
            if result.bounds.is_bool:
                other = bool(other)
            other = V.kernel.cse.generate(
                V.kernel.compute,
                f"tl.full({result}.shape, {constant_repr(other)}, {result}.dtype)",
                bounds=ValueRanges.wrap(other),
            )
            ret = ops.where(new_mask, result, other)
        else:
            ret = result

        # 丢弃掩码变量
        ret.mask_vars.discard(new_mask)
        return ret

    @staticmethod
    def load_seed(name, offset):
        # 加载种子值
        var = V.kernel.args.input(name)
        return (
            f"tl.load({var} + {V.kernel.args.seed_offset('load_seed_offset', offset)})"
        )
    # 定义一个函数 frexp，用于返回浮点数 x 的尾数和指数的元组
    def frexp(x):
        # 构建缓存键，用于存储函数调用结果以提高性能
        cache_key = f"frexp({x})"
        # 如果缓存键已存在于 V.kernel.cse.cache 中，则直接返回缓存的结果
        if cache_key in V.kernel.cse.cache:
            return V.kernel.cse.cache[cache_key]

        # 生成新的变量名，用于存储尾数和指数
        mantissa = V.kernel.cse.newvar()
        exponent = V.kernel.cse.newvar()
        
        # 在 kernel 上下文中，调用 triton_helpers.frexp 函数，并将结果分配给 mantissa 和 exponent
        V.kernel.compute.writeline(
            f"{mantissa}, {exponent} = triton_helpers.frexp({x})"
        )
        
        # 将计算结果存入缓存，以便下次使用
        V.kernel.cse.cache[cache_key] = (mantissa, exponent)
        
        # 返回尾数和指数的元组
        return (mantissa, exponent)
# 使用 mypy 检查协议是否正确实现
def _typecheck_TritonKernelOverrides(h: TritonKernelOverrides) -> OpsHandler[str]:
    return h


class HelperFunctions:
    """一个有序的辅助函数集合。"""

    _templates_seen: Dict[str, str]  # 模板代码到函数名的映射
    finalized_helpers: List[str]

    def __init__(self):
        self._templates_seen = {}
        self.finalized_helpers = []

    def add(self, template_code: str, *, base_name="_triton_helper_fn") -> str:
        """接受一个带有函数名格式说明符的函数定义，例如

            @triton.jit
            def {name}(arg0, arg1):
                return arg0 + arg1

        我们将模板代码添加到函数集合中，并返回分配给该函数的名称。

        """
        existing_name = self._templates_seen.get(template_code)
        if existing_name is not None:
            # 不重复添加现有的辅助函数
            return existing_name

        name = f"{base_name}{len(self.finalized_helpers)}"
        self._templates_seen[template_code] = name
        self.finalized_helpers.append(template_code.format(name=name))
        return name

    def __iter__(self):
        return iter(self.finalized_helpers)

    def __getitem__(self, idx):
        return self.finalized_helpers[idx]


@dataclasses.dataclass
class BlockParameters:
    """
    代表 ND 块维度的类，用于块指针分析。
    """

    shape: List[sympy.Expr] = dataclasses.field(default_factory=list)
    block_shape: List[sympy.Expr] = dataclasses.field(default_factory=list)
    strides: List[sympy.Expr] = dataclasses.field(default_factory=list)
    offsets: List[sympy.Expr] = dataclasses.field(default_factory=list)

    def __add__(self, other: BlockParameters) -> BlockParameters:
        """
        连接块参数。
        """
        cls = type(self)
        a, b = tuple(dataclasses.asdict(x) for x in (self, other))
        return cls(**{key: a[key] + b[key] for key in a})


class TritonKernel(SIMDKernel):
    overrides = TritonKernelOverrides  # type: ignore[assignment]
    helper_functions: HelperFunctions
    kexpr: Callable[[sympy.Expr], str] = texpr
    allow_block_ptr = True

    def __init__(
        self,
        *groups,
        index_dtype: str,
        mutations: Optional[Set[str]] = None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        min_elem_per_thread=0,
        override_persistent_reduction=None,
        ):
            super().__init__(
                *groups,
                index_dtype=index_dtype,
                mutations=mutations,
                reduction_hint=reduction_hint,
                pid_cache=pid_cache,
                override_persistent_reduction=override_persistent_reduction,
            )
            self.suffix: IndentedBuffer = IndentedBuffer()  # type: ignore[assignment]
            self.outside_loop_vars: Set[Any] = set()
            self.min_elem_per_thread = min_elem_per_thread
            self.block_ptr_id = itertools.count()
            self.helper_functions = HelperFunctions()

            # A set of autotuning hints to pass as part of triton_meta
            self.autotune_hints: Set[AutotuneHint] = set()
            self.triton_meta: Optional[Dict[str, object]] = None

            # Generate the range tree for code generation
            self.codegen_range_tree()

        def _get_symt(self, tree: IterationRangesEntry) -> SymT:
            # Create a mapping from prefix strings to symbolic representations
            prefix_to_symt = {prefix: symt for symt, prefix in prefix_str.items()}
            return prefix_to_symt[tree.prefix]

        def _get_block_size(self, tree: IterationRangesEntry) -> sympy.Symbol:
            # Retrieve the block size symbol for a given tree entry
            return block_sizes[self._get_symt(tree)]

        def _get_block_offset(self, tree: IterationRangesEntry) -> sympy.Symbol:
            # Retrieve the block offset symbol for a given tree entry
            return block_offsets[self._get_symt(tree)]

        def _max_block_size(self, tree: IterationRangesEntry) -> int:
            # Retrieve the maximum block size for a given tree entry
            return TRITON_MAX_BLOCK[tree.prefix.upper()]

        def codegen_range_tree(self):
            # Generate code for iteration ranges within the range_trees collection
            for tree in self.range_trees:
                # If the tree is not a loop, generate header information for iteration ranges
                if not tree.is_loop:
                    self.iteration_ranges_codegen_header(tree, self.body)
            # Check if the last tree is a loop and if we are inside a reduction
            if self.inside_reduction and self.range_trees[-1].is_loop:
                # Append a workaround statement for a specific issue related to ranges
                self.body.writeline(
                    f"rbase = {self.iteration_ranges_ranges_code(self.range_trees[-1])}"
                )

        def need_numel_args(self):
            r"""
            Indicate whether we need to provide numel as arguments for the generated
            kernel calls in the benchmark.

            Should be true for pointwise/reduction kernels but false for triton
            matmul kernels.
            """
            return True
    # 判断是否应该使用持久化的归约策略，设置 self.persistent_reduction 并添加必要的保护条件
    def should_use_persistent_reduction(self) -> bool:
        if not (self.inside_reduction and config.triton.persistent_reductions):
            # 如果不在归约内部或者 Triton 的持久化归约未启用，则返回 False
            return False
        threshold = {
            ReductionHint.INNER: 1024,
        }.get(self.reduction_hint, 64)
        # 如果启用了 multi_kernel，增加阈值的值以进行更激进的持久化归约
        if config.triton.multi_kernel:
            threshold *= 16
        last_numel = self.numels[-1]
        # 返回最后一个元素的大小是否静态小于阈值
        return V.graph.sizevars.statically_known_leq(last_numel, threshold)  # type: ignore[arg-types]

    # 判断是否需要排除 X 维度的索引
    def want_no_x_dim(self):
        return (
            self.reduction_hint == ReductionHint.INNER
            and self.persistent_reduction
            and len(self.numels) == 2
            and V.graph.sizevars.statically_known_geq(self.numels[-1], 256)  # type: ignore[arg-types]
        )

    # 返回断言函数的名称字符串
    @property
    def assert_function(self) -> str:
        return "tl.device_assert"

    # 处理索引操作，返回块指针、延迟行和其他选项的元组
    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape=None,
        dense_indexing=False,
        override_mask=None,
        block_ptr=False,
    ):
        # 生成块指针的代码
        def codegen_block_ptr(
            self, name: str, var: str, indexing: BlockPtrOptions, other=""
        ) -> Tuple[str, Optional[DeferredLine], str]:
            advance_block_ptr = None
            check = indexing.boundary_check()
            if not check:
                # 临时解决 https://github.com/openai/triton/issues/2813
                other = ""
            elif other:
                assert other == ", other=0.0"
                other = f", boundary_check={check!r}, padding_option='zero'"
            else:
                other = f", boundary_check={check!r}"
            if (
                self.inside_reduction
                and self.range_trees[-1].is_loop
                and indexing.has_rindex()
            ):
                # 如果在归约内部且最后一个范围树是循环并且索引具有 rindex
                block_ptr = f"block_ptr{next(self.block_ptr_id)}"
                self.body.writeline(
                    DeferredLine(
                        name, f"{block_ptr} = {indexing.format(var, roffset=False)}"
                    )
                )
                advance_block_ptr = DeferredLine(
                    name,
                    f"{block_ptr} = tl.advance({block_ptr}, {indexing.advance_roffset()})",
                )
            else:
                block_ptr = indexing.format(var)
            # 返回块指针、延迟行和其他选项的元组
            return block_ptr, advance_block_ptr, other
    # 定义一个方法，用于生成代码块存储行，包括名称、索引、块指针、数值等参数
    def codegen_block_ptr_store_line(self, name, indexing, block_ptr, value, other=""):
        # 对于块指针，不隐式进行广播
        value = (
            f"tl.broadcast_to({value}, {self.index_to_str(indexing.reshape_suffix)})"
        )
        # 去除大小为1的额外维度
        block_shape = [V.kernel.index_to_str(expr) for expr in indexing.block_shape]
        value = triton_reshape(value, indexing.reshape_suffix, block_shape)
        # 解决问题 https://github.com/openai/triton/issues/2814 的临时解决方案
        value = f"{value}.to({triton_store_type(V.graph.get_dtype(name))})"
        # 返回生成的代码行，包括块指针和数值等信息
        return f"tl.store({block_ptr}, {value}{other})"

    # 检查边界条件，包括表达式、大小、下界和上界等参数
    def check_bounds(
        self,
        expr: sympy.Expr,
        size: sympy.Expr,
        lower: bool,
        upper: bool,
    ):
        # 如果既没有下界也没有上界条件，直接返回
        if not (lower or upper):
            return

        # 断言表达式类型为 sympy.Expr
        assert isinstance(expr, sympy.Expr)
        # 获取索引选项，不包括块指针
        indexing = self.indexing(expr, block_ptr=False)
        assert isinstance(indexing, IndexingOptions)

        # 获取索引字符串、掩码字符串（如果有）、大小字符串（如果是上界）
        index_str = indexing.index_str
        mask_str = indexing.mask_str if indexing.has_mask() else None
        size_str = V.kernel.sexpr(self.rename_indexing(size)) if upper else None

        # 生成间接断言行，包括索引字符串、大小字符串、掩码字符串
        line = self.indirect_assert(
            index_str, "0" if lower else None, size_str, mask_str
        )

        # 检查表达式是否已经被包装，或者是否存在间接索引或变量
        indirect = self.is_indirect_indexing(expr) or any(
            isinstance(m, TritonCSEVariable) for m in indexing.mask_vars
        )
        # 获取加载缓冲区
        buffer = self.get_load_buffer(indexing)
        self.cse.generate(buffer, line, assignment=False)

    # 根据索引选项获取加载缓冲区
    def get_load_buffer(self, indexing):
        if indexing.has_indirect() or indexing.has_tmpmask():
            # 对于存在间接索引或临时掩码的情况，加载必须在计算掩码之后进行
            return self.compute
        elif (
            self.inside_reduction
            and self.range_trees[-1].is_loop
            and not indexing.has_rindex()
        ):
            # 可以将常见加载提升到约简循环之外，但不包括间接加载的情况
            return self.body
        else:
            # 默认情况下返回普通加载缓冲区
            return self.loads
    ) -> None:
        # 将输出结果写入指定名称的变量
        var = self.args.output(name)
        # 保存原始索引值
        original_index = index
        # 执行索引操作，使用稠密索引方式，如果模式为 None，则使用块指针索引
        indexing = self.indexing(index, dense_indexing=True, block_ptr=mode is None)

        # 针对 Triton 中的读后写入问题进行保护
        # 参考：https://github.com/openai/triton/issues/1615
        # Triton 中的这个 bug 意味着如果一个 load 操作在多个 warps 中广播，
        # 可能会看到稍后在 Triton 程序中发生的 store 操作的结果。
        # 解决方法是在存储之前添加一个 barrier，确保所有 warps 已经读取了数据。
        is_inplace = name in self.args.inplace_buffers
        is_broadcasted = self.is_broadcasted(original_index)
        if is_inplace and is_broadcasted:
            # 在存储之前添加一个 barrier
            self.stores.writeline(DeferredLine(name, "tl.debug_barrier()"))

        advance_block_ptr = None
        if isinstance(indexing, BlockPtrOptions):
            # 如果索引类型是 BlockPtrOptions，则生成块指针相关的代码
            block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                name, var, indexing
            )
            # 生成块指针存储行的代码
            line = self.codegen_block_ptr_store_line(
                name, indexing, block_ptr, value, other
            )
        elif mode is None:
            # 普通存储模式，生成对应的存储行代码
            line = f"tl.store({var} + ({indexing.index_str}), {value}, {indexing.mask_str})"
        elif mode == "atomic_add":
            # 原子加法模式，生成对应的原子加法存储行代码
            line = f"tl.atomic_add({var} + ({indexing.index_str}), {value}, {indexing.mask_str}, sem='relaxed')"
        else:
            # 抛出未实现的存储模式异常
            raise NotImplementedError(f"store mode={mode}")
        # 将生成的存储行代码写入存储对象
        self.stores.writeline(DeferredLine(name, line))
        if advance_block_ptr:
            # 如果存在块指针的进阶操作，则写入对应的存储行代码
            self.stores.writeline(advance_block_ptr)

        # 如果不在规约内部，将 value 添加到外部循环变量集合中
        if not self.inside_reduction:
            self.outside_loop_vars.add(value)
    ) -> CSEVariable:
        """
        See [Note: Inductor bucketize op]
        """

        # Triton performance for bucketize_binary_search is much better when the number
        # of threads equals the number of elements.
        # If we're trying to use a bucketize kernel, we should make sure that an
        # autotuning config with num_elements_per_warp=32 exists.
        # 将 num_elements_per_warp=32 的自动调优配置添加到自动调优提示中
        self.autotune_hints.add(AutotuneHint.ELEMENTS_PER_WARP_32)

        # 获取偏移指针
        offsets_ptr = self.args.input(offsets_name)
        # 获取密集大小的字符串表示
        block_size = self.dense_size_str()
        # 将偏移大小转换为字符串表示
        offsets_size_str = self.index_to_str(offsets_size)

        # 根据索引数据类型选择 Triton 的数据类型
        if indexing_dtype == torch.int32:
            triton_dtype = "tl.int32"
        elif indexing_dtype == torch.int64:
            triton_dtype = "tl.int64"
        else:
            # 不支持的索引数据类型，抛出异常
            raise NotImplementedError(
                "Bucketize only supports indexing with int32 and int64"
            )

        # 使用 CSE 模块生成 bucketize_binary_search 的计算表达式
        result = self.cse.generate(
            self.compute,
            f"triton_helpers.bucketize_binary_search({values}, {offsets_ptr}, {triton_dtype}, {right}, {offsets_size_str}, {block_size})",  # noqa: B950 line too long
        )

        # 返回生成的结果
        return result

    def reduction_resize(self, value):
        # 获取 Triton 张量的维数
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            # 如果维数为 1，则将值升级为张量
            return f"triton_helpers.promote_to_tensor({value})"

        # 否则，生成多维张量的字符串表示
        sizes = [":"] * ndims
        sizes[-1] = "None"
        return f"{value}[{', '.join(sizes)}]"

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    ):
        # 未提供具体实现的函数，只有函数签名，无需进一步注释
        pass

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
        # 确保当前处于缩减操作内部
        assert self.inside_reduction
        self.inside_reduction = False
        # 获取索引的表达式选项
        indexing = self.indexing(index, block_ptr=True)
        self.inside_reduction = True
        # 获取输出变量名对应的参数
        var = self.args.output(name)

        if isinstance(indexing, BlockPtrOptions):
            # 如果索引选项为块指针选项，生成相应的存储行代码
            self.suffix.writeline(
                DeferredLine(
                    name,
                    self.codegen_block_ptr_store_line(
                        name,
                        indexing,
                        indexing.format(var),
                        value,
                        f", boundary_check={indexing.boundary_check()!r}",
                    ),
                )
            )
        else:
            # 否则，索引选项为索引选项，生成相应的存储代码行
            assert isinstance(indexing, IndexingOptions)
            self.suffix.writeline(
                DeferredLine(
                    name,
                    f"tl.store({var} + ({indexing.index_str}), {value}, {indexing.mask_str})",
                )
            )
    def _lift_helper(self, fn, num_args) -> str:
        # Lift IR function for scan operations into a triton function
        # in the global namespace

        # 创建一个缓冲区来存储助手函数的代码
        helper = IndentedBuffer()
        helper.writeline("@triton.jit")

        # 生成参数元组列表，用于函数签名
        args = [tuple(f"arg{i}_{n}" for n in range(num_args)) for i in range(2)]
        signature = ", ".join(itertools.chain.from_iterable(args))

        # 定义函数的头部，包括函数名和参数签名
        helper.writeline(f"def {{name}}({signature}):")

        # 创建一个公共子表达式消除(CSE)的实例
        cse = CSE(prefix="", suffix="")

        # 创建一个模拟的 Triton 处理程序覆盖
        overrides = TritonOverrides(V.MockHandler())

        # 为了解决 Triton 缓存中的 bug，根据 fn 动态生成助手函数的名称
        helper_name = "_triton_helper_fn"

        # 创建一个 CSE 代理类，用于动态生成 CSE 变量
        class CSEProxy:
            def __getattr__(self, name: str) -> Callable[..., CSEVariable]:
                def inner(*args, **kwargs):
                    nonlocal helper_name
                    helper_name += f"_{name}"
                    return cse.generate(
                        helper,
                        getattr(overrides, name)(*args, **kwargs),
                    )
                return inner

        # 使用 CSEProxy 设置操作处理程序，并缩进 helper 的代码块
        with helper.indent(), V.set_ops_handler(CSEProxy()):
            # 调用输入的 fn 函数，获取其输出
            outputs = fn(*args)
            outputs = ", ".join(str(output) for output in outputs)
            helper.writeline(f"return {outputs}")

        # 将生成的助手函数添加到 helper_functions 中，并返回其名称
        return self.helper_functions.add(helper.getvalue(), base_name=helper_name)

    def scan(
        self,
        dtypes: Tuple[torch.dtype, ...],
        combine_fn: Callable[
            [Tuple[CSEVariable, ...], Tuple[CSEVariable, ...]], Tuple[CSEVariable, ...]
        ],
        values: Tuple[CSEVariable, ...],
    ):
        # 执行扫描操作的函数定义，接受数据类型、组合函数和值元组作为参数
        pass  # 实际实现被省略

    def sort(
        self,
        dtypes: Tuple[torch.dtype, ...],
        values: Tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ):
        # 执行排序操作的函数定义，接受数据类型、值元组以及排序稳定性和降序标志作为参数
        pass  # 实际实现被省略
    ) -> Tuple[CSEVariable, ...]:
        # 在内部约简操作中，确保处于约简状态
        assert self.inside_reduction
        # 从所有范围树中获取带有特定前缀的掩码名称集合
        masks = {f"{tree.prefix}mask" for tree in self.range_trees}
        # 使用获取的掩码集合来过滤掩码
        self.filter_masks(masks)
        # 对掩码集合进行排序
        masks = sorted(masks)
        # 确保不在 ops.masked 内部使用 ops.sort
        assert not self._load_mask, "ops.sort not supported inside ops.masked"
        # 确保只有在持久约简操作中支持 ops.sort
        assert self.persistent_reduction, "ops.sort is only supported in persistent reductions"
        # 获取最后一个范围树的前缀作为约简范围前缀
        reduction_range_prefix = self.range_trees[-1].prefix

        # 创建部分函数以便计算共享子表达式
        cse_compute = functools.partial(self.cse.generate, self.compute)
        # 计算张量的维度减去1
        dim = self.triton_tensor_ndim() - 1

        # 对值进行广播，生成广播后的值列表
        broadcasted_values = [
            cse_compute(f"tl.broadcast_to({value}, {self.dense_size_str()})")
            for value in values
        ]

        # 定义函数，将值转换为逗号分隔的字符串
        def csv(values):
            return " ".join(f"{value}," for value in values)

        # 定义函数，处理多个共享子表达式的生成和缓存
        def cse_multiple(line, n, masks):
            # 生成多个缓存键
            cache_keys = [f"{line}, {i}, {masks}" for i in range(n)]
            # 如果所有缓存键都在缓存中，则直接返回缓存结果
            if all(cache_key in self.cse.cache for cache_key in cache_keys):
                return [self.cse.cache[cache_key] for cache_key in cache_keys]
            # 否则生成新的结果变量列表，并将生成的表达式写入计算流
            result_vars = [self.cse.newvar() for _ in range(n)]
            self.compute.writeline(
                f"{csv(result_vars)} = {line}",
            )
            # 将生成的结果变量与缓存键关联，并设置掩码变量（如果有）
            for result_var, cache_key in zip(result_vars, cache_keys):
                if masks:
                    result_var.mask_vars = masks  # type: ignore[attr-defined]
                self.cse.cache[cache_key] = result_var
            return tuple(result_vars)

        # 确保最后一个范围树的前缀为 "r"
        assert self.range_trees[-1].prefix == "r"
        # 如果最后一个范围树没有常量掩码，则设置 rmask 为 "None"
        rmask = "None" if self._has_constant_mask(self.range_trees[-1]) else "rmask"

        # 如果值的数量为2，则生成排序的表达式并处理共享子表达式
        if len(values) == 2:
            line = (
                f"triton_helpers.sort_with_index({broadcasted_values[0]}, {broadcasted_values[1]},"
                f" {rmask}, {dim}, stable={stable}, descending={descending})"
            )
            result_vars = cse_multiple(line, len(values), masks)
        else:
            # 否则抛出异常，表明未处理的排序情况
            raise AssertionError("Unhandled sort")

        # 将生成的结果变量与输入变量关联，并设置结果变量的边界
        for result_var, input_var in zip(result_vars, values):
            result_var.mask_vars = masks  # type: ignore[attr-defined]
            result_var.bounds = input_var.bounds

        # 返回生成的结果变量元组
        return tuple(result_vars)
    def codegen_body(self):
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        # 检查是否有任何代码段存在，如果全为空，则直接返回
        if not (
            self.indexing_code
            or self.loads
            or self.stores
            or self.compute
            or self.suffix
        ):
            return

        # 如果处于内部缩减且最后一个范围树是循环
        if self.inside_reduction and self.range_trees[-1].is_loop:
            # 在self.body中生成缩减循环的代码
            self.body.writeline("for roffset in range(0, rnumel, RBLOCK):")
            with self.body.indent():
                # 头部代码生成范围树的迭代
                self.iteration_ranges_codegen_header(self.range_trees[-1], self.body)
                # 插入索引代码
                self.body.splice(self.indexing_code)
                # 插入加载代码
                self.body.splice(self.loads)
                # 插入计算代码
                self.body.splice(self.compute)
                # 插入存储代码
                self.body.splice(self.stores)

            # 使来自缩减循环内部的任何缓存无效
            self.cse.invalidate(self.outside_loop_vars)
            # 清除最后一个范围树的缓存
            self.range_trees[-1].cache_clear()
        else:
            # 插入索引代码
            self.body.splice(self.indexing_code)
            # 插入加载代码
            self.body.splice(self.loads)
            # 插入计算代码
            self.body.splice(self.compute)
            # 插入存储代码
            self.body.splice(self.stores)

        # 插入后缀代码
        self.body.splice(self.suffix)
        # 清空索引代码
        self.indexing_code.clear()
        # 清空加载代码
        self.loads.clear()
        # 清空计算代码
        self.compute.clear()
        # 清空存储代码
        self.stores.clear()
        # 清空后缀代码
        self.suffix.clear()

    def imports_for_benchmark_kernel(self):
        return textwrap.dedent(
            """
            from torch._dynamo.testing import rand_strided
            {}
            import torch
            from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid
        """.format(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        )

    def _get_heuristic(self):
        # 如果启用持续缩减，则返回"persistent_reduction"
        if self.persistent_reduction:
            assert self.inside_reduction
            return "persistent_reduction"
        # 如果处于缩减中，则返回"reduction"
        elif self.inside_reduction:
            return "reduction"
        # 否则返回"pointwise"
        return "pointwise"
    # 定义一个函数，用于生成一些与感应器相关的元数据信息，返回一个字典
    def inductor_meta_common():
        # 创建一个包含各种元数据信息的字典
        inductor_meta = {
            "backend_hash": torch.utils._triton.triton_hash_with_backend(),  # 获取与后端相关的哈希值
            "are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),  # 检查确定性算法是否启用
            "assert_indirect_indexing": config.assert_indirect_indexing,  # 从配置中获取是否启用间接索引断言
            "autotune_local_cache": config.autotune_local_cache,  # 从配置中获取本地缓存自动调优选项
            "autotune_pointwise": config.triton.autotune_pointwise,  # 从 Triton 配置中获取逐点自动调优选项
            "autotune_remote_cache": config.autotune_remote_cache,  # 从配置中获取远程缓存自动调优选项
            "force_disable_caches": config.force_disable_caches,  # 从配置中获取是否强制禁用缓存选项
            "dynamic_scale_rblock": config.dynamic_scale_rblock,  # 从配置中获取动态比例 RBlock 的选项
            "max_autotune": config.max_autotune,  # 从配置中获取最大自动调优次数
            "max_autotune_pointwise": config.max_autotune_pointwise,  # 从配置中获取逐点操作的最大自动调优次数
            "min_split_scan_rblock": config.triton.min_split_scan_rblock,  # 从 Triton 配置中获取最小拆分扫描 RBlock 的选项
            "spill_threshold": config.triton.spill_threshold,  # 从 Triton 配置中获取溢出阈值
            "store_cubin": config.triton.store_cubin,  # 从 Triton 配置中获取是否存储 cubin 文件
        }
        # 如果当前环境是 HIP 环境，设置 "is_hip" 属性为 True
        if torch.version.hip is not None:
            inductor_meta["is_hip"] = True
        # 如果当前环境是 FBCode 环境，设置 "is_fbcode" 属性为 True
        if config.is_fbcode():
            inductor_meta["is_fbcode"] = True
        # 如果配置中开启了 bandwidth profiling，添加相关配置信息到字典中
        if config.profile_bandwidth:
            inductor_meta["profile_bandwidth"] = config.profile_bandwidth
            inductor_meta["profile_bandwidth_regex"] = config.profile_bandwidth_regex
            inductor_meta["profile_bandwidth_output"] = config.profile_bandwidth_output
        # 如果配置中开启了 coordinate descent tuning，添加相关配置信息到字典中
        if config.coordinate_descent_tuning:
            inductor_meta[
                "coordinate_descent_tuning"
            ] = config.coordinate_descent_tuning
            inductor_meta[
                "coordinate_descent_search_radius"
            ] = config.coordinate_descent_search_radius
            inductor_meta[
                "coordinate_descent_check_all_directions"
            ] = config.coordinate_descent_check_all_directions
        # 返回生成的元数据字典
        return inductor_meta

    # 定义一个方法，用于获取持久的 RBLOCK 值，根据给定的 rnumel 参数计算并返回一个整数值
    def _get_persistent_RBLOCK(self, rnumel):
        # 简化 rnumel 参数，并确保其为整数类型
        rnumel = V.graph.sizevars.simplify(rnumel)
        # 如果 rnumel 是整数类型，转换为整数并计算下一个最接近的 2 的幂次方数值
        if isinstance(rnumel, (sympy.Integer, int)):
            val = int(rnumel)
            val = next_power_of_2(val)
        else:
            # 如果 rnumel 不是整数类型，默认使用 128 作为初始值，逐步扩大直到满足静态限制条件
            val = 128
            while not V.graph.sizevars.statically_known_leq(rnumel, val):
                assert val <= 16 * 1024, f"Failed to find static RBLOCK for {rnumel}"
                val *= 2
        # 返回计算得到的 RBLOCK 值
        return val
    def codegen_static_numels(self, code):
        """
        We get a small speedup from hard coding numels if they are static.

        This code stomps on the passed-in values by writing an constant to the top of the kernel.

        In a kernel like:
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        rnumel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """
        # 遍历范围树列表中的每棵树
        for tree in self.range_trees:
            # 如果树的前缀不是'r'或者当前处于内部约简状态
            if tree.prefix != "r" or self.inside_reduction:
                # 简化树的元素数（numel）
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                # 如果简化后的元素数是整数类型
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    # 写入一行代码，将树的前缀加上"numel"赋值为简化后的整数元素数
                    code.writeline(f"{tree.prefix}numel = {int(simplified_tree_numel)}")

            # 如果树的前缀是'r'且持久约简状态为真
            if tree.prefix == "r" and self.persistent_reduction:
                # 获取持久RBLOCK的值
                val = self._get_persistent_RBLOCK(tree.numel)
                # 写入一行代码，将"RBLOCK"声明为tl.constexpr，并赋值为持久RBLOCK的值
                code.writeline(f"RBLOCK: tl.constexpr = {val}")

            # 如果树的前缀是'x'且不存在x维度
            if tree.prefix == "x" and self.no_x_dim:
                # 写入一行代码，将"XBLOCK"声明为tl.constexpr，并赋值为1
                code.writeline("XBLOCK: tl.constexpr = 1")

    def _get_grid_fn(self):
        # 返回字符串"grid"作为网格函数的名称
        return "grid"

    def add_numel_to_call_args_and_grid(self, name, call_args, arg_types, grid):
        # TODO(jansel): if there are constants, we shouldn't bother passing them as args
        # 遍历范围树列表中的每棵树
        for tree in self.range_trees:
            # 如果树的元素数是整数类型或符号类型
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                # 表达式为树的元素数
                expr = tree.numel
            else:
                # 否则，生成树的元素数表达式
                expr = V.graph.wrapper_code.generate_numel_expr(name, tree)

            # 如果树的前缀不是'r'或者当前处于内部约简状态
            if tree.prefix != "r" or self.inside_reduction:
                # 将表达式添加到调用参数列表中
                call_args.append(expr)
                # 将表达式类型添加到参数类型列表中
                arg_types.append(type(expr))
            # 如果树的网格维度不为None
            if tree.grid_dim is not None:
                # 将表达式添加到网格列表中
                grid.append(expr)
    # 调用内核函数，执行与节点相关的计算任务
    def call_kernel(self, name: str, node: Optional[IRNode] = None):
        # 获取图形计算图的包装器代码
        wrapper = V.graph.wrapper_code
        # 写入 Triton 头部信息（如果尚未写入）
        wrapper.write_triton_header_once()
        # 获取参数定义的 Python 代码字符串，以及调用参数、参数类型信息
        _, call_args, _, arg_types = self.args.python_argdefs()
        # 用于存储并行计算网格的列表
        grid: List[Any] = []
        # 添加元素数量到调用参数和网格中
        self.add_numel_to_call_args_and_grid(name, call_args, arg_types, grid)
        # 获取当前设备信息
        current_device = V.graph.scheduler.get_current_device_or_throw()

        # 如果存在工作空间参数，生成工作空间分配代码
        if self.args.workspace_arg is not None:
            ws = self.args.workspace_arg
            wrapper.generate_workspace_allocation(
                ws.nbytes, current_device, ws.zero_fill
            )

        # 生成默认的计算网格
        grid = wrapper.generate_default_grid(name, grid)
        # 生成内核调用代码，包括名称、调用参数、网格、设备索引等信息
        wrapper.generate_kernel_call(
            name,
            call_args,
            grid,
            current_device.index,
            cuda=True,
            triton=True,
            arg_types=arg_types,
            grid_fn=self._get_grid_fn(),
            triton_meta=self.triton_meta,
        )

        # 如果存在工作空间参数，生成释放工作空间的代码
        if self.args.workspace_arg is not None:
            wrapper.writeline(wrapper.make_free_by_names(["workspace"]))

    # 生成 NaN 检查的代码
    def codegen_nan_check(self):
        # 获取图形计算图的包装器代码
        wrapper = V.graph.wrapper_code
        # 获取参数定义的 Python 代码字符串，以及调用参数、参数类型信息
        _, call_args, arg_types, _ = self.args.python_argdefs()
        # 遍历调用参数和其对应的类型
        for arg, arg_type in zip(call_args, arg_types):
            # 如果参数类型是 TensorArg 类型
            if isinstance(arg_type, TensorArg):
                # 如果使用了 C++ 包装器，并且 ABI 兼容性已开启
                if V.graph.cpp_wrapper:
                    if config.abi_compatible:
                        # 生成 AOTI_TORCH_ERROR_CODE_CHECK 宏的调用
                        wrapper.writeline(
                            f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan("{arg}", {arg}));'
                        )
                    else:
                        # 生成断言 NaN 和 Inf 的代码
                        wrapper.writeline(f'assert_inf_and_nan("{arg}", {arg});')
                else:
                    # 生成检查张量是否包含 NaN 的代码
                    line = f"assert not {arg}.isnan().any().item()"
                    wrapper.writeline(line)
                    # 生成检查张量是否包含 Inf 的代码
                    line = f"assert not {arg}.isinf().any().item()"
                    wrapper.writeline(line)

    # 创建公共子表达式变量
    def create_cse_var(self, *args, **kwargs):
        return TritonCSEVariable(*args, **kwargs)

    # 生成迭代范围的入口代码
    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        # 生成表达式赋值语句
        line = f"{entry.name} = {self.kexpr(self.rename_indexing(entry.expr))}"
        # 如果入口是循环，则将索引代码写入索引代码块
        if entry.root.is_loop:
            self.indexing_code.writeline(line)
        else:
            # 否则将非约简存储提升到循环外部
            self.body.writeline(line)

    # 生成迭代范围的范围代码
    def iteration_ranges_ranges_code(self, entry):
        # 确保张量维度非空
        assert entry.tensor_dim is not None
        # 获取索引大小的字符串表示
        size = self.indexing_size_str(entry.tensor_dim)
        # 获取索引数据类型
        index_dtype = self.index_dtype
        # 如果索引数据类型不是 'tl.int32'，则进行转换
        convert = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
        return f"tl.arange(0, {entry.prefix.upper()}BLOCK){size}{convert}"

    # 生成迭代范围的标量代码
    def iteration_ranges_scalar_code(self, entry, value):
        # 获取索引数据类型
        index_dtype = self.index_dtype
        # 获取 Triton 张量的维度数量
        ndim = self.triton_tensor_ndim()
        # 创建指定值的标量张量
        size = [1] * ndim
        return f"tl.full({size}, {value}, {index_dtype})"
    # 获取程序标识符（pid），基于给定的迭代条目（entry）
    def iteration_ranges_get_pid(self, entry):
        # 断言确保 grid_dim 不为 None
        assert entry.grid_dim is not None
        # 根据 grid_dim 构建用于缓存键的字符串
        key = f"tl.program_id({entry.grid_dim})"
        
        # 如果 grid_dim 为 1，并且没有 z 维度，并且 numel 不超过最大的 y 网格数
        if (
            entry.grid_dim == 1
            and not entry.has_zdim
            and not (isinstance(entry.numel, int) and entry.numel <= get_max_y_grid())
        ):
            # 对于超过最大 y 网格数的 numel，需要使用 z 维度
            # 每个 z 维度，有 tl.num_programs(1) 个 y 块，通过 grad(x, y, z) 传递
            # 因此，需要添加 tl.program_id(z) * tl.num_programs(y) * YBLOCK 来获得正确的 y 偏移量
            key = f"({key} + tl.program_id({entry.grid_dim + 1}) * tl.num_programs({entry.grid_dim}))"
        
        # 从 pid_cache 中获取 pid，如果缓存中不存在则返回 key 本身
        pid = entry.pid_cache.get(key, key)
        
        # 如果 index_dtype 不是 "tl.int32"，则将 pid 转换为指定的 index_dtype 类型
        if self.index_dtype != "tl.int32":
            return f"{pid}.to({self.index_dtype})"
        
        # 返回获取的 pid
        return pid

    # 检查是否具有常量掩码
    def _has_constant_mask(self, tree: IterationRangesRoot):
        # 如果 numel 静态已知且等于 1，则返回 True
        if V.graph.sizevars.statically_known_equals(tree.numel, 1):  # type: ignore[arg-type]
            return True
        
        # 如果 prefix 是 "r" 并且启用了持久化归约，计算最大的 BLOCK
        elif tree.prefix == "r" and self.persistent_reduction:
            max_block = self._get_persistent_RBLOCK(tree.numel)
        
        # 如果 prefix 是 "x" 并且没有 x 维度，则最大的 BLOCK 为 1
        elif tree.prefix == "x" and self.no_x_dim:
            max_block = 1
        
        # 否则，根据 prefix 在 TRITON_MAX_BLOCK 中查找最大的 BLOCK
        else:
            if tree.prefix.upper() not in TRITON_MAX_BLOCK:
                return False
            max_block = TRITON_MAX_BLOCK[tree.prefix.upper()]
        
        # 可选优化：如果 BLOCK 精确地整除 numel，则永远不需要进行掩码加载来处理末尾的 stragglers
        # 避免掩码加载速度更快，但总是进行掩码加载是安全的
        return V.graph.sizevars.statically_known_multiple_of(tree.numel, max_block)

    # 过滤掩码变量
    def filter_masks(self, mask_vars):
        # 遍历 range_trees 中的每棵树
        for tree in self.range_trees:
            # 如果树具有常量掩码，则从 mask_vars 中移除相应的掩码变量
            if self._has_constant_mask(tree):
                mask_vars.discard(f"{tree.prefix}mask")
    # 定义一个方法用于生成迭代范围的代码头部，接受两个参数：entry 和 code
    def iteration_ranges_codegen_header(self, entry, code):
        # 将 entry 的前缀赋给变量 x
        x = entry.prefix
        # 如果 entry 是一个循环
        if entry.is_loop:
            # 在 code 中写入一个表达式，形如 "{entry.name} = {x}offset + {x}base"
            code.writeline(f"{entry.name} = {x}offset + {x}base")
        # 如果 entry 的 grid_dim 为 None
        elif entry.grid_dim is None:
            # 不需要 "{x}offset = " 的前缀
            # 在 code 中写入两行代码：第一行是调用 self.iteration_ranges_ranges_code 方法，第二行是设置 {x}offset = 0
            code.writeline(f"{entry.name} = {self.iteration_ranges_ranges_code(entry)}")
            code.writeline(f"{x}offset = 0")
        else:
            # 如果 entry 的 tensor_dim 不为 None
            if entry.tensor_dim is not None:
                # 构建一个表达式 line，形如 "{x}offset + {self.iteration_ranges_ranges_code(entry)}"
                line = f"{x}offset + {self.iteration_ranges_ranges_code(entry)}"
            else:
                # 否则调用 self.iteration_ranges_scalar_code 方法，生成一个标量代码表达式
                line = self.iteration_ranges_scalar_code(entry, f"{x}offset")
            # 在 code 中写入两行代码：第一行是计算 {x}offset，第二行是设定 {entry.name} 的值为先前构建的 line
            code.writelines(
                [
                    f"{x}offset = {self.iteration_ranges_get_pid(entry)} * {x.upper()}BLOCK",
                    f"{entry.name} = {line}",
                ]
            )

        # 如果 entry 具有常数掩码
        if self._has_constant_mask(entry):
            # 生成一个全为真的掩码 {x}mask = tl.full({sizes}, True, tl.int1)
            sizes = self.dense_size_str()
            code.writeline(f"{x}mask = tl.full({sizes}, True, tl.int1)")
        else:
            # 否则生成一个掩码 {x}mask = {entry.name} < {x}numel
            code.writeline(f"{x}mask = {entry.name} < {x}numel")
class TritonScheduling(SIMDScheduling):
    int32_type = "tl.int32"  # 定义 int32 类型的字符串常量
    int64_type = "tl.int64"  # 定义 int64 类型的字符串常量
    kernel_type = TritonKernel  # 指定使用 TritonKernel 作为内核类型
    backend_features = dict.fromkeys(  # 创建后端特性字典，以确保顺序一致
        [
            BackendFeature.FOREACH,  # 支持 FOREACH 特性
            BackendFeature.BUCKETIZE,  # 支持 BUCKETIZE 特性
            BackendFeature.INPLACE_BUFFERS,  # 支持 INPLACE_BUFFERS 特性
            BackendFeature.MASKED_SCATTER_WITH_INDEX,  # 支持 MASKED_SCATTER_WITH_INDEX 特性
            BackendFeature.SCAN,  # 支持 SCAN 特性
            BackendFeature.TRITON_TEMPLATES,  # 支持 TRITON_TEMPLATES 特性
        ]
    )
    if torch.version.hip is None:
        backend_features.update(  # 在没有 HIP 版本时，更新后端特性字典
            dict.fromkeys(
                [
                    # TODO: 当 ROCm Triton 添加对多输入支持时，将此移至上方
                    BackendFeature.TUPLE_REDUCTION,  # 支持 TUPLE_REDUCTION 特性
                    BackendFeature.SORT,  # 支持 SORT 特性
                ]
            )
        )

    @classmethod
    def get_backend_features(cls, device: torch.device):
        return cls.backend_features  # 返回后端特性字典

    def codegen_comment(self, node_schedule):
        wrapper = V.graph.wrapper_code  # 获取图形包装器
        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)  # 获取节点调度的内核元数据
        if origins:
            wrapper.writeline(origins)  # 将内核元数据写入包装器

        if config.debug_fusion:
            from torch._inductor.scheduler import (  # 导入调度器节点类
                BaseSchedulerNode,
                ForeachKernelSchedulerNode,
            )

            if not any(
                isinstance(n, ForeachKernelSchedulerNode) for n in node_schedule
            ):
                # 可能需要查看 foreach 节点内部的节点
                node_names = [
                    n.get_name()
                    for n in node_schedule
                    if isinstance(n, BaseSchedulerNode)
                ]
                wrapper.writeline(
                    f"{wrapper.comment} Fused node name list: {', '.join(node_names)}"  # 将融合节点名称列表写入包装器
                )
    # 定义一个方法，用于为特定的源代码、节点调度和内核定义内核函数
    def define_kernel(self, src_code, node_schedule, kernel):
        # 获取图形对象的包装器代码
        wrapper = V.graph.wrapper_code
        
        # 如果源代码已经在包装器的源码到内核名字的映射中存在
        if src_code in wrapper.src_to_kernel:
            # 获取已存在的内核名字
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            # 否则，根据源代码生成新的内核名字
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_category = get_kernel_category_by_source_code(src_code)[:3]
            kernel_name = "_".join(
                ["triton", kernel_category, fused_name, wrapper.next_kernel_suffix()]
            )
            
            # 将原始的源代码作为键，将新生成的内核名字存入映射表
            wrapper.src_to_kernel[src_code] = kernel_name
            subs_name = kernel_name if config.triton.unique_kernel_names else "triton_"

            # 根据配置替换源代码中的特定占位符
            src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), subs_name)

            # 在源代码中处理特定的代码段，这是一个临时的修复方法
            src_code = src_code.replace("#pragma CMT", "#")

            # 根据源代码的哈希值生成文件路径相关信息
            basename, _, kernel_path = get_path(code_hash(src_code.strip()), "py")

            # 创建编译器包装对象，生成异步编译的调用代码
            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.triton({subs_name!r}, '''")
            compile_wrapper.splice(src_code, strip=True)
            current_device = V.graph.scheduler.get_current_device_or_throw()
            compile_wrapper.writeline(f"''', device_str='{current_device.type}')")

            # 生成包含内核路径和元数据的注释
            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            
            # 在包装器中定义内核，包括编译器的代码和元数据的注释
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )

            # 如果启用了内核元数据记录功能，记录内核的信息
            if is_metric_table_enabled("kernel_metadata"):
                log_kernel_metadata(kernel_name, kernel_path, src_code)

        # 返回最终确定的内核名字
        return kernel_name

    # 保持随机数生成器状态的装饰器函数
    @preserve_rng_state()
    # 定义一个方法用于对一组节点进行性能基准测试
    def benchmark_fused_nodes(self, nodes):
        # 从节点生成核心代码，设置为基准测试模式
        src_code = self.generate_kernel_code_from_nodes(nodes, benchmark_kernel=True)
        # 加载生成的代码模块
        mod = PyCodeCache.load(src_code)

        # 定义缓存文件路径生成函数
        def cache_file_path():
            # 确保模块有文件路径
            assert mod.__file__ is not None
            # 返回模块文件路径的内核性能缓存文件路径
            return os.path.splitext(mod.__file__)[0] + ".kernel_perf"

        # 加载缓存中的性能数据
        def load_cache():
            path = cache_file_path()
            if os.path.exists(path):
                with open(path) as fd:
                    return float(fd.read())
            return None

        # 将性能数据存入缓存
        def store_cache():
            path = cache_file_path()
            with open(path, "w") as fd:
                fd.write(str(ms))

        # 记录调试信息，显示生成的核心代码的文件路径
        log.debug(
            "kernel src code for %s written to: %s",
            {n.get_name() for n in nodes},
            mod.__file__,
        )

        # 从缓存中加载性能数据
        ms = load_cache()
        if ms is not None:
            return ms, mod.__file__

        # 获取模块的参数并准备调用
        args = mod.get_args()
        call = mod.call
        wrapped_jit_function = mod.triton_

        # 调用一次以触发编译过程
        try:
            call(wrapped_jit_function.clone_args(*args)[0])
        except Exception as e:
            # 记录异常情况，通常在编译时出现问题
            log.debug(
                "Exception (%s) in compiling fused nodes %s",
                e,
                {n.get_name() for n in nodes},
            )
            ms = float("inf")
            store_cache()
            return ms, mod.__file__

        # 获取模块的启动器信息
        launchers = wrapped_jit_function.launchers
        assert len(launchers) == 1
        if launchers[0].n_spills > 0:
            # 如果有寄存器溢出，跳过核心性能测试
            ms = float("inf")
        else:
            # 执行GPU基准测试，评估核心代码的性能
            ms = do_bench_gpu(lambda: call(wrapped_jit_function.clone_args(*args)[0]))

            # 考虑参数克隆的开销，以便在原地更新时生成核心代码
            ms = ms - do_bench_gpu(lambda: wrapped_jit_function.clone_args(*args))

        # 记录核心代码的性能测试结果
        log.debug(
            "The fused kernel for %s took %.3f ms to run",
            {n.get_name() for n in nodes},
            ms,
        )
        # 将性能数据存入缓存
        store_cache()
        return ms, mod.__file__
```