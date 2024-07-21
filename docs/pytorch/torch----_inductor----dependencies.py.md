# `.\pytorch\torch\_inductor\dependencies.py`

```
# 设置 mypy：允许未类型化的定义
import abc  # 导入抽象基类模块
import collections  # 导入集合模块
import dataclasses  # 导入数据类模块
import itertools  # 导入迭代工具模块
import logging  # 导入日志模块
import re  # 导入正则表达式模块
import typing  # 导入类型提示模块
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union  # 导入多种类型提示

from unittest.mock import patch  # 从单元测试模块导入 patch 函数

import sympy  # 导入符号计算模块

import torch  # 导入 PyTorch 模块
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols  # 导入符号形状模块

from .codegen.common import index_prevent_reordering  # 从当前目录下的 codegen 模块导入 index_prevent_reordering 函数
from .utils import (  # 从当前目录下的 utils 模块导入以下函数：
    get_dtype_size,
    reduction_num_outputs,
    sympy_index_symbol,
    sympy_str,
    sympy_subs,
    VarRanges,
)
from .virtualized import OpsHandler, ReductionType, V  # 从当前目录下的 virtualized 模块导入 OpsHandler, ReductionType, V 类

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
is_indirect = re.compile(r"indirect|tmp").search  # 编译正则表达式用于检测字符串中是否包含 "indirect" 或 "tmp"


class Dep(abc.ABC):
    name: str  # 依赖的名称，字符串类型
    index: sympy.Expr  # 依赖的索引表达式，符号表达式类型

    @abc.abstractmethod
    def rename(self, renames: Dict[str, str]) -> "Dep":
        pass  # 抽象方法：重命名依赖对象

    @abc.abstractmethod
    def get_numel(self) -> sympy.Expr:
        pass  # 抽象方法：获取依赖对象的元素数量表达式

    @abc.abstractmethod
    def numbytes_hint(self):
        pass  # 抽象方法：获取依赖对象的字节提示信息

    @abc.abstractmethod
    def has_unbacked_symbols(self) -> bool:
        pass  # 抽象方法：检查依赖对象是否含有未支持的符号

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        pass  # 抽象方法：检查依赖对象是否连续


@dataclasses.dataclass(frozen=True)
class MemoryDep(Dep):
    name: str  # 内存依赖的名称，字符串类型
    index: sympy.Expr  # 内存依赖的索引表达式，符号表达式类型
    var_names: Tuple[sympy.Symbol, ...]  # 变量名称元组，包含符号变量
    size: Tuple[sympy.Expr, ...]  # 大小元组，包含符号表达式
    mode: Optional[str] = None  # 模式，可选的字符串，默认为 None

    def __repr__(self):
        return f"MemoryDep({self.name!r}, {self.index}, {self.ranges}, {self.mode})"
        # 返回对象的字符串表示，包括名称、索引、范围和模式信息

    def get_offset(self):
        """
        Return the offset by setting every variable to be 0.
        """
        return sympy_subs(self.index, {v: 0 for v in self.var_names})
        # 返回通过将每个变量设为 0 来计算的偏移量
    def normalize_with_stride_order(self, prefix="t"):
        r"""
        Used to decide if two MemoryDep does not equal due to different loop orders.
        More specifically, when dep1 and dep2 are not equal, we can normalize
        both and check if they are equal after that. If yes, then the mismatch is
        caused by different loop orders.
        """
        # 导入此处以避免循环导入
        from torch._inductor import ir
        
        # 获取变量的步长提示
        strides = V.graph.sizevars.stride_hints(self.index, self.var_names)
        
        # 按步长降序选择循环顺序
        order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        stride_reorder = ir.same_reorder(order)
        sizes = self.size
        var_names = self.var_names
        
        # 使用步长重新排序变量名称和大小
        new_reordered_sizes = stride_reorder(sizes)
        new_reordered_var_names = stride_reorder(var_names)
        
        # 简化循环并获取新的简化大小和重新索引
        new_simplified_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            new_reordered_var_names,
            new_reordered_sizes,
            index_prevent_reordering(
                [self.index], new_reordered_var_names, new_reordered_sizes
            ),
        )
        
        # 使用传入的前缀创建新的符号
        var_ranges, add_var = var_builder(prefix)
        replacement = dict(
            zip(
                new_reordered_var_names,
                reindex([add_var(x) for x in new_simplified_sizes]),
            )
        )
        
        # 使用符号替换并扩展现有的索引
        new_index = sympy_subs(sympy.expand(self.index), replacement)
        
        # 创建并返回新的 MemoryDep 对象
        out = MemoryDep(self.name, new_index, tuple(var_ranges.keys()), tuple(var_ranges.values()))  # type: ignore[arg-type]
        return out

    @property
    def ranges(self) -> Dict[sympy.Symbol, sympy.Expr]:
        """{c0: 128, c1: 512, ...}"""
        return dict(zip(self.var_names, self.size))

    def get_numel(self) -> sympy.Expr:
        if self.is_indirect():
            # 如果是间接索引，则获取元素数量
            numel = V.graph.get_numel(self.name)
        else:
            vars = set(self.index.free_symbols)
            numel = sympy.Integer(1)
            for var, size in zip(self.var_names, self.size):
                if var in vars:
                    numel = numel * size
        return numel

    def rename(self, renames: Dict[str, str]) -> "MemoryDep":
        if self.name in renames:
            # 如果名称在重命名字典中，则返回重命名后的 MemoryDep 对象
            return MemoryDep(
                renames[self.name],
                self.index,
                var_names=self.var_names,
                size=self.size,
                mode=self.mode,
            )
        return self

    def numbytes_hint(self):
        # 返回基于元素数量的字节大小提示
        return V.graph.sizevars.size_hint(self.get_numel()) * get_dtype_size(
            V.graph.get_dtype(self.name)
        )

    def has_unbacked_symbols(self):
        # 检查是否存在未支持的符号
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        # 检查索引是否是符号，并且存在于变量名称中，用以确定是否是连续的
        return isinstance(self.index, sympy.Symbol) and self.index in self.var_names
    # 检查最后一个维度的步长是否为1
    def stride1_for_last_dim(self, result_for_complex_expression=True) -> bool:
        """
        Whether the stride for the last dimension is 1.
        """
        # 如果变量名列表为空，直接返回True，表示最后一个维度的步长为1
        if len(self.var_names) == 0:
            return True

        # 如果索引是 sympy.Add 类型，则将其各项作为 terms
        terms = self.index.args if isinstance(self.index, sympy.Add) else [self.index]

        # 取出最后一个符号变量
        last_sym = self.var_names[-1]
        
        # 遍历 terms 中的每一项
        for term in terms:
            # 如果 term 是最后一个符号变量，则返回 True，表示最后一个维度的步长为1
            if term is last_sym:
                return True

            # 如果 term 是一个乘积且其第二个参数是最后一个符号变量，
            # 且第一个参数是整数类型且大于1，则返回 False，表示步长大于1，性能较差
            if (
                isinstance(term, sympy.Mul)
                and len(term.args) == 2
                and term.args[1] is last_sym
                and isinstance(term.args[0], (int, sympy.Integer))
                and term.args[0] > 1
            ):
                return False

        # 如果以上条件都不满足，则返回 result_for_complex_expression
        return result_for_complex_expression

    # 检查索引是否为标量
    def is_scalar(self) -> bool:
        # 如果索引是 sympy.Symbol 类型，并且不在变量名列表中，并且不是间接索引，则返回 True
        if isinstance(self.index, sympy.Symbol):
            return self.index not in self.var_names and not self.is_indirect()
        # 如果索引是整数类型或者 sympy.Integer 类型，则返回 True
        return isinstance(self.index, (int, sympy.Integer))

    # 检查索引是否为间接索引
    def is_indirect(self) -> bool:
        # 检查索引中的每个自由符号是否是间接索引（使用 is_indirect 函数判断）
        return any(is_indirect(v.name) for v in self.index.free_symbols)  # type: ignore[attr-defined]
# StarDep 类，用于表示某种依赖关系，继承自 Dep 类
@dataclasses.dataclass(frozen=True)
class StarDep(Dep):
    name: str  # 名称属性，表示依赖的名称
    mode: Optional[str] = None  # 模式属性，可选，表示依赖的模式

    # index 属性的 getter 方法，抛出未实现异常，因为 StarDep 没有具体的索引
    @property
    def index(self):
        raise NotImplementedError("StarDep does not have an index")

    # 获取元素数量的方法，返回一个 sympy 表达式
    def get_numel(self) -> sympy.Expr:
        return V.graph.get_numel(self.name)

    # 根据给定的重命名字典进行重命名，返回一个新的 StarDep 实例
    def rename(self, renames: Dict[str, str]) -> "StarDep":
        if self.name in renames:
            return StarDep(renames[self.name], self.mode)
        return self

    # 返回估算的字节数提示，用于排序，不是实际的依赖
    def numbytes_hint(self):
        return V.graph.sizevars.size_hint(self.get_numel()) * get_dtype_size(
            V.graph.get_dtype(self.name)
        )

    # 判断是否存在未支持的符号
    def has_unbacked_symbols(self):
        return len(free_unbacked_symbols(self.get_numel())) > 0

    # 判断是否是连续的依赖
    def is_contiguous(self) -> bool:
        return False

    # 判断是否是标量
    def is_scalar(self) -> bool:
        return False

    # 判断是否是间接依赖
    def is_indirect(self) -> bool:
        return False


# WeakDep 类，用于表示一种弱依赖关系，继承自 Dep 类
# 该类不具备索引属性
@dataclasses.dataclass(frozen=True)
class WeakDep(Dep):
    name: str  # 名称属性，表示依赖的名称

    # index 属性的 getter 方法，抛出未实现异常，因为 WeakDep 没有具体的索引
    @property
    def index(self):
        raise NotImplementedError("WeakDep does not have an index")

    # 获取元素数量的方法，返回一个固定值 1 的 sympy 表达式
    def get_numel(self) -> sympy.Expr:
        return sympy.Integer(1)

    # 根据给定的重命名字典进行重命名，返回一个新的 WeakDep 实例
    def rename(self, renames: Dict[str, str]) -> "WeakDep":
        if self.name in renames:
            return WeakDep(renames[self.name])
        return self

    # 返回固定值 1，用于排序，不是实际的依赖
    def numbytes_hint(self):
        return 1  # Purely inserted for ordering, not an actual dep

    # 判断是否存在未支持的符号，该方法始终返回 False
    def has_unbacked_symbols(self):
        return False

    # 判断是否是连续的依赖，该方法始终返回 False
    def is_contiguous(self) -> bool:
        return False


# IndexExprDep 类，表示一个索引表达式的依赖
@dataclasses.dataclass
class IndexExprDep:
    index: sympy.Expr  # 索引表达式，使用 sympy.Expr 类型
    var_names: Tuple[sympy.Symbol, ...]  # 变量名称元组，包含 sympy.Symbol 类型的元素
    size: Tuple[sympy.Expr, ...]  # 大小元组，包含 sympy.Expr 类型的元素


# ReadWrites 类，用于存储读写操作的依赖关系集合
@dataclasses.dataclass
class ReadWrites:
    reads: Set[Dep]  # 读操作的依赖集合，包含 Dep 类型的元素
    writes: Set[Dep]  # 写操作的依赖集合，包含 Dep 类型的元素
    index_exprs: Set[IndexExprDep]  # 索引表达式的依赖集合，包含 IndexExprDep 类型的元素
    range_vars: Optional[List[sympy.Expr]] = None  # 可选的范围变量列表，元素为 sympy.Expr 类型
    var_ranges: Optional[VarRanges] = None  # 可选的变量范围，类型为 VarRanges
    op_counts: typing.Counter[str] = dataclasses.field(
        default_factory=collections.Counter  # 操作计数器，使用 collections.Counter 初始化
    )

    # 根据给定的重命名字典进行重命名，返回一个新的 ReadWrites 实例
    def rename(self, renames: typing.Dict[str, str]) -> "ReadWrites":
        return ReadWrites(
            {dep.rename(renames) for dep in self.reads},  # 对 reads 集合中的每个 dep 进行重命名
            {dep.rename(renames) for dep in self.writes},  # 对 writes 集合中的每个 dep 进行重命名
            self.index_exprs,  # 索引表达式集合保持不变
            self.range_vars,  # 范围变量保持不变
            self.var_ranges,  # 变量范围保持不变
            op_counts=self.op_counts,  # 操作计数器保持不变
        )
    # 返回一个新的 ReadWrites 对象，包含当前对象的读操作集合并加入 dep 对象
    def with_read(self, dep: Dep) -> "ReadWrites":
        # 断言 dep 是 WeakDep 或 StarDep 类的实例
        assert isinstance(dep, (WeakDep, StarDep))
        # 创建并返回新的 ReadWrites 对象，包含更新后的读操作集合
        return ReadWrites(
            set.union(self.reads, {dep}),  # 合并当前 reads 集合和新 dep 对象，生成新的 reads 集合
            self.writes,                   # 复用当前对象的 writes 集合
            self.index_exprs,              # 复用当前对象的 index_exprs 集合
            self.range_vars,               # 复用当前对象的 range_vars 集合
            self.var_ranges,               # 复用当前对象的 var_ranges 集合
            op_counts=self.op_counts,      # 复用当前对象的 op_counts 计数器
        )

    # 合并当前对象与另一个 ReadWrites 对象，并返回合并后的新对象
    def merge(self, other: "ReadWrites"):
        # 合并两个对象的 reads、writes 和 index_exprs 集合
        reads = set.union(self.reads, other.reads)
        writes = set.union(self.writes, other.writes)
        index_exprs = set.union(self.index_exprs, other.index_exprs)
        # 创建并初始化一个新的操作计数器，复制当前对象的 op_counts
        op_counts = collections.Counter(self.op_counts)
        # 更新操作计数器，加入另一个对象的 op_counts
        op_counts.update(other.op_counts)
        # 返回一个新的 ReadWrites 对象，包含合并后的 reads、writes、index_exprs 和 op_counts
        return ReadWrites(reads - writes, writes, index_exprs, op_counts=op_counts)

    # 静态方法：合并给定列表中的多个 ReadWrites 对象，并返回合并后的新对象
    @staticmethod
    def merge_list(read_writes: List["ReadWrites"]):
        # 合并所有对象的 writes 集合
        all_writes = set.union(*[rw.writes for rw in read_writes])
        # 合并所有对象的 reads 集合，减去 writes，得到所有独立的 reads
        all_reads = set.union(*[rw.reads for rw in read_writes]) - all_writes
        # 合并所有对象的 index_exprs 集合
        all_index_exprs = set.union(*[rw.index_exprs for rw in read_writes])

        # 初始化一个操作计数器
        op_counts: typing.Counter[Any] = collections.Counter()
        # 遍历所有 ReadWrites 对象，更新操作计数器
        for rw in read_writes:
            op_counts.update(rw.op_counts)

        # 返回一个新的 ReadWrites 对象，包含合并后的 reads、writes、index_exprs 和 op_counts
        return ReadWrites(all_reads, all_writes, all_index_exprs, op_counts=op_counts)

    # 移除当前对象中的指定 reads 集合元素，并返回更新后的新对象
    def remove_reads(self, rem_reads):
        return ReadWrites(
            self.reads - rem_reads,        # 移除指定的 reads 元素后的新 reads 集合
            self.writes,                   # 复用当前对象的 writes 集合
            self.index_exprs,              # 复用当前对象的 index_exprs 集合
            self.range_vars,               # 复用当前对象的 range_vars 集合
            self.var_ranges,               # 复用当前对象的 var_ranges 集合
            op_counts=self.op_counts,      # 复用当前对象的 op_counts 计数器
        )

    # 返回当前对象的 reads 和 writes 的迭代器链表
    def reads_and_writes(self):
        return itertools.chain(self.reads, self.writes)

    # 返回当前对象中所有 MemoryDep 的名称集合，可选择是否忽略整数索引
    def buffer_names(self, ignore_integer_index=True):
        """
        Integer index is used for load_seed.
        """
        # 初始化一个空集合 names
        names = set()
        # 遍历当前对象的 reads_and_writes 迭代器链表
        for dep in self.reads_and_writes():
            # 如果 dep 不是 MemoryDep 类的实例，则跳过当前循环
            if not isinstance(dep, MemoryDep):
                continue
            # 如果不忽略整数索引或者 dep 的索引不是整数类型，则将 dep 的名称添加到 names 集合中
            if not ignore_integer_index or not isinstance(
                dep.index, (int, sympy.Integer)
            ):
                names.add(dep.name)
        # 返回包含 MemoryDep 名称的集合 names
        return names
class _RecordLoadStoreInner(V.MockHandler):  # type: ignore[name-defined]
    def __init__(self, var_ranges: VarRanges, normalize: bool):
        super().__init__()
        # 初始化空集合，用于记录读取操作的依赖关系
        self._reads: Set[Dep] = set()
        # 初始化空集合，用于记录写入操作的内存依赖关系
        self._writes: Set[MemoryDep] = set()
        # 初始化空集合，用于记录索引表达式的依赖关系
        self._index_exprs: Set[IndexExprDep] = set()
        # 初始化变量范围和是否标准化的属性
        self._var_ranges: VarRanges = var_ranges
        self._normalize: bool = normalize

    def canonicalize(
        self, index: sympy.Expr
    ) -> Tuple[sympy.Expr, Tuple[sympy.Symbol, ...], Tuple[sympy.Expr, ...]]:
        if not self._normalize:
            # 简化变量范围的大小，排除大小为1的变量
            sizes = [V.graph.sizevars.simplify(x) for x in self._var_ranges.values()]
            var_names = tuple(
                k for k, v in zip(self._var_ranges.keys(), sizes) if v != 1
            )
            sizes = tuple(v for v in sizes if v != 1)
            return index, var_names, sizes  # type: ignore[return-value]

        # 尝试进一步简化索引，即使simplify_loops未能将其转换为最简形式，因为受不同索引公式的干扰。
        free_symbols = index.free_symbols
        # 简化变量范围，并生成新的变量范围字典
        var_ranges = {
            k: V.graph.sizevars.simplify(v)
            for k, v in self._var_ranges.items()
            # TODO(jansel): 进一步探索此标准化
            # if k in free_symbols
        }
        index_vars = [*var_ranges.keys()]
        sizes = tuple(var_ranges.values())
        # 使用_simplify_loops方法简化循环
        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            index_vars,
            sizes,
            index_prevent_reordering([index], index_vars, sizes),
        )

        # 分配新的变量以处理编号不匹配的维度
        new_vars, add_var = var_builder(canonicalization_prefix())
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))
        index = sympy_subs(sympy.expand(index), replacement)

        new_vars = [*new_vars.keys()]
        new_sizes = [*new_sizes]
        free_symbols = index.free_symbols
        while new_vars and new_vars[-1] not in free_symbols:
            # 简化掉最后一个维度（已简化的），但下游用户不会这样做。标准化这一点。
            new_vars.pop()
            new_sizes.pop()
        return index, tuple(new_vars), tuple(new_sizes)  # type: ignore[arg-type]

    def load(self, name: str, index: sympy.Expr) -> str:
        # 将读操作添加到_reads集合中，记录内存依赖
        self._reads.add(MemoryDep(name, *self.canonicalize(index)))
        return f"load({name}, {sympy_str(index)})"

    def load_seed(self, name: str, index: int):
        assert isinstance(index, int)
        # 将load_seed转换为load操作，使用整数索引
        return self.load(name, sympy.Integer(index))

    def store(self, name: str, index: sympy.Expr, value: str, mode=None) -> str:
        # 将写操作添加到_writes集合中，记录内存依赖和模式（如果有）
        self._writes.add(MemoryDep(name, *self.canonicalize(index), mode=mode))
        return f"store({name}, {sympy_str(index)}, {value}, {mode})"
    # 将指定的值存储到数据库中，返回存储操作的字符串表示
    def store_reduction(self, name: str, index, value) -> str:
        return self.store(name, index, f"store_reduction({value})")

    # 将索引表达式添加到索引表达式集合中，并返回表达式的字符串表示
    def index_expr(self, index: sympy.Expr, dtype) -> str:
        self._index_exprs.add(IndexExprDep(*self.canonicalize(index)))
        return f"index_expr({sympy_str(index)}, {dtype})"

    # 将值分桶，记录对偏移名称的读取，并返回分桶操作的字符串表示
    def bucketize(
        self,
        values,
        offsets_name: str,
        offsets_size: sympy.Expr,
        indexing_dtype: torch.dtype,
        right: bool,
    ):
        self._reads.add(StarDep(offsets_name))
        return f"bucketize({values}, {offsets_name}, {sympy_str(offsets_size)}, {indexing_dtype}, {right})"
class _OpCounter:
    """用于计算每个操作使用次数的包装器类"""

    def __init__(self, inner):
        super().__init__()
        self.parent_handler = inner
        self._op_counts: typing.Counter[Any] = collections.Counter()

    def __getattr__(self, name):
        # 当调用不存在的属性时，增加对应操作的计数并委托给父处理器
        self._op_counts[name] += 1
        return getattr(self.parent_handler, name)


class RecordLoadStore(V.KernelFormatterHandler):  # type: ignore[name-defined]
    def __init__(self, var_ranges: VarRanges, normalize: bool):
        # 创建内部处理器并包装成 OpCounter 对象
        parent_handler = _RecordLoadStoreInner(
            var_ranges=var_ranges, normalize=normalize
        )
        parent_handler = _OpCounter(parent_handler)
        super().__init__(parent_handler=parent_handler)


# TODO: check call sites
def var_builder(prefix: str) -> Tuple[VarRanges, Callable[[sympy.Expr], sympy.Symbol]]:
    cnt = itertools.count()
    var_ranges: VarRanges = dict()

    def add_var(length: sympy.Expr) -> sympy.Symbol:
        # 创建带有指定前缀的新符号，并将其长度添加到 var_ranges 中
        v = sympy_index_symbol(f"{prefix}{next(cnt)}")
        var_ranges[v] = length
        return v

    return var_ranges, add_var


def index_vars_no_squeeze(*argsizes: Tuple[sympy.Expr, ...], prefix: str):
    # 使用给定前缀构建变量，并返回其列表及其范围
    var_ranges, add_var = var_builder(prefix)
    args: List[List[sympy.Symbol]] = []
    for size in argsizes:
        args.append(list(map(add_var, size)))
    return args, var_ranges


def index_vars_squeeze(*argsizes: Tuple[sympy.Expr, ...], prefix: str = "d"):
    from .ir import SqueezeView

    # 使用给定前缀构建变量，并返回经过压缩后的变量列表及其范围
    var_ranges, add_var = var_builder(prefix)
    args: List[List[sympy.Expr]] = []
    new_sizes: List[List[sympy.Expr]] = []
    for size in argsizes:
        new_size, reindex = SqueezeView.squeezer(size)
        new_sizes.append(new_size)
        args.append(reindex(list(map(add_var, new_size))))
    return args, var_ranges


def extract_read_writes(
    fn: Callable[..., Any],
    *argsizes: Tuple[sympy.Expr, ...],
    normalize: bool = False,
    prefix: str = "d",
):
    # 提取读写操作，并返回相关信息
    args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)
    rw = RecordLoadStore(var_ranges, normalize=normalize)
    with V.set_ops_handler(rw):
        fn(*args)

    if normalize:
        range_vars = []  # 由于标准化，变量数量可能不同
    else:
        range_vars = list(itertools.chain.from_iterable(args))

    inner = rw.parent_handler.parent_handler
    return ReadWrites(
        set(inner._reads),
        set(inner._writes),
        inner._index_exprs,
        range_vars,
        var_ranges,
        rw.parent_handler._op_counts,
    )


def extract_input_node_reduction_ranges(
    input_node: "torch._inductor.ir.TensorBox",
) -> Tuple[Optional[List[sympy.Expr]], Optional[List[sympy.Expr]]]:
    """
    返回所有输入节点的大小和减少大小，如果大小和减少大小（如果存在）都相同。
    可能一个节点有多个输入，一些是减少节点，其他是逐点节点。
    在这种情况下，减少节点的减少大小需要相同。
    """
    # 实现略
    Otherwise returns (None, None).
    """

    from .ir import ComputedBuffer, Loops  # 导入必要的模块，包括ComputedBuffer和Loops

    if isinstance(input_node.data, ComputedBuffer):
        # 如果输入节点的数据已经被实现过，返回其大小和减少大小
        size = input_node.get_size()  # 获取节点大小
        reduction_size = input_node.get_reduction_size()  # 获取节点减少大小
        if len(reduction_size) > 0:
            return (size, reduction_size)  # 如果存在减少大小，返回大小和减少大小
        else:
            return (None, None)  # 否则返回空

    if not isinstance(input_node.data.data, Loops):  # type: ignore[attr-defined]
        # 其他的IR节点没有减少范围
        return (None, None)  # 返回空

    # 存在一个问题：如果在输入节点和其依赖的实现节点之间存在视图/排列，会怎么样？
    # 当前方法仍然使用依赖实现节点的减少范围，这不是理想的情况。
    # 是否有一种方法可以检查中间是否有排列？
    reads = input_node.get_reads()  # 获取输入节点的读取操作
    reduction_size = None  # 初始化减少大小为None
    size = None  # 初始化大小为None
    while reduction_size is None and len(reads) > 0:
        seen = set()  # 初始化一个集合，用于记录已经处理过的读取名称
        new_reads = []  # 初始化一个空列表，用于存储新的读取操作
        for read in reads:
            if not isinstance(read, MemoryDep):
                continue  # 如果读取操作不是MemoryDep类型，则继续下一个循环
            if read.name in seen:
                continue  # 如果读取操作的名称已经在集合中存在，继续下一个循环
            seen.add(read.name)  # 将读取操作的名称添加到集合中
            buffer = V.graph.get_buffer(read.name)  # 从图中获取名称对应的缓冲区
            if buffer is None:
                continue  # 如果缓冲区为None，则继续下一个循环
            if (
                isinstance(buffer, ComputedBuffer)
                and len(buffer.get_reduction_size()) > 0
            ):
                if reduction_size is None:
                    reduction_size = buffer.get_reduction_size()  # 获取缓冲区的减少大小
                    size = buffer.get_size()  # 获取缓冲区的大小
                elif (
                    reduction_size != buffer.get_reduction_size()
                    or size != buffer.get_size()
                ):
                    return (None, None)  # 如果发现不一致的减少大小或大小，返回空
            else:
                new_reads.extend(buffer.get_reads())  # 如果缓冲区不是ComputedBuffer类型或没有减少大小，则获取其读取操作
        if reads == new_reads:
            return (size, reduction_size)  # 如果当前读取操作列表与新读取操作列表相同，返回大小和减少大小
        else:
            reads = new_reads  # 否则更新读取操作列表为新的读取操作列表
    return (size, reduction_size)  # 返回大小和减少大小
# 返回字符串 "c" 作为规范化的前缀
def canonicalization_prefix():
    return "c"


# ops handler which computes all the free unbacked symbols for an IR
class FreeUnbackedSymbolsOpsHandler:
    symbols: Set[sympy.Symbol]  # 类成员变量 symbols，用于存储符号集合

    def __init__(self):
        self.symbols = set()  # 初始化 symbols 为空集合

    def __getattr__(self, name: str) -> Callable[..., Any]:
        # 动态属性访问器，用于捕获方法调用并处理其中的符号
        def inner(*args, **kwargs):
            for a in itertools.chain(args, kwargs.values()):
                if isinstance(a, (sympy.Expr, sympy.logic.boolalg.Boolean)):
                    self.symbols |= free_unbacked_symbols(a)

        return inner

    def indirect_indexing(self, index_var, size, check=True) -> sympy.Symbol:
        assert not isinstance(index_var, (sympy.Expr, sympy.logic.boolalg.Boolean))
        self.symbols |= free_unbacked_symbols(size)
        return sympy_index_symbol(f"({str(index_var)})")  # 返回索引符号的表示形式

    def frexp(self, x):
        return (None,) * 2  # 返回空结果元组

    def scan(self, dtypes, combine_fn, values):
        return (None,) * len(values)  # 返回与 values 相同长度的空结果元组

    def sort(self, dtypes, values, stable, descending):
        return (None,) * len(values)  # 返回与 values 相同长度的空结果元组

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[None, Tuple[None, ...]],
    ) -> Union[None, Tuple[None, ...]]:
        num_values = reduction_num_outputs(reduction_type)
        return (None,) * num_values if num_values > 1 else None  # 根据输出数量返回相应长度的空结果元组或单个空值


# 对 FreeUnbackedSymbolsOpsHandler 类进行类型检查的辅助函数
def _typecheck_FreeUnbackedSymbolsOpsHandler(
    h: FreeUnbackedSymbolsOpsHandler,
) -> OpsHandler[None]:
    return h  # 直接返回传入的 FreeUnbackedSymbolsOpsHandler 实例


# 提取给定函数 fn 在指定索引和可选反向索引 rindex 下的所有自由未备份符号
def extract_free_unbacked_symbols(fn: Callable[..., Any], index, rindex=None):
    from .ir import FlexibleLayout

    args = [index, rindex] if rindex is not None else [index]
    handler = FreeUnbackedSymbolsOpsHandler()  # 创建 FreeUnbackedSymbolsOpsHandler 实例
    # 使用特定的操作处理器 handler 运行函数 fn，捕获其中的符号
    # NB: 我模仿了这里的 allow_indexing 补丁，我不理解为什么人们到处都这样做
    with V.set_ops_handler(handler), patch.object(
        FlexibleLayout, "allow_indexing", True
    ):
        fn(*args)
    return handler.symbols  # 返回处理器中收集到的所有符号集合
```