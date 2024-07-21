# `.\pytorch\torch\_inductor\codegen\simd.py`

```py
# mypy: allow-untyped-defs
from __future__ import annotations

import collections  # 导入collections模块，用于高级容器数据类型的支持
import contextlib  # 导入contextlib模块，用于创建和管理上下文管理器的工具函数和对象
import dataclasses  # 导入dataclasses模块，用于简化创建和操作数据类的工具
import functools  # 导入functools模块，用于高阶函数和操作可调用对象的工具
import itertools  # 导入itertools模块，用于创建和操作迭代器的工具
import logging  # 导入logging模块，用于记录日志消息
import math  # 导入math模块，提供标准数学函数
import operator  # 导入operator模块，提供对内置运算符的函数形式访问
from typing import (  # 导入typing模块中的各种类型提示
    Any,
    Callable,
    Counter,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import sympy  # 导入sympy库，用于符号数学计算

import torch  # 导入torch库，用于张量计算和神经网络
import torch._logging  # 导入torch._logging模块，用于处理torch的内部日志

from torch.utils._sympy.functions import FloorDiv, ModularIndexing  # 从torch.utils._sympy.functions导入特定函数
from torch.utils._sympy.symbol import (  # 从torch.utils._sympy.symbol导入特定符号处理工具
    free_symbol_is_type,
    symbol_is_type,
    SymT,
)
from ..._dynamo.utils import counters  # 从指定位置导入counters工具
from .. import config, ir, scheduler  # 从指定位置导入config、ir、scheduler模块
from ..codecache import code_hash  # 从指定位置导入code_hash模块

from ..dependencies import Dep, MemoryDep, StarDep, WeakDep  # 从指定位置导入多个依赖模块
from ..ir import TritonTemplateBuffer  # 从ir模块导入TritonTemplateBuffer类
from ..optimize_indexing import indexing_dtype_strength_reduction  # 从optimize_indexing模块导入指定函数
from ..runtime.hints import ReductionHint  # 从runtime.hints模块导入ReductionHint类
from ..runtime.runtime_utils import green_text, yellow_text  # 从runtime_utils模块导入指定函数
from ..scheduler import (  # 从scheduler模块导入多个类和函数
    BaseSchedulerNode,
    BaseScheduling,
    WhyNoFuse,
)
from ..utils import (  # 从utils模块导入多个实用函数
    get_dtype_size,
    IndentedBuffer,
    Placeholder,
    sympy_index_symbol,
    sympy_product,
    sympy_subs,
    unique,
)
from ..virtualized import ops, OpsWrapper, V  # 从virtualized模块导入ops、OpsWrapper、V

from .common import (  # 从common模块导入多个类和函数
    CSEVariable,
    index_prevent_reordering,
    Kernel,
    PythonPrinter,
)
from .multi_kernel import MultiKernel  # 从multi_kernel模块导入MultiKernel类


log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")  # 获取性能提示的日志记录器对象
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")  # 获取调度相关的日志记录器对象
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")  # 获取融合相关的日志记录器对象


pexpr = PythonPrinter().doprint  # 创建PythonPrinter实例并调用doprint方法，返回打印表达式的函数


@dataclasses.dataclass
class IterationRanges:
    """
    Each range tree represents multiple sets of iteration indexing
    in a single tiled dimension in the output kernel.

    If you have two loops ranges one (4, 3, 2) and another (4, 6),
    then the range tree will be:
            4 (i0)
        3 (i1)  6 (i3)
        2 (i2)
    Where i0 is shared between both loops, but then the split into
    different indexing vars.  All loop ranges must iterate over
    the same number of elements.
    """

    def __init__(
        self,
        name: str,
        var_list: List[sympy.Symbol],
        var_ranges: Dict[sympy.Symbol, sympy.Expr],
        numel: sympy.Expr,
        prefix: str,
        *,
        kernel: SIMDKernel,
        divisor=sympy.Integer(1),
        length=sympy.Integer(1),
        root: IterationRangesRoot,
    ):
        super().__init__()  # 调用父类的初始化方法
        self.name = name  # 设置迭代范围的名称
        self.var_list = var_list  # 设置变量符号列表
        self.var_ranges = var_ranges  # 设置变量范围的表达式字典
        self.numel = numel  # 设置元素数的表达式
        self.prefix = prefix  # 设置前缀字符串
        self.divisor = divisor  # 设置除数，默认为1
        self.length = length  # 设置长度，默认为1
        self.kernel = kernel  # 设置SIMD内核对象
        self.root = root  # 设置迭代范围的根节点

    def symbol(self):
        return sympy_index_symbol(self.name)  # 返回符号表达式


class IterationRangesRoot(IterationRanges):
    pass  # 迭代范围的根节点类继承自迭代范围类
    def __init__(
        self,
        name: str,
        numel: sympy.Expr,
        # TODO: this is probably SymTy.INDEX and SymTy.RINDEX
        prefix: str,
        index: int,
        kernel: SIMDKernel,
        pid_cache=None,
        *,
        is_loop: bool,
        tensor_dim: Optional[int],
        grid_dim: Optional[int],
        has_zdim: bool,
    ):
        # 如果 pid_cache 为 None，则初始化为空字典
        if pid_cache is None:
            pid_cache = {}
        # 调用父类的构造方法，传递必要的参数
        super().__init__(
            name=name,
            var_list=[],  # 变量列表初始化为空列表
            var_ranges={},  # 变量范围初始化为空字典
            numel=numel,  # 元素数量
            prefix=prefix,  # 前缀
            kernel=kernel,  # SIMD 内核对象
            root=self,
        )
        self.index = index  # 存储索引
        # 存储所有节点的字典，节点为 sympy.Expr 对象，值为 IterationRangesEntry 对象
        self.nodes: Dict[sympy.Expr, IterationRangesEntry] = {}
        # 用于在 Triton mm 模板中重新排序程序 ID 的缓存
        self.pid_cache: Dict[str, str] = pid_cache

        # 如果维度被实现为一个单独的程序循环遍历整个维度，则为 True
        # （目前仅用于非持久化约简）
        assert not is_loop or (prefix == "r" and grid_dim is None)
        self.is_loop = is_loop  # 是否是循环
        self.tensor_dim = tensor_dim  # 对应于 Triton 张量的维度索引
        self.grid_dim = grid_dim  # 对应于 Triton 网格的维度索引
        self.has_zdim = has_zdim  # 是否有 z 维度

    def __repr__(self):
        # 返回对象的字符串表示形式，用于调试和输出
        return f"IterationRangesRoot({self.name!r}, {self.numel}, ...)"

    def cache_clear(self):
        # 清除所有节点的缓存
        for node in self.nodes.values():
            node.cache_clear()

    def index_sym(self):
        # 返回与当前索引相关的 sympy 符号
        return sympy_index_symbol(f"{self.prefix}index")

    def lookup(self, divisor, length):
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
        # 如果 divisor * length 等于 numel，则静态知道的变量大小相等
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = FloorDiv(self.index_sym(), divisor)  # 整除运算的 sympy 表达式
        else:
            expr = ModularIndexing(self.index_sym(), divisor, length)  # mod 索引的 sympy 表达式

        if expr not in self.nodes:
            # 创建新的迭代范围条目对象
            node = IterationRangesEntry(
                f"{self.prefix}{next(V.kernel.iter_vars_count)}",
                divisor,
                length,
                expr,
                self,
            )
            V.kernel.range_tree_nodes[node.symbol()] = node  # 将节点添加到范围树中
            self.var_list.append(node.symbol())  # 将节点符号添加到变量列表中
            self.var_ranges[node.symbol()] = length  # 设置节点符号的长度
            self.nodes[expr] = node  # 将节点添加到节点字典中
        return self.nodes[expr]  # 返回节点对象

    def construct_entries(self, lengths: List[sympy.Expr]):
        # 构造迭代范围条目对象列表
        divisor = sympy.Integer(1)
        itervars = []
        for length in reversed(lengths):
            itervars.append(self.lookup(divisor, length))  # 查找或创建迭代范围条目
            divisor = divisor * length  # 更新除数
        return list(reversed(itervars))  # 返回反转后的迭代范围条目列表

    def construct(self, lengths: List[sympy.Expr]):
        # 构造迭代范围条目符号列表
        return [e.symbol() for e in self.construct_entries(lengths)]  # 返回迭代范围条目符号列表
    def vars_and_sizes(self, index: sympy.Expr):
        """Figure out vars from this tree used in index"""
        # 从这棵树中找出在 index 中使用的变量
        nodes = [V.kernel.range_tree_nodes.get(s) for s in index.free_symbols]
        # 筛选出前缀匹配且存在的节点列表
        nodes = [n for n in nodes if n and n.prefix == self.prefix]
        # 根据节点的 sizevars 大小提示对节点列表进行排序
        nodes.sort(key=lambda x: V.graph.sizevars.size_hint(x.divisor))
        divisor = sympy.Integer(1)
        index_vars = []
        sizes = []

        def add(node):
            nonlocal divisor
            # 添加节点的符号到索引变量列表中
            index_vars.append(node.symbol())
            # 添加节点的长度到大小列表中
            sizes.append(node.length)
            # 更新除数
            divisor = divisor * node.length

        for node in nodes:
            if not V.graph.sizevars.statically_known_equals(node.divisor, divisor):
                # 如果节点的除数与当前除数不同，则填充未使用的索引变量
                add(self.lookup(divisor, FloorDiv(node.divisor, divisor)))
                divisor = node.divisor
            add(node)
        if not V.graph.sizevars.statically_known_equals(self.numel, divisor):
            # 如果 self.numel 与当前除数不同，则填充未使用的索引变量
            add(self.lookup(divisor, FloorDiv(self.numel, divisor)))

        # 返回反转后的索引变量列表和大小列表
        return list(reversed(index_vars)), list(reversed(sizes))
class IterationRangesEntry(IterationRanges):
    # IterationRangesEntry 类继承自 IterationRanges 类
    def __init__(
        self,
        name: str,
        divisor: sympy.Expr,
        length: sympy.Expr,
        expr: sympy.Expr,
        parent: IterationRanges,
    ):
        # 调用父类 IterationRanges 的构造函数进行初始化
        super().__init__(
            name=name,
            numel=parent.numel / length,
            var_list=parent.var_list,
            var_ranges=parent.var_ranges,
            prefix=parent.prefix,
            divisor=divisor,
            length=length,
            kernel=parent.kernel,
            root=parent.root,
        )
        # 设置实例变量 parent 和 codegen
        self.parent = parent
        self.codegen = functools.lru_cache(None)(self._codegen)
        self.expr = expr

    def __repr__(self):
        # 返回对象的字符串表示，包括 name, divisor, length, expr 和 var_ranges
        return f"IterationRangesEntry({self.name}, {self.divisor}, {self.length}, {self.expr}, {self.var_ranges})"

    def set_name(self, name):
        # 设置对象的名称，并更新 codegen 函数以生成新名称
        self.codegen = lambda: name  # type: ignore[assignment]
        self.codegen.cache_clear = lambda: None  # type: ignore[method-assign]
        self.name = name

    def cache_clear(self):
        # 清除 codegen 函数的缓存
        self.codegen.cache_clear()

    def _codegen(self):
        # 调用 V.kernel 的 codegen_iteration_ranges_entry 方法生成代码，并返回名称
        V.kernel.codegen_iteration_ranges_entry(self)
        return self.name

    def precomputed_args(self):
        # 为动态形状找到需要预先计算的索引表达式部分
        precomputed_args: List[sympy.Expr] = []
        if isinstance(self.expr, sympy.Symbol):
            return precomputed_args
        assert isinstance(self.expr, (FloorDiv, ModularIndexing)), type(self.expr)
        for arg in self.expr.args[1:]:
            if not isinstance(arg, (sympy.Integer, sympy.Symbol)):
                symbols = arg.free_symbols
                if len(symbols) > 0 and all(
                    symbol_is_type(s, SymT.SIZE) for s in symbols
                ):
                    precomputed_args.append(arg)
        return precomputed_args

    def __hash__(self):
        # 返回对象的哈希值，基于名称的哈希
        return hash(self.name)

    def __eq__(self, other):
        # 比较两个对象是否相等，基于名称
        return self.name == other.name


def constant_repr(value):
    # 返回常量的字符串表示，处理特殊值如无穷大、负无穷大和 NaN
    if value == float("inf"):
        return 'float("inf")'
    elif value == float("-inf"):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    return repr(value)


class SIMDKernel(Kernel):
    """
    Common base class for Triton/Halide codegen which both use flattened indexing rather than loop nests.
    """

    sexpr = pexpr
    kexpr: Callable[[sympy.Expr], str]
    allow_block_ptr = False

    def __init__(
        self,
        *groups,
        index_dtype: str,
        mutations: Optional[Set[str]] = None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        override_persistent_reduction=None,
    ):
        # SIMDKernel 类的构造函数，接受多个参数并进行初始化
        super().__init__(*groups)
        self.index_dtype = index_dtype
        self.mutations = mutations
        self.pid_cache = pid_cache
        self.reduction_hint = reduction_hint
        self.override_persistent_reduction = override_persistent_reduction
        ):
            # 如果 pid_cache 为 None，则初始化为空字典
            if pid_cache is None:
                pid_cache = {}
            # 调用父类的初始化方法
            super().__init__()
            # 创建一个缓冲区对象来存储代码主体
            self.body = IndentedBuffer()
            # 创建一个缓冲区对象用于存储索引代码
            self.indexing_code = IndentedBuffer()
            # 计算每个组的元素数量，并简化为最简形式
            self.numels = [V.graph.sizevars.simplify(s) for s in groups]
            # 初始化 mutations 属性为传入的 mutations 集合，如果为 None 则为空集合
            self.mutations: Set[str] = mutations if mutations is not None else set()
            # 初始化 range_trees 为空列表，用于存储迭代范围的树结构
            self.range_trees: List[IterationRangesRoot] = []
            # 初始化 range_tree_nodes 为空字典，用于存储符号到迭代范围条目的映射
            self.range_tree_nodes: Dict[sympy.Symbol, IterationRangesEntry] = {}
            # 使用 itertools.count() 创建一个迭代器对象，用于生成迭代变量
            self.iter_vars_count = itertools.count()
            # 根据 numels 最后一个元素是否为 1，设置 inside_reduction 属性
            self.inside_reduction = self.numels[-1] != 1
            # 初始化 reduction_hint 属性为传入的 reduction_hint
            self.reduction_hint = reduction_hint
            # 初始化 index_dtype 属性为传入的 index_dtype
            self.index_dtype: str = index_dtype
            # 初始化 last_usage 属性为空集合，用于存储最后使用的标识符
            self.last_usage: Set[str] = set()
            # 初始化 buf_accesses 属性为默认字典，用于存储缓冲区访问的依赖关系列表
            self.buf_accesses: DefaultDict[str, List[Dep]] = collections.defaultdict(list)
            # 初始化 persistent_reduction 属性，根据传入的 override_persistent_reduction 或者默认策略决定
            self.persistent_reduction: bool = (
                override_persistent_reduction
                if override_persistent_reduction is not None
                else self.should_use_persistent_reduction()
            )
            # 初始化 no_x_dim 属性，根据 want_no_x_dim 方法的返回值决定
            self.no_x_dim = self.want_no_x_dim()
            # 初始化 code_hash 属性为空
            self.code_hash = None

            # 定义一个闭包函数 simplify_indexing，使用 functools.lru_cache 进行结果缓存
            @functools.lru_cache(None)
            def simplify_indexing(index: sympy.Expr):
                # 使用 sizevars.simplify_with_ranges 简化索引表达式
                index = V.graph.sizevars.simplify_with_ranges(index, self.var_ranges())
                # 对每棵 range_trees 树进行迭代，组合连续的维度
                for tree in self.range_trees:
                    index = self.combine_contiguous_dims(index, tree)

                # 组合模块化索引对
                return self.combine_modular_indexing_pairs(index)

            # 将 simplify_indexing 函数赋值给对象的 simplify_indexing 属性
            self.simplify_indexing = simplify_indexing
            # 初始化 range_trees 对象，根据传入的 pid_cache
            self.initialize_range_tree(pid_cache)

    def want_no_x_dim(self):
        # 默认返回 False，表示不希望去除 x 维度
        return False

    def initialize_range_tree(self, pid_cache):
        # 计算是否没有 r 维度
        no_r_dim = not self.inside_reduction or self.numels[-1] == 1

        # 定义一些前缀和活动前缀，用于确定 tensor_dims
        prefixes = "zyxr"
        active_prefixes = prefixes[-len(self.numels):]

        # 根据条件设置 grid_dims 和 tensor_dims
        grid_dims = "xyz"
        if self.no_x_dim:
            tensor_dims = "r"
        elif no_r_dim:
            tensor_dims = "xyz"
        else:
            tensor_dims = "xyzr"

        # 筛选出 active_prefixes 中包含的 tensor_dims 的前缀
        tensor_dims = "".join(p for p in tensor_dims if p in active_prefixes)

        # 根据 active_prefixes 和 tensor_dims 创建 range_trees 对象
        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix == "r"
            tensor_dim = tensor_dims.find(prefix) if prefix in tensor_dims else None
            grid_dim = None if is_reduction else grid_dims.find(prefix)
            index = i if grid_dim is None else grid_dim
            self.range_trees.append(
                IterationRangesRoot(
                    f"{prefix}index",
                    self.numels[i],
                    prefix,
                    index,
                    self,
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                    has_zdim="z" in active_prefixes,
                )
            )
    def finalize_indexing(self, indices: Sequence[sympy.Expr]):
        """
        Hook called right before codegen with every index that will be
        used in the fused kernel.
        """
        pass

# 定义一个方法finalize_indexing，接受一个Sympy表达式序列indices作为参数，用于在生成融合内核代码之前调用的钩子函数。


    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
        prior = self.inside_reduction
        self.inside_reduction = False
        try:
            return self.store(name, index, value)
        finally:
            self.inside_reduction = prior

# 定义一个方法store_reduction，用于存储缩减操作的结果。它接受名称name（字符串）、索引index（Sympy表达式）、值value（CSEVariable对象）作为参数，并在存储之前将inside_reduction标记为False，最后恢复其先前的值。


    def should_use_persistent_reduction(self) -> bool:
        return False  # defined in subclass

# 定义一个方法should_use_persistent_reduction，返回布尔值False，子类中需要重新定义该方法。


    def var_ranges(self):
        return dict(
            itertools.chain.from_iterable(
                tree.var_ranges.items() for tree in self.range_trees
            )
        )

# 定义一个方法var_ranges，返回一个字典，包含所有range_trees中各个tree的var_ranges属性项的链式迭代结果。


    def triton_tensor_ndim(self):
        return sum(int(tree.tensor_dim is not None) for tree in self.range_trees)

# 定义一个方法triton_tensor_ndim，返回一个整数，表示所有range_trees中tensor_dim不为None的tree的数量之和。


    def indexing_size_str(self, i):
        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[i] = ":"
        return f"[{', '.join(sizes)}]"

# 定义一个方法indexing_size_str，接受整数参数i，返回一个格式化的字符串，表示在第i个位置将所有tensor的尺寸大小用None填充，并在指定位置用冒号":"替换。


    def dense_size_list(self) -> List[str]:
        sizes = ["1"] * self.triton_tensor_ndim()
        for tree in self.range_trees:
            if tree.tensor_dim is None:
                continue

            if tree.prefix != "r" or self.inside_reduction:
                sizes[tree.tensor_dim] = f"{tree.prefix.upper()}BLOCK"
        return sizes

# 定义一个方法dense_size_list，返回一个字符串列表，表示所有range_trees中每个tensor的密集尺寸列表，使用1初始化。如果tree的tensor_dim为None，则跳过；否则，根据条件设置sizes中相应的位置。


    def dense_size_str(self):
        sizes = self.dense_size_list()
        return f"[{', '.join(sizes)}]"

# 定义一个方法dense_size_str，返回一个格式化的字符串，表示所有dense_size_list中的密集尺寸大小。


    def combine_modular_indexing_pairs(self, index):
        if not isinstance(index, ModularIndexing):
            return index
        x = index.args[0]
        if (tree_node := self.range_tree_nodes.get(x)) is None:
            return index
        new_index = sympy_subs(index, {x: tree_node.expr})
        new_index = V.graph.sizevars.combine_modular_indexing_pairs(new_index)
        # the index now contains xindex/etc, which is nonstandard, fix it up
        return sympy_subs(
            new_index,
            {
                tree_node.root.index_sym(): tree_node.root.lookup(
                    sympy.Integer(1), tree_node.root.numel
                ).symbol()
            },
        )

# 定义一个方法combine_modular_indexing_pairs，用于组合模块化索引对。如果index不是ModularIndexing类型，则直接返回。获取index的第一个参数x，如果tree_node中未找到x，则返回原始索引。将index中的x替换为tree_node.expr，并使用V.graph.sizevars.combine_modular_indexing_pairs处理新索引。最后修正新索引中的非标准部分。


    def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
        if expand_res := V.graph.sizevars.expand_floor_div(index):
            new_index, denominator = expand_res  # type: ignore[misc]
            return FloorDiv(self._combine_contiguous_dims(new_index, tree), denominator)
        else:
            return self._combine_contiguous_dims(index, tree)

# 定义一个方法combine_contiguous_dims，用于组合连续的维度。如果可以通过V.graph.sizevars.expand_floor_div扩展index，则返回一个FloorDiv对象，否则直接返回_combine_contiguous_dims的结果。
    def _combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
        """
        More aggressive simplification to merge contiguous dims
        """
        # 如果索引是整数或符号，则直接返回
        if isinstance(index, (sympy.Integer, sympy.Symbol)):
            return index
        # 获取索引中的变量和大小信息
        index_vars, sizes = tree.vars_and_sizes(index)
        # 如果维度数量小于等于1，则返回原始索引
        if len(sizes) <= 1:
            return index
        # 进行循环变量简化，获取新的大小、重新索引函数和修剪函数
        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, index_prevent_reordering([index], index_vars, sizes)
        )
        # 如果新的大小和旧的大小相同，则返回原始索引
        if new_sizes == sizes:
            return index
        # 根据新的大小构造新的索引变量
        new_index_vars = tree.construct(new_sizes)
        # 使用重新索引函数更新索引表达式
        new_index = sympy_subs(index, dict(zip(index_vars, reindex(new_index_vars))))
        return new_index

    def set_last_usage(self, nodes):
        # 如果不在归约内部或者是持久归约，则直接返回
        if not self.inside_reduction or self.persistent_reduction:
            return
        # 设置最后使用的节点集合
        self.last_usage = set(
            itertools.chain.from_iterable(
                n.last_usage for n in nodes if n is not EnableReduction
            )
        )

    def disable_reduction(self):
        # 判断是否需要刷新
        should_flush = self.range_trees[-1].is_loop

        @contextlib.contextmanager
        def ctx():
            # 如果最后一个元素个数为1，则确保不在归约内部，并直接返回
            if self.numels[-1] == 1:
                assert not self.inside_reduction
                yield
                return
            # 如果需要刷新，则调用codegen_body()刷新所有挂起的缓冲区，并生成归约循环
            if should_flush:
                # calling codegen_body() will flush all the pending buffers
                # and write out a reduction loop
                self.codegen_body()
            # 离开归约内部状态
            self.inside_reduction = False
            try:
                yield
                # 如果需要刷新，则再次调用codegen_body()刷新所有挂起的缓冲区
                if should_flush:
                    # flush out any code before opening the next loop
                    self.codegen_body()
            finally:
                # 恢复归约内部状态
                self.inside_reduction = True

        return ctx()

    def set_ranges(self, *lengths):
        # 断言长度与范围树的长度相同
        assert len(lengths) == len(self.range_trees)
        # 根据给定长度构造范围
        return [
            ranges.construct(length)
            for length, ranges in zip(lengths, self.range_trees)
        ]

    @staticmethod
    def _split_iteration_ranges(
        groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]
        # 将分组和长度信息作为输入参数，返回分割后的迭代范围
    ):
        # 获取图形对象的大小变量
        sv = V.graph.sizevars
        # 初始化一个二维列表，用于存储新的范围表达式
        new_ranges: List[List[sympy.Expr]] = [[] for _ in groups]
        # 复制待处理的大小列表
        remaining = [sv.simplify(g) for g in groups]
        # 用于生成唯一变量的计数器
        var_count = itertools.count()

        def add_range(i, expr):
            # 简化表达式
            expr = sv.simplify(expr)
            # 如果无法静态确定剩余大小是否是表达式的倍数，则抛出异常
            if not sv.statically_known_multiple_of(remaining[i], expr):
                raise CantSplit
            # 将剩余大小除以表达式，获取商并赋给剩余大小
            remaining[i] = FloorDiv(remaining[i], expr)
            # 将表达式添加到对应的新范围中
            new_ranges[i].append(expr)
            # 返回下一个变量的编号
            return next(var_count)

        def make_combined(size, idx1, idx2):
            # 创建组合函数，用于获取组合变量的值
            def getter(flat_vars):
                return size * flat_vars[idx1] + flat_vars[idx2]

            return getter

        # 返回值获取器列表的初始化
        return_getters_groups = []
        # 当前处理的组索引
        current_group = 0
        # 遍历长度组列表
        for length_group in lengths:
            # 单个返回值获取器列表的初始化
            return_getters = []
            # 遍历每个大小
            for size in length_group:
                # 如果大小已知为1，则直接返回0的lambda函数
                if sv.statically_known_equals(size, 1):  # type: ignore[arg-type]
                    return_getters.append(lambda _: sympy.Integer(0))
                    continue

                # 寻找下一个仍有剩余元素的组
                while current_group < len(remaining) and sv.statically_known_equals(
                    remaining[current_group], 1  # type: ignore[arg-type]
                ):
                    current_group += 1  # 滚动到下一个仍有剩余元素的组

                # 如果当前组和下一组的大小都已知，并且大小大于当前组的剩余大小
                if current_group + 1 < len(remaining) and sv.statically_known_gt(
                    size, remaining[current_group]
                ):
                    # 需要将大小分割为两部分
                    if not sv.statically_known_multiple_of(
                        size, remaining[current_group]
                    ):
                        raise CantSplit
                    size1 = remaining[current_group]
                    size2 = FloorDiv(size, remaining[current_group])
                    # 创建组合函数，并添加到返回值获取器列表中
                    return_getters.append(
                        make_combined(
                            size2,
                            add_range(current_group, size1),
                            add_range(current_group + 1, size2),
                        )
                    )
                else:
                    # 否则直接返回单个索引的获取函数
                    return_getters.append(
                        operator.itemgetter(add_range(current_group, size))
                    )
            # 将当前组的返回值获取器列表添加到总列表中
            return_getters_groups.append(return_getters)

        # 断言所有剩余大小都满足图形对象的大小提示为1，否则抛出异常
        assert all(
            V.graph.sizevars.size_hint(s) == 1 for s in remaining
        ), f"failed to set ranges {remaining} {lengths}"

        # 返回新的范围列表和返回值获取器列表
        return new_ranges, return_getters_groups

    @classmethod
    def is_compatible(
        cls, groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]
    ):
        try:
            # 尝试调用内部方法进行迭代范围的分割
            cls._split_iteration_ranges(groups, lengths)
            # 如果成功，返回True
            return True
        except CantSplit:
            # 如果抛出异常，则返回False
            return False
    def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
        # 获取每个迭代空间的尺寸
        groups = [rt.numel for rt in self.range_trees]
        # 如果不是在归约内部，则最后一个尺寸设为1
        if not self.inside_reduction:
            groups[-1] = sympy.Integer(1)

        # 如果长度与每个迭代空间的尺寸匹配，并且长度乘积减去组尺寸等于0
        if len(lengths) == len(self.range_trees) and all(
            V.graph.sizevars.simplify(sympy_product(x) - g) == 0
            for x, g in zip(lengths, groups)
        ):
            # 设置新的迭代空间范围
            return self.set_ranges(*lengths)

        # 分割迭代范围以匹配这个内核的组，使得能够进行瓷砖化和非瓷砖化融合
        new_ranges, return_getters_groups = self._split_iteration_ranges(
            groups, lengths
        )
        # 设置新的迭代范围并返回迭代变量
        itervars = list(itertools.chain.from_iterable(self.set_ranges(*new_ranges)))
        # 对返回的获取器组中的每个函数应用迭代变量，并返回结果
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def is_indirect_indexing(self, index: sympy.Expr):
        # 检查索引是否属于间接索引（tmpX）
        return free_symbol_is_type(index, SymT.TMP)

    def is_broadcasted(self, index: sympy.Expr):
        # 注意：当存在间接索引时，此函数可能不正确
        if self.is_indirect_indexing(index):
            return False

        # 初始化索引尺寸列表为1
        index_numels = [1] * len(self.numels)
        # 遍历索引的自由符号
        for symbol in index.free_symbols:
            # 如果符号不在迭代树节点中，则跳过（例如步长等非迭代变量）
            if symbol not in self.range_tree_nodes:
                continue
            # 获取迭代树节点中的条目长度，并更新索引尺寸列表
            entry = self.range_tree_nodes[symbol]  # type: ignore[index]
            assert isinstance(entry.parent, IterationRangesRoot)
            index_numels[entry.parent.index] *= entry.length

        # 如果索引变量只在内核的一部分尺寸上迭代，则说明它是广播的
        simplify = V.graph.sizevars.simplify
        return any(
            simplify(idx_range) != simplify(iter_range)  # type: ignore[arg-type]
            for idx_range, iter_range in zip(index_numels, self.numels)
        )

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in output code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the generated kernel.

        Index expressions often need to be passed in as arguments to the triton kernel.
        Rename_indexing and codegen_indexing keep track of the needed indices and add
        new parameters to the function signature.
        """
        # 如果索引是列表，则递归地转换每个索引并用逗号分隔
        if isinstance(index, list):
            return f"[{', '.join(map(self.index_to_str, index))}]"
        # 使用重命名后的索引表达式生成输出代码中可用的字符串
        return self.kexpr(self.rename_indexing(index))  # type: ignore[call-arg]

    def prepare_indexing(
        self,
        index: sympy.Expr,
        ):
            # 简化索引，确保其符合预期格式
            index = self.simplify_indexing(index)
            # 使用预先计算的替换项，对索引中的符号进行替换
            index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
            # 如果简单替换未能消除 floor/ceil 函数，尝试进行完整替换
            if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
                index = index.subs(V.graph.sizevars.precomputed_replacements)
            # 最后的备选方案，如果表达式中没有范围变量，则将其提升
            # TODO: 不要盲目查找复杂表达式，应在生成索引时提升输入/输出大小和步长，
            # 但目前内核的输入和输出尚未设置，需要进行更深层次的重构
            if len(index.atoms(sympy.ceiling)):
                for a in index.atoms(sympy.ceiling):
                    # 对于嵌套表达式，atoms 方法按照顶层优先顺序生成
                    # 如果一切顺利，较低级别的替换将得到空结果
                    symbols = a.free_symbols
                    if len(symbols) > 0 and all(
                        symbol_is_type(s, (SymT.SIZE, SymT.PRECOMPUTED_SIZE))
                        for s in symbols
                    ):
                        # 替换 ceil 函数中的符号为预计算的大小
                        replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                        index = sympy_subs(index, replacements)

        # 生成优化后的索引表达式的代码
        return self.codegen_indexing(self.simplify_indexing(index))


    def active_range_trees(self, reorder=False):
        # 获取活跃的范围树，排除规约内部的树
        trees = [
            t for t in self.range_trees if t.prefix != "r" or self.inside_reduction
        ]
        # 如果需要重新排序且树的数量大于 1
        if reorder and len(trees) > 1:
            # 统计前缀为 "xyz" 的树的数量
            count = sum(t.prefix in "xyz" for t in trees)
            # 断言前缀顺序是 "zyx" 的倒序
            assert "".join(t.prefix for t in trees[:count]) == "zyx"[-count:], [
                t.prefix for t in trees[:count]
            ]
            # 对前缀为 "xyz" 的树进行倒序排列
            trees[:count] = reversed(trees[:count])
        # 返回活跃的范围树列表
        return trees


    def codegen_indexing(self, expr: sympy.Expr):
        # 简化表达式中的索引，使用变量范围进行简化
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        # 对表达式中的自由符号按名称排序处理
        for sym in sorted(expr.free_symbols, key=str):
            if sym in self.range_tree_nodes:
                # 如果索引表达式复杂，我们在主机端预先计算它，并作为内核参数发送结果
                replacements = {}
                for ps in self.range_tree_nodes[sym].precomputed_args():  # type: ignore[index]
                    # 使用预先计算的大小替换变量
                    replacements[ps] = V.graph.sizevars.lookup_precomputed_size(ps)
                # 如果有替换项，则对表达式进行替换
                if len(replacements) > 0:
                    self.range_tree_nodes[sym].expr = sympy_subs(  # type: ignore[index]
                        self.range_tree_nodes[sym].expr, replacements  # type: ignore[index]
                    )
                # 生成范围树节点的代码
                self.range_tree_nodes[sym].codegen()  # type: ignore[index]
        # 返回处理后的表达式
        return expr

    @contextlib.contextmanager
    def mask_loads(self, mask, value):
        """
        Context manager to add an additional mask to tl.load/store

        Args:
            mask: Tensor or boolean mask to apply.
            value: Value to store during loading.

        Returns:
            Generator yielding the applied mask.

        """
        prior = self._load_mask
        prior_val = self._load_other
        
        # Combine current mask with prior mask using logical AND
        if prior:
            mask = ops.logical_and(mask, prior)

        # Unwrap OpsWrapper if mask is wrapped
        mask = OpsWrapper._unwrap(mask)

        # Set the current load mask and value
        self._load_mask = mask
        self._load_other = value
        
        try:
            # Yield the current mask being applied
            yield mask
        finally:
            # Restore previous load mask and value
            self._load_mask = prior
            self._load_other = prior_val

    def get_strides_of_load(self, index: sympy.Expr):
        """
        Get the stride of the index for each tiling variable at index 0.

        Args:
            index: Sympy expression representing the index.

        Returns:
            dict: Mapping of tiling variable to its stride.

        """
        # Map range tree node names to their expressions
        index_to_tile_indexes = {k: v.expr for k, v in self.range_tree_nodes.items()}
        
        # Substitute index using range tree node expressions
        index_in_tile_vars = sympy_subs(index, index_to_tile_indexes)  # type: ignore[arg-type]
        
        strides = {}
        # Calculate stride for each range tree
        for range_tree in self.range_trees:
            s = sympy_index_symbol(range_tree.name)
            stride_value = sympy_subs(index_in_tile_vars, {s: 1}) - sympy_subs(
                index_in_tile_vars, {s: 0}
            )
            strides[s] = stride_value
        
        return strides

    @staticmethod
    def _map_tuple_or_scalar(fn, value):
        """
        Map a function over a tuple or scalar value.

        Args:
            fn: Function to apply.
            value: Tuple or scalar value.

        Returns:
            tuple or scalar: Result of applying the function to value.

        """
        if isinstance(value, tuple):
            return tuple(map(fn, value))
        return fn(value)
    # 当输入的内核存在混合布局时打印消息
    def warn_mix_layout(self, kernel_name):
        """
        Print message if the kernel have mixed layout inputs.
        Only care about 4D tensor for now.
        """
        # 检查只有一个输入缓冲区且只有一个输出缓冲区，并且没有就地缓冲区
        if (
            len(self.args.input_buffers) == 1
            and len(self.args.output_buffers) == 1
            and len(self.args.inplace_buffers) == 0
        ):
            # 即使输入缓冲区和输出缓冲区具有不同的布局，这可能是一个布局转换内核，无需警告混合布局。
            return

        # 获取参数的默认值、调用参数、签名及其它信息
        argdefs, call_args, signature, _ = self.args.python_argdefs()
        uniform_stride_order = None
        # 遍历调用参数
        for arg_name in call_args:
            # 获取缓冲区对象
            buf = V.graph.get_buffer(arg_name)
            if buf and len(buf.layout.size) == 4:
                # 如果只有一个维度不为零，则忽略此张量
                if len([x for x in buf.layout.size if x == 1]) == 3:
                    continue
                # 获取缓冲区布局的步幅顺序
                stride_order = ir.get_stride_order(buf.layout.stride)
                if uniform_stride_order is None:
                    uniform_stride_order = stride_order
                elif uniform_stride_order != stride_order:
                    # 打印警告消息，指示发现不一致的步幅顺序
                    msg = yellow_text(
                        f"Expected stride order {uniform_stride_order}, but found stride order"
                        + f" {stride_order} for kernel {kernel_name}"
                    )
                    log.warning(msg)

                    # 获取调用参数中每个缓冲区的步幅顺序列表、大小列表和来源列表
                    stride_order_list = [
                        ir.get_stride_order(V.graph.get_buffer(name).layout.stride)
                        if V.graph.get_buffer(name)
                        else None
                        for name in call_args
                    ]
                    size_list = [
                        V.graph.get_buffer(name).layout.size
                        if V.graph.get_buffer(name)
                        else None
                        for name in call_args
                    ]
                    source_list = [
                        "GraphInput"
                        if name in V.graph.graph_inputs
                        else "IntermediateBuffer"
                        if name in V.graph.name_to_buffer
                        else None
                        for name in call_args
                    ]

                    # 打印详细信息
                    msg = yellow_text(
                        f"  param names {argdefs}\n  buf names {call_args}\n  strides {stride_order_list}"
                        + f"\n  sizes {size_list}\n  sources {source_list}\n"
                    )
                    log.warning(msg)
                    return
        # 所有输入的布局均一致，打印消息
        msg = green_text(
            f"All the inputs for the triton kernel {kernel_name} have uniform layout"
        )
        log.warning(msg)
    # 使用 Welford 方法对值进行归约操作，计算值的总和
    def welford_reduce_fallback(self, dtype, value):
        sum_ = ops.reduction(dtype, dtype, "sum", value)
        # 标记内部归约过程结束
        self.inside_reduction = False
        # 获取当前处理的数据元素数量，并将其转换为指定的数据类型
        rnumel = ops.index_expr(self.numels[-1], dtype)
        # 计算值的均值
        mean = ops.truediv(sum_, rnumel)

        # 标记内部归约过程开始
        self.inside_reduction = True
        # 计算值与均值之间的差值
        dx = ops.sub(value, mean)
        # 计算差值的平方
        dx2 = ops.mul(dx, dx)
        # 对平方差值进行总和归约
        m2 = ops.reduction(dtype, dtype, "sum", dx2)
        # 返回包含均值、平方差总和及数据元素数量的元组
        return OpsWrapper._unwrap((mean, m2, rnumel))

    # 抛出未实现错误，提示需要在子类中实现具体的代码生成核心方法
    def codegen_kernel(self):
        raise NotImplementedError

    # 空方法体，用于子类中实现具体的代码生成主体逻辑
    def codegen_body(self):
        pass

    # 接受迭代范围条目作为参数的方法，用于子类中生成迭代范围的入口点
    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        pass
# 定义一个 SIMDScheduling 类，继承自 BaseScheduling 类
class SIMDScheduling(BaseScheduling):
    # 设置 kernel_type 属性为 SIMDKernel，在子类中会被覆盖
    kernel_type = SIMDKernel  # override in subclass
    # 设置 int32_type 属性为 "torch.int32"
    int32_type = "torch.int32"
    # 设置 int64_type 属性为 "torch.int64"
    int64_type = "torch.int64"

    # 初始化方法，接受一个 scheduler 参数，并调用父类的初始化方法
    def __init__(self, scheduler):
        super().__init__()
        # 将传入的 scheduler 参数赋值给实例变量 self.scheduler
        self.scheduler = scheduler

    # 定义一个 group_fn 方法，接受 sizes 参数，返回一个元组
    def group_fn(self, sizes):
        # 使用列表推导式，对 sizes 中的每个 s 进行处理，简化操作并返回元组
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    # 将 can_fuse_vertical 和 can_fuse_horizontal 设置为 can_fuse 的值
    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    # 定义一个 codegen_node 方法，接受一个 node 参数，可能是 FusedSchedulerNode 或 SchedulerNode 类型的对象
    def codegen_node(
        self, node: Union[scheduler.FusedSchedulerNode, scheduler.SchedulerNode]
    ):
        """
        给定一组预先融合的节点，生成一个 Triton 内核。
        """

        # 获取节点集合，并使用 max 函数找到具有最大 is_reduction() 值的节点
        nodes: List[scheduler.SchedulerNode] = node.get_nodes()  # type: ignore[assignment]

        # 从最大节点中获取 group 的第二个元素，分别赋值给 numel 和 rnumel
        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group

        # 使用节点生成调度计划
        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        
        # 创建一个 defaultdict 对象，用于存储节点的缓冲区访问信息
        buf_accesses = collections.defaultdict(list)
        for node in nodes:
            # 遍历节点的读写访问，将访问的名称作为键，访问对象列表作为值存入 buf_accesses 中
            for access in node.read_writes.reads | node.read_writes.writes:
                buf_accesses[access.name].append(access)

        # 记录调度日志，输出调度计划的信息
        schedule_log.debug("Schedule:\n %s", node_schedule)

        # 调用 codegen_node_schedule 方法，生成节点调度的代码
        return self.codegen_node_schedule(node_schedule, buf_accesses, numel, rnumel)

    # 静态方法：返回节点是否为 reduction 的提示
    @staticmethod
    def reduction_hint(node):
        assert node.is_reduction()  # 断言节点是 reduction 类型
        # 如果节点的所有依赖都是连续的，则返回 ReductionHint.INNER；否则返回节点数据的 reduction_hint 属性
        if all(
            dep.is_contiguous()
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint

    # 静态方法：判断是否可以使用 32 位索引
    @staticmethod
    def can_use_32bit_indexing(
        numel: sympy.Expr, buffers: Iterable[Union[ir.Buffer, ir.TensorBox]]
    ):
    ) -> bool:
        # 获取整数的最大值
        int_max = torch.iinfo(torch.int32).max
        # 获取图的大小变量的大小提示
        size_hint = V.graph.sizevars.size_hint
        # 检查图的大小变量的形状环境是否有提示
        has_hint = V.graph.sizevars.shape_env.has_hint

        def within_32bit(e):
            # 允许未提示的 e，只要我们仍然可以通过 ValueRanges 静态证明它仍在界限内
            if V.graph.sizevars.is_expr_static_and_true(e <= int_max):
                return True
            # 否则，提示必须存在并且在范围内
            return has_hint(e) and size_hint(e) <= int_max

        # 如果 numel 不在 32 位范围内，返回 False
        if not within_32bit(numel):
            return False

        # 使用 MultiOutputLayout 将创建一个考虑大小的缓冲区
        buf_sizes = [
            buf.get_layout().storage_size()
            for buf in buffers
            if not isinstance(buf.get_layout(), ir.MultiOutputLayout)
        ]

        # 检查所有 buf_sizes 中的大小是否都在 32 位范围内
        if not all(within_32bit(size) for size in buf_sizes):
            return False

        # 仅为 32 位索引安装保护，因为对于所有内容使用 64 位没有正确性问题
        V.graph.sizevars.guard_leq(numel, int_max)  # type: ignore[arg-type]
        for size in buf_sizes:
            V.graph.sizevars.guard_leq(size, int_max)  # type: ignore[arg-type]
        return True

    @classmethod
    def select_index_dtype(cls, node_schedule, numel, reduction_numel):
        # 收集所有使用的缓冲区名称
        buffer_names = set()
        for node in node_schedule:
            if not isinstance(node, scheduler.BaseSchedulerNode):
                continue

            # 更新缓冲区名称集合
            buffer_names.update(node.get_names())
            buffer_names.update(node.used_buffer_names())

        # 获取缓冲区对象
        def _get_buffer(name: str) -> Union[ir.Buffer, ir.TensorBox]:
            buf = V.graph.get_buffer(name)
            if buf is None:
                raise RuntimeError(f"Failed to find buffer matching name {name}")
            return buf

        # 根据缓冲区名称获取缓冲区对象列表
        buffers = [_get_buffer(name) for name in buffer_names]

        # 计算总 numel
        total_numel = numel * reduction_numel

        # 检查是否可以使用 32 位索引
        if SIMDScheduling.can_use_32bit_indexing(total_numel, buffers):
            return cls.int32_type
        # 默认返回 64 位索引类型
        return cls.int64_type
    def has_non_contiguous_pw_in_reduction_kernel(self, node_schedule, numel, rnumel):
        # 过滤出所有非连续的点积节点，这些节点不是启用或禁用归约节点，并且不是归约节点，且其组大小等于 numel * rnumel
        pointwise_nodes = list(
            filter(
                lambda n: n not in (EnableReduction, DisableReduction)
                and not n.is_reduction()
                and n.group[1][0] == numel * rnumel,
                node_schedule,
            )
        )
        # 遍历所有点积节点
        for node in pointwise_nodes:
            # 如果存在任何一个依赖不是连续的内存依赖或者依赖的索引不是整数，或者是最后维度的步幅不为1，则返回 True
            if not all(
                not isinstance(dep, MemoryDep)
                or dep.is_contiguous()
                or isinstance(dep.index, (sympy.Integer, int))
                or dep.stride1_for_last_dim()
                for dep in itertools.chain(
                    node.read_writes.reads, node.read_writes.writes
                )
            ):
                return True
        # 如果所有点积节点都符合条件，则返回 False
        return False

    def get_kernel_args(self, node_schedule, numel, reduction_numel):
        # 过滤出所有归约节点
        reductions = list(
            filter(
                lambda n: n not in (EnableReduction, DisableReduction)
                and n.is_reduction(),
                node_schedule,
            )
        )
        # 如果存在归约节点
        if len(reductions) > 0:
            # 获取所有归约节点的提示信息
            hints = [self.reduction_hint(n) for n in reductions]
            # 如果所有提示信息都相同，则取第一个提示信息作为归约提示值，否则取默认值
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT

            # 如果提示值为 ReductionHint.INNER 并且在归约内核中存在非连续的点积节点，则设置为默认提示值
            if (
                reduction_hint_val == ReductionHint.INNER
                and self.has_non_contiguous_pw_in_reduction_kernel(
                    node_schedule, numel, reduction_numel
                )
            ):
                reduction_hint_val = ReductionHint.DEFAULT
        else:
            # 如果不存在归约节点，则设置默认的归约提示值
            reduction_hint_val = ReductionHint.DEFAULT

        # 收集所有节点的突变
        mutations = set()
        for node in node_schedule:
            if hasattr(node, "get_mutations"):
                mutations.update(node.get_mutations())

        # 选择适当的索引数据类型
        index_dtype = self.select_index_dtype(node_schedule, numel, reduction_numel)

        # 返回归约提示值、突变集合和索引数据类型
        return reduction_hint_val, mutations, index_dtype

    def codegen_node_schedule(
        self, node_schedule, buf_accesses, numel, reduction_numel
    ):
        # 省略的函数体，未提供足够信息以添加注释
        pass
    # 使用给定的节点调度和内核生成代码
    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        # 定义一个函数，用于从节点序列中获取当前的减少节点（不包括 DisableReduction）
        def current_reduction_nodes(nodes):
            return itertools.takewhile(lambda n: n is not DisableReduction, nodes)

        # 使用给定的内核上下文
        with kernel:
            # 创建一个上下文管理器堆栈
            stack = contextlib.ExitStack()
            # 设置内核的最后使用情况为当前减少节点
            kernel.set_last_usage(current_reduction_nodes(node_schedule))
            # 用于存储所有索引信息的字典
            all_indexing = {}

            # 第一遍循环用于收集索引和确定就地更新
            for node in node_schedule:
                if node is DisableReduction:
                    # 进入禁用减少的上下文
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    # 关闭当前堆栈上下文
                    stack.close()
                else:
                    # 决定节点是否就地更新
                    node.decide_inplace_update()
                    # 拆分并设置节点范围，获取索引变量
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    # 更新所有索引字典，以节点参数的索引为键
                    all_indexing.update(
                        dict.fromkeys(
                            node._body.indexing_from_args(index_vars).values()
                        )
                    )

            # 完成索引的最终处理
            kernel.finalize_indexing(all_indexing.keys())

            # 第二遍循环用于生成代码
            for i, node in enumerate(node_schedule):
                if node is DisableReduction:
                    # 进入禁用减少的上下文
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    # 关闭当前堆栈上下文，并设置内核的最后使用情况为后续减少节点
                    stack.close()
                    kernel.set_last_usage(current_reduction_nodes(node_schedule[i:]))
                else:
                    # 使用分割的范围来处理索引数据类型强度减少
                    indexing_dtype_strength_reduction(node._body)
                    # 拆分并设置节点范围，获取索引变量
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    # 为节点生成代码
                    node.codegen(index_vars)
    ) -> Optional[str]:
        """
        Codegen a triton template

        If `only_gen_src_code` the src code will be returned instead of codegen'd into the wrapper
        """
        # 获取模板节点的尺寸信息 (numel, rnumel)，并确保 rnumel 等于 1
        _, (numel, rnumel) = template_node.group
        assert rnumel == 1
        # 生成核函数和渲染器对象
        kernel, render = template_node.node.make_kernel_render(template_node.node)
        # 进入核函数的上下文管理器
        with kernel:
            if not only_gen_src_code:
                # 如果不仅生成源代码，则标记模板节点和所有后续节点为运行状态
                for node in [template_node, *epilogue_nodes]:
                    node.mark_run()
            # 渲染部分代码片段
            partial_code = render()
            # 设置子图形体为 "<STORE_OUTPUT>"，并为每个后续节点生成代码
            with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                for node in epilogue_nodes:
                    node.codegen(kernel.split_and_set_ranges(node.get_ranges()))

        if not isinstance(partial_code, str):
            # 如果 partial_code 不是字符串，则调用其 finalize_hook 方法，标记 "<DEF_KERNEL>"
            partial_code.finalize_hook("<DEF_KERNEL>")
        # 在添加后续节点之后必须调用 finalize
        with V.set_kernel_handler(kernel):
            # 设置子图形体为 "<STORE_OUTPUT>"
            with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                if isinstance(partial_code, str):
                    # 如果 partial_code 是字符串，则将其设置为 src_code
                    src_code = partial_code
                else:
                    # 否则，调用 partial_code 的 finalize_hook 方法，标记 "<STORE_OUTPUT>"，并将其代码设置为 src_code
                    partial_code.finalize_hook("<STORE_OUTPUT>")
                    src_code = partial_code.code
            # 设置节点调度顺序
            node_schedule = [template_node, *epilogue_nodes]

            if config.benchmark_kernel:
                # 如果配置为基准测试核函数，则估算核函数的字节数量并生成网格参数
                num_gb = kernel.estimate_kernel_num_bytes() / 1e9
                grid_args = V.graph.sizevars.size_hints(kernel.call_sizes)
                assert kernel.meta is not None, "meta is None"
                # 通过核函数的网格函数和参数获取网格
                grid = kernel.grid_fn(*grid_args, kernel.meta)
                # 生成基准测试的源代码，包含导入语句、核函数源码以及性能评估代码
                src_code = (
                    f"{kernel.imports_for_benchmark_kernel()}\n"
                    f"{src_code}\n"
                    f"{kernel.codegen_kernel_benchmark(num_gb, grid).getvalue()}"
                )

            if only_gen_src_code:
                # 如果只生成源代码，则返回 src_code
                return src_code

            # 定义核函数的名称并返回其值
            kernel_name = self.define_kernel(src_code, node_schedule, kernel)

        # 为节点调度计划添加注释
        self.codegen_comment(node_schedule)
        # 调用核函数的名称，将模板节点传递给核函数调用
        kernel.call_kernel(kernel_name, template_node.node)
        # 更新图形移除的缓冲区和待移除的替换缓冲区
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        # 释放调度器的缓冲区
        self.scheduler.free_buffers()
        # 返回空值
        return None

    def codegen_sync(self):
        # 将设备操作的同步写入包装器代码
        V.graph.wrapper_code.writeline(V.graph.device_ops.synchronize())
    # 定义一个实例方法 `codegen_foreach`，用于生成 foreach 循环的代码
    def codegen_foreach(self, foreach_node):
        # 导入 Triton 中的 ForeachKernel 类
        from .triton_foreach import ForeachKernel
        
        # 使用 ForeachKernel 类的 horizontal_partition 方法对 foreach_node 的子内核节点进行水平分区
        for partitions_with_metadata in ForeachKernel.horizontal_partition(
            foreach_node.get_subkernel_nodes(), self
        ):
            # 创建一个新的 ForeachKernel 实例
            kernel = ForeachKernel()
            # 遍历每个分区及其元数据
            for nodes, tiled_groups, numel, rnumel in partitions_with_metadata:
                # 生成节点调度信息
                node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
                
                # 获取内核参数，包括缩减提示值、变异和索引数据类型
                (
                    reduction_hint_val,
                    mutations,
                    index_dtype,
                ) = self.get_kernel_args(node_schedule, numel, rnumel)

                # 创建子内核
                subkernel = kernel.create_sub_kernel(
                    *tiled_groups,
                    reduction_hint=reduction_hint_val,
                    mutations=mutations,
                    index_dtype=index_dtype,
                )

                # 使用子内核生成节点调度与内核代码
                self.codegen_node_schedule_with_kernel(
                    node_schedule,
                    subkernel,
                )

                # 设置子内核的内核处理程序
                with V.set_kernel_handler(subkernel):
                    # 对节点调度中的每个节点进行标记，表示可以执行
                    for node in node_schedule:
                        if node not in (EnableReduction, DisableReduction):
                            node.mark_run()
                
                # 将子内核移除的缓冲区添加到 V.graph.removed_buffers 中
                V.graph.removed_buffers |= subkernel.removed_buffers
                # 将子内核中需要移除的 inplaced 标记添加到 V.graph.inplaced_to_remove 中
                V.graph.inplaced_to_remove |= subkernel.inplaced_to_remove

            # 生成整个 kernel 的源代码
            src_code = kernel.codegen_kernel()
            # 定义 kernel 的名称，并为其定义
            kernel_name = self.define_kernel(src_code, [foreach_node], kernel)
            # 生成对 `foreach_node` 的注释
            self.codegen_comment([foreach_node])
            # 调用生成的 kernel
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)

        # 释放调度器中的缓冲区资源
        self.scheduler.free_buffers()

    # 静态方法修饰器，使用 functools 提供的 LRU 缓存，缓存上限为 32
    @staticmethod
    @functools.lru_cache(32)
    # 定义一个静态方法 candidate_tilings，接受一个节点对象作为参数
    def candidate_tilings(node):
        # 从节点对象中获取范围和减少范围的信息
        ranges, reduction_ranges = node.get_ranges()
        
        # 如果范围数量小于等于1，返回空元组
        if len(ranges) <= 1:
            return ()

        # 获取节点的逐点读写操作信息
        rw = node.pointwise_read_writes()
        
        # 断言读写范围变量的数量与范围数量相等
        assert len(rw.range_vars) == len(ranges)

        # dep_sources 包含所有的读和写依赖
        dep_sources = [rw.reads, rw.writes]
        
        # 断言所有依赖都是 MemoryDep 或 StarDep 类型
        assert all(
            isinstance(dep, (MemoryDep, StarDep))
            for dep in itertools.chain.from_iterable(dep_sources)
        )
        
        # 筛选出不在已移除缓冲区列表中且为 MemoryDep 类型的依赖
        deps = [
            dep
            for dep in itertools.chain.from_iterable(dep_sources)
            if dep.name not in V.graph.removed_buffers and isinstance(dep, MemoryDep)
        ]
        
        # 获取所有写操作的名称集合
        write_names = {dep.name for dep in rw.writes}

        # 初始化候选切片列表
        tilings: List[CandidateTiling] = []

        # 遍历每个依赖
        for dep in deps:
            # 获取依赖的步长提示
            strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
            
            # 断言步长数量与范围数量相等
            assert len(strides) == len(ranges)
            
            try:
                # 尝试找到第一个步长为1的索引位置
                split = strides.index(1) + 1
                
                # 如果 split 等于范围数量，跳过当前依赖
                if split == len(ranges):
                    continue
                
                # 如果 split 后的所有步长都为0，表示这不是一个真正的切片
                if all(s == 0 for s in strides[split:]):
                    continue

            except ValueError:
                continue
            
            # 根据切片信息简化后的大小分组
            tiled_groups = (
                V.graph.sizevars.simplify(sympy_product(ranges[:split])),
                V.graph.sizevars.simplify(sympy_product(ranges[split:])),
            )
            
            # 根据元素数量对切片进行评分
            score = V.graph.sizevars.size_hint(
                sympy_product(
                    size for size, stride in zip(ranges, strides) if stride != 0
                )
            )
            
            # 如果依赖的名称在写操作名称集合中，将评分加倍
            if dep.name in write_names:
                score *= 2
            
            # 如果第一个切片组是一个好的大小，将评分加倍
            if CandidateTiling.is_good_size(tiled_groups[0]):
                score *= 2
            
            # 如果第二个切片组是一个好的大小，将评分加倍
            if CandidateTiling.is_good_size(tiled_groups[1]):
                score *= 2
            
            # 如果评分减去范围和减少范围的乘积后的大小提示大于等于0，将切片信息添加到候选切片列表中
            if (
                V.graph.sizevars.size_hint(
                    score - sympy_product(itertools.chain(ranges, reduction_ranges))
                )
                >= 0
            ):
                tilings.append(CandidateTiling(tiled_groups, score, dep.name))
        
        # 返回候选切片列表
        return tilings

    @classmethod
    # 类方法，用于选择如何分割内核
    @classmethod
    def select_tiling(cls, node_schedule, numel, reduction_numel=sympy.Integer(1)):
        """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`
        """

        # 如果reduction_numel不等于1或者配置中的最大瓦片数小于等于1，则直接返回(numel, reduction_numel)
        if reduction_numel != 1 or config.triton.max_tiles <= 1:
            # TODO(jansel): should we tile reductions?
            # 在这里进行性能提示，如果步长为1的维度未被减少
            if perf_hint_log.level <= logging.WARNING:
                # 遍历节点调度中的节点，过滤出启用了减少操作的节点
                for node in EnableReduction.filter(node_schedule):
                    # 如果某节点有候选的瓦片方案，则记录性能提示日志并中断
                    if len(cls.candidate_tilings(node)) > 0:
                        perf_hint_log.info("reduction over non-contiguous dims")
                        break
            # 返回(numel, reduction_numel)
            return (numel, reduction_numel)

        # 初始化一个集合，用于记录已经处理过的瓦片名字
        seen_names = set()
        # 初始化一个Counter，用于计数候选瓦片的出现次数
        candidate_tiles: Counter[Any] = collections.Counter()

        # 遍历节点调度中的节点，过滤出启用了减少操作的节点
        for node in EnableReduction.filter(node_schedule):
            # 遍历节点的候选瓦片方案
            for tiling in cls.candidate_tilings(node):
                # 如果该瓦片名字已经处理过，则跳过
                if tiling.name in seen_names:
                    continue
                # 将瓦片名字加入已处理集合
                seen_names.add(tiling.name)
                # 将该瓦片方案的得分累加到Counter中
                candidate_tiles[tiling.tiling] += tiling.score

        # 按照出现次数最多的顺序排列瓦片方案
        ranked_tilings = [tiling for tiling, score in candidate_tiles.most_common()]

        # 如果配置中的最大瓦片数大于等于3，则考虑添加第三个维度的瓦片
        if config.triton.max_tiles >= 3:
            # 考虑添加第三个维度的瓦片选择
            for i in range(1, len(ranked_tilings)):
                a0, a1 = ranked_tilings[0]
                b0, b1 = ranked_tilings[i]
                # 如果a1是b1的倍数，则添加该三维瓦片选择
                if V.graph.sizevars.size_hint(a1 - b1) == 0:
                    continue
                if V.graph.sizevars.size_hint(a1 - b1) < 0:
                    a0, a1 = ranked_tilings[i]
                    b0, b1 = ranked_tilings[0]
                assert V.graph.sizevars.size_hint(a1 - b1) > 0
                if V.graph.sizevars.statically_known_multiple_of(a1, b1):
                    tiling = (a0, FloorDiv(a1, b1), b1)
                    ranked_tilings = [tiling] + ranked_tilings
                    break  # 只添加一种选择

        # 如果有多于一种瓦片方案，则记录性能提示日志
        if len(ranked_tilings) > 1:
            perf_hint_log.info("possibly bad tiling: %s", ranked_tilings)

        # 遍历排序好的瓦片方案
        for tiled_groups in ranked_tilings:
            # 在当前瓦片方案后添加减少数量，组成新的瓦片组合
            new_groups = (*tiled_groups, reduction_numel)
            # 对于调度中的每个节点，检查新的瓦片组合是否兼容
            if all(
                SIMDKernel.is_compatible(new_groups, node.get_ranges())
                for node in node_schedule
                if isinstance(node, scheduler.SchedulerNode)
            ):
                return new_groups

        # 如果没有找到合适的瓦片组合，则返回(numel, reduction_numel)
        return (numel, reduction_numel)

    # 空的flush方法，未实现具体功能
    def flush(self):
        pass
    # 返回 False，表示未准备好刷新
    def ready_to_flush(self) -> bool:
        return False

    # 从节点生成内核代码
    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False):
        # 定义用于持有最后使用情况的数据类
        @dataclasses.dataclass
        class LastUsageHolder:
            n: Any  # 节点
            last_usage: Any  # 最后使用情况

            def __del__(self):
                self.n.last_usage = self.last_usage  # 在对象销毁时更新节点的最后使用情况

        # 创建节点列表中每个节点的最后使用情况持有者
        last_usage_holders = [LastUsageHolder(n, n.last_usage) for n in nodes]

        # 清空所有节点的最后使用情况集合。可能导致更积极的 'evict_last'，但应该没问题。
        for n in nodes:
            n.last_usage = set()

        # 如果第一个节点不是模板节点
        if not nodes[0].is_template():
            # 找到具有最大 'is_reduction' 值的节点，并获取其分组中的数据
            _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
            # 生成节点调度表
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
            # 选择节点的切片组
            tiled_groups = self.select_tiling(node_schedule, numel, rnumel)
            # 获取内核参数：减少提示值、变异和索引数据类型
            reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
                node_schedule, numel, rnumel
            )
            # 创建内核对象
            kernel = self.kernel_type(
                *tiled_groups,
                reduction_hint=reduction_hint_val,
                mutations=mutations,
                index_dtype=index_dtype,
            )
            # 使用内核生成节点调度表的代码
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)
            # 使用内核处理器生成内核代码
            with config.patch(
                "benchmark_kernel", benchmark_kernel
            ), V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
        else:
            # 如果第一个节点是模板节点，使用模板代码生成器生成源代码
            template_node = nodes[0]
            epilogue_nodes = nodes[1:]
            with config.patch("benchmark_kernel", benchmark_kernel):
                src_code = self.codegen_template(
                    template_node, epilogue_nodes, only_gen_src_code=True
                )

        # 替换源代码中的占位符 'KERNEL_NAME' 为 'triton_'，并返回生成的源代码
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), "triton_")
        return src_code

    # 生成代码的注释，但函数体为空
    def codegen_comment(self, node_schedule):
        pass

    # 定义内核的接口函数，但抛出未实现的错误
    def define_kernel(self, src_code, node_schedule, kernel):
        raise NotImplementedError
# 使用 dataclasses 模块创建一个候选平铺类，用于表示一个平铺及其评分
@dataclasses.dataclass
class CandidateTiling:
    tiling: Tuple[sympy.Expr, sympy.Expr]  # 平铺表达式的元组
    score: int  # 评分，分数越高越好
    name: Optional[str] = None  # 可选的名称字段

    @staticmethod
    def is_good_size(s):
        """用于一些大小的启发式方法，用于增加某些大小的评分"""
        # 获取图的大小提示，并对大小进行调整
        s = V.graph.sizevars.size_hint(s)
        # 返回大小是否符合要求的布尔值
        return s >= 32 and (s % 32 == 0)


# 表示禁用规约的标记类，用于调用 kernel.disable_reduction()
class DisableReduction:
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """


# 表示启用规约的标记类，用于结束 DisableReduction 块
class EnableReduction:
    @staticmethod
    def filter(node_schedule):
        """
        从节点调度中获取节点，跳过 DisableReduction 块中的节点。
        """
        disabled = False  # 标记是否处于禁用规约块内
        for node in node_schedule:
            if node in (EnableReduction, DisableReduction):
                # 不对主规约循环外的内容进行平铺处理
                disabled = node is DisableReduction
            elif disabled:
                pass  # 如果处于禁用规约状态，则跳过该节点
            else:
                yield node  # 如果未处于禁用规约状态，则产出该节点


# 表示无法进行分裂的异常类
class CantSplit(Exception):
    pass
```