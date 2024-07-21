# `.\pytorch\torch\_inductor\codegen\memory_planning.py`

```
# mypy: allow-untyped-defs
# 使用未类型化的定义来允许类型推断

from __future__ import annotations
# 允许在类定义中使用自身类作为类型注释

import collections
# 导入集合模块，用于操作集合数据类型

import dataclasses
# 导入dataclasses模块，用于创建不可变数据对象

import itertools
# 导入itertools模块，用于创建迭代器的函数

import pprint
# 导入pprint模块，用于格式化打印数据结构

from typing import Any, Dict, Iterable, List, Optional, Protocol
# 导入类型注释模块，用于指定变量和函数参数的类型

import sympy
# 导入sympy模块，用于符号计算

import torch
# 导入torch模块，用于深度学习任务

from .. import config, ir
# 从相对路径导入config和ir模块

from ..utils import _align, align, cache_on_self, CachedMethod, IndentedBuffer
# 从相对路径导入_align, align, cache_on_self, CachedMethod, IndentedBuffer函数

from ..virtualized import V
# 从相对路径导入V模块

from .wrapper import (
    AllocateLine,
    FreeIfNotReusedLine,
    MemoryPlanningLine,
    NullLine,
    ReuseLine,
)
# 从当前路径导入wrapper模块中的AllocateLine, FreeIfNotReusedLine, MemoryPlanningLine, NullLine, ReuseLine类

@dataclasses.dataclass
# 使用dataclass装饰LiveRange类，以便自动创建__init__方法和__repr__方法
class LiveRange:
    """
    A range where a given tensor is live.  Begin and end are both counters
    representing points in the program of grouped memory operations.
    Begin is inclusive, end is exclusive.

    Invariant: begin <= end
    """
    begin: float  # int | +/-inf
    # 开始位置，是一个浮点数，表示内存操作计划中的起始点
    end: float  # int | +/-inf
    # 结束位置，是一个浮点数，表示内存操作计划中的结束点

    def contains(self, other: LiveRange):
        """Is other entirely within self"""
        # 判断另一个LiveRange对象other是否完全包含在当前对象self中
        return self.begin <= other.begin and other.end <= self.end

    def join(self, other: LiveRange):
        """Combine two ranges using a union operation"""
        # 将两个LiveRange对象self和other合并成一个更大的范围
        return LiveRange(min(self.begin, other.begin), max(self.end, other.end))

    def __len__(self):
        # 返回当前LiveRange对象的长度
        return self.end - self.begin

class LiveRanges:
    """
    A collection of LiveRange regions, allowing for non-contiguous
    live regions.

    Invariant: LiveRanges.ranges is in sorted order and non-overlapping
    """
    def __init__(self, ranges: Iterable[LiveRange]):
        # 初始化函数，接受一个LiveRange对象的可迭代集合作为参数
        ranges = [*sorted(ranges, key=lambda x: x.begin)]
        # 将ranges按照begin属性排序，并转换成列表
        self.ranges = ranges[:1]
        # 初始化LiveRanges对象的ranges属性为排序后的第一个LiveRange对象
        for r in ranges[1:]:
            assert self.ranges[-1].begin <= r.begin
            # 确保前一个LiveRange对象的begin属性小于等于当前LiveRange对象的begin属性
            if self.ranges[-1].end >= r.begin:
                self.ranges[-1] = LiveRange.join(self.ranges[-1], r)
                # 如果当前LiveRange对象与前一个LiveRange对象有重叠，则合并它们
            else:
                self.ranges.append(r)
                # 否则直接将当前LiveRange对象加入到ranges列表中

    def overlaps(self, other: LiveRanges):
        """Check if any pair of ranges in self and other overlap"""
        # 检查当前LiveRanges对象与另一个LiveRanges对象other是否有任何重叠的区间
        left = collections.deque(self.ranges)
        right = collections.deque(other.ranges)
        while left and right:
            if left[0].begin > right[0].begin:
                left, right = right, left
            assert left[0].begin <= right[0].begin
            if left[0].end > right[0].begin:
                return True
            left.popleft()
        return False

    @property
    def begin(self):
        # 返回当前LiveRanges对象中第一个LiveRange对象的begin属性
        return self.ranges[0].begin

    @property
    def end(self):
        # 返回当前LiveRanges对象中最后一个LiveRange对象的end属性
        return self.ranges[-1].end

    def __repr__(self):
        # 返回当前LiveRanges对象的字符串表示形式
        return f"{self.__class__.__name__}([{', '.join(map(repr, self.ranges))}])"


class AllocationTreeNode:
    """
    Abstract base class for nodes in allocation pool.
    """

    def allocate(self, block: Allocation, is_last: bool) -> bool:
        """
        Try to assign block to a memory location in this bool.  Return True if
        an assignment was made.
        """
        # 尝试将block分配给当前AllocationTreeNode对象中的一个内存位置，并返回是否成功分配的布尔值
        return False
        # 默认情况下，分配失败，返回False
    # 返回此对象以下所有对象的 LiveRanges 的聚合结果
    def get_live_ranges(self) -> LiveRanges:
        """Aggregate LiveRanges for all objects below this in tree"""
        raise NotImplementedError

    # 返回示例输入所使用的字节数
    def get_size_hint(self) -> int:
        """Number of bytes used for example inputs"""
        raise NotImplementedError

    # 返回运行时需要的字节数表达式
    def get_symbolic_size(self) -> sympy.Expr:
        """Number of bytes needed at runtime"""
        raise NotImplementedError

    # 在所有分配完成后调用，返回 AllocationTreeNode 对象
    def finalize(self, pool, offset) -> AllocationTreeNode:
        """Called after all allocations have been made"""
        return self

    # 判断对象是否为空，总是返回 False
    def is_empty(self):
        return False
# 使用 dataclasses 装饰器创建 Allocation 类，表示分配给分配池中给定节点的内存。
@dataclasses.dataclass
class Allocation(AllocationTreeNode):
    """
    Represents memory allocated to a given node in the allocation pool.
    """

    # 分配的缓冲区节点
    node: ir.Buffer
    # 活跃范围
    live_range: LiveRange
    # 大小的暗示
    size_hint: int
    # 符号大小表达式
    symbolic_size: sympy.Expr
    # 是否已分配的标志
    allocated: bool = False
    # 分配池（可选）
    pool: Optional[AllocationPool] = None
    # 偏移量（可选）
    offset: Optional[sympy.Expr] = None

    # 获取设备属性
    @property
    def device(self):
        return self.node.get_device()

    # 获取活跃范围列表
    def get_live_ranges(self):
        return LiveRanges([self.live_range])

    # 获取大小的暗示
    def get_size_hint(self):
        return self.size_hint

    # 获取符号大小表达式
    def get_symbolic_size(self):
        return self.symbolic_size

    # 标记为已分配状态
    def mark_allocated(self):
        assert not self.allocated
        self.allocated = True

    # 最终化分配，设置分配池和偏移量
    def finalize(self, pool, offset):
        assert self.pool is None and self.offset is None
        self.pool = pool
        self.offset = offset
        return self

    # 从分配池生成分配代码
    def codegen_alloc_from_pool(self, wrapper):
        assert self.pool
        node = self.node
        shape = tuple(node.get_size())
        stride = tuple(node.get_stride())
        return wrapper.codegen_alloc_from_pool(
            self.pool.name, self.offset, node.get_dtype(), shape, stride
        )

    # 返回对象的字符串表示形式
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node={self.node.get_name()}, "
            f"live_range={self.live_range}, "
            f"size_hint={self.size_hint}, "
            f"symbolic_size={self.symbolic_size}, "
            f"pool={self.pool.name if self.pool else None}, "
            f"offset={self.offset})"
        )


# 使用 dataclasses 装饰器创建 Empty 类，表示分配池中空闲空间的占位符。
@dataclasses.dataclass
class Empty(AllocationTreeNode):
    """
    Placeholder to represent empty space in the allocation pool.
    Only exists to get the size_hint correct in parent nodes.
    """

    # 大小的暗示
    size_hint: int

    # 获取空活跃范围列表
    def get_live_ranges(self):
        return LiveRanges([])

    # 获取大小的暗示
    def get_size_hint(self):
        return self.size_hint

    # 获取符号大小为零
    def get_symbolic_size(self):
        return 0

    # 判断是否为空
    def is_empty(self):
        return True


# 定义 MemorySplitProtocol 协议，定义了获取活跃范围、大小暗示和符号大小的方法
class MemorySplitProtocol(Protocol):
    get_live_ranges: CachedMethod[[], LiveRanges]
    get_size_hint: CachedMethod[[], int]
    get_symbolic_size: CachedMethod[[], sympy.Expr]

    # 分配方法，必须被子类实现
    def _allocate(self, block: Allocation, is_last: bool) -> bool:
        ...


# ClearCacheOnAllocateMixin 类继承自 MemorySplitProtocol 类，
# 提供了清除缓存的辅助方法，用于 get_live_ranges、get_size_hint 和 get_symbolic_size。
class ClearCacheOnAllocateMixin(MemorySplitProtocol):
    """
    Helper to assist in caching get_live_ranges, get_size_hint, and
    get_symbolic_size.
    """

    # 分配方法，调用 _allocate 方法，如果成功分配则清除缓存
    def allocate(self, block: Allocation, is_last: bool):
        is_allocated = self._allocate(block, is_last)
        if is_allocated:
            self.clear_cache()
        return is_allocated

    # 清除缓存的方法
    def clear_cache(self):
        self.get_live_ranges.clear_cache(self)
        self.get_size_hint.clear_cache(self)
        self.get_symbolic_size.clear_cache(self)


# 使用 dataclasses 装饰器创建 TemporalSplit 类，继承 ClearCacheOnAllocateMixin 和 AllocationTreeNode 类，
# 用于表示在分配池中不重叠的分配列表。
@dataclasses.dataclass
class TemporalSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains a list of allocations not overlapping in LiveRanges.
    """

    # 代码块，用于分配
    def _allocate(self, block: Allocation, is_last: bool) -> bool:
        ...
    """
    Invariant: no pair (a,b) in self.allocations will have:
         a.get_live_ranges().overlaps(b.get_live_ranges())
    """

    allocations: List[AllocationTreeNode]  # 声明一个成员变量 allocations，类型为 AllocationTreeNode 的列表

    def _allocate(self, block: Allocation, is_last: bool):
        # 获取分配的槽大小和块大小的提示信息
        slot_size = self.get_size_hint()
        block_size = block.get_size_hint()
        # 如果不是最后一个位置且块大小超过槽大小，则返回 False，表示不符合条件
        if not is_last and block_size > slot_size:
            return False  # doesn't fit

        # 获取当前块的活跃范围
        block_live = block.get_live_ranges()
        # 查找与当前块存在重叠活跃范围的已分配节点
        overlapping = [
            s for s in self.allocations if s.get_live_ranges().overlaps(block_live)
        ]
        # 如果存在多个重叠节点，则暂时无法分配当前块
        if len(overlapping) > 1:
            # TODO(jansel): 在此处可以尝试通过空间合并重叠的节点
            return False
        # 如果存在一个重叠节点，则将当前块分配给该节点
        elif len(overlapping) == 1:
            return overlapping[0].allocate(block, is_last)
        else:
            # 标记当前块已分配
            block.mark_allocated()

            # 如果只有一个空节点，且当前节点是最后一个，则移除该空节点
            if len(self.allocations) == 1 and isinstance(self.allocations[-1], Empty):
                self.allocations.pop()

            # 根据块大小和槽大小的关系进行分配
            if slot_size == block_size:
                # 完美匹配，直接将块添加到分配列表中
                self.allocations.append(block)
            elif slot_size > block_size:
                # 槽大小大于块大小，创建空间分裂节点，并将块添加到分配列表中
                self.allocations.append(
                    SpatialSplit.create(block, slot_size - block_size)
                )
            else:  # 如果需要扩展此分配
                assert is_last
                # 将当前所有分配节点分别与块大小和槽大小的差额创建空间分裂节点，然后将当前块添加到分配列表中
                self.allocations = [
                    *(
                        SpatialSplit.create(a, block_size - slot_size)
                        for a in self.allocations
                    ),
                    block,
                ]
            return True

    @cache_on_self
    def get_live_ranges(self) -> LiveRanges:
        # 返回所有已分配节点的活跃范围的联合
        return LiveRanges(
            itertools.chain.from_iterable(
                x.get_live_ranges().ranges for x in self.allocations
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        # 如果没有已分配节点，则返回 0
        if not self.allocations:
            return 0
        # 返回所有已分配节点中大小提示的最大值
        return max(x.get_size_hint() for x in self.allocations)

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        # 如果没有已分配节点，则返回 0
        if not self.allocations:
            return 0  # type: ignore[return-value]
        # 返回所有已分配节点中符号大小的最大值
        return sympy.Max(*[x.get_symbolic_size() for x in self.allocations])

    def is_empty(self):
        # 检查是否只有一个空节点
        return len(self.allocations) == 1 and self.allocations[0].is_empty()

    def finalize(self, pool, offset):
        # 对所有已分配节点进行最终处理，将其池化和偏移
        self.allocations = [block.finalize(pool, offset) for block in self.allocations]
        # 清除缓存
        self.clear_cache()
        # 如果只有一个已分配节点，则直接返回该节点
        if len(self.allocations) == 1:
            return self.allocations[0]
        # 否则返回整个对象
        return self
@dataclasses.dataclass
class SpatialSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains two allocations, left and right, that do not overlap in space.
    Right will be allocated immediately after left in memory.
    """

    left: TemporalSplit  # 左子树，包含时间分割的节点
    right: TemporalSplit  # 右子树，包含时间分割的节点

    @staticmethod
    def create(left, extra_space):
        assert isinstance(left, AllocationTreeNode)  # 确保 left 是 AllocationTreeNode 的实例
        assert isinstance(extra_space, int) and extra_space >= 1  # 确保 extra_space 是整数且大于等于1
        return SpatialSplit(TemporalSplit([left]), TemporalSplit([Empty(extra_space)]))

    def _allocate(self, block: Allocation, is_last: bool):
        # 尝试在左子树分配，若失败则在右子树分配
        return self.left.allocate(block, False) or self.right.allocate(block, is_last)

    @cache_on_self
    def get_live_ranges(self):
        # 返回左右子树的活跃范围的迭代器的合并
        return LiveRanges(
            itertools.chain(
                self.left.get_live_ranges().ranges, self.right.get_live_ranges().ranges
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        # 返回左子树大小的对齐值加上右子树大小的提示值
        return _align(self.left.get_size_hint()) + self.right.get_size_hint()

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        # 返回左子树符号大小的对齐值加上右子树符号大小的表达式
        return align(self.left.get_symbolic_size()) + self.right.get_symbolic_size()

    def finalize(self, pool, offset):
        # 完成对池中分配的最终操作
        self.left = self.left.finalize(pool, offset)  # 最终化左子树
        self.right = self.right.finalize(
            pool, offset + align(self.left.get_symbolic_size())
        )  # 最终化右子树，偏移值为左子树符号大小的对齐值加上偏移
        self.clear_cache()  # 清除缓存
        if self.right.is_empty():  # 若右子树为空
            return self.left  # 返回左子树
        return self  # 否则返回自身


@dataclasses.dataclass
class AllocationPool:
    """
    Represents a pool of allocations that will be generated by a single
    call to torch.empty.
    """

    device: torch.device  # 分配池的设备
    root: TemporalSplit  # 根节点，包含时间分割的节点
    can_expand: bool = True  # 是否可以扩展
    restrict_live_range: Optional[LiveRange] = None  # 限制活跃范围的可选参数
    name: Optional[str] = None  # 名称的可选参数
    names_to_del: List[str] = dataclasses.field(default_factory=list)  # 待删除名称的列表
    creation_cache: Dict[str, str] = dataclasses.field(default_factory=dict)  # 创建缓存的字典

    def allocate(self, block: Allocation, is_last: bool):
        if self.restrict_live_range and not self.restrict_live_range.contains(
            block.live_range
        ):
            return False

        is_last = self.can_expand and is_last  # 若可以扩展且为最后分配，则设置为真
        if self.root.allocate(block, is_last):  # 尝试在根节点分配
            return True

        if is_last:  # 若为最后分配
            return self.allocate_at_end(block)  # 在末尾分配

        return False

    def allocate_at_end(self, block):
        block.mark_allocated()  # 标记块已分配
        self.root = TemporalSplit([SpatialSplit(self.root, TemporalSplit([block]))])  # 在根节点末尾分配
        return True

    def finalize(self, name):
        assert not self.name  # 确保没有名称
        self.name = name  # 设置名称
        self.names_to_del.append(name)  # 将名称添加到待删除列表
        self.root.finalize(self, 0)  # 最终化根节点的分配
    # 定义一个方法，用于生成创建代码，接受两个参数：wrapper和code
    def codegen_create(self, wrapper, code: IndentedBuffer):
        # 断言确保对象有名称
        assert self.name
        # 获取符号化大小
        nbytes = self.root.get_symbolic_size()
        # 遍历根对象的分配块
        for block in self.root.allocations:
            # 检查是否为分配对象，并且大小与符号化大小相等
            if isinstance(block, Allocation) and nbytes == block.get_symbolic_size():
                # 优化：合并第一个分配和池创建
                node = block.node
                # 生成分配代码并写入到code中
                code.writeline(
                    wrapper.make_allocation(
                        self.name,
                        device=self.device,
                        dtype=node.get_dtype(),
                        shape=tuple(node.get_size()),
                        stride=tuple(node.get_stride()),
                    )
                )
                # 缓存创建的块，使用block来自池的分配代码作为键，名称作为值
                self.creation_cache[block.codegen_alloc_from_pool(wrapper)] = self.name
                return  # 结束方法
        else:
            # 如果没有找到相符的分配块，生成默认的分配代码
            code.writeline(
                wrapper.make_allocation(
                    self.name,
                    device=self.device,
                    dtype=torch.uint8,
                    shape=(nbytes,),
                    stride=(1,),
                )
            )

    # 定义一个方法，用于生成销毁代码，接受两个参数：wrapper和code
    def codegen_destroy(self, wrapper, code: IndentedBuffer):
        # 生成释放名称列表中对象的代码并写入到code中
        code.writeline(wrapper.make_free_by_names(self.names_to_del))

    # 定义相等运算符重载方法，用于比较两个对象是否为同一实例
    def __eq__(self, other):
        return self is other

    # 定义哈希方法，返回对象的ID作为哈希值
    def __hash__(self):
        return id(self)
@dataclasses.dataclass
class AllocationPools:
    """
    Collection of many AllocationPool objects grouped by device.
    """

    # 设备到分配池列表的映射字典，默认为空字典
    device_to_pools: Dict[torch.device, List[AllocationPool]] = dataclasses.field(
        default_factory=dict
    )

    def get_pools(self, block):
        # 如果块所在设备不在device_to_pools中，初始化一个空列表
        if block.device not in self.device_to_pools:
            self.device_to_pools[block.device] = []
        # 返回指定设备的分配池列表
        return self.device_to_pools[block.device]

    def allocate(self, block: Allocation):
        # 获取块所在设备的分配池列表
        pools = self.get_pools(block)

        # 遍历设备的分配池
        for pool in pools:
            # 如果在当前池中成功分配了块，则返回
            if pool.allocate(block, is_last=pool is pools[-1]):
                return

        # 如果所有池都已满，创建一个新的分配池
        pools.append(
            AllocationPool(
                block.device,
                TemporalSplit([block]),
                can_expand=config.memory_pool != "none",
            )
        )
        # 标记块已分配
        block.mark_allocated()

    def allocate_output(self, block: Allocation):
        """Outputs get different pools so memory gets freed properly"""
        # 获取块所在设备的分配池列表
        pools = self.get_pools(block)
        # 如果存在池并且配置允许在outputs或combined中进行内存池分配
        if pools and config.memory_pool in ("outputs", "combined"):
            # 在最后一个池中末尾分配块
            pools[-1].allocate_at_end(block)
        else:
            # 否则创建一个新的分配池，并标记块已分配
            block.mark_allocated()
            pools.append(
                AllocationPool(
                    block.device,
                    TemporalSplit([block]),
                    can_expand=config.memory_pool == "combined",
                )
            )

    def finalize(self):
        """Called at the end of allocation process"""
        # 遍历所有设备的所有分配池，进行最终处理
        for i, pool in enumerate(
            itertools.chain.from_iterable(self.device_to_pools.values())
        ):
            pool.finalize(f"pool{i}")

    def pprint(self):
        # 打印所有设备的所有分配池的信息
        for pool in itertools.chain.from_iterable(self.device_to_pools.values()):
            print()
            print(pool.name)
            print(pool.root.get_live_ranges())
            pprint.pprint(pool.root)


class BufferGroup:
    """
    Due to inplace reuse an allocated buffer can have many names.
    This tracks these collections of buffers sharing underlying memory.
    """

    def __init__(self, node: ir.Buffer):
        # 初始化缓冲组，以IR节点作为参数
        self.node = node
        # 初始缓冲组的名称列表
        self.names = [node.get_name()]
        # 是否为输出
        self.is_output = False
        # 分配信息，初始为None
        self.allocation: Optional[Allocation] = None
        # 生命周期范围，初始为无穷大到负无穷
        self.live_range = LiveRange(float("inf"), -float("inf"))

    def update_usage(self, timestep: int):
        """Expand self.live_range to include timestep"""
        # 扩展生命周期范围以包括给定的时间步
        self.live_range = LiveRange(
            min(timestep, self.live_range.begin),
            max(timestep, self.live_range.end),
        )

    def sym_nbytes(self):
        # 返回符号字节数，由布局大小和数据类型大小计算得出
        return self.node.get_layout().storage_size() * self.node.get_dtype().itemsize
    # 创建分配（allocation）的方法，用于为对象分配内存空间
    def make_allocation(self):
        # 断言确保分配操作仅执行一次，避免重复分配内存
        assert not self.allocation, "multiple allocations"
        # 断言确保 self.live_range.begin 是整数，表示存活范围已计算
        assert isinstance(self.live_range.begin, int), "live ranges not computed"
        # 获取符号的字节数
        nbytes = self.sym_nbytes()
        
        # 计算大小的提示值，如果遇到未支持的 SymInt，将使用后备值。长期计划是改进未支持 SymInt 的大小提示。
        size_hint = V.graph.sizevars.size_hint(nbytes, fallback=64)
        
        # 创建 Allocation 对象，用于存储分配的相关信息
        self.allocation = Allocation(
            self.node,
            self.live_range,
            size_hint=size_hint,
            symbolic_size=nbytes,
        )

    # 返回对象的字符串表示形式，包括类名、名称、是否为输出、存活范围等信息
    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.names!r}, is_output={self.is_output}, "
            f"live_range={self.live_range}"
        )
@dataclasses.dataclass
class PoolMemoryPlanningLine(MemoryPlanningLine):
    """Abstract base class for {Alloc,Dealloc}FromPoolLine"""

    group: BufferGroup
    timestep: Optional[int] = None

    @property
    def node(self):
        return self.group.node



@dataclasses.dataclass
class AllocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to AllocationLine, but takes memory from a pool"""

    is_first_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        allocation = self.group.allocation
        assert allocation and allocation.pool
        pool = allocation.pool
        name = self.node.get_name()

        if self.is_first_pool_usage:
            # Generate code to create pool if it's the first usage
            pool.codegen_create(self.wrapper, code)

        # Extend list of names to delete from the pool
        pool.names_to_del.extend(self.group.names)

        # Generate allocation code and handle caching
        alloc_from_pool = allocation.codegen_alloc_from_pool(self.wrapper)
        if alloc_from_pool in pool.creation_cache:
            # Use cached allocation if available
            code.writeline(
                self.wrapper.make_tensor_alias(
                    name, pool.creation_cache[alloc_from_pool], "alloc"
                )
            )
        else:
            # Cache allocation and generate code to declare variable
            pool.creation_cache[alloc_from_pool] = name
            code.writeline(
                f"{self.wrapper.declare}{name} = {alloc_from_pool}{self.wrapper.ending}"
            )



@dataclasses.dataclass
class DeallocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to FreeIfNotReusedLine, but takes memory from a pool"""

    is_last_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        if self.is_last_pool_usage:
            assert self.group.allocation and self.group.allocation.pool
            # Generate code to destroy pool if it's the last usage
            self.group.allocation.pool.codegen_destroy(self.wrapper, code)



@dataclasses.dataclass
class MemoryPlanner:
    """
    Coordination object to run memory planning passes during wrapper
    codegen.
    """

    wrapper: Any
    pools: AllocationPools = dataclasses.field(default_factory=AllocationPools)
    buffer_groups: Optional[List[BufferGroup]] = None

    def plan(self, lines: List[Any]) -> List[Any]:
        """Call all the memory planning passes in sequence"""
        lines = [*lines]
        self.drop_removed_buffers(lines)
        self.convert_to_pool_lines(lines)
        self.compute_live_ranges(lines)
        self.allocate_groups()
        self.mark_first_last_usage(lines)
        return lines

    def drop_removed_buffers(self, lines):
        """
        Replace any memory planning lines in V.graph.removed_buffers with NullLine
        """
        # drop any removed buffers
        for i, line in enumerate(lines):
            if isinstance(line, (AllocateLine, FreeIfNotReusedLine, ReuseLine)):
                if line.node.get_name() in V.graph.removed_buffers:
                    # Replace with NullLine if buffer is removed
                    lines[i] = NullLine(self.wrapper)
    def compute_buffer_groups(self, lines):
        """
        Populates self.buffer_groups with BufferGroup objects that join
        allocations with common storage (due to inplace reuse) into a
        single object.
        """
        # 创建一个空字典，用于存储节点名到BufferGroup对象的映射关系
        name_to_group = {}
        
        # 遍历传入的lines列表
        for line in lines:
            # 如果当前行是AllocateLine类型的实例
            if isinstance(line, AllocateLine):
                # 获取节点的名称
                name = line.node.get_name()
                # 确保节点名称在name_to_group中不存在，防止重复
                assert name not in name_to_group
                # 创建一个新的BufferGroup对象，并添加到name_to_group字典中
                name_to_group[name] = BufferGroup(line.node)
            
            # 如果当前行是ReuseLine类型的实例
            elif isinstance(line, ReuseLine):
                # 获取旧节点和新节点的名称
                old_name = line.node.get_name()
                new_name = line.reused_as.get_name()
                # 确保新节点名称在name_to_group中不存在
                assert new_name not in name_to_group
                # 如果旧节点在name_to_group中已存在
                if old_name in name_to_group:
                    # 将新节点名称添加到旧节点对应的BufferGroup的names列表中
                    name_to_group[old_name].names.append(new_name)
                    # 将新节点也映射到旧节点对应的BufferGroup对象上
                    name_to_group[new_name] = name_to_group[old_name]

        # 获取图的输出节点名称集合
        outputs = set(V.graph.get_output_names())
        # 使用字典推导式获取所有唯一的BufferGroup对象列表
        unique_groups = [*{id(g): g for g in name_to_group.values()}.values()]
        
        # 遍历唯一的BufferGroup对象列表
        for group in unique_groups:
            # 检查当前组中是否有节点是图的输出节点
            group.is_output = any(x in outputs for x in group.names)

        # 确保self.buffer_groups属性为空
        assert self.buffer_groups is None
        # 将唯一的BufferGroup对象列表赋值给self.buffer_groups属性
        self.buffer_groups = unique_groups
        # 返回name_to_group字典，这个字典包含了所有节点名称到对应BufferGroup对象的映射
        return name_to_group

    def convert_to_pool_lines(self, lines):
        """
        Convert AllocateLine/FreeIfNotReusedLine/ReuseLine into their
        pool-based counterparts.
        """
        # 计算缓冲区组，并获取节点名称到BufferGroup对象的映射关系
        name_to_group = self.compute_buffer_groups(lines)
        
        # 遍历传入的lines列表的每一行
        for i, line in enumerate(lines):
            # 如果当前行是AllocateLine类型的实例
            if isinstance(line, AllocateLine):
                # 如果当前AllocateLine对象的节点名称存在于name_to_group映射中
                if line.node.get_name() in name_to_group:
                    # 将当前行替换为AllocFromPoolLine对象
                    lines[i] = AllocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            
            # 如果当前行是FreeIfNotReusedLine类型的实例
            elif isinstance(line, FreeIfNotReusedLine):
                # 确保当前行不是被重用的
                assert not line.is_reused
                # 如果当前FreeIfNotReusedLine对象的节点名称存在于name_to_group映射中
                if line.node.get_name() in name_to_group:
                    # 将当前行替换为DeallocFromPoolLine对象
                    lines[i] = DeallocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            
            # 如果当前行是ReuseLine类型的实例
            elif isinstance(line, ReuseLine):
                # 如果当前ReuseLine对象的节点名称存在于name_to_group映射中
                if line.node.get_name() in name_to_group:
                    # 设置当前行的delete_old属性为False，表示不删除旧节点
                    line.delete_old = False
    def compute_live_ranges(self, lines):
        """根据首次和最后使用情况填充每个BufferGroup.live_ranges字段"""
        timestep = 0
        worklist = collections.deque(lines)  # 创建一个双端队列，初始化为lines参数
        while worklist:
            if isinstance(worklist[0], MemoryPlanningLine):
                timestep += 1  # 时间步长加一
                while worklist and isinstance(worklist[0], MemoryPlanningLine):
                    line = worklist.popleft()  # 从队列左侧移除一个元素
                    if isinstance(line, PoolMemoryPlanningLine):
                        line.group.update_usage(timestep)  # 更新内存池组的使用情况
                        line.timestep = timestep  # 设置当前行的时间步长
            else:
                worklist.popleft()  # 移除非MemoryPlanningLine类型的元素（忽略）

        timestep += 1  # 时间步长再加一
        assert self.buffer_groups is not None  # 断言self.buffer_groups不为空
        for group in self.buffer_groups:
            if group.is_output:
                group.update_usage(timestep)  # 更新输出组的使用情况

    def allocate_groups(self):
        """
        将每个分配分配到特定的位置和特定的AllocationPool中。
        """
        assert config.memory_pool in ("none", "intermediates", "outputs", "combined")  # 断言config.memory_pool在指定的选项中
        assert self.buffer_groups is not None  # 断言self.buffer_groups不为空

        for group in self.buffer_groups:
            group.make_allocation()  # 调用BufferGroup对象的make_allocation方法

        outputs: List[Allocation] = []  # 初始化一个输出的Allocation列表
        intermediates: List[Allocation] = []  # 初始化一个中间结果的Allocation列表
        for group in self.buffer_groups:
            assert group.allocation  # 断言group有分配
            if group.is_output and config.memory_pool != "combined":
                outputs.append(group.allocation)  # 如果是输出且内存池不是combined，则加入到outputs列表
            else:
                intermediates.append(group.allocation)  # 否则加入到intermediates列表

        for block in sorted(
            outputs,
            key=lambda x: (
                x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate_output(block)  # 对outputs列表中的Allocation对象进行分配输出操作

        for block in sorted(
            intermediates,
            key=lambda x: (
                -x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate(block)  # 对intermediates列表中的Allocation对象进行分配操作

        self.pools.finalize()  # 最终处理分配的结果
    # 初始化一个空集合，用于存储已经处理过的池子对象
    seen = set()
    # 遍历传入的 lines 列表中的每一行
    for line in lines:
        # 检查当前行是否为 AllocFromPoolLine 的实例
        if isinstance(line, AllocFromPoolLine):
            # 断言分组对象已经被分配
            assert line.group.allocation
            # 获取分配的池子对象
            pool = line.group.allocation.pool
            # 断言池子对象不为 None
            assert pool is not None
            # 如果当前池子对象不在 seen 集合中，则将该行标记为首次使用该池子
            if pool not in seen:
                line.is_first_pool_usage = True
                seen.add(pool)

    # 重新初始化一个空集合，用于存储已经处理过的池子对象
    seen = set()
    # 反向遍历传入的 lines 列表中的每一行
    for line in reversed(lines):
        # 检查当前行是否为 DeallocFromPoolLine 的实例
        if isinstance(line, DeallocFromPoolLine):
            # 断言分组对象已经被分配
            assert line.group.allocation
            # 获取分配的池子对象
            pool = line.group.allocation.pool
            # 断言池子对象不为 None
            assert pool is not None
            # 如果当前池子对象不在 seen 集合中，则将该行标记为最后一次使用该池子，并根据条件赋值 is_last_pool_usage
            if pool not in seen:
                line.is_last_pool_usage = (
                    pool.root.get_live_ranges().end <= line.timestep
                )
                seen.add(pool)


这段代码的作用是为给定的 `lines` 列表中的特定类型的行对象（`AllocFromPoolLine` 和 `DeallocFromPoolLine`）设置属性，以标记每个池子对象的首次使用和最后一次使用。
```