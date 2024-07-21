# `.\pytorch\torch\_inductor\scheduler.py`

```
# mypy: disallow-untyped-defs
# 从未来导入的注释，禁止未注释的函数类型
from __future__ import annotations

import collections  # 导入collections模块，用于容器数据类型的高性能实现
import dataclasses  # 导入dataclasses模块，用于定义和操作数据类
import functools  # 导入functools模块，用于高阶函数操作
import itertools  # 导入itertools模块，用于创建和操作迭代器的函数
import logging  # 导入logging模块，用于记录日志信息
import math  # 导入math模块，提供对数学函数的访问
import operator  # 导入operator模块，提供对内置操作符的函数形式访问
import os  # 导入os模块，提供与操作系统交互的功能
import pprint  # 导入pprint模块，提供数据结构的美观打印功能
import textwrap  # 导入textwrap模块，提供文本包装和填充功能
import typing  # 导入typing模块，用于支持类型提示
from typing import (  # 从typing模块导入多个类型注解
    Any,
    Counter,
    DefaultDict,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import sympy  # 导入sympy库，用于符号计算

import torch  # 导入torch库，用于深度学习框架
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.utils import counters, dynamo_timed  # 导入torch._dynamo.utils模块的计数器和计时器函数
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled  # 导入torch._inductor.metrics模块的指标表和指标表启用函数
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols  # 导入torch.fx.experimental.symbolic_shapes模块的自由未支持符号
from torch.utils._sympy.symbol import free_symbol_is_type, SymT  # 导入torch.utils._sympy.symbol模块的类型自由符号和SymT类型
from torch.utils._triton import has_triton  # 导入torch.utils._triton模块的是否有Triton函数

from . import comms, config, dependencies, ir, metrics  # 导入当前包内的指定模块
from .codecache import write_text  # 从当前包导入codecache模块的写文本函数
from .codegen.common import (  # 从当前包导入codegen.common模块的多个符号
    BackendFeature,
    get_scheduling_for_device,
    Kernel,
)
from .comm_analysis import estimate_nccl_collective_runtime  # 从当前包导入comm_analysis模块的估算NCCL集合运行时函数
from .dependencies import Dep, MemoryDep, StarDep, WeakDep  # 从当前包导入dependencies模块的多个依赖类型
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout  # 从当前包导入ir模块的计算缓冲、多输出和多输出布局
from .runtime.runtime_utils import green_text, red_text  # 从当前包导入runtime.runtime_utils模块的绿色文本和红色文本函数
from .sizevars import SimplifyIndexing  # 从当前包导入sizevars模块的简化索引类
from .utils import (  # 从当前包导入utils模块的多个实用函数
    cache_on_self,
    cmp,
    device_need_guard,
    get_device_tflops,
    get_dtype_size,
    get_gpu_dram_gbps,
    IndentedBuffer,
    is_collective,
    is_gpu,
    is_wait,
    sympy_product,
)
from .virtualized import V  # 从当前包导入virtualized模块的V符号

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")  # 获取融合日志记录器对象

# 定义基础调度节点类
class BaseSchedulerNode:
    group: Tuple[torch.device, Tuple[Tuple[sympy.Expr, ...], ...]]  # 声明属性group为设备和符号表达式的嵌套元组
    read_writes: dependencies.ReadWrites  # 声明属性read_writes为依赖的读写操作
    unmet_dependencies: Set[Dep]  # 声明属性unmet_dependencies为未满足的依赖集合

    # 初始化方法，接受调度器和IR缓冲区节点作为参数
    def __init__(self, scheduler: Scheduler, node: ir.Buffer) -> None:
        self.scheduler: Scheduler = scheduler  # 设置调度器属性
        self.node: Optional[ir.Buffer] = node  # 设置IR缓冲区节点属性
        self.users: List[NodeUser] = []  # 初始化用户列表属性为空列表
        self.set_read_writes(node.get_read_writes())  # 调用方法设置读写操作
        self.ancestors: Set[str] = set()  # 初始化祖先集合属性为空集合
        self.min_order: int  # 声明最小顺序属性
        self.max_order: int  # 声明最大顺序属性
        self.last_usage: Set[str] = set()  # 初始化最后使用集合属性为空集合，包含不会在此内核之后使用的缓冲区
        self.written = False  # 初始化written属性为False，表示未写入状态

    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.get_name()!r})"
    def debug_str(self) -> str:
        """生成用于跟踪日志的详细输出字符串"""
        # 获取节点名称
        name = self.get_name()
        # 构建输出行列表
        lines = [
            f"{name}: {type(self).__name__}({type(getattr(self, 'node', None)).__name__})",  # 打印对象类型及其节点类型
            f"{name}.writes = {pformat(self.read_writes.writes)}",  # 打印写操作的详细信息
            f"{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}",  # 打印未满足的依赖关系
            f"{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}",  # 打印满足的依赖关系
            f"{name}.users = {self.users}",  # 打印使用该对象的用户列表
        ]
        try:
            lines += [
                self.debug_str_extra(),  # 尝试获取额外的调试信息
            ]
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)  # 记录并忽略在获取额外信息时可能出现的异常

        return "\n".join(lines).rstrip()  # 返回拼接的所有输出行，并移除末尾的空白字符

    def debug_str_extra(self) -> str:
        """额外的调试信息，默认为空字符串"""
        return ""

    def debug_str_short(self) -> str:
        """生成简短的调试输出字符串"""
        maybe_data = getattr(self.node, "data", None)
        data_str = ""
        if isinstance(maybe_data, torch._inductor.ir.Pointwise):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_size()], shorten=False, multiline=False
            )  # 如果数据类型为 Pointwise，获取其大小信息并转换为字符串
        elif isinstance(maybe_data, torch._inductor.ir.Reduction):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_reduction_size(), maybe_data.get_reduction_type()],
                shorten=False,
                multiline=False,
            )  # 如果数据类型为 Reduction，获取其大小和类型信息并转换为字符串
        return f"{self}{data_str}"  # 返回对象的字符串表示形式及其相关数据信息

    def log_details(self) -> None:
        """记录详细信息到日志"""
        log.info(
            "%s: unmet_dependencies = %s, writes = %s",
            self,
            self.unmet_dependencies,
            self.read_writes.writes,
        )  # 记录对象的名称、未满足的依赖关系和写操作到日志

    def update_mutated_names(self, renames: Dict[str, str]) -> None:
        """更新变异名称"""
        self.set_read_writes(self.read_writes.rename(renames))  # 使用给定的重命名字典更新读写操作

    def add_fake_dep(self, dep: Dep) -> None:
        """添加虚假依赖"""
        self.set_read_writes(self.read_writes.with_read(dep))  # 向读写操作中添加虚假依赖对象

    def set_users(self, users: List[NodeUser]) -> None:
        """设置对象的用户列表"""
        # 去重用户列表
        result: Dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = use.merge(result[id(use.node)])
            else:
                result[id(use.node)] = use
        self.users = list(result.values())  # 更新对象的用户列表为去重后的结果列表

    def set_last_usage(
        self, future_used_buffers: Set[str], mutation_real_name: Dict[str, str]
    ) -> None:
        """设置最后一次使用的缓冲区"""
        used_buffers = self.used_or_aliased_buffer_names()  # 获取已使用或已别名的缓冲区名称集合
        used_buffers = {mutation_real_name.get(k, k) for k in used_buffers}  # 使用变异真实名称字典更新缓冲区名称
        self.last_usage = used_buffers - future_used_buffers  # 计算最后一次使用的缓冲区名称集合

    def get_aliases(self) -> Sequence[str]:
        """获取对象的别名列表"""
        assert self.node is not None
        return self.node.get_inputs_that_alias_output()  # 获取节点的输入别名列表

    def get_mutations(self) -> List[str]:
        """获取对象的变异名称列表"""
        assert self.node is not None
        return self.node.get_mutation_names()  # 获取节点的变异名称列表

    def has_aliasing_or_mutation(self) -> bool:
        """检查对象是否存在别名或变异"""
        return bool(self.get_aliases() or self.get_mutations())  # 返回对象是否具有别名或变异的布尔值判断
    # 设置读写依赖关系，将给定的依赖关系对象赋值给当前对象的读写属性
    def set_read_writes(self, rw: dependencies.ReadWrites) -> None:
        self.read_writes = rw
        # 将读依赖设置为未满足的依赖
        self.unmet_dependencies = self.read_writes.reads
        # 剪枝依赖关系中不符合条件的依赖
        self.prune_deps()

    # 返回操作计数的计数器对象
    def op_counts(self) -> Counter[str]:
        return self.read_writes.op_counts

    # 返回使用的缓冲区名称的集合
    def used_buffer_names(self) -> Set[str]:
        return {
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        }

    # 返回使用或别名的缓冲区名称的集合
    def used_or_aliased_buffer_names(self) -> Set[str]:
        used_names = set()

        # 获取所有读写依赖的名称
        deps = [
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        ]
        # 遍历依赖，处理可能的别名关系
        while len(deps) > 0:
            dep = deps.pop()
            used_names.add(dep)
            # 如果存在别名关系，将别名添加到待处理依赖列表中
            if V.graph.name_to_buffer.get(dep):
                for alias in V.graph.name_to_buffer[dep].get_inputs_that_alias_output():
                    if alias not in used_names:
                        deps.append(alias)
        return used_names

    # 剪枝未满足的依赖关系，移除不在调度器可用缓冲区名称中的依赖
    def prune_deps(self) -> None:
        self.unmet_dependencies = {
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        }

    # 剪枝弱依赖关系，移除已移除的缓冲区上的弱依赖
    def prune_weak_deps(self) -> None:
        # 定义判断是否应该剪枝依赖的函数
        def should_prune(dep: Dep) -> bool:
            return isinstance(dep, WeakDep) and dep.name in V.graph.removed_buffers

        # 根据判断条件移除弱依赖
        to_remove = {dep for dep in self.read_writes.reads if should_prune(dep)}
        self.set_read_writes(self.read_writes.remove_reads(to_remove))

    # 剪枝冗余依赖关系，根据给定的名称到融合节点映射进行剪枝
    def prune_redundant_deps(
        self, name_to_fused_node: Dict[str, BaseSchedulerNode]
    ) -> None:
        _prune_redundant_deps(self, name_to_fused_node)

    # 获取节点名称，确保节点不为空
    def get_name(self) -> str:
        assert self.node is not None
        return self.node.get_name()

    # 获取第一个名称，直接调用获取名称方法
    def get_first_name(self) -> str:
        return self.get_name()

    # 获取名称集合，返回包含当前节点名称的集合
    def get_names(self) -> Set[str]:
        return {self.get_name()}

    # 获取节点列表，返回包含当前节点的列表
    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        return [self]

    # 获取设备对象，确保节点不为空
    def get_device(self) -> torch.device:
        assert self.node is not None
        return self.node.get_device()

    # 判断是否为减少操作
    def is_reduction(self) -> bool:
        return False

    # 判断是否为分割扫描操作
    def is_split_scan(self) -> bool:
        return False

    # 判断是否为模板操作
    def is_template(self) -> bool:
        return False

    # 判断是否为外部操作
    def is_extern(self) -> bool:
        return False

    # 判断是否为foreach操作
    def is_foreach(self) -> bool:
        return False

    # 判断是否可以在原地进行操作，总是返回False
    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        return False

    # 判断是否具有副作用，总是返回False
    def has_side_effects(self) -> bool:
        return False
    # 分配操作，确保节点存在
    def allocate(self) -> None:
        assert self.node is not None
        # 如果节点不需要分配，则直接返回
        if not self.node.should_allocate():
            return

        # 如果当前对象是 SchedulerNode 类型，并且存在以下情况：
        # 1. 输入与输出存在别名关系
        # 2. 存在变异名称
        # 则调用代码生成器进行分配操作，并返回
        if isinstance(self, (SchedulerNode,)) and (
            self.node.get_inputs_that_alias_output() or self.node.get_mutation_names()
        ):
            V.graph.wrapper_code.codegen_allocation(self.node)
            return

        # 用于检查 V.kernel 是否为真实内核或 NullHandler 的简单检查方法
        if (
            hasattr(V.kernel, "args")
            and self.get_name() in V.kernel.inplace_update_buffers
        ):
            # 如果当前对象的名称存在于 inplace_update_buffers 中，则调用代码生成器进行原地重用操作
            V.graph.wrapper_code.codegen_inplace_reuse(
                self.scheduler.name_to_node[
                    V.kernel.inplace_update_buffers[self.get_name()]
                ].node,
                self.node,
            )
        else:
            # 否则调用代码生成器进行分配操作
            V.graph.wrapper_code.codegen_allocation(self.node)

    # 判断是否可以释放当前对象
    def can_free(self) -> bool:
        # 确保节点存在
        assert self.node is not None
        # 如果节点的布局为 ir.NoneLayout，则无需释放
        if isinstance(self.node.layout, ir.NoneLayout):
            return False
        # 检查所有使用当前对象的用户
        for use in self.users:
            # 如果使用者是 OutputNode 类型，则不可以释放
            if isinstance(use.node, OutputNode):
                return False
        # 可以释放
        return True

    # 生成与源信息相关的代码注释
    def codegen_originating_info(
        self, buffer: IndentedBuffer, only_once: bool = True
    ) -> None:
        # 如果配置中不需要源信息的注释，则直接返回
        if not config.comment_origin:
            return

        # 如果 only_once 为真且已经生成过注释，则直接返回
        if only_once and self.written:
            return
        # 确保节点存在
        assert self.node is not None
        # 获取节点的源信息
        origins = self.node.origins
        # 存储生成的注释行
        out_lines = []

        # 遍历节点的源信息
        for o in origins:
            # 如果操作为 "output"，则跳过生成注释，因为这些信息通常是重复和无聊的
            if o.op == "output":
                continue

            out_lines.append("")
            # 添加源信息的 pragma 注释
            out_lines.append("#pragma CMT ORIGIN:")
            op_info_str = f"#pragma CMT {o.op} {o.target}"
            # 如果 meta 中包含 'seq_nr' 字段，则添加序列号信息
            if "seq_nr" in o.meta:
                op_info_str = op_info_str + f" seq_nr:{o.meta['seq_nr']}"
            out_lines.append(op_info_str)
            # 如果 meta 中包含 'stack_trace' 字段，则添加堆栈跟踪信息
            if "stack_trace" in o.meta:
                stack_trace = f"{o.meta['stack_trace']}"
                stack_trace_last_line = stack_trace.split("|")[-1]
                out_lines.append(
                    "#pragma CMT "
                    + stack_trace_last_line.replace("{", "{{")
                    .replace("}", "}}")
                    .replace("\n", "\\")
                )
            out_lines.append("#pragma CMT END ORIGIN")
            out_lines.append("")

        # 如果没有生成任何注释行，则直接返回
        if len(out_lines) == 0:
            return

        # 将生成的注释行写入到指定缓冲区中
        buffer.writelines(out_lines)
        self.written = True

    # 获取模板节点，可能返回 None
    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        return None
class WhyNoFuse:
    # 当我们停止支持 Python < 3.10 时，可以使用 @dataclass(slots=True) 替代手动指定 __slots__。
    # 这个类表示为什么两个调度节点不能融合的原因和参数
    __slots__ = ["node1", "node2", "reason", "args"]
    reason: str  # 描述不能融合的具体原因的字符串
    args: Tuple[Any, ...]  # 传递给融合失败原因的参数元组

    def __init__(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        self.node1 = node1  # 第一个调度节点
        self.node2 = node2  # 第二个调度节点

    def __call__(self, reason: str, *args: Any) -> None:
        self.reason = reason  # 设置融合失败的原因
        self.args = args  # 设置相关参数
        fusion_log.debug(self)  # 记录调试信息到日志

    def __str__(self) -> str:
        return f"cannot fuse {self.node1.get_name()} with {self.node2.get_name()}: " + (
            self.reason % self.args
        )


def pformat(obj: Any) -> str:
    if isinstance(obj, set):
        # pformat 对 sympy 表达式集合的处理有问题
        obj = sorted(obj, key=str)  # 对集合进行排序
    result = pprint.pformat(obj, indent=4)  # 使用 pprint 格式化对象
    if "\n" in result:
        return f"\n{textwrap.indent(result, ' '*4)}"  # 如果结果包含换行，进行缩进处理
    return result


class OutputNode:
    def __init__(self, dep: StarDep) -> None:
        self.unmet_dependencies = {dep}  # 初始化未满足的依赖集合

    def is_reduction(self) -> bool:
        return False  # 输出节点不是减少操作

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return ()  # 返回空元组，表示没有与输出别名相关的输入

    def get_name(self) -> str:
        return "OUTPUT"  # 返回节点名称为 "OUTPUT"

    __repr__ = get_name  # 将 __repr__ 方法重载为 get_name 方法


def _prune_redundant_deps(
    node: BaseSchedulerNode, name_to_fused_node: Dict[str, BaseSchedulerNode]
) -> None:
    """
    修剪弱依赖项，以便在上游已融合节点之后进行变异排序

    实质上，这强制执行融合的顺序。随着融合的发生，弱依赖项将逐步删除，
    从而启用其他的融合，并确保按顺序融合。

    Args:
    - node: 要处理的调度节点
    - name_to_fused_node: 映射每个节点名到已融合节点对象的字典
    """
    name_to_dep_count: Counter[str] = collections.Counter()

    for dep in node.unmet_dependencies:
        if not isinstance(dep, WeakDep):
            name_to_dep_count[name_to_fused_node[dep.name].get_name()] += 1

    def should_prune(dep: Dep) -> bool:
        if isinstance(dep, WeakDep):
            is_redundant = (
                name_to_dep_count[name_to_fused_node[dep.name].get_name()] > 0
            )
            # 这些可能会发生，因为融合节点总是从它们的子节点获取依赖
            # 如果 B 对 A 有弱依赖
            # B 与 C 融合后，任何时候 BC 被融合，弱依赖将重新出现
            is_self_dep = name_to_fused_node[dep.name] == node
            return is_redundant or is_self_dep
        else:
            return False

    deps_to_prune = {dep for dep in node.unmet_dependencies if should_prune(dep)}

    if deps_to_prune:
        node.unmet_dependencies = node.unmet_dependencies - deps_to_prune
        node.set_read_writes(node.read_writes.remove_reads(deps_to_prune))


# TODO(xmfan): 如果存在的话，重用此映射，否则将其正式化为 ir.py:ExternKernel 中的一个映射
kernel_name_to_op = {
    # 将外部实现的卷积操作注册到指定名称，使用 torch.ops.aten.convolution
    "extern_kernels.convolution": torch.ops.aten.convolution,
    # 将外部实现的矩阵乘法操作注册到指定名称，使用 torch.ops.aten.mm
    "extern_kernels.mm": torch.ops.aten.mm,
    # 将外部实现的批量矩阵乘法操作注册到指定名称，使用 torch.ops.aten.bmm
    "extern_kernels.bmm": torch.ops.aten.bmm,
    # 将外部实现的矩阵加法乘法操作注册到指定名称，使用 torch.ops.aten.addmm
    "extern_kernels.addmm": torch.ops.aten.addmm,
}


class ExternKernelSchedulerNode(BaseSchedulerNode):
    # 继承自 BaseSchedulerNode 的外部内核调度节点类

    def debug_str_extra(self) -> str:
        # 返回节点的调试信息的额外部分字符串
        return f"{self.get_name()}.node.kernel = {getattr(self.node, 'python_kernel_name', None)}"

    def is_extern(self) -> bool:
        # 判断是否为外部节点的方法，始终返回 True
        return True

    def has_side_effects(self) -> bool:
        # 断言节点不为空，检查节点是否具有副作用属性
        assert self.node is not None
        return hasattr(self.node, "has_side_effects") and self.node.has_side_effects()


class NopKernelSchedulerNode(BaseSchedulerNode):
    # 继承自 BaseSchedulerNode 的空操作内核调度节点类
    pass


class SchedulerNode(BaseSchedulerNode):
    # 继承自 BaseSchedulerNode 的调度节点类

    def __init__(
        self,
        scheduler: Scheduler,
        node: Union[ir.ComputedBuffer, ir.TemplateBuffer],
    ) -> None:
        # 初始化方法，接受调度器对象和节点对象参数
        super().__init__(scheduler, node)
        self._compute_attrs()  # 调用计算属性的方法

    def _compute_attrs(
        self,
        extra_indexing_constraints: Optional[Tuple[Dict[Any, Any], List[Any]]] = None,
    ) -> None:
        # 计算节点属性的方法
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer))
        self._sizes, self._body = self.node.simplify_and_reorder(
            extra_indexing_constraints=extra_indexing_constraints
        )

        group_fn = self.scheduler.get_backend(self.node.get_device()).group_fn
        self.group = (self.node.get_device(), group_fn(self._sizes))

        if isinstance(self.node, ir.TemplateBuffer):
            self.set_read_writes(self.node.normalized_read_writes())
        else:
            self.set_read_writes(
                dependencies.extract_read_writes(
                    self._body, *self._sizes, normalize=True
                )
            )

    def recompute_size_and_body(
        self, extra_indexing_constraints: Tuple[Dict[Any, Any], List[Any]]
    ) -> None:
        # 重新计算节点大小和主体方法
        self._compute_attrs(extra_indexing_constraints=extra_indexing_constraints)

    def debug_str_extra(self) -> str:
        # 返回节点调试信息的额外部分字符串
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        for dep in self.read_writes.reads_and_writes():
            buf_name = dep.name
            buf = V.graph.get_buffer(buf_name)
            lines.append(f"{buf_name}_layout = {pformat(buf.layout)}")
        if self.get_aliases():
            lines.append(f"{name}.aliases = {pformat(self.get_aliases())}")
        if self.get_mutations():
            lines.append(f"{name}.mutations = {pformat(self.get_mutations())}")
        if isinstance(self._body, ir.LoopBody):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))

        assert self.node is not None
        if ir.is_triton(self.node.get_device()):
            lines.extend(debug_triton_code(self))

        return "\n".join(lines)

    def get_ranges(self) -> Sequence[Sequence[sympy.Expr]]:
        # 获取节点范围的方法，返回节点的大小属性
        return self._sizes
    def is_reduction(self) -> bool:
        # 断言节点类型为 ComputedBuffer 或 TemplateBuffer
        assert isinstance(
            self.node, (ir.ComputedBuffer, ir.TemplateBuffer)
        ), f"{type(self.node)=}"
        # 返回节点是否为 reduction 类型的布尔值
        return bool(self.node.get_reduction_type())

    def is_split_scan(self) -> bool:
        # 断言节点类型为 ComputedBuffer 或 TemplateBuffer
        assert isinstance(
            self.node, (ir.ComputedBuffer, ir.TemplateBuffer)
        ), f"{type(self.node)=}"
        # 返回节点是否为 ComputedBuffer 且其数据类型为 SplitScan 的布尔值
        return isinstance(self.node, ir.ComputedBuffer) and isinstance(
            self.node.data, ir.SplitScan
        )

    def is_template(self) -> bool:
        # 返回节点是否为 TemplateBuffer 的布尔值
        return isinstance(self.node, ir.TemplateBuffer)

    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        # 如果节点是 TemplateBuffer 类型则返回节点，否则返回 None
        return self.node if isinstance(self.node, ir.TemplateBuffer) else None

    def run(self, *index_vars: Sequence[sympy.Expr]) -> None:
        # 执行决定是否原地更新
        self.decide_inplace_update()
        # 执行标记运行
        self.mark_run()
        # 调用代码生成函数
        self.codegen(index_vars)

    def mark_run(self) -> None:
        # 分配资源
        self.allocate()

    def ranges_from_index_vars(
        self, index_vars: Sequence[Sequence[sympy.Expr]]
    ) -> Dict[sympy.Expr, sympy.Expr]:
        # 检查索引变量和尺寸的总数是否相等
        sizes = self._sizes
        assert sum(map(len, sizes)) == sum(map(len, index_vars))
        # 创建索引变量范围的字典映射
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars: Sequence[Sequence[sympy.Expr]]) -> None:
        # 根据索引变量生成代码
        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            # 使用 SimplifyIndexing 处理操作，并设置当前节点为当前内核节点
            with V.set_ops_handler(
                SimplifyIndexing(V.get_ops_handler(), var_ranges)
            ), V.kernel.set_current_node(self):
                # 调用内部代码生成函数
                self._body(*index_vars)
        except Exception:
            # 记录错误日志并抛出异常
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_read_writes(self) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in the non-reduction axis.
        """
        # 获取尺寸和 reduction 尺寸
        sizes, reduction_sizes = self._sizes

        def fn(index: Sequence[sympy.Symbol]) -> str:
            # 返回在非 reduction 轴上的内存依赖
            return self._body(index, [sympy.Integer(0) for _ in reduction_sizes])

        # 提取读写依赖
        return dependencies.extract_read_writes(fn, sizes)

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        # 如果存在别名或者节点是模板，则不能原地更新
        if self.get_aliases() or self.is_template():
            return False
        # 如果只有一个写入操作且读依赖为内存依赖，则尝试进行原地更新
        if len(self.read_writes.writes) == 1 and isinstance(
            read_dep, dependencies.MemoryDep
        ):
            write_dep = next(iter(self.read_writes.writes))
            assert isinstance(write_dep, dependencies.MemoryDep), f"{type(write_dep)=}"
            # 检查读写依赖的索引和尺寸是否相同
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    @cache_on_self
    # 获取使用原子加法操作的缓冲区集合
    def _get_atomic_add_buffers(self) -> Set[str]:
        # 创建一个空集合，用于存储使用原子加法操作的缓冲区名称
        buffers_store_as_atomic_add = set()
        # 如果对象的身体部分是一个循环体
        if isinstance(self._body, ir.LoopBody):
            # 遍历循环体中的所有节点
            for node in self._body.get_nodes():
                # 检查节点操作为 "call_method"、目标为 "store"，且满足以下条件之一：
                # 1. 存在 "mode" 参数且其值为 "atomic_add"
                # 2. 参数列表长度为 5 且第五个参数为 "atomic_add"
                if (
                    node.op == "call_method"
                    and node.target == "store"
                    and (
                        ("mode" in node.kwargs and node.kwargs["mode"] == "atomic_add")
                        or (len(node.args) == 5 and node.args[4] == "atomic_add")
                    )
                ):
                    # 将节点中的缓冲区名称添加到集合中
                    buffers_store_as_atomic_add.add(
                        node.kwargs["name"]
                        if "name" in node.kwargs
                        else (node.args[1] if len(node.args) >= 2 else "")
                    )
        # 返回存储原子加法操作缓冲区名称的集合
        return buffers_store_as_atomic_add
    def set_last_usage(
        self, future_used_buffers: Set[str], mutation_real_name: Dict[str, str]
    ) -> None:
        """
        Set the last usage information for the fused scheduler node.
        
        Args:
            future_used_buffers: Set of names of buffers that will be used in the future.
            mutation_real_name: Mapping of mutation names to their real names.
        """
        # Calculate the set of unmet dependencies considering the names of all nodes in the group
        self.unmet_dependencies = {
            dep
            for dep in set.union(*[x.unmet_dependencies for x in self.snodes])
            if dep.name not in self.get_names()
        } - self.read_writes.writes
        
        # Determine the minimum and maximum order among the constituent nodes
        self.min_order = min(x.min_order for x in self.snodes)
        self.max_order = max(x.max_order for x in self.snodes)
    ) -> None:
        """
        设置 self.last_usage 使用全局信息
        将用于内核间优化
        """
        super().set_last_usage(future_used_buffers, mutation_real_name)
        """
        设置 self.last_usage 在 snodes 上
        将用于内核内优化
        """
        future_used_buffers: Set[str] = set()
        """
        反向遍历 self.snodes
        设置每个节点的 self.last_usage
        """
        for node in reversed(self.snodes):
            node.set_last_usage(future_used_buffers, mutation_real_name)
            future_used_buffers.update(node.last_usage)

    @cache_on_self
    def used_buffer_names(self) -> Set[str]:
        """
        返回所有 snodes 使用的缓冲区名称的并集
        """
        return set.union(*[x.used_buffer_names() for x in self.snodes])

    @cache_on_self
    def used_or_aliased_buffer_names(self) -> Set[str]:
        """
        返回所有 snodes 使用或别名的缓冲区名称的并集
        """
        return set.union(*[x.used_or_aliased_buffer_names() for x in self.snodes])

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        """
        返回 self.snodes
        """
        return self.snodes

    def __repr__(self) -> str:
        """
        返回对象的字符串表示形式，包含节点名称
        """
        return f"{type(self).__name__}(nodes={self.get_name()})"

    @cache_on_self
    def is_reduction(self) -> bool:
        """
        检查是否存在任何 snode 是 reduction
        """
        return any(x.is_reduction() for x in self.snodes)

    @cache_on_self
    def is_split_scan(self) -> bool:
        """
        检查是否存在任何 snode 是 split scan
        """
        return any(x.is_split_scan() for x in self.snodes)

    @cache_on_self
    def is_template(self) -> bool:
        """
        检查是否存在任何 snode 是 template
        """
        return any(x.is_template() for x in self.snodes)

    @cache_on_self
    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        """
        返回第一个发现的 template snode 的模板节点
        """
        for node in self.snodes:
            if node.is_template():
                return node.get_template_node()
        return None

    def get_device(self) -> torch.device:
        """
        返回组中的第一个设备
        """
        return self.group[0]

    @cache_on_self
    def has_aliasing_or_mutation(self) -> bool:
        """
        检查是否存在任何 snode 有别名或变异
        """
        return any(x.has_aliasing_or_mutation() for x in self.snodes)

    @cache_on_self
    def op_counts(self) -> Counter[str]:
        """
        返回所有 snodes 的操作计数的总和
        """
        op_counts: Counter[str] = collections.Counter()
        for node in self.snodes:
            op_counts.update(node.op_counts())
        return op_counts

    # None of these need to be implemented, as a FusedSchedulerNode is just an
    # abstraction for scheduling purposes
    def update_mutated_names(self, renames: Dict[str, str]) -> None:
        """
        更新变异名称，抛出未实现错误
        """
        raise NotImplementedError

    def add_fake_dep(self, name: Dep) -> None:
        """
        添加虚假依赖，抛出未实现错误
        """
        raise NotImplementedError

    def set_users(self, users: List[NodeUser]) -> None:
        """
        设置用户，抛出未实现错误
        """
        raise NotImplementedError

    def get_aliases(self) -> Sequence[str]:
        """
        获取别名，抛出未实现错误
        """
        raise NotImplementedError

    def get_mutations(self) -> List[str]:
        """
        获取变异，抛出未实现错误
        """
        raise NotImplementedError

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        """
        检查是否可以就地操作，抛出未实现错误
        """
        raise NotImplementedError

    def allocate(self) -> None:
        """
        分配资源，抛出未实现错误
        """
        raise NotImplementedError

    def can_free(self) -> bool:
        """
        检查是否可以释放资源，抛出未实现错误
        """
        raise NotImplementedError
    def debug_str(self) -> str:
        """生成用于跟踪日志的详细打印输出"""
        # 获取当前对象的名称
        name = self.get_name()
        # 生成节点类型字符串，包含所有子节点的类型名称
        node_typestr = ",".join(type(n).__name__ for n in self.snodes)
        # 构建输出行列表
        lines = [
            f"{name}: {type(self).__name__}({node_typestr})",  # 输出对象名称和类名
            f"{name}.writes = {pformat(self.read_writes.writes)}",  # 输出写操作的详细信息
            f"{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}",  # 输出未满足的依赖关系
            f"{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}",  # 输出已满足的依赖关系
            f"{name}.users = {self.users}",  # 输出当前对象的用户信息
        ]
        try:
            lines += [
                self.debug_str_extra(),  # 尝试获取额外的调试信息并加入到输出行中
            ]
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)  # 如果获取额外调试信息时出错，则记录警告信息

        # 将所有行组合成最终的输出字符串，并移除末尾的空白字符
        return "\n".join(lines).rstrip()
class ForeachKernelSchedulerNode(FusedSchedulerNode):
    """Scheduler node which consists of a list of scheduler nodes that each operate on a
    distinct tensor in a list of tensors."""

    # 返回一个消费者子节点，用于给定的生产者节点
    def get_consumer_subnode_for(
        self, producer: BaseSchedulerNode
    ) -> Optional[BaseSchedulerNode]:
        if producer.get_name() in self.read_to_node:
            return self.read_to_node[producer.get_name()]

        return None

    # 返回一个生产者子节点，用于给定的消费者节点
    def get_producer_subnode_for(
        self, consumer: BaseSchedulerNode
    ) -> Optional[BaseSchedulerNode]:
        for rd in consumer.read_writes.reads:
            if rd.name in self.name_to_node:
                return self.name_to_node[rd.name]

        return None

    @classmethod
    # 判断是否可以融合两个节点，返回布尔值
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        why = WhyNoFuse(producer, consumer)  # 创建一个 WhyNoFuse 实例
        if producer.is_foreach() and consumer.is_foreach():  # 如果生产者和消费者都是 foreach 节点
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            foreach_match = len(producer.snodes) == len(consumer.snodes)  # 检查 foreach 节点的子节点数量是否相同
            if not foreach_match:
                why("foreach do not have same length")  # 如果数量不同，记录原因
            return foreach_match and all(
                producer.scheduler.can_fuse(l, r)  # 对每对子节点调用 can_fuse 方法检查是否可以融合
                for l, r in zip(producer.snodes, consumer.snodes)
            )
        elif consumer.is_foreach():  # 如果消费者是 foreach 节点
            if producer.is_reduction():  # 如果生产者是 reduction 节点
                why(
                    "candidate producer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            if consumer_subnode is not None:
                return consumer.scheduler.can_fuse(producer, consumer_subnode)

            why("candidate producer is not dep of any foreach consumer")
            return False

        elif producer.is_foreach():  # 如果生产者是 foreach 节点
            if consumer.is_reduction():  # 如果消费者是 reduction 节点
                why(
                    "candidate consumer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            producer_subnode = producer.get_producer_subnode_for(consumer)
            if producer_subnode is not None:
                return producer.scheduler.can_fuse(producer_subnode, consumer)

            why("candidate consumer has no dep in any foreach producer")
            return False

        raise AssertionError(
            "At least one node passed to ForeachKernelSchedulerNode.can_fuse should be a foreach node"
        )

    @classmethod
    # 实施节点融合操作
    def fuse(
        cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode
    ) -> ForeachKernelSchedulerNode:
        # 声明此方法返回类型为 ForeachKernelSchedulerNode
        assert producer.is_foreach() or consumer.is_foreach()
        # 断言 producer 或 consumer 必须是 ForeachKernelSchedulerNode 类型的对象

        prev_node_1 = None
        prev_node_2 = None
        # 初始化两个变量 prev_node_1 和 prev_node_2，初始值为 None

        fused_nodes: List[BaseSchedulerNode]
        # 声明 fused_nodes 变量为 List[BaseSchedulerNode] 类型

        if producer.is_foreach() and consumer.is_foreach():
            # 如果 producer 和 consumer 都是 ForeachKernelSchedulerNode 类型
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            # 强制将 producer 转换为 ForeachKernelSchedulerNode 类型
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            # 强制将 consumer 转换为 ForeachKernelSchedulerNode 类型

            # 使用 zip 函数遍历 producer 和 consumer 的 snodes，并通过 FusedSchedulerNode.fuse 方法进行融合
            fused_nodes = [
                FusedSchedulerNode.fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            ]

        elif producer.is_foreach():
            # 如果只有 producer 是 ForeachKernelSchedulerNode 类型
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            # 强制将 producer 转换为 ForeachKernelSchedulerNode 类型
            producer_subnode = producer.get_producer_subnode_for(consumer)
            # 获取 producer 对应于 consumer 的子节点

            fused_nodes = []
            prev_node_1 = producer
            prev_node_2 = None

            # 遍历 producer 的 snodes，对于 producer_subnode 进行特殊处理并融合
            for node in producer.snodes:
                if node is producer_subnode:
                    new_node = FusedSchedulerNode.fuse(node, consumer)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)

        elif consumer.is_foreach():
            # 如果只有 consumer 是 ForeachKernelSchedulerNode 类型
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            # 强制将 consumer 转换为 ForeachKernelSchedulerNode 类型
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            # 获取 consumer 对应于 producer 的子节点

            fused_nodes = []
            prev_node_1 = consumer
            prev_node_2 = None

            # 遍历 consumer 的 snodes，对于 consumer_subnode 进行特殊处理并融合
            for node in consumer.snodes:
                if node is consumer_subnode:
                    new_node = FusedSchedulerNode.fuse(producer, node)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)

        return cls(producer.scheduler, fused_nodes, prev_node_1, prev_node_2)
        # 返回一个新的实例，包含 producer 的调度器、融合后的节点列表 fused_nodes，以及前两个节点 prev_node_1 和 prev_node_2

    def __init__(
        self,
        scheduler: Scheduler,
        nodes: Sequence[BaseSchedulerNode],
        prev_node_1: Optional[BaseSchedulerNode] = None,
        prev_node_2: Optional[BaseSchedulerNode] = None,
    ) -> None:
        # 初始化两个空字典，用于存储读取到的节点和节点名称映射关系
        self.read_to_node = {}
        self.name_to_node = {}

        # 如果前两个节点任意一个为None，则调用父类构造函数进行初始化
        if prev_node_1 is None or prev_node_2 is None:
            super().__init__(scheduler, nodes)

            # 遍历所有节点
            for node in nodes:
                # 将每个节点的读取操作映射到self.read_to_node字典中
                for read in node.read_writes.reads:
                    self.read_to_node[read.name] = node

                # 获取节点的名称列表，并将每个名称映射到self.name_to_node字典中
                for name in node.get_names():
                    self.name_to_node[name] = node
        else:
            # 如果两个前置节点均不为None，则执行以下逻辑

            # 设置调度器和节点列表
            self.scheduler = scheduler
            self.snodes = nodes
            self.node = None
            self.users: List[NodeUser] = []

            # 合并两个前置节点的读写依赖关系
            self.set_read_writes(
                dependencies.ReadWrites.merge_list(
                    [prev_node_1.read_writes, prev_node_2.read_writes]
                )
            )

            # 计算未满足的依赖项集合
            self.unmet_dependencies = {
                dep
                for dep in set.union(
                    prev_node_1.unmet_dependencies, prev_node_2.unmet_dependencies
                )
                # 过滤掉已经存在于当前节点名称集合中的依赖项
                if dep.name not in self.get_names()
            } - self.read_writes.writes

            # 计算最小和最大执行顺序
            self.min_order = min([prev_node_1.min_order, prev_node_2.min_order])
            self.max_order = max([prev_node_1.max_order, prev_node_2.max_order])

            # 确定是否为Foreach节点
            if prev_node_1.is_foreach():
                assert isinstance(prev_node_1, ForeachKernelSchedulerNode)
                foreach_node, other_node = prev_node_1, prev_node_2
            else:
                assert isinstance(prev_node_2, ForeachKernelSchedulerNode)
                foreach_node, other_node = prev_node_2, prev_node_1

            # 合并祖先节点集合
            self.ancestors = foreach_node.ancestors
            self.ancestors.update(other_node.ancestors)

            # 将名称到节点的映射关系设置为第一个Foreach节点的名称到节点的映射关系
            self.name_to_node = foreach_node.name_to_node
            # 将第二个节点的所有名称映射到self.name_to_node字典中
            for name in other_node.get_names():
                self.name_to_node[name] = other_node

        # 设置节点组信息，包含设备和特定元组
        self.group = (nodes[0].get_device(), ((sympy.Expr("foreach"),),))

        # 初始化原始节点集合为空集
        self.origins: Set[torch.fx.Node] = set()

    def mark_run(self) -> None:
        # 抛出未实现错误，子类需实现此方法
        raise NotImplementedError

    def codegen(self) -> None:
        # 断言当前节点是一个ComputedBuffer类型的节点，否则抛出异常
        assert isinstance(self.node, ir.ComputedBuffer), f"{type(self.node)=}"
        # 调用节点的存储函数和加载器函数，实现代码生成
        self.node.get_store_function()(self.node.make_loader()())

    def can_free(self) -> bool:
        # 抛出未实现错误，子类需实现此方法
        raise NotImplementedError

    def is_foreach(self) -> bool:
        # 始终返回True，表示当前节点为Foreach节点
        return True

    def get_subkernel_nodes(self) -> List[BaseSchedulerNode]:
        """Returns a list of nodes which comprise the foreach kernel, operating on corresponding elements of our input lists.
        These nodes may be vertically fused."""
        # 返回当前节点所包含的子内核节点列表
        return list(self.snodes)

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        """Returns all nodes contained in this kernel, unpacking fused nodes
        into their constituent scheduler nodes."""
        # 返回当前内核中的所有节点，展开融合的节点为其各自的调度器节点
        return list(itertools.chain.from_iterable(x.get_nodes() for x in self.snodes))

    def get_first_name(self) -> str:
        # 返回第一个子节点的第一个名称
        return self.snodes[0].get_first_name()
    # 定义一个方法，用于修剪冗余依赖关系。方法接受一个字典参数，将其传递给内部函数。
    def prune_redundant_deps(
        self, name_to_fused_node: Dict[str, BaseSchedulerNode]
    ) -> None:
        # 调用内部函数 _prune_redundant_deps，传递当前对象和给定的节点字典作为参数
        _prune_redundant_deps(self, name_to_fused_node)

        # 遍历当前对象的 snodes 列表
        for node in self.snodes:
            # 对每个节点调用其 prune_redundant_deps 方法，传递给定的节点字典作为参数
            node.prune_redundant_deps(name_to_fused_node)
def pick_loop_order(
    stride_lengths: List[List[int]],
    sizes: List[sympy.Expr],
    priority_idx: Tuple[int, ...] = (),
) -> List[int]:
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

    # 定义一个比较函数，用于比较两个索引的优先级
    @functools.cmp_to_key
    def index_cmp(a: int, b: int) -> int:
        # 如果其中一个维度大小为1，则不影响顺序，将其放在最后
        if sizes[a] == 1 or sizes[b] == 1:
            return cmp(sizes[a] == 1, sizes[b] == 1)

        # 计算每个维度的步长长度的绝对值
        stride_len_a = [abs(sl[a]) for sl in stride_lengths]
        stride_len_b = [abs(sl[b]) for sl in stride_lengths]

        # 计算在所有尺寸上，a 在 b 之前的次数
        a_first = sum(
            sl_b == 0 or sl_a < sl_b for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        # 计算在所有尺寸上，b 在 a 之前的次数
        b_first = sum(
            sl_a == 0 or sl_b < sl_a for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        # 根据优先级次数比较 a 和 b 的顺序
        if a_first > b_first:
            return -1
        if b_first > a_first:
            return 1

        # 如果优先级相同，则按索引大小比较
        return cmp(b, a)

    # 初始顺序为逆序的索引列表
    order = list(reversed(range(len(stride_lengths[0]))))
    # 如果有指定优先级索引，则只使用这些索引的顺序
    if len(priority_idx) > 0:
        stride_lengths = [stride_lengths[pi] for pi in priority_idx]
    # 如果配置允许，根据定义的比较函数对顺序进行排序
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order


@dataclasses.dataclass
class NodeUser:
    node: Union[BaseSchedulerNode, OutputNode]
    can_inplace: bool = False

    # A weak user must be scheduled after a given node, but doesn't actually
    # use the result
    is_weak: bool = False

    def __hash__(self) -> int:
        # 返回对象的哈希值，以便在集合中使用
        return hash((self.node.get_name(), self.can_inplace, self.is_weak))

    def __eq__(self, other: object) -> bool:
        # 比较对象是否相等
        return (
            isinstance(other, NodeUser)
            and self.get_name() == other.get_name()
            and self.can_inplace == other.can_inplace
            and self.is_weak == other.is_weak
        )

    def get_name(self) -> str:
        # 获取节点的名称
        return self.node.get_name()

    def merge(self, other: NodeUser) -> NodeUser:
        # 合并两个 NodeUser 对象，要求它们属于同一个节点
        assert self.node is other.node
        return NodeUser(
            self.node,
            self.can_inplace and other.can_inplace,
            self.is_weak and other.is_weak,
        )


# 计数器对象，用于生成唯一的后续图计数
_post_grad_graph_counter = itertools.count()


class Scheduler:
    # 缓存依赖关系大小的提示，以便快速访问
    __dep_size_hint_cache: Dict[Dep, int]

    @dynamo_timed
    def get_current_device_or_throw(self) -> torch.device:
        # 获取当前设备，如果不存在则抛出异常
        if device := self.current_device:
            return device
        else:
            raise RuntimeError("No current device")
    # 生成用于调试的图形像文件的方法
    def debug_draw_graph(self) -> None:
        """Generate an image of the graph for debugging"""
        # 检查环境变量是否设置了生成调度器图形的选项
        if os.environ.get("INDUCTOR_WRITE_SCHEDULER_GRAPH", None) == "1":
            # 导入绘制缓冲区的调试函数
            from .debug import draw_buffers

            # 调用绘制函数，并打印调试图形
            draw_buffers(self.nodes, print_graph=True)

    # 打印节点详细信息的调试方法
    def debug_print_nodes(self, label: str) -> None:
        # 如果 INFO 级别日志启用
        if log.isEnabledFor(logging.INFO):
            # 记录标签信息到日志
            log.info("%s:", label)
            # 遍历所有节点，记录每个节点的详细信息到日志
            for node in self.nodes:
                node.log_details()

    # 创建调度器节点的方法
    def create_scheduler_node(self, node: ir.Buffer) -> BaseSchedulerNode:
        # 断言节点的来源不为空，确保所有传递到调度的节点都有一个来源
        assert (
            node.origins is not None
        ), "All nodes passed to scheduling must have an origin"
        # 如果节点是空操作节点，返回空操作内核调度节点对象
        if node.is_no_op():
            return NopKernelSchedulerNode(self, node)
        # 如果节点是计算缓冲区或模板缓冲区，返回普通调度节点对象
        elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            return SchedulerNode(self, node)
        # 如果节点是外部内核，返回外部内核调度节点对象
        elif isinstance(node, ir.ExternKernel):
            return ExternKernelSchedulerNode(self, node)
        else:
            # 抛出未实现错误，表明当前节点类型未被处理
            raise NotImplementedError(node)

    # 创建 foreach 节点的方法
    def create_foreach_nodes(self) -> None:
        # 存储被移除的节点名称的集合
        removed_node_names = set()
        # 存储 foreach 节点的列表
        fe_nodes = []
        # 获取保留的节点名称的集合
        kept_node_names = self.name_to_fused_node.keys()

        # 遍历图中的每个节点列表
        for names in V.graph.lists.values():
            # 过滤掉不在保留节点集合中或者是空操作节点的名称
            names = [
                name
                for name in names
                if name in kept_node_names
                and not isinstance(self.name_to_node[name], NopKernelSchedulerNode)
            ]
            # 如果过滤后的名称列表为空，则表示所有节点都被消除
            if not names:
                continue

            # 更新被移除节点名称集合
            removed_node_names.update(names)
            # 获取每个名称对应的节点对象列表
            snodes = [self.name_to_node[name] for name in names]

            # 创建 foreach 节点对象
            fe_node = ForeachKernelSchedulerNode(self, snodes)

            # 将创建的 foreach 节点添加到列表中
            fe_nodes.append(fe_node)

            # 将每个名称映射到对应的 foreach 节点对象
            for name in names:
                self.name_to_fused_node[name] = fe_node

        # 更新节点列表，移除已被消除的节点，并添加新创建的 foreach 节点
        self.nodes = [
            node for node in self.nodes if node.get_name() not in removed_node_names
        ] + list(fe_nodes)
    def dead_node_elimination(self) -> None:
        """
        Remove any nodes without users
        """
        # self.nodes is in topological order, so by iterating in reverse order
        # we have visited (and potentially removed) all users before visiting a
        # given node.
        updated_nodes = []
        for node in reversed(self.nodes):
            # Define a function to determine if a user node can be eliminated
            def can_eliminate_user(user: NodeUser) -> bool:
                return user.is_weak or user.get_name() in V.graph.removed_buffers
            
            # Check if the node can be eliminated based on absence of side effects
            # and all users being eligible for elimination
            can_eliminate = not node.has_side_effects() and all(
                can_eliminate_user(u) for u in node.users
            )

            if not can_eliminate:
                updated_nodes.append(node)
            else:
                # Log the removal of a dead node and add its name to removed_buffers
                log.debug("removed dead node: %s", node.get_name())
                V.graph.removed_buffers.add(node.get_name())

        self.nodes = list(reversed(updated_nodes))

        # Prune any WeakDeps no longer needed for each node in topological order
        for node in self.nodes:
            node.prune_weak_deps()

    def topological_sort_schedule(self) -> None:
        """
        Ensure self.nodes is in topologically sorted order
        """
        seen: Set[BaseSchedulerNode] = set()
        name_to_node: Dict[str, BaseSchedulerNode] = dict()
        result: List[BaseSchedulerNode] = []

        # Define a recursive function to perform depth-first search
        def visit(n: BaseSchedulerNode) -> None:
            if n not in seen:
                seen.add(n)
                # Visit each unmet dependency in sorted order
                for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                    visit(name_to_node[dep.name])
                result.append(n)

        # Build name_to_node mapping for quick lookup
        for node in self.nodes:
            for name in node.get_names():
                name_to_node[name] = node

        # Perform DFS on each node to populate result in topological order
        for node in self.nodes:
            visit(node)

        self.nodes = result

    def compute_ancestors(self) -> None:
        """
        Populate each node.ancestors
        """
        # Note: self.nodes is topologically sorted
        name_to_ancestors: Dict[str, Set[str]] = {}
        
        # Iterate over each node to compute ancestors
        for node in self.nodes:
            ancestors = set()
            for dep in node.unmet_dependencies:
                ancestors.add(dep.name)
                ancestors |= name_to_ancestors[dep.name]
            name_to_ancestors[node.get_name()] = ancestors
            node.ancestors = ancestors

        # Set min_order and max_order for each node based on its position in topological order
        for order, node in enumerate(self.nodes):
            node.min_order = order
            node.max_order = order
    # 方法用于将节点合并成 FusedSchedulerNodes，修改了 self.nodes
    def fuse_nodes(self) -> None:
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.
        """
        # 进行 10 次节点融合尝试
        for i in range(10):
            # 记录每次融合尝试的起始节点数
            old_len = len(self.nodes)
            # 输出调试信息，显示当前融合尝试的进度和节点数
            fusion_log.debug(
                "===== attempting fusion (%d/10): %d nodes =====",
                i + 1,
                old_len,
            )
            # 调用实例方法进行一次节点融合
            self.fuse_nodes_once()
            # 计算融合后的节点数
            new_len = len(self.nodes)
            # 输出调试信息，显示融合本轮后的节点数变化
            fusion_log.debug(
                "completed fusion round (%d/10): fused %d nodes into %d nodes\n",
                i + 1,
                old_len,
                new_len,
            )
            # 如果融合后的节点数没有变化或者变为 1，则认为融合完成
            if new_len == old_len or new_len == 1:
                fusion_log.debug("===== fusion complete (%d iterations) =====", i + 1)
                break

    # 方法用于对融合后的节点列表进行基准测试，并返回执行时间和描述信息
    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> Tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        # 断言节点列表长度大于 0
        assert len(nodes) > 0
        # 获取第一个节点的设备信息
        device = nodes[0].get_device()
        # 将当前设备信息存储到实例变量中
        self.current_device = device
        # 根据设备获取后端信息
        backend = self.get_backend(device)
        # 调用后端方法对融合后的节点进行基准测试，返回执行时间和描述信息
        return backend.benchmark_fused_nodes(nodes)
    def finalize_multi_template_buffers(self) -> None:
        def replace_buffer(
            orig_node: ir.MultiTemplateBuffer, new_node: ir.Buffer
        ) -> None:
            # 获取新节点的名称并替换原始节点的名称
            replaced_name = new_node.name
            orig_name = orig_node.get_name()
            assert isinstance(orig_name, str) and isinstance(replaced_name, str)

            # 删除图中原名称对应的缓冲区
            del V.graph.name_to_buffer[replaced_name]
            # 更新新节点的名称为原始节点的名称
            new_node.name = orig_name

            # 找到原始节点在缓冲区列表中的位置并替换为新节点
            orig = V.graph.buffers.index(orig_node)
            V.graph.buffers.remove(new_node)
            V.graph.buffers[orig] = new_node
            # 更新图中原始节点名称到新节点的映射
            V.graph.name_to_buffer[orig_name] = new_node

        # 遍历所有节点
        for i, node in enumerate(self.nodes):
            # 检查节点类型是否为 SchedulerNode 并且其包含的节点类型为 MultiTemplateBuffer
            if isinstance(node, SchedulerNode) and isinstance(
                node.node, ir.MultiTemplateBuffer
            ):
                multi_node = node.node
                # 获取最小选择的节点并进行处理
                min_node_unfused, _ = multi_node.get_min_choice()

                # 如果最小选择的节点是 TritonTemplateCallerBase 类型，则进行特定处理
                if isinstance(
                    min_node_unfused,
                    torch._inductor.ir.TritonTemplateCallerBase,
                ):
                    node.node.finalize_as_triton_caller(min_node_unfused)
                    continue

                # 获取输出节点中的缓存对象
                out_tensorbox = min_node_unfused.output_node()
                out_storage = out_tensorbox.data
                assert isinstance(out_storage, ir.StorageBox)
                out_buffer = out_storage.data
                assert isinstance(out_buffer, ir.Buffer)

                # 将输出缓存的布局设置为多模板缓冲的布局
                out_buffer.layout = multi_node.layout
                # 替换多模板缓冲节点为输出缓存节点
                replace_buffer(multi_node, out_buffer)
                # 创建新的调度节点
                new_scheduler_node = self.create_scheduler_node(out_buffer)

                # 更新当前节点列表中的节点为新的调度节点
                self.nodes[i] = new_scheduler_node
                # 更新名称到节点的映射
                self.name_to_node[node.get_name()] = new_scheduler_node
                # 更新名称到融合节点的映射
                self.name_to_fused_node[node.get_name()] = new_scheduler_node

                # 复制原始节点的用户、最小顺序、最大顺序和最后使用时间
                new_scheduler_node.users = node.users
                new_scheduler_node.min_order = node.min_order
                new_scheduler_node.max_order = node.max_order
                new_scheduler_node.last_usage = node.last_usage
    # 将节点融合成 FusedSchedulerNodes，并更新 self.nodes
    def fuse_nodes_once(self) -> None:
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuse(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
        # 创建一个包含所有节点的集合，作为候选融合节点
        fused_nodes = set(self.nodes)
        
        # 如果启用了 DEBUG 级别的日志，记录候选节点信息
        if fusion_log.isEnabledFor(logging.DEBUG):
            fusion_log.debug("fuse_nodes_once, candidates:")
            for node in fused_nodes:
                fusion_log.debug("  " + node.debug_str_short())  # noqa: G003
        
        # 遍历所有可能的节点对，尝试进行融合
        for node1, node2 in self.get_possible_fusions():
            # 将节点名称映射到对应的融合节点
            node1 = self.name_to_fused_node[node1.get_first_name()]
            node2 = self.name_to_fused_node[node2.get_first_name()]
            
            # 检查节点是否可以融合，并且不会创建循环依赖
            if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
                node1, node2
            ):
                # 尝试通过融合提升性能
                if not self.speedup_by_fusion(node1, node2):
                    continue
                
                # 记录融合操作的调试信息
                fusion_log.debug(
                    "fusing %s with %s", node1.get_name(), node2.get_name()
                )

                # 根据 node1 的设备获取后端对象，并进行节点融合
                device = node1.get_device()
                node3 = self.get_backend(device).fuse(node1, node2)
                
                # 更新节点集合：移除融合的原始节点，添加新的融合节点
                fused_nodes.remove(node1)
                fused_nodes.remove(node2)
                fused_nodes.add(node3)
                
                # 更新名称到融合节点的映射
                self.name_to_fused_node.update(
                    {n.get_name(): node3 for n in node3.get_nodes()}
                )
        
        # 根据最小顺序对融合后的节点集合进行排序
        self.nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        
        # 对节点集合执行拓扑排序的调度
        self.topological_sort_schedule()
        
        # 剪除冗余依赖关系
        self.prune_redundant_deps()

    # 对 self.nodes 中的每个节点执行冗余依赖关系的剪除操作
    def prune_redundant_deps(self) -> None:
        for node in self.nodes:
            node.prune_redundant_deps(self.name_to_fused_node)
    def get_possible_fusions(self) -> List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        """
        Helper to find all legal fusion opportunities, sorted by self.score_fusion()
        """
        # 初始化一个空列表，用于存储所有可能的融合机会
        possible_fusions = []
        # 用于记录已经检查过的节点对，以避免重复检查
        seen = set()

        def check_all_pairs(nodes: List[BaseSchedulerNode]) -> None:
            # 遍历节点列表，找出所有可能的节点对
            for node1_index, node1 in enumerate(nodes):
                for node2 in nodes[node1_index + 1 :]:
                    key = (node1, node2)
                    if key in seen:
                        continue
                    seen.add(key)

                    if self.can_fuse(node1, node2):
                        # 如果两个节点可以融合，则将其添加到可能的融合机会列表中
                        possible_fusions.append(key)
                    elif (node2.is_template() or node2.is_foreach()) and self.can_fuse(
                        node2, node1
                    ):
                        # 对于模板节点或foreach节点，融合顺序是有关的，也添加到可能的融合机会列表中
                        possible_fusions.append((node2, node1))

        # 使用缓冲区名字作为键，将节点分组
        buffer_names_grouping = collections.defaultdict(list)
        for node in self.nodes:
            for buf in node.used_buffer_names():
                buffer_names_grouping[buf].append(node)
        # 遍历每个缓冲区名字的节点分组，检查其中的所有节点对
        for node_grouping in buffer_names_grouping.values():
            check_all_pairs(node_grouping)

        # 如果配置启用了激进的融合策略
        if config.aggressive_fusion:
            # 使用节点的组信息进行分组
            group_grouping = collections.defaultdict(list)
            for node in self.nodes:
                group = getattr(node, "group", None)
                if group:
                    group_grouping[group].append(node)
            # 遍历每个组的节点分组，检查其中的所有节点对
            for node_grouping in group_grouping.values():
                check_all_pairs(node_grouping)

        # 根据融合机会的优先级获取可能的融合机会，并按照得分函数排序
        possible_fusions = self.get_possible_fusions_with_highest_priority(
            possible_fusions
        )
        # 按照特定的排序键对融合机会列表进行逆序排序
        possible_fusions.sort(key=self.score_fusion_key, reverse=True)
        # 记录调试信息，表示找到了多少个可能的融合机会
        fusion_log.debug("found %d possible fusions", len(possible_fusions))
        # 返回所有可能的融合机会列表
        return possible_fusions

    def will_fusion_create_cycle(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        判断从 node1 到 node2（或反之）是否存在由其他融合节点间接引起的路径。
        """

        visited = set()

        def found_path(node: BaseSchedulerNode) -> bool:
            # 只有融合节点可以引入新的祖先节点。
            if isinstance(node, FusedSchedulerNode) and node not in visited:
                visited.add(node)
                if node.get_names().issubset(combined_ancestors):
                    # 所有融合输出都在 node1 和 node2 的祖先节点中，因此不能引入新路径：
                    #
                    # 1. 如果输出既不是 node1 的后代也不是 node2 的后代，则该输出不能引入路径。
                    # 2. 根据 [can_fuse]：如果假设输出是 node1 的后代，则它不能位于路径（node1->node2）上，因此它也不能是 node2 的祖先。
                    # 3. 根据 [acyclic]：如果假设输出是 node1 的后代，则它不能是 node1 的祖先。
                    return False
                else:
                    # 继续对融合引入的新祖先节点进行深度优先搜索（DFS）
                    return bool(combined_names & node.ancestors) or any(
                        found_path(self.name_to_fused_node[n])
                        for n in node.ancestors - combined_ancestors
                    )
            return False

        combined_names = node1.get_names() | node2.get_names()
        combined_ancestors = (node1.ancestors | node2.ancestors) - combined_names
        cycle = any(found_path(self.name_to_fused_node[n]) for n in combined_ancestors)
        if cycle:
            WhyNoFuse(node1, node2)("将会创建循环")
        return cycle

    def can_fusion_increase_peak_memory(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        This function prevents fusion for nodes that can increase memory
        footprint. This problem is more common in horizontal fusion, where nodes
        that are far apart in the original order get fused, lengthening the live
        intervals of tensors. This is very evident in models with activation
        checkpointing, where the recomputed nodes from different checkpointed
        regions get fused and significantly increase the memory footprint.

        The current attempt is a quick, possibly hacky, heuristic to prevent the
        fusion of nodes that are far away in the original order.

        A better but difficult to implement heuristic would be to use live
        intervals of the buffers, find the region of peak pressure in the original
        program, and prevent fusion that crosses that peak region. We might need
        special care or a good approximation in this implementation, as fusion of
        nodes changes live intervals, and recomputing live intervals and peak
        memory after each fusion can introduce large compilation overhead.
        """
        # Calculate the proximity score between node1 and node2 based on their scheduling order
        proximity_score = max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        # Return True if the proximity score is greater than 64, indicating nodes are too far apart for fusion
        return proximity_score > 64

    def decide_fusion_fail_reason(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        common_buf_names: Tuple[str, ...],
    ) -> str:
        """
        尝试确定融合失败的原因，即使有共同的缓冲区也没有共享内存。
        """
        # 初始化一个空字典，用于存储融合失败的原因
        reasons = {}

        # 创建 node1 的名称到依赖对象的映射字典
        node1_name2dep = {dep.name: dep for dep in node1.read_writes.reads_and_writes()}
        
        # 创建 node2 的名称到依赖对象的映射字典
        node2_name2dep = {dep.name: dep for dep in node2.read_writes.reads_and_writes()}

        # 遍历共同缓冲区的名称列表
        for buf_name in common_buf_names:
            # 获取缓冲区对象
            buf = V.graph.get_buffer(buf_name)
            
            # 获取 node1 中缓冲区名称对应的依赖对象
            lhs_dep = node1_name2dep[buf_name]
            
            # 获取 node2 中缓冲区名称对应的依赖对象
            rhs_dep = node2_name2dep[buf_name]

            # 检查依赖对象中元素数量是否不同
            if lhs_dep.get_numel() != rhs_dep.get_numel():
                reasons[
                    buf_name
                ] = f"different numel: {lhs_dep.get_numel()} v.s. {rhs_dep.get_numel()}"
                continue

            # 检查依赖对象的尺寸是否导致广播操作
            if sympy_product(lhs_dep.size) != sympy_product(rhs_dep.size):
                reasons[buf_name] = "broadcast"
                continue

            # 检查依赖对象是否为 MemoryDep 类型
            if not isinstance(lhs_dep, MemoryDep) or not isinstance(rhs_dep, MemoryDep):
                reasons[
                    buf_name
                ] = f"not MemoryDep: {type(lhs_dep)} v.s. {type(rhs_dep)}"
                continue

            # 检查依赖对象的偏移量是否不同
            lhs_off = lhs_dep.get_offset()
            rhs_off = rhs_dep.get_offset()
            if lhs_off != rhs_off:
                # 举例说明偏移量不同的情况
                reasons[buf_name] = f"different offset: {lhs_off} v.s. {rhs_off}"
                continue

            # 检查依赖对象的循环顺序是否匹配
            if (
                lhs_dep.normalize_with_stride_order()
                == rhs_dep.normalize_with_stride_order()
            ):
                reasons[buf_name] = f"Mismatch loop orders: {lhs_dep} v.s. {rhs_dep}"
                continue

            # 默认情况下，添加未知原因的条目
            reasons[
                buf_name
            ] = f"Unknown reason: {lhs_dep} v.s. {rhs_dep}. Layout: {buf.layout}"

        # 将字典转换为字符串并返回
        return str(reasons)
    ) -> bool:
        """
        检查是否可以将一个消费者节点（node2）融合到一个生产者节点（node1）中。

        如果node2的所有读取要么与node1中对应的写入匹配，要么由可以在融合node1和node2之前调度的节点写入，则可以进行融合。

        如果读取和写入不对齐，则禁用写入后续读取的融合。
        """
        # 获取node1的名称集合
        node1_names = node1.get_names()
        # 初始化计算后的依赖集合
        computed_deps = set()
        # 创建为何不融合的对象，传入node1和node2
        why = WhyNoFuse(node1, node2)

        # 遍历node1的读写操作中的写入
        for cd in node1.read_writes.writes:
            # 如果不是内存依赖，则跳过
            if not isinstance(cd, MemoryDep):
                continue
            # 遍历node2的未满足依赖
            for rd in node2.unmet_dependencies:
                # 如果可以融合读写操作
                if self.fusable_read_and_write(rd, cd):
                    computed_deps.add(rd)

        # 计算剩余的依赖名称
        remaining_deps = {dep.name for dep in node2.unmet_dependencies - computed_deps}
        # 如果剩余依赖名称与node1的名称有交集
        if remaining_deps & node1_names:
            # 内存依赖不匹配，并且读取了同一缓冲区的不同位置。
            # 示例包括：
            #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
            #   - MemoryDep("foo", x) != StarDep("foo")
            why("memory deps did not match")
            return False
        # 对于每个剩余依赖名称
        for name in remaining_deps:
            # 如果node1的名称与该依赖的融合节点的祖先有交集
            if node1_names & self.name_to_fused_node[name].ancestors:
                why("intermediate nodes between node1 & node2")
                return False

        # 类似于can_inplace，如果我们要将写入融合到读取后
        # 要求索引和大小相同
        for write in node2.read_writes.writes:
            # 如果不是内存依赖，则跳过
            if not isinstance(write, MemoryDep):
                continue
            # 遍历node1的读写操作中的读取
            for read in node1.read_writes.reads:
                # 如果写入名称与读取名称的变异重命名不匹配，则继续下一个循环
                if write.name != self.mutation_renames.get(read.name, read.name):
                    continue

                # 如果不能融合读写操作
                if not self.fusable_read_and_write(read, write):
                    why("fusing a write into a read with different indexing formula")
                    return False

        # 如果所有条件都满足，则返回True，表示可以融合
        return True

    # StarDep不匹配MemoryDep，不同的索引不匹配
    # 然而，广播有时会剥离维度，如果是这种情况，则仍然可以匹配未满足的依赖
    # 如果有间接索引，则不匹配它
    def fusable_read_and_write(self, read: Dep, write: MemoryDep) -> bool:
        # 如果 read 是 MemoryDep 类型
        if isinstance(read, MemoryDep):
            # 检查读和写的模式是否相同，并且写的模式不为空
            if read.mode == write.mode and write.mode is not None:
                return True
            # 获取读取操作的名称
            read_name = read.name
            # 如果读取操作的名称在 mutation_renames 中，使用其重命名后的名称
            if read_name in self.mutation_renames:
                read_name = self.mutation_renames[read_name]
            # 检查以下条件是否都成立以确定可融合性：
            return (
                read_name == write.name
                and not free_symbol_is_type(read.index, SymT.TMP)
                and not free_symbol_is_type(write.index, SymT.TMP)
                and read.index == write.index
                and len(read.size) >= len(write.size)
                and read.size[: len(write.size)] == write.size
            )
        # 如果 read 是 StarDep 类型
        elif isinstance(read, StarDep):
            # 获取读和写的重命名后的名称
            read_name = self.mutation_renames.get(read.name, read.name)
            write_name = self.mutation_renames.get(write.name, write.name)
            # 如果读和写的模式相同，并且重命名后的名称也相同，则返回 True
            if (
                read.mode == write.mode
                and write.mode is not None
                and read_name == write_name
            ):
                return True
        # 默认返回 False，表示不可融合
        return False

    def score_fusion(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> Tuple[bool, bool, int, int]:
        """
        为节点 node1 和 node2 分配一个分数（分数高的优先）。当不同的融合冲突时，
        这是我们决定运行它们的顺序的方式。

        当前的分数基于：
        - 估计的内存操作节省量
        - 原始顺序中融合的距离
        """
        # 计算内存操作分数
        memory_score = self.score_fusion_memory(node1, node2)
        # 计算原始顺序中融合的距离分数
        proximity_score = -max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        # 返回结果元组，包括两个布尔值和两个整数分数
        return (
            node1.is_template() == config.epilogue_fusion_first and memory_score > 0,
            node1.is_reduction() == node2.is_reduction() and memory_score > 0,
            memory_score,
            proximity_score,
        )

    def dep_size_hint(self, dep: Dep) -> int:
        res = 0
        # 如果 dep 不在缓存中
        if dep not in self.__dep_size_hint_cache:
            try:
                # 如果 dep 没有未备份的符号
                if not dep.has_unbacked_symbols():
                    # 获取 dep 的字节大小的提示
                    res = dep.numbytes_hint()
            except KeyError:
                # 在至少一个测试中（test/inductor/test_torchbind.py），
                # 我们创建了一个在图中不存在的 StarDep，并且调用 `has_unbacked_symbols()` 会抛出错误。
                pass
            # 将结果存入缓存中
            self.__dep_size_hint_cache[dep] = res
        else:
            # 如果 dep 在缓存中，直接获取缓存中的值
            res = self.__dep_size_hint_cache[dep]
        # 返回 dep 的大小提示
        return res

    def score_fusion_memory(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        # 实现计算两个节点之间内存操作分数的具体逻辑，这部分代码在提供的片段中未包含
        pass
    ) -> int:
        """
        The first term in our fusion score that estimates number of saved
        memory operations.
        """
        # Calculate common memory dependencies between node1 and node2
        common_memory_deps = (node1.read_writes.reads | node1.read_writes.writes) & (
            node2.read_writes.reads | node2.read_writes.writes
        )
        # Sum up the size hints for each common memory dependency
        return sum(self.dep_size_hint(dep) for dep in common_memory_deps)

    def get_possible_fusions_with_highest_priority(
        self, possible_fusions: List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]
    ) -> List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        # Group possible fusions by their priority from the backend
        # Return only the group of possible fusions with the highest priority
        if len(possible_fusions) == 0:
            return possible_fusions
        possible_fusions_group_by_priority: Dict[
            int, List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]
        ] = {}

        for node1, node2 in possible_fusions:
            # Ensure both nodes are on the same device
            assert node1.get_device() == node2.get_device()
            device = node1.get_device()
            # Get fusion pair priority from the backend for nodes node1 and node2
            fusion_pair_priority = int(
                self.get_backend(device).get_fusion_pair_priority(node1, node2)
            )
            if fusion_pair_priority not in possible_fusions_group_by_priority:
                possible_fusions_group_by_priority[fusion_pair_priority] = [
                    (node1, node2),
                ]
            else:
                possible_fusions_group_by_priority[fusion_pair_priority].append(
                    (node1, node2)
                )
        # Return the list of possible fusions with the highest priority
        possible_fusions_with_highest_priority = min(
            possible_fusions_group_by_priority.items(), key=operator.itemgetter(0)
        )[1]
        assert len(possible_fusions_with_highest_priority) > 0
        return possible_fusions_with_highest_priority

    def score_fusion_key(
        self, nodes: Tuple[BaseSchedulerNode, BaseSchedulerNode]
    ) -> Tuple[bool, bool, int, int]:
        """
        Shim for list.sort(key=...)
        """
        node1, node2 = nodes
        # Call score_fusion method to score the fusion of node1 and node2
        return self.score_fusion(node1, node2)

    def compute_last_usage(self) -> None:
        """
        Populate node.last_usage recursively (also for the nodes within a FusedSchedulerNode)
        """
        # Get the set of future used buffers from the graph's output names
        future_used_buffers = set(V.graph.get_output_names())

        # Iterate over nodes in reverse order and set their last_usage attributes
        for node in reversed(self.nodes):
            # Set last_usage for the current node considering future used buffers
            node.set_last_usage(future_used_buffers, self.mutation_real_name)
            # Update future_used_buffers to include the last_usage of the current node
            future_used_buffers.update(node.last_usage)
    def free_buffers(self) -> None:
        """Free any buffers that are no longer needed"""
        # 遍历需要释放的缓冲区名称集合，按名称排序
        for name in sorted(
            self.buffer_names_to_free
            - V.graph.removed_buffers
            - V.graph.wrapper_code.freed
        ):
            # 检查缓冲区名称是否存在于当前对象的名称到节点映射中
            if name in self.name_to_node:
                # 获取节点对象
                node = self.name_to_node[name]
                # 检查节点是否可以释放
                if node.can_free():
                    # 调用代码生成器释放节点对应的资源
                    V.graph.wrapper_code.codegen_free(node.node)
            # 如果名称存在于图输入的名称集合中
            elif name in V.graph.graph_inputs:
                # 获取存储对象
                storage = V.graph.graph_inputs[name].data
                # 断言存储对象为 StorageBox 类型且是输入缓冲区
                assert isinstance(storage, ir.StorageBox) and storage.is_input_buffer()
                # 调用代码生成器释放存储对象的数据
                V.graph.wrapper_code.codegen_free(storage.data)

        # 清空待释放缓冲区名称集合
        self.buffer_names_to_free.clear()

    def remove_kernel_local_buffers(self) -> None:
        """
        Any buffers that are both created and have a last use in the
        same kernel can be removed.
        """

        # V.kernel.store_buffer_names should represent the set of nodes
        # get fused
        # 获取已融合节点名称集合
        fused_node_names = V.kernel.store_buffer_names
        # 待移除的缓冲区名称列表
        names_to_remove = []
        # 遍历已融合节点名称集合
        for out_buf in V.kernel.store_buffer_names:
            # 获取节点的用户集合
            users = self.name_to_node[out_buf].users
            # 断言用户集合不为空
            assert users is not None
            # 转换用户集合为名称集合，过滤掉弱引用的用户
            users = {user.get_name() for user in users if not user.is_weak}
            # 如果节点的用户集合是已融合节点名称集合的子集
            if users.issubset(fused_node_names):
                # 将缓冲区名称添加到待移除列表中
                names_to_remove.append(out_buf)

        # 定义移除过滤函数
        def remove_filter(n: str) -> bool:
            return (
                n not in V.kernel.must_keep_buffers
                and n not in V.kernel.args.input_buffers
                and n not in self.mutation_renames
                and n not in self.mutation_real_name
            )

        # 使用移除过滤函数过滤待移除名称列表
        names_to_remove = list(filter(remove_filter, names_to_remove))

        # 遍历待移除名称列表
        for name in names_to_remove:
            # 如果名称存在于原地操作缓冲区集合中
            if name in V.kernel.args.inplace_buffers:
                # 获取原地操作缓冲区对象
                buf = V.kernel.args.inplace_buffers[name]
                # 如果缓冲区是字符串并且以 "REMOVED" 开头，则继续下一轮循环
                if isinstance(buf, str) and buf.startswith("REMOVED"):
                    continue
                # 检查缓冲区的其他名称是否都在待移除名称列表中
                remove = all(n in names_to_remove for n in buf.other_names)
                # 如果需要移除，则调用方法移除原地操作缓冲区
                if remove:
                    self.remove_inplace_buffer(name)
                # 将缓冲区名称添加到已标记待移除的原地操作缓冲区集合中
                V.kernel.inplaced_to_remove.add(name)
            else:
                # 否则调用方法移除缓冲区
                self.remove_buffer(name)

    def remove_buffer(self, name: str) -> None:
        # Assign a special value instead of deleting the entry
        # because we still rely on output_buffers's length to
        # generate unique arg name.
        # 记录调试日志
        log.debug("remove_buffer(%r)", name)
        # 将特殊值赋给输出缓冲区字典中的对应条目，而不是删除该条目
        # 因为我们仍然依赖于 output_buffers 的长度来生成唯一的参数名称
        V.kernel.args.output_buffers[name] = "REMOVED"
        # 将名称添加到已移除缓冲区集合中
        V.kernel.removed_buffers.add(name)

    def remove_inplace_buffer(self, name: str) -> None:
        # 记录调试日志
        log.debug("removing_inplace_buffer(%r)", name)
        # 获取原地操作缓冲区对象的内部名称
        inner_name = V.kernel.args.inplace_buffers[name].inner_name
        # 将原地操作缓冲区对象中的内部名称替换为 "REMOVED"
        V.kernel.args.inplace_buffers[name] = inner_name.replace(
            "in_out_ptr", "REMOVED"
        )
        # 将名称添加到已移除缓冲区集合中
        V.kernel.removed_buffers.add(name)
    # 刷新所有后端的状态
    def flush(self) -> None:
        for backend in self.backends.values():
            backend.flush()
        self.free_buffers()

    # 生成外部调用的代码，用于调度外部内核
    def codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode) -> None:
        assert isinstance(scheduler_node, ExternKernelSchedulerNode)
        # 'decide_inplace_update' 方法在当前内核中存储就地更新的决策，
        # 这些决策将由 'allocate' 方法检索使用。
        # 我们必须确保有一个非空的内核处理程序来存储这些就地更新的决策。
        counters["inductor"]["extern_calls"] += 1
        # 使用虚拟环境中的内核处理程序，不增加内核计数
        with V.set_kernel_handler(Kernel(increase_kernel_count=False)):
            scheduler_node.decide_inplace_update()
            scheduler_node.allocate()
        node = scheduler_node.node
        assert isinstance(node, ir.ExternKernel), f"{type(node)=}"
        node.codegen(V.graph.wrapper_code)
        self.free_buffers()

    # 根据设备创建后端调度器对象
    def create_backend(self, device: torch.device) -> BaseScheduling:
        assert (
            not is_gpu(device.type) or device.index is not None
        ), f"{device} should have been normalized in lowering"
        # 添加设备信息到虚拟图中
        V.graph.add_device_info(device)

        # 获取适用于设备类型的调度器
        device_scheduling = get_scheduling_for_device(device.type)
        if device_scheduling is None:
            raise RuntimeError(f"Unsupported device type: {device.type}")

        # 检查是否有 Triton 支持
        if not has_triton():
            if (
                device.type == "cuda"
                and (device_props := torch.cuda.get_device_properties(device)).major < 7
            ):
                raise RuntimeError(
                    f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability {device_props.major}.{device_props.minor}"  # noqa: B950
                )
            elif is_gpu(device.type):
                raise RuntimeError(
                    "Cannot find a working triton installation. More information on installing Triton can be found at https://github.com/openai/triton"  # noqa: B950
                )

        # 使用设备调度器创建后端调度对象
        return device_scheduling(self)

    # 获取特定设备的后端调度对象，如果不存在则创建
    def get_backend(self, device: torch.device) -> BaseScheduling:
        if device not in self.backends:
            self.backends[device] = self.create_backend(device)
        return self.backends[device]
    # 将指定节点添加到上下文中，以便处理节点的顺序
    def enter_context(self, node: BaseSchedulerNode) -> None:
        # 获取节点的顺序号
        def get_order(n: torch.fx.Node) -> int:
            # 如果节点不在 origin_to_index 字典中，则更新字典
            if n not in self.origin_to_index:
                self.origin_to_index.update({n: i for i, n in enumerate(n.graph.nodes)})
            return self.origin_to_index[n]

        # 使用字典来保证顺序
        origins = {
            # 对于每个节点 n 中的每个来源 e，按顺序获取节点
            (get_order(e), e): None
            for n in node.get_nodes()  # 对于每个节点 n
            if n.node is not None  # 如果节点存在
            for e in n.node.origins  # 对于节点 n 的每个来源 e
        }
        # 将字典的键转换为列表
        origins = list(origins.keys())
        # 如果 origins 非空
        if origins:
            # 获取最大顺序号的节点
            _, last = max(origins, key=operator.itemgetter(0))
            # 进入节点的包装代码上下文
            V.graph.wrapper_code.enter_context(last)

    # 使用动态时间装饰器，获取指定缓冲区名称的布局信息
    @dynamo_timed
    def get_buffer_layout(self, buf_name: str) -> ir.Layout:
        # 获取缓冲区名称对应的节点
        node = self.name_to_node[buf_name]
        # 断言节点存在
        assert node.node is not None
        # 返回节点的布局信息
        return node.node.get_layout()
class BaseScheduling:
    @classmethod
    def get_backend_features(cls, device: torch.device) -> Sequence[BackendFeature]:
        """Return a set of .codegen.common.BackendFeature()"""
        return ()

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be vertically fused or not.
        """
        raise NotImplementedError

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be horizontally fused or not.
        """
        raise NotImplementedError

    def fuse(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> FusedSchedulerNode:
        """
        Fuse two nodes

        If either node1 or node2 is a foreach node, use ForeachKernelSchedulerNode to fuse them,
        otherwise use FusedSchedulerNode.
        """
        if node1.is_foreach() or node2.is_foreach():
            return ForeachKernelSchedulerNode.fuse(node1, node2)
        else:
            return FusedSchedulerNode.fuse(node1, node2)

    def group_fn(
        self, sizes: Sequence[Sequence[sympy.Expr]]
    ) -> Tuple[Tuple[sympy.Expr, ...], ...]:
        """
        Process the iteration sizes in case a transformation needs to be applied.
        """
        raise NotImplementedError

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
    ) -> Optional[str]:
        """
        Given a template node, generate a kernel.

        This function is only available for triton now. If the third-party backend behaves as a sub-class
        of TritonScheduling, it can override it or reuse it.
        """
        raise NotImplementedError

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]) -> None:
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        raise NotImplementedError

    def codegen_sync(self) -> None:
        """
        Generate synchronization code for the kernel. This method depends on the hardware characteristics.
        """
        raise NotImplementedError

    def ready_to_flush(self) -> bool:
        """
        Check whether the backend is requesting the scheduler to flush the generated kernel.
        If not supported, please return False.
        """
        return False

    def flush(self) -> None:
        """
        Flush the generated kernel and python wrapper code to the source code file.
        """
        raise NotImplementedError

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> Tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        raise NotImplementedError

    def get_fusion_pair_priority(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        """
        Return the priority of fusing node1 and node2.

        This is used to determine the order of fusion when multiple fusion opportunities exist.
        """
        raise NotImplementedError
    ) -> int:
        """
        返回一个表示融合对优先级的无符号整数。
        数值越小，优先级越高。
        """
        return 0
# 定义一个函数，用于调试 Triton 代码生成的函数
def debug_triton_code(node: Union[SchedulerNode, FusedSchedulerNode]) -> List[str]:
    # 初始化一个空列表，用于存储输出的调试信息的每一行
    lines = []
    
    # 获取节点的模板节点（如果有的话）
    multi_template = node.get_template_node()
    
    # 断言：如果 multi_template 不是 None，则它必须是 MultiTemplateBuffer 类的实例
    assert multi_template is None or isinstance(multi_template, ir.MultiTemplateBuffer)
    
    # 检查是否存在未完成的模板，如果存在，则输出相应的调试信息
    if multi_template and multi_template.make_kernel_render is None:
        lines.append(f"{node.get_name()} Unfinalized multi template buffer")
    else:
        # 导入相关的模块和类
        from torch._inductor.codegen.cuda_combined_scheduling import (
            CUDACombinedScheduling,
        )
        from .codegen.simd import SIMDScheduling

        # 确定要处理的节点列表
        snodes = (node,) if isinstance(node, SchedulerNode) else node.snodes
        
        # 获取节点的设备信息
        device = snodes[0].get_device()
        
        # 获取节点的后端调度器信息
        backend = node.scheduler.get_backend(device)
        
        # 断言：后端调度器必须是 SIMDScheduling 或 CUDACombinedScheduling 类的实例
        assert isinstance(backend, (SIMDScheduling, CUDACombinedScheduling))
        
        # 设置当前的设备为节点的设备
        V.graph.scheduler.current_device = device

        # 在生成调试字符串时不增加内核计数
        # 这会导致某些单元测试检查生成的内核数量时出现混乱
        old_generated_kernel_count = metrics.generated_kernel_count
        
        # 从节点生成内核代码，并去除两侧空白字符
        triton_code = backend.generate_kernel_code_from_nodes(snodes).strip()
        
        # 恢复旧的生成内核计数
        metrics.generated_kernel_count = old_generated_kernel_count

        # 将节点名称和 Triton 生成的代码添加到输出行列表中
        lines.append(f"{node.get_name()} Triton code:")
        lines.append(textwrap.indent(triton_code, "    "))
    
    # 返回包含调试信息的行列表
    return lines
```