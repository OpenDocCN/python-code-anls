# `.\pytorch\torch\_inductor\codegen\rocm\rocm_cpp_scheduling.py`

```
# mypy: allow-untyped-defs
# 引入日志模块
import logging
# 引入类型注解相关模块
from typing import cast, Sequence

# 引入配置模块
from ... import config
# 引入代码缓存相关模块
from ...codecache import code_hash, get_path
# 引入调度器相关模块
from ...scheduler import BaseSchedulerNode, BaseScheduling, Scheduler, SchedulerNode
# 引入实用工具相关模块
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
# 引入虚拟化相关模块
from ...virtualized import V
# 引入通用模块
from ..common import IndentedBuffer

# 引入 ROCm 模板缓冲类
from .rocm_template_buffer import ROCmTemplateBuffer

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


class ROCmCPPScheduling(BaseScheduling):
    """
    ROCm C++ 内核的部分调度实现。
    此类旨在与 TritonScheduling 结合使用，并由 CUDACombinedScheduling 委托使用。

    处理融合决策和 ROCm C++ 特定模板代码生成。
    """

    def __init__(self, scheduler: Scheduler):
        # 调用父类构造函数
        super().__init__()
        # 初始化调度器属性
        self.scheduler = scheduler

    def group_fn(self, sizes):
        # 简化尺寸变量并组成元组返回
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_rocm_cpp_template(node: BaseSchedulerNode) -> bool:
        # 判断节点是否为 ROCm 模板缓冲节点
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, ROCmTemplateBuffer
        )

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # 垂直融合判断，始终返回 False
        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
        # 获取图形封装器代码
        wrapper = V.graph.wrapper_code
        # 如果源代码已在映射中，则获取内核名称
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            # 根据配置决定是否使用描述性名称
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            # 构建内核名称
            kernel_name = "_".join(["rocm", fused_name, wrapper.next_kernel_suffix()])
            # 使用源代码作为键将内核名称映射到图形封装器中
            wrapper.src_to_kernel[src_code] = kernel_name
            # 替换源代码中的 KERNEL_NAME 为内核名称
            src_code = src_code.replace("KERNEL_NAME", kernel_name)

            # 计算代码的哈希值并获取代码路径
            _, _, kernel_path = get_path(code_hash(src_code), "py")

            # 创建缓冲区用于编译包装
            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline("async_compile.rocm(r'''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline("''', 'so')")

            # 构建内核的元数据注释
            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            # 定义内核到图形封装器中
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        # 返回内核名称
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a ROCm template, possibly with fused epilogues
        """
        # 断言传入的模板节点是 ROCmScheduler 生成的 ROCmTemplateBuffer 包装的 SchedulerNode
        assert self.is_rocm_cpp_template(
            template_node
        ), "Template node passed to ROCmScheduler.codegen_template must be a SchedulerNode that wraps a ROCmTemplateBuffer"
        # 将 template_node 强制转换为 SchedulerNode 类型
        template_node = cast(SchedulerNode, template_node)
        # 从 template_node 的组属性中获取 numel 和 rnumel 的值
        _, (numel, rnumel) = template_node.group
        # 断言 rnumel 的值为 1
        assert rnumel == 1
        # 将 template_node.node 强制转换为 ROCmTemplateBuffer 类型
        ctb: ROCmTemplateBuffer = cast(ROCmTemplateBuffer, template_node.node)
        # 使用 ctb.make_kernel_render 方法生成 kernel 和 render 函数
        kernel, render = ctb.make_kernel_render(ctb)
        # 进入 kernel 上下文
        with kernel:
            # 标记 template_node 为已运行状态
            template_node.mark_run()
            # 调用 render 函数生成源代码
            src_code = render()

        # 使用 kernel 设置为当前的 kernel 处理器
        with V.set_kernel_handler(kernel):
            # 将 template_node 添加到 node_schedule 中
            node_schedule = [template_node]
            # 使用生成的 src_code 和 node_schedule 定义 kernel 名称
            kernel_name = self.define_kernel(src_code, node_schedule)
        # 调用 kernel 的 call_kernel 方法，并传入 kernel_name 和 ctb
        kernel.call_kernel(kernel_name, ctb)
        # 将 kernel 的 removed_buffers 加入 V.graph.removed_buffers
        V.graph.removed_buffers |= kernel.removed_buffers
        # 释放 scheduler 中的缓冲区
        self.scheduler.free_buffers()
```