# `.\pytorch\torch\_inductor\codegen\cuda\cuda_cpp_scheduling.py`

```py
# mypy: allow-untyped-defs
# 导入日志模块，用于记录程序运行信息
import logging
# 引入类型转换函数和序列类型
from typing import cast, Sequence

# 引入计数器工具模块
from ...._dynamo.utils import counters

# 引入配置模块
from ... import config
# 引入代码缓存相关模块
from ...codecache import code_hash, get_path

# 引入IR模块中的CUDA模板缓冲类
from ...ir import CUDATemplateBuffer
# 引入调度器相关模块和节点类
from ...scheduler import BaseSchedulerNode, BaseScheduling, Scheduler, SchedulerNode
# 引入工具函数模块
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
# 引入虚拟化模块
from ...virtualized import V
# 引入通用缩进缓冲类
from ..common import IndentedBuffer

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


class CUDACPPScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for CUDA C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling.

    It handles fusion decisions and CUDA C++ specific template code generation.
    """

    def __init__(self, scheduler: Scheduler):
        super().__init__()
        # 初始化调度器对象
        self.scheduler = scheduler

    @classmethod
    def get_backend_features(cls, device):
        # 返回空字典，表示不返回任何后端特性信息
        return {}

    def group_fn(self, sizes):
        # 对给定大小进行分组处理，并返回简化后的大小元组
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_cuda_cpp_template(node: BaseSchedulerNode) -> bool:
        # 判断节点是否为CUDA C++模板节点
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, CUDATemplateBuffer
        )

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # 垂直融合判定函数，始终返回False，表示不允许融合
        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
        # 定义内核函数，根据源代码生成内核名称并进行编译
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            # 如果源代码已存在对应的内核名称，直接获取内核名称
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            # 否则根据调度信息生成融合内核名称
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_name = "_".join(["cuda", fused_name, wrapper.next_kernel_suffix()])
            # 将原始的src_code用作键存储内核名称映射
            wrapper.src_to_kernel[src_code] = kernel_name
            # 替换源代码中的"KERNEL_NAME"为生成的内核名称
            src_code = src_code.replace("KERNEL_NAME", kernel_name)

            # 计算源代码的哈希值和生成对应的路径
            _, _, kernel_path = get_path(code_hash(src_code), "py")

            # 创建编译包装器，并进行异步CUDA编译
            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline("async_compile.cuda(r'''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline("''', 'so')")

            # 添加内核元数据注释
            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            # 在包装器中定义内核
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        """
        Codegen a CUDA template, possibly with fused epilogues
        """
        # 增加 CUDA 模板的计数器，用于融合结尾节点
        counters["inductor"]["cuda_epilogue_fusion_counter"] += len(epilogue_nodes)
        
        # 断言模板节点是 CUDA C++ 模板，并确保其包装的是 CUDATemplateBuffer 的 SchedulerNode
        assert self.is_cuda_cpp_template(
            template_node
        ), "Template node passed to CUDAScheduler.codegen_template must be a SchedulerNode that wraps a CUDATemplateBuffer"
        
        # 将模板节点强制类型转换为 SchedulerNode，获取其中的 group 属性中的元组 (numel, rnumel)
        template_node = cast(SchedulerNode, template_node)
        _, (numel, rnumel) = template_node.group
        
        # 断言 rnumel 等于 1
        assert rnumel == 1
        
        # 将 template_node.node 强制类型转换为 CUDATemplateBuffer，命名为 ctb
        ctb: CUDATemplateBuffer = cast(CUDATemplateBuffer, template_node.node)
        
        # 使用 CUDATemplateBuffer 的 make_kernel_render 方法生成 kernel 和 render 函数
        kernel, render = ctb.make_kernel_render(ctb)
        
        # 进入 kernel 的上下文管理器
        with kernel:
            template_node.mark_run()  # 标记模板节点已运行
            src_code = render()  # 调用 render 函数获取源代码
        
        # 进入 V.set_kernel_handler(kernel) 的上下文管理器
        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]  # 将 template_node 添加到节点调度列表
            kernel_name = self.define_kernel(src_code, node_schedule)  # 定义 kernel 名称
        kernel.call_kernel(kernel_name, ctb)  # 调用 kernel 的 call_kernel 方法执行 kernel
        V.graph.removed_buffers |= kernel.removed_buffers  # 将 kernel 移除的缓冲区添加到 removed_buffers
        self.scheduler.free_buffers()  # 释放调度器中的缓冲区
```