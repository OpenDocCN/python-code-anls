# `.\pytorch\torch\_inductor\codegen\triton_foreach.py`

```
# mypy: allow-untyped-defs
# 引入 itertools 模块，用于生成迭代器和处理迭代器操作的工具函数
import itertools
# 从 collections 模块中引入 defaultdict 类，用于创建默认字典，简化字典数据处理
from collections import defaultdict
# 从 dataclasses 模块中引入 dataclass 装饰器，用于简化创建和管理数据类的过程
from dataclasses import dataclass
# 从 typing 模块中引入需要的类型提示：Dict（字典类型）、List（列表类型）、Tuple（元组类型）
from typing import Dict, List, Tuple

# 从 sympy 模块中引入 Integer 类，用于处理大整数
from sympy import Integer

# 从上级模块中引入 metrics 模块
from .. import metrics
# 从 runtime.hints 模块中引入 DeviceProperties 类型提示
from ..runtime.hints import DeviceProperties
# 从 scheduler 模块中引入 SchedulerNode 类
from ..scheduler import SchedulerNode
# 从 utils 模块中引入 ceildiv 函数（向上取整除法）、Placeholder 类
from ..utils import ceildiv, Placeholder
# 从 virtualized 模块中引入 V 对象
from ..virtualized import V
# 从当前目录的 common 模块中引入 IndentedBuffer 类和 Kernel 类
from .common import IndentedBuffer, Kernel
# 从当前目录的 triton 模块中引入 gen_common_triton_imports 函数和 TritonKernel 类
from .triton import gen_common_triton_imports, TritonKernel
# 从当前目录的 triton_utils 模块中引入 config_of 函数和 signature_to_meta 函数
from .triton_utils import config_of, signature_to_meta


@dataclass
# 定义 PartitionState 数据类，包含 partitions（分区列表）、cur_partition（当前分区）、cur_count（当前计数）
class PartitionState:
    # partitions 属性是一个列表，每个元素是一个列表，包含多个 SchedulerNode 节点、元组、整数和整数
    partitions: List[
        List[Tuple[List[SchedulerNode], Tuple[Integer, ...], Integer, Integer]]
    ]
    # cur_partition 属性是一个列表，包含多个 SchedulerNode 节点、元组、整数和整数
    cur_partition: List[
        Tuple[List[SchedulerNode], Tuple[Integer, ...], Integer, Integer]
    ]
    # cur_count 属性表示当前计数，是一个整数
    cur_count: int

    # finalize 方法用于完成分区，如果 cur_partition 不为空，则将其添加到 partitions 中
    def finalize(self):
        if self.cur_partition:
            self.partitions.append(self.cur_partition)


# 定义 ForeachKernel 类，继承自 Kernel 类
class ForeachKernel(Kernel):
    # MAX_NUM_ARGS 类属性定义最大参数数量，当节点数量超过此值时不再出现 Triton 错误
    MAX_NUM_ARGS = 250  # number where I would no longer get triton errors

    # _update_partition 方法用于更新分区状态
    @staticmethod
    def _update_partition(partition_state, node_rw_count, node_info):
        # 如果当前计数加上节点读写次数超过最大参数数量
        if partition_state.cur_count + node_rw_count > ForeachKernel.MAX_NUM_ARGS:
            # 将当前分区添加到 partitions 中
            partition_state.partitions.append(partition_state.cur_partition)
            # 重置当前分区为包含当前节点信息的新列表
            partition_state.cur_partition = [node_info]
            # 更新当前计数为节点读写次数
            partition_state.cur_count = node_rw_count
        else:
            # 否则，增加当前计数
            partition_state.cur_count += node_rw_count
            # 将当前节点信息添加到当前分区中
            partition_state.cur_partition.append(node_info)

    # 静态方法定义结束
    def horizontal_partition(subkernel_nodes, triton_scheduling):
        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)
        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args
        (read/writes) and to have the same 2D or 1D blocking strategy."""
        # 确保子内核节点列表不为空
        assert len(subkernel_nodes) >= 1

        # 初始化一维分区状态
        partition_state_1d = PartitionState([], [], 0)
        # 创建一个字典，用于存储二维分区状态，键为 y 元素，值为分区状态对象
        yelem_to_partition_state_2d: Dict[Integer, PartitionState] = defaultdict(
            lambda: PartitionState([], [], 0)
        )

        # 遍历子内核节点
        for node in subkernel_nodes:
            # 获取融合节点
            fused_nodes = node.get_nodes()
            # 获取融合节点中具有最大 numel 和 rnumel 的节点
            _, (numel, rnumel) = max(
                fused_nodes, key=lambda x: int(x.is_reduction())
            ).group
            # 选择 tiling 策略
            tiled_groups = triton_scheduling.select_tiling(fused_nodes, numel, rnumel)
            # 构建节点信息元组
            node_info = fused_nodes, tiled_groups, numel, rnumel

            # 获取节点的读写操作
            read_writes = node.read_writes
            read_write_count = len(read_writes.reads) + len(read_writes.writes)

            # 根据 tiling 策略更新分区状态
            if tiled_groups[1] == 1:
                ForeachKernel._update_partition(
                    partition_state_1d, read_write_count, node_info
                )
            else:
                y_elem = tiled_groups[0]
                partition_state_2d = yelem_to_partition_state_2d[y_elem]
                ForeachKernel._update_partition(
                    partition_state_2d, read_write_count, node_info
                )

        # 完成一维分区状态
        partition_state_1d.finalize()
        all_partitions = partition_state_1d.partitions
        # 完成二维分区状态
        for partition_state_2d in yelem_to_partition_state_2d.values():
            partition_state_2d.finalize()
            all_partitions.extend(partition_state_2d.partitions)

        return all_partitions

    def __init__(self):
        super().__init__()
        # 初始化属性
        self.blocking_2d = False
        self.block_size_1d = 1024  # 尝试调整此值
        self.block_size_2d = 32
        self.num_warps = 8
        self.sub_kernels = []
        self.iter_vars_count = itertools.count()
        self.x_block_count = 0
        self.y_block_count = 0

    def get_block_size(self):
        # 根据 blocking_2d 属性返回相应的块大小
        if self.blocking_2d:
            return self.block_size_2d
        else:
            return self.block_size_1d

    @staticmethod
    def codegen_pid_offsets(code, block_count, lower_bound, prefix):
        # 根据块数量生成代码中的偏移量
        if block_count == 0:
            code.splice(f"{prefix}pid_offset = {prefix}pid")
        else:
            code.splice(f"{prefix}pid_offset = {prefix}pid - {lower_bound}")
    # 生成指定代码和元素数量相关的 PID 范围
    def codegen_pid_range(self, code, x_elems):
        # 计算需要的 x 块数目，ceildiv 是向上取整除法函数
        num_x_blocks = ceildiv(x_elems, self.get_block_size())
        # 计算上界 x PID，当前 x 块数目加上计算得到的 x 块数目
        upper_bound_x_pid = self.x_block_count + num_x_blocks
        # 计算下界 x PID，当前 x 块数目
        lower_bound_x_pid = self.x_block_count

        # 根据当前 x 块数目确定条件
        if self.x_block_count == 0:
            cond = "if"
        else:
            cond = "elif"

        # 生成 x PID 范围检查条件字符串
        x_pid_bounds_check = (
            f"xpid >= {lower_bound_x_pid} and xpid < {upper_bound_x_pid}"
        )
        # 在代码对象中插入条件语句
        code.splice(f"{cond} {x_pid_bounds_check}:")

        # 缩进处理以下代码块
        with code.indent():
            # 调用方法生成 x PID 偏移
            ForeachKernel.codegen_pid_offsets(
                code, num_x_blocks, lower_bound_x_pid, "x"
            )
            # 更新 x 块数目
            self.x_block_count += num_x_blocks

    # 创建子内核
    def create_sub_kernel(self, *groups, index_dtype, mutations, reduction_hint):
        # 创建 TritonKernel 对象，传入参数进行初始化
        sub_kernel = TritonKernel(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            pid_cache={
                "tl.program_id(0)": "xpid_offset",
                "tl.program_id(1)": "ypid",
            },
            reduction_hint=reduction_hint,
        )
        # 如果启用了二维阻塞，则验证 groups 的长度为 3
        if self.blocking_2d:
            assert len(groups) == 3

        # 更新二维阻塞状态
        self.blocking_2d |= groups[1] != 1 and len(groups) == 3
        # 生成的内核计数减一
        metrics.generated_kernel_count -= 1
        # 设置子内核的参数
        sub_kernel.args = self.args
        # 设置子内核的迭代变量计数
        sub_kernel.iter_vars_count = self.iter_vars_count
        # 复制公共子表达式的迭代缓冲区 ID
        sub_kernel.cse.iter_buffer_ids = self.cse.iter_buffer_ids
        # 将子内核添加到父内核的子内核列表中
        self.sub_kernels.append(sub_kernel)
        # 返回创建的子内核对象
        return sub_kernel

    # JIT 行
    def jit_lines(self):
        # 检查是否所有子内核的索引类型都是 "tl.int32"
        can_use_32bit = all(k.index_dtype == "tl.int32" for k in self.sub_kernels)
        # 确定 size_dtype 的类型，取决于是否可以使用 32 位
        size_dtype = "tl.int32" if can_use_32bit else "tl.int64"
        # 从参数中获取函数的 Python 签名信息
        _, _, signature, _ = self.args.python_argdefs()
        # 构建 Triton 元数据字典
        triton_meta = {
            "signature": signature_to_meta(signature, size_dtype=size_dtype),
            "device": DeviceProperties.create(
                V.graph.scheduler.get_current_device_or_throw()
            ),
            "constants": {},
        }
        # 将配置信息添加到 Triton 元数据中
        triton_meta["configs"] = [config_of(signature)]
        # 构建感应器元数据字典
        inductor_meta = {
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            **TritonKernel.inductor_meta_common(),
        }
        # 返回 JIT 行字符串
        return f"""
            @triton_heuristics.foreach(
                num_warps={self.num_warps},
                triton_meta={triton_meta!r},
                inductor_meta={inductor_meta!r},
            )
            @triton.jit
        """

    # 网格函数
    def grid(self):
        # 返回网格大小元组
        return (
            self.x_block_count,
            ceildiv(int(self.sub_kernels[0].numels[0]), self.block_size_2d)
            if self.blocking_2d
            else 1,
            1,
        )
    # 定义一个方法用于生成代码块，初始化一个缓冲区
    def codegen_kernel(self, name=None):
        code = IndentedBuffer()

        # 将共通的 Triton 导入代码添加到缓冲区中
        code.splice(gen_common_triton_imports())
        # 获取参数的默认定义，并将 JIT 编译后的代码添加到缓冲区中
        argdefs, _, _, _ = self.args.python_argdefs()
        code.splice(self.jit_lines())
        # 定义一个函数，函数名为给定的 name 参数或默认的 Placeholder.KERNEL_NAME
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
        )

        # 进入代码块的缩进部分
        with code.indent():
            # 添加代码到缓冲区，设置程序的 XPID
            code.splice("xpid = tl.program_id(0)")
            # 如果使用二维阻塞模式
            if self.blocking_2d:
                # 设置程序的 YPID
                code.splice("ypid = tl.program_id(1)")
                # 设置 XBLOCK 和 YBLOCK 的常量表达式为 block_size_2d 的值
                code.splice(f"XBLOCK: tl.constexpr = {self.block_size_2d}")
                code.splice(f"YBLOCK: tl.constexpr = {self.block_size_2d}")
            else:
                # 设置 XBLOCK 的常量表达式为 block_size_1d 的值
                code.splice(f"XBLOCK: tl.constexpr = {self.block_size_1d}")

            # 遍历所有子内核
            for sub_kernel in self.sub_kernels:
                # 断言子内核的 numels 数组长度不超过 3
                assert len(sub_kernel.numels) <= 3
                # TODO mlazos: 支持动态形状
                # 根据阻塞模式选择 numels 数组中的索引
                numel_ind = 0 if not self.blocking_2d else 1
                # 生成 pid 范围代码，并添加到缓冲区
                self.codegen_pid_range(code, int(sub_kernel.numels[numel_ind]))
                with code.indent():
                    # 如果是二维阻塞模式，设置 ynumel 和 xnumel 的值
                    if self.blocking_2d:
                        code.splice(f"ynumel = {sub_kernel.numels[0]}")
                        code.splice(f"xnumel = {sub_kernel.numels[1]}")
                    else:
                        # 否则，设置 xnumel 的值
                        code.splice(f"xnumel = {sub_kernel.numels[0]}")

                    # 生成子内核的主体代码，并添加到缓冲区
                    sub_kernel.codegen_body()
                    code.splice(sub_kernel.body)

            # 如果没有匹配的情况
            code.splice("else:")
            with code.indent():
                # 添加 pass 语句到缓冲区
                code.splice("pass")

        # 返回生成的完整代码
        return code.getvalue()

    # 定义一个方法用于调用内核函数
    def call_kernel(self, code, name: str):
        # 获取参数的调用定义，调用函数，并添加到缓冲区中
        _, call_args, _, arg_types = self.args.python_argdefs()
        V.graph.wrapper_code.generate_kernel_call(
            name,
            call_args,
            grid=self.grid(),
            arg_types=arg_types,
            grid_fn="",
        )
```