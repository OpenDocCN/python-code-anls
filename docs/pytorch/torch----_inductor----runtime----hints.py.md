# `.\pytorch\torch\_inductor\runtime\hints.py`

```py
# 设置 mypy：允许未打印的定义，这允许在未显式声明类型的情况下使用函数。
import collections
import typing
from dataclasses import fields
from enum import auto, Enum
from typing import Dict, List, Optional, Union

# 定义 Triton 最大的块大小字典，指定不同维度的最大尺寸限制
TRITON_MAX_BLOCK = {
    "X": 2048,
    "Y": 1024,
    "Z": 1024,
    "R": 4096 * 16,  # * 16 表示仅适用于多内核场景
}

# 定义减少提示枚举
class ReductionHint(Enum):
    INNER = 0
    OUTER = 1
    OUTER_TINY = 2
    DEFAULT = 3

# 定义切片提示枚举
class TileHint(Enum):
    SQUARE = 0
    DEFAULT = 1

# 尝试从 Triton 中导入 AttrsDescriptor
try:
    from triton.compiler.compiler import AttrsDescriptor

    attrs_descriptor_available = True
    # 确定 'ids_of_folded_args' 是否为 AttrsDescriptor 的有效字段
    attr_desc_fields = {f.name for f in fields(AttrsDescriptor)}
    ids_of_folded_args_available = "ids_of_folded_args" in attr_desc_fields
    divisible_by_8_available = "divisible_by_8" in attr_desc_fields
except ImportError:
    attrs_descriptor_available = False

# 当 AttrsDescriptor 不可用时定义命名元组作为后备
instance_descriptor = None

# 如果 attrs_descriptor_available 为真，则定义 instance_descriptor 函数来处理条件分支
if attrs_descriptor_available:

    def instance_descriptor(
        divisible_by_16=None,
        equal_to_1=None,
        ids_of_folded_args=None,
        divisible_by_8=None,
    ):
        # 准备用于 AttrsDescriptor 的参数
        kwargs = {
            "divisible_by_16": divisible_by_16,
            "equal_to_1": equal_to_1,
        }

        # 根据 AttrsDescriptor 的可用性，条件添加 'ids_of_folded_args'
        if ids_of_folded_args_available:
            kwargs["ids_of_folded_args"] = ids_of_folded_args
        if divisible_by_8_available:
            kwargs["divisible_by_8"] = divisible_by_8

        # 使用准备好的参数实例化 AttrsDescriptor
        return AttrsDescriptor(**kwargs)

else:
    # 在 AttrsDescriptor 不可用时，定义命名元组作为 instance_descriptor 的后备
    instance_descriptor = collections.namedtuple(  # type: ignore[no-redef]
        "instance_descriptor",
        ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"],
        defaults=[tuple(), tuple(), tuple(), tuple()],
    )

# 每个线程束中的线程数目
_NUM_THREADS_PER_WARP = 32

# 定义启发类型枚举
class HeuristicType(Enum):
    PERSISTENT_REDUCTION = auto()
    POINTWISE = auto()
    REDUCTION = auto()
    SPLIT_SCAN = auto()
    TEMPLATE = auto()
    USER_AUTOTUNE = auto()

# 定义自动调整提示枚举
class AutotuneHint(Enum):
    ELEMENTS_PER_WARP_32 = 0

    # Triton codegen 尝试生成 AutotuneHints 集合。
    # Enum.__repr__ 返回类似 "<AutotuneHint.ELEMENTS_PER_WARP_32: 0>" 的字符串，
    # 这不是有效的 Python 语法。
    # Enum.__str__ 将返回 "AutotuneHint.ELEMENTS_PER_WARP_32"。
    __repr__ = Enum.__str__

# 定义设备属性命名元组，复制设备属性到一个不需要导入 torch 的数据结构中
class DeviceProperties(typing.NamedTuple):
    type: str  # 设备类型
    index: int  # 索引
    cc: int  # 计算能力
    major: Optional[int] = None  # 主版本（可选）
    regs_per_multiprocessor: Optional[int] = None  # 每个多处理器寄存器数（可选）
    # 可选参数，表示每个多处理器的最大线程数
    max_threads_per_multi_processor: Optional[int] = None
    # 可选参数，表示多处理器的数量
    multi_processor_count: Optional[int] = None

    @classmethod
    # 类方法，用于创建一个新的实例
    def create(cls, device):
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        # 根据设备类型获取设备类型，如果是 HIP 则设备类型为 "hip"
        device_type = device.type if torch.version.hip is None else "hip"
        # 获取适用于设备的接口
        device_interface = get_interface_for_device(device)
        
        # 如果设备类型是 "cuda"
        if device_type == "cuda":
            # 获取设备的性能属性
            props = device_interface.get_device_properties(device)
            # 返回一个新的类实例，包括设备类型、索引、计算能力、主要特性以及设备属性
            return cls(
                type=device_type,
                index=device.index,
                cc=device_interface.get_compute_capability(device),
                major=props.major,
                regs_per_multiprocessor=props.regs_per_multiprocessor,
                max_threads_per_multi_processor=props.max_threads_per_multi_processor,
                multi_processor_count=props.multi_processor_count,
            )
        
        # 如果不是 "cuda" 设备类型，则返回一个新的类实例，包括设备类型、索引和计算能力
        return cls(
            type=device_type,
            index=device.index,
            cc=device_interface.get_compute_capability(device),
        )
class HalideInputSpec(typing.NamedTuple):
    ctype: str  # 定义变量，表示数据类型，如 float*、int* 等
    name: str  # 定义变量，表示变量名
    shape: Optional[List[str]] = None  # 可选变量，表示数据形状的列表
    stride: Optional[List[str]] = None  # 可选变量，表示数据步长的列表
    offset: Optional[str] = None  # 可选变量，表示数据偏移量的字符串
    alias_of: Optional[str] = None  # 可选变量，表示别名的字符串

    def bindings_type(self):
        if self.ctype in ("half*", "bfloat16*"):
            return "uint16_t*"  # 如果数据类型为 "half*" 或 "bfloat16*"，返回 "uint16_t*" 类型
        return self.ctype  # 否则返回定义的数据类型本身

    def halide_type(self):
        if self.ctype == "half*":
            return "halide_type_t(halide_type_float, 16)"  # 如果数据类型为 "half*"，返回对应的 Halide 数据类型
        if self.ctype == "bfloat16*":
            return "halide_type_t(halide_type_bfloat, 16)"  # 如果数据类型为 "bfloat16*"，返回对应的 Halide 数据类型
        return f"halide_type_of<{self.ctype.replace('*', '')}>()"  # 否则返回对应的 Halide 数据类型字符串表示

    def is_scalar(self):
        return self.shape is None  # 判断是否为标量，即数据形状是否为 None

    def is_buffer(self):
        return self.shape is not None  # 判断是否为缓冲区，即数据形状是否不为 None


class HalideMeta(typing.NamedTuple):
    argtypes: List[HalideInputSpec]  # 定义变量，表示 HalideInputSpec 对象的列表
    target: str  # 定义变量，表示目标字符串
    scheduler: Optional[str] = None  # 可选变量，表示调度器字符串
    scheduler_flags: Optional[Dict[str, Union[int, str]]] = None  # 可选变量，表示调度器标志的字典
    cuda_device: Optional[int] = None  # 可选变量，表示 CUDA 设备号

    def args(self):
        """Command line args to pass to halide generator"""
        args = [f"target={self.target}"]  # 初始化参数列表，包含目标字符串
        if self.scheduler:
            args.append(f"autoscheduler={self.scheduler}")  # 如果存在调度器，则添加到参数列表中
        if self.scheduler_flags:
            assert self.scheduler  # 断言调度器存在
            for k, v in self.scheduler_flags.items():
                args.append(f"autoscheduler.{k}={v}")  # 遍历调度器标志的字典，将每对键值添加到参数列表中
        return args  # 返回生成的参数列表

    def is_cuda(self):
        return self.cuda_device is not None  # 判断是否为 CUDA 设备模式
```