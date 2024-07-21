# `.\pytorch\torch\_inductor\codegen\triton_utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型和模块
from typing import Any, Dict, List, Optional

# 导入 sympy 库
import sympy

# 导入 PyTorch 库
import torch

# 导入上级目录中的 config 模块
from .. import config

# 导入 instance_descriptor 类型提示
from ..runtime.hints import instance_descriptor

# 导入 _type_of 工具函数
from ..utils import _type_of

# 导入 V 对象
from ..virtualized import V

# 导入本地 common 模块中的类型提示
from .common import KernelArgType, SizeArg, TensorArg, WorkspaceArg


def signature_of(arg: KernelArgType, *, size_dtype: str) -> str:
    # 如果参数是 TensorArg 类型
    if isinstance(arg, TensorArg):
        # TODO: 当 Triton 支持 PyTorch fp8 数据类型时，移除 fp8 特殊处理
        if arg.dtype == torch.float8_e4m3fn:
            tye = "*fp8e4nv"
        elif arg.dtype == torch.float8_e5m2:
            tye = "*fp8e5"
        elif arg.dtype == torch.float8_e4m3fnuz:
            tye = "*fp8e4b8"
        elif arg.dtype == torch.float8_e5m2fnuz:
            tye = "*fp8e5b16"
        else:
            # 根据数据类型获取对应的字符串表示
            tye = _type_of(arg.dtype)
        
        # 如果是未指定参数，则将 0d 张量视为标量处理
        if V.graph.is_unspec_arg(arg.buffer):
            # 根据类型移除开头的 '*'
            new_tye = tye.lstrip("*")
            if new_tye in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_tye
        else:
            return tye
    
    # 如果参数是 SizeArg 类型
    if isinstance(arg, SizeArg):
        if arg.expr is None:
            # 来自 triton/runtime/jit.py，None 表示 nullptr，隐式转换为 *i8
            return "*i8"
        elif isinstance(arg.expr, (float, sympy.Float)):
            return "fp32"
        
        # 根据 size_dtype 返回对应的数据类型字符串
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    
    # 如果参数是 WorkspaceArg 类型，则返回 '*i8'
    if isinstance(arg, WorkspaceArg):
        return "*i8"
    
    # 抛出未处理的参数类型异常
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def signature_to_meta(
    signature: List[KernelArgType],
    *,
    size_dtype: str,
    indices: Optional[List[int]] = None,
) -> Dict[int, str]:
    # 如果 indices 未提供，则使用所有参数的索引
    if indices is None:
        indices = list(range(len(signature)))
    
    # 返回参数索引到对应类型字符串的字典
    return {
        i: signature_of(arg, size_dtype=size_dtype)
        for i, arg in zip(indices, signature)
    }


def is_unaligned_buffer(arg: TensorArg):
    # 获取缓冲区名称
    buf_name = arg.buffer
    
    # 如果缓冲区在图输入中
    if buf_name in V.graph.graph_inputs:
        # 查看输入对齐处理
        return buf_name not in V.graph.aligned_inputs
    
    # 如果缓冲区在常量中
    if buf_name in V.graph.constants:
        # 所有常量假定为已对齐
        return False
    
    # 如果有调度器，则获取缓冲区布局
    if V.graph.scheduler:
        layout = V.graph.scheduler.get_buffer_layout(buf_name)
    else:
        # 否则获取缓冲区布局
        buffer = V.graph.get_buffer(buf_name)
        # 输出参数
        if not buffer:
            assert buf_name == V.kernel.output_node.name
            layout = V.kernel.output_node.layout
        else:
            layout = buffer.get_layout()
    
    # 如果布局为 NonOwningLayout，则检查是否需要对齐
    if isinstance(layout, torch._inductor.ir.NonOwningLayout):
        return not layout.maybe_guard_aligned()
    else:
        return False


def config_of(
    # 声明一个参数 args，类型为 List[KernelArgType]
    args: List[KernelArgType],
    # 使用 '*' 表示后续的参数都是关键字参数
    # 声明一个可选参数 indices，类型为 Optional[List[int]]，默认值为 None
    *, 
    indices: Optional[List[int]] = None,
def compute_packed_indices(
    indices: Optional[List[int]] = None, *args: KernelArgType, config: Config
) -> Any:
    # 如果未提供indices参数，则默认为0到args列表长度的整数列表
    if indices is None:
        indices = list(range(len(args)))

    def is_aligned(x: KernelArgType, alignment: int, include_tensor: bool) -> bool:
        """
        Roughly follow triton code here:
        https://github.com/openai/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        """
        # 如果x是TensorArg类型
        if isinstance(x, TensorArg):
            if include_tensor:
                # 计算偏移是否按alignment对齐，并检查是否为非对齐的缓冲区
                offset_aligned = V.graph.sizevars.statically_known_multiple_of(
                    x.offset * x.dtype.itemsize, alignment  # type: ignore[arg-type]
                )
                return offset_aligned and not is_unaligned_buffer(x)
            else:
                return False
        # 如果x是SizeArg类型
        if isinstance(x, SizeArg):
            # 以下情况不需要对齐检查
            if x.name.startswith("load_seed_offset"):
                return False
            if x.expr is None:
                return False
            if isinstance(x.expr, float):
                return False
            # 检查表达式是否可以静态确定为alignment的倍数
            return V.graph.sizevars.statically_known_multiple_of(x.expr, alignment)  # type: ignore[arg-type]
        # 如果x是WorkspaceArg类型
        if isinstance(x, WorkspaceArg):
            # 检查WorkspaceArg的nbytes是否可以静态确定为alignment的倍数
            return V.graph.sizevars.statically_known_multiple_of(x.nbytes, alignment)  # type: ignore[arg-type]
        # 抛出未实现的类型错误
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    # 根据config.triton.divisible_by_16的设置，选择是否包含对tensor的对齐检查
    if config.triton.divisible_by_16:
        # 构建divisible_by_16元组，包含所有对齐到16的indices及其对应的args
        divisible_by_16 = tuple(
            i
            for i, arg in zip(indices, args)
            if is_aligned(arg, alignment=16, include_tensor=True)
        )
    else:
        # 如果config.triton.divisible_by_16为False，则divisible_by_16为空元组
        divisible_by_16 = ()

    # 构建divisible_by_8元组，包含所有对齐到8的indices及其对应的args，不包含tensor类型的检查
    divisible_by_8 = tuple(
        i
        for i, arg in zip(indices, args)
        if is_aligned(arg, alignment=8, include_tensor=False)
    )

    # 构建equal_to_1元组，包含所有是SizeArg且表达式为整数1的indices及其对应的args
    equal_to_1 = tuple(
        i
        for i, arg in zip(indices, args)
        if isinstance(arg, SizeArg)
        and isinstance(arg.expr, (int, sympy.Integer))
        and V.graph.sizevars.statically_known_equals(arg.expr, 1)  # type: ignore[arg-type]
    )

    # ids_of_folded_args从equal_to_1和空args中设置
    ids_of_folded_args = tuple(equal_to_1)

    # 调用instance_descriptor函数，返回计算得到的结果元组
    return instance_descriptor(
        divisible_by_16, equal_to_1, ids_of_folded_args, divisible_by_8
    )
```