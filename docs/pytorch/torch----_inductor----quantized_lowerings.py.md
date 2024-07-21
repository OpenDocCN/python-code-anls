# `.\pytorch\torch\_inductor\quantized_lowerings.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 lowering 模块
from . import lowering

# 定义 quantized 变量，引用 torch 库中的 quantized 操作
quantized = torch.ops.quantized
# 定义 _quantized 变量，引用 torch 库中的 _quantized 操作
_quantized = torch.ops._quantized
# 定义 aten 变量，引用 torch 库中的 aten 操作
aten = torch.ops.aten

# 注册量化操作函数
def register_quantized_ops():
    # 将需要实现的输入添加到 lowering 模块中
    lowering.add_needs_realized_inputs(
        [
            quantized.max_pool2d,
            _quantized.wrapped_fbgemm_pack_gemm_matrix_fp16,
            _quantized.wrapped_fbgemm_linear_fp16_weight,
        ]
    )

    # 为 quantized.max_pool2d 函数创建回退机制
    lowering.make_fallback(quantized.max_pool2d)
    # 为 _quantized.wrapped_fbgemm_pack_gemm_matrix_fp16 函数创建回退机制
    lowering.make_fallback(_quantized.wrapped_fbgemm_pack_gemm_matrix_fp16)
    # 为 _quantized.wrapped_fbgemm_linear_fp16_weight 函数创建回退机制
    lowering.make_fallback(_quantized.wrapped_fbgemm_linear_fp16_weight)


# 注册无权重量化矩阵乘法操作函数
def register_woq_mm_ops():
    # 将需要实现的输入添加到 lowering 模块中
    lowering.add_needs_realized_inputs(
        [
            aten._weight_int8pack_mm,
        ]
    )

    # 为 aten._weight_int8pack_mm 函数创建回退机制
    lowering.make_fallback(aten._weight_int8pack_mm)
```