# `.\pytorch\torch\distributed\_tensor\ops\experimental_ops.py`

```
# 实现矩阵相关的操作用于分布式张量
from typing import List

import torch
from torch.distributed._tensor._op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta

# 使用 torch.ops.aten 别名为 aten，提供对底层 ATen 操作的访问
aten = torch.ops.aten

# 尝试导入 numpy 库，如果不存在则将 np 设置为 None
try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

# 注册 aten.slice_backward.default 操作的属性规则
@register_prop_rule(aten.slice_backward.default)
def slice_backward_rules(op_schema: OpSchema) -> OutputSharding:
    # 解构 OpSchema 对象的参数模式
    grad_output_spec, input_sizes, dim, start, end, step = op_schema.args_schema
    
    # 断言参数类型
    assert isinstance(grad_output_spec, DTensorSpec)
    assert isinstance(input_sizes, List)
    
    # 确保 grad_output_spec.tensor_meta 不为 None
    assert grad_output_spec.tensor_meta is not None
    
    # 计算梯度输入的步长
    grad_input_stride = list(np.cumprod(input_sizes[::-1])[:-1][::-1])
    grad_input_stride.append(1)
    
    # 获取 grad_output_spec 的维度映射和求和信息
    dim_map = grad_output_spec.dim_map
    sums = grad_output_spec.sums
    
    # 构造梯度输入的 TensorMeta 对象
    grad_input_tensor_meta = TensorMeta(
        torch.Size(input_sizes),
        tuple(grad_input_stride),
        grad_output_spec.tensor_meta.dtype,
    )
    
    # 使用 dim_map、sums 和构造的 grad_input_tensor_meta 构建 DTensorSpec 对象
    grad_input_spec = DTensorSpec.from_dim_map(
        grad_output_spec.mesh,
        dim_map,
        sums,
        tensor_meta=grad_input_tensor_meta,
    )
    
    # 返回 OutputSharding 对象，包含 grad_input_spec
    return OutputSharding(grad_input_spec)

# 注册 aten.bernoulli.default 和 aten.bernoulli_.float 操作的属性规则
@register_prop_rule(aten.bernoulli.default)
@register_prop_rule(aten.bernoulli_.float)
def bernoulli_rules(op_schema: OpSchema) -> OutputSharding:
    # 获取 OpSchema 对象的第一个参数，即输入的 DTensorSpec
    input_spec = op_schema.args_schema[0]
    
    # 断言参数类型
    assert isinstance(input_spec, DTensorSpec)
    
    # 返回 OutputSharding 对象，包含 input_spec
    return OutputSharding(input_spec)
```