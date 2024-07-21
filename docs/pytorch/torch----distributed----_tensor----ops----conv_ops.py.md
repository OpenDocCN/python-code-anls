# `.\pytorch\torch\distributed\_tensor\ops\conv_ops.py`

```
# 导入必要的模块和类型声明
from typing import List

import torch
from torch.distributed._tensor._op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta

# 获取 torch.ops.aten 的引用
aten = torch.ops.aten

# 注册卷积操作的属性规则
@register_prop_rule(aten.convolution.default)
def convolution_rules(op_schema: OpSchema) -> OutputSharding:
    # 解构 OpSchema 对象以获取参数
    (
        input_spec,
        weight_spec,
        bias_spec,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ) = op_schema.args_schema

    # 断言参数的类型和属性
    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    assert isinstance(bias_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None
    assert weight_spec.tensor_meta is not None

    # 获取输入张量和权重张量的形状
    in_shape = input_spec.tensor_meta.shape
    weight_shape = weight_spec.tensor_meta.shape

    # 断言和计算卷积的步长、填充、扩展
    assert isinstance(stride, List)
    assert isinstance(padding, List)
    assert isinstance(dilation, List)
    assert isinstance(weight_shape, torch.Size)
    N, C_in, H_in, W_in = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
    C_out = weight_shape[0]
    H_out = (H_in + 2 * padding[0] - dilation[0] * (weight_shape[2] - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (weight_shape[3] - 1) - 1) // stride[1] + 1

    # 计算输出张量的形状和步长
    output_shape = [N, C_out, H_out, W_out]
    output_stride = (C_out * H_out * W_out, H_out * W_out, W_out, 1)
    output_dim_map = input_spec.dim_map
    pending_sums = input_spec.sums

    # 创建输出张量的元数据
    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        output_stride,
        input_spec.tensor_meta.dtype,
    )

    # 返回输出分片对象
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            output_dim_map,
            pending_sums,
            tensor_meta=tensor_meta,
        )
    )


# 注册卷积反向操作的属性规则
@register_prop_rule(aten.convolution_backward.default)
def convolution_backward_rules(op_schema: OpSchema) -> OutputSharding:
    # 解构 OpSchema 对象以获取参数
    input_spec = op_schema.args_schema[0]
    (
        grad_output_spec,
        input_spec,
        weight_spec,
        bias_shape_opt,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    ) = op_schema.args_schema

    # 断言参数的类型和属性
    assert isinstance(grad_output_spec, DTensorSpec)
    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    assert isinstance(bias_shape_opt, List)
    assert input_spec.tensor_meta is not None

    # 获取权重和偏置张量的元数据
    weight_tensor_meta = weight_spec.tensor_meta
    bias_tensor_meta = TensorMeta(
        torch.Size(bias_shape_opt),
        (1,),
        input_spec.tensor_meta.dtype,
    )

    # 设置梯度输入张量的规格
    grad_input_spec = input_spec
    # 根据输入规格创建梯度权重的张量规格对象
    grad_weight_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,  # 使用输入规格的网格信息
        [-1, -1, -1, -1],  # 设置权重张量的维度映射为 [-1, -1, -1, -1]
        [0],  # 设置权重张量的分片为 [0]
        tensor_meta=weight_tensor_meta,  # 使用权重张量的元数据
    )
    
    # 根据输入规格创建梯度偏置的张量规格对象
    grad_bias_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,  # 使用输入规格的网格信息
        [-1],  # 设置偏置张量的维度映射为 [-1]
        [0],  # 设置偏置张量的分片为 [0]
        tensor_meta=bias_tensor_meta,  # 使用偏置张量的元数据
    )
    
    # 返回一个包含梯度输入规格、梯度权重规格和梯度偏置规格的输出分片对象
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])
```