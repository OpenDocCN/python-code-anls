# `.\pytorch\torch\_inductor\fx_passes\efficient_conv_bn_eval.py`

```py
# mypy: allow-untyped-defs
# 导入PyTorch库中需要使用的模块
import torch
import torch.nn as nn

# 导入Dynamo工具包中的计数器
from torch._dynamo.utils import counters
# 导入Inductor配置模块
from torch._inductor import config as inductor_config
# 导入功能调用模块
from torch.func import functional_call

# 导入模式匹配器中的相关类和函数
from ..pattern_matcher import (
    CallFunctionVarArgs,
    CallModuleVarArgs,
    Match,
    register_graph_pattern,
)

# 导入预处理梯度模块中的评估通道函数
from .pre_grad import efficient_conv_bn_eval_pass


def efficient_conv_bn_eval(
    bn: nn.modules.batchnorm._BatchNorm, conv: nn.modules.conv._ConvNd, x: torch.Tensor
):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Efficient ConvBN Blocks for Transfer Learning and Beyond"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for **training** as well, but only if one sets `bn.training=False`. It
     reduces memory footprint and computation cost, at the cost of slightly
     reduced numerical stability.
    Args:
        bn (nn.modules.batchnorm._BatchNorm): a BatchNorm module.
        conv (nn.modules.conv._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """

    assert bn.running_var is not None

    # 根据情况处理卷积层和批归一化层的参数，确保不缺少任何必需的参数
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    # 对于Conv2d，目标形状为[C_out, 1, 1, 1]，根据卷积类型调整目标形状
    target_shape = [-1] + [1] * (conv.weight.ndim - 1)
    if isinstance(conv, nn.modules.conv._ConvTransposeNd):
        # 对于转置卷积，C_out应该在索引1位置
        target_shape[:2] = [target_shape[1], target_shape[0]]
    # 计算权重系数
    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape(target_shape)
    # 计算动态调整的系数
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # 调整卷积层的权重
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # 调整卷积层的偏置
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (
        bias_on_the_fly - bn.running_mean
    )

    # 进行功能调用，传入调整后的参数进行卷积操作
    input = x
    params = {"weight": weight_on_the_fly, "bias": bias_on_the_fly}
    output = functional_call(conv, params, input)
    return output


def efficient_conv_bn_eval_decomposed(
    bn_weight,
    bn_bias,
    bn_running_mean,
    bn_running_var,
    bn_eps,
    conv: torch._ops.OpOverload,
    conv_weight,
    conv_bias,
    x,
    conv_remainging_args,
):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    """
    """
    "Efficient ConvBN Blocks for Transfer Learning and Beyond"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for **training** as well, but only if one sets `bn.training=False`. It
    reduces memory footprint and computation cost, at the cost of slightly
    reduced numerical stability.
    Args:
    bn_running_var: Running variance of batch normalization.
    conv_weight: Weight tensor of the convolutional layer.
    conv_bias: Bias tensor of the convolutional layer.
    bn_weight: Weight tensor of batch normalization.
    bn_bias: Bias tensor of batch normalization.
    bn_eps: Epsilon value for numerical stability in batch normalization.
    conv: Convolutional layer function (e.g., torch.nn.functional.conv2d).
    x: Input tensor to the convolutional layer.
    conv_remaining_args: Additional arguments passed to the convolutional layer function.
    """
    # 断言保证 bn_running_var 不为空
    assert bn_running_var is not None

    # 处理各种情况：当卷积层没有偏置时，使用零初始化偏置
    weight_on_the_fly = conv_weight
    if conv_bias is not None:
        bias_on_the_fly = conv_bias
    else:
        bias_on_the_fly = torch.zeros_like(bn_running_var)

    # 如果存在 bn_weight，则使用其值；否则，使用与 bn_running_var 相同形状的全一张量
    if bn_weight is not None:
        bn_weight = bn_weight
    else:
        bn_weight = torch.ones_like(bn_running_var)

    # 如果存在 bn_bias，则使用其值；否则，使用与 bn_running_var 相同形状的全零张量
    if bn_bias is not None:
        bn_bias = bn_bias
    else:
        bn_bias = torch.zeros_like(bn_running_var)

    # 对权重的形状进行处理，确保与卷积层的输出通道数匹配
    target_shape = [-1] + [1] * (conv_weight.ndim - 1)
    if "conv_transpose" in conv.__str__():
        # 对于转置卷积，输出通道数应该在索引 1 处
        target_shape[:2] = [target_shape[1], target_shape[0]]
    # 计算权重系数，用于归一化卷积层的权重
    weight_coeff = torch.rsqrt(bn_running_var + bn_eps).reshape(target_shape)

    # 计算最终应用于权重和偏置的系数
    coeff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # 应用系数到权重和偏置上
    weight_on_the_fly = weight_on_the_fly * coeff_on_the_fly
    bias_on_the_fly = bn_bias + coeff_on_the_fly.flatten() * (
        bias_on_the_fly - bn_running_mean
    )

    # 将输入变量重命名为 input
    input = x

    # 返回卷积层函数的调用结果，参数包括处理后的输入、权重和偏置，以及可能的额外参数
    return conv(*((input, weight_on_the_fly, bias_on_the_fly) + conv_remaining_args))
@register_graph_pattern(
    CallFunctionVarArgs(
        [
            torch.nn.functional.batch_norm,
        ]
    ),
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: not inductor_config.freezing
    and inductor_config.efficient_conv_bn_eval_fx_passes,
)
def efficient_conv_bn_eval_graph_transform_inlined(match: Match, *args, **kwargs):
    bn_node = match.nodes[0]  # 获取匹配中的第一个节点，即 BatchNorm 节点
    graph = match.graph  # 获取匹配中的图对象
    assert len(bn_node.args) == 8  # 断言 BatchNorm 节点的参数长度为 8

    # We can only use efficient conv-bn for eval mode with track_running_stats
    # bn_node.args is `training`
    # 只有在评估模式且 track_running_stats 为 True 时才能使用高效的 conv-bn
    if bn_node.args[-3]:  # 如果 BatchNorm 节点的倒数第三个参数为 True
        return  # 返回，不进行优化

    # Check if the input is Conv
    # 检查输入是否为 Conv 层
    input_node = bn_node.args[0]  # 获取 BatchNorm 节点的第一个参数作为输入节点

    if input_node.op != "call_function":  # 如果输入节点不是函数调用
        return  # 返回，不进行优化

    input_fn = input_node.target  # 获取输入节点的目标函数对象

    supported_convs = [
        torch._C._nn.linear,
        torch.conv1d,
        torch.conv2d,
        torch.conv3d,
        torch.conv_transpose1d,
        torch.conv_transpose2d,
        torch.conv_transpose3d,
    ]

    # 检查输入函数是否为支持的卷积操作之一
    if not any(input_fn is cls for cls in supported_convs):
        return  # 返回，不进行优化

    conv_node = input_node  # 将输入节点视为卷积节点

    # Output of conv is used by other nodes, cannot optimize
    # 如果卷积节点的输出被其他节点使用，则无法进行优化
    if len(conv_node.users) > 1:  # 如果卷积节点的使用者数大于1
        return  # 返回，不进行优化

    counters["inductor"]["efficient_conv_bn_eval"] += 1  # 计数器记录高效 conv-bn 优化次数加一

    with graph.inserting_before(bn_node):
        # prepare args for the fused function
        # 准备融合函数的参数
        bn_running_mean = bn_node.args[1]  # BatchNorm 节点的 running_mean 参数
        bn_running_var = bn_node.args[2]  # BatchNorm 节点的 running_var 参数
        bn_weight = bn_node.args[3]  # BatchNorm 节点的 weight 参数
        bn_bias = bn_node.args[4]  # BatchNorm 节点的 bias 参数
        bn_eps = bn_node.args[7]  # BatchNorm 节点的 eps 参数
        assert len(conv_node.args) >= 2  # 断言卷积节点的参数长度至少为2
        conv_input = conv_node.args[0]  # 卷积节点的输入参数
        conv_weight = conv_node.args[1]  # 卷积节点的 weight 参数
        conv_bias = conv_node.args[2] if len(conv_node.args) >= 3 else None  # 卷积节点的 bias 参数（如果存在的话）
        conv_remainging_args = conv_node.args[3:]  # 卷积节点的其余参数
        args = (
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            bn_eps,
            conv_node.target,  # 卷积节点的目标函数
            conv_weight,
            conv_bias,
            conv_input,
            conv_remainging_args,
        )

        # create a new node
        # 创建一个新节点，调用融合后的函数
        new_node = graph.create_node(
            op="call_function",
            target=efficient_conv_bn_eval_decomposed,
            args=args,
            name="efficient_conv_bn_eval",
        )

    # this node replaces the original conv + bn, and therefore
    # should replace the uses of bn_node
    # 将新节点替换原始的 conv + bn，因此应替换 bn_node 的使用情况
    bn_node.replace_all_uses_with(new_node)
    # take care of the deletion order:
    # delete bn_node first, and then conv_node
    # 处理删除顺序：
    # 先删除 bn_node，然后是 conv_node
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)

    return
    # 使用 efficient_conv_bn_eval_pass 作为 pass_dict 的值
    pass_dict=efficient_conv_bn_eval_pass,
    # 定义额外的检查函数 extra_check，确保条件不冻结并且使用了 efficient_conv_bn_eval_fx_passes
    extra_check=lambda match: not inductor_config.freezing
    and inductor_config.efficient_conv_bn_eval_fx_passes,
)
def efficient_conv_bn_eval_graph_transform_decomposed(match: Match, *args, **kwargs):
    bn_node = match.nodes[0]  # 从匹配对象中获取批归一化节点
    graph = match.graph  # 获取匹配所在的图对象
    assert len(bn_node.args) == 9  # 断言批归一化节点的参数个数为9个

    # We can only use efficient conv-bn for eval mode with track_running_stats
    # bn_node.args is `training`
    if bn_node.args[-4]:  # 如果批归一化节点的倒数第4个参数为True，则返回，表示不能优化
        return

    # Check if the input is Conv
    input_node = bn_node.args[0]  # 获取批归一化节点的第一个参数作为输入节点

    if input_node.op != "call_function":  # 如果输入节点不是函数调用类型，则返回，无法优化
        return

    input_fn = input_node.target  # 获取输入节点的目标函数

    supported_convs = [  # 支持的卷积操作列表
        torch.ops.aten.linear.default,
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv3d.default,
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.conv_transpose3d.input,
    ]

    if not any(input_fn is cls for cls in supported_convs):  # 如果输入函数不在支持的卷积操作列表中，则返回
        return

    conv_node = input_node  # 将输入节点作为卷积节点

    # Output of conv is used by other nodes, cannot optimize
    if len(conv_node.users) > 1:  # 如果卷积节点的使用者大于1个，则不能优化，返回
        return

    counters["inductor"]["efficient_conv_bn_eval"] += 1  # 统计计数器中的优化次数加1

    with graph.inserting_before(bn_node):  # 在批归一化节点之前插入新节点
        # prepare args for the fused function
        bn_weight = bn_node.args[1]  # 批归一化权重参数
        bn_bias = bn_node.args[2]  # 批归一化偏置参数
        bn_running_mean = bn_node.args[3]  # 批归一化运行均值参数
        bn_running_var = bn_node.args[4]  # 批归一化运行方差参数
        bn_eps = bn_node.args[7]  # 批归一化epsilon参数
        assert len(conv_node.args) >= 2  # 断言卷积节点的参数至少有2个
        conv_input = conv_node.args[0]  # 卷积的输入参数
        conv_weight = conv_node.args[1]  # 卷积的权重参数
        conv_bias = conv_node.args[2] if len(conv_node.args) >= 3 else None  # 卷积的偏置参数，如果存在的话
        conv_remainging_args = conv_node.args[3:]  # 卷积的剩余参数
        args = (
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            bn_eps,
            conv_node.target,  # 卷积节点的目标函数
            conv_weight,
            conv_bias,
            conv_input,
            conv_remainging_args,
        )

        # create a new node
        new_node = graph.create_node(
            op="call_function",
            target=efficient_conv_bn_eval_decomposed,
            args=args,
            name="efficient_conv_bn_eval",
        )  # 创建新的节点，调用优化后的函数

    # this node replaces the original conv + bn, and therefore
    # should replace the uses of bn_node
    bn_node.replace_all_uses_with(new_node)  # 替换所有使用批归一化节点的地方为新节点
    # take care of the deletion order:
    # delete bn_node first, and then conv_node
    graph.erase_node(bn_node)  # 删除批归一化节点
    graph.erase_node(conv_node)  # 删除卷积节点

    return
    # 定义一个匿名函数 extra_check，接受一个参数 match
    # 检查 inductor_config.freezing 为假且 inductor_config.efficient_conv_bn_eval_fx_passes 为真
    extra_check=lambda match: not inductor_config.freezing
    and inductor_config.efficient_conv_bn_eval_fx_passes,
)
def efficient_conv_bn_eval_graph_transform(match: Match, *args, **kwargs):
    # 匹配到了一个 Batch Normalization (BN) 节点
    bn_node = match.nodes[0]
    # 获取匹配对象所属的计算图
    graph = match.graph
    # 获取计算图所属的模块
    gm = graph.owning_module
    # 从模块中获取 Batch Normalization 模块
    bn_mod = getattr(gm, bn_node.target)  # type: ignore[arg-type]

    # 只能在评估模式且 track_running_stats 启用时使用高效的 conv-bn 优化
    if not bn_mod.track_running_stats or bn_mod.training:
        return

    # 检查输入是否为 Conv 层
    if bn_node.args:
        input_node = bn_node.args[0]
    else:
        input_node = bn_node.kwargs["input"]
    if input_node.op != "call_module":  # type: ignore[union-attr]
        return
    if not hasattr(gm, input_node.target):  # type: ignore[arg-type, union-attr]
        return
    # 获取输入节点对应的模块
    input_mod = getattr(gm, input_node.target)  # type: ignore[arg-type, union-attr]
    # 支持的 Conv 层类型
    supported_convs = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]
    # 如果输入模块不是支持的 Conv 类型之一，则返回
    if not any(isinstance(input_mod, cls) for cls in supported_convs):
        return
    # 将输入节点标记为 Conv 节点
    conv_node = input_node
    # Conv 输出被其他节点使用，无法优化
    if len(conv_node.users) > 1:  # type: ignore[union-attr]
        return

    # 寻找需要优化的 Conv 和 BN 计算节点对
    counters["inductor"]["efficient_conv_bn_eval"] += 1

    with graph.inserting_before(conv_node):
        # 创建 `get_attr` 节点以访问模块
        # 注意，我们直接调用 `create_node` 填充 `name` 参数，
        # `graph.get_attr` 和 `graph.call_function` 不允许 `name` 参数。
        conv_get_node = graph.create_node(
            op="get_attr", target=conv_node.target, name="get_conv"  # type: ignore[union-attr]
        )
        bn_get_node = graph.create_node(
            op="get_attr", target=bn_node.target, name="get_bn"
        )
        if conv_node.args:  # type: ignore[union-attr]
            conv_input = conv_node.args[0]  # type: ignore[union-attr]
        else:
            conv_input = conv_node.kwargs["input"]  # type: ignore[union-attr]
        # 准备融合函数的参数
        args = (bn_get_node, conv_get_node, conv_input)
        # 创建新节点
        new_node = graph.create_node(
            op="call_function",
            target=efficient_conv_bn_eval,
            args=args,
            name="efficient_conv_bn_eval",
        )
    # 新节点替换原始的 Conv + BN，因此应替换 bn_node 的所有使用
    bn_node.replace_all_uses_with(new_node)
    # 处理删除顺序：
    # 首先删除 bn_node，然后删除 conv_node
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)
```