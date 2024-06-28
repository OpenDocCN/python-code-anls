# `.\integrations\aqlm.py`

```
# 版权声明和许可信息
# 版权所有 2024 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 无论是明示的还是暗示的条件，包括但不限于适销性和特定用途的适用性。
# 有关详细信息，请参阅许可证。
"AQLM (Additive Quantization of Language Model) integration file"

# 导入所需的模块和函数
from ..utils import is_accelerate_available, is_aqlm_available, is_torch_available

# 如果 torch 可用，则导入 torch.nn 模块
if is_torch_available():
    import torch.nn as nn

# 替换模型中的线性层为 AQLM 量化层的公共方法
def replace_with_aqlm_linear(
    model,
    quantization_config=None,
    linear_weights_not_to_quantize=None,
    current_key_name=None,
    has_been_replaced=False,
):
    """
    Public method that recursively replaces the Linear layers of the given model with AQLM quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successfull or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AqlmConfig`):
            The quantization config object that contains the quantization parameters.
        linear_weights_not_to_quantize (`list[str]`, *optional*):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """

    # 检查是否安装了 AQLM
    if not is_aqlm_available():
        raise ValueError("AQLM is not available. Please install it with `pip install aqlm[cpu,gpu]`")

    # 检查是否安装了 Accelerate
    if not is_accelerate_available():
        raise ValueError("AQLM requires Accelerate to be installed: `pip install accelerate`")

    # 如果未提供 linear_weights_not_to_quantize 参数，则初始化为空列表
    if linear_weights_not_to_quantize is None:
        linear_weights_not_to_quantize = []

    # 导入所需的函数和类
    from accelerate import init_empty_weights
    from aqlm import QuantizedLinear
    # 遍历模型的每个子模块的名称和模块本身
    for name, module in model.named_children():
        # 如果当前键名为 None，则初始化为空列表
        if current_key_name is None:
            current_key_name = []
        # 将当前模块名称添加到当前键名列表中
        current_key_name.append(name)

        # 如果当前模块是线性层（nn.Linear）
        if isinstance(module, nn.Linear):
            # 构造当前模块权重的完整路径，以便检查是否不需要量化
            if ".".join(current_key_name) + ".weight" not in linear_weights_not_to_quantize:
                # 使用 init_empty_weights 上下文管理器初始化空权重
                with init_empty_weights():
                    # 获取输入和输出特征数
                    in_features = module.in_features
                    out_features = module.out_features

                    # 替换当前模块为量化后的 QuantizedLinear 模块
                    model._modules[name] = QuantizedLinear(
                        in_features,
                        out_features,
                        bias=module.bias is not None,
                        in_group_size=quantization_config.in_group_size,
                        out_group_size=quantization_config.out_group_size,
                        num_codebooks=quantization_config.num_codebooks,
                        nbits_per_codebook=quantization_config.nbits_per_codebook,
                    )
                    # 标记模块已被替换
                    has_been_replaced = True

                    # 存储原始模块类以备稍后可能需要对权重进行转置
                    model._modules[name].source_cls = type(module)
                    # 将 requires_grad 设置为 False，避免意外错误
                    model._modules[name].requires_grad_(False)

        # 如果当前模块有子模块
        if len(list(module.children())) > 0:
            # 递归调用 replace_with_aqlm_linear 函数替换子模块中的线性层
            _, has_been_replaced = replace_with_aqlm_linear(
                module,
                quantization_config=quantization_config,
                linear_weights_not_to_quantize=linear_weights_not_to_quantize,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        
        # 递归结束后，移除当前键名列表中的最后一个键名
        current_key_name.pop(-1)

    # 返回替换后的模型及替换状态标志
    return model, has_been_replaced
```