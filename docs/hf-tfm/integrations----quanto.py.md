# `.\integrations\quanto.py`

```py
# 导入 is_torch_available 函数，检查是否可以导入 torch
from ..utils import is_torch_available

# 如果 torch 可用，则导入 torch 库
if is_torch_available():
    import torch

# 定义函数 replace_with_quanto_layers，用于递归替换给定模型的线性层为 Quanto 量化层，并返回转换后的模型及是否成功的布尔值
def replace_with_quanto_layers(
    model,  # 输入参数：待转换的模型，必须是 torch.nn.Module 的实例
    quantization_config=None,  # 输入参数：量化配置对象，包含量化参数，默认为 None
    modules_to_not_convert=None,  # 输入参数：不转换的模块列表，默认为 None
    current_key_name=None,  # 输入参数：当前键名列表，用于递归，用户不应传递此参数，默认为 None
    has_been_replaced=False,  # 输入参数：指示转换是否成功的布尔值，用于递归，用户不应传递此参数，默认为 False
):
    """
    Public method that recursively replaces the Linear layers of the given model with Quanto quantized layers.
    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AqlmConfig`, defaults to `None`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list`, *optional*, defaults to `None`):
            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.
        current_key_name (`list`, *optional*, defaults to `None`):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*, defaults to `None`):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """

    # 从 accelerate 库中导入 init_empty_weights 函数
    from accelerate import init_empty_weights
    # 从 quanto 库中导入 QLayerNorm, QLinear, qfloat8, qint2, qint4, qint8 等类和函数
    from quanto import QLayerNorm, QLinear, qfloat8, qint2, qint4, qint8

    # 定义权重映射表和激活映射表
    w_mapping = {"float8": qfloat8, "int8": qint8, "int4": qint4, "int2": qint2}
    a_mapping = {None: None, "float8": qfloat8, "int8": qint8}

    # 如果 modules_to_not_convert 为 None，则设为空列表
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    # 遍历模型的每个子模块，获取子模块的名称和对象
    for name, module in model.named_children():
        # 如果当前键名为 None，则初始化为空列表
        if current_key_name is None:
            current_key_name = []
        # 将当前子模块的名称添加到当前键名列表中
        current_key_name.append(name)

        # 检查当前模块的全限定名是否包含在不转换的模块列表中的任何项
        if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
            # 使用空权重初始化上下文管理器
            with init_empty_weights():
                # 如果模块是线性层
                if isinstance(module, torch.nn.Linear):
                    # 替换为量化后的线性层 QLinear
                    model._modules[name] = QLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        dtype=module.weight.dtype,
                        weights=w_mapping[quantization_config.weights],
                        activations=a_mapping[quantization_config.activations],
                    )
                    # 设置新模块不需要梯度计算
                    model._modules[name].requires_grad_(False)
                    # 标记替换已完成
                    has_been_replaced = True
                # 如果模块是 LayerNorm 层
                elif isinstance(module, torch.nn.LayerNorm):
                    # 如果存在激活量化配置，则替换为量化后的 LayerNorm 层 QLayerNorm
                    if quantization_config.activations is not None:
                        model._modules[name] = QLayerNorm(
                            module.normalized_shape,
                            module.eps,
                            module.elementwise_affine,
                            module.bias is not None,
                            activations=a_mapping[quantization_config.activations],
                        )
                        # 标记替换已完成
                        has_been_replaced = True

        # 如果当前模块有子模块，则递归替换子模块中的量化层
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_quanto_layers(
                module,
                quantization_config=quantization_config,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        
        # 递归完成后，移除当前键名列表中的最后一个键，准备处理下一个子模块
        current_key_name.pop(-1)

    # 返回替换后的模型和替换操作是否发生过的标志
    return model, has_been_replaced
```