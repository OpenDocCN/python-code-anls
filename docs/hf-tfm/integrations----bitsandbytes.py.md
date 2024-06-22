# `.\transformers\integrations\bitsandbytes.py`

```py
# 导入模块元数据
import importlib.metadata
# 导入警告模块
import warnings
# 导入深拷贝函数
from copy import deepcopy

# 从 packaging 模块中导入 version 类
from packaging import version

# 从当前包的 utils 模块中导入 is_accelerate_available、is_bitsandbytes_available 和 logging 函数
from ..utils import is_accelerate_available, is_bitsandbytes_available, logging

# 如果 bitsandbytes 可用，则导入相应模块
if is_bitsandbytes_available():
    # 导入 bitsandbytes 模块，并将其重命名为 bnb
    import bitsandbytes as bnb
    # 导入 PyTorch 模块
    import torch
    # 导入 PyTorch 神经网络模块
    import torch.nn as nn

    # 从当前包的 pytorch_utils 模块中导入 Conv1D 类
    from ..pytorch_utils import Conv1D

# 如果 accelerate 可用，则导入相应模块
if is_accelerate_available():
    # 从 accelerate 模块中导入 init_empty_weights 函数
    from accelerate import init_empty_weights
    # 从 accelerate.utils 模块中导入 find_tied_parameters 函数

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义函数，将模块的量化张量设置到指定设备上
def set_module_quantized_tensor_to_device(module, tensor_name, device, value=None, quantized_stats=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function). The
    function is adapted from `set_module_tensor_to_device` function from accelerate that is adapted to support the
    class `Int8Params` from `bitsandbytes`.

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        quantized_stats (`dict[str, Any]`, *optional*):
            Dict with items for either 4-bit or 8-bit serialization
    """
    # 如果张量名中包含"."，则递归处理
    if "." in tensor_name:
        # 按"."分割张量名
        splits = tensor_name.split(".")
        # 遍历分割后的部分
        for split in splits[:-1]:
            # 获取模块的子模块
            new_module = getattr(module, split)
            # 如果子模块不存在，则引发异常
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            # 更新模块为当前子模块
            module = new_module
        # 将张量名更新为最后一部分
        tensor_name = splits[-1]

    # 如果张量名不在模块的参数或缓冲区中，则引发异常
    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    # 判断张量是否为缓冲区
    is_buffer = tensor_name in module._buffers
    # 获取旧值
    old_value = getattr(module, tensor_name)

    # 如果旧值在元设备上，并且目标设备不是元设备，并且没有给定值，则引发异常
    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    # 如果是量化加载前的加载，则设置量化标志
    prequantized_loading = quantized_stats is not None
    # 如果是缓冲区或者 bitsandbytes 模块不可用，则标志位都为 False
    if is_buffer or not is_bitsandbytes_available():
        is_8bit = False
        is_4bit = False
    else:
        # 判断是否为 4 位量化张量
        is_4bit = hasattr(bnb.nn, "Params4bit") and isinstance(module._parameters[tensor_name], bnb.nn.Params4bit)
        # 判断是否为 8 位量化张量
        is_8bit = isinstance(module._parameters[tensor_name], bnb.nn.Int8Params)
    # 如果参数是8位或4位
    if is_8bit or is_4bit:
        # 获取参数
        param = module._parameters[tensor_name]
        # 如果参数不在 CUDA 设备上
        if param.device.type != "cuda":
            # 如果值为空
            if value is None:
                # 将旧值转移到指定设备
                new_value = old_value.to(device)
            # 如果值是 torch.Tensor 类型
            elif isinstance(value, torch.Tensor):
                # 将值转移到 CPU 设备
                new_value = value.to("cpu")
            else:
                # 将值转为 torch.Tensor 类型，并指定设备为 CPU
                new_value = torch.tensor(value, device="cpu")

            # 如果模型使用 `Conv1D` 替代 `nn.Linear`（例如 gpt2），在量化之前对权重矩阵进行转置
            # 由于权重以正确的“方向”保存，因此在加载时跳过转置
            if issubclass(module.source_cls, Conv1D) and not prequantized_loading:
                new_value = new_value.T

            # 复制旧值的属性
            kwargs = old_value.__dict__

            # 如果预量化加载与新值的数据类型不兼容
            if prequantized_loading != (new_value.dtype in (torch.int8, torch.uint8)):
                raise ValueError(
                    f"Value dtype `{new_value.dtype}` is not compatible with parameter quantization status."
                )

            # 如果是8位
            if is_8bit:
                # 检查是否支持8位序列化
                is_8bit_serializable = version.parse(importlib.metadata.version("bitsandbytes")) > version.parse(
                    "0.37.2"
                )
                # 如果新值的数据类型是 int8 或 uint8 且不支持8位序列化
                if new_value.dtype in (torch.int8, torch.uint8) and not is_8bit_serializable:
                    raise ValueError(
                        "Detected int8 weights but the version of bitsandbytes is not compatible with int8 serialization. "
                        "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                    )
                # 创建 Int8Params 对象，并指定设备为 CPU
                new_value = bnb.nn.Int8Params(new_value, requires_grad=False, **kwargs).to(device)
                # 如果预量化加载，设置 SCB 属性
                if prequantized_loading:
                    setattr(new_value, "SCB", quantized_stats["SCB"].to(device))
            # 如果是4位
            elif is_4bit:
                # 如果预量化加载
                if prequantized_loading:
                    # 检查是否支持4位序列化
                    is_4bit_serializable = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse(
                        "0.41.3"
                    )
                    # 如果新值的数据类型是 int8 或 uint8 且不支持4位序列化
                    if new_value.dtype in (torch.int8, torch.uint8) and not is_4bit_serializable:
                        raise ValueError(
                            "Detected 4-bit weights but the version of bitsandbytes is not compatible with 4-bit serialization. "
                            "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                        )
                    # 从预量化数据创建 Params4bit 对象
                    new_value = bnb.nn.Params4bit.from_prequantized(
                        data=new_value,
                        quantized_stats=quantized_stats,
                        requires_grad=False,
                        device=device,
                        **kwargs,
                    )
                else:
                    # 创建 Params4bit 对象，并指定设备为 CPU
                    new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(device)
            # 更新模块的参数
            module._parameters[tensor_name] = new_value
    else:
        # 如果数值为 None，则将旧数值转移到指定设备
        if value is None:
            new_value = old_value.to(device)
        # 如果数值是 torch.Tensor 类型，则将其转移到指定设备
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        # 否则，将数值转换为 torch.Tensor 类型，并移动到指定设备
        else:
            new_value = torch.tensor(value, device=device)

        # 如果是缓冲区，则更新模块的缓冲区
        if is_buffer:
            module._buffers[tensor_name] = new_value
        # 否则，将新数值转换为 nn.Parameter 类型，并更新模块的参数
        else:
            new_value = nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value
# 定义一个私有方法，用于模块替换的递归包装
def _replace_with_bnb_linear(
    model,
    modules_to_not_convert=None,  # 不需要转换的模块列表，默认为 None
    current_key_name=None,  # 当前键名，默认为 None
    quantization_config=None,  # 量化配置，默认为 None
    has_been_replaced=False,  # 标志是否已替换，默认为 False
):
    """
    返回已转换的模型以及指示转换是否成功的布尔值。
    """
    # 遍历模型的子模块，同时获取模块的名称和实例
    for name, module in model.named_children():
        # 如果当前键名为 None，则将其初始化为空列表
        if current_key_name is None:
            current_key_name = []
        # 将当前模块的名称添加到键名列表中
        current_key_name.append(name)

        # 如果当前模块为线性层（nn.Linear）或者一维卷积层（Conv1D）且不在不转换的模块列表中
        if (isinstance(module, nn.Linear) or isinstance(module, Conv1D)) and name not in modules_to_not_convert:
            # 检查当前键名不在`modules_to_not_convert`中
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                # 使用空权重初始化上下文管理器
                with init_empty_weights():
                    # 如果当前模块为一维卷积层
                    if isinstance(module, Conv1D):
                        # 获取输入和输出特征的形状
                        in_features, out_features = module.weight.shape
                    else:
                        # 获取输入和输出特征的数量
                        in_features = module.in_features
                        out_features = module.out_features

                    # 如果量化配置的量化方法是 "llm_int8"
                    if quantization_config.quantization_method() == "llm_int8":
                        # 用 bnb.nn.Linear8bitLt 替换模型的当前模块
                        model._modules[name] = bnb.nn.Linear8bitLt(
                            in_features,
                            out_features,
                            module.bias is not None,
                            has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                            threshold=quantization_config.llm_int8_threshold,
                        )
                        # 设置已替换标志为 True
                        has_been_replaced = True
                    else:
                        # 如果跳过的模块不为空且当前模块在跳过模块列表中
                        if (
                            quantization_config.llm_int8_skip_modules is not None
                            and name in quantization_config.llm_int8_skip_modules
                        ):
                            pass
                        else:
                            # 用 bnb.nn.Linear4bit 替换模型的当前模块
                            model._modules[name] = bnb.nn.Linear4bit(
                                in_features,
                                out_features,
                                module.bias is not None,
                                quantization_config.bnb_4bit_compute_dtype,
                                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                                quant_type=quantization_config.bnb_4bit_quant_type,
                            )
                            # 设置已替换标志为 True
                            has_been_replaced = True
                    # 将模块类存储以备后续可能需要转置权重
                    model._modules[name].source_cls = type(module)
                    # 强制 requires_grad 设置为 False，以避免意外错误
                    model._modules[name].requires_grad_(False)
        
        # 如果当前模块有子模块
        if len(list(module.children())) > 0:
            # 递归调用 _replace_with_bnb_linear 函数替换子模块
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        
        # 移除用于递归的键名列表的最后一个键
        current_key_name.pop(-1)
    
    # 返回替换后的模型及替换标志
    return model, has_been_replaced
# 用于替换所有 `torch.nn.Linear` 模块为 `bnb.nn.Linear8bit` 模块的辅助函数，以实现使用混合 int8 精度运行模型
def replace_with_bnb_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    """
    一个辅助函数，用于将所有 `torch.nn.Linear` 模块替换为 `bitsandbytes` 库中的 `bnb.nn.Linear8bit` 模块。
    这将使您的模型能够使用混合 int8 精度，如论文 `LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale` 所述。
    在运行此函数之前，请确保已安装了正确 CUDA 版本的硬件的 `bitsandbytes`。`pip install -i https://test.pypi.org/simple/
    bitsandbytes`

    该函数将递归运行，并替换所有 `torch.nn.Linear` 模块，除了应保留为 `torch.nn.Linear` 模块的 `lm_head`。
    替换是在 `init_empty_weights` 上下文管理器下完成的，因此运行此函数不需要 CPU/GPU 内存。
    Int8 混合精度矩阵分解通过将矩阵乘法分为两个流进行工作：(1) 和系统特征异常值流矩阵在 fp16 中相乘 (0.01%)，
    (2) 一个常规的 int8 矩阵乘法流 (99.9%)。通过这种方法，对于非常大的模型（>=176B 参数），可以进行 int8 推断而不会出现预测降级。

    参数:
        model (`torch.nn.Module`):
            输入模型或 `torch.nn.Module`，因为函数是递归运行的。
        modules_to_not_convert (`List[`str`]`, *可选*, 默认为 `["lm_head"]`):
            不要在 `Linear8bitLt` 中转换的模块的名称。在实践中，我们保留 `lm_head` 以全精度的原因是为了数值稳定性。
        current_key_name (`List[`str`]`, *可选*):
            用于跟踪递归的当前键的数组。这用于检查当前键（部分）是否不在不转换模块列表中（例如转移到 `cpu` 或 `disk` 的模块）。
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    model, has_been_replaced = _replace_with_bnb_linear(
        model, modules_to_not_convert, current_key_name, quantization_config
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


# 为了向后兼容
def replace_8bit_linear(*args, **kwargs):
    warnings.warn(
        "`replace_8bit_linear` 将在将来的版本中被弃用，请改用 `replace_with_bnb_linear`",
        FutureWarning,
    )
    return replace_with_bnb_linear(*args, **kwargs)


# 为了向后兼容
def set_module_8bit_tensor_to_device(*args, **kwargs):
    # 发出警告，提示`set_module_8bit_tensor_to_device`将在将来的版本中被弃用，请使用`set_module_quantized_tensor_to_device`代替
    warnings.warn(
        "`set_module_8bit_tensor_to_device` will be deprecated in a future version, please use `set_module_quantized_tensor_to_device` instead",
        FutureWarning,
    )
    # 返回调用`set_module_quantized_tensor_to_device`函数的结果
    return set_module_quantized_tensor_to_device(*args, **kwargs)
# 获取在转换为 int8 时不需要进行转换的模块的键列表的实用函数
def get_keys_to_not_convert(model):
    """
    获取模块中需要保持完整精度的模块键的实用函数，例如对于 CausalLM 模块，
    我们可能希望出于数值稳定性原因保持 lm_head 的完整精度。
    对于其他架构，我们希望保留模型的绑定权重。该函数将返回不转换为 int8 的模块键的列表。

    参数:
    model (`torch.nn.Module`):
        输入模型
    """
    # 创建模型的副本并绑定权重，然后检查是否包含绑定的权重
    tied_model = deepcopy(model)  # 这在 `init_empty_weights` 上下文管理器内完成，成本为 0
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # 兼容 Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # 如果没有绑定的权重，我们希望保留 lm_head（output_embedding）的完整精度
    if not has_tied_params:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            list_last_module = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            return list_last_module

    # 否则，没有绑定的权重，没有定义输出嵌入，简单地保持最后一个模块的完整精度
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]
    # 将最后一个模块与绑定的权重一起添加
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)

    # 从键中移除 ".weight"
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names
```