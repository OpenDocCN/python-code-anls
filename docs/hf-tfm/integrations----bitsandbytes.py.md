# `.\integrations\bitsandbytes.py`

```
import importlib.metadata  # 导入元数据模块，用于获取包的版本信息
import warnings  # 导入警告模块，用于处理警告信息
from copy import deepcopy  # 导入深拷贝函数，用于复制对象
from inspect import signature  # 导入签名模块，用于获取函数的参数签名信息

from packaging import version  # 导入版本模块，用于处理版本号

from ..utils import is_accelerate_available, is_bitsandbytes_available, logging  # 导入自定义工具函数和日志模块


if is_bitsandbytes_available():
    import bitsandbytes as bnb  # 如果bitsandbytes可用，导入bitsandbytes库
    import torch  # 导入PyTorch库
    import torch.nn as nn  # 导入PyTorch的神经网络模块

    from ..pytorch_utils import Conv1D  # 导入自定义的Conv1D模块

if is_accelerate_available():
    from accelerate import init_empty_weights  # 如果accelerate可用，导入初始化空权重函数
    from accelerate.utils import find_tied_parameters  # 导入查找绑定参数的函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


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
    # 如果张量名包含点号，递归访问模块的子模块直到找到张量名
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    # 如果张量名不在参数或缓冲区中，抛出值错误异常
    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers  # 标记张量名是否在缓冲区中
    old_value = getattr(module, tensor_name)  # 获取模块中的旧张量值

    # 如果旧张量值在meta设备上，但目标设备不是meta，且没有提供值，则抛出值错误异常
    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    prequantized_loading = quantized_stats is not None  # 标记是否为预量化加载

    # 如果是缓冲区或bitsandbytes库不可用，则不是4位或8位量化
    if is_buffer or not is_bitsandbytes_available():
        is_8bit = False
        is_4bit = False
    else:
        # 检查是否是4位参数并且模块参数是4位
        is_4bit = hasattr(bnb.nn, "Params4bit") and isinstance(module._parameters[tensor_name], bnb.nn.Params4bit)
        # 检查是否是8位参数
        is_8bit = isinstance(module._parameters[tensor_name], bnb.nn.Int8Params)
    # 检查是否为8位或4位量化模型
    if is_8bit or is_4bit:
        # 获取模块中指定张量名称的参数
        param = module._parameters[tensor_name]
        # 如果参数不在CUDA设备上，则需要进行数据迁移
        if param.device.type != "cuda":
            # 根据值的类型和情况，将旧值转移到指定设备上或者转换为CPU上的张量
            if value is None:
                new_value = old_value.to(device)
            elif isinstance(value, torch.Tensor):
                new_value = value.to("cpu")
            else:
                new_value = torch.tensor(value, device="cpu")

            # 如果模块源类型是Conv1D，并且不是预量化加载情况下，需要转置权重矩阵以支持Conv1D替代nn.Linear的模型
            if issubclass(module.source_cls, Conv1D) and not prequantized_loading:
                new_value = new_value.T

            # 将旧值的属性作为关键字参数传递给新值
            kwargs = old_value.__dict__

            # 检查新值的dtype是否与参数量化状态兼容
            if prequantized_loading != (new_value.dtype in (torch.int8, torch.uint8)):
                raise ValueError(
                    f"Value dtype `{new_value.dtype}` is not compatible with parameter quantization status."
                )

            # 如果是8位量化模型
            if is_8bit:
                # 检查bitsandbytes库版本是否支持int8的序列化
                is_8bit_serializable = version.parse(importlib.metadata.version("bitsandbytes")) > version.parse(
                    "0.37.2"
                )
                # 如果新值的dtype是int8或uint8且bitsandbytes版本不支持int8序列化，则抛出错误
                if new_value.dtype in (torch.int8, torch.uint8) and not is_8bit_serializable:
                    raise ValueError(
                        "Detected int8 weights but the version of bitsandbytes is not compatible with int8 serialization. "
                        "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                    )
                # 使用bitsandbytes库中的Int8Params将新值转换为int8参数，设置不需要梯度，并应用到指定设备上
                new_value = bnb.nn.Int8Params(new_value, requires_grad=False, **kwargs).to(device)
                # 如果是预量化加载情况下，将quantized_stats中的SCB属性设置到新值的SCB属性上
                if prequantized_loading:
                    setattr(new_value, "SCB", quantized_stats["SCB"].to(device))
            # 如果是4位量化模型
            elif is_4bit:
                # 如果是预量化加载情况下，检查bitsandbytes库版本是否支持4位的序列化
                if prequantized_loading:
                    is_4bit_serializable = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse(
                        "0.41.3"
                    )
                    # 如果新值的dtype是int8或uint8且bitsandbytes版本不支持4位序列化，则抛出错误
                    if new_value.dtype in (torch.int8, torch.uint8) and not is_4bit_serializable:
                        raise ValueError(
                            "Detected 4-bit weights but the version of bitsandbytes is not compatible with 4-bit serialization. "
                            "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                        )
                    # 使用bitsandbytes库中的Params4bit.from_prequantized方法从预量化数据创建4位参数，设置不需要梯度，并应用到指定设备上
                    new_value = bnb.nn.Params4bit.from_prequantized(
                        data=new_value,
                        quantized_stats=quantized_stats,
                        requires_grad=False,
                        device=device,
                        **kwargs,
                    )
                else:
                    # 使用bitsandbytes库中的Params4bit将新值转换为4位参数，设置不需要梯度，并应用到指定设备上
                    new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(device)
            # 将模块中指定张量名称的参数更新为新值
            module._parameters[tensor_name] = new_value
    else:
        # 如果value为None，则将old_value转移到指定设备（device）
        if value is None:
            new_value = old_value.to(device)
        # 如果value是torch.Tensor类型，则将其移动到指定设备（device）
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        # 否则，将value转换为torch.tensor，并移动到指定设备（device）
        else:
            new_value = torch.tensor(value, device=device)

        # 如果是缓冲区（buffer），则更新module的_buffers字典
        if is_buffer:
            module._buffers[tensor_name] = new_value
        # 否则，将new_value封装为nn.Parameter，并将其存储在module的_parameters字典中
        else:
            new_value = nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value
# 定义一个私有方法，用于递归替换模块的功能。返回替换后的模型和一个布尔值，指示替换是否成功。
def _replace_with_bnb_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    return model, has_been_replaced


# 定义一个函数，用于将所有 `torch.nn.Linear` 模块替换为 `bnb.nn.Linear8bit` 模块。
# 这样可以实现使用混合 int8 精度运行模型，如论文 `LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale` 所述。
# 在运行此函数之前，请确保已正确安装支持正确 CUDA 版本的 `bitsandbytes` 库。
# `pip install -i https://test.pypi.org/simple/bitsandbytes`
def replace_with_bnb_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `LLM.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes`

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Int8 mixed-precision matrix decomposition works by separating a
    matrix multiplication into two streams: (1) and systematic feature outlier stream matrix multiplied in fp16
    (0.01%), (2) a regular stream of int8 matrix multiplication (99.9%). With this method, int8 inference with no
    predictive degradation is possible for very large models (>=176B parameters).

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `Linear8bitLt`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    # 调用私有方法 `_replace_with_bnb_linear` 进行实际的替换操作
    model, has_been_replaced = _replace_with_bnb_linear(
        model, modules_to_not_convert, current_key_name, quantization_config
    )

    # 如果没有替换成功，则记录警告信息，提示可能出现了问题
    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    # 返回替换后的模型
    return model


# 为了向后兼容而定义的占位符注释
# 引发 FutureWarning 警告，提示 `replace_8bit_linear` 将来会被弃用，建议使用 `replace_with_bnb_linear` 替代
def replace_8bit_linear(*args, **kwargs):
    warnings.warn(
        "`replace_8bit_linear` will be deprecated in a future version, please use `replace_with_bnb_linear` instead",
        FutureWarning,
    )
    # 调用 `replace_with_bnb_linear` 函数并返回其结果
    return replace_with_bnb_linear(*args, **kwargs)


# 为了向后兼容性而设立的函数
# 引发 FutureWarning 警告，提示 `set_module_8bit_tensor_to_device` 将来会被弃用，建议使用 `set_module_quantized_tensor_to_device` 替代
def set_module_8bit_tensor_to_device(*args, **kwargs):
    warnings.warn(
        "`set_module_8bit_tensor_to_device` will be deprecated in a future version, please use `set_module_quantized_tensor_to_device` instead",
        FutureWarning,
    )
    # 调用 `set_module_quantized_tensor_to_device` 函数并返回其结果
    return set_module_quantized_tensor_to_device(*args, **kwargs)


def get_keys_to_not_convert(model):
    r"""
    获取模块的键列表，用于指定不转换为 int8 的模块。例如对于 CausalLM 模块，
    我们可能希望保持 lm_head 以完整精度，以确保数值稳定性。对于其他架构，
    我们可能希望保持模型的 tied weights。该函数将返回一个不需要转换为 int8 的模块键列表。

    Parameters:
    model (`torch.nn.Module`):
        输入的模型
    """
    # 复制模型并绑定权重，然后检查是否包含绑定的权重
    tied_model = deepcopy(model)  # 这个操作在 `init_empty_weights` 上下文管理器内部不会有额外开销
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # 兼容 Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # 如果没有绑定的权重，我们希望保持 lm_head（output_embedding）以完整精度
    if not has_tied_params:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            list_last_module = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            return list_last_module

    # 否则，没有绑定的权重，也没有定义输出嵌入，简单地保持最后一个模块以完整精度
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]
    # 将最后一个模块与绑定的权重一起添加到列表中
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)

    # 从键中移除 ".weight" 和 ".bias"
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names
```