# `.\transformers\modeling_utils.py`

```
# 导入必要的库和模块
import collections  # 导入 collections 模块，用于提供额外的数据结构和工具
import copy  # 导入 copy 模块，用于对象的浅拷贝和深拷贝操作
import functools  # 导入 functools 模块，用于高阶函数和操作函数的工具
import gc  # 导入 gc 模块，用于垃圾回收
import importlib.metadata  # 导入 importlib.metadata 模块，用于读取元数据
import inspect  # 导入 inspect 模块，用于检查活动对象
import itertools  # 导入 itertools 模块，用于创建迭代器的工具
import json  # 导入 json 模块，用于 JSON 数据的编解码
import os  # 导入 os 模块，提供操作系统相关的功能
import re  # 导入 re 模块，用于正则表达式匹配
import shutil  # 导入 shutil 模块，提供文件操作相关的函数
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import warnings  # 导入 warnings 模块，用于警告处理
from contextlib import contextmanager  # 导入 contextmanager 类，用于创建上下文管理器
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于创建数据类
from functools import partial, wraps  # 导入 partial 和 wraps 函数，用于部分应用和装饰器
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类和函数
from zipfile import is_zipfile  # 导入 is_zipfile 函数，用于检查文件是否为 ZIP 格式

import torch  # 导入 torch 库，用于深度学习模型的构建和训练
from packaging import version  # 导入 version 类，用于版本号处理
from torch import Tensor, nn  # 导入 Tensor 和 nn 类，用于张量和神经网络相关的操作
from torch.nn import CrossEntropyLoss, Identity  # 导入 CrossEntropyLoss 和 Identity 类，用于损失函数和身份映射
from torch.utils.checkpoint import checkpoint  # 导入 checkpoint 函数，用于模型的检查点操作

# 导入自定义模块和函数
from .activations import get_activation  # 导入 get_activation 函数，用于获取激活函数
from .configuration_utils import PretrainedConfig  # 导入 PretrainedConfig 类，用于预训练配置
from .dynamic_module_utils import custom_object_save  # 导入 custom_object_save 函数，用于自定义对象保存
from .generation import GenerationConfig, GenerationMixin  # 导入 GenerationConfig 和 GenerationMixin 类，用于生成相关的配置和混合类
from .integrations import (  # 导入 integrations 模块，用于集成其他框架或库
    PeftAdapterMixin,  # 导入 PeftAdapterMixin 类，用于 PEFT 适配器混合类
    deepspeed_config,  # 导入 deepspeed_config 函数，用于 DeepSpeed 配置
    is_deepspeed_zero3_enabled,  # 导入 is_deepspeed_zero3_enabled 函数，用于检查是否启用 DeepSpeed Zero3
)
from .pytorch_utils import (  # 导入 pytorch_utils 模块，提供 PyTorch 工具函数
    Conv1D,  # 导入 Conv1D 类，用于一维卷积操作
    apply_chunking_to_forward,  # 导入 apply_chunking_to_forward 函数，用于将前向传播分块应用到模型中
    find_pruneable_heads_and_indices,  # 导入 find_pruneable_heads_and_indices 函数，用于查找可剪枝的头和索引
    id_tensor_storage,  # 导入 id_tensor_storage 函数，用于创建张量存储
    is_torch_greater_or_equal_than_1_13,  # 导入 is_torch_greater_or_equal_than_1_13 函数，用于检查 PyTorch 版本是否大于等于 1.13
    prune_conv1d_layer,  # 导入 prune_conv1d_layer 函数，用于剪枝一维卷积层
    prune_layer,  # 导入 prune_layer 函数，用于剪枝层
    prune_linear_layer,  # 导入 prune_linear_layer 函数，用于剪枝线性层
)
from .safetensors_conversion import auto_conversion  # 导入 auto_conversion 函数，用于安全张量的转换
from .utils import (  # 导入 utils 模块，提供各种实用函数和工具
    ADAPTER_SAFE_WEIGHTS_NAME,  # 导入 ADAPTER_SAFE_WEIGHTS_NAME 常量，表示适配器安全权重的名称
    ADAPTER_WEIGHTS_NAME,  # 导入 ADAPTER_WEIGHTS_NAME 常量，表示适配器权重的名称
    CONFIG_NAME,  # 导入 CONFIG_NAME 常量，表示配置文件的名称
    DUMMY_INPUTS,  # 导入 DUMMY_INPUTS 常量，表示虚拟输入
    FLAX_WEIGHTS_NAME,  # 导入 FLAX_WEIGHTS_NAME 常量，表示 FLAX 权重的名称
    SAFE_WEIGHTS_INDEX_NAME,  # 导入 SAFE_WEIGHTS_INDEX_NAME 常量，表示安全权重索引的名称
    SAFE_WEIGHTS_NAME,  # 导入 SAFE_WEIGHTS_NAME 常量，表示安全权重的名称
    TF2_WEIGHTS_NAME,  # 导入 TF2_WEIGHTS_NAME 常量，表示 TensorFlow 2.x 权重的名称
    TF_WEIGHTS_NAME,  # 导入 TF_WEIGHTS_NAME 常量，表示 TensorFlow 权重的名称
    WEIGHTS_INDEX_NAME,  # 导入 WEIGHTS_INDEX_NAME 常量，表示权重索引的名称
    WEIGHTS_NAME,  # 导入 WEIGHTS_NAME 常量，表示权重的名称
    ContextManagers,  # 导入 Context
# 从 utils.quantization_config 模块中导入 AwqConfig, BitsAndBytesConfig, GPTQConfig, QuantizationMethod
from .utils.quantization_config import AwqConfig, BitsAndBytesConfig, GPTQConfig, QuantizationMethod
# 获取环境变量 XLA_USE_BF16 的值，如果不存在则默认为 "0"，并转换为大写
XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
# 获取环境变量 XLA_DOWNCAST_BF16 的值，如果不存在则默认为 "0"，并转换为大写

XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()

# 如果加速库可用
if is_accelerate_available():
    # 从 accelerate 模块中导入 dispatch_model, infer_auto_device_map, init_empty_weights
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    # 从 accelerate.hooks 模块中导入 add_hook_to_module
    from accelerate.hooks import add_hook_to_module
    # 从 accelerate.utils 模块中导入一系列函数
    from accelerate.utils import (
        check_tied_parameters_on_same_device,
        find_tied_parameters,
        get_balanced_memory,
        get_max_memory,
        load_offloaded_weights,
        offload_weight,
        save_offload_index,
        set_module_tensor_to_device,
    )

# 如果 SafeTensors 可用
if is_safetensors_available():
    # 从 safetensors 模块中导入 safe_open
    from safetensors import safe_open
    # 从 safetensors.torch 模块中导入 load_file 和 save_file
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 初始化权重标志
_init_weights = True

# 判断是否启用 FSDP
def is_fsdp_enabled():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )

# 判断是否为本地分布式训练的第一个进程
def is_local_dist_rank_0():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and int(os.environ.get("LOCAL_RANK", -1)) == 0
    )

# 如果 SageMaker Model Parallel 可用
if is_sagemaker_mp_enabled():
    # 从 smdistributed.modelparallel.torch 模块中导入 smp
    import smdistributed.modelparallel.torch as smp
    # 从 smdistributed.modelparallel 模块中导入 __version__ 并重命名为 SMP_VERSION
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    # 判断 SageMaker Model Parallel 版本是否大于等于 1.10
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

# 如果 PEFT 可用
if is_peft_available():
    # 从 utils 模块中导入 find_adapter_config_file

# 初始化 Torch 初始化函数字典
TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}

# 定义上下文管理器 no_init_weights
@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights

    if _enable:
        _init_weights = False

        def _skip_init(*args, **kwargs):
            pass

        # # Save the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, _skip_init)
    # 尝试执行代码块
    try:
        yield
    # 无论是否发生异常，都会执行以下代码
    finally:
        # 恢复原始的初始化权重函数
        _init_weights = old_init_weights
        # 如果启用了功能
        if _enable:
            # 恢复原始的初始化函数
            for name, init_func in TORCH_INIT_FUNCTIONS.items():
                setattr(torch.nn.init, name, init_func)
# 获取参数的设备信息
def get_parameter_device(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    try:
        # 返回参数的第一个参数的设备信息
        return next(parameter.parameters()).device
    except StopIteration:
        # 对于 PyTorch 1.5 中的 nn.DataParallel 兼容性

        # 查找模块中的张量属性
        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        # 生成器获取模块中的张量属性
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        # 获取第一个元组
        first_tuple = next(gen)
        return first_tuple[1].device


# 获取第一个参数的数据类型
def get_first_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    """
    返回第一个参数的数据类型（可以是非浮点数），如果没有找到则断言。
    """
    try:
        # 返回第一个参数的数据类型
        return next(parameter.parameters()).dtype
    except StopIteration:
        # 对于 PyTorch > 1.5 中的 nn.DataParallel 兼容性

        # 查找模块中的张量属性
        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        # 生成器获取模块中的张量属性
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        # 获取第一个元组
        first_tuple = next(gen)
        return first_tuple[1].dtype


# 获取参数的数据类型
def get_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    """
    如果找到浮点数据类型，则返回参数中找到的第一个浮点数据类型，否则返回找到的最后一个数据类型。
    """
    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            # 为 https://github.com/pytorch/xla/issues/4152 添加修复
            # 修复模型代码传递的值超出 XLA_USE_BF16=1 和 XLA_DOWNCAST_BF16=1 范围的问题，导致转换为 -inf
            # 注意：`is_torch_tpu_available()` 在最后检查，因为它会在 torch dynamo 中引入图断点
            if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_tpu_available():
                return torch.bfloat16
            if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_tpu_available():
                if t.dtype == torch.float:
                    return torch.bfloat16
                if t.dtype == torch.double:
                    return torch.float32
            return t.dtype

    if last_dtype is not None:
        # 如果没有找到浮点数据类型，则返回第一个数据类型
        return last_dtype

    # 对于 PyTorch > 1.5 中的 nn.DataParallel 兼容性
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    # 生成器获取模块中的张量属性
    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype
    # 如果上一个元组不为空，则返回上一个元组中的数据类型
    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype

    # 如果上一个元组为空，则使用缓冲区数据类型作为备选
    for t in parameter.buffers():
        # 获取当前缓冲区的数据类型
        last_dtype = t.dtype
        # 如果当前缓冲区是浮点数类型，则返回当前缓冲区的数据类型
        if t.is_floating_point():
            return t.dtype
    # 返回最后一个缓冲区的数据类型作为备选
    return last_dtype
def get_state_dict_float_dtype(state_dict):
    """
    返回`state_dict`中第一个找到的浮点数据类型，如果没有找到则断言。
    """
    # 遍历state_dict的值
    for t in state_dict.values():
        # 如果值是浮点数
        if t.is_floating_point():
            # 返回数据类型
            return t.dtype

    # 如果在state_dict中找不到浮点数据类型，则引发错误
    raise ValueError("couldn't find any floating point dtypes in state_dict")


def get_state_dict_dtype(state_dict):
    """
    如果找到浮点数据类型，则返回`state_dict`中第一个找到的浮点数据类型，否则返回第一个数据类型。
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # 如果找不到浮点数据类型，则返回第一个数据类型
    else:
        return next(state_dict.values()).dtype


def dtype_byte_size(dtype):
    """
    返回`dtype`类型参数占用的字节数大小。

    示例：

    ```py
    >>> dtype_byte_size(torch.float32)
    4
    ```
    """
    # 如果数据类型是torch.bool
    if dtype == torch.bool:
        return 1 / 8
    # 使用正则表达式查找数据类型的位数
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def shard_checkpoint(
    state_dict: Dict[str, torch.Tensor], max_shard_size: Union[int, str] = "10GB", weights_name: str = WEIGHTS_NAME
):
    """
    将模型状态字典分割成子检查点，使每个子检查点的最终大小不超过给定大小。

    子检查点是通过按键的顺序迭代`state_dict`来确定的，因此没有优化使每个子检查点尽可能接近传递的最大大小。例如，如果限制为10GB，且权重大小为[6GB, 6GB, 2GB, 6GB, 2GB, 2GB]，它们将被分割为[6GB]，[6+2GB]，[6+2+2GB]，而不是[6+2+2GB]，[6+2GB]，[6GB]。

    <Tip warning={true}>

    如果模型的某个权重大于`max_shard_size`，它将最终位于自己的子检查点中，其大小大于`max_shard_size`。

    </Tip>

    Args:
        state_dict (`Dict[str, torch.Tensor]`): 要保存的模型的状态字典。
        max_shard_size (`int` or `str`，*可选*，默认为`"10GB"`):
            每个子检查点的最大大小。如果表示为字符串，需要是数字后跟一个单位（如`"5MB"`）。
        weights_name (`str`，*可选*，默认为`"pytorch_model.bin"`):
            模型保存文件的名称。
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = [{}]
    last_block_size = 0
    total_size = 0
    storage_id_to_block = {}
    # 遍历状态字典中的键值对，键为权重名称，值为权重数据
    for key, weight in state_dict.items():
        # 当使用 bnb 序列化时，状态字典中的权重可能是字符串
        # 参考：https://github.com/huggingface/transformers/pull/24416 获取更多细节
        if isinstance(weight, str):
            # 如果权重是字符串类型，则跳过
            continue
        else:
            # 获取权重数据的存储 ID
            storage_id = id_tensor_storage(weight)

        # 如果一个权重与另一个张量共享相同的底层存储，则将其放入相同的“块”中
        if storage_id in storage_id_to_block:
            block_id = storage_id_to_block[storage_id]
            sharded_state_dicts[block_id][key] = weight
            # 继续下一个循环
            continue

        # 计算权重数据的大小
        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # 如果该权重将超过最大大小，则进行分割，但仅在当前块中至少放入一个权重时才执行
        if last_block_size + weight_size > max_shard_size and len(sharded_state_dicts[-1]) > 0:
            sharded_state_dicts.append({})
            last_block_size = 0

        # 将权重数据添加到当前块中
        sharded_state_dicts[-1][key] = weight
        last_block_size += weight_size
        total_size += weight_size
        storage_id_to_block[storage_id] = len(sharded_state_dicts) - 1

    # 如果只有一个块，则返回该块
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # 否则，构建索引
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        # 为每个块创建文件名
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        # 更新权重映射
        for key in shard.keys():
            weight_map[key] = shard_file

    # 添加元数据
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index
# 加载分片检查点到模型中
def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    """
    This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`)
            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True  # load safe since we have no other choice

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    # 如果启用了严格模式并且存在缺失的键或者多余的键
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        # 生成错误信息，指明加载状态字典时出现的错误
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        # 如果存在缺失的键
        if len(missing_keys) > 0:
            # 将缺失的键转换为字符串，并拼接到错误信息中
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        # 如果存在多余的键
        if len(unexpected_keys) > 0:
            # 将多余的键转换为字符串，并拼接到错误信息中
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        # 抛出运行时错误，包含错误信息
        raise RuntimeError(error_message)

    # 根据加载安全标志选择加载器
    loader = (
        safe_load_file
        if load_safe
        else partial(torch.load, map_location="cpu", weights_only=is_torch_greater_or_equal_than_1_13)
    )

    # 遍历分片文件列表
    for shard_file in shard_files:
        # 使用加载器加载状态字典
        state_dict = loader(os.path.join(folder, shard_file))
        # 将加载的状态字典应用到模型中，不进行严格匹配
        model.load_state_dict(state_dict, strict=False)

        # 确保在加载下一个状态字典之前释放内存
        del state_dict
        gc.collect()

    # 返回与 PyTorch load_state_dict 函数相同的结果
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """
    # 检查文件名是否以 ".safetensors" 结尾并且 safetensors 可用
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # 检查存档的格式
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
        # 如果存档的格式不是 "pt", "tf", "flax" 中的一种，则抛出异常
        if metadata.get("format") not in ["pt", "tf", "flax"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        return safe_load_file(checkpoint_file)
    try:
        # 如果启用了 DeepSpeed Zero3 并且分布式已初始化且分布式排名大于 0，或者启用了 FSDP 并且不是本地分布式排名为 0
        if (
            is_deepspeed_zero3_enabled() and torch.distributed.is_initialized() and torch.distributed.get_rank() > 0
        ) or (is_fsdp_enabled() and not is_local_dist_rank_0()):
            map_location = "meta"
        else:
            map_location = "cpu"
        extra_args = {}
        # 只有在使用基于 zipfile 的格式序列化的文件时才能使用 mmap
        if (
            isinstance(checkpoint_file, str)
            and map_location != "meta"
            and version.parse(torch.__version__) >= version.parse("2.1.0")
            and is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}
        return torch.load(
            checkpoint_file,
            map_location=map_location,
            weights_only=is_torch_greater_or_equal_than_1_13,
            **extra_args,
        )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                # 如果文件的前 7 个字符是 "version"，则抛出异常
                if f.read(7) == "version":
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


def set_initialized_submodules(model, state_dict_keys):
    """
    Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
    dict.
    """
    not_initialized_submodules = {}
    # 遍历模型的命名模块及其对应的模块对象
    for module_name, module in model.named_modules():
        # 从状态字典键集合中获取以当前模块名开头的键，并去除模块名前缀，形成加载的键集合
        loaded_keys = {k.replace(f"{module_name}.", "") for k in state_dict_keys if k.startswith(f"{module_name}.")}
        # 如果加载的键集合包含当前模块的状态字典的所有键，则将当前模块标记为已经由 Hugging Face 初始化
        if loaded_keys.issuperset(module.state_dict()):
            module._is_hf_initialized = True
        # 如果加载的键集合不包含当前模块的状态字典的所有键，则将当前模块添加到未初始化的子模块字典中
        else:
            not_initialized_submodules[module_name] = module
    # 返回未初始化的子模块字典
    return not_initialized_submodules
# 将状态字典加载到模型中的内部函数
def _load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    # 如果需要，将旧格式转换为新格式，来自PyTorch状态字典
    old_keys = []  # 存储旧键的列表
    new_keys = []  # 存储新键的列表
    # 遍历状态字典中的键
    for key in state_dict.keys():
        new_key = None
        # 如果键中包含"gamma"，则将其替换为"weight"
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        # 如果键中包含"beta"，则将其替换为"bias"
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        # 如果有新键，则将旧键和新键分别存储到相应的列表中
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    # 使用新键替换旧键
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # 复制状态字典以便_load_from_state_dict函数可以修改它
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []  # 存储错误消息的列表

    # PyTorch的`_load_from_state_dict`不会复制模块后代中的参数，因此我们需要递归应用该函数。
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # 模块及其子模块的参数将以prefix开头。如果在这个state_dict中没有这些参数，我们可以提前退出
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            # 如果启用了DeepSpeed Zero3
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # 在分片模型中，每个分片只有完整state_dict的一部分，因此只收集当前state_dict中存在的参数
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                # 如果有参数需要收集
                if len(params_to_gather) > 0:
                    # 因为zero3在模型参数中放置了占位符，这个上下文管理器收集（解析）当前层的参数，然后从state_dict加载，
                    # 然后再次将它们重新分区
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            # 加载来自state_dict的参数
                            module._load_from_state_dict(*args)
            else:
                # 加载来自state_dict的参数
                module._load_from_state_dict(*args)

        # 递归地加载子模块
        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    # 从给定的前缀开始递归加载模型及其子模块
    load(model_to_load, state_dict, prefix=start_prefix)
    # 删除state_dict，以便它可以更早地被垃圾回收。请注意，state_dict是参数的副本，因此可以安全地删除它。
    del state_dict

    return error_msgs


# 查找模型中的子模块和参数名称
def find_submodule_and_param_name(model, long_key, start_prefix):
    """
    # 一个辅助函数，用于查找最后一个子模块及其参数/缓冲区名称。如果提供了`start_prefix`，则会从键的开头删除它。
    if len(start_prefix) > 0 and long_key.startswith(start_prefix):
        # 如果提供了`start_prefix`并且长键以它开头，则从长键中删除`start_prefix`
        long_key = ".".join(long_key.split(".")[1:])
    
    # 将长键按`.`分割成列表
    split_key = long_key.split(".")
    # 将子模块设置为模型
    submodule = model
    # 当分割后的键的长度大于1时执行循环
    while len(split_key) > 1:
        # 如果子模块有split_key的第一个元素属性
        if hasattr(submodule, split_key[0]):
            # 将子模块设置为split_key的第一个属性的值
            submodule = getattr(submodule, split_key[0])
            # 删除split_key的第一个元素
            del split_key[0]
        else:
            # 如果子模块没有split_key的第一个属性，则将子模块设为None并跳出循环
            submodule = None
            break
    # 如果子模块等于模型，则将子模块设为None
    if submodule == model:
        submodule = None
    # 返回子模块和剩余的键列表中的第一个元素
    return submodule, split_key[0]
# 将模型中的参数移动到元设备(meta device)，以释放这些参数占用的内存空间
def _move_model_to_meta(model, loaded_state_dict_keys, start_prefix):
    """
    Moves `loaded_state_dict_keys` in model to meta device which frees up the memory taken by those params.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """

    # 将要被状态字典(state_dict)替换的参数存储在元设备上，实现参数的去材料化(dematerialize)
    for k in loaded_state_dict_keys:
        # 找到模型中子模块和参数名，以便后续处理
        submodule, param_name = find_submodule_and_param_name(model, k, start_prefix)
        if submodule is not None:
            # 仅选择性地将下一个要从状态字典替换的参数/缓冲区切换到元设备，
            # 这是一种复杂的方式，因为我们没有用于张量的原地(in-place) to_ 方法。
            new_val = getattr(submodule, param_name)
            if isinstance(new_val, torch.nn.Parameter):
                # 如果是参数，则先将参数转移到元设备上
                new_val = torch.nn.Parameter(new_val.to("meta"))
            else:
                # 如果是普通张量，则直接转移到元设备上
                new_val = new_val.to("meta")
            # 将参数/缓冲区替换为在元设备上的新值
            setattr(submodule, param_name, new_val)


def _load_state_dict_into_meta_model(
    model,
    state_dict,
    loaded_state_dict_keys,  # left for now but could be removed, see below
    start_prefix,
    expected_keys,
    device_map=None,
    offload_folder=None,
    offload_index=None,
    state_dict_folder=None,
    state_dict_index=None,
    dtype=None,
    is_quantized=False,
    is_safetensors=False,
    keep_in_fp32_modules=None,
    unexpected_keys=None,  # passing `unexpected` for cleanup from quantization items
):
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
    params back to the normal device, but only for `loaded_state_dict_keys`.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """

    # XXX: remaining features to implement to be fully compatible with _load_state_dict_into_model
    # - deepspeed zero 3 support
    # - need to copy metadata if any - see _load_state_dict_into_model
    # - handling error_msgs - mimicking the error handling in module._load_from_state_dict()
    # - Is there a situation where some keys aren't in `loaded_state_dict_keys` and in which case
    #   they won't get loaded.

    if is_quantized:
        from .integrations import set_module_quantized_tensor_to_device

    error_msgs = []

    old_keys = []
    new_keys = []
    # 遍历状态字典中的键
    for key in state_dict.keys():
        # 初始化新键为 None
        new_key = None
        # 如果键中包含 "gamma"，则替换为 "weight"
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        # 如果键中包含 "beta"，则替换为 "bias"
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        # 如果存在新键
        if new_key:
            # 将旧键添加到旧键列表中
            old_keys.append(key)
            # 将新键添加到新键列表中
            new_keys.append(new_key)
    # 遍历旧键和新键列表，将状态字典中的键替换
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # 返回错误消息、卸载索引和状态字典索引
    return error_msgs, offload_index, state_dict_index
def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    # 如果 variant 参数不为空
    if variant is not None:
        # 使用 '.' 分割 weights_name 字符串，并将 variant 插入到倒数第二个位置
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        # 将列表重新拼接成字符串
        weights_name = ".".join(splits)

    # 返回处理后的 weights_name
    return weights_name


class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        # 导入 psutil 库，如果导入失败，抛出 ImportError
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        # 获取当前进程的内存信息
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        # 记录当前模块的内存占用（预前向传播）
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        # 导入 psutil 库，如果导入失败，抛出 ImportError
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        # 获取当前进程的内存信息
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        # 记录当前模块的内存占用（后向传播）
        module.mem_rss_post_forward = mem.rss
        # 计算内存占用差值
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        # 更新模块的内存占用差值属性
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None

    def add_memory_hooks(self):
        """
        Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

        Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero
        with `model.reset_memory_hooks_state()`.
        """
        # 遍历模块中的每个子模块
        for module in self.modules():
            # 注册前向传播前钩子函数
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            # 注册前向传播后钩子函数
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        # 重置内存钩子状态
        self.reset_memory_hooks_state()

    def reset_memory_hooks_state(self):
        """
        Reset the `mem_rss_diff` attribute of each module (see [`~modeling_utils.ModuleUtilsMixin.add_memory_hooks`]).
        """
        # 遍历模块中的每个子模块
        for module in self.modules():
            # 将模块的内存属性重置为零
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # 返回模块的设备信息
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        # 返回模块的数据类型信息
        return get_parameter_dtype(self)
    # 反转注意力掩码（例如，将0和1互换）
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        # 如果注意力掩码的维度为3
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        # 如果注意力掩码的维度为2
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5有一个可以比较序列ID的掩码，我们可以通过这个转置来模拟
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        # 将注意力掩码转换为指定数据类型（用于fp16兼容性）
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
        # 将注意力掩码反转并乘以最小值
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask

    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        # 如果设备参数不为空，则发出警告
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
        else:
            device = attention_mask.device
        # 获取输入形状的批量大小和序列长度
        batch_size, seq_length = input_shape
        # 创建序列ID
        seq_ids = torch.arange(seq_length, device=device)
        # 创建因果掩码
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # 将因果掩码转换为与注意力掩码相同的数据类型
        causal_mask = causal_mask.to(attention_mask.dtype)

        # 如果因果掩码的形状小于注意力掩码的形状
        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            # 添加前缀1掩码��因果掩码
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        # 创建扩展的注意力掩码
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    # 获取扩展的注意力掩码
    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # 如果未提供 dtype，则使用 self.dtype
        if dtype is None:
            dtype = self.dtype

        # 如果注意力掩码维度不是2，并且模型不是decoder，则显示警告
        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # 只有在 `create_extended_attention_mask_for_decoder` 中没有显示警告时才显示警告
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        
        # 如果注意力掩码维度为3，则将其扩展为 [batch_size, 1, from_seq_length, to_seq_length]
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        # 如果注意力掩码维度为2
        elif attention_mask.dim() == 2:
            # 如果模型是decoder，则在padding掩码之外还应用一个因果掩码
            # 如果模型是encoder，则将掩码扩展为 [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            # 如果输入形状（input_ids）或注意力掩码（attention_mask）形状不正确，则引发错误
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # 由于注意力掩码对于我们想要注意的位置为1.0，对于掩码位置为0.0，
        # 此操作将创建一个张量，对于我们想要注意的位置为0.0，掩码位置为dtype的最小值。
        # 由于我们在softmax之前将其添加到原始分数中，这实际上等同于完全删除这些位置。
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # 为了fp16兼容性
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        # 如果需要，准备头部掩码
        if head_mask is not None:
            # 将头部掩码转换为5维张量
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            # 如果注意力分块计算，则在最后一维上增加一个维度
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            # 如果头部掩码为None，则创建一个包含None的列表，长度为num_hidden_layers
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        # 如果头部掩码的维度为1，则扩展维度为5维
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        # 如果头部掩码的维度为2，则在指定每层的头部掩码
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        # 断言头部掩码的维度为5
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        # 将头部掩码转换为指定的数据类型，以便与fp16兼容
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask
    # 计算模块中的参数数量，可选择是否仅包括可训练参数或非嵌入层参数
    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        # 如果要排除嵌入层参数
        if exclude_embeddings:
            # 获取所有嵌入层参数的名称列表
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            # 获取排除嵌入层参数后的所有参数列表
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            # 获取所有参数列表
            total_parameters = list(self.parameters())

        # 存储所有参数的元素数量
        total_numel = []
        # 检查是否以4位精度加载模型
        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)
        # 如果以4位精度加载模型
        if is_loaded_in_4bit:
            # 检查是否安装了 bitsandbytes 库
            if is_bitsandbytes_available():
                import bitsandbytes as bnb
            else:
                # 若未安装 bitsandbytes 库，则引发异常
                raise ValueError(
                    "bitsandbytes is not installed but it seems that the model has been loaded in 4bit precision, something went wrong"
                    " make sure to install bitsandbytes with `pip install bitsandbytes`. You also need a GPU. "
                )

        # 遍历所有参数
        for param in total_parameters:
            # 如果参数需要梯度或者不仅需要可训练参数
            if param.requires_grad or not only_trainable:
                # 对于以4位精度加载的模型，将参数数量乘以2，因为一半的参数用于4位量化（uint8张量存储）
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    total_numel.append(param.numel() * 2)
                else:
                    total_numel.append(param.numel())

        # 返回所有参数的元素总数
        return sum(total_numel)

    # 估算模型输入中的总令牌数
    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        # 如果模型实例中不存在警告已发出的标志，则初始化一个空字典
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        # 如果主要输入名称在输入字典中
        if self.main_input_name in input_dict:
            # 返回主要输入张量的元素数量（令牌数）
            return input_dict[self.main_input_name].numel()
        # 如果主要输入名称不在输入字典中且未发出“estimate_tokens”警告
        elif "estimate_tokens" not in self.warnings_issued:
            # 记录“estimate_tokens”警告已发出
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        # 返回令牌数为0
        return 0
    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
        paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (`int`):
                The batch size for the forward pass.

            sequence_length (`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (`bool`, *optional*, defaults to `True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            `int`: The number of floating-point operations.
        """

        # 返回估计的浮点运算次数，根据输入字典中的数据估计tokens数量和是否排除嵌入操作
        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)
class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a PyTorch model,
          taking as arguments:

            - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
            - **config** ([`PreTrainedConfig`]) -- An instance of the configuration associated to the model.
            - **path** (`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None

    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    is_parallelizable = False
    supports_gradient_checkpointing = False

    # Flash Attention 2 support
    _supports_flash_attn_2 = False

    # SDPA support
    _supports_sdpa = False

    # Has support for a `Cache` instance as `past_key_values`
    _supports_cache_class = False

    @property
    def dummy_inputs(self) -> Dict[str, torch.Tensor]:
        """
        `Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        # 返回一个包含虚拟输入数据的字典，用于在网络中进行前向传播
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a PyTorch model.
        """
        # 返回字符串 "pt"，表示这是一个 PyTorch 模型
        return "pt"

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 保存配置和预训练权重的来源（如果在模型中给出）
        config = self._autoset_attn_implementation(
            config, torch_dtype=torch.get_default_dtype(), check_device_map=False
        )
        self.config = config

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        # 重写类属性以使其成为实例属性，这样像 `InstructBlipForConditionalGeneration` 这样的模型可以动态更新它，而无需修改类属性
        # 当使用不同的组件（例如 language_model）时。
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        # 在每个 Transformer 模型初始化结束时执行的方法，执行需要模型模块正确初始化的代码（例如权重初始化）
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # 删除已经使用的属性，以便不保存在配置中
            delattr(self.config, "gradient_checkpointing")
    def add_model_tags(self, tags: Union[List[str], str]) -> None:
        r"""
        Add custom tags into the model that gets pushed to the Hugging Face Hub. Will
        not overwrite existing tags in the model.

        Args:
            tags (`Union[List[str], str]`):
                The desired tags to inject in the model

        Examples:

        ```python
        from transformers import AutoModel

        model = AutoModel.from_pretrained("bert-base-cased")

        model.add_model_tags(["custom", "custom-bert"])

        # Push the model to your namespace with the name "my-custom-bert".
        model.push_to_hub("my-custom-bert")
        ```
        """
        # 如果输入的 tags 是字符串，则转换为列表
        if isinstance(tags, str):
            tags = [tags]

        # 如果 model_tags 属性为空，则初始化为空列表
        if self.model_tags is None:
            self.model_tags = []

        # 遍历输入的标签列表
        for tag in tags:
            # 如果标签不在模型标签列表中，则添加到模型标签列表中
            if tag not in self.model_tags:
                self.model_tags.append(tag)

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        # 从 kwargs 中获取 torch_dtype 参数，默认为 None
        torch_dtype = kwargs.pop("torch_dtype", None)
        # 从 kwargs 中获取 use_flash_attention_2 参数，默认为 False
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        # 如果指定了 torch_dtype，则修改默认的 torch.dtype
        dtype_orig = None
        if torch_dtype is not None:
            # 保存原始的 torch.dtype
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        # 深拷贝配置对象，避免在 _from_config 中修改原始配置
        config = copy.deepcopy(config)
        # 从 kwargs 中获取 attn_implementation 参数，并设置给配置对象
        config._attn_implementation = kwargs.pop("attn_implementation", None)
        # 自动设置注意力机制的实现方式
        config = cls._autoset_attn_implementation(
            config, use_flash_attention_2=use_flash_attention_2, check_device_map=False
        )

        # 如果启用了 DeepSpeed ZeRO-3
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # 输出日志信息
            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # 使用 DeepSpeed 进行初始化
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                # 创建模型实例
                model = cls(config, **kwargs)
        else:
            # 创建模型实例
            model = cls(config, **kwargs)

        # 恢复默认的 torch.dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """
        修改默认的dtype并返回先前的dtype。在想要以特定dtype实例化模型时需要使用此功能。

        Args:
            dtype (`torch.dtype`):
                要设置的浮点dtype。

        Returns:
            `torch.dtype`: 可用于恢复`torch.set_default_dtype(dtype)`的原始`dtype`。如果未修改，则返回`None`。

        注意 `set_default_dtype` 目前仅适用于浮点类型，并且如果例如传递了`torch.int64`，则会断言。因此，如果传递了非浮点`dtype`，此函数将引发异常。
        """
        如果dtype不是浮点数：
            引发值错误异常，指示无法在dtype={dtype}下实例化{cls.__name__}模型，因为它不是浮点dtype

        logger.info(f"正在以默认dtype {dtype} 实例化{cls.__name__}模型。")
        获取当前默认的dtype并存储在dtype_orig中
        dtype_orig = torch.get_default_dtype()
        设置默认dtype为传入的dtype
        torch.set_default_dtype(dtype)
        返回原始dtype
        return dtype_orig

    @property
    def base_model(self) -> nn.Module:
        """
        `torch.nn.Module`: 模型的主体部分。
        """
        返回模型的主体部分，即`self.base_model_prefix`指定的模型，如果没有则返回self

    @classmethod
    def can_generate(cls) -> bool:
        """
        返回此模型是否能够生成序列。

        Returns:
            `bool`: 是否能够使用`.generate()`生成序列。
        """
        # 检测`prepare_inputs_for_generation`是否已被覆盖，这是生成的要求之一。
        # 另外，模型还可以具有自定义的`generate`函数。
        如果`prepare_inputs_for_generation`和`generate`都在`GenerationMixin`中：
            返回False
        返回True

    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
        hard_check_only: bool = False,
    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
        """
        Checks the availability of SDPA for a given model.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `_attn_implementation` to "flash_attention_2" so that the model can initialize the correct attention module.
        """
        # 如果仅进行硬检查，并且当前模型不支持 SDPA，则抛出异常
        if hard_check_only:
            if not cls._supports_sdpa:
                raise ValueError(
                    f"{cls.__name__} does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet."
                    " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/28005. If you believe"
                    ' this error is a bug, please open an issue in Transformers GitHub repository and load your model with the argument `attn_implementation="eager"` meanwhile. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
                )
            # 如果 PyTorch SDPA 不可用，则抛出 ImportError
            if not is_torch_sdpa_available():
                raise ImportError(
                    "PyTorch SDPA requirements in Transformers are not met. Please install torch>=2.1.1."
                )

        # 如果 PyTorch SDPA 不可用或当前模型不支持 SDPA，则返回配置对象
        if not is_torch_sdpa_available() or not cls._supports_sdpa:
            return config

        # 检查是否使用了 BetterTransformer，若是，则返回配置对象
        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
        if _is_bettertransformer:
            return config

        # 如果非硬检查，将配置对象的 _attn_implementation 属性设置为 "sdpa"
        if not hard_check_only:
            config._attn_implementation = "sdpa"
        return config

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        # 定义函数，用于设置输入梯度
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        # 注册前向钩子，使输入梯度生效
        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self):
        """
        Removes the `_require_grads_hook`.
        """
        # 移除注册的前向钩子，停止设置输入梯度
        self._require_grads_hook.remove()

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        # 获取模型的输入嵌入层
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError
    def set_input_embeddings(self, value: nn.Module):
        """
        设置模型的输入嵌入。

        Args:
            value (`nn.Module`): 将词汇映射到隐藏状态的模块。
        """
        # 获取基础模型
        base_model = getattr(self, self.base_model_prefix, self)
        # 如果基础模型不是自身，则设置其输入嵌入
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            # 否则，抛出未实现的异常
            raise NotImplementedError

    def get_output_embeddings(self) -> nn.Module:
        """
        返回模型的输出嵌入。

        Returns:
            `nn.Module`: 一个将隐藏状态映射到词汇的 torch 模块。
        """
        return None  # 对具有输出嵌入的模型进行重写

    def _init_weights(self, module):
        """
        初始化权重。此方法应该被派生类重写，并且是唯一在使用 `from_pretrained` 加载检查点时调用的初始化方法。
        任何尝试在此函数之外初始化的尝试都将无效，因为所有 torch.nn.init 函数都被替换为跳过。
        """
        pass

    def _initialize_weights(self, module):
        """
        如果权重尚未初始化，则初始化权重。
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    def tie_weights(self):
        """
        在输入嵌入和输出嵌入之间绑定权重。

        如果配置中设置了 `torchscript` 标志，则不能处理参数共享，因此我们会克隆权重。
        """
        # 如果配置中设置了 `tie_word_embeddings` 标志
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            # 如果存在输出嵌入
            if output_embeddings is not None:
                # 绑定或克隆权重
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        # 如果配置中设置了 `is_encoder_decoder` 和 `tie_encoder_decoder` 标志
        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            # 如果基础模型存在
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            # 绑定编码器和解码器之间的权重
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        # 对模块进行循环
        for module in self.modules():
            # 如果模块具有 `_tie_weights` 属性
            if hasattr(module, "_tie_weights"):
                # 绑定权重
                module._tie_weights()

    @staticmethod
    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        # 如果使用 TorchScript，则克隆输入嵌入权重到输出嵌入权重
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            # 否则，将输入嵌入权重直接赋值给输出嵌入权重
            output_embeddings.weight = input_embeddings.weight

        # 如果输出嵌入层有偏置项
        if getattr(output_embeddings, "bias", None) is not None:
            # 使用常数值0填充，调整偏置项的大小以匹配输出嵌入权重的形状
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        # 如果输出嵌入层具有"out_features"属性且输入嵌入层具有"num_embeddings"属性
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            # 将输出嵌入层的输出特征数设置为输入嵌入层的嵌入数量
            output_embeddings.out_features = input_embeddings.num_embeddings

    def _get_no_split_modules(self, device_map: str):
        """
        Get the modules of the model that should not be spit when using device_map. We iterate through the modules to
        get the underlying `_no_split_modules`.

        Args:
            device_map (`str`):
                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: List of modules that should not be split
        """
        _no_split_modules = set()
        modules_to_check = [self]
        while len(modules_to_check) > 0:
            module = modules_to_check.pop(-1)
            # 如果模块不在不分割模块集合中，则继续检查其子模块
            if module.__class__.__name__ not in _no_split_modules:
                # 如果是预训练模型的子类
                if isinstance(module, PreTrainedModel):
                    # 如果模块的不分割模块属性为None，则抛出异常
                    if module._no_split_modules is None:
                        raise ValueError(
                            f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model "
                            "class needs to implement the `_no_split_modules` attribute."
                        )
                    else:
                        # 否则将模块的不分割模块属性加入不分割模块集合
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                # 将当前模块的子模块加入待检查模块列表
                modules_to_check += list(module.children())
        # 返回不分割模块集合的列表表示形式
        return list(_no_split_modules)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        # 调整模型的输入标记嵌入矩阵大小，如果`new_num_tokens != config.vocab_size`。
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 如果`new_num_tokens`为`None`且`pad_to_multiple_of`为`None`，则直接返回模型的输入标记嵌入模块指针
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # 更新基础模型和当前模型配置的词汇量大小
        self.config.vocab_size = model_embeds.weight.shape[0]
        self.vocab_size = model_embeds.weight.shape[0]

        # 如果需要，重新绑定权重
        self.tie_weights()

        return model_embeds
    # 调整模型中的 token embeddings 大小，以适应新的 token 数量，并可选择将其填充到指定的倍数
    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        # 获取旧的 embeddings
        old_embeddings = self.get_input_embeddings()
        # 调整 embeddings 大小
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        # 如果旧的 embeddings 有 "_hf_hook" 属性，则将其 hook 添加到新的 embeddings 上
        if hasattr(old_embeddings, "_hf_hook"):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        # 保留旧的 embeddings 是否需要梯度信息，并将新的 embeddings 设置相同的梯度信息
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        # 设置新的 embeddings 为输入 embeddings
        self.set_input_embeddings(new_embeddings)

        # 更新 new_num_tokens 为新的 embeddings 的实际大小
        if pad_to_multiple_of is not None:
            if is_deepspeed_zero3_enabled():
                import deepspeed

                with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                    new_num_tokens = new_embeddings.weight.shape[0]
            else:
                new_num_tokens = new_embeddings.weight.shape[0]

        # 如果 word embeddings 没有被绑定，确保 lm head 也被调整大小
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            # 调整 lm head 大小
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            # 如果旧的 lm head 有 "_hf_hook" 属性，则将其 hook 添加到新的 lm head 上
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            # 保留旧的 lm head 是否需要梯度信息，并将新的 lm head 设置相同的梯度信息
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            # 设置新的 lm head 为输出 embeddings
            self.set_output_embeddings(new_lm_head)

        # 返回输入 embeddings
        return self.get_input_embeddings()

    # 获取调整大小后的 embeddings
    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    # 获取调整大小后的 lm head
    def _get_resized_lm_head(
        self, old_lm_head: nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    # 复制原始 lm head 到调整大小后的 lm head
    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # 复制旧的 lm head 权重到新的 lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # 复制偏置权重到新的 lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
    # 抛出未实现错误，提示需要在子类中实现该方法
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    # 抛出未实现错误，提示需要在子类中实现该方法
    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    # 初始化权重的方法
    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # 如果需要修剪头部
        if self.config.pruned_heads:
            # 调用修剪头部的方法
            self.prune_heads(self.config.pruned_heads)

        # 如果存在初始化权重的方法
        if _init_weights:
            # 初始化权重
            self.apply(self._initialize_weights)

            # 当不初始化所有权重时，应跳过绑定权重
            # 因为 from_pretrained(...) 方法会自动绑定权重
            self.tie_weights()

    # 修剪模型头部的方法
    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        # 保存新修剪头部的集合，作为之前存储的修剪头部和新修剪头部的并集
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # 不幸的是，我们必须将其存储为列表以便于 JSON

        # 调用基础模型的修剪头部方法
        self.base_model._prune_heads(heads_to_prune)
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        激活当前模型的梯度检查点。

        注意，在其他框架中，此功能可能称为“激活检查点”或“检查点激活”。

        我们传递模块的 `__call__` 方法而不是 `forward`，因为 `__call__` 会附加模块的所有钩子。
        https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                传递给 `torch.utils.checkpoint.checkpoint` 函数的额外关键字参数。
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} 不支持梯度检查点。")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # 对于旧的梯度检查点格式（transformers < 4.35.0）以及位于 Hub 上的模型，我们将退回到重写的 `_set_gradient_checkpointing` 方法
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            logger.warn(
                "您正在使用已弃用的检查点格式的旧版本（如果您传递了 `gradient_checkpointing_kwargs`，我们还将静默地忽略它）。"
                "请更新模型文件中的新格式。要使用新格式，您需要完全删除模型中 `_set_gradient_checkpointing` 方法的定义。"
            )

        if getattr(self, "_hf_peft_config_loaded", False):
            # 当使用 PEFT + 梯度检查点 + Trainer 时，我们需要确保输入的 requires_grad=True
            # 我们在 PEFT 中也这样做：https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # 在使用 PEFT 进行训练时，只有 LoRA 层的 requires_grad 设置为 True，但是冻结层的输出需要传播梯度，以确保梯度流动。
            self.enable_input_require_grads()
    # 设置梯度检查点功能，可以选择是否启用，设置梯度检查点函数
    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        # 初始化梯度检查点是否设置的标志
        is_gradient_checkpointing_set = False

        # 如果模型具有"gradient_checkpointing"属性，则在顶层模块上应用梯度检查点
        # 例如，LongT5Stack继承自`PreTrainedModel`
        if hasattr(self, "gradient_checkpointing"):
            # 设置梯度检查点函数和是否启用梯度检查点
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        # 遍历模型的所有模块
        for module in self.modules():
            # 如果模块具有"gradient_checkpointing"属性
            if hasattr(module, "gradient_checkpointing"):
                # 设置梯度检查点函数和是否启用梯度检查点
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        # 如果没有设置梯度检查点，则引发值错误
        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    # 禁用梯度检查点功能
    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        # 如果模型支持梯度检查点
        if self.supports_gradient_checkpointing:
            # 对于旧的GC格式（transformers < 4.35.0）和位于Hub上的模型，我们将回退到重写的`_set_gradient_checkpointing`方法
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                logger.warn(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                self.apply(partial(self._set_gradient_checkpointing, value=False))

        # 如果存在"_hf_peft_config_loaded"属性
        if getattr(self, "_hf_peft_config_loaded", False):
            # 禁用输入要求梯度
            self.disable_input_require_grads()

    # 返回梯度检查点是否激活的属性
    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        # 返回模型中任何模块是否具有"gradient_checkpointing"属性且梯度检查点是否已启用
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())
    # 将模型保存到指定目录
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    @wraps(PushToHubMixin.push_to_hub)
    # 将模型推送到模型中心
    def push_to_hub(self, *args, **kwargs):
        # 获取模型标签
        tags = self.model_tags if self.model_tags is not None else []

        # 获取传入的标签参数
        tags_kwargs = kwargs.get("tags", [])
        if isinstance(tags_kwargs, str):
            tags_kwargs = [tags_kwargs]

        # 将传入的标签添加到模型标签列表中
        for tag in tags_kwargs:
            if tag not in tags:
                tags.append(tag)

        # 如果存在标签，则将标签参数更新为模型标签列表
        if tags:
            kwargs["tags"] = tags
        # 调用父类的推送方法，并传递参数
        return super().push_to_hub(*args, **kwargs)

    # 获取模型的内存占用
    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
        """
        # 计算模型参数的内存占用
        mem = sum([param.nelement() * param.element_size() for param in self.parameters()])
        # 如果指定返回缓冲区的大小，则计算缓冲区的内存占用并加到总内存中
        if return_buffers:
            mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
            mem = mem + mem_bufs
        # 返回模型的内存占用
        return mem

    # 将模型移动到 CUDA 设备上
    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        # 检查模型是否以8位为单位进行加载
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            # 如果模型以4位或8位为单位进行量化，则不支持将其移动到 CUDA 设备上，抛出错误
            raise ValueError(
                "Calling `cuda()` is not supported for `4-bit` or `8-bit` quantized models. Please use the model as it is, since the"
                " model has already been set to the correct devices and casted to the correct `dtype`."
            )
        else:
            # 否则，调用父类的 cuda() 方法，并传递参数
            return super().cuda(*args, **kwargs)

    # 将模型转换到指定设备上
    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        # 检查模型是否已经以8位加载
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            # 如果模型已经以4位或8位位和字节加载，则不支持`.to`方法
            raise ValueError(
                "`.to` is not supported for `4-bit` or `8-bit` bitsandbytes models. Please use the model as it is, since the"
                " model has already been set to the correct devices and casted to the correct `dtype`."
            )
        elif getattr(self, "quantization_method", None) == QuantizationMethod.GPTQ:
            # 对于 GPTQ 模型，阻止用户将模型转换为其他数据类型以限制不必要的行为
            # 正确的 API 应该是通过 `from_pretrained` 直接加载具有所需数据类型的模型
            dtype_present_in_args = False

            if "dtype" not in kwargs:
                for arg in args:
                    if isinstance(arg, torch.dtype):
                        dtype_present_in_args = True
                        break
            else:
                dtype_present_in_args = True

            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a GPTQ model in a new `dtype`. Make sure to load the model using `from_pretrained` using the desired"
                    " `dtype` by passing the correct `torch_dtype` argument."
                )
        # 将参数传递给父类的 `to` 方法
        return super().to(*args, **kwargs)

    def half(self, *args):
        # 检查模型是否已量化
        if getattr(self, "is_quantized", False):
            # 不支持对量化模型使用`.half()`方法
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            # 将参数传递给父类的 `half` 方法
            return super().half(*args)

    def float(self, *args):
        # 检查模型是否已量化
        if getattr(self, "is_quantized", False):
            # 不支持对量化模型使用`.float()`方法
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            # 将参数传递给父类的 `float` 方法
            return super().float(*args)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    @classmethod
    # 加载预训练模型的方法，用于从给定的状态字典中加载模型的参数
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        low_cpu_mem_usage=False,
        device_map=None,
        offload_folder=None,
        offload_state_dict=None,
        dtype=None,
        is_quantized=False,
        keep_in_fp32_modules=None,
    # 从模型中检索具有指定名称的模块
    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        # 将给定名称中的模块键提取出来，用于后续比较
        module_keys = {".".join(key.split(".")[:-1]) for key in names}

        # torch.nn.ParameterList 是一种特殊情况，其中两个参数关键字被附加到模块名称上，例如 bert.special_embeddings.0
        # 如果名称以数字结尾，将其切割为模块键
        module_keys = module_keys.union(
            {".".join(key.split(".")[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()}
        )

        # 存储检索到的模块
        retrieved_modules = []
        # 遍历模型的所有模块，找到匹配的模块并加入到检索结果中
        for name, module in self.named_modules():
            if remove_prefix:
                _prefix = f"{self.base_model_prefix}."
                # 如果需要移除前缀，则移除前缀后再比较
                name = name[len(_prefix) :] if name.startswith(_prefix) else name
            elif add_prefix:
                # 如果需要添加前缀，则将前缀添加到模块名称前
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            # 如果模块名称在模块键中，则将其加入到检索结果中
            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    # 静态方法，用于以低内存消耗加载预训练模型
    @staticmethod
    def _load_pretrained_model_low_mem(model, loaded_state_dict_keys, resolved_archive_file, start_prefix=""):
        """
        This is an experimental function that loads the model using ~1.x model size CPU memory

        Before you call it do:

        1. save which state_dict keys are available
        2. drop state_dict before model is created, since the latter takes 1x model size memory

        Here then we continue:

        3. switch to the meta device all params/buffers that are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it doesn't handle missing_keys, unexpected_keys, mismatched_keys. It can't handle deepspeed.
        """

        # 将模型参数移动到元设备，以准备从加载的状态字典中替换
        _move_model_to_meta(model, loaded_state_dict_keys, start_prefix)
        # 加载状态字典
        state_dict = load_state_dict(resolved_archive_file)
        # 将加载的状态字典中的参数替换到模型中
        error_msgs = _load_state_dict_into_meta_model(model, state_dict, loaded_state_dict_keys, start_prefix)
        return error_msgs

    # 类方法
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):
                The auto class to register this new model with.
        """
        # 如果 auto_class 不是字符串类型，则将其转换为类名字符串
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入 transformers.models.auto 模块
        import transformers.models.auto as auto_module

        # 检查 auto_module 中是否存在指定的 auto_class
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将 auto_class 赋值给当前类的 _auto_class 属性
        cls._auto_class = auto_class

    def to_bettertransformer(self) -> "PreTrainedModel":
        """
        Converts the model to use [PyTorch's native attention
        implementation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), integrated to
        Transformers through [Optimum library](https://huggingface.co/docs/optimum/bettertransformer/overview). Only a
        subset of all Transformers models are supported.

        PyTorch's attention fastpath allows to speed up inference through kernel fusions and the use of [nested
        tensors](https://pytorch.org/docs/stable/nested.html). Detailed benchmarks can be found in [this blog
        post](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2).

        Returns:
            [`PreTrainedModel`]: The model converted to BetterTransformer.
        """
        # 检查是否安装了 optimum 包
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        # 导入 optimum 版本信息
        from optimum.version import __version__ as optimum_version

        # 检查 optimum 版本是否符合要求
        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        # 导入 BetterTransformer 类
        from optimum.bettertransformer import BetterTransformer

        # 将当前模型转换为 BetterTransformer 模型
        return BetterTransformer.transform(self)
    def reverse_bettertransformer(self):
        """
        Reverts the transformation from [`~PreTrainedModel.to_bettertransformer`] so that the original modeling is
        used, for example in order to save the model.

        Returns:
            [`PreTrainedModel`]: The model converted back to the original modeling.
        """
        # 检查是否安装了 `optimum` 包，若未安装则引发 ImportError
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        # 导入 `optimum` 的版本信息
        from optimum.version import __version__ as optimum_version

        # 检查 `optimum` 版本是否满足要求，若不满足则引发 ImportError
        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        # 导入 `BetterTransformer` 类
        from optimum.bettertransformer import BetterTransformer

        # 使用 `BetterTransformer` 类中的 reverse 方法还原模型转换
        return BetterTransformer.reverse(self)

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """

        # 在 Torch FX 代理、Torch JIT 追踪模式或 TorchDynamo 编译模式下，跳过检查
        if is_torch_fx_proxy(input_ids) or torch.jit.is_tracing() or is_torchdynamo_compiling():
            return

        # 若提供了 attention_mask 参数，或者模型配置中的 pad_token_id 为 None，则不发出警告
        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # 仅检查第一个和最后一个输入 ID，以减少开销
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            # 构建警告信息字符串
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "
                "https://huggingface.co/docs/transformers/troubleshooting"
                "#incorrect-output-when-padding-tokens-arent-masked."
            )

            # 若 pad_token_id 等于 BOS、EOS 或 SEP 中的任意一个，提示用户忽略警告
            if (
                (self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id)
                or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id)
                or (self.config.sep_token_id is not None and self.config.sep_token_id == self.config.pad_token_id)
            ):
                warn_string += (
                    f"\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical "
                    f"to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), "
                    f"or the `sep_token_id` ({self.config.sep_token_id}), and your input is not padded."
                )

            # 发出一次性警告
            logger.warning_once(warn_string)
```  
# 将 PreTrainedModel 类的 push_to_hub 方法复制一份，并重新赋值给自身，以避免修改原始方法
PreTrainedModel.push_to_hub = copy_func(PreTrainedModel.push_to_hub)
# 如果 push_to_hub 方法有文档字符串
if PreTrainedModel.push_to_hub.__doc__ is not None:
    # 格式化 push_to_hub 方法的文档字符串，替换文档字符串中的占位符为实际内容
    PreTrainedModel.push_to_hub.__doc__ = PreTrainedModel.push_to_hub.__doc__.format(
        object="model", object_class="AutoModel", object_files="model file"
    )


class PoolerStartLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # 初始化一个全连接层，输入尺寸为模型的 hidden_size，输出尺寸为1
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(
        self, hidden_states: torch.FloatTensor, p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            `torch.FloatTensor`: The start logits for SQuAD.
        """
        # 将隐藏状态通过全连接层得到 logits，然后压缩最后一个维度
        x = self.dense(hidden_states).squeeze(-1)

        # 如果存在 p_mask
        if p_mask is not None:
            # 如果模型参数类型是 float16
            if get_parameter_dtype(self) == torch.float16:
                # 使用 p_mask 来处理 logits，1-p_mask 表示未被掩盖的部分，-65500 * p_mask 用于将被掩盖的部分设置为一个较小的值
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                # 使用 p_mask 来处理 logits，1-p_mask 表示未被掩盖的部分，-1e30 * p_mask 用于将被掩盖的部分设置为一个较小的值
                x = x * (1 - p_mask) - 1e30 * p_mask

        # 返回处理后的 logits
        return x


class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # 初始化两个全连接层和一个 LayerNorm 层，用于生成 end logits
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `torch.FloatTensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        # Concatenate hidden_states and start_states along the last dimension
        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        # Apply activation function
        x = self.activation(x)
        # Apply Layer Normalization
        x = self.LayerNorm(x)
        # Apply another dense layer and squeeze the last dimension
        x = self.dense_1(x).squeeze(-1)

        # Apply masking if p_mask is provided
        if p_mask is not None:
            # Check the data type of the model parameters
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x
# 定义一个类，用于计算 SQuAD 2.0 的答案类别，基于分类和起始标记的隐藏状态
class PoolerAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入维度为 config.hidden_size * 2，输出维度为 config.hidden_size
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        # 创建一个线性层，输入维度为 config.hidden_size，输出维度为 1，无偏置
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `torch.FloatTensor`: The SQuAD 2.0 answer class.
        """
        # 获取隐藏状态的最后一个维度大小
        hsz = hidden_states.shape[-1]
        # 检查 start_states 和 start_positions 中至少有一个不为 None
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        # 将 start_states 和 cls_token_state 连接后输入到第一个线性层
        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x


@dataclass
class SquadHeadOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a [`~modeling_utils.SQuADHead`].
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.

    """

    # 定义可选的变量，用于存储损失、起始token的log概率、起始token的索引、结束token的log概率、结束token的索引、答案的"is_impossible"标签的log概率
    loss: Optional[torch.FloatTensor] = None
    start_top_log_probs: Optional[torch.FloatTensor] = None
    start_top_index: Optional[torch.LongTensor] = None
    end_top_log_probs: Optional[torch.FloatTensor] = None
    end_top_index: Optional[torch.LongTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
class SQuADHead(nn.Module):
    r"""
    A SQuAD head inspired by XLNet.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config):
        super().__init__()
        # 初始化开始位置的 top K 值
        self.start_n_top = config.start_n_top
        # 初始化结束位置的 top K 值
        self.end_n_top = config.end_n_top

        # 创建用于预测起始位置的池化器
        self.start_logits = PoolerStartLogits(config)
        # 创建用于预测结束位置的池化器
        self.end_logits = PoolerEndLogits(config)
        # 创建用于预测答案类别的池化器
        self.answer_class = PoolerAnswerClass(config)

    # 前向传播函数
    @replace_return_docstrings(output_type=SquadHeadOutput, config_class=PretrainedConfig)
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
        is_impossible: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
        return_dict: bool = False,


        # SequenceSummary 类定义开始
class SequenceSummary(nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """
    # 初始化函数，接受一个预训练配置对象作为参数
    def __init__(self, config: PretrainedConfig):
        # 调用父类的初始化函数
        super().__init__()

        # 获取配置对象中的摘要类型，默认为"last"
        self.summary_type = getattr(config, "summary_type", "last")
        # 如果摘要类型为"attn"
        if self.summary_type == "attn":
            # 抛出未实现的错误
            raise NotImplementedError

        # 初始化摘要为一个恒等映射
        self.summary = Identity()
        # 如果配置对象中有"summary_use_proj"属性且为True
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            # 如果配置对象中有"summary_proj_to_labels"属性且为True，并且num_labels大于0
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            # 将摘要映射为一个线性层
            self.summary = nn.Linear(config.hidden_size, num_classes)

        # 获取配置对象中的摘要激活函数，默认为None
        activation_string = getattr(config, "summary_activation", None)
        # 根据激活函数字符串获取对应的激活函数，如果没有则使用恒等映射
        self.activation: Callable = get_activation(activation_string) if activation_string else Identity()

        # 初始化第一个dropout为恒等映射
        self.first_dropout = Identity()
        # 如果配置对象中有"summary_first_dropout"属性且大于0
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            # 将第一个dropout设置为一个Dropout层
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        # 初始化最后一个dropout为恒等映射
        self.last_dropout = Identity()
        # 如果配置对象中有"summary_last_dropout"属性且大于0
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            # 将最后一个dropout设置为一个Dropout层
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    # 前向传播函数，接受隐藏状态和类别索引作为参数
    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `torch.FloatTensor`: The summary of the sequence hidden states.
        """
        # 根据不同的summary_type计算序列隐藏状态的摘要
        if self.summary_type == "last":
            output = hidden_states[:, -1]  # 取最后一个隐藏状态作为摘要
        elif self.summary_type == "first":
            output = hidden_states[:, 0]  # 取第一个隐藏状态作为摘要
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)  # 计算隐藏状态的平均值作为摘要
        elif self.summary_type == "cls_index":
            if cls_index is None:
                # 如果cls_index为空，则使用最后一个token作为分类token
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # cls_index的形状为(bsz, XX, 1, hidden_size)，其中XX是hidden_states的可选前导维度
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # 获取指定位置的隐藏状态作为摘要，形状为(bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        # 对输出进行首次丢弃
        output = self.first_dropout(output)
        # 对输出进行摘要处理
        output = self.summary(output)
        # 对输出进行激活函数处理
        output = self.activation(output)
        # 对输出进行最后一次丢弃
        output = self.last_dropout(output)

        return output
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    递归地从潜在的容器中解包模型（如在分布式训练中使用）。

    Args:
        model (`torch.nn.Module`): 要解包的模型。
    """
    # 如果模型具有 `module` 属性，则递归解包
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def expand_device_map(device_map, param_names, start_prefix):
    """
    扩展设备映射以返回参数名称与设备的对应关系。
    """
    new_device_map = {}
    # 提取以指定前缀开头的参数名称，并移除前缀
    param_names = [p[len(start_prefix) :] for p in param_names if p.startswith(start_prefix)]
    for module, device in device_map.items():
        # 更新设备映射，如果参数名称与模块名称匹配，或者参数名称以模块名称加点开头，或者模块名称为空
        new_device_map.update(
            {p: device for p in param_names if p == module or p.startswith(f"{module}.") or module == ""}
        )
    return new_device_map


def get_disk_only_shard_files(device_map, sharded_metadata, start_prefix):
    """
    返回仅包含已转移到磁盘的权重的分片文件列表。
    """

    # 提取以指定前缀开头的参数名称与文件名的映射，并移除前缀
    weight_map = {
        p[len(start_prefix) :]: v for p, v in sharded_metadata["weight_map"].items() if p.startswith(start_prefix)
    }
    files_content = collections.defaultdict(list)
    for weight_name, filename in weight_map.items():
        # 将权重名称逐渐缩减，直到找到与设备映射匹配的名称
        while len(weight_name) > 0 and weight_name not in device_map:
            weight_name = ".".join(weight_name.split(".")[:-1])
        files_content[filename].append(device_map[weight_name])

    # 返回只包含磁盘设备的分片文件列表
    return [fname for fname, devices in files_content.items() if set(devices) == {"disk"}]
```