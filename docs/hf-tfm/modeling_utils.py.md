# `.\modeling_utils.py`

```
# 导入 Python 内置和第三方库
import collections  # 导入 collections 模块，用于扩展内置容器数据类型
import copy  # 导入 copy 模块，用于对象复制操作
import functools  # 导入 functools 模块，用于高阶函数操作
import gc  # 导入 gc 模块，Python 的垃圾回收模块
import importlib.metadata  # 导入 importlib.metadata 模块，用于元数据获取
import inspect  # 导入 inspect 模块，用于解析源码
import itertools  # 导入 itertools 模块，用于创建和操作迭代器的函数
import json  # 导入 json 模块，用于 JSON 数据的编解码
import os  # 导入 os 模块，用于与操作系统交互
import re  # 导入 re 模块，用于正则表达式操作
import shutil  # 导入 shutil 模块，用于文件操作的高级函数
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import warnings  # 导入 warnings 模块，用于警告控制

# 导入 typing 模块中的类型
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入第三方库 torch
import torch  # 导入 PyTorch 深度学习库
from packaging import version  # 从 packaging 模块导入 version 子模块
from torch import Tensor, nn  # 从 torch 模块导入 Tensor 和 nn（神经网络）子模块
from torch.nn import CrossEntropyLoss, Identity  # 从 torch.nn 模块导入 CrossEntropyLoss 和 Identity 类
from torch.utils.checkpoint import checkpoint  # 从 torch.utils.checkpoint 模块导入 checkpoint 函数

# 导入本地的模块和函数
from .activations import get_activation  # 从当前目录的 activiations 模块导入 get_activation 函数
from .configuration_utils import PretrainedConfig  # 从当前目录的 configuration_utils 模块导入 PretrainedConfig 类
from .dynamic_module_utils import custom_object_save  # 从当前目录的 dynamic_module_utils 模块导入 custom_object_save 函数
from .generation import GenerationConfig, GenerationMixin  # 从当前目录的 generation 模块导入 GenerationConfig 和 GenerationMixin 类
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled  # 从当前目录的 integrations 模块导入若干函数和类
from .pytorch_utils import (  # 从当前目录的 pytorch_utils 模块导入若干函数和类，忽略 F401 错误
    Conv1D,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    id_tensor_storage,
    is_torch_greater_or_equal_than_1_13,
    prune_conv1d_layer,
    prune_layer,
    prune_linear_layer,
)
from .quantizers import AutoHfQuantizer, HfQuantizer  # 从当前目录的 quantizers 模块导入 AutoHfQuantizer 和 HfQuantizer 类
from .quantizers.quantizers_utils import get_module_from_name  # 从当前目录的 quantizers.quantizers_utils 模块导入 get_module_from_name 函数
from .safetensors_conversion import auto_conversion  # 从当前目录的 safetensors_conversion 模块导入 auto_conversion 函数
from .utils import (  # 从当前目录的 utils 模块导入若干函数、常量和类
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    ModelOutput,
    PushToHubMixin,
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    has_file,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_flash_attn_2_available,
    is_offline_mode,
    is_optimum_available,
    is_peft_available,
    is_remote_url,
    is_safetensors_available,
    is_torch_sdpa_available,
    is_torch_xla_available,
    logging,
    replace_return_docstrings,
    strtobool,
)
from .utils.hub import convert_file_size_to_int, create_and_tag_model_card, get_checkpoint_shard_files  # 从当前目录的 utils.hub 模块导入若干函数
from .utils.import_utils import (  # 从当前目录的 utils.import_utils 模块导入若干函数和常量
    ENV_VARS_TRUE_VALUES,
    is_sagemaker_mp_enabled,
    is_torch_fx_proxy,
)
    is_torchdynamo_compiling,
# 导入所需模块和变量
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
# 设置环境变量 XLA_USE_BF16，指定默认值为 "0" 并转换为大写
XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
# 设置环境变量 XLA_DOWNCAST_BF16，指定默认值为 "0" 并转换为大写
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()

# 如果加速库可用
if is_accelerate_available():
    # 导入加速库相关模块和函数
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    from accelerate.hooks import add_hook_to_module
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

# 如果 SafeTensors 库可用
if is_safetensors_available():
    # 导入 SafeTensors 库相关函数
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file

# 获取日志记录器
logger = logging.get_logger(__name__)

# 初始化权重标记
_init_weights = True

# 检查是否启用了 FSDP（Fully Sharded Data Parallelism）
def is_fsdp_enabled():
    return (
        torch.distributed.is_available()  # 检查是否支持分布式训练
        and torch.distributed.is_initialized()  # 检查是否已初始化分布式环境
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1  # 检查是否启用 FSDP
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1  # 检查是否启用 FSDP CPU 和 RAM 的高效加载
    )

# 检查当前进程是否是本地分布式训练的主进程（rank 0）
def is_local_dist_rank_0():
    return (
        torch.distributed.is_available()  # 检查是否支持分布式训练
        and torch.distributed.is_initialized()  # 检查是否已初始化分布式环境
        and int(os.environ.get("LOCAL_RANK", -1)) == 0  # 检查本地进程的分布式训练排名是否为 0
    )

# 如果 SageMaker Model Parallelism 可用
if is_sagemaker_mp_enabled():
    # 导入 SageMaker Model Parallelism 相关模块和函数
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    # 检查是否为 SageMaker MP 1.10 版本之后
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

# 如果 PEFT 可用
if is_peft_available():
    # 从 utils 模块中导入 find_adapter_config_file 函数
    from .utils import find_adapter_config_file

# 定义 Torch 初始化函数字典
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

# 上下文管理器，用于全局禁用模型初始化权重以加快大模型加载速度
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

        # 临时替换 Torch 初始化函数为 _skip_init 函数
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        # 恢复原始的初始化权重函数
        _init_weights = old_init_weights
        if _enable:
            # 如果启用了初始化函数替换
            # 遍历 TORCH_INIT_FUNCTIONS 字典中的每一项
            for name, init_func in TORCH_INIT_FUNCTIONS.items():
                # 将 torch.nn.init 中的初始化函数名 name 恢复为原始函数 init_func
                setattr(torch.nn.init, name, init_func)
def get_parameter_device(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    try:
        # 尝试获取参数的第一个参数并返回其设备信息
        return next(parameter.parameters()).device
    except StopIteration:
        # 对于 nn.DataParallel 在 PyTorch 1.5 及以上版本的兼容性处理

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        # 从参数中获取命名成员的生成器
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        # 获取第一个生成器产生的元组，并返回其设备信息
        first_tuple = next(gen)
        return first_tuple[1].device


def get_first_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    """
    Returns the first parameter dtype (can be non-floating) or asserts if none were found.
    """
    try:
        # 尝试获取参数的第一个参数并返回其数据类型
        return next(parameter.parameters()).dtype
    except StopIteration:
        # 对于 nn.DataParallel 在 PyTorch 大于 1.5 版本的兼容性处理

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        # 从参数中获取命名成员的生成器
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        # 获取第一个生成器产生的元组，并返回其数据类型
        first_tuple = next(gen)
        return first_tuple[1].dtype


def get_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    # 遍历参数的所有参数
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            # 添加修复 https://github.com/pytorch/xla/issues/4152
            # 修复模型代码传递的数值超出 XLA_USE_BF16=1 和 XLA_DOWNCAST_BF16=1 的范围，导致转换为 -inf 的问题
            # 注意: `is_torch_xla_available()` 是最后检查的，因为它会在 torch dynamo 中引入图形断裂
            if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                return torch.bfloat16
            if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                if t.dtype == torch.float:
                    return torch.bfloat16
                if t.dtype == torch.double:
                    return torch.float32
            return t.dtype

    # 如果找不到浮点数据类型，则返回最后一个找到的数据类型
    if last_dtype is not None:
        return last_dtype

    # 对于 nn.DataParallel 在 PyTorch 大于 1.5 版本的兼容性处理
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    # 从参数中获取命名成员的生成器
    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    # 遍历生成器中的元组
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype
    # 如果 last_tuple 不是 None，则返回 last_tuple 中第二个元素的数据类型作为结果
    if last_tuple is not None:
        return last_tuple[1].dtype
    
    # 如果 last_tuple 是 None，则尝试使用 parameter 中的缓冲区的数据类型作为结果
    for t in parameter.buffers():
        # 记录每次迭代中 t 的数据类型到 last_dtype
        last_dtype = t.dtype
        # 如果 t 是浮点数类型，则返回 t 的数据类型作为结果
        if t.is_floating_point():
            return t.dtype
    
    # 如果所有缓冲区都不是浮点数类型，则返回最后一次迭代中记录的数据类型作为结果
    return last_dtype
# 返回 `state_dict` 中第一个浮点数据类型，如果没有则抛出异常
def get_state_dict_float_dtype(state_dict):
    for t in state_dict.values():  # 遍历 `state_dict` 中的每个值
        if t.is_floating_point():  # 检查当前值是否为浮点数类型
            return t.dtype  # 返回该值的数据类型

    raise ValueError("couldn't find any floating point dtypes in state_dict")  # 如果没有找到浮点数据类型则抛出异常


# 返回 `state_dict` 中第一个浮点数据类型，如果没有则返回第一个数据类型
def get_state_dict_dtype(state_dict):
    for t in state_dict.values():  # 遍历 `state_dict` 中的每个值
        if t.is_floating_point():  # 检查当前值是否为浮点数类型
            return t.dtype  # 返回该值的数据类型

    # 如果没有找到浮点数据类型，则返回 `state_dict` 中第一个值的数据类型
    else:
        return next(state_dict.values()).dtype


# 返回指定数据类型 `dtype` 的参数占据的字节数
def dtype_byte_size(dtype):
    if dtype == torch.bool:  # 如果数据类型是布尔类型
        return 1 / 8  # 返回布尔类型参数占据的字节数
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))  # 从数据类型字符串中搜索位数信息
    if bit_search is None:  # 如果未找到有效的数据类型
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")  # 抛出数据类型无效的异常
    bit_size = int(bit_search.groups()[0])  # 提取数据类型的位数
    return bit_size // 8  # 返回数据类型参数占据的字节数


# 将模型状态字典 `state_dict` 分割为多个子检查点，使每个子检查点的最终大小不超过指定大小
def shard_checkpoint(
    state_dict: Dict[str, torch.Tensor], max_shard_size: Union[int, str] = "10GB", weights_name: str = WEIGHTS_NAME
):
    max_shard_size = convert_file_size_to_int(max_shard_size)  # 将最大分片大小转换为整数形式

    sharded_state_dicts = [{}]  # 初始化一个空的分片状态字典列表
    last_block_size = 0  # 初始化最后一个分片的大小
    total_size = 0  # 初始化总大小
    storage_id_to_block = {}  # 初始化存储 ID 到分片索引的映射表
    # 遍历状态字典中的每个键值对，其中键为参数名，值为参数的权重
    for key, weight in state_dict.items():
        # 如果权重是字符串类型，跳过当前循环，因为在序列化时使用了 BNB，可能出现这种情况
        # 可参考：https://github.com/huggingface/transformers/pull/24416 获取更多细节
        if isinstance(weight, str):
            continue
        else:
            # 获取权重张量的存储 ID
            storage_id = id_tensor_storage(weight)

        # 如果某个权重共享相同的底层存储，则将该权重放入相同的“块”中
        if storage_id in storage_id_to_block:
            block_id = storage_id_to_block[storage_id]
            sharded_state_dicts[block_id][key] = weight
            continue

        # 计算当前权重的字节大小
        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # 如果当前块的总大小加上当前权重的大小超过了最大分片大小，并且当前块中至少有一个权重，
        # 则将当前块分片，创建一个新的空字典作为新的块，并重置当前块的大小
        if last_block_size + weight_size > max_shard_size and len(sharded_state_dicts[-1]) > 0:
            sharded_state_dicts.append({})
            last_block_size = 0

        # 将当前权重添加到当前块中
        sharded_state_dicts[-1][key] = weight
        # 更新当前块的总大小
        last_block_size += weight_size
        # 将当前权重的存储 ID 映射到对应的块索引
        storage_id_to_block[storage_id] = len(sharded_state_dicts) - 1

    # 如果只有一个分片，直接返回该分片
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # 否则，构建索引
    weight_map = {}
    shards = {}
    # 遍历所有分片，为每个分片创建一个文件名，并将分片及其对应的键添加到 shards 和 weight_map 中
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # 添加元数据
    metadata = {"total_size": total_size}
    # 构建索引结构，包括元数据和权重映射
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index
# 加载分片检查点的函数，用于从文件夹中加载模型的状态字典
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
    # 拼接索引文件的路径
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    # 拼接安全索引文件的路径
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    # 检查索引文件和安全索引文件是否存在
    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    # 如果既没有索引文件也没有安全索引文件，则抛出错误
    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    # 根据 prefer_safe 的设置确定加载哪种索引文件
    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # 根据偏好加载安全索引文件
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True  # 因为没有其他选择，所以加载安全索引文件

    load_index = safe_index_file if load_safe else index_file

    # 使用 utf-8 编码打开加载索引文件，并解析为 JSON 格式
    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    # 获取所有分片文件的路径
    shard_files = list(set(index["weight_map"].values()))

    # 如果 strict=True，则在加载任何状态字典之前检查错误
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()

    # 查找模型中缺失的键和索引中未预料到的键
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    # 如果 strict 为 True 并且存在缺失的键或者不期望的键
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        # 构建错误信息，指明加载 state_dict 时出错的模型类名
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        
        # 如果存在缺失的键
        if len(missing_keys) > 0:
            # 构建缺失键的字符串表示，用逗号分隔
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        
        # 如果存在不期望的键
        if len(unexpected_keys) > 0:
            # 构建不期望键的字符串表示，用逗号分隔
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        
        # 抛出运行时异常，显示错误信息
        raise RuntimeError(error_message)

    # 根据 torch 版本创建用于加载文件的 loader 函数
    weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
    loader = safe_load_file if load_safe else partial(torch.load, map_location="cpu", **weights_only_kwarg)

    # 遍历每个分片文件
    for shard_file in shard_files:
        # 使用 loader 加载分片文件的 state_dict
        state_dict = loader(os.path.join(folder, shard_file))
        
        # 将加载的 state_dict 应用到模型中，strict 设置为 False
        model.load_state_dict(state_dict, strict=False)

        # 在加载下一个 state_dict 之前确保释放内存
        del state_dict
        gc.collect()

    # 返回与 PyTorch load_state_dict 函数相同的对象，用于处理不兼容键
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
def load_state_dict(checkpoint_file: Union[str, os.PathLike], is_quantized: bool = False):
    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """

    # 如果检查点文件以 ".safetensors" 结尾且安全张量可用
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # 检查归档格式
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
        # 如果归档中的元数据格式不在有效列表 ["pt", "tf", "flax", "mlx"] 中，则抛出异常
        if metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        # 加载安全张量文件
        return safe_load_file(checkpoint_file)

    try:
        # 处理特定条件下的 `map_location`
        if (
            (is_deepspeed_zero3_enabled() and torch.distributed.is_initialized() and torch.distributed.get_rank() > 0)
            or (is_fsdp_enabled() and not is_local_dist_rank_0())
        ) and not is_quantized:
            map_location = "meta"
        else:
            map_location = "cpu"

        extra_args = {}
        # 如果 `checkpoint_file` 是字符串，并且不是 `meta` `map_location`，且 PyTorch 版本 >= 2.1.0，并且是 Zip 格式文件，则启用 `mmap`
        if (
            isinstance(checkpoint_file, str)
            and map_location != "meta"
            and version.parse(torch.__version__) >= version.parse("2.1.0")
            and is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}

        weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
        
        # 使用 PyTorch 加载检查点文件
        return torch.load(
            checkpoint_file,
            map_location=map_location,
            **weights_only_kwarg,
            **extra_args,
        )

    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                # 检查文件是否以 "version" 开头，如果是，则可能是未安装 git-lfs 的情况
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
            # 如果无法读取文件内容，抛出加载异常
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
    # 创建一个空字典，用于存储未初始化的子模块
    not_initialized_submodules = {}
    # 遍历模型中所有命名的模块及其对应的名称
    for module_name, module in model.named_modules():
        # 从状态字典键集合中提取加载的键集合，去除模块名称前缀
        loaded_keys = {k.replace(f"{module_name}.", "") for k in state_dict_keys if k.startswith(f"{module_name}.")}
        # 检查加载的键集合是否完全包含模块的状态字典的所有键
        if loaded_keys.issuperset(module.state_dict()):
            # 如果是，则标记模块为已由Hugging Face初始化
            module._is_hf_initialized = True
        else:
            # 否则将未初始化的模块添加到未初始化子模块字典中
            not_initialized_submodules[module_name] = module
    # 返回所有未初始化的子模块字典
    return not_initialized_submodules
# 将给定的模型加载状态字典到模型中，修改旧格式为新格式（如果需要）
def _load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    # 查找所有含有特定关键词的键，将其转换为新的键名
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    # 替换旧键为新键
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # 复制状态字典以便 _load_from_state_dict 可以修改它
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch 的 `_load_from_state_dict` 不会复制模块子类中的参数，
    # 所以需要递归应用该函数
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # 模块及其子模块的参数将以给定的前缀开头，如果在状态字典中不存在这些参数，则可以提前退出
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # 在分片模型中，每个分片只有部分完整状态字典，因此只收集当前状态字典中存在的参数
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                if len(params_to_gather) > 0:
                    # 因为 zero3 在模型参数中放置占位符，所以这个上下文管理器会收集（取消分片）当前层的参数，
                    # 然后从状态字典中加载，再重新分片
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        # 递归加载子模块的参数
        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    # 开始递归加载模型
    load(model_to_load, state_dict, prefix=start_prefix)
    # 删除 `state_dict`，以便更早地由 GC 回收。注意 `state_dict` 是参数的副本，因此可以安全删除它。
    del state_dict

    return error_msgs
    # 辅助函数：查找最后一个子模块及其参数/缓冲区名称。如果提供了 `start_prefix`，则将其从键的开头移除。
    if len(start_prefix) > 0 and long_key.startswith(start_prefix):
        # 如果 `start_prefix` 长度大于零且 `long_key` 以 `start_prefix` 开头，则移除 `start_prefix`
        long_key = ".".join(long_key.split(".")[1:])
    
    # 按照点号分割长键名
    split_key = long_key.split(".")
    # 从模型开始查找子模块
    submodule = model
    while len(split_key) > 1:
        if hasattr(submodule, split_key[0]):
            # 如果模块具有当前分割键名对应的属性，则获取该属性作为下一级子模块
            submodule = getattr(submodule, split_key[0])
            # 删除已处理的键名
            del split_key[0]
        else:
            # 如果模块不具有当前分割键名对应的属性，则子模块置为 None，跳出循环
            submodule = None
            break
    
    # 如果最终找到的子模块仍然是初始的模型，说明未找到匹配的子模块
    if submodule == model:
        submodule = None
    # 返回最后找到的子模块及剩余的键名
    return submodule, split_key[0]
    # 将 `loaded_state_dict_keys` 中的参数移到模型的元设备上，从而释放这些参数占用的内存空间。
    # `start_prefix` 用于包含模型名称的模型键，例如在 `bert.pooler.dense.weight` 中的 `bert`。

    # 初始化错误信息列表
    error_msgs = []

    # 初始化旧键和新键列表，用于处理特定的参数重命名情况
    old_keys = []
    new_keys = []

    # 检查是否进行了量化操作
    is_quantized = hf_quantizer is not None

    # 遍历 `state_dict` 中的所有键
    for key in state_dict.keys():
        new_key = None

        # 替换特定键名中的 "gamma" 为 "weight"
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")

        # 替换特定键名中的 "beta" 为 "bias"
        if "beta" in key:
            new_key = key.replace("beta", "bias")

        # 如果有新的键名生成，则将原键名添加到旧键列表，将新键名添加到新键列表
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    # 遍历两个列表 old_keys 和 new_keys，依次将 state_dict 中 old_key 对应的值替换为 new_key，并更新 state_dict。
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    
    # 返回三个变量作为结果：error_msgs（错误消息列表）、offload_index（卸载索引）、state_dict_index（状态字典索引）。
    return error_msgs, offload_index, state_dict_index
def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    # 如果 variant 参数不为 None，则修改 weights_name 中的文件扩展名
    if variant is not None:
        # 将 weights_name 按照 '.' 分割成列表
        splits = weights_name.split(".")
        # 替换列表中倒数第二项为 variant
        splits = splits[:-1] + [variant] + splits[-1:]
        # 将列表重新组合成字符串形式的 weights_name
        weights_name = ".".join(splits)

    # 返回修改后的 weights_name
    return weights_name


class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            # 如果导入 psutil 失败，则抛出 ImportError
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        # 获取当前进程的 psutil.Process 对象
        process = psutil.Process(os.getpid())
        # 获取当前进程的内存信息
        mem = process.memory_info()
        # 将当前进程的内存占用 RSS 存储到 module 对象的 mem_rss_pre_forward 属性中
        module.mem_rss_pre_forward = mem.rss
        # 返回 None
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            # 如果导入 psutil 失败，则抛出 ImportError
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        # 获取当前进程的 psutil.Process 对象
        process = psutil.Process(os.getpid())
        # 获取当前进程的内存信息
        mem = process.memory_info()
        # 将当前进程的内存占用 RSS 存储到 module 对象的 mem_rss_post_forward 属性中
        module.mem_rss_post_forward = mem.rss
        # 计算前后两次内存占用的差值
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        # 将差值累加到 module 对象的 mem_rss_diff 属性中
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        # 返回 None
        return None

    def add_memory_hooks(self):
        """
        Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

        Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero
        with `model.reset_memory_hooks_state()`.
        """
        # 遍历当前对象的所有子模块
        for module in self.modules():
            # 注册前向传播前的钩子函数 _hook_rss_memory_pre_forward
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            # 注册前向传播后的钩子函数 _hook_rss_memory_post_forward
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        # 调用 reset_memory_hooks_state 方法，重置所有模块的内存钩子状态
        self.reset_memory_hooks_state()

    def reset_memory_hooks_state(self):
        """
        Reset the `mem_rss_diff` attribute of each module (see [`~modeling_utils.ModuleUtilsMixin.add_memory_hooks`]).
        """
        # 遍历当前对象的所有子模块
        for module in self.modules():
            # 将每个模块的 mem_rss_diff、mem_rss_post_forward 和 mem_rss_pre_forward 属性重置为 0
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # 调用 get_parameter_device 函数获取当前模块所在的设备，并返回设备对象
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        # 调用 get_parameter_dtype 函数获取当前模块的数据类型，并返回数据类型对象
        return get_parameter_dtype(self)
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        # 如果注意力遮罩是三维的，则在第二个维度上扩展为四维
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        # 如果注意力遮罩是二维的，则在第二个和第三个维度上扩展为四维
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        
        # T5有一个可以比较序列ID的遮罩，这里通过转置来模拟
        # 参考：https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # 将注意力遮罩转换为模型数据类型，以支持fp16（半精度浮点数）计算
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
        # 计算反转的注意力遮罩，将0变为最小的负浮点数
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask

    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
        else:
            device = attention_mask.device
        
        batch_size, seq_length = input_shape
        # 创建一个序列ID张量，长度为seq_length，设备为指定的设备
        seq_ids = torch.arange(seq_length, device=device)
        # 创建一个因果遮罩，用于decoder，形状为[batch_size, seq_length, seq_length]
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # 将因果遮罩转换为与注意力遮罩相同的数据类型
        causal_mask = causal_mask.to(attention_mask.dtype)

        # 如果因果遮罩的长度小于注意力遮罩的长度，则需要在因果遮罩前添加一个全1的遮罩
        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        # 创建扩展的注意力遮罩，是因果遮罩和输入的注意力遮罩的点积
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ):
        # 略过此方法的注释，因为未提供代码块
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype  # 如果未指定 dtype，则使用对象自身的 dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # 如果 attention_mask 的维度不是二维或模型不是解码器，发出警告
            # 仅在不在 `create_extended_attention_mask_for_decoder` 中显示时才显示此警告
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        
        # 如果 attention_mask 的维度是三维，则扩展为 [batch_size, 1, from_seq_length, to_seq_length]
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # 如果提供了维度为 [batch_size, seq_length] 的填充 mask
            # - 如果模型是解码器，则除了填充 mask 外还应用因果 mask
            # - 如果模型是编码器，则将 mask 扩展为 [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            # 如果 attention_mask 维度不符合要求，抛出 ValueError
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # 将 extended_attention_mask 转换为指定的 dtype，用于 fp16 兼容性
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        # 将所有值为 1.0 的位置变为 0.0，所有值为 0.0 的位置变为 dtype 的最小值
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask
    def prepare_head_mask(self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attention scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            # Convert head_mask to a 5-dimensional tensor if it's 1-dimensional
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            # Modify head_mask shape if attention scores are computed by chunks
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            # Set head_mask to a list of None for each layer if head_mask is None
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """
        Convert `head_mask` to a 5-dimensional tensor `[num_hidden_layers x batch x num_heads x seq_length x seq_length]`.

        Args:
            head_mask (`torch.Tensor`):
                The input head_mask tensor with shape `[num_heads]` or `[num_hidden_layers x num_heads]`.
            num_hidden_layers (`int`):
                The number of hidden layers in the model.

        Returns:
            `torch.Tensor`:
                The converted head_mask tensor with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]`.
        """
        if head_mask.dim() == 1:
            # Expand the head_mask tensor to match the desired shape
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            # Expand the head_mask tensor to include each layer if it's 2-dimensional
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # Convert to specified dtype for compatibility
        return head_mask
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

        # Check if embeddings should be excluded from the parameter count
        if exclude_embeddings:
            # Generate a list of parameter names that belong to embedding layers
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            # Filter out embedding parameters from the total parameters
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            # If not excluding embeddings, include all parameters of the module
            total_parameters = list(self.parameters())

        # Initialize an empty list to store the number of elements (numel) in each parameter tensor
        total_numel = []
        
        # Check if the model has been loaded in 4bit precision
        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)

        # If loaded in 4bit precision, additional considerations are needed
        if is_loaded_in_4bit:
            # Check if the bitsandbytes library is available
            if is_bitsandbytes_available():
                import bitsandbytes as bnb
            else:
                # Raise an error if bitsandbytes is not installed but 4bit precision is indicated
                raise ValueError(
                    "bitsandbytes is not installed but it seems that the model has been loaded in 4bit precision, something went wrong"
                    " make sure to install bitsandbytes with `pip install bitsandbytes`. You also need a GPU. "
                )

        # Iterate through each parameter to calculate the number of elements (numel)
        for param in total_parameters:
            # Check if the parameter requires gradient or if only trainable parameters are considered
            if param.requires_grad or not only_trainable:
                # For 4bit models, adjust the numel calculation due to storage considerations
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    total_numel.append(
                        param.numel() * 2 * self.hf_quantizer.quantization_config.bnb_4bit_quant_storage.itemsize
                    )
                else:
                    # Standard numel calculation for regular tensors
                    total_numel.append(param.numel())

        # Return the sum of all calculated numels, representing the total number of parameters
        return sum(total_numel)
    # Helper function to estimate the total number of tokens from the model inputs.
    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        # Initialize a dictionary to track warnings if not already initialized
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        
        # Check if the main input name exists in the input dictionary
        if self.main_input_name in input_dict:
            # Return the number of elements in the tensor corresponding to the main input
            return input_dict[self.main_input_name].numel()
        # If main input name does not exist, issue a warning
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            # Mark that a warning for 'estimate_tokens' has been issued
            self.warnings_issued["estimate_tokens"] = True
        
        # Return 0 if unable to estimate tokens
        return 0

    # Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
    # batch with this transformer model.
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

        # Calculate the number of floating-point operations based on an approximation
        # 6 operations per token times the estimated number of tokens times the number of model parameters
        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)
# 定义一个继承自多个Mixin类的模型基类，用于所有模型的基础功能实现
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

    # 配置类，派生类需覆盖
    config_class = None
    # 基础模型前缀，派生类需覆盖
    base_model_prefix = ""
    # 主要输入名称，默认为 `input_ids`
    main_input_name = "input_ids"
    # 模型标签，初始化为 None
    model_tags = None

    # 内部使用的属性，以下几个属性初始化为 None
    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None

    # 用于加载时忽略的 `state_dict` 键的模式列表，初始化为 None
    _keys_to_ignore_on_load_missing = None
    # 用于加载时忽略的 `state_dict` 键的模式列表，初始化为 None
    _keys_to_ignore_on_load_unexpected = None
    # 用于保存模型时忽略的 `state_dict` 键的列表，初始化为 None
    _keys_to_ignore_on_save = None
    # 可能与另一个键绑定的 `state_dict` 键的列表，初始化为 None
    _tied_weights_keys = None

    # 是否支持模型并行化，默认为 False
    is_parallelizable = False
    # 是否支持梯度检查点，默认为 False
    supports_gradient_checkpointing = False

    # 是否支持 Flash Attention 2，默认为 False
    _supports_flash_attn_2 = False

    # 是否支持 SDPA，默认为 False
    _supports_sdpa = False

    # 是否支持将 `Cache` 实例用作 `past_key_values`，默认为 False
    _supports_cache_class = False

    @property
    def dummy_inputs(self) -> Dict[str, torch.Tensor]:
        """
        `Dict[str, torch.Tensor]`: 返回用于网络前向传播的虚拟输入数据字典。
        """
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}

    @property
    def framework(self) -> str:
        """
        :str: 标识这是一个基于 PyTorch 的模型。
        """
        return "pt"

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 保存配置和预训练权重的来源，如果在模型中给出的话
        config = self._autoset_attn_implementation(
            config, torch_dtype=torch.get_default_dtype(), check_device_map=False
        )
        self.config = config

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        # 如果模型支持生成，将生成配置从模型配置中创建
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        # 重写类属性以将其变为实例属性，这样像 `InstructBlipForConditionalGeneration` 这样的模型可以动态更新它，
        # 而不需要修改类属性，当使用不同的组件（例如语言模型）时。
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)

    def post_init(self):
        """
        在每次 Transformer 模型初始化结束时执行的方法，用于执行需要模型模块正确初始化的代码（例如权重初始化）。
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # 现在已经使用了该属性，从配置中删除它，这样它就不会被保存在配置中。
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

        model = AutoModel.from_pretrained("google-bert/bert-base-cased")

        model.add_model_tags(["custom", "custom-bert"])

        # Push the model to your namespace with the name "my-custom-bert".
        model.push_to_hub("my-custom-bert")
        ```
        """
        if isinstance(tags, str):
            tags = [tags]  # 如果tags是字符串，转换为单元素列表

        if self.model_tags is None:
            self.model_tags = []  # 如果当前模型标签为空，初始化为空列表

        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)  # 添加不重复的标签到模型标签列表

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        torch_dtype = kwargs.pop("torch_dtype", None)
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)  # 如果指定了torch_dtype，则设置默认dtype为指定的dtype

        config = copy.deepcopy(config)  # 创建配置的深拷贝，避免在_from_config中直接修改原始配置
        config._attn_implementation = kwargs.pop("attn_implementation", None)  # 设置配置中的注意力实现方式

        config = cls._autoset_attn_implementation(
            config,
            use_flash_attention_2=use_flash_attention_2,
            check_device_map=False,
            torch_dtype=torch_dtype,
        )  # 调用自动设置注意力实现的方法，根据参数设置config的相关属性

        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                model = cls(config, **kwargs)  # 在DeepSpeed ZeRO-3环境下使用deepseed.zero.Init初始化模型
        else:
            model = cls(config, **kwargs)  # 在非DeepSpeed ZeRO-3环境下常规初始化模型

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)  # 如果修改了默认dtype，则恢复为修改前的dtype

        return model

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        """
        Automatically sets the attention implementation in the provided config.

        Args:
            config: The model configuration to modify.
            use_flash_attention_2: Whether to use the Flash Attention 2 implementation.
            torch_dtype: Optional, override the default torch.dtype for initialization.
            device_map: Optional device mapping.
            check_device_map: Whether to check device map validity.

        Returns:
            The modified config with the attention implementation set.
        """
        # Set attention implementation based on parameters
        if use_flash_attention_2:
            config.attention_type = "flash_attention_2"
        elif config._attn_implementation is not None:
            config.attention_type = config._attn_implementation

        if device_map is not None and check_device_map:
            cls._validate_device_map(device_map)

        return config
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """
        Change the default dtype and return the previous one. This is needed when wanting to instantiate the model
        under specific dtype.

        Args:
            dtype (`torch.dtype`):
                a floating dtype to set to.

        Returns:
            `torch.dtype`: the original `dtype` that can be used to restore `torch.set_default_dtype(dtype)` if it was
            modified. If it wasn't, returns `None`.

        Note `set_default_dtype` currently only works with floating-point types and asserts if for example,
        `torch.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.
        """
        if not dtype.is_floating_point:
            raise ValueError(
                f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
            )

        logger.info(f"Instantiating {cls.__name__} model under default dtype {dtype}.")
        # 获取当前的默认 dtype
        dtype_orig = torch.get_default_dtype()
        # 设置新的默认 dtype
        torch.set_default_dtype(dtype)
        return dtype_orig

    @property
    def base_model(self) -> nn.Module:
        """
        `torch.nn.Module`: The main body of the model.
        """
        # 返回当前实例的 `base_model_prefix` 属性，如果不存在则返回自身
        return getattr(self, self.base_model_prefix, self)

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # 检查是否定义了 `prepare_inputs_for_generation` 或 `generate` 函数
        # 如果没有定义 `prepare_inputs_for_generation` 或 `generate`，则返回 True
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True

    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
        hard_check_only: bool = False,
    ):
        """
        Check and potentially enable the Flash Attention 2 features based on the provided configuration.

        Args:
            config: The configuration object for the model.
            torch_dtype (Optional[torch.dtype]): The desired dtype to set as default.
            device_map (Optional[Union[str, Dict[str, int]]]): Device mapping information.
            check_device_map (bool): Whether to check device map.
            hard_check_only (bool): Whether to perform a hard check only.

        This function checks if certain conditions are met in the provided configuration to enable Flash Attention 2.
        """
        # 此处应该有代码实现，用于检查和启用 Flash Attention 2 的相关特性
        pass
    # 检查并启用 SDPA（Scaled Dot-Product Attention）功能的静态方法。如果所有检查通过且 `hard_check_only` 为 False，
    # 则设置配置属性 `_attn_implementation` 为 "flash_attention_2"，以便模型可以正确初始化相应的注意力模块。
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
        if hard_check_only:
            # 如果仅进行严格检查并且当前类不支持 SDPA，则抛出值错误
            if not cls._supports_sdpa:
                raise ValueError(
                    f"{cls.__name__} does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet."
                    " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/28005. If you believe"
                    ' this error is a bug, please open an issue in Transformers GitHub repository and load your model with the argument `attn_implementation="eager"` meanwhile. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
                )
            # 如果未安装 PyTorch SDPA，则抛出导入错误
            if not is_torch_sdpa_available():
                raise ImportError(
                    "PyTorch SDPA requirements in Transformers are not met. Please install torch>=2.1.1."
                )

        # 如果未安装 PyTorch SDPA 或当前类不支持 SDPA，则直接返回配置
        if not is_torch_sdpa_available() or not cls._supports_sdpa:
            return config

        # 获取类属性 `_is_bettertransformer`，判断是否使用 BetterTransformer 模式
        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
        # 如果是 BetterTransformer 模式，则返回配置
        if _is_bettertransformer:
            return config

        # 如果不是严格检查模式，将配置的 `_attn_implementation` 设置为 "sdpa"
        if not hard_check_only:
            config._attn_implementation = "sdpa"
        # 返回更新后的配置
        return config

    # 启用输入嵌入的梯度计算的方法。用于在固定模型权重的同时微调适配器权重。
    def enable_input_require_grads(self):
        # 定义一个函数 `make_inputs_require_grads`，用于设置输出的梯度要求为 True
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        # 注册前向钩子 `_require_grads_hook` 到输入嵌入模块上
        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    # 移除输入嵌入梯度计算的方法。
    def disable_input_require_grads(self):
        # 移除前向钩子 `_require_grads_hook`
        self._require_grads_hook.remove()

    # 获取模型的输入嵌入的方法，返回一个 `nn.Module` 模块，将词汇映射到隐藏状态。
    def get_input_embeddings(self) -> nn.Module:
        # 获取基础模型，若存在，则递归调用其 `get_input_embeddings` 方法
        base_model = getattr(self, self.base_model_prefix, self)
        # 若 `base_model` 不是当前对象本身，则调用其 `get_input_embeddings` 方法
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            # 否则抛出未实现错误
            raise NotImplementedError
    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (`nn.Module`): A module mapping vocabulary to hidden states.
        """
        # 获取当前模型的基础模型（可能是自身或者其它模型）
        base_model = getattr(self, self.base_model_prefix, self)
        # 如果基础模型不是当前对象本身，则递归调用基础模型的设置输入嵌入方法
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            # 如果基础模型是当前对象本身，则抛出未实现的错误
            raise NotImplementedError

    def get_output_embeddings(self) -> nn.Module:
        """
        Returns the model's output embeddings.

        Returns:
            `nn.Module`: A torch module mapping hidden states to vocabulary.
        """
        # 对于没有输出嵌入的模型，返回空值
        return None  # Overwrite for models with output embeddings

    def _init_weights(self, module):
        """
        Initialize the weights. This method should be overridden by derived class and is
        the only initialization method that will be called when loading a checkpoint
        using `from_pretrained`. Any attempt to initialize outside of this function
        will be useless as the torch.nn.init function are all replaced with skip.
        """
        # 初始化权重的方法，应当由派生类重写。在使用 `from_pretrained` 加载检查点时，这是唯一会被调用的初始化方法。

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        # 如果模块已经被初始化，则直接返回
        if getattr(module, "_is_hf_initialized", False):
            return
        # 否则调用初始化权重的具体方法
        self._init_weights(module)
        # 标记模块已经被初始化
        module._is_hf_initialized = True

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        # 如果配置中设置了 `tie_word_embeddings`，则尝试绑定输入嵌入和输出嵌入的权重
        if getattr(self.config, "tie_word_embeddings", True):
            # 获取输出嵌入
            output_embeddings = self.get_output_embeddings()
            # 如果输出嵌入不为空，则尝试绑定或克隆权重
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        # 如果配置中设置了 `is_encoder_decoder` 和 `tie_encoder_decoder`，则尝试绑定编码器-解码器的权重
        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            # 如果存在基础模型前缀，则将当前对象替换为基础模型
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            # 调用内部方法绑定编码器-解码器权重
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        # 对于模型中的每一个模块，如果模块具有 `_tie_weights` 属性，则调用其绑定权重方法
        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()
    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """根据是否使用 TorchScript 来共享或克隆模块的权重"""
        if self.config.torchscript:
            # 如果使用 TorchScript，则克隆输入 embeddings 的权重到输出 embeddings
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            # 否则，直接共享输入 embeddings 的权重给输出 embeddings
            output_embeddings.weight = input_embeddings.weight

        # 如果输出 embeddings 存在偏置项
        if getattr(output_embeddings, "bias", None) is not None:
            # 对输出 embeddings 的偏置进行填充，以匹配权重的形状
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        # 如果输出 embeddings 具有 'out_features' 属性，并且输入 embeddings 具有 'num_embeddings' 属性
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            # 设置输出 embeddings 的 out_features 属性为输入 embeddings 的 num_embeddings
            output_embeddings.out_features = input_embeddings.num_embeddings

    def _get_no_split_modules(self, device_map: str):
        """
        获取在使用 device_map 时不应分割的模块。我们遍历模块以获取底层的 `_no_split_modules`。

        Args:
            device_map (`str`):
                设备映射值。选项有 ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: 不应分割的模块列表
        """
        _no_split_modules = set()
        modules_to_check = [self]
        while len(modules_to_check) > 0:
            module = modules_to_check.pop(-1)
            # 如果模块不在 _no_split_modules 中，则继续检查其子模块
            if module.__class__.__name__ not in _no_split_modules:
                if isinstance(module, PreTrainedModel):
                    if module._no_split_modules is None:
                        raise ValueError(
                            f"{module.__class__.__name__} 不支持 `device_map='{device_map}'`。要实现支持，模型类需要实现 `_no_split_modules` 属性。"
                        )
                    else:
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                # 将当前模块的子模块加入待检查列表
                modules_to_check += list(module.children())
        return list(_no_split_modules)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
        ):
    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        调整模型的输入 token embeddings 矩阵大小，如果 `new_num_tokens != config.vocab_size` 的话。

        调整后负责在需要时绑定权重 embeddings，如果模型类有 `tie_weights()` 方法的话。

        参数:
            new_num_tokens (`int`, *可选*):
                embedding 矩阵中的新 token 数量。增加大小会在末尾添加新初始化的向量。减少大小会从末尾移除向量。
                如果未提供或为 `None`，仅返回指向模型输入 token 的 `torch.nn.Embedding` 模块的指针，不执行任何操作。
            pad_to_multiple_of (`int`, *可选*):
                如果设置，将填充 embedding 矩阵至提供的值的倍数。如果 `new_num_tokens` 设置为 `None`，则仅将 embedding
                填充至 `pad_to_multiple_of` 的倍数。

                这对于启用 NVIDIA 硬件的 Tensor Cores（计算能力 `>= 7.5`，Volta）或者利用 TPUs 时特别有用，这些硬件
                在序列长度为 128 的倍数时效果最佳。有关更多详细信息或调整大小的正确值的帮助，请参考此指南：
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        返回:
            `torch.nn.Embedding`: 指向模型输入 tokens Embedding 模块的指针。
        """
        # 调整 token embeddings 大小并返回
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # 如果 new_num_tokens 和 pad_to_multiple_of 都为 None，直接返回调整后的模型 embeddings
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # 更新基础模型和当前模型配置中的词汇大小
        self.config.vocab_size = model_embeds.weight.shape[0]
        self.vocab_size = model_embeds.weight.shape[0]

        # 如果需要，重新绑定权重
        self.tie_weights()

        # 返回调整后的模型 embeddings
        return model_embeds
    # 调整模型的 token embeddings 的大小
    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        # 获取当前的输入 embeddings
        old_embeddings = self.get_input_embeddings()
        # 调整 embeddings 的大小
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        
        # 如果旧的 embeddings 带有 _hf_hook 属性，将其挂钩移到新的 embeddings 上
        if hasattr(old_embeddings, "_hf_hook"):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        
        # 复制旧的 embeddings 是否需要梯度到新的 embeddings
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        
        # 设置模型的输入 embeddings 为新调整大小后的 embeddings
        self.set_input_embeddings(new_embeddings)
        
        # 检查是否量化了模型
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        
        # 更新 new_num_tokens，确保其与新 embeddings 的实际大小一致
        if pad_to_multiple_of is not None:
            # 如果使用了 deepspeed.zero3 并且未量化，则使用 deepspeed.zero.GatheredParameters 调整大小
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                    new_num_tokens = new_embeddings.weight.shape[0]
            else:
                # 否则，直接使用新 embeddings 的大小
                new_num_tokens = new_embeddings.weight.shape[0]
        
        # 如果输出 embeddings 存在且未绑定 word embeddings，调整 lm head 的大小
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            # 获取旧的 lm head
            old_lm_head = self.get_output_embeddings()
            # 调整 lm head 的大小
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            
            # 如果旧的 lm head 带有 _hf_hook 属性，将其挂钩移到新的 lm head 上
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            
            # 复制旧的 lm head 是否需要梯度到新的 lm head
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            
            # 设置模型的输出 embeddings 为新调整大小后的 lm head
            self.set_output_embeddings(new_lm_head)
        
        # 返回调整后的输入 embeddings
        return self.get_input_embeddings()

    # 获取调整大小后的 embeddings
    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        ...

    # 获取调整大小后的 lm head
    def _get_resized_lm_head(
        self, old_lm_head: nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ):
        ...

    # 将原始 lm head 复制到调整大小后的 lm head
    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # 将旧的 lm head 权重复制到新的 lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # 如果新的 lm head 存在偏置，将旧的 lm head 偏置复制到新的 lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        # 抛出未实现错误，提示用户在子类中实现这个方法来调整位置嵌入
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        # 抛出未实现错误，提示用户在子类中实现这个方法来获取位置嵌入
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # 如果需要修剪头部，则调用修剪方法
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        # 如果定义了初始化权重的方法，则执行权重初始化
        if _init_weights:
            # 调用_apply方法来初始化权重
            self.apply(self._initialize_weights)

            # 如果不是初始化所有权重，则不应该绑定权重
            # 因为from_pretrained(...)方法会自动绑定权重
            self.tie_weights()

    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        # 将新修剪的头部集合保存为先前存储的修剪头部集合与新修剪头部集合的并集
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # 不幸的是，我们必须将其存储为列表以便进行JSON序列化

        # 调用基础模型的内部方法来修剪头部
        self.base_model._prune_heads(heads_to_prune)
    # 激活当前模型的梯度检查点功能。
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        # 如果当前模型不支持梯度检查点，则抛出异常。
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        # 如果未提供梯度检查点参数，则使用默认值 {"use_reentrant": True}。
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}

        # 创建一个偏函数，用于调用 `torch.utils.checkpoint.checkpoint` 函数，并传入梯度检查点参数。
        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # 对于旧的梯度检查点格式（transformers < 4.35.0），对于在Hub上存在的模型，我们将回退到重写的 `_set_gradient_checkpointing` 方法。
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        # 如果不是使用旧格式，则调用 `self._set_gradient_checkpointing` 方法启用梯度检查点。
        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        # 否则，应用部分应用 `self._set_gradient_checkpointing` 方法，传入参数 `value=True`。
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            # 记录警告信息，提示使用了已废弃的梯度检查点格式。
            logger.warn(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

        # 如果存在 `_hf_peft_config_loaded` 属性，则需要确保输入的 `requires_grad` 为 True。
        if getattr(self, "_hf_peft_config_loaded", False):
            # 当使用 PEFT + 梯度检查点 + Trainer 时，需要确保输入的 `requires_grad` 为 True。
            # 这也适用于 PEFT：https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # 在使用 PEFT 进行训练时，只有 LoRA 层的 `requires_grad` 被设置为 True，但冻结层的输出需要传播梯度，以确保梯度的流动。
            self.enable_input_require_grads()
    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        is_gradient_checkpointing_set = False

        # Apply gradient checkpointing setting to the top-level module if supported,
        # such as LongT5Stack inheriting from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            # Set the checkpointing function for the top-level module
            self._gradient_checkpointing_func = gradient_checkpointing_func
            # Enable or disable gradient checkpointing
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        # Apply gradient checkpointing setting to all modules recursively
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                # Set the checkpointing function for the current module
                module._gradient_checkpointing_func = gradient_checkpointing_func
                # Enable or disable gradient checkpointing for the current module
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        # If no module supports gradient checkpointing, raise an error
        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        # Check if gradient checkpointing is supported
        if self.supports_gradient_checkpointing:
            # For older format (transformers < 4.35.0) or models on the Hub,
            # fall back to the deprecated `_set_gradient_checkpointing` method
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                # Disable gradient checkpointing using the modern method
                self._set_gradient_checkpointing(enable=False)
            else:
                # Warn about using deprecated checkpointing format
                logger.warn(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                # Apply partial method to disable gradient checkpointing
                self.apply(partial(self._set_gradient_checkpointing, value=False))

        # Disable input require gradients if Half precision config loaded
        if getattr(self, "_hf_peft_config_loaded", False):
            self.disable_input_require_grads()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        # Check if any module in the model has gradient checkpointing enabled
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())
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
    ):
        """
        Save the model to the specified directory.

        Arguments:
            save_directory (`Union[str, os.PathLike]`):
                Directory where the model should be saved.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Flag indicating if the current process is the main one.
            state_dict (`Optional[dict]`, *optional*):
                Optional dictionary containing the state of the model.
            save_function (`Callable`, *optional*):
                Function used for saving the model (default is `torch.save`).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether to push the saved model to a model hub (if supported).
            max_shard_size (`Union[int, str]`, *optional*, defaults to `"5GB"`):
                Maximum size of each shard when saving large models.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to ensure safe serialization of the model.
            variant (`Optional[str]`, *optional*):
                Variant of the model being saved (if applicable).
            token (`Optional[Union[str, bool]]`, *optional*):
                Token used for authentication or authorization.
            save_peft_format (`bool`, *optional*, defaults to `True`):
                Whether to save the model in PEFT format.
            **kwargs:
                Additional keyword arguments for customizing the saving process.
        """
        @wraps(PushToHubMixin.push_to_hub)
        def push_to_hub(self, *args, **kwargs):
            """
            Push the model to a model hub with specified tags.

            Arguments:
                *args:
                    Positional arguments for the push operation.
                **kwargs:
                    Keyword arguments for customizing the push operation.

            Returns:
                Result of the super class's `push_to_hub` method.
            """
            tags = self.model_tags if self.model_tags is not None else []

            tags_kwargs = kwargs.get("tags", [])
            if isinstance(tags_kwargs, str):
                tags_kwargs = [tags_kwargs]

            for tag in tags_kwargs:
                if tag not in tags:
                    tags.append(tag)

            if tags:
                kwargs["tags"] = tags
            return super().push_to_hub(*args, **kwargs)

        def get_memory_footprint(self, return_buffers=True):
            """
            Get the memory footprint of the model.

            Arguments:
                return_buffers (`bool`, *optional*, defaults to `True`):
                    Whether to include buffer tensors in the memory footprint calculation.

            Returns:
                Memory footprint of the model in bytes.
            """
            mem = sum([param.nelement() * param.element_size() for param in self.parameters()])
            if return_buffers:
                mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
                mem = mem + mem_bufs
            return mem

        @wraps(torch.nn.Module.cuda)
        def cuda(self, *args, **kwargs):
            """
            Move the model to CUDA device, if not quantized.

            Arguments:
                *args:
                    Positional arguments for the CUDA operation.
                **kwargs:
                    Keyword arguments for customizing the CUDA operation.

            Returns:
                Result of the super class's `cuda` method.
            
            Raises:
                ValueError: If the model is 4-bit or 8-bit quantized.
            """
            if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
                raise ValueError(
                    "Calling `cuda()` is not supported for `4-bit` or `8-bit` quantized models. "
                    "Please use the model as it is, since the model has already been set to the "
                    "correct devices and casted to the correct `dtype`."
                )
            else:
                return super().cuda(*args, **kwargs)

        @wraps(torch.nn.Module.to)
    # 定义一个类方法，用于从预训练模型加载模型实例
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
    ):
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
        hf_quantizer=None,
        keep_in_fp32_modules=None,
    ):
        """
        Load a pretrained model using the provided state_dict and configuration.

        Args:
            model: The model to load the pretrained weights into.
            state_dict: The pretrained weights as a state dictionary.
            loaded_keys: Keys of the loaded state_dict.
            resolved_archive_file: Path to the resolved archive file.
            pretrained_model_name_or_path: Name or path of the pretrained model.
            ignore_mismatched_sizes: If True, ignore mismatched tensor sizes.
            sharded_metadata: Metadata related to sharding.
            _fast_init: Whether to perform fast initialization.
            low_cpu_mem_usage: If True, use low CPU memory mode.
            device_map: Mapping of devices.
            offload_folder: Folder for offloading.
            offload_state_dict: State dictionary for offloading.
            dtype: Data type of the model weights.
            hf_quantizer: Quantizer for Hugging Face models.
            keep_in_fp32_modules: Modules to keep in FP32 format.

        Returns:
            None
        """

        # Implementation of pretrained model loading logic
        _move_model_to_meta(model, loaded_keys, "")  # Move model to meta device

        # Load state_dict from resolved archive file
        state_dict = load_state_dict(resolved_archive_file)

        # Placeholder for expected keys handling
        expected_keys = loaded_keys  # TODO: Replace with proper expected keys handling

        # Load state_dict into meta model and retrieve error messages if any
        error_msgs = _load_state_dict_into_meta_model(
            model,
            state_dict,
            loaded_keys,
            "",
            expected_keys=expected_keys,
            hf_quantizer=hf_quantizer,
        )

        return error_msgs

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        """
        Retrieve modules from the model based on provided module names.

        Args:
            names: List of module names to retrieve.
            add_prefix: Whether to add a prefix to retrieved module names.
            remove_prefix: Whether to remove a prefix from retrieved module names.

        Returns:
            List: Retrieved modules based on the provided names.
        """

        # Create a set of module keys from the provided names
        module_keys = {".".join(key.split(".")[:-1]) for key in names}

        # Special case handling for torch.nn.ParameterList
        module_keys = module_keys.union(
            {".".join(key.split(".")[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()}
        )

        retrieved_modules = []

        # Retrieve modules that match the module keys
        for name, module in self.named_modules():
            if remove_prefix:
                _prefix = f"{self.base_model_prefix}."
                name = name[len(_prefix) :] if name.startswith(_prefix) else name
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    @staticmethod
    def _load_pretrained_model_low_mem(
        model, loaded_state_dict_keys, resolved_archive_file, start_prefix="", hf_quantizer=None
    ):
        """
        This is an experimental function that loads the model using ~1.x model size CPU memory

        Before you call it do:

        1. save which state_dict keys are available
        2. drop state_dict before model is created, since the latter takes 1x model size memory

        Here then we continue:

        3. switch to the meta device all params/buffers that are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it doesn't handle missing_keys, unexpected_keys, mismatched_keys. It can't handle deepspeed. To
        handle bitsandbytes, needs non-empty hf_quantizer argument.
        """
        _move_model_to_meta(model, loaded_state_dict_keys, start_prefix)  # Move model to meta device
        state_dict = load_state_dict(resolved_archive_file)  # Load state_dict from archive file
        expected_keys = loaded_state_dict_keys  # Placeholder for expected keys
        error_msgs = _load_state_dict_into_meta_model(
            model,
            state_dict,
            loaded_state_dict_keys,
            start_prefix,
            expected_keys=expected_keys,
            hf_quantizer=hf_quantizer,
        )
        return error_msgs
    # 注册自定义模型类到指定的自动模型类中
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
        # 如果 `auto_class` 不是字符串，则将其转换为类名字符串
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入自动模型模块
        import transformers.models.auto as auto_module

        # 检查是否存在给定名称的自动模型类
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将自动模型类名赋值给当前类的 `_auto_class` 属性
        cls._auto_class = auto_class

    # 将模型转换为 BetterTransformer
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
        # 检查是否安装了 Optimum 库，如果没有则抛出 ImportError
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        # 导入 Optimum 库的版本信息
        from optimum.version import __version__ as optimum_version

        # 检查 Optimum 库的版本是否满足要求，如果不满足则抛出 ImportError
        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        # 导入 BetterTransformer 类
        from optimum.bettertransformer import BetterTransformer

        # 使用 BetterTransformer 类将当前模型转换为 BetterTransformer
        return BetterTransformer.transform(self)
    def reverse_bettertransformer(self):
        """
        Reverts the transformation from [`~PreTrainedModel.to_bettertransformer`] so that the original modeling is
        used, for example in order to save the model.

        Returns:
            [`PreTrainedModel`]: The model converted back to the original modeling.
        """
        # 检查是否已安装 optimum 包，否则抛出 ImportError
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        # 导入 optimum 版本信息，并检查是否符合最低要求版本 1.7.0
        from optimum.version import __version__ as optimum_version

        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        # 导入 BetterTransformer 类并调用其 reverse 方法，将模型转换回原始建模
        from optimum.bettertransformer import BetterTransformer

        return BetterTransformer.reverse(self)

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """

        # 在 TorchFX 代理或 Torch 脚本跟踪时跳过检查
        if is_torch_fx_proxy(input_ids) or torch.jit.is_tracing() or is_torchdynamo_compiling():
            return

        # 如果 attention_mask 不为 None 或者模型配置中 pad_token_id 为 None，则跳过警告
        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # 仅检查输入中的第一个和最后一个 token 是否包含 pad_token_id，以减少开销
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "
                "https://huggingface.co/docs/transformers/troubleshooting"
                "#incorrect-output-when-padding-tokens-arent-masked."
            )

            # 如果 pad_token_id 等于 BOS、EOS 或 SEP 中的任何一个，显示额外警告信息
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

            # 发出一次性的警告，用 logger 记录
            logger.warning_once(warn_string)

    @property
    # 发出警告，提醒用户 `_is_quantized_training_enabled` 函数将在 transformers 4.39.0 版本中弃用，建议使用 `model.hf_quantizer.is_trainable` 替代
    warnings.warn(
        "`_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead",
        FutureWarning,
    )

    # 检查当前对象是否具有属性 `hf_quantizer`
    if not hasattr(self, "hf_quantizer"):
        # 如果没有 `hf_quantizer` 属性，则返回 False
        return False

    # 返回 `hf_quantizer` 对象的 `is_trainable` 属性值
    return self.hf_quantizer.is_trainable
# 将 PreTrainedModel 类的 push_to_hub 方法复制一份，赋值给自身，以备后续修改
PreTrainedModel.push_to_hub = copy_func(PreTrainedModel.push_to_hub)

# 如果 push_to_hub 方法有文档字符串，则格式化文档字符串，插入模型、AutoModel 和模型文件的相关信息
if PreTrainedModel.push_to_hub.__doc__ is not None:
    PreTrainedModel.push_to_hub.__doc__ = PreTrainedModel.push_to_hub.__doc__.format(
        object="model", object_class="AutoModel", object_files="model file"
    )

# 定义一个计算 SQuAD 起始位置 logit 的神经网络模块
class PoolerStartLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # 使用全连接层将隐藏状态映射到一个数值
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
        # 使用全连接层计算起始位置的 logit，并将结果压缩维度
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            # 根据模型参数的数据类型，对无效位置的 logit 进行处理，使用不同的填充值
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


# 定义一个计算 SQuAD 结束位置 logit 的神经网络模块
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
        # 第一个全连接层将两倍的隐藏状态映射到隐藏大小
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # 激活函数为双曲正切函数
        self.activation = nn.Tanh()
        # 使用 LayerNorm 对隐藏大小进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 第二个全连接层将隐藏状态映射到一个数值
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
                模型的最终隐藏状态。
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                标记范围内第一个标记的隐藏状态。
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                标记范围内第一个标记的位置。
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                用于无效位置的掩码，如查询和特殊符号（PAD、SEP、CLS）。1.0 表示该标记应被屏蔽。

        <Tip>

        `start_states` 或 `start_positions` 中的一个必须不为 `None`。如果两者都设置了，`start_positions` 会覆盖 `start_states`。

        </Tip>

        Returns:
            `torch.FloatTensor`: SQuAD 任务的结束位置logits。
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x
class PoolerAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config):
        super().__init__()
        # Initialize a linear layer that maps concatenated hidden states to the hidden size
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # Activation function for the dense layer
        self.activation = nn.Tanh()
        # Final linear layer to compute logits for SQuAD answer class
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
        # Ensure the hidden state size is retrieved correctly
        hsz = hidden_states.shape[-1]
        # Ensure that either start_states or start_positions is provided
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"

        # If start_positions is provided, derive start_states from hidden_states
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        # If cls_index is provided, derive cls_token_state from hidden_states
        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            # Otherwise, take the last token's hidden state as cls_token_state
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        # Concatenate start_states and cls_token_state, apply dense layers and activation
        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        # Apply final linear layer and squeeze to get SQuAD answer class logits
        x = self.dense_1(x).squeeze(-1)

        return x


@dataclass
class SquadHeadOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a [`~modeling_utils.SQuADHead`].
    """
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

    # Optional: 可选参数，以下各变量用于存储模型的不同输出结果，如果未提供`start_positions`或`end_positions`，则可能为空
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
        # 初始化 SQuAD 头部模块，设置起始和结束位置的 top k 值
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        # 初始化起始位置的 logits 池化层
        self.start_logits = PoolerStartLogits(config)
        # 初始化结束位置的 logits 池化层
        self.end_logits = PoolerEndLogits(config)
        # 初始化答案分类的池化层
        self.answer_class = PoolerAnswerClass(config)

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
    ):
        """
        Perform forward pass of the SQuAD head module.

        Args:
            hidden_states (torch.FloatTensor): Sequence of hidden states.
            start_positions (Optional[torch.LongTensor]): Tensor of start positions for the answer spans.
            end_positions (Optional[torch.LongTensor]): Tensor of end positions for the answer spans.
            cls_index (Optional[torch.LongTensor]): Index of the classification token if used.
            is_impossible (Optional[torch.LongTensor]): Tensor indicating if the question is unanswerable.
            p_mask (Optional[torch.FloatTensor]): Mask indicating which elements in the input sequence should not be attended to.
            return_dict (bool): Whether to return a dictionary.

        Returns:
            SquadHeadOutput: Output of the SQuAD head module.
        """
        # 实现 SQuAD 头部的前向传播逻辑
        # 这里应该包含具体的模型逻辑，根据输入参数计算输出
        pass


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
        # 调用父类的初始化方法
        super().__init__()

        # 从配置对象中获取摘要类型，如果未指定则默认为"last"
        self.summary_type = getattr(config, "summary_type", "last")
        
        # 如果摘要类型为"attn"，则抛出未实现错误，建议使用标准的多头注意力模块
        if self.summary_type == "attn":
            raise NotImplementedError

        # 初始化摘要为一个Identity对象，这个对象在前向传播中不做任何操作
        self.summary = Identity()

        # 如果配置中指定了使用投影进行摘要操作
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            # 如果配置中指定了将投影映射到标签并且标签数大于0，则num_classes为标签数
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            # 否则num_classes为隐藏大小
            else:
                num_classes = config.hidden_size
            # 使用线性层将隐藏状态映射到num_classes维度
            self.summary = nn.Linear(config.hidden_size, num_classes)

        # 根据配置中指定的激活函数字符串，获取对应的激活函数或者使用Identity作为激活函数
        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = get_activation(activation_string) if activation_string else Identity()

        # 初始化第一个dropout层为Identity对象，如果配置中指定了第一个dropout的概率，则使用nn.Dropout进行初始化
        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        # 初始化最后一个dropout层为Identity对象，如果配置中指定了最后一个dropout的概率，则使用nn.Dropout进行初始化
        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)
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
        # 根据选择的汇总类型进行汇总操作
        if self.summary_type == "last":
            # 取每个序列的最后一个隐藏状态
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            # 取每个序列的第一个隐藏状态
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            # 对整个序列的隐藏状态进行平均
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                # 如果没有提供 cls_index，则默认选择每个序列的最后一个 token 作为分类 token
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                # 将 cls_index 扩展为与 hidden_states 相同的维度
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # 从 hidden_states 中根据 cls_index 提取对应的隐藏状态
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            # 如果选择了注意力汇总类型，目前尚未实现此功能，抛出未实现错误
            raise NotImplementedError

        # 对输出进行第一个 dropout 操作
        output = self.first_dropout(output)
        # 将汇总后的向量传递给汇总层
        output = self.summary(output)
        # 对汇总后的向量应用激活函数
        output = self.activation(output)
        # 对最终输出进行最后一个 dropout 操作
        output = self.last_dropout(output)

        return output
# 递归地解包模型，从可能的容器中解开（如在分布式训练中使用的容器）。
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # 如果模型具有 `module` 属性，说明模型被包装，需要递归解包
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


# 展开设备映射，返回对应参数名到设备的映射。
def expand_device_map(device_map, param_names, start_prefix):
    """
    Expand a device map to return the correspondance parameter name to device.
    """
    # 创建新的设备映射字典
    new_device_map = {}
    # 过滤参数名列表，仅保留以给定前缀开头的参数名，并去除前缀
    param_names = [p[len(start_prefix) :] for p in param_names if p.startswith(start_prefix)]
    # 遍历设备映射，更新新的设备映射字典
    for module, device in device_map.items():
        new_device_map.update(
            # 对于每个参数名，如果与模块名匹配，或者以模块名加点开头，或者模块名为空，则更新映射
            {p: device for p in param_names if p == module or p.startswith(f"{module}.") or module == ""}
        )
    return new_device_map


# 获取仅包含已转移到磁盘的权重的碎片文件列表。
def get_disk_only_shard_files(device_map, sharded_metadata, start_prefix):
    """
    Returns the list of shard files containing only weights offloaded to disk.
    """
    # 从权重映射中提取与给定前缀匹配的权重名称及其对应的文件名
    weight_map = {
        p[len(start_prefix) :]: v for p, v in sharded_metadata["weight_map"].items() if p.startswith(start_prefix)
    }
    # 创建一个默认值为列表的字典，用于存储每个文件的设备列表
    files_content = collections.defaultdict(list)
    # 遍历权重映射，为每个权重名称找到对应的设备列表并存储到 files_content 中
    for weight_name, filename in weight_map.items():
        while len(weight_name) > 0 and weight_name not in device_map:
            weight_name = ".".join(weight_name.split(".")[:-1])
        files_content[filename].append(device_map[weight_name])

    # 返回仅包含磁盘设备的文件列表
    return [fname for fname, devices in files_content.items() if set(devices) == {"disk"}]
```