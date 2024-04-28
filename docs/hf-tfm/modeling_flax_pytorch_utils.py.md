# `.\transformers\modeling_flax_pytorch_utils.py`

```
# 设置编码格式为 UTF-8
# 版权声明
# 版权所有 2021 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权;
# 除非符合许可证的要求，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 按“原样”分发，不附带任何明示或暗示的保证或条件。
# 有关许可证的详细信息，请参阅
# 许可证。


# 导入所需的库和模块
import os  # 导入操作系统相关功能的模块
from pickle import UnpicklingError  # 导入用于处理 pickle 反序列化错误的异常
from typing import Dict, Tuple  # 导入用于类型提示的 Dict 和 Tuple 类型

import jax  # 导入 JAX 库
import jax.numpy as jnp  # 导入 JAX 中的 NumPy 接口
import numpy as np  # 导入 NumPy 库
from flax.serialization import from_bytes  # 导入从字节流反序列化对象的函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入用于扁平化和反扁平化字典的函数

import transformers  # 导入 transformers 库

from . import is_safetensors_available  # 导入判断 safetensors 是否可用的函数
from .utils import logging  # 从当前包中导入 logging 模块

# 如果 safetensors 可用，则导入相关功能
if is_safetensors_available():
    from safetensors import safe_open  # 导入安全打开文件的函数
    from safetensors.flax import load_file as safe_load_file  # 导入安全加载文件的函数

# 获取日志记录器对象
logger = logging.get_logger(__name__)


#####################
# PyTorch => Flax #
#####################


# 将 PyTorch 检查点加载到 Flax 模型中的函数
def load_pytorch_checkpoint_in_flax_state_dict(
    flax_model, pytorch_checkpoint_path, is_sharded, allow_missing_keys=False
):
    """Load pytorch checkpoints in a flax model"""
    try:
        import torch  # 导入 PyTorch 库
        from .pytorch_utils import is_torch_greater_or_equal_than_1_13  # 导入用于检查 PyTorch 版本的函数
    except (ImportError, ModuleNotFoundError):  # 处理导入错误
        # 如果 PyTorch 或 Flax 未安装，则输出错误消息并引发异常
        logger.error(
            "Loading a PyTorch model in Flax, requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
            " instructions."
        )
        raise

    # 如果模型没有被分片
    if not is_sharded:
        # 获取 PyTorch 检查点路径
        pt_path = os.path.abspath(pytorch_checkpoint_path)
        logger.info(f"Loading PyTorch weights from {pt_path}")

        # 如果 PyTorch 检查点是 .safetensors 格式
        if pt_path.endswith(".safetensors"):
            # 初始化一个空字典来保存 PyTorch 状态字典
            pt_state_dict = {}
            # 使用安全打开文件，读取字典并保存为 PyTorch 的张量
            with safe_open(pt_path, framework="pt") as f:
                for k in f.keys():
                    pt_state_dict[k] = f.get_tensor(k)
        else:  # 如果 PyTorch 检查点不是 .safetensors 格式
            # 使用 PyTorch 加载检查点文件，保存为 PyTorch 的状态字典
            pt_state_dict = torch.load(pt_path, map_location="cpu", weights_only=is_torch_greater_or_equal_than_1_13)
        # 记录 PyTorch 检查点中包含的参数数量
        logger.info(f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters.")

        # 将 PyTorch 状态字典转换为 Flax 状态字典
        flax_state_dict = convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)
    else:  # 如果模型被分片
        # 将 PyTorch 分片状态字典转换为 Flax 状态字典
        flax_state_dict = convert_pytorch_sharded_state_dict_to_flax(pytorch_checkpoint_path, flax_model)
    # 返回 Flax 状态字典
    return flax_state_dict


# 重命名键并重塑张量的函数
def rename_key_and_reshape_tensor(
    pt_tuple_key: Tuple[str],
    pt_tensor: np.ndarray,
    random_flax_state_dict: Dict[str, jnp.ndarray],
    model_prefix: str,
    # 定义函数，将 PyTorch 权重名称重命名为对应的 Flax 权重名称，并在必要时重塑张量
    def is_key_or_prefix_key_in_dict(key: Tuple[str]) -> bool:
        # 检查 `(prefix,) + key` 是否在 random_flax_state_dict 中
        return len(set(random_flax_state_dict) & {key, (model_prefix,) + key}) > 0

    # 对层归一化进行处理
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    if pt_tuple_key[-1] in ["weight", "gamma"] and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # 处理批归一化层的均值
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("mean",)
    if pt_tuple_key[-1] == "running_mean" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # 处理批归一化层的方差
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("var",)
    if pt_tuple_key[-1] == "running_var" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # 处理嵌入层
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
    if pt_tuple_key[-1] == "weight" and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # 处理卷积层
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4 and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor

    # 处理线性层
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor

    # 处理旧的 PyTorch 层归一化权重
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # 处理旧的 PyTorch 层归一化偏置
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    # 处理来自 https://github.com/huggingface/transformers/pull/24030 的新 `weight_norm`
    name = None
    if pt_tuple_key[-3::2] == ("parametrizations", "original0"):
        name = pt_tuple_key[-2] + "_g"
    elif pt_tuple_key[-3::2] == ("parametrizations", "original1"):
        name = pt_tuple_key[-2] + "_v"
    if name is not None:
        renamed_pt_tuple_key = pt_tuple_key[:-3] + (name,)
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor

# �� PyTorch 状态字典转换为 Flax 模型
def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    # 将 PyTorch 张量转换为 numpy
    # 当前 numpy 不支持 bfloat16，需要在这种情况下转换为 float32 以避免丢失精度
    try:
        import torch  # noqa: F401
    # 处理 ImportError 和 ModuleNotFoundError 异常
    except (ImportError, ModuleNotFoundError):
        # 记录错误日志，指示加载 PyTorch 模型需要安装 PyTorch 和 Flax
        logger.error(
            "Loading a PyTorch model in Flax, requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
            " instructions."
        )
        # 抛出异常
        raise

    # 创建一个字典，包含 PyTorch 模型参数的名称及其数据类型
    weight_dtypes = {k: v.dtype for k, v in pt_state_dict.items()}
    # 将 PyTorch 模型参数转换为 NumPy 数组，如果数据类型是 torch.bfloat16，则先转换为 float 再转换为 NumPy 数组
    pt_state_dict = {
        k: v.numpy() if not v.dtype == torch.bfloat16 else v.float().numpy() for k, v in pt_state_dict.items()
    }

    # 获取 Flax 模型的基本模型前缀
    model_prefix = flax_model.base_model_prefix

    # 如果 Flax 模型包含批处理标准化层，则使用参数字典
    if "params" in flax_model.params:
        flax_model_params = flax_model.params["params"]
    else:
        flax_model_params = flax_model.params
    # 将 Flax 模型参数扁平化为字典
    random_flax_state_dict = flatten_dict(flax_model_params)

    # 如果 Flax 模型参数包含批处理统计信息，则将其添加到字典中
    if "batch_stats" in flax_model.params:
        flax_batch_stats = flatten_dict(flax_model.params["batch_stats"])
        random_flax_state_dict.update(flax_batch_stats)

    # 初始化一个空的 Flax 模型参数字典
    flax_state_dict = {}

    # 检查是否需要将头部模型加载到基本模型中
    load_model_with_head_into_base_model = (model_prefix not in flax_model_params) and (
        model_prefix in {k.split(".")[0] for k in pt_state_dict.keys()}
    )
    # 检查是否需要将基本模型加载到带有头部的模型中
    load_base_model_into_model_with_head = (model_prefix in flax_model_params) and (
        model_prefix not in {k.split(".")[0] for k in pt_state_dict.keys()}
    )

    # 需要修改一些参数名称以匹配 Flax 的名称
    # 遍历 PyTorch 状态字典中的键值对，键为参数名，值为参数张量
    for pt_key, pt_tensor in pt_state_dict.items():
        # 将 PyTorch 参数名拆分为元组形式
        pt_tuple_key = tuple(pt_key.split("."))
        # 检查当前参数是否为 bfloat16 类型
        is_bfloat_16 = weight_dtypes[pt_key] == torch.bfloat16

        # 如果需要，移除模型前缀
        has_base_model_prefix = pt_tuple_key[0] == model_prefix
        if load_model_with_head_into_base_model and has_base_model_prefix:
            pt_tuple_key = pt_tuple_key[1:]

        # 通过重命名函数来正确更改权重参数的名称并重新调整张量形状
        flax_key, flax_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, pt_tensor, random_flax_state_dict, model_prefix
        )

        # 如果需要，添加模型前缀
        require_base_model_prefix = (model_prefix,) + flax_key in random_flax_state_dict
        if load_base_model_into_model_with_head and require_base_model_prefix:
            flax_key = (model_prefix,) + flax_key

        # 检查是否存在相应的 Flax 参数
        if flax_key in random_flax_state_dict:
            # 检查张量形状是否与 Flax 参数的形状相匹配，若不匹配则引发 ValueError
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )

        # 如果模型包含批次规范化层，则添加批次统计信息
        if "batch_stats" in flax_model.params:
            if "mean" in flax_key[-1] or "var" in flax_key[-1]:
                # 添加批次均值和方差
                flax_state_dict[("batch_stats",) + flax_key] = jnp.asarray(flax_tensor)
                continue
            # 移除追踪批次数的键
            if "num_batches_tracked" in flax_key[-1]:
                flax_state_dict.pop(flax_key, None)
                continue

            # 添加意外的权重以引发警告
            flax_state_dict[("params",) + flax_key] = (
                jnp.asarray(flax_tensor) if not is_bfloat_16 else jnp.asarray(flax_tensor, dtype=jnp.bfloat16)
            )

        else:
            # 添加意外的权重以引发警告
            flax_state_dict[flax_key] = (
                jnp.asarray(flax_tensor) if not is_bfloat_16 else jnp.asarray(flax_tensor, dtype=jnp.bfloat16)
            )

    # 返回转换后的 Flax 状态字典
    return unflatten_dict(flax_state_dict)
############################
# Sharded Pytorch => Flax #
############################

# 将分片的 PyTorch 状态字典转换为 Flax 模型
def convert_pytorch_sharded_state_dict_to_flax(shard_filenames, flax_model):
    import torch

    # 导入判断函数，检查 PyTorch 版本是否大于等于 1.13
    from .pytorch_utils import is_torch_greater_or_equal_than_1_13

    # 加载索引
    flax_state_dict = {}
    # 返回展开的字典
    return unflatten_dict(flax_state_dict)


#####################
# Flax => PyTorch #
#####################

# 在 PyTorch 模型中加载 Flax 检查点
def load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path):
    """Load flax checkpoints in a PyTorch model"""
    # 将 Flax 检查点路径转换为绝对路径
    flax_checkpoint_path = os.path.abspath(flax_checkpoint_path)
    logger.info(f"Loading Flax weights from {flax_checkpoint_path}")

    # 导入正确的 Flax 类
    flax_cls = getattr(transformers, "Flax" + model.__class__.__name__)

    # 加载 Flax 权重字典
    if flax_checkpoint_path.endswith(".safetensors"):
        # 如果文件路径以 ".safetensors" 结尾，则安全加载文件
        flax_state_dict = safe_load_file(flax_checkpoint_path)
        # 使用指定分隔符展开字典
        flax_state_dict = unflatten_dict(flax_state_dict, sep=".")
    else:
        with open(flax_checkpoint_path, "rb") as state_f:
            try:
                # 尝试从字节流中加载 Flax 状态
                flax_state_dict = from_bytes(flax_cls, state_f.read())
            except UnpicklingError:
                raise EnvironmentError(f"Unable to convert {flax_checkpoint_path} to Flax deserializable object. ")

    # 将 Flax 模型权重加载到 PyTorch 模型中
    return load_flax_weights_in_pytorch_model(model, flax_state_dict)


def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """Load flax checkpoints in a PyTorch model"""

    try:
        import torch  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        logger.error(
            "Loading a Flax weights in PyTorch, requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
            " instructions."
        )
        raise

    # 检查是否存在 bf16 类型的权重
    is_type_bf16 = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype == jnp.bfloat16, flax_state)).values()
    if any(is_type_bf16):
        # 如果存在 bf16 类型的权重，则将所有权重转换为 float32 类型，因为 torch.from_numpy 无法处理 bf16 类型
        # 而且 PyTorch 尚不完全支持 bf16 类型
        logger.warning(
            "Found ``bfloat16`` weights in Flax model. Casting all ``bfloat16`` weights to ``float32`` "
            "before loading those in PyTorch model."
        )
        flax_state = jax.tree_util.tree_map(
            lambda params: params.astype(np.float32) if params.dtype == jnp.bfloat16 else params, flax_state
        )

    # 展开 Flax 状态字典
    flax_state_dict = flatten_dict(flax_state)
    # 获取 PyTorch 模型的状态字典
    pt_model_dict = pt_model.state_dict()

    # 检查是否将具有头部的模型加载到基础模型中
    load_model_with_head_into_base_model = (pt_model.base_model_prefix in flax_state) and (
        pt_model.base_model_prefix not in {k.split(".")[0] for k in pt_model_dict.keys()}
    )
    # 检查是否需要将基础模型加载到带有头部的模型中，条件是Flax状态中没有基础模型前缀，同时在PyTorch模型字典中存在基础模型前缀
    load_base_model_into_model_with_head = (pt_model.base_model_prefix not in flax_state) and (
        pt_model.base_model_prefix in {k.split(".")[0] for k in pt_model_dict.keys()}
    )

    # 用于跟踪未预期和缺失的键
    unexpected_keys = []
    missing_keys = set(pt_model_dict.keys())

    # 加载PyTorch模型的状态字典
    pt_model.load_state_dict(pt_model_dict)

    # 将缺失的键重新转换为列表
    missing_keys = list(missing_keys)

    # 如果有未预期的键，则发出警告
    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the Flax model were not used when initializing the PyTorch model"
            f" {pt_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {pt_model.__class__.__name__} from a Flax model trained on another task or with another architecture"
            " (e.g. initializing a BertForSequenceClassification model from a FlaxBertForPreTraining model).\n- This"
            f" IS NOT expected if you are initializing {pt_model.__class__.__name__} from a Flax model that you expect"
            " to be exactly identical (e.g. initializing a BertForSequenceClassification model from a"
            " FlaxBertForSequenceClassification model)."
        )
    else:
        # 所有的Flax模型权重都被用于初始化PyTorch模型
        logger.warning(f"All Flax model weights were used when initializing {pt_model.__class__.__name__}.\n")
    
    # 如果有缺失的键，则发出警告
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the Flax model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )
    else:
        # 所有的PyTorch模型权重都被从Flax模型初始化
        logger.warning(
            f"All the weights of {pt_model.__class__.__name__} were initialized from the Flax model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {pt_model.__class__.__name__} for predictions without further training."
        )

    # 返回PyTorch模型
    return pt_model
```