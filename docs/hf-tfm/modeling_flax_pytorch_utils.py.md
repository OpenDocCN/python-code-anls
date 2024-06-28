# `.\modeling_flax_pytorch_utils.py`

```py
# 设置编码为 UTF-8，确保文件能正确处理非 ASCII 字符
# 版权声明及许可协议，指定本文件的版权归属及使用许可
# 注意事项：根据 Apache 许可证 2.0 版本，除非符合许可协议，否则禁止使用本文件
# 获取许可协议的详细信息，请访问指定的 URL
# 如果适用法律要求或书面同意，软件将按“原样”分发，不提供任何明示或暗示的担保或条件
# 查看许可协议以了解具体语言和限制条件

""" PyTorch - Flax general utilities."""
# 导入所需的标准库和模块

import os  # 导入操作系统功能
from pickle import UnpicklingError  # 导入反序列化错误异常
from typing import Dict, Tuple  # 导入类型提示

import jax  # 导入 JAX 库
import jax.numpy as jnp  # 导入 JAX 的 NumPy 接口
import numpy as np  # 导入 NumPy 库
from flax.serialization import from_bytes  # 从字节流中反序列化对象
from flax.traverse_util import flatten_dict, unflatten_dict  # 对字典进行扁平化和展开操作

import transformers  # 导入 Transformers 库

from . import is_safetensors_available, is_torch_available  # 导入本地模块
from .utils import logging  # 从本地工具模块中导入日志功能


if is_torch_available():  # 如果 Torch 可用
    import torch  # 导入 Torch 库

if is_safetensors_available():  # 如果 SafeTensors 可用
    from safetensors import safe_open  # 导入安全打开文件函数
    from safetensors.flax import load_file as safe_load_file  # 导入安全加载文件函数


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


#####################
# PyTorch => Flax #
#####################

# 定义一个函数，将 PyTorch 的检查点加载到 Flax 模型的状态字典中
def load_pytorch_checkpoint_in_flax_state_dict(
    flax_model, pytorch_checkpoint_path, is_sharded, allow_missing_keys=False
):
    """Load pytorch checkpoints in a flax model"""

    if not is_sharded:  # 如果不是分片加载
        pt_path = os.path.abspath(pytorch_checkpoint_path)  # 获取 PyTorch 检查点的绝对路径
        logger.info(f"Loading PyTorch weights from {pt_path}")  # 记录日志，显示正在加载的 PyTorch 权重文件路径

        if pt_path.endswith(".safetensors"):  # 如果文件路径以 ".safetensors" 结尾
            pt_state_dict = {}  # 初始化一个空字典，用于存储 PyTorch 的状态字典
            with safe_open(pt_path, framework="flax") as f:  # 使用安全方式打开文件
                for k in f.keys():  # 遍历文件中的键
                    pt_state_dict[k] = f.get_tensor(k)  # 将文件中的张量数据存储到状态字典中
        else:  # 如果文件路径不以 ".safetensors" 结尾
            try:
                import torch  # 尝试导入 Torch 库
                from .pytorch_utils import is_torch_greater_or_equal_than_1_13  # 导入版本比较工具函数
            except (ImportError, ModuleNotFoundError):  # 处理导入错误或模块未找到异常
                logger.error(
                    "Loading a PyTorch model in Flax, requires both PyTorch and Flax to be installed. Please see"
                    " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
                    " instructions."
                )  # 记录加载错误信息，指出安装 PyTorch 和 Flax 的必要性
                raise  # 抛出异常

            # 根据不同的 Torch 版本加载权重数据
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            pt_state_dict = torch.load(pt_path, map_location="cpu", **weights_only_kwarg)  # 使用 Torch 加载权重数据到状态字典
            logger.info(f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters.")  # 记录日志，显示加载的参数数量

        flax_state_dict = convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)  # 将 PyTorch 的状态字典转换为 Flax 的状态字典
    else:
        # 如果模型是分片的，并且 pytorch_checkpoint_path 已经包含了 .pt 分片文件的列表
        # 则将使用 convert_pytorch_sharded_state_dict_to_flax 函数将其转换为 Flax 模型的状态字典
        flax_state_dict = convert_pytorch_sharded_state_dict_to_flax(pytorch_checkpoint_path, flax_model)
    # 返回转换后的 Flax 模型状态字典
    return flax_state_dict
# 将 PyTorch 权重名称重命名为对应的 Flax 权重名称并在必要时重塑张量
def rename_key_and_reshape_tensor(
    pt_tuple_key: Tuple[str],
    pt_tensor: np.ndarray,
    random_flax_state_dict: Dict[str, jnp.ndarray],
    model_prefix: str,
) -> (Tuple[str], np.ndarray):
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""

    def is_key_or_prefix_key_in_dict(key: Tuple[str]) -> bool:
        """Checks if `key` of `(prefix,) + key` is in random_flax_state_dict"""
        return len(set(random_flax_state_dict) & {key, (model_prefix,) + key}) > 0

    # layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    if pt_tuple_key[-1] in ["weight", "gamma"] and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # batch norm layer mean
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("mean",)
    if pt_tuple_key[-1] == "running_mean" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # batch norm layer var
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("var",)
    if pt_tuple_key[-1] == "running_var" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
    if pt_tuple_key[-1] == "weight" and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # conv layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4 and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor

    # linear layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm weight
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm bias
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    # New `weight_norm` from https://github.com/huggingface/transformers/pull/24030
    name = None
    if pt_tuple_key[-3::2] == ("parametrizations", "original0"):
        name = pt_tuple_key[-2] + "_g"
    elif pt_tuple_key[-3::2] == ("parametrizations", "original1"):
        name = pt_tuple_key[-2] + "_v"
    if name is not None:
        renamed_pt_tuple_key = pt_tuple_key[:-3] + (name,)
        return renamed_pt_tuple_key, pt_tensor

    # 默认情况，返回原始的 tuple key 和张量
    return pt_tuple_key, pt_tensor
    # 根据条件确定要使用的 bfloat16 类型，如果 from_bin 是 True 则使用 torch.bfloat16，否则使用字符串 "bfloat16"
    bfloat16 = torch.bfloat16 if from_bin else "bfloat16"

    # 创建一个字典，将 PyTorch 模型状态字典中每个键对应的数据类型收集起来
    weight_dtypes = {k: v.dtype for k, v in pt_state_dict.items()}

    # 如果 from_bin 是 True，则需要将 PyTorch 模型状态字典中的 bfloat16 类型转换为 float32，以避免精度损失问题
    if from_bin:
        for k, v in pt_state_dict.items():
            # 当前 numpy 不支持 bfloat16，因此在此情况下需要将其转换为 float32 类型
            if v.dtype == bfloat16:
                v = v.float()
            pt_state_dict[k] = v.numpy()

    # 获取 Flax 模型的基础模型前缀
    model_prefix = flax_model.base_model_prefix

    # 如果模型中包含批归一化层，则使用 params 字典
    if "params" in flax_model.params:
        flax_model_params = flax_model.params["params"]
    else:
        flax_model_params = flax_model.params

    # 将 Flax 模型参数展平为字典
    random_flax_state_dict = flatten_dict(flax_model_params)

    # 如果模型参数中包含 batch_stats，则将其展平并加入随机状态字典中
    if "batch_stats" in flax_model.params:
        flax_batch_stats = flatten_dict(flax_model.params["batch_stats"])
        random_flax_state_dict.update(flax_batch_stats)

    # 初始化一个空的 Flax 状态字典
    flax_state_dict = {}

    # 根据条件判断是将头部模型加载到基础模型中，还是将基础模型加载到带头部模型中
    load_model_with_head_into_base_model = (model_prefix not in flax_model_params) and (
        model_prefix in {k.split(".")[0] for k in pt_state_dict.keys()}
    )
    load_base_model_into_model_with_head = (model_prefix in flax_model_params) and (
        model_prefix not in {k.split(".")[0] for k in pt_state_dict.keys()}
    )

    # 需要修改一些参数名称以匹配 Flax 模型的命名规范
    # 此处的注释指出了需要进行参数名称匹配的必要性，但没有具体说明如何实现
    # 遍历 PyTorch 状态字典中的每个键值对
    for pt_key, pt_tensor in pt_state_dict.items():
        # 将点分割的键名转换为元组形式
        pt_tuple_key = tuple(pt_key.split("."))
        # 检查当前权重数据类型是否为 bfloat16
        is_bfloat_16 = weight_dtypes[pt_key] == bfloat16

        # 如果需要，移除基础模型前缀
        has_base_model_prefix = pt_tuple_key[0] == model_prefix
        if load_model_with_head_into_base_model and has_base_model_prefix:
            pt_tuple_key = pt_tuple_key[1:]

        # 使用指定函数重命名键名并调整张量形状
        flax_key, flax_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, pt_tensor, random_flax_state_dict, model_prefix
        )

        # 如果需要，添加模型前缀
        require_base_model_prefix = (model_prefix,) + flax_key in random_flax_state_dict
        if load_base_model_into_model_with_head and require_base_model_prefix:
            flax_key = (model_prefix,) + flax_key

        # 检查重命名后的键是否存在于随机化的 Flax 状态字典中
        if flax_key in random_flax_state_dict:
            # 检查张量形状是否与期望的 Flax 模型权重形状一致，否则抛出 ValueError
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )

        # 如果 Flax 模型包含批次归一化层，添加批次统计信息
        if "batch_stats" in flax_model.params:
            if "mean" in flax_key[-1] or "var" in flax_key[-1]:
                # 将 Flax 张量转换为 JAX 数组，并存储在新的位置
                flax_state_dict[("batch_stats",) + flax_key] = jnp.asarray(flax_tensor)
                continue
            # 移除 num_batches_tracked 键
            if "num_batches_tracked" in flax_key[-1]:
                flax_state_dict.pop(flax_key, None)
                continue

            # 否则，将权重添加到 params 键下
            flax_state_dict[("params",) + flax_key] = (
                jnp.asarray(flax_tensor) if not is_bfloat_16 else jnp.asarray(flax_tensor, dtype=jnp.bfloat16)
            )
        else:
            # 如果模型不包含批次归一化层，也将权重添加到状态字典中
            flax_state_dict[flax_key] = (
                jnp.asarray(flax_tensor) if not is_bfloat_16 else jnp.asarray(flax_tensor, dtype=jnp.bfloat16)
            )

    # 返回经过重新构造的 Flax 状态字典
    return unflatten_dict(flax_state_dict)
############################
# Sharded Pytorch => Flax #
############################

# 将分片的 PyTorch 状态字典转换为 Flax 格式
def convert_pytorch_sharded_state_dict_to_flax(shard_filenames, flax_model):
    import torch

    from .pytorch_utils import is_torch_greater_or_equal_than_1_13

    # Load the index
    flax_state_dict = {}
    # 调用函数 unflatten_dict 将 Flax 状态字典展开并返回
    return unflatten_dict(flax_state_dict)


#####################
# Flax => PyTorch #
#####################

# 在 PyTorch 模型中加载 Flax 检查点
def load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path):
    """Load flax checkpoints in a PyTorch model"""
    flax_checkpoint_path = os.path.abspath(flax_checkpoint_path)
    logger.info(f"Loading Flax weights from {flax_checkpoint_path}")

    # import correct flax class
    flax_cls = getattr(transformers, "Flax" + model.__class__.__name__)

    # load flax weight dict
    if flax_checkpoint_path.endswith(".safetensors"):
        # 使用 safe_load_file 函数加载安全张量文件
        flax_state_dict = safe_load_file(flax_checkpoint_path)
        # 使用分隔符 "." 对 Flax 状态字典进行展开
        flax_state_dict = unflatten_dict(flax_state_dict, sep=".")
    else:
        with open(flax_checkpoint_path, "rb") as state_f:
            try:
                # 尝试从文件中读取并解析 Flax 序列化对象
                flax_state_dict = from_bytes(flax_cls, state_f.read())
            except UnpicklingError:
                raise EnvironmentError(f"Unable to convert {flax_checkpoint_path} to Flax deserializable object. ")

    return load_flax_weights_in_pytorch_model(model, flax_state_dict)


# 在 PyTorch 模型中加载 Flax 权重
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

    # check if we have bf16 weights
    # 检查是否存在 bf16 类型的权重，并转换为 fp32 类型以便 PyTorch 加载
    is_type_bf16 = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype == jnp.bfloat16, flax_state)).values()
    if any(is_type_bf16):
        # 如果发现 Flax 模型中包含 bf16 类型的权重，则警告并将它们转换为 fp32 类型
        logger.warning(
            "Found ``bfloat16`` weights in Flax model. Casting all ``bfloat16`` weights to ``float32`` "
            "before loading those in PyTorch model."
        )
        flax_state = jax.tree_util.tree_map(
            lambda params: params.astype(np.float32) if params.dtype == jnp.bfloat16 else params, flax_state
        )

    # 将 Flax 状态字典展开为一维字典
    flax_state_dict = flatten_dict(flax_state)
    # 获取 PyTorch 模型的状态字典
    pt_model_dict = pt_model.state_dict()

    # 判断是否需要将模型头加载到基础模型中
    load_model_with_head_into_base_model = (pt_model.base_model_prefix in flax_state) and (
        pt_model.base_model_prefix not in {k.split(".")[0] for k in pt_model_dict.keys()}
    )
    # 检查是否需要将基础模型加载到带头部的模型中
    load_base_model_into_model_with_head = (pt_model.base_model_prefix not in flax_state) and (
        pt_model.base_model_prefix in {k.split(".")[0] for k in pt_model_dict.keys()}
    )

    # 用于跟踪未预期和丢失的键
    unexpected_keys = []
    missing_keys = set(pt_model_dict.keys())

    # 加载 PyTorch 模型的状态字典
    pt_model.load_state_dict(pt_model_dict)

    # 将缺失的键重新转换为列表
    missing_keys = list(missing_keys)

    # 如果存在未预期的键，则发出警告
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
        # 所有 Flax 模型的权重均已用于初始化 PyTorch 模型
        logger.warning(f"All Flax model weights were used when initializing {pt_model.__class__.__name__}.\n")

    # 如果存在丢失的键，则发出警告
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the Flax model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )
    else:
        # 所有的权重已从 Flax 模型初始化
        logger.warning(
            f"All the weights of {pt_model.__class__.__name__} were initialized from the Flax model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {pt_model.__class__.__name__} for predictions without further training."
        )

    # 返回加载后的 PyTorch 模型
    return pt_model
```