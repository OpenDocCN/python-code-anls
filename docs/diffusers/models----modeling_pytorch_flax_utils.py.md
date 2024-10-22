# `.\diffusers\models\modeling_pytorch_flax_utils.py`

```py
# 指定文件编码为 UTF-8
# coding=utf-8
# 版权所有 2024 The HuggingFace Inc. 团队。
#
# 根据 Apache 许可证版本 2.0（"许可证"）许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件在许可证下以“原样”方式分发，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 有关许可证下的特定权限和限制，请参见许可证。
"""PyTorch - Flax 一般实用工具。"""

# 从 pickle 模块导入 UnpicklingError 异常
from pickle import UnpicklingError

# 导入 jax 库及其 numpy 模块
import jax
import jax.numpy as jnp
# 导入 numpy 库
import numpy as np
# 从 flax.serialization 导入 from_bytes 函数
from flax.serialization import from_bytes
# 从 flax.traverse_util 导入 flatten_dict 函数
from flax.traverse_util import flatten_dict

# 从 utils 模块导入 logging
from ..utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

#####################
# Flax => PyTorch #
#####################

# 从指定模型文件加载 Flax 检查点到 PyTorch 模型
# 来源：https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flax_pytorch_utils.py#L224-L352
def load_flax_checkpoint_in_pytorch_model(pt_model, model_file):
    # 尝试打开模型文件以读取 Flax 状态
    try:
        with open(model_file, "rb") as flax_state_f:
            # 从字节流中反序列化 Flax 状态
            flax_state = from_bytes(None, flax_state_f.read())
    # 捕获反序列化错误
    except UnpicklingError as e:
        try:
            # 以文本模式打开模型文件
            with open(model_file) as f:
                # 检查文件内容是否以 "version" 开头
                if f.read().startswith("version"):
                    # 如果是，抛出 OSError，提示缺少 git-lfs
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please"
                        " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                        " folder you cloned."
                    )
                else:
                    # 否则，抛出 ValueError
                    raise ValueError from e
        # 捕获 Unicode 解码错误和其他值错误
        except (UnicodeDecodeError, ValueError):
            # 抛出环境错误，提示无法转换文件
            raise EnvironmentError(f"Unable to convert {model_file} to Flax deserializable object. ")

    # 返回加载的 Flax 权重到 PyTorch 模型
    return load_flax_weights_in_pytorch_model(pt_model, flax_state)

# 从 Flax 状态加载权重到 PyTorch 模型
def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """将 Flax 检查点加载到 PyTorch 模型中"""

    # 尝试导入 PyTorch
    try:
        import torch  # noqa: F401
    # 捕获导入错误
    except ImportError:
        # 记录错误信息，提示需要安装 PyTorch 和 Flax
        logger.error(
            "Loading Flax weights in PyTorch requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
            " instructions."
        )
        # 抛出异常
        raise

    # 检查是否存在 bf16 权重
    is_type_bf16 = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype == jnp.bfloat16, flax_state)).values()
    # 如果存在 bf16 类型的权重
    if any(is_type_bf16):
        # 如果权重是 bf16 类型，转换为 fp32，因为 torch.from_numpy 无法处理 bf16
        
        # 而且 bf16 在 PyTorch 中尚未完全支持。
        logger.warning(
            "Found ``bfloat16`` weights in Flax model. Casting all ``bfloat16`` weights to ``float32`` "
            "before loading those in PyTorch model."
        )
        # 使用 tree_map 遍历 flax_state，将 bf16 权重转换为 float32
        flax_state = jax.tree_util.tree_map(
            lambda params: params.astype(np.float32) if params.dtype == jnp.bfloat16 else params, flax_state
        )

    # 将基础模型前缀设为空
    pt_model.base_model_prefix = ""

    # 将 flax_state 字典扁平化，使用 "." 作为分隔符
    flax_state_dict = flatten_dict(flax_state, sep=".")
    # 获取 PyTorch 模型的状态字典
    pt_model_dict = pt_model.state_dict()

    # 记录意外和缺失的键
    unexpected_keys = []  # 存储意外键
    missing_keys = set(pt_model_dict.keys())  # 存储缺失键的集合

    # 遍历 flax_state_dict 中的每个键值对
    for flax_key_tuple, flax_tensor in flax_state_dict.items():
        # 将键元组转换为数组形式
        flax_key_tuple_array = flax_key_tuple.split(".")

        # 如果键的最后一个元素是 "kernel" 且张量维度为 4
        if flax_key_tuple_array[-1] == "kernel" and flax_tensor.ndim == 4:
            # 将最后一个元素替换为 "weight"，并调整张量的维度顺序
            flax_key_tuple_array = flax_key_tuple_array[:-1] + ["weight"]
            flax_tensor = jnp.transpose(flax_tensor, (3, 2, 0, 1))
        # 如果键的最后一个元素是 "kernel"
        elif flax_key_tuple_array[-1] == "kernel":
            # 将最后一个元素替换为 "weight"，并转置张量
            flax_key_tuple_array = flax_key_tuple_array[:-1] + ["weight"]
            flax_tensor = flax_tensor.T
        # 如果键的最后一个元素是 "scale"
        elif flax_key_tuple_array[-1] == "scale":
            # 将最后一个元素替换为 "weight"
            flax_key_tuple_array = flax_key_tuple_array[:-1] + ["weight"]

        # 如果键数组中不包含 "time_embedding"
        if "time_embedding" not in flax_key_tuple_array:
            # 遍历键数组，替换下划线为点
            for i, flax_key_tuple_string in enumerate(flax_key_tuple_array):
                flax_key_tuple_array[i] = (
                    flax_key_tuple_string.replace("_0", ".0")
                    .replace("_1", ".1")
                    .replace("_2", ".2")
                    .replace("_3", ".3")
                    .replace("_4", ".4")
                    .replace("_5", ".5")
                    .replace("_6", ".6")
                    .replace("_7", ".7")
                    .replace("_8", ".8")
                    .replace("_9", ".9")
                )

        # 将键数组重新连接为字符串
        flax_key = ".".join(flax_key_tuple_array)

        # 如果当前键在 PyTorch 模型的字典中
        if flax_key in pt_model_dict:
            # 如果权重形状不匹配，抛出错误
            if flax_tensor.shape != pt_model_dict[flax_key].shape:
                raise ValueError(
                    f"Flax checkpoint seems to be incorrect. Weight {flax_key_tuple} was expected "
                    f"to be of shape {pt_model_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )
            else:
                # 将权重添加到 PyTorch 字典中
                flax_tensor = np.asarray(flax_tensor) if not isinstance(flax_tensor, np.ndarray) else flax_tensor
                pt_model_dict[flax_key] = torch.from_numpy(flax_tensor)
                # 从缺失键中移除当前键
                missing_keys.remove(flax_key)
        else:
            # 权重不是 PyTorch 模型所期望的
            unexpected_keys.append(flax_key)

    # 将状态字典加载到 PyTorch 模型中
    pt_model.load_state_dict(pt_model_dict)

    # 将缺失键重新转换为列表
    # 将 missing_keys 转换为列表，以便后续处理
    missing_keys = list(missing_keys)

    # 检查 unexpected_keys 的长度，如果大于 0，表示有未使用的权重
    if len(unexpected_keys) > 0:
        # 记录警告信息，提示某些权重未被使用
        logger.warning(
            "Some weights of the Flax model were not used when initializing the PyTorch model"
            f" {pt_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {pt_model.__class__.__name__} from a Flax model trained on another task or with another architecture"
            " (e.g. initializing a BertForSequenceClassification model from a FlaxBertForPreTraining model).\n- This"
            f" IS NOT expected if you are initializing {pt_model.__class__.__name__} from a Flax model that you expect"
            " to be exactly identical (e.g. initializing a BertForSequenceClassification model from a"
            " FlaxBertForSequenceClassification model)."
        )
    # 检查 missing_keys 的长度，如果大于 0，表示有权重未被初始化
    if len(missing_keys) > 0:
        # 记录警告信息，提示某些权重是新初始化的
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the Flax model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )

    # 返回初始化后的 PyTorch 模型
    return pt_model
```