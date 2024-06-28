# `.\modeling_tf_pytorch_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明，分别归属于 Google AI Language Team 和 HuggingFace Inc. 团队以及 NVIDIA 公司
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证要求，否则禁止使用此文件
# 可以在以下链接找到完整的许可证文本：http://www.apache.org/licenses/LICENSE-2.0
#
# 如果不符合适用法律或未经书面同意，软件将按“原样”分发，没有任何形式的担保或条件
# 详见许可证以了解更多信息

""" PyTorch - TF 2.0 通用实用工具 """

# 导入必要的库
import os
import re

import numpy  # 导入 numpy 库

# 从本地模块中导入以下工具函数
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义枚举类 TransposeType，表示转置类型的枚举值
class TransposeType(ExplicitEnum):
    """
    可能的...
    """
    NO = "no"
    SIMPLE = "simple"
    CONV1D = "conv1d"
    CONV2D = "conv2d"

# 定义函数 convert_tf_weight_name_to_pt_weight_name，将 TF 2.0 模型变量名转换为 PyTorch 模型权重名
def convert_tf_weight_name_to_pt_weight_name(
    tf_name, start_prefix_to_remove="", tf_weight_shape=None, name_scope=None
):
    """
    将 TF 2.0 模型变量名转换为 PyTorch 模型权重名。

    TF2.0 范围 -> PyTorch 属性名转换的约定：

        - '$1___$2' 被 $2 替换（可用于在 TF2.0 vs PyTorch 中复制或删除层）
        - '_._' 被新的级别分隔替换（可用于在 PyTorch nn.ModulesList 中转换 TF2.0 列表）

    返回一个元组，包含：

        - PyTorch 模型权重名
        - transpose：表示 TF2.0 和 PyTorch 权重矩阵之间是否以及如何进行转置的 `TransposeType` 成员
    """
    if name_scope is not None:
        if not tf_name.startswith(name_scope) and "final_logits_bias" not in tf_name:
            raise ValueError(
                f"Weight name {tf_name} does not start with name_scope {name_scope}. This is an internal error "
                "in Transformers, so (unless you were doing something really evil) please open an issue to report it!"
            )
        tf_name = tf_name[len(name_scope) :]
        tf_name = tf_name.lstrip("/")
    tf_name = tf_name.replace(":0", "")  # 移除设备 ID
    tf_name = re.sub(
        r"/[^/]*___([^/]*)/", r"/\1/", tf_name
    )  # '$1___$2' 被 $2 替换（可用于在 TF2.0 vs PyTorch 中复制或删除层）
    tf_name = tf_name.replace(
        "_._", "/"
    )  # '_._' 被级别分隔符替换（可用于在 PyTorch nn.ModulesList 中转换 TF2.0 列表）
    tf_name = re.sub(r"//+", "/", tf_name)  # 移除末尾的空级别
    tf_name = tf_name.split("/")  # 从 TF2.0 '/' 分隔符转换为 PyTorch '.' 分隔符
    # 检查 TensorFlow 权重名是否为多层级结构，如 BART 中的 final_logits_bias
    if len(tf_name) > 1:
        # 如果是多层级结构，移除第一层级的名称
        tf_name = tf_name[1:]  # Remove level zero
    
    # 将 TensorFlow 权重形状转换为列表形式
    tf_weight_shape = list(tf_weight_shape)
    
    # 判断是否需要转置权重
    if tf_name[-1] == "kernel" and tf_weight_shape is not None and len(tf_weight_shape) == 4:
        # 如果权重名称以 "kernel" 结尾且形状为四维，则选择转置类型为 CONV2D
        transpose = TransposeType.CONV2D
    elif tf_name[-1] == "kernel" and tf_weight_shape is not None and len(tf_weight_shape) == 3:
        # 如果权重名称以 "kernel" 结尾且形状为三维，则选择转置类型为 CONV1D
        transpose = TransposeType.CONV1D
    elif bool(
        tf_name[-1] in ["kernel", "pointwise_kernel", "depthwise_kernel"]
        or "emb_projs" in tf_name
        or "out_projs" in tf_name
    ):
        # 如果权重名称以 "kernel", "pointwise_kernel", "depthwise_kernel" 结尾，或者包含 "emb_projs" 或 "out_projs"，
        # 则选择转置类型为 SIMPLE
        transpose = TransposeType.SIMPLE
    else:
        # 否则，选择不进行转置
        transpose = TransposeType.NO
    
    # 将标准的 TensorFlow 2.0 权重名称转换为 PyTorch 权重名称
    if tf_name[-1] == "kernel" or tf_name[-1] == "embeddings" or tf_name[-1] == "gamma":
        tf_name[-1] = "weight"
    if tf_name[-1] == "beta":
        tf_name[-1] = "bias"
    
    # 对于 SeparableConv1D TF 层，将两个权重转换为 PyTorch Conv1D 的形式
    if tf_name[-1] == "pointwise_kernel" or tf_name[-1] == "depthwise_kernel":
        tf_name[-1] = tf_name[-1].replace("_kernel", ".weight")
    
    # 将列表形式的名称拼接为字符串形式
    tf_name = ".".join(tf_name)
    
    # 如果需要移除前缀，则移除指定的前缀
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, "", 1)
    
    # 返回转换后的 PyTorch 权重名称和转置类型
    return tf_name, transpose
def apply_transpose(transpose: TransposeType, weight, match_shape=None, pt_to_tf=True):
    """
    Apply a transpose operation to a weight tensor and optionally reshape it to match a target shape, in a framework-agnostic manner.
    """
    # 根据 transpose 类型选择不同的转置方式
    if transpose is TransposeType.CONV2D:
        # Conv2D 权重转置说明：
        #    PT: (num_out_channel, num_in_channel, kernel[0], kernel[1])
        # -> TF: (kernel[0], kernel[1], num_in_channel, num_out_channel)
        axes = (2, 3, 1, 0) if pt_to_tf else (3, 2, 0, 1)
        weight = transpose_func(weight, axes=axes)
    elif transpose is TransposeType.CONV1D:
        # Conv1D 权重转置说明：
        #    PT: (num_out_channel, num_in_channel, kernel)
        # -> TF: (kernel, num_in_channel, num_out_channel)
        weight = transpose_func(weight, axes=(2, 1, 0))
    elif transpose is TransposeType.SIMPLE:
        # 简单转置操作
        weight = transpose_func(weight)

    # 如果没有指定匹配的形状，直接返回转置后的权重
    if match_shape is None:
        return weight

    # 调整权重的形状以匹配目标形状
    if len(match_shape) < len(weight.shape):
        weight = squeeze(weight)  # 如果目标形状的维度少于当前权重的维度，则进行压缩操作
    elif len(match_shape) > len(weight.shape):
        weight = expand_dims(weight, axis=0)  # 如果目标形状的维度多于当前权重的维度，则在指定轴上扩展维度

    # 如果权重的形状与目标形状不匹配，则尝试重新调整形状
    if list(match_shape) != list(weight.shape):
        try:
            weight = reshape(weight, match_shape)  # 重新调整权重的形状为目标形状
        except AssertionError as e:
            e.args += (match_shape, match_shape)
            raise e  # 抛出异常

    return weight


#####################
# PyTorch => TF 2.0 #
#####################


def load_pytorch_checkpoint_in_tf2_model(
    tf_model,
    pytorch_checkpoint_path,
    tf_inputs=None,
    allow_missing_keys=False,
    output_loading_info=False,
    _prefix=None,
    tf_to_pt_weight_rename=None,
):
    """Load pytorch checkpoints into a TF 2.0 model"""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
        from safetensors.torch import load_file as safe_load_file  # noqa: F401

        from .pytorch_utils import is_torch_greater_or_equal_than_1_13  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a PyTorch model in TensorFlow requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 将单个文件路径处理为一个包含单个片段的集合
    if isinstance(pytorch_checkpoint_path, str):
        pytorch_checkpoint_path = [pytorch_checkpoint_path]

    # 将所有片段加载到单个状态字典中
    pt_state_dict = {}
    for path in pytorch_checkpoint_path:
        pt_path = os.path.abspath(path)
        logger.info(f"Loading PyTorch weights from {pt_path}")
        # 根据文件后缀选择加载方式
        if pt_path.endswith(".safetensors"):
            state_dict = safe_load_file(pt_path)
        else:
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            state_dict = torch.load(pt_path, map_location="cpu", **weights_only_kwarg)

        pt_state_dict.update(state_dict)
    # 使用日志记录器输出 PyTorch 检查点中包含的参数总数，格式化为千位分隔的字符串
    logger.info(f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters")
    
    # 调用函数，将 PyTorch 模型权重加载到 TensorFlow 2 模型中
    return load_pytorch_weights_in_tf2_model(
        tf_model,                     # TensorFlow 2 模型对象
        pt_state_dict,                # PyTorch 模型的状态字典
        tf_inputs=tf_inputs,          # 可选参数：传递给 TensorFlow 加载函数的输入
        allow_missing_keys=allow_missing_keys,  # 可选参数：允许缺失的键
        output_loading_info=output_loading_info,  # 可选参数：控制加载过程中的信息输出
        _prefix=_prefix,              # 可选参数：加载时的前缀
        tf_to_pt_weight_rename=tf_to_pt_weight_rename,  # 可选参数：重命名 TensorFlow 到 PyTorch 权重的映射
    )
# 载入 PyTorch 模型权重到 TensorFlow 2.0 模型
def load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=None, allow_missing_keys=False):
    """Load pytorch checkpoints in a TF 2.0 model"""
    # 获取 PyTorch 模型的状态字典
    pt_state_dict = pt_model.state_dict()

    # 调用函数载入 PyTorch 权重到 TensorFlow 模型
    return load_pytorch_weights_in_tf2_model(
        tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys
    )


# 载入 PyTorch 状态字典到 TensorFlow 2.0 模型
def load_pytorch_weights_in_tf2_model(
    tf_model,
    pt_state_dict,
    tf_inputs=None,
    allow_missing_keys=False,
    output_loading_info=False,
    _prefix=None,
    tf_to_pt_weight_rename=None,
):
    """Load pytorch state_dict in a TF 2.0 model."""
    try:
        import tensorflow as tf  # 导入 TensorFlow 库
        import torch  # 导入 PyTorch 库
    except ImportError:
        # 若导入失败，输出错误信息并抛出异常
        logger.error(
            "Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 将 PyTorch 状态字典中的张量转换为 NumPy 数组
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}
    # 调用函数加载 PyTorch 状态字典到 TensorFlow 模型
    return load_pytorch_state_dict_in_tf2_model(
        tf_model,
        pt_state_dict,
        tf_inputs=tf_inputs,
        allow_missing_keys=allow_missing_keys,
        output_loading_info=output_loading_info,
        _prefix=_prefix,
        tf_to_pt_weight_rename=tf_to_pt_weight_rename,
    )


# 加载 PyTorch 状态字典到 TensorFlow 2.0 模型
def load_pytorch_state_dict_in_tf2_model(
    tf_model,
    pt_state_dict,
    tf_inputs=None,
    allow_missing_keys=False,
    output_loading_info=False,
    _prefix=None,
    tf_to_pt_weight_rename=None,
    ignore_mismatched_sizes=False,
):
    """Load a pytorch state_dict in a TF 2.0 model. pt_state_dict can be either an actual dict or a lazy-loading
    safetensors archive created with the safe_open() function."""
    import tensorflow as tf

    # 如果未指定输入数据，使用模型的虚拟输入
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    # 如果未指定前缀，设为空字符串
    if _prefix is None:
        _prefix = ""

    # 如果有输入数据，确保模型已构建
    if tf_inputs:
        with tf.name_scope(_prefix):
            tf_model(tf_inputs, training=False)  # 确保模型已构建

    # 转换从 TensorFlow 键到 PyTorch 键的映射
    tf_keys_to_pt_keys = {}
    # 遍历输入字典的键
    for key in pt_state_dict.keys():
        new_key = None
        # 如果键名中包含 "gamma"，替换为 "weight"
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        # 如果键名中包含 "beta"，替换为 "bias"
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        # 如果键名中包含 "running_var"，替换为 "moving_variance"
        if "running_var" in key:
            new_key = key.replace("running_var", "moving_variance")
        # 如果键名中包含 "running_mean"，替换为 "moving_mean"
        if "running_mean" in key:
            new_key = key.replace("running_mean", "moving_mean")

        # 处理新的 `weight_norm` 命名，来源于 https://github.com/huggingface/transformers/pull/24030
        key_components = key.split(".")
        name = None
        # 检查键名的特定模式，根据模式生成新的命名
        if key_components[-3::2] == ["parametrizations", "original0"]:
            name = key_components[-2] + "_g"
        elif key_components[-3::2] == ["parametrizations", "original1"]:
            name = key_components[-2] + "_v"
        if name is not None:
            key_components = key_components[:-3] + [name]
            new_key = ".".join(key_components)

        # 如果没有匹配到任何替换规则，保持原来的键名不变
        if new_key is None:
            new_key = key
        # 将新旧键名的对应关系存入字典
        tf_keys_to_pt_keys[new_key] = key

    # Matt: 所有 TF 模型都在 MainLayer 类中存储实际模型，包括基础模型。
    # 在 PT 中，派生模型（带头部的模型）使用基础模型类作为主干，没有 MainLayer 类。
    # 这意味着 TF 基础模型的权重名中有一个额外的层级，对应于 MainLayer 类。
    # 以下代码块用于补偿这一差异。
    
    # 如果没有任何 TF 键名以 tf_model.base_model_prefix 开头，则需要移除的前缀为 tf_model.base_model_prefix + "."
    start_prefix_to_remove = ""
    if not any(s.startswith(tf_model.base_model_prefix) for s in tf_keys_to_pt_keys.keys()):
        start_prefix_to_remove = tf_model.base_model_prefix + "."

    # 获取 TF 模型的所有符号权重（可训练和不可训练的）
    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    # 初始化 TF 加载的权重数目
    tf_loaded_numel = 0
    # 获取所有 PyTorch 键名的集合
    all_pytorch_weights = set(tf_keys_to_pt_keys.keys())
    # 存储缺失的键名列表
    missing_keys = []
    # 存储不匹配的键名列表
    mismatched_keys = []
    # 检查 pt_state_dict 是否具有 "get_tensor" 方法，用于确定是否为 SafeTensor 存档
    is_safetensor_archive = hasattr(pt_state_dict, "get_tensor")
    # 遍历符号权重列表中的每个符号权重对象
    for symbolic_weight in symbolic_weights:
        # 获取当前符号权重的名称
        sw_name = symbolic_weight.name
        # 将 TensorFlow 的权重名称转换为 PyTorch 的权重名称，并获取转换后的名称及是否需要转置的信息
        name, transpose = convert_tf_weight_name_to_pt_weight_name(
            sw_name,
            start_prefix_to_remove=start_prefix_to_remove,
            tf_weight_shape=symbolic_weight.shape,
            name_scope=_prefix,
        )
        
        # 如果指定了 TensorFlow 到 PyTorch 权重重命名函数，则使用它来获取可能的别名
        if tf_to_pt_weight_rename is not None:
            aliases = tf_to_pt_weight_rename(name)  # 返回一个元组以处理可能的名称别名
            # 遍历别名列表，按优先顺序使用第一个匹配的别名
            for alias in aliases:
                if alias in tf_keys_to_pt_keys:
                    name = alias
                    break
            else:
                # 如果没有别名匹配，使用列表中的第一个名称（将被报告为缺失）
                name = aliases[0]

        # 在 PyTorch 模型状态字典中查找对应的 NumPy 数组
        if name not in tf_keys_to_pt_keys:
            # 如果允许缺失键，则将名称添加到缺失键列表中并继续下一个符号权重
            if allow_missing_keys:
                missing_keys.append(name)
                continue
            # 如果定义了可以在加载时忽略的键列表，则根据列表判断是否需要忽略当前键
            elif tf_model._keys_to_ignore_on_load_missing is not None:
                if any(re.search(pat, name) is not None for pat in tf_model._keys_to_ignore_on_load_missing):
                    continue
                # 如果不符合忽略条件，则抛出异常，指出在 PyTorch 模型中找不到该键
            raise AttributeError(f"{name} not found in PyTorch model")
        
        # 获取 PyTorch 模型状态字典中对应键的数组
        state_dict_name = tf_keys_to_pt_keys[name]
        # 如果是安全张量归档模式，则从 PyTorch 状态字典中获取张量
        if is_safetensor_archive:
            array = pt_state_dict.get_tensor(state_dict_name)
        else:
            array = pt_state_dict[state_dict_name]
        
        # 尝试将数组按照转置信息应用到符号权重的形状上
        try:
            array = apply_transpose(transpose, array, symbolic_weight.shape)
        except tf.errors.InvalidArgumentError as e:
            # 如果出现尺寸不匹配的错误，并且不忽略尺寸不匹配，则抛出异常
            if not ignore_mismatched_sizes:
                error_msg = str(e)
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
                raise tf.errors.InvalidArgumentError(error_msg)
            else:
                # 否则将不匹配的键和形状添加到不匹配键列表中并继续下一个符号权重
                mismatched_keys.append((name, array.shape, symbolic_weight.shape))
                continue
        
        # 计算加载的 TensorFlow 权重的元素数量并累加到 tf_loaded_numel 中
        tf_loaded_numel += tensor_size(array)
        
        # 将 PyTorch 数组转换为符号权重的数据类型，并分配给符号权重对象
        symbolic_weight.assign(tf.cast(array, symbolic_weight.dtype))
        # 立即释放数组以尽可能保持内存使用低峰
        del array
        # 从所有 PyTorch 权重集合中移除当前处理的键
        all_pytorch_weights.discard(name)

    # 记录加载了多少个参数到 TF 2.0 模型中
    logger.info(f"Loaded {tf_loaded_numel:,} parameters in the TF 2.0 model.")

    # 将未预期的键列表转换为列表形式
    unexpected_keys = list(all_pytorch_weights)

    # 如果定义了在加载时忽略的缺失键列表，则根据列表中的模式匹配规则进行过滤
    if tf_model._keys_to_ignore_on_load_missing is not None:
        for pat in tf_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
    
    # 如果定义了在加载时忽略的未预期键列表，则根据列表中的模式匹配规则进行过滤
    if tf_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in tf_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
    # 如果存在未预期的键（权重），记录警告信息到日志
    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the PyTorch model were not used when initializing the TF 2.0 model"
            f" {tf_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {tf_model.__class__.__name__} from a PyTorch model trained on another task or with another architecture"
            " (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).\n- This IS"
            f" NOT expected if you are initializing {tf_model.__class__.__name__} from a PyTorch model that you expect"
            " to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a"
            " BertForSequenceClassification model)."
        )
    else:
        # 如果所有 PyTorch 模型的权重都被使用，记录相应信息到日志
        logger.warning(f"All PyTorch model weights were used when initializing {tf_model.__class__.__name__}.\n")
    
    # 如果存在未初始化的键（权重或缓冲区），记录警告信息到日志
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights or buffers of the TF 2.0 model {tf_model.__class__.__name__} were not initialized from the"
            f" PyTorch model and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a"
            " down-stream task to be able to use it for predictions and inference."
        )
    else:
        # 如果所有权重都从 PyTorch 模型初始化，记录相应信息到日志
        logger.warning(
            f"All the weights of {tf_model.__class__.__name__} were initialized from the PyTorch model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {tf_model.__class__.__name__} for predictions without further training."
        )
    
    # 如果存在形状不匹配的键，生成对应的警告信息，并记录到日志
    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        logger.warning(
            f"Some weights of {tf_model.__class__.__name__} were not initialized from the model checkpoint"
            f" are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
            " to use it for predictions and inference."
        )
    
    # 如果需要输出加载信息，返回 TensorFlow 模型及加载信息
    if output_loading_info:
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
        }
        return tf_model, loading_info
    
    # 返回加载后的 TensorFlow 模型
    return tf_model
#####################
# TF 2.0 => PyTorch #
#####################

# 在 PyTorch 模型中加载 TF 2.0 的检查点
def load_tf2_checkpoint_in_pytorch_model(
    pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False, output_loading_info=False
):
    """
    Load TF 2.0 HDF5 checkpoint in a PyTorch model We use HDF5 to easily do transfer learning (see
    https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    import transformers

    from .modeling_tf_utils import load_tf_weights

    logger.info(f"Loading TensorFlow weights from {tf_checkpoint_path}")

    # 实例化并加载相关的 TF 2.0 模型
    tf_model_class_name = "TF" + pt_model.__class__.__name__  # 在类名前加上 "TF"
    tf_model_class = getattr(transformers, tf_model_class_name)
    tf_model = tf_model_class(pt_model.config)

    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # 确保模型已构建

    load_tf_weights(tf_model, tf_checkpoint_path)

    return load_tf2_model_in_pytorch_model(
        pt_model, tf_model, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info
    )


# 在 PyTorch 模型中加载 TF 2.0 模型
def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=False, output_loading_info=False):
    """Load TF 2.0 model in a pytorch model"""
    weights = tf_model.weights

    return load_tf2_weights_in_pytorch_model(
        pt_model, weights, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info
    )


# 在 PyTorch 模型中加载 TF 2.0 权重
def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False, output_loading_info=False):
    """Load TF2.0 symbolic weights in a PyTorch model"""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 将 TF 2.0 的权重转换为字典形式
    tf_state_dict = {tf_weight.name: tf_weight.numpy() for tf_weight in tf_weights}
    return load_tf2_state_dict_in_pytorch_model(
        pt_model, tf_state_dict, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info
    )


# 在 PyTorch 模型中加载 TF 2.0 的状态字典
def load_tf2_state_dict_in_pytorch_model(pt_model, tf_state_dict, allow_missing_keys=False, output_loading_info=False):
    import torch

    new_pt_params_dict = {}
    # 获取当前 PyTorch 模型的所有命名参数，并转换成字典形式
    current_pt_params_dict = dict(pt_model.named_parameters())

    # 确保能够加载 PyTorch 基础模型和派生模型（带有头部）
    # TF 模型总是有一个前缀，而一些 PyTorch 基础模型则没有
    start_prefix_to_remove = ""
    if not any(s.startswith(pt_model.base_model_prefix) for s in current_pt_params_dict.keys()):
        start_prefix_to_remove = pt_model.base_model_prefix + "."

    # 构建一个从潜在的 PyTorch 权重名称到 TF 2.0 变量的映射
    tf_weights_map = {}
    for name, tf_weight in tf_state_dict.items():
        # 转换 TF 的权重名称到 PyTorch 的权重名称
        pt_name, transpose = convert_tf_weight_name_to_pt_weight_name(
            name, start_prefix_to_remove=start_prefix_to_remove, tf_weight_shape=tf_weight.shape
        )
        tf_weights_map[pt_name] = (tf_weight, transpose)

    # 获取所有 TF 权重名称的集合
    all_tf_weights = set(tf_weights_map.keys())
    
    # 用于存储已加载的 PyTorch 权重数据指针的字典
    loaded_pt_weights_data_ptr = {}
    
    # 存储缺失的 PyTorch 键列表
    missing_keys_pt = []

    # 遍历当前 PyTorch 模型的所有参数
    for pt_weight_name, pt_weight in current_pt_params_dict.items():
        # 处理 PyTorch 共享权重（在 TF 2.0 中不重复）
        if pt_weight.data_ptr() in loaded_pt_weights_data_ptr:
            new_pt_params_dict[pt_weight_name] = loaded_pt_weights_data_ptr[pt_weight.data_ptr()]
            continue

        # 准备用于检查的 PyTorch 权重名称
        pt_weight_name_to_check = pt_weight_name
        
        # 处理新的 `weight_norm`（来自 https://github.com/huggingface/transformers/pull/24030）
        key_components = pt_weight_name.split(".")
        name = None
        if key_components[-3::2] == ["parametrizations", "original0"]:
            name = key_components[-2] + "_g"
        elif key_components[-3::2] == ["parametrizations", "original1"]:
            name = key_components[-2] + "_v"
        if name is not None:
            key_components = key_components[:-3] + [name]
            pt_weight_name_to_check = ".".join(key_components)

        # 检查 PyTorch 权重名称是否在 TF 2.0 权重映射中
        if pt_weight_name_to_check not in tf_weights_map:
            # 如果允许缺失的键，则将其添加到缺失的 PyTorch 键列表中
            if allow_missing_keys:
                missing_keys_pt.append(pt_weight_name)
                continue

            # 否则，抛出属性错误，指明找不到对应的 TF 2.0 模型的键
            raise AttributeError(f"{pt_weight_name} not found in TF 2.0 model")

        # 获取对应的 numpy 数组和转置信息
        array, transpose = tf_weights_map[pt_weight_name_to_check]

        # 应用转置（如果需要），将 TF 数组转换为 PyTorch 数组
        array = apply_transpose(transpose, array, pt_weight.shape, pt_to_tf=False)

        # 如果数组是标量，转换为 numpy 数组
        if numpy.isscalar(array):
            array = numpy.array(array)
        # 如果不是 torch 张量也不是 numpy 数组，则假定为 numpy 数组并转换为 torch 张量
        if not is_torch_tensor(array) and not is_numpy_array(array):
            array = array.numpy()
        if is_numpy_array(array):
            # 转换为 torch 张量
            array = torch.from_numpy(array)

        # 将转换后的数组存储到新的 PyTorch 参数字典中
        new_pt_params_dict[pt_weight_name] = array
        # 将已加载的 PyTorch 权重数据指针存储到字典中，以避免重复加载
        loaded_pt_weights_data_ptr[pt_weight.data_ptr()] = array
        # 从所有 TF 权重集合中移除当前处理的 PyTorch 权重名称
        all_tf_weights.discard(pt_weight_name)

    # 使用新的 PyTorch 参数字典加载模型状态，允许缺失的键
    missing_keys, unexpected_keys = pt_model.load_state_dict(new_pt_params_dict, strict=False)
    # 将缺失的 PyTorch 键列表添加到总的缺失键列表中
    missing_keys += missing_keys_pt
    # 如果模型定义了要在加载时忽略的键，将这些键从缺失键列表中移除，避免不必要地向用户发出警告。
    if pt_model._keys_to_ignore_on_load_missing is not None:
        for pat in pt_model._keys_to_ignore_on_load_missing:
            # 使用正则表达式模式匹配并移除缺失键列表中与模式匹配的键
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    # 如果模型定义了要在加载时忽略的意外键，将这些键从意外键列表中移除，同样避免不必要的警告。
    if pt_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in pt_model._keys_to_ignore_on_load_unexpected:
            # 使用正则表达式模式匹配并移除意外键列表中与模式匹配的键
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    # 如果存在未使用的权重（意外键），向日志记录警告信息，说明这在某些情况下是预期的，比如模型从不同任务或架构的 TF 2.0 模型初始化时。
    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the TF 2.0 model were not used when initializing the PyTorch model"
            f" {pt_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {pt_model.__class__.__name__} from a TF 2.0 model trained on another task or with another architecture"
            " (e.g. initializing a BertForSequenceClassification model from a TFBertForPreTraining model).\n- This IS"
            f" NOT expected if you are initializing {pt_model.__class__.__name__} from a TF 2.0 model that you expect"
            " to be exactly identical (e.g. initializing a BertForSequenceClassification model from a"
            " TFBertForSequenceClassification model)."
        )
    else:
        # 如果没有未使用的权重，向日志记录警告信息，说明所有 TF 2.0 模型权重都已使用。
        logger.warning(f"All TF 2.0 model weights were used when initializing {pt_model.__class__.__name__}.\n")

    # 如果存在未初始化的权重（缺失键），向日志记录警告信息，建议用户在下游任务上训练模型以便进行预测和推断。
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the TF 2.0 model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )
    else:
        # 如果没有未初始化的权重，向日志记录警告信息，说明所有权重都已从 TF 2.0 模型初始化。
        logger.warning(
            f"All the weights of {pt_model.__class__.__name__} were initialized from the TF 2.0 model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {pt_model.__class__.__name__} for predictions without further training."
        )

    # 向日志记录加载信息，显示哪些 TF 2.0 模型的权重或缓冲区未加载。
    logger.info(f"Weights or buffers not loaded from TF 2.0 model: {all_tf_weights}")

    # 如果需要输出加载信息，返回模型及加载信息的字典。
    if output_loading_info:
        loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}
        return pt_model, loading_info

    # 否则，只返回加载后的 PyTorch 模型。
    return pt_model
```