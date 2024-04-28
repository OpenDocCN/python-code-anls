# `.\transformers\modeling_tf_pytorch_utils.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 Google AI Language Team 作者和 HuggingFace Inc. 团队所有，以及 NVIDIA 公司所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" PyTorch - TF 2.0 通用工具。"""

# 导入必要的库
import os
import re
import numpy

# 从 utils 模块中导入所需的函数和类
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义一个枚举类，表示转置类型
class TransposeType(ExplicitEnum):
    """
    可能的...
    """
    NO = "no"
    SIMPLE = "simple"
    CONV1D = "conv1d"
    CONV2D = "conv2d"

# 将 TF 2.0 模型变量名称转换为 PyTorch 模型权重名称
def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove="", tf_weight_shape=None, name_scope=None):
    """
    Convert a TF 2.0 model variable name in a pytorch model weight name.

    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:

        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
        - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

    return tuple with:

        - pytorch model weight name
        - transpose: `TransposeType` member indicating whether and how TF2.0 and PyTorch weights matrices should be
          transposed with regards to each other
    """
    # 如果指定了名称范围，则检查是否以名称范围开头，否则引发异常
    if name_scope is not None:
        if not tf_name.startswith(name_scope) and "final_logits_bias" not in tf_name:
            raise ValueError(
                f"Weight name {tf_name} does not start with name_scope {name_scope}. This is an internal error "
                "in Transformers, so (unless you were doing something really evil) please open an issue to report it!"
            )
        tf_name = tf_name[len(name_scope) :]
        tf_name = tf_name.lstrip("/")
    tf_name = tf_name.replace(":0", "")  # 移除设备 id
    tf_name = re.sub(
        r"/[^/]*___([^/]*)/", r"/\1/", tf_name
    )  # '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
    tf_name = tf_name.replace(
        "_._", "/"
    )  # '_._' is replaced by a level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)
    tf_name = re.sub(r"//+", "/", tf_name)  # 移除末尾的空级别
    tf_name = tf_name.split("/")  # 将 TF2.0 的 '/' 分隔符转换为 PyTorch 的 '.' 分隔符
    # 如果 TensorFlow 的权重名称长度大于1，则去掉第一个元素，即去掉最外层的名称
    if len(tf_name) > 1:
        tf_name = tf_name[1:]  # Remove level zero

    # 将 TensorFlow 的权重形状转换为列表形式
    tf_weight_shape = list(tf_weight_shape)

    # 判断何时需要转置权重
    if tf_name[-1] == "kernel" and tf_weight_shape is not None and len(tf_weight_shape) == 4:
        transpose = TransposeType.CONV2D
    elif tf_name[-1] == "kernel" and tf_weight_shape is not None and len(tf_weight_shape) == 3:
        transpose = TransposeType.CONV1D
    elif bool(
        tf_name[-1] in ["kernel", "pointwise_kernel", "depthwise_kernel"]
        or "emb_projs" in tf_name
        or "out_projs" in tf_name
    ):
        transpose = TransposeType.SIMPLE
    else:
        transpose = TransposeType.NO

    # 将标准的 TensorFlow 2.0 名称转换为 PyTorch 名称
    if tf_name[-1] == "kernel" or tf_name[-1] == "embeddings" or tf_name[-1] == "gamma":
        tf_name[-1] = "weight"
    if tf_name[-1] == "beta":
        tf_name[-1] = "bias"

    # SeparableConv1D TF 层包含两个权重，这里将其转换为 PyTorch Conv1D
    if tf_name[-1] == "pointwise_kernel" or tf_name[-1] == "depthwise_kernel":
        tf_name[-1] = tf_name[-1].replace("_kernel", ".weight")

    # 如果需要，去掉前缀
    tf_name = ".".join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, "", 1)

    # 返回转换后的名称和转置类型
    return tf_name, transpose
def apply_transpose(transpose: TransposeType, weight, match_shape=None, pt_to_tf=True):
    """
    Apply a transpose to some weight then tries to reshape the weight to the same shape as a given shape, all in a
    framework agnostic way.
    """
    # 根据转置类型对权重进行转置，并根据需要重塑权重的形状
    if transpose is TransposeType.CONV2D:
        # Conv2D 权重:
        #    PT: (num_out_channel, num_in_channel, kernel[0], kernel[1])
        # -> TF: (kernel[0], kernel[1], num_in_channel, num_out_channel)
        axes = (2, 3, 1, 0) if pt_to_tf else (3, 2, 0, 1)
        weight = transpose_func(weight, axes=axes)
    elif transpose is TransposeType.CONV1D:
        # Conv1D 权重:
        #    PT: (num_out_channel, num_in_channel, kernel)
        # -> TF: (kernel, num_in_channel, num_out_channel)
        weight = transpose_func(weight, axes=(2, 1, 0))
    elif transpose is TransposeType.SIMPLE:
        # 简单转置
        weight = transpose_func(weight)

    if match_shape is None:
        # 若无需匹配形状，则直接返回权重
        return weight

    if len(match_shape) < len(weight.shape):
        # 如果目标形状的维数少于权重的维数，则压缩权重
        weight = squeeze(weight)
    elif len(match_shape) > len(weight.shape):
        # 如果目标形状的维数多于权重的维数，则在权重上添加新的维度
        weight = expand_dims(weight, axis=0)

    if list(match_shape) != list(weight.shape):
        # 若目标形状与权重形状不匹配，则尝试重塑权重
        try:
            weight = reshape(weight, match_shape)
        except AssertionError as e:
            # 如果重塑失败，则引发异常
            e.args += (match_shape, match_shape)
            raise e

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
    """Load pytorch checkpoints in a TF 2.0 model"""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
        from safetensors.torch import load_file as safe_load_file  # noqa: F401

        from .pytorch_utils import is_torch_greater_or_equal_than_1_13  # noqa: F401
    except ImportError:
        # 导入必要的库，如果失败，则引发 ImportError
        logger.error(
            "Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 将单个文件视为只有 1 个分片的集合
    if isinstance(pytorch_checkpoint_path, str):
        pytorch_checkpoint_path = [pytorch_checkpoint_path]

    # 将所有分片加载到单个状态字典中
    pt_state_dict = {}
    for path in pytorch_checkpoint_path:
        pt_path = os.path.abspath(path)
        logger.info(f"Loading PyTorch weights from {pt_path}")
        if pt_path.endswith(".safetensors"):
            state_dict = safe_load_file(pt_path)
        else:
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=is_torch_greater_or_equal_than_1_13)

        pt_state_dict.update(state_dict)
    # 使用 logger 记录 PyTorch 检查点中包含的参数数量
    logger.info(f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters")

    # 调用函数加载 PyTorch 权重到 TensorFlow 2 模型中
    return load_pytorch_weights_in_tf2_model(
        tf_model,  # TensorFlow 2 模型
        pt_state_dict,  # PyTorch 状态字典
        tf_inputs=tf_inputs,  # TensorFlow 输入
        allow_missing_keys=allow_missing_keys,  # 是否允许缺失键
        output_loading_info=output_loading_info,  # 输出加载信息
        _prefix=_prefix,  # 前缀
        tf_to_pt_weight_rename=tf_to_pt_weight_rename,  # TensorFlow 到 PyTorch 权重重命名
    )
# 加载 PyTorch 模型的权重到 TensorFlow 2.0 模型中
def load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=None, allow_missing_keys=False):
    """Load pytorch checkpoints in a TF 2.0 model"""
    # 获取 PyTorch 模型的状态字典
    pt_state_dict = pt_model.state_dict()

    # 调用函数加载 PyTorch 权重到 TensorFlow 模型中
    return load_pytorch_weights_in_tf2_model(
        tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys
    )


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
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        # 打印错误信息，要求安装 PyTorch 和 TensorFlow
        logger.error(
            "Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 将 PyTorch 的状态字典转换为 NumPy 数组
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}
    
    # 调用函数加载 PyTorch 权重到 TensorFlow 模型中
    return load_pytorch_state_dict_in_tf2_model(
        tf_model,
        pt_state_dict,
        tf_inputs=tf_inputs,
        allow_missing_keys=allow_missing_keys,
        output_loading_info=output_loading_info,
        _prefix=_prefix,
        tf_to_pt_weight_rename=tf_to_pt_weight_rename,
    )


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
    from keras import backend as K

    # 如果没有指定 TensorFlow 输入，则使用模型的虚拟输入
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    # 如果没有指定前缀，则设为空字符串
    if _prefix is None:
        _prefix = ""
    
    # 如果存在 TensorFlow 输入
    if tf_inputs:
        with tf.name_scope(_prefix):
            # 确保模型已构建
            tf_model(tf_inputs, training=False)  # Make sure model is built
    
    # 用于存储 TensorFlow 权重与 PyTorch 权重之间的映射关系
    tf_keys_to_pt_keys = {}
    # 遍历 PyTorch 状态字典中的键（模型权重的名称）
    for key in pt_state_dict.keys():
        # 初始化新键为 None
        new_key = None
        # 如果键中包含 "gamma"
        if "gamma" in key:
            # 将 "gamma" 替换为 "weight"，用于对应 PyTorch 模型的权重
            new_key = key.replace("gamma", "weight")
        # 如果键中包含 "beta"
        if "beta" in key:
            # 将 "beta" 替换为 "bias"，用于对应 PyTorch 模型的偏置
            new_key = key.replace("beta", "bias")
        # 如果键中包含 "running_var"
        if "running_var" in key:
            # 将 "running_var" 替换为 "moving_variance"，用于对应 PyTorch 模型的移动方差
            new_key = key.replace("running_var", "moving_variance")
        # 如果键中包含 "running_mean"
        if "running_mean" in key:
            # 将 "running_mean" 替换为 "moving_mean"，用于对应 PyTorch 模型的移动均值
            new_key = key.replace("running_mean", "moving_mean")
    
        # 从 TF 模型的权重名中提取新键名
        key_components = key.split(".")
        name = None
        # 如果 TF 模型的权重名符合某种模式
        if key_components[-3::2] == ["parametrizations", "original0"]:
            # 设置新键名为 "parametrizations" + "_g"
            name = key_components[-2] + "_g"
        # 如果 TF 模型的权重名符合另一种模式
        elif key_components[-3::2] == ["parametrizations", "original1"]:
            # 设置新键名为 "parametrizations" + "_v"
            name = key_components[-2] + "_v"
        # 如果新键名存在
        if name is not None:
            # 将键名中特定部分替换为新键名
            key_components = key_components[:-3] + [name]
            # 重新构建新键
            new_key = ".".join(key_components)
    
        # 如果新键名不存在
        if new_key is None:
            # 使用原键名作为新键名
            new_key = key
        # 将 TF 模型的键名与 PyTorch 模型的键名对应关系保存到字典中
        tf_keys_to_pt_keys[new_key] = key
    
    # 获取 TF 模型的基础模型前缀，用于后续处理
    start_prefix_to_remove = ""
    # 如果 TF 模型的键名中没有以基础模型前缀开头的键
    if not any(s.startswith(tf_model.base_model_prefix) for s in tf_keys_to_pt_keys.keys()):
        # 设置要移除的前缀为 TF 模型的基础模型前缀加上 "."
        start_prefix_to_remove = tf_model.base_model_prefix + "."
    
    # 提取 TF 模型的所有权重（包括可训练权重和不可训练权重）
    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    # 初始化 TF 加载的权重数量为 0
    tf_loaded_numel = 0
    # 将所有 PyTorch 模型的权重键名保存到集合中
    all_pytorch_weights = set(tf_keys_to_pt_keys.keys())
    # 初始化缺失的键列表
    missing_keys = []
    # 初始化不匹配的键列表
    mismatched_keys = []
    # 检查 PyTorch 状态字典是否为 SafeTensor 归档格式
    is_safetensor_archive = hasattr(pt_state_dict, "get_tensor")
    # 遍历 TensorFlow 模型中的符号权重
    for symbolic_weight in symbolic_weights:
        # 获取符号权重的名称
        sw_name = symbolic_weight.name
        # 将 TensorFlow 权重名称转换为 PyTorch 权重名称，并进行可能的转置
        name, transpose = convert_tf_weight_name_to_pt_weight_name(
            sw_name,
            start_prefix_to_remove=start_prefix_to_remove,
            tf_weight_shape=symbolic_weight.shape,
            name_scope=_prefix,
        )
        # 如果存在 TensorFlow 到 PyTorch 权重的重命名函数
        if tf_to_pt_weight_rename is not None:
            # 获取可能的重命名列表（元组形式以考虑可能的别名）
            aliases = tf_to_pt_weight_rename(name)
            # 遍历可能的别名，按优先顺序使用第一个匹配的别名
            for alias in aliases:
                if alias in tf_keys_to_pt_keys:
                    name = alias
                    break
            else:
                # 如果没有别名匹配，则使用第一个别名（会被报告为丢失）
                name = aliases[0]
    
        # 查找 PyTorch 模型状态字典中关联的 numpy 数组
        if name not in tf_keys_to_pt_keys:
            # 如果允许丢失的键，则将键添加到丢失键列表中并继续下一个循环
            if allow_missing_keys:
                missing_keys.append(name)
                continue
            # 如果存在可以忽略加载丢失的键的列表，则检查是否匹配并继续下一个循环
            elif tf_model._keys_to_ignore_on_load_missing is not None:
                if any(re.search(pat, name) is not None for pat in tf_model._keys_to_ignore_on_load_missing):
                    continue
            # 抛出属性错误，指明在 PyTorch 模型中找不到对应键
            raise AttributeError(f"{name} not found in PyTorch model")
        # 获取 PyTorch 模型状态字典中对应的数组
        state_dict_name = tf_keys_to_pt_keys[name]
        # 如果是安全张量存档，则从中获取张量
        if is_safetensor_archive:
            array = pt_state_dict.get_tensor(state_dict_name)
        else:
            array = pt_state_dict[state_dict_name]
        try:
            # 应用可能的转置操作，并处理大小不匹配的异常
            array = apply_transpose(transpose, array, symbolic_weight.shape)
        except tf.errors.InvalidArgumentError as e:
            # 如果不忽略大小不匹配，则抛出异常；否则将键添加到不匹配键列表中并继续下一个循环
            if not ignore_mismatched_sizes:
                error_msg = str(e)
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
                raise tf.errors.InvalidArgumentError(error_msg)
            else:
                mismatched_keys.append((name, array.shape, symbolic_weight.shape))
                continue
    
        # 更新 TensorFlow 加载的张量数量
        tf_loaded_numel += tensor_size(array)
    
        # 将 PyTorch 数组的值设置为符号权重的值
        K.set_value(symbolic_weight, array)
        # 立即释放内存，以保持峰值使用量尽可能低
        del array  
    
    # 记录加载的 TF 2.0 模型中的参数数量
    logger.info(f"Loaded {tf_loaded_numel:,} parameters in the TF 2.0 model.")
    
    # 获取未预期的键列表
    unexpected_keys = list(all_pytorch_weights)
    
    # 如果存在可以忽略加载丢失的键的列表，则从丢失键列表中移除匹配的键
    if tf_model._keys_to_ignore_on_load_missing is not None:
        for pat in tf_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
    # 如果存在可以忽略加载不期望的键的列表，则从未预期的键列表中移除匹配的键
    if tf_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in tf_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
    # 检查是否存在未预期的键
    if len(unexpected_keys) > 0:
        # 如果存在未使用的 PyTorch 模型权重，则发出警告，说明未使用的键，并给出可能的原因
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
        # 如果所有 PyTorch 模型权重都被使用，则发出警告
        logger.warning(f"All PyTorch model weights were used when initializing {tf_model.__class__.__name__}.\n")
    
    # 检查是否存在缺失的键
    if len(missing_keys) > 0:
        # 如果存在 TF 2.0 模型的权重或缓冲区没有从 PyTorch 模型初始化，则发出警告，说明缺失的键，并提供训练建议
        logger.warning(
            f"Some weights or buffers of the TF 2.0 model {tf_model.__class__.__name__} were not initialized from the"
            f" PyTorch model and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a"
            " down-stream task to be able to use it for predictions and inference."
        )
    else:
        # 如果所有 TF 2.0 模型的权重和缓冲区都已从 PyTorch 模型初始化，则发出警告
        logger.warning(
            f"All the weights of {tf_model.__class__.__name__} were initialized from the PyTorch model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {tf_model.__class__.__name__} for predictions without further training."
        )
    
    # 检查是否存在不匹配的键
    if len(mismatched_keys) > 0:
        # 如果存在权重不匹配的情况，则发出警告，说明不匹配的键以及形状，给出建议
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
    
    # 如果需要输出加载信息，则返回加载信息
    if output_loading_info:
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
        }
        return tf_model, loading_info
    
    # 返回 TF 模型
    return tf_model
# 载入 TensorFlow 2.0 HDF5 检查点到 PyTorch 模型中
def load_tf2_checkpoint_in_pytorch_model(
    pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False, output_loading_info=False
):
    """
    Load TF 2.0 HDF5 checkpoint in a PyTorch model We use HDF5 to easily do transfer learning (see
    https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    尝试导入 TensorFlow 和 PyTorch 库
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    导入 transformers 模块和 load_tf_weights 函数
    import transformers

    from .modeling_tf_utils import load_tf_weights

    记录日志，显示正在加载 TensorFlow 权重的路径
    logger.info(f"Loading TensorFlow weights from {tf_checkpoint_path}")

    # 实例化并加载相关的 TF 2.0 模型
    tf_model_class_name = "TF" + pt_model.__class__.__name__  # 在类名前面添加 "TF"
    tf_model_class = getattr(transformers, tf_model_class_name)
    tf_model = tf_model_class(pt_model.config)

    如果 tf_inputs 为 None，则使用 tf_model 的虚拟输入
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    如果 tf_inputs 不为 None，则确保模型已构建
    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # 确保模型已构建

    载入 TF 模型的权重
    load_tf_weights(tf_model, tf_checkpoint_path)

    返回载入 TF2 模型到 PyTorch 模型的结果
    return load_tf2_model_in_pytorch_model(
        pt_model, tf_model, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info
    )


# 载入 TF2 模型到 PyTorch 模型
def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=False, output_loading_info=False):
    """Load TF 2.0 model in a pytorch model"""
    获取 TF 模型的权重
    weights = tf_model.weights

    返回载入 TF2 模型权重到 PyTorch 模型的结果
    return load_tf2_weights_in_pytorch_model(
        pt_model, weights, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info
    )


# 载入 TF2 模型权重到 PyTorch 模型
def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False, output_loading_info=False):
    """Load TF2.0 symbolic weights in a PyTorch model"""
    尝试导入 TensorFlow 和 PyTorch 库
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    创建 TF 状态字典，包含 TF 权重的名称和值
    tf_state_dict = {tf_weight.name: tf_weight.numpy() for tf_weight in tf_weights}
    返回载入 TF2 状态字典到 PyTorch 模型的结果
    return load_tf2_state_dict_in_pytorch_model(
        pt_model, tf_state_dict, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info
    )


# 载入 TF2 状态字典到 PyTorch 模型
def load_tf2_state_dict_in_pytorch_model(pt_model, tf_state_dict, allow_missing_keys=False, output_loading_info=False):
    import torch

    创建新的 PyTorch 参数字典
    new_pt_params_dict = {}
    # 创建一个包含当前 PyTorch 模型所有参数的字典
    current_pt_params_dict = dict(pt_model.named_parameters())
    
    # 确保我们能够加载 PyTorch 基础模型以及衍生模型（带有头部）
    # TF 模型总是有一个前缀，而一些 PyTorch 模型（基础模型）没有
    start_prefix_to_remove = ""
    if not any(s.startswith(pt_model.base_model_prefix) for s in current_pt_params_dict.keys()):
        start_prefix_to_remove = pt_model.base_model_prefix + "."
    
    # 构建一个从可能的 PyTorch 权重名称到 TF 2.0 变量的映射
    tf_weights_map = {}
    for name, tf_weight in tf_state_dict.items():
        pt_name, transpose = convert_tf_weight_name_to_pt_weight_name(
            name, start_prefix_to_remove=start_prefix_to_remove, tf_weight_shape=tf_weight.shape
        )
        tf_weights_map[pt_name] = (tf_weight, transpose)
    
    # 获取所有 TF 权重的集合
    all_tf_weights = set(tf_weights_map.keys())
    # 用于存储加载的 PyTorch 权重数据指针的字典
    loaded_pt_weights_data_ptr = {}
    # 用于存储缺失的 PyTorch 权重键的列表
    missing_keys_pt = []
    
    # 遍历当前 PyTorch 参数字典
    for pt_weight_name, pt_weight in current_pt_params_dict.items():
        # 处理 PyTorch 共享权重（在 TF 2.0 中不重复）
        if pt_weight.data_ptr() in loaded_pt_weights_data_ptr:
            new_pt_params_dict[pt_weight_name] = loaded_pt_weights_data_ptr[pt_weight.data_ptr()]
            continue
    
        pt_weight_name_to_check = pt_weight_name
        # 处理新的 `weight_norm`，来自 https://github.com/huggingface/transformers/pull/24030
        key_components = pt_weight_name.split(".")
        name = None
        if key_components[-3::2] == ["parametrizations", "original0"]:
            name = key_components[-2] + "_g"
        elif key_components[-3::2] == ["parametrizations", "original1"]:
            name = key_components[-2] + "_v"
        if name is not None:
            key_components = key_components[:-3] + [name]
            pt_weight_name_to_check = ".".join(key_components)
    
        # 查找 PyTorch 模型状态字典中关联的 numpy 数组
        if pt_weight_name_to_check not in tf_weights_map:
            if allow_missing_keys:
                missing_keys_pt.append(pt_weight_name)
                continue
    
            raise AttributeError(f"{pt_weight_name} not found in TF 2.0 model")
    
        array, transpose = tf_weights_map[pt_weight_name_to_check]
    
        # 应用转置
        array = apply_transpose(transpose, array, pt_weight.shape, pt_to_tf=False)
    
        if numpy.isscalar(array):
            array = numpy.array(array)
        if not is_torch_tensor(array) and not is_numpy_array(array):
            array = array.numpy()
        if is_numpy_array(array):
            # 转换为 torch 张量
            array = torch.from_numpy(array)
    
        new_pt_params_dict[pt_weight_name] = array
        loaded_pt_weights_data_ptr[pt_weight.data_ptr()] = array
        all_tf_weights.discard(pt_weight_name)
    
    # 使用新的 PyTorch 参数字典加载模型状态，并且不严格要求完全匹配
    missing_keys, unexpected_keys = pt_model.load_state_dict(new_pt_params_dict, strict=False)
    missing_keys += missing_keys_pt
    # 某些模型可能在设计上具有不在状态中的键，在不必要地警告用户之前删除它们。
    if pt_model._keys_to_ignore_on_load_missing is not None:
        # 遍历需要忽略的丢失键的模式列表，过滤出不匹配模式的丢失键
        for pat in pt_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    if pt_model._keys_to_ignore_on_load_unexpected is not None:
        # 遍历需要忽略的意外键的模式列表，过滤出不匹配模式的意外键
        for pat in pt_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    if len(unexpected_keys) > 0:
        # 如果存在未使用的权重，则发出警告
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
        # 如果所有权重都被使用，则发出相应的警告
        logger.warning(f"All TF 2.0 model weights were used when initializing {pt_model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        # 如果存在未初始化的键，则发出警告
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the TF 2.0 model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )
    else:
        # 如果所有权重都被初始化，则发出相应的警告
        logger.warning(
            f"All the weights of {pt_model.__class__.__name__} were initialized from the TF 2.0 model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {pt_model.__class__.__name__} for predictions without further training."
        )

    # 记录未从 TF 2.0 模型加载的权重或缓冲区
    logger.info(f"Weights or buffers not loaded from TF 2.0 model: {all_tf_weights}")

    if output_loading_info:
        # 如果需要输出加载信息，则返回加载信息字典
        loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}
        return pt_model, loading_info

    # 返回已初始化的 PyTorch 模型
    return pt_model
```