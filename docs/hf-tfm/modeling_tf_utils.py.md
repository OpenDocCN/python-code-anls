# `.\transformers\modeling_tf_utils.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，包括作者和团队信息
# 版权声明，版权所有，保留所有权利
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""TF general model utils."""

from __future__ import annotations

import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import h5py
import numpy as np
import tensorflow as tf
from huggingface_hub import Repository, list_repo_files
from keras import backend as K
from packaging.version import parse

from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
    expand_1d,
    load_attributes_from_hdf5_group,
    save_attributes_to_hdf5_group,
    shape_list,
)
from .utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_INDEX_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ModelOutput,
    PushToHubMixin,
    cached_file,
    download_url,
    find_labels,
    has_file,
    is_offline_mode,
    is_remote_url,
    is_safetensors_available,
    is_tf_symbolic_tensor,
    logging,
    requires_backends,
    working_or_temp_dir,
)
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files

# 如果安装了 safetensors 库，则导入相关模块
if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.tensorflow import save_file as safe_save_file

# 如果是类型检查，导入 PreTrainedTokenizerBase
if TYPE_CHECKING:
    from . import PreTrainedTokenizerBase

# 获取日志记录器
logger = logging.get_logger(__name__)
tf_logger = tf.get_logger()

# 定义 TFModelInputType 类型
TFModelInputType = Union[
    List[tf.Tensor],
    List[np.ndarray],
    Dict[str, tf.Tensor],
    Dict[str, np.ndarray],
    tf.Tensor,
    np.ndarray,
]

# 定义一个虚拟损失函数
def dummy_loss(y_true, y_pred):
    if y_pred.shape.rank <= 1:
        return y_pred
    else:
        reduction_axes = list(range(1, y_pred.shape.rank))
        return tf.reduce_mean(y_pred, axis=reduction_axes)

# 定义 TFModelUtilsMixin 类
class TFModelUtilsMixin:
    """
    A few utilities for `tf.keras.Model`, to be used as a mixin.
    """
    # 定义一个方法，用于获取模型中的参数数量（可选择是否只包括可训练参数）
    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get the number of (optionally, trainable) parameters in the model.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

        Returns:
            `int`: The number of parameters.
        """
        # 如果只包括可训练参数，则计算可训练参数的数量并返回
        if only_trainable:
            return int(sum(np.prod(w.shape.as_list()) for w in self.trainable_variables))
        # 否则，调用 count_params 方法获取所有参数的数量并返回
        else:
            return self.count_params()
# 用于装饰 Keras Layer 类以支持 Keras 序列化的装饰器函数
def keras_serializable(cls):
    # 保存原始的初始化方法
    initializer = cls.__init__

    # 获取配置类，通常是一个 TF.MainLayer 类
    config_class = getattr(cls, "config_class", None)
    # 如果配置类未设置，则抛出 AttributeError
    if config_class is None:
        raise AttributeError("Must set `config_class` to use @keras_serializable")

    # 重新定义初始化方法
    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        # 从参数中获取配置对象
        config = args[0] if args and isinstance(args[0], PretrainedConfig) else kwargs.pop("config", None)

        # 根据传入的配置参数类型进行初始化
        if isinstance(config, dict):
            # 将字典类型的配置转换为配置对象
            config = config_class.from_dict(config)
            initializer(self, config, *args, **kwargs)
        elif isinstance(config, PretrainedConfig):
            # 如果传入的是预训练配置对象，则直接使用
            if len(args) > 0:
                initializer(self, *args, **kwargs)
            else:
                initializer(self, config, *args, **kwargs)
        else:
            # 如果既不是字典也不是预训练配置对象，则抛出 ValueError
            raise ValueError("Must pass either `config` (PretrainedConfig) or `config` (dict)")

        # 保存配置对象和额外参数
        self._config = config
        self._kwargs = kwargs

    # 用重新定义的初始化方法替换原始的初始化方法
    cls.__init__ = wrapped_init

    # 检查是否有 get_config 方法
    if not hasattr(cls, "get_config"):
        raise TypeError("Only use @keras_serializable on tf.keras.layers.Layer subclasses")
    # 检查 get_config 方法是否为默认方法
    if hasattr(cls.get_config, "_is_default"):

        # 重新定义 get_config 方法
        def get_config(self):
            # 获取父类的配置
            cfg = super(cls, self).get_config()
            # 将配置对象转换为字典并加入配置
            cfg["config"] = self._config.to_dict()
            cfg.update(self._kwargs)
            return cfg

        # 用重新定义的 get_config 方法替换原始的 get_config 方法
        cls.get_config = get_config

    # 标记类为可序列化
    cls._keras_serializable = True
    # 如果支持 Tensorflow 版本，则在 Keras 中注册类为自定义对象
    if hasattr(tf.keras.utils, "register_keras_serializable"):
        cls = tf.keras.utils.register_keras_serializable()(cls)
    return cls


# 用于序列模型的损失函数，适用于因果语言建模（CLM）任务，即猜测下一个标记
class TFCausalLanguageModelingLoss:
    """
    Loss function suitable for causal language modeling (CLM), that is, the task of guessing the next token.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """
    # 计算损失函数
    def hf_compute_loss(self, labels, logits):
        # 定义交叉熵损失函数，from_logits=True表示logits未经过softmax，reduction指定不进行损失的汇总
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        if self.config.tf_legacy_loss:
            # 确保只有不等于-100的标签会影响损失
            active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2]), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
            return loss_fn(labels, reduced_logits)

        # 将负标签裁剪为零，以避免NaN和错误 - 这些位置将在后面被掩盖
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # 确保只有不等于-100的标签会影响损失
        loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))
class TFQuestionAnsweringLoss:
    """
    Loss function suitable for question answering.
    """

    def hf_compute_loss(self, labels, logits):
        # 定义交叉熵损失函数，适用于稀疏分类，从logits计算，不进行损失缩减
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # 计算起始位置的损失
        start_loss = loss_fn(labels["start_position"], logits[0])
        # 计算结束位置的损失
        end_loss = loss_fn(labels["end_position"], logits[1])

        # 返回起始位置和结束位置损失的平均值作为最终损失
        return (start_loss + end_loss) / 2.0


class TFTokenClassificationLoss:
    """
    Loss function suitable for token classification.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        # 定义交叉熵损失函数，适用于稀疏分类，从logits计算，不进行损失缩减
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # 如果处于执行阶段（eager execution），警告使用-1来屏蔽标记的损失已被弃用，请改用-100
        if tf.executing_eagerly():  # Data-dependent conditionals are forbidden in XLA
            if tf.math.reduce_any(labels == -1):
                tf.print("Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.")

        if self.config.tf_legacy_loss:
            # 确保只有不等于-100的标签被考虑在损失计算中
            if tf.math.reduce_any(labels == -1):
                tf.print("Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.")
                active_loss = tf.reshape(labels, (-1,)) != -1
            else:
                active_loss = tf.reshape(labels, (-1,)) != -100
            # 将不等于-100的标签和对应的logits提取出来
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

            return loss_fn(labels, reduced_logits)

        # 在这里将负标签裁剪为零以避免NaN和错误，这些位置稍后会被掩盖 - 这些位置将在后面被掩盖
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # 确保只有不等于-100或-1的标签被考虑在损失计算中
        loss_mask = tf.cast(labels >= 0, dtype=unmasked_loss.dtype)
        # 避免后面可能的除零操作
        # 掩盖的位置将损失NaN，因为-100和-1不是有效的标签
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))


class TFSequenceClassificationLoss:
    """
    Loss function suitable for sequence classification.
    """
    # 计算损失函数
    def hf_compute_loss(self, labels, logits):
        # 如果 logits 的维度为 1 或者 logits 的第二维为 1
        if logits.shape.rank == 1 or logits.shape[1] == 1:
            # 使用均方误差损失函数，不进行汇总
            loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            # 如果 labels 的维度为 1
            if labels.shape.rank == 1:
                # 将 labels 扩展一个维度，避免 MeanSquaredError 返回标量损失
                labels = tf.expand_dims(labels, axis=-1)
        else:
            # 使用稀疏分类交叉熵损失函数，从 logits 计算，不进行汇总
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )

        # 返回计算得到的损失值
        return loss_fn(labels, logits)
class TFMultipleChoiceLoss:
    """Loss function suitable for multiple choice tasks."""

    def hf_compute_loss(self, labels, logits):
        # 使用稀疏分类交叉熵作为损失函数，适用于多分类任务
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        return loss_fn(labels, logits)


class TFMaskedLanguageModelingLoss(TFCausalLanguageModelingLoss):
    """
    Loss function suitable for masked language modeling (MLM), that is, the task of guessing the masked tokens.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """


class TFNextSentencePredictionLoss:
    """
    Loss function suitable for next sentence prediction (NSP), that is, the task of guessing the next sentence.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        # 使用稀疏分类交叉熵作为损失函数，适用于多分类任务
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        if self.config.tf_legacy_loss:
            # 确保只有标签不等于-100时才被计入损失
            next_sentence_active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            next_sentence_reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, 2)), next_sentence_active_loss)
            next_sentence_label = tf.boolean_mask(tf.reshape(labels, (-1,)), next_sentence_active_loss)

            return loss_fn(next_sentence_label, next_sentence_reduced_logits)

        # 确保只有标签不等于-100时才被计入损失

        # 在这里将负标签剪切为零，以避免NaN和错误——这些位置后面会被屏蔽
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels), y_pred=logits)
        ns_loss_mask = tf.cast(labels != -100, dtype=unmasked_ns_loss.dtype)
        # 只将标签为-100的样本归零，不进行损失的降维
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask

        return masked_ns_loss


def booleans_processing(config, **kwargs):
    """
    Process the input booleans of each model.

    Args:
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The boolean parameters

    Returns:
        A dictionary with the proper values for each boolean
    """
    final_booleans = {}

    # 纯卷积模型（例如ConvNext）没有`output_attentions`。如果签名中有`output_attentions`，则它将在此处作为`kwargs`的一部分出现，即使未设置（在这种情况下，为`None`）
    if "output_attentions" in kwargs:
        final_booleans["output_attentions"] = (
            kwargs["output_attentions"] if kwargs["output_attentions"] is not None else config.output_attentions
        )
    # 设置是否输出隐藏状态的布尔值，如果kwargs中有指定则使用kwargs中的值，否则使用config中的默认值
    final_booleans["output_hidden_states"] = (
        kwargs["output_hidden_states"] if kwargs["output_hidden_states"] is not None else config.output_hidden_states
    )
    # 设置是否返回字典的布尔值，如果kwargs中有指定则使用kwargs中的值，否则使用config中的默认值
    final_booleans["return_dict"] = kwargs["return_dict"] if kwargs["return_dict"] is not None else config.return_dict

    # 如果kwargs中有指定"use_cache"，则设置是否使用缓存的布尔值，如果kwargs中有指定则使用kwargs中的值，否则使用config中的默认值或者None
    if "use_cache" in kwargs:
        final_booleans["use_cache"] = (
            kwargs["use_cache"] if kwargs["use_cache"] is not None else getattr(config, "use_cache", None)
        )
    # 返回最终的布尔值字典
    return final_booleans
# 用装饰器处理 Keras 层的输入，将它们作为关键字参数传递给层，这样可以通过它们的变量名在下游使用，即使它们作为字典打包在第一个输入中（在 Keras 中很常见）
def unpack_inputs(func):
    """
    Decorator that processes the inputs to a Keras layer, passing them to the layer as keyword arguments. This enables
    downstream use of the inputs by their variable name, even if they arrive packed as a dictionary in the first input
    (common case in Keras).

    Args:
        func (`callable`):
            The callable function of the TensorFlow model.


    Returns:
        A callable that wraps the original `func` with the behavior described above.
    """

    # 获取原始函数的签名
    original_signature = inspect.signature(func)

    @functools.wraps(func)
    def run_call_with_unpacked_inputs(self, *args, **kwargs):
        # 为装饰函数隔离实际的 `**kwargs`
        kwargs_call = {key: val for key, val in kwargs.items() if key not in dict(original_signature.parameters)}
        fn_args_and_kwargs = {key: val for key, val in kwargs.items() if key not in kwargs_call}
        fn_args_and_kwargs.update({"kwargs_call": kwargs_call})

        # 如果存在任何参数，则将其移动到 kwargs 中
        fn_args_and_kwargs.update(dict(zip(func.__code__.co_varnames[1:], args)))

        # Encoder Decoder 模型将配置选项的应用委托给其内部模型
        if "EncoderDecoder" in self.__class__.__name__:
            config = None
        else:
            config = self.config

        # 处理输入
        unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
        return func(self, **unpacked_inputs)

    # Keras 要求传递第一个层参数，并通过 `inspect.getfullargspec()` 进行检查
    # 该函数不遵循包装器链（即忽略 `functools.wraps()`），因此没有下面的行，Keras 将尝试根据包装器的字面签名检查第一个参数
    run_call_with_unpacked_inputs.__signature__ = original_signature

    return run_call_with_unpacked_inputs


# 处理每个 TensorFlow 模型的输入，包括布尔值
def input_processing(func, config, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_ids = tf.keras.Input(shape=(128,), dtype='int32',
    name="input_ids")` otherwise the order of the tensors will not be guaranteed during the training.

    Args:
        func (`callable`):
            The callable function of the TensorFlow model.
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    # 获取函数的签名
    signature = dict(inspect.signature(func).parameters)
    # 检查是否有 kwargs 参数
    has_kwargs = bool(signature.pop("kwargs", None))
    signature.pop("self", None)
    parameter_names = list(signature.keys())
    main_input_name = parameter_names[0]
    main_input = kwargs.pop(main_input_name, None)
    output = {}
    # 定义允许的数据类型
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)

    # 如果参数中包含"inputs"，发出警告并将其替换为"input_ids"
    if "inputs" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
            FutureWarning,
        )
        output["input_ids"] = kwargs["kwargs_call"].pop("inputs")

    # 如果参数中包含"decoder_cached_states"，发出警告并将其替换为"past_key_values"
    if "decoder_cached_states" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
            " `past_key_values` instead.",
            FutureWarning,
        )
        output["past_key_values"] = kwargs["kwargs_call"].pop("decoder_cached_states")

    # 如果参数中包含"past"且"past_key_values"在参数名称中，发出警告并进行替换
    if "past" in kwargs["kwargs_call"] and "past_key_values" in parameter_names:
        warnings.warn(
            "The `past` argument is deprecated and will be removed in a future version, use `past_key_values`"
            " instead.",
            FutureWarning,
        )
        kwargs["past_key_values"] = kwargs["kwargs_call"].pop("past")
    # 如果参数中包含"past_key_values"且"past"在参数名称中，进行替换
    elif "past_key_values" in kwargs["kwargs_call"] and "past" in parameter_names:
        kwargs["past"] = kwargs["kwargs_call"].pop("past_key_values")

    # 如果存在kwargs参数，将其存储到输出中并移除原参数
    if has_kwargs:
        output["kwargs"] = kwargs.pop("kwargs_call", {})
    else:
        # 如果kwargs中还有未处理的参数，抛出异常
        if len(kwargs["kwargs_call"]) > 0:
            raise ValueError(
                "The following keyword arguments are not supported by this model:"
                f" {list(kwargs['kwargs_call'].keys())}."
            )
        kwargs.pop("kwargs_call")

    # 遍历kwargs中的参数，根据类型判断是否允许存储到输出中
    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or tf.is_tensor(v) or v is None:
            output[k] = v
        else:
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    # 如果main_input是元组或列表，处理每个输入
    if isinstance(main_input, (tuple, list)):
        for i, input in enumerate(main_input):
            # 对于Tensor类型的输入，根据名称存储到输出中
            if is_tf_symbolic_tensor(input):
                tensor_name = input.name.split(":")[0]
                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            # 对于允许的数据类型或None类型的输入，存储到输出中
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for"
                    f" {parameter_names[i]}."
                )
    # 如果主输入是映射类型（如字典）
    elif isinstance(main_input, Mapping):
        # 如果主输入字典中包含键"inputs"
        if "inputs" in main_input:
            # 发出警告，指出`inputs`参数已被弃用，并将在将来的版本中移除，建议使用`input_ids`代替
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids`"
                " instead.",
                FutureWarning,
            )

            # 将`inputs`对应的值赋给输出字典的键"input_ids"，并从主输入字典中移除"inputs"键值对
            output["input_ids"] = main_input.pop("inputs")

        # 如果主输入字典中包含键"decoder_cached_states"
        if "decoder_cached_states" in main_input:
            # 发出警告，指出`decoder_cached_states`参数已被弃用，并将在将来的版本中移除，建议使用`past_key_values`代替
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
                " `past_key_values` instead.",
                FutureWarning,
            )
            # 将`decoder_cached_states`对应的值赋给输出字典的键"past_key_values"，并从主输入字典中移除"decoder_cached_states"键值对
            output["past_key_values"] = main_input.pop("decoder_cached_states")

        # 遍历主输入字典中的键值对
        for k, v in dict(main_input).items():
            # 如果值属于允许的类型或者为None
            if isinstance(v, allowed_types) or v is None:
                # 将键值对添加到输出字典中
                output[k] = v
            # 如果键不在参数名列表中，并且"args"不在参数名列表中
            elif k not in parameter_names and "args" not in parameter_names:
                # 发出警告，指出参数不属于参数列表，并将被忽略
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                # 继续下一次循环
                continue
            else:
                # 抛出异常，指出该类型的数据不允许，并提供相应的参数和键
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        # 如果主输入是TensorFlow张量或者为None
        if tf.is_tensor(main_input) or main_input is None:
            # 将主输入赋给输出字典的键`main_input_name`
            output[main_input_name] = main_input
        else:
            # 抛出异常，指出该类型的数据不允许，并提供相应的参数和键
            raise ValueError(
                f"Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for"
                f" {main_input_name}."
            )

    # 根据函数签名，使用默认值填充任何未指定的参数
    for name in parameter_names:
        # 如果参数名不在输出字典的键列表中，并且参数名不是"args"
        if name not in list(output.keys()) and name != "args":
            # 如果参数名在签名中有默认值，则使用默认值填充输出字典中的对应键
            output[name] = kwargs.pop(name, signature[name].default)

    # 当创建SavedModel时，TF调用LayerCall.__call__(args, **kwargs)方法
    # 为了保持正确的输出，我们必须添加此异常
    if "args" in output:
        # 如果输出字典中的"args"不为None，并且是TensorFlow符号张量
        if output["args"] is not None and is_tf_symbolic_tensor(output["args"]):
            # 获取Tensor的名称
            tensor_name = output["args"].name.split(":")[0]
            # 将`args`对应的张量添加到输出字典中
            output[tensor_name] = output["args"]
        else:
            # 在这种情况下，`args`始终是第一个参数，然后是`input_ids`
            # 将`args`对应的值赋给输出字典的键"input_ids"
            output["input_ids"] = output["args"]

        # 从输出字典中删除"args"键值对
        del output["args"]

    # 从输出字典中删除"kwargs"键值对
    if "kwargs" in output:
        del output["kwargs"]

    # 将输出字典中的值进行类型转换，确保所有整数类型为int32
    cast_output = {}
    for key, val in output.items():
        if isinstance(val, tf.Tensor) and val.dtype == tf.int64:
            # 将TensorFlow张量中的int64类型转换为int32
            cast_output[key] = tf.cast(val, tf.int32)
        elif isinstance(val, np.ndarray) and val.dtype == np.int64:
            # 将NumPy数组中的int64类型转换为int32
            cast_output[key] = val.astype(np.int32)
        else:
            # 其他情况下，保持原值不变
            cast_output[key] = val

    # 将类型转换后的输出字典赋给output，并删除类型转换后的字典
    output = cast_output
    del cast_output
    # 如果配置不为空
    if config is not None:
        # 从输出字典中筛选出指定键的键值对，组成新的字典
        boolean_dict = {
            k: v
            for k, v in output.items()
            if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
        }

        # 调用 booleans_processing 函数处理布尔值相关配置，并更新到输出字典中
        output.update(
            booleans_processing(
                config=config,
                **boolean_dict,
            )
        )

    # 返回更新后的输出字典
    return output
# 定义函数，用于计算指定数据类型的参数所占用的字节数
def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(tf.float32)
    4
    ```py
    """
    # 如果数据类型是布尔型，则返回占用字节数为 1/8
    if dtype == tf.bool:
        return 1 / 8
    # 通过正则表达式从数据类型名称中提取位数
    bit_search = re.search(r"[^\d](\d+)$", dtype.name)
    # 如果未找到位数，则抛出 ValueError 异常
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])  # 获取位数
    return bit_size // 8  # 返回字节数


# 定义函数，用于剥离模型名称和前缀
def strip_model_name_and_prefix(name, _prefix=None):
    # 如果存在前缀并且名称以该前缀开头，则将前缀从名称中去除
    if _prefix is not None and name.startswith(_prefix):
        name = name[len(_prefix) :]
        # 如果去除前缀后的名称以 "/" 开头，则再次去除 "/"
        if name.startswith("/"):
            name = name[1:]
    # 如果名称中不包含 "model."，且分割后的部分大于 1，则只保留第二部分
    if "model." not in name and len(name.split("/")) > 1:
        name = "/".join(name.split("/")[1:])
    return name


# 定义函数，用于将模型状态字典分割为子检查点，使每个子检查点的大小不超过给定大小
def tf_shard_checkpoint(weights, max_shard_size="10GB"):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        weights (`Dict[str, tf.RessourceVariable]`): The list of tf.RessourceVariable of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
    """
    # 将最大子检查点大小转换为整数
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []  # 存储分割后的子检查点列表
    current_block = []  # 当前检查点块
    current_block_size = 0  # 当前检查点块的大小
    total_size = 0  # 总大小

    # 遍历权重列表
    for item in weights:
        # 计算权重的大小
        weight_size = item.numpy().size * dtype_byte_size(item.dtype)

        # 如果当前检查点块的大小加上该权重的大小超过最大子检查点大小，则进行分割
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)  # 将当前检查点块添加到分割后的子检查点列表中
            current_block = []  # 重置当前检查点块
            current_block_size = 0  # 重置当前检查点块的大小

        # 将权重添加到当前检查点块中
        current_block.append(item)
        current_block_size += weight_size  # 更新当前检查点块的大小
        total_size += weight_size  # 更新总大小

    # 添加最后一个检查点块
    sharded_state_dicts.append(current_block)

    # 如果只有一个子检查点，直接返回该子检查点
    if len(sharded_state_dicts) == 1:
        return {TF2_WEIGHTS_NAME: sharded_state_dicts[0]}, None

    # 否则，构建索引
    weight_map = {}  # 权重映射字典
    shards = {}  # 子检查点字典
    # 遍历分片状态字典列表，获取索引和每个分片
    for idx, shard in enumerate(sharded_state_dicts):
        # 根据索引和总分片数生成分片文件名
        shard_file = TF2_WEIGHTS_NAME.replace(".h5", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.h5")
        # 将分片数据和文件名添加到分片字典中
        shards[shard_file] = shard
        # 遍历分片中的权重，获取权重名称并映射到分片文件名
        for weight in shard:
            weight_name = weight.name
            weight_map[weight_name] = shard_file

    # 添加元数据
    metadata = {"total_size": total_size}
    # 创建索引字典，包含元数据和权重映射
    index = {"metadata": metadata, "weight_map": weight_map}
    # 返回分片字典和索引字典
    return shards, index
# 加载 TF 分片权重，根据名称和形状检测缺失和意外层，并根据其名称和形状从分片文件中加载 TF 权重
def load_tf_sharded_weights(model, shard_files, ignore_mismatched_sizes=False, strict=False, _prefix=None):
    """
    This is the same as `load_tf_weights` but for a sharded checkpoint. Detect missing and unexpected layers and load
    the TF weights from the shard file accordingly to their names and shapes.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`tf.keras.models.Model`): The model in which to load the checkpoint.
        shard_files (`str` or `os.PathLike`): A list containing the sharded checkpoint names.
        ignore_mismatched_sizes`bool`, *optional`, defaults to `True`):
            Whether or not to ignore the mismatch between the sizes
        strict (`bool`, *optional*, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
        mismatched layers.
    """

    # 加载索引
    unexpected_keys = set()
    saved_keys = set()
    mismatched_keys = set()

    # 由于 TF 将类的名称添加到其权重中，并使用索引而不是层的名称来加载权重，因此我们必须去掉层名称的第一个前缀
    model_keys = set()
    model_layer_map = {}
    for i, k in enumerate(model.weights):
        layer_name = k.name
        if _prefix is not None and layer_name.startswith(_prefix):
            layer_name = layer_name[len(_prefix) :]
            layer_name = layer_name.lstrip("/")
        if not ("model." in layer_name or len(layer_name.split("/")) == 1):
            layer_name = "/".join(layer_name.split("/")[1:])
        model_keys.add(layer_name)
        model_layer_map[layer_name] = i

    for shard_file in shard_files:
        saved_weight_names_set, unexpected_keys_set, mismatched_keys_set = load_tf_shard(
            model,
            model_layer_map,
            shard_file,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            _prefix=_prefix,
        )
        saved_keys.update(saved_weight_names_set)
        unexpected_keys.update(unexpected_keys_set)
        mismatched_keys.update(mismatched_keys_set)
        gc.collect()

    missing_keys = model_keys - saved_keys
    # 如果启用了严格模式，并且存在缺失的键或者多余的键
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        # 构造错误信息字符串，指示状态字典加载中发生错误，包含模型类名
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        # 如果存在缺失的键
        if len(missing_keys) > 0:
            # 构造缺失键列表的字符串表示，并添加到错误信息中
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        # 如果存在多余的键
        if len(unexpected_keys) > 0:
            # 构造多余键列表的字符串表示，并添加到错误信息中
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        # 抛出运行时错误，包含错误信息
        raise RuntimeError(error_message)

    # 返回缺失键、多余键和不匹配键的元组
    return missing_keys, unexpected_keys, mismatched_keys
# 从分片的检查点文件加载一个分片。处理丢失的键和意外的键。
def load_tf_shard(model, model_layer_map, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    # 保存权重名的集合
    saved_weight_names_set = set()
    # 保存权重的字典
    saved_weights = {}
    # 不匹配的键的集合
    mismatched_keys = set()
    # 意外的键的集合
    unexpected_keys = set()
    # 读取 H5 文件
    # 尝试打开 H5 文件，只读模式
    try:
        with h5py.File(resolved_archive_file, "r") as sharded_checkpoint_file:
            # 从 H5 文件中获取每个层的名称
            saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, "layer_names"))
            weight_value_tuples = []

            # 计算缺失和意外的子层
            # 将权重存储在类似 [(权重对象, 权重值), ...] 的元组列表中
            for layer_name in saved_h5_model_layers_name:
                # 获取 H5 文件中的层对象
                h5_layer_object = sharded_checkpoint_file[layer_name]
                saved_weights[layer_name] = np.asarray(h5_layer_object)

                saved_weight_names_set.add(layer_name)

                # 如果层名称不在模型层映射中
                if layer_name not in model_layer_map:
                    unexpected_keys.add(layer_name)
                else:
                    # 获取符号权重
                    symbolic_weight = model.weights[model_layer_map[layer_name]]

                    saved_weight_value = saved_weights[layer_name]
                    # 如果找到当前权重
                    if saved_weight_value is not None:
                        # 检查当前权重的形状和 H5 文件中的权重形状是否不同
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            # 如果是，我们根据当前权重重新调整 H5 文件中的权重
                            # 如果两个形状不兼容，我们引发问题
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_keys.add(
                                        (layer_name, saved_weight_value.shape, K.int_shape(symbolic_weight))
                                    )
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value

                    # 创建将要加载的元组并将其添加到最终列表中
                    weight_value_tuples.append((symbolic_weight, array))

        # 批量设置权重值
        K.batch_set_value(weight_value_tuples)

        # 返回结果集
        return saved_weight_names_set, unexpected_keys, mismatched_keys
    # 捕获任何异常并将其存储在变量e中
    except Exception as e:
        # 尝试打开解析后的存档文件
        try:
            with open(resolved_archive_file) as f:
                # 如果文件内容以"version"开头，则引发OSError异常
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                # 否则，引发ValueError异常
                else:
                    raise ValueError(
                        f"Unable to locate the file {resolved_archive_file} which is necessary to load this pretrained"
                        " model. Make sure you have saved the model properly."
                    ) from e
        # 捕获UnicodeDecodeError和ValueError异常
        except (UnicodeDecodeError, ValueError):
            # 引发OSError异常，指示无法从TF检查点文件加载权重
            raise OSError(
                f"Unable to load weights from TF checkpoint file for '{resolved_archive_file}' "
                f"at '{resolved_archive_file}'. "
                "If you tried to load a TF model from a sharded checkpoint, you should try converting the model "
                "by loading it in pytorch and saving it localy. A convertion script should be realeased soon."
            )
# 加载 TensorFlow 模型的权重，根据文件名和形状匹配加载权重
def load_tf_weights(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    # 如果文件名以 ".safetensors" 结尾，则使用 load_tf_weights_from_safetensors 函数
    if resolved_archive_file.endswith(".safetensors"):
        load_function = load_tf_weights_from_safetensors
    else:
        # 否则使用 load_tf_weights_from_h5 函数
        load_function = load_tf_weights_from_h5

    # 调用相应的加载函数
    return load_function(
        model, resolved_archive_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=_prefix
    )


# 从 H5 文件加载 TensorFlow 模型的权重
def load_tf_weights_from_h5(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    mismatched_layers = []

    # 读取 H5 文件
    # 加载所有权重
    K.batch_set_value(weight_value_tuples)

    # 计算缺失和意外的层
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set))
    unexpected_layers.extend(list(saved_weight_names_set - symbolic_weights_names))

    return missing_layers, unexpected_layers, mismatched_layers


# 从 safetensors 文件加载 TensorFlow 模型的权重
def load_tf_weights_from_safetensors(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    # 读取 safetensors 文件
    # 使用安全方式打开解析后的归档文件，以确保在不同框架下的兼容性
    with safe_open(resolved_archive_file, framework="tf") as safetensors_archive:
        # 用于存储不匹配的层
        mismatched_layers = []
        # 提取模型权重的名称，并去除前缀
        weight_names = [strip_model_name_and_prefix(w.name, _prefix=_prefix) for w in model.weights]
        # 提取已加载权重的名称列表
        loaded_weight_names = list(safetensors_archive.keys())
        # 找到在高级层列表中缺失的层
        missing_layers = list(set(weight_names) - set(loaded_weight_names))
        # 找到在高级层列表中意外存在的层
        unexpected_layers = list(set(loaded_weight_names) - set(weight_names))

        # 遍历模型的权重
        for weight in model.weights:
            # 获取去除前缀后的权重名称
            weight_name = strip_model_name_and_prefix(weight.name, _prefix=_prefix)
            # 如果权重名称在已加载的权重名称列表中
            if weight_name in loaded_weight_names:
                # 获取归档文件中相应权重的值
                weight_value = safetensors_archive.get_tensor(weight_name)
                # 检查当前权重与归档文件中的权重形状是否不同
                if K.int_shape(weight) != weight_value.shape:
                    # 如果不同，将归档文件中的权重形状调整为当前权重的形状
                    # 如果两个形状不兼容，引发问题
                    try:
                        weight_value = tf.reshape(weight_value, K.int_shape(weight))
                    except (ValueError, tf.errors.InvalidArgumentError) as e:
                        if ignore_mismatched_sizes:
                            # 如果忽略大小不匹配，将其记录到不匹配的层列表中
                            mismatched_layers.append((weight_name, weight_value.shape, K.int_shape(weight)))
                            continue
                        else:
                            raise e

                # 将权重值设置为已加载的权重值
                K.set_value(weight, weight_value)  # weight.assign() might break if weight is a DTensor

    # 返回缺失的层、意外存在的层和大小不匹配的层
    return missing_layers, unexpected_layers, mismatched_layers
# 初始化复制嵌入向量函数，用于在 new_num_tokens < old_num_tokens 时减少嵌入向量，或在 new_num_tokens > old_num_tokens 时使用 -1 进行填充。
# 同时计算一个掩码，以确定哪些嵌入权重应该保留或丢弃。
def init_copy_embeddings(old_embeddings, new_num_tokens):
    # 获取旧嵌入向量的形状
    old_num_tokens, old_embedding_dim = shape_list(old_embeddings)
    # 计算新旧嵌入向量之间的大小差异
    size_diff = new_num_tokens - old_num_tokens

    # 初始化新的嵌入向量
    # 从旧的嵌入向量中复制标记嵌入
    if tf.math.greater(size_diff, 0):
        # 如果新大小大于旧大小，则使用填充扩展当前嵌入向量直到达到新的大小
        # 同时创建一个掩码，以正确识别填充值，并用新创建的嵌入向量的值替换填充值
        current_weights = tf.pad(
            old_embeddings.value(), tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=-1
        )
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        mask = tf.fill(tf.convert_to_tensor([num_tokens_to_copy, 1]), True)
        mask = tf.pad(mask, tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=False)
    else:
        # 如果新大小小于旧大小，则取当前嵌入向量直到新大小
        current_weights = tf.slice(
            old_embeddings.value(),
            tf.convert_to_tensor([0, 0]),
            tf.convert_to_tensor([new_num_tokens, old_embedding_dim]),
        )
        mask = tf.fill(tf.convert_to_tensor([new_num_tokens, 1]), True)

    return mask, current_weights


# TF 模型的基类，负责存储模型的配置，并处理加载、下载和保存模型的方法，以及所有模型共有的一些方法：
# - 调整输入嵌入向量的大小
# - 在自注意力头中修剪头
class TFPreTrainedModel(tf.keras.Model, TFModelUtilsMixin, TFGenerationMixin, PushToHubMixin):
    # 所有 TF 模型的基类。

    # [`TFPreTrainedModel`] 负责存储模型的配置，并处理加载、下载和保存模型的方法，
    # 以及一些所有模型共有的方法：
    # - 调整输入嵌入的大小
    # - 在自注意力头中修剪头
    # 类属性（由派生类覆盖）：

        # - **config_class** ([`PretrainedConfig`]) -- 用作该模型架构的配置类的子类 [`PretrainedConfig`]。
        # - **base_model_prefix** (`str`) -- 一个字符串，指示与同一架构的派生类中基础模型相关联的属性，该属性在基础模型之上添加模块。
        # - **main_input_name** (`str`) -- 模型的主要输入的名称（通常是 NLP 模型的 `input_ids`，视觉模型的 `pixel_values` 和语音模型的 `input_values`）。
    """

    # 配置类，默认为 None
    config_class = None
    # 基础模型前缀，默认为空字符串
    base_model_prefix = ""
    # 主要输入的名称，默认为 "input_ids"
    main_input_name = "input_ids"
    # 自动类，默认为 None
    _auto_class = None
    # 使用虚拟损失，默认为 None
    _using_dummy_loss = None
    # 标签到输出映射，默认为 None
    _label_to_output_map = None

    # 加载模型权重时要忽略的张量名称的正则表达式列表（避免不必要的警告）
    _keys_to_ignore_on_load_missing = None
    # 加载模型权重时要忽略的权重张量名称的正则表达式列表（避免不必要的警告）
    _keys_to_ignore_on_load_unexpected = None
    # 是否需要加载权重前缀，默认为 False
    _requires_load_weight_prefix = False

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        构建网络的虚拟输入。

        Returns:
            `Dict[str, tf.Tensor]`: 虚拟输入。
        """
        # 初始化一个空字典，用于存储虚拟输入
        dummies = {}
        # 遍历输入签名的项
        for key, spec in self.input_signature.items():
            # 生成虚拟输入的形状，如果维度未指定，则使用 2（这是最正确的任意大小。不接受问题）
            dummy_shape = [dim if dim is not None else 2 for dim in spec.shape]
            # 如果批量大小未指定，则将其设置为 1 以节省内存
            if spec.shape[0] is None:
                dummy_shape[0] = 1
            # 根据规格创建一个全为 1 的张量作为虚拟输入
            dummies[key] = tf.ones(shape=dummy_shape, dtype=spec.dtype)
            # 对于特定的键 "token_type_ids"，如果存在，则将其设置为全为 0 的张量
            if key == "token_type_ids":
                dummies[key] = tf.zeros_like(dummies[key])
        # 如果模型具有交叉注意力且在调用中具有 "encoder_hidden_states" 参数
        if self.config.add_cross_attention and "encoder_hidden_states" in inspect.signature(self.call).parameters:
            # 如果 "encoder_hidden_states" 不在虚拟输入中
            if "encoder_hidden_states" not in dummies:
                # 如果主要输入名称为 "input_ids"，则创建一个形状为 (1, 2, self.config.hidden_size) 的全为 1 的张量作为 "encoder_hidden_states"
                if self.main_input_name == "input_ids":
                    dummies["encoder_hidden_states"] = tf.ones(
                        shape=(1, 2, self.config.hidden_size), dtype=tf.float32, name="encoder_hidden_states"
                    )
                # 否则，抛出 NotImplementedError
                else:
                    raise NotImplementedError(
                        "Model has cross-attention but we couldn't infer the shape for the encoder hidden states. Please manually override dummy_inputs!"
                    )
        # 返回虚拟输入字典
        return dummies

    # 在名称范围内构建模型
    def build_in_name_scope(self):
        # 在名称范围内构建模型
        with tf.name_scope(self.name):
            self.build(input_shape=None)

    @property
    def framework(self) -> str:
        """
        :str: 标识这是一个 TensorFlow 模型。
        """
        # 返回标识这是 TensorFlow 框架的字符串
        return "tf"
    def build(self, input_shape=None):
        pass  # This is just here to make sure we don't call the superclass build()
    # 重写 build 方法，但不实际执行任何操作，只是为了确保不调用父类的 build() 方法

    def __init__(self, config, *inputs, **kwargs):
        # 初始化方法，接受配置参数 config 和可变数量的位置参数 inputs 以及关键字参数 kwargs
        super().__init__(*inputs, **kwargs)
        # 调用父类的初始化方法
        if not isinstance(config, PretrainedConfig):
            # 如果 config 不是 PretrainedConfig 类的实例，则抛出 ValueError
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 将 config 和预训练权重的来源保存到当前对象中
        self.config = config
        self.name_or_path = config.name_or_path
        # 如果模型可以生成，则将生成配置保存到 generation_config 中
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        # 设置保存规范，即输入签名
        self._set_save_spec(self.input_signature)

    def get_config(self):
        # 返回模型的配置字典
        return self.config.to_dict()

    @classmethod
    def from_config(cls, config, **kwargs):
        # 从配置中重建模型的类方法
        if isinstance(config, PretrainedConfig):
            return cls._from_config(config, **kwargs)
        return cls._from_config(cls.config_class.from_dict(config, **kwargs))

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        # 从配置重建模型的类方法
        return cls(config, **kwargs)
    # 该方法将模型的初始化上下文管理器放在此处

    def get_head_mask(self, head_mask: tf.Tensor | None, num_hidden_layers: int) -> tf.Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`tf.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.

        Returns:
            `tf.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        # 准备头部掩码（如果需要的话）

        if head_mask is not None:
            # 如果 head_mask 不为 None，则调用 _convert_head_mask_to_5d 方法将其转换为 5 维张量
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            # 如果 head_mask 为 None，则创建一个列表，长度为 num_hidden_layers，每个元素都是 None
            head_mask = [None] * num_hidden_layers

        return head_mask
    # 返回头部掩码，如果需要的话

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        # 将头部掩码转换为 5 维张量

        if head_mask.shape.rank == 1:
            # 如果头部掩码的 rank 为 1，则增加维度使其成为 5 维张量
            head_mask = head_mask[None, None, :, None, None]
            # 在合适的位置复制 head_mask，使其与 num_hidden_layers 的数量匹配
            head_mask = tf.repeat(head_mask, repeats=num_hidden_layers, axis=0)
        elif head_mask.shape.rank == 2:
            # 如果头部掩码的 rank 为 2，则在特定位置添加维度
            head_mask = head_mask[:, None, :, None, None]
        assert head_mask.shape.rank == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        # 将 head_mask 转换为 float32 类型，以确保兼容性
        head_mask = tf.cast(head_mask, tf.float32)  # switch to float if need + fp16 compatibility
        return head_mask
    # 返回转换后的头部掩码，确保其为 5 维张量，用于模型的使用

    @tf.function
    # TensorFlow 函数装饰器，将函数转换为 TensorFlow 图执行模式，提高性能
    # 定义用于提供模型服务的方法，没有特定的签名，但在使用 `save_pretrained` 保存时会被专门化为具体函数
    def serving(self, inputs):
        """
        Args:
            inputs (`Dict[str, tf.Tensor]`):
                作为张量字典的保存模型的输入。
        """
        # 调用模型的 call 方法，传入输入，获取输出
        output = self.call(inputs)

        # 调用 serving_output 方法处理输出并返回结果
        return self.serving_output(output)

    # 定义用于提供模型服务的方法，此方法已被弃用，将在 Transformers 的 4.32.0 版本中移除
    def eager_serving(self, inputs):
        """
        Args:
            inputs (`Dict[str, tf.Tensor]`):
                作为张量字典的保存模型的输入。
        """
        # 发出警告，提示该方法已被弃用，将在未来版本中移除
        warnings.warn(
            "The function `eager_serving` is deprecated and will be removed in version 4.32.0 of Transformers",
            FutureWarning,
        )
        # 调用模型的 call 方法，传入输入，获取输出
        output = self.call(inputs)

        # 调用 serving_output 方法处理输出并返回结果
        return self.serving_output(output)

    # 属性
    # 定义一个方法，返回一个字典，将输入名称映射到 tf.TensorSpec 对象，表示模型输入的预期形状和数据类型
    def input_signature(self) -> Dict[str, tf.TensorSpec]:
        """
        This property should return a dict mapping input names to tf.TensorSpec objects, representing the expected
        shape and dtype for model inputs. It is used for both serving and for generating the dummy inputs used to build
        the model.
        """
        # 获取调用方法的参数列表
        model_inputs = list(inspect.signature(self.call).parameters)
        # 初始化一个空字典用于存储输入签名
        sig = {}
        # 如果模型输入中包含 "input_ids"
        if "input_ids" in model_inputs:
            # 根据模型类名结尾是否为 "ForMultipleChoice" 来确定文本维度
            if self.__class__.__name__.endswith("ForMultipleChoice"):
                text_dims = 3
            else:
                text_dims = 2
            # 遍历指定的输入名称列表
            for input_name in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "decoder_input_ids",
                "decoder_attention_mask",
            ):
                # 如果输入名称在模型输入中
                if input_name in model_inputs:
                    # 将输入名称和对应的 TensorSpec 对象添加到字典中
                    sig[input_name] = tf.TensorSpec([None] * text_dims, tf.int32, name=input_name)
        # 如果模型输入中包含 "pixel_values"
        if "pixel_values" in model_inputs:
            # 初始化像素值的形状
            pixel_values_shape = [None, None, None, None]
            # 根据配置获取视觉配置
            if hasattr(self.config, "vision_config"):
                vision_config = self.config.vision_config
            else:
                vision_config = self.config
            # 如果视觉配置中包含通道数
            if hasattr(vision_config, "num_channels"):
                pixel_values_shape[1] = vision_config.num_channels
            else:
                # 抛出异常，需要手动定义输入签名以指定输入形状
                raise NotImplementedError(
                    "Could not infer number of channels from config, please override input_signature to specify input shapes."
                )
            # 根据视觉配置设置像素值的高度和宽度
            if hasattr(vision_config, "image_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.image_size
            elif hasattr(vision_config, "input_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.input_size
            else:
                # 抛出异常，需要手动定义输入签名以指定输入形状
                raise NotImplementedError(
                    "Could not infer input image shape from config, please override input_signature to specify input shapes."
                )
            # 将像素值和对应的 TensorSpec 对象添加到字典中
            sig["pixel_values"] = tf.TensorSpec(pixel_values_shape, tf.float32, name="pixel_values")
        # 如果模型输入中包含 "input_features"
        if "input_features" in model_inputs:
            # 抛出异常，音频模型需要手动定义输入签名
            raise NotImplementedError("Audio models need a manually defined input_signature")
        # 返回输入签名字典
        return sig
    def serving_output(self, output):
        """
        Prepare the output of the saved model. Can be overridden if specific serving modifications are required.
        准备保存模型的输出。如果需要特定的服务修改，可以重写此方法。
        """
        if not isinstance(output, ModelOutput):
            return output
        # 遍历输出的键值对
        for key in output:
            # 如果键以"hidden_states"结尾且配置中未设置输出隐藏状态，则将其值设为None
            if key.endswith("hidden_states") and not getattr(self.config, "output_hidden_states", False):
                output[key] = None
            # 如果键以"attentions"结尾且配置中未设置输出注意力，则将其值设为None
            elif key.endswith("attentions") and not getattr(self.config, "output_attentions", False):
                output[key] = None
            # 如果键为"past_key_values"且配置中未设置使用缓存，则将其值设为None
            elif key == "past_key_values" and not getattr(self.config, "use_cache", False):
                output[key] = None
            # 如果键为"cross_attentions"且配置中未同时设置输出注意力和添加交叉注意力，则将其值设为None
            elif key == "cross_attentions" and not (
                getattr(self.config, "output_attentions", False) and getattr(self.config, "add_cross_attention", False)
            ):
                output[key] = None
            # 如果输出的值为元组或列表，则尝试将其转换为张量
            if isinstance(output[key], (tuple, list)):
                try:
                    output[key] = tf.convert_to_tensor(output[key])
                except (ValueError, tf.errors.InvalidArgumentError):
                    pass  # Layers may not have the same dimensions
        return output

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        返回该模型是否可以使用 `.generate()` 生成序列。

        Returns:
            `bool`: 返回该模型是否可以使用 `.generate()` 生成序列。
        """
        # 检测是否重写了`prepare_inputs_for_generation`，这是生成的要求之一。
        # 或者，模型也可以具有自定义的`generate`函数。
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        """
        Returns the model's input embeddings layer.

        Returns:
            `tf.Variable`: The embeddings layer mapping vocabulary to hidden states.
        返回模型的输入嵌入层。

        Returns:
            `tf.Variable`: 将词汇映射到隐藏状态的嵌入层。
        """
        main_layer = getattr(self, self.base_model_prefix, self)

        if main_layer is not self:
            # 如果主层不是当前层，则返回主层的输入嵌入层
            return main_layer.get_input_embeddings()
        else:
            # 否则，抛出未实现错误
            raise NotImplementedError
    # 保存模型的检查点信息，包括权重和额外数据
    def _save_checkpoint(self, checkpoint_dir, epoch):
        # 如果检查点目录不存在，则创建
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # 避免使用 tf.train.checkpoint 或保存 TF 格式的权重，因为这需要特殊处理自定义损失等对象
        # 我们使用内部和用户可能也会使用的对象
        # 保存权重到指定路径
        weights_path = os.path.join(checkpoint_dir, "weights.h5")
        self.save_weights(weights_path)
        # 保存额外数据，包括当前轮次和优化器状态
        extra_data = {"epoch": epoch, "optimizer_state": self.optimizer.get_weights()}
        extra_data_path = os.path.join(checkpoint_dir, "extra_data.pickle")
        # 使用 pickle 序列化额外数据并保存到文件
        with open(extra_data_path, "wb") as f:
            pickle.dump(extra_data, f)
    def load_repo_checkpoint(self, repo_path_or_name):
        """
        Loads a saved checkpoint (model weights and optimizer state) from a repo. Returns the current epoch count when
        the checkpoint was made.

        Args:
            repo_path_or_name (`str`):
                Can either be a repository name for your {object} in the Hub or a path to a local folder (in which case
                the repository will have the name of that local folder).

        Returns:
            `dict`: A dictionary of extra metadata from the checkpoint, most commonly an "epoch" count.
        """
        # 检查是否存在优化器，若不存在，则抛出运行时错误
        if getattr(self, "optimizer", None) is None:
            raise RuntimeError(
                "Checkpoint loading failed as no optimizer is attached to the model. "
                "This is most likely caused by the model not being compiled."
            )
        # 如果传入的路径是一个目录，则使用该目录作为本地目录
        if os.path.isdir(repo_path_or_name):
            local_dir = repo_path_or_name
        else:
            # 如果传入的不是本地路径，则检查远程仓库是否存在，并且其中包含一个检查点文件
            repo_files = list_repo_files(repo_path_or_name)
            # 检查是否有权重文件和额外数据文件
            for file in ("checkpoint/weights.h5", "checkpoint/extra_data.pickle"):
                if file not in repo_files:
                    raise FileNotFoundError(f"Repo {repo_path_or_name} does not contain checkpoint file {file}!")
            # 克隆远程仓库到本地，并获取本地路径
            repo = Repository(repo_path_or_name.split("/")[-1], clone_from=repo_path_or_name)
            local_dir = repo.local_dir

        # 确保仓库中存在检查点文件
        checkpoint_dir = os.path.join(local_dir, "checkpoint")
        weights_file = os.path.join(checkpoint_dir, "weights.h5")
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f"Could not find checkpoint file weights.h5 in repo {repo_path_or_name}!")
        extra_data_file = os.path.join(checkpoint_dir, "extra_data.pickle")
        if not os.path.isfile(extra_data_file):
            raise FileNotFoundError(f"Could not find checkpoint file extra_data.pickle in repo {repo_path_or_name}!")

        # 假设仓库存在且有检查点，加载权重和优化器状态到模型中
        # 优化器状态包括迭代计数，因此学习率调度也会正常恢复
        self.load_weights(weights_file)
        with open(extra_data_file, "rb") as f:
            extra_data = pickle.load(f)
        self.optimizer.set_weights(extra_data["optimizer_state"])

        # 最后，从检查点返回 epoch 数。这不是模型的属性，因此我们无法直接设置它，但用户可以将其传递给 fit()。
        return {"epoch": extra_data["epoch"]}
    # 准备 TensorFlow 数据集
    def prepare_tf_dataset(
        self,
        dataset: "datasets.Dataset",  # noqa:F821
        batch_size: int = 8,  # 设置批量大小，默认为 8
        shuffle: bool = True,  # 是否对数据进行随机打乱，默认为 True
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,  # 分词器，默认为空
        collate_fn: Optional[Callable] = None,  # 收集函数，默认为空
        collate_fn_args: Optional[Dict[str, Any]] = None,  # 收集函数参数，默认为空
        drop_remainder: Optional[bool] = None,  # 是否丢弃剩余的数据，默认为空
        prefetch: bool = True,  # 是否预取数据，默认为 True
    # 编译模型
    def compile(
        self,
        optimizer="rmsprop",  # 优化器，默认为 "rmsprop"
        loss="auto_with_warning",  # 损失函数，默认为 "auto_with_warning"
        metrics=None,  # 度量指标，默认为空
        loss_weights=None,  # 损失权重，默认为空
        weighted_metrics=None,  # 加权度量指标，默认为空
        run_eagerly=None,  # 是否立即运行，默认为空
        steps_per_execution=None,  # 每次执行的步骤数，默认为空
        **kwargs,  # 其他参数
    ):
        """
        这是一个薄包装器，如果用户没有指定损失函数，将模型的损失输出头设置为损失。
        """
        # 如果损失函数为 "auto_with_warning" 或 "passthrough"（为了向后兼容性而设定的），则使用模型的内部损失计算作为损失
        if loss in ("auto_with_warning", "passthrough"):
            logger.info(
                "No loss specified in compile() - the model's internal loss computation will be used as the "
                "loss. Don't panic - this is a common way to train TensorFlow models in Transformers! "
                "To disable this behaviour please pass a loss argument, or explicitly pass "
                "`loss=None` if you do not want your model to compute a loss. You can also specify `loss='auto'` to "
                "get the internal loss without printing this info string."
            )
            loss = "auto"  # 将损失函数设为 "auto"
        # 如果损失函数为 "auto"，则使用虚拟损失函数
        if loss == "auto":
            loss = dummy_loss  # 虚拟损失函数
            self._using_dummy_loss = True
        else:
            self._using_dummy_loss = False
        parent_args = list(inspect.signature(tf.keras.Model.compile).parameters.keys())  # 获取父类编译方法的参数列表
        # 这个参数已经更名，我们需要同时支持两个版本
        if "steps_per_execution" in parent_args:
            # 调用父类的编译方法，设置参数
            super().compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                steps_per_execution=steps_per_execution,
                **kwargs,
            )
        else:
            # 调用父类的编译方法，设置参数（兼容性处理）
            super().compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                experimental_steps_per_execution=steps_per_execution,
                **kwargs,
            )
```  
    # 定义一个方法用于计算损失，支持不定数量的位置参数和关键字参数
    def compute_loss(self, *args, **kwargs):
        # 如果 tf.keras.Model 类中有 compute_loss 方法
        if hasattr(tf.keras.Model, "compute_loss"):
            # 在 TensorFlow 2.8 或更新版本中会执行这里的逻辑
            return super().compute_loss(*args, **kwargs)  # 调用父类的 compute_loss 方法
        else:
            # 如果旧的 compute_loss 方法已弃用，因为在 TF 2.8 中新增了与 Keras 中 compute_loss 方法冲突的方法
            warnings.warn(
                "The old compute_loss method is deprecated as it conflicts with the Keras compute_loss "
                "method added in TF 2.8. If you want the original HF compute_loss, please call "
                "hf_compute_loss() instead. From TF versions >= 2.8, or Transformers versions >= 5, "
                "calling compute_loss() will get the Keras method instead.",
                FutureWarning,
            )
            # 返回调用 hf_compute_loss 方法的结果
            return self.hf_compute_loss(*args, **kwargs)

    # 定义一个方法用于获取标签到输出名称的映射关系
    def get_label_to_output_name_mapping(self):
        # 获取调用签名的参数名列表
        arg_names = list(inspect.signature(self.call).parameters)
        # 如果已经存在标签到输出名称的映射关系
        if self._label_to_output_map is not None:
            return self._label_to_output_map
        # 如果 call 方法的参数中有 "start_positions"
        elif "start_positions" in arg_names:
            # 返回指定的映射关系
            return {"start_positions": "start_logits", "end_positions": "end_logits"}
        # 如果 call 方法的参数中有 "sentence_order_label"
        elif "sentence_order_label" in arg_names:
            # 返回指定的映射关系
            return {"labels": "prediction_logits", "sentence_order_label": "sop_logits"}
        # 如果 call 方法的参数中有 "next_sentence_label"
        elif "next_sentence_label" in arg_names:
            # 返回指定的映射关系
            return {"labels": "prediction_logits", "next_sentence_label": "seq_relationship_logits"}
        # 如果 call 方法的参数中有 "mc_labels"
        elif "mc_labels" in arg_names:
            # 返回指定的映射关系
            return {"labels": "logits", "mc_labels": "mc_logits"}
        # 如果没有符合条件的情况
        else:
            # 返回空字典
            return {}

    # 定义一个方法用于创建模型卡片
    def create_model_card(
        self,
        output_dir,  # 输出目录
        model_name: str,  # 模型名称
        language: Optional[str] = None,  # 语言（可选）
        license: Optional[str] = None,  # 许可证（可选）
        tags: Optional[str] = None,  # 标签（可选）
        finetuned_from: Optional[str] = None,  # 微调来源（可选）
        tasks: Optional[str] = None,  # 任务（可选）
        dataset_tags: Optional[Union[str, List[str]]] = None,  # 数据集标签（可选）
        dataset: Optional[Union[str, List[str]]] = None,  # 数据集（可选）
        dataset_args: Optional[Union[str, List[str]]] = None,  # 数据集参数（可选）
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            output_dir (`str` or `os.PathLike`):
                The folder in which to create the model card.
            model_name (`str`, *optional*):
                The name of the model.
            language (`str`, *optional*):
                The language of the model (if applicable)
            license (`str`, *optional*):
                The license of the model. Will default to the license of the pretrained model used, if the original
                model given to the `Trainer` comes from a repo on the Hub.
            tags (`str` or `List[str]`, *optional*):
                Some tags to be included in the metadata of the model card.
            finetuned_from (`str`, *optional*):
                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
                of the original model given to the `Trainer` (if it comes from the Hub).
            tasks (`str` or `List[str]`, *optional*):
                One or several task identifiers, to be included in the metadata of the model card.
            dataset_tags (`str` or `List[str]`, *optional*):
                One or several dataset tags, to be included in the metadata of the model card.
            dataset (`str` or `List[str]`, *optional*):
                One or several dataset identifiers, to be included in the metadata of the model card.
            dataset_args (`str` or `List[str]`, *optional*):
               One or several dataset arguments, to be included in the metadata of the model card.
        """
        # 避免循环引入，根据需要执行此操作。
        from .modelcard import TrainingSummary  # tests_ignore

        # 使用训练摘要对象从 Keras 历史中获取训练摘要信息
        training_summary = TrainingSummary.from_keras(
            self,
            keras_history=self.history,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        # 将训练摘要转换为模型卡对象
        model_card = training_summary.to_model_card()
        # 将模型卡写入到输出目录下的 README.md 文件中
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(model_card)
    def set_input_embeddings(self, value):
        """
        Set model's input embeddings

        Args:
            value (`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        # 获取主层，例如 GPT2Model
        main_layer = getattr(self, self.base_model_prefix)

        # 如果主层为空，抛出未实现错误
        if main_layer is None:
            raise NotImplementedError("The model does not implements the base_model_prefix attribute.")

        try:
            # 尝试设置输入嵌入
            main_layer.set_input_embeddings(value)
        except AttributeError:
            # 如果出现属性错误，记录日志并构建模型
            logger.info("Building the model")
            self.build_in_name_scope()
            main_layer.set_input_embeddings(value)

    def get_output_embeddings(self) -> Union[None, tf.keras.layers.Layer]:
        """
        Returns the model's output embeddings

        Returns:
            `tf.Variable`: The new weights mapping vocabulary to hidden states.
        """
        # 如果存在语言模型头，则获取它
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()

            try:
                # 尝试获取输出嵌入
                return lm_head.get_output_embeddings()
            except AttributeError:
                # 如果出现属性错误，记录日志并构建模型
                logger.info("Building the model")
                self.build_in_name_scope()

                return lm_head().get_output_embeddings()

        return None  # 对于具有输出嵌入的模型进行覆盖

    def set_output_embeddings(self, value):
        """
        Set model's output embeddings

        Args:
            value (`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        # 如果存在语言模型头，则获取它
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                # 尝试设置输出嵌入
                lm_head.set_output_embeddings(value)
            except AttributeError:
                # 如果出现属性错误，记录日志并构建模型
                logger.info("Building the model")
                self.build_in_name_scope()
                lm_head.set_output_embeddings(value)

    def get_output_layer_with_bias(self) -> Union[None, tf.keras.layers.Layer]:
        """
        Get the layer that handles a bias attribute in case the model has an LM head with weights tied to the
        embeddings

        Return:
            `tf.keras.layers.Layer`: The layer that handles the bias, None if not an LM model.
        """
        # 发出警告，此方法已过时，请使用`get_lm_head`代替
        warnings.warn(
            "The method get_output_layer_with_bias is deprecated. Please use `get_lm_head` instead.", FutureWarning
        )
        # 返回语言模型头
        return self.get_lm_head()

    def get_prefix_bias_name(self) -> Union[None, str]:
        """
        Get the concatenated _prefix name of the bias from the model name to the parent layer

        Return:
            `str`: The _prefix name of the bias.
        """
        # 发出警告，此方法已过时，请使用`get_bias`代替
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return None
    def get_bias(self) -> Union[None, Dict[str, tf.Variable]]:
        """
        Dict of bias attached to an LM head. The key represents the name of the bias attribute.

        Return:
            `tf.Variable`: The weights representing the bias, None if not an LM model.
        """
        # 检查模型是否具有 LM 头
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                # 尝试获取 LM 头的偏置
                return lm_head.get_bias()
            except AttributeError:
                # 如果 LM 头没有 get_bias 方法，则尝试在名称空间内建立
                self.build_in_name_scope()
                # 再次尝试获取偏置
                return lm_head.get_bias()
        # 如果模型不是 LM 模型，则返回 None
        return None

    def set_bias(self, value):
        """
        Set all the bias in the LM head.

        Args:
            value (`Dict[tf.Variable]`):
                All the new bias attached to an LM head.
        """
        # 检查模型是否具有 LM 头
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                # 尝试设置 LM 头的偏置
                lm_head.set_bias(value)
            except AttributeError:
                # 如果 LM 头没有 set_bias 方法，则尝试在名称空间内建立
                self.build_in_name_scope()
                # 再次尝试设置偏置
                lm_head.set_bias(value)

    def get_lm_head(self) -> tf.keras.layers.Layer:
        """
        The LM Head layer. This method must be overwritten by all the models that have a lm head.

        Return:
            `tf.keras.layers.Layer`: The LM head layer if the model has one, None if not.
        """
        # 返回模型的 LM 头层，如果没有则返回 None
        return None

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None
    ) -> Union[tf.keras.layers.Embedding, tf.Variable]:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens without doing anything.

        Return:
            `tf.Variable` or `tf.keras.layers.Embedding`: Pointer to the input tokens of the model.
        """
        # TODO (joao): flagged for replacement (by `_v2_resized_token_embeddings`) due to embeddings refactor

        # 如果模型具有 keras embeddings 层，则运行新的代码路径
        if isinstance(self.get_input_embeddings(), tf.keras.layers.Embedding):
            return self._v2_resized_token_embeddings(new_num_tokens)

        # 如果 new_num_tokens 为 None 或者与配置中的 vocab_size 相同，则返回输入 token 的指针
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self._get_word_embedding_weight(self.get_input_embeddings())

        # 调整 token embeddings 矩阵的大小
        model_embeds = self._resize_token_embeddings(new_num_tokens)

        # 更新基础模型和当前模型配置中的 vocab_size
        self.config.vocab_size = new_num_tokens

        return model_embeds
    def _v2_resized_token_embeddings(self, new_num_tokens: Optional[int] = None) -> tf.keras.layers.Embedding:
        """
        调整模型的输入标记嵌入矩阵，如果 `new_num_tokens != config.vocab_size`。

        参数:
            new_num_tokens (`int`, *可选*):
                嵌入矩阵中新标记的数量。增加大小将在末尾添加新初始化的向量。减小大小将从末尾删除向量。如果未提供或为 `None`，则只返回指向输入标记的指针，不执行任何操作。

        返回:
            `tf.keras.layers.Embedding`: 模型的输入标记指针。
        """
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self.get_input_embeddings()

        model_embeds = self._v2_resize_token_embeddings(new_num_tokens)

        # 更新基础模型和当前模型配置
        self.config.vocab_size = new_num_tokens

        return model_embeds

    def _get_word_embedding_weight(model, embedding_layer):
        # TODO (joao): 标记为删除，因为嵌入重构

        # 如果变量保存权重本身，则返回它们
        if isinstance(embedding_layer, tf.Tensor):
            return embedding_layer
        # 否则，尝试从层的属性中获取它们

        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        # 属性不存在的原因可能是因为模型尚未构建，因此在构建模型后重试获取参数
        model.build_in_name_scope()

        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        return None
    # 调整模型中的 token embeddings 大小，用于支持新的 token 数量
    def _resize_token_embeddings(self, new_num_tokens):
        # TODO (joao): flagged for replacement (by `_v2_resize_token_embeddings`) due to embeddings refactor
        # 获取当前的 word embeddings 权重
        old_embeddings = self._get_word_embedding_weight(self.get_input_embeddings())
        # 调整 embeddings 大小以适应新的 token 数量
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)

        # 如果 word embeddings 没有被绑定，确保 lm head bias 也被调整大小
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)

            self.set_bias(new_lm_head_bias)

        # 如果 word embeddings 没有被绑定，确保 lm head decoder 也被调整大小
        if self.get_output_embeddings() is not None:
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)

            self.set_output_embeddings(new_lm_head_decoder)

        # 设置调整后的 embeddings
        self.set_input_embeddings(new_embeddings)

        # 返回调整后的输入 embeddings
        return self.get_input_embeddings()

    # 新版本的调整 token embeddings 大小函数
    def _v2_resize_token_embeddings(self, new_num_tokens):
        # 获取当前的输入 embeddings
        old_embeddings = self.get_input_embeddings()
        # 调整 embeddings 大小以适应新的 token 数量
        new_embeddings = self._v2_get_resized_embeddings(old_embeddings, new_num_tokens)
        # 设置调整后的输入 embeddings
        self.set_input_embeddings(new_embeddings)

        # 如果 word embeddings 没有被绑定，确保 lm head bias 也被调整大小
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._v2_get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
            self.set_bias(new_lm_head_bias)

        # 如果 word embeddings 没有被绑定，确保 lm head decoder 也被调整大小
        tied_weights = self.get_input_embeddings() == self.get_output_embeddings()
        if self.get_output_embeddings() is not None and not tied_weights:
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            # TODO (joao): this one probably needs a v2 version with other models
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)
            self.set_output_embeddings(new_lm_head_decoder)

        # 返回调整后的输入 embeddings
        return self.get_input_embeddings()
    # 从旧的 lm 头偏置构建一个调整大小后的偏置。增加大小将在末尾添加新初始化的向量。减小大小将从末尾移除向量
    def _get_resized_lm_head_bias(self, old_lm_head_bias, new_num_tokens):
        """
        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_bias (`tf.Variable`):
                Old lm head bias to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns None

        Return:
            `tf.Variable`: Pointer to the resized bias.
        """
        # TODO (joao): flagged for replacement (by `_v2_get_resized_lm_head_bias`) due to embeddings refactor
        # 创建一个新的 lm 头偏置的字典
        new_lm_head_bias = {}

        # 遍历旧的 lm 头偏置字典
        for attr, weight in old_lm_head_bias.items():
            # 获取权重的形状信息
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            # 计算大小差异
            size_diff = new_num_tokens - old_num_tokens
            final_shape = [new_num_tokens] if first_dim is None else [first_dim, new_num_tokens]

            # 初始化新的偏置
            if tf.math.greater(size_diff, 0):
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                current_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape), constant_values=-1)
                num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
                mask_shape = [num_tokens_to_copy] if first_dim is None else [1, num_tokens_to_copy]
                bias_mask = tf.fill(tf.convert_to_tensor(mask_shape), True)
                bias_mask = tf.pad(bias_mask, tf.convert_to_tensor(padding_shape), constant_values=False)
            else:
                slice_from = [0] if first_dim is None else [0, 0]
                current_bias = tf.slice(
                    weight.value(), tf.convert_to_tensor(slice_from), tf.convert_to_tensor(final_shape)
                )
                bias_mask = tf.fill(tf.convert_to_tensor(final_shape), True)

            # 添加新的权重
            new_bias = self.add_weight(
                shape=final_shape,
                initializer="zeros",
                trainable=True,
                name=weight.name.split(":")[0],
            )
            init_bias = tf.where(bias_mask, current_bias, new_bias.value())

            # 将初始化后的偏置赋值给新的偏置
            new_bias.assign(init_bias)
            new_lm_head_bias[attr] = new_bias

        return new_lm_head_bias

    def _v2_get_resized_lm_head_bias(
        self, old_lm_head_bias: Dict[str, tf.Variable], new_num_tokens: int
    ) -> Dict[str, tf.Tensor]:
        """
        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_bias (`Dict[str, tf.Variable]`):
                Old lm head bias to be resized.
            new_num_tokens (`int`):
                New number of tokens in the linear matrix. Increasing the size will add newly initialized vectors at
                the end. Reducing the size will remove vectors from the end.

        Return:
            `tf.Tensor`: Values for the resized bias.
        """
        # Initialize an empty dictionary to store the resized bias values
        new_lm_head_bias = {}

        # Iterate through each attribute and corresponding weight in the old_lm_head_bias dictionary
        for attr, weight in old_lm_head_bias.items():
            # Determine the size difference (depending on the shape)
            # If the rank of the weight tensor is 1, set the first_dim to None, otherwise extract the first dimension
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            # Calculate the size difference between the new and old number of tokens
            size_diff = new_num_tokens - old_num_tokens

            # Copy the old bias values to the new bias
            if old_num_tokens > new_num_tokens:
                # If the old number of tokens is greater than the new number, slice the old bias tensor to match the new size
                new_bias = weight.value()[..., :new_num_tokens]
            else:
                # If the new number of tokens is greater than the old number, pad the old bias tensor to match the new size
                # Define the padding shape based on whether the first_dim is None or not
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                # Pad the old bias tensor with zeros to match the new size
                new_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape))

            # Store the resized bias tensor in the new_lm_head_bias dictionary
            new_lm_head_bias[attr] = new_bias
        
        # Return the dictionary containing the resized bias tensors
        return new_lm_head_bias
    def _get_resized_lm_head_decoder(self, old_lm_head_decoder, new_num_tokens):
        """
        Build a resized decoder from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_decoder (`tf.Variable`):
                Old lm head decoder to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns None

        Return:
            `tf.Variable`: Pointer to the resized decoder or None if the output embeddings are different from the input
            ones.
        """
        # 将传入的旧 lm 头部解码器赋值给新的解码器
        new_lm_head_decoder = old_lm_head_decoder
        # 检查输入和输出是否相等
        is_input_output_equals = tf.reduce_any(
            self._get_word_embedding_weight(self.get_input_embeddings()) == old_lm_head_decoder
        )

        # 如果旧的头部解码器不为空且输入和输出不相等
        if old_lm_head_decoder is not None and not is_input_output_equals:
            # 获取旧解码器的维度
            old_embedding_dim = shape_list(old_lm_head_decoder)[1]
            # 初始化解码器掩码和当前解码器
            decoder_mask, current_decoder = init_copy_embeddings(old_lm_head_decoder, new_num_tokens)
            # 添加新的权重，形状为 (新标记数, 旧嵌入维度)
            new_lm_head_decoder = self.add_weight(
                shape=(new_num_tokens, old_embedding_dim),
                initializer="zeros",
                trainable=True,
                name=old_lm_head_decoder.name.split(":")[0],
            )
            # 根据解码器掩码，选择初始化的解码器或新的解码器值
            init_decoder = tf.where(decoder_mask, current_decoder, new_lm_head_decoder.value())

            # 将初始化的解码器赋值给新的解码器
            new_lm_head_decoder.assign(init_decoder)

        # 返回新的解码器
        return new_lm_head_decoder
    # 从给定的 token Embedding 权重构建一个调整大小的 Embedding 权重。增加大小将在末尾添加新初始化的向量，减小大小将从末尾移除向量。
    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None) -> tf.Variable:
        """
        Build a resized Embedding weights from a provided token Embedding weights. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`tf.Variable`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `tf.Variable` module of the model without doing anything.

        Return:
            `tf.Variable`: Pointer to the resized Embedding Module or the old Embedding Module if `new_num_tokens` is
            `None`
        """
        # 获取旧嵌入维度
        old_embedding_dim = shape_list(old_embeddings)[1]
        # 获取初始化范围，默认为 0.02
        init_range = getattr(self.config, "initializer_range", 0.02)
        # 初始化嵌入掩码和当前嵌入
        embeddings_mask, current_embeddings = init_copy_embeddings(old_embeddings, new_num_tokens)
        # 创建一个新的权重变量，形状为 [new_num_tokens, old_embedding_dim]
        new_embeddings = self.add_weight(
            name=old_embeddings.name.split(":")[0],
            shape=[new_num_tokens, old_embedding_dim],
            initializer=get_initializer(init_range),
            dtype=tf.float32,
        )
        # 用当前嵌入或新嵌入的值来初始化新嵌入，根据嵌入掩码进行选择
        init_embeddings = tf.where(embeddings_mask, current_embeddings, new_embeddings.value())
        # 将初始化的新嵌入赋值给新嵌入变量
        new_embeddings.assign(init_embeddings)

        # 返回新的嵌入变量
        return new_embeddings

    # TODO (joao): flagged for replacement (by `_v2_get_resized_embeddings`) due to embeddings refactor
    # 获取调整大小后的嵌入（版本2）
    def _v2_get_resized_embeddings(
        self, old_embeddings: tf.keras.layers.Embedding, new_num_tokens: int
    def resize_embedding(self, old_embeddings: tf.keras.layers.Embedding, new_num_tokens: int = None) -> tf.keras.layers.Embedding:
        """
        Build a resized Embedding layer from a provided Embedding layer. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end.

        Args:
            old_embeddings (`tf.keras.layers.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

        Return:
            `tf.keras.layers.Embedding`: Resized Embedding layer.
        """

        # Get the initialization range for the embeddings
        init_range = 0.02  # default value
        potential_initialization_variable_names = [
            "initializer_range",  # most common
            "initializer_factor",  # e.g. T5
            "init_std",  # e.g BART
        ]
        for var_name in potential_initialization_variable_names:
            if hasattr(self.config, var_name):
                init_range = getattr(self.config, var_name)

        # Get a new (initialized) embeddings layer
        new_embeddings = tf.keras.layers.Embedding(
            input_dim=new_num_tokens,
            output_dim=old_embeddings.output_dim,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=init_range),
            name=old_embeddings.embeddings.name[:-13],  # exact same scoped name except "/embeddings:0"
        )
        new_embeddings(tf.constant([[0]]))

        # Copy the old embeddings to the new embeddings
        if old_embeddings.input_dim >= new_num_tokens:
            init_embeddings = old_embeddings.embeddings[:new_num_tokens]
        else:
            init_embeddings = tf.concat(
                [old_embeddings.embeddings, new_embeddings.embeddings[old_embeddings.input_dim :]], axis=0
            )
        new_embeddings.embeddings.assign(init_embeddings)
        return new_embeddings

    def prune_heads(self, heads_to_prune):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        raise NotImplementedError

    def save_pretrained(
        self,
        save_directory,
        saved_model=False,
        version=1,
        push_to_hub=False,
        signatures=None,
        max_shard_size: Union[int, str] = "10GB",
        create_pr: bool = False,
        safe_serialization: bool = False,
        token: Optional[Union[str, bool]] = None,
        **kwargs,
    @classmethod
    # 从预训练模型名称或路径创建一个实例
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
    # 将模型推送到 Hub
    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        token: Optional[Union[bool, str]] = None,
        # (`use_auth_token` is deprecated: we have to keep it here as we don't have **kwargs)
        use_auth_token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        **base_model_card_args,
    # 为自动类注册类
    @classmethod
    def register_for_auto_class(cls, auto_class="TFAutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"TFAutoModel"`):
                The auto class to register this new model with.
        """
        # 如果 auto_class 不是字符串，则获取其类名
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入自动模块
        import transformers.models.auto as auto_module

        # 检查 auto_class 是否存在于自动模块中
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将 auto_class 设置为当前类的自动类
        cls._auto_class = auto_class
# 定义一个 1D 卷积层，与 Radford 等人为 OpenAI GPT（也用于 GPT-2）定义的一维卷积层相同。
# 基本上，它的工作原理类似于线性层，但权重是转置的。
class TFConv1D(tf.keras.layers.Layer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`):
            The number of output features. 输出特征的数量。
        nx (`int`):
            The number of input features. 输入特征的数量。
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation to use to initialize the weights. 用于初始化权重的标准差。
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`. 传递给`tf.keras.layers.Layer`的`__init__`的额外关键字参数。
    """

    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        # 如果已经构建了层，直接返回
        if self.built:
            return
        self.built = True
        # 添加权重，初始化为给定形状的矩阵，使用指定的标准差初始化
        self.weight = self.add_weight(
            "weight", shape=[self.nx, self.nf], initializer=get_initializer(self.initializer_range)
        )
        # 添加偏置，初始化为给定形状的零矩阵
        self.bias = self.add_weight("bias", shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        # 获取输入张量的批量大小和序列长度
        bz, sl = shape_list(x)[:2]

        # 将输入张量重塑为二维张量
        x = tf.reshape(x, [-1, self.nx])
        # 执行矩阵乘法操作，并加上偏置
        x = tf.matmul(x, self.weight) + self.bias

        # 将结果重新塑造回原来的形状
        x = tf.reshape(x, [bz, sl, self.nf])

        # 返回结果张量
        return x


class TFSharedEmbeddings(tf.keras.layers.Layer):
    r"""
    Construct shared token embeddings.

    The weights of the embedding layer is usually shared with the weights of the linear decoder when doing language
    modeling.

    Args:
        vocab_size (`int`):
            The size of the vocabulary, e.g., the number of unique tokens.
        hidden_size (`int`):
            The size of the embedding vectors.
        initializer_range (`float`, *optional*):
            The standard deviation to use when initializing the weights. If no value is provided, it will default to
            \\(1/\sqrt{hidden\_size}\\).
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`.
    """

    # TODO (joao): flagged for delection due to embeddings refactor

    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: Optional[float] = None, **kwargs):
        # 初始化函数
        super().__init__(**kwargs)
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置嵌入向量的大小
        self.hidden_size = hidden_size
        # 如果未提供初始化范围，则默认为 hidden_size 的倒数平方
        self.initializer_range = hidden_size**-0.5 if initializer_range is None else initializer_range
        # 发出警告，TFSharedEmbeddings 将在 v4.32 版本中删除，请使用 `tf.keras.layers.Embedding` 替代。
        warnings.warn(
            "`TFSharedEmbeddings` is scheduled for deletion in v4.32, use `tf.keras.layers.Embedding` instead.",
            DeprecationWarning,
        )
    def build(self, input_shape):
        """
        Build shared token embedding layer. Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        # 添加权重参数，用于嵌入层
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)
        )
        # 调用父类的 build 方法
        super().build(input_shape)

    def get_config(self):
        # 获取层的配置参数
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        }
        # 调用父类的 get_config 方法获取基础配置参数
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor, mode: str = "embedding") -> tf.Tensor:
        """
        获取输入的标记嵌入或解码最终隐藏状态。

        Args:
            inputs (`tf.Tensor`):
                在嵌入模式下，应该是形状为 `[batch_size, length]` 的 int64 张量。

                在线性模式下，应该是形状为 `[batch_size, length, hidden_size]` 的 float 张量。
            mode (`str`, 默认为 `"embedding"`):
               有效值为 `"embedding"` 或 `"linear"`，第一个表示该层应该用作嵌入层，第二个表示该层应该用作线性解码器。

        Returns:
            `tf.Tensor`: 在嵌入模式下，输出是一个形状为 `[batch_size, length, embedding_size]` 的 float32 嵌入张量。

            在线性模式下，输出是一个形状为 `[batch_size, length, vocab_size]` 的 float32 张量。

        Raises:
            ValueError: 如果 `mode` 不合法。

        共享权重逻辑从此处调整:
        [here](https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24).
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, input_ids):
        """基于输入张量应用嵌入。"""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """
        通过线性层运行输入来计算 logits。

        Args:
            inputs: 形状为 [..., hidden_size] 的 float32 张量。

        Returns:
            形状为 [..., vocab_size] 的 float32 张量。
        """
        first_dims = shape_list(inputs)[:-1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])
class TFSequenceSummary(tf.keras.layers.Layer):
    """
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

        initializer_range (`float`, defaults to 0.02): The standard deviation to use to initialize the weights.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`.
    """
    # 初始化函数，接受预训练配置和初始化范围作为参数
    def __init__(self, config: PretrainedConfig, initializer_range: float = 0.02, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 根据配置确定摘要类型
        self.summary_type = config.summary_type if hasattr(config, "summary_use_proj") else "last"
        # 如果摘要类型为"attn"，则抛出未实现的错误
        if self.summary_type == "attn":
            raise NotImplementedError

        # 检查是否有摘要，并根据配置决定是否使用投影
        self.has_summary = hasattr(config, "summary_use_proj") and config.summary_use_proj
        if self.has_summary:
            # 如果配置中指定了投影到标签并且标签数大于0，则使用标签数作为类别数，否则使用隐藏层大小
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            # 创建一个全连接层作为摘要层
            self.summary = tf.keras.layers.Dense(
                num_classes, kernel_initializer=get_initializer(initializer_range), name="summary"
            )

        # 检查是否有激活函数，并根据配置确定激活函数类型
        self.has_activation = False
        activation_string = getattr(config, "summary_activation", None)
        if activation_string is not None:
            self.has_activation = True
            self.activation = get_tf_activation(activation_string)

        # 检查是否有第一个dropout，并根据配置确定dropout比例
        self.has_first_dropout = hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0
        if self.has_first_dropout:
            self.first_dropout = tf.keras.layers.Dropout(config.summary_first_dropout)

        # 检查是否有最后一个dropout，并根据配置确定dropout比例
        self.has_last_dropout = hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0
        if self.has_last_dropout:
            self.last_dropout = tf.keras.layers.Dropout(config.summary_last_dropout)
        # 记录隐藏层大小
        self.hidden_size = config.hidden_size
    # 定义一个方法，用于处理输入数据并返回输出结果
    def call(self, inputs, cls_index=None, training=False):
        # 检查输入数据类型，如果不是字典、元组或列表，则将其作为隐藏状态
        if not isinstance(inputs, (dict, tuple, list)):
            hidden_states = inputs
        # 如果输入数据是元组或列表，则将第一个元素作为隐藏状态，第二个元素作为cls_index
        elif isinstance(inputs, (tuple, list)):
            hidden_states = inputs[0]
            cls_index = inputs[1] if len(inputs) > 1 else None
            assert len(inputs) <= 2, "Too many inputs."
        # 如果输入数据是字典，则从中获取hidden_states和cls_index
        else:
            hidden_states = inputs.get("hidden_states")
            cls_index = inputs.get("cls_index", None)

        # 根据summary_type的不同进行不同的汇总操作
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = tf.reduce_mean(hidden_states, axis=1)
        elif self.summary_type == "cls_index":
            # 根据cls_index获取对应的hidden_states
            hidden_shape = shape_list(hidden_states)  # e.g. [batch, num choices, seq length, hidden dims]
            if cls_index is None:
                cls_index = tf.fill(
                    hidden_shape[:-2], hidden_shape[-2] - 1
                )  # A tensor full of shape [batch] or [batch, num choices] full of sequence length
            cls_shape = shape_list(cls_index)
            if len(cls_shape) <= len(hidden_shape) - 2:
                cls_index = tf.expand_dims(cls_index, axis=-1)
            output = tf.gather(hidden_states, cls_index, batch_dims=len(hidden_shape) - 2)
            output = tf.squeeze(
                output, axis=len(hidden_shape) - 2
            )  # shape of output: (batch, num choices, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        # 如果有第一个dropout操作，则对output进行处理
        if self.has_first_dropout:
            output = self.first_dropout(output, training=training)

        # 如果有summary操作，则对output进行处理
        if self.has_summary:
            output = self.summary(output)

        # 如果有激活函数，则对output进行处理
        if self.has_activation:
            output = self.activation(output)

        # 如果有最后一个dropout操作，��对output进行处理
        if self.has_last_dropout:
            output = self.last_dropout(output, training=training)

        # 返回处理后的output
        return output

    # 构建模型
    def build(self, input_shape):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果有summary操作，则构建summary
        if getattr(self, "summary", None) is not None:
            with tf.name_scope("summary"):
                self.summary.build(self.hidden_size)
# 定义一个函数，用于创建具有给定范围的 `tf.keras.initializers.TruncatedNormal` 初始化器
def get_initializer(initializer_range: float = 0.02) -> tf.keras.initializers.TruncatedNormal:
    """
    创建一个具有给定范围的 `tf.keras.initializers.TruncatedNormal` 初始化器。

    Args:
        initializer_range (*float*, defaults to 0.02): 初始化器范围的标准差。

    Returns:
        `tf.keras.initializers.TruncatedNormal`: 截断正态分布的初始化器。
    """
    # 返回一个截断正态分布初始化器，标准差为给定的初始化器范围
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
```