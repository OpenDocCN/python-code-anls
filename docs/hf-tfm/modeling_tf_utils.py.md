# `.\modeling_tf_utils.py`

```py
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TF general model utils."""

from __future__ import annotations  # Future import to allow forward references

import functools  # Importing functools module for higher-order functions
import gc  # Importing gc module for garbage collection utilities
import inspect  # Importing inspect module for examining live objects
import json  # Importing json module for JSON encoding and decoding
import os  # Importing os module for operating system functionalities
import pickle  # Importing pickle module for object serialization
import re  # Importing re module for regular expressions
import warnings  # Importing warnings module for issuing warnings

from collections.abc import Mapping  # Importing Mapping from collections.abc for ABCs of collections
from pathlib import Path  # Importing Path from pathlib for object-oriented filesystem paths
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union  # Importing typing modules for type hints

import h5py  # Importing h5py for HDF5 file support
import numpy as np  # Importing numpy for numerical computing
import tensorflow as tf  # Importing tensorflow library

from packaging.version import parse  # Importing parse from packaging.version for version parsing

from . import DataCollatorWithPadding, DefaultDataCollator  # Importing local modules
from .activations_tf import get_tf_activation  # Importing get_tf_activation from activations_tf module
from .configuration_utils import PretrainedConfig  # Importing PretrainedConfig from configuration_utils module
from .dynamic_module_utils import custom_object_save  # Importing custom_object_save from dynamic_module_utils module
from .generation import GenerationConfig, TFGenerationMixin  # Importing GenerationConfig and TFGenerationMixin
from .tf_utils import (
    convert_batch_encoding,  # Importing convert_batch_encoding function
    expand_1d,  # Importing expand_1d function
    load_attributes_from_hdf5_group,  # Importing load_attributes_from_hdf5_group function
    save_attributes_to_hdf5_group,  # Importing save_attributes_to_hdf5_group function
    shape_list,  # Importing shape_list function
)

from .utils import (
    SAFE_WEIGHTS_INDEX_NAME,  # Importing constants from utils module
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_INDEX_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ModelOutput,  # Importing ModelOutput class
    PushToHubMixin,  # Importing PushToHubMixin class
    cached_file,  # Importing cached_file function
    download_url,  # Importing download_url function
    find_labels,  # Importing find_labels function
    has_file,  # Importing has_file function
    is_offline_mode,  # Importing is_offline_mode function
    is_remote_url,  # Importing is_remote_url function
    is_safetensors_available,  # Importing is_safetensors_available function
    is_tf_symbolic_tensor,  # Importing is_tf_symbolic_tensor function
    logging,  # Importing logging utilities
    requires_backends,  # Importing requires_backends decorator
    working_or_temp_dir,  # Importing working_or_temp_dir function
)
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files  # Importing hub-related utilities

# Checking if safetensors library is available and importing related functions if so
if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.tensorflow import save_file as safe_save_file

# Checking if TYPE_CHECKING is True, then importing PreTrainedTokenizerBase from local module
if TYPE_CHECKING:
    from . import PreTrainedTokenizerBase

# Getting logger from logging utilities
logger = logging.get_logger(__name__)

# Setting TF_USE_LEGACY_KERAS environment variable to '1' for compatibility with Keras 2
if "TF_USE_LEGACY_KERAS" not in os.environ:
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
elif os.environ["TF_USE_LEGACY_KERAS"] != "1":
    # Warning if TF_USE_LEGACY_KERAS is set to '0' explicitly, which may cause issues with Transformers models
    logger.warning(
        "Transformers is only compatible with Keras 2, but you have explicitly set `TF_USE_LEGACY_KERAS` to `0`. "
        "This may result in unexpected behaviour or errors if Keras 3 objects are passed to Transformers models."
    )

# Attempting to import tf_keras as keras and backend as K, falling back to keras and keras.backend if not available
try:
    import tf_keras as keras
    from tf_keras import backend as K
except (ModuleNotFoundError, ImportError):
    import keras
    from keras import backend as K
    # 检查导入的 Keras 版本是否大于 2
    if parse(keras.__version__).major > 2:
        # 如果版本大于 2，则抛出值错误异常
        raise ValueError(
            "Your currently installed version of Keras is Keras 3, but this is not yet supported in "
            "Transformers. Please install the backwards-compatible tf-keras package with "
            "`pip install tf-keras`."
        )
# 获取 TensorFlow 的日志记录器对象
tf_logger = tf.get_logger()

# 定义一个类型别名，表示可以作为 TF 模型的输入的多种可能类型
TFModelInputType = Union[
    List[tf.Tensor],         # 列表中包含 TensorFlow 张量
    List[np.ndarray],        # 列表中包含 NumPy 数组
    Dict[str, tf.Tensor],    # 字典，键是字符串，值是 TensorFlow 张量
    Dict[str, np.ndarray],   # 字典，键是字符串，值是 NumPy 数组
    tf.Tensor,               # 单个 TensorFlow 张量
    np.ndarray,              # 单个 NumPy 数组
]

# 定义一个简单的损失函数，如果预测值的维度小于等于 1，则直接返回预测值，否则返回沿指定轴的均值
def dummy_loss(y_true, y_pred):
    if y_pred.shape.rank <= 1:
        return y_pred
    else:
        reduction_axes = list(range(1, y_pred.shape.rank))
        return tf.reduce_mean(y_pred, axis=reduction_axes)


class TFModelUtilsMixin:
    """
    `keras.Model` 的几个实用工具方法，作为 Mixin 使用。
    """

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        获取模型中参数的数量（可选只计算可训练的参数）。

        Args:
            only_trainable (`bool`, *optional*, 默认为 `False`):
                是否只返回可训练参数的数量。

        Returns:
            `int`: 参数的数量。
        """
        if only_trainable:
            return int(sum(np.prod(w.shape.as_list()) for w in self.trainable_variables))
        else:
            return self.count_params()


def keras_serializable(cls):
    """
    装饰一个 Keras 层类，以支持 Keras 序列化。

    这是通过以下方式实现的：

    1. 在 `get_config` 中为 Keras 配置字典添加 `transformers_config` 字典（在序列化时由 Keras 调用）。
    2. 包装 `__init__` 方法以接受 `transformers_config` 字典（在反序列化时由 Keras 传递）并将其转换为实际层初始化器的配置对象。
    3. 在 Keras 中注册该类作为自定义对象（如果 Tensorflow 版本支持），因此在调用 `keras.models.load_model` 时不需要在 `custom_objects` 中提供它。

    Args:
        cls (a `keras.layers.Layers subclass`):
            通常是项目中的 `TF.MainLayer` 类，一般必须接受 `config` 参数作为其初始化器。

    Returns:
        经过修改以支持 Keras 反序列化的同一类对象。
    """
    initializer = cls.__init__

    config_class = getattr(cls, "config_class", None)
    if config_class is None:
        raise AttributeError("Must set `config_class` to use @keras_serializable")

    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        config = args[0] if args and isinstance(args[0], PretrainedConfig) else kwargs.pop("config", None)

        if isinstance(config, dict):
            config = config_class.from_dict(config)
            initializer(self, config, *args, **kwargs)
        elif isinstance(config, PretrainedConfig):
            if len(args) > 0:
                initializer(self, *args, **kwargs)
            else:
                initializer(self, config, *args, **kwargs)
        else:
            raise ValueError("Must pass either `config` (PretrainedConfig) or `config` (dict)")

        self._config = config
        self._kwargs = kwargs

    cls.__init__ = wrapped_init
    # 检查类 cls 是否具有 get_config 方法，如果没有，则抛出 TypeError 异常
    if not hasattr(cls, "get_config"):
        raise TypeError("Only use @keras_serializable on keras.layers.Layer subclasses")
    
    # 检查 cls 的 get_config 方法是否具有 "_is_default" 属性
    if hasattr(cls.get_config, "_is_default"):
        
        # 定义新的 get_config 方法，用于序列化对象的配置信息
        def get_config(self):
            # 调用父类的 get_config 方法，获取默认配置
            cfg = super(cls, self).get_config()
            # 将当前对象的配置转换为字典，并存储在 cfg["config"] 中
            cfg["config"] = self._config.to_dict()
            # 将对象的关键字参数更新到 cfg 中
            cfg.update(self._kwargs)
            return cfg
        
        # 将新定义的 get_config 方法赋值给 cls 的 get_config 属性
        cls.get_config = get_config
    
    # 将 _keras_serializable 标记设置为 True，表示对象已经被序列化
    cls._keras_serializable = True
    
    # 如果 keras.utils 中存在 register_keras_serializable 方法，则注册 cls
    if hasattr(keras.utils, "register_keras_serializable"):
        cls = keras.utils.register_keras_serializable()(cls)
    
    # 返回经过处理的 cls 对象
    return cls
# 定义一个适用于因果语言建模（CLM）的损失函数类，即猜测下一个标记的任务。
class TFCausalLanguageModelingLoss:
    """
    Loss function suitable for causal language modeling (CLM), that is, the task of guessing the next token.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    # 使用标签和logits计算损失的方法
    def hf_compute_loss(self, labels, logits):
        # 定义稀疏分类交叉熵损失函数，from_logits=True 表示输入为 logits
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        
        # 如果配置为 tf_legacy_loss，则仅仅处理不等于 -100 的标签
        if self.config.tf_legacy_loss:
            # 创建一个布尔掩码，标记所有不等于 -100 的位置
            active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            # 使用布尔掩码过滤 logits，并降维处理
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            # 使用布尔掩码过滤标签，并降维处理
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
            return loss_fn(labels, reduced_logits)
        
        # 将负标签裁剪为零，以避免 NaN 和错误，这些位置将在后续被掩码
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # 创建一个损失掩码，确保仅处理不等于 -100 的标签
        loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
        # 应用损失掩码到未掩码的损失
        masked_loss = unmasked_loss * loss_mask
        # 计算平均掩码后的损失
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))


class TFQuestionAnsweringLoss:
    """
    Loss function suitable for question answering.
    """

    # 使用标签和logits计算损失的方法
    def hf_compute_loss(self, labels, logits):
        # 定义稀疏分类交叉熵损失函数，from_logits=True 表示输入为 logits
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        # 计算起始位置的损失
        start_loss = loss_fn(labels["start_position"], logits[0])
        # 计算结束位置的损失
        end_loss = loss_fn(labels["end_position"], logits[1])
        # 返回起始和结束位置损失的平均值
        return (start_loss + end_loss) / 2.0


class TFTokenClassificationLoss:
    """
    Loss function suitable for token classification.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """
    # 定义一个方法用于计算损失，需要传入标签和对数概率
    def hf_compute_loss(self, labels, logits):
        # 使用稀疏分类交叉熵损失函数，设置为从对数概率计算，不进行损失的汇总
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        
        # 如果当前是即时执行模式（eager execution），则执行以下条件判断
        if tf.executing_eagerly():  # Data-dependent conditionals are forbidden in XLA
            # 如果标签中存在值为 -1 的情况，打印警告信息，建议使用 -100 替代 -1 来屏蔽损失
            if tf.math.reduce_any(labels == -1):
                tf.print("Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.")
        
        # 如果配置中指定使用传统的 TensorFlow 损失计算方法
        if self.config.tf_legacy_loss:
            # 如果标签中存在值为 -1 的情况，打印警告信息，建议使用 -100 替代 -1 来屏蔽损失
            if tf.math.reduce_any(labels == -1):
                tf.print("Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.")
                # 将标签中不等于 -1 的位置筛选出来，作为有效的损失位置
                active_loss = tf.reshape(labels, (-1,)) != -1
            else:
                # 将标签中不等于 -100 的位置筛选出来，作为有效的损失位置
                active_loss = tf.reshape(labels, (-1,)) != -100
            
            # 从 logits 中筛选出有效的预测值，并且展平为一维数组
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            # 从标签中筛选出有效的标签值，并且展平为一维数组
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
            
            # 返回计算后的损失值
            return loss_fn(labels, reduced_logits)
        
        # 对负数标签进行裁剪，转换为零，避免出现 NaN 和错误，这些位置之后会被屏蔽掉
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        
        # 确保只有标签不等于 -100 或 -1 的位置被计入损失计算
        loss_mask = tf.cast(labels >= 0, dtype=unmasked_loss.dtype)
        
        # 避免之后可能出现的除以零错误
        # 屏蔽掉的位置将因为 -100 和 -1 不是有效标签而导致损失为 NaN
        masked_loss = unmasked_loss * loss_mask
        
        # 计算屏蔽后的损失总和，并除以有效损失位置的总数来得到平均损失
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        
        # 将结果重新整形为长度为 1 的张量，并返回
        return tf.reshape(reduced_masked_loss, (1,))
class TFSequenceClassificationLoss:
    """
    Loss function suitable for sequence classification.
    """

    def hf_compute_loss(self, labels, logits):
        # 如果 logits 的形状是 1 维或者第二维是 1，使用均方误差损失函数
        if logits.shape.rank == 1 or logits.shape[1] == 1:
            loss_fn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
            if labels.shape.rank == 1:
                # 如果 labels 是 1 维的，则将其扩展为二维
                labels = tf.expand_dims(labels, axis=-1)
        else:
            # 否则使用稀疏分类交叉熵损失函数
            loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=keras.losses.Reduction.NONE
            )

        return loss_fn(labels, logits)


class TFMultipleChoiceLoss:
    """Loss function suitable for multiple choice tasks."""

    def hf_compute_loss(self, labels, logits):
        # 使用稀疏分类交叉熵损失函数
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
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
        # 使用稀疏分类交叉熵损失函数
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            # 确保仅计算不等于 -100 的标签作为损失
            next_sentence_active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            next_sentence_reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, 2)), next_sentence_active_loss)
            next_sentence_label = tf.boolean_mask(tf.reshape(labels, (-1,)), next_sentence_active_loss)

            return loss_fn(next_sentence_label, next_sentence_reduced_logits)

        # 确保仅计算不等于 -100 的标签作为损失

        # 在这里将负标签剪切为零，以避免 NaN 和错误 - 这些位置后续将被屏蔽
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels), y_pred=logits)
        ns_loss_mask = tf.cast(labels != -100, dtype=unmasked_ns_loss.dtype)
        # 将标签为 -100 的样本归零，不进行减少
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask

        return masked_ns_loss


def booleans_processing(config, **kwargs):
    """
    Process the input booleans of each model.
    """
    # 创建一个空字典，用于存储最终的布尔值选项
    final_booleans = {}
    
    # 如果在传入的参数 kwargs 中存在 "output_attentions"，则处理其布尔值设定：
    # 如果 kwargs["output_attentions"] 不为 None，则使用它；否则使用 config.output_attentions 的值
    if "output_attentions" in kwargs:
        final_booleans["output_attentions"] = (
            kwargs["output_attentions"] if kwargs["output_attentions"] is not None else config.output_attentions
        )
    
    # 处理 "output_hidden_states" 的布尔值设定：
    # 如果 kwargs["output_hidden_states"] 不为 None，则使用它；否则使用 config.output_hidden_states 的值
    final_booleans["output_hidden_states"] = (
        kwargs["output_hidden_states"] if kwargs["output_hidden_states"] is not None else config.output_hidden_states
    )
    
    # 处理 "return_dict" 的布尔值设定：
    # 如果 kwargs["return_dict"] 不为 None，则使用它；否则使用 config.return_dict 的值
    final_booleans["return_dict"] = kwargs["return_dict"] if kwargs["return_dict"] is not None else config.return_dict
    
    # 如果在 kwargs 中有 "use_cache" 参数，则处理其布尔值设定：
    # 如果 kwargs["use_cache"] 不为 None，则使用它；否则尝试使用 config.use_cache 的值，如果 config 没有 use_cache 属性则为 None
    if "use_cache" in kwargs:
        final_booleans["use_cache"] = (
            kwargs["use_cache"] if kwargs["use_cache"] is not None else getattr(config, "use_cache", None)
        )
    
    # 返回存储了所有布尔选项的字典
    return final_booleans
# 定义一个装饰器函数，用于处理传递给 Keras 层的输入参数，将它们作为关键字参数传递给层。这样可以通过它们的变量名在下游使用这些输入，即使它们作为字典打包在第一个输入中（在 Keras 中很常见）。

original_signature = inspect.signature(func)
# 获取传入函数的原始签名信息

@functools.wraps(func)
def run_call_with_unpacked_inputs(self, *args, **kwargs):
    # 从装饰函数的 kwargs 中隔离出实际的 `**kwargs`
    kwargs_call = {key: val for key, val in kwargs.items() if key not in dict(original_signature.parameters)}
    # 从 kwargs 中分离出用于函数调用的参数和关键字参数
    fn_args_and_kwargs = {key: val for key, val in kwargs.items() if key not in kwargs_call}
    fn_args_and_kwargs.update({"kwargs_call": kwargs_call})

    # 如果存在任何参数，将其移动到 kwargs 中
    fn_args_and_kwargs.update(dict(zip(func.__code__.co_varnames[1:], args)))

    # 对于 EncoderDecoder 模型，将配置选项应用于其内部模型。
    if "EncoderDecoder" in self.__class__.__name__:
        config = None
    else:
        config = self.config

    # 调用 input_processing 函数处理输入
    unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
    # 调用原始函数并传递解包后的输入
    return func(self, **unpacked_inputs)

# Keras 要求传递第一个层参数，并通过 `inspect.getfullargspec()` 进行检查。这个函数不遵循装饰器链（即不考虑 `functools.wraps()`），因此必须使用以下行以确保 Keras 检查第一个参数与原始签名匹配。
run_call_with_unpacked_inputs.__signature__ = original_signature

return run_call_with_unpacked_inputs
    # 定义允许的数据类型元组，包括 TensorFlow 张量、布尔值、整数、模型输出、元组、列表、字典和 NumPy 数组
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)
    
    # 如果 kwargs 字典中包含键 "kwargs_call" 中的 "inputs"，发出警告并将其替换为 "input_ids"
    if "inputs" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
            FutureWarning,
        )
        output["input_ids"] = kwargs["kwargs_call"].pop("inputs")
    
    # 如果 kwargs 字典中包含键 "kwargs_call" 中的 "decoder_cached_states"，发出警告并将其替换为 "past_key_values"
    if "decoder_cached_states" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
            " `past_key_values` instead.",
            FutureWarning,
        )
        output["past_key_values"] = kwargs["kwargs_call"].pop("decoder_cached_states")
    
    # 如果 kwargs 字典中同时包含 "past" 和 "past_key_values"，根据参数名称列表作相应处理
    if "past" in kwargs["kwargs_call"] and "past_key_values" in parameter_names:
        warnings.warn(
            "The `past` argument is deprecated and will be removed in a future version, use `past_key_values`"
            " instead.",
            FutureWarning,
        )
        kwargs["past_key_values"] = kwargs["kwargs_call"].pop("past")
    elif "past_key_values" in kwargs["kwargs_call"] and "past" in parameter_names:
        kwargs["past"] = kwargs["kwargs_call"].pop("past_key_values")
    
    # 如果存在额外的关键字参数（kwargs_call），将其从 kwargs 中弹出并存储在 output 字典中的 "kwargs" 键下
    if has_kwargs:
        output["kwargs"] = kwargs.pop("kwargs_call", {})
    else:
        # 如果 kwargs_call 不为空，则引发 ValueError 异常，指示模型不支持这些关键字参数
        if len(kwargs["kwargs_call"]) > 0:
            raise ValueError(
                "The following keyword arguments are not supported by this model:"
                f" {list(kwargs['kwargs_call'].keys())}."
            )
        kwargs.pop("kwargs_call")
    
    # 遍历 kwargs 字典，检查每个键值对的值是否是允许的数据类型之一，如果是则存储在 output 字典中对应的键下
    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or tf.is_tensor(v) or v is None:
            output[k] = v
        else:
            # 如果值的类型不允许，则引发 ValueError 异常，指出具体类型和不允许的数据类型列表
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    
    # 如果 main_input 是元组或列表，则遍历其中的每个输入
    if isinstance(main_input, (tuple, list)):
        for i, input in enumerate(main_input):
            # 如果输入是 TensorFlow 符号张量，并且输入的名称在 parameter_names 中，则存储在 output 中对应的键下
            if is_tf_symbolic_tensor(input):
                # TensorFlow 张量的名称通常是 `name:id` 格式，这里只提取 `name` 部分
                tensor_name = input.name.split(":")[0]
    
                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            # 如果输入是允许的数据类型之一或为 None，则存储在 output 中对应的键下
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                # 如果输入的类型不允许，则引发 ValueError 异常，指出具体类型和不允许的数据类型列表
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for"
                    f" {parameter_names[i]}."
                )
    # 如果 main_input 是一个 Mapping 类型（如字典），则执行以下操作
    elif isinstance(main_input, Mapping):
        # 如果 main_input 中包含键 "inputs"
        if "inputs" in main_input:
            # 发出警告，说明 `inputs` 参数已废弃，并在将来的版本中会移除，建议使用 `input_ids` 替代
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids`"
                " instead.",
                FutureWarning,
            )
            # 将 main_input 中的 "inputs" 弹出并放入 output 的 "input_ids" 中
            output["input_ids"] = main_input.pop("inputs")

        # 如果 main_input 中包含键 "decoder_cached_states"
        if "decoder_cached_states" in main_input:
            # 发出警告，说明 `decoder_cached_states` 参数已废弃，并在将来的版本中会移除，建议使用 `past_key_values` 替代
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
                " `past_key_values` instead.",
                FutureWarning,
            )
            # 将 main_input 中的 "decoder_cached_states" 弹出并放入 output 的 "past_key_values" 中
            output["past_key_values"] = main_input.pop("decoder_cached_states")

        # 遍历 main_input 中的键值对
        for k, v in dict(main_input).items():
            # 如果值 v 的类型属于允许的类型 allowed_types 或者为 None
            if isinstance(v, allowed_types) or v is None:
                # 将键值对放入 output 中
                output[k] = v
            # 如果键 k 不在参数名称列表 parameter_names 中，且 "args" 不在参数名称列表中
            elif k not in parameter_names and "args" not in parameter_names:
                # 记录警告日志，说明参数 k 不属于参数列表 parameter_names 中，并将被忽略
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                # 抛出数值错误，说明类型为 type(v) 的数据不允许，只有 allowed_types 类型允许传递给参数 k
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    
    # 如果 main_input 不是 Mapping 类型，则执行以下操作
    else:
        # 如果 main_input 是 TensorFlow 的张量或者为 None
        if tf.is_tensor(main_input) or main_input is None:
            # 将 main_input 放入 output 中，键为 main_input_name
            output[main_input_name] = main_input
        else:
            # 抛出数值错误，说明类型为 type(main_input) 的数据不允许，只有 allowed_types 类型允许传递给 main_input_name
            raise ValueError(
                f"Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for {main_input_name}."
            )

    # 将未指定的参数按照签名的默认值填充到 output 中
    for name in parameter_names:
        # 如果参数名称 name 不在 output 的键列表中，且不为 "args"
        if name not in list(output.keys()) and name != "args":
            # 将参数名称 name 的默认值（来自 kwargs 或者签名中）填充到 output 中
            output[name] = kwargs.pop(name, signature[name].default)

    # 当创建 SavedModel 时，TF 会通过 LayerCall.__call__(args, **kwargs) 调用方法
    # 因此为了正确输出，需要处理此异常情况
    if "args" in output:
        # 如果 output 中的 "args" 不为 None，并且是 TensorFlow 符号张量
        if output["args"] is not None and is_tf_symbolic_tensor(output["args"]):
            # 获取张量的名称
            tensor_name = output["args"].name.split(":")[0]
            # 将 output 中的 "args" 放入 output 中，键为张量的名称
            output[tensor_name] = output["args"]
        else:
            # 在这种情况下，"args" 总是第一个参数，然后是 "input_ids"
            output["input_ids"] = output["args"]

        # 从 output 中删除 "args"
        del output["args"]

    # 如果 output 中存在 "kwargs"，从 output 中删除 "kwargs"
    if "kwargs" in output:
        del output["kwargs"]

    # 创建一个新的字典 cast_output
    cast_output = {}
    # 遍历 output 中的键值对
    for key, val in output.items():
        # 如果值 val 是 TensorFlow 的张量且数据类型为 tf.int64
        if isinstance(val, tf.Tensor) and val.dtype == tf.int64:
            # 将 val 转换为 tf.int32 类型，并放入 cast_output 中
            cast_output[key] = tf.cast(val, tf.int32)
        # 如果值 val 是 NumPy 的数组且数据类型为 np.int64
        elif isinstance(val, np.ndarray) and val.dtype == np.int64:
            # 将 val 转换为 np.int32 类型，并放入 cast_output 中
            cast_output[key] = val.astype(np.int32)
        else:
            # 否则直接将 val 放入 cast_output 中
            cast_output[key] = val

    # 将 cast_output 赋值给 output
    output = cast_output
    # 删除 cast_output
    del cast_output
    # 如果配置对象不为空，则从输出字典中提取指定键的键值对，形成布尔类型的字典
    boolean_dict = {
        k: v
        for k, v in output.items()
        if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
    }

    # 调用 booleans_processing 函数处理布尔类型的配置，更新输出字典
    output.update(
        booleans_processing(
            config=config,
            **boolean_dict,
        )
    )

    # 返回更新后的输出字典
    return output
def tf_shard_checkpoint(weights, max_shard_size="10GB"):
    """
    Splits a model state dictionary into sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `weights` in the order of its keys, ensuring that each
    sub-checkpoint does not exceed `max_shard_size`.

    Args:
        weights (`Dict[str, tf.ResourceVariable]`): The dictionary of tf.ResourceVariable objects representing weights
            of a model.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
    
    Returns:
        Tuple[Dict[str, List[tf.ResourceVariable]], Optional[Dict[str, List[int]]]]:
            A tuple containing:
                - A dictionary mapping from a checkpoint name (e.g., `"TF2_WEIGHTS_NAME"`) to a list of tf.ResourceVariable objects,
                  representing each sub-checkpoint.
                - Optionally, a dictionary mapping from each checkpoint name to a list of sizes (in bytes) of the corresponding
                  sub-checkpoints.
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)  # Convert `max_shard_size` string to integer bytes

    sharded_state_dicts = []  # Initialize list to hold sub-checkpoints
    current_block = []  # Initialize current sub-checkpoint
    current_block_size = 0  # Initialize current sub-checkpoint size
    total_size = 0  # Initialize total size accumulator

    for item in weights:  # Iterate through each weight item
        weight_size = item.numpy().size * dtype_byte_size(item.dtype)  # Calculate size of current weight in bytes

        # Check if adding current weight would exceed `max_shard_size`, if so, start a new sub-checkpoint
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)  # Append current sub-checkpoint to list
            current_block = []  # Reset current sub-checkpoint
            current_block_size = 0  # Reset current sub-checkpoint size

        current_block.append(item)  # Add current weight to current sub-checkpoint
        current_block_size += weight_size  # Update current sub-checkpoint size
        total_size += weight_size  # Update total size accumulator

    sharded_state_dicts.append(current_block)  # Append the last sub-checkpoint

    # If only one sub-checkpoint exists, return it directly
    if len(sharded_state_dicts) == 1:
        return {TF2_WEIGHTS_NAME: sharded_state_dicts[0]}, None

    # Otherwise, prepare and return a dictionary mapping each checkpoint name to its corresponding list of weights
    weight_map = {}
    shards = {}
    # 遍历分片状态字典列表，同时追踪索引号和每个状态字典
    for idx, shard in enumerate(sharded_state_dicts):
        # 根据索引号生成分片文件名，将 ".h5" 替换为格式化的编号
        shard_file = TF2_WEIGHTS_NAME.replace(".h5", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.h5")
        # 将当前分片存入分片字典，以生成的文件名作为键，分片数据作为值
        shards[shard_file] = shard
        # 遍历当前分片中的每个权重，并将权重名映射到对应的分片文件名
        for weight in shard:
            weight_name = weight.name
            weight_map[weight_name] = shard_file

    # 创建元数据字典，包含总大小信息
    metadata = {"total_size": total_size}
    # 创建索引字典，包含元数据和权重映射信息
    index = {"metadata": metadata, "weight_map": weight_map}
    # 返回分片字典和索引字典作为结果
    return shards, index
# 加载 TensorFlow 分片权重的函数，用于从分片检查点中加载模型的权重。检测缺失和意外的层，并根据它们的名称和形状从分片文件中加载 TensorFlow 权重。
def load_tf_sharded_weights(model, shard_files, ignore_mismatched_sizes=False, strict=False, _prefix=None):
    """
    This is the same as `load_tf_weights` but for a sharded checkpoint. Detect missing and unexpected layers and load
    the TF weights from the shard file accordingly to their names and shapes.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`keras.models.Model`): The model in which to load the checkpoint.
        shard_files (`str` or `os.PathLike`): A list containing the sharded checkpoint names.
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `True`):
            Whether or not to ignore the mismatch between the sizes.
        strict (`bool`, *optional*, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
        mismatched layers.
    """

    # 创建空集合来存储意外的键、保存的键和不匹配的键
    unexpected_keys = set()
    saved_keys = set()
    mismatched_keys = set()

    # 由于 TensorFlow 将其权重的类名添加到权重中，并使用索引而不是层名称加载权重，因此我们必须去掉层名称的第一个前缀。
    # 创建模型键集合和映射字典
    model_keys = set()
    model_layer_map = {}
    for i, k in enumerate(model.weights):
        layer_name = k.name
        # 如果有前缀，并且层名称以前缀开头，则去除前缀和斜杠
        if _prefix is not None and layer_name.startswith(_prefix):
            layer_name = layer_name[len(_prefix):]
            layer_name = layer_name.lstrip("/")
        # 如果层名称中包含 "model." 或只有一个部分，则保持不变；否则，只保留第二部分作为层名称
        if not ("model." in layer_name or len(layer_name.split("/")) == 1):
            layer_name = "/".join(layer_name.split("/")[1:])
        # 将处理后的层名称添加到模型键集合和映射字典中
        model_keys.add(layer_name)
        model_layer_map[layer_name] = i

    # 遍历每个分片文件，并加载权重
    for shard_file in shard_files:
        # 调用 load_tf_shard 函数加载分片文件中的权重
        saved_weight_names_set, unexpected_keys_set, mismatched_keys_set = load_tf_shard(
            model,
            model_layer_map,
            shard_file,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            _prefix=_prefix,
        )
        # 更新保存的键、意外的键和不匹配的键集合
        saved_keys.update(saved_weight_names_set)
        unexpected_keys.update(unexpected_keys_set)
        mismatched_keys.update(mismatched_keys_set)
        # 手动进行垃圾回收
        gc.collect()

    # 计算缺失的键集合
    missing_keys = model_keys - saved_keys
    # 如果 strict 为 True 并且存在缺失的键或意外的键，则抛出运行时错误
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nUnexpected key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)
    # 返回三个变量：missing_keys（缺失的键列表）、unexpected_keys（意外的键列表）、mismatched_keys（不匹配的键列表）
    return missing_keys, unexpected_keys, mismatched_keys
# 从分片的检查点文件中加载一个分片。处理缺失的键和意外的键。

def load_tf_shard(model, model_layer_map, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    """
    Loads a shard from a sharded checkpoint file. Handles the missing keys and unexpected keys.

    Args:
        model (`keras.models.Model`): Model in which the weights are loaded
        model_layer_map (`Dict`): A dictionary mapping the layer name to the index of the layer in the model.
        resolved_archive_file (`str`): Path to the checkpoint file from which the weights will be loaded
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`): Whether to ignore the mismatched keys

    Returns:
        `keras.models.Model`: Three lists, one for the layers that were found and succesfully restored (from the
        shard file), one for the mismatched layers, and another one for the unexpected layers.
    """

    # 保存已读取的权重名称的集合
    saved_weight_names_set = set()
    # 存储已加载的权重数据的字典
    saved_weights = {}
    # 存储不匹配的键的集合
    mismatched_keys = set()
    # 存储意外的键的集合
    unexpected_keys = set()

    # 读取 H5 文件
    try:
        # 使用 "r" 模式打开 H5 文件作为 sharded_checkpoint_file，使用 with 语句确保文件操作后自动关闭
        with h5py.File(resolved_archive_file, "r") as sharded_checkpoint_file:
            # 从 H5 文件中加载每个层的名称，并存储为集合 saved_h5_model_layers_name
            saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, "layer_names"))
            # 初始化空列表，用于存储权重的元组 [(权重对象, 权重值), ...]
            weight_value_tuples = []

            # 遍历每个保存的层名称
            for layer_name in saved_h5_model_layers_name:
                # 获取 H5 文件中的层对象
                h5_layer_object = sharded_checkpoint_file[layer_name]
                # 将 H5 文件中的权重转换为 NumPy 数组，并存储在 saved_weights 字典中
                saved_weights[layer_name] = np.asarray(h5_layer_object)

                # 将当前层名称添加到 saved_weight_names_set 集合中
                saved_weight_names_set.add(layer_name)

                # 如果层名称不在 model_layer_map 中，将其添加到 unexpected_keys 集合中
                if layer_name not in model_layer_map:
                    unexpected_keys.add(layer_name)
                else:
                    # 从 model_layer_map 中获取符号权重并赋值给 symbolic_weight
                    symbolic_weight = model.weights[model_layer_map[layer_name]]

                    # 获取保存的权重值
                    saved_weight_value = saved_weights[layer_name]
                    # 如果保存的权重值不为空
                    if saved_weight_value is not None:
                        # 检查当前权重的形状与 H5 文件中的形状是否不同
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            # 如果形状不兼容，尝试重新调整保存的权重值的形状以匹配当前权重
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                # 如果 ignore_mismatched_sizes 为 True，则将不兼容的形状添加到 mismatched_keys 中
                                if ignore_mismatched_sizes:
                                    mismatched_keys.add(
                                        (layer_name, saved_weight_value.shape, K.int_shape(symbolic_weight))
                                    )
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value

                    # 创建权重元组 (symbolic_weight, array)，并添加到 weight_value_tuples 列表中
                    weight_value_tuples.append((symbolic_weight, array))

        # 使用 K.batch_set_value 批量设置模型权重
        K.batch_set_value(weight_value_tuples)

        # 返回结果：保存的权重名称集合、未预期的键集合和不匹配的键集合
        return saved_weight_names_set, unexpected_keys, mismatched_keys
    # 捕获任何异常，并尝试处理
    except Exception as e:
        # 尝试打开已解析的归档文件
        try:
            # 使用上下文管理器打开文件
            with open(resolved_archive_file) as f:
                # 如果文件内容以 "version" 开头
                if f.read().startswith("version"):
                    # 抛出 OSError，提示缺少 git-lfs
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    # 否则，抛出 ValueError，提示无法找到必要的预训练模型文件
                    raise ValueError(
                        f"Unable to locate the file {resolved_archive_file} which is necessary to load this pretrained"
                        " model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            # 捕获 UnicodeDecodeError 或 ValueError 异常，抛出 OSError
            raise OSError(
                f"Unable to load weights from TF checkpoint file for '{resolved_archive_file}' "
                f"at '{resolved_archive_file}'. "
                "If you tried to load a TF model from a sharded checkpoint, you should try converting the model "
                "by loading it in pytorch and saving it localy. A convertion script should be realeased soon."
            )
# 根据文件后缀判断使用哪种函数加载 TF 权重：如果是 ".safetensors" 后缀，则使用安全张量的加载函数，否则使用 H5 文件的加载函数
if resolved_archive_file.endswith(".safetensors"):
    load_function = load_tf_weights_from_safetensors
else:
    load_function = load_tf_weights_from_h5

# 调用相应的加载函数，加载模型的权重并返回结果
return load_function(
    model, resolved_archive_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=_prefix
)



def load_tf_weights_from_h5(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    # 初始化一个空列表来存放形状不匹配的层
    mismatched_layers = []

    # 从 H5 文件中读取权重值，并批量设置到模型中
    K.batch_set_value(weight_value_tuples)

    # 计算缺失的和意外的层
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set))
    unexpected_layers.extend(list(saved_weight_names_set - symbolic_weights_names))

    # 返回缺失的层、意外的层和形状不匹配的层
    return missing_layers, unexpected_layers, mismatched_layers



def load_tf_weights_from_safetensors(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    # 从安全张量文件中读取权重
    # 使用安全的方式打开解析后的存档文件，支持 TensorFlow 框架
    with safe_open(resolved_archive_file, framework="tf") as safetensors_archive:
        # 初始化一个空列表，用于存储不匹配的层信息
        mismatched_layers = []
        
        # 获取模型所有权重的名称列表（去除模型名称和前缀）
        weight_names = [strip_model_name_and_prefix(w.name, _prefix=_prefix) for w in model.weights]
        
        # 获取加载的权重文件中所有的键（即权重名称）
        loaded_weight_names = list(safetensors_archive.keys())
        
        # 找出在高级层列表中存在但在加载的权重中不存在的层
        missing_layers = list(set(weight_names) - set(loaded_weight_names))
        
        # 找出在加载的权重中存在但在高级层列表中不存在的层
        unexpected_layers = list(set(loaded_weight_names) - set(weight_names))
        
        # 遍历模型的每一个权重
        for weight in model.weights:
            # 获取去除模型名称和前缀后的权重名称
            weight_name = strip_model_name_and_prefix(weight.name, _prefix=_prefix)
            
            # 如果该权重在加载的权重名称列表中
            if weight_name in loaded_weight_names:
                # 从安全存档中获取该权重的值
                weight_value = safetensors_archive.get_tensor(weight_name)
                
                # 检查当前权重和从H5文件中读取的权重形状是否不同
                if K.int_shape(weight) != weight_value.shape:
                    # 如果形状不同，尝试将从文件中读取的权重值重塑为当前权重的形状
                    try:
                        weight_value = tf.reshape(weight_value, K.int_shape(weight))
                    except (ValueError, tf.errors.InvalidArgumentError) as e:
                        # 如果无法重塑且不忽略形状不匹配，则抛出异常
                        if ignore_mismatched_sizes:
                            # 如果忽略形状不匹配，则将当前权重和文件中权重的不匹配信息添加到列表中
                            mismatched_layers.append((weight_name, weight_value.shape, K.int_shape(weight)))
                            continue
                        else:
                            raise e
                
                # 将重新形状后的权重值赋值给当前权重
                K.set_value(weight, weight_value)  # weight.assign() might break if weight is a DTensor
    
    # 返回缺失的层列表、意外的层列表和不匹配的层列表
    return missing_layers, unexpected_layers, mismatched_layers
def init_copy_embeddings(old_embeddings, new_num_tokens):
    r"""
    This function aims to reduce the embeddings in case new_num_tokens < old_num_tokens or to pad with -1 in case
    new_num_tokens > old_num_tokens. A mask is also computed in order to know which weight in the embeddings should be
    kept or not. Example:

        - if new_num_tokens=5 and old_num_tokens=4 and old_embeddings=[w1,w2,w3,w4]

            -  mask=[True,True,True,True,False] and current_weights=[w1,w2,w3,w4,-1]
        - if new_num_tokens=4 and old_num_tokens=5 and old_embeddings=[w1,w2,w3,w4,w5]

            - mask=[True,True,True,True] and current_weights=[w1,w2,w3,w4]
    """
    # Get the number of tokens and embedding dimension from the old embeddings
    old_num_tokens, old_embedding_dim = shape_list(old_embeddings)
    
    # Calculate the difference in size between old and new embeddings
    size_diff = new_num_tokens - old_num_tokens

    # initialize new embeddings
    # Copy token embeddings from the previous ones
    if tf.math.greater(size_diff, 0):
        # if the new size is greater than the old one, we extend the current embeddings with a padding until getting new size
        # and we create a mask to properly identify the padded values and be replaced by the values of the newly created
        # embeddings
        
        # Pad the old embeddings with -1 to extend to the new size
        current_weights = tf.pad(
            old_embeddings.value(), tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=-1
        )
        
        # Determine how many tokens to copy and create a mask to identify them
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        mask = tf.fill(tf.convert_to_tensor([num_tokens_to_copy, 1]), True)
        
        # Pad the mask to match the extended embeddings size
        mask = tf.pad(mask, tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=False)
    else:
        # if the new size if lower than the old one, we take the current embeddings until the new size
        
        # Slice the old embeddings to match the new size
        current_weights = tf.slice(
            old_embeddings.value(),
            tf.convert_to_tensor([0, 0]),
            tf.convert_to_tensor([new_num_tokens, old_embedding_dim]),
        )
        
        # Create a mask for the entire new size
        mask = tf.fill(tf.convert_to_tensor([new_num_tokens, 1]), True)

    # Return the mask and the current weights
    return mask, current_weights
    """
    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """

    # 配置类，用作该模型架构的配置类，应该是 PretrainedConfig 的子类
    config_class = None

    # 基础模型前缀，表示在相同架构的派生类中与基础模型相关联的属性字符串
    base_model_prefix = ""

    # 主要输入名称，模型的主要输入名称，通常为 `input_ids`（用于 NLP 模型）、`pixel_values`（用于视觉模型）和 `input_values`（用于语音模型）
    main_input_name = "input_ids"

    # 自动分类，未指定
    _auto_class = None

    # 使用虚拟损失，未指定
    _using_dummy_loss = None

    # 标签到输出映射，未指定
    _label_to_output_map = None

    # 在加载模型权重时要忽略的张量名称的正则表达式列表，避免不必要的警告
    _keys_to_ignore_on_load_missing = None

    # 在加载模型权重时要忽略的权重中张量名称的正则表达式列表，避免不必要的警告
    _keys_to_ignore_on_load_unexpected = None

    # 是否需要加载权重前缀，默认为 False
    _requires_load_weight_prefix = False

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        dummies = {}
        for key, spec in self.input_signature.items():
            # 2 是最正确的任意大小。我不会回答这个问题
            dummy_shape = [dim if dim is not None else 2 for dim in spec.shape]
            if spec.shape[0] is None:
                # 但是，为了节省内存，让批量大小为 1
                dummy_shape[0] = 1
            dummies[key] = tf.ones(shape=dummy_shape, dtype=spec.dtype)
            if key == "token_type_ids":
                # 一些模型具有 token_type_ids，但 vocab_size 为 1
                dummies[key] = tf.zeros_like(dummies[key])
        if self.config.add_cross_attention and "encoder_hidden_states" in inspect.signature(self.call).parameters:
            if "encoder_hidden_states" not in dummies:
                if self.main_input_name == "input_ids":
                    dummies["encoder_hidden_states"] = tf.ones(
                        shape=(1, 2, self.config.hidden_size), dtype=tf.float32, name="encoder_hidden_states"
                    )
                else:
                    raise NotImplementedError(
                        "Model has cross-attention but we couldn't infer the shape for the encoder hidden states. Please manually override dummy_inputs!"
                    )
        return dummies

    def build_in_name_scope(self):
        with tf.name_scope(self.name):
            self.build(input_shape=None)

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a TensorFlow model.
        """
        return "tf"
    # 定义一个方法 `build`，用于构建模型，接受一个可选的输入形状参数 `input_shape`
    def build(self, input_shape=None):
        pass  # 这里只是为了确保不调用父类的 `build()`

    # 初始化方法 `__init__`，接受一个配置参数 `config` 和可变数量的位置参数 `inputs` 和关键字参数 `kwargs`
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*inputs, **kwargs)
        # 如果 `config` 不是 `PretrainedConfig` 类的实例，则抛出异常
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 将 `config` 和预训练权重的原始来源（如果在模型中给出）保存在实例中
        self.config = config
        self.name_or_path = config.name_or_path
        # 如果模型可以生成文本，则根据 `config` 创建 `GenerationConfig` 实例并保存在 `generation_config` 中，否则设为 `None`
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        # 设置保存规范为输入签名 `input_signature` 的保存规范
        self._set_save_spec(self.input_signature)

    # 获取模型配置的方法，返回配置的字典表示
    def get_config(self):
        return self.config.to_dict()

    # 使用 `convert_batch_encoding` 转换参数，然后调用父类的 `fit` 方法
    @functools.wraps(keras.Model.fit)
    def fit(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().fit(*args, **kwargs)

    # 使用 `convert_batch_encoding` 转换参数，然后调用父类的 `train_on_batch` 方法
    @functools.wraps(keras.Model.train_on_batch)
    def train_on_batch(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().train_on_batch(*args, **kwargs)

    # 使用 `convert_batch_encoding` 转换参数，然后调用父类的 `test_on_batch` 方法
    @functools.wraps(keras.Model.test_on_batch)
    def test_on_batch(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().test_on_batch(*args, **kwargs)

    # 使用 `convert_batch_encoding` 转换参数，然后调用父类的 `predict_on_batch` 方法
    @functools.wraps(keras.Model.predict_on_batch)
    def predict_on_batch(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().predict_on_batch(*args, **kwargs)

    # 使用 `convert_batch_encoding` 转换参数，然后调用父类的 `predict` 方法
    @functools.wraps(keras.Model.predict)
    def predict(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().predict(*args, **kwargs)

    # 使用 `convert_batch_encoding` 转换参数，然后调用父类的 `evaluate` 方法
    @functools.wraps(keras.Model.evaluate)
    def evaluate(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().evaluate(*args, **kwargs)

    # 类方法 `from_config`，接受 `config` 和其他关键字参数 `kwargs`
    @classmethod
    def from_config(cls, config, **kwargs):
        # 如果 `config` 是 `PretrainedConfig` 类的实例，则调用 `_from_config` 方法
        if isinstance(config, PretrainedConfig):
            return cls._from_config(config, **kwargs)
        # 否则，根据 `config` 字典创建 `config_class` 实例，并调用 `_from_config` 方法
        return cls._from_config(cls.config_class.from_dict(config, **kwargs))

    # 类方法 `_from_config`，接受 `config` 和其他关键字参数 `kwargs`
    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        所有模型初始化时应置于其下的上下文管理器都在这里。
        """
        # 使用 `config` 和其他关键字参数初始化类 `cls` 的实例
        return cls(config, **kwargs)
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
        # 如果传入的头部掩码不为 None，则调用 _convert_head_mask_to_5d 方法将其转换为 5 维张量
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            # 如果头部掩码为 None，则创建一个列表，包含 num_hidden_layers 个 None 元素
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        # 如果头部掩码的维度为 1，将其扩展为 [1 x 1 x num_heads x 1 x 1] 的形式，并复制为 num_hidden_layers 个
        if head_mask.shape.rank == 1:
            head_mask = head_mask[None, None, :, None, None]
            head_mask = tf.repeat(head_mask, repeats=num_hidden_layers, axis=0)
        # 如果头部掩码的维度为 2，将其扩展为 [num_hidden_layers x 1 x num_heads x 1 x 1] 的形式
        elif head_mask.shape.rank == 2:
            head_mask = head_mask[:, None, :, None, None]
        # 断言头部掩码的维度必须为 5，否则抛出异常
        assert head_mask.shape.rank == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        # 将头部掩码转换为 float32 类型，以支持 float16 兼容性
        head_mask = tf.cast(head_mask, tf.float32)
        return head_mask

    @tf.function
    def serving(self, inputs):
        """
        Args:
        Method used for serving the model. Does not have a specific signature, but will be specialized as concrete
        functions when saving with `save_pretrained`.
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        # 调用模型的 call 方法进行推理，获取输出
        output = self.call(inputs)

        # 返回推理输出的 serving_output 结果
        return self.serving_output(output)

    @property
    # 定义一个方法，返回一个字典，将模型输入的名称映射到 tf.TensorSpec 对象，用于描述模型输入的预期形状和数据类型。
    def input_signature(self) -> Dict[str, tf.TensorSpec]:
        """
        This property should return a dict mapping input names to tf.TensorSpec objects, representing the expected
        shape and dtype for model inputs. It is used for both serving and for generating dummy inputs.
        """
        # 获取调用方法 self.call 的参数列表
        model_inputs = list(inspect.signature(self.call).parameters)
        # 初始化一个空字典用于存储输入签名
        sig = {}
        
        # 检查是否存在 "input_ids" 作为模型输入的一部分
        if "input_ids" in model_inputs:
            # 如果模型类名以 "ForMultipleChoice" 结尾，则文本维度为 3
            if self.__class__.__name__.endswith("ForMultipleChoice"):
                text_dims = 3
            else:
                text_dims = 2
            # 遍历预定义的输入名称列表
            for input_name in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "decoder_input_ids",
                "decoder_attention_mask",
            ):
                # 如果当前遍历的输入名称存在于模型输入中
                if input_name in model_inputs:
                    # 将输入名称作为键，创建对应的 tf.TensorSpec 对象，指定形状和数据类型
                    sig[input_name] = tf.TensorSpec([None] * text_dims, tf.int32, name=input_name)
        
        # 检查是否存在 "pixel_values" 作为模型输入的一部分
        if "pixel_values" in model_inputs:
            # 初始化像素值的形状，None 表示任意长度或尺寸
            pixel_values_shape = [None, None, None, None]
            # 根据配置获取视觉输入的配置信息
            if hasattr(self.config, "vision_config"):
                vision_config = self.config.vision_config
            else:
                vision_config = self.config
            # 如果配置中包含 num_channels 属性，则将其设置为像素值形状的第二维度
            if hasattr(vision_config, "num_channels"):
                pixel_values_shape[1] = vision_config.num_channels
            else:
                # 如果无法从配置中推断出通道数，则抛出未实现错误
                raise NotImplementedError(
                    "Could not infer number of channels from config, please override input_signature to specify input shapes."
                )
            # 根据配置中的图像大小信息设置像素值的高度和宽度
            if hasattr(vision_config, "image_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.image_size
            elif hasattr(vision_config, "input_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.input_size
            else:
                # 如果无法推断输入图像的形状，则抛出未实现错误
                raise NotImplementedError(
                    "Could not infer input image shape from config, please override input_signature to specify input shapes."
                )
            # 将 "pixel_values" 添加到输入签名字典中，创建对应的 tf.TensorSpec 对象
            sig["pixel_values"] = tf.TensorSpec(pixel_values_shape, tf.float32, name="pixel_values")
        
        # 如果模型需要 "input_features" 作为输入，则抛出未实现错误，要求手动定义输入签名
        if "input_features" in model_inputs:
            raise NotImplementedError("Audio models need a manually defined input_signature")
        
        # 返回构建好的输入签名字典
        return sig
    def serving_output(self, output):
        """
        Prepare the output of the saved model. Can be overridden if specific serving modifications are required.
        """
        # 检查输出是否为ModelOutput类型，如果不是，则直接返回输出
        if not isinstance(output, ModelOutput):
            return output
        # 遍历输出的键
        for key in output:
            # 如果键以"hidden_states"结尾且配置中未设置输出隐藏状态，则将对应值设为None
            if key.endswith("hidden_states") and not getattr(self.config, "output_hidden_states", False):
                output[key] = None
            # 如果键以"attentions"结尾且配置中未设置输出注意力权重，则将对应值设为None
            elif key.endswith("attentions") and not getattr(self.config, "output_attentions", False):
                output[key] = None
            # 如果键为"past_key_values"且配置中未设置使用缓存，则将对应值设为None
            elif key == "past_key_values" and not getattr(self.config, "use_cache", False):
                output[key] = None
            # 如果键为"cross_attentions"且配置中未同时设置输出注意力权重和使用交叉注意力，则将对应值设为None
            elif key == "cross_attentions" and not (
                getattr(self.config, "output_attentions", False) and getattr(self.config, "add_cross_attention", False)
            ):
                output[key] = None
            # 如果值为tuple或list类型，尝试将其转换为TensorFlow张量
            if isinstance(output[key], (tuple, list)):
                try:
                    output[key] = tf.convert_to_tensor(output[key])
                except (ValueError, tf.errors.InvalidArgumentError):
                    pass  # 可能由于层的维度不同而无法转换
        return output

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # 检测是否已覆盖了`prepare_inputs_for_generation`方法，这是生成序列的要求之一
        # 或者模型可能有自定义的`generate`函数
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True

    def get_input_embeddings(self) -> keras.layers.Layer:
        """
        Returns the model's input embeddings layer.

        Returns:
            `tf.Variable`: The embeddings layer mapping vocabulary to hidden states.
        """
        # 获取模型的输入嵌入层
        main_layer = getattr(self, self.base_model_prefix, self)

        # 如果main_layer不是self，即存在基础模型前缀，则返回其输入嵌入层
        if main_layer is not self:
            return main_layer.get_input_embeddings()
        else:
            # 否则抛出未实现错误，要求子类实现该方法
            raise NotImplementedError
    # 定义一个方法用于保存模型检查点，将模型参数保存到指定的目录中
    def _save_checkpoint(self, checkpoint_dir, epoch):
        # 如果指定的检查点目录不存在，则创建该目录
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        
        # 定义权重文件的保存路径为指定目录下的"weights.h5"
        weights_path = os.path.join(checkpoint_dir, "weights.h5")
        # 调用模型的保存权重方法，将模型的权重保存到weights_path中
        self.save_weights(weights_path)
        
        # 准备额外的数据，包括当前的训练轮数(epoch)和优化器的状态
        extra_data = {"epoch": epoch, "optimizer_state": self.optimizer.get_weights()}
        # 定义额外数据文件的保存路径为指定目录下的"extra_data.pickle"
        extra_data_path = os.path.join(checkpoint_dir, "extra_data.pickle")
        
        # 使用 pickle 序列化额外数据，并保存到extra_data_path中
        with open(extra_data_path, "wb") as f:
            pickle.dump(extra_data, f)

    # 定义一个方法用于准备 TensorFlow 数据集
    def prepare_tf_dataset(
        self,
        dataset: "datasets.Dataset",  # noqa:F821
        batch_size: int = 8,
        shuffle: bool = True,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        collate_fn: Optional[Callable] = None,
        collate_fn_args: Optional[Dict[str, Any]] = None,
        drop_remainder: Optional[bool] = None,
        prefetch: bool = True,
    ):
    
    # 定义一个方法用于编译模型，设置优化器、损失函数、评估指标等
    def compile(
        self,
        optimizer="rmsprop",
        loss="auto_with_warning",
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs,
    ):
    ):
        """
        This is a thin wrapper that sets the model's loss output head as the loss if the user does not specify a loss
        function themselves.
        """
        # 如果用户没有指定损失函数，则将模型的损失输出头部设置为损失函数
        if loss in ("auto_with_warning", "passthrough"):  # "passthrough" for workflow backward compatibility
            # 如果在compile()中没有指定损失函数，将使用模型的内部损失计算作为损失
            logger.info(
                "No loss specified in compile() - the model's internal loss computation will be used as the "
                "loss. Don't panic - this is a common way to train TensorFlow models in Transformers! "
                "To disable this behaviour please pass a loss argument, or explicitly pass "
                "`loss=None` if you do not want your model to compute a loss. You can also specify `loss='auto'` to "
                "get the internal loss without printing this info string."
            )
            # 设置损失为"auto"，表示使用默认的虚拟损失函数
            loss = "auto"
        if loss == "auto":
            # 如果损失为"auto"，则将损失设置为虚拟损失函数dummy_loss，并标记为使用了虚拟损失函数
            loss = dummy_loss
            self._using_dummy_loss = True
        else:
            # 否则，标记为没有使用虚拟损失函数
            self._using_dummy_loss = False
        # 获取父类方法compile()的参数列表
        parent_args = list(inspect.signature(keras.Model.compile).parameters.keys())
        # 检查是否支持参数"steps_per_execution"
        if "steps_per_execution" in parent_args:
            # 如果支持，调用父类方法compile()，使用参数"steps_per_execution"
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
            # 否则，调用父类方法compile()，使用参数"experimental_steps_per_execution"（兼容旧版本命名）
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

    def compute_loss(self, *args, **kwargs):
        # 检查是否有方法"compute_loss"存在于keras.Model中
        if hasattr(keras.Model, "compute_loss"):
            # 如果是True（TF 2.8或更高版本），调用父类方法compute_loss()
            return super().compute_loss(*args, **kwargs)
        else:
            # 否则，发出警告，指出旧版本的compute_loss方法已弃用，建议使用hf_compute_loss()方法
            warnings.warn(
                "The old compute_loss method is deprecated as it conflicts with the Keras compute_loss "
                "method added in TF 2.8. If you want the original HF compute_loss, please call "
                "hf_compute_loss() instead. From TF versions >= 2.8, or Transformers versions >= 5, "
                "calling compute_loss() will get the Keras method instead.",
                FutureWarning,
            )
            # 返回使用hf_compute_loss()方法计算的损失值
            return self.hf_compute_loss(*args, **kwargs)
    # 获取标签到输出名称的映射关系函数
    def get_label_to_output_name_mapping(self):
        # 使用 Python inspect 模块获取当前函数调用的参数名列表
        arg_names = list(inspect.signature(self.call).parameters)
        # 如果已经存在标签到输出映射关系，直接返回
        if self._label_to_output_map is not None:
            return self._label_to_output_map
        # 根据不同的参数名情况，返回对应的映射关系字典
        elif "start_positions" in arg_names:
            return {"start_positions": "start_logits", "end_positions": "end_logits"}
        elif "sentence_order_label" in arg_names:
            return {"labels": "prediction_logits", "sentence_order_label": "sop_logits"}
        elif "next_sentence_label" in arg_names:
            return {"labels": "prediction_logits", "next_sentence_label": "seq_relationship_logits"}
        elif "mc_labels" in arg_names:
            return {"labels": "logits", "mc_labels": "mc_logits"}
        else:
            # 默认情况下，返回空的映射关系字典
            return {}

    # 创建模型卡函数，用于生成模型卡片的描述
    def create_model_card(
        self,
        output_dir,
        model_name: str,
        language: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Optional[str] = None,
        dataset_tags: Optional[Union[str, List[str]]] = None,
        dataset: Optional[Union[str, List[str]]] = None,
        dataset_args: Optional[Union[str, List[str]]] = None,
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
            # Avoids a circular import by doing this when necessary.
            from .modelcard import TrainingSummary  # tests_ignore

            # 使用 TrainingSummary 类的静态方法 from_keras 创建训练摘要
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
            # 将训练摘要转换为模型卡
            model_card = training_summary.to_model_card()
            # 打开指定路径下的 README.md 文件，以写入模型卡内容
            with open(os.path.join(output_dir, "README.md"), "w") as f:
                f.write(model_card)
    # 设置模型的输入嵌入
    def set_input_embeddings(self, value):
        """
        Set model's input embeddings

        Args:
            value (`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        # 获取主要的模型层
        main_layer = getattr(self, self.base_model_prefix)

        # 如果主模型层为空，抛出未实现错误
        if main_layer is None:
            raise NotImplementedError("The model does not implements the base_model_prefix attribute.")

        try:
            # 尝试设置输入嵌入到主模型层
            main_layer.set_input_embeddings(value)
        except AttributeError:
            # 如果出现属性错误，记录日志并构建模型
            logger.info("Building the model")
            self.build_in_name_scope()
            # 再次尝试设置输入嵌入到主模型层
            main_layer.set_input_embeddings(value)

    # 获取模型的输出嵌入
    def get_output_embeddings(self) -> Union[None, keras.layers.Layer]:
        """
        Returns the model's output embeddings

        Returns:
            `tf.Variable`: The new weights mapping vocabulary to hidden states.
        """
        # 如果模型有语言模型头部
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()

            try:
                # 尝试获取输出嵌入层
                return lm_head.get_output_embeddings()
            except AttributeError:
                # 如果出现属性错误，记录日志并构建模型
                logger.info("Building the model")
                self.build_in_name_scope()

                # 再次尝试获取输出嵌入层
                return lm_head().get_output_embeddings()

        # 如果没有语言模型头部，返回None（适用于没有输出嵌入的模型）
        return None  # Overwrite for models with output embeddings

    # 设置模型的输出嵌入
    def set_output_embeddings(self, value):
        """
        Set model's output embeddings

        Args:
            value (`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        # 如果模型有语言模型头部
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                # 尝试设置输出嵌入到语言模型头部
                lm_head.set_output_embeddings(value)
            except AttributeError:
                # 如果出现属性错误，记录日志并构建模型，然后再次尝试设置输出嵌入
                logger.info("Building the model")
                self.build_in_name_scope()
                lm_head.set_output_embeddings(value)

    # 获取带有偏置的输出层，用于处理模型带有与嵌入权重绑定的偏置属性
    def get_output_layer_with_bias(self) -> Union[None, keras.layers.Layer]:
        """
        Get the layer that handles a bias attribute in case the model has an LM head with weights tied to the
        embeddings

        Return:
            `keras.layers.Layer`: The layer that handles the bias, None if not an LM model.
        """
        warnings.warn(
            "The method get_output_layer_with_bias is deprecated. Please use `get_lm_head` instead.", FutureWarning
        )
        # 返回语言模型头部（如果有）
        return self.get_lm_head()

    # 获取模型名称到父层的前缀偏置名称
    def get_prefix_bias_name(self) -> Union[None, str]:
        """
        Get the concatenated _prefix name of the bias from the model name to the parent layer

        Return:
            `str`: The _prefix name of the bias.
        """
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回None，因为这个方法已经被废弃
        return None
    def get_bias(self) -> Union[None, Dict[str, tf.Variable]]:
        """
        获取 LM 头部的偏置字典。键表示偏置属性的名称。

        Return:
            `tf.Variable`: 表示偏置的权重，如果不是 LM 模型则返回 None。
        """
        if self.get_lm_head() is not None:
            # 获取 LM 头部的引用
            lm_head = self.get_lm_head()
            try:
                # 尝试获取 LM 头部的偏置
                return lm_head.get_bias()
            except AttributeError:
                # 如果 LM 头部没有 get_bias 方法，则建立名称作用域并尝试再次获取偏置
                self.build_in_name_scope()
                return lm_head.get_bias()
        return None

    def set_bias(self, value):
        """
        设置 LM 头部所有的偏置。

        Args:
            value (`Dict[tf.Variable]`):
                LM 头部新的偏置字典。
        """
        if self.get_lm_head() is not None:
            # 获取 LM 头部的引用
            lm_head = self.get_lm_head()
            try:
                # 尝试设置 LM 头部的偏置
                lm_head.set_bias(value)
            except AttributeError:
                # 如果 LM 头部没有 set_bias 方法，则建立名称作用域并尝试再次设置偏置
                self.build_in_name_scope()
                lm_head.set_bias(value)

    def get_lm_head(self) -> keras.layers.Layer:
        """
        LM 头部层。所有包含 LM 头部的模型必须重写此方法。

        Return:
            `keras.layers.Layer`: 如果模型有 LM 头部则返回该层，否则返回 None。
        """
        return None

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None
    ) -> Union[keras.layers.Embedding, tf.Variable]:
        """
        调整模型输入标记嵌入矩阵的大小，如果 `new_num_tokens != config.vocab_size`。

        在之后处理权重嵌入时要注意是否模型类有 `tie_weights()` 方法。

        Arguments:
            new_num_tokens (`int`, *optional*):
                嵌入矩阵中的新标记数量。增加大小将在末尾添加新初始化的向量，减小大小将从末尾删除向量。如果未提供或为 `None`，则仅返回输入标记的指针而不执行任何操作。

        Return:
            `tf.Variable` 或 `keras.layers.Embedding`: 模型输入标记的指针。
        """
        # TODO (joao): 因嵌入重构标记为替换标记（由 `_v2_resized_token_embeddings`）

        # 如果模型具有 keras 嵌入层，则运行新代码路径
        if isinstance(self.get_input_embeddings(), keras.layers.Embedding):
            return self._v2_resized_token_embeddings(new_num_tokens)

        # 如果 new_num_tokens 为 None 或等于 config.vocab_size，则返回当前输入标记的权重
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self._get_word_embedding_weight(self.get_input_embeddings())

        # 否则调整标记嵌入大小并返回模型嵌入
        model_embeds = self._resize_token_embeddings(new_num_tokens)

        # 更新基础模型和当前模型配置的词汇大小
        self.config.vocab_size = new_num_tokens

        return model_embeds
    # 调整模型的输入标记嵌入矩阵大小，如果 `new_num_tokens != config.vocab_size`。
    # 如果 `new_num_tokens` 为 `None` 或者与当前配置中的词汇表大小相同，则返回模型的输入标记嵌入指针。
    def _v2_resized_token_embeddings(self, new_num_tokens: Optional[int] = None) -> keras.layers.Embedding:
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self.get_input_embeddings()

        # 调整标记嵌入矩阵的大小，并获取调整后的模型嵌入层
        model_embeds = self._v2_resize_token_embeddings(new_num_tokens)

        # 更新基础模型和当前模型配置中的词汇表大小
        self.config.vocab_size = new_num_tokens

        # 返回调整后的模型嵌入层
        return model_embeds

    # 获取词嵌入权重的函数
    def _get_word_embedding_weight(model, embedding_layer):
        # TODO (joao): 根据嵌入重构的需求标记为删除

        # 如果 `embedding_layer` 是 `tf.Tensor` 类型，则返回它本身
        if isinstance(embedding_layer, tf.Tensor):
            return embedding_layer

        # 否则，尝试从层的属性中获取权重
        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        # 尝试从层的 `decoder` 属性获取权重
        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        # 如果属性不存在可能是因为模型尚未构建，因此尝试在构建模型后再次获取
        model.build_in_name_scope()

        # 再次尝试从层的 `weight` 属性获取权重
        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        # 再次尝试从层的 `decoder` 属性获取权重
        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        # 如果无法获取权重，则返回 `None`
        return None
    def _resize_token_embeddings(self, new_num_tokens):
        # TODO (joao): flagged for replacement (by `_v2_resize_token_embeddings`) due to embeddings refactor
        # 获取当前模型的词嵌入权重
        old_embeddings = self._get_word_embedding_weight(self.get_input_embeddings())
        # 调用私有方法，根据新的词汇量大小调整词嵌入权重
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)

        # 如果词嵌入没有被绑定，确保语言模型头部偏置也被调整大小
        if self.get_bias() is not None:
            # 获取当前的语言模型头部偏置
            old_lm_head_bias = self.get_bias()
            # 根据新的词汇量大小调整语言模型头部偏置
            new_lm_head_bias = self._get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
            # 设置调整后的语言模型头部偏置
            self.set_bias(new_lm_head_bias)

        # 如果词嵌入没有被绑定，确保语言模型头部解码器也被调整大小
        if self.get_output_embeddings() is not None:
            # 获取当前语言模型头部解码器的词嵌入权重
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            # 根据新的词汇量大小调整语言模型头部解码器
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)
            # 设置调整后的语言模型头部解码器
            self.set_output_embeddings(new_lm_head_decoder)

        # 设置调整后的输入词嵌入
        self.set_input_embeddings(new_embeddings)

        # 返回调整后的输入词嵌入
        return self.get_input_embeddings()

    def _v2_resize_token_embeddings(self, new_num_tokens):
        # 获取当前模型的输入词嵌入权重
        old_embeddings = self.get_input_embeddings()
        # 根据新的词汇量大小调整输入词嵌入权重
        new_embeddings = self._v2_get_resized_embeddings(old_embeddings, new_num_tokens)
        # 设置调整后的输入词嵌入权重
        self.set_input_embeddings(new_embeddings)

        # 如果词嵌入没有被绑定，确保语言模型头部偏置也被调整大小
        if self.get_bias() is not None:
            # 获取当前的语言模型头部偏置
            old_lm_head_bias = self.get_bias()
            # 根据新的词汇量大小调整语言模型头部偏置
            new_lm_head_bias = self._v2_get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
            # 设置调整后的语言模型头部偏置
            self.set_bias(new_lm_head_bias)

        # 如果词嵌入没有被绑定，确保语言模型头部解码器也被调整大小
        tied_weights = self.get_input_embeddings() == self.get_output_embeddings()
        if self.get_output_embeddings() is not None and not tied_weights:
            # 获取当前语言模型头部解码器的词嵌入权重
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            # TODO (joao): this one probably needs a v2 version with other models
            # 根据新的词汇量大小调整语言模型头部解码器
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)
            # 设置调整后的语言模型头部解码器
            self.set_output_embeddings(new_lm_head_decoder)

        # 返回调整后的输入词嵌入权重
        return self.get_input_embeddings()
   `
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
        # Initialize an empty dictionary to store new biases
        new_lm_head_bias = {}

        # Iterate through each attribute and its corresponding weight in old_lm_head_bias
        for attr, weight in old_lm_head_bias.items():
            # Determine the shape of the weight tensor
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            # Calculate the difference in size between old and new number of tokens
            size_diff = new_num_tokens - old_num_tokens
            # Define the final shape of the bias tensor after resizing
            final_shape = [new_num_tokens] if first_dim is None else [first_dim, new_num_tokens]

            # Initialize or slice the current bias based on size difference
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

            # Create a new bias variable and initialize it
            new_bias = self.add_weight(
                shape=final_shape,
                initializer="zeros",
                trainable=True,
                name=weight.name.split(":")[0],
            )
            init_bias = tf.where(bias_mask, current_bias, new_bias.value())

            # Assign the initialized bias to the new_bias variable and store it in new_lm_head_bias
            new_bias.assign(init_bias)
            new_lm_head_bias[attr] = new_bias

        # Return the dictionary containing resized biases
        return new_lm_head_bias
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
        # Initialize an empty dictionary to store resized biases
        new_lm_head_bias = {}

        # Iterate over each attribute and its corresponding weight in the old_lm_head_bias dictionary
        for attr, weight in old_lm_head_bias.items():
            # Determine the shape of the weight tensor and calculate the size difference
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            size_diff = new_num_tokens - old_num_tokens

            # Copy the old bias values to the new bias tensor
            if old_num_tokens > new_num_tokens:
                # Trim the weight tensor if reducing size
                new_bias = weight.value()[..., :new_num_tokens]
            else:
                # Pad the weight tensor with zeros if increasing size
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                new_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape))

            # Store the resized bias tensor in the new_lm_head_bias dictionary
            new_lm_head_bias[attr] = new_bias

        # Return the dictionary containing resized bias tensors
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
        # 将新的 lm head 解码器初始化为旧的 lm head 解码器
        new_lm_head_decoder = old_lm_head_decoder

        # 检查输入嵌入矩阵和旧 lm head 解码器是否相同
        is_input_output_equals = tf.reduce_any(
            self._get_word_embedding_weight(self.get_input_embeddings()) == old_lm_head_decoder
        )

        # 如果旧 lm head 解码器不为 None 并且输入输出不相同
        if old_lm_head_decoder is not None and not is_input_output_equals:
            # 获取旧 lm head 解码器的维度
            old_embedding_dim = shape_list(old_lm_head_decoder)[1]

            # 初始化复制嵌入和解码器掩码
            decoder_mask, current_decoder = init_copy_embeddings(old_lm_head_decoder, new_num_tokens)

            # 创建新的 lm head 解码器，使用零初始化，可训练
            new_lm_head_decoder = self.add_weight(
                shape=(new_num_tokens, old_embedding_dim),
                initializer="zeros",
                trainable=True,
                name=old_lm_head_decoder.name.split(":")[0],
            )

            # 根据解码器掩码选择初始化策略
            init_decoder = tf.where(decoder_mask, current_decoder, new_lm_head_decoder.value())

            # 将初始化的解码器赋给新的 lm head 解码器
            new_lm_head_decoder.assign(init_decoder)

        # 返回新的 lm head 解码器
        return new_lm_head_decoder
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
        # TODO (joao): flagged for replacement (by `_v2_get_resized_embeddings`) due to embeddings refactor
        # 获取旧嵌入维度
        old_embedding_dim = shape_list(old_embeddings)[1]
        # 从配置中获取初始化范围
        init_range = getattr(self.config, "initializer_range", 0.02)
        # 初始化嵌入层并生成嵌入掩码和当前嵌入
        embeddings_mask, current_embeddings = init_copy_embeddings(old_embeddings, new_num_tokens)
        # 添加新权重，根据指定的形状和初始化器
        new_embeddings = self.add_weight(
            name=old_embeddings.name.split(":")[0],
            shape=[new_num_tokens, old_embedding_dim],
            initializer=get_initializer(init_range),
            dtype=tf.float32,
        )
        # 根据嵌入掩码选择初始化的嵌入值
        init_embeddings = tf.where(embeddings_mask, current_embeddings, new_embeddings.value())

        # 将初始化的嵌入值赋给新的嵌入层
        new_embeddings.assign(init_embeddings)

        # 返回新的嵌入层
        return new_embeddings
    ) -> keras.layers.Embedding:
        """
        Build a resized Embedding layer from a provided Embedding layer. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end.

        Args:
            old_embeddings (`keras.layers.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

        Return:
            `keras.layers.Embedding`: Resized Embedding layer.
        """

        # Get the initialization range for the embeddings
        init_range = 0.02  # default value

        # Define potential variable names for initialization range
        potential_initialization_variable_names = [
            "initializer_range",  # most common
            "initializer_factor",  # e.g. T5
            "init_std",  # e.g BART
        ]

        # Iterate through potential variable names to find the correct initialization range
        for var_name in potential_initialization_variable_names:
            if hasattr(self.config, var_name):
                init_range = getattr(self.config, var_name)

        # Create a new Embedding layer with the specified parameters
        new_embeddings = keras.layers.Embedding(
            input_dim=new_num_tokens,
            output_dim=old_embeddings.output_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=init_range),
            name=old_embeddings.embeddings.name[:-13],  # exact same scoped name except "/embeddings:0"
        )
        
        # Initialize the new Embedding layer with a dummy input
        new_embeddings(tf.constant([[0]]))

        # Copy the old embeddings to the new embeddings
        if old_embeddings.input_dim >= new_num_tokens:
            init_embeddings = old_embeddings.embeddings[:new_num_tokens]
        else:
            init_embeddings = tf.concat(
                [old_embeddings.embeddings, new_embeddings.embeddings[old_embeddings.input_dim :]], axis=0
            )
        # Assign the initialized embeddings to the new embeddings layer
        new_embeddings.embeddings.assign(init_embeddings)
        
        # Return the resized Embedding layer
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
        """
        从预训练模型加载模型或者模型配置。

        Args:
            pretrained_model_name_or_path (str or os.PathLike):
                预训练模型名称或路径。
            *model_args:
                模型特定的额外参数。
            config (PretrainedConfig, str, os.PathLike, optional):
                可选的模型配置。
            cache_dir (str or os.PathLike, optional):
                可选的缓存目录。
            ignore_mismatched_sizes (bool):
                是否忽略大小不匹配的警告，默认为 False。
            force_download (bool):
                是否强制下载模型，默认为 False。
            local_files_only (bool):
                是否只使用本地文件，默认为 False。
            token (str or bool, optional):
                可选的身份验证令牌。
            revision (str):
                模型版本，默认为 "main"。
            use_safetensors (bool, optional):
                是否使用安全张量，默认为 None。
            **kwargs:
                其他未指定的关键字参数。
        """

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
    ):
        """
        将模型推送到模型中心（Hub）的指定仓库。

        Args:
            repo_id (str):
                仓库的唯一标识符。
            use_temp_dir (bool, optional):
                是否使用临时目录，默认为 None。
            commit_message (str, optional):
                提交消息，用于版本控制。
            private (bool, optional):
                是否将仓库设置为私有，默认为 None。
            max_shard_size (int or str, optional):
                最大的分片大小限制，默认为 "10GB"。
            token (bool or str, optional):
                身份验证令牌。
            use_auth_token (bool or str, optional):
                （已弃用）身份验证令牌，用于兼容目的。
            create_pr (bool):
                是否创建 Pull Request，默认为 False。
            **base_model_card_args:
                其他基本模型卡片参数。
    @classmethod
    def register_for_auto_class(cls, auto_class="TFAutoModel"):
        """
        注册当前类到给定的自动加载类中。这仅用于自定义模型，因为库中的模型已经与自动加载类映射。

        <Tip warning={true}>
        该 API 是实验性的，可能在未来的发布中有些许更改。
        </Tip>

        Args:
            auto_class (str or type, optional, defaults to "TFAutoModel"):
                要注册新模型的自动加载类。
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} 不是有效的自动加载类名。")

        cls._auto_class = auto_class
# 定义一个自定义的 1D 卷积层，按照 Radford 等人在 OpenAI GPT 中定义的方式（也用于 GPT-2）。

class TFConv1D(keras.layers.Layer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`):
            The number of output features.
        nx (`int`):
            The number of input features.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation to use to initialize the weights.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `keras.layers.Layer`.
    """

    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf  # 输出特征的数量
        self.nx = nx  # 输入特征的数量
        self.initializer_range = initializer_range  # 初始化权重时的标准差

    def build(self, input_shape):
        if self.built:
            return
        self.built = True
        # 添加权重变量：weight 的形状为 [nx, nf]，使用指定标准差的初始化器初始化
        self.weight = self.add_weight(
            "weight", shape=[self.nx, self.nf], initializer=get_initializer(self.initializer_range)
        )
        # 添加偏置变量：bias 的形状为 [1, nf]，使用零初始化器初始化
        self.bias = self.add_weight("bias", shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        bz, sl = shape_list(x)[:2]  # 获取输入张量 x 的批量大小和序列长度

        x = tf.reshape(x, [-1, self.nx])  # 将输入张量 x 重塑为二维张量
        x = tf.matmul(x, self.weight) + self.bias  # 执行矩阵乘法和偏置加法操作

        x = tf.reshape(x, [bz, sl, self.nf])  # 将结果重新塑造为原始序列张量的形状

        return x


class TFSharedEmbeddings(keras.layers.Layer):
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
            Additional keyword arguments passed along to the `__init__` of `keras.layers.Layer`.
    """

    # TODO (joao): flagged for delection due to embeddings refactor

    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size  # 词汇表大小，即唯一标记的数量
        self.hidden_size = hidden_size  # 嵌入向量的大小
        self.initializer_range = hidden_size**-0.5 if initializer_range is None else initializer_range
        # 如果未提供初始化标准差，则默认为 1/√hidden_size
        warnings.warn(
            "`TFSharedEmbeddings` is scheduled for deletion in v4.32, use `keras.layers.Embedding` instead.",
            DeprecationWarning,
        )
    def build(self, input_shape):
        """
        Build shared token embedding layer.

        This method initializes the layer's weight matrix based on the specified vocabulary size and hidden size.
        The weight matrix is initialized using a custom initializer within the specified range.

        Args:
            input_shape (tuple): Shape tuple describing the input shape.

        Returns:
            None
        """
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)
        )
        super().build(input_shape)

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: Configuration dictionary containing vocab_size, hidden_size, and initializer_range.
        """
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor, mode: str = "embedding") -> tf.Tensor:
        """
        Get token embeddings of inputs or decode final hidden state.

        Args:
            inputs (`tf.Tensor`):
                In embedding mode, should be an int64 tensor with shape `[batch_size, length]`.
                In linear mode, should be a float tensor with shape `[batch_size, length, hidden_size]`.
            mode (`str`, defaults to `"embedding"`):
                A valid value is either `"embedding"` or `"linear"`, indicating the layer's usage mode.

        Returns:
            `tf.Tensor`: Depending on mode,
                - In embedding mode: float32 embedding tensor, shape `[batch_size, length, embedding_size]`.
                - In linear mode: float32 tensor, shape `[batch_size, length, vocab_size]`.

        Raises:
            ValueError: if `mode` is not valid.
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, input_ids):
        """
        Applies embedding based on input_ids tensor.

        Args:
            input_ids: Tensor containing token indices.

        Returns:
            `tf.Tensor`: Float32 embedding tensor.
        """
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [..., hidden_size]

        Returns:
            `tf.Tensor`: Float32 tensor with shape [..., vocab_size].
        """
        first_dims = shape_list(inputs)[:-1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])
class TFSequenceSummary(keras.layers.Layer):
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
            Additional keyword arguments passed along to the `__init__` of `keras.layers.Layer`.
    """
    # 初始化函数，接受预训练配置和其他可选参数
    def __init__(self, config: PretrainedConfig, initializer_range: float = 0.02, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 根据配置确定摘要类型，如果配置中有 summary_use_proj 属性则使用其值，否则默认为 "last"
        self.summary_type = config.summary_type if hasattr(config, "summary_use_proj") else "last"
        
        # 如果摘要类型为 "attn"，抛出未实现错误，建议使用标准的多头注意力模块
        if self.summary_type == "attn":
            raise NotImplementedError
       
        # 判断配置中是否有 summary_use_proj 属性并且其值为 True，表示需要进行投影操作
        self.has_summary = hasattr(config, "summary_use_proj") and config.summary_use_proj
        if self.has_summary:
            # 如果配置中定义了 summary_proj_to_labels 并且其值为 True，并且 num_labels 大于 0
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            # 创建一个全连接层，用于摘要投影，输出维度为 num_classes
            self.summary = keras.layers.Dense(
                num_classes, kernel_initializer=get_initializer(initializer_range), name="summary"
            )

        # 判断配置中是否定义了 summary_activation 属性，如果有则设置相应的激活函数
        self.has_activation = False
        activation_string = getattr(config, "summary_activation", None)
        if activation_string is not None:
            self.has_activation = True
            self.activation = get_tf_activation(activation_string)

        # 判断配置中是否定义了 summary_first_dropout 属性并且其值大于 0，如果是则创建首层 Dropout
        self.has_first_dropout = hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0
        if self.has_first_dropout:
            self.first_dropout = keras.layers.Dropout(config.summary_first_dropout)

        # 判断配置中是否定义了 summary_last_dropout 属性并且其值大于 0，如果是则创建末层 Dropout
        self.has_last_dropout = hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0
        if self.has_last_dropout:
            self.last_dropout = keras.layers.Dropout(config.summary_last_dropout)
        
        # 将隐藏大小设置为配置中定义的 hidden_size
        self.hidden_size = config.hidden_size
    # 定义一个方法 `call`，用于执行模型的前向传播
    def call(self, inputs, cls_index=None, training=False):
        # 检查输入是否为字典、元组或列表，若不是则直接使用 `inputs` 作为隐藏状态
        if not isinstance(inputs, (dict, tuple, list)):
            hidden_states = inputs
        elif isinstance(inputs, (tuple, list)):
            # 若输入为元组或列表，则将第一个元素作为隐藏状态，第二个元素作为 `cls_index`（若有的话）
            hidden_states = inputs[0]
            cls_index = inputs[1] if len(inputs) > 1 else None
            assert len(inputs) <= 2, "Too many inputs."  # 断言输入的长度不超过2，否则报错
        else:
            # 若输入为字典，则从中获取 `hidden_states` 和 `cls_index`（默认为 None）
            hidden_states = inputs.get("hidden_states")
            cls_index = inputs.get("cls_index", None)

        # 根据 `summary_type` 选择如何汇总隐藏状态
        if self.summary_type == "last":
            output = hidden_states[:, -1]  # 取最后一个时间步的隐藏状态作为输出
        elif self.summary_type == "first":
            output = hidden_states[:, 0]  # 取第一个时间步的隐藏状态作为输出
        elif self.summary_type == "mean":
            output = tf.reduce_mean(hidden_states, axis=1)  # 对隐藏状态在第一维（batch 维度）上取平均
        elif self.summary_type == "cls_index":
            # 根据给定的 `cls_index` 从隐藏状态中取出对应位置的向量
            hidden_shape = shape_list(hidden_states)  # 获取隐藏状态的形状信息
            if cls_index is None:
                # 若 `cls_index` 为 None，则默认选择每个样本序列的最后一个位置
                cls_index = tf.fill(
                    hidden_shape[:-2], hidden_shape[-2] - 1
                )  # 创建一个张量，形状为 [batch] 或 [batch, num choices]，填充为序列长度
            cls_shape = shape_list(cls_index)
            if len(cls_shape) <= len(hidden_shape) - 2:
                cls_index = tf.expand_dims(cls_index, axis=-1)  # 在最后一维上扩展 `cls_index`
            # output shape: (batch, num choices, hidden_size)
            output = tf.gather(hidden_states, cls_index, batch_dims=len(hidden_shape) - 2)
            output = tf.squeeze(
                output, axis=len(hidden_shape) - 2
            )  # 压缩维度，输出形状为 (batch, num choices, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError  # 如果 `summary_type` 是 "attn"，则抛出未实现错误

        # 若模型具有第一个 dropout 层，则在输出上应用该 dropout
        if self.has_first_dropout:
            output = self.first_dropout(output, training=training)

        # 若模型具有汇总方法，则将输出传递给汇总方法
        if self.has_summary:
            output = self.summary(output)

        # 若模型具有激活函数，则将输出传递给激活函数
        if self.has_activation:
            output = self.activation(output)

        # 若模型具有最后一个 dropout 层，则在输出上应用该 dropout
        if self.has_last_dropout:
            output = self.last_dropout(output, training=training)

        return output

    # 构建模型，在输入形状已知的情况下进行构建
    def build(self, input_shape):
        if self.built:
            return  # 如果模型已经构建过，则直接返回
        self.built = True  # 标记模型已经构建
        if getattr(self, "summary", None) is not None:
            with tf.name_scope("summary"):
                self.summary.build(self.hidden_size)  # 使用汇总方法构建汇总层
# 定义一个函数，用于创建具有指定范围的截断正态分布初始化器
def get_initializer(initializer_range: float = 0.02) -> keras.initializers.TruncatedNormal:
    """
    Creates a `keras.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        `keras.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    # 返回一个截断正态分布初始化器对象，其标准差由参数 initializer_range 指定
    return keras.initializers.TruncatedNormal(stddev=initializer_range)
```