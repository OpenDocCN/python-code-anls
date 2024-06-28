# `.\tf_utils.py`

```
# 导入必要的库和模块
from typing import List, Optional, Union

import numpy as np  # 导入NumPy库，用于数值计算
import tensorflow as tf  # 导入TensorFlow库，用于机器学习模型构建和训练

from .feature_extraction_utils import BatchFeature  # 导入自定义的特征提取工具类
from .tokenization_utils_base import BatchEncoding  # 导入自定义的编码工具类
from .utils import logging  # 导入自定义的日志工具模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    处理 TensorFlow 中的动态形状。

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): 要获取形状的张量或数组。

    Returns:
        `List[int]`: 张量的形状列表。
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)  # 返回数组的形状列表

    dynamic = tf.shape(tensor)  # 获取 TensorFlow 张量的动态形状

    if tensor.shape == tf.TensorShape(None):  # 如果张量的静态形状未知
        return dynamic  # 返回动态形状

    static = tensor.shape.as_list()  # 获取张量的静态形状列表

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]  # 返回静态形状或动态形状的组合

def stable_softmax(logits: tf.Tensor, axis: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    """
    稳定的 softmax 函数，用于解决 TensorFlow 在 CPU 上与 XLA 结合时的问题。

    Args:
        logits (`tf.Tensor`): 输入的对数概率张量。
        axis (`int`, *optional*): 执行 softmax 操作的维度，默认为 -1 表示最后一个维度。
        name (`str`, *optional*): 操作的名称。

    Returns:
        `tf.Tensor`: 与 logits 具有相同类型和形状的张量，经过 softmax 处理。
    """
    # TODO: 当上述问题得到解决后，检查 TensorFlow 版本并使用原始函数，最终移除这个函数。
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)  # 添加一个小量以确保数值稳定性后进行 softmax 操作

def functional_layernorm(inputs, weight, bias, epsilon=1e-5, axis=-1):
    # 这是一个简化的功能性 layernorm，用于在需要时复制 PyTorch nn.functional.layer_norm 的功能
    # （待补充完整，具体实现未提供）
    # 检查权重和偏置的维度是否为1，以及轴是否为整数，若不符合则抛出未实现的错误
    if weight.shape.rank != 1 or bias.shape.rank != 1 or not isinstance(axis, int):
        raise NotImplementedError("Only 1D weight and bias tensors are supported for now, with only a single axis.")

    # 计算在指定轴上输入数据的均值和方差
    mean, variance = tf.nn.moments(inputs, axes=[axis], keepdims=True)

    if axis != -1:
        # 若轴不是最后一个轴（-1），则重塑权重和偏置的形状，使其与输入数据具有相同的秩，但在除了指定轴外的所有维度上都是1
        shape = [1] * inputs.shape.rank
        shape[axis] = shape_list(inputs)[axis]
        weight = tf.reshape(weight, shape)
        bias = tf.reshape(bias, shape)

    # 使用批量归一化函数 tf.nn.batch_normalization 计算层归一化
    outputs = tf.nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=bias,
        scale=weight,
        variance_epsilon=epsilon,
    )
    # 返回归一化后的输出结果
    return outputs
def flatten(input, start_dim=0, end_dim=-1):
    # Replicates the behavior of torch.flatten in TF

    # If end_dim or start_dim is negative, count them from the end
    # 如果 end_dim 或 start_dim 是负数，则从末尾开始计数
    if end_dim < 0:
        end_dim += input.shape.rank
    if start_dim < 0:
        start_dim += input.shape.rank

    # Return input tensor if start_dim equals end_dim
    # 如果 start_dim 等于 end_dim，则返回输入张量
    if start_dim == end_dim:
        return input

    # Get the shape of the input tensor
    # 获取输入张量的形状
    in_shape = tf.shape(input)
    
    # Calculate the total size of the flattened dimensions
    # 计算被展平维度的总大小
    flattened_dim = tf.math.reduce_prod(in_shape[start_dim : end_dim + 1])
    
    # Construct the output shape with the flattened dimensions
    # 使用展平后的维度构造输出形状
    out_shape = tf.concat([in_shape[:start_dim], [flattened_dim], in_shape[end_dim + 1 :]], axis=0)
    
    # Reshape the input tensor to the calculated output shape
    # 将输入张量重塑为计算得到的输出形状
    return tf.reshape(input, out_shape)


def invert_attention_mask(encoder_attention_mask: tf.Tensor) -> tf.Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `tf.Tensor`: The inverted attention mask.
    """
    if not isinstance(encoder_attention_mask, tf.Tensor):
        encoder_attention_mask = tf.convert_to_tensor(encoder_attention_mask)  # Catches stray NumPy inputs
    
    # Extend the attention mask tensor based on its rank
    # 根据张量的秩扩展注意力掩码张量
    if encoder_attention_mask.shape.rank == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.shape.rank == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    
    # Invert the extended attention mask values
    # 反转扩展后的注意力掩码值
    encoder_extended_attention_mask = (
        tf.cast(1, encoder_attention_mask.dtype) - encoder_extended_attention_mask
    ) * encoder_extended_attention_mask.dtype.min

    return encoder_extended_attention_mask


def check_embeddings_within_bounds(tensor: tf.Tensor, embed_dim: int, tensor_name: str = "input_ids") -> None:
    """
    `tf.gather`, on which TF embedding layers are based, won't check positive out of bound indices on GPU, returning
    zeros instead. This function adds a check against that dangerous silent behavior.

    Args:
        tensor (`tf.Tensor`): The tensor of indices to check.
        embed_dim (`int`): The embedding dimension.
        tensor_name (`str`, *optional*): The name of the tensor to use in the error message.
    """
    # Assert that all indices in tensor are less than embed_dim
    # 断言张量中所有的索引都小于 embed_dim
    tf.debugging.assert_less(
        tensor,
        tf.cast(embed_dim, dtype=tensor.dtype),
        message=(
            f"The maximum value of {tensor_name} ({tf.math.reduce_max(tensor)}) must be smaller than the embedding "
            f"layer's input dimension ({embed_dim}). The likely cause is some problem at tokenization time."
        ),
    )


def save_attributes_to_hdf5_group(group, name, data):
    """Saves attributes (data) of the specified name into the HDF5 group.
    
    This function saves attributes (data) with a given name into the specified HDF5 group.

    Args:
        group: HDF5 group where the attributes will be saved.
        name: Name of the attribute to save.
        data: Data to be saved as the attribute.
    """
    """
    This method deals with an inherent problem of HDF5 file which is not able to store data larger than
    HDF5_OBJECT_HEADER_LIMIT bytes.

    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to save.
        data: Attributes data to store.

    Raises:
        RuntimeError: If any single attribute is too large to be saved.

    Copied from Keras to Transformers to avoid versioning issues.
    """
    # 定义 HDF5 文件中对象头部限制大小为 64512 字节
    HDF5_OBJECT_HEADER_LIMIT = 64512

    # 检查所有数据项是否超过 HDF5_OBJECT_HEADER_LIMIT 字节
    # 如果超过，则无论如何切块都无法保存
    bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

    # 如果存在超过限制的属性，则抛出 RuntimeError 异常
    if bad_attributes:
        raise RuntimeError(
            "The following attributes cannot be saved to HDF5 file because "
            f"they are larger than {HDF5_OBJECT_HEADER_LIMIT} "
            f"bytes: {bad_attributes}"
        )

    # 将数据转换为 NumPy 数组
    data_npy = np.asarray(data)

    # 初始时只有一个块
    num_chunks = 1

    # 将数据切分成多个块
    chunked_data = np.array_split(data_npy, num_chunks)

    # 如果有任何一个块的大小超过 HDF5_OBJECT_HEADER_LIMIT，则继续切块
    while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
        num_chunks += 1
        chunked_data = np.array_split(data_npy, num_chunks)

    # 如果切块数大于 1，则逐个保存每个块
    if num_chunks > 1:
        for chunk_id, chunk_data in enumerate(chunked_data):
            group.attrs["%s%d" % (name, chunk_id)] = chunk_data
    else:
        # 否则直接保存数据到 HDF5 文件
        group.attrs[name] = data
# 从指定的 HDF5 组中加载指定名称的属性数据
def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem of HDF5 file which is not able to store data larger than
    HDF5_OBJECT_HEADER_LIMIT bytes.

    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    Returns:
        data: Attributes data.

    Copied from Keras to Transformers to avoid versioning issues.
    """
    # 检查属性名是否存在于 HDF5 组的属性中
    if name in group.attrs:
        # 将属性数据解码为 UTF-8 格式的字符串，如果属性是字节流则直接返回
        data = [n.decode("utf8") if hasattr(n, "decode") else n for n in group.attrs[name]]
    else:
        data = []  # 如果属性名不存在，初始化一个空列表
        chunk_id = 0
        # 持续循环直到找不到以 name + chunk_id 命名的属性
        while "%s%d" % (name, chunk_id) in group.attrs:
            # 将属性数据解码为 UTF-8 格式的字符串，如果属性是字节流则直接返回，扩展到 data 列表中
            data.extend(
                [n.decode("utf8") if hasattr(n, "decode") else n for n in group.attrs["%s%d" % (name, chunk_id)]]
            )
            chunk_id += 1
    # 返回加载的属性数据
    return data


# 将 1 维张量扩展为 2 维张量
def expand_1d(data):
    """Expands 1-dimensional `Tensor`s into 2-dimensional `Tensor`s.
    Copied from Keras to here to avoid versioning issues."""

    def _expand_single_1d_tensor(t):
        # 如果输入是 TensorFlow 的张量且为 1 维，则在最后一个维度上扩展为 2 维张量
        if isinstance(t, tf.Tensor) and t.shape.rank == 1:
            return tf.expand_dims(t, axis=-1)
        return t

    # 使用 tf.nest.map_structure 对数据结构中的每个元素应用 _expand_single_1d_tensor 函数
    return tf.nest.map_structure(_expand_single_1d_tensor, data)


# 将 HF BatchEncoding/BatchFeature 对象转换为 Keras 可理解的字典格式
def convert_batch_encoding(*args, **kwargs):
    # 如果参数中存在且第一个参数是 BatchEncoding 或 BatchFeature 类型的对象，则转换为字典
    if args and isinstance(args[0], (BatchEncoding, BatchFeature)):
        args = list(args)
        args[0] = dict(args[0])
    # 如果 kwargs 中存在键 "x" 且其值是 BatchEncoding 或 BatchFeature 类型的对象，则转换为字典
    elif "x" in kwargs and isinstance(kwargs["x"], (BatchEncoding, BatchFeature)):
        kwargs["x"] = dict(kwargs["x"])
    # 返回转换后的参数和关键字参数
    return args, kwargs
```