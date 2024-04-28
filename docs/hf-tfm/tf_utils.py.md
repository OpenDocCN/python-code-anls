# `.\transformers\tf_utils.py`

```
# 导入所需模块和类型提示
from typing import List, Optional, Union
# 导入 NumPy 库
import numpy as np
# 导入 TensorFlow 库
import tensorflow as tf
# 从当前包中导入 logging 模块
from .utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义函数 shape_list，用于获取张量的形状
def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    # 若输入为 NumPy 数组，则返回其形状的列表表示
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    # 获取 TensorFlow 张量的动态形状
    dynamic = tf.shape(tensor)

    # 若张量形状为未知，则返回动态形状
    if tensor.shape == tf.TensorShape(None):
        return dynamic

    # 获取 TensorFlow 张量的静态形状
    static = tensor.shape.as_list()

    # 返回静态形状的列表表示，若某一维度为 None，则用对应的动态形状替代
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


# 定义函数 stable_softmax，用于实现稳定的 softmax 操作
def stable_softmax(logits: tf.Tensor, axis: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    """
    Stable wrapper that returns the same output as `tf.nn.softmax`, but that works reliably with XLA on CPU. It is
    meant as a workaround for the [following issue](https://github.com/tensorflow/tensorflow/issues/55682), and will be
    removed after it gets fixed. The arguments and outputs are the same as `tf.nn.softmax`, and relies on the fact that
    `softmax(x) = softmax(x + c)` (see https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html).

    Args:
        logits (`tf.Tensor`):
            Must be one of the following types: half, float32, float64.
        axis (`int`, *optional*):
            The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name (`str`, *optional*):
            A name for the operation.

    Returns:
        `tf.Tensor`:
            A Tensor. Has the same type and shape as logits.
    """
    # 使用带偏移的 logits 计算 softmax，以解决在 CPU 上使用 XLA 时的稳定性问题
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)


# 定义函数 functional_layernorm，用于实现功能性的 LayerNorm 操作
def functional_layernorm(inputs, weight, bias, epsilon=1e-5, axis=-1):
    # This is a very simplified functional layernorm, designed to duplicate
    # the functionality of PyTorch nn.functional.layer_norm when this is needed to port
    # models in Transformers.

    # 这是一个非常简化的功能性 LayerNorm，旨在在需要转换 Transformers 模型时复制 PyTorch nn.functional.layer_norm 的功能
```  
    # 检查权重和偏置的形状是否为一维，以及轴是否为整数类型，如果不满足条件，则抛出NotImplementedError异常
    if weight.shape.rank != 1 or bias.shape.rank != 1 or not isinstance(axis, int):
        raise NotImplementedError("Only 1D weight and bias tensors are supported for now, with only a single axis.")

    # 计算在需要归一化的轴上的均值和方差
    mean, variance = tf.nn.moments(inputs, axes=[axis], keepdims=True)

    # 如果轴不等于-1，则重塑比例和权重，使其具有与输入相同的秩，但在除轴之外的每个维度上具有1个维度
    if axis != -1:
        shape = [1] * inputs.shape.rank
        shape[axis] = shape_list(inputs)[axis]
        weight = tf.reshape(weight, shape)
        bias = tf.reshape(bias, shape)

    # 使用batch_normalization函数计算层归一化
    outputs = tf.nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=bias,
        scale=weight,
        variance_epsilon=epsilon,
    )
    # 返回归一化后的输出
    return outputs
# 将输入张量展平，复制了 torch.flatten 在 TF 中的行为
def flatten(input, start_dim=0, end_dim=-1):
    # 如果 end_dim 或 start_dim 为负数，则从末尾开始计算
    if end_dim < 0:
        end_dim += input.shape.rank
    if start_dim < 0:
        start_dim += input.shape.rank

    # 如果 start_dim 等于 end_dim，则返回输入张量
    if start_dim == end_dim:
        return input

    # 获取输入张量的形状
    in_shape = tf.shape(input)
    # 计算展平后的维度
    flattened_dim = tf.math.reduce_prod(in_shape[start_dim : end_dim + 1])
    # 构建展平后的形状
    out_shape = tf.concat([in_shape[:start_dim], [flattened_dim], in_shape[end_dim + 1 :]], axis=0)
    # 返回展平后的张量
    return tf.reshape(input, out_shape)


def invert_attention_mask(encoder_attention_mask: tf.Tensor) -> tf.Tensor:
    """
    反转注意力掩码（例如，将 0 和 1 互换）。

    Args:
        encoder_attention_mask (`torch.Tensor`): 注意力掩码。

    Returns:
        `tf.Tensor`: 反转后的注意力掩码。
    """
    # 如果输入不是张量，则将其转换为张量（捕捉到误输入的 NumPy）
    if not isinstance(encoder_attention_mask, tf.Tensor):
        encoder_attention_mask = tf.convert_to_tensor(encoder_attention_mask)
    # 如果注意力掩码的秩为 3
    if encoder_attention_mask.shape.rank == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    # 如果注意力掩码的秩为 2
    if encoder_attention_mask.shape.rank == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 模型有一个可以比较序列 ID 的掩码，我们可以通过这种转置来模拟
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = (
        tf.cast(1, encoder_attention_mask.dtype) - encoder_extended_attention_mask
    ) * encoder_extended_attention_mask.dtype.min

    return encoder_extended_attention_mask


def check_embeddings_within_bounds(tensor: tf.Tensor, embed_dim: int, tensor_name: str = "input_ids") -> None:
    """
    `tf.gather`，TF 嵌入层的基础，不会在 GPU 上检查正值超出范围的索引，而是返回零。此函数添加了对该危险的静默行为的检查。

    Args:
        tensor (`tf.Tensor`): 要检查的索引张量。
        embed_dim (`int`): 嵌入维度。
        tensor_name (`str`, *optional*): 在错误消息中使用的张量名称。
    """
    # 断言索引张量的值小于嵌入层的输入维度
    tf.debugging.assert_less(
        tensor,
        tf.cast(embed_dim, dtype=tensor.dtype),
        message=(
            f"{tensor_name} 的最大值 ({tf.math.reduce_max(tensor)}) 必须小于嵌入层的输入维度 ({embed_dim})。"
            "可能的原因是在分词时出现了一些问题。"
        ),
    )


def save_attributes_to_hdf5_group(group, name, data):
    """将指定名称的属性（数据）保存到 HDF5 组中。"""
    This method deals with an inherent problem of HDF5 file which is not able to store data larger than
    HDF5_OBJECT_HEADER_LIMIT bytes.

    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to save.
         Attributes data to store.

    Raises:
      RuntimeError: If any single attribute is too large to be saved.

    Copied from Keras to Transformers to avoid versioning issues.
    """
    # 定义 HDF5_OBJECT_HEADER_LIMIT 常量，表示 HDF5 文件对象头限制
    HDF5_OBJECT_HEADER_LIMIT = 64512
    # 检查 `data` 中的任何一项是否大于 `HDF5_OBJECT_HEADER_LIMIT`，如果是，则无法保存
    # 因为即使对数组进行分块，也无法使保存变得可能。
    bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

    # 期望这个条件永远不会为真。
    if bad_attributes:
        # 如果有大于 HDF5_OBJECT_HEADER_LIMIT 的属性，则抛出异常
        raise RuntimeError(
            "The following attributes cannot be saved to HDF5 file because "
            f"they are larger than {HDF5_OBJECT_HEADER_LIMIT} "
            f"bytes: {bad_attributes}"
        )

    # 将 data 转换为 NumPy 数组
    data_npy = np.asarray(data)

    # 初始化分块数为1
    num_chunks = 1
    # 将数据分成多个块
    chunked_data = np.array_split(data_npy, num_chunks)

    # 这个循环永远不会无限循环，因为上面的测试已经排除了这种可能性。
    while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
        # 如果任何一个块的大小超过了 HDF5_OBJECT_HEADER_LIMIT，就增加块的数量，重新分块
        num_chunks += 1
        chunked_data = np.array_split(data_npy, num_chunks)

    # 如果分成多个块
    if num_chunks > 1:
        # 遍历每个块并将其存储到 HDF5 group 的属性中
        for chunk_id, chunk_data in enumerate(chunked_data):
            group.attrs["%s%d" % (name, chunk_id)] = chunk_data
    else:
        # 如果只有一个块，则直接存储到 HDF5 group 的属性中
        group.attrs[name] = data
```  
# 从 HDF5 组中加载指定名称的属性
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
    # 检查属性是否存在于组中
    if name in group.attrs:
        # 如果存在，将属性数据解码为 UTF-8 格式
        data = [n.decode("utf8") if hasattr(n, "decode") else n for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        # 处理属性数据分块的情况
        while "%s%d" % (name, chunk_id) in group.attrs:
            # 将属性数据解码为 UTF-8 格式并添加到结果列表中
            data.extend(
                [n.decode("utf8") if hasattr(n, "decode") else n for n in group.attrs["%s%d" % (name, chunk_id)]]
            )
            chunk_id += 1
    return data


# 将 1 维张量扩展为 2 维张量
def expand_1d(data):
    """Expands 1-dimensional `Tensor`s into 2-dimensional `Tensor`s.
    Copied from Keras to here to avoid versioning issues."""

    # 定义内部函数，用于扩展单个 1 维张量
    def _expand_single_1d_tensor(t):
        # 如果是 TensorFlow 张量且为 1 维，则在最后一个维度上扩展为 2 维
        if isinstance(t, tf.Tensor) and t.shape.rank == 1:
            return tf.expand_dims(t, axis=-1)
        return t

    # 使用 map_structure 函数将内部函数应用于数据结构中的每个元素
    return tf.nest.map_structure(_expand_single_1d_tensor, data)
```