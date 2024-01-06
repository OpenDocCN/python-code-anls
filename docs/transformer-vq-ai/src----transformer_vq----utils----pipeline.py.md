# `transformer_vq\src\transformer_vq\utils\pipeline.py`

```
# 导入所需的库
import time
from typing import Iterator
from typing import Tuple
import jax
import numpy as np
import seqio
import tensorflow as tf

# 设置 TensorFlow 日志级别为 ERROR
tf.get_logger().setLevel("ERROR")

# 定义函数 pack_examples，用于将数据集打包成指定长度的序列
def pack_examples(ds, sequence_len, vocab, append_eos):
    # 定义内部函数 func，用于对数据进行填充，使其长度符合指定的序列长度
    def func(r):
        # 定义填充参数
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.eos_id)
        # 对数据进行填充
        return tf.pad(r, [[0, 1]], **pad_kwargs)

    # 如果需要在序列末尾添加结束符号
    if append_eos:
# 对数据集中的每个元素应用函数func
ds = ds.map(func)
# 取消数据集的批处理
ds = ds.unbatch()
# 对数据集进行批处理，指定批处理大小为sequence_len，不丢弃剩余数据
ds = ds.batch(batch_size=sequence_len, drop_remainder=False)
# 返回处理后的数据集

# 对数据集中的每个元素进行填充，使其长度达到sequence_len
def pad_examples(ds, sequence_len, vocab):
    # 填充示例使其达到长度sequence_len
    # 对于打包后可能存在的“剩余”序列而言，这不是一个空操作
    # 对于图像来说，由于打包是一个空操作且没有剩余部分，这总是一个空操作
    def func(r):
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.pad_id)
        return tf.pad(r, [[0, sequence_len - tf.shape(r)[0]]], **pad_kwargs)
    return ds.map(func)

# 将示例转换为特征
def examples_to_features(ds, sequence_len, vocab):
    def func(r):
        # 使用eos_id来提示序列；我们假设词汇表中没有bos_id
        # 定义填充参数，使用常量值填充，常量值为词汇表中的结束符号的索引
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.eos_id)
        # 对输入进行填充，将其长度扩展到 sequence_len，使用指定的填充参数
        inputs = tf.pad(r[:-1], [[1, 0]], **pad_kwargs)
        # 目标数据为原始数据 r
        targets = r
        # 判断输入是否为结束符号的索引，返回布尔值
        input_is_eos = tf.equal(inputs, vocab.eos_id * tf.ones_like(inputs))
        # 判断目标数据是否为填充值的索引，返回布尔值
        target_is_pad = tf.equal(targets, vocab.pad_id * tf.ones_like(targets))
        # 计算文档 ID，使用输入是否为结束符号的布尔值进行累加
        doc_ids = tf.cumsum(tf.cast(input_is_eos, tf.int32), axis=-1)
        # 计算损失掩码，目标数据不为填充值的索引为 1，否则为 0
        loss_mask = 1 - tf.cast(target_is_pad, tf.int32)
        # 返回填充后的数据，确保数据的形状符合指定的 sequence_len
        return dict(
            inputs=tf.ensure_shape(inputs, [sequence_len]),
            targets=tf.ensure_shape(targets, [sequence_len]),
            doc_ids=tf.ensure_shape(doc_ids, [sequence_len]),
            loss_mask=tf.ensure_shape(loss_mask, [sequence_len]),
        )

    # 对数据集中的每个元素应用 func 函数
    return ds.map(func)


def pad_batches(ds, batch_size, sequence_len, vocab):
    # 对批次进行填充，确保每个批次中的序列长度一致
    def func(r):
        # 创建一个填充规范，用于将输入和目标下方填充到指定的批量大小
        pad_spec = [[0, batch_size - tf.shape(r["targets"])[0]], [0, 0]]
        # 使用指定的填充值对输入和目标进行填充
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.pad_id)
        r["inputs"] = tf.pad(r["inputs"], pad_spec, **pad_kwargs)
        r["targets"] = tf.pad(r["targets"], pad_spec, **pad_kwargs)
        # 使用零对损失掩码和文档 ID 进行填充
        pad_kwargs = dict(mode="CONSTANT", constant_values=0)
        r["loss_mask"] = tf.pad(r["loss_mask"], pad_spec, **pad_kwargs)
        r["doc_ids"] = tf.pad(r["doc_ids"], pad_spec, **pad_kwargs)
        # 检查形状是否符合要求
        output_shape = [batch_size, sequence_len]
        r["inputs"] = tf.ensure_shape(r["inputs"], output_shape)
        r["targets"] = tf.ensure_shape(r["targets"], output_shape)
        r["doc_ids"] = tf.ensure_shape(r["doc_ids"], output_shape)
        r["loss_mask"] = tf.ensure_shape(r["loss_mask"], output_shape)
        # 返回结果
        return r

    # 将函数应用到数据集上
    return ds.map(func)
# 从数据集中获取批次数据
# 每个批次是一个包含四个形状为 [batch_size, sequence_len] 的张量的元组
# 这些张量分别是输入、目标、文档 ID 和损失掩码
options = tf.data.Options()
options.autotune.enabled = True
common = dict(sequence_len=sequence_len, vocab=vocab)
# 将示例打包成数据集，根据参数是否添加结束符
ds = pack_examples(ds, append_eos=append_eos, **common)
# 对示例进行填充，使它们的长度一致
ds = pad_examples(ds, **common)
# 将示例转换为特征
ds = examples_to_features(ds, **common)
if is_train:
    # 如果是训练模式，循环无限次，产生大小为 batch_size 的批次
    ds = ds.repeat()
# 生成随机种子，确保每次洗牌的结果都不同
shuffle_seed = int(time.time() + (10**9) * jax.process_index())
# 对数据集进行洗牌操作，设置缓冲区大小为100000，使用生成的随机种子
ds = ds.shuffle(buffer_size=100_000, seed=shuffle_seed)
# 对数据集进行分批处理，设置批大小为batch_size，丢弃不足一个批次的数据
ds = ds.batch(batch_size=batch_size, drop_remainder=True)
# 如果是评估模式
else:
    # 生成所有批次，对不足一个批次的数据进行填充，以确保完整的评估覆盖
    ds = ds.batch(batch_size=batch_size, drop_remainder=False)
    # 对批次进行填充，使用pad_batches函数，设置批大小为batch_size，传入common参数
    ds = pad_batches(ds, batch_size=batch_size, **common)
# 预取数据，使用tf.data.AUTOTUNE自动调整缓冲区大小
ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
# 将数据集转换为numpy迭代器并返回
return ds.as_numpy_iterator()
```