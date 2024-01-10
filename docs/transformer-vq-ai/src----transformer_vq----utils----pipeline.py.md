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

# 将数据集中的文档打包成指定长度的序列。对于图像数据集，我们将使用 seq_len = prod(img_shape) 和 append_eos=False 来使其成为一个无操作。
def pack_examples(ds, sequence_len, vocab, append_eos):
    def func(r):
        # 使用 vocab.eos_id 作为填充值，将序列填充到指定长度
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.eos_id)
        return tf.pad(r, [[0, 1]], **pad_kwargs)

    # 如果需要在序列末尾添加 eos 标记，则进行处理
    if append_eos:
        ds = ds.map(func)
    # 将数据集拆分为单个样本
    ds = ds.unbatch()
    # 将数据集重新打包成指定长度的批次
    ds = ds.batch(batch_size=sequence_len, drop_remainder=False)
    return ds

# 将样本填充到指定长度
def pad_examples(ds, sequence_len, vocab):
    def func(r):
        # 使用 vocab.pad_id 作为填充值，将样本填充到指定长度
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.pad_id)
        return tf.pad(r, [[0, sequence_len - tf.shape(r)[0]]], **pad_kwargs)

    return ds.map(func)

# 将样本转换为特征
def examples_to_features(ds, sequence_len, vocab):
    def func(r):
        # 使用 eos_id 作为序列的起始标记；我们假设 vocab 中没有 bos_id。
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.eos_id)
        # 生成输入序列和目标序列
        inputs = tf.pad(r[:-1], [[1, 0]], **pad_kwargs)
        targets = r
        # 检查输入序列中的 eos 标记和目标序列中的填充标记
        input_is_eos = tf.equal(inputs, vocab.eos_id * tf.ones_like(inputs))
        target_is_pad = tf.equal(targets, vocab.pad_id * tf.ones_like(targets))
        # 计算文档 ID 和损失掩码
        doc_ids = tf.cumsum(tf.cast(input_is_eos, tf.int32), axis=-1)
        loss_mask = 1 - tf.cast(target_is_pad, tf.int32)
        # 返回包含特征的字典
        return dict(
            inputs=tf.ensure_shape(inputs, [sequence_len]),
            targets=tf.ensure_shape(targets, [sequence_len]),
            doc_ids=tf.ensure_shape(doc_ids, [sequence_len]),
            loss_mask=tf.ensure_shape(loss_mask, [sequence_len]),
        )
    # 对数据集中的每个元素应用给定的函数，并返回结果数据集
    return ds.map(func)
def pad_batches(ds, batch_size, sequence_len, vocab):
    # 用于填充批次中过少的示例
    def func(r):
        # 定义填充规范，用于指定填充的方式和值
        pad_spec = [[0, batch_size - tf.shape(r["targets"])[0]], [0, 0]]
        # 使用 vocab.pad_id 在输入和目标的下方进行填充
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.pad_id)
        r["inputs"] = tf.pad(r["inputs"], pad_spec, **pad_kwargs)
        r["targets"] = tf.pad(r["targets"], pad_spec, **pad_kwargs)
        # 在 loss_mask 和 doc_ids 的下方使用零进行填充
        pad_kwargs = dict(mode="CONSTANT", constant_values=0)
        r["loss_mask"] = tf.pad(r["loss_mask"], pad_spec, **pad_kwargs)
        r["doc_ids"] = tf.pad(r["doc_ids"], pad_spec, **pad_kwargs)
        # 检查形状
        output_shape = [batch_size, sequence_len]
        r["inputs"] = tf.ensure_shape(r["inputs"], output_shape)
        r["targets"] = tf.ensure_shape(r["targets"], output_shape)
        r["doc_ids"] = tf.ensure_shape(r["doc_ids"], output_shape)
        r["loss_mask"] = tf.ensure_shape(r["loss_mask"], output_shape)
        return r

    return ds.map(func)


def get_batches(
    ds: tf.data.Dataset,
    batch_size: int,
    sequence_len: int,
    is_train: bool,
    vocab: seqio.Vocabulary,
    append_eos: bool,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    # 从数据集中提取批次
    # 每个批次是一个包含四个形状为 [batch_size, sequence_len] 的张量的元组
    # 这些张量分别是输入、目标、文档 ID 和损失掩码
    options = tf.data.Options()
    options.autotune.enabled = True
    common = dict(sequence_len=sequence_len, vocab=vocab)
    ds = pack_examples(ds, append_eos=append_eos, **common)
    ds = pad_examples(ds, **common)
    ds = examples_to_features(ds, **common)
    # 如果是训练模式
    if is_train:
        # 无限循环，产生大小为 batch_size 的批次数据
        ds = ds.repeat()
        # 使用当前时间和进程索引生成随机种子，对数据集进行洗牌
        shuffle_seed = int(time.time() + (10**9) * jax.process_index())
        ds = ds.shuffle(buffer_size=100_000, seed=shuffle_seed)
        # 将数据集划分为大小为 batch_size 的批次，丢弃不足一个批次的数据
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    else:
        # 产生所有批次数据，对不足一个批次的数据进行填充以覆盖完整的评估
        ds = ds.batch(batch_size=batch_size, drop_remainder=False)
        # 对数据集进行填充，使得所有批次数据大小均为 batch_size
        ds = pad_batches(ds, batch_size=batch_size, **common)
    # 预取数据，提前准备好下一个批次的数据以加速训练
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    # 将数据集转换为 numpy 迭代器并返回
    return ds.as_numpy_iterator()
```