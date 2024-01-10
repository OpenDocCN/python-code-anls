# `transformer_vq\tests\utils\test_pipeline.py`

```
# 导入需要的库
import jax
import numpy as np
import pytest
import tensorflow as tf

# noreorder
# 导入自定义模块中的函数
from transformer_vq.utils.pipeline import pack_examples
from transformer_vq.utils.pipeline import pad_examples
from transformer_vq.utils.pipeline import examples_to_features
from transformer_vq.utils.pipeline import pad_batches
from tests.utils.fixtures import vocab

# 使用 pytest 的参数化装饰器，测试 pack_examples 函数
@pytest.mark.parametrize("append_eos", [True, False])
def test_pack_examples(vocab, append_eos):
    # 创建包含 tf.range(7) 的数据集
    ds = tf.data.Dataset.from_tensors(tf.range(7))
    # 调用 pack_examples 函数
    ds = pack_examples(ds, sequence_len=5, vocab=vocab, append_eos=append_eos)
    # 将数据集转换为列表
    actual = list(ds.as_numpy_iterator())
    actual_row0 = actual[0]
    actual_row1 = actual[1]
    expected_row0 = tf.constant([0, 1, 2, 3, 4])
    expected_row1 = tf.constant([5, 6])
    # 如果 append_eos 为 True，则在 expected_row1 后面填充 eos_id
    if append_eos:
        expected_row1 = tf.pad(expected_row1, [[0, 1]], constant_values=vocab.eos_id)
    # 使用 numpy.testing.assert_allclose 函数比较 actual_row0 和 expected_row0
    np.testing.assert_allclose(actual=actual_row0, desired=expected_row0)
    # 使用 numpy.testing.assert_allclose 函数比较 actual_row1 和 expected_row1

@pytest.mark.parametrize("append_eos", [True, False])
def test_pad_examples(vocab, append_eos):
    # 创建包含 tf.range(7) 的数据集
    ds = tf.data.Dataset.from_tensors(tf.range(7))
    # 调用 pack_examples 函数
    ds = pack_examples(ds, sequence_len=5, vocab=vocab, append_eos=append_eos)
    # 调用 pad_examples 函数
    ds = pad_examples(ds, sequence_len=5, vocab=vocab)
    # 将数据集转换为列表
    actual = list(ds.as_numpy_iterator())
    actual_row0 = actual[0]
    actual_row1 = actual[1]
    expected_row0 = tf.constant([0, 1, 2, 3, 4])
    expected_row1 = tf.constant([5, 6])
    # 如果 append_eos 为 True，则在 expected_row1 后面填充 eos_id
    if append_eos:
        expected_row1 = tf.pad(expected_row1, [[0, 1]], constant_values=vocab.eos_id)
        expected_row1 = tf.pad(expected_row1, [[0, 2]], constant_values=vocab.pad_id)
    else:
        expected_row1 = tf.pad(expected_row1, [[0, 3]], constant_values=vocab.pad_id)
    # 使用 numpy.testing.assert_allclose 函数比较 actual_row0 和 expected_row0
    np.testing.assert_allclose(actual=actual_row0, desired=expected_row0)
    # 使用 numpy.testing.assert_allclose 函数比较 actual_row1 和 expected_row1

# 测试 examples_to_features 函数
def test_examples_to_features(vocab):
    pad_id = vocab.pad_id
    # 获取词汇表中的结束符号的 ID
    eos_id = vocab.eos_id
    # 创建包含指定张量的数据集
    ds = tf.data.Dataset.from_tensors(tf.constant([10, 20, 30, pad_id, pad_id]))
    # 将数据集中的示例转换为特征
    ds = examples_to_features(ds=ds, sequence_len=5, vocab=vocab)
    # 将数据集转换为 numpy 迭代器，并获取第一个元素
    actual = list(ds.as_numpy_iterator())[0]
    # 使用 numpy.testing.assert_allclose 函数检查 actual["inputs"] 是否等于指定的张量
    np.testing.assert_allclose(
        actual=actual["inputs"],
        desired=tf.constant([eos_id, 10, 20, 30, pad_id]),
    )
    # 使用 numpy.testing.assert_allclose 函数检查 actual["targets"] 是否等于指定的张量
    np.testing.assert_allclose(
        actual=actual["targets"],
        desired=tf.constant([10, 20, 30, pad_id, pad_id]),
    )
    # 使用 numpy.testing.assert_allclose 函数检查 actual["loss_mask"] 是否等于指定的张量
    np.testing.assert_allclose(
        actual=actual["loss_mask"],
        desired=tf.constant([1, 1, 1, 0, 0]),
    )
# 测试填充批次函数，传入词汇表参数
def test_pad_batches(vocab):
    # 获取填充符的 ID
    pad_id = vocab.pad_id
    # 期望的填充前数据，包括输入、目标、文档 ID 和损失掩码
    expected = dict(
        inputs=tf.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, pad_id], [pad_id] * 5]),
        targets=tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, pad_id, pad_id], [pad_id] * 5]),
        doc_ids=tf.constant([[0] * 5, [0] * 5, [0] * 5]),
        loss_mask=tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0] * 5]),
    )
    # 从期望的数据中去除填充部分
    unpadded = jax.tree_util.tree_map(lambda x: x[0:2], expected)
    # 从给定的数据创建数据集
    ds = tf.data.Dataset.from_tensors(unpadded)
    # 调用填充批次函数，传入数据集、批次大小、序列长度和词汇表
    ds = pad_batches(ds=ds, batch_size=3, sequence_len=5, vocab=vocab)
    # 获取填充后的数据
    actual = list(ds.as_numpy_iterator())[0]
    # 遍历填充后的数据，对比每个键的值是否与期望一致
    for key in actual:
        np.testing.assert_allclose(actual=actual[key], desired=expected[key])
```