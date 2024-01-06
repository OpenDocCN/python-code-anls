# `transformer_vq\tests\utils\test_pipeline.py`

```
# 导入所需的库
import jax
import numpy as np
import pytest
import tensorflow as tf

# 导入自定义模块
from transformer_vq.utils.pipeline import pack_examples
from transformer_vq.utils.pipeline import pad_examples
from transformer_vq.utils.pipeline import examples_to_features
from transformer_vq.utils.pipeline import pad_batches
from tests.utils.fixtures import vocab

# 使用 pytest 的参数化装饰器，定义测试用例参数化
@pytest.mark.parametrize("append_eos", [True, False])
# 定义测试函数，测试 pack_examples 函数
def test_pack_examples(vocab, append_eos):
    # 创建一个包含从 0 到 6 的张量的数据集
    ds = tf.data.Dataset.from_tensors(tf.range(7))
    # 调用 pack_examples 函数，对数据集进行打包处理
    ds = pack_examples(ds, sequence_len=5, vocab=vocab, append_eos=append_eos)
    # 将处理后的数据集转换为列表
    actual = list(ds.as_numpy_iterator())
    # 获取处理后数据集的第一行和第二行
    actual_row0 = actual[0]
    actual_row1 = actual[1]
# 创建一个张量，表示预期的第一行数据
expected_row0 = tf.constant([0, 1, 2, 3, 4])
# 创建一个张量，表示预期的第二行数据
expected_row1 = tf.constant([5, 6])
# 如果需要在第二行数据末尾添加结束符，则使用 tf.pad 函数在末尾添加结束符
if append_eos:
    expected_row1 = tf.pad(expected_row1, [[0, 1]], constant_values=vocab.eos_id)
# 使用 np.testing.assert_allclose 函数检查实际结果和预期结果是否一致
np.testing.assert_allclose(actual=actual_row0, desired=expected_row0)
np.testing.assert_allclose(actual=actual_row1, desired=expected_row1)

# 使用 pytest.mark.parametrize 注解来指定参数化测试的参数
@pytest.mark.parametrize("append_eos", [True, False])
# 定义测试函数，测试 pad_examples 函数的功能
def test_pad_examples(vocab, append_eos):
    # 创建一个包含一个张量的数据集
    ds = tf.data.Dataset.from_tensors(tf.range(7))
    # 调用 pack_examples 函数对数据集进行处理
    ds = pack_examples(ds, sequence_len=5, vocab=vocab, append_eos=append_eos)
    # 调用 pad_examples 函数对数据集进行处理
    ds = pad_examples(ds, sequence_len=5, vocab=vocab)
    # 将数据集转换为列表
    actual = list(ds.as_numpy_iterator())
    # 获取实际结果中的第一行数据
    actual_row0 = actual[0]
    # 获取实际结果中的第二行数据
    actual_row1 = actual[1]
    # 创建一个张量，表示预期的第一行数据
    expected_row0 = tf.constant([0, 1, 2, 3, 4])
    # 创建一个张量，表示预期的第二行数据
    expected_row1 = tf.constant([5, 6])
    # 如果需要在第二行数据末尾添加结束符，则使用 tf.pad 函数在末尾添加结束符
    if append_eos:
        expected_row1 = tf.pad(expected_row1, [[0, 1]], constant_values=vocab.eos_id)
# 如果条件成立，对 expected_row1 进行填充操作，使其在第一维上增加 2 个元素，填充值为 vocab.pad_id
expected_row1 = tf.pad(expected_row1, [[0, 2]], constant_values=vocab.pad_id)
# 如果条件不成立，对 expected_row1 进行填充操作，使其在第一维上增加 3 个元素，填充值为 vocab.pad_id
else:
    expected_row1 = tf.pad(expected_row1, [[0, 3]], constant_values=vocab.pad_id)
# 使用 np.testing.assert_allclose 函数对 actual_row0 和 expected_row0 进行数值比较
np.testing.assert_allclose(actual=actual_row0, desired=expected_row0)
# 使用 np.testing.assert_allclose 函数对 actual_row1 和 expected_row1 进行数值比较
np.testing.assert_allclose(actual=actual_row1, desired=expected_row1)

# 定义测试函数 test_examples_to_features，参数为 vocab
def test_examples_to_features(vocab):
    # 获取 vocab 中的 pad_id 和 eos_id
    pad_id = vocab.pad_id
    eos_id = vocab.eos_id
    # 创建一个包含单个张量的数据集 ds
    ds = tf.data.Dataset.from_tensors(tf.constant([10, 20, 30, pad_id, pad_id]))
    # 调用 examples_to_features 函数对数据集 ds 进行处理，设置序列长度为 5，vocab 为参数
    ds = examples_to_features(ds=ds, sequence_len=5, vocab=vocab)
    # 将处理后的数据集转换为 numpy 迭代器，并取出第一个元素
    actual = list(ds.as_numpy_iterator())[0]
    # 使用 np.testing.assert_allclose 函数对 actual["inputs"] 和 tf.constant([eos_id, 10, 20, 30, pad_id]) 进行数值比较
    np.testing.assert_allclose(
        actual=actual["inputs"],
        desired=tf.constant([eos_id, 10, 20, 30, pad_id]),
    )
    # 使用 np.testing.assert_allclose 函数对 actual["targets"] 和 tf.constant([10, 20, 30, pad_id, pad_id]) 进行数值比较
    np.testing.assert_allclose(
        actual=actual["targets"],
        desired=tf.constant([10, 20, 30, pad_id, pad_id]),
    )
    # 使用 NumPy 测试库中的 assert_allclose 函数，比较实际值和期望值的近似程度
    np.testing.assert_allclose(
        actual=actual["loss_mask"],
        desired=tf.constant([1, 1, 1, 0, 0]),
    )


def test_pad_batches(vocab):
    # 获取填充符的 ID
    pad_id = vocab.pad_id
    # 创建期望的填充后的批次数据
    expected = dict(
        inputs=tf.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, pad_id], [pad_id] * 5]),
        targets=tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, pad_id, pad_id], [pad_id] * 5]),
        doc_ids=tf.constant([[0] * 5, [0] * 5, [0] * 5]),
        loss_mask=tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0] * 5]),
    )
    # 从期望的填充前数据中去掉填充部分，得到未填充的数据
    unpadded = jax.tree_util.tree_map(lambda x: x[0:2], expected)
    # 从未填充的数据创建 TensorFlow 数据集
    ds = tf.data.Dataset.from_tensors(unpadded)
    # 对数据集进行填充
    ds = pad_batches(ds=ds, batch_size=3, sequence_len=5, vocab=vocab)
    # 从数据集中获取实际的填充后数据
    actual = list(ds.as_numpy_iterator())[0]
    # 遍历实际数据的键
    for key in actual:
# 使用 NumPy 测试模块中的 assert_allclose 函数，比较 actual 字典中 key 对应的值和 expected 字典中 key 对应的值是否在允许误差范围内相等。
```