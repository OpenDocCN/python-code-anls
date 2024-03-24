# `.\lucidrains\progen\progen_transformer\data.py`

```
# 导入所需的库
import tensorflow as tf
import numpy as np
from functools import partial
from pathlib import Path
from contextlib import contextmanager

# 写入 tfrecords

# 定义写入函数，将值写入 tfrecord 文件
def write(writer, values):
    # 将值序列化为字节流
    record_bytes = tf.train.Example(features = tf.train.Features(feature={
        'seq': tf.train.Feature(bytes_list = tf.train.BytesList(value=[values]))
    })).SerializeToString()

    # 写入字节流到 tfrecord 文件
    writer.write(record_bytes)

# 定义上下文管理器，用于创建 tfrecord 文件写入器
@contextmanager
def with_tfrecord_writer(path):
    # 设置 TFRecordWriter 的选项，使用 GZIP 压缩
    options = tf.io.TFRecordOptions(compression_type = 'GZIP')

    # 创建 TFRecordWriter 对象
    with tf.io.TFRecordWriter(path, options = options) as writer:
        # 使用 partial 函数创建写入函数的偏函数
        yield partial(write, writer)

# 读取 tfrecords

# 解析 tfrecord 样本的函数
def parse_fn(sample):
    return tf.io.parse_single_example(sample, {
        'seq': tf.io.FixedLenFeature([], tf.string)
    })

# 对批次数据进行整理的函数
def collate_fn(batch, pad_length, offset = 0):
    # 将字节流转换为 numpy 数组
    tensors = [np.frombuffer(el, dtype = np.uint8).astype(np.uint16) for el in batch.numpy()]
    tensors = map(lambda t: t[..., :pad_length], tensors)
    tensors = map(lambda t: t + offset, tensors)
    padded_tensors = map(lambda t: np.pad(t, (0, pad_length - t.shape[-1])), tensors)
    return np.stack(list(padded_tensors))

# 从 tfrecords 文件夹创建迭代器的函数
def iterator_from_tfrecords_folder(folder, data_type = 'train'):
    # 判断是否为 GCS 路径
    is_gcs_path = folder.startswith('gs://')

    # 根据路径获取 tfrecord 文件名列表
    if is_gcs_path:
        filenames = tf.io.gfile.glob(f'{folder}/*.{data_type}.tfrecord.gz')
    else:
        folder = Path(folder)
        filenames = [str(p) for p in folder.glob(f'**/*.{data_type}.tfrecord.gz')]

    # 计算总序列数
    num_seqs = sum(map(lambda t: int(t.split('.')[-4]), filenames))

    # 定义迭代器函数
    def iter_fn(
        seq_len,
        batch_size,
        skip = 0,
        loop = False
    ):
        # 创建 TFRecordDataset 对象
        dataset = tf.data.TFRecordDataset(filenames, compression_type = 'GZIP')

        # 跳过指定数量的样本
        dataset = dataset.skip(skip)
        dataset = dataset.map(parse_fn)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # 如果需要循环迭代，则重复数据集
        if loop:
            dataset = dataset.repeat()

        # 遍历数据集，整理数据并返回
        for batch in dataset:
            seq = batch['seq']
            batch_size = seq.shape[0]
            seq = collate_fn(seq, pad_length = seq_len, offset = 1)
            bos = np.zeros((batch_size, 1), dtype = np.uint16)
            seq = np.concatenate((bos, seq), axis = 1)
            yield seq

    return num_seqs, iter_fn

# 标记化

# 编码单个标记的函数
def encode_token(token):
    return ord(token) + 1

# 解码单个标记的函数
def decode_token(token):
    if token < 0:
        return ''
    return str(chr(token))

# 编码标记序列的函数
def encode_tokens(tokens):
    return list(map(encode_token, tokens))

# 解码标记序列的函数
def decode_tokens(tokens, offset = 1):
    return ''.join(list(map(decode_token, tokens.astype(np.int16) - offset))
```