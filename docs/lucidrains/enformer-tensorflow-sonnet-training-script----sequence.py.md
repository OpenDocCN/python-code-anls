# `.\lucidrains\enformer-tensorflow-sonnet-training-script\sequence.py`

```
# 导入所需的库
import tensorflow as tf
import numpy as np
import pandas as pd
from pyfaidx import Fasta

from functools import partial
from random import randrange

# 创建一个用于存储 DNA 序列的独热编码的嵌入表
# 基于 https://gist.github.com/hannes-brt/54ca5d4094b3d96237fa2e820c0945dd 进行修改
embed = np.zeros([89, 4], np.float32)
embed[ord('A')] = np.array([1, 0, 0, 0])
embed[ord('C')] = np.array([0, 1, 0, 0])
embed[ord('G')] = np.array([0, 0, 1, 0])
embed[ord('T')] = np.array([0, 0, 0, 1])
embed[ord('a')] = np.array([1, 0, 0, 0])
embed[ord('c')] = np.array([0, 1, 0, 0])
embed[ord('g')] = np.array([0, 0, 1, 0])
embed[ord('t')] = np.array([0, 0, 0, 1])
embed[ord('.')] = np.array([.25, .25, .25, .25])

# 将嵌入表转换为 TensorFlow 张量
embedding_table = tf.convert_to_tensor(embed)

# 定义一个函数，将 DNA 序列进行独热编码
def one_hot_encode_seq(dna_input, embed, name = "encode_seq"):
  with tf.name_scope(name):
    # 将 DNA 序列转换为字节流
    b = bytearray()
    b.extend(map(ord, str(dna_input)))
    t = tf.convert_to_tensor(b)
    t = tf.cast(t, tf.int32)
    # 使用嵌入表进行独热编码
    encoded_dna = tf.nn.embedding_lookup(embedding_table, t)

  return encoded_dna

# 根据 fasta 文件和 pyfaidx 获取更长的上下文
def get_datum(
  ind,
  fasta_ref,
  bed_df,
  context_length = None,
  rand_shift_range = None
):
  # 从 bed 数据框中获取行信息
  row = bed_df.iloc[ind]
  chrname, start, end, t = bed_df.iloc[ind].tolist()
  interval_length = end - start

  chromosome = fasta_ref[chrname]
  chromosome_length = len(chromosome)

  if rand_shift_range is not None:
    min_shift, max_shift = rand_shift_range

    adj_min_shift = max(start + min_shift, 0) - start
    adj_max_shift = min(end + max_shift, chromosome_length) - end

    left_padding = adj_min_shift - min_shift
    right_padding = max_shift - adj_max_shift

    start += adj_min_shift
    end += adj_max_shift

  if context_length is None or context_length <= interval_length:
    seq = chromosome[start:end]
    return one_hot_encode_seq(seq, embed)

  left_padding = right_padding = 0
  
  extra_seq = context_length - interval_length

  extra_left_seq = extra_seq // 2
  extra_right_seq = extra_seq - extra_left_seq

  start -= extra_left_seq
  end += extra_right_seq

  if start < 0:
    left_padding = -start
    start = 0

  if end > chromosome_length:
    right_padding = end - chromosome_length
    end = chromosome_length

  seq = ('.' * left_padding) + str(chromosome[start:end]) + ('.' * right_padding)
  return one_hot_encode_seq(seq, embed)

# 获取 DNA 样本数据
def get_dna_sample(
  bed_file,
  fasta_file,
  filter_type = None,
  context_length = None,
  rand_shift_range = (-2, 2)
):
  # 从 bed 文件中读取数据
  df = pd.read_csv(bed_file, sep = '\t', header = None)

  if filter_type is not None:
    df = df[df[3] == filter_type]

  # 读取 fasta 文件
  fasta = Fasta(fasta_file, sequence_always_upper = True)
  yield_data_fn = partial(get_datum, fasta_ref = fasta, bed_df = df, context_length = context_length, rand_shift_range = rand_shift_range)

  def inner():
    for ind in range(len(df)):
      yield yield_data_fn(ind)

  return inner

# 主函数
if __name__ == '__main__':

  # 获取 DNA 样本数据生成器
  generator_fn = get_dna_sample(
    bed_file = './human-sequences.bed',
    fasta_file = './hg38.ml.fa',
    filter_type = 'valid',
    context_length = 196_608
  )

  # 创建 TensorFlow 数据集
  dataset = tf.data.Dataset.from_generator(generator_fn, tf.float32)
  # 打印数据集中第一个元素的形状
  print(next(iter(dataset)).shape)
```