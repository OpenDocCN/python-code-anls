# `.\lucidrains\progen\generate_data.py`

```
# 导入所需的库
import os
import gzip
import click
import re
import random
from math import ceil
from functools import partial
from itertools import islice, chain
from operator import itemgetter

from pyfaidx import Faidx

import numpy as np
from random import random
from pathlib import Path

import toml
from google.cloud import storage

from prefect import Parameter, task, Flow

from progen_transformer.data import with_tfrecord_writer
from progen_transformer.utils import clear_directory_

# 常量定义
GCS_WRITE_TIMEOUT = 60 * 30
TMP_DIR = Path('./.tmp')

# 定义函数

# 根据给定的排序函数对字典进行排序
def order_dict_by(d, fn):
    keys = fn(d.keys())
    return dict(tuple(map(lambda k: (k, d[k]), keys)))

# 从描述中提取注释信息
def get_annotations_from_description(config, description):
    taxonomy_matches = re.findall(r'Tax=([a-zA-Z\s]*)\s[a-zA-Z\=]', description)
    annotations = dict()

    if len(taxonomy_matches) > 0:
        annotations['tax'] = taxonomy_matches[0]

    return annotations

# 将 fasta 行转换为序列字符串
def fasta_row_to_sequence_strings(config, fa, uid):
    seq_len = fa.index[uid].rlen
    seq = str(fa.fetch(uid, 1, seq_len))
    description = fa.get_long_name(uid)

    sequences = []
    annotations = get_annotations_from_description(config, description)
    # todo: gather annotations from GO

    if len(annotations) > 0:
        sort_annot_by = random.shuffle if not config['sort_annotations'] else sorted
        annotations = order_dict_by(annotations, sort_annot_by)

        annotation_str = [f"[{annot_name}={annot}]" for annot_name, annot in annotations.items()]
        annotation_str = ' '.join(annotation_str)

        seq_annot_pair = (annotation_str, seq)

        if random() <= config['prob_invert_seq_annotation']:
            seq_annot_pair = tuple(reversed(seq_annot_pair))

        sequence = ' # '.join(seq_annot_pair)
        sequence = sequence.encode('utf-8')
        sequences.append(sequence)

    sequence = f'# {seq}'
    sequence = sequence.encode('utf-8')
    sequences.append(sequence)

    return sequences

# 处理并写入临时文件
def process_and_write_to_tmp_file(i, seq_str):
    filename = TMP_DIR / str(i)
    with gzip.open(str(filename), 'wb') as f:
        f.write(seq_str)

# 对每个元素应用函数
def foreach(fn, it):
    for el in it:
        fn(*el)

# DAG 函数

# 将 fasta 文件转换为临时文件
@task
def fasta_to_tmp_files(config):
    clear_directory_(TMP_DIR)

    print('reading from fasta')
    fa = Faidx(config['read_from'], sequence_always_upper = True)

    print('filtering by length')
    it = iter(fa.index.items())
    it = filter(lambda el: el[1].rlen <= config['max_seq_len'], it)

    print('parallel processing to tmp files')
    it = islice(it, 0, config['num_samples'])
    it = map(itemgetter(0), it)

    fasta_to_seq_fn = partial(fasta_row_to_sequence_strings, config, fa)
    it = map(fasta_to_seq_fn, it)
    it = enumerate(chain.from_iterable(it))
    foreach(process_and_write_to_tmp_file, it)

# 将文件转换为 tfrecords
@task
def files_to_tfrecords(config):
    filenames = [*TMP_DIR.glob('**/*')]
    num_samples = len(filenames)
    num_valids = ceil(config['fraction_valid_data'] * num_samples)

    num_sequences_per_file = config['num_sequences_per_file']

    # 分割出验证序列

    permuted_sequences = np.random.permutation(num_samples)
    valid_seqs, train_seqs = np.split(permuted_sequences, [num_valids])

    # 清空写入目录

    write_to = config['write_to']
    upload_gcs = write_to.startswith('gs://')

    if upload_gcs:
        write_to = write_to[5:]
        client = storage.Client()
        bucket_name = write_to

        bucket = client.get_bucket(bucket_name)
        bucket.delete_blobs(list(bucket.list_blobs()))

    write_to_path = Path(write_to)
    clear_directory_(write_to_path)

    # 循环并将所有训练和验证文件写入 tfrecords
    # 遍历训练集和验证集，每个元组包含序列类型和序列数据
    for (seq_type, seqs) in (('train', train_seqs), ('valid', valid_seqs)):
        # 计算需要拆分的文件数量
        num_split = ceil(seqs.shape[0] / num_sequences_per_file)
        # 对序列数据进行拆分，每个文件包含 num_sequences_per_file 个序列
        for file_index, indices in enumerate(np.array_split(seqs, num_split)):
            # 获取当前文件中序列的数量
            num_sequences = len(indices)
            # 构建 TFRecord 文件名，包含文件索引、序列数量和序列类型
            tfrecord_filename = f'{file_index}.{num_sequences}.{seq_type}.tfrecord.gz'
            # 构建 TFRecord 文件路径
            tfrecord_path = str(write_to_path / tfrecord_filename)

            # 使用 TFRecord 写入器打开文件，写入序列数据
            with with_tfrecord_writer(tfrecord_path) as write:
                # 遍历当前文件中的序列索引
                for index in indices:
                    # 获取当前序列对应的文件名
                    filename = filenames[index]
                    # 使用 gzip 打开文件，读取数据并写入 TFRecord 文件
                    with gzip.open(filename, 'rb') as f:
                        write(f.read())

            # 如果需要上传到 Google Cloud Storage
            if upload_gcs:
                # 创建一个存储桶对象
                blob = bucket.blob(tfrecord_filename)
                # 从本地文件上传 TFRecord 文件到存储桶，设置超时时间
                blob.upload_from_filename(tfrecord_path, timeout = GCS_WRITE_TIMEOUT)
# 创建一个名为'parse-fasta'的Flow对象
with Flow('parse-fasta') as flow:
    # 创建一个名为'config'的参数，必须提供数值
    config = Parameter('config', required = True)
    # 调用fasta_to_tmp_files函数，传入config参数
    fasta_to_tmp_files(config = config)
    # 调用files_to_tfrecords函数，传入config参数

@click.command()
# 添加一个名为'data_dir'的命令行选项，默认值为'./configs/data'
@click.option('--data_dir', default = './configs/data')
# 添加一个名为'name'的命令行选项，默认值为'default'
@click.option('--name', default = 'default')
def main(
    data_dir,
    name
):
    # 将data_dir转换为Path对象
    data_dir = Path(data_dir)
    # 构建配置文件路径
    config_path = data_dir / f'{name}.toml'
    # 断言配置文件路径存在
    assert config_path.exists(), f'config does not exist at {str(config_path)}'

    # 读取配置文件内容并解析为字典
    config = toml.loads(config_path.read_text())
    # 运行Flow对象，传入config参数
    flow.run(config = config)

# 如果当前脚本作为主程序运行，则执行main函数
if __name__ == '__main__':
    main()
```