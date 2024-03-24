# `.\lucidrains\enformer-tensorflow-sonnet-training-script\create_tfrecords.py`

```
# 导入所需的模块
from itertools import islice
from functools import partial
import tensorflow as tf

# 旧的 get_dataset 函数，但只返回标签以便在新的更长序列中进行压缩
def organism_path(organism):
  return os.path.join(f'gs://basenji_barnyard/data', organism)

# 获取数据集
def get_dataset(organism, subset, num_threads=8):
  # 获取元数据
  metadata = get_metadata(organism)
  # 获取 TFRecord 文件
  files = tfrecord_files(organism, subset)
  # 创建 TFRecord 数据集
  dataset = tf.data.TFRecordDataset(files, compression_type='ZLIB', num_parallel_reads=None)
  
  # 映射数据集
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata), num_parallel_calls=num_threads)
  return dataset

# 获取元数据
def get_metadata(organism):
  path = os.path.join(organism_path(organism), 'statistics.json')
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)

# 获取 TFRecord 文件
def tfrecord_files(organism, subset):
  return sorted(tf.io.gfile.glob(os.path.join(organism_path(organism), 'tfrecords', f'{subset}-*.tfr')), key=lambda x: int(x.split('-')[-1].split('.')[0]))

# 反序列化
def deserialize(serialized_example, metadata):
  feature_map = {
    'sequence': tf.io.FixedLenFeature([], tf.string),
    'target': tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_example(serialized_example, feature_map)
  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target, (metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)
  return target

# 分块函数
def chunk(it, size):
  it = iter(it)
  return iter(lambda: tuple(islice(it, size)), ())

# 创建 float 特征
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# 解析单个示例
def parse_single_example(seq, target):
  seq = seq.numpy()
  target = target.numpy()

  data = {
      'seq' : _float_feature(seq.flatten()),
      'target' : _float_feature(target.flatten()),
  }

  out = tf.train.Example(features=tf.train.Features(feature=data))
  return out

# 物种配置
NUM_TRACKS_CONFIG = dict(human = 5313, mouse = 1643)

# 映射序列和目标
def map_seq_target(
  element,
  seq_len,
  species,  # 'human' or 'mouse'
  shifts = None
):
  assert species in NUM_TRACKS_CONFIG, f'{species} not found in config'
  num_tracks = NUM_TRACKS_CONFIG[species]

  num_shifts = 0 if shifts is None else len(list(range(shifts[0], shifts[1] + 1)))

  data = {
    'seq':tf.io.FixedLenFeature([(seq_len + num_shifts) * 4], tf.float32),
    'target':tf.io.FixedLenFeature([896 * num_tracks], tf.float32),
  }
  
  content = tf.io.parse_single_example(element, data)
  return content

# 创建 TFRecord
def create_tfrecords(ds, path = './', chunk_size = 256):
  for ind, batch in enumerate(chunk(iter(ds), chunk_size)):
    writer = tf.io.TFRecordWriter(f'{path}{ind}.tfrecord', 'ZLIB')

    for seq, target in batch:
      features = parse_single_example(seq, target)
      writer.write(features.SerializeToString())

    writer.close()

if __name__ == '__main__':

  # 写入示例
  generator_fn = get_dna_sample(
    bed_file = './human-sequences.bed',
    fasta_file = './hg38.ml.fa',
    filter_type = 'train',
    context_length = 196_608
  )

  seq_ds = tf.data.Dataset.from_generator(generator_fn, tf.float32)
  label_ds = get_dataset('human', 'train')

  zipped_ds = tf.data.Dataset.zip((seq_ds, label_ds))
  create_tfrecords(zipped_ds, 'gs://enformer-new-data-path/')

  # 读取
  dataset = tf.data.TFRecordDataset(['./0.tfrecord', './1.tfrecord'], compression_type = 'ZLIB')
  map_element_fn = partial(map_seq_target, seq_len = 196608, species = 'human', shifts = (-2, 2))
  dataset = dataset.map(map_element_fn)
```