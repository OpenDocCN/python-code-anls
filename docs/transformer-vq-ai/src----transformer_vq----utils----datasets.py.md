# `transformer_vq\src\transformer_vq\utils\datasets.py`

```
# 导入必要的库
import abc  # 导入抽象基类模块
import functools  # 导入函数工具模块
from typing import Dict  # 导入类型提示模块，用于声明字典类型
from typing import Iterable  # 导入类型提示模块，用于声明可迭代对象类型
from typing import Tuple  # 导入类型提示模块，用于声明元组类型

import jax  # 导入 JAX 库
import numpy as np  # 导入 NumPy 库
import seqio  # 导入 seqio 库
import tensorflow as tf  # 导入 TensorFlow 库

from transformer_vq.utils.pipeline import get_batches  # 从自定义模块中导入 get_batches 函数
from transformer_vq.utils.vocab import maybe_train_sentencepiece_model  # 从自定义模块中导入 maybe_train_sentencepiece_model 函数

# 定义特殊标记的数量
NUM_SPECIAL = 3
# 定义一个函数用于将图像展平
def image_flatten(r):
    """for flattening images"""
    # 将图像展平并转换为整数类型
    return {"targets": tf.cast(tf.reshape(r["targets"], [-1]), tf.int32)}


# 定义一个函数用于处理展平后的图像
def image_offset(r):
    """
    for use with flattened images.
    this is equivalent to converting the bytes to a string.
    then encoding them as a tensor using seqio.ByteVocab(),
    which introduces special tokens 0, 1, 2 and offsets all valid bytes by 3.
    """
    # 对展平后的图像进行处理，添加特殊标记并偏移所有有效字节
    return {"targets": r["targets"] + NUM_SPECIAL}


# 定义一个函数用于将数据添加到seqio任务中
def add_to_seqio(task_name, tfds_data_dir, tfds_id, tfds_col, vocab, modality):
    preprocessors = list()
    preprocessors.append(
# 使用 functools.partial 函数创建一个新的函数，用于重新映射数据的键名
functools.partial(seqio.preprocessors.rekey, key_map={"targets": tfds_col})
# 如果数据类型是图像，添加图像预处理函数
if modality == "image":
    preprocessors.append(seqio.utils.map_over_dataset(image_flatten))
    preprocessors.append(seqio.utils.map_over_dataset(image_offset))
# 如果数据类型是文本，添加文本预处理函数
if modality == "text":
    preprocessors.append(seqio.preprocessors.tokenize)
# 将任务添加到任务注册表中，包括数据源、预处理函数和输出特征
seqio.TaskRegistry.add(
    task_name,
    source=seqio.TfdsDataSource(tfds_id, tfds_data_dir=tfds_data_dir),
    preprocessors=preprocessors,
    output_features={"targets": seqio.Feature(vocabulary=vocab)},
)

# 定义一个抽象基类 Dataset
class Dataset(metaclass=abc.ABCMeta):
    task_name: str
    modality: str
    registry: Dict = dict()
# 初始化函数，用于设置对象的属性
def __init__(
    self,
    tfds_data_dir,  # TFDS 数据目录
    tfds_id,  # TFDS ID
    tfds_col,  # TFDS 列
    spm_vocab_path,  # SPM 词汇表路径
    spm_vocab_size,  # SPM 词汇表大小
    spm_uds,  # SPM UDS
    img_shape=None,  # 图像形状，默认为 None
):
    # 设置对象属性
    self.tfds_data_dir = tfds_data_dir
    self.tfds_id = tfds_id
    self.tfds_col = tfds_col
    self.spm_vocab_path = spm_vocab_path
    self.spm_vocab_size = spm_vocab_size
    self.spm_uds = spm_uds
    self.img_shape = img_shape
    # 获取词汇表
    self.vocab = self._get_vocab()

# 初始化子类的函数
def __init_subclass__(cls, **kwargs):
    # 当子类被定义时，根据PEP 487将子类添加到注册表中
    """adds subclasses to registry when they are defined -- see pep 487"""
    super().__init_subclass__(**kwargs)
    # 将子类添加到注册表中，使用任务名称作为键
    cls.registry[cls.task_name] = cls

    # 获取词汇表
    def _get_vocab(self):
        # 如果spm_vocab_size为None，则返回ByteVocabulary对象
        if self.spm_vocab_size is None:
            return seqio.ByteVocabulary()
        else:
            # 否则，根据参数获取SentencePiece词汇表路径，并返回SentencePieceVocabulary对象
            vocab_path = maybe_train_sentencepiece_model(
                tfds_data_dir=self.tfds_data_dir,
                tfds_id=self.tfds_id,
                tfds_col=self.tfds_col,
                spm_vocab_path=self.spm_vocab_path,
                spm_vocab_size=self.spm_vocab_size,
                spm_uds=self.spm_uds,
            )
            return seqio.SentencePieceVocabulary(vocab_path)

    # 获取分片信息的静态方法
    @staticmethod
    def _get_shard_info():
# 获取当前主机的 ID
host_id = jax.process_index()
# 获取主机数量
n_host = jax.process_count()
# 返回主机的分片信息
return seqio.ShardInfo(index=host_id, num_shards=n_host)

# 获取分词后的数据
def _get_tokenized(self, split_name, shard_by_host):
    # 如果任务名称不在任务注册表中，则添加到任务注册表中
    if self.task_name not in seqio.TaskRegistry.names():
        add_to_seqio(
            task_name=self.task_name,
            tfds_data_dir=self.tfds_data_dir,
            tfds_id=self.tfds_id,
            tfds_col=self.tfds_col,
            vocab=self.vocab,
            modality=self.modality,
        )
    # 从任务注册表中获取任务
    task = seqio.TaskRegistry.get(self.task_name)
    # 如果按主机分片，则获取分片信息，否则为 None
    shard_info = Dataset._get_shard_info() if shard_by_host else None
    # 获取数据集
    ds = task.get_dataset(
        sequence_length=None,
        split=split_name,
        shuffle=False,
    # 返回数据集的目标值
    return ds.map(lambda x: tf.cast(x["targets"], tf.int32))

    # 返回词汇表的大小
    @property
    def vocab_size(self):
        return self.vocab.vocab_size

    # 将整数序列解码为文本
    @staticmethod
    def decode_text(ints: Iterable[int], vocab: seqio.Vocabulary):
        return vocab.decode(ints)

    # 将整数序列解码为图像
    @staticmethod
    def decode_image(ints: Iterable[int], image_shape: Tuple[int, int, int]):
        assert isinstance(image_shape, tuple) and len(image_shape) == 3
        # 将特殊标记去除后的整数序列转换为图像数组
        ints = [i - NUM_SPECIAL for i in ints if i >= NUM_SPECIAL]
        return np.reshape(np.array(ints, dtype=np.uint8), image_shape)

    # 解码整数序列
    def decode(self, ints):
        # 如果数据类型是文本
        if self.modality == "text":
        # 如果数据类型是文本，则调用decode_text方法解码数据
        return Dataset.decode_text(ints, self.vocab)
        # 如果数据类型是图像，则调用decode_image方法解码数据
        if self.modality == "image":
            return Dataset.decode_image(ints, self.img_shape)
        # 如果数据类型不是文本或图像，则抛出NotImplementedError
        raise NotImplementedError

    @abc.abstractmethod
    # 获取数据迭代器的抽象方法
    def get_iter(self, split_name, batch_size, sequence_len):
        raise NotImplementedError


class Enwik8(Dataset):
    # 数据集名称为enwik8
    task_name = "enwik8"
    # 数据类型为文本
    modality = "text"

    def __init__(self, vocab_path, data_dir):
        # 调用父类的初始化方法
        super().__init__(
            tfds_data_dir=data_dir,
            tfds_id="huggingface:enwik8/enwik8-raw:1.1.0",
            tfds_col="text",
            spm_vocab_path=None,  # bytes
# 定义一个函数，用于获取数据迭代器
def get_iter(self, split_name, batch_size, sequence_len):
    # 手动划分数据集，因为huggingface数据集将所有100m enwik8字节作为训练集
    # 获取经过标记的数据集，不进行主机分片，因为数据集较小
    ds = self._get_tokenized("train", shard_by_host=False)
    # 将数据集中的第一个元素转换为numpy迭代器，并转换为列表
    token_ints = list(ds.take(1).as_numpy_iterator())[0]
    # 划分数据集的索引
    split_indices = [90_000_000, 95_000_000]
    # 划分的数据集名称
    split_names = ["train", "validation", "test"]
    # 根据划分索引将数据集划分为不同的部分
    splits = np.split(token_ints, split_indices)
    # 从划分后的数据集中获取指定名称的数据集
    ds = tf.data.Dataset.from_tensors(dict(zip(split_names, splits))[split_name])
    # 返回批次数据
    return get_batches(
        ds=ds.map(lambda r: tf.cast(r, tf.int32)),  # 将数据集中的元素转换为tf.int32类型
        batch_size=batch_size,  # 批次大小
        sequence_len=sequence_len,  # 序列长度
        is_train=split_name == "train",  # 是否为训练集
        vocab=self.vocab,  # 词汇表
        append_eos=False,  # 是否添加结束符
    )
# 定义一个名为PG19的类，继承自Dataset类
class PG19(Dataset):
    # 定义任务名称为"pg19"
    task_name = "pg19"
    # 定义数据类型为"text"
    modality = "text"

    # 初始化方法，接收词汇表路径和数据目录作为参数
    def __init__(self, vocab_path, data_dir):
        # 调用父类的初始化方法
        super().__init__(
            # 设置tfds_data_dir为数据目录
            tfds_data_dir=data_dir,
            # 设置tfds_id为"pg19:0.1.1"
            tfds_id="pg19:0.1.1",
            # 设置tfds_col为"book_text"
            tfds_col="book_text",
            # 设置spm_vocab_path为词汇表路径
            spm_vocab_path=vocab_path,
            # 设置spm_vocab_size为32,000
            spm_vocab_size=32_000,
            # 设置spm_uds为空列表
            spm_uds=[],
        )

    # 获取迭代器的方法，接收数据集名称、批量大小和序列长度作为参数
    def get_iter(self, split_name, batch_size, sequence_len):
        # 调用get_batches函数，传入tokenized数据集、批量大小和是否按主机分片的标志
        return get_batches(
            ds=self._get_tokenized(split_name, shard_by_host=split_name == "train"),
            batch_size=batch_size,
# 设置序列长度
sequence_len=sequence_len,
# 判断是否为训练集，根据split_name参数判断
is_train=split_name == "train",
# 词汇表路径
vocab=self.vocab,
# 是否在序列末尾添加结束符
append_eos=True,
)

# 定义一个名为Imagenet64的数据集类
class Imagenet64(Dataset):
    # 任务名称
    task_name = "imagenet64"
    # 数据类型为图像
    modality = "image"

    # 初始化函数
    def __init__(self, vocab_path, data_dir):
        # 调用父类的初始化函数
        super().__init__(
            # 数据集目录
            tfds_data_dir=data_dir,
            # 数据集ID
            tfds_id="imagenet_resized/64x64:0.1.0",
            # 数据集列名
            tfds_col="image",
            # 字节类型的词汇表路径
            spm_vocab_path=None,
            # 字节类型的词汇表大小
            spm_vocab_size=None,
            # 未知数据集
            spm_uds=[],
            # 图像形状
            img_shape=(64, 64, 3),
        )

    def get_iter(self, split_name, batch_size, sequence_len):
        if split_name == "train":
            # 如果是训练集，使用官方训练集的前120万个示例进行训练
            ds = self._get_tokenized("train", shard_by_host=True)
            ds = ds.take(1_200_000)
            return get_batches(
                ds=ds,
                batch_size=batch_size,
                sequence_len=sequence_len,
                is_train=True,
                vocab=self.vocab,
                append_eos=False,
            )
        if split_name == "validation":
            # 如果是验证集，使用训练集中剩余的示例进行验证
            ds = self._get_tokenized("train", shard_by_host=False)
            ds = ds.skip(1_200_000)
            return get_batches(
# 设置数据集、批处理大小、序列长度、训练标志、词汇表和是否追加结束符号
ds=ds,
batch_size=batch_size,
sequence_len=sequence_len,
is_train=False,
vocab=self.vocab,
append_eos=False,

# 如果是测试集，使用官方验证集进行基准测试；ImageNet 没有公开的测试集
if split_name == "test":
    # 获取经过标记的验证集，不按主机分片
    ds = self._get_tokenized("validation", shard_by_host=False)
    # 返回批处理
    return get_batches(
        ds=ds,
        batch_size=batch_size,
        sequence_len=sequence_len,
        is_train=False,
        vocab=self.vocab,
        append_eos=False,
    )

# 如果不是测试集，抛出未实现错误
raise NotImplementedError
```