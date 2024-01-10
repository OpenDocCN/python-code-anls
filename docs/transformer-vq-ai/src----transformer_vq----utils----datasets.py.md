# `transformer_vq\src\transformer_vq\utils\datasets.py`

```
"""
Dataset definitions.
"""
# 导入必要的库
import abc
import functools
from typing import Dict
from typing import Iterable
from typing import Tuple

import jax
import numpy as np
import seqio
import tensorflow as tf

# 导入自定义的模块
from transformer_vq.utils.pipeline import get_batches
from transformer_vq.utils.vocab import maybe_train_sentencepiece_model

# 定义特殊标记的数量
NUM_SPECIAL = 3

# 定义用于将图像展平的函数
def image_flatten(r):
    """for flattening images"""
    return {"targets": tf.cast(tf.reshape(r["targets"], [-1]), tf.int32)}

# 定义用于处理展平后图像的函数
def image_offset(r):
    """
    for use with flattened images.
    this is equivalent to converting the bytes to a string.
    then encoding them as a tensor using seqio.ByteVocab(),
    which introduces special tokens 0, 1, 2 and offsets all valid bytes by 3.
    """
    return {"targets": r["targets"] + NUM_SPECIAL}

# 定义用于添加任务到seqio的函数
def add_to_seqio(task_name, tfds_data_dir, tfds_id, tfds_col, vocab, modality):
    preprocessors = list()
    preprocessors.append(
        functools.partial(seqio.preprocessors.rekey, key_map={"targets": tfds_col})
    )
    if modality == "image":
        preprocessors.append(seqio.utils.map_over_dataset(image_flatten))
        preprocessors.append(seqio.utils.map_over_dataset(image_offset))
    if modality == "text":
        preprocessors.append(seqio.preprocessors.tokenize)
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TfdsDataSource(tfds_id, tfds_data_dir=tfds_data_dir),
        preprocessors=preprocessors,
        output_features={"targets": seqio.Feature(vocabulary=vocab)},
    )

# 定义抽象基类Dataset
class Dataset(metaclass=abc.ABCMeta):
    task_name: str
    modality: str
    registry: Dict = dict()

    def __init__(
        self,
        tfds_data_dir,
        tfds_id,
        tfds_col,
        spm_vocab_path,
        spm_vocab_size,
        spm_uds,
        img_shape=None,
    ):
        # 初始化函数，设置各个属性值
        self.tfds_data_dir = tfds_data_dir
        self.tfds_id = tfds_id
        self.tfds_col = tfds_col
        self.spm_vocab_path = spm_vocab_path
        self.spm_vocab_size = spm_vocab_size
        self.spm_uds = spm_uds
        self.img_shape = img_shape
        self.vocab = self._get_vocab()

    def __init_subclass__(cls, **kwargs):
        """当子类被定义时，将子类添加到注册表中 -- 参见 pep 487"""
        super().__init_subclass__(**kwargs)
        cls.registry[cls.task_name] = cls

    def _get_vocab(self):
        # 如果spm_vocab_size为None，则返回ByteVocabulary对象，否则调用maybe_train_sentencepiece_model函数获取SentencePieceVocabulary对象
        if self.spm_vocab_size is None:
            return seqio.ByteVocabulary()
        else:
            vocab_path = maybe_train_sentencepiece_model(
                tfds_data_dir=self.tfds_data_dir,
                tfds_id=self.tfds_id,
                tfds_col=self.tfds_col,
                spm_vocab_path=self.spm_vocab_path,
                spm_vocab_size=self.spm_vocab_size,
                spm_uds=self.spm_uds,
            )
            return seqio.SentencePieceVocabulary(vocab_path)

    @staticmethod
    def _get_shard_info():
        # 获取当前进程的ID和进程总数，返回ShardInfo对象
        host_id = jax.process_index()
        n_host = jax.process_count()
        return seqio.ShardInfo(index=host_id, num_shards=n_host)

    def _get_tokenized(self, split_name, shard_by_host):
        # 如果任务名称不在TaskRegistry的名称列表中，则将任务添加到TaskRegistry中
        if self.task_name not in seqio.TaskRegistry.names():
            add_to_seqio(
                task_name=self.task_name,
                tfds_data_dir=self.tfds_data_dir,
                tfds_id=self.tfds_id,
                tfds_col=self.tfds_col,
                vocab=self.vocab,
                modality=self.modality,
            )
        # 从TaskRegistry中获取任务对象
        task = seqio.TaskRegistry.get(self.task_name)
        # 如果shard_by_host为True，则获取ShardInfo对象，否则为None
        shard_info = Dataset._get_shard_info() if shard_by_host else None
        # 获取数据集并进行映射，返回处理后的数据集
        ds = task.get_dataset(
            sequence_length=None,
            split=split_name,
            shuffle=False,
            shard_info=shard_info,
        )
        return ds.map(lambda x: tf.cast(x["targets"], tf.int32))
    # 返回词汇表的大小
    @property
    def vocab_size(self):
        return self.vocab.vocab_size
    
    # 解码整数序列为文本
    @staticmethod
    def decode_text(ints: Iterable[int], vocab: seqio.Vocabulary):
        return vocab.decode(ints)
    
    # 解码整数序列为图像
    @staticmethod
    def decode_image(ints: Iterable[int], image_shape: Tuple[int, int, int]):
        # 检查图像形状是否为三元组
        assert isinstance(image_shape, tuple) and len(image_shape) == 3
        # 将特殊标记后的整数转换为图像数据
        ints = [i - NUM_SPECIAL for i in ints if i >= NUM_SPECIAL]
        return np.reshape(np.array(ints, dtype=np.uint8), image_shape)
    
    # 解码整数序列为文本或图像
    def decode(self, ints):
        # 如果是文本类型，则调用decode_text方法
        if self.modality == "text":
            return Dataset.decode_text(ints, self.vocab)
        # 如果是图像类型，则调用decode_image方法
        if self.modality == "image":
            return Dataset.decode_image(ints, self.img_shape)
        # 如果既不是文本也不是图像类型，则抛出未实现错误
        raise NotImplementedError
    
    # 获取数据集迭代器的抽象方法
    @abc.abstractmethod
    def get_iter(self, split_name, batch_size, sequence_len):
        raise NotImplementedError
# 定义 Enwik8 数据集类，继承自 Dataset 类
class Enwik8(Dataset):
    # 设置任务名称为 "enwik8"
    task_name = "enwik8"
    # 设置数据类型为文本
    modality = "text"

    # 初始化方法，接受词汇表路径和数据目录作为参数
    def __init__(self, vocab_path, data_dir):
        # 调用父类的初始化方法
        super().__init__(
            # 设置 TF 数据集目录为数据目录
            tfds_data_dir=data_dir,
            # 设置 TF 数据集 ID
            tfds_id="huggingface:enwik8/enwik8-raw:1.1.0",
            # 设置 TF 数据集列为文本
            tfds_col="text",
            # 设置 SPM 词汇表路径为 None
            spm_vocab_path=None,  # bytes
            # 设置 SPM 词汇表大小为 None
            spm_vocab_size=None,  # bytes
            # 设置 SPM 未知标记列表为空
            spm_uds=[],
        )

    # 获取数据迭代器的方法，接受数据集名称、批量大小和序列长度作为参数
    def get_iter(self, split_name, batch_size, sequence_len):
        # 手动分割数据集，因为 huggingface 数据集将所有 100m enwik8 字节作为训练集
        ds = self._get_tokenized("train", shard_by_host=False)  # never shard; small
        # 获取数据集中的令牌整数列表
        token_ints = list(ds.take(1).as_numpy_iterator())[0]
        # 设置分割索引
        split_indices = [90_000_000, 95_000_000]
        # 设置分割名称
        split_names = ["train", "validation", "test"]
        # 根据分割索引将令牌整数列表分割成不同的数据集
        splits = np.split(token_ints, split_indices)
        # 创建 TF 数据集
        ds = tf.data.Dataset.from_tensors(dict(zip(split_names, splits))[split_name])
        # 返回批量数据
        return get_batches(
            # 数据集映射为整数类型
            ds=ds.map(lambda r: tf.cast(r, tf.int32)),
            # 设置批量大小
            batch_size=batch_size,
            # 设置序列长度
            sequence_len=sequence_len,
            # 判断是否为训练集
            is_train=split_name == "train",
            # 设置词汇表
            vocab=self.vocab,
            # 不添加结束符
            append_eos=False,
        )


# 定义 PG19 数据集类，继承自 Dataset 类
class PG19(Dataset):
    # 设置任务名称为 "pg19"
    task_name = "pg19"
    # 设置数据类型为文本
    modality = "text"

    # 初始化方法，接受词汇表路径和数据目录作为参数
    def __init__(self, vocab_path, data_dir):
        # 调用父类的初始化方法
        super().__init__(
            # 设置 TF 数据集目录为数据目录
            tfds_data_dir=data_dir,
            # 设置 TF 数据集 ID
            tfds_id="pg19:0.1.1",
            # 设置 TF 数据集列为书籍文本
            tfds_col="book_text",
            # 设置 SPM 词汇表路径为给定路径
            spm_vocab_path=vocab_path,
            # 设置 SPM 词汇表大小为 32,000
            spm_vocab_size=32_000,
            # 设置 SPM 未知标记列表为空
            spm_uds=[],
        )

    # 获取数据迭代器的方法，接受数据集名称、批量大小和序列长度作为参数
    def get_iter(self, split_name, batch_size, sequence_len):
        # 返回批量数据
        return get_batches(
            # 获取经过标记化的数据集
            ds=self._get_tokenized(split_name, shard_by_host=split_name == "train"),
            # 设置批量大小
            batch_size=batch_size,
            # 设置序列长度
            sequence_len=sequence_len,
            # 判断是否为训练集
            is_train=split_name == "train",
            # 设置词汇表
            vocab=self.vocab,
            # 添加结束符
            append_eos=True,
        )


# 定义 Imagenet64 数据集类，继承自 Dataset 类
class Imagenet64(Dataset):
    # 设置任务名称为 "imagenet64"
    task_name = "imagenet64"
    # 设置数据类型为图像
    modality = "image"
    # 初始化函数，接受词汇表路径和数据目录作为参数
    def __init__(self, vocab_path, data_dir):
        # 调用父类的初始化函数
        super().__init__(
            tfds_data_dir=data_dir,  # 设置 TensorFlow 数据集的数据目录
            tfds_id="imagenet_resized/64x64:0.1.0",  # 设置 TensorFlow 数据集的 ID
            tfds_col="image",  # 设置 TensorFlow 数据集的列名
            spm_vocab_path=None,  # bytes  # 设置字节流词汇表路径为 None
            spm_vocab_size=None,  # bytes  # 设置字节流词汇表大小为 None
            spm_uds=[],  # 设置字节流 UDS 为空列表
            img_shape=(64, 64, 3),  # 设置图像形状为 (64, 64, 3)
        )

    # 获取数据迭代器的函数，接受数据集名称、批量大小和序列长度作为参数
    def get_iter(self, split_name, batch_size, sequence_len):
        # 如果数据集名称为 "train"
        if split_name == "train":
            # 使用官方训练集的前 1.2m 个示例进行训练
            ds = self._get_tokenized("train", shard_by_host=True)  # 获取经过标记的数据集
            ds = ds.take(1_200_000)  # 从数据集中取出前 1.2m 个示例
            return get_batches(
                ds=ds,  # 数据集
                batch_size=batch_size,  # 批量大小
                sequence_len=sequence_len,  # 序列长度
                is_train=True,  # 是否为训练集
                vocab=self.vocab,  # 词汇表
                append_eos=False,  # 是否添加 EOS 标记
            )
        # 如果数据集名称为 "validation"
        if split_name == "validation":
            # 使用训练集中剩余的示例进行验证
            ds = self._get_tokenized("train", shard_by_host=False)  # 获取经过标记的数据集
            ds = ds.skip(1_200_000)  # 跳过前 1.2m 个示例
            return get_batches(
                ds=ds,  # 数据集
                batch_size=batch_size,  # 批量大小
                sequence_len=sequence_len,  # 序列长度
                is_train=False,  # 是否为训练集
                vocab=self.vocab,  # 词汇表
                append_eos=False,  # 是否添加 EOS 标记
            )
        # 如果数据集名称为 "test"
        if split_name == "test":
            # 使用官方验证集进行基准测试；imagenet 没有公共测试集
            ds = self._get_tokenized("validation", shard_by_host=False)  # 获取经过标记的数据集
            return get_batches(
                ds=ds,  # 数据集
                batch_size=batch_size,  # 批量大小
                sequence_len=sequence_len,  # 序列长度
                is_train=False,  # 是否为训练集
                vocab=self.vocab,  # 词汇表
                append_eos=False,  # 是否添加 EOS 标记
            )
        # 如果数据集名称不在 "train", "validation", "test" 中
        raise NotImplementedError  # 抛出未实现的错误
```