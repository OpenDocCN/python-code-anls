# `transformer_vq\src\transformer_vq\utils\vocab.py`

```
# 导入所需的模块
import os
import tempfile
import time
from typing import List
from typing import Optional

import jax
import sentencepiece as spm
import tensorflow as tf
import tensorflow_datasets as tfds

# 将数据集中的字符写入临时文件
def dump_chars_to_tempfile(ds, maxchars):
    char_count = 0
    # 创建一个临时文件，用于存储字符数据
    with tempfile.NamedTemporaryFile(delete=False, prefix="/tmp/ds_chars") as outfp:
        # 遍历数据集中的字符
        for document_chars in ds:
            # 如果指定了最大字符数，并且已经达到最大字符数，则停止遍历
            if (maxchars is not None) and (char_count >= maxchars):
                break
            # 将字符写入临时文件，并在每个字符后添加换行符
            outfp.write(document_chars + b"\n")
            # 更新字符计数
            char_count += len(document_chars)
# 返回输出文件名和字符计数
def maybe_train_sentencepiece_model(
    tfds_data_dir: Optional[str],  # TFDS 数据目录
    tfds_id: str,  # TFDS ID
    tfds_col: str,  # TFDS 列名
    spm_vocab_path: str,  # SentencePiece 词汇表路径
    spm_vocab_size: int,  # SentencePiece 词汇表大小
    spm_uds: List[str],  # SentencePiece 未定义符号列表
    maxchars: Optional[int] = int(1e9),  # 最大字符数，默认为 10^9
):
    # 如果 SentencePiece 词汇表路径以 "gs://" 开头，则使用绝对路径
    if spm_vocab_path.startswith("gs://"):
        abs_spm_vocab_path = spm_vocab_path
    else:
        # 否则，获取绝对路径并创建目录
        abs_spm_vocab_path = os.path.abspath(os.path.expanduser(spm_vocab_path))
        os.makedirs(os.path.dirname(abs_spm_vocab_path), exist_ok=True)
    # 如果 SentencePiece 词汇表路径存在，则返回该路径
    if tf.io.gfile.exists(abs_spm_vocab_path):
        return abs_spm_vocab_path
    # 如果是 JAX 进程的第一个进程
    if jax.process_index() == 0:
# 加载数据集，并将数据集中的指定列映射为numpy迭代器
chardump_ds = (
    tfds.load(tfds_id, split="train", data_dir=tfds_data_dir, try_gcs=True)
    .map(lambda r: r[tfds_col])
    .as_numpy_iterator()
)
# 将字符转储到临时文件中，并获取文件名
fname, _ = dump_chars_to_tempfile(ds=chardump_ds, maxchars)
# 使用临时文件创建一个命名的临时文件，用于保存训练后的模型
with tempfile.NamedTemporaryFile(delete=False, prefix="/tmp/sp_tmp") as temp_fp:
    # 为了与seqio.ByteVocabulary保持一致，使用0=PAD, 1=EOS, 2=UNK
    # 使用SentencePieceTrainer训练模型
    spm.SentencePieceTrainer.Train(
        input=fname,
        vocab_size=spm_vocab_size,
        character_coverage=1.0,
        model_prefix=temp_fp.name,
        model_type="bpe",
        user_defined_symbols=spm_uds,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
    )
# 创建一个新的文件路径，用于复制和重命名文件
copy_rename_path = abs_spm_vocab_path + ".rntmp"
# 打印临时文件名加上".model"的字符串
print(temp_fp.name + ".model")
# 使用tf.io.gfile.copy()函数复制临时文件到新路径，如果新路径已存在则覆盖
tf.io.gfile.copy(temp_fp.name + ".model", copy_rename_path, overwrite=True)
# 使用tf.io.gfile.rename()函数将复制的文件重命名为目标文件路径，如果目标文件路径已存在则覆盖
tf.io.gfile.rename(copy_rename_path, abs_spm_vocab_path, overwrite=True)
# 如果目标文件路径不存在，则进入循环等待，直到文件存在为止
else:
    while not tf.io.gfile.exists(abs_spm_vocab_path):
        time.sleep(1)
    # 等待1秒
    time.sleep(1)
# 返回目标文件路径
return abs_spm_vocab_path
```