# `transformer_vq\src\transformer_vq\utils\vocab.py`

```
# 导入所需的库
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
    # 创建一个临时文件，用于存储字符
    with tempfile.NamedTemporaryFile(delete=False, prefix="/tmp/ds_chars") as outfp:
        # 遍历数据集中的字符
        for document_chars in ds:
            # 如果指定了最大字符数，并且已经达到最大字符数，则退出循环
            if (maxchars is not None) and (char_count >= maxchars):
                break
            # 将字符写入临时文件，并在末尾添加换行符
            outfp.write(document_chars + b"\n")
            char_count += len(document_chars)
        # 返回临时文件的路径和字符数
        return outfp.name, char_count

# 可能训练 sentencepiece 模型
def maybe_train_sentencepiece_model(
    tfds_data_dir: Optional[str],
    tfds_id: str,
    tfds_col: str,
    spm_vocab_path: str,
    spm_vocab_size: int,
    spm_uds: List[str],
    maxchars: Optional[int] = int(1e9),
):
    # 如果 spm_vocab_path 是以 "gs://" 开头的，则直接使用该路径
    if spm_vocab_path.startswith("gs://"):
        abs_spm_vocab_path = spm_vocab_path
    else:
        # 否则，获取绝对路径，并创建目录（如果不存在）
        abs_spm_vocab_path = os.path.abspath(os.path.expanduser(spm_vocab_path))
        os.makedirs(os.path.dirname(abs_spm_vocab_path), exist_ok=True)
    # 如果指定的 spm_vocab_path 已经存在，则直接返回该路径
    if tf.io.gfile.exists(abs_spm_vocab_path):
        return abs_spm_vocab_path
    # 检查当前进程是否为第一个进程
    if jax.process_index() == 0:
        # 加载指定的 TF 数据集，并将数据集中的指定列映射为 NumPy 迭代器
        chardump_ds = (
            tfds.load(tfds_id, split="train", data_dir=tfds_data_dir, try_gcs=True)
            .map(lambda r: r[tfds_col])
            .as_numpy_iterator()
        )
        # 将字符转储到临时文件中，并返回文件名和文件对象
        fname, _ = dump_chars_to_tempfile(ds=chardump_ds, maxchars)
        # 使用命名的临时文件创建 SentencePiece 模型
        with tempfile.NamedTemporaryFile(delete=False, prefix="/tmp/sp_tmp") as temp_fp:
            # 为了与 seqio.ByteVocabulary 保持一致，使用 0=PAD, 1=EOS, 2=UNK
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
            # 复制并重命名临时文件中的模型文件
            copy_rename_path = abs_spm_vocab_path + ".rntmp"
            print(temp_fp.name + ".model")
            tf.io.gfile.copy(temp_fp.name + ".model", copy_rename_path, overwrite=True)
            tf.io.gfile.rename(copy_rename_path, abs_spm_vocab_path, overwrite=True)
    else:
        # 如果当前进程不是第一个进程，则等待直到生成的 SentencePiece 模型文件存在
        while not tf.io.gfile.exists(abs_spm_vocab_path):
            time.sleep(1)
        # 等待一秒钟
        time.sleep(1)
    # 返回生成的 SentencePiece 模型文件的路径
    return abs_spm_vocab_path
```