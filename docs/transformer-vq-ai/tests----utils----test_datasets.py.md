# `transformer_vq\tests\utils\test_datasets.py`

```
# 导入所需的库
import numpy as np
import pytest
import seqio
import tensorflow as tf

# 导入自定义的数据集相关模块
from transformer_vq.utils.datasets import Dataset
from transformer_vq.utils.datasets import image_flatten
from transformer_vq.utils.datasets import image_offset
from tests.utils.fixtures import img

# 测试字节词汇表的特殊符号
def test_byte_vocab_specials():
    # 创建字节词汇表对象
    vocab = seqio.ByteVocabulary()
    # 断言特殊符号的索引值
    assert vocab.pad_id == 0
    assert vocab.eos_id == 1
    assert vocab.unk_id == 2
    # 测试未实现的特殊符号
    with pytest.raises(NotImplementedError):
        print(f"byte-level bos_id: {vocab.bos_id}")

# 测试图像处理流水线
def test_img_pipeline(img):
    # 编码图像
    x = {"targets": tf.constant(img, dtype=tf.uint8)}
    x = image_flatten(x)
    x = image_offset(x)["targets"]
    # 解码图像
    y = Dataset.decode_image(x, tuple(img.shape))
    # 检查解码后的图像与原图像是否一致
    np.testing.assert_allclose(actual=y, desired=img)
```