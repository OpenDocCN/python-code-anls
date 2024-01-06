# `transformer_vq\tests\utils\test_datasets.py`

```
# 导入所需的库
import numpy as np
import pytest
import seqio
import tensorflow as tf

# 导入自定义的模块和函数
from transformer_vq.utils.datasets import Dataset
from transformer_vq.utils.datasets import image_flatten
from transformer_vq.utils.datasets import image_offset
from tests.utils.fixtures import img

# 定义测试函数，测试字节词汇表的特殊符号
def test_byte_vocab_specials():
    # 创建一个字节词汇表对象
    vocab = seqio.ByteVocabulary()
    # 断言特殊符号的索引值
    assert vocab.pad_id == 0
    assert vocab.eos_id == 1
    assert vocab.unk_id == 2
    # 测试是否抛出预期的异常
    with pytest.raises(NotImplementedError):
        print(f"byte-level bos_id: {vocab.bos_id}")
# 定义一个测试图像处理的函数
def test_img_pipeline(img):
    # 将图像转换为 TensorFlow 常量，并指定数据类型为无符号整型
    x = {"targets": tf.constant(img, dtype=tf.uint8)}
    # 对图像进行扁平化处理
    x = image_flatten(x)
    # 对图像进行偏移处理
    x = image_offset(x)["targets"]
    # 解码图像
    y = Dataset.decode_image(x, tuple(img.shape))
    # 检查解码后的图像与原图像是否一致
    np.testing.assert_allclose(actual=y, desired=img)
```