# `transformer_vq\tests\utils\fixtures.py`

```
# 导入所需的模块
import os
import pytest
import seqio
import tensorflow as tf

# 定义一个 fixture，返回一个 ByteVocabulary 对象
@pytest.fixture
def vocab():
    return seqio.ByteVocabulary()

# 定义一个 fixture，返回一个解码后的 PNG 图像数据
@pytest.fixture
def img():
    # 获取当前文件所在目录的绝对路径
    dir1 = os.path.dirname(os.path.realpath(__file__))
    # 定义图像文件名
    fname1 = "ref_img.png"
    # 使用 TensorFlow 的文件操作工具打开图像文件，并以二进制方式读取数据
    with tf.io.gfile.GFile(tf.io.gfile.join(dir1, fname1), "rb") as f1:
        data1 = f1.read()
    # 使用 TensorFlow 解码 PNG 数据，指定通道数为 3
    return tf.io.decode_png(data1, channels=3)
```