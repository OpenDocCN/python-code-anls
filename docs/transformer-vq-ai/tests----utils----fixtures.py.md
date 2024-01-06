# `transformer_vq\tests\utils\fixtures.py`

```
# 导入所需的模块
import os
import pytest
import seqio
import tensorflow as tf

# 定义 fixture，返回一个 ByteVocabulary 对象
@pytest.fixture
def vocab():
    return seqio.ByteVocabulary()

# 定义 fixture，返回一个解码后的 PNG 图像数据
@pytest.fixture
def img():
    # 获取当前文件所在目录
    dir1 = os.path.dirname(os.path.realpath(__file__))
    # 定义图像文件名
    fname1 = "ref_img.png"
    # 使用 TensorFlow 的文件操作函数打开图像文件，并以二进制形式读取数据
    with tf.io.gfile.GFile(tf.io.gfile.join(dir1, fname1), "rb") as f1:
        data1 = f1.read()
    # 使用 TensorFlow 的图像解码函数对图像数据进行解码，返回解码后的图像数据
    return tf.io.decode_png(data1, channels=3)
```