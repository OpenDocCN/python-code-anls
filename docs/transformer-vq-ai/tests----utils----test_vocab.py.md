# `transformer_vq\tests\utils\test_vocab.py`

```
# 导入 TensorFlow 模块
import tensorflow as tf

# 从transformer_vq.utils.vocab模块中导入dump_chars_to_tempfile函数
from transformer_vq.utils.vocab import dump_chars_to_tempfile

# 定义测试函数test_dump_chars_to_tempfile
def test_dump_chars_to_tempfile():
    # 创建包含字符串张量的数据集
    ds = (
        tf.data.Dataset.from_tensors(tf.constant(["abc", "def", "ghi", "jkl"]))
        .unbatch()
        .as_numpy_iterator()
    )
    # 调用dump_chars_to_tempfile函数，将数据集中的字符转储到临时文件中
    fp, count = dump_chars_to_tempfile(ds, maxchars=10)
    # 初始化实际字符列表
    actual_chars = []
    # 打开临时文件，读取其中的内容
    with open(fp, "rb") as f:
        for line in f:
            actual_chars.append(line)
    # 将实际字符列表转换为字节流
    actual_chars = b"".join(actual_chars)
    # 断言实际字符与预期字符相等
    assert actual_chars == b"abc\ndef\nghi\njkl\n"
```