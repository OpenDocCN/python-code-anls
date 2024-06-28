# `.\utils\dummy_speech_objects.py`

```py
# 这个文件是由命令 `make fix-copies` 自动生成的，不要编辑。
# 从上级目录的 utils 模块中导入 DummyObject 和 requires_backends 函数
from ..utils import DummyObject, requires_backends

# 定义一个元类为 DummyObject 的类 ASTFeatureExtractor
class ASTFeatureExtractor(metaclass=DummyObject):
    # 类属性 _backends，指定为包含字符串 "speech" 的列表
    _backends = ["speech"]

    # 初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保 self 对象依赖 "speech" 后端
        requires_backends(self, ["speech"])

# 定义一个元类为 DummyObject 的类 Speech2TextFeatureExtractor
class Speech2TextFeatureExtractor(metaclass=DummyObject):
    # 类属性 _backends，指定为包含字符串 "speech" 的列表
    _backends = ["speech"]

    # 初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保 self 对象依赖 "speech" 后端
        requires_backends(self, ["speech"])
```