# `.\utils\dummy_music_objects.py`

```py
# 该文件由命令 `make fix-copies` 自动生成，禁止编辑。
# 导入依赖模块：从上级目录的 utils 模块中导入 DummyObject 和 requires_backends 函数
from ..utils import DummyObject, requires_backends

# 定义 Pop2PianoFeatureExtractor 类，使用 DummyObject 元类
class Pop2PianoFeatureExtractor(metaclass=DummyObject):
    # 类变量 _backends，指定支持的后端为 "music"
    _backends = ["music"]

    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保实例支持 "music" 后端
        requires_backends(self, ["music"])

# 定义 Pop2PianoTokenizer 类，使用 DummyObject 元类
class Pop2PianoTokenizer(metaclass=DummyObject):
    # 类变量 _backends，指定支持的后端为 "music"
    _backends = ["music"]

    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保实例支持 "music" 后端
        requires_backends(self, ["music"])
```