# `.\transformers\utils\dummy_speech_objects.py`

```
# 该文件是由命令 `make fix-copies` 自动生成的，请勿编辑。
# 导入依赖的模块
from ..utils import DummyObject, requires_backends

# 定义一个虚拟类 ASTFeatureExtractor，其元类为 DummyObject
class ASTFeatureExtractor(metaclass=DummyObject):
    # 定义该类依赖的后端为 "speech"
    _backends = ["speech"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 检查是否满足依赖的后端 "speech"
        requires_backends(self, ["speech"])

# 定义一个虚拟类 Speech2TextFeatureExtractor，其元类为 DummyObject
class Speech2TextFeatureExtractor(metaclass=DummyObject):
    # 定义该类依赖的后端为 "speech"
    _backends = ["speech"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 检查是否满足依赖的后端 "speech"
        requires_backends(self, ["speech"])
```