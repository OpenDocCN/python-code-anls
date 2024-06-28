# `.\utils\dummy_sentencepiece_objects.py`

```
# 以下代码由命令 `make fix-copies` 自动生成，不要修改。
from ..utils import DummyObject, requires_backends

# 定义 AlbertTokenizer 类，使用 DummyObject 元类
class AlbertTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 BarthezTokenizer 类，使用 DummyObject 元类
class BarthezTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 BartphoTokenizer 类，使用 DummyObject 元类
class BartphoTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 BertGenerationTokenizer 类，使用 DummyObject 元类
class BertGenerationTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 BigBirdTokenizer 类，使用 DummyObject 元类
class BigBirdTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 CamembertTokenizer 类，使用 DummyObject 元类
class CamembertTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 CodeLlamaTokenizer 类，使用 DummyObject 元类
class CodeLlamaTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 CpmTokenizer 类，使用 DummyObject 元类
class CpmTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 DebertaV2Tokenizer 类，使用 DummyObject 元类
class DebertaV2Tokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 ErnieMTokenizer 类，使用 DummyObject 元类
class ErnieMTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 FNetTokenizer 类，使用 DummyObject 元类
class FNetTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 GemmaTokenizer 类，使用 DummyObject 元类
class GemmaTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 GPTSw3Tokenizer 类，使用 DummyObject 元类
class GPTSw3Tokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 LayoutXLMTokenizer 类，使用 DummyObject 元类
class LayoutXLMTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 LlamaTokenizer 类，使用 DummyObject 元类
class LlamaTokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 M2M100Tokenizer 类，使用 DummyObject 元类
class M2M100Tokenizer(metaclass=DummyObject):
    # 定义支持的后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 要求当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])

# 定义 MarianTokenizer 类，使用 DummyObject 元类
class MarianTokenizer(metaclass=DummyObject):
    # 该类的定义未完整，暂无初始化方法和后端定义
    pass
    # 定义一个类属性 `_backends`，该列表包含字符串 "sentencepiece"
    _backends = ["sentencepiece"]
    
    # 定义类的初始化方法 `__init__`，接受任意数量的位置参数 `args` 和关键字参数 `kwargs`
    def __init__(self, *args, **kwargs):
        # 调用 `requires_backends` 函数，传入当前对象以及要求的后端列表 ["sentencepiece"]
        requires_backends(self, ["sentencepiece"])
# 定义一个 MBart50Tokenizer 类，使用 DummyObject 元类
class MBart50Tokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 MBartTokenizer 类，使用 DummyObject 元类
class MBartTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 MLukeTokenizer 类，使用 DummyObject 元类
class MLukeTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 MT5Tokenizer 类，使用 DummyObject 元类
class MT5Tokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 NllbTokenizer 类，使用 DummyObject 元类
class NllbTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 PegasusTokenizer 类，使用 DummyObject 元类
class PegasusTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 PLBartTokenizer 类，使用 DummyObject 元类
class PLBartTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 ReformerTokenizer 类，使用 DummyObject 元类
class ReformerTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 RemBertTokenizer 类，使用 DummyObject 元类
class RemBertTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 SeamlessM4TTokenizer 类，使用 DummyObject 元类
class SeamlessM4TTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 SiglipTokenizer 类，使用 DummyObject 元类
class SiglipTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 Speech2TextTokenizer 类，使用 DummyObject 元类
class Speech2TextTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 SpeechT5Tokenizer 类，使用 DummyObject 元类
class SpeechT5Tokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 T5Tokenizer 类，使用 DummyObject 元类
class T5Tokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 UdopTokenizer 类，使用 DummyObject 元类
class UdopTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 XGLMTokenizer 类，使用 DummyObject 元类
class XGLMTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖 "sentencepiece" 后端
        requires_backends(self, ["sentencepiece"])


# 定义一个 XLMProphetNetTokenizer 类，使用 DummyObject 元类
class XLMProphetNetTokenizer(metaclass=DummyObject):
    # 类变量，指定后端为 "sentencepiece"
    _backends = ["sentencepiece"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        #
# 定义 XLMRobertaTokenizer 类，使用 DummyObject 元类
class XLMRobertaTokenizer(metaclass=DummyObject):
    # 类变量 _back
```