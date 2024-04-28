# `.\transformers\utils\dummy_tokenizers_objects.py`

```
# 该文件是由命令 `make fix-copies` 自动生成的，不要编辑。
# 导入必要的模块
from ..utils import DummyObject, requires_backends

# 定义 AlbertTokenizerFast 类
class AlbertTokenizerFast(metaclass=DummyObject):
    # 指定后端为 "tokenizers"
    _backends = ["tokenizers"]

    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 检查是否存在必要的后端 "tokenizers"
        requires_backends(self, ["tokenizers"])

# 定义 BartTokenizerFast 类
class BartTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 BarthezTokenizerFast 类
class BarthezTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 BertTokenizerFast 类
class BertTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 BigBirdTokenizerFast 类
class BigBirdTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 BlenderbotTokenizerFast 类
class BlenderbotTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 BlenderbotSmallTokenizerFast 类
class BlenderbotSmallTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 BloomTokenizerFast 类
class BloomTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 CamembertTokenizerFast 类
class CamembertTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 CLIPTokenizerFast 类
class CLIPTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 CodeLlamaTokenizerFast 类
class CodeLlamaTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 CodeGenTokenizerFast 类
class CodeGenTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 ConvBertTokenizerFast 类
class ConvBertTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 CpmTokenizerFast 类
class CpmTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 DebertaTokenizerFast 类
class DebertaTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 DebertaV2TokenizerFast 类
class DebertaV2TokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])

# 定义 RetriBertTokenizerFast 类
class RetriBertTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]
    # 初始化函数，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 检查当前对象是否需要特定的后端支持，这里需要"tokenizers"后端支持
        requires_backends(self, ["tokenizers"])
# 定义 DistilBertTokenizerFast 类，使用 DummyObject 元类
class DistilBertTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 DPRContextEncoderTokenizerFast 类，使用 DummyObject 元类
class DPRContextEncoderTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 DPRQuestionEncoderTokenizerFast 类，使用 DummyObject 元类
class DPRQuestionEncoderTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 DPRReaderTokenizerFast 类，使用 DummyObject 元类
class DPRReaderTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 ElectraTokenizerFast 类，使用 DummyObject 元类
class ElectraTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 FNetTokenizerFast 类，使用 DummyObject 元类
class FNetTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 FunnelTokenizerFast 类，使用 DummyObject 元类
class FunnelTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 GPT2TokenizerFast 类，使用 DummyObject 元类
class GPT2TokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 GPTNeoXTokenizerFast 类，使用 DummyObject 元类
class GPTNeoXTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 GPTNeoXJapaneseTokenizer 类，使用 DummyObject 元类
class GPTNeoXJapaneseTokenizer(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 HerbertTokenizerFast 类，使用 DummyObject 元类
class HerbertTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 LayoutLMTokenizerFast 类，使用 DummyObject 元类
class LayoutLMTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 LayoutLMv2TokenizerFast 类，使用 DummyObject 元类
class LayoutLMv2TokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 LayoutLMv3TokenizerFast 类，使用 DummyObject 元类
class LayoutLMv3TokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 LayoutXLMTokenizerFast 类，使用 DummyObject 元类
class LayoutXLMTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 LEDTokenizerFast 类，使用 DummyObject 元类
class LEDTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 LlamaTokenizerFast 类，使用 DummyObject 元类
class LlamaTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求后端为["tokenizers"]
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])
# 定义 LongformerTokenizerFast 类，使用 DummyObject 元类
class LongformerTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 LxmertTokenizerFast 类，使用 DummyObject 元类
class LxmertTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 MarkupLMTokenizerFast 类，使用 DummyObject 元类
class MarkupLMTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 MBartTokenizerFast 类，使用 DummyObject 元类
class MBartTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 MBart50TokenizerFast 类，使用 DummyObject 元类
class MBart50TokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 MobileBertTokenizerFast 类，使用 DummyObject 元类
class MobileBertTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 MPNetTokenizerFast 类，使用 DummyObject 元类
class MPNetTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 MT5TokenizerFast 类���使用 DummyObject 元类
class MT5TokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 MvpTokenizerFast 类，使用 DummyObject 元类
class MvpTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 NllbTokenizerFast 类，使用 DummyObject 元类
class NllbTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 NougatTokenizerFast 类，使用 DummyObject 元类
class NougatTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 OpenAIGPTTokenizerFast 类，使用 DummyObject 元类
class OpenAIGPTTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 PegasusTokenizerFast 类，使用 DummyObject 元类
class PegasusTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 Qwen2TokenizerFast 类，使用 DummyObject 元类
class Qwen2TokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 RealmTokenizerFast 类，使用 DummyObject 元类
class RealmTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 ReformerTokenizerFast 类，使用 DummyObject 元类
class ReformerTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 RemBertTokenizerFast 类，使用 DummyObject 元类
class RemBertTokenizerFast(metaclass=DummyObject):
    # 指定_backends��性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数，要求必须有"tokenizers"后端
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


# 定义 RobertaTokenizerFast 类，未完整定义，缺少初始化方法
class RobertaTokenizerFast(metaclass=DummyObject):
    # 定义私有属性_backends，包含"tokenizers"字符串
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用requires_backends函数，确保当前对象拥有"tokenizers"后端
        requires_backends(self, ["tokenizers"])
# 定义 RoFormerTokenizerFast 类，使用 DummyObject 元类
class RoFormerTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])


# 定义 SeamlessM4TTokenizerFast 类，使用 DummyObject 元类
class SeamlessM4TTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])


# 定义 SplinterTokenizerFast 类，使用 DummyObject 元类
class SplinterTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])


# 定义 SqueezeBertTokenizerFast 类，使用 DummyObject 元类
class SqueezeBertTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])


# 定义 T5TokenizerFast 类，使用 DummyObject 元类
class T5TokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])


# 定义 WhisperTokenizerFast 类，使用 DummyObject 元类
class WhisperTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])


# 定义 XGLMTokenizerFast 类，使用 DummyObject 元类
class XGLMTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])


# 定义 XLMRobertaTokenizerFast 类，使用 DummyObject 元类
class XLMRobertaTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])


# 定义 XLNetTokenizerFast 类，使用 DummyObject 元类
class XLNetTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])


# 定义 PreTrainedTokenizerFast 类，使用 DummyObject 元类
class PreTrainedTokenizerFast(metaclass=DummyObject):
    # 指定_backends属性为["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数
    def __init__(self, *args, **kwargs):
        # 要求存在 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])
```