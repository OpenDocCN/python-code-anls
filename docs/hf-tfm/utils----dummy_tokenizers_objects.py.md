# `.\utils\dummy_tokenizers_objects.py`

```
# 此文件由命令 `make fix-copies` 自动生成，不要编辑。

# 从上级目录的 utils 模块中导入 DummyObject 类和 requires_backends 函数
from ..utils import DummyObject, requires_backends

# 定义 AlbertTokenizerFast 类，使用 DummyObject 元类
class AlbertTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 BartTokenizerFast 类，使用 DummyObject 元类
class BartTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 BarthezTokenizerFast 类，使用 DummyObject 元类
class BarthezTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 BertTokenizerFast 类，使用 DummyObject 元类
class BertTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 BigBirdTokenizerFast 类，使用 DummyObject 元类
class BigBirdTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 BlenderbotTokenizerFast 类，使用 DummyObject 元类
class BlenderbotTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 BlenderbotSmallTokenizerFast 类，使用 DummyObject 元类
class BlenderbotSmallTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 BloomTokenizerFast 类，使用 DummyObject 元类
class BloomTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 CamembertTokenizerFast 类，使用 DummyObject 元类
class CamembertTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 CLIPTokenizerFast 类，使用 DummyObject 元类
class CLIPTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 CodeLlamaTokenizerFast 类，使用 DummyObject 元类
class CodeLlamaTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 CodeGenTokenizerFast 类，使用 DummyObject 元类
class CodeGenTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 CohereTokenizerFast 类，使用 DummyObject 元类
class CohereTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 ConvBertTokenizerFast 类，使用 DummyObject 元类
class ConvBertTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 CpmTokenizerFast 类，使用 DummyObject 元类
class CpmTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖于 "tokenizers" 后端
        requires_backends(self, ["tokenizers"])

# 定义 DebertaTokenizerFast 类，使用 DummyObject 元类
class DebertaTokenizerFast(metaclass=DummyObject):
    # 定义 _backends 类变量，指定为 ["tokenizers"]
    _backends = ["tokenizers"]

    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保对象依赖
    # 初始化方法，用于对象的初始化
    def __init__(self, *args, **kwargs):
        # 调用 requires_backends 函数，确保当前对象依赖的 "tokenizers" 后端可用
        requires_backends(self, ["tokenizers"])
class RetriBertTokenizerFast(metaclass=DummyObject):
    # 定义一个名为 RetriBertTokenizerFast 的类，使用 DummyObject 作为元类
    _backends = ["tokenizers"]
    # 类属性 _backends，存储字符串列表 ["tokenizers"]

    def __init__(self, *args, **kwargs):
        # 构造函数，接受任意位置参数 *args 和任意关键字参数 **kwargs
        requires_backends(self, ["tokenizers"])
        # 调用 requires_backends 函数，确保 self 实例依赖于 "tokenizers" 后端


class DistilBertTokenizerFast(metaclass=DummyObject):
    # 定义一个名为 DistilBertTokenizerFast 的类，使用 DummyObject 作为元类
    _backends = ["tokenizers"]
    # 类属性 _backends，存储字符串列表 ["tokenizers"]

    def __init__(self, *args, **kwargs):
        # 构造函数，接受任意位置参数 *args 和任意关键字参数 **kwargs
        requires_backends(self, ["tokenizers"])
        # 调用 requires_backends 函数，确保 self 实例依赖于 "tokenizers" 后端


# 后续类的结构与上述类似，均包含一个名为 _backends 的类属性和一个构造函数
# 每个构造函数都调用 requires_backends 函数确保依赖于 "tokenizers" 后端

class DPRContextEncoderTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class DPRQuestionEncoderTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class DPRReaderTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class ElectraTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class FNetTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class FunnelTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class GemmaTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class GPT2TokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class GPTNeoXTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class GPTNeoXJapaneseTokenizer(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class HerbertTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class LayoutLMTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class LayoutLMv2TokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class LayoutLMv3TokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class LayoutXLMTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])
class LEDTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class LlamaTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class LongformerTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class LxmertTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class MarkupLMTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class MBartTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class MBart50TokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class MobileBertTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class MPNetTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class MT5TokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class MvpTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class NllbTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class NougatTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class OpenAIGPTTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class PegasusTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class Qwen2TokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class RealmTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])  # 调用函数 requires_backends，确保实例依赖 "tokenizers" 后端


class ReformerTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]  # 定义类属性 _backends，值为包含字符串 "tokenizers" 的列表
    # 定义一个类变量 `_backends`，其值是包含字符串 "tokenizers" 的列表，可能用于指示类依赖的后端组件。
    _backends = ["tokenizers"]
    
    # 类的初始化方法，接受任意数量的位置参数和关键字参数。
    def __init__(self, *args, **kwargs):
        # 调用 `requires_backends` 函数，验证当前类实例是否具有所需的后端组件 "tokenizers"。
        requires_backends(self, ["tokenizers"])
# 定义一个自定义元类 DummyObject，用于创建带有特定后端依赖的快速标记器类
class RemBertTokenizerFast(metaclass=DummyObject):
    # 定义类属性 _backends，表示此标记器类依赖于名为 "tokenizers" 的后端
    _backends = ["tokenizers"]

    # 初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用函数 requires_backends，确保实例具有必需的后端依赖
        requires_backends(self, ["tokenizers"])


class RobertaTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class RoFormerTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class SeamlessM4TTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class SplinterTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class SqueezeBertTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class T5TokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class UdopTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class WhisperTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class XGLMTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class XLMRobertaTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class XLNetTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])


class PreTrainedTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])
```