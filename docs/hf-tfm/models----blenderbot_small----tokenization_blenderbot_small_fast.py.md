# `.\models\blenderbot_small\tokenization_blenderbot_small_fast.py`

```
# 指定 Python 文件使用 UTF-8 编码
# 声明版权和许可相关信息，使用 Apache License 2.0
# 文件头部版权和许可说明
# 许可证相关网址
# 提示许可证 "按原样" 提供，不提供任何担保或保证
"""快速 BlenderbotSmall 的分词类。"""
from typing import List, Optional  # 引入类型提示库，指定变量或函数参数的类型

from tokenizers import ByteLevelBPETokenizer  # 引入字节级 BPE 分词器

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 引入预训练分词器的快速版本
from ...utils import logging  # 引入日志记录工具
from .tokenization_blenderbot_small import BlenderbotSmallTokenizer  # 引入慢速 BlenderbotSmall 分词器

logger = logging.get_logger(__name__)  # 获取日志记录器实例

# 定义字典，包含词汇文件、合并文件、分词器配置文件的名称
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_config_file": "tokenizer_config.json",
}

# 定义预训练模型中词汇文件的位置
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/vocab.json"
    },
    "merges_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/merges.txt"
    },
    "tokenizer_config_file": {
        "facebook/blenderbot_small-90M": (
            "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/tokenizer_config.json"
        )
    },
}

# 定义预训练模型中的最大位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/blenderbot_small-90M": 512,  # 最大嵌入大小为 512
}

# 定义一个快速的 BlenderbotSmall 分词器类
class BlenderbotSmallTokenizerFast(PreTrainedTokenizerFast):
    """
    构造一个快速的 BlenderbotSmall 分词器（基于 HuggingFace 的 *tokenizers* 库）。

    参数:
        vocab_file (`str`):
            词汇文件的路径。
    """

    # 定义分词器的词汇文件名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练模型中词汇文件的映射
    pretrained_vocab_files_map = PRETRAINED_VOC_FILES_MAP
    # 定义预训练模型中最大输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 引用慢速分词器类
    slow_tokenizer_class = BlenderbotSmallTokenizer

    # 定义构造函数，初始化分词器
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        unk_token="<|endoftext|>",  # 默认未知词汇符号
        bos_token="<|endoftext|>",  # 默认起始符号
        eos_token="<|endoftext|>",  # 默认结束符号
        add_prefix_space=False,  # 是否在词汇前添加前缀空格
        trim_offsets=True,  # 是否修剪偏移
        **kwargs,
    ):
        # 调用父类构造函数，初始化 ByteLevelBPETokenizer
        super().__init__(
            ByteLevelBPETokenizer(
                vocab=vocab_file,
                merges=merges_file,
                add_prefix_space=add_prefix_space,
                trim_offsets=trim_offsets,
            ),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs,
        )
        # 是否添加前缀空格
        self.add_prefix_space = add_prefix_space
    # 根据给定的token_ids_0和token_ids_1构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 将bos_token_id添加到token_ids_0的开头，将eos_token_id添加到token_ids_0的结尾
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        # 如果token_ids_1为空，则直接返回output
        if token_ids_1 is None:
            return output
        # 如果token_ids_1不为空，则在output后面添加eos_token_id，token_ids_1，和eos_token_id，并返回
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    # 根据token_ids_0和token_ids_1创建token type ids，用于序列对分类任务
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BlenderbotSmall
        does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 定义分隔符和类别标记
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果token_ids_1为空，则返回长度为len(cls + token_ids_0 + sep)的全0列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 如果token_ids_1不为空，则返回长度为len(cls + token_ids_0 + sep + sep + token_ids_1 + sep)的全0列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    # 获取默认的聊天模板
    @property
    # Copied from transformers.models.blenderbot.tokenization_blenderbot.BlenderbotTokenizer.default_chat_template
    def default_chat_template(self):
        """
        A very simple chat template that just adds whitespace between messages.
        """
        # 如果没有为tokenizer定义聊天模板，则使用默认模板并发出警告
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )
```