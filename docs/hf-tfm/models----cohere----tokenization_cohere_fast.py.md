# `.\models\cohere\tokenization_cohere_fast.py`

```py
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
from typing import Dict, List, Literal, Union  # 导入类型提示相关的模块

from tokenizers import processors  # 从tokenizers模块导入processors

from ...pipelines.conversational import Conversation  # 导入对话处理相关模块
from ...tokenization_utils_base import BatchEncoding  # 导入批量编码相关模块
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入预训练的快速分词器
from ...utils import logging  # 导入日志记录工具
from ...utils.versions import require_version  # 导入版本要求检查函数

require_version("tokenizers>=0.13.3")  # 要求tokenizers版本至少为0.13.3

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}  # 定义词汇文件名映射

PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "Cohere/Command-nightly": "https://huggingface.co/Cohere/Command-nightly/blob/main/tokenizer.json",
    },  # 预训练词汇文件的映射，指定了Cohere/Command-nightly模型的tokenizer.json文件位置
}

# fmt: off
DEFAULT_SYSTEM_PROMPT = "You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere."
DEFAULT_RAG_PREAMBLE = """## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling."""
# fmt: on


class CohereTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Cohere tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and NFC normalization.

    ```
    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")
    >>> tokenizer.encode("Hello this is a test")
    [5, 28339, 2075, 1801, 1671, 3282]
    ```

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None
    # No `max_model_input_sizes`

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<UNK>",
        bos_token="<BOS_TOKEN>",
        eos_token="<|END_OF_TURN_TOKEN|>",
        add_bos_token=True,
        add_eos_token=False,
        use_default_system_prompt=False,
        add_prefix_space=False,
        **kwargs,
    ):


        vocab_files_names = VOCAB_FILES_NAMES
        # 初始化类的属性 `vocab_files_names`，使用预定义的常量 `VOCAB_FILES_NAMES`

        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 初始化类的属性 `pretrained_vocab_files_map`，使用预定义的常量 `PRETRAINED_VOCAB_FILES_MAP`

        padding_side = "left"
        # 初始化类的属性 `padding_side`，设置为字符串 "left"

        model_input_names = ["input_ids", "attention_mask"]
        # 初始化类的属性 `model_input_names`，设置为包含两个字符串元素的列表

        slow_tokenizer_class = None
        # 初始化类的属性 `slow_tokenizer_class`，设置为 `None`

        # No `max_model_input_sizes`
        # 没有定义 `max_model_input_sizes` 属性

        def __init__(
            self,
            vocab_file=None,
            merges_file=None,
            tokenizer_file=None,
            clean_up_tokenization_spaces=False,
            unk_token="<UNK>",
            bos_token="<BOS_TOKEN>",
            eos_token="<|END_OF_TURN_TOKEN|>",
            add_bos_token=True,
            add_eos_token=False,
            use_default_system_prompt=False,
            add_prefix_space=False,
            **kwargs,
        ):
        # 类的初始化方法，定义了多个可选参数和默认值，用于实例化一个 tokenizer 对象
        ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_default_system_prompt=use_default_system_prompt,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.use_default_system_prompt = use_default_system_prompt
        self.vocab_file = vocab_file
        self.grounded_generation_template = kwargs.pop("grounded_generation_template", None)
        self.tool_use_template = kwargs.pop("tool_use_template", None)

        # TODO @ArthurZucker this can only work one way for now, to update later-on. Tests should also properly
        # check this as they were green before.
        # 序列化并存储当前后端分词器的预处理器和解码器状态
        pre_tok_state = pickle.dumps(self.backend_tokenizer.pre_tokenizer)
        decoder_state = pickle.dumps(self.backend_tokenizer.decoder)

        # 如果设置了 add_prefix_space 为 True，则修改序列化状态中的相应配置
        if add_prefix_space:
            pre_tok_state = pre_tok_state.replace(b'"add_prefix_space":false', b'"add_prefix_space": true')
            decoder_state = decoder_state.replace(b'"add_prefix_space":false', b'"add_prefix_space": true')
        
        # 从序列化状态中恢复后端分词器的预处理器和解码器
        self.backend_tokenizer.pre_tokenizer = pickle.loads(pre_tok_state)
        self.backend_tokenizer.decoder = pickle.loads(decoder_state)

        self.add_prefix_space = add_prefix_space

    # 对输入进行批量编码，并返回 BatchEncoding 对象
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        
        # 如果未设置 add_prefix_space 为 True 或者输入未经预分词，则抛出异常
        if not (self.add_prefix_space or not is_split_into_words):
            raise Exception(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True to use it with"
                " pretokenized inputs."
            )

        return super()._batch_encode_plus(*args, **kwargs)

    # 对单个输入进行编码，并返回 BatchEncoding 对象
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 如果未设置 add_prefix_space 为 True 或者输入未经预分词，则抛出异常
        if not (self.add_prefix_space or not is_split_into_words):
            raise Exception(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True to use it with"
                " pretokenized inputs."
            )

        return super()._encode_plus(*args, **kwargs)
    def update_post_processor(self):
        """
        更新底层的后处理器，使用当前的 `bos_token` 和 `eos_token`。
        """
        bos = self.bos_token  # 获取开始词（bos_token）
        bos_token_id = self.bos_token_id  # 获取开始词的 ID（bos_token_id）
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")  # 如果 add_bos_token 为 True 但 bos_token 为 None，则引发错误

        eos = self.eos_token  # 获取结束词（eos_token）
        eos_token_id = self.eos_token_id  # 获取结束词的 ID（eos_token_id）
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")  # 如果 add_eos_token 为 True 但 eos_token 为 None，则引发错误

        # 创建单个和双语句模板
        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))  # 如果需要添加开始词，则将开始词及其 ID 添加到特殊词列表中
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))  # 如果需要添加结束词，则将结束词及其 ID 添加到特殊词列表中
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )  # 更新 Tokenizer 的后处理器使用新的模板和特殊词列表

    @property
    def add_eos_token(self):
        return self._add_eos_token  # 返回是否添加结束词的属性值

    @property
    def add_bos_token(self):
        return self._add_bos_token  # 返回是否添加开始词的属性值

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value  # 设置是否添加结束词的属性值
        self.update_post_processor()  # 更新后处理器以反映属性变化

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value  # 设置是否添加开始词的属性值
        self.update_post_processor()  # 更新后处理器以反映属性变化

    @property
    def apply_tool_use_template(
        self,
        conversation: Union[List[Dict[str, str]], "Conversation"],
        tools: List[Dict],
        **kwargs,
    ):
        """
        应用工具使用模板到给定对话和工具列表中的工具。
        """
        # TODO ArthurZ let's rely on the template processor instead, refactor all fast tokenizers

    def apply_grounded_generation_template(
        self,
        conversation: Union[List[Dict[str, str]], "Conversation"],
        documents: List[Dict],
        citation_mode: Literal["fast", "accurate"] = "accurate",
        **kwargs,
    ):
        """
        应用基于文档生成的模板到给定对话和文档列表中的文档。
        """
        # TODO ArthurZ let's rely on the template processor instead, refactor all fast tokenizers

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        构建包含特殊标记的输入序列。

        Args:
        - token_ids_0: 第一个输入序列的 token IDs
        - token_ids_1: 可选，第二个输入序列的 token IDs

        Returns:
        - 包含特殊标记的输入序列
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []  # 如果需要添加开始词，则创建开始词的 ID 列表，否则为空列表
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []  # 如果需要添加结束词，则创建结束词的 ID 列表，否则为空列表

        output = bos_token_id + token_ids_0 + eos_token_id  # 构建输出序列，包含开始词 ID、第一个输入序列的 token IDs、结束词 ID

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id  # 如果存在第二个输入序列，则将第二个输入序列的 token IDs 同样添加到输出序列中

        return output  # 返回包含特殊标记的完整输入序列
```