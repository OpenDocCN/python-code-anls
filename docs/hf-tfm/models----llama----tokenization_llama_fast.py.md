# `.\models\llama\tokenization_llama_fast.py`

```
# 设置编码格式为 UTF-8

# 导入所需的模块和函数
import os  # 导入操作系统相关的功能
from shutil import copyfile  # 从 shutil 模块导入 copyfile 函数
from typing import Optional, Tuple  # 导入类型提示相关的类和函数

from tokenizers import processors  # 从 tokenizers 模块导入 processors

# 导入所需的自定义模块和函数
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入预训练的快速分词器
from ...utils import is_sentencepiece_available, logging  # 从 utils 模块导入检查是否安装了 sentencepiece 的函数和日志功能
from ...utils.versions import require_version  # 从 utils.versions 模块导入版本要求函数

# 要求使用的 tokenizers 版本至少为 0.13.3
require_version("tokenizers>=0.13.3")

# 如果安装了 sentencepiece，则导入 LlamaTokenizer；否则置为 None
if is_sentencepiece_available():
    from .tokenization_llama import LlamaTokenizer
else:
    LlamaTokenizer = None

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}

# 定义特定格式的起始和结束标记
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# fmt: off
# 默认系统提示文本，采用三重引号多行字符串
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class LlamaTokenizerFast(PreTrainedTokenizerFast):
    """
    构建 Llama 快速分词器，基于字节级别的 Byte-Pair-Encoding。

    这个分词器使用了 ByteFallback 和不进行任何标准化处理。

    ```python
    >>> from transformers import LlamaTokenizerFast

    >>> tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    >>> tokenizer.encode("Hello this is a test")
    [1, 15043, 445, 338, 263, 1243]
    ```

    如果需要修改 `bos_token` 或 `eos_token`，请在初始化模型时指定，或调用 `tokenizer.update_post_processor()` 确保后处理正确执行
    （否则编码序列的第一个和最后一个标记的值将不正确）。更多详情，请参阅
    # 定义了一个名为 `vocab_files_names` 的变量，其值来自外部常量 VOCAB_FILES_NAMES
    vocab_files_names = VOCAB_FILES_NAMES
    
    # 定义了一个名为 `pretrained_vocab_files_map` 的变量，其值来自外部常量 PRETRAINED_VOCAB_FILES_MAP
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    
    # 将类 `LlamaTokenizer` 赋值给变量 `slow_tokenizer_class`
    slow_tokenizer_class = LlamaTokenizer
    
    # 指定填充操作在输入的左侧进行
    padding_side = "left"
    
    # 定义了一个包含字符串元素 "input_ids" 和 "attention_mask" 的列表，并赋值给变量 `model_input_names`
    model_input_names = ["input_ids", "attention_mask"]
    
    # 定义了一个初始化函数 `__init__`，用于实例化一个新的 Tokenizer 对象
    def __init__(
        self,
        vocab_file=None,  # 可选参数：指定包含词汇的文件名
        tokenizer_file=None,  # 可选参数：指定包含 tokenizer 配置的文件名
        clean_up_tokenization_spaces=False,  # 可选参数：是否清理解码后的空格
        unk_token="<unk>",  # 可选参数：未知 token，默认为 "<unk>"
        bos_token="<s>",  # 可选参数：序列开头 token，默认为 "<s>"
        eos_token="</s>",  # 可选参数：序列结尾 token，默认为 "</s>"
        add_bos_token=True,  # 可选参数：是否在序列开头添加 bos_token，默认为 True
        add_eos_token=False,  # 可选参数：是否在序列结尾添加 eos_token，默认为 False
        use_default_system_prompt=False,  # 可选参数：是否使用默认的系统提示语（针对 Llama）
        add_prefix_space=None,  # 可选参数：是否自动在 tokenizer 前添加空格
        **kwargs,  # 其他未指定的关键字参数
    ):
    ):
        if add_prefix_space is not None:
            # 如果设置了 add_prefix_space，则发出警告信息，说明需要将分词器从慢速分词器转换过来
            logger.warning_once(
                "You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers"
            )
            # 设置参数 from_slow=True，以便在初始化时使用
            kwargs["from_slow"] = True

        # 调用父类的初始化方法，传入各种参数来初始化对象
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_default_system_prompt=use_default_system_prompt,
            **kwargs,
        )
        # 将 add_bos_token 和 add_eos_token 设置到当前对象的属性中
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        # 调用更新后处理器的方法，确保后处理器与当前的 bos_token 和 eos_token 保持同步
        self.update_post_processor()
        # 设置 use_default_system_prompt 属性
        self.use_default_system_prompt = use_default_system_prompt
        # 设置 vocab_file 属性
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查 vocab_file 是否存在，从而判断是否可以保存慢速分词器的状态
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def update_post_processor(self):
        """
        更新后处理器，使用当前的 bos_token 和 eos_token。
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        # 如果 add_bos_token 为 True，但 bos_token 为 None，则抛出错误
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        # 如果 add_eos_token 为 True，但 eos_token 为 None，则抛出错误
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        # 根据 add_bos_token 和 add_eos_token 的设置，生成单句和双句的模板
        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        special_tokens = []
        # 如果 add_bos_token 为 True，则将 bos 和其对应的 token_id 加入特殊标记列表
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        # 如果 add_eos_token 为 True，则将 eos 和其对应的 token_id 加入特殊标记列表
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        # 将生成的模板和特殊标记设置到 tokenizer 的后处理器中
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        # 返回当前对象的 add_eos_token 属性值
        return self._add_eos_token

    @property
    def add_bos_token(self):
        # 返回当前对象的 add_bos_token 属性值
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        # 设置 add_eos_token 的值，并更新后处理器
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        # 设置 add_bos_token 的值，并更新后处理器
        self._add_bos_token = value
        self.update_post_processor()
    # 保存词汇表到指定目录和文件名前缀下的文件中，并返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表，则抛出数值错误异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建输出词汇表文件的完整路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不一致，则复制词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回保存的词汇表文件路径的元组
        return (out_vocab_file,)

    @property
    # 从 LlamaTokenizer.default_chat_template 复制而来，建议使用模板处理器，重构所有快速分词器
    # 从 LlamaTokenizer.build_inputs_with_special_tokens 复制而来
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 如果需要添加开头特殊标记，则使用开头的标记 ID；否则为空列表
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        # 如果需要添加结尾特殊标记，则使用结尾的标记 ID；否则为空列表
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # 构建输出的特殊标记输入，包括开头标记、第一个序列的标记和结尾标记
        output = bos_token_id + token_ids_0 + eos_token_id

        # 如果有第二个序列的标记存在，则添加第二个序列的开头标记、标记和结尾标记
        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        # 返回构建好的特殊标记输入列表
        return output
```