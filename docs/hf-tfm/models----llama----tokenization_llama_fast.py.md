# `.\transformers\models\llama\tokenization_llama_fast.py`

```
# 导入所需的模块和库
import os  # 导入操作系统模块
from shutil import copyfile  # 从 shutil 模块中导入 copyfile 函数
from typing import Optional, Tuple  # 导入类型提示模块中的可选、元组类型

from tokenizers import processors  # 导入 tokenizers 模块中的 processors 子模块

# 从 tokenization_utils_fast 模块中导入 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从 utils 模块中导入 is_sentencepiece_available、logging 函数
from ...utils import is_sentencepiece_available, logging
# 从 utils.versions 模块中导入 require_version 函数
from ...utils.versions import require_version

# 要求 tokenizers 版本 >= 0.13.3
require_version("tokenizers>=0.13.3")

# 如果 sentencepiece 可用，则从 tokenization_llama 模块中导入 LlamaTokenizer 类，否则设为 None
if is_sentencepiece_available():
    from .tokenization_llama import LlamaTokenizer
else:
    LlamaTokenizer = None

# 获取 logger 对象
logger = logging.get_logger(__name__)
# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件的映射关系
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}
# 定义标记的起始和结束字符串
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n
    class LlamaTokenizer(PreTrainedTokenizerFast):
        """
        LlamaTokenizer 是一个基于 SentencePiece 和 tokenizers 的快速分词器。
    
        这个分词器继承自 `PreTrainedTokenizerFast`，其中包含大部分主要方法。用户应该参考这个超类以获取有关这些方法的更多信息。
    
        Args:
            vocab_file (`str`, *可选*):
                包含用于实例化分词器的词汇表的 [SentencePiece](https://github.com/google/sentencepiece) 文件（通常具有 .model 扩展名）。
            tokenizer_file (`str`, *可选*):
                包含加载分词器所需内容的 [tokenizers](https://github.com/huggingface/tokenizers) 文件（通常具有 .json 扩展名）。
            clean_up_tokenization_spaces (`bool`, *可选*, 默认为 `False`):
                是否清理解码后的空格，清理操作包括删除额外的空格等可能的痕迹。
            unk_token (`str` 或 `tokenizers.AddedToken`, *可选*, 默认为 `"<unk>"`):
                未知标记。词汇表中没有的标记无法转换为 ID，并设置为此标记。
            bos_token (`str` 或 `tokenizers.AddedToken`, *可选*, 默认为 `"<s>"`):
                用于预训练的序列开始标记。可以用作序列分类器标记。
            eos_token (`str` 或 `tokenizers.AddedToken`, *可选*, 默认为 `"</s>"`):
                序列结束标记。
            add_bos_token (`bool`, *可选*, 默认为 `True`):
                是否在序列开头添加一个 `bos_token`。
            add_eos_token (`bool`, *可选*, 默认为 `False`):
                是否在序列结尾添加一个 `eos_token`。
            use_default_system_prompt (`bool`, *可选*, 默认为 `False`):
                是否使用 Llama 的默认系统提示。
    
        """
    
        vocab_files_names = VOCAB_FILES_NAMES  # 词汇表文件名列表
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇表文件映射
        slow_tokenizer_class = LlamaTokenizer  # 慢速分词器类
        padding_side = "left"  # 填充位置为左侧
        model_input_names = ["input_ids", "attention_mask"]  # 模型输入名称列表
    
        def __init__(
            self,
            vocab_file=None,
            tokenizer_file=None,
            clean_up_tokenization_spaces=False,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            add_bos_token=True,
            add_eos_token=False,
            use_default_system_prompt=False,
            **kwargs,
        ):
            """
            初始化 LlamaTokenizer 对象。
    
            Args:
                vocab_file (`str`, *可选*):
                    包含用于实例化分词器的词汇表的 [SentencePiece](https://github.com/google/sentencepiece) 文件（通常具有 .model 扩展名）。
                tokenizer_file (`str`, *可选*):
                    包含加载分词器所需内容的 [tokenizers](https://github.com/huggingface/tokenizers) 文件（通常具有 .json 扩展名）。
                clean_up_tokenization_spaces (`bool`, *可选*, 默认为 `False`):
                    是否清理解码后的空格，清理操作包括删除额外的空格等可能的痕迹。
                unk_token (`str` 或 `tokenizers.AddedToken`, *可选*, 默认为 `"<unk>"`):
                    未知标记。词汇表中没有的标记无法转换为 ID，并设置为此标记。
                bos_token (`str` 或 `tokenizers.AddedToken`, *可选*, 默认为 `"<s>"`):
                    用于预训练的序列开始标记。可以用作序列分类器标记。
                eos_token (`str` 或 `tokenizers.AddedToken`, *可选*, 默认为 `"</s>"`):
                    序列结束标记。
                add_bos_token (`bool`, *可选*, 默认为 `True`):
                    是否在序列开头添加一个 `bos_token`。
                add_eos_token (`bool`, *可选*, 默认为 `False`):
                    是否在序列结尾添加一个 `eos_token`。
                use_default_system_prompt (`bool`, *可选*, 默认为 `False`):
                    是否使用 Llama 的默认系统提示。
            """
    # 调用父类的构造函数初始化对象
    ):
        super().__init__(
            vocab_file=vocab_file,  # 传入词汇表文件路径
            tokenizer_file=tokenizer_file,  # 传入分词器文件路径
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,  # 是否清理分词空格
            unk_token=unk_token,  # 未知标记
            bos_token=bos_token,  # 句子起始标记
            eos_token=eos_token,  # 句子结束标记
            add_bos_token=add_bos_token,  # 是否添加句子起始标记
            add_eos_token=add_eos_token,  # 是否添加句子结束标记
            use_default_system_prompt=use_default_system_prompt,  # 是否使用默认系统提示
            **kwargs,  # 其他关键字参数
        )
        # 初始化句子起始标记属性
        self._add_bos_token = add_bos_token
        # 初始化句子结束标记属性
        self._add_eos_token = add_eos_token
        # 更新后处理器
        self.update_post_processor()
        # 设置是否使用默认系统提示
        self.use_default_system_prompt = use_default_system_prompt
        # 设置词汇表文件路径
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查词汇表文件是否存在
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        # 获取句子起始标记和其对应的 ID
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        # 若添加句子起始标记但未设置其值，则引发异常
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        # 获取句子结束标记和其对应的 ID
        eos = self.eos_token
        eos_token_id = self.eos_token_id
        # 若添加句子结束标记但未设置其值，则引发异常
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        # 构建单句模板
        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        # 构建双句模板
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        # 准备特殊标记列表
        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        # 更新后处理器的模板处理器
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        # 返回是否添加句子结束标记
        return self._add_eos_token

    @property
    def add_bos_token(self):
        # 返回是否添加句子起始标记
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        # 设置是否添加句子结束标记，并更新后处理器
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        # 设置是否添加句子起始标记，并更新后处理器
        self._add_bos_token = value
        self.update_post_processor()
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速标记器的词汇表，则引发值错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 定义输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇表文件路径与输出词汇表文件路径不同，则复制词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回输出词汇表文件路径的元组
        return (out_vocab_file,)

    @property
    # 生成包含特殊标记的输入序列
    # 从transformers.models.llama.tokenization_llama.LlamaTokenizer.default_chat_template复制而来
    # TODO ArthurZ 让我们依赖模板处理器，重构所有快速标记器
    # 从transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens复制而来
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 如果需要添加bos标记，则创建包含bos标记的列表，否则为空列表
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        # 如果需要添加eos标记，则创建包含eos标记的列表，否则为空列表
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # 将输入标记序列与bos标记和eos标记拼接起来
        output = bos_token_id + token_ids_0 + eos_token_id

        # 如果有第二个输入标记序列，则拼接第二个输入标记序列的bos标记、标记和eos标记
        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        # 返回拼接后的标记序列
        return output
```