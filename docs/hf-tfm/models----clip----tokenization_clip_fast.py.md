# `.\models\clip\tokenization_clip_fast.py`

```py
# 导入所需模块和类
from typing import List, Optional, Tuple
from tokenizers import pre_tokenizers  # 导入tokenizers模块中的pre_tokenizers类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 从tokenization_utils_fast模块导入PreTrainedTokenizerFast类
from ...utils import logging  # 从utils模块导入logging类
from .tokenization_clip import CLIPTokenizer  # 从当前目录中的tokenization_clip模块导入CLIPTokenizer类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义用于存储词汇文件名称的字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
    },
    "merges_file": {
        "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "openai/clip-vit-base-patch32": (
            "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json"
        ),
    },
}

# 定义预训练位置嵌入尺寸的映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai/clip-vit-base-patch32": 77,
}

# 定义CLIPTokenizerFast类，继承自PreTrainedTokenizerFast类
class CLIPTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" CLIP tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    # 定义词汇文件名称字典
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速标记器类为CLIPTokenizer
    slow_tokenizer_class = CLIPTokenizer

    # 初始化方法
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",  # hack to enable padding
        **kwargs,
    ):
        # 调用父类初始化方法
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        # 如果 backend_tokenizer 的预标记器不是 pre_tokenizers.Sequence，则抛出值错误
        if not isinstance(self.backend_tokenizer.pre_tokenizer, pre_tokenizers.Sequence):
            raise ValueError(
                "The `backend_tokenizer` provided does not match the expected format. The CLIP tokenizer has been"
                " heavily modified from transformers version 4.17.0. You need to convert the tokenizer you are using"
                " to be compatible with this version.The easiest way to do so is"
                ' `CLIPTokenizerFast.from_pretrained("path_to_local_folder_or_hub_repo, from_slow=True)`. If you want'
                " to use your existing tokenizer, you will have to revert to a version prior to 4.17.0 of"
                " transformers."
            )

        # 对 backend_tokenizer 的解码方法进行封装
        self._wrap_decode_method_backend_tokenizer()

    # 很丑的Hack方法，用于使填充正确解码
    def _wrap_decode_method_backend_tokenizer(self):
        # 保存原始解码方法
        orig_decode_method = self.backend_tokenizer.decode

        # 定义新的解码方法
        def new_decode_method(*args, **kwargs):
            text = orig_decode_method(*args, **kwargs)
            # 替换文本中的后缀，并去除首尾空格
            text = text.replace(self.backend_tokenizer.model.end_of_word_suffix, " ").strip()
            return text

        # 更新backend_tokenizer的解码方法
        self.backend_tokenizer.decode = new_decode_method

    # 构建带有特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义函数的返回类型为 List[int]
        ) -> List[int]:
            """
            # 用于序列分类任务，构建带有特殊标记的模型输入
            # CLIP 序列格式如下：
            #
            # - 单个序列: `<|startoftext|> X <|endoftext|>`
            #
            # 序列对的情况并不常见，但处理时不会添加分隔符
            #
            # 参数:
            #   token_ids_0 (`List[int]`):
            #       要添加特殊标记的 ID 列表
            #   token_ids_1 (`List[int]`, *可选*):
            #       序列对的可选第二个 ID 列表
            #
            # 返回:
            #   `List[int]`: 带有适当特殊标记的输入 ID 列表
            """
            # 定义序列起始标记
            bos_token = [self.bos_token_id]
            # 定义序列结束标记
            eos_token = [self.eos_token_id]
    
            # 如果只有一个序列
            if token_ids_1 is None:
                # 返回加上起始和结束标记的序列
                return bos_token + token_ids_0 + eos_token
            # 如果有两个序列
            return bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token
    
        # 定义返回类型为 List[int] 的函数，生成序列的掩码
        def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
            """
            # 根据传入的两个序列创建掩码
            # CLIP 不使用 token type ID，因此返回全为零的列表
            #
            # 参数:
            #   token_ids_0 (`List[int]`):
            #       ID 列表
            #   token_ids_1 (`List[int]`, *可选*):
            #       序列对的可选第二个 ID 列表
            #
            # 返回:
            #   `List[int]`: 全为零的列表
            """
            # 定义序列起始标记
            bos_token = [self.bos_token_id]
            # 定义序列结束标记
            eos_token = [self.eos_token_id]
    
            # 如果只有一个序列
            if token_ids_1 is None:
                # 返回和序列长度相同的零列表
                return len(bos_token + token_ids_0 + eos_token) * [0]
            # 如果有两个序列
            return len(bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token) * [0]
    
        # 定义返回类型为 Tuple[str] 的函数，用于保存词汇表
        def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
            # 使用 _tokenizer 对象的 model 方法保存词汇表，传入保存路径和文件名前缀
            files = self._tokenizer.model.save(save_directory, name=filename_prefix)
            # 返回保存的文件名元组
            return tuple(files)
```