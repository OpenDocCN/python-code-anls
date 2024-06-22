# `.\models\esm\tokenization_esm.py`

```py
# 设置文件编码为 UTF-8

# 导入所需模块和函数
import os
from typing import List, Optional, Union

# 从 tokenization_utils 模块中导入 PreTrainedTokenizer 类
from ...tokenization_utils import PreTrainedTokenizer
# 从 tokenization_utils_base 模块中导入 AddedToken 类
from ...tokenization_utils_base import AddedToken
# 从 utils 模块中导入 logging 函数
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义文件名常量
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练词汇文件映射常量
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/esm2_t6_8M_UR50D": "https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main/vocab.txt",
        "facebook/esm2_t12_35M_UR50D": "https://huggingface.co/facebook/esm2_t12_35M_UR50D/resolve/main/vocab.txt",
    },
}

# 定义预训练位置嵌入大小常量
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/esm2_t6_8M_UR50D": 1024,
    "facebook/esm2_t12_35M_UR50D": 1024,
}

# 定义加载词汇文件的函数
def load_vocab_file(vocab_file):
    # 打开词汇文件，并按行读取
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        # 返回去除每行空格后的列表
        return [l.strip() for l in lines]

# 定义 EsmTokenizer 类，继承自 PreTrainedTokenizer
class EsmTokenizer(PreTrainedTokenizer):
    """
    Constructs an ESM tokenizer.
    """

    # 定义类属性
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )
        # 加载词汇文件中的所有词汇
        self.all_tokens = load_vocab_file(vocab_file)
        # 创建 ID 到词汇的映射字典
        self._id_to_token = dict(enumerate(self.all_tokens))
        # 创建词汇到 ID 的映射字典
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}

        # TODO, all the tokens are added? But they are also part of the vocab... bit strange.
        # none of them are special, but they all need special splitting.

        # 定义不需拆分的唯一词汇列表
        self.unique_no_split_tokens = self.all_tokens
        # 更新 Trie 数据结构以支持词汇拆分
        self._update_trie(self.unique_no_split_tokens)

    # 将 ID 转换为词汇的方法
    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)
    # 将给定token转换为对应的id，如果找不到则返回未知token对应的id
    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    # 将输入的文本进行分词处理，以空格为分隔符
    def _tokenize(self, text, **kwargs):
        return text.split()

    # 获取词汇表的大小，包括添加的特殊token
    def get_vocab_size(self, with_added_tokens=False):
        return len(self._id_to_token)

    # 返回词汇表，将token映射为对应的id
    def get_vocab(self):
        return {token: i for i, token in enumerate(self.all_tokens)}

    # 将给定token转换为对应的id，如果找不到则返回未知token对应的id
    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    # 根据index返回对应的token，如果找不到则返回未知token
    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    # 构建包含特殊token的输入，支持单输入和双输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.eos_token_id]  # ESM词汇表中没有sep token
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + sep  # 多输入总是有一个EOS token

    # 获取特殊token的mask，支持单输入和双输入
    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask
    # 保存词汇表到指定目录，并指定文件名前缀
    def save_vocabulary(self, save_directory, filename_prefix):
        # 构建文件路径
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        # 打开文件，并写入所有的词汇
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        # 返回保存的文件路径
        return (vocab_file,)
    
    # 获取词汇表大小
    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size(with_added_tokens=False)
    
    # 添加新的词汇到词汇表中
    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        # 调用父类方法添加新的词汇，设置special_tokens为True
        return super()._add_tokens(new_tokens, special_tokens=True)
```