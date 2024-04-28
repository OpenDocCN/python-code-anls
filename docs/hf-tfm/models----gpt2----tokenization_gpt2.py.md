# `.\models\gpt2\tokenization_gpt2.py`

```py
# 指定编码格式为 UTF-8
# 版权声明
# 著作权 2018 年由 Open AI 团队和 HuggingFace Inc. 团队拥有。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按原样”提供的，
# 不提供任何形式的担保或条件，无论是明示的还是暗示的。
# 请查阅许可证以获取特定语言的权限和限制。
"""OpenAI GPT 的分词类。"""


# 导入必要的库
import json
import os
from functools import lru_cache  # 导入用于缓存函数结果的装饰器
from typing import List, Optional, Tuple

import regex as re  # 导入正则表达式库

# 从 tokenization_utils 模块中导入必要的类和函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging  # 导入日志模块


# 获取 logger 对象用于日志记录
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/vocab.json",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/vocab.json",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/vocab.json",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/vocab.json",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/vocab.json",
    },
    "merges_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/merges.txt",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/merges.txt",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/merges.txt",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/merges.txt",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/merges.txt",
    },
}

# 预训练位置嵌入的尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt2": 1024,
    "gpt2-medium": 1024,
    "gpt2-large": 1024,
    "gpt2-xl": 1024,
    "distilgpt2": 1024,
}


@lru_cache()  # 使用 lru_cache 装饰器缓存函数结果
def bytes_to_unicode():
    """
    返回 utf-8 字节列表和映射到 Unicode 字符串的映射。我们明确避免了映射到空格/控制字符，
    因为 bpe 代码在这些字符上报错。
    
    可逆的 bpe 代码适用于 Unicode 字符串。这意味着如果您希望避免 UNKs，您需要在词汇表中有大量的 Unicode 字符。
    当您拥有大约 100 亿个令牌数据集时，您最终需要大约 5K 个字符来获得良好的覆盖率。
    这是您正常的，比如，32K bpe 词汇表的显着百分比。
    为了避免这种情况，我们希望在 utf-8 字节和 Unicode 字符串之间建立查找表。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )  # 包含常见 Unicode 字节范围的列表
    cs = bs[:]  # 将 bs 复制到 cs
    n = 0  # 初始化计数器
    for b in range(2**8):  # 遍历 0 到 255 的整数范围
        if b not in bs:  # 如果当前字节不在 bs 中
            bs.append(b)  # 将当前字节添加到 bs
            cs.append(2**8 + n)  # 将扩展的 Unicode 编码添加到 cs
            n += 1  # 计数器加 1
    将 cs 列表中的每个整数转换为相应的字符
    使用 bs 列表中的元素作为键，cs 列表中的元素作为值，创建一个字典并返回
def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词以符号元组形式表示（符号为可变长度的字符串）。
    """
    # 初始化符号对集合
    pairs = set()
    # 记录前一个字符
    prev_char = word[0]
    # 遍历单词中的每个字符
    for char in word[1:]:
        # 将前一个字符和当前字符作为一个符号对加入集合
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符
        prev_char = char
    # 返回符号对集合
    return pairs


class GPT2Tokenizer(PreTrainedTokenizer):
    """
    构建 GPT-2 分词器。基于字节级字节对编码。

    该分词器已经训练，将空格视为标记的一部分（有点像 sentencepiece），因此一个单词的编码方式会根据它是否在句子开头（无空格）而变化：

    ```python
    >>> from transformers import GPT2Tokenizer

    >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```py

    你可以通过在实例化此分词器或在对文本进行编码时传入 `add_prefix_space=True` 来绕过此行为，但由于模型没有按照这种方式预训练，这可能会降低性能。

    <Tip>

    当与 `is_split_into_words=True` 一起使用时，此分词器将在每个单词前添加一个空格（即使是第一个单词）。

    </Tip>

    此分词器继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。

    Args:
        vocab_file (`str`):
            词汇表文件路径。
        merges_file (`str`):
            合并文件路径。
        errors (`str`, *optional*, 默认为 `"replace"`):
            解码字节为 UTF-8 时要遵循的范例。更多信息请参阅[bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode)。
        unk_token (`str`, *optional*, 默认为 `"<|endoftext|>"`):
            未知标记。不在词汇表中的标记无法转换为 ID，并被设置为此标记。
        bos_token (`str`, *optional*, 默认为 `"<|endoftext|>"`):
            序列开始标记。
        eos_token (`str`, *optional*, 默认为 `"<|endoftext|>"`):
            序列结束标记。
        pad_token (`str`, *optional*):
            用于填充的标记，例如当批处理不同长度的序列时。
        add_prefix_space (`bool`, *optional*, 默认为 `False`):
            是否在输入前添加一个初始空格。这样可以将前导词视为任何其他词。（GPT2 分词器通过前导空格检测词的开头）。
        add_bos_token (`bool`, *optional*, 默认为 `False`):
            是否在输入前添加一个初始句子开始标记。这样可以将前导词视为任何其他词。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # 加载预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 加载预训练位置嵌入的最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs,
    ):
        # 如果 bos_token 是字符串，则将其转换为 AddedToken 对象
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串，则将其转换为 AddedToken 对象
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果 unk_token 是字符串，则将其转换为 AddedToken 对象
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则将其转换为 AddedToken 对象
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 设置是否添加 bos_token
        self.add_bos_token = add_bos_token

        # 从词汇文件中读取编码器（encoder）字典
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 从编码器字典生成解码器（decoder）字典
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置解码时的错误处理方式
        self.errors = errors
        # 将字节转换为 Unicode 的编码器和解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 从合并文件中读取 BPE 合并规则
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建 BPE 合并规则的字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存
        self.cache = {}
        # 设置是否在前缀空格前添加空格
        self.add_prefix_space = add_prefix_space

        # 编译用于分词的正则表达式模式
        # 该模式包括对缩写词、词性、数字、标点符号以及空格的匹配
        # 忽略大小写，以便 BPE 合并可以针对缩写词的大写版本进行合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化方法
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表大小
        return len(self.encoder)

    def get_vocab(self):
        # 返回词汇表
        return dict(self.encoder, **self.added_tokens_encoder)
    # 对给定的 token 进行 BPE（字节对编码）处理
    def bpe(self, token):
        # 如果 token 已经在缓存中，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换为元组
        word = tuple(token)
        # 获取 token 的所有可能的字符对
        pairs = get_pairs(word)

        # 如果不存在字符对，则直接返回原始 token
        if not pairs:
            return token

        # 迭代处理字符对，直到无法继续拆分
        while True:
            # 根据字符对的编码频率，选择最小的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果字符对不在 BPE 编码的频率表中，停止拆分
            if bigram not in self.bpe_ranks:
                break
            # 分别获取字符对的两个字符
            first, second = bigram
            new_word = []
            i = 0
            # 遍历 token 的字符，合并相邻的字符对
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到字符对的第一个字符，直接将剩余字符添加到新 token 中
                    new_word.extend(word[i:])
                    break
                else:
                    # 将第一个字符之前的字符添加到新 token 中
                    new_word.extend(word[i:j])
                    i = j

                # 如果当前字符和下一个字符组成字符对，则合并到新 token 中
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则，将当前字符添加到新 token 中
                    new_word.append(word[i])
                    i += 1
            # 更新 token 为新的字符对合并后的结果
            new_word = tuple(new_word)
            word = new_word
            # 如果新 token 的长度为 1，则停止拆分
            if len(word) == 1:
                break
            else:
                # 否则，继续获取新的字符对
                pairs = get_pairs(word)
        # 将 token 转换为字符串形式
        word = " ".join(word)
        # 将处理后的 token 添加到缓存中
        self.cache[token] = word
        # 返回 BPE 处理后的结果
        return word

    # 构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 如果需要添加起始标记
        if self.add_bos_token:
            # 将起始标记添加到列表中
            bos_token_ids = [self.bos_token_id]
        else:
            # 否则，空列表
            bos_token_ids = []

        # 将 token_ids_0 添加到输出列表中
        output = bos_token_ids + token_ids_0

        # 如果有第二个序列
        if token_ids_1 is None:
            # 直接返回输出列表
            return output

        # 如果有第二个序列，将起始标记和 token_ids_1 添加到输出列表中
        return output + bos_token_ids + token_ids_1

    # 获取特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。在使用分词器的 `prepare_for_model` 或 `encode_plus` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def _tokenize(self, text):
        """对字符串进行分词。"""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # 将所有字节映射为 Unicode 字符串，避免 BPE 的控制标记（在我们的情况下是空格）
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """使用词汇表将标记（str）转换为 ID。"""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为标记（str）。"""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """将标记序列（字符串）转换为单个字符串。"""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 保存词汇表到指定目录下，返回文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则输出错误日志并返回空
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 打开词汇文件并写入词汇表内容
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 打开合并文件并写入内容
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历BPE标记和标记索引，按索引排序并写入内容
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # 准备用于分词的文本
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 获取add_prefix_space参数，默认为self.add_prefix_space
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经被分割成单词或需要在文本前加空格，则在文本前加一个空格
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    # 获取默认的聊天模板
    @property
    def default_chat_template(self):
        """
        一个简单的聊天模板，忽略角色信息，仅将消息与EOS标记连接起来。
        """
        # 输出警告信息并返回默认的聊天模板字符串
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
```