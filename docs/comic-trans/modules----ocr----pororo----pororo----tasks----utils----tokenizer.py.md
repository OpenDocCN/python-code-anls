# `.\comic-translate\modules\ocr\pororo\pororo\tasks\utils\tokenizer.py`

```py
from typing import List, Optional, Union  # 引入需要使用的类型提示

from tokenizers import Tokenizer, decoders, pre_tokenizers  # 导入 Tokenizer 相关模块
from tokenizers.implementations import BaseTokenizer  # 导入基础的 Tokenizer 类
from tokenizers.models import BPE, Unigram  # 导入 BPE 和 Unigram 模型
from tokenizers.normalizers import NFKC  # 导入 NFKC 规范化器


class CustomTokenizer(BaseTokenizer):  # 自定义的 Tokenizer 继承自 BaseTokenizer

    def __init__(
        self,
        vocab: Union[str, List],  # 词汇表可以是字符串或字符串列表
        merges: Union[str, None],  # 合并文件的路径，可以是字符串或空值
        unk_token: str = "<unk>",  # 未知标记的默认值为 "<unk>"
        replacement: str = "▁",  # 替换标记的默认值为 "▁"
        add_prefix_space: bool = True,  # 是否在词前添加空格，默认为 True
        dropout: Optional[float] = None,  # 可选的 dropout 参数，默认为 None
        normalize: bool = True,  # 是否进行规范化，默认为 True
    ):
        if merges:
            n_model = "BPE"  # 如果提供了合并文件，则使用 BPE 模型
            tokenizer = Tokenizer(
                BPE(
                    vocab,  # BPE 模型的词汇表
                    merges,  # BPE 模型的合并文件
                    unk_token=unk_token,  # 未知标记
                    fuse_unk=True,  # 是否将未知标记融合为一个特殊标记
                ))
        else:
            n_model = "Unigram"  # 如果没有提供合并文件，则使用 Unigram 模型
            tokenizer = Tokenizer(Unigram(vocab, 1))  # 创建 Unigram 模型的 Tokenizer 对象

        if normalize:
            tokenizer.normalizer = NFKC()  # 如果需要规范化，则使用 NFKC 规范化器

        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement=replacement,  # 使用 Metaspace 预处理器，设置替换标记和是否添加前缀空格
            add_prefix_space=add_prefix_space,
        )

        tokenizer.decoder = decoders.Metaspace(
            replacement=replacement,  # 使用 Metaspace 解码器，设置替换标记和是否添加前缀空格
            add_prefix_space=add_prefix_space,
        )

        parameters = {
            "model": f"SentencePiece{n_model}",  # 模型名称，根据 n_model 决定是 SentencePieceBPE 还是 SentencePieceUnigram
            "unk_token": unk_token,  # 未知标记
            "replacement": replacement,  # 替换标记
            "add_prefix_space": add_prefix_space,  # 是否添加前缀空格
            "dropout": dropout,  # dropout 参数
        }
        super().__init__(tokenizer, parameters)  # 调用父类的构造函数初始化 Tokenizer

    @staticmethod
    def from_file(
        vocab_filename: str,  # 词汇文件的路径
        merges_filename: Union[str, None],  # 合并文件的路径，可以是字符串或空值
        **kwargs,  # 其他参数
    ):
        # BPE 模型
        if merges_filename:
            vocab, merges = BPE.read_file(vocab_filename, merges_filename)  # 从文件中读取 BPE 模型的词汇表和合并文件

        # Unigram 模型
        else:
            vocab = []  # 初始化词汇表为空列表
            merges = None  # 合并文件为空
            with open(vocab_filename, "r") as f_in:
                for line in f_in.readlines():
                    token, score = line.strip().split("\t")
                    vocab.append((token, float(score)))  # 从文件中读取 Unigram 模型的词汇表

        return CustomTokenizer(vocab, merges, **kwargs)  # 返回一个自定义 Tokenizer 对象

    def segment(self, text: str) -> List[str]:
        """
        Segment text into subword list

        Args:
            text (str): input text to be segmented

        Returns:
            List[str]: segmented subword list

        """
        encoding = self.encode(text)  # 使用 Tokenizer 对文本进行编码

        offsets = encoding.offsets  # 获取编码后的偏移量
        tokens = encoding.tokens  # 获取编码后的标记列表

        result = []
        for offset, token in zip(offsets, tokens):
            if token != "<unk>":
                result.append(token)  # 如果标记不是 "<unk>"，直接添加到结果列表中
                continue
            s, e = offset
            result.append(text[s:e])  # 如果标记是 "<unk>"，根据偏移量从原文本中截取子串并添加到结果列表中
        return result  # 返回分段后的子词列表
```