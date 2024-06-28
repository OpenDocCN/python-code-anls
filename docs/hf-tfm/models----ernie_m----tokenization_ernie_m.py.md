# `.\models\ernie_m\tokenization_ernie_m.py`

```py
# coding=utf-8
# 上面的注释声明了文件的编码格式为 UTF-8，并非代码实际操作，仅为信息说明

# 版权声明，指出该文件的版权归属于 Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang 以及 HuggingFace Inc. 团队所有
# 版权声明的文本通常包括对软件使用的限制和许可，这里声明使用 Apache License, Version 2.0，详细信息可在指定网址查看
# http://www.apache.org/licenses/LICENSE-2.0

# 导入所需的模块和类
import io
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 模块，用于分词
import sentencepiece as spm

# 从 HuggingFace 库中导入必要的类和函数
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 定义常量 SPIECE_UNDERLINE，用于表示子词之间的连接符
SPIECE_UNDERLINE = "▁"

# 定义词汇文件的默认名称映射，包括词汇文件和 sentencepiece 模型文件
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "sentencepiece_model_ckpt": "sentencepiece.bpe.model"}

# 定义资源文件的默认名称映射，包括 sentencepiece 模型文件和词汇文件
RESOURCE_FILES_NAMES = {
    "sentencepiece_model_file": "sentencepiece.bpe.model",
    "vocab_file": "vocab.txt",
}

# 预训练模型的词汇文件映射，根据模型名称指定对应的下载链接
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "ernie-m-base": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/vocab.txt",
        "ernie-m-large": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/vocab.txt",
    },
    "sentencepiece_model_file": {
        "ernie-m-base": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/sentencepiece.bpe.model",
        "ernie-m-large": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/sentencepiece.bpe.model",
    },
}

# 预训练模型的位置嵌入大小映射，根据模型名称指定对应的嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "ernie-m-base": 514,
    "ernie-m-large": 514,
}

# 预训练模型的初始化配置映射，根据模型名称指定对应的初始化参数
PRETRAINED_INIT_CONFIGURATION = {
    "ernie-m-base": {"do_lower_case": False},
    "ernie-m-large": {"do_lower_case": False},
}

# 以下是一个类的定义，用于构建 Ernie-M 的分词器
class ErnieMTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a Ernie-M tokenizer. It uses the `sentencepiece` tools to cut the words to sub-words.
    """
    # 这里是类的初始化方法和构造函数，用于初始化一个 Ernie-M 分词器对象
    def __init__(
        self,
        # 指定分词器的初始化参数
        vocab_file: Optional[str] = None,
        sentencepiece_model_file: Optional[str] = None,
        do_lower_case=False,
        **kwargs
    ):
        """
        :param vocab_file: 词汇文件的路径（可选）
        :param sentencepiece_model_file: sentencepiece 模型文件的路径（可选）
        :param do_lower_case: 是否将所有输入转换为小写（默认为 False）
        """
        # 调用父类 PreTrainedTokenizer 的构造函数，初始化分词器
        super().__init__(
            # 指定初始化参数
            vocab_file=vocab_file,
            sentencepiece_model_file=sentencepiece_model_file,
            do_lower_case=do_lower_case,
            **kwargs
        )
    # 类描述了用于构建和初始化Ernie-M模型所需的参数与配置, 包括预训练语言模型的超参数和初始化工具.
    """
    Args:
        sentencepiece_model_ckpt (`str`):
            某句段语法模型检查点的路径, 用于序列到序列的编码和解码任务.
    
        vocab_file (`str`, *optional*):
            字典文件路径, 若未提供则继承默认词汇表.
    
        do_lower_case (`str`, *optional*, defaults to `True`):
            是否将输入文本转换为小写, 当在数据预处理阶段处理文本时启用.
    
        encoding (`str`, *optional*, defaults to `utf8`):
            编码方式, 默认使用UTF-8用于解析输入数据.
    
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知词汇(外域词汇)的标记, 用于替换未在词汇表中的词汇.
    
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            用于分隔不同句子在同一批文本序列中.
    
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充序列, 使所有序列长度相等适用于批处理.
    
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器的标志符, 表示序列开始的典型符号.
    
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于替换的标记符号, 该模型将其视为需要预测原始未掩码的令牌的例证.
    
        sp_model_kwargs: `Optional[Dict[str, Any]]` = None:
            用于初始化句段模型的可选参数字典.
    
        kwargs:
            其他可能的初始化参数, 用于扩展上述参数的功能.
    """
    
    # 定义用于Ernie-M模型关键输入名称的列表.
    model_input_names: List[str] = ["input_ids"]
    
    # 载入预训练配置集合与需要提供的初始配置工具
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
    # 载入预先构建的词汇表文件映射表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    
    # 载入额外资源文件的定义与配置
    resource_files_names = RESOURCE_FILES_NAMES
    
    # 构建模型实例以初始化, 包括指定的参数和可能的额外配置项.
    def __init__(
        self,
        sentencepiece_model_ckpt,
        vocab_file=None,
        do_lower_case=False,
        encoding="utf8",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        pass
    ) -> None:
        # 定义一个初始化方法，接受多个参数，用于初始化对象的各种属性和参数

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 如果 sp_model_kwargs 为 None，则将其设置为空字典，否则使用传入的 sp_model_kwargs

        self.do_lower_case = do_lower_case
        # 初始化一个属性 do_lower_case，表示是否进行小写处理

        self.sentencepiece_model_ckpt = sentencepiece_model_ckpt
        # 初始化 sentencepiece_model_ckpt 属性，表示 SentencePiece 模型的检查点路径

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 使用 sp_model_kwargs 初始化 SentencePieceProcessor 对象，并赋值给 sp_model 属性

        self.sp_model.Load(sentencepiece_model_ckpt)
        # 载入 SentencePiece 模型的检查点文件

        # 模仿 paddlenlp.transformers.ernie_m.tokenizer.ErnieMTokenizer 的功能
        if vocab_file is not None:
            self.vocab = self.load_vocab(filepath=vocab_file)
            # 如果提供了 vocab_file 参数，则调用 load_vocab 方法加载词汇表
        else:
            self.vocab = {self.sp_model.id_to_piece(id): id for id in range(self.sp_model.get_piece_size())}
            # 否则，根据 SentencePiece 模型的大小构建词汇表，使用 id_to_piece 方法和 get_piece_size 方法

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        # 创建反向词汇表，将 id 映射到词汇

        super().__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            vocab_file=vocab_file,
            encoding=encoding,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )
        # 调用父类的初始化方法，传递相关参数和关键字参数

    def get_offset_mapping(self, text):
        if text is None:
            return None
        # 如果文本为空，则返回 None

        split_tokens = self.tokenize(text)
        # 使用当前对象的 tokenize 方法对文本进行分词，得到分词后的列表 split_tokens

        normalized_text, char_mapping = "", []
        # 初始化 normalized_text 和 char_mapping

        for i, ch in enumerate(text):
            if ch in self.SP_CHAR_MAPPING:
                ch = self.SP_CHAR_MAPPING.get(ch)
                # 如果字符在 SP_CHAR_MAPPING 中，使用映射后的字符替换原字符
            else:
                ch = unicodedata.normalize("NFKC", ch)
                # 否则，使用 NFKC 规范化处理字符

            if self.is_whitespace(ch):
                continue
            # 如果字符是空白字符，则跳过

            normalized_text += ch
            # 将处理后的字符追加到 normalized_text 中
            char_mapping.extend([i] * len(ch))
            # 根据字符长度，将相应索引追加到 char_mapping 中

        text, token_mapping, offset = normalized_text, [], 0
        # 将处理后的文本赋值给 text，初始化 token_mapping 和 offset

        if self.do_lower_case:
            text = text.lower()
            # 如果需要进行小写处理，则将文本转换为小写

        for token in split_tokens:
            if token[:1] == "▁":
                token = token[1:]
                # 如果 token 以 "▁" 开头，去除 "▁"
            start = text[offset:].index(token) + offset
            # 找到 token 在 text 中的起始索引
            end = start + len(token)
            # 计算 token 在 text 中的结束索引

            token_mapping.append((char_mapping[start], char_mapping[end - 1] + 1))
            # 将 token 的字符映射加入 token_mapping 中
            offset = end
            # 更新 offset

        return token_mapping
        # 返回字符映射列表

    @property
    def vocab_size(self):
        return len(self.vocab)
        # 返回词汇表大小

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)
        # 返回词汇表及额外的 token 编码器

    def __getstate__(self):
        state = self.__dict__.copy()
        # 复制对象的属性字典到 state 中
        state["sp_model"] = None
        # 将 sp_model 设置为 None
        return state
        # 返回对象的状态字典

    def __setstate__(self, d):
        self.__dict__ = d
        # 将状态字典 d 中的内容赋值给对象的属性字典

        # 用于向后兼容
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
            # 如果对象中不存在 sp_model_kwargs 属性，则设置为空字典

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 使用 sp_model_kwargs 初始化 SentencePieceProcessor 对象，并赋值给 sp_model 属性
        self.sp_model.Load(self.sentencepiece_model_ckpt)
        # 载入 SentencePiece 模型的检查点文件
    # 对文本进行清洗，去除无效字符并清理空白
    def clean_text(self, text):
        return "".join((self.SP_CHAR_MAPPING.get(c, c) for c in text))

    # 对字符串进行分词处理
    def _tokenize(self, text, enable_sampling=False, nbest_size=64, alpha=0.1):
        """Tokenize a string."""

        # 如果在参数中启用了采样，则将 enable_sampling 设置为 True
        if self.sp_model_kwargs.get("enable_sampling") is True:
            enable_sampling = True
        # 如果参数中指定了 alpha，则使用参数中的值
        if self.sp_model_kwargs.get("alpha") is not None:
            alpha = self.sp_model_kwargs.get("alpha")
        # 如果参数中指定了 nbest_size，则使用参数中的值
        if self.sp_model_kwargs.get("nbest_size") is not None:
            nbest_size = self.sp_model_kwargs.get("nbest_size")

        # 根据是否启用采样来选择使用 EncodeAsPieces 还是 SampleEncodeAsPieces 方法
        if not enable_sampling:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, nbest_size, alpha)
        
        new_pieces = []
        # 遍历分词后的片段
        for pi, piece in enumerate(pieces):
            # 处理特殊标记 SPIECE_UNDERLINE
            if piece == SPIECE_UNDERLINE:
                # 如果当前标记是 SPIECE_UNDERLINE 且下一个标记不以 SPIECE_UNDERLINE 开头且不是第一个标记，则添加 SPIECE_UNDERLINE
                if not pieces[pi + 1].startswith(SPIECE_UNDERLINE) and pi != 0:
                    new_pieces.append(SPIECE_UNDERLINE)
                    continue
                else:
                    continue
            lst_i = 0
            # 遍历当前片段中的每个字符
            for i, chunk in enumerate(piece):
                # 跳过 SPIECE_UNDERLINE
                if chunk == SPIECE_UNDERLINE:
                    continue
                # 判断字符是否为中文字符或标点符号
                if self.is_ch_char(chunk) or self.is_punct(chunk):
                    # 如果当前字符不是第一个且前一个字符不是 SPIECE_UNDERLINE，则添加前面的部分到 new_pieces
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    # 添加当前字符到 new_pieces
                    new_pieces.append(chunk)
                    # 更新 lst_i 为当前索引加 1
                    lst_i = i + 1
                # 如果字符是数字且不是第一个，并且前一个字符不是数字，则添加前面的部分到 new_pieces
                elif chunk.isdigit() and i > 0 and not piece[i - 1].isdigit():
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    lst_i = i
                # 如果字符不是数字且不是第一个，并且前一个字符是数字，则添加前面的部分到 new_pieces
                elif not chunk.isdigit() and i > 0 and piece[i - 1].isdigit():
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    lst_i = i
            # 如果片段长度大于 lst_i，则添加剩余部分到 new_pieces
            if len(piece) > lst_i:
                new_pieces.append(piece[lst_i:])
        
        # 返回处理后的片段列表
        return new_pieces

    # 将分词后的 tokens 列表转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    # 将 token ids 转换为单个字符串
    def convert_ids_to_string(self, ids):
        # 将 token ids 转换为 tokens
        tokens = self.convert_ids_to_tokens(ids)
        # 将 tokens 列表转换为单个字符串，并替换 SPIECE_UNDERLINE
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    # 模仿 paddlenlp.transformers.ernie_m.tokenizer.ErnieMTokenizer 的功能，将 token 转换为其对应的 id
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 模仿 paddlenlp.transformers.ernie_m.tokenizer.ErnieMTokenizer 的功能
    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) into a token (str) using the vocabulary.
        
        Args:
            index (int): Index to convert into a token.
        
        Returns:
            str: The corresponding token if found in the vocabulary, otherwise returns the unknown token (self.unk_token).
        """
        # 使用反向词汇表将索引转换为对应的标记，如果索引不存在则返回未知标记
        return self.reverse_vocab.get(index, self.unk_token)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        构建用于序列分类任务的模型输入，通过连接和添加特殊标记。ErnieM 序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] [SEP] B [SEP]`

        Args:
            token_ids_0 (List[int]): 要添加特殊标记的 ID 列表。
            token_ids_1 (List[int], optional): 第二个序列的 ID 列表（可选）。

        Returns:
            List[int]: 包含适当特殊标记的输入 ID 列表。
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self, offset_mapping_0, offset_mapping_1=None):
        """
        构建偏移映射，通过连接和添加特殊标记的偏移量。Ernie-M 偏移映射的格式如下：

        - 单个序列：`(0,0) X (0,0)`
        - 序列对：`(0,0) A (0,0) (0,0) B (0,0)`

        Args:
            offset_mapping_ids_0 (List[tuple]): 要添加特殊标记的字符偏移列表。
            offset_mapping_ids_1 (List[tuple], optional): 第二个序列的单词片段偏移列表（可选）。

        Returns:
            List[tuple]: 包含适当特殊标记偏移量的单词片段偏移列表。
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0), (0, 0)] + offset_mapping_1 + [(0, 0)]
    # 检查给定的 token_ids_0 是否已经包含特殊标记，如果是则返回特殊标记掩码
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        r"""
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `encode` method.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`str`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`:
                The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            # 如果 token_ids_0 已经包含了特殊标记，并且 token_ids_1 也被提供了，则抛出 ValueError
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            # 返回 token_ids_0 中特殊标记的掩码
            return [1 if x in [self.sep_token_id, self.cls_token_id] else 0 for x in token_ids_0]

        if token_ids_1 is not None:
            # 如果 token_ids_1 存在，则创建包含特殊标记的掩码列表，用于处理序列对
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]
        # 否则，仅处理 token_ids_0，返回包含特殊标记的掩码列表
        return [1] + ([0] * len(token_ids_0)) + [1]

    # 根据传入的 token_ids_0 和 token_ids_1 创建对应的 token 类型 ID 列表
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids) Should be overridden in a subclass if the model has a special way of
        building: those.

        Args:
            token_ids_0 (`List[int]`):
                The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*):
                The second tokenized sequence.
        Returns:
            `List[int]`: The token type ids.
        """
        # 当 `add_special_tokens` 为 True 时调用，因此需要与 `build_inputs_with_special_tokens` 方法对齐
        if token_ids_1 is None:
            # [CLS] X [SEP] 的序列，对应的 token 类型 ID 全为 0
            return (len(token_ids_0) + 2) * [0]

        # [CLS] A [SEP] [SEP] B [SEP] 的序列，构建对应的 token 类型 ID 列表
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 3)

    # 检查字符是否为中文字符
    def is_ch_char(self, char):
        """
        is_ch_char
        """
        if "\u4e00" <= char <= "\u9fff":
            return True
        return False

    # 检查字符是否为字母
    def is_alpha(self, char):
        """
        is_alpha
        """
        if ("a" <= char <= "z") or ("A" <= char <= "Z"):
            return True
        return False

    # 检查字符是否为标点符号
    def is_punct(self, char):
        """
        is_punct
        """
        if char in ",;:.?!~，；：。？！《》【】":
            return True
        return False
    # 判断字符是否为空白字符
    def is_whitespace(self, char):
        """
        is whitespace
        """
        # 检查字符是否为空格、制表符、换行符或回车符
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        # 如果字符长度为1，则使用 unicodedata 模块检查其分类是否为 Zs（空格分隔符）
        if len(char) == 1:
            cat = unicodedata.category(char)
            if cat == "Zs":
                return True
        # 若以上条件都不满足，则返回 False，表示字符不是空白字符
        return False

    # 加载词汇表文件，并返回 token 到索引的映射字典
    def load_vocab(self, filepath):
        token_to_idx = {}
        # 使用 utf-8 编码打开文件，并逐行读取
        with io.open(filepath, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                # 去除行尾的换行符，并将 token 作为键，索引作为值存入字典
                token = line.rstrip("\n")
                token_to_idx[token] = int(index)
        # 返回 token 到索引的映射字典
        return token_to_idx

    # 将词汇表保存到指定目录，返回保存的文件路径元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        # 检查保存目录是否存在，若不存在则直接使用保存文件名
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 使用 utf-8 编码打开 vocab_file 文件，并写入词汇表
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 按照词汇表中的索引顺序排序词汇，并写入文件
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果发现索引不连续，记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将 token 写入文件，并在末尾添加换行符
                writer.write(token + "\n")
                index += 1

        # 将 tokenizer 模型保存为二进制文件
        tokenizer_model_file = os.path.join(save_directory, "sentencepiece.bpe.model")
        with open(tokenizer_model_file, "wb") as fi:
            # 获取序列化的 tokenizer 模型，并写入文件
            content_spiece_model = self.sp_model.serialized_model_proto()
            fi.write(content_spiece_model)

        # 返回保存的词汇表文件路径的元组
        return (vocab_file,)
```