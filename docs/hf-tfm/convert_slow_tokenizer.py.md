# `.\convert_slow_tokenizer.py`

```py
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to convert slow tokenizers in their fast tokenizers counterparts.

All the conversions are grouped here to gather SentencePiece dependencies outside of the fast tokenizers files and
allow to make our dependency on SentencePiece optional.
"""

import warnings
from typing import Dict, List, Tuple

from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece

from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR


def import_protobuf(error_message=""):
    # 检查是否可以导入 protobuf 库
    if is_protobuf_available():
        import google.protobuf

        # 如果 protobuf 版本低于 4.0.0，则使用旧版的 sentencepiece_model_pb2
        if version.parse(google.protobuf.__version__) < version.parse("4.0.0"):
            from transformers.utils import sentencepiece_model_pb2
        else:
            from transformers.utils import sentencepiece_model_pb2_new as sentencepiece_model_pb2

        # 返回 sentencepiece_model_pb2 模块
        return sentencepiece_model_pb2
    else:
        # 如果无法导入 protobuf，则抛出 ImportError 异常
        raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        # 检查是否已经导入 sentencepiece 库，如果没有则引发异常
        requires_backends(self, "sentencepiece")
        # 导入 SentencePieceProcessor 类
        from sentencepiece import SentencePieceProcessor

        # 创建 SentencePieceProcessor 实例并加载模型
        self.sp = SentencePieceProcessor()
        self.sp.Load(model)
    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        # 获取 SentencePiece 对象的实例
        sp = self.sp
        # 创建一个字典，将每个索引映射到对应的 Piece（词汇）
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        # 根据传入的 vocab_scores 是否为 None 来决定使用哪种排序方式
        if vocab_scores is not None:
            # 如果 vocab_scores 不为 None，则将其转换为字典并设置 reverse 为 True
            vocab_scores, reverse = dict(vocab_scores), True
        else:
            # 如果 vocab_scores 为 None，则使用默认的 vocab，并将 reverse 设置为 False
            vocab_scores, reverse = vocab, False

        # Merges（合并操作）
        merges = []
        # 遍历 vocab_scores 中的每个 merge 和对应的 piece_score
        for merge, piece_score in vocab_scores.items():
            local = []
            # 将 merge 分解为 piece_l 和 piece_r 的组合，并检查其在 vocab 中是否存在
            for index in range(1, len(merge)):
                piece_l, piece_r = merge[:index], merge[index:]
                if piece_l in vocab and piece_r in vocab:
                    local.append((piece_l, piece_r, piece_score))
            # 对 local 按照 vocab 中 piece_l 和 piece_r 的索引排序
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            # 将 local 的内容扩展到 merges 列表中
            merges.extend(local)

        # 按照 piece_score 进行降序排序，并转换为 (piece_l, piece_r) 的形式
        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
        merges = [(val[0], val[1]) for val in merges]
        # 返回 vocab 和 merges
        return vocab, merges
class GemmaSentencePieceExtractor(SentencePieceExtractor):
    # GemmaSentencePieceExtractor 类继承自 SentencePieceExtractor，用于实现定制的 SentencePiece 提取器

    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        # extract 方法用于从 SentencePiece 模型中提取词汇表和合并列表
        sp = self.sp  # 获取 SentencePiece 对象
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}
        # 根据索引从 SentencePiece 对象中获取词汇，并创建词汇到索引的映射字典

        # 补充一个缺失的特殊词汇 "<0x09>" 作为 "\t" 的字节回退表示
        vocab["\t"] = vocab.pop("<0x09>")
        
        if vocab_scores is not None:
            vocab_scores, reverse = dict(vocab_scores), True
        else:
            vocab_scores, reverse = vocab, False

        # Merges
        merges = []
        for merge, piece_score in vocab_scores.items():
            local = []
            for index in range(1, len(merge)):
                piece_l, piece_r = merge[:index], merge[index:]
                if piece_l in vocab and piece_r in vocab:
                    local.append((piece_l, piece_r, piece_score))
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            merges.extend(local)

        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
        merges = [(val[0], val[1]) for val in merges]
        return vocab, merges
        # 返回提取后的词汇表和合并列表


def check_number_comma(piece: str) -> bool:
    # check_number_comma 函数用于检查给定的字符串 piece 是否满足特定条件（结尾不是逗号或倒数第二个字符不是数字）
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


class Converter:
    # Converter 类用作基类，包含一些基本结构但没有实现具体的转换逻辑

    def __init__(self, original_tokenizer):
        # 初始化方法，接受一个原始的 tokenizer 并存储在实例变量中
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        # converted 方法声明但未实现，用于子类重写实现具体的转换逻辑
        raise NotImplementedError()


class BertConverter(Converter):
    # BertConverter 类继承自 Converter 类，用于实现针对 Bert 模型的具体转换逻辑
    # 定义一个方法，用于将原始的 tokenizer 转换为新的 Tokenizer 对象，并返回
    def converted(self) -> Tokenizer:
        # 获取原始 tokenizer 的词汇表
        vocab = self.original_tokenizer.vocab
        # 使用 WordPiece 模型和未知标记符初始化新的 Tokenizer 对象
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        # 初始化用于标准化文本的参数，默认为 False
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        
        # 如果原始 tokenizer 中包含 basic_tokenizer 属性，则从中获取相关参数值
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 设置新 Tokenizer 对象的标准化器为 BertNormalizer，使用指定的参数
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        
        # 设置新 Tokenizer 对象的预处理器为 BertPreTokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # 从原始 tokenizer 中获取特殊标记的字符串表示和标记 ID
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 设置新 Tokenizer 对象的后处理器为 TemplateProcessing，指定单句和双句处理模板及特殊标记
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )

        # 设置新 Tokenizer 对象的解码器为 WordPiece 解码器，前缀为 '##'
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # 返回配置好的新 Tokenizer 对象
        return tokenizer
# 定义一个名为 SplinterConverter 的类，它继承自 Converter 类
class SplinterConverter(Converter):
    
    # 重写父类方法 converted，返回一个 Tokenizer 对象
    def converted(self) -> Tokenizer:
        
        # 获取原始分词器的词汇表
        vocab = self.original_tokenizer.vocab
        
        # 使用 WordPiece 模型和未知标记初始化 Tokenizer 对象
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))
        
        # 初始化变量用于存储是否分词中文字符、是否去除重音符号、是否小写化的标志
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        
        # 如果原始分词器具有 basic_tokenizer 属性，获取其相应的属性值
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case
        
        # 设置 Tokenizer 的 normalizer 为 BertNormalizer 对象，配置各项参数
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        
        # 设置 Tokenizer 的 pre_tokenizer 为 BertPreTokenizer 对象
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        
        # 获取特殊标记（如 CLS、SEP、QUESTION 和 DOT）的字符串形式和对应的标记 ID
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        question = str(self.original_tokenizer.question_token)
        dot = "."
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id
        question_token_id = self.original_tokenizer.question_token_id
        
        # 使用原始分词器将 DOT 转换为其对应的标记 ID
        dot_token_id = self.original_tokenizer.convert_tokens_to_ids(".")
        
        # 根据原始分词器的填充位置确定 pair 的模板字符串
        if self.original_tokenizer.padding_side == "right":
            pair = f"{cls}:0 $A:0 {question} {dot} {sep}:0 $B:1 {sep}:1"
        else:
            pair = f"{cls}:0 $A:0 {sep}:0 $B:1 {question} {dot} {sep}:1"
        
        # 设置 Tokenizer 的 post_processor 为 TemplateProcessing 对象，配置各项参数
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=pair,
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
                (question, question_token_id),
                (dot, dot_token_id),
            ],
        )
        
        # 设置 Tokenizer 的 decoder 为 WordPiece 对象，配置前缀为 "##"
        tokenizer.decoder = decoders.WordPiece(prefix="##")
        
        # 返回配置好的 Tokenizer 对象
        return tokenizer
    # 定义一个方法，用于将原始的分词器转换为新的 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 获取原始分词器的词汇表
        vocab = self.original_tokenizer.vocab
        # 使用 WordPiece 模型和未知标记来初始化 Tokenizer 对象
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        # 初始化用于标准化文本的参数
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        # 检查原始分词器是否有基本分词器属性，设置标志位
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 设置 Tokenizer 对象的标准化器为 BertNormalizer，配置参数包括是否清洗文本、处理中文字符、去除重音、小写化
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # 设置 Tokenizer 对象的预处理器为 BertPreTokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # 获取原始分词器的特殊标记（例如 [CLS] 和 [SEP]）
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 设置 Tokenizer 对象的后处理器为 TemplateProcessing，配置单句和双句模板及特殊标记
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:2 $A:0 {sep}:0",  # token_type_id is 2 for Funnel transformer
            pair=f"{cls}:2 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        # 设置 Tokenizer 对象的解码器为 WordPiece 解码器，前缀为 "##"
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # 返回配置完成的 Tokenizer 对象
        return tokenizer
class MPNetConverter(Converter):
    # MPNetConverter 类继承自 Converter 类，用于将原始 tokenizer 转换为 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 获取原始 tokenizer 的词汇表
        vocab = self.original_tokenizer.vocab
        # 创建一个 Tokenizer 对象，使用 WordPiece 模型，设置未知标记为原始 tokenizer 的未知标记
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        # 初始化一些变量用于记录是否执行特定的文本清洗和处理操作
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        
        # 检查原始 tokenizer 是否具有 basic_tokenizer 属性，如果有，则更新相关变量
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 设置 tokenizer 的文本清洗器为 BertNormalizer，配置其参数
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # 设置 tokenizer 的预处理器为 BertPreTokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # 获取特殊标记的字符串形式
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 设置 tokenizer 的后处理器为 TemplateProcessing，配置其参数
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 {sep}:0 $B:1 {sep}:1",  # MPNet 使用两个 [SEP] 标记
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        # 设置 tokenizer 的解码器为 WordPiece 解码器，前缀为 "##"
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # 返回转换后的 Tokenizer 对象
        return tokenizer


class OpenAIGPTConverter(Converter):
    # OpenAIGPTConverter 类继承自 Converter 类，用于将原始 tokenizer 转换为 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 获取原始 tokenizer 的编码器和 BPE 合并列表
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        unk_token = self.original_tokenizer.unk_token

        # 创建一个 Tokenizer 对象，使用 BPE 模型，设置未知标记为原始 tokenizer 的未知标记
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                unk_token=str(unk_token),
                end_of_word_suffix="</w>",
                fuse_unk=False,
            )
        )

        # 如果 tokenizer 中已经包含原始 tokenizer 的未知标记，则添加到特殊标记列表中
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])

        # 设置 tokenizer 的文本清洗器为 BertNormalizer，只设置小写处理
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        # 设置 tokenizer 的预处理器为 BertPreTokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        # 设置 tokenizer 的解码器为 BPEDecoder，后缀为 "</w>"
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        # 返回转换后的 Tokenizer 对象
        return tokenizer


class GPT2Converter(Converter):
    # GPT2Converter 类继承自 Converter 类，用于将原始 tokenizer 转换为 Tokenizer 对象
    # 定义一个方法 converted，返回类型为 Tokenizer
    def converted(self) -> Tokenizer:
        # 获取原始分词器的词汇表
        vocab = self.original_tokenizer.encoder
        # 获取原始分词器的 BPE 合并列表
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        # 创建一个新的 Tokenizer 对象，使用 BPE 分词器
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,  # 设置词汇表
                merges=merges,  # 设置合并列表
                dropout=None,  # 没有 dropout
                continuing_subword_prefix="",  # 子词前缀为空字符串
                end_of_word_suffix="",  # 词尾后缀为空字符串
                fuse_unk=False,  # 不融合未知标记
            )
        )

        # 设置 Tokenizer 的预分词器为 ByteLevel，并根据原始分词器设置是否添加前缀空格
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        # 设置 Tokenizer 的解码器为 ByteLevel
        tokenizer.decoder = decoders.ByteLevel()

        # 如果原始分词器设置了开始词头（bos），则设置后处理器为 TemplateProcessing
        if self.original_tokenizer.add_bos_token:
            bos = self.original_tokenizer.bos_token
            bos_token_id = self.original_tokenizer.bos_token_id
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos}:0 $A:0",  # 单句模板，以 bos 开始
                pair=f"{bos}:0 $A:0 $B:1",  # 双句模板，以 bos 开始
                special_tokens=[  # 特殊标记列表，包括 bos 和其对应的 id
                    (bos, bos_token_id),
                ],
            )
        else:
            # 如果没有设置开始词头，设置后处理器为 ByteLevel，trim_offsets=False 表示不修剪偏移量
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        # 返回创建的 Tokenizer 对象
        return tokenizer
# HerbertConverter 类，继承自 Converter 类
class HerbertConverter(Converter):
    # 覆盖了 converted 方法，返回一个 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # tokenizer_info_str 字符串，用于版本信息
        tokenizer_info_str = "#version:"
        # token_suffix 字符串，用于表示词尾
        token_suffix = "</w>"

        # 获取原始分词器的编码器字典
        vocab = self.original_tokenizer.encoder
        # 获取原始分词器的 BPE 合并操作列表
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        
        # 如果 merges 的第一个元素包含 tokenizer_info_str，则从 merges 中移除该元素
        if tokenizer_info_str in merges[0][0]:
            merges = merges[1:]

        # 创建 Tokenizer 对象，使用 BPE 分词器
        tokenizer = Tokenizer(
            BPE(
                vocab,
                merges,
                dropout=None,
                unk_token=self.original_tokenizer.unk_token,
                end_of_word_suffix=token_suffix,
            )
        )

        # 设置 Tokenizer 对象的正规化器为 BertNormalizer
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False)
        # 设置 Tokenizer 对象的预处理器为 BertPreTokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        # 设置 Tokenizer 对象的解码器为 BPEDecoder，指定词尾后缀
        tokenizer.decoder = decoders.BPEDecoder(suffix=token_suffix)
        # 设置 Tokenizer 对象的后处理器为 BertProcessing，指定特殊标记
        tokenizer.post_processor = processors.BertProcessing(
            sep=(self.original_tokenizer.sep_token, self.original_tokenizer.sep_token_id),
            cls=(self.original_tokenizer.cls_token, self.original_tokenizer.cls_token_id),
        )

        # 返回创建的 Tokenizer 对象
        return tokenizer


# Qwen2Converter 类，继承自 Converter 类
class Qwen2Converter(Converter):
    # 覆盖了 converted 方法，返回一个 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 获取原始分词器的编码器字典
        vocab = self.original_tokenizer.encoder
        # 获取原始分词器的 BPE 合并操作列表
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        # 创建 Tokenizer 对象，使用 BPE 分词器
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )

        # 设置 Tokenizer 对象的正规化器为 NFC（Unicode 标准化）
        tokenizer.normalizer = normalizers.NFC()

        # 设置 Tokenizer 对象的预处理器为 Sequence，包含两个预处理步骤
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                # 第一个预处理步骤：使用正则表达式拆分，匹配单词和数字，以及特定标点
                pre_tokenizers.Split(
                    Regex(
                        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
                    ),
                    behavior="isolated",
                    invert=False,
                ),
                # 第二个预处理步骤：使用 ByteLevel 拆分字节级别的处理
                pre_tokenizers.ByteLevel(
                    add_prefix_space=getattr(self.original_tokenizer, "add_prefix_space", False),
                    use_regex=False,
                ),
            ]
        )

        # 设置 Tokenizer 对象的解码器为 ByteLevel
        tokenizer.decoder = decoders.ByteLevel()
        # 设置 Tokenizer 对象的后处理器为 ByteLevel，不修剪偏移量
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # 返回创建的 Tokenizer 对象
        return tokenizer


# RobertaConverter 类，继承自 Converter 类，该部分代码尚未提供完整
    # 定义一个方法 `converted`，返回一个 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 获取原始的分词器对象
        ot = self.original_tokenizer
        # 获取原始分词器的词汇表
        vocab = ot.encoder
        # 获取原始分词器的合并列表
        merges = list(ot.bpe_ranks.keys())

        # 创建一个新的 Tokenizer 对象，使用 BPE 分词器
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,  # 设置词汇表
                merges=merges,  # 设置合并列表
                dropout=None,  # 不使用 dropout
                continuing_subword_prefix="",  # 设置持续子词前缀为空字符串
                end_of_word_suffix="",  # 设置词尾后缀为空字符串
                fuse_unk=False,  # 不融合未知标记
            )
        )

        # 设置 Tokenizer 的预分词器为 ByteLevel，并保留原始分词器的前缀空格设置
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        # 设置 Tokenizer 的解码器为 ByteLevel
        tokenizer.decoder = decoders.ByteLevel()
        # 设置 Tokenizer 的后处理器为 RobertaProcessing，设置分隔符、CLS和SEP标记及其ID，同时修剪偏移量
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(ot.sep_token, ot.sep_token_id),  # 设置分隔符及其ID
            cls=(ot.cls_token, ot.cls_token_id),  # 设置CLS标记及其ID
            add_prefix_space=ot.add_prefix_space,  # 保留原始分词器的前缀空格设置
            trim_offsets=True,  # 在Roberta上默认为True（历史遗留）
        )

        # 返回配置好的 Tokenizer 对象
        return tokenizer
class RoFormerConverter(Converter):
    # RoFormerConverter 类继承自 Converter 类，用于转换器功能
    def converted(self) -> Tokenizer:
        # 返回类型为 Tokenizer 的 converted 方法
        from .models.roformer.tokenization_utils import JiebaPreTokenizer

        # 导入 RoFormer 所需的 JiebaPreTokenizer

        # 获取原始分词器的词汇表
        vocab = self.original_tokenizer.vocab
        # 创建一个 Tokenizer 实例，使用 WordPiece 方法和未知标记
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        # 初始化 strip_accents 和 do_lower_case 为 False
        strip_accents = False
        do_lower_case = False
        # 如果原始分词器具有 basic_tokenizer 属性
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            # 获取 strip_accents 和 do_lower_case 的设置
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 设置 tokenizer 的 normalizer 为 BertNormalizer 实例
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # 设置 tokenizer 的 pre_tokenizer 为 JiebaPreTokenizer 实例
        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(JiebaPreTokenizer(vocab))

        # 获取 cls 和 sep 的字符串表示
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        # 获取 cls_token_id 和 sep_token_id
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 设置 tokenizer 的 post_processor 为 TemplateProcessing 实例
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        # 设置 tokenizer 的 decoder 为 WordPiece 实例，前缀为 "##"
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # 返回设置好的 tokenizer 实例
        return tokenizer


class DebertaConverter(Converter):
    # DebertaConverter 类继承自 Converter 类，用于转换器功能
    def converted(self) -> Tokenizer:
        # 返回类型为 Tokenizer 的 converted 方法
        ot = self.original_tokenizer
        # 获取原始分词器的 encoder 和 bpe_ranks

        # 创建一个 Tokenizer 实例，使用 BPE 方法和给定的参数
        tokenizer = Tokenizer(
            BPE(
                vocab=ot.encoder,
                merges=list(ot.bpe_ranks.keys()),
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        # 设置 tokenizer 的 pre_tokenizer 为 ByteLevel 实例
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        # 设置 tokenizer 的 decoder 为 ByteLevel 实例
        tokenizer.decoder = decoders.ByteLevel()
        # 设置 tokenizer 的 post_processor 为 TemplateProcessing 实例
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )

        # 返回设置好的 tokenizer 实例
        return tokenizer


class SpmConverter(Converter):
    # SpmConverter 类继承自 Converter 类，用于转换器功能
    # 初始化方法，接受任意数量参数
    def __init__(self, *args):
        # 检查是否需要后端支持protobuf，如果不支持则抛出异常
        requires_backends(self, "protobuf")

        # 调用父类的初始化方法，传入所有参数
        super().__init__(*args)

        # 导入protobuf模型，此处调用import_protobuf函数，返回的模型对象赋值给model_pb2
        model_pb2 = import_protobuf()

        # 创建一个新的ModelProto对象m，从self.original_tokenizer.vocab_file文件中解析数据到m对象
        with open(self.original_tokenizer.vocab_file, "rb") as f:
            m.ParseFromString(f.read())

        # 将解析后的m对象赋值给self.proto
        self.proto = m

        # 如果self.proto.trainer_spec.byte_fallback为True，则进行以下处理
        if self.proto.trainer_spec.byte_fallback:
            # 如果没有定义handle_byte_fallback属性，则发出警告
            if not getattr(self, "handle_byte_fallback", None):
                warnings.warn(
                    "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                    " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                    " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                    "unknown tokens into a sequence of byte tokens matching the original piece of text."
                )

    # 返回proto中pieces属性的列表，每个元素为(piece.piece, piece.score)元组
    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    # 返回proto中trainer_spec属性的unk_id
    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    # 根据proto的trainer_spec.model_type选择合适的Tokenizer类型，并返回对应的实例
    def tokenizer(self, proto):
        # 获取model_type值
        model_type = proto.trainer_spec.model_type
        # 获取vocab信息
        vocab_scores = self.vocab(proto)
        # 获取unk_id信息
        unk_id = self.unk_id(proto)

        # 根据model_type的值选择合适的Tokenizer类型
        if model_type == 1:
            # 使用Unigram模型创建Tokenizer实例
            tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
        elif model_type == 2:
            # 从self.original_tokenizer.vocab_file中提取_, merges变量
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            # 创建BPE类型的Tokenizer实例，使用给定的参数
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                )
            )
        else:
            # 如果model_type不是1或2，则抛出异常
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        # 返回创建的Tokenizer实例
        return tokenizer

    # 根据proto的normalizer_spec属性返回合适的normalizer对象
    def normalizer(self, proto):
        # 获取precompiled_charsmap信息
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        # 定义_normalizers列表，包含两种normalizers对象
        _normalizers = [
            normalizers.Strip(left=False, right=True),  # 去除空格，保留左侧，右侧去除
            normalizers.Replace(Regex(" {2,}"), "▁"),  # 替换多个空格为特殊字符"▁"
        ]
        # 如果没有预编译字符映射，则返回一个Sequence对象，包含_normalizers中的内容
        if not precompiled_charsmap:
            return normalizers.Sequence(_normalizers)
        else:
            # 否则返回一个Sequence对象，包含precompiled_charsmap映射后的内容和_normalizers
            return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    # 根据replacement和add_prefix_space创建并返回pre_tokenizers.Metaspace对象
    def pre_tokenizer(self, replacement, add_prefix_space):
        # 初始化prepend_scheme为"always"
        prepend_scheme = "always"
        # 如果self.original_tokenizer存在legacy属性且为False，则设置prepend_scheme为"first"
        if hasattr(self.original_tokenizer, "legacy") and not self.original_tokenizer.legacy:
            prepend_scheme = "first"
        # 返回一个Metaspace对象，使用给定的参数
        return pre_tokenizers.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space, prepend_scheme=prepend_scheme
        )
    # 定义一个方法 `post_processor`，返回 `None`
    def post_processor(self):
        return None

    # 定义一个方法 `decoder`，接受 `replacement` 和 `add_prefix_space` 两个参数，返回一个 `decoders.Metaspace` 对象
    def decoder(self, replacement, add_prefix_space):
        return decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

    # 定义一个方法 `converted`，返回一个 `Tokenizer` 对象
    def converted(self) -> Tokenizer:
        # 使用 `self.tokenizer` 类型创建一个 `tokenizer` 对象，使用 `self.proto` 作为参数
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer 组装过程
        # 使用 `self.normalizer` 类型创建一个 `normalizer` 对象，使用 `self.proto` 作为参数
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        # 设置 `replacement` 和 `add_prefix_space` 的默认值
        replacement = "▁"
        add_prefix_space = True

        # 检查 `self.original_tokenizer` 是否有 `add_prefix_space` 属性，更新 `add_prefix_space` 变量
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        # 使用 `self.pre_tokenizer` 类型创建一个 `pre_tokenizer` 对象，使用 `replacement` 和 `add_prefix_space` 作为参数
        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        # 使用 `self.decoder` 方法创建一个 `decoder` 对象，使用 `replacement` 和 `add_prefix_space` 作为参数
        tokenizer.decoder = self.decoder(replacement, add_prefix_space)

        # 调用 `self.post_processor` 方法获取 `post_processor` 对象
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        # 返回最终组装好的 `tokenizer` 对象
        return tokenizer
# AlbertConverter 类，继承自 SpmConverter 类
class AlbertConverter(SpmConverter):
    
    # 重写 vocab 方法，返回一个包含单词片段和分数的列表
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    # 重写 normalizer 方法，返回一个正则化序列对象
    def normalizer(self, proto):
        # 列出要应用的正则化器列表
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        # 如果不保留重音符号，添加相应的正则化器
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        # 如果执行小写化，添加小写化正则化器
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())
        
        # 获取预编译字符映射表
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        
        # 如果存在预编译字符映射表，添加预编译正则化器
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        
        # 添加空格合并的正则化器
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        
        # 返回正则化序列对象
        return normalizers.Sequence(list_normalizers)

    # 重写 post_processor 方法，返回一个模板处理对象
    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


# BarthezConverter 类，继承自 SpmConverter 类
class BarthezConverter(SpmConverter):
    
    # 重写 unk_id 方法，返回未知标记的 ID
    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    # 重写 post_processor 方法，返回一个模板处理对象
    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


# CamembertConverter 类，继承自 SpmConverter 类
class CamembertConverter(SpmConverter):
    
    # 重写 vocab 方法，返回一个词汇表，包含词汇和分数的元组列表
    def vocab(self, proto):
        vocab = [
            ("<s>NOTUSED", 0.0),
            ("<pad>", 0.0),
            ("</s>NOTUSED", 0.0),
            ("<unk>", 0.0),
            ("<unk>NOTUSED", -100),
        ]
        # 将 proto.pieces 中的片段和分数添加到词汇表中
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[1:]]
        # 添加 "<mask>" 到词汇表中
        vocab += [("<mask>", 0.0)]
        return vocab

    # 重写 unk_id 方法，返回未知标记的 ID
    def unk_id(self, proto):
        # 见 vocab 方法中的 unk 位置
        return 3

    # 重写 post_processor 方法，返回一个模板处理对象
    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


# DebertaV2Converter 类，继承自 SpmConverter 类
    # 定义一个预处理器函数，用于生成一个包含预处理器序列的对象
    def pre_tokenizer(self, replacement, add_prefix_space):
        # 初始化一个空列表，用于存储预处理器对象
        list_pretokenizers = []
        # 如果原始分词器支持按标点符号切分，则添加一个按独立标点切分的预处理器
        if self.original_tokenizer.split_by_punct:
            list_pretokenizers.append(pre_tokenizers.Punctuation(behavior="isolated"))
        # 添加一个 Metaspace 预处理器，用于处理元空间
        list_pretokenizers.append(pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space))
        # 返回一个预处理器序列对象，其中包含以上构建的预处理器列表
        return pre_tokenizers.Sequence(list_pretokenizers)

    # 定义一个正则化器函数，用于生成一个包含正则化器序列的对象
    def normalizer(self, proto):
        # 初始化一个空列表，用于存储正则化器对象
        list_normalizers = []
        # 如果原始分词器需要进行小写处理，则添加一个小写化正则化器
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())
        # 添加一个去除空格的正则化器
        list_normalizers.append(normalizers.Strip())

        # 获取预编译字符映射表
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        # 如果存在预编译字符映射表，则添加一个预编译正则化器
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        # 添加一个替换连续空格为单个空格的正则化器
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))

        # 返回一个正则化器序列对象，其中包含以上构建的正则化器列表
        return normalizers.Sequence(list_normalizers)

    # 定义一个后处理器函数，用于生成一个模板处理器对象
    def post_processor(self):
        return processors.TemplateProcessing(
            # 单文本处理模板，用特定标记替换各个部分
            single="[CLS]:0 $A:0 [SEP]:0",
            # 双文本处理模板，用特定标记替换各个部分，包括两个分隔符
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            # 特殊标记的映射，将特殊标记与其在原始分词器中对应的 ID 关联起来
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
# 定义一个名为 MBartConverter 的类，继承自 SpmConverter 类
class MBartConverter(SpmConverter):
    
    # 定义一个名为 vocab 的方法，接收 proto 参数，返回一个词汇表列表
    def vocab(self, proto):
        # 初始化词汇表，包括常见特殊 token 和初始权重
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        # 将 proto 对象中的子词片段（从第四个开始）添加到词汇表中
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # 添加特定语言标识符和对应的初始权重
        vocab += [
            ("ar_AR", 0.0),
            ("cs_CZ", 0.0),
            ("de_DE", 0.0),
            ("en_XX", 0.0),
            ("es_XX", 0.0),
            ("et_EE", 0.0),
            ("fi_FI", 0.0),
            ("fr_XX", 0.0),
            ("gu_IN", 0.0),
            ("hi_IN", 0.0),
            ("it_IT", 0.0),
            ("ja_XX", 0.0),
            ("kk_KZ", 0.0),
            ("ko_KR", 0.0),
            ("lt_LT", 0.0),
            ("lv_LV", 0.0),
            ("my_MM", 0.0),
            ("ne_NP", 0.0),
            ("nl_XX", 0.0),
            ("ro_RO", 0.0),
            ("ru_RU", 0.0),
            ("si_LK", 0.0),
            ("tr_TR", 0.0),
            ("vi_VN", 0.0),
            ("zh_CN", 0.0),
        ]
        # 添加一个特殊的 mask 标识符和初始权重
        vocab += [("<mask>", 0.0)]
        # 返回完整的词汇表
        return vocab
    
    # 定义一个名为 unk_id 的方法，接收 proto 参数，返回未知 token 的 id（这里固定为 3）
    def unk_id(self, proto):
        return 3

    # 定义一个名为 post_processor 的方法，返回一个 TemplateProcessing 的处理器对象
    def post_processor(self):
        return processors.TemplateProcessing(
            # 单文本模板，使用 $A 作为占位符，并以 "</s> en_XX" 结尾
            single="$A </s> en_XX",
            # 双文本模板，使用 $A 和 $B 作为占位符，并以 "</s> en_XX" 结尾
            pair="$A $B </s> en_XX",
            # 特殊标记列表，包括 en_XX 和 </s> 的 token 到 id 映射
            special_tokens=[
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


# 定义一个名为 MBart50Converter 的类，继承自 SpmConverter 类
class MBart50Converter(SpmConverter):
    
    # 定义一个名为 vocab 的方法，接收 proto 参数，返回一个词汇表列表
    def vocab(self, proto):
        # 初始化词汇表，包括常见特殊 token 和初始权重
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        # 将 proto 对象中的子词片段（从第四个开始）添加到词汇表中
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # 添加多种语言的标识符和对应的初始权重
        vocab += [
            ("ar_AR", 0.0), ("cs_CZ", 0.0), ("de_DE", 0.0), ("en_XX", 0.0), ("es_XX", 0.0), ("et_EE", 0.0), ("fi_FI", 0.0),
            ("fr_XX", 0.0), ("gu_IN", 0.0), ("hi_IN", 0.0), ("it_IT", 0.0), ("ja_XX", 0.0), ("kk_KZ", 0.0), ("ko_KR", 0.0),
            ("lt_LT", 0.0), ("lv_LV", 0.0), ("my_MM", 0.0), ("ne_NP", 0.0), ("nl_XX", 0.0), ("ro_RO", 0.0), ("ru_RU", 0.0),
            ("si_LK", 0.0), ("tr_TR", 0.0), ("vi_VN", 0.0), ("zh_CN", 0.0), ("af_ZA", 0.0), ("az_AZ", 0.0), ("bn_IN", 0.0),
            ("fa_IR", 0.0), ("he_IL", 0.0), ("hr_HR", 0.0), ("id_ID", 0.0), ("ka_GE", 0.0), ("km_KH", 0.0), ("mk_MK", 0.0),
            ("ml_IN", 0.0), ("mn_MN", 0.0), ("mr_IN", 0.0), ("pl_PL", 0.0), ("ps_AF", 0.0), ("pt_XX", 0.0), ("sv_SE", 0.0),
            ("sw_KE", 0.0), ("ta_IN", 0.0), ("te_IN", 0.0), ("th_TH", 0.0), ("tl_XX", 0.0), ("uk_UA", 0.0), ("ur_PK", 0.0),
            ("xh_ZA", 0.0), ("gl_ES", 0.0), ("sl_SI", 0.0)  # fmt: skip
        ]
        # 添加一个特殊的 mask 标识符和初始权重
        vocab += [("<mask>", 0.0)]
        # 返回完整的词汇表
        return vocab
    
    # 定义一个名为 unk_id 的方法，接收 proto 参数，返回未知 token 的 id（这里固定为 3）
    def unk_id(self, proto):
        return 3
    # 定义一个方法 `post_processor`，用于生成处理器对象
    def post_processor(self):
        # 返回一个模板处理器对象，配置了单句和双句模板以及特殊令牌信息
        return processors.TemplateProcessing(
            single="en_XX $A </s>",  # 单句模板，用 `en_XX $A </s>` 表示
            pair="en_XX $A $B </s>",  # 双句模板，用 `en_XX $A $B </s>` 表示
            special_tokens=[
                # 定义特殊令牌列表，包括 ("en_XX", en_XX 对应的 ID) 和 ("</s>", </s> 对应的 ID)
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )
# 定义 NllbConverter 类，继承自 SpmConverter 类
class NllbConverter(SpmConverter):

    # 定义 vocab 方法，接受 proto 参数
    def vocab(self, proto):
        # 初始化词汇表，包括四个特殊标记和 proto 中的 piece 的内容与得分（从第四个 piece 开始）
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]  # 添加 proto 中的 piece 的内容与得分
        return vocab  # 返回词汇表

    # 定义 unk_id 方法，接受 proto 参数
    def unk_id(self, proto):
        return 3  # 返回未知标记的 id，这里始终为 3

    # 定义 post_processor 方法
    def post_processor(self):
        # 返回 TemplateProcessing 处理器的实例，用于后处理文本
        return processors.TemplateProcessing(
            single="eng_Latn $A </s>",  # 单句模板
            pair="eng_Latn $A $B </s>",  # 双句模板
            special_tokens=[
                ("eng_Latn", self.original_tokenizer.convert_tokens_to_ids("eng_Latn")),  # 特殊标记：eng_Latn 的 id
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),  # 特殊标记：</s> 的 id
            ],
        )


# 定义 SeamlessM4TConverter 类，继承自 SpmConverter 类
class SeamlessM4TConverter(SpmConverter):

    # 定义 vocab 方法，接受 proto 参数
    def vocab(self, proto):
        # 初始化词汇表，包括四个特殊标记和 proto 中的 piece 的内容与得分（从第四个 piece 开始）
        vocab = [
            ("<pad>", 0.0),
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]  # 添加 proto 中的 piece 的内容与得分
        return vocab  # 返回词汇表

    # 定义 unk_id 方法，接受 proto 参数
    def unk_id(self, proto):
        return self.original_tokenizer.unk_token_id  # 返回原始 tokenizer 的未知标记 id

    # 定义 post_processor 方法
    def post_processor(self):
        # 返回 TemplateProcessing 处理器的实例，用于后处理文本
        return processors.TemplateProcessing(
            single="__eng__ $A </s>",  # 单句模板
            pair="__eng__ $A $B </s>",  # 双句模板
            special_tokens=[
                ("__eng__", self.original_tokenizer.convert_tokens_to_ids("__eng__")),  # 特殊标记：__eng__ 的 id
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),  # 特殊标记：</s> 的 id
            ],
        )


# 定义 XLMRobertaConverter 类，继承自 SpmConverter 类
class XLMRobertaConverter(SpmConverter):

    # 定义 vocab 方法，接受 proto 参数
    def vocab(self, proto):
        # 初始化词汇表，包括五个特殊标记、proto 中的 piece 的内容与得分（从第四个 piece 开始）以及额外的 <mask> 标记
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]  # 添加 proto 中的 piece 的内容与得分
        vocab += [("<mask>", 0.0)]  # 添加 <mask> 标记
        return vocab  # 返回词汇表

    # 定义 unk_id 方法，接受 proto 参数
    def unk_id(self, proto):
        unk_id = 3  # 设置未知标记的 id
        return unk_id  # 返回未知标记的 id

    # 定义 post_processor 方法
    def post_processor(self):
        # 返回 TemplateProcessing 处理器的实例，用于后处理文本
        return processors.TemplateProcessing(
            single="<s> $A </s>",  # 单句模板
            pair="<s> $A </s> </s> $B </s>",  # 双句模板
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),  # 特殊标记：<s> 的 id
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),  # 特殊标记：</s> 的 id
            ],
        )


# 定义 XLNetConverter 类，继承自 SpmConverter 类
class XLNetConverter(SpmConverter):

    # 定义 vocab 方法，接受 proto 参数
    def vocab(self, proto):
        # 返回根据 piece.piece 是否包含数字或逗号来决定是否减去 100 分的词汇表列表
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]
    # 定义一个方法用于文本规范化处理，接受参数 proto 作为输入
    def normalizer(self, proto):
        # 定义一个列表，包含一系列的文本规范化器，用于处理文本中的特定模式替换
        list_normalizers = [
            normalizers.Replace("``", '"'),  # 替换双反引号为双引号
            normalizers.Replace("''", '"'),  # 替换单反引号为双引号
        ]
        # 如果原始分词器不保留重音符号，添加将 Unicode 数据标准化为分解形式 (NFKD) 的规范化器
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            # 添加去除重音符号的规范化器
            list_normalizers.append(normalizers.StripAccents())
        # 如果原始分词器需要小写化处理，添加小写化规范化器
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        # 获取预编译字符映射表
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        # 如果存在预编译字符映射表，添加预编译规范化器
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        # 添加用正则表达式替换多个连续空格为单个空格的规范化器
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        
        # 返回一个组合了所有规范化器的序列化规范化器对象
        return normalizers.Sequence(list_normalizers)

    # 定义一个方法用于后处理器，返回一个模板处理对象
    def post_processor(self):
        return processors.TemplateProcessing(
            # 单个文本序列的模板，使用特定占位符和分隔符
            single="$A:0 <sep>:0 <cls>:2",
            # 成对文本序列的模板，使用特定占位符和分隔符
            pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
            # 定义特殊标记及其对应的 token ID
            special_tokens=[
                ("<sep>", self.original_tokenizer.convert_tokens_to_ids("<sep>")),
                ("<cls>", self.original_tokenizer.convert_tokens_to_ids("<cls>")),
            ],
        )
class ReformerConverter(SpmConverter):
    pass



class RemBertConverter(SpmConverter):
    # 受 AlbertConverter 启发

    # 标准化器方法，处理给定的 proto 对象
    def normalizer(self, proto):
        # 定义一组标准化器列表，用于处理文本
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
        # 如果不保留重音符号，则添加相应的标准化器
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        # 如果执行小写转换，则添加小写化标准化器
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        # 从 proto 中获取预编译的字符映射，如果存在，则添加预编译标准化器
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        # 返回一个序列化的标准化器对象
        return normalizers.Sequence(list_normalizers)

    # 后处理器方法，返回一个模板处理器对象
    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )



class BertGenerationConverter(SpmConverter):
    pass



class PegasusConverter(SpmConverter):
    # 词汇表方法，生成给定 proto 的词汇表
    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0.0),
            (self.original_tokenizer.eos_token, 0.0),
        ]

        # 如果存在 mask_token_sent，则添加到词汇表
        if self.original_tokenizer.mask_token_sent is not None:
            vocab += [(self.original_tokenizer.mask_token_sent, 0.0)]

        # 如果存在 mask_token 并且其 ID 小于偏移值，则添加到词汇表
        if (
            self.original_tokenizer.mask_token is not None
            and self.original_tokenizer.mask_token_id < self.original_tokenizer.offset
        ):
            vocab += [(self.original_tokenizer.mask_token, 0.0)]

        # 添加未知词标记，对于从 2 到偏移值的范围，使用固定的负分数
        vocab += [(f"<unk_{i}>", -100.0) for i in range(2, self.original_tokenizer.offset)]
        # 添加 proto 对象中第二个元素之后的所有片段和它们的分数
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[2:]]
        return vocab

    # 未知词 ID 方法，根据 proto 对象返回未知词 ID
    def unk_id(self, proto):
        return proto.trainer_spec.unk_id + self.original_tokenizer.offset

    # 预分词器方法，返回一个序列化的预分词器对象
    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            ]
        )

    # 后处理器方法，返回一个模板处理器对象
    def post_processor(self):
        eos = self.original_tokenizer.eos_token
        special_tokens = [
            (eos, self.original_tokenizer.eos_token_id),
        ]
        return processors.TemplateProcessing(single=["$A", eos], pair=["$A", "$B", eos], special_tokens=special_tokens)



class T5Converter(SpmConverter):
    pass
    # 定义一个方法用于生成词汇表，接收一个 proto 参数
    def vocab(self, proto):
        # 获取原始分词器的额外 ID 数量
        num_extra_ids = self.original_tokenizer._extra_ids
        # 从 proto 的 pieces 属性中提取词汇和对应的分数，组成列表
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        # 添加额外的特殊标记到词汇表中，这些标记是以 "<extra_id_i>" 格式的字符串
        vocab += [(f"<extra_id_{i}>", 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        # 返回生成的词汇表
        return vocab

    # 定义一个方法用于生成后处理器
    def post_processor(self):
        # 返回一个模板处理器对象，配置了不同长度的模板以及特殊标记的转换
        return processors.TemplateProcessing(
            single=["$A", "</s>"],  # 单文本模板，包含 "$A" 和 "</s>"
            pair=["$A", "</s>", "$B", "</s>"],  # 双文本模板，包含 "$A", "</s>", "$B", "</s>"
            special_tokens=[  # 特殊标记的配置，将 "</s>" 映射到其对应的 ID
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )
# 定义名为 UdopConverter 的类，继承自 SpmConverter 类
class UdopConverter(SpmConverter):
    
    # 定义 post_processor 方法，用于创建处理器对象
    def post_processor(self):
        # 返回 TemplateProcessing 处理器对象，配置如下参数：
        return processors.TemplateProcessing(
            # 单句模板，使用变量 $A 和结束标记 </s>
            single=["$A", "</s>"],
            # 双句模板，使用变量 $A 和 $B，并以 </s> 作为结束标记
            pair=["$A", "</s>", "$B", "</s>"],
            # 特殊 token 配置，包含结束标记 </s> 的 ID 映射
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


# 定义名为 WhisperConverter 的类，继承自 Converter 类
class WhisperConverter(Converter):
    
    # 定义 converted 方法，返回 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 获取原始分词器的词汇表和合并列表
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        # 创建 Tokenizer 对象，配置如下参数：
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        # 设置 Tokenizer 的预处理器和解码器为 ByteLevel
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()

        # 获取原始分词器的前缀 token ID 和对应的 token 列表
        prefix_token_ids = self.original_tokenizer.prefix_tokens
        prefixes = self.original_tokenizer.convert_ids_to_tokens(prefix_token_ids)
        eos = self.original_tokenizer.eos_token
        eos_token_id = self.original_tokenizer.eos_token_id
        
        # 构建前缀模板字符串，以及设置 Tokenizer 的后处理器
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        tokenizer.post_processor = processors.TemplateProcessing(
            # 单句模板，包含前缀模板、变量 $A 和结束标记的 ID 映射
            single=f"{prefix_template} $A:0 {eos}:0",
            # 双句模板，包含前缀模板、变量 $A 和 $B，以及结束标记的 ID 映射
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            # 特殊 token 配置，包含结束标记和前缀 token 的 ID 映射
            special_tokens=[
                (eos, eos_token_id),
                *zip(prefixes, prefix_token_ids),
            ],
        )

        # 返回配置完成的 Tokenizer 对象
        return tokenizer


# 定义名为 BigBirdConverter 的类，继承自 SpmConverter 类
class BigBirdConverter(SpmConverter):
    
    # 定义 post_processor 方法，用于创建处理器对象
    def post_processor(self):
        # 返回 TemplateProcessing 处理器对象，配置如下参数：
        return processors.TemplateProcessing(
            # 单句模板，使用固定 token [CLS] 和变量 $A，以及固定 token [SEP]
            single="[CLS]:0 $A:0 [SEP]:0",
            # 双句模板，使用固定 token [CLS]、变量 $A 和 $B，以及两个 [SEP] 标记
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            # 特殊 token 配置，包含 [CLS] 和 [SEP] 的 ID 映射
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class CLIPConverter(Converter):
    # 这里是未完成的类定义，需要在此处继续补充代码
    # 定义一个方法 `converted`，返回一个 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 从原始分词器获取词汇表
        vocab = self.original_tokenizer.encoder
        # 从原始分词器获取BPE合并操作列表
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        # 获取原始分词器的未知标记
        unk_token = self.original_tokenizer.unk_token

        # 创建一个 Tokenizer 对象，使用 BPE 分词器
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,  # 设置词汇表
                merges=merges,  # 设置合并操作列表
                dropout=None,  # 不使用dropout
                continuing_subword_prefix="",  # 继续子词前缀为空
                end_of_word_suffix="</w>",  # 设置词尾标记
                fuse_unk=False,  # 禁用未知标记融合
                unk_token=str(unk_token),  # 设置未知标记
            )
        )

        # 设置标准化器为 NFC、替换多余空格为单个空格、转换为小写的序列
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()]
        )

        # 设置预分词器序列，包括使用正则表达式和字节级预处理器
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""),
                    behavior="removed",  # 移除匹配的内容
                    invert=True,  # 反转操作
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),  # 字节级处理，无前缀空格
            ]
        )

        # 设置解码器为字节级解码器
        tokenizer.decoder = decoders.ByteLevel()

        # 使用 RobertaProcessing 处理器进行后处理，设置分隔符和特殊标记
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(self.original_tokenizer.eos_token, self.original_tokenizer.eos_token_id),  # 分隔符设定
            cls=(self.original_tokenizer.bos_token, self.original_tokenizer.bos_token_id),  # 类标记设定
            add_prefix_space=False,  # 不添加前缀空格
            trim_offsets=False,  # 不修剪偏移量
        )

        # 返回创建的 Tokenizer 对象
        return tokenizer
class LayoutLMv2Converter(Converter):
    # LayoutLMv2Converter 类，继承自 Converter 类，用于实现转换器功能
    def converted(self) -> Tokenizer:
        # 转换方法，返回一个 Tokenizer 对象
        vocab = self.original_tokenizer.vocab
        # 获取原始 tokenizer 的词汇表

        # 创建 Tokenizer 对象，使用 WordPiece 模型，并传入 unk_token 作为未知标记
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = True

        # 检查原始 tokenizer 是否具有 basic_tokenizer 属性，根据属性值设置相应变量
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 设置 tokenizer 的正则化器为 BertNormalizer，配置各种参数
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )

        # 设置 tokenizer 的预处理器为 BertPreTokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # 获取特殊标记的字符串表示，并分配给相应变量
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 设置 tokenizer 的后处理器为 TemplateProcessing，根据单句和双句模板配置
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )

        # 设置 tokenizer 的解码器为 WordPiece 解码器，前缀为 "##"
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer
        # 返回配置好的 Tokenizer 对象


class BlenderbotConverter(Converter):
    # BlenderbotConverter 类，继承自 Converter 类，用于实现转换器功能
    def converted(self) -> Tokenizer:
        # 转换方法，返回一个 Tokenizer 对象
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        # 创建 Tokenizer 对象，使用 BPE 模型，并传入相应的参数
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        # 设置 tokenizer 的预处理器为 ByteLevel，并根据原始 tokenizer 的属性配置添加前缀空格
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)

        # 设置 tokenizer 的解码器为 ByteLevel 解码器
        tokenizer.decoder = decoders.ByteLevel()

        # 设置 tokenizer 的后处理器为 TemplateProcessing，根据单句模板配置
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A:0 {ot.eos_token}:0",
            special_tokens=[
                (ot.eos_token, ot.eos_token_id),
            ],
        )

        return tokenizer
        # 返回配置好的 Tokenizer 对象


class XGLMConverter(SpmConverter):
    # XGLMConverter 类，继承自 SpmConverter 类，用于实现转换器功能
    def vocab(self, proto):
        # 生成词汇表的方法，接受一个 proto 参数
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        # 将 proto 中的 pieces 转换为词汇表的元组，从第三个元素开始
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # 添加一些假词汇到词汇表中

        return vocab
        # 返回生成的词汇表


    def unk_id(self, proto):
        # 获取未知标记的 ID 的方法，接受一个 proto 参数
        unk_id = 3
        return unk_id
        # 返回未知标记的 ID
    # 定义一个方法 `post_processor`，用于生成一个处理器对象 `TemplateProcessing`
    def post_processor(self):
        # 返回一个 TemplateProcessing 对象，配置如下参数：
        return processors.TemplateProcessing(
            # 当处理单个句子时的模板，插入特殊标记 `$A`
            single="</s> $A",
            # 当处理句对时的模板，插入特殊标记 `$A` 和 `$B`
            pair="</s> $A </s> </s> $B",
            # 定义一些特殊标记及其对应的 ID，使用了 `original_tokenizer` 中的方法将特殊标记转换为 ID
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )
# GemmaConvert 类继承自 SpmConverter，用于特定文本转换任务的定制化处理
class GemmaConvert(SpmConverter):
    # 设置字节处理回退选项为 True
    handle_byte_fallback = True

    # 下面是一个多行字符串，可能用于配置参数，未直接使用于代码逻辑中
    """"
    split_by_unicode_script: true
    split_by_number: true
    split_by_whitespace: true
    treat_whitespace_as_suffix: false
    allow_whitespace_only_pieces: true
    split_digits: true
    byte_fallback: true
    """

    # 标准化器函数，返回一个替换空格为特定符号的标准化器对象
    def normalizer(self, proto):
        return normalizers.Replace(" ", "▁")

    # 词汇表函数，根据 proto 对象的 pieces 属性生成词汇表列表
    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0.0),  # 添加填充标记和对应的得分
            (self.original_tokenizer.eos_token, 0.0),  # 添加结束标记和对应的得分
            (self.original_tokenizer.bos_token, 0.0),  # 添加起始标记和对应的得分
        ]
        # 遍历 proto 对象的 pieces 属性，从第四个元素开始添加到词汇表中
        for piece in proto.pieces[3:]:
            if piece.piece == "<0x09>":
                vocab += [("\t", piece.score)]  # 如果词素是 "<0x09>"，则用制表符 "\t" 替代
            else:
                vocab += [(piece.piece, piece.score)]  # 否则直接添加词素和得分
        # 返回生成的词汇表
        return vocab

    # 预处理分词器函数，返回 None 表示没有预处理分词器
    def pre_tokenizer(self, replacement, add_prefix_space):
        return None

    # 未知标记 ID 函数，始终返回整数值 3 作为未知标记 ID
    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    # 解码器函数，返回一个序列解码器对象，按顺序执行替换、字节回退和融合操作
    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),  # 将特定符号 "▁" 替换为空格
                decoders.ByteFallback(),  # 字节回退解码器
                decoders.Fuse(),  # 融合解码器
            ]
        )
    # 定义一个方法 `tokenizer`，接受一个参数 `proto`
    def tokenizer(self, proto):
        # 从参数 `proto` 的 `trainer_spec` 属性中获取 `model_type`
        model_type = proto.trainer_spec.model_type
        # 调用当前对象的 `vocab` 方法，获取词汇表和分数
        vocab_scores = self.vocab(proto)
        
        # 根据 `model_type` 的值进行条件判断
        if model_type == 1:
            # 如果 `model_type` 为 1，导入 `tokenizers` 模块
            import tokenizers

            # 检查 `tokenizers` 模块的版本是否小于 "0.14.0"
            if version.parse(tokenizers.__version__) < version.parse("0.14.0"):
                # 如果版本小于 "0.14.0"，创建一个 `Tokenizer` 对象，使用 `Unigram` 模型和词汇分数
                tokenizer = Tokenizer(Unigram(vocab_scores, 0))
            else:
                # 如果版本大于等于 "0.14.0"，创建一个 `Tokenizer` 对象，使用 `Unigram` 模型、词汇分数和字节回退
                tokenizer = Tokenizer(Unigram(vocab_scores, 0, byte_fallback=True))

        elif model_type == 2:
            # 如果 `model_type` 为 2，调用 `GemmaSentencePieceExtractor`，提取词汇分数和合并列表
            _, merges = GemmaSentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            # 创建 BPE 词汇表，将词汇与索引对应起来
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}

            # 创建一个 `Tokenizer` 对象，使用 BPE 模型、词汇表、合并列表和其他参数
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                    byte_fallback=True,
                    dropout=None,
                )
            )
            # 向 `tokenizer` 添加特殊标记
            tokenizer.add_special_tokens(
                [
                    AddedToken("<pad>", normalized=False, special=True),
                    AddedToken("<eos>", normalized=False, special=True),
                    AddedToken("<bos>", normalized=False, special=True),
                    AddedToken("<unk>", normalized=False, special=True),
                ]
            )
        else:
            # 如果 `model_type` 既不是 1 也不是 2，抛出异常
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )
        
        # 根据 `proto` 的 `trainer_spec.user_defined_symbols` 创建用户自定义符号的 `AddedToken` 列表
        user_defined_symbols = [
            AddedToken(token, normalized=False, special=False) for token in proto.trainer_spec.user_defined_symbols
        ]
        # 向 `tokenizer` 添加用户自定义符号
        tokenizer.add_tokens(user_defined_symbols)
        
        # 返回创建的 `tokenizer` 对象
        return tokenizer
class LlamaConverter(SpmConverter):
    # 设置处理字节回退的开关为 True
    handle_byte_fallback = True

    # 构建词汇表的方法，接受一个 proto 参数
    def vocab(self, proto):
        # 初始词汇表包含特殊标记和默认得分
        vocab = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        # 将 proto 中第三个位置之后的词片段及其得分加入词汇表
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    # 返回未知标记的 ID，默认为 0
    def unk_id(self, proto):
        unk_id = 0
        return unk_id

    # 返回解码器对象，用于文本序列的解析和替换
    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.Replace("▁", " "),  # 将 "▁" 替换为 " "
            decoders.ByteFallback(),  # 字节回退处理器
            decoders.Fuse(),  # 合并处理器
        ]
        # 如果需要在前缀空格之前添加处理器，则添加处理器去除左边的空格
        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    # 返回标记器对象，根据模型类型选择不同的标记化方法
    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)

        # 根据模型类型选择合适的标记器
        if model_type == 1:
            import tokenizers

            # 根据 tokenizers 库的版本选择 Unigram 标记器
            if version.parse(tokenizers.__version__) < version.parse("0.14.0"):
                tokenizer = Tokenizer(Unigram(vocab_scores, 0))
            else:
                tokenizer = Tokenizer(Unigram(vocab_scores, 0, byte_fallback=True))

        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            # 使用 BPE 标记器，并添加特殊标记
            tokenizer = Tokenizer(
                BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True)
            )
            tokenizer.add_special_tokens(
                [
                    AddedToken("<unk>", normalized=False, special=True),
                    AddedToken("<s>", normalized=False, special=True),
                    AddedToken("</s>", normalized=False, special=True),
                ]
            )
        else:
            # 抛出异常，提示模型类型与训练算法不匹配
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    # 返回正规化器对象，处理文本的规范化过程
    def normalizer(self, proto):
        sequence = []
        # 如果原始标记器具有添加前缀空格的功能，则在序列中添加前缀处理器
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            if self.original_tokenizer.add_prefix_space:
                sequence += [normalizers.Prepend(prepend="▁")]
        # 将空格替换为 "▁" 的处理器添加到序列中
        sequence += [normalizers.Replace(pattern=" ", content="▁")]
        return normalizers.Sequence(sequence)

    # 返回预标记器对象，用于预处理文本中的特定标记
    def pre_tokenizer(self, replacement, add_prefix_space):
        # 返回空值，表示没有预标记器
        return None

    # 返回后处理器对象，用于进一步处理标记化后的文本
    def post_processor(self):
        # 返回空值，表示没有后处理器
        # 后处理器在 LlamaTokenizerFast 类中定义
        return None
    # 定义一个方法 `converted`，返回一个 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 获取原始的分词器对象
        ot = self.original_tokenizer
        # 获取分词器的词汇表
        vocab = ot.encoder
        # 获取分词器的合并列表
        merges = list(ot.bpe_ranks.keys())

        # 创建一个新的 Tokenizer 对象，使用 BPE 分词器
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,  # 设定词汇表
                merges=merges,  # 设定合并列表
                dropout=None,  # 不使用 dropout
                continuing_subword_prefix="",  # 设定连续子词的前缀
                end_of_word_suffix="",  # 设定单词结束后缀
                fuse_unk=False,  # 不融合未知标记
                unk_token=self.original_tokenizer.unk_token,  # 设定未知标记
            )
        )

        # 设定预分词器为 ByteLevel，并根据原始分词器的参数设定
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        # 设定解码器为 ByteLevel
        tokenizer.decoder = decoders.ByteLevel()

        # 获取原始分词器的特殊标记（如 `[CLS]` 和 `[SEP]`）
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 设定后处理器为 TemplateProcessing，根据原始分词器的特殊标记设定模板
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls} $A {sep}",  # 单句模板
            pair=f"{cls} $A {sep} $B {sep}",  # 双句模板
            special_tokens=[
                (cls, cls_token_id),  # 添加 `[CLS]` 特殊标记
                (sep, sep_token_id),  # 添加 `[SEP]` 特殊标记
            ],
        )

        # 返回创建好的 Tokenizer 对象
        return tokenizer
# 定义一个映射字典，将慢速tokenizer的类名映射到相应的快速converter类
SLOW_TO_FAST_CONVERTERS = {
    "AlbertTokenizer": AlbertConverter,
    "BartTokenizer": RobertaConverter,
    "BarthezTokenizer": BarthezConverter,
    "BertTokenizer": BertConverter,
    "BigBirdTokenizer": BigBirdConverter,
    "BlenderbotTokenizer": BlenderbotConverter,
    "CamembertTokenizer": CamembertConverter,
    "CLIPTokenizer": CLIPConverter,
    "CodeGenTokenizer": GPT2Converter,
    "ConvBertTokenizer": BertConverter,
    "DebertaTokenizer": DebertaConverter,
    "DebertaV2Tokenizer": DebertaV2Converter,
    "DistilBertTokenizer": BertConverter,
    "DPRReaderTokenizer": BertConverter,
    "DPRQuestionEncoderTokenizer": BertConverter,
    "DPRContextEncoderTokenizer": BertConverter,
    "ElectraTokenizer": BertConverter,
    "FNetTokenizer": AlbertConverter,
    "FunnelTokenizer": FunnelConverter,
    "GPT2Tokenizer": GPT2Converter,
    "HerbertTokenizer": HerbertConverter,
    "LayoutLMTokenizer": BertConverter,
    "LayoutLMv2Tokenizer": BertConverter,
    "LayoutLMv3Tokenizer": RobertaConverter,
    "LayoutXLMTokenizer": XLMRobertaConverter,
    "LongformerTokenizer": RobertaConverter,
    "LEDTokenizer": RobertaConverter,
    "LxmertTokenizer": BertConverter,
    "MarkupLMTokenizer": MarkupLMConverter,
    "MBartTokenizer": MBartConverter,
    "MBart50Tokenizer": MBart50Converter,
    "MPNetTokenizer": MPNetConverter,
    "MobileBertTokenizer": BertConverter,
    "MvpTokenizer": RobertaConverter,
    "NllbTokenizer": NllbConverter,
    "OpenAIGPTTokenizer": OpenAIGPTConverter,
    "PegasusTokenizer": PegasusConverter,
    "Qwen2Tokenizer": Qwen2Converter,
    "RealmTokenizer": BertConverter,
    "ReformerTokenizer": ReformerConverter,
    "RemBertTokenizer": RemBertConverter,
    "RetriBertTokenizer": BertConverter,
    "RobertaTokenizer": RobertaConverter,
    "RoFormerTokenizer": RoFormerConverter,
    "SeamlessM4TTokenizer": SeamlessM4TConverter,
    "SqueezeBertTokenizer": BertConverter,
    "T5Tokenizer": T5Converter,
    "UdopTokenizer": UdopConverter,
    "WhisperTokenizer": WhisperConverter,
    "XLMRobertaTokenizer": XLMRobertaConverter,
    "XLNetTokenizer": XLNetConverter,
    "SplinterTokenizer": SplinterConverter,
    "XGLMTokenizer": XGLMConverter,
    "LlamaTokenizer": LlamaConverter,
    "CodeLlamaTokenizer": LlamaConverter,
    "GemmaTokenizer": GemmaConvert,
}

# 定义函数，将慢速tokenizer实例转换为对应的快速tokenizer实例
def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """

    # 获取tokenizer的类名
    tokenizer_class_name = transformer_tokenizer.__class__.__name__
    # 检查要转换的分词器类名是否存在于SLOW_TO_FAST_CONVERTERS字典中
    if tokenizer_class_name not in SLOW_TO_FAST_CONVERTERS:
        # 如果不存在，则抛出值错误异常，指明无法将该分词器类转换为快速分词器实例
        raise ValueError(
            f"An instance of tokenizer class {tokenizer_class_name} cannot be converted in a Fast tokenizer instance."
            " No converter was found. Currently available slow->fast convertors:"
            f" {list(SLOW_TO_FAST_CONVERTERS.keys())}"
        )

    # 根据分词器类名从SLOW_TO_FAST_CONVERTERS字典中获取对应的转换器类
    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]

    # 返回通过转换器类对transformer_tokenizer进行转换后的结果
    return converter_class(transformer_tokenizer).converted()
```