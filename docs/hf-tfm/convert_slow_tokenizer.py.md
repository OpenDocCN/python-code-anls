# `.\transformers\convert_slow_tokenizer.py`

```
# 设置编码为 UTF-8
# 版权声明
# 该代码的版权归 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 按“原样”提供，不提供任何形式的明示或暗示担保，
# 包括但不限于适销性、特定用途适用性和不侵权担保。
# 有关详细信息，请参阅许可证。

"""
将慢速分词器转换为其快速分词器对应项的实用工具。

所有的转换都在此处分组，以便在快速分词器文件之外收集 SentencePiece 依赖项，
并且允许我们将对 SentencePiece 的依赖关系设置为可选。
"""

# 引入警告模块
import warnings
# 引入类型提示相关的模块
from typing import Dict, List, Tuple

# 引入版本处理相关的模块
from packaging import version
# 引入 tokenizers 库
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
# 引入 tokenizers.models 下的模块
from tokenizers.models import BPE, Unigram, WordPiece

# 引入一些工具函数
from .utils import is_protobuf_available, requires_backends
# 引入 import_utils 中的 PROTOBUF_IMPORT_ERROR
from .utils.import_utils import PROTOBUF_IMPORT_ERROR

# 定义一个函数用于导入 protobuf
def import_protobuf(error_message=""):
    # 如果 protobuf 可用
    if is_protobuf_available():
        # 导入 google.protobuf 模块
        import google.protobuf
        # 如果 protobuf 的版本小于 4.0.0
        if version.parse(google.protobuf.__version__) < version.parse("4.0.0"):
            # 从 transformers.utils 中导入 sentencepiece_model_pb2
            from transformers.utils import sentencepiece_model_pb2
        else:
            # 从 transformers.utils 中导入 sentencepiece_model_pb2_new 并重命名为 sentencepiece_model_pb2
            from transformers.utils import sentencepiece_model_pb2_new as sentencepiece_model_pb2
        # 返回 sentencepiece_model_pb2
        return sentencepiece_model_pb2
    else:
        # 抛出 ImportError，提示缺少 SentencePiece 依赖
        raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))

# 定义 SentencePieceExtractor 类
class SentencePieceExtractor:
    """
    SentencePiece 训练模型的提取器实现。 https://github.com/google/sentencepiece
    """

    # 初始化函数
    def __init__(self, model: str):
        # 检查是否需要 SentencePiece 依赖
        requires_backends(self, "sentencepiece")
        # 从 sentencepiece 模块中导入 SentencePieceProcessor 类
        from sentencepiece import SentencePieceProcessor

        # 创建 SentencePieceProcessor 实例
        self.sp = SentencePieceProcessor()
        # 加载 SentencePiece 模型
        self.sp.Load(model)
    # 定义一个方法用于提取 Subword 分词器的词汇表及合并列表
    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        默认情况下，将返回词汇表及其顺序的合并列表，通过发送 `vocab_scores`，我们将按照片段分数的顺序来排序合并列表。
        """
        # 获取 Subword 分词器对象
        sp = self.sp
        # 创建一个字典，将词汇的索引映射到其对应的词汇
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}
        # 如果提供了 `vocab_scores`，则使用其来排序合并列表
        if vocab_scores is not None:
            # 将 vocab_scores 转换为字典，并标记为反向排序
            vocab_scores, reverse = dict(vocab_scores), True
        else:
            # 否则使用默认的词汇及其顺序来排序
            vocab_scores, reverse = vocab, False

        # 合并
        merges = []
        # 遍历词汇及其分数
        for merge, piece_score in vocab_scores.items():
            # 存储合并后的局部列表
            local = []
            # 遍历合并的位置
            for index in range(1, len(merge)):
                # 将词汇分为左右两部分
                piece_l, piece_r = merge[:index], merge[index:]
                # 检查左右两部分是否在词汇表中
                if piece_l in vocab and piece_r in vocab:
                    # 如果在词汇表中，则添加到局部列表中
                    local.append((piece_l, piece_r, piece_score))
            # 将局部列表按照词汇表索引排序
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            # 将局部列表扩展到合并列表中
            merges.extend(local)

        # 根据分数排序合并列表，并根据需要进行反向排序
        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
        # 仅保留合并列表中的合并操作，去除分数信息
        merges = [(val[0], val[1]) for val in merges]
        # 返回词汇表及合并列表
        return vocab, merges
# 检查给定字符串是否符合特定条件
def check_number_comma(piece: str) -> bool:
    # 若字符串长度小于 2 或最后一个字符不是逗号或倒数第二个字符不是数字，则返回 True
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


# 转换器基类，用于定义转换器的基本行为
class Converter:
    def __init__(self, original_tokenizer):
        # 初始化转换器时保存原始的分词器
        self.original_tokenizer = original_tokenizer

    # 抽象方法，子类需要实现具体的转换逻辑
    def converted(self) -> Tokenizer:
        # 抛出未实现错误，子类需重写该方法
        raise NotImplementedError()


# 基于 BERT 的转换器类，继承自 Converter
class BertConverter(Converter):
    def converted(self) -> Tokenizer:
        # 获取原始分词器的词汇表
        vocab = self.original_tokenizer.vocab
        # 使用 WordPiece 构造分词器，并指定未登录词标记
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        # 初始化变量，用于保存是否进行中文字符分词、是否删除重音符号、是否转换为小写等配置
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        # 检查原始分词器是否具有基本分词器属性，并获取对应配置
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 配置分词器的正规化器，用于规范化输入文本
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # 配置分词器的预处理器，用于处理输入文本
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # 获取特殊标记的字符串形式和对应的标记 ID
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 配置分词器的后处理器，用于处理分词结果
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        # 配置分词器的解码器，用于解码分词结果
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # 返回配置好的分词器
        return tokenizer


class SplinterConverter(Converter):
    # 将原始的分词器转换为新的分词器
    def converted(self) -> Tokenizer:
        # 获取原始分词器的词汇表
        vocab = self.original_tokenizer.vocab
        # 创建新的分词器，使用WordPiece模型和未知标记
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        # 初始化变量，用于记录是否需要处理中文字符、是否去除重音符号、是否转换为小写
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        # 检查原始分词器是否有基本分词器属性
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            # 获取基本分词器的处理标志
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 设置新分词器的规范化器，包括清理文本、处理中文字符、去除重音符号、转换为小写
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # 设置新分词器的预处理器为Bert预处理器
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # 获取特殊标记的字符串形式
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        question = str(self.original_tokenizer.question_token)
        dot = "."
        # 获取特殊标记的ID
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id
        question_token_id = self.original_tokenizer.question_token_id
        dot_token_id = self.original_tokenizer.convert_tokens_to_ids(".")

        # 根据填充位置设置句子对的模板
        if self.original_tokenizer.padding_side == "right":
            pair = f"{cls}:0 $A:0 {question} {dot} {sep}:0 $B:1 {sep}:1"
        else:
            pair = f"{cls}:0 $A:0 {sep}:0 $B:1 {question} {dot} {sep}:1"

        # 设置新分词器的后处理器，包括单句和句子对的模板，以及特殊标记
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
        # 设置新分词器的解码器为WordPiece解码器
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # 返回新的分词器
        return tokenizer
# 定义 FunnelConverter 类，继承自 Converter 类
class FunnelConverter(Converter):
    # 定义 converted 方法，返回 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 获取原始分词器的词汇表
        vocab = self.original_tokenizer.vocab
        # 使用原始分词器的词汇表创建 Tokenizer 对象
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        # 初始化中文字符分词、去除重音符号和小写转换标志
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        # 如果原始分词器具有 basic_tokenizer 属性
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            # 获取 basic_tokenizer 中的标志
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 配置 Tokenizer 的标准化器
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # 配置 Tokenizer 的预处理器
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # 获取原始分词器的 [CLS] 和 [SEP] 标记
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 配置 Tokenizer 的后处理器
        tokenizer.post_processor = processors.TemplateProcessing(
            # 单句模板，使用 Funnel transformer 的 token_type_id 为 2
            single=f"{cls}:2 $A:0 {sep}:0",
            # 双句模板，使用 Funnel transformer 的 token_type_id 为 2
            pair=f"{cls}:2 $A:0 {sep}:0 $B:1 {sep}:1",
            # 添加特殊标记 [CLS] 和 [SEP]
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        # 配置 Tokenizer 的解码器
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # 返回配置好的 Tokenizer 对象
        return tokenizer

# 定义 MPNetConverter 类，继承自 Converter 类
class MPNetConverter(Converter):
    # 将原始的分词器转换为 Tokenizer 对象
    def converted(self) -> Tokenizer:
        # 获取原始分词器的词汇表
        vocab = self.original_tokenizer.vocab
        # 创建一个新的 Tokenizer 对象，使用 WordPiece 模型和未知标记符号
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        # 初始化变量用于存储是否需要处理中文字符、是否去除重音符号、是否转换为小写
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        # 检查原始分词器是否有 basic_tokenizer 属性
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            # 获取处理中文字符、去除重音符号、转换为小写的设置
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 设置 Tokenizer 的正规化器，根据设置处理文本
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # 设置 Tokenizer 的预处理器，使用 Bert 预处理器
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # 获取原始分词器的 [CLS] 和 [SEP] 标记符号及其对应的 ID
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 设置 Tokenizer 的后处理器，根据模板处理特殊标记符号
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 {sep}:0 $B:1 {sep}:1",  # MPNet 使用两个 [SEP] 标记符号
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        # 设置 Tokenizer 的解码器，使用 WordPiece 解码器
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # 返回转换后的 Tokenizer 对象
        return tokenizer
# 定义一个名为 OpenAIGPTConverter 的类，继承自 Converter 类
class OpenAIGPTConverter(Converter):
    # 定义一个方法 converted，返回类型为 Tokenizer
    def converted(self) -> Tokenizer:
        # 获取原始分词器的编码器
        vocab = self.original_tokenizer.encoder
        # 获取原始分词器的 BPE 合并列表
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        # 获取原始分词器的未知标记
        unk_token = self.original_tokenizer.unk_token

        # 创建一个 Tokenizer 对象
        tokenizer = Tokenizer(
            # 使用 BPE 分词器，传入编码器、合并列表、未知标记等参数
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                unk_token=str(unk_token),
                end_of_word_suffix="</w>",
                fuse_unk=False,
            )
        )

        # 如果未知标记在分词器中存在，则添加特殊标记
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])

        # 设置分词器的规范化器为 BertNormalizer，设置小写化为 True
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        # 设置分词器的预处理器为 BertPreTokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        # 设置分词器的解码器为 BPEDecoder，设置后缀为 "</w>"
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        # 返回创建的 Tokenizer 对象
        return tokenizer


# 定义一个名为 GPT2Converter 的类，继承自 Converter 类
class GPT2Converter(Converter):
    # 定义一个方法 converted，返回类型为 Tokenizer
    def converted(self) -> Tokenizer:
        # 获取原始分词器的编码器
        vocab = self.original_tokenizer.encoder
        # 获取原始分词器的 BPE 合并列表
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        # 创建一个 Tokenizer 对象
        tokenizer = Tokenizer(
            # 使用 BPE 分词器，传入编码器、合并列表等参数
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        # 设置分词器的预处理器为 ByteLevel，根据原始分词器是否添加前缀空格来决定是否添加前缀空格
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        # 设置分词器的解���器为 ByteLevel
        tokenizer.decoder = decoders.ByteLevel()

        # 如果原始分词器添加了开始标记，则设置后处理器为 TemplateProcessing，根据开始标记和标记 ID 进行处理
        if self.original_tokenizer.add_bos_token:
            bos = self.original_tokenizer.bos_token
            bos_token_id = self.original_tokenizer.bos_token_id
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos}:0 $A:0",
                pair=f"{bos}:0 $A:0 $B:1",
                special_tokens=[
                    (bos, bos_token_id),
                ],
            )
        else:
            # 如果未添加开始标记，则设置后处理器为 ByteLevel，trim_offsets=False 表示后处理器不做任何处理
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        # 返回创建的 Tokenizer 对象
        return tokenizer


# 定义一个名为 HerbertConverter 的类，继承自 Converter ���
class HerbertConverter(Converter):
    # 将转换后的tokenizer返回，其类型为Tokenizer
    def converted(self) -> Tokenizer:
        # 定义一个字符串，用于标识tokenizer的版本信息
        tokenizer_info_str = "#version:"
        # 定义token后缀，用于标识单词的结束
        token_suffix = "</w>"

        # 从原始tokenizer中获取词汇表和合并规则
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        # 检查是否第一个合并规则包含了版本信息字符串，如果是则移除该合并规则
        if tokenizer_info_str in merges[0][0]:
            merges = merges[1:]

        # 使用词汇表和合并规则初始化一个基于BPE的tokenizer对象
        tokenizer = Tokenizer(
            BPE(
                vocab,
                merges,
                dropout=None,  # 不使用dropout
                unk_token=self.original_tokenizer.unk_token,  # 未知token
                end_of_word_suffix=token_suffix,  # token后缀
            )
        )

        # 设置tokenizer的正则化器为BertNormalizer，不转换为小写，不去除重音符号
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False)
        # 设置tokenizer的预处理器为BertPreTokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        # 设置tokenizer的解码器为BPEDecoder，使用token后缀
        tokenizer.decoder = decoders.BPEDecoder(suffix=token_suffix)
        # 设置tokenizer的后处理器为BertProcessing，使用原始tokenizer的特殊token作为分隔符和类别标识符
        tokenizer.post_processor = processors.BertProcessing(
            sep=(self.original_tokenizer.sep_token, self.original_tokenizer.sep_token_id),
            cls=(self.original_tokenizer.cls_token, self.original_tokenizer.cls_token_id),
        )

        # 返回转换后的tokenizer对象
        return tokenizer
# 定义一个名为 Qwen2Converter 的类，继承自 Converter 类
class Qwen2Converter(Converter):
    # 定义一个 converted 方法，返回类型为 Tokenizer
    def converted(self) -> Tokenizer:
        # 获取原始分词器的编码器和合并列表
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        # 创建一个 Tokenizer 对象，使用 BPE 分词器
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

        # 设置分词器的规范化器为 NFC 规范化
        tokenizer.normalizer = normalizers.NFC()

        # 设置分词器的预处理器为 Sequence，包含 Split 和 ByteLevel 两个预处理器
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(
                        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
                    ),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=getattr(self.original_tokenizer, "add_prefix_space", False),
                    use_regex=False,
                ),
            ]
        )

        # 设置分词器的解码器为 ByteLevel
        tokenizer.decoder = decoders.ByteLevel()
        # 设置分词器的后处理器为 ByteLevel，不修剪偏移量
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # 返回创建的 Tokenizer 对象
        return tokenizer


# 定义一个名为 RobertaConverter 的类，继承自 Converter 类
class RobertaConverter(Converter):
    # 定义一个 converted 方法，返回类型为 Tokenizer
    def converted(self) -> Tokenizer:
        # 获取原始分词器的信息
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        # 创建一个 Tokenizer 对象，使用 BPE 分词器
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

        # 设置分词器的预处理器为 ByteLevel，根据原始分词器是否有前缀空格来添加前缀空格
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        # 设置分词器的解码器为 ByteLevel
        tokenizer.decoder = decoders.ByteLevel()
        # 设置分词器的后处理器为 RobertaProcessing，包含特殊标记和修剪偏移量等参数
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(ot.sep_token, ot.sep_token_id),
            cls=(ot.cls_token, ot.cls_token_id),
            add_prefix_space=ot.add_prefix_space,
            trim_offsets=True,  # True by default on Roberta (historical)
        )

        # 返回创建的 Tokenizer 对象
        return tokenizer


# 定义一个名为 RoFormerConverter 的类，继承自 Converter 类
class RoFormerConverter(Converter):
    # 定义一个方法，将原始的 Tokenizer 转换为新的 Tokenizer
    def converted(self) -> Tokenizer:
        # 导入 JiebaPreTokenizer 类
        from .models.roformer.tokenization_utils import JiebaPreTokenizer

        # 获取原始 Tokenizer 的词汇表
        vocab = self.original_tokenizer.vocab
        # 创建一个新的 Tokenizer 对象，使用 WordPiece 模型和未知标记
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        # 初始化 strip_accents 和 do_lower_case 变量
        strip_accents = False
        do_lower_case = False
        # 检查原始 Tokenizer 是否有 basic_tokenizer 属性
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            # 获取 strip_accents 和 do_lower_case 的值
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        # 设置 Tokenizer 的正规化器，使用 BertNormalizer
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # 设置 Tokenizer 的预处理器，使用自定义的 JiebaPreTokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(JiebaPreTokenizer(vocab))

        # 获取 cls 和 sep 标记，以及它们的标记 ID
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 设置 Tokenizer 的后处理器，使用 TemplateProcessing
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        # 设置 Tokenizer 的解码器，使用 WordPiece 模型
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # 返回转换后的 Tokenizer 对象
        return tokenizer
class DebertaConverter(Converter):
    # 定义一个名为DebertaConverter的类，继承自Converter类
    def converted(self) -> Tokenizer:
        # 定义一个名为converted的方法，返回类型为Tokenizer
        ot = self.original_tokenizer
        # 将self.original_tokenizer赋值给ot
        vocab = ot.encoder
        # 将ot.encoder赋值给vocab
        merges = list(ot.bpe_ranks.keys())
        # 将ot.bpe_ranks的键转换为列表赋值给merges

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
        # 创建一个Tokenizer对象，使用BPE作为参数，设置各种属性

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        # 设置tokenizer的pre_tokenizer属性为ByteLevel，设置add_prefix_space属性为ot.add_prefix_space
        tokenizer.decoder = decoders.ByteLevel()
        # 设置tokenizer的decoder属性为ByteLevel
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
        # 设置tokenizer的post_processor属性为TemplateProcessing，设置各种特殊token

        return tokenizer
        # 返回tokenizer对象


class SpmConverter(Converter):
    # 定义一个名为SpmConverter的类，继承自Converter类
    def __init__(self, *args):
        # 定义一个初始化方法，接受任意数量的参数
        requires_backends(self, "protobuf")
        # 调用requires_backends函数，传入self和"protobuf"

        super().__init__(*args)
        # 调用父类的初始化方法，传入所有参数

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()
        # 调用import_protobuf函数，将返回值赋值给model_pb2

        m = model_pb2.ModelProto()
        # 创建一个model_pb2.ModelProto对象赋值给m
        with open(self.original_tokenizer.vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        # 打开self.original_tokenizer.vocab_file文件，读取内容并解析为m对象
        self.proto = m
        # 将m赋值给self.proto

        if self.proto.trainer_spec.byte_fallback:
            # 如果self.proto.trainer_spec.byte_fallback为真
            if not getattr(self, "handle_byte_fallback", None):
                # 如果self没有handle_byte_fallback属性
                warnings.warn(
                    "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                    " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                    " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                    "unknown tokens into a sequence of byte tokens matching the original piece of text."
                )
                # 发出警告信息

    def vocab(self, proto):
        # 定义一个名为vocab的方法，接受proto参数
        return [(piece.piece, piece.score) for piece in proto.pieces]
        # 返回proto.pieces中每个piece的piece和score组成的列表

    def unk_id(self, proto):
        # 定义一个名为unk_id的方���，接受proto参数
        return proto.trainer_spec.unk_id
        # 返回proto.trainer_spec.unk_id
    # 根据给定的协议(proto)生成分词器
    def tokenizer(self, proto):
        # 获取模型类型
        model_type = proto.trainer_spec.model_type
        # 获取词汇表和分数
        vocab_scores = self.vocab(proto)
        # 获取未知标记的ID
        unk_id = self.unk_id(proto)

        # 根据模型类型选择不同的分词器
        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
        elif model_type == 2:
            # 提取 SentencePiece 模型的词汇表和合并规则
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            # 创建 BPE 分词器的词汇表
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            # 创建 BPE 分词器
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                )
            )
        else:
            # 抛出异常，提示模型类型不匹配
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        # 返回生成的分词器
        return tokenizer

    # 根据给定的协议(proto)生成规范化器
    def normalizer(self, proto):
        # 获取预编译字符映射
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        # 定义规范化器列表
        _normalizers = [
            normalizers.Strip(left=False, right=True),  # 去除空格
            normalizers.Replace(Regex(" {2,}"), "▁"),  # 替换多个空格为特殊符号
        ]
        # 如果没有预编译字符映射，则返回规范化器序列
        if not precompiled_charsmap:
            return normalizers.Sequence(_normalizers)
        else:
            # 否则返回包含预编译字符映射的规范化器序列
            return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    # 根据替换和是否添加前缀空格生成预分词器
    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = "always"
        # 如果原始分词器不是旧版本，则设置前缀方案为"first"
        if hasattr(self.original_tokenizer, "legacy") and not self.original_tokenizer.legacy:
            prepend_scheme = "first"
        # 返回 Metaspace 预分词器
        return pre_tokenizers.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space, prepend_scheme=prepend_scheme
        )

    # 返回空的后处理器
    def post_processor(self):
        return None

    # 根据替换和是否添加前缀空格生成解码器
    def decoder(self, replacement, add_prefix_space):
        return decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

    # 生成转换后的分词器
    def converted(self) -> Tokenizer:
        # 生成分词器
        tokenizer = self.tokenizer(self.proto)

        # 分词器组装
        # 生成规范化器
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        # 生成预分词器
        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        # 设置解码器
        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        # 生成后处理器
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        # 返回转换后的分词器
        return tokenizer
# 定义一个名为 AlbertConverter 的类，继承自 SpmConverter 类
class AlbertConverter(SpmConverter):
    # 定义一个名为 vocab 的方法，接受 proto 参数
    def vocab(self, proto):
        # 返回一个列表，列表元素是元组，每个元组包含 piece.piece 和 piece.score，如果 piece.piece 符合 check_number_comma 条件，否则将 score 减去 100
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    # 定义一个名为 normalizer 的方法，接受 proto 参数
    def normalizer(self, proto):
        # 初始化一个列表 list_normalizers，包含两个 normalizers.Replace 对象，分别用于替换 "``" 和 "''" 为 '"'
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        # 如果不保留重音符号，添加两个 normalizers 对象 NFKD 和 StripAccents 到 list_normalizers
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        # 如果需要小写化，添加 normalizers.Lowercase 到 list_normalizers
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        # 获取 proto.normalizer_spec.precompiled_charsmap，并赋值给 precompiled_charsmap
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        # 如果 precompiled_charsmap 存在，添加 normalizers.Precompiled(precompiled_charsmap) 到 list_normalizers
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        # 添加一个 normalizers.Replace 对象，用于替换多个连续空格为一个空格
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        # 返回一个 normalizers.Sequence 对象，其中包含 list_normalizers 中的所有 normalizers 对象
        return normalizers.Sequence(list_normalizers)

    # 定义一个名为 post_processor 的方法
    def post_processor(self):
        # 返回一个 processors.TemplateProcessing 对象，用于处理模板
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


# 定义一个名为 BarthezConverter 的类，继承自 SpmConverter 类
class BarthezConverter(SpmConverter):
    # 定义一个名为 unk_id 的方法，接受 proto 参数
    def unk_id(self, proto):
        # 设置 unk_id 的值为 3
        unk_id = 3
        # 返回 unk_id
        return unk_id

    # 定义一个名为 post_processor 的方法
    def post_processor(self):
        # 返回一个 processors.TemplateProcessing 对象，用于处理模板
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


# 定义一个名为 CamembertConverter 的类，继承自 SpmConverter 类
class CamembertConverter(SpmConverter):
    # 定义一个名为 vocab 的方法，接受 proto 参数
    def vocab(self, proto):
        # 初始化一个列表 vocab，包含预定义的特殊 token 和它们的分数
        vocab = [
            ("<s>NOTUSED", 0.0),
            ("<pad>", 0.0),
            ("</s>NOTUSED", 0.0),
            ("<unk>", 0.0),
            ("<unk>NOTUSED", -100),
        ]
        # 将 proto.pieces 中除了第一个元素之外的每个 piece 的 piece 和 score 组成元组，添加到 vocab 中
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[1:]]
        # 添加 ("<mask>", 0.0) 到 vocab
        vocab += [("<mask>", 0.0)]
        # 返回 vocab
        return vocab

    # 定义一个名为 unk_id 的方法，接受 proto 参数
    def unk_id(self, proto):
        # 返回 3，表示未知 token 的 id
        return 3

    # 定义一个名为 post_processor 的方法
    def post_processor(self):
        # 返回一个 processors.TemplateProcessing 对象，用于处理模板
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


# 定义一个名为 DebertaV2Converter 的类，继承自 SpmConverter 类，但是这段代码不完整，缺少了方法的定义
class DebertaV2Converter(SpmConverter):
    # 定义一个预处理器函数，用于生成预处理器序列
    def pre_tokenizer(self, replacement, add_prefix_space):
        # 初始化预处理器列表
        list_pretokenizers = []
        # 如果原始分词器按标点符号分割
        if self.original_tokenizer.split_by_punct:
            # 添加一个标点符号预处理器到列表中
            list_pretokenizers.append(pre_tokenizers.Punctuation(behavior="isolated"))
        # 添加一个空格预处理器到列表中
        list_pretokenizers.append(pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space))
        # 返回预处理器序列
        return pre_tokenizers.Sequence(list_pretokenizers)

    # 定义一个规范化器函数，用于生成规范化器序列
    def normalizer(self, proto):
        # 初始化规范化器列表
        list_normalizers = []
        # 如果原始分词器执行小写转换
        if self.original_tokenizer.do_lower_case:
            # 添加一个小写转换规范化器到列表中
            list_normalizers.append(normalizers.Lowercase())
        # 添加一个去除空格规范化器到列表中
        list_normalizers.append(normalizers.Strip())

        # 获取预编译字符映射
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        # 如果存在预编译字符映射
        if precompiled_charsmap:
            # 添加一个预编译规范化器到列表中
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        # 添加一个替换规范化器到列表中，用于替换多余空格
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))

        # 返回规范化器序列
        return normalizers.Sequence(list_normalizers)

    # 定义一个后处理器函数，用于生成后处理器对象
    def post_processor(self):
        # 返回一个模板处理器对象，用于处理单句和双句情况
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
# 定义一个类 MBartConverter，继承自 SpmConverter
class MBartConverter(SpmConverter):
    # 定义一个方法 vocab，用于返回词汇表
    def vocab(self, proto):
        # 初始化词汇表，包含一些特殊标记和对应的分数
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        # 将 proto 中的 pieces 转换为词汇表的形式，从第四个元素开始添加到词汇表中
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # 添加一些特殊标记和对应的分数到词汇表中
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
        # 添加一个特殊标记 "<mask>" 到词汇表中
        vocab += [("<mask>", 0.0)]
        # 返回词汇表
        return vocab

    # 定义一个方法 unk_id，用于返回未知标记的 id
    def unk_id(self, proto):
        # 返回未知标记的 id，这里返回的是 3
        return 3

    # 定义一个方法 post_processor，用于返回后处理器
    def post_processor(self):
        # 返回一个模板处理器，包含单句和双句的模板，以及一些特殊标记和对应的 id
        return processors.TemplateProcessing(
            single="$A </s> en_XX",
            pair="$A $B </s> en_XX",
            special_tokens=[
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


# 定义一个类 MBart50Converter，继承自 SpmConverter
class MBart50Converter(SpmConverter):
    # 定义一个方法 vocab，用于返回词汇表
    def vocab(self, proto):
        # 初始化词汇表，包含一些特殊标记和对应的分数
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        # 将 proto 中的 pieces 转换为词汇表的形式，从第四个元素开始添加到词汇表中
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # 添加一些特殊标记和对应的分数到词汇表中
        vocab += [("ar_AR", 0.0), ("cs_CZ", 0.0), ("de_DE", 0.0), ("en_XX", 0.0), ("es_XX", 0.0), ("et_EE", 0.0), ("fi_FI", 0.0), ("fr_XX", 0.0), ("gu_IN", 0.0), ("hi_IN", 0.0), ("it_IT", 0.0), ("ja_XX", 0.0), ("kk_KZ", 0.0), ("ko_KR", 0.0), ("lt_LT", 0.0), ("lv_LV", 0.0), ("my_MM", 0.0), ("ne_NP", 0.0), ("nl_XX", 0.0), ("ro_RO", 0.0), ("ru_RU", 0.0), ("si_LK", 0.0), ("tr_TR", 0.0), ("vi_VN", 0.0), ("zh_CN", 0.0), ("af_ZA", 0.0), ("az_AZ", 0.0), ("bn_IN", 0.0), ("fa_IR", 0.0), ("he_IL", 0.0), ("hr_HR", 0.0), ("id_ID", 0.0), ("ka_GE", 0.0), ("km_KH", 0.0), ("mk_MK", 0.0), ("ml_IN", 0.0), ("mn_MN", 0.0), ("mr_IN", 0.0), ("pl_PL", 0.0), ("ps_AF", 0.0), ("pt_XX", 0.0), ("sv_SE", 0.0), ("sw_KE", 0.0), ("ta_IN", 0.0), ("te_IN", 0.0), ("th_TH", 0.0), ("tl_XX", 0.0), ("uk_UA", 0.0), ("ur_PK", 0.0), ("xh_ZA", 0.0), ("gl_ES", 0.0), ("sl_SI", 0.0)]  # fmt: skip
        # 添加一个特殊标记 "<mask>" 到词汇表中
        vocab += [("<mask>", 0.0)]
        # 返回词汇表
        return vocab

    # 定义一个方法 unk_id，用于返回未知标记的 id
    def unk_id(self, proto):
        # 返回未知标记的 id，这里返回的是 3
        return 3
    # 定义一个方法用于后处理
    def post_processor(self):
        # 返回一个模板处理器对象
        return processors.TemplateProcessing(
            # 设置单句模板，包含en_XX $A </s>
            single="en_XX $A </s>",
            # 设置双句模板，包含en_XX $A $B </s>
            pair="en_XX $A $B </s>",
            # 设置特殊标记的对应关系，包括en_XX和</s>的id映射关系
            special_tokens=[
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )
# 定义 NllbConverter 类，继承自 SpmConverter 类
class NllbConverter(SpmConverter):
    # 定义 unk_id 方法，返回值为 3
    def unk_id(self, proto):
        return 3

    # 定义 post_processor 方法，返回 TemplateProcessing 处理器对象
    def post_processor(self):
        return processors.TemplateProcessing(
            single="eng_Latn $A </s>",  # 单句模板
            pair="eng_Latn $A $B </s>",  # 双句模板
            special_tokens=[  # 特殊标记列表
                ("eng_Latn", self.original_tokenizer.convert_tokens_to_ids("eng_Latn")),  # eng_Latn 标记
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),  # </s> 标记
            ],
        )


# 定义 SeamlessM4TConverter 类，继承自 SpmConverter 类
class SeamlessM4TConverter(SpmConverter):
    # 定义 vocab 方法，返回词汇表
    def vocab(self, proto):
        vocab = [
            ("<pad>", 0.0),  # 填充标记
            ("<unk>", 0.0),  # 未知标记
            ("<s>", 0.0),  # 起始标记
            ("</s>", 0.0),  # 结束标记
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]  # 将 proto 中的词片段添加到词汇表
        return vocab

    # 定义 unk_id 方法，返回原始分词器的未知标记 ID
    def unk_id(self, proto):
        return self.original_tokenizer.unk_token_id

    # 定义 post_processor 方法，返回 TemplateProcessing 处理器对象
    def post_processor(self):
        return processors.TemplateProcessing(
            single="__eng__ $A </s>",  # 单句模板
            pair="__eng__ $A $B </s>",  # 双句模板
            special_tokens=[  # 特殊标记列表
                ("__eng__", self.original_tokenizer.convert_tokens_to_ids("__eng__")),  # __eng__ 标记
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),  # </s> 标记
            ],
        )


# 定义 XLMRobertaConverter 类，继承自 SpmConverter 类
class XLMRobertaConverter(SpmConverter):
    # 定义 vocab 方法，返回词汇表
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),  # 起始标记
            ("<pad>", 0.0),  # 填充标记
            ("</s>", 0.0),  # 结束标记
            ("<unk>", 0.0),  # 未知标记
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]  # 将 proto 中的词片段添加到词汇表
        vocab += [("<mask>", 0.0)]  # 添加 <mask> 标记
        return vocab

    # 定义 unk_id 方法，返回值为 3
    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    # 定义 post_processor 方法，返回 TemplateProcessing 处理器对象
    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A </s>",  # 单句模板
            pair="<s> $A </s> </s> $B </s>",  # 双句模板
            special_tokens=[  # 特殊标记列表
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),  # <s> 标记
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),  # </s> 标记
            ],
        )


# 定义 XLNetConverter 类，继承自 SpmConverter 类
class XLNetConverter(SpmConverter):
    # 定义 vocab 方法，返回词汇表
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]
    # 定义文本标准化函数，接受一个标准化器对象和协议参数
    def normalizer(self, proto):
        # 创建一个标准化器列表，包含两个替换标准化器，用于将双反引号替换为双引号
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        # 如果原始分词器不保留重音符号，则添加 NFKD 和去除重音符号的标准化器
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        # 如果原始分词器进行小写处理，则添加小写化标准化器
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        # 获取预编译的字符映射
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        # 如果存在预编译字符映射，则添加预编译标准化器
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        # 添加替换多个连续空格为单个空格的标准化器
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        # 返回一个标准化器序列，按顺序应用列表中的标准化器
        return normalizers.Sequence(list_normalizers)

    # 定义后处理器函数
    def post_processor(self):
        # 返回一个模板处理器，用于对单个文本和文本对进行处理
        return processors.TemplateProcessing(
            # 定义单个文本的模板，将A标记为0，分隔符标记为0，CLS标记为2
            single="$A:0 <sep>:0 <cls>:2",
            # 定义文本对的模板，将A标记为0，分隔符标记为0，B标记为1，分隔符标记为1，CLS标记为2
            pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
            # 定义特殊标记，将分隔符和CLS标记转换为其对应的词汇ID
            special_tokens=[
                ("<sep>", self.original_tokenizer.convert_tokens_to_ids("<sep>")),
                ("<cls>", self.original_tokenizer.convert_tokens_to_ids("<cls>")),
            ],
        )
class ReformerConverter(SpmConverter):
    pass



class RemBertConverter(SpmConverter):
    # 从AlbertConverter中获得灵感
    def normalizer(self, proto):
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        return normalizers.Sequence(list_normalizers)

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
    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0.0),
            (self.original_tokenizer.eos_token, 0.0),
        ]

        if self.original_tokenizer.mask_token_sent is not None:
            vocab += [(self.original_tokenizer.mask_token_sent, 0.0)]

        if (
            self.original_tokenizer.mask_token is not None
            and self.original_tokenizer.mask_token_id < self.original_tokenizer.offset
        ):
            vocab += [(self.original_tokenizer.mask_token, 0.0)]

        vocab += [(f"<unk_{i}>", -100.0) for i in range(2, self.original_tokenizer.offset)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[2:]]
        return vocab

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id + self.original_tokenizer.offset

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            ]
        )

    def post_processor(self):
        eos = self.original_tokenizer.eos_token
        special_tokens = [
            (eos, self.original_tokenizer.eos_token_id),
        ]
        return processors.TemplateProcessing(single=["$A", eos], pair=["$A", "$B", eos], special_tokens=special_tokens)



class T5Converter(SpmConverter):
    # 定义一个方法用于生成词汇表，参数为预训练模型的 protobuf 对象
    def vocab(self, proto):
        # 获取额外标识符的数量
        num_extra_ids = self.original_tokenizer._extra_ids
        # 从给定的 protobuf 对象中提取词汇表并转换成列表，每个元素是一个包含词片段和对应得分的元组
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        # 将额外标识符添加到词汇表中，每个额外标识符的得分设置为0.0
        vocab += [(f"<extra_id_{i}>", 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        # 返回生成的词汇表
        return vocab

    # 定义一个方法用于创建后处理器对象
    def post_processor(self):
        # 返回一个模板处理器对象，用于处理单个和成对的序列
        return processors.TemplateProcessing(
            # 定义单个序列的处理模板，包含起始标记"$A"和终止标记"</s>"
            single=["$A", "</s>"],
            # 定义成对序列的处理模板，包含起始标记"$A"、终止标记"</s>"、起始标记"$B"和终止标记"</s>"
            pair=["$A", "</s>", "$B", "</s>"],
            # 定义特殊标记及其对应的标识符，此处仅包含终止标记"</s>"
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )
# 定义一个名为WhisperConverter的类，继承自Converter类
class WhisperConverter(Converter):
    # 定义一个名为converted的方法，返回一个Tokenizer对象
    def converted(self) -> Tokenizer:
        # 获取原始分词器的编码器
        vocab = self.original_tokenizer.encoder
        # 获取原始分词器的BPE合并列表
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        # 创建一个Tokenizer对象，使用BPE分词器
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

        # 设置分词器的预处理器为ByteLevel
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        # 设置分词器的解码器为ByteLevel
        tokenizer.decoder = decoders.ByteLevel()

        # 获取原始分词器的前缀标记ID
        prefix_token_ids = self.original_tokenizer.prefix_tokens
        # 将前缀标记ID转换为标记
        prefixes = self.original_tokenizer.convert_ids_to_tokens(prefix_token_ids)
        # 获取原始分词器的结束标记和结束标记ID
        eos = self.original_tokenizer.eos_token
        eos_token_id = self.original_tokenizer.eos_token_id
        # 创建前缀模板
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        # 设置分词器的后处理器为TemplateProcessing
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{prefix_template} $A:0 {eos}:0",
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            special_tokens=[
                (eos, eos_token_id),
                *zip(prefixes, prefix_token_ids),
            ],
        )

        # 返回创建的分词器对象
        return tokenizer


# 定义一个名为BigBirdConverter的类，继承自SpmConverter类
class BigBirdConverter(SpmConverter):
    # 定义一个名为post_processor的方法
    def post_processor(self):
        # 返回一个TemplateProcessing对象
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


# 定义一个名为CLIPConverter的类，继承自Converter类
class CLIPConverter(Converter):
    # 将原始的分词器转换为新的分词器格式
    def converted(self) -> Tokenizer:
        # 获取原始分词器的词汇表
        vocab = self.original_tokenizer.encoder
        # 获取原始分词器的BPE合并列表
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        # 获取原始分词器的未知标记
        unk_token = self.original_tokenizer.unk_token

        # 创建新的分词器对象
        tokenizer = Tokenizer(
            # 使用BPE分词器
            BPE(
                vocab=vocab,  # 原始分词器的词汇表
                merges=merges,  # 原始分词器的BPE合并列表
                dropout=None,  # 丢弃率设置为None
                continuing_subword_prefix="",  # 子词前缀设置为空字符串
                end_of_word_suffix="</w>",  # 单词结束后缀设置为"</w>"
                fuse_unk=False,  # 不融合未知标记
                unk_token=str(unk_token),  # 未知标记转换为字符串
            )
        )

        # 设置分词器的规范化器
        tokenizer.normalizer = normalizers.Sequence(
            # 使用一系列规范化器
            [
                normalizers.NFC(),  # 使用NFC规范化器
                normalizers.Replace(Regex(r"\s+"), " "),  # 替换连续空格为单个空格
                normalizers.Lowercase()  # 转换为小写
            ]
        )
        # 设置分词器的预处理器
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            # 使用一系列预处理器
            [
                pre_tokenizers.Split(
                    Regex(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""),
                    behavior="removed",  # 移除匹配到的模式
                    invert=True,  # 不反转结果
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),  # 使用字节级别分词器
            ]
        )
        # 设置分词器的解码器
        tokenizer.decoder = decoders.ByteLevel()

        # 用于处理ByteLevel和TemplaceProcessor的Hack
        tokenizer.post_processor = processors.RobertaProcessing(
            # 设置句子结束的标记和对应的ID
            sep=(self.original_tokenizer.eos_token, self.original_tokenizer.eos_token_id),
            # 设置类开始的标记和对应的ID
            cls=(self.original_tokenizer.bos_token, self.original_tokenizer.bos_token_id),
            add_prefix_space=False,  # 不在前缀空格上添加
            trim_offsets=False,  # 不修剪偏移
        )
        # 返回新的分词器对象
        return tokenizer
class LayoutLMv2Converter(Converter):
    # LayoutLMv2 转换器类，继承自 Converter 类
    def converted(self) -> Tokenizer:
        # 转换方法，返回 Tokenizer 对象
        vocab = self.original_tokenizer.vocab
        # 获取原始分词器的词汇表
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))
        # 创建 Tokenizer 对象，使用 WordPiece 分词器和未知标记

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = True
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            # 检查原始分词器是否有 basic_tokenizer 属性
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # 设置 Tokenizer 的正规化器为 BertNormalizer

        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        # 设置 Tokenizer 的预处理器为 BertPreTokenizer

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        # 设置 Tokenizer 的后处理器为 TemplateProcessing

        tokenizer.decoder = decoders.WordPiece(prefix="##")
        # 设置 Tokenizer 的解码器为 WordPiece，前缀为 "##"

        return tokenizer
        # 返回 Tokenizer 对象


class BlenderbotConverter(Converter):
    # Blenderbot 转换器类，继承自 Converter 类
    def converted(self) -> Tokenizer:
        # 转换方法，返回 Tokenizer 对象
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

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
        # 创建 Tokenizer 对象，使用 BPE 分词器和相关参数

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        # 设置 Tokenizer 的预处理器为 ByteLevel

        tokenizer.decoder = decoders.ByteLevel()
        # 设置 Tokenizer 的解码器为 ByteLevel

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A:0 {ot.eos_token}:0",
            special_tokens=[
                (ot.eos_token, ot.eos_token_id),
            ],
        )
        # 设置 Tokenizer 的后处理器为 TemplateProcessing

        return tokenizer
        # 返回 Tokenizer 对象


class XGLMConverter(SpmConverter):
    # XGLM 转换器类，继承自 SpmConverter 类
    def vocab(self, proto):
        # 获取词汇表方法
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        # 初始化词汇表，包含特殊标记
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # 将 proto 中的词片段添加到词汇表
        vocab += [("<madeupword0>", 0.0), ("<madeupword1>", 0.0), ("<madeupword2>", 0.0), ("<madeupword3>", 0.0), ("<madeupword4>", 0.0), ("<madeupword5>", 0.0), ("<madeupword6>", 0.0)]  # fmt: skip
        # 添加虚构词到词汇表
        return vocab
        # 返回词汇表

    def unk_id(self, proto):
        # 获取未知标记 ID 方法
        unk_id = 3
        # 设置未知标记 ID 为 3
        return unk_id
        # 返回未知标记 ID
    # 定义一个后处理器函数，用于处理生成的文本
    def post_processor(self):
        # 返回一个模板处理器对象，用于处理单个和成对的文本
        return processors.TemplateProcessing(
            # 定义单个文本的模板，将 $A 替换为生成的文本
            single="</s> $A",
            # 定义成对文本的模板，将 $A 和 $B 替换为生成的文本
            pair="</s> $A </s> </s> $B",
            # 定义特殊标记和对应的标记 ID，用于后续处理
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )
# 定义了一个名为 LlamaConverter 的类，它继承自 SpmConverter 类
class LlamaConverter(SpmConverter):
    # 设置一个布尔型属性，表示是否使用字节回退
    handle_byte_fallback = True

    # 定义了一个方法，用于生成词汇表
    def vocab(self, proto):
        # 初始化词汇表，包含三个特殊标记
        vocab = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        # 将从 proto 中获取的片段添加到词汇表中
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # 返回生成的词汇表
        return vocab

    # 定义了一个方法，用于获取未知标记的 ID
    def unk_id(self, proto):
        # 未知标记的 ID 默认为 0
        unk_id = 0
        # 返回未知标记的 ID
        return unk_id

    # 定义了一个方法，用于生成解码器
    def decoder(self, replacement, add_prefix_space):
        # 返回一个序列解码器，其中包含了一系列解码器操作
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(content=" ", left=1),
            ]
        )

    # 定义了一个方法，用于生成分词器
    def tokenizer(self, proto):
        # 获取模型类型
        model_type = proto.trainer_spec.model_type
        # 获取词汇表和分数
        vocab_scores = self.vocab(proto)
        # 根据模型类型进行分词器的初始化
        if model_type == 1:
            # 导入 tokenizers 模块
            import tokenizers
            # 根据 tokenizers 模块版本选择不同的初始化方式
            if version.parse(tokenizers.__version__) < version.parse("0.14.0"):
                tokenizer = Tokenizer(Unigram(vocab_scores, 0))
            else:
                tokenizer = Tokenizer(Unigram(vocab_scores, 0, byte_fallback=True))

        elif model_type == 2:
            # 导入 SentencePieceExtractor 类，并根据词汇表和分数进行初始化
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            # 根据 BPE 算法进行初始化
            tokenizer = Tokenizer(
                BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True)
            )
            # 添加特殊标记到分词器中
            tokenizer.add_special_tokens(
                [
                    AddedToken("<unk>", normalized=False, special=True),
                    AddedToken("<s>", normalized=False, special=True),
                    AddedToken("</s>", normalized=False, special=True),
                ]
            )
        else:
            # 抛出异常，表示不支持该模型类型
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        # 返回初始化后的分词器
        return tokenizer

    # 定义了一个方法，用于生成规范化器
    def normalizer(self, proto):
        # 返回一个序列规范化器，其中包含了一系列规范化器操作
        return normalizers.Sequence(
            [
                normalizers.Prepend(prepend="▁"),
                normalizers.Replace(pattern=" ", content="▁"),
            ]
        )

    # 定义了一个方法，用于生成预分词器
    def pre_tokenizer(self, replacement, add_prefix_space):
        # 返回空值，表示不使用预分词器
        return None

    # 定义了一个方法，用于生成后处理器
    def post_processor(self):
        # 返回空值，表示后处理器在 LlamaTokenizerFast 类中定义
        return None


class MarkupLMConverter(Converter):


How do you feel about the explanations I provided?
    # 将原始的分词器转换为新的分词器
    def converted(self) -> Tokenizer:
        # 获取原始分词器的编码器和BPE合并列表
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        # 创建新的分词器对象
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                unk_token=self.original_tokenizer.unk_token,
            )
        )

        # 设置新分词器的预处理器和解码器
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()

        # 获取原始分词器的特殊标记和对应的ID
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        # 设置新分词器的后处理器，包括单句和双句模板以及特殊标记
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls} $A {sep}",
            pair=f"{cls} $A {sep} $B {sep}",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )

        # 返回新的分词器对象
        return tokenizer
# 定义一个字典，将慢速分词器类名映射到相应的转换器类上
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
    "WhisperTokenizer": WhisperConverter,
    "XLMRobertaTokenizer": XLMRobertaConverter,
    "XLNetTokenizer": XLNetConverter,
    "SplinterTokenizer": SplinterConverter,
    "XGLMTokenizer": XGLMConverter,
    "LlamaTokenizer": LlamaConverter,
    "CodeLlamaTokenizer": LlamaConverter,
}

# 将慢速分词器转换为快速分词器的实用程序函数
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

    # 获取传入分词器实例的类名
    tokenizer_class_name = transformer_tokenizer.__class__.__name__
    # 检查 tokenizer_class_name 是否在 SLOW_TO_FAST_CONVERTERS 字典中
    if tokenizer_class_name not in SLOW_TO_FAST_CONVERTERS:
        # 如果不在字典中，则抛出数值错误异常，提示无法将指定的慢速 tokenizer 类转换为快速 tokenizer 实例
        raise ValueError(
            f"An instance of tokenizer class {tokenizer_class_name} cannot be converted in a Fast tokenizer instance."
            " No converter was found. Currently available slow->fast convertors:"
            f" {list(SLOW_TO_FAST_CONVERTERS.keys())}"
        )

    # 获取对应的转换器类
    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]

    # 返回转换后的快速 tokenizer 实例
    return converter_class(transformer_tokenizer).converted()
```