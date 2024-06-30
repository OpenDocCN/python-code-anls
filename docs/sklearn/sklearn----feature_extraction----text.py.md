# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\text.py`

```
"""Utilities to build feature vectors from text documents."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import array
import re
import unicodedata
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from numbers import Integral
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..exceptions import NotFittedError
from ..preprocessing import normalize
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from ..utils.fixes import _IS_32BIT
from ..utils.validation import FLOAT_DTYPES, check_array, check_is_fitted
from ._hash import FeatureHasher
from ._stop_words import ENGLISH_STOP_WORDS

__all__ = [
    "HashingVectorizer",
    "CountVectorizer",
    "ENGLISH_STOP_WORDS",
    "TfidfTransformer",
    "TfidfVectorizer",
    "strip_accents_ascii",
    "strip_accents_unicode",
    "strip_tags",
]


def _preprocess(doc, accent_function=None, lower=False):
    """Chain together an optional series of text preprocessing steps to
    apply to a document.

    Parameters
    ----------
    doc: str
        The string to preprocess
    accent_function: callable, default=None
        Function for handling accented characters. Common strategies include
        normalizing and removing.
    lower: bool, default=False
        Whether to use str.lower to lowercase all of the text

    Returns
    -------
    doc: str
        preprocessed string
    """
    if lower:
        doc = doc.lower()
    if accent_function is not None:
        doc = accent_function(doc)
    return doc


def _analyze(
    doc,
    analyzer=None,
    tokenizer=None,
    ngrams=None,
    preprocessor=None,
    decoder=None,
    stop_words=None,
):
    """Chain together an optional series of text processing steps to go from
    a single document to ngrams, with or without tokenizing or preprocessing.

    If analyzer is used, only the decoder argument is used, as the analyzer is
    intended to replace the preprocessor, tokenizer, and ngrams steps.

    Parameters
    ----------
    analyzer: callable, default=None
        Function to analyze the document directly, overriding other steps.
    tokenizer: callable, default=None
        Function to tokenize the document into words or tokens.
    ngrams: callable, default=None
        Function to generate ngrams from tokens.
    preprocessor: callable, default=None
        Function to preprocess the document before tokenization.
    decoder: callable, default=None
        Function to decode the document, if necessary.
    stop_words: list, default=None
        List of words to be excluded from tokenization.

    Returns
    -------
    ngrams: list
        A sequence of tokens, possibly with pairs, triples, etc.
    """

    if decoder is not None:
        doc = decoder(doc)  # Decode the document if a decoder function is provided
    if analyzer is not None:
        doc = analyzer(doc)  # Analyze the document directly if an analyzer function is provided
    else:
        if preprocessor is not None:
            doc = preprocessor(doc)  # Preprocess the document if a preprocessor function is provided
        if tokenizer is not None:
            doc = tokenizer(doc)  # Tokenize the document if a tokenizer function is provided
        if ngrams is not None:
            if stop_words is not None:
                doc = ngrams(doc, stop_words)  # Generate ngrams from tokens with stop words removal if provided
            else:
                doc = ngrams(doc)  # Generate ngrams from tokens if no stop words list is provided
    # 返回函数中存储的变量 doc 的值作为函数的输出结果
    return doc
# 将带重音符号的Unicode符号转换为它们的简单对应项。
# 注意：Python级别的循环和连接操作使得这种实现比strip_accents_ascii基本的规范化慢20倍。

def strip_accents_unicode(s):
    try:
        # 如果`s`是ASCII兼容的，则不包含任何重音字符，可以避免昂贵的列表理解
        s.encode("ASCII", errors="strict")
        return s
    except UnicodeEncodeError:
        # 使用NFKD规范化`s`，将其转换为分解形式
        normalized = unicodedata.normalize("NFKD", s)
        # 从规范化后的字符串中过滤掉组合字符，以移除重音符号
        return "".join([c for c in normalized if not unicodedata.combining(c)])


# 将带重音符号的Unicode符号转换为ASCII或空字符。
# 注意：此解决方案仅适用于能直接转换为ASCII符号的语言。

def strip_accents_ascii(s):
    # 使用NFKD规范化`s`，将其转换为分解形式
    nkfd_form = unicodedata.normalize("NFKD", s)
    # 忽略非ASCII字符，然后将结果解码为ASCII字符串
    return nkfd_form.encode("ASCII", "ignore").decode("ASCII")


# 基于正则表达式的HTML/XML标签去除函数。
# 对于严格的HTML/XML预处理，建议使用外部库，如lxml或BeautifulSoup。

def strip_tags(s):
    # 使用正则表达式移除所有HTML/XML标签，并用空格替换
    return re.compile(r"<([^>]+)>", flags=re.UNICODE).sub(" ", s)


# 检查停用词列表，返回相应的集合或None。
# 如果`stop`为"english"，则返回内置的英文停用词列表ENGLISH_STOP_WORDS。
# 如果`stop`是字符串，则引发值错误。
# 如果`stop`为None，则返回None。
# 否则，假定它是一个集合，返回其冻结集合形式。

def _check_stop_list(stop):
    if stop == "english":
        return ENGLISH_STOP_WORDS
    elif isinstance(stop, str):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    else:  # 假设它是一个集合
        return frozenset(stop)


class _VectorizerMixin:
    """提供文本向量化器（分词逻辑）的通用代码。"""

    # 用于匹配多个空白字符的正则表达式
    _white_spaces = re.compile(r"\s\s+")
    def decode(self, doc):
        """Decode the input into a string of unicode symbols.

        The decoding strategy depends on the vectorizer parameters.

        Parameters
        ----------
        doc : bytes or str
            The string to decode.

        Returns
        -------
        doc: str
            A string of unicode symbols.
        """
        # 如果输入为文件名，使用二进制模式打开文件并读取内容
        if self.input == "filename":
            with open(doc, "rb") as fh:
                doc = fh.read()

        # 如果输入为文件对象，直接读取其内容
        elif self.input == "file":
            doc = doc.read()

        # 如果输入是字节类型，则根据指定的编码和解码错误方式解码为字符串
        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)

        # 如果文档为 np.nan，则抛出异常，要求输入必须是字节或unicode字符串
        if doc is np.nan:
            raise ValueError(
                "np.nan is an invalid document, expected byte or unicode string."
            )

        # 返回解码后的字符串文档
        return doc

    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # 处理停用词
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # 处理词语的 n-gram
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # 对于单字词不需要进行切片，直接遍历原始 tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # 将方法绑定到循环外以减少开销
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i : i + n]))

        return tokens

    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        # 规范化空白字符
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        min_n, max_n = self.ngram_range
        if min_n == 1:
            # 对于单字符 n-gram 不需要进行切片，直接遍历字符串
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []

        # 将方法绑定到循环外以减少开销
        ngrams_append = ngrams.append

        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngrams_append(text_document[i : i + n])
        return ngrams
    def _char_wb_ngrams(self, text_document):
        """Whitespace sensitive char-n-gram tokenization.

        Tokenize text_document into a sequence of character n-grams
        operating only inside word boundaries. n-grams at the edges
        of words are padded with space."""
        # 将文本中的多个空白字符规范化为单个空格
        text_document = self._white_spaces.sub(" ", text_document)

        min_n, max_n = self.ngram_range
        ngrams = []

        # 将方法绑定到循环外以减少开销
        ngrams_append = ngrams.append

        # 对文本按空格分割后的每个单词进行处理
        for w in text_document.split():
            w = " " + w + " "  # 在单词两侧添加空格，以便处理边界情况
            w_len = len(w)
            for n in range(min_n, max_n + 1):
                offset = 0
                # 提取当前单词中的所有 n-grams，并添加到 ngrams 列表中
                ngrams_append(w[offset : offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams_append(w[offset : offset + n])
                if offset == 0:  # 当单词长度小于 n 时，只计算一次
                    break
        return ngrams

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization.

        Returns
        -------
        preprocessor: callable
              A function to preprocess the text before tokenization.
        """
        if self.preprocessor is not None:
            return self.preprocessor

        # 去除重音符号
        if not self.strip_accents:
            strip_accents = None
        elif callable(self.strip_accents):
            strip_accents = self.strip_accents
        elif self.strip_accents == "ascii":
            strip_accents = strip_accents_ascii
        elif self.strip_accents == "unicode":
            strip_accents = strip_accents_unicode
        else:
            raise ValueError(
                'Invalid value for "strip_accents": %s' % self.strip_accents
            )

        # 返回预处理函数，包括去除重音和转换为小写（如果设置）
        return partial(_preprocess, accent_function=strip_accents, lower=self.lowercase)

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens.

        Returns
        -------
        tokenizer: callable
              A function to split a string into a sequence of tokens.
        """
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)

        if token_pattern.groups > 1:
            raise ValueError(
                "More than 1 capturing group in token pattern. Only a single "
                "group should be captured."
            )

        # 返回一个函数，将字符串分割成 token 序列
        return token_pattern.findall

    def get_stop_words(self):
        """Build or fetch the effective stop words list.

        Returns
        -------
        stop_words: list or None
                A list of stop words.
        """
        # 获取或构建有效的停止词列表
        return _check_stop_list(self.stop_words)
    def _check_stop_words_consistency(self, stop_words, preprocess, tokenize):
        """Check if stop words are consistent
        
        Returns
        -------
        is_consistent : True if stop words are consistent with the preprocessor
                        and tokenizer, False if they are not, None if the check
                        was previously performed, "error" if it could not be
                        performed (e.g. because of the use of a custom
                        preprocessor / tokenizer)
        """
        # 如果当前实例的停用词与之前的停用词相同，则说明停用词已经验证过
        if id(self.stop_words) == getattr(self, "_stop_words_id", None):
            # 停用词已经验证过
            return None

        # 注意：stop_words 是被验证过的，而不是 self.stop_words
        try:
            inconsistent = set()
            # 遍历停用词列表（如果有的话），或者空列表
            for w in stop_words or ():
                # 对每个停用词进行预处理后进行分词
                tokens = list(tokenize(preprocess(w)))
                # 检查每个分词后的 token 是否在停用词列表中
                for token in tokens:
                    if token not in stop_words:
                        inconsistent.add(token)
            # 记录当前实例的停用词 id
            self._stop_words_id = id(self.stop_words)

            # 如果存在不一致的 token，则发出警告
            if inconsistent:
                warnings.warn(
                    "Your stop_words may be inconsistent with "
                    "your preprocessing. Tokenizing the stop "
                    "words generated tokens %r not in "
                    "stop_words." % sorted(inconsistent)
                )
            # 返回是否一致的布尔值
            return not inconsistent
        except Exception:
            # 处理停用词一致性检查失败的情况（例如由于使用了自定义预处理器或分词器）
            self._stop_words_id = id(self.stop_words)
            return "error"
    # 构建并返回一个可调用对象，用于处理输入数据
    def build_analyzer(self):
        """Return a callable to process input data.

        The callable handles preprocessing, tokenization, and n-grams generation.

        Returns
        -------
        analyzer: callable
            A function to handle preprocessing, tokenization
            and n-grams generation.
        """

        # 如果 self.analyzer 是可调用对象，则返回部分应用的 _analyze 函数
        if callable(self.analyzer):
            return partial(_analyze, analyzer=self.analyzer, decoder=self.decode)

        # 构建预处理器
        preprocess = self.build_preprocessor()

        # 如果 self.analyzer 是 "char"，则返回部分应用的 _analyze 函数，使用字符 n-grams
        if self.analyzer == "char":
            return partial(
                _analyze,
                ngrams=self._char_ngrams,
                preprocessor=preprocess,
                decoder=self.decode,
            )

        # 如果 self.analyzer 是 "char_wb"，则返回部分应用的 _analyze 函数，使用带边界的字符 n-grams
        elif self.analyzer == "char_wb":
            return partial(
                _analyze,
                ngrams=self._char_wb_ngrams,
                preprocessor=preprocess,
                decoder=self.decode,
            )

        # 如果 self.analyzer 是 "word"，则返回部分应用的 _analyze 函数，使用词语 n-grams
        elif self.analyzer == "word":
            # 获取停用词集合和构建的分词器
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()
            # 检查停用词的一致性
            self._check_stop_words_consistency(stop_words, preprocess, tokenize)
            return partial(
                _analyze,
                ngrams=self._word_ngrams,
                tokenizer=tokenize,
                preprocessor=preprocess,
                decoder=self.decode,
                stop_words=stop_words,
            )

        # 如果 self.analyzer 不是以上三种值，则引发 ValueError 异常
        else:
            raise ValueError(
                "%s is not a valid tokenization scheme/analyzer" % self.analyzer
            )

    # 验证词汇表的有效性
    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        # 如果词汇表不为 None
        if vocabulary is not None:
            # 如果词汇表是集合，则将其排序为列表
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            # 如果词汇表不是 Mapping 对象，则转换为索引到词项的映射
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(vocabulary.values())
                # 检查映射值的唯一性
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in range(len(vocabulary)):
                    if i not in indices:
                        msg = "Vocabulary of size %d doesn't contain index %d." % (
                            len(vocabulary),
                            i,
                        )
                        raise ValueError(msg)
            # 如果词汇表为空，则引发 ValueError 异常
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            # 设置标志指示词汇表已固定，并保存词汇表的副本
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False
    # 检查词汇表是否为空或缺失（未拟合）
    def _check_vocabulary(self):
        """Check if vocabulary is empty or missing (not fitted)"""
        # 如果对象中没有"vocabulary_"属性，表示词汇表为空或未拟合，需要进行验证
        if not hasattr(self, "vocabulary_"):
            self._validate_vocabulary()
            # 如果词汇表不是固定的，则抛出未拟合错误
            if not self.fixed_vocabulary_:
                raise NotFittedError("Vocabulary not fitted or provided")

        # 如果词汇表长度为0，则抛出值错误
        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary is empty")

    # 验证 ngram_range 参数的有效性
    def _validate_ngram_range(self):
        """Check validity of ngram_range parameter"""
        # 获取 ngram_range 的最小值和最大值
        min_n, max_m = self.ngram_range
        # 如果最小值大于最大值，抛出值错误
        if min_n > max_m:
            raise ValueError(
                "Invalid value for ngram_range=%s "
                "lower boundary larger than the upper boundary." % str(self.ngram_range)
            )

    # 警告未使用的参数
    def _warn_for_unused_params(self):
        # 如果 tokenizer 和 token_pattern 都不为 None，则发出警告，token_pattern 参数不会被使用
        if self.tokenizer is not None and self.token_pattern is not None:
            warnings.warn(
                "The parameter 'token_pattern' will not be used"
                " since 'tokenizer' is not None'"
            )

        # 如果 preprocessor 不为 None 并且 analyzer 是可调用的，则发出警告，preprocessor 参数不会被使用
        if self.preprocessor is not None and callable(self.analyzer):
            warnings.warn(
                "The parameter 'preprocessor' will not be used"
                " since 'analyzer' is callable'"
            )

        # 如果 ngram_range 不是 (1, 1) 并且 analyzer 是可调用的，则发出警告，ngram_range 参数不会被使用
        if (
            self.ngram_range != (1, 1)
            and self.ngram_range is not None
            and callable(self.analyzer)
        ):
            warnings.warn(
                "The parameter 'ngram_range' will not be used"
                " since 'analyzer' is callable'"
            )

        # 如果 analyzer 不是 "word" 或者 analyzer 是可调用的，则根据不同情况发出相关警告
        if self.analyzer != "word" or callable(self.analyzer):
            # 如果 stop_words 不为 None，则发出警告，stop_words 参数不会被使用
            if self.stop_words is not None:
                warnings.warn(
                    "The parameter 'stop_words' will not be used"
                    " since 'analyzer' != 'word'"
                )
            # 如果 token_pattern 不为 None 并且不是默认的正则表达式，则发出警告，token_pattern 参数不会被使用
            if (
                self.token_pattern is not None
                and self.token_pattern != r"(?u)\b\w\w+\b"
            ):
                warnings.warn(
                    "The parameter 'token_pattern' will not be used"
                    " since 'analyzer' != 'word'"
                )
            # 如果 tokenizer 不为 None，则发出警告，tokenizer 参数不会被使用
            if self.tokenizer is not None:
                warnings.warn(
                    "The parameter 'tokenizer' will not be used"
                    " since 'analyzer' != 'word'"
                )
class HashingVectorizer(
    TransformerMixin, _VectorizerMixin, BaseEstimator, auto_wrap_output_keys=None
):
    r"""Convert a collection of text documents to a matrix of token occurrences.

    It turns a collection of text documents into a scipy.sparse matrix holding
    token occurrence counts (or binary occurrence information), possibly
    normalized as token frequencies if norm='l1' or projected on the euclidean
    unit sphere if norm='l2'.

    This text vectorizer implementation uses the hashing trick to find the
    token string name to feature integer index mapping.

    This strategy has several advantages:

    - it is very low memory scalable to large datasets as there is no need to
      store a vocabulary dictionary in memory.

    - it is fast to pickle and un-pickle as it holds no state besides the
      constructor parameters.

    - it can be used in a streaming (partial fit) or parallel pipeline as there
      is no state computed during fit.

    There are also a couple of cons (vs using a CountVectorizer with an
    in-memory vocabulary):

    - there is no way to compute the inverse transform (from feature indices to
      string feature names) which can be a problem when trying to introspect
      which features are most important to a model.

    - there can be collisions: distinct tokens can be mapped to the same
      feature index. However in practice this is rarely an issue if n_features
      is large enough (e.g. 2 ** 18 for text classification problems).

    - no IDF weighting as this would render the transformer stateful.

    The hash function employed is the signed 32-bit version of Murmurhash3.

    For an efficiency comparison of the different feature extractors, see
    :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

    For an example of document clustering and comparison with
    :class:`~sklearn.feature_extraction.text.TfidfVectorizer`, see
    :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    input : {'filename', 'file', 'content'}, default='content'
        - If `'filename'`, the sequence passed as an argument to fit is
          expected to be a list of filenames that need reading to fetch
          the raw content to analyze.

        - If `'file'`, the sequence items must have a 'read' method (file-like
          object) that is called to fetch the bytes in memory.

        - If `'content'`, the input is expected to be a sequence of items that
          can be of type string or byte.

    encoding : str, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.
    # decode_error 参数指定了在分析包含非给定编码字符的字节序列时的处理方式。
    # 默认为 'strict'，表示会引发 UnicodeDecodeError 错误。
    # 可选的值包括 'ignore'（忽略错误）和 'replace'（替换错误）。
    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    # strip_accents 参数指定了是否在预处理步骤中去除重音符号并进行其他字符规范化。
    # 'ascii' 是一种快速的方法，仅适用于直接具有 ASCII 映射的字符。
    # 'unicode' 是一种稍慢的方法，适用于任何字符。
    # None（默认）表示不进行字符规范化。
    # 'ascii' 和 'unicode' 方法都使用 :func:`unicodedata.normalize` 中的 NFKD 规范化。
    strip_accents : {'ascii', 'unicode'} or callable, default=None
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        a direct ASCII mapping.
        'unicode' is a slightly slower method that works on any character.
        None (default) means no character normalization is performed.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    # lowercase 参数指定是否在分词之前将所有字符转换为小写。
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.

    # preprocessor 参数允许在保留分词和 n-grams 生成步骤的同时，重写预处理（字符串转换）阶段。
    # 仅当 ``analyzer`` 不是可调用对象时适用。
    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer`` is not callable.

    # tokenizer 参数允许在保留预处理和 n-grams 生成步骤的同时，重写字符串的分词步骤。
    # 仅当 ``analyzer == 'word'`` 时适用。
    tokenizer : callable, default=None
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    # stop_words 参数指定停用词列表的使用方式。
    # 如果为 'english'，则使用内置的英文停用词列表。
    # 'english' 存在一些已知问题，建议考虑替代方案（参见 :ref:`stop_words`）。
    # 如果是一个列表，则假定该列表包含所有要从结果标记中移除的停用词。
    # 仅当 ``analyzer == 'word'`` 时适用。
    stop_words : {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

    # token_pattern 参数是一个正则表达式，用于确定什么构成一个“标记”。
    # 仅当 ``analyzer == 'word'`` 时使用。
    # 默认的正则表达式选择包含至少两个字母数字字符的标记（完全忽略标点符号并始终将其视为标记分隔符）。
    # 如果 token_pattern 中有一个捕获组，则捕获组内容而不是整个匹配成为标记。
    # 最多允许一个捕获组。
    token_pattern : str or None, default=r"(?u)\\b\\w\\w+\\b"
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

        If there is a capturing group in token_pattern then the
        captured group content, not the entire match, becomes the token.
        At most one capturing group is permitted.

    # ngram_range 参数指定从文本中提取的 n-gram 的 n 值范围。
    # min_n 和 max_n 分别表示最小和最大 n 值的边界。
    # 所有 n 的值满足 min_n <= n <= max_n 将被使用。
    # 例如，``ngram_range`` 为 ``(1, 1)`` 表示只使用单个词（unigrams），
    # ``(1, 2)`` 表示同时使用单个词和双词组合（unigrams 和 bigrams），
    # ``(2, 2)`` 表示只使用双词组合（bigrams）。
    # 仅当 ``analyzer`` 不是可调用对象时适用。
    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams.
        Only applies if ``analyzer`` is not callable.
    analyzer : {'word', 'char', 'char_wb'} or callable, default='word'
        # 分析器类型，可以是 'word'、'char'、'char_wb' 或可调用对象，默认为'word'
        # 'char_wb'选项仅从单词边界内的文本创建字符 n-gram；单词边缘的 n-gram 使用空格填充
        # 如果传入可调用对象，则用于从原始未处理的输入中提取特征序列
        .. versionchanged:: 0.21
            # 自版本0.21起，如果输入为'filename'或'file'，则先从文件中读取数据，然后传递给指定的分析器

    n_features : int, default=(2 ** 20)
        # 输出矩阵中的特征数（列数），较小的特征数可能导致哈希冲突，但较大的特征数会导致线性学习器中较大的系数维度

    binary : bool, default=False
        # 如果为True，则所有非零计数将设置为1，这对于离散概率模型（而不是整数计数）很有用

    norm : {'l1', 'l2'}, default='l2'
        # 用于规范化术语向量的规范。None表示不规范化

    alternate_sign : bool, default=True
        # 当为True时，将交替的符号添加到特征中，以在哈希空间中大约保持内积。类似于稀疏随机投影的方法

        .. versionadded:: 0.19
            # 自版本0.19起添加的功能

    dtype : type, default=np.float64
        # fit_transform()或transform()返回的矩阵的类型

    See Also
    --------
    CountVectorizer : 将文本文档集转换为令牌计数矩阵
    TfidfVectorizer : 将原始文档集转换为TF-IDF特征矩阵

    Notes
    -----
    This estimator is :term:`stateless` and does not need to be fitted.
    However, we recommend to call :meth:`fit_transform` instead of
    :meth:`transform`, as parameter validation is only performed in
    :meth:`fit`.
        # 这个估计器是`无状态`的，不需要拟合。但是，我们建议调用`fit_transform`而不是`transform`，因为参数验证仅在`fit`方法中执行

    Examples
    --------
    >>> from sklearn.feature_extraction.text import HashingVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = HashingVectorizer(n_features=2**4)
    >>> X = vectorizer.fit_transform(corpus)
    >>> print(X.shape)
    (4, 16)
    # 示例代码：使用HashingVectorizer将文本文档集转换为矩阵，打印出矩阵的形状
    _parameter_constraints: dict = {
        "input": [StrOptions({"filename", "file", "content"})],
        "encoding": [str],
        "decode_error": [StrOptions({"strict", "ignore", "replace"})],
        "strip_accents": [StrOptions({"ascii", "unicode"}), None, callable],
        "lowercase": ["boolean"],
        "preprocessor": [callable, None],
        "tokenizer": [callable, None],
        "stop_words": [StrOptions({"english"}), list, None],
        "token_pattern": [str, None],
        "ngram_range": [tuple],
        "analyzer": [StrOptions({"word", "char", "char_wb"}), callable],
        "n_features": [Interval(Integral, 1, np.iinfo(np.int32).max, closed="left")],
        "binary": ["boolean"],
        "norm": [StrOptions({"l1", "l2"}), None],
        "alternate_sign": ["boolean"],
        "dtype": "no_validation",  # delegate to numpy
    }

    # 初始化方法，设置对象的参数
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        n_features=(2**20),
        binary=False,
        norm="l2",
        alternate_sign=True,
        dtype=np.float64,
    ):
        # 设置对象的各个参数
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.alternate_sign = alternate_sign
        self.dtype = dtype

    # partial_fit 方法，用于部分拟合和验证估算器的参数
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : ndarray of shape [n_samples, n_features]
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            HashingVectorizer instance.
        """
        # 这里仅返回 self，表示仅验证参数，不执行实际拟合
        return self

    # 另一个 partial_fit 方法，用于部分拟合和验证估算器的参数
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : ndarray of shape [n_samples, n_features]
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            HashingVectorizer instance.
        """
        # 如果 X 是字符串，则抛出值错误，要求传入的是可迭代的原始文本文档
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        # 调用警告未使用参数的方法
        self._warn_for_unused_params()
        # 验证 ngram 范围的有效性
        self._validate_ngram_range()

        # 获取特征哈希器对象，并对训练数据进行拟合
        self._get_hasher().fit(X, y=y)
        return self

    def transform(self, X):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        # 如果 X 是字符串，则抛出值错误，要求传入的是可迭代的原始文本文档
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        # 再次验证 ngram 范围的有效性
        self._validate_ngram_range()

        # 构建分析器
        analyzer = self.build_analyzer()
        # 使用特征哈希器对文档进行转换，生成文档-术语矩阵
        X = self._get_hasher().transform(analyzer(doc) for doc in X)
        # 如果设置了二进制模式，则将数据的值填充为 1
        if self.binary:
            X.data.fill(1)
        # 如果设置了归一化选项，则对 X 进行归一化处理
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    def fit_transform(self, X, y=None):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        # 调用 fit 方法拟合数据，然后对拟合后的结果进行转换
        return self.fit(X, y).transform(X)

    def _get_hasher(self):
        # 返回一个特征哈希器对象，配置了指定的参数
        return FeatureHasher(
            n_features=self.n_features,
            input_type="string",
            dtype=self.dtype,
            alternate_sign=self.alternate_sign,
        )

    def _more_tags(self):
        # 返回一个包含额外标签信息的字典，说明处理的数据类型是字符串
        return {"X_types": ["string"]}
# 定义一个函数 `_document_frequency`，用于计算稀疏矩阵 X 中每个特征非零值的数量。
def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    # 检查输入的 X 是否是稀疏矩阵并且格式为 CSR 格式
    if sp.issparse(X) and X.format == "csr":
        # 使用 numpy 的 bincount 函数统计非零值的数量，minlength 参数设定矩阵的列数
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        # 如果 X 不是稀疏矩阵或者不是 CSR 格式，则计算相邻元素的差分
        return np.diff(X.indptr)


# 定义一个类 CountVectorizer，继承自 _VectorizerMixin 和 BaseEstimator
class CountVectorizer(_VectorizerMixin, BaseEstimator):
    r"""Convert a collection of text documents to a matrix of token counts.

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    If you do not provide an a-priori dictionary and you do not use an analyzer
    that does some kind of feature selection then the number of features will
    be equal to the vocabulary size found by analyzing the data.

    For an efficiency comparison of the different feature extractors, see
    :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    input : {'filename', 'file', 'content'}, default='content'
        - If `'filename'`, the sequence passed as an argument to fit is
          expected to be a list of filenames that need reading to fetch
          the raw content to analyze.

        - If `'file'`, the sequence items must have a 'read' method (file-like
          object) that is called to fetch the bytes in memory.

        - If `'content'`, the input is expected to be a sequence of items that
          can be of type string or byte.

    encoding : str, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode'} or callable, default=None
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        a direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) means no character normalization is performed.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable, default=None
        Override the preprocessing (strip_accents and lowercase) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer`` is not callable.
    # tokenizer : callable, default=None
    # 定义了一个可调用对象 tokenizer，默认为 None
    # 用于在保留预处理和 n-gram 生成步骤的同时，覆盖字符串的分词步骤
    # 仅当 analyzer == 'word' 时有效。

    # stop_words : {'english'}, list, default=None
    # 定义停用词列表。如果为 'english'，则使用内置的英文停用词列表。
    # 'english' 存在一些已知问题，建议考虑使用替代方案（参见 :ref:`stop_words`）。
    # 如果是列表，则假定列表包含要删除的所有停用词。
    # 仅当 analyzer == 'word' 时有效。

    # token_pattern : str or None, default=r"(?u)\\b\\w\\w+\\b"
    # 定义什么构成一个“标记”的正则表达式模式，仅当 analyzer == 'word' 时使用。
    # 默认正则表达式选择包含 2 个或更多个字母数字字符的标记
    # （标点符号完全被忽略，并始终被视为标记分隔符）。
    # 如果 token_pattern 中有捕获组，则捕获组内容而不是整个匹配成为标记。
    # 最多允许一个捕获组。

    # ngram_range : tuple (min_n, max_n), default=(1, 1)
    # 提取不同词 n-gram 或字符 n-gram 的 n 值范围的下限和上限。
    # 所有满足 min_n <= n <= max_n 的 n 值将被使用。
    # 例如，ngram_range (1, 1) 表示只有单个词项，(1, 2) 表示单个词项和双词项，
    # (2, 2) 表示只有双词项。
    # 仅当 analyzer 不是可调用对象时有效。

    # analyzer : {'word', 'char', 'char_wb'} or callable, default='word'
    # 定义了特征应该是由单词 n-gram 还是字符 n-gram 组成。
    # 选项 'char_wb' 仅从单词边界内的文本创建字符 n-gram；
    # 词的边缘的 n-gram 用空格填充。
    # 如果传递了可调用对象，则用于从原始未处理的输入中提取特征序列。
    # 自版本 0.21 起，如果输入是“filename”或“file”，则先从文件中读取数据，
    # 然后传递给给定的可调用分析器函数。

    # max_df : float in range [0.0, 1.0] or int, default=1.0
    # 在构建词汇表时，忽略文档频率严格高于给定阈值的术语（特定于语料库的停用词）。
    # 如果是浮点数，则表示文档的比例；如果是整数，则表示绝对计数。
    # 如果 vocabulary 不是 None，则忽略此参数。
    # min_df参数用于构建词汇表时忽略文档频率低于给定阈值的词语。
    # 它可以是一个介于0.0和1.0之间的浮点数，表示文档比例，也可以是整数，表示绝对文档计数。
    # 在没有提供自定义词汇表的情况下，此参数被忽略。

    # max_features参数指定构建词汇表时只考虑按词频在整个语料库中排名前`max_features`的词语。
    # 如果为None，则使用所有的特征。

    # 在没有提供自定义词汇表的情况下，此参数被忽略。

    # vocabulary参数可以是一个Mapping（例如字典），其中键是词语，值是特征矩阵中的索引，或者是一个词语的可迭代对象。
    # 如果没有提供，则从输入文档中确定词汇表。
    # 映射中的索引不应重复，并且在0到最大索引之间不应有任何间隙。

    # binary参数如果为True，则所有非零计数都设置为1。
    # 这对于模型二进制事件的概率模型很有用，而不是整数计数。

    # dtype参数指定fit_transform()或transform()返回的矩阵类型。

    # vocabulary_属性：词汇表的词语到特征索引的映射。

    # fixed_vocabulary_属性：如果用户提供了固定的词汇表，即词语到索引的映射，则为True。

    # HashingVectorizer参见：将文本文档集合转换为令牌计数矩阵。

    # TfidfVectorizer参见：将原始文档集合转换为TF-IDF特征矩阵。
    """
    Parameter constraints for initializing a class instance.
    Specifies the expected types and constraints for each parameter.
    
    _parameter_constraints: dict = {
        "input": [StrOptions({"filename", "file", "content"})],
        "encoding": [str],
        "decode_error": [StrOptions({"strict", "ignore", "replace"})],
        "strip_accents": [StrOptions({"ascii", "unicode"}), None, callable],
        "lowercase": ["boolean"],
        "preprocessor": [callable, None],
        "tokenizer": [callable, None],
        "stop_words": [StrOptions({"english"}), list, None],
        "token_pattern": [str, None],
        "ngram_range": [tuple],
        "analyzer": [StrOptions({"word", "char", "char_wb"}), callable],
        "max_df": [
            Interval(RealNotInt, 0, 1, closed="both"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "min_df": [
            Interval(RealNotInt, 0, 1, closed="both"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_features": [Interval(Integral, 1, None, closed="left"), None],
        "vocabulary": [Mapping, HasMethods("__iter__"), None],
        "binary": ["boolean"],
        "dtype": "no_validation",  # delegate to numpy
    }
    
    Class initializer with keyword-only arguments.
    Initializes an instance of the class with specified default values for parameters.
    
    def __init__(
        self,
        *,
        input="content",              # 默认参数：输入内容为"content"
        encoding="utf-8",             # 默认参数：编码方式为"utf-8"
        decode_error="strict",        # 默认参数：解码错误处理方式为"strict"
        strip_accents=None,           # 默认参数：不去除重音符号
        lowercase=True,               # 默认参数：将文本转为小写
        preprocessor=None,            # 默认参数：预处理器为None
        tokenizer=None,               # 默认参数：分词器为None
        stop_words=None,              # 默认参数：停用词为None
        token_pattern=r"(?u)\b\w\w+\b", # 默认参数：匹配单词的正则表达式
        ngram_range=(1, 1),           # 默认参数：n-gram范围为(1, 1)
        analyzer="word",              # 默认参数：分析方式为单词级别
        max_df=1.0,                   # 默认参数：文档频率上限为1.0
        min_df=1,                     # 默认参数：文档频率下限为1
        max_features=None,            # 默认参数：最大特征数为None
        vocabulary=None,              # 默认参数：词汇表为None
        binary=False,                 # 默认参数：不使用二值化
        dtype=np.int64,               # 默认参数：数据类型为np.int64
    ):
        Initializes instance attributes with provided or default values.
    
        self.input = input             # 将输入参数赋值给实例属性self.input
        self.encoding = encoding       # 将编码方式赋值给实例属性self.encoding
        self.decode_error = decode_error  # 将解码错误处理方式赋值给实例属性self.decode_error
        self.strip_accents = strip_accents  # 将去重音符号的方式赋值给实例属性self.strip_accents
        self.preprocessor = preprocessor    # 将预处理器赋值给实例属性self.preprocessor
        self.tokenizer = tokenizer          # 将分词器赋值给实例属性self.tokenizer
        self.analyzer = analyzer            # 将分析方式赋值给实例属性self.analyzer
        self.lowercase = lowercase          # 将是否转为小写赋值给实例属性self.lowercase
        self.token_pattern = token_pattern  # 将正则表达式赋值给实例属性self.token_pattern
        self.stop_words = stop_words        # 将停用词赋值给实例属性self.stop_words
        self.max_df = max_df                # 将文档频率上限赋值给实例属性self.max_df
        self.min_df = min_df                # 将文档频率下限赋值给实例属性self.min_df
        self.max_features = max_features    # 将最大特征数赋值给实例属性self.max_features
        self.ngram_range = ngram_range      # 将n-gram范围赋值给实例属性self.ngram_range
        self.vocabulary = vocabulary        # 将词汇表赋值给实例属性self.vocabulary
        self.binary = binary                # 将是否二值化赋值给实例属性self.binary
        self.dtype = dtype                  # 将数据类型赋值给实例属性self.dtype
    """
    def _sort_features(self, X, vocabulary):
        """Sort features by name

        Returns a reordered matrix and modifies the vocabulary in place
        """
        # 对词汇表按名称排序，返回排序后的矩阵，并在原地修改词汇表
        sorted_features = sorted(vocabulary.items())
        # 创建一个与排序后特征数量相同的索引映射数组
        map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
        # 遍历排序后的特征，更新词汇表和索引映射数组
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        # 使用索引映射数组调整特征矩阵的索引
        X.indices = map_index.take(X.indices, mode="clip")
        return X

    def _limit_features(self, X, vocabulary, high=None, low=None, limit=None):
        """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.

        This does not prune samples with zero features.
        """
        # 如果未指定任何限制条件，则直接返回原始特征矩阵和空集合
        if high is None and low is None and limit is None:
            return X, set()

        # 计算基于文档频率的掩码
        dfs = _document_frequency(X)
        mask = np.ones(len(dfs), dtype=bool)
        # 根据指定的上限删除文档频率高于 high 的特征
        if high is not None:
            mask &= dfs <= high
        # 根据指定的下限删除文档频率低于 low 的特征
        if low is not None:
            mask &= dfs >= low
        # 如果指定了限制并且掩码的数量超过限制，则根据频率对特征进行修剪
        if limit is not None and mask.sum() > limit:
            tfs = np.asarray(X.sum(axis=0)).ravel()
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        # 创建新的索引数组，将旧索引映射到新索引
        new_indices = np.cumsum(mask) - 1
        # 更新词汇表，删除不在掩码内的特征
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
        # 获取保留的特征索引
        kept_indices = np.where(mask)[0]
        # 如果没有保留任何特征，则引发异常
        if len(kept_indices) == 0:
            raise ValueError(
                "After pruning, no terms remain. Try a lower min_df or a higher max_df."
            )
        return X[:, kept_indices]
    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False"""
        # 如果 fixed_vocab=True，则使用现有的词汇表 self.vocabulary_
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # 当 fixed_vocab=False 时，创建一个默认字典作为词汇表
            # 默认字典的工厂方法设置为返回当前词汇表的长度
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        # 使用 self.build_analyzer() 创建分析器 analyze
        analyze = self.build_analyzer()
        
        # 初始化 j_indices 和 indptr 列表
        j_indices = []
        indptr = []

        # 初始化 values 数组，用于存储特征计数
        values = _make_int_array()

        # 将 0 添加到 indptr，表示第一个文档的起始点
        indptr.append(0)

        # 遍历原始文档集合 raw_documents
        for doc in raw_documents:
            # 创建特征计数器，用于统计每个特征在当前文档中的出现次数
            feature_counter = {}
            # 遍历分析器 analyze 处理后的特征列表
            for feature in analyze(doc):
                try:
                    # 获取特征在词汇表中的索引 feature_idx
                    feature_idx = vocabulary[feature]
                    # 更新特征计数器中特征索引对应的计数
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # 当 fixed_vocab=True 时，忽略词汇表外的特征项
                    continue

            # 将特征计数器中的特征索引和计数值扩展到 j_indices 和 values 中
            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            # 将当前文档处理后 j_indices 的长度添加到 indptr 中，表示新文档的起始位置
            indptr.append(len(j_indices))

        # 当 fixed_vocab=False 时，将默认字典类型的词汇表转换为普通字典类型
        if not fixed_vocab:
            vocabulary = dict(vocabulary)
            # 如果词汇表为空，则抛出 ValueError 异常
            if not vocabulary:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only contain stop words"
                )

        # 检查最后一个 indptr 是否超过 np.int32 的最大值（2**31 - 1）
        if indptr[-1] > np.iinfo(np.int32).max:
            # 如果是 32 位 Python 环境，则抛出异常，因为需要 64 位索引
            if _IS_32BIT:
                raise ValueError(
                    (
                        "sparse CSR array has {} non-zero "
                        "elements and requires 64 bit indexing, "
                        "which is unsupported with 32 bit Python."
                    ).format(indptr[-1])
                )
            # 否则，设置索引的数据类型为 np.int64
            indices_dtype = np.int64
        else:
            # 否则，设置索引的数据类型为 np.int32
            indices_dtype = np.int32

        # 将 j_indices 转换为 numpy 数组，并设置数据类型为 indices_dtype
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        # 将 indptr 转换为 numpy 数组，并设置数据类型为 indices_dtype
        indptr = np.asarray(indptr, dtype=indices_dtype)
        # 将 values 转换为 numpy 数组，并设置数据类型为 np.intc
        values = np.frombuffer(values, dtype=np.intc)

        # 创建稀疏 CSR 矩阵 X，使用 values、j_indices 和 indptr
        # 矩阵形状为 (文档数, 词汇表长度)，数据类型为 self.dtype
        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(vocabulary)),
            dtype=self.dtype,
        )

        # 对矩阵 X 的索引进行排序
        X.sort_indices()

        # 返回词汇表和稀疏矩阵 X
        return vocabulary, X

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        y : None
            This parameter is ignored.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        # 调用 self.fit_transform(raw_documents) 学习文档集中所有标记的词汇表
        self.fit_transform(raw_documents)
        # 返回拟合后的向量化器对象 self
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return document-term matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        y : None
            This parameter is ignored.

        Returns
        -------
        X : array of shape (n_samples, n_features)
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.

        # 如果 raw_documents 是字符串，则抛出错误，期望接收的是原始文本文档的可迭代对象
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        # 调用内部方法，验证 ngram 范围是否有效
        self._validate_ngram_range()
        # 如果有未使用的参数，发出警告
        self._warn_for_unused_params()
        # 验证词汇表的有效性
        self._validate_vocabulary()

        # 备份参数到局部变量
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        # 如果词汇表固定且转换为小写，检查词汇表中是否包含大写字符并发出警告
        if self.fixed_vocabulary_ and self.lowercase:
            for term in self.vocabulary:
                if any(map(str.isupper, term)):
                    warnings.warn(
                        "Upper case characters found in"
                        " vocabulary while 'lowercase'"
                        " is True. These entries will not"
                        " be matched with any documents"
                    )
                    break

        # 计算词汇表和文档-词频矩阵
        vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)

        # 如果 binary=True，将 X 中的非零元素设置为1
        if self.binary:
            X.data.fill(1)

        # 如果词汇表不固定，进一步处理特征限制和排序
        if not self.fixed_vocabulary_:
            n_doc = X.shape[0]
            max_doc_count = max_df if isinstance(max_df, Integral) else max_df * n_doc
            min_doc_count = min_df if isinstance(min_df, Integral) else min_df * n_doc
            if max_doc_count < min_doc_count:
                raise ValueError("max_df corresponds to < documents than min_df")
            if max_features is not None:
                X = self._sort_features(X, vocabulary)
            X = self._limit_features(
                X, vocabulary, max_doc_count, min_doc_count, max_features
            )
            if max_features is None:
                X = self._sort_features(X, vocabulary)
            self.vocabulary_ = vocabulary

        # 返回处理后的文档-词频矩阵 X
        return X
    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )
        self._check_vocabulary()  # 检查词汇表的有效性

        # 使用与 fit_transform 相同的矩阵构建策略
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)  # 计算文档-词项矩阵
        if self.binary:
            X.data.fill(1)  # 如果是二值化模式，将所有非零元素设置为1
        return X  # 返回文档-词项矩阵

    def inverse_transform(self, X):
        """Return terms per document with nonzero entries in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document-term matrix.

        Returns
        -------
        X_inv : list of arrays of shape (n_samples,)
            List of arrays of terms.
        """
        self._check_vocabulary()  # 检查词汇表的有效性
        # 需要 CSR 格式进行快速行操作
        X = check_array(X, accept_sparse="csr")  # 确保 X 是 CSR 稀疏矩阵格式
        n_samples = X.shape[0]

        terms = np.array(list(self.vocabulary_.keys()))  # 从词汇表中获取词项
        indices = np.array(list(self.vocabulary_.values()))  # 获取词汇表中的索引
        inverse_vocabulary = terms[np.argsort(indices)]  # 根据索引排序获取逆词汇表

        if sp.issparse(X):
            return [
                inverse_vocabulary[X[i, :].nonzero()[1]].ravel()
                for i in range(n_samples)
            ]  # 返回每个文档中非零条目对应的词项数组
        else:
            return [
                inverse_vocabulary[np.flatnonzero(X[i, :])].ravel()
                for i in range(n_samples)
            ]  # 返回每个文档中非零条目对应的词项数组

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        self._check_vocabulary()  # 检查词汇表的有效性
        return np.asarray(
            [t for t, i in sorted(self.vocabulary_.items(), key=itemgetter(1))],
            dtype=object,
        )  # 返回经过排序后的转换后的特征名称数组

    def _more_tags(self):
        return {"X_types": ["string"]}  # 返回更多的标签信息，表示处理的输入类型为字符串
# 创建一个适合用于 scipy.sparse 索引的 array.array 对象，类型为整型
def _make_int_array():
    return array.array(str("i"))


class TfidfTransformer(
    OneToOneFeatureMixin, TransformerMixin, BaseEstimator, auto_wrap_output_keys=None
):
    """将计数矩阵转换为归一化的 tf 或 tf-idf 表示形式。

    Tf 表示词项频率，而 tf-idf 表示词项频率乘以逆文档频率。这是信息检索中常用的词项加权方案，
    在文档分类中也有很好的应用。

    使用 tf-idf 而不是文档中词项的原始频率的目的是缩小在给定语料库中频繁出现的词项对特征的影响，
    因此这些词项在训练语料库中的信息量相对较少。

    Tf-idf 的计算公式为 tf-idf(t, d) = tf(t, d) * idf(t)，其中 idf(t) 的计算方式为
    idf(t) = log [ n / df(t) ] + 1（如果 ``smooth_idf=False``），其中 n 是文档集合中的总文档数，
    df(t) 是包含词项 t 的文档数；文档频率是包含词项 t 的文档数。在上述公式中，将 "1" 添加到 idf 
    的效果是对于 idf 为零的词项（即出现在训练集中的所有文档中的词项），不会完全被忽略。
    （注意上述 idf 公式与标准教科书定义的 idf 稍有不同，标准教科书定义为 idf(t) = log [ n / (df(t) + 1) ]）。

    如果 ``smooth_idf=True``（默认情况下），在 idf 的分子和分母中分别添加常数 "1"，就好像看到了一个额外的文档，
    其中包含集合中的每个词项正好一次，这样可以防止零除：idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1。

    此外，用于计算 tf 和 idf 的公式依赖于参数设置，这些设置对应于信息检索中使用的 SMART 符号表示：

    - 当 ``sublinear_tf=True`` 时，Tf 是 "l"（对数），默认是 "n"（自然）。
    - 当使用 ``use_idf=True`` 时，Idf 是 "t"，否则是 "n"（无）。
    - 当 ``norm='l2'`` 时，标准化是 "c"（余弦），否则是 "n"（无）。

    更多信息请参阅 :ref:`用户指南 <text_feature_extraction>`。

    Parameters
    ----------
    norm : {'l1', 'l2'} 或 None，默认='l2'
        每个输出行将具有单位范数，可以是：

        - 'l2': 向量元素的平方和为1。当应用 l2 范数后，两个向量之间的余弦相似度是它们的点积。
        - 'l1': 向量元素的绝对值之和为1。
        - None: 不进行标准化。

    use_idf : bool，默认=True
        启用逆文档频率重新加权。如果为 False，则 idf(t) = 1。
    smooth_idf : bool, default=True
        # 平滑 idf 权重，通过将文档频率加一，就好像看到一个包含集合中每个术语的额外文档一次。防止零除法。

    sublinear_tf : bool, default=False
        # 应用子线性 tf 缩放，即用 1 + log(tf) 替换 tf。

    Attributes
    ----------
    idf_ : array of shape (n_features)
        # 逆文档频率（IDF）向量；仅在 ``use_idf`` 为 True 时定义。

        .. versionadded:: 0.20

    n_features_in_ : int
        # 在 :term:`fit` 过程中看到的特征数量。

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 :term:`fit` 过程中看到的特征名称。仅当 `X` 具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    CountVectorizer : 将文本转换为 n-gram 计数的稀疏矩阵。

    TfidfVectorizer : 将一组原始文档转换为 TF-IDF 特征的矩阵。

    HashingVectorizer : 将一组文本文档转换为令牌出现的矩阵。

    References
    ----------
    .. [Yates2011] R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern
                   Information Retrieval. Addison Wesley, pp. 68-74.

    .. [MRS2008] C.D. Manning, P. Raghavan and H. Schütze  (2008).
                   Introduction to Information Retrieval. Cambridge University
                   Press, pp. 118-120.

    Examples
    --------
    >>> from sklearn.feature_extraction.text import TfidfTransformer
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.pipeline import Pipeline
    >>> corpus = ['this is the first document',
    ...           'this document is the second document',
    ...           'and this is the third one',
    ...           'is this the first document']
    >>> vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
    ...               'and', 'one']
    >>> pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
    ...                  ('tfid', TfidfTransformer())]).fit(corpus)
    >>> pipe['count'].transform(corpus).toarray()
    array([[1, 1, 1, 1, 0, 1, 0, 0],
           [1, 2, 0, 1, 1, 1, 0, 0],
           [1, 0, 0, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 0, 1, 0, 0])
    >>> pipe['tfid'].idf_
    array([1.        , 1.22314355, 1.51082562, 1.        , 1.91629073,
           1.        , 1.91629073, 1.91629073])
    >>> pipe.transform(corpus).shape
    (4, 8)
    """

    _parameter_constraints: dict = {
        "norm": [StrOptions({"l1", "l2"}), None],
        "use_idf": ["boolean"],
        "smooth_idf": ["boolean"],
        "sublinear_tf": ["boolean"],
    }
    # 初始化函数，设置 TF-IDF 转换器的参数
    def __init__(self, *, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False):
        # 设置标准化方式，默认为 l2 范数
        self.norm = norm
        # 是否使用 IDF 权重，默认为 True
        self.use_idf = use_idf
        # 是否对 IDF 进行平滑处理，默认为 True
        self.smooth_idf = smooth_idf
        # 是否使用子线性 TF，默认为 False
        self.sublinear_tf = sublinear_tf

    # 装饰器函数，用于拟合 IDF 向量（全局词项权重）
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Learn the idf vector (global term weights).

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            Term/token counts matrix.

        y : None
            不需要此参数来计算 TF-IDF。

        Returns
        -------
        self : object
            拟合后的转换器对象。
        """
        # 验证输入数据 X，接受稀疏格式 ("csr", "csc")，并根据平台是否为 32 位进行处理
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), accept_large_sparse=not _IS_32BIT
        )
        # 如果 X 不是稀疏矩阵，则转换为 csr 格式的稀疏矩阵
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        # 确定 X 的数据类型为 np.float64 或 np.float32，否则为 np.float64
        dtype = X.dtype if X.dtype in (np.float64, np.float32) else np.float64

        # 如果使用 IDF 权重
        if self.use_idf:
            # 获取样本数和特征数
            n_samples, _ = X.shape
            # 计算文档频率
            df = _document_frequency(X)
            # 将 df 转换为指定的数据类型
            df = df.astype(dtype, copy=False)

            # 如果需要进行 IDF 平滑处理
            if self.smooth_idf:
                # 对文档频率 df 进行平滑处理
                df += float(self.smooth_idf)
                n_samples += int(self.smooth_idf)

            # 使用 log+1 而不是 log，以确保具有零 IDF 的术语不会完全被抑制
            # `np.log` 保持 `df` 的数据类型，因此也保持 `dtype` 不变。
            self.idf_ = np.log(n_samples / df) + 1.0

        # 返回拟合后的对象
        return self
    # 使用该方法将输入的计数矩阵转换成 TF 或 TF-IDF 表示

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation.

        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.

        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations. `copy=False` will only be effective with CSR sparse matrix.

        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        
        # 验证输入数据 X，接受 CSR 格式的稀疏矩阵，并可以指定数据类型为 np.float64 或 np.float32
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            copy=copy,
            reset=False,
        )
        
        # 如果 X 不是稀疏矩阵，则将其转换为 CSR 格式
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=X.dtype)

        # 如果设置了 sublinear_tf，对 X 中的数据应用对数变换并加 1
        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1.0

        # 如果模型已经拟合并具有 idf_ 属性，则将 X 中每列对应的索引乘以对应的 idf 值
        if hasattr(self, "idf_"):
            # X 的列可以通过 `X.indices` 访问，并与对应的 `idf` 值相乘
            X.data *= self.idf_[X.indices]

        # 如果设置了 norm，则对 X 进行规范化处理，使用指定的规范化方法
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)

        # 返回转换后的矩阵 X
        return X

    def _more_tags(self):
        # 返回附加的标签信息，指定 X_types 支持的数据类型为 2darray 和 sparse
        return {
            "X_types": ["2darray", "sparse"],
            # 修正：如果 _inplace_csr_row_normalize_l2 接受 np.float16，则可以保留此数据类型
            "preserves_dtype": [np.float64, np.float32],
        }
class TfidfVectorizer(CountVectorizer):
    r"""Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to :class:`CountVectorizer` followed by
    :class:`TfidfTransformer`.

    For an example of usage, see
    :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`.

    For an efficiency comparison of the different feature extractors, see
    :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

    For an example of document clustering and comparison with
    :class:`~sklearn.feature_extraction.text.HashingVectorizer`, see
    :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    input : {'filename', 'file', 'content'}, default='content'
        - If `'filename'`, the sequence passed as an argument to fit is
          expected to be a list of filenames that need reading to fetch
          the raw content to analyze.

        - If `'file'`, the sequence items must have a 'read' method (file-like
          object) that is called to fetch the bytes into memory.

        - If `'content'`, the input is expected to be a sequence of items that
          can be of type string or byte.

    encoding : str, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode'} or callable, default=None
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        a direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) means no character normalization is performed.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer`` is not callable.

    tokenizer : callable, default=None
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    """
    analyzer : {'word', 'char', 'char_wb'} or callable, default='word'
        # 分析器类型，可以是字符串 'word', 'char', 'char_wb'，或者一个可调用对象，默认为 'word'
        Whether the feature should be made of word or character n-grams.
        # 确定特征是由单词还是字符 n-gram 组成
        Option 'char_wb' creates character n-grams only from text inside
        # 'char_wb' 选项仅从单词边界内的文本创建字符 n-gram；单词边缘的 n-gram 会用空格填充
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        # 如果传递了一个可调用对象，则用于从原始未处理的输入中提取特征序列
        out of the raw, unprocessed input.

        .. versionchanged:: 0.21
            # 从版本 0.21 开始，如果输入是 'filename' 或 'file'，则首先从文件中读取数据，然后传递给给定的 callable analyzer.
            Since v0.21, if ``input`` is ``'filename'`` or ``'file'``, the data
            is first read from the file and then passed to the given callable
            analyzer.

    stop_words : {'english'}, list, default=None
        # 停用词选项，可以是字符串 'english'，目前只支持此字符串值
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        # 如果是字符串 'english'，将传递给 _check_stop_list 并返回适当的停用词列表

        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).
        # 'english' 存在一些已知问题，建议考虑替代方案（参见 :ref:`stop_words`）

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        # 如果是列表，则假定该列表包含所有要从结果令牌中删除的停用词

        Only applies if ``analyzer == 'word'``.
        # 仅在 analyzer 为 'word' 时适用

        If None, no stop words will be used. In this case, setting `max_df`
        to a higher value, such as in the range (0.7, 1.0), can automatically detect
        and filter stop words based on intra corpus document frequency of terms.
        # 如果为 None，则不使用停用词。在这种情况下，将 `max_df` 设置为较高的值（如 (0.7, 1.0) 范围内），可以根据文档频率自动检测和过滤停用词

    token_pattern : str, default=r"(?u)\\b\\w\\w+\\b"
        # 表示令牌的正则表达式模式，仅在 `analyzer == 'word'` 时使用。默认正则表达式选择包含至少两个字母数字字符的令牌（标点符号完全被忽略并始终视为令牌分隔符）

        If there is a capturing group in token_pattern then the
        captured group content, not the entire match, becomes the token.
        # 如果在 token_pattern 中有捕获组，则捕获组的内容而不是整个匹配成为令牌。最多只允许一个捕获组。

        At most one capturing group is permitted.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        # n-gram 提取的 n 值范围的下限和上限。所有满足 min_n <= n <= max_n 的 n 值都将被使用。例如，(1, 1) 表示仅有单个词，(1, 2) 表示单个词和二元组，(2, 2) 表示仅有二元组
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams.

        Only applies if ``analyzer`` is not callable.
        # 仅当 analyzer 不是可调用对象时适用

    max_df : float or int, default=1.0
        # 构建词汇表时忽略文档频率严格高于给定阈值的词语（特定于语料库的停用词）
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        # 当构建词汇表时，忽略文档频率高于给定阈值的术语

        If float in range [0.0, 1.0], the parameter represents a proportion of
        documents, integer absolute counts.
        # 如果是在 [0.0, 1.0] 范围内的浮点数，则表示文档的比例；如果是整数，则表示绝对计数

        This parameter is ignored if vocabulary is not None.
        # 如果词汇表不是 None，则忽略此参数
    # 最小文档频率阈值，用于构建词汇表，忽略文档频率低于该阈值的词语
    min_df : float or int, default=1
        # 如果是浮点数且在 [0.0, 1.0] 范围内，则表示文档的比例；如果是整数，则表示绝对文档数
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        # 如果词汇表不为 None，则忽略此参数

    # 最大特征数，如果不为 None，则构建一个只考虑语料库中按词项频率排序的前 `max_features` 个词汇表
    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        `max_features` ordered by term frequency across the corpus.
        Otherwise, all features are used.

        # 如果词汇表不为 None，则忽略此参数

    # 词汇表，可以是映射（如字典）或可迭代的词语集合。如果未给出，则从输入文档中确定词汇表
    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.

        # 如果词汇表不为 None，则忽略此参数

    # 是否使用二值化，如果为 True，则所有非零词频被设置为 1
    binary : bool, default=False
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set `binary` to True, `use_idf` to False and
        `norm` to None to get 0/1 outputs).

    # fit_transform() 或 transform() 返回的矩阵的数据类型
    dtype : dtype, default=float64
        Type of the matrix returned by fit_transform() or transform().

    # 规范化方式，'l2' 表示每行向量的平方和为 1；'l1' 表示每行向量的绝对值和为 1；None 表示不规范化
    norm : {'l1', 'l2'} or None, default='l2'
        Each output row will have unit norm, either:

        - 'l2': Sum of squares of vector elements is 1. The cosine
          similarity between two vectors is their dot product when l2 norm has
          been applied.
        - 'l1': Sum of absolute values of vector elements is 1.
          See :func:`~sklearn.preprocessing.normalize`.
        - None: No normalization.

    # 是否启用逆文档频率重加权
    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting. If False, idf(t) = 1.

    # 是否平滑逆文档频率权重
    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    # 是否应用子线性 tf 缩放，即用 1 + log(tf) 替换 tf
    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    # 属性
    Attributes
    ----------
    # 词汇表的映射，将词语映射到特征矩阵中的索引
    vocabulary_ : dict
        A mapping of terms to feature indices.

    # 是否存在由用户提供的固定词汇表
    fixed_vocabulary_ : bool
        True if a fixed vocabulary of term to indices mapping
        is provided by the user.

    # 逆文档频率 (IDF) 向量数组；仅在 ``use_idf`` 为 True 时定义
    idf_ : array of shape (n_features,)
        The inverse document frequency (IDF) vector; only defined
        if ``use_idf`` is True.

    See Also
    --------
    # 将文本转换为 n-gram 计数的稀疏矩阵
    CountVectorizer : Transforms text into a sparse matrix of n-gram counts.

    # 从提供的计数矩阵执行 TF-IDF 转换
    TfidfTransformer : Performs the TF-IDF transformation from a provided
        matrix of counts.

    Examples
    --------
    # 使用示例
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> vectorizer.get_feature_names_out()
    array(['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third',
           'this'], ...)
    >>> print(X.shape)
    (4, 9)
    """

    # 初始化一个参数约束字典，继承自CountVectorizer的参数约束
    _parameter_constraints: dict = {**CountVectorizer._parameter_constraints}
    _parameter_constraints.update(
        {
            "norm": [StrOptions({"l1", "l2"}), None],
            "use_idf": ["boolean"],
            "smooth_idf": ["boolean"],
            "sublinear_tf": ["boolean"],
        }
    )

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        # 调用父类的构造函数初始化文本向量化器的参数
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )
        # 设置TF-IDF向量化器的特定参数
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    # 将TF-IDF的参数广播到底层的转换器实例，以便于网格搜索和表示

    @property
    def idf_(self):
        """逆文档频率向量，仅在 `use_idf=True` 时定义。

        Returns
        -------
        ndarray of shape (n_features,)
        """
        # 如果尚未拟合（即未调用`fit`方法），则抛出NotFittedError异常
        if not hasattr(self, "_tfidf"):
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this attribute."
            )
        return self._tfidf.idf_

    @idf_.setter
    # 如果未启用 IDF（逆文档频率），则抛出数值错误异常
    def idf_(self, value):
        if not self.use_idf:
            raise ValueError("`idf_` cannot be set when `user_idf=False`.")
        
        # 如果尚未创建 _tfidf 实例，则创建一个新的 TfidfTransformer 实例
        if not hasattr(self, "_tfidf"):
            # 支持从另一个 TfidfTransformer 转移 `idf_`，因此需要在尚不存在时创建转换器实例
            self._tfidf = TfidfTransformer(
                norm=self.norm,
                use_idf=self.use_idf,
                smooth_idf=self.smooth_idf,
                sublinear_tf=self.sublinear_tf,
            )
        
        # 验证词汇表
        self._validate_vocabulary()
        
        # 如果已经存在属性 `vocabulary_`，则检查其长度是否与 value 相等
        if hasattr(self, "vocabulary_"):
            if len(self.vocabulary_) != len(value):
                raise ValueError(
                    "idf length = %d must be equal to vocabulary size = %d"
                    % (len(value), len(self.vocabulary))
                )
        
        # 将 value 赋给 _tfidf 对象的 idf_
        self._tfidf.idf_ = value

    # 检查参数设置，如果 `dtype` 不在 FLOAT_DTYPES 中则发出警告
    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )

    # 使用装饰器进行上下文拟合，支持嵌套验证时跳过
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, raw_documents, y=None):
        """从训练集中学习词汇表和 IDF。

        Parameters
        ----------
        raw_documents : iterable
            生成字符串、unicode 或文件对象的可迭代对象。

        y : None
            计算 tfidf 不需要此参数。

        Returns
        -------
        self : object
            拟合后的向量化器。
        """
        # 检查参数设置
        self._check_params()
        
        # 发出未使用参数警告
        self._warn_for_unused_params()
        
        # 创建 TfidfTransformer 实例
        self._tfidf = TfidfTransformer(
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )
        
        # 调用超类的 fit_transform 方法，获取转换后的数据 X
        X = super().fit_transform(raw_documents)
        
        # 使用 _tfidf 对象拟合 X
        self._tfidf.fit(X)
        
        # 返回自身对象
        return self
    # 定义一个方法用于拟合和转换数据，学习词汇和逆文档频率，并返回文档-术语矩阵
    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return document-term matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        y : None
            This parameter is ignored.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        # 检查模型参数的有效性
        self._check_params()
        # 使用给定的参数初始化 TfidfTransformer 对象
        self._tfidf = TfidfTransformer(
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )
        # 对原始文档进行拟合和转换，返回文档-术语矩阵 X
        X = super().fit_transform(raw_documents)
        # 使用 X 对 TfidfTransformer 进行拟合，学习逆文档频率
        self._tfidf.fit(X)
        # 因为 X 已经是原始文档的转换视图，所以设置 copy=False
        return self._tfidf.transform(X, copy=False)

    # 定义一个方法用于将文档转换为文档-术语矩阵
    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        # 检查模型是否已经拟合，若未拟合则引发异常
        check_is_fitted(self, msg="The TF-IDF vectorizer is not fitted")

        # 对原始文档进行转换，返回文档-术语矩阵 X
        X = super().transform(raw_documents)
        # 使用 TfidfTransformer 对象将 X 进行转换，得到 tf-idf 加权的文档-术语矩阵
        return self._tfidf.transform(X, copy=False)

    # 定义一个方法返回更多标签，用于描述对象特性
    def _more_tags(self):
        return {"X_types": ["string"], "_skip_test": True}
```