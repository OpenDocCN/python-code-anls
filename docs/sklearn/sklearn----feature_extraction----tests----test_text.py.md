# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\tests\test_text.py`

```
# 导入必要的模块和函数
import pickle  # 用于序列化和反序列化 Python 对象
import re  # 用于正则表达式操作
import warnings  # 用于管理警告信息
from collections import defaultdict  # 提供了一个默认字典的类
from collections.abc import Mapping  # ABC（Abstract Base Classes）模块中的映射类
from functools import partial  # 用于创建带有部分参数的函数
from io import StringIO  # 用于内存中读写文本数据
from itertools import product  # 用于生成迭代器的函数

import numpy as np  # 提供多维数组对象和一些操作数组的函数
import pytest  # 测试框架
from numpy.testing import assert_array_almost_equal, assert_array_equal  # numpy 测试工具
from scipy import sparse  # 稀疏矩阵和相关的运算

from sklearn.base import clone  # 复制估计器基类
from sklearn.feature_extraction.text import (  # 文本特征提取器
    ENGLISH_STOP_WORDS,  # 停用词集合
    CountVectorizer,  # 计数向量化器
    HashingVectorizer,  # 哈希向量化器
    TfidfTransformer,  # TF-IDF 转换器
    TfidfVectorizer,  # TF-IDF 向量化器
    strip_accents_ascii,  # 移除 ASCII 编码的重音符号
    strip_accents_unicode,  # 移除 Unicode 编码的重音符号
    strip_tags,  # 移除 HTML/XML 标签
)
from sklearn.model_selection import (  # 模型选择工具
    GridSearchCV,  # 网格搜索交叉验证
    cross_val_score,  # 交叉验证评分
    train_test_split,  # 数据集分割
)
from sklearn.pipeline import Pipeline  # 管道类，用于将多个转换器和估计器链在一起
from sklearn.svm import LinearSVC  # 线性支持向量分类器
from sklearn.utils._testing import (  # 测试工具
    assert_allclose_dense_sparse,  # 检查密集和稀疏矩阵的近似相等性
    assert_almost_equal,  # 检查两个数字的近似相等性
    skip_if_32bit,  # 如果是 32 位平台则跳过测试
)
from sklearn.utils.fixes import _IS_WASM, CSC_CONTAINERS, CSR_CONTAINERS  # 修复工具

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",  # 垃圾食品文档样本
    "the pizza burger beer copyright",  # 垃圾食品文档样本
    "the the pizza beer beer copyright",  # 垃圾食品文档样本
    "the burger beer beer copyright",  # 垃圾食品文档样本
    "the coke burger coke copyright",  # 垃圾食品文档样本
)

NOTJUNK_FOOD_DOCS = (
    "the salad celeri copyright",  # 非垃圾食品文档样本
    "the salad salad sparkling water copyright",  # 非垃圾食品文档样本
    "the the celeri celeri copyright",  # 非垃圾食品文档样本
    "the tomato tomato salad water",  # 非垃圾食品文档样本
    "the tomato salad water copyright",  # 非垃圾食品文档样本
)

ALL_FOOD_DOCS = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS  # 所有食品文档样本的合集


def uppercase(s):
    return strip_accents_unicode(s).upper()  # 移除重音符号并转换为大写


def strip_eacute(s):
    return s.replace("é", "e")  # 将字符串中的 'é' 替换为 'e'


def split_tokenize(s):
    return s.split()  # 按空格分割字符串


def lazy_analyze(s):
    return ["the_ultimate_feature"]  # 返回包含一个特征的列表


def test_strip_accents():
    # 检查一些经典的拉丁重音符号
    a = "àáâãäåçèéêë"
    expected = "aaaaaaceeee"
    assert strip_accents_unicode(a) == expected

    a = "ìíîïñòóôõöùúûüý"
    expected = "iiiinooooouuuuy"
    assert strip_accents_unicode(a) == expected

    # 检查一些阿拉伯文
    a = "\u0625"  # 带下面的哈姆扎的阿勒夫：إ
    expected = "\u0627"  # 简单的阿勒夫：ا
    assert strip_accents_unicode(a) == expected

    # 混合使用有重音符号和没有的字母
    a = "this is à test"
    expected = "this is a test"
    assert strip_accents_unicode(a) == expected

    # 已分解的字符串
    a = "o\u0308"  # 带两点的 o
    expected = "o"
    assert strip_accents_unicode(a) == expected

    # 单独的组合符号
    a = "\u0300\u0301\u0302\u0303"
    expected = ""
    assert strip_accents_unicode(a) == expected

    # 在一个字符上的多个组合符号
    a = "o\u0308\u0304"
    expected = "o"
    assert strip_accents_unicode(a) == expected


def test_to_ascii():
    # 检查一些经典的拉丁重音符号
    a = "àáâãäåçèéêë"
    expected = "aaaaaaceeee"
    assert strip_accents_ascii(a) == expected

    a = "ìíîïñòóôõöùúûüý"
    expected = "iiiinooooouuuuy"
    assert strip_accents_ascii(a) == expected
    # 检查一些阿拉伯语字符
    a = "\u0625"  # halef带有下面的hamza
    expected = ""  # halef没有直接的ASCII匹配
    assert strip_accents_ascii(a) == expected
    
    # 混合带重音和不带重音的字母
    a = "this is à test"
    expected = "this is a test"
    assert strip_accents_ascii(a) == expected
@pytest.mark.parametrize("Vectorizer", (CountVectorizer, HashingVectorizer))
def test_word_analyzer_unigrams(Vectorizer):
    # 使用 pytest 的 parametrize 装饰器，允许使用不同的 Vectorizer 类进行参数化测试

    # 创建指定 Vectorizer 类的文本分析器，并设置去除非 ASCII 字符
    wa = Vectorizer(strip_accents="ascii").build_analyzer()

    # 定义测试文本和预期输出的单词列表
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    expected = [
        "ai",
        "mange",
        "du",
        "kangourou",
        "ce",
        "midi",
        "etait",
        "pas",
        "tres",
        "bon",
    ]

    # 断言分析器处理文本后的输出是否符合预期
    assert wa(text) == expected

    # 第二组测试文本和预期输出的单词列表
    text = "This is a test, really.\n\n I met Harry yesterday."
    expected = ["this", "is", "test", "really", "met", "harry", "yesterday"]

    # 断言分析器处理文本后的输出是否符合预期
    assert wa(text) == expected

    # 使用文件输入的方式创建 Vectorizer 的文本分析器
    wa = Vectorizer(input="file").build_analyzer()

    # 使用 StringIO 模拟文件内容
    text = StringIO("This is a test with a file-like object!")
    expected = ["this", "is", "test", "with", "file", "like", "object"]

    # 断言分析器处理文本后的输出是否符合预期
    assert wa(text) == expected

    # 使用自定义预处理器创建 Vectorizer 的文本分析器
    wa = Vectorizer(preprocessor=uppercase).build_analyzer()

    # 定义测试文本和预期输出的单词列表
    text = "J'ai mangé du kangourou  ce midi,  c'était pas très bon."
    expected = [
        "AI",
        "MANGE",
        "DU",
        "KANGOUROU",
        "CE",
        "MIDI",
        "ETAIT",
        "PAS",
        "TRES",
        "BON",
    ]

    # 断言分析器处理文本后的输出是否符合预期
    assert wa(text) == expected

    # 使用自定义分词器创建 Vectorizer 的文本分析器
    wa = Vectorizer(tokenizer=split_tokenize, strip_accents="ascii").build_analyzer()

    # 定义测试文本和预期输出的单词列表
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    expected = [
        "j'ai",
        "mange",
        "du",
        "kangourou",
        "ce",
        "midi,",
        "c'etait",
        "pas",
        "tres",
        "bon.",
    ]

    # 断言分析器处理文本后的输出是否符合预期
    assert wa(text) == expected


def test_word_analyzer_unigrams_and_bigrams():
    # 创建支持单字和双字分析的 CountVectorizer 对象，并设置字符编码为 Unicode，分析器为单词级别
    wa = CountVectorizer(
        analyzer="word", strip_accents="unicode", ngram_range=(1, 2)
    ).build_analyzer()

    # 定义测试文本和预期输出的单词列表及双字列表
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    expected = [
        "ai",
        "mange",
        "du",
        "kangourou",
        "ce",
        "midi",
        "etait",
        "pas",
        "tres",
        "bon",
        "ai mange",
        "mange du",
        "du kangourou",
        "kangourou ce",
        "ce midi",
        "midi etait",
        "etait pas",
        "pas tres",
        "tres bon",
    ]

    # 断言分析器处理文本后的输出是否符合预期
    assert wa(text) == expected


def test_unicode_decode_error():
    # 设置 CountVectorizer 使用 ASCII 编码，预期处理 Unicode 字符串会导致解码错误
    # 创建 Unicode 编码的测试字符串
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    text_bytes = text.encode("utf-8")

    # 使用 ASCII 编码尝试解析 Unicode 字符串，预期会抛出解码错误异常
    wa = CountVectorizer(ngram_range=(1, 2), encoding="ascii").build_analyzer()
    with pytest.raises(UnicodeDecodeError):
        wa(text_bytes)

    # 创建字符级别分析器，使用 ASCII 编码，预期处理 Unicode 字符串会导致解码错误
    ca = CountVectorizer(
        analyzer="char", ngram_range=(3, 6), encoding="ascii"
    ).build_analyzer()
    with pytest.raises(UnicodeDecodeError):
        ca(text_bytes)
# 定义一个测试函数，用于测试字符级 n-gram 分析器
def test_char_ngram_analyzer():
    # 创建 CountVectorizer 对象，使用字符级分析器，处理时保留 Unicode 编码的重音符号，生成长度为 3 到 6 的 n-gram
    cnga = CountVectorizer(
        analyzer="char", strip_accents="unicode", ngram_range=(3, 6)
    ).build_analyzer()

    # 定义测试文本
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon"
    # 预期的前五个字符级 n-gram
    expected = ["j'a", "'ai", "ai ", "i m", " ma"]
    # 断言生成的前五个字符级 n-gram 与预期结果相同
    assert cnga(text)[:5] == expected

    # 预期的后五个字符级 n-gram
    expected = ["s tres", " tres ", "tres b", "res bo", "es bon"]
    # 断言生成的后五个字符级 n-gram 与预期结果相同
    assert cnga(text)[-5:] == expected

    # 更换测试文本
    text = "This \n\tis a test, really.\n\n I met Harry yesterday"
    # 预期的前五个字符级 n-gram
    expected = ["thi", "his", "is ", "s i", " is"]
    # 断言生成的前五个字符级 n-gram 与预期结果相同
    assert cnga(text)[:5] == expected

    # 预期的后五个字符级 n-gram
    expected = [" yeste", "yester", "esterd", "sterda", "terday"]
    # 断言生成的后五个字符级 n-gram 与预期结果相同
    assert cnga(text)[-5:] == expected

    # 创建新的 CountVectorizer 对象，从文件输入，使用字符级分析器，生成长度为 3 到 6 的 n-gram
    cnga = CountVectorizer(
        input="file", analyzer="char", ngram_range=(3, 6)
    ).build_analyzer()
    # 使用 StringIO 创建一个文件类似对象的文本流
    text = StringIO("This is a test with a file-like object!")
    # 预期的前五个字符级 n-gram
    expected = ["thi", "his", "is ", "s i", " is"]
    # 断言生成的前五个字符级 n-gram 与预期结果相同
    assert cnga(text)[:5] == expected


# 定义一个测试函数，用于测试字符级 word boundaries n-gram 分析器
def test_char_wb_ngram_analyzer():
    # 创建 CountVectorizer 对象，使用字符级 word boundaries 分析器，处理时保留 Unicode 编码的重音符号，生成长度为 3 到 6 的 n-gram
    cnga = CountVectorizer(
        analyzer="char_wb", strip_accents="unicode", ngram_range=(3, 6)
    ).build_analyzer()

    # 定义测试文本
    text = "This \n\tis a test, really.\n\n I met Harry yesterday"
    # 预期的前五个字符级 word boundaries n-gram
    expected = [" th", "thi", "his", "is ", " thi"]
    # 断言生成的前五个字符级 word boundaries n-gram 与预期结果相同
    assert cnga(text)[:5] == expected

    # 预期的后五个字符级 word boundaries n-gram
    expected = ["yester", "esterd", "sterda", "terday", "erday "]
    # 断言生成的后五个字符级 word boundaries n-gram 与预期结果相同
    assert cnga(text)[-5:] == expected

    # 创建新的 CountVectorizer 对象，从文件输入，使用字符级 word boundaries 分析器，生成长度为 3 到 6 的 n-gram
    cnga = CountVectorizer(
        input="file", analyzer="char_wb", ngram_range=(3, 6)
    ).build_analyzer()
    # 使用 StringIO 创建一个文件类似对象的文本流
    text = StringIO("A test with a file-like object!")
    # 预期的前六个字符级 word boundaries n-gram
    expected = [" a ", " te", "tes", "est", "st ", " tes"]
    # 断言生成的前六个字符级 word boundaries n-gram 与预期结果相同
    assert cnga(text)[:6] == expected


# 定义一个测试函数，用于测试单词级 n-gram 分析器
def test_word_ngram_analyzer():
    # 创建 CountVectorizer 对象，使用单词级分析器，处理时保留 Unicode 编码的重音符号，生成长度为 3 到 6 的 n-gram
    cnga = CountVectorizer(
        analyzer="word", strip_accents="unicode", ngram_range=(3, 6)
    ).build_analyzer()

    # 定义测试文本
    text = "This \n\tis a test, really.\n\n I met Harry yesterday"
    # 预期的前三个单词级 n-gram
    expected = ["this is test", "is test really", "test really met"]
    # 断言生成的前三个单词级 n-gram 与预期结果相同
    assert cnga(text)[:3] == expected

    # 预期的后三个单词级 n-gram
    expected = [
        "test really met harry yesterday",
        "this is test really met harry",
        "is test really met harry yesterday",
    ]
    # 断言生成的后三个单词级 n-gram 与预期结果相同
    assert cnga(text)[-3:] == expected

    # 创建新的 CountVectorizer 对象，从文件输入，使用单词级分析器，生成长度为 3 到 6 的 n-gram
    cnga_file = CountVectorizer(
        input="file", analyzer="word", ngram_range=(3, 6)
    ).build_analyzer()
    # 使用 StringIO 创建一个文件类似对象的文本流
    file = StringIO(text)
    # 断言文件输入生成的单词级 n-gram 与文本输入生成的单词级 n-gram 相同
    assert cnga_file(file) == cnga(text)


# 定义一个测试函数，用于测试 CountVectorizer 自定义词汇表功能
def test_countvectorizer_custom_vocabulary():
    # 定义一个词汇表
    vocab = {"pizza": 0, "beer": 1}
    # 将词汇表中的词汇存入集合中
    terms = set(vocab.keys())

    # 尝试几种支持的类型。
    # 遍历包含四种类型（dict, list, iter, partial(defaultdict, int)）的迭代器typ
    for typ in [dict, list, iter, partial(defaultdict, int)]:
        # 使用当前类型typ和给定的词汇表vocab创建对象v
        v = typ(vocab)
        # 使用CountVectorizer初始化向量化器vect，并指定词汇表为v
        vect = CountVectorizer(vocabulary=v)
        # 对JUNK_FOOD_DOCS进行拟合，构建词频矩阵
        vect.fit(JUNK_FOOD_DOCS)
        # 如果v是映射类型（Mapping），则验证vect的词汇表与vocab相同
        if isinstance(v, Mapping):
            assert vect.vocabulary_ == vocab
        else:
            # 否则验证vect的词汇表是terms的集合
            assert set(vect.vocabulary_) == terms
        # 使用向量化器vect对JUNK_FOOD_DOCS进行转换，得到特征矩阵X
        X = vect.transform(JUNK_FOOD_DOCS)
        # 验证特征矩阵X的列数与terms的长度相同
        assert X.shape[1] == len(terms)
        # 使用当前类型typ和给定的词汇表vocab创建对象v
        v = typ(vocab)
        # 使用CountVectorizer初始化向量化器vect，并指定词汇表为v
        vect = CountVectorizer(vocabulary=v)
        # 使用向量化器vect对特征矩阵X进行逆转换，得到逆向转换结果inv
        inv = vect.inverse_transform(X)
        # 验证逆向转换结果inv的长度与特征矩阵X的行数相同
        assert len(inv) == X.shape[0]
def test_countvectorizer_custom_vocabulary_pipeline():
    what_we_like = ["pizza", "beer"]  # 定义自定义词汇表
    pipe = Pipeline(
        [
            ("count", CountVectorizer(vocabulary=what_we_like)),  # 使用自定义词汇表创建 CountVectorizer 实例
            ("tfidf", TfidfTransformer()),  # 添加 TF-IDF 转换器到 Pipeline
        ]
    )
    X = pipe.fit_transform(ALL_FOOD_DOCS)  # 在文档集合上拟合 Pipeline，并转换成稀疏矩阵 X
    assert set(pipe.named_steps["count"].vocabulary_) == set(what_we_like)  # 断言 CountVectorizer 的词汇表与自定义词汇表一致
    assert X.shape[1] == len(what_we_like)  # 断言稀疏矩阵 X 的列数等于自定义词汇表的长度


def test_countvectorizer_custom_vocabulary_repeated_indices():
    vocab = {"pizza": 0, "beer": 0}  # 重复索引的自定义词汇表
    msg = "Vocabulary contains repeated indices"  # 报错消息
    with pytest.raises(ValueError, match=msg):  # 断言会抛出 ValueError，并匹配指定的报错消息
        vect = CountVectorizer(vocabulary=vocab)  # 使用自定义词汇表创建 CountVectorizer 实例
        vect.fit(["pasta_siziliana"])  # 在一个文档上拟合 CountVectorizer


def test_countvectorizer_custom_vocabulary_gap_index():
    vocab = {"pizza": 1, "beer": 2}  # 存在索引间隙的自定义词汇表
    with pytest.raises(ValueError, match="doesn't contain index"):  # 断言会抛出 ValueError，并匹配指定的报错消息
        vect = CountVectorizer(vocabulary=vocab)  # 使用自定义词汇表创建 CountVectorizer 实例
        vect.fit(["pasta_verdura"])  # 在一个文档上拟合 CountVectorizer


def test_countvectorizer_stop_words():
    cv = CountVectorizer()  # 创建 CountVectorizer 实例
    cv.set_params(stop_words="english")  # 设置停用词为英语常见停用词
    assert cv.get_stop_words() == ENGLISH_STOP_WORDS  # 断言获取的停用词集合与预定义的英语常见停用词集合相等
    cv.set_params(stop_words="_bad_str_stop_")  # 设置无效的停用词字符串
    with pytest.raises(ValueError):  # 断言会抛出 ValueError
        cv.get_stop_words()  # 尝试获取停用词集合
    cv.set_params(stop_words="_bad_unicode_stop_")  # 设置无效的停用词 Unicode 字符串
    with pytest.raises(ValueError):  # 断言会抛出 ValueError
        cv.get_stop_words()  # 尝试获取停用词集合
    stoplist = ["some", "other", "words"]  # 自定义的停用词列表
    cv.set_params(stop_words=stoplist)  # 设置停用词为自定义列表
    assert cv.get_stop_words() == set(stoplist)  # 断言获取的停用词集合与自定义列表转换成的集合相等


def test_countvectorizer_empty_vocabulary():
    with pytest.raises(ValueError, match="empty vocabulary"):  # 断言会抛出 ValueError，并匹配指定的报错消息
        vect = CountVectorizer(vocabulary=[])  # 使用空的自定义词汇表创建 CountVectorizer 实例
        vect.fit(["foo"])  # 在一个文档上拟合 CountVectorizer

    with pytest.raises(ValueError, match="empty vocabulary"):  # 断言会抛出 ValueError，并匹配指定的报错消息
        v = CountVectorizer(max_df=1.0, stop_words="english")  # 使用参数设置创建 CountVectorizer 实例
        v.fit(["to be or not to be", "and me too", "and so do you"])  # 在多个文档上拟合 CountVectorizer


def test_fit_countvectorizer_twice():
    cv = CountVectorizer()  # 创建 CountVectorizer 实例
    X1 = cv.fit_transform(ALL_FOOD_DOCS[:5])  # 在部分文档集合上拟合 CountVectorizer，并转换成稀疏矩阵 X1
    X2 = cv.fit_transform(ALL_FOOD_DOCS[5:])  # 在另一部分文档集合上拟合 CountVectorizer，并转换成稀疏矩阵 X2
    assert X1.shape[1] != X2.shape[1]  # 断言两次拟合后得到的稀疏矩阵列数不相等


def test_countvectorizer_custom_token_pattern():
    """Check `get_feature_names_out()` when a custom token pattern is passed.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/12971
    """
    corpus = [
        "This is the 1st document in my corpus.",
        "This document is the 2nd sample.",
        "And this is the 3rd one.",
        "Is this the 4th document?",
    ]  # 文档语料库
    token_pattern = r"[0-9]{1,3}(?:st|nd|rd|th)\s\b(\w{2,})\b"  # 自定义的 token 匹配模式
    vectorizer = CountVectorizer(token_pattern=token_pattern)  # 使用自定义 token 模式创建 CountVectorizer 实例
    vectorizer.fit_transform(corpus)  # 在语料库上拟合 CountVectorizer
    expected = ["document", "one", "sample"]  # 预期的特征名列表
    feature_names_out = vectorizer.get_feature_names_out()  # 获取转换后的特征名列表
    assert_array_equal(feature_names_out, expected)  # 断言转换后的特征名列表与预期列表相等


def test_countvectorizer_custom_token_pattern_with_several_group():
    """Check that we raise an error if token pattern capture several groups.
    Non-regression test for:
    ```
    Check that we raise an error if token pattern capture several groups.
    Non-regression test for:
    ```
    # 定义一个示例语料库，包含四个字符串作为文档
    corpus = [
        "This is the 1st document in my corpus.",
        "This document is the 2nd sample.",
        "And this is the 3rd one.",
        "Is this the 4th document?",
    ]

    # 定义一个正则表达式模式，用于匹配带有序数词的单词
    token_pattern = r"([0-9]{1,3}(?:st|nd|rd|th))\s\b(\w{2,})\b"

    # 定义错误消息字符串，用于匹配时捕获组多于一个时抛出的异常信息
    err_msg = "More than 1 capturing group in token pattern"

    # 使用CountVectorizer创建一个文本向量化的对象，指定了token_pattern作为词条化的规则
    vectorizer = CountVectorizer(token_pattern=token_pattern)

    # 使用pytest测试框架的pytest.raises来确保在fit操作中捕获到指定的ValueError异常，并匹配给定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        # 对语料库进行拟合操作，如果匹配到超过一个捕获组的情况，将抛出预期的异常
        vectorizer.fit(corpus)
def test_countvectorizer_uppercase_in_vocab():
    # 检查提供的词汇表中是否包含大写字母的检查仅在fit时进行，而不是在transform时进行 (#21251)
    vocabulary = ["Sample", "Upper", "Case", "Vocabulary"]
    # 警告消息内容
    message = (
        "Upper case characters found in"
        " vocabulary while 'lowercase'"
        " is True. These entries will not"
        " be matched with any documents"
    )

    # 创建一个CountVectorizer对象，设置lowercase为True，并使用给定的词汇表
    vectorizer = CountVectorizer(lowercase=True, vocabulary=vocabulary)

    # 在使用给定词汇表进行fit时，期望产生UserWarning，并匹配指定的警告消息
    with pytest.warns(UserWarning, match=message):
        vectorizer.fit(vocabulary)

    # 在transform时，设定一个警告过滤器，检查是否有UserWarning产生
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        vectorizer.transform(vocabulary)


def test_tf_transformer_feature_names_out():
    """检查TfidfTransformer的get_feature_names_out方法是否正常工作"""
    X = [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    # 创建一个TfidfTransformer对象，设置smooth_idf为True，norm为l2，并对输入X进行fit
    tr = TfidfTransformer(smooth_idf=True, norm="l2").fit(X)

    feature_names_in = ["a", "c", "b"]
    # 调用get_feature_names_out方法，检查输出的特征名是否与输入一致
    feature_names_out = tr.get_feature_names_out(feature_names_in)
    assert_array_equal(feature_names_in, feature_names_out)


def test_tf_idf_smoothing():
    X = [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    # 创建一个TfidfTransformer对象，设置smooth_idf为True，norm为l2，并对输入X进行fit_transform
    tr = TfidfTransformer(smooth_idf=True, norm="l2")
    tfidf = tr.fit_transform(X).toarray()
    assert (tfidf >= 0).all()

    # 检查归一化
    assert_array_almost_equal((tfidf**2).sum(axis=1), [1.0, 1.0, 1.0])

    # 对于只包含零的特征，这是稳健的
    X = [[1, 1, 0], [1, 1, 0], [1, 0, 0]]
    tr = TfidfTransformer(smooth_idf=True, norm="l2")
    tfidf = tr.fit_transform(X).toarray()
    assert (tfidf >= 0).all()


@pytest.mark.xfail(
    _IS_WASM,
    reason=(
        "no floating point exceptions, see"
        " https://github.com/numpy/numpy/pull/21895#issuecomment-1311525881"
    ),
)
def test_tfidf_no_smoothing():
    X = [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    # 创建一个TfidfTransformer对象，设置smooth_idf为False，norm为l2，并对输入X进行fit_transform
    tr = TfidfTransformer(smooth_idf=False, norm="l2")
    tfidf = tr.fit_transform(X).toarray()
    assert (tfidf >= 0).all()

    # 检查归一化
    assert_array_almost_equal((tfidf**2).sum(axis=1), [1.0, 1.0, 1.0])

    # 在没有平滑的情况下，对于只包含零的特征，IDF的稳健性受到影响
    X = [[1, 1, 0], [1, 1, 0], [1, 0, 0]]
    tr = TfidfTransformer(smooth_idf=False, norm="l2")

    in_warning_message = "divide by zero"
    # 在此处期望引发RuntimeWarning，并匹配指定的警告消息
    with pytest.warns(RuntimeWarning, match=in_warning_message):
        tr.fit_transform(X).toarray()


def test_sublinear_tf():
    X = [[1], [2], [3]]
    # 创建一个TfidfTransformer对象，设置sublinear_tf为True，use_idf为False，norm为None，并对输入X进行fit_transform
    tr = TfidfTransformer(sublinear_tf=True, use_idf=False, norm=None)
    tfidf = tr.fit_transform(X).toarray()
    assert tfidf[0] == 1
    assert tfidf[1] > tfidf[0]
    assert tfidf[2] > tfidf[1]
    assert tfidf[1] < 2
    assert tfidf[2] < 3


def test_vectorizer():
    # 将原始文档作为迭代器传入训练数据
    train_data = iter(ALL_FOOD_DOCS[:-1])
    test_data = [ALL_FOOD_DOCS[-1]]
    n_train = len(ALL_FOOD_DOCS) - 1

    # 测试没有提供词汇表的情况
    v1 = CountVectorizer(max_df=0.5)
    # 使用 v1 对训练数据进行向量化转换
    counts_train = v1.fit_transform(train_data)

    # 如果 counts_train 拥有 tocsr 方法，则将其转换为 CSR 格式的稀疏矩阵
    if hasattr(counts_train, "tocsr"):
        counts_train = counts_train.tocsr()

    # 断言 counts_train 的第一行中包含两次单词 "pizza"
    assert counts_train[0, v1.vocabulary_["pizza"]] == 2

    # 使用与 v1 相同的词汇表构建一个新的向量化器 v2
    v2 = CountVectorizer(vocabulary=v1.vocabulary_)

    # 比较两个向量化器在测试样本上的输出是否相同
    for v in (v1, v2):
        # 使用当前向量化器 v 转换测试数据，得到 counts_test
        counts_test = v.transform(test_data)

        # 如果 counts_test 拥有 tocsr 方法，则将其转换为 CSR 格式的稀疏矩阵
        if hasattr(counts_test, "tocsr"):
            counts_test = counts_test.tocsr()

        # 获取当前向量化器 v 的词汇表
        vocabulary = v.vocabulary_

        # 断言 counts_test 的第一行中单词 "salad" 出现一次
        assert counts_test[0, vocabulary["salad"]] == 1
        # 断言 counts_test 的第一行中单词 "tomato" 出现一次
        assert counts_test[0, vocabulary["tomato"]] == 1
        # 断言 counts_test 的第一行中单词 "water" 出现一次
        assert counts_test[0, vocabulary["water"]] == 1

        # 断言词汇表中不存在单词 "the"
        assert "the" not in vocabulary

        # 断言词汇表中不存在单词 "copyright"
        # 这些单词通常是高频出现在整个语料库中，不具备信息量
        assert "copyright" not in vocabulary

        # 断言 counts_test 的第一行中单词 "coke" 未出现
        assert counts_test[0, vocabulary["coke"]] == 0
        # 断言 counts_test 的第一行中单词 "burger" 未出现
        assert counts_test[0, vocabulary["burger"]] == 0
        # 断言 counts_test 的第一行中单词 "beer" 未出现
        assert counts_test[0, vocabulary["beer"]] == 0
        # 断言 counts_test 的第一行中单词 "pizza" 未出现
        assert counts_test[0, vocabulary["pizza"]] == 0

    # 测试 TF-IDF 转换
    t1 = TfidfTransformer(norm="l1")
    tfidf = t1.fit(counts_train).transform(counts_train).toarray()

    # 断言 IDF 向量的长度与 v1 的词汇表长度相同
    assert len(t1.idf_) == len(v1.vocabulary_)
    # 断言 tfidf 的形状为 (n_train, len(v1.vocabulary_))
    assert tfidf.shape == (n_train, len(v1.vocabulary_))

    # 使用新数据测试 TF-IDF 转换
    tfidf_test = t1.transform(counts_test).toarray()
    # 断言 tfidf_test 的形状为 (len(test_data), len(v1.vocabulary_))
    assert tfidf_test.shape == (len(test_data), len(v1.vocabulary_))

    # 测试仅使用 TF 转换
    t2 = TfidfTransformer(norm="l1", use_idf=False)
    tf = t2.fit(counts_train).transform(counts_train).toarray()

    # 断言 t2 没有 idf_ 属性
    assert not hasattr(t2, "idf_")

    # 测试带有未学习 idf 向量的 IDF 转换
    t3 = TfidfTransformer(use_idf=True)
    with pytest.raises(ValueError):
        t3.transform(counts_train)

    # 断言 L1 归一化后的 TF 向量总和为 1
    assert_array_almost_equal(np.sum(tf, axis=1), [1.0] * n_train)

    # 测试直接使用 TF-IDF 向量化器
    train_data = iter(ALL_FOOD_DOCS[:-1])
    tv = TfidfVectorizer(norm="l1")

    # 将 tv 的 max_df 设置为 v1 的 max_df
    tv.max_df = v1.max_df
    tfidf2 = tv.fit_transform(train_data).toarray()

    # 断言 tv 不使用固定的词汇表
    assert not tv.fixed_vocabulary_
    # 断言 tfidf 与 tfidf2 几乎相等
    assert_array_almost_equal(tfidf, tfidf2)

    # 使用新数据测试直接 TF-IDF 向量化器
    tfidf_test2 = tv.transform(test_data).toarray()
    # 断言 tfidf_test 与 tfidf_test2 几乎相等
    assert_array_almost_equal(tfidf_test, tfidf_test2)

    # 测试对未设置词汇表的未训练向量化器进行转换
    v3 = CountVectorizer(vocabulary=None)
    with pytest.raises(ValueError):
        v3.transform(train_data)

    # 设置 v3 的参数：去除重音符号，不转换为小写
    v3.set_params(strip_accents="ascii", lowercase=False)
    # 创建一个名为 processor 的对象，调用 v3.build_preprocessor() 方法返回的预处理器
    processor = v3.build_preprocessor()

    # 定义一个包含特定文本的字符串变量
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."

    # 使用 strip_accents_ascii 函数处理文本，返回预期的处理结果
    expected = strip_accents_ascii(text)

    # 将 text 传递给 processor 对象，得到处理后的结果
    result = processor(text)

    # 使用断言检查处理后的结果是否与预期相同
    assert expected == result

    # 当 strip_accents 参数设置为无效值 "_gabbledegook_" 时，设置 v3 的参数并期望抛出 ValueError 异常
    v3.set_params(strip_accents="_gabbledegook_", preprocessor=None)
    with pytest.raises(ValueError):
        v3.build_preprocessor()

    # 当设置 v3 的参数为无效的分析器类型 "_invalid_analyzer_type_" 时，预期抛出 ValueError 异常
    v3.set_params = "_invalid_analyzer_type_"
    with pytest.raises(ValueError):
        v3.build_analyzer()
def test_tfidf_vectorizer_setters():
    # 设置初始参数
    norm, use_idf, smooth_idf, sublinear_tf = "l2", False, False, False
    # 创建 TfidfVectorizer 实例，使用给定的参数进行初始化
    tv = TfidfVectorizer(
        norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
    )
    # 对文档集合 JUNK_FOOD_DOCS 进行拟合
    tv.fit(JUNK_FOOD_DOCS)
    # 断言初始参数是否正确传递到内部的 _tfidf 对象
    assert tv._tfidf.norm == norm
    assert tv._tfidf.use_idf == use_idf
    assert tv._tfidf.smooth_idf == smooth_idf
    assert tv._tfidf.sublinear_tf == sublinear_tf

    # 修改 TfidfVectorizer 的参数，预期不会影响内部 _tfidf 对象，直到重新拟合
    tv.norm = "l1"
    tv.use_idf = True
    tv.smooth_idf = True
    tv.sublinear_tf = True
    assert tv._tfidf.norm == norm
    assert tv._tfidf.use_idf == use_idf
    assert tv._tfidf.smooth_idf == smooth_idf
    assert tv._tfidf.sublinear_tf == sublinear_tf

    # 再次对文档集合拟合，检查参数是否更新到内部 _tfidf 对象
    tv.fit(JUNK_FOOD_DOCS)
    assert tv._tfidf.norm == tv.norm
    assert tv._tfidf.use_idf == tv.use_idf
    assert tv._tfidf.smooth_idf == tv.smooth_idf
    assert tv._tfidf.sublinear_tf == tv.sublinear_tf


def test_hashing_vectorizer():
    # 创建 HashingVectorizer 实例
    v = HashingVectorizer()
    # 对所有文档集 ALL_FOOD_DOCS 进行转换
    X = v.transform(ALL_FOOD_DOCS)
    # 检查转换后的矩阵形状和数据类型是否正确
    token_nnz = X.nnz
    assert X.shape == (len(ALL_FOOD_DOCS), v.n_features)
    assert X.dtype == v.dtype

    # 默认情况下，哈希后的值会获得随机符号并进行 l2 归一化
    assert np.min(X.data) > -1
    assert np.min(X.data) < 0
    assert np.max(X.data) > 0
    assert np.max(X.data) < 1

    # 检查每行数据是否被归一化
    for i in range(X.shape[0]):
        assert_almost_equal(np.linalg.norm(X[0].data, 2), 1.0)

    # 使用一些非默认参数重新创建 HashingVectorizer 实例
    v = HashingVectorizer(ngram_range=(1, 2), norm="l1")
    X = v.transform(ALL_FOOD_DOCS)
    # 检查转换后的矩阵形状和数据类型是否正确
    assert X.shape == (len(ALL_FOOD_DOCS), v.n_features)
    assert X.dtype == v.dtype

    # ngrams 生成更多的非零值
    ngrams_nnz = X.nnz
    assert ngrams_nnz > token_nnz
    assert ngrams_nnz < 2 * token_nnz

    # 确保特征值被归一化
    assert np.min(X.data) > -1
    assert np.max(X.data) < 1

    # 检查每行数据是否被归一化
    for i in range(X.shape[0]):
        assert_almost_equal(np.linalg.norm(X[0].data, 1), 1.0)


def test_feature_names():
    # 创建 CountVectorizer 实例，设置 max_df 参数
    cv = CountVectorizer(max_df=0.5)

    # 测试未拟合或空词汇表时是否会引发 ValueError
    with pytest.raises(ValueError):
        cv.get_feature_names_out()
    assert not cv.fixed_vocabulary_

    # 对数据集 ALL_FOOD_DOCS 进行拟合并转换
    X = cv.fit_transform(ALL_FOOD_DOCS)
    n_samples, n_features = X.shape
    # 断言词汇表的长度与特征数相同
    assert len(cv.vocabulary_) == n_features

    # 获取特征名称数组，并进行相关断言
    feature_names = cv.get_feature_names_out()
    assert isinstance(feature_names, np.ndarray)
    assert feature_names.dtype == object
    assert len(feature_names) == n_features
    # 断言：检查特征名列表是否与预期相等
    assert_array_equal(
        [
            "beer",
            "burger",
            "celeri",
            "coke",
            "pizza",
            "salad",
            "sparkling",
            "tomato",
            "water",
        ],
        feature_names,  # 实际特征名列表
    )

    # 遍历特征名列表，确保每个特征名的索引与 CountVectorizer 的词汇表中对应的索引相等
    for idx, name in enumerate(feature_names):
        assert idx == cv.vocabulary_.get(name)

    # 测试自定义词汇表
    vocab = [
        "beer",
        "burger",
        "celeri",
        "coke",
        "pizza",
        "salad",
        "sparkling",
        "tomato",
        "water",
    ]

    # 使用自定义词汇表创建 CountVectorizer 对象
    cv = CountVectorizer(vocabulary=vocab)
    # 获取 CountVectorizer 对象生成的特征名列表
    feature_names = cv.get_feature_names_out()
    # 断言：检查特征名列表是否与预期相等
    assert_array_equal(
        [
            "beer",
            "burger",
            "celeri",
            "coke",
            "pizza",
            "salad",
            "sparkling",
            "tomato",
            "water",
        ],
        feature_names,  # 实际特征名列表
    )
    # 断言：检查 CountVectorizer 是否使用了固定的词汇表
    assert cv.fixed_vocabulary_

    # 遍历特征名列表，确保每个特征名的索引与 CountVectorizer 的词汇表中对应的索引相等
    for idx, name in enumerate(feature_names):
        assert idx == cv.vocabulary_.get(name)
@pytest.mark.parametrize("Vectorizer", (CountVectorizer, TfidfVectorizer))
def test_vectorizer_max_features(Vectorizer):
    expected_vocabulary = {"burger", "beer", "salad", "pizza"}

    # test bounded number of extracted features
    # 创建一个指定类型的文本向量化器（CountVectorizer 或 TfidfVectorizer），限制提取的最大特征数为4
    vectorizer = Vectorizer(max_df=0.6, max_features=4)
    # 使用指定的文本数据集 ALL_FOOD_DOCS 对向量化器进行拟合
    vectorizer.fit(ALL_FOOD_DOCS)
    # 断言向量化后的词汇表与预期的词汇表一致
    assert set(vectorizer.vocabulary_) == expected_vocabulary


def test_count_vectorizer_max_features():
    # Regression test: max_features didn't work correctly in 0.14.

    # 创建三个不同配置的 CountVectorizer 实例，分别限制最大特征数为1、3 和不限制
    cv_1 = CountVectorizer(max_features=1)
    cv_3 = CountVectorizer(max_features=3)
    cv_None = CountVectorizer(max_features=None)

    # 对 JUNK_FOOD_DOCS 进行拟合，并计算每个实例的特征出现次数的总和
    counts_1 = cv_1.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)
    counts_3 = cv_3.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)
    counts_None = cv_None.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)

    # 获取每个实例的特征名称列表
    features_1 = cv_1.get_feature_names_out()
    features_3 = cv_3.get_feature_names_out()
    features_None = cv_None.get_feature_names_out()

    # 断言每个实例中最常见的特征出现次数为7（测试数据中最频繁的特征为 "the"，出现7次）
    assert 7 == counts_1.max()
    assert 7 == counts_3.max()
    assert 7 == counts_None.max()

    # 断言每个实例中最常见的特征应该都是 "the"
    assert "the" == features_1[np.argmax(counts_1)]
    assert "the" == features_3[np.argmax(counts_3)]
    assert "the" == features_None[np.argmax(counts_None)]


def test_vectorizer_max_df():
    test_data = ["abc", "dea", "eat"]
    # 创建一个字符级的 CountVectorizer 实例，并设置最大文档频率为1.0
    vect = CountVectorizer(analyzer="char", max_df=1.0)
    vect.fit(test_data)
    # 断言 'a' 存在于词汇表的键中
    assert "a" in vect.vocabulary_.keys()
    # 断言词汇表中的键的数量为6
    assert len(vect.vocabulary_.keys()) == 6

    # 修改 max_df 为 0.5，重新拟合向量化器
    vect.max_df = 0.5  # 0.5 * 3 documents -> max_doc_count == 1.5
    vect.fit(test_data)
    # 断言 'a' 不在词汇表的键中（因为 'a' 的文档频率超过了新的 max_df）
    assert "a" not in vect.vocabulary_.keys()  # {ae} ignored
    # 断言词汇表中的键的数量为4
    assert len(vect.vocabulary_.keys()) == 4  # {bcdt} remain

    # 将 max_df 重置为 1，重新拟合向量化器
    vect.max_df = 1
    vect.fit(test_data)
    # 断言 'a' 不在词汇表的键中（因为 'a' 的文档频率超过了新的 max_df）
    assert "a" not in vect.vocabulary_.keys()  # {ae} ignored
    # 断言词汇表中的键的数量为4
    assert len(vect.vocabulary_.keys()) == 4  # {bcdt} remain


def test_vectorizer_min_df():
    test_data = ["abc", "dea", "eat"]
    # 创建一个字符级的 CountVectorizer 实例，并设置最小文档频率为1
    vect = CountVectorizer(analyzer="char", min_df=1)
    vect.fit(test_data)
    # 断言 'a' 存在于词汇表的键中
    assert "a" in vect.vocabulary_.keys()
    # 断言词汇表中的键的数量为6
    assert len(vect.vocabulary_.keys()) == 6

    # 修改 min_df 为 2，重新拟合向量化器
    vect.min_df = 2
    vect.fit(test_data)
    # 断言 'c' 不在词汇表的键中（因为 'c' 的文档频率不满足新的 min_df）
    assert "c" not in vect.vocabulary_.keys()  # {bcdt} ignored
    # 断言词汇表中的键的数量为2
    assert len(vect.vocabulary_.keys()) == 2  # {ae} remain

    # 将 min_df 重置为 0.8，重新拟合向量化器
    vect.min_df = 0.8  # 0.8 * 3 documents -> min_doc_count == 2.4
    vect.fit(test_data)
    # 断言 'c' 不在词汇表的键中（因为 'c' 的文档频率不满足新的 min_df）
    assert "c" not in vect.vocabulary_.keys()  # {bcdet} ignored
    # 断言词汇表中的键的数量为1
    assert len(vect.vocabulary_.keys()) == 1  # {a} remains


def test_count_binary_occurrences():
    # 默认情况下多次出现的特征将作为长整型计数
    test_data = ["aaabc", "abbde"]
    # 创建一个字符级的 CountVectorizer 实例，并设置最大文档频率为1.0
    vect = CountVectorizer(analyzer="char", max_df=1.0)
    X = vect.fit_transform(test_data).toarray()
    # 断言特征名称数组与预期的特征名称数组相等
    assert_array_equal(["a", "b", "c", "d", "e"], vect.get_feature_names_out())
    # 断言转换后的数组 X 与预期的数组相等
    assert_array_equal([[3, 1, 1, 0, 0], [1, 2, 0, 1, 1]], X)
    # 使用布尔特征，可以获取二进制出现信息而非计数信息。
    # 创建一个字符级别的计数向量化器，设置最大文档频率为1.0，使用二进制表示特征。
    vect = CountVectorizer(analyzer="char", max_df=1.0, binary=True)
    # 对测试数据进行向量化处理，并转换为数组表示
    X = vect.fit_transform(test_data).toarray()
    # 断言检查向量化结果是否符合预期
    assert_array_equal([[1, 1, 1, 0, 0], [1, 1, 0, 1, 1]], X)

    # 检查是否能够改变数据类型(dtype)
    # 创建一个字符级别的计数向量化器，设置最大文档频率为1.0，使用二进制表示特征，并指定数据类型为np.float32
    vect = CountVectorizer(analyzer="char", max_df=1.0, binary=True, dtype=np.float32)
    # 对测试数据进行向量化处理
    X_sparse = vect.fit_transform(test_data)
    # 断言检查向量化结果的数据类型是否为np.float32
    assert X_sparse.dtype == np.float32
# 定义一个测试函数，用于测试哈希向量化器的二进制出现次数功能
def test_hashed_binary_occurrences():
    # 默认情况下，多次出现的字符被计为长整型
    test_data = ["aaabc", "abbde"]
    
    # 创建一个字符级哈希向量化器对象，不使用替代符号，不进行规范化
    vect = HashingVectorizer(alternate_sign=False, analyzer="char", norm=None)
    
    # 对测试数据进行向量化转换
    X = vect.transform(test_data)
    
    # 断言：第一条数据的最大值应为3
    assert np.max(X[0:1].data) == 3
    # 断言：第二条数据的最大值应为2
    assert np.max(X[1:2].data) == 2
    # 断言：转换后的数据类型应为 np.float64
    assert X.dtype == np.float64

    # 使用布尔特性，可以获取二进制出现信息
    vect = HashingVectorizer(
        analyzer="char", alternate_sign=False, binary=True, norm=None
    )
    X = vect.transform(test_data)
    
    # 断言：所有数据的最大值应为1
    assert np.max(X.data) == 1
    # 断言：转换后的数据类型应为 np.float64
    assert X.dtype == np.float64

    # 检查更改数据类型的能力
    vect = HashingVectorizer(
        analyzer="char", alternate_sign=False, binary=True, norm=None, dtype=np.float64
    )
    X = vect.transform(test_data)
    
    # 断言：转换后的数据类型应为 np.float64
    assert X.dtype == np.float64


# 使用参数化测试，测试向量化器的逆转换功能
@pytest.mark.parametrize("Vectorizer", (CountVectorizer, TfidfVectorizer))
def test_vectorizer_inverse_transform(Vectorizer):
    # 原始文档数据
    data = ALL_FOOD_DOCS
    
    # 创建特定的向量化器对象
    vectorizer = Vectorizer()
    
    # 对原始文档数据进行拟合和转换
    transformed_data = vectorizer.fit_transform(data)
    
    # 对转换后的数据进行逆转换
    inversed_data = vectorizer.inverse_transform(transformed_data)
    
    # 断言：逆转换后的结果应为列表类型
    assert isinstance(inversed_data, list)

    # 获取文本分析器对象
    analyze = vectorizer.build_analyzer()
    
    # 逐一检查每个文档的词条和逆转换后的词条是否一致
    for doc, inversed_terms in zip(data, inversed_data):
        terms = np.sort(np.unique(analyze(doc)))
        inversed_terms = np.sort(np.unique(inversed_terms))
        assert_array_equal(terms, inversed_terms)
    
    # 断言：转换后的数据应为稀疏矩阵
    assert sparse.issparse(transformed_data)
    # 断言：转换后的稀疏矩阵格式应为 "csr"
    assert transformed_data.format == "csr"

    # 测试逆转换在 numpy 数组和 scipy 中的工作情况
    transformed_data2 = transformed_data.toarray()
    inversed_data2 = vectorizer.inverse_transform(transformed_data2)
    
    # 逐一检查逆转换结果是否一致
    for terms, terms2 in zip(inversed_data, inversed_data2):
        assert_array_equal(np.sort(terms), np.sort(terms2))

    # 检查逆转换在非 CSR 稀疏数据上的工作情况
    transformed_data3 = transformed_data.tocsc()
    inversed_data3 = vectorizer.inverse_transform(transformed_data3)
    
    # 逐一检查逆转换结果是否一致
    for terms, terms3 in zip(inversed_data, inversed_data3):
        assert_array_equal(np.sort(terms), np.sort(terms3))


# 测试计数向量化器管道和网格搜索选择最佳参数
def test_count_vectorizer_pipeline_grid_selection():
    # 原始文档数据
    data = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS

    # 标签：将垃圾食品标记为 -1，其他标记为 +1
    target = [-1] * len(JUNK_FOOD_DOCS) + [1] * len(NOTJUNK_FOOD_DOCS)

    # 将数据集分为训练集和测试集
    train_data, test_data, target_train, target_test = train_test_split(
        data, target, test_size=0.2, random_state=0
    )

    # 创建管道，包括计数向量化器和线性支持向量分类器
    pipeline = Pipeline([("vect", CountVectorizer()), ("svc", LinearSVC())])

    # 定义参数网格，用于搜索最佳参数
    parameters = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "svc__loss": ("hinge", "squared_hinge"),
    }

    # 查找特征提取和分类器的最佳参数
    # 创建一个网格搜索对象，用于参数优化
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, cv=3)

    # 检查网格搜索找到的最佳模型在保留的评估集上是否100%正确
    pred = grid_search.fit(train_data, target_train).predict(test_data)
    assert_array_equal(pred, target_test)

    # 在这个玩具数据集上，认为在网格搜索的最后阶段使用的二元组表示是最佳估计器，
    # 因为它们都收敛到100%准确率的模型
    assert grid_search.best_score_ == 1.0
    best_vectorizer = grid_search.best_estimator_.named_steps["vect"]
    assert best_vectorizer.ngram_range == (1, 1)
# 测试向量化和分类管道的网格搜索选择
def test_vectorizer_pipeline_grid_selection():
    # 原始文档
    data = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS

    # 将垃圾食品标记为-1，其他标记为+1
    target = [-1] * len(JUNK_FOOD_DOCS) + [1] * len(NOTJUNK_FOOD_DOCS)

    # 将数据集分割为模型开发和最终评估集
    train_data, test_data, target_train, target_test = train_test_split(
        data, target, test_size=0.1, random_state=0
    )

    # 创建管道，包括文本特征提取和分类器
    pipeline = Pipeline([("vect", TfidfVectorizer()), ("svc", LinearSVC())])

    # 定义参数网格
    parameters = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "vect__norm": ("l1", "l2"),
        "svc__loss": ("hinge", "squared_hinge"),
    }

    # 使用网格搜索寻找最佳参数
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1)

    # 检查网格搜索找到的最佳模型在保留的评估集上是否完全正确
    pred = grid_search.fit(train_data, target_train).predict(test_data)
    assert_array_equal(pred, target_test)

    # 在这个玩具数据集上，考虑到所有模型都收敛到100%准确率，因此使用最后一个
    # 网格搜索中的双字母组合表示器作为最佳估计器
    assert grid_search.best_score_ == 1.0
    best_vectorizer = grid_search.best_estimator_.named_steps["vect"]
    assert best_vectorizer.ngram_range == (1, 1)
    assert best_vectorizer.norm == "l2"
    assert not best_vectorizer.fixed_vocabulary_


# 测试向量化和分类管道的交叉验证
def test_vectorizer_pipeline_cross_validation():
    # 原始文档
    data = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS

    # 将垃圾食品标记为-1，其他标记为+1
    target = [-1] * len(JUNK_FOOD_DOCS) + [1] * len(NOTJUNK_FOOD_DOCS)

    # 创建管道，包括文本特征提取和分类器
    pipeline = Pipeline([("vect", TfidfVectorizer()), ("svc", LinearSVC())])

    # 进行交叉验证
    cv_scores = cross_val_score(pipeline, data, target, cv=3)
    assert_array_equal(cv_scores, [1.0, 1.0, 1.0])


# 测试向量化器在处理Unicode字符（如西里尔字母）时的表现
def test_vectorizer_unicode():
    # 测试文档，包含俄语的西里尔字母
    document = (
        "Машинное обучение — обширный подраздел искусственного "
        "интеллекта, изучающий методы построения алгоритмов, "
        "способных обучаться."
    )

    # 使用CountVectorizer进行向量化
    vect = CountVectorizer()
    X_counted = vect.fit_transform([document])
    assert X_counted.shape == (1, 12)

    # 使用HashingVectorizer进行向量化
    vect = HashingVectorizer(norm=None, alternate_sign=False)
    X_hashed = vect.transform([document])
    assert X_hashed.shape == (1, 2**20)

    # 在这个小数据集上没有碰撞
    assert X_counted.nnz == X_hashed.nnz

    # 当norm为None且不使用alternate_sign时，tokens会被计数直到碰撞
    assert_array_equal(np.sort(X_counted.data), np.sort(X_hashed.data))


# 测试带有固定词汇表的TF-IDF向量化器
def test_tfidf_vectorizer_with_fixed_vocabulary():
    # 非回归测试，用于检查继承问题
    vocabulary = ["pizza", "celeri"]
    vect = TfidfVectorizer(vocabulary=vocabulary)
    X_1 = vect.fit_transform(ALL_FOOD_DOCS)
    X_2 = vect.transform(ALL_FOOD_DOCS)
    # 断言两个稀疏矩阵的数组几乎相等
    assert_array_almost_equal(X_1.toarray(), X_2.toarray())
    # 断言向量化器具有固定的词汇表
    assert vect.fixed_vocabulary_
# 定义测试函数 test_pickling_vectorizer，用于测试文本向量化器的序列化和反序列化
def test_pickling_vectorizer():
    # 创建多个向量化器实例的列表
    instances = [
        HashingVectorizer(),  # 默认配置的 HashingVectorizer 实例
        HashingVectorizer(norm="l1"),  # 指定 norm="l1" 的 HashingVectorizer 实例
        HashingVectorizer(binary=True),  # 指定 binary=True 的 HashingVectorizer 实例
        HashingVectorizer(ngram_range=(1, 2)),  # 指定 ngram_range=(1, 2) 的 HashingVectorizer 实例
        CountVectorizer(),  # 默认配置的 CountVectorizer 实例
        CountVectorizer(preprocessor=strip_tags),  # 使用 strip_tags 预处理器的 CountVectorizer 实例
        CountVectorizer(analyzer=lazy_analyze),  # 使用 lazy_analyze 分析器的 CountVectorizer 实例
        CountVectorizer(preprocessor=strip_tags).fit(JUNK_FOOD_DOCS),  # 在 JUNK_FOOD_DOCS 上拟合的 CountVectorizer 实例
        CountVectorizer(strip_accents=strip_eacute).fit(JUNK_FOOD_DOCS),  # 在 JUNK_FOOD_DOCS 上拟合的 CountVectorizer 实例
        TfidfVectorizer(),  # 默认配置的 TfidfVectorizer 实例
        TfidfVectorizer(analyzer=lazy_analyze),  # 使用 lazy_analyze 分析器的 TfidfVectorizer 实例
        TfidfVectorizer().fit(JUNK_FOOD_DOCS),  # 在 JUNK_FOOD_DOCS 上拟合的 TfidfVectorizer 实例
    ]

    # 对每个向量化器实例进行测试
    for orig in instances:
        # 序列化原始实例
        s = pickle.dumps(orig)
        # 反序列化并获得复制的实例
        copy = pickle.loads(s)
        # 断言复制的实例与原始实例的类型相同
        assert type(copy) == orig.__class__
        # 断言复制的实例与原始实例具有相同的参数设置
        assert copy.get_params() == orig.get_params()
        # 断言复制的实例在相同数据上的稀疏和密集表示近似相等
        assert_allclose_dense_sparse(
            copy.fit_transform(JUNK_FOOD_DOCS),
            orig.fit_transform(JUNK_FOOD_DOCS),
        )


# 使用 pytest 的 parametrize 装饰器对多个工厂函数进行参数化测试
@pytest.mark.parametrize(
    "factory",
    [
        CountVectorizer.build_analyzer,  # CountVectorizer 的 build_analyzer 工厂函数
        CountVectorizer.build_preprocessor,  # CountVectorizer 的 build_preprocessor 工厂函数
        CountVectorizer.build_tokenizer,  # CountVectorizer 的 build_tokenizer 工厂函数
    ],
)
# 定义测试函数 test_pickling_built_processors，用于测试 CountVectorizer 的内建处理器的序列化和反序列化
def test_pickling_built_processors(factory):
    """Tokenizers cannot be pickled
    https://github.com/scikit-learn/scikit-learn/issues/12833
    """
    # 创建一个 CountVectorizer 实例
    vec = CountVectorizer()
    # 使用给定工厂函数构建一个处理器函数
    function = factory(vec)
    # 待处理的文本
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    # 对处理器函数进行序列化和反序列化，获取回程处理器函数
    roundtripped_function = pickle.loads(pickle.dumps(function))
    # 使用原始处理器函数处理文本
    expected = function(text)
    # 使用回程处理器函数处理文本
    result = roundtripped_function(text)
    # 断言处理结果与预期结果相等
    assert result == expected


# 定义测试函数 test_countvectorizer_vocab_sets_when_pickling，用于测试 CountVectorizer 在使用集合类型词汇时的序列化和反序列化
def test_countvectorizer_vocab_sets_when_pickling():
    # 确保在反序列化后保留词汇集合的迭代顺序，将集合类型的词汇转换为列表
    rng = np.random.RandomState(0)
    vocab_words = np.array(
        [
            "beer",
            "burger",
            "celeri",
            "coke",
            "pizza",
            "salad",
            "sparkling",
            "tomato",
            "water",
        ]
    )
    # 多次测试
    for x in range(0, 100):
        # 从 vocab_words 中随机选择 5 个单词组成的词汇集合
        vocab_set = set(rng.choice(vocab_words, size=5, replace=False))
        # 创建 CountVectorizer 实例，指定词汇为 vocab_set
        cv = CountVectorizer(vocabulary=vocab_set)
        # 对 CountVectorizer 实例进行序列化和反序列化
        unpickled_cv = pickle.loads(pickle.dumps(cv))
        # 在 ALL_FOOD_DOCS 上拟合原始 CountVectorizer 实例
        cv.fit(ALL_FOOD_DOCS)
        # 在 ALL_FOOD_DOCS 上拟合反序列化的 CountVectorizer 实例
        unpickled_cv.fit(ALL_FOOD_DOCS)
        # 断言两个 CountVectorizer 实例的特征名称数组相等
        assert_array_equal(
            cv.get_feature_names_out(), unpickled_cv.get_feature_names_out()
        )


# 定义测试函数 test_countvectorizer_vocab_dicts_when_pickling，用于测试 CountVectorizer 在使用字典类型词汇时的序列化和反序列化
def test_countvectorizer_vocab_dicts_when_pickling():
    rng = np.random.RandomState(0)
    vocab_words = np.array(
        [
            "beer",
            "burger",
            "celeri",
            "coke",
            "pizza",
            "salad",
            "sparkling",
            "tomato",
            "water",
        ]
    )
    # 循环迭代100次，每次创建一个新的空字典 vocab_dict
    for x in range(0, 100):
        # 初始化一个空的词汇字典 vocab_dict
        vocab_dict = dict()
        # 从 vocab_words 中随机选择5个单词，不放回
        words = rng.choice(vocab_words, size=5, replace=False)
        # 构建以选定单词为键，以其在列表中的索引为值的字典
        for y in range(0, 5):
            vocab_dict[words[y]] = y
        # 使用 vocab_dict 创建 CountVectorizer 对象 cv
        cv = CountVectorizer(vocabulary=vocab_dict)
        # 通过 pickle 序列化和反序列化 cv 对象，以确保其状态保存与恢复的一致性
        unpickled_cv = pickle.loads(pickle.dumps(cv))
        # 使用 cv 对象拟合 ALL_FOOD_DOCS 的文档并转换为词频向量
        cv.fit(ALL_FOOD_DOCS)
        # 使用反序列化的 unpickled_cv 对象拟合 ALL_FOOD_DOCS 的文档并转换为词频向量
        unpickled_cv.fit(ALL_FOOD_DOCS)
        # 断言两个 CountVectorizer 对象的输出特征名称是否相等
        assert_array_equal(
            cv.get_feature_names_out(), unpickled_cv.get_feature_names_out()
        )
# 测试将文档集合转换为词频向量并进行逆文档频率（IDF）转换
def test_pickling_transformer():
    # 使用 CountVectorizer 对文档集合 JUNK_FOOD_DOCS 进行词频向量化处理，并转换为稀疏矩阵 X
    X = CountVectorizer().fit_transform(JUNK_FOOD_DOCS)
    # 使用 TfidfTransformer 对 X 进行 IDF 转换
    orig = TfidfTransformer().fit(X)
    # 将 orig 对象序列化为字节流 s
    s = pickle.dumps(orig)
    # 从字节流 s 中反序列化出一个新的对象 copy
    copy = pickle.loads(s)
    # 断言 copy 是与 orig 同一类的对象
    assert type(copy) == orig.__class__
    # 断言经过 copy.fit_transform(X) 转换后的稀疏矩阵数组与 orig.fit_transform(X) 的结果相等
    assert_array_equal(copy.fit_transform(X).toarray(), orig.fit_transform(X).toarray())


# 测试 Transformer 对象的 IDF 设置
def test_transformer_idf_setter():
    # 使用 CountVectorizer 对文档集合 JUNK_FOOD_DOCS 进行词频向量化处理，并转换为稀疏矩阵 X
    X = CountVectorizer().fit_transform(JUNK_FOOD_DOCS)
    # 使用 TfidfTransformer 对 X 进行 IDF 转换
    orig = TfidfTransformer().fit(X)
    # 创建一个新的 TfidfTransformer 对象 copy
    copy = TfidfTransformer()
    # 将 copy 的 IDF 属性设置为 orig 的 IDF 属性值
    copy.idf_ = orig.idf_
    # 断言经过 copy.transform(X) 转换后的稀疏矩阵数组与 orig.transform(X) 的结果相等
    assert_array_equal(copy.transform(X).toarray(), orig.transform(X).toarray())


# 测试 TfidfVectorizer 对象的 IDF 设置
def test_tfidf_vectorizer_setter():
    # 创建一个使用 IDF 的 TfidfVectorizer 对象 orig
    orig = TfidfVectorizer(use_idf=True)
    # 对文档集合 JUNK_FOOD_DOCS 进行拟合
    orig.fit(JUNK_FOOD_DOCS)
    # 创建一个新的 TfidfVectorizer 对象 copy，使用 orig 的词汇表和 IDF 属性值
    copy = TfidfVectorizer(vocabulary=orig.vocabulary_, use_idf=True)
    copy.idf_ = orig.idf_
    # 断言经过 copy.transform(JUNK_FOOD_DOCS) 转换后的稀疏矩阵数组与 orig.transform(JUNK_FOOD_DOCS) 的结果相等
    assert_array_equal(
        copy.transform(JUNK_FOOD_DOCS).toarray(),
        orig.transform(JUNK_FOOD_DOCS).toarray(),
    )
    # 当 use_idf=False 时，不能设置 idf_ 属性
    copy = TfidfVectorizer(vocabulary=orig.vocabulary_, use_idf=False)
    err_msg = "`idf_` cannot be set when `user_idf=False`."
    # 使用 pytest 检查设置 idf_ 属性时的 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        copy.idf_ = orig.idf_


# 测试 TfidfVectorizer 对象的无效 IDF 属性设置
def test_tfidfvectorizer_invalid_idf_attr():
    # 创建一个使用 IDF 的 TfidfVectorizer 对象 vect
    vect = TfidfVectorizer(use_idf=True)
    # 对文档集合 JUNK_FOOD_DOCS 进行拟合
    vect.fit(JUNK_FOOD_DOCS)
    # 创建一个新的 TfidfVectorizer 对象 copy，使用 vect 的词汇表
    copy = TfidfVectorizer(vocabulary=vect.vocabulary_, use_idf=True)
    # 设置一个无效长度的 IDF 属性值
    expected_idf_len = len(vect.idf_)
    invalid_idf = [1.0] * (expected_idf_len + 1)
    # 使用 pytest 检查设置 idf_ 属性时的 ValueError 异常
    with pytest.raises(ValueError):
        setattr(copy, "idf_", invalid_idf)


# 测试 CountVectorizer 的词汇表不能有重复项
def test_non_unique_vocab():
    # 创建一个包含重复词汇的词汇表 vocab
    vocab = ["a", "b", "c", "a", "a"]
    # 使用 CountVectorizer 并使用 vocab 进行初始化
    vect = CountVectorizer(vocabulary=vocab)
    # 使用 pytest 检查 fit 方法中的 ValueError 异常
    with pytest.raises(ValueError):
        vect.fit([])


# 测试 HashingVectorizer 处理文档中的 NaN 值
def test_hashingvectorizer_nan_in_docs():
    # 当从 CSV 文件加载文本字段时，可能会出现 NaN 值
    message = "np.nan is an invalid document, expected byte or unicode string."
    exception = ValueError

    def func():
        # 创建一个 HashingVectorizer 对象 hv
        hv = HashingVectorizer()
        # 使用 pytest 检查 fit_transform 方法中的 ValueError 异常
        hv.fit_transform(["hello world", np.nan, "hello hello"])

    with pytest.raises(exception, match=message):
        func()


# 测试 TfidfVectorizer 的 binary 参数设置是否生效
def test_tfidfvectorizer_binary():
    # 非回归测试：TfidfVectorizer 之前忽略其 binary 参数
    v = TfidfVectorizer(binary=True, use_idf=False, norm=None)
    # 断言 v 的 binary 参数为 True
    assert v.binary

    # 对文档集合 ["hello world", "hello hello"] 使用 v 进行拟合和转换
    X = v.fit_transform(["hello world", "hello hello"]).toarray()
    # 断言转换后的数组与预期的结果数组相等
    assert_array_equal(X.ravel(), [1, 1, 1, 0])
    # 再次使用 v 对同一文档集合进行转换，断言结果数组与预期的结果数组相等
    X2 = v.transform(["hello world", "hello hello"]).toarray()
    assert_array_equal(X2.ravel(), [1, 1, 1, 0])


# 测试 TfidfVectorizer 导出 IDF 属性
def test_tfidfvectorizer_export_idf():
    # 创建一个使用 IDF 的 TfidfVectorizer 对象 vect
    vect = TfidfVectorizer(use_idf=True)
    # 对文档集合 JUNK_FOOD_DOCS 进行拟合
    vect.fit(JUNK_FOOD_DOCS)
    # 断言 vect 的 IDF 属性与其内部 _tfidf 对象的 IDF 属性近似相等
    assert_array_almost_equal(vect.idf_, vect._tfidf.idf_)


# 测试 Vectorizer 对象的词汇表克隆
def test_vectorizer_vocab_clone():
    # 创建一个使用特定词汇表的 TfidfVectorizer 对象 vect_vocab
    vect_vocab = TfidfVectorizer(vocabulary=["the"])
    # 克隆 vect_vocab 得到 vect_vocab_clone
    vect_vocab_clone = clone(vect_vocab)
    # 使用 ALL_FOOD_DOCS 对 vect_vocab 和 vect_vocab_clone 进行拟合
    vect_vocab.fit(ALL_FOOD_DOCS)
    vect_vocab_clone.fit(ALL_FOOD_DOCS)
    # 断言 vect_vocab_clone 的词汇表与 vect_vocab 的词汇表相等
    assert vect_vocab_clone.vocabulary_ == vect_vocab.vocabulary_
# 使用 pytest.mark.parametrize 装饰器标记参数化测试函数，测试不同的 Vectorizer 类型
@pytest.mark.parametrize(
    "Vectorizer", (CountVectorizer, TfidfVectorizer, HashingVectorizer)
)
def test_vectorizer_string_object_as_input(Vectorizer):
    # 错误消息，表示期望迭代器遍历原始文本文档，但收到了字符串对象
    message = "Iterable over raw text documents expected, string object received."
    # 创建指定类型的 Vectorizer 对象
    vec = Vectorizer()

    # 测试 fit_transform 方法对单个字符串输入的行为
    with pytest.raises(ValueError, match=message):
        vec.fit_transform("hello world!")

    # 测试 fit 方法对单个字符串输入的行为
    with pytest.raises(ValueError, match=message):
        vec.fit("hello world!")

    # 测试 fit 方法对字符串列表输入的行为
    vec.fit(["some text", "some other text"])

    # 测试 transform 方法对单个字符串输入的行为
    with pytest.raises(ValueError, match=message):
        vec.transform("hello world!")


# 使用 pytest.mark.parametrize 装饰器标记参数化测试函数，测试不同的 X_dtype 类型
@pytest.mark.parametrize("X_dtype", [np.float32, np.float64])
def test_tfidf_transformer_type(X_dtype):
    # 创建稀疏矩阵 X，指定类型为 X_dtype
    X = sparse.rand(10, 20000, dtype=X_dtype, random_state=42)
    # 使用 TfidfTransformer 对象进行拟合转换
    X_trans = TfidfTransformer().fit_transform(X)
    # 断言转换后的数据类型与原始数据类型相同
    assert X_trans.dtype == X.dtype


# 使用 pytest.mark.parametrize 装饰器标记参数化测试函数，测试不同的 CSC_CONTAINERS 和 CSR_CONTAINERS 组合
@pytest.mark.parametrize(
    "csc_container, csr_container", product(CSC_CONTAINERS, CSR_CONTAINERS)
)
def test_tfidf_transformer_sparse(csc_container, csr_container):
    # 创建稀疏矩阵 X，数据类型为 np.float64
    X = sparse.rand(10, 20000, dtype=np.float64, random_state=42)
    # 将 X 转换为 CSC 和 CSR 格式
    X_csc = csc_container(X)
    X_csr = csr_container(X)

    # 使用 TfidfTransformer 对象分别对 CSC 和 CSR 格式的 X 进行拟合转换
    X_trans_csc = TfidfTransformer().fit_transform(X_csc)
    X_trans_csr = TfidfTransformer().fit_transform(X_csr)

    # 断言稠密和稀疏格式的转换结果相近
    assert_allclose_dense_sparse(X_trans_csc, X_trans_csr)
    # 断言转换后的格式与原始格式相同
    assert X_trans_csc.format == X_trans_csr.format


# 使用 pytest.mark.parametrize 装饰器标记参数化测试函数，测试不同的 vectorizer_dtype, output_dtype 和 warning_expected 组合
@pytest.mark.parametrize(
    "vectorizer_dtype, output_dtype, warning_expected",
    [
        (np.int32, np.float64, True),
        (np.int64, np.float64, True),
        (np.float32, np.float32, False),
        (np.float64, np.float64, False),
    ],
)
def test_tfidf_vectorizer_type(vectorizer_dtype, output_dtype, warning_expected):
    # 创建包含字符串的 numpy 数组 X
    X = np.array(["numpy", "scipy", "sklearn"])
    # 创建指定 dtype 的 TfidfVectorizer 对象
    vectorizer = TfidfVectorizer(dtype=vectorizer_dtype)

    # 期望的警告消息，指示应使用 'dtype'
    warning_msg_match = "'dtype' should be used."

    # 如果预期有警告消息，则测试是否会发出 UserWarning
    if warning_expected:
        with pytest.warns(UserWarning, match=warning_msg_match):
            X_idf = vectorizer.fit_transform(X)
    else:
        # 如果不预期警告消息，则测试是否会抛出 UserWarning 异常
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            X_idf = vectorizer.fit_transform(X)

    # 断言转换后的数据类型与预期的输出数据类型相同
    assert X_idf.dtype == output_dtype


# 使用 pytest.mark.parametrize 装饰器标记参数化测试函数，测试不同的 vectorizer 对象
@pytest.mark.parametrize(
    "vec",
    [
        HashingVectorizer(ngram_range=(2, 1)),
        CountVectorizer(ngram_range=(2, 1)),
        TfidfVectorizer(ngram_range=(2, 1)),
    ],
)
def test_vectorizers_invalid_ngram_range(vec):
    # vectorizers 可能会使用无效的 ngram 范围进行初始化
    # 测试是否会引发错误消息
    invalid_range = vec.ngram_range
    message = re.escape(
        f"Invalid value for ngram_range={invalid_range} "
        "lower boundary larger than the upper boundary."
    )

    # 测试 fit 方法对单个输入的行为
    with pytest.raises(ValueError, match=message):
        vec.fit(["good news everyone"])

    # 测试 fit_transform 方法对单个输入的行为
    with pytest.raises(ValueError, match=message):
        vec.fit_transform(["good news everyone"])
    # 如果 vec 是 HashingVectorizer 的实例，则执行以下代码块
    if isinstance(vec, HashingVectorizer):
        # 使用 pytest 的断言来捕获 ValueError 异常，并检查异常消息是否匹配指定的 message
        with pytest.raises(ValueError, match=message):
            # 对输入数据 ["good news everyone"] 进行转换操作
            vec.transform(["good news everyone"])
def _check_stop_words_consistency(estimator):
    # 获取估计器的停用词列表
    stop_words = estimator.get_stop_words()
    # 获取估计器的分词器
    tokenize = estimator.build_tokenizer()
    # 获取估计器的预处理器
    preprocess = estimator.build_preprocessor()
    # 调用估计器的停用词一致性检查方法，并返回结果
    return estimator._check_stop_words_consistency(stop_words, preprocess, tokenize)


def test_vectorizer_stop_words_inconsistent():
    lstr = r"\['and', 'll', 've'\]"
    # 构建警告信息字符串
    message = (
        "Your stop_words may be inconsistent with your "
        "preprocessing. Tokenizing the stop words generated "
        "tokens %s not in stop_words." % lstr
    )
    # 针对多个向量化器进行测试
    for vec in [CountVectorizer(), TfidfVectorizer(), HashingVectorizer()]:
        # 设置停用词参数
        vec.set_params(stop_words=["you've", "you", "you'll", "AND"])
        # 检查是否触发警告，并匹配警告信息
        with pytest.warns(UserWarning, match=message):
            vec.fit_transform(["hello world"])
        # 重置停用词验证状态
        del vec._stop_words_id
        # 断言停用词一致性检查为假
        assert _check_stop_words_consistency(vec) is False

    # 每个停用词列表仅触发一次警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        vec.fit_transform(["hello world"])
    # 断言停用词一致性检查为None
    assert _check_stop_words_consistency(vec) is None

    # 测试不一致性评估的缓存
    vec.set_params(stop_words=["you've", "you", "you'll", "blah", "AND"])
    with pytest.warns(UserWarning, match=message):
        vec.fit_transform(["hello world"])


@skip_if_32bit
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_countvectorizer_sort_features_64bit_sparse_indices(csr_container):
    """
    检查CountVectorizer._sort_features方法保留其稀疏特征矩阵的dtype。

    在32位平台上跳过此测试，请参见：
        https://github.com/scikit-learn/scikit-learn/pull/11295
    获取更多详细信息。
    """

    X = csr_container((5, 5), dtype=np.int64)

    # 强制indices和indptr为int64类型
    INDICES_DTYPE = np.int64
    X.indices = X.indices.astype(INDICES_DTYPE)
    X.indptr = X.indptr.astype(INDICES_DTYPE)

    # 设置词汇表
    vocabulary = {"scikit-learn": 0, "is": 1, "great!": 2}

    # 调用_countvectorizer._sort_features方法
    Xs = CountVectorizer()._sort_features(X, vocabulary)

    # 断言indices的dtype与预期一致
    assert INDICES_DTYPE == Xs.indices.dtype


@pytest.mark.parametrize(
    "Estimator", [CountVectorizer, TfidfVectorizer, HashingVectorizer]
)
def test_stop_word_validation_custom_preprocessor(Estimator):
    data = [{"text": "some text"}]

    # 初始化向量化器
    vec = Estimator()
    # 断言停用词一致性检查为真
    assert _check_stop_words_consistency(vec) is True

    # 使用自定义预处理器和停用词列表初始化向量化器
    vec = Estimator(preprocessor=lambda x: x["text"], stop_words=["and"])
    # 断言停用词一致性检查为"error"
    assert _check_stop_words_consistency(vec) == "error"
    # 检查是否缓存检查结果
    assert _check_stop_words_consistency(vec) is None
    # 对数据进行拟合
    vec.fit_transform(data)

    # 使用自定义估计器类初始化向量化器
    class CustomEstimator(Estimator):
        def build_preprocessor(self):
            return lambda x: x["text"]

    vec = CustomEstimator(stop_words=["and"])
    # 断言停用词一致性检查为"error"
    assert _check_stop_words_consistency(vec) == "error"

    # 使用自定义分词器和停用词列表初始化向量化器
    vec = Estimator(
        tokenizer=lambda doc: re.compile(r"\w{1,}").findall(doc), stop_words=["and"]
    )
    # 使用断言检查给定向量 vec 的停用词一致性，断言条件为真
    assert _check_stop_words_consistency(vec) is True
@pytest.mark.parametrize(
    "Estimator", [CountVectorizer, TfidfVectorizer, HashingVectorizer]
)
@pytest.mark.parametrize(
    "input_type, err_type, err_msg",
    [
        ("filename", FileNotFoundError, ""),  # 参数化测试：使用文件名时，预期捕获 FileNotFoundError 异常
        ("file", AttributeError, "'str' object has no attribute 'read'"),  # 参数化测试：使用文件时，预期捕获 AttributeError 异常
    ],
)
def test_callable_analyzer_error(Estimator, input_type, err_type, err_msg):
    data = ["this is text, not file or filename"]
    with pytest.raises(err_type, match=err_msg):
        Estimator(analyzer=lambda x: x.split(), input=input_type).fit_transform(data)


@pytest.mark.parametrize(
    "Estimator",
    [
        CountVectorizer,
        TfidfVectorizer,
        pytest.param(HashingVectorizer),  # 参数化测试：包括 HashingVectorizer 作为估算器之一
    ],
)
@pytest.mark.parametrize(
    "analyzer", [lambda doc: open(doc, "r"), lambda doc: doc.read()]  # 参数化测试：定义不同的分析器函数
)
@pytest.mark.parametrize("input_type", ["file", "filename"])  # 参数化测试：定义不同的输入类型
def test_callable_analyzer_change_behavior(Estimator, analyzer, input_type):
    data = ["this is text, not file or filename"]
    with pytest.raises((FileNotFoundError, AttributeError)):
        Estimator(analyzer=analyzer, input=input_type).fit_transform(data)


@pytest.mark.parametrize(
    "Estimator", [CountVectorizer, TfidfVectorizer, HashingVectorizer]
)
def test_callable_analyzer_reraise_error(tmpdir, Estimator):
    # 检查分析器抛出的自定义异常是否向用户显示
    def analyzer(doc):
        raise Exception("testing")

    f = tmpdir.join("file.txt")
    f.write("sample content\n")

    with pytest.raises(Exception, match="testing"):
        Estimator(analyzer=analyzer, input="file").fit_transform([f])


@pytest.mark.parametrize(
    "Vectorizer", [CountVectorizer, HashingVectorizer, TfidfVectorizer]
)
@pytest.mark.parametrize(
    (
        "stop_words, tokenizer, preprocessor, ngram_range, token_pattern,"
        "analyzer, unused_name, ovrd_name, ovrd_msg"
    ),
    # 第一个元组条件：当指定了 'stop_words' 参数但未指定 'analyzer' 参数为 'word' 时
    (
        ["you've", "you'll"],   # 'stop_words' 参数的值列表
        None,                   # 未指定 'analyzer' 参数
        None,                   # 未指定 'tokenizer' 参数
        (1, 1),                 # 'ngram_range' 参数的值为 (1, 1)
        None,                   # 未指定 'token_pattern' 参数
        "char",                 # 'analyzer' 参数的值为 'char'
        "'stop_words'",         # 字符串，表示参数 'stop_words'
        "'analyzer'",           # 字符串，表示参数 'analyzer'
        "!= 'word'",            # 条件表达式，判断 'analyzer' 参数不等于 'word'
    ),
    
    # 第二个元组条件：当未指定 'stop_words' 参数，但指定了自定义的 'tokenizer' 函数
    (
        None,                   # 未指定 'stop_words' 参数
        lambda s: s.split(),    # 自定义的 'tokenizer' 函数，用于分割字符串 s
        None,                   # 未指定 'analyzer' 参数
        (1, 1),                 # 'ngram_range' 参数的值为 (1, 1)
        None,                   # 未指定 'token_pattern' 参数
        "char",                 # 'analyzer' 参数的值为 'char'
        "'tokenizer'",          # 字符串，表示参数 'tokenizer'
        "'analyzer'",           # 字符串，表示参数 'analyzer'
        "!= 'word'",            # 条件表达式，判断 'analyzer' 参数不等于 'word'
    ),
    
    # 第三个元组条件：当未指定 'stop_words' 参数，但指定了自定义的 'token_pattern' 正则表达式
    (
        None,                   # 未指定 'stop_words' 参数
        lambda s: s.split(),    # 自定义的 'tokenizer' 函数，用于分割字符串 s
        None,                   # 未指定 'analyzer' 参数
        (1, 1),                 # 'ngram_range' 参数的值为 (1, 1)
        r"\w+",                 # 自定义的 'token_pattern' 参数，表示匹配单词字符的正则表达式
        "word",                 # 'analyzer' 参数的值为 'word'
        "'token_pattern'",      # 字符串，表示参数 'token_pattern'
        "'tokenizer'",          # 字符串，表示参数 'tokenizer'
        "is not None",          # 条件表达式，判断 'tokenizer' 参数不为 None
    ),
    
    # 第四个元组条件：当未指定 'stop_words' 参数，但指定了自定义的 'preprocessor' 函数
    (
        None,                   # 未指定 'stop_words' 参数
        None,                   # 未指定 'tokenizer' 参数
        lambda s: s.upper(),    # 自定义的 'preprocessor' 函数，用于将字符串 s 转为大写
        (1, 1),                 # 'ngram_range' 参数的值为 (1, 1)
        r"\w+",                 # 自定义的 'token_pattern' 参数，表示匹配单词字符的正则表达式
        lambda s: s.upper(),    # 自定义的 'analyzer' 函数，用于将字符串 s 转为大写
        "'preprocessor'",       # 字符串，表示参数 'preprocessor'
        "'analyzer'",           # 字符串，表示参数 'analyzer'
        "is callable",          # 条件表达式，判断 'analyzer' 参数是可调用的
    ),
    
    # 第五个元组条件：当未指定 'stop_words' 参数，但指定了自定义的 'ngram_range' 函数
    (
        None,                   # 未指定 'stop_words' 参数
        None,                   # 未指定 'tokenizer' 参数
        None,                   # 未指定 'analyzer' 参数
        (1, 2),                 # 'ngram_range' 参数的值为 (1, 2)
        None,                   # 未指定 'token_pattern' 参数
        lambda s: s.upper(),    # 自定义的 'analyzer' 函数，用于将字符串 s 转为大写
        "'ngram_range'",        # 字符串，表示参数 'ngram_range'
        "'analyzer'",           # 字符串，表示参数 'analyzer'
        "is callable",          # 条件表达式，判断 'analyzer' 参数是可调用的
    ),
    
    # 第六个元组条件：当未指定 'stop_words' 参数，但指定了自定义的 'token_pattern' 正则表达式
    (
        None,                   # 未指定 'stop_words' 参数
        None,                   # 未指定 'tokenizer' 参数
        None,                   # 未指定 'analyzer' 参数
        (1, 1),                 # 'ngram_range' 参数的值为 (1, 1)
        r"\w+",                 # 自定义的 'token_pattern' 参数，表示匹配单词字符的正则表达式
        "char",                 # 'analyzer' 参数的值为 'char'
        "'token_pattern'",      # 字符串，表示参数 'token_pattern'
        "'analyzer'",           # 字符串，表示参数 'analyzer'
        "!= 'word'",            # 条件表达式，判断 'analyzer' 参数不等于 'word'
    ),
# 定义测试函数，用于测试未使用参数时是否会发出警告
def test_unused_parameters_warn(
    Vectorizer,         # 向量化器类，用于创建向量化器对象
    stop_words,         # 停用词列表或 None，用于向量化器参数设置
    tokenizer,          # 分词器函数或 None，用于向量化器参数设置
    preprocessor,       # 预处理函数或 None，用于向量化器参数设置
    ngram_range,        # 元组，表示 n-gram 范围，用于向量化器参数设置
    token_pattern,      # 正则表达式，用于向量化器参数设置
    analyzer,           # 字符串，指定分析器类型，用于向量化器参数设置
    unused_name,        # 字符串，指示未使用的参数名
    ovrd_name,          # 字符串，指示覆盖的参数名
    ovrd_msg,           # 字符串，指示覆盖的原因信息
):
    # 使用预定义的文本数据作为训练数据
    train_data = JUNK_FOOD_DOCS
    # 创建指定类型的向量化器对象
    vect = Vectorizer()
    # 设置向量化器对象的参数
    vect.set_params(
        stop_words=stop_words,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
        analyzer=analyzer,
    )
    # 构建警告消息，指示未使用的参数将被忽略
    msg = "The parameter %s will not be used since %s %s" % (
        unused_name,
        ovrd_name,
        ovrd_msg,
    )
    # 使用 pytest 的 warn 模块检查是否会发出 UserWarning，且消息匹配预期消息
    with pytest.warns(UserWarning, match=msg):
        # 对向量化器对象进行拟合操作
        vect.fit(train_data)


# 使用参数化测试装饰器，测试不同的向量化器及其输入数据
@pytest.mark.parametrize(
    "Vectorizer, X",
    (
        (HashingVectorizer, [{"foo": 1, "bar": 2}, {"foo": 3, "baz": 1}]),
        (CountVectorizer, JUNK_FOOD_DOCS),
    ),
)
def test_n_features_in(Vectorizer, X):
    # 对于向量化器，n_features_in_ 属性没有意义
    vectorizer = Vectorizer()
    # 断言向量化器对象不具有 n_features_in_ 属性
    assert not hasattr(vectorizer, "n_features_in_")
    # 对向量化器对象进行拟合操作
    vectorizer.fit(X)
    # 再次断言向量化器对象不具有 n_features_in_ 属性
    assert not hasattr(vectorizer, "n_features_in_")


# 测试样本顺序不变性时，设置 max_features 参数的影响
def test_tie_breaking_sample_order_invariance():
    # 创建一个 CountVectorizer 对象，设置 max_features=1
    vec = CountVectorizer(max_features=1)
    # 分别用 ["hello", "world"] 和 ["world", "hello"] 进行拟合并获取词汇表
    vocab1 = vec.fit(["hello", "world"]).vocabulary_
    vocab2 = vec.fit(["world", "hello"]).vocabulary_
    # 断言两次拟合后的词汇表相同
    assert vocab1 == vocab2


# 测试非负哈希向量化器结果索引的情况
def test_nonnegative_hashing_vectorizer_result_indices():
    # 创建一个 HashingVectorizer 对象，设置 n_features 和 ngram_range
    hashing = HashingVectorizer(n_features=1000000, ngram_range=(2, 3))
    # 对输入文本 "22pcs efuture" 进行转换并获取非零元素的索引
    indices = hashing.transform(["22pcs efuture"]).indices
    # 断言索引值大于等于 0
    assert indices[0] >= 0


# 使用参数化测试装饰器，检查向量化器类是否定义了 set_output 方法
@pytest.mark.parametrize(
    "Estimator", [CountVectorizer, TfidfVectorizer, TfidfTransformer, HashingVectorizer]
)
def test_vectorizers_do_not_have_set_output(Estimator):
    """检查向量化器类是否不定义 set_output 方法。"""
    est = Estimator()
    # 断言向量化器对象不具有 set_output 方法
    assert not hasattr(est, "set_output")


# 使用参数化测试装饰器，检查 TfidfTransformer 在复制参数（copy）设置下的行为
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_tfidf_transformer_copy(csr_container):
    """检查 TfidfTransformer.transform 方法在复制参数（copy）设置下的行为。"""
    # 创建一个稀疏矩阵 X，类型为 np.float64
    X = sparse.rand(10, 20000, dtype=np.float64, random_state=42)
    # 使用 csr_container 将 X 转换为稀疏矩阵格式
    X_csr = csr_container(X)
    # 记录原始矩阵 X_csr 的副本，以备后续比较
    X_csr_original = X_csr.copy()
    # 创建 TfidfTransformer 对象并对输入进行拟合
    transformer = TfidfTransformer().fit(X_csr)
    # 使用复制参数（copy=True）进行转换操作
    X_transform = transformer.transform(X_csr, copy=True)
    # 断言转换后的稀疏矩阵与原始矩阵 X_csr 相等
    assert_allclose_dense_sparse(X_csr, X_csr_original)
    # 断言转换后的对象不是原始矩阵 X_csr
    assert X_transform is not X_csr
    # 使用复制参数（copy=False）进行转换操作
    X_transform = transformer.transform(X_csr, copy=False)
    # 断言转换后的对象是原始矩阵 X_csr
    assert X_transform is X_csr
    # 使用 pytest 的 raises 断言，检查是否引发了 AssertionError
    with pytest.raises(AssertionError):
        assert_allclose_dense_sparse(X_csr, X_csr_original)
```