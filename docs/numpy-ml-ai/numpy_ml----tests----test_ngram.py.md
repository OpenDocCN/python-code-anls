# `numpy-ml\numpy_ml\tests\test_ngram.py`

```
# 禁用 flake8 检查
# 导入临时文件模块
import tempfile

# 导入 nltk 和 numpy 模块
import nltk
import numpy as np

# 从上级目录中导入 tokenize_words 函数
from ..preprocessing.nlp import tokenize_words
# 从上级目录中导入 AdditiveNGram 和 MLENGram 类
from ..ngram import AdditiveNGram, MLENGram
# 从上级目录中导入 random_paragraph 函数
from ..utils.testing import random_paragraph

# 定义 MLEGold 类
class MLEGold:
    def __init__(
        self, N, K=1, unk=True, filter_stopwords=True, filter_punctuation=True
    ):
        # 初始化类的属性
        self.N = N
        self.K = K
        self.unk = unk
        self.filter_stopwords = filter_stopwords
        self.filter_punctuation = filter_punctuation

        # 设置超参数字典
        self.hyperparameters = {
            "N": N,
            "K": K,
            "unk": unk,
            "filter_stopwords": filter_stopwords,
            "filter_punctuation": filter_punctuation,
        }

    # 计算给定 N-gram 的对数概率
    def log_prob(self, words, N):
        assert N in self.counts, "You do not have counts for {}-grams".format(N)

        if N > len(words):
            err = "Not enough words for a gram-size of {}: {}".format(N, len(words))
            raise ValueError(err)

        total_prob = 0
        for ngram in nltk.ngrams(words, N):
            total_prob += self._log_ngram_prob(ngram)
        return total_prob

    # 计算给定 N-gram 的对数概率
    def _log_ngram_prob(self, ngram):
        N = len(ngram)
        return self._models[N].logscore(ngram[-1], ngram[:-1])

# 定义 AdditiveGold 类
class AdditiveGold:
    def __init__(
        self, N, K=1, unk=True, filter_stopwords=True, filter_punctuation=True
    ):
        # 初始化类的属性
        self.N = N
        self.K = K
        self.unk = unk
        self.filter_stopwords = filter_stopwords
        self.filter_punctuation = filter_punctuation

        # 设置超参数字典
        self.hyperparameters = {
            "N": N,
            "K": K,
            "unk": unk,
            "filter_stopwords": filter_stopwords,
            "filter_punctuation": filter_punctuation,
        }
    # 计算给定单词列表的对数概率和
    def log_prob(self, words, N):
        # 检查是否存在 N-grams 的计数
        assert N in self.counts, "You do not have counts for {}-grams".format(N)

        # 如果单词数量不足以形成 N-grams，则引发异常
        if N > len(words):
            err = "Not enough words for a gram-size of {}: {}".format(N, len(words))
            raise ValueError(err)

        # 初始化总概率
        total_prob = 0
        # 遍历生成给定单词列表的 N-grams，并计算其对数概率
        for ngram in nltk.ngrams(words, N):
            total_prob += self._log_ngram_prob(ngram)
        # 返回总概率
        return total_prob

    # 计算给定 N-gram 的对数概率
    def _log_ngram_prob(self, ngram):
        # 获取 N-gram 的长度
        N = len(ngram)
        # 调用模型对象的 logscore 方法计算 N-gram 的对数概率
        return self._models[N].logscore(ngram[-1], ngram[:-1])
# 测试最大似然估计模型
def test_mle():
    # 生成一个随机整数 N，范围在 [2, 5) 之间
    N = np.random.randint(2, 5)
    # 创建一个最大似然估计的金标准对象
    gold = MLEGold(N, unk=True, filter_stopwords=False, filter_punctuation=False)
    # 创建一个最大似然估计的自己实现的对象
    mine = MLENGram(N, unk=True, filter_stopwords=False, filter_punctuation=False)

    # 使用临时文件进行训练
    with tempfile.NamedTemporaryFile() as temp:
        # 将随机生成的一个包含 1000 个单词的段落写入临时文件
        temp.write(bytes(" ".join(random_paragraph(1000)), encoding="utf-8-sig"))
        # 使用金标准对象进行训练
        gold.train(temp.name, encoding="utf-8-sig")
        # 使用自己实现的对象进行训练
        mine.train(temp.name, encoding="utf-8-sig")

    # 遍历自己实现的对象中 N 阶计数的键
    for k in mine.counts[N].keys():
        # 如果键的第一个和第二个元素相等，并且在 ("<bol>", "<eol>") 中，则跳过
        if k[0] == k[1] and k[0] in ("<bol>", "<eol>"):
            continue

        # 错误信息字符串模板
        err_str = "{}, mine: {}, gold: {}"
        # 断言自己实现的对象中的计数与金标准对象中的计数相等
        assert mine.counts[N][k] == gold.counts[N][k], err_str.format(
            k, mine.counts[N][k], gold.counts[N][k]
        )

        # 计算自己实现的对象中 k 的对数概率
        M = mine.log_prob(k, N)
        # 计算金标准对象中 k 的对数概率，并转换为以自然对数为底
        G = gold.log_prob(k, N) / np.log2(np.e)
        # 使用 np.testing.assert_allclose 检查 M 和 G 是否接近
        np.testing.assert_allclose(M, G)
        # 打印 "PASSED"
        print("PASSED")


# 测试加法平滑模型
def test_additive():
    # 生成一个随机浮点数 K
    K = np.random.rand()
    # 生成一个随机整数 N，范围在 [2, 5) 之间
    N = np.random.randint(2, 5)
    # 创建一个加法平滑的金标准对象
    gold = AdditiveGold(
        N, K, unk=True, filter_stopwords=False, filter_punctuation=False
    )
    # 创建一个加法平滑的自己实现的对象
    mine = AdditiveNGram(
        N, K, unk=True, filter_stopwords=False, filter_punctuation=False
    )

    # 使用临时文件进行训练
    with tempfile.NamedTemporaryFile() as temp:
        # 将随机生成的一个包含 1000 个单词的段落写入临时文件
        temp.write(bytes(" ".join(random_paragraph(1000)), encoding="utf-8-sig"))
        # 使用金标准对象进行训练
        gold.train(temp.name, encoding="utf-8-sig")
        # 使用自己实现的对象进行训练
        mine.train(temp.name, encoding="utf-8-sig")

    # 遍历自己实现的对象中 N 阶计数的键
    for k in mine.counts[N].keys():
        # 如果键的第一个和第二个元素相等，并且在 ("<bol>", "<eol>") 中，则跳过
        if k[0] == k[1] and k[0] in ("<bol>", "<eol>"):
            continue

        # 错误信息字符串模板
        err_str = "{}, mine: {}, gold: {}"
        # 断言自己实现的对象中的计数与金标准对象中的计数相等
        assert mine.counts[N][k] == gold.counts[N][k], err_str.format(
            k, mine.counts[N][k], gold.counts[N][k]
        )

        # 计算自己实现的对象中 k 的对数概率
        M = mine.log_prob(k, N)
        # 计算金标准对象中 k 的对数概率，并转换为以自然对数为底
        G = gold.log_prob(k, N) / np.log2(np.e)
        # 使用 np.testing.assert_allclose 检查 M 和 G 是否接近
        np.testing.assert_allclose(M, G)
        # 打印 "PASSED"
        print("PASSED")
```