# `numpy-ml\numpy_ml\ngram\ngram.py`

```py
# 导入文本包装、抽象基类和计数器模块
import textwrap
from abc import ABC, abstractmethod
from collections import Counter

# 导入 NumPy 库
import numpy as np

# 从自定义的线性回归模块中导入 LinearRegression 类
from numpy_ml.linear_models import LinearRegression
# 从自定义的自然语言处理预处理模块中导入 tokenize_words、ngrams 和 strip_punctuation 函数
from numpy_ml.preprocessing.nlp import tokenize_words, ngrams, strip_punctuation

# 定义一个抽象基类 NGramBase
class NGramBase(ABC):
    def __init__(self, N, unk=True, filter_stopwords=True, filter_punctuation=True):
        """
        A simple word-level N-gram language model.

        Notes
        -----
        This is not optimized code and will be slow for large corpora. To see
        how industry-scale NGram models are handled, see the SRLIM-format:

            http://www.speech.sri.com/projects/srilm/
        """
        # 初始化 N-gram 模型的参数
        self.N = N
        self.unk = unk
        self.filter_stopwords = filter_stopwords
        self.filter_punctuation = filter_punctuation

        # 存储超参数信息
        self.hyperparameters = {
            "N": N,
            "unk": unk,
            "filter_stopwords": filter_stopwords,
            "filter_punctuation": filter_punctuation,
        }

        # 调用父类的构造函数
        super().__init__()
    # 训练语言模型，统计语料库中文本的 n-gram 计数
    def train(self, corpus_fp, vocab=None, encoding=None):
        """
        Compile the n-gram counts for the text(s) in `corpus_fp`.

        Notes
        -----
        After running `train`, the ``self.counts`` attribute will store
        dictionaries of the `N`, `N-1`, ..., 1-gram counts.

        Parameters
        ----------
        corpus_fp : str
            The path to a newline-separated text corpus file.
        vocab : :class:`~numpy_ml.preprocessing.nlp.Vocabulary` instance or None
            If not None, only the words in `vocab` will be used to construct
            the language model; all out-of-vocabulary words will either be
            mappend to ``<unk>`` (if ``self.unk = True``) or removed (if
            ``self.unk = False``). Default is None.
        encoding : str or None
            Specifies the text encoding for corpus. Common entries are 'utf-8',
            'utf-8-sig', 'utf-16'. Default is None.
        """
        # 调用内部方法 _train 来实际执行训练过程
        return self._train(corpus_fp, vocab=vocab, encoding=encoding)
    # 定义一个用于训练 N-gram 模型的方法，接受语料文件路径、词汇表和编码方式作为参数
    def _train(self, corpus_fp, vocab=None, encoding=None):
        # 获取超参数
        H = self.hyperparameters
        # 初始化存储 N-gram 的字典
        grams = {N: [] for N in range(1, self.N + 1)}
        # 初始化计数器字典
        counts = {N: Counter() for N in range(1, self.N + 1)}
        # 获取是否过滤停用词和标点符号的设置
        filter_stop, filter_punc = H["filter_stopwords"], H["filter_punctuation"]

        # 初始化词数计数器
        _n_words = 0
        # 初始化词汇集合
        tokens = {"<unk>"}
        # 初始化句子起始和结束标记
        bol, eol = ["<bol>"], ["<eol>"]

        # 打开语料文件进行读取
        with open(corpus_fp, "r", encoding=encoding) as text:
            # 逐行处理文本
            for line in text:
                # 如果需要过滤标点符号，则去除标点符号
                line = strip_punctuation(line) if filter_punc else line
                # 对文本进行分词处理，根据设置过滤停用词
                words = tokenize_words(line, filter_stopwords=filter_stop)

                # 如果提供了词汇表，则根据词汇表过滤词汇
                if vocab is not None:
                    words = vocab.filter(words, H["unk"])

                # 如果分词结果为空，则继续处理下一行
                if len(words) == 0:
                    continue

                # 更新词数计数器
                _n_words += len(words)
                # 更新词汇集合
                tokens.update(words)

                # 计算 n, n-1, ... 1-gram
                for N in range(1, self.N + 1):
                    # 对词汇进行填充，添加起始和结束标记
                    words_padded = bol * max(1, N - 1) + words + eol * max(1, N - 1)
                    # 将 n-gram 添加到对应的 grams 中
                    grams[N].extend(ngrams(words_padded, N))

        # 统计每个 N-gram 的出现次数
        for N in counts.keys():
            counts[N].update(grams[N])

        # 统计总词数
        n_words = {N: np.sum(list(counts[N].values())) for N in range(1, self.N + 1)}
        n_words[1] = _n_words

        # 统计词汇量
        n_tokens = {N: len(counts[N]) for N in range(2, self.N + 1)}
        n_tokens[1] = len(vocab) if vocab is not None else len(tokens)

        # 更新模型的计数器、总词数和词汇量
        self.counts = counts
        self.n_words = n_words
        self.n_tokens = n_tokens
    # 定义一个方法，用于返回在 N-gram 语言模型下提议的下一个单词的分布
    def completions(self, words, N):
        """
        Return the distribution over proposed next words under the `N`-gram
        language model.

        Parameters
        ----------
        words : list or tuple of strings
            The initial sequence of words
        N : int
            The gram-size of the language model to use to generate completions

        Returns
        -------
        probs : list of (word, log_prob) tuples
            The list of possible next words and their log probabilities under
            the `N`-gram language model (unsorted)
        """
        # 确保 N 不超过 words 的长度加一
        N = min(N, len(words) + 1)
        # 检查 self.counts 中是否有 N-grams 的计数
        assert N in self.counts, "You do not have counts for {}-grams".format(N)
        # 检查 words 中是否至少有 N-1 个单词
        assert len(words) >= N - 1, "`words` must have at least {} words".format(N - 1)

        # 初始化一个空列表用于存储下一个单词的概率
        probs = []
        # 获取基础元组，包含最后 N-1 个单词的小写形式
        base = tuple(w.lower() for w in words[-N + 1 :])
        # 遍历 N-grams 的计数字典中的键
        for k in self.counts[N].keys():
            # 如果当前键的前 N-1 个元素与基础元组相同
            if k[:-1] == base:
                # 计算当前键的概率，并添加到概率列表中
                c_prob = self._log_ngram_prob(base + k[-1:])
                probs.append((k[-1], c_prob))
        # 返回概率列表
        return probs
    def generate(self, N, seed_words=["<bol>"], n_sentences=5):
        """
        使用 N-gram 语言模型生成句子。

        Parameters
        ----------
        N : int
            要生成的模型的 gram 大小
        seed_words : list of strs
            用于初始化句子生成的种子词列表。默认为 ["<bol>"]。
        sentences : int
            从 N-gram 模型中生成的句子数量。默认为 50。

        Returns
        -------
        sentences : str
            从 N-gram 模型中抽样生成的句子，用空格连接，句子之间用换行符分隔。
        """
        counter = 0
        sentences = []  # 存储生成的句子
        words = seed_words.copy()  # 复制种子词列表
        while counter < n_sentences:
            # 获取下一个词和对应的概率
            nextw, probs = zip(*self.completions(words, N))
            # 如果进行了平滑处理，则重新归一化概率
            probs = np.exp(probs) / np.exp(probs).sum()
            # 根据概率选择下一个词
            next_word = np.random.choice(nextw, p=probs)

            # 如果到达句子结尾，保存句子并开始新句子
            if next_word == "<eol>":
                # 将词列表连接成句子
                S = " ".join([w for w in words if w != "<bol>"])
                # 将句子格式化为指定宽度，以便显示
                S = textwrap.fill(S, 90, initial_indent="", subsequent_indent="   ")
                print(S)  # 打印句子
                words.append(next_word)
                sentences.append(words)  # 将句子添加到结果列表
                words = seed_words.copy()  # 重置词列表为种子词列表
                counter += 1
                continue

            words.append(next_word)
        return sentences  # 返回生成的句子列表
    # 计算给定单词序列的模型困惑度
    def perplexity(self, words, N):
        r"""
        Calculate the model perplexity on a sequence of words.

        Notes
        -----
        Perplexity, `PP`, is defined as

        .. math::

            PP(W)  =  \left( \frac{1}{p(W)} \right)^{1 / n}

        or simply

        .. math::

            PP(W)  &=  \exp(-\log p(W) / n) \\
                   &=  \exp(H(W))

        where :math:`W = [w_1, \ldots, w_k]` is a sequence of words, `H(w)` is
        the cross-entropy of `W` under the current model, and `n` is the number
        of `N`-grams in `W`.

        Minimizing perplexity is equivalent to maximizing the probability of
        `words` under the `N`-gram model. It may also be interpreted as the
        average branching factor when predicting the next word under the
        language model.

        Parameters
        ----------
        N : int
            The gram-size of the model to calculate perplexity with.
        words : list or tuple of strings
            The sequence of words to compute perplexity on.

        Returns
        -------
        perplexity : float
            The model perlexity for the words in `words`.
        """
        # 返回给定单词序列的交叉熵的指数值，即模型困惑度
        return np.exp(self.cross_entropy(words, N))
    # 计算模型在一系列单词上的交叉熵，与样本中单词的经验分布进行比较
    def cross_entropy(self, words, N):
        r"""
        Calculate the model cross-entropy on a sequence of words against the
        empirical distribution of words in a sample.

        Notes
        -----
        Model cross-entropy, `H`, is defined as

        .. math::

            H(W) = -\frac{\log p(W)}{n}

        where :math:`W = [w_1, \ldots, w_k]` is a sequence of words, and `n` is
        the number of `N`-grams in `W`.

        The model cross-entropy is proportional (not equal, since we use base
        `e`) to the average number of bits necessary to encode `W` under the
        model distribution.

        Parameters
        ----------
        N : int
            The gram-size of the model to calculate cross-entropy on.
        words : list or tuple of strings
            The sequence of words to compute cross-entropy on.

        Returns
        -------
        H : float
            The model cross-entropy for the words in `words`.
        """
        # 计算 n-gram 的数量
        n_ngrams = len(ngrams(words, N))
        # 返回交叉熵结果
        return -(1 / n_ngrams) * self.log_prob(words, N)

    # 计算序列单词在 N-gram 模型下的对数概率
    def _log_prob(self, words, N):
        """
        Calculate the log probability of a sequence of words under the
        `N`-gram model
        """
        # 检查是否有 N-gram 的计数
        assert N in self.counts, "You do not have counts for {}-grams".format(N)

        # 如果单词数量不足以形成 N-gram，则引发异常
        if N > len(words):
            err = "Not enough words for a gram-size of {}: {}".format(N, len(words))
            raise ValueError(err)

        # 初始化总概率
        total_prob = 0
        # 遍历所有 N-gram，计算对数概率并累加
        for ngram in ngrams(words, N):
            total_prob += self._log_ngram_prob(ngram)
        return total_prob
    # 返回在未平滑的 N 元语言模型下，可以跟随序列 `words` 的唯一单词标记的数量
    def _n_completions(self, words, N):
        # 检查是否存在 N 元组的计数
        assert N in self.counts, "You do not have counts for {}-grams".format(N)
        # 检查是否需要大于 N-2 个单词才能使用 N 元组
        assert len(words) <= N - 1, "Need > {} words to use {}-grams".format(N - 2, N)

        # 如果输入的单词是列表，则转换为元组
        if isinstance(words, list):
            words = tuple(words)

        # 获取基础单词序列
        base = words[-N + 1 :]
        # 返回在基础单词序列之后出现的唯一单词标记的数量
        return len([k[-1] for k in self.counts[N].keys() if k[:-1] == base])

    # 返回出现次数为 `C` 的唯一 `N` 元组标记的数量
    def _num_grams_with_count(self, C, N):
        # 确保出现次数大于 0
        assert C > 0
        # 检查是否存在 N 元组的计数
        assert N in self.counts, "You do not have counts for {}-grams".format(N)
        # 为将来的调用缓存计数值
        if not hasattr(self, "_NC"):
            self._NC = {N: {} for N in range(1, self.N + 1)}
        # 如果计数值不在缓存中，则计算并存储
        if C not in self._NC[N]:
            self._NC[N][C] = len([k for k, v in self.counts[N].items() if v == C])
        return self._NC[N][C]

    @abstractmethod
    # 计算在未平滑、最大似然的 `N` 元语言模型下，单词序列的对数概率
    def log_prob(self, words, N):
        raise NotImplementedError

    @abstractmethod
    # 返回 `ngram` 的未平滑对数概率
    def _log_ngram_prob(self, ngram):
        raise NotImplementedError
class MLENGram(NGramBase):
    # MLENGram 类继承自 NGramBase 类，表示一个简单的未平滑的 N-gram 语言模型

    def __init__(self, N, unk=True, filter_stopwords=True, filter_punctuation=True):
        """
        A simple, unsmoothed N-gram model.

        Parameters
        ----------
        N : int
            The maximum length (in words) of the context-window to use in the
            langauge model. Model will compute all n-grams from 1, ..., N.
        unk : bool
            Whether to include the ``<unk>`` (unknown) token in the LM. Default
            is True.
        filter_stopwords : bool
            Whether to remove stopwords before training. Default is True.
        filter_punctuation : bool
            Whether to remove punctuation before training. Default is True.
        """
        # 初始化函数，设置模型的参数

        super().__init__(N, unk, filter_stopwords, filter_punctuation)

        # 设置模型的超参数
        self.hyperparameters["id"] = "MLENGram"

    def log_prob(self, words, N):
        """
        Compute the log probability of a sequence of words under the
        unsmoothed, maximum-likelihood `N`-gram language model.

        Parameters
        ----------
        words : list of strings
            A sequence of words
        N : int
            The gram-size of the language model to use when calculating the log
            probabilities of the sequence

        Returns
        -------
        total_prob : float
            The total log-probability of the sequence `words` under the
            `N`-gram language model
        """
        # 计算给定序列在未平滑的最大似然 N-gram 语言模型下的对数概率
        return self._log_prob(words, N)

    def _log_ngram_prob(self, ngram):
        """Return the unsmoothed log probability of the ngram"""
        # 返回 ngram 的未平滑对数概率
        N = len(ngram)
        num = self.counts[N][ngram]
        den = self.counts[N - 1][ngram[:-1]] if N > 1 else self.n_words[1]
        return np.log(num) - np.log(den) if (den > 0 and num > 0) else -np.inf


class AdditiveNGram(NGramBase):
    def __init__(
        self, N, K=1, unk=True, filter_stopwords=True, filter_punctuation=True,
    ):
        """
        An N-Gram model with smoothed probabilities calculated via additive /
        Lidstone smoothing.

        Notes
        -----
        The resulting estimates correspond to the expected value of the
        posterior, `p(ngram_prob | counts)`, when using a symmetric Dirichlet
        prior on counts with parameter `K`.

        Parameters
        ----------
        N : int
            The maximum length (in words) of the context-window to use in the
            langauge model. Model will compute all n-grams from 1, ..., N
        K : float
            The pseudocount to add to each observation. Larger values allocate
            more probability toward unseen events. When `K` = 1, the model is
            known as Laplace smoothing.  When `K` = 0.5, the model is known as
            expected likelihood estimation (ELE) or the Jeffreys-Perks law.
            Default is 1.
        unk : bool
            Whether to include the ``<unk>`` (unknown) token in the LM. Default
            is True.
        filter_stopwords : bool
            Whether to remove stopwords before training. Default is True.
        filter_punctuation : bool
            Whether to remove punctuation before training. Default is True.
        """
        # 调用父类的初始化方法，传入参数 N, unk, filter_stopwords, filter_punctuation
        super().__init__(N, unk, filter_stopwords, filter_punctuation)

        # 设置模型的超参数 id 为 "AdditiveNGram"
        self.hyperparameters["id"] = "AdditiveNGram"
        # 设置模型的超参数 K 为传入的参数 K
        self.hyperparameters["K"] = K
    def log_prob(self, words, N):
        r"""
        Compute the smoothed log probability of a sequence of words under the
        `N`-gram language model with additive smoothing.

        Notes
        -----
        For a bigram, additive smoothing amounts to:

        .. math::

            P(w_i \mid w_{i-1}) = \frac{A + K}{B + KV}

        where

        .. math::

            A  &=  \text{Count}(w_{i-1}, w_i) \\
            B  &=  \sum_j \text{Count}(w_{i-1}, w_j) \\
            V  &= |\{ w_j \ : \ \text{Count}(w_{i-1}, w_j) > 0 \}|

        This is equivalent to pretending we've seen every possible `N`-gram
        sequence at least `K` times.

        Additive smoothing can be problematic, as it:
            - Treats each predicted word in the same way
            - Can assign too much probability mass to unseen `N`-grams

        Parameters
        ----------
        words : list of strings
            A sequence of words.
        N : int
            The gram-size of the language model to use when calculating the log
            probabilities of the sequence.

        Returns
        -------
        total_prob : float
            The total log-probability of the sequence `words` under the
            `N`-gram language model.
        """
        # 调用内部方法计算给定序列的平滑对数概率
        return self._log_prob(words, N)

    def _log_ngram_prob(self, ngram):
        """Return the smoothed log probability of the ngram"""
        # 获取 ngram 的长度
        N = len(ngram)
        # 获取超参数 K
        K = self.hyperparameters["K"]
        # 获取各种计数和词汇量
        counts, n_words, n_tokens = self.counts, self.n_words[1], self.n_tokens[1]

        # 获取 ngram 的上下文
        ctx = ngram[:-1]
        # 计算分子
        num = counts[N][ngram] + K
        # 计算上下文的计数
        ctx_count = counts[N - 1][ctx] if N > 1 else n_words
        # 计算分母
        den = ctx_count + K * n_tokens
        # 返回平滑后的对数概率，如果分母为零则返回负无穷
        return np.log(num / den) if den != 0 else -np.inf
# 定义一个 GoodTuringNGram 类，继承自 NGramBase 类
class GoodTuringNGram(NGramBase):
    # 初始化方法，接受多个参数
    def __init__(
        self, N, conf=1.96, unk=True, filter_stopwords=True, filter_punctuation=True,
    ):
        """
        An N-Gram model with smoothed probabilities calculated with the simple
        Good-Turing estimator from Gale (2001).

        Parameters
        ----------
        N : int
            The maximum length (in words) of the context-window to use in the
            langauge model. Model will compute all n-grams from 1, ..., N.
        conf: float
            The multiplier of the standard deviation of the empirical smoothed
            count (the default, 1.96, corresponds to a 95% confidence
            interval). Controls how many datapoints are smoothed using the
            log-linear model.
        unk : bool
            Whether to include the ``<unk>`` (unknown) token in the LM. Default
            is True.
        filter_stopwords : bool
            Whether to remove stopwords before training. Default is True.
        filter_punctuation : bool
            Whether to remove punctuation before training. Default is True.
        """
        # 调用父类的初始化方法，传入参数 N, unk, filter_stopwords, filter_punctuation
        super().__init__(N, unk, filter_stopwords, filter_punctuation)

        # 设置超参数字典中的 id 键值对为 "GoodTuringNGram"
        self.hyperparameters["id"] = "GoodTuringNGram"
        # 设置超参数字典中的 conf 键值对为传入的 conf 参数值
        self.hyperparameters["conf"] = conf
    # 训练语言模型，统计文本语料库中的 n-gram 计数。完成后，self.counts 属性将存储 N、N-1、...、1-gram 计数的字典。
    def train(self, corpus_fp, vocab=None, encoding=None):
        """
        编译 `corpus_fp` 中文本的 n-gram 计数。完成后，`self.counts` 属性将存储 `N`、`N-1`、...、1-gram 计数的字典。

        Parameters
        ----------
        corpus_fp : str
            新行分隔的文本语料库文件的路径
        vocab : :class:`~numpy_ml.preprocessing.nlp.Vocabulary` 实例或 None。
            如果不是 None，则只使用 `vocab` 中的单词来构建语言模型；所有超出词汇表的单词将被映射到 `<unk>`（如果 `self.unk = True`）或删除（如果 `self.unk = False`）。默认为 None。
        encoding : str  or None
            指定语料库的文本编码。常见的条目有 'utf-8'、'utf-8-sig'、'utf-16'。默认为 None。
        """
        # 调用 _train 方法，训练语言模型
        self._train(corpus_fp, vocab=vocab, encoding=encoding)
        # 计算平滑后的计数
        self._calc_smoothed_counts()
    # 计算在具有 Good-Turing 平滑的 N-gram 语言模型下，给定单词序列的平滑对数概率
    def log_prob(self, words, N):
        r"""
        Compute the smoothed log probability of a sequence of words under the
        `N`-gram language model with Good-Turing smoothing.

        Notes
        -----
        For a bigram, Good-Turing smoothing amounts to:

        .. math::

            P(w_i \mid w_{i-1}) = \frac{C^*}{\text{Count}(w_{i-1})}

        where :math:`C^*` is the Good-Turing smoothed estimate of the bigram
        count:

        .. math::

            C^* = \frac{(c + 1) \text{NumCounts}(c + 1, 2)}{\text{NumCounts}(c, 2)}

        where

        .. math::

            c  &=  \text{Count}(w_{i-1}, w_i) \\
            \text{NumCounts}(r, k)  &=
                |\{ k\text{-gram} : \text{Count}(k\text{-gram}) = r \}|

        In words, the probability of an `N`-gram that occurs `r` times in the
        corpus is estimated by dividing up the probability mass occupied by
        N-grams that occur `r+1` times.

        For large values of `r`, NumCounts becomes unreliable. In this case, we
        compute a smoothed version of NumCounts using a power law function:

        .. math::

            \log \text{NumCounts}(r) = b + a \log r

        Under the Good-Turing estimator, the total probability assigned to
        unseen `N`-grams is equal to the relative occurrence of `N`-grams that
        appear only once.

        Parameters
        ----------
        words : list of strings
            A sequence of words.
        N : int
            The gram-size of the language model to use when calculating the log
            probabilities of the sequence.

        Returns
        -------
        total_prob : float
            The total log-probability of the sequence `words` under the
            `N`-gram language model.
        """
        # 调用内部方法 _log_prob 来计算给定单词序列在 N-gram 语言模型下的总对数概率
        return self._log_prob(words, N)
    # 计算 ngram 的平滑对数概率并返回
    def _log_ngram_prob(self, ngram):
        """Return the smoothed log probability of the ngram"""
        N = len(ngram)
        sc, T = self._smooth_counts[N], self._smooth_totals[N]
        n_tokens, n_seen = self.n_tokens[N], len(self.counts[N])

        # 计算未出现在词汇表中的 ngram 的概率（即 p0 的一部分）
        n_unseen = max((n_tokens ** N) - n_seen, 1)
        prob = np.log(self._p0[N] / n_unseen)

        # 如果 ngram 在计数中存在，则重新计算概率
        if ngram in self.counts[N]:
            C = self.counts[N][ngram]
            prob = np.log(1 - self._p0[N]) + np.log(sc[C]) - np.log(T)
        return prob

    # 拟合计数模型
    def _fit_count_models(self):
        """
        Perform the averaging transform proposed by Church and Gale (1991):
        estimate the expected count-of-counts by the *density* of
        count-of-count values.
        """
        self._count_models = {}
        NC = self._num_grams_with_count
        for N in range(1, self.N + 1):
            X, Y = [], []
            sorted_counts = sorted(set(self.counts[N].values()))  # r

            # 计算平均转换后的值
            for ix, j in enumerate(sorted_counts):
                i = 0 if ix == 0 else sorted_counts[ix - 1]
                k = 2 * j - i if ix == len(sorted_counts) - 1 else sorted_counts[ix + 1]
                y = 2 * NC(j, N) / (k - i)
                X.append(j)
                Y.append(y)

            # 拟合对数线性模型：log(counts) ~ log(average_transform(counts))
            self._count_models[N] = LinearRegression(fit_intercept=True)
            self._count_models[N].fit(np.log(X), np.log(Y))
            b, a = self._count_models[N].beta

            # 如果斜率大于 -1，则输出警告
            if a > -1:
                fstr = "[Warning] Log-log averaging transform has slope > -1 for N={}"
                print(fstr.format(N))
```