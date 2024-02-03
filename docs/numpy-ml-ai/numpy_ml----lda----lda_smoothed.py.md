# `numpy-ml\numpy_ml\lda\lda_smoothed.py`

```
import numpy as np

# 定义 SmoothedLDA 类
class SmoothedLDA(object):
    def __init__(self, T, **kwargs):
        """
        A smoothed LDA model trained using collapsed Gibbs sampling. Generates
        posterior mean estimates for model parameters `phi` and `theta`.

        Parameters
        ----------
        T : int
            Number of topics

        Attributes
        ----------
        D : int
            Number of documents
        N : int
            Total number of words across all documents
        V : int
            Number of unique word tokens across all documents
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(N[d], T)`
            The word-topic distribution
        theta : :py:class:`ndarray <numpy.ndarray>` of shape `(D, T)`
            The document-topic distribution
        alpha : :py:class:`ndarray <numpy.ndarray>` of shape `(1, T)`
            Parameter for the Dirichlet prior on the document-topic distribution
        beta  : :py:class:`ndarray <numpy.ndarray>` of shape `(V, T)`
            Parameter for the Dirichlet prior on the topic-word distribution
        """
        # 初始化 T 属性
        self.T = T

        # 初始化 alpha 属性，如果 kwargs 中包含 alpha，则使用 kwargs 中的值，否则使用默认值
        self.alpha = (50.0 / self.T) * np.ones(self.T)
        if "alpha" in kwargs.keys():
            self.alpha = (kwargs["alpha"]) * np.ones(self.T)

        # 初始化 beta 属性，如果 kwargs 中包含 beta，则使用 kwargs 中的值，否则使用默认值
        self.beta = 0.01
        if "beta" in kwargs.keys():
            self.beta = kwargs["beta"]
    # 初始化模型参数
    def _init_params(self, texts, tokens):
        # 设置模型的 tokens 属性
        self.tokens = tokens
        # 获取文档数量 D
        self.D = len(texts)
        # 获取唯一 tokens 的数量 V
        self.V = len(np.unique(self.tokens))
        # 获取文档中 token 的总数 N
        self.N = np.sum(np.array([len(doc) for doc in texts]))
        # 初始化 word_document 矩阵
        self.word_document = np.zeros(self.N)

        # 根据 tokens 数量设置 beta
        self.beta = self.beta * np.ones(self.V)

        # 初始化计数器
        count = 0
        # 遍历文档和文档中的单词
        for doc_idx, doc in enumerate(texts):
            for word_idx, word in enumerate(doc):
                # 更新 word_idx
                word_idx = word_idx + count
                # 将文档索引存入 word_document 矩阵
                self.word_document[word_idx] = doc_idx
            # 更新计数器
            count = count + len(doc)

    # 训练主题模型
    def train(self, texts, tokens, n_gibbs=2000):
        """
        Trains a topic model on the documents in texts.

        Parameters
        ----------
        texts : array of length `(D,)`
            The training corpus represented as an array of subarrays, where
            each subarray corresponds to the tokenized words of a single
            document.
        tokens : array of length `(V,)`
            The set of unique tokens in the documents in `texts`.
        n_gibbs : int
            The number of steps to run the collapsed Gibbs sampler during
            training. Default is 2000.

        Returns
        -------
        C_wt : :py:class:`ndarray <numpy.ndarray>` of shape (V, T)
            The word-topic count matrix
        C_dt : :py:class:`ndarray <numpy.ndarray>` of shape (D, T)
            The document-topic count matrix
        assignments : :py:class:`ndarray <numpy.ndarray>` of shape (N, n_gibbs)
            The topic assignments for each word in the corpus on each Gibbs
            step.
        """
        # 初始化模型参数
        self._init_params(texts, tokens)
        # 运行 Gibbs 采样器
        C_wt, C_dt, assignments = self._gibbs_sampler(n_gibbs, texts)
        # 拟合参数
        self.fit_params(C_wt, C_dt)
        # 返回结果
        return C_wt, C_dt, assignments
    def what_did_you_learn(self, top_n=10):
        """
        打印每个主题下概率最高的 `top_n` 个单词
        """
        遍历每个主题
        for tt in range(self.T):
            对每个主题的单词分布进行排序，找出概率最高的 `top_n` 个单词的索引
            top_idx = np.argsort(self.phi[:, tt])[::-1][:top_n]
            根据索引获取对应的单词
            top_tokens = self.tokens[top_idx]
            打印当前主题下的概率最高的单词
            print("\nTop Words for Topic %s:\n" % (str(tt)))
            遍历每个概率最高的单词并打印
            for token in top_tokens:
                print("\t%s\n" % (str(token)))

    def fit_params(self, C_wt, C_dt):
        """
        估计 `phi`，即单词-主题分布，和 `theta`，即主题-文档分布。

        参数
        ----------
        C_wt : :py:class:`ndarray <numpy.ndarray>` of shape (V, T)
            单词-主题计数矩阵
        C_dt : :py:class:`ndarray <numpy.ndarray>` of shape (D, T)
            文档-主题计数矩阵

        返回
        -------
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(V, T)`
            单词-主题分布
        theta : :py:class:`ndarray <numpy.ndarray>` of shape `(D, T)`
            主题-文档分布
        """
        初始化单词-主题分布矩阵 phi 和主题-文档分布矩阵 theta
        self.phi = np.zeros([self.V, self.T])
        self.theta = np.zeros([self.D, self.T])

        获取超参数 beta 和 alpha
        b, a = self.beta[0], self.alpha[0]
        遍历每个单词
        for ii in range(self.V):
            遍历每个主题
            for jj in range(self.T):
                计算单词-主题分布 phi
                self.phi[ii, jj] = (C_wt[ii, jj] + b) / (
                    np.sum(C_wt[:, jj]) + self.V * b
                )

        遍历每个文档
        for dd in range(self.D):
            遍历每个主题
            for jj in range(self.T):
                计算主题-文档分布 theta
                self.theta[dd, jj] = (C_dt[dd, jj] + a) / (
                    np.sum(C_dt[dd, :]) + self.T * a
                )
        返回计算得到的 phi 和 theta
        return self.phi, self.theta
    # 估算给定条件下 token ii 被分配到主题 jj 的概率的近似值
    def _estimate_topic_prob(self, ii, d, C_wt, C_dt):
        """
        Compute an approximation of the conditional probability that token ii
        is assigned to topic jj given all previous topic assignments and the
        current document d: p(t_i = j | t_{-i}, w_i, d_i)
        """
        # 初始化概率向量
        p_vec = np.zeros(self.T)
        # 获取 beta 和 alpha 参数
        b, a = self.beta[0], self.alpha[0]
        # 遍历所有主题
        for jj in range(self.T):
            # 计算在主题 jj 下 token ii 的概率
            frac1 = (C_wt[ii, jj] + b) / (np.sum(C_wt[:, jj]) + self.V * b)
            # 计算在文档 d 下主题 jj 的概率
            frac2 = (C_dt[d, jj] + a) / (np.sum(C_dt[d, :]) + self.T * a)
            # 计算 token ii 被分配到主题 jj 的概率
            p_vec[jj] = frac1 * frac2
        # 对概率向量进行归一化处理
        return p_vec / np.sum(p_vec)
    # 定义一个私有方法，使用Collapsed Gibbs采样器估计主题分配的后验分布
    def _gibbs_sampler(self, n_gibbs, texts):
        """
        Collapsed Gibbs sampler for estimating the posterior distribution over
        topic assignments.
        """
        # 初始化计数矩阵
        C_wt = np.zeros([self.V, self.T])
        C_dt = np.zeros([self.D, self.T])
        assignments = np.zeros([self.N, n_gibbs + 1])

        # 为单词随机初始化主题分配
        for ii in range(self.N):
            token_idx = np.concatenate(texts)[ii]
            assignments[ii, 0] = np.random.randint(0, self.T)

            doc = self.word_document[ii]
            C_dt[doc, assignments[ii, 0]] += 1
            C_wt[token_idx, assignments[ii, 0]] += 1

        # 运行Collapsed Gibbs采样器
        for gg in range(n_gibbs):
            print("Gibbs iteration {} of {}".format(gg + 1, n_gibbs))
            for jj in range(self.N):
                token_idx = np.concatenate(texts)[jj]

                # 将计数矩阵减1
                doc = self.word_document[jj]
                C_wt[token_idx, assignments[jj, gg]] -= 1
                C_dt[doc, assignments[jj, gg]] -= 1

                # 从条件分布的近似中抽取新的主题
                p_topics = self._estimate_topic_prob(token_idx, doc, C_wt, C_dt)
                sampled_topic = np.nonzero(np.random.multinomial(1, p_topics))[0][0]

                # 更新计数矩阵
                C_wt[token_idx, sampled_topic] += 1
                C_dt[doc, sampled_topic] += 1
                assignments[jj, gg + 1] = sampled_topic
        return C_wt, C_dt, assignments
```