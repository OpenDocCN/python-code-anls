# `numpy-ml\numpy_ml\neural_nets\models\w2v.py`

```
# 从 time 模块中导入 time 函数
from time import time

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 从上一级目录中导入 layers 模块中的 Embedding 类
from ..layers import Embedding

# 从上一级目录中导入 losses 模块中的 NCELoss 类
from ..losses import NCELoss

# 从 preprocessing.nlp 模块中导入 Vocabulary 和 tokenize_words 函数
from ...preprocessing.nlp import Vocabulary, tokenize_words

# 从 utils.data_structures 模块中导入 DiscreteSampler 类
from ...utils.data_structures import DiscreteSampler

# 定义 Word2Vec 类
class Word2Vec(object):
    # 初始化方法
    def __init__(
        self,
        context_len=5,
        min_count=None,
        skip_gram=False,
        max_tokens=None,
        embedding_dim=300,
        filter_stopwords=True,
        noise_dist_power=0.75,
        init="glorot_uniform",
        num_negative_samples=64,
        optimizer="SGD(lr=0.1)",
    # 初始化参数方法
    def _init_params(self):
        # 初始化词典
        self._dv = {}
        # 构建噪声分布
        self._build_noise_distribution()

        # 初始化 Embedding 层
        self.embeddings = Embedding(
            init=self.init,
            vocab_size=self.vocab_size,
            n_out=self.embedding_dim,
            optimizer=self.optimizer,
            pool=None if self.skip_gram else "mean",
        )

        # 初始化 NCELoss 损失函数
        self.loss = NCELoss(
            init=self.init,
            optimizer=self.optimizer,
            n_classes=self.vocab_size,
            subtract_log_label_prob=False,
            noise_sampler=self._noise_sampler,
            num_negative_samples=self.num_negative_samples,
        )

    # 参数属性，返回模型参数
    @property
    def parameters(self):
        """Model parameters"""
        param = {"components": {"embeddings": {}, "loss": {}}}
        if hasattr(self, "embeddings"):
            param["components"] = {
                "embeddings": self.embeddings.parameters,
                "loss": self.loss.parameters,
            }
        return param

    @property
    # 返回模型的超参数，包括模型结构、初始化方式、优化器、最大标记数、上下文长度、嵌入维度、噪声分布幂、是否过滤停用词、负采样数、词汇表大小等信息
    def hyperparameters(self):
        """Model hyperparameters"""
        hp = {
            "layer": "Word2Vec",
            "init": self.init,
            "skip_gram": self.skip_gram,
            "optimizer": self.optimizer,
            "max_tokens": self.max_tokens,
            "context_len": self.context_len,
            "embedding_dim": self.embedding_dim,
            "noise_dist_power": self.noise_dist_power,
            "filter_stopwords": self.filter_stopwords,
            "num_negative_samples": self.num_negative_samples,
            "vocab_size": self.vocab_size if hasattr(self, "vocab_size") else None,
            "components": {"embeddings": {}, "loss": {}},
        }

        # 如果模型包含嵌入层和损失函数，则更新超参数信息
        if hasattr(self, "embeddings"):
            hp["components"] = {
                "embeddings": self.embeddings.hyperparameters,
                "loss": self.loss.hyperparameters,
            }
        return hp

    # 返回模型操作期间计算的变量，包括嵌入层和损失函数的派生变量
    @property
    def derived_variables(self):
        """Variables computed during model operation"""
        dv = {"components": {"embeddings": {}, "loss": {}}}
        dv.update(self._dv)

        # 如果模型包含嵌入层和损失函数，则更新派生变量信息
        if hasattr(self, "embeddings"):
            dv["components"] = {
                "embeddings": self.embeddings.derived_variables,
                "loss": self.loss.derived_variables,
            }
        return dv

    # 返回模型参数的梯度信息，包括嵌入层和损失函数的梯度
    @property
    def gradients(self):
        """Model parameter gradients"""
        grad = {"components": {"embeddings": {}, "loss": {}}}
        if hasattr(self, "embeddings"):
            grad["components"] = {
                "embeddings": self.embeddings.gradients,
                "loss": self.loss.gradients,
            }
        return grad
    def forward(self, X, targets, retain_derived=True):
        """
        Evaluate the network on a single minibatch.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Layer input, representing a minibatch of `n_ex` examples, each
            consisting of `n_in` integer word indices
        targets : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`
            Target word index for each example in the minibatch.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If `False`, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            True.

        Returns
        -------
        loss : float
            The loss associated with the current minibatch
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`
            The conditional probabilities of the words in `targets` given the
            corresponding example / context in `X`.
        """
        # 计算输入层的嵌入表示
        X_emb = self.embeddings.forward(X, retain_derived=True)
        # 计算损失和预测值
        loss, y_pred = self.loss.loss(X_emb, targets.flatten(), retain_derived=True)
        return loss, y_pred

    def backward(self):
        """
        Compute the gradient of the loss wrt the current network parameters.
        """
        # 计算损失相对于当前网络参数的梯度
        dX_emb = self.loss.grad(retain_grads=True, update_params=False)
        # 反向传播梯度到嵌入层
        self.embeddings.backward(dX_emb)

    def update(self, cur_loss=None):
        """Perform gradient updates"""
        # 更新损失
        self.loss.update(cur_loss)
        # 更新嵌入层
        self.embeddings.update(cur_loss)
        # 清空梯度
        self.flush_gradients()

    def flush_gradients(self):
        """Reset parameter gradients after update"""
        # 清空损失的梯度
        self.loss.flush_gradients()
        # 清空嵌入层的梯度
        self.embeddings.flush_gradients()
    def get_embedding(self, word_ids):
        """
        Retrieve the embeddings for a collection of word IDs.

        Parameters
        ----------
        word_ids : :py:class:`ndarray <numpy.ndarray>` of shape `(M,)`
            An array of word IDs to retrieve embeddings for.

        Returns
        -------
        embeddings : :py:class:`ndarray <numpy.ndarray>` of shape `(M, n_out)`
            The embedding vectors for each of the `M` word IDs.
        """
        # 如果输入的word_ids是列表，则转换为NumPy数组
        if isinstance(word_ids, list):
            word_ids = np.array(word_ids)
        # 调用embeddings对象的lookup方法来获取word_ids对应的嵌入向量
        return self.embeddings.lookup(word_ids)

    def _build_noise_distribution(self):
        """
        Construct the noise distribution for use during negative sampling.

        For a word ``w`` in the corpus, the noise distribution is::

            P_n(w) = Count(w) ** noise_dist_power / Z

        where ``Z`` is a normalizing constant, and `noise_dist_power` is a
        hyperparameter of the model. Mikolov et al. report best performance
        using a `noise_dist_power` of 0.75.
        """
        # 检查是否已经存在vocab属性，如果不存在则抛出异常
        if not hasattr(self, "vocab"):
            raise ValueError("Must call `fit` before constructing noise distribution")

        # 初始化一个全零数组来存储噪声分布的概率
        probs = np.zeros(len(self.vocab))
        # 获取噪声分布的幂指数
        power = self.hyperparameters["noise_dist_power"]

        # 遍历词汇表中的每个词，计算其出现次数的幂指数作为噪声分布的概率
        for ix, token in enumerate(self.vocab):
            count = token.count
            probs[ix] = count ** power

        # 对概率进行归一化处理
        probs /= np.sum(probs)
        # 使用DiscreteSampler类来构建噪声分布的采样器
        self._noise_sampler = DiscreteSampler(probs, log=False, with_replacement=False)
    # 训练一个 epoch 的数据集
    def _train_epoch(self, corpus_fps, encoding):
        # 初始化总损失
        total_loss = 0
        # 获取一个批次的数据生成器
        batch_generator = self.minibatcher(corpus_fps, encoding)
        # 遍历每个批次的数据
        for ix, (X, target) in enumerate(batch_generator):
            # 计算当前批次的损失
            loss = self._train_batch(X, target)
            # 累加总损失
            total_loss += loss
            # 如果设置了 verbose，则输出当前批次的损失
            if self.verbose:
                # 计算平滑损失
                smooth_loss = 0.99 * smooth_loss + 0.01 * loss if ix > 0 else loss
                fstr = "[Batch {}] Loss: {:.5f} | Smoothed Loss: {:.5f}"
                print(fstr.format(ix + 1, loss, smooth_loss))
        # 返回平均损失
        return total_loss / (ix + 1)

    # 训练一个批次的数据
    def _train_batch(self, X, target):
        # 前向传播计算损失
        loss, _ = self.forward(X, target)
        # 反向传播
        self.backward()
        # 更新参数
        self.update(loss)
        # 返回当前批次的损失
        return loss

    # 拟合模型
    def fit(
        self, corpus_fps, encoding="utf-8-sig", n_epochs=20, batchsize=128, verbose=True
        ):
        """
        Learn word2vec embeddings for the examples in `X_train`.

        Parameters
        ----------
        corpus_fps : str or list of strs
            The filepath / list of filepaths to the document(s) to be encoded.
            Each document is expected to be encoded as newline-separated
            string of text, with adjacent tokens separated by a whitespace
            character.
        encoding : str
            Specifies the text encoding for corpus. Common entries are either
            'utf-8' (no header byte), or 'utf-8-sig' (header byte).  Default
            value is 'utf-8-sig'.
        n_epochs : int
            The maximum number of training epochs to run. Default is 20.
        batchsize : int
            The desired number of examples in each training batch. Default is
            128.
        verbose : bool
            Print batch information during training. Default is True.
        """
        # 设置是否打印训练信息
        self.verbose = verbose
        # 设置最大训练轮数
        self.n_epochs = n_epochs
        # 设置每个训练批次的样本数量
        self.batchsize = batchsize

        # 初始化词汇表对象
        self.vocab = Vocabulary(
            lowercase=True,
            min_count=self.min_count,
            max_tokens=self.max_tokens,
            filter_stopwords=self.filter_stopwords,
        )
        # 根据语料库文件路径和编码方式构建词汇表
        self.vocab.fit(corpus_fps, encoding=encoding)
        # 获取词汇表大小
        self.vocab_size = len(self.vocab)

        # 在训练模型时忽略特殊字符
        for sp in self.special_chars:
            self.vocab.counts[sp] = 0

        # 初始化词嵌入参数
        self._init_params()

        # 初始化上一次损失值
        prev_loss = np.inf
        # 开始训练
        for i in range(n_epochs):
            # 初始化损失值和时间
            loss, estart = 0.0, time()
            # 训练一个轮次，计算损失值
            loss = self._train_epoch(corpus_fps, encoding)

            # 打印每轮训练的平均损失值和时间
            fstr = "[Epoch {}] Avg. loss: {:.3f}  Delta: {:.3f} ({:.2f}m/epoch)"
            print(fstr.format(i + 1, loss, prev_loss - loss, (time() - estart) / 60.0))
            prev_loss = loss
```