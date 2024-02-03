# `numpy-ml\numpy_ml\linear_models\naive_bayes.py`

```py
# 导入 numpy 库
import numpy as np

# 定义高斯朴素贝叶斯分类器类
class GaussianNBClassifier:
    # 拟合模型参数，通过最大似然估计
    def fit(self, X, y):
        """
        Fit the model parameters via maximum likelihood.

        Notes
        -----
        The model parameters are stored in the :py:attr:`parameters
        <numpy_ml.linear_models.GaussianNBClassifier.parameters>` attribute.
        The following keys are present:

            "mean": :py:class:`ndarray <numpy.ndarray>` of shape `(K, M)`
                Feature means for each of the `K` label classes
            "sigma": :py:class:`ndarray <numpy.ndarray>` of shape `(K, M)`
                Feature variances for each of the `K` label classes
            "prior": :py:class:`ndarray <numpy.ndarray>` of shape `(K,)`
                Prior probability of each of the `K` label classes, estimated
                empirically from the training data

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`
        y: :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The class label for each of the `N` examples in `X`

        Returns
        -------
        self : :class:`GaussianNBClassifier <numpy_ml.linear_models.GaussianNBClassifier>` instance
        """  # noqa: E501
        
        # 获取模型参数和超参数
        P = self.parameters
        H = self.hyperparameters

        # 获取唯一的类别标签
        self.labels = np.unique(y)

        # 获取类别数量和特征维度
        K = len(self.labels)
        N, M = X.shape

        # 初始化均值、方差和先验概率
        P["mean"] = np.zeros((K, M))
        P["sigma"] = np.zeros((K, M))
        P["prior"] = np.zeros((K,))

        # 遍历每个类别，计算均值、方差和先验概率
        for i, c in enumerate(self.labels):
            X_c = X[y == c, :]

            P["mean"][i, :] = np.mean(X_c, axis=0)
            P["sigma"][i, :] = np.var(X_c, axis=0) + H["eps"]
            P["prior"][i] = X_c.shape[0] / N
        return self
    # 使用训练好的分类器对输入数据集 X 进行预测，返回每个样本的预测类别标签
    def predict(self, X):
        """
        Use the trained classifier to predict the class label for each example
        in **X**.

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`

        Returns
        -------
        labels : :py:class:`ndarray <numpy.ndarray>` of shape `(N)`
            The predicted class labels for each example in `X`
        """
        # 返回每个样本的预测类别标签，通过计算后验概率的对数值并取最大值确定
        return self.labels[self._log_posterior(X).argmax(axis=1)]

    # 计算每个类别的（未归一化的）对数后验概率
    def _log_posterior(self, X):
        r"""
        Compute the (unnormalized) log posterior for each class.

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`

        Returns
        -------
        log_posterior : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            Unnormalized log posterior probability of each class for each
            example in `X`
        """
        # 获取类别数量 K
        K = len(self.labels)
        # 初始化一个全零数组，用于存储每个样本的对数后验概率
        log_posterior = np.zeros((X.shape[0], K))
        # 遍历每个类别，计算每个样本的对数后验概率
        for i in range(K):
            log_posterior[:, i] = self._log_class_posterior(X, i)
        # 返回每个样本的对数后验概率
        return log_posterior
    # 计算给定类别索引下的（未归一化的）对数后验概率
    def _log_class_posterior(self, X, class_idx):
        r"""
        Compute the (unnormalized) log posterior for the label at index
        `class_idx` in :py:attr:`labels <numpy_ml.linear_models.GaussianNBClassifier.labels>`.

        Notes
        -----
        Unnormalized log posterior for example :math:`\mathbf{x}_i` and class
        :math:`c` is::

        .. math::

            \log P(y_i = c \mid \mathbf{x}_i, \theta)
                &\propto \log P(y=c \mid \theta) +
                    \log P(\mathbf{x}_i \mid y_i = c, \theta) \\
                &\propto \log P(y=c \mid \theta)
                    \sum{j=1}^M \log P(x_j \mid y_i = c, \theta)

        In the Gaussian naive Bayes model, the feature likelihood for class
        :math:`c`, :math:`P(\mathbf{x}_i \mid y_i = c, \theta)` is assumed to
        be normally distributed

        .. math::

            \mathbf{x}_i \mid y_i = c, \theta \sim \mathcal{N}(\mu_c, \Sigma_c)

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`
        class_idx : int
            The index of the current class in :py:attr:`labels`

        Returns
        -------
        log_class_posterior : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            Unnormalized log probability of the label at index `class_idx`
            in :py:attr:`labels <numpy_ml.linear_models.GaussianNBClassifier.labels>`
            for each example in `X`
        """  # noqa: E501
        # 获取模型参数
        P = self.parameters
        # 获取当前类别的均值
        mu = P["mean"][class_idx]
        # 获取当前类别的先验概率
        prior = P["prior"][class_idx]
        # 获取当前类别的方差
        sigsq = P["sigma"][class_idx]

        # 计算对数似然 = 对数 X | N(mu, sigsq)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigsq))
        log_likelihood -= 0.5 * np.sum(((X - mu) ** 2) / sigsq, axis=1)
        # 返回对数似然加上对数先验概率，即未归一化的对数后验概率
        return log_likelihood + np.log(prior)
```