# `numpy-ml\numpy_ml\preprocessing\general.py`

```
import json
import hashlib
import warnings

import numpy as np

try:
    from scipy.sparse import csr_matrix

    _SCIPY = True
except ImportError:
    # 如果导入失败，则发出警告并设置_SCIPY为False
    warnings.warn("Scipy not installed. FeatureHasher can only create dense matrices")
    _SCIPY = False


def minibatch(X, batchsize=256, shuffle=True):
    """
    Compute the minibatch indices for a training dataset.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)`
        The dataset to divide into minibatches. Assumes the first dimension
        represents the number of training examples.
    batchsize : int
        The desired size of each minibatch. Note, however, that if ``X.shape[0] %
        batchsize > 0`` then the final batch will contain fewer than batchsize
        entries. Default is 256.
    shuffle : bool
        Whether to shuffle the entries in the dataset before dividing into
        minibatches. Default is True.

    Returns
    -------
    mb_generator : generator
        A generator which yields the indices into `X` for each batch.
    n_batches: int
        The number of batches.
    """
    # 获取数据集的样本数量
    N = X.shape[0]
    # 创建包含样本索引的数组
    ix = np.arange(N)
    # 计算总共的批次数量
    n_batches = int(np.ceil(N / batchsize))

    # 如果需要打乱数据集
    if shuffle:
        np.random.shuffle(ix)

    # 定义生成器函数，用于生成每个批次的索引
    def mb_generator():
        for i in range(n_batches):
            yield ix[i * batchsize : (i + 1) * batchsize]

    return mb_generator(), n_batches


class OneHotEncoder:
    def __init__(self):
        """
        Convert between category labels and their one-hot vector
        representations.

        Parameters
        ----------
        categories : list of length `C`
            List of the unique category labels for the items to encode.
        """
        # 初始化OneHotEncoder类的属性
        self._is_fit = False
        self.hyperparameters = {}
        self.parameters = {"categories": None}

    def __call__(self, labels):
        # 调用transform方法
        return self.transform(labels)
    def fit(self, categories):
        """
        Create mappings between columns and category labels.

        Parameters
        ----------
        categories : list of length `C`
            List of the unique category labels for the items to encode.
        """
        # 将传入的类别标签列表存储在参数字典中
        self.parameters["categories"] = categories
        # 创建类别标签到索引的映射字典
        self.cat2idx = {c: i for i, c in enumerate(categories)}
        # 创建索引到类别标签的映射字典
        self.idx2cat = {i: c for i, c in enumerate(categories)}
        # 标记已经进行了fit操作
        self._is_fit = True

    def transform(self, labels, categories=None):
        """
        Convert a list of labels into a one-hot encoding.

        Parameters
        ----------
        labels : list of length `N`
            A list of category labels.
        categories : list of length `C`
            List of the unique category labels for the items to encode. Default
            is None.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            The one-hot encoded labels. Each row corresponds to an example,
            with a single 1 in the column corresponding to the respective
            label.
        """
        # 如果还没有进行fit操作，则根据传入的标签列表进行fit操作
        if not self._is_fit:
            categories = set(labels) if categories is None else categories
            self.fit(categories)

        # 检查是否有未识别的标签
        unknown = list(set(labels) - set(self.cat2idx.keys()))
        assert len(unknown) == 0, "Unrecognized label(s): {}".format(unknown)

        # 获取标签列表的长度和类别标签的数量
        N, C = len(labels), len(self.cat2idx)
        # 将标签列表转换为对应的索引列表
        cols = np.array([self.cat2idx[c] for c in labels])

        # 创建一个全零矩阵，用于存储one-hot编码后的标签
        Y = np.zeros((N, C))
        # 在每行中对应标签的位置设置为1
        Y[np.arange(N), cols] = 1
        return Y
    # 将一个独热编码转换回对应的标签

    def inverse_transform(self, Y):
        """
        Convert a one-hot encoding back into the corresponding labels

        Parameters
        ----------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            One-hot encoded labels. Each row corresponds to an example, with a
            single 1 in the column associated with the label for that example

        Returns
        -------
        labels : list of length `N`
            The list of category labels corresponding to the nonzero columns in
            `Y`
        """
        # 获取类别数量
        C = len(self.cat2idx)
        # 检查 Y 的维度是否为 2
        assert Y.ndim == 2, "Y must be 2D, but has shape {}".format(Y.shape)
        # 检查 Y 的列数是否与类别数量相等
        assert Y.shape[1] == C, "Y must have {} columns, got {}".format(C, Y.shape[1])
        # 返回 Y 中非零元素对应的类别标签
        return [self.idx2cat[ix] for ix in Y.nonzero()[1]]
class Standardizer:
    def __init__(self, with_mean=True, with_std=True):
        """
        Feature-wise standardization for vector inputs.

        Notes
        -----
        Due to the sensitivity of empirical mean and standard deviation
        calculations to extreme values, `Standardizer` cannot guarantee
        balanced feature scales in the presence of outliers. In particular,
        note that because outliers for each feature can have different
        magnitudes, the spread of the transformed data on each feature can be
        very different.

        Similar to sklearn, `Standardizer` uses a biased estimator for the
        standard deviation: ``numpy.std(x, ddof=0)``.

        Parameters
        ----------
        with_mean : bool
            Whether to scale samples to have 0 mean during transformation.
            Default is True.
        with_std : bool
            Whether to scale samples to have unit variance during
            transformation. Default is True.
        """
        # 初始化 Standardizer 类，设置是否计算均值和标准差
        self.with_mean = with_mean
        self.with_std = with_std
        self._is_fit = False

    @property
    def hyperparameters(self):
        # 返回超参数字典，包括是否计算均值和标准差
        H = {"with_mean": self.with_mean, "with_std": self.with_std}
        return H

    @property
    def parameters(self):
        # 返回参数字典，包括均值和标准差（如果已计算）
        params = {
            "mean": self._mean if hasattr(self, "mean") else None,
            "std": self._std if hasattr(self, "std") else None,
        }
        return params

    def __call__(self, X):
        # 调用 transform 方法对输入数据进行标准化
        return self.transform(X)
    def fit(self, X):
        """
        Store the feature-wise mean and standard deviation across the samples
        in `X` for future scaling.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            An array of N samples, each with dimensionality `C`
        """
        # 如果输入不是 numpy 数组，则将其转换为 numpy 数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # 检查样本数量是否至少为 2
        if X.shape[0] < 2:
            raise ValueError("`X` must contain at least 2 samples")

        # 初始化标准差为 1，均值为 0
        std = np.ones(X.shape[1])
        mean = np.zeros(X.shape[1])

        # 如果需要计算均值，则计算样本的均值
        if self.with_mean:
            mean = np.mean(X, axis=0)

        # 如果需要计算标准差，则计算样本的标准差
        if self.with_std:
            std = np.std(X, axis=0, ddof=0)

        # 存储计算得到的均值和标准差
        self._mean = mean
        self._std = std
        self._is_fit = True

    def transform(self, X):
        """
        Standardize features by removing the mean and scaling to unit variance.

        For a sample `x`, the standardized score is calculated as:

        .. math::

            z = (x - u) / s

        where `u` is the mean of the training samples or zero if `with_mean` is
        False, and `s` is the standard deviation of the training samples or 1
        if `with_std` is False.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            An array of N samples, each with dimensionality `C`.

        Returns
        -------
        Z : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            The feature-wise standardized version of `X`.
        """
        # 如果没有进行拟合，则抛出异常
        if not self._is_fit:
            raise Exception("Must call `fit` before using the `transform` method")
        # 返回标准化后的结果
        return (X - self._mean) / self._std
    # 将标准化后的特征集合转换回原始特征空间
    def inverse_transform(self, Z):
        """
        Convert a collection of standardized features back into the original
        feature space.

        For a standardized sample `z`, the unstandardized score is calculated as:

        .. math::

            x = z s + u

        where `u` is the mean of the training samples or zero if `with_mean` is
        False, and `s` is the standard deviation of the training samples or 1
        if `with_std` is False.

        Parameters
        ----------
        Z : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            An array of `N` standardized samples, each with dimensionality `C`.

        Returns
        -------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            The unstandardized samples from `Z`.
        """
        # 检查是否已经拟合了`Standardizer`，如果没有则抛出异常
        assert self._is_fit, "Must fit `Standardizer` before calling inverse_transform"
        # 获取参数字典
        P = self.parameters
        # 获取平均值和标准差
        mean, std = P["mean"], P["std"]
        # 返回原始特征空间中的未标准化样本
        return Z * std + mean
# 定义一个特征哈希器类
class FeatureHasher:
    # 初始化方法，设置特征维度和稀疏性
    def __init__(self, n_dim=256, sparse=True):
        """
        Convert a collection of features to a fixed-dimensional matrix using
        the hashing trick.

        Notes
        -----
        Uses the md5 hash.

        Parameters
        ----------
        n_dim : int
            The dimensionality of each example in the output feature matrix.
            Small numbers of features are likely to cause hash collisions, but
            large numbers will cause larger overall parameter dimensions for
            any (linear) learning agent. Default is 256.
        sparse : bool
            Whether the resulting feature matrix should be a sparse
            :py:class:`csr_matrix <scipy.sparse.csr_matrix>` or dense
            :py:class:`ndarray <numpy.ndarray>`. Default is True.
        """
        # 设置特征维度
        self.n_dim = n_dim
        # 使用 md5 哈希算法
        self.hash = hashlib.md5
        # 设置是否使用稀疏矩阵
        self.sparse = sparse and _SCIPY
    # 将一组多特征示例编码为一个 `n_dim` 维特征矩阵，通过特征哈希

    # 特征哈希通过将哈希函数应用于示例的特征，并使用哈希值作为结果特征矩阵中的列索引来工作。
    # 每个哈希特征列上的条目对应于该示例和特征的值。例如，给定以下两个输入示例：

    # 定义示例
    >>> examples = [
        {"furry": 1, "quadruped": 1, "domesticated": 1},
        {"nocturnal": 1, "quadruped": 1},
    ]

    # 定义特征矩阵
    >>> feature_mat = zeros(2, 128)
    >>> ex1_cols = [H("furry"), H("quadruped"), H("domesticated")]
    >>> ex2_cols = [H("nocturnal"), H("quadruped")]
    >>> feat_mat[0, ex1_cols] = 1
    >>> feat_mat[1, ex2_cols] = 1

    # 为了更好地处理哈希冲突，通常将特征值乘以相应特征名称的摘要的符号。

    # 参数：
    # examples : dict or list of dicts
    #     一组 `N` 个示例，每个示例表示为一个字典，其中键对应于特征名称，值对应于特征值。

    # 返回：
    # table : :py:class:`ndarray <numpy.ndarray>` or :py:class:`csr_matrix <scipy.sparse.csr_matrix>` of shape `(N, n_dim)`
    #     编码后的特征矩阵
    def encode(self, examples):
        # 如果示例是字典，则转换为列表
        if isinstance(examples, dict):
            examples = [examples]

        # 根据稀疏性选择编码方式
        sparse = self.sparse
        return self._encode_sparse(examples) if sparse else self._encode_dense(examples)
    # 将稠密特征编码为稀疏矩阵
    def _encode_dense(self, examples):
        # 获取样本数量
        N = len(examples)
        # 创建一个全零矩阵，用于存储稠密特征
        table = np.zeros(N, self.n_dim)  # dense

        # 遍历每个样本
        for row, feat_dict in enumerate(examples):
            # 遍历每个特征及其值
            for f_id, val in feat_dict.items():
                # 如果特征ID是字符串，则转换为UTF-8编码
                if isinstance(f_id, str):
                    f_id = f_id.encode("utf-8")

                # 使用json模块将特征ID转换为与缓冲区API兼容的唯一字符串（哈希算法所需）
                if isinstance(f_id, (tuple, dict, list)):
                    f_id = json.dumps(f_id, sort_keys=True).encode("utf-8")

                # 计算特征ID的哈希值，并取模得到列索引
                h = int(self.hash(f_id).hexdigest(), base=16)
                col = h % self.n_dim
                # 更新稠密特征矩阵的值
                table[row, col] += np.sign(h) * val

        # 返回稠密特征矩阵
        return table

    # 将稀疏特征编码为稀疏矩阵
    def _encode_sparse(self, examples):
        # 获取样本数量
        N = len(examples)
        # 初始化索引和数据列表
        idxs, data = [], []

        # 遍历每个样本
        for row, feat_dict in enumerate(examples):
            # 遍历每个特征及其值
            for f_id, val in feat_dict.items():
                # 如果特征ID是字符串，则转换为UTF-8编码
                if isinstance(f_id, str):
                    f_id = f_id.encode("utf-8")

                # 使用json模块将特征ID转换为与缓冲区API兼容的唯一字符串（哈希算法所需）
                if isinstance(f_id, (tuple, dict, list)):
                    f_id = json.dumps(f_id, sort_keys=True).encode("utf-8")

                # 计算特征ID的哈希值，并取模得到列索引
                h = int(self.hash(f_id).hexdigest(), base=16)
                col = h % self.n_dim
                # 将行索引和列索引添加到索引列表，将值添加到数据列表
                idxs.append((row, col))
                data.append(np.sign(h) * val)

        # 使用稀疏矩阵的构造函数创建稀疏矩阵
        table = csr_matrix((data, zip(*idxs)), shape=(N, self.n_dim))
        # 返回稀疏特征矩阵
        return table
```