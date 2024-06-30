# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\_hash.py`

```
# 从 itertools 模块导入 chain 函数，用于将多个可迭代对象连接成一个迭代器
# 从 numbers 模块导入 Integral 类型，用于检查对象是否为整数类型
# 导入 numpy 库并重命名为 np，用于处理数值计算
# 导入 scipy.sparse 库并重命名为 sp，用于稀疏矩阵操作
from itertools import chain
from numbers import Integral

import numpy as np
import scipy.sparse as sp

# 从 ..base 模块中导入 BaseEstimator、TransformerMixin 和 _fit_context
# 从 ..utils._param_validation 模块中导入 Interval 和 StrOptions
# 从 ._hashing_fast 模块中导入 transform 函数并重命名为 _hashing_transform
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils._param_validation import Interval, StrOptions
from ._hashing_fast import transform as _hashing_transform


def _iteritems(d):
    """Like d.iteritems, but accepts any collections.Mapping."""
    # 如果对象 d 具有 iteritems 方法，则调用 d.iteritems()，否则调用 d.items()
    return d.iteritems() if hasattr(d, "iteritems") else d.items()


class FeatureHasher(TransformerMixin, BaseEstimator):
    """Implements feature hashing, aka the hashing trick.

    This class turns sequences of symbolic feature names (strings) into
    scipy.sparse matrices, using a hash function to compute the matrix column
    corresponding to a name. The hash function employed is the signed 32-bit
    version of Murmurhash3.

    Feature names of type byte string are used as-is. Unicode strings are
    converted to UTF-8 first, but no Unicode normalization is done.
    Feature values must be (finite) numbers.

    This class is a low-memory alternative to DictVectorizer and
    CountVectorizer, intended for large-scale (online) learning and situations
    where memory is tight, e.g. when running prediction code on embedded
    devices.

    For an efficiency comparison of the different feature extractors, see
    :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

    Read more in the :ref:`User Guide <feature_hashing>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_features : int, default=2**20
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.
    input_type : str, default='dict'
        Choose a string from {'dict', 'pair', 'string'}.
        Either "dict" (the default) to accept dictionaries over
        (feature_name, value); "pair" to accept pairs of (feature_name, value);
        or "string" to accept single strings.
        feature_name should be a string, while value should be a number.
        In the case of "string", a value of 1 is implied.
        The feature_name is hashed to find the appropriate column for the
        feature. The value's sign might be flipped in the output (but see
        non_negative, below).
    dtype : numpy dtype, default=np.float64
        The type of feature values. Passed to scipy.sparse matrix constructors
        as the dtype argument. Do not set this to bool, np.boolean or any
        unsigned integer type.
    """
    # 类 FeatureHasher 继承了 TransformerMixin 和 BaseEstimator 类

    def __init__(self, n_features=2**20, input_type='dict', dtype=np.float64):
        self.n_features = n_features
        self.input_type = input_type
        self.dtype = dtype

    def fit(self, X, y=None):
        """No-op.

        This method does nothing and is only present for compatibility
        with the scikit-learn transformer API.
        """
        return self

    def transform(self, raw_X):
        """Transform raw_X into a sparse matrix.

        Parameters
        ----------
        raw_X : iterable
            An iterable (e.g. a list or generator) containing input data to
            be transformed.

        Returns
        -------
        X : scipy.sparse matrix, shape (n_samples, self.n_features)
            Sparse matrix representing the transformed input.
        """
        # 初始化一个稀疏矩阵的列表
        indices = []
        # 初始化一个值的列表
        values = []
        # 对 raw_X 中的每个元素进行迭代
        for x in raw_X:
            # 根据输入类型选择不同的处理方式
            if self.input_type == 'dict':
                # 如果输入类型是字典，则直接使用 _iteritems 函数遍历字典的键值对
                x_iter = _iteritems(x)
            elif self.input_type == 'pair':
                # 如果输入类型是键值对，则直接使用 x 迭代
                x_iter = iter(x)
            elif self.input_type == 'string':
                # 如果输入类型是字符串，则以 x 和 1 组成的元组迭代
                x_iter = [(x, 1)]
            else:
                raise ValueError("Invalid input_type. Expected one of: "
                                 "'dict', 'pair', 'string'. Got {0}"
                                 .format(self.input_type))
            # 对每个键值对进行处理
            for key, value in x_iter:
                # 确保特征值为有限的数字
                if not isinstance(value, Integral):
                    if not np.isfinite(value):
                        continue
                    # 如果值不是整数，将其转换为浮点数
                    value = self.dtype(value)
                # 使用 Murmurhash3 算法对特征名进行哈希处理，得到列索引
                feature_index = abs(hash(key)) % self.n_features
                # 根据特征索引和特征值添加到稀疏矩阵的列表中
                indices.append(feature_index)
                values.append(value)
        # 根据 indices 和 values 创建稀疏矩阵
        X = sp.csr_matrix((values, indices, [0, len(values)]),
                          shape=(len(raw_X), self.n_features),
                          dtype=self.dtype)
        return X
    # 参数约束字典，定义了 FeatureHasher 类的参数及其取值约束
    _parameter_constraints: dict = {
        "n_features": [Interval(Integral, 1, np.iinfo(np.int32).max, closed="both")],
        "input_type": [StrOptions({"dict", "pair", "string"})],
        "dtype": "no_validation",  # 委托给 numpy 进行数据类型验证
        "alternate_sign": ["boolean"],
    }

    # FeatureHasher 类的构造函数，初始化对象实例时调用
    def __init__(
        self,
        n_features=(2**20),  # 特征哈希后的特征数，默认为 2^20
        *,
        input_type="dict",   # 输入类型，默认为字典
        dtype=np.float64,    # 数组数据类型，默认为 np.float64
        alternate_sign=True, # 是否使用交替符号，默认为 True
    ):
        self.dtype = dtype                # 设置数据类型
        self.input_type = input_type      # 设置输入类型
        self.n_features = n_features      # 设置特征数
        self.alternate_sign = alternate_sign  # 设置是否使用交替符号

    # 使用 @_fit_context 装饰器定义的 fit 方法，用于验证参数
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X=None, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FeatureHasher class instance.
        """
        return self
    def transform(self, raw_X):
        """Transform a sequence of instances to a scipy.sparse matrix.

        Parameters
        ----------
        raw_X : iterable over iterable over raw features, length = n_samples
            Samples. Each sample must be iterable an (e.g., a list or tuple)
            containing/generating feature names (and optionally values, see
            the input_type constructor argument) which will be hashed.
            raw_X need not support the len function, so it can be the result
            of a generator; n_samples is determined on the fly.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Feature matrix, for use with estimators or further transformers.
        """
        # 将raw_X转换为迭代器
        raw_X = iter(raw_X)
        
        # 如果输入类型是字典，将raw_X转换为生成(key, value)元组的迭代器
        if self.input_type == "dict":
            raw_X = (_iteritems(d) for d in raw_X)
        
        # 如果输入类型是字符串
        elif self.input_type == "string":
            # 获取第一个样本
            first_raw_X = next(raw_X)
            # 如果第一个样本是字符串，则抛出异常
            if isinstance(first_raw_X, str):
                raise ValueError(
                    "Samples can not be a single string. The input must be an iterable"
                    " over iterables of strings."
                )
            # 将第一个样本与原始迭代器raw_X合并
            raw_X_ = chain([first_raw_X], raw_X)
            # 将每个样本转换为(feature_name, 1)的元组迭代器
            raw_X = (((f, 1) for f in x) for x in raw_X_)

        # 使用_hashing_transform函数将raw_X转换为稀疏矩阵的三元组(indices, indptr, values)
        indices, indptr, values = _hashing_transform(
            raw_X, self.n_features, self.dtype, self.alternate_sign, seed=0
        )
        # 计算样本数
        n_samples = indptr.shape[0] - 1

        # 如果样本数为0，则抛出异常
        if n_samples == 0:
            raise ValueError("Cannot vectorize empty sequence.")

        # 创建稀疏矩阵X，使用csr_matrix存储格式
        X = sp.csr_matrix(
            (values, indices, indptr),
            dtype=self.dtype,
            shape=(n_samples, self.n_features),
        )
        # 对矩阵X进行去重操作，同时排序索引
        X.sum_duplicates()

        # 返回稀疏矩阵X作为转换后的结果
        return X

    # 返回更多标签，主要用于指定输入数据的类型
    def _more_tags(self):
        return {"X_types": [self.input_type]}
```