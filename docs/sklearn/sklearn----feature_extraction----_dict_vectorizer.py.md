# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\_dict_vectorizer.py`

```
# 从 Python 标准库导入所需模块和类
from array import array
from collections.abc import Iterable, Mapping
from numbers import Number
from operator import itemgetter

# 导入第三方科学计算库模块，并使用别名 np 表示
import numpy as np
import scipy.sparse as sp

# 从 scikit-learn 库中导入基础模块和相关工具函数
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array
from ..utils.validation import check_is_fitted

# 定义一个名为 DictVectorizer 的类，继承自 TransformerMixin 和 BaseEstimator
class DictVectorizer(TransformerMixin, BaseEstimator):
    """Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    If a feature value is a sequence or set of strings, this transformer
    will iterate over the values and will count the occurrences of each string
    value.

    However, note that this transformer will only do a binary one-hot encoding
    when feature values are of type string. If categorical features are
    represented as numeric values such as int or iterables of strings, the
    DictVectorizer can be followed by
    :class:`~sklearn.preprocessing.OneHotEncoder` to complete
    binary one-hot encoding.

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    For an efficiency comparison of the different feature extractors, see
    :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

    Read more in the :ref:`User Guide <dict_feature_extraction>`.

    Parameters
    ----------
    dtype : dtype, default=np.float64
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator : str, default="="
        Separator string used when constructing new features for one-hot
        coding.
    sparse : bool, default=True
        Whether transform should produce scipy.sparse matrices.
    sort : bool, default=True
        Whether ``feature_names_`` and ``vocabulary_`` should be
        sorted when fitting.

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.

    feature_names_ : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    See Also
    --------
    FeatureHasher : Performs vectorization using only a hash function.
    sklearn.preprocessing.OrdinalEncoder : Handles nominal/categorical
        features encoded as columns of arbitrary data types.

    Examples
    --------
    """
    # 定义类的初始化方法，接收几个参数并设置默认值
    def __init__(self, dtype=np.float64, separator='=', sparse=True, sort=True):
        # 调用父类的初始化方法，设置默认参数
        super().__init__()
        # 设置属性 dtype，表示特征值的数据类型，默认为 np.float64
        self.dtype = dtype
        # 设置属性 separator，用于构造新特征时的分隔符，默认为 '='
        self.separator = separator
        # 设置属性 sparse，表示是否生成稀疏矩阵，默认为 True
        self.sparse = sparse
        # 设置属性 sort，表示是否在拟合时对特征名称和词汇表进行排序，默认为 True
        self.sort = sort

    # 定义 fit 方法，用于拟合模型
    def fit(self, X, y=None):
        # 调用内部方法 _fit_transform，返回转换后的矩阵和特征名称
        X, feature_names = self._fit_transform(X)
        # 返回自身对象
        return self

    # 定义 transform 方法，用于转换数据
    def transform(self, X):
        # 检查模型是否已经拟合
        check_is_fitted(self, 'vocabulary_')
        # 将输入数据 X 转换为矩阵
        X = self._transform(X)
        # 返回转换后的矩阵
        return X

    # 定义 fit_transform 方法，同时进行拟合和转换
    def fit_transform(self, X, y=None, **fit_params):
        # 调用内部方法 _fit_transform，返回转换后的矩阵和特征名称
        X, self.feature_names_ = self._fit_transform(X)
        # 返回转换后的矩阵
        return X

    # 内部方法 _fit_transform，用于拟合和转换过程的具体实现
    def _fit_transform(self, X):
        # 初始化特征名称列表
        feature_names = []
        # 初始化特征索引计数器
        idx = 0
        # 初始化词汇表字典
        vocabulary = {}
        
        # 遍历输入数据 X 中的每个样本
        for x in X:
            # 检查样本 x 是否为映射类型（类似字典的对象）
            if not isinstance(x, Mapping):
                # 如果不是映射类型，则抛出类型错误
                raise TypeError("Each sample must be a dictionary-like object.")
            
            # 遍历样本 x 中的每个特征名和对应的特征值
            for f, v in x.items():
                # 检查特征值是否为字符串类型
                if isinstance(v, str):
                    # 如果特征值是字符串，则生成新特征名
                    fname = "{}{}{}".format(f, self.separator, v)
                    # 检查新特征名是否已经在词汇表中
                    if fname not in vocabulary:
                        # 如果不在词汇表中，则添加到词汇表，并分配索引值
                        vocabulary[fname] = idx
                        # 增加索引计数器
                        idx += 1
                        # 添加新特征名到特征名称列表
                        feature_names.append(fname)
                # 如果特征值是序列或集合类型的字符串
                elif isinstance(v, Iterable) and all(isinstance(item, str) for item in v):
                    # 对序列或集合中的每个字符串进行处理
                    for item in v:
                        # 生成新特征名
                        fname = "{}{}{}".format(f, self.separator, item)
                        # 检查新特征名是否已经在词汇表中
                        if fname not in vocabulary:
                            # 如果不在词汇表中，则添加到词汇表，并分配索引值
                            vocabulary[fname] = idx
                            # 增加索引计数器
                            idx += 1
                            # 添加新特征名到特征名称列表
                            feature_names.append(fname)
                # 如果特征值是数值类型
                elif isinstance(v, Number):
                    # 生成新特征名，不考虑特征值类型，直接使用数值
                    fname = "{}{}{}".format(f, self.separator, v)
                    # 检查新特征名是否已经在词汇表中
                    if fname not in vocabulary:
                        # 如果不在词汇表中，则添加到词汇表，并分配索引值
                        vocabulary[fname] = idx
                        # 增加索引计数器
                        idx += 1
                        # 添加新特征名到特征名称列表
                        feature_names.append(fname)
                # 如果特征值不符合以上类型，则抛出类型错误
                else:
                    raise ValueError("Unsupported feature type for %s: %s" % (f, type(v)))

        # 根据 sort 参数决定是否对特征名称列表和词汇表进行排序
        if self.sort:
            feature_names.sort()
            # 重新构建词汇表，按照特征名称重新排列
            vocabulary = {fname: idx for idx, fname in enumerate(feature_names)}

        # 设置模型的 vocabulary_ 属性为词汇表
        self.vocabulary_ = vocabulary

        # 根据 sparse 参数决定是否生成稀疏矩阵
        if self.sparse:
            # 初始化稀疏
    _parameter_constraints: dict = {
        "dtype": "no_validation",  # 设置数据类型参数的约束为不进行验证，由 numpy 处理验证
        "separator": [str],         # 设置分隔符参数的约束为字符串类型列表
        "sparse": ["boolean"],      # 设置稀疏矩阵参数的约束为布尔值
        "sort": ["boolean"],        # 设置排序参数的约束为布尔值
    }

    def __init__(self, *, dtype=np.float64, separator="=", sparse=True, sort=True):
        # 初始化方法，设置类的属性
        self.dtype = dtype          # 设置数据类型属性
        self.separator = separator  # 设置分隔符属性
        self.sparse = sparse        # 设置稀疏矩阵属性
        self.sort = sort            # 设置排序属性

    def _add_iterable_element(
        self,
        f,
        v,
        feature_names,
        vocab,
        *,
        fitting=True,
        transforming=False,
        indices=None,
        values=None,
    ):
        """Add feature names for iterable of strings"""
        # 添加可迭代字符串的特征名
        for vv in v:
            if isinstance(vv, str):  # 如果当前元素是字符串
                feature_name = "%s%s%s" % (f, self.separator, vv)  # 构造特征名
                vv = 1  # 将值设为1
            else:
                # 如果不是字符串，抛出类型错误异常
                raise TypeError(
                    f"Unsupported type {type(vv)} in iterable "
                    "value. Only iterables of string are "
                    "supported."
                )
            if fitting and feature_name not in vocab:
                vocab[feature_name] = len(feature_names)  # 添加到词汇表
                feature_names.append(feature_name)       # 添加到特征名列表

            if transforming and feature_name in vocab:
                indices.append(vocab[feature_name])      # 添加索引
                values.append(self.dtype(vv))            # 添加值，并根据数据类型进行转换

    @_fit_context(prefer_skip_nested_validation=True)
    # 定义一个方法用于学习特征名到索引的映射关系，并为 DictVectorizer 类实例化一个字典
    def fit(self, X, y=None):
        """Learn a list of feature name -> indices mappings.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

            .. versionchanged:: 0.24
               Accepts multiple string values for one categorical feature.

        y : (ignored)
            Ignored parameter.

        Returns
        -------
        self : object
            DictVectorizer class instance.
        """
        # 初始化空的特征名列表和词汇表字典
        feature_names = []
        vocab = {}

        # 遍历输入的 X
        for x in X:
            # 遍历每个 x 中的特征名 f 和对应的值 v
            for f, v in x.items():
                # 如果值 v 是字符串类型，则特征名由特征名 f、分隔符和值 v 组成
                if isinstance(v, str):
                    feature_name = "%s%s%s" % (f, self.separator, v)
                # 如果值 v 是数字类型或者 None，则特征名只由特征名 f 组成
                elif isinstance(v, Number) or (v is None):
                    feature_name = f
                # 如果值 v 是 Mapping 类型，则抛出类型错误异常
                elif isinstance(v, Mapping):
                    raise TypeError(
                        f"Unsupported value type {type(v)} "
                        f"for {f}: {v}.\n"
                        "Mapping objects are not supported."
                    )
                # 如果值 v 是可迭代对象，则特征名为 None，调用 _add_iterable_element 方法处理
                elif isinstance(v, Iterable):
                    feature_name = None
                    self._add_iterable_element(f, v, feature_names, vocab)

                # 如果特征名不为 None，则将特征名加入词汇表并记录在特征名列表中
                if feature_name is not None:
                    if feature_name not in vocab:
                        vocab[feature_name] = len(feature_names)
                        feature_names.append(feature_name)

        # 如果需要排序特征名列表，则进行排序并更新词汇表
        if self.sort:
            feature_names.sort()
            vocab = {f: i for i, f in enumerate(feature_names)}

        # 将学习到的特征名列表和词汇表分别赋值给对象的属性
        self.feature_names_ = feature_names
        self.vocabulary_ = vocab

        # 返回自身实例
        return self

    # 使用装饰器 _fit_context 标记，定义一个方法用于学习特征名到索引的映射关系并同时对 X 进行转换
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """Learn a list of feature name -> indices mappings and transform X.

        Like fit(X) followed by transform(X), but does not require
        materializing X in memory.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

            .. versionchanged:: 0.24
               Accepts multiple string values for one categorical feature.

        y : (ignored)
            Ignored parameter.

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        # 调用 _transform 方法进行转换操作，标记 fitting=True 表示在学习过程中
        return self._transform(X, fitting=True)
    def inverse_transform(self, X, dict_type=dict):
        """Transform array or sparse matrix X back to feature mappings.
        
        X must have been produced by this DictVectorizer's transform or
        fit_transform method; it may only have passed through transformers
        that preserve the number of features and their order.
        
        In the case of one-hot/one-of-K coding, the constructed feature
        names and values are returned rather than the original ones.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Sample matrix.
        dict_type : type, default=dict
            Constructor for feature mappings. Must conform to the
            collections.Mapping API.
        
        Returns
        -------
        D : list of dict_type objects of shape (n_samples,)
            Feature mappings for the samples in X.
        """
        # 检查是否已拟合，即是否已经通过fit或fit_transform方法获取了特征名
        check_is_fitted(self, "feature_names_")
        
        # 如果X是COO格式的稀疏矩阵，则不支持使用下标访问，需要转换为支持的格式
        X = check_array(X, accept_sparse=["csr", "csc"])
        n_samples = X.shape[0]
        
        # 获取特征名列表
        names = self.feature_names_
        # 创建空字典列表，用于存放每个样本的特征映射
        dicts = [dict_type() for _ in range(n_samples)]
        
        # 如果X是稀疏矩阵
        if sp.issparse(X):
            # 遍历非零元素的索引，将对应的特征值填入字典中
            for i, j in zip(*X.nonzero()):
                dicts[i][names[j]] = X[i, j]
        else:
            # 如果X是稠密矩阵，直接遍历每个样本的特征值
            for i, d in enumerate(dicts):
                for j, v in enumerate(X[i, :]):
                    if v != 0:
                        d[names[j]] = X[i, j]
        
        # 返回特征映射字典列表
        return dicts

    def transform(self, X):
        """Transform feature->value dicts to array or sparse matrix.
        
        Named features not encountered during fit or fit_transform will be
        silently ignored.
        
        Parameters
        ----------
        X : Mapping or iterable over Mappings of shape (n_samples,)
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        
        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        # 检查是否已拟合，即是否已经通过fit或fit_transform方法获取了特征名和词汇表
        check_is_fitted(self, ["feature_names_", "vocabulary_"])
        # 调用_transform方法进行实际的转换操作，不进行拟合过程
        return self._transform(X, fitting=False)

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
        # 检查是否已拟合，即是否已经通过fit或fit_transform方法获取了特征名
        check_is_fitted(self, "feature_names_")
        # 如果特征名中有非字符串类型的，强制转换为字符串
        if any(not isinstance(name, str) for name in self.feature_names_):
            feature_names = [str(name) for name in self.feature_names_]
        else:
            feature_names = self.feature_names_
        # 将特征名列表转换为NumPy数组并返回
        return np.asarray(feature_names, dtype=object)
    def restrict(self, support, indices=False):
        """
        Restrict the features to those in support using feature selection.

        This function modifies the estimator in-place.

        Parameters
        ----------
        support : array-like
            Boolean mask or list of indices (as returned by the get_support
            member of feature selectors).
        indices : bool, default=False
            Whether support is a list of indices.

        Returns
        -------
        self : object
            DictVectorizer class instance.

        Examples
        --------
        >>> from sklearn.feature_extraction import DictVectorizer
        >>> from sklearn.feature_selection import SelectKBest, chi2
        >>> v = DictVectorizer()
        >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        >>> X = v.fit_transform(D)
        >>> support = SelectKBest(chi2, k=2).fit(X, [0, 1])
        >>> v.get_feature_names_out()
        array(['bar', 'baz', 'foo'], ...)
        >>> v.restrict(support.get_support())
        DictVectorizer()
        >>> v.get_feature_names_out()
        array(['bar', 'foo'], ...)
        """

        # Ensure the instance has been fitted and has feature names available
        check_is_fitted(self, "feature_names_")

        # Convert support to indices if it is not already in index form
        if not indices:
            support = np.where(support)[0]

        # Retrieve current feature names
        names = self.feature_names_

        # Create a new vocabulary dictionary based on the supported features
        new_vocab = {}
        for i in support:
            new_vocab[names[i]] = len(new_vocab)

        # Update the vocabulary of the DictVectorizer instance
        self.vocabulary_ = new_vocab

        # Update the feature names in sorted order based on the new vocabulary
        self.feature_names_ = [
            f for f, i in sorted(new_vocab.items(), key=itemgetter(1))
        ]

        # Return the modified instance of DictVectorizer
        return self
```