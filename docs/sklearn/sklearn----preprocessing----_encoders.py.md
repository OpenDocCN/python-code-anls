# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\_encoders.py`

```
"""
# 作者: scikit-learn 开发者
# SPDX 许可证: BSD-3-Clause

# 导入必要的模块和库
import numbers  # 导入 numbers 模块
import warnings  # 导入 warnings 模块
from numbers import Integral  # 从 numbers 模块导入 Integral 类

import numpy as np  # 导入 NumPy 库，并重命名为 np
from scipy import sparse  # 从 SciPy 库导入 sparse 模块

# 从 scikit-learn 内部模块导入需要的类和函数
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..utils import _safe_indexing, check_array  # 导入 _safe_indexing 和 check_array 函数
from ..utils._encode import _check_unknown, _encode, _get_counts, _unique  # 导入编码相关函数
from ..utils._mask import _get_mask  # 导入 _get_mask 函数
from ..utils._missing import is_scalar_nan  # 导入 is_scalar_nan 函数
from ..utils._param_validation import Interval, RealNotInt, StrOptions  # 导入参数验证相关类
from ..utils._set_output import _get_output_config  # 导入 _get_output_config 函数
from ..utils.validation import _check_feature_names_in, check_is_fitted  # 导入验证函数

__all__ = ["OneHotEncoder", "OrdinalEncoder"]  # 设置模块的公开接口

class _BaseEncoder(TransformerMixin, BaseEstimator):
    """
    编码器的基类，包括对输入特征进行分类和转换的代码。
    """

    def _check_X(self, X, force_all_finite=True):
        """
        自定义的 check_array 执行以下操作：
        - 将字符串列表转换为对象 dtype
        - 对于对象 dtype 数据，检查缺失值（check_array 本身不执行此操作）
        - 返回特征列表（数组），保留 pandas DataFrame 列的数据类型信息，
          否则信息将丢失并且无法用于例如 `categories_` 属性。

        """
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            # 如果不是 DataFrame，则执行常规的 check_array 验证
            X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
            if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object, force_all_finite=force_all_finite)
            else:
                X = X_temp
            needs_validation = False
        else:
            # 是 pandas DataFrame，稍后逐列验证，以保留编码器中使用的数据类型信息。
            needs_validation = force_all_finite

        n_samples, n_features = X.shape
        X_columns = []

        for i in range(n_features):
            Xi = _safe_indexing(X, indices=i, axis=1)
            Xi = check_array(
                Xi, ensure_2d=False, dtype=None, force_all_finite=needs_validation
            )
            X_columns.append(Xi)

        return X_columns, n_samples, n_features

    def _fit(
        self,
        X,
        handle_unknown="error",
        force_all_finite=True,
        return_counts=False,
        return_and_ignore_missing_for_infrequent=False,
    ):
        # 进行拟合操作的核心函数，具体实现依赖于子类的实现。
        pass

    def _transform(
        self,
        X,
        handle_unknown="error",
        force_all_finite=True,
        warn_on_unknown=False,
        ignore_category_indices=None,
"""
        ):
        # 使用 self._check_X 方法验证输入 X 是否合法，并返回处理后的 X_list、样本数 n_samples、特征数 n_features
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )
        # 检查特征名，不重置
        self._check_feature_names(X, reset=False)
        # 检查特征数，不重置
        self._check_n_features(X, reset=False)

        # 初始化一个全零矩阵 X_int，用于存储整数编码后的特征
        X_int = np.zeros((n_samples, n_features), dtype=int)
        # 初始化一个全一矩阵 X_mask，用于标记有效值的掩码
        X_mask = np.ones((n_samples, n_features), dtype=bool)

        # 初始化一个空列表，用于存储含有未知类别的列索引
        columns_with_unknown = []
        # 遍历每个特征
        for i in range(n_features):
            # 获取第 i 列特征 Xi
            Xi = X_list[i]
            # 调用 _check_unknown 函数检查 Xi 中的未知类别，并返回未知类别 diff 和有效掩码 valid_mask
            diff, valid_mask = _check_unknown(Xi, self.categories_[i], return_mask=True)

            # 如果存在未知类别
            if not np.all(valid_mask):
                # 如果 handle_unknown 设置为 "error"，抛出 ValueError 异常
                if handle_unknown == "error":
                    msg = (
                        "Found unknown categories {0} in column {1}"
                        " during transform".format(diff, i)
                    )
                    raise ValueError(msg)
                else:
                    # 如果 warn_on_unknown 设置为 True，将列索引 i 添加到 columns_with_unknown 中
                    if warn_on_unknown:
                        columns_with_unknown.append(i)
                    # 将无效的行标记为可接受的值，并在 X_mask 中标记这些行
                    X_mask[:, i] = valid_mask
                    # 将 Xi 强制转换为能够处理不同长度 numpy 字符串的最大字符串类型
                    if (
                        self.categories_[i].dtype.kind in ("U", "S")
                        and self.categories_[i].itemsize > Xi.itemsize
                    ):
                        Xi = Xi.astype(self.categories_[i].dtype)
                    elif self.categories_[i].dtype.kind == "O" and Xi.dtype.kind == "U":
                        # 如果类别是对象而 Xi 是 numpy 字符串，则将 Xi 转换为对象类型以防止截断
                        Xi = Xi.astype("O")
                    else:
                        Xi = Xi.copy()

                    # 将无效的值设置为类别列表中的第一个值
                    Xi[~valid_mask] = self.categories_[i][0]
            # 由于上面已经调用过 _check_unknown，因此这里使用 check_unknown=False
            # 对 Xi 进行编码，将结果存入 X_int 中的第 i 列
            X_int[:, i] = _encode(Xi, uniques=self.categories_[i], check_unknown=False)
        # 如果存在未知类别的列，发出警告
        if columns_with_unknown:
            warnings.warn(
                (
                    "Found unknown categories in columns "
                    f"{columns_with_unknown} during transform. These "
                    "unknown categories will be encoded as all zeros"
                ),
                UserWarning,
            )

        # 对罕见的类别进行映射处理
        self._map_infrequent_categories(X_int, X_mask, ignore_category_indices)
        # 返回整数编码后的特征矩阵 X_int 和有效值掩码 X_mask
        return X_int, X_mask
    def infrequent_categories_(self):
        """Return infrequent categories for each feature.

        Returns
        -------
        list of ndarray or None
            List containing infrequent categories for each feature if available,
            otherwise None.
        """
        # 获取已定义的 _infrequent_indices 属性，如果未定义则引发 AttributeError
        infrequent_indices = self._infrequent_indices
        return [
            # 根据 _infrequent_indices 中的索引获取相应的 infrequent categories
            None if indices is None else category[indices]
            for category, indices in zip(self.categories_, infrequent_indices)
        ]

    def _check_infrequent_enabled(self):
        """
        Check if infrequent category filtering is enabled.

        This function checks whether _infrequent_enabled is True or False.
        This should be called after parameter validation in the fit function.
        """
        max_categories = getattr(self, "max_categories", None)
        min_frequency = getattr(self, "min_frequency", None)
        # 根据 max_categories 和 min_frequency 的存在与否设定 _infrequent_enabled
        self._infrequent_enabled = (
            max_categories is not None and max_categories >= 1
        ) or min_frequency is not None

    def _identify_infrequent(self, category_count, n_samples, col_idx):
        """Compute infrequent category indices.

        Parameters
        ----------
        category_count : ndarray of shape (n_cardinality,)
            Counts of each category.

        n_samples : int
            Number of samples.

        col_idx : int
            Index of the current feature column (used for error messages).

        Returns
        -------
        ndarray or None
            Indices of infrequent categories if any, otherwise None.
        """
        if isinstance(self.min_frequency, numbers.Integral):
            # 根据整数型的 min_frequency 进行判断
            infrequent_mask = category_count < self.min_frequency
        elif isinstance(self.min_frequency, numbers.Real):
            # 根据实数型的 min_frequency 计算绝对频率阈值
            min_frequency_abs = n_samples * self.min_frequency
            infrequent_mask = category_count < min_frequency_abs
        else:
            # 如果 min_frequency 未定义，则认为没有 infrequent categories
            infrequent_mask = np.zeros(category_count.shape[0], dtype=bool)

        # 计算当前有效特征数目（排除 infrequent categories）
        n_current_features = category_count.size - infrequent_mask.sum() + 1

        if self.max_categories is not None and self.max_categories < n_current_features:
            # 如果设定了 max_categories，并且有效特征数目超过了设定值
            frequent_category_count = self.max_categories - 1
            if frequent_category_count == 0:
                # 所有类别都是 infrequent categories
                infrequent_mask[:] = True
            else:
                # 使用稳定排序来保持原始计数的顺序
                smallest_levels = np.argsort(category_count, kind="mergesort")[
                    :-frequent_category_count
                ]
                infrequent_mask[smallest_levels] = True

        # 返回 infrequent categories 的索引数组
        output = np.flatnonzero(infrequent_mask)
        return output if output.size > 0 else None

    def _fit_infrequent_category_mapping(
        self, n_samples, category_counts, missing_indices
    ):
        """Fit mapping for infrequent categories.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        category_counts : list of ndarray
            Counts of categories for each feature.

        missing_indices : list of ndarray
            Indices of missing values for each feature.

        """
    # 将罕见的类别映射到表示罕见类别的整数值，直接修改 X_int
    def _map_infrequent_categories(self, X_int, X_mask, ignore_category_indices):
        """Map infrequent categories to integer representing the infrequent category.

        This modifies X_int in-place. Values that were invalid based on `X_mask`
        are mapped to the infrequent category if there was an infrequent
        category for that feature.

        Parameters
        ----------
        X_int: ndarray of shape (n_samples, n_features)
            Integer encoded categories.

        X_mask: ndarray of shape (n_samples, n_features)
            Bool mask for valid values in `X_int`.

        ignore_category_indices : dict
            Dictionary mapping from feature_idx to category index to ignore.
            Ignored indexes will not be grouped and the original ordinal encoding
            will remain.
        """
        # 如果未启用罕见类别处理，则直接返回
        if not self._infrequent_enabled:
            return

        # 如果 ignore_category_indices 为空，则初始化为空字典
        ignore_category_indices = ignore_category_indices or {}

        # 遍历每一列
        for col_idx in range(X_int.shape[1]):
            # 获取当前列的罕见类别索引
            infrequent_idx = self._infrequent_indices[col_idx]
            # 如果当前列没有罕见类别，跳过当前循环
            if infrequent_idx is None:
                continue

            # 将不在 X_mask 中有效的值映射到罕见类别索引的第一个值
            X_int[~X_mask[:, col_idx], col_idx] = infrequent_idx[0]

            # 如果处理未知值的策略是 "infrequent_if_exist"
            if self.handle_unknown == "infrequent_if_exist":
                # 所有未知的值现在都映射到 infrequent_idx[0]，使得未知值变为有效值
                # 这在 `transform` 中形成编码时使用 `X_mask` 是必需的
                X_mask[:, col_idx] = True

        # 重新映射 `X_int` 中的编码，其中罕见类别被分组在一起
        for i, mapping in enumerate(self._default_to_infrequent_mappings):
            # 如果当前特征的映射为空，则跳过当前循环
            if mapping is None:
                continue

            # 如果当前列索引在 ignore_category_indices 中
            if i in ignore_category_indices:
                # 更新 **不** 被忽略的行
                rows_to_update = X_int[:, i] != ignore_category_indices[i]
            else:
                rows_to_update = slice(None)

            # 使用 np.take 将映射应用到 X_int 中的相应列
            X_int[rows_to_update, i] = np.take(mapping, X_int[rows_to_update, i])

    # 返回更多的标签信息，指明输入类型和是否允许 NaN 值
    def _more_tags(self):
        return {"X_types": ["2darray", "categorical"], "allow_nan": True}
class OneHotEncoder(_BaseEncoder):
    """
    Encode categorical features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse_output``
    parameter).

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    For a comparison of different encoders, refer to:
    :ref:`sphx_glr_auto_examples_preprocessing_plot_target_encoder.py`.

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

        .. versionadded:: 0.20
    """
    # drop 参数用于指定丢弃分类变量中的某些类别的方法
    # 对于具有完全共线特征可能引起问题的情况很有用，比如在将结果数据输入未正则化的线性回归模型时
    # 但是，丢弃一个类别会破坏原始表示的对称性，因此可能会在下游模型中引入偏差，例如对于带有惩罚的线性分类或回归模型

    # None: 保留所有特征（默认行为）
    # 'first': 在每个特征中丢弃第一个类别。如果每个特征只有一个类别，则整个特征将被丢弃
    # 'if_binary': 在每个具有两个类别的特征中丢弃第一个类别。保持具有1个或超过2个类别的特征不变
    # array: ``drop[i]`` 是要在特征 ``X[:, i]`` 中丢弃的类别

    # 当 `max_categories` 或 `min_frequency` 配置为对不常见的类别进行分组时，丢弃行为将在分组后处理

    # .. versionadded:: 0.21
    #    参数 `drop` 在版本 0.21 中添加

    # .. versionchanged:: 0.23
    #    选项 `drop='if_binary'` 在版本 0.23 中添加

    # .. versionchanged:: 1.1
    #    支持丢弃不常见类别
    drop: {'first', 'if_binary'} or an array-like of shape (n_features,),
            default=None

    # sparse_output 参数控制返回结果是否是稀疏矩阵的输出格式
    # 当为 ``True`` 时，返回一个 :class:`scipy.sparse.csr_matrix`，即压缩稀疏行 (CSR) 格式的稀疏矩阵

    # .. versionadded:: 1.2
    #    `sparse` 重命名为 `sparse_output`
    sparse_output: bool, default=True

    # dtype 参数指定输出的期望数据类型，默认为 np.float64
    dtype: number type, default=np.float64
    handle_unknown : {'error', 'ignore', 'infrequent_if_exist'}, \
                     default='error'
    处理未知类别的方式，用于 :meth:`transform` 方法。

        - 'error' : 如果在 transform 过程中遇到未知类别，则抛出错误。
        - 'ignore' : 如果在 transform 过程中遇到未知类别，则生成的独热编码列将全为零。在逆转换中，未知类别将表示为 None。
        - 'infrequent_if_exist' : 如果在 transform 过程中遇到未知类别，则生成的独热编码列将映射到频率低的类别（如果存在）。频率低的类别将被映射到编码的最后位置。在逆转换中，未知类别将映射到标记为 `'infrequent'` 的类别（如果存在）。如果 `'infrequent'` 类别不存在，则 :meth:`transform` 和 :meth:`inverse_transform` 将像 `handle_unknown='ignore'` 一样处理未知类别。频率低的类别基于 `min_frequency` 和 `max_categories`。详细信息请参阅 :ref:`User Guide <encoder_infrequent_categories>`。

        .. versionchanged:: 1.1
            添加了 `'infrequent_if_exist'` 以自动处理未知类别和频率低的类别。

    min_frequency : int or float, default=None
    指定被视为频率低的类别的最小频率。

        - 如果是 `int`，则具有更小基数的类别将被视为频率低。
        - 如果是 `float`，则具有比 `min_frequency * n_samples` 更小基数的类别将被视为频率低。

        .. versionadded:: 1.1
            详细信息请参阅 :ref:`User Guide <encoder_infrequent_categories>`。

    max_categories : int, default=None
    在考虑频率低的类别时，指定每个输入特征的输出特征数量的上限。如果存在频率低的类别，`max_categories` 包括代表频率低的类别以及频繁类别。如果为 `None`，则输出特征数量没有限制。

        .. versionadded:: 1.1
            详细信息请参阅 :ref:`User Guide <encoder_infrequent_categories>`。

    feature_name_combiner : "concat" or callable, default="concat"
    用于创建 :meth:`get_feature_names_out` 返回的特征名称的可调用对象，具有签名 `def callable(input_feature, category)`。

        `"concat"` 将编码的特征名称和类别连接起来，使用 `feature + "_" + str(category)`。例如，特征 X 的值为 1、6、7，创建的特征名称为 `X_1, X_6, X_7`。

        .. versionadded:: 1.3
    Attributes
    ----------
    categories_ : list of arrays
        每个特征在拟合期间确定的类别
        （按照X中特征的顺序，并与“transform”的输出对应）。
        包括在`drop`参数指定的类别（如果有的话）。

    drop_idx_ : array of shape (n_features,)
        - ``drop_idx_[i]`` 是`categories_[i]`中要删除的类别的索引
          对于每个特征。
        - 如果没有要从特征`i`中删除的类别，例如当`drop='if_binary'`且特征不是二进制时，
          ``drop_idx_[i] = None``。
        - 如果所有转换后的特征都将被保留，则 ``drop_idx_ = None``。

        如果通过将`min_frequency`或`max_categories`设置为非默认值启用了不常见的类别，
        并且`drop_idx[i]`对应于一个不常见的类别，则整个不常见的类别将被删除。

        .. versionchanged:: 0.23
           添加了包含`None`值的可能性。

    infrequent_categories_ : list of ndarray
        仅在通过将`min_frequency`或`max_categories`设置为非默认值启用了不常见的类别时定义。
        `infrequent_categories_[i]` 是特征`i`的不常见类别。
        如果特征`i`没有不常见的类别，则 `infrequent_categories_[i]` 为None。

        .. versionadded:: 1.1

    n_features_in_ : int
        在拟合期间观察到的特征数量。

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在`fit`期间观察到的特征的名称。仅当`X`具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    feature_name_combiner : callable or None
        具有签名 `def callable(input_feature, category)` 的可调用对象，
        返回一个字符串。用于创建由:meth:`get_feature_names_out`返回的特征名称。

        .. versionadded:: 1.3

    See Also
    --------
    OrdinalEncoder : 对分类特征执行序数（整数）编码。
    TargetEncoder : 使用目标对分类特征进行编码。
    sklearn.feature_extraction.DictVectorizer : 对字典项执行一种独热编码（也处理字符串值特征）。
    sklearn.feature_extraction.FeatureHasher : 对字典项或字符串执行近似独热编码。
    LabelBinarizer : 以一对所有方式对标签进行二元编码。
    MultiLabelBinarizer : 在可迭代的可迭代对象和多标签格式之间进行转换，
      例如（样本 x 类别）二进制矩阵，指示类标签的存在。

    Examples
    --------
    假设数据集有两个特征，我们让编码器找到每个特征的唯一值，并将数据转换为二进制的独热编码。
    _parameter_constraints: dict = {
        # 定义参数约束字典，用于描述 OneHotEncoder 的参数限制条件
        "categories": [StrOptions({"auto"}), list],
        # `categories` 参数可接受的取值为字符串 "auto" 或列表类型
        "drop": [StrOptions({"first", "if_binary"}), "array-like", None],
        # `drop` 参数可接受的取值为字符串 "first" 或 "if_binary"，或者类似数组的对象，也可以为 None
        "dtype": "no_validation",  # validation delegated to numpy
        # `dtype` 参数的类型不做额外验证，由 numpy 负责验证
        "handle_unknown": [StrOptions({"error", "ignore", "infrequent_if_exist"})],
        # `handle_unknown` 参数可接受的取值为字符串 "error"、"ignore" 或 "infrequent_if_exist"
        "max_categories": [Interval(Integral, 1, None, closed="left"), None],
        # `max_categories` 参数为整数且必须大于等于 1，或者可以为 None
        "min_frequency": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="neither"),
            None,
        ],
        # `min_frequency` 参数可以是整数且必须大于等于 1，或者是实数但不是整数且大于等于 0 且小于等于 1，或者可以为 None
        "sparse_output": ["boolean"],
        # `sparse_output` 参数为布尔值类型
        "feature_name_combiner": [StrOptions({"concat"}), callable],
        # `feature_name_combiner` 参数可接受的取值为字符串 "concat" 或可调用对象
    }
    def __init__(
        self,
        *,
        categories="auto",
        drop=None,
        sparse_output=True,
        dtype=np.float64,
        handle_unknown="error",
        min_frequency=None,
        max_categories=None,
        feature_name_combiner="concat",
    ):
        # 初始化方法，用于初始化分类编码器的各种参数
        self.categories = categories  # 设定分类编码的方式，默认为自动选择
        self.sparse_output = sparse_output  # 是否使用稀疏矩阵输出，默认为True
        self.dtype = dtype  # 数据类型，默认为np.float64
        self.handle_unknown = handle_unknown  # 处理未知分类的策略，默认为抛出错误
        self.drop = drop  # 要删除的特征列表，默认为None
        self.min_frequency = min_frequency  # 最小频率阈值，默认为None
        self.max_categories = max_categories  # 最大分类数限制，默认为None
        self.feature_name_combiner = feature_name_combiner  # 特征名称组合方式，默认为连接字符串方式

    def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
        """Convert `drop_idx` into the index for infrequent categories.

        If there are no infrequent categories, then `drop_idx` is
        returned. This method is called in `_set_drop_idx` when the `drop`
        parameter is an array-like.
        """
        if not self._infrequent_enabled:  # 如果没有启用低频分类
            return drop_idx

        default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
        if default_to_infrequent is None:
            return drop_idx

        # 当明确删除一个低频分类时引发错误
        infrequent_indices = self._infrequent_indices[feature_idx]
        if infrequent_indices is not None and drop_idx in infrequent_indices:
            categories = self.categories_[feature_idx]
            raise ValueError(
                f"Unable to drop category {categories[drop_idx].item()!r} from"
                f" feature {feature_idx} because it is infrequent"
            )
        return default_to_infrequent[drop_idx]

    def _compute_transformed_categories(self, i, remove_dropped=True):
        """Compute the transformed categories used for column `i`.

        1. If there are infrequent categories, the category is named
        'infrequent_sklearn'.
        2. Dropped columns are removed when remove_dropped=True.
        """
        cats = self.categories_[i]  # 获取第 i 列的分类列表

        if self._infrequent_enabled:  # 如果启用了低频分类
            infreq_map = self._default_to_infrequent_mappings[i]
            if infreq_map is not None:
                frequent_mask = infreq_map < infreq_map.max()
                infrequent_cat = "infrequent_sklearn"
                # 低频分类总是位于末尾
                cats = np.concatenate(
                    (cats[frequent_mask], np.array([infrequent_cat], dtype=object))
                )

        if remove_dropped:
            cats = self._remove_dropped_categories(cats, i)  # 移除已删除的分类
        return cats

    def _remove_dropped_categories(self, categories, i):
        """Remove dropped categories."""
        if (
            self._drop_idx_after_grouping is not None
            and self._drop_idx_after_grouping[i] is not None
        ):
            return np.delete(categories, self._drop_idx_after_grouping[i])
        return categories
    def _compute_n_features_outs(self):
        """计算每个输入特征的输出特征数目。"""
        # 对于每个特征的类别列表，计算其长度作为输出特征数目
        output = [len(cats) for cats in self.categories_]

        if self._drop_idx_after_grouping is not None:
            # 如果存在需要删除的索引，则相应地减少输出特征数目
            for i, drop_idx in enumerate(self._drop_idx_after_grouping):
                if drop_idx is not None:
                    output[i] -= 1

        if not self._infrequent_enabled:
            return output

        # 如果启用了不常见类别的处理，相应地减少输出特征数目
        for i, infreq_idx in enumerate(self._infrequent_indices):
            if infreq_idx is None:
                continue
            output[i] -= infreq_idx.size - 1

        return output

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        将 OneHotEncoder 适配到 X 上。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            需要确定每个特征的类别的数据。

        y : None
            忽略。此参数仅用于与 :class:`~sklearn.pipeline.Pipeline` 兼容。

        Returns
        -------
        self
            已适配的编码器。
        """
        # 调用 _fit 方法进行适配，设置处理未知值的方式和处理所有有限值的方式
        self._fit(
            X,
            handle_unknown=self.handle_unknown,
            force_all_finite="allow-nan",
        )
        # 设置需要删除的索引
        self._set_drop_idx()
        # 计算每个输入特征的输出特征数目
        self._n_features_outs = self._compute_n_features_outs()
        return self

    def get_feature_names_out(self, input_features=None):
        """获取转换后的特征名称。

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            输入特征。

            - 如果 `input_features` 为 `None`，则使用 `feature_names_in_` 作为输入特征名称。
              如果 `feature_names_in_` 未定义，则生成以下输入特征名称：
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`。
            - 如果 `input_features` 是数组形式，则必须与 `feature_names_in_` 匹配（如果已定义）。

        Returns
        -------
        feature_names_out : ndarray of str objects
            转换后的特征名称。
        """
        # 检查模型是否已经适配
        check_is_fitted(self)
        # 检查输入特征名称
        input_features = _check_feature_names_in(self, input_features)
        # 对每个特征的类别进行转换，生成转换后的类别列表
        cats = [
            self._compute_transformed_categories(i)
            for i, _ in enumerate(self.categories_)
        ]

        # 获取特征名称组合器
        name_combiner = self._check_get_feature_name_combiner()
        feature_names = []
        # 对于每个特征的转换后类别，生成相应的特征名称
        for i in range(len(cats)):
            names = [name_combiner(input_features[i], t) for t in cats[i]]
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)
    # 定义一个方法用于检查和获取特征名组合器
    def _check_get_feature_name_combiner(self):
        # 如果特征名组合器是 "concat"，返回一个 lambda 函数
        if self.feature_name_combiner == "concat":
            return lambda feature, category: feature + "_" + str(category)
        else:  # 如果特征名组合器是一个可调用对象
            # 进行一次模拟运行，获取组合器的返回值
            dry_run_combiner = self.feature_name_combiner("feature", "category")
            # 检查返回值类型，如果不是字符串则抛出类型错误
            if not isinstance(dry_run_combiner, str):
                raise TypeError(
                    "When `feature_name_combiner` is a callable, it should return a "
                    f"Python string. Got {type(dry_run_combiner)} instead."
                )
            # 返回特征名组合器本身
            return self.feature_name_combiner
# 定义一个名为 OrdinalEncoder 的类，继承自 OneToOneFeatureMixin 和 _BaseEncoder
class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):
    """
    Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    For a comparison of different encoders, refer to:
    :ref:`sphx_glr_auto_examples_preprocessing_plot_target_encoder.py`.

    .. versionadded:: 0.20

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values, and should be sorted in case of numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default=np.float64
        Desired dtype of output.

    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        When set to 'error' an error will be raised in case an unknown
        categorical feature is present during transform. When set to
        'use_encoded_value', the encoded value of unknown categories will be
        set to the value given for the parameter `unknown_value`. In
        :meth:`inverse_transform`, an unknown category will be denoted as None.

        .. versionadded:: 0.24

    unknown_value : int or np.nan, default=None
        When the parameter handle_unknown is set to 'use_encoded_value', this
        parameter is required and will set the encoded value of unknown
        categories. It has to be distinct from the values used to encode any of
        the categories in `fit`. If set to np.nan, the `dtype` parameter must
        be a float dtype.

        .. versionadded:: 0.24

    encoded_missing_value : int or np.nan, default=np.nan
        Encoded value of missing categories. If set to `np.nan`, then the `dtype`
        parameter must be a float dtype.

        .. versionadded:: 1.1

    min_frequency : int or float, default=None
        Specifies the minimum frequency below which a category will be
        considered infrequent.

        - If `int`, categories with a smaller cardinality will be considered
          infrequent.

        - If `float`, categories with a smaller cardinality than
          `min_frequency * n_samples`  will be considered infrequent.

        .. versionadded:: 1.3
            Read more in the :ref:`User Guide <encoder_infrequent_categories>`.
    """
    # max_categories: int, default=None
    #     每个输入特征在考虑不常见类别时的输出类别数的上限。
    #     如果存在不常见类别，`max_categories`包括表示不常见类别的类别，
    #     以及常见类别。如果为 `None`，则输出特征数量没有限制。
    #     
    #     `max_categories` 不考虑缺失或未知类别。
    #     将 `unknown_value` 或 `encoded_missing_value` 设置为整数将使唯一整数编码的数量增加一个。
    #     这可能导致最多 `max_categories + 2` 个整数编码。
    #     
    #     .. versionadded:: 1.3
    #         请在 :ref:`用户指南 <encoder_infrequent_categories>` 中阅读更多内容。

    # Attributes
    # ----------
    # categories_ : list of arrays
    #     在 ``fit`` 过程中确定的每个特征的类别（与 ``transform`` 的输出顺序相对应）。
    #     这不包括在 ``fit`` 过程中未见过的类别。
    
    # n_features_in_ : int
    #     在 :term:`fit` 过程中看到的特征数量。
    #     
    #     .. versionadded:: 1.0

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    #     在 `X` 具有全部为字符串的特征名时定义。
    #     
    #     .. versionadded:: 1.0

    # infrequent_categories_ : list of ndarray
    #     仅在通过将 `min_frequency` 或 `max_categories` 设置为非默认值启用不常见类别时定义。
    #     `infrequent_categories_[i]` 是特征 `i` 的不常见类别。
    #     如果特征 `i` 没有不常见类别，则 `infrequent_categories_[i]` 为 None。
    #     
    #     .. versionadded:: 1.3

    # See Also
    # --------
    # OneHotEncoder : 对分类特征执行一对一编码。此编码适用于低到中基数的分类变量，无论是在监督还是非监督设置中。
    # TargetEncoder : 使用分类特征的监督信号在分类或回归管道中进行编码。此编码通常适用于高基数的分类变量。
    # LabelEncoder : 将目标标签编码为介于 0 和 ``n_classes-1`` 之间的值。

    # Notes
    # -----
    # 在高比例的 `nan` 值情况下，使用 Python 3.10 之前的版本推断类别会变慢。
    # 从 Python 3.10 开始，处理 `nan` 值得到了改进（参见 `bpo-43475 <https://github.com/python/cpython/issues/87641>`_）。

    # Examples
    # --------
    # 给定一个具有两个特征的数据集，我们让编码器找到每个特征的唯一值，并将数据转换为序数编码。
    # 
    # >>> from sklearn.preprocessing import OrdinalEncoder
    # >>> enc = OrdinalEncoder()
    # >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    _parameter_constraints: dict = {
        # 参数约束字典，定义了每个参数的有效取值范围或类型
        "categories": [StrOptions({"auto"}), list],  # 可以是字符串"auto"或列表类型
        "dtype": "no_validation",  # 数据类型不进行额外验证，由numpy处理
        "encoded_missing_value": [Integral, type(np.nan)],  # 缺失值的编码可以是整数或np.nan类型
        "handle_unknown": [StrOptions({"error", "use_encoded_value"})],  # 处理未知值的方式，可以是"error"或"use_encoded_value"
        "unknown_value": [Integral, type(np.nan), None],  # 未知值的表示可以是整数、np.nan类型或者None
        "max_categories": [Interval(Integral, 1, None, closed="left"), None],  # 最大类别数的约束，必须是大于等于1的整数或None
        "min_frequency": [
            Interval(Integral, 1, None, closed="left"),  # 最小频率的约束，必须是大于等于1的整数
            Interval(RealNotInt, 0, 1, closed="neither"),  # 或者是一个非整数的实数，范围在(0, 1)
            None,
        ],
    }

    def __init__(
        self,
        *,
        categories="auto",  # 类别，默认为"auto"
        dtype=np.float64,  # 数据类型，默认为np.float64
        handle_unknown="error",  # 处理未知值的策略，默认为"error"
        unknown_value=None,  # 未知值的表示，默认为None
        encoded_missing_value=np.nan,  # 编码的缺失值，默认为np.nan
        min_frequency=None,  # 最小频率阈值，默认为None
        max_categories=None,  # 最大类别数，默认为None
    ):
        # 初始化方法，设置对象的各个属性
        self.categories = categories  # 设置类别属性
        self.dtype = dtype  # 设置数据类型属性
        self.handle_unknown = handle_unknown  # 设置处理未知值的策略属性
        self.unknown_value = unknown_value  # 设置未知值的表示属性
        self.encoded_missing_value = encoded_missing_value  # 设置编码的缺失值属性
        self.min_frequency = min_frequency  # 设置最小频率阈值属性
        self.max_categories = max_categories  # 设置最大类别数属性

    @_fit_context(prefer_skip_nested_validation=True)
    # 定义一个方法用于将输入数据 X 转换为序数编码

    """
    Transform X to ordinal codes.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to encode.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Transformed input.
    """
    
    # 检查当前对象是否已经拟合，即是否已计算出类别信息
    check_is_fitted(self, "categories_")
    
    # 调用内部方法 _transform，将输入数据 X 转换为整数编码的形式，并返回转换后的结果以及缺失值的掩码
    X_int, X_mask = self._transform(
        X,
        handle_unknown=self.handle_unknown,
        force_all_finite="allow-nan",
        ignore_category_indices=self._missing_indices,
    )
    
    # 将转换后的整数编码的数据类型转换为指定的 dtype，并且在原地修改而不创建副本
    X_trans = X_int.astype(self.dtype, copy=False)

    # 对于每个分类索引和其对应的缺失值索引，将对应的 X_trans 中的缺失值位置用指定的编码值进行替换
    for cat_idx, missing_idx in self._missing_indices.items():
        X_missing_mask = X_int[:, cat_idx] == missing_idx
        X_trans[X_missing_mask, cat_idx] = self.encoded_missing_value

    # 如果 handle_unknown 设置为 "use_encoded_value"，则将所有未知值的位置替换为指定的未知值编码
    if self.handle_unknown == "use_encoded_value":
        X_trans[~X_mask] = self.unknown_value
    
    # 返回转换后的数据 X_trans
    return X_trans
```