# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\_target_encoder.py`

```
# 导入必要的模块和类
from numbers import Integral, Real
import numpy as np
# 导入所需的基类和函数
from ..base import OneToOneFeatureMixin, _fit_context
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import type_of_target
from ..utils.validation import (
    _check_feature_names_in,
    _check_y,
    check_consistent_length,
    check_is_fitted,
)
# 导入编码器基类和快速编码函数
from ._encoders import _BaseEncoder
from ._target_encoder_fast import _fit_encoding_fast, _fit_encoding_fast_auto_smooth

class TargetEncoder(OneToOneFeatureMixin, _BaseEncoder):
    """Target Encoder for regression and classification targets.

    Each category is encoded based on a shrunk estimate of the average target
    values for observations belonging to the category. The encoding scheme mixes
    the global target mean with the target mean conditioned on the value of the
    category (see [MIC]_).

    When the target type is "multiclass", encodings are based
    on the conditional probability estimate for each class. The target is first
    binarized using the "one-vs-all" scheme via
    :class:`~sklearn.preprocessing.LabelBinarizer`, then the average target
    value for each class and each category is used for encoding, resulting in
    `n_features` * `n_classes` encoded output features.

    :class:`TargetEncoder` considers missing values, such as `np.nan` or `None`,
    as another category and encodes them like any other category. Categories
    that are not seen during :meth:`fit` are encoded with the target mean, i.e.
    `target_mean_`.

    For a demo on the importance of the `TargetEncoder` internal cross-fitting,
    see
    :ref:`sphx_glr_auto_examples_preprocessing_plot_target_encoder_cross_val.py`.
    For a comparison of different encoders, refer to
    :ref:`sphx_glr_auto_examples_preprocessing_plot_target_encoder.py`. Read
    more in the :ref:`User Guide <target_encoder>`.

    .. note::
        `fit(X, y).transform(X)` does not equal `fit_transform(X, y)` because a
        :term:`cross fitting` scheme is used in `fit_transform` for encoding.
        See the :ref:`User Guide <target_encoder>` for details.

    .. versionadded:: 1.3

    Parameters
    ----------
    categories : "auto" or list of shape (n_features,) of array-like, default="auto"
        Categories (unique values) per feature:

        - `"auto"` : Determine categories automatically from the training data.
        - list : `categories[i]` holds the categories expected in the i-th column. The
          passed categories should not mix strings and numeric values within a single
          feature, and should be sorted in case of numeric values.

        The used categories are stored in the `categories_` fitted attribute.
"""
    target_type : {"auto", "continuous", "binary", "multiclass"}, default="auto"
        Type of target.

        - `"auto"` : Type of target is inferred with
          :func:`~sklearn.utils.multiclass.type_of_target`.
          # 使用 sklearn.utils.multiclass.type_of_target 推断目标变量的类型

        - `"continuous"` : Continuous target
          # 连续型目标变量

        - `"binary"` : Binary target
          # 二元分类目标变量

        - `"multiclass"` : Multiclass target
          # 多类别分类目标变量

        .. note::
            The type of target inferred with `"auto"` may not be the desired target
            type used for modeling. For example, if the target consisted of integers
            between 0 and 100, then :func:`~sklearn.utils.multiclass.type_of_target`
            will infer the target as `"multiclass"`. In this case, setting
            `target_type="continuous"` will specify the target as a regression
            problem. The `target_type_` attribute gives the target type used by the
            encoder.
            # 使用 "auto" 推断的目标类型可能不是建模所需的目标类型。例如，如果目标变量是介于0和100之间的整数，
            # 则 :func:`~sklearn.utils.multiclass.type_of_target` 将其推断为 "multiclass"。在这种情况下，
            # 设置 `target_type="continuous"` 将指定目标为回归问题。`target_type_` 属性给出了编码器使用的目标类型。

        .. versionchanged:: 1.4
           Added the option 'multiclass'.

    smooth : "auto" or float, default="auto"
        The amount of mixing of the target mean conditioned on the value of the
        category with the global target mean. A larger `smooth` value will put
        more weight on the global target mean.
        If `"auto"`, then `smooth` is set to an empirical Bayes estimate.
        # 目标均值在类别值与全局目标均值条件下的混合程度。较大的 `smooth` 值会更加依赖全局目标均值。
        # 如果是 `"auto"`，则 `smooth` 被设置为一个经验贝叶斯估计值。

    cv : int, default=5
        Determines the number of folds in the :term:`cross fitting` strategy used in
        :meth:`fit_transform`. For classification targets, `StratifiedKFold` is used
        and for continuous targets, `KFold` is used.
        # 确定在 `fit_transform` 中使用的交叉拟合策略中的折叠数量。对于分类目标，使用 `StratifiedKFold`，
        # 对于连续型目标，使用 `KFold`。

    shuffle : bool, default=True
        Whether to shuffle the data in :meth:`fit_transform` before splitting into
        folds. Note that the samples within each split will not be shuffled.
        # 在将数据拆分为折叠之前，是否在 `fit_transform` 中对数据进行洗牌。注意，每个拆分内的样本不会被洗牌。

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
        # 当 `shuffle` 为 True 时，`random_state` 影响索引的排序，控制每个折叠的随机性。否则，此参数无效。
        # 传递一个整数可实现多次函数调用的可重现输出。参见 :term:`Glossary <random_state>`。

    Attributes
    ----------
    encodings_ : list of shape (n_features,) or (n_features * n_classes) of \
                    ndarray
        Encodings learnt on all of `X`.
        For feature `i`, `encodings_[i]` are the encodings matching the
        categories listed in `categories_[i]`. When `target_type_` is
        "multiclass", the encoding for feature `i` and class `j` is stored in
        `encodings_[j + (i * len(classes_))]`. E.g., for 2 features (f) and
        3 classes (c), encodings are ordered:
        f0_c0, f0_c1, f0_c2, f1_c0, f1_c1, f1_c2,
        # 在 `X` 的所有数据上学习到的编码。
        # 对于特征 `i`，`encodings_[i]` 是与 `categories_[i]` 中列出的类别相匹配的编码。
        # 当 `target_type_` 为 "multiclass" 时，特征 `i` 和类 `j` 的编码存储在 `encodings_[j + (i * len(classes_))]` 中。
        # 例如，对于 2 个特征 (f) 和 3 个类别 (c)，编码顺序为：f0_c0, f0_c1, f0_c2, f1_c0, f1_c1, f1_c2,

    categories_ : list of shape (n_features,) of ndarray
        The categories of each input feature determined during fitting or
        specified in `categories`
        (in order of the features in `X` and corresponding with the output
        of :meth:`transform`).
        # 在拟合过程中确定或在 `categories` 中指定的每个输入特征的类别。
        # (按照 `X` 中的特征顺序，并与 :meth:`transform` 的输出对应)

    target_type_ : str
        Type of target.
        # 目标变量的类型。
    _parameter_constraints: dict = {
        "categories": [StrOptions({"auto"}), list],
        # 参数 `categories` 可以接受字符串选项集合 {"auto"} 或者列表类型
        "target_type": [StrOptions({"auto", "continuous", "binary", "multiclass"})],
        # 参数 `target_type` 可以接受字符串选项集合 {"auto", "continuous", "binary", "multiclass"}
        "smooth": [StrOptions({"auto"}), Interval(Real, 0, None, closed="left")],
        # 参数 `smooth` 可以接受字符串选项集合 {"auto"}，或者一个实数区间 [0, ∞)，左闭右开
        "cv": [Interval(Integral, 2, None, closed="left")],
        # 参数 `cv` 是一个整数区间 [2, ∞)，左闭右开
        "shuffle": ["boolean"],
        # 参数 `shuffle` 是布尔类型
        "random_state": ["random_state"],
        # 参数 `random_state` 是一个随机状态对象
    }
    def __init__(
        self,
        categories="auto",
        target_type="auto",
        smooth="auto",
        cv=5,
        shuffle=True,
        random_state=None,
    ):
        # 初始化方法，用于创建TargetEncoder对象并设置初始参数

        self.categories = categories
        # 设定类别编码的策略，默认为"auto"

        self.smooth = smooth
        # 设定平滑参数的策略，默认为"auto"

        self.target_type = target_type
        # 设定目标类型，用于编码目标数据，默认为"auto"

        self.cv = cv
        # 设定交叉验证的折数，默认为5

        self.shuffle = shuffle
        # 设定是否在拟合时对数据进行洗牌，默认为True

        self.random_state = random_state
        # 设定随机数生成器的种子，用于复现随机过程，默认为None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the :class:`TargetEncoder` to X and y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : array-like of shape (n_samples,)
            The target data used to encode the categories.

        Returns
        -------
        self : object
            Fitted encoder.
        """
        # 将TargetEncoder适配到输入的X和y上，学习编码规则

        self._fit_encodings_all(X, y)
        # 调用内部方法 _fit_encodings_all 来进行真正的拟合过程

        return self
    def fit_transform(self, X, y):
        """Fit :class:`TargetEncoder` and transform X with the target encoding.

        .. note::
            `fit(X, y).transform(X)` does not equal `fit_transform(X, y)` because a
            :term:`cross fitting` scheme is used in `fit_transform` for encoding.
            See the :ref:`User Guide <target_encoder>`. for details.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : array-like of shape (n_samples,)
            The target data used to encode the categories.

        Returns
        -------
        X_trans : ndarray of shape (n_samples, n_features) or \
                    (n_samples, (n_features * n_classes))
            Transformed input.
        """
        from ..model_selection import KFold, StratifiedKFold  # avoid circular import

        # Encode X, identify known values, encode y, and count unique categories
        X_ordinal, X_known_mask, y_encoded, n_categories = self._fit_encodings_all(X, y)

        # Define cross-validation strategy
        if self.target_type_ == "continuous":
            cv = KFold(self.cv, shuffle=self.shuffle, random_state=self.random_state)
        else:
            cv = StratifiedKFold(
                self.cv, shuffle=self.shuffle, random_state=self.random_state
            )

        # Prepare output array based on target type
        if self.target_type_ == "multiclass":
            X_out = np.empty(
                (X_ordinal.shape[0], X_ordinal.shape[1] * len(self.classes_)),
                dtype=np.float64,
            )
        else:
            X_out = np.empty_like(X_ordinal, dtype=np.float64)

        # Perform cross-validation
        for train_idx, test_idx in cv.split(X, y):
            # Select training data and corresponding encoded targets
            X_train, y_train = X_ordinal[train_idx, :], y_encoded[train_idx]
            # Compute mean of encoded targets
            y_train_mean = np.mean(y_train, axis=0)

            # Fit encoding based on target type
            if self.target_type_ == "multiclass":
                encodings = self._fit_encoding_multiclass(
                    X_train,
                    y_train,
                    n_categories,
                    y_train_mean,
                )
            else:
                encodings = self._fit_encoding_binary_or_continuous(
                    X_train,
                    y_train,
                    n_categories,
                    y_train_mean,
                )
            
            # Transform X using fitted encodings and update X_out
            self._transform_X_ordinal(
                X_out,
                X_ordinal,
                ~X_known_mask,
                test_idx,
                encodings,
                y_train_mean,
            )
        
        # Return the transformed input
        return X_out
    def transform(self, X):
        """Transform X with the target encoding.

        .. note::
            `fit(X, y).transform(X)` does not equal `fit_transform(X, y)` because a
            :term:`cross fitting` scheme is used in `fit_transform` for encoding.
            See the :ref:`User Guide <target_encoder>` for details.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        Returns
        -------
        X_trans : ndarray of shape (n_samples, n_features) or \
                    (n_samples, (n_features * n_classes))
            Transformed input.
        """
        # 使用 `_transform` 方法进行数据转换，处理未知值为 'ignore'，允许无限大数值
        X_ordinal, X_known_mask = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )

        # 如果目标类型是多类别（multiclass），则根据类别数目扩展轴为 1，否则保持原有形状
        if self.target_type_ == "multiclass":
            X_out = np.empty(
                (X_ordinal.shape[0], X_ordinal.shape[1] * len(self.classes_)),
                dtype=np.float64,
            )
        else:
            X_out = np.empty_like(X_ordinal, dtype=np.float64)

        # 使用 `_transform_X_ordinal` 方法对 `X_out` 进行填充
        self._transform_X_ordinal(
            X_out,
            X_ordinal,
            ~X_known_mask,
            slice(None),
            self.encodings_,
            self.target_mean_,
        )
        # 返回转换后的数据数组
        return X_out
    # 在所有数据上拟合目标编码
    def _fit_encodings_all(self, X, y):
        """Fit a target encoding with all the data."""
        # 避免循环导入，导入必要的模块
        from ..preprocessing import (
            LabelBinarizer,
            LabelEncoder,
        )

        # 检查输入特征 X 和目标 y 的长度是否一致
        check_consistent_length(X, y)

        # 调用 _fit 方法拟合编码，处理未知值并允许 NaN
        self._fit(X, handle_unknown="ignore", force_all_finite="allow-nan")

        # 如果目标类型是自动检测
        if self.target_type == "auto":
            # 支持的目标类型包括二元、多类和连续
            accepted_target_types = ("binary", "multiclass", "continuous")
            # 推断目标类型
            inferred_type_of_target = type_of_target(y, input_name="y")
            # 如果推断出的目标类型不在支持的类型列表中，抛出错误
            if inferred_type_of_target not in accepted_target_types:
                raise ValueError(
                    "Unknown label type: Target type was inferred to be "
                    f"{inferred_type_of_target!r}. Only {accepted_target_types} are "
                    "supported."
                )
            # 将推断得到的目标类型作为实际的目标类型
            self.target_type_ = inferred_type_of_target
        else:
            # 否则，直接使用设定的目标类型
            self.target_type_ = self.target_type

        # 初始化类别属性为空
        self.classes_ = None

        # 根据目标类型处理 y
        if self.target_type_ == "binary":
            # 对二元目标进行标签编码
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            self.classes_ = label_encoder.classes_
        elif self.target_type_ == "multiclass":
            # 对多类目标进行标签二值化处理
            label_binarizer = LabelBinarizer()
            y = label_binarizer.fit_transform(y)
            self.classes_ = label_binarizer.classes_
        else:
            # 否则，处理连续型目标变量
            y = _check_y(y, y_numeric=True, estimator=self)

        # 计算目标变量 y 的均值
        self.target_mean_ = np.mean(y, axis=0)

        # 对输入特征 X 进行变换，处理未知值并允许 NaN，得到有序的 X 和已知掩码
        X_ordinal, X_known_mask = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )

        # 计算每个特征的类别数量，并将结果转换为 numpy 数组
        n_categories = np.fromiter(
            (len(category_for_feature) for category_for_feature in self.categories_),
            dtype=np.int64,
            count=len(self.categories_),
        )

        # 根据目标类型选择合适的编码方法
        if self.target_type_ == "multiclass":
            # 如果目标类型是多类，使用多类编码方法拟合
            encodings = self._fit_encoding_multiclass(
                X_ordinal,
                y,
                n_categories,
                self.target_mean_,
            )
        else:
            # 否则，使用二元或连续目标编码方法拟合
            encodings = self._fit_encoding_binary_or_continuous(
                X_ordinal,
                y,
                n_categories,
                self.target_mean_,
            )

        # 将拟合得到的编码结果保存在 encodings_ 属性中
        self.encodings_ = encodings

        # 返回拟合后的结果：有序的 X，已知掩码，处理后的 y，以及类别数量
        return X_ordinal, X_known_mask, y, n_categories
    ):
        """学习目标编码。"""
        # 如果平滑参数为 "auto"，计算目标变量的方差，并使用快速自动平滑方法学习编码
        if self.smooth == "auto":
            y_variance = np.var(y)
            encodings = _fit_encoding_fast_auto_smooth(
                X_ordinal,
                y,
                n_categories,
                target_mean,
                y_variance,
            )
        else:
            # 否则，使用指定的平滑方法学习编码
            encodings = _fit_encoding_fast(
                X_ordinal,
                y,
                n_categories,
                self.smooth,
                target_mean,
            )
        # 返回学习到的编码结果
        return encodings

    def _fit_encoding_multiclass(self, X_ordinal, y, n_categories, target_mean):
        """学习多类别编码。

        为每个类别学习编码，然后重新排序编码，以便相同的特征被分组在一起。
        `reorder_index` 可以将编码从：
        f0_c0, f1_c0, f0_c1, f1_c1, f0_c2, f1_c2
        重新排序为：
        f0_c0, f0_c1, f0_c2, f1_c0, f1_c1, f1_c2
        """
        # 获取特征数量和类别数量
        n_features = self.n_features_in_
        n_classes = len(self.classes_)

        encodings = []
        # 对每个类别进行循环，学习对应的二元或连续编码
        for i in range(n_classes):
            y_class = y[:, i]
            encoding = self._fit_encoding_binary_or_continuous(
                X_ordinal,
                y_class,
                n_categories,
                target_mean[i],
            )
            # 将学习到的编码加入到总编码列表中
            encodings.extend(encoding)

        # 生成重新排序的索引，使得相同特征在编码列表中被正确分组
        reorder_index = (
            idx
            for start in range(n_features)
            for idx in range(start, (n_classes * n_features), n_features)
        )
        # 按照重新排序的索引返回编码列表
        return [encodings[idx] for idx in reorder_index]

    def _transform_X_ordinal(
        self,
        X_out,
        X_ordinal,
        X_unknown_mask,
        row_indices,
        encodings,
        target_mean,
    ):
        """
        Transform X_ordinal using encodings.

        In the multiclass case, `X_ordinal` and `X_unknown_mask` have column
        (axis=1) size `n_features`, while `encodings` has length of size
        `n_features * n_classes`. `feat_idx` deals with this by repeating
        feature indices by `n_classes` E.g., for 3 features, 2 classes:
        0,0,1,1,2,2

        Additionally, `target_mean` is of shape (`n_classes`,) so `mean_idx`
        cycles through 0 to `n_classes` - 1, `n_features` times.
        """
        if self.target_type_ == "multiclass":
            # Determine the number of classes
            n_classes = len(self.classes_)
            # Iterate through each encoding
            for e_idx, encoding in enumerate(encodings):
                # Repeat feature indices by n_classes
                feat_idx = e_idx // n_classes
                # Cycle through each class
                mean_idx = e_idx % n_classes
                # Transform X_ordinal using encodings and assign to X_out
                X_out[row_indices, e_idx] = encoding[X_ordinal[row_indices, feat_idx]]
                # Assign target_mean to X_out for X_unknown_mask entries
                X_out[X_unknown_mask[:, feat_idx], e_idx] = target_mean[mean_idx]
        else:
            # Handle the binary or continuous target case
            for e_idx, encoding in enumerate(encodings):
                # Transform X_ordinal using encodings and assign to X_out
                X_out[row_indices, e_idx] = encoding[X_ordinal[row_indices, e_idx]]
                # Assign target_mean to X_out for X_unknown_mask entries
                X_out[X_unknown_mask[:, e_idx], e_idx] = target_mean

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names. `feature_names_in_` is used unless it is
            not defined, in which case the following input feature names are
            generated: `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            When `type_of_target_` is "multiclass" the names are of the format
            '<feature_name>_<class_name>'.
        """
        # Ensure the estimator is fitted and get input feature names
        check_is_fitted(self, "n_features_in_")
        feature_names = _check_feature_names_in(self, input_features)
        if self.target_type_ == "multiclass":
            # Generate feature names for each class and feature
            feature_names = [
                f"{feature_name}_{class_name}"
                for feature_name in feature_names
                for class_name in self.classes_
            ]
            return np.asarray(feature_names, dtype=object)
        else:
            return feature_names

    def _more_tags(self):
        """
        Provide additional tags for the estimator.

        Returns
        -------
        tags : dict
            Dictionary containing additional tags. In this case, indicates that
            the estimator requires y values during operation.
        """
        return {
            "requires_y": True,
        }
```