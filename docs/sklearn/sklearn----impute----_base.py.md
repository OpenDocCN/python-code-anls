# `D:\src\scipysrc\scikit-learn\sklearn\impute\_base.py`

```
        self, *, missing_values=np.nan, add_indicator=False, keep_empty_features=False
    ):
        # 初始化 imputer 对象，设置缺失值、是否添加指示器、是否保留空特征等参数
        super().__init__()
        self.missing_values = missing_values
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _validate_input(self, X):
        """Validate input X and ensure it has the correct data type."""
        # 检查输入 X 的数据类型是否为浮点数或密集矩阵
        if not isinstance(X, (np.ndarray, sp.csr_matrix, sp.csc_matrix)):
            raise TypeError(
                f"Input X should be a numpy array or sparse matrix, "
                f"got {type(X)} instead."
            )
        if ma.isMaskedArray(X):
            raise ValueError("Masked arrays are not supported.")
        if hasattr(X, "dtype") and X.dtype.kind not in "fiub":
            raise ValueError(
                f"Data with dtype {X.dtype} does not comply with expected types: "
                f"float64, int64, uint8, or boolean."
            )

    def _more_tags(self):
        """Define tags related to the estimator's capabilities."""
        # 定义与估计器功能相关的标签，如支持的输入输出格式
        return {"requires_y": False}

    def _check_output(self, Xt):
        """Validate the output after imputation."""
        # 检查输出 Xt 的数据类型和形状是否与输入 X 一致
        self._validate_input(Xt)
        if Xt.shape[1] != self.statistics_.shape[0]:
            raise ValueError(
                f"Xt has {Xt.shape[1]} features, "
                f"but this Imputer is expecting {self.statistics_.shape[0]} features"
            )
        return Xt

    def _check_statistics(self, X):
        """Check the statistics computed across features."""
        # 检查统计信息（如均值、中位数等）是否与输入 X 的特征数一致
        if getattr(self, "statistics_", None) is None:
            raise AttributeError("Statistics for imputation have not been computed.")
        if X.shape[1] != self.statistics_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but this Imputer is expecting {self.statistics_.shape[0]} features"
            )

    def _check_fill_value(self, value):
        """Check the fill value for missing values."""
        # 检查填充缺失值的值是否为有效值
        if not isinstance(value, (numbers.Real, str)):
            raise TypeError(
                f"Fill value {value} does not comply with expected types: "
                f"float, int, string, or None."
            )

    def _check_add_indicator(self):
        """Check the add_indicator parameter."""
        # 检查是否添加指示器参数是否为布尔类型
        if not isinstance(self.add_indicator, bool):
            raise TypeError("add_indicator should be a boolean value.")

    def _check_keep_empty_features(self):
        """Check the keep_empty_features parameter."""
        # 检查是否保留空特征参数是否为布尔类型
        if not isinstance(self.keep_empty_features, bool):
            raise TypeError("keep_empty_features should be a boolean value.")
    ):
        self.missing_values = missing_values
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features



        """Initialize the MissingIndicatorMixin.

        Parameters:
        missing_values : int or float, default=np.nan
            The placeholder for the missing values.
        add_indicator : bool, default=False
            Whether to add a missing indicator for each feature.
        keep_empty_features : bool, default=False
            Whether to keep features with only missing values.
        """



    def _fit_indicator(self, X):



        """Fit a MissingIndicator.

        Parameters:
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data used to compute the missing values indicator.
        """



        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values, error_on_new=False
            )
            self.indicator_._fit(X, precomputed=True)
        else:
            self.indicator_ = None



        """Fit the missing values indicator on the input data X.

        Parameters:
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data used to compute the missing values indicator.
        """



    def _transform_indicator(self, X):



        """Transform the input data X to include the missing indicator.

        Parameters:
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to transform.

        Returns:
        X_transformed : {array-like, sparse matrix}, shape (n_samples, n_features + n_features_missing)
            Transformed array with the missing indicator added.
        """



        if self.add_indicator:
            if not hasattr(self, "indicator_"):
                raise ValueError(
                    "Make sure to call _fit_indicator before _transform_indicator"
                )
            return self.indicator_.transform(X)



        """Transform the input data X to include the missing indicator.

        Raises:
        ValueError:
            If _fit_indicator has not been called before _transform_indicator.

        Returns:
        X_transformed : {array-like, sparse matrix}, shape (n_samples, n_features + n_features_missing)
            Transformed array with the missing indicator added.
        """



    def _concatenate_indicator(self, X_imputed, X_indicator):



        """Concatenate the indicator mask with the imputed data.

        Parameters:
        X_imputed : {array-like, sparse matrix}, shape (n_samples, n_features)
            The imputed data.
        X_indicator : {array-like, sparse matrix}, shape (n_samples, n_features_missing)
            The missing indicator mask.

        Returns:
        X_combined : {array-like, sparse matrix}, shape (n_samples, n_features + n_features_missing)
            Concatenated array of imputed data and missing indicator.
        """



        if not self.add_indicator:
            return X_imputed



        """Return the imputed data X_imputed if add_indicator is False.

        Returns:
        X_imputed : {array-like, sparse matrix}, shape (n_samples, n_features)
            The original imputed data.
        """



        if sp.issparse(X_imputed):
            # sp.hstack may result in different formats between sparse arrays and
            # matrices; specify the format to keep consistent behavior
            hstack = partial(sp.hstack, format=X_imputed.format)
        else:
            hstack = np.hstack



        """Define the hstack function based on the sparsity of X_imputed.

        Returns:
        hstack : function
            The appropriate hstack function for concatenating arrays.
        """



        if X_indicator is None:
            raise ValueError(
                "Data from the missing indicator are not provided. Call "
                "_fit_indicator and _transform_indicator in the imputer "
                "implementation."
            )



        """Check if X_indicator is None and raise ValueError if True.

        Raises:
        ValueError:
            If X_indicator is None, indicating that _fit_indicator and 
            _transform_indicator were not called before concatenation.
        """



        return hstack((X_imputed, X_indicator))



        """Concatenate X_imputed and X_indicator using the defined hstack function.

        Returns:
        X_combined : {array-like, sparse matrix}, shape (n_samples, n_features + n_features_missing)
            Concatenated array of imputed data and missing indicator.
        """



    def _concatenate_indicator_feature_names_out(self, names, input_features):



        """Concatenate feature names with the indicator feature names.

        Parameters:
        names : list
            List of feature names.
        input_features : list
            List of input features.

        Returns:
        concatenated_names : ndarray
            Concatenated array of feature names and indicator feature names.
        """



        if not self.add_indicator:
            return names



        """Return the feature names if add_indicator is False.

        Returns:
        names : list
            List of feature names.
        """



        indicator_names = self.indicator_.get_feature_names_out(input_features)
        return np.concatenate([names, indicator_names])



        """Concatenate feature names with the indicator feature names.

        Returns:
        concatenated_names : ndarray
            Concatenated array of feature names and indicator feature names.
        """



    def _more_tags(self):



        """Provide additional tags for the imputer.

        Returns:
        tags : dict
            Dictionary of additional tags, e.g., {"allow_nan": True/False}.
        """
class SimpleImputer(_BaseImputer):
    """Univariate imputer for completing missing values with simple strategies.

    Replace missing values using a descriptive statistic (e.g. mean, median, or
    most frequent) along each column, or using a constant value.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20
       `SimpleImputer` replaces the previous `sklearn.preprocessing.Imputer`
       estimator which is now removed.

    Parameters
    ----------
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.

    strategy : str or Callable, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.
        - If an instance of Callable, then replace missing values using the
          scalar statistic returned by running the callable over a dense 1d
          array containing non-missing values of each column.

        .. versionadded:: 0.20
           strategy="constant" for fixed value imputation.

        .. versionadded:: 1.5
           strategy=callable for custom value imputation.

    fill_value : str or numerical value, default=None
        When strategy == "constant", `fill_value` is used to replace all
        occurrences of missing_values. For string or object data types,
        `fill_value` must be a string.
        If `None`, `fill_value` will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If `X` is not an array of floating values;
        - If `X` is encoded as a CSR matrix;
        - If `add_indicator=True`.
    """
    # 继承自 _BaseImputer 类，用于单变量的缺失值填充
    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None, copy=True):
        # 调用父类的构造方法
        super().__init__(missing_values=missing_values, strategy=strategy, fill_value=fill_value, copy=copy)
        # 设置类的文档字符串，描述了如何用简单的策略来填充缺失值
        self.__doc__ = self.__class__.__doc__
    add_indicator : bool, default=False
        # 是否添加缺失指示器，如果为True，则在缺失值填充后添加一个MissingIndicator转换
        # 这允许预测估计器考虑缺失值，即使进行了填充
        # 如果特征在拟合/训练时没有缺失值，则即使在转换/测试时存在缺失值，该特征也不会出现在缺失指示器中

    keep_empty_features : bool, default=False
        # 如果为True，则在调用`transform`时，当`fit`调用时由纯缺失值组成的特征将在结果中返回
        # 填充值始终为`0`，除非`strategy="constant"`，在这种情况下将使用`fill_value`

        .. versionadded:: 1.2

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        # 每个特征的填充值
        # 计算统计信息可能导致`np.nan`值
        # 在`transform`期间，与`np.nan`统计信息对应的特征将被丢弃

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        # 用于添加缺失值的二进制指示器
        # 如果`add_indicator=False`，则为`None`

    n_features_in_ : int
        # 在`fit`期间看到的特征数量

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在`fit`期间看到的特征的名称
        # 仅在`X`具有全部为字符串的特征名称时定义

        .. versionadded:: 1.0

    See Also
    --------
    IterativeImputer : 从所有其他特征的缺失值中估计要填充的每个特征的多变量填充器
    KNNImputer : 使用最近样本估计缺失特征的多变量填充器

    Notes
    -----
    如果策略不是`"constant"`，则在`transform`时仅包含在`fit`时仅包含缺失值的列将被丢弃

    在预测环境中，当与弱学习器关联时，简单填充通常表现不佳
    但是，对于强学习器，它可以导致与复杂填充（如`IterativeImputer`或`KNNImputer`）一样好甚至更好的性能

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    SimpleImputer()
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]

    有关更详细的示例，请参见
    :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.
    """
    # 定义参数约束字典，继承自_BaseImputer的参数约束，并增加了额外的约束项
    _parameter_constraints: dict = {
        **_BaseImputer._parameter_constraints,
        "strategy": [
            StrOptions({"mean", "median", "most_frequent", "constant"}),  # 策略参数应为指定字符串集合
            callable,  # 策略参数也可以是可调用对象
        ],
        "fill_value": "no_validation",  # 填充值可以是任何对象，无需验证
        "copy": ["boolean"],  # 复制参数应为布尔值
    }

    # 初始化方法，设置缺失值处理器的初始参数
    def __init__(
        self,
        *,
        missing_values=np.nan,  # 缺失值默认为 np.nan
        strategy="mean",  # 缺失值填充策略，默认为 "mean"
        fill_value=None,  # 自定义填充值，默认为 None
        copy=True,  # 是否复制输入数据，默认为 True
        add_indicator=False,  # 是否添加指示器特征，默认为 False
        keep_empty_features=False,  # 是否保留空特征，默认为 False
    ):
        # 调用父类的初始化方法，设置通用参数
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        self.strategy = strategy  # 设置策略参数
        self.fill_value = fill_value  # 设置填充值参数
        self.copy = copy  # 设置复制参数

    # 使用装饰器定义的上下文环境，跳过嵌套验证优先
    @_fit_context(prefer_skip_nested_validation=True)
    # 拟合方法，用于在数据上拟合缺失值处理器
    def fit(self, X, y=None):
        """Fit the imputer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_input(X, in_fit=True)  # 验证输入数据的有效性，用于拟合过程

        # 默认的填充值取决于输入数据类型：数值类型默认为 0，其他类型默认为 "missing_value"
        if self.fill_value is None:
            if X.dtype.kind in ("i", "u", "f"):
                fill_value = 0
            else:
                fill_value = "missing_value"
        else:
            fill_value = self.fill_value

        # 根据数据稀疏性选择相应的拟合方法，设置self.statistics_
        if sp.issparse(X):
            self.statistics_ = self._sparse_fit(
                X, self.strategy, self.missing_values, fill_value
            )
        else:
            self.statistics_ = self._dense_fit(
                X, self.strategy, self.missing_values, fill_value
            )

        return self  # 返回已拟合的估计器对象
    def _sparse_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on sparse data."""
        # 获取稀疏数据中的缺失值掩码
        missing_mask = _get_mask(X, missing_values)
        # 获取掩码中的数据部分
        mask_data = missing_mask.data
        # 计算每列隐式零值的数量
        n_implicit_zeros = X.shape[0] - np.diff(X.indptr)

        # 创建一个与列数相同大小的空数组，用于存储统计数据
        statistics = np.empty(X.shape[1])

        # 根据策略填充统计数据数组
        if strategy == "constant":
            # 对于常数策略，使用 fill_value 填充 statistics 数组中的每一列
            statistics.fill(fill_value)
        else:
            # 遍历每一列数据
            for i in range(X.shape[1]):
                # 获取当前列的数据
                column = X.data[X.indptr[i] : X.indptr[i + 1]]
                # 获取当前列的掩码数据
                mask_column = mask_data[X.indptr[i] : X.indptr[i + 1]]
                # 从列中移除掩码中为真的元素
                column = column[~mask_column]

                # 合并显式和隐式零值
                mask_zeros = _get_mask(column, 0)
                column = column[~mask_zeros]
                n_explicit_zeros = mask_zeros.sum()
                n_zeros = n_implicit_zeros[i] + n_explicit_zeros

                if len(column) == 0 and self.keep_empty_features:
                    # 如果希望保留只有缺失值的列，则将 statistics[i] 设置为 0
                    statistics[i] = 0
                else:
                    # 根据策略计算统计数据
                    if strategy == "mean":
                        s = column.size + n_zeros
                        statistics[i] = np.nan if s == 0 else column.sum() / s
                    elif strategy == "median":
                        statistics[i] = _get_median(column, n_zeros)
                    elif strategy == "most_frequent":
                        statistics[i] = _most_frequent(column, 0, n_zeros)
                    elif isinstance(strategy, Callable):
                        statistics[i] = self.strategy(column)

        # 调用父类方法，处理缺失值指示器的拟合
        super()._fit_indicator(missing_mask)

        # 返回统计数据数组
        return statistics
    def _dense_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on dense data."""
        # 获取缺失值的掩码
        missing_mask = _get_mask(X, missing_values)
        # 使用缺失值掩码创建被屏蔽的数组
        masked_X = ma.masked_array(X, mask=missing_mask)

        # 调用父类的 _fit_indicator 方法来处理缺失值的指示器
        super()._fit_indicator(missing_mask)

        # Mean 策略
        if strategy == "mean":
            # 计算每列的均值，返回未屏蔽元素的数据
            mean_masked = np.ma.mean(masked_X, axis=0)
            # 处理警告："Warning: converting a masked element to nan."
            mean = np.ma.getdata(mean_masked)
            # 将屏蔽的元素设置为 0 或 NaN，取决于 self.keep_empty_features
            mean[np.ma.getmask(mean_masked)] = 0 if self.keep_empty_features else np.nan

            return mean

        # Median 策略
        elif strategy == "median":
            # 计算每列的中位数，返回未屏蔽元素的数据
            median_masked = np.ma.median(masked_X, axis=0)
            # 处理警告："Warning: converting a masked element to nan."
            median = np.ma.getdata(median_masked)
            # 将屏蔽的元素设置为 0 或 NaN，取决于 self.keep_empty_features
            median[np.ma.getmaskarray(median_masked)] = (
                0 if self.keep_empty_features else np.nan
            )

            return median

        # Most frequent 策略
        elif strategy == "most_frequent":
            # 避免使用 scipy.stats.mstats.mode，因为它需要额外的开销和较慢的基准性能。
            # 参见 Issue 14325 和 PR 14399 进行全面讨论。

            # 转置 X 和掩码，以便按列访问元素
            X = X.transpose()
            mask = missing_mask.transpose()

            # 初始化数组以存储每列的最频繁值
            if X.dtype.kind == "O":
                most_frequent = np.empty(X.shape[0], dtype=object)
            else:
                most_frequent = np.empty(X.shape[0])

            # 遍历 X 和 mask 的每一行
            for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
                # 将 row_mask 转换为逻辑反，并转换为布尔类型
                row_mask = np.logical_not(row_mask).astype(bool)
                # 获取未被屏蔽的行数据
                row = row[row_mask]
                # 如果行为空且 self.keep_empty_features 为真，则填充 0
                # 否则，计算行中的最频繁值并填充
                if len(row) == 0 and self.keep_empty_features:
                    most_frequent[i] = 0
                else:
                    most_frequent[i] = _most_frequent(row, np.nan, 0)

            return most_frequent

        # Constant 策略
        elif strategy == "constant":
            # 对于 constant 策略，使用 self.statistics_ 存储每列的 fill_value
            return np.full(X.shape[1], fill_value, dtype=X.dtype)

        # Custom 策略
        elif isinstance(strategy, Callable):
            # 初始化统计数组以存储每列的统计值
            statistics = np.empty(masked_X.shape[1])
            # 遍历每列，调用 self.strategy 处理未被屏蔽的数据
            for i in range(masked_X.shape[1]):
                statistics[i] = self.strategy(masked_X[:, i].compressed())
            return statistics
    # 将数据转换回原始表示形式的方法

    # 检查模型是否已拟合
    check_is_fitted(self)

    # 如果未设置 add_indicator=True，则无法执行反转操作，抛出错误
    if not self.add_indicator:
        raise ValueError(
            "'inverse_transform' works only when "
            "'SimpleImputer' is instantiated with "
            "'add_indicator=True'. "
            f"Got 'add_indicator={self.add_indicator}' "
            "instead."
        )

    # 计算具有缺失值二进制指示符的特征数量
    n_features_missing = len(self.indicator_.features_)
    # 计算没有缺失值指示符的特征数量
    non_empty_feature_count = X.shape[1] - n_features_missing
    # 复制被填充的数据部分，不包括缺失值指示符
    array_imputed = X[:, :non_empty_feature_count].copy()
    # 提取缺失值指示符的布尔掩码
    missing_mask = X[:, non_empty_feature_count:].astype(bool)

    # 原始特征数量为模型统计数据的长度
    n_features_original = len(self.statistics_)
    # 创建原始数据形状的零数组
    shape_original = (X.shape[0], n_features_original)
    X_original = np.zeros(shape_original)
    # 根据缺失值指示符的位置，将缺失值的布尔掩码填充到原始数据中
    X_original[:, self.indicator_.features_] = missing_mask
    # 创建整体布尔掩码
    full_mask = X_original.astype(bool)

    # 将填充的数据逐列复制回原始数据中
    imputed_idx, original_idx = 0, 0
    while imputed_idx < len(array_imputed.T):
        if not np.all(X_original[:, original_idx]):
            X_original[:, original_idx] = array_imputed.T[imputed_idx]
            imputed_idx += 1
            original_idx += 1
        else:
            original_idx += 1

    # 将完整的布尔掩码位置的数据填充为缺失值
    X_original[full_mask] = self.missing_values

    # 返回原始的 X 数据，包含之前的缺失值
    return X_original
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        # 确保模型已经拟合，获取输入特征的数量
        check_is_fitted(self, "n_features_in_")
        
        # 根据输入的特征名列表，进行验证和处理
        input_features = _check_feature_names_in(self, input_features)
        
        # 生成一个布尔掩码，标记缺失值的位置
        non_missing_mask = np.logical_not(_get_mask(self.statistics_, np.nan))
        
        # 根据掩码从输入特征列表中选择非缺失值的特征名
        names = input_features[non_missing_mask]
        
        # 调用内部方法，将选择的特征名和输入特征列表连接起来，生成输出特征名
        return self._concatenate_indicator_feature_names_out(names, input_features)
# 定义一个自定义的转换器类，用于生成缺失值的二进制指示器

class MissingIndicator(TransformerMixin, BaseEstimator):
    """Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`~sklearn.pipeline.Pipeline` consisting of transformers and a
    classifier, but rather could be added using a
    :class:`~sklearn.pipeline.FeatureUnion` or
    :class:`~sklearn.compose.ColumnTransformer`.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.
    
    features : {'missing-only', 'all'}, default='missing-only'
        Whether the imputer mask should represent all or a subset of
        features.
        
        - If `'missing-only'` (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If `'all'`, the imputer mask will represent all features.
    
    sparse : bool or 'auto', default='auto'
        Whether the imputer mask format should be sparse or dense.
        
        - If `'auto'` (default), the imputer mask will be of same type as
          input.
        - If `True`, the imputer mask will be a sparse matrix.
        - If `False`, the imputer mask will be a numpy array.
    
    error_on_new : bool, default=True
        If `True`, :meth:`transform` will raise an error when there are
        features with missing values that have no missing values in
        :meth:`fit`. This is applicable only when `features='missing-only'`.
    
    Attributes
    ----------
    features_ : ndarray of shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling
        :meth:`transform`. They are computed during :meth:`fit`. If
        `features='all'`, `features_` is equal to `range(n_features)`.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        
        .. versionadded:: 1.0
    
    See Also
    --------
    SimpleImputer : Univariate imputation of missing values.
    IterativeImputer : Multivariate imputation of missing values.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)
    
    """
    
    def __init__(self, missing_values=np.nan, features='missing-only',
                 sparse='auto', error_on_new=True):
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new
    
    def fit(self, X, y=None):
        """Fit the MissingIndicator transformer to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        
        y : None
            Ignored variable.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 计算在fit过程中出现缺失值的特征的索引
        if self.features == 'missing-only':
            # 如果仅考虑有缺失值的特征，找出这些特征的索引
            self.features_ = np.where(np.any(pd.isna(X), axis=0))[0]
        else:
            # 考虑所有特征，直接使用所有特征的索引
            self.features_ = np.arange(X.shape[1])
        
        self.n_features_in_ = X.shape[1]
        
        # 如果输入X具有全部为字符串的特征名，记录这些特征名
        if isinstance(X, pd.DataFrame) and X.columns.is_all_strings:
            self.feature_names_in_ = np.asarray(X.columns)
        
        return self
    
    def transform(self, X):
        """Transform X by adding a binary indicator for missing values.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        
        Returns
        -------
        X_indicator : {array-like, sparse matrix}
            A matrix of shape (n_samples, n_features) containing binary
            indicators for missing values.
        """
        # 创建一个与X形状相同的零矩阵或稀疏矩阵，用于存储缺失值的二进制指示器
        if self.sparse == 'auto':
            sparse = sp.issparse(X)
        else:
            sparse = self.sparse
        
        if sparse:
            # 使用稀疏矩阵表示，初始化一个稀疏矩阵
            X_indicator = sp.csr_matrix((X.shape[0], self.n_features_in_), dtype=bool)
        else:
            # 使用密集数组表示，初始化一个全零数组
            X_indicator = np.zeros((X.shape[0], self.n_features_in_), dtype=bool)
        
        # 根据fit时记录的特征索引，为X中的每个样本添加缺失值指示器
        for idx in self.features_:
            X_indicator[:, idx] = pd.isna(X[:, idx])
        
        return X_indicator
    # 创建一个 MissingIndicator 实例，用于检测输入数据中的缺失值
    MissingIndicator()

    # 使用 MissingIndicator 实例对输入数据 X2 进行转换，生成转换后的结果 X2_tr
    >>> X2_tr = indicator.transform(X2)

    # 打印转换后的结果 X2_tr，展示其中的布尔值数组
    >>> X2_tr
    array([[False,  True],  # 第一个样本中第二个特征是缺失的
           [ True, False],  # 第二个样本中第一个特征是缺失的
           [False, False]])  # 第三个样本中没有缺失特征
    """

    # 定义参数约束字典 _parameter_constraints
    _parameter_constraints: dict = {
        "missing_values": [MissingValues()],  # 缺失值参数的约束，使用 MissingValues 类
        "features": [StrOptions({"missing-only", "all"})],  # 特征参数的约束，可以是 "missing-only" 或 "all"
        "sparse": ["boolean", StrOptions({"auto"})],  # 稀疏性参数的约束，可以是布尔值或 "auto"
        "error_on_new": ["boolean"],  # 是否在新特征出现时报错的约束，为布尔值
    }

    # 定义 MissingIndicator 类的初始化方法
    def __init__(
        self,
        *,
        missing_values=np.nan,  # 缺失值的默认值为 NaN
        features="missing-only",  # 特征默认只考虑缺失值
        sparse="auto",  # 稀疏性默认自动判断
        error_on_new=True,  # 默认在出现新特征时报错
    ):
        self.missing_values = missing_values  # 初始化缺失值属性
        self.features = features  # 初始化特征属性
        self.sparse = sparse  # 初始化稀疏性属性
        self.error_on_new = error_on_new  # 初始化报错属性

    # 定义获取含缺失值特征信息的方法 _get_missing_features_info
    def _get_missing_features_info(self, X):
        """Compute the imputer mask and the indices of the features
        containing missing values.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data with missing values. Note that `X` has been
            checked in :meth:`fit` and :meth:`transform` before to call this
            function.

        Returns
        -------
        imputer_mask : {ndarray, sparse matrix} of shape \
        (n_samples, n_features)
            The imputer mask of the original data.

        features_with_missing : ndarray of shape (n_features_with_missing)
            The features containing missing values.
        """
        # 如果没有预计算的数据，则计算缺失值的掩码
        if not self._precomputed:
            imputer_mask = _get_mask(X, self.missing_values)
        else:
            imputer_mask = X

        # 如果输入数据 X 是稀疏矩阵
        if sp.issparse(X):
            imputer_mask.eliminate_zeros()  # 清除稀疏矩阵中的零元素

            # 如果特征选择为 "missing-only"
            if self.features == "missing-only":
                n_missing = imputer_mask.getnnz(axis=0)  # 统计每列中非零元素的个数

            # 如果稀疏性参数为 False，则转换为密集矩阵
            if self.sparse is False:
                imputer_mask = imputer_mask.toarray()
            elif imputer_mask.format == "csr":
                imputer_mask = imputer_mask.tocsc()
        else:
            # 如果不是稀疏矩阵
            if not self._precomputed:
                imputer_mask = _get_mask(X, self.missing_values)
            else:
                imputer_mask = X

            # 如果特征选择为 "missing-only"
            if self.features == "missing-only":
                n_missing = imputer_mask.sum(axis=0)  # 沿列求和，统计缺失值个数

            # 如果稀疏性参数为 True，则转换为稀疏矩阵
            if self.sparse is True:
                imputer_mask = sp.csc_matrix(imputer_mask)

        # 如果特征选择为 "all"，则返回所有特征的索引，否则返回含缺失值的特征索引
        if self.features == "all":
            features_indices = np.arange(X.shape[1])
        else:
            features_indices = np.flatnonzero(n_missing)

        return imputer_mask, features_indices
    def _validate_input(self, X, in_fit):
        # 如果 missing_values 不是标量 NaN，则强制将所有数据视为有限的
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        # 验证输入数据 X 的有效性，根据参数设置进行验证
        X = self._validate_data(
            X,
            reset=in_fit,
            accept_sparse=("csc", "csr"),
            dtype=None,
            force_all_finite=force_all_finite,
        )

        # 检查输入数据 X 的数据类型是否符合预期
        _check_inputs_dtype(X, self.missing_values)

        # 如果输入数据 X 的数据类型不在 ('i', 'u', 'f', 'O') 中，抛出 ValueError
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError(
                "MissingIndicator does not support data with "
                "dtype {0}. Please provide either a numeric array"
                " (with a floating point or integer dtype) or "
                "categorical data represented either as an array "
                "with integer dtype or an array of string values "
                "with an object dtype.".format(X.dtype)
            )

        # 如果 X 是稀疏矩阵并且 missing_values 等于 0，则抛出 ValueError
        if sp.issparse(X) and self.missing_values == 0:
            raise ValueError(
                "Sparse input with missing_values=0 is "
                "not supported. Provide a dense "
                "array instead."
            )

        return X

    def _fit(self, X, y=None, precomputed=False):
        """Fit the transformer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            If `precomputed=True`, then `X` is a mask of the input data.

        precomputed : bool
            Whether the input data is a mask.

        Returns
        -------
        imputer_mask : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The imputer mask of the original data.
        """
        # 如果 precomputed 为 True，则验证输入数据 X 是否为布尔掩码
        if precomputed:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")
            self._precomputed = True
        else:
            self._precomputed = False

        # 如果没有预计算过，则进行输入数据 X 的验证
        if not self._precomputed:
            X = self._validate_input(X, in_fit=True)
        else:
            # 在预计算的情况下，仅在需要时创建 `n_features_in_`
            self._check_n_features(X, reset=True)

        # 记录输入数据 X 的特征数
        self._n_features = X.shape[1]

        # 获取输入数据 X 中缺失特征的信息
        missing_features_info = self._get_missing_features_info(X)
        # 将特征信息中的缺失特征标记存储在 self.features_ 中
        self.features_ = missing_features_info[1]

        # 返回输入数据 X 的缺失特征掩码
        return missing_features_info[0]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the transformer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # 调用内部方法 `_fit` 进行拟合操作
        self._fit(X, y)

        # 返回拟合后的自身对象
        return self

    def transform(self, X):
        """Generate missing values indicator for `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of `Xt`
            will be boolean.
        """
        # 检查是否已经拟合，若未拟合则引发异常
        check_is_fitted(self)

        # 若未预先计算，则验证输入数据 X
        if not self._precomputed:
            X = self._validate_input(X, in_fit=False)
        else:
            # 若预先计算标志为真，但输入数据 X 的数据类型不是布尔型，则引发异常
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")

        # 获取输入数据 X 的缺失特征信息和特征名称
        imputer_mask, features = self._get_missing_features_info(X)

        # 若特征选择为 "missing-only"
        if self.features == "missing-only":
            # 计算在 transform 中存在但在 fit 中不存在的特征差异
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            # 若出现新特征且错误标志为真，则引发异常
            if self.error_on_new and features_diff_fit_trans.size > 0:
                raise ValueError(
                    "The features {} have missing values "
                    "in transform but have no missing values "
                    "in fit.".format(features_diff_fit_trans)
                )

            # 若 fit 时的特征数小于总特征数，则只保留拟合时使用的特征的缺失指示器
            if self.features_.size < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]

        # 返回输入数据 X 的缺失指示器
        return imputer_mask

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """Generate missing values indicator for `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data to complete.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of `Xt`
            will be boolean.
        """
        # 执行拟合操作并返回结果
        imputer_mask = self._fit(X, y)

        # 若拟合时使用的特征数小于总特征数，则只保留拟合时使用的特征的缺失指示器
        if self.features_.size < self._n_features:
            imputer_mask = imputer_mask[:, self.features_]

        # 返回输入数据 X 的缺失指示器
        return imputer_mask
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        # Ensure that the model is fitted and get the number of input features
        check_is_fitted(self, "n_features_in_")
        
        # Validate and possibly transform the input feature names
        input_features = _check_feature_names_in(self, input_features)
        
        # Generate a prefix for the feature names based on the class name
        prefix = self.__class__.__name__.lower()
        
        # Create transformed feature names by appending the prefix to each feature name
        return np.asarray(
            [
                f"{prefix}_{feature_name}"
                for feature_name in input_features[self.features_]
            ],
            dtype=object,
        )

    def _more_tags(self):
        """
        Provide additional tags for the estimator.

        Returns
        -------
        tags : dict
            Dictionary containing additional tags.
        """
        # Return additional tags specifying properties of the estimator
        return {
            "allow_nan": True,
            "X_types": ["2darray", "string"],
            "preserves_dtype": [],
        }
```