# `D:\src\scipysrc\scikit-learn\sklearn\compose\_column_transformer.py`

```
    _ERR_MSG_1DCOLUMN = (
        "1D data passed to a transformer that expects 2D data. "
        "Try to specify the column selection as a list of one "
        "item instead of a scalar."
    )
    transformers : list of tuples
        # 一个元组列表，每个元组包含三个元素：name（名称）、transformer（转换器）、columns（列名或索引）
        List of (name, transformer, columns) tuples specifying the
        # 指定要应用于数据子集的转换器对象的列表

        name : str
            # 字符串，用于在Pipeline和FeatureUnion中设置转换器及其参数，可以使用set_params进行设置，也可以在网格搜索中进行搜索
            Like in Pipeline and FeatureUnion, this allows the transformer and
            its parameters to be set using ``set_params`` and searched in grid
            search.

        transformer : {'drop', 'passthrough'} or estimator
            # 可选值为{'drop', 'passthrough'}或估计器对象，估计器必须支持fit和transform方法
            Estimator must support :term:`fit` and :term:`transform`.
            # 'drop'和'passthrough'分别表示丢弃列或直接传递未转换的列

        columns :  str, array-like of str, int, array-like of int, \
                array-like of bool, slice or callable
            # 指定数据的第二维度的索引方式，整数表示位置列，字符串可以按DataFrame列名引用列
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name.
            # 一个标量字符串或整数用于转换器期望X是一维数组（向量），否则将传递二维数组给转换器
            A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            # 可调用对象传递输入数据`X`，可以返回上述任何一种方式
            A callable is passed the input data `X` and can return any of the
            above.
            # 若要通过名称或数据类型选择多列，可以使用make_column_selector

    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        # 默认情况下，只有`transformers`中指定的列进行转换并组合到输出中，未指定的列将被丢弃（默认为'drop'）
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        # 通过指定`remainder='passthrough'`，将自动传递所有在`fit`中未指定在`transformers`中的剩余列
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers`, but present in the data passed
        to `fit` will be automatically passed through.
        # 对于数据框，`fit`时未见过的额外列将从`transform`的输出中排除
        For dataframes,
        extra columns not seen during `fit` will be excluded from the output
        of `transform`.
        # 通过设置`remainder`为估计器，未指定的剩余列将使用`remainder`估计器进行处理
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator.
        # 估计器必须支持fit和transform方法
        The estimator must support :term:`fit` and :term:`transform`.
        # 使用此功能要求DataFrame列在`fit`和`transform`时具有相同的顺序
        Note that using this feature requires that the DataFrame columns
        input at :term:`fit` and :term:`transform` have identical order.

    sparse_threshold : float, default=0.3
        # 如果不同转换器的输出包含稀疏矩阵，则当总体密度低于此值时，这些矩阵将堆叠为稀疏矩阵
        If the output of the different transformers contains sparse matrices,
        these will be stacked as a sparse matrix if the overall density is
        lower than this value.
        # 使用`sparse_threshold=0`始终返回密集矩阵
        Use ``sparse_threshold=0`` to always return dense.
        # 当转换后的输出全部为密集数据时，堆叠结果将为密集矩阵，此关键字将被忽略
        When the transformed output consists of all dense data, the
        stacked result will be dense, and this keyword will be ignored.
    n_jobs : int, default=None
        # 并行运行的作业数量，默认为 None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    transformer_weights : dict, default=None
        # 特征变换器的乘法权重，默认为 None
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    verbose : bool, default=False
        # 是否输出详细信息，默认为 False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    verbose_feature_names_out : bool, default=True
        # 是否为输出的特征名称添加变换器名称前缀，默认为 True
        If True, :meth:`ColumnTransformer.get_feature_names_out` will prefix
        all feature names with the name of the transformer that generated that
        feature.
        If False, :meth:`ColumnTransformer.get_feature_names_out` will not
        prefix any feature names and will error if feature names are not
        unique.

        .. versionadded:: 1.0

    force_int_remainder_cols : bool, default=True
        # 是否强制将最后一个 "remainder" 变换器的列索引存储为整数，默认为 True
        Force the columns of the last entry of `transformers_`, which
        corresponds to the "remainder" transformer, to always be stored as
        indices (int) rather than column names (str). See description of the
        `transformers_` attribute for details.

        .. note::
            # 注意：
            If you do not access the list of columns for the remainder columns
            in the `transformers_` fitted attribute, you do not need to set
            this parameter.

        .. versionadded:: 1.5

        .. versionchanged:: 1.7
           # 版本更改说明：1.7 版本中 `force_int_remainder_cols` 的默认值将从 `True` 改为 `False`

    Attributes
    ----------
    transformers_ : list
        # 存储已拟合的转换器的集合，每个元素为元组 (name, fitted_transformer, column)。
        # fitted_transformer 可能是一个评估器，或者是 'drop'；'passthrough' 将被替换为等效的 FunctionTransformer。
        # 如果没有选择列，则这将是未拟合的转换器。
        # 如果有剩余的列，则最后一个元素是一个形如 ('remainder', transformer, remaining_columns) 的元组，
        # 对应于 remainder 参数。如果有剩余的列，则 len(transformers_)==len(transformers)+1，否则 len(transformers_)==len(transformers)。
        .. versionchanged:: 1.5
            # 如果有剩余的列并且 force_int_remainder_cols 为 True，则剩余的列总是以它们在输入 X 中的位置索引表示（与旧版本相同）。
            # 如果 force_int_remainder_cols 为 False，则格式尝试匹配其他转换器的格式：
            # 如果所有列都以列名（str）提供，则剩余的列将以列名存储；
            # 如果所有列都以掩码数组（bool）提供，则剩余的列也是掩码数组；
            # 在所有其他情况下，剩余的列将以索引（int）形式存储。

    named_transformers_ : :class:`~sklearn.utils.Bunch`
        # 只读属性，用于通过给定名称访问任何转换器。
        # 键是转换器名称，值是已拟合的转换器对象。

    sparse_output_ : bool
        # 布尔标志，指示 `transform` 的输出是稀疏矩阵还是密集的 numpy 数组，
        # 这取决于各个转换器的输出和 sparse_threshold 参数。

    output_indices_ : dict
        # 从每个转换器名称到一个切片的字典，切片对应于转换输出中的索引。
        # 这对于查看哪个转换器负责哪些转换特征非常有用。
        .. versionadded:: 1.0

    n_features_in_ : int
        # 在拟合过程中看到的特征数。仅在底层转换器在拟合时公开此属性时定义。
        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合过程中看到的特征的名称。仅当 `X` 的特征名称全为字符串时定义。

    See Also
    --------
    make_column_transformer : 用于将多个转换器对象应用于原始特征空间的列子集的便捷函数。
    make_column_selector : 基于数据类型或列名与正则表达式模式选择列的便捷函数。
    """
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the `transformers` list.
    Columns of the original feature matrix that are not specified are
    dropped from the resulting transformed feature matrix, unless specified
    in the `passthrough` keyword. Those columns specified with `passthrough`
    are added at the right to the output of the transformers.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.preprocessing import Normalizer
    >>> ct = ColumnTransformer(
    ...     [("norm1", Normalizer(norm='l1'), [0, 1]),
    ...      ("norm2", Normalizer(norm='l1'), slice(2, 4))])
    >>> X = np.array([[0., 1., 2., 2.],
    ...               [1., 1., 0., 1.]])
    >>> # Normalizer scales each row of X to unit norm. A separate scaling
    >>> # is applied for the two first and two last elements of each
    >>> # row independently.
    >>> ct.fit_transform(X)
    array([[0. , 1. , 0.5, 0.5],
           [0.5, 0.5, 0. , 1. ]])

    :class:`ColumnTransformer` can be configured with a transformer that requires
    a 1d array by setting the column to a string:

    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> import pandas as pd   # doctest: +SKIP
    >>> X = pd.DataFrame({
    ...     "documents": ["First item", "second one here", "Is this the last?"],
    ...     "width": [3, 4, 5],
    ... })  # doctest: +SKIP
    >>> # "documents" is a string which configures ColumnTransformer to
    >>> # pass the documents column as a 1d array to the CountVectorizer
    >>> ct = ColumnTransformer(
    ...     [("text_preprocess", CountVectorizer(), "documents"),
    ...      ("num_preprocess", MinMaxScaler(), ["width"])])
    >>> X_trans = ct.fit_transform(X)  # doctest: +SKIP

    For a more detailed example of usage, see
    :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py`.
    """

    # Required parameters for the ColumnTransformer class
    _required_parameters = ["transformers"]

    # Parameter constraints for validation
    _parameter_constraints: dict = {
        "transformers": [list, Hidden(tuple)],
        "remainder": [
            StrOptions({"drop", "passthrough"}),
            HasMethods(["fit", "transform"]),
            HasMethods(["fit_transform", "transform"]),
        ],
        "sparse_threshold": [Interval(Real, 0, 1, closed="both")],
        "n_jobs": [Integral, None],
        "transformer_weights": [dict, None],
        "verbose": ["verbose"],
        "verbose_feature_names_out": ["boolean"],
        "force_int_remainder_cols": ["boolean"],
    }

    # Constructor method for the ColumnTransformer class
    def __init__(
        self,
        transformers,
        *,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
        force_int_remainder_cols=True,
    ):
        """
        Initialize a ColumnTransformer instance.

        Parameters:
        -----------
        transformers : list
            List of tuples specifying the transformers to be applied.
        remainder : {'drop', 'passthrough'}, default='drop'
            Strategy for handling columns not specified in transformers.
        sparse_threshold : float, default=0.3
            Threshold for feature sparsity.
        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
        transformer_weights : dict or None, default=None
            Weights associated with transformers.
        verbose : bool, default=False
            Enable verbose output.
        verbose_feature_names_out : bool, default=True
            Control whether to include feature names in verbose output.
        force_int_remainder_cols : bool, default=True
            Force integer conversion for remainder columns.
        """
    ):
        self.transformers = transformers  # 设置对象的转换器列表，用于数据转换
        self.remainder = remainder  # 设置对象的剩余处理方式
        self.sparse_threshold = sparse_threshold  # 设置稀疏矩阵阈值
        self.n_jobs = n_jobs  # 设置并行处理任务数
        self.transformer_weights = transformer_weights  # 设置转换器权重
        self.verbose = verbose  # 设置详细程度
        self.verbose_feature_names_out = verbose_feature_names_out  # 设置输出特征名称的详细程度
        self.force_int_remainder_cols = force_int_remainder_cols  # 设置强制整数剩余列处理方式

    @property
    def _transformers(self):
        """
        Internal list of transformer only containing the name and
        transformers, dropping the columns.

        DO NOT USE: This is for the implementation of get_params via
        BaseComposition._get_params which expects lists of tuples of len 2.

        To iterate through the transformers, use ``self._iter`` instead.
        """
        try:
            return [(name, trans) for name, trans, _ in self.transformers]
        except (TypeError, ValueError):
            return self.transformers

    @_transformers.setter
    def _transformers(self, value):
        """DO NOT USE: This is for the implementation of set_params via
        BaseComposition._get_params which gives lists of tuples of len 2.
        """
        try:
            self.transformers = [
                (name, trans, col)
                for ((name, trans), (_, _, col)) in zip(value, self.transformers)
            ]
        except (TypeError, ValueError):
            self.transformers = value

    def set_output(self, *, transform=None):
        """Set the output container when `"transform"` and `"fit_transform"` are called.

        Calling `set_output` will set the output of all estimators in `transformers`
        and `transformers_`.

        Parameters
        ----------
        transform : {"default", "pandas", "polars"}, default=None
            Configure output of `transform` and `fit_transform`.

            - `"default"`: Default output format of a transformer
            - `"pandas"`: DataFrame output
            - `"polars"`: Polars output
            - `None`: Transform configuration is unchanged

            .. versionadded:: 1.4
                `"polars"` option was added.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        super().set_output(transform=transform)  # 调用父类方法设置输出

        transformers = (
            trans
            for _, trans, _ in chain(
                self.transformers, getattr(self, "transformers_", [])
            )
            if trans not in {"passthrough", "drop"}  # 过滤掉 "passthrough" 和 "drop" 的转换器
        )
        for trans in transformers:
            _safe_set_output(trans, transform=transform)  # 设置转换器的输出格式

        if self.remainder not in {"passthrough", "drop"}:
            _safe_set_output(self.remainder, transform=transform)  # 设置剩余处理的输出格式

        return self  # 返回当前对象实例
    # 获取该估计器的参数
    def get_params(self, deep=True):
        """Get parameters for this estimator.
    
        返回构造函数中给定的参数以及`ColumnTransformer`的`transformers`中包含的估计器。
    
        Parameters
        ----------
        deep : bool, default=True
            如果为True，则返回该估计器及其包含的子对象（估计器）的参数。
    
        Returns
        -------
        params : dict
            参数名称映射到其值。
        """
        return self._get_params("_transformers", deep=deep)
    
    # 设置该估计器的参数
    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
    
        可以使用`get_params()`列出有效的参数键。请注意，您可以直接设置`ColumnTransformer`的`transformers`中包含的估计器的参数。
    
        Parameters
        ----------
        **kwargs : dict
            估计器参数。
    
        Returns
        -------
        self : ColumnTransformer
            该估计器。
        """
        self._set_params("_transformers", **kwargs)
        return self
    def _iter(self, fitted, column_as_labels, skip_drop, skip_empty_columns):
        """
        Generate (name, trans, columns, weight) tuples.

        Parameters
        ----------
        fitted : bool
            If True, use the fitted transformers (``self.transformers_``) to
            iterate through transformers, else use the transformers passed by
            the user (``self.transformers``).

        column_as_labels : bool
            If True, columns are returned as string labels. If False, columns
            are returned as they were given by the user. This can only be True
            if the ``ColumnTransformer`` is already fitted.

        skip_drop : bool
            If True, 'drop' transformers are filtered out.

        skip_empty_columns : bool
            If True, transformers with empty selected columns are filtered out.

        Yields
        ------
        A generator of tuples containing:
            - name : the name of the transformer
            - transformer : the transformer object
            - columns : the columns for that transformer
            - weight : the weight of the transformer
        """

        if fitted:
            # 如果 fitted 为 True，则使用已经拟合的转换器 self.transformers_
            transformers = self.transformers_
        else:
            # 否则使用用户传入的未拟合转换器 self.transformers，并与列标识符交错
            transformers = [
                (name, trans, column)
                for (name, trans, _), column in zip(self.transformers, self._columns)
            ]
            # 如果存在剩余部分转换器，则添加剩余部分的转换器元组
            if self._remainder[2]:
                transformers = chain(transformers, [self._remainder])

        # 禁用关于剩余列数据类型将来更改的警告，仅在用户直接访问时显示
        # 该警告，而不是在 ColumnTransformer 自身使用时显示
        transformers = _with_dtype_warning_enabled_set_to(False, transformers)

        # 获取转换器权重的函数，如果未定义则返回 None
        get_weight = (self.transformer_weights or {}).get

        for name, trans, columns in transformers:
            if skip_drop and trans == "drop":
                # 如果 skip_drop 为 True 并且转换器为 "drop"，则跳过该转换器
                continue
            if skip_empty_columns and _is_empty_column_selection(columns):
                # 如果 skip_empty_columns 为 True 并且选择的列为空，则跳过该转换器
                continue

            if column_as_labels:
                # 如果 column_as_labels 为 True，则将所有列转换为它们的字符串标签
                columns_is_scalar = np.isscalar(columns)

                # 获取转换器对应的输入索引，从中获取特征名称
                indices = self._transformer_to_input_indices[name]
                columns = self.feature_names_in_[indices]

                if columns_is_scalar:
                    # 如果 columns 是标量，则选择的是一个维度
                    columns = columns[0]

            # 生成器返回每个转换器的名称、转换器对象、对应列、以及权重（如果定义）
            yield (name, trans, columns, get_weight(name))
    def _validate_transformers(self):
        """
        Validate names of transformers and the transformers themselves.

        This checks whether given transformers have the required methods, i.e.
        `fit` or `fit_transform` and `transform` implemented.
        """
        # 如果没有任何transformers，则直接返回
        if not self.transformers:
            return

        # 解压self.transformers元组中的names, transformers, _
        names, transformers, _ = zip(*self.transformers)

        # 验证transformers的名称
        self._validate_names(names)

        # 验证每个transformer
        for t in transformers:
            # 如果是"drop"或"passthrough"，则跳过验证
            if t in ("drop", "passthrough"):
                continue
            # 如果transformer没有fit或fit_transform方法，或者没有transform方法，则抛出TypeError
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All estimators should implement fit and "
                    "transform, or can be 'drop' or 'passthrough' "
                    "specifiers. '%s' (type %s) doesn't." % (t, type(t))
                )

    def _validate_column_callables(self, X):
        """
        Converts callable column specifications.

        This stores a dictionary of the form `{step_name: column_indices}` and
        calls the `columns` on `X` if `columns` is a callable for a given
        transformer.

        The results are then stored in `self._transformer_to_input_indices`.
        """
        # 初始化空列表用于存储所有的列
        all_columns = []
        # 初始化空字典，用于存储transformer到输入索引的映射关系
        transformer_to_input_indices = {}
        
        # 遍历self.transformers中的每个transformer
        for name, _, columns in self.transformers:
            # 如果columns是callable，调用它，并传入X
            if callable(columns):
                columns = columns(X)
            # 将处理后的columns加入到all_columns列表中
            all_columns.append(columns)
            # 根据X和columns获取列的索引，并存入transformer_to_input_indices字典中
            transformer_to_input_indices[name] = _get_column_indices(X, columns)

        # 将所有列存入self._columns
        self._columns = all_columns
        # 将transformer到输入索引的映射关系存入self._transformer_to_input_indices
        self._transformer_to_input_indices = transformer_to_input_indices

    def _validate_remainder(self, X):
        """
        Validates ``remainder`` and defines ``_remainder`` targeting
        the remaining columns.
        """
        # 获取所有已经使用的列的索引集合
        cols = set(chain(*self._transformer_to_input_indices.values()))
        # 计算剩余未使用的列的索引列表，并进行排序
        remaining = sorted(set(range(self.n_features_in_)) - cols)
        # 将剩余未使用的列的索引列表存入self._transformer_to_input_indices字典中的"remainder"键
        self._transformer_to_input_indices["remainder"] = remaining
        # 获取剩余列的数据，并存入self._remainder
        remainder_cols = self._get_remainder_cols(remaining)
        self._remainder = ("remainder", self.remainder, remainder_cols)

    def _get_remainder_cols_dtype(self):
        try:
            # 获取所有transformers的数据类型集合
            all_dtypes = {_determine_key_type(c) for (*_, c) in self.transformers}
            # 如果所有数据类型相同，则返回该数据类型
            if len(all_dtypes) == 1:
                return next(iter(all_dtypes))
        except ValueError:
            # 如果某些transformer的列是callable，则返回"int"
            return "int"
        # 默认返回"int"
        return "int"
    # 获取剩余列的方法，根据给定的索引列表返回适当的结果
    def _get_remainder_cols(self, indices):
        # 获取剩余列的数据类型
        dtype = self._get_remainder_cols_dtype()
        # 如果强制要求剩余列为整数类型且当前类型不是整数，则返回一个_RemainderColsList对象
        if self.force_int_remainder_cols and dtype != "int":
            return _RemainderColsList(indices, future_dtype=dtype)
        # 如果数据类型为字符串，则返回特征名称列表中索引对应的内容
        if dtype == "str":
            return list(self.feature_names_in_[indices])
        # 如果数据类型为布尔值，则返回一个布尔列表，表示每个索引是否在给定的索引列表中
        if dtype == "bool":
            return [i in indices for i in range(self.n_features_in_)]
        # 否则直接返回索引列表本身
        return indices

    @property
    def named_transformers_(self):
        """按名称访问已拟合的转换器。

        只读属性，通过给定名称访问任何转换器。
        键是转换器名称，值是已拟合的转换器对象。
        """
        # 使用Bunch对象来改进自动完成功能
        return Bunch(**{name: trans for name, trans, _ in self.transformers_})

    def _get_feature_name_out_for_transformer(self, name, trans, feature_names_in):
        """获取转换器的特征名称。

        与self._iter(fitted=True)一起在get_feature_names_out中使用。
        """
        # 获取转换器对应的输入索引
        column_indices = self._transformer_to_input_indices[name]
        # 根据输入特征名称获取转换后的特征名称
        names = feature_names_in[column_indices]
        # 如果转换器没有提供get_feature_names_out方法，则引发错误
        if not hasattr(trans, "get_feature_names_out"):
            raise AttributeError(
                f"Transformer {name} (type {type(trans).__name__}) does "
                "not provide get_feature_names_out."
            )
        # 返回转换器的输出特征名称
        return trans.get_feature_names_out(names)
    def get_feature_names_out(self, input_features=None):
        """获取转换后的特征名称列表。

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            输入特征。

            - 如果 `input_features` 是 `None`，则使用 `feature_names_in_` 作为输入特征名称。
              如果 `feature_names_in_` 未定义，则生成如下输入特征名称：
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`。
            - 如果 `input_features` 是一个数组类对象，则 `input_features` 必须与
              `feature_names_in_` 匹配（如果 `feature_names_in_` 已定义）。

        Returns
        -------
        feature_names_out : ndarray of str objects
            转换后的特征名称数组。
        """
        # 确保模型已拟合
        check_is_fitted(self)
        # 检查并规范化输入特征名称
        input_features = _check_feature_names_in(self, input_features)

        # 用于存储转换器名称及其输出特征名称的列表
        transformer_with_feature_names_out = []
        # 遍历模型中已拟合的转换器
        for name, trans, *_ in self._iter(
            fitted=True,
            column_as_labels=False,
            skip_empty_columns=True,
            skip_drop=True,
        ):
            # 获取当前转换器的输出特征名称
            feature_names_out = self._get_feature_name_out_for_transformer(
                name, trans, input_features
            )
            # 如果特征名称为空，跳过当前转换器
            if feature_names_out is None:
                continue
            # 将转换器名称及其输出特征名称添加到列表中
            transformer_with_feature_names_out.append((name, feature_names_out))

        # 如果没有转换器输出任何特征名称，则返回空的 ndarray
        if not transformer_with_feature_names_out:
            return np.array([], dtype=object)

        # 对转换器输出的特征名称添加前缀处理，并返回结果
        return self._add_prefix_for_feature_names_out(
            transformer_with_feature_names_out
        )
    # 添加前缀到包含转换器名称的特征名输出中

    # 如果设置了详细特征名输出
    if self.verbose_feature_names_out:
        # 使用链式迭代器，为每个特征名添加转换器名称前缀
        names = list(
            chain.from_iterable(
                (f"{name}__{i}" for i in feature_names_out)
                for name, feature_names_out in transformer_with_feature_names_out
            )
        )
        # 将结果转换为 ndarray 类型并返回
        return np.asarray(names, dtype=object)

    # 如果详细特征名输出为假
    # 检查特征名是否全部不带前缀且唯一
    feature_names_count = Counter(
        chain.from_iterable(s for _, s in transformer_with_feature_names_out)
    )
    # 找出重叠次数最多的前 6 个特征名
    top_6_overlap = [
        name for name, count in feature_names_count.most_common(6) if count > 1
    ]
    # 对重叠的特征名进行排序
    top_6_overlap.sort()
    # 如果存在重叠的特征名
    if top_6_overlap:
        if len(top_6_overlap) == 6:
            # 如果超过 5 个重叠特征名，只显示前 5 个和省略符号
            names_repr = str(top_6_overlap[:5])[:-1] + ", ...]"
        else:
            names_repr = str(top_6_overlap)
        # 抛出值错误，说明输出特征名不唯一，建议设置 verbose_feature_names_out=True 添加前缀
        raise ValueError(
            f"Output feature names: {names_repr} are not unique. Please set "
            "verbose_feature_names_out=True to add prefixes to feature names"
        )

    # 返回转换器与特征名的组合中的特征名组成的 ndarray
    return np.concatenate(
        [name for _, name in transformer_with_feature_names_out],
    )
    def _update_fitted_transformers(self, transformers):
        """Set self.transformers_ from given transformers.

        Parameters
        ----------
        transformers : list of estimators
            The fitted estimators as the output of
            `self._call_func_on_transformers(func=_fit_transform_one, ...)`.
            That function doesn't include 'drop' or transformers for which no
            column is selected. 'drop' is kept as is, and for the no-column
            transformers the unfitted transformer is put in
            `self.transformers_`.
        """
        # 从给定的 transformers 设置 self.transformers_
        
        # transformers 是已经拟合的变换器；不包括 'drop' 的情况
        fitted_transformers = iter(transformers)
        transformers_ = []

        # 遍历所有的变换器
        for name, old, column, _ in self._iter(
            fitted=False,
            column_as_labels=False,
            skip_drop=False,
            skip_empty_columns=False,
        ):
            # 如果变换器是 'drop'，则将其标记为 "drop"
            if old == "drop":
                trans = "drop"
            # 如果列选择为空，则保留原始变换器
            elif _is_empty_column_selection(column):
                trans = old
            # 否则，使用下一个已拟合的变换器
            else:
                trans = next(fitted_transformers)
            # 将处理后的变换器及其相关信息添加到 transformers_ 列表中
            transformers_.append((name, trans, column))

        # 检查确保 fitted_transformers 已耗尽
        assert not list(fitted_transformers)
        
        # 将生成的 transformers_ 列表赋值给 self.transformers_
        self.transformers_ = _with_dtype_warning_enabled_set_to(True, transformers_)
    def _validate_output(self, result):
        """
        Ensure that the output of each transformer is 2D. Otherwise
        hstack can raise an error or produce incorrect results.
        """
        # 获取所有转换器的名称列表
        names = [
            name
            for name, _, _, _ in self._iter(
                fitted=True,
                column_as_labels=False,
                skip_drop=True,
                skip_empty_columns=True,
            )
        ]
        # 遍历每个转换器的输出结果和对应的名称
        for Xs, name in zip(result, names):
            # 检查输出结果是否为二维数组或者是 Pandas DataFrame
            if not getattr(Xs, "ndim", 0) == 2 and not hasattr(Xs, "__dataframe__"):
                # 如果不是，抛出数值错误
                raise ValueError(
                    "The output of the '{0}' transformer should be 2D (numpy array, "
                    "scipy sparse array, dataframe).".format(name)
                )
        # 如果输出配置指定为 Pandas，直接返回
        if _get_output_config("transform", self)["dense"] == "pandas":
            return
        try:
            # 尝试导入 pandas 库
            import pandas as pd
        except ImportError:
            # 如果导入失败，直接返回
            return
        # 再次遍历每个转换器的输出结果和名称
        for Xs, name in zip(result, names):
            # 如果输出结果不是 Pandas DataFrame，继续下一个循环
            if not _is_pandas_df(Xs):
                continue
            # 遍历 DataFrame 的列名和对应的数据类型
            for col_name, dtype in Xs.dtypes.to_dict().items():
                # 如果数据类型具有 na_value 属性且不是 pd.NA，则继续下一个循环
                if getattr(dtype, "na_value", None) is not pd.NA:
                    continue
                # 如果 pd.NA 不在列的值中，则继续下一个循环
                if pd.NA not in Xs[col_name].values:
                    continue
                # 获取当前类名
                class_name = self.__class__.__name__
                # 发出警告，指出当前输出结果中存在使用 pandas.NA 表示空值的情况
                warnings.warn(
                    (
                        f"The output of the '{name}' transformer for column"
                        f" '{col_name}' has dtype {dtype} and uses pandas.NA to"
                        " represent null values. Storing this output in a numpy array"
                        " can cause errors in downstream scikit-learn estimators, and"
                        " inefficiencies. Starting with scikit-learn version 1.6, this"
                        " will raise a ValueError. To avoid this problem you can (i)"
                        " store the output in a pandas DataFrame by using"
                        f" {class_name}.set_output(transform='pandas') or (ii) modify"
                        f" the input data or the '{name}' transformer to avoid the"
                        " presence of pandas.NA (for example by using"
                        " pandas.DataFrame.astype)."
                    ),
                    FutureWarning,
                )
    # 记录每个转换器生成的列的索引范围
    def _record_output_indices(self, Xs):
        """
        Record which transformer produced which column.
        记录每个转换器生成了哪些列。
        """
        # 初始化索引起始位置
        idx = 0
        # 初始化存储输出索引的字典
        self.output_indices_ = {}

        # 遍历已经安装的转换器，枚举得到每个转换器的信息
        for transformer_idx, (name, _, _, _) in enumerate(
            self._iter(
                fitted=True,
                column_as_labels=False,
                skip_drop=True,
                skip_empty_columns=True,
            )
        ):
            # 获取当前转换器生成的列数
            n_columns = Xs[transformer_idx].shape[1]
            # 记录转换器名和其生成列的索引范围
            self.output_indices_[name] = slice(idx, idx + n_columns)
            # 更新下一个转换器的起始索引位置
            idx += n_columns

        # `_iter` 只生成有非空选择的转换器。这里为那些没有输出的转换器设置空的切片，
        # 这些切片对索引是安全的。
        all_names = [t[0] for t in self.transformers] + ["remainder"]
        for name in all_names:
            # 对于没有在输出索引中的转换器，设置一个空的切片
            if name not in self.output_indices_:
                self.output_indices_[name] = slice(0, 0)

    # 打印日志信息，显示当前处理的转换器
    def _log_message(self, name, idx, total):
        """
        Generate a log message indicating processing of a transformer.
        生成一个日志信息，指示转换器的处理过程。
        """
        # 如果非详细模式，直接返回 None
        if not self.verbose:
            return None
        # 返回格式化的处理信息字符串
        return "(%d of %d) Processing %s" % (idx, total, name)
    def _call_func_on_transformers(self, X, y, func, column_as_labels, routed_params):
        """
        Private function to fit and/or transform on demand.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be used in fit and/or transform.

        y : array-like of shape (n_samples,)
            Targets.

        func : callable
            Function to call, which can be _fit_transform_one or
            _transform_one.

        column_as_labels : bool
            Used to iterate through transformers. If True, columns are returned
            as strings. If False, columns are returned as they were given by
            the user. Can be True only if the ``ColumnTransformer`` is already
            fitted.

        routed_params : dict
            The routed parameters as the output from ``process_routing``.

        Returns
        -------
        Return value (transformers and/or transformed X data) depends
        on the passed function.
        """
        # 根据传入的函数确定是否已经拟合
        if func is _fit_transform_one:
            fitted = False
        else:  # func is _transform_one
            fitted = True

        # 获取所有的转换器
        transformers = list(
            self._iter(
                fitted=fitted,
                column_as_labels=column_as_labels,
                skip_drop=True,
                skip_empty_columns=True,
            )
        )
        try:
            jobs = []
            # 遍历所有转换器
            for idx, (name, trans, columns, weight) in enumerate(transformers, start=1):
                if func is _fit_transform_one:
                    # 如果转换器是"passthrough"，则创建一个FunctionTransformer
                    if trans == "passthrough":
                        output_config = _get_output_config("transform", self)
                        trans = FunctionTransformer(
                            accept_sparse=True,
                            check_inverse=False,
                            feature_names_out="one-to-one",
                        ).set_output(transform=output_config["dense"])

                    extra_args = dict(
                        message_clsname="ColumnTransformer",
                        message=self._log_message(name, idx, len(transformers)),
                    )
                else:  # func is _transform_one
                    extra_args = {}
                # 添加任务到jobs列表中
                jobs.append(
                    delayed(func)(
                        transformer=clone(trans) if not fitted else trans,
                        X=_safe_indexing(X, columns, axis=1),
                        y=y,
                        weight=weight,
                        **extra_args,
                        params=routed_params[name],
                    )
                )

            # 并行执行jobs中的任务
            return Parallel(n_jobs=self.n_jobs)(jobs)

        except ValueError as e:
            # 捕获特定异常并重新抛出
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError(_ERR_MSG_1DCOLUMN) from e
            else:
                raise
    # 定义一个方法 `fit`，用于拟合所有的转换器，根据输入数据 X 进行拟合

    def fit(self, X, y=None, **params):
        """Fit all transformers using X.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.

        **params : dict, default=None
            Parameters to be passed to the underlying transformers' ``fit`` and
            ``transform`` methods.

            You can only pass this if metadata routing is enabled, which you
            can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

            .. versionadded:: 1.4

        Returns
        -------
        self : ColumnTransformer
            This estimator.
        """

        # 检查参数并引发异常，确保参数在估计器中是合法的
        _raise_for_params(params, self, "fit")

        # 使用 fit_transform 方法来确保设置 sparse_output_（需要转换后的数据）以在预测中保持一致的输出类型
        self.fit_transform(X, y=y, **params)

        # 返回估计器自身
        return self

    @_fit_context(
        # estimators in ColumnTransformer.transformers are not validated yet
        prefer_skip_nested_validation=False
    )
    # 拟合所有的转换器，并对数据进行转换，然后将结果进行连接。

    # 参数说明：
    # X : 形状为 (n_samples, n_features) 的数组或数据框架
    #     输入数据，其中使用指定的子集来拟合转换器。
    # y : 形状为 (n_samples,) 的数组，默认为 None
    #     监督学习的目标。
    # **params : 字典，默认为 None
    #     要传递给底层转换器的 ``fit`` 和 ``transform`` 方法的参数。
    #     只有在启用元数据路由时才能传递这些参数，可以通过 ``sklearn.set_config(enable_metadata_routing=True)`` 启用。

    # 返回结果：
    # X_t : 形状为 (n_samples, sum_n_components) 的数组或稀疏矩阵
    #     转换器的水平堆叠结果。sum_n_components 是转换器上的输出维度总和。
    #     如果任何结果是稀疏矩阵，则所有内容都将转换为稀疏矩阵。

    _raise_for_params(params, self, "fit_transform")
    # 检查参数，如果有问题会引发异常

    self._check_feature_names(X, reset=True)
    # 检查特征名称，根据需要重置

    X = _check_X(X)
    # 检查输入数据 X 的格式

    # 设置 n_features_in_ 属性
    self._check_n_features(X, reset=True)

    self._validate_transformers()
    # 验证所有转换器

    n_samples = _num_samples(X)
    # 获取样本数量

    self._validate_column_callables(X)
    # 验证列可调用对象

    self._validate_remainder(X)
    # 验证剩余的部分

    if _routing_enabled():
        routed_params = process_routing(self, "fit_transform", **params)
    else:
        routed_params = self._get_empty_routing()
    # 如果启用了路由，处理路由参数；否则，获取空的路由参数

    result = self._call_func_on_transformers(
        X,
        y,
        _fit_transform_one,
        column_as_labels=False,
        routed_params=routed_params,
    )
    # 对转换器执行 _fit_transform_one 方法，并返回结果

    if not result:
        self._update_fitted_transformers([])
        # 如果所有转换器都为 None，则更新已拟合转换器为空列表
        return np.zeros((n_samples, 0))
        # 返回一个零数组，形状为 (n_samples, 0)

    Xs, transformers = zip(*result)
    # 解压缩结果元组为 Xs 和 transformers

    # 确定连接后的输出是否为稀疏矩阵
    if any(sparse.issparse(X) for X in Xs):
        nnz = sum(X.nnz if sparse.issparse(X) else X.size for X in Xs)
        total = sum(
            X.shape[0] * X.shape[1] if sparse.issparse(X) else X.size for X in Xs
        )
        density = nnz / total
        self.sparse_output_ = density < self.sparse_threshold
    else:
        self.sparse_output_ = False
    # 计算稀疏性并更新 sparse_output_

    self._update_fitted_transformers(transformers)
    # 更新已拟合转换器

    self._validate_output(Xs)
    # 验证输出格式

    self._record_output_indices(Xs)
    # 记录输出索引

    return self._hstack(list(Xs), n_samples=n_samples)
    # 对 Xs 列表进行水平堆叠，返回结果
    def _sk_visual_block_(self):
        # 检查 self.remainder 是否为字符串且其值为 "drop"
        if isinstance(self.remainder, str) and self.remainder == "drop":
            # 如果是，则使用已有的 transformers
            transformers = self.transformers
        # 如果 self._remainder 存在
        elif hasattr(self, "_remainder"):
            # 获取 self._remainder 的第三个元素作为 remainder_columns
            remainder_columns = self._remainder[2]
            # 如果存在 feature_names_in_ 属性，并且 remainder_columns 不为空且不是全部都是字符串
            if (
                hasattr(self, "feature_names_in_")
                and remainder_columns
                and not all(isinstance(col, str) for col in remainder_columns)
            ):
                # 将 remainder_columns 转换为列表形式
                remainder_columns = self.feature_names_in_[remainder_columns].tolist()
            # 使用 chain 函数将 self.transformers 和 [("remainder", self.remainder, remainder_columns)] 合并
            transformers = chain(
                self.transformers, [("remainder", self.remainder, remainder_columns)]
            )
        else:
            # 否则使用 chain 函数将 self.transformers 和 [("remainder", self.remainder, "")] 合并
            transformers = chain(self.transformers, [("remainder", self.remainder, "")])

        # 解压 transformers 变量的元素并分配给 names, transformers, name_details
        names, transformers, name_details = zip(*transformers)
        # 返回一个 _VisualBlock 对象，类型为 "parallel"
        return _VisualBlock(
            "parallel", transformers, names=names, name_details=name_details
        )

    def __getitem__(self, key):
        try:
            # 尝试返回指定键名的 named_transformers_ 中的值
            return self.named_transformers_[key]
        except AttributeError as e:
            # 如果发生 AttributeError，则抛出 TypeError，提示 ColumnTransformer 在拟合后可以进行下标访问
            raise TypeError(
                "ColumnTransformer is subscriptable after it is fitted"
            ) from e
        except KeyError as e:
            # 如果发生 KeyError，则抛出 KeyError，提示指定的键名不是有效的转换器名称
            raise KeyError(f"'{key}' is not a valid transformer name") from e

    def _get_empty_routing(self):
        """Return empty routing.

        Used while routing can be disabled.

        TODO: Remove when ``set_config(enable_metadata_routing=False)`` is no
        more an option.
        """
        # 返回一个 Bunch 对象，包含空的路由信息，具体内容为生成的字典推导式
        return Bunch(
            **{
                name: Bunch(**{method: {} for method in METHODS})
                for name, step, _, _ in self._iter(
                    fitted=False,
                    column_as_labels=False,
                    skip_drop=True,
                    skip_empty_columns=True,
                )
            }
        )
    def get_metadata_routing(self):
        """获取此对象的元数据路由。

        请查看 :ref:`User Guide <metadata_routing>` 了解路由机制的工作原理。

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            封装了路由信息的 :class:`~sklearn.utils.metadata_routing.MetadataRouter` 对象。
        """
        # 创建一个 MetadataRouter 对象，将所有步骤的元数据路由到这个对象中
        router = MetadataRouter(owner=self.__class__.__name__)
        
        # 获取所有转换器链表，包括可能的“remainder”转换器
        transformers = chain(self.transformers, [("remainder", self.remainder, None)])
        
        # 遍历每个转换器及其对应的名称和步骤
        for name, step, _ in transformers:
            # 创建一个 MethodMapping 对象，用于映射各种方法调用
            method_mapping = MethodMapping()
            
            # 如果步骤具有 "fit_transform" 方法
            if hasattr(step, "fit_transform"):
                # 添加 fit 和 fit_transform 方法的映射关系
                method_mapping.add(caller="fit", callee="fit_transform").add(
                    caller="fit_transform", callee="fit_transform"
                )
            else:
                # 否则，添加 fit、transform 方法的映射关系
                method_mapping.add(caller="fit", callee="fit").add(
                    caller="fit", callee="transform"
                ).add(caller="fit_transform", callee="fit").add(
                    caller="fit_transform", callee="transform"
                )
            
            # 添加 transform 方法的映射关系
            method_mapping.add(caller="transform", callee="transform")
            
            # 将 method_mapping 和对应的步骤名称添加到路由器中
            router.add(method_mapping=method_mapping, **{name: step})

        # 返回元数据路由器对象
        return router
# 当需要时才使用 check_array，例如在列表和其他非数组类对象上
def _check_X(X):
    # 检查 X 是否具有 "__array__" 属性，或者 "__dataframe__" 属性，或者是稀疏矩阵
    if hasattr(X, "__array__") or hasattr(X, "__dataframe__") or sparse.issparse(X):
        return X
    # 否则，调用 check_array 函数，允许 NaN 存在，数据类型为 object
    return check_array(X, force_all_finite="allow-nan", dtype=object)


# 检查列选择是否为空（空列表或全为 False 的布尔数组），返回 True 或 False
def _is_empty_column_selection(column):
    # 如果 column 具有 "dtype" 属性，并且是布尔类型
    if hasattr(column, "dtype") and np.issubdtype(column.dtype, np.bool_):
        # 返回是否所有值都为 False
        return not column.any()
    # 如果 column 是可迭代对象
    elif hasattr(column, "__len__"):
        # 返回是否列表长度为 0，或者所有元素为布尔类型且全为 False
        return (
            len(column) == 0
            or all(isinstance(col, bool) for col in column)
            and not any(column)
        )
    else:
        # 其他情况返回 False
        return False


# 从给定的转换器列表构造 (name, trans, column) 元组列表
def _get_transformer_list(estimators):
    # 解压缩 estimators 中的转换器和列信息
    transformers, columns = zip(*estimators)
    # 获取转换器的名称列表和空列表
    names, _ = zip(*_name_estimators(transformers))
    # 将名称、转换器和列信息组成三元组列表
    transformer_list = list(zip(names, transformers, columns))
    return transformer_list


# 该函数没有使用 validate_params 进行验证，因为它仅用作 ColumnTransformer 的工厂函数
def make_column_transformer(
    *transformers,
    remainder="drop",
    sparse_threshold=0.3,
    n_jobs=None,
    verbose=False,
    verbose_feature_names_out=True,
    force_int_remainder_cols=True,
):
    """从给定的转换器构造一个 ColumnTransformer 对象。

    这是 ColumnTransformer 构造函数的一种简写形式；不需要也不能使用名称来指定转换器。
    转换器的名称将根据其类型自动分配。它也不允许使用 `transformer_weights` 进行加权。

    详细信息请参阅 :ref:`User Guide <make_column_transformer>`。

    Parameters
    ----------
    *transformers : tuples
        形如 (transformer, columns) 的元组，指定要应用于数据子集的转换器对象。

        transformer : {'drop', 'passthrough'} 或者 estimator
            转换器必须支持 `fit` 和 `transform` 方法。特殊字符串 'drop' 和 'passthrough'
            也被接受，分别表示丢弃列或者不进行任何转换直接传递。
        columns : str, array-like of str, int, array-like of int, slice,
                array-like of bool 或者 callable
            索引数据的第二个轴。整数被解释为位置列，而字符串可以按名称引用 DataFrame 列。
            在 `transformer` 需要 X 为 1 维数组时应使用标量字符串或整数，否则将传递 2 维数组给转换器。
            可以使用 `make_column_selector` 来通过名称或 dtype 选择多列。

    """
    # 函数内部没有具体代码，因此这里不需要进一步的注释
    pass
    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        # `remainder`参数指定了处理未指定的列的方式，可以是字符串{'drop', 'passthrough'}或者一个estimator，默认为'drop'。
        # 默认情况下，只有在`transformers`中指定的列会被转换并组合到输出中，未指定的列会被丢弃（默认为'drop'）。
        # 通过指定`remainder='passthrough'`，所有未在`transformers`中指定的剩余列将自动通过。
        # 如果将`remainder`设置为一个estimator，未指定的剩余列将使用该estimator进行处理。
        # Estimator必须支持`fit`和`transform`方法。

    sparse_threshold : float, default=0.3
        # 如果转换后的输出同时包含稀疏和密集数据，则当密度低于此值时，将以稀疏矩阵形式堆叠。
        # 使用`sparse_threshold=0`可始终返回密集矩阵。
        # 当转换后的输出完全由稀疏数据或完全由密集数据组成时，堆叠结果将分别为稀疏或密集，并且此关键字将被忽略。

    n_jobs : int, default=None
        # 并行运行的作业数。
        # `None`表示除非在`joblib.parallel_backend`上下文中，否则为1。
        # `-1`表示使用所有处理器。详细信息请参见术语表中的“n_jobs”。

    verbose : bool, default=False
        # 如果为True，则在完成每个转换器的拟合时打印经过的时间。

    verbose_feature_names_out : bool, default=True
        # 如果为True，则:meth:`ColumnTransformer.get_feature_names_out`将所有特征名称前缀为生成该特征的转换器的名称。
        # 如果为False，则:meth:`ColumnTransformer.get_feature_names_out`将不会前缀任何特征名称，并且如果特征名称不唯一，则会报错。

        # .. versionadded:: 1.0

    force_int_remainder_cols : bool, default=True
        # 强制将`transformers_`的最后一个条目对应的“remainder”转换器的列始终存储为索引（int）而不是列名（str）。
        # 有关:attr:`ColumnTransformer.transformers_`属性的详细描述，请参阅该属性的描述。

        # .. note::
        #     如果您不访问:attr:`ColumnTransformer.transformers_`中剩余列的列列表，则不需要设置此参数。

        # .. versionadded:: 1.5

        # .. versionchanged:: 1.7
        #    在版本1.7中，默认值`force_int_remainder_cols`将从`True`更改为`False`。

    Returns
    -------
    ct : ColumnTransformer
        # 返回一个:class:`ColumnTransformer`对象。

    See Also
    --------
    """
        # ColumnTransformer：允许将多个变换器对象的输出应用于数据的列子集，将它们合并成单一的特征空间。
    
        Examples
        --------
        >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
        >>> from sklearn.compose import make_column_transformer
        >>> make_column_transformer(
        ...     (StandardScaler(), ['numerical_column']),
        ...     (OneHotEncoder(), ['categorical_column']))
        ColumnTransformer(transformers=[('standardscaler', StandardScaler(...),
                                         ['numerical_column']),
                                        ('onehotencoder', OneHotEncoder(...),
                                         ['categorical_column'])])
        """
        # transformer_weights 关键字没有传递，因为用户需要知道变换器的自动生成名称
        transformer_list = _get_transformer_list(transformers)
        # 创建并返回 ColumnTransformer 对象，使用给定的参数进行配置
        return ColumnTransformer(
            transformer_list,
            n_jobs=n_jobs,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
            force_int_remainder_cols=force_int_remainder_cols,
        )
class make_column_selector:
    """Create a callable to select columns to be used with
    :class:`ColumnTransformer`.

    :func:`make_column_selector` can select columns based on datatype or the
    columns name with a regex. When using multiple selection criteria, **all**
    criteria must match for a column to be selected.

    For an example of how to use :func:`make_column_selector` within a
    :class:`ColumnTransformer` to select columns based on data type (i.e.
    `dtype`), refer to
    :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py`.

    Parameters
    ----------
    pattern : str, default=None
        Name of columns containing this regex pattern will be included. If
        None, column selection will not be selected based on pattern.

    dtype_include : column dtype or list of column dtypes, default=None
        A selection of dtypes to include. For more details, see
        :meth:`pandas.DataFrame.select_dtypes`.

    dtype_exclude : column dtype or list of column dtypes, default=None
        A selection of dtypes to exclude. For more details, see
        :meth:`pandas.DataFrame.select_dtypes`.

    Returns
    -------
    selector : callable
        Callable for column selection to be used by a
        :class:`ColumnTransformer`.

    See Also
    --------
    ColumnTransformer : Class that allows combining the
        outputs of multiple transformer objects used on column subsets
        of the data into a single feature space.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
    >>> from sklearn.compose import make_column_transformer
    >>> from sklearn.compose import make_column_selector
    >>> import numpy as np
    >>> import pandas as pd  # doctest: +SKIP
    >>> X = pd.DataFrame({'city': ['London', 'London', 'Paris', 'Sallisaw'],
    ...                   'rating': [5, 3, 4, 5]})  # doctest: +SKIP
    >>> ct = make_column_transformer(
    ...       (StandardScaler(),
    ...        make_column_selector(dtype_include=np.number)),  # rating
    ...       (OneHotEncoder(),
    ...        make_column_selector(dtype_include=object)))  # city
    >>> ct.fit_transform(X)  # doctest: +SKIP
    array([[ 0.90453403,  1.        ,  0.        ,  0.        ],
           [-1.50755672,  1.        ,  0.        ,  0.        ],
           [-0.30151134,  0.        ,  1.        ,  0.        ],
           [ 0.90453403,  0.        ,  0.        ,  1.        ]])

    """

    def __init__(self, pattern=None, *, dtype_include=None, dtype_exclude=None):
        # 初始化函数，用于创建一个列选择器对象
        self.pattern = pattern
        # 设置正则表达式模式，用于列名的选择
        self.dtype_include = dtype_include
        # 设置包含的数据类型，用于选择列
        self.dtype_exclude = dtype_exclude
        # 设置排除的数据类型，用于选择列
    def __call__(self, df):
        """
        在:class:`ColumnTransformer`中使用的列选择器的可调用方法。

        Parameters
        ----------
        df : dataframe of shape (n_features, n_samples)
            要从中选择列的DataFrame。
        """
        # 检查df是否具有'iloc'属性，否则引发值错误
        if not hasattr(df, "iloc"):
            raise ValueError(
                "make_column_selector can only be applied to pandas dataframes"
            )
        # 从df中提取第一行数据
        df_row = df.iloc[:1]
        # 如果指定了dtype_include或dtype_exclude，则根据这些条件选择列
        if self.dtype_include is not None or self.dtype_exclude is not None:
            df_row = df_row.select_dtypes(
                include=self.dtype_include, exclude=self.dtype_exclude
            )
        # 获取列名
        cols = df_row.columns
        # 如果指定了pattern，则根据正则表达式筛选列名
        if self.pattern is not None:
            cols = cols[cols.str.contains(self.pattern, regex=True)]
        # 将列名转换为列表并返回
        return cols.tolist()
class _RemainderColsList(UserList):
    """
    A list that raises a warning whenever items are accessed.

    It is used to store the columns handled by the "remainder" entry of
    ``ColumnTransformer.transformers_``, ie ``transformers_[-1][-1]``.

    For some values of the ``ColumnTransformer`` ``transformers`` parameter,
    this list of indices will be replaced by either a list of column names or a
    boolean mask; in those cases we emit a ``FutureWarning`` the first time an
    element is accessed.

    Parameters
    ----------
    columns : list of int
        The remainder columns.

    future_dtype : {'str', 'bool'}, default=None
        The dtype that will be used by a ColumnTransformer with the same inputs
        in a future release. There is a default value because providing a
        constructor that takes a single argument is a requirement for
        subclasses of UserList, but we do not use it in practice. It would only
        be used if a user called methods that return a new list such are
        copying or concatenating `_RemainderColsList`.

    warning_was_emitted : bool, default=False
        Whether the warning for that particular list was already shown, so we
        only emit it once.

    warning_enabled : bool, default=True
        When False, the list never emits the warning nor updates
        `warning_was_emitted``. This is used to obtain a quiet copy of the list
        for use by the `ColumnTransformer` itself, so that the warning is only
        shown when a user accesses it directly.
    """

    def __init__(
        self,
        columns,
        *,
        future_dtype=None,
        warning_was_emitted=False,
        warning_enabled=True,
    ):
        # 调用父类 UserList 的构造方法初始化列表
        super().__init__(columns)
        # 设置未来数据类型的属性
        self.future_dtype = future_dtype
        # 设置是否已经发出警告的属性，默认为 False
        self.warning_was_emitted = warning_was_emitted
        # 设置警告是否启用的属性，默认为 True
        self.warning_enabled = warning_enabled

    def __getitem__(self, index):
        # 调用私有方法 _show_remainder_cols_warning() 发出警告
        self._show_remainder_cols_warning()
        # 调用父类 UserList 的 __getitem__() 方法获取索引处的元素
        return super().__getitem__(index)
    # 如果警告已经被发出或者警告功能未启用，则直接返回，不进行警告
    if self.warning_was_emitted or not self.warning_enabled:
        return
    
    # 标记警告已被发出，避免重复发出警告
    self.warning_was_emitted = True
    
    # 根据 self.future_dtype 获取对应的未来数据类型描述，用于警告信息中
    future_dtype_description = {
        "str": "column names (of type str)",
        "bool": "a mask array (of type bool)",
        # 因为我们总是使用非默认的 future_dtype 进行初始化，所以不应该发生为 None 的情况
        None: "a different type depending on the ColumnTransformer inputs",
    }.get(self.future_dtype, self.future_dtype)

    # 发出警告，说明 'remainder' transformer 在未来版本中的格式将发生变化
    # 版本 1.7 中更新警告内容，说明旧行为将在 1.9 中移除
    warnings.warn(
        (
            "\nThe format of the columns of the 'remainder' transformer in"
            " ColumnTransformer.transformers_ will change in version 1.7 to"
            " match the format of the other transformers.\nAt the moment the"
            " remainder columns are stored as indices (of type int). With the same"
            " ColumnTransformer configuration, in the future they will be stored"
            f" as {future_dtype_description}.\nTo use the new behavior now and"
            " suppress this warning, use"
            " ColumnTransformer(force_int_remainder_cols=False).\n"
        ),
        category=FutureWarning,
    )

def _repr_pretty_(self, printer, *_):
    """Override display in ipython console, otherwise the class name is shown."""
    # 在 ipython 控制台中，重写显示方式，以便展示对象的 repr() 结果而非类名
    printer.text(repr(self.data))
# 定义函数，设置数据类型警告使能状态并返回转换器列表
def _with_dtype_warning_enabled_set_to(warning_enabled, transformers):
    # 初始化结果列表
    result = []
    # 遍历 transformers 列表中的每个元素，每个元素包含 name、trans、columns 三个值
    for name, trans, columns in transformers:
        # 检查 columns 是否为 _RemainderColsList 类型的实例
        if isinstance(columns, _RemainderColsList):
            # 如果是 _RemainderColsList 类型，则创建一个新的 _RemainderColsList 对象
            # 在创建时更新 warning_enabled 参数的值
            columns = _RemainderColsList(
                columns.data,
                future_dtype=columns.future_dtype,
                warning_was_emitted=columns.warning_was_emitted,
                warning_enabled=warning_enabled,
            )
        # 将处理后的 name、trans、columns 组成的元组加入结果列表中
        result.append((name, trans, columns))
    # 返回处理后的结果列表
    return result
```