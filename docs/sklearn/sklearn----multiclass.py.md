# `D:\src\scipysrc\scikit-learn\sklearn\multiclass.py`

```
# 定义一个私有函数，用于部分拟合（适用于增量学习）一个二元分类器
def _partial_fit_binary(estimator, X, y, partial_fit_params):
    # 部分拟合单个二元估算器（estimator）。
    # 调用 estimator 对象的 partial_fit 方法，用于部分拟合模型。
    # 参数 X 是训练数据集，y 是目标变量，classes 是类别数组，partial_fit_params 是额外的部分拟合参数。
    estimator.partial_fit(X, y, classes=np.array((0, 1)), **partial_fit_params)
    # 返回部分拟合后的 estimator 对象。
    return estimator
def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    # 如果估计器是回归器，则直接返回其预测结果
    if is_regressor(estimator):
        return estimator.predict(X)
    
    try:
        # 尝试获取决策函数的分数（decision_function），并将其展平
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # 如果无法获取决策函数，则假定为分类器，返回正类的概率
        score = estimator.predict_proba(X)[:, 1]
    
    return score


def _threshold_for_binary_predict(estimator):
    """Threshold for predictions from binary estimator."""
    # 如果估计器有决策函数并且是分类器，则返回阈值0.0
    if hasattr(estimator, "decision_function") and is_classifier(estimator):
        return 0.0
    else:
        # 否则返回预测概率的默认阈值0.5
        return 0.5


class _ConstantPredictor(BaseEstimator):
    """Helper predictor to be used when only one class is present."""

    def fit(self, X, y):
        # 检查参数的合法性
        check_params = dict(
            force_all_finite=False, dtype=None, ensure_2d=False, accept_sparse=True
        )
        # 验证数据的合法性
        self._validate_data(
            X, y, reset=True, validate_separately=(check_params, check_params)
        )
        self.y_ = y
        return self

    def predict(self, X):
        # 确保模型已经拟合
        check_is_fitted(self)
        # 验证数据的合法性
        self._validate_data(
            X,
            force_all_finite=False,
            dtype=None,
            accept_sparse=True,
            ensure_2d=False,
            reset=False,
        )
        # 返回预测结果，重复使用已知的类标签
        return np.repeat(self.y_, _num_samples(X))

    def decision_function(self, X):
        # 确保模型已经拟合
        check_is_fitted(self)
        # 验证数据的合法性
        self._validate_data(
            X,
            force_all_finite=False,
            dtype=None,
            accept_sparse=True,
            ensure_2d=False,
            reset=False,
        )
        # 返回决策函数的结果，重复使用已知的类标签
        return np.repeat(self.y_, _num_samples(X))

    def predict_proba(self, X):
        # 确保模型已经拟合
        check_is_fitted(self)
        # 验证数据的合法性
        self._validate_data(
            X,
            force_all_finite=False,
            dtype=None,
            accept_sparse=True,
            ensure_2d=False,
            reset=False,
        )
        # 将已知的类标签转换为浮点数，并返回重复使用的预测概率
        y_ = self.y_.astype(np.float64)
        return np.repeat([np.hstack([1 - y_, y_])], _num_samples(X), axis=0)


def _estimators_has(attr):
    """Check if self.estimator or self.estimators_[0] has attr.

    If `self.estimators_[0]` has the attr, then its safe to assume that other
    estimators have it too. We raise the original `AttributeError` if `attr`
    does not exist. This function is used together with `available_if`.
    """
    
    def check(self):
        if hasattr(self, "estimators_"):
            # 如果self.estimators_[0]有attr属性，则假定其他估计器也有它
            getattr(self.estimators_[0], attr)
        else:
            # 否则检查self.estimator是否有attr属性
            getattr(self.estimator, attr)

        return True

    return check


class OneVsRestClassifier(
    MultiOutputMixin,
    ClassifierMixin,
    MetaEstimatorMixin,
    BaseEstimator,
):
    """One-vs-the-rest (OvR) multiclass strategy.

    Also known as one-vs-all, this strategy consists in fitting one classifier
    per class. For each classifier, the class is fitted against all the other
    # 实现一对多分类的策略，其中每个类别对应一个分类器，以增加计算效率（仅需要 `n_classes` 个分类器），
    # 同时具备可解释性。通过检查每个分类器可以获取关于该类别的知识。
    # 这是多类别分类中最常用的策略，也是一个合理的默认选择。
    
    # OneVsRestClassifier 也可以用于多标签分类。使用这个功能时，在调用 `.fit` 时需提供目标 `y` 的指示器矩阵。
    # 换句话说，目标标签应该格式化为一个二维的二进制（0/1）矩阵，其中 [i, j] == 1 表示样本 i 中存在标签 j。
    # 这个估计器使用二进制相关性方法执行多标签分类，即独立为每个标签训练一个二进制分类器。
    
    # 更多信息请参阅：:ref:`用户指南 <ovr_classification>`。
    
    # 参数
    # ----------
    # estimator : estimator 对象
    #     实现 :term:`fit` 的回归器或分类器。
    #     当传递分类器时，优先使用 :term:`decision_function`，如果不可用则退而使用 :term:`predict_proba`。
    #     当传递回归器时，使用 :term:`predict`。
    
    # n_jobs : int, 默认为 None
    #     用于计算的作业数：`n_classes` 个一对多问题并行计算。
    
    #     ``None`` 表示 1，除非在 :obj:`joblib.parallel_backend` 上下文中。
    #     ``-1`` 表示使用所有处理器。有关更多详细信息，请参见 :term:`术语表 <n_jobs>`。
    
    #     .. versionchanged:: 0.20
    #        `n_jobs` 默认从 1 更改为 None
    
    # verbose : int, 默认为 0
    #     详细级别，如果非零，则打印进度消息。
    #     小于 50，输出发送到 stderr。否则，输出发送到 stdout。
    #     随着详细级别的增加，消息的频率增加，每 10 次报告一次迭代。
    #     有关更多详细信息，请参见 :class:`joblib.Parallel`。
    
    #     .. versionadded:: 1.1
    
    # 属性
    # ----------
    # estimators_ : `n_classes` 个估计器的列表
    #     用于预测的估计器。
    
    # classes_ : 数组，形状为 [`n_classes`]
    #     类别标签。
    
    # n_classes_ : int
    #     类别的数量。
    
    # label_binarizer_ : LabelBinarizer 对象
    #     用于将多类标签转换为二进制标签及其反向转换的对象。
    
    # multilabel_ : 布尔值
    #     一个 OneVsRestClassifier 是否为多标签分类器。
    
    # n_features_in_ : int
    #     在 :term:`fit` 过程中看到的特征数。仅在底层估计器在 fit 时暴露此属性时才定义。
    
    #     .. versionadded:: 0.24
    # 存储特征名称的 ndarray，形状为 (`n_features_in_`,)
    # 在 :term:`fit` 过程中观察到的特征名称。只有在底层估计器在拟合时暴露此属性时才定义。

    # .. versionadded:: 1.0
    # 版本新增功能：从版本 1.0 开始可用

    See Also
    --------
    OneVsOneClassifier : 一对一多类策略。
    OutputCodeClassifier : (纠错) 输出码多类策略。
    sklearn.multioutput.MultiOutputClassifier : 扩展估计器进行多标签分类的另一种方法。
    sklearn.preprocessing.MultiLabelBinarizer : 将可迭代的可迭代对象转换为二进制指示器矩阵。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.multiclass import OneVsRestClassifier
    >>> from sklearn.svm import SVC
    >>> X = np.array([
    ...     [10, 10],
    ...     [8, 10],
    ...     [-5, 5.5],
    ...     [-5.4, 5.5],
    ...     [-20, -20],
    ...     [-15, -20]
    ... ])
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> clf = OneVsRestClassifier(SVC()).fit(X, y)
    >>> clf.predict([[-19, -20], [9, 9], [-5, 5]])
    array([2, 0, 1])
    """

    # 存储参数约束的字典
    _parameter_constraints = {
        "estimator": [HasMethods(["fit"])],  # 估计器需要具有 "fit" 方法
        "n_jobs": [Integral, None],  # 可接受整数或 None
        "verbose": ["verbose"],  # 可接受 "verbose" 字符串
    }

    # 初始化方法
    def __init__(self, estimator, *, n_jobs=None, verbose=0):
        self.estimator = estimator  # 初始化估计器
        self.n_jobs = n_jobs  # 初始化 n_jobs 参数
        self.verbose = verbose  # 初始化 verbose 参数

    @_fit_context(
        # OneVsRestClassifier.estimator 尚未经过验证
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **fit_params):
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        **fit_params : dict
            Parameters passed to the ``estimator.fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        _raise_for_params(fit_params, self, "fit")  # 检查并确保 fit_params 合法性

        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )
        # 使用稀疏输出的 LabelBinarizer，性能优于或与密集版本相匹配，并且在 fit_ovr 函数中也导致内存消耗更少或相等。
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)  # 对目标变量 y 进行二进制编码
        Y = Y.tocsc()  # 将 Y 转换为压缩稀疏列格式
        self.classes_ = self.label_binarizer_.classes_  # 记录类别标签

        columns = (col.toarray().ravel() for col in Y.T)
        # 对每一列进行并行训练，生成 OneVsRest 分类器的子估计器列表
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_binary)(
                self.estimator,
                X,
                column,
                fit_params=routed_params.estimator.fit,
                classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i],
                ],
            )
            for i, column in enumerate(columns)
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_  # 记录特征数目
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_  # 记录特征名称

        return self
    def partial_fit(self, X, y, classes=None, **partial_fit_params):
        """Partially fit underlying estimators.

        Should be used when memory is inefficient to train all data.
        Chunks of data can be passed in several iterations.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        **partial_fit_params : dict
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            Instance of partially fitted estimator.
        """
        # Validate and raise error for invalid parameters in partial_fit_params
        _raise_for_params(partial_fit_params, self, "partial_fit")

        # Process routing parameters for partial fitting
        routed_params = process_routing(
            self,
            "partial_fit",
            **partial_fit_params,
        )

        # Initialize estimators if this is the first call to partial_fit
        if _check_partial_fit_first_call(self, classes):
            # Create a list of cloned estimators based on the number of classes
            self.estimators_ = [clone(self.estimator) for _ in range(self.n_classes_)]

            # Initialize a LabelBinarizer for multi-label classification
            # using sparse output for efficient memory usage
            self.label_binarizer_ = LabelBinarizer(sparse_output=True)
            self.label_binarizer_.fit(self.classes_)

        # Check if y contains classes not present in self.classes_
        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(
                (
                    "Mini-batch contains {0} while classes " + "must be subset of {1}"
                ).format(np.unique(y), self.classes_)
            )

        # Transform multi-class target y into a sparse binary matrix Y
        Y = self.label_binarizer_.transform(y)
        Y = Y.tocsc()
        # Create an iterator over columns of Y for parallel processing
        columns = (col.toarray().ravel() for col in Y.T)

        # Perform parallel partial fitting on each estimator with respective column of Y
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_binary)(
                estimator,
                X,
                column,
                partial_fit_params=routed_params.estimator.partial_fit,
            )
            for estimator, column in zip(self.estimators_, columns)
        )

        # Set n_features_in_ attribute if available in estimators_[0]
        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_

        # Return the updated estimator instance
        return self
    # 使用预测方法进行多类别目标预测
    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Predicted multi-class targets.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 获取样本数
        n_samples = _num_samples(X)
        
        # 如果标签二值化器的类型是多类别
        if self.label_binarizer_.y_type_ == "multiclass":
            # 初始化一个数组来保存每个样本的最大预测值
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            # 初始化一个数组来保存对应于最大预测值的分类器索引
            argmaxima = np.zeros(n_samples, dtype=int)
            
            # 对于每个分类器进行预测
            for i, e in enumerate(self.estimators_):
                # 预测当前分类器的二进制输出
                pred = _predict_binary(e, X)
                # 更新最大预测值
                np.maximum(maxima, pred, out=maxima)
                # 更新对应最大预测值的分类器索引
                argmaxima[maxima == pred] = i
            
            # 返回对应于最大预测值的类别
            return self.classes_[argmaxima]
        
        else:
            # 对于二进制预测，计算阈值
            thresh = _threshold_for_binary_predict(self.estimators_[0])
            # 使用数组来存储预测结果的列索引
            indices = array.array("i")
            # 使用数组来存储每个分类器的指针索引
            indptr = array.array("i", [0])
            
            # 对于每个分类器进行预测
            for e in self.estimators_:
                # 将超过阈值的预测结果的索引扩展到 indices 数组中
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                # 更新 indptr 数组，指向新的 indices 结束位置
                indptr.append(len(indices))
            
            # 创建稀疏矩阵表示预测结果
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix(
                (data, indices, indptr), shape=(n_samples, len(self.estimators_))
            )
            
            # 对稀疏矩阵进行逆转换，返回预测的目标
            return self.label_binarizer_.inverse_transform(indicator)

    @available_if(_estimators_has("predict_proba"))
    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Note that in the multilabel case, each sample can have any number of
        labels. This returns the marginal probability that the given sample has
        the label in question. For example, it is entirely consistent that two
        labels both have a 90% probability of applying to a given sample.

        In the single label multiclass case, the rows of the returned matrix
        sum to 1.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        # Ensure that the model is fitted
        check_is_fitted(self)

        # Y[i, j] gives the probability that sample i has the label j.
        # In the multi-label case, these are not disjoint.
        # Predict probabilities from each estimator and transpose the result
        Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T

        if len(self.estimators_) == 1:
            # Only one estimator, but we still want to return probabilities
            # for two classes. Concatenate probabilities for the complementary class.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        if not self.multilabel_:
            # If not a multilabel classifier, normalize probabilities to sum to 1
            Y /= np.sum(Y, axis=1)[:, np.newaxis]

        return Y

    @available_if(_estimators_has("decision_function"))
    def decision_function(self, X):
        """Decision function for the OneVsRestClassifier.

        Return the distance of each sample from the decision boundary for each
        class. This can only be used with estimators which implement the
        `decision_function` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes) or (n_samples,) for \
            binary classification.
            Result of calling `decision_function` on the final estimator.

            .. versionchanged:: 0.19
                output shape changed to ``(n_samples,)`` to conform to
                scikit-learn conventions for binary classification.
        """
        # Ensure that the model is fitted
        check_is_fitted(self)

        if len(self.estimators_) == 1:
            # If there's only one estimator, return decision function directly
            return self.estimators_[0].decision_function(X)

        # Compute decision function for each estimator and transpose the result
        return np.array(
            [est.decision_function(X).ravel() for est in self.estimators_]
        ).T

    @property
    def multilabel_(self):
        """Whether this is a multilabel classifier."""
        # Check if the label binarizer indicates multilabel classification
        return self.label_binarizer_.y_type_.startswith("multilabel")

    @property
    def n_classes_(self):
        """Number of classes."""
        # Return the number of unique classes in the model
        return len(self.classes_)
    def _more_tags(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        # 返回一个字典，指示包装的估计器是否使用预先计算的 Gram 矩阵
        return {"pairwise": _safe_tags(self.estimator, key="pairwise")}

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，指定其所有者为当前对象的类名
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            # 添加当前对象自身的请求到路由器
            .add_self_request(self)
            # 添加估计器及其方法的映射到路由器
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping()
                # 添加调用者为 "fit"，被调用者为 "fit" 的映射
                .add(caller="fit", callee="fit")
                # 添加调用者为 "partial_fit"，被调用者为 "partial_fit" 的映射
                .add(caller="partial_fit", callee="partial_fit"),
            )
        )
        # 返回包含路由信息的 MetadataRouter 对象
        return router
def _fit_ovo_binary(estimator, X, y, i, j, fit_params):
    """Fit a single binary estimator (one-vs-one)."""
    # 创建一个布尔数组，标记所有等于类别 i 或 j 的样本
    cond = np.logical_or(y == i, y == j)
    # 从标记了 i 或 j 的样本中提取出对应的标签 y
    y = y[cond]
    # 创建一个与 y 形状相同的整数数组，用于表示二进制分类（0 或 1）
    y_binary = np.empty(y.shape, int)
    y_binary[y == i] = 0  # 将类别 i 对应的样本标记为 0
    y_binary[y == j] = 1  # 将类别 j 对应的样本标记为 1
    # 提取符合条件的样本的索引，并生成一个范围数组
    indcond = np.arange(_num_samples(X))[cond]

    # 检查方法参数并返回适用于当前子集的参数
    fit_params_subset = _check_method_params(X, params=fit_params, indices=indcond)
    # 调用 _fit_binary 函数进行二进制分类器的拟合
    return (
        _fit_binary(
            estimator,
            _safe_split(estimator, X, None, indices=indcond)[0],  # 安全地拆分 X，并选择对应的子集
            y_binary,
            fit_params=fit_params_subset,
            classes=[i, j],  # 指定当前分类器的类别标签
        ),
        indcond,  # 返回符合条件的样本的索引
    )


def _partial_fit_ovo_binary(estimator, X, y, i, j, partial_fit_params):
    """Partially fit a single binary estimator(one-vs-one)."""
    # 创建一个布尔数组，标记所有等于类别 i 或 j 的样本
    cond = np.logical_or(y == i, y == j)
    # 从标记了 i 或 j 的样本中提取出对应的标签 y
    y = y[cond]
    if len(y) != 0:
        # 创建一个与 y 形状相同的零填充整数数组，用于表示二进制分类（0 或 1）
        y_binary = np.zeros_like(y)
        y_binary[y == j] = 1  # 将类别 j 对应的样本标记为 1
        # 检查方法参数并返回适用于当前子集的参数
        partial_fit_params_subset = _check_method_params(
            X, params=partial_fit_params, indices=cond
        )
        # 调用 _partial_fit_binary 函数进行部分拟合
        return _partial_fit_binary(
            estimator, X[cond], y_binary, partial_fit_params=partial_fit_params_subset
        )
    return estimator  # 如果符合条件的样本为空，则直接返回原始估计器


class OneVsOneClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """One-vs-one multiclass strategy.

    This strategy consists in fitting one classifier per class pair.
    At prediction time, the class which received the most votes is selected.
    Since it requires to fit `n_classes * (n_classes - 1) / 2` classifiers,
    this method is usually slower than one-vs-the-rest, due to its
    O(n_classes^2) complexity. However, this method may be advantageous for
    algorithms such as kernel algorithms which don't scale well with
    `n_samples`. This is because each individual learning problem only involves
    a small subset of the data whereas, with one-vs-the-rest, the complete
    dataset is used `n_classes` times.

    Read more in the :ref:`User Guide <ovo_classification>`.

    Parameters
    ----------
    estimator : estimator object
        A regressor or a classifier that implements :term:`fit`.
        When a classifier is passed, :term:`decision_function` will be used
        in priority and it will fallback to :term:`predict_proba` if it is not
        available.
        When a regressor is passed, :term:`predict` is used.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the `n_classes * (
        n_classes - 1) / 2` OVO problems are computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    estimators_ : list of ``n_classes * (n_classes - 1) / 2`` estimators
        Estimators used for predictions.

    classes_ : numpy array of shape [n_classes]
        Array containing labels.

    n_classes_ : int
        Number of classes.
    """
    """
    pairwise_indices_ : list, length = ``len(estimators_)``, or ``None``
        训练估计器时使用的样本索引列表。
        当估计器的 `pairwise` 标签为 False 时为 `None`。
    
    n_features_in_ : int
        在 `fit` 过程中观察到的特征数量。
        
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 `fit` 过程中观察到的特征名称。仅当 `X` 的特征名称全为字符串时定义。
        
        .. versionadded:: 1.0
    
    See Also
    --------
    OneVsRestClassifier : 一对多的多类策略。
    OutputCodeClassifier : （纠错）输出代码的多类策略。
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.multiclass import OneVsOneClassifier
    >>> from sklearn.svm import LinearSVC
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, shuffle=True, random_state=0)
    >>> clf = OneVsOneClassifier(
    ...     LinearSVC(random_state=0)).fit(X_train, y_train)
    >>> clf.predict(X_test[:10])
    array([2, 1, 0, 2, 0, 2, 0, 1, 1, 1])
    """
    
    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "n_jobs": [Integral, None],
    }
    
    def __init__(self, estimator, *, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
    
    @_fit_context(
        # OneVsOneClassifier.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    # 使用给定的数据 X 和目标 y，使用 fit_params 字典中的参数拟合底层估计器
    def fit(self, X, y, **fit_params):
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Multi-class targets.

        **fit_params : dict
            Parameters passed to the ``estimator.fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            The fitted underlying estimator.
        """
        # 检查并抛出参数错误，如果有问题则引发异常
        _raise_for_params(fit_params, self, "fit")

        # 处理路由参数，根据路由策略调整传递给底层估计器的参数
        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )

        # 需要验证数据，因为稍后会进行安全索引
        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc"], force_all_finite=False
        )
        # 检查分类目标的有效性
        check_classification_targets(y)

        # 确定类别的唯一值
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError(
                "OneVsOneClassifier can not be fit when only one class is present."
            )
        n_classes = self.classes_.shape[0]

        # 并行处理每对类别之间的估计器拟合
        estimators_indices = list(
            zip(
                *(
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(_fit_ovo_binary)(
                            self.estimator,
                            X,
                            y,
                            self.classes_[i],
                            self.classes_[j],
                            fit_params=routed_params.estimator.fit,
                        )
                        for i in range(n_classes)
                        for j in range(i + 1, n_classes)
                    )
                )
            )
        )

        # 存储拟合好的估计器索引
        self.estimators_ = estimators_indices[0]

        # 根据是否是一对一分类器，决定是否存储相关索引
        pairwise = self._get_tags()["pairwise"]
        self.pairwise_indices_ = estimators_indices[1] if pairwise else None

        # 返回已拟合的分类器对象本身
        return self

    @available_if(_estimators_has("partial_fit"))
    @_fit_context(
        # OneVsOneClassifier.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def partial_fit(self, X, y, classes=None, **partial_fit_params):
        """部分拟合底层估计器。

        当内存不足以训练所有数据时应使用此方法。可以在多次迭代中传递数据块，
        其中第一次调用应传入所有目标变量的数组。

        Parameters
        ----------
        X : {array-like, sparse matrix) of shape (n_samples, n_features)
            数据。

        y : array-like of shape (n_samples,)
            多类别目标。

        classes : array, shape (n_classes, )
            所有 partial_fit 调用中的类。
            可以通过 `np.unique(y_all)` 获得，其中 y_all 是整个数据集的目标向量。
            此参数仅在 partial_fit 的第一次调用时需要，并且在后续调用中可以省略。

        **partial_fit_params : dict
            传递给每个子估计器的 ``estimator.partial_fit`` 方法的参数。

            .. versionadded:: 1.4
                仅在 `enable_metadata_routing=True` 时可用。有关更多详情，请参阅
                :ref:`元数据路由用户指南 <metadata_routing>`。

        Returns
        -------
        self : object
            部分拟合的底层估计器。
        """
        # 检查参数
        _raise_for_params(partial_fit_params, self, "partial_fit")

        # 处理路由参数
        routed_params = process_routing(
            self,
            "partial_fit",
            **partial_fit_params,
        )

        # 检查是否是第一次调用 partial_fit
        first_call = _check_partial_fit_first_call(self, classes)
        if first_call:
            # 初始化估计器列表
            self.estimators_ = [
                clone(self.estimator)
                for _ in range(self.n_classes_ * (self.n_classes_ - 1) // 2)
            ]

        # 检查 y 是否属于 self.classes_
        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(
                "Mini-batch contains {0} while it must be subset of {1}".format(
                    np.unique(y), self.classes_
                )
            )

        # 数据验证和预处理
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            force_all_finite=False,
            reset=first_call,
        )

        # 检查分类目标
        check_classification_targets(y)

        # 生成类别对的组合
        combinations = itertools.combinations(range(self.n_classes_), 2)

        # 并行拟合每个一对一（ovo）的二元分类器
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_ovo_binary)(
                estimator,
                X,
                y,
                self.classes_[i],
                self.classes_[j],
                partial_fit_params=routed_params.estimator.partial_fit,
            )
            for estimator, (i, j) in zip(self.estimators_, combinations)
        )

        # 清空 pairwise_indices_
        self.pairwise_indices_ = None

        # 如果估计器具有 n_features_in_ 属性，则设置 self.n_features_in_
        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_

        return self
    def predict(self, X):
        """
        Estimate the best class label for each sample in X.

        This is implemented as ``argmax(decision_function(X), axis=1)`` which
        will return the label of the class with most votes by estimators
        predicting the outcome of a decision for each possible class pair.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        # Calculate decision values for samples
        Y = self.decision_function(X)
        
        # Handle binary classification case
        if self.n_classes_ == 2:
            # Compute threshold for binary prediction
            thresh = _threshold_for_binary_predict(self.estimators_[0])
            # Return classes based on decision values and threshold
            return self.classes_[(Y > thresh).astype(int)]
        
        # Return classes with maximum decision values
        return self.classes_[Y.argmax(axis=1)]

    def decision_function(self, X):
        """
        Decision function for the OneVsOneClassifier.

        The decision values for the samples are computed by adding the
        normalized sum of pair-wise classification confidence levels to the
        votes in order to disambiguate between the decision values when the
        votes for all the classes are equal leading to a tie.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Y : array-like of shape (n_samples, n_classes) or (n_samples,)
            Result of calling `decision_function` on the final estimator.

            .. versionchanged:: 0.19
                output shape changed to ``(n_samples,)`` to conform to
                scikit-learn conventions for binary classification.
        """
        # Ensure the estimator is fitted
        check_is_fitted(self)
        
        # Validate input data
        X = self._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )

        # Get pairwise indices if available
        indices = self.pairwise_indices_
        
        # Prepare data subsets for each estimator
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            Xs = [X[:, idx] for idx in indices]

        # Make predictions for each subset of data
        predictions = np.vstack(
            [est.predict(Xi) for est, Xi in zip(self.estimators_, Xs)]
        ).T
        
        # Compute confidence scores for binary predictions
        confidences = np.vstack(
            [_predict_binary(est, Xi) for est, Xi in zip(self.estimators_, Xs)]
        ).T
        
        # Compute final decision function
        Y = _ovr_decision_function(predictions, confidences, len(self.classes_))
        
        # Return appropriate shape based on number of classes
        if self.n_classes_ == 2:
            return Y[:, 1]  # For binary classification, return positive class decision value
        return Y  # For multi-class classification, return decision values

    @property
    def n_classes_(self):
        """Number of classes."""
        return len(self.classes_)

    def _more_tags(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return {"pairwise": _safe_tags(self.estimator, key="pairwise")}
    def get_metadata_routing(self):
        """获取此对象的元数据路由。

        请查看 :ref:`用户指南 <metadata_routing>` 了解路由机制的工作原理。

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            一个 :class:`~sklearn.utils.metadata_routing.MetadataRouter` 对象，封装了路由信息。
        """

        # 创建一个 MetadataRouter 对象，指定所有者为当前类名
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            # 添加当前对象自身的请求到路由中
            .add_self_request(self)
            # 添加估计器的元数据路由信息
            .add(
                estimator=self.estimator,
                # 使用 MethodMapping 对象添加方法映射关系
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")  # 添加 fit 方法的映射
                .add(caller="partial_fit", callee="partial_fit"),  # 添加 partial_fit 方法的映射
            )
        )
        # 返回构建好的路由对象
        return router
# 定义一个多类输出码分类器，继承自MetaEstimatorMixin、ClassifierMixin和BaseEstimator类
class OutputCodeClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """(Error-Correcting) Output-Code multiclass strategy.

    Output-code based strategies consist in representing each class with a
    binary code (an array of 0s and 1s). At fitting time, one binary
    classifier per bit in the code book is fitted.  At prediction time, the
    classifiers are used to project new points in the class space and the class
    closest to the points is chosen. The main advantage of these strategies is
    that the number of classifiers used can be controlled by the user, either
    for compressing the model (0 < `code_size` < 1) or for making the model more
    robust to errors (`code_size` > 1). See the documentation for more details.

    Read more in the :ref:`User Guide <ecoc>`.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and one of
        :term:`decision_function` or :term:`predict_proba`.

    code_size : float, default=1.5
        Percentage of the number of classes to be used to create the code book.
        A number between 0 and 1 will require fewer classifiers than
        one-vs-the-rest. A number greater than 1 will require more classifiers
        than one-vs-the-rest.

    random_state : int, RandomState instance, default=None
        The generator used to initialize the codebook.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the multiclass problems
        are computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    estimators_ : list of `int(n_classes * code_size)` estimators
        Estimators used for predictions.

    classes_ : ndarray of shape (n_classes,)
        Array containing labels.

    code_book_ : ndarray of shape (n_classes, `len(estimators_)`)
        Binary array containing the code of each class.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    OneVsRestClassifier : One-vs-all multiclass strategy.
    OneVsOneClassifier : One-vs-one multiclass strategy.

    References
    ----------

    .. [1] "Solving multiclass learning problems via error-correcting output
       codes",
       Dietterich T., Bakiri G.,
       Journal of Artificial Intelligence Research 2,
       1995.

    """
    # 定义参数约束字典，用于描述输出编码分类器的各个参数的类型和限制条件
    _parameter_constraints: dict = {
        "estimator": [
            HasMethods(["fit", "decision_function"]),  # estimator 参数要求具有 fit 和 decision_function 方法
            HasMethods(["fit", "predict_proba"]),      # estimator 参数要求具有 fit 和 predict_proba 方法
        ],
        "code_size": [Interval(Real, 0.0, None, closed="neither")],  # code_size 参数要求为实数，大于0
        "random_state": ["random_state"],  # random_state 参数只接受 random_state 类型
        "n_jobs": [Integral, None],  # n_jobs 参数可以为整数或 None
    }
    
    # 定义输出编码分类器类
    def __init__(self, estimator, *, code_size=1.5, random_state=None, n_jobs=None):
        self.estimator = estimator  # 初始化 estimator 参数
        self.code_size = code_size  # 初始化 code_size 参数，默认为 1.5
        self.random_state = random_state  # 初始化 random_state 参数
        self.n_jobs = n_jobs  # 初始化 n_jobs 参数
    
    # 应用 _fit_context 装饰器来包装方法
    @_fit_context(
        prefer_skip_nested_validation=False  # 设置 prefer_skip_nested_validation 参数为 False
    )
    def fit(self, X, y, **fit_params):
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            数据集。

        y : array-like of shape (n_samples,)
            多类别目标值。

        **fit_params : dict
            传递给每个子估计器的 ``estimator.fit`` 方法的参数。

            .. versionadded:: 1.4
                仅当 `enable_metadata_routing=True` 时可用。详见
                :ref:`Metadata Routing User Guide <metadata_routing>` 获取更多详情。

        Returns
        -------
        self : object
            返回已拟合的对象实例。
        """
        _raise_for_params(fit_params, self, "fit")  # 检查参数合法性

        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )  # 处理路由参数

        y = self._validate_data(X="no_validation", y=y)  # 验证数据有效性

        random_state = check_random_state(self.random_state)  # 检查并获取随机数生成器
        check_classification_targets(y)  # 检查分类目标的有效性

        self.classes_ = np.unique(y)  # 获取类别标签的唯一值
        n_classes = self.classes_.shape[0]  # 获取类别数
        if n_classes == 0:
            raise ValueError(
                "OutputCodeClassifier can not be fit when no class is present."
            )  # 如果没有类别标签，则抛出异常

        n_estimators = int(n_classes * self.code_size)  # 计算估计器数量

        # FIXME: there are more elaborate methods than generating the codebook
        # randomly.
        self.code_book_ = random_state.uniform(size=(n_classes, n_estimators))
        self.code_book_[self.code_book_ > 0.5] = 1.0  # 随机生成编码簿，并将大于0.5的值设为1.0

        if hasattr(self.estimator, "decision_function"):
            self.code_book_[self.code_book_ != 1] = -1.0  # 如果估计器有决策函数，则非1的值设为-1.0
        else:
            self.code_book_[self.code_book_ != 1] = 0.0  # 否则非1的值设为0.0

        classes_index = {c: i for i, c in enumerate(self.classes_)}  # 创建类别索引字典

        Y = np.array(
            [self.code_book_[classes_index[y[i]]] for i in range(_num_samples(y))],
            dtype=int,
        )  # 构建编码矩阵Y，根据类别索引和编码簿

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_binary)(
                self.estimator, X, Y[:, i], fit_params=routed_params.estimator.fit
            )  # 并行拟合二进制分类器
            for i in range(Y.shape[1])
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_  # 设置输入特征数属性
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_  # 设置输入特征名称属性

        return self  # 返回已拟合的对象实例
    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted multi-class targets.
        """
        # 确保模型已经拟合，即已经训练过
        check_is_fitted(self)
        
        # 为了使用 ArgKmin，需要一个 C-contiguous 数组。聚合预测结果需要进行转置。
        # 因此我们创建一个 F-contiguous 数组，以避免复制，并在转置操作后得到
        # 一个 C-contiguous 数组。
        Y = np.array(
            [_predict_binary(e, X) for e in self.estimators_],
            order="F",
            dtype=np.float64,
        ).T
        
        # 使用欧氏距离找到最接近的聚类中心的索引
        pred = pairwise_distances_argmin(Y, self.code_book_, metric="euclidean")
        
        # 返回预测的类别标签
        return self.classes_[pred]

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，用于管理元数据的路由信息
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        
        # 返回封装了路由信息的 MetadataRouter 对象
        return router
```