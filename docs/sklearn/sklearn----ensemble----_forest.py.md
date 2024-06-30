# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_forest.py`

```
    # 计算用于bootstrap采样的样本数量
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    ```
    # 如果 max_samples 参数为 None，则直接返回原始数据集中的样本数
    if max_samples is None:
        return n_samples
    
    # 如果 max_samples 参数是整数类型
    if isinstance(max_samples, Integral):
        # 检查 max_samples 是否大于数据集中的样本数，若是则抛出数值错误异常
        if max_samples > n_samples:
            msg = "`max_samples` must be <= n_samples={} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        # 否则返回 max_samples 作为 bootstrap 样本的样本数
        return max_samples
    
    # 如果 max_samples 参数是实数类型
    if isinstance(max_samples, Real):
        # 计算出用于 bootstrap 的样本数，取整并确保至少为1个样本
        return max(round(n_samples * max_samples), 1)
# 生成一个随机数生成器实例，用于随机选择样本索引
def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    random_instance = check_random_state(random_state)
    # 生成一个包含 n_samples_bootstrap 个随机整数的数组，表示样本的索引
    sample_indices = random_instance.randint(
        0, n_samples, n_samples_bootstrap, dtype=np.int32
    )

    return sample_indices


# 生成一个未被抽样的样本索引数组
def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    # 调用 _generate_sample_indices 函数生成抽样索引
    sample_indices = _generate_sample_indices(
        random_state, n_samples, n_samples_bootstrap
    )
    # 统计每个样本索引出现的次数
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    # 创建一个布尔掩码数组，表示未被抽样的样本索引
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    # 从索引范围中选择未被抽样的索引
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


# 并行构建单棵树的私有函数
def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
    missing_values_in_feature_mask=None,
):
    # 如果 verbose 大于 1，打印正在构建的树的信息
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    # 如果使用 bootstrap 方法
    if bootstrap:
        n_samples = X.shape[0]
        # 如果未指定样本权重，创建一个全为1的样本权重数组
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        # 生成抽样索引
        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )
        # 统计每个样本索引出现的次数，并更新样本权重
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        # 如果类别权重为 "subsample"，根据样本索引更新样本权重
        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        # 如果类别权重为 "balanced_subsample"，根据样本索引平衡更新样本权重
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        # 调用 tree 对象的 _fit 方法，拟合数据
        tree._fit(
            X,
            y,
            sample_weight=curr_sample_weight,
            check_input=False,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )
    else:
        # 使用给定的样本权重调用 tree 对象的 _fit 方法，拟合数据
        tree._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=False,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )

    return tree


class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    # 定义参数约束字典，指定每个参数的类型和取值范围
    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "bootstrap": ["boolean"],
        "oob_score": ["boolean", callable],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
        "max_samples": [
            None,
            Interval(RealNotInt, 0.0, 1.0, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
    }

    @abstractmethod
    # 构造函数初始化方法，继承自父类并设置了多个参数
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        # 调用父类构造函数，设置基础的估计器和估计器参数
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

        # 初始化随机森林对象的属性
        self.bootstrap = bootstrap  # 是否启用自助法
        self.oob_score = oob_score  # 是否计算 out-of-bag 分数
        self.n_jobs = n_jobs  # 并行运行的作业数
        self.random_state = random_state  # 控制随机性的随机种子
        self.verbose = verbose  # 控制详细程度的整数值
        self.warm_start = warm_start  # 是否使用前一个调用的解决方案来拟合新增的估计器
        self.class_weight = class_weight  # 类别权重
        self.max_samples = max_samples  # 每个估计器的最大样本数

    # 应用方法，将输入的样本 X 应用于森林中的每棵树，并返回叶子索引
    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        # 验证并预测输入的 X 数据
        X = self._validate_X_predict(X)
        # 并行地应用每棵树对 X 进行预测，返回每个树的叶子索引结果
        results = Parallel(
            n_jobs=self.n_jobs,  # 并行作业数
            verbose=self.verbose,  # 详细程度
            prefer="threads",  # 使用线程作为首选方法
        )(delayed(tree.apply)(X, check_input=False) for tree in self.estimators_)

        # 将结果转换为 ndarray 并返回，结果的转置表示每个样本对应每棵树的叶子索引
        return np.array(results).T
    # 定义决策路径函数，返回随机森林中每棵树的决策路径

    @_fit_context(prefer_skip_nested_validation=True)
    @abstractmethod
    # 使用装饰器设置上下文以跳过嵌套验证，并声明该方法为抽象方法
    def _set_oob_score_and_attributes(self, X, y, scoring_function=None):
        """Compute and set the OOB score and attributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
            数据矩阵，包含 n_samples 行和 n_features 列
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
            目标矩阵，包含 n_samples 行和 n_outputs 列
        scoring_function : callable, default=None
            Scoring function for OOB score. Default depends on whether
            this is a regression (R2 score) or classification problem
            (accuracy score).
            用于计算 OOB 分数的评分函数，默认根据是回归（R2 分数）还是分类问题（准确率分数）而定
        """
    # 计算并设置OOB（Out-of-Bag）得分。

    def _compute_oob_predictions(self, X, y):
        """Compute and set the OOB score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            数据矩阵。
        y : ndarray of shape (n_samples, n_outputs)
            目标矩阵。

        Returns
        -------
        oob_pred : ndarray of shape (n_samples, n_classes, n_outputs) or \
                (n_samples, 1, n_outputs)
            OOB 预测结果。
        """
        # 需要将 X 转换成 CSR 格式才能进行预测
        if issparse(X):
            X = X.tocsr()

        n_samples = y.shape[0]
        n_outputs = self.n_outputs_
        if is_classifier(self) and hasattr(self, "n_classes_"):
            # 在这个阶段，n_classes_ 是一个 ndarray
            # 所有支持的目标类型在所有输出中都具有相同数量的类别
            oob_pred_shape = (n_samples, self.n_classes_[0], n_outputs)
        else:
            # 对于回归问题，n_classes_ 不存在，我们创建一个空轴以保持与分类情况一致，
            # 并使数组操作与两种设置兼容
            oob_pred_shape = (n_samples, 1, n_outputs)

        oob_pred = np.zeros(shape=oob_pred_shape, dtype=np.float64)
        n_oob_pred = np.zeros((n_samples, n_outputs), dtype=np.int64)

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples,
            self.max_samples,
        )
        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state,
                n_samples,
                n_samples_bootstrap,
            )

            y_pred = self._get_oob_predictions(estimator, X[unsampled_indices, :])
            oob_pred[unsampled_indices, ...] += y_pred
            n_oob_pred[unsampled_indices, :] += 1

        for k in range(n_outputs):
            if (n_oob_pred == 0).any():
                warn(
                    (
                        "Some inputs do not have OOB scores. This probably means "
                        "too few trees were used to compute any reliable OOB "
                        "estimates."
                    ),
                    UserWarning,
                )
                n_oob_pred[n_oob_pred == 0] = 1
            oob_pred[..., k] /= n_oob_pred[..., [k]]

        return oob_pred

    def _validate_y_class_weight(self, y):
        # 默认实现，直接返回 y 和 None
        return y, None
    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        
        # 如果第一个估计器支持处理缺失值，则设置允许NaN，否则设置为True
        if self.estimators_[0]._support_missing_values(X):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True
        
        # 验证输入数据X的格式和类型
        X = self._validate_data(
            X,
            dtype=DTYPE,
            accept_sparse="csr",
            reset=False,
            force_all_finite=force_all_finite,
        )
        
        # 如果X是稀疏矩阵且索引不是np.intc类型，则抛出错误
        if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based sparse matrices")
        
        return X

    @property
    def feature_importances_(self):
        """
        The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)

        # 并行计算每棵树的特征重要性
        all_importances = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(getattr)(tree, "feature_importances_")
            for tree in self.estimators_
            if tree.tree_.node_count > 1
        )

        # 如果没有任何特征重要性，则返回全零数组
        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)

        # 计算所有树的平均特征重要性，并进行归一化
        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    def _get_estimators_indices(self):
        """
        Get drawn indices along both sample and feature axes.
        """
        # 遍历所有估计器，生成样本索引
        for tree in self.estimators_:
            if not self.bootstrap:
                yield np.arange(self._n_samples, dtype=np.int32)
            else:
                # 使用每棵树的随机种子生成样本索引
                seed = tree.random_state
                yield _generate_sample_indices(
                    seed, self._n_samples, self._n_samples_bootstrap
                )

    @property
    def estimators_samples_(self):
        """
        返回每个基本估算器所用的抽样样本的子集。

        返回一个动态生成的索引列表，用于标识用于拟合集合的每个成员的样本，即袋内样本。

        注意：为了通过不存储抽样数据来减少对象的内存占用，每次调用该属性时都会重新创建列表。
        因此，获取属性可能比预期的要慢。
        """
        return [sample_indices for sample_indices in self._get_estimators_indices()]

    def _more_tags(self):
        """
        只有标准是必需的，以确定树是否支持缺失值。

        estimator 是使用 self.estimator 类型创建的对象，其中 criterion 参数设为 self.criterion。
        返回一个字典，包含有关估算器特性的信息，如是否允许 NaN。
        """
        estimator = type(self.estimator)(criterion=self.criterion)
        return {"allow_nan": _safe_tags(estimator, key="allow_nan")}
# 这是一个用于 joblib 的 Parallel 的实用函数，用于并行计算预测结果。

def _accumulate_prediction(predict, X, out, lock):
    # 使用给定的预测函数 `predict` 对输入数据 `X` 进行预测，禁用输入检查
    prediction = predict(X, check_input=False)
    
    # 使用锁确保多线程安全地更新输出结果 `out`
    with lock:
        # 如果 `out` 只有一个元素，则将预测结果累加到这个元素上
        if len(out) == 1:
            out[0] += prediction
        else:
            # 否则，对每个输出元素逐个累加预测结果中的对应部分
            for i in range(len(out)):
                out[i] += prediction[i]


class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
    """
    基于树的森林分类器的基类。

    警告: 不应直接使用此类，请使用派生类。
    """

    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        # 调用父类构造函数初始化森林分类器的参数
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

    @staticmethod
    def _get_oob_predictions(tree, X):
        """计算单棵决策树的袋外预测。

        Parameters
        ----------
        tree : DecisionTreeClassifier 对象
            单棵决策树分类器。
        X : 形状为 (n_samples, n_features) 的 ndarray
            袋外样本。

        Returns
        -------
        y_pred : 形状为 (n_samples, n_classes, n_outputs) 的 ndarray
            与袋外样本相关联的预测结果。
        """
        # 使用决策树 `tree` 对输入数据 `X` 进行概率预测，禁用输入检查
        y_pred = tree.predict_proba(X, check_input=False)
        y_pred = np.asarray(y_pred)
        
        # 如果预测结果维度为 2，则为二元分类或多类分类
        if y_pred.ndim == 2:
            y_pred = y_pred[..., np.newaxis]
        else:
            # 将第一个 `n_outputs` 轴滚动到最后一个轴上，将形状从 (n_outputs, n_samples, n_classes) 转换为 (n_samples, n_classes, n_outputs)
            y_pred = np.rollaxis(y_pred, axis=0, start=3)
        
        return y_pred
    # 计算并设置袋外（Out-of-Bag）得分和相关属性

    # 使用父类方法计算袋外预测函数值并赋给实例属性
    self.oob_decision_function_ = super()._compute_oob_predictions(X, y)

    # 如果袋外预测函数值的最后一个维度长度为1，则去除该维度，适用于单一输出的情况
    if self.oob_decision_function_.shape[-1] == 1:
        self.oob_decision_function_ = self.oob_decision_function_.squeeze(axis=-1)

    # 如果未指定评分函数，则默认使用 accuracy_score
    if scoring_function is None:
        scoring_function = accuracy_score

    # 计算袋外得分并赋给实例属性 oob_score_
    self.oob_score_ = scoring_function(
        y, np.argmax(self.oob_decision_function_, axis=1)
    )
    # 验证分类目标的有效性，检查是否是有效的分类目标
    check_classification_targets(y)

    # 复制 y 数组，防止原始数据被修改
    y = np.copy(y)
    # 初始化扩展的类别权重为空
    expanded_class_weight = None

    # 如果指定了 class_weight 参数
    if self.class_weight is not None:
        # 复制一份原始的 y 数组备份
        y_original = np.copy(y)

    # 初始化存储唯一索引的数组，用于存储每个输出的唯一索引
    y_store_unique_indices = np.zeros(y.shape, dtype=int)

    # 对于每个输出，进行处理
    for k in range(self.n_outputs_):
        # 获取当前输出的唯一类别和对应的索引
        classes_k, y_store_unique_indices[:, k] = np.unique(
            y[:, k], return_inverse=True
        )
        # 将当前输出的类别列表加入 self.classes_
        self.classes_.append(classes_k)
        # 记录当前输出的类别数量
        self.n_classes_.append(classes_k.shape[0])

    # 更新 y 为存储唯一索引的数组
    y = y_store_unique_indices

    # 如果指定了 class_weight 参数
    if self.class_weight is not None:
        # 允许的预设值
        valid_presets = ("balanced", "balanced_subsample")
        # 如果 class_weight 是字符串类型
        if isinstance(self.class_weight, str):
            # 如果指定的 class_weight 不在允许的预设值内，抛出 ValueError 异常
            if self.class_weight not in valid_presets:
                raise ValueError(
                    "Valid presets for class_weight include "
                    '"balanced" and "balanced_subsample".'
                    'Given "%s".' % self.class_weight
                )
            # 如果使用了 warm_start 参数，则警告不推荐使用 "balanced" 和 "balanced_subsample"
            if self.warm_start:
                warn(
                    'class_weight presets "balanced" or '
                    '"balanced_subsample" are '
                    "not recommended for warm_start if the fitted data "
                    "differs from the full dataset. In order to use "
                    '"balanced" weights, use compute_class_weight '
                    '("balanced", classes, y). In place of y you can use '
                    "a large enough sample of the full training set "
                    "target to properly estimate the class frequency "
                    "distributions. Pass the resulting weights as the "
                    "class_weight parameter."
                )

        # 如果不是 "balanced_subsample" 或者不使用 bootstrap
        if self.class_weight != "balanced_subsample" or not self.bootstrap:
            # 如果 class_weight 是 "balanced_subsample"，将其替换为 "balanced"
            if self.class_weight == "balanced_subsample":
                class_weight = "balanced"
            else:
                class_weight = self.class_weight
            # 计算扩展的类别权重
            expanded_class_weight = compute_sample_weight(class_weight, y_original)

    # 返回处理后的 y 和扩展的类别权重
    return y, expanded_class_weight
    def predict(self, X):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        # 使用预测概率函数进行预测
        proba = self.predict_proba(X)

        # 如果模型只有一个输出，则直接返回具有最大概率的类别
        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            # 否则，处理多输出的情况
            n_samples = proba[0].shape[0]
            # 所有类别的数据类型应该相同，因此只需取第一个类别的数据类型
            class_type = self.classes_[0].dtype
            # 创建一个空的预测数组，用于存储每个输出的预测结果
            predictions = np.empty((n_samples, self.n_outputs_), dtype=class_type)

            # 对每个输出进行循环，选择具有最大概率的类别作为预测结果
            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(
                    np.argmax(proba[k], axis=1), axis=0
                )

            return predictions
    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # Ensure the model is fitted
        check_is_fitted(self)
        # Validate and preprocess input data
        X = self._validate_X_predict(X)

        # Determine number of jobs and tree estimators partitioning
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Initialize array to accumulate probabilities from all estimators
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]

        # Thread-safe lock for accumulation operation
        lock = threading.Lock()

        # Parallel prediction aggregation from each tree estimator
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_
        )

        # Average the accumulated probabilities across all estimators
        for proba in all_proba:
            proba /= len(self.estimators_)

        # Return either a single array of probabilities or a list of arrays
        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict_log_proba(self, X):
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # Compute class probabilities using predict_proba method
        proba = self.predict_proba(X)

        # Apply logarithm to the probabilities for each output dimension
        if self.n_outputs_ == 1:
            return np.log(proba)
        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

    def _more_tags(self):
        """
        Returns additional tags for the estimator.

        Returns
        -------
        dict
            A dictionary with additional tags, e.g., {'multilabel': True}.
        """
        return {"multilabel": True}
    """
    Base class for forest of trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    # 林木回归器的基类，继承自RegressorMixin和BaseForest，使用ABCMeta元类

    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        # 构造函数，初始化回归器参数和森林的基本属性
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        # 检查模型是否已经拟合数据
        check_is_fitted(self)

        # 验证预测数据的有效性
        X = self._validate_X_predict(X)

        # 将森林中的树分配给工作任务
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # 如果有多个输出，初始化预测值数组
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # 并行执行预测
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.estimators_
        )

        # 计算预测的平均值
        y_hat /= len(self.estimators_)

        return y_hat

    @staticmethod
    def _get_oob_predictions(tree, X):
        """Compute the OOB predictions for an individual tree.

        Parameters
        ----------
        tree : DecisionTreeRegressor object
            A single decision tree regressor.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1, n_outputs)
            The OOB associated predictions.
        """
        # Predict using the provided tree on out-of-bag samples
        y_pred = tree.predict(X, check_input=False)
        if y_pred.ndim == 1:
            # If there's only one output, reshape to (n_samples, 1, 1)
            y_pred = y_pred[:, np.newaxis, np.newaxis]
        else:
            # If there are multiple outputs, reshape to (n_samples, 1, n_outputs)
            y_pred = y_pred[:, np.newaxis, :]
        return y_pred

    def _set_oob_score_and_attributes(self, X, y, scoring_function=None):
        """Compute and set the OOB score and attributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        scoring_function : callable, default=None
            Scoring function for OOB score. Defaults to `r2_score`.
        """
        # Compute the OOB predictions and squeeze to remove axis of size 1
        self.oob_prediction_ = super()._compute_oob_predictions(X, y).squeeze(axis=1)
        if self.oob_prediction_.shape[-1] == 1:
            # If there's only one output, remove the axis
            self.oob_prediction_ = self.oob_prediction_.squeeze(axis=-1)

        if scoring_function is None:
            scoring_function = r2_score

        # Compute the OOB score using the specified scoring function
        self.oob_score_ = scoring_function(y, self.oob_prediction_)

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features), dtype=DTYPE
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features), dtype=np.intp
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape (n_samples,)
            The value of the partial dependence function on each grid point.
        """
        # Convert grid and target_features to appropriate types and orders
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        target_features = np.asarray(target_features, dtype=np.intp, order="C")

        # Initialize an array to store averaged predictions
        averaged_predictions = np.zeros(
            shape=grid.shape[0], dtype=np.float64, order="C"
        )

        # Compute partial dependence for each tree in the forest
        for tree in self.estimators_:
            # Note: Calculation is done sequentially due to GIL constraints
            tree.tree_.compute_partial_dependence(
                grid, target_features, averaged_predictions
            )
        
        # Average predictions over all trees in the forest
        averaged_predictions /= len(self.estimators_)

        return averaged_predictions
    # 定义一个方法 `_more_tags`，返回一个包含 `multilabel` 键且其值为 `True` 的字典
    def _more_tags(self):
        # 返回包含 `multilabel` 键的字典，其值为 `True`
        return {"multilabel": True}
class RandomForestClassifier(ForestClassifier):
    """
    A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    Trees in the forest use the best split strategy, i.e. equivalent to passing
    `splitter="best"` to the underlying :class:`~sklearn.tree.DecisionTreeRegressor`.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.

    For a comparison between tree-based ensemble models see the example
    :ref:`sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py`.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.
        Note: This parameter is tree-specific.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.
    """
    # 定义随机森林分类器类，继承自ForestClassifier

    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1):
        """
        Initialize a random forest classifier.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.

        criterion : {"gini", "entropy", "log_loss"}, default="gini"
            The function to measure the quality of a split.

        max_depth : int, default=None
            The maximum depth of the trees.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
        """
        # 初始化方法，设置随机森林分类器的各种参数
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth",
                              "min_samples_split", "min_samples_leaf"),
            splitter="best",
            bootstrap=True,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf)
        # 调用父类的初始化方法，设定基础决策树分类器、树的数量、参数等
    min_weight_fraction_leaf : float, default=0.0
        # 最小的加权样本权重总和比例，用于作为叶节点的条件。当未提供 sample_weight 时，所有样本的权重相等。

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        # 在寻找最佳分割时考虑的最大特征数：

        - 如果是 int，则每次分割考虑 `max_features` 个特征。
        - 如果是 float，则 `max_features` 是一个分数，每次分割考虑 `max(1, int(max_features * n_features_in_))` 个特征。
        - 如果是 "sqrt"，则 `max_features=sqrt(n_features)`。
        - 如果是 "log2"，则 `max_features=log2(n_features)`。
        - 如果是 None，则 `max_features=n_features`。

        .. versionchanged:: 1.1
            `max_features` 的默认值从 `"auto"` 改为 `"sqrt"`。

        注意：即使需要实际检查超过 `max_features` 个特征才能找到有效的节点样本分区，搜索分割也不会停止。

    max_leaf_nodes : int, default=None
        # 以最佳优先方式生长具有 `max_leaf_nodes` 个叶节点的树。最佳节点的定义是基于杂质的相对减少。
        如果为 None，则叶节点数目不受限制。

    min_impurity_decrease : float, default=0.0
        # 如果此分割导致杂质减少大于或等于此值，则会分裂节点。

        加权杂质减少的方程如下所示：

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        其中，`N` 是样本的总数，`N_t` 是当前节点的样本数，`N_t_L` 是左子节点的样本数，`N_t_R` 是右子节点的样本数。

        如果传递了 `sample_weight`，所有的 `N`、`N_t`、`N_t_R` 和 `N_t_L` 都是加权和。

        .. versionadded:: 0.19

    bootstrap : bool, default=True
        # 构建树时是否使用自助样本。如果为 False，则使用整个数据集构建每棵树。

    oob_score : bool or callable, default=False
        # 是否使用袋外样本来估计泛化分数。默认使用 :func:`~sklearn.metrics.accuracy_score`。
        如果想使用自定义度量标准，提供一个带有 `metric(y_true, y_pred)` 签名的可调用对象。仅在 `bootstrap=True` 时可用。

    n_jobs : int, default=None
        # 并行运行的作业数。:meth:`fit`, :meth:`predict`, :meth:`decision_path` 和 :meth:`apply` 都在树上并行化。
        `None` 表示使用 1 个作业，除非在 :obj:`joblib.parallel_backend` 上下文中。`-1` 表示使用所有处理器。有关更多详细信息，请参阅 :term:`术语表 <n_jobs>`。
    random_state : int, RandomState instance or None, default=None
        # 控制随机性，用于样本的自助抽样（如果 bootstrap=True），以及在每个节点查找最佳分裂时考虑的特征的抽样。
        # 详细信息请参见术语表中的 "random_state"。

    verbose : int, default=0
        # 控制拟合和预测过程中的冗余输出级别。

    warm_start : bool, default=False
        # 如果设置为 True，则重复利用上一次调用 fit 的解决方案，并添加更多的估计器到集成中；否则，重新拟合整个随机森林。
        # 详细信息请参见术语表中的 "warm_start" 和 "tree_ensemble_warm_start"。

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        # 类别权重，形式为 `{类标签: 权重}`。
        # 如果未提供，则假定所有类别的权重均为一。对于多输出问题，可以按列提供一组字典，与 y 的列顺序相同。
        # 注意，对于多输出（包括多标签）问题，应为每列中的每个类别定义权重。例如，对于四类多标签分类，权重应为
        # [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]，而不是 [{1:1}, {2:5}, {3:1}, {4:1}]。
        # "balanced" 模式根据输入数据中类别频率的倒数自动调整权重，公式为 `n_samples / (n_classes * np.bincount(y))`。
        # "balanced_subsample" 模式与 "balanced" 相同，但权重基于每棵树的自助样本计算。
        # 对于多输出问题，将乘以 y 的每列的权重。
        # 注意，如果指定了 sample_weight（通过 fit 方法传递），这些权重将与 sample_weight 相乘。

    ccp_alpha : non-negative float, default=0.0
        # 用于最小成本复杂度剪枝的复杂度参数。选择比 `ccp_alpha` 小的最大成本复杂度的子树。默认情况下，不执行剪枝。
        # 详细信息请参见 "minimal_cost_complexity_pruning"。

        .. versionadded:: 0.22

    max_samples : int or float, default=None
        # 如果 bootstrap=True，则从 X 中抽取的样本数，用于训练每个基本估计器。

        - 如果为 None（默认），则抽取 `X.shape[0]` 个样本。
        - 如果为整数，则抽取 `max_samples` 个样本。
        - 如果为浮点数，则抽取 `max(round(n_samples * max_samples), 1)` 个样本。因此，
          `max_samples` 应在区间 `(0.0, 1.0]` 内。

        .. versionadded:: 0.22
    # monotonic_cst是一个数组，用于指定每个特征要强制执行的单调性约束。
    # 对于每个特征：
    #   - 1: 要求单调递增
    #   - 0: 没有约束
    #   - -1: 要求单调递减
    # 如果monotonic_cst为None，则不应用任何约束。
    # 单调性约束不支持以下情况：
    #   - 多类别分类（即 `n_classes > 2` 时）
    #   - 多输出分类（即 `n_outputs_ > 1` 时）
    #   - 训练数据中存在缺失值的分类。
    # 这些约束适用于正类别的概率。
    # 在 :ref:`User Guide <monotonic_cst_gbdt>` 中可以阅读更多相关内容。
    # 添加于版本 1.4

Attributes
----------

    # estimator_是用于创建已安装子估算器集合的子估算器模板。
    # 添加于版本 1.2，`base_estimator_` 更名为 `estimator_`。

    # estimators_是已安装子估算器的集合。

    # classes_是类标签数组（单输出问题），或者类标签数组的列表（多输出问题）。

    # n_classes_是类别的数量（单输出问题），或者包含每个输出的类别数的列表（多输出问题）。

    # n_features_in_是在 `fit` 过程中观察到的特征数。
    # 添加于版本 0.24

    # feature_names_in_是在 `fit` 过程中观察到的特征名称数组。
    # 仅在 `X` 具有所有字符串特征名称时定义。
    # 添加于版本 1.0

    # n_outputs_是执行 `fit` 时的输出数量。

    # feature_importances_是基于不纯度的特征重要性。
    # 数值越高，特征越重要。
    # 特征的重要性是通过该特征带来的（归一化的）准则总减少来计算的。
    # 也称为基尼重要性。
    # 警告：基于不纯度的特征重要性对于基数高的特征（具有许多唯一值）可能具有误导性。
    # 可以考虑使用 :func:`sklearn.inspection.permutation_importance` 作为替代方法。

    # oob_score_是使用袋外估计获得的训练数据集得分。
    # 仅在 `oob_score` 为True时存在。
    # _parameter_constraints 是一个字典，包含了随机森林分类器和决策树分类器的参数约束
    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,  # 继承随机森林分类器的参数约束
        **DecisionTreeClassifier._parameter_constraints,  # 继承决策树分类器的参数约束
        "class_weight": [  # 对于 "class_weight" 参数的约束列表
            StrOptions({"balanced_subsample", "balanced"}),  # 可选的字符串选项集合
            dict,  # 字典类型
            list,  # 列表类型
            None,  # 可以为 None
        ],
    }
    # 移除参数约束中的 "splitter" 键
    _parameter_constraints.pop("splitter")
    # 初始化方法，用于构造一个随机森林分类器对象
    def __init__(
        self,
        n_estimators=100,  # 决策树的数量，默认为100
        *,
        criterion="gini",  # 决策树划分标准，默认为基尼系数
        max_depth=None,  # 决策树的最大深度，默认不限制
        min_samples_split=2,  # 内部节点再划分所需最小样本数，默认为2
        min_samples_leaf=1,  # 叶子节点最少样本数，默认为1
        min_weight_fraction_leaf=0.0,  # 叶子节点样本权重的最小加权分数，默认为0.0
        max_features="sqrt",  # 每棵树的最大特征数，默认为平方根
        max_leaf_nodes=None,  # 最大叶子节点数，默认不限制
        min_impurity_decrease=0.0,  # 如果节点分裂会导致杂质减少大于或等于该值，则分裂，默认为0.0
        bootstrap=True,  # 是否使用有放回抽样，默认为True
        oob_score=False,  # 是否使用袋外样本评估模型性能，默认为False
        n_jobs=None,  # 并行运行的作业数，默认为None，表示单线程
        random_state=None,  # 控制随机性的种子，默认为None
        verbose=0,  # 控制拟合和预测过程的详细程度，默认为0，不输出信息
        warm_start=False,  # 设置为True时，使用之前的解作为初始拟合结果，默认为False
        class_weight=None,  # 各个类别的权重，可以是字典、'balanced'或None，默认为None
        ccp_alpha=0.0,  # 用于剪枝的复杂度参数，默认为0.0，即不进行剪枝
        max_samples=None,  # 每个基础估计器的最大样本数，默认为None，表示全部样本
        monotonic_cst=None,  # 用于指定每个特征的单调约束，默认为None
    ):
        # 调用父类的初始化方法，传入决策树分类器作为基础估计器及相关参数
        super().__init__(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,  # 设定基础估计器的数量
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,  # 传递是否使用有放回抽样的参数
            oob_score=oob_score,  # 传递是否使用袋外样本评估模型性能的参数
            n_jobs=n_jobs,  # 传递并行运行的作业数
            random_state=random_state,  # 传递随机数种子
            verbose=verbose,  # 传递详细程度的参数
            warm_start=warm_start,  # 传递是否使用前一个解作为初始拟合结果的参数
            class_weight=class_weight,  # 传递各个类别的权重
            max_samples=max_samples,  # 传递每个基础估计器的最大样本数
        )

        # 设置当前对象的属性，与初始化参数对应
        self.criterion = criterion  # 设置划分标准
        self.max_depth = max_depth  # 设置最大深度
        self.min_samples_split = min_samples_split  # 设置内部节点最小样本数
        self.min_samples_leaf = min_samples_leaf  # 设置叶子节点最小样本数
        self.min_weight_fraction_leaf = min_weight_fraction_leaf  # 设置叶子节点样本权重的最小加权分数
        self.max_features = max_features  # 设置每棵树的最大特征数
        self.max_leaf_nodes = max_leaf_nodes  # 设置最大叶子节点数
        self.min_impurity_decrease = min_impurity_decrease  # 设置杂质减少阈值
        self.monotonic_cst = monotonic_cst  # 设置单调约束
        self.ccp_alpha = ccp_alpha  # 设置剪枝复杂度参数
class RandomForestRegressor(ForestRegressor):
    """
    A random forest regressor.

    A random forest is a meta estimator that fits a number of decision tree
    regressors on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    Trees in the forest use the best split strategy, i.e. equivalent to passing
    `splitter="best"` to the underlying :class:`~sklearn.tree.DecisionTreeRegressor`.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.

    For a comparison between tree-based ensemble models see the example
    :ref:`sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py`.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.
        Training using "absolute_error" is significantly slower
        than when using "squared_error".

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 1.0
           Poisson criterion.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.
    """
    # min_samples_leaf : int or float, default=1
    # 叶子节点所需的最小样本数。任何深度的分裂点只有在左右分支中至少有 `min_samples_leaf` 训练样本时才考虑。
    # 这可能会平滑模型，特别是在回归问题中。
    # 
    # - 如果是 int，则将 `min_samples_leaf` 视为最小样本数。
    # - 如果是 float，则 `min_samples_leaf` 是一个分数，`ceil(min_samples_leaf * n_samples)` 是每个节点的最小样本数。
    # 
    # .. versionchanged:: 0.18
    #    添加了用于分数的 float 值。

    min_weight_fraction_leaf : float, default=0.0
    # 叶子节点所需的加权样本总权重的最小分数。当没有提供 sample_weight 时，样本权重相等。

    max_features : {"sqrt", "log2", None}, int or float, default=1.0
    # 在寻找最佳分裂时考虑的特征数：
    # 
    # - 如果是 int，则在每次分裂时考虑 `max_features` 个特征。
    # - 如果是 float，则 `max_features` 是一个分数，每次分裂时考虑 `max(1, int(max_features * n_features_in_))` 个特征。
    # - 如果是 "sqrt"，则 `max_features=sqrt(n_features)`。
    # - 如果是 "log2"，则 `max_features=log2(n_features)`。
    # - 如果是 None 或 1.0，则 `max_features=n_features`。
    # 
    # .. note::
    #     默认值 1.0 相当于袋装树，通过设置较小的值（如 0.3），可以实现更多的随机性。
    # 
    # .. versionchanged:: 1.1
    #     `max_features` 的默认值从 `"auto"` 改为了 1.0。

    max_leaf_nodes : int, default=None
    # 以最佳优先方式生长具有 `max_leaf_nodes` 个叶节点的树。
    # 最佳节点定义为不纯度的相对减少。
    # 如果为 None，则叶节点数量不受限制。

    min_impurity_decrease : float, default=0.0
    # 如果此分裂导致不纯度减少大于或等于此值，则会分裂节点。
    # 加权不纯度减少方程如下所示：
    # 
    #     N_t / N * (impurity - N_t_R / N_t * right_impurity
    #                         - N_t_L / N_t * left_impurity)
    # 
    # 其中，`N` 是样本总数，`N_t` 是当前节点的样本数，`N_t_L` 是左子节点中的样本数，`N_t_R` 是右子节点中的样本数。
    # 如果传递了 `sample_weight`，则所有这些数量都是加权总和。
    # 
    # .. versionadded:: 0.19
    bootstrap : bool, default=True
        # 是否使用自助采样构建树。如果为 False，则使用整个数据集构建每棵树。

    oob_score : bool or callable, default=False
        # 是否使用袋外样本来估计泛化分数。
        # 默认情况下，使用 sklearn.metrics.r2_score。
        # 提供一个具有签名 `metric(y_true, y_pred)` 的可调用对象以使用自定义度量标准。
        # 仅在 `bootstrap=True` 时可用。

    n_jobs : int, default=None
        # 并行运行的作业数量。
        # `fit`，`predict`，`decision_path` 和 `apply` 方法都会并行化处理树。
        # `None` 表示 1，除非在 `joblib.parallel_backend` 上下文中。
        # `-1` 表示使用所有处理器。详见 "术语表" 中的 "n_jobs" 一词。

    random_state : int, RandomState instance or None, default=None
        # 控制在构建树时用于自助采样的样本的随机性（如果 `bootstrap=True`），
        # 以及在查找每个节点的最佳分割时用于考虑的特征的采样的随机性
        # （如果 `max_features < n_features`）。
        # 详见 "术语表" 中的 "random_state" 一词。

    verbose : int, default=0
        # 控制拟合和预测时的详细程度。

    warm_start : bool, default=False
        # 当设置为 `True` 时，重用上一次调用 `fit` 的解决方案，并向集成中添加更多的估计器，
        # 否则，只拟合一个全新的森林。详见 "术语表" 中的 "warm_start" 和 "tree_ensemble_warm_start" 一词。

    ccp_alpha : non-negative float, default=0.0
        # 用于最小成本复杂度剪枝的复杂性参数。将选择小于 `ccp_alpha` 的最大成本复杂度的子树。
        # 默认情况下，不执行剪枝。详见 "最小成本复杂度剪枝" 一文。

        .. versionadded:: 0.22

    max_samples : int or float, default=None
        # 如果 `bootstrap=True`，则从 X 中抽取的样本数，用于训练每个基估计器。

        - 如果为 None（默认），则抽取 `X.shape[0]` 个样本。
        - 如果为整数，则抽取 `max_samples` 个样本。
        - 如果为浮点数，则抽取 `max(round(n_samples * max_samples), 1)` 个样本。
          因此，`max_samples` 应在区间 `(0.0, 1.0]` 内。

        .. versionadded:: 0.22
    monotonic_cst : array-like of int of shape (n_features), default=None
        # 定义一个数组，表示特征的单调性约束，默认为None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonically increasing
          - 0: no constraint
          - -1: monotonically decreasing

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multioutput regressions (i.e. when `n_outputs_ > 1`),
          - regressions trained on data with missing values.

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

        .. versionadded:: 1.4

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.DecisionTreeRegressor`
        # 子估计器模板，用于创建一组拟合的子估计器集合
        The child estimator template used to create the collection of fitted
        sub-estimators.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of DecisionTreeRegressor
        # 拟合的子估计器集合
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        # 基于不纯度的特征重要性
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        # 在拟合过程中观察到的特征数目
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合过程中观察到的特征名称
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        # 执行`fit`时的输出数目
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        # 使用包外估计得到的训练数据集得分
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        # 使用包外估计在训练集上计算得到的预测值
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    estimators_samples_ : list of arrays
        # 每个基础估计器的抽样子集（即袋外样本）
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

        .. versionadded:: 1.4

    See Also
    --------
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    sklearn.ensemble.ExtraTreesRegressor : Ensemble of extremely randomized
        tree regressors.
    sklearn.ensemble.HistGradientBoostingRegressor : A Histogram-based Gradient
        Boosting Regression Tree, very fast for big datasets (n_samples >=
        10_000).

    Notes
    -----
    """
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    """
    
    _parameter_constraints: dict = {
        **ForestRegressor._parameter_constraints,
        **DecisionTreeRegressor._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")
    
    class RandomForestRegressor:
        """
        The features are always randomly permuted at each split. Therefore,
        the best found split may vary, even with the same training data,
        ``max_features=n_features`` and ``bootstrap=False``, if the improvement
        of the criterion is identical for several splits enumerated during the
        search of the best split. To obtain a deterministic behaviour during
        fitting, ``random_state`` has to be fixed.
    
        The default value ``max_features=1.0`` uses ``n_features``
        rather than ``n_features / 3``. The latter was originally suggested in
        [1], whereas the former was more recently justified empirically in [2].
    
        References
        ----------
        .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    
        .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
               trees", Machine Learning, 63(1), 3-42, 2006.
    
        Examples
        --------
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_features=4, n_informative=2,
        ...                        random_state=0, shuffle=False)
        >>> regr = RandomForestRegressor(max_depth=2, random_state=0)
        >>> regr.fit(X, y)
        RandomForestRegressor(...)
        >>> print(regr.predict([[0, 0, 0, 0]]))
        [-8.32987858]
        """
    
        def __init__(
            self,
            n_estimators=100,
            *,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
    ):
        """
        Initialize a RandomForestRegressor instance.
    
        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.
    
        criterion : {"squared_error"}, default="squared_error"
            The function to measure the quality of a split.
    
        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
    
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.
    
        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node.
    
        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            input samples) required to be at a leaf node.
    
        max_features : float, default=1.0
            The number of features to consider when looking for the best split:
            - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
    
        max_leaf_nodes : int, default=None
            Grow trees with `max_leaf_nodes` in best-first fashion.
    
        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
    
        bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees.
    
        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate the R^2 on unseen data.
    
        n_jobs : int, default=None
            The number of jobs to run in parallel. -1 means using all processors.
    
        random_state : int, RandomState instance, default=None
            Controls both the randomness of the bootstrapping of the samples used
            when building trees (if `bootstrap=True`) and the sampling of the
            features to consider when looking for the best split at each node.
    
        verbose : int, default=0
            Controls the verbosity when fitting and predicting.
    
        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit and
            add more estimators to the ensemble, otherwise, just fit a whole new
            forest.
    
        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            `ccp_alpha` will be chosen.
    
        max_samples : int or float, default=None
            If bootstrap is True, the number of samples to draw from X to train
            each base estimator. If None (default), then draw `X.shape[0]`
            samples.
    
        monotonic_cst : array-like of shape (n_features,), default=None
            The constants in the monotonic constraints. 1 means increasing,
            -1 means decreasing, and 0 means no constraint.
    
        """
        super().__init__(
            estimator=DecisionTreeRegressor(),  # 使用决策树回归器作为基础估计器
            n_estimators=n_estimators,  # 设定集成器中决策树的数量
            estimator_params=(
                "criterion",  # 决策树的分裂标准
                "max_depth",  # 决策树的最大深度
                "min_samples_split",  # 分裂内部节点所需的最小样本数
                "min_samples_leaf",  # 叶节点所需的最小样本数
                "min_weight_fraction_leaf",  # 叶节点的最小加权分数
                "max_features",  # 在每棵树分裂时考虑的最大特征数
                "max_leaf_nodes",  # 每棵树允许的最大叶节点数
                "min_impurity_decrease",  # 分裂节点的最小杂质减少量
                "random_state",  # 控制随机性的种子
                "ccp_alpha",  # 最小代价复杂度剪枝的复杂度参数
                "monotonic_cst",  # 强制特征的单调性的约束
            ),
            bootstrap=bootstrap,  # 是否使用自助法进行抽样
            oob_score=oob_score,  # 是否计算袋外估计
            n_jobs=n_jobs,  # 并行处理的作业数
            random_state=random_state,  # 控制估计器的随机性的种子
            verbose=verbose,  # 控制详细程度的整数值
            warm_start=warm_start,  # 是否使用前一次调用的解决方案作为初始值
            max_samples=max_samples,  # 每棵树的最大样本数
        )

        self.criterion = criterion  # 将传入的决策树分裂标准存储在实例中
        self.max_depth = max_depth  # 将传入的决策树最大深度存储在实例中
        self.min_samples_split = min_samples_split  # 将传入的决策树分裂内部节点所需的最小样本数存储在实例中
        self.min_samples_leaf = min_samples_leaf  # 将传入的决策树叶节点所需的最小样本数存储在实例中
        self.min_weight_fraction_leaf = min_weight_fraction_leaf  # 将传入的决策树叶节点的最小加权分数存储在实例中
        self.max_features = max_features  # 将传入的决策树每棵树分裂时考虑的最大特征数存储在实例中
        self.max_leaf_nodes = max_leaf_nodes  # 将传入的决策树每棵树允许的最大叶节点数存储在实例中
        self.min_impurity_decrease = min_impurity_decrease  # 将传入的决策树分裂节点的最小杂质减少量存储在实例中
        self.ccp_alpha = ccp_alpha  # 将传入的决策树最小代价复杂度剪枝的复杂度参数存储在实例中
        self.monotonic_cst = monotonic_cst  # 将传入的决策树强制特征单调性的约束存储在实例中
class ExtraTreesClassifier(ForestClassifier):
    """
    An extra-trees classifier.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.
        Note: This parameter is tree-specific.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    """

    # 初始化方法，设置各种参数
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, **kwargs):
        # 调用父类的初始化方法，设置通用的参数
        super().__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "random_state"),
            **kwargs)

        # 设置额外树分类器的特定参数
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
    # max_features: 控制每次分裂时考虑的特征数量限制，可以是整数、浮点数或特定字符串
    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:
    
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
    
        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to `"sqrt"`.
    
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    
    # max_leaf_nodes: 以最佳优先方式生长树，限制最大叶子节点数
    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    
    # min_impurity_decrease: 控制节点分裂的最小不纯度减少值
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
        The weighted impurity decrease equation is the following::
    
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
    
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
    
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
    
        .. versionadded:: 0.19
    
    # bootstrap: 是否使用自助法样本构建树
    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.
    
    # oob_score: 是否使用袋外样本来估计泛化分数
    oob_score : bool or callable, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        By default, :func:`~sklearn.metrics.accuracy_score` is used.
        Provide a callable with signature `metric(y_true, y_pred)` to use a
        custom metric. Only available if `bootstrap=True`.
    
    # n_jobs: 并行运行的作业数量
    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.
    # 控制随机性的种子，影响三个方面：
    # - 当构建树时，用于引导样本的抽样（如果 `bootstrap=True`）
    # - 在每个节点寻找最佳分割时，考虑的特征的抽样（如果 `max_features < n_features`）
    # - 每个 `max_features` 中的分割的抽样
    random_state : int, RandomState instance or None, default=None

    # 控制拟合和预测时的详细程度
    verbose : int, default=0

    # 当设置为 True 时，重用上次调用 `fit` 的解决方案，并添加更多的估计器到集成中；
    # 否则，只是拟合一个全新的森林。参见术语表中的 `warm_start` 和 `tree_ensemble_warm_start` 获取详细信息。
    warm_start : bool, default=False

    # 类别权重，支持以下几种形式：
    # - "balanced": 使用 `y` 的值自动调整权重，与输入数据中类别频率成反比计算为 `n_samples / (n_classes * np.bincount(y))`
    # - "balanced_subsample": 与 "balanced" 相同，但是基于每棵树生长的自举样本计算权重
    # - 字典或字典列表：形如 `{class_label: weight}`，适用于多输出问题时，可按 `y` 的列顺序提供多个字典
    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, default=None

    # 最小成本复杂度剪枝所用的复杂度参数，选择小于 `ccp_alpha` 的最大成本复杂度子树
    # 默认情况下，不执行剪枝。参见 `minimal_cost_complexity_pruning` 获取详细信息。
    ccp_alpha : non-negative float, default=0.0

    # 如果 `bootstrap` 为 True，则从 `X` 中抽取的样本数目用于训练每个基础估计器
    # - 如果为 None（默认），则抽取 `X.shape[0]` 个样本
    # - 如果为整数，则抽取 `max_samples` 个样本
    # - 如果为浮点数，则抽取 `max_samples * X.shape[0]` 个样本，因此 `max_samples` 应在 `(0.0, 1.0]` 区间内
    max_samples : int or float, default=None
    monotonic_cst : array-like of int of shape (n_features), default=None
        # 指示每个特征上要强制执行的单调性约束。
        #   - 1: 单调递增
        #   - 0: 无约束
        #   - -1: 单调递减

        # 如果 monotonic_cst 为 None，则不应用任何约束。

        # 不支持以下情况的单调性约束：
        #   - 多类别分类（即 `n_classes > 2`），
        #   - 多输出分类（即 `n_outputs_ > 1`），
        #   - 训练数据包含缺失值。

        # 约束适用于正类的概率。

        # 在 :ref:`User Guide <monotonic_cst_gbdt>` 中可以进一步了解。

        .. versionadded:: 1.4

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeClassifier`
        # 用于创建拟合子估计器集合的子估计器模板。

        .. versionadded:: 1.2
           `base_estimator_` 被重命名为 `estimator_`.

    estimators_ : list of DecisionTreeClassifier
        # 拟合的子估计器集合。

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        # 类别标签（单输出问题），或类别标签数组的列表（多输出问题）。

    n_classes_ : int or list
        # 类别数量（单输出问题），或包含每个输出类别数量的列表（多输出问题）。

    feature_importances_ : ndarray of shape (n_features,)
        # 基于不纯度的特征重要性。
        # 数值越高，特征越重要。
        # 特征的重要性计算为该特征带来的（标准化的）准则总减少量。
        # 也称为基尼重要性。

        # 警告：基于不纯度的特征重要性对高基数特征（具有许多唯一值）可能具有误导性。
        # 可以查看 :func:`sklearn.inspection.permutation_importance` 作为替代方法。

    n_features_in_ : int
        # 在 `fit` 过程中观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 过程中观察到的特征名称。
        # 仅当 `X` 的特征名称全为字符串时定义。

        .. versionadded:: 1.0

    n_outputs_ : int
        # 执行 `fit` 时的输出数量。

    oob_score_ : float
        # 使用袋外估计获得的训练数据集得分。
        # 只有在 `oob_score` 设置为 True 时存在。
    # `oob_decision_function_` : ndarray of shape (n_samples, n_classes) or \
    #         (n_samples, n_classes, n_outputs)
    #     训练集上使用袋外估计计算的决策函数。如果 `n_estimators` 较小，可能会出现在引导过程中某个数据点从未被排除的情况。
    #     在这种情况下，`oob_decision_function_` 可能包含 NaN。仅当 `oob_score` 为 True 时才存在此属性。

    # `estimators_samples_` : list of arrays
    #     每个基本估计器的抽样子集（即袋内样本）。每个子集由选择的索引数组定义。

    #     .. versionadded:: 1.4

    # See Also
    # --------
    # ExtraTreesRegressor : 使用随机分割的极端随机树回归器。
    # RandomForestClassifier : 使用最优分割的随机森林分类器。
    # RandomForestRegressor : 使用最优分割的树集成回归器。

    # Notes
    # -----
    # 控制树的大小的参数的默认值（例如 `max_depth`、`min_samples_leaf` 等）会导致完全生长且未修剪的树，
    # 在某些数据集上可能非常大。为了减少内存消耗，应通过设置这些参数值来控制树的复杂性和大小。

    # References
    # ----------
    # .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
    #        trees", Machine Learning, 63(1), 3-42, 2006.

    # Examples
    # --------
    # >>> from sklearn.ensemble import ExtraTreesClassifier
    # >>> from sklearn.datasets import make_classification
    # >>> X, y = make_classification(n_features=4, random_state=0)
    # >>> clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    # >>> clf.fit(X, y)
    # ExtraTreesClassifier(random_state=0)
    # >>> clf.predict([[0, 0, 0, 0]])
    # array([1])

    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **DecisionTreeClassifier._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }
    # "splitter" 不再适用于 ExtraTreesClassifier，因此从参数约束中移除
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        """
        初始化 ExtraTreesClassifier 实例。

        Parameters
        ----------
        n_estimators : int, default=100
            森林中树的数量。

        criterion : {"gini", "entropy"}, default="gini"
            用于评估分裂质量的标准。

        max_depth : int, default=None
            树的最大深度。如果为 None，则节点会扩展直到所有叶子都是纯的，
            或者直到所有叶子包含的样本少于 `min_samples_split`。

        min_samples_split : int or float, default=2
            拆分内部节点所需的最小样本数。
            如果为 float，则它表示作为百分比的 `min_samples_split` 的分数。
        
        （以下省略余下的参数说明，具体细节可参阅 sklearn 文档）

        """
        super().__init__(
            estimator=ExtraTreeClassifier(),  # 调用父类的初始化方法，使用ExtraTreeClassifier作为基础评估器
            n_estimators=n_estimators,  # 设定集成模型中树的数量
            estimator_params=(
                "criterion",  # 决策树的分裂标准
                "max_depth",  # 决策树的最大深度
                "min_samples_split",  # 内部节点分裂所需的最小样本数
                "min_samples_leaf",  # 叶子节点最少样本数
                "min_weight_fraction_leaf",  # 叶子节点样本权重的最小加权分数
                "max_features",  # 每棵树分裂时考虑的最大特征数
                "max_leaf_nodes",  # 最大叶子节点数
                "min_impurity_decrease",  # 分裂节点时最小不纯度下降量
                "random_state",  # 随机数种子
                "ccp_alpha",  # 用于剪枝的复杂性参数
                "monotonic_cst",  # 用于强制单调性的约束
            ),
            bootstrap=bootstrap,  # 是否进行自助采样
            oob_score=oob_score,  # 是否计算袋外评分
            n_jobs=n_jobs,  # 并行运行时使用的作业数
            random_state=random_state,  # 全局随机数种子
            verbose=verbose,  # 控制详细程度的输出
            warm_start=warm_start,  # 是否重用前一次训练的解决方案以适应增量数据
            class_weight=class_weight,  # 类别权重
            max_samples=max_samples,  # 每棵树的最大样本数
        )

        self.criterion = criterion  # 将传入的决策树分裂标准赋给实例变量
        self.max_depth = max_depth  # 将传入的最大深度赋给实例变量
        self.min_samples_split = min_samples_split  # 将传入的内部节点最小分裂样本数赋给实例变量
        self.min_samples_leaf = min_samples_leaf  # 将传入的叶子节点最小样本数赋给实例变量
        self.min_weight_fraction_leaf = min_weight_fraction_leaf  # 将传入的叶子节点样本权重最小加权分数赋给实例变量
        self.max_features = max_features  # 将传入的每棵树分裂时考虑的最大特征数赋给实例变量
        self.max_leaf_nodes = max_leaf_nodes  # 将传入的最大叶子节点数赋给实例变量
        self.min_impurity_decrease = min_impurity_decrease  # 将传入的分裂节点时最小不纯度下降量赋给实例变量
        self.ccp_alpha = ccp_alpha  # 将传入的用于剪枝的复杂性参数赋给实例变量
        self.monotonic_cst = monotonic_cst  # 将传入的用于强制单调性的约束赋给实例变量
class ExtraTreesRegressor(ForestRegressor):
    """
    An extra-trees regressor.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.
        Training using "absolute_error" is significantly slower
        than when using "squared_error".

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.
    """
    # 最小叶节点权重的分数阈值，默认为0.0
    min_weight_fraction_leaf : float, default=0.0
        # 叶节点所需的样本权重总和的最小加权分数比例（所有输入样本的权重）
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    # 最大特征数，用于寻找最佳分割的时候，默认为1.0
    max_features : {"sqrt", "log2", None}, int or float, default=1.0
        # 在寻找最佳分割时考虑的特征数目：

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None or 1.0, then `max_features=n_features`.

        .. note::
            默认值1.0相当于装袋树，通过设置较小的值（例如0.3），可以增加更多的随机性。

        .. versionchanged:: 1.1
            默认的 `max_features` 从 `"auto"` 改为 1.0。

        Note: 即使需要实际检查超过 ``max_features`` 特征，寻找分割也不会在未找到有效分割的情况下停止。

    # 最大叶节点数，默认为None
    max_leaf_nodes : int, default=None
        # 以最佳优先方式扩展具有 ``max_leaf_nodes`` 的树。
        最佳节点定义为杂质的相对减少。
        如果为None，则叶节点数目不限。

    # 最小杂质减少阈值，默认为0.0
    min_impurity_decrease : float, default=0.0
        # 如果此分割导致杂质减少大于或等于此值，则节点将被分割。

        加权杂质减少方程式如下::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        其中 ``N`` 是样本的总数，``N_t`` 是当前节点的样本数，``N_t_L`` 是左子节点中的样本数，``N_t_R`` 是右子节点中的样本数。

        如果传递了 ``sample_weight``，则所有这些都是加权和。

        .. versionadded:: 0.19

    # 是否在构建树时使用自助采样，默认为False
    bootstrap : bool, default=False
        # 是否在构建树时使用自助采样。如果为False，则使用整个数据集来构建每棵树。

    # 是否使用袋外样本来估计泛化分数，默认为False
    oob_score : bool or callable, default=False
        # 是否使用袋外样本来估计泛化分数。
        默认情况下使用 :func:`~sklearn.metrics.r2_score`。
        提供一个带有 `metric(y_true, y_pred)` 签名的可调用函数来使用自定义度量标准。
        仅在 `bootstrap=True` 时可用。
    n_jobs : int, default=None
        # 并行运行的作业数量。:meth:`fit`, :meth:`predict`,
        # :meth:`decision_path` 和 :meth:`apply` 方法都可以在多棵树之间并行化执行。
        # 如果未指定，为1，除非在 :obj:`joblib.parallel_backend` 上下文中，此时为 `None`。
        # `-1` 表示使用所有处理器。详细信息请参阅 :term:`术语表 <n_jobs>`。

    random_state : int, RandomState instance or None, default=None
        # 控制三个随机源：

        # - 构建树时用于自举采样的样本
          (如果 ``bootstrap=True``)
        # - 在每个节点查找最佳分割时考虑的特征样本的抽样
          (如果 ``max_features < n_features``)
        # - 为 `max_features` 的每个特征绘制分割的抽样

        # 详细信息请参阅 :term:`术语表 <random_state>`。

    verbose : int, default=0
        # 控制 `fit` 和 `predict` 时的冗长程度。

    warm_start : bool, default=False
        # 当设置为 ``True`` 时，重用先前调用 `fit` 的解决方案，并添加更多估计器到集成中，
        # 否则，只需适合一个全新的森林。详细信息请参阅 :term:`术语表 <warm_start>` 和
        # :ref:`tree_ensemble_warm_start`。

    ccp_alpha : non-negative float, default=0.0
        # 用于最小成本复杂性修剪的复杂度参数。将选择小于 ``ccp_alpha`` 的最大成本复杂性的子树。
        # 默认情况下，不执行修剪。详细信息请参阅 :ref:`minimal_cost_complexity_pruning`。

        # .. versionadded:: 0.22

    max_samples : int or float, default=None
        # 如果 `bootstrap` 为 True，则从 X 中抽取的样本数量用于训练每个基本估计器。

        # - 如果为 None（默认值），则抽取 `X.shape[0]` 个样本。
        # - 如果为 int，则抽取 `max_samples` 个样本。
        # - 如果为 float，则抽取 `max_samples * X.shape[0]` 个样本。因此，
          `max_samples` 应在区间 `(0.0, 1.0]` 内。

        # .. versionadded:: 0.22

    monotonic_cst : array-like of int of shape (n_features), default=None
        # 指示要强制施加在每个特征上的单调性约束。
          - 1: 单调递增
          - 0: 无约束
          - -1: 单调递减

        # 如果 `monotonic_cst` 为 None，则不应用任何约束。

        # 单调性约束不支持：
          - 多输出回归（即 `n_outputs_ > 1`），
          - 在带有缺失值的数据上训练的回归。

        # 详细信息请阅读 :ref:`User Guide <monotonic_cst_gbdt>`。

        # .. versionadded:: 1.4

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor`
        # 用于创建已安装子估计器集合的子估计器模板。

        # .. versionadded:: 1.2
           `base_estimator_` 已重命名为 `estimator_`。
    estimators_ : list of DecisionTreeRegressor
        # 存储训练后的决策树回归器对象的列表

    feature_importances_ : ndarray of shape (n_features,)
        # 基于不纯度的特征重要性评估结果
        # 数组形状为 (n_features,)，每个元素表示对应特征的重要性
        # 数值越高，特征越重要，特征重要性通过该特征对准则总减少量的归一化计算得出，也称为基尼重要性

        # 警告：基于不纯度的特征重要性对于基数较高的特征（具有许多唯一值）可能具有误导性。参见
        # :func:`sklearn.inspection.permutation_importance` 作为一种替代方法。

    n_features_in_ : int
        # 在拟合过程中观察到的特征数量
        # .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合过程中观察到的特征名称
        # 仅在 `X` 具有全部为字符串的特征名称时定义
        # .. versionadded:: 1.0

    n_outputs_ : int
        # 输出的数量

    oob_score_ : float
        # 使用袋外估计计算的训练数据集得分
        # 仅在 `oob_score` 为 True 时存在

    oob_prediction_ : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        # 使用袋外估计在训练集上计算的预测值
        # 仅在 `oob_score` 为 True 时存在

    estimators_samples_ : list of arrays
        # 每个基本估计器的抽取样本子集（即袋内样本）
        # 每个子集由选定的索引数组定义
        # .. versionadded:: 1.4

    See Also
    --------
    ExtraTreesClassifier : 具有随机分割的极端随机树分类器。
    RandomForestClassifier : 使用最优分割的随机森林分类器。
    RandomForestRegressor : 使用最优分割的树的集成回归器。

    Notes
    -----
    默认参数值（例如 `max_depth`, `min_samples_leaf` 等）控制树的大小，
    导致完全生长且未修剪的树可能在某些数据集上非常大。为了减少内存消耗，
    应通过设置这些参数值来控制树的复杂性和大小。

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(
    ...    X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.2727...
    # 继承 ForestRegressor 和 DecisionTreeRegressor 的参数约束字典，并移除 "splitter" 参数约束
    _parameter_constraints: dict = {
        **ForestRegressor._parameter_constraints,
        **DecisionTreeRegressor._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")

    # 初始化方法，定义了 ExtraTreesRegressor 的超参数和设置
    def __init__(
        self,
        n_estimators=100,  # 弱学习器数量，默认为 100
        *,
        criterion="squared_error",  # 划分标准，默认为平方误差
        max_depth=None,  # 树的最大深度，默认不限制
        min_samples_split=2,  # 划分内部节点所需的最小样本数，默认为 2
        min_samples_leaf=1,  # 叶子节点最少样本数，默认为 1
        min_weight_fraction_leaf=0.0,  # 叶子节点的最小加权分数，默认为 0
        max_features=1.0,  # 每个决策树最大的特征数，默认为全部特征
        max_leaf_nodes=None,  # 叶子节点的最大数量，默认不限制
        min_impurity_decrease=0.0,  # 划分节点的最小杂质减少量，默认为 0
        bootstrap=False,  # 是否进行自助采样，默认不采用
        oob_score=False,  # 是否使用 out-of-bag 数据计算决策树的评分，默认不使用
        n_jobs=None,  # 并行运行的作业数，默认不并行
        random_state=None,  # 随机种子，控制随机性，默认无特定种子
        verbose=0,  # 控制详细程度的标志，默认不输出详细信息
        warm_start=False,  # 是否热启动，默认不启用
        ccp_alpha=0.0,  # 用于复杂性剪枝的成本复杂度参数，默认为 0
        max_samples=None,  # 用于 bootstrap 采样的样本数，默认不限制
        monotonic_cst=None,  # 用于约束特征的单调性，默认无约束
    ):
        # 调用父类的初始化方法，设置参数到 ExtraTreeRegressor 模型
        super().__init__(
            estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        # 设置对象的各个超参数值
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
# 定义一个新的类 RandomTreesEmbedding，它同时继承了 TransformerMixin 和 BaseForest 类
class RandomTreesEmbedding(TransformerMixin, BaseForest):
    """
    An ensemble of totally random trees.

    An unsupervised transformation of a dataset to a high-dimensional
    sparse representation. A datapoint is coded according to which leaf of
    each tree it is sorted into. Using a one-hot encoding of the leaves,
    this leads to a binary coding with as many ones as there are trees in
    the forest.

    The dimensionality of the resulting representation is
    ``n_out <= n_estimators * max_leaf_nodes``. If ``max_leaf_nodes == None``,
    the number of leaf nodes is at most ``n_estimators * 2 ** max_depth``.

    Read more in the :ref:`User Guide <random_trees_embedding>`.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    max_depth : int, default=5
        The maximum depth of each tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` is the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` is the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    """
    # 初始化方法，设置类的初始属性
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_leaf_nodes=None):
        # 调用父类的初始化方法
        super().__init__()
        # 设置属性 n_estimators 为传入的 n_estimators 参数值
        self.n_estimators = n_estimators
        # 设置属性 max_depth 为传入的 max_depth 参数值
        self.max_depth = max_depth
        # 设置属性 min_samples_split 为传入的 min_samples_split 参数值
        self.min_samples_split = min_samples_split
        # 设置属性 min_samples_leaf 为传入的 min_samples_leaf 参数值
        self.min_samples_leaf = min_samples_leaf
        # 设置属性 min_weight_fraction_leaf 为传入的 min_weight_fraction_leaf 参数值
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        # 设置属性 max_leaf_nodes 为传入的 max_leaf_nodes 参数值
        self.max_leaf_nodes = max_leaf_nodes
    # min_impurity_decrease: 节点分裂的条件之一，如果分裂导致的不纯度减少大于等于此值，则进行分裂。
    min_impurity_decrease : float, default=0.0
        节点将会被分裂，如果这次分裂导致的不纯度减少大于或等于此值。

        加权不纯度减少的计算公式如下::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        其中 ``N`` 是样本总数，``N_t`` 是当前节点的样本数，``N_t_L`` 是左子节点的样本数，
        ``N_t_R`` 是右子节点的样本数。

        如果传递了 ``sample_weight``，则 ``N``、``N_t``、``N_t_R`` 和 ``N_t_L`` 都是加权总和。

        .. versionadded:: 0.19

    # sparse_output: 是否返回稀疏的 CSR 矩阵，默认为 True，即返回与密集管道操作兼容的密集数组。
    sparse_output : bool, default=True
        是否返回稀疏的 CSR 矩阵，作为默认行为，或者返回与密集管道操作兼容的密集数组。

    # n_jobs: 并行运行的作业数。fit、transform、decision_path 和 apply 方法在所有树上都是并行执行的。
    n_jobs : int, default=None
        并行运行的作业数。fit、transform、decision_path 和 apply 方法在所有树上都是并行执行的。
        ``None`` 表示使用 1 个作业，除非在 :obj:`joblib.parallel_backend` 上下文中。``-1`` 表示使用所有处理器。
        更多详情见 :term:`术语表 <n_jobs>`。

    # random_state: 控制用于拟合树和节点处每个特征的分割的随机 `y` 的生成。
    random_state : int, RandomState instance or None, default=None
        控制用于拟合树和节点处每个特征的分割的随机 `y` 的生成。
        详见 :term:`术语表 <random_state>`。

    # verbose: 控制拟合和预测时的详细程度。
    verbose : int, default=0
        控制拟合和预测时的详细程度。

    # warm_start: 当设置为 True 时，重用上一次调用 fit 的解决方案，并添加更多的估计器到集成中。
    warm_start : bool, default=False
        当设置为 True 时，重用上一次调用 fit 的解决方案，并添加更多的估计器到集成中。
        详见 :term:`术语表 <warm_start>` 和 :ref:`tree_ensemble_warm_start`。

    Attributes
    ----------

    # estimator_: 用于创建已拟合子估计器集合的子估计器模板。
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor` instance
        用于创建已拟合子估计器集合的子估计器模板。

        .. versionadded:: 1.2
           `base_estimator_` 更名为 `estimator_`。

    # estimators_: 拟合的子估计器集合。
    estimators_ : list of :class:`~sklearn.tree.ExtraTreeRegressor` instances
        拟合的子估计器集合。

    # feature_importances_: 特征重要性，值越高表示特征越重要。
    feature_importances_ : ndarray of shape (n_features,)
        特征重要性（值越高，特征越重要）。

    # n_features_in_: 在拟合过程中观察到的特征数量。
    n_features_in_ : int
        在 :term:`fit` 过程中观察到的特征数量。

        .. versionadded:: 0.24

    # feature_names_in_: 在 `fit` 过程中观察到的特征名称数组。
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 `fit` 过程中观察到的特征名称数组。仅当 `X` 的特征名称全为字符串时定义。

        .. versionadded:: 1.0

    # n_outputs_: 执行 `fit` 时的输出数量。
    n_outputs_ : int
        执行 `fit` 时的输出数量。
    # OneHotEncoder 实例，用于创建稀疏嵌入
    one_hot_encoder_ : OneHotEncoder instance

    # 每个基础估计器抽取的样本子集（即袋内样本），每个子集由选定索引的数组定义
    estimators_samples_ : list of arrays

    # 版本新增：1.4
    .. versionadded:: 1.4

    See Also
    --------
    # 额外树分类器
    ExtraTreesClassifier : An extra-trees classifier.
    # 额外树回归器
    ExtraTreesRegressor : An extra-trees regressor.
    # 随机森林分类器
    RandomForestClassifier : A random forest classifier.
    # 随机森林回归器
    RandomForestRegressor : A random forest regressor.
    # 极端随机树分类器
    sklearn.tree.ExtraTreeClassifier: An extremely randomized
        tree classifier.
    # 极端随机树回归器
    sklearn.tree.ExtraTreeRegressor : An extremely randomized
        tree regressor.

    References
    ----------
    # 参考文献
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    .. [2] Moosmann, F. and Triggs, B. and Jurie, F.  "Fast discriminative
           visual codebooks using randomized clustering forests"
           NIPS 2007

    Examples
    --------
    # 示例
    >>> from sklearn.ensemble import RandomTreesEmbedding
    >>> X = [[0,0], [1,0], [0,1], [-1,0], [0,-1]]
    >>> random_trees = RandomTreesEmbedding(
    ...    n_estimators=5, random_state=0, max_depth=1).fit(X)
    >>> X_sparse_embedding = random_trees.transform(X)
    >>> X_sparse_embedding.toarray()
    array([[0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
           [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.]])

    """

    # 参数约束字典，指定了决策树随机森林的各种参数限制
    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
        **BaseDecisionTree._parameter_constraints,  # 继承自 BaseDecisionTree 的参数约束
        "sparse_output": ["boolean"],
    }

    # 移除一些特定参数的约束
    for param in ("max_features", "ccp_alpha", "splitter", "monotonic_cst"):
        _parameter_constraints.pop(param)

    # 损失函数为平方误差
    criterion = "squared_error"
    
    # 最大特征数为 1
    max_features = 1

    # 初始化方法
    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        sparse_output=True,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ):
        # 调用父类初始化方法，设置 ExtraTreeRegressor 为基本估计器，设置其他参数
        super().__init__(
            estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
            ),
            bootstrap=False,  # 禁用 bootstrap 方法
            oob_score=False,  # 禁用 Out-of-Bag 分数
            n_jobs=n_jobs,  # 并行工作的任务数
            random_state=random_state,  # 随机数种子
            verbose=verbose,  # 控制详细程度的输出
            warm_start=warm_start,  # 是否重用上一次调用的解
            max_samples=None,  # 最大样本数设为空
        )

        # 设置决策树的最大深度
        self.max_depth = max_depth
        # 设置内部节点分裂所需的最小样本数
        self.min_samples_split = min_samples_split
        # 设置叶子节点所需的最小样本数
        self.min_samples_leaf = min_samples_leaf
        # 设置叶子节点最小权重
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        # 设置决策树最大叶子节点数
        self.max_leaf_nodes = max_leaf_nodes
        # 设置最小不纯度减少量
        self.min_impurity_decrease = min_impurity_decrease
        # 设置是否稀疏输出
        self.sparse_output = sparse_output

    def _set_oob_score_and_attributes(self, X, y, scoring_function=None):
        # 抛出异常，因为树嵌入不支持 Out-of-Bag 分数
        raise NotImplementedError("OOB score not supported by tree embedding")

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 在 fit_transform 中验证参数
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Fit estimator and transform dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data used to build forests. Use ``dtype=np.float32`` for
            maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_out)
            Transformed dataset.
        """

        # Initialize a random state using the provided or default random state
        rnd = check_random_state(self.random_state)
        
        # Generate random uniform values for y based on the number of samples in X
        y = rnd.uniform(size=_num_samples(X))
        
        # Call the fit method of the superclass with the provided arguments
        super().fit(X, y, sample_weight=sample_weight)

        # Initialize a OneHotEncoder with specified sparse output configuration
        self.one_hot_encoder_ = OneHotEncoder(sparse_output=self.sparse_output)
        
        # Apply the encoder to the transformed dataset X and store the output
        output = self.one_hot_encoder_.fit_transform(self.apply(X))
        
        # Determine the number of output features from the shape of the transformed output
        self._n_features_out = output.shape[1]
        
        # Return the transformed dataset
        return output


    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Only used to validate feature names with the names seen in :meth:`fit`.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names, in the format of
            `randomtreesembedding_{tree}_{leaf}`, where `tree` is the tree used
            to generate the leaf and `leaf` is the index of a leaf node
            in that tree. Note that the node indexing scheme is used to
            index both nodes with children (split nodes) and leaf nodes.
            Only the latter can be present as output features.
            As a consequence, there are missing indices in the output
            feature names.
        """

        # Ensure that the estimator has been fitted and _n_features_out is available
        check_is_fitted(self, "_n_features_out")
        
        # Validate input_features if provided, but do not generate new feature names
        _check_feature_names_in(
            self, input_features=input_features, generate_names=False
        )

        # Generate feature names based on the structure of the embedded trees and leaves
        feature_names = [
            f"randomtreesembedding_{tree}_{leaf}"
            for tree in range(self.n_estimators)
            for leaf in self.one_hot_encoder_.categories_[tree]
        ]
        
        # Convert the list of feature names into a numpy array for output
        return np.asarray(feature_names, dtype=object)
    # 定义一个方法 `transform`，用于对数据集进行转换操作
    def transform(self, X):
        """
        Transform dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csr_matrix`` for maximum efficiency.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_out)
            Transformed dataset.
        """
        # 检查模型是否已拟合（即训练过程中是否已调用 `fit` 方法）
        check_is_fitted(self)
        # 调用 `apply` 方法对输入数据 `X` 进行处理，并使用 `one_hot_encoder_` 进行独热编码转换
        return self.one_hot_encoder_.transform(self.apply(X))
```