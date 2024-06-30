# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_public_functions.py`

```
# 从 importlib 模块导入 import_module 函数，用于动态导入模块
# 从 inspect 模块导入 signature 函数，用于获取函数的签名信息
# 从 numbers 模块导入 Integral 和 Real 类型，用于参数类型检查
import pytest  # 导入 pytest 模块，用于测试

# 从 sklearn.utils._param_validation 模块导入多个函数和类
from sklearn.utils._param_validation import (
    Interval,  # 导入 Interval 类，表示参数的区间范围
    InvalidParameterError,  # 导入 InvalidParameterError 类，表示参数错误异常
    generate_invalid_param_val,  # 导入 generate_invalid_param_val 函数，用于生成无效参数值
    generate_valid_param,  # 导入 generate_valid_param 函数，用于生成有效参数值
    make_constraint,  # 导入 make_constraint 函数，用于生成参数约束条件
)


def _get_func_info(func_module):
    # 根据函数模块名拆分出模块名和函数名
    module_name, func_name = func_module.rsplit(".", 1)
    # 动态导入模块
    module = import_module(module_name)
    # 获取模块中的函数对象
    func = getattr(module, func_name)

    # 获取函数的签名信息
    func_sig = signature(func)
    # 获取函数的所有参数名（不包括 *args 和 **kwargs）
    func_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    # 忽略 `*args` 和 `**kwargs` 参数，因为无法生成约束条件
    required_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    return func, func_name, func_params, required_params


def _check_function_param_validation(
    func, func_name, func_params, required_params, parameter_constraints
):
    """Check that an informative error is raised when the value of a parameter does not
    have an appropriate type or value.
    """
    # 为必需参数生成有效的值
    valid_required_params = {}
    for param_name in required_params:
        if parameter_constraints[param_name] == "no_validation":
            valid_required_params[param_name] = 1  # 如果无需验证，则设定为任意有效值（这里为 1）
        else:
            # 生成有效的参数值，基于参数的约束条件
            valid_required_params[param_name] = generate_valid_param(
                make_constraint(parameter_constraints[param_name][0])
            )

    # 检查每个参数是否都有约束条件
    if func_params:
        validation_params = parameter_constraints.keys()
        # 找出不在函数参数列表中的约束参数
        unexpected_params = set(validation_params) - set(func_params)
        # 找出缺少约束条件的函数参数
        missing_params = set(func_params) - set(validation_params)
        # 构建错误信息，指出参数约束和函数参数之间的不匹配
        err_msg = (
            "Mismatch between _parameter_constraints and the parameters of"
            f" {func_name}.\nConsider the unexpected parameters {unexpected_params} and"
            f" expected but missing parameters {missing_params}\n"
        )
        # 断言确保所有的参数约束都与函数参数匹配，否则抛出错误信息
        assert set(validation_params) == set(func_params), err_msg

    # 创建一个对象，用于表示所有参数类型都不正确的情况
    param_with_bad_type = type("BadType", (), {})()
    # 遍历函数参数列表中的每个参数名
    for param_name in func_params:
        # 获取当前参数名对应的约束条件
        constraints = parameter_constraints[param_name]

        # 如果约束条件为 "no_validation"，表示该参数不需要验证，跳过当前循环
        if constraints == "no_validation":
            # This parameter is not validated
            continue

        # 检查是否存在同时包含整数和实数类型区间的约束条件，如果是则抛出数值错误异常
        if any(
            isinstance(constraint, Interval) and constraint.type == Integral
            for constraint in constraints
        ) and any(
            isinstance(constraint, Interval) and constraint.type == Real
            for constraint in constraints
        ):
            raise ValueError(
                f"The constraint for parameter {param_name} of {func_name} can't have a"
                " mix of intervals of Integral and Real types. Use the type"
                " RealNotInt instead of Real."
            )

        # 构造用于匹配错误信息的正则表达式模式
        match = (
            rf"The '{param_name}' parameter of {func_name} must be .* Got .* instead."
        )

        # 构造用于提示错误消息不足的错误信息
        err_msg = (
            f"{func_name} does not raise an informative error message when the "
            f"parameter {param_name} does not have a valid type. If any Python type "
            "is valid, the constraint should be 'no_validation'."
        )

        # 首先，检查是否会因参数类型不匹配而引发错误
        with pytest.raises(InvalidParameterError, match=match):
            func(**{**valid_required_params, param_name: param_with_bad_type})
            pytest.fail(err_msg)

        # 然后，对于复杂约束条件，检查参数值是否满足任何有效值的要求
        constraints = [make_constraint(constraint) for constraint in constraints]

        # 遍历每个约束条件，生成无效的参数值并验证是否会引发错误
        for constraint in constraints:
            try:
                bad_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue

            # 构造用于提示错误消息不足的错误信息
            err_msg = (
                f"{func_name} does not raise an informative error message when the "
                f"parameter {param_name} does not have a valid value.\n"
                "Constraints should be disjoint. For instance "
                "[StrOptions({'a_string'}), str] is not a acceptable set of "
                "constraint because generating an invalid string for the first "
                "constraint will always produce a valid string for the second "
                "constraint."
            )

            # 检查是否会因参数值不匹配约束条件而引发错误
            with pytest.raises(InvalidParameterError, match=match):
                func(**{**valid_required_params, param_name: bad_value})
                pytest.fail(err_msg)
# 参数验证函数列表，包含多个字符串形式的函数名，每个字符串代表一个函数的全限定名
PARAM_VALIDATION_FUNCTION_LIST = [
    "sklearn.calibration.calibration_curve",
    "sklearn.cluster.cluster_optics_dbscan",
    "sklearn.cluster.compute_optics_graph",
    "sklearn.cluster.estimate_bandwidth",
    "sklearn.cluster.kmeans_plusplus",
    "sklearn.cluster.cluster_optics_xi",
    "sklearn.cluster.ward_tree",
    "sklearn.covariance.empirical_covariance",
    "sklearn.covariance.ledoit_wolf_shrinkage",
    "sklearn.covariance.log_likelihood",
    "sklearn.covariance.shrunk_covariance",
    "sklearn.datasets.clear_data_home",
    "sklearn.datasets.dump_svmlight_file",
    "sklearn.datasets.fetch_20newsgroups",
    "sklearn.datasets.fetch_20newsgroups_vectorized",
    "sklearn.datasets.fetch_california_housing",
    "sklearn.datasets.fetch_covtype",
    "sklearn.datasets.fetch_kddcup99",
    "sklearn.datasets.fetch_lfw_pairs",
    "sklearn.datasets.fetch_lfw_people",
    "sklearn.datasets.fetch_olivetti_faces",
    "sklearn.datasets.fetch_rcv1",
    "sklearn.datasets.fetch_openml",
    "sklearn.datasets.fetch_species_distributions",
    "sklearn.datasets.get_data_home",
    "sklearn.datasets.load_breast_cancer",
    "sklearn.datasets.load_diabetes",
    "sklearn.datasets.load_digits",
    "sklearn.datasets.load_files",
    "sklearn.datasets.load_iris",
    "sklearn.datasets.load_linnerud",
    "sklearn.datasets.load_sample_image",
    "sklearn.datasets.load_svmlight_file",
    "sklearn.datasets.load_svmlight_files",
    "sklearn.datasets.load_wine",
    "sklearn.datasets.make_biclusters",
    "sklearn.datasets.make_blobs",
    "sklearn.datasets.make_checkerboard",
    "sklearn.datasets.make_circles",
    "sklearn.datasets.make_classification",
    "sklearn.datasets.make_friedman1",
    "sklearn.datasets.make_friedman2",
    "sklearn.datasets.make_friedman3",
    "sklearn.datasets.make_gaussian_quantiles",
    "sklearn.datasets.make_hastie_10_2",
    "sklearn.datasets.make_low_rank_matrix",
    "sklearn.datasets.make_moons",
    "sklearn.datasets.make_multilabel_classification",
    "sklearn.datasets.make_regression",
    "sklearn.datasets.make_s_curve",
    "sklearn.datasets.make_sparse_coded_signal",
    "sklearn.datasets.make_sparse_spd_matrix",
    "sklearn.datasets.make_sparse_uncorrelated",
    "sklearn.datasets.make_spd_matrix",
    "sklearn.datasets.make_swiss_roll",
    "sklearn.decomposition.sparse_encode",
    "sklearn.feature_extraction.grid_to_graph",
    "sklearn.feature_extraction.img_to_graph",
    "sklearn.feature_extraction.image.extract_patches_2d",
    "sklearn.feature_extraction.image.reconstruct_from_patches_2d",
    "sklearn.feature_selection.chi2",
    "sklearn.feature_selection.f_classif",
    "sklearn.feature_selection.f_regression",
    "sklearn.feature_selection.mutual_info_classif",
    "sklearn.feature_selection.mutual_info_regression",
    "sklearn.feature_selection.r_regression",
    "sklearn.inspection.partial_dependence",
    "sklearn.inspection.permutation_importance",
]
    # 导入所有的 sklearn.metrics 模块中的函数和类
    from sklearn.metrics import (
        accuracy_score,           # 准确率评估
        auc,                      # AUC
        average_precision_score,  # 平均精度
        balanced_accuracy_score,  # 平衡精度
        brier_score_loss,         # Brier 评分损失
        calinski_harabasz_score,  # Calinski-Harabasz 指数
        check_scoring,            # 检查评分器
        completeness_score,       # 完整度评分
        class_likelihood_ratios,  # 类别似然比
        classification_report,    # 分类报告
        cluster_adjusted_mutual_info_score,  # 聚类调整后的互信息
        cluster_contingency_matrix,          # 聚类列联表
        cluster_entropy,                     # 聚类熵
        cluster_fowlkes_mallows_score,       # 聚类 Fowlkes-Mallows 指数
        cluster_homogeneity_completeness_v_measure,  # 聚类一致性、完整性和 V 度量
        cluster_normalized_mutual_info_score,        # 聚类归一化互信息
        cluster_silhouette_samples,                  # 聚类轮廓系数样本
        cluster_silhouette_score,                   # 聚类轮廓系数
        cohen_kappa_score,          # Cohen's Kappa 系数
        confusion_matrix,           # 混淆矩阵
        consensus_score,            # 一致性得分
        coverage_error,             # 覆盖误差
        d2_absolute_error_score,    # D2 绝对误差得分
        d2_log_loss_score,          # D2 对数损失得分
        d2_pinball_score,           # D2 Pinball 损失得分
        d2_tweedie_score,           # D2 Tweedie 损失得分
        davies_bouldin_score,       # Davies-Bouldin 指数
        dcg_score,                  # 折损累计增益
        det_curve,                  # DET 曲线
        explained_variance_score,   # 解释方差得分
        f1_score,                   # F1 分数
        fbeta_score,                # F-beta 分数
        get_scorer,                 # 获取评分器
        hamming_loss,               # 汉明损失
        hinge_loss,                 # Hinge 损失
        homogeneity_score,          # 同质性得分
        jaccard_score,              # Jaccard 分数
        label_ranking_average_precision_score,  # 标签排序平均精度得分
        label_ranking_loss,         # 标签排序损失
        log_loss,                   # 对数损失
        make_scorer,                # 创建评分器
        matthews_corrcoef,          # Matthews 相关系数
        max_error,                  # 最大误差
        mean_absolute_error,        # 平均绝对误差
        mean_absolute_percentage_error,  # 平均绝对百分比误差
        mean_gamma_deviance,        # 平均 Gamma 偏差
        mean_pinball_loss,          # 平均 Pinball 损失
        mean_poisson_deviance,      # 平均 Poisson 偏差
        mean_squared_error,         # 均方误差
        mean_squared_log_error,     # 均方对数误差
        mean_tweedie_deviance,      # 平均 Tweedie 偏差
        median_absolute_error,      # 中位数绝对误差
        multilabel_confusion_matrix,  # 多标签混淆矩阵
        mutual_info_score,          # 互信息得分
        ndcg_score,                 # 标准化折损累计增益
        pair_confusion_matrix,      # 对分混淆矩阵
        adjusted_rand_score         # 调整后兰德指数
    )
    # 导入需要的模块和函数
    import sklearn.metrics.pairwise
    import sklearn.metrics
    import sklearn.model_selection
    import sklearn.neighbors
    import sklearn.preprocessing
    import sklearn.random_projection
    import sklearn.svm
    import sklearn.tree
    import sklearn.utils
    
    # 下面是一系列的模块和函数的导入，用于科学计算和机器学习任务
    "sklearn.metrics.pairwise.additive_chi2_kernel",  # 导入加性卡方核函数
    "sklearn.metrics.pairwise.chi2_kernel",  # 导入卡方核函数
    "sklearn.metrics.pairwise.cosine_distances",  # 导入余弦距离计算函数
    "sklearn.metrics.pairwise.cosine_similarity",  # 导入余弦相似度计算函数
    "sklearn.metrics.pairwise.euclidean_distances",  # 导入欧氏距离计算函数
    "sklearn.metrics.pairwise.haversine_distances",  # 导入球面距离（haversine）计算函数
    "sklearn.metrics.pairwise.laplacian_kernel",  # 导入拉普拉斯核函数
    "sklearn.metrics.pairwise.linear_kernel",  # 导入线性核函数
    "sklearn.metrics.pairwise.manhattan_distances",  # 导入曼哈顿距离计算函数
    "sklearn.metrics.pairwise.nan_euclidean_distances",  # 导入处理 NaN 的欧氏距离计算函数
    "sklearn.metrics.pairwise.paired_cosine_distances",  # 导入成对余弦距离计算函数
    "sklearn.metrics.pairwise.paired_distances",  # 导入成对距离计算函数
    "sklearn.metrics.pairwise.paired_euclidean_distances",  # 导入成对欧氏距离计算函数
    "sklearn.metrics.pairwise.paired_manhattan_distances",  # 导入成对曼哈顿距离计算函数
    "sklearn.metrics.pairwise.pairwise_distances_argmin_min",  # 导入计算最小距离及其索引函数
    "sklearn.metrics.pairwise.pairwise_kernels",  # 导入成对核函数计算函数
    "sklearn.metrics.pairwise.polynomial_kernel",  # 导入多项式核函数
    "sklearn.metrics.pairwise.rbf_kernel",  # 导入高斯径向基核函数
    "sklearn.metrics.pairwise.sigmoid_kernel",  # 导入 sigmoid 核函数
    "sklearn.metrics.pairwise_distances",  # 导入成对距离计算函数
    "sklearn.metrics.pairwise_distances_argmin",  # 导入距离最小化的索引函数
    "sklearn.metrics.pairwise_distances_chunked",  # 导入分块距离计算函数
    "sklearn.metrics.precision_recall_curve",  # 导入精确率-召回率曲线计算函数
    "sklearn.metrics.precision_recall_fscore_support",  # 导入精确率、召回率、F1 值及支持度计算函数
    "sklearn.metrics.precision_score",  # 导入精确率计算函数
    "sklearn.metrics.r2_score",  # 导入 R^2 得分计算函数
    "sklearn.metrics.rand_score",  # 导入兰德指数计算函数
    "sklearn.metrics.recall_score",  # 导入召回率计算函数
    "sklearn.metrics.roc_auc_score",  # 导入 ROC 曲线下面积计算函数
    "sklearn.metrics.roc_curve",  # 导入 ROC 曲线计算函数
    "sklearn.metrics.root_mean_squared_error",  # 导入均方根误差计算函数
    "sklearn.metrics.root_mean_squared_log_error",  # 导入均方根对数误差计算函数
    "sklearn.metrics.top_k_accuracy_score",  # 导入前 k 个准确率计算函数
    "sklearn.metrics.v_measure_score",  # 导入 V-Measure 计算函数
    "sklearn.metrics.zero_one_loss",  # 导入 0-1 损失计算函数
    "sklearn.model_selection.cross_val_predict",  # 导入交叉验证预测函数
    "sklearn.model_selection.cross_val_score",  # 导入交叉验证得分计算函数
    "sklearn.model_selection.cross_validate",  # 导入交叉验证评估函数
    "sklearn.model_selection.learning_curve",  # 导入学习曲线计算函数
    "sklearn.model_selection.permutation_test_score",  # 导入置换检验得分计算函数
    "sklearn.model_selection.train_test_split",  # 导入训练集与测试集划分函数
    "sklearn.model_selection.validation_curve",  # 导入验证曲线计算函数
    "sklearn.neighbors.kneighbors_graph",  # 导入 k 近邻图计算函数
    "sklearn.neighbors.radius_neighbors_graph",  # 导入半径近邻图计算函数
    "sklearn.neighbors.sort_graph_by_row_values",  # 导入按行值排序图函数
    "sklearn.preprocessing.add_dummy_feature",  # 导入添加虚拟特征函数
    "sklearn.preprocessing.binarize",  # 导入二值化函数
    "sklearn.preprocessing.label_binarize",  # 导入标签二值化函数
    "sklearn.preprocessing.normalize",  # 导入归一化函数
    "sklearn.preprocessing.scale",  # 导入标准化函数
    "sklearn.random_projection.johnson_lindenstrauss_min_dim",  # 导入约翰逊-林登斯特劳斯最小维度计算函数
    "sklearn.svm.l1_min_c",  # 导入线性 SVM 的最小 L1 惩罚系数计算函数
    "sklearn.tree.export_graphviz",  # 导入导出决策树图形描述函数
    "sklearn.tree.export_text",  # 导入导出决策树文本描述函数
    "sklearn.tree.plot_tree",  # 导入绘制决策树函数
    "sklearn.utils.gen_batches",  # 导入生成批次函数
    "sklearn.utils.gen_even_slices",  # 导入生成均匀切片函数
    "sklearn.utils.resample",  # 导入重采样函数
    "sklearn.utils.safe_mask",  # 导入安全掩码函数
    "sklearn.utils.extmath.randomized_svd",  # 导入随机化奇异值分解函数
    "sklearn.utils.class_weight.compute_class_weight",  # 导入计算类别权重函数
    "sklearn.utils.class_weight.compute_sample_weight",  # 导入计算样本权重函数
    "sklearn.utils.graph.single_source_shortest_path_length",  # 导入单源最短路径长度计算函数
# 使用 pytest.mark.parametrize 装饰器为 test_function_param_validation 函数生成多个参数化测试
@pytest.mark.parametrize("func_module", PARAM_VALIDATION_FUNCTION_LIST)
def test_function_param_validation(func_module):
    """Check param validation for public functions that are not wrappers around
    estimators.
    """
    # 从 func_module 中获取函数信息：函数本身 func，函数名称 func_name，函数参数 func_params，必需参数 required_params
    func, func_name, func_params, required_params = _get_func_info(func_module)

    # 获取函数的参数约束信息
    parameter_constraints = getattr(func, "_skl_parameter_constraints")

    # 调用 _check_function_param_validation 函数，验证函数参数的有效性
    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )


# 定义 PARAM_VALIDATION_CLASS_WRAPPER_LIST 列表，包含类包装器的信息
PARAM_VALIDATION_CLASS_WRAPPER_LIST = [
    ("sklearn.cluster.affinity_propagation", "sklearn.cluster.AffinityPropagation"),
    ("sklearn.cluster.dbscan", "sklearn.cluster.DBSCAN"),
    ("sklearn.cluster.k_means", "sklearn.cluster.KMeans"),
    ("sklearn.cluster.mean_shift", "sklearn.cluster.MeanShift"),
    ("sklearn.cluster.spectral_clustering", "sklearn.cluster.SpectralClustering"),
    ("sklearn.covariance.graphical_lasso", "sklearn.covariance.GraphicalLasso"),
    ("sklearn.covariance.ledoit_wolf", "sklearn.covariance.LedoitWolf"),
    ("sklearn.covariance.oas", "sklearn.covariance.OAS"),
    ("sklearn.decomposition.dict_learning", "sklearn.decomposition.DictionaryLearning"),
    (
        "sklearn.decomposition.dict_learning_online",
        "sklearn.decomposition.MiniBatchDictionaryLearning",
    ),
    ("sklearn.decomposition.fastica", "sklearn.decomposition.FastICA"),
    ("sklearn.decomposition.non_negative_factorization", "sklearn.decomposition.NMF"),
    ("sklearn.preprocessing.maxabs_scale", "sklearn.preprocessing.MaxAbsScaler"),
    ("sklearn.preprocessing.minmax_scale", "sklearn.preprocessing.MinMaxScaler"),
    ("sklearn.preprocessing.power_transform", "sklearn.preprocessing.PowerTransformer"),
    (
        "sklearn.preprocessing.quantile_transform",
        "sklearn.preprocessing.QuantileTransformer",
    ),
    ("sklearn.preprocessing.robust_scale", "sklearn.preprocessing.RobustScaler"),
]


# 使用 pytest.mark.parametrize 装饰器为 test_class_wrapper_param_validation 函数生成多个参数化测试
@pytest.mark.parametrize(
    "func_module, class_module", PARAM_VALIDATION_CLASS_WRAPPER_LIST
)
def test_class_wrapper_param_validation(func_module, class_module):
    """Check param validation for public functions that are wrappers around
    estimators.
    """
    # 从 func_module 中获取函数信息：函数本身 func，函数名称 func_name，函数参数 func_params，必需参数 required_params
    func, func_name, func_params, required_params = _get_func_info(func_module)

    # 解析类包装器的模块和类名称
    module_name, class_name = class_module.rsplit(".", 1)
    module = import_module(module_name)
    klass = getattr(module, class_name)

    # 获取函数和类的参数约束信息
    parameter_constraints_func = getattr(func, "_skl_parameter_constraints")
    parameter_constraints_class = getattr(klass, "_parameter_constraints")

    # 合并函数和类的参数约束信息
    parameter_constraints = {
        **parameter_constraints_class,
        **parameter_constraints_func,
    }

    # 筛选出 func_params 中存在的参数约束
    parameter_constraints = {
        k: v for k, v in parameter_constraints.items() if k in func_params
    }

    # 调用 _check_function_param_validation 函数，验证函数参数的有效性
    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )
```