# `D:\src\scipysrc\scikit-learn\doc\api_reference.py`

```
"""
Configuration for the API reference documentation.
"""


def _get_guide(*refs, is_developer=False):
    """Get the rst to refer to user/developer guide.

    `refs` is several references that can be used in the :ref:`...` directive.
    """
    # Determine the description based on the number of references provided
    if len(refs) == 1:
        ref_desc = f":ref:`{refs[0]}` section"
    elif len(refs) == 2:
        ref_desc = f":ref:`{refs[0]}` and :ref:`{refs[1]}` sections"
    else:
        ref_desc = ", ".join(f":ref:`{ref}`" for ref in refs[:-1])
        ref_desc += f", and :ref:`{refs[-1]}` sections"

    # Determine whether the guide is for Developer or User
    guide_name = "Developer" if is_developer else "User"
    # Construct the final guide message
    return f"**{guide_name} guide.** See the {ref_desc} for further details."


def _get_submodule(module_name, submodule_name):
    """Get the submodule docstring and automatically add the hook.

    `module_name` is e.g. `sklearn.feature_extraction`, and `submodule_name` is e.g.
    `image`, so we get the docstring and hook for `sklearn.feature_extraction.image`
    submodule. `module_name` is used to reset the current module because autosummary
    automatically changes the current module.
    """
    # Prepare lines for automodule directive and currentmodule directive
    lines = [
        f".. automodule:: {module_name}.{submodule_name}",
        f".. currentmodule:: {module_name}",
    ]
    # Join lines with double newline separator
    return "\n\n".join(lines)


"""
CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to a dictionary that consists of the following
components:

short_summary (required)
    The text to be printed on the index page; it has nothing to do the API reference
    page of each module.
description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - title (required, `None` if not needed): the section title, commonly it should
      not be `None` except for the first section of a module,
    - description (optional): the optional additional description for the section,
    - autosummary (required): an autosummary block, assuming current module is the
      current module name.

Essentially, the rendered page would look like the following:

|---------------------------------------------------------------------------------|
|     {{ module_name }}                                                           |
|     =================                                                           |
|     {{ module_docstring }}                                                      |
|     {{ description }}                                                           |
|                                                                                 |
|     {{ section_title_1 }}   <-------------- Optional if one wants the first     |
|     ---------------------                   section to directly follow          |
|     {{ section_description_1 }}             without a second-level heading.     |

"""
# 定义一个API参考的字典
API_REFERENCE = {
    # scikit-learn库的主要部分
    "sklearn": {
        # 简短的概述信息
        "short_summary": "Settings and information tools.",
        # 描述信息为空
        "description": None,
        # 包含的各个章节列表
        "sections": [
            {
                # 章节的标题为空
                "title": None,
                # 自动摘要的列表
                "autosummary": [
                    "config_context",
                    "get_config",
                    "set_config",
                    "show_versions",
                ],
            },
        ],
    },
    # scikit-learn库的基础部分
    "sklearn.base": {
        # 简短的概述信息
        "short_summary": "Base classes and utility functions.",
        # 描述信息为空
        "description": None,
        # 包含的各个章节列表
        "sections": [
            {
                # 章节的标题为空
                "title": None,
                # 自动摘要的列表
                "autosummary": [
                    "BaseEstimator",
                    "BiclusterMixin",
                    "ClassNamePrefixFeaturesOutMixin",
                    "ClassifierMixin",
                    "ClusterMixin",
                    "DensityMixin",
                    "MetaEstimatorMixin",
                    "OneToOneFeatureMixin",
                    "OutlierMixin",
                    "RegressorMixin",
                    "TransformerMixin",
                    "clone",
                    "is_classifier",
                    "is_clusterer",
                    "is_regressor",
                ],
            }
        ],
    },
}
    "sklearn.calibration": {
        "short_summary": "Probability calibration.",
        "description": _get_guide("calibration"),  # 获取概率校准指南的描述内容
        "sections": [
            {
                "title": None,  # 没有特定的标题
                "autosummary": ["CalibratedClassifierCV", "calibration_curve"],  # 自动摘要列表包括 CalibratedClassifierCV 和 calibration_curve
            },
            {
                "title": "Visualization",  # 可视化部分的标题为 "Visualization"
                "autosummary": ["CalibrationDisplay"],  # 自动摘要列表包括 CalibrationDisplay
            },
        ],
    },
    "sklearn.cluster": {
        "short_summary": "Clustering.",  # 聚类的简短摘要
        "description": _get_guide("clustering", "biclustering"),  # 获取聚类和双聚类指南的描述内容
        "sections": [
            {
                "title": None,  # 没有特定的标题
                "autosummary": [  # 自动摘要列表包括多个聚类算法的名称
                    "AffinityPropagation",
                    "AgglomerativeClustering",
                    "Birch",
                    "BisectingKMeans",
                    "DBSCAN",
                    "FeatureAgglomeration",
                    "HDBSCAN",
                    "KMeans",
                    "MeanShift",
                    "MiniBatchKMeans",
                    "OPTICS",
                    "SpectralBiclustering",
                    "SpectralClustering",
                    "SpectralCoclustering",
                    "affinity_propagation",
                    "cluster_optics_dbscan",
                    "cluster_optics_xi",
                    "compute_optics_graph",
                    "dbscan",
                    "estimate_bandwidth",
                    "k_means",
                    "kmeans_plusplus",
                    "mean_shift",
                    "spectral_clustering",
                    "ward_tree",
                ],
            },
        ],
    },
    "sklearn.compose": {
        "short_summary": "Composite estimators.",  # 复合估计器的简短摘要
        "description": _get_guide("combining_estimators"),  # 获取组合估计器指南的描述内容
        "sections": [
            {
                "title": None,  # 没有特定的标题
                "autosummary": [  # 自动摘要列表包括多个组合估计器相关的名称
                    "ColumnTransformer",
                    "TransformedTargetRegressor",
                    "make_column_selector",
                    "make_column_transformer",
                ],
            },
        ],
    },
    "sklearn.covariance": {
        "short_summary": "Covariance estimation.",  # 协方差估计的简短摘要
        "description": _get_guide("covariance"),  # 获取协方差指南的描述内容
        "sections": [
            {
                "title": None,  # 没有特定的标题
                "autosummary": [  # 自动摘要列表包括多个协方差估计相关的名称
                    "EllipticEnvelope",
                    "EmpiricalCovariance",
                    "GraphicalLasso",
                    "GraphicalLassoCV",
                    "LedoitWolf",
                    "MinCovDet",
                    "OAS",
                    "ShrunkCovariance",
                    "empirical_covariance",
                    "graphical_lasso",
                    "ledoit_wolf",
                    "ledoit_wolf_shrinkage",
                    "oas",
                    "shrunk_covariance",
                ],
            },
        ],
    },
    {
        # 定义一个字典条目，键为"sklearn.cross_decomposition"
        "sklearn.cross_decomposition": {
            # 提供"sklearn.cross_decomposition"的简短摘要
            "short_summary": "Cross decomposition.",
            # 获取"cross_decomposition"的指南描述，并赋给"description"
            "description": _get_guide("cross_decomposition"),
            # 包含"sklearn.cross_decomposition"的章节列表
            "sections": [
                {
                    # 第一个章节的标题为None
                    "title": None,
                    # 包含自动摘要的列表，包括"CCA", "PLSCanonical", "PLSRegression", "PLSSVD"
                    "autosummary": ["CCA", "PLSCanonical", "PLSRegression", "PLSSVD"],
                },
            ],
        },
        # 定义另一个字典条目，键为"sklearn.datasets"
        "sklearn.datasets": {
            # 提供"sklearn.datasets"的简短摘要
            "short_summary": "Datasets.",
            # 获取"datasets"的指南描述，并赋给"description"
            "description": _get_guide("datasets"),
            # 包含"sklearn.datasets"的章节列表
            "sections": [
                {
                    # 第一个章节的标题为"Loaders"
                    "title": "Loaders",
                    # 包含自动摘要的列表，列出数据集加载相关函数
                    "autosummary": [
                        "clear_data_home",
                        "dump_svmlight_file",
                        "fetch_20newsgroups",
                        "fetch_20newsgroups_vectorized",
                        "fetch_california_housing",
                        "fetch_covtype",
                        "fetch_kddcup99",
                        "fetch_lfw_pairs",
                        "fetch_lfw_people",
                        "fetch_olivetti_faces",
                        "fetch_openml",
                        "fetch_rcv1",
                        "fetch_species_distributions",
                        "get_data_home",
                        "load_breast_cancer",
                        "load_diabetes",
                        "load_digits",
                        "load_files",
                        "load_iris",
                        "load_linnerud",
                        "load_sample_image",
                        "load_sample_images",
                        "load_svmlight_file",
                        "load_svmlight_files",
                        "load_wine",
                    ],
                },
                {
                    # 第二个章节的标题为"Sample generators"
                    "title": "Sample generators",
                    # 包含自动摘要的列表，列出数据集生成器相关函数
                    "autosummary": [
                        "make_biclusters",
                        "make_blobs",
                        "make_checkerboard",
                        "make_circles",
                        "make_classification",
                        "make_friedman1",
                        "make_friedman2",
                        "make_friedman3",
                        "make_gaussian_quantiles",
                        "make_hastie_10_2",
                        "make_low_rank_matrix",
                        "make_moons",
                        "make_multilabel_classification",
                        "make_regression",
                        "make_s_curve",
                        "make_sparse_coded_signal",
                        "make_sparse_spd_matrix",
                        "make_sparse_uncorrelated",
                        "make_spd_matrix",
                        "make_swiss_roll",
                    ],
                },
            ],
        },
    }
    # 定义"sklearn.decomposition"模块的描述信息和自动生成的文档部分
    "sklearn.decomposition": {
        "short_summary": "Matrix decomposition.",  # 模块的简短摘要
        "description": _get_guide("decompositions"),  # 根据指定的引导函数获取模块的详细描述
        "sections": [  # 模块的文档章节
            {
                "title": None,  # 章节标题为空
                "autosummary": [  # 自动摘要部分，列出自动化生成的类和函数摘要列表
                    "DictionaryLearning",
                    "FactorAnalysis",
                    "FastICA",
                    "IncrementalPCA",
                    "KernelPCA",
                    "LatentDirichletAllocation",
                    "MiniBatchDictionaryLearning",
                    "MiniBatchNMF",
                    "MiniBatchSparsePCA",
                    "NMF",
                    "PCA",
                    "SparseCoder",
                    "SparsePCA",
                    "TruncatedSVD",
                    "dict_learning",
                    "dict_learning_online",
                    "fastica",
                    "non_negative_factorization",
                    "sparse_encode",
                ],
            },
        ],
    },
    # 定义"sklearn.discriminant_analysis"模块的描述信息和自动生成的文档部分
    "sklearn.discriminant_analysis": {
        "short_summary": "Discriminant analysis.",  # 模块的简短摘要
        "description": _get_guide("lda_qda"),  # 根据指定的引导函数获取模块的详细描述
        "sections": [  # 模块的文档章节
            {
                "title": None,  # 章节标题为空
                "autosummary": [  # 自动摘要部分，列出自动化生成的类和函数摘要列表
                    "LinearDiscriminantAnalysis",
                    "QuadraticDiscriminantAnalysis",
                ],
            },
        ],
    },
    # 定义"sklearn.dummy"模块的描述信息和自动生成的文档部分
    "sklearn.dummy": {
        "short_summary": "Dummy estimators.",  # 模块的简短摘要
        "description": _get_guide("model_evaluation"),  # 根据指定的引导函数获取模块的详细描述
        "sections": [  # 模块的文档章节
            {
                "title": None,  # 章节标题为空
                "autosummary": [  # 自动摘要部分，列出自动化生成的类和函数摘要列表
                    "DummyClassifier",
                    "DummyRegressor",
                ],
            },
        ],
    },
    # 定义"sklearn.ensemble"模块的描述信息和自动生成的文档部分
    "sklearn.ensemble": {
        "short_summary": "Ensemble methods.",  # 模块的简短摘要
        "description": _get_guide("ensemble"),  # 根据指定的引导函数获取模块的详细描述
        "sections": [  # 模块的文档章节
            {
                "title": None,  # 章节标题为空
                "autosummary": [  # 自动摘要部分，列出自动化生成的类和函数摘要列表
                    "AdaBoostClassifier",
                    "AdaBoostRegressor",
                    "BaggingClassifier",
                    "BaggingRegressor",
                    "ExtraTreesClassifier",
                    "ExtraTreesRegressor",
                    "GradientBoostingClassifier",
                    "GradientBoostingRegressor",
                    "HistGradientBoostingClassifier",
                    "HistGradientBoostingRegressor",
                    "IsolationForest",
                    "RandomForestClassifier",
                    "RandomForestRegressor",
                    "RandomTreesEmbedding",
                    "StackingClassifier",
                    "StackingRegressor",
                    "VotingClassifier",
                    "VotingRegressor",
                ],
            },
        ],
    },
    {
        "sklearn.exceptions": {
            "short_summary": "Exceptions and warnings.",
            "description": None,  # 无描述信息
            "sections": [  # 开始处理sections列表
                {
                    "title": None,  # 子节标题为空
                    "autosummary": [  # autosummary包含以下条目
                        "ConvergenceWarning",  # 收敛警告
                        "DataConversionWarning",  # 数据转换警告
                        "DataDimensionalityWarning",  # 数据维度警告
                        "EfficiencyWarning",  # 效率警告
                        "FitFailedWarning",  # 拟合失败警告
                        "InconsistentVersionWarning",  # 版本不一致警告
                        "NotFittedError",  # 未拟合错误
                        "UndefinedMetricWarning",  # 未定义指标警告
                    ],
                },
            ],
        },
        "sklearn.experimental": {
            "short_summary": "Experimental tools.",  # 实验性工具
            "description": None,  # 无描述信息
            "sections": [  # 开始处理sections列表
                {
                    "title": None,  # 子节标题为空
                    "autosummary": [  # autosummary包含以下条目
                        "enable_halving_search_cv",  # 启用减半搜索交叉验证
                        "enable_iterative_imputer",  # 启用迭代式填充器
                    ],
                },
            ],
        },
        "sklearn.feature_extraction": {
            "short_summary": "Feature extraction.",  # 特征提取
            "description": _get_guide("feature_extraction"),  # 获取特征提取指南的描述
            "sections": [  # 开始处理sections列表
                {
                    "title": None,  # 子节标题为空
                    "autosummary": [  # autosummary包含以下条目
                        "DictVectorizer",  # 字典向量化器
                        "FeatureHasher",  # 特征哈希器
                    ],
                },
                {
                    "title": "From images",  # 图像部分标题
                    "description": _get_submodule("sklearn.feature_extraction", "image"),  # 获取图像子模块的描述
                    "autosummary": [  # autosummary包含以下条目
                        "image.PatchExtractor",  # 图像补丁提取器
                        "image.extract_patches_2d",  # 提取2D图像补丁
                        "image.grid_to_graph",  # 网格转图形
                        "image.img_to_graph",  # 图像转图形
                        "image.reconstruct_from_patches_2d",  # 从2D补丁重建图像
                    ],
                },
                {
                    "title": "From text",  # 文本部分标题
                    "description": _get_submodule("sklearn.feature_extraction", "text"),  # 获取文本子模块的描述
                    "autosummary": [  # autosummary包含以下条目
                        "text.CountVectorizer",  # 计数向量化器
                        "text.HashingVectorizer",  # 哈希向量化器
                        "text.TfidfTransformer",  # TF-IDF转换器
                        "text.TfidfVectorizer",  # TF-IDF向量化器
                    ],
                },
            ],
        },
        "sklearn.feature_selection": {
            "short_summary": "Feature selection.",  # 特征选择
            "description": _get_guide("feature_selection"),  # 获取特征选择指南的描述
            "sections": [  # 开始处理sections列表
                {
                    "title": None,  # 子节标题为空
                    "autosummary": [  # autosummary包含以下条目
                        "GenericUnivariateSelect",  # 通用单变量选择器
                        "RFE",  # 递归特征消除
                        "RFECV",  # 递归特征消除交叉验证
                        "SelectFdr",  # FDR选择器
                        "SelectFpr",  # FPR选择器
                        "SelectFromModel",  # 基于模型选择
                        "SelectFwe",  # FWE选择器
                        "SelectKBest",  # K最佳选择器
                        "SelectPercentile",  # 百分位选择器
                        "SelectorMixin",  # 选择器混合器
                        "SequentialFeatureSelector",  # 顺序特征选择器
                        "VarianceThreshold",  # 方差阈值
                        "chi2",  # 卡方统计量
                        "f_classif",  # F统计量
                        "f_regression",  # F回归
                        "mutual_info_classif",  # 互信息分类
                        "mutual_info_regression",  # 互信息回归
                        "r_regression",  # R回归
                    ],
                },
            ],
        },
    }
    {
        # 定义一个字典，每个键对应于一个模块的名称，值是一个包含有关该模块的信息的字典
        "sklearn.gaussian_process": {
            # 简要总结高斯过程的功能
            "short_summary": "Gaussian processes.",
            # 使用_get_guide函数获取关于高斯过程的详细指南
            "description": _get_guide("gaussian_process"),
            # 列出模块的各个部分，每个部分由标题和自动摘要组成
            "sections": [
                {
                    # 第一个部分没有标题
                    "title": None,
                    # 列出与高斯过程相关的分类器和回归器
                    "autosummary": [
                        "GaussianProcessClassifier",
                        "GaussianProcessRegressor",
                    ],
                },
                {
                    # 第二个部分标题为“Kernels”
                    "title": "Kernels",
                    # 获取关于内核的子模块信息
                    "description": _get_submodule("sklearn.gaussian_process", "kernels"),
                    # 列出各种可用的内核函数
                    "autosummary": [
                        "kernels.CompoundKernel",
                        "kernels.ConstantKernel",
                        "kernels.DotProduct",
                        "kernels.ExpSineSquared",
                        "kernels.Exponentiation",
                        "kernels.Hyperparameter",
                        "kernels.Kernel",
                        "kernels.Matern",
                        "kernels.PairwiseKernel",
                        "kernels.Product",
                        "kernels.RBF",
                        "kernels.RationalQuadratic",
                        "kernels.Sum",
                        "kernels.WhiteKernel",
                    ],
                },
            ],
        },
        # sklearn.impute模块的信息
        "sklearn.impute": {
            # 简要总结缺失值填充的功能
            "short_summary": "Imputation.",
            # 使用_get_guide函数获取关于缺失值填充的详细指南
            "description": _get_guide("impute"),
            # 列出模块的部分，只有一个没有标题的部分
            "sections": [
                {
                    "title": None,
                    # 列出不同的填充方法和指示器
                    "autosummary": [
                        "IterativeImputer",
                        "KNNImputer",
                        "MissingIndicator",
                        "SimpleImputer",
                    ],
                },
            ],
        },
        # sklearn.inspection模块的信息
        "sklearn.inspection": {
            # 简要总结模型检验的功能
            "short_summary": "Inspection.",
            # 使用_get_guide函数获取关于模型检验的详细指南
            "description": _get_guide("inspection"),
            # 列出模块的部分，第一个部分没有标题，第二个部分标题为“Plotting”
            "sections": [
                {
                    "title": None,
                    # 列出部分依赖和排列重要性的自动摘要
                    "autosummary": ["partial_dependence", "permutation_importance"],
                },
                {
                    "title": "Plotting",
                    # 列出决策边界和部分依赖展示的自动摘要
                    "autosummary": ["DecisionBoundaryDisplay", "PartialDependenceDisplay"],
                },
            ],
        },
        # sklearn.isotonic模块的信息
        "sklearn.isotonic": {
            # 简要总结保序回归的功能
            "short_summary": "Isotonic regression.",
            # 使用_get_guide函数获取关于保序回归的详细指南
            "description": _get_guide("isotonic"),
            # 列出模块的部分，第一个部分没有标题
            "sections": [
                {
                    "title": None,
                    # 列出保序回归相关的类和函数
                    "autosummary": [
                        "IsotonicRegression",
                        "check_increasing",
                        "isotonic_regression",
                    ],
                },
            ],
        },
        # sklearn.kernel_approximation模块的信息
        "sklearn.kernel_approximation": {
            # 简要总结核近似的功能
            "short_summary": "Isotonic regression.",
            # 使用_get_guide函数获取关于核近似的详细指南
            "description": _get_guide("kernel_approximation"),
            # 列出模块的部分，第一个部分没有标题
            "sections": [
                {
                    "title": None,
                    # 列出各种核近似的自动摘要
                    "autosummary": [
                        "AdditiveChi2Sampler",
                        "Nystroem",
                        "PolynomialCountSketch",
                        "RBFSampler",
                        "SkewedChi2Sampler",
                    ],
                },
            ],
        },
    }
    "sklearn.kernel_ridge": {
        "short_summary": "Kernel ridge regression.",
        "description": _get_guide("kernel_ridge"),  # 使用函数 _get_guide 获取 "kernel_ridge" 的说明文档
        "sections": [
            {
                "title": None,  # 没有指定标题
                "autosummary": ["KernelRidge"],  # 自动摘要列表中包含 "KernelRidge"
            },
        ],
    },
    },
    "sklearn.manifold": {
        "short_summary": "Manifold learning.",
        "description": _get_guide("manifold"),  # 使用函数 _get_guide 获取 "manifold" 的说明文档
        "sections": [
            {
                "title": None,  # 没有指定标题
                "autosummary": [
                    "Isomap",  # Isomap
                    "LocallyLinearEmbedding",  # LocallyLinearEmbedding
                    "MDS",  # MDS
                    "SpectralEmbedding",  # SpectralEmbedding
                    "TSNE",  # TSNE
                    "locally_linear_embedding",  # locally_linear_embedding
                    "smacof",  # smacof
                    "spectral_embedding",  # spectral_embedding
                    "trustworthiness",  # trustworthiness
                ],
            },
        ],
    },
    },
    "sklearn.mixture": {
        "short_summary": "Gaussian mixture models.",
        "description": _get_guide("mixture"),  # 使用函数 _get_guide 获取 "mixture" 的说明文档
        "sections": [
            {
                "title": None,  # 没有指定标题
                "autosummary": ["BayesianGaussianMixture", "GaussianMixture"],  # 自动摘要列表中包含 "BayesianGaussianMixture" 和 "GaussianMixture"
            },
        ],
    },
    {
        # 定义了名为 "sklearn.model_selection" 的字典，包含模型选择相关信息
        "sklearn.model_selection": {
            # 简短总结，描述为模型选择
            "short_summary": "Model selection.",
            # 描述，调用 _get_guide 函数获取有关交叉验证、网格搜索和学习曲线的指南
            "description": _get_guide("cross_validation", "grid_search", "learning_curve"),
            # 包含多个部分的列表，每个部分有标题和自动摘要列表
            "sections": [
                {
                    # 分类器
                    "title": "Splitters",
                    # 自动摘要列表，包含各种数据分割器类的名称
                    "autosummary": [
                        "GroupKFold",
                        "GroupShuffleSplit",
                        "KFold",
                        "LeaveOneGroupOut",
                        "LeaveOneOut",
                        "LeavePGroupsOut",
                        "LeavePOut",
                        "PredefinedSplit",
                        "RepeatedKFold",
                        "RepeatedStratifiedKFold",
                        "ShuffleSplit",
                        "StratifiedGroupKFold",
                        "StratifiedKFold",
                        "StratifiedShuffleSplit",
                        "TimeSeriesSplit",
                        "check_cv",
                        "train_test_split",
                    ],
                },
                {
                    # 超参数优化器
                    "title": "Hyper-parameter optimizers",
                    # 自动摘要列表，包含各种超参数优化器类的名称
                    "autosummary": [
                        "GridSearchCV",
                        "HalvingGridSearchCV",
                        "HalvingRandomSearchCV",
                        "ParameterGrid",
                        "ParameterSampler",
                        "RandomizedSearchCV",
                    ],
                },
                {
                    # 拟合后模型调优
                    "title": "Post-fit model tuning",
                    # 自动摘要列表，包含各种模型调优类的名称
                    "autosummary": [
                        "FixedThresholdClassifier",
                        "TunedThresholdClassifierCV",
                    ],
                },
                {
                    # 模型验证
                    "title": "Model validation",
                    # 自动摘要列表，包含各种模型验证函数的名称
                    "autosummary": [
                        "cross_val_predict",
                        "cross_val_score",
                        "cross_validate",
                        "learning_curve",
                        "permutation_test_score",
                        "validation_curve",
                    ],
                },
                {
                    # 可视化
                    "title": "Visualization",
                    # 自动摘要列表，包含各种可视化类的名称
                    "autosummary": ["LearningCurveDisplay", "ValidationCurveDisplay"],
                },
            ],
        },
        {
            # 定义了名为 "sklearn.multiclass" 的字典，包含多类分类相关信息
            "sklearn.multiclass": {
                # 简短总结，描述为多类分类
                "short_summary": "Multiclass classification.",
                # 描述，调用 _get_guide 函数获取有关多类分类的指南
                "description": _get_guide("multiclass_classification"),
                # 包含单个部分的列表，无标题，包含多类分类器的名称
                "sections": [
                    {
                        # 无标题
                        "title": None,
                        # 自动摘要列表，包含各种多类分类器类的名称
                        "autosummary": [
                            "OneVsOneClassifier",
                            "OneVsRestClassifier",
                            "OutputCodeClassifier",
                        ],
                    },
                ],
            },
        },
    }
    {
        "sklearn.multioutput": {
            "short_summary": "Multioutput regression and classification.",
            "description": _get_guide(
                "multilabel_classification",
                "multiclass_multioutput_classification",
                "multioutput_regression",
            ),
            "sections": [  # 开始描述 sklearn.multioutput 的各个部分
                {
                    "title": None,  # 无特定标题
                    "autosummary": [  # 自动摘要列表开始
                        "ClassifierChain",  # 分类链模型
                        "MultiOutputClassifier",  # 多输出分类器
                        "MultiOutputRegressor",  # 多输出回归器
                        "RegressorChain",  # 回归链模型
                    ],  # 自动摘要列表结束
                },
            ],  # 结束描述 sklearn.multioutput 的各个部分
        },  # sklearn.multioutput 结束
    
        "sklearn.naive_bayes": {
            "short_summary": "Naive Bayes.",  # 简短摘要：朴素贝叶斯
            "description": _get_guide("naive_bayes"),  # 获取朴素贝叶斯的指南信息
            "sections": [  # 开始描述 sklearn.naive_bayes 的各个部分
                {
                    "title": None,  # 无特定标题
                    "autosummary": [  # 自动摘要列表开始
                        "BernoulliNB",  # 伯努利朴素贝叶斯
                        "CategoricalNB",  # 分类朴素贝叶斯
                        "ComplementNB",  # 补充朴素贝叶斯
                        "GaussianNB",  # 高斯朴素贝叶斯
                        "MultinomialNB",  # 多项式朴素贝叶斯
                    ],  # 自动摘要列表结束
                },
            ],  # 结束描述 sklearn.naive_bayes 的各个部分
        },  # sklearn.naive_bayes 结束
    
        "sklearn.neighbors": {
            "short_summary": "Nearest neighbors.",  # 简短摘要：最近邻
            "description": _get_guide("neighbors"),  # 获取最近邻的指南信息
            "sections": [  # 开始描述 sklearn.neighbors 的各个部分
                {
                    "title": None,  # 无特定标题
                    "autosummary": [  # 自动摘要列表开始
                        "BallTree",  # Ball 树
                        "KDTree",  # KD 树
                        "KNeighborsClassifier",  # K 近邻分类器
                        "KNeighborsRegressor",  # K 近邻回归器
                        "KNeighborsTransformer",  # K 近邻转换器
                        "KernelDensity",  # 核密度估计
                        "LocalOutlierFactor",  # 局部异常因子
                        "NearestCentroid",  # 最近质心分类器
                        "NearestNeighbors",  # 最近邻分类器
                        "NeighborhoodComponentsAnalysis",  # 邻域组分分析
                        "RadiusNeighborsClassifier",  # 半径最近邻分类器
                        "RadiusNeighborsRegressor",  # 半径最近邻回归器
                        "RadiusNeighborsTransformer",  # 半径最近邻转换器
                        "kneighbors_graph",  # K 近邻图
                        "radius_neighbors_graph",  # 半径最近邻图
                        "sort_graph_by_row_values",  # 按行值排序图
                    ],  # 自动摘要列表结束
                },
            ],  # 结束描述 sklearn.neighbors 的各个部分
        },  # sklearn.neighbors 结束
    
        "sklearn.neural_network": {
            "short_summary": "Neural network models.",  # 简短摘要：神经网络模型
            "description": _get_guide(  # 获取神经网络模型的指南信息
                "neural_networks_supervised", "neural_networks_unsupervised"
            ),
            "sections": [  # 开始描述 sklearn.neural_network 的各个部分
                {
                    "title": None,  # 无特定标题
                    "autosummary": [  # 自动摘要列表开始
                        "BernoulliRBM",  # 伯努利受限玻尔兹曼机
                        "MLPClassifier",  # 多层感知器分类器
                        "MLPRegressor",  # 多层感知器回归器
                    ],  # 自动摘要列表结束
                },
            ],  # 结束描述 sklearn.neural_network 的各个部分
        },  # sklearn.neural_network 结束
    
        "sklearn.pipeline": {
            "short_summary": "Pipeline.",  # 简短摘要：管道
            "description": _get_guide("combining_estimators"),  # 获取组合估计器的指南信息
            "sections": [  # 开始描述 sklearn.pipeline 的各个部分
                {
                    "title": None,  # 无特定标题
                    "autosummary": [  # 自动摘要列表开始
                        "FeatureUnion",  # 特征联合
                        "Pipeline",  # 管道
                        "make_pipeline",  # 创建管道
                        "make_union",  # 创建联合
                    ],  # 自动摘要列表结束
                },
            ],  # 结束描述 sklearn.pipeline 的各个部分
        },  # sklearn.pipeline 结束
    }
    "sklearn.preprocessing": {
        "short_summary": "数据预处理和归一化。",
        "description": _get_guide("preprocessing"),  # 调用_get_guide函数获取预处理指南的描述信息
        "sections": [  # 开始描述预处理模块的各个部分
            {
                "title": None,  # 没有特定的标题
                "autosummary": [  # 自动摘要部分列出了预处理模块中的类和函数
                    "Binarizer",  # 二值化
                    "FunctionTransformer",  # 函数转换器
                    "KBinsDiscretizer",  # K桶离散化
                    "KernelCenterer",  # 核心居中器
                    "LabelBinarizer",  # 标签二值化
                    "LabelEncoder",  # 标签编码器
                    "MaxAbsScaler",  # 最大绝对值缩放
                    "MinMaxScaler",  # 最小-最大缩放
                    "MultiLabelBinarizer",  # 多标签二值化
                    "Normalizer",  # 归一化
                    "OneHotEncoder",  # 独热编码器
                    "OrdinalEncoder",  # 顺序编码器
                    "PolynomialFeatures",  # 多项式特征
                    "PowerTransformer",  # 功率变换
                    "QuantileTransformer",  # 分位数变换
                    "RobustScaler",  # 鲁棒缩放
                    "SplineTransformer",  # 样条变换
                    "StandardScaler",  # 标准化缩放
                    "TargetEncoder",  # 目标编码器
                    "add_dummy_feature",  # 添加虚拟特征
                    "binarize",  # 二值化
                    "label_binarize",  # 标签二值化
                    "maxabs_scale",  # 最大绝对值缩放
                    "minmax_scale",  # 最小-最大缩放
                    "normalize",  # 归一化
                    "power_transform",  # 功率变换
                    "quantile_transform",  # 分位数变换
                    "robust_scale",  # 鲁棒缩放
                    "scale",  # 缩放
                ],
            },
        ],
    },
    "sklearn.random_projection": {
        "short_summary": "随机投影。",
        "description": _get_guide("random_projection"),  # 调用_get_guide函数获取随机投影指南的描述信息
        "sections": [  # 开始描述随机投影模块的各个部分
            {
                "title": None,  # 没有特定的标题
                "autosummary": [  # 自动摘要部分列出了随机投影模块中的类
                    "GaussianRandomProjection",  # 高斯随机投影
                    "SparseRandomProjection",  # 稀疏随机投影
                    "johnson_lindenstrauss_min_dim",  # 约翰逊-林登斯特劳斯最小维数
                ],
            },
        ],
    },
    "sklearn.semi_supervised": {
        "short_summary": "半监督学习。",
        "description": _get_guide("semi_supervised"),  # 调用_get_guide函数获取半监督学习指南的描述信息
        "sections": [  # 开始描述半监督学习模块的各个部分
            {
                "title": None,  # 没有特定的标题
                "autosummary": [  # 自动摘要部分列出了半监督学习模块中的类
                    "LabelPropagation",  # 标签传播
                    "LabelSpreading",  # 标签扩散
                    "SelfTrainingClassifier",  # 自训练分类器
                ],
            },
        ],
    },
    "sklearn.svm": {
        "short_summary": "支持向量机。",
        "description": _get_guide("svm"),  # 调用_get_guide函数获取支持向量机指南的描述信息
        "sections": [  # 开始描述支持向量机模块的各个部分
            {
                "title": None,  # 没有特定的标题
                "autosummary": [  # 自动摘要部分列出了支持向量机模块中的类
                    "LinearSVC",  # 线性支持向量分类器
                    "LinearSVR",  # 线性支持向量回归器
                    "NuSVC",  # Nu支持向量分类器
                    "NuSVR",  # Nu支持向量回归器
                    "OneClassSVM",  # 单类支持向量机
                    "SVC",  # 支持向量分类器
                    "SVR",  # 支持向量回归器
                    "l1_min_c",  # l1最小C值
                ],
            },
        ],
    },
    "sklearn.tree": {  # 创建一个名为 "sklearn.tree" 的字典条目
        "short_summary": "Decision trees.",  # 键为 "short_summary"，值为 "Decision trees."
        "description": _get_guide("tree"),  # 键为 "description"，值为调用 _get_guide 函数返回的内容，参数为 "tree"
        "sections": [  # 键为 "sections"，值为包含多个字典的列表
            {  # 列表中的第一个字典
                "title": None,  # 第一个字典的键 "title" 的值为 None
                "autosummary": [  # 第一个字典的键 "autosummary" 的值为包含多个字符串的列表
                    "DecisionTreeClassifier",  # 列表中的第一个字符串
                    "DecisionTreeRegressor",  # 列表中的第二个字符串
                    "ExtraTreeClassifier",  # 列表中的第三个字符串
                    "ExtraTreeRegressor",  # 列表中的第四个字符串
                ],
            },
            {  # 列表中的第二个字典
                "title": "Exporting",  # 第二个字典的键 "title" 的值为 "Exporting"
                "autosummary": ["export_graphviz", "export_text"],  # 第二个字典的键 "autosummary" 的值为包含两个字符串的列表
            },
            {  # 列表中的第三个字典
                "title": "Plotting",  # 第三个字典的键 "title" 的值为 "Plotting"
                "autosummary": ["plot_tree"],  # 第三个字典的键 "autosummary" 的值为包含一个字符串的列表
            },
        ],
    },
# DEPRECATED_API_REFERENCE 是一个字典，用于记录已弃用的 API 相关信息
# 键是版本号，值是该版本中的已弃用 API 列表
DEPRECATED_API_REFERENCE = {
    # 版本号 "1.7" 对应的已弃用 API 列表
    "1.7": [
        # 已弃用 API: utils.parallel_backend
        "utils.parallel_backend",
        # 已弃用 API: utils.register_parallel_backend
        "utils.register_parallel_backend",
    ]
}  # type: ignore
```