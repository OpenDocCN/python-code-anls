# `D:\src\scipysrc\scikit-learn\sklearn\metrics\tests\test_common.py`

```
# 从 functools 模块导入 partial 函数，用于创建部分函数应用
# 从 inspect 模块导入 signature 函数，用于获取函数的签名信息
# 从 itertools 模块导入 chain、permutations、product 函数，用于迭代器操作

import numpy as np
# 导入 NumPy 库，用于数值计算

import pytest
# 导入 pytest 库，用于编写和运行测试用例

from sklearn._config import config_context
# 从 sklearn._config 模块导入 config_context 函数，用于配置 sklearn 库的全局参数

from sklearn.datasets import make_multilabel_classification
# 从 sklearn.datasets 模块导入 make_multilabel_classification 函数，用于生成多标签分类数据集

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    coverage_error,
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    dcg_score,
    det_curve,
    explained_variance_score,
    f1_score,
    fbeta_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_pinball_loss,
    mean_poisson_deviance,
    mean_squared_error,
    mean_tweedie_deviance,
    median_absolute_error,
    multilabel_confusion_matrix,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
    zero_one_loss,
)
# 从 sklearn.metrics 模块导入多个评估指标函数，用于评估分类和回归模型的性能指标

from sklearn.metrics._base import _average_binary_score
# 从 sklearn.metrics._base 模块导入 _average_binary_score 函数，用于二分类平均评分计算

from sklearn.metrics.pairwise import (
    additive_chi2_kernel,
    chi2_kernel,
    cosine_similarity,
    paired_cosine_distances,
)
# 从 sklearn.metrics.pairwise 模块导入多个核函数和距离函数，用于计算核矩阵和距离

from sklearn.preprocessing import LabelBinarizer
# 从 sklearn.preprocessing 模块导入 LabelBinarizer 类，用于标签二值化处理

from sklearn.utils import shuffle
# 从 sklearn.utils 模块导入 shuffle 函数，用于打乱数据顺序

from sklearn.utils._array_api import (
    _atol_for_type,
    _convert_to_numpy,
    yield_namespace_device_dtype_combinations,
)
# 从 sklearn.utils._array_api 模块导入多个数组处理函数，用于数组操作和类型转换

from sklearn.utils._testing import (
    _array_api_for_tests,
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_array_less,
    ignore_warnings,
)
# 从 sklearn.utils._testing 模块导入多个测试辅助函数，用于单元测试和断言验证

from sklearn.utils.fixes import COO_CONTAINERS
# 从 sklearn.utils.fixes 模块导入 COO_CONTAINERS 常量，用于稀疏矩阵数据的容器选择

from sklearn.utils.multiclass import type_of_target
# 从 sklearn.utils.multiclass 模块导入 type_of_target 函数，用于推断目标类型

from sklearn.utils.validation import _num_samples, check_random_state
# 从 sklearn.utils.validation 模块导入 _num_samples 和 check_random_state 函数，用于数据验证和随机状态检查

# 开发者关于度量测试的注意事项
# -------------------------------------------
# 常常可以为多个度量编写一个通用的测试：
#
#   - 不变性属性，例如样本顺序的不变性
#   - 具有相同参数的常见行为，例如 "normalize" 参数为 True 时返回度量的均值，为 False 时返回度量的总和
#
# 为了改进整体的度量测试，最好先为给定的度量编写特定的测试，然后再为所有具有相同行为的度量添加通用测试。
#
# 该系统使用两种数据结构实现：
# 度量字典和具有共同属性的度量列表。

# 度量字典
# ------------------------
# 使用这些字典的目的是为了方便调用特定的度量函数，并为每个函数关联一个名称：
#
#   - REGRESSION_METRICS: 所有回归度量函数。
#   - CLASSIFICATION_METRICS: 包含所有的分类指标，用于比较分类器返回的估计目标和实际目标之间的表现。
#   - THRESHOLDED_METRICS: 包含所有的分类指标，这些指标比较实际目标和分数，例如估计的概率或决策函数（格式可能有所不同）。

# 这些字典将被用来系统地测试某些不变性属性，例如对多种输入布局的不变性。

REGRESSION_METRICS = {
    "max_error": max_error,  # 最大误差指标
    "mean_absolute_error": mean_absolute_error,  # 平均绝对误差指标
    "mean_squared_error": mean_squared_error,  # 均方误差指标
    "mean_pinball_loss": mean_pinball_loss,  # 平均损失捏球损失指标
    "median_absolute_error": median_absolute_error,  # 中位数绝对误差指标
    "mean_absolute_percentage_error": mean_absolute_percentage_error,  # 平均百分比误差指标
    "explained_variance_score": explained_variance_score,  # 解释方差得分指标
    "r2_score": partial(r2_score, multioutput="variance_weighted"),  # R²得分指标（加权方差）
    "mean_normal_deviance": partial(mean_tweedie_deviance, power=0),  # 平均正态偏差指标
    "mean_poisson_deviance": mean_poisson_deviance,  # 平均泊松偏差指标
    "mean_gamma_deviance": mean_gamma_deviance,  # 平均伽马偏差指标
    "mean_compound_poisson_deviance": partial(mean_tweedie_deviance, power=1.4),  # 平均复合泊松偏差指标
    "d2_tweedie_score": partial(d2_tweedie_score, power=1.4),  # D² Tweedie得分指标
    "d2_pinball_score": d2_pinball_score,  # D² 捏球得分指标
    "d2_absolute_error_score": d2_absolute_error_score,  # D² 绝对误差得分指标
}

CLASSIFICATION_METRICS = {
    "accuracy_score": accuracy_score,  # 准确率指标
    "balanced_accuracy_score": balanced_accuracy_score,  # 平衡精度指标
    "adjusted_balanced_accuracy_score": partial(balanced_accuracy_score, adjusted=True),  # 调整后的平衡精度指标
    "unnormalized_accuracy_score": partial(accuracy_score, normalize=False),  # 未标准化的准确率指标
    # `confusion_matrix` 返回绝对值，因此表现为未标准化。使用未标准化前缀命名对于此模块来说是必要的，
    # 以跳过样本权重缩放检查，因为对于未标准化的指标，这些检查会失败。
    "unnormalized_confusion_matrix": confusion_matrix,  # 未标准化的混淆矩阵
    "normalized_confusion_matrix": lambda *args, **kwargs: (
        confusion_matrix(*args, **kwargs).astype("float")
        / confusion_matrix(*args, **kwargs).sum(axis=1)[:, np.newaxis]
    ),  # 标准化的混淆矩阵
    "unnormalized_multilabel_confusion_matrix": multilabel_confusion_matrix,  # 未标准化的多标签混淆矩阵
    "unnormalized_multilabel_confusion_matrix_sample": partial(
        multilabel_confusion_matrix, samplewise=True
    ),  # 样本方式的未标准化多标签混淆矩阵
    "hamming_loss": hamming_loss,  # 汉明损失指标
    "zero_one_loss": zero_one_loss,  # 0-1损失指标
    "unnormalized_zero_one_loss": partial(zero_one_loss, normalize=False),  # 未标准化的0-1损失指标
    # 这些指标用于测试平均化
    "jaccard_score": jaccard_score,  # Jaccard分数
    "precision_score": precision_score,  # 精确率指标
    "recall_score": recall_score,  # 召回率指标
    "f1_score": f1_score,  # F1分数
    "f2_score": partial(fbeta_score, beta=2),  # F2分数
    "f0.5_score": partial(fbeta_score, beta=0.5),  # F0.5分数
    "matthews_corrcoef_score": matthews_corrcoef,  # Matthews相关系数得分指标
    "weighted_f0.5_score": partial(fbeta_score, average="weighted", beta=0.5),  # 加权F0.5分数
    "weighted_f1_score": partial(f1_score, average="weighted"),  # 加权F1分数
    "weighted_f2_score": partial(fbeta_score, average="weighted", beta=2),  # 加权F2分数
}
    "weighted_precision_score": partial(precision_score, average="weighted"),
    # 创建一个偏函数，用于计算加权精确率（weighted precision）
    "weighted_recall_score": partial(recall_score, average="weighted"),
    # 创建一个偏函数，用于计算加权召回率（weighted recall）
    "weighted_jaccard_score": partial(jaccard_score, average="weighted"),
    # 创建一个偏函数，用于计算加权 Jaccard 系数（weighted Jaccard score）
    "micro_f0.5_score": partial(fbeta_score, average="micro", beta=0.5),
    # 创建一个偏函数，用于计算微平均 F-beta 分数（micro-average F-beta score）中的 F0.5 分数
    "micro_f1_score": partial(f1_score, average="micro"),
    # 创建一个偏函数，用于计算微平均 F1 分数（micro-average F1 score）
    "micro_f2_score": partial(fbeta_score, average="micro", beta=2),
    # 创建一个偏函数，用于计算微平均 F-beta 分数中的 F2 分数
    "micro_precision_score": partial(precision_score, average="micro"),
    # 创建一个偏函数，用于计算微平均精确率（micro-average precision）
    "micro_recall_score": partial(recall_score, average="micro"),
    # 创建一个偏函数，用于计算微平均召回率（micro-average recall）
    "micro_jaccard_score": partial(jaccard_score, average="micro"),
    # 创建一个偏函数，用于计算微平均 Jaccard 系数（micro-average Jaccard score）
    "macro_f0.5_score": partial(fbeta_score, average="macro", beta=0.5),
    # 创建一个偏函数，用于计算宏平均 F-beta 分数（macro-average F-beta score）中的 F0.5 分数
    "macro_f1_score": partial(f1_score, average="macro"),
    # 创建一个偏函数，用于计算宏平均 F1 分数（macro-average F1 score）
    "macro_f2_score": partial(fbeta_score, average="macro", beta=2),
    # 创建一个偏函数，用于计算宏平均 F-beta 分数中的 F2 分数
    "macro_precision_score": partial(precision_score, average="macro"),
    # 创建一个偏函数，用于计算宏平均精确率（macro-average precision）
    "macro_recall_score": partial(recall_score, average="macro"),
    # 创建一个偏函数，用于计算宏平均召回率（macro-average recall）
    "macro_jaccard_score": partial(jaccard_score, average="macro"),
    # 创建一个偏函数，用于计算宏平均 Jaccard 系数（macro-average Jaccard score）
    "samples_f0.5_score": partial(fbeta_score, average="samples", beta=0.5),
    # 创建一个偏函数，用于计算样本平均 F-beta 分数（samples-average F-beta score）中的 F0.5 分数
    "samples_f1_score": partial(f1_score, average="samples"),
    # 创建一个偏函数，用于计算样本平均 F1 分数（samples-average F1 score）
    "samples_f2_score": partial(fbeta_score, average="samples", beta=2),
    # 创建一个偏函数，用于计算样本平均 F-beta 分数中的 F2 分数
    "samples_precision_score": partial(precision_score, average="samples"),
    # 创建一个偏函数，用于计算样本平均精确率（samples-average precision）
    "samples_recall_score": partial(recall_score, average="samples"),
    # 创建一个偏函数，用于计算样本平均召回率（samples-average recall）
    "samples_jaccard_score": partial(jaccard_score, average="samples"),
    # 创建一个偏函数，用于计算样本平均 Jaccard 系数（samples-average Jaccard score）
    "cohen_kappa_score": cohen_kappa_score,
    # 使用 cohen_kappa_score 函数计算 Cohen's kappa 系数
}

# precision_recall_curve_padded_thresholds 函数定义，用于处理 precision-recall 曲线的维度问题和阈值数组的填充
def precision_recall_curve_padded_thresholds(*args, **kwargs):
    """
    precision_recall_curve 返回的精确度-召回率对和阈值数组的维度不匹配。
    参见 func:`sklearn.metrics.precision_recall_curve`

    这会阻止返回值三元组隐式转换为更高维度的 np.array，其 dtype 为('float64')（而会是 dtype('object')）。
    这会影响 assert_array_equal 的正确工作。

    作为一种解决方法，我们使用 NaN 值填充阈值数组，使其与精确度和召回率数组的维度相匹配。
    """
    # 调用 precision_recall_curve 函数，获取精确度、召回率和阈值数组
    precision, recall, thresholds = precision_recall_curve(*args, **kwargs)

    # 计算需要填充的 NaN 数量，以使阈值数组与精确度数组的长度相等
    pad_threshholds = len(precision) - len(thresholds)

    # 返回一个包含精确度、召回率和填充后阈值数组的 np.array
    return np.array(
        [
            precision,
            recall,
            np.pad(
                thresholds.astype(np.float64),  # 将阈值数组转换为 np.float64 类型
                pad_width=(0, pad_threshholds),  # 在阈值数组两端填充 NaN，使其长度达到精确度数组长度
                mode="constant",  # 填充模式为常数填充
                constant_values=[np.nan],  # 填充值为 NaN
            ),
        ]
    )


# CURVE_METRICS 字典，包含曲线相关的度量名称及其对应的函数引用
CURVE_METRICS = {
    "roc_curve": roc_curve,  # ROC 曲线函数
    "precision_recall_curve": precision_recall_curve_padded_thresholds,  # 修正后的精确度-召回率曲线函数
    "det_curve": det_curve,  # DET 曲线函数
}

# THRESHOLDED_METRICS 字典，包含阈值相关的度量名称及其对应的函数引用
THRESHOLDED_METRICS = {
    "coverage_error": coverage_error,
    "label_ranking_loss": label_ranking_loss,
    "log_loss": log_loss,
    "unnormalized_log_loss": partial(log_loss, normalize=False),
    "hinge_loss": hinge_loss,
    "brier_score_loss": brier_score_loss,
    "roc_auc_score": roc_auc_score,  # ROC AUC 曲线下面积函数，默认使用 'macro' 平均值
    "weighted_roc_auc": partial(roc_auc_score, average="weighted"),
    "samples_roc_auc": partial(roc_auc_score, average="samples"),
    "micro_roc_auc": partial(roc_auc_score, average="micro"),
    "ovr_roc_auc": partial(roc_auc_score, average="macro", multi_class="ovr"),
    "weighted_ovr_roc_auc": partial(
        roc_auc_score, average="weighted", multi_class="ovr"
    ),
    "ovo_roc_auc": partial(roc_auc_score, average="macro", multi_class="ovo"),
    "weighted_ovo_roc_auc": partial(
        roc_auc_score, average="weighted", multi_class="ovo"
    ),
    "partial_roc_auc": partial(roc_auc_score, max_fpr=0.5),
    "average_precision_score": average_precision_score,  # 平均精度（average_precision）得分函数，默认使用 'macro' 平均值
    "weighted_average_precision_score": partial(
        average_precision_score, average="weighted"
    ),
    "samples_average_precision_score": partial(
        average_precision_score, average="samples"
    ),
    "micro_average_precision_score": partial(average_precision_score, average="micro"),
    "label_ranking_average_precision_score": label_ranking_average_precision_score,
    "ndcg_score": ndcg_score,
    "dcg_score": dcg_score,
    "top_k_accuracy_score": top_k_accuracy_score,
}

# ALL_METRICS 字典，包含所有度量名称及其对应的函数引用，结合了阈值度量、分类度量和回归度量以及曲线度量
ALL_METRICS = dict()
ALL_METRICS.update(THRESHOLDED_METRICS)  # 添加阈值度量
ALL_METRICS.update(CLASSIFICATION_METRICS)  # 添加分类度量
ALL_METRICS.update(REGRESSION_METRICS)  # 添加回归度量
ALL_METRICS.update(CURVE_METRICS)  # 添加曲线度量

# Lists of metrics with common properties
# ---------------------------------------
# Lists of metrics with common properties are used to test systematically some
# functionalities and invariance, e.g. SYMMETRIC_METRICS lists all metrics that
# are symmetric with respect to their input argument y_true and y_pred.
#
# When you add a new metric or functionality, check if a general test
# is already written.

# Those metrics don't support binary inputs
METRIC_UNDEFINED_BINARY = {
    "samples_f0.5_score",
    "samples_f1_score",
    "samples_f2_score",
    "samples_precision_score",
    "samples_recall_score",
    "samples_jaccard_score",
    "coverage_error",
    "unnormalized_multilabel_confusion_matrix_sample",
    "label_ranking_loss",
    "label_ranking_average_precision_score",
    "dcg_score",
    "ndcg_score",
}

# Those metrics don't support multiclass inputs
METRIC_UNDEFINED_MULTICLASS = {
    "brier_score_loss",
    "micro_roc_auc",
    "samples_roc_auc",
    "partial_roc_auc",
    "roc_auc_score",
    "weighted_roc_auc",
    "jaccard_score",
    # with default average='binary', multiclass is prohibited
    "precision_score",
    "recall_score",
    "f1_score",
    "f2_score",
    "f0.5_score",
    # curves
    "roc_curve",
    "precision_recall_curve",
    "det_curve",
}

# Metric undefined with "binary" or "multiclass" input
METRIC_UNDEFINED_BINARY_MULTICLASS = METRIC_UNDEFINED_BINARY.union(
    METRIC_UNDEFINED_MULTICLASS
)

# Metrics with an "average" argument
METRICS_WITH_AVERAGING = {
    "precision_score",
    "recall_score",
    "f1_score",
    "f2_score",
    "f0.5_score",
    "jaccard_score",
}

# Threshold-based metrics with an "average" argument
THRESHOLDED_METRICS_WITH_AVERAGING = {
    "roc_auc_score",
    "average_precision_score",
    "partial_roc_auc",
}

# Metrics with a "pos_label" argument
METRICS_WITH_POS_LABEL = {
    "roc_curve",
    "precision_recall_curve",
    "det_curve",
    "brier_score_loss",
    "precision_score",
    "recall_score",
    "f1_score",
    "f2_score",
    "f0.5_score",
    "jaccard_score",
    "average_precision_score",
    "weighted_average_precision_score",
    "micro_average_precision_score",
    "samples_average_precision_score",
}

# Metrics with a "labels" argument
# TODO: Handle multi_class metrics that has a labels argument as well as a
# decision function argument. e.g hinge_loss
METRICS_WITH_LABELS = {
    "unnormalized_confusion_matrix",
    "normalized_confusion_matrix",
    "roc_curve",
    "precision_recall_curve",
    "det_curve",
    "precision_score",
    "recall_score",
    "f1_score",
    "f2_score",
    "f0.5_score",
    "jaccard_score",
    "weighted_f0.5_score",
    "weighted_f1_score",
    "weighted_f2_score",
    "weighted_precision_score",
    "weighted_recall_score",
    "weighted_jaccard_score",
    "micro_f0.5_score",
    "micro_f1_score",
    "micro_f2_score",
    "micro_precision_score",
    "micro_recall_score",
    "micro_jaccard_score",
    "macro_f0.5_score",
    "macro_f1_score",
}
    # 这些字符串代表不同的评估指标名称，用于多标签分类任务的评估
    "macro_f2_score",
    "macro_precision_score",
    "macro_recall_score",
    "macro_jaccard_score",
    "unnormalized_multilabel_confusion_matrix",
    "unnormalized_multilabel_confusion_matrix_sample",
    "cohen_kappa_score",
# Metrics with a "normalize" option
# 具有“normalize”选项的指标集合
METRICS_WITH_NORMALIZE_OPTION = {
    "accuracy_score",
    "top_k_accuracy_score",
    "zero_one_loss",
}

# Threshold-based metrics with "multilabel-indicator" format support
# 支持“multilabel-indicator”格式的基于阈值的指标集合
THRESHOLDED_MULTILABEL_METRICS = {
    "log_loss",
    "unnormalized_log_loss",
    "roc_auc_score",
    "weighted_roc_auc",
    "samples_roc_auc",
    "micro_roc_auc",
    "partial_roc_auc",
    "average_precision_score",
    "weighted_average_precision_score",
    "samples_average_precision_score",
    "micro_average_precision_score",
    "coverage_error",
    "label_ranking_loss",
    "ndcg_score",
    "dcg_score",
    "label_ranking_average_precision_score",
}

# Classification metrics with "multilabel-indicator" format
# 使用“multilabel-indicator”格式的分类指标集合
MULTILABELS_METRICS = {
    "accuracy_score",
    "unnormalized_accuracy_score",
    "hamming_loss",
    "zero_one_loss",
    "unnormalized_zero_one_loss",
    "weighted_f0.5_score",
    "weighted_f1_score",
    "weighted_f2_score",
    "weighted_precision_score",
    "weighted_recall_score",
    "weighted_jaccard_score",
    "macro_f0.5_score",
    "macro_f1_score",
    "macro_f2_score",
    "macro_precision_score",
    "macro_recall_score",
    "macro_jaccard_score",
    "micro_f0.5_score",
    "micro_f1_score",
    "micro_f2_score",
    "micro_precision_score",
    "micro_recall_score",
    "micro_jaccard_score",
    "unnormalized_multilabel_confusion_matrix",
    "samples_f0.5_score",
    "samples_f1_score",
    "samples_f2_score",
    "samples_precision_score",
    "samples_recall_score",
    "samples_jaccard_score",
}

# Regression metrics with "multioutput-continuous" format support
# 支持“multioutput-continuous”格式的回归指标集合
MULTIOUTPUT_METRICS = {
    "mean_absolute_error",
    "median_absolute_error",
    "mean_squared_error",
    "r2_score",
    "explained_variance_score",
    "mean_absolute_percentage_error",
    "mean_pinball_loss",
    "d2_pinball_score",
    "d2_absolute_error_score",
}

# Symmetric with respect to their input arguments y_true and y_pred
# metric(y_true, y_pred) == metric(y_pred, y_true).
# 关于其输入参数y_true和y_pred对称的指标集合，即metric(y_true, y_pred) == metric(y_pred, y_true)
SYMMETRIC_METRICS = {
    "accuracy_score",
    "unnormalized_accuracy_score",
    "hamming_loss",
    "zero_one_loss",
    "unnormalized_zero_one_loss",
    "micro_jaccard_score",
    "macro_jaccard_score",
    "jaccard_score",
    "samples_jaccard_score",
    "f1_score",
    "micro_f1_score",
    "macro_f1_score",
    "weighted_recall_score",
    # P = R = F = accuracy in multiclass case
    # 多类别情况下，P = R = F = accuracy
    "micro_f0.5_score",
    "micro_f1_score",
    "micro_f2_score",
    "micro_precision_score",
    "micro_recall_score",
    "matthews_corrcoef_score",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
    "max_error",
    # Pinball loss is only symmetric for alpha=0.5 which is the default.
    # Pinball损失仅在alpha=0.5（默认值）时对称。
    "mean_pinball_loss",
    "cohen_kappa_score",
    "mean_normal_deviance",
}

# Asymmetric with respect to their input arguments y_true and y_pred
# metric(y_true, y_pred) != metric(y_pred, y_true).
# 关于其输入参数y_true和y_pred不对称的指标集合，即metric(y_true, y_pred) != metric(y_pred, y_true)
NOT_SYMMETRIC_METRICS = {
    # 定义一个包含多种评估指标名称的元组或列表
    (
        "balanced_accuracy_score",                       # 平衡精度评分
        "adjusted_balanced_accuracy_score",              # 调整后的平衡精度评分
        "explained_variance_score",                      # 解释方差分数
        "r2_score",                                      # R^2（确定系数）评分
        "unnormalized_confusion_matrix",                 # 未归一化混淆矩阵
        "normalized_confusion_matrix",                   # 归一化混淆矩阵
        "roc_curve",                                     # ROC 曲线
        "precision_recall_curve",                        # 精确率-召回率曲线
        "det_curve",                                     # DET 曲线
        "precision_score",                               # 精确率评分
        "recall_score",                                  # 召回率评分
        "f2_score",                                      # F2 分数
        "f0.5_score",                                    # F0.5 分数
        "weighted_f0.5_score",                           # 加权 F0.5 分数
        "weighted_f1_score",                             # 加权 F1 分数
        "weighted_f2_score",                             # 加权 F2 分数
        "weighted_precision_score",                      # 加权精确率评分
        "weighted_jaccard_score",                        # 加权 Jaccard 分数
        "unnormalized_multilabel_confusion_matrix",       # 未归一化多标签混淆矩阵
        "macro_f0.5_score",                              # 宏平均 F0.5 分数
        "macro_f2_score",                                # 宏平均 F2 分数
        "macro_precision_score",                         # 宏平均精确率评分
        "macro_recall_score",                            # 宏平均召回率评分
        "hinge_loss",                                    # Hinge 损失
        "mean_gamma_deviance",                           # 平均 Gamma 偏差
        "mean_poisson_deviance",                         # 平均 Poisson 偏差
        "mean_compound_poisson_deviance",                # 平均复合 Poisson 偏差
        "d2_tweedie_score",                              # D2 Tweedie 分数
        "d2_pinball_score",                              # D2 Pinball 分数
        "d2_absolute_error_score",                       # D2 绝对误差分数
        "mean_absolute_percentage_error",                # 平均绝对百分比误差
    )
# No Sample weight support
# 没有样本权重支持的度量指标集合
METRICS_WITHOUT_SAMPLE_WEIGHT = {
    "median_absolute_error",
    "max_error",
    "ovo_roc_auc",
    "weighted_ovo_roc_auc",
}

# 要求目标变量为正的度量指标集合
METRICS_REQUIRE_POSITIVE_Y = {
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "mean_compound_poisson_deviance",
    "d2_tweedie_score",
}

# 函数用于使目标变量严格为正数
def _require_positive_targets(y1, y2):
    """Make targets strictly positive"""
    # 计算使得目标变量严格为正的偏移量
    offset = abs(min(y1.min(), y2.min())) + 1
    # 将目标变量加上偏移量
    y1 += offset
    y2 += offset
    return y1, y2


# 测试函数，用于验证度量指标的对称性和一致性
def test_symmetry_consistency():
    # 验证所有度量指标的完整性，确保没有遗漏
    assert (
        SYMMETRIC_METRICS
        | NOT_SYMMETRIC_METRICS
        | set(THRESHOLDED_METRICS)
        | METRIC_UNDEFINED_BINARY_MULTICLASS
    ) == set(ALL_METRICS)

    # 验证对称度量和非对称度量集合的交集为空集
    assert (SYMMETRIC_METRICS & NOT_SYMMETRIC_METRICS) == set()


# 使用参数化测试对所有对称度量进行测试
@pytest.mark.parametrize("name", sorted(SYMMETRIC_METRICS))
def test_symmetric_metric(name):
    # 测试评分和损失函数的对称性
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(20,))
    y_pred = random_state.randint(0, 2, size=(20,))

    # 如果度量指标要求目标变量为正，则调用函数进行调整
    if name in METRICS_REQUIRE_POSITIVE_Y:
        y_true, y_pred = _require_positive_targets(y_true, y_pred)

    y_true_bin = random_state.randint(0, 2, size=(20, 25))
    y_pred_bin = random_state.randint(0, 2, size=(20, 25))

    # 获取当前度量指标的函数对象
    metric = ALL_METRICS[name]
    if name in METRIC_UNDEFINED_BINARY:
        # 如果度量指标未定义为二进制多类别，则进行特殊处理
        if name in MULTILABELS_METRICS:
            # 对于多标签度量指标，验证其对称性
            assert_allclose(
                metric(y_true_bin, y_pred_bin),
                metric(y_pred_bin, y_true_bin),
                err_msg="%s is not symmetric" % name,
            )
        else:
            # 对于未处理的情况，抛出断言错误
            assert False, "This case is currently unhandled"
    else:
        # 验证度量指标的对称性
        assert_allclose(
            metric(y_true, y_pred),
            metric(y_pred, y_true),
            err_msg="%s is not symmetric" % name,
        )


# 使用参数化测试对所有非对称度量进行测试
@pytest.mark.parametrize("name", sorted(NOT_SYMMETRIC_METRICS))
def test_not_symmetric_metric(name):
    # 测试评分和损失函数的对称性
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(20,))
    y_pred = random_state.randint(0, 2, size=(20,))

    # 如果度量指标要求目标变量为正，则调用函数进行调整
    if name in METRICS_REQUIRE_POSITIVE_Y:
        y_true, y_pred = _require_positive_targets(y_true, y_pred)

    # 获取当前度量指标的函数对象
    metric = ALL_METRICS[name]

    # 使用上下文管理器提供自定义错误消息
    with pytest.raises(AssertionError):
        # 验证度量指标的非对称性
        assert_array_equal(metric(y_true, y_pred), metric(y_pred, y_true))
        raise ValueError("%s seems to be symmetric" % name)


# 使用参数化测试对除了二进制多类别未定义的所有度量进行测试
@pytest.mark.parametrize(
    "name", sorted(set(ALL_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS)
)
def test_sample_order_invariance(name):
    # 生成随机数种子
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(20,))
    y_pred = random_state.randint(0, 2, size=(20,))

    # 如果度量指标要求目标变量为正，则调用函数进行调整
    if name in METRICS_REQUIRE_POSITIVE_Y:
        y_true, y_pred = _require_positive_targets(y_true, y_pred)
    # 使用 shuffle 函数对 y_true 和 y_pred 进行洗牌，使用 random_state=0 确保结果可重复
    y_true_shuffle, y_pred_shuffle = shuffle(y_true, y_pred, random_state=0)

    # 使用 ignore_warnings 上下文管理器来忽略警告
    with ignore_warnings():
        # 从 ALL_METRICS 字典中获取特定名称（name）对应的评估指标函数
        metric = ALL_METRICS[name]

        # 断言：对于给定的评估指标，经过洗牌后的预测结果应该和原始预测结果在该指标下的计算结果非常接近
        assert_allclose(
            metric(y_true, y_pred),  # 计算原始数据下的评估指标值
            metric(y_true_shuffle, y_pred_shuffle),  # 计算洗牌后数据下的评估指标值
            err_msg="%s is not sample order invariant" % name,  # 如果不满足近似条件，输出错误信息
        )
# 用装饰器忽略警告，将下面的函数标记为忽略警告的函数
@ignore_warnings
# 测试函数，验证多标签和多输出问题中的样本顺序不变性
def test_sample_order_invariance_multilabel_and_multioutput():
    # 使用随机状态生成随机数生成器
    random_state = check_random_state(0)

    # 生成一些数据
    y_true = random_state.randint(0, 2, size=(20, 25))  # 生成随机的真实标签
    y_pred = random_state.randint(0, 2, size=(20, 25))  # 生成随机的预测标签
    y_score = random_state.uniform(size=y_true.shape)  # 生成随机的得分

    # 有些指标（例如 log_loss）要求 y_score 是概率（总和为1）
    y_score /= y_score.sum(axis=1, keepdims=True)

    # 对真实标签、预测标签和得分进行随机排列
    y_true_shuffle, y_pred_shuffle, y_score_shuffle = shuffle(
        y_true, y_pred, y_score, random_state=0
    )

    # 对多标签指标进行循环测试
    for name in MULTILABELS_METRICS:
        metric = ALL_METRICS[name]
        # 验证样本顺序不变性
        assert_allclose(
            metric(y_true, y_pred),
            metric(y_true_shuffle, y_pred_shuffle),
            err_msg="%s is not sample order invariant" % name,
        )

    # 对阈值多标签指标进行循环测试
    for name in THRESHOLDED_MULTILABEL_METRICS:
        metric = ALL_METRICS[name]
        # 验证样本顺序不变性
        assert_allclose(
            metric(y_true, y_score),
            metric(y_true_shuffle, y_score_shuffle),
            err_msg="%s is not sample order invariant" % name,
        )

    # 对多输出指标进行循环测试
    for name in MULTIOUTPUT_METRICS:
        metric = ALL_METRICS[name]
        # 验证样本顺序不变性
        assert_allclose(
            metric(y_true, y_score),
            metric(y_true_shuffle, y_score_shuffle),
            err_msg="%s is not sample order invariant" % name,
        )
        # 再次验证样本顺序不变性，这次针对真实标签和预测标签
        assert_allclose(
            metric(y_true, y_pred),
            metric(y_true_shuffle, y_pred_shuffle),
            err_msg="%s is not sample order invariant" % name,
        )


# 使用参数化测试标记，测试与二元多类别未定义指标不相交的所有指标
@pytest.mark.parametrize(
    "name", sorted(set(ALL_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS)
)
def test_format_invariance_with_1d_vectors(name):
    # 使用随机状态生成随机数生成器
    random_state = check_random_state(0)
    # 生成两个长度为20的随机一维向量
    y1 = random_state.randint(0, 2, size=(20,))
    y2 = random_state.randint(0, 2, size=(20,))

    # 如果指标要求正向目标，则调用 _require_positive_targets 函数
    if name in METRICS_REQUIRE_POSITIVE_Y:
        y1, y2 = _require_positive_targets(y1, y2)

    # 将向量转换为列表形式
    y1_list = list(y1)
    y2_list = list(y2)

    # 将向量转换为 numpy 数组形式
    y1_1d, y2_1d = np.array(y1), np.array(y2)
    # 验证数组维度为1
    assert_array_equal(y1_1d.ndim, 1)
    assert_array_equal(y2_1d.ndim, 1)
    
    # 将一维数组转换为列向量和行向量
    y1_column = np.reshape(y1_1d, (-1, 1))
    y2_column = np.reshape(y2_1d, (-1, 1))
    y1_row = np.reshape(y1_1d, (1, -1))
    y2_row = np.reshape(y2_1d, (1, -1))
    # 使用上下文管理器 ignore_warnings() 来忽略警告
    with ignore_warnings():
        # 从 ALL_METRICS 字典中获取指定名称 name 对应的度量标准对象
        metric = ALL_METRICS[name]

        # 使用 metric 对象计算 y1 和 y2 之间的度量值，并赋给 measure 变量
        measure = metric(y1, y2)

        # 断言 metric 度量 y1_list 和 y2_list 之间的度量值与之前计算的 measure 相等
        assert_allclose(
            metric(y1_list, y2_list),
            measure,
            err_msg="%s is not representation invariant with list" % name,
        )

        # 断言 metric 度量 y1_1d 和 y2_1d 之间的度量值与之前计算的 measure 相等
        assert_allclose(
            metric(y1_1d, y2_1d),
            measure,
            err_msg="%s is not representation invariant with np-array-1d" % name,
        )

        # 断言 metric 度量 y1_column 和 y2_column 之间的度量值与之前计算的 measure 相等
        assert_allclose(
            metric(y1_column, y2_column),
            measure,
            err_msg="%s is not representation invariant with np-array-column" % name,
        )

        # 断言 metric 度量 y1_1d 和 y2_list 之间的度量值与之前计算的 measure 相等
        assert_allclose(
            metric(y1_1d, y2_list),
            measure,
            err_msg="%s is not representation invariant with mix np-array-1d and list"
            % name,
        )

        # 断言 metric 度量 y1_list 和 y2_1d 之间的度量值与之前计算的 measure 相等
        assert_allclose(
            metric(y1_list, y2_1d),
            measure,
            err_msg="%s is not representation invariant with mix np-array-1d and list"
            % name,
        )

        # 断言 metric 度量 y1_1d 和 y2_column 之间的度量值与之前计算的 measure 相等
        assert_allclose(
            metric(y1_1d, y2_column),
            measure,
            err_msg=(
                "%s is not representation invariant with mix "
                "np-array-1d and np-array-column"
            )
            % name,
        )

        # 断言 metric 度量 y1_column 和 y2_1d 之间的度量值与之前计算的 measure 相等
        assert_allclose(
            metric(y1_column, y2_1d),
            measure,
            err_msg=(
                "%s is not representation invariant with mix "
                "np-array-1d and np-array-column"
            )
            % name,
        )

        # 断言 metric 度量 y1_list 和 y2_column 之间的度量值与之前计算的 measure 相等
        assert_allclose(
            metric(y1_list, y2_column),
            measure,
            err_msg=(
                "%s is not representation invariant with mix list and np-array-column"
            )
            % name,
        )

        # 断言 metric 度量 y1_column 和 y2_list 之间的度量值与之前计算的 measure 相等
        assert_allclose(
            metric(y1_column, y2_list),
            measure,
            err_msg=(
                "%s is not representation invariant with mix list and np-array-column"
            )
            % name,
        )

        # 这些混合表示方式不被允许，会引发 ValueError 异常
        with pytest.raises(ValueError):
            metric(y1_1d, y2_row)
        with pytest.raises(ValueError):
            metric(y1_row, y2_1d)
        with pytest.raises(ValueError):
            metric(y1_list, y2_row)
        with pytest.raises(ValueError):
            metric(y1_row, y2_list)
        with pytest.raises(ValueError):
            metric(y1_column, y2_row)
        with pytest.raises(ValueError):
            metric(y1_row, y2_column)

        # 注意事项：我们不测试 y1_row 和 y2_row，因为这些可能被解释为多标签或多输出数据。
        if name not in (
            MULTIOUTPUT_METRICS | THRESHOLDED_MULTILABEL_METRICS | MULTILABELS_METRICS
        ):
            # 断言 metric 度量 y1_row 和 y2_row 之间的度量值会引发 ValueError 异常
            with pytest.raises(ValueError):
                metric(y1_row, y2_row)
# 使用 pytest 框架的 parametrize 装饰器，对 test_classification_invariance_string_vs_numbers_labels 函数进行参数化测试
@pytest.mark.parametrize(
    "name", sorted(set(CLASSIFICATION_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS)
)
def test_classification_invariance_string_vs_numbers_labels(name):
    # 确保具有字符串标签的分类指标是不变的
    random_state = check_random_state(0)
    # 生成两个长度为 20 的随机整数数组，作为分类标签
    y1 = random_state.randint(0, 2, size=(20,))
    y2 = random_state.randint(0, 2, size=(20,))

    # 使用字符串数组作为映射，创建字符串标签的分类结果
    y1_str = np.array(["eggs", "spam"])[y1]
    y2_str = np.array(["eggs", "spam"])[y2]

    # 正类标签为 "spam"，定义标签集合为 ["eggs", "spam"]
    pos_label_str = "spam"
    labels_str = ["eggs", "spam"]

    # 忽略警告上下文
    with ignore_warnings():
        # 获取当前分类指标的函数对象
        metric = CLASSIFICATION_METRICS[name]
        # 使用数值标签计算分类指标
        measure_with_number = metric(y1, y2)

        # 处理带有正类标签和标签的情况
        metric_str = metric
        if name in METRICS_WITH_POS_LABEL:
            # 如果分类指标需要正类标签，使用字符串形式的正类标签重新定义函数对象
            metric_str = partial(metric_str, pos_label=pos_label_str)

        # 使用字符串标签计算分类指标
        measure_with_str = metric_str(y1_str, y2_str)

        # 断言数值标签和字符串标签的分类结果应当相等
        assert_array_equal(
            measure_with_number,
            measure_with_str,
            err_msg="{0} failed string vs number invariance test".format(name),
        )

        # 使用字符串对象数组计算分类指标
        measure_with_strobj = metric_str(y1_str.astype("O"), y2_str.astype("O"))
        # 断言数值标签和字符串对象数组的分类结果应当相等
        assert_array_equal(
            measure_with_number,
            measure_with_strobj,
            err_msg="{0} failed string object vs number invariance test".format(name),
        )

        # 如果分类指标需要标签集合，使用字符串形式的标签集合重新定义函数对象
        if name in METRICS_WITH_LABELS:
            metric_str = partial(metric_str, labels=labels_str)
            # 使用字符串标签计算分类指标
            measure_with_str = metric_str(y1_str, y2_str)
            # 断言数值标签和字符串标签的分类结果应当相等
            assert_array_equal(
                measure_with_number,
                measure_with_str,
                err_msg="{0} failed string vs number  invariance test".format(name),
            )

            # 使用字符串对象数组计算分类指标
            measure_with_strobj = metric_str(y1_str.astype("O"), y2_str.astype("O"))
            # 断言数值标签和字符串对象数组的分类结果应当相等
            assert_array_equal(
                measure_with_number,
                measure_with_strobj,
                err_msg="{0} failed string vs number  invariance test".format(name),
            )


# 使用 pytest 框架的 parametrize 装饰器，对 test_thresholded_invariance_string_vs_numbers_labels 函数进行参数化测试
@pytest.mark.parametrize("name", THRESHOLDED_METRICS)
def test_thresholded_invariance_string_vs_numbers_labels(name):
    # 确保具有字符串标签的阈值指标是不变的
    random_state = check_random_state(0)
    # 生成两个长度为 20 的随机整数数组，作为分类标签
    y1 = random_state.randint(0, 2, size=(20,))
    y2 = random_state.randint(0, 2, size=(20,))

    # 使用字符串数组作为映射，创建字符串标签的分类结果
    y1_str = np.array(["eggs", "spam"])[y1]

    # 正类标签为 "spam"
    pos_label_str = "spam"
    # 使用 ignore_warnings 上下文管理器来忽略警告
    with ignore_warnings():
        # 从 THRESHOLDED_METRICS 字典中获取指标函数
        metric = THRESHOLDED_METRICS[name]
        
        # 如果指标名不在 METRIC_UNDEFINED_BINARY 中
        if name not in METRIC_UNDEFINED_BINARY:
            # 处理具有 pos_label 和 label 的情况
            # 复制指标函数以备后用
            metric_str = metric
            
            # 如果指标名在 METRICS_WITH_POS_LABEL 中
            if name in METRICS_WITH_POS_LABEL:
                # 部分应用指标函数，设置 pos_label 参数为 pos_label_str
                metric_str = partial(metric_str, pos_label=pos_label_str)
            
            # 使用数值数组 y1 和 y2 计算指标值
            measure_with_number = metric(y1, y2)
            # 使用字符串数组 y1_str 和 y2 计算指标值
            measure_with_str = metric_str(y1_str, y2)
            
            # 断言数值计算结果与字符串计算结果相等
            assert_array_equal(
                measure_with_number,
                measure_with_str,
                err_msg="{0} failed string vs number invariance test".format(name),
            )
            
            # 使用字符串对象数组 y1_str.astype("O") 和 y2 计算指标值
            measure_with_strobj = metric_str(y1_str.astype("O"), y2)
            
            # 断言数值计算结果与字符串对象计算结果相等
            assert_array_equal(
                measure_with_number,
                measure_with_strobj,
                err_msg="{0} failed string object vs number invariance test".format(
                    name
                ),
            )
        else:
            # TODO: 这些指标目前不支持字符串标签
            # 使用 pytest.raises 检测 ValueError 异常
            with pytest.raises(ValueError):
                metric(y1_str, y2)
            with pytest.raises(ValueError):
                metric(y1_str.astype("O"), y2)
# 定义包含无效值情况的列表，每个元素包含一个 y_true 和一个 y_score 列表
invalids_nan_inf = [
    ([0, 1], [np.inf, np.inf]),  # y_true=[0, 1], y_score=[inf, inf]
    ([0, 1], [np.nan, np.nan]),  # y_true=[0, 1], y_score=[nan, nan]
    ([0, 1], [np.nan, np.inf]),  # y_true=[0, 1], y_score=[nan, inf]
    ([0, 1], [np.inf, 1]),       # y_true=[0, 1], y_score=[inf, 1]
    ([0, 1], [np.nan, 1]),       # y_true=[0, 1], y_score=[nan, 1]
]

# 使用 pytest 的 parametrize 装饰器，对 metric 进行参数化，参数为 THRESHOLDED_METRICS 和 REGRESSION_METRICS 中的所有值
# 并对 y_true 和 y_score 进行参数化，参数为 invalids_nan_inf 列表中的所有值
@pytest.mark.parametrize(
    "metric", chain(THRESHOLDED_METRICS.values(), REGRESSION_METRICS.values())
)
@pytest.mark.parametrize("y_true, y_score", invalids_nan_inf)
def test_regression_thresholded_inf_nan_input(metric, y_true, y_score):
    # 如果 metric 是 coverage_error，则需要将 y_true 和 y_score 转换成列表，因为 coverage_error 只接受二维数组作为输入
    if metric == coverage_error:
        y_true = [y_true]
        y_score = [y_score]
    # 使用 pytest.raises 检查是否抛出 ValueError 异常，异常消息中应包含 "contains (NaN|infinity)"
    with pytest.raises(ValueError, match=r"contains (NaN|infinity)"):
        metric(y_true, y_score)

# 使用 pytest 的 parametrize 装饰器，对 metric 进行参数化，参数为 CLASSIFICATION_METRICS 中的所有值
# 并对 y_true 和 y_score 进行参数化，参数为 invalids_nan_inf 列表中的所有值和额外的分类相关的无效输入情况
@pytest.mark.parametrize("metric", CLASSIFICATION_METRICS.values())
@pytest.mark.parametrize(
    "y_true, y_score",
    invalids_nan_inf +
    [
        ([np.nan, 1, 2], [1, 2, 3]),  # y_true=[nan, 1, 2], y_score=[1, 2, 3]
        ([np.inf, 1, 2], [1, 2, 3]),  # y_true=[inf, 1, 2], y_score=[1, 2, 3]
    ],
)
def test_classification_inf_nan_input(metric, y_true, y_score):
    """检查分类指标是否引发包含非有限值的警告信息。"""
    # 根据 y_true 是否包含非有限值，确定输入名称和异常消息中的非期望值
    if not np.isfinite(y_true).all():
        input_name = "y_true"
        if np.isnan(y_true).any():
            unexpected_value = "NaN"
        else:
            unexpected_value = "infinity or a value too large"
    else:
        input_name = "y_pred"
        if np.isnan(y_score).any():
            unexpected_value = "NaN"
        else:
            unexpected_value = "infinity or a value too large"

    err_msg = f"Input {input_name} contains {unexpected_value}"

    # 使用 pytest.raises 检查是否抛出 ValueError 异常，异常消息应匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        metric(y_true, y_score)

# 使用 pytest 的 parametrize 装饰器，对 metric 进行参数化，参数为 CLASSIFICATION_METRICS 中的所有值
def test_classification_binary_continuous_input(metric):
    """检查分类指标是否引发包含二进制和连续目标向量混合类型数据的异常信息。"""
    y_true, y_score = ["a", "b", "a"], [0.1, 0.2, 0.3]
    err_msg = (
        "Classification metrics can't handle a mix of binary and continuous targets"
    )
    # 使用 pytest.raises 检查是否抛出 ValueError 异常，异常消息应匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        metric(y_true, y_score)

# 装饰器 ignore_warnings 标记的函数，测试单个样本的评分是否正常运行
def check_single_sample(name):
    """非回归测试：分数应该与单个样本一起工作。这对于留一交叉验证非常重要。"""
    metric = ALL_METRICS[name]

    # 断言不应抛出异常
    if name in METRICS_REQUIRE_POSITIVE_Y:
        values = [1, 2]
    else:
        values = [0, 1]
    for i, j in product(values, repeat=2):
        metric([i], [j])

# 装饰器 ignore_warnings 标记的函数，测试多输出的单个样本是否正常运行
def check_single_sample_multioutput(name):
    """测试多输出的单个样本是否正常运行。"""
    metric = ALL_METRICS[name]
    # 使用 itertools.product 函数生成四个元素的笛卡尔积，元素来自 [0, 1]，即生成所有可能的二进制组合
    for i, j, k, l in product([0, 1], repeat=4):
        # 调用 metric 函数，计算两个二维数组的某种度量，每次传入不同的组合 (i, j) 和 (k, l)
        metric(np.array([[i, j]]), np.array([[k, l]]))
@pytest.mark.parametrize(
    "name",
    sorted(
        set(ALL_METRICS)
        # 过滤掉在单样本或多类别分类中未定义的指标
        - METRIC_UNDEFINED_BINARY_MULTICLASS
        - set(THRESHOLDED_METRICS)
    ),
)
def test_single_sample(name):
    # 调用单样本检查函数，传入指标名称作为参数
    check_single_sample(name)


@pytest.mark.parametrize("name", sorted(MULTIOUTPUT_METRICS | MULTILABELS_METRICS))
def test_single_sample_multioutput(name):
    # 调用多输出单样本检查函数，传入指标名称作为参数
    check_single_sample_multioutput(name)


@pytest.mark.parametrize("name", sorted(MULTIOUTPUT_METRICS))
def test_multioutput_number_of_output_differ(name):
    # 创建包含不同输出数量的真实值和预测值
    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0], [1, 0], [0, 0]])

    # 获取指定指标的计算函数
    metric = ALL_METRICS[name]
    # 确保引发 ValueError 异常
    with pytest.raises(ValueError):
        metric(y_true, y_pred)


@pytest.mark.parametrize("name", sorted(MULTIOUTPUT_METRICS))
def test_multioutput_regression_invariance_to_dimension_shuffling(name):
    # 测试对维度重排列的不变性
    random_state = check_random_state(0)
    # 使用随机状态生成真实值和预测值
    y_true = random_state.uniform(0, 2, size=(20, 5))
    y_pred = random_state.uniform(0, 2, size=(20, 5))

    # 获取指定指标的计算函数
    metric = ALL_METRICS[name]
    # 计算原始情况下的误差
    error = metric(y_true, y_pred)

    # 针对三种不同维度的排列情况进行测试
    for _ in range(3):
        perm = random_state.permutation(y_true.shape[1])
        assert_allclose(
            metric(y_true[:, perm], y_pred[:, perm]),
            error,
            err_msg="%s is not dimension shuffling invariant" % (name),
        )


@ignore_warnings
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_multilabel_representation_invariance(coo_container):
    # 生成一些数据
    n_classes = 4
    n_samples = 50

    # 使用随机状态生成多标签分类数据
    _, y1 = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=0,
        n_samples=n_samples,
        allow_unlabeled=True,
    )
    _, y2 = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=1,
        n_samples=n_samples,
        allow_unlabeled=True,
    )

    # 确保至少存在一个空标签
    y1 = np.vstack([y1, [[0] * n_classes]])
    y2 = np.vstack([y2, [[0] * n_classes]])

    # 使用稀疏容器创建稀疏表示
    y1_sparse_indicator = coo_container(y1)
    y2_sparse_indicator = coo_container(y2)

    # 将数组列表转换为列表数组表示
    y1_list_array_indicator = list(y1)
    y2_list_array_indicator = list(y2)

    # 将列表数组转换为列表列表表示
    y1_list_list_indicator = [list(a) for a in y1_list_array_indicator]
    y2_list_list_indicator = [list(a) for a in y2_list_array_indicator]
    # 遍历 MULTILABELS_METRICS 列表中的每个指标名称
    for name in MULTILABELS_METRICS:
        # 获取指标名称对应的度量函数对象
        metric = ALL_METRICS[name]

        # 对于偏函数进行一个恶意的修改以便处理
        # XXX: 这是一个针对部分函数的粗暴修改方法
        if isinstance(metric, partial):
            # 修改偏函数的模块名为 "tmp"
            metric.__module__ = "tmp"
            # 修改偏函数的名称为当前处理的指标名称
            metric.__name__ = name

        # 使用当前指标对 y1 和 y2 进行度量
        measure = metric(y1, y2)

        # 检查表示不变性
        # 检验稀疏指示符格式和密集指示符格式之间的表示不变性
        assert_allclose(
            metric(y1_sparse_indicator, y2_sparse_indicator),
            measure,
            err_msg=(
                "%s failed representation invariance between "
                "dense and sparse indicator formats."
            )
            % name,
        )
        # 检验列表列表指示符格式和密集数组格式之间的表示不变性
        assert_almost_equal(
            metric(y1_list_list_indicator, y2_list_list_indicator),
            measure,
            err_msg=(
                "%s failed representation invariance  "
                "between dense array and list of list "
                "indicator formats."
            )
            % name,
        )
        # 检验列表数组指示符格式和密集数组格式之间的表示不变性
        assert_almost_equal(
            metric(y1_list_array_indicator, y2_list_array_indicator),
            measure,
            err_msg=(
                "%s failed representation invariance  "
                "between dense and list of array "
                "indicator formats."
            )
            % name,
        )
# 对每个名字按照字母顺序排序，使用参数化测试，逐个执行测试函数
@pytest.mark.parametrize("name", sorted(MULTILABELS_METRICS))
def test_raise_value_error_multilabel_sequences(name):
    # 确保多标签序列格式引发 ValueError 异常
    multilabel_sequences = [
        [[1], [2], [0, 1]],  # 包含非空标签的列表
        [(), (2), (0, 1)],    # 包含非空标签的元组
        [[]],                 # 空列表
        [()],                 # 单个空元组
        np.array([[], [1, 2]], dtype="object"),  # 包含非空标签的 NumPy 对象数组
    ]

    # 获取指定名称的度量函数
    metric = ALL_METRICS[name]
    # 对于每个序列，确保调用度量函数时会引发 ValueError 异常
    for seq in multilabel_sequences:
        with pytest.raises(ValueError):
            metric(seq, seq)


# 对每个名称按字母顺序排序，使用参数化测试，逐个执行测试函数
@pytest.mark.parametrize("name", sorted(METRICS_WITH_NORMALIZE_OPTION))
def test_normalize_option_binary_classification(name):
    # 在二分类情况下进行测试
    n_classes = 2
    n_samples = 20
    random_state = check_random_state(0)

    # 生成随机真实标签、预测标签和分数
    y_true = random_state.randint(0, n_classes, size=(n_samples,))
    y_pred = random_state.randint(0, n_classes, size=(n_samples,))
    y_score = random_state.normal(size=y_true.shape)

    # 获取指定名称的度量函数
    metrics = ALL_METRICS[name]
    # 根据度量函数是否支持阈值化，选择相应的预测结果
    pred = y_score if name in THRESHOLDED_METRICS else y_pred
    # 测量带有标准化选项的结果
    measure_normalized = metrics(y_true, pred, normalize=True)
    measure_not_normalized = metrics(y_true, pred, normalize=False)

    # 断言标准化后的度量结果小于0
    assert_array_less(
        -1.0 * measure_normalized,
        0,
        err_msg="We failed to test correctly the normalize option",
    )

    # 断言标准化后的度量结果与非标准化结果除以样本数的近似相等性
    assert_allclose(
        measure_normalized,
        measure_not_normalized / n_samples,
        err_msg=f"Failed with {name}",
    )


# 对每个名称按字母顺序排序，使用参数化测试，逐个执行测试函数
@pytest.mark.parametrize("name", sorted(METRICS_WITH_NORMALIZE_OPTION))
def test_normalize_option_multiclass_classification(name):
    # 在多分类情况下进行测试
    n_classes = 4
    n_samples = 20
    random_state = check_random_state(0)

    # 生成随机真实标签、预测标签和分数
    y_true = random_state.randint(0, n_classes, size=(n_samples,))
    y_pred = random_state.randint(0, n_classes, size=(n_samples,))
    y_score = random_state.uniform(size=(n_samples, n_classes))

    # 获取指定名称的度量函数
    metrics = ALL_METRICS[name]
    # 根据度量函数是否支持阈值化，选择相应的预测结果
    pred = y_score if name in THRESHOLDED_METRICS else y_pred
    # 测量带有标准化选项的结果
    measure_normalized = metrics(y_true, pred, normalize=True)
    measure_not_normalized = metrics(y_true, pred, normalize=False)

    # 断言标准化后的度量结果小于0
    assert_array_less(
        -1.0 * measure_normalized,
        0,
        err_msg="We failed to test correctly the normalize option",
    )

    # 断言标准化后的度量结果与非标准化结果除以样本数的近似相等性
    assert_allclose(
        measure_normalized,
        measure_not_normalized / n_samples,
        err_msg=f"Failed with {name}",
    )


# 对每个名称按字母顺序排序，使用参数化测试，逐个执行测试函数
@pytest.mark.parametrize(
    "name", sorted(METRICS_WITH_NORMALIZE_OPTION.intersection(MULTILABELS_METRICS))
)
def test_normalize_option_multilabel_classification(name):
    # 在多标签情况下进行测试
    n_classes = 4
    n_samples = 100
    random_state = check_random_state(0)

    # 生成随机多标签分类数据
    # 为确保包含至少一个未标记的条目，使用两个随机状态
    _, y_true = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=0,
        allow_unlabeled=True,
        n_samples=n_samples,
    )
    # 使用 make_multilabel_classification 函数生成多标签分类数据，其中只有一个特征，n_classes 指定类别数量，random_state 设置随机种子，allow_unlabeled 允许存在未标记样本，n_samples 指定样本数量
    _, y_pred = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=1,
        allow_unlabeled=True,
        n_samples=n_samples,
    )

    # 使用随机数生成器 random_state 生成一个与 y_true 形状相同的随机分数向量作为 y_score
    y_score = random_state.uniform(size=y_true.shape)

    # 确保 y_true 和 y_pred 至少包含一个空标签，将每个向量扩展添加 n_classes 个零
    y_true += [0] * n_classes
    y_pred += [0] * n_classes

    # 从 ALL_METRICS 字典中获取指定名称 name 对应的评估指标函数
    metrics = ALL_METRICS[name]
    
    # 如果指标名称 name 存在于 THRESHOLDED_METRICS 中，则使用 y_score 作为预测值，否则使用 y_pred
    pred = y_score if name in THRESHOLDED_METRICS else y_pred
    
    # 使用指定的评估指标函数 metrics 对 y_true 和 pred 进行归一化计算，normalize=True 表示进行归一化处理
    measure_normalized = metrics(y_true, pred, normalize=True)
    
    # 使用指定的评估指标函数 metrics 对 y_true 和 pred 进行非归一化计算，normalize=False 表示不进行归一化处理
    measure_not_normalized = metrics(y_true, pred, normalize=False)

    # 断言：确保 measure_normalized 的值小于 -1.0，用以验证归一化选项的正确性
    assert_array_less(
        -1.0 * measure_normalized,
        0,
        err_msg="We failed to test correctly the normalize option",
    )

    # 断言：确保 measure_normalized 与 measure_not_normalized 除以样本数量后的值接近，用以验证计算的准确性
    assert_allclose(
        measure_normalized,
        measure_not_normalized / n_samples,
        err_msg=f"Failed with {name}",
    )
# 忽略警告的装饰器函数
@ignore_warnings
# 检查平均化参数的内部函数
def _check_averaging(
    metric, y_true, y_pred, y_true_binarize, y_pred_binarize, is_multilabel
):
    # 获取二进制化后的真实标签数组的形状
    n_samples, n_classes = y_true_binarize.shape

    # 没有平均化的测量
    label_measure = metric(y_true, y_pred, average=None)
    # 断言每个类别的测量值近似相等
    assert_allclose(
        label_measure,
        [
            metric(y_true_binarize[:, i], y_pred_binarize[:, i])
            for i in range(n_classes)
        ],
    )

    # 微平均测量
    micro_measure = metric(y_true, y_pred, average="micro")
    # 断言微平均测量值近似等于展平后的二进制化标签的测量值
    assert_allclose(
        micro_measure, metric(y_true_binarize.ravel(), y_pred_binarize.ravel())
    )

    # 宏平均测量
    macro_measure = metric(y_true, y_pred, average="macro")
    # 断言宏平均测量值近似等于标签测量值的平均值
    assert_allclose(macro_measure, np.mean(label_measure))

    # 加权平均测量
    weights = np.sum(y_true_binarize, axis=0, dtype=int)

    if np.sum(weights) != 0:
        weighted_measure = metric(y_true, y_pred, average="weighted")
        # 断言加权平均测量值近似等于标签测量值按权重加权后的平均值
        assert_allclose(weighted_measure, np.average(label_measure, weights=weights))
    else:
        weighted_measure = metric(y_true, y_pred, average="weighted")
        # 断言加权平均测量值近似等于0，因为权重和为0
        assert_allclose(weighted_measure, 0)

    # 样本平均测量
    if is_multilabel:
        sample_measure = metric(y_true, y_pred, average="samples")
        # 断言样本平均测量值近似等于每个样本的测量值的平均值
        assert_allclose(
            sample_measure,
            np.mean(
                [
                    metric(y_true_binarize[i], y_pred_binarize[i])
                    for i in range(n_samples)
                ]
            ),
        )

    # 断言未知平均化参数会引发 ValueError 异常
    with pytest.raises(ValueError):
        metric(y_true, y_pred, average="unknown")
    # 断言无效的平均化参数会引发 ValueError 异常
    with pytest.raises(ValueError):
        metric(y_true, y_pred, average="garbage")


# 检查指定测量名称的平均化多类别测量函数
def check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score):
    # 判断是否为多标签数据类型
    is_multilabel = type_of_target(y_true).startswith("multilabel")

    # 获取指定名称的测量函数
    metric = ALL_METRICS[name]

    # 如果测量名称在具有平均化选项的列表中
    if name in METRICS_WITH_AVERAGING:
        # 调用内部函数进行平均化检查
        _check_averaging(
            metric, y_true, y_pred, y_true_binarize, y_pred_binarize, is_multilabel
        )
    # 如果测量名称在具有阈值的平均化选项的列表中
    elif name in THRESHOLDED_METRICS_WITH_AVERAGING:
        # 调用内部函数进行平均化检查，使用阈值化后的预测分数
        _check_averaging(
            metric, y_true, y_score, y_true_binarize, y_score, is_multilabel
        )
    else:
        # 抛出异常，测量不记录为具有平均选项
        raise ValueError("Metric is not recorded as having an average option")


# 参数化测试，对具有平均化选项的测量名称进行排序并依次执行
@pytest.mark.parametrize("name", sorted(METRICS_WITH_AVERAGING))
def test_averaging_multiclass(name):
    # 定义样本数和类别数
    n_samples, n_classes = 50, 3
    # 设定随机种子
    random_state = check_random_state(0)
    # 随机生成真实标签和预测标签
    y_true = random_state.randint(0, n_classes, size=(n_samples,))
    y_pred = random_state.randint(0, n_classes, size=(n_samples,))
    # 随机生成预测分数
    y_score = random_state.uniform(size=(n_samples, n_classes))

    # 标签二进制化
    lb = LabelBinarizer().fit(y_true)
    y_true_binarize = lb.transform(y_true)
    y_pred_binarize = lb.transform(y_pred)

    # 调用检查平均化函数
    check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score)


# 参数化测试，对具有平均化选项和阈值化的测量名称进行排序并依次执行
@pytest.mark.parametrize(
    "name", sorted(METRICS_WITH_AVERAGING | THRESHOLDED_METRICS_WITH_AVERAGING)
)
# 定义一个测试函数，用于测试多标签分类的平均指标计算
def test_averaging_multilabel(name):
    # 设置样本数和类数
    n_samples, n_classes = 40, 5
    # 生成多标签分类数据，返回特征和标签
    _, y = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=5,
        n_samples=n_samples,
        allow_unlabeled=False,
    )
    # 将标签分为真实标签和预测标签
    y_true = y[:20]
    y_pred = y[20:]
    # 生成随机得分作为预测分数
    y_score = check_random_state(0).normal(size=(20, n_classes))
    # 设置二进制化后的真实标签和预测标签
    y_true_binarize = y_true
    y_pred_binarize = y_pred

    # 调用函数检查平均指标
    check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score)


# 使用参数化测试装饰器，遍历所有需要平均的指标名称
@pytest.mark.parametrize("name", sorted(METRICS_WITH_AVERAGING))
def test_averaging_multilabel_all_zeroes(name):
    # 生成全零数组作为真实标签、预测标签和预测分数
    y_true = np.zeros((20, 3))
    y_pred = np.zeros((20, 3))
    y_score = np.zeros((20, 3))
    # 设置二进制化后的真实标签和预测标签
    y_true_binarize = y_true
    y_pred_binarize = y_pred

    # 调用函数检查平均指标
    check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score)


# 定义一个测试函数，用于测试二进制多标签分类中所有值为零的情况
def test_averaging_binary_multilabel_all_zeroes():
    # 生成全零数组作为真实标签和预测标签
    y_true = np.zeros((20, 3))
    y_pred = np.zeros((20, 3))
    # 设置二进制化后的真实标签和预测标签
    y_true_binarize = y_true
    y_pred_binarize = y_pred
    # 定义一个 lambda 函数，用于测试在权重和为零时的平均二进制评分
    binary_metric = lambda y_true, y_score, average="macro": _average_binary_score(
        precision_score, y_true, y_score, average
    )
    # 调用函数检查平均指标
    _check_averaging(
        binary_metric,
        y_true,
        y_pred,
        y_true_binarize,
        y_pred_binarize,
        is_multilabel=True,
    )


# 使用参数化测试装饰器，遍历所有需要平均的指标名称
@pytest.mark.parametrize("name", sorted(METRICS_WITH_AVERAGING))
def test_averaging_multilabel_all_ones(name):
    # 生成全一数组作为真实标签、预测标签和预测分数
    y_true = np.ones((20, 3))
    y_pred = np.ones((20, 3))
    y_score = np.ones((20, 3))
    # 设置二进制化后的真实标签和预测标签
    y_true_binarize = y_true
    y_pred_binarize = y_pred

    # 调用函数检查平均指标
    check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score)


# 忽略警告装饰器，定义一个检查样本权重不变性的函数
@ignore_warnings
def check_sample_weight_invariance(name, metric, y1, y2):
    # 设定随机种子生成随机数发生器
    rng = np.random.RandomState(0)
    # 生成随机样本权重
    sample_weight = rng.randint(1, 10, size=len(y1))

    # 对于二进制情况，当 name 是 "top_k_accuracy_score" 时，设置 k=1
    metric = partial(metric, k=1) if name == "top_k_accuracy_score" else metric

    # 检查单位权重和无权重的情况下得分相同
    unweighted_score = metric(y1, y2, sample_weight=None)

    # 断言检查两者得分相等
    assert_allclose(
        unweighted_score,
        metric(y1, y2, sample_weight=np.ones(shape=len(y1))),
        err_msg="For %s sample_weight=None is not equivalent to sample_weight=ones"
        % name,
    )

    # 检查加权和未加权得分不相等
    weighted_score = metric(y1, y2, sample_weight=sample_weight)

    # 使用上下文管理器提供自定义错误消息
    with pytest.raises(AssertionError):
        assert_allclose(unweighted_score, weighted_score)
        raise ValueError(
            "Unweighted and weighted scores are unexpectedly "
            "almost equal (%s) and (%s) "
            "for %s" % (unweighted_score, weighted_score, name)
        )

    # 检查样本权重可以是一个列表
    # 计算带权重的评分列表，使用给定的权重列表
    weighted_score_list = metric(y1, y2, sample_weight=sample_weight.tolist())

    # 断言带权重的评分与列表输入的带权重评分相等，如果不相等则抛出错误信息
    assert_allclose(
        weighted_score,
        weighted_score_list,
        err_msg=(
            "Weighted scores for array and list "
            "sample_weight input are not equal (%s != %s) for %s"
        )
        % (weighted_score, weighted_score_list, name),
    )

    # 检查使用整数权重是否等同于重复样本的权重
    repeat_weighted_score = metric(
        np.repeat(y1, sample_weight, axis=0),
        np.repeat(y2, sample_weight, axis=0),
        sample_weight=None,
    )
    assert_allclose(
        weighted_score,
        repeat_weighted_score,
        err_msg="Weighting %s is not equal to repeating samples" % name,
    )

    # 检查忽略部分样本的权重与将相应权重设为零是否等效
    sample_weight_subset = sample_weight[1::2]
    sample_weight_zeroed = np.copy(sample_weight)
    sample_weight_zeroed[::2] = 0
    y1_subset = y1[1::2]
    y2_subset = y2[1::2]
    weighted_score_subset = metric(
        y1_subset, y2_subset, sample_weight=sample_weight_subset
    )
    weighted_score_zeroed = metric(y1, y2, sample_weight=sample_weight_zeroed)
    assert_allclose(
        weighted_score_subset,
        weighted_score_zeroed,
        err_msg=(
            "Zeroing weights does not give the same result as "
            "removing the corresponding samples (%s != %s) for %s"
        )
        % (weighted_score_zeroed, weighted_score_subset, name),
    )

    # 如果名称不以"unnormalized"开头，检查评分在权重乘以一个公因子时是否不变
    if not name.startswith("unnormalized"):
        for scaling in [2, 0.3]:
            assert_allclose(
                weighted_score,
                metric(y1, y2, sample_weight=sample_weight * scaling),
                err_msg="%s sample_weight is not invariant under scaling" % name,
            )

    # 检查如果y_true和sample_weight中的样本数不相等，则引发有意义的错误
    error_message = (
        r"Found input variables with inconsistent numbers of "
        r"samples: \[{}, {}, {}\]".format(
            _num_samples(y1), _num_samples(y2), _num_samples(sample_weight) * 2
        )
    )
    with pytest.raises(ValueError, match=error_message):
        metric(y1, y2, sample_weight=np.hstack([sample_weight, sample_weight]))
# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_regression_sample_weight_invariance 参数化
@pytest.mark.parametrize(
    "name",
    # 对 ALL_METRICS 和 REGRESSION_METRICS 取交集，再去掉 METRICS_WITHOUT_SAMPLE_WEIGHT 中的元素，并进行排序
    sorted(
        set(ALL_METRICS).intersection(set(REGRESSION_METRICS))
        - METRICS_WITHOUT_SAMPLE_WEIGHT
    ),
)
def test_regression_sample_weight_invariance(name):
    # 设置样本数为 50
    n_samples = 50
    # 创建随机数生成器，种子为 0
    random_state = check_random_state(0)
    # 生成回归任务的真实值和预测值数组
    y_true = random_state.random_sample(size=(n_samples,))
    y_pred = random_state.random_sample(size=(n_samples,))
    # 获取指定名称的度量函数
    metric = ALL_METRICS[name]
    # 调用辅助函数，检查样本权重不变性
    check_sample_weight_invariance(name, metric, y_true, y_pred)


# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_binary_sample_weight_invariance 参数化
@pytest.mark.parametrize(
    "name",
    # 对 ALL_METRICS 减去 REGRESSION_METRICS、METRICS_WITHOUT_SAMPLE_WEIGHT 和 METRIC_UNDEFINED_BINARY 的集合，并进行排序
    sorted(
        set(ALL_METRICS)
        - set(REGRESSION_METRICS)
        - METRICS_WITHOUT_SAMPLE_WEIGHT
        - METRIC_UNDEFINED_BINARY
    ),
)
def test_binary_sample_weight_invariance(name):
    # 设置样本数为 50
    n_samples = 50
    # 创建随机数生成器，种子为 0
    random_state = check_random_state(0)
    # 生成二分类任务的真实值、预测值和得分数组
    y_true = random_state.randint(0, 2, size=(n_samples,))
    y_pred = random_state.randint(0, 2, size=(n_samples,))
    y_score = random_state.random_sample(size=(n_samples,))
    # 获取指定名称的度量函数
    metric = ALL_METRICS[name]
    # 如果度量名称在 THRESHOLDED_METRICS 中，则调用辅助函数，检查样本权重不变性，传入真实值和得分数组
    if name in THRESHOLDED_METRICS:
        check_sample_weight_invariance(name, metric, y_true, y_score)
    else:
        # 否则，调用辅助函数，检查样本权重不变性，传入真实值和预测值数组
        check_sample_weight_invariance(name, metric, y_true, y_pred)


# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_multiclass_sample_weight_invariance 参数化
@pytest.mark.parametrize(
    "name",
    # 对 ALL_METRICS 减去 REGRESSION_METRICS、METRICS_WITHOUT_SAMPLE_WEIGHT 和 METRIC_UNDEFINED_BINARY_MULTICLASS 的集合，并进行排序
    sorted(
        set(ALL_METRICS)
        - set(REGRESSION_METRICS)
        - METRICS_WITHOUT_SAMPLE_WEIGHT
        - METRIC_UNDEFINED_BINARY_MULTICLASS
    ),
)
def test_multiclass_sample_weight_invariance(name):
    # 设置样本数为 50
    n_samples = 50
    # 创建随机数生成器，种子为 0
    random_state = check_random_state(0)
    # 生成多分类任务的真实值、预测值和得分数组
    y_true = random_state.randint(0, 5, size=(n_samples,))
    y_pred = random_state.randint(0, 5, size=(n_samples,))
    y_score = random_state.random_sample(size=(n_samples, 5))
    # 获取指定名称的度量函数
    metric = ALL_METRICS[name]
    # 如果度量名称在 THRESHOLDED_METRICS 中，则进行 softmax 操作，再调用辅助函数，检查样本权重不变性，传入真实值和归一化后的得分数组
    if name in THRESHOLDED_METRICS:
        temp = np.exp(-y_score)
        y_score_norm = temp / temp.sum(axis=-1).reshape(-1, 1)
        check_sample_weight_invariance(name, metric, y_true, y_score_norm)
    else:
        # 否则，调用辅助函数，检查样本权重不变性，传入真实值和预测值数组
        check_sample_weight_invariance(name, metric, y_true, y_pred)


# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_multilabel_sample_weight_invariance 参数化
@pytest.mark.parametrize(
    "name",
    # 对 MULTILABELS_METRICS、THRESHOLDED_MULTILABEL_METRICS 和 MULTIOUTPUT_METRICS 的并集减去 METRICS_WITHOUT_SAMPLE_WEIGHT 的集合
    sorted(
        (MULTILABELS_METRICS | THRESHOLDED_MULTILABEL_METRICS | MULTIOUTPUT_METRICS)
        - METRICS_WITHOUT_SAMPLE_WEIGHT
    ),
)
def test_multilabel_sample_weight_invariance(name):
    # 创建随机数生成器，种子为 0
    random_state = check_random_state(0)
    # 生成多标签指示任务的真实值数组
    _, ya = make_multilabel_classification(
        n_features=1, n_classes=10, random_state=0, n_samples=50, allow_unlabeled=False
    )
    _, yb = make_multilabel_classification(
        n_features=1, n_classes=10, random_state=1, n_samples=50, allow_unlabeled=False
    )
    y_true = np.vstack([ya, yb])
    # 生成多标签指示任务的预测值数组
    y_pred = np.vstack([ya, ya])
    # 生成随机的得分数组
    y_score = random_state.uniform(size=y_true.shape)
    # 一些度量函数（例如 log_loss）要求 y_score 是概率（和为1）
    y_score /= y_score.sum(axis=1, keepdims=True)
    # 获取指定名称的度量函数
    metric = ALL_METRICS[name]
    # 如果给定的 `name` 存在于 THRESHOLDED_METRICS 中，则调用 check_sample_weight_invariance 函数并传入相应的参数 y_true 和 y_score
    if name in THRESHOLDED_METRICS:
        check_sample_weight_invariance(name, metric, y_true, y_score)
    # 否则，调用 check_sample_weight_invariance 函数并传入相应的参数 y_true 和 y_pred
    else:
        check_sample_weight_invariance(name, metric, y_true, y_pred)
@ignore_warnings
def test_no_averaging_labels():
    # 带有 @ignore_warnings 装饰器的测试函数，用于测试在多类和多标签情况下未使用平均化时的标签参数
    y_true_multilabel = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    y_pred_multilabel = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
    y_true_multiclass = np.array([0, 1, 2])
    y_pred_multiclass = np.array([0, 2, 3])
    labels = np.array([3, 0, 1, 2])
    _, inverse_labels = np.unique(labels, return_inverse=True)

    for name in METRICS_WITH_AVERAGING:
        for y_true, y_pred in [
            [y_true_multiclass, y_pred_multiclass],
            [y_true_multilabel, y_pred_multilabel],
        ]:
            # 如果指标不在 MULTILABELS_METRICS 中，并且 y_pred 的维度大于1，则跳过
            if name not in MULTILABELS_METRICS and y_pred.ndim > 1:
                continue

            metric = ALL_METRICS[name]

            # 使用给定的标签计算非平均化的指标分数
            score_labels = metric(y_true, y_pred, labels=labels, average=None)
            # 计算非平均化的指标分数
            score = metric(y_true, y_pred, average=None)
            # 断言使用反向标签的 score_labels 等于 score 中的值
            assert_array_equal(score_labels, score[inverse_labels])


@pytest.mark.parametrize(
    "name", sorted(MULTILABELS_METRICS - {"unnormalized_multilabel_confusion_matrix"})
)
def test_multilabel_label_permutations_invariance(name):
    # 参数化测试，测试多标签指标的标签置换不变性
    random_state = check_random_state(0)
    n_samples, n_classes = 20, 4

    y_true = random_state.randint(0, 2, size=(n_samples, n_classes))
    y_score = random_state.randint(0, 2, size=(n_samples, n_classes))

    metric = ALL_METRICS[name]
    # 计算指标分数
    score = metric(y_true, y_score)

    for perm in permutations(range(n_classes), n_classes):
        # 对 y_true 和 y_score 进行标签置换
        y_score_perm = y_score[:, perm]
        y_true_perm = y_true[:, perm]

        # 计算当前标签置换下的指标分数
        current_score = metric(y_true_perm, y_score_perm)
        # 断言 score 等于当前标签置换下的分数
        assert_almost_equal(score, current_score)


@pytest.mark.parametrize(
    "name", sorted(THRESHOLDED_MULTILABEL_METRICS | MULTIOUTPUT_METRICS)
)
def test_thresholded_multilabel_multioutput_permutations_invariance(name):
    # 参数化测试，测试阈值化的多标签多输出指标的置换不变性
    random_state = check_random_state(0)
    n_samples, n_classes = 20, 4
    y_true = random_state.randint(0, 2, size=(n_samples, n_classes))
    y_score = random_state.uniform(size=y_true.shape)

    # 一些指标（如 log_loss）要求 y_score 是概率（总和为1）
    y_score /= y_score.sum(axis=1, keepdims=True)

    # 确保所有样本至少有一个标签。这是为了解决在 average="sample" 模式下运行指标时的错误
    y_true[y_true.sum(1) == 4, 0] = 0
    y_true[y_true.sum(1) == 0, 0] = 1

    metric = ALL_METRICS[name]
    # 计算指标分数
    score = metric(y_true, y_score)
    # 对于给定的类别数目 n_classes，生成所有可能的排列组合
    for perm in permutations(range(n_classes), n_classes):
        # 根据当前排列组合重新排列 y_score 和 y_true 的列顺序
        y_score_perm = y_score[:, perm]
        y_true_perm = y_true[:, perm]

        # 计算当前排列组合下的评估指标得分
        current_score = metric(y_true_perm, y_score_perm)
        
        # 如果评估指标是平均绝对百分比误差（MAPE）
        if metric == mean_absolute_percentage_error:
            # 断言当前得分是有限数值
            assert np.isfinite(current_score)
            # 断言当前得分大于 1e6
            assert current_score > 1e6
            # 在计算 MAPE 时，如果 y_true 的值恰好为零，MAPE 的值不具有实际意义。
            # 因此在这种情况下，我们期望得到一个非常大的有限值。
            # 这里我们不比较具体的数值。
        else:
            # 断言当前得分与预期得分几乎相等
            assert_almost_equal(score, current_score)
# 使用 pytest 的 parametrize 装饰器，为测试函数 test_thresholded_metric_permutation_invariance 添加参数化测试数据
@pytest.mark.parametrize(
    "name", sorted(set(THRESHOLDED_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS)
)
def test_thresholded_metric_permutation_invariance(name):
    # 设置样本数和类别数
    n_samples, n_classes = 100, 3
    # 使用随机种子初始化随机状态
    random_state = check_random_state(0)

    # 生成随机的分类器得分
    y_score = random_state.rand(n_samples, n_classes)
    # 将分类器得分转换为概率
    temp = np.exp(-y_score)
    y_score = temp / temp.sum(axis=-1).reshape(-1, 1)
    # 随机生成真实标签
    y_true = random_state.randint(0, n_classes, size=n_samples)

    # 获取指定名称的度量函数
    metric = ALL_METRICS[name]
    # 计算度量得分
    score = metric(y_true, y_score)

    # 对类别的排列进行全排列并测试不变性
    for perm in permutations(range(n_classes), n_classes):
        # 构造反向排列索引
        inverse_perm = np.zeros(n_classes, dtype=int)
        inverse_perm[list(perm)] = np.arange(n_classes)
        # 使用反向排列对分类器得分和真实标签进行排列
        y_score_perm = y_score[:, inverse_perm]
        y_true_perm = np.take(perm, y_true)

        # 计算排列后的度量得分
        current_score = metric(y_true_perm, y_score_perm)
        # 断言原始得分与排列后得分几乎相等
        assert_almost_equal(score, current_score)


# 使用 pytest 的 parametrize 装饰器，为测试函数 test_metrics_consistent_type_error 添加参数化测试数据
@pytest.mark.parametrize("metric_name", CLASSIFICATION_METRICS)
def test_metrics_consistent_type_error(metric_name):
    # 检查当 y_true 和 y_pred 类型不匹配时是否会引发合适的错误消息
    rng = np.random.RandomState(42)
    # 创建包含字符串的数组作为 y_true
    y1 = np.array(["spam"] * 3 + ["eggs"] * 2, dtype=object)
    y2 = rng.randint(0, 2, size=y1.size)

    # 预期的错误消息
    err_msg = "Labels in y_true and y_pred should be of the same type."
    # 使用 pytest 的 assertRaises 来检测是否引发指定类型的异常及其错误消息
    with pytest.raises(TypeError, match=err_msg):
        CLASSIFICATION_METRICS[metric_name](y1, y2)


# 使用 pytest 的 parametrize 装饰器，为测试函数 test_metrics_pos_label_error_str 添加参数化测试数据
@pytest.mark.parametrize(
    "metric, y_pred_threshold",
    [
        (average_precision_score, True),
        (brier_score_loss, True),
        (f1_score, False),
        (partial(fbeta_score, beta=1), False),
        (jaccard_score, False),
        (precision_recall_curve, True),
        (precision_score, False),
        (recall_score, False),
        (roc_curve, True),
    ],
)
@pytest.mark.parametrize("dtype_y_str", [str, object])
def test_metrics_pos_label_error_str(metric, y_pred_threshold, dtype_y_str):
    # 检查当没有指定 pos_label 且目标是字符串时是否会引发正确的错误消息
    rng = np.random.RandomState(42)
    # 创建包含字符串的数组作为 y_true
    y1 = np.array(["spam"] * 3 + ["eggs"] * 2, dtype=dtype_y_str)
    y2 = rng.randint(0, 2, size=y1.size)

    # 如果不需要 y_pred_threshold，则使用预定义的字符串数组
    if not y_pred_threshold:
        y2 = np.array(["spam", "eggs"], dtype=dtype_y_str)[y2]

    # 根据度量函数的签名获取预期的错误消息
    pos_label_default = signature(metric).parameters["pos_label"].default
    err_msg_pos_label_None = (
        "y_true takes value in {'eggs', 'spam'} and pos_label is not "
        "specified: either make y_true take value in {0, 1} or {-1, 1} or "
        "pass pos_label explicit"
    )
    err_msg_pos_label_1 = (
        r"pos_label=1 is not a valid label. It should be one of " r"\['eggs', 'spam'\]"
    )
    err_msg = err_msg_pos_label_1 if pos_label_default == 1 else err_msg_pos_label_None

    # 使用 pytest 的 assertRaises 来检测是否引发指定类型的异常及其错误消息
    with pytest.raises(ValueError, match=err_msg):
        metric(y1, y2)


# 定义函数 check_array_api_metric，用于检查数组 API 的度量函数
def check_array_api_metric(
    metric, array_namespace, device, dtype_name, a_np, b_np, **metric_kwargs
):
    # 此处尚未提供代码，无法为其添加注释
    # 调用 _array_api_for_tests 函数，返回适当的数组 API 对象 xp
    xp = _array_api_for_tests(array_namespace, device)
    
    # 使用 xp.asarray 将 a_np 转换为特定设备上的数组 a_xp
    a_xp = xp.asarray(a_np, device=device)
    # 使用 xp.asarray 将 b_np 转换为特定设备上的数组 b_xp
    b_xp = xp.asarray(b_np, device=device)
    
    # 调用 metric 函数计算 a_np 和 b_np 之间的度量指标，使用 metric_kwargs 作为参数
    metric_np = metric(a_np, b_np, **metric_kwargs)
    
    # 如果 metric_kwargs 中存在 "sample_weight" 参数，则将其转换为特定设备上的数组
    if metric_kwargs.get("sample_weight") is not None:
        metric_kwargs["sample_weight"] = xp.asarray(
            metric_kwargs["sample_weight"], device=device
        )
    
    # 获取 metric_kwargs 中的 "multioutput" 参数
    multioutput = metric_kwargs.get("multioutput")
    # 如果 "multioutput" 是 numpy 数组，则将其转换为特定设备上的数组
    if isinstance(multioutput, np.ndarray):
        metric_kwargs["multioutput"] = xp.asarray(multioutput, device=device)
    
    # 使用 config_context 开启数组 API 分发设置
    with config_context(array_api_dispatch=True):
        # 使用 xp 计算 a_xp 和 b_xp 之间的度量指标，使用 metric_kwargs 作为参数
        metric_xp = metric(a_xp, b_xp, **metric_kwargs)
    
        # 断言 _convert_to_numpy 将 metric_xp 转换为 numpy 数组，并与 metric_np 进行比较
        assert_allclose(
            _convert_to_numpy(xp.asarray(metric_xp), xp),
            metric_np,
            atol=_atol_for_type(dtype_name),
        )
# 检查数组API的二分类指标函数
def check_array_api_binary_classification_metric(
    metric, array_namespace, device, dtype_name
):
    # 创建包含真实标签的NumPy数组
    y_true_np = np.array([0, 0, 1, 1])
    # 创建包含预测标签的NumPy数组
    y_pred_np = np.array([0, 1, 0, 1])

    # 调用通用的数组API检查函数，计算二分类指标，不考虑样本权重
    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        sample_weight=None,
    )

    # 创建带有样本权重的NumPy数组
    sample_weight = np.array([0.0, 0.1, 2.0, 1.0], dtype=dtype_name)

    # 调用通用的数组API检查函数，计算二分类指标，考虑样本权重
    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        sample_weight=sample_weight,
    )


# 检查数组API的多分类指标函数
def check_array_api_multiclass_classification_metric(
    metric, array_namespace, device, dtype_name
):
    # 创建包含真实标签的NumPy数组（多分类）
    y_true_np = np.array([0, 1, 2, 3])
    # 创建包含预测标签的NumPy数组（多分类）
    y_pred_np = np.array([0, 1, 0, 2])

    # 调用通用的数组API检查函数，计算多分类指标，不考虑样本权重
    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        sample_weight=None,
    )

    # 创建带有样本权重的NumPy数组
    sample_weight = np.array([0.0, 0.1, 2.0, 1.0], dtype=dtype_name)

    # 调用通用的数组API检查函数，计算多分类指标，考虑样本权重
    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        sample_weight=sample_weight,
    )


# 检查数组API的多标签分类指标函数
def check_array_api_multilabel_classification_metric(
    metric, array_namespace, device, dtype_name
):
    # 创建包含真实标签的NumPy数组（多标签分类）
    y_true_np = np.array([[1, 1], [0, 1], [0, 0]], dtype=dtype_name)
    # 创建包含预测标签的NumPy数组（多标签分类）
    y_pred_np = np.array([[1, 1], [1, 1], [1, 1]], dtype=dtype_name)

    # 调用通用的数组API检查函数，计算多标签分类指标，不考虑样本权重
    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        sample_weight=None,
    )

    # 创建带有样本权重的NumPy数组
    sample_weight = np.array([0.0, 0.1, 2.0], dtype=dtype_name)

    # 调用通用的数组API检查函数，计算多标签分类指标，考虑样本权重
    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        sample_weight=sample_weight,
    )


# 检查数组API的回归指标函数
def check_array_api_regression_metric(metric, array_namespace, device, dtype_name):
    # 创建包含真实值的NumPy数组（回归）
    y_true_np = np.array([2.0, 0.1, 1.0, 4.0], dtype=dtype_name)
    # 创建包含预测值的NumPy数组（回归）
    y_pred_np = np.array([0.5, 0.5, 2, 2], dtype=dtype_name)

    # 获取指标函数的参数信息
    metric_kwargs = {}
    metric_params = signature(metric).parameters

    # 如果指标函数支持样本权重，则设置为None
    if "sample_weight" in metric_params:
        metric_kwargs["sample_weight"] = None

    # 调用通用的数组API检查函数，计算回归指标，不考虑样本权重
    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        **metric_kwargs,
    )

    # 如果指标函数支持样本权重，则创建带有样本权重的NumPy数组
    if "sample_weight" in metric_params:
        metric_kwargs["sample_weight"] = np.array(
            [0.1, 2.0, 1.5, 0.5], dtype=dtype_name
        )

        # 调用通用的数组API检查函数，计算回归指标，考虑样本权重
        check_array_api_metric(
            metric,
            array_namespace,
            device,
            dtype_name,
            a_np=y_true_np,
            b_np=y_pred_np,
            **metric_kwargs,
        )


# 检查数组API的多输出回归指标函数
def check_array_api_regression_metric_multioutput(
    # 从给定的代码中，依次解包得到 metric, array_namespace, device, dtype_name 四个变量
    metric, array_namespace, device, dtype_name
    y_true_np = np.array([[1, 3, 2], [1, 2, 2]], dtype=dtype_name)
    y_pred_np = np.array([[1, 4, 4], [1, 1, 1]], dtype=dtype_name)
    # 创建包含真实值的 NumPy 数组
    # dtype_name 是数据类型的参数

    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        sample_weight=None,
    )
    # 调用函数 check_array_api_metric，传递真实值和预测值的 NumPy 数组作为参数
    # sample_weight 参数为 None，表示未使用样本权重

    sample_weight = np.array([0.1, 2.0], dtype=dtype_name)
    # 创建包含样本权重的 NumPy 数组

    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        sample_weight=sample_weight,
    )
    # 再次调用函数 check_array_api_metric，这次使用了样本权重数组作为参数

    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        multioutput=np.array([0.1, 0.3, 0.7], dtype=dtype_name),
    )
    # 再次调用函数 check_array_api_metric，使用多输出参数作为参数
    # multioutput 参数是一个 NumPy 数组

    check_array_api_metric(
        metric,
        array_namespace,
        device,
        dtype_name,
        a_np=y_true_np,
        b_np=y_pred_np,
        multioutput="raw_values",
    )
    # 最后一次调用函数 check_array_api_metric，使用字符串 "raw_values" 作为多输出参数
    # 这里的 "raw_values" 是一个特殊值，表示返回未经处理的原始值
    max_error: [check_array_api_regression_metric],
    # 定义了一个字典键值对，键为 max_error，值为包含单一元素的列表 [check_array_api_regression_metric]

    chi2_kernel: [check_array_api_metric_pairwise],
    # 定义了另一个字典键值对，键为 chi2_kernel，值为包含单一元素的列表 [check_array_api_metric_pairwise]
}

# 定义一个生成器函数，用于生成每个指标对应的检查器组合
def yield_metric_checker_combinations(metric_checkers=array_api_metric_checkers):
    # 遍历指标检查器字典的每个指标及其对应的检查器列表
    for metric, checkers in metric_checkers.items():
        # 遍历每个指标的检查器列表
        for checker in checkers:
            # 生成器返回每个指标和其对应的检查器
            yield metric, checker

# 使用 pytest 的 parametrize 装饰器，为测试函数参数化
@pytest.mark.parametrize(
    # 使用 yield_namespace_device_dtype_combinations() 生成的命名空间、设备和数据类型组合作为参数
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
)
# 使用 yield_metric_checker_combinations() 生成的指标和检查器组合作为参数
@pytest.mark.parametrize("metric, check_func", yield_metric_checker_combinations())
# 定义测试函数，测试数组 API 的兼容性
def test_array_api_compliance(metric, array_namespace, device, dtype_name, check_func):
    # 调用指定的检查函数，检查指标在指定的命名空间、设备和数据类型下的兼容性
    check_func(metric, array_namespace, device, dtype_name)
```