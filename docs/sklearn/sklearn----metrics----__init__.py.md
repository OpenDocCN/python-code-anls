# `D:\src\scipysrc\scikit-learn\sklearn\metrics\__init__.py`

```
# 导入模块：集成了得分函数、性能度量、成对度量和距离计算
from . import cluster
# 导入分类相关的函数和类
from ._classification import (
    accuracy_score,  # 计算准确率得分
    balanced_accuracy_score,  # 计算平衡准确率得分
    brier_score_loss,  # 计算布里尔分数损失
    class_likelihood_ratios,  # 计算类别似然比
    classification_report,  # 生成分类报告
    cohen_kappa_score,  # 计算Cohen's Kappa系数
    confusion_matrix,  # 计算混淆矩阵
    d2_log_loss_score,  # 计算D2对数损失得分
    f1_score,  # 计算F1分数
    fbeta_score,  # 计算F-beta分数
    hamming_loss,  # 计算汉明损失
    hinge_loss,  # 计算铰链损失
    jaccard_score,  # 计算Jaccard相似度分数
    log_loss,  # 计算对数损失
    matthews_corrcoef,  # 计算Matthews相关系数
    multilabel_confusion_matrix,  # 计算多标签混淆矩阵
    precision_recall_fscore_support,  # 计算精确率、召回率、F1分数和支持度
    precision_score,  # 计算精确率
    recall_score,  # 计算召回率
    zero_one_loss,  # 计算0-1损失
)
# 导入距离度量类
from ._dist_metrics import DistanceMetric
# 导入绘图模块：混淆矩阵展示
from ._plot.confusion_matrix import ConfusionMatrixDisplay
# 导入绘图模块：DET曲线展示
from ._plot.det_curve import DetCurveDisplay
# 导入绘图模块：精确率-召回率曲线展示
from ._plot.precision_recall_curve import PrecisionRecallDisplay
# 导入绘图模块：预测误差展示
from ._plot.regression import PredictionErrorDisplay
# 导入绘图模块：ROC曲线展示
from ._plot.roc_curve import RocCurveDisplay
# 导入排序度量函数和类
from ._ranking import (
    auc,  # 计算AUC值
    average_precision_score,  # 计算平均精度得分
    coverage_error,  # 计算覆盖误差
    dcg_score,  # 计算折损累计增益
    det_curve,  # 计算DET曲线
    label_ranking_average_precision_score,  # 计算标签排名平均精度得分
    label_ranking_loss,  # 计算标签排名损失
    ndcg_score,  # 计算归一化折损累计增益
    precision_recall_curve,  # 计算精确率-召回率曲线
    roc_auc_score,  # 计算ROC AUC值
    roc_curve,  # 计算ROC曲线
    top_k_accuracy_score,  # 计算Top-k准确率得分
)
# 导入回归相关的函数和类
from ._regression import (
    d2_absolute_error_score,  # 计算D2绝对误差得分
    d2_pinball_score,  # 计算D2 Pinball损失
    d2_tweedie_score,  # 计算D2 Tweedie损失
    explained_variance_score,  # 计算解释方差得分
    max_error,  # 计算最大误差
    mean_absolute_error,  # 计算平均绝对误差
    mean_absolute_percentage_error,  # 计算平均绝对百分比误差
    mean_gamma_deviance,  # 计算平均Gamma偏差
    mean_pinball_loss,  # 计算平均Pinball损失
    mean_poisson_deviance,  # 计算平均Poisson偏差
    mean_squared_error,  # 计算均方误差
    mean_squared_log_error,  # 计算均方对数误差
    mean_tweedie_deviance,  # 计算平均Tweedie偏差
    median_absolute_error,  # 计算中位数绝对误差
    r2_score,  # 计算R^2得分
    root_mean_squared_error,  # 计算均方根误差
    root_mean_squared_log_error,  # 计算均方根对数误差
)
# 导入评分函数和类
from ._scorer import check_scoring, get_scorer, get_scorer_names, make_scorer
# 导入聚类相关的函数和类
from .cluster import (
    adjusted_mutual_info_score,  # 计算调整后的互信息得分
    adjusted_rand_score,  # 计算调整后的Rand指数
    calinski_harabasz_score,  # 计算Calinski-Harabasz指数
    completeness_score,  # 计算完整性得分
    consensus_score,  # 计算一致性得分
    davies_bouldin_score,  # 计算Davies-Bouldin指数
    fowlkes_mallows_score,  # 计算Fowlkes-Mallows指数
    homogeneity_completeness_v_measure,  # 计算同质性、完整性和V-measure
    homogeneity_score,  # 计算同质性得分
    mutual_info_score,  # 计算互信息得分
    normalized_mutual_info_score,  # 计算归一化互信息得分
    pair_confusion_matrix,  # 计算成对混淆矩阵
    rand_score,  # 计算Rand指数
    silhouette_samples,  # 计算轮廓系数样本
    silhouette_score,  # 计算轮廓系数
    v_measure_score,  # 计算V-measure得分
)
# 导入成对距离和核函数相关的函数和类
from .pairwise import (
    euclidean_distances,  # 计算欧氏距离矩阵
    nan_euclidean_distances,  # 计算带有NaN值的欧氏距离矩阵
    pairwise_distances,  # 计算成对距离
    pairwise_distances_argmin,  # 计算成对距离的最小索引
    pairwise_distances_argmin_min,  # 计算成对距离的最小索引和值
    pairwise_distances_chunked,  # 分块计算成对距离
    pairwise_kernels,  # 计算成对核函数
)
    # 导入DetCurveDisplay类
    "DetCurveDisplay",
    # 导入det_curve函数
    "det_curve",
    # 导入DistanceMetric类
    "DistanceMetric",
    # 导入euclidean_distances函数
    "euclidean_distances",
    # 导入explained_variance_score函数
    "explained_variance_score",
    # 导入f1_score函数
    "f1_score",
    # 导入fbeta_score函数
    "fbeta_score",
    # 导入fowlkes_mallows_score函数
    "fowlkes_mallows_score",
    # 导入get_scorer函数
    "get_scorer",
    # 导入hamming_loss函数
    "hamming_loss",
    # 导入hinge_loss函数
    "hinge_loss",
    # 导入homogeneity_completeness_v_measure函数
    "homogeneity_completeness_v_measure",
    # 导入homogeneity_score函数
    "homogeneity_score",
    # 导入jaccard_score函数
    "jaccard_score",
    # 导入label_ranking_average_precision_score函数
    "label_ranking_average_precision_score",
    # 导入label_ranking_loss函数
    "label_ranking_loss",
    # 导入log_loss函数
    "log_loss",
    # 导入make_scorer函数
    "make_scorer",
    # 导入nan_euclidean_distances函数
    "nan_euclidean_distances",
    # 导入matthews_corrcoef函数
    "matthews_corrcoef",
    # 导入max_error函数
    "max_error",
    # 导入mean_absolute_error函数
    "mean_absolute_error",
    # 导入mean_squared_error函数
    "mean_squared_error",
    # 导入mean_squared_log_error函数
    "mean_squared_log_error",
    # 导入mean_pinball_loss函数
    "mean_pinball_loss",
    # 导入mean_poisson_deviance函数
    "mean_poisson_deviance",
    # 导入mean_gamma_deviance函数
    "mean_gamma_deviance",
    # 导入mean_tweedie_deviance函数
    "mean_tweedie_deviance",
    # 导入median_absolute_error函数
    "median_absolute_error",
    # 导入mean_absolute_percentage_error函数
    "mean_absolute_percentage_error",
    # 导入multilabel_confusion_matrix函数
    "multilabel_confusion_matrix",
    # 导入mutual_info_score函数
    "mutual_info_score",
    # 导入ndcg_score函数
    "ndcg_score",
    # 导入normalized_mutual_info_score函数
    "normalized_mutual_info_score",
    # 导入pair_confusion_matrix函数
    "pair_confusion_matrix",
    # 导入pairwise_distances函数
    "pairwise_distances",
    # 导入pairwise_distances_argmin函数
    "pairwise_distances_argmin",
    # 导入pairwise_distances_argmin_min函数
    "pairwise_distances_argmin_min",
    # 导入pairwise_distances_chunked函数
    "pairwise_distances_chunked",
    # 导入pairwise_kernels函数
    "pairwise_kernels",
    # 导入PrecisionRecallDisplay类
    "PrecisionRecallDisplay",
    # 导入precision_recall_curve函数
    "precision_recall_curve",
    # 导入precision_recall_fscore_support函数
    "precision_recall_fscore_support",
    # 导入precision_score函数
    "precision_score",
    # 导入PredictionErrorDisplay类
    "PredictionErrorDisplay",
    # 导入r2_score函数
    "r2_score",
    # 导入rand_score函数
    "rand_score",
    # 导入recall_score函数
    "recall_score",
    # 导入RocCurveDisplay类
    "RocCurveDisplay",
    # 导入roc_auc_score函数
    "roc_auc_score",
    # 导入roc_curve函数
    "roc_curve",
    # 导入root_mean_squared_log_error函数
    "root_mean_squared_log_error",
    # 导入root_mean_squared_error函数
    "root_mean_squared_error",
    # 导入get_scorer_names函数
    "get_scorer_names",
    # 导入silhouette_samples函数
    "silhouette_samples",
    # 导入silhouette_score函数
    "silhouette_score",
    # 导入top_k_accuracy_score函数
    "top_k_accuracy_score",
    # 导入v_measure_score函数
    "v_measure_score",
    # 导入zero_one_loss函数
    "zero_one_loss",
    # 导入brier_score_loss函数
    "brier_score_loss",
]
```