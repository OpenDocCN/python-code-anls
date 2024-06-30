# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\__init__.py`

```
"""Evaluation metrics for cluster analysis results.

- Supervised evaluation uses a ground truth class values for each sample.
- Unsupervised evaluation does use ground truths and measures the "quality" of the
  model itself.
"""

# 导入评估聚类分析结果的相关模块和函数

from ._bicluster import consensus_score
from ._supervised import (
    adjusted_mutual_info_score,  # 计算调整后的互信息得分
    adjusted_rand_score,         # 计算调整后的兰德指数得分
    completeness_score,          # 计算完整性得分
    contingency_matrix,          # 计算列联表
    entropy,                     # 计算熵
    expected_mutual_information, # 计算期望互信息
    fowlkes_mallows_score,       # 计算Fowlkes-Mallows指数
    homogeneity_completeness_v_measure,  # 计算均一性、完整性和V-measure
    homogeneity_score,           # 计算均一性得分
    mutual_info_score,           # 计算互信息得分
    normalized_mutual_info_score,  # 计算归一化互信息得分
    pair_confusion_matrix,       # 计算配对混淆矩阵
    rand_score,                  # 计算兰德指数
    v_measure_score,             # 计算V-measure得分
)
from ._unsupervised import (
    calinski_harabasz_score,     # 计算Calinski-Harabasz指数
    davies_bouldin_score,        # 计算Davies-Bouldin指数
    silhouette_samples,          # 计算轮廓系数样本值
    silhouette_score,            # 计算轮廓系数
)

__all__ = [
    "adjusted_mutual_info_score",
    "normalized_mutual_info_score",
    "adjusted_rand_score",
    "rand_score",
    "completeness_score",
    "pair_confusion_matrix",
    "contingency_matrix",
    "expected_mutual_information",
    "homogeneity_completeness_v_measure",
    "homogeneity_score",
    "mutual_info_score",
    "v_measure_score",
    "fowlkes_mallows_score",
    "entropy",
    "silhouette_samples",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "consensus_score",
]
```