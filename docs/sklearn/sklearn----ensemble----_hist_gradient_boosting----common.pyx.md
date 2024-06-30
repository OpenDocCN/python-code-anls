# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\common.pyx`

```
# 导入NumPy库，用于处理数值计算
import numpy as np

# Y_DTYPE是目标变量y转换的数据类型，也是叶子节点值、增益、梯度/海森值之和的数据类型。
# 使用np.float64以确保足够的浮点精度。
Y_DTYPE = np.float64

# X_DTYPE是特征变量X的数据类型，使用np.float64以确保足够的浮点精度。
X_DTYPE = np.float64

# X_BINNED_DTYPE是特征经过分箱后的数据类型，使用np.uint8，因此最大分箱数为256。
X_BINNED_DTYPE = np.uint8

# G_H_DTYPE是梯度和海森值数组的数据类型，使用np.float32以节省内存空间。
G_H_DTYPE = np.float32

# X_BITSET_INNER_DTYPE是特征的位集合内部表示的数据类型，使用np.uint32。
X_BITSET_INNER_DTYPE = np.uint32

# HISTOGRAM_DTYPE是直方图数据的结构化数据类型，包括sum_gradients（在箱中的样本梯度总和）、
# sum_hessians（在箱中的样本海森总和）和count（在箱中的样本数）。
HISTOGRAM_DTYPE = np.dtype([
    ('sum_gradients', Y_DTYPE),  # 在箱中样本梯度的总和
    ('sum_hessians', Y_DTYPE),   # 在箱中样本海森的总和
    ('count', np.uint32),        # 在箱中的样本数
])

# PREDICTOR_RECORD_DTYPE是预测器记录的结构化数据类型，包括值、样本数、特征索引、
# 阈值数量、缺失值处理、左右子节点、增益、深度、是否叶子节点、分箱阈值、是否分类等信息。
PREDICTOR_RECORD_DTYPE = np.dtype([
    ('value', Y_DTYPE),             # 预测值
    ('count', np.uint32),           # 样本数
    ('feature_idx', np.intp),       # 特征索引
    ('num_threshold', X_DTYPE),     # 阈值数量
    ('missing_go_to_left', np.uint8),  # 缺失值处理，向左子节点
    ('left', np.uint32),            # 左子节点
    ('right', np.uint32),           # 右子节点
    ('gain', Y_DTYPE),              # 增益
    ('depth', np.uint32),           # 深度
    ('is_leaf', np.uint8),          # 是否叶子节点
    ('bin_threshold', X_BINNED_DTYPE),  # 分箱阈值
    ('is_categorical', np.uint8),   # 是否是分类特征
    ('bitset_idx', np.uint32)       # 位集合数组中的索引，仅在is_categorical为True时使用
])

# ALMOST_INF定义为1e300，用于LightGBM中的避免无穷大值。
ALMOST_INF = 1e300  # 参见LightGBM中的AvoidInf()
```