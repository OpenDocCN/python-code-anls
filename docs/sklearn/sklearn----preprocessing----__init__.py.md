# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\__init__.py`

```
"""
Methods for scaling, centering, normalization, binarization, and more.
"""

# 从 _data 模块导入需要的类和函数
from ._data import (
    Binarizer,               # 二值化器
    KernelCenterer,          # 核心心中心化器
    MaxAbsScaler,            # 最大绝对值缩放器
    MinMaxScaler,            # 最小-最大缩放器
    Normalizer,              # 归一化器
    PowerTransformer,        # 幂变换器
    QuantileTransformer,     # 分位数变换器
    RobustScaler,            # 鲁棒缩放器
    StandardScaler,          # 标准缩放器
    add_dummy_feature,       # 添加虚拟特征
    binarize,                # 二值化函数
    maxabs_scale,            # 最大绝对值缩放函数
    minmax_scale,            # 最小-最大缩放函数
    normalize,               # 标准化函数
    power_transform,         # 幂转换函数
    quantile_transform,      # 分位数转换函数
    robust_scale,            # 鲁棒缩放函数
    scale,                   # 缩放函数
)

# 从 _discretization 模块导入 KBinsDiscretizer 类
from ._discretization import KBinsDiscretizer

# 从 _encoders 模块导入 OneHotEncoder 和 OrdinalEncoder 类
from ._encoders import OneHotEncoder, OrdinalEncoder

# 从 _function_transformer 模块导入 FunctionTransformer 类
from ._function_transformer import FunctionTransformer

# 从 _label 模块导入 LabelBinarizer、LabelEncoder、MultiLabelBinarizer 类以及 label_binarize 函数
from ._label import LabelBinarizer, LabelEncoder, MultiLabelBinarizer, label_binarize

# 从 _polynomial 模块导入 PolynomialFeatures 和 SplineTransformer 类
from ._polynomial import PolynomialFeatures, SplineTransformer

# 从 _target_encoder 模块导入 TargetEncoder 类
from ._target_encoder import TargetEncoder

# 定义模块中公开的所有对象列表
__all__ = [
    "Binarizer",              # 二值化器类
    "FunctionTransformer",    # 函数转换器类
    "KBinsDiscretizer",       # K 段离散化器类
    "KernelCenterer",         # 核心中心化器类
    "LabelBinarizer",         # 标签二值化器类
    "LabelEncoder",           # 标签编码器类
    "MultiLabelBinarizer",    # 多标签二值化器类
    "MinMaxScaler",           # 最小-最大缩放器类
    "MaxAbsScaler",           # 最大绝对值缩放器类
    "QuantileTransformer",    # 分位数变换器类
    "Normalizer",             # 归一化器类
    "OneHotEncoder",          # 独热编码器类
    "OrdinalEncoder",         # 有序编码器类
    "PowerTransformer",       # 幂变换器类
    "RobustScaler",           # 鲁棒缩放器类
    "SplineTransformer",      # 样条变换器类
    "StandardScaler",         # 标准缩放器类
    "TargetEncoder",          # 目标编码器类
    "add_dummy_feature",      # 添加虚拟特征函数
    "PolynomialFeatures",     # 多项式特征生成器类
    "binarize",               # 二值化函数
    "normalize",              # 标准化函数
    "scale",                  # 缩放函数
    "robust_scale",           # 鲁棒缩放函数
    "maxabs_scale",           # 最大绝对值缩放函数
    "minmax_scale",           # 最小-最大缩放函数
    "label_binarize",         # 标签二值化函数
    "quantile_transform",     # 分位数转换函数
    "power_transform",        # 幂转换函数
]
```