# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_estimator_representation.py`

```
"""
===========================================
Displaying estimators and complex pipelines
===========================================

This example illustrates different ways estimators and pipelines can be
displayed.
"""

from sklearn.compose import make_column_transformer  # 导入列转换器模块
from sklearn.impute import SimpleImputer  # 导入缺失值填充模块
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.pipeline import make_pipeline  # 导入创建管道模块
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入独热编码和标准化模块

# %%
# Compact text representation
# ---------------------------
#
# Estimators will only show the parameters that have been set to non-default
# values when displayed as a string. This reduces the visual noise and makes it
# easier to spot what the differences are when comparing instances.

lr = LogisticRegression(penalty="l1")  # 创建一个带有L1正则化惩罚的逻辑回归模型
print(lr)  # 打印逻辑回归模型的信息

# %%
# Rich HTML representation
# ------------------------
# In notebooks estimators and pipelines will use a rich HTML representation.
# This is particularly useful to summarise the
# structure of pipelines and other composite estimators, with interactivity to
# provide detail.  Click on the example image below to expand Pipeline
# elements.  See :ref:`visualizing_composite_estimators` for how you can use
# this feature.

num_proc = make_pipeline(  # 创建数值处理管道
    SimpleImputer(strategy="median"),  # 使用中位数填充缺失值
    StandardScaler()  # 标准化处理
)

cat_proc = make_pipeline(  # 创建分类处理管道
    SimpleImputer(strategy="constant", fill_value="missing"),  # 使用常数填充缺失值，填充值为"missing"
    OneHotEncoder(handle_unknown="ignore")  # 独热编码处理，未知值忽略
)

preprocessor = make_column_transformer(  # 创建列转换器
    (num_proc, ("feat1", "feat3")),  # 对feat1和feat3列应用数值处理管道
    (cat_proc, ("feat0", "feat2"))  # 对feat0和feat2列应用分类处理管道
)

clf = make_pipeline(preprocessor, LogisticRegression())  # 创建整体的管道，包括预处理和逻辑回归模型
clf  # 打印整体的管道结构信息
```