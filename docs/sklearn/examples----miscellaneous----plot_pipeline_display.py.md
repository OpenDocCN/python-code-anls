# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_pipeline_display.py`

```
# %%
# Displaying a Pipeline with a Preprocessing Step and Classifier
################################################################################
# This section constructs a :class:`~sklearn.pipeline.Pipeline` with a preprocessing
# step, :class:`~sklearn.preprocessing.StandardScaler`, and classifier,
# :class:`~sklearn.linear_model.LogisticRegression`, and displays its visual
# representation.

from sklearn import set_config   # 导入设置配置函数
from sklearn.linear_model import LogisticRegression   # 导入逻辑回归模型
from sklearn.pipeline import Pipeline   # 导入Pipeline类
from sklearn.preprocessing import StandardScaler   # 导入标准化预处理器

steps = [
    ("preprocessing", StandardScaler()),   # 定义预处理步骤，使用标准化预处理器
    ("classifier", LogisticRegression()),   # 定义分类器步骤，使用逻辑回归分类器
]
pipe = Pipeline(steps)   # 创建Pipeline对象

# %%
# To visualize the diagram, the default is `display='diagram'`.
set_config(display="diagram")
pipe  # click on the diagram below to see the details of each step

# %%
# To view the text pipeline, change to `display='text'`.
set_config(display="text")
pipe

# %%
# Put back the default display
set_config(display="diagram")

# %%
# Displaying a Pipeline Chaining Multiple Preprocessing Steps & Classifier
################################################################################
# This section constructs a :class:`~sklearn.pipeline.Pipeline` with multiple
# preprocessing steps, :class:`~sklearn.preprocessing.PolynomialFeatures` and
# :class:`~sklearn.preprocessing.StandardScaler`, and a classifier step,
# :class:`~sklearn.linear_model.LogisticRegression`, and displays its visual
# representation.

from sklearn.linear_model import LogisticRegression   # 导入逻辑回归模型
from sklearn.pipeline import Pipeline   # 导入Pipeline类
from sklearn.preprocessing import PolynomialFeatures, StandardScaler   # 导入多项式特征生成器和标准化预处理器

steps = [
    ("standard_scaler", StandardScaler()),   # 定义标准化预处理步骤
    ("polynomial", PolynomialFeatures(degree=3)),   # 定义多项式特征生成步骤
    ("classifier", LogisticRegression(C=2.0)),   # 定义分类器步骤，使用参数C=2.0的逻辑回归分类器
]
pipe = Pipeline(steps)   # 创建Pipeline对象
pipe  # click on the diagram below to see the details of each step

# %%
# Displaying a Pipeline and Dimensionality Reduction and Classifier
################################################################################
# This section constructs a :class:`~sklearn.pipeline.Pipeline` with a
# dimensionality reduction step, :class:`~sklearn.decomposition.PCA`,
# a classifier, :class:`~sklearn.svm.SVC`, and displays its visual
# representation.

from sklearn.decomposition import PCA   # 导入PCA降维模型
from sklearn.pipeline import Pipeline   # 导入Pipeline类
from sklearn.svm import SVC   # 导入支持向量分类器模型

steps = [("reduce_dim", PCA(n_components=4)),   # 定义降维步骤，使用PCA降维至4个主成分
         ("classifier", SVC(kernel="linear"))]   # 定义分类器步骤，使用线性核的支持向量分类器
pipe = Pipeline(steps)   # 创建Pipeline对象
pipe  # click on the diagram below to see the details of each step
# 这里声明了一个名为 `pipe` 的变量，用于存储后续构建的Pipeline对象

# %%
# Displaying a Complex Pipeline Chaining a Column Transformer
################################################################################
# This section constructs a complex :class:`~sklearn.pipeline.Pipeline` with a
# :class:`~sklearn.compose.ColumnTransformer` and a classifier,
# :class:`~sklearn.linear_model.LogisticRegression`, and displays its visual
# representation.

import numpy as np
# 导入NumPy库，用于处理数值计算

from sklearn.compose import ColumnTransformer
# 导入ColumnTransformer类，用于处理特征转换

from sklearn.impute import SimpleImputer
# 导入SimpleImputer类，用于处理数据缺失值

from sklearn.linear_model import LogisticRegression
# 导入LogisticRegression类，用于逻辑回归分类器

from sklearn.pipeline import Pipeline, make_pipeline
# 导入Pipeline和make_pipeline类，用于构建数据处理流水线

from sklearn.preprocessing import OneHotEncoder, StandardScaler
# 导入OneHotEncoder和StandardScaler类，用于类别变量编码和数据标准化

numeric_preprocessor = Pipeline(
    steps=[
        ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)
# 创建处理数值特征的Pipeline对象，包括均值填充和标准化步骤

categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputation_constant",
            SimpleImputer(fill_value="missing", strategy="constant"),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# 创建处理类别特征的Pipeline对象，包括常数填充和独热编码步骤

preprocessor = ColumnTransformer(
    [
        ("categorical", categorical_preprocessor, ["state", "gender"]),
        ("numerical", numeric_preprocessor, ["age", "weight"]),
    ]
)
# 创建ColumnTransformer对象，按照指定列应用对应的数据处理Pipeline

pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
# 创建Pipeline对象，将数据处理Pipeline和LogisticRegression分类器连接起来

pipe  # click on the diagram below to see the details of each step
# 这里声明了一个名为 `pipe` 的Pipeline对象，用于存储后续构建的数据处理和分类流水线

# %%
# Displaying a Grid Search over a Pipeline with a Classifier
################################################################################
# This section constructs a :class:`~sklearn.model_selection.GridSearchCV`
# over a :class:`~sklearn.pipeline.Pipeline` with
# :class:`~sklearn.ensemble.RandomForestClassifier` and displays its visual
# representation.

import numpy as np
# 导入NumPy库，用于处理数值计算

from sklearn.compose import ColumnTransformer
# 导入ColumnTransformer类，用于处理特征转换

from sklearn.ensemble import RandomForestClassifier
# 导入RandomForestClassifier类，用于随机森林分类器

from sklearn.impute import SimpleImputer
# 导入SimpleImputer类，用于处理数据缺失值

from sklearn.model_selection import GridSearchCV
# 导入GridSearchCV类，用于进行网格搜索交叉验证

from sklearn.pipeline import Pipeline, make_pipeline
# 导入Pipeline和make_pipeline类，用于构建数据处理流水线

from sklearn.preprocessing import OneHotEncoder, StandardScaler
# 导入OneHotEncoder和StandardScaler类，用于类别变量编码和数据标准化

numeric_preprocessor = Pipeline(
    steps=[
        ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)
# 创建处理数值特征的Pipeline对象，包括均值填充和标准化步骤

categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputation_constant",
            SimpleImputer(fill_value="missing", strategy="constant"),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# 创建处理类别特征的Pipeline对象，包括常数填充和独热编码步骤

preprocessor = ColumnTransformer(
    [
        ("categorical", categorical_preprocessor, ["state", "gender"]),
        ("numerical", numeric_preprocessor, ["age", "weight"]),
    ]
)
# 创建ColumnTransformer对象，按照指定列应用对应的数据处理Pipeline

pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
)
# 创建Pipeline对象，包括数据处理Pipeline和随机森林分类器

param_grid = {
    "classifier__n_estimators": [200, 500],
    "classifier__max_features": ["auto", "sqrt", "log2"],
    # 定义随机森林分类器的参数网格


注释：
    # 定义参数网格中决策树最大深度的候选值
    "classifier__max_depth": [4, 5, 6, 7, 8],
    # 定义参数网格中决策树划分标准的候选值，可以是基尼系数或熵
    "classifier__criterion": ["gini", "entropy"],
}

# 这里是一个单独的代码块结束标志，表示前面的代码段已经完成。

grid_search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=1)
# 创建一个网格搜索对象GridSearchCV，用于执行带交叉验证的网格搜索
# pipe: 要执行的管道对象，param_grid: 参数网格，n_jobs: 并行作业数

grid_search  # click on the diagram below to see the details of each step
# 输出grid_search对象，提示用户点击下面的图表以查看每个步骤的详细信息
```