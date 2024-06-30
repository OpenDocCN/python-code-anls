# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_set_output.py`

```
"""
================================
Introducing the `set_output` API
================================

.. currentmodule:: sklearn

This example will demonstrate the `set_output` API to configure transformers to
output pandas DataFrames. `set_output` can be configured per estimator by calling
the `set_output` method or globally by setting `set_config(transform_output="pandas")`.
For details, see
`SLEP018 <https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep018/proposal.html>`__.
"""  # noqa

# %%
# 首先，我们加载鸢尾花数据集作为 DataFrame，以演示 `set_output` API。
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(as_frame=True, return_X_y=True)
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
# 显示训练集的前几行数据
X_train.head()

# %%
# 要配置如 `preprocessing.StandardScaler` 这样的估算器返回 DataFrames，调用 `set_output`。
# 这个功能需要安装 pandas。

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform="pandas")

scaler.fit(X_train)
# 对测试集数据进行标准化转换
X_test_scaled = scaler.transform(X_test)
# 显示转换后的数据前几行
X_test_scaled.head()

# %%
# `set_output` 可以在 `fit` 后调用，以在事后配置 `transform`。
scaler2 = StandardScaler()

scaler2.fit(X_train)
# 使用默认的输出类型进行转换，并显示其类型
X_test_np = scaler2.transform(X_test)
print(f"Default output type: {type(X_test_np).__name__}")

# 配置输出类型为 pandas 后再次进行转换
scaler2.set_output(transform="pandas")
X_test_df = scaler2.transform(X_test)
print(f"Configured pandas output type: {type(X_test_df).__name__}")

# %%
# 在 :class:`pipeline.Pipeline` 中，`set_output` 配置所有步骤以输出 DataFrames。
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    StandardScaler(), SelectPercentile(percentile=75), LogisticRegression()
)
clf.set_output(transform="pandas")
clf.fit(X_train, y_train)

# %%
# 管道中的每个转换器都配置为返回 DataFrames。这意味着最终的逻辑回归步骤包含输入的特征名称。
clf[-1].feature_names_in_

# %%
# .. note:: 如果使用 `set_params` 方法，转换器将被一个带有默认输出格式的新转换器替换。
clf.set_params(standardscaler=StandardScaler())
clf.fit(X_train, y_train)
clf[-1].feature_names_in_

# %%
# 要保持预期的行为，先在新转换器上使用 `set_output`。
scaler = StandardScaler().set_output(transform="pandas")
clf.set_params(standardscaler=scaler)
clf.fit(X_train, y_train)
clf[-1].feature_names_in_

# %%
# 接下来，我们加载泰坦尼克号数据集，以演示如何在 :class:`compose.ColumnTransformer` 和异构数据中使用 `set_output`。
from sklearn.datasets import fetch_openml

X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
# %%
# 使用 `set_output` API 可以通过 :func:`set_config` 全局配置，将 `transform_output` 设置为 `"pandas"`。
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 设置全局配置，输出转换为 Pandas DataFrame
set_config(transform_output="pandas")

# 创建数值特征处理管道，包括缺失值填充和标准化
num_pipe = make_pipeline(SimpleImputer(), StandardScaler())
# 定义数值列
num_cols = ["age", "fare"]
# 创建列转换器，将数值列用 num_pipe 处理，将分类列用 OneHotEncoder 处理
ct = ColumnTransformer(
    (
        ("numerical", num_pipe, num_cols),
        (
            "categorical",
            OneHotEncoder(
                sparse_output=False, drop="if_binary", handle_unknown="ignore"
            ),
            ["embarked", "sex", "pclass"],
        ),
    ),
    verbose_feature_names_out=False,
)
# 创建包含列转换器和 Logistic 回归的管道
clf = make_pipeline(ct, SelectPercentile(percentile=50), LogisticRegression())
# 在训练集上拟合模型
clf.fit(X_train, y_train)
# 在测试集上评估模型准确率
clf.score(X_test, y_test)

# %%
# 在全局配置下，所有转换器输出的都是 DataFrame。这使得我们可以轻松地使用对应的特征名绘制 Logistic 回归的系数。
import pandas as pd

# 获取 Logistic 回归模型
log_reg = clf[-1]
# 提取并排序系数
coef = pd.Series(log_reg.coef_.ravel(), index=log_reg.feature_names_in_)
# 按照系数大小绘制水平条形图
_ = coef.sort_values().plot.barh()

# %%
# 为了在下面演示 :func:`config_context` 的功能，首先将 `transform_output` 重置为默认值。
set_config(transform_output="default")

# %%
# 当使用 :func:`config_context` 配置输出类型时，关键是在调用 `transform` 或 `fit_transform` 时的配置。
# 在构造或拟合转换器时设置这些参数不会产生影响。
from sklearn import config_context

# 创建标准化器
scaler = StandardScaler()
# 在训练集上拟合标准化器
scaler.fit(X_train[num_cols])

# %%
# 在 config_context 上下文管理器中，`transform` 的输出将是一个 Pandas DataFrame。
with config_context(transform_output="pandas"):
    X_test_scaled = scaler.transform(X_test[num_cols])
# 显示标准化后的测试集前几行
X_test_scaled.head()

# %%
# 在上下文管理器之外，`transform` 的输出将是一个 NumPy 数组。
X_test_scaled = scaler.transform(X_test[num_cols])
# 显示标准化后的测试集前几行
X_test_scaled[:5]
```