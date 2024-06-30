# `D:\src\scipysrc\scikit-learn\examples\inspection\plot_permutation_importance.py`

```
# %%
# Data Loading and Feature Engineering
# ------------------------------------
# 加载数据并进行特征工程处理
# ------------------------------------
# 使用 pandas 加载泰坦尼克号数据集的副本。下面展示了如何对数值和分类特征分别进行预处理。
#
# 我们进一步包括两个与目标变量（"survived"）无关的随机变量：
#
# - "random_num" 是一个高基数数值变量（唯一值数量与记录数相同）。
# - "random_cat" 是一个低基数分类变量（有 3 个可能的取值）。
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 从 OpenML 加载泰坦尼克号数据集
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# 设置随机种子
rng = np.random.RandomState(seed=42)

# 添加随机生成的两个变量到特征集中
X["random_cat"] = rng.randint(3, size=X.shape[0])
X["random_num"] = rng.randn(X.shape[0])

# 指定分类和数值列
categorical_columns = ["pclass", "sex", "embarked", "random_cat"]
numerical_columns = ["age", "sibsp", "parch", "fare", "random_num"]

# 从特征集中选择所需的分类和数值列
X = X[categorical_columns + numerical_columns]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# %%
# Model Definition and Preprocessing
# ----------------------------------
# 定义预测模型并进行预处理
# ----------------------------------
# 我们基于随机森林定义一个预测模型。因此，我们将进行以下预处理步骤：
#
# - 使用 :class:`~sklearn.preprocessing.OrdinalEncoder` 对分类特征进行编码；
# - 使用 :class:`~sklearn.impute.SimpleImputer` 采用均值策略填充数值特征的缺失值。
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# 设置分类特征的编码器
categorical_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
)

# 设置数值特征的填充器
numerical_pipe = SimpleImputer(strategy="mean")
# 创建预处理流水线，包括类别特征编码器和数值特征处理管道
preprocessing = ColumnTransformer(
    [
        ("cat", categorical_encoder, categorical_columns),  # 对类别特征进行编码
        ("num", numerical_pipe, numerical_columns),  # 对数值特征进行处理
    ],
    verbose_feature_names_out=False,  # 禁用详细特征名称输出
)

# 创建随机森林分类器的管道，包括预处理和分类器部分
rf = Pipeline(
    [
        ("preprocess", preprocessing),  # 使用预处理流水线
        ("classifier", RandomForestClassifier(random_state=42)),  # 使用随机森林分类器
    ]
)

# 使用训练数据拟合随机森林模型
rf.fit(X_train, y_train)

# %%
# 模型的准确率
# ---------------------
# 在检查特征重要性之前，首先检查模型的预测性能是否足够高。确实，检查一个非预测性模型的重要特征将没有什么意义。
#
# 可以观察到训练准确率非常高（随机森林模型具有足够的能力完全记住训练集），但它仍然能够通过随机森林的内置装袋机制很好地泛化到测试集。
#
# 通过限制树的能力（例如设置 ``min_samples_leaf=5`` 或 ``min_samples_leaf=10``），可以在训练集上牺牲一些准确性，以在测试集上略微提高准确性，这样可以限制过拟合而不引入过多的欠拟合。
#
# 然而，为了展示具有许多唯一值的变量的一些特征重要性的缺陷，我们暂时保留高容量的随机森林模型。
print(f"RF train accuracy: {rf.score(X_train, y_train):.3f}")
print(f"RF test accuracy: {rf.score(X_test, y_test):.3f}")


# %%
# 基于不纯度减少（MDI）的树特征重要性
# --------------------------------------------------------------
# 基于不纯度的特征重要性将数值特征排名为最重要的特征。因此，非预测性的 ``random_num`` 变量被排名为最重要的特征之一！
#
# 这个问题源于基于不纯度的特征重要性的两个限制：
#
# - 基于不纯度的重要性偏向于高基数特征；
# - 基于不纯度的重要性是基于训练集统计数据计算的，因此不反映特征对于生成泛化到测试集的预测的能力（当模型具有足够容量时）。
#
# 对高基数特征的偏向性解释了为什么 `random_num` 的重要性远远大于 `random_cat`，而我们期望两个随机特征的重要性为零。
#
# 我们使用训练集统计数据的事实解释了为什么 `random_num` 和 `random_cat` 特征都具有非零的重要性。
import pandas as pd

# 获取特征名称
feature_names = rf[:-1].get_feature_names_out()

# 使用MDI计算特征重要性并排序
mdi_importances = pd.Series(
    rf[-1].feature_importances_, index=feature_names
).sort_values(ascending=True)

# %%
# 创建水平条形图显示MDI的随机森林特征重要性
ax = mdi_importances.plot.barh()
ax.set_title("Random Forest Feature Importances (MDI)")
ax.figure.tight_layout()

# %%
# 作为替代方案，计算 ``rf`` 的排列重要性
# 导入 permutation_importance 函数从 sklearn.inspection 模块
from sklearn.inspection import permutation_importance

# 计算在测试集上的排列重要性，包括模型预测准确率的下降情况
result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)

# 按照平均重要性排序特征索引
sorted_importances_idx = result.importances_mean.argsort()

# 创建一个 DataFrame 显示特征的重要性
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)

# 绘制特征重要性的箱线图
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()

# %%
# 在训练集上计算排列重要性，显示随机数值和分类特征的重要性
result = permutation_importance(
    rf, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
)

# 按照平均重要性排序特征索引
sorted_importances_idx = result.importances_mean.argsort()

# 创建一个 DataFrame 显示特征的重要性
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)

# 绘制特征重要性的箱线图
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (train set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()

# %%
# 尝试通过设置 `min_samples_leaf` 参数为 20 来限制树的过拟合能力
rf.set_params(classifier__min_samples_leaf=20).fit(X_train, y_train)

# %%
# 观察训练集和测试集上的准确率分数，验证模型是否过拟合
print(f"RF train accuracy: {rf.score(X_train, y_train):.3f}")
print(f"RF test accuracy: {rf.score(X_test, y_test):.3f}")

# %%
# 计算使用新模型的训练集上的排列重要性
train_result = permutation_importance(
    rf, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
)

# 计算使用新模型的测试集上的排列重要性
test_results = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)

# 按照平均重要性排序特征索引
sorted_importances_idx = train_result.importances_mean.argsort()

# %%
# 创建训练集上特征重要性的 DataFrame
train_importances = pd.DataFrame(
    train_result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)

# 创建测试集上特征重要性的 DataFrame
test_importances = pd.DataFrame(
    test_results.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)

# %%
# 遍历两个列表的元素，分别为 ["train", "test"] 和 [train_importances, test_importances]
for name, importances in zip(["train", "test"], [train_importances, test_importances]):
    # 使用 importances 对象的 plot 方法创建箱线图，水平方向显示，箱线图须须延伸至10个IQR之外
    ax = importances.plot.box(vert=False, whis=10)
    # 设置图表标题，包含当前数据集的名称（train 或 test）
    ax.set_title(f"Permutation Importances ({name} set)")
    # 设置 x 轴标签，表示精度分数的降低
    ax.set_xlabel("Decrease in accuracy score")
    # 在图上绘制垂直于 x 轴的虚线，位置为 x=0，颜色为黑色，线型为虚线
    ax.axvline(x=0, color="k", linestyle="--")
    # 调整图表布局，使得元素紧凑显示
    ax.figure.tight_layout()

# %%
# 现在，我们可以观察到在两个数据集上，`random_num` 和 `random_cat`
# 特征的重要性都低于过拟合的随机森林模型。然而，关于其他特征重要性的结论
# 仍然是有效的。
```