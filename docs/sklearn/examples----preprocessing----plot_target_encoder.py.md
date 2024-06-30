# `D:\src\scipysrc\scikit-learn\examples\preprocessing\plot_target_encoder.py`

```
"""
============================================
Comparing Target Encoder with Other Encoders
============================================

.. currentmodule:: sklearn.preprocessing

The :class:`TargetEncoder` uses the value of the target to encode each
categorical feature. In this example, we will compare three different approaches
for handling categorical features: :class:`TargetEncoder`,
:class:`OrdinalEncoder`, :class:`OneHotEncoder` and dropping the category.

.. note::
    `fit(X, y).transform(X)` does not equal `fit_transform(X, y)` because a
    cross fitting scheme is used in `fit_transform` for encoding. See the
    :ref:`User Guide <target_encoder>` for details.
"""

# %%
# Loading Data from OpenML
# ========================
# First, we load the wine reviews dataset, where the target is the points given
# be a reviewer:
from sklearn.datasets import fetch_openml

wine_reviews = fetch_openml(data_id=42074, as_frame=True)

df = wine_reviews.frame
df.head()

# %%
# For this example, we use the following subset of numerical and categorical
# features in the data. The target are continuous values from 80 to 100:
numerical_features = ["price"]
categorical_features = [
    "country",
    "province",
    "region_1",
    "region_2",
    "variety",
    "winery",
]
target_name = "points"

X = df[numerical_features + categorical_features]
y = df[target_name]

_ = y.hist()

# %%
# Training and Evaluating Pipelines with Different Encoders
# =========================================================
# In this section, we will evaluate pipelines with
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` with different encoding
# strategies. First, we list out the encoders we will be using to preprocess
# the categorical features:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

categorical_preprocessors = [
    ("drop", "drop"),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    (
        "one_hot",
        OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False),
    ),
    ("target", TargetEncoder(target_type="continuous")),
]

# %%
# Next, we evaluate the models using cross validation and record the results:
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

n_cv_folds = 3
max_iter = 20
results = []


def evaluate_model_and_store(name, pipe):
    """
    Evaluate the specified pipeline using cross-validation and store the results.

    Parameters:
    - name (str): Name of the pipeline configuration.
    - pipe (Pipeline): Pipeline object containing preprocessing and model.

    This function computes the RMSE (root mean squared error) scores for both
    training and testing sets using cross-validation (3 folds) and stores the results.
    """
    result = cross_validate(
        pipe,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=n_cv_folds,
        return_train_score=True,
    )
    rmse_test_score = -result["test_score"]
    rmse_train_score = -result["train_score"]
    results.append(
        {
            "preprocessor": name,               # 添加预处理器名称到结果字典中
            "rmse_test_mean": rmse_test_score.mean(),   # 添加测试集均方根误差的平均值到结果字典中
            "rmse_test_std": rmse_train_score.std(),   # 添加测试集均方根误差的标准差到结果字典中（此处可能应为测试集的标准差）
            "rmse_train_mean": rmse_train_score.mean(),  # 添加训练集均方根误差的平均值到结果字典中
            "rmse_train_std": rmse_train_score.std(),    # 添加训练集均方根误差的标准差到结果字典中
        }
    )
# 遍历每个分类预处理器及其名称
for name, categorical_preprocessor in categorical_preprocessors:
    # 创建一个列转换器，将数值特征直接传递，将分类特征应用对应的预处理器
    preprocessor = ColumnTransformer(
        [
            ("numerical", "passthrough", numerical_features),  # 数值特征直接传递
            ("categorical", categorical_preprocessor, categorical_features),  # 应用分类预处理器
        ]
    )
    # 创建一个管道，依次应用列转换器和梯度增强回归器
    pipe = make_pipeline(
        preprocessor, HistGradientBoostingRegressor(random_state=0, max_iter=max_iter)
    )
    # 评估模型性能并保存结果
    evaluate_model_and_store(name, pipe)

# %%
# Native Categorical Feature Support
# ==================================
# In this section, we build and evaluate a pipeline that uses native categorical
# feature support in :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,
# which only supports up to 255 unique categories. In our dataset, the most of
# the categorical features have more than 255 unique categories:
# 计算每个分类特征的唯一值数量并按降序排序
n_unique_categories = df[categorical_features].nunique().sort_values(ascending=False)
n_unique_categories

# %%
# To workaround the limitation above, we group the categorical features into
# low cardinality and high cardinality features. The high cardinality features
# will be target encoded and the low cardinality features will use the native
# categorical feature in gradient boosting.
# 根据唯一值数量将分类特征分成低基数和高基数特征
high_cardinality_features = n_unique_categories[n_unique_categories > 255].index
low_cardinality_features = n_unique_categories[n_unique_categories <= 255].index
# 创建列转换器，用于组合数值特征、高基数特征（目标编码）和低基数特征（原生分类特征）
mixed_encoded_preprocessor = ColumnTransformer(
    [
        ("numerical", "passthrough", numerical_features),  # 数值特征直接传递
        (
            "high_cardinality",
            TargetEncoder(target_type="continuous"),  # 高基数特征目标编码
            high_cardinality_features,
        ),
        (
            "low_cardinality",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),  # 低基数特征使用原生分类特征
            low_cardinality_features,
        ),
    ],
    verbose_feature_names_out=False,  # 不输出详细特征名称
)

# The output of the of the preprocessor must be set to pandas so the
# gradient boosting model can detect the low cardinality features.
# 设置预处理器的输出为 pandas 格式，以便梯度增强模型能够识别低基数特征
mixed_encoded_preprocessor.set_output(transform="pandas")
# 创建管道，依次应用混合编码的列转换器和梯度增强回归器
mixed_pipe = make_pipeline(
    mixed_encoded_preprocessor,
    HistGradientBoostingRegressor(
        random_state=0, max_iter=max_iter, categorical_features=low_cardinality_features
    ),
)
mixed_pipe

# %%
# Finally, we evaluate the pipeline using cross validation and record the results:
# 最后，使用交叉验证评估管道并记录结果
evaluate_model_and_store("mixed_target", mixed_pipe)

# %%
# Plotting the Results
# ====================
# In this section, we display the results by plotting the test and train scores:
# 在此部分，通过绘制测试和训练分数来展示结果
import matplotlib.pyplot as plt
import pandas as pd

# 将评估结果转换为 DataFrame 并按预处理器排序
results_df = (
    pd.DataFrame(results).set_index("preprocessor").sort_values("rmse_test_mean")
)

# 创建包含两个子图的图形窗口
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 8), sharey=True, constrained_layout=True
)
xticks = range(len(results_df))
# 将预处理器名称映射到颜色
name_to_color = dict(
    zip((r["preprocessor"] for r in results), ["C0", "C1", "C2", "C3", "C4"])
)

# 针对测试集和训练集分别绘制结果子图
for subset, ax in zip(["test", "train"], [ax1, ax2]):
    # 根据给定的子集名称（subset）构造均值和标准差的列名
    mean, std = f"rmse_{subset}_mean", f"rmse_{subset}_std"
    
    # 从结果数据框中选择均值和标准差两列，并按均值列（mean）进行排序
    data = results_df[[mean, std]].sort_values(mean)
    
    # 在图形上绘制柱状图，设置柱子的参数包括位置（x轴ticks）、高度（均值）、误差（标准差）、宽度、颜色
    ax.bar(
        x=xticks,
        height=data[mean],
        yerr=data[std],
        width=0.9,
        color=[name_to_color[name] for name in data.index],
    )
    
    # 设置图形的标题、x轴标签、x轴刻度、x轴刻度标签
    ax.set(
        title=f"RMSE ({subset.title()})",  # 设置标题，标题显示包括子集名称
        xlabel="Encoding Scheme",  # 设置x轴标签
        xticks=xticks,  # 设置x轴刻度的位置
        xticklabels=data.index,  # 设置x轴刻度的标签内容为data的索引
    )
# %%
# 在评估测试集的预测性能时，删除表现最差的类别特征，而目标编码器表现最佳。这可以如下解释：
#
# - 删除分类特征使得管道的表达能力降低，因此容易出现欠拟合；
# - 由于高基数和为了减少训练时间，使用 `max_categories=20` 的独热编码方案，防止特征扩展过多，这可以避免欠拟合；
# - 如果没有设置 `max_categories=20`，独热编码方案可能会导致过拟合，因为特征数量会随着罕见类别的出现而爆炸增长，这些类别仅在训练集中与目标变量有偶然相关性；
# - 序数编码对特征施加了任意顺序，然后由 :class:`~sklearn.ensemble.HistGradientBoostingRegressor` 将其视为数值。由于这个模型将每个特征分组为 256 个箱，许多不相关的类别可能会被合并在一起，从而导致整体管道欠拟合；
# - 使用目标编码器时，也会进行相同的分箱处理，但由于编码后的值按照与目标变量的边际关联性排序，因此由 :class:`~sklearn.ensemble.HistGradientBoostingRegressor` 进行的分箱是合理的，并且导致良好的结果：平滑目标编码和分箱的结合作为一种良好的正则化策略，防止过拟合，同时不会过度限制管道的表达能力。
```