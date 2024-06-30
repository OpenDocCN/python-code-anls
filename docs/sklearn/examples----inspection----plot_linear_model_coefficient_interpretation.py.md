# `D:\src\scipysrc\scikit-learn\examples\inspection\plot_linear_model_coefficient_interpretation.py`

```
# %%
# 导入绘图库 matplotlib.pyplot，并简写为 plt
import matplotlib.pyplot as plt
# 导入数值计算库 numpy，并简写为 np
import numpy as np
# 导入数据处理库 pandas，并简写为 pd
import pandas as pd
# 导入科学计算库 scipy，并简写为 sp
import scipy as sp
# 导入数据可视化库 seaborn，并简写为 sns
import seaborn as sns

# %%
# 数据集: 薪水
# ------------------
#
# 从 `OpenML <http://openml.org/>`_ 获取数据。
# 注意设置参数 `as_frame` 为 True 将以 pandas dataframe 形式获取数据。
from sklearn.datasets import fetch_openml

# 使用 fetch_openml 函数获取 ID 为 534 的数据集，并将其作为 dataframe 存储在 survey 变量中
survey = fetch_openml(data_id=534, as_frame=True)

# %%
# 接下来，我们识别特征 `X` 和目标 `y`：列 WAGE 是我们的目标变量（即我们想要预测的变量）。
X = survey.data[survey.feature_names]
# 使用 describe 方法显示特征 `X` 的统计描述信息
X.describe(include="all")

# %%
# 注意数据集包含分类和数值变量。
# 在之后预处理数据集时需要考虑这一点。
X.head()

# %%
# 我们的预测目标：工资。
# 工资以每小时美元的浮点数形式描述。

# %%
# 目标变量 `y` 是我们要预测的工资值，以每小时美元为单位。
y = survey.target.values.ravel()
# 显示 survey.target 的前几行数据
survey.target.head()

# %%
# 将样本数据集分割为训练集和测试集。
# 导入 train_test_split 函数，用于将数据集 X 和 y 划分为训练集和测试集
from sklearn.model_selection import train_test_split

# 使用 train_test_split 函数将数据集 X 和 y 划分为训练集和测试集，设置随机种子为 42
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
# 首先，通过查看变量的分布和它们之间的成对关系来获取一些见解。
# 只使用数值变量。在以下图中，每个点代表一个样本。
#
#   .. _marginal_dependencies:
#
# 复制 X_train 数据集用于分析，并在第一列插入目标变量 "WAGE"
_ = sns.pairplot(train_dataset, kind="reg", diag_kind="kde")

# %%
# 仔细观察 WAGE 的分布发现它呈长尾分布。因此，我们应当取其对数，
# 使其近似成为正态分布（线性模型如岭回归或套索对正态分布的误差效果更佳）。
#
# WAGE 随 EDUCATION 的增加而增加。
# 注意，这里表示的 WAGE 和 EDUCATION 之间的依赖是边际依赖，
# 即描述了特定变量的行为，而不固定其他变量。
#
# 此外，EXPERIENCE 和 AGE 强烈线性相关。
#
# .. _the-pipeline:
#
# 机器学习流水线
# -----------------------------
#
# 设计机器学习流水线之前，首先手动检查我们正在处理的数据类型：
survey.data.info()

# %%
# 如前所述，数据集包含不同数据类型的列，我们需要对每种数据类型应用特定的预处理。
# 特别是，如果不将分类变量编码为整数，它们无法包含在线性模型中。
# 此外，为了避免将分类特征视为有序值，我们需要对其进行独热编码。
# 我们的预处理器将
#
# - 对非二进制分类变量进行独热编码（即每个分类生成一个列）；
# - 保持数值变量不变（初步方案，在分析数值归一化如何影响后可能会调整）。
#
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

categorical_columns = ["RACE", "OCCUPATION", "SECTOR", "MARR", "UNION", "SEX", "SOUTH"]
numerical_columns = ["EDUCATION", "EXPERIENCE", "AGE"]

# 创建列转换器，其中对非二进制分类列使用独热编码，保留其余列
preprocessor = make_column_transformer(
    (OneHotEncoder(drop="if_binary"), categorical_columns),
    remainder="passthrough",
    verbose_feature_names_out=False,  # 避免添加前缀作为预处理器名称
)

# %%
# 为了描述数据集作为线性模型，我们使用岭回归器，并将目标变量设置为 WAGE 的对数值。
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
# 导入make_pipeline函数，用于创建一个机器学习管道
from sklearn.pipeline import make_pipeline

# 创建一个机器学习管道模型，包括数据预处理器和转换目标回归器
model = make_pipeline(
    preprocessor,
    TransformedTargetRegressor(
        regressor=Ridge(alpha=1e-10), func=np.log10, inverse_func=sp.special.exp10
    ),
)

# %%
# 处理数据集
# ----------------------
#
# 首先，对模型进行拟合。

model.fit(X_train, y_train)

# %%
# 然后，我们检查计算模型的性能，绘制其在测试集上的预测结果，
# 并计算模型的中位绝对误差等指标。

from sklearn.metrics import PredictionErrorDisplay, median_absolute_error

# 计算训练集的中位绝对误差
mae_train = median_absolute_error(y_train, model.predict(X_train))

# 对测试集进行预测
y_pred = model.predict(X_test)

# 计算测试集的中位绝对误差
mae_test = median_absolute_error(y_test, y_pred)

# 存储模型评估指标
scores = {
    "MedAE on training set": f"{mae_train:.2f} $/hour",
    "MedAE on testing set": f"{mae_test:.2f} $/hour",
}

# %%
# 创建一个绘图空间，并根据预测结果绘制预测误差图，
# 同时显示模型评估指标。

_, ax = plt.subplots(figsize=(5, 5))
display = PredictionErrorDisplay.from_predictions(
    y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
)
ax.set_title("Ridge model, small regularization")
for name, score in scores.items():
    ax.plot([], [], " ", label=f"{name}: {score}")
ax.legend(loc="upper left")
plt.tight_layout()

# %%
# 从上图中可以看出，学习到的模型远非一个能够准确预测的好模型：
# 这一点在上图中显而易见，好的预测应该在黑色虚线上。
#
# 在接下来的部分，我们将解释模型的系数。
# 在解释时，我们应该牢记，我们得出的任何结论都是关于我们建立的模型，
# 而不是关于数据的真实生成过程。
#
# 解释系数：尺度的重要性
# ----------------------------------------
#
# 首先，我们可以查看已拟合回归器的系数值。
feature_names = model[:-1].get_feature_names_out()

# 创建一个DataFrame来显示回归器的系数
coefs = pd.DataFrame(
    model[-1].regressor_.coef_,
    columns=["Coefficients"],
    index=feature_names,
)

coefs

# %%
# AGE系数表示为“每年生活年龄增加1年减少0.030867美元/小时”，
# EDUCATION系数表示为“每年教育年限增加1年增加0.054699美元/小时”。
# 这种系数的表示方式有助于清楚地了解模型的实际预测。
# 另外，联合(UNION)或性别(SEX)等分类变量是无量纲数，取值为0或1。
# 它们的系数以美元/小时表示。因此，我们不能直接比较不同系数的大小，
# 因为这些特征具有不同的自然尺度和值范围。
# 如果我们绘制系数，这一点会更加明显。

coefs.plot.barh(figsize=(9, 7))
plt.title("Ridge model, small regularization")
plt.axvline(x=0, color=".5")
# 在图中添加垂直线，x轴位置为0，颜色为灰色

plt.xlabel("Raw coefficient values")
# 设置x轴标签为"Raw coefficient values"

plt.subplots_adjust(left=0.3)
# 调整子图布局，左侧空白部分宽度为0.3

# %%
# 实际上，从上面的图中可以看出决定工资（WAGE）最重要的因素是变量UNION，
# 尽管我们的直觉可能告诉我们像EXPERIENCE这样的变量应该影响更大。
#
# 通过系数图来衡量特征的重要性可能会误导，因为一些特征变化范围很小，
# 而像AGE这样的特征变化范围很大，相差数十年。
#
# 如果比较不同特征的标准偏差，这一点就很明显了。

X_train_preprocessed = pd.DataFrame(
    model[:-1].transform(X_train), columns=feature_names
)
# 将训练数据X_train进行预处理转换，并存储在DataFrame中，列名为feature_names

X_train_preprocessed.std(axis=0).plot.barh(figsize=(9, 7))
# 绘制水平条形图，显示各特征值的标准偏差，设置图形尺寸为(9, 7)
plt.title("Feature ranges")
# 设置图标题为"Feature ranges"
plt.xlabel("Std. dev. of feature values")
# 设置x轴标签为"Std. dev. of feature values"
plt.subplots_adjust(left=0.3)
# 调整子图布局，左侧空白部分宽度为0.3

# %%
# 将系数乘以相关特征的标准偏差可以将所有系数缩放到相同的单位。
# 正如我们将在:ref:`scaling_num`中看到的那样，这等效于将数值变量标准化为它们的标准偏差，
# 即:math:`y = \sum{coef_i \times X_i} = \sum{(coef_i \times std_i) \times (X_i / std_i)}`。
#
# 通过这种方式，我们强调了一个特征的方差越大，其对输出的系数权重就越大，其他条件不变。

coefs = pd.DataFrame(
    model[-1].regressor_.coef_ * X_train_preprocessed.std(axis=0),
    columns=["Coefficient importance"],
    index=feature_names,
)
# 计算并存储各特征系数乘以对应特征标准偏差后的值，存储在DataFrame中，列名为"Coefficient importance"，索引为特征名

coefs.plot(kind="barh", figsize=(9, 7))
# 绘制水平条形图，显示特征系数校正后的重要性，设置图形尺寸为(9, 7)
plt.xlabel("Coefficient values corrected by the feature's std. dev.")
# 设置x轴标签为"Coefficient values corrected by the feature's std. dev."
plt.title("Ridge model, small regularization")
# 设置图标题为"Ridge model, small regularization"
plt.axvline(x=0, color=".5")
# 在图中添加垂直线，x轴位置为0，颜色为灰色
plt.subplots_adjust(left=0.3)
# 调整子图布局，左侧空白部分宽度为0.3

# %%
# 现在系数已经被缩放，我们可以安全地进行比较。
#
# .. warning::
#
#   为什么上面的图表暗示年龄增长导致工资减少？为什么:ref:`initial pairplot
#   <marginal_dependencies>`告诉了相反的情况？
#
# 上面的图表告诉我们特定特征与目标之间的依赖关系，当所有其他特征保持不变时，即**条件依赖关系**。
# 当所有其他特征保持不变时，AGE的增加将导致WAGE的减少。相反，EXPERIENCE的增加将导致WAGE的增加。
# 此外，AGE、EXPERIENCE和EDUCATION是最影响模型的三个变量之一。
#
# 解释系数：对因果关系要保持谨慎
# ---------------------------------------------------------
#
# 线性模型是衡量统计关联的重要工具，但在提出因果关系时应保持谨慎，毕竟相关性并不总意味着因果关系。
# 这在社会科学中尤其困难，因为我们观察到的变量仅作为潜在因果过程的代理。
#
# 导入需要的库：RepeatedKFold 用于重复 K 折交叉验证，cross_validate 用于交叉验证模型
from sklearn.model_selection import RepeatedKFold, cross_validate

# 设定交叉验证的参数：n_splits=5 表示每次划分训练集和验证集时使用 5 折交叉验证，
# n_repeats=5 表示重复 5 次整个交叉验证过程，random_state=0 为随机种子，保证可重复性
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)

# 进行交叉验证并返回每个模型的估计器
cv_model = cross_validate(
    model,         # 待验证的模型
    X,             # 特征数据集
    y,             # 目标数据集
    cv=cv,         # 指定交叉验证策略
    return_estimator=True,  # 返回每折交叉验证的模型估计器
    n_jobs=2,      # 并行运行的作业数量
)

# 计算系数的变化性并存储在 DataFrame 中
coefs = pd.DataFrame(
    [
        # 计算每个模型的系数乘以标准化后的特征数据的标准差
        est[-1].regressor_.coef_ * est[:-1].transform(X.iloc[train_idx]).std(axis=0)
        for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(X, y))
    ],
    columns=feature_names,  # 列名为特征名
)

# %%
# 创建一个绘图窗口，设置大小为 9x7 英寸
plt.figure(figsize=(9, 7))

# 绘制条形图显示系数的重要性，水平方向
sns.stripplot(data=coefs, orient="h", palette="dark:k", alpha=0.5)

# 绘制箱线图显示系数的分布情况，水平方向，填充色为青色，饱和度为 0.5，whis=10 表示箱线图的范围
sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5, whis=10)

# 在 x=0 的位置画一条垂直虚线，颜色为灰色
plt.axvline(x=0, color=".5")

# 设置 x 轴标签为 "Coefficient importance"
plt.xlabel("Coefficient importance")

# 设置标题为 "Coefficient importance and its variability"
plt.title("Coefficient importance and its variability")

# 设置总标题为 "Ridge model, small regularization"
plt.suptitle("Ridge model, small regularization")

# 调整子图布局，左边界为 0.3
plt.subplots_adjust(left=0.3)

# %%
# 问题：相关变量的问题
# -----------------------------------
#
# AGE 和 EXPERIENCE 系数受强烈变异的影响，可能是由于这两个特征之间的共线性：
# 因为在数据中 AGE 和 EXPERIENCE 一起变化，它们的效果难以分离。
#
# 为了验证这一解释，我们绘制了 AGE 和 EXPERIENCE 系数的变异性。
#
# .. _covariation:
plt.ylabel("Age coefficient")  # 设置 y 轴标签为 "Age coefficient"
plt.xlabel("Experience coefficient")  # 设置 x 轴标签为 "Experience coefficient"
plt.grid(True)  # 打开网格线
plt.xlim(-0.4, 0.5)  # 设置 x 轴范围从 -0.4 到 0.5
plt.ylim(-0.4, 0.5)
plt.scatter(coefs["AGE"], coefs["EXPERIENCE"])
_ = plt.title("Co-variations of coefficients for AGE and EXPERIENCE across folds")


# 设置 y 轴的范围在 -0.4 到 0.5 之间
# 绘制散点图，显示 AGE 和 EXPERIENCE 的系数变化情况
# 设置图表标题为 "Co-variations of coefficients for AGE and EXPERIENCE across folds"



# %%
# 当 EXPERIENCE 系数为正时，AGE 系数为负，反之亦然。
#
# 为了进一步分析，我们移除其中一个特征，观察对模型稳定性的影响。

column_to_drop = ["AGE"]

cv_model = cross_validate(
    model,
    X.drop(columns=column_to_drop),
    y,
    cv=cv,
    return_estimator=True,
    n_jobs=2,
)


# %%
# 从代码中可以看出，我们移除了 AGE 这个特征，以检查其对模型稳定性的影响。
# 使用交叉验证评估模型在删除 AGE 特征后的效果，返回了每折交叉验证的估计器对象。

coefs = pd.DataFrame(
    [
        est[-1].regressor_.coef_
        * est[:-1].transform(X.drop(columns=column_to_drop).iloc[train_idx]).std(axis=0)
        for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(X, y))
    ],
    columns=feature_names[:-1],
)


# %%
plt.figure(figsize=(9, 7))
sns.stripplot(data=coefs, orient="h", palette="dark:k", alpha=0.5)
sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
plt.axvline(x=0, color=".5")
plt.title("Coefficient importance and its variability")
plt.xlabel("Coefficient importance")
plt.suptitle("Ridge model, small regularization, AGE dropped")
plt.subplots_adjust(left=0.3)


# %%
# 现在，经验系数的估计显示出显著降低的变异性。
# 在交叉验证期间训练的所有模型中，经验仍然是重要的。
#
# .. _scaling_num:
#
# 数值变量的预处理
# ---------------------------------
#
# 如前所述（参见“:ref:`the-pipeline`”），我们也可以选择在训练模型之前对数值进行缩放。
# 当我们对所有特征应用类似的正则化时，在岭回归中这可能会很有用。
# 重新定义预处理器以便减去均值并将变量缩放为单位方差。

from sklearn.preprocessing import StandardScaler

preprocessor = make_column_transformer(
    (OneHotEncoder(drop="if_binary"), categorical_columns),
    (StandardScaler(), numerical_columns),
)


# %%
# 模型将保持不变。

model = make_pipeline(
    preprocessor,
    TransformedTargetRegressor(
        regressor=Ridge(alpha=1e-10), func=np.log10, inverse_func=sp.special.exp10
    ),
)
model.fit(X_train, y_train)


# %%
# 再次检查模型的性能，例如模型的中位绝对误差和 R 平方系数。

mae_train = median_absolute_error(y_train, model.predict(X_train))
y_pred = model.predict(X_test)
mae_test = median_absolute_error(y_test, y_pred)
scores = {
    "MedAE on training set": f"{mae_train:.2f} $/hour",
    "MedAE on testing set": f"{mae_test:.2f} $/hour",
}

_, ax = plt.subplots(figsize=(5, 5))
display = PredictionErrorDisplay.from_predictions(
    y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
)
ax.set_title("Ridge model, small regularization")
for name, score in scores.items():
    ax.plot([], [], " ", label=f"{name}: {score}")
ax.legend(loc="upper left")
plt.tight_layout()


# %%
# 通过例子展示模型的性能，例如模型在训练集上的中位绝对误差和 R 平方系数。
# 绘制预测误差图，显示实际值与预测值的关系，同时显示性能评分。
# %%
# 对于系数分析，这次不需要进行缩放，因为在预处理步骤中已经执行了缩放。

coefs = pd.DataFrame(
    model[-1].regressor_.coef_,
    columns=["Coefficients importance"],
    index=feature_names,
)
coefs.plot.barh(figsize=(9, 7))
plt.title("Ridge model, small regularization, normalized variables")
plt.xlabel("Raw coefficient values")
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)

# %%
# 现在我们跨多个交叉验证折叠检查系数。与上面的例子类似，我们不需要通过特征值的标准差来缩放系数，
# 因为这个缩放已经在管道的预处理步骤中完成了。

cv_model = cross_validate(
    model,
    X,
    y,
    cv=cv,
    return_estimator=True,
    n_jobs=2,
)
coefs = pd.DataFrame(
    [est[-1].regressor_.coef_ for est in cv_model["estimator"]], columns=feature_names
)

# %%
plt.figure(figsize=(9, 7))
sns.stripplot(data=coefs, orient="h", palette="dark:k", alpha=0.5)
sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5, whis=10)
plt.axvline(x=0, color=".5")
plt.title("Coefficient variability")
plt.subplots_adjust(left=0.3)

# %%
# 结果与非标准化情况非常相似。
#
# 具有正则化的线性模型
# ------------------------------
#
# 在机器学习实践中，岭回归通常与显著的正则化一起使用。
#
# 在上述情况下，我们将正则化限制在很小的量上。正则化提高了问题的条件性，并减少了估计值的方差。
# :class:`~sklearn.linear_model.RidgeCV` 使用交叉验证来确定哪个正则化参数 (`alpha`) 的值
# 最适合预测。

from sklearn.linear_model import RidgeCV

alphas = np.logspace(-10, 10, 21)  # 通过交叉验证选择的alpha值
model = make_pipeline(
    preprocessor,
    TransformedTargetRegressor(
        regressor=RidgeCV(alphas=alphas),
        func=np.log10,
        inverse_func=sp.special.exp10,
    ),
)
model.fit(X_train, y_train)

# %%
# 首先我们检查选择了哪个 :math:`\alpha` 值。

model[-1].regressor_.alpha_

# %%
# 然后我们检查预测的质量。
mae_train = median_absolute_error(y_train, model.predict(X_train))
y_pred = model.predict(X_test)
mae_test = median_absolute_error(y_test, y_pred)
scores = {
    "MedAE on training set": f"{mae_train:.2f} $/hour",
    "MedAE on testing set": f"{mae_test:.2f} $/hour",
}

_, ax = plt.subplots(figsize=(5, 5))
display = PredictionErrorDisplay.from_predictions(
    y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
)
ax.set_title("Ridge model, optimum regularization")
for name, score in scores.items():
    ax.plot([], [], " ", label=f"{name}: {score}")
ax.legend(loc="upper left")
plt.tight_layout()

# %%
# 创建一个 DataFrame 对象，存储正则化模型的系数，以及它们对应的特征名字作为索引
coefs = pd.DataFrame(
    model[-1].regressor_.coef_,
    columns=["Coefficients importance"],  # 列名为 "Coefficients importance"
    index=feature_names,  # 索引为特征名字列表
)
# 绘制水平条形图，显示正则化岭回归模型的系数重要性
coefs.plot.barh(figsize=(9, 7))
plt.title("Ridge model, with regularization, normalized variables")  # 设置图表标题
plt.xlabel("Raw coefficient values")  # 设置 x 轴标签
plt.axvline(x=0, color=".5")  # 在 x=0 处绘制垂直参考线，颜色为灰色
plt.subplots_adjust(left=0.3)  # 调整子图布局，左侧空出 0.3

# %%
# 系数值有显著变化。
# AGE 和 EXPERIENCE 的系数都是正值，但它们现在对预测的影响较小。
#
# 正则化通过在模型中共享权重来减少相关变量对模型的影响，
# 因此单个变量不会有强烈的权重。
#
# 另一方面，正则化得到的权重更加稳定（参见 :ref:`ridge_regression` 用户指南部分）。
# 从通过数据扰动在交叉验证中得到的图表中可以看到这种增强的稳定性。
# 这张图表可以与 :ref:`之前的图表<covariation>` 进行比较。
cv_model = cross_validate(
    model,
    X,
    y,
    cv=cv,
    return_estimator=True,
    n_jobs=2,
)
# 创建 DataFrame 对象，存储交叉验证中每个估计器的系数，列名为特征名字
coefs = pd.DataFrame(
    [est[-1].regressor_.coef_ for est in cv_model["estimator"]], columns=feature_names
)

# %%
plt.ylabel("Age coefficient")  # 设置 y 轴标签为 "Age coefficient"
plt.xlabel("Experience coefficient")  # 设置 x 轴标签为 "Experience coefficient"
plt.grid(True)  # 显示网格线
plt.xlim(-0.4, 0.5)  # 设置 x 轴范围
plt.ylim(-0.4, 0.5)  # 设置 y 轴范围
# 绘制散点图，显示 AGE 和 EXPERIENCE 系数在不同交叉验证折叠中的协变化
plt.scatter(coefs["AGE"], coefs["EXPERIENCE"])
_ = plt.title("Co-variations of coefficients for AGE and EXPERIENCE across folds")  # 设置图表标题

# %%
# 稀疏系数的线性模型
# ------------------------
#
# 在数据集中考虑相关变量的另一种可能性是估计稀疏系数。
# 当我们在先前的岭回归估计中删除 AGE 列时，我们已经手动执行了这种操作。
#
# Lasso 模型（参见 :ref:`lasso` 用户指南部分）估计稀疏系数。
# :class:`~sklearn.linear_model.LassoCV` 应用交叉验证来确定哪个正则化参数（`alpha`）最适合模型估计。
from sklearn.linear_model import LassoCV

alphas = np.logspace(-10, 10, 21)  # 通过交叉验证选择的 alpha 值的范围
# 创建管道对象，包括预处理器和 TransformedTargetRegressor 包装的 LassoCV 模型
model = make_pipeline(
    preprocessor,
    TransformedTargetRegressor(
        regressor=LassoCV(alphas=alphas, max_iter=100_000),  # LassoCV 模型配置
        func=np.log10,  # 目标变量的变换函数
        inverse_func=sp.special.exp10,  # 目标变量的逆变换函数
    ),
)

_ = model.fit(X_train, y_train)  # 拟合模型

# %%
# 首先，验证所选的 :math:`\alpha` 值。

model[-1].regressor_.alpha_

# %%
# 然后，检查预测质量。

mae_train = median_absolute_error(y_train, model.predict(X_train))  # 计算训练集的中位绝对误差
y_pred = model.predict(X_test)  # 对测试集进行预测
mae_test = median_absolute_error(y_test, y_pred)  # 计算测试集的中位绝对误差
# 存储中位绝对误差的结果
scores = {
    "MedAE on training set": f"{mae_train:.2f} $/hour",
    "MedAE on testing set": f"{mae_test:.2f} $/hour",
}
_, ax = plt.subplots(figsize=(6, 6))
# 创建一个新的 subplot，大小为 6x6 英寸，将其赋值给变量 ax

display = PredictionErrorDisplay.from_predictions(
    y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
)
# 使用 y_test 和 y_pred 数据创建一个 PredictionErrorDisplay 对象，
# 并将其显示在之前创建的 ax subplot 上，设置散点图参数 alpha 为 0.5

ax.set_title("Lasso model, optimum regularization")
# 设置 subplot ax 的标题为 "Lasso model, optimum regularization"

for name, score in scores.items():
    ax.plot([], [], " ", label=f"{name}: {score}")
    # 对于 scores 字典中的每一项，绘制一个空的线条图，使用空格字符串来创建空线条
    # 在图例中显示每个名称和分数的标签

ax.legend(loc="upper left")
# 在左上角显示图例

plt.tight_layout()
# 调整 subplot 的布局，使其紧凑显示

# %%
# 对于我们的数据集，再次注意到模型的预测能力并不是很强。

coefs = pd.DataFrame(
    model[-1].regressor_.coef_,
    columns=["Coefficients importance"],
    index=feature_names,
)
# 创建一个包含模型系数的 DataFrame，使用最后一个步骤的回归器的系数，
# 列名为 "Coefficients importance"，行索引为特征名称

coefs.plot(kind="barh", figsize=(9, 7))
# 绘制水平条形图，图形大小为 9x7 英寸

plt.title("Lasso model, optimum regularization, normalized variables")
# 设置图形的标题为 "Lasso model, optimum regularization, normalized variables"

plt.axvline(x=0, color=".5")
# 在 x 轴上加一条垂直线，位置为 0，颜色为灰色

plt.subplots_adjust(left=0.3)
# 调整子图布局，左边距为 0.3

# %%
# Lasso 模型识别 AGE 和 EXPERIENCE 之间的相关性，并抑制其中一个以进行预测。
#
# 需要注意的是，已经被抑制的系数可能本身与结果相关：
# 模型选择抑制它们是因为它们除了其他特征外带来的信息少或没有。
# 此外，对于相关特征，这种选择是不稳定的，应谨慎解释。

cv_model = cross_validate(
    model,
    X,
    y,
    cv=cv,
    return_estimator=True,
    n_jobs=2,
)
# 使用交叉验证评估模型的性能，返回每个折叠的估计器

coefs = pd.DataFrame(
    [est[-1].regressor_.coef_ for est in cv_model["estimator"]], columns=feature_names
)
# 创建一个 DataFrame，包含每个交叉验证折叠中模型的系数，
# 列名为特征名称

# %%
plt.figure(figsize=(9, 7))
# 创建一个图形，大小为 9x7 英寸

sns.stripplot(data=coefs, orient="h", palette="dark:k", alpha=0.5)
# 绘制水平方向的散点图，使用 dark:k 调色板，透明度为 0.5

sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5, whis=100)
# 绘制水平方向的箱线图，颜色为青色，饱和度为 0.5，whis 参数设置为 100

plt.axvline(x=0, color=".5")
# 在 x 轴上加一条垂直线，位置为 0，颜色为灰色

plt.title("Coefficient variability")
# 设置图形的标题为 "Coefficient variability"

plt.subplots_adjust(left=0.3)
# 调整子图布局，左边距为 0.3

# %%
# 我们观察到 AGE 和 EXPERIENCE 的系数在不同的折叠中变化很大。
#
# 错误的因果解释
# ---------------------------
#
# 政策制定者可能希望知道教育对工资的影响，以评估鼓励人们追求更多教育的某些政策是否经济上合理。
# 虽然机器学习模型非常适合测量统计关联，但它们通常无法推断因果效应。
#
# 可能会诱人的是，查看我们最后一个模型（或任何模型）中教育对工资的系数，并得出它捕捉了
# 标准化教育变量对工资的真实影响的结论。
#
# 不幸的是，可能存在未观察到的混杂变量，这些变量会使该系数膨胀或减小。
# 混杂变量是同时导致教育和工资的变量。其中一个例子是能力。
# 据推测，能力更强的人更有可能追求教育，同时更有可能在任何教育水平下赚取更高的时薪。
# 在这种情况下，能力引起了正的“遗漏变量偏差”。
# <https://en.wikipedia.org/wiki/Omitted-variable_bias>`_ (OVB) on the EDUCATION
# coefficient, thereby exaggerating the effect of education on wages.
#
# See the :ref:`sphx_glr_auto_examples_inspection_plot_causal_interpretation.py`
# for a simulated case of ability OVB.
#
# Lessons learned
# ---------------
#
# * Coefficients must be scaled to the same unit of measure to retrieve
#   feature importance. Scaling them with the standard-deviation of the
#   feature is a useful proxy.
# * Interpreting causality is difficult when there are confounding effects. If
#   the relationship between two variables is also affected by something
#   unobserved, we should be careful when making conclusions about causality.
# * Coefficients in multivariate linear models represent the dependency
#   between a given feature and the target, **conditional** on the other
#   features.
# * Correlated features induce instabilities in the coefficients of linear
#   models and their effects cannot be well teased apart.
# * Different linear models respond differently to feature correlation and
#   coefficients could significantly vary from one another.
# * Inspecting coefficients across the folds of a cross-validation loop
#   gives an idea of their stability.
# * Coefficients are unlikely to have any causal meaning. They tend
#   to be biased by unobserved confounders.
# * Inspection tools may not necessarily provide insights on the true
#   data generating process.
```