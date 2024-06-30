# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_display_object_visualization.py`

```
# %%
# Load Data and train model
# -------------------------
# For this example, we load a blood transfusion service center data set from
# `OpenML <https://www.openml.org/d/1464>`. This is a binary classification
# problem where the target is whether an individual donated blood. Then the
# data is split into a train and test dataset and a logistic regression is
# fitted with the train dataset.
from sklearn.datasets import fetch_openml  # 导入用于获取数据集的函数
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.model_selection import train_test_split  # 导入数据集分割函数
from sklearn.pipeline import make_pipeline  # 导入管道构建函数
from sklearn.preprocessing import StandardScaler  # 导入数据标准化函数

X, y = fetch_openml(data_id=1464, return_X_y=True)  # 获取数据集并加载特征 X 和标签 y
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)  # 将数据集分割为训练集和测试集

clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))  # 创建包含数据标准化和逻辑回归的管道
clf.fit(X_train, y_train)  # 在训练集上拟合模型

# %%
# Create :class:`ConfusionMatrixDisplay`
##############################################################################
# With the fitted model, we compute the predictions of the model on the test
# dataset. These predictions are used to compute the confusion matrix which
# is plotted with the :class:`ConfusionMatrixDisplay`
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # 导入混淆矩阵相关函数

y_pred = clf.predict(X_test)  # 使用模型对测试集进行预测
cm = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵

cm_display = ConfusionMatrixDisplay(cm).plot()  # 创建并绘制混淆矩阵展示对象

# %%
# Create :class:`RocCurveDisplay`
##############################################################################
# The roc curve requires either the probabilities or the non-thresholded
# decision values from the estimator. Since the logistic regression provides
# a decision function, we will use it to plot the roc curve:
from sklearn.metrics import RocCurveDisplay, roc_curve  # 导入 ROC 曲线相关函数

y_score = clf.decision_function(X_test)  # 使用模型的决策函数计算预测分数

fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])  # 计算 ROC 曲线的假阳率和真阳率
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()  # 创建并绘制 ROC 曲线展示对象

# %%
# Create :class:`PrecisionRecallDisplay`
##############################################################################
# Similarly, the precision recall curve can be plotted using `y_score` from
# the previous sections.
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve  # 导入精确率-召回率曲线相关函数

prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])  # 计算精确率-召回率曲线的精确率和召回率
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()  # 创建并绘制精确率-召回率曲线展示对象

# %%
# Combining the display objects into a single plot
##############################################################################
# The display objects store the computed values that were passed as arguments.
# This allows for the visualizations to be easily combined using matplotlib's
# API. In the following example, we place the displays next to each other in a
# row.

# Import the matplotlib.pyplot module under the alias plt
import matplotlib.pyplot as plt

# Create a figure with two subplots arranged in a single row
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Plot the first display object (assumed to be roc_display) on the first subplot
roc_display.plot(ax=ax1)

# Plot the second display object (assumed to be pr_display) on the second subplot
pr_display.plot(ax=ax2)

# Display the combined plot with both subplots
plt.show()
```