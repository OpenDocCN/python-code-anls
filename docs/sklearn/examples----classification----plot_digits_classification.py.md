# `D:\src\scipysrc\scikit-learn\examples\classification\plot_digits_classification.py`

```
"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Standard scientific Python imports
import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm  # 导入数据集、分类器和性能度量工具
from sklearn.model_selection import train_test_split  # 导入数据集分割函数

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()  # 加载手写数字数据集

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))  # 创建一个包含 1 行 4 列子图的图形窗口
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()  # 关闭坐标轴
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")  # 显示灰度图像
    ax.set_title("Training: %i" % label)  # 设置子图标题为对应数字标签

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)  # 获取样本数量
data = digits.images.reshape((n_samples, -1))  # 将图像数据展平为一维数组

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)  # 创建支持向量机分类器对象，设置 gamma 值为 0.001

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)  # 将数据集分割为训练集和测试集，测试集占总数据集的 50%

# Learn the digits on the train subset
clf.fit(X_train, y_train)  # 在训练集上训练分类器模型

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)  # 使用训练好的模型对测试集进行预测

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))  # 创建一个包含 1 行 4 列子图的图形窗口
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()  # 关闭坐标轴
    image = image.reshape(8, 8)  # 将一维图像数据转换回二维数组形式
    # 在Axes对象ax上显示灰度图像，使用反转的灰度颜色映射，使用最近邻插值进行显示
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    # 设置图像标题，标题包含预测结果的信息
    ax.set_title(f"Prediction: {prediction}")
###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.
# 打印分类器的分类报告，显示主要的分类指标。

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
# 打印给定分类器的分类报告，包括其在测试集上的预测结果。

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.
# 我们也可以绘制一个真实数字值和预测数字值的混淆矩阵。

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# 使用预测结果和真实标签创建混淆矩阵展示对象

disp.figure_.suptitle("Confusion Matrix")
# 设置混淆矩阵图的标题为 "Confusion Matrix"

print(f"Confusion matrix:\n{disp.confusion_matrix}")
# 打印混淆矩阵的内容

plt.show()
# 展示混淆矩阵图形

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:
# 如果从评估分类器的结果以混淆矩阵的形式存储，而不是 `y_true` 和 `y_pred` 的形式，仍然可以如下构建分类报告。

# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
# 从混淆矩阵重建分类报告，输出基于混淆矩阵构建的分类指标报告。
```