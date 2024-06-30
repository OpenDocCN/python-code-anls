# `D:\src\scipysrc\scikit-learn\examples\semi_supervised\plot_label_propagation_digits_active_learning.py`

```
"""
========================================
Label Propagation digits active learning
========================================

Demonstrates an active learning technique to learn handwritten digits
using label propagation.

We start by training a label propagation model with only 10 labeled points,
then we select the top five most uncertain points to label. Next, we train
with 15 labeled points (original 10 + 5 new ones). We repeat this process
four times to have a model trained with 30 labeled examples. Note you can
increase this to label more than 30 by changing `max_iterations`. Labeling
more than 30 can be useful to get a sense for the speed of convergence of
this active learning technique.

A plot will appear showing the top 5 most uncertain digits for each iteration
of training. These may or may not contain mistakes, but we will train the next
model with their true labels.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算
from scipy import stats  # 导入 scipy.stats 库，用于统计计算

from sklearn import datasets  # 导入 sklearn 中的数据集模块
from sklearn.metrics import classification_report, confusion_matrix  # 导入分类报告和混淆矩阵计算函数
from sklearn.semi_supervised import LabelSpreading  # 导入标签传播模型

digits = datasets.load_digits()  # 载入手写数字数据集
rng = np.random.RandomState(0)  # 创建随机数生成器对象 rng
indices = np.arange(len(digits.data))  # 创建数据集索引数组
rng.shuffle(indices)  # 打乱数据集索引顺序

X = digits.data[indices[:330]]  # 提取数据集特征数据
y = digits.target[indices[:330]]  # 提取数据集标签数据
images = digits.images[indices[:330]]  # 提取数据集图像数据

n_total_samples = len(y)  # 计算总样本数
n_labeled_points = 40  # 初始标记样本数
max_iterations = 5  # 最大迭代次数

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]  # 创建未标记样本索引数组
f = plt.figure()  # 创建图形对象 f，用于绘图

for i in range(max_iterations):
    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break
    y_train = np.copy(y)  # 复制标签数据到 y_train
    y_train[unlabeled_indices] = -1  # 将未标记样本的标签设置为 -1

    lp_model = LabelSpreading(gamma=0.25, max_iter=20)  # 创建标签传播模型对象 lp_model
    lp_model.fit(X, y_train)  # 使用 X 和 y_train 训练模型

    predicted_labels = lp_model.transduction_[unlabeled_indices]  # 预测未标记样本的标签
    true_labels = y[unlabeled_indices]  # 获取未标记样本的真实标签

    cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)  # 计算混淆矩阵

    print("Iteration %i %s" % (i, 70 * "_"))  # 打印迭代次数信息
    print(
        "Label Spreading model: %d labeled & %d unlabeled (%d total)"
        % (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples)
    )  # 打印模型的标记信息

    print(classification_report(true_labels, predicted_labels))  # 打印分类报告

    print("Confusion matrix")  # 打印混淆矩阵标题
    print(cm)  # 打印混淆矩阵内容

    # compute the entropies of transduced label distributions
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)  # 计算标签分布的熵

    # select up to 5 digit examples that the classifier is most uncertain about
    uncertainty_index = np.argsort(pred_entropies)[::-1]  # 根据熵值降序排列索引
    uncertainty_index = uncertainty_index[
        np.isin(uncertainty_index, unlabeled_indices)
    ][:5]  # 获取未标记样本中熵值最高的前五个索引

    # keep track of indices that we get labels for
    delete_indices = np.array([], dtype=int)  # 创建空数组用于记录获取标签的索引
    # 如果当前迭代次数 i 小于 5，则执行以下代码块
    if i < 5:
        # 在图形上绘制文本信息，指定位置和内容
        f.text(
            0.05,  # 横坐标位置
            (1 - (i + 1) * 0.183),  # 纵坐标位置，根据当前迭代次数计算
            "model %d\n\nfit with\n%d labels" % ((i + 1), i * 5 + 10),  # 显示的文本内容
            size=10,  # 文本字体大小
        )
    
    # 遍历 uncertainty_index 中的索引和对应的 image_index
    for index, image_index in enumerate(uncertainty_index):
        # 根据 image_index 获取对应的图像数据
        image = images[image_index]

        # 如果当前迭代次数 i 小于 5，则执行以下代码块
        if i < 5:
            # 添加子图到图形中，每行 5 列，计算子图位置
            sub = f.add_subplot(5, 5, index + 1 + (5 * i))
            # 在子图中显示灰度图像，指定颜色映射和插值方式
            sub.imshow(image, cmap=plt.cm.gray_r, interpolation="none")
            # 设置子图标题，显示预测结果和真实标签信息
            sub.set_title(
                "predict: %i\ntrue: %i"
                % (lp_model.transduction_[image_index], y[image_index]),  # 使用 transduction_ 和 y 的对应索引值
                size=10,  # 标题字体大小
            )
            # 关闭子图的坐标轴显示
            sub.axis("off")

        # 在 unlabeled_indices 中找到与 image_index 相同的索引，将其添加到 delete_indices
        (delete_index,) = np.where(unlabeled_indices == image_index)
        delete_indices = np.concatenate((delete_indices, delete_index))

    # 从 unlabeled_indices 中删除 delete_indices 中包含的索引
    unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
    # 增加已标记点的数量，加上 uncertainty_index 中的索引数量
    n_labeled_points += len(uncertainty_index)
# 设置图表的总标题，描述了使用标签传播的主动学习过程
# 标题包括两行文本：第一行描述主题，第二行指明每行显示最不确定的5个标签以便下一个模型学习
f.suptitle(
    (
        "Active learning with Label Propagation.\nRows show 5 most "
        "uncertain labels to learn with the next model."
    ),
    y=1.15,  # 将标题放置在图像顶部的偏移量
)

# 调整子图布局，以便在显示前进行视觉优化
plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2, hspace=0.85)

# 显示当前所有绘图
plt.show()
```