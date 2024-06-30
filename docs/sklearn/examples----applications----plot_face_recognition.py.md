# `D:\src\scipysrc\scikit-learn\examples\applications\plot_face_recognition.py`

```
# %%
# 导入所需的库和模块
from time import time  # 导入时间模块中的time函数

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot用于绘图
from scipy.stats import loguniform  # 导入scipy.stats中的loguniform分布

from sklearn.datasets import fetch_lfw_people  # 导入sklearn中的人脸数据集
from sklearn.decomposition import PCA  # 导入PCA主成分分析模块
from sklearn.metrics import ConfusionMatrixDisplay, classification_report  # 导入评估模型性能的相关模块
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # 导入随机搜索和数据集划分模块
from sklearn.preprocessing import StandardScaler  # 导入数据标准化模块
from sklearn.svm import SVC  # 导入支持向量机分类器

# %%
# 下载数据集（如果尚未下载）并加载为numpy数组

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 检查图像数组以获取形状信息（用于绘图）
n_samples, h, w = lfw_people.images.shape

# 对于机器学习，我们直接使用数据（忽略像素位置信息）
X = lfw_people.data
n_features = X.shape[1]

# 要预测的标签是人物的ID
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# %%
# 将数据集分割为训练集和测试集，保留25%的数据作为测试集

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# 计算PCA（特征脸）以进行面部数据集的无监督特征提取/降维

n_components = 150

print(
    "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
)
t0 = time()
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# %%
# 训练一个SVM分类模型

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}
clf = RandomizedSearchCV(
    SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# %%
# 在测试集上定量评估模型质量

print("Predicting people's names on the test set")
t0 = time()
# 使用分类器预测测试集上的结果
y_pred = clf.predict(X_test_pca)
# 打印预测完成所需的时间
print("done in %0.3fs" % (time() - t0))

# 打印分类报告，包括精确度、召回率、F1 值等信息
print(classification_report(y_test, y_pred, target_names=target_names))

# 使用 ConfusionMatrixDisplay 可视化分类器的混淆矩阵
ConfusionMatrixDisplay.from_estimator(
    clf, X_test_pca, y_test, display_labels=target_names, xticks_rotation="vertical"
)
# 调整图像布局
plt.tight_layout()
# 展示图像
plt.show()

# %%
# 使用 matplotlib 对预测结果进行定性评估


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    # 创建一个图像窗口，指定每个子图的大小和间距
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    # 遍历并绘制图像及其标题
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# %%
# 绘制测试集中部分样本的预测结果


def title(y_pred, y_test, target_names, i):
    # 根据索引 i 获取预测和真实标签的名字部分
    pred_name = target_names[y_pred[i]].rsplit(" ", 1)[-1]
    true_name = target_names[y_test[i]].rsplit(" ", 1)[-1]
    return "predicted: %s\ntrue:      %s" % (pred_name, true_name)


# 生成每个预测样本的标题
prediction_titles = [
    title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])
]

# 调用 plot_gallery 函数绘制预测结果的图像画廊
plot_gallery(X_test, prediction_titles, h, w)
# %%
# 绘制最重要的特征脸图像的画廊

# 创建特征脸的标题列表
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# 使用 plot_gallery 函数绘制特征脸图像
plot_gallery(eigenfaces, eigenface_titles, h, w)

# 展示图像
plt.show()

# %%
# 面部识别问题可以通过训练卷积神经网络更有效地解决，
# 但是这类模型超出了 scikit-learn 库的范围。
# 有兴趣的读者应该尝试使用 pytorch 或 tensorflow 来实现这些模型。
```