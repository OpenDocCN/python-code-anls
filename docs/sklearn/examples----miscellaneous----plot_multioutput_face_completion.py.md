# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_multioutput_face_completion.py`

```
"""
==============================================
Face completion with a multi-output estimators
==============================================

This example shows the use of multi-output estimator to complete images.
The goal is to predict the lower half of a face given its upper half.

The first column of images shows true faces. The next columns illustrate
how extremely randomized trees, k nearest neighbors, linear
regression and ridge regression complete the lower half of those faces.

"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入用于绘图的 matplotlib 库
import numpy as np  # 导入处理数组和矩阵的 numpy 库

from sklearn.datasets import fetch_olivetti_faces  # 从 sklearn 中导入加载 Olivetti Faces 数据集的函数
from sklearn.ensemble import ExtraTreesRegressor  # 从 sklearn 中导入极端随机树回归器
from sklearn.linear_model import LinearRegression, RidgeCV  # 导入线性回归和岭回归的函数
from sklearn.neighbors import KNeighborsRegressor  # 导入 k 近邻回归器
from sklearn.utils.validation import check_random_state  # 从 sklearn 中导入用于生成随机状态的函数

# 加载人脸数据集
data, targets = fetch_olivetti_faces(return_X_y=True)

train = data[targets < 30]  # 选择训练集，仅包含目标标签小于 30 的数据
test = data[targets >= 30]  # 选择测试集，仅包含目标标签大于等于 30 的数据，即独立的人物

# 在测试集中选择一个子集进行测试
n_faces = 5
rng = check_random_state(4)  # 使用随机状态生成器创建随机数生成器对象
face_ids = rng.randint(test.shape[0], size=(n_faces,))
test = test[face_ids, :]

n_pixels = data.shape[1]  # 获取每张人脸图像的像素数量
# 人脸的上半部分作为训练数据
X_train = train[:, : (n_pixels + 1) // 2]
# 人脸的下半部分作为训练目标
y_train = train[:, n_pixels // 2 :]
# 测试集中人脸的上半部分作为测试数据
X_test = test[:, : (n_pixels + 1) // 2]
# 测试集中人脸的下半部分作为测试目标
y_test = test[:, n_pixels // 2 :]

# 初始化各个估计器
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(
        n_estimators=10, max_features=32, random_state=0
    ),  # 极端随机树回归器
    "K-nn": KNeighborsRegressor(),  # k 近邻回归器
    "Linear regression": LinearRegression(),  # 线性回归
    "Ridge": RidgeCV(),  # 岭回归
}

y_test_predict = dict()  # 创建一个空字典用于存储预测结果
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)  # 使用训练数据训练估计器
    y_test_predict[name] = estimator.predict(X_test)  # 对测试数据进行预测并存储结果

# 绘制完整的人脸图像
image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)  # 列数为估计器数量加上真实人脸的列
plt.figure(figsize=(2.0 * n_cols, 2.26 * n_faces))  # 设置绘图窗口的大小
plt.suptitle("Face completion with multi-output estimators", size=16)  # 设置总标题

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))  # 将真实人脸的上下部分合并为一张完整的人脸图像

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")

    sub.axis("off")  # 关闭坐标轴
    sub.imshow(
        true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest"
    )  # 绘制真实人脸图像

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))  # 将预测结果的上下部分合并为一张完整的人脸图像

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)

        sub.axis("off")  # 关闭坐标轴
        sub.imshow(
            completed_face.reshape(image_shape),
            cmap=plt.cm.gray,
            interpolation="nearest",
        )  # 绘制预测结果的人脸图像

plt.show()  # 显示绘制的所有图像
```