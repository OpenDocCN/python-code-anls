# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_incremental_pca.py`

```
# 导入绘图和数据处理库
import matplotlib.pyplot as plt
import numpy as np

# 导入鸢尾花数据集和PCA相关类
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 提取特征数据
y = iris.target  # 提取目标数据

# 指定主成分的数量
n_components = 2

# 创建IncrementalPCA对象，指定主成分数目和批处理大小
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
# 对数据进行IncrementalPCA降维处理
X_ipca = ipca.fit_transform(X)

# 创建PCA对象，指定主成分数目
pca = PCA(n_components=n_components)
# 对数据进行PCA降维处理
X_pca = pca.fit_transform(X)

# 定义用于绘图的颜色
colors = ["navy", "turquoise", "darkorange"]

# 遍历两种降维结果，分别绘制图像
for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    # 根据目标数据绘制散点图
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(
            X_transformed[y == i, 0],  # 提取第一个主成分数据
            X_transformed[y == i, 1],  # 提取第二个主成分数据
            color=color,
            lw=2,
            label=target_name,
        )

    # 根据标题判断是否为Incremental PCA，计算PCA和Incremental PCA之间的误差
    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error %.6f" % err)
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)  # 添加图例
    plt.axis([-4, 4, -1.5, 1.5])  # 设置坐标轴范围

plt.show()  # 显示图形
```