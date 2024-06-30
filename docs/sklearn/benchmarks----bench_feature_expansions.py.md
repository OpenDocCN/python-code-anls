# `D:\src\scipysrc\scikit-learn\benchmarks\bench_feature_expansions.py`

```
from time import time  # 导入时间模块中的time函数，用于计时

import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块，用于绘图
import numpy as np  # 导入NumPy库，用于数值计算
import scipy.sparse as sparse  # 导入SciPy库中的稀疏矩阵模块

from sklearn.preprocessing import PolynomialFeatures  # 从sklearn库中的preprocessing模块导入PolynomialFeatures类，用于生成多项式特征

degree = 2  # 定义多项式特征的阶数为2
trials = 3  # 定义试验次数为3次
num_rows = 1000  # 定义数据集的行数为1000行
dimensionalities = np.array([1, 2, 8, 16, 32, 64])  # 定义数据集的维度数组
densities = np.array([0.01, 0.1, 1.0])  # 定义数据集的密度数组
csr_times = {d: np.zeros(len(dimensionalities)) for d in densities}  # 创建一个字典，用于存储CSR格式运行时间
dense_times = {d: np.zeros(len(dimensionalities)) for d in densities}  # 创建一个字典，用于存储稠密格式运行时间
transform = PolynomialFeatures(
    degree=degree, include_bias=False, interaction_only=False
)  # 创建PolynomialFeatures对象，配置为不包含偏置项和只生成非交互特征

for trial in range(trials):  # 开始试验循环
    for density in densities:  # 遍历密度数组
        for dim_index, dim in enumerate(dimensionalities):  # 遍历维度数组的索引和值
            print(trial, density, dim)  # 打印当前试验次数、密度和维度
            X_csr = sparse.random(num_rows, dim, density).tocsr()  # 生成稀疏矩阵并转换为CSR格式
            X_dense = X_csr.toarray()  # 将稀疏矩阵转换为稠密数组
            # CSR格式计时开始
            t0 = time()  # 记录当前时间
            transform.fit_transform(X_csr)  # 对CSR格式数据进行多项式特征转换
            csr_times[density][dim_index] += time() - t0  # 计算并累加CSR格式运行时间
            # 稠密格式计时开始
            t0 = time()  # 记录当前时间
            transform.fit_transform(X_dense)  # 对稠密数组进行多项式特征转换
            dense_times[density][dim_index] += time() - t0  # 计算并累加稠密格式运行时间

csr_linestyle = (0, (3, 1, 1, 1, 1, 1))  # 定义CSR格式的线型，密集的长短点划线
dense_linestyle = (0, ())  # 定义稠密格式的线型，实线

fig, axes = plt.subplots(nrows=len(densities), ncols=1, figsize=(8, 10))  # 创建子图布局，根据密度数组的长度确定行数，1列，图像尺寸为8x10英寸
for density, ax in zip(densities, axes):  # 遍历密度数组和子图对象
    ax.plot(
        dimensionalities,
        csr_times[density] / trials,  # 绘制CSR格式的平均运行时间
        label="csr",  # 设置标签为"csr"
        linestyle=csr_linestyle,  # 设置线型为预定义的CSR线型
    )
    ax.plot(
        dimensionalities,
        dense_times[density] / trials,  # 绘制稠密格式的平均运行时间
        label="dense",  # 设置标签为"dense"
        linestyle=dense_linestyle,  # 设置线型为预定义的稠密格式线型
    )
    ax.set_title("density %0.2f, degree=%d, n_samples=%d" % (density, degree, num_rows))  # 设置子图标题
    ax.legend()  # 显示图例
    ax.set_xlabel("Dimensionality")  # 设置X轴标签
    ax.set_ylabel("Time (seconds)")  # 设置Y轴标签

plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图形
```