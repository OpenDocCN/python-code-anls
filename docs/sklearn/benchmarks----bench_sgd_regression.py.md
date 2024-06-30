# `D:\src\scipysrc\scikit-learn\benchmarks\bench_sgd_regression.py`

```
# 导入垃圾回收和计时功能
import gc
from time import time

# 导入 matplotlib 和 numpy 库
import matplotlib.pyplot as plt
import numpy as np

# 导入生成回归数据的函数和线性模型
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error

"""
SGD 回归的基准测试

比较 SGD 回归与坐标下降和 Ridge 回归在合成数据上的性能。
"""

# 打印文档字符串
print(__doc__)

if __name__ == "__main__":
    # 创建包含从 100 到 10000 之间的 5 个整数的数组
    list_n_samples = np.linspace(100, 10000, 5).astype(int)
    # 定义特征数的数组
    list_n_features = [10, 100, 1000]
    n_test = 1000  # 测试集大小
    max_iter = 1000  # 最大迭代次数
    noise = 0.1  # 噪声水平
    alpha = 0.01  # 正则化参数 alpha

    # 初始化各种回归结果的空数组
    sgd_results = np.zeros((len(list_n_samples), len(list_n_features), 2))
    elnet_results = np.zeros((len(list_n_samples), len(list_n_features), 2))
    ridge_results = np.zeros((len(list_n_samples), len(list_n_features), 2))
    asgd_results = np.zeros((len(list_n_samples), len(list_n_features), 2))

    # 绘制结果图表
    i = 0
    m = len(list_n_features)
    plt.figure("scikit-learn SGD regression benchmark results", figsize=(5 * 2, 4 * m))
    for j in range(m):
        # 绘制测试误差图表
        plt.subplot(m, 2, i + 1)
        plt.plot(list_n_samples, np.sqrt(elnet_results[:, j, 0]), label="ElasticNet")
        plt.plot(list_n_samples, np.sqrt(sgd_results[:, j, 0]), label="SGDRegressor")
        plt.plot(list_n_samples, np.sqrt(asgd_results[:, j, 0]), label="A-SGDRegressor")
        plt.plot(list_n_samples, np.sqrt(ridge_results[:, j, 0]), label="Ridge")
        plt.legend(prop={"size": 10})
        plt.xlabel("n_train")
        plt.ylabel("RMSE")
        plt.title("Test error - %d features" % list_n_features[j])
        i += 1

        # 绘制训练时间图表
        plt.subplot(m, 2, i + 1)
        plt.plot(list_n_samples, np.sqrt(elnet_results[:, j, 1]), label="ElasticNet")
        plt.plot(list_n_samples, np.sqrt(sgd_results[:, j, 1]), label="SGDRegressor")
        plt.plot(list_n_samples, np.sqrt(asgd_results[:, j, 1]), label="A-SGDRegressor")
        plt.plot(list_n_samples, np.sqrt(ridge_results[:, j, 1]), label="Ridge")
        plt.legend(prop={"size": 10})
        plt.xlabel("n_train")
        plt.ylabel("Time [sec]")
        plt.title("Training time - %d features" % list_n_features[j])
        i += 1

    # 调整子图之间的垂直间距
    plt.subplots_adjust(hspace=0.30)

    # 展示图表
    plt.show()
```