# `numpy-ml\numpy_ml\tests\test_nonparametric.py`

```
# 禁用 flake8 的警告
# 导入 numpy 库并重命名为 np
import numpy as np

# 从 sklearn 库中导入 KNeighborsRegressor 和 KNeighborsClassifier 类
# 从 sklearn.gaussian_process 库中导入 GaussianProcessRegressor 类
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

# 从 numpy_ml.nonparametric.knn 模块中导入 KNN 类
# 从 numpy_ml.nonparametric.gp 模块中导入 GPRegression 类
# 从 numpy_ml.utils.distance_metrics 模块中导入 euclidean 函数
from numpy_ml.nonparametric.knn import KNN
from numpy_ml.nonparametric.gp import GPRegression
from numpy_ml.utils.distance_metrics import euclidean

# 定义测试 KNN 回归的函数，参数 N 默认值为 15
def test_knn_regression(N=15):
    # 设置随机种子为 12345
    np.random.seed(12345)

    # 初始化循环计数器 i
    i = 0
    # 循环执行 N 次
    while i < N:
        # 生成随机数 N 和 M
        N = np.random.randint(2, 100)
        M = np.random.randint(2, 100)
        # 生成随机数 k，ls，weights
        k = np.random.randint(1, N)
        ls = np.min([np.random.randint(1, 10), N - 1])
        weights = np.random.choice(["uniform", "distance"])

        # 生成随机数据 X，X_test，y
        X = np.random.rand(N, M)
        X_test = np.random.rand(N, M)
        y = np.random.rand(N)

        # 创建 KNN 模型对象 knn
        knn = KNN(
            k=k, leaf_size=ls, metric=euclidean, classifier=False, weights=weights
        )
        # 使用 X，y 训练 KNN 模型
        knn.fit(X, y)
        # 对 X_test 进行预测
        preds = knn.predict(X_test)

        # 创建 sklearn 中的 KNeighborsRegressor 模型对象 gold
        gold = KNeighborsRegressor(
            p=2,
            leaf_size=ls,
            n_neighbors=k,
            weights=weights,
            metric="minkowski",
            algorithm="ball_tree",
        )
        # 使用 X，y 训练 gold 模型
        gold.fit(X, y)
        # 对 X_test 进行预测
        gold_preds = gold.predict(X_test)

        # 检查预测结果是否几乎相等
        for mine, theirs in zip(preds, gold_preds):
            np.testing.assert_almost_equal(mine, theirs)
        # 打印测试通过信息
        print("PASSED")
        # 更新循环计数器
        i += 1

# 定义测试 KNN 分类的函数，参数 N 默认值为 15
def test_knn_clf(N=15):
    # 设置随机种子为 12345
    np.random.seed(12345)

    # 初始化循环计数器 i
    i = 0
    # 当 i 小于 N 时执行循环
    while i < N:
        # 生成一个介于 2 到 100 之间的随机整数作为 N
        N = np.random.randint(2, 100)
        # 生成一个介于 2 到 100 之间的随机整数作为 M
        M = np.random.randint(2, 100)
        # 生成一个介于 1 到 N 之间的随机整数作为 k
        k = np.random.randint(1, N)
        # 生成一个介于 2 到 10 之间的随机整数作为 n_classes
        n_classes = np.random.randint(2, 10)
        # 生成一个介于 1 到 10 之间的随机整数和 N-1 中的最小值作为 ls
        ls = np.min([np.random.randint(1, 10), N - 1])
        # 设置权重为 "uniform"

        # 生成一个 N 行 M 列的随机数组作为 X
        X = np.random.rand(N, M)
        # 生成一个 N 行 M 列的随机数组作为 X_test
        X_test = np.random.rand(N, M)
        # 生成一个长度为 N 的随机整数数组作为 y
        y = np.random.randint(0, n_classes, size=N)

        # 创建 KNN 对象，设置参数 k, leaf_size, metric, classifier 和 weights
        knn = KNN(k=k, leaf_size=ls, metric=euclidean, classifier=True, weights=weights)
        # 使用 X 和 y 训练 KNN 模型
        knn.fit(X, y)
        # 对 X_test 进行预测
        preds = knn.predict(X_test)

        # 创建 KNeighborsClassifier 对象，设置参数 p, metric, leaf_size, n_neighbors, weights 和 algorithm
        gold = KNeighborsClassifier(
            p=2,
            metric="minkowski",
            leaf_size=ls,
            n_neighbors=k,
            weights=weights,
            algorithm="ball_tree",
        )
        # 使用 X 和 y 训练 KNeighborsClassifier 模型
        gold.fit(X, y)
        # 对 X_test 进行预测
        gold_preds = gold.predict(X_test)

        # 对 preds 和 gold_preds 进行逐一比较，检查是否几乎相等
        for mine, theirs in zip(preds, gold_preds):
            np.testing.assert_almost_equal(mine, theirs)
        # 打印 "PASSED" 表示测试通过
        print("PASSED")
        # i 自增 1
        i += 1
# 定义一个测试高斯过程回归的函数，参数 N 默认为 15
def test_gp_regression(N=15):
    # 设置随机种子
    np.random.seed(12345)

    # 初始化循环计数器 i
    i = 0
    # 循环执行 N 次
    while i < N:
        # 生成随机的 alpha
        alpha = np.random.rand()
        # 生成随机的 N
        N = np.random.randint(2, 100)
        # 生成随机的 M
        M = np.random.randint(2, 100)
        # 生成随机的 K
        K = np.random.randint(1, N)
        # 生成随机的 J
        J = np.random.randint(1, 3)

        # 生成随机的 N 行 M 列的数据矩阵 X
        X = np.random.rand(N, M)
        # 生成随机的 N 行 J 列的数据矩阵 y
        y = np.random.rand(N, J)
        # 生成随机的 K 行 M 列的数据矩阵 X_test
        X_test = np.random.rand(K, M)

        # 创建一个高斯过程回归对象 gp，使用 RBF 核函数和指定的 alpha
        gp = GPRegression(kernel="RBFKernel(sigma=1)", alpha=alpha)
        # 创建一个高斯过程回归对象 gold，使用默认参数
        gold = GaussianProcessRegressor(
            kernel=None, alpha=alpha, optimizer=None, normalize_y=False
        )

        # 使用 X, y 训练 gp 模型
        gp.fit(X, y)
        # 使用 X, y 训练 gold 模型
        gold.fit(X, y)

        # 对 X_test 进行预测，返回预测值 preds 和方差
        preds, _ = gp.predict(X_test)
        # 使用 gold 模型对 X_test 进行预测，返回预测值 gold_preds
        gold_preds = gold.predict(X_test)
        # 检查预测值 preds 和 gold_preds 是否几乎相等
        np.testing.assert_almost_equal(preds, gold_preds)

        # 计算 gp 模型的边缘对数似然
        mll = gp.marginal_log_likelihood()
        # 计算 gold 模型的对数边缘似然
        gold_mll = gold.log_marginal_likelihood()
        # 检查 mll 和 gold_mll 是否几乎相等
        np.testing.assert_almost_equal(mll, gold_mll)

        # 打印 "PASSED"，表示测试通过
        print("PASSED")
        # 更新循环计数器 i
        i += 1
```