# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\linear_model.py`

```
# 导入所需的线性模型库
from sklearn.linear_model import (
    ElasticNet,  # 弹性网络模型
    Lasso,        # 套索回归模型
    LinearRegression,  # 线性回归模型
    LogisticRegression,  # 逻辑回归模型
    Ridge,        # 岭回归模型
    SGDRegressor,  # 随机梯度下降回归模型
)

# 导入通用函数和类
from .common import Benchmark, Estimator, Predictor
from .datasets import (
    _20newsgroups_highdim_dataset,    # 高维度20新闻组数据集
    _20newsgroups_lowdim_dataset,     # 低维度20新闻组数据集
    _synth_regression_dataset,        # 合成回归数据集
    _synth_regression_sparse_dataset, # 稀疏合成回归数据集
)
from .utils import make_gen_classif_scorers, make_gen_reg_scorers


class LogisticRegressionBenchmark(Predictor, Estimator, Benchmark):
    """
    LogisticRegression 的基准测试类。
    """

    param_names = ["representation", "solver", "n_jobs"]
    params = (["dense", "sparse"], ["lbfgs", "saga"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        super().setup_cache()

    def make_data(self, params):
        representation, solver, n_jobs = params

        if Benchmark.data_size == "large":
            if representation == "sparse":
                data = _20newsgroups_highdim_dataset(n_samples=10000)  # 获取大规模稀疏数据集
            else:
                data = _20newsgroups_lowdim_dataset(n_components=1e3)  # 获取大规模低维度数据集
        else:
            if representation == "sparse":
                data = _20newsgroups_highdim_dataset(n_samples=2500)   # 获取小规模稀疏数据集
            else:
                data = _20newsgroups_lowdim_dataset()                   # 获取小规模低维度数据集

        return data

    def make_estimator(self, params):
        representation, solver, n_jobs = params

        penalty = "l2" if solver == "lbfgs" else "l1"  # 根据 solver 设置正则化类型

        estimator = LogisticRegression(
            solver=solver,
            penalty=penalty,
            tol=0.01,
            n_jobs=n_jobs,
            random_state=0,
        )

        return estimator

    def make_scorers(self):
        make_gen_classif_scorers(self)  # 使用通用分类评分器


class RidgeBenchmark(Predictor, Estimator, Benchmark):
    """
    Ridge 的基准测试类。
    """

    param_names = ["representation", "solver"]
    params = (
        ["dense", "sparse"],    # 数据表示类型
        ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],  # solver 类型
    )

    def setup_cache(self):
        super().setup_cache()

    def make_data(self, params):
        representation, solver = params

        if representation == "dense":
            data = _synth_regression_dataset(n_samples=500000, n_features=100)  # 创建密集型回归数据集
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=100000, n_features=10000, density=0.005  # 创建稀疏型回归数据集
            )

        return data

    def make_estimator(self, params):
        representation, solver = params

        estimator = Ridge(solver=solver, fit_intercept=False, random_state=0)  # 创建 Ridge 回归器

        return estimator

    def make_scorers(self):
        make_gen_reg_scorers(self)  # 使用通用回归评分器

    def skip(self, params):
        representation, solver = params

        if representation == "sparse" and solver == "svd":
            return True  # 跳过稀疏表示和 svd solver 的组合
        return False


class LinearRegressionBenchmark(Predictor, Estimator, Benchmark):
    """
    线性回归的基准测试类。
    """

    param_names = ["representation"]  # 数据表示类型
    # 参数定义为包含一个元组的列表，元组包含字符串 "dense" 和 "sparse"
    params = (["dense", "sparse"],)

    # 设置缓存的初始化方法，调用父类的设置缓存方法
    def setup_cache(self):
        super().setup_cache()

    # 根据参数生成数据集的方法
    def make_data(self, params):
        # 从参数元组中解包得到表示形式（representation）
        (representation,) = params

        # 根据表示形式选择生成稠密或稀疏的回归数据集
        if representation == "dense":
            data = _synth_regression_dataset(n_samples=1000000, n_features=100)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=10000, n_features=100000, density=0.01
            )

        # 返回生成的数据集
        return data

    # 创建估算器（estimator）的方法
    def make_estimator(self, params):
        # 创建线性回归估算器对象
        estimator = LinearRegression()

        # 返回估算器对象
        return estimator

    # 创建评分器（scorer）的方法
    def make_scorers(self):
        # 调用通用的生成回归评分器的方法
        make_gen_reg_scorers(self)
class SGDRegressorBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmark for SGD
    """

    # 参数名字列表
    param_names = ["representation"]
    # 参数组合
    params = (["dense", "sparse"],)

    # 设置缓存
    def setup_cache(self):
        super().setup_cache()  # 调用父类的设置缓存方法

    # 生成数据
    def make_data(self, params):
        (representation,) = params  # 解包参数元组

        # 根据表示类型生成合成回归数据集
        if representation == "dense":
            data = _synth_regression_dataset(n_samples=100000, n_features=200)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=100000, n_features=1000, density=0.01
            )

        return data

    # 创建评估器
    def make_estimator(self, params):
        (representation,) = params  # 解包参数元组

        # 根据表示类型选择最大迭代次数
        max_iter = 60 if representation == "dense" else 300

        # 创建 SGD 回归器对象
        estimator = SGDRegressor(max_iter=max_iter, tol=None, random_state=0)

        return estimator

    # 创建评分器
    def make_scorers(self):
        make_gen_reg_scorers(self)  # 调用通用回归评分器生成函数


class ElasticNetBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for ElasticNet.
    """

    # 参数名字列表
    param_names = ["representation", "precompute"]
    # 参数组合
    params = (["dense", "sparse"], [True, False])

    # 设置缓存
    def setup_cache(self):
        super().setup_cache()  # 调用父类的设置缓存方法

    # 生成数据
    def make_data(self, params):
        representation, precompute = params  # 解包参数元组

        # 根据表示类型生成合成回归数据集
        if representation == "dense":
            data = _synth_regression_dataset(n_samples=1000000, n_features=100)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=50000, n_features=5000, density=0.01
            )

        return data

    # 创建评估器
    def make_estimator(self, params):
        representation, precompute = params  # 解包参数元组

        # 创建 ElasticNet 回归器对象
        estimator = ElasticNet(precompute=precompute, alpha=0.001, random_state=0)

        return estimator

    # 创建评分器
    def make_scorers(self):
        make_gen_reg_scorers(self)  # 调用通用回归评分器生成函数

    # 跳过特定参数组合的方法
    def skip(self, params):
        representation, precompute = params  # 解包参数元组

        # 如果表示类型为稀疏且不预计算，则跳过
        if representation == "sparse" and precompute is False:
            return True
        return False


class LassoBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for Lasso.
    """

    # 参数名字列表
    param_names = ["representation", "precompute"]
    # 参数组合
    params = (["dense", "sparse"], [True, False])

    # 设置缓存
    def setup_cache(self):
        super().setup_cache()  # 调用父类的设置缓存方法

    # 生成数据
    def make_data(self, params):
        representation, precompute = params  # 解包参数元组

        # 根据表示类型生成合成回归数据集
        if representation == "dense":
            data = _synth_regression_dataset(n_samples=1000000, n_features=100)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=50000, n_features=5000, density=0.01
            )

        return data

    # 创建评估器
    def make_estimator(self, params):
        representation, precompute = params  # 解包参数元组

        # 创建 Lasso 回归器对象
        estimator = Lasso(precompute=precompute, alpha=0.001, random_state=0)

        return estimator

    # 创建评分器
    def make_scorers(self):
        make_gen_reg_scorers(self)  # 调用通用回归评分器生成函数
    # 定义一个方法 `skip`，接受一个参数 `params`
    def skip(self, params):
        # 解构参数 `params`，将其分解为 `representation` 和 `precompute`
        representation, precompute = params
        
        # 检查 `representation` 是否为 "sparse"，并且 `precompute` 是 False
        if representation == "sparse" and precompute is False:
            # 如果满足条件，返回 True
            return True
        
        # 如果不满足条件，返回 False
        return False
```