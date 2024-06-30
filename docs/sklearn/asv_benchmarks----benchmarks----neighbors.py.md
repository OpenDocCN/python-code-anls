# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\neighbors.py`

```
# 导入KNeighborsClassifier类，用于实现K近邻分类器
from sklearn.neighbors import KNeighborsClassifier

# 从当前目录中导入Benchmark、Estimator、Predictor类
from .common import Benchmark, Estimator, Predictor
# 从.datasets模块中导入_20newsgroups_lowdim_dataset函数
from .datasets import _20newsgroups_lowdim_dataset
# 从.utils模块中导入make_gen_classif_scorers函数
from .utils import make_gen_classif_scorers


# 定义KNeighborsClassifierBenchmark类，继承自Predictor、Estimator、Benchmark类
class KNeighborsClassifierBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for KNeighborsClassifier.
    """

    # 参数名称列表
    param_names = ["algorithm", "dimension", "n_jobs"]
    # 参数值元组，包含了三个列表：算法、维度、工作线程数
    params = (["brute", "kd_tree", "ball_tree"], ["low", "high"], Benchmark.n_jobs_vals)

    # 初始化方法，继承自Benchmark类的setup_cache方法
    def setup_cache(self):
        super().setup_cache()

    # 根据参数生成数据的方法
    def make_data(self, params):
        # 解包参数元组
        algorithm, dimension, n_jobs = params

        # 根据数据大小设置n_components的值
        if Benchmark.data_size == "large":
            n_components = 40 if dimension == "low" else 200
        else:
            n_components = 10 if dimension == "low" else 50

        # 调用_20newsgroups_lowdim_dataset函数生成数据
        data = _20newsgroups_lowdim_dataset(n_components=n_components)

        return data

    # 根据参数生成评估器的方法
    def make_estimator(self, params):
        # 解包参数元组
        algorithm, dimension, n_jobs = params

        # 创建KNeighborsClassifier对象作为评估器
        estimator = KNeighborsClassifier(algorithm=algorithm, n_jobs=n_jobs)

        return estimator

    # 生成分类器评分函数的方法
    def make_scorers(self):
        # 调用make_gen_classif_scorers函数，生成分类器评分函数
        make_gen_classif_scorers(self)
```