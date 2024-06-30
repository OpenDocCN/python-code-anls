# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\ensemble.py`

```
# 导入需要的类和函数
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

# 导入本地模块中的类和函数
from .common import Benchmark, Estimator, Predictor
from .datasets import (
    _20newsgroups_highdim_dataset,
    _20newsgroups_lowdim_dataset,
    _synth_classification_dataset,
)
from .utils import make_gen_classif_scorers


class RandomForestClassifierBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for RandomForestClassifier.
    """

    # 参数名列表
    param_names = ["representation", "n_jobs"]
    # 参数组合
    params = (["dense", "sparse"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        # 调用父类的缓存设置方法
        super().setup_cache()

    def make_data(self, params):
        # 解包参数
        representation, n_jobs = params

        # 根据表示类型选择数据集
        if representation == "sparse":
            data = _20newsgroups_highdim_dataset()
        else:
            data = _20newsgroups_lowdim_dataset()

        return data

    def make_estimator(self, params):
        # 解包参数
        representation, n_jobs = params

        # 根据数据大小设置决策树数量
        n_estimators = 500 if Benchmark.data_size == "large" else 100

        # 创建随机森林分类器对象
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=10,
            max_features="log2",
            n_jobs=n_jobs,
            random_state=0,
        )

        return estimator

    def make_scorers(self):
        # 使用工具函数创建分类器评分器
        make_gen_classif_scorers(self)


class GradientBoostingClassifierBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for GradientBoostingClassifier.
    """

    # 参数名列表
    param_names = ["representation"]
    # 参数组合
    params = (["dense", "sparse"],)

    def setup_cache(self):
        # 调用父类的缓存设置方法
        super().setup_cache()

    def make_data(self, params):
        # 解包参数
        (representation,) = params

        # 根据表示类型选择数据集
        if representation == "sparse":
            data = _20newsgroups_highdim_dataset()
        else:
            data = _20newsgroups_lowdim_dataset()

        return data

    def make_estimator(self, params):
        # 解包参数
        (representation,) = params

        # 根据数据大小设置弱学习器数量
        n_estimators = 100 if Benchmark.data_size == "large" else 10

        # 创建梯度提升分类器对象
        estimator = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_features="log2",
            subsample=0.5,
            random_state=0,
        )

        return estimator

    def make_scorers(self):
        # 使用工具函数创建分类器评分器
        make_gen_classif_scorers(self)


class HistGradientBoostingClassifierBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for HistGradientBoostingClassifier.
    """

    # 参数名列表为空
    param_names = []
    # 参数组合为空
    params = ()

    def setup_cache(self):
        # 调用父类的缓存设置方法
        super().setup_cache()

    def make_data(self, params):
        # 创建合成分类数据集
        data = _synth_classification_dataset(
            n_samples=10000, n_features=100, n_classes=5
        )

        return data

    def make_estimator(self, params):
        # 创建直方图梯度提升分类器对象
        estimator = HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, early_stopping=False, random_state=0
        )

        return estimator

    def make_scorers(self):
        # 使用工具函数创建分类器评分器
        make_gen_classif_scorers(self)
```