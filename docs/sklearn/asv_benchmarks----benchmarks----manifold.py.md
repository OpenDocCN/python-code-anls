# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\manifold.py`

```
from sklearn.manifold import TSNE  # 导入 t-SNE 模型

from .common import Benchmark, Estimator  # 导入 Benchmark 和 Estimator 类
from .datasets import _digits_dataset  # 导入数据集函数 _digits_dataset


class TSNEBenchmark(Estimator, Benchmark):
    """
    Benchmarks for t-SNE.
    """

    param_names = ["method"]  # 参数名称列表，这里只有一个参数 "method"
    params = (["exact", "barnes_hut"],)  # 参数取值，分别为 "exact" 和 "barnes_hut"

    def setup_cache(self):
        super().setup_cache()  # 调用父类 Estimator 的 setup_cache 方法

    def make_data(self, params):
        (method,) = params  # 解包参数元组，获取 method 参数值

        n_samples = 500 if method == "exact" else None  # 根据 method 参数确定样本数

        return _digits_dataset(n_samples=n_samples)  # 调用数据集生成函数 _digits_dataset

    def make_estimator(self, params):
        (method,) = params  # 解包参数元组，获取 method 参数值

        estimator = TSNE(random_state=0, method=method)  # 创建 t-SNE 估计器对象

        return estimator  # 返回估计器对象

    def make_scorers(self):
        self.train_scorer = lambda _, __: self.estimator.kl_divergence_  # 定义训练得分函数
        self.test_scorer = lambda _, __: self.estimator.kl_divergence_  # 定义测试得分函数
```