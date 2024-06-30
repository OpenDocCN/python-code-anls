# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\svm.py`

```
from sklearn.svm import SVC  # 导入支持向量机（SVC）模型

from .common import Benchmark, Estimator, Predictor  # 导入本地模块中的Benchmark、Estimator、Predictor类
from .datasets import _synth_classification_dataset  # 导入本地模块中的_synth_classification_dataset函数
from .utils import make_gen_classif_scorers  # 导入本地模块中的make_gen_classif_scorers函数


class SVCBenchmark(Predictor, Estimator, Benchmark):
    """Benchmarks for SVC."""

    param_names = ["kernel"]  # 定义参数名列表，只有一个元素'kernel'
    params = (["linear", "poly", "rbf", "sigmoid"],)  # 定义参数取值列表，包含四个内核选项

    def setup_cache(self):
        super().setup_cache()  # 调用父类的setup_cache方法，设置缓存

    def make_data(self, params):
        return _synth_classification_dataset()  # 调用_synth_classification_dataset函数生成分类合成数据集

    def make_estimator(self, params):
        (kernel,) = params  # 从参数元组中解包获取kernel值

        estimator = SVC(
            max_iter=100, tol=1e-16, kernel=kernel, random_state=0, gamma="scale"
        )  # 创建一个SVC分类器实例，设置最大迭代次数、容忍度、内核类型、随机种子和gamma参数

        return estimator  # 返回创建的分类器实例

    def make_scorers(self):
        make_gen_classif_scorers(self)  # 调用make_gen_classif_scorers函数，生成通用分类评分器
```