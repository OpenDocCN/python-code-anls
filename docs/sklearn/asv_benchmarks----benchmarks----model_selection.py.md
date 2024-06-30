# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\model_selection.py`

```
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器模型
from sklearn.model_selection import GridSearchCV, cross_val_score  # 导入网格搜索交叉验证和交叉验证评分函数

from .common import Benchmark, Estimator, Predictor  # 导入本地的Benchmark、Estimator和Predictor类
from .datasets import _synth_classification_dataset  # 导入本地的_synth_classification_dataset函数
from .utils import make_gen_classif_scorers  # 导入本地的make_gen_classif_scorers函数


class CrossValidationBenchmark(Benchmark):
    """
    交叉验证的基准类。
    """

    timeout = 20000  # 设置超时时间为20,000毫秒

    param_names = ["n_jobs"]  # 参数名为n_jobs
    params = (Benchmark.n_jobs_vals,)  # 参数列表从Benchmark类的n_jobs_vals属性获取

    def setup(self, *params):
        (n_jobs,) = params  # 解包参数元组，获取n_jobs

        data = _synth_classification_dataset(n_samples=50000, n_features=100)  # 创建合成分类数据集，样本数为50,000，特征数为100
        self.X, self.X_val, self.y, self.y_val = data  # 将数据集分配给实例变量self.X、self.X_val、self.y、self.y_val

        self.clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)  # 初始化随机森林分类器对象

        cv = 16 if Benchmark.data_size == "large" else 4  # 如果数据集大小为"large"，设置交叉验证折数为16，否则为4

        self.cv_params = {"n_jobs": n_jobs, "cv": cv}  # 设定交叉验证的参数字典包括n_jobs和cv


    def time_crossval(self, *args):
        cross_val_score(self.clf, self.X, self.y, **self.cv_params)  # 进行时间度量的交叉验证评分


    def peakmem_crossval(self, *args):
        cross_val_score(self.clf, self.X, self.y, **self.cv_params)  # 进行内存峰值度量的交叉验证评分


    def track_crossval(self, *args):
        return float(cross_val_score(self.clf, self.X, self.y, **self.cv_params).mean())  # 跟踪并返回交叉验证评分的平均值


class GridSearchBenchmark(Predictor, Estimator, Benchmark):
    """
    网格搜索的基准类。
    """

    timeout = 20000  # 设置超时时间为20,000毫秒

    param_names = ["n_jobs"]  # 参数名为n_jobs
    params = (Benchmark.n_jobs_vals,)  # 参数列表从Benchmark类的n_jobs_vals属性获取

    def setup_cache(self):
        super().setup_cache()  # 调用父类Benchmark的setup_cache方法

    def make_data(self, params):
        data = _synth_classification_dataset(n_samples=10000, n_features=100)  # 创建合成分类数据集，样本数为10,000，特征数为100

        return data  # 返回创建的数据集

    def make_estimator(self, params):
        (n_jobs,) = params  # 解包参数元组，获取n_jobs

        clf = RandomForestClassifier(random_state=0)  # 初始化随机森林分类器对象

        if Benchmark.data_size == "large":
            n_estimators_list = [10, 25, 50, 100, 500]  # 大数据集时的n_estimators候选列表
            max_depth_list = [5, 10, None]  # 大数据集时的max_depth候选列表
            max_features_list = [0.1, 0.4, 0.8, 1.0]  # 大数据集时的max_features候选列表
        else:
            n_estimators_list = [10, 25, 50]  # 小数据集时的n_estimators候选列表
            max_depth_list = [5, 10]  # 小数据集时的max_depth候选列表
            max_features_list = [0.1, 0.4, 0.8]  # 小数据集时的max_features候选列表

        param_grid = {
            "n_estimators": n_estimators_list,  # 网格搜索的n_estimators参数范围
            "max_depth": max_depth_list,  # 网格搜索的max_depth参数范围
            "max_features": max_features_list,  # 网格搜索的max_features参数范围
        }

        estimator = GridSearchCV(clf, param_grid, n_jobs=n_jobs, cv=4)  # 创建网格搜索交叉验证对象

        return estimator  # 返回创建的网格搜索交叉验证对象

    def make_scorers(self):
        make_gen_classif_scorers(self)  # 调用本地函数make_gen_classif_scorers，生成通用分类器评分函数
```