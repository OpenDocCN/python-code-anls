# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_parallel.py`

```
import time  # 导入时间模块

import joblib  # 导入joblib用于并行处理
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架
from numpy.testing import assert_array_equal  # 导入NumPy的数组相等断言函数

from sklearn import config_context, get_config  # 导入sklearn的配置相关模块
from sklearn.compose import make_column_transformer  # 导入构建列转换器的函数
from sklearn.datasets import load_iris  # 导入鸢尾花数据集加载函数
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.model_selection import GridSearchCV  # 导入网格搜索交叉验证
from sklearn.pipeline import make_pipeline  # 导入构建管道的函数
from sklearn.preprocessing import StandardScaler  # 导入数据标准化函数
from sklearn.utils.parallel import Parallel, delayed  # 导入并行处理相关函数


def get_working_memory():
    return get_config()["working_memory"]  # 获取当前配置中的工作内存设置


@pytest.mark.parametrize("n_jobs", [1, 2])  # 参数化测试，测试n_jobs参数为1和2
@pytest.mark.parametrize("backend", ["loky", "threading", "multiprocessing"])  # 参数化测试，测试不同的并行后端
def test_configuration_passes_through_to_joblib(n_jobs, backend):
    # 测试全局配置是否传递给joblib的任务

    with config_context(working_memory=123):  # 使用配置上下文设置工作内存为123
        results = Parallel(n_jobs=n_jobs, backend=backend)(  # 并行执行任务
            delayed(get_working_memory)() for _ in range(2)
        )

    assert_array_equal(results, [123] * 2)  # 断言结果数组是否为两个123的列表


def test_parallel_delayed_warnings():
    """Informative warnings should be raised when mixing sklearn and joblib API"""
    # 在混合使用sklearn和joblib API时应发出警告
    # 当使用sklearn.utils.fixes.Parallel和joblib.delayed时应发出警告，配置不会传播到工作进程中
    warn_msg = "`sklearn.utils.parallel.Parallel` needs to be used in conjunction"
    with pytest.warns(UserWarning, match=warn_msg) as records:
        Parallel()(joblib.delayed(time.sleep)(0) for _ in range(10))
    assert len(records) == 10

    # 当使用sklearn.utils.fixes.delayed和joblib.Parallel时应发出警告
    warn_msg = (
        "`sklearn.utils.parallel.delayed` should be used with "
        "`sklearn.utils.parallel.Parallel` to make it possible to propagate"
    )
    with pytest.warns(UserWarning, match=warn_msg) as records:
        joblib.Parallel()(delayed(time.sleep)(0) for _ in range(10))
    assert len(records) == 10


@pytest.mark.parametrize("n_jobs", [1, 2])  # 参数化测试，测试n_jobs参数为1和2
def test_dispatch_config_parallel(n_jobs):
    """Check that we properly dispatch the configuration in parallel processing.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/25239
    """
    pd = pytest.importorskip("pandas")  # 导入并检查pandas模块是否可用
    iris = load_iris(as_frame=True)  # 加载鸢尾花数据集，返回DataFrame格式的数据

    class TransformerRequiredDataFrame(StandardScaler):
        def fit(self, X, y=None):
            assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
            return super().fit(X, y)

        def transform(self, X, y=None):
            assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
            return super().transform(X, y)

    dropper = make_column_transformer(
        ("drop", [0]),  # 移除第一列特征
        remainder="passthrough",  # 其余特征保持不变
        n_jobs=n_jobs,  # 并行处理的工作数
    )
    param_grid = {"randomforestclassifier__max_depth": [1, 2, 3]}  # 随机森林分类器的超参数网格搜索
    # 创建一个 GridSearchCV 对象，用于模型超参数搜索
    search_cv = GridSearchCV(
        # 使用管道来依次执行数据处理步骤和随机森林分类器的初始化
        make_pipeline(
            dropper,  # 数据预处理步骤：删除指定的特征列
            TransformerRequiredDataFrame(),  # 转换器：确保数据类型为 DataFrame
            RandomForestClassifier(n_estimators=5, n_jobs=n_jobs),  # 随机森林分类器
        ),
        param_grid,  # 超参数网格
        cv=5,  # 交叉验证的折数
        n_jobs=n_jobs,  # 并行工作的 CPU 核数
        error_score="raise",  # 如果搜索失败则抛出异常
    )

    # 确保在不请求 DataFrame 的情况下 `fit` 方法会失败
    with pytest.raises(AssertionError, match="X should be a DataFrame"):
        search_cv.fit(iris.data, iris.target)

    # 设置输出转换为 pandas 数据框上下文，预期每个中间步骤输出的是 DataFrame
    with config_context(transform_output="pandas"):
        search_cv.fit(iris.data, iris.target)  # 执行拟合操作

    # 断言网格搜索的交叉验证结果中不含有 NaN 值
    assert not np.isnan(search_cv.cv_results_["mean_test_score"]).any()
```