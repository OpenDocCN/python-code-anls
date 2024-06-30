# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\common.py`

```
import itertools  # 导入 itertools 模块，用于高效的迭代操作
import json  # 导入 json 模块，用于 JSON 数据的处理
import os  # 导入 os 模块，用于操作系统相关功能
import pickle  # 导入 pickle 模块，用于对象的序列化和反序列化
import timeit  # 导入 timeit 模块，用于测量代码执行时间
from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 抽象基类和 abstractmethod 装饰器
from multiprocessing import cpu_count  # 导入 multiprocessing 模块的 cpu_count 函数，用于获取 CPU 核心数
from pathlib import Path  # 导入 pathlib 模块中的 Path 类，用于处理路径操作

import numpy as np  # 导入 NumPy 库，用于科学计算

def get_from_config():
    """Get benchmarks configuration from the config.json file"""
    current_path = Path(__file__).resolve().parent  # 获取当前脚本所在目录的绝对路径

    config_path = current_path / "config.json"  # 拼接配置文件的路径
    with open(config_path, "r") as config_file:
        config_file = "".join(line for line in config_file if line and "//" not in line)  # 读取配置文件内容，并排除注释
        config = json.loads(config_file)  # 解析 JSON 格式的配置文件内容

    profile = os.getenv("SKLBENCH_PROFILE", config["profile"])  # 获取或设置性能配置文件的配置文件

    n_jobs_vals_env = os.getenv("SKLBENCH_NJOBS")  # 获取环境变量 SKLBENCH_NJOBS
    if n_jobs_vals_env:
        n_jobs_vals = json.loads(n_jobs_vals_env)  # 解析环境变量中的 JSON 格式数据
    else:
        n_jobs_vals = config["n_jobs_vals"]  # 否则使用配置文件中的 n_jobs_vals

    if not n_jobs_vals:
        n_jobs_vals = list(range(1, 1 + cpu_count()))  # 如果 n_jobs_vals 为空，则生成默认的 CPU 核心数范围列表

    cache_path = current_path / "cache"  # 缓存目录的路径
    cache_path.mkdir(exist_ok=True)  # 如果缓存目录不存在，则创建该目录
    (cache_path / "estimators").mkdir(exist_ok=True)  # 创建存放估算器模型的子目录
    (cache_path / "tmp").mkdir(exist_ok=True)  # 创建临时文件存放的子目录

    save_estimators = os.getenv("SKLBENCH_SAVE_ESTIMATORS", config["save_estimators"])  # 获取或设置是否保存估算器模型
    save_dir = os.getenv("ASV_COMMIT", "new")[:8]  # 获取 ASV_COMMIT 环境变量的前 8 位，用于保存估算器模型的目录名

    if save_estimators:
        (cache_path / "estimators" / save_dir).mkdir(exist_ok=True)  # 如果保存估算器模型，创建对应的目录

    base_commit = os.getenv("SKLBENCH_BASE_COMMIT", config["base_commit"])  # 获取或设置基准提交的哈希值

    bench_predict = os.getenv("SKLBENCH_PREDICT", config["bench_predict"])  # 获取或设置是否进行预测基准测试
    bench_transform = os.getenv("SKLBENCH_TRANSFORM", config["bench_transform"])  # 获取或设置是否进行转换基准测试

    return (
        profile,
        n_jobs_vals,
        save_estimators,
        save_dir,
        base_commit,
        bench_predict,
        bench_transform,
    )  # 返回从配置中获取的各项配置信息


def get_estimator_path(benchmark, directory, params, save=False):
    """Get path of pickled fitted estimator"""
    path = Path(__file__).resolve().parent / "cache"  # 获取缓存目录的路径
    path = (path / "estimators" / directory) if save else (path / "tmp")  # 如果保存模型，路径为 estimators 目录下的对应目录，否则为 tmp 目录

    filename = (
        benchmark.__class__.__name__  # 获取基准类的名称
        + "_estimator_"
        + "_".join(list(map(str, params)))  # 将参数列表转换为字符串并用下划线连接
        + ".pkl"  # 拼接成最终的文件名，以 .pkl 结尾
    )

    return path / filename  # 返回最终的文件路径


def clear_tmp():
    """Clean the tmp directory"""
    path = Path(__file__).resolve().parent / "cache" / "tmp"  # 获取临时文件目录的路径
    for child in path.iterdir():
        child.unlink()  # 遍历目录中的所有文件并删除


class Benchmark(ABC):
    """Abstract base class for all the benchmarks"""

    timer = timeit.default_timer  # 设置计时器为默认的 wall time 计时器
    processes = 1  # 设置进程数为 1
    timeout = 500  # 设置超时时间为 500 秒

    (
        profile,
        n_jobs_vals,
        save_estimators,
        save_dir,
        base_commit,
        bench_predict,
        bench_transform,
    ) = get_from_config()  # 使用 get_from_config 函数获取配置信息

    if profile == "fast":
        warmup_time = 0  # 如果配置为快速模式，设置预热时间为 0
        repeat = 1  # 设置重复运行次数为 1
        number = 1  # 设置运行次数为 1
        min_run_count = 1  # 设置最小运行次数为 1
        data_size = "small"  # 设置数据大小为小型数据集
    elif profile == "regular":
        warmup_time = 1  # 如果配置为常规模式，设置预热时间为 1
        repeat = (3, 100, 30)  # 设置重复运行次数为元组 (3, 100, 30)
        data_size = "small"  # 设置数据大小为小型数据集
    elif profile == "large_scale":
        warmup_time = 1  # 如果配置为大规模模式，设置预热时间为 1
        repeat = 3  # 设置重复运行次数为 3
        number = 1  # 设置运行次数为 1
        data_size = "large"  # 设置数据大小为大型数据集
    # 声明一个抽象方法 `params`，用于获取对象的参数信息
    @property
    @abstractmethod
    def params(self):
        # 抽象方法体，需要在具体子类中实现，用于返回对象的参数信息
        pass
class Estimator(ABC):
    """Abstract base class for all benchmarks of estimators"""

    @abstractmethod
    def make_data(self, params):
        """Return the dataset for a combination of parameters"""
        # The datasets are cached using joblib.Memory so it's fast and can be
        # called for each repeat
        pass

    @abstractmethod
    def make_estimator(self, params):
        """Return an instance of the estimator for a combination of parameters"""
        pass

    def skip(self, params):
        """Return True if the benchmark should be skipped for these params"""
        return False

    def setup_cache(self):
        """Pickle a fitted estimator for all combinations of parameters"""
        # This is run once per benchmark class.

        # 清除临时数据
        clear_tmp()

        # 生成参数网格
        param_grid = list(itertools.product(*self.params))

        # 对每个参数组合进行操作
        for params in param_grid:
            if self.skip(params):
                continue

            # 创建估计器实例并拟合数据
            estimator = self.make_estimator(params)
            X, _, y, _ = self.make_data(params)
            estimator.fit(X, y)

            # 获取估计器的保存路径并写入文件
            est_path = get_estimator_path(
                self, Benchmark.save_dir, params, Benchmark.save_estimators
            )
            with est_path.open(mode="wb") as f:
                pickle.dump(estimator, f)

    def setup(self, *params):
        """Generate dataset and load the fitted estimator"""
        # This is run once per combination of parameters and per repeat so we
        # need to avoid doing expensive operations there.

        if self.skip(params):
            # 如果跳过该参数组合，则抛出未实现错误
            raise NotImplementedError

        # 生成数据集并加载拟合好的估计器
        self.X, self.X_val, self.y, self.y_val = self.make_data(params)

        # 加载估计器对象
        est_path = get_estimator_path(
            self, Benchmark.save_dir, params, Benchmark.save_estimators
        )
        with est_path.open(mode="rb") as f:
            self.estimator = pickle.load(f)

        # 创建评分器
        self.make_scorers()

    def time_fit(self, *args):
        # 训练时间统计函数
        self.estimator.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        # 记录内存峰值函数
        self.estimator.fit(self.X, self.y)

    def track_train_score(self, *args):
        # 跟踪训练集评分函数
        if hasattr(self.estimator, "predict"):
            y_pred = self.estimator.predict(self.X)
        else:
            y_pred = None
        return float(self.train_scorer(self.y, y_pred))

    def track_test_score(self, *args):
        # 跟踪测试集评分函数
        if hasattr(self.estimator, "predict"):
            y_val_pred = self.estimator.predict(self.X_val)
        else:
            y_val_pred = None
        return float(self.test_scorer(self.y_val, y_val_pred))


class Predictor(ABC):
    """Abstract base class for benchmarks of estimators implementing predict"""
    # 如果 Benchmark.bench_predict 为真，则执行以下代码块
    if Benchmark.bench_predict:

        # 定义一个方法 time_predict，用于预测结果，但不返回任何值
        def time_predict(self, *args):
            self.estimator.predict(self.X)

        # 定义一个方法 peakmem_predict，用于预测结果，但不返回任何值
        def peakmem_predict(self, *args):
            self.estimator.predict(self.X)

        # 如果 Benchmark.base_commit 不为 None，则执行以下代码块
        if Benchmark.base_commit is not None:

            # 定义一个方法 track_same_prediction，用于跟踪是否有相同的预测结果
            def track_same_prediction(self, *args):
                # 获取基准提交时的估计器路径
                est_path = get_estimator_path(self, Benchmark.base_commit, args, True)
                # 以二进制读取估计器基准文件
                with est_path.open(mode="rb") as f:
                    estimator_base = pickle.load(f)

                # 使用基准估计器预测验证数据集的目标值
                y_val_pred_base = estimator_base.predict(self.X_val)
                # 使用当前估计器预测验证数据集的目标值
                y_val_pred = self.estimator.predict(self.X_val)

                # 返回两个预测结果是否全部接近的布尔值
                return np.allclose(y_val_pred_base, y_val_pred)

    # 抽象属性装饰器，用于声明一个抽象方法 params
    @property
    @abstractmethod
    def params(self):
        pass
class Transformer(ABC):
    """Abstract base class for benchmarks of estimators implementing transform"""

    # 如果 Benchmark.bench_transform 为真，则定义以下方法
    if Benchmark.bench_transform:

        # 计算 transform 方法的执行时间
        def time_transform(self, *args):
            self.estimator.transform(self.X)

        # 计算 transform 方法的内存峰值
        def peakmem_transform(self, *args):
            self.estimator.transform(self.X)

        # 如果 Benchmark.base_commit 不为空，则定义以下方法
        if Benchmark.base_commit is not None:

            # 跟踪相同 transform 方法的行为
            def track_same_transform(self, *args):
                # 获取基准版本的估计器路径
                est_path = get_estimator_path(self, Benchmark.base_commit, args, True)
                # 使用二进制模式打开文件
                with est_path.open(mode="rb") as f:
                    # 从文件中加载基准版本的估计器
                    estimator_base = pickle.load(f)

                # 对验证集进行基准版本估计器的 transform 操作
                X_val_t_base = estimator_base.transform(self.X_val)
                # 对验证集进行当前估计器的 transform 操作
                X_val_t = self.estimator.transform(self.X_val)

                # 检查两个 transform 结果是否近似相等
                return np.allclose(X_val_t_base, X_val_t)

    @property
    @abstractmethod
    def params(self):
        pass
```