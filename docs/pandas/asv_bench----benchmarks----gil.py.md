# `D:\src\scipysrc\pandas\asv_bench\benchmarks\gil.py`

```
from functools import wraps  # 导入装饰器相关的函数
import threading  # 导入多线程模块

import numpy as np  # 导入NumPy库

from pandas import (  # 从Pandas库中导入多个模块和函数
    DataFrame,
    Index,
    Series,
    date_range,
    factorize,
    read_csv,
)
from pandas.core.algorithms import take_nd  # 导入Pandas核心算法中的函数

try:
    from pandas import (  # 尝试从Pandas库中导入滚动统计函数
        rolling_kurt,
        rolling_max,
        rolling_mean,
        rolling_median,
        rolling_min,
        rolling_skew,
        rolling_std,
        rolling_var,
    )
    have_rolling_methods = True  # 设置滚动统计函数可用标志
except ImportError:
    have_rolling_methods = False  # 如果导入失败，设置滚动统计函数不可用标志

try:
    from pandas._libs import algos  # 尝试导入Pandas内部算法库中的algos模块
except ImportError:
    from pandas import algos  # 如果导入失败，则从Pandas库中导入algos模块

from .pandas_vb_common import BaseIO  # 导入本地模块中的BaseIO类（跳过isort排序）

def test_parallel(num_threads=2, kwargs_list=None):
    """
    Decorator to run the same function multiple times in parallel.

    Parameters
    ----------
    num_threads : int, optional
        The number of times the function is run in parallel.
    kwargs_list : list of dicts, optional
        The list of kwargs to update original
        function kwargs on different threads.

    Notes
    -----
    This decorator does not pass the return value of the decorated function.

    Original from scikit-image:

    https://github.com/scikit-image/scikit-image/pull/1519

    """
    assert num_threads > 0  # 断言确保num_threads大于0
    has_kwargs_list = kwargs_list is not None  # 检查是否有kwargs_list参数
    if has_kwargs_list:
        assert len(kwargs_list) == num_threads  # 断言确保kwargs_list长度与num_threads一致

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if has_kwargs_list:
                update_kwargs = lambda i: dict(kwargs, **kwargs_list[i])  # 更新kwargs为指定索引的kwargs_list
            else:
                update_kwargs = lambda i: kwargs  # 否则保持原有kwargs

            threads = []
            for i in range(num_threads):
                updated_kwargs = update_kwargs(i)  # 获取更新后的kwargs
                thread = threading.Thread(target=func, args=args, kwargs=updated_kwargs)  # 创建线程
                threads.append(thread)  # 添加线程到列表

            for thread in threads:
                thread.start()  # 启动所有线程

            for thread in threads:
                thread.join()  # 等待所有线程执行完毕

        return inner

    return wrapper


class ParallelGroupbyMethods:
    params = ([2, 4, 8], ["count", "last", "max", "mean", "min", "prod", "sum", "var"])  # 定义参数范围
    param_names = ["threads", "method"]  # 定义参数名称列表

    def setup(self, threads, method):
        N = 10**6  # 设置数据量N
        ngroups = 10**3  # 设置分组数量ngroups
        df = DataFrame(  # 创建DataFrame对象df
            {"key": np.random.randint(0, ngroups, size=N), "data": np.random.randn(N)}
        )

        @test_parallel(num_threads=threads)  # 使用test_parallel装饰器并指定线程数
        def parallel():
            getattr(df.groupby("key")["data"], method)()  # 获取并调用指定方法

        self.parallel = parallel  # 将parallel方法赋值给实例属性

        def loop():
            getattr(df.groupby("key")["data"], method)()  # 在单线程下执行相同方法

        self.loop = loop  # 将loop方法赋值给实例属性

    def time_parallel(self, threads, method):
        self.parallel()  # 执行并行运行方法

    def time_loop(self, threads, method):
        for i in range(threads):
            self.loop()  # 循环执行单线程方法


class ParallelGroups:
    params = [2, 4, 8]  # 定义线程数参数范围
    param_names = ["threads"]  # 定义参数名称
    # 设置函数，用于初始化测试环境，接收线程数作为参数
    def setup(self, threads):
        # 定义数据大小为2的22次方
        size = 2**22
        # 定义分组数为1000
        ngroups = 10**3
        # 生成随机数据序列，数值范围在0到ngroups-1之间，大小为size
        data = Series(np.random.randint(0, ngroups, size=size))
    
        # 使用装饰器声明并定义一个并行测试函数
        @test_parallel(num_threads=threads)
        def get_groups():
            # 对数据进行分组并获取分组信息
            data.groupby(data).groups
    
        # 将并行测试函数赋值给实例的属性
        self.get_groups = get_groups
    
    # 定义计时函数，用于执行获取分组的操作，并计时
    def time_get_groups(self, threads):
        # 调用之前设置的并行测试函数，获取分组信息
        self.get_groups()
class ParallelTake1D:
    # 定义参数列表
    params = ["int64", "float64"]
    # 定义参数名称列表
    param_names = ["dtype"]

    # 初始化设置方法，接收dtype参数
    def setup(self, dtype):
        # 设置数组大小
        N = 10**6
        # 创建DataFrame对象，包含一个从0到N-1的列，数据类型由dtype指定
        df = DataFrame({"col": np.arange(N, dtype=dtype)})
        # 创建索引器，从100到df长度减100的范围
        indexer = np.arange(100, len(df) - 100)

        # 定义并行测试函数，使用test_parallel装饰器，设置线程数为2
        @test_parallel(num_threads=2)
        def parallel_take1d():
            # 调用take_nd函数，传入df["col"].values数组和indexer索引器
            take_nd(df["col"].values, indexer)

        # 将定义的并行测试函数赋值给实例变量self.parallel_take1d
        self.parallel_take1d = parallel_take1d

    # 计时方法，接收dtype参数
    def time_take1d(self, dtype):
        # 执行self.parallel_take1d函数
        self.parallel_take1d()


class ParallelKth:
    # 这部分代码完全依赖于_libs/中的代码，可以放在libs.py文件中

    # 定义测试次数
    number = 1
    # 定义重复次数
    repeat = 5

    # 初始化设置方法
    def setup(self):
        # 设置数组大小
        N = 10**7
        # 设置k值
        k = 5 * 10**5
        # 创建kwargs_list列表，包含两个字典，每个字典包含一个名为arr的随机数组
        kwargs_list = [{"arr": np.random.randn(N)}, {"arr": np.random.randn(N)}]

        # 定义并行测试函数，使用test_parallel装饰器，设置线程数为2，传入kwargs_list参数
        @test_parallel(num_threads=2, kwargs_list=kwargs_list)
        def parallel_kth_smallest(arr):
            # 调用algos.kth_smallest函数，传入arr数组和k值
            algos.kth_smallest(arr, k)

        # 将定义的并行测试函数赋值给实例变量self.parallel_kth_smallest
        self.parallel_kth_smallest = parallel_kth_smallest

    # 计时方法
    def time_kth_smallest(self):
        # 执行self.parallel_kth_smallest函数
        self.parallel_kth_smallest()


class ParallelDatetimeFields:
    # 初始化设置方法
    def setup(self):
        # 设置日期范围大小
        N = 10**6
        # 创建日期范围对象dti，从1900-01-01开始，周期为1分钟，总共N个周期
        self.dti = date_range("1900-01-01", periods=N, freq="min")
        # 将日期范围对象转换为周期对象，每个周期表示1天
        self.period = self.dti.to_period("D")

    # 年份字段操作计时方法
    def time_datetime_field_year(self):
        # 定义并行测试函数，使用test_parallel装饰器，设置线程数为2，接收dti参数
        @test_parallel(num_threads=2)
        def run(dti):
            # 访问日期范围对象的年份字段
            dti.year

        # 执行run函数，传入self.dti参数
        run(self.dti)

    # 日字段操作计时方法
    def time_datetime_field_day(self):
        # 定义并行测试函数，使用test_parallel装饰器，设置线程数为2，接收dti参数
        @test_parallel(num_threads=2)
        def run(dti):
            # 访问日期范围对象的日字段
            dti.day

        # 执行run函数，传入self.dti参数
        run(self.dti)

    # 每月天数字段操作计时方法
    def time_datetime_field_daysinmonth(self):
        # 定义并行测试函数，使用test_parallel装饰器，设置线程数为2，接收dti参数
        @test_parallel(num_threads=2)
        def run(dti):
            # 访问日期范围对象的每月天数字段
            dti.days_in_month

        # 执行run函数，传入self.dti参数
        run(self.dti)

    # 规范化日期操作计时方法
    def time_datetime_field_normalize(self):
        # 定义并行测试函数，使用test_parallel装饰器，设置线程数为2，接收dti参数
        @test_parallel(num_threads=2)
        def run(dti):
            # 规范化日期范围对象
            dti.normalize()

        # 执行run函数，传入self.dti参数
        run(self.dti)

    # 日期转换为周期操作计时方法
    def time_datetime_to_period(self):
        # 定义并行测试函数，使用test_parallel装饰器，设置线程数为2，接收dti参数
        @test_parallel(num_threads=2)
        def run(dti):
            # 将日期范围对象转换为以秒为周期单位的周期对象
            dti.to_period("s")

        # 执行run函数，传入self.dti参数
        run(self.dti)

    # 周期转换为日期操作计时方法
    def time_period_to_datetime(self):
        # 定义并行测试函数，使用test_parallel装饰器，设置线程数为2，接收period参数
        @test_parallel(num_threads=2)
        def run(period):
            # 将周期对象转换为时间戳对象
            period.to_timestamp()

        # 执行run函数，传入self.period参数
        run(self.period)


class ParallelRolling:
    # 定义参数列表，包含不同的滚动操作方法
    params = ["median", "mean", "min", "max", "var", "skew", "kurt", "std"]
    # 定义参数名称列表
    param_names = ["method"]
    def setup(self, method):
        # 设置窗口大小为100
        win = 100
        # 生成一个包含10万个随机数的数组
        arr = np.random.rand(100000)
        
        # 检查 DataFrame 类是否具有 rolling 方法
        if hasattr(DataFrame, "rolling"):
            # 如果有 rolling 方法，则创建 DataFrame 对象，并进行滚动窗口操作
            df = DataFrame(arr).rolling(win)

            @test_parallel(num_threads=2)
            def parallel_rolling():
                # 调用传入方法对滚动窗口进行并行操作
                getattr(df, method)()

            # 将并行滚动函数赋值给实例变量
            self.parallel_rolling = parallel_rolling
        
        # 如果没有 DataFrame 的 rolling 方法但有其他滚动方法
        elif have_rolling_methods:
            # 定义滚动方法字典
            rolling = {
                "median": rolling_median,
                "mean": rolling_mean,
                "min": rolling_min,
                "max": rolling_max,
                "var": rolling_var,
                "skew": rolling_skew,
                "kurt": rolling_kurt,
                "std": rolling_std,
            }

            @test_parallel(num_threads=2)
            def parallel_rolling():
                # 调用指定方法的滚动函数并进行并行操作
                rolling[method](arr, win)

            # 将并行滚动函数赋值给实例变量
            self.parallel_rolling = parallel_rolling
        
        # 如果都不满足，则抛出未实现错误
        else:
            raise NotImplementedError

    def time_rolling(self, method):
        # 调用并执行并行滚动函数
        self.parallel_rolling()
# 定义一个类 ParallelReadCSV，继承自 BaseIO
class ParallelReadCSV(BaseIO):
    # 类变量：设定默认值为 1 的 number 和 5 的 repeat
    number = 1
    repeat = 5
    # 类变量：参数列表包括三种数据类型："float"、"object"、"datetime"
    params = ["float", "object", "datetime"]
    # 类变量：参数名称列表，只有一个元素 "dtype"
    param_names = ["dtype"]

    # 方法：设置函数，在测试之前的准备工作
    def setup(self, dtype):
        # 设定行数为 10000，列数为 50
        rows = 10000
        cols = 50
        # 根据 dtype 参数的不同选择生成不同的 DataFrame
        if dtype == "float":
            # 生成随机数填充的 DataFrame
            df = DataFrame(np.random.randn(rows, cols))
        elif dtype == "datetime":
            # 生成带有时间索引的随机数填充的 DataFrame
            df = DataFrame(
                np.random.randn(rows, cols), index=date_range("1/1/2000", periods=rows)
            )
        elif dtype == "object":
            # 生成特定值填充的 DataFrame
            df = DataFrame(
                "foo", index=range(rows), columns=[f"object{num:03d}" for num in range(5)]
            )
        else:
            # 如果未知的 dtype 参数，抛出未实现错误
            raise NotImplementedError
        
        # 根据 dtype 参数生成文件名
        self.fname = f"__test_{dtype}__.csv"
        # 将 DataFrame 写入 CSV 文件
        df.to_csv(self.fname)
        
        # 嵌套函数：使用 @test_parallel 装饰器设定并行读取 CSV 文件的测试函数
        @test_parallel(num_threads=2)
        def parallel_read_csv():
            read_csv(self.fname)
        
        # 将并行读取 CSV 的函数赋值给实例变量
        self.parallel_read_csv = parallel_read_csv

    # 方法：测试读取 CSV 文件的时间
    def time_read_csv(self, dtype):
        # 调用并行读取 CSV 文件的测试函数
        self.parallel_read_csv()


# 定义一个类 ParallelFactorize
class ParallelFactorize:
    # 类变量：设定默认值为 1 的 number 和 5 的 repeat
    number = 1
    repeat = 5
    # 类变量：参数列表包括 2、4、8 三种线程数
    params = [2, 4, 8]
    # 类变量：参数名称列表，只有一个元素 "threads"
    param_names = ["threads"]

    # 方法：设置函数，在测试之前的准备工作
    def setup(self, threads):
        # 创建一个包含 100000 个字符串的索引对象
        strings = Index([f"i-{i}" for i in range(100000)], dtype=object)

        # 嵌套函数：使用 @test_parallel 装饰器设定并行执行 factorize 的测试函数
        @test_parallel(num_threads=threads)
        def parallel():
            factorize(strings)
        
        # 将并行执行 factorize 的函数赋值给实例变量
        self.parallel = parallel

        # 嵌套函数：顺序执行 factorize 的测试函数
        def loop():
            factorize(strings)
        
        # 将顺序执行 factorize 的函数赋值给实例变量
        self.loop = loop

    # 方法：测试并行执行 factorize 的时间
    def time_parallel(self, threads):
        # 调用并行执行 factorize 的测试函数
        self.parallel()

    # 方法：测试顺序执行 factorize 的时间
    def time_loop(self, threads):
        # 循环调用顺序执行 factorize 的测试函数，次数等于线程数
        for i in range(threads):
            self.loop()

# 导入 pandas_vb_common 模块的 setup 函数，跳过 isort 排序和 F401 noqa 标记
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```