# `D:\src\scipysrc\pandas\asv_bench\benchmarks\pandas_vb_common.py`

```
from importlib import import_module  # 导入模块导入函数 import_module

import os  # 导入操作系统相关功能的模块

import numpy as np  # 导入 NumPy 数值计算库

import pandas as pd  # 导入 Pandas 数据处理库

# 为了兼容性引入 lib 模块
for imp in ["pandas._libs.lib", "pandas.lib"]:
    try:
        lib = import_module(imp)  # 尝试导入指定的模块
        break  # 成功导入后退出循环
    except (ImportError, TypeError, ValueError):
        pass  # 处理导入错误

# 兼容性导入测试模块
try:
    import pandas._testing as tm  # 尝试导入 Pandas 的测试模块
except ImportError:
    import pandas.util.testing as tm  # 若导入失败，则使用备选的测试模块，忽略 F401 错误


numeric_dtypes = [
    np.int64,
    np.int32,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.int16,
    np.int8,
    np.uint16,
    np.uint8,
]
datetime_dtypes = [np.datetime64, np.timedelta64]  # 定义日期时间类型和时间差类型的列表
string_dtypes = [object]  # 定义字符串类型的列表
try:
    extension_dtypes = [
        pd.Int8Dtype,
        pd.Int16Dtype,
        pd.Int32Dtype,
        pd.Int64Dtype,
        pd.UInt8Dtype,
        pd.UInt16Dtype,
        pd.UInt32Dtype,
        pd.UInt64Dtype,
        pd.CategoricalDtype,
        pd.IntervalDtype,
        pd.DatetimeTZDtype("ns", "UTC"),
        pd.PeriodDtype("D"),
    ]  # 尝试定义扩展数据类型的列表，包括整数类型、分类数据类型、时间区间等
except AttributeError:
    extension_dtypes = []  # 若属性错误，则为空列表


def setup(*args, **kwargs):
    # 此函数被导入到每个基准文件中，用于在每个函数运行前设置随机种子。
    # 参考：https://asv.readthedocs.io/en/latest/writing_benchmarks.html
    np.random.seed(1234)  # 设置 NumPy 随机种子为 1234


class BaseIO:
    """
    IO 基准测试的基类
    """

    fname = None  # 文件名属性默认为空

    def remove(self, f):
        """删除已创建的文件"""
        try:
            os.remove(f)  # 尝试删除指定的文件
        except OSError:
            # 在 Windows 上，尝试删除正在使用的文件会引发异常
            pass

    def teardown(self, *args, **kwargs):
        self.remove(self.fname)  # 调用 remove 方法删除存储在 fname 属性中的文件名
```