# `D:\src\scipysrc\pandas\asv_bench\benchmarks\stat_ops.py`

```
import numpy as np  # 导入NumPy库，用于数值计算

import pandas as pd  # 导入Pandas库，用于数据处理和分析

ops = ["mean", "sum", "median", "std", "skew", "kurt", "prod", "sem", "var"]
# 定义操作列表，包括平均值、求和、中位数、标准差、偏度、峰度、乘积、标准误差、方差

class FrameOps:
    params = [ops, ["float", "int", "Int64"], [0, 1, None]]
    param_names = ["op", "dtype", "axis"]

    def setup(self, op, dtype, axis):
        # 生成一个形状为(100000, 4)的随机数值数组
        values = np.random.randn(100000, 4)
        if dtype == "Int64":
            values = values.astype(int)  # 如果dtype为"Int64"，则将数组转换为整数类型
        df = pd.DataFrame(values).astype(dtype)  # 创建DataFrame，并根据dtype转换数据类型
        self.df_func = getattr(df, op)  # 获取DataFrame对象中指定操作的方法

    def time_op(self, op, dtype, axis):
        self.df_func(axis=axis)  # 执行DataFrame对象中指定操作的方法，传入axis参数


class FrameMixedDtypesOps:
    params = [ops, [0, 1, None]]
    param_names = ["op", "axis"]

    def setup(self, op, axis):
        if op in ("sum", "skew", "kurt", "prod", "sem", "var") or (
            (op, axis)
            in (
                ("mean", 1),
                ("mean", None),
                ("median", 1),
                ("median", None),
                ("std", 1),
                ("std", None),
            )
        ):
            # 如果操作为特定的聚合函数且涉及日期时间类型，抛出未实现错误
            raise NotImplementedError

        N = 1_000_000
        # 创建DataFrame，包含列"f"为标准正态分布随机数，列"i"为指定范围内的随机整数，列"ts"为日期时间序列
        df = pd.DataFrame(
            {
                "f": np.random.normal(0.0, 1.0, N),
                "i": np.random.randint(0, N, N),
                "ts": pd.date_range(start="1/1/2000", periods=N, freq="h"),
            }
        )

        self.df_func = getattr(df, op)  # 获取DataFrame对象中指定操作的方法

    def time_op(self, op, axis):
        self.df_func(axis=axis)  # 执行DataFrame对象中指定操作的方法，传入axis参数


class FrameMultiIndexOps:
    params = [ops]
    param_names = ["op"]

    def setup(self, op):
        # 创建多级索引，包含三个级别，每级对应的标签和代码
        levels = [np.arange(10), np.arange(100), np.arange(100)]
        codes = [
            np.arange(10).repeat(10000),
            np.tile(np.arange(100).repeat(100), 10),
            np.tile(np.tile(np.arange(100), 100), 10),
        ]
        index = pd.MultiIndex(levels=levels, codes=codes)  # 创建多级索引对象
        df = pd.DataFrame(np.random.randn(len(index), 4), index=index)  # 创建DataFrame，数据为随机数值
        self.df_func = getattr(df, op)  # 获取DataFrame对象中指定操作的方法

    def time_op(self, op):
        self.df_func()  # 执行DataFrame对象中指定操作的方法


class SeriesOps:
    params = [ops, ["float", "int"]]
    param_names = ["op", "dtype"]

    def setup(self, op, dtype):
        # 创建Series，包含随机数值，并根据dtype指定的数据类型转换
        s = pd.Series(np.random.randn(100000)).astype(dtype)
        self.s_func = getattr(s, op)  # 获取Series对象中指定操作的方法

    def time_op(self, op, dtype):
        self.s_func()  # 执行Series对象中指定操作的方法


class SeriesMultiIndexOps:
    params = [ops]
    param_names = ["op"]

    def setup(self, op):
        # 创建多级索引Series，包含随机数值，并根据op指定的操作方法
        levels = [np.arange(10), np.arange(100), np.arange(100)]
        codes = [
            np.arange(10).repeat(10000),
            np.tile(np.arange(100).repeat(100), 10),
            np.tile(np.tile(np.arange(100), 100), 10),
        ]
        index = pd.MultiIndex(levels=levels, codes=codes)  # 创建多级索引对象
        s = pd.Series(np.random.randn(len(index)), index=index)  # 创建Series，数据为随机数值
        self.s_func = getattr(s, op)  # 获取Series对象中指定操作的方法

    def time_op(self, op):
        self.s_func()  # 执行Series对象中指定操作的方法


class Rank:
    params = [["DataFrame", "Series"], [True, False]]
    param_names = ["constructor", "pct"]
    # 设置函数，初始化数据
    def setup(self, constructor, pct):
        # 生成包含10^5个随机数的数组
        values = np.random.randn(10**5)
        # 使用给定的构造函数名，通过getattr调用pandas库中对应的构造函数，生成数据结构并赋值给self.data
        self.data = getattr(pd, constructor)(values)

    # 计时函数，用于计算数据排名
    def time_rank(self, constructor, pct):
        # 对self.data进行排名，根据参数pct指定的百分比
        self.data.rank(pct=pct)

    # 计时函数，用于计算旧版平均值
    def time_average_old(self, constructor, pct):
        # 对self.data进行排名，根据参数pct指定的百分比，并除以数据长度，计算平均值
        self.data.rank(pct=pct) / len(self.data)
class Correlation:
    # 参数列表，包含一个子列表，用于存储相关系数计算方法的名称
    params = [["spearman", "kendall", "pearson"]]
    # 参数名称列表，仅包含一个元素 "method"
    param_names = ["method"]

    # 初始化方法，用于设置测试环境
    def setup(self, method):
        # 创建一个500行、15列的随机数数据框
        self.df = pd.DataFrame(np.random.randn(500, 15))
        # 创建第二个500行、15列的随机数数据框
        self.df2 = pd.DataFrame(np.random.randn(500, 15))
        # 创建一个500行、100列的随机数数据框
        self.df_wide = pd.DataFrame(np.random.randn(500, 100))
        # 在 df_wide 数据框中，根据指定的条件生成 NaN 值
        self.df_wide_nans = self.df_wide.where(np.random.random((500, 100)) < 0.9)
        # 创建一个包含500个随机数的 Series 对象
        self.s = pd.Series(np.random.randn(500))
        # 创建第二个包含500个随机数的 Series 对象
        self.s2 = pd.Series(np.random.randn(500))

    # 计算数据框 df 的相关系数，使用给定的方法
    def time_corr(self, method):
        self.df.corr(method=method)

    # 计算宽格式数据框 df_wide 的相关系数，使用给定的方法
    def time_corr_wide(self, method):
        self.df_wide.corr(method=method)

    # 计算带有 NaN 值的宽格式数据框 df_wide_nans 的相关系数，使用给定的方法
    def time_corr_wide_nans(self, method):
        self.df_wide_nans.corr(method=method)

    # 计算 Series 对象 s 和 s2 之间的相关系数，使用给定的方法
    def time_corr_series(self, method):
        self.s.corr(self.s2, method=method)

    # 计算数据框 df 与 df2 之间的列相关系数，使用给定的方法
    def time_corrwith_cols(self, method):
        self.df.corrwith(self.df2, method=method)

    # 计算数据框 df 与 df2 之间的行相关系数，使用给定的方法
    def time_corrwith_rows(self, method):
        self.df.corrwith(self.df2, axis=1, method=method)


class Covariance:
    # 参数列表，为空列表
    params = []
    # 参数名称列表，为空列表
    param_names = []

    # 初始化方法，用于设置测试环境
    def setup(self):
        # 创建一个包含100000个随机数的 Series 对象
        self.s = pd.Series(np.random.randn(100000))
        # 创建第二个包含100000个随机数的 Series 对象
        self.s2 = pd.Series(np.random.randn(100000))

    # 计算 Series 对象 s 和 s2 之间的协方差
    def time_cov_series(self):
        self.s.cov(self.s2)
```