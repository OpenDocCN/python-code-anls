# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\sql.py`

```
import sqlite3  # 导入 SQLite3 库

import numpy as np  # 导入 NumPy 库
from sqlalchemy import create_engine  # 从 SQLAlchemy 库导入 create_engine 函数

from pandas import (  # 从 Pandas 库导入以下对象
    DataFrame,  # 数据帧对象
    Index,  # 索引对象
    date_range,  # 日期范围生成函数
    read_sql_query,  # 读取 SQL 查询结果函数
    read_sql_table,  # 读取 SQL 表格函数
)


class SQL:
    params = ["sqlalchemy", "sqlite"]  # 支持的连接参数列表
    param_names = ["connection"]  # 参数名称列表

    def setup(self, connection):
        N = 10000  # 数据行数
        con = {  # 创建不同连接方式的字典
            "sqlalchemy": create_engine("sqlite:///:memory:"),  # SQLAlchemy 内存连接
            "sqlite": sqlite3.connect(":memory:"),  # SQLite 内存连接
        }
        self.table_name = "test_type"  # 数据表名称
        self.query_all = f"SELECT * FROM {self.table_name}"  # 查询所有数据的 SQL 语句
        self.con = con[connection]  # 根据参数选择连接对象
        self.df = DataFrame(  # 创建数据帧对象
            {
                "float": np.random.randn(N),  # 浮点数列
                "float_with_nan": np.random.randn(N),  # 带 NaN 值的浮点数列
                "string": ["foo"] * N,  # 字符串列
                "bool": [True] * N,  # 布尔值列
                "int": np.random.randint(0, N, size=N),  # 整数列
                "datetime": date_range("2000-01-01", periods=N, freq="s"),  # 日期时间列
            },
            index=Index([f"i-{i}" for i in range(N)], dtype=object),  # 设置索引
        )
        self.df.iloc[1000:3000, 1] = np.nan  # 将部分数据置为 NaN
        self.df["date"] = self.df["datetime"].dt.date  # 提取日期列
        self.df["time"] = self.df["datetime"].dt.time  # 提取时间列
        self.df["datetime_string"] = self.df["datetime"].astype(str)  # 将日期时间列转换为字符串
        self.df.to_sql(self.table_name, self.con, if_exists="replace")  # 将数据帧写入 SQL 表格

    def time_to_sql_dataframe(self, connection):
        self.df.to_sql("test1", self.con, if_exists="replace")  # 将数据帧写入 SQL 表格

    def time_read_sql_query(self, connection):
        read_sql_query(self.query_all, self.con)  # 从 SQL 数据库中读取数据


class WriteSQLDtypes:
    params = (  # 支持的连接参数和数据类型列表
        ["sqlalchemy", "sqlite"],  # 连接参数列表
        [  # 数据类型列表
            "float",
            "float_with_nan",
            "string",
            "bool",
            "int",
            "date",
            "time",
            "datetime",
        ],
    )
    param_names = ["connection", "dtype"]  # 参数名称列表

    def setup(self, connection, dtype):
        N = 10000  # 数据行数
        con = {  # 创建不同连接方式的字典
            "sqlalchemy": create_engine("sqlite:///:memory:"),  # SQLAlchemy 内存连接
            "sqlite": sqlite3.connect(":memory:"),  # SQLite 内存连接
        }
        self.table_name = "test_type"  # 数据表名称
        self.query_col = f"SELECT {dtype} FROM {self.table_name}"  # 查询特定列数据的 SQL 语句
        self.con = con[connection]  # 根据参数选择连接对象
        self.df = DataFrame(  # 创建数据帧对象
            {
                "float": np.random.randn(N),  # 浮点数列
                "float_with_nan": np.random.randn(N),  # 带 NaN 值的浮点数列
                "string": ["foo"] * N,  # 字符串列
                "bool": [True] * N,  # 布尔值列
                "int": np.random.randint(0, N, size=N),  # 整数列
                "datetime": date_range("2000-01-01", periods=N, freq="s"),  # 日期时间列
            },
            index=Index([f"i-{i}" for i in range(N)], dtype=object),  # 设置索引
        )
        self.df.iloc[1000:3000, 1] = np.nan  # 将部分数据置为 NaN
        self.df["date"] = self.df["datetime"].dt.date  # 提取日期列
        self.df["time"] = self.df["datetime"].dt.time  # 提取时间列
        self.df["datetime_string"] = self.df["datetime"].astype(str)  # 将日期时间列转换为字符串
        self.df.to_sql(self.table_name, self.con, if_exists="replace")  # 将数据帧写入 SQL 表格
    # 将DataFrame中指定的列以指定的数据类型写入SQL数据库中的"test1"表格，如果表格存在则替换
    def time_to_sql_dataframe_column(self, connection, dtype):
        self.df[[dtype]].to_sql("test1", self.con, if_exists="replace")

    # 从SQL数据库中读取执行查询所返回的结果集中的指定列数据
    def time_read_sql_query_select_column(self, connection, dtype):
        read_sql_query(self.query_col, self.con)
class ReadSQLTable:
    # 定义一个用于读取 SQL 表的类

    def setup(self):
        # 设置函数，初始化测试数据和数据库连接
        N = 10000
        self.table_name = "test"
        self.con = create_engine("sqlite:///:memory:")  # 在内存中创建 SQLite 引擎
        self.df = DataFrame(
            {
                "float": np.random.randn(N),  # 创建随机浮点数列
                "float_with_nan": np.random.randn(N),  # 创建含有 NaN 的随机浮点数列
                "string": ["foo"] * N,  # 创建字符串列
                "bool": [True] * N,  # 创建布尔值列
                "int": np.random.randint(0, N, size=N),  # 创建随机整数列
                "datetime": date_range("2000-01-01", periods=N, freq="s"),  # 创建时间序列
            },
            index=Index([f"i-{i}" for i in range(N)], dtype=object),  # 创建对象索引
        )
        self.df.iloc[1000:3000, 1] = np.nan  # 将部分数据设置为 NaN
        self.df["date"] = self.df["datetime"].dt.date  # 从 datetime 列中提取日期
        self.df["time"] = self.df["datetime"].dt.time  # 从 datetime 列中提取时间
        self.df["datetime_string"] = self.df["datetime"].astype(str)  # 将 datetime 列转换为字符串
        self.df.to_sql(self.table_name, self.con, if_exists="replace")  # 将 DataFrame 写入 SQLite 表中

    def time_read_sql_table_all(self):
        # 读取整个 SQL 表的性能测试函数
        read_sql_table(self.table_name, self.con)

    def time_read_sql_table_parse_dates(self):
        # 读取 SQL 表并解析日期时间的性能测试函数
        read_sql_table(
            self.table_name,
            self.con,
            columns=["datetime_string"],  # 仅选择 datetime_string 列
            parse_dates=["datetime_string"],  # 解析 datetime_string 列为日期时间格式
        )


class ReadSQLTableDtypes:
    # 定义一个用于测试不同数据类型列的 SQL 表读取类
    params = [
        "float",
        "float_with_nan",
        "string",
        "bool",
        "int",
        "date",
        "time",
        "datetime",
    ]
    param_names = ["dtype"]

    def setup(self, dtype):
        # 设置函数，初始化测试数据和数据库连接
        N = 10000
        self.table_name = "test"
        self.con = create_engine("sqlite:///:memory:")  # 在内存中创建 SQLite 引擎
        self.df = DataFrame(
            {
                "float": np.random.randn(N),  # 创建随机浮点数列
                "float_with_nan": np.random.randn(N),  # 创建含有 NaN 的随机浮点数列
                "string": ["foo"] * N,  # 创建字符串列
                "bool": [True] * N,  # 创建布尔值列
                "int": np.random.randint(0, N, size=N),  # 创建随机整数列
                "datetime": date_range("2000-01-01", periods=N, freq="s"),  # 创建时间序列
            },
            index=Index([f"i-{i}" for i in range(N)], dtype=object),  # 创建对象索引
        )
        self.df.iloc[1000:3000, 1] = np.nan  # 将部分数据设置为 NaN
        self.df["date"] = self.df["datetime"].dt.date  # 从 datetime 列中提取日期
        self.df["time"] = self.df["datetime"].dt.time  # 从 datetime 列中提取时间
        self.df["datetime_string"] = self.df["datetime"].astype(str)  # 将 datetime 列转换为字符串
        self.df.to_sql(self.table_name, self.con, if_exists="replace")  # 将 DataFrame 写入 SQLite 表中

    def time_read_sql_table_column(self, dtype):
        # 测试读取指定数据类型列的性能函数
        read_sql_table(self.table_name, self.con, columns=[dtype])


from ..pandas_vb_common import setup  # 导入设置函数，跳过 isort 排序检查
```