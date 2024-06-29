# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\hdf.py`

```
# 导入 numpy 库，并将其重命名为 np
import numpy as np

# 从 pandas 库中导入以下子模块和函数
# DataFrame：用于创建和操作数据帧的类
# HDFStore：用于读写 HDF5 文件的类
# Index：用于创建和操作索引的类
# date_range：用于生成日期范围的函数
# read_hdf：用于从 HDF5 文件中读取数据的函数
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    date_range,
    read_hdf,
)

# 从 ..pandas_vb_common 模块中导入 BaseIO 类
from ..pandas_vb_common import BaseIO

# 定义一个继承自 BaseIO 类的 HDFStoreDataFrame 类
class HDFStoreDataFrame(BaseIO):
    # 初始化方法，用于设置数据和变量
    def setup(self):
        # 设置一个整数 N，表示数据集的大小
        N = 25000
        # 创建一个索引对象 index，包含 N 个以字符串形式命名的索引值
        index = Index([f"i-{i}" for i in range(N)], dtype=object)
        
        # 创建一个名为 self.df 的 DataFrame 对象，包含两列随机浮点数数据，使用上面创建的索引
        self.df = DataFrame(
            {"float1": np.random.randn(N), "float2": np.random.randn(N)}, index=index
        )
        
        # 创建一个名为 self.df_mixed 的 DataFrame 对象，包含多种数据类型的列，使用上面创建的索引
        self.df_mixed = DataFrame(
            {
                "float1": np.random.randn(N),
                "float2": np.random.randn(N),
                "string1": ["foo"] * N,
                "bool1": [True] * N,
                "int1": np.random.randint(0, N, size=N),
            },
            index=index,
        )
        
        # 创建一个名为 self.df_wide 的 DataFrame 对象，包含随机数据，行数为 N，列数为 100
        self.df_wide = DataFrame(np.random.randn(N, 100))
        
        # 设置 self.start_wide 和 self.stop_wide 变量，分别表示 wide 数据帧的起始和结束索引
        self.start_wide = self.df_wide.index[10000]
        self.stop_wide = self.df_wide.index[15000]
        
        # 创建一个名为 self.df2 的 DataFrame 对象，包含随机浮点数数据，使用日期范围作为索引
        self.df2 = DataFrame(
            {"float1": np.random.randn(N), "float2": np.random.randn(N)},
            index=date_range("1/1/2000", periods=N),
        )
        
        # 设置 self.start 和 self.stop 变量，分别表示 df2 数据帧的起始和结束索引
        self.start = self.df2.index[10000]
        self.stop = self.df2.index[15000]
        
        # 创建一个名为 self.df_wide2 的 DataFrame 对象，包含随机数据，行数为 N，列数为 100，使用日期范围作为索引
        self.df_wide2 = DataFrame(
            np.random.randn(N, 100), index=date_range("1/1/2000", periods=N)
        )
        
        # 创建一个名为 self.df_dc 的 DataFrame 对象，包含随机数据，行数为 N，列数为 10，列名为 C000 到 C009
        self.df_dc = DataFrame(
            np.random.randn(N, 10), columns=[f"C{i:03d}" for i in range(10)]
        )

        # 设置文件名变量 self.fname，用于存储 HDF5 文件
        self.fname = "__test__.h5"

        # 创建一个 HDFStore 对象 self.store，打开指定文件名的 HDF5 存储文件
        self.store = HDFStore(self.fname)
        
        # 将 self.df 存储到 HDF5 文件中，键名为 "fixed"
        self.store.put("fixed", self.df)
        
        # 将 self.df_mixed 存储到 HDF5 文件中，键名为 "fixed_mixed"
        self.store.put("fixed_mixed", self.df_mixed)
        
        # 将 self.df2 追加存储到 HDF5 文件中，键名为 "table"
        self.store.append("table", self.df2)
        
        # 将 self.df_mixed 追加存储到 HDF5 文件中，键名为 "table_mixed"
        self.store.append("table_mixed", self.df_mixed)
        
        # 将 self.df_wide 追加存储到 HDF5 文件中，键名为 "table_wide"
        self.store.append("table_wide", self.df_wide)
        
        # 将 self.df_wide2 追加存储到 HDF5 文件中，键名为 "table_wide2"
        self.store.append("table_wide2", self.df_wide2)

    # 结束方法，用于清理和关闭 HDF5 存储文件
    def teardown(self):
        # 关闭 HDFStore 对象 self.store
        self.store.close()
        # 调用父类的 remove 方法，删除指定文件名的 HDF5 文件
        self.remove(self.fname)

    # 读取固定键名为 "fixed" 的数据
    def time_read_store(self):
        self.store.get("fixed")

    # 读取固定键名为 "fixed_mixed" 的数据
    def time_read_store_mixed(self):
        self.store.get("fixed_mixed")

    # 写入固定键名为 "fixed_write" 的数据
    def time_write_store(self):
        self.store.put("fixed_write", self.df)

    # 写入固定键名为 "fixed_mixed_write" 的数据
    def time_write_store_mixed(self):
        self.store.put("fixed_mixed_write", self.df_mixed)

    # 读取键名为 "table_mixed" 的数据
    def time_read_store_table_mixed(self):
        self.store.select("table_mixed")

    # 写入键名为 "table_mixed_write" 的数据
    def time_write_store_table_mixed(self):
        self.store.append("table_mixed_write", self.df_mixed)

    # 读取键名为 "table" 的数据
    def time_read_store_table(self):
        self.store.select("table")

    # 写入键名为 "table_write" 的数据
    def time_write_store_table(self):
        self.store.append("table_write", self.df)

    # 读取键名为 "table_wide" 的数据
    def time_read_store_table_wide(self):
        self.store.select("table_wide")

    # 写入键名为 "table_wide_write" 的数据
    def time_write_store_table_wide(self):
        self.store.append("table_wide_write", self.df_wide)

    # 写入带有数据列索引的键名为 "table_dc_write" 的数据
    def time_write_store_table_dc(self):
        self.store.append("table_dc_write", self.df_dc, data_columns=True)
    # 定义一个方法，用于执行广泛表的查询操作
    def time_query_store_table_wide(self):
        # 调用存储对象的select方法，查询名为"table_wide"的表，
        # 并指定条件为index大于self.start_wide并且小于self.stop_wide
        self.store.select(
            "table_wide", where="index > self.start_wide and index < self.stop_wide"
        )

    # 定义一个方法，用于执行普通表的查询操作
    def time_query_store_table(self):
        # 调用存储对象的select方法，查询名为"table"的表，
        # 并指定条件为index大于self.start并且小于self.stop
        self.store.select("table", where="index > self.start and index < self.stop")

    # 定义一个方法，返回存储对象的字符串表示形式
    def time_store_repr(self):
        # 调用存储对象的repr方法，返回其字符串表示形式
        repr(self.store)

    # 定义一个方法，返回存储对象的字符串表示形式
    def time_store_str(self):
        # 调用存储对象的str方法，返回其字符串表示形式
        str(self.store)

    # 定义一个方法，获取存储对象的信息
    def time_store_info(self):
        # 调用存储对象的info方法，获取其信息
        self.store.info()
class HDF(BaseIO):
    # HDF 类继承自 BaseIO 类
    params = ["table", "fixed"]
    # 定义类变量 params，包含表和固定两个参数
    param_names = ["format"]
    # 定义类变量 param_names，包含格式参数名称为 format

    def setup(self, format):
        # 定义实例方法 setup，接受参数 format
        self.fname = "__test__.h5"
        # 设置实例变量 fname 为 "__test__.h5"
        N = 100000
        C = 5
        # 初始化变量 N 和 C 分别为 100000 和 5
        self.df = DataFrame(
            np.random.randn(N, C),
            columns=[f"float{i}" for i in range(C)],
            index=date_range("20000101", periods=N, freq="h"),
        )
        # 创建 DataFrame 实例 df，包含随机数数据，列名为 "float0" 到 "float4"，索引为从 "20000101" 开始的 N 个小时频率的日期范围
        self.df["object"] = Index([f"i-{i}" for i in range(N)], dtype=object)
        # 在 df 中添加名为 "object" 的列，包含对象类型的索引值，形如 "i-0" 到 "i-99999"
        self.df.to_hdf(self.fname, key="df", format=format)
        # 将 df 对象以 HDF5 格式写入文件 fname 中，使用 key "df"，指定写入格式为 format

        # Numeric df
        self.df1 = self.df.copy()
        # 复制 df 到 df1
        self.df1 = self.df1.reset_index()
        # 重置 df1 的索引
        self.df1.to_hdf(self.fname, key="df1", format=format)
        # 将 df1 对象以 HDF5 格式写入文件 fname 中，使用 key "df1"，指定写入格式为 format

    def time_read_hdf(self, format):
        # 定义实例方法 time_read_hdf，接受参数 format
        read_hdf(self.fname, "df")
        # 调用 read_hdf 函数，读取 fname 文件中 key 为 "df" 的数据

    def peakmem_read_hdf(self, format):
        # 定义实例方法 peakmem_read_hdf，接受参数 format
        read_hdf(self.fname, "df")
        # 调用 read_hdf 函数，读取 fname 文件中 key 为 "df" 的数据

    def time_write_hdf(self, format):
        # 定义实例方法 time_write_hdf，接受参数 format
        self.df.to_hdf(self.fname, key="df", format=format)
        # 将 df 对象以 HDF5 格式写入文件 fname 中，使用 key "df"，指定写入格式为 format
```