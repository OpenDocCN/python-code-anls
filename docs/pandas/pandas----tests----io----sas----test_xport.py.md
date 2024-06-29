# `D:\src\scipysrc\pandas\pandas\tests\io\sas\test_xport.py`

```
import numpy as np  # 导入NumPy库，用于处理数值数据
import pytest  # 导入pytest库，用于编写和运行测试用例

import pandas as pd  # 导入Pandas库，用于数据分析和处理
import pandas._testing as tm  # 导入Pandas的测试模块

from pandas.io.sas.sasreader import read_sas  # 从Pandas的SAS读取模块中导入read_sas函数

# CSV versions of test xpt files were obtained using the R foreign library
# 使用R的foreign库获取的测试xpt文件的CSV版本

# Numbers in a SAS xport file are always float64, so need to convert
# before making comparisons.
# SAS xport文件中的数字始终为float64类型，因此在进行比较之前需要进行转换


def numeric_as_float(data):
    for v in data.columns:
        if data[v].dtype is np.dtype("int64"):
            data[v] = data[v].astype(np.float64)
            # 如果数据类型是int64，则将其转换为float64类型


class TestXport:
    @pytest.mark.slow  # 标记为慢速测试用例
    def test1_basic(self, datapath):
        # Tests with DEMO_G.xpt (all numeric file)
        # 使用DEMO_G.xpt进行测试（全数值文件）

        # Compare to this
        # 与以下CSV文件进行比较
        file01 = datapath("io", "sas", "data", "DEMO_G.xpt")
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        numeric_as_float(data_csv)

        # Read full file
        # 读取完整文件
        data = read_sas(file01, format="xport")
        tm.assert_frame_equal(data, data_csv)
        num_rows = data.shape[0]

        # Test reading beyond end of file
        # 测试超出文件末尾的读取
        with read_sas(file01, format="xport", iterator=True) as reader:
            data = reader.read(num_rows + 100)
        assert data.shape[0] == num_rows

        # Test incremental read with `read` method.
        # 使用`read`方法进行增量读取测试
        with read_sas(file01, format="xport", iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

        # Test incremental read with `get_chunk` method.
        # 使用`get_chunk`方法进行增量读取测试
        with read_sas(file01, format="xport", chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

        # Test read in loop
        # 在循环中进行读取测试
        m = 0
        with read_sas(file01, format="xport", chunksize=100) as reader:
            for x in reader:
                m += x.shape[0]
        assert m == num_rows

        # Read full file with `read_sas` method
        # 使用`read_sas`方法读取完整文件
        data = read_sas(file01)
        tm.assert_frame_equal(data, data_csv)

    def test1_index(self, datapath):
        # Tests with DEMO_G.xpt using index (all numeric file)
        # 使用带索引的DEMO_G.xpt进行测试（全数值文件）

        # Compare to this
        # 与以下CSV文件进行比较
        file01 = datapath("io", "sas", "data", "DEMO_G.xpt")
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        data_csv = data_csv.set_index("SEQN")
        numeric_as_float(data_csv)

        # Read full file
        # 读取完整文件
        data = read_sas(file01, index="SEQN", format="xport")
        tm.assert_frame_equal(data, data_csv, check_index_type=False)

        # Test incremental read with `read` method.
        # 使用`read`方法进行增量读取测试
        with read_sas(file01, index="SEQN", format="xport", iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

        # Test incremental read with `get_chunk` method.
        # 使用`get_chunk`方法进行增量读取测试
        with read_sas(file01, index="SEQN", format="xport", chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)
    # 定义测试函数 test1_incremental
    def test1_incremental(self, datapath):
        # 用 DEMO_G.xpt 文件进行测试，以增量方式读取完整文件
        file01 = datapath("io", "sas", "data", "DEMO_G.xpt")
        # 读取 csv 文件的数据
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        # 将数据以 "SEQN" 为索引
        data_csv = data_csv.set_index("SEQN")
        # 将浮点数数据转换为浮点型
        numeric_as_float(data_csv)
        
        # 使用 read_sas 函数读取文件，并以 "SEQN" 为索引，每次读取 1000 行数据
        with read_sas(file01, index="SEQN", chunksize=1000) as reader:
            all_data = list(reader)
        # 将所有数据拼接成一个数据框
        data = pd.concat(all_data, axis=0)
        
        # 检查两个数据框是否相等
        tm.assert_frame_equal(data, data_csv, check_index_type=False)
    
    # 定义测试函数 test2
    def test2(self, datapath):
        # 用 SSHSV1_A.xpt 文件进行测试
        file02 = datapath("io", "sas", "data", "SSHSV1_A.xpt")
        # 读取对应的 csv 文件的数据
        data_csv = pd.read_csv(file02.replace(".xpt", ".csv"))
        # 将数据转换为浮点型
        numeric_as_float(data_csv)
        
        # 使用 read_sas 函数读取文件
        data = read_sas(file02)
        # 检查两个数据框是否相等
        tm.assert_frame_equal(data, data_csv)
    
    # 定义测试函数 test2_binary
    def test2_binary(self, datapath):
        # 用 SSHSV1_A.xpt 文件进行测试，以二进制文件的方式读取
        file02 = datapath("io", "sas", "data", "SSHSV1_A.xpt")
        # 读取对应的 csv 文件的数据
        data_csv = pd.read_csv(file02.replace(".xpt", ".csv"))
        # 将数据转换为浮点型
        numeric_as_float(data_csv)
        
        # 以二进制文件的方式打开文件
        with open(file02, "rb") as fd:
            # 确保如果我们传递了一个打开的文件，我们不会在 read_sas 中错误地关闭它
            data = read_sas(fd, format="xport")
        
        # 检查两个数据框是否相等
        tm.assert_frame_equal(data, data_csv)
    
    # 定义测试函数 test_multiple_types
    def test_multiple_types(self, datapath):
        # 用 DRXFCD_G.xpt 文件进行测试（包含文本和数值变量）
        file03 = datapath("io", "sas", "data", "DRXFCD_G.xpt")
        # 读取对应的 csv 文件的数据
        data_csv = pd.read_csv(file03.replace(".xpt", ".csv"))
        # 使用 utf-8 编码读取文件
        data = read_sas(file03, encoding="utf-8")
        # 检查两个数据框是否相等
        tm.assert_frame_equal(data, data_csv)
    
    # 定义测试函数 test_truncated_float_support
    def test_truncated_float_support(self, datapath):
        # 用 paxraw_d_short.xpt 文件进行测试，这是 http://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/PAXRAW_D.ZIP 的缩短版本，包含了截断的浮点数（在这种情况下是 5 个字节）
        file04 = datapath("io", "sas", "data", "paxraw_d_short.xpt")
        # 读取对应的 csv 文件的数据
        data_csv = pd.read_csv(file04.replace(".xpt", ".csv"))
        # 使用 xport 格式读取文件
        data = read_sas(file04, format="xport")
        # 检查两个数据框是否相等，并且类型为 int64
        tm.assert_frame_equal(data.astype("int64"), data_csv)
    
    # 定义测试函数 test_cport_header_found_raises
    def test_cport_header_found_raises(self, datapath):
        # 用 DEMO_PUF.cpt 文件进行测试，这是 https://www.cms.gov/files/zip/puf2019.zip 中 puf2019_1_fall.xpt 的开头（尽管扩展名是 .cpt）
        msg = "Header record indicates a CPORT file, which is not readable."
        # 确保当我们传入了格式为 cpt 的文件时，会引发 ValueError 错误，错误信息为 msg
        with pytest.raises(ValueError, match=msg):
            read_sas(datapath("io", "sas", "data", "DEMO_PUF.cpt"), format="xport")
```