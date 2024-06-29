# `D:\src\scipysrc\pandas\pandas\tests\io\sas\test_sas.py`

```
from io import StringIO  # 导入StringIO类，用于操作字符串缓冲区

import pytest  # 导入pytest模块，用于编写和运行测试用例

from pandas import read_sas  # 导入pandas中的read_sas函数，用于读取SAS文件
import pandas._testing as tm  # 导入pandas内部测试工具模块

class TestSas:
    def test_sas_buffer_format(self):
        # 测试用例：验证在缓冲区对象上读取SAS文件时必须指定格式字符串
        b = StringIO("")  # 创建一个空的StringIO对象

        msg = (
            "If this is a buffer object rather than a string "
            "name, you must specify a format string"
        )
        with pytest.raises(ValueError, match=msg):  # 使用pytest断言捕获ValueError异常，验证错误信息与msg匹配
            read_sas(b)  # 调用read_sas函数尝试在StringIO对象上读取SAS文件

    def test_sas_read_no_format_or_extension(self):
        # 测试用例：验证无法推断SAS文件格式时抛出错误
        msg = "unable to infer format of SAS file.+"  # 设置期望的错误信息模式
        with tm.ensure_clean("test_file_no_extension") as path:  # 使用测试工具确保测试路径干净
            with pytest.raises(ValueError, match=msg):  # 使用pytest断言捕获ValueError异常，验证错误信息与msg匹配
                read_sas(path)  # 调用read_sas函数尝试读取未指定格式或扩展名的SAS文件


def test_sas_archive(datapath):
    fname_uncompressed = datapath("io", "sas", "data", "airline.sas7bdat")  # 获取未压缩的SAS文件路径
    df_uncompressed = read_sas(fname_uncompressed)  # 使用read_sas读取未压缩的SAS文件数据
    fname_compressed = datapath("io", "sas", "data", "airline.sas7bdat.gz")  # 获取压缩的SAS文件路径
    df_compressed = read_sas(fname_compressed, format="sas7bdat")  # 使用read_sas读取压缩的SAS文件数据，指定格式为sas7bdat
    tm.assert_frame_equal(df_uncompressed, df_compressed)  # 使用测试工具验证未压缩和压缩的数据框是否相等
```