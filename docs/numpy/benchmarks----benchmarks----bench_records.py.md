# `.\numpy\benchmarks\benchmarks\bench_records.py`

```py
# 导入Benchmark类从common模块中
from .common import Benchmark

# 导入NumPy库并重命名为np
import numpy as np

# Records类继承Benchmark类，用于性能基准测试
class Records(Benchmark):
    
    # 设置方法，初始化测试数据
    def setup(self):
        # 创建一个包含1000个元素的NumPy数组，范围从0到999
        self.l50 = np.arange(1000)
        # 设置字段数量为10000
        self.fields_number = 10000
        # 创建包含self.l50数组的列表，长度为self.fields_number
        self.arrays = [self.l50 for _ in range(self.fields_number)]
        # 创建包含self.l50数组dtype字符串的列表，长度为self.fields_number
        self.formats = [self.l50.dtype.str for _ in range(self.fields_number)]
        # 将self.formats列表中的所有字符串用逗号连接成一个字符串
        self.formats_str = ','.join(self.formats)
        # 创建NumPy结构化数据类型，包含10000个字段，每个字段名为'field_i'（i为0到9999），类型为self.l50.dtype.str
        self.dtype_ = np.dtype(
            [
                ('field_{}'.format(i), self.l50.dtype.str)
                for i in range(self.fields_number)
            ]
        )
        # 将self.l50数组转换为字符串，然后复制self.fields_number次，赋值给self.buffer
        self.buffer = self.l50.tostring() * self.fields_number
    
    # 测试函数，使用指定的dtype创建记录数组
    def time_fromarrays_w_dtype(self):
        np._core.records.fromarrays(self.arrays, dtype=self.dtype_)
    
    # 测试函数，使用默认dtype创建记录数组
    def time_fromarrays_wo_dtype(self):
        np._core.records.fromarrays(self.arrays)
    
    # 测试函数，使用格式列表创建记录数组
    def time_fromarrays_formats_as_list(self):
        np._core.records.fromarrays(self.arrays, formats=self.formats)
    
    # 测试函数，使用格式字符串创建记录数组
    def time_fromarrays_formats_as_string(self):
        np._core.records.fromarrays(self.arrays, formats=self.formats_str)
    
    # 测试函数，使用指定的dtype从字符串创建记录数组
    def time_fromstring_w_dtype(self):
        np._core.records.fromstring(self.buffer, dtype=self.dtype_)
    
    # 测试函数，使用格式列表从字符串创建记录数组
    def time_fromstring_formats_as_list(self):
        np._core.records.fromstring(self.buffer, formats=self.formats)
    
    # 测试函数，使用格式字符串从字符串创建记录数组
    def time_fromstring_formats_as_string(self):
        np._core.records.fromstring(self.buffer, formats=self.formats_str)
```