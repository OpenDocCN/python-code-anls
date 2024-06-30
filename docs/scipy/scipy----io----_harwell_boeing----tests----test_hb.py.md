# `D:\src\scipysrc\scipy\scipy\io\_harwell_boeing\tests\test_hb.py`

```
# 导入所需的模块和库
from io import StringIO  # 导入字符串IO模块中的StringIO类，用于操作内存中的文本数据
import tempfile  # 导入临时文件模块，用于创建临时文件和目录

import numpy as np  # 导入NumPy库，用于数值计算

from numpy.testing import assert_equal, \  # 导入NumPy测试模块中的断言函数
    assert_array_almost_equal_nulp  # 导入NumPy测试模块中的近似相等检查函数

from scipy.sparse import coo_matrix, csc_matrix, rand  # 导入SciPy稀疏矩阵和随机数生成函数

from scipy.io import hb_read, hb_write  # 导入SciPy IO模块中的Harwell-Boeing文件读写函数


SIMPLE = """\
No Title                                                                |No Key
             9             4             1             4
RUA                      100           100            10             0
(26I3)          (26I3)          (3E23.15)
1  2  2  2  2  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3
3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3
3  3  3  3  3  3  3  4  4  4  6  6  6  6  6  6  6  6  6  6  6  8  9  9  9  9
9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9 11
37 71 89 18 30 45 70 19 25 52
2.971243799687726e-01  3.662366682877375e-01  4.786962174699534e-01
6.490068647991184e-01  ```python
SIMPLE_MATRIX = coo_matrix(
    ((0.297124379969, 0.366236668288, 0.47869621747, 0.649006864799,
      0.0661749042483, 0.887037034319, 0.419647859016,
      0.564960307211, 0.993442388709, 0.691233499152,),
     (np.array([[36, 70, 88, 17, 29, 44, 69, 18, 24, 51],
                [0, 4, 58, 61, 61, 72, 72, 73, 99, 99]]))))



def assert_csc_almost_equal(r, l):
    # 将输入的稀疏矩阵r和l转换为压缩列格式（CSC），并进行比较
    r = csc_matrix(r)
    l = csc_matrix(l)
    # 使用NumPy的断言函数比较两个稀疏矩阵的indptr、indices和data是否相等
    assert_equal(r.indptr, l.indptr)
    assert_equal(r.indices, l.indices)
    assert_array_almost_equal_nulp(r.data, l.data, 10000)



class TestHBReader:
    def test_simple(self):
        # 从SIMPLE字符串创建Harwell-Boeing格式的稀疏矩阵m
        m = hb_read(StringIO(SIMPLE))
        # 使用自定义的断言函数assert_csc_almost_equal检查m与预先定义的SIMPLE_MATRIX的近似相等性
        assert_csc_almost_equal(m, SIMPLE_MATRIX)



class TestHBReadWrite:

    def check_save_load(self, value):
        # 使用临时文件进行测试，写入value到文件并再次读取并比较
        with tempfile.NamedTemporaryFile(mode='w+t') as file:
            hb_write(file, value)  # 将value写入临时文件
            file.file.seek(0)  # 将文件指针移动到文件开头
            value_loaded = hb_read(file)  # 从临时文件读取数据
        # 使用自定义的断言函数assert_csc_almost_equal检查value和value_loaded的近似相等性
        assert_csc_almost_equal(value, value_loaded)

    def test_simple(self):
        # 创建一个随机稀疏矩阵random_matrix
        random_matrix = rand(10, 100, 0.1)
        # 针对不同的稀疏矩阵格式进行测试
        for matrix_format in ('coo', 'csc', 'csr', 'bsr', 'dia', 'dok', 'lil'):
            matrix = random_matrix.asformat(matrix_format, copy=False)  # 将random_matrix转换为指定格式的稀疏矩阵
            self.check_save_load(matrix)  # 调用check_save_load方法，测试保存和加载的功能
```