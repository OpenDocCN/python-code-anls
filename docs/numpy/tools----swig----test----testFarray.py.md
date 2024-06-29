# `.\numpy\tools\swig\test\testFarray.py`

```
#!/usr/bin/env python3
# System imports
# 导入系统模块
from distutils.util import get_platform
import os
import sys
import unittest

# Import NumPy
# 导入 NumPy 库
import numpy as np
# 解析 NumPy 版本号
major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
# 根据 NumPy 的主版本号判断错误类型
if major == 0: BadListError = TypeError
else:          BadListError = ValueError

# Add the distutils-generated build directory to the python search path and then
# import the extension module
# 将 distutils 生成的构建目录添加到 Python 搜索路径中，并导入扩展模块
libDir = "lib.{}-{}.{}".format(get_platform(), *sys.version_info[:2])
sys.path.insert(0, os.path.join("build", libDir))
import Farray

######################################################################

class FarrayTestCase(unittest.TestCase):

    def setUp(self):
        # 设置测试用例的初始条件
        self.nrows = 5
        self.ncols = 4
        # 创建 Farray 对象
        self.array = Farray.Farray(self.nrows, self.ncols)

    def testConstructor1(self):
        "Test Farray size constructor"
        # 测试 Farray 的大小构造函数
        self.assertTrue(isinstance(self.array, Farray.Farray))

    def testConstructor2(self):
        "Test Farray copy constructor"
        # 测试 Farray 的复制构造函数
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array[i, j] = i + j
        # 创建 Farray 对象的副本
        arrayCopy = Farray.Farray(self.array)
        # 断言副本与原始对象相等
        self.assertTrue(arrayCopy == self.array)

    def testConstructorBad1(self):
        "Test Farray size constructor, negative nrows"
        # 测试 Farray 的大小构造函数，负数行数
        self.assertRaises(ValueError, Farray.Farray, -4, 4)

    def testConstructorBad2(self):
        "Test Farray size constructor, negative ncols"
        # 测试 Farray 的大小构造函数，负数列数
        self.assertRaises(ValueError, Farray.Farray, 4, -4)

    def testNrows(self):
        "Test Farray nrows method"
        # 测试 Farray 的 nrows 方法
        self.assertTrue(self.array.nrows() == self.nrows)

    def testNcols(self):
        "Test Farray ncols method"
        # 测试 Farray 的 ncols 方法
        self.assertTrue(self.array.ncols() == self.ncols)

    def testLen(self):
        "Test Farray __len__ method"
        # 测试 Farray 的 __len__ 方法
        self.assertTrue(len(self.array) == self.nrows*self.ncols)

    def testSetGet(self):
        "Test Farray __setitem__, __getitem__ methods"
        # 测试 Farray 的 __setitem__ 和 __getitem__ 方法
        m = self.nrows
        n = self.ncols
        for i in range(m):
            for j in range(n):
                self.array[i, j] = i*j
        for i in range(m):
            for j in range(n):
                self.assertTrue(self.array[i, j] == i*j)

    def testSetBad1(self):
        "Test Farray __setitem__ method, negative row"
        # 测试 Farray 的 __setitem__ 方法，负数行号
        self.assertRaises(IndexError, self.array.__setitem__, (-1, 3), 0)

    def testSetBad2(self):
        "Test Farray __setitem__ method, negative col"
        # 测试 Farray 的 __setitem__ 方法，负数列号
        self.assertRaises(IndexError, self.array.__setitem__, (1, -3), 0)

    def testSetBad3(self):
        "Test Farray __setitem__ method, out-of-range row"
        # 测试 Farray 的 __setitem__ 方法，超出范围的行号
        self.assertRaises(IndexError, self.array.__setitem__, (self.nrows+1, 0), 0)

    def testSetBad4(self):
        "Test Farray __setitem__ method, out-of-range col"
        # 测试 Farray 的 __setitem__ 方法，超出范围的列号
        self.assertRaises(IndexError, self.array.__setitem__, (0, self.ncols+1), 0)
    # 测试 Farray 的 __getitem__ 方法，检查负索引行为是否引发 IndexError 异常
    def testGetBad1(self):
        "Test Farray __getitem__ method, negative row"
        self.assertRaises(IndexError, self.array.__getitem__, (-1, 3))

    # 测试 Farray 的 __getitem__ 方法，检查负索引列是否引发 IndexError 异常
    def testGetBad2(self):
        "Test Farray __getitem__ method, negative col"
        self.assertRaises(IndexError, self.array.__getitem__, (1, -3))

    # 测试 Farray 的 __getitem__ 方法，检查超出范围的行索引是否引发 IndexError 异常
    def testGetBad3(self):
        "Test Farray __getitem__ method, out-of-range row"
        self.assertRaises(IndexError, self.array.__getitem__, (self.nrows+1, 0))

    # 测试 Farray 的 __getitem__ 方法，检查超出范围的列索引是否引发 IndexError 异常
    def testGetBad4(self):
        "Test Farray __getitem__ method, out-of-range col"
        self.assertRaises(IndexError, self.array.__getitem__, (0, self.ncols+1))

    # 测试 Farray 的 asString 方法
    def testAsString(self):
        "Test Farray asString method"
        result = """\
"""
[ [ 0, 1, 2, 3 ],
  [ 1, 2, 3, 4 ],
  [ 2, 3, 4, 5 ],
  [ 3, 4, 5, 6 ],
  [ 4, 5, 6, 7 ] ]
"""

# 遍历每个元素的行和列索引，为二维数组填充递增的整数值
for i in range(self.nrows):
    for j in range(self.ncols):
        self.array[i, j] = i+j
self.assertTrue(self.array.asString() == result)

def testStr(self):
    "Test Farray __str__ method"
    result = """\
[ [ 0, -1, -2, -3 ],
  [ 1, 0, -1, -2 ],
  [ 2, 1, 0, -1 ],
  [ 3, 2, 1, 0 ],
  [ 4, 3, 2, 1 ] ]
"""

# 遍历每个元素的行和列索引，为二维数组填充递减的整数值
for i in range(self.nrows):
    for j in range(self.ncols):
        self.array[i, j] = i-j
self.assertTrue(str(self.array) == result)

def testView(self):
    "Test Farray view method"
    
# 遍历每个元素的行和列索引，为二维数组填充递增的整数值
for i in range(self.nrows):
    for j in range(self.ncols):
        self.array[i, j] = i+j

# 调用数组的视图方法，并进行断言检查其返回的对象是否是 np.ndarray 类型
a = self.array.view()
self.assertTrue(isinstance(a, np.ndarray))

# 断言检查数组是否是列优先（Fortran）存储
self.assertTrue(a.flags.f_contiguous)

# 遍历每个元素的行和列索引，检查视图数组中的元素值是否正确
for i in range(self.nrows):
    for j in range(self.ncols):
        self.assertTrue(a[i, j] == i+j)

######################################################################

if __name__ == "__main__":

    # 构建测试套件
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FarrayTestCase))

    # 执行测试套件
    print("Testing Classes of Module Farray")
    print("NumPy version", np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))
```