# `.\numpy\tools\swig\test\testArray.py`

```py
#!/usr/bin/env python3
# System imports
import sys  # 导入系统模块sys，用于访问系统相关功能
import unittest  # 导入unittest模块，用于编写和运行单元测试

# Import NumPy
import numpy as np  # 导入NumPy库，用于处理数值数组和数据

major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
if major == 0:
    BadListError = TypeError  # 如果NumPy的主版本号为0，定义BadListError为TypeError
else:
    BadListError = ValueError  # 否则定义BadListError为ValueError

import Array  # 导入自定义的Array模块

######################################################################

class Array1TestCase(unittest.TestCase):
    """定义Array1类的单元测试用例"""

    def setUp(self):
        """每个测试方法执行前的初始化操作"""
        self.length = 5  # 设置数组长度为5
        self.array1 = Array.Array1(self.length)  # 创建Array1对象

    def testConstructor0(self):
        """测试Array1的默认构造函数"""
        a = Array.Array1()  # 调用默认构造函数创建Array1对象a
        self.assertTrue(isinstance(a, Array.Array1))  # 断言a是Array1的实例
        self.assertTrue(len(a) == 0)  # 断言a的长度为0

    def testConstructor1(self):
        """测试Array1的长度构造函数"""
        self.assertTrue(isinstance(self.array1, Array.Array1))  # 断言self.array1是Array1的实例

    def testConstructor2(self):
        """测试Array1的数组构造函数"""
        na = np.arange(self.length)  # 创建一个NumPy数组na，包含从0到self.length-1的整数
        aa = Array.Array1(na)  # 使用数组na创建Array1对象aa
        self.assertTrue(isinstance(aa, Array.Array1))  # 断言aa是Array1的实例

    def testConstructor3(self):
        """测试Array1的拷贝构造函数"""
        for i in range(self.array1.length()): self.array1[i] = i  # 设置self.array1的元素为索引值
        arrayCopy = Array.Array1(self.array1)  # 使用self.array1创建Array1对象arrayCopy
        self.assertTrue(arrayCopy == self.array1)  # 断言arrayCopy等于self.array1

    def testConstructorBad(self):
        """测试Array1的长度构造函数，负值情况"""
        self.assertRaises(ValueError, Array.Array1, -4)  # 断言创建Array1对象时传入负数会引发ValueError异常

    def testLength(self):
        """测试Array1的length方法"""
        self.assertTrue(self.array1.length() == self.length)  # 断言self.array1的长度为预期长度

    def testLen(self):
        """测试Array1的__len__方法"""
        self.assertTrue(len(self.array1) == self.length)  # 断言调用len函数返回self.array1的长度

    def testResize0(self):
        """测试Array1的resize方法，长度参数"""
        newLen = 2 * self.length  # 设置新的长度为当前长度的两倍
        self.array1.resize(newLen)  # 调整self.array1的长度为newLen
        self.assertTrue(len(self.array1) == newLen)  # 断言self.array1的长度为newLen

    def testResize1(self):
        """测试Array1的resize方法，数组参数"""
        a = np.zeros((2*self.length,), dtype='l')  # 创建一个NumPy数组a，包含长度为2*self.length的零数组
        self.array1.resize(a)  # 使用数组a调整self.array1的长度
        self.assertTrue(len(self.array1) == a.size)  # 断言self.array1的长度等于数组a的大小

    def testResizeBad(self):
        """测试Array1的resize方法，负值长度"""
        self.assertRaises(ValueError, self.array1.resize, -5)  # 断言调用resize方法时传入负数会引发ValueError异常

    def testSetGet(self):
        """测试Array1的__setitem__和__getitem__方法"""
        n = self.length  # 设置n为数组的长度
        for i in range(n):
            self.array1[i] = i*i  # 设置索引i处的元素为i的平方
        for i in range(n):
            self.assertTrue(self.array1[i] == i*i)  # 断言索引i处的元素等于i的平方

    def testSetBad1(self):
        """测试Array1的__setitem__方法，负索引"""
        self.assertRaises(IndexError, self.array1.__setitem__, -1, 0)  # 断言设置负索引会引发IndexError异常

    def testSetBad2(self):
        """测试Array1的__setitem__方法，超出范围的索引"""
        self.assertRaises(IndexError, self.array1.__setitem__, self.length+1, 0)  # 断言设置超出范围的索引会引发IndexError异常

    def testGetBad1(self):
        """测试Array1的__getitem__方法，负索引"""
        self.assertRaises(IndexError, self.array1.__getitem__, -1)  # 断言访问负索引会引发IndexError异常
    # 测试 Array1 的 __getitem__ 方法，测试索引超出范围的情况
    def testGetBad2(self):
        "Test Array1 __getitem__ method, out-of-range index"
        # 断言是否抛出 IndexError 异常，访问超出范围的索引 self.length+1
        self.assertRaises(IndexError, self.array1.__getitem__, self.length+1)

    # 测试 Array1 的 asString 方法
    def testAsString(self):
        "Test Array1 asString method"
        # 设置 Array1 的元素值为索引加一
        for i in range(self.array1.length()): self.array1[i] = i+1
        # 断言 Array1 转换为字符串后是否等于预期的字符串 "[ 1, 2, 3, 4, 5 ]"
        self.assertTrue(self.array1.asString() == "[ 1, 2, 3, 4, 5 ]")

    # 测试 Array1 的 __str__ 方法
    def testStr(self):
        "Test Array1 __str__ method"
        # 设置 Array1 的元素值为索引减二
        for i in range(self.array1.length()): self.array1[i] = i-2
        # 断言 Array1 转换为字符串后是否等于预期的字符串 "[ -2, -1, 0, 1, 2 ]"
        self.assertTrue(str(self.array1) == "[ -2, -1, 0, 1, 2 ]")

    # 测试 Array1 的 view 方法
    def testView(self):
        "Test Array1 view method"
        # 设置 Array1 的元素值为索引加一
        for i in range(self.array1.length()): self.array1[i] = i+1
        # 调用 Array1 的 view 方法
        a = self.array1.view()
        # 断言返回的视图 a 是否为 numpy.ndarray 类型
        self.assertTrue(isinstance(a, np.ndarray))
        # 断言视图 a 的长度是否与 Array1 的长度相等
        self.assertTrue(len(a) == self.length)
        # 断言视图 a 的所有元素是否与预期的数组 [1, 2, 3, 4, 5] 相等
        self.assertTrue((a == [1, 2, 3, 4, 5]).all())
######################################################################

class Array2TestCase(unittest.TestCase):

    def setUp(self):
        # 设置测试用例中使用的行数和列数
        self.nrows = 5
        self.ncols = 4
        # 创建一个 Array2 类的实例对象
        self.array2 = Array.Array2(self.nrows, self.ncols)

    def testConstructor0(self):
        "Test Array2 default constructor"
        # 测试默认构造函数是否返回了 Array2 类的实例
        a = Array.Array2()
        self.assertTrue(isinstance(a, Array.Array2))
        # 测试默认构造函数返回的对象是否长度为 0
        self.assertTrue(len(a) == 0)

    def testConstructor1(self):
        "Test Array2 nrows, ncols constructor"
        # 测试指定行数和列数的构造函数是否返回了 Array2 类的实例
        self.assertTrue(isinstance(self.array2, Array.Array2))

    def testConstructor2(self):
        "Test Array2 array constructor"
        # 使用给定的 numpy 数组创建 Array2 类的实例
        na = np.zeros((3, 4), dtype="l")
        aa = Array.Array2(na)
        self.assertTrue(isinstance(aa, Array.Array2))

    def testConstructor3(self):
        "Test Array2 copy constructor"
        # 测试复制构造函数是否能够正确复制 Array2 类的实例
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array2[i][j] = i * j
        # 使用复制构造函数创建一个新的 Array2 实例
        arrayCopy = Array.Array2(self.array2)
        self.assertTrue(arrayCopy == self.array2)

    def testConstructorBad1(self):
        "Test Array2 nrows, ncols constructor, negative nrows"
        # 测试当指定的行数为负数时是否会引发 ValueError 异常
        self.assertRaises(ValueError, Array.Array2, -4, 4)

    def testConstructorBad2(self):
        "Test Array2 nrows, ncols constructor, negative ncols"
        # 测试当指定的列数为负数时是否会引发 ValueError 异常
        self.assertRaises(ValueError, Array.Array2, 4, -4)

    def testNrows(self):
        "Test Array2 nrows method"
        # 测试 nrows 方法是否返回正确的行数
        self.assertTrue(self.array2.nrows() == self.nrows)

    def testNcols(self):
        "Test Array2 ncols method"
        # 测试 ncols 方法是否返回正确的列数
        self.assertTrue(self.array2.ncols() == self.ncols)

    def testLen(self):
        "Test Array2 __len__ method"
        # 测试 __len__ 方法是否返回正确的数组长度
        self.assertTrue(len(self.array2) == self.nrows*self.ncols)

    def testResize0(self):
        "Test Array2 resize method, size"
        # 测试 resize 方法是否能够正确调整数组大小（指定行数和列数）
        newRows = 2 * self.nrows
        newCols = 2 * self.ncols
        self.array2.resize(newRows, newCols)
        self.assertTrue(len(self.array2) == newRows * newCols)

    def testResize1(self):
        "Test Array2 resize method, array"
        # 测试 resize 方法是否能够正确调整数组大小（使用给定的 numpy 数组）
        a = np.zeros((2*self.nrows, 2*self.ncols), dtype='l')
        self.array2.resize(a)
        self.assertTrue(len(self.array2) == a.size)

    def testResizeBad1(self):
        "Test Array2 resize method, negative nrows"
        # 测试当调整行数为负数时，是否会引发 ValueError 异常
        self.assertRaises(ValueError, self.array2.resize, -5, 5)

    def testResizeBad2(self):
        "Test Array2 resize method, negative ncols"
        # 测试当调整列数为负数时，是否会引发 ValueError 异常
        self.assertRaises(ValueError, self.array2.resize, 5, -5)

    def testSetGet1(self):
        "Test Array2 __setitem__, __getitem__ methods"
        m = self.nrows
        n = self.ncols
        array1 = []
        a = np.arange(n, dtype="l")
        for i in range(m):
            # 创建一个 Array1 类的实例列表
            array1.append(Array.Array1(i * a))
        for i in range(m):
            # 测试 __setitem__ 方法是否能够正确设置元素
            self.array2[i] = array1[i]
        for i in range(m):
            # 测试 __getitem__ 方法是否能够正确获取元素
            self.assertTrue(self.array2[i] == array1[i])
    # 测试 Array2 类的链式 __setitem__ 和 __getitem__ 方法
    def testSetGet2(self):
        "Test Array2 chained __setitem__, __getitem__ methods"
        m = self.nrows  # 获取行数
        n = self.ncols  # 获取列数
        for i in range(m):  # 遍历行
            for j in range(n):  # 遍历列
                self.array2[i][j] = i * j  # 设置数组中索引 (i, j) 处的值为 i * j
        for i in range(m):  # 再次遍历行
            for j in range(n):  # 再次遍历列
                self.assertTrue(self.array2[i][j] == i * j)  # 断言数组中索引 (i, j) 处的值等于 i * j

    # 测试 Array2 类的 __setitem__ 方法，使用负索引
    def testSetBad1(self):
        "Test Array2 __setitem__ method, negative index"
        a = Array.Array1(self.ncols)  # 创建一个 Array1 类型的对象 a
        self.assertRaises(IndexError, self.array2.__setitem__, -1, a)  # 断言调用 __setitem__ 方法时使用负索引会抛出 IndexError 异常

    # 测试 Array2 类的 __setitem__ 方法，使用超出范围的索引
    def testSetBad2(self):
        "Test Array2 __setitem__ method, out-of-range index"
        a = Array.Array1(self.ncols)  # 创建一个 Array1 类型的对象 a
        self.assertRaises(IndexError, self.array2.__setitem__, self.nrows + 1, a)  # 断言调用 __setitem__ 方法时使用超出范围的索引会抛出 IndexError 异常

    # 测试 Array2 类的 __getitem__ 方法，使用负索引
    def testGetBad1(self):
        "Test Array2 __getitem__ method, negative index"
        self.assertRaises(IndexError, self.array2.__getitem__, -1)  # 断言调用 __getitem__ 方法时使用负索引会抛出 IndexError 异常

    # 测试 Array2 类的 __getitem__ 方法，使用超出范围的索引
    def testGetBad2(self):
        "Test Array2 __getitem__ method, out-of-range index"
        self.assertRaises(IndexError, self.array2.__getitem__, self.nrows + 1)  # 断言调用 __getitem__ 方法时使用超出范围的索引会抛出 IndexError 异常

    # 测试 Array2 类的 asString 方法
    def testAsString(self):
        "Test Array2 asString method"
        result = """\
# 定义一个二维列表，表示一个包含整数的二维数组
[ [ 0, 1, 2, 3 ],
  [ 1, 2, 3, 4 ],
  [ 2, 3, 4, 5 ],
  [ 3, 4, 5, 6 ],
  [ 4, 5, 6, 7 ] ]



        # 遍历二维数组的行
        for i in range(self.nrows):
            # 遍历二维数组的列
            for j in range(self.ncols):
                # 为每个元素赋值为行索引和列索引的和
                self.array2[i][j] = i+j
        # 使用断言验证数组转换为字符串后是否与指定的结果字符串相等
        self.assertTrue(self.array2.asString() == result)



    def testStr(self):
        "Test Array2 __str__ method"
        # 定义预期的结果字符串，表示包含整数的二维数组
        result = """\
[ [ 0, -1, -2, -3 ],
  [ 1, 0, -1, -2 ],
  [ 2, 1, 0, -1 ],
  [ 3, 2, 1, 0 ],
  [ 4, 3, 2, 1 ] ]
"""
        # 遍历二维数组的行
        for i in range(self.nrows):
            # 遍历二维数组的列
            for j in range(self.ncols):
                # 为每个元素赋值为行索引减去列索引
                self.array2[i][j] = i-j
        # 使用断言验证数组转换为字符串后是否与指定的结果字符串相等
        self.assertTrue(str(self.array2) == result)



    def testView(self):
        "Test Array2 view method"
        # 调用数组的视图方法
        a = self.array2.view()
        # 使用断言验证返回的视图对象是否是 NumPy 数组
        self.assertTrue(isinstance(a, np.ndarray))
        # 使用断言验证返回的视图数组的长度是否与指定的行数相等
        self.assertTrue(len(a) == self.nrows)



######################################################################

class ArrayZTestCase(unittest.TestCase):

    def setUp(self):
        # 初始化测试用例中数组的长度
        self.length = 5
        # 创建一个长度为 self.length 的 ArrayZ 对象
        self.array3 = Array.ArrayZ(self.length)


（以下测试函数可以根据需要进行类似的注释，但已超出了示例的范围，这里仅展示示例代码部分的注释。）
    def testSetBad1(self):
        "Test ArrayZ __setitem__ method, negative index"
        # 断言会抛出 IndexError 异常，因为索引为负数
        self.assertRaises(IndexError, self.array3.__setitem__, -1, 0)

    def testSetBad2(self):
        "Test ArrayZ __setitem__ method, out-of-range index"
        # 断言会抛出 IndexError 异常，因为索引超出范围
        self.assertRaises(IndexError, self.array3.__setitem__, self.length+1, 0)

    def testGetBad1(self):
        "Test ArrayZ __getitem__ method, negative index"
        # 断言会抛出 IndexError 异常，因为索引为负数
        self.assertRaises(IndexError, self.array3.__getitem__, -1)

    def testGetBad2(self):
        "Test ArrayZ __getitem__ method, out-of-range index"
        # 断言会抛出 IndexError 异常，因为索引超出范围
        self.assertRaises(IndexError, self.array3.__getitem__, self.length+1)

    def testAsString(self):
        "Test ArrayZ asString method"
        # 为 ArrayZ 对象的元素赋值为复数
        for i in range(self.array3.length()): self.array3[i] = complex(i+1,-i-1)
        # 断言 ArrayZ 对象的字符串表示与预期的字符串相等
        self.assertTrue(self.array3.asString() == "[ (1,-1), (2,-2), (3,-3), (4,-4), (5,-5) ]")

    def testStr(self):
        "Test ArrayZ __str__ method"
        # 为 ArrayZ 对象的元素赋值为复数
        for i in range(self.array3.length()): self.array3[i] = complex(i-2,(i-2)*2)
        # 断言 ArrayZ 对象的字符串表示与预期的字符串相等
        self.assertTrue(str(self.array3) == "[ (-2,-4), (-1,-2), (0,0), (1,2), (2,4) ]")

    def testView(self):
        "Test ArrayZ view method"
        # 为 ArrayZ 对象的元素赋值为复数
        for i in range(self.array3.length()): self.array3[i] = complex(i+1,i+2)
        # 获取 ArrayZ 对象的视图
        a = self.array3.view()
        # 断言视图对象是 numpy.ndarray 类型
        self.assertTrue(isinstance(a, np.ndarray))
        # 断言视图对象的长度与 ArrayZ 对象的长度相等
        self.assertTrue(len(a) == self.length)
        # 断言视图对象的所有元素与预期的复数数组相等
        self.assertTrue((a == [1+2j, 2+3j, 3+4j, 4+5j, 5+6j]).all())
######################################################################

if __name__ == "__main__":
    # 如果当前脚本被直接执行而非被导入，则执行以下代码块

    # 构建测试套件
    suite = unittest.TestSuite()
    # 将 Array1TestCase 的测试添加到测试套件中
    suite.addTest(unittest.makeSuite(Array1TestCase))
    # 将 Array2TestCase 的测试添加到测试套件中
    suite.addTest(unittest.makeSuite(Array2TestCase))
    # 将 ArrayZTestCase 的测试添加到测试套件中
    suite.addTest(unittest.makeSuite(ArrayZTestCase))

    # 执行测试套件
    print("Testing Classes of Module Array")
    # 打印 NumPy 版本信息
    print("NumPy version", np.__version__)
    print()
    # 运行测试套件，并返回测试结果
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # 根据测试结果中是否存在错误或失败，决定退出状态
    sys.exit(bool(result.errors + result.failures))
```