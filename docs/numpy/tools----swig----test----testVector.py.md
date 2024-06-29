# `.\numpy\tools\swig\test\testVector.py`

```py
#!/usr/bin/env python3
# System imports
import sys
import unittest

# Import NumPy
import numpy as np
major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
if major == 0: BadListError = TypeError
else:          BadListError = ValueError

# Import Vector module
import Vector

######################################################################

class VectorTestCase(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"

    # Test the (type IN_ARRAY1[ANY]) typemap
    def testLength(self):
        "Test length function"
        # 输出测试类型和测试开始信息到标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的长度函数并测试
        length = Vector.__dict__[self.typeStr + "Length"]
        self.assertEqual(length([5, 12, 0]), 13)

    # Test the (type IN_ARRAY1[ANY]) typemap
    def testLengthBadList(self):
        "Test length function with bad list"
        # 输出测试类型和测试开始信息到标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的长度函数并测试，预期抛出 BadListError 异常
        length = Vector.__dict__[self.typeStr + "Length"]
        self.assertRaises(BadListError, length, [5, "twelve", 0])

    # Test the (type IN_ARRAY1[ANY]) typemap
    def testLengthWrongSize(self):
        "Test length function with wrong size"
        # 输出测试类型和测试开始信息到标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的长度函数并测试，预期抛出 TypeError 异常
        length = Vector.__dict__[self.typeStr + "Length"]
        self.assertRaises(TypeError, length, [5, 12])

    # Test the (type IN_ARRAY1[ANY]) typemap
    def testLengthWrongDim(self):
        "Test length function with wrong dimensions"
        # 输出测试类型和测试开始信息到标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的长度函数并测试，预期抛出 TypeError 异常
        length = Vector.__dict__[self.typeStr + "Length"]
        self.assertRaises(TypeError, length, [[1, 2], [3, 4]])

    # Test the (type IN_ARRAY1[ANY]) typemap
    def testLengthNonContainer(self):
        "Test length function with non-container"
        # 输出测试类型和测试开始信息到标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的长度函数并测试，预期抛出 TypeError 异常
        length = Vector.__dict__[self.typeStr + "Length"]
        self.assertRaises(TypeError, length, None)

    # Test the (type* IN_ARRAY1, int DIM1) typemap
    def testProd(self):
        "Test prod function"
        # 输出测试类型和测试开始信息到标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的乘积函数并测试
        prod = Vector.__dict__[self.typeStr + "Prod"]
        self.assertEqual(prod([1, 2, 3, 4]), 24)

    # Test the (type* IN_ARRAY1, int DIM1) typemap
    def testProdBadList(self):
        "Test prod function with bad list"
        # 输出测试类型和测试开始信息到标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的乘积函数并测试，预期抛出 BadListError 异常
        prod = Vector.__dict__[self.typeStr + "Prod"]
        self.assertRaises(BadListError, prod, [[1, "two"], ["e", "pi"]])

    # Test the (type* IN_ARRAY1, int DIM1) typemap
    def testProdWrongDim(self):
        "Test prod function with wrong dimensions"
        # 输出测试类型和测试开始信息到标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的乘积函数并测试，预期抛出 TypeError 异常
        prod = Vector.__dict__[self.typeStr + "Prod"]
        self.assertRaises(TypeError, prod, [[1, 2], [8, 9]])

    # Test the (type* IN_ARRAY1, int DIM1) typemap
    # 测试非容器对象的 prod 函数
    def testProdNonContainer(self):
        "Test prod function with non-container"
        # 打印测试类型字符串到标准错误，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的 prod 函数
        prod = Vector.__dict__[self.typeStr + "Prod"]
        # 断言调用 prod 函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, prod, None)

    # 测试 sum 函数，参数为 (int DIM1, type* IN_ARRAY1) typemap
    def testSum(self):
        "Test sum function"
        # 打印测试类型字符串到标准错误，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的 sum 函数
        sum = Vector.__dict__[self.typeStr + "Sum"]
        # 断言 sum 函数对 [5, 6, 7, 8] 的计算结果为 26
        self.assertEqual(sum([5, 6, 7, 8]), 26)

    # 测试 sum 函数，参数为 (int DIM1, type* IN_ARRAY1) typemap，使用错误的列表
    def testSumBadList(self):
        "Test sum function with bad list"
        # 打印测试类型字符串到标准错误，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的 sum 函数
        sum = Vector.__dict__[self.typeStr + "Sum"]
        # 断言调用 sum 函数时会抛出 BadListError 异常，因为列表包含非数值类型元素
        self.assertRaises(BadListError, sum, [3, 4, 5, "pi"])

    # 测试 sum 函数，参数为 (int DIM1, type* IN_ARRAY1) typemap，使用错误的维度
    def testSumWrongDim(self):
        "Test sum function with wrong dimensions"
        # 打印测试类型字符串到标准错误，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的 sum 函数
        sum = Vector.__dict__[self.typeStr + "Sum"]
        # 断言调用 sum 函数时会抛出 TypeError 异常，因为输入的是二维列表而不是一维列表
        self.assertRaises(TypeError, sum, [[3, 4], [5, 6]])

    # 测试 sum 函数，参数为 (int DIM1, type* IN_ARRAY1) typemap，使用非容器对象
    def testSumNonContainer(self):
        "Test sum function with non-container"
        # 打印测试类型字符串到标准错误，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的 sum 函数
        sum = Vector.__dict__[self.typeStr + "Sum"]
        # 断言调用 sum 函数时会抛出 TypeError 异常，因为输入参数是 True，而不是可迭代对象
        self.assertRaises(TypeError, sum, True)

    # 测试 reverse 函数，参数为 (type INPLACE_ARRAY1[ANY]) typemap
    def testReverse(self):
        "Test reverse function"
        # 打印测试类型字符串到标准错误，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的 reverse 函数
        reverse = Vector.__dict__[self.typeStr + "Reverse"]
        # 创建一个包含 [1, 2, 4] 的 NumPy 数组，并进行反转操作
        vector = np.array([1, 2, 4], self.typeCode)
        reverse(vector)
        # 断言 vector 是否等于 [4, 2, 1] 的所有元素，即反转是否成功
        self.assertEqual((vector == [4, 2, 1]).all(), True)

    # 测试 reverse 函数，参数为 (type INPLACE_ARRAY1[ANY]) typemap，使用错误的维度
    def testReverseWrongDim(self):
        "Test reverse function with wrong dimensions"
        # 打印测试类型字符串到标准错误，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的 reverse 函数
        reverse = Vector.__dict__[self.typeStr + "Reverse"]
        # 创建一个二维 NumPy 数组作为参数，断言调用 reverse 函数时会抛出 TypeError 异常
        vector = np.array([[1, 2], [3, 4]], self.typeCode)
        self.assertRaises(TypeError, reverse, vector)

    # 测试 reverse 函数，参数为 (type INPLACE_ARRAY1[ANY]) typemap，使用错误的大小
    def testReverseWrongSize(self):
        "Test reverse function with wrong size"
        # 打印测试类型字符串到标准错误，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的 reverse 函数
        reverse = Vector.__dict__[self.typeStr + "Reverse"]
        # 创建一个包含 [9, 8, 7, 6, 5, 4] 的 NumPy 数组作为参数，断言调用 reverse 函数时会抛出 TypeError 异常
        vector = np.array([9, 8, 7, 6, 5, 4], self.typeCode)
        self.assertRaises(TypeError, reverse, vector)

    # 测试 reverse 函数，参数为 (type INPLACE_ARRAY1[ANY]) typemap，使用错误的类型
    def testReverseWrongType(self):
        "Test reverse function with wrong type"
        # 打印测试类型字符串到标准错误，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取对应类型的 reverse 函数
        reverse = Vector.__dict__[self.typeStr + "Reverse"]
        # 创建一个包含 [1, 2, 4] 的 NumPy 数组，但指定错误的类型 'c'，断言调用 reverse 函数时会抛出 TypeError 异常
        vector = np.array([1, 2, 4], 'c')
        self.assertRaises(TypeError, reverse, vector)
    def testReverseNonArray(self):
        "Test reverse function with non-array"
        # 输出测试类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取指定类型的反转函数
        reverse = Vector.__dict__[self.typeStr + "Reverse"]
        # 断言调用反转函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, reverse, [2, 4, 6])

    # Test the (type* INPLACE_ARRAY1, int DIM1) typemap
    def testOnes(self):
        "Test ones function"
        # 输出测试类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取指定类型的ones函数
        ones = Vector.__dict__[self.typeStr + "Ones"]
        # 创建一个全零的向量数组
        vector = np.zeros(5, self.typeCode)
        # 调用ones函数将向量数组填充为全1
        ones(vector)
        # 使用 NumPy 测试工具断言向量数组与期望的全1数组相等
        np.testing.assert_array_equal(vector, np.array([1, 1, 1, 1, 1]))

    # Test the (type* INPLACE_ARRAY1, int DIM1) typemap
    def testOnesWrongDim(self):
        "Test ones function with wrong dimensions"
        # 输出测试类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取指定类型的ones函数
        ones = Vector.__dict__[self.typeStr + "Ones"]
        # 创建一个维度为(5, 5)的全零向量数组
        vector = np.zeros((5, 5), self.typeCode)
        # 断言调用ones函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, ones, vector)

    # Test the (type* INPLACE_ARRAY1, int DIM1) typemap
    def testOnesWrongType(self):
        "Test ones function with wrong type"
        # 输出测试类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取指定类型的ones函数
        ones = Vector.__dict__[self.typeStr + "Ones"]
        # 创建一个类型为'c'的向量数组
        vector = np.zeros((5, 5), 'c')
        # 断言调用ones函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, ones, vector)

    # Test the (type* INPLACE_ARRAY1, int DIM1) typemap
    def testOnesNonArray(self):
        "Test ones function with non-array"
        # 输出测试类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取指定类型的ones函数
        ones = Vector.__dict__[self.typeStr + "Ones"]
        # 断言调用ones函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, ones, [2, 4, 6, 8])

    # Test the (int DIM1, type* INPLACE_ARRAY1) typemap
    def testZeros(self):
        "Test zeros function"
        # 输出测试类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取指定类型的zeros函数
        zeros = Vector.__dict__[self.typeStr + "Zeros"]
        # 创建一个全1的向量数组
        vector = np.ones(5, self.typeCode)
        # 调用zeros函数将向量数组填充为全0
        zeros(vector)
        # 使用 NumPy 测试工具断言向量数组与期望的全0数组相等
        np.testing.assert_array_equal(vector, np.array([0, 0, 0, 0, 0]))

    # Test the (int DIM1, type* INPLACE_ARRAY1) typemap
    def testZerosWrongDim(self):
        "Test zeros function with wrong dimensions"
        # 输出测试类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取指定类型的zeros函数
        zeros = Vector.__dict__[self.typeStr + "Zeros"]
        # 创建一个维度为(5, 5)的全1向量数组
        vector = np.ones((5, 5), self.typeCode)
        # 断言调用zeros函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, zeros, vector)

    # Test the (int DIM1, type* INPLACE_ARRAY1) typemap
    def testZerosWrongType(self):
        "Test zeros function with wrong type"
        # 输出测试类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取指定类型的zeros函数
        zeros = Vector.__dict__[self.typeStr + "Zeros"]
        # 创建一个类型为'c'的全1向量数组
        vector = np.ones(6, 'c')
        # 断言调用zeros函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, zeros, vector)
    # 定义一个测试方法，测试当参数不是数组时的情况
    def testZerosNonArray(self):
        "Test zeros function with non-array"
        # 打印当前类型字符串，指示测试开始，使用标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Vector 类的对应类型加上 "Zeros" 的方法，并引发 TypeError 异常，期望传入参数为列表
        zeros = Vector.__dict__[self.typeStr + "Zeros"]
        self.assertRaises(TypeError, zeros, [1, 3, 5, 7, 9])

    # 测试 (type ARGOUT_ARRAY1[ANY]) 类型映射的 typemap
    def testEOSplit(self):
        "Test eoSplit function"
        # 打印当前类型字符串，指示测试开始，使用标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Vector 类的对应类型加上 "EOSplit" 的方法，并测试其返回的 even 和 odd 是否符合预期
        eoSplit = Vector.__dict__[self.typeStr + "EOSplit"]
        even, odd = eoSplit([1, 2, 3])
        self.assertEqual((even == [1, 0, 3]).all(), True)
        self.assertEqual((odd  == [0, 2, 0]).all(), True)

    # 测试 (type* ARGOUT_ARRAY1, int DIM1) 类型映射的 typemap
    def testTwos(self):
        "Test twos function"
        # 打印当前类型字符串，指示测试开始，使用标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Vector 类的对应类型加上 "Twos" 的方法，并测试其返回的向量是否为全是 2 的数组
        twos = Vector.__dict__[self.typeStr + "Twos"]
        vector = twos(5)
        self.assertEqual((vector == [2, 2, 2, 2, 2]).all(), True)

    # 测试 (type* ARGOUT_ARRAY1, int DIM1) 类型映射的 typemap，验证传入非整数维度时的行为
    def testTwosNonInt(self):
        "Test twos function with non-integer dimension"
        # 打印当前类型字符串，指示测试开始，使用标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Vector 类的对应类型加上 "Twos" 的方法，并引发 TypeError 异常，期望传入参数为整数
        twos = Vector.__dict__[self.typeStr + "Twos"]
        self.assertRaises(TypeError, twos, 5.0)

    # 测试 (int DIM1, type* ARGOUT_ARRAY1) 类型映射的 typemap
    def testThrees(self):
        "Test threes function"
        # 打印当前类型字符串，指示测试开始，使用标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Vector 类的对应类型加上 "Threes" 的方法，并测试其返回的向量是否为全是 3 的数组
        threes = Vector.__dict__[self.typeStr + "Threes"]
        vector = threes(6)
        self.assertEqual((vector == [3, 3, 3, 3, 3, 3]).all(), True)

    # 测试 (type* ARGOUT_ARRAY1, int DIM1) 类型映射的 typemap，验证传入非整数维度时的行为
    def testThreesNonInt(self):
        "Test threes function with non-integer dimension"
        # 打印当前类型字符串，指示测试开始，使用标准错误输出
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Vector 类的对应类型加上 "Threes" 的方法，并引发 TypeError 异常，期望传入参数为整数
        threes = Vector.__dict__[self.typeStr + "Threes"]
        self.assertRaises(TypeError, threes, "threes")
######################################################################

class scharTestCase(VectorTestCase):
    # scharTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'schar'
        self.typeStr  = "schar"
        # 设置类型代码为 'b'
        self.typeCode = "b"

######################################################################

class ucharTestCase(VectorTestCase):
    # ucharTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'uchar'
        self.typeStr  = "uchar"
        # 设置类型代码为 'B'
        self.typeCode = "B"

######################################################################

class shortTestCase(VectorTestCase):
    # shortTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'short'
        self.typeStr  = "short"
        # 设置类型代码为 'h'
        self.typeCode = "h"

######################################################################

class ushortTestCase(VectorTestCase):
    # ushortTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'ushort'
        self.typeStr  = "ushort"
        # 设置类型代码为 'H'
        self.typeCode = "H"

######################################################################

class intTestCase(VectorTestCase):
    # intTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'int'
        self.typeStr  = "int"
        # 设置类型代码为 'i'
        self.typeCode = "i"

######################################################################

class uintTestCase(VectorTestCase):
    # uintTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'uint'
        self.typeStr  = "uint"
        # 设置类型代码为 'I'
        self.typeCode = "I"

######################################################################

class longTestCase(VectorTestCase):
    # longTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'long'
        self.typeStr  = "long"
        # 设置类型代码为 'l'
        self.typeCode = "l"

######################################################################

class ulongTestCase(VectorTestCase):
    # ulongTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'ulong'
        self.typeStr  = "ulong"
        # 设置类型代码为 'L'
        self.typeCode = "L"

######################################################################

class longLongTestCase(VectorTestCase):
    # longLongTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'longLong'
        self.typeStr  = "longLong"
        # 设置类型代码为 'q'
        self.typeCode = "q"

######################################################################

class ulongLongTestCase(VectorTestCase):
    # ulongLongTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'ulongLong'
        self.typeStr  = "ulongLong"
        # 设置类型代码为 'Q'
        self.typeCode = "Q"

######################################################################

class floatTestCase(VectorTestCase):
    # floatTestCase 类继承自 VectorTestCase 类
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的初始化方法
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 'float'
        self.typeStr  = "float"
        # 设置类型代码为 'f'
        self.typeCode = "f"

######################################################################
######################################################################

class doubleTestCase(VectorTestCase):
    # 定义 doubleTestCase 类，继承自 VectorTestCase
    def __init__(self, methodName="runTest"):
        # 调用父类 VectorTestCase 的构造函数初始化
        VectorTestCase.__init__(self, methodName)
        # 设置类型字符串为 "double"
        self.typeStr  = "double"
        # 设置类型代码为 "d"
        self.typeCode = "d"

######################################################################

if __name__ == "__main__":
    # 如果作为主程序运行

    # 创建一个测试套件
    suite = unittest.TestSuite()
    # 将各个类型的测试用例添加到测试套件中
    suite.addTest(unittest.makeSuite(    scharTestCase))
    suite.addTest(unittest.makeSuite(    ucharTestCase))
    suite.addTest(unittest.makeSuite(    shortTestCase))
    suite.addTest(unittest.makeSuite(   ushortTestCase))
    suite.addTest(unittest.makeSuite(      intTestCase))
    suite.addTest(unittest.makeSuite(     uintTestCase))
    suite.addTest(unittest.makeSuite(     longTestCase))
    suite.addTest(unittest.makeSuite(    ulongTestCase))
    suite.addTest(unittest.makeSuite( longLongTestCase))
    suite.addTest(unittest.makeSuite(ulongLongTestCase))
    suite.addTest(unittest.makeSuite(    floatTestCase))
    suite.addTest(unittest.makeSuite(   doubleTestCase))

    # 执行测试套件中的测试用例
    print("Testing 1D Functions of Module Vector")
    # 打印 NumPy 的版本号
    print("NumPy version", np.__version__)
    print()
    # 运行测试并获取测试结果
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # 根据测试结果，如果有错误或失败，以非零状态退出，否则以零状态退出
    sys.exit(bool(result.errors + result.failures))
```