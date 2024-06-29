# `.\numpy\tools\swig\test\testMatrix.py`

```py
# 指定 Python 解释器路径，使脚本可以在环境中独立运行
#!/usr/bin/env python3

# 导入系统相关模块
import sys
import unittest

# 导入 NumPy 库，并获取其主版本号和次版本号
import numpy as np
major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
if major == 0: BadListError = TypeError
else:          BadListError = ValueError

# 导入 Matrix 模块
import Matrix

######################################################################

# 定义测试类 MatrixTestCase，继承自 unittest.TestCase
class MatrixTestCase(unittest.TestCase):

    # 构造方法，初始化测试实例
    def __init__(self, methodName="runTests"):
        # 调用父类构造方法初始化
        unittest.TestCase.__init__(self, methodName)
        # 设置测试用例类型字符串和类型代码
        self.typeStr  = "double"
        self.typeCode = "d"

    # 测试用例，测试 det 函数的正常情况
    # Test (type IN_ARRAY2[ANY][ANY]) typemap
    def testDet(self):
        "Test det function"
        # 输出类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 模块中特定类型的 det 函数
        det = Matrix.__dict__[self.typeStr + "Det"]
        # 定义测试矩阵
        matrix = [[8, 7], [6, 9]]
        # 断言调用 det 函数后的返回值为预期值 30
        self.assertEqual(det(matrix), 30)

    # 测试用例，测试 det 函数处理不良列表的情况
    # Test (type IN_ARRAY2[ANY][ANY]) typemap
    def testDetBadList(self):
        "Test det function with bad list"
        # 输出类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 模块中特定类型的 det 函数
        det = Matrix.__dict__[self.typeStr + "Det"]
        # 定义包含非法类型的测试矩阵
        matrix = [[8, 7], ["e", "pi"]]
        # 断言调用 det 函数时会引发 BadListError 异常
        self.assertRaises(BadListError, det, matrix)

    # 测试用例，测试 det 函数处理维度错误的情况
    # Test (type IN_ARRAY2[ANY][ANY]) typemap
    def testDetWrongDim(self):
        "Test det function with wrong dimensions"
        # 输出类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 模块中特定类型的 det 函数
        det = Matrix.__dict__[self.typeStr + "Det"]
        # 定义维度不符合要求的测试矩阵
        matrix = [8, 7]
        # 断言调用 det 函数时会引发 TypeError 异常
        self.assertRaises(TypeError, det, matrix)

    # 测试用例，测试 det 函数处理尺寸错误的情况
    # Test (type IN_ARRAY2[ANY][ANY]) typemap
    def testDetWrongSize(self):
        "Test det function with wrong size"
        # 输出类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 模块中特定类型的 det 函数
        det = Matrix.__dict__[self.typeStr + "Det"]
        # 定义尺寸不符合要求的测试矩阵
        matrix = [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
        # 断言调用 det 函数时会引发 TypeError 异常
        self.assertRaises(TypeError, det, matrix)

    # 测试用例，测试 det 函数处理非容器的情况
    # Test (type IN_ARRAY2[ANY][ANY]) typemap
    def testDetNonContainer(self):
        "Test det function with non-container"
        # 输出类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 模块中特定类型的 det 函数
        det = Matrix.__dict__[self.typeStr + "Det"]
        # 断言调用 det 函数时会引发 TypeError 异常，因为传入了非容器对象
        self.assertRaises(TypeError, det, None)

    # 测试用例，测试 max 函数的正常情况
    # Test (type* IN_ARRAY2, int DIM1, int DIM2) typemap
    def testMax(self):
        "Test max function"
        # 输出类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 模块中特定类型的 max 函数
        max = Matrix.__dict__[self.typeStr + "Max"]
        # 定义测试矩阵
        matrix = [[6, 5, 4], [3, 2, 1]]
        # 断言调用 max 函数后的返回值为预期值 6
        self.assertEqual(max(matrix), 6)

    # 测试用例，测试 max 函数处理不良列表的情况
    # Test (type* IN_ARRAY2, int DIM1, int DIM2) typemap
    def testMaxBadList(self):
        "Test max function with bad list"
        # 输出类型字符串到标准错误流
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 模块中特定类型的 max 函数
        max = Matrix.__dict__[self.typeStr + "Max"]
        # 定义包含非法类型的测试矩阵
        matrix = [[6, "five", 4], ["three", 2, "one"]]
        # 断言调用 max 函数时会引发 BadListError 异常
        self.assertRaises(BadListError, max, matrix)

    # 其他测试用例待补充
    # 定义一个测试函数，用于测试在非容器对象上使用 max 函数
    def testMaxNonContainer(self):
        # 打印当前测试类型字符串，指示正在进行的测试，输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 类中与当前类型字符串相关的最大值函数
        max = Matrix.__dict__[self.typeStr + "Max"]
        # 断言调用该最大值函数时传入 None 会引发 TypeError 异常
        self.assertRaises(TypeError, max, None)

    # 测试函数签名为 (type* IN_ARRAY2, int DIM1, int DIM2)，验证 max 函数对错误维度输入的处理
    def testMaxWrongDim(self):
        # 打印当前测试类型字符串，指示正在进行的测试，输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 类中与当前类型字符串相关的最大值函数
        max = Matrix.__dict__[self.typeStr + "Max"]
        # 断言调用该最大值函数时传入维度错误的列表会引发 TypeError 异常
        self.assertRaises(TypeError, max, [0, 1, 2, 3])

    # 测试函数签名为 (int DIM1, int DIM2, type* IN_ARRAY2)，验证 min 函数的正常工作
    def testMin(self):
        # 打印当前测试类型字符串，指示正在进行的测试，输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 类中与当前类型字符串相关的最小值函数
        min = Matrix.__dict__[self.typeStr + "Min"]
        # 创建一个二维列表作为矩阵输入，并验证其最小值为 4
        matrix = [[9, 8], [7, 6], [5, 4]]
        self.assertEqual(min(matrix), 4)

    # 测试函数签名为 (int DIM1, int DIM2, type* IN_ARRAY2)，验证 min 函数对包含非数值元素的列表的处理
    def testMinBadList(self):
        # 打印当前测试类型字符串，指示正在进行的测试，输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 类中与当前类型字符串相关的最小值函数
        min = Matrix.__dict__[self.typeStr + "Min"]
        # 创建一个包含非数值元素的二维列表，并验证调用最小值函数时会引发 BadListError 异常
        matrix = [["nine", "eight"], ["seven", "six"]]
        self.assertRaises(BadListError, min, matrix)

    # 测试函数签名为 (int DIM1, int DIM2, type* IN_ARRAY2)，验证 min 函数对错误维度输入的处理
    def testMinWrongDim(self):
        # 打印当前测试类型字符串，指示正在进行的测试，输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 类中与当前类型字符串相关的最小值函数
        min = Matrix.__dict__[self.typeStr + "Min"]
        # 断言调用该最小值函数时传入维度错误的列表会引发 TypeError 异常
        self.assertRaises(TypeError, min, [1, 3, 5, 7, 9])

    # 测试函数签名为 (int DIM1, int DIM2, type* IN_ARRAY2)，验证 min 函数在非容器对象上的处理
    def testMinNonContainer(self):
        # 打印当前测试类型字符串，指示正在进行的测试，输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 类中与当前类型字符串相关的最小值函数
        min = Matrix.__dict__[self.typeStr + "Min"]
        # 断言调用该最小值函数时传入 False 会引发 TypeError 异常
        self.assertRaises(TypeError, min, False)

    # 测试函数签名为 (type INPLACE_ARRAY2[ANY][ANY])，验证 scale 函数的正常工作
    def testScale(self):
        # 打印当前测试类型字符串，指示正在进行的测试，输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 类中与当前类型字符串相关的缩放函数
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        # 创建一个 NumPy 数组作为矩阵输入，并验证缩放后的结果是否符合预期
        matrix = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]], self.typeCode)
        scale(matrix, 4)
        # 断言缩放后矩阵的元素是否全部符合预期值
        self.assertEqual((matrix == [[4, 8, 12], [8, 4, 8], [12, 8, 4]]).all(), True)

    # 测试函数签名为 (type INPLACE_ARRAY2[ANY][ANY])，验证 scale 函数对错误维度输入的处理
    def testScaleWrongDim(self):
        # 打印当前测试类型字符串，指示正在进行的测试，输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 类中与当前类型字符串相关的缩放函数
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        # 创建一个维度错误的 NumPy 数组，并断言调用缩放函数时会引发 TypeError 异常
        matrix = np.array([1, 2, 2, 1], self.typeCode)
        self.assertRaises(TypeError, scale, matrix)

    # 测试函数签名为 (type INPLACE_ARRAY2[ANY][ANY])，验证 scale 函数对错误大小输入的处理
    def testScaleWrongSize(self):
        # 打印当前测试类型字符串，指示正在进行的测试，输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Matrix 类中与当前类型字符串相关的缩放函数
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        # 创建一个大小错误的 NumPy 数组，并断言调用缩放函数时会引发 TypeError 异常
        matrix = np.array([[1, 2], [2, 1]], self.typeCode)
        self.assertRaises(TypeError, scale, matrix)
    # Test (type INPLACE_ARRAY2[ANY][ANY]) typemap
    def testScaleWrongType(self):
        "Test scale function with wrong type"
        # 打印当前类型信息到标准错误输出流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取与当前类型相关的缩放函数
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        # 创建一个 NumPy 数组作为测试矩阵，使用 'c' 类型码
        matrix = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]], 'c')
        # 断言调用缩放函数时会引发 TypeError 异常
        self.assertRaises(TypeError, scale, matrix)
    
    # Test (type INPLACE_ARRAY2[ANY][ANY]) typemap
    def testScaleNonArray(self):
        "Test scale function with non-array"
        # 打印当前类型信息到标准错误输出流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取与当前类型相关的缩放函数
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        # 创建一个普通的 Python 列表作为测试矩阵
        matrix = [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
        # 断言调用缩放函数时会引发 TypeError 异常
        self.assertRaises(TypeError, scale, matrix)
    
    # Test (type* INPLACE_ARRAY2, int DIM1, int DIM2) typemap
    def testFloor(self):
        "Test floor function"
        # 打印当前类型信息到标准错误输出流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取与当前类型相关的取底函数
        floor = Matrix.__dict__[self.typeStr + "Floor"]
        # 创建一个 NumPy 数组作为测试矩阵，使用 self.typeCode 类型码
        matrix = np.array([[6, 7], [8, 9]], self.typeCode)
        # 调用取底函数，并验证矩阵的预期变化
        floor(matrix, 7)
        np.testing.assert_array_equal(matrix, np.array([[7, 7], [8, 9]]))
    
    # Test (type* INPLACE_ARRAY2, int DIM1, int DIM2) typemap
    def testFloorWrongDim(self):
        "Test floor function with wrong dimensions"
        # 打印当前类型信息到标准错误输出流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取与当前类型相关的取底函数
        floor = Matrix.__dict__[self.typeStr + "Floor"]
        # 创建一个一维 NumPy 数组作为测试矩阵，使用 self.typeCode 类型码
        matrix = np.array([6, 7, 8, 9], self.typeCode)
        # 断言调用取底函数时会引发 TypeError 异常
        self.assertRaises(TypeError, floor, matrix)
    
    # Test (type* INPLACE_ARRAY2, int DIM1, int DIM2) typemap
    def testFloorWrongType(self):
        "Test floor function with wrong type"
        # 打印当前类型信息到标准错误输出流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取与当前类型相关的取底函数
        floor = Matrix.__dict__[self.typeStr + "Floor"]
        # 创建一个 NumPy 数组作为测试矩阵，使用 'c' 类型码
        matrix = np.array([[6, 7], [8, 9]], 'c')
        # 断言调用取底函数时会引发 TypeError 异常
        self.assertRaises(TypeError, floor, matrix)
    
    # Test (type* INPLACE_ARRAY2, int DIM1, int DIM2) typemap
    def testFloorNonArray(self):
        "Test floor function with non-array"
        # 打印当前类型信息到标准错误输出流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取与当前类型相关的取底函数
        floor = Matrix.__dict__[self.typeStr + "Floor"]
        # 创建一个普通的 Python 列表作为测试矩阵
        matrix = [[6, 7], [8, 9]]
        # 断言调用取底函数时会引发 TypeError 异常
        self.assertRaises(TypeError, floor, matrix)
    
    # Test (int DIM1, int DIM2, type* INPLACE_ARRAY2) typemap
    def testCeil(self):
        "Test ceil function"
        # 打印当前类型信息到标准错误输出流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取与当前类型相关的取上函数
        ceil = Matrix.__dict__[self.typeStr + "Ceil"]
        # 创建一个 NumPy 数组作为测试矩阵，使用 self.typeCode 类型码
        matrix = np.array([[1, 2], [3, 4]], self.typeCode)
        # 调用取上函数，并验证矩阵的预期变化
        ceil(matrix, 3)
        np.testing.assert_array_equal(matrix, np.array([[1, 2], [3, 3]]))
    
    # Test (int DIM1, int DIM2, type* INPLACE_ARRAY2) typemap
    def testCeilWrongDim(self):
        "Test ceil function with wrong dimensions"
        # 打印当前类型信息到标准错误输出流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取与当前类型相关的取上函数
        ceil = Matrix.__dict__[self.typeStr + "Ceil"]
        # 创建一个一维 NumPy 数组作为测试矩阵，使用 self.typeCode 类型码
        matrix = np.array([1, 2, 3, 4], self.typeCode)
        # 断言调用取上函数时会引发 TypeError 异常
        self.assertRaises(TypeError, ceil, matrix)
    # Test (int DIM1, int DIM2, type* INPLACE_ARRAY2) typemap
    def testCeilWrongType(self):
        # 测试 ceil 函数对于维度错误的矩阵
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 ceil 函数
        ceil = Matrix.__dict__[self.typeStr + "Ceil"]
        # 创建一个以字符串 'c' 表示的数组矩阵
        matrix = np.array([[1, 2], [3, 4]], 'c')
        # 断言会抛出 TypeError 异常
        self.assertRaises(TypeError, ceil, matrix)
    
    # Test (int DIM1, int DIM2, type* INPLACE_ARRAY2) typemap
    def testCeilNonArray(self):
        # 测试 ceil 函数对于非数组输入的情况
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 ceil 函数
        ceil = Matrix.__dict__[self.typeStr + "Ceil"]
        # 创建一个普通的嵌套列表表示的矩阵
        matrix = [[1, 2], [3, 4]]
        # 断言会抛出 TypeError 异常
        self.assertRaises(TypeError, ceil, matrix)
    
    # Test (type ARGOUT_ARRAY2[ANY][ANY]) typemap
    def testLUSplit(self):
        # 测试 luSplit 函数
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 luSplit 函数
        luSplit = Matrix.__dict__[self.typeStr + "LUSplit"]
        # 对一个特定的 3x3 矩阵进行 LU 分解
        lower, upper = luSplit([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # 断言下三角矩阵是否符合预期
        self.assertEqual((lower == [[1, 0, 0], [4, 5, 0], [7, 8, 9]]).all(), True)
        # 断言上三角矩阵是否符合预期
        self.assertEqual((upper == [[0, 2, 3], [0, 0, 6], [0, 0, 0]]).all(), True)
######################################################################

# 创建一个名为 scharTestCase 的类，继承自 MatrixTestCase
class scharTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "schar"
        self.typeStr  = "schar"
        # 设置实例属性 typeCode 为字符 "b"
        self.typeCode = "b"

######################################################################

# 创建一个名为 ucharTestCase 的类，继承自 MatrixTestCase
class ucharTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "uchar"
        self.typeStr  = "uchar"
        # 设置实例属性 typeCode 为字符 "B"
        self.typeCode = "B"

######################################################################

# 创建一个名为 shortTestCase 的类，继承自 MatrixTestCase
class shortTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "short"
        self.typeStr  = "short"
        # 设置实例属性 typeCode 为字符 "h"
        self.typeCode = "h"

######################################################################

# 创建一个名为 ushortTestCase 的类，继承自 MatrixTestCase
class ushortTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "ushort"
        self.typeStr  = "ushort"
        # 设置实例属性 typeCode 为字符 "H"
        self.typeCode = "H"

######################################################################

# 创建一个名为 intTestCase 的类，继承自 MatrixTestCase
class intTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "int"
        self.typeStr  = "int"
        # 设置实例属性 typeCode 为字符 "i"
        self.typeCode = "i"

######################################################################

# 创建一个名为 uintTestCase 的类，继承自 MatrixTestCase
class uintTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "uint"
        self.typeStr  = "uint"
        # 设置实例属性 typeCode 为字符 "I"
        self.typeCode = "I"

######################################################################

# 创建一个名为 longTestCase 的类，继承自 MatrixTestCase
class longTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "long"
        self.typeStr  = "long"
        # 设置实例属性 typeCode 为字符 "l"
        self.typeCode = "l"

######################################################################

# 创建一个名为 ulongTestCase 的类，继承自 MatrixTestCase
class ulongTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "ulong"
        self.typeStr  = "ulong"
        # 设置实例属性 typeCode 为字符 "L"
        self.typeCode = "L"

######################################################################

# 创建一个名为 longLongTestCase 的类，继承自 MatrixTestCase
class longLongTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "longLong"
        self.typeStr  = "longLong"
        # 设置实例属性 typeCode 为字符 "q"
        self.typeCode = "q"

######################################################################

# 创建一个名为 ulongLongTestCase 的类，继承自 MatrixTestCase
class ulongLongTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "ulongLong"
        self.typeStr  = "ulongLong"
        # 设置实例属性 typeCode 为字符 "Q"
        self.typeCode = "Q"

######################################################################

# 创建一个名为 floatTestCase 的类，继承自 MatrixTestCase
class floatTestCase(MatrixTestCase):
    # 初始化方法，接受一个 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        MatrixTestCase.__init__(self, methodName)
        # 设置实例属性 typeStr 为字符串 "float"
        self.typeStr  = "float"
        # 设置实例属性 typeCode 为字符 "f"
        self.typeCode = "f"

######################################################################
######################################################################

class doubleTestCase(MatrixTestCase):
    # 定义一个名为 doubleTestCase 的测试用例类，继承自 MatrixTestCase
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法，设置测试方法的名称
        MatrixTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "double"
        self.typeStr  = "double"
        # 设置测试类型代码为 "d"
        self.typeCode = "d"

######################################################################

if __name__ == "__main__":

    # 构建测试套件
    suite = unittest.TestSuite()
    # 将各个测试用例类添加到测试套件中
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

    # 执行测试套件
    print("Testing 2D Functions of Module Matrix")
    print("NumPy version", np.__version__)
    print()
    # 运行测试套件并获取测试结果
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # 根据测试结果，如果有错误或者失败的测试，以非零退出码退出程序
    sys.exit(bool(result.errors + result.failures))


这段代码是一个用于测试的主程序，它构建了一个包含各种数据类型测试用例的测试套件，并执行这些测试用例。
```