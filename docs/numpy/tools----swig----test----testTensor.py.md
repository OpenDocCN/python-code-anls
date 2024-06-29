# `.\numpy\tools\swig\test\testTensor.py`

```
#!/usr/bin/env python3
# System imports
# 导入数学库中的平方根函数
from   math           import sqrt
# 导入系统库
import sys
# 导入单元测试框架
import unittest

# Import NumPy
# 导入 NumPy 库，并获取其主要和次要版本号
import numpy as np
major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
# 根据 NumPy 版本号设置异常类型
if major == 0: BadListError = TypeError
else:          BadListError = ValueError

# 导入自定义的 Tensor 模块
import Tensor

######################################################################

# 定义 TensorTestCase 类，继承自 unittest.TestCase
class TensorTestCase(unittest.TestCase):

    # 构造函数，初始化测试用例
    def __init__(self, methodName="runTests"):
        # 调用父类的构造函数
        unittest.TestCase.__init__(self, methodName)
        # 初始化类型字符串
        self.typeStr  = "double"
        # 初始化类型代码
        self.typeCode = "d"
        # 初始化预期结果，为计算平方根后的值
        self.result   = sqrt(28.0/8)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    # 测试 norm 函数
    def testNorm(self):
        "Test norm function"
        # 输出类型字符串到标准错误
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Tensor 模块中对应类型的 norm 函数
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        # 定义测试用的三维张量
        tensor = [[[0, 1], [2, 3]],
                  [[3, 2], [1, 0]]]
        # 根据结果类型进行断言检查
        if isinstance(self.result, int):
            self.assertEqual(norm(tensor), self.result)
        else:
            self.assertAlmostEqual(norm(tensor), self.result, 6)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    # 测试包含不良数据的 norm 函数
    def testNormBadList(self):
        "Test norm function with bad list"
        # 输出类型字符串到标准错误
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Tensor 模块中对应类型的 norm 函数
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        # 定义包含不良数据的测试三维张量
        tensor = [[[0, "one"], [2, 3]],
                  [[3, "two"], [1, 0]]]
        # 检查是否引发预期的异常
        self.assertRaises(BadListError, norm, tensor)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    # 测试维度错误的 norm 函数
    def testNormWrongDim(self):
        "Test norm function with wrong dimensions"
        # 输出类型字符串到标准错误
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Tensor 模块中对应类型的 norm 函数
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        # 定义维度错误的测试二维张量
        tensor = [[0, 1, 2, 3],
                  [3, 2, 1, 0]]
        # 检查是否引发预期的异常
        self.assertRaises(TypeError, norm, tensor)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    # 测试大小错误的 norm 函数
    def testNormWrongSize(self):
        "Test norm function with wrong size"
        # 输出类型字符串到标准错误
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Tensor 模块中对应类型的 norm 函数
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        # 定义大小错误的测试三维张量
        tensor = [[[0, 1, 0], [2, 3, 2]],
                  [[3, 2, 3], [1, 0, 1]]]
        # 检查是否引发预期的异常
        self.assertRaises(TypeError, norm, tensor)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    # 测试非容器类型的 norm 函数
    def testNormNonContainer(self):
        "Test norm function with non-container"
        # 输出类型字符串到标准错误
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Tensor 模块中对应类型的 norm 函数
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        # 检查是否引发预期的异常
        self.assertRaises(TypeError, norm, None)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    # 测试 max 函数
    def testMax(self):
        "Test max function"
        # 输出类型字符串到标准错误
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取 Tensor 模块中对应类型的 max 函数
        max = Tensor.__dict__[self.typeStr + "Max"]
        # 定义测试用的三维张量
        tensor = [[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]]
        # 检查最大值计算是否正确
        self.assertEqual(max(tensor), 8)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    # 定义一个测试方法，用于测试在存在不良列表情况下的最大函数
    def testMaxBadList(self):
        # 打印当前类型字符串，作为测试标识输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的最大值函数
        max = Tensor.__dict__[self.typeStr + "Max"]
        # 创建一个三维列表，其中包含混合类型的元素
        tensor = [[[1, "two"], [3, 4]],
                  [[5, "six"], [7, 8]]]
        # 断言调用最大值函数时抛出不良列表错误
        self.assertRaises(BadListError, max, tensor)

    # 测试具有非容器类型参数的最大函数
    def testMaxNonContainer(self):
        # 打印当前类型字符串，作为测试标识输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的最大值函数
        max = Tensor.__dict__[self.typeStr + "Max"]
        # 断言调用最大值函数时抛出类型错误，传入参数为 None
        self.assertRaises(TypeError, max, None)

    # 测试具有错误维度的最大函数
    def testMaxWrongDim(self):
        # 打印当前类型字符串，作为测试标识输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的最大值函数
        max = Tensor.__dict__[self.typeStr + "Max"]
        # 断言调用最大值函数时抛出类型错误，传入参数为一维列表
        self.assertRaises(TypeError, max, [0, -1, 2, -3])

    # 测试最小函数
    def testMin(self):
        # 打印当前类型字符串，作为测试标识输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的最小值函数
        min = Tensor.__dict__[self.typeStr + "Min"]
        # 创建一个三维列表，包含整数元素
        tensor = [[[9, 8], [7, 6]],
                  [[5, 4], [3, 2]]]
        # 断言调用最小值函数返回的结果为 2
        self.assertEqual(min(tensor), 2)

    # 测试具有不良列表的最小函数
    def testMinBadList(self):
        # 打印当前类型字符串，作为测试标识输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的最小值函数
        min = Tensor.__dict__[self.typeStr + "Min"]
        # 创建一个三维列表，其中包含混合类型的元素
        tensor = [[["nine", 8], [7, 6]],
                  [["five", 4], [3, 2]]]
        # 断言调用最小值函数时抛出不良列表错误
        self.assertRaises(BadListError, min, tensor)

    # 测试具有非容器类型参数的最小函数
    def testMinNonContainer(self):
        # 打印当前类型字符串，作为测试标识输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的最小值函数
        min = Tensor.__dict__[self.typeStr + "Min"]
        # 断言调用最小值函数时抛出类型错误，传入参数为布尔值 True
        self.assertRaises(TypeError, min, True)

    # 测试具有错误维度的最小函数
    def testMinWrongDim(self):
        # 打印当前类型字符串，作为测试标识输出到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的最小值函数
        min = Tensor.__dict__[self.typeStr + "Min"]
        # 断言调用最小值函数时抛出类型错误，传入参数为二维列表
        self.assertRaises(TypeError, min, [[1, 3], [5, 7]])

    # 测试 (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    # 定义一个名为 testScale 的测试方法，用于测试 scale 函数
    def testScale(self):
        # 打印测试类型的字符串，输出到标准错误流中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 根据类型字符串获取对应的 scale 函数
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        # 创建一个三维 NumPy 数组作为测试用例，使用给定的类型码
        tensor = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                          [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                          [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], self.typeCode)
        # 调用 scale 函数对 tensor 进行缩放操作
        scale(tensor, 4)
        # 使用断言检查 tensor 是否等于预期的值
        self.assertEqual((tensor == [[[4, 0, 4], [0, 4, 0], [4, 0, 4]],
                                      [[0, 4, 0], [4, 0, 4], [0, 4, 0]],
                                      [[4, 0, 4], [0, 4, 0], [4, 0, 4]]]).all(), True)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    # 定义一个名为 testScaleWrongType 的测试方法，测试 scale 函数对于错误类型的处理
    def testScaleWrongType(self):
        # 打印测试类型的字符串，输出到标准错误流中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 根据类型字符串获取对应的 scale 函数
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        # 创建一个三维 NumPy 数组作为测试用例，使用错误的类型码 'c'
        tensor = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                          [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                          [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], 'c')
        # 使用断言检查 scale 函数是否能正确抛出 TypeError 异常
        self.assertRaises(TypeError, scale, tensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    # 定义一个名为 testScaleWrongDim 的测试方法，测试 scale 函数对于错误维度的处理
    def testScaleWrongDim(self):
        # 打印测试类型的字符串，输出到标准错误流中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 根据类型字符串获取对应的 scale 函数
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        # 创建一个二维 NumPy 数组作为测试用例，使用给定的类型码
        tensor = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1],
                          [0, 1, 0], [1, 0, 1], [0, 1, 0]], self.typeCode)
        # 使用断言检查 scale 函数是否能正确抛出 TypeError 异常
        self.assertRaises(TypeError, scale, tensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    # 定义一个名为 testScaleWrongSize 的测试方法，测试 scale 函数对于错误大小的处理
    def testScaleWrongSize(self):
        # 打印测试类型的字符串，输出到标准错误流中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 根据类型字符串获取对应的 scale 函数
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        # 创建一个二维 NumPy 数组作为测试用例，使用给定的类型码
        tensor = np.array([[[1, 0], [0, 1], [1, 0]],
                          [[0, 1], [1, 0], [0, 1]],
                          [[1, 0], [0, 1], [1, 0]]], self.typeCode)
        # 使用断言检查 scale 函数是否能正确抛出 TypeError 异常
        self.assertRaises(TypeError, scale, tensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    # 定义一个名为 testScaleNonArray 的测试方法，测试 scale 函数对于非数组输入的处理
    def testScaleNonArray(self):
        # 打印测试类型的字符串，输出到标准错误流中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 根据类型字符串获取对应的 scale 函数
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        # 使用断言检查 scale 函数是否能正确抛出 TypeError 异常，传入 True 作为参数
        self.assertRaises(TypeError, scale, True)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    # 定义一个名为 testFloor 的测试方法，测试 floor 函数
    def testFloor(self):
        # 打印测试类型的字符串，输出到标准错误流中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 根据类型字符串获取对应的 floor 函数
        floor = Tensor.__dict__[self.typeStr + "Floor"]
        # 创建一个三维 NumPy 数组作为测试用例，使用给定的类型码
        tensor = np.array([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]]], self.typeCode)
        # 调用 floor 函数对 tensor 进行 floor 操作，将大于等于 4 的元素替换为 4
        floor(tensor, 4)
        # 使用 NumPy 的断言函数检查 tensor 是否与预期值相等
        np.testing.assert_array_equal(tensor, np.array([[[4, 4], [4, 4]],
                                                      [[5, 6], [7, 8]]]))

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    # 测试 floor 函数对于错误类型的输入
    def testFloorWrongType(self):
        # 打印类型字符串，指示测试正在进行中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 floor 函数
        floor = Tensor.__dict__[self.typeStr + "Floor"]
        # 创建一个三维 NumPy 数组，使用 'c' 表示以 C 风格存储
        tensor = np.array([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]]], 'c')
        # 断言调用 floor 函数时会引发 TypeError 异常
        self.assertRaises(TypeError, floor, tensor)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    # 测试 floor 函数对于维度错误的输入
    def testFloorWrongDim(self):
        # 打印类型字符串，指示测试正在进行中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 floor 函数
        floor = Tensor.__dict__[self.typeStr + "Floor"]
        # 创建一个二维 NumPy 数组，使用 self.typeCode 表示数据类型
        tensor = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], self.typeCode)
        # 断言调用 floor 函数时会引发 TypeError 异常
        self.assertRaises(TypeError, floor, tensor)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    # 测试 floor 函数对于非数组输入
    def testFloorNonArray(self):
        # 打印类型字符串，指示测试正在进行中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 floor 函数
        floor = Tensor.__dict__[self.typeStr + "Floor"]
        # 断言调用 floor 函数时会引发 TypeError 异常，输入为 object 类型
        self.assertRaises(TypeError, floor, object)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    # 测试 ceil 函数的正常使用
    def testCeil(self):
        # 打印类型字符串，指示测试正在进行中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 ceil 函数
        ceil = Tensor.__dict__[self.typeStr + "Ceil"]
        # 创建一个三维 NumPy 数组，使用 self.typeCode 表示数据类型
        tensor = np.array([[[9, 8], [7, 6]],
                          [[5, 4], [3, 2]]], self.typeCode)
        # 调用 ceil 函数对 tensor 进行操作
        ceil(tensor, 5)
        # 使用 NumPy 的断言方法验证数组是否符合预期
        np.testing.assert_array_equal(tensor, np.array([[[5, 5], [5, 5]],
                                                      [[5, 4], [3, 2]]]))

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    # 测试 ceil 函数对于错误类型的输入
    def testCeilWrongType(self):
        # 打印类型字符串，指示测试正在进行中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 ceil 函数
        ceil = Tensor.__dict__[self.typeStr + "Ceil"]
        # 创建一个三维 NumPy 数组，使用 'c' 表示以 C 风格存储
        tensor = np.array([[[9, 8], [7, 6]],
                          [[5, 4], [3, 2]]], 'c')
        # 断言调用 ceil 函数时会引发 TypeError 异常
        self.assertRaises(TypeError, ceil, tensor)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    # 测试 ceil 函数对于维度错误的输入
    def testCeilWrongDim(self):
        # 打印类型字符串，指示测试正在进行中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 ceil 函数
        ceil = Tensor.__dict__[self.typeStr + "Ceil"]
        # 创建一个二维 NumPy 数组，使用 self.typeCode 表示数据类型
        tensor = np.array([[9, 8], [7, 6], [5, 4], [3, 2]], self.typeCode)
        # 断言调用 ceil 函数时会引发 TypeError 异常
        self.assertRaises(TypeError, ceil, tensor)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    # 测试 ceil 函数对于非数组输入
    def testCeilNonArray(self):
        # 打印类型字符串，指示测试正在进行中
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        # 获取当前类型对应的 ceil 函数
        ceil = Tensor.__dict__[self.typeStr + "Ceil"]
        # 创建一个嵌套列表表示的三维数组
        tensor = [[[9, 8], [7, 6]],
                  [[5, 4], [3, 2]]]
        # 断言调用 ceil 函数时会引发 TypeError 异常
        self.assertRaises(TypeError, ceil, tensor)

    # Test (type ARGOUT_ARRAY3[ANY][ANY][ANY]) typemap
    # 这个测试用例的注释未提供，可能需要补充
    def testLUSplit(self):
        "Test luSplit function"
        # 打印测试类型字符串到标准错误流，不换行
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        
        # 获取当前类型的 luSplit 函数
        luSplit = Tensor.__dict__[self.typeStr + "LUSplit"]
        
        # 对输入张量进行 LU 分解，分别得到下三角矩阵 lower 和上三角矩阵 upper
        lower, upper = luSplit([[[1, 1], [1, 1]],
                                [[1, 1], [1, 1]]])
        
        # 断言下三角矩阵 lower 是否与预期相等
        self.assertEqual((lower == [[[1, 1], [1, 0]],
                                     [[1, 0], [0, 0]]]).all(), True)
        
        # 断言上三角矩阵 upper 是否与预期相等
        self.assertEqual((upper == [[[0, 0], [0, 1]],
                                     [[0, 1], [1, 1]]]).all(), True)
######################################################################

class scharTestCase(TensorTestCase):
    # scharTestCase 类，继承自 TensorTestCase 类，用于测试有符号字符类型的测试用例
    def __init__(self, methodName="runTest"):
        # 初始化方法，调用父类的初始化方法，并设置类型字符串、类型代码和结果值
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "schar"
        self.typeCode = "b"
        self.result   = int(self.result)

######################################################################

class ucharTestCase(TensorTestCase):
    # ucharTestCase 类，继承自 TensorTestCase 类，用于测试无符号字符类型的测试用例
    def __init__(self, methodName="runTest"):
        # 初始化方法，调用父类的初始化方法，并设置类型字符串、类型代码和结果值
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "uchar"
        self.typeCode = "B"
        self.result   = int(self.result)

######################################################################

class shortTestCase(TensorTestCase):
    # shortTestCase 类，继承自 TensorTestCase 类，用于测试短整型类型的测试用例
    def __init__(self, methodName="runTest"):
        # 初始化方法，调用父类的初始化方法，并设置类型字符串、类型代码和结果值
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "short"
        self.typeCode = "h"
        self.result   = int(self.result)

######################################################################

class ushortTestCase(TensorTestCase):
    # ushortTestCase 类，继承自 TensorTestCase 类，用于测试无符号短整型类型的测试用例
    def __init__(self, methodName="runTest"):
        # 初始化方法，调用父类的初始化方法，并设置类型字符串、类型代码和结果值
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "ushort"
        self.typeCode = "H"
        self.result   = int(self.result)

######################################################################

class intTestCase(TensorTestCase):
    # intTestCase 类，继承自 TensorTestCase 类，用于测试整型类型的测试用例
    def __init__(self, methodName="runTest"):
        # 初始化方法，调用父类的初始化方法，并设置类型字符串、类型代码和结果值
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "int"
        self.typeCode = "i"
        self.result   = int(self.result)

######################################################################

class uintTestCase(TensorTestCase):
    # uintTestCase 类，继承自 TensorTestCase 类，用于测试无符号整型类型的测试用例
    def __init__(self, methodName="runTest"):
        # 初始化方法，调用父类的初始化方法，并设置类型字符串、类型代码和结果值
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "uint"
        self.typeCode = "I"
        self.result   = int(self.result)

######################################################################

class longTestCase(TensorTestCase):
    # longTestCase 类，继承自 TensorTestCase 类，用于测试长整型类型的测试用例
    def __init__(self, methodName="runTest"):
        # 初始化方法，调用父类的初始化方法，并设置类型字符串、类型代码和结果值
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "long"
        self.typeCode = "l"
        self.result   = int(self.result)

######################################################################

class ulongTestCase(TensorTestCase):
    # ulongTestCase 类，继承自 TensorTestCase 类，用于测试无符号长整型类型的测试用例
    def __init__(self, methodName="runTest"):
        # 初始化方法，调用父类的初始化方法，并设置类型字符串、类型代码和结果值
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "ulong"
        self.typeCode = "L"
        self.result   = int(self.result)

######################################################################

class longLongTestCase(TensorTestCase):
    # longLongTestCase 类，继承自 TensorTestCase 类，用于测试长长整型类型的测试用例
    def __init__(self, methodName="runTest"):
        # 初始化方法，调用父类的初始化方法，并设置类型字符串、类型代码和结果值
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "longLong"
        self.typeCode = "q"
        self.result   = int(self.result)

######################################################################

class ulongLongTestCase(TensorTestCase):
    # ulongLongTestCase 类，继承自 TensorTestCase 类，用于测试无符号长长整型类型的测试用例
    # 初始化方法，用于创建对象实例
    def __init__(self, methodName="runTest"):
        # 调用父类 TensorTestCase 的初始化方法，传入当前对象实例和方法名
        TensorTestCase.__init__(self, methodName)
        # 设置对象实例的 typeStr 属性为字符串 "ulongLong"
        self.typeStr  = "ulongLong"
        # 设置对象实例的 typeCode 属性为字符串 "Q"
        self.typeCode = "Q"
        # 将对象实例的 result 属性转换为整数类型
        self.result   = int(self.result)
######################################################################

# 定义一个继承自 TensorTestCase 的浮点数测试用例类
class floatTestCase(TensorTestCase):
    # 初始化方法，接受一个可选的 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类 TensorTestCase 的初始化方法
        TensorTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "float"
        self.typeStr  = "float"
        # 设置测试类型代码为 "f"
        self.typeCode = "f"

######################################################################

# 定义一个继承自 TensorTestCase 的双精度浮点数测试用例类
class doubleTestCase(TensorTestCase):
    # 初始化方法，接受一个可选的 methodName 参数，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类 TensorTestCase 的初始化方法
        TensorTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "double"
        self.typeStr  = "double"
        # 设置测试类型代码为 "d"

######################################################################

# 确保当前文件是主程序时执行以下代码块
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
    print("Testing 3D Functions of Module Tensor")
    # 打印 NumPy 版本信息
    print("NumPy version", np.__version__)
    print()
    # 运行测试套件，并获取测试结果
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # 根据测试结果，退出程序并返回是否有错误或失败的布尔值
    sys.exit(bool(result.errors + result.failures))
```