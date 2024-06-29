# `.\numpy\tools\swig\test\testSuperTensor.py`

```
#!/usr/bin/env python3
# System imports
import sys  # 导入系统模块sys，用于处理系统相关的功能
import unittest  # 导入unittest模块，用于编写和运行单元测试

# Import NumPy
import numpy as np  # 导入NumPy库，用于数值计算

major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
if major == 0: BadListError = TypeError  # 如果NumPy的主版本号为0，则定义BadListError为TypeError
else:          BadListError = ValueError  # 否则定义BadListError为ValueError

import SuperTensor  # 导入SuperTensor模块，假设这是一个自定义的张量处理模块

######################################################################

class SuperTensorTestCase(unittest.TestCase):

    def __init__(self, methodName="runTests"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr  = "double"  # 定义测试用例中的类型字符串为"double"
        self.typeCode = "d"       # 定义测试用例中的类型代码为"d"

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNorm(self):
        "Test norm function"
        print(self.typeStr, "... ", file=sys.stderr)  # 打印当前类型字符串到标准错误流
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]  # 获取SuperTensor模块中的对应类型的Norm函数
        supertensor = np.arange(2*2*2*2, dtype=self.typeCode).reshape((2, 2, 2, 2))  # 创建一个4维数组作为输入张量
        #Note: cludge to get an answer of the same type as supertensor.
        #Answer is simply sqrt(sum(supertensor*supertensor)/16)
        answer = np.array([np.sqrt(np.sum(supertensor.astype('d')*supertensor)/16.)], dtype=self.typeCode)[0]
        self.assertAlmostEqual(norm(supertensor), answer, 6)  # 断言计算的norm函数值接近预期值，精确到小数点后6位

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormBadList(self):
        "Test norm function with bad list"
        print(self.typeStr, "... ", file=sys.stderr)  # 打印当前类型字符串到标准错误流
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]  # 获取SuperTensor模块中的对应类型的Norm函数
        supertensor = [[[[0, "one"], [2, 3]], [[3, "two"], [1, 0]]], [[[0, "one"], [2, 3]], [[3, "two"], [1, 0]]]]
        # 创建一个包含不合法元素的多维列表作为输入张量
        self.assertRaises(BadListError, norm, supertensor)  # 断言调用norm函数时抛出BadListError异常

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormWrongDim(self):
        "Test norm function with wrong dimensions"
        print(self.typeStr, "... ", file=sys.stderr)  # 打印当前类型字符串到标准错误流
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]  # 获取SuperTensor模块中的对应类型的Norm函数
        supertensor = np.arange(2*2*2, dtype=self.typeCode).reshape((2, 2, 2))  # 创建一个3维数组作为输入张量
        self.assertRaises(TypeError, norm, supertensor)  # 断言调用norm函数时抛出TypeError异常

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormWrongSize(self):
        "Test norm function with wrong size"
        print(self.typeStr, "... ", file=sys.stderr)  # 打印当前类型字符串到标准错误流
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]  # 获取SuperTensor模块中的对应类型的Norm函数
        supertensor = np.arange(3*2*2, dtype=self.typeCode).reshape((3, 2, 2))  # 创建一个不符合尺寸要求的输入张量
        self.assertRaises(TypeError, norm, supertensor)  # 断言调用norm函数时抛出TypeError异常

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormNonContainer(self):
        "Test norm function with non-container"
        print(self.typeStr, "... ", file=sys.stderr)  # 打印当前类型字符串到标准错误流
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]  # 获取SuperTensor模块中的对应类型的Norm函数
        self.assertRaises(TypeError, norm, None)  # 断言调用norm函数时抛出TypeError异常

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMax(self):
        "Test max function"
        print(self.typeStr, "... ", file=sys.stderr)  # 打印当前类型字符串到标准错误流
        max = SuperTensor.__dict__[self.typeStr + "Max"]  # 获取SuperTensor模块中的对应类型的Max函数
        supertensor = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]
        # 创建一个多维列表作为输入张量
        self.assertEqual(max(supertensor), 8)  # 断言调用max函数计算的最大值为预期的8
    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMaxBadList(self):
        "Test max function with bad list"
        # 输出正在进行的测试类型到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取当前类型对应的最大值函数
        max = SuperTensor.__dict__[self.typeStr + "Max"]
        # 创建一个包含非法元素的超级张量
        supertensor = [[[[1, "two"], [3, 4]], [[5, "six"], [7, 8]]], [[[1, "two"], [3, 4]], [[5, "six"], [7, 8]]]]
        # 断言调用最大值函数时会抛出 BadListError 异常
        self.assertRaises(BadListError, max, supertensor)
    
    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMaxNonContainer(self):
        "Test max function with non-container"
        # 输出正在进行的测试类型到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取当前类型对应的最大值函数
        max = SuperTensor.__dict__[self.typeStr + "Max"]
        # 断言调用最大值函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, max, None)
    
    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMaxWrongDim(self):
        "Test max function with wrong dimensions"
        # 输出正在进行的测试类型到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取当前类型对应的最大值函数
        max = SuperTensor.__dict__[self.typeStr + "Max"]
        # 断言调用最大值函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, max, [0, -1, 2, -3])
    
    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMin(self):
        "Test min function"
        # 输出正在进行的测试类型到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取当前类型对应的最小值函数
        min = SuperTensor.__dict__[self.typeStr + "Min"]
        # 创建一个正常的超级张量
        supertensor = [[[[9, 8], [7, 6]], [[5, 4], [3, 2]]], [[[9, 8], [7, 6]], [[5, 4], [3, 2]]]]
        # 断言调用最小值函数返回值为预期的最小值
        self.assertEqual(min(supertensor), 2)
    
    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMinBadList(self):
        "Test min function with bad list"
        # 输出正在进行的测试类型到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取当前类型对应的最小值函数
        min = SuperTensor.__dict__[self.typeStr + "Min"]
        # 创建一个包含非法元素的超级张量
        supertensor = [[[["nine", 8], [7, 6]], [["five", 4], [3, 2]]], [[["nine", 8], [7, 6]], [["five", 4], [3, 2]]]]
        # 断言调用最小值函数时会抛出 BadListError 异常
        self.assertRaises(BadListError, min, supertensor)
    
    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMinNonContainer(self):
        "Test min function with non-container"
        # 输出正在进行的测试类型到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取当前类型对应的最小值函数
        min = SuperTensor.__dict__[self.typeStr + "Min"]
        # 断言调用最小值函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, min, True)
    
    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMinWrongDim(self):
        "Test min function with wrong dimensions"
        # 输出正在进行的测试类型到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取当前类型对应的最小值函数
        min = SuperTensor.__dict__[self.typeStr + "Min"]
        # 断言调用最小值函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, min, [[1, 3], [5, 7]])
    
    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScale(self):
        "Test scale function"
        # 输出正在进行的测试类型到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取当前类型对应的缩放函数
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        # 创建一个 NumPy 超级张量，用于测试
        supertensor = np.arange(3*3*3*3, dtype=self.typeCode).reshape((3, 3, 3, 3))
        # 创建预期的结果，通过复制当前张量并缩放
        answer = supertensor.copy()*4
        # 调用缩放函数
        scale(supertensor, 4)
        # 断言张量缩放后的结果符合预期
        self.assertEqual((supertensor == answer).all(), True)
    # 定义一个测试方法，用于测试在错误类型情况下调用 scale 函数
    def testScaleWrongType(self):
        # 打印当前对象的 typeStr 属性到标准错误输出流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 scale 函数
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        # 创建一个三维的 numpy 数组作为测试输入
        supertensor = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                          [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                          [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], 'c')
        # 断言调用 scale 函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, scale, supertensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    # 定义一个测试方法，用于测试在错误维度情况下调用 scale 函数
    def testScaleWrongDim(self):
        # 打印当前对象的 typeStr 属性到标准错误输出流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 scale 函数
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        # 创建一个错误维度的 numpy 数组作为测试输入
        supertensor = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1],
                          [0, 1, 0], [1, 0, 1], [0, 1, 0]], self.typeCode)
        # 断言调用 scale 函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, scale, supertensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    # 定义一个测试方法，用于测试在错误尺寸情况下调用 scale 函数
    def testScaleWrongSize(self):
        # 打印当前对象的 typeStr 属性到标准错误输出流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 scale 函数
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        # 创建一个错误尺寸的 numpy 数组作为测试输入
        supertensor = np.array([[[1, 0], [0, 1], [1, 0]],
                          [[0, 1], [1, 0], [0, 1]],
                          [[1, 0], [0, 1], [1, 0]]], self.typeCode)
        # 断言调用 scale 函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, scale, supertensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    # 定义一个测试方法，用于测试在非数组类型情况下调用 scale 函数
    def testScaleNonArray(self):
        # 打印当前对象的 typeStr 属性到标准错误输出流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 scale 函数
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        # 断言调用 scale 函数时会抛出 TypeError 异常，传入的参数为 True
        self.assertRaises(TypeError, scale, True)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    # 定义一个测试方法，用于测试 floor 函数的正常操作
    def testFloor(self):
        # 打印当前对象的 typeStr 属性到标准错误输出流
        print(self.typeStr, "... ", file=sys.stderr)
        # 使用 typeCode 创建一个特定类型的 numpy 数组
        supertensor = np.arange(2*2*2*2, dtype=self.typeCode).reshape((2, 2, 2, 2))
        # 复制 supertensor 以备后续比较
        answer = supertensor.copy()
        # 将 answer 数组中小于 4 的元素设置为 4
        answer[answer < 4] = 4

        # 获取对应类型的 floor 函数
        floor = SuperTensor.__dict__[self.typeStr + "Floor"]
        # 调用 floor 函数，将 supertensor 中小于 4 的元素设置为 4
        floor(supertensor, 4)
        # 使用 numpy.testing.assert_array_equal 检查 supertensor 是否等于 answer
        np.testing.assert_array_equal(supertensor, answer)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    # 定义一个测试方法，用于测试在错误类型情况下调用 floor 函数
    def testFloorWrongType(self):
        # 打印当前对象的 typeStr 属性到标准错误输出流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 floor 函数
        floor = SuperTensor.__dict__[self.typeStr + "Floor"]
        # 创建一个错误类型的 numpy 数组作为测试输入
        supertensor = np.ones(2*2*2*2, dtype='c').reshape((2, 2, 2, 2))
        # 断言调用 floor 函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, floor, supertensor)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    # 定义一个测试方法，用于测试在错误维度情况下调用 floor 函数
    def testFloorWrongDim(self):
        # 打印当前对象的 typeStr 属性到标准错误输出流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 floor 函数
        floor = SuperTensor.__dict__[self.typeStr + "Floor"]
        # 创建一个错误维度的 numpy 数组作为测试输入
        supertensor = np.arange(2*2*2, dtype=self.typeCode).reshape((2, 2, 2))
        # 断言调用 floor 函数时会抛出 TypeError 异常
        self.assertRaises(TypeError, floor, supertensor)
    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testFloorNonArray(self):
        "Test floor function with non-array"
        # 打印当前类型字符串到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 floor 函数并调用，预期会抛出 TypeError 异常
        floor = SuperTensor.__dict__[self.typeStr + "Floor"]
        self.assertRaises(TypeError, floor, object)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeil(self):
        "Test ceil function"
        # 打印当前类型字符串到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 创建一个指定类型和形状的超级张量，并复制一个相同的答案张量
        supertensor = np.arange(2*2*2*2, dtype=self.typeCode).reshape((2, 2, 2, 2))
        answer = supertensor.copy()
        # 将答案张量中大于5的元素设置为5
        answer[answer > 5] = 5
        # 获取对应类型的 ceil 函数并调用，修改 supertensor
        ceil = SuperTensor.__dict__[self.typeStr + "Ceil"]
        ceil(supertensor, 5)
        # 使用 NumPy 测试断言确保 supertensor 和答案张量相等
        np.testing.assert_array_equal(supertensor, answer)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeilWrongType(self):
        "Test ceil function with wrong type"
        # 打印当前类型字符串到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 ceil 函数并调用，传入一个不合适类型的超级张量
        ceil = SuperTensor.__dict__[self.typeStr + "Ceil"]
        supertensor = np.ones(2*2*2*2, 'c').reshape((2, 2, 2, 2))
        # 预期会抛出 TypeError 异常
        self.assertRaises(TypeError, ceil, supertensor)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeilWrongDim(self):
        "Test ceil function with wrong dimensions"
        # 打印当前类型字符串到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 ceil 函数并调用，传入一个维度不匹配的超级张量
        ceil = SuperTensor.__dict__[self.typeStr + "Ceil"]
        supertensor = np.arange(2*2*2, dtype=self.typeCode).reshape((2, 2, 2))
        # 预期会抛出 TypeError 异常
        self.assertRaises(TypeError, ceil, supertensor)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeilNonArray(self):
        "Test ceil function with non-array"
        # 打印当前类型字符串到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 ceil 函数并调用，传入一个转换为列表的超级张量
        ceil = SuperTensor.__dict__[self.typeStr + "Ceil"]
        supertensor = np.arange(2*2*2*2, dtype=self.typeCode).reshape((2, 2, 2, 2)).tolist()
        # 预期会抛出 TypeError 异常
        self.assertRaises(TypeError, ceil, supertensor)

    # Test (type ARGOUT_ARRAY3[ANY][ANY][ANY]) typemap
    def testLUSplit(self):
        "Test luSplit function"
        # 打印当前类型字符串到标准错误流
        print(self.typeStr, "... ", file=sys.stderr)
        # 获取对应类型的 luSplit 函数并调用，传入一个全为1的超级张量
        luSplit = SuperTensor.__dict__[self.typeStr + "LUSplit"]
        supertensor = np.ones(2*2*2*2, dtype=self.typeCode).reshape((2, 2, 2, 2))
        # 预期 lower 和 upper 分解的结果与设定的答案相等
        answer_upper = [[[[0, 0], [0, 1]], [[0, 1], [1, 1]]], [[[0, 1], [1, 1]], [[1, 1], [1, 1]]]]
        answer_lower = [[[[1, 1], [1, 0]], [[1, 0], [0, 0]]], [[[1, 0], [0, 0]], [[0, 0], [0, 0]]]]
        lower, upper = luSplit(supertensor)
        # 使用断言确保 lower 和 answer_lower 相等
        self.assertEqual((lower == answer_lower).all(), True)
        # 使用断言确保 upper 和 answer_upper 相等
        self.assertEqual((upper == answer_upper).all(), True)
######################################################################

class scharTestCase(SuperTensorTestCase):
    # scharTestCase 类，继承自 SuperTensorTestCase 类

    def __init__(self, methodName="runTest"):
        # 初始化方法，接受一个 methodName 参数，默认为 "runTest"

        SuperTensorTestCase.__init__(self, methodName)
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数

        self.typeStr  = "schar"
        # 设置实例变量 typeStr 为字符串 "schar"
        
        self.typeCode = "b"
        # 设置实例变量 typeCode 为字符 "b"

        #self.result   = int(self.result)
        # 注释掉的代码，可能是待解开的功能性代码或注释

######################################################################

class ucharTestCase(SuperTensorTestCase):
    # ucharTestCase 类，继承自 SuperTensorTestCase 类

    def __init__(self, methodName="runTest"):
        # 初始化方法，接受一个 methodName 参数，默认为 "runTest"

        SuperTensorTestCase.__init__(self, methodName)
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数

        self.typeStr  = "uchar"
        # 设置实例变量 typeStr 为字符串 "uchar"
        
        self.typeCode = "B"
        # 设置实例变量 typeCode 为字符 "B"

        #self.result   = int(self.result)
        # 注释掉的代码，可能是待解开的功能性代码或注释

######################################################################

class shortTestCase(SuperTensorTestCase):
    # shortTestCase 类，继承自 SuperTensorTestCase 类

    def __init__(self, methodName="runTest"):
        # 初始化方法，接受一个 methodName 参数，默认为 "runTest"

        SuperTensorTestCase.__init__(self, methodName)
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数

        self.typeStr  = "short"
        # 设置实例变量 typeStr 为字符串 "short"
        
        self.typeCode = "h"
        # 设置实例变量 typeCode 为字符 "h"

        #self.result   = int(self.result)
        # 注释掉的代码，可能是待解开的功能性代码或注释

######################################################################

class ushortTestCase(SuperTensorTestCase):
    # ushortTestCase 类，继承自 SuperTensorTestCase 类

    def __init__(self, methodName="runTest"):
        # 初始化方法，接受一个 methodName 参数，默认为 "runTest"

        SuperTensorTestCase.__init__(self, methodName)
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数

        self.typeStr  = "ushort"
        # 设置实例变量 typeStr 为字符串 "ushort"
        
        self.typeCode = "H"
        # 设置实例变量 typeCode 为字符 "H"

        #self.result   = int(self.result)
        # 注释掉的代码，可能是待解开的功能性代码或注释

######################################################################

class intTestCase(SuperTensorTestCase):
    # intTestCase 类，继承自 SuperTensorTestCase 类

    def __init__(self, methodName="runTest"):
        # 初始化方法，接受一个 methodName 参数，默认为 "runTest"

        SuperTensorTestCase.__init__(self, methodName)
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数

        self.typeStr  = "int"
        # 设置实例变量 typeStr 为字符串 "int"
        
        self.typeCode = "i"
        # 设置实例变量 typeCode 为字符 "i"

        #self.result   = int(self.result)
        # 注释掉的代码，可能是待解开的功能性代码或注释

######################################################################

class uintTestCase(SuperTensorTestCase):
    # uintTestCase 类，继承自 SuperTensorTestCase 类

    def __init__(self, methodName="runTest"):
        # 初始化方法，接受一个 methodName 参数，默认为 "runTest"

        SuperTensorTestCase.__init__(self, methodName)
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数

        self.typeStr  = "uint"
        # 设置实例变量 typeStr 为字符串 "uint"
        
        self.typeCode = "I"
        # 设置实例变量 typeCode 为字符 "I"

        #self.result   = int(self.result)
        # 注释掉的代码，可能是待解开的功能性代码或注释

######################################################################

class longTestCase(SuperTensorTestCase):
    # longTestCase 类，继承自 SuperTensorTestCase 类

    def __init__(self, methodName="runTest"):
        # 初始化方法，接受一个 methodName 参数，默认为 "runTest"

        SuperTensorTestCase.__init__(self, methodName)
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数

        self.typeStr  = "long"
        # 设置实例变量 typeStr 为字符串 "long"
        
        self.typeCode = "l"
        # 设置实例变量 typeCode 为字符 "l"

        #self.result   = int(self.result)
        # 注释掉的代码，可能是待解开的功能性代码或注释

######################################################################

class ulongTestCase(SuperTensorTestCase):
    # ulongTestCase 类，继承自 SuperTensorTestCase 类

    def __init__(self, methodName="runTest"):
        # 初始化方法，接受一个 methodName 参数，默认为 "runTest"

        SuperTensorTestCase.__init__(self, methodName)
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数

        self.typeStr  = "ulong"
        # 设置实例变量 typeStr 为字符串 "ulong"
        
        self.typeCode = "L"
        # 设置实例变量 typeCode 为字符 "L"

        #self.result   = int(self.result)
        # 注释掉的代码，可能是待解开的功能性代码或注释

######################################################################

class longLongTestCase(SuperTensorTestCase):
    # longLongTestCase 类，继承自 SuperTensorTestCase 类

    def __init__(self, methodName="runTest"):
        # 初始化方法，接受一个 methodName 参数，默认为 "runTest"

        SuperTensorTestCase.__init__(self, methodName)
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数

        self.typeStr  = "longLong"
        # 设置实例变量 typeStr 为字符串 "longLong"
        
        self.typeCode = "q"
        # 设置实例变量 typeCode 为字符 "q"

        #self.result   = int(self.result)
        # 注释掉的代码，可能是待解开的功能性代码或注释

######################################################################

class ulongLongTestCase(SuperTensorTestCase):
    # ulongLongTestCase 类，继承自 SuperTensorTestCase 类

    # 此处省略了类的具体实现，可能是意图留待后续添加
    # 定义类的初始化方法，接受一个可选参数 methodName，默认为"runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类 SuperTensorTestCase 的初始化方法，并传入 methodName 参数
        SuperTensorTestCase.__init__(self, methodName)
        # 设置实例变量 typeStr 为字符串 "ulongLong"
        self.typeStr  = "ulongLong"
        # 设置实例变量 typeCode 为字符串 "Q"
        self.typeCode = "Q"
        # 注释掉的代码行，原来是设置实例变量 result 为整数化后的 self.result，已经被注释掉
        #self.result   = int(self.result)
######################################################################

class floatTestCase(SuperTensorTestCase):
    # floatTestCase 类，继承自 SuperTensorTestCase
    def __init__(self, methodName="runTest"):
        # 调用父类构造函数初始化对象
        SuperTensorTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "float"
        self.typeStr  = "float"
        # 设置测试类型代码为 "f"
        self.typeCode = "f"

######################################################################

class doubleTestCase(SuperTensorTestCase):
    # doubleTestCase 类，继承自 SuperTensorTestCase
    def __init__(self, methodName="runTest"):
        # 调用父类构造函数初始化对象
        SuperTensorTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "double"
        self.typeStr  = "double"
        # 设置测试类型代码为 "d"
        self.typeCode = "d"

######################################################################

if __name__ == "__main__":

    # 构建测试套件
    suite = unittest.TestSuite()
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
    print("Testing 4D Functions of Module SuperTensor")
    print("NumPy version", np.__version__)
    print()
    # 运行测试套件并获取结果
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # 根据测试结果判断是否有错误或失败，并退出程序
    sys.exit(bool(result.errors + result.failures))
```