# `.\numpy\tools\swig\test\testFortran.py`

```py
#!/usr/bin/env python3
# System imports
import sys  # 导入系统模块 sys
import unittest  # 导入单元测试模块 unittest

# Import NumPy
import numpy as np  # 导入 NumPy 库，并命名为 np
major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]  # 解析 NumPy 版本信息
if major == 0:  # 如果主版本号为 0
    BadListError = TypeError  # 则定义 BadListError 为 TypeError
else:  # 否则
    BadListError = ValueError  # 定义 BadListError 为 ValueError

import Fortran  # 导入 Fortran 模块

######################################################################

class FortranTestCase(unittest.TestCase):
    # 初始化方法
    def __init__(self, methodName="runTests"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr  = "double"  # 设置测试用例中的数据类型字符串为 "double"
        self.typeCode = "d"       # 设置测试用例中的数据类型代码为 "d"

    # Test (type* IN_FARRAY2, int DIM1, int DIM2) typemap
    def testSecondElementFortran(self):
        "Test Fortran matrix initialized from reshaped NumPy fortranarray"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)  # 打印当前数据类型字符串到标准错误输出
        second = Fortran.__dict__[self.typeStr + "SecondElement"]  # 获取 Fortran 模块中对应数据类型的第二个元素函数
        matrix = np.asfortranarray(np.arange(9).reshape(3, 3),
                                   self.typeCode)  # 生成一个 Fortran 风格的 NumPy 数组
        self.assertEqual(second(matrix), 3)  # 断言第二个元素函数对该数组的调用结果为 3

    def testSecondElementObject(self):
        "Test Fortran matrix initialized from nested list fortranarray"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)  # 打印当前数据类型字符串到标准错误输出
        second = Fortran.__dict__[self.typeStr + "SecondElement"]  # 获取 Fortran 模块中对应数据类型的第二个元素函数
        matrix = np.asfortranarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], self.typeCode)  # 生成一个 Fortran 风格的 NumPy 数组
        self.assertEqual(second(matrix), 3)  # 断言第二个元素函数对该数组的调用结果为 3

######################################################################

class scharTestCase(FortranTestCase):
    # 初始化方法
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "schar"  # 设置测试用例中的数据类型字符串为 "schar"
        self.typeCode = "b"      # 设置测试用例中的数据类型代码为 "b"

######################################################################

class ucharTestCase(FortranTestCase):
    # 初始化方法
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "uchar"  # 设置测试用例中的数据类型字符串为 "uchar"
        self.typeCode = "B"      # 设置测试用例中的数据类型代码为 "B"

######################################################################

class shortTestCase(FortranTestCase):
    # 初始化方法
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "short"  # 设置测试用例中的数据类型字符串为 "short"
        self.typeCode = "h"      # 设置测试用例中的数据类型代码为 "h"

######################################################################

class ushortTestCase(FortranTestCase):
    # 初始化方法
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "ushort"  # 设置测试用例中的数据类型字符串为 "ushort"
        self.typeCode = "H"       # 设置测试用例中的数据类型代码为 "H"

######################################################################

class intTestCase(FortranTestCase):
    # 初始化方法
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "int"  # 设置测试用例中的数据类型字符串为 "int"
        self.typeCode = "i"    # 设置测试用例中的数据类型代码为 "i"

######################################################################

class uintTestCase(FortranTestCase):
    # 初始化方法
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "uint"  # 设置测试用例中的数据类型字符串为 "uint"
        self.typeCode = "I"     # 设置测试用例中的数据类型代码为 "I"
######################################################################

class longTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        FortranTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "long"
        self.typeStr  = "long"
        # 设置测试类型码为 "l"
        self.typeCode = "l"

######################################################################

class ulongTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        FortranTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "ulong"
        self.typeStr  = "ulong"
        # 设置测试类型码为 "L"
        self.typeCode = "L"

######################################################################

class longLongTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        FortranTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "longLong"
        self.typeStr  = "longLong"
        # 设置测试类型码为 "q"
        self.typeCode = "q"

######################################################################

class ulongLongTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        FortranTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "ulongLong"
        self.typeStr  = "ulongLong"
        # 设置测试类型码为 "Q"
        self.typeCode = "Q"

######################################################################

class floatTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        FortranTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "float"
        self.typeStr  = "float"
        # 设置测试类型码为 "f"
        self.typeCode = "f"

######################################################################

class doubleTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        FortranTestCase.__init__(self, methodName)
        # 设置测试类型字符串为 "double"
        self.typeStr  = "double"
        # 设置测试类型码为 "d"
        self.typeCode = "d"

######################################################################

if __name__ == "__main__":

    # 创建测试套件对象
    suite = unittest.TestSuite()
    # 将各个测试类添加到测试套件中
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

    # 执行测试套件中的测试
    print("Testing 2D Functions of Module Matrix")
    print("NumPy version", np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # 根据测试结果返回相应的退出状态码
    sys.exit(bool(result.errors + result.failures))
```