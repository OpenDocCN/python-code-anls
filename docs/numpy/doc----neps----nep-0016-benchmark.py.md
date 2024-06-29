# `.\numpy\doc\neps\nep-0016-benchmark.py`

```
# 导入性能测试模块 perf
import perf
# 导入抽象基类模块 abc
import abc
# 导入 NumPy 库并使用 np 别名
import numpy as np

# 定义一个名为 NotArray 的空类
class NotArray:
    pass

# 定义一个名为 AttrArray 的类，并设置类属性 __array_implementer__ 为 True
class AttrArray:
    __array_implementer__ = True

# 定义一个名为 ArrayBase 的抽象基类
class ArrayBase(abc.ABC):
    pass

# 定义一个名为 ABCArray1 的类，继承自 ArrayBase 抽象基类
class ABCArray1(ArrayBase):
    pass

# 定义一个名为 ABCArray2 的类，没有明确指定继承关系
class ABCArray2:
    pass

# 将 ABCArray2 注册为 ArrayBase 的虚拟子类
ArrayBase.register(ABCArray2)

# 创建 NotArray 类的实例
not_array = NotArray()
# 创建 AttrArray 类的实例
attr_array = AttrArray()
# 创建 ABCArray1 类的实例
abc_array_1 = ABCArray1()
# 创建 ABCArray2 类的实例
abc_array_2 = ABCArray2()

# 确保抽象基类 ABC 的缓存被预先加载

# 测试 isinstance 函数，检查 not_array 是否为 ArrayBase 的实例
isinstance(not_array, ArrayBase)
# 测试 isinstance 函数，检查 abc_array_1 是否为 ArrayBase 的实例
isinstance(abc_array_1, ArrayBase)
# 测试 isinstance 函数，检查 abc_array_2 是否为 ArrayBase 的实例

# 创建性能测试的 Runner 对象
runner = perf.Runner()

# 定义函数 t，用于执行性能测试并记录时间
def t(name, statement):
    runner.timeit(name, statement, globals=globals())

# 测试 np.asarray([]) 的性能
t("np.asarray([])", "np.asarray([])")
# 创建一个空 NumPy 数组 arrobj
arrobj = np.array([])
# 测试 np.asarray(arrobj) 的性能
t("np.asarray(arrobj)", "np.asarray(arrobj)")

# 测试 getattr 函数，获取 not_array 的 '__array_implementer__' 属性，如果不存在返回 False
t("attr, False", "getattr(not_array, '__array_implementer__', False)")
# 测试 getattr 函数，获取 attr_array 的 '__array_implementer__' 属性，如果不存在返回 False
t("attr, True", "getattr(attr_array, '__array_implementer__', False)")

# 测试 isinstance 函数，检查 not_array 是否为 ArrayBase 的实例，预期结果为 False
t("ABC, False", "isinstance(not_array, ArrayBase)")
# 测试 isinstance 函数，检查 abc_array_1 是否为 ArrayBase 的实例，预期结果为 True（通过继承实现）
t("ABC, True, via inheritance", "isinstance(abc_array_1, ArrayBase)")
# 测试 isinstance 函数，检查 abc_array_2 是否为 ArrayBase 的实例，预期结果为 True（通过注册实现）
t("ABC, True, via register", "isinstance(abc_array_2, ArrayBase)")
```