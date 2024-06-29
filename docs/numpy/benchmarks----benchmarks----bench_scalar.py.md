# `.\numpy\benchmarks\benchmarks\bench_scalar.py`

```
# 从共享库中导入 Benchmark 类和 TYPES1 常量
from .common import Benchmark, TYPES1

# 导入 numpy 库，并将其命名为 np
import numpy as np

# 定义 ScalarMath 类，继承自 Benchmark 类
class ScalarMath(Benchmark):
    # 测试标量数学运算，每个测试会多次运行以抵消函数调用的开销
    params = [TYPES1]  # 参数列表，包含 TYPES1 常量
    param_names = ["type"]  # 参数名列表，包含 "type"

    # 设置函数，在每次测试前执行
    def setup(self, typename):
        # 使用给定的 typename 创建一个 numpy 数据类型对象，并初始化为 2
        self.num = np.dtype(typename).type(2)
        # 创建一个 np.int32 类型的对象，并初始化为 2
        self.int32 = np.int32(2)
        # 创建一个包含单个元素为 2 的 np.int32 数组
        self.int32arr = np.array(2, dtype=np.int32)

    # 测试函数：加法运算
    def time_addition(self, typename):
        # 将 self.num 赋值给 n
        n = self.num
        # 进行连续的加法操作
        res = n + n + n + n + n + n + n + n + n + n

    # 测试函数：加法运算（包含 Python 整数）
    def time_addition_pyint(self, typename):
        # 将 self.num 赋值给 n
        n = self.num
        # 进行连续的加法操作，其中包含 Python 整数
        res = n + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1

    # 测试函数：乘法运算
    def time_multiplication(self, typename):
        # 将 self.num 赋值给 n
        n = self.num
        # 进行连续的乘法操作
        res = n * n * n * n * n * n * n * n * n * n

    # 测试函数：计算平方数
    def time_power_of_two(self, typename):
        # 将 self.num 赋值给 n
        n = self.num
        # 计算多个 n 的平方数
        res = n**2, n**2, n**2, n**2, n**2, n**2, n**2, n**2, n**2, n**2

    # 测试函数：计算绝对值
    def time_abs(self, typename):
        # 将 self.num 赋值给 n
        n = self.num
        # 进行多次绝对值运算
        res = abs(abs(abs(abs(abs(abs(abs(abs(abs(abs(n))))))))))

    # 测试函数：int32 类型与其他数值相加
    def time_add_int32_other(self, typename):
        # 一些混合情况的测试，有些快，有些慢，这里记录了它们的差异。
        # 当编写时，如果结果类型是输入之一，则速度较快。
        # 将 self.int32 赋值给 int32
        int32 = self.int32
        # 将 self.num 赋值给 other
        other = self.num
        # 执行多次 int32 和 other 的加法运算
        int32 + other
        int32 + other
        int32 + other
        int32 + other
        int32 + other

    # 测试函数：int32arr 数组与其他数值相加
    def time_add_int32arr_and_other(self, typename):
        # `arr + scalar` 会触发正常的 ufunc（数组）路径。
        # 将 self.int32arr 赋值给 int32
        int32 = self.int32arr
        # 将 self.num 赋值给 other
        other = self.num
        # 执行多次 int32arr 和 other 的加法运算
        int32 + other
        int32 + other
        int32 + other
        int32 + other
        int32 + other

    # 测试函数：其他数值与 int32arr 数组相加
    def time_add_other_and_int32arr(self, typename):
        # `scalar + arr` 在某些情况下会触发标量路径，这些路径可以更容易优化
        # 将 self.int32arr 赋值给 int32
        int32 = self.int32arr
        # 将 self.num 赋值给 other
        other = self.num
        # 执行多次 other 和 int32arr 的加法运算
        other + int32
        other + int32
        other + int32
        other + int32
        other + int32


# 定义 ScalarStr 类，继承自 Benchmark 类
class ScalarStr(Benchmark):
    # 测试标量到字符串的转换
    params = [TYPES1]  # 参数列表，包含 TYPES1 常量
    param_names = ["type"]  # 参数名列表，包含 "type"

    # 设置函数，在每次测试前执行
    def setup(self, typename):
        # 创建一个包含 100 个值为 100 的数组，数据类型为 typename
        self.a = np.array([100] * 100, dtype=typename)

    # 测试函数：执行数组元素的字符串表示
    def time_str_repr(self, typename):
        # 对数组 self.a 中的每个元素执行字符串表示操作
        res = [str(x) for x in self.a]
```