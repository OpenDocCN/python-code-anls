# `D:\src\scipysrc\scipy\scipy\special\_test_internal.pyx`

```
# 设置 Cython 编译器指令，启用 cpow 功能
# 导入必要的模块和函数，包括 numpy 和 numpy.testing 中的 assert_
import numpy as np
from numpy.testing import assert_

# 导入需要使用的 C 函数和常量定义
from libc.math cimport isnan

# 从 _round.h 中导入 C 函数和常量定义
cdef extern from "_round.h":
    double add_round_up(double, double) nogil  # 导入 add_round_up 函数声明
    double add_round_down(double, double) nogil  # 导入 add_round_down 函数声明
    int fesetround(int) nogil  # 导入 fesetround 函数声明
    int fegetround() nogil  # 导入 fegetround 函数声明
    int FE_UPWARD  # 导入 FE_UPWARD 常量定义
    int FE_DOWNWARD  # 导入 FE_DOWNWARD 常量定义

# 从 dd_real_wrappers.h 中导入 C 函数声明
cdef extern from "dd_real_wrappers.h" nogil:
    cdef struct double2:  # 定义 double2 结构体
        double hi  # 结构体成员 hi，双精度浮点数
        double lo  # 结构体成员 lo，双精度浮点数
    double2 dd_create(double a, double b)  # 导入 dd_create 函数声明
    double2 dd_exp(const double2* x)  # 导入 dd_exp 函数声明
    double2 dd_log(const double2* x)  # 导入 dd_log 函数声明

# 检查是否支持浮点环境处理
def have_fenv():
    old_round = fegetround()  # 获取当前的浮点舍入模式
    have_getround = True if old_round >= 0 else False  # 检查是否成功获取舍入模式
    if not have_getround:
        return False

    have_setround = True
    try:
        # 尝试设置舍入模式为向上和向下，并检查是否成功
        if fesetround(FE_UPWARD) != 0:
            have_setround = False
        if fesetround(FE_DOWNWARD) != 0:
            have_setround = False
    finally:
        fesetround(old_round)  # 恢复之前的舍入模式
    return have_setround  # 返回是否成功设置浮点舍入模式的布尔值

# 生成随机双精度浮点数数组的函数
def random_double(size):
    # 生成随机整数数组，转换为无符号 16 位整数，再转换为双精度浮点数
    x = np.random.randint(low=0, high=2**16, size=4*size)
    return x.astype(np.uint16).view(np.float64)  # 返回双精度浮点数视图

# 测试函数，用于测试 add_round_up 和 add_round_down 函数
def test_add_round(size, mode):
    cdef:
        int i, old_round, status  # 定义循环计数器、旧的舍入模式和状态变量
        double[:] sample1 = random_double(size)  # 创建第一个随机双精度浮点数数组
        double[:] sample2 = random_double(size)  # 创建第二个随机双精度浮点数数组
        double res, std  # 定义结果和标准结果变量

    nfail = 0  # 记录失败测试数量的计数器
    msg = []  # 存储失败消息的列表
    for i in range(size):  # 循环处理每个元素
        old_round = fegetround()  # 获取当前的浮点舍入模式
        if old_round < 0:  # 如果获取失败，抛出运行时错误
            raise RuntimeError("Couldn't get rounding mode")
        try:
            if mode == 'up':  # 根据模式选择使用向上或向下舍入函数
                res = add_round_up(sample1[i], sample2[i])
                status = fesetround(FE_UPWARD)  # 设置舍入模式为向上
            elif mode == 'down':
                res = add_round_down(sample1[i], sample2[i])
                status = fesetround(FE_DOWNWARD)  # 设置舍入模式为向下
            else:
                raise ValueError("Invalid rounding mode")  # 模式无效时抛出值错误
            if status != 0:  # 如果设置舍入模式失败，抛出运行时错误
                raise RuntimeError("Failed to set rounding mode")
            std = sample1[i] + sample2[i]  # 计算标准结果
        finally:
            fesetround(old_round)  # 恢复之前的浮点舍入模式
        if isnan(res) and isnan(std):  # 如果结果和标准结果都是 NaN，则继续下一个循环
            continue
        if res != std:  # 如果结果与标准结果不相等，记录失败的详细消息
            nfail += 1
            msg.append("{:.21g} + {:.21g} = {:.21g} != {:.21g}"
                       .format(sample1[i], sample2[i], std, res))  # 添加消息到列表
    # 如果存在失败的情况（nfail 不为零），则执行以下操作
    s = "{}/{} failures with mode {}.".format(nfail, size, mode)
    # 构建失败信息的字符串，包含失败数、总数和模式
    msg = [s] + msg
    # 将失败信息字符串和已有消息列表合并成新的消息列表
    assert_(False, "\n".join(msg))
    # 断言失败，并输出合并后的消息列表作为错误信息
# Python wrappers for a few of the "double-double" C functions defined
# in cephes/dd_*.  The wrappers are not part of the public API; they are
# for use in scipy.special unit tests only.

# 定义了 Python 封装函数，用于调用 C 库中的 "double-double" 函数。这些函数
# 不是公共 API 的一部分，仅用于 scipy.special 单元测试中。

def _dd_exp(double xhi, double xlo):
    # 将传入的两个 double 型参数 xhi 和 xlo 封装成 double2 结构
    cdef double2 x = dd_create(xhi, xlo)
    # 调用 C 库中的 dd_exp 函数对 x 进行指数函数计算
    cdef double2 y = dd_exp(&x)
    # 返回计算结果的高位和低位部分
    return y.hi, y.lo


def _dd_log(double xhi, double xlo):
    # 将传入的两个 double 型参数 xhi 和 xlo 封装成 double2 结构
    cdef double2 x = dd_create(xhi, xlo)
    # 调用 C 库中的 dd_log 函数对 x 进行对数函数计算
    cdef double2 y = dd_log(&x)
    # 返回计算结果的高位和低位部分
    return y.hi, y.lo
```