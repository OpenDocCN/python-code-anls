# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\testing.pyi`

```
# 导入 NumPy 库，并引入 NumPy 类型注解
import numpy as np
import numpy.typing as npt

# 声明 AR_U 变量，类型为 NumPy 字符串数组
AR_U: npt.NDArray[np.str_]

# 声明一个空函数 func，返回类型为布尔值
def func() -> bool: ...

# 使用 NumPy.testing.assert_ 进行断言，预期为 True，但给出错误信息 1
np.testing.assert_(True, msg=1)  # E: incompatible type

# 使用 NumPy.testing.build_err_msg 构建错误消息，给出错误代码 1 和消息 "test"
np.testing.build_err_msg(1, "test")  # E: incompatible type

# 使用 NumPy.testing.assert_almost_equal 进行近似相等断言，比较 AR_U 与 AR_U
np.testing.assert_almost_equal(AR_U, AR_U)  # E: incompatible type

# 使用 NumPy.testing.assert_approx_equal 进行近似相等断言，比较两个列表 [1, 2, 3] 与 [1, 2, 3]
np.testing.assert_approx_equal([1, 2, 3], [1, 2, 3])  # E: incompatible type

# 使用 NumPy.testing.assert_array_almost_equal 进行数组近似相等断言，比较 AR_U 与 AR_U
np.testing.assert_array_almost_equal(AR_U, AR_U)  # E: incompatible type

# 使用 NumPy.testing.assert_array_less 进行数组大小比较断言，比较 AR_U 与 AR_U
np.testing.assert_array_less(AR_U, AR_U)  # E: incompatible type

# 使用 NumPy.testing.assert_string_equal 进行字符串相等断言，比较字节串 b"a" 与 b"a"
np.testing.assert_string_equal(b"a", b"a")  # E: incompatible type

# 使用 NumPy.testing.assert_raises 断言函数 func 调用会引发 TypeError 异常
np.testing.assert_raises(expected_exception=TypeError, callable=func)  # E: No overload variant

# 使用 NumPy.testing.assert_raises_regex 断言函数 func 调用会引发 TypeError 异常，且异常消息匹配 "T"
np.testing.assert_raises_regex(expected_exception=TypeError, expected_regex="T", callable=func)  # E: No overload variant

# 使用 NumPy.testing.assert_allclose 进行全部近似相等断言，比较 AR_U 与 AR_U
np.testing.assert_allclose(AR_U, AR_U)  # E: incompatible type

# 使用 NumPy.testing.assert_array_almost_equal_nulp 进行数组近似相等断言，比较 AR_U 与 AR_U
np.testing.assert_array_almost_equal_nulp(AR_U, AR_U)  # E: incompatible type

# 使用 NumPy.testing.assert_array_max_ulp 进行数组最大误差比较断言，比较 AR_U 与 AR_U
np.testing.assert_array_max_ulp(AR_U, AR_U)  # E: incompatible type

# 使用 NumPy.testing.assert_warns 断言函数 func 调用会产生 RuntimeWarning 警告
np.testing.assert_warns(warning_class=RuntimeWarning, func=func)  # E: No overload variant

# 使用 NumPy.testing.assert_no_warnings 断言函数 func 调用不会产生任何警告
np.testing.assert_no_warnings(func=func)  # E: No overload variant

# 使用 NumPy.testing.assert_no_warnings 断言函数 func 调用不会产生任何警告，但给出了多余的参数 None
np.testing.assert_no_warnings(func, None)  # E: Too many arguments

# 使用 NumPy.testing.assert_no_warnings 断言函数 func 调用不会产生任何警告，但提供了未预期的关键字参数 test=None
np.testing.assert_no_warnings(func, test=None)  # E: Unexpected keyword argument

# 使用 NumPy.testing.assert_no_gc_cycles 断言函数 func 调用期间没有垃圾回收循环
np.testing.assert_no_gc_cycles(func=func)  # E: No overload variant
```