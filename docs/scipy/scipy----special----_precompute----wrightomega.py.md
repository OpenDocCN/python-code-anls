# `D:\src\scipysrc\scipy\scipy\special\_precompute\wrightomega.py`

```
# 导入 NumPy 库，通常用于科学计算中的数组操作和数学函数
import numpy as np

# 尝试导入 mpmath 库，这是一个用于高精度数学计算的库
try:
    import mpmath
# 如果导入失败（抛出 ImportError 异常），则什么也不做，继续执行后续代码
except ImportError:
    pass


# 定义一个函数 mpmath_wrightomega，计算 Wright Omega 函数的值
def mpmath_wrightomega(x):
    return mpmath.lambertw(mpmath.exp(x), mpmath.mpf('-0.5'))


# 定义一个函数 wrightomega_series_error，计算级数逼近误差
def wrightomega_series_error(x):
    # 将 x 赋值给 series
    series = x
    # 计算 mpmath_wrightomega(x) 的值作为期望值
    desired = mpmath_wrightomega(x)
    # 返回级数逼近误差的绝对值与期望值之比
    return abs(series - desired) / desired


# 定义一个函数 wrightomega_exp_error，计算指数逼近误差
def wrightomega_exp_error(x):
    # 计算 mpmath.exp(x) 的值作为指数逼近
    exponential_approx = mpmath.exp(x)
    # 计算 mpmath_wrightomega(x) 的值作为期望值
    desired = mpmath_wrightomega(x)
    # 返回指数逼近误差的绝对值与期望值之比
    return abs(exponential_approx - desired) / desired


# 主函数 main，用于执行主要的计算和输出
def main():
    # 计算所需误差的设定值，使用 NumPy 提供的浮点数精度（eps）
    desired_error = 2 * np.finfo(float).eps
    # 输出级数逼近误差计算结果
    print('Series Error')
    # 遍历一组 x 值进行计算和输出
    for x in [1e5, 1e10, 1e15, 1e20]:
        # 在 mpmath 中设置工作精度为 100 位小数
        with mpmath.workdps(100):
            # 调用 wrightomega_series_error 计算误差
            error = wrightomega_series_error(x)
        # 输出 x 值、误差及是否小于设定的期望误差
        print(x, error, error < desired_error)

    # 输出指数逼近误差计算结果
    print('Exp error')
    # 遍历另一组 x 值进行计算和输出
    for x in [-10, -25, -50, -100, -200, -400, -700, -740]:
        # 在 mpmath 中设置工作精度为 100 位小数
        with mpmath.workdps(100):
            # 调用 wrightomega_exp_error 计算误差
            error = wrightomega_exp_error(x)
        # 输出 x 值、误差及是否小于设定的期望误差
        print(x, error, error < desired_error)


# 如果当前脚本作为主程序运行，则执行主函数 main
if __name__ == '__main__':
    main()
```