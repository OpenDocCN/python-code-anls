# `D:\src\scipysrc\scipy\scipy\special\_comb.pyx`

```
# 从 "limits.h" 头文件中导入 ULONG_MAX 常量
cdef extern from "limits.h":
    unsigned long ULONG_MAX

# 定义 _comb_int 函数，计算组合数 C(N, k)
def _comb_int(N, k):
    # 使用机器整数进行快速计算
    try:
        # 调用 _comb_int_long 函数计算组合数
        r = _comb_int_long(N, k)
        # 如果结果不为零，直接返回结果
        if r != 0:
            return r
    # 捕获溢出错误或类型错误异常
    except (OverflowError, TypeError):
        pass

    # 如果出现异常或结果为零，则使用回退方法
    # 将 N 和 k 转换为整数类型
    N = int(N)
    k = int(k)

    # 如果 k 大于 N 或者 N 或 k 小于零，则返回 0
    if k > N or N < 0 or k < 0:
        return 0

    # 计算 M 为 N + 1
    M = N + 1
    # 计算 nterms 为 k 和 N-k 中的较小值
    nterms = min(k, N - k)

    # 初始化 numerator 和 denominator 为 1
    numerator = 1
    denominator = 1
    # 计算组合数的分子和分母
    for j in range(1, nterms + 1):
        numerator *= M - j
        denominator *= j

    # 返回组合数结果，采用整数除法以保证结果为整数
    return numerator // denominator


# 定义 _comb_int_long 函数，使用机器长整数计算组合数
cdef unsigned long _comb_int_long(unsigned long N, unsigned long k) noexcept:
    """
    Compute binom(N, k) for integers.
    Returns 0 if error/overflow encountered.
    """
    cdef unsigned long val, j, M, nterms

    # 如果 k 大于 N 或者 N 等于 ULONG_MAX，直接返回 0
    if k > N or N == ULONG_MAX:
        return 0

    # 计算 M 为 N + 1
    M = N + 1
    # 计算 nterms 为 k 和 N-k 中的较小值
    nterms = min(k, N - k)

    # 初始化 val 为 1
    val = 1

    # 计算组合数的值
    for j in range(1, nterms + 1):
        # 检查是否会发生溢出
        if val > ULONG_MAX // (M - j):
            return 0

        val *= M - j
        val //= j

    # 返回组合数的计算结果
    return val
```