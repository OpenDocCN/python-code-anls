# `D:\src\scipysrc\scipy\scipy\special\_convex_analysis.pxd`

```
# 导入需要的 C 库函数和常量
from libc.math cimport log, fabs, expm1, log1p, isnan, NAN, INFINITY
from libc.float cimport DBL_MIN
# 导入 Cython 编译器
import cython

# 定义计算信息熵的内联函数
cdef inline double entr(double x) noexcept nogil:
    # 如果 x 是 NaN，则返回 x
    if isnan(x):
        return x
    # 如果 x 大于 0，则计算并返回 -x * log(x)
    elif x > 0:
        return -x * log(x)
    # 如果 x 等于 0，则返回 0
    elif x == 0:
        return 0
    # 其他情况返回负无穷大
    else:
        return -INFINITY

# 定义计算 KL 散度的内联函数
cdef inline double kl_div(double x, double y) noexcept nogil:
    # 如果 x 或 y 是 NaN，则返回 NaN
    if isnan(x) or isnan(y):
        return NAN
    # 如果 x 和 y 都大于 0，则计算并返回 x * log(x / y) - x + y
    elif x > 0 and y > 0:
        return x * log(x / y) - x + y
    # 如果 x 等于 0 且 y 大于等于 0，则返回 y
    elif x == 0 and y >= 0:
        return y
    # 其他情况返回正无穷大
    else:
        return INFINITY

# 定义计算相对熵的内联函数
@cython.cdivision(True)
cdef inline double rel_entr(double x, double y) noexcept nogil:
    cdef double ratio
    # 如果 x 或 y 是 NaN，则返回 NaN
    if isnan(x) or isnan(y):
        return NAN
    # 如果 x 或 y 小于等于 0，则根据条件返回对应的值
    if x <= 0 or y <= 0:
        if x == 0 and y >= 0:
            return 0
        return INFINITY
    # 计算 x 与 y 的比率
    ratio = x / y
    # 如果 ratio 在 0.5 到 2 之间，使用更精确的计算方式
    if 0.5 < ratio < 2:
        # 当 x 和 y 接近时，这种计算方式更为精确
        return x * log1p((x - y) / y)
    # 如果 ratio 处于 DBL_MIN 到 INFINITY 之间，则没有下溢/上溢问题
    if DBL_MIN < ratio < INFINITY:
        # 进行常规的对数计算
        return x * log(ratio)
    # 如果 x 和 y 相差太大，直接计算对数
    # 以避免下溢、上溢或次正常数的问题
    return x * (log(x) - log(y))

# 定义 Huber 损失函数的内联函数
cdef inline double huber(double delta, double r) noexcept nogil:
    # 如果 delta 小于 0，则返回正无穷大
    if delta < 0:
        return INFINITY
    # 如果 r 的绝对值小于等于 delta，则返回 0.5 * r * r
    elif fabs(r) <= delta:
        return 0.5 * r * r;
    # 其他情况返回 delta * (fabs(r) - 0.5 * delta)
    else:
        return delta * (fabs(r) - 0.5 * delta);

# 定义 Pseudo-Huber 损失函数的内联函数
cdef inline double pseudo_huber(double delta, double r) noexcept nogil:
    cdef double u, v
    # 如果 delta 小于 0，则返回正无穷大
    if delta < 0:
        return INFINITY
    # 如果 delta 或 r 等于 0，则返回 0
    elif delta == 0 or r == 0:
        return 0
    else:
        # 计算 u 和 v 的值
        u = delta
        v = r / delta
        # 根据公式计算 Pseudo-Huber 损失
        # 使用优化的方式来保持小 v 的精度
        return u*u*expm1(0.5*log1p(v*v))
```