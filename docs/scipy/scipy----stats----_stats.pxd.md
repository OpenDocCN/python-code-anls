# `D:\src\scipysrc\scipy\scipy\stats\_stats.pxd`

```
# 在一个低级调用中使用的函数，计算逆高斯分布的概率密度函数
cdef double _geninvgauss_pdf(double x, void *user_data) noexcept nogil

# 在一个低级调用中使用的函数，计算学生化范围的累积分布函数
cdef double _studentized_range_cdf(int n, double[2] x, void *user_data) noexcept nogil

# 在一个低级调用中使用的函数，计算学生化范围的渐近累积分布函数
cdef double _studentized_range_cdf_asymptotic(double z, void *user_data) noexcept nogil

# 在一个低级调用中使用的函数，计算学生化范围的概率密度函数
cdef double _studentized_range_pdf(int n, double[2] x, void *user_data) noexcept nogil

# 在一个低级调用中使用的函数，计算学生化范围的渐近概率密度函数
cdef double _studentized_range_pdf_asymptotic(double z, void *user_data) noexcept nogil

# 在一个低级调用中使用的函数，计算学生化范围的矩
cdef double _studentized_range_moment(int n, double[3] x_arg, void *user_data) noexcept nogil

# 在一个低级调用中使用的函数，计算广义双曲线分布的概率密度函数
cdef double _genhyperbolic_pdf(double x, void *user_data) noexcept nogil

# 在一个低级调用中使用的函数，计算广义双曲线分布的对数概率密度函数
cdef double _genhyperbolic_logpdf(double x, void *user_data) noexcept nogil
```