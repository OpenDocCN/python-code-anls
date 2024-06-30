# `D:\src\scipysrc\scipy\scipy\special\cython_special.pxd`

```
# 定义一个融合类型 `number_t`，可以是复数双精度或者双精度浮点数
ctypedef fused number_t:
    double complex
    double

# 定义用于求解球面贝塞尔函数的函数原型，接受整数 `n`、类型为 `number_t` 的参数 `z`，并可选地指定导数，默认不抛出异常，无 GIL
cpdef number_t spherical_jn(long n, number_t z, bint derivative=*) noexcept nogil
cpdef number_t spherical_yn(long n, number_t z, bint derivative=*) noexcept nogil
cpdef number_t spherical_in(long n, number_t z, bint derivative=*) noexcept nogil
cpdef number_t spherical_kn(long n, number_t z, bint derivative=*) noexcept nogil

# 定义另一个融合类型 `Dd_number_t`，可以是复数双精度或者双精度浮点数
ctypedef fused Dd_number_t:
    double complex
    double

# 定义融合类型 `df_number_t`，可以是单精度或双精度浮点数
ctypedef fused df_number_t:
    double
    float

# 定义融合类型 `dfg_number_t`，可以是单精度、双精度或长双精度浮点数
ctypedef fused dfg_number_t:
    double
    float
    long double

# 定义融合类型 `dl_number_t`，可以是双精度浮点数或长整数
ctypedef fused dl_number_t:
    double
    long

# 定义 Voigt 剖面的函数原型，接受双精度参数 `x0`, `x1`, `x2`，并可选地指定不抛出异常，无 GIL
cpdef double voigt_profile(double x0, double x1, double x2) noexcept nogil

# 定义算术-几何均值的函数原型，接受双精度参数 `x0`, `x1`，并可选地指定不抛出异常，无 GIL
cpdef double agm(double x0, double x1) noexcept nogil

# 定义 Airy 函数的 C 函数版本，接受双精度或复数双精度参数 `x0`，并分别在 `y0`, `y1`, `y2`, `y3` 中返回结果，无 GIL
cdef void airy(Dd_number_t x0, Dd_number_t *y0, Dd_number_t *y1, Dd_number_t *y2, Dd_number_t *y3) noexcept nogil

# 定义 Airy 函数的拓展版本，接受双精度或复数双精度参数 `x0`，并分别在 `y0`, `y1`, `y2`, `y3` 中返回结果，无 GIL
cdef void airye(Dd_number_t x0, Dd_number_t *y0, Dd_number_t *y1, Dd_number_t *y2, Dd_number_t *y3) noexcept nogil

# 定义二项分布累积分布函数的函数原型，接受双精度参数 `x0`, `x2` 和长整数参数 `x1`，并可选地指定不抛出异常，无 GIL
cpdef double bdtr(double x0, dl_number_t x1, double x2) noexcept nogil
cpdef double bdtrc(double x0, dl_number_t x1, double x2) noexcept nogil
cpdef double bdtri(double x0, dl_number_t x1, double x2) noexcept nogil
cpdef double bdtrik(double x0, double x1, double x2) noexcept nogil
cpdef double bdtrin(double x0, double x1, double x2) noexcept nogil

# 定义贝塞尔函数的扩展第一类的第二个类型的参数 `x0`，并可选地指定不抛出异常，无 GIL
cpdef double bei(double x0) noexcept nogil
cpdef double beip(double x0) noexcept nogil
cpdef double ber(double x0) noexcept nogil
cpdef double berp(double x0) noexcept nogil

# 定义贝塞尔多项式函数的函数原型，接受双精度参数 `x0`, `x1`, `x2`，并可选地指定不抛出异常，无 GIL
cpdef double besselpoly(double x0, double x1, double x2) noexcept nogil

# 定义贝塔函数的函数原型，接受双精度参数 `x0`, `x1`，并可选地指定不抛出异常，无 GIL
cpdef double beta(double x0, double x1) noexcept nogil

# 定义不完全贝塔函数的函数原型，接受双精度参数 `x0`, `x1`, `x2`，并可选地指定不抛出异常，无 GIL
cpdef df_number_t betainc(df_number_t x0, df_number_t x1, df_number_t x2) noexcept nogil
cpdef df_number_t betaincc(df_number_t x0, df_number_t x1, df_number_t x2) noexcept nogil
cpdef df_number_t betaincinv(df_number_t x0, df_number_t x1, df_number_t x2) noexcept nogil
cpdef df_number_t betainccinv(df_number_t x0, df_number_t x1, df_number_t x2) noexcept nogil

# 定义贝塔自然对数的函数原型，接受双精度参数 `x0`, `x1`，并可选地指定不抛出异常，无 GIL
cpdef double betaln(double x0, double x1) noexcept nogil

# 定义二项式分布的函数原型，接受双精度参数 `x0`, `x1`，并可选地指定不抛出异常，无 GIL
cpdef double binom(double x0, double x1) noexcept nogil

# 定义 Box-Cox 变换的函数原型，接受双精度参数 `x0`, `x1`，并可选地指定不抛出异常，无 GIL
cpdef double boxcox(double x0, double x1) noexcept nogil
cpdef double boxcox1p(double x0, double x1) noexcept nogil

# 定义贝塔分布累积分布函数的函数原型，接受双精度参数 `x0`, `x1`, `x2`，并可选地指定不抛出异常，无 GIL
cpdef double btdtr(double x0, double x1, double x2) noexcept nogil
cpdef double btdtri(double x0, double x1, double x2) noexcept nogil

# 定义贝塔分布的函数原型，接受双精度参数 `x0`, `x1`, `x2`，并可选地指定不抛出异常，无 GIL
cpdef double btdtria(double x0, double x1, double x2) noexcept nogil
cpdef double btdtrib(double x0, double x1, double x2) noexcept nogil

# 定义立方根函数的函数原型，接受双精度参数 `x0`，并可选地指定不抛出异常，无 GIL
cpdef double cbrt(double x0) noexcept nogil

# 定义卡方分布的函数原型，接受双精度参数 `x0`, `x1`，并可选地指定不抛出异常，无 GIL
cpdef double chdtr(double x0, double x1) noexcept nogil
cpdef double chdtrc(double x0, double x1) noexcept nogil
cpdef double chdtri(double x0, double x1) noexcept nogil
cpdef double chdtriv(double x0, double x1) noexcept nogil

# 定义非中心卡方分布的函数原型，接受双精度参数 `x0`, `x1`, `x2`，并可选地指定不抛出异常，无 GIL
cpdef double chndtr(double x0, double x1, double x2) noexcept nog
# 计算三个参数的阶乘之和并返回结果，使用Cython的功能，无异常处理，无GIL
cpdef double chndtrix(double x0, double x1, double x2) noexcept nogil

# 计算给定角度的余弦值，使用Cython的功能，无异常处理，无GIL
cpdef double cosdg(double x0) noexcept nogil

# 计算给定角度的余弦值减一，使用Cython的功能，无异常处理，无GIL
cpdef double cosm1(double x0) noexcept nogil

# 计算给定角度的余切值，使用Cython的功能，无异常处理，无GIL
cpdef double cotdg(double x0) noexcept nogil

# 计算Dawson函数的值，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t dawsn(Dd_number_t x0) noexcept nogil

# 计算椭圆积分第一类的完全椭圆积分值，使用Cython的功能，无异常处理，无GIL
cpdef double ellipe(double x0) noexcept nogil

# 计算椭圆积分第二类的不完全椭圆积分值，使用Cython的功能，无异常处理，无GIL
cpdef double ellipeinc(double x0, double x1) noexcept nogil

# 计算椭圆积分Jacobian椭圆函数的值，使用Cython的功能，无异常处理，无GIL
cdef void ellipj(double x0, double x1, double *y0, double *y1, double *y2, double *y3) noexcept nogil

# 计算椭圆积分第二类的不完全椭圆积分补充值，使用Cython的功能，无异常处理，无GIL
cpdef double ellipkinc(double x0, double x1) noexcept nogil

# 计算椭圆积分第一类的不完全椭圆积分补充值，使用Cython的功能，无异常处理，无GIL
cpdef double ellipkm1(double x0) noexcept nogil

# 计算椭圆积分第一类的完全椭圆积分值，使用Cython的功能，无异常处理，无GIL
cpdef double ellipk(double x0) noexcept nogil

# 计算椭圆积分第一类的完全椭圆积分RC函数的值，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t elliprc(Dd_number_t x0, Dd_number_t x1) noexcept nogil

# 计算椭圆积分第二类的不完全椭圆积分RD函数的值，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t elliprd(Dd_number_t x0, Dd_number_t x1, Dd_number_t x2) noexcept nogil

# 计算椭圆积分第一类的不完全椭圆积分RF函数的值，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t elliprf(Dd_number_t x0, Dd_number_t x1, Dd_number_t x2) noexcept nogil

# 计算椭圆积分第二类的不完全椭圆积分RG函数的值，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t elliprg(Dd_number_t x0, Dd_number_t x1, Dd_number_t x2) noexcept nogil

# 计算椭圆积分第一类的不完全椭圆积分RJ函数的值，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t elliprj(Dd_number_t x0, Dd_number_t x1, Dd_number_t x2, Dd_number_t x3) noexcept nogil

# 计算熵函数的值，使用Cython的功能，无异常处理，无GIL
cpdef double entr(double x0) noexcept nogil

# 计算误差函数的值，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t erf(Dd_number_t x0) noexcept nogil

# 计算互补误差函数的值，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t erfc(Dd_number_t x0) noexcept nogil

# 计算互补误差函数的值的复指数，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t erfcx(Dd_number_t x0) noexcept nogil

# 计算反误差函数的值，接受复数参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t erfi(Dd_number_t x0) noexcept nogil

# 计算误差函数的反函数的值，接受双精度浮点数参数，使用Cython的功能，无异常处理，无GIL
cpdef df_number_t erfinv(df_number_t x0) noexcept nogil

# 计算互补误差函数的反函数的值，接受双精度浮点数参数，使用Cython的功能，无异常处理，无GIL
cpdef double erfcinv(double x0) noexcept nogil

# 计算Chebyshev多项式的值，接受双精度和复双精度参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t eval_chebyc(dl_number_t x0, Dd_number_t x1) noexcept nogil

# 计算Chebyshev多项式的值，接受双精度和复双精度参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t eval_chebys(dl_number_t x0, Dd_number_t x1) noexcept nogil

# 计算Chebyshev多项式的值，接受双精度和复双精度参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t eval_chebyt(dl_number_t x0, Dd_number_t x1) noexcept nogil

# 计算Chebyshev多项式的值，接受双精度和复双精度参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t eval_chebyu(dl_number_t x0, Dd_number_t x1) noexcept nogil

# 计算Gegenbauer多项式的值，接受双精度和复双精度参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t eval_gegenbauer(dl_number_t x0, double x1, Dd_number_t x2) noexcept nogil

# 计算Generalized Laguerre多项式的值，接受双精度和复双精度参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t eval_genlaguerre(dl_number_t x0, double x1, Dd_number_t x2) noexcept nogil

# 计算Hermite多项式的值，接受长整型和双精度参数，使用Cython的功能，无异常处理，无GIL
cpdef double eval_hermite(long x0, double x1) noexcept nogil

# 计算Hermite归一化多项式的值，接受长整型和双精度参数，使用Cython的功能，无异常处理，无GIL
cpdef double eval_hermitenorm(long x0, double x1) noexcept nogil

# 计算Jacobi多项式的值，接受双精度和复双精度参数，使用Cython的功能，无异常处理，无GIL
cpdef Dd_number_t eval_jacobi(dl_number_t x0,
# 定义一个 C/C++ 扩展的函数，计算指数函数 exp(x1) 或 expm1(x1)
cpdef double expn(dl_number_t x0, double x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算相对误差 exp(x0) - 1
cpdef double exprel(double x0) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 F 分布的累积分布函数值
cpdef double fdtr(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 F 分布的补充累积分布函数值
cpdef double fdtrc(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 F 分布的分位数函数值
cpdef double fdtri(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 F 分布的逆差分函数值
cpdef double fdtridfd(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 Fresnel 积分函数的两个输出值
cdef void fresnel(Dd_number_t x0, Dd_number_t *y0, Dd_number_t *y1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算伽玛函数的值
cpdef Dd_number_t gamma(Dd_number_t x0) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算不完全伽玛函数的值
cpdef double gammainc(double x0, double x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算不完全伽玛函数的补充值
cpdef double gammaincc(double x0, double x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算不完全伽玛函数的补充值的逆函数
cpdef double gammainccinv(double x0, double x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算不完全伽玛函数的逆函数
cpdef double gammaincinv(double x0, double x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算伽玛函数的自然对数值
cpdef double gammaln(double x0) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算伽玛函数的符号值
cpdef double gammasgn(double x0) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 G 分布的累积分布函数值
cpdef double gdtr(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 G 分布的补充累积分布函数值
cpdef double gdtrc(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 G 分布的分位数函数值
cpdef double gdtria(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 G 分布的逆累积分布函数值
cpdef double gdtrib(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 G 分布的逆差分函数值
cpdef double gdtrix(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算汉克尔函数 H1(x1)
cpdef double complex hankel1(double x0, double complex x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正汉克尔函数 H1(x1)
cpdef double complex hankel1e(double x0, double complex x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算汉克尔函数 H2(x1)
cpdef double complex hankel2(double x0, double complex x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正汉克尔函数 H2(x1)
cpdef double complex hankel2e(double x0, double complex x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算 Huber 函数的值
cpdef double huber(double x0, double x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算超几何函数 0F1(x1)
cpdef Dd_number_t hyp0f1(double x0, Dd_number_t x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算超几何函数 1F1(x1, x2)
cpdef Dd_number_t hyp1f1(double x0, double x1, Dd_number_t x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算超几何函数 2F1(x1, x2, x3)
cpdef Dd_number_t hyp2f1(double x0, double x1, double x2, Dd_number_t x3) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算超几何函数 U(x1, x2)
cpdef double hyperu(double x0, double x1, double x2) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正贝塞尔函数 I0(x0)
cpdef double i0(double x0) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正贝塞尔函数 I0e(x0)
cpdef double i0e(double x0) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正贝塞尔函数 I1(x0)
cpdef double i1(double x0) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正贝塞尔函数 I1e(x0)
cpdef double i1e(double x0) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算逆 Box-Cox 变换的值
cpdef double inv_boxcox(double x0, double x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算逆 Box-Cox-1 变换的值
cpdef double inv_boxcox1p(double x0, double x1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正特殊函数 It2I0K0(x0)
cdef void it2i0k0(double x0, double *y0, double *y1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正特殊函数 It2J0Y0(x0)
cdef void it2j0y0(double x0, double *y0, double *y1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正斯特劳夫函数 It2Struve0(x0)
cpdef double it2struve0(double x0) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正艾里函数 ItAiry(x0)
cdef void itairy(double x0, double *y0, double *y1, double *y2, double *y3) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正贝塞尔函数 ItI0K0(x0)
cdef void iti0k0(double x0, double *y0, double *y1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正贝塞尔函数 ItJ0Y0(x0)
cdef void itj0y0(double x0, double *y0, double *y1) noexcept nogil
# 定义一个 C/C++ 扩展的函数，计算修正斯特劳夫函数 ItModStruve0(x0)
cpdef double itmod
# 定义函数 k0，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double k0(double x0) noexcept nogil

# 定义函数 k0e，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double k0e(double x0) noexcept nogil

# 定义函数 k1，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double k1(double x0) noexcept nogil

# 定义函数 k1e，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double k1e(double x0) noexcept nogil

# 定义函数 kei，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double kei(double x0) noexcept nogil

# 定义函数 keip，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double keip(double x0) noexcept nogil

# 定义函数 kelvin，计算输入参数 x0 的某个数学函数，将结果存储在复数指针 y0, y1, y2, y3 所指向的内存位置，无异常抛出，无需全局解释器锁 (GIL)
cdef void kelvin(double x0, double complex *y0, double complex *y1, double complex *y2, double complex *y3) noexcept nogil

# 定义函数 ker，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double ker(double x0) noexcept nogil

# 定义函数 kerp，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double kerp(double x0) noexcept nogil

# 定义函数 kl_div，计算输入参数 x0 和 x1 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double kl_div(double x0, double x1) noexcept nogil

# 定义函数 kn，计算输入参数 x0 和 x1 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double kn(dl_number_t x0, double x1) noexcept nogil

# 定义函数 kolmogi，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double kolmogi(double x0) noexcept nogil

# 定义函数 kolmogorov，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double kolmogorov(double x0) noexcept nogil

# 定义函数 kv，计算输入参数 x0 和 x1 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef Dd_number_t kv(double x0, Dd_number_t x1) noexcept nogil

# 定义函数 kve，计算输入参数 x0 和 x1 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef Dd_number_t kve(double x0, Dd_number_t x1) noexcept nogil

# 定义函数 log1p，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef Dd_number_t log1p(Dd_number_t x0) noexcept nogil

# 定义函数 log_expit，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef dfg_number_t log_expit(dfg_number_t x0) noexcept nogil

# 定义函数 log_ndtr，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef Dd_number_t log_ndtr(Dd_number_t x0) noexcept nogil

# 定义函数 loggamma，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef Dd_number_t loggamma(Dd_number_t x0) noexcept nogil

# 定义函数 logit，计算输入参数 x0 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef dfg_number_t logit(dfg_number_t x0) noexcept nogil

# 定义函数 lpmv，计算输入参数 x0, x1, x2 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double lpmv(double x0, double x1, double x2) noexcept nogil

# 定义函数 mathieu_a，计算输入参数 x0, x1 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double mathieu_a(double x0, double x1) noexcept nogil

# 定义函数 mathieu_b，计算输入参数 x0, x1 的某个数学函数并返回结果，无异常抛出，无需全局解释器锁 (GIL)
cpdef double mathieu_b(double x0, double x1) noexcept nogil

# 定义函数 mathieu_cem，计算输入参数 x0, x1, x2 的某个数学函数，将结果存储在指针 y0, y1 所指向的内存位置，无异常抛出，无需全局解释器锁 (GIL)
cdef void mathieu_cem(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil

# 定义函数 mathieu_modcem1，计算输入参数 x0, x1, x2 的某个数学函数，将结果存储在指针 y0, y1 所指向的内存位置，无异常抛出，无需全局解释器锁 (GIL)
cdef void mathieu_modcem1(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil

# 定义函数 mathieu_modcem2，计算输入参数 x0, x1, x2 的某个数学函数，将结果存储在指针 y0, y1 所指向的内存位置，无异常抛出，无需全局解释器锁 (GIL)
cdef void mathieu_modcem2(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil

# 定义函数 mathieu_modsem1，计算输入参数 x0, x1, x2 的某个数学函数，将结果存储在指针 y0, y1 所指向的内存位置，无异常抛出，无需全局解释器锁 (GIL)
cdef void mathieu_modsem1(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil

# 定义函数 mathieu_modsem2，计算输入参数 x0, x1, x2 的某个数学函数，将结果存储在指针 y0, y1 所指向的内存位置，无异常抛出，无需全局解释器锁 (GIL)
cdef void mathieu_modsem2(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil

# 定义函数 mathieu_sem，计算输入参数 x0, x1, x2 的某个数学函数，将结果
# 定义一个 CPython 函数，计算并返回 nctdtridf 的结果，参数为 x0, x1, x2，无异常抛出，无全局解释器锁（GIL）限制
cpdef double nctdtridf(double x0, double x1, double x2) noexcept nogil

# 定义一个 CPython 函数，计算并返回 nctdtrinc 的结果，参数为 x0, x1, x2，无异常抛出，无全局解释器锁（GIL）限制
cpdef double nctdtrinc(double x0, double x1, double x2) noexcept nogil

# 定义一个 CPython 函数，计算并返回 nctdtrit 的结果，参数为 x0, x1, x2，无异常抛出，无全局解释器锁（GIL）限制
cpdef double nctdtrit(double x0, double x1, double x2) noexcept nogil

# 定义一个 CPython 函数，计算并返回 ndtr 的结果，参数为 x0，无异常抛出，无全局解释器锁（GIL）限制
cpdef Dd_number_t ndtr(Dd_number_t x0) noexcept nogil

# 定义一个 CPython 函数，计算并返回 ndtri 的结果，参数为 x0，无异常抛出，无全局解释器锁（GIL）限制
cpdef double ndtri(double x0) noexcept nogil

# 定义一个 CPython 函数，计算并返回 nrdtrimn 的结果，参数为 x0, x1, x2，无异常抛出，无全局解释器锁（GIL）限制
cpdef double nrdtrimn(double x0, double x1, double x2) noexcept nogil

# 定义一个 CPython 函数，计算并返回 nrdtrisd 的结果，参数为 x0, x1, x2，无异常抛出，无全局解释器锁（GIL）限制
cpdef double nrdtrisd(double x0, double x1, double x2) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1, x2, x3，无异常抛出，无全局解释器锁（GIL）限制
cdef void obl_ang1(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1, x2, x3, x4，无异常抛出，无全局解释器锁（GIL）限制
cdef void obl_ang1_cv(double x0, double x1, double x2, double x3, double x4, double *y0, double *y1) noexcept nogil

# 定义一个 CPython 函数，计算并返回 obl_cv 的结果，参数为 x0, x1, x2，无异常抛出，无全局解释器锁（GIL）限制
cpdef double obl_cv(double x0, double x1, double x2) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1, x2, x3，无异常抛出，无全局解释器锁（GIL）限制
cdef void obl_rad1(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1, x2, x3, x4，无异常抛出，无全局解释器锁（GIL）限制
cdef void obl_rad1_cv(double x0, double x1, double x2, double x3, double x4, double *y0, double *y1) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1, x2, x3，无异常抛出，无全局解释器锁（GIL）限制
cdef void obl_rad2(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1, x2, x3, x4，无异常抛出，无全局解释器锁（GIL）限制
cdef void obl_rad2_cv(double x0, double x1, double x2, double x3, double x4, double *y0, double *y1) noexcept nogil

# 定义一个 CPython 函数，计算并返回 owens_t 的结果，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cpdef double owens_t(double x0, double x1) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cdef void pbdv(double x0, double x1, double *y0, double *y1) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cdef void pbvv(double x0, double x1, double *y0, double *y1) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cdef void pbwa(double x0, double x1, double *y0, double *y1) noexcept nogil

# 定义一个 CPython 函数，计算并返回 pdtr 的结果，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cpdef double pdtr(double x0, double x1) noexcept nogil

# 定义一个 CPython 函数，计算并返回 pdtrc 的结果，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cpdef double pdtrc(double x0, double x1) noexcept nogil

# 定义一个 CPython 函数，计算并返回 pdtri 的结果，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cpdef double pdtri(dl_number_t x0, double x1) noexcept nogil

# 定义一个 CPython 函数，计算并返回 pdtrik 的结果，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cpdef double pdtrik(double x0, double x1) noexcept nogil

# 定义一个 CPython 函数，计算并返回 poch 的结果，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cpdef double poch(double x0, double x1) noexcept nogil

# 定义一个 CPython 函数，计算并返回 powm1 的结果，参数为 x0, x1，无异常抛出，无全局解释器锁（GIL）限制
cpdef df_number_t powm1(df_number_t x0, df_number_t x1) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1, x2, x3，无异常抛出，无全局解释器锁（GIL）限制
cdef void pro_ang1(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil

# 定义一个 Cython 函数，计算并更新 y0 和 y1 的值，参数为 x0, x1, x2, x3, x4，无异常抛出，无全局解释器锁（GIL）限制
cdef void pro_ang1_cv(double x0, double x1, double x2, double x3, double
# 定义 C 函数 `sici`，接受一个 `Dd_number_t` 类型参数和两个 `Dd_number_t` 指针作为输出，不会释放全局解释器锁
cdef void sici(Dd_number_t x0, Dd_number_t *y0, Dd_number_t *y1) noexcept nogil

# 定义 C 函数 `sindg`，接受一个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double sindg(double x0) noexcept nogil

# 定义 C 函数 `smirnov`，接受一个 `dl_number_t` 和一个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double smirnov(dl_number_t x0, double x1) noexcept nogil

# 定义 C 函数 `smirnovi`，接受一个 `dl_number_t` 和一个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double smirnovi(dl_number_t x0, double x1) noexcept nogil

# 定义 C 函数 `spence`，接受一个 `Dd_number_t` 类型参数，返回 `Dd_number_t` 类型结果，不会释放全局解释器锁
cpdef Dd_number_t spence(Dd_number_t x0) noexcept nogil

# 定义 C 函数 `sph_harm`，接受两个 `dl_number_t` 类型参数和两个 `double` 类型参数，返回 `double complex` 类型结果，不会释放全局解释器锁
cpdef double complex sph_harm(dl_number_t x0, dl_number_t x1, double x2, double x3) noexcept nogil

# 定义 C 函数 `stdtr`，接受两个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double stdtr(double x0, double x1) noexcept nogil

# 定义 C 函数 `stdtridf`，接受两个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double stdtridf(double x0, double x1) noexcept nogil

# 定义 C 函数 `stdtrit`，接受两个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double stdtrit(double x0, double x1) noexcept nogil

# 定义 C 函数 `struve`，接受两个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double struve(double x0, double x1) noexcept nogil

# 定义 C 函数 `tandg`，接受一个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double tandg(double x0) noexcept nogil

# 定义 C 函数 `tklmbda`，接受两个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double tklmbda(double x0, double x1) noexcept nogil

# 定义 C 函数 `wofz`，接受一个 `double complex` 类型参数，返回 `double complex` 类型结果，不会释放全局解释器锁
cpdef double complex wofz(double complex x0) noexcept nogil

# 定义 C 函数 `wrightomega`，接受一个 `Dd_number_t` 类型参数，返回 `Dd_number_t` 类型结果，不会释放全局解释器锁
cpdef Dd_number_t wrightomega(Dd_number_t x0) noexcept nogil

# 定义 C 函数 `xlog1py`，接受两个 `Dd_number_t` 类型参数，返回 `Dd_number_t` 类型结果，不会释放全局解释器锁
cpdef Dd_number_t xlog1py(Dd_number_t x0, Dd_number_t x1) noexcept nogil

# 定义 C 函数 `xlogy`，接受两个 `Dd_number_t` 类型参数，返回 `Dd_number_t` 类型结果，不会释放全局解释器锁
cpdef Dd_number_t xlogy(Dd_number_t x0, Dd_number_t x1) noexcept nogil

# 定义 C 函数 `y0`，接受一个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double y0(double x0) noexcept nogil

# 定义 C 函数 `y1`，接受一个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double y1(double x0) noexcept nogil

# 定义 C 函数 `yn`，接受一个 `dl_number_t` 和一个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double yn(dl_number_t x0, double x1) noexcept nogil

# 定义 C 函数 `yv`，接受一个 `double` 和一个 `Dd_number_t` 类型参数，返回 `Dd_number_t` 类型结果，不会释放全局解释器锁
cpdef Dd_number_t yv(double x0, Dd_number_t x1) noexcept nogil

# 定义 C 函数 `yve`，接受一个 `double` 和一个 `Dd_number_t` 类型参数，返回 `Dd_number_t` 类型结果，不会释放全局解释器锁
cpdef Dd_number_t yve(double x0, Dd_number_t x1) noexcept nogil

# 定义 C 函数 `zetac`，接受一个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double zetac(double x0) noexcept nogil

# 定义 C 函数 `wright_bessel`，接受三个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double wright_bessel(double x0, double x1, double x2) noexcept nogil

# 定义 C 函数 `log_wright_bessel`，接受三个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double log_wright_bessel(double x0, double x1, double x2) noexcept nogil

# 定义 C 函数 `ndtri_exp`，接受一个 `double` 类型参数，返回 `double` 类型结果，不会释放全局解释器锁
cpdef double ndtri_exp(double x0) noexcept nogil
```