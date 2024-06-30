# `D:\src\scipysrc\scipy\scipy\special\_specfun.pyx`

```
# 导入libcpp库中的complex模块，并将其重命名为ccomplex
from libcpp.complex cimport complex as ccomplex

# 导入numpy库，并导入其数组接口模块
cimport numpy as cnp
# 调用numpy的import_array函数，初始化numpy数组接口

# 从特定路径下的头文件"special/airy.h"中，使用nogil标记，声明specfun_airyzo函数
cdef extern from "special/airy.h" nogil:
    void specfun_airyzo 'special::airyzo'(int nt, int kf, double *xa, double *xb, double *xc, double *xd)

# 从特定路径下的头文件"special/fresnel.h"中，使用nogil标记，声明specfun_fcszo函数
cdef extern from "special/fresnel.h" nogil:
    void specfun_fcszo 'special::fcszo'(int kf, int nt, ccomplex[double] *zo)

# 从特定路径下的头文件"special/kelvin.h"中，使用nogil标记，声明specfun_klvnzo函数
cdef extern from "special/kelvin.h" nogil:
    void specfun_klvnzo 'special::klvnzo'(int nt, int kd, double *zo)

# 从特定路径下的头文件"special/par_cyl.h"中，使用nogil标记，声明多个函数
cdef extern from "special/par_cyl.h" nogil:
    void specfun_pbdv 'special::detail::pbdv'(double x, double v, double *dv, double *dp, double *pdf, double *pdd)
    void specfun_pbvv 'special::detail::pbvv'(double x, double v, double *vv, double *vp, double *pvf, double *pvd)

# 从特定路径下的头文件"special/specfun/specfun.h"中，使用nogil标记，声明多个函数
cdef extern from "special/specfun/specfun.h" nogil:
    void specfun_bernob 'special::specfun::bernob'(int n, double *bn)
    void specfun_cerzo 'special::specfun::cerzo'(int nt, ccomplex[double] *zo)
    void specfun_cpbdn 'special::specfun::cpbdn'(int n, ccomplex[double] z, ccomplex[double] *cpb, ccomplex[double] *cpd)
    void specfun_cyzo 'special::specfun::cyzo'(int nt, int kf, int kc, ccomplex[double] *zo, ccomplex[double] *zv)
    void specfun_eulerb 'special::specfun::eulerb'(int n, double *en)
    void specfun_fcoef 'special::specfun::fcoef'(int kd, int m, double q, double a, double *fc)
    void specfun_jdzo 'special::specfun::jdzo'(int nt, double *zo, int *n, int *m, int *p)
    void specfun_jyzo 'special::specfun::jyzo'(int n, int nt, double *rj0, double *rj1, double *ry0, double *ry1)
    void specfun_lamn 'special::specfun::lamn'(int n, double x, int *nm, double *bl, double *dl)
    void specfun_lamv 'special::specfun::lamv'(double v, double x, double *vm, double *vl, double *dl)
    void specfun_lqnb 'special::specfun::lqnb'(int n, double x, double* qn, double* qd)
    void specfun_pbdv 'special::specfun::pbdv'(double x, double v, double *dv, double *dp, double *pdf, double *pdd)
    void specfun_pbvv 'special::specfun::pbvv'(double x, double v, double *vv, double *vp, double *pvf, double *pvd)
    void specfun_sdmn 'special::specfun::sdmn'(int m, int n, double c, double cv, double kd, double *df)
    void specfun_segv 'special::specfun::segv'(int m, int n, double c, int kd, double *cv, double *eg)

# 定义函数airyzo，计算Airy函数的前NT个零点及其相关值
def airyzo(int nt, int kf):
    """
    Compute the first NT zeros of Airy functions
    Ai(x) and Ai'(x), a and a', and the associated
    values of Ai(a') and Ai'(a); and the first NT
    zeros of Airy functions Bi(x) and Bi'(x), b and
    b', and the associated values of Bi(b') and
    Bi'(b).

    This is a wrapper for the function 'specfun_airyzo'.
    """
    # 定义C语言中的双精度指针变量
    cdef double *xxa
    cdef double *xxb
    cdef double *xxc
    cdef double *xxd
    # 定义numpy数组维度为1维的变量dims，长度为NT
    cdef cnp.npy_intp dims[1]
    dims[0] = nt

    # 创建numpy数组xa、xb、xc，初始化为零，数据类型为float64
    xa = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    xb = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    xc = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    # 创建一个形状为 dims 的双精度浮点类型的零数组 xd
    xd = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    
    # 获取数组 xa 的数据指针，并将其类型转换为 cnp.float64_t*
    xxa = <cnp.float64_t *>cnp.PyArray_DATA(xa)
    # 获取数组 xb 的数据指针，并将其类型转换为 cnp.float64_t*
    xxb = <cnp.float64_t *>cnp.PyArray_DATA(xb)
    # 获取数组 xc 的数据指针，并将其类型转换为 cnp.float64_t*
    xxc = <cnp.float64_t *>cnp.PyArray_DATA(xc)
    # 获取数组 xd 的数据指针，并将其类型转换为 cnp.float64_t*
    xxd = <cnp.float64_t *>cnp.PyArray_DATA(xd)
    
    # 调用 specfun_airyzo 函数，传入参数 nt, kf, xxa, xxb, xxc, xxd，对数组进行处理
    specfun_airyzo(nt, kf, xxa, xxb, xxc, xxd)
    
    # 返回更新后的数组 xa, xb, xc, xd
    return xa, xb, xc, xd
# 计算 Bernoulli 数 Bn，对应 n >= 2。这是函数 'specfun_bernob' 的包装器。
def bernob(int n):
    cdef double *bbn  # 定义指向 Bernoulli 数组的指针
    cdef cnp.npy_intp dims[1]  # 定义一个维度数组，长度为 n + 1
    dims[0] = n + 1
    bn = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)  # 创建一个类型为 float64 的全零数组
    bbn = <cnp.float64_t *>cnp.PyArray_DATA(bn)  # 将数组数据转换为 float64 指针类型
    specfun_bernob(n, bbn)  # 调用 specfun_bernob 函数计算 Bernoulli 数
    return bn  # 返回计算结果的数组


# 计算误差函数 erf(z) 的复根，使用修改的牛顿迭代法。这是函数 'specfun_cerzo' 的包装器。
def cerzo(int nt):
    cdef ccomplex[double] *zzo  # 定义指向复根数组的指针
    cdef cnp.npy_intp dims[1]  # 定义一个维度数组，长度为 nt
    dims[0] = nt
    zo = cnp.PyArray_ZEROS(1, dims, cnp.NPY_COMPLEX128, 0)  # 创建一个类型为 complex128 的全零数组
    zzo = <ccomplex[double] *>cnp.PyArray_DATA(zo)  # 将数组数据转换为 complex128 指针类型
    specfun_cerzo(nt, zzo)  # 调用 specfun_cerzo 函数计算复根
    return zo  # 返回计算结果的数组


# 计算复参数下的抛物线柱函数 Dn(z) 和 Dn'(z)。这是函数 'specfun_cpbdn' 的包装器。
def cpbdn(int n, ccomplex[double] z):
    cdef ccomplex[double] *ccpb  # 定义指向 Dn(z) 的指针
    cdef ccomplex[double] *ccpd  # 定义指向 Dn'(z) 的指针
    cdef cnp.npy_intp dims[1]  # 定义一个维度数组，长度为 |n| + 2
    dims[0] = abs(n) + 2

    cpb = cnp.PyArray_ZEROS(1, dims, cnp.NPY_COMPLEX128, 0)  # 创建一个类型为 complex128 的全零数组，存储 Dn(z)
    cpd = cnp.PyArray_ZEROS(1, dims, cnp.NPY_COMPLEX128, 0)  # 创建一个类型为 complex128 的全零数组，存储 Dn'(z)
    ccpb = <ccomplex[double] *>cnp.PyArray_DATA(cpb)  # 将数组数据转换为 complex128 指针类型
    ccpd = <ccomplex[double] *>cnp.PyArray_DATA(cpd)  # 将数组数据转换为 complex128 指针类型
    specfun_cpbdn(n, <ccomplex[double]> z, ccpb, ccpd)  # 调用 specfun_cpbdn 函数计算抛物线柱函数
    return cpb, cpd  # 返回计算结果的两个数组


# 计算 Y0(z)，Y1(z) 和 Y1'(z) 的复根，并返回这些根处的值。这是函数 'specfun_cyzo' 的包装器。
def cyzo(int nt, int kf, int kc):
    cdef ccomplex[double] *zzo  # 定义指向复根数组的指针
    cdef ccomplex[double] *zzv  # 定义指向复根处值的数组的指针
    cdef cnp.npy_intp dims[1]  # 定义一个维度数组，长度为 nt
    dims[0] = nt

    zo = cnp.PyArray_ZEROS(1, dims, cnp.NPY_COMPLEX128, 0)  # 创建一个类型为 complex128 的全零数组，存储复根
    zv = cnp.PyArray_ZEROS(1, dims, cnp.NPY_COMPLEX128, 0)  # 创建一个类型为 complex128 的全零数组，存储复根处值
    zzo = <ccomplex[double] *>cnp.PyArray_DATA(zo)  # 将数组数据转换为 complex128 指针类型
    zzv = <ccomplex[double] *>cnp.PyArray_DATA(zv)  # 将数组数据转换为 complex128 指针类型
    specfun_cyzo(nt, kf, kc, zzo, zzv)  # 调用 specfun_cyzo 函数计算复根及其对应值
    return zo, zv  # 返回计算结果的两个数组


# 计算 Euler 数 Bn，对应 n >= 2。这是函数 'specfun_eulerb' 的包装器。
def eulerb(int n):
    cdef double *een  # 定义指向 Euler 数组的指针
    cdef cnp.npy_intp dims[1]  # 定义一个维度数组，长度为 n + 1
    dims[0] = n + 1
    en = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)  # 创建一个类型为 float64 的全零数组
    een = <cnp.float64_t *>cnp.PyArray_DATA(en)  # 将数组数据转换为 float64 指针类型
    specfun_eulerb(n, een)  # 调用 specfun_eulerb 函数计算 Euler 数
    return en  # 返回计算结果的数组


# 计算 Mathieu 函数和修改的 Mathieu 函数的展开系数。
def fcoef(int kd, int m, double q, double a):
    cdef double *ffc  # 定义指向展开系数数组的指针

    cdef cnp.npy_intp dims[1]  # 定义一个维度数组，长度为 251
    dims[0] = 251

    fc = cnp.PyArray_SimpleNew(1, dims, cnp.NPY_FLOAT64)  # 创建一个类型为 float64 的新数组
    ffc = <cnp.float64_t *>cnp.PyArray_DATA(fc)  # 将数组数据转换为 float64 指针类型
    specfun_fcoef(kd, m, q, a, ffc)  # 调用 specfun_fcoef 函数计算展开系数
    return fc  # 返回计算结果的数组


# 计算 Fresnel 积分 C(z) 或 S(z) 的复根，使用修改的牛顿迭代法。这是函数 'specfun_fcszo' 的包装器。
def fcszo(int kf, int nt):
    """
    Compute the complex zeros of Fresnel integral C(z) or S(z) using
    modified Newton's iteration method. This is a wrapper for the
    function 'specfun_fcszo'.
    """
    # 定义一个指向复数数组的指针变量 zzo
    cdef ccomplex[double] *zzo
    # 定义一个包含一个元素的整数数组 dims
    cdef cnp.npy_intp dims[1]
    # 将 nt 赋值给 dims 数组的第一个元素，用于指定数组的维度大小
    dims[0] = nt
    # 调用 cnp.PyArray_ZEROS 函数创建一个包含 nt 个元素的复数数组 zo，
    # 并初始化所有元素为零，数组类型为 NPY_COMPLEX128
    zo = cnp.PyArray_ZEROS(1, dims, cnp.NPY_COMPLEX128, 0)
    # 将 zo 数组的数据指针转换为 ccomplex[double]* 类型，并赋值给 zzo 指针变量
    zzo = <ccomplex[double] *>cnp.PyArray_DATA(zo)
    # 调用 specfun_fcszo 函数，传递 kf、nt 和 zzo 作为参数
    specfun_fcszo(kf, nt, zzo)
    # 返回创建的复数数组 zo
    return zo
# 计算贝塞尔函数 Jn(x) 和 Jn'(x) 的零点，并按其大小顺序排列。
# 这是函数 'specfun_jdzo' 的包装器。nt 和所需数组大小之间的关系如下：
#     nt 在以下范围内    所需数组大小
#    -----------------------------------
#      0  -  100   ->    (nt + 10)
#    100  -  200   ->    (nt + 14)
#    200  -  300   ->    (nt + 16)
#    300  -  400   ->    (nt + 18)
#    400  -  500   ->    (nt + 21)
#    500  -  600   ->    (nt + 25)
#    600  -  700   ->    (nt + 11)
#    700  -  800   ->    (nt +  9)
#    800  -  900   ->    (nt +  9)
#    900  - 1000   ->    (nt + 10)
#   1000 - 1100   ->    (nt + 10)
#   1100 - 1200   ->    (nt + 11)
def jdzo(int nt):
    """
    Compute the zeros of Bessel functions Jn(x) and Jn'(x), and
    arrange them in the order of their magnitudes.

    This is a wrapper for the function 'specfun_jdzo'. The
    relationship between nt and the required array sizes is

         nt between     required array size
        -----------------------------------
          0  -  100   ->    (nt + 10)
        100  -  200   ->    (nt + 14)
        200  -  300   ->    (nt + 16)
        300  -  400   ->    (nt + 18)
        400  -  500   ->    (nt + 21)
        500  -  600   ->    (nt + 25)
        600  -  700   ->    (nt + 11)
        700  -  800   ->    (nt +  9)
        800  -  900   ->    (nt +  9)
        900  - 1000   ->    (nt + 10)
        1000 - 1100   ->    (nt + 10)
        1100 - 1200   ->    (nt + 11)

    It can be made a bit more granular but a generic +25 slack seems
    like an easy option instead of costly 1400-long arrays as Fortran
    code did originally, independent from the value of 'nt'.
    """

    cdef double *zzo
    cdef int *nn
    cdef int *mm
    cdef int *pp
    cdef cnp.npy_intp dims[1]

    # 设置数组的维度为 nt + 25
    dims[0] = nt + 25

    # 创建四个数组，用于存储计算结果
    zo = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    n = cnp.PyArray_ZEROS(1, dims, cnp.NPY_INT32, 0)
    m = cnp.PyArray_ZEROS(1, dims, cnp.NPY_INT32, 0)
    p = cnp.PyArray_ZEROS(1, dims, cnp.NPY_INT32, 0)

    # 获取数组的数据指针
    zzo = <cnp.float64_t *>cnp.PyArray_DATA(zo)
    nn = <int*>cnp.PyArray_DATA(n)
    mm = <int*>cnp.PyArray_DATA(m)
    pp = <int*>cnp.PyArray_DATA(p)

    # 调用 C 函数 specfun_jdzo 进行计算
    specfun_jdzo(nt, zzo, nn, mm, pp)

    # 返回计算结果的四个数组
    return n, m, p, zo


# 计算贝塞尔函数 Jn(x), Yn(x) 及其导数的零点。
# 这是函数 'specfun_jyzo' 的包装器。
def jyzo(int n, int nt):
    """
    Compute the zeros of Bessel functions Jn(x), Yn(x), and their
    derivatives. This is a wrapper for the function 'specfun_jyzo'.
    """
    cdef double *rrj0
    cdef double *rrj1
    cdef double *rry0
    cdef double *rry1
    cdef cnp.npy_intp dims[1]

    # 设置数组的维度为 nt
    dims[0] = nt

    # 创建四个数组，用于存储计算结果
    rj0 = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    rj1 = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    ry0 = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    ry1 = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)

    # 获取数组的数据指针
    rrj0 = <cnp.float64_t *>cnp.PyArray_DATA(rj0)
    rrj1 = <cnp.float64_t *>cnp.PyArray_DATA(rj1)
    rry0 = <cnp.float64_t *>cnp.PyArray_DATA(ry0)
    rry1 = <cnp.float64_t *>cnp.PyArray_DATA(ry1)

    # 调用 C 函数 specfun_jyzo 进行计算
    specfun_jyzo(n, nt, rrj0, rrj1, rry0, rry1)

    # 返回计算结果的四个数组
    return rj0, rj1, ry0, ry1


# 计算开尔文函数的零点。
# 这是函数 'specfun_klvnzo' 的包装器。
def klvnzo(int nt, int kd):
    """
    Compute the zeros of Kelvin functions. This is a wrapper for
    the function 'specfun_klvnzo'.
    """
    cdef double *zzo
    cdef cnp.npy_intp dims[1]

    # 设置数组的维度为 nt
    dims[0] = nt

    # 创建一个数组，用于存储计算结果
    zo = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)

    # 获取数组的数据指针
    zzo = <cnp.float64_t *>cnp.PyArray_DATA(zo)

    # 调用 C 函数 specfun_klvnzo 进行计算
    specfun_klvnzo(nt, kd, zzo)

    # 返回计算结果的数组
    return zo


# 计算 lambda 函数及其导数。
# 这是函数 'specfun_lamn' 的包装器。
def lamn(int n, double x):
    """
    Compute lambda functions and their derivatives. This is a wrapper
    for the function 'specfun_lamn'.
    """
    cdef int nm
    cdef double *bbl
    cdef double *ddl
    cdef cnp.npy_intp dims[1]
    # 将 dims 列表中的第一个元素设为 n+1，修改数组的第一个维度大小
    dims[0] = n + 1

    # 创建一个元素类型为 NPY_FLOAT64 的一维数组 bl，用于存储浮点数，初始化为零
    bl = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)

    # 创建一个元素类型为 NPY_FLOAT64 的一维数组 dl，用于存储浮点数，初始化为零
    dl = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)

    # 获取 bl 数组的数据指针，并转换为 cnp.float64_t 类型指针 bbl
    bbl = <cnp.float64_t *>cnp.PyArray_DATA(bl)

    # 获取 dl 数组的数据指针，并转换为 cnp.float64_t 类型指针 ddl
    ddl = <cnp.float64_t *>cnp.PyArray_DATA(dl)

    # 调用 specfun_lamn 函数，计算特定参数下的结果，并将结果写入 bbl 和 ddl 指向的内存中
    specfun_lamn(n, x, &nm, bbl, ddl)

    # 返回 nm（计算结果）、bl（包含计算结果的数组）、dl（包含计算结果的数组）
    return nm, bl, dl
def lamv(double v, double x):
    """
    Compute lambda function with arbitrary order v, and their derivative.
    This is a wrapper for the function 'specfun_lamv'.
    """
    # 声明双精度变量
    cdef double vm
    # 声明双精度指针
    cdef double *vvl
    cdef double *ddl
    # 定义一个整型数组，用于存储维度信息
    cdef cnp.npy_intp dims[1]
    dims[0] = int(v) + 1

    # 创建一个大小为 dims 的零数组，数据类型为 NPY_FLOAT64
    vl = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    dl = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    # 将数组数据转换为双精度指针
    vvl = <cnp.float64_t *>cnp.PyArray_DATA(vl)
    ddl = <cnp.float64_t *>cnp.PyArray_DATA(dl)
    # 调用 specfun_lamv 函数计算 lambda 函数及其导数
    specfun_lamv(v, x, &vm, vvl, ddl)
    # 返回结果 vm, vl, dl
    return vm, vl, dl


def pbdv(double v, double x):
    # 声明双精度变量
    cdef double pdf
    cdef double pdd
    cdef double *ddv
    cdef double *ddp
    # 定义一个整型数组，用于存储维度信息
    cdef cnp.npy_intp dims[1]
    dims[0] = abs(<int>v) + 2

    # 创建一个大小为 dims 的零数组，数据类型为 NPY_FLOAT64
    dv = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    dp = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    # 将数组数据转换为双精度指针
    ddv = <cnp.float64_t *>cnp.PyArray_DATA(dv)
    ddp = <cnp.float64_t *>cnp.PyArray_DATA(dp)
    # 调用 specfun_pbdv 函数计算 pbdv
    specfun_pbdv(x, v, ddv, ddp, &pdf, &pdd)
    # 返回结果 dv, dp, pdf, pdd
    return dv, dp, pdf, pdd


def pbvv(double v, double x):
    # 声明双精度变量
    cdef double pvf
    cdef double pvd
    cdef double *dvv
    cdef double *dvp
    # 定义一个整型数组，用于存储维度信息
    cdef cnp.npy_intp dims[1]
    dims[0] = abs(<int>v) + 2

    # 创建一个大小为 dims 的零数组，数据类型为 NPY_FLOAT64
    vv = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    vp = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    # 将数组数据转换为双精度指针
    dvv = <cnp.float64_t *>cnp.PyArray_DATA(vv)
    dvp = <cnp.float64_t *>cnp.PyArray_DATA(vp)
    # 调用 specfun_pbvv 函数计算 pbvv
    specfun_pbvv(x, v, dvv, dvp, &pvf, &pvd)
    # 返回结果 vv, vp, pvf, pvd
    return vv, vp, pvf, pvd


def sdmn(int m, int n, double c, double cv, int kd):
    """
    Compute the expansion coefficients of the prolate and oblate
    spheroidal functions, dk. This is a wrapper for the function
    'specfun_sdmn'.
    """
    # 声明双精度指针
    cdef double *ddf
    # 计算数组的大小
    cdef int nm = 25 + (int)(0.5 * (n - m) + c);
    cdef cnp.npy_intp dims[1]
    dims[0] = nm + 1

    # 创建一个大小为 dims 的零数组，数据类型为 NPY_FLOAT64
    df = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    # 将数组数据转换为双精度指针
    ddf = <cnp.float64_t *>cnp.PyArray_DATA(df)
    # 调用 specfun_sdmn 函数计算 sdmn
    specfun_sdmn(m, n, c, cv, kd, ddf)
    # 返回结果 df
    return df


def segv(int m, int n, double c, int kd):
    """
    Compute the characteristic values of spheroidal wave functions.
    This is a wrapper for the function 'specfun_segv'.
    """
    # 声明双精度变量
    cdef double cv
    cdef double *eeg
    cdef cnp.npy_intp dims[1]
    dims[0] = n - m + 1

    # 创建一个大小为 dims 的零数组，数据类型为 NPY_FLOAT64
    eg = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 0)
    # 将数组数据转换为双精度指针
    eeg = <cnp.float64_t *>cnp.PyArray_DATA(eg)
    # 调用 specfun_segv 函数计算 segv
    specfun_segv(m, n, c, kd, &cv, eeg)
    # 返回结果 cv, eg
    return cv, eg
```