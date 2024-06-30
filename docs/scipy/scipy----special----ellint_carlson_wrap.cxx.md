# `D:\src\scipysrc\scipy\scipy\special\ellint_carlson_wrap.cxx`

```
// 包含自定义头文件 "ellint_carlson_wrap.hh" 和 "sf_error.h"
#include "ellint_carlson_wrap.hh"
#include "sf_error.h"

// 静态常量定义，用于指定椭圆积分的相对误差
static constexpr double ellip_rerr = 5e-16;

// 声明一个 C 风格的外部函数
extern "C" {

// 实数参数的完全椭圆积分 RC 函数
double fellint_RC(double x, double y)
{
    sf_error_t status; // 定义 sf_error_t 类型的 status 变量
    double res; // 定义存储结果的 double 类型变量

    // 调用 ellint_carlson 命名空间中的 rc 函数进行计算，存储错误状态到 status
    status = static_cast<sf_error_t>(ellint_carlson::rc(x, y, ellip_rerr, res));
    // 输出椭圆积分函数的错误信息
    sf_error("elliprc (real)", status, NULL);
    // 返回计算结果
    return res;
}

// 复数参数的完全椭圆积分 RC 函数
npy_cdouble cellint_RC(npy_cdouble x, npy_cdouble y)
{
    sf_error_t status; // 定义 sf_error_t 类型的 status 变量
    // 将 npy_cdouble 转换为 std::complex<double> 类型
    std::complex<double> xx{npy_creal(x), npy_cimag(x)};
    std::complex<double> yy{npy_creal(y), npy_cimag(y)};
    std::complex<double> res; // 定义存储结果的复数变量

    // 调用 ellint_carlson 命名空间中的 rc 函数进行计算，存储错误状态到 status
    status = static_cast<sf_error_t>(ellint_carlson::rc(xx, yy, ellip_rerr, res));
    // 输出椭圆积分函数的错误信息
    sf_error("elliprc (complex)", status, NULL);
    // 返回计算结果，将 std::complex<double> 转换为 npy_cdouble 类型
    return npy_cpack(res.real(), res.imag());
}

// 实数参数的第一类完全椭圆积分 RD 函数
double fellint_RD(double x, double y, double z)
{
    sf_error_t status; // 定义 sf_error_t 类型的 status 变量
    double res; // 定义存储结果的 double 类型变量

    // 调用 ellint_carlson 命名空间中的 rd 函数进行计算，存储错误状态到 status
    status = static_cast<sf_error_t>(ellint_carlson::rd(x, y, z, ellip_rerr, res));
    // 输出椭圆积分函数的错误信息
    sf_error("elliprd (real)", status, NULL);
    // 返回计算结果
    return res;
}

// 复数参数的第一类完全椭圆积分 RD 函数
npy_cdouble cellint_RD(npy_cdouble x, npy_cdouble y, npy_cdouble z)
{
    sf_error_t status; // 定义 sf_error_t 类型的 status 变量
    // 将 npy_cdouble 转换为 std::complex<double> 类型
    std::complex<double> xx{npy_creal(x), npy_cimag(x)};
    std::complex<double> yy{npy_creal(y), npy_cimag(y)};
    std::complex<double> zz{npy_creal(z), npy_cimag(z)};
    std::complex<double> res; // 定义存储结果的复数变量

    // 调用 ellint_carlson 命名空间中的 rd 函数进行计算，存储错误状态到 status
    status = static_cast<sf_error_t>(ellint_carlson::rd(xx, yy, zz, ellip_rerr, res));
    // 输出椭圆积分函数的错误信息
    sf_error("elliprd (complex)", status, NULL);
    // 返回计算结果，将 std::complex<double> 转换为 npy_cdouble 类型
    return npy_cpack(res.real(), res.imag());
}

// 实数参数的第一类完全椭圆积分 RF 函数
double fellint_RF(double x, double y, double z)
{
    sf_error_t status; // 定义 sf_error_t 类型的 status 变量
    double res; // 定义存储结果的 double 类型变量

    // 调用 ellint_carlson 命名空间中的 rf 函数进行计算，存储错误状态到 status
    status = static_cast<sf_error_t>(ellint_carlson::rf(x, y, z, ellip_rerr, res));
    // 输出椭圆积分函数的错误信息
    sf_error("elliprf (real)", status, NULL);
    // 返回计算结果
    return res;
}

// 复数参数的第一类完全椭圆积分 RF 函数
npy_cdouble cellint_RF(npy_cdouble x, npy_cdouble y, npy_cdouble z)
{
    sf_error_t status; // 定义 sf_error_t 类型的 status 变量
    // 将 npy_cdouble 转换为 std::complex<double> 类型
    std::complex<double> xx{npy_creal(x), npy_cimag(x)};
    std::complex<double> yy{npy_creal(y), npy_cimag(y)};
    std::complex<double> zz{npy_creal(z), npy_cimag(z)};
    std::complex<double> res; // 定义存储结果的复数变量

    // 调用 ellint_carlson 命名空间中的 rf 函数进行计算，存储错误状态到 status
    status = static_cast<sf_error_t>(ellint_carlson::rf(xx, yy, zz, ellip_rerr, res));
    // 输出椭圆积分函数的错误信息
    sf_error("elliprf (complex)", status, NULL);
    // 返回计算结果，将 std::complex<double> 转换为 npy_cdouble 类型
    return npy_cpack(res.real(), res.imag());
}

// 实数参数的第一类完全椭圆积分 RG 函数
double fellint_RG(double x, double y, double z)
{
    sf_error_t status; // 定义 sf_error_t 类型的 status 变量
    double res; // 定义存储结果的 double 类型变量

    // 调用 ellint_carlson 命名空间中的 rg 函数进行计算，存储错误状态到 status
    status = static_cast<sf_error_t>(ellint_carlson::rg(x, y, z, ellip_rerr, res));
    // 输出椭圆积分函数的错误信息
    sf_error("elliprg (real)", status, NULL);
    // 返回计算结果
    return res;
}

// 复数参数的第一类完全椭圆积分 RG 函数
npy_cdouble cellint_RG(npy_cdouble x, npy_cdouble y, npy_cdouble z)
{
    sf_error_t status; // 定义 sf_error_t 类型的 status 变量
    // 将 npy_cdouble 转换为 std::complex<double> 类型
    std::complex<double> xx{npy_creal(x), npy_cimag(x)};
    std::complex<double> yy{npy_creal(y), npy_cimag(y)};
    std::complex<double> zz{npy_creal(z), npy_cimag(z)};
    std::complex<double> res; // 定义存储结果的复数变量
    // 将输入的复数 y 转换为 C++ 的复数类型
    std::complex<double> yy{npy_creal(y), npy_cimag(y)};
    
    // 将输入的复数 z 转换为 C++ 的复数类型
    std::complex<double> zz{npy_creal(z), npy_cimag(z)};
    
    // 定义一个变量 res，用于存储计算结果的复数
    std::complex<double> res;

    // 调用 ellint_carlson 命名空间中的 rg 函数计算 Carlson's elliptic integral rg
    // 使用输入的 xx, yy, zz 参数进行计算，同时传递 ellip_rerr 作为误差容限
    // 将计算结果存储在 res 中，同时将返回的状态值转换为 sf_error_t 类型并存储在 status 变量中
    status = static_cast<sf_error_t>(ellint_carlson::rg(xx, yy, zz,
                                                        ellip_rerr, res));
    
    // 打印名为 "elliprg (complex)" 的函数调用的状态信息
    sf_error("elliprg (complex)", status, NULL);
    
    // 将复数 res 的实部和虚部打包成一个复数对象并返回
    return npy_cpack(res.real(), res.imag());
// 结束当前代码块，用于表示 C/C++ 中的函数定义或其他代码块的结束
}

// 定义名为 `fellint_RJ` 的函数，计算实数参数的 RJ 椭圆积分
double fellint_RJ(double x, double y, double z, double p)
{
    // 定义 `status` 变量，用于存储函数调用返回的错误状态
    sf_error_t status;
    // 定义 `res` 变量，用于存储函数调用返回的结果值
    double res;

    // 调用 `ellint_carlson::rj` 函数计算 RJ 椭圆积分，返回错误状态
    status = static_cast<sf_error_t>(ellint_carlson::rj(x, y, z, p,
                                                        ellip_rerr, res));
    // 输出与椭圆积分相关的错误信息
    sf_error("elliprj (real)", status, NULL);
    // 返回计算得到的实数结果
    return res;
}

// 定义名为 `cellint_RJ` 的函数，计算复数参数的 RJ 椭圆积分
npy_cdouble cellint_RJ(npy_cdouble x, npy_cdouble y, npy_cdouble z, npy_cdouble p)
{
    // 定义 `status` 变量，用于存储函数调用返回的错误状态
    sf_error_t status;
    // 将复数参数转换为 C++ 的复数类型
    std::complex<double> xx{npy_creal(x), npy_cimag(x)};
    std::complex<double> yy{npy_creal(y), npy_cimag(y)};
    std::complex<double> zz{npy_creal(z), npy_cimag(z)};
    std::complex<double> pp{npy_creal(p), npy_cimag(p)};
    // 定义 `res` 变量，用于存储函数调用返回的复数结果
    std::complex<double> res;

    // 调用 `ellint_carlson::rj` 函数计算 RJ 椭圆积分，返回错误状态
    status = static_cast<sf_error_t>(ellint_carlson::rj(xx, yy, zz, pp,
                                                        ellip_rerr, res));
    // 输出与椭圆积分相关的错误信息
    sf_error("elliprj (complex)", status, NULL);
    // 返回计算得到的复数结果，转换为 NumPy 复数结构
    return npy_cpack(res.real(), res.imag());
}

// 结束 C/C++ 函数定义的外部链接
}  // extern "C"
```