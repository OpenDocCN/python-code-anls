# `D:\src\scipysrc\scipy\scipy\special\_faddeeva.cxx`

```
#include "_faddeeva.h"
// 引入外部头文件 "_faddeeva.h"，假设这是一个实现特殊函数 Faddeeva 的头文件

#include <complex>
// 引入复数运算库，用于处理复数类型和计算

#include <cmath>
// 引入数学函数库，包括常数和数学函数

using namespace std;
// 使用标准命名空间，避免每次使用标准库时都需要加前缀 std::

extern "C" {
// 使用 C 语言风格的外部函数接口，确保编译器正确识别函数名和参数

npy_cdouble faddeeva_w(npy_cdouble zp)
{
    // 将输入的复数转换为 C++ 中的复数类型
    complex<double> z(npy_creal(zp), npy_cimag(zp));
    // 调用 Faddeeva 类的 w 函数，计算 Faddeeva 函数 w(z)
    std::complex<double> w = Faddeeva::w(z);
    // 将计算结果封装成 numpy 复数返回给调用者
    return npy_cpack(real(w), imag(w));
}

npy_cdouble faddeeva_erf(npy_cdouble zp)
{
    // 将输入的复数转换为 C++ 中的复数类型
    complex<double> z(npy_creal(zp), npy_cimag(zp));
    // 调用 Faddeeva 类的 erf 函数，计算 Faddeeva 函数 erf(z)
    complex<double> w = Faddeeva::erf(z);
    // 将计算结果封装成 numpy 复数返回给调用者
    return npy_cpack(real(w), imag(w));
}

double faddeeva_erfc(double x)
{
    // 调用 Faddeeva 类的 erfc 函数，计算 Faddeeva 函数 erfc(x)
    return Faddeeva::erfc(x);
}

npy_cdouble faddeeva_erfc_complex(npy_cdouble zp)
{
    // 将输入的复数转换为 C++ 中的复数类型
    complex<double> z(npy_creal(zp), npy_cimag(zp));
    // 调用 Faddeeva 类的 erfc 函数，计算 Faddeeva 函数 erfc(z)
    complex<double> w = Faddeeva::erfc(z);
    // 将计算结果封装成 numpy 复数返回给调用者
    return npy_cpack(real(w), imag(w));
}

double faddeeva_erfcx(double x)
{
    // 调用 Faddeeva 类的 erfcx 函数，计算 Faddeeva 函数 erfcx(x)
    return Faddeeva::erfcx(x);
}

npy_cdouble faddeeva_erfcx_complex(npy_cdouble zp)
{
    // 将输入的复数转换为 C++ 中的复数类型
    complex<double> z(npy_creal(zp), npy_cimag(zp));
    // 调用 Faddeeva 类的 erfcx 函数，计算 Faddeeva 函数 erfcx(z)
    complex<double> w = Faddeeva::erfcx(z);
    // 将计算结果封装成 numpy 复数返回给调用者
    return npy_cpack(real(w), imag(w));
}

double faddeeva_erfi(double x)
{
    // 调用 Faddeeva 类的 erfi 函数，计算 Faddeeva 函数 erfi(x)
    return Faddeeva::erfi(x);
}

npy_cdouble faddeeva_erfi_complex(npy_cdouble zp)
{
    // 将输入的复数转换为 C++ 中的复数类型
    complex<double> z(npy_creal(zp), npy_cimag(zp));
    // 调用 Faddeeva 类的 erfi 函数，计算 Faddeeva 函数 erfi(z)
    complex<double> w = Faddeeva::erfi(z);
    // 将计算结果封装成 numpy 复数返回给调用者
    return npy_cpack(real(w), imag(w));
}

double faddeeva_dawsn(double x)
{
    // 调用 Faddeeva 类的 Dawson 函数，计算 Faddeeva 函数 Dawson(x)
    return Faddeeva::Dawson(x);
}

npy_cdouble faddeeva_dawsn_complex(npy_cdouble zp)
{
    // 将输入的复数转换为 C++ 中的复数类型
    complex<double> z(npy_creal(zp), npy_cimag(zp));
    // 调用 Faddeeva 类的 Dawson 函数，计算 Faddeeva 函数 Dawson(z)
    complex<double> w = Faddeeva::Dawson(z);
    // 将计算结果封装成 numpy 复数返回给调用者
    return npy_cpack(real(w), imag(w));
}

/*
 * A wrapper for a normal CDF for complex argument
 */
// 为复数参数的标准正态分布函数的包装器

npy_cdouble faddeeva_ndtr(npy_cdouble zp)
{
    // 将输入的复数转换为 C++ 中的复数类型
    complex<double> z(npy_creal(zp), npy_cimag(zp));
    // 对 z 进行缩放
    z *= M_SQRT1_2;
    // 调用 Faddeeva 类的 erfc 函数，计算 -z 的 Faddeeva 函数 erfc(-z)，并乘以 0.5
    complex<double> w = 0.5 * Faddeeva::erfc(-z);
    // 将计算结果封装成 numpy 复数返回给调用者
    return npy_cpack(real(w), imag(w));
}

/*
 * Log of the CDF of the normal distribution for double x.
 *
 * Let F(x) be the CDF of the standard normal distribution.
 * This implementation of log(F(x)) is based on the identities
 *
 *   F(x) = erfc(-x/√2)/2
 *        = 1 - erfc(x/√2)/2
 *
 * We use the first formula for x < -1, with erfc(z) replaced
 * by erfcx(z)*exp(-z**2) to ensure high precision for large
 * negative values when we take the logarithm:
 *
 *   log F(x) = log(erfcx(-x/√2)/2) - x**2/2
 *
 * For x >= -1, we use the second formula for F(x):
 *
 *   log F(x) = log1p(-erfc(x/√2)/2)
 */
// 对于双精度数 x 的标准正态分布函数的对数。

double faddeeva_log_ndtr(double x)
{
    // 对 x 进行缩放
    double t = x * M_SQRT1_2;
    if (x < -1.0) {
        // 使用第一个公式计算，对于 x < -1，使用 erfcx(-t) 替换 erfc(-t) 来确保计算精度
        return log(faddeeva_erfcx(-t) / 2) - t * t;
    }
    else {
        // 使用第二个公式计算，对于 x >= -1，使用 erfc(x) 计算
        return log1p(-faddeeva_erfc(t) / 2);
    }
}

/*
 * Log of the normal CDF for complex arguments.
 *
 * This is equivalent to log(ndtr(z)), but is more robust to overflow at $z\to\infty$.
 * This implementation uses the Faddeva computation, $\erfc(z) = \exp(-z^2) w(iz)$,
 * taking special care to select the principal branch of the log function
 *           log( exp(-z^2) w(i z) )
 */
// 对于复数参数的正态分布函数的对数。
npy_cdouble faddeeva_log_ndtr_complex(npy_cdouble zp)
{
    // 将复数转换为 C++ 标准库中的复数对象
    complex<double> z(npy_creal(zp), npy_cimag(zp));

    if (npy_creal(zp) > 6) {
        // 当实部大于6时，发生下溢。在接近实轴的情况下，使用 log(1 - ndtr(-z)) 展开对数。
        complex<double> w = -0.5 * Faddeeva::erfc(z*M_SQRT1_2);
        if (abs(w) < 1e-8) {
            // 如果 w 的模小于1e-8，则返回其实部和虚部构成的复数
            return npy_cpack(real(w), imag(w));
        }
    }

    // 对 z 应用变换
    z *= -M_SQRT1_2;
    double x = real(z), y = imag(z);

    /* 计算 $log(exp(-z^2))$ 的主分支，利用 $log(e^t) = log|e^t| + i Arg(e^t)$ 的性质，
     * 其中如果 $t = r + is$，那么 $e^t = e^r (\cos(s) + i \sin(s))$。
     */
    double mRe_z2 = (y - x) * (x + y); // Re(-z^2)，注意溢出情况
    double mIm_z2 = -2*x*y; // Im(-z^2)

    // 计算 mIm_z2 对 2.0*M_PI 取模的结果
    double im = fmod(mIm_z2, 2.0*M_PI);
    if (im > M_PI) {im -= 2.0*M_PI;}

    // 构造复数 val1，使用 mRe_z2 和 im 作为其实部和虚部
    complex<double> val1 = complex<double>(mRe_z2, im);

    // 计算复数 val2，其为 log(Faddeeva::w(complex<double>(-y, x))) 的结果
    complex<double> val2 = log(Faddeeva::w(complex<double>(-y, x)));

    // 计算最终结果，加上 val1、val2 并减去 NPY_LOGE2
    complex<double> result = val1 + val2 - NPY_LOGE2;

    /* 再次选择主分支：log(z) = log|z| + i arg(z)，因此结果的虚部应该在 [-pi, pi] 范围内。*/
    im = imag(result);
    if (im >= M_PI){ im -= 2*M_PI; }
    if (im < -M_PI){ im += 2*M_PI; }

    // 返回复数结果，使用 npy_cpack 函数将其打包成 npy_cdouble 类型
    return npy_cpack(real(result), im);
}

double faddeeva_voigt_profile(double x, double sigma, double gamma)
{
    const double INV_SQRT_2 = 0.707106781186547524401;
    const double SQRT_2PI = 2.5066282746310002416123552393401042;

    if(sigma == 0){
        if (gamma == 0){
            if (std::isnan(x))
                return x;
            if (x == 0)
                return INFINITY;
            // 若 x 不为 NaN 且不为零，则返回 0
            return 0;
        }
        // 计算 Voigt 分布在 sigma 为零时的值
        return gamma / M_PI / (x*x + gamma*gamma);
    }
    if (gamma == 0){
        // 计算 Voigt 分布在 gamma 为零时的值
        return 1 / SQRT_2PI / sigma * exp(-(x/sigma)*(x/sigma) / 2);
    }

    // 计算 Voigt 分布的一般情况下的值，使用复数 z 进行计算
    double zreal = x / sigma * INV_SQRT_2;
    double zimag = gamma / sigma * INV_SQRT_2;
    std::complex<double> z(zreal, zimag);
    std::complex<double> w = Faddeeva::w(z);

    // 返回 Voigt 分布的实部值
    return real(w) / sigma / SQRT_2PI;
}
```