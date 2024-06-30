# `D:\src\scipysrc\scipy\scipy\special\special\specfun.h`

```
#pragma once
// 通过宏定义实现的特殊函数错误处理，当实数部分为1.0e300时，将错误类型设为溢出，并将实数部分设为正无穷大
#define SPECFUN_ZCONVINF(func, z)                                                                                      \
    do {                                                                                                               \
        if ((double) (z).real() == (double) 1.0e300) {                                                                 \
            set_error(func, SF_ERROR_OVERFLOW, NULL);                                                                  \
            (z).real(std::numeric_limits<double>::infinity());                                                         \
        }                                                                                                              \
        // 当实数部分为-1.0e300时，将错误类型设为溢出，并将实数部分设为负无穷大                                                          \
        if ((double) (z).real() == (double) -1.0e300) {                                                                \
            set_error(func, SF_ERROR_OVERFLOW, NULL);                                                                  \
            (z).real(-std::numeric_limits<double>::infinity());                                                        \
        }                                                                                                              \
    } while (0)
// 通过宏定义实现的特殊函数错误处理，当实数部分为1.0e300时，将错误类型设为溢出，并将实数设为正无穷大
#define SPECFUN_CONVINF(func, x)                                                                                       \
    do {                                                                                                               \
        if ((double) (x) == (double) 1.0e300) {                                                                        \
            set_error(func, SF_ERROR_OVERFLOW, NULL);                                                                  \
            (x) = std::numeric_limits<double>::infinity();                                                             \
        }                                                                                                              \
        // 当实数部分为-1.0e300时，将错误类型设为溢出，并将实数设为负无穷大
        if ((double) (x) == (double) -1.0e300) {                                                                       \
            set_error(func, SF_ERROR_OVERFLOW, NULL);                                                                  \
            (x) = -std::numeric_limits<double>::infinity();                                                            \
        }                                                                                                              \
    } while (0)

namespace special {

// 定义特殊函数 chyp2f1，接受四个参数 a, b, c 和 z，返回一个复数结果
inline std::complex<double> chyp2f1(double a, double b, double c, std::complex<double> z) {
    // 判断条件 l0 和 l1 用于特殊情况的错误处理和返回值设定
    int l0 = ((c == floor(c)) && (c < 0));
    int l1 = ((fabs(1 - z.real()) < 1e-15) && (z.imag() == 0) && (c - a - b <= 0));
    // 如果满足 l0 或 l1 中的任意一个条件，则设定错误类型为溢出，并返回正无穷大作为结果
    if (l0 || l1) {
        set_error("chyp2f1", SF_ERROR_OVERFLOW, NULL);
        return std::numeric_limits<double>::infinity();
    }

    // isfer 用于指示特殊情况
    int isfer = 0;
    // 调用 specfun 命名空间下的 hygfz 函数计算结果，并将 isfer 的指针传递给该函数
    std::complex<double> outz = specfun::hygfz(a, b, c, z, &isfer);
    # 如果 isfer 等于 3，表示出现溢出错误
    if (isfer == 3) {
        # 设置错误类型为 SF_ERROR_OVERFLOW
        set_error("chyp2f1", SF_ERROR_OVERFLOW, NULL);
        # 将 outz 设为正无穷大实部，虚部为 0.0
        outz.real(std::numeric_limits<double>::infinity());
        outz.imag(0.0);
    # 如果 isfer 等于 5，表示出现损失错误
    } else if (isfer == 5) {
        # 设置错误类型为 SF_ERROR_LOSS
        set_error("chyp2f1", SF_ERROR_LOSS, NULL);
    # 如果 isfer 不等于 0（且不等于 3 或 5），表示出现其他错误
    } else if (isfer != 0) {
        # 根据 isfer 的值设置具体的错误类型
        set_error("chyp2f1", static_cast<sf_error_t>(isfer), NULL);
        # 将 outz 设为 quiet NaN（非数值）
        outz.real(std::numeric_limits<double>::quiet_NaN());
        outz.imag(std::numeric_limits<double>::quiet_NaN());
    }
    # 返回计算结果 outz
    return outz;
}

// 定义了一系列用于特殊函数计算的函数，都在 special 命名空间中

inline std::complex<double> chyp1f1(double a, double b, std::complex<double> z) {
    // 调用 specfun::cchg 函数计算复数参数的超几何函数 1F1
    std::complex<double> outz = specfun::cchg(a, b, z);
    // 如果结果的实部为极大值 1e300，设置溢出错误并将实部设为正无穷大
    if (outz.real() == 1e300) {
        set_error("chyp1f1", SF_ERROR_OVERFLOW, NULL);
        outz.real(std::numeric_limits<double>::infinity());
    }

    return outz;
}

inline double hypu(double a, double b, double x) {
    double out;
    int md; // 方法代码，但不会返回
    int isfer = 0; // 用于记录特殊函数的返回状态

    // 调用 specfun::chgu 函数计算约化超几何函数 U
    out = specfun::chgu(x, a, b, &md, &isfer);
    // 如果结果为极大值 1e300，设置溢出错误并将结果设为正无穷大
    if (out == 1e300) {
        set_error("hypU", SF_ERROR_OVERFLOW, NULL);
        out = std::numeric_limits<double>::infinity();
    }
    // 根据 isfer 的值设置相应的错误状态
    if (isfer == 6) {
        set_error("hypU", SF_ERROR_NO_RESULT, NULL);
        out = std::numeric_limits<double>::quiet_NaN();
    } else if (isfer != 0) {
        set_error("hypU", static_cast<sf_error_t>(isfer), NULL);
        out = std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}

inline double hyp1f1(double a, double b, double x) {
    double outy;

    // 调用 specfun::chgm 函数计算约化超几何函数 1F1
    outy = specfun::chgm(x, a, b);
    return outy;
}

inline std::complex<double> cerf(std::complex<double> z) {
    // 调用 specfun::cerror 函数计算复数误差函数
    return specfun::cerror(z);
}

inline double pmv(double m, double v, double x) {
    int int_m;
    double out;

    // 如果 m 不是整数，返回 NaN
    if (m != floor(m)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // 将 m 转换为整数类型
    int_m = (int) m;
    // 调用 specfun::lpmv 函数计算关联勒让德函数 Pmv
    out = specfun::lpmv(x, int_m, v);
    // 处理特殊函数的无穷大和错误状态
    SPECFUN_CONVINF("pmv", out);
    return out;
}

} // namespace special
```