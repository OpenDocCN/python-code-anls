# `D:\src\scipysrc\scipy\scipy\special\special\iv_ratio.h`

```
#pragma once
// 只允许该头文件被编译一次

#include "config.h"
// 包含自定义的配置文件

#include "tools.h"
// 包含自定义的工具函数

#include "error.h"
// 包含自定义的错误处理函数

namespace special {

/* Generates the "tail" of Perron's continued fraction for `iv(v,x)/iv(v-1,x)`
 * for v >= 1 and x >= 0.
 *
 * The Perron continued fraction is studied in [1].  It is given by
 *
 *         iv(v, x)      x    -(2v+1)x   -(2v+3)x   -(2v+5)x
 *   R := --------- = ------ ---------- ---------- ---------- ...
 *        iv(v-1,x)   2v+x + 2(v+x)+1 + 2(v+x)+2 + 2(v+x)+3 +
 *
 * Rearrange the expression by making an equivalent transform to prevent
 * floating point overflow and extracting the first fraction to simplify
 * the recurrence relation.  This leads to
 *
 *        xc                -(2vc+c)(xc) -(2vc+3c)(xc) -(2vc+5c)(xc)
 *   R = -----,  fc = 2vc + ------------ ------------- ------------- ...
 *       xc+fc              2(vc+xc)+c + 2(vc+xc)+2c + 2(vc+xc)+3c +
 *
 * This class generates the fractions of fc after 2vc.
 *
 * [1] Gautschi, W. and Slavik, J. (1978). "On the computation of modified
 *     Bessel function ratios." Mathematics of Computation, 32(143):865-875.
 */
struct IvRatioCFTailGenerator {

    // It is assumed that v >= 1, x >= 0, c > 0, and all are finite.
    // 假设 v >= 1, x >= 0, c > 0，且均为有限数值

    IvRatioCFTailGenerator(double vc, double xc, double c) noexcept {
        // 初始化 Perron 连分数的尾部生成器
        a0_ = -(2*vc-c)*xc;
        as_ = -2*c*xc;
        b0_ = 2*(vc+xc);
        bs_ = c;
        k_ = 0;
    }

    std::pair<double, double> operator()() {
        // 返回一对数值，表示生成的下一个分数
        ++k_;
        return {std::fma(k_, as_, a0_), std::fma(k_, bs_, b0_)};
    }

private:
    double a0_, as_;  // a[k] == a0 + as*k, k >= 1
    double b0_, bs_;  // b[k] == b0 + bs*k, k >= 1
    std::uint64_t k_; // current index
};

SPECFUN_HOST_DEVICE inline double iv_ratio(double v, double x) {

    if (std::isnan(v) || std::isnan(x)) {
        // 如果 v 或 x 是 NaN，则返回 NaN
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (v < 1 || x < 0) {
        // 如果 v < 1 或 x < 0，则设置错误并返回 NaN
        set_error("iv_ratio", SF_ERROR_DOMAIN, NULL);
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(v) && std::isinf(x)) {
        // 如果 v 和 x 均为无穷大，则没有唯一的极限，设置错误并返回 NaN
        set_error("iv_ratio", SF_ERROR_DOMAIN, NULL);
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) {
        // 如果 x 为 +/-0.0，则根据极限行为返回 +/-0.0
        return x;
    }
    if (std::isinf(v)) {
        // 如果 v 是无穷大，则返回 0.0
        return 0.0;
    }
    if (std::isinf(x)) {
        // 如果 x 是无穷大，则返回 1.0
        return 1.0;
    }

    // 现在 v >= 1 且 x >= 0，且均为有限数值
    int e;
    std::frexp(std::fmax(v, x), &e);
    double c = std::ldexp(1, 2-e); // rescaling multiplier
    double vc = v * c;
    double xc = x * c;

    IvRatioCFTailGenerator cf(vc, xc, c);
    auto [fc, terms] = detail::series_eval_kahan(
        detail::continued_fraction_series(cf),
        std::numeric_limits<double>::epsilon() * 0.5,
        1000,
        2*vc);
    // 如果 terms 等于 0，表示未收敛；这种情况不应该发生
    if (terms == 0) { // failed to converge; should not happen
        // 设置错误消息，指示未找到结果
        set_error("iv_ratio", SF_ERROR_NO_RESULT, NULL);
        // 返回 NaN（非数字），表示无效的结果
        return std::numeric_limits<double>::quiet_NaN();
    }

    // 返回 xc / (xc + fc) 的计算结果
    return xc / (xc + fc);
}  // 结束 special 命名空间

SPECFUN_HOST_DEVICE inline float iv_ratio(float v, float x) {
    // 调用双精度版本的 iv_ratio 函数，将 float 类型的参数转换为 double 类型
    return iv_ratio(static_cast<double>(v), static_cast<double>(x));
}

}  // 结束特定命名空间
```