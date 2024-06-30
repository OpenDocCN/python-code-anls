# `D:\src\scipysrc\scipy\scipy\special\special\wright_bessel.h`

```
/*
 * Translated from Cython into C++ by SciPy developers in 2023.
 * Original header with Copyright information appears below.
 */

/*
 * Implementation of Wright's generalized Bessel function Phi, see
 * https://dlmf.nist.gov/10.46.E1
 *
 * Copyright: Christian Lorentzen
 *
 * Distributed under the same license as SciPy
 *
 *
 * Implementation Overview:
 *
 * First, different functions are implemented valid for certain domains of the
 * three arguments.
 * Finally they are put together in wright_bessel. See the docstring of
 * that function for more details.
 */

#pragma once

#include "cephes/lanczos.h"
#include "cephes/polevl.h"
#include "cephes/rgamma.h"
#include "config.h"
#include "digamma.h"
#include "error.h"

namespace special {

namespace detail {
    // rgamma_zero: smallest value x for which rgamma(x) == 0 as x gets large
    constexpr double rgamma_zero = 178.47241115886637;

    /*
     * Compute exp(x) / gamma(y) = exp(x) * rgamma(y).
     *
     * This helper function avoids overflow by using the lanczos
     * approximation of the gamma function.
     */
    SPECFUN_HOST_DEVICE inline double exp_rgamma(double x, double y) {
        return std::exp(x + (1 - std::log(y + cephes::lanczos_g - 0.5)) * (y - 0.5)) /
               cephes::lanczos_sum_expg_scaled(y);
    }

    /*
     * Taylor series expansion in x=0 for x <= 1.
     *
     * Phi(a, b, x) = sum_k x^k / k! / Gamma(a*k + b)
     *
     * Note that every term, and therefore also Phi(a, b, x) is
     * monotone decreasing with increasing a or b.
     */
    SPECFUN_HOST_DEVICE inline double wb_series(double a, double b, double x, unsigned int nstart, unsigned int nstop) {
        double xk_k = std::pow(x, nstart) * cephes::rgamma(nstart + 1); // x^k/k!
        double res = xk_k * cephes::rgamma(nstart * a + b);
        
        // term k=nstart+1, +2, +3, ...
        if (nstop > nstart) {
            // series expansion until term k such that a*k+b <= rgamma_zero
            unsigned int k_max = std::floor((rgamma_zero - b) / a);
            if (nstop > k_max) {
                nstop = k_max;
            }
            for (unsigned int k = nstart + 1; k < nstop; k++) {
                xk_k *= x / k;
                res += xk_k * cephes::rgamma(a * k + b);
            }
        }
        return res;
    }

    template<bool log_wb>
    SPECFUN_HOST_DEVICE inline double wb_large_a(double a, double b, double x, int n) {
        /* 2. Taylor series expansion in x=0, for large a.
         *
         * Phi(a, b, x) = sum_k x^k / k! / Gamma(a*k + b)
         *
         * Use Stirling's formula to find k=k_max, the maximum term.
         * Then use n terms of Taylor series around k_max.
         */

        // 计算 k_max，使用 Stirling 公式确定最大项的 k 值
        int k_max = static_cast<int>(std::pow(std::pow(a, -a) * x, 1.0 / (1 + a)));

        // 计算起始的 k 值，确保不超出范围
        int nstart = k_max - n / 2;
        if (nstart < 0) {
            nstart = 0;
        }

        double res = 0;
        double lnx = std::log(x);

        // 为了数值稳定性，通过将最大项 exp(..)（k=k_max）因子分出来
        // 如果该项大于 0
        double max_exponent = std::fmax(0, k_max * lnx - cephes::lgam(k_max + 1) - cephes::lgam(a * k_max + b));

        // 计算 Taylor 级数的 n 项和
        for (int k = nstart; k < nstart + n; k++) {
            res += std::exp(k * lnx - cephes::lgam(k + 1) - cephes::lgam(a * k + b) - max_exponent);
        }

        // 根据 log_wb 的值进行后续处理
        if (!log_wb) {
            res *= std::exp(max_exponent);
        } else {
            // Wright 函数的对数
            res = max_exponent + std::log(res);
        }

        // 返回计算结果
        return res;
    }

    template<bool log_wb>
    }

    template<bool log_wb>
    }

    SPECFUN_HOST_DEVICE inline double wb_Kmod(double exp_term, double eps, double a, double b, double x, double r) {
        /* Compute integrand Kmod(eps, a, b, x, r) for Gauss-Laguerre quadrature.
         *
         * K(a, b, x, r+eps) = exp(-r-eps) * Kmod(eps, a, b, x, r)
         * 
         * Kmod(eps, a, b, x, r) = exp(x * (r+eps)^(-a) * cos(pi*a)) * (r+eps)^(-b)
         *                       * sin(x * (r+eps)^(-a) * sin(pi*a) + pi * b)
         * 
         * Note that we additionally factor out exp(exp_term) which helps with large
         * terms in the exponent of exp(...)
         */

        // 计算 Kmod 的积分项，用于 Gauss-Laguerre 积分
        double x_r_a = x * std::pow(r + eps, -a);
        return std::exp(x_r_a * cephes::cospi(a) + exp_term) * std::pow(r + eps, -b) *
               std::sin(x_r_a * cephes::sinpi(a) + M_PI * b);
    }

    SPECFUN_HOST_DEVICE inline double wb_P(double exp_term, double eps, double a, double b, double x, double phi) {
        /* Compute integrand P for Gauss-Legendre quadrature.
         *
         * P(eps, a, b, x, phi) = exp(eps * cos(phi) + x * eps^(-a) * cos(a*phi))
         *                      * cos(eps * sin(phi) - x * eps^(-a) * sin(a*phi)
         *                            + (1-b)*phi)
         * 
         * Note that we additionally factor out exp(exp_term) which helps with large
         * terms in the exponent of exp(...)
         */

        // 计算 P 的积分项，用于 Gauss-Legendre 积分
        double x_eps_a = x * std::pow(eps, -a);
        return std::exp(eps * std::cos(phi) + x_eps_a * std::cos(a * phi) + exp_term) *
               std::cos(eps * std::sin(phi) - x_eps_a * std::sin(a * phi) + (1 - b) * phi);
    }
    /* Laguerre 多项式的根，阶数为 50
     * 使用 scipy.special.roots_laguerre(50)[0] 或 sympy.integrals.quadrature.import gauss_laguerre(50, 16)[0] 获得 */
    constexpr double wb_x_laguerre[] = {
        0.02863051833937908, 0.1508829356769337, 0.3709487815348964, 0.6890906998810479, 1.105625023539913,
        1.620961751102501,   2.23561037591518,   2.950183366641835,  3.765399774405782,  4.682089387559285,
        5.70119757478489,    6.823790909794551,  8.051063669390792,  9.384345308258407,  10.82510903154915,
        12.37498160875746,   14.03575459982991,  15.80939719784467,  17.69807093335025,  19.70414653546156,
        21.83022330657825,   24.0791514444115,   26.45405784125298,  28.95837601193738,  31.59588095662286,
        34.37072996309045,   37.28751061055049,  40.35129757358607,  43.56772026999502,  46.94304399160304,
        50.48426796312992,   54.19924488016862,  58.09682801724853,  62.18705417568891,  66.48137387844482,
        70.99294482661949,   75.73701154772731,  80.73140480247769,  85.99721113646323,  91.55969041253388,
        97.44956561485056,   103.7048912366923,  110.3738588076403,  117.5191982031112,  125.2254701334734,
        133.6120279227287,   142.8583254892541,  153.2603719726036,  165.3856433166825,  180.6983437092145
    };

    /* Laguerre 多项式的权重，阶数为 50
     * 使用 sympy.integrals.quadrature.import gauss_laguerre(50, 16)[1] 获得 */
    constexpr double wb_w_laguerre[] = {
        0.07140472613518988,   0.1471486069645884,    0.1856716275748313,    0.1843853825273539,
        0.1542011686063556,    0.1116853699022688,    0.07105288549019586,   0.04002027691150833,
        0.02005062308007171,   0.008960851203646281,  0.00357811241531566,   0.00127761715678905,
        0.0004080302449837189, 0.0001165288322309724, 2.974170493694165e-5,  6.777842526542028e-6,
        1.37747950317136e-6,   2.492886181720092e-7,  4.010354350427827e-8,  5.723331748141425e-9,
        7.229434249182665e-10, 8.061710142281779e-11, 7.913393099943723e-12, 6.81573661767678e-13,
        5.13242671658949e-14,  3.365624762437814e-15, 1.913476326965035e-16, 9.385589781827253e-18,
        3.950069964503411e-19, 1.417749517827512e-20, 4.309970276292175e-22, 1.101257519845548e-23,
        2.344617755608987e-25, 4.11854415463823e-27,  5.902246763596448e-29, 6.812008916553065e-31,
        6.237449498812102e-33, 4.452440579683377e-35, 2.426862352250487e-37, 9.852971481049686e-40,
        2.891078872318428e-42, 5.906162708112361e-45, 8.01287459750397e-48,  6.789575424396417e-51,
        3.308173010849252e-54, 8.250964876440456e-58, 8.848728128298018e-62, 3.064894889844417e-66,
        1.988708229330752e-71, 6.049567152238783e-78
    };

    /* Legendre 多项式的根，阶数为 50
     * 使用 sympy.integrals.quadrature.import gauss_legendre(50, 16)[0] 获得 */
    # Legendre多项式的节点，用于数值积分，共50个点
    constexpr double wb_x_legendre[] = {
        -0.998866404420071,  -0.9940319694320907, -0.9853540840480059, -0.9728643851066921,  -0.9566109552428079,
        -0.9366566189448779, -0.9130785566557919, -0.885967979523613,  -0.8554297694299461,  -0.8215820708593359,
        -0.7845558329003993, -0.7444943022260685, -0.7015524687068223, -0.6558964656854394,  -0.6077029271849502,
        -0.5571583045146501, -0.5044581449074642, -0.4498063349740388, -0.3934143118975651,  -0.3355002454194374,
        -0.276288193779532,  -0.2160072368760418, -0.1548905899981459, -0.09317470156008614, -0.03109833832718888,
        0.03109833832718888, 0.09317470156008614, 0.1548905899981459,  0.2160072368760418,   0.276288193779532,
        0.3355002454194374,  0.3934143118975651,  0.4498063349740388,  0.5044581449074642,   0.5571583045146501,
        0.6077029271849502,  0.6558964656854394,  0.7015524687068223,  0.7444943022260685,   0.7845558329003993,
        0.8215820708593359,  0.8554297694299461,  0.885967979523613,   0.9130785566557919,   0.9366566189448779,
        0.9566109552428079,  0.9728643851066921,  0.9853540840480059,  0.9940319694320907,   0.998866404420071
    };
    
    # Legendre多项式的权重，用于数值积分，共50个权重
    constexpr double wb_w_legendre[] = {
        0.002908622553155141, 0.006759799195745401, 0.01059054838365097, 0.01438082276148557,  0.01811556071348939,
        0.02178024317012479,  0.02536067357001239,  0.0288429935805352,  0.03221372822357802,  0.03545983561514615,
        0.03856875661258768,  0.0415284630901477,   0.04432750433880328, 0.04695505130394843,  0.04940093844946632,
        0.05165570306958114,  0.05371062188899625,  0.05555774480621252, 0.05718992564772838,  0.05860084981322245,
        0.05978505870426546,  0.06073797084177022,  0.06145589959031666, 0.06193606742068324,  0.06217661665534726,
        0.06217661665534726,  0.06193606742068324,  0.06145589959031666, 0.06073797084177022,  0.05978505870426546,
        0.05860084981322245,  0.05718992564772838,  0.05555774480621252, 0.05371062188899625,  0.05165570306958114,
        0.04940093844946632,  0.04695505130394843,  0.04432750433880328, 0.0415284630901477,   0.03856875661258768,
        0.03545983561514615,  0.03221372822357802,  0.0288429935805352,  0.02536067357001239,  0.02178024317012479,
        0.01811556071348939,  0.01438082276148557,  0.01059054838365097, 0.006759799195745401, 0.002908622553155141
    };
    
    # 通过使用特定的参数计算出的Wright-Bessel函数的最佳eps选择的拟合参数
    # 调用命令：python _precompute/wright_bessel.py 4
    constexpr double wb_A[] = {0.41037, 0.30833, 6.9952, 18.382, -2.8566, 2.1122};
    
    # 用于模板的布尔类型参数，控制日志记录
    template<bool log_wb>
    /* Compute Wright's generalized Bessel function for scalar arguments.
     *
     * According to [1], it is an entire function defined as
     *
     * .. math:: \Phi(a, b; x) = \sum_{k=0}^\infty \frac{x^k}{k! \Gamma(a k + b)}
     *
     * So far, only non-negative values of rho=a, beta=b and z=x are implemented.
     * There are 5 different approaches depending on the ranges of the arguments:
     *
     * 1. Taylor series expansion in x=0 [1], for x <= 1.
     *    Involves gamma functions in each term.
     * 2. Taylor series expansion in x=0 [2], for large a.
     * 3. Taylor series in a=0, for tiny a and not too large x.
     * 4. Asymptotic expansion for large x [3, 4].
     *    Suitable for large x while still small a and b.
     * 5. Integral representation [5], in principle for all arguments.
     *
     * References
     * ----------
     * [1] https://dlmf.nist.gov/10.46.E1
     * [2] P. K. Dunn, G. K. Smyth (2005), Series evaluation of Tweedie exponential
     *     dispersion model densities. Statistics and Computing 15 (2005): 267-280.
     * [3] E. M. Wright (1935), The asymptotic expansion of the generalized Bessel
     *     function. Proc. London Math. Soc. (2) 38, pp. 257-270.
     *     https://doi.org/10.1112/plms/s2-38.1.257
     * [4] R. B. Paris (2017), The asymptotics of the generalised Bessel function,
     *     Mathematica Aeterna, Vol. 7, 2017, no. 4, 381 - 406,
     *     https://arxiv.org/abs/1711.03006
     * [5] Y. F. Luchko (2008), Algorithms for Evaluation of the Wright Function for
     *     the Real Arguments' Values, Fractional Calculus and Applied Analysis 11(1)
     *     http://sci-gems.math.bas.bg/jspui/bitstream/10525/1298/1/fcaa-vol11-num1-2008-57p-75p.pdf
     */
    // Check for NaN values in a, b, or x
    if (std::isnan(a) || std::isnan(b) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Check for negative values in a, b, or x
    if (a < 0 || b < 0 || x < 0) {
        // Set error for negative arguments and return NaN
        set_error("wright_bessel", SF_ERROR_DOMAIN, NULL);
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Check for infinite values in x and if a or b are also infinite
    if (std::isinf(x)) {
        if (std::isinf(a) || std::isinf(b)) {
            // Return NaN if a or b are infinite and x is infinite
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Return positive infinity if x is infinite and a, b are finite
        return std::numeric_limits<double>::infinity();
    }
    // Check for infinite values in a or b
    if (std::isinf(a) || std::isinf(b)) {
        // Return NaN or 0 depending on the context
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Check if a or b are beyond a specific limit
    if (a >= detail::rgamma_zero || b >= detail::rgamma_zero) {
        // Set overflow error and return NaN
        set_error("wright_bessel", SF_ERROR_OVERFLOW, NULL);
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Handle the case when x equals 0
    if (x == 0) {
        // Return the value of the gamma function of b or its logarithm
        if (!log_wb) {
            return cephes::rgamma(b);
        } else {
            // Return the negative logarithm of the gamma function of b
            return -cephes::lgam(b);
        }
    }
    // 如果 a 等于 0
    if (a == 0) {
        // 返回 exp(x) * rgamma(b)
        if (!log_wb) {
            // 如果不是求对数的情况，调用 detail 命名空间中的 exp_rgamma 函数
            return detail::exp_rgamma(x, b);
        } else {
            // 如果是求对数的情况，返回 x - cephes::lgam(b)，即 Wright 函数的对数
            // 使用 cephes 命名空间中的 lgam 函数
            return x - cephes::lgam(b);
        }
    }

    // 设定一个常量，表示 exp(x) 的上限值
    constexpr double exp_inf = 709.78271289338403;
    int order;
    // 根据条件设定变量 order 的值
    if ((a <= 1e-3 && b <= 50 && x <= 9) || (a <= 1e-4 && b <= 70 && x <= 100) ||
        (a <= 1e-5 && b <= 170 && (x < exp_inf || (log_wb && x <= 1e3)))) {
        /* 在 a=0 时的 Taylor 级数展开，order=order => 精度 <= 1e-11
         * 如果 beta 也很小 => 精度 <= 1e-11.
         * 最大 order = 5 */
        if (a <= 1e-5) {
            if (x <= 1) {
                order = 2;
            } else if (x <= 10) {
                order = 3;
            } else if (x <= 100) {
                order = 4;
            } else { // x < exp_inf
                order = 5;
            }
        } else if (a <= 1e-4) {
            if (x <= 1e-2) {
                order = 2;
            } else if (x <= 1) {
                order = 3;
            } else if (x <= 10) {
                order = 4;
            } else { // x <= 100
                order = 5;
            }
        } else { // a <= 1e-3
            if (x <= 1e-5) {
                order = 2;
            } else if (x <= 1e-1) {
                order = 3;
            } else if (x <= 1) {
                order = 4;
            } else { // x <= 9
                order = 5;
            }
        }

        // 调用 detail 命名空间中的 wb_small_a 函数，返回计算结果
        return detail::wb_small_a<log_wb>(a, b, x, order);
    }

    // 对于 x <= 1 的情况，使用 18 项 Taylor 级数展开 => 错误主要小于 5e-14
    if (x <= 1) {
        double res = detail::wb_series(a, b, x, 0, 18);
        // 如果是求对数的情况，取结果的对数
        if (log_wb) res = std::log(res);
        return res;
    }

    // 对于 x <= 2 的情况，使用 20 项 Taylor 级数展开 => 错误主要小于 1e-12 到 1e-13
    if (x <= 2) {
        return detail::wb_series(a, b, x, 0, 20);
    }

    // 对于 a >= 5 的情况，设定变量 order 的值
    if (a >= 5) {
        /* 在大约最大项周围的 Taylor 级数展开。
         * 设定项数=order。 */
        if (a >= 10) {
            if (x <= 1e11) {
                order = 6;
            } else {
                // 计算 order，确保其不超过给定的值
                order = static_cast<int>(std::fmin(std::log10(x) - 5 + b / 10, 30));
            }
        } else {
            if (x <= 1e4) {
                order = 6;
            } else if (x <= 1e8) {
                order = static_cast<int>(2 * std::log10(x));
            } else if (x <= 1e10) {
                order = static_cast<int>(4 * std::log10(x) - 16);
            } else {
                order = static_cast<int>(std::fmin(6 * std::log10(x) - 36, 100));
            }
        }

        // 调用 detail 命名空间中的 wb_large_a 函数，返回计算结果
        return detail::wb_large_a<log_wb>(a, b, x, order);
    }
    // 如果满足条件：(a * x)^(1 / (1. + a)) >= 14 + b * b / (2 * (1 + a))
    // 则使用渐近展开，展开到Z = (a*x)^(1/(1+a))的8阶项，其中包含1/Z^8。
    // 对于1/Z^k，b的最高次数为 b^(2*k) * a0 / (2^k k! (1+a)^k)。
    // 由于a0是所有阶数的共同因子，这解释了上述好收敛域的范围。
    // => 精度约为1e-11，但可能下降至约1e-8或1e-7。
    // 注意：我们确保 a <= 5，因为对于较大的a，这是一个较差的近似。
    return detail::wb_asymptotic<log_wb>(a, b, x);

    // 如果满足条件：0.5 <= a <= 1.8 并且 100 <= b 并且 1e5 <= x
    // 这是一个非常困难的领域。此条件放置在wb_asymptotic之后。
    // TODO: 探索覆盖此领域的方法。
    return std::numeric_limits<double>::quiet_NaN();

    // 如果以上两个条件均不满足，则计算Wright-Bessel积分
    return detail::wright_bessel_integral<log_wb>(a, b, x);
}  // 结束 special 命名空间

SPECFUN_HOST_DEVICE inline double wright_bessel(double a, double b, double x) {
    // 调用模板函数 wright_bessel_t，使用双精度参数 a, b, x
    return wright_bessel_t<false>(a, b, x);
}

SPECFUN_HOST_DEVICE inline float wright_bessel(float a, float b, float x) {
    // 将单精度参数转换为双精度，然后调用双精度版本的 wright_bessel 函数
    return wright_bessel(static_cast<double>(a), static_cast<double>(b), static_cast<double>(x));
}

SPECFUN_HOST_DEVICE inline double log_wright_bessel(double a, double b, double x) {
    // 调用模板函数 wright_bessel_t，使用双精度参数 a, b, x，并要求返回对数值
    return wright_bessel_t<true>(a, b, x);
}

SPECFUN_HOST_DEVICE inline float log_wright_bessel(float a, float b, float x) {
    // 将单精度参数转换为双精度，然后调用双精度版本的 log_wright_bessel 函数
    return log_wright_bessel(static_cast<double>(a), static_cast<double>(b), static_cast<double>(x));
}

}  // 结束 special 命名空间
```