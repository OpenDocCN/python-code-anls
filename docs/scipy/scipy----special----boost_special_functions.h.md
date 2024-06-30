# `D:\src\scipysrc\scipy\scipy\special\boost_special_functions.h`

```
#ifndef BOOST_SPECIAL_FUNCTIONS_H
#define BOOST_SPECIAL_FUNCTIONS_H

#include <cmath>  // 包含数学函数库
#include <stdexcept>  // 包含标准异常类
#include "sf_error.h"  // 包含自定义的错误处理头文件

// Override some default BOOST policies.
// These are required to ensure that the Boost function ibeta_inv
// handles extremely small p values with precision comparable to the
// Cephes incbi function.

#include "boost/math/special_functions/beta.hpp"  // 包含 Boost beta 函数头文件
#include "boost/math/special_functions/erf.hpp"  // 包含 Boost erf 函数头文件
#include "boost/math/special_functions/powm1.hpp"  // 包含 Boost powm1 函数头文件
#include "boost/math/special_functions/hypergeometric_1F1.hpp"  // 包含 Boost 1F1 超几何函数头文件
#include "boost/math/special_functions/hypergeometric_pFq.hpp"  // 包含 Boost pFq 超几何函数头文件

#include "boost/math/distributions.hpp"  // 包含 Boost 分布函数库
#include <boost/math/distributions/inverse_gaussian.hpp>  // 包含 Boost 逆高斯分布函数库

typedef boost::math::policies::policy<
    boost::math::policies::promote_float<false >,  // 不自动提升到浮点数
    boost::math::policies::promote_double<false >,  // 不自动提升到双精度浮点数
    boost::math::policies::max_root_iterations<400 > > SpecialPolicy;  // 最大根迭代次数为 400 的特殊策略

// Round up to achieve correct ppf(cdf) round-trips for discrete distributions
typedef boost::math::policies::policy<
    boost::math::policies::domain_error<boost::math::policies::ignore_error >,  // 忽略域错误
    boost::math::policies::overflow_error<boost::math::policies::user_error >,  // 溢出错误处理为用户自定义错误
    boost::math::policies::evaluation_error<boost::math::policies::user_error >,  // 评估错误处理为用户自定义错误
    boost::math::policies::promote_float<false >,  // 不自动提升到浮点数
    boost::math::policies::promote_double<false >,  // 不自动提升到双精度浮点数
    boost::math::policies::discrete_quantile<
        boost::math::policies::integer_round_up > > StatsPolicy;  // 整数向上舍入的离散量子策略

// Raise a RuntimeWarning making users aware that something went wrong during
// evaluation of the function, but return the best guess
template <class RealType>
RealType
boost::math::policies::user_evaluation_error(const char* function, const char* message, const RealType& val) {
    std::string msg("Error in function ");
    std::string haystack {function};
    const std::string needle {"%1%"};
    msg += haystack.replace(haystack.find(needle), needle.length(), typeid(RealType).name()) + ": ";
    // "message" may have %1%, but arguments don't always contain all
    // required information, so don't call boost::format for now
    msg += message;
    PyGILState_STATE save = PyGILState_Ensure();  // 保存 Python 全局解释器锁状态
    PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 1);  // 触发运行时警告并传递消息
    PyGILState_Release(save);  // 释放 Python 全局解释器锁
    return val;  // 返回最佳猜测值
}

template <class RealType>
RealType
boost::math::policies::user_overflow_error(const char* function, const char* message, const RealType& val) {
    std::string msg("Error in function ");
    std::string haystack {function};
    const std::string needle {"%1%"};
    msg += haystack.replace(haystack.find(needle), needle.length(), typeid(RealType).name()) + ": ";
    // From Boost docs: "overflow and underflow messages do not contain this %1% specifier
    //                   (since the value of value is immaterial in these cases)."
    if (message) {
        msg += message;
    }
    PyGILState_STATE save = PyGILState_Ensure();  // 保存 Python 全局解释器锁状态
    PyErr_SetString(PyExc_OverflowError, msg.c_str());  // 设置溢出错误并传递消息
    // 释放全局解释器锁（GIL）状态，恢复之前保存的状态
    PyGILState_Release(save);
    // 返回整数 0，表示函数正常结束
    return 0;
template<typename Real, typename Policy>
static inline
Real ibeta_inv_wrap(Real a, Real b, Real p, const Policy& policy_)
{
    Real y;

    // 检查输入参数是否有 NaN，如果有，返回 NaN
    if (isnan(a) || isnan(b) || isnan(p)) {
        return NAN;
    }
    // 检查参数是否在定义域内，若不在，则报错并返回 NaN
    if ((a <= 0) || (b <= 0) || (p < 0) || (p > 1)) {
        sf_error("betaincinv", SF_ERROR_DOMAIN, NULL);
        return NAN;
    }
    try {
        // 调用 boost::math 库中的 ibeta_inv 函数计算逆 beta 分布
        y = boost::math::ibeta_inv(a, b, p, policy_);
    } catch (const std::domain_error& e) {
        // 捕获域错误异常，报告错误类型，并返回 NaN
        sf_error("betaincinv", SF_ERROR_DOMAIN, NULL);
        y = NAN;
    } catch (const std::overflow_error& e) {
        // 捕获溢出错误异常，报告溢出错误类型，并返回无穷大
        sf_error("betaincinv", SF_ERROR_OVERFLOW, NULL);
        y = INFINITY;
    } catch (const std::underflow_error& e) {
        // 捕获下溢错误异常，报告下溢错误类型，并返回 0
        sf_error("betaincinv", SF_ERROR_UNDERFLOW, NULL);
        y = 0;
    } catch (...) {
        // 捕获其他未知异常，报告其他错误类型，并返回 NaN
        sf_error("betaincinv", SF_ERROR_OTHER, NULL);
        y = NAN;
    }
    // 返回计算结果
    return y;
}

// 用于单精度浮点数的逆 beta 分布函数调用
float
ibeta_inv_float(float a, float b, float p)
{
    return ibeta_inv_wrap(a, b, p, SpecialPolicy());
}

// 用于双精度浮点数的逆 beta 分布函数调用
double
ibeta_inv_double(double a, double b, double p)
{
    return ibeta_inv_wrap(a, b, p, SpecialPolicy());
}
// 调用 ibeta_inv_wrap 函数计算不完全贝塔函数的逆
ibeta_inv_double(double a, double b, double p)
{
    return ibeta_inv_wrap(a, b, p, SpecialPolicy());
}


// 对于 Real 类型的参数，计算不完全贝塔函数的逆的封装函数
template<typename Real>
static inline
Real ibetac_inv_wrap(Real a, Real b, Real p)
{
    Real y;

    // 检查参数是否包含 NaN，若是则返回 NaN
    if (isnan(a) || isnan(b) || isnan(p)) {
        return NAN;
    }
    // 检查参数范围，若不符合要求则报错并返回 NaN
    if ((a <= 0) || (b <= 0) || (p < 0) || (p > 1)) {
        sf_error("betainccinv", SF_ERROR_DOMAIN, NULL);
        return NAN;
    }
    try {
        // 使用 boost 库计算不完全贝塔函数的逆
        y = boost::math::ibetac_inv(a, b, p, SpecialPolicy());
    } catch (const std::domain_error& e) {
        sf_error("betainccinv", SF_ERROR_DOMAIN, NULL);
        y = NAN;
    } catch (const std::overflow_error& e) {
        sf_error("betainccinv", SF_ERROR_OVERFLOW, NULL);
        y = INFINITY;
    } catch (const std::underflow_error& e) {
        sf_error("betainccinv", SF_ERROR_UNDERFLOW, NULL);
        y = 0;
    } catch (...) {
        sf_error("betainccinv", SF_ERROR_OTHER, NULL);
        y = NAN;
    }
    return y;
}

// 返回 float 类型的不完全贝塔函数的逆
float
ibetac_inv_float(float a, float b, float p)
{
    return ibetac_inv_wrap(a, b, p);
}

// 返回 double 类型的不完全贝塔函数的逆
double
ibetac_inv_double(double a, double b, double p)
{
    return ibetac_inv_wrap(a, b, p);
}


// 对于 Real 类型的参数，计算误差函数的逆的封装函数
template<typename Real>
static inline
Real erfinv_wrap(Real x)
{
    Real y;

    // 处理特殊情况：x 等于 -1 时返回负无穷
    if (x == -1) {
        return -INFINITY;
    }
    // 处理特殊情况：x 等于 1 时返回正无穷
    if (x == 1) {
        return INFINITY;
    }

    try {
        // 使用 boost 库计算误差函数的逆
        y = boost::math::erf_inv(x, SpecialPolicy());
    } catch (const std::domain_error& e) {
        sf_error("erfinv", SF_ERROR_DOMAIN, NULL);
        y = NAN;
    } catch (const std::overflow_error& e) {
        sf_error("erfinv", SF_ERROR_OVERFLOW, NULL);
        y = INFINITY;
    } catch (const std::underflow_error& e) {
        sf_error("erfinv", SF_ERROR_UNDERFLOW, NULL);
        y = 0;
    } catch (...) {
        sf_error("erfinv", SF_ERROR_OTHER, NULL);
        y = NAN;
    }
    return y;
}

// 返回 float 类型的误差函数的逆
float
erfinv_float(float x)
{
    return erfinv_wrap(x);
}

// 返回 double 类型的误差函数的逆
double
erfinv_double(double x)
{
    return erfinv_wrap(x);
}


// 对于 Real 类型的参数，计算 x 的 y 次幂减 1 的封装函数
template<typename Real>
static inline
Real powm1_wrap(Real x, Real y)
{
    Real z;

    // 处理边界情况：y 为 0 或 x 为 1 时返回 0
    if (y == 0 || x == 1) {
        // (anything)**0 is 1
        // 1**(anything) is 1
        // This includes the case 0**0, and 'anything' includes inf and nan.
        return 0;
    }
    // 处理特殊情况：x 为 0 且 y 小于 0 时报错并返回正无穷
    if (x == 0) {
        if (y < 0) {
            sf_error("powm1", SF_ERROR_DOMAIN, NULL);
            return INFINITY;
        }
        else if (y > 0) {
            return -1;
        }
    }
    // 处理特殊情况：x 小于 0 且 y 不是整数时报错并返回 NaN
    if (x < 0 && std::trunc(y) != y) {
        // To compute x**y with x < 0, y must be an integer.
        sf_error("powm1", SF_ERROR_DOMAIN, NULL);
        return NAN;
    }

    try {
        // 使用 boost 库计算 x 的 y 次幂减 1
        z = boost::math::powm1(x, y, SpecialPolicy());
    } catch (const std::domain_error& e) {
        sf_error("powm1", SF_ERROR_DOMAIN, NULL);
        z = NAN;
        z = NAN;
    } catch (const std::overflow_error& e) {
        sf_error("powm1", SF_ERROR_OVERFLOW, NULL);
        z = INFINITY;
    } catch (const std::underflow_error& e) {
        sf_error("powm1", SF_ERROR_UNDERFLOW, NULL);
        z = 0;
    } catch (...) {
        sf_error("powm1", SF_ERROR_OTHER, NULL);
        z = NAN;
    }
    return z;
}
    } catch (const std::overflow_error& e) {
        // 如果计算溢出，记录溢出错误
        sf_error("powm1", SF_ERROR_OVERFLOW, NULL);

        // 根据输入参数 x 和 y 的不同情况计算结果 z
        // 参考：https://en.cppreference.com/w/cpp/numeric/math/pow

        if (x > 0) {
            // 当 x > 0 时
            if (y < 0) {
                // 当 y < 0 时，结果 z 为 0
                z = 0;
            }
            else if (y == 0) {
                // 当 y 等于 0 时，结果 z 为 1
                z = 1;
            }
            else {
                // 当 y 大于 0 时，结果 z 为正无穷大
                z = INFINITY;
            }
        }
        else if (x == 0) {
            // 当 x 等于 0 时，结果 z 为正无穷大
            z = INFINITY;
        }
        else {
            // 当 x 小于 0 时
            if (y < 0) {
                // 当 y < 0 且 y 是偶数时，结果 z 为 0
                if (std::fmod(y, 2) == 0) {
                    z = 0;
                }
                // 当 y < 0 且 y 是奇数时，结果 z 为负零
                else {
                    z = -0;
                }
            }
            else if (y == 0) {
                // 当 y 等于 0 时，结果 z 为 1
                z = 1;
            }
            else {
                // 当 y 大于 0 且 y 是偶数时，结果 z 为正无穷大
                if (std::fmod(y, 2) == 0) {
                    z = INFINITY;
                }
                // 当 y 大于 0 且 y 是奇数时，结果 z 为负无穷大
                else {
                    z = -INFINITY;
                }
            }
        }
    } catch (const std::underflow_error& e) {
        // 如果计算下溢，记录下溢错误
        sf_error("powm1", SF_ERROR_UNDERFLOW, NULL);
        // 结果 z 设为 0
        z = 0;
    } catch (...) {
        // 对于其它异常，记录其它错误
        sf_error("powm1", SF_ERROR_OTHER, NULL);
        // 结果 z 设为 NaN（非数值）
        z = NAN;
    }
    // 返回最终计算结果 z
    return z;
}

// 返回 powm1_wrap 函数的浮点数版本结果
float
powm1_float(float x, float y)
{
    return powm1_wrap(x, y);
}

// 返回 powm1_wrap 函数的双精度版本结果
double
powm1_double(double x, double y)
{
    return powm1_wrap(x, y);
}


//
// 因为 Boost 版本 1.80 及更早版本中的 hypergeometric_1F1 在某些边缘情况下存在 bug 或不一致行为，
// 所以这里包装了 hypergeometric_pFq 函数。在这些情况下，hypergeometric_pFq 函数表现正确，
// 因此我们会在更新 Boost 版本之前使用它。
//
template<typename Real>
static inline
Real call_hypergeometric_pFq(Real a, Real b, Real x)
{
    Real y;

    try {
        // 初始化误差变量为零
        Real p_abs_error = 0;
        // 调用 Boost 的 hypergeometric_pFq 函数计算超几何函数值，并获取绝对误差
        y = boost::math::hypergeometric_pFq({a}, {b}, x, &p_abs_error, SpecialPolicy());
    } catch (const std::domain_error& e) {
        // 如果捕获到域错误，使用特定的错误信息报告，并将 y 设为无穷大
        sf_error("hyp1f1", SF_ERROR_DOMAIN, NULL);
        y = INFINITY;
    } catch (const std::overflow_error& e) {
        // 如果捕获到溢出错误，使用特定的错误信息报告，并将 y 设为无穷大
        sf_error("hyp1f1", SF_ERROR_OVERFLOW, NULL);
        y = INFINITY;
    } catch (const std::underflow_error& e) {
        // 如果捕获到下溢错误，使用特定的错误信息报告，并将 y 设为零
        sf_error("hyp1f1", SF_ERROR_UNDERFLOW, NULL);
        y = 0;
    } catch (...) {
        // 如果捕获到其他类型的错误，使用特定的错误信息报告，并将 y 设为 NaN
        sf_error("hyp1f1", SF_ERROR_OTHER, NULL);
        y = NAN;
    }
    // 返回计算结果 y
    return y;
}

template<typename Real>
static inline
Real hyp1f1_wrap(Real a, Real b, Real x)
{
    Real y;

    // 如果 a、b 或 x 中有任何一个为 NaN，则直接返回 NaN
    if (isnan(a) || isnan(b) || isnan(x)) {
        return NAN;
    }
    // 如果 b 是非正整数，并且是整数，则处理特殊情况
    if (b <= 0 && std::trunc(b) == b) {
        // b 是非正整数。
        // 注意：此处的代码设计保留了在这种边缘情况下 hyp1f1 的历史行为。
        // 其他软件如 Boost、mpmath 和 Mathematica 对某些子情况使用不同的约定。
        if (b != 0  && a == b) {
            // 在这种情况下，使用 Boost 的 hypergeometric_pFq 函数而不是 hypergeometric_1F1 函数。
            // 这避免了 Boost 1.80 及更早版本中的不一致性问题；详见 https://github.com/boostorg/math/issues/829。
            return call_hypergeometric_pFq(a, b, x);
        }
        if (!(a < 0 && std::trunc(a) == a && a >= b)) {
            return INFINITY;
        }
        // 继续执行，让 Boost 函数处理剩余情况。
    }
    if (a < 0 && std::trunc(a) == a && b > 0 && b == x) {
        // 避免 Boost 1.80 及更早版本中的 bug，当 a 是负整数，b 是正数，且 b == x 时发生 bug。
        // 详见 https://github.com/boostorg/math/issues/833。
        return call_hypergeometric_pFq(a, b, x);
    }

    // 使用 Boost 的 hypergeometric_1F1 函数进行基本计算。
    // 它应该能够处理大多数情况。
    // 如果需要更多的特定情况处理，可以在此添加代码。
    // 也正确处理上面未涵盖的任何其他特殊情况。
    // 捕获所有异常，并使用 `special` 错误处理函数处理它们。
    try {
        // Real p_abs_error = 0;
        // 调用 Boost 库中的超几何函数 hypergeometric_1F1，使用特定的策略 SpecialPolicy
        y = boost::math::hypergeometric_1F1(a, b, x, SpecialPolicy());
    } catch (const std::domain_error& e) {
        // 如果捕获到 std::domain_error 异常，使用 sf_error 函数报告 SF_ERROR_DOMAIN 错误
        sf_error("hyp1f1", SF_ERROR_DOMAIN, NULL);
        // 将 y 设为正无穷
        y = INFINITY;
    } catch (const std::overflow_error& e) {
        // 如果捕获到 std::overflow_error 异常，使用 sf_error 函数报告 SF_ERROR_OVERFLOW 错误
        sf_error("hyp1f1", SF_ERROR_OVERFLOW, NULL);
        // 将 y 设为正无穷
        y = INFINITY;
    } catch (const std::underflow_error& e) {
        // 如果捕获到 std::underflow_error 异常，使用 sf_error 函数报告 SF_ERROR_UNDERFLOW 错误
        sf_error("hyp1f1", SF_ERROR_UNDERFLOW, NULL);
        // 将 y 设为 0
        y = 0;
    } catch (...) {
        // 如果捕获到任何其他类型的异常，使用 sf_error 函数报告 SF_ERROR_OTHER 错误
        sf_error("hyp1f1", SF_ERROR_OTHER, NULL);
        // 将 y 设为 NaN（Not-a-Number）
        y = NAN;
    }
    // 返回计算结果 y
    return y;
}

double
hyp1f1_double(double a, double b, double x)
{
    return hyp1f1_wrap(a, b, x);
}
// 计算给定非中心卡方分布的概率密度函数值，输入参数为实数类型
ncx2_pdf_wrap(const Real x, const Real k, const Real l)
{
    // 检查 x 是否有限
    if (std::isfinite(x)) {
        // 调用 Boost 库计算非中心卡方分布的概率密度函数值
        return boost::math::pdf(
            boost::math::non_central_chi_squared_distribution<Real, StatsPolicy>(k, l), x);
    }
    // 若 x 为无穷大或无穷小，则返回 NAN
    return NAN; // inf or -inf returns NAN
}

// 计算给定浮点数类型的非中心卡方分布的概率密度函数值
float
ncx2_pdf_float(float x, float k, float l)
{
    // 调用通用的 ncx2_pdf_wrap 函数计算非中心卡方分布的概率密度函数值
    return ncx2_pdf_wrap(x, k, l);
}

// 计算给定双精度浮点数类型的非中心卡方分布的概率密度函数值
double
ncx2_pdf_double(double x, double k, double l)
{
    // 调用通用的 ncx2_pdf_wrap 函数计算非中心卡方分布的概率密度函数值
    return ncx2_pdf_wrap(x, k, l);
}

// 计算给定泛型类型 Real 的非中心卡方分布的累积分布函数值
template<typename Real>
Real
ncx2_cdf_wrap(const Real x, const Real k, const Real l)
{
    // 检查 x 是否有限
    if (std::isfinite(x)) {
        // 调用 Boost 库计算非中心卡方分布的累积分布函数值
        return boost::math::cdf(
            boost::math::non_central_chi_squared_distribution<Real, StatsPolicy>(k, l), x);
    }
    // 若 x 为无穷大或无穷小，返回 0 或 1
    // -inf => 0, inf => 1
    return 1 - std::signbit(x);
}

// 计算给定浮点数类型的非中心卡方分布的累积分布函数值
float
ncx2_cdf_float(float x, float k, float l)
{
    // 调用通用的 ncx2_cdf_wrap 函数计算非中心卡方分布的累积分布函数值
    return ncx2_cdf_wrap(x, k, l);
}

// 计算给定双精度浮点数类型的非中心卡方分布的累积分布函数值
double
ncx2_cdf_double(double x, double k, double l)
{
    // 调用通用的 ncx2_cdf_wrap 函数计算非中心卡方分布的累积分布函数值
    return ncx2_cdf_wrap(x, k, l);
}

// 计算给定泛型类型 Real 的非中心卡方分布的百分位点函数值
template<typename Real>
Real
ncx2_ppf_wrap(const Real x, const Real k, const Real l)
{
    // 调用 Boost 库计算非中心卡方分布的百分位点函数值
    return boost::math::quantile(
        boost::math::non_central_chi_squared_distribution<Real, StatsPolicy>(k, l), x);
}

// 计算给定浮点数类型的非中心卡方分布的百分位点函数值
float
ncx2_ppf_float(float x, float k, float l)
{
    // 调用通用的 ncx2_ppf_wrap 函数计算非中心卡方分布的百分位点函数值
    return ncx2_ppf_wrap(x, k, l);
}

// 计算给定双精度浮点数类型的非中心卡方分布的百分位点函数值
double
ncx2_ppf_double(double x, double k, double l)
{
    // 调用通用的 ncx2_ppf_wrap 函数计算非中心卡方分布的百分位点函数值
    return ncx2_ppf_wrap(x, k, l);
}

// 计算给定泛型类型 Real 的非中心卡方分布的生存函数值
template<typename Real>
Real
ncx2_sf_wrap(const Real x, const Real k, const Real l)
{
    // 调用 Boost 库计算非中心卡方分布的生存函数值
    return boost::math::cdf(boost::math::complement(
        boost::math::non_central_chi_squared_distribution<Real, StatsPolicy>(k, l), x));
}

// 计算给定浮点数类型的非中心卡方分布的生存函数值
float
ncx2_sf_float(float x, float k, float l)
{
    // 调用通用的 ncx2_sf_wrap 函数计算非中心卡方分布的生存函数值
    return ncx2_sf_wrap(x, k, l);
}

// 计算给定双精度浮点数类型的非中心卡方分布的生存函数值
double
ncx2_sf_double(double x, double k, double l)
{
    // 调用通用的 ncx2_sf_wrap 函数计算非中心卡方分布的生存函数值
    return ncx2_sf_wrap(x, k, l);
}

// 计算给定泛型类型 Real 的非中心卡方分布的逆累积分布函数值
template<typename Real>
Real
ncx2_isf_wrap(const Real x, const Real k, const Real l)
{
    // 调用 Boost 库计算非中心卡方分布的逆累积分布函数值
    return boost::math::quantile(boost::math::complement(
        boost::math::non_central_chi_squared_distribution<Real, StatsPolicy>(k, l), x));
}

// 计算给定浮点数类型的非中心卡方分布的逆累积分布函数值
float
ncx2_isf_float(float x, float k, float l)
{
    // 调用通用的 ncx2_isf_wrap 函数计算非中心卡方分布的逆累积分布函数值
    return ncx2_isf_wrap(x, k, l);
}

// 计算给定双精度浮点数类型的非中心卡方分布的逆累积分布函数值
double
ncx2_isf_double(double x, double k, double l)
{
    // 调用通用的 ncx2_isf_wrap 函数计算非中心卡方分布的逆累积分布函数值
    return ncx2_isf_wrap(x, k, l);
}

// 计算给定泛型类型 Real 的非中心 F 分布的概率密度函数值
template<typename Real>
Real
ncf_pdf_wrap(const Real x, const Real v1, const Real v2, const Real l)
{
    // 检查 x 是否有限
    if (std::isfinite(x)) {
        // 调用 Boost 库计算非中心 F 分布的概率密度函数值
        return boost::math::pdf(
            boost::math::non_central_f_distribution<Real, StatsPolicy>(v1, v2, l), x);
    }
    // 若 x 为无穷大或无穷小，则返回 NAN
    return NAN; // inf or -inf returns NAN
}

// 计算给定浮点数类型的非中心 F 分布的概率密度函数值
float
ncf_pdf_float(float x, float v1, float v2, float l)
{
    // 调用通用的 ncf_pdf_wrap 函数计算非中心 F 分布的概率密度函数值
    return ncf_pdf_wrap(x, v1, v2, l);
}

// 计算给定双精度浮点数类型的非中心 F 分布的概率密度函数值
double
ncf_pdf_double(double x, double v1, double v2, double l)
{
    // 调用通用的 ncf_pdf_wrap 函数计算非中心 F 分布的概率密度函数
// 返回非中心 F 分布的累积分布函数值，用于 float 类型
float
ncf_cdf_float(float x, float v1, float v2, float l)
{
    return ncf_cdf_wrap(x, v1, v2, l);
}

// 返回非中心 F 分布的累积分布函数值，用于 double 类型
double
ncf_cdf_double(double x, double v1, double v2, double l)
{
    return ncf_cdf_wrap(x, v1, v2, l);
}

// 返回非中心 F 分布的分位数函数值，用于任意 Real 类型
template<typename Real>
Real
ncf_ppf_wrap(const Real x, const Real v1, const Real v2, const Real l)
{
    return boost::math::quantile<Real, StatsPolicy>(
        boost::math::non_central_f_distribution<Real, StatsPolicy>(v1, v2, l), x);
}

// 返回非中心 F 分布的分位数函数值，用于 float 类型
float
ncf_ppf_float(float x, float v1, float v2, float l)
{
    return ncf_ppf_wrap(x, v1, v2, l);
}

// 返回非中心 F 分布的分位数函数值，用于 double 类型
double
ncf_ppf_double(double x, double v1, double v2, double l)
{
    return ncf_ppf_wrap(x, v1, v2, l);
}

// 返回非中心 F 分布的生存函数值，用于任意 Real 类型
template<typename Real>
Real
ncf_sf_wrap(const Real x, const Real v1, const Real v2, const Real l)
{
    return boost::math::cdf(boost::math::complement(
        boost::math::non_central_f_distribution<Real, StatsPolicy>(v1, v2, l), x));
}

// 返回非中心 F 分布的生存函数值，用于 float 类型
float
ncf_sf_float(float x, float v1, float v2, float l)
{
    return ncf_sf_wrap(x, v1, v2, l);
}

// 返回非中心 F 分布的生存函数值，用于 double 类型
double
ncf_sf_double(double x, double v1, double v2, double l)
{
    return ncf_sf_wrap(x, v1, v2, l);
}

// 返回非中心 F 分布的逆生存函数值，用于任意 Real 类型
template<typename Real>
Real
ncf_isf_wrap(const Real x, const Real v1, const Real v2, const Real l)
{
    return boost::math::quantile(boost::math::complement(
        boost::math::non_central_f_distribution<Real, StatsPolicy>(v1, v2, l), x));
}

// 返回非中心 F 分布的逆生存函数值，用于 float 类型
float
ncf_isf_float(float x, float v1, float v2, float l)
{
    return ncf_isf_wrap(x, v1, v2, l);
}

// 返回非中心 F 分布的逆生存函数值，用于 double 类型
double
ncf_isf_double(double x, double v1, double v2, double l)
{
    return ncf_isf_wrap(x, v1, v2, l);
}

// 返回非中心 F 分布的均值，用于 float 类型
float
ncf_mean_float(float v1, float v2, float l)
{
    // 如果 v2 小于等于 2.0，则返回 NaN
    RETURN_NAN(v2, 2.0);
    return boost::math::mean(boost::math::non_central_f_distribution<float, StatsPolicy>(v1, v2, l));
}

// 返回非中心 F 分布的均值，用于 double 类型
double
ncf_mean_double(double v1, double v2, double l)
{
    // 如果 v2 小于等于 2.0，则返回 NaN
    RETURN_NAN(v2, 2.0);
    return boost::math::mean(boost::math::non_central_f_distribution<double, StatsPolicy>(v1, v2, l));
}

// 返回非中心 F 分布的方差，用于 float 类型
float
ncf_variance_float(float v1, float v2, float l)
{
    // 如果 v2 小于等于 4.0，则返回 NaN
    RETURN_NAN(v2, 4.0);
    return boost::math::variance(boost::math::non_central_f_distribution<float, StatsPolicy>(v1, v2, l));
}

// 返回非中心 F 分布的方差，用于 double 类型
double
ncf_variance_double(double v1, double v2, double l)
{
    // 如果 v2 小于等于 4.0，则返回 NaN
    RETURN_NAN(v2, 4.0);
    return boost::math::variance(boost::math::non_central_f_distribution<double, StatsPolicy>(v1, v2, l));
}

// 返回非中心 F 分布的偏度，用于 float 类型
float
ncf_skewness_float(float v1, float v2, float l)
{
    // 如果 v2 小于等于 6.0，则返回 NaN
    RETURN_NAN(v2, 6.0);
    return boost::math::skewness(boost::math::non_central_f_distribution<float, StatsPolicy>(v1, v2, l));
}

// 返回非中心 F 分布的偏度，用于 double 类型
double
ncf_skewness_double(double v1, double v2, double l)
{
    // 如果 v2 小于等于 6.0，则返回 NaN
    RETURN_NAN(v2, 6.0);
    return boost::math::skewness(boost::math::non_central_f_distribution<double, StatsPolicy>(v1, v2, l));
}

// 返回非中心 F 分布的峰度（超额峰度），用于 float 类型
float
ncf_kurtosis_excess_float(float v1, float v2, float l)
{
    // 如果 v2 小于等于 8.0，则返回 NaN
    RETURN_NAN(v2, 8.0);
    return boost::math::kurtosis_excess(boost::math::non_central_f_distribution<float, StatsPolicy>(v1, v2, l));
}

// 返回非中心 F 分布的峰度（超额峰度），用于 double 类型
double
ncf_kurtosis_excess_double(double v1, double v2, double l)
{
    // 如果 v2 小于等于 8.0，则返回 NaN
    RETURN_NAN(v2, 8.0);
    return boost::math::kurtosis_excess(boost::math::non_central_f_distribution<double, StatsPolicy>(v1, v2, l));
}
float
nct_kurtosis_excess_float(float v, float l)
{
    // 如果输入参数 v 是 NaN（非数字），则返回 8.0
    RETURN_NAN(v, 4.0);
    // 使用 Boost.Math 计算非中心 t 分布的峰度偏移
    return boost::math::kurtosis_excess(boost::math::non_central_t_distribution<float, StatsPolicy>(v, l));
}

double
nct_kurtosis_excess_double(double v1, double v2, double l)
{
    // 如果输入参数 v2 是 NaN（非数字），则返回 8.0
    RETURN_NAN(v2, 8.0);
    // 使用 Boost.Math 计算非中心 F 分布的峰度偏移
    return boost::math::kurtosis_excess(boost::math::non_central_f_distribution<double, StatsPolicy>(v1, v2, l));
}

template<typename Real>
Real
nct_cdf_wrap(const Real x, const Real v, const Real l)
{
    // 如果 x 是有限的数值
    if (std::isfinite(x)) {
        // 使用 Boost.Math 计算非中心 t 分布的累积分布函数（CDF）
        return boost::math::cdf(
            boost::math::non_central_t_distribution<Real, StatsPolicy>(v, l), x);
    }
    // 如果 x 是 -inf，则返回 0；如果 x 是 inf，则返回 1
    return 1.0 - std::signbit(x);
}

float
nct_cdf_float(float x, float v, float l)
{
    // 调用通用的累积分布函数封装函数，使用 float 类型
    return nct_cdf_wrap(x, v, l);
}

double
nct_cdf_double(double x, double v, double l)
{
    // 调用通用的累积分布函数封装函数，使用 double 类型
    return nct_cdf_wrap(x, v, l);
}

template<typename Real>
Real
nct_ppf_wrap(const Real x, const Real v, const Real l)
{
    // 使用 Boost.Math 计算非中心 t 分布的分位数函数（PPF）
    return boost::math::quantile(
        boost::math::non_central_t_distribution<Real, StatsPolicy>(v, l), x);
}

float
nct_ppf_float(float x, float v, float l)
{
    // 调用通用的分位数函数封装函数，使用 float 类型
    return nct_ppf_wrap(x, v, l);
}

double
nct_ppf_double(double x, double v, double l)
{
    // 调用通用的分位数函数封装函数，使用 double 类型
    return nct_ppf_wrap(x, v, l);
}

template<typename Real>
Real
nct_sf_wrap(const Real x, const Real v, const Real l)
{
    // 使用 Boost.Math 计算非中心 t 分布的生存函数（SF）
    return boost::math::cdf(boost::math::complement(
        boost::math::non_central_t_distribution<Real, StatsPolicy>(v, l), x));
}

float
nct_sf_float(float x, float v, float l)
{
    // 调用通用的生存函数封装函数，使用 float 类型
    return nct_sf_wrap(x, v, l);
}

double
nct_sf_double(double x, double v, double l)
{
    // 调用通用的生存函数封装函数，使用 double 类型
    return nct_sf_wrap(x, v, l);
}

template<typename Real>
Real
nct_isf_wrap(const Real x, const Real v, const Real l)
{
    // 使用 Boost.Math 计算非中心 t 分布的逆生存函数（ISF）
    return boost::math::quantile(boost::math::complement(
        boost::math::non_central_t_distribution<Real, StatsPolicy>(v, l), x));
}

float
nct_isf_float(float x, float v, float l)
{
    // 调用通用的逆生存函数封装函数，使用 float 类型
    return nct_isf_wrap(x, v, l);
}

double
nct_isf_double(double x, double v, double l)
{
    // 调用通用的逆生存函数封装函数，使用 double 类型
    return nct_isf_wrap(x, v, l);
}

float
nct_mean_float(float v, float l)
{
    // 如果输入参数 v 是 NaN（非数字），则返回 1.0
    RETURN_NAN(v, 1.0);
    // 使用 Boost.Math 计算非中心 t 分布的均值
    return boost::math::mean(boost::math::non_central_t_distribution<float, StatsPolicy>(v, l));
}

double
nct_mean_double(double v, double l)
{
    // 如果输入参数 v 是 NaN（非数字），则返回 1.0
    RETURN_NAN(v, 1.0);
    // 使用 Boost.Math 计算非中心 t 分布的均值
    return boost::math::mean(boost::math::non_central_t_distribution<double, StatsPolicy>(v, l));
}

float
nct_variance_float(float v, float l)
{
    // 如果输入参数 v 是 NaN（非数字），则返回 2.0
    RETURN_NAN(v, 2.0);
    // 使用 Boost.Math 计算非中心 t 分布的方差
    return boost::math::variance(boost::math::non_central_t_distribution<float, StatsPolicy>(v, l));
}

double
nct_variance_double(double v, double l)
{
    // 如果输入参数 v 是 NaN（非数字），则返回 2.0
    RETURN_NAN(v, 2.0);
    // 使用 Boost.Math 计算非中心 t 分布的方差
    return boost::math::variance(boost::math::non_central_t_distribution<double, StatsPolicy>(v, l));
}

float
nct_skewness_float(float v, float l)
{
    // 如果输入参数 v 是 NaN（非数字），则返回 3.0
    RETURN_NAN(v, 3.0);
    // 使用 Boost.Math 计算非中心 t 分布的偏度
    return boost::math::skewness(boost::math::non_central_t_distribution<float, StatsPolicy>(v, l));
}

double
nct_skewness_double(double v, double l)
{
    // 如果输入参数 v 是 NaN（非数字），则返回 3.0
    RETURN_NAN(v, 3.0);
    // 使用 Boost.Math 计算非中心 t 分布的偏度
    return boost::math::skewness(boost::math::non_central_t_distribution<double, StatsPolicy>(v, l));
}
    # 调用 Boost 库中的数学函数，计算给定非中心 t 分布的峰度（超额峰度）
    return boost::math::kurtosis_excess(
        boost::math::non_central_t_distribution<float, StatsPolicy>(v, l)
    );
}

// 返回值为 NAN，如果参数 v 不是有限数，则返回 4.0
double
nct_kurtosis_excess_double(double v, double l)
{
    RETURN_NAN(v, 4.0);
    // 调用 Boost 数学库中的 kurtosis_excess 函数，计算非中心 t 分布的峰度超额
    return boost::math::kurtosis_excess(boost::math::non_central_t_distribution<double, StatsPolicy>(v, l));
}

// 计算 Skew-Normal 分布的累积分布函数，返回值为 Real 类型
template<typename Real>
Real
skewnorm_cdf_wrap(const Real x, const Real l, const Real sc, const Real sh)
{
    // 如果 x 是有限数，则计算 Skew-Normal 分布的累积分布函数值
    if (std::isfinite(x)) {
        return boost::math::cdf(
            boost::math::skew_normal_distribution<Real, StatsPolicy>(l, sc, sh), x);
    }
    // 如果 x 是无穷大或负无穷大，返回 1 或 0
    return 1 - std::signbit(x);
}

// 返回 Skew-Normal 分布的累积分布函数，参数为 float 类型
float
skewnorm_cdf_float(float x, float l, float sc, float sh)
{
    return skewnorm_cdf_wrap(x, l, sc, sh);
}

// 返回 Skew-Normal 分布的累积分布函数，参数为 double 类型
double
skewnorm_cdf_double(double x, double l, double sc, double sh)
{
    return skewnorm_cdf_wrap(x, l, sc, sh);
}

// 计算 Binomial 分布的概率质量函数，返回值为 Real 类型
template<typename Real>
Real
binom_pmf_wrap(const Real x, const Real n, const Real p)
{
    // 如果 x 是有限数，则计算 Binomial 分布的概率质量函数值
    if (std::isfinite(x)) {
        return boost::math::pdf(
            boost::math::binomial_distribution<Real, StatsPolicy>(n, p), x);
    }
    // 如果 x 是无穷大或负无穷大，返回 NAN
    return NAN; // inf or -inf returns NAN
}

// 返回 Binomial 分布的概率质量函数，参数为 float 类型
float
binom_pmf_float(float x, float n, float p)
{
    return binom_pmf_wrap(x, n, p);
}

// 返回 Binomial 分布的概率质量函数，参数为 double 类型
double
binom_pmf_double(double x, double n, double p)
{
    return binom_pmf_wrap(x, n, p);
}

// 计算 Binomial 分布的累积分布函数，返回值为 Real 类型
template<typename Real>
Real
binom_cdf_wrap(const Real x, const Real n, const Real p)
{
    // 如果 x 是有限数，则计算 Binomial 分布的累积分布函数值
    if (std::isfinite(x)) {
        return boost::math::cdf(
            boost::math::binomial_distribution<Real, StatsPolicy>(n, p), x);
    }
    // 如果 x 是无穷大或负无穷大，返回 1 或 0
    return 1 - std::signbit(x);
}

// 返回 Binomial 分布的累积分布函数，参数为 float 类型
float
binom_cdf_float(float x, float n, float p)
{
    return binom_cdf_wrap(x, n, p);
}

// 返回 Binomial 分布的累积分布函数，参数为 double 类型
double
binom_cdf_double(double x, double n, double p)
{
    return binom_cdf_wrap(x, n, p);
}

// 计算 Binomial 分布的分位数函数，返回值为 Real 类型
template<typename Real>
Real
binom_ppf_wrap(const Real x, const Real n, const Real p)
{
    return boost::math::quantile(
        boost::math::binomial_distribution<Real, StatsPolicy>(n, p), x);
}

// 返回 Binomial 分布的分位数函数，参数为 float 类型
float
binom_ppf_float(float x, float n, float p)
{
    return binom_ppf_wrap(x, n, p);
}

// 返回 Binomial 分布的分位数函数，参数为 double 类型
double
binom_ppf_double(double x, double n, double p)
{
    # 调用函数 binom_ppf_wrap，并返回其结果
    return binom_ppf_wrap(x, n, p);
}

template<typename Real>
Real
binom_sf_wrap(const Real x, const Real n, const Real p)
{
    // 使用 Boost.Math 库计算二项分布的生存函数值
    return boost::math::cdf(boost::math::complement(
        boost::math::binomial_distribution<Real, StatsPolicy>(n, p), x));
}

float
binom_sf_float(float x, float n, float p)
{
    // 调用通用模板函数 binom_sf_wrap，处理单精度浮点数参数
    return binom_sf_wrap(x, n, p);
}

double
binom_sf_double(double x, double n, double p)
{
    // 调用通用模板函数 binom_sf_wrap，处理双精度浮点数参数
    return binom_sf_wrap(x, n, p);
}

template<typename Real>
Real
binom_isf_wrap(const Real x, const Real n, const Real p)
{
    // 使用 Boost.Math 库计算二项分布的逆生存函数值
    return boost::math::quantile(boost::math::complement(
        boost::math::binomial_distribution<Real, StatsPolicy>(n, p), x));
}

float
binom_isf_float(float x, float n, float p)
{
    // 调用通用模板函数 binom_isf_wrap，处理单精度浮点数参数
    return binom_isf_wrap(x, n, p);
}

double
binom_isf_double(double x, double n, double p)
{
    // 调用通用模板函数 binom_isf_wrap，处理双精度浮点数参数
    return binom_isf_wrap(x, n, p);
}

template<typename Real>
Real
nbinom_pmf_wrap(const Real x, const Real r, const Real p)
{
    // 如果 x 是有限值，则使用 Boost.Math 库计算负二项分布的概率质量函数值
    if (std::isfinite(x)) {
        return boost::math::pdf(
            boost::math::negative_binomial_distribution<Real, StatsPolicy>(r, p), x);
    }
    // 如果 x 是无限值，则返回 NAN，表示无法计算
    return NAN; // inf or -inf returns NAN
}

float
nbinom_pmf_float(float x, float r, float p)
{
    // 调用通用模板函数 nbinom_pmf_wrap，处理单精度浮点数参数
    return nbinom_pmf_wrap(x, r, p);
}

double
nbinom_pmf_double(double x, double r, double p)
{
    // 调用通用模板函数 nbinom_pmf_wrap，处理双精度浮点数参数
    return nbinom_pmf_wrap(x, r, p);
}

template<typename Real>
Real
nbinom_cdf_wrap(const Real x, const Real r, const Real p)
{
    // 如果 x 是有限值，则使用 Boost.Math 库计算负二项分布的累积分布函数值
    if (std::isfinite(x)) {
        return boost::math::cdf(
            boost::math::negative_binomial_distribution<Real, StatsPolicy>(r, p), x);
    }
    // 如果 x 是无限值，则返回 0 或 1，表示累积分布函数的边界情况
    // -inf => 0, inf => 1
    return 1 - std::signbit(x);
}

float
nbinom_cdf_float(float x, float r, float p)
{
    // 调用通用模板函数 nbinom_cdf_wrap，处理单精度浮点数参数
    return nbinom_cdf_wrap(x, r, p);
}

double
nbinom_cdf_double(double x, double r, double p)
{
    // 调用通用模板函数 nbinom_cdf_wrap，处理双精度浮点数参数
    return nbinom_cdf_wrap(x, r, p);
}

template<typename Real>
Real
nbinom_ppf_wrap(const Real x, const Real r, const Real p)
{
    // 使用 Boost.Math 库计算负二项分布的分位点函数值
    return boost::math::quantile(
        boost::math::negative_binomial_distribution<Real, StatsPolicy>(r, p), x);
}

float
nbinom_ppf_float(float x, float r, float p)
{
    // 调用通用模板函数 nbinom_ppf_wrap，处理单精度浮点数参数
    return nbinom_ppf_wrap(x, r, p);
}

double
nbinom_ppf_double(double x, double r, double p)
{
    // 调用通用模板函数 nbinom_ppf_wrap，处理双精度浮点数参数
    return nbinom_ppf_wrap(x, r, p);
}

template<typename Real>
Real
nbinom_sf_wrap(const Real x, const Real r, const Real p)
{
    // 使用 Boost.Math 库计算负二项分布的生存函数值
    return boost::math::cdf(boost::math::complement(
        boost::math::negative_binomial_distribution<Real, StatsPolicy>(r, p), x));
}

float
nbinom_sf_float(float x, float r, float p)
{
    // 调用通用模板函数 nbinom_sf_wrap，处理单精度浮点数参数
    return nbinom_sf_wrap(x, r, p);
}

double
nbinom_sf_double(double x, double r, double p)
{
    // 调用通用模板函数 nbinom_sf_wrap，处理双精度浮点数参数
    return nbinom_sf_wrap(x, r, p);
}

template<typename Real>
Real
nbinom_isf_wrap(const Real x, const Real r, const Real p)
{
    // 使用 Boost.Math 库计算负二项分布的逆生存函数值
    return boost::math::quantile(boost::math::complement(
        boost::math::negative_binomial_distribution<Real, StatsPolicy>(r, p), x));
}

float
nbinom_isf_float(float x, float r, float p)
{
    // 调用通用模板函数 nbinom_isf_wrap，处理单精度浮点数参数
    return nbinom_isf_wrap(x, r, p);
}

double
nbinom_isf_double(double x, double r, double p)
{
    // 调用通用模板函数 nbinom_isf_wrap，处理双精度浮点数参数
    return nbinom_isf_wrap(x, r, p);
}
    # 调用 nbinom_isf_wrap 函数，并返回其结果
    return nbinom_isf_wrap(x, r, p);
}

float
nbinom_mean_float(float r, float p)
{
    // 返回负二项分布（浮点型）的均值
    return boost::math::mean(boost::math::negative_binomial_distribution<float, StatsPolicy>(r, p));
}

double
nbinom_mean_double(double r, double p)
{
    // 返回负二项分布（双精度型）的均值
    return boost::math::mean(boost::math::negative_binomial_distribution<double, StatsPolicy>(r, p));
}

float
nbinom_variance_float(float r, float p)
{
    // 返回负二项分布（浮点型）的方差
    return boost::math::variance(boost::math::negative_binomial_distribution<float, StatsPolicy>(r, p));
}

double
nbinom_variance_double(double r, double p)
{
    // 返回负二项分布（双精度型）的方差
    return boost::math::variance(boost::math::negative_binomial_distribution<double, StatsPolicy>(r, p));
}

float
nbinom_skewness_float(float r, float p)
{
    // 返回负二项分布（浮点型）的偏度
    return boost::math::skewness(boost::math::negative_binomial_distribution<float, StatsPolicy>(r, p));
}

double
nbinom_skewness_double(double r, double p)
{
    // 返回负二项分布（双精度型）的偏度
    return boost::math::skewness(boost::math::negative_binomial_distribution<double, StatsPolicy>(r, p));
}

float
nbinom_kurtosis_excess_float(float r, float p)
{
    // 返回负二项分布（浮点型）的峰度（超额）
    return boost::math::kurtosis_excess(boost::math::negative_binomial_distribution<float, StatsPolicy>(r, p));
}

double
nbinom_kurtosis_excess_double(double r, double p)
{
    // 返回负二项分布（双精度型）的峰度（超额）
    return boost::math::kurtosis_excess(boost::math::negative_binomial_distribution<double, StatsPolicy>(r, p));
}

template<typename Real>
Real
hypergeom_pmf_wrap(const Real k, const Real n, const Real N, const Real M)
{
    if (std::isfinite(k)) {
        // 返回超几何分布的概率质量函数值
        return boost::math::pdf(
            boost::math::hypergeometric_distribution<Real, StatsPolicy>(n, N, M), k);
    }
    // 若 k 是无穷，返回 NAN （inf 或 -inf 返回 NAN）
    return NAN;
}

float
hypergeom_pmf_float(float k, float n, float N, float M)
{
    // 返回浮点型超几何分布的概率质量函数值
    return hypergeom_pmf_wrap(k, n, N, M);
}

double
hypergeom_pmf_double(double k, double n, double N, double M)
{
    // 返回双精度型超几何分布的概率质量函数值
    return hypergeom_pmf_wrap(k, n, N, M);
}

template<typename Real>
Real
hypergeom_cdf_wrap(const Real k, const Real n, const Real N, const Real M)
{
    if (std::isfinite(k)) {
        // 返回超几何分布的累积分布函数值
        return boost::math::cdf(
            boost::math::hypergeometric_distribution<Real, StatsPolicy>(n, N, M), k);
    }
    // 若 k 是无穷，按照 -inf => 0, inf => 1 的规则返回值
    return 1 - std::signbit(k);
}

float
hypergeom_cdf_float(float k, float n, float N, float M)
{
    // 返回浮点型超几何分布的累积分布函数值
    return hypergeom_cdf_wrap(k, n, N, M);
}

double
hypergeom_cdf_double(double k, double n, double N, double M)
{
    // 返回双精度型超几何分布的累积分布函数值
    return hypergeom_cdf_wrap(k, n, N, M);
}

template<typename Real>
Real
hypergeom_sf_wrap(const Real k, const Real n, const Real N, const Real M)
{
    // 返回超几何分布的生存函数值
    return boost::math::cdf(boost::math::complement(
        boost::math::hypergeometric_distribution<Real, StatsPolicy>(n, N, M), k));
}

float
hypergeom_sf_float(float k, float n, float N, float M)
{
    // 返回浮点型超几何分布的生存函数值
    return hypergeom_sf_wrap(k, n, N, M);
}

double
hypergeom_sf_double(double k, double n, double N, double M)
{
    // 返回双精度型超几何分布的生存函数值
    return hypergeom_sf_wrap(k, n, N, M);
}

float
hypergeom_mean_float(float n, float N, float M)
{
    // 返回浮点型超几何分布的均值
    return boost::math::mean(boost::math::hypergeometric_distribution<float, StatsPolicy>(n, N, M));
}

double
# 计算双精度参数下超几何分布的均值
double
hypergeom_mean_double(double n, double N, double M)
{
    # 使用 Boost 库计算双精度超几何分布的均值
    return boost::math::mean(boost::math::hypergeometric_distribution<double, StatsPolicy>(n, N, M));
}

# 计算单精度参数下超几何分布的方差
float
hypergeom_variance_float(float n, float N, float M)
{
    # 使用 Boost 库计算单精度超几何分布的方差
    return boost::math::variance(boost::math::hypergeometric_distribution<float, StatsPolicy>(n, N, M));
}

# 计算双精度参数下超几何分布的方差
double
hypergeom_variance_double(double n, double N, double M)
{
    # 使用 Boost 库计算双精度超几何分布的方差
    return boost::math::variance(boost::math::hypergeometric_distribution<double, StatsPolicy>(n, N, M));
}

# 计算单精度参数下超几何分布的偏度
float
hypergeom_skewness_float(float n, float N, float M)
{
    # 使用 Boost 库计算单精度超几何分布的偏度
    return boost::math::skewness(boost::math::hypergeometric_distribution<float, StatsPolicy>(n, N, M));
}

# 计算双精度参数下超几何分布的偏度
double
hypergeom_skewness_double(double n, double N, double M)
{
    # 使用 Boost 库计算双精度超几何分布的偏度
    return boost::math::skewness(boost::math::hypergeometric_distribution<double, StatsPolicy>(n, N, M));
}

#endif
```