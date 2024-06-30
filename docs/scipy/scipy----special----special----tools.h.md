# `D:\src\scipysrc\scipy\scipy\special\special\tools.h`

```
/* Building blocks for implementing special functions */

#pragma once

#include "config.h"
#include "error.h"

namespace special {
namespace detail {

    /* Result type of a "generator", a callable object that produces a value
     * each time it is called.
     */
    template <typename Generator>
    using generator_result_t = std::decay_t<std::invoke_result_t<Generator>>;

    /* Used to deduce the type of the numerator/denominator of a fraction. */
    template <typename Pair>
    struct pair_traits;

    template <typename T>
    struct pair_traits<std::pair<T, T>> {
        using value_type = T;
    };

    template <typename Pair>
    using pair_value_t = typename pair_traits<Pair>::value_type;

    /* Used to extract the "value type" of a complex type. */
    template <typename T>
    struct real_type {
        using type = T;
    };

    template <typename T>
    struct real_type<std::complex<T>> {
        using type = T;
    };

    template <typename T>
    using real_type_t = typename real_type<T>::type;

    // Return NaN, handling both real and complex types.
    template <typename T>
    SPECFUN_HOST_DEVICE inline std::enable_if_t<std::is_floating_point_v<T>, T> maybe_complex_NaN() {
        return std::numeric_limits<T>::quiet_NaN();
    }

    template <typename T>
    SPECFUN_HOST_DEVICE inline std::enable_if_t<!std::is_floating_point_v<T>, T> maybe_complex_NaN() {
        using V = typename T::value_type;
        return {std::numeric_limits<V>::quiet_NaN(), std::numeric_limits<V>::quiet_NaN()};
    }

    // Series evaluators.
    template <typename Generator, typename T = generator_result_t<Generator>>
    SPECFUN_HOST_DEVICE T series_eval(Generator &g, T init_val, real_type_t<T> tol, std::uint64_t max_terms,
                                      const char *func_name) {
        /* Sum an infinite series to a given precision.
         *
         * g : a generator of terms for the series.
         *
         * init_val : A starting value that terms are added to. This argument determines the
         *     type of the result.
         *
         * tol : relative tolerance for stopping criterion.
         *
         * max_terms : The maximum number of terms to add before giving up and declaring
         *     non-convergence.
         *
         * func_name : The name of the function within SciPy where this call to series_eval
         *     will ultimately be used. This is needed to pass to set_error in case
         *     of non-convergence.
         */
        T result = init_val;  // Initialize the result with the initial value
        T term;  // Variable to store each term of the series
        for (std::uint64_t i = 0; i < max_terms; ++i) {  // Loop through max_terms iterations
            term = g();  // Generate the next term in the series
            result += term;  // Add the term to the current result
            if (std::abs(term) < std::abs(result) * tol) {  // Check if the term is sufficiently small relative to the result
                return result;  // If convergence criteria met, return the result
            }
        }
        // Exceeded max terms without converging. Set error and return NaN.
        set_error(func_name, SF_ERROR_NO_RESULT, NULL);  // Set error code indicating non-convergence
        return maybe_complex_NaN<T>();  // Return NaN value appropriate to the type T
    }
    /* Sum a fixed number of terms from a series.
     *
     * g : a generator of terms for the series.
     *
     * init_val : A starting value that terms are added to. This argument determines the
     *     type of the result.
     *
     * num_terms : The number of terms from the series to sum.
     *
     */
    template <typename Generator, typename T = generator_result_t<Generator>>
    SPECFUN_HOST_DEVICE T series_eval_fixed_length(Generator &g, T init_val, std::uint64_t num_terms) {
        // Initialize the result with the initial value
        T result = init_val;
        // Iterate over num_terms and accumulate values from the generator
        for (std::uint64_t i = 0; i < num_terms; ++i) {
            result += g();
        }
        // Return the accumulated result
        return result;
    }

    /* Performs one step of Kahan summation. */
    template <typename T>
    SPECFUN_HOST_DEVICE void kahan_step(T& sum, T& comp, T x) {
        // Compute the correction term y
        T y = x - comp;
        // Update the tentative sum
        T t = sum + y;
        // Update the compensation term
        comp = (t - sum) - y;
        // Update the sum
        sum = t;
    }

    /* Evaluates an infinite series using Kahan summation.
     *
     * Denote the series by
     *
     *   S = a[0] + a[1] + a[2] + ...
     *
     * And for n = 0, 1, 2, ..., denote its n-th partial sum by
     *
     *   S[n] = a[0] + a[1] + ... + a[n]
     *
     * This function computes S[0], S[1], ... until a[n] is sufficiently
     * small or if the maximum number of terms have been evaluated.
     *
     * Parameters
     * ----------
     *   g
     *       Reference to generator that yields the sequence of values a[1],
     *       a[2], a[3], ...
     *
     *   tol
     *       Relative tolerance for convergence.  Specifically, stop iteration
     *       as soon as `abs(a[n]) <= tol * abs(S[n])` for some n >= 1.
     *
     *   max_terms
     *       Maximum number of terms after a[0] to evaluate.  It should be set
     *       large enough such that the convergence criterion is guaranteed
     *       to have been satisfied within that many terms if there is no
     *       rounding error.
     *
     *   init_val
     *       a[0].  Default is zero.  The type of this parameter (T) is used
     *       for intermediary computations as well as the result.
     *
     * Return Value
     * ------------
     * If the convergence criterion is satisfied by some `n <= max_terms`,
     * returns `(S[n], n)`.  Otherwise, returns `(S[max_terms], 0)`.
     */
    template <typename Generator, typename T = generator_result_t<Generator>>
    SPECFUN_HOST_DEVICE std::pair<T, std::uint64_t> series_eval_kahan(
        Generator &&g, real_type_t<T> tol, std::uint64_t max_terms, T init_val = T(0)) {

        // Initialize sum with the initial value
        T sum = init_val;
        // Initialize compensation term with zero
        T comp = 0;
        // Iterate up to max_terms to evaluate the series
        for (std::uint64_t i = 0; i < max_terms; ++i) {
            // Get the next term from the generator
            T term = g();
            // Perform one step of Kahan summation
            kahan_step(sum, comp, term);
            // Check the convergence criterion
            if (std::abs(term) <= tol * std::abs(sum)) {
                // Return the current sum and the number of terms used
                return {sum, i + 1};
            }
        }
        // If max_terms is reached without satisfying the criterion, return the current sum and 0
        return {sum, 0};
    }
    /* Generator that yields the difference of successive convergents of a
     * continued fraction.
     *
     * Let f[n] denote the n-th convergent of a continued fraction:
     *
     *                 a[1]   a[2]       a[n]
     *   f[n] = b[0] + ------ ------ ... ----
     *                 b[1] + b[2] +     b[n]
     *
     * with f[0] = b[0].  This generator yields the sequence of values
     * f[1]-f[0], f[2]-f[1], f[3]-f[2], ...
     *
     * Constructor Arguments
     * ---------------------
     *   cf
     *       Reference to generator that yields the terms of the continued
     *       fraction as (numerator, denominator) pairs, starting from
     *       (a[1], b[1]).
     *
     *       `cf` must outlive the ContinuedFractionSeriesGenerator object.
     *
     *       The constructed object always eagerly retrieves the next term
     *       of the continued fraction.  Specifically, (a[1], b[1]) is
     *       retrieved upon construction, and (a[n], b[n]) is retrieved after
     *       (n-1) calls of `()`.
     *
     * Type Arguments
     * --------------
     *   T
     *       Type in which computations are performed and results are turned.
     *
     * Remarks
     * -------
     * The series is computed using the recurrence relation described in [1].
     *
     * No error checking is performed.  The caller must ensure that all terms
     * are finite and that intermediary computations do not trigger floating
     * point exceptions such as overflow.
     *
     * The numerical stability of this method depends on the characteristics
     * of the continued fraction being evaluated.
     *
     * Reference
     * ---------
     * [1] Gautschi, W. (1967). “Computational Aspects of Three-Term
     *     Recurrence Relations.” SIAM Review, 9(1):24-82.
     */
    template <typename Generator, typename T = pair_value_t<generator_result_t<Generator>>>
    class ContinuedFractionSeriesGenerator {

    public:
        // Constructor initializing with a reference to the continued fraction generator
        explicit ContinuedFractionSeriesGenerator(Generator &cf) : cf_(cf) {
            init(); // Initialize internal state using the first terms of the continued fraction
        }

        // Function call operator to retrieve the next value in the sequence
        double operator()() {
            double v = v_; // Store current value v[n]
            advance(); // Advance to the next value in the sequence
            return v; // Return the stored current value v[n]
        }

    private:
        // Initialize internal state with the first terms of the continued fraction
        void init() {
            auto [num, denom] = cf_(); // Retrieve numerator and denominator of the continued fraction
            T a = num; // Store numerator as type T
            T b = denom; // Store denominator as type T
            u_ = T(1); // Initialize u[1] = 1
            v_ = a / b; // Initialize v[1] = a[1] / b[1]
            b_ = b; // Store the last denominator b[0]
        }

        // Compute the next value in the sequence using the recurrence relation
        void advance() {
            auto [num, denom] = cf_(); // Retrieve next numerator and denominator
            T a = num; // Store next numerator as type T
            T b = denom; // Store next denominator as type T
            u_ = T(1) / (T(1) + (a * u_) / (b * b_)); // Update u[n] using the recurrence relation
            v_ *= (u_ - T(1)); // Update v[n] = f[n] - f[n-1] using u[n] and v[n-1]
            b_ = b; // Update the last denominator b[n-1]
        }

        Generator& cf_; // Reference to the generator for continued fraction terms
        T v_; // Difference of successive convergents f[n] - f[n-1], n >= 1
        T u_; // Sequence u[n] = v[n] / v[n-1], with u[1] = 1
        T b_; // Last denominator b[n-1] used in the computation
    };
    /* 将一个连分数转换为一系列项，这些项是其连分数收敛项的差值序列。
     *
     * 详见 ContinuedFractionSeriesGenerator 的详细说明。
     */
    template <typename Generator, typename T = pair_value_t<generator_result_t<Generator>>>
    SPECFUN_HOST_DEVICE ContinuedFractionSeriesGenerator<Generator, T>
    continued_fraction_series(Generator &cf) {
        // 返回一个 ContinuedFractionSeriesGenerator 对象，用给定的连分数生成器初始化
        return ContinuedFractionSeriesGenerator<Generator, T>(cf);
    }
} // 结束 detail 命名空间的定义
} // 结束 special 命名空间的定义
```