# `D:\src\scipysrc\scipy\scipy\stats\_levy_stable\c_src\levyst.c`

```
/*
 * Implements special functions for stable distribution calculations.
 *
 * A function g appears in the integrand in Nolan's method for calculating
 * stable densities and distribution functions. It takes a different form for
 * for alpha = 1 vs alpha ≠ 1. See [NO] for more info.
 *
 * References
 * [NO] John P. Nolan (1997) Numerical calculation of stable densities and
 *      distribution functions.
 */

#define _USE_MATH_DEFINES  // Define to enable math constants like M_PI
#include <math.h>
#include <stdlib.h>
#include "levyst.h"  // Include the header file for the implementation

/* M_PI et al. are not defined in math.h in C99, even with _USE_MATH_DEFINES */
#ifndef M_PI_2
# define M_PI_2  1.57079632679489661923  /* pi/2 */
# define M_1_PI  0.31830988618379067154  /* 1/pi */
# define M_2_PI  0.63661977236758134308  /* 2/pi */
#endif

/* Computes g(theta) for alpha not equal to 1 */
double
g_alpha_ne_one(struct nolan_precanned *sp, double theta)
{
    if (theta == -sp->xi) {  // Check if theta is equal to negative xi
        if (sp->alpha < 1) {
            return 0;  // Return 0 if alpha is less than 1
        }
        else {
            return INFINITY;  // Return infinity if alpha is greater than or equal to 1
        }
    }
    if (theta == M_PI_2) {  // Check if theta is equal to pi/2
        if (sp->alpha < 1) {
            return INFINITY;  // Return infinity if alpha is less than 1
        }
        else {
            return 0;  // Return 0 if alpha is greater than or equal to 1
        }
    }

    double cos_theta = cos(theta);  // Compute cosine of theta
    return (
        sp->zeta_prefactor
        * pow(
            cos_theta
            / sin(sp->alpha_xi + sp->alpha * theta)
            * sp->zeta_offset, sp->alpha_exp)
        * cos(sp->alpha_xi + (sp->alpha - 1) * theta)
        / cos_theta
    );  // Return the computed expression involving prefactors and trigonometric functions
}

/* Computes g(theta) for alpha equal to 1 */
double
g_alpha_eq_one(struct nolan_precanned *sp, double theta)
{
    if (theta == -sp->xi) {  // Check if theta is equal to negative xi
        return 0;  // Return 0 if alpha is equal to 1
    }

    if (theta == M_PI_2) {  // Check if theta is equal to pi/2
        return INFINITY;  // Return infinity if alpha is equal to 1
    }

    return (
        (1 + theta * sp->two_beta_div_pi)
        * exp((sp->pi_div_two_beta + theta) * tan(theta) - sp->x0_div_term)
        / cos(theta)
    );  // Return the computed expression involving prefactors and trigonometric functions
}

/* Precomputes values required for g function to optimize numerical integration */
struct nolan_precanned *
nolan_precan(double alpha, double beta, double x0)
{
    /* Stores results of intermediate computations so they need not be
     * recomputed when g is called many times during numerical integration
     * through QUADPACK.
     */
    struct nolan_precanned *sp = malloc(sizeof(struct nolan_precanned));  // Allocate memory for struct
    if (!sp) {
        abort();  // Abort program if memory allocation fails
    }
    sp->alpha = alpha;  // Assign alpha value to struct member
    sp->zeta = -beta * tan(M_PI_2 * alpha);  // Compute zeta value based on beta and alpha

    if (alpha != 1.) {  // Check if alpha is not equal to 1
        sp->xi = atan(-sp->zeta) / alpha;  // Compute xi value
        sp->zeta_prefactor = pow(
            pow(sp->zeta, 2.) + 1., -1. / (2. * (alpha - 1.)));  // Compute zeta prefactor
        sp->alpha_exp = alpha / (alpha - 1.);  // Compute alpha exponent
        sp->alpha_xi = atan(-sp->zeta);  // Compute alpha xi
        sp->zeta_offset = x0 - sp->zeta;  // Compute zeta offset
        if (alpha < 1.) {
            sp->c1 = 0.5 - sp->xi * M_1_PI;  // Compute constant c1
            sp->c3 = M_1_PI;  // Assign constant c3
        }
        else {
            sp->c1 = 1.;  // Assign constant c1
            sp->c3 = -M_1_PI;  // Assign constant c3
        }
        sp->c2 = alpha * M_1_PI / fabs(alpha - 1.) / (x0 - sp->zeta);  // Compute constant c2
        sp->g = &g_alpha_ne_one;  // Set g function pointer to g_alpha_ne_one
    }
    else {
        // 设置 sp 结构体的 xi 字段为 π/2
        sp->xi = M_PI_2;
        // 设置 sp 结构体的 two_beta_div_pi 字段为 beta 乘以 2/π 的值
        sp->two_beta_div_pi = beta * M_2_PI;
        // 设置 sp 结构体的 pi_div_two_beta 字段为 π/2 除以 beta 的值
        sp->pi_div_two_beta = M_PI_2 / beta;
        // 设置 sp 结构体的 x0_div_term 字段为 x0 除以 two_beta_div_pi 的值
        sp->x0_div_term = x0 / sp->two_beta_div_pi;
        // 设置 sp 结构体的 c1 字段为 0.0
        sp->c1 = 0.;
        // 设置 sp 结构体的 c2 字段为 0.5 除以 beta 绝对值的值
        sp->c2 = .5 / fabs(beta);
        // 设置 sp 结构体的 c3 字段为 1/π 的值
        sp->c3 = M_1_PI;
        // 设置 sp 结构体的 g 字段为指向 g_alpha_eq_one 函数的指针
        sp->g = &g_alpha_eq_one;
    }
    // 返回 sp 结构体指针
    return sp;
}



# 这是一个单独的右大括号 '}'，用于结束一个代码块或数据结构的定义。
```