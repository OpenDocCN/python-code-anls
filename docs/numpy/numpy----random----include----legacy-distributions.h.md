# `.\numpy\numpy\random\include\legacy-distributions.h`

```py
#ifndef _RANDOMDGEN__DISTRIBUTIONS_LEGACY_H_
#define _RANDOMDGEN__DISTRIBUTIONS_LEGACY_H_

// 包含 numpy 随机数分布头文件
#include "numpy/random/distributions.h"

// 定义增强型比特生成器结构体
typedef struct aug_bitgen {
  bitgen_t *bit_generator; // 指向比特生成器的指针
  int has_gauss;           // 是否已生成高斯数标志
  double gauss;            // 存储上一个生成的高斯数值
} aug_bitgen_t;

// 下面是各种随机数分布的函数声明

// 产生服从正态分布的随机数
extern double legacy_gauss(aug_bitgen_t *aug_state);

// 产生标准指数分布的随机数
extern double legacy_standard_exponential(aug_bitgen_t *aug_state);

// 产生帕累托分布的随机数
extern double legacy_pareto(aug_bitgen_t *aug_state, double a);

// 产生威布尔分布的随机数
extern double legacy_weibull(aug_bitgen_t *aug_state, double a);

// 产生幂律分布的随机数
extern double legacy_power(aug_bitgen_t *aug_state, double a);

// 产生伽马分布的随机数
extern double legacy_gamma(aug_bitgen_t *aug_state, double shape, double scale);

// 产生卡方分布的随机数
extern double legacy_chisquare(aug_bitgen_t *aug_state, double df);

// 产生雷利分布的随机数
extern double legacy_rayleigh(bitgen_t *bitgen_state, double mode);

// 产生非中心卡方分布的随机数
extern double legacy_noncentral_chisquare(aug_bitgen_t *aug_state, double df,
                                          double nonc);

// 产生非中心 F 分布的随机数
extern double legacy_noncentral_f(aug_bitgen_t *aug_state, double dfnum,
                                  double dfden, double nonc);

// 产生瓦尔德分布的随机数
extern double legacy_wald(aug_bitgen_t *aug_state, double mean, double scale);

// 产生对数正态分布的随机数
extern double legacy_lognormal(aug_bitgen_t *aug_state, double mean,
                               double sigma);

// 产生标准 t 分布的随机数
extern double legacy_standard_t(aug_bitgen_t *aug_state, double df);

// 产生标准柯西分布的随机数
extern double legacy_standard_cauchy(aug_bitgen_t *state);

// 产生贝塔分布的随机数
extern double legacy_beta(aug_bitgen_t *aug_state, double a, double b);

// 产生 F 分布的随机数
extern double legacy_f(aug_bitgen_t *aug_state, double dfnum, double dfden);

// 产生正态分布的随机数
extern double legacy_normal(aug_bitgen_t *aug_state, double loc, double scale);

// 产生标准伽马分布的随机数
extern double legacy_standard_gamma(aug_bitgen_t *aug_state, double shape);

// 产生指数分布的随机数
extern double legacy_exponential(aug_bitgen_t *aug_state, double scale);

// 产生冯·米塞斯分布的随机数
extern double legacy_vonmises(bitgen_t *bitgen_state, double mu, double kappa);

// 产生二项分布的随机数
extern int64_t legacy_random_binomial(bitgen_t *bitgen_state, double p,
                                      int64_t n, binomial_t *binomial);

// 产生负二项分布的随机数
extern int64_t legacy_negative_binomial(aug_bitgen_t *aug_state, double n,
                                        double p);

// 产生超几何分布的随机数
extern int64_t legacy_random_hypergeometric(bitgen_t *bitgen_state,
                                            int64_t good, int64_t bad,
                                            int64_t sample);

// 产生对数级数分布的随机数
extern int64_t legacy_logseries(bitgen_t *bitgen_state, double p);

// 产生泊松分布的随机数
extern int64_t legacy_random_poisson(bitgen_t *bitgen_state, double lam);

// 产生 Zipf 分布的随机数
extern int64_t legacy_random_zipf(bitgen_t *bitgen_state, double a);

// 产生几何分布的随机数
extern int64_t legacy_random_geometric(bitgen_t *bitgen_state, double p);

// 多项式分布的随机数生成函数声明
void legacy_random_multinomial(bitgen_t *bitgen_state, RAND_INT_TYPE n,
                               RAND_INT_TYPE *mnix, double *pix, npy_intp d,
                               binomial_t *binomial);

#endif // _RANDOMDGEN__DISTRIBUTIONS_LEGACY_H_
```