# `.\numpy\numpy\_core\include\numpy\random\distributions.h`

```py
#ifndef NUMPY_CORE_INCLUDE_NUMPY_RANDOM_DISTRIBUTIONS_H_
#define NUMPY_CORE_INCLUDE_NUMPY_RANDOM_DISTRIBUTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include "numpy/npy_common.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "numpy/npy_math.h"
#include "numpy/random/bitgen.h"

/*
 * RAND_INT_TYPE is used to share integer generators with RandomState which
 * used long in place of int64_t. If changing a distribution that uses
 * RAND_INT_TYPE, then the original unmodified copy must be retained for
 * use in RandomState by copying to the legacy distributions source file.
 */
#ifdef NP_RANDOM_LEGACY
#define RAND_INT_TYPE long
#define RAND_INT_MAX LONG_MAX
#else
#define RAND_INT_TYPE int64_t
#define RAND_INT_MAX INT64_MAX
#endif

#ifdef _MSC_VER
#define DECLDIR __declspec(dllexport)
#else
#define DECLDIR extern
#endif

#ifndef MIN
// 定义一个宏，用于返回两个数中的最小值
#define MIN(x, y) (((x) < (y)) ? x : y)
// 定义一个宏，用于返回两个数中的最大值
#define MAX(x, y) (((x) > (y)) ? x : y)
#endif

#ifndef M_PI
// 如果 M_PI 未定义，则定义为圆周率的数值
#define M_PI 3.14159265358979323846264338328
#endif

// 定义结构体 s_binomial_t，用于存储二项分布的参数
typedef struct s_binomial_t {
  int has_binomial; /* !=0: following parameters initialized for binomial */
  double psave;
  RAND_INT_TYPE nsave;
  double r;
  double q;
  double fm;
  RAND_INT_TYPE m;
  double p1;
  double xm;
  double xl;
  double xr;
  double c;
  double laml;
  double lamr;
  double p2;
  double p3;
  double p4;
} binomial_t;

// 声明以下函数为外部函数，供外部调用
DECLDIR float random_standard_uniform_f(bitgen_t *bitgen_state);
DECLDIR double random_standard_uniform(bitgen_t *bitgen_state);
DECLDIR void random_standard_uniform_fill(bitgen_t *, npy_intp, double *);
DECLDIR void random_standard_uniform_fill_f(bitgen_t *, npy_intp, float *);

DECLDIR int64_t random_positive_int64(bitgen_t *bitgen_state);
DECLDIR int32_t random_positive_int32(bitgen_t *bitgen_state);
DECLDIR int64_t random_positive_int(bitgen_t *bitgen_state);
DECLDIR uint64_t random_uint(bitgen_t *bitgen_state);

DECLDIR double random_standard_exponential(bitgen_t *bitgen_state);
DECLDIR float random_standard_exponential_f(bitgen_t *bitgen_state);
DECLDIR void random_standard_exponential_fill(bitgen_t *, npy_intp, double *);
DECLDIR void random_standard_exponential_fill_f(bitgen_t *, npy_intp, float *);
DECLDIR void random_standard_exponential_inv_fill(bitgen_t *, npy_intp, double *);
DECLDIR void random_standard_exponential_inv_fill_f(bitgen_t *, npy_intp, float *);

DECLDIR double random_standard_normal(bitgen_t *bitgen_state);
DECLDIR float random_standard_normal_f(bitgen_t *bitgen_state);
DECLDIR void random_standard_normal_fill(bitgen_t *, npy_intp, double *);
DECLDIR void random_standard_normal_fill_f(bitgen_t *, npy_intp, float *);
DECLDIR double random_standard_gamma(bitgen_t *bitgen_state, double shape);
DECLDIR float random_standard_gamma_f(bitgen_t *bitgen_state, float shape);

DECLDIR double random_normal(bitgen_t *bitgen_state, double loc, double scale);

DECLDIR double random_gamma(bitgen_t *bitgen_state, double shape, double scale);

#endif /* NUMPY_CORE_INCLUDE_NUMPY_RANDOM_DISTRIBUTIONS_H_ */

#ifdef __cplusplus
}
#endif


注释：

#ifndef NUMPY_CORE_INCLUDE_NUMPY_RANDOM_DISTRIBUTIONS_H_
#define NUMPY_CORE_INCLUDE_NUMPY_RANDOM_DISTRIBUTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>                 // 包含 Python 标准头文件，提供 Python API 支持
#include "numpy/npy_common.h"       // 包含 NumPy 的通用头文件
#include <stddef.h>                 // 包含标准定义头文件，提供大小定义
#include <stdbool.h>                // 包含标准布尔类型头文件
#include <stdint.h>                 // 包含标准整数类型头文件

#include "numpy/npy_math.h"         // 包含 NumPy 的数学函数头文件
#include "numpy/random/bitgen.h"    // 包含 NumPy 随机数生成器位级生成器头文件

/*
 * RAND_INT_TYPE 用于与使用 long 替换 int64_t 的 RandomState 共享整数生成器。
 * 如果更改使用 RAND_INT_TYPE 的分布，则必须保留原始未修改的副本，以供在 RandomState 中使用，通过复制到旧的分布源文件。
 */
#ifdef NP_RANDOM_LEGACY
#define RAND_INT_TYPE long           // 如果定义了 NP_RANDOM_LEGACY，则使用 long 类型
#define RAND_INT_MAX LONG_MAX        // RAND_INT_TYPE 类型的最大值
#else
#define RAND_INT_TYPE int64_t        // 否则使用 int64_t 类型
#define RAND_INT_MAX INT64_MAX       // RAND_INT_TYPE 类型的最大值
#endif

#ifdef _MSC_VER
#define DECLDIR __declspec(dllexport) // 如果是在 Microsoft Visual Studio 下编译，则定义 DECLDIR 为 dllexport
#else
#define DECLDIR extern                // 否则定义 DECLDIR 为 extern
#endif

#ifndef MIN
// 定义 MIN 宏，返回两个数中的最小值
#define MIN(x, y) (((x) < (y)) ? x : y)
// 定义 MAX 宏，返回两个数中的最大值
#define MAX(x, y) (((x) > (y)) ? x : y)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328 // 如果 M_PI 未定义，则定义为圆周率的数值
#endif

// 定义结构体 s_binomial_t，用于存储二项分布的参数
typedef struct s_binomial_t {
  int has_binomial;   // 是否初始化为二项分布的标志，非0表示已初始化
  double psave;       // 保存的概率值
  RAND_INT_TYPE nsave; // 保存的整数值
  double r;
  double q;
  double fm;
  RAND_INT_TYPE m;
  double p1;
  double xm;
  double xl;
  double xr;
  double c;
  double laml;
  double lamr;
  double p2;
  double p3;
  double p4;
} binomial_t;

// 声明以下函数为外部函数，供外部调用
DECLDIR float random_standard_uniform_f(bitgen_t *bitgen_state); // 声明随机标准均匀分布的浮点数版本函数
DECLDIR double random_standard_uniform(bitgen_t *bitgen_state);   // 声明随机标准均匀分布的双精度版本函数
DECLDIR void random_standard_uniform_fill(bitgen_t *, npy_intp, double *);  // 声明填充数组的随机标准均匀分布函数
DECLDIR void random_standard_uniform_fill_f(bitgen_t *, npy_intp, float *);  // 声明填充数组的随机标准均匀分布的浮点数版本函数

DECLDIR int64_t random_positive_int64(bitgen_t *bitgen_state);   // 声明生成正整数（int64_t类型）的函数
DECLDIR int32_t random_positive_int32(bitgen_t *bitgen_state);   // 声明生成正整数（int32_t类型）的函数
DECLDIR int64_t random_positive_int(bitgen_t *bitgen_state);     // 声明生成正整数（int64_t类型）的函数
DECLDIR uint64_t random_uint(bitgen_t *bitgen_state);            // 声明生成无符号整数（uint64_t类型）的函数

DECLDIR double random_standard_exponential(bitgen_t *bitgen_state);   // 声明随机标准指数分布的双精度版本函数
DECLDIR float random_standard_exponential_f(bitgen_t *bitgen_state);  // 声明随机标准指数分布的浮点数版本函数
DECLDIR void random_standard_exponential_fill(bitgen_t *, npy_intp, double *);  // 声明填充数组的随机标准指数分布函数
DECLDIR void random_standard_exponential_fill_f(bitgen_t *, npy_intp,
// 声明一个返回 float 类型的函数 random_gamma_f，接受一个指向 bitgen_t 结构体的指针和两个 float 类型的参数
DECLDIR float random_gamma_f(bitgen_t *bitgen_state, float shape, float scale);

// 声明一个返回 double 类型的函数 random_exponential，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR double random_exponential(bitgen_t *bitgen_state, double scale);

// 声明一个返回 double 类型的函数 random_uniform，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_uniform(bitgen_t *bitgen_state, double lower, double range);

// 声明一个返回 double 类型的函数 random_beta，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_beta(bitgen_t *bitgen_state, double a, double b);

// 声明一个返回 double 类型的函数 random_chisquare，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR double random_chisquare(bitgen_t *bitgen_state, double df);

// 声明一个返回 double 类型的函数 random_f，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_f(bitgen_t *bitgen_state, double dfnum, double dfden);

// 声明一个返回 double 类型的函数 random_standard_cauchy，接受一个指向 bitgen_t 结构体的指针
DECLDIR double random_standard_cauchy(bitgen_t *bitgen_state);

// 声明一个返回 double 类型的函数 random_pareto，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR double random_pareto(bitgen_t *bitgen_state, double a);

// 声明一个返回 double 类型的函数 random_weibull，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR double random_weibull(bitgen_t *bitgen_state, double a);

// 声明一个返回 double 类型的函数 random_power，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR double random_power(bitgen_t *bitgen_state, double a);

// 声明一个返回 double 类型的函数 random_laplace，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_laplace(bitgen_t *bitgen_state, double loc, double scale);

// 声明一个返回 double 类型的函数 random_gumbel，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_gumbel(bitgen_t *bitgen_state, double loc, double scale);

// 声明一个返回 double 类型的函数 random_logistic，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_logistic(bitgen_t *bitgen_state, double loc, double scale);

// 声明一个返回 double 类型的函数 random_lognormal，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_lognormal(bitgen_t *bitgen_state, double mean, double sigma);

// 声明一个返回 double 类型的函数 random_rayleigh，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR double random_rayleigh(bitgen_t *bitgen_state, double mode);

// 声明一个返回 double 类型的函数 random_standard_t，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR double random_standard_t(bitgen_t *bitgen_state, double df);

// 声明一个返回 double 类型的函数 random_noncentral_chisquare，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_noncentral_chisquare(bitgen_t *bitgen_state, double df, double nonc);

// 声明一个返回 double 类型的函数 random_noncentral_f，接受一个指向 bitgen_t 结构体的指针和三个 double 类型的参数
DECLDIR double random_noncentral_f(bitgen_t *bitgen_state, double dfnum, double dfden, double nonc);

// 声明一个返回 double 类型的函数 random_wald，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_wald(bitgen_t *bitgen_state, double mean, double scale);

// 声明一个返回 double 类型的函数 random_vonmises，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR double random_vonmises(bitgen_t *bitgen_state, double mu, double kappa);

// 声明一个返回 double 类型的函数 random_triangular，接受一个指向 bitgen_t 结构体的指针和三个 double 类型的参数
DECLDIR double random_triangular(bitgen_t *bitgen_state, double left, double mode, double right);

// 声明一个返回 RAND_INT_TYPE 类型的函数 random_poisson，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR RAND_INT_TYPE random_poisson(bitgen_t *bitgen_state, double lam);

// 声明一个返回 RAND_INT_TYPE 类型的函数 random_negative_binomial，接受一个指向 bitgen_t 结构体的指针和两个 double 类型的参数
DECLDIR RAND_INT_TYPE random_negative_binomial(bitgen_t *bitgen_state, double n, double p);

// 声明一个返回 int64_t 类型的函数 random_binomial，接受一个指向 bitgen_t 结构体的指针、一个 double 类型的参数、一个 int64_t 类型的参数和一个指向 binomial_t 结构体的指针
DECLDIR int64_t random_binomial(bitgen_t *bitgen_state, double p, int64_t n, binomial_t *binomial);

// 声明一个返回 int64_t 类型的函数 random_logseries，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR int64_t random_logseries(bitgen_t *bitgen_state, double p);

// 声明一个返回 int64_t 类型的函数 random_geometric，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR int64_t random_geometric(bitgen_t *bitgen_state, double p);

// 声明一个返回 RAND_INT_TYPE 类型的函数 random_geometric_search，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR RAND_INT_TYPE random_geometric_search(bitgen_t *bitgen_state, double p);

// 声明一个返回 RAND_INT_TYPE 类型的函数 random_zipf，接受一个指向 bitgen_t 结构体的指针和一个 double 类型的参数
DECLDIR RAND_INT_TYPE random_zipf(bitgen_t *bitgen_state, double a);

// 声明一个返回 int64_t 类型的函数 random_hypergeometric，接受一个指向 bitgen_t 结构体的指针和三个 int64_t 类型的参数
DECLDIR int64_t random_hypergeometric(bitgen_t *bitgen_state, int64_t good, int64_t bad, int64_t sample);

// 声明一个返回 uint64_t 类型的函数 random_interval，接受一个指向 bitgen_t 结构体的指针和一个 uint64_t 类型的参数
DECLDIR uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max);

// 声明一个返回 uint64_t 类型的函数 random_bounded_uint64，接受一个指向 bitgen_t 结构体的指针和四个 uint64_t 类型的参数以及一个 bool 类型的参数
DECLDIR uint64_t random_bounded_uint64(bitgen_t *bitgen_state, uint64_t off, uint64_t rng, uint64_t mask, bool use_masked);
# 声明一个返回 uint32_t 类型的函数，用于生成有界的随机数
DECLDIR uint32_t random_buffered_bounded_uint32(bitgen_t *bitgen_state,
                                                uint32_t off, uint32_t rng,
                                                uint32_t mask, bool use_masked,
                                                int *bcnt, uint32_t *buf);

# 声明一个返回 uint16_t 类型的函数，用于生成有界的随机数
DECLDIR uint16_t random_buffered_bounded_uint16(bitgen_t *bitgen_state,
                                                uint16_t off, uint16_t rng,
                                                uint16_t mask, bool use_masked,
                                                int *bcnt, uint32_t *buf);

# 声明一个返回 uint8_t 类型的函数，用于生成有界的随机数
DECLDIR uint8_t random_buffered_bounded_uint8(bitgen_t *bitgen_state, uint8_t off,
                                              uint8_t rng, uint8_t mask,
                                              bool use_masked, int *bcnt,
                                              uint32_t *buf);

# 声明一个返回 npy_bool 类型的函数，用于生成有界的随机布尔值
DECLDIR npy_bool random_buffered_bounded_bool(bitgen_t *bitgen_state, npy_bool off,
                                              npy_bool rng, npy_bool mask,
                                              bool use_masked, int *bcnt,
                                              uint32_t *buf);

# 声明一个 void 类型的函数，用于填充一个 uint64_t 类型的数组，生成有界的随机数
DECLDIR void random_bounded_uint64_fill(bitgen_t *bitgen_state, uint64_t off,
                                        uint64_t rng, npy_intp cnt,
                                        bool use_masked, uint64_t *out);

# 声明一个 void 类型的函数，用于填充一个 uint32_t 类型的数组，生成有界的随机数
DECLDIR void random_bounded_uint32_fill(bitgen_t *bitgen_state, uint32_t off,
                                        uint32_t rng, npy_intp cnt,
                                        bool use_masked, uint32_t *out);

# 声明一个 void 类型的函数，用于填充一个 uint16_t 类型的数组，生成有界的随机数
DECLDIR void random_bounded_uint16_fill(bitgen_t *bitgen_state, uint16_t off,
                                        uint16_t rng, npy_intp cnt,
                                        bool use_masked, uint16_t *out);

# 声明一个 void 类型的函数，用于填充一个 uint8_t 类型的数组，生成有界的随机数
DECLDIR void random_bounded_uint8_fill(bitgen_t *bitgen_state, uint8_t off,
                                       uint8_t rng, npy_intp cnt,
                                       bool use_masked, uint8_t *out);

# 声明一个 void 类型的函数，用于填充一个 npy_bool 类型的数组，生成有界的随机布尔值
DECLDIR void random_bounded_bool_fill(bitgen_t *bitgen_state, npy_bool off,
                                      npy_bool rng, npy_intp cnt,
                                      bool use_masked, npy_bool *out);

# 声明一个 void 类型的函数，用于生成多项式分布的随机数
DECLDIR void random_multinomial(bitgen_t *bitgen_state, RAND_INT_TYPE n, RAND_INT_TYPE *mnix,
                                double *pix, npy_intp d, binomial_t *binomial);

# 声明一个 int 类型的函数，用于生成多元超几何分布的随机数（使用"count"方法）
DECLDIR int random_multivariate_hypergeometric_count(bitgen_t *bitgen_state,
                              int64_t total,
                              size_t num_colors, int64_t *colors,
                              int64_t nsample,
                              size_t num_variates, int64_t *variates);

# 声明一个多元超几何分布的随机数生成函数（使用"marginals"方法），返回 int 类型
DECLDIR int random_multivariate_hypergeometric_marginals(bitgen_t *bitgen_state,
                              int64_t total,
                              size_t num_colors, int64_t *colors,
                              int64_t nsample,
                              size_t num_variates, int64_t *variates);
/* 声明一个函数 random_multivariate_hypergeometric_marginals，无返回值，可能使用 bitgen_t 类型的状态，接受如下参数：
   - bitgen_state: 位生成器状态的指针
   - total: 整数，表示总数
   - num_colors: 大小类型，表示颜色的数量
   - colors: 整数的指针，表示颜色数组
   - nsample: 整数，表示样本数
   - num_variates: 大小类型，表示变量的数量
   - variates: 整数的指针，表示变量数组 */
DECLDIR void random_multivariate_hypergeometric_marginals(bitgen_t *bitgen_state,
                                   int64_t total,
                                   size_t num_colors, int64_t *colors,
                                   int64_t nsample,
                                   size_t num_variates, int64_t *variates);

/* random_binomial_btpe 函数的声明，该函数在 legacy-distributions.c 和 distributions.c 中使用，
   但不会被导出 */

/* 使用 bitgen_state 位生成器状态，接受如下参数：
   - bitgen_state: 位生成器状态的指针
   - n: 随机整数的类型
   - p: 双精度浮点数，表示概率
   - binomial: 二项分布结构体指针 */
RAND_INT_TYPE random_binomial_btpe(bitgen_t *bitgen_state,
                                   RAND_INT_TYPE n,
                                   double p,
                                   binomial_t *binomial);

/* random_binomial_inversion 函数的声明，该函数在 legacy-distributions.c 和 distributions.c 中使用 */

/* 使用 bitgen_state 位生成器状态，接受如下参数：
   - bitgen_state: 位生成器状态的指针
   - n: 随机整数的类型
   - p: 双精度浮点数，表示概率
   - binomial: 二项分布结构体指针 */
RAND_INT_TYPE random_binomial_inversion(bitgen_t *bitgen_state,
                                        RAND_INT_TYPE n,
                                        double p,
                                        binomial_t *binomial);

/* random_loggam 函数的声明 */

/* 接受一个双精度浮点数 x 作为参数 */
double random_loggam(double x);

/* 内联函数 next_double 的声明，返回一个双精度浮点数 */

/* 使用 bitgen_state 位生成器状态，接受如下参数：
   - bitgen_state: 位生成器状态的指针 */
static inline double next_double(bitgen_t *bitgen_state) {
    return bitgen_state->next_double(bitgen_state->state);
}

#ifdef __cplusplus
}
#endif

/* 结束条件编译的 endif 指令 */

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_RANDOM_DISTRIBUTIONS_H_ */
```