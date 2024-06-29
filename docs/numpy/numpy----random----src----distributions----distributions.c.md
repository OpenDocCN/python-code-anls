# `.\numpy\numpy\random\src\distributions\distributions.c`

```
/* 包含头文件 numpy/random/distributions.h */
#include "numpy/random/distributions.h"
/* 包含头文件 ziggurat_constants.h */
#include "ziggurat_constants.h"
/* 包含头文件 logfactorial.h */
#include "logfactorial.h"

/* 如果编译器是 MSC 并且是 64 位 Windows */
#if defined(_MSC_VER) && defined(_WIN64)
/* 包含内置函数头文件 */
#include <intrin.h>
#endif

/* 包含标准断言头文件 */
#include <assert.h>

/* 内联函数定义，生成内部使用的 uint32_t 类型随机数 */
static inline uint32_t next_uint32(bitgen_t *bitgen_state) {
  return bitgen_state->next_uint32(bitgen_state->state);
}

/* 内联函数定义，生成内部使用的 uint64_t 类型随机数 */
static inline uint64_t next_uint64(bitgen_t *bitgen_state) {
  return bitgen_state->next_uint64(bitgen_state->state);
}

/* 内联函数定义，生成内部使用的 float 类型随机数 */
static inline float next_float(bitgen_t *bitgen_state) {
  return (next_uint32(bitgen_state) >> 8) * (1.0f / 16777216.0f);
}

/* 外部使用的随机数生成器，生成标准均匀分布的 float 类型随机数 */
float random_standard_uniform_f(bitgen_t *bitgen_state) {
    return next_float(bitgen_state);
}

/* 外部使用的随机数生成器，生成标准均匀分布的 double 类型随机数 */
double random_standard_uniform(bitgen_t *bitgen_state) {
    return next_double(bitgen_state);
}

/* 外部使用的随机数生成器，将标准均匀分布的 double 类型随机数填充到数组中 */
void random_standard_uniform_fill(bitgen_t *bitgen_state, npy_intp cnt, double *out) {
  npy_intp i;
  for (i = 0; i < cnt; i++) {
    out[i] = next_double(bitgen_state);
  }
}

/* 外部使用的随机数生成器，将标准均匀分布的 float 类型随机数填充到数组中 */
void random_standard_uniform_fill_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) {
  npy_intp i;
  for (i = 0; i < cnt; i++) {
    out[i] = next_float(bitgen_state);
  }
}

/* 内部静态函数定义，生成不常见的指数分布随机数 */
static double standard_exponential_unlikely(bitgen_t *bitgen_state,
                                                uint8_t idx, double x) {
  if (idx == 0) {
    /* 切换到 1.0 - U 以避免 log(0.0)，参见 GitHub 13361 */
    return ziggurat_exp_r - npy_log1p(-next_double(bitgen_state));
  } else if ((fe_double[idx - 1] - fe_double[idx]) * next_double(bitgen_state) +
                 fe_double[idx] <
             exp(-x)) {
    return x;
  } else {
    return random_standard_exponential(bitgen_state);
  }
}

/* 外部使用的随机数生成器，生成标准指数分布的 double 类型随机数 */
double random_standard_exponential(bitgen_t *bitgen_state) {
  uint64_t ri;
  uint8_t idx;
  double x;
  ri = next_uint64(bitgen_state);
  ri >>= 3;
  idx = ri & 0xFF;
  ri >>= 8;
  x = ri * we_double[idx];
  if (ri < ke_double[idx]) {
    return x; /* 98.9% 的时间我们在第一次尝试时返回这里 */
  }
  return standard_exponential_unlikely(bitgen_state, idx, x);
}

/* 外部使用的随机数生成器，将标准指数分布的 double 类型随机数填充到数组中 */
void random_standard_exponential_fill(bitgen_t * bitgen_state, npy_intp cnt, double * out)
{
  npy_intp i;
  for (i = 0; i < cnt; i++) {
    out[i] = random_standard_exponential(bitgen_state);
  }
}

/* 内部静态函数定义，生成不常见的指数分布随机数 */
static float standard_exponential_unlikely_f(bitgen_t *bitgen_state,
                                                 uint8_t idx, float x) {
  if (idx == 0) {
    /* 切换到 1.0 - U 以避免 log(0.0)，参见 GitHub 13361 */
    return ziggurat_exp_r_f - npy_log1pf(-next_float(bitgen_state));
  } else if ((fe_float[idx - 1] - fe_float[idx]) * next_float(bitgen_state) +
                 fe_float[idx] <
             expf(-x)) {
    return x;
  } else {
    return random_standard_exponential_f(bitgen_state);
  }
}
float random_standard_exponential_f(bitgen_t *bitgen_state) {
  uint32_t ri;  /* 声明一个无符号32位整数变量 ri */
  uint8_t idx;   /* 声明一个无符号8位整数变量 idx */
  float x;       /* 声明一个单精度浮点数变量 x */
  ri = next_uint32(bitgen_state);  /* 调用函数 next_uint32 获取一个随机的32位无符号整数并赋值给 ri */
  ri >>= 1;      /* 将 ri 右移1位 */
  idx = ri & 0xFF;  /* 将 ri 的低8位保存到 idx 中 */
  ri >>= 8;      /* 将 ri 右移8位 */
  x = ri * we_float[idx];  /* 计算 x = ri * we_float[idx]，其中 we_float 是一个预先定义的数组 */
  if (ri < ke_float[idx]) {  /* 如果 ri 小于 ke_float[idx]，返回 x */
    return x; /* 98.9% of the time we return here 1st try */
  }
  return standard_exponential_unlikely_f(bitgen_state, idx, x);  /* 否则调用函数 standard_exponential_unlikely_f 处理 */
}

void random_standard_exponential_fill_f(bitgen_t * bitgen_state, npy_intp cnt, float * out)
{
  npy_intp i;  /* 声明一个用于循环的变量 i */
  for (i = 0; i < cnt; i++) {  /* 循环 cnt 次 */
    out[i] = random_standard_exponential_f(bitgen_state);  /* 调用函数 random_standard_exponential_f 获取一个随机指数分布数值并存入 out 数组 */
  }
}

void random_standard_exponential_inv_fill(bitgen_t * bitgen_state, npy_intp cnt, double * out)
{
  npy_intp i;  /* 声明一个用于循环的变量 i */
  for (i = 0; i < cnt; i++) {  /* 循环 cnt 次 */
    out[i] = -npy_log1p(-next_double(bitgen_state));  /* 使用 next_double 获取一个随机的双精度浮点数，计算其负对数并存入 out 数组 */
  }
}

void random_standard_exponential_inv_fill_f(bitgen_t * bitgen_state, npy_intp cnt, float * out)
{
  npy_intp i;  /* 声明一个用于循环的变量 i */
  for (i = 0; i < cnt; i++) {  /* 循环 cnt 次 */
    out[i] = -npy_log1p(-next_float(bitgen_state));  /* 使用 next_float 获取一个随机的单精度浮点数，计算其负对数并存入 out 数组 */
  }
}


double random_standard_normal(bitgen_t *bitgen_state) {
  uint64_t r;   /* 声明一个无符号64位整数变量 r */
  int sign;      /* 声明一个整数变量 sign */
  uint64_t rabs;  /* 声明一个无符号64位整数变量 rabs */
  int idx;        /* 声明一个整数变量 idx */
  double x, xx, yy;  /* 声明三个双精度浮点数变量 x, xx, yy */
  for (;;) {   /* 无限循环 */
    /* r = e3n52sb8 */
    r = next_uint64(bitgen_state);  /* 调用函数 next_uint64 获取一个随机的64位无符号整数并赋值给 r */
    idx = r & 0xff;  /* 将 r 的低8位保存到 idx 中 */
    r >>= 8;   /* 将 r 右移8位 */
    sign = r & 0x1;  /* 将 r 的最低位保存到 sign 中 */
    rabs = (r >> 1) & 0x000fffffffffffff;  /* 将 r 的除最低位外的部分保存到 rabs 中 */
    x = rabs * wi_double[idx];  /* 计算 x = rabs * wi_double[idx]，其中 wi_double 是一个预先定义的数组 */
    if (sign & 0x1)  /* 如果 sign 的最低位为1 */
      x = -x;   /* 将 x 取负 */
    if (rabs < ki_double[idx])  /* 如果 rabs 小于 ki_double[idx] */
      return x; /* 99.3% of the time return here */
    if (idx == 0) {  /* 如果 idx 等于0 */
      for (;;) {  /* 无限循环 */
        /* Switch to 1.0 - U to avoid log(0.0), see GH 13361 */
        xx = -ziggurat_nor_inv_r * npy_log1p(-next_double(bitgen_state));  /* 使用 next_double 获取一个随机的双精度浮点数，计算其负对数并乘以常数 xx */
        yy = -npy_log1p(-next_double(bitgen_state));  /* 使用 next_double 获取一个随机的双精度浮点数，计算其负对数并保存到 yy */
        if (yy + yy > xx * xx)  /* 如果 yy 的两倍大于 xx 的平方 */
          return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx) : ziggurat_nor_r + xx;  /* 返回 ziggurat_nor_r 和 xx 的和或差 */
      }
    } else {
      if (((fi_double[idx - 1] - fi_double[idx]) * next_double(bitgen_state) + fi_double[idx]) < exp(-0.5 * x * x))
        return x;  /* 返回 x */
    }
  }
}

void random_standard_normal_fill(bitgen_t *bitgen_state, npy_intp cnt, double *out) {
  npy_intp i;  /* 声明一个用于循环的变量 i */
  for (i = 0; i < cnt; i++) {  /* 循环 cnt 次 */
    out[i] = random_standard_normal(bitgen_state);  /* 调用函数 random_standard_normal 获取一个随机正态分布数值并存入 out 数组 */
  }
}

float random_standard_normal_f(bitgen_t *bitgen_state) {
  uint32_t r;   /* 声明一个无符号32位整数变量 r */
  int sign;      /* 声明一个整数变量 sign */
  uint32_t rabs;  /* 声明一个无符号32位整数变量 rabs */
  int idx;        /* 声明一个整数变量 idx */
  float x, xx, yy;  /* 声明三个单精度浮点数变量 x, xx, yy */
  for (;;) {   /* 无限循环 */
    /* r = n23sb8 */
    r = next_uint32(bitgen_state);  /* 调用函数 next_uint32 获取一个随机的32位无符号整数并赋值给 r */
    idx = r & 0xff;  /* 将 r 的低8位保存到 idx 中 */
    sign = (r >> 8) & 0x1;  /* 将 r 的第9位保存到 sign 中 */
    rabs = (r >> 9) & 0x0007fffff;  /* 将 r 的第10位到第30位保存到 rabs 中 */
    x = rabs * wi_float[idx];  /* 计算 x = rabs * wi_float[idx]，其中 wi_float 是一个预先定义的数组 */
    if (sign & 0x1)  /* 如果 sign 的最低位为1 */
      x = -x;   /* 将 x 取负 */
    if (rabs < ki_float[idx])  /* 如果 rabs 小于 ki_float[idx] */
      return x; /* # 99.3% of the time return here */
    if (idx == 0) {  /* 如果 idx 等于0 */
      for (;;) {  /* 无限循环 */
        /* Switch to 1.0 - U to avoid log(0.0), see GH 13361 */
        xx = -ziggurat_nor_inv_r_f * npy_log1pf(-next_float(bitgen_state));  /* 使用 next_float 获取一个随机的单精度浮点数，计算其负对数并乘以常数 xx */
        yy = -npy_log1pf(-next_float(bitgen_state));
    } else {
      # 如果条件不满足上述第一个条件，执行以下代码块
      if (((fi_float[idx - 1] - fi_float[idx]) * next_float(bitgen_state) +
           fi_float[idx]) < exp(-0.5 * x * x))
        # 如果计算出的新值小于指数函数的结果，则返回当前的 x 值
        return x;
    }
  }
/* 从正态分布中生成随机数填充到浮点数数组中 */
void random_standard_normal_fill_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) {
  npy_intp i;
  // 循环生成指定数量的随机数
  for (i = 0; i < cnt; i++) {
    // 调用 random_standard_normal_f 函数生成一个标准正态分布的随机数，并填充到数组中
    out[i] = random_standard_normal_f(bitgen_state);
  }
}

/* 生成标准 Gamma 分布的随机数 */
double random_standard_gamma(bitgen_t *bitgen_state, double shape) {
  double b, c;
  double U, V, X, Y;

  // 如果形状参数 shape 等于 1.0，则返回标准指数分布的随机数
  if (shape == 1.0) {
    return random_standard_exponential(bitgen_state);
  }
  // 如果形状参数 shape 等于 0.0，则直接返回 0.0
  else if (shape == 0.0) {
    return 0.0;
  }
  // 如果形状参数 shape 小于 1.0
  else if (shape < 1.0) {
    // 开始无限循环，直到生成符合条件的随机数
    for (;;) {
      // 生成 [0, 1) 区间内的均匀分布随机数 U 和标准指数分布随机数 V
      U = next_double(bitgen_state);
      V = random_standard_exponential(bitgen_state);
      if (U <= 1.0 - shape) {
        // 如果 U 小于等于 1-shape，则根据逆变换法计算 X
        X = pow(U, 1. / shape);
        if (X <= V) {
          return X;
        }
      } else {
        // 否则，根据逆变换法计算 Y 和 X
        Y = -log((1 - U) / shape);
        X = pow(1.0 - shape + shape * Y, 1. / shape);
        if (X <= (V + Y)) {
          return X;
        }
      }
    }
  }
  // 如果形状参数 shape 大于等于 1.0
  else {
    // 计算辅助变量 b 和 c
    b = shape - 1. / 3.;
    c = 1. / sqrt(9 * b);
    // 开始无限循环，直到生成符合条件的随机数
    for (;;) {
      do {
        // 生成标准正态分布的随机数 X 和 V
        X = random_standard_normal(bitgen_state);
        V = 1.0 + c * X;
      } while (V <= 0.0);

      V = V * V * V;
      // 生成 [0, 1) 区间内的均匀分布随机数 U
      U = next_double(bitgen_state);
      if (U < 1.0 - 0.0331 * (X * X) * (X * X))
        return (b * V);
      // log(0.0) 在这里是允许的，用于比较
      if (log(U) < 0.5 * X * X + b * (1. - V + log(V)))
        return (b * V);
    }
  }
}

/* 生成标准 Gamma 分布的随机数（单精度浮点数版本） */
float random_standard_gamma_f(bitgen_t *bitgen_state, float shape) {
  float b, c;
  float U, V, X, Y;

  // 如果形状参数 shape 等于 1.0f，则返回标准指数分布的随机数（单精度）
  if (shape == 1.0f) {
    return random_standard_exponential_f(bitgen_state);
  }
  // 如果形状参数 shape 等于 0.0，则直接返回 0.0（单精度）
  else if (shape == 0.0) {
    return 0.0;
  }
  // 如果形状参数 shape 小于 1.0f
  else if (shape < 1.0f) {
    // 开始无限循环，直到生成符合条件的随机数
    for (;;) {
      // 生成 [0, 1) 区间内的均匀分布随机数 U 和标准指数分布随机数 V
      U = next_float(bitgen_state);
      V = random_standard_exponential_f(bitgen_state);
      if (U <= 1.0f - shape) {
        // 如果 U 小于等于 1-shape，则根据逆变换法计算 X
        X = powf(U, 1.0f / shape);
        if (X <= V) {
          return X;
        }
      } else {
        // 否则，根据逆变换法计算 Y 和 X
        Y = -logf((1.0f - U) / shape);
        X = powf(1.0f - shape + shape * Y, 1.0f / shape);
        if (X <= (V + Y)) {
          return X;
        }
      }
    }
  }
  // 如果形状参数 shape 大于等于 1.0f
  else {
    // 计算辅助变量 b 和 c
    b = shape - 1.0f / 3.0f;
    c = 1.0f / sqrtf(9.0f * b);
    // 开始无限循环，直到生成符合条件的随机数
    for (;;) {
      do {
        // 生成标准正态分布的随机数 X 和 V
        X = random_standard_normal_f(bitgen_state);
        V = 1.0f + c * X;
      } while (V <= 0.0f);

      V = V * V * V;
      // 生成 [0, 1) 区间内的均匀分布随机数 U
      U = next_float(bitgen_state);
      if (U < 1.0f - 0.0331f * (X * X) * (X * X))
        return (b * V);
      // logf(0.0f) 在这里是允许的，用于比较
      if (logf(U) < 0.5f * X * X + b * (1.0f - V + logf(V)))
        return (b * V);
    }
  }
}

/* 生成正整数类型的随机数（64位） */
int64_t random_positive_int64(bitgen_t *bitgen_state) {
  return next_uint64(bitgen_state) >> 1;
}

/* 生成正整数类型的随机数（32位） */
int32_t random_positive_int32(bitgen_t *bitgen_state) {
  return next_uint32(bitgen_state) >> 1;
}

/* 生成正整数类型的随机数，根据平台确定返回类型 */
int64_t random_positive_int(bitgen_t *bitgen_state) {
#if ULONG_MAX <= 0xffffffffUL
  return (int64_t)(next_uint32(bitgen_state) >> 1);
#else
  return (int64_t)(next_uint64(bitgen_state) >> 1);
#endif
}
#if ULONG_MAX <= 0xffffffffUL
  // 如果 ULONG_MAX 小于等于 0xffffffffUL，则返回一个 32 位随机数
  return next_uint32(bitgen_state);
#else
  // 否则返回一个 64 位随机数
  return next_uint64(bitgen_state);
#endif
}

/*
 * log-gamma 函数用于支持某些分布。该算法来源于张善杰和金建明的《Computation of Special Functions》。
 * 如果 random_loggam(k+1) 用于计算整数 k 的 log(k!)，建议使用 logfactorial(k)。
 */
double random_loggam(double x) {
  double x0, x2, lg2pi, gl, gl0;
  RAND_INT_TYPE k, n;

  static double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                         7.936507936507937e-04, -5.952380952380952e-04,
                         8.417508417508418e-04, -1.917526917526918e-03,
                         6.410256410256410e-03, -2.955065359477124e-02,
                         1.796443723688307e-01, -1.39243221690590e+00};

  if ((x == 1.0) || (x == 2.0)) {
    // 当 x 为 1.0 或 2.0 时，直接返回 0.0
    return 0.0;
  } else if (x < 7.0) {
    // 当 x 小于 7.0 时，计算 n
    n = (RAND_INT_TYPE)(7 - x);
  } else {
    // 否则 n 等于 0
    n = 0;
  }
  x0 = x + n;
  x2 = (1.0 / x0) * (1.0 / x0);
  /* log(2 * M_PI) */
  lg2pi = 1.8378770664093453e+00;
  gl0 = a[9];
  for (k = 8; k >= 0; k--) {
    // 使用级数计算 gamma 函数的对数值
    gl0 *= x2;
    gl0 += a[k];
  }
  // 计算 gamma 函数的对数值
  gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * log(x0) - x0;
  if (x < 7.0) {
    // 如果 x 小于 7.0，进一步调整 gl 的值
    for (k = 1; k <= n; k++) {
      gl -= log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  // 返回 gamma 函数的对数值
  return gl;
}

/*
double random_normal(bitgen_t *bitgen_state, double loc, double scale) {
  return loc + scale * random_gauss(bitgen_state);
}
*/

// 使用正态分布函数 random_standard_normal 计算正态分布随机数
double random_normal(bitgen_t *bitgen_state, double loc, double scale) {
  return loc + scale * random_standard_normal(bitgen_state);
}

// 使用指数分布函数 random_standard_exponential 计算指数分布随机数
double random_exponential(bitgen_t *bitgen_state, double scale) {
  return scale * random_standard_exponential(bitgen_state);
}

// 使用均匀分布函数 next_double 计算均匀分布随机数
double random_uniform(bitgen_t *bitgen_state, double lower, double range) {
  return lower + range * next_double(bitgen_state);
}

// 使用 gamma 分布函数 random_standard_gamma 计算 gamma 分布随机数
double random_gamma(bitgen_t *bitgen_state, double shape, double scale) {
  return scale * random_standard_gamma(bitgen_state, shape);
}

// 使用 gamma 分布函数 random_standard_gamma_f 计算 gamma 分布随机数（单精度浮点数版本）
float random_gamma_f(bitgen_t *bitgen_state, float shape, float scale) {
  return scale * random_standard_gamma_f(bitgen_state, shape);
}

#define BETA_TINY_THRESHOLD 3e-103

/*
 *  注意：random_beta 假设 a != 0 且 b != 0。
 */
double random_beta(bitgen_t *bitgen_state, double a, double b) {
  double Ga, Gb;

  if ((a <= 1.0) && (b <= 1.0)) {
    double U, V, X, Y, XpY;

    if (a < BETA_TINY_THRESHOLD && b < BETA_TINY_THRESHOLD) {
      /*
       * 当 a 和 b 很小时，生成随机数的概率非常低，因此使用 a/(a + b) 和一个均匀随机数 U 来生成结果。
       */
      U = next_double(bitgen_state);
      return (a + b)*U < a;
    }

    /* 使用 Johnk's 算法 */

      U = next_double(bitgen_state);
      V = next_double(bitgen_state);
      X = pow(U, 1.0 / a);
      Y = pow(V, 1.0 / b);
      XpY = X + Y;
      if (XpY <= 1.0) {
        return X / XpY;
      } else {
        return 1.0;
      }
    }

    /* 使用 Gauss 近似法 */
    Ga = random_gamma(bitgen_state, a, 1.0);
    Gb = random_gamma(bitgen_state, b, 1.0);
    return Ga / (Ga + Gb);
  } else {
    double alpha, beta, r;

    if (a < BETA_TINY_THRESHOLD) {
      alpha = 1.0 + b;
      beta = 1.0 + b * tan(M_PI * a / 2.0);
      r = pow(next_double(bitgen_state), 1.0 / a);
      return r / (r + pow(next_double(bitgen_state), 1.0 / (alpha * a)));
    }

    if (b < BETA_TINY_THRESHOLD) {
      alpha = 1.0 + a;
      beta = 1.0 + a * tan(M_PI * b / 2.0);
      r = pow(next_double(bitgen_state), 1.0 / (alpha * b));
      return 1.0 - r / (r + pow(next_double(bitgen_state), 1.0 / b));
    }

    alpha = a + b;
    beta = sqrt((alpha - 2.0) / (2.0 * a * b - alpha));
    r = random_normal(bitgen_state, 1.0, beta);
    return (1.0 + a * r) / (alpha + alpha * r);
  }
}
    while (1) {
      // 从随机数生成器中获取下一个双精度随机数 U 和 V
      U = next_double(bitgen_state);
      V = next_double(bitgen_state);
      // 计算 U 和 V 的 a 次方根 X 和 b 次方根 Y
      X = pow(U, 1.0 / a);
      Y = pow(V, 1.0 / b);
      // 计算 X 和 Y 的和
      XpY = X + Y;
      /* 如果 XpY 小于等于 1.0 并且 U + V 大于 0.0，则接受生成的随机数 */
      if ((XpY <= 1.0) && (U + V > 0.0)) {
        if (XpY > 0) {
          // 如果 XpY 大于 0，则返回 X / XpY
          return X / XpY;
        } else {
          // 否则计算对数变换的加权平均数，返回对应的概率密度函数值
          double logX = log(U) / a;
          double logY = log(V) / b;
          double logM = logX > logY ? logX : logY;
          logX -= logM;
          logY -= logM;
          return exp(logX - log(exp(logX) + exp(logY)));
        }
      }
    }
  } else {
    // 使用标准 Gamma 分布生成器生成随机变量 Ga 和 Gb
    Ga = random_standard_gamma(bitgen_state, a);
    Gb = random_standard_gamma(bitgen_state, b);
    // 返回 Ga / (Ga + Gb) 作为生成的随机数
    return Ga / (Ga + Gb);
  }
}

// 生成自由度为 df 的卡方分布随机变量的函数
double random_chisquare(bitgen_t *bitgen_state, double df) {
    // 调用标准伽马分布函数生成自由度为 df/2 的随机变量，再进行线性变换得到卡方分布随机变量
    return 2.0 * random_standard_gamma(bitgen_state, df / 2.0);
}

// 生成自由度为 dfnum 和 dfden 的 F 分布随机变量的函数
double random_f(bitgen_t *bitgen_state, double dfnum, double dfden) {
    // 使用随机卡方分布函数生成自由度为 dfnum 和 dfden 的两个卡方分布随机变量，并进行线性变换得到 F 分布随机变量
    return ((random_chisquare(bitgen_state, dfnum) * dfden) /
            (random_chisquare(bitgen_state, dfden) * dfnum));
}

// 生成标准柯西分布随机变量的函数
double random_standard_cauchy(bitgen_t *bitgen_state) {
    // 生成两个标准正态分布随机变量，并用其比值得到标准柯西分布随机变量
    return random_standard_normal(bitgen_state) / random_standard_normal(bitgen_state);
}

// 生成参数为 a 的帕累托分布随机变量的函数
double random_pareto(bitgen_t *bitgen_state, double a) {
    // 生成标准指数分布随机变量，进行线性变换得到帕累托分布随机变量
    return expm1(random_standard_exponential(bitgen_state) / a);
}

// 生成参数为 a 的威布尔分布随机变量的函数
double random_weibull(bitgen_t *bitgen_state, double a) {
    if (a == 0.0) {
        return 0.0;
    }
    // 生成标准指数分布随机变量，进行幂运算得到威布尔分布随机变量
    return pow(random_standard_exponential(bitgen_state), 1. / a);
}

// 生成参数为 a 的幂律分布随机变量的函数
double random_power(bitgen_t *bitgen_state, double a) {
    // 生成标准指数分布随机变量，进行幂运算和逆运算得到幂律分布随机变量
    return pow(-expm1(-random_standard_exponential(bitgen_state)), 1. / a);
}

// 生成参数为 loc 和 scale 的拉普拉斯分布随机变量的函数
double random_laplace(bitgen_t *bitgen_state, double loc, double scale) {
    double U;

    U = next_double(bitgen_state);
    if (U >= 0.5) {
        // 生成两个均匀分布随机变量，进行对数变换得到拉普拉斯分布随机变量
        U = loc - scale * log(2.0 - U - U);
    } else if (U > 0.0) {
        // 生成两个均匀分布随机变量，进行对数变换得到拉普拉斯分布随机变量
        U = loc + scale * log(U + U);
    } else {
        /* 拒绝 U == 0.0 并重新调用以获取下一个值 */
        U = random_laplace(bitgen_state, loc, scale);
    }
    return U;
}

// 生成参数为 loc 和 scale 的冈伯尔分布随机变量的函数
double random_gumbel(bitgen_t *bitgen_state, double loc, double scale) {
    double U;

    U = 1.0 - next_double(bitgen_state);
    if (U < 1.0) {
        // 生成均匀分布随机变量，进行对数变换得到冈伯尔分布随机变量
        return loc - scale * log(-log(U));
    }
    /* 拒绝 U == 1.0 并重新调用以获取下一个值 */
    return random_gumbel(bitgen_state, loc, scale);
}

// 生成参数为 loc 和 scale 的逻辑斯蒂分布随机变量的函数
double random_logistic(bitgen_t *bitgen_state, double loc, double scale) {
    double U;

    U = next_double(bitgen_state);
    if (U > 0.0) {
        // 生成均匀分布随机变量，进行对数变换得到逻辑斯蒂分布随机变量
        return loc + scale * log(U / (1.0 - U));
    }
    /* 拒绝 U == 0.0 并重新调用以获取下一个值 */
    return random_logistic(bitgen_state, loc, scale);
}

// 生成参数为 mean 和 sigma 的对数正态分布随机变量的函数
double random_lognormal(bitgen_t *bitgen_state, double mean, double sigma) {
    // 生成正态分布随机变量，进行指数变换得到对数正态分布随机变量
    return exp(random_normal(bitgen_state, mean, sigma));
}

// 生成参数为 mode 的瑞利分布随机变量的函数
double random_rayleigh(bitgen_t *bitgen_state, double mode) {
    // 生成标准指数分布随机变量，进行线性变换得到瑞利分布随机变量
    return mode * sqrt(2.0 * random_standard_exponential(bitgen_state));
}

// 生成自由度为 df 的标准 t 分布随机变量的函数
double random_standard_t(bitgen_t *bitgen_state, double df) {
    double num, denom;

    num = random_standard_normal(bitgen_state);
    denom = random_standard_gamma(bitgen_state, df / 2);
    // 对标准正态分布和标准卡方分布随机变量进行线性变换得到标准 t 分布随机变量
    return sqrt(df / 2) * num / sqrt(denom);
}

// 用于生成泊松随机变量的转换拒绝法函数
static RAND_INT_TYPE random_poisson_mult(bitgen_t *bitgen_state, double lam) {
    RAND_INT_TYPE X;
    double prod, U, enlam;

    enlam = exp(-lam);
    X = 0;
    prod = 1.0;
    while (1) {
        U = next_double(bitgen_state);
        prod *= U;
        if (prod > enlam) {
            X += 1;
        } else {
            return X;
        }
    }
}

/*
 * 用于生成泊松随机变量的变换拒绝法函数
 * W. Hoermann
 * Insurance: Mathematics and Economics 12, 39-45 (1993)
 */
#define LS2PI 0.91893853320467267
#define TWELFTH 0.083333333333333333333333
static RAND_INT_TYPE random_poisson_ptrs(bitgen_t *bitgen_state, double lam) {
  RAND_INT_TYPE k;
  double U, V, slam, loglam, a, b, invalpha, vr, us;

  slam = sqrt(lam);  // 计算 lam 的平方根
  loglam = log(lam);  // 计算 lam 的自然对数
  b = 0.931 + 2.53 * slam;  // 根据 slam 计算 b 的值
  a = -0.059 + 0.02483 * b;  // 根据 b 计算 a 的值
  invalpha = 1.1239 + 1.1328 / (b - 3.4);  // 根据 b 计算 invalpha 的值
  vr = 0.9277 - 3.6224 / (b - 2);  // 根据 b 计算 vr 的值

  while (1) {
    U = next_double(bitgen_state) - 0.5;  // 生成 U，范围在 [-0.5, 0.5]
    V = next_double(bitgen_state);  // 生成 V，范围在 [0, 1)
    us = 0.5 - fabs(U);  // 计算 us，范围在 [0, 0.5]
    k = (RAND_INT_TYPE)floor((2 * a / us + b) * U + lam + 0.43);  // 根据 U、us、a、b、lam 计算 k

    // 根据生成的 U、V、us、vr 进行条件判断
    if ((us >= 0.07) && (V <= vr)) {
      return k;  // 返回 k
    }
    if ((k < 0) || ((us < 0.013) && (V > us))) {
      continue;  // 如果不满足条件则继续循环
    }
    /* log(V) == log(0.0) ok here */
    /* if U==0.0 so that us==0.0, log is ok since always returns */
    // 根据生成的 V、invalpha、a、us 进行条件判断
    if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
        (-lam + k * loglam - random_loggam(k + 1))) {
      return k;  // 返回 k
    }
  }
}

RAND_INT_TYPE random_poisson(bitgen_t *bitgen_state, double lam) {
  if (lam >= 10) {
    return random_poisson_ptrs(bitgen_state, lam);  // 如果 lam 大于等于 10，调用 random_poisson_ptrs 函数
  } else if (lam == 0) {
    return 0;  // 如果 lam 等于 0，返回 0
  } else {
    return random_poisson_mult(bitgen_state, lam);  // 否则调用 random_poisson_mult 函数
  }
}

RAND_INT_TYPE random_negative_binomial(bitgen_t *bitgen_state, double n,
                                       double p) {
  double Y = random_gamma(bitgen_state, n, (1 - p) / p);  // 调用 random_gamma 函数生成 Y
  return random_poisson(bitgen_state, Y);  // 调用 random_poisson 函数，传入 Y 作为参数
}

RAND_INT_TYPE random_binomial_btpe(bitgen_t *bitgen_state, RAND_INT_TYPE n,
                                   double p, binomial_t *binomial) {
  double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
  double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
  RAND_INT_TYPE m, y, k, i;

  if (!(binomial->has_binomial) || (binomial->nsave != n) ||
      (binomial->psave != p)) {
    /* initialize */
    binomial->nsave = n;
    binomial->psave = p;
    binomial->has_binomial = 1;
    binomial->r = r = MIN(p, 1.0 - p);
    binomial->q = q = 1.0 - r;
    binomial->fm = fm = n * r + r;
    binomial->m = m = (RAND_INT_TYPE)floor(binomial->fm);
    binomial->p1 = p1 = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5;
    binomial->xm = xm = m + 0.5;
    binomial->xl = xl = xm - p1;
    binomial->xr = xr = xm + p1;
    binomial->c = c = 0.134 + 20.5 / (15.3 + m);
    a = (fm - xl) / (fm - xl * r);
    binomial->laml = laml = a * (1.0 + a / 2.0);
    a = (xr - fm) / (xr * q);
    binomial->lamr = lamr = a * (1.0 + a / 2.0);
    binomial->p2 = p2 = p1 * (1.0 + 2.0 * c);
    binomial->p3 = p3 = p2 + c / laml;
    binomial->p4 = p4 = p3 + c / lamr;
  } else {
    r = binomial->r;
    q = binomial->q;
    fm = binomial->fm;
    m = binomial->m;
    p1 = binomial->p1;
    xm = binomial->xm;
    xl = binomial->xl;
    xr = binomial->xr;
    c = binomial->c;
    laml = binomial->laml;
    lamr = binomial->lamr;
    p2 = binomial->p2;
    p3 = binomial->p3;
    p4 = binomial->p4;
  }

/* sigh ... */
Step10:
  nrq = n * r * q;  // 计算 n * r * q
  u = next_double(bitgen_state) * p4;  // 生成 u，范围在 [0, p4)
  v = next_double(bitgen_state);  // 生成 v，范围在 [0, 1)
  if (u > p1)  // 判断 u 是否大于 p1
    # 转到步骤20的标签，即代码跳转到后续步骤的标记点
    goto Step20;
  # 计算 y 的值，使用 floor 函数对 xm - p1 * v + u 进行向下取整，并将结果转换为 RAND_INT_TYPE 类型
  y = (RAND_INT_TYPE)floor(xm - p1 * v + u);
  # 转到步骤60的标签，即代码跳转到后续步骤的标记点
  goto Step60;
Step20:
  // 如果随机数大于 p2，则跳转到 Step30
  if (u > p2)
    goto Step30;
  // 计算随机数对应的变量 x
  x = xl + (u - p1) / c;
  // 更新 v，这里的操作是根据变量 m 和 x 计算 v 的新值
  v = v * c + 1.0 - fabs(m - x + 0.5) / p1;
  // 如果 v 大于 1.0，则跳转到 Step10
  if (v > 1.0)
    goto Step10;
  // 计算 x 的整数部分作为 y，并跳转到 Step50
  y = (RAND_INT_TYPE)floor(x);
  goto Step50;

Step30:
  // 如果随机数大于 p3，则跳转到 Step40
  if (u > p3)
    goto Step40;
  // 根据公式计算 y 的值
  y = (RAND_INT_TYPE)floor(xl + log(v) / laml);
  /* 如果 v 等于 0.0，则拒绝，因为前面的转换是未定义的 */
  if ((y < 0) || (v == 0.0))
    goto Step10;
  // 更新 v
  v = v * (u - p2) * laml;
  goto Step50;

Step40:
  // 根据公式计算 y 的值
  y = (RAND_INT_TYPE)floor(xr - log(v) / lamr);
  /* 如果 v 等于 0.0，则拒绝，因为前面的转换是未定义的 */
  if ((y > n) || (v == 0.0))
    goto Step10;
  // 更新 v
  v = v * (u - p3) * lamr;

Step50:
  // 计算 k 的绝对值
  k = llabs(y - m);
  // 如果 k 在指定的范围内，则跳转到 Step52
  if ((k > 20) && (k < ((nrq) / 2.0 - 1)))
    goto Step52;

  // 计算 s，a 和 F
  s = r / q;
  a = s * (n + 1);
  F = 1.0;
  // 根据条件更新 F 的值
  if (m < y) {
    for (i = m + 1; i <= y; i++) {
      F *= (a / i - s);
    }
  } else if (m > y) {
    for (i = y + 1; i <= m; i++) {
      F /= (a / i - s);
    }
  }
  // 如果 v 大于 F，则跳转到 Step10
  if (v > F)
    goto Step10;
  // 跳转到 Step60
  goto Step60;

Step52:
  // 计算 rho 和 t
  rho =
      (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
  t = -k * k / (2 * nrq);
  /* log(0.0) 在这里是允许的 */
  A = log(v);
  // 根据条件判断是否跳转到 Step60 或 Step10
  if (A < (t - rho))
    goto Step60;
  if (A > (t + rho))
    goto Step10;

  // 根据一系列公式计算和比较，判断是否跳转到 Step10
  x1 = y + 1;
  f1 = m + 1;
  z = n + 1 - m;
  w = n - y + 1;
  x2 = x1 * x1;
  f2 = f1 * f1;
  z2 = z * z;
  w2 = w * w;
  if (A > (xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) +
           (y - m) * log(w * r / (x1 * q)) +
           (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w /
               166320.)) {
    goto Step10;
  }

Step60:
  // 如果 p 大于 0.5，则更新 y
  if (p > 0.5) {
    y = n - y;
  }

  // 返回最终的 y 值
  return y;
}

RAND_INT_TYPE random_binomial_inversion(bitgen_t *bitgen_state, RAND_INT_TYPE n,
                                        double p, binomial_t *binomial) {
  double q, qn, np, px, U;
  RAND_INT_TYPE X, bound;

  // 如果 binomial 结构体中的参数与当前传入的不一致，则重新初始化
  if (!(binomial->has_binomial) || (binomial->nsave != n) ||
      (binomial->psave != p)) {
    binomial->nsave = n;
    binomial->psave = p;
    binomial->has_binomial = 1;
    binomial->q = q = 1.0 - p;
    binomial->r = qn = exp(n * log(q));
    binomial->c = np = n * p;
    binomial->m = bound = (RAND_INT_TYPE)MIN(n, np + 10.0 * sqrt(np * q + 1));
  } else {
    // 否则，使用已存储的参数
    q = binomial->q;
    qn = binomial->r;
    np = binomial->c;
    bound = binomial->m;
  }
  X = 0;
  px = qn;
  // 使用指定算法生成服从二项分布的随机数 X
  U = next_double(bitgen_state);
  while (U > px) {
    X++;
    if (X > bound) {
      X = 0;
      px = qn;
      U = next_double(bitgen_state);
    } else {
      U -= px;
      px = ((n - X + 1) * p * px) / (X * q);
    }
  }
  // 返回生成的随机数 X
  return X;
}
// 计算二项分布随机变量的函数，返回计算结果
int64_t random_binomial(bitgen_t *bitgen_state, double p, int64_t n,
                        binomial_t *binomial) {
  double q;

  // 如果 n 为 0 或者 p 为 0，则直接返回 0
  if ((n == 0LL) || (p == 0.0f))
    return 0;

  // 如果 p <= 0.5，根据 p * n 的大小选择不同的方法计算
  if (p <= 0.5) {
    // 如果 p * n <= 30.0，使用反向变换法计算二项分布
    if (p * n <= 30.0) {
      return random_binomial_inversion(bitgen_state, n, p, binomial);
    } else {
      // 否则使用 BTPE 算法计算二项分布
      return random_binomial_btpe(bitgen_state, n, p, binomial);
    }
  } else {
    // 如果 p > 0.5，计算 q = 1 - p
    q = 1.0 - p;
    // 如果 q * n <= 30.0，使用反向变换法计算 n - 二项分布的值
    if (q * n <= 30.0) {
      return n - random_binomial_inversion(bitgen_state, n, q, binomial);
    } else {
      // 否则使用 BTPE 算法计算 n - 二项分布的值
      return n - random_binomial_btpe(bitgen_state, n, q, binomial);
    }
  }
}

// 计算非中心卡方分布的随机变量，返回计算结果
double random_noncentral_chisquare(bitgen_t *bitgen_state, double df,
                                   double nonc) {
  // 如果 nonc 是 NaN，则返回 NaN
  if (npy_isnan(nonc)) {
    return NPY_NAN;
  }
  // 如果 nonc 为 0，则返回 df 自由度的卡方分布的随机变量
  if (nonc == 0) {
    return random_chisquare(bitgen_state, df);
  }
  // 如果 df > 1，则进行非中心卡方分布的计算
  if (1 < df) {
    const double Chi2 = random_chisquare(bitgen_state, df - 1);
    const double n = random_standard_normal(bitgen_state) + sqrt(nonc);
    return Chi2 + n * n;
  } else {
    // 如果 df <= 1，则使用泊松分布和卡方分布计算非中心卡方分布
    const RAND_INT_TYPE i = random_poisson(bitgen_state, nonc / 2.0);
    return random_chisquare(bitgen_state, df + 2 * i);
  }
}

// 计算非中心 F 分布的随机变量，返回计算结果
double random_noncentral_f(bitgen_t *bitgen_state, double dfnum, double dfden,
                           double nonc) {
  // 计算非中心卡方分布随机变量乘以 dfden
  double t = random_noncentral_chisquare(bitgen_state, dfnum, nonc) * dfden;
  return t / (random_chisquare(bitgen_state, dfden) * dfnum);
}

// 计算瓦尔德分布的随机变量，返回计算结果
double random_wald(bitgen_t *bitgen_state, double mean, double scale) {
  double U, X, Y;
  double mu_2l;

  // 计算 mu / (2 * scale)
  mu_2l = mean / (2 * scale);
  // 生成标准正态分布的随机变量 Y
  Y = random_standard_normal(bitgen_state);
  // 计算 X 和 U
  Y = mean * Y * Y;
  X = mean + mu_2l * (Y - sqrt(4 * scale * Y + Y * Y));
  U = next_double(bitgen_state);
  // 根据 U 的值选择返回 X 或者 mean * mean / X
  if (U <= mean / (mean + X)) {
    return X;
  } else {
    return mean * mean / X;
  }
}

// 计算冯·米塞斯分布的随机变量，返回计算结果
double random_vonmises(bitgen_t *bitgen_state, double mu, double kappa) {
  double s;
  double U, V, W, Y, Z;
  double result, mod;
  int neg;
  
  // 如果 kappa 是 NaN，则返回 NaN
  if (npy_isnan(kappa)) {
    return NPY_NAN;
  }
  // 如果 kappa < 1e-8，使用均匀分布生成随机变量
  if (kappa < 1e-8) {
    return M_PI * (2 * next_double(bitgen_state) - 1);
  } else {
    // 对于其他情况，根据 kappa 的大小选择不同的计算方法
    if (kappa < 1e-5) {
      // 当 1e-5 <= kappa <= 1e-6 时，使用二阶泰勒展开计算
      s = (1. / kappa + kappa);
    } else {
      if (kappa <= 1e6) {
        // 当 1e-5 < kappa <= 1e6 时，使用特定的数学公式计算
        double r = 1 + sqrt(1 + 4 * kappa * kappa);
        double rho = (r - sqrt(2 * r)) / (2 * kappa);
        s = (1 + rho * rho) / (2 * rho);
      } else {
        // 当 kappa > 1e6 时，使用包裹的正态分布计算
        result = mu + sqrt(1. / kappa) * random_standard_normal(bitgen_state);
        // 确保结果在 -π 到 π 之间
        if (result < -M_PI) {
          result += 2*M_PI;
        }
        if (result > M_PI) {
          result -= 2*M_PI;
        }
        return result;
      }
    }
    // 返回计算结果
    return result;
  }
}
    }

这是一个 do-while 循环的结束标志，表示前面的循环体结束。


    while (1) {

开始一个无限循环，直到满足条件跳出循环。


      U = next_double(bitgen_state);
      Z = cos(M_PI * U);
      W = (1 + s * Z) / (s + Z);
      Y = kappa * (s - W);
      V = next_double(bitgen_state);

生成随机数并计算出变量 Z, W, Y 和 V，这些变量将用于接下来的条件判断。


      /*
       * V==0.0 is ok here since Y >= 0 always leads
       * to accept, while Y < 0 always rejects
       */
      if ((Y * (2 - Y) - V >= 0) || (log(Y / V) + 1 - Y >= 0)) {
        break;
      }

条件判断语句，根据 Y 和 V 的值来决定是否跳出循环。注释解释了为什么 V 可以等于 0 的情况。


    U = next_double(bitgen_state);

生成下一个随机数 U。


    result = acos(W);
    if (U < 0.5) {
      result = -result;
    }
    result += mu;

根据 U 的值对 result 进行调整，先求 acos(W)，然后根据 U 的大小确定是否取反，最后加上 mu。


    neg = (result < 0);
    mod = fabs(result);
    mod = (fmod(mod + M_PI, 2 * M_PI) - M_PI);
    if (neg) {
      mod *= -1;
    }

处理 result 的符号和将其调整到 [-π, π] 范围内。


    return mod;
  }

返回处理后的 mod 值，作为函数的输出结果。
}

/*
 * 生成符合对数级数分布的随机数
 * bitgen_state: 随机数生成器状态
 * p: 成功概率
 */
int64_t random_logseries(bitgen_t *bitgen_state, double p) {
  double q, r, U, V;
  int64_t result;

  r = npy_log1p(-p);

  while (1) {
    V = next_double(bitgen_state);  // 生成一个均匀分布的随机数 V
    if (V >= p) {  // 如果 V 大于等于成功概率 p，返回 1
      return 1;
    }
    U = next_double(bitgen_state);  // 生成另一个均匀分布的随机数 U
    q = -expm1(r * U);  // 计算 q 值
    if (V <= q * q) {  // 如果 V 小于等于 q 的平方，执行以下操作
      result = (int64_t)floor(1 + log(V) / log(q));  // 计算结果并返回
      if ((result < 1) || (V == 0.0)) {  // 如果结果小于 1 或者 V 等于 0，继续循环
        continue;
      } else {
        return result;  // 返回结果
      }
    }
    if (V >= q) {  // 如果 V 大于等于 q，返回 1
      return 1;
    }
    return 2;  // 默认情况下返回 2
  }
}

/*
 * 生成符合几何分布的随机数
 * bitgen_state: 随机数生成器状态
 * p: 成功概率
 */
RAND_INT_TYPE random_geometric_search(bitgen_t *bitgen_state, double p) {
  double U;
  RAND_INT_TYPE X;
  double sum, prod, q;

  X = 1;
  sum = prod = p;
  q = 1.0 - p;
  U = next_double(bitgen_state);  // 生成一个均匀分布的随机数 U
  while (U > sum) {  // 循环直到 U 大于 sum
    prod *= q;  // 更新 prod
    sum += prod;  // 更新 sum
    X++;  // X 自增
  }
  return X;  // 返回生成的随机数 X
}

/*
 * 使用反转方法生成符合几何分布的随机数
 * bitgen_state: 随机数生成器状态
 * p: 成功概率
 */
int64_t random_geometric_inversion(bitgen_t *bitgen_state, double p) {
  double z = ceil(-random_standard_exponential(bitgen_state) / npy_log1p(-p));
  /*
   * 常量 9.223372036854776e+18 是大于 INT64_MAX 的最小双精度浮点数。
   * 如果 z 大于等于此常量，返回 INT64_MAX。
   */
  if (z >= 9.223372036854776e+18) {
    return INT64_MAX;
  }
  return (int64_t) z;  // 返回生成的随机数 z
}

/*
 * 生成符合几何分布的随机数
 * bitgen_state: 随机数生成器状态
 * p: 成功概率
 */
int64_t random_geometric(bitgen_t *bitgen_state, double p) {
  if (p >= 0.333333333333333333333333) {
    return random_geometric_search(bitgen_state, p);  // 使用搜索方法生成
  } else {
    return random_geometric_inversion(bitgen_state, p);  // 使用反转方法生成
  }
}

/*
 * 生成符合 Zipf 分布的随机数
 * bitgen_state: 随机数生成器状态
 * a: 分布参数
 */
RAND_INT_TYPE random_zipf(bitgen_t *bitgen_state, double a) {
  double am1, b;

  am1 = a - 1.0;
  b = pow(2.0, am1);
  while (1) {
    double T, U, V, X;

    U = 1.0 - next_double(bitgen_state);  // 生成一个均匀分布的随机数 U
    V = next_double(bitgen_state);  // 生成另一个均匀分布的随机数 V
    X = floor(pow(U, -1.0 / am1));  // 计算 X 值
    /*
     * 如果 X 大于 RAND_INT_MAX 或者小于 1.0，则继续循环。
     * 因为这是一个简单的拒绝采样算法，所以可以拒绝这个值。
     * 此函数模拟了一个被截断到 sys.maxint 的 Zipf 分布。
     */
    if (X > (double)RAND_INT_MAX || X < 1.0) {
      continue;
    }

    T = pow(1.0 + 1.0 / X, am1);  // 计算 T 值
    if (V * X * (T - 1.0) / (b - 1.0) <= T / b) {  // 如果条件满足，返回 X
      return (RAND_INT_TYPE)X;
    }
  }
}

/*
 * 生成符合三角分布的随机数
 * bitgen_state: 随机数生成器状态
 * left: 左边界
 * mode: 众数
 * right: 右边界
 */
double random_triangular(bitgen_t *bitgen_state, double left, double mode,
                         double right) {
  double base, leftbase, ratio, leftprod, rightprod;
  double U;

  base = right - left;  // 计算基础值
  leftbase = mode - left;  // 计算左基础值
  ratio = leftbase / base;  // 计算比率
  leftprod = leftbase * base;  // 计算左乘积
  rightprod = (right - mode) * base;  // 计算右乘积

  U = next_double(bitgen_state);  // 生成一个均匀分布的随机数 U
  if (U <= ratio) {  // 如果 U 小于等于比率，返回左边界加上开方后的值
    return left + sqrt(U * leftprod);
  } else {  // 否则返回右边界减去开方后的值
    return right - sqrt((1.0 - U) * rightprod);
  }
}
/* 生成一个指定范围内的随机数
 * 使用掩码来确保生成的随机数不超出指定的最大值
 * 如果最大值为0，则直接返回0
 */
uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max) {
  uint64_t mask, value;
  if (max == 0) {
    return 0;
  }

  mask = max;

  /* 找到最小的大于等于 max 的位掩码 */
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;

  /* 在 [0..mask] 范围内搜索一个随机值使其 <= max */
  if (max <= 0xffffffffUL) {
    while ((value = (next_uint32(bitgen_state) & mask)) > max)
      ;
  } else {
    while ((value = (next_uint64(bitgen_state) & mask)) > max)
      ;
  }
  return value;
}

/* 生成一个用于生成随机数的掩码
 * 掩码用于确保生成的随机数在指定的范围内
 */
static inline uint64_t gen_mask(uint64_t max) {
  uint64_t mask = max;
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;
  return mask;
}

/* 使用 32 位缓冲区生成 16 位随机数 */
static inline uint16_t buffered_uint16(bitgen_t *bitgen_state, int *bcnt,
                                       uint32_t *buf) {
  if (!(bcnt[0])) {
    buf[0] = next_uint32(bitgen_state);
    bcnt[0] = 1;
  } else {
    buf[0] >>= 16;
    bcnt[0] -= 1;
  }

  return (uint16_t)buf[0];
}

/* 使用 32 位缓冲区生成 8 位随机数 */
static inline uint8_t buffered_uint8(bitgen_t *bitgen_state, int *bcnt,
                                     uint32_t *buf) {
  if (!(bcnt[0])) {
    buf[0] = next_uint32(bitgen_state);
    bcnt[0] = 3;
  } else {
    buf[0] >>= 8;
    bcnt[0] -= 1;
  }

  return (uint8_t)buf[0];
}

/* 静态的 `masked rejection` 函数，由 random_bounded_uint64(...) 调用 */
static inline uint64_t bounded_masked_uint64(bitgen_t *bitgen_state,
                                             uint64_t rng, uint64_t mask) {
  uint64_t val;

  while ((val = (next_uint64(bitgen_state) & mask)) > rng)
    ;

  return val;
}

/* 静态的 `masked rejection` 函数，由 random_buffered_bounded_uint32(...) 调用 */
static inline uint32_t
buffered_bounded_masked_uint32(bitgen_t *bitgen_state, uint32_t rng,
                               uint32_t mask, int *bcnt, uint32_t *buf) {
  /*
   * 缓冲区和缓冲区计数在此处未使用，但包含在内是为了与类似的 uint8 和 uint16 函数进行模板化
   */

  uint32_t val;

  while ((val = (next_uint32(bitgen_state) & mask)) > rng)
    ;

  return val;
}

/* 静态的 `masked rejection` 函数，由 random_buffered_bounded_uint16(...) 调用 */
static inline uint16_t
buffered_bounded_masked_uint16(bitgen_t *bitgen_state, uint16_t rng,
                               uint16_t mask, int *bcnt, uint32_t *buf) {
  uint16_t val;

  while ((val = (buffered_uint16(bitgen_state, bcnt, buf) & mask)) > rng)
    ;

  return val;
}

/* 静态的 `masked rejection` 函数，由 random_buffered_bounded_uint8(...) 调用 */
/* 
 * 从缓冲区获取一个受限制和掩码的无符号8位整数，并进行Lemire拒绝采样以确保在指定范围内。
 * 这个函数是静态的，用于内部调用。
 */
static inline uint8_t buffered_bounded_masked_uint8(bitgen_t *bitgen_state,
                                                    uint8_t rng,
                                                    uint8_t mask,
                                                    int *bcnt,
                                                    uint32_t *buf) {
  uint8_t val;

  // 使用 buffered_uint8 函数获取一个无符号8位整数，并与掩码进行按位与操作
  while ((val = (buffered_uint8(bitgen_state, bcnt, buf) & mask)) > rng)
    ; // 循环直到获取的值在指定的范围内

  return val; // 返回满足条件的值
}

/*
 * 从缓冲区获取一个受限制的布尔值，并进行Lemire拒绝采样以确保在指定范围内。
 * 这个函数是静态的，用于内部调用。
 */
static inline npy_bool buffered_bounded_bool(bitgen_t *bitgen_state,
                                             npy_bool off, npy_bool rng,
                                             npy_bool mask, int *bcnt,
                                             uint32_t *buf) {
  if (rng == 0)
    return off; // 如果范围为0，则直接返回 off

  if (!(bcnt[0])) {
    buf[0] = next_uint32(bitgen_state);
    bcnt[0] = 31;
  } else {
    buf[0] >>= 1;
    bcnt[0] -= 1;
  }

  // 返回缓冲区中的布尔值，确保在指定的范围内
  return (buf[0] & 0x00000001UL) != 0;
}

/* 
 * 由 random_bounded_uint64(...) 调用的静态 `Lemire拒绝` 函数。
 * 使用Lemire算法，确保生成的随机数在指定范围内。
 */
static inline uint64_t bounded_lemire_uint64(bitgen_t *bitgen_state,
                                             uint64_t rng) {
  /*
   * 使用Lemire的算法 - https://arxiv.org/abs/1805.10941
   *
   * 注意: `rng` 不应为 0xFFFFFFFFFFFFFFFF。当发生这种情况时，`rng_excl` 变为零。
   */
  const uint64_t rng_excl = rng + 1;

  // 断言，确保 `rng` 不等于 0xFFFFFFFFFFFFFFFFULL
  assert(rng != 0xFFFFFFFFFFFFFFFFULL);

#if __SIZEOF_INT128__
  /* 128位无符号整数可用 (例如GCC/clang)。`m` 是缩放后的 __uint128_t 整数。 */
  __uint128_t m;
  uint64_t leftover;

  // 生成一个缩放后的随机数
  m = ((__uint128_t)next_uint64(bitgen_state)) * rng_excl;

  // 使用拒绝采样消除任何偏差
  leftover = m & 0xFFFFFFFFFFFFFFFFULL;

  // 如果余数小于 `rng_excl`，继续循环直到符合条件
  if (leftover < rng_excl) {
    const uint64_t threshold = (UINT64_MAX - rng) % rng_excl;
    
    while (leftover < threshold) {
      m = ((__uint128_t)next_uint64(bitgen_state)) * rng_excl;
      leftover = m & 0xFFFFFFFFFFFFFFFFULL;
    }
  }

  // 返回缩放后的整数的高64位
  return (m >> 64);
#else
  /* 128位无符号整数不可用 (例如MSVS)。`m1` 是缩放后的整数的高64位。 */
  uint64_t m1;
  uint64_t x;
  uint64_t leftover;

  x = next_uint64(bitgen_state);

  // 使用拒绝采样消除任何偏差
  leftover = x * rng_excl;

  // 如果余数小于 `rng_excl`，继续循环直到符合条件
  if (leftover < rng_excl) {
    const uint64_t threshold = (UINT64_MAX - rng) % rng_excl;

    while (leftover < threshold) {
      x = next_uint64(bitgen_state);
      leftover = x * rng_excl;
    }
  }

#if defined(_MSC_VER) && defined(_WIN64)
  // _WIN64 架构。使用 __umulh 内置函数计算 `m1`。
  m1 = __umulh(x, rng_excl);
#else
  // 32位架构。模拟 __umulh 函数计算 `m1`。
  {
    uint64_t x0, x1, rng_excl0, rng_excl1;
    uint64_t w0, w1, w2, t;

    x0 = x & 0xFFFFFFFFULL;
    x1 = x >> 32;
    # 将 rng_excl 的低 32 位保存到 rng_excl0 中
    rng_excl0 = rng_excl & 0xFFFFFFFFULL;
    # 将 rng_excl 的高 32 位保存到 rng_excl1 中
    rng_excl1 = rng_excl >> 32;
    # 计算 x0 乘以 rng_excl0 的结果，存入 w0 中
    w0 = x0 * rng_excl0;
    # 计算 x1 乘以 rng_excl0 的结果，加上 w0 的高 32 位部分，存入 t 中
    t = x1 * rng_excl0 + (w0 >> 32);
    # 提取 t 的低 32 位保存到 w1 中
    w1 = t & 0xFFFFFFFFULL;
    # 提取 t 的高 32 位保存到 w2 中
    w2 = t >> 32;
    # 计算 x0 乘以 rng_excl1 的结果，加上 w1 的高 32 位部分，存入 w1 中
    w1 += x0 * rng_excl1;
    # 计算 x1 乘以 rng_excl1 的结果，加上 w2 和 w1 的高 32 位部分，存入 m1 中
    m1 = x1 * rng_excl1 + w2 + (w1 >> 32);
}
#endif

  return m1;
#endif
}

/* Static `Lemire rejection` function called by
 * random_buffered_bounded_uint32(...) */
static inline uint32_t buffered_bounded_lemire_uint32(
    bitgen_t *bitgen_state, uint32_t rng, int *bcnt, uint32_t *buf) {
  /*
   * 使用Lemire算法 - https://arxiv.org/abs/1805.10941
   *
   * 缓冲区和缓冲区计数在这里未使用，但包含在此函数中以便与相似的uint8和uint16函数模板化
   *
   * 注意：`rng`不应为0xFFFFFFFF。当这种情况发生时，`rng_excl`变为零。
   */
  const uint32_t rng_excl = rng + 1;

  uint64_t m;
  uint32_t leftover;

  assert(rng != 0xFFFFFFFFUL);

  /* 生成一个经过缩放的随机数。*/
  m = ((uint64_t)next_uint32(bitgen_state)) * rng_excl;

  /* 拒绝采样以消除任何偏差 */
  leftover = m & 0xFFFFFFFFUL;

  if (leftover < rng_excl) {
    /* `rng_excl`是`threshold`的简单上限。*/
    const uint32_t threshold = (UINT32_MAX - rng) % rng_excl;

    while (leftover < threshold) {
      m = ((uint64_t)next_uint32(bitgen_state)) * rng_excl;
      leftover = m & 0xFFFFFFFFUL;
    }
  }

  return (m >> 32);
}

/* Static `Lemire rejection` function called by
 * random_buffered_bounded_uint16(...) */
static inline uint16_t buffered_bounded_lemire_uint16(
    bitgen_t *bitgen_state, uint16_t rng, int *bcnt, uint32_t *buf) {
  /*
   * 使用Lemire算法 - https://arxiv.org/abs/1805.10941
   *
   * 注意：`rng`不应为0xFFFF。当这种情况发生时，`rng_excl`变为零。
   */
  const uint16_t rng_excl = rng + 1;

  uint32_t m;
  uint16_t leftover;

  assert(rng != 0xFFFFU);

  /* 生成一个经过缩放的随机数。*/
  m = ((uint32_t)buffered_uint16(bitgen_state, bcnt, buf)) * rng_excl;

  /* 拒绝采样以消除任何偏差 */
  leftover = m & 0xFFFFUL;

  if (leftover < rng_excl) {
    /* `rng_excl`是`threshold`的简单上限。*/
    const uint16_t threshold = (UINT16_MAX - rng) % rng_excl;

    while (leftover < threshold) {
      m = ((uint32_t)buffered_uint16(bitgen_state, bcnt, buf)) * rng_excl;
      leftover = m & 0xFFFFUL;
    }
  }

  return (m >> 16);
}

/* Static `Lemire rejection` function called by
 * random_buffered_bounded_uint8(...) */
static inline uint8_t buffered_bounded_lemire_uint8(bitgen_t *bitgen_state,
                                                    uint8_t rng, int *bcnt,
                                                    uint32_t *buf) {
  /*
   * 使用Lemire算法 - https://arxiv.org/abs/1805.10941
   *
   * 注意：`rng`不应为0xFF。当这种情况发生时，`rng_excl`变为零。
   */
  const uint8_t rng_excl = rng + 1;

  uint16_t m;
  uint8_t leftover;

  assert(rng != 0xFFU);


  /* 生成一个经过缩放的随机数。*/
  m = ((uint16_t)buffered_uint8(bitgen_state, bcnt, buf)) * rng_excl;

  /* 拒绝采样以消除任何偏差 */
  leftover = m & 0xFFUL;

  if (leftover < rng_excl) {
    /* `rng_excl`是`threshold`的简单上限。*/
    const uint8_t threshold = (UINT8_MAX - rng) % rng_excl;

      m = ((uint16_t)buffered_uint8(bitgen_state, bcnt, buf)) * rng_excl;
      leftover = m & 0xFFUL;
    }
  }

  return (m >> 8);
}
    /* `rng_excl` 是 `threshold` 的简单上限。*/
    const uint8_t threshold = (UINT8_MAX - rng) % rng_excl;

    /* 当 `leftover` 小于 `threshold` 时执行循环。*/
    while (leftover < threshold) {
      /* 使用 `bitgen_state`、`bcnt` 和 `buf` 生成一个缓冲区中的无符号 8 位整数，
         将其转换为无符号 16 位整数乘以 `rng_excl`。 */
      m = ((uint16_t)buffered_uint8(bitgen_state, bcnt, buf)) * rng_excl;
      /* 更新 `leftover` 为 `m` 和 0xFF 之间的位与运算结果。 */
      leftover = m & 0xFFUL;
    }

    /* 返回 `m` 右移 8 位后的结果。 */
    return (m >> 8);
/*
 * 返回一个在 off 和 off + rng 之间（包括边界）的随机 npy_uint64 数字。
 * 如果 rng 足够大，数字将会循环。
 */
uint64_t random_bounded_uint64(bitgen_t *bitgen_state, uint64_t off,
                               uint64_t rng, uint64_t mask, bool use_masked) {
  if (rng == 0) {
    // 如果范围为0，直接返回 off
    return off;
  } else if (rng <= 0xFFFFFFFFUL) {
    /* 如果范围在32位内，则调用32位生成器。 */
    if (rng == 0xFFFFFFFFUL) {
      /*
       * 32位 Lemire 方法不支持 rng=0xFFFFFFFF，因此直接调用 next_uint32。
       * 这也适用于 use_masked 为 True 的情况，因此这里处理两种情况。
       */
      return off + (uint64_t) next_uint32(bitgen_state);
    }
    if (use_masked) {
      // 使用掩码的缓冲有界32位无符号整数生成器
      return off + buffered_bounded_masked_uint32(bitgen_state, rng, mask, NULL,
                                                  NULL);
    } else {
      // 使用 Lemire 方法的缓冲有界32位无符号整数生成器
      return off +
             buffered_bounded_lemire_uint32(bitgen_state, rng, NULL, NULL);
    }
  } else if (rng == 0xFFFFFFFFFFFFFFFFULL) {
    /* Lemire64 不支持包含 rng = 0xFFFFFFFFFFFFFFFF 的情况。 */
    return off + next_uint64(bitgen_state);
  } else {
    if (use_masked) {
      // 使用掩码的有界64位无符号整数生成器
      return off + bounded_masked_uint64(bitgen_state, rng, mask);
    } else {
      // 使用 Lemire 方法的有界64位无符号整数生成器
      return off + bounded_lemire_uint64(bitgen_state, rng);
    }
  }
}

/*
 * 返回一个在 off 和 off + rng 之间（包括边界）的随机 npy_uint32 数字。
 * 如果 rng 足够大，数字将会循环。
 */
uint32_t random_buffered_bounded_uint32(bitgen_t *bitgen_state, uint32_t off,
                                        uint32_t rng, uint32_t mask,
                                        bool use_masked, int *bcnt,
                                        uint32_t *buf) {
  /*
   * 未使用的 bcnt 和 buf 仅用于允许与其他 uint 生成器进行模板化。
   */
  if (rng == 0) {
    // 如果范围为0，直接返回 off
    return off;
  } else if (rng == 0xFFFFFFFFUL) {
    /* Lemire32 不支持包含 rng = 0xFFFFFFFF 的情况。 */
    return off + next_uint32(bitgen_state);
  } else {
    if (use_masked) {
      // 使用掩码的缓冲有界32位无符号整数生成器
      return off +
             buffered_bounded_masked_uint32(bitgen_state, rng, mask, bcnt, buf);
    } else {
      // 使用 Lemire 方法的缓冲有界32位无符号整数生成器
      return off + buffered_bounded_lemire_uint32(bitgen_state, rng, bcnt, buf);
    }
  }
}

/*
 * 返回一个在 off 和 off + rng 之间（包括边界）的随机 npy_uint16 数字。
 * 如果 rng 足够大，数字将会循环。
 */
uint16_t random_buffered_bounded_uint16(bitgen_t *bitgen_state, uint16_t off,
                                        uint16_t rng, uint16_t mask,
                                        bool use_masked, int *bcnt,
                                        uint32_t *buf) {
  if (rng == 0) {
    // 如果范围为0，直接返回 off
    return off;
  } else if (rng == 0xFFFFUL) {
    /* Lemire16 不支持包含 rng = 0xFFFF 的情况。 */
    return off + buffered_uint16(bitgen_state, bcnt, buf);
  } else {
    if (use_masked) {
      // 使用掩码的缓冲有界16位无符号整数生成器
      return off +
             buffered_bounded_masked_uint16(bitgen_state, rng, mask, bcnt, buf);
    } else {
      # 如果不满足上述条件，执行以下代码块
      # 返回调用函数 off + buffered_bounded_lemire_uint16 的结果
      return off + buffered_bounded_lemire_uint16(bitgen_state, rng, bcnt, buf);
    }
/*
 * 返回一个介于 off 和 off + rng 之间（包括边界）的随机 npy_uint8 数字。
 * 如果 rng 很大，则数字会循环。
 */
uint8_t random_buffered_bounded_uint8(bitgen_t *bitgen_state, uint8_t off,
                                      uint8_t rng, uint8_t mask,
                                      bool use_masked, int *bcnt,
                                      uint32_t *buf) {
  if (rng == 0) {
    // 如果 rng 为 0，则直接返回 off
    return off;
  } else if (rng == 0xFFUL) {
    /* Lemire8 不支持 inclusive rng = 0xFF。 */
    // 如果 rng 是 0xFF，调用 buffered_uint8 获取随机数
    return off + buffered_uint8(bitgen_state, bcnt, buf);
  } else {
    if (use_masked) {
      // 如果 use_masked 为 true，使用 masked 方法获取随机数
      return off +
             buffered_bounded_masked_uint8(bitgen_state, rng, mask, bcnt, buf);
    } else {
      // 否则使用 Lemire 方法获取随机数
      return off + buffered_bounded_lemire_uint8(bitgen_state, rng, bcnt, buf);
    }
  }
}

/*
 * 返回一个随机的 npy_bool 值，介于 off 和 off + rng 之间（包括边界）。
 */
npy_bool random_buffered_bounded_bool(bitgen_t *bitgen_state, npy_bool off,
                                      npy_bool rng, npy_bool mask,
                                      bool use_masked, int *bcnt,
                                      uint32_t *buf) {
  // 调用 buffered_bounded_bool 函数返回随机的 npy_bool 值
  return buffered_bounded_bool(bitgen_state, off, rng, mask, bcnt, buf);
}

/*
 * 用 cnt 个随机 npy_uint64 数字填充数组 out，这些数字介于 off 和 off + rng 之间（包括边界）。
 * 如果 rng 很大，则数字会循环。
 */
void random_bounded_uint64_fill(bitgen_t *bitgen_state, uint64_t off,
                                uint64_t rng, npy_intp cnt, bool use_masked,
                                uint64_t *out) {
  npy_intp i;

  if (rng == 0) {
    // 如果 rng 为 0，将数组 out 填充为 off
    for (i = 0; i < cnt; i++) {
      out[i] = off;
    }
  } else if (rng <= 0xFFFFFFFFUL) {
    /* 如果范围在 32 位内，则调用 32 位生成器。 */

    /*
     * 32 位 Lemire 方法不处理 rng=0xFFFFFFFF 的情况，因此我们直接调用 next_uint32。
     * 当 use_masked 为 true 时，这也适用，因此我们在此处理两种情况。
     */
    if (rng == 0xFFFFFFFFUL) {
      // 如果 rng 是 0xFFFFFFFF，使用 next_uint32 填充数组 out
      for (i = 0; i < cnt; i++) {
        out[i] = off + (uint64_t) next_uint32(bitgen_state);
      }
    } else {
      uint32_t buf = 0;
      int bcnt = 0;

      if (use_masked) {
        /* 最小的位掩码 >= max */
        uint64_t mask = gen_mask(rng);

        // 使用 masked 方法填充数组 out
        for (i = 0; i < cnt; i++) {
          out[i] = off + buffered_bounded_masked_uint32(bitgen_state, rng, mask,
                                                        &bcnt, &buf);
        }
      } else {
        // 使用 Lemire 方法填充数组 out
        for (i = 0; i < cnt; i++) {
          out[i] = off +
                   buffered_bounded_lemire_uint32(bitgen_state, rng, &bcnt, &buf);
        }
      }
    }
  } else if (rng == 0xFFFFFFFFFFFFFFFFULL) {
    /* Lemire64 不支持 rng = 0xFFFFFFFFFFFFFFFF。 */
    // 如果 rng 是 0xFFFFFFFFFFFFFFFF，使用 next_uint64 填充数组 out
    for (i = 0; i < cnt; i++) {
      out[i] = off + next_uint64(bitgen_state);
    }
  } else {
    if (use_masked) {
      /* 最小的位掩码 >= max */
      uint64_t mask = gen_mask(rng);

      // 使用 masked 方法填充数组 out
      for (i = 0; i < cnt; i++) {
        out[i] = off + bounded_masked_uint64(bitgen_state, rng, mask);
      }
    } else {
      // 使用 Lemire 方法填充数组 out
      for (i = 0; i < cnt; i++) {
        out[i] = off + buffered_bounded_lemire_uint64(bitgen_state, rng, &bcnt, &buf);
      }
    }
  }
}
    } else {
      # 如果条件不满足，则执行以下代码块
      for (i = 0; i < cnt; i++) {
        # 循环，从 i = 0 开始，直到 i < cnt 结束
        out[i] = off + bounded_lemire_uint64(bitgen_state, rng);
        # 将 out 数组的第 i 个元素赋值为 off 加上 bounded_lemire_uint64 函数返回的随机数
      }
    }
  }
/*
 * Fills an array with cnt random npy_uint32 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint32_fill(bitgen_t *bitgen_state, uint32_t off,
                                uint32_t rng, npy_intp cnt, bool use_masked,
                                uint32_t *out) {
  npy_intp i;
  uint32_t buf = 0;
  int bcnt = 0;

  if (rng == 0) {
    // 如果范围为0，填充数组所有元素为 off
    for (i = 0; i < cnt; i++) {
      out[i] = off;
    }
  } else if (rng == 0xFFFFFFFFUL) {
    /* Lemire32 不支持 rng = 0xFFFFFFFF。*/
    // 如果范围为 0xFFFFFFFF，使用 next_uint32 填充数组元素
    for (i = 0; i < cnt; i++) {
      out[i] = off + next_uint32(bitgen_state);
    }
  } else {
    if (use_masked) {
      // 如果使用掩码方式
      /* Smallest bit mask >= max */
      uint32_t mask = (uint32_t)gen_mask(rng);  // 生成掩码

      // 使用 buffered_bounded_masked_uint32 填充数组元素
      for (i = 0; i < cnt; i++) {
        out[i] = off + buffered_bounded_masked_uint32(bitgen_state, rng, mask,
                                                      &bcnt, &buf);
      }
    } else {
      // 否则使用 buffered_bounded_lemire_uint32 填充数组元素
      for (i = 0; i < cnt; i++) {
        out[i] = off +
                 buffered_bounded_lemire_uint32(bitgen_state, rng, &bcnt, &buf);
      }
    }
  }
}

/*
 * Fills an array with cnt random npy_uint16 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint16_fill(bitgen_t *bitgen_state, uint16_t off,
                                uint16_t rng, npy_intp cnt, bool use_masked,
                                uint16_t *out) {
  npy_intp i;
  uint32_t buf = 0;
  int bcnt = 0;

  if (rng == 0) {
    // 如果范围为0，填充数组所有元素为 off
    for (i = 0; i < cnt; i++) {
      out[i] = off;
    }
  } else if (rng == 0xFFFFUL) {
    /* Lemire16 不支持 rng = 0xFFFF。*/
    // 如果范围为 0xFFFF，使用 buffered_uint16 填充数组元素
    for (i = 0; i < cnt; i++) {
      out[i] = off + buffered_uint16(bitgen_state, &bcnt, &buf);
    }
  } else {
    if (use_masked) {
      // 如果使用掩码方式
      /* Smallest bit mask >= max */
      uint16_t mask = (uint16_t)gen_mask(rng);  // 生成掩码

      // 使用 buffered_bounded_masked_uint16 填充数组元素
      for (i = 0; i < cnt; i++) {
        out[i] = off + buffered_bounded_masked_uint16(bitgen_state, rng, mask,
                                                      &bcnt, &buf);
      }
    } else {
      // 否则使用 buffered_bounded_lemire_uint16 填充数组元素
      for (i = 0; i < cnt; i++) {
        out[i] = off +
                 buffered_bounded_lemire_uint16(bitgen_state, rng, &bcnt, &buf);
      }
    }
  }
}

/*
 * Fills an array with cnt random npy_uint8 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint8_fill(bitgen_t *bitgen_state, uint8_t off, uint8_t rng,
                               npy_intp cnt, bool use_masked, uint8_t *out) {
  npy_intp i;
  uint32_t buf = 0;
  int bcnt = 0;

  if (rng == 0) {
    // 如果范围为0，填充数组所有元素为 off
    for (i = 0; i < cnt; i++) {
      out[i] = off;
    }
  } else if (rng == 0xFFUL) {
    /* Lemire8 不支持 rng = 0xFF。*/
    // 如果范围为 0xFF，使用 buffered_uint8 填充数组元素
    for (i = 0; i < cnt; i++) {
      out[i] = off + buffered_uint8(bitgen_state, &bcnt, &buf);
    }
  } else {
    if (use_masked) {
      /* 如果 use_masked 为真，则使用掩码生成随机数 */
      /* 生成大于等于最小值的掩码 */
      uint8_t mask = (uint8_t)gen_mask(rng);

      /* 对于给定的数量 cnt，循环生成掩码限定范围内的随机数 */
      for (i = 0; i < cnt; i++) {
        /* 调用函数生成带掩码的随机数，并将结果存入 out 数组 */
        out[i] = off + buffered_bounded_masked_uint8(bitgen_state, rng, mask,
                                                     &bcnt, &buf);
      }
    } else {
      /* 如果 use_masked 为假，则使用默认的 Lemire 方法生成随机数 */
      /* 对于给定的数量 cnt，循环生成 Lemire 方法限定范围内的随机数 */
      for (i = 0; i < cnt; i++) {
        /* 调用函数生成 Lemire 方法的随机数，并将结果存入 out 数组 */
        out[i] =
            off + buffered_bounded_lemire_uint8(bitgen_state, rng, &bcnt, &buf);
      }
    }
/*
 * Fills an array with cnt random npy_bool between off and off + rng
 * inclusive.
 */
void random_bounded_bool_fill(bitgen_t *bitgen_state, npy_bool off,
                              npy_bool rng, npy_intp cnt, bool use_masked,
                              npy_bool *out) {
  npy_bool mask = 0; // 初始化一个 npy_bool 类型的变量 mask，初始值为 0
  npy_intp i; // 定义一个 npy_intp 类型的变量 i，用于循环计数

  // 循环 cnt 次，填充数组 out，每次调用 buffered_bounded_bool 函数获取随机值
  for (i = 0; i < cnt; i++) {
    out[i] = buffered_bounded_bool(bitgen_state, off, rng, mask, &bcnt, &buf);
  }
}

/*
 * Generates multinomial random numbers based on given probabilities.
 */
void random_multinomial(bitgen_t *bitgen_state, RAND_INT_TYPE n,
                        RAND_INT_TYPE *mnix, double *pix, npy_intp d,
                        binomial_t *binomial) {
  double remaining_p = 1.0; // 初始化 remaining_p 变量为 1.0，表示剩余的概率
  npy_intp j; // 定义一个 npy_intp 类型的变量 j，用于循环计数
  RAND_INT_TYPE dn = n; // 将参数 n 赋值给变量 dn

  // 循环 d-1 次，生成多项式随机数，每次调用 random_binomial 函数
  for (j = 0; j < (d - 1); j++) {
    mnix[j] = random_binomial(bitgen_state, pix[j] / remaining_p, dn, binomial);
    dn = dn - mnix[j]; // 更新剩余的随机数总数
    if (dn <= 0) {
      break; // 如果剩余随机数总数小于等于 0，跳出循环
    }
    remaining_p -= pix[j]; // 更新剩余的概率
  }

  // 如果仍有剩余随机数总数大于 0，将剩余随机数放入最后一个元素
  if (dn > 0) {
      mnix[d - 1] = dn;
  }
}
```