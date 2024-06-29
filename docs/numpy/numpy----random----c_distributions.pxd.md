# `D:\src\scipysrc\numpy\numpy\random\c_distributions.pxd`

```
# 导入必要的模块和类型定义
# 设置 Cython 编译选项：禁用循环包装、空指针检查、边界检查，启用 C 除法，语言级别为 Python 3
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3

# 从 numpy 中导入指定类型 npy_intp
from numpy cimport npy_intp

# 从 C 标准库中导入指定类型和结构体
from libc.stdint cimport (uint64_t, int32_t, int64_t)
# 从 numpy.random 模块中导入 bitgen_t 类型
from numpy.random cimport bitgen_t

# 从 numpy/random/distributions.h 头文件中引入以下函数和结构体的定义

# 定义二项分布结构体 s_binomial_t，包含多个字段
cdef extern from "numpy/random/distributions.h":

    struct s_binomial_t:
        int has_binomial
        double psave
        int64_t nsave
        double r
        double q
        double fm
        int64_t m
        double p1
        double xm
        double xl
        double xr
        double c
        double laml
        double lamr
        double p2
        double p3
        double p4

    # 为 s_binomial_t 结构体定义别名 binomial_t
    ctypedef s_binomial_t binomial_t

    # 下面列出了一系列用于生成随机数的函数声明，这些函数使用 bitgen_t 类型的参数，并在无 GIL 的环境下执行

    # 生成标准均匀分布的随机浮点数，双精度版本
    float random_standard_uniform_f(bitgen_t *bitgen_state) nogil
    # 生成标准均匀分布的随机双精度数，无 GIL
    double random_standard_uniform(bitgen_t *bitgen_state) nogil
    # 生成标准均匀分布的随机浮点数数组，无 GIL
    void random_standard_uniform_fill(bitgen_t* bitgen_state, npy_intp cnt, double *out) nogil
    # 生成标准均匀分布的随机单精度数数组，无 GIL
    void random_standard_uniform_fill_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) nogil
    
    # 生成标准指数分布的随机双精度数，无 GIL
    double random_standard_exponential(bitgen_t *bitgen_state) nogil
    # 生成标准指数分布的随机单精度数，无 GIL
    float random_standard_exponential_f(bitgen_t *bitgen_state) nogil
    # 生成标准指数分布的随机双精度数数组，无 GIL
    void random_standard_exponential_fill(bitgen_t *bitgen_state, npy_intp cnt, double *out) nogil
    # 生成标准指数分布的随机单精度数数组，无 GIL
    void random_standard_exponential_fill_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) nogil
    # 生成标准指数分布的逆函数的随机双精度数数组，无 GIL
    void random_standard_exponential_inv_fill(bitgen_t *bitgen_state, npy_intp cnt, double *out) nogil
    # 生成标准指数分布的逆函数的随机单精度数数组，无 GIL
    void random_standard_exponential_inv_fill_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) nogil
    
    # 生成标准正态分布的随机双精度数，无 GIL
    double random_standard_normal(bitgen_t* bitgen_state) nogil
    # 生成标准正态分布的随机单精度数，无 GIL
    float random_standard_normal_f(bitgen_t *bitgen_state) nogil
    # 生成标准正态分布的随机双精度数数组，无 GIL
    void random_standard_normal_fill(bitgen_t *bitgen_state, npy_intp count, double *out) nogil
    # 生成标准正态分布的随机单精度数数组，无 GIL
    void random_standard_normal_fill_f(bitgen_t *bitgen_state, npy_intp count, float *out) nogil
    # 生成标准 Gamma 分布的随机双精度数，无 GIL
    double random_standard_gamma(bitgen_t *bitgen_state, double shape) nogil
    # 生成标准 Gamma 分布的随机单精度数，无 GIL
    float random_standard_gamma_f(bitgen_t *bitgen_state, float shape) nogil

    # 下面是重复声明，已经在前面出现过的函数，这里再次声明，保持完整性

    float random_standard_uniform_f(bitgen_t *bitgen_state) nogil
    void random_standard_uniform_fill_f(bitgen_t* bitgen_state, npy_intp cnt, float *out) nogil
    float random_standard_normal_f(bitgen_t* bitgen_state) nogil
    float random_standard_gamma_f(bitgen_t *bitgen_state, float shape) nogil

    # 生成正整数范围内的随机 int64_t 数，无 GIL
    int64_t random_positive_int64(bitgen_t *bitgen_state) nogil
    # 生成正整数范围内的随机 int32_t 数，无 GIL
    int32_t random_positive_int32(bitgen_t *bitgen_state) nogil
    # 生成正整数范围内的随机 int64_t 数，无 GIL
    int64_t random_positive_int(bitgen_t *bitgen_state) nogil
    # 生成随机的 uint64_t 数，无 GIL
    uint64_t random_uint(bitgen_t *bitgen_state) nogil

    # 生成正态分布的随机双精度数，指定均值和标准差，无 GIL
    double random_normal(bitgen_t *bitgen_state, double loc, double scale) nogil

    # 生成 Gamma 分布的随机双精度数，指定形状和尺度参数，无 GIL
    double random_gamma(bitgen_t *bitgen_state, double shape, double scale) nogil
    # 生成 Gamma 分布的随机单精度数，指定形状和尺度参数，无 GIL
    float random_gamma_f(bitgen_t *bitgen_state, float shape, float scale) nogil

    # 生成指数分布的随机双精度数，指定尺度参数，无 GIL
    double random_exponential(bitgen_t *bitgen_state, double scale) nogil
    # 生成均匀分布的随机双精度数，指定下限和范围，无 GIL
    double random_uniform(bitgen_t *bitgen_state, double lower, double range) nogil
    // 生成符合 Beta 分布的随机数
    double random_beta(bitgen_t *bitgen_state, double a, double b) nogil
    
    // 生成符合自由度为 df 的卡方分布的随机数
    double random_chisquare(bitgen_t *bitgen_state, double df) nogil
    
    // 生成符合自由度为 dfnum 和 dfden 的 F 分布的随机数
    double random_f(bitgen_t *bitgen_state, double dfnum, double dfden) nogil
    
    // 生成符合标准 Cauchy 分布的随机数
    double random_standard_cauchy(bitgen_t *bitgen_state) nogil
    
    // 生成符合参数 a 的 Pareto 分布的随机数
    double random_pareto(bitgen_t *bitgen_state, double a) nogil
    
    // 生成符合参数 a 的 Weibull 分布的随机数
    double random_weibull(bitgen_t *bitgen_state, double a) nogil
    
    // 生成符合参数 a 的 Power 分布的随机数
    double random_power(bitgen_t *bitgen_state, double a) nogil
    
    // 生成符合参数 loc 和 scale 的 Laplace (Double Exponential) 分布的随机数
    double random_laplace(bitgen_t *bitgen_state, double loc, double scale) nogil
    
    // 生成符合参数 loc 和 scale 的 Gumbel (Extreme Value Type I) 分布的随机数
    double random_gumbel(bitgen_t *bitgen_state, double loc, double scale) nogil
    
    // 生成符合参数 loc 和 scale 的 Logistic 分布的随机数
    double random_logistic(bitgen_t *bitgen_state, double loc, double scale) nogil
    
    // 生成符合参数 mean 和 sigma 的对数正态分布的随机数
    double random_lognormal(bitgen_t *bitgen_state, double mean, double sigma) nogil
    
    // 生成符合参数 mode 的 Rayleigh 分布的随机数
    double random_rayleigh(bitgen_t *bitgen_state, double mode) nogil
    
    // 生成符合自由度为 df 的标准 t 分布的随机数
    double random_standard_t(bitgen_t *bitgen_state, double df) nogil
    
    // 生成符合非中心卡方分布的随机数，参数为自由度 df 和非中心参数 nonc
    double random_noncentral_chisquare(bitgen_t *bitgen_state, double df, double nonc) nogil
    
    // 生成符合非中心 F 分布的随机数，参数为自由度 dfnum、dfden 和非中心参数 nonc
    double random_noncentral_f(bitgen_t *bitgen_state, double dfnum, double dfden, double nonc) nogil
    
    // 生成符合参数 mean 和 scale 的 Wald (Inverse Gaussian) 分布的随机数
    double random_wald(bitgen_t *bitgen_state, double mean, double scale) nogil
    
    // 生成符合参数 mu 和 kappa 的 von Mises 分布的随机数
    double random_vonmises(bitgen_t *bitgen_state, double mu, double kappa) nogil
    
    // 生成符合参数 left、mode 和 right 的三角分布的随机数
    double random_triangular(bitgen_t *bitgen_state, double left, double mode, double right) nogil
    
    // 生成符合参数 lam 的泊松分布的随机整数
    int64_t random_poisson(bitgen_t *bitgen_state, double lam) nogil
    
    // 生成符合参数 n 和 p 的负二项分布的随机整数
    int64_t random_negative_binomial(bitgen_t *bitgen_state, double n, double p) nogil
    
    // 生成符合参数 p 和 n 的二项分布的随机整数
    int64_t random_binomial(bitgen_t *bitgen_state, double p, int64_t n, binomial_t *binomial) nogil
    
    // 生成符合参数 p 的对数级数分布的随机整数
    int64_t random_logseries(bitgen_t *bitgen_state, double p) nogil
    
    // 使用搜索法生成符合参数 p 的几何分布的随机整数
    int64_t random_geometric_search(bitgen_t *bitgen_state, double p) nogil
    
    // 使用反演法生成符合参数 p 的几何分布的随机整数
    int64_t random_geometric_inversion(bitgen_t *bitgen_state, double p) nogil
    
    // 生成符合参数 p 的几何分布的随机整数
    int64_t random_geometric(bitgen_t *bitgen_state, double p) nogil
    
    // 生成符合参数 a 的 Zipf 分布的随机整数
    int64_t random_zipf(bitgen_t *bitgen_state, double a) nogil
    
    // 生成符合参数 good、bad 和 sample 的超几何分布的随机整数
    int64_t random_hypergeometric(bitgen_t *bitgen_state, int64_t good, int64_t bad, int64_t sample) nogil
    
    // 生成在闭区间 [0, max] 内的随机整数
    uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max) nogil
    
    // 生成在闭区间 [off, off + rng] 内的随机 uint64 数字
    uint64_t random_bounded_uint64(bitgen_t *bitgen_state, uint64_t off, uint64_t rng, uint64_t mask, bint use_masked) nogil
    
    // 生成符合多项式分布的随机数，参数为 n（试验次数）、mnix（结果数组）、pix（概率数组）、d（数组长度）、binomial（二项分布辅助对象）
    void random_multinomial(bitgen_t *bitgen_state, int64_t n, int64_t *mnix, double *pix, npy_intp d, binomial_t *binomial) nogil
    # 计算多元超几何分布的随机数计数
    int random_multivariate_hypergeometric_count(bitgen_t *bitgen_state,
                              int64_t total,
                              size_t num_colors, int64_t *colors,
                              int64_t nsample,
                              size_t num_variates, int64_t *variates) nogil
    {
        # 用于存储每个变量的超几何分布计数结果
        int result;
        # 使用快速判断方法，初始化bitgen_state
        setup_bitgen_for_normal_distribution(bitgen_state);
        for (size_t i = 0; i < num_variates; ++i) {
            # 采用较大的数，use  max_value from the colors[i]
            to_max_value[i] += total;
        }
      '
```