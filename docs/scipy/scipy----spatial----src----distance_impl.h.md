# `D:\src\scipysrc\scipy\scipy\spatial\src\distance_impl.h`

```
/**
 * Author: Damian Eads
 * Date:   September 22, 2007 (moved to new file on June 8, 2008)
 *
 * Copyright (c) 2007, 2008, Damian Eads. All rights reserved.
 * Adapted for incorporation into Scipy, April 9, 2008.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   - Redistributions of source code must retain the above
 *     copyright notice, this list of conditions and the
 *     following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer
 *     in the documentation and/or other materials provided with the
 *     distribution.
 *   - Neither the name of the author nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>

static inline void
_row_norms(const double *X, npy_intp num_rows, const npy_intp num_cols, double *norms_buff){
    /* Compute the row norms. */
    npy_intp i, j;
    for (i = 0; i < num_rows; ++i) {
        for (j = 0; j < num_cols; ++j, ++X) {
            const double curr_val = *X;
            norms_buff[i] += curr_val * curr_val;
        }
        norms_buff[i] = sqrt(norms_buff[i]);
    }
}

static inline double
sqeuclidean_distance_double(const double *u, const double *v, const npy_intp n)
{
    double s = 0.0;
    npy_intp i;

    for (i = 0; i < n; ++i) {
        const double d = u[i] - v[i];
        s += d * d;
    }
    return s;
}

static inline double
euclidean_distance_double(const double *u, const double *v, const npy_intp n)
{
    /* Calculate the Euclidean distance between vectors u and v. */
    return sqrt(sqeuclidean_distance_double(u, v, n));
}

#if 0   /* XXX unused */
static inline double
ess_distance(const double *u, const double *v, const npy_intp n)
{
    double s = 0.0, d;
    npy_intp i;

    for (i = 0; i < n; ++i) {
        d = fabs(u[i] - v[i]);
        s += d * d;
    }
    return s;
}
#endif

static inline double
chebyshev_distance_double(const double *u, const double *v, const npy_intp n)
{
    /* Calculate the Chebyshev distance between vectors u and v. */
    double maxv = 0.0;
    npy_intp i;
    # 循环遍历索引 i 从 0 到 n-1
    for (i = 0; i < n; ++i) {
        # 计算 u[i] 和 v[i] 之间的绝对差值
        const double d = fabs(u[i] - v[i]);
        # 如果 d 大于当前的最大值 maxv，则更新 maxv
        if (d > maxv) {
            maxv = d;
        }
    }
    # 返回最大差值 maxv
    return maxv;
/*
 * 计算加权切比雪夫距离（双精度浮点数版本）
 * 输入参数:
 * - u: 第一个向量的指针
 * - v: 第二个向量的指针
 * - n: 向量的长度
 * - w: 权重数组的指针，用于加权距离计算
 * 返回值:
 * - 最大距离值
 */
static inline double
weighted_chebyshev_distance_double(const double *u, const double *v,
                                   const npy_intp n, const double *w)
{
    npy_intp i;
    double maxv = 0.0;
    
    for (i = 0; i < n; ++i) {
        if (w[i] == 0.0) continue;  // 如果权重为零，则跳过当前维度的计算
        const double d = fabs(u[i] - v[i]);  // 计算当前维度上的绝对差值
        if (d > maxv) {
            maxv = d;  // 更新最大距离值
        }
    }
    
    return maxv;
}

/*
 * 计算Canberra距离（双精度浮点数版本）
 * 输入参数:
 * - u: 第一个向量的指针
 * - v: 第二个向量的指针
 * - n: 向量的长度
 * 返回值:
 * - Canberra距离总和
 */
static inline double
canberra_distance_double(const double *u, const double *v, const npy_intp n)
{
    double tot = 0.;
    npy_intp i;

    for (i = 0; i < n; ++i) {
        const double x = u[i], y = v[i];
        const double snum = fabs(x - y);  // 计算当前维度上的绝对差值
        const double sdenom = fabs(x) + fabs(y);  // 计算当前维度上的绝对值之和
        if (sdenom > 0.) {
            tot += snum / sdenom;  // 更新Canberra距离总和
        }
    }
    
    return tot;
}

/*
 * 计算Bray-Curtis距离（双精度浮点数版本）
 * 输入参数:
 * - u: 第一个向量的指针
 * - v: 第二个向量的指针
 * - n: 向量的长度
 * 返回值:
 * - Bray-Curtis距离
 */
static inline double
bray_curtis_distance_double(const double *u, const double *v, const npy_intp n)
{
    double s1 = 0.0, s2 = 0.0;
    npy_intp i;

    for (i = 0; i < n; ++i) {
        s1 += fabs(u[i] - v[i]);  // 计算当前维度上的绝对差值之和
        s2 += fabs(u[i] + v[i]);  // 计算当前维度上的绝对值之和
    }
    
    return s1 / s2;  // 返回Bray-Curtis距离
}

/*
 * 计算点积（双精度浮点数版本）
 * 输入参数:
 * - u: 第一个向量的指针
 * - v: 第二个向量的指针
 * - n: 向量的长度
 * 返回值:
 * - 点积结果
 */
static inline double
dot_product(const double *u, const double *v, const npy_intp n)
{
    double s = 0.0;
    npy_intp i;

    for (i = 0; i < n; ++i) {
        s += u[i] * v[i];  // 计算点积
    }
    
    return s;  // 返回点积结果
}

/*
 * 计算马氏距离（双精度浮点数版本）
 * 输入参数:
 * - u: 第一个向量的指针
 * - v: 第二个向量的指针
 * - covinv: 协方差矩阵的逆的指针（用于马氏距离计算）
 * - dimbuf1: 第一个缓冲区的指针
 * - dimbuf2: 第二个缓冲区的指针
 * - n: 向量的长度
 * 返回值:
 * - 马氏距离
 */
static inline double
mahalanobis_distance(const double *u, const double *v, const double *covinv,
                     double *dimbuf1, double *dimbuf2, const npy_intp n)
{
    npy_intp i;

    for (i = 0; i < n; ++i) {
        dimbuf1[i] = u[i] - v[i];  // 计算两个向量的差值
    }
    
    /*
     * 注意: 矩阵-向量乘法（GEMV）。对于高维数据，OpenBLAS可以加速这一过程。
     */
    for (i = 0; i < n; ++i) {
        const double *covrow = covinv + (i * n);
        dimbuf2[i] = dot_product(dimbuf1, covrow, n);  // 计算马氏距离所需的中间向量
    }
    
    return sqrt(dot_product(dimbuf1, dimbuf2, n));  // 返回马氏距离的平方根
}

/*
 * 计算汉明距离（双精度浮点数版本）
 * 输入参数:
 * - u: 第一个向量的指针
 * - v: 第二个向量的指针
 * - n: 向量的长度
 * - w: 权重数组的指针，用于加权距离计算
 * 返回值:
 * - 加权汉明距离
 */
static inline double
hamming_distance_double(const double *u, const double *v, const npy_intp n, const double *w)
{
    npy_intp i;
    double s = 0;
    double w_sum = 0;

    for (i = 0; i < n; ++i) {
        s += ((double) (u[i] != v[i])) * w[i];  // 计算不同位置上的加权汉明距离
        w_sum += w[i];  // 计算权重的总和
    }

    return s / w_sum;  // 返回加权汉明距离
}

/*
 * 计算汉明距离（字符版本）
 * 输入参数:
 * - u: 第一个字符串的指针
 * - v: 第二个字符串的指针
 * - n: 字符串的长度
 * - w: 权重数组的指针，用于加权距离计算
 * 返回值:
 * - 加权汉明距离
 */
static inline double
hamming_distance_char(const char *u, const char *v, const npy_intp n, const double *w)
{
    npy_intp i;
    double s = 0;
    double w_sum = 0;

    for (i = 0; i < n; ++i) {
        s += ((double) (u[i] != v[i])) * w[i];  // 计算不同位置上的加权汉明距离
        w_sum += w[i];  // 计算权重的总和
    }

    return s / w_sum;  // 返回加权汉明距离
}

/*
 * 计算Yule距离（字符版本）
 * 输入参数:
 * - u: 第一个字符串的指针
 * - v: 第二个字符串的指针
 * - n: 字符串的长度
 * 返回值:
 * - Yule距离
 */
static inline double
yule_distance_char(const char *u, const char *v, const npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, nff = 0, nft = 0, ntf = 0;

    for (i = 0; i < n; ++i) {
        const npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y;   // 计算两个字符串同时为真的数量
        ntf += x & (!y);  //
    # 计算非交叉表格元素 nff
    nff = n - ntt - ntf - nft;
    # 计算 ntf 和 nft 的乘积，并转换为 double 类型
    double half_R = (double)ntf * nft;
    # 如果 half_R 等于 0.0，返回 0.0
    if (half_R == 0.0) {
        return 0.0;
    }
    # 计算 G-test 统计量的值并返回
    return (2. * half_R) / ((double)ntt * nff + half_R);
}

// 计算字符数组 u 和 v 之间的 Dice 距离
static inline double
dice_distance_char(const char *u, const char *v, const npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    // 遍历字符数组，统计相同位置上非零字符的个数和不同字符的个数
    for (i = 0; i < n; ++i) {
        const npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y; // 计算相同位置上都为非零字符的个数
        ndiff += (x != y); // 计算不同位置上字符是否相同
    }
    // 返回 Dice 距离
    return ndiff / (2. * ntt + ndiff);
}

// 计算字符数组 u 和 v 之间的 Rogers-Tanimoto 距离
static inline double
rogerstanimoto_distance_char(const char *u, const char *v, const npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    // 遍历字符数组，统计相同位置上非零字符的个数和不同字符的个数
    for (i = 0; i < n; ++i) {
        const npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y; // 计算相同位置上都为非零字符的个数
        ndiff += (x != y); // 计算不同位置上字符是否相同
    }
    // 返回 Rogers-Tanimoto 距离
    return (2. * ndiff) / ((double)n + ndiff);
}

// 计算字符数组 u 和 v 之间的 Russell-Rao 距离
static inline double
russellrao_distance_char(const char *u, const char *v, const npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0;

    // 遍历字符数组，统计相同位置上都为非零字符的个数
    for (i = 0; i < n; ++i) {
        ntt += (u[i] != 0) & (v[i] != 0); // 计算相同位置上都为非零字符的个数
    }
    // 返回 Russell-Rao 距离
    return (double)(n - ntt) / n;
}

// 计算字符数组 u 和 v 之间的 Kulczynski1 距离
static inline double
kulczynski1_distance_char(const char *u, const char *v, const npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    // 遍历字符数组，统计相同位置上非零字符的个数和不同字符的个数
    for (i = 0; i < n; ++i) {
        const npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y; // 计算相同位置上都为非零字符的个数
        ndiff += (x != y); // 计算不同位置上字符是否相同
    }
    // 返回 Kulczynski1 距离
    return ((double)ntt) / ((double)ndiff);
}

// 计算字符数组 u 和 v 之间的 Sokal-Sneath 距离
static inline double
sokalsneath_distance_char(const char *u, const char *v, const npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    // 遍历字符数组，统计相同位置上非零字符的个数和不同字符的个数
    for (i = 0; i < n; ++i) {
        const npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y; // 计算相同位置上都为非零字符的个数
        ndiff += (x != y); // 计算不同位置上字符是否相同
    }
    // 返回 Sokal-Sneath 距离
    return (2. * ndiff) / (2. * ndiff + ntt);
}

// 计算字符数组 u 和 v 之间的 Sokal-Michener 距离
static inline double
sokalmichener_distance_char(const char *u, const char *v, const npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    // 遍历字符数组，统计相同位置上非零字符的个数和不同字符的个数
    for (i = 0; i < n; ++i) {
        const npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y; // 计算相同位置上都为非零字符的个数
        ndiff += (x != y); // 计算不同位置上字符是否相同
    }
    // 返回 Sokal-Michener 距离
    return (2. * ndiff) / ((double)ndiff + n);
}

// 计算双精度数组 u 和 v 之间的 Jaccard 距离
static inline double
jaccard_distance_double(const double *u, const double *v, const npy_intp n)
{
    npy_intp denom = 0, num = 0;
    npy_intp i;

    // 遍历双精度数组，统计不同位置上数值是否相等且至少一个不为零的个数
    for (i = 0; i < n; ++i) {
        const double x = u[i], y = v[i];
        num += (x != y) & ((x != 0.0) | (y != 0.0)); // 计算不同位置上数值是否相等且至少一个不为零的个数
        denom += (x != 0.0) | (y != 0.0); // 计算至少一个不为零的个数
    }
    // 返回 Jaccard 距离
    return denom == 0.0 ? 0.0 : (double)num / denom;
}

// 计算字符数组 u 和 v 之间的 Jaccard 距离
static inline double
jaccard_distance_char(const char *u, const char *v, const npy_intp n)
{
    npy_intp num = 0, denom = 0;
    npy_intp i;

    // 遍历字符数组，统计不同位置上字符是否相等且至少一个不为零的个数
    for (i = 0; i < n; ++i) {
        const npy_bool x = (u[i] != 0), y = (v[i] != 0);
        num += (x != y); // 计算不同位置上字符是否相等的个数
        denom += x | y; // 计算至少一个不为零的个数
    }
    // 返回 Jaccard 距离
    return denom == 0.0 ? 0.0 : (double)num / denom;
}

// 计算双精度数组 u 和 v 之间的 Jensen-Shannon 距离
static inline double
jensenshannon_distance_double(const double *p, const double *q, const npy_intp n)
{
    npy_intp i;
    double s = 0.0;
    double p_sum = 0.0;
    double q_sum = 0.0;

    // 遍历双精度数组，计算 p 和 q 的总和以及各自元素的和
    for (i = 0; i < n; ++i) {
        if (p[i] < 0.0 || q[i] < 0.0)
            return HUGE_VAL; // 如果数组中出现负数，返回无穷大
        p_sum += p[i]; // 计算数组 p 的总和
        q_sum += q[i]; // 计算数组 q 的总和
    }
    # 如果 p_sum 或 q_sum 为 0，返回一个极大值
    if (p_sum == 0.0 || q_sum == 0.0)
        return HUGE_VAL;

    # 初始化累加器 s，用于计算散度值
    for (i = 0; i < n; ++i) {
        # 计算归一化后的概率 p_i 和 q_i
        const double p_i = p[i] / p_sum;
        const double q_i = q[i] / q_sum;
        # 计算中间值 m_i
        const double m_i = (p_i + q_i) / 2.0;
        # 如果 p_i 大于 0，则累加 KL 散度的一部分
        if (p_i > 0.0)
            s += p_i * log(p_i / m_i);
        # 如果 q_i 大于 0，则累加 KL 散度的一部分
        if (q_i > 0.0)
            s += q_i * log(q_i / m_i);
    }

    # 返回计算得到的 JS 散度的平方根除以 2
    return sqrt(s / 2.0);
static inline double
seuclidean_distance(const double *var, const double *u, const double *v,
                    const npy_intp n)
{
    // 初始化距离累加器
    double s = 0.0;
    // 循环计算每个维度上的标准化欧几里得距离
    npy_intp i;
    for (i = 0; i < n; ++i) {
        // 计算两个向量在当前维度上的差值
        const double d = u[i] - v[i];
        // 累加该维度的平方差值除以方差的结果
        s += (d * d) / var[i];
    }
    // 返回平方根结果作为距离
    return sqrt(s);
}

static inline double
city_block_distance_double(const double *u, const double *v, const npy_intp n)
{
    // 初始化距离累加器
    double s = 0.0;
    // 循环计算曼哈顿距离
    npy_intp i;
    for (i = 0; i < n; ++i) {
        // 累加每个维度上的绝对差值
        s += fabs(u[i] - v[i]);
    }
    // 返回曼哈顿距离结果
    return s;
}

static inline double
minkowski_distance(const double *u, const double *v, const npy_intp n, const double p)
{
    // 初始化距离累加器
    double s = 0.0;
    // 循环计算闵可夫斯基距离
    npy_intp i;
    for (i = 0; i < n; ++i) {
        // 计算两个向量在当前维度上的绝对差值
        const double d = fabs(u[i] - v[i]);
        // 累加绝对差值的 p 次方
        s += pow(d, p);
    }
    // 返回 p 次方根的结果作为距离
    return pow(s, 1.0 / p);
}

static inline double
weighted_minkowski_distance(const double *u, const double *v, const npy_intp n,
                            const double p, const double *w)
{
    // 初始化权重乘积累加器和索引计数器
    npy_intp i = 0;
    double s = 0.0;
    // 循环计算加权闵可夫斯基距离
    for (i = 0; i < n; ++i) {
        // 计算两个向量在当前维度上的绝对差值
        const double d = fabs(u[i] - v[i]);
        // 累加加权绝对差值的 p 次方
        s += pow(d, p) * w[i];
    }
    // 返回 p 次方根的结果作为加权闵可夫斯基距离
    return pow(s, 1.0 / p);
}

#define DEFINE_CDIST(name, type) \
    static int cdist_ ## name ## _ ## type(const type *XA, const type *XB, \
                                           double *dm,                     \
                                           const npy_intp num_rowsA,       \
                                           const npy_intp num_rowsB,       \
                                           const npy_intp num_cols)        \
    {                                                                      \
        // 定义循环索引变量
        Py_ssize_t i, j;                                                   \
        // 循环计算给定距离度量的两组向量之间的距离矩阵
        for (i = 0; i < num_rowsA; ++i) {                                  \
            // 获取第一个向量的起始地址
            const type *u = XA + num_cols * i;                             \
            // 循环处理第二组向量
            for (j = 0; j < num_rowsB; ++j, ++dm) {                        \
                // 获取第二个向量的起始地址
                const type *v = XB + num_cols * j;                         \
                // 计算并存储两个向量之间的距离
                *dm = name ## _distance_ ## type(u, v, num_cols);          \
            }                                                              \
        }                                                                  \
        // 返回成功
        return 0; \
    }

DEFINE_CDIST(bray_curtis, double)
DEFINE_CDIST(canberra, double)
DEFINE_CDIST(chebyshev, double)
DEFINE_CDIST(city_block, double)
DEFINE_CDIST(euclidean, double)
DEFINE_CDIST(jaccard, double)
DEFINE_CDIST(jensenshannon, double)
#define DEFINE_CDIST(sqeuclidean, double)


// 定义一个宏，用于声明平方欧几里得距离的计算函数，参数类型为双精度浮点数
#define DEFINE_CDIST(sqeuclidean, double)

#define DEFINE_CDIST(dice, char)


// 定义一个宏，用于声明 Dice 距离的计算函数，参数类型为字符型
#define DEFINE_CDIST(dice, char)

#define DEFINE_CDIST(jaccard, char)


// 定义一个宏，用于声明 Jaccard 距离的计算函数，参数类型为字符型
#define DEFINE_CDIST(jaccard, char)

#define DEFINE_CDIST(kulczynski1, char)


// 定义一个宏，用于声明 Kulczynski1 距离的计算函数，参数类型为字符型
#define DEFINE_CDIST(kulczynski1, char)

#define DEFINE_CDIST(rogerstanimoto, char)


// 定义一个宏，用于声明 Rogerstanimoto 距离的计算函数，参数类型为字符型
#define DEFINE_CDIST(rogerstanimoto, char)

#define DEFINE_CDIST(russellrao, char)


// 定义一个宏，用于声明 Russellrao 距离的计算函数，参数类型为字符型
#define DEFINE_CDIST(russellrao, char)

#define DEFINE_CDIST(sokalmichener, char)


// 定义一个宏，用于声明 Sokalmichener 距离的计算函数，参数类型为字符型
#define DEFINE_CDIST(sokalmichener, char)

#define DEFINE_CDIST(sokalsneath, char)


// 定义一个宏，用于声明 Sokalsneath 距离的计算函数，参数类型为字符型
#define DEFINE_CDIST(sokalsneath, char)

#define DEFINE_CDIST(yule, char)


// 定义一个宏，用于声明 Yule 距离的计算函数，参数类型为字符型
#define DEFINE_CDIST(yule, char)

#define DEFINE_PDIST(name, type) \
    static int pdist_ ## name ## _ ## type(const type *X, double *dm,       \
                                           const npy_intp num_rows,         \
                                           const npy_intp num_cols)         \
    {                                                                       \
        Py_ssize_t i, j;                                                    \
        double *it = dm;                                                    \
        for (i = 0; i < num_rows; ++i) {                                    \
            const type *u = X + num_cols * i;                               \
            for (j = i + 1; j < num_rows; ++j, it++) {                      \
                const type *v = X + num_cols * j;                           \
                *it = name ## _distance_ ## type(u, v, num_cols);           \
            }                                                               \
        }                                                                   \
        return 0; \
    }


// 定义一个宏，用于生成基于给定距离名称和数据类型的距离计算函数
#define DEFINE_PDIST(name, type) \
    static int pdist_ ## name ## _ ## type(const type *X, double *dm,       \
                                           const npy_intp num_rows,         \
                                           const npy_intp num_cols)         \
    {                                                                       \
        Py_ssize_t i, j;                                                    \
        double *it = dm;                                                    \
        // 遍历每对样本，计算它们之间的距离
        for (i = 0; i < num_rows; ++i) {                                    \
            const type *u = X + num_cols * i;                               \
            for (j = i + 1; j < num_rows; ++j, it++) {                      \
                const type *v = X + num_cols * j;                           \
                *it = name ## _distance_ ## type(u, v, num_cols);           \
            }                                                               \
        }                                                                   \
        return 0;                                                           \
    }

DEFINE_PDIST(bray_curtis, double)


// 定义 Bray-Curtis 距离计算函数，参数类型为双精度浮点数
DEFINE_PDIST(bray_curtis, double)

DEFINE_PDIST(canberra, double)


// 定义 Canberra 距离计算函数，参数类型为双精度浮点数
DEFINE_PDIST(canberra, double)

DEFINE_PDIST(chebyshev, double)


// 定义 Chebyshev 距离计算函数，参数类型为双精度浮点数
DEFINE_PDIST(chebyshev, double)

DEFINE_PDIST(city_block, double)


// 定义 City Block (Manhattan) 距离计算函数，参数类型为双精度浮点数
DEFINE_PDIST(city_block, double)

DEFINE_PDIST(euclidean, double)


// 定义 Euclidean 距离计算函数，参数类型为双精度浮点数
DEFINE_PDIST(euclidean, double)

DEFINE_PDIST(jaccard, double)


// 定义 Jaccard 距离计算函数，参数类型为双精度浮点数
DEFINE_PDIST(jaccard, double)

DEFINE_PDIST(jensenshannon, double)


// 定义 Jensen-Shannon 距离计算函数，参数类型为双精度浮点数
DEFINE_PDIST(jensenshannon, double)

DEFINE_PDIST(sqeuclidean, double)


// 定义平方欧几里得距离计算函数，参数类型为双精度浮点数
DEFINE_PDIST(sqeuclidean, double)

DEFINE_PDIST(dice, char)


// 定义 Dice 距离计算函数，参数类型为字符型
DEFINE_PDIST(dice, char)

DEFINE_PDIST(jaccard, char)


// 定义 Jaccard 距离计算函数，参数类型为字符型
DEFINE_PDIST(jaccard, char)

DEFINE_PDIST(kulczynski1, char)


// 定义 Kulczynski1 距离计算函数，参数类型为字符型
DEFINE_PDIST(kulczynski1, char)

DEFINE_PDIST(rogerstanimoto, char)


// 定义 Rogerstanimoto 距离计算函数，参数类型为字符型
DEFINE_PDIST(rogerstanimoto, char)

DEFINE_PDIST(russellrao, char)


// 定义 Russellrao 距离计算函数，参数类型为字符型
DEFINE_PDIST(russellrao, char)

DEFINE_PDIST(sokalmichener, char)


// 定义 Sokalmichener 距离计算函数，参数类型为字符型
DEFINE_PDIST(sokalmichener, char)

DEFINE_PDIST(sokalsneath, char)


// 定义 Sokalsneath 距离计算函数，参数类型为字符型
DEFINE_PDIST(sokalsneath, char)

DEFINE_PDIST(yule, char)


// 定义 Yule 距离计算函数，参数类型为字符型
DEFINE_PDIST(yule, char)

static inline int
pdist_mahalanobis(const double *X, double *dm, const npy_intp num_rows,
                  const npy_intp num_cols, const double *covinv)
{
    npy_intp i, j;
    double *dimbuf1 = calloc(2 * num_cols, sizeof(double));
    double *dimbuf2;
    if (!dimbuf1) {
        return -1;
    }

    dimbuf2 = dimbuf1 + num_cols;

    for (i = 0; i < num_rows; ++i) {
        const double *u = X + (num_cols * i);
        for (j = i + 1; j < num_rows; ++j, ++dm) {
            const double *v = X + (num_cols * j);
            *dm = mahalanobis_distance(u, v, covinv, dimbuf1, dimbuf2, num_cols);
        }
    }
    free(dimbuf1);
    return 0;
}


// 定义 Mahalanobis 距离计算函数，参数类型为双精度浮点数
static inline int
pdist_mahalanobis(const double *X, double *dm, const npy_intp num_rows,
                  const npy
static inline int
pdist_cosine(const double *X, double *dm, const npy_intp num_rows,
             const npy_intp num_cols)
{
    double cosine;                  // 存储余弦相似度
    npy_intp i, j;                  // 循环变量

    double * norms_buff = calloc(num_rows, sizeof(double));  // 分配存储行范数的数组内存
    if (!norms_buff)
        return -1;                  // 如果内存分配失败，则返回错误代码

    _row_norms(X, num_rows, num_cols, norms_buff);  // 计算每行的范数

    for (i = 0; i < num_rows; ++i) {
        const double *u = X + (num_cols * i);  // 获取第 i 行的起始地址
        for (j = i + 1; j < num_rows; ++j, ++dm) {
            const double *v = X + (num_cols * j);  // 获取第 j 行的起始地址
            cosine = dot_product(u, v, num_cols) / (norms_buff[i] * norms_buff[j]);  // 计算余弦相似度
            if (fabs(cosine) > 1.) {
                /* Clip to correct rounding error. */
                cosine = copysign(1, cosine);  // 如果余弦相似度超过1，进行修正
            }
            *dm = 1. - cosine;  // 计算余弦距离并存储到距离矩阵中
        }
    }
    free(norms_buff);  // 释放行范数数组的内存
    return 0;  // 返回成功标志
}

static inline int
pdist_seuclidean(const double *X, const double *var, double *dm,
                 const npy_intp num_rows, const npy_intp num_cols)
{
    npy_intp i, j;  // 循环变量

    for (i = 0; i < num_rows; ++i) {
        const double *u = X + (num_cols * i);  // 获取第 i 行的起始地址
        for (j = i + 1; j < num_rows; ++j, ++dm) {
            const double *v = X + (num_cols * j);  // 获取第 j 行的起始地址
            *dm = seuclidean_distance(var, u, v, num_cols);  // 计算标准化欧氏距离并存储到距离矩阵中
        }
    }
    return 0;  // 返回成功标志
}

static inline int
pdist_minkowski(const double *X, double *dm, npy_intp num_rows,
                const npy_intp num_cols, const double p)
{
    npy_intp i, j;  // 循环变量
    if (p == 1.0) {
        return pdist_city_block_double(X, dm, num_rows, num_cols);  // 如果 p 为 1，计算曼哈顿距离
    }
    if (p == 2.0) {
        return pdist_euclidean_double(X, dm, num_rows, num_cols);  // 如果 p 为 2，计算欧氏距离
    }
    if (isinf(p)) {
        return pdist_chebyshev_double(X, dm, num_rows, num_cols);  // 如果 p 为无穷大，计算切比雪夫距离
    }

    for (i = 0; i < num_rows; ++i) {
        const double *u = X + (num_cols * i);  // 获取第 i 行的起始地址
        for (j = i + 1; j < num_rows; ++j, ++dm) {
            const double *v = X + (num_cols * j);  // 获取第 j 行的起始地址
            *dm = minkowski_distance(u, v, num_cols, p);  // 计算闵可夫斯基距离并存储到距离矩阵中
        }
    }
    return 0;  // 返回成功标志
}

static inline int
pdist_weighted_minkowski(const double *X, double *dm, npy_intp num_rows,
                         const npy_intp num_cols, const double p, const double *w)
{
    npy_intp i, j;  // 循环变量

    if (isinf(p)) {
        return pdist_weighted_chebyshev(X, dm, num_rows, num_cols, w);  // 如果 p 为无穷大，计算加权切比雪夫距离
    }

    for (i = 0; i < num_rows; ++i) {
        const double *u = X + (num_cols * i);  // 获取第 i 行的起始地址
        for (j = i + 1; j < num_rows; ++j, ++dm) {
            const double *v = X + (num_cols * j);  // 获取第 j 行的起始地址
            *dm = weighted_minkowski_distance(u, v, num_cols, p, w);  // 计算加权闵可夫斯基距离并存储到距离矩阵中
        }
    }
    return 0;  // 返回成功标志
}

static inline int
pdist_hamming_double(const double *X, double *dm, npy_intp num_rows,
                         const npy_intp num_cols, const double *w)
{
    npy_intp i, j;  // 循环变量

    for (i = 0; i < num_rows; ++i) {
        const double *u = X + (num_cols * i);  // 获取第 i 行的起始地址
        for (j = i + 1; j < num_rows; ++j, ++dm) {
            const double *v = X + (num_cols * j);  // 获取第 j 行的起始地址
            *dm = hamming_distance_double(u, v, num_cols, w);  // 计算汉明距离并存储到距离矩阵中
        }
    }
    // 注意：此函数似乎缺少最后的大括号 '}'，可能是复制错误或代码截断的问题
}
    }
    // 函数执行完毕，返回整数值 0
    return 0;
/** cdist */
static inline int
cdist_cosine(const double *XA, const double *XB, double *dm, const npy_intp num_rowsA,
             const npy_intp num_rowsB, const npy_intp num_cols)
{
    // 声明变量 cosine、i、j，用于循环和计算余弦距离
    double cosine;
    npy_intp i, j;

    // 分配并初始化一个大小为 num_rowsA + num_rowsB 的双精度浮点数数组
    double * norms_buffA = calloc(num_rowsA + num_rowsB, sizeof(double));
    double * norms_buffB;
    // 如果分配内存失败，返回错误码 -1
    if (!norms_buffA)
        return -1;

    // 将 norms_buffB 指针指向 norms_buffA 后面的一半空间
    norms_buffB = norms_buffA + num_rowsA;

    // 计算 XA 矩阵的每行的范数并存储在 norms_buffA 中
    _row_norms(XA, num_rowsA, num_cols, norms_buffA);
    // 计算 XB 矩阵的每行的范数并存储在 norms_buffB 中
    _row_norms(XB, num_rowsB, num_cols, norms_buffB);
    // 遍历矩阵A的每一行
    for (i = 0; i < num_rowsA; ++i) {
        // 获取矩阵A第i行的起始地址
        const double *u = XA + (num_cols * i);
        // 遍历矩阵B的每一行
        for (j = 0; j < num_rowsB; ++j, ++dm) {
            // 获取矩阵B第j行的起始地址
            const double *v = XB + (num_cols * j);
            // 计算两行向量的余弦相似度
            cosine = dot_product(u, v, num_cols) / (norms_buffA[i] * norms_buffB[j]);
            // 如果余弦相似度超过1，进行修正以避免由于舍入误差而导致的错误值
            if (fabs(cosine) > 1.) {
                /* Clip to correct rounding error. */
                cosine = copysign(1, cosine);
            }
            // 计算两行向量的距离度量，并存储在dm指向的位置
            *dm = 1. - cosine;
        }
    }
    // 释放 norms_buffA 数组占用的内存
    free(norms_buffA);
    // 返回操作成功的标志
    return 0;
// 静态内联函数：计算马氏距离
static inline int
cdist_mahalanobis(const double *XA, const double *XB, double *dm,
                  const npy_intp num_rowsA, const npy_intp num_rowsB,
                  const npy_intp num_cols, const double *covinv)
{
    npy_intp i, j;

    // 分配临时内存空间，用于计算马氏距离
    double *dimbuf1 = calloc(2 * num_cols, sizeof(double));
    double *dimbuf2;
    if (!dimbuf1) {
        return -1;  // 分配失败时返回错误码
    }
    dimbuf2 = dimbuf1 + num_cols;

    // 计算每对向量之间的马氏距离
    for (i = 0; i < num_rowsA; ++i) {
        const double *u = XA + (num_cols * i);
        for (j = 0; j < num_rowsB; ++j, ++dm) {
            const double *v = XB + (num_cols * j);
            *dm = mahalanobis_distance(u, v, covinv, dimbuf1, dimbuf2, num_cols);
        }
    }
    
    free(dimbuf1);  // 释放临时内存空间
    return 0;  // 返回计算成功
}

// 静态内联函数：计算标准化欧几里得距离
static inline int
cdist_seuclidean(const double *XA, const double *XB, const double *var,
                 double *dm, const npy_intp num_rowsA, const npy_intp num_rowsB,
                 const npy_intp num_cols)
{
    npy_intp i, j;

    // 计算每对向量之间的标准化欧几里得距离
    for (i = 0; i < num_rowsA; ++i) {
        const double *u = XA + (num_cols * i);
        for (j = 0; j < num_rowsB; ++j, ++dm) {
            const double *v = XB + (num_cols * j);
            *dm = seuclidean_distance(var, u, v, num_cols);
        }
    }
    return 0;  // 返回计算成功
}

// 静态内联函数：计算闵可夫斯基距离
static inline int
cdist_minkowski(const double *XA, const double *XB, double *dm,
                const npy_intp num_rowsA, const npy_intp num_rowsB,
                const npy_intp num_cols, const double p)
{
    npy_intp i, j;

    // 根据不同的 p 值选择相应的距离计算方法
    if (p == 1.0) {
        return cdist_city_block_double(XA, XB, dm, num_rowsA, num_rowsB, num_cols);
    }
    if (p == 2.0) {
        return cdist_euclidean_double(XA, XB, dm, num_rowsA, num_rowsB, num_cols);
    }
    if (isinf(p)) {
        return cdist_chebyshev_double(XA, XB, dm, num_rowsA, num_rowsB, num_cols);
    }

    // 计算每对向量之间的闵可夫斯基距离
    for (i = 0; i < num_rowsA; ++i) {
        const double *u = XA + (num_cols * i);
        for (j = 0; j < num_rowsB; ++j, ++dm) {
            const double *v = XB + (num_cols * j);
            *dm = minkowski_distance(u, v, num_cols, p);
        }
    }
    return 0;  // 返回计算成功
}

// 静态内联函数：计算加权闵可夫斯基距离
static inline int
cdist_weighted_minkowski(const double *XA, const double *XB, double *dm,
                         const npy_intp num_rowsA, const npy_intp num_rowsB,
                         const npy_intp num_cols, const double p,
                         const double *w)
{
    npy_intp i, j;

    // 根据 p 的值选择相应的加权闵可夫斯基距离计算方法
    if (isinf(p)) {
        return cdist_weighted_chebyshev(XA, XB, dm, num_rowsA, num_rowsB, num_cols, w);
    }

    // 计算每对向量之间的加权闵可夫斯基距离
    for (i = 0; i < num_rowsA; ++i) {
        const double *u = XA + (num_cols * i);
        for (j = 0; j < num_rowsB; ++j, ++dm) {
            const double *v = XB + (num_cols * j);
            *dm = weighted_minkowski_distance(u, v, num_cols, p, w);
        }
    }
    return 0;  // 返回计算成功
}
# 计算两个双精度浮点型数组之间的汉明距离，并将结果写入到给定的数组中
cdist_hamming_double(const double *XA, const double *XB, double *dm,
                         const npy_intp num_rowsA, const npy_intp num_rowsB,
                         const npy_intp num_cols,
                         const double *w)
{
    npy_intp i, j;

    # 遍历第一个输入数组的行
    for (i = 0; i < num_rowsA; ++i) {
        # 指向当前行的起始位置
        const double *u = XA + (num_cols * i);
        # 遍历第二个输入数组的行
        for (j = 0; j < num_rowsB; ++j, ++dm) {
            # 指向当前行的起始位置
            const double *v = XB + (num_cols * j);
            # 调用函数计算两行数据的汉明距离，并将结果写入到dm中
            *dm = hamming_distance_double(u, v, num_cols, w);
        }
    }
    # 返回操作成功的标志
    return 0;
}

# 计算两个字符型数组之间的汉明距离，并将结果写入到给定的数组中
static inline int
cdist_hamming_char(const char *XA, const char *XB, double *dm,
                         const npy_intp num_rowsA, const npy_intp num_rowsB,
                         const npy_intp num_cols,
                         const double *w)
{
    npy_intp i, j;

    # 遍历第一个输入数组的行
    for (i = 0; i < num_rowsA; ++i) {
        # 指向当前行的起始位置
        const char *u = XA + (num_cols * i);
        # 遍历第二个输入数组的行
        for (j = 0; j < num_rowsB; ++j, ++dm) {
            # 指向当前行的起始位置
            const char *v = XB + (num_cols * j);
            # 调用函数计算两行数据的汉明距离，并将结果写入到dm中
            *dm = hamming_distance_char(u, v, num_cols, w);
        }
    }
    # 返回操作成功的标志
    return 0;
}
```