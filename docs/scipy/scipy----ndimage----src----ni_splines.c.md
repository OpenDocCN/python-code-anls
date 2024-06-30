# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_splines.c`

```
#include "ni_support.h"
#include "ni_splines.h"
#include <math.h>


int
get_spline_interpolation_weights(double x, int order, double *weights)
{
    int i;
    double y, z, t;

    /* Convert x to the delta to the middle knot. */
    x -= floor(order & 1 ? x : x + 0.5);
    y = x;
    z = 1.0 - x;

    /* 所有插值权重总和为1.0，因此最后一个权重为1.0 */
    weights[order] = 1.0;
    for (i = 0; i < order; ++i) {
        weights[order] -= weights[i];
    }

    return 0;
}


int
get_filter_poles(int order, int *npoles, double *poles)
{
    *npoles = order / 2;
    /*
     * 如果触发此断言，则表示有人在此处添加了更多的阶数，
     * 但没有与 MAX_SPLIINE_FILTER_POLES 保持同步。
     */
    assert(*npoles <= MAX_SPLINE_FILTER_POLES);

    switch (order) {
        case 2:
            /* sqrt(8.0) - 3.0 */
            poles[0] = -0.171572875253809902396622551580603843;
            break;
        case 3:
            /* sqrt(3.0) - 2.0 */
            poles[0] = -0.267949192431122706472553658494127633;
            break;
        case 4:
            /* sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0 */
            poles[0] = -0.361341225900220177092212841325675255;
            /* sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0 */
            poles[1] = -0.013725429297339121360331226939128204;
            break;
        case 5:
            /* sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5 */
            poles[0] = -0.430575347099973791851434783493520110;
            /* sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5 */
            poles[1] = -0.043096288203264653822712376822550182;
            break;
        default:
            return 1; /* 不支持的阶数。 */
    };

    return 0;
}


typedef void (init_fn)(npy_double*, const npy_intp, const double);


static void
_init_causal_mirror(double *c, const npy_intp n, const double z)
{
    npy_intp i;
    double z_i = z;
    const double z_n_1 = pow(z, n - 1);

    c[0] = c[0] + z_n_1 * c[n - 1];
    for (i = 1; i < n - 1; ++i) {
        c[0] += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] /= 1 - z_n_1 * z_n_1;
}


static void
_init_anticausal_mirror(double *c, const npy_intp n, const double z)
{
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1);
}


static void
_init_causal_wrap(double *c, const npy_intp n, const double z)
{
    npy_intp i;
    double z_i = z;

    for (i = 1; i < n; ++i) {
        c[0] += z_i * c[n - i];
        z_i *= z;
    }
    c[0] /= 1 - z_i; /* z_i = pow(z, n) */
}


static void
_init_anticausal_wrap(double *c, const npy_intp n, const double z)
{
    npy_intp i;
    double z_i = z;

    for (i = 0; i < n - 1; ++i) {
        c[n - 1] += z_i * c[i];
        z_i *= z;
    }
    c[n - 1] *= z / (z_i - 1); /* z_i = pow(z, n) */
}


static void
_init_causal_reflect(double *c, const npy_intp n, const double z)
{
    npy_intp i;
    double z_i = z;
    const double z_n = pow(z, n);
    const double c0 = c[0];

    /* 将 c[0] 初始化为反射边界条件下的值 */
    c[0] = c0;
    for (i = 1; i < n; ++i) {
        c[0] += z_i * c[i];
        z_i *= z;
    }
    c[0] /= 1 - z_n; /* z_n = pow(z, n) */
}
    # 计算生成的序列中的第一个系数 c[0]，基于最后一个元素 c[n-1] 和 z_n 的乘积
    c[0] = c[0] + z_n * c[n - 1];

    # 循环计算生成序列中其他系数 c[1] 到 c[n-1]
    for (i = 1; i < n; ++i) {
        # 计算 c[0] 的累加值，加上当前 c[i] 与对称位置 c[n-1-i] 的乘积，乘以 z_i
        c[0] += z_i * (c[i] + z_n * c[n - 1 - i]);
        # 更新 z_i 的值，乘以 z
        z_i *= z;
    }

    # 最后计算 c[0] 的最终值，乘以 z/(1 - z_n * z_n)
    c[0] *= z / (1 - z_n * z_n);

    # 最后加上常数项 c0
    c[0] += c0;
/*
 * 初始化反向反射边界条件下的第一个系数。
 * 该函数对系数数组执行两次滤波，一次正向，一次反向。
 * 详细讨论方法参见：
 * Unser, Michael, Akram Aldroubi, and Murray Eden. "Fast B-spline
 * transforms for continuous image representation and interpolation."
 * IEEE Transactions on pattern analysis and machine intelligence 13.3
 * (1991): 277-285.
 * 
 * 过程的关键部分是初始化每次滤波过程中的第一个系数，这取决于选择的边界条件以扩展输入图像。
 * NI_EXTEND_MIRROR 模式下的初始化方法见上述论文。
 * 对于 NI_EXTEND_WRAP 和 NI_EXTEND_REFLECT，未发表的方法从 Philippe Thévenaz 博士处获得。
 */
static void
_apply_filter(double *c, npy_intp n, double z, init_fn *causal_init,
              init_fn *anticausal_init)
{
    npy_intp i;

    // 初始化正向滤波的第一个系数
    causal_init(c, n, z);
    // 正向滤波
    for (i = 1; i < n; ++i) {
        c[i] += z * c[i - 1];
    }
    // 初始化反向滤波的第一个系数
    anticausal_init(c, n, z);
    // 反向滤波
    for (i = n - 2; i >= 0; --i) {
        c[i] = z * (c[i + 1] - c[i]);
    }
}


/*
 * 计算滤波器的增益，该滤波器由给定的极点组成。
 * 增益的计算使用了所有极点的乘积。
 */
static void
_apply_filter_gain(double *c, npy_intp n, const double *zs, int nz)
{
    double gain = 1.0;

    // 计算增益，涉及所有极点的乘积
    while (nz--) {
        const double z = *zs++;
        gain *= (1.0 - z) * (1.0 - 1.0 / z);
    }

    // 将每个系数乘以计算出的增益
    while (n--) {
        *c++ *= gain;
    }
}


/*
 * 应用滤波器到系数数组上，使用给定的极点和边界扩展模式。
 * 根据边界扩展模式选择合适的初始化函数。
 */
void
apply_filter(double *coefficients, const npy_intp len, const double *poles,
             int npoles, NI_ExtendMode mode)
{
    init_fn *causal = NULL;
    init_fn *anticausal = NULL;

    // 注意：此 switch 语句应与 NI_GeometricTransform 中的 spline_mode 变量设置相匹配
    switch(mode) {
        case NI_EXTEND_GRID_CONSTANT:
        case NI_EXTEND_CONSTANT:
        case NI_EXTEND_MIRROR:
        case NI_EXTEND_WRAP:
            causal = &_init_causal_mirror;
            anticausal = &_init_anticausal_mirror;
            break;
        case NI_EXTEND_GRID_WRAP:
            causal = &_init_causal_wrap;
            anticausal = &_init_anticausal_wrap;
            break;
        case NI_EXTEND_NEAREST:
        case NI_EXTEND_REFLECT:
            causal = &_init_causal_reflect;
            anticausal = &_init_anticausal_reflect;
            break;
        default:
            assert(0); /* 我们不应该到达这里。 */
    }

    // 应用滤波器的增益到系数数组上
    _apply_filter_gain(coefficients, len, poles, npoles);

    // 对每个极点应用滤波器
    while (npoles--) {
        _apply_filter(coefficients, len, *poles++, causal, anticausal);
    }
}
```