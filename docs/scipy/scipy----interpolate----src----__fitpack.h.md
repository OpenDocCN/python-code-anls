# `D:\src\scipysrc\scipy\scipy\interpolate\src\__fitpack.h`

```
/*
 * B-spline evaluation routine.
 */

static inline void
_deBoor_D(const double *t, double x, int k, int ell, int m, double *result) {
    /*
     * On completion the result array stores
     * the k+1 non-zero values of beta^(m)_i,k(x):  for i=ell, ell-1, ell-2, ell-k.
     * Where t[ell] <= x < t[ell+1].
     */
    // 初始化中间结果数组指针
    double *hh = result + k + 1;
    // 初始化当前结果数组指针
    double *h = result;
    // 临时变量，存储右侧节点和左侧节点的值
    double xb, xa, w;
    // 循环中的索引变量，节点计数
    int ind, j, n;

    /*
     * Perform k-m "standard" deBoor iterations
     * so that h contains the k+1 non-zero values of beta_{ell,k-m}(x)
     * needed to calculate the remaining derivatives.
     */
    // 初始化基础情况，设置 beta_{ell,k}(x) 的初始值
    result[0] = 1.0;
    // 进行 k-m 次迭代，计算 beta_{ell,k-m}(x) 的值
    for (j = 1; j <= k - m; j++) {
        // 复制当前 h 数组到 hh 数组
        memcpy(hh, h, j*sizeof(double));
        // 将 h[0] 置零
        h[0] = 0.0;
        // 对每个节点 n 计算相应的 beta 值
        for (n = 1; n <= j; n++) {
            ind = ell + n;
            xb = t[ind];
            xa = t[ind - j];
            // 处理相同节点的情况
            if (xb == xa) {
                h[n] = 0.0;
                continue;
            }
            // 计算 beta 值
            w = hh[n - 1] / (xb - xa);
            h[n - 1] += w * (xb - x);
            h[n] = w * (x - xa);
        }
    }

    /*
     * Now do m "derivative" recursions
     * to convert the values of beta into the mth derivative
     */
    // 进行 m 次 "导数" 递归，计算 m 阶导数的值
    for (j = k - m + 1; j <= k; j++) {
        // 复制当前 h 数组到 hh 数组
        memcpy(hh, h, j*sizeof(double));
        // 将 h[0] 置零
        h[0] = 0.0;
        // 对每个节点 n 计算相应的 m 阶导数值
        for (n = 1; n <= j; n++) {
            ind = ell + n;
            xb = t[ind];
            xa = t[ind - j];
            // 处理相同节点的情况
            if (xb == xa) {
                h[m] = 0.0;
                continue;
            }
            // 计算 m 阶导数值
            w = j * hh[n - 1] / (xb - xa);
            h[n - 1] -= w;
            h[n] = w;
        }
    }
}
```