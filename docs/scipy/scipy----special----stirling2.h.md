# `D:\src\scipysrc\scipy\scipy\special\stirling2.h`

```
#ifndef STIRLING_H
#define STIRLING_H

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "special/binom.h"
#include "special/lambertw.h"

/*     Stirling numbers of the second kind
 *
 * SYNOPSIS: Stirling numbers of the second kind count the
 *  number of ways to make a partition of n distinct elements
 *  into k non-empty subsets.
 *
 * DESCRIPTION: n is the number of distinct elements of the set
 *  to be partitioned and k is the number of non-empty subsets.
 *  The values for n < 0 or k < 0 are interpreted as 0. If you
 *  ACCURACY: The method returns a double type.
 *
 * NOTE: this file was added by Lucas Roberts
 */

// Dynamic programming

// 计算第二类 Stirling 数，使用动态规划方法
double _stirling2_dp(double n, double k){
    // 特殊情况处理：n 和 k 同时为 0 或者 1
    if ((n == 0 && k == 0) || (n == 1 && k == 1)) {
        return 1.;
    }
    // 边界条件检查：k <= 0 或者 k > n 或者 n < 0 时返回 0
    if (k <= 0 || k > n || n < 0){
        return 0.;
    }
    // 确定动态规划数组的大小
    int arraySize = k <= n - k + 1 ? k : n - k + 1;
    // 分配动态规划数组内存
    double *curr = (double *) malloc(arraySize * sizeof(double));
    // 初始化动态规划数组
    for (int i = 0; i < arraySize; i++){
        curr[i] = 1.;
    }
    // 根据 k 和 n - k + 1 的大小关系选择不同的动态规划计算方式
    if (k <= n - k + 1) {
        for (int i = 1; i < n - k + 1; i++){
            for (int j = 1; j < k; j++){
                curr[j] = (j + 1) * curr[j] + curr[j - 1];
                // 检查数值是否溢出
                if (isinf(curr[j])){
                    free(curr);
                    return INFINITY; // 数值溢出
                }
            }
        }
    } else {
        for (int i = 1; i < k; i++){
            for (int j = 1; j < n - k + 1; j++){
                curr[j] = (i + 1) * curr[j - 1] + curr[j];
                // 检查数值是否溢出
                if (isinf(curr[j])){
                    free(curr);
                    return INFINITY; // 数值溢出
                }
            }
        }
    }
    // 保存计算结果并释放内存
    double output = curr[arraySize - 1];
    free(curr);
    return output;
}

// second order Temme approximation

#endif  // STIRLING_H
/*
 * 计算第二类 Stirling 数的不精确版本的函数。
 * 当 n 小于或等于 50 时，调用 _stirling2_dp 函数进行精确计算；
 * 否则，调用 _stirling2_temme 函数进行近似计算。
 */
double _stirling2_inexact(double n, double k) {
    // 如果 n 小于等于 50，则调用 _stirling2_dp 函数进行精确计算
    if (n <= 50) {
        return _stirling2_dp(n, k);
    } else {
        // 否则，调用 _stirling2_temme 函数进行近似计算
        return _stirling2_temme(n, k);
    }
}
```