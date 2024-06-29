# `.\numpy\numpy\_core\src\common\mem_overlap.c`

```
/*
  Solving memory overlap integer programs and bounded Diophantine equations with
  positive coefficients.

  Asking whether two strided arrays `a` and `b` overlap is equivalent to
  asking whether there is a solution to the following problem::

      sum(stride_a[i] * x_a[i] for i in range(ndim_a))
      -
      sum(stride_b[i] * x_b[i] for i in range(ndim_b))
      ==
      base_b - base_a

      0 <= x_a[i] < shape_a[i]
      0 <= x_b[i] < shape_b[i]

  for some integer x_a, x_b.  Itemsize needs to be considered as an additional
  dimension with stride 1 and size itemsize.

  Negative strides can be changed to positive (and vice versa) by changing
  variables x[i] -> shape[i] - 1 - x[i], and zero strides can be dropped, so
  that the problem can be recast into a bounded Diophantine equation with
  positive coefficients::

     sum(a[i] * x[i] for i in range(n)) == b

     a[i] > 0

     0 <= x[i] <= ub[i]

  This problem is NP-hard --- runtime of algorithms grows exponentially with
  increasing ndim.


  *Algorithm description*

  A straightforward algorithm that excludes infeasible solutions using GCD-based
  pruning is outlined in Ref. [1]. It is implemented below. A number of other
  algorithms exist in the literature; however, this one seems to have
  performance satisfactory for the present purpose.

  The idea is that an equation::

      a_1 x_1 + a_2 x_2 + ... + a_n x_n = b
      0 <= x_i <= ub_i, i = 1...n

  implies::

      a_2' x_2' + a_3 x_3 + ... + a_n x_n = b

      0 <= x_i <= ub_i, i = 2...n

      0 <= x_1' <= c_1 ub_1 + c_2 ub_2

  with a_2' = gcd(a_1, a_2) and x_2' = c_1 x_1 + c_2 x_2 with c_1 = (a_1/a_1'),
  and c_2 = (a_2/a_1').  This procedure can be repeated to obtain::

      a_{n-1}' x_{n-1}' + a_n x_n = b

      0 <= x_{n-1}' <= ub_{n-1}'

      0 <= x_n <= ub_n

  Now, one can enumerate all candidate solutions for x_n.  For each, one can use
  the previous-level equation to enumerate potential solutions for x_{n-1}, with
  transformed right-hand side b -> b - a_n x_n.  And so forth, until after n-1
  nested for loops we either arrive at a candidate solution for x_1 (in which
  case we have found one solution to the problem), or find that the equations do
  not allow any solutions either for x_1 or one of the intermediate x_i (in
  which case we have proved there is no solution for the upper-level candidates
  chosen). If no solution is found for any candidate x_n, we have proved the
  problem is infeasible --- which for the memory overlap problem means there is
  no overlap.


  *Performance*

  Some common ndarray cases are easy for the algorithm:

  - Two arrays whose memory ranges do not overlap.

    These will be excluded by the bounds on x_n, with max_work=1. We also add
    this check as a fast path, to avoid computing GCDs needlessly, as this can
    take some time.

  - Arrays produced by continuous slicing of a continuous parent array (no
*/
  *Integer overflows*

  # 算法使用固定宽度整数编写，如果检测到整数溢出，可能会以失败结束（实现中捕获所有情况）。潜在的失败模式：

  - Array extent sum(stride*(shape-1)) is too large (for int64).
  # 数组的范围和 sum(stride*(shape-1)) 太大（对于 int64）。

  - Minimal solutions to a_i x_i + a_j x_j == b are too large,
  # 最小解 a_i x_i + a_j x_j == b 太大，
    in some of the intermediate equations.
    # 这段文字描述了算法中某些中间方程的使用情况。

    We do this part of the computation in 128-bit integers.
    # 在这部分计算中，我们使用128位整数。

  In general, overflows are expected only if array size is close to
  NPY_INT64_MAX, requiring ~exabyte size arrays, which is usually not possible.
  # 通常情况下，只有在数组大小接近NPY_INT64_MAX时才会发生溢出，这需要大约百亿亿字节大小的数组，这通常是不可能的。

  References
  ----------
  .. [1] P. Ramachandran, ''Use of Extended Euclidean Algorithm in Solving
         a System of Linear Diophantine Equations with Bounded Variables''.
         Algorithmic Number Theory, Lecture Notes in Computer Science **4076**,
         182-192 (2006). doi:10.1007/11792086_14
  # 参考文献1：Ramachandran在《算法数论》中介绍了扩展欧几里得算法在解有界变量线性丢番图方程组中的应用。

  .. [2] Cornuejols, Urbaniak, Weismantel, and Wolsey,
         ''Decomposition of integer programs and of generating sets.'',
         Lecture Notes in Computer Science 1284, 92-103 (1997).
  # 参考文献2：Cornuejols等人在《计算机科学讲义》中讨论了整数程序和生成集的分解。

  .. [3] K. Aardal, A.K. Lenstra,
         ''Hard equality constrained integer knapsacks'',
         Lecture Notes in Computer Science 2337, 350-366 (2002).
  # 参考文献3：Aardal和Lenstra在《计算机科学讲义》中探讨了硬等式约束整数背包问题。
/*
  Copyright (c) 2015 Pauli Virtanen
  All rights reserved.
  Licensed under 3-clause BSD license, see LICENSE.txt.
*/
/* 设置 NumPy 的 API 版本，禁用过时的 API */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* 清理 Py_ssize_t 宏定义，以支持最新的 Python 对象 API */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* 引入 NumPy 的头文件 */
#include "numpy/ndarrayobject.h"

/* 引入自定义的内存重叠检测头文件 */
#include "mem_overlap.h"

/* 引入处理扩展整数 128 位的头文件 */
#include "npy_extint128.h"

/* 引入标准库头文件 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

/* 定义取最大值的宏函数 */
#define MAX(a, b) (((a) >= (b)) ? (a) : (b))

/* 定义取最小值的宏函数 */
#define MIN(a, b) (((a) <= (b)) ? (a) : (b))

/**
 * 欧几里得算法求最大公约数 (GCD) 的函数
 *
 * 解决方程 gamma*a1 + epsilon*a2 == gcd(a1, a2)
 * 其中 |gamma| < |a2|/gcd, |epsilon| < |a1|/gcd.
 */
static void
euclid(npy_int64 a1, npy_int64 a2, npy_int64 *a_gcd, npy_int64 *gamma, npy_int64 *epsilon)
{
    npy_int64 gamma1, gamma2, epsilon1, epsilon2, r;

    assert(a1 > 0);
    assert(a2 > 0);

    gamma1 = 1;
    gamma2 = 0;
    epsilon1 = 0;
    epsilon2 = 1;

    /* 在迭代过程中，a1 和 a2 保持在 |a1|, |a2| 的界限内，因此没有整数溢出 */
    while (1) {
        if (a2 > 0) {
            r = a1/a2;
            a1 -= r*a2;
            gamma1 -= r*gamma2;
            epsilon1 -= r*epsilon2;
        }
        else {
            *a_gcd = a1;
            *gamma = gamma1;
            *epsilon = epsilon1;
            break;
        }

        if (a1 > 0) {
            r = a2/a1;
            a2 -= r*a1;
            gamma2 -= r*gamma1;
            epsilon2 -= r*epsilon1;
        }
        else {
            *a_gcd = a2;
            *gamma = gamma2;
            *epsilon = epsilon2;
            break;
        }
    }
}

/**
 * 预计算最大公约数 (GCD) 和边界转换的函数
 */
static int
diophantine_precompute(unsigned int n,
                       diophantine_term_t *E,
                       diophantine_term_t *Ep,
                       npy_int64 *Gamma, npy_int64 *Epsilon)
{
    npy_int64 a_gcd, gamma, epsilon, c1, c2;
    unsigned int j;
    char overflow = 0;

    assert(n >= 2);

    /* 使用欧几里得算法计算第一和第二项的最大公约数和相应的 gamma, epsilon */
    euclid(E[0].a, E[1].a, &a_gcd, &gamma, &epsilon);
    Ep[0].a = a_gcd;
    Gamma[0] = gamma;
    Epsilon[0] = epsilon;

    if (n > 2) {
        c1 = E[0].a / a_gcd;
        c2 = E[1].a / a_gcd;

        /* 计算 Ep[0].ub = E[0].ub * c1 + E[1].ub * c2 的安全加法 */
        Ep[0].ub = safe_add(safe_mul(E[0].ub, c1, &overflow),
                            safe_mul(E[1].ub, c2, &overflow), &overflow);
        if (overflow) {
            return 1; /* 溢出情况，返回错误 */
        }
    }

    /* 循环计算剩余项的最大公约数和相应的 gamma, epsilon */
    for (j = 2; j < n; ++j) {
        euclid(Ep[j-2].a, E[j].a, &a_gcd, &gamma, &epsilon);
        Ep[j-1].a = a_gcd;
        Gamma[j-1] = gamma;
        Epsilon[j-1] = epsilon;

        if (j < n - 1) {
            c1 = Ep[j-2].a / a_gcd;
            c2 = E[j].a / a_gcd;

            /* 计算 Ep[j-1].ub = c1 * Ep[j-2].ub + c2 * E[j].ub 的安全加法 */
            Ep[j-1].ub = safe_add(safe_mul(c1, Ep[j-2].ub, &overflow),
                                  safe_mul(c2, E[j].ub, &overflow), &overflow);

            if (overflow) {
                return 1; /* 溢出情况，返回错误 */
            }
        }
    }

    return 0; /* 所有项计算完成，无溢出情况，返回成功 */
}
/**
 * Depth-first bounded Euclid search
 */
static mem_overlap_t
diophantine_dfs(unsigned int n,
                unsigned int v,
                diophantine_term_t *E,
                diophantine_term_t *Ep,
                npy_int64 *Gamma, npy_int64 *Epsilon,
                npy_int64 b,
                Py_ssize_t max_work,
                int require_ub_nontrivial,
                npy_int64 *x,
                Py_ssize_t *count)
{
    npy_int64 a_gcd, gamma, epsilon, a1, u1, a2, u2, c, r, c1, c2, t, t_l, t_u, b2, x1, x2;
    npy_extint128_t x10, x20, t_l1, t_l2, t_u1, t_u2;
    mem_overlap_t res;
    char overflow = 0;

    if (max_work >= 0 && *count >= max_work) {
        return MEM_OVERLAP_TOO_HARD;
    }

    /* Fetch precomputed values for the reduced problem */
    // 根据问题的减少，获取预先计算的值
    if (v == 1) {
        a1 = E[0].a;
        u1 = E[0].ub;
    }
    else {
        a1 = Ep[v-2].a;
        u1 = Ep[v-2].ub;
    }

    a2 = E[v].a;
    u2 = E[v].ub;

    a_gcd = Ep[v-1].a;
    gamma = Gamma[v-1];
    epsilon = Epsilon[v-1];

    /* Generate set of allowed solutions */
    // 生成允许的解集合
    c = b / a_gcd;
    r = b % a_gcd;
    if (r != 0) {
        ++*count;
        return MEM_OVERLAP_NO;
    }

    c1 = a2 / a_gcd;
    c2 = a1 / a_gcd;

    /*
      The set to enumerate is:
      x1 = gamma*c + c1*t
      x2 = epsilon*c - c2*t
      t integer
      0 <= x1 <= u1
      0 <= x2 <= u2
      and we have c, c1, c2 >= 0
     */
    // 枚举的集合为：
    // x1 = gamma*c + c1*t
    // x2 = epsilon*c - c2*t
    // 其中 t 是整数
    // 0 <= x1 <= u1
    // 0 <= x2 <= u2
    // 同时 c, c1, c2 >= 0

    x10 = mul_64_64(gamma, c);
    x20 = mul_64_64(epsilon, c);

    t_l1 = ceildiv_128_64(neg_128(x10), c1);
    t_l2 = ceildiv_128_64(sub_128(x20, to_128(u2), &overflow), c2);

    t_u1 = floordiv_128_64(sub_128(to_128(u1), x10, &overflow), c1);
    t_u2 = floordiv_128_64(x20, c2);

    if (overflow) {
        return MEM_OVERLAP_OVERFLOW;
    }

    if (gt_128(t_l2, t_l1)) {
        t_l1 = t_l2;
    }

    if (gt_128(t_u1, t_u2)) {
        t_u1 = t_u2;
    }

    if (gt_128(t_l1, t_u1)) {
        ++*count;
        return MEM_OVERLAP_NO;
    }

    t_l = to_64(t_l1, &overflow);
    t_u = to_64(t_u1, &overflow);

    x10 = add_128(x10, mul_64_64(c1, t_l), &overflow);
    x20 = sub_128(x20, mul_64_64(c2, t_l), &overflow);

    t_u = safe_sub(t_u, t_l, &overflow);
    t_l = 0;
    x1 = to_64(x10, &overflow);
    x2 = to_64(x20, &overflow);

    if (overflow) {
        return MEM_OVERLAP_OVERFLOW;
    }

    /* The bounds t_l, t_u ensure the x computed below do not overflow */
    // t_l, t_u 的边界确保下面计算的 x 不会溢出
    if (v == 1) {
        /* 如果当前深度 v 等于 1，表示到达递归的基本情况 */
        /* Base case */
        if (t_u >= t_l) {
            /* 如果上界 t_u 大于等于下界 t_l */
            /* Calculate x[0] and x[1] based on linear equations */
            x[0] = x1 + c1*t_l;
            x[1] = x2 - c2*t_l;
            /* 如果需要检查上界是否为非平凡解 */
            if (require_ub_nontrivial) {
                unsigned int j;
                int is_ub_trivial;

                is_ub_trivial = 1;
                /* 检查每个变量是否满足上界的一半 */
                for (j = 0; j < n; ++j) {
                    if (x[j] != E[j].ub/2) {
                        is_ub_trivial = 0;
                        break;
                    }
                }

                /* 如果上界为平凡解，则忽略 */
                if (is_ub_trivial) {
                    ++*count;
                    return MEM_OVERLAP_NO;
                }
            }
            /* 返回存在内存重叠的标志 */
            return MEM_OVERLAP_YES;
        }
        /* 增加计数并返回内存未重叠的标志 */
        ++*count;
        return MEM_OVERLAP_NO;
    }
    else {
        /* 如果当前深度 v 大于 1，递归到所有可能的候选解 */
        /* Recurse to all candidates */
        for (t = t_l; t <= t_u; ++t) {
            /* 计算当前变量 x[v] 的值 */
            x[v] = x2 - c2*t;

            /* 计算剩余的线性方程式的右侧值 b2 */
            b2 = safe_sub(b, safe_mul(a2, x[v], &overflow), &overflow);
            /* 如果计算溢出 */
            if (overflow) {
                return MEM_OVERLAP_OVERFLOW;
            }

            /* 递归调用解决剩余变量的线性方程组 */
            res = diophantine_dfs(n, v-1, E, Ep, Gamma, Epsilon,
                                  b2, max_work, require_ub_nontrivial,
                                  x, count);
            /* 如果找到内存重叠的解，则返回结果 */
            if (res != MEM_OVERLAP_NO) {
                return res;
            }
        }
        /* 增加计数并返回内存未重叠的标志 */
        ++*count;
        return MEM_OVERLAP_NO;
    }
/**
 * 解决有界丢番图方程
 *
 * 考虑的问题是：
 *     A[0] x[0] + A[1] x[1] + ... + A[n-1] x[n-1] == b
 *     0 <= x[i] <= U[i]
 *     A[i] > 0
 *
 * 使用深度优先的欧几里德算法解决，如[1]中所述。
 *
 * 如果 require_ub_nontrivial!=0，则寻找满足以下条件的解：
 * 当 b = A[0]*(U[0]/2) + ... + A[n-1]*(U[n-1]/2)，但忽略 x[i] = U[i]/2 的平凡解。
 * 所有的 U[i] 必须能被 2 整除。在这种情况下，给定的 b 值会被忽略。
 */
NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_diophantine(unsigned int n, diophantine_term_t *E, npy_int64 b,
                  Py_ssize_t max_work, int require_ub_nontrivial, npy_int64 *x)
{
    mem_overlap_t res;
    unsigned int j;

    // 检查每个项的系数和上界
    for (j = 0; j < n; ++j) {
        if (E[j].a <= 0) {
            return MEM_OVERLAP_ERROR; // 如果系数小于等于0，则返回错误状态
        }
        else if (E[j].ub < 0) {
            return MEM_OVERLAP_NO; // 如果上界小于0，则返回无重叠状态
        }
    }

    // 如果需要非平凡的上界解
    if (require_ub_nontrivial) {
        npy_int64 ub_sum = 0;
        char overflow = 0;
        // 检查所有项的上界是否能被2整除，并计算修正后的 b 值
        for (j = 0; j < n; ++j) {
            if (E[j].ub % 2 != 0) {
                return MEM_OVERLAP_ERROR; // 如果某个上界不能被2整除，则返回错误状态
            }
            ub_sum = safe_add(ub_sum,
                              safe_mul(E[j].a, E[j].ub/2, &overflow),
                              &overflow); // 计算修正后的 b 值
        }
        if (overflow) {
            return MEM_OVERLAP_ERROR; // 如果计算过程中发生溢出，则返回错误状态
        }
        b = ub_sum; // 更新 b 的值为修正后的值
    }

    // 如果 b 小于0，则返回无重叠状态
    if (b < 0) {
        return MEM_OVERLAP_NO;
    }

    // 对于没有变量的情况
    if (n == 0) {
        if (require_ub_nontrivial) {
            /* 对于0个变量的情况只有平凡解 */
            return MEM_OVERLAP_NO;
        }
        if (b == 0) {
            return MEM_OVERLAP_YES; // 如果 b 为0，则返回重叠状态
        }
        return MEM_OVERLAP_NO; // 否则返回无重叠状态
    }
    // 对于只有一个变量的情况
    else if (n == 1) {
        if (require_ub_nontrivial) {
            /* 对于1个变量的情况只有平凡解 */
            return MEM_OVERLAP_NO;
        }
        if (b % E[0].a == 0) {
            x[0] = b / E[0].a;
            if (x[0] >= 0 && x[0] <= E[0].ub) {
                return MEM_OVERLAP_YES; // 如果计算出的解在合理范围内，则返回重叠状态
            }
        }
        return MEM_OVERLAP_NO; // 否则返回无重叠状态
    }
    // 对于多于一个变量的情况
    else {
        Py_ssize_t count = 0;
        diophantine_term_t *Ep = NULL;
        npy_int64 *Epsilon = NULL, *Gamma = NULL;

        // 分配内存并检查分配情况
        Ep = malloc(n * sizeof(diophantine_term_t));
        Epsilon = malloc(n * sizeof(npy_int64));
        Gamma = malloc(n * sizeof(npy_int64));
        if (Ep == NULL || Epsilon == NULL || Gamma == NULL) {
            res = MEM_OVERLAP_ERROR; // 如果内存分配失败，则返回错误状态
        }
        else if (diophantine_precompute(n, E, Ep, Gamma, Epsilon)) {
            res = MEM_OVERLAP_OVERFLOW; // 如果预计算过程中发生溢出，则返回溢出状态
        }
        else {
            // 进行深度优先搜索解方程
            res = diophantine_dfs(n, n-1, E, Ep, Gamma, Epsilon, b, max_work,
                                  require_ub_nontrivial, x, &count);
        }
        free(Ep); // 释放动态分配的内存
        free(Gamma);
        free(Epsilon);
        return res; // 返回计算结果状态
    }
}


static int
diophantine_sort_A(const void *xp, const void *yp)
{
    npy_int64 xa = ((diophantine_term_t*)xp)->a;
    // 比较函数，按照结构体中的 a 成员排序
    // 用于在排序算法中对结构体数组进行排序
    # 从指针 yp 强制转换为 diophantine_term_t 类型的指针，并取出其成员变量 a 的值
    npy_int64 ya = ((diophantine_term_t*)yp)->a;
    
    # 如果 xa 小于 ya，则返回 1，表示 xa 比 ya 小
    if (xa < ya) {
        return 1;
    }
    # 如果 ya 小于 xa，则返回 -1，表示 ya 比 xa 小
    else if (ya < xa) {
        return -1;
    }
    # 否则，返回 0，表示 xa 和 ya 相等
    else {
        return 0;
    }
/**
 * Simplify Diophantine decision problem.
 *
 * Combine identical coefficients, remove unnecessary variables, and trim
 * bounds.
 *
 * The feasible/infeasible decision result is retained.
 *
 * Returns: 0 (success), -1 (integer overflow).
 */
NPY_VISIBILITY_HIDDEN int
diophantine_simplify(unsigned int *n, diophantine_term_t *E, npy_int64 b)
{
    unsigned int i, j, m;
    char overflow = 0;

    /* Skip obviously infeasible cases */
    for (j = 0; j < *n; ++j) {
        if (E[j].ub < 0) {
            return 0;
        }
    }

    if (b < 0) {
        return 0;
    }

    /* Sort vs. coefficients */
    qsort(E, *n, sizeof(diophantine_term_t), diophantine_sort_A);

    /* Combine identical coefficients */
    m = *n;
    i = 0;
    for (j = 1; j < m; ++j) {
        if (E[i].a == E[j].a) {
            E[i].ub = safe_add(E[i].ub, E[j].ub, &overflow);
            --*n;
        }
        else {
            ++i;
            if (i != j) {
                E[i] = E[j];
            }
        }
    }

    /* Trim bounds and remove unnecessary variables */
    m = *n;
    i = 0;
    for (j = 0; j < m; ++j) {
        E[j].ub = MIN(E[j].ub, b / E[j].a);
        if (E[j].ub == 0) {
            /* If the problem is feasible at all, x[i]=0 */
            --*n;
        }
        else {
            if (i != j) {
                E[i] = E[j];
            }
            ++i;
        }
    }

    if (overflow) {
        return -1;
    }
    else {
        return 0;
    }
}


/**
 * Gets a half-open range [start, end) of offsets from the data pointer
 */
NPY_VISIBILITY_HIDDEN void
offset_bounds_from_strides(const int itemsize, const int nd,
                           const npy_intp *dims, const npy_intp *strides,
                           npy_intp *lower_offset, npy_intp *upper_offset)
{
    npy_intp max_axis_offset;
    npy_intp lower = 0;
    npy_intp upper = 0;
    int i;

    for (i = 0; i < nd; i++) {
        if (dims[i] == 0) {
            /* If the array size is zero, return an empty range */
            *lower_offset = 0;
            *upper_offset = 0;
            return;
        }
        /* Expand either upwards or downwards depending on stride */
        max_axis_offset = strides[i] * (dims[i] - 1);
        if (max_axis_offset > 0) {
            upper += max_axis_offset;
        }
        else {
            lower += max_axis_offset;
        }
    }
    /* Return a half-open range */
    upper += itemsize;
    *lower_offset = lower;
    *upper_offset = upper;
}


/**
 * Gets a half-open range [start, end) which contains the array data
 */
static void
get_array_memory_extents(PyArrayObject *arr,
                         npy_uintp *out_start, npy_uintp *out_end,
                         npy_uintp *num_bytes)
{
    npy_intp low, upper;
    int j;
    offset_bounds_from_strides(PyArray_ITEMSIZE(arr), PyArray_NDIM(arr),
                               PyArray_DIMS(arr), PyArray_STRIDES(arr),
                               &low, &upper);
    # 计算指向数组数据开始位置的指针
    *out_start = (npy_uintp)PyArray_DATA(arr) + (npy_uintp)low;
    
    # 计算指向数组数据结束位置的指针
    *out_end = (npy_uintp)PyArray_DATA(arr) + (npy_uintp)upper;
    
    # 计算数组每个元素的字节大小
    *num_bytes = PyArray_ITEMSIZE(arr);
    
    # 根据数组的维度信息，计算数组总共占用的字节数
    for (j = 0; j < PyArray_NDIM(arr); ++j) {
        *num_bytes *= PyArray_DIM(arr, j);
    }
/**
 * 将数组的步长转换为项集合。
 *
 * Args:
 *     arr: NumPy 数组对象指针
 *     terms: 存储转换后项的数组
 *     nterms: 项的数量，通过指针传递
 *     skip_empty: 是否跳过空数组维度的标志
 *
 * Returns:
 *     0 表示成功，1 表示整数溢出
 *
 * 该函数根据数组的维度和步长信息，将每个维度的步长转换为项集合，存储在 terms 数组中。
 * 如果 skip_empty 标志被设置且某维度的尺寸为 1 或步长为 0，则跳过该维度的处理。
 * 对于步长为负数的情况，将其转换为正数处理，并检查是否存在整数溢出。
 */
static int
strides_to_terms(PyArrayObject *arr, diophantine_term_t *terms,
                 unsigned int *nterms, int skip_empty)
{
    int i;

    for (i = 0; i < PyArray_NDIM(arr); ++i) {
        if (skip_empty) {
            if (PyArray_DIM(arr, i) <= 1 || PyArray_STRIDE(arr, i) == 0) {
                continue;
            }
        }

        terms[*nterms].a = PyArray_STRIDE(arr, i);

        if (terms[*nterms].a < 0) {
            terms[*nterms].a = -terms[*nterms].a;
        }

        if (terms[*nterms].a < 0) {
            /* 整数溢出 */
            return 1;
        }

        terms[*nterms].ub = PyArray_DIM(arr, i) - 1;
        ++*nterms;
    }

    return 0;
}



/**
 * 判断两个数组是否共享内存。
 *
 * Returns:
 *     0 (不共享内存), 1 (共享内存), 或 < 0 (解决失败)
 *
 * Notes:
 *     解决失败可能是由于整数溢出或解决问题所需工作量超过 max_work 导致。
 *     该问题是 NP-难的，最坏情况下的运行时间与维度数量呈指数关系。
 *     max_work 控制处理的工作量，可以是精确的 (max_work == -1)，
 *     也可以仅仅是一个简单的内存范围检查 (max_work == 0)，或者设置一个上限
 *     max_work > 0 用于考虑的解决方案候选数量。
 *
 *     函数的主要目的是检查两个数组的内存是否重叠。
 */
NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_may_share_memory(PyArrayObject *a, PyArrayObject *b,
                       Py_ssize_t max_work)
{
    npy_int64 rhs;
    diophantine_term_t terms[2*NPY_MAXDIMS + 2];
    npy_uintp start1 = 0, end1 = 0, size1 = 0;
    npy_uintp start2 = 0, end2 = 0, size2 = 0;
    npy_uintp uintp_rhs;
    npy_int64 x[2*NPY_MAXDIMS + 2];
    unsigned int nterms;

    get_array_memory_extents(a, &start1, &end1, &size1);
    get_array_memory_extents(b, &start2, &end2, &size2);

    if (!(start1 < end2 && start2 < end1 && start1 < end1 && start2 < end2)) {
        /* 内存范围不重叠 */
        return MEM_OVERLAP_NO;
    }

    if (max_work == 0) {
        /* 需要的工作量太大，放弃 */
        return MEM_OVERLAP_TOO_HARD;
    }

    /* 将问题转换为具有正系数的丢番图方程形式。
       由 offset_bounds_from_strides 计算的边界对应于所有正步长。

       start1 + sum(abs(stride1)*x1)
       == start2 + sum(abs(stride2)*x2)
       == end1 - 1 - sum(abs(stride1)*x1')
       == end2 - 1 - sum(abs(stride2)*x2')

       <=>

       sum(abs(stride1)*x1) + sum(abs(stride2)*x2')
       == end2 - 1 - start1

       OR

       sum(abs(stride1)*x1') + sum(abs(stride2)*x2)
       == end1 - 1 - start2

       我们选择具有较小 RHS 的问题（由于上面的范围检查，它们都是非负的）。
    */

    uintp_rhs = MIN(end2 - 1 - start1, end1 - 1 - start2);
    if (uintp_rhs > NPY_MAX_INT64) {
        /* 整数溢出 */
        return MEM_OVERLAP_OVERFLOW;
    }
    rhs = (npy_int64)uintp_rhs;

    nterms = 0;
    # 如果数组 a 的步幅转换为对应的项失败，则返回内存重叠溢出错误码
    if (strides_to_terms(a, terms, &nterms, 1)) {
        return MEM_OVERLAP_OVERFLOW;
    }
    # 如果数组 b 的步幅转换为对应的项失败，则返回内存重叠溢出错误码
    if (strides_to_terms(b, terms, &nterms, 1)) {
        return MEM_OVERLAP_OVERFLOW;
    }
    # 如果数组 a 的元素字节大小大于 1
    if (PyArray_ITEMSIZE(a) > 1) {
        # 将项中的 a 设为 1
        terms[nterms].a = 1;
        # 将项中的 ub 设为数组 a 的元素字节大小减 1
        terms[nterms].ub = PyArray_ITEMSIZE(a) - 1;
        # 项的数量加一
        ++nterms;
    }
    # 如果数组 b 的元素字节大小大于 1
    if (PyArray_ITEMSIZE(b) > 1) {
        # 将项中的 a 设为 1
        terms[nterms].a = 1;
        # 将项中的 ub 设为数组 b 的元素字节大小减 1
        terms[nterms].ub = PyArray_ITEMSIZE(b) - 1;
        # 项的数量加一
        ++nterms;
    }

    """ 简化，如果可能 """
    # 简化二次方程组，如果失败则返回内存重叠溢出错误码
    if (diophantine_simplify(&nterms, terms, rhs)) {
        """ 整数溢出 """
        return MEM_OVERLAP_OVERFLOW;
    }

    """ 求解 """
    # 调用函数解二次方程组并返回结果
    return solve_diophantine(nterms, terms, rhs, max_work, 0, x);
/**
 * Determine whether an array has internal overlap.
 *
 * Returns: 0 (no overlap), 1 (overlap), or < 0 (failed to solve).
 *
 * max_work and reasons for solver failures are as in solve_may_share_memory.
 */
NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_may_have_internal_overlap(PyArrayObject *a, Py_ssize_t max_work)
{
    // 定义用于解决的二次方程项和解向量
    diophantine_term_t terms[NPY_MAXDIMS+1];
    npy_int64 x[NPY_MAXDIMS+1];
    unsigned int i, j, nterms;

    // 检查数组是否是连续的，是的话快速返回无重叠
    if (PyArray_ISCONTIGUOUS(a)) {
        /* Quick case */
        return MEM_OVERLAP_NO;
    }

    // 内存重叠问题是寻找两个不同的解决方案
    // 初始化二次方程的项
    nterms = 0;
    if (strides_to_terms(a, terms, &nterms, 0)) {
        // 如果转换 strides 到方程项时溢出，返回溢出错误
        return MEM_OVERLAP_OVERFLOW;
    }
    if (PyArray_ITEMSIZE(a) > 1) {
        // 如果数组元素大小大于1，添加额外的项来处理
        terms[nterms].a = 1;
        terms[nterms].ub = PyArray_ITEMSIZE(a) - 1;
        ++nterms;
    }

    // 清除零系数和空项
    i = 0;
    for (j = 0; j < nterms; ++j) {
        if (terms[j].ub == 0) {
            continue;
        }
        else if (terms[j].ub < 0) {
            // 如果上界小于0，表示无重叠
            return MEM_OVERLAP_NO;
        }
        else if (terms[j].a == 0) {
            // 如果系数为0，表示有重叠
            return MEM_OVERLAP_YES;
        }
        if (i != j) {
            terms[i] = terms[j];
        }
        ++i;
    }
    nterms = i;

    // 扩展上界以处理内部重叠问题
    for (j = 0; j < nterms; ++j) {
        terms[j].ub *= 2;
    }

    // 根据系数排序；不能调用 diophantine_simplify，因为它可能改变决策问题的不等式部分
    qsort(terms, nterms, sizeof(diophantine_term_t), diophantine_sort_A);

    // 解决二次方程
    return solve_diophantine(nterms, terms, -1, max_work, 1, x);
}
```