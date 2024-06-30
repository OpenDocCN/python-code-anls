# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dlacon2.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dlacon2.c
 * \brief Estimates the 1-norm
 *
 * <pre>
 * -- SuperLU routine (version 5.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * July 24, 2022
 * </pre>
 */
#include <math.h>
#include "slu_Cnames.h"

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   DLACON2 estimates the 1-norm of a square matrix A.   
 *   Reverse communication is used for evaluating matrix-vector products. 
 * 
 *   This is a thread safe version of CLACON, which uses the array ISAVE
 *   in place of a STATIC variables, as follows:
 *
 *     DLACON     DLACON2
 *      jump     isave[0]
 *      j        isave[1]
 *      iter     isave[2]
 *
 *
 *   Arguments   
 *   =========   
 *
 *   N      (input) INT
 *          The order of the matrix.  N >= 1.   
 *
 *   V      (workspace) DOUBLE PRECISION array, dimension (N)   
 *          On the final return, V = A*W,  where  EST = norm(V)/norm(W)   
 *          (W is not returned).   
 *
 *   X      (input/output) DOUBLE PRECISION array, dimension (N)   
 *          On an intermediate return, X should be overwritten by   
 *                A * X,   if KASE=1,   
 *                A' * X,  if KASE=2,
 *         and DLACON must be re-called with all the other parameters   
 *          unchanged.   
 *
 *   ISGN   (workspace) INT array, dimension (N)
 *
 *   EST    (output) DOUBLE PRECISION   
 *          An estimate (a lower bound) for norm(A).   
 *
 *   KASE   (input/output) INT
 *          On the initial call to DLACON, KASE should be 0.   
 *          On an intermediate return, KASE will be 1 or 2, indicating   
 *          whether X should be overwritten by A * X  or A' * X.   
 *          On the final return from DLACON, KASE will again be 0.   
 *
 *   isave  (input/output) int [3]
 *          ISAVE is INTEGER array, dimension (3)
 *          ISAVE is used to save variables between calls to DLACON2
 *
 *   Further Details   
 *   ===============   
 *
 *   Contributed by Nick Higham, University of Manchester.   
 *   Originally named CONEST, dated March 16, 1988.   
 *
 *   Reference: N.J. Higham, "FORTRAN codes for estimating the one-norm of 
 *   a real or complex matrix, with applications to condition estimation", 
 *   ACM Trans. Math. Soft., vol. 14, no. 4, pp. 381-396, December 1988.   
 *   ===================================================================== 
 * </pre>
 */

int
dlacon2_(int *n, double *v, double *x, int *isgn, double *est, int *kase, int isave[3])
{
    /* Table of constant values */
    int c__1 = 1; // 常量1，用于表示整数1

    double zero = 0.0; // 双精度浮点数0
    double one = 1.0; // 双精度浮点数1
    
    /* Local variables */
    // 局部变量声明

    // 返回值为整数类型，函数名为dlacon2_，接受以下参数
    // n: 矩阵的阶数，输入参数
    // v: 双精度浮点数数组，用作工作空间，维度为N，返回时V = A*W，其中EST = norm(V)/norm(W)
    // x: 双精度浮点数数组，维度为N，输入/输出参数，在中间返回时，如果KASE=1，则被A * X覆盖；如果KASE=2，则被A' * X覆盖
    // isgn: 整数数组，维度为N，工作空间
    // est: 双精度浮点数，输出参数，对矩阵A的一范数的估计（下界）
    // kase: 整数，输入/输出参数，初始调用时应为0；中间返回时，值为1或2，指示X应该被A * X或A' * X覆盖；最终返回时，值再次为0
    // isave: 整数数组，维度为3，输入/输出参数，用于在调用dlacon2之间保存变量
    
    // 函数DLACON2的目的是估计方阵A的一范数
    // 使用反向通信来评估矩阵-向量乘积
    // 这是CLACON的线程安全版本，它使用数组ISAVE代替静态变量
    
    // 函数起始
    # 定义整数变量 jlast，用于存储上一次迭代的值
    int jlast;
    # 定义双精度浮点变量 altsgn 和 estold，用于存储算法中的中间值
    double altsgn, estold;
    # 定义整数变量 i，用于循环迭代计数
    int i;
    # 定义双精度浮点变量 temp，用于临时存储计算结果
    double temp;
#ifdef _CRAY
    extern int ISAMAX(int *, double *, int *);
    extern double SASUM(int *, double *, int *);
    extern int SCOPY(int *, double *, int *, double *, int *);
#else
    extern int idamax_(int *, double *, int *);
    extern double dasum_(int *, double *, int *);
    extern int dcopy_(int *, double *, int *, double *, int *);
#endif

#define d_sign(a, b) (b >= 0 ? fabs(a) : -fabs(a))    /* Copy sign */
#define i_dnnt(a) \
    ( a>=0 ? floor(a+.5) : -floor(.5-a) ) /* Round to nearest integer */

if ( *kase == 0 ) {
    for (i = 0; i < *n; ++i) {
        x[i] = 1. / (double) (*n);   /* 初始化 x 数组为 1/n */
    }
    *kase = 1;   /* 设置 kase 为 1 */
    isave[0] = 1;    /* 设置 isave[0] 为 1，表示跳转位置 */
    return 0;   /* 返回 0 */
}

switch (isave[0]) {
case 1:  goto L20;
case 2:  goto L40;
case 3:  goto L70;
case 4:  goto L110;
case 5:  goto L140;
}

/*     ................ ENTRY   (isave[0] == 1)   
   FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X. */
L20:
if (*n == 1) {
    v[0] = x[0];    /* 将 x[0] 赋值给 v[0] */
    *est = fabs(v[0]);   /* 计算 v[0] 的绝对值并赋给 est */
    /*        ... QUIT */
    goto L150;    /* 跳转到标签 L150 */
}

#ifdef _CRAY
*est = SASUM(n, x, &c__1);   /* 使用 SASUM 计算 x 数组的和并赋给 est */
#else
*est = dasum_(n, x, &c__1);   /* 使用 dasum_ 计算 x 数组的和并赋给 est */
#endif

for (i = 0; i < *n; ++i) {
    x[i] = d_sign(one, x[i]);   /* 根据 x[i] 的符号给 x[i] 赋值 */
    isgn[i] = i_dnnt(x[i]);   /* 将 x[i] 四舍五入到最近的整数并赋给 isgn[i] */
}
*kase = 2;   /* 设置 kase 为 2 */
isave[0] = 2;  /* 设置 isave[0] 为 2，表示跳转位置 */
return 0;   /* 返回 0 */

/*     ................ ENTRY   (isave[0] == 2)   
   FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X. */
L40:
#ifdef _CRAY
isave[1] = ISAMAX(n, &x[0], &c__1);  /* 使用 ISAMAX 找到 x 数组中的最大值的索引并赋给 isave[1] */
#else
isave[1] = idamax_(n, &x[0], &c__1);  /* 使用 idamax_ 找到 x 数组中的最大值的索引并赋给 isave[1] */
#endif
--isave[1];  /* 将 isave[1] 减 1，即 --j */
isave[2] = 2; /* 设置 isave[2] 为 2，表示迭代次数 */

/*     MAIN LOOP - ITERATIONS 2,3,...,ITMAX. */
L50:
for (i = 0; i < *n; ++i) x[i] = zero;   /* 将 x 数组置零 */
x[isave[1]] = one;   /* 将 x[isave[1]] 设置为 1 */
*kase = 1;   /* 设置 kase 为 1 */
isave[0] = 3;  /* 设置 isave[0] 为 3，表示跳转位置 */
return 0;   /* 返回 0 */

/*     ................ ENTRY   (isave[0] == 3)   
   X HAS BEEN OVERWRITTEN BY A*X. */
L70:
#ifdef _CRAY
SCOPY(n, x, &c__1, v, &c__1);   /* 复制 x 到 v */
#else
dcopy_(n, x, &c__1, v, &c__1);   /* 复制 x 到 v */
#endif
estold = *est;   /* 将 est 赋给 estold */
#ifdef _CRAY
*est = SASUM(n, v, &c__1);   /* 使用 SASUM 计算 v 数组的和并赋给 est */
#else
*est = dasum_(n, v, &c__1);   /* 使用 dasum_ 计算 v 数组的和并赋给 est */
#endif

for (i = 0; i < *n; ++i)
if (i_dnnt(d_sign(one, x[i])) != isgn[i])
    goto L90;

/*     REPEATED SIGN VECTOR DETECTED, HENCE ALGORITHM HAS CONVERGED. */
goto L120;

L90:
/*     TEST FOR CYCLING. */
if (*est <= estold) goto L120;

for (i = 0; i < *n; ++i) {
x[i] = d_sign(one, x[i]);   /* 根据 x[i] 的符号给 x[i] 赋值 */
isgn[i] = i_dnnt(x[i]);   /* 将 x[i] 四舍五入到最近的整数并赋给 isgn[i] */
}
*kase = 2;   /* 设置 kase 为 2 */
isave[0] = 4;  /* 设置 isave[0] 为 4，表示跳转位置 */
return 0;   /* 返回 0 */

/*     ................ ENTRY   (isave[0] == 4)
   X HAS BEEN OVERWRITTEN BY TRANDPOSE(A)*X. */
L110:
jlast = isave[1];  /* 将 isave[1] 赋给 jlast */
#ifdef _CRAY
isave[1] = ISAMAX(n, &x[0], &c__1);  /* 使用 ISAMAX 找到 x 数组中的最大值的索引并赋给 isave[1] */
#else
isave[1] = idamax_(n, &x[0], &c__1);  /* 使用 idamax_ 找到 x 数组中的最大值的索引并赋给 isave[1] */
#endif
isave[1] = isave[1] - 1;  /* 将 isave[1] 减 1，即 --j */
if (x[jlast] != fabs(x[isave[1]]) && isave[2] < 5) {
isave[2] = isave[2] + 1;  /* 将 isave[2] 加 1，即 ++iter */
goto L50;   /* 跳转到标签 L50 */
}
    /*     迭代完成。最终阶段。 */
L120:
    altsgn = 1.;
    // 初始化交替符号为正值
    for (i = 1; i <= *n; ++i) {
        // 设置数组 x 的元素，交替使用正负符号
        x[i-1] = altsgn * ((double)(i - 1) / (double)(*n - 1) + 1.);
        // 切换符号
        altsgn = -altsgn;
    }
    *kase = 1;
    // 设置 kase 为 1
    isave[0] = 5;  /* jump = 5; */
    // 设置 isave[0] 为 5，用作跳转标记

    return 0;
    
    /*     ................ ENTRY   (isave[0] = 5)   
       X HAS BEEN OVERWRITTEN BY A*X. */
L140:
#ifdef _CRAY
    // 在 CRAY 平台上计算 x 的一范数的一半与 n 和 3 的乘积的比值，再乘以 2
    temp = SASUM(n, x, &c__1) / (double)(*n * 3) * 2.;
#else
    // 在非 CRAY 平台上使用 BLAS 函数计算 x 的一范数的一半与 n 和 3 的乘积的比值，再乘以 2
    temp = dasum_(n, x, &c__1) / (double)(*n * 3) * 2.;
#endif
    // 如果计算得到的 temp 大于当前的 est，则更新 est，并复制 x 到 v
    if (temp > *est) {
#ifdef _CRAY
        // 在 CRAY 平台上复制 x 到 v
        SCOPY(n, &x[0], &c__1, &v[0], &c__1);
#else
        // 在非 CRAY 平台上使用 BLAS 函数复制 x 到 v
        dcopy_(n, &x[0], &c__1, &v[0], &c__1);
#endif
        *est = temp;
    }

L150:
    *kase = 0;
    // 设置 kase 为 0
    return 0;

} /* dlacon2_ */
// 函数 dlacon2_ 的结束标记
```