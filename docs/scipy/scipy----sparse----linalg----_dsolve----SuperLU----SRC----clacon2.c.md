# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\clacon2.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file clacon2.c
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
#include "slu_scomplex.h"

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   CLACON2 estimates the 1-norm of a square matrix A.   
 *   Reverse communication is used for evaluating matrix-vector products. 
 * 
 *   This is a thread safe version of CLACON, which uses the array ISAVE
 *   in place of a STATIC variables, as follows:
 *
 *     CLACON     CLACON2
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
 *   V      (workspace) COMPLEX PRECISION array, dimension (N)   
 *          On the final return, V = A*W,  where  EST = norm(V)/norm(W)   
 *          (W is not returned).   
 *
 *   X      (input/output) COMPLEX PRECISION array, dimension (N)   
 *          On an intermediate return, X should be overwritten by   
 *                A * X,   if KASE=1,   
 *                A' * X,  if KASE=2,
 *          where A' is the conjugate transpose of A,
 *         and CLACON must be re-called with all the other parameters   
 *          unchanged.   
 *
 *
 *   EST    (output) FLOAT PRECISION   
 *          An estimate (a lower bound) for norm(A).   
 *
 *   KASE   (input/output) INT
 *          On the initial call to CLACON, KASE should be 0.   
 *          On an intermediate return, KASE will be 1 or 2, indicating   
 *          whether X should be overwritten by A * X  or A' * X.   
 *          On the final return from CLACON, KASE will again be 0.   
 *
 *   isave  (input/output) int [3]
 *          ISAVE is INTEGER array, dimension (3)
 *          ISAVE is used to save variables between calls to CLACON2
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
clacon2_(int *n, singlecomplex *v, singlecomplex *x, float *est, int *kase, int isave[3])
{
    /* Table of constant values */
    int c__1 = 1;
    singlecomplex      zero = {0.0, 0.0};

    // 定义常量 c__1 为整数值 1，用于后续计算
    // 定义复数变量 zero，并初始化为 {0.0, 0.0}
    /* 定义一个复数类型变量 one，并初始化为 {1.0, 0.0} */
    singlecomplex      one = {1.0, 0.0};

    /* 系统生成的局部变量 */
    float d__1;
    
    /* 局部变量声明 */
    int jlast;
    float altsgn, estold;
    int i;
    float temp;
    float safmin;

    /* 外部函数声明 */
    extern float smach(char *);
    extern int icmax1_slu(int *, singlecomplex *, int *);
    extern double scsum1_slu(int *, singlecomplex *, int *);
    extern int ccopy_(int *, singlecomplex *, int *, singlecomplex *, int *);

    /* 调用 smach 函数获取 "Safe minimum" 值，赋给 safmin */
    safmin = smach("Safe minimum");

    /* 如果 kase 等于 0 */
    if ( *kase == 0 ) {
        /* 初始化向量 x，使每个元素的实部为 1/n，虚部为 0 */
        for (i = 0; i < *n; ++i) {
            x[i].r = 1. / (float) (*n);
            x[i].i = 0.;
        }
        /* 将 kase 设置为 1 */
        *kase = 1;
        /* 设置 isave[0] 为 1，表示 jump = 1 */
        isave[0] = 1;    /* jump = 1; */
        /* 返回 0 */
        return 0;
    }

    /* 根据 isave[0] 的不同值执行不同的跳转 */
    switch (isave[0]) {
    case 1:  goto L20;
    case 2:  goto L40;
    case 3:  goto L70;
    case 4:  goto L110;
    case 5:  goto L140;
    }

    /*     ................ ENTRY   (isave[0] == 1)   
       第一次迭代。X 已经被 A*X 覆盖。 */
  L20:
    /* 如果 n 等于 1 */
    if (*n == 1) {
        /* 将 x[0] 赋给 v[0] */
        v[0] = x[0];
        /* 计算估计值 est 为 v[0] 的绝对值 */
        *est = c_abs(&v[0]);
        /* 跳转到 L150 结束 */
        goto L150;
    }
    /* 计算 scsum1_slu 函数的返回值赋给 est */
    *est = scsum1_slu(n, x, &c__1);

    /* 对 x 中的每个元素进行处理 */
    for (i = 0; i < *n; ++i) {
        /* 计算 x[i] 的绝对值 */
        d__1 = c_abs(&x[i]);
        /* 如果绝对值大于 safmin */
        if (d__1 > safmin) {
            /* 计算倒数并乘以 x[i] */
            d__1 = 1 / d__1;
            x[i].r *= d__1;
            x[i].i *= d__1;
        } else {
            /* 否则将 x[i] 赋值为 one */
            x[i] = one;
        }
    }
    /* 设置 kase 为 2 */
    *kase = 2;
    /* 设置 isave[0] 为 2，表示 jump = 2 */
    isave[0] = 2;  /* jump = 2; */
    /* 返回 0 */
    return 0;

    /*     ................ ENTRY   (isave[0] == 2)   
       第一次迭代。X 已经被 A^T*X 覆盖。 */
L40:
    isave[1] = icmax1_slu(n, &x[0], &c__1);  /* 获取列维度最大的元素的索引，并将其存储在isave数组的第一个位置 */
    --isave[1];  /* 将isave数组的第一个位置的值减1，即执行 --j 操作 */
    isave[2] = 2; /* 设置isave数组的第二个位置为2，表示迭代次数为2 */

    /*     主循环 - 迭代 2,3,...,ITMAX. */
L50:
    for (i = 0; i < *n; ++i) x[i] = zero;  /* 将数组x中的所有元素置为零 */
    x[isave[1]] = one;  /* 将x数组中isave数组的第一个位置索引指示的元素设为1 */
    *kase = 1;  /* 将kase设为1 */
    isave[0] = 3;  /* 设置isave数组的第一个位置为3，表示跳转目标为3 */

    return 0;

    /*     ................ ENTRY   (isave[0] == 3)   
       X HAS BEEN OVERWRITTEN BY A*X. */
L70:
#ifdef _CRAY
    CCOPY(n, x, &c__1, v, &c__1);  /* 如果是Cray系统，执行向量复制操作 */
#else
    ccopy_(n, x, &c__1, v, &c__1);  /* 否则调用ccopy_函数执行向量复制操作 */
#endif
    estold = *est;  /* 将est的值存储在estold中 */
    *est = scsum1_slu(n, v, &c__1);  /* 计算向量v的元素和，并将结果存储在est中 */


/* L90: */
    /*     检查循环性. */
    if (*est <= estold) goto L120;  /* 如果当前的est值小于等于之前的estold值，则跳转到标签L120 */

    for (i = 0; i < *n; ++i) {
    d__1 = c_abs(&x[i]);  /* 计算x数组中第i个元素的复数绝对值 */
    if (d__1 > safmin) {
        d__1 = 1 / d__1;  /* 计算复数的倒数 */
        x[i].r *= d__1;  /* 对x数组中第i个元素的实部乘以d__1 */
        x[i].i *= d__1;  /* 对x数组中第i个元素的虚部乘以d__1 */
    } else {
        x[i] = one;  /* 如果绝对值小于等于safmin，则将x数组中第i个元素设为1 */
    }
    }
    *kase = 2;  /* 将kase设为2 */
    isave[0] = 4;  /* 设置isave数组的第一个位置为4，表示跳转目标为4 */

    return 0;

    /*     ................ ENTRY   (isave[0] == 4)
       X HAS BEEN OVERWRITTEN BY TRANDPOSE(A)*X. */
L110:
    jlast = isave[1];  /* 将isave数组的第一个位置的值赋给jlast */
    isave[1] = icmax1_slu(n, &x[0], &c__1);  /* 获取列维度最大的元素的索引，并将其存储在isave数组的第一个位置 */
    isave[1] = isave[1] - 1;  /* 将isave数组的第一个位置的值减1，即执行 --j 操作 */
    if (x[jlast].r != (d__1 = x[isave[1]].r, fabs(d__1)) && isave[2] < 5) {
    isave[2] = isave[2] + 1;  /* 将isave数组的第二个位置的值加1，即执行 ++iter 操作 */
    goto L50;  /* 跳转到标签L50 */
    }

    /*     迭代完成.  最后阶段. */
L120:
    altsgn = 1.;  /* 将altsgn设为1.0 */
    for (i = 1; i <= *n; ++i) {
    x[i-1].r = altsgn * ((float)(i - 1) / (float)(*n - 1) + 1.);  /* 根据i计算x数组中每个元素的实部 */
    x[i-1].i = 0.;  /* 将x数组中每个元素的虚部设为0 */
    altsgn = -altsgn;  /* 切换altsgn的符号 */
    }
    *kase = 1;  /* 将kase设为1 */
    isave[0] = 5;  /* 设置isave数组的第一个位置为5，表示跳转目标为5 */

    return 0;
    
    /*     ................ ENTRY   (isave[0] = 5)   
       X HAS BEEN OVERWRITTEN BY A*X. */
L140:
    temp = scsum1_slu(n, x, &c__1) / (float)(*n * 3) * 2.;  /* 计算x数组元素和的1/3倍，并乘以2，将结果存储在temp中 */
    if (temp > *est) {
#ifdef _CRAY
    CCOPY(n, &x[0], &c__1, &v[0], &c__1);  /* 如果是Cray系统，执行向量复制操作 */
#else
    ccopy_(n, &x[0], &c__1, &v[0], &c__1);  /* 否则调用ccopy_函数执行向量复制操作 */
#endif
    *est = temp;  /* 更新est的值为temp */
    }

L150:
    *kase = 0;  /* 将kase设为0 */
    return 0;

} /* clacon2_ */
```