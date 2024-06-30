# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\slacon2.c`

```
/*! @file slacon2.c
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
 *   SLACON2 estimates the 1-norm of a square matrix A.   
 *   Reverse communication is used for evaluating matrix-vector products. 
 * 
 *   This is a thread safe version of CLACON, which uses the array ISAVE
 *   in place of a STATIC variables, as follows:
 *
 *     SLACON     SLACON2
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
 *   V      (workspace) FLOAT PRECISION array, dimension (N)   
 *          On the final return, V = A*W,  where  EST = norm(V)/norm(W)   
 *          (W is not returned).   
 *
 *   X      (input/output) FLOAT PRECISION array, dimension (N)   
 *          On an intermediate return, X should be overwritten by   
 *                A * X,   if KASE=1,   
 *                A' * X,  if KASE=2,
 *         and SLACON must be re-called with all the other parameters   
 *          unchanged.   
 *
 *   ISGN   (workspace) INT array, dimension (N)
 *
 *   EST    (output) FLOAT PRECISION   
 *          An estimate (a lower bound) for norm(A).   
 *
 *   KASE   (input/output) INT
 *          On the initial call to SLACON, KASE should be 0.   
 *          On an intermediate return, KASE will be 1 or 2, indicating   
 *          whether X should be overwritten by A * X  or A' * X.   
 *          On the final return from SLACON, KASE will again be 0.   
 *
 *   isave  (input/output) int [3]
 *          ISAVE is INTEGER array, dimension (3)
 *          ISAVE is used to save variables between calls to SLACON2
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

/*! \brief
 *  Estimates the 1-norm of a square matrix A using reverse communication.
 *
 *  This function estimates the 1-norm of a square matrix A using iterative
 *  methods with reverse communication, utilizing the array ISAVE to store
 *  intermediate values across calls.
 *
 *  Arguments
 *  ---------
 *  N      (input) INT
 *         The order of the matrix. N >= 1.
 *
 *  V      (workspace) FLOAT PRECISION array, dimension (N)
 *         Workspace for intermediate calculations.
 *
 *  X      (input/output) FLOAT PRECISION array, dimension (N)
 *         On input, X represents the vector to be multiplied by A or A'.
 *         On output, depending on KASE:
 *           - If KASE=1, X is overwritten by A * X.
 *           - If KASE=2, X is overwritten by A' * X.
 *         After each computation, SLACON2 should be called again with
 *         unchanged parameters.
 *
 *  ISGN   (workspace) INT array, dimension (N)
 *         Workspace to store signs of elements.
 *
 *  EST    (output) FLOAT PRECISION
 *         An estimate (lower bound) for the 1-norm of matrix A.
 *
 *  KASE   (input/output) INT
 *         On input:
 *           - 0: Initial call.
 *         On output (after each intermediate return):
 *           - 1: X should be multiplied by A.
 *           - 2: X should be multiplied by A'.
 *         On final return, KASE is reset to 0.
 *
 *  isave  (input/output) int [3]
 *         Integer array of dimension (3) to store intermediate values:
 *           - isave[0]: Jump
 *           - isave[1]: j
 *           - isave[2]: iter
 *         These variables are used internally to maintain state across calls.
 */
int
slacon2_(int *n, float *v, float *x, int *isgn, float *est, int *kase, int isave[3])
{
    /* Table of constant values */
    int c__1 = 1;
    float      zero = 0.0; // 常量值 0.0
    float      one = 1.0;  // 常量值 1.0
    
    /* Local variables */
    int jlast;  // 上一个 j 的值
    // 声明浮点型变量 altsgn 和 estold，以及整型变量 i
    float altsgn, estold;
    int i;
    // 声明浮点型变量 temp
    float temp;
#ifdef _CRAY
    extern int ISAMAX(int *, float *, int *);
    extern float SASUM(int *, float *, int *);
    extern int SCOPY(int *, float *, int *, float *, int *);
#else
    extern int isamax_(int *, float *, int *);
    extern float sasum_(int *, float *, int *);
    extern int scopy_(int *, float *, int *, float *, int *);
#endif

#define d_sign(a, b) (b >= 0 ? fabs(a) : -fabs(a))    /* Copy sign */
#define i_dnnt(a) \
    ( a>=0 ? floor(a+.5) : -floor(.5-a) ) /* Round to nearest integer */

// 判断 kase 的值，根据不同的情况执行相应的操作
if ( *kase == 0 ) {
    for (i = 0; i < *n; ++i) {
        x[i] = 1. / (float) (*n);    // 将 x 数组中的元素初始化为 1/n
    }
    *kase = 1;    // 设置 kase 为 1
    isave[0] = 1;    /* jump = 1; */    // 设置 isave[0] 为 1，表示跳转到标签 L20 处
    return 0;    // 返回 0
}

// 根据 isave[0] 的值执行不同的跳转
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
    v[0] = x[0];
    *est = fabs(v[0]);
    /*        ... QUIT */
    goto L150;    // 跳转到标签 L150 处，结束
}

#ifdef _CRAY
*est = SASUM(n, x, &c__1);    // 计算 x 数组的绝对值和，使用 SASUM 函数
#else
*est = sasum_(n, x, &c__1);    // 计算 x 数组的绝对值和，使用 sasum_ 函数
#endif

for (i = 0; i < *n; ++i) {
    x[i] = d_sign(one, x[i]);    // 根据 x[i] 的正负号来更新 x[i] 的值
    isgn[i] = i_dnnt(x[i]);    // 将 x[i] 转换为最接近的整数，并保存在 isgn[i] 中
}
*kase = 2;    // 设置 kase 为 2
isave[0] = 2;  /* jump = 2; */    // 设置 isave[0] 为 2，表示跳转到标签 L40 处
return 0;    // 返回 0

/*     ................ ENTRY   (isave[0] == 2)   
   FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X. */
L40:
#ifdef _CRAY
isave[1] = ISAMAX(n, &x[0], &c__1);  /* j */
#else
isave[1] = isamax_(n, &x[0], &c__1);  /* j */
#endif
--isave[1];  /* --j; */    // 将 isave[1] 减少 1，表示 j 的索引

isave[2] = 2; /* iter = 2; */    // 设置 isave[2] 为 2，表示当前迭代次数为 2

/*     MAIN LOOP - ITERATIONS 2,3,...,ITMAX. */
L50:
for (i = 0; i < *n; ++i) x[i] = zero;    // 将 x 数组所有元素置零
x[isave[1]] = one;    // 将 x[isave[1]] 设置为 1
*kase = 1;    // 设置 kase 为 1
isave[0] = 3;  /* jump = 3; */    // 设置 isave[0] 为 3，表示跳转到标签 L70 处
return 0;    // 返回 0

/*     ................ ENTRY   (isave[0] == 3)   
   X HAS BEEN OVERWRITTEN BY A*X. */
L70:
#ifdef _CRAY
SCOPY(n, x, &c__1, v, &c__1);
#else
scopy_(n, x, &c__1, v, &c__1);
#endif
estold = *est;    // 将当前的 est 值保存到 estold 中

#ifdef _CRAY
*est = SASUM(n, v, &c__1);    // 计算 v 数组的绝对值和，使用 SASUM 函数
#else
*est = sasum_(n, v, &c__1);    // 计算 v 数组的绝对值和，使用 sasum_ 函数
#endif

for (i = 0; i < *n; ++i)
if (i_dnnt(d_sign(one, x[i])) != isgn[i])
    goto L90;    // 如果某些条件不满足，则跳转到标签 L90 处

/*     REPEATED SIGN VECTOR DETECTED, HENCE ALGORITHM HAS CONVERGED. */
goto L120;    // 跳转到标签 L120 处

L90:
/*     TEST FOR CYCLING. */
if (*est <= estold) goto L120;    // 如果估计值小于等于前一个估计值，则跳转到标签 L120 处

for (i = 0; i < *n; ++i) {
x[i] = d_sign(one, x[i]);    // 根据 x[i] 的正负号来更新 x[i] 的值
isgn[i] = i_dnnt(x[i]);    // 将 x[i] 转换为最接近的整数，并保存在 isgn[i] 中
}
*kase = 2;    // 设置 kase 为 2
isave[0] = 4;  /* jump = 4; */    // 设置 isave[0] 为 4，表示跳转到标签 L110 处
return 0;    // 返回 0

/*     ................ ENTRY   (isave[0] == 4)
   X HAS BEEN OVERWRITTEN BY TRANDPOSE(A)*X. */
L110:
jlast = isave[1];  /* j; */    // 将 isave[1] 的值保存到 jlast 中

#ifdef _CRAY
isave[1] = ISAMAX(n, &x[0], &c__1);/* j */
#else
isave[1] = isamax_(n, &x[0], &c__1);  /* j */
#endif
isave[1] = isave[1] - 1;  /* --j; */    // 将 isave[1] 减少 1，表示 j 的索引

if (x[jlast] != fabs(x[isave[1]]) && isave[2] < 5) {
isave[2] = isave[2] + 1;  /* ++iter; */    // 增加迭代次数计数器 isave[2]
goto L50;    // 跳转到标签 L50 处
}

/*     ITERATION COMPLETE.  FINAL STAGE. */
L120:
    altsgn = 1.;  /* 初始化交替符号为正 */
    for (i = 1; i <= *n; ++i) {  /* 循环迭代从1到n */
        x[i-1] = altsgn * ((float)(i - 1) / (float)(*n - 1) + 1.);  /* 计算并存储 x[i-1] 的值 */
        altsgn = -altsgn;  /* 切换交替符号 */
    }
    *kase = 1;  /* 设置 kase 为 1 */
    isave[0] = 5;  /* 设置 isave[0] 为 5，标记跳转位置 */
    return 0;  /* 返回 */

    /*     ................ ENTRY   (isave[0] = 5)   
       X HAS BEEN OVERWRITTEN BY A*X. */
L140:
#ifdef _CRAY
    temp = SASUM(n, x, &c__1) / (float)(*n * 3) * 2.;  /* 计算 x 的绝对和的一部分 */
#else
    temp = sasum_(n, x, &c__1) / (float)(*n * 3) * 2.;  /* 调用外部函数计算 x 的绝对和的一部分 */
#endif
    if (temp > *est) {  /* 如果计算得到的值大于 est */
#ifdef _CRAY
    SCOPY(n, &x[0], &c__1, &v[0], &c__1);  /* 复制 x 到 v */
#else
    scopy_(n, &x[0], &c__1, &v[0], &c__1);  /* 调用外部函数复制 x 到 v */
#endif
    *est = temp;  /* 更新 est */
    }

L150:
    *kase = 0;  /* 设置 kase 为 0 */
    return 0;  /* 返回 */

} /* slacon2_ */
```