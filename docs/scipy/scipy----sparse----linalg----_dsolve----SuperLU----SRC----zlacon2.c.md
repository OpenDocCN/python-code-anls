# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zlacon2.c`

```
/*! @file zlacon2.c
 * \brief Estimates the 1-norm
 *
 * <pre>
 * -- SuperLU routine (version 5.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * July 24, 2022
 * </pre>
 */
#include <math.h>              // 包含数学函数库
#include "slu_Cnames.h"        // 包含 SLU 的 C 名称定义
#include "slu_dcomplex.h"      // 包含双精度复数类型定义

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   ZLACON2 estimates the 1-norm of a square matrix A.   
 *   Reverse communication is used for evaluating matrix-vector products. 
 * 
 *   This is a thread safe version of CLACON, which uses the array ISAVE
 *   in place of a STATIC variables, as follows:
 *
 *     ZLACON     ZLACON2
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
 *   V      (workspace) DOUBLE COMPLEX PRECISION array, dimension (N)   
 *          On the final return, V = A*W,  where  EST = norm(V)/norm(W)   
 *          (W is not returned).   
 *
 *   X      (input/output) DOUBLE COMPLEX PRECISION array, dimension (N)   
 *          On an intermediate return, X should be overwritten by   
 *                A * X,   if KASE=1,   
 *                A' * X,  if KASE=2,
 *          where A' is the conjugate transpose of A,
 *         and ZLACON must be re-called with all the other parameters   
 *          unchanged.   
 *
 *
 *   EST    (output) DOUBLE PRECISION   
 *          An estimate (a lower bound) for norm(A).   
 *
 *   KASE   (input/output) INT
 *          On the initial call to ZLACON, KASE should be 0.   
 *          On an intermediate return, KASE will be 1 or 2, indicating   
 *          whether X should be overwritten by A * X  or A' * X.   
 *          On the final return from ZLACON, KASE will again be 0.   
 *
 *   isave  (input/output) int [3]
 *          ISAVE is INTEGER array, dimension (3)
 *          ISAVE is used to save variables between calls to ZLACON2
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
zlacon2_(int *n, doublecomplex *v, doublecomplex *x, double *est, int *kase, int isave[3])
{
    /* Table of constant values */
    int c__1 = 1;              // 设置整型常量 c__1 为 1
    doublecomplex      zero = {0.0, 0.0};  // 创建复数常量 zero，值为 {0.0, 0.0}
    // 定义一个复数结构体 one，赋值为 {1.0, 0.0}
    doublecomplex one = {1.0, 0.0};

    /* System generated locals */
    // 系统自动生成的局部变量
    double d__1;
    
    /* Local variables */
    // 本地变量声明
    int jlast;
    double altsgn, estold;
    int i;
    double temp;
    double safmin;
    // 外部函数声明
    extern double dmach(char *);
    extern int izmax1_slu(int *, doublecomplex *, int *);
    extern double dzsum1_slu(int *, doublecomplex *, int *);
    extern int zcopy_(int *, doublecomplex *, int *, doublecomplex *, int *);

    // 获取安全最小值并赋给 safmin
    safmin = dmach("Safe minimum");
    // 如果 kase 等于 0
    if ( *kase == 0 ) {
        // 初始化 x 数组的值，使其每个元素的实部为 1/(double)(*n)，虚部为 0
        for (i = 0; i < *n; ++i) {
            x[i].r = 1. / (double) (*n);
            x[i].i = 0.;
        }
        // 设置 kase 为 1
        *kase = 1;
        // 设置 isave[0] 为 1，表示跳转目标为 1
        isave[0] = 1;    /* jump = 1; */
        // 返回 0
        return 0;
    }

    // 根据 isave[0] 的值进行跳转
    switch (isave[0]) {
    case 1:  goto L20;
    case 2:  goto L40;
    case 3:  goto L70;
    case 4:  goto L110;
    case 5:  goto L140;
    }

    //     ................ ENTRY   (isave[0] == 1)   
    // 第一次迭代。X 已被 A*X 覆盖。
  L20:
    // 如果 n 等于 1
    if (*n == 1) {
        // 将 x[0] 复制给 v[0]
        v[0] = x[0];
        // 计算 v[0] 的绝对值并赋给 est
        *est = z_abs(&v[0]);
        //        ... QUIT
        // 跳转到 L150
        goto L150;
    }
    // 计算 x 数组的 L1 范数并赋给 est
    *est = dzsum1_slu(n, x, &c__1);

    // 遍历 x 数组的每个元素
    for (i = 0; i < *n; ++i) {
        // 计算 x[i] 的绝对值
        d__1 = z_abs(&x[i]);
        // 如果绝对值大于 safmin
        if (d__1 > safmin) {
            // 计算倒数
            d__1 = 1 / d__1;
            // 对 x[i] 进行缩放
            x[i].r *= d__1;
            x[i].i *= d__1;
        } else {
            // 否则，将 x[i] 设置为 one
            x[i] = one;
        }
    }
    // 设置 kase 为 2
    *kase = 2;
    // 设置 isave[0] 为 2，表示跳转目标为 2
    isave[0] = 2;  /* jump = 2; */
    // 返回 0
    return 0;

    //     ................ ENTRY   (isave[0] == 2)   
    // 第一次迭代。X 已被转置(A)*X 覆盖。
    isave[1] = izmax1_slu(n, &x[0], &c__1);  /* 将最大绝对值元素的索引保存到isave[1]中 */
    --isave[1];  /* 将j减一，即--j */

    isave[2] = 2; /* 设置迭代次数为2 */

    /*     主循环 - 迭代次数2,3,...,ITMAX. */
L50:
    for (i = 0; i < *n; ++i) x[i] = zero;  /* 将x数组清零 */
    x[isave[1]] = one;  /* 将x中isave[1]位置设为1 */
    *kase = 1;  /* 设置kase为1，表示下一个步骤将进行A*x运算 */
    isave[0] = 3;  /* 设置跳转目标为3 */

    return 0;

    /*     ................ ENTRY   (isave[0] == 3)
       X 已被 A*X 覆盖. */
L70:
#ifdef _CRAY
    CCOPY(n, x, &c__1, v, &c__1);  /* 使用CCOPY复制x到v（针对_Cray系统） */
#else
    zcopy_(n, x, &c__1, v, &c__1);  /* 使用zcopy_复制x到v */
#endif
    estold = *est;  /* 保存旧的估计值到estold */
    *est = dzsum1_slu(n, v, &c__1);  /* 计算向量v的绝对值之和，保存到*est中 */

/* L90: */
    /*     测试是否进入循环. */
    if (*est <= estold) goto L120;  /* 如果估计值不再改善，跳转到L120 */

    for (i = 0; i < *n; ++i) {
        d__1 = z_abs(&x[i]);  /* 计算x[i]的复数绝对值 */
        if (d__1 > safmin) {  /* 如果绝对值大于safmin */
            d__1 = 1 / d__1;  /* 计算倒数 */
            x[i].r *= d__1;  /* 对x[i]的实部进行缩放 */
            x[i].i *= d__1;  /* 对x[i]的虚部进行缩放 */
        } else {
            x[i] = one;  /* 否则将x[i]设为1 */
        }
    }
    *kase = 2;  /* 设置kase为2，表示下一个步骤将进行A^T*x运算 */
    isave[0] = 4;  /* 设置跳转目标为4 */

    return 0;

    /*     ................ ENTRY   (isave[0] == 4)
       X 已被 A^T*X 覆盖. */
L110:
    jlast = isave[1];  /* 将上次迭代中最大绝对值元素的索引保存到jlast中 */
    isave[1] = izmax1_slu(n, &x[0], &c__1);  /* 计算最大绝对值元素的索引，保存到isave[1]中 */
    isave[1] = isave[1] - 1;  /* 将j减一，即--j */
    if (x[jlast].r != (d__1 = x[isave[1]].r, fabs(d__1)) && isave[2] < 5) {
        isave[2] = isave[2] + 1;  /* 迭代次数加一 */
        goto L50;  /* 跳转到L50，进行下一次迭代 */
    }

    /*     迭代完成. 最终阶段. */
L120:
    altsgn = 1.;  /* 初始化交替符号为1 */
    for (i = 1; i <= *n; ++i) {
        x[i-1].r = altsgn * ((double)(i - 1) / (double)(*n - 1) + 1.);  /* 设置x的实部 */
        x[i-1].i = 0.;  /* 设置x的虚部为0 */
        altsgn = -altsgn;  /* 切换交替符号 */
    }
    *kase = 1;  /* 设置kase为1，表示下一个步骤将进行A*x运算 */
    isave[0] = 5;  /* 设置跳转目标为5 */

    return 0;

    /*     ................ ENTRY   (isave[0] == 5)
       X 已被 A*X 覆盖. */
L140:
    temp = dzsum1_slu(n, x, &c__1) / (double)(*n * 3) * 2.;  /* 计算x的绝对值之和的一种估计 */
    if (temp > *est) {
#ifdef _CRAY
    CCOPY(n, &x[0], &c__1, &v[0], &c__1);  /* 使用CCOPY复制x到v（针对_Cray系统） */
#else
    zcopy_(n, &x[0], &c__1, &v[0], &c__1);  /* 使用zcopy_复制x到v */
#endif
    *est = temp;  /* 更新估计值 */
    }

L150:
    *kase = 0;  /* 设置kase为0，表示算法结束 */
    return 0;

} /* zlacon2_ */
```