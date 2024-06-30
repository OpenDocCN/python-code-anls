# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zgscon.c`

```
/*! @file zgscon.c
 * \brief Estimates reciprocal of the condition number of a general matrix
 * 
 * <pre>
 * -- SuperLU routine (version 5.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * July 25, 2015
 *
 * Modified from lapack routines ZGECON.
 * </pre> 
 */

/*
 * File name:    zgscon.c
 * History:     Modified from lapack routines ZGECON.
 */
#include <math.h>
#include "slu_zdefs.h"

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   ZGSCON estimates the reciprocal of the condition number of a general 
 *   real matrix A, in either the 1-norm or the infinity-norm, using   
 *   the LU factorization computed by ZGETRF.   *
 *
 *   An estimate is obtained for norm(inv(A)), and the reciprocal of the   
 *   condition number is computed as   
 *      RCOND = 1 / ( norm(A) * norm(inv(A)) ).   
 *
 *   See supermatrix.h for the definition of 'SuperMatrix' structure.
 * 
 *   Arguments   
 *   =========   
 *
 *    NORM    (input) char*
 *            Specifies whether the 1-norm condition number or the   
 *            infinity-norm condition number is required:   
 *            = '1' or 'O':  1-norm;   
 *            = 'I':         Infinity-norm.
 *        
 *    L       (input) SuperMatrix*
 *            The factor L from the factorization Pr*A*Pc=L*U as computed by
 *            zgstrf(). Use compressed row subscripts storage for supernodes,
 *            i.e., L has types: Stype = SLU_SC, Dtype = SLU_Z, Mtype = SLU_TRLU.
 * 
 *    U       (input) SuperMatrix*
 *            The factor U from the factorization Pr*A*Pc=L*U as computed by
 *            zgstrf(). Use column-wise storage scheme, i.e., U has types:
 *            Stype = SLU_NC, Dtype = SLU_Z, Mtype = SLU_TRU.
 *        
 *    ANORM   (input) double
 *            If NORM = '1' or 'O', the 1-norm of the original matrix A.   
 *            If NORM = 'I', the infinity-norm of the original matrix A.
 *        
 *    RCOND   (output) double*
 *           The reciprocal of the condition number of the matrix A,   
 *           computed as RCOND = 1/(norm(A) * norm(inv(A))).
 *        
 *    INFO    (output) int*
 *           = 0:  successful exit   
 *           < 0:  if INFO = -i, the i-th argument had an illegal value   
 *
 *    ===================================================================== 
 * </pre>
 */

void
zgscon(char *norm, SuperMatrix *L, SuperMatrix *U,
       double anorm, double *rcond, SuperLUStat_t *stat, int *info)
{


    /* Local variables */
    int    kase, kase1, onenrm;
    double ainvnm;
    doublecomplex *work;
    int    isave[3];

    // 根据 LU 分解的结果计算矩阵 A 的条件数的倒数
    // 判断使用的是 1-norm 还是 infinity-norm
    onenrm = lsame_(norm, "1") || lsame_(norm, "O");
    // 初始化返回值为 0
    *info = 0;
    // 如果计算条件数的类型不合法，返回错误信息
    if (!onenrm && !lsame_(norm, "I")) {
        *info = -1;
    } else if (L->nrow < 0) {
        *info = -2;
    } else if (U->nrow < 0 || U->ncol < 0 || L->nrow != U->nrow) {
        *info = -3;
    } else if (anorm < 0.) {
        *info = -4;
    }
    // 如果有错误信息，则直接返回
    if (*info != 0) {
        xerbla_("zgscon", -*info);
        return;
    }

    // 分配工作空间
    work = doublecomplexMalloc(U->ncol);
    // 如果分配失败，返回错误信息
    if (work == NULL) {
        *info = -5;
        xerbla_("zgscon", *info);
        return;
    }

    // 估算 A 的逆的范数
    zlacon_(&U->ncol, work, work + U->ncol, isave, &ainvnm);

    // 释放工作空间
    doublecomplexFree(work);

    // 如果逆的范数为 0，则条件数的倒数设为 0
    if (ainvnm == 0.) {
        *rcond = 0.;
    } else {
        // 计算条件数的倒数
        *rcond = (1. / anorm) / ainvnm;
    }

    return;
}
    /* 调用外部函数 zrscl_ 对工作数组进行缩放 */
    extern int zrscl_(int *, doublecomplex *, doublecomplex *, int *);

    /* 调用外部函数 zlacon2_ 进行条件数估计 */
    extern int zlacon2_(int *, doublecomplex *, doublecomplex *, double *, int *, int []);

    /* Test the input parameters. */
    /* 检查输入参数。初始化 info 为 0 */
    *info = 0;
    /* 检查 norm 的值是否为 '1' 或者以 "O" 开头 */
    onenrm = *(unsigned char *)norm == '1' || strncmp(norm, "O", 1)==0;
    /* 如果 norm 不是 '1' 且不是以 "I" 开头，则将 info 设置为 -1 */
    if (! onenrm && strncmp(norm, "I", 1)!=0) *info = -1;
    /* 检查矩阵 L 的行数是否小于 0，或者行列不相等，或者稀疏类型不是 SLU_SC，数据类型不是 SLU_Z，存储类型不是 SLU_TRLU */
    else if (L->nrow < 0 || L->nrow != L->ncol ||
             L->Stype != SLU_SC || L->Dtype != SLU_Z || L->Mtype != SLU_TRLU)
     *info = -2;
    /* 检查矩阵 U 的行数是否小于 0，或者行列不相等，或者稀疏类型不是 SLU_NC，数据类型不是 SLU_Z，存储类型不是 SLU_TRU */
    else if (U->nrow < 0 || U->nrow != U->ncol ||
             U->Stype != SLU_NC || U->Dtype != SLU_Z || U->Mtype != SLU_TRU) 
    *info = -3;
    /* 如果 info 不等于 0，则说明有错误，调用 input_error 函数并返回 */
    if (*info != 0) {
    int ii = -(*info);
    input_error("zgscon", &ii);
    return;
    }

    /* Quick return if possible */
    /* 如果 L 或 U 的行数为 0，则直接返回条件数 rcond 为 1 */
    *rcond = 0.;
    if ( L->nrow == 0 || U->nrow == 0) {
    *rcond = 1.;
    return;
    }

    /* 为工作数组分配内存 */
    work = doublecomplexCalloc( 3*L->nrow );

    /* 如果分配失败，则调用 ABORT 函数终止程序 */
    if ( !work )
    ABORT("Malloc fails for work arrays in zgscon.");
    
    /* Estimate the norm of inv(A). */
    /* 估计 inv(A) 的范数 */
    ainvnm = 0.;
    /* 如果 onenrm 为真，则 kase1 设为 1，否则设为 2 */
    if ( onenrm ) kase1 = 1;
    else kase1 = 2;
    /* 初始化 kase 为 0 */
    kase = 0;

    /* 重复进行直到 kase 为 0 */
    int nrow = L->nrow;
    do {
    /* 调用 zlacon2_ 函数估计条件数，并更新 kase 和 ainvnm */
    zlacon2_(&nrow, &work[L->nrow], &work[0], &ainvnm, &kase, isave);

    /* 如果 kase 为 0，则跳出循环 */
    if (kase == 0) break;

    /* 根据 kase 的值选择相应的操作 */
    if (kase == kase1) {
        /* Multiply by inv(L). */
        /* 调用 sp_ztrsv 函数，对工作数组进行 L 的逆操作 */

        /* Multiply by inv(U). */
        /* 调用 sp_ztrsv 函数，对工作数组进行 U 的逆操作 */
        
    } else {

        /* Multiply by inv(U'). */
        /* 调用 sp_ztrsv 函数，对工作数组进行 U 的转置逆操作 */

        /* Multiply by inv(L'). */
        /* 调用 sp_ztrsv 函数，对工作数组进行 L 的转置逆操作 */
        
    }

    } while ( kase != 0 );

    /* Compute the estimate of the reciprocal condition number. */
    /* 计算倒数条件数的估计值 */
    if (ainvnm != 0.) *rcond = (1. / ainvnm) / anorm;

    /* 释放工作数组的内存 */
    SUPERLU_FREE (work);
    /* 返回结果 */
    return;
} /* zgscon */
```