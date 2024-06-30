# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cgscon.c`

```
/*
 * File name:    cgscon.c
 * History:     Modified from lapack routines CGECON.
 */

#include <math.h>
#include "slu_cdefs.h"

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   CGSCON estimates the reciprocal of the condition number of a general 
 *   real matrix A, in either the 1-norm or the infinity-norm, using   
 *   the LU factorization computed by CGETRF.   *
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
 *            cgstrf(). Use compressed row subscripts storage for supernodes,
 *            i.e., L has types: Stype = SLU_SC, Dtype = SLU_C, Mtype = SLU_TRLU.
 * 
 *    U       (input) SuperMatrix*
 *            The factor U from the factorization Pr*A*Pc=L*U as computed by
 *            cgstrf(). Use column-wise storage scheme, i.e., U has types:
 *            Stype = SLU_NC, Dtype = SLU_C, Mtype = SLU_TRU.
 *        
 *    ANORM   (input) float
 *            If NORM = '1' or 'O', the 1-norm of the original matrix A.   
 *            If NORM = 'I', the infinity-norm of the original matrix A.
 *        
 *    RCOND   (output) float*
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
cgscon(char *norm, SuperMatrix *L, SuperMatrix *U,
       float anorm, float *rcond, SuperLUStat_t *stat, int *info)
{
    /* Local variables */
    int    kase, kase1, onenrm;
    float ainvnm;
    singlecomplex *work;
    int    isave[3];

    /*! \brief
     * 
     * <pre>
     * Purpose:
     * -------
     * 
     * CGSCON estimates the reciprocal of the condition number of a general 
     * real matrix A, in either the 1-norm or the infinity-norm, using   
     * the LU factorization computed by CGETRF.   
     * 
     * Arguments:
     * ---------
     * 
     * NORM    (input) char*
     *         Specifies whether the 1-norm condition number or the   
     *         infinity-norm condition number is required:   
     *         = '1' or 'O':  1-norm;   
     *         = 'I':         Infinity-norm.
     * 
     * L       (input) SuperMatrix*
     *         The factor L from the factorization Pr*A*Pc=L*U as computed by
     *         cgstrf(). Use compressed row subscripts storage for supernodes,
     *         i.e., L has types: Stype = SLU_SC, Dtype = SLU_C, Mtype = SLU_TRLU.
     * 
     * U       (input) SuperMatrix*
     *         The factor U from the factorization Pr*A*Pc=L*U as computed by
     *         cgstrf(). Use column-wise storage scheme, i.e., U has types:
     *         Stype = SLU_NC, Dtype = SLU_C, Mtype = SLU_TRU.
     * 
     * ANORM   (input) float
     *         If NORM = '1' or 'O', the 1-norm of the original matrix A.   
     *         If NORM = 'I', the infinity-norm of the original matrix A.
     * 
     * RCOND   (output) float*
     *         The reciprocal of the condition number of the matrix A,   
     *         computed as RCOND = 1/(norm(A) * norm(inv(A))).
     * 
     * STAT    (input) SuperLUStat_t*
     *         SuperLU structure containing statistics about the factorization.
     * 
     * INFO    (output) int*
     *         = 0:  successful exit   
     *         < 0:  if INFO = -i, the i-th argument had an illegal value   
     * 
     * </pre>
     */

    // Initialization of local variables
    kase = 0;
    kase1 = 1;
    onenrm = (*norm == '1' || *norm == 'O');

    // Allocate memory for work array
    work = (singlecomplex *) SUPERLU_MALLOC(3 * sizeof(singlecomplex));

    // Check if memory allocation was successful
    if (!work) {
        *info = -1;
        return;
    }

    // Estimate the norm of the inverse of A using CGECON routine
    cgecon(norm, L->nrow, U->nzval, L->fstnz, work, isave, &ainvnm, info);

    // Free allocated memory for work array
    SUPERLU_FREE(work);

    // If successful, compute the reciprocal of the condition number
    if (*info == 0) {
        if (onenrm) {
            *rcond = (float) (1.0 / ((anorm * ainvnm)));
        } else {
            *rcond = (float) (1.0 / ((anorm * ainvnm)));
        }
    }

    return;
}
    /* 调用外部函数 crscl_，对输入参数进行缩放 */
    extern int crscl_(int *, singlecomplex *, singlecomplex *, int *);

    /* 调用外部函数 clacon2_，检验输入参数 */
    extern int clacon2_(int *, singlecomplex *, singlecomplex *, float *, int *, int []);

    /* 测试输入参数的有效性 */
    *info = 0;  // 将错误信息初始化为0
    onenrm = *(unsigned char *)norm == '1' || strncmp(norm, "O", 1)==0;  // 检查 norm 是否为 '1' 或 "O"
    if (! onenrm && strncmp(norm, "I", 1)!=0) *info = -1;  // 如果 norm 既不是 '1' 也不是 "I"，设置错误信息为 -1
    else if (L->nrow < 0 || L->nrow != L->ncol ||
             L->Stype != SLU_SC || L->Dtype != SLU_C || L->Mtype != SLU_TRLU)
        *info = -2;  // 如果 L 的行数小于0，或者 L 不是 SC 类型，或者不是 C 类型，或者不是 TRLU 类型，设置错误信息为 -2
    else if (U->nrow < 0 || U->nrow != U->ncol ||
             U->Stype != SLU_NC || U->Dtype != SLU_C || U->Mtype != SLU_TRU)
        *info = -3;  // 如果 U 的行数小于0，或者 U 不是 NC 类型，或者不是 C 类型，或者不是 TRU 类型，设置错误信息为 -3
    if (*info != 0) {
        int ii = -(*info);
        input_error("cgscon", &ii);  // 调用 input_error 函数报告输入错误
        return;  // 返回
    }

    /* 如果可能，快速返回 */
    *rcond = 0.;  // 初始化 rcond 为 0
    if ( L->nrow == 0 || U->nrow == 0) {
        *rcond = 1.;  // 如果 L 或 U 的行数为 0，则设置 rcond 为 1，直接返回
        return;
    }

    work = complexCalloc( 3*L->nrow );  // 分配复数类型的工作数组空间，大小为 3*L->nrow

    if ( !work )
        ABORT("Malloc fails for work arrays in cgscon.");  // 如果分配失败，调用 ABORT 函数终止程序

    /* 估算 inv(A) 的范数 */
    ainvnm = 0.;  // 初始化 ainvnm 为 0
    if ( onenrm ) kase1 = 1;  // 如果使用 1-范数，设置 kase1 为 1
    else kase1 = 2;  // 否则设置 kase1 为 2
    kase = 0;  // 初始化 kase 为 0

    int nrow = L->nrow;  // 将 L 的行数赋给 nrow

    do {
        clacon2_(&nrow, &work[L->nrow], &work[0], &ainvnm, &kase, isave);  // 调用 clacon2_ 函数

        if (kase == 0) break;  // 如果 kase 等于 0，跳出循环

        if (kase == kase1) {
            /* 乘以 inv(L) */
            sp_ctrsv("L", "No trans", "Unit", L, U, &work[0], stat, info);  // 调用 sp_ctrsv 函数，进行 L 的逆运算

            /* 乘以 inv(U) */
            sp_ctrsv("U", "No trans", "Non-unit", L, U, &work[0], stat, info);  // 调用 sp_ctrsv 函数，进行 U 的逆运算

        } else {

            /* 乘以 inv(U') */
            sp_ctrsv("U", "Transpose", "Non-unit", L, U, &work[0], stat, info);  // 调用 sp_ctrsv 函数，进行 U 的转置逆运算

            /* 乘以 inv(L') */
            sp_ctrsv("L", "Transpose", "Unit", L, U, &work[0], stat, info);  // 调用 sp_ctrsv 函数，进行 L 的转置逆运算

        }

    } while ( kase != 0 );  // 当 kase 不为 0 时继续循环

    /* 计算倒数条件数的估计 */
    if (ainvnm != 0.) *rcond = (1. / ainvnm) / anorm;  // 如果 ainvnm 不为 0，则计算倒数条件数的估计值

    SUPERLU_FREE (work);  // 释放工作数组的内存空间
    return;  // 返回
} /* cgscon */
```