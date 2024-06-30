# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dgscon.c`

```
/*! @file dgscon.c
 * \brief Estimates reciprocal of the condition number of a general matrix
 * 
 * <pre>
 * -- SuperLU routine (version 5.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * July 25, 2015
 *
 * Modified from lapack routines DGECON.
 * </pre> 
 */

/*
 * File name:    dgscon.c
 * History:     Modified from lapack routines DGECON.
 */
#include <math.h>
#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   DGSCON estimates the reciprocal of the condition number of a general 
 *   real matrix A, in either the 1-norm or the infinity-norm, using   
 *   the LU factorization computed by DGETRF.   *
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
 *            dgstrf(). Use compressed row subscripts storage for supernodes,
 *            i.e., L has types: Stype = SLU_SC, Dtype = SLU_D, Mtype = SLU_TRLU.
 * 
 *    U       (input) SuperMatrix*
 *            The factor U from the factorization Pr*A*Pc=L*U as computed by
 *            dgstrf(). Use column-wise storage scheme, i.e., U has types:
 *            Stype = SLU_NC, Dtype = SLU_D, Mtype = SLU_TRU.
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
dgscon(char *norm, SuperMatrix *L, SuperMatrix *U,
       double anorm, double *rcond, SuperLUStat_t *stat, int *info)
{


    /* Local variables */
    int    kase, kase1, onenrm;
    double ainvnm;
    double *work;
    int    *iwork;
    int    isave[3];

    // Determine whether to use 1-norm or infinity-norm for condition number estimation
    onenrm = lsame_(norm, "1") || lsame_(norm, "O");

    // Allocate workspace for the algorithm
    work = doubleMalloc(2 * L->ncol);
    iwork = intMalloc(L->ncol);

    // Estimate the reciprocal of the condition number
    dgsrfs(anorm, L, U, stat, work, iwork, rcond, &ainvnm, info);

    // Free allocated workspace
    SUPERLU_FREE(work);
    SUPERLU_FREE(iwork);

    return;
}


注释：
    /* 外部函数声明，drscl_ 用于对向量进行缩放，dlacon2_ 用于估计矩阵的条件数。 */
    extern int drscl_(int *, double *, double *, int *);
    extern int dlacon2_(int *, double *, double *, int *, double *, int *, int []);

    /* 测试输入参数的有效性。 */
    *info = 0;
    onenrm = *(unsigned char *)norm == '1' || strncmp(norm, "O", 1)==0;
    if (! onenrm && strncmp(norm, "I", 1)!=0) *info = -1;
    else if (L->nrow < 0 || L->nrow != L->ncol ||
             L->Stype != SLU_SC || L->Dtype != SLU_D || L->Mtype != SLU_TRLU)
     *info = -2;
    else if (U->nrow < 0 || U->nrow != U->ncol ||
             U->Stype != SLU_NC || U->Dtype != SLU_D || U->Mtype != SLU_TRU) 
    *info = -3;
    if (*info != 0) {
    int ii = -(*info);
    input_error("dgscon", &ii);
    return;
    }

    /* 若可能，快速返回 */
    *rcond = 0.;
    if ( L->nrow == 0 || U->nrow == 0) {
    *rcond = 1.;
    return;
    }

    /* 分配工作空间和整数工作数组 */
    work = doubleCalloc( 3*L->nrow );
    iwork = int32Malloc( L->nrow );

    /* 检查分配是否成功 */
    if ( !work || !iwork )
    ABORT("Malloc fails for work arrays in dgscon.");
    
    /* 估计 inv(A) 的范数 */
    ainvnm = 0.;
    if ( onenrm ) kase1 = 1;
    else kase1 = 2;
    kase = 0;

    int nrow = L->nrow;

    /* 循环计算 inv(A) 的范数估计 */
    do {
    dlacon2_(&nrow, &work[L->nrow], &work[0], &iwork[0], &ainvnm, &kase, isave);

    /* 根据 kase 的值选择适当的操作 */
    if (kase == 0) break;

    if (kase == kase1) {
        /* 乘以 inv(L) */
        sp_dtrsv("L", "No trans", "Unit", L, U, &work[0], stat, info);

        /* 乘以 inv(U) */
        sp_dtrsv("U", "No trans", "Non-unit", L, U, &work[0], stat, info);
        
    } else {

        /* 乘以 inv(U') */
        sp_dtrsv("U", "Transpose", "Non-unit", L, U, &work[0], stat, info);

        /* 乘以 inv(L') */
        sp_dtrsv("L", "Transpose", "Unit", L, U, &work[0], stat, info);
        
    }

    } while ( kase != 0 );

    /* 计算倒数条件数的估计值 */
    if (ainvnm != 0.) *rcond = (1. / ainvnm) / anorm;

    /* 释放工作空间和整数工作数组 */
    SUPERLU_FREE (work);
    SUPERLU_FREE (iwork);
    return;
} /* dgscon */
```