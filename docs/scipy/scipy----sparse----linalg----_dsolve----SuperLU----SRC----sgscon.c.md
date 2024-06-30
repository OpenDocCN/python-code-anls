# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sgscon.c`

```
/*
 * File name:    sgscon.c
 * History:     Modified from lapack routines SGECON.
 */
#include <math.h>
#include "slu_sdefs.h"

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   SGSCON estimates the reciprocal of the condition number of a general 
 *   real matrix A, in either the 1-norm or the infinity-norm, using   
 *   the LU factorization computed by SGETRF.   *
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
 *            sgstrf(). Use compressed row subscripts storage for supernodes,
 *            i.e., L has types: Stype = SLU_SC, Dtype = SLU_S, Mtype = SLU_TRLU.
 * 
 *    U       (input) SuperMatrix*
 *            The factor U from the factorization Pr*A*Pc=L*U as computed by
 *            sgstrf(). Use column-wise storage scheme, i.e., U has types:
 *            Stype = SLU_NC, Dtype = SLU_S, Mtype = SLU_TRU.
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
sgscon(char *norm, SuperMatrix *L, SuperMatrix *U,
       float anorm, float *rcond, SuperLUStat_t *stat, int *info)
{
    /* Local variables */
    int    kase, kase1, onenrm;
    float ainvnm;
    float *work;
    int    *iwork;
    int    isave[3];

    // 开始 SGSCON 函数，用于估算矩阵 A 的条件数的倒数

    // 初始化工作空间和整型工作数组
    work = NULL;
    iwork = NULL;

    // 获取一范数或无穷范数的标志
    onenrm = lsame_(norm, "1") || lsame_(norm, "O");

    // 如果要计算矩阵 A 的一范数条件数
    if (onenrm) {
        // 使用 LAPACK 函数 slacon_ 估算矩阵 A 的条件数倒数
        slacon_(norm, &L->ncol, &U->ncol, &anorm, rcond, &kase, work, iwork, info);
    } else {  // 否则，计算矩阵 A 的无穷范数条件数
        // 使用 LAPACK 函数 slacon_ 估算矩阵 A 的条件数倒数
        slacon_(norm, &L->ncol, &U->ncol, &anorm, rcond, &kase, work, iwork, info);
    }

    // 释放工作空间
    SUPERLU_FREE(work);
    SUPERLU_FREE(iwork);
}
    extern int srscl_(int *, float *, float *, int *);

    extern int slacon2_(int *, float *, float *, int *, float *, int *, int []);
    
    /* 测试输入参数的有效性 */
    *info = 0;  // 初始化 info 为 0
    onenrm = *(unsigned char *)norm == '1' || strncmp(norm, "O", 1)==0;  // 判断 norm 是否为 '1' 或者以 "O" 开头
    if (! onenrm && strncmp(norm, "I", 1)!=0) *info = -1;  // 如果 norm 不是 '1' 且不以 "I" 开头，则将 info 设为 -1
    else if (L->nrow < 0 || L->nrow != L->ncol ||
             L->Stype != SLU_SC || L->Dtype != SLU_S || L->Mtype != SLU_TRLU)
     *info = -2;  // 如果 L 不是方阵，或者类型不符合预期，则将 info 设为 -2
    else if (U->nrow < 0 || U->nrow != U->ncol ||
             U->Stype != SLU_NC || U->Dtype != SLU_S || U->Mtype != SLU_TRU) 
    *info = -3;  // 如果 U 不是方阵，或者类型不符合预期，则将 info 设为 -3
    if (*info != 0) {
    int ii = -(*info);
    input_error("sgscon", &ii);  // 调用错误处理函数 input_error，并传递错误码的负值
    return;  // 返回，终止函数执行
    }

    /* 如果可能，进行快速返回 */
    *rcond = 0.;  // 初始化 rcond 为 0
    if ( L->nrow == 0 || U->nrow == 0) {
    *rcond = 1.;  // 如果 L 或 U 的维度为 0，则将 rcond 设为 1，表示条件数的倒数为 1
    return;  // 返回，终止函数执行
    }

    work = floatCalloc( 3*L->nrow );  // 分配存储空间给 work 数组
    iwork = int32Malloc( L->nrow );  // 分配存储空间给 iwork 数组

    if ( !work || !iwork )
    ABORT("Malloc fails for work arrays in sgscon.");  // 如果分配失败，则调用 ABORT 函数终止程序

    /* 估算 inv(A) 的范数 */
    ainvnm = 0.;  // 初始化 ainvnm 为 0
    if ( onenrm ) kase1 = 1;  // 如果使用的是 1-范数，则设置 kase1 为 1
    else kase1 = 2;  // 否则设置 kase1 为 2
    kase = 0;  // 初始化 kase 为 0

    int nrow = L->nrow;  // 将 L 的行数赋给 nrow

    do {
    slacon2_(&nrow, &work[L->nrow], &work[0], &iwork[0], &ainvnm, &kase, isave);  // 调用 slacon2_ 函数进行迭代计算

    if (kase == 0) break;  // 如果 kase 为 0，则跳出循环

    if (kase == kase1) {
        /* 乘以 inv(L) */
        sp_strsv("L", "No trans", "Unit", L, U, &work[0], stat, info);  // 调用 sp_strsv 函数，乘以 inv(L)

        /* 乘以 inv(U) */
        sp_strsv("U", "No trans", "Non-unit", L, U, &work[0], stat, info);  // 调用 sp_strsv 函数，乘以 inv(U)
        
    } else {

        /* 乘以 inv(U') */
        sp_strsv("U", "Transpose", "Non-unit", L, U, &work[0], stat, info);  // 调用 sp_strsv 函数，乘以 inv(U')

        /* 乘以 inv(L') */
        sp_strsv("L", "Transpose", "Unit", L, U, &work[0], stat, info);  // 调用 sp_strsv 函数，乘以 inv(L')
        
    }

    } while ( kase != 0 );  // 循环直到 kase 变为 0

    /* 计算条件数的估计值 */
    if (ainvnm != 0.) *rcond = (1. / ainvnm) / anorm;  // 如果 ainvnm 不为 0，则计算条件数的估计值

    SUPERLU_FREE (work);  // 释放 work 数组的内存
    SUPERLU_FREE (iwork);  // 释放 iwork 数组的内存
    return;  // 返回，结束函数执行
} /* sgscon */
```