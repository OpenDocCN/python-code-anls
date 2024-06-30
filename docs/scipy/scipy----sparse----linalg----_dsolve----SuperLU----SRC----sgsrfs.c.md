# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sgsrfs.c`

```
/*
 * File name:    sgsrfs.c
 * History:     Modified from lapack routine SGERFS
 */

#include <math.h>               // 包含数学函数库，例如 fabs 等
#include "slu_sdefs.h"          // 包含定义了 SuperLU 结构和函数的头文件

void
sgsrfs(trans_t trans, SuperMatrix *A, SuperMatrix *L, SuperMatrix *U,
       int *perm_c, int *perm_r, char *equed, float *R, float *C,
       SuperMatrix *B, SuperMatrix *X, float *ferr, float *berr,
       SuperLUStat_t *stat, int *info)
{
#define ITMAX 5                 // 定义常量 ITMAX 为 5

    /* Table of constant values */
    int    ione = 1, nrow = A->nrow;    // 定义整数常量 ione 和 nrow
    float ndone = -1.;                  // 定义浮点数常量 ndone
    float done = 1.;                    // 定义浮点数常量 done
    
    /* Local variables */
    NCformat *Astore;                   // 定义指向 A 矩阵的 NCformat 结构指针
    float   *Aval;                      // 定义指向 A 矩阵数值部分的浮点数指针
    SuperMatrix Bjcol;                  // 定义超级矩阵 Bjcol
    DNformat *Bstore, *Xstore, *Bjcol_store;  // 定义指向 B, X, Bjcol 的 DNformat 结构指针
    float   *Bmat, *Xmat, *Bptr, *Xptr;  // 定义指向 Bmat, Xmat, Bptr, Xptr 的浮点数指针
    int      kase;                      // 定义整数变量 kase
    float   safe1, safe2;               // 定义安全常量 safe1 和 safe2
    int      i, j, k, irow, nz, count, notran, rowequ, colequ;  // 定义整数变量
    int      ldb, ldx, nrhs;             // 定义整数变量 ldb, ldx, nrhs
    float   s, xk, lstres, eps, safmin;  // 定义浮点数变量 s, xk, lstres, eps, safmin
    char     transc[1];                 // 定义字符数组 transc
    trans_t  transt;                    // 定义 trans_t 类型变量 transt
    float   *work;                      // 定义指向 work 的浮点数指针
    float   *rwork;                     // 定义指向 rwork 的浮点数指针
    int      *iwork;                     // 定义指向 iwork 的整数指针
    int      isave[3];                   // 定义整数数组 isave

    extern int slacon2_(int *, float *, float *, int *, float *, int *, int []);  // 声明外部函数 slacon2_
#ifdef _CRAY
    extern int SCOPY(int *, float *, int *, float *, int *);  // 声明外部函数 SCOPY
    extern int SSAXPY(int *, float *, float *, int *, float *, int *);  // 声明外部函数 SSAXPY
#else
    extern int scopy_(int *, float *, int *, float *, int *);  // 声明外部函数 scopy_
    extern int saxpy_(int *, float *, float *, int *, float *, int *);  // 声明外部函数 saxpy_
#endif

    Astore = A->Store;                  // 获取 A 矩阵的存储格式
    Aval   = Astore->nzval;             // 获取 A 矩阵的非零值数组
    Bstore = B->Store;                  // 获取 B 矩阵的存储格式
    Xstore = X->Store;                  // 获取 X 矩阵的存储格式
    Bmat   = Bstore->nzval;             // 获取 B 矩阵的非零值数组
    Xmat   = Xstore->nzval;             // 获取 X 矩阵的非零值数组
    ldb    = Bstore->lda;               // 获取 B 矩阵的 leading dimension
    ldx    = Xstore->lda;               // 获取 X 矩阵的 leading dimension
    nrhs   = B->ncol;                   // 获取 B 矩阵的列数
    
    /* Test the input parameters */
    *info = 0;                          // 初始化 info 为 0
    notran = (trans == NOTRANS);         // 判断是否为非转置
    if ( !notran && trans != TRANS && trans != CONJ ) *info = -1;  // 检查 trans 参数是否合法
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
          A->Stype != SLU_NC || A->Dtype != SLU_S || A->Mtype != SLU_GE )
    *info = -2;                         // 检查 A 矩阵的参数是否合法
    else if ( L->nrow != L->ncol || L->nrow < 0 ||
           L->Stype != SLU_SC || L->Dtype != SLU_S || L->Mtype != SLU_TRLU )
    *info = -3;                         // 检查 L 矩阵的参数是否合法
    else if ( U->nrow != U->ncol || U->nrow < 0 ||
           U->Stype != SLU_NC || U->Dtype != SLU_S || U->Mtype != SLU_TRU )
    *info = -4;                         // 检查 U 矩阵的参数是否合法
    else if ( ldb < SUPERLU_MAX(0, A->nrow) ||
           B->Stype != SLU_DN || B->Dtype != SLU_S || B->Mtype != SLU_GE )
        *info = -10;
    else if ( ldx < SUPERLU_MAX(0, A->nrow) ||
           X->Stype != SLU_DN || X->Dtype != SLU_S || X->Mtype != SLU_GE )
    *info = -11;
    if (*info != 0) {
    i = -(*info);
    input_error("sgsrfs", &i);
    return;
    }


# 检查输入参数的有效性，包括矩阵和向量的属性
else if ( ldb < SUPERLU_MAX(0, A->nrow) ||    # 如果 ldb 小于 0 或者超出 A 的行数，则设定错误码为 -10
       B->Stype != SLU_DN || B->Dtype != SLU_S || B->Mtype != SLU_GE )
    *info = -10;
else if ( ldx < SUPERLU_MAX(0, A->nrow) ||    # 如果 ldx 小于 0 或者超出 A 的行数，则设定错误码为 -11
       X->Stype != SLU_DN || X->Dtype != SLU_S || X->Mtype != SLU_GE )
*info = -11;
if (*info != 0) {    # 如果错误码不为 0，则输出错误消息并返回
i = -(*info);
input_error("sgsrfs", &i);
return;
}
#ifdef _CRAY
        SCOPY(&nrow, Bptr, &ione, work, &ione);
#else
        scopy_(&nrow, Bptr, &ione, work, &ione);
#endif
        // 使用 SCOPY 或 scopy_ 复制向量 Bptr 到 work 中，nrow 是向量的长度

        sp_sgemv(transc, ndone, A, Xptr, ione, done, work, ione);
        // 调用 sp_sgemv 函数进行向量乘法操作：work = op(A) * Xptr

        /* Compute componentwise relative backward error from formula 
           max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )   
           where abs(Z) is the componentwise absolute value of the matrix
           or vector Z.  If the i-th component of the denominator is less
           than SAFE2, then SAFE1 is added to the i-th component of the   
           numerator before dividing. */
        // 计算组件相对后向误差，根据公式计算每个分量的误差

        for (i = 0; i < A->nrow; ++i) rwork[i] = fabs( Bptr[i] );
        // 初始化 rwork 数组，存储向量 Bptr 的绝对值

        /* Compute abs(op(A))*abs(X) + abs(B). */
        if ( notran ) {
            // 若不需要转置操作
            for (k = 0; k < A->ncol; ++k) {
                xk = fabs( Xptr[k] );
                for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i)
                    rwork[Astore->rowind[i]] += fabs(Aval[i]) * xk;
                // 计算 abs(op(A))*abs(X) + abs(B) 中的每个分量
            }
        } else {  /* trans = TRANS or CONJ */
            // 若需要转置操作
            for (k = 0; k < A->ncol; ++k) {
                s = 0.;
                for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) {
                    irow = Astore->rowind[i];
                    s += fabs(Aval[i]) * fabs(Xptr[irow]);
                }
                rwork[k] += s;
                // 计算转置操作后的 abs(op(A))*abs(X) + abs(B) 中的每个分量
            }
        }

        s = 0.;
        // 初始化 s，用于存储最大的相对误差

        for (i = 0; i < A->nrow; ++i) {
            if (rwork[i] > safe2) {
                s = SUPERLU_MAX( s, fabs(work[i]) / rwork[i] );
            } else if ( rwork[i] != 0.0 ) {
                /* Adding SAFE1 to the numerator guards against
                   spuriously zero residuals (underflow). */
                s = SUPERLU_MAX( s, (safe1 + fabs(work[i])) / rwork[i] );
            }
            /* If rwork[i] is exactly 0.0, then we know the true 
               residual also must be exactly 0.0. */
            // 处理特殊情况，如果 rwork[i] 精确为 0.0，则相对误差也为 0.0
        }

        berr[j] = s;
        // 存储当前迭代的相对误差到数组 berr 的第 j 个位置

        /* Test stopping criterion. Continue iterating if   
           1) The residual BERR(J) is larger than machine epsilon, and   
           2) BERR(J) decreased by at least a factor of 2 during the   
              last iteration, and   
           3) At most ITMAX iterations tried. */
        // 检测停止条件：如果当前的误差大于机器 epsilon，并且上次迭代的误差至少减少了一半，并且迭代次数未超过 ITMAX

        if (berr[j] > eps && berr[j] * 2. <= lstres && count < ITMAX) {
            /* Update solution and try again. */
            sgstrs (trans, L, U, perm_c, perm_r, &Bjcol, stat, info);
            // 调用 sgstrs 函数更新解向量

#ifdef _CRAY
            SAXPY(&nrow, &done, work, &ione,
                   &Xmat[j*ldx], &ione);
#else
            saxpy_(&nrow, &done, work, &ione,
                   &Xmat[j*ldx], &ione);
#endif
            // 使用 SAXPY 或 saxpy_ 更新解向量 Xmat[j*ldx]

            lstres = berr[j];
            // 更新上一次的误差为当前误差
            ++count;
            // 增加迭代计数器
        } else {
            break;
            // 如果不满足继续迭代的条件，则退出循环
        }
        
    } /* end while */

    stat->RefineSteps = count;
    // 将迭代次数存储到 stat 结构体中
    /* 计算公式导致的边界错误：
       norm(X - XTRUE) / norm(X) .le. FERR = norm( abs(inv(op(A)))*   
       ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)   
          其中   
            norm(Z) 是 Z 中最大分量的幅度   
            inv(op(A)) 是 op(A) 的逆矩阵   
            abs(Z) 是矩阵或向量 Z 按分量取绝对值   
            NZ 是矩阵 A 中任何行的最大非零元素个数加一   
            EPS 是机器精度   

          abs(R)+NZ*EPS*(abs(op(A))*abs(X)+abs(B)) 的第 i 个分量   
          如果 abs(op(A))*abs(X) + abs(B) 的第 i 个分量小于 SAFE2，就增加 SAFE1。   

          使用 SLACON2 估计矩阵 inv(op(A)) * diag(W) 的无穷范数，   
          其中 W = abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) */
    
    for (i = 0; i < A->nrow; ++i) rwork[i] = fabs( Bptr[i] );
    /* 将 rwork 的每个元素初始化为 Bptr 对应元素的绝对值 */

    /* 计算 abs(op(A))*abs(X) + abs(B). */
    if ( notran ) {
        for (k = 0; k < A->ncol; ++k) {
        xk = fabs( Xptr[k] );
        for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i)
            rwork[Astore->rowind[i]] += fabs(Aval[i]) * xk;
        }
    } else {  /* trans == TRANS or CONJ */
        for (k = 0; k < A->ncol; ++k) {
        s = 0.;
        for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) {
            irow = Astore->rowind[i];
            xk = fabs( Xptr[irow] );
            s += fabs(Aval[i]) * xk;
        }
        rwork[k] += s;
        }
    }
    
    for (i = 0; i < A->nrow; ++i)
        if (rwork[i] > safe2)
        rwork[i] = fabs(work[i]) + (iwork[i]+1)*eps*rwork[i];
        else
        rwork[i] = fabs(work[i])+(iwork[i]+1)*eps*rwork[i]+safe1;

    kase = 0;

    do {
        slacon2_(&nrow, &work[A->nrow], work,
            &iwork[A->nrow], &ferr[j], &kase, isave);
        if (kase == 0) break;

        if (kase == 1) {
        /* 乘以 diag(W)*inv(op(A)**T)*(diag(C) 或 diag(R)). */
        if ( notran && colequ )
            for (i = 0; i < A->ncol; ++i) work[i] *= C[i];
        else if ( !notran && rowequ )
            for (i = 0; i < A->nrow; ++i) work[i] *= R[i];
        
        sgstrs (transt, L, U, perm_c, perm_r, &Bjcol, stat, info);
        
        for (i = 0; i < A->nrow; ++i) work[i] *= rwork[i];
        } else {
        /* 乘以 (diag(C) 或 diag(R))*inv(op(A))*diag(W). */
        for (i = 0; i < A->nrow; ++i) work[i] *= rwork[i];
        
        sgstrs (trans, L, U, perm_c, perm_r, &Bjcol, stat, info);
        
        if ( notran && colequ )
            for (i = 0; i < A->ncol; ++i) work[i] *= C[i];
        else if ( !notran && rowequ )
            for (i = 0; i < A->ncol; ++i) work[i] *= R[i];  
        }
        
    } while ( kase != 0 );


    /* 规范化误差。 */
    lstres = 0.;
    // 初始化 lstres 变量为 0

     if ( notran && colequ ) {
        // 如果不是转置操作且列等价标志为真，则执行以下操作
        for (i = 0; i < A->nrow; ++i)
            // 循环遍历 A 矩阵的行数次
            lstres = SUPERLU_MAX( lstres, C[i] * fabs( Xptr[i]) );
            // 更新 lstres 为当前 lstres 与 C[i] * |Xptr[i]| 的最大值
      } else if ( !notran && rowequ ) {
        // 否则，如果是转置操作且行等价标志为真，则执行以下操作
        for (i = 0; i < A->nrow; ++i)
            // 循环遍历 A 矩阵的行数次
            lstres = SUPERLU_MAX( lstres, R[i] * fabs( Xptr[i]) );
            // 更新 lstres 为当前 lstres 与 R[i] * |Xptr[i]| 的最大值
    } else {
        // 否则执行以下操作（即不是转置操作且列等价标志为假，或者是转置操作但行等价标志为假）
        for (i = 0; i < A->nrow; ++i)
            // 循环遍历 A 矩阵的行数次
            lstres = SUPERLU_MAX( lstres, fabs( Xptr[i]) );
            // 更新 lstres 为当前 lstres 与 |Xptr[i]| 的最大值
    }
    // 如果 lstres 不为 0，则执行以下操作
    if ( lstres != 0. )
        ferr[j] /= lstres;
        // 将 ferr[j] 除以 lstres

    } /* for each RHS j ... */
    // 结束对每个 RHS j 的循环

    SUPERLU_FREE(work);
    // 释放工作空间 work
    SUPERLU_FREE(rwork);
    // 释放实数工作空间 rwork
    SUPERLU_FREE(iwork);
    // 释放整数工作空间 iwork
    SUPERLU_FREE(Bjcol.Store);
    // 释放 Bjcol.Store 所占用的空间

    return;
    // 返回
} /* sgsrfs */
```