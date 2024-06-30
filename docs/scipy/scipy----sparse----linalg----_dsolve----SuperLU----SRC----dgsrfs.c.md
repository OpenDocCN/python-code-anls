# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dgsrfs.c`

```
/*
 * File name:    dgsrfs.c
 * History:     Modified from lapack routine DGERFS
 */

#include <math.h>               // 导入数学函数库，用于数学计算
#include "slu_ddefs.h"          // 导入 SuperLU 的定义文件

void
dgsrfs(trans_t trans, SuperMatrix *A, SuperMatrix *L, SuperMatrix *U,
       int *perm_c, int *perm_r, char *equed, double *R, double *C,
       SuperMatrix *B, SuperMatrix *X, double *ferr, double *berr,
       SuperLUStat_t *stat, int *info)
{

#define ITMAX 5                 // 定义常量 ITMAX 为 5

    /* Table of constant values */
    int    ione = 1, nrow = A->nrow;    // 定义整型常量 ione 为 1，nrow 为 A 矩阵的行数
    double ndone = -1.;                 // 定义浮点型常量 ndone 为 -1.0
    double done = 1.;                   // 定义浮点型常量 done 为 1.0
    
    /* Local variables */
    NCformat *Astore;                   // 定义 A 的存储格式为 NCformat 结构体指针
    double   *Aval;                     // 定义 A 的值数组指针
    SuperMatrix Bjcol;                  // 定义 Bjcol 为 SuperMatrix 结构体
    DNformat *Bstore, *Xstore, *Bjcol_store;  // 定义 B、X、Bjcol 的存储格式为 DNformat 结构体指针
    double   *Bmat, *Xmat, *Bptr, *Xptr;  // 定义 Bmat、Xmat、Bptr、Xptr 为双精度浮点型指针
    int      kase;                      // 定义整型变量 kase
    double   safe1, safe2;              // 定义双精度浮点型变量 safe1、safe2
    int      i, j, k, irow, nz, count, notran, rowequ, colequ;  // 定义整型变量 i、j、k 等等
    int      ldb, ldx, nrhs;             // 定义整型变量 ldb、ldx、nrhs
    double   s, xk, lstres, eps, safmin; // 定义双精度浮点型变量 s、xk、lstres、eps、safmin
    char     transc[1];                 // 定义字符数组 transc，长度为 1
    trans_t  transt;                    // 定义 trans_t 类型变量 transt
    double   *work;                     // 定义双精度浮点型指针 work
    double   *rwork;                    // 定义双精度浮点型指针 rwork
    int      *iwork;                    // 定义整型指针 iwork
    int      isave[3];                  // 定义整型数组 isave，长度为 3

    extern int dlacon2_(int *, double *, double *, int *, double *, int *, int []);  // 声明外部函数 dlacon2_

#ifdef _CRAY
    extern int SCOPY(int *, double *, int *, double *, int *);     // 声明外部函数 SCOPY
    extern int SSAXPY(int *, double *, double *, int *, double *, int *);   // 声明外部函数 SSAXPY
#else
    extern int dcopy_(int *, double *, int *, double *, int *);    // 声明外部函数 dcopy_
    extern int daxpy_(int *, double *, double *, int *, double *, int *);  // 声明外部函数 daxpy_
#endif

    Astore = A->Store;                  // 获取 A 矩阵的存储格式
    Aval   = Astore->nzval;             // 获取 A 矩阵的值数组指针
    Bstore = B->Store;                  // 获取 B 矩阵的存储格式
    Xstore = X->Store;                  // 获取 X 矩阵的存储格式
    Bmat   = Bstore->nzval;             // 获取 B 矩阵的值数组指针
    Xmat   = Xstore->nzval;             // 获取 X 矩阵的值数组指针
    ldb    = Bstore->lda;               // 获取 B 矩阵的 leading dimension
    ldx    = Xstore->lda;               // 获取 X 矩阵的 leading dimension
    nrhs   = B->ncol;                   // 获取 B 矩阵的列数
    
    /* Test the input parameters */
    *info = 0;                          // 初始化 info 为 0
    notran = (trans == NOTRANS);         // 检查是否不需要转置
    if ( !notran && trans != TRANS && trans != CONJ ) *info = -1;   // 检查 trans 参数是否合法
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
              A->Stype != SLU_NC || A->Dtype != SLU_D || A->Mtype != SLU_GE )
        *info = -2;                     // 检查 A 矩阵的属性是否合法
    else if ( L->nrow != L->ncol || L->nrow < 0 ||
               L->Stype != SLU_SC || L->Dtype != SLU_D || L->Mtype != SLU_TRLU )
        *info = -3;                     // 检查 L 矩阵的属性是否合法
    else if ( U->nrow != U->ncol || U->nrow < 0 ||
               U->Stype != SLU_NC || U->Dtype != SLU_D || U->Mtype != SLU_TRU )
        *info = -4;                     // 检查 U 矩阵的属性是否合法
    else if ( ldb < SUPERLU_MAX(0, A->nrow) ||
           B->Stype != SLU_DN || B->Dtype != SLU_D || B->Mtype != SLU_GE )
        *info = -10;
    else if ( ldx < SUPERLU_MAX(0, A->nrow) ||
           X->Stype != SLU_DN || X->Dtype != SLU_D || X->Mtype != SLU_GE )
    *info = -11;
    if (*info != 0) {
    i = -(*info);
    input_error("dgsrfs", &i);
    return;
    }


    /* 检查输入参数的有效性，如果不满足条件，则返回相应的错误代码 */
    else if ( ldb < SUPERLU_MAX(0, A->nrow) || 
           B->Stype != SLU_DN || B->Dtype != SLU_D || B->Mtype != SLU_GE )
        *info = -10;
    else if ( ldx < SUPERLU_MAX(0, A->nrow) || 
           X->Stype != SLU_DN || X->Dtype != SLU_D || X->Mtype != SLU_GE )
        *info = -11;
    if (*info != 0) {
        i = -(*info);
        input_error("dgsrfs", &i);
        return;
    }



    /* 如果 A 的行数为 0 或者右侧向量个数 nrhs 为 0，则直接返回 */
    if ( A->nrow == 0 || nrhs == 0) {
    for (j = 0; j < nrhs; ++j) {
        ferr[j] = 0.;
        berr[j] = 0.;
    }
    return;
    }


    /* 如果 A 的行数为 0 或者右侧向量个数 nrhs 为 0，则直接返回零误差 */
    if ( A->nrow == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; ++j) {
            ferr[j] = 0.;
            berr[j] = 0.;
        }
        return;
    }



    rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
    colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;


    /* 检查 equed 字符串以确定行均衡和列均衡的设置 */
    rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
    colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;



    /* 分配工作空间 */
    work = doubleMalloc(2*A->nrow);
    rwork = (double *) SUPERLU_MALLOC( A->nrow * sizeof(double) );
    iwork = int32Malloc(2*A->nrow);
    if ( !work || !rwork || !iwork ) 
        ABORT("Malloc fails for work/rwork/iwork.");


    /* 分配双精度浮点数、双精度浮点数数组和整型数组的工作空间 */
    work = doubleMalloc(2*A->nrow);
    rwork = (double *) SUPERLU_MALLOC( A->nrow * sizeof(double) );
    iwork = int32Malloc(2*A->nrow);
    if ( !work || !rwork || !iwork ) 
        ABORT("Malloc fails for work/rwork/iwork.");



    if ( notran ) {
    *(unsigned char *)transc = 'N';
        transt = TRANS;
    } else if ( trans == TRANS ) {
    *(unsigned char *)transc = 'T';
    transt = NOTRANS;
    } else if ( trans == CONJ ) {
    *(unsigned char *)transc = 'C';
    transt = NOTRANS;
    }    


    /* 根据转置参数设置 trans 和 transc 变量 */
    if ( notran ) {
        *(unsigned char *)transc = 'N';
        transt = TRANS;
    } else if ( trans == TRANS ) {
        *(unsigned char *)transc = 'T';
        transt = NOTRANS;
    } else if ( trans == CONJ ) {
        *(unsigned char *)transc = 'C';
        transt = NOTRANS;
    }



    /* 计算每行（或列）非零元素的数量 */
    for (i = 0; i < A->nrow; ++i) iwork[i] = 0;
    if ( notran ) {
    for (k = 0; k < A->ncol; ++k)
        for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) 
        ++iwork[Astore->rowind[i]];
    } else {
    for (k = 0; k < A->ncol; ++k)
        iwork[k] = Astore->colptr[k+1] - Astore->colptr[k];
    }    


    /* 初始化 iwork 数组为零，然后根据不同的转置模式统计每行（或列）的非零元素个数 */
    for (i = 0; i < A->nrow; ++i) iwork[i] = 0;
    if ( notran ) {
        for (k = 0; k < A->ncol; ++k)
            for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) 
                ++iwork[Astore->rowind[i]];
    } else {
        for (k = 0; k < A->ncol; ++k)
            iwork[k] = Astore->colptr[k+1] - Astore->colptr[k];
    }



    /* 复制 RHS B 的一列到 Bjcol */
    Bjcol.Stype = B->Stype;
    Bjcol.Dtype = B->Dtype;
    Bjcol.Mtype = B->Mtype;
    Bjcol.nrow  = B->nrow;
    Bjcol.ncol  = 1;
    Bjcol.Store = (void *) SUPERLU_MALLOC( sizeof(DNformat) );
    if ( !Bjcol.Store ) ABORT("SUPERLU_MALLOC fails for Bjcol.Store");
    Bjcol_store = Bjcol.Store;
    Bjcol_store->lda = ldb;
    Bjcol_store->nzval = work; /* address aliasing */


    /* 初始化 Bjcol 结构体以保存 B 的列向量 */
    Bjcol.Stype = B->Stype;
    Bjcol.Dtype = B->Dtype;
    Bjcol.Mtype = B->Mtype;
    Bjcol.nrow  = B->nrow;
    Bjcol.ncol  = 1;
    Bjcol.Store = (void *) SUPERLU_MALLOC( sizeof(DNformat) );
    if ( !Bjcol.Store ) ABORT("SUPERLU_MALLOC fails for Bjcol.Store");
    Bjcol_store = Bjcol.Store;
    Bjcol_store->lda = ldb;
    Bjcol_store->nzval = work; /* 使用 work 数组作为 Bjcol 的 nzval 字段 */



    /* 对每个右侧向量进行处理 */
    for (j = 0; j < nrhs; ++j) {
    count = 0;
    lstres = 3.;
    Bptr = &Bmat[j*ldb];
    Xptr = &Xmat[j*ldx];

    while (1) { /* Loop until stopping criterion is satisfied. */


    /* 对每个右侧向量进行处理，使用迭代方法直到满足停止条件 */
    for (j = 0; j < nrhs; ++j) {
        count = 0;
        lstres = 3.;
        Bptr = &Bmat[j*ldb];
        Xptr = &Xmat[j*ldx];

        while (1) { /* 循环直到满足停止条件 */



        /* 计算残差 R = B - op(A) * X，
           其中 op(A) = A, A**T, 或 A**H，取决于 TRANS 参数。 */


        /* 计算残差 R = B - op(A) * X，
           其中 op(A) = A, A**T, 或 A**H，取决于 TRANS 参数。 */
#ifdef _CRAY
        SCOPY(&nrow, Bptr, &ione, work, &ione);
#else
        dcopy_(&nrow, Bptr, &ione, work, &ione);
#endif
        // 根据宏定义选择不同的函数复制向量Bptr到work中，nrow是向量的长度
        sp_dgemv(transc, ndone, A, Xptr, ione, done, work, ione);

        /* 计算相对后向误差，使用公式
           max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
           其中 abs(Z) 表示矩阵或向量 Z 的逐元素绝对值。
           如果分母的第i个分量小于 SAFE2，则在除法前给分子的第i个分量加上 SAFE1。 */
        
        for (i = 0; i < A->nrow; ++i) rwork[i] = fabs( Bptr[i] );
        
        /* 计算 abs(op(A))*abs(X) + abs(B) */
        if ( notran ) {
            for (k = 0; k < A->ncol; ++k) {
                xk = fabs( Xptr[k] );
                for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i)
                    rwork[Astore->rowind[i]] += fabs(Aval[i]) * xk;
            }
        } else {  /* trans = TRANS or CONJ */
            for (k = 0; k < A->ncol; ++k) {
                s = 0.;
                for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) {
                    irow = Astore->rowind[i];
                    s += fabs(Aval[i]) * fabs(Xptr[irow]);
                }
                rwork[k] += s;
            }
        }
        s = 0.;
        for (i = 0; i < A->nrow; ++i) {
            if (rwork[i] > safe2) {
                s = SUPERLU_MAX( s, fabs(work[i]) / rwork[i] );
            } else if ( rwork[i] != 0.0 ) {
                /* 如果 rwork[i] 不为 0.0，则给分子加上 SAFE1 防止出现虚假的零残差（下溢）。 */
                s = SUPERLU_MAX( s, (safe1 + fabs(work[i])) / rwork[i] );
            }
            /* 如果 rwork[i] 确实为 0.0，则真实的残差也必须是确实的 0.0。 */
        }
        berr[j] = s;

        /* 测试停止条件。如果满足以下条件，则继续迭代：
           1) 残差 BERR(J) 大于机器精度 eps，
           2) BERR(J) 上一次迭代减少了至少一半，
           3) 迭代次数不超过 ITMAX。 */
        
        if (berr[j] > eps && berr[j] * 2. <= lstres && count < ITMAX) {
            /* 更新解并再次尝试。 */
            dgstrs (trans, L, U, perm_c, perm_r, &Bjcol, stat, info);
            
#ifdef _CRAY
            SAXPY(&nrow, &done, work, &ione,
                  &Xmat[j*ldx], &ione);
#else
            daxpy_(&nrow, &done, work, &ione,
                  &Xmat[j*ldx], &ione);
#endif
            lstres = berr[j];
            ++count;
        } else {
            break;
        }
        
    } /* end while */

    stat->RefineSteps = count;
    /* 计算绝对误差上界 */
    for (i = 0; i < A->nrow; ++i) rwork[i] = fabs( Bptr[i] );
    
    /* 计算 abs(op(A))*abs(X) + abs(B) */
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
    
    /* 更新 rwork，根据条件判断是否增加 SAFE1 */
    for (i = 0; i < A->nrow; ++i) {
        if (rwork[i] > safe2)
            rwork[i] = fabs(work[i]) + (iwork[i]+1)*eps*rwork[i];
        else
            rwork[i] = fabs(work[i]) + (iwork[i]+1)*eps*rwork[i] + safe1;
    }

    kase = 0;

    /* 使用 dlacon2 估计矩阵的无穷范数 */
    do {
        dlacon2_(&nrow, &work[A->nrow], work,
            &iwork[A->nrow], &ferr[j], &kase, isave);
        if (kase == 0) break;

        /* 根据 kase 值进行不同的操作 */
        if (kase == 1) {
            /* 乘以 diag(W)*inv(op(A)**T)*(diag(C) or diag(R)) */
            if ( notran && colequ )
                for (i = 0; i < A->ncol; ++i) work[i] *= C[i];
            else if ( !notran && rowequ )
                for (i = 0; i < A->nrow; ++i) work[i] *= R[i];
        
            dgstrs (transt, L, U, perm_c, perm_r, &Bjcol, stat, info);
        
            for (i = 0; i < A->nrow; ++i) work[i] *= rwork[i];
        } else {
            /* 乘以 (diag(C) or diag(R))*inv(op(A))*diag(W) */
            for (i = 0; i < A->nrow; ++i) work[i] *= rwork[i];
        
            dgstrs (trans, L, U, perm_c, perm_r, &Bjcol, stat, info);
        
            if ( notran && colequ )
                for (i = 0; i < A->ncol; ++i) work[i] *= C[i];
            else if ( !notran && rowequ )
                for (i = 0; i < A->ncol; ++i) work[i] *= R[i];  
        }
        
    } while ( kase != 0 );

    /* 归一化误差 */
    lstres = 0.;
    // 初始化残差的最大值为0

    if ( notran && colequ ) {
        // 如果未转置且列等价性成立
        for (i = 0; i < A->nrow; ++i)
            // 遍历矩阵A的行数次循环
            lstres = SUPERLU_MAX( lstres, C[i] * fabs( Xptr[i]) );
            // 更新残差的最大值为当前值与C[i]乘以Xptr[i]绝对值的较大值
    } else if ( !notran && rowequ ) {
        // 否则，如果转置且行等价性成立
        for (i = 0; i < A->nrow; ++i)
            // 遍历矩阵A的行数次循环
            lstres = SUPERLU_MAX( lstres, R[i] * fabs( Xptr[i]) );
            // 更新残差的最大值为当前值与R[i]乘以Xptr[i]绝对值的较大值
    } else {
        // 否则
        for (i = 0; i < A->nrow; ++i)
            // 遍历矩阵A的行数次循环
            lstres = SUPERLU_MAX( lstres, fabs( Xptr[i]) );
            // 更新残差的最大值为当前值与Xptr[i]绝对值的较大值
    }
    if ( lstres != 0. )
        // 如果最大残差不为0
        ferr[j] /= lstres;
        // 对于第j个右手边向量，除以最大残差值

    } /* for each RHS j ... */
    // 结束对每个右手边向量j的循环

    SUPERLU_FREE(work);
    // 释放工作区域的内存
    SUPERLU_FREE(rwork);
    // 释放实数工作区域的内存
    SUPERLU_FREE(iwork);
    // 释放整数工作区域的内存
    SUPERLU_FREE(Bjcol.Store);
    // 释放Bjcol.Store的内存空间

    return;
    // 返回
} /* dgsrfs */


注释：

# 这行代码是一个注释，使用大括号和星号包围的文本
# 在程序中通常用来注释掉或解释代码块的作用
```