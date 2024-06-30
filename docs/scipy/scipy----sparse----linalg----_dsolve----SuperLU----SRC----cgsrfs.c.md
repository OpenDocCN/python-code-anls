# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cgsrfs.c`

```
    /* 定义最大迭代次数 */
#define ITMAX 5
    
    /* 常量表 */
    int    ione = 1, nrow = A->nrow;
    singlecomplex ndone = {-1., 0.};
    singlecomplex done = {1., 0.};
    
    /* 本地变量 */
    NCformat *Astore;
    singlecomplex   *Aval;
    SuperMatrix Bjcol;
    DNformat *Bstore, *Xstore, *Bjcol_store;
    singlecomplex   *Bmat, *Xmat, *Bptr, *Xptr;
    int      kase;
    float   safe1, safe2;
    int      i, j, k, irow, nz, count, notran, rowequ, colequ;
    int      ldb, ldx, nrhs;
    float   s, xk, lstres, eps, safmin;
    char     transc[1];
    trans_t  transt;
    singlecomplex   *work;
    float   *rwork;
    int      *iwork;
    int      isave[3];

    extern int clacon2_(int *, singlecomplex *, singlecomplex *, float *, int *, int []);
#ifdef _CRAY
    extern int CCOPY(int *, singlecomplex *, int *, singlecomplex *, int *);
    extern int CSAXPY(int *, singlecomplex *, singlecomplex *, int *, singlecomplex *, int *);
#else
    extern int ccopy_(int *, singlecomplex *, int *, singlecomplex *, int *);
    extern int caxpy_(int *, singlecomplex *, singlecomplex *, int *, singlecomplex *, int *);
#endif

    /* 获取输入参数 */
    Astore = A->Store;  // 获取矩阵 A 的存储结构
    Aval   = Astore->nzval;  // 获取矩阵 A 的非零元素数组
    Bstore = B->Store;  // 获取矩阵 B 的存储结构
    Xstore = X->Store;  // 获取矩阵 X 的存储结构
    Bmat   = Bstore->nzval;  // 获取矩阵 B 的非零元素数组
    Xmat   = Xstore->nzval;  // 获取矩阵 X 的非零元素数组
    ldb    = Bstore->lda;  // 获取矩阵 B 的列数
    ldx    = Xstore->lda;  // 获取矩阵 X 的列数
    nrhs   = B->ncol;  // 获取矩阵 B 的列数
    
    /* 检查输入参数 */
    *info = 0;  // 将 info 初始化为 0
    notran = (trans == NOTRANS);  // 检查是否为无转置操作
    if ( !notran && trans != TRANS && trans != CONJ ) *info = -1;  // 检查转置类型的合法性
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
          A->Stype != SLU_NC || A->Dtype != SLU_C || A->Mtype != SLU_GE )
    *info = -2;  // 检查矩阵 A 的属性和尺寸
    else if ( L->nrow != L->ncol || L->nrow < 0 ||
           L->Stype != SLU_SC || L->Dtype != SLU_C || L->Mtype != SLU_TRLU )
    *info = -3;  // 检查矩阵 L 的属性和尺寸
    else if ( U->nrow != U->ncol || U->nrow < 0 ||
           U->Stype != SLU_NC || U->Dtype != SLU_C || U->Mtype != SLU_TRU )
    *info = -4;  // 检查矩阵 U 的属性和尺寸
    *info = -4;  # 设置错误信息为-4（这里缺少条件判断语句开头的if/else）
    else if ( ldb < SUPERLU_MAX(0, A->nrow) ||  # 如果 ldb 小于等于 0 或者 B 不是稠密存储类型或者数据类型不是双精度复数或者矩阵类型不是一般矩阵，则设置错误信息为-10
           B->Stype != SLU_DN || B->Dtype != SLU_C || B->Mtype != SLU_GE )
        *info = -10;
    else if ( ldx < SUPERLU_MAX(0, A->nrow) ||  # 如果 ldx 小于等于 0 或者 X 不是稠密存储类型或者数据类型不是双精度复数或者矩阵类型不是一般矩阵，则设置错误信息为-11
           X->Stype != SLU_DN || X->Dtype != SLU_C || X->Mtype != SLU_GE )
    *info = -11;
    if (*info != 0) {  # 如果错误信息不为0，则执行下面的错误处理和返回
    i = -(*info);  # 取错误信息的负值
    input_error("cgsrfs", &i);  # 调用输入错误处理函数
    return;  # 直接返回
    }

    /* Quick return if possible */  # 如果可能的话，快速返回
    if ( A->nrow == 0 || nrhs == 0) {  # 如果 A 的行数为0或者右手边的数量为0
    for (j = 0; j < nrhs; ++j) {  # 循环处理每个右手边
        ferr[j] = 0.;  # 设置 ferr[j] 为0
        berr[j] = 0.;  # 设置 berr[j] 为0
    }
    return;  # 直接返回
    }

    rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;  # 检查是否对行进行了等价化
    colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;  # 检查是否对列进行了等价化
    
    /* Allocate working space */  # 分配工作空间
    work = complexMalloc(2*A->nrow);  # 分配复杂数据类型的工作空间
    rwork = (float *) SUPERLU_MALLOC( A->nrow * sizeof(float) );  # 分配浮点型数组的工作空间
    iwork = int32Malloc(A->nrow);  # 分配32位整数数组的工作空间
    if ( !work || !rwork || !iwork )   # 如果任意一个工作空间分配失败
        ABORT("Malloc fails for work/rwork/iwork.");  # 输出错误信息并中止程序
    
    if ( notran ) {  # 如果没有转置
    *(unsigned char *)transc = 'N';  # 将 trans 转换为字符 'N'
        transt = TRANS;  # 设置 transt 为 TRANS
    } else if ( trans == TRANS ) {  # 如果转置类型是 TRANS
    *(unsigned char *)transc = 'T';  # 将 trans 转换为字符 'T'
    transt = NOTRANS;  # 设置 transt 为 NOTRANS
    } else if ( trans == CONJ ) {  # 如果转置类型是 CONJ
    *(unsigned char *)transc = 'C';  # 将 trans 转换为字符 'C'
    transt = NOTRANS;  # 设置 transt 为 NOTRANS
    }    

    /* NZ = maximum number of nonzero elements in each row of A, plus 1 */  # NZ = A 的每行中的最大非零元素数目加1
    nz     = A->ncol + 1;  # 计算 nz
    
    eps    = smach("Epsilon");  # 获取机器精度
    safmin = smach("Safe minimum");  # 获取安全最小值

    /* Set SAFE1 essentially to be the underflow threshold times the
       number of additions in each row. */  # 将 SAFE1 设置为下溢阈值乘以每行中的加法次数
    safe1  = nz * safmin;  # 计算 SAFE1
    safe2  = safe1 / eps;  # 计算 SAFE2

    /* Compute the number of nonzeros in each row (or column) of A */  # 计算 A 的每行（或列）中的非零元素数目
    for (i = 0; i < A->nrow; ++i) iwork[i] = 0;  # 初始化 iwork 数组为0
    if ( notran ) {  # 如果没有转置
    for (k = 0; k < A->ncol; ++k)  # 遍历 A 的每一列
        for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i)   # 遍历列 k 中的每个非零元素
        ++iwork[Astore->rowind[i]];  # 统计每行中的非零元素个数
    } else {  # 如果有转置
    for (k = 0; k < A->ncol; ++k)  # 遍历 A 的每一列
        iwork[k] = Astore->colptr[k+1] - Astore->colptr[k];  # 计算每列中的非零元素个数
    }    

    /* Copy one column of RHS B into Bjcol. */  # 将右手边矩阵 B 的一列复制到 Bjcol 中
    Bjcol.Stype = B->Stype;  # 设置 Bjcol 的存储类型与 B 相同
    Bjcol.Dtype = B->Dtype;  # 设置 Bjcol 的数据类型与 B 相同
    Bjcol.Mtype = B->Mtype;  # 设置 Bjcol 的矩阵类型与 B 相同
    Bjcol.nrow  = B->nrow;   # 设置 Bjcol 的行数与 B 相同
    Bjcol.ncol  = 1;         # 设置 Bjcol 的列数为1
    Bjcol.Store = (void *) SUPERLU_MALLOC( sizeof(DNformat) );  # 分配 Bjcol.Store 的内存空间
    if ( !Bjcol.Store ) ABORT("SUPERLU_MALLOC fails for Bjcol.Store");  # 如果分配失败，则输出错误信息并中止程序
    Bjcol_store = Bjcol.Store;  # 将 Bjcol.Store 的地址保存到 Bjcol_store 中
    Bjcol_store->lda = ldb;  # 设置 Bjcol 的 lda 属性为 ldb
    Bjcol_store->nzval = work; /* address aliasing */  # 设置 Bjcol 的 nzval 指向 work 数组

    /* Do for each right hand side ... */  # 对每个右手边进行操作
    for (j = 0; j < nrhs; ++j) {  # 遍历每个右手边
    count = 0;  # 初始化计数器为0
    lstres = 3.;  # 初始化上一个残差为3.0
    Bptr = &Bmat[j*ldb];  # 获取右手边矩阵 B 的第 j 列的指针
    Xptr = &Xmat[j*ldx];  # 获取解矩阵 X 的第 j 列的指针

    while (1) { /* Loop until stopping criterion is satisfied. */  # 循环直到满足停止准则

        /* Compute residual R = B - op(A) * X,   
           where op(A) = A, A**T, or A**H, depending on TRANS. */  # 计算残差 R = B - op(A) * X，其中 op(A) 可能是 A, A**T, 或 A**H，取决于 TRANS
#ifdef _CRAY
        // 如果定义了_CRAY宏，使用CCOPY函数复制向量数据
        CCOPY(&nrow, Bptr, &ione, work, &ione);
#else
        // 如果未定义_CRAY宏，使用ccopy_函数复制向量数据
        ccopy_(&nrow, Bptr, &ione, work, &ione);
#endif

        // 调用BLAS库中的复数通用矩阵向量乘法函数
        sp_cgemv(transc, ndone, A, Xptr, ione, done, work, ione);

        /* 计算每个分量的相对后向误差，公式为
           max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )   
           其中 abs(Z) 表示矩阵或向量 Z 的逐分量绝对值。
           如果分母的第i个分量小于 SAFE2，则在除法之前将 SAFE1 加到分子的第i个分量上。 */

        // 初始化rwork数组，存放分量的绝对值
        for (i = 0; i < A->nrow; ++i) rwork[i] = c_abs1( &Bptr[i] );

        /* 计算 abs(op(A))*abs(X) + abs(B) */
        if ( notran ) {
            // 非转置操作时
            for (k = 0; k < A->ncol; ++k) {
                xk = c_abs1( &Xptr[k] );
                for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i)
                    rwork[Astore->rowind[i]] += c_abs1(&Aval[i]) * xk;
            }
        } else {  /* trans = TRANS or CONJ */
            // 转置或共轭转置操作时
            for (k = 0; k < A->ncol; ++k) {
                s = 0.;
                for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) {
                    irow = Astore->rowind[i];
                    s += c_abs1(&Aval[i]) * c_abs1(&Xptr[irow]);
                }
                rwork[k] += s;
            }
        }

        // 计算最大的相对后向误差
        s = 0.;
        for (i = 0; i < A->nrow; ++i) {
            if (rwork[i] > safe2) {
                s = SUPERLU_MAX( s, c_abs1(&work[i]) / rwork[i] );
            } else if ( rwork[i] != 0.0 ) {
                s = SUPERLU_MAX( s, (c_abs1(&work[i]) + safe1) / rwork[i] );
            }
            /* 如果 rwork[i] 恰好为0.0，则真实的残差也必定为0.0。 */
        }

        // 将计算得到的相对后向误差存入berr数组
        berr[j] = s;

        /* 测试停止条件。如果满足以下条件，则继续迭代：
           1) 残差 BERR(J) 大于机器精度 eps，
           2) BERR(J) 在上一次迭代中至少减少了一半，
           3) 迭代次数 count 小于 ITMAX。 */
        if (berr[j] > eps && berr[j] * 2. <= lstres && count < ITMAX) {
            /* 更新解并再次尝试。 */
            cgstrs (trans, L, U, perm_c, perm_r, &Bjcol, stat, info);

#ifdef _CRAY
            // 如果定义了_CRAY宏，使用CAXPY函数更新向量
            CAXPY(&nrow, &done, work, &ione,
                  &Xmat[j*ldx], &ione);
#else
            // 如果未定义_CRAY宏，使用caxpy_函数更新向量
            caxpy_(&nrow, &done, work, &ione,
                   &Xmat[j*ldx], &ione);
#endif

            // 更新最后一个残差值为当前的 berr[j]
            lstres = berr[j];
            // 迭代次数加一
            ++count;
        } else {
            // 不满足停止条件，跳出循环
            break;
        }
        
    } /* end while */

    // 将迭代次数记录到stat结构体的RefineSteps成员中
    stat->RefineSteps = count;
    /* 计算矩阵范数误差的修正：
       norm(X - XTRUE) / norm(X) .le. FERR = norm( abs(inv(op(A)))*   
       ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)   
          其中   
            norm(Z) 是向量或矩阵 Z 的最大分量的幅度   
            inv(op(A)) 是 op(A) 的逆矩阵   
            abs(Z) 是向量或矩阵 Z 的逐分量绝对值   
            NZ 是矩阵 A 中任意一行中非零元素的最大数目加1   
            EPS 是机器精度   

          abs(R)+NZ*EPS*(abs(op(A))*abs(X)+abs(B)) 的第 i 个分量，   
          如果 abs(op(A))*abs(X) + abs(B) 的第 i 个分量小于 SAFE2，则增加 SAFE1。   

          使用 CLACON2 估计矩阵 inv(op(A)) * diag(W) 的无穷范数，   
          其中 W = abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) */
    
    /* 初始化 rwork 数组为 Bptr 中各元素的绝对值 */
    for (i = 0; i < A->nrow; ++i) rwork[i] = c_abs1( &Bptr[i] );
    
    /* 计算 abs(op(A))*abs(X) + abs(B) */
    if ( notran ) {
        /* 非转置情况下的计算 */
        for (k = 0; k < A->ncol; ++k) {
            xk = c_abs1( &Xptr[k] );
            for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i)
                rwork[Astore->rowind[i]] += c_abs1(&Aval[i]) * xk;
        }
    } else {  /* trans == TRANS or CONJ */
        /* 转置或共轭转置情况下的计算 */
        for (k = 0; k < A->ncol; ++k) {
            s = 0.;
            for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) {
                irow = Astore->rowind[i];
                xk = c_abs1( &Xptr[irow] );
                s += c_abs1(&Aval[i]) * xk;
            }
            rwork[k] += s;
        }
    }
    
    /* 对 rwork 数组进行修正 */
    for (i = 0; i < A->nrow; ++i) {
        if (rwork[i] > safe2)
            rwork[i] = c_abs(&work[i]) + (iwork[i]+1)*eps*rwork[i];
        else
            rwork[i] = c_abs(&work[i]) + (iwork[i]+1)*eps*rwork[i] + safe1;
    }
    
    /* 初始化 kase 为 0 */
    kase = 0;
    do {
        // 调用 LAPACK 函数 clacon2_，计算工作向量中的元素，并更新 ferr 数组
        clacon2_(&nrow, &work[A->nrow], work, &ferr[j], &kase, isave);
        // 如果 kase 为 0，退出循环
        if (kase == 0) break;

        // 根据 kase 的值选择不同的操作
        if (kase == 1) {
            /* Multiply by diag(W)*inv(op(A)**T)*(diag(C) or diag(R)). */
            // 如果使用非转置且需要列平衡，则对工作向量中的每个元素进行乘法操作
            if ( notran && colequ )
                for (i = 0; i < A->ncol; ++i) {
                    cs_mult(&work[i], &work[i], C[i]);
                }
            // 如果使用转置且需要行平衡，则对工作向量中的每个元素进行乘法操作
            else if ( !notran && rowequ )
                for (i = 0; i < A->nrow; ++i) {
                    cs_mult(&work[i], &work[i], R[i]);
                }

            // 调用 cgstrs 函数，执行求解线性系统的步骤
            cgstrs (transt, L, U, perm_c, perm_r, &Bjcol, stat, info);
            
            // 对工作向量中的每个元素进行乘法操作
            for (i = 0; i < A->nrow; ++i) {
                cs_mult(&work[i], &work[i], rwork[i]);
            }
        } else {
            /* Multiply by (diag(C) or diag(R))*inv(op(A))*diag(W). */
            // 对工作向量中的每个元素进行乘法操作
            for (i = 0; i < A->nrow; ++i) {
                cs_mult(&work[i], &work[i], rwork[i]);
            }
            
            // 调用 cgstrs 函数，执行求解线性系统的步骤
            cgstrs (trans, L, U, perm_c, perm_r, &Bjcol, stat, info);
            
            // 如果使用非转置且需要列平衡，则对工作向量中的每个元素进行乘法操作
            if ( notran && colequ )
                for (i = 0; i < A->ncol; ++i) {
                    cs_mult(&work[i], &work[i], C[i]);
                }
            // 如果使用转置且需要行平衡，则对工作向量中的每个元素进行乘法操作
            else if ( !notran && rowequ )
                for (i = 0; i < A->ncol; ++i) {
                    cs_mult(&work[i], &work[i], R[i]);  
                }
        }
        
    } while ( kase != 0 );

    /* Normalize error. */
    // 初始化 lstres 为 0
    lstres = 0.;
    // 根据不同的情况计算 lstres 的值
    if ( notran && colequ ) {
        for (i = 0; i < A->nrow; ++i)
            // 计算 C[i] * |Xptr[i]|
            lstres = SUPERLU_MAX( lstres, C[i] * c_abs1( &Xptr[i]) );
    } else if ( !notran && rowequ ) {
        for (i = 0; i < A->nrow; ++i)
            // 计算 R[i] * |Xptr[i]|
            lstres = SUPERLU_MAX( lstres, R[i] * c_abs1( &Xptr[i]) );
    } else {
        for (i = 0; i < A->nrow; ++i)
            // 计算 |Xptr[i]|
            lstres = SUPERLU_MAX( lstres, c_abs1( &Xptr[i]) );
    }
    // 如果 lstres 不为 0，则将 ferr[j] 除以 lstres
    if ( lstres != 0. )
        ferr[j] /= lstres;

    } /* for each RHS j ... */

    // 释放动态分配的内存空间
    SUPERLU_FREE(work);
    SUPERLU_FREE(rwork);
    SUPERLU_FREE(iwork);
    SUPERLU_FREE(Bjcol.Store);

    // 函数结束
    return;
} /* cgsrfs */
```