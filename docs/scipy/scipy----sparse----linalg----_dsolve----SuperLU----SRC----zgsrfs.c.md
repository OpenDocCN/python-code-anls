# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zgsrfs.c`

```
    *info = -4;
    else if ( perm_c == NULL || perm_r == NULL )
        *info = -5;
    else if ( equed == NULL || (equed[0] != 'N' && equed[0] != 'R' &&
                             equed[0] != 'C' && equed[0] != 'B') )
        *info = -6;
    else if ( R == NULL || C == NULL )
        *info = -7;
    else if ( B->nrow < 0 || B->ncol < 0 || ldb < max(1,B->nrow) )
        *info = -10;
    else if ( X->nrow < 0 || X->ncol < 0 || ldx < max(1,X->nrow) )
        *info = -12;
    if ( *info != 0 ) {
        i = -(*info);
        xerbla_("zgsrfs", &i);
        return;
    }

    /* Quick return if possible */
    if ( nrow == 0 || nrhs == 0 ) {
        for (j = 0; j < nrhs; ++j) {
            ferr[j] = 0.;
            berr[j] = 0.;
        }
        return;
    }

    /* Set constants */
    transc[0] = trans;
    transt = (trans == TRANS) ? CONJ : TRANS;
    eps = dlamch_("Epsilon");
    safmin = dlamch_("Safe minimum");
    safe1 = ITMAX * eps;
    safe2 = safe1 / safmin;
    nrhs = B->ncol;
    
    /* Initialize statistics variables */
    stat->utime = 0.0;
    stat->ops = 0.0;

    /* Set up the workspace arrays */
    work = doublecomplexMalloc(nrow);
    rwork = (double *) SUPERLU_MALLOC(nrow * sizeof(double));
    iwork = (int *) SUPERLU_MALLOC(nrow * sizeof(int));

    /* Main loop for each right-hand side */
    for (k = 0; k < nrhs; ++k) {
        Bptr = &Bmat[k * ldb];
        Xptr = &Xmat[k * ldx];
        Bjcol.Store = NULL;
        
        /* Apply permutations and equilibrations */
        if ( notran ) {
            zlacon2_(&nrow, &Bptr[0], &Xptr[0], &lstres, &kase, isave);
        } else {
            if ( k == 0 ) {
                if ( !equil ) {
                    rowequ = 0;
                    colequ = 0;
                } else if ( equed[0] == 'R' || equed[0] == 'B' ) {
                    rowequ = 1;
                    colequ = 0;
                } else if ( equed[0] == 'C' ) {
                    rowequ = 0;
                    colequ = 1;
                } else if ( equed[0] == 'N' ) {
                    rowequ = 0;
                    colequ = 0;
                }
            }
        }
    }
}


注释：


    *info = -4;
    else if ( perm_c == NULL || perm_r == NULL )
        *info = -5;
    else if ( equed == NULL || (equed[0] != 'N' && equed[0] != 'R' &&
                             equed[0] != 'C' && equed[0] != 'B') )
        *info = -6;
    else if ( R == NULL || C == NULL )
        *info = -7;
    else if ( B->nrow < 0 || B->ncol < 0 || ldb < max(1,B->nrow) )
        *info = -10;
    else if ( X->nrow < 0 || X->ncol < 0 || ldx < max(1,X->nrow) )
        *info = -12;
    if ( *info != 0 ) {
        i = -(*info);
        xerbla_("zgsrfs", &i);
        return;
    }
检查输入参数是否有效，如果不是，则设置错误码并返回。perm_c 和 perm_r 不能为 NULL，equed 必须是有效字符，R 和 C 不能为 NULL，B 和 X 的行列数以及 ldb 和 ldx 的值必须满足条件。

    /* Quick return if possible */
如果 nrow 或 nrhs 为 0，则直接将 ferr 和 berr 设置为零并返回。

    /* Set constants */
设置常数和变量。transc 是 trans 的字符形式，transt 是 trans 的转置或共轭转置形式，eps 和 safmin 是机器精度和安全最小值的常数。

    /* Initialize statistics variables */
初始化统计变量，包括 utime 和 ops。

    /* Set up the workspace arrays */
设置工作空间数组，包括 work、rwork 和 iwork。

    /* Main loop for each right-hand side */
主循环，处理每个右侧向量。

        /* Apply permutations and equilibrations */
应用排列和均衡化，根据 notran 的值决定是否调用 zlacon2_ 函数，以及根据 equed 的值确定 rowequ 和 colequ 的设置。
    *info = -4;
    // 如果输入错误码为-4，则表示参数异常
    else if ( ldb < SUPERLU_MAX(0, A->nrow) ||
           B->Stype != SLU_DN || B->Dtype != SLU_Z || B->Mtype != SLU_GE )
        *info = -10;
    // 如果ldb小于0或者B不是期望的类型（稠密复双精度矩阵），则设置错误码为-10
    else if ( ldx < SUPERLU_MAX(0, A->nrow) ||
           X->Stype != SLU_DN || X->Dtype != SLU_Z || X->Mtype != SLU_GE )
    *info = -11;
    // 如果ldx小于0或者X不是期望的类型（稠密复双精度矩阵），则设置错误码为-11
    if (*info != 0) {
    i = -(*info);
    // 将错误码转换为正数
    input_error("zgsrfs", &i);
    // 报告输入错误
    return;
    // 函数结束
    }

    /* Quick return if possible */
    // 如果A的行数为0或者右手边的数量为0，则快速返回
    if ( A->nrow == 0 || nrhs == 0) {
    // 针对每个右手边的向量，设置残差和误差为0
    for (j = 0; j < nrhs; ++j) {
        ferr[j] = 0.;
        berr[j] = 0.;
    }
    return;
    // 函数结束
    }

    rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
    // 检查equed是否以'R'或者'B'开头
    colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
    // 检查equed是否以'C'或者'B'开头
    
    /* Allocate working space */
    // 分配工作空间
    work = doublecomplexMalloc(2*A->nrow);
    // 分配大小为2*A->nrow的复双精度空间
    rwork = (double *) SUPERLU_MALLOC( A->nrow * sizeof(double) );
    // 分配大小为A->nrow的双精度空间
    iwork = int32Malloc(A->nrow);
    // 分配大小为A->nrow的32位整数空间
    if ( !work || !rwork || !iwork ) 
        ABORT("Malloc fails for work/rwork/iwork.");
    // 如果分配失败，则终止程序
    
    if ( notran ) {
    *(unsigned char *)transc = 'N';
        transt = TRANS;
    }
    // 如果不是转置操作，设置transc为'N'，transt为TRANS
    else if ( trans == TRANS ) {
    *(unsigned char *)transc = 'T';
    transt = NOTRANS;
    }
    // 如果是转置操作，设置transc为'T'，transt为NOTRANS
    else if ( trans == CONJ ) {
    *(unsigned char *)transc = 'C';
    transt = NOTRANS;
    }    
    // 如果是共轭转置操作，设置transc为'C'，transt为NOTRANS

    /* NZ = maximum number of nonzero elements in each row of A, plus 1 */
    // NZ = A每行非零元素的最大数量加1
    nz     = A->ncol + 1;
    // 设置nz为A的列数加1
    eps    = dmach("Epsilon");
    // 计算机器精度
    safmin = dmach("Safe minimum");
    // 计算安全最小值

    /* Set SAFE1 essentially to be the underflow threshold times the
       number of additions in each row. */
    // 设置SAFE1为下溢阈值乘以每行的加法次数
    safe1  = nz * safmin;
    // 计算safe1
    safe2  = safe1 / eps;
    // 计算safe2

    /* Compute the number of nonzeros in each row (or column) of A */
    // 计算A每行（或列）的非零元素数量
    for (i = 0; i < A->nrow; ++i) iwork[i] = 0;
    // 初始化iwork为0
    if ( notran ) {
    for (k = 0; k < A->ncol; ++k)
        for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) 
        ++iwork[Astore->rowind[i]];
    }
    // 如果不是转置操作，计算每行的非零元素数量
    else {
    for (k = 0; k < A->ncol; ++k)
        iwork[k] = Astore->colptr[k+1] - Astore->colptr[k];
    }    
    // 如果是转置操作，计算每列的非零元素数量

    /* Copy one column of RHS B into Bjcol. */
    // 将右手边矩阵B的一列复制到Bjcol中
    Bjcol.Stype = B->Stype;
    Bjcol.Dtype = B->Dtype;
    Bjcol.Mtype = B->Mtype;
    Bjcol.nrow  = B->nrow;
    Bjcol.ncol  = 1;
    Bjcol.Store = (void *) SUPERLU_MALLOC( sizeof(DNformat) );
    // 分配大小为DNformat结构体的内存空间给Bjcol.Store
    if ( !Bjcol.Store ) ABORT("SUPERLU_MALLOC fails for Bjcol.Store");
    // 如果分配失败，则终止程序
    Bjcol_store = Bjcol.Store;
    Bjcol_store->lda = ldb;
    Bjcol_store->nzval = work; /* address aliasing */
    // 设置Bjcol_store的lda和nzval为相应的值（地址别名）

    /* Do for each right hand side ... */
    // 针对每个右手边的向量执行以下操作
    for (j = 0; j < nrhs; ++j) {
    count = 0;
    // 初始化计数为0
    lstres = 3.;
    // 设置初始残差为3
    Bptr = &Bmat[j*ldb];
    // 指向右手边矩阵B的第j列
    Xptr = &Xmat[j*ldx];
    // 指向解向量矩阵X的第j列

    while (1) { /* Loop until stopping criterion is satisfied. */

        /* Compute residual R = B - op(A) * X,   
           where op(A) = A, A**T, or A**H, depending on TRANS. */
        // 计算残差R = B - op(A) * X，其中op(A) = A, A**T, 或 A**H，取决于TRANS
#ifdef _CRAY
    // 使用 Cray 特定的复制函数 CCOPY 复制向量 Bptr 到 work 中
    CCOPY(&nrow, Bptr, &ione, work, &ione);
#else
    // 使用 BLAS 函数 zcopy_ 复制向量 Bptr 到 work 中
    zcopy_(&nrow, Bptr, &ione, work, &ione);
#endif

// 调用稀疏矩阵-稠密向量乘法函数 sp_zgemv 计算 A*X 或者 A^T*X，结果存储在 work 中
sp_zgemv(transc, ndone, A, Xptr, ione, done, work, ione);

/* 计算组件级别的相对后向误差，公式为
   max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
   其中 abs(Z) 表示矩阵或向量 Z 的逐元素绝对值。
   如果分母的第 i 个分量小于 SAFE2，则将 SAFE1 添加到分子的第 i 个分量再进行除法。 */

// 初始化 rwork 数组，存储向量 Bptr 的绝对值
for (i = 0; i < A->nrow; ++i)
    rwork[i] = z_abs1( &Bptr[i] );

/* 计算 abs(op(A))*abs(X) + abs(B) */
if ( notran ) {
    // 当不需要转置时的计算
    for (k = 0; k < A->ncol; ++k) {
        // 计算向量 Xptr 的绝对值 xk
        xk = z_abs1( &Xptr[k] );
        // 遍历 A 的列 k，并更新 rwork 中对应行的值
        for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i)
            rwork[Astore->rowind[i]] += z_abs1(&Aval[i]) * xk;
    }
} else {  /* trans = TRANS or CONJ */
    // 当需要转置时的计算
    for (k = 0; k < A->ncol; ++k) {
        s = 0.;
        // 遍历 A 的列 k，并计算对应元素的乘积加和 s
        for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) {
            irow = Astore->rowind[i];
            s += z_abs1(&Aval[i]) * z_abs1(&Xptr[irow]);
        }
        // 更新 rwork 中对应列的值
        rwork[k] += s;
    }
}

s = 0.;
// 计算每行的相对后向误差，并更新到向量 berr 中的第 j 个位置
for (i = 0; i < A->nrow; ++i) {
    if (rwork[i] > safe2) {
        s = SUPERLU_MAX( s, z_abs1(&work[i]) / rwork[i] );
    } else if ( rwork[i] != 0.0 ) {
        s = SUPERLU_MAX( s, (z_abs1(&work[i]) + safe1) / rwork[i] );
    }
    /* 如果 rwork[i] 精确为 0.0，则真实残差也必须精确为 0.0。 */
}

// 将计算得到的相对后向误差存储到 berr 的第 j 个位置
berr[j] = s;

/* 测试停止准则。继续迭代条件是：
   1) 残差 berr[j] 大于机器精度 eps，并且
   2) berr[j] 在上次迭代中至少减少了一半，并且
   3) 迭代次数 count 小于最大迭代次数 ITMAX。 */

if (berr[j] > eps && berr[j] * 2. <= lstres && count < ITMAX) {
    /* 更新解并继续迭代 */
    zgstrs (trans, L, U, perm_c, perm_r, &Bjcol, stat, info);
    
#ifdef _CRAY
    // 使用 Cray 特定的 axpy 函数 CAXPY 更新解 Xmat[j*ldx] = Xmat[j*ldx] + done*work
    CAXPY(&nrow, &done, work, &ione,
           &Xmat[j*ldx], &ione);
#else
    // 使用 BLAS 函数 zaxpy_ 更新解 Xmat[j*ldx] = Xmat[j*ldx] + done*work
    zaxpy_(&nrow, &done, work, &ione,
           &Xmat[j*ldx], &ione);
#endif
    // 更新最后一个残差值 lstres 为当前 berr[j]
    lstres = berr[j];
    // 增加迭代计数器 count
    ++count;
} else {
    // 不满足迭代条件，跳出循环
    break;
}

} /* end while */

// 将迭代次数 count 存储到 stat->RefineSteps
stat->RefineSteps = count;
    /* 根据公式计算误差边界：
       norm(X - XTRUE) / norm(X) ≤ FERR = norm( abs(inv(op(A))) *
       ( abs(R) + NZ*EPS*( abs(op(A))*abs(X) + abs(B) ))) / norm(X)
          其中：
            norm(Z) 是向量 Z 中最大分量的大小
            inv(op(A)) 是 op(A) 的逆矩阵
            abs(Z) 是矩阵或向量 Z 的逐分量绝对值
            NZ 是 A 中任何行的非零元素的最大数目加一
            EPS 是机器精度

          如果 abs(op(A))*abs(X) + abs(B) 的第 i 个分量小于 SAFE2，则将
          abs(R) + NZ*EPS*(abs(op(A))*abs(X) + abs(B)) 的第 i 个分量增加 SAFE1。

          使用 ZLACON2 估算矩阵 inv(op(A)) * diag(W) 的无穷范数，
          其中 W = abs(R) + NZ*EPS*( abs(op(A))*abs(X) + abs(B) ))) */

    for (i = 0; i < A->nrow; ++i)
        // 计算 rwork 数组的每个元素，即 abs(R)
        rwork[i] = z_abs1( &Bptr[i] );

    /* 计算 abs(op(A))*abs(X) + abs(B)。*/
    if ( notran ) {
        for (k = 0; k < A->ncol; ++k) {
            // 计算每列 k 的 xk = abs(X[k])，并累加到相应的行
            xk = z_abs1( &Xptr[k] );
            for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i)
                rwork[Astore->rowind[i]] += z_abs1(&Aval[i]) * xk;
        }
    } else {  /* trans == TRANS or CONJ */
        for (k = 0; k < A->ncol; ++k) {
            s = 0.;
            // 对于转置或共轭传输情况，计算每列 k 的 s，并累加到相应的行
            for (i = Astore->colptr[k]; i < Astore->colptr[k+1]; ++i) {
                irow = Astore->rowind[i];
                xk = z_abs1( &Xptr[irow] );
                s += z_abs1(&Aval[i]) * xk;
            }
            rwork[k] += s;
        }
    }

    for (i = 0; i < A->nrow; ++i) {
        // 根据条件判断 rwork[i] 大小，并进行调整
        if (rwork[i] > safe2)
            rwork[i] = z_abs(&work[i]) + (iwork[i]+1)*eps*rwork[i];
        else
            rwork[i] = z_abs(&work[i]) + (iwork[i]+1)*eps*rwork[i] + safe1;
    }
    kase = 0;
    do {
        zlacon2_(&nrow, &work[A->nrow], work, &ferr[j], &kase, isave);
        // 调用 zlacon2_ 函数计算特征值估计量，并确定处理情况 kase
        if (kase == 0) break;

        if (kase == 1) {
            /* Multiply by diag(W)*inv(op(A)**T)*(diag(C) or diag(R)). */
            // 根据 kase 的值选择不同的处理方式
            if ( notran && colequ )
                // 如果 notran 为真且 colequ 为真，则对每列进行特定操作
                for (i = 0; i < A->ncol; ++i) {
                    zd_mult(&work[i], &work[i], C[i]);
                }
            else if ( !notran && rowequ )
                // 如果 notran 为假且 rowequ 为真，则对每行进行特定操作
                for (i = 0; i < A->nrow; ++i) {
                    zd_mult(&work[i], &work[i], R[i]);
                }

            // 解线性方程组
            zgstrs (transt, L, U, perm_c, perm_r, &Bjcol, stat, info);

            // 对工作数组中的每个元素进行处理
            for (i = 0; i < A->nrow; ++i) {
                zd_mult(&work[i], &work[i], rwork[i]);
            }
        } else {
            /* Multiply by (diag(C) or diag(R))*inv(op(A))*diag(W). */
            // 根据 kase 的值选择另一种处理方式
            for (i = 0; i < A->nrow; ++i) {
                zd_mult(&work[i], &work[i], rwork[i]);
            }

            // 解线性方程组
            zgstrs (trans, L, U, perm_c, perm_r, &Bjcol, stat, info);

            // 根据不同的条件对工作数组中的每个元素进行处理
            if ( notran && colequ )
                for (i = 0; i < A->ncol; ++i) {
                    zd_mult(&work[i], &work[i], C[i]);
                }
            else if ( !notran && rowequ )
                for (i = 0; i < A->ncol; ++i) {
                    zd_mult(&work[i], &work[i], R[i]);
                }
        }
        
    } while ( kase != 0 );  // 继续循环直到 kase 为 0

    /* Normalize error. */
    lstres = 0.;
    // 根据不同的条件计算误差的标准化值
    if ( notran && colequ ) {
        for (i = 0; i < A->nrow; ++i)
            lstres = SUPERLU_MAX( lstres, C[i] * z_abs1( &Xptr[i]) );
    } else if ( !notran && rowequ ) {
        for (i = 0; i < A->nrow; ++i)
            lstres = SUPERLU_MAX( lstres, R[i] * z_abs1( &Xptr[i]) );
    } else {
        for (i = 0; i < A->nrow; ++i)
            lstres = SUPERLU_MAX( lstres, z_abs1( &Xptr[i]) );
    }
    if ( lstres != 0. )
        ferr[j] /= lstres;

    } /* for each RHS j ... */

    // 释放动态分配的内存
    SUPERLU_FREE(work);
    SUPERLU_FREE(rwork);
    SUPERLU_FREE(iwork);
    SUPERLU_FREE(Bjcol.Store);

    return;
} /* zgsrfs */
```