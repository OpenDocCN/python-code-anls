# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\smemory.c`

```
/*!
 * Setup the memory model to be used for factorization.
 *  
 *    lwork = 0: use system malloc;
 *    lwork > 0: use user-supplied work[] space.
 */
void sSetupSpace(void *work, int_t lwork, GlobalLU_t *Glu)
{
    // 如果 lwork 为 0，则使用系统的 malloc/free
    if ( lwork == 0 ) {
        Glu->MemModel = SYSTEM; /* malloc/free */
    } 
    // 如果 lwork 大于 0，则使用用户提供的 work[] 空间
    else if ( lwork > 0 ) {
        Glu->MemModel = USER;   /* user provided space */
        // 初始化堆栈的使用情况和顶部指针
        Glu->stack.used = 0;
        Glu->stack.top1 = 0;
        // 计算栈顶部2的位置，并确保是按字地址对齐的
        Glu->stack.top2 = (lwork/4)*4; /* must be word addressable */
        Glu->stack.size = Glu->stack.top2;
        Glu->stack.array = (void *) work;
    }
}



/*!
 * Allocate memory from the user-provided stack.
 *
 * \param bytes Number of bytes to allocate.
 * \param which_end Specifies whether to allocate from the head or tail of the stack.
 * \param Glu GlobalLU_t structure containing stack information.
 * \return Pointer to the allocated memory, or NULL if allocation fails.
 */
void *suser_malloc(int bytes, int which_end, GlobalLU_t *Glu)
{
    void *buf;
    
    // 检查堆栈是否已满
    if ( StackFull(bytes) ) return (NULL);

    // 根据 which_end 指定的位置从堆栈头或尾分配内存
    if ( which_end == HEAD ) {
        buf = (char*) Glu->stack.array + Glu->stack.top1;
        Glu->stack.top1 += bytes;
    } else {
        Glu->stack.top2 -= bytes;
        buf = (char*) Glu->stack.array + Glu->stack.top2;
    }
    
    // 更新堆栈的使用情况
    Glu->stack.used += bytes;
    return buf;
}


/*!
 * Free memory allocated from the user-provided stack.
 *
 * \param bytes Number of bytes to free.
 * \param which_end Specifies whether to free memory from the head or tail of the stack.
 * \param Glu GlobalLU_t structure containing stack information.
 */
void suser_free(int bytes, int which_end, GlobalLU_t *Glu)
{
    // 根据 which_end 指定的位置释放内存
    if ( which_end == HEAD ) {
        Glu->stack.top1 -= bytes;
    } else {
        Glu->stack.top2 += bytes;
    }
    // 更新堆栈的使用情况
    Glu->stack.used -= bytes;
}
/*!
 * Query the space required for the LU factorization.
 *
 * \param L SuperMatrix pointer to the lower triangular matrix L.
 * \param U SuperMatrix pointer to the upper triangular matrix U.
 * \param mem_usage Pointer to mem_usage_t structure to store memory usage details.
 *
 * \return Returns 0 upon successful completion.
 */
int sQuerySpace(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage)
{
    SCformat *Lstore;
    NCformat *Ustore;
    register int n, iword, dword, panel_size = sp_ienv(1);

    // Assign the store formats of L and U matrices
    Lstore = L->Store;
    Ustore = U->Store;
    // Number of columns in matrix L
    n = L->ncol;
    // Size of int and float (or double) in bytes
    iword = sizeof(int);
    dword = sizeof(float);

    /* For LU factors */
    // Calculate memory usage for LU factors in bytes
    mem_usage->for_lu = (float)( (4.0 * n + 3.0) * iword +
                                 Lstore->nzval_colptr[n] * dword +
                                 Lstore->rowind_colptr[n] * iword );
    mem_usage->for_lu += (float)( (n + 1.0) * iword +
                                 Ustore->colptr[n] * (dword + iword) );

    /* Working storage to support factorization */
    // Calculate total needed memory including working storage
    mem_usage->total_needed = mem_usage->for_lu +
    (float)( (2.0 * panel_size + 4.0 + NO_MARKER) * n * iword +
            (panel_size + 1.0) * n * dword );

    return 0;
} /* sQuerySpace */


/*!
 * Query the space required for ILU factorization.
 *
 * \param L SuperMatrix pointer to the lower triangular matrix L.
 * \param U SuperMatrix pointer to the upper triangular matrix U.
 * \param mem_usage Pointer to mem_usage_t structure to store memory usage details.
 *
 * \return Returns 0 upon successful completion.
 */
int ilu_sQuerySpace(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage)
{
    SCformat *Lstore;
    NCformat *Ustore;
    register int n, panel_size = sp_ienv(1);
    register float iword, dword;

    // Assign the store formats of L and U matrices
    Lstore = L->Store;
    Ustore = U->Store;
    // Number of columns in matrix L
    n = L->ncol;
    // Size of int and double in bytes
    iword = sizeof(int);
    dword = sizeof(double);

    /* For LU factors */
    // Calculate memory usage for ILU factors in bytes
    mem_usage->for_lu = (float)( (4.0f * n + 3.0f) * iword +
                                 Lstore->nzval_colptr[n] * dword +
                                 Lstore->rowind_colptr[n] * iword );
    mem_usage->for_lu += (float)( (n + 1.0f) * iword +
                                 Ustore->colptr[n] * (dword + iword) );

    /* Working storage to support factorization.
       ILU needs 5*n more integers than LU */
    // Calculate total needed memory for ILU including working storage
    mem_usage->total_needed = mem_usage->for_lu +
    (float)( (2.0f * panel_size + 9.0f + NO_MARKER) * n * iword +
            (panel_size + 1.0f) * n * dword );

    return 0;
} /* ilu_sQuerySpace */


/*! \brief Initialize memory for the LU factorization.
 *
 * This function estimates and allocates memory for data structures common to all factorization routines.
 * It also supports a special mode (lwork = -1) for estimating memory requirements without allocating.
 *
 * \param fact Type of factorization to perform.
 * \param work Pointer to workspace.
 * \param lwork Size of workspace allocated.
 * \param m Number of rows in the matrix.
 * \param n Number of columns in the matrix.
 * \param annz Number of non-zero entries in the matrix A.
 * \param panel_size Size of the panel used in the factorization.
 * \param fill_ratio Fill ratio for estimating workspace.
 * \param L SuperMatrix pointer to the lower triangular matrix L.
 * \param U SuperMatrix pointer to the upper triangular matrix U.
 * \param Glu Pointer to GlobalLU_t structure for global LU information.
 * \param iwork Pointer to integer workspace.
 * \param dwork Pointer to floating-point workspace.
 *
 * \return Estimated memory space required if lwork = -1, actual allocated memory on success, or an error code on failure.
 */
int_t sLUMemInit(fact_t fact, void *work, int_t lwork, int m, int n, int_t annz,
                 int panel_size, float fill_ratio, SuperMatrix *L, SuperMatrix *U,
                 GlobalLU_t *Glu, int **iwork, float **dwork)
{
    int      info, iword, dword;
    SCformat *Lstore;
    NCformat *Ustore;
    int      *xsup, *supno;
    int_t    *lsub, *xlsub;
    float   *lusup;
    int_t    *xlusup;
    float   *ucol;
    int_t    *usub, *xusub;
    int_t    nzlmax, nzumax, nzlumax;
    
    // Size of int in bytes
    iword     = sizeof(int);
    // Assign the store formats of L and U matrices
    Lstore = L->Store;
    Ustore = U->Store;
    // Allocate memory for various data structures used in factorization
    // (details omitted due to complexity and specificity of implementation)
    
    # 定义一个变量 dword，存储 float 类型的大小（字节数）
    dword     = sizeof(float);
    # 设置 Glu 结构体的成员变量 n 为给定的 n
    Glu->n    = n;
    # 将 Glu 结构体的 num_expansions 成员变量设为 0
    Glu->num_expansions = 0;

    # 为 Glu 结构体的 expanders 成员分配内存空间，大小为 NO_MEMTYPE * sizeof(ExpHeader)
    Glu->expanders = (ExpHeader *) SUPERLU_MALLOC( NO_MEMTYPE *
                                                     sizeof(ExpHeader) );
    # 如果内存分配失败，则打印错误信息并中止程序
    if ( !Glu->expanders ) ABORT("SUPERLU_MALLOC fails for expanders");
    
    # 如果 fact 不等于 SamePattern_SameRowPerm，则进行以下操作
    /* Guess for L\U factors */
    # 计算预估的 L/U 因子的非零元素数量上限
    nzumax = nzlumax = nzlmax = fill_ratio * annz;
    # 如果 lwork 等于 -1，则返回所需的工作空间大小
    if ( lwork == -1 ) {
        return ( GluIntArray(n) * iword + TempSpace(m, panel_size)
            + (nzlmax+nzumax)*iword + (nzlumax+nzumax)*dword + n );
        } else {
        # 否则，配置工作空间
        sSetupSpace(work, lwork, Glu);
    }
#if ( PRNTlevel >= 1 )
    // 如果打印级别大于等于1，输出初始化信息
    printf("sLUMemInit() called: fill_ratio %.0f, nzlmax %lld, nzumax %lld\n", 
           fill_ratio, (long long) nzlmax, (long long) nzumax);
    // 刷新标准输出流
    fflush(stdout);
#endif    
    
    /* Integer pointers for L\U factors */
    // 为 L/U 因子分配整数指针
    if ( Glu->MemModel == SYSTEM ) {
        xsup   = int32Malloc(n+1);   // 分配 n+1 个 int32_t 大小的内存
        supno  = int32Malloc(n+1);   // 分配 n+1 个 int32_t 大小的内存
        xlsub  = intMalloc(n+1);     // 分配 n+1 个 int 大小的内存
        xlusup = intMalloc(n+1);     // 分配 n+1 个 int 大小的内存
        xusub  = intMalloc(n+1);     // 分配 n+1 个 int 大小的内存
    } else {
        xsup   = (int *)suser_malloc((n+1) * iword, HEAD, Glu);   // 使用用户提供的内存分配函数分配内存
        supno  = (int *)suser_malloc((n+1) * iword, HEAD, Glu);   // 使用用户提供的内存分配函数分配内存
        xlsub  = suser_malloc((n+1) * iword, HEAD, Glu);          // 使用用户提供的内存分配函数分配内存
        xlusup = suser_malloc((n+1) * iword, HEAD, Glu);          // 使用用户提供的内存分配函数分配内存
        xusub  = suser_malloc((n+1) * iword, HEAD, Glu);          // 使用用户提供的内存分配函数分配内存
    }

    lusup = (float *) sexpand( &nzlumax, LUSUP, 0, 0, Glu );   // 扩展 LUSUP 内存空间并返回其地址
    ucol  = (float *) sexpand( &nzumax, UCOL, 0, 0, Glu );    // 扩展 UCOL 内存空间并返回其地址
    lsub  = (int_t *) sexpand( &nzlmax, LSUB, 0, 0, Glu );    // 扩展 LSUB 内存空间并返回其地址
    usub  = (int_t *) sexpand( &nzumax, USUB, 0, 1, Glu );    // 扩展 USUB 内存空间并返回其地址

    while ( !lusup || !ucol || !lsub || !usub ) {
        if ( Glu->MemModel == SYSTEM ) {
            // 如果内存模型为 SYSTEM，释放超节点内存
            SUPERLU_FREE(lusup); 
            SUPERLU_FREE(ucol); 
            SUPERLU_FREE(lsub); 
            SUPERLU_FREE(usub);
        } else {
            // 使用用户提供的内存释放函数释放超节点内存
            suser_free((nzlumax+nzumax)*dword+(nzlmax+nzumax)*iword,
                            HEAD, Glu);
        }
        // 减半各种内存大小
        nzlumax /= 2;
        nzumax /= 2;
        nzlmax /= 2;
        if ( nzlumax < annz ) {
            // 如果内存不足以执行因子分解，输出错误信息并返回内存使用情况和 n
            printf("Not enough memory to perform factorization.\n");
            return (smemory_usage(nzlmax, nzumax, nzlumax, n) + n);
        }
#if ( PRNTlevel >= 1)
        // 如果打印级别大于等于1，输出减小尺寸后的内存信息
        printf("sLUMemInit() reduce size: nzlmax %ld, nzumax %ld\n", 
           (long) nzlmax, (long) nzumax);
        // 刷新标准输出流
        fflush(stdout);
#endif
        // 再次扩展各种内存大小并返回其地址
        lusup = (float *) sexpand( &nzlumax, LUSUP, 0, 0, Glu );
        ucol  = (float *) sexpand( &nzumax, UCOL, 0, 0, Glu );
        lsub  = (int_t *) sexpand( &nzlmax, LSUB, 0, 0, Glu );
        usub  = (int_t *) sexpand( &nzumax, USUB, 0, 1, Glu );
    }
    
    } else {
    /* fact == SamePattern_SameRowPerm */
    // 如果 fact 等于 SamePattern_SameRowPerm
    Lstore   = L->Store;         // 将 L 的存储指针赋给 Lstore
    Ustore   = U->Store;         // 将 U 的存储指针赋给 Ustore
    xsup     = Lstore->sup_to_col;   // 将 Lstore 的 sup_to_col 指针赋给 xsup
    supno    = Lstore->col_to_sup;   // 将 Lstore 的 col_to_sup 指针赋给 supno
    xlsub    = Lstore->rowind_colptr;    // 将 Lstore 的 rowind_colptr 指针赋给 xlsub
    xlusup   = Lstore->nzval_colptr;     // 将 Lstore 的 nzval_colptr 指针赋给 xlusup
    xusub    = Ustore->colptr;    // 将 Ustore 的 colptr 指针赋给 xusub
    nzlmax   = Glu->nzlmax;       // 将 Glu 的 nzlmax 赋给 nzlmax（来自前次因子分解的最大值）
    nzumax   = Glu->nzumax;       // 将 Glu 的 nzumax 赋给 nzumax（来自前次因子分解的最大值）
    nzlumax  = Glu->nzlumax;      // 将 Glu 的 nzlumax 赋给 nzlumax
    
    if ( lwork == -1 ) {
        // 如果 lwork 等于 -1，返回用于因子分解的内存估计值
        return ( GluIntArray(n) * iword + TempSpace(m, panel_size)
            + (nzlmax+nzumax)*iword + (nzlumax+nzumax)*dword + n );
        } else if ( lwork == 0 ) {
        // 如果 lwork 等于 0，将内存模型设置为 SYSTEM
        Glu->MemModel = SYSTEM;
    } else {
        // 否则，将内存模型设置为 USER，并设置堆栈顶部
        Glu->MemModel = USER;
        Glu->stack.top2 = (lwork/4)*4;   // 必须是字地址可寻址的
        Glu->stack.size = Glu->stack.top2;   // 设置堆栈大小为 top2
    }
    
    lsub  = Glu->expanders[LSUB].mem  = Lstore->rowind;   // 设置 LSUB 内存地址
    lusup = Glu->expanders[LUSUP].mem = Lstore->nzval;    // 设置 LUSUP 内存地址
    usub  = Glu->expanders[USUB].mem  = Ustore->rowind;   // 设置 USUB 内存地址
    ucol  = Glu->expanders[UCOL].mem  = Ustore->nzval;;
    // 设置Glu结构体中expanders数组中UCOL位置的mem成员为Ustore结构体的nzval，同时将其赋值给ucol

    Glu->expanders[LSUB].size         = nzlmax;
    // 设置Glu结构体中expanders数组中LSUB位置的size成员为nzlmax，表示LSUB的大小

    Glu->expanders[LUSUP].size        = nzlumax;
    // 设置Glu结构体中expanders数组中LUSUP位置的size成员为nzlumax，表示LUSUP的大小

    Glu->expanders[USUB].size         = nzumax;
    // 设置Glu结构体中expanders数组中USUB位置的size成员为nzumax，表示USUB的大小

    Glu->expanders[UCOL].size         = nzumax;
    // 设置Glu结构体中expanders数组中UCOL位置的size成员为nzumax，表示UCOL的大小

    }

    Glu->xsup    = xsup;
    // 将xsup赋值给Glu结构体中的xsup成员，表示超节点的起始索引

    Glu->supno   = supno;
    // 将supno赋值给Glu结构体中的supno成员，表示超节点的编号

    Glu->lsub    = lsub;
    // 将lsub赋值给Glu结构体中的lsub成员，表示L分解的行索引数组

    Glu->xlsub   = xlsub;
    // 将xlsub赋值给Glu结构体中的xlsub成员，表示L分解的行偏移数组

    Glu->lusup   = (void *) lusup;
    // 将lusup转换为void指针后赋值给Glu结构体中的lusup成员，表示L分解的数据存储数组

    Glu->xlusup  = xlusup;
    // 将xlusup赋值给Glu结构体中的xlusup成员，表示L分解的列偏移数组

    Glu->ucol    = (void *) ucol;
    // 将ucol转换为void指针后赋值给Glu结构体中的ucol成员，表示U分解的列索引数组

    Glu->usub    = usub;
    // 将usub赋值给Glu结构体中的usub成员，表示U分解的行索引数组

    Glu->xusub   = xusub;
    // 将xusub赋值给Glu结构体中的xusub成员，表示U分解的行偏移数组

    Glu->nzlmax  = nzlmax;
    // 将nzlmax赋值给Glu结构体中的nzlmax成员，表示L分解的最大非零元数

    Glu->nzumax  = nzumax;
    // 将nzumax赋值给Glu结构体中的nzumax成员，表示U分解的最大非零元数

    Glu->nzlumax = nzlumax;
    // 将nzlumax赋值给Glu结构体中的nzlumax成员，表示L和U分解的最大非零元数

    info = sLUWorkInit(m, n, panel_size, iwork, dwork, Glu);
    // 调用sLUWorkInit函数初始化LU分解的工作空间，返回初始化信息

    if ( info )
        return ( info + smemory_usage(nzlmax, nzumax, nzlumax, n) + n);
    // 如果初始化失败，返回错误码加上计算的内存使用量

    ++Glu->num_expansions;
    // 增加Glu结构体中的num_expansions计数器，表示扩展操作的次数

    return 0;
    // 成功初始化后返回0表示没有错误
/*! \brief Allocate known working storage.
 * Returns 0 if success, otherwise
 * returns the number of bytes allocated so far when failure occurred.
 */
int
sLUWorkInit(int m, int n, int panel_size, int **iworkptr, 
            float **dworkptr, GlobalLU_t *Glu)
{
    int    isize, dsize, extra;
    float *old_ptr;
    int    maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) ),
           rowblk   = sp_ienv(4);

    /* xplore[m] and xprune[n] can be 64-bit; they are allocated separately */
    // isize = ( (2 * panel_size + 3 + NO_MARKER ) * m + n ) * sizeof(int);
    isize = ( (2 * panel_size + 2 + NO_MARKER ) * m ) * sizeof(int);
    // Calculate size needed for integer working memory

    dsize = (m * panel_size +
         NUM_TEMPV(m,panel_size,maxsuper,rowblk)) * sizeof(float);
    // Calculate size needed for floating-point working memory

    if ( Glu->MemModel == SYSTEM ) 
        *iworkptr = (int *) int32Calloc(isize/sizeof(int));
    else
        *iworkptr = (int *) suser_malloc(isize, TAIL, Glu);
    // Allocate memory for integer working array based on memory model

    if ( ! *iworkptr ) {
        fprintf(stderr, "sLUWorkInit: malloc fails for local iworkptr[]\n");
        return (isize + n);
    }
    // Check if integer working array allocation failed

    if ( Glu->MemModel == SYSTEM )
        *dworkptr = (float *) SUPERLU_MALLOC(dsize);
    else {
        *dworkptr = (float *) suser_malloc(dsize, TAIL, Glu);
        // Allocate memory for floating-point working array based on memory model

        if ( NotDoubleAlign(*dworkptr) ) {
            old_ptr = *dworkptr;
            *dworkptr = (float*) DoubleAlign(*dworkptr);
            *dworkptr = (float*) ((double*)*dworkptr - 1);
            extra = (char*)old_ptr - (char*)*dworkptr;
#if ( DEBUGlevel>=1 )
            printf("sLUWorkInit: not aligned, extra %d\n", extra); fflush(stdout);
#endif        
            Glu->stack.top2 -= extra;
            Glu->stack.used += extra;
            // Handle alignment issues and adjust stack usage
        }
    }

    if ( ! *dworkptr ) {
        fprintf(stderr, "malloc fails for local dworkptr[].");
        return (isize + dsize + n);
    }
    // Check if floating-point working array allocation failed

    return 0;
} /* end sLUWorkInit */


/*! \brief Set up pointers for real working arrays.
 */
void
sSetRWork(int m, int panel_size, float *dworkptr,
     float **dense, float **tempv)
{
    float zero = 0.0;

    int maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) ),
        rowblk   = sp_ienv(4);
    *dense = dworkptr;
    *tempv = *dense + panel_size*m;
    // Set pointers to dense and temporary vectors in the working memory
    sfill (*dense, m * panel_size, zero);
    sfill (*tempv, NUM_TEMPV(m,panel_size,maxsuper,rowblk), zero);     
    // Initialize dense and temporary vectors with zero
}
    
/*! \brief Free the working storage used by factor routines.
 */
void sLUWorkFree(int *iwork, float *dwork, GlobalLU_t *Glu)
{
    if ( Glu->MemModel == SYSTEM ) {
        SUPERLU_FREE (iwork);
        SUPERLU_FREE (dwork);
        // Free memory using system-specific deallocation
    } else {
        Glu->stack.used -= (Glu->stack.size - Glu->stack.top2);
        Glu->stack.top2 = Glu->stack.size;
        // Adjust stack usage for custom memory model
/*    sStackCompress(Glu);  */
    }
    
    SUPERLU_FREE (Glu->expanders);    
    Glu->expanders = NULL;
    // Free expanders array and nullify pointer
}

/*! \brief Expand the data structures for L and U during the factorization.
 * 
 * <pre>
 * Return value:   0 - successful return
 *               > 0 - number of bytes allocated when run out of space
 * </pre>
 */
int_t
sLUMemXpand(int jcol,
       int_t next,          /* number of elements currently in the factors */
       MemType mem_type,  /* which type of memory to expand  */
       int_t *maxlen,       /* modified - maximum length of a data structure */
       GlobalLU_t *Glu    /* modified - global LU data structures */
       )
{
    void   *new_mem;
    
#if ( DEBUGlevel>=1 ) 
    // 如果调试级别大于等于1，打印调试信息
    printf("sLUMemXpand[1]: jcol %d, next %lld, maxlen %lld, MemType %d\n",
       jcol, (long long) next, (long long) *maxlen, mem_type);
#endif    

    // 根据内存类型调用sexpand函数进行内存扩展
    if (mem_type == USUB) 
        new_mem = sexpand(maxlen, mem_type, next, 1, Glu);
    else
        new_mem = sexpand(maxlen, mem_type, next, 0, Glu);
    
    // 如果内存扩展失败，则输出错误信息并返回当前内存使用量
    if ( !new_mem ) {
        int_t    nzlmax  = Glu->nzlmax;
        int_t    nzumax  = Glu->nzumax;
        int_t    nzlumax = Glu->nzlumax;
        fprintf(stderr, "Can't expand MemType %d: jcol %d\n", mem_type, jcol);
        return (smemory_usage(nzlmax, nzumax, nzlumax, Glu->n) + Glu->n);
    }

    // 根据不同的内存类型，将扩展后的内存赋值给对应的全局LU数据结构
    switch ( mem_type ) {
      case LUSUP:
        Glu->lusup   = (void *) new_mem;
        Glu->nzlumax = *maxlen;
        break;
      case UCOL:
        Glu->ucol   = (void *) new_mem;
        Glu->nzumax = *maxlen;
        break;
      case LSUB:
        Glu->lsub   = (int_t *) new_mem;
        Glu->nzlmax = *maxlen;
        break;
      case USUB:
        Glu->usub   = (int_t *) new_mem;
        Glu->nzumax = *maxlen;
        break;
      default: break;
    }
    
    // 返回成功状态
    return 0;
}

void
copy_mem_float(int_t howmany, void *old, void *new)
{
    register int_t i;
    float *dold = old;
    float *dnew = new;
    // 将旧内存中的数据复制到新内存中
    for (i = 0; i < howmany; i++) dnew[i] = dold[i];
}

/*! \brief Expand the existing storage to accommodate more fill-ins.
 */
void
*sexpand (
     int_t *prev_len,   /* length used from previous call */
     MemType type,    /* which part of the memory to expand */
     int_t len_to_copy, /* size of the memory to be copied to new store */
     int keep_prev,   /* = 1: use prev_len;
                 = 0: compute new_len to expand */
     GlobalLU_t *Glu  /* modified - global LU data structures */
    )
{
    float    EXPAND = 1.5;
    float    alpha;
    void     *new_mem, *old_mem;
    int_t    new_len, bytes_to_copy;
    int      tries, lword, extra;
    ExpHeader *expanders = Glu->expanders; /* Array of 4 types of memory */

    alpha = EXPAND;

    // 根据当前扩展次数和keep_prev的值决定是否使用之前的长度
    if ( Glu->num_expansions == 0 || keep_prev ) {
        /* First time allocate requested */
        new_len = *prev_len;
    } else {
        // 计算新的长度
        new_len = alpha * *prev_len;
    }
    
    // 根据不同的内存类型选择数据类型字节大小
    if ( type == LSUB || type == USUB ) lword = sizeof(int_t);
    else lword = sizeof(float);

    // 根据内存模型分配新的内存空间
    if ( Glu->MemModel == SYSTEM ) {
        new_mem = (void *) SUPERLU_MALLOC((size_t)new_len * lword);
        ```
    # 如果 Glu 结构体中的 num_expansions 不为 0，则执行以下代码块
    if ( Glu->num_expansions != 0 ) {
        tries = 0;  # 初始化尝试次数为 0
        if ( keep_prev ) {  # 如果 keep_prev 标志为真
        if ( !new_mem ) return (NULL);  # 如果 new_mem 为 NULL，则返回空指针
        } else {
        while ( !new_mem ) {  # 当 new_mem 为 NULL 时循环
            if ( ++tries > 10 ) return (NULL);  # 如果尝试次数超过 10 次，则返回空指针
            alpha = Reduce(alpha);  # 调用 Reduce 函数对 alpha 进行处理
            new_len = alpha * *prev_len;  # 计算新的长度
            new_mem = (void *) SUPERLU_MALLOC((size_t)new_len * lword);  # 分配新的内存空间
        }
        }
        if ( type == LSUB || type == USUB ) {  # 如果 type 是 LSUB 或 USUB
        copy_mem_int(len_to_copy, expanders[type].mem, new_mem);  # 复制整数数据到新内存
        } else {
        copy_mem_float(len_to_copy, expanders[type].mem, new_mem);  # 复制浮点数数据到新内存
        }
        SUPERLU_FREE (expanders[type].mem);  # 释放原来的内存空间
    }
    expanders[type].mem = (void *) new_mem;  # 将 expanders[type].mem 指向新分配的内存

    } else { /* MemModel == USER */  # 如果 MemModel 是 USER

    if ( Glu->num_expansions == 0 ) { /* First time initialization */  # 如果是首次初始化

        new_mem = suser_malloc(new_len * lword, HEAD, Glu);  # 调用 suser_malloc 分配新内存
        if ( NotDoubleAlign(new_mem) &&  # 如果新内存不是双对齐，并且类型为 LUSUP 或 UCOL
        (type == LUSUP || type == UCOL) ) {
        old_mem = new_mem;  # 保存旧的内存指针
        new_mem = (void *)DoubleAlign(new_mem);  # 对新内存进行双对齐处理
        extra = (char*)new_mem - (char*)old_mem;  # 计算额外的偏移量
#if ( DEBUGlevel>=1 )
        printf("expand(): not aligned, extra %d\n", extra);
#endif        
        // 增加堆栈顶部指针和已使用大小，以扩展堆栈空间
        Glu->stack.top1 += extra;
        Glu->stack.used += extra;
        }
        
        // 将新分配的内存赋给对应类型的扩展器
        expanders[type].mem = (void *) new_mem;
        
    } else { /* CASE: num_expansions != 0 */
    
        tries = 0;
        // 计算需要额外扩展的字节数
        extra = (new_len - *prev_len) * lword;
        if ( keep_prev ) {
        // 如果保留上一次扩展的结果，检查堆栈是否已满
        if ( StackFull(extra) ) return (NULL);
        } else {
        // 如果不保留上一次扩展的结果，尝试减少alpha值，直到堆栈有足够空间
        while ( StackFull(extra) ) {
            if ( ++tries > 10 ) return (NULL);
            alpha = Reduce(alpha);
            new_len = alpha * *prev_len;
            extra = (new_len - *prev_len) * lword;        
        }
        }

          /* Need to expand the memory: moving the content after the current MemType
               to make extra room for the current MemType.
                   Memory layout: [ LUSUP || UCOL || LSUB || USUB ]
          */
          // 需要扩展内存：将当前MemType之后的内容移动，为当前MemType腾出额外空间
          if ( type != USUB ) {
        // 计算新的内存起始位置，并复制数据以扩展到此位置
        new_mem = (void*)((char*)expanders[type + 1].mem + extra);
        bytes_to_copy = (char*)Glu->stack.array + Glu->stack.top1
            - (char*)expanders[type + 1].mem;
        user_bcopy(expanders[type+1].mem, new_mem, bytes_to_copy);

        // 根据类型更新相关的指针到新的内存位置
        if ( type < USUB ) {
            Glu->usub = expanders[USUB].mem =
            (void*)((char*)expanders[USUB].mem + extra);
        }
        if ( type < LSUB ) {
            Glu->lsub = expanders[LSUB].mem =
            (void*)((char*)expanders[LSUB].mem + extra);
        }
        if ( type < UCOL ) {
            Glu->ucol = expanders[UCOL].mem =
            (void*)((char*)expanders[UCOL].mem + extra);
        }
        // 更新堆栈顶部指针和已使用大小
        Glu->stack.top1 += extra;
        Glu->stack.used += extra;
        // 特殊情况处理：如果类型为UCOL，则USUB也要增加相同的额外空间
        if ( type == UCOL ) {
            Glu->stack.top1 += extra;   /* Add same amount for USUB */
            Glu->stack.used += extra;
        }
        
        } /* end expansion */

    } /* else ... */
    }

    // 更新扩展器的大小和前一个长度，增加扩展次数计数
    expanders[type].size = new_len;
    *prev_len = new_len;
    if ( Glu->num_expansions ) ++Glu->num_expansions;
    
    // 返回当前MemType的内存指针
    return (void *) expanders[type].mem;
    
} /* sexpand */


/*! \brief Compress the work[] array to remove fragmentation.
 */
void
sStackCompress(GlobalLU_t *Glu)
{
    register int iword, dword, ndim;
    char     *last, *fragment;
    int_t    *ifrom, *ito;
    float   *dfrom, *dto;
    int_t    *xlsub, *lsub, *xusub, *usub, *xlusup;
    float   *ucol, *lusup;
    
    // 设置基本类型大小和维度
    iword = sizeof(int);
    dword = sizeof(float);
    ndim = Glu->n;

    // 初始化各种数组指针
    xlsub  = Glu->xlsub;
    lsub   = Glu->lsub;
    xusub  = Glu->xusub;
    usub   = Glu->usub;
    xlusup = Glu->xlusup;
    ucol   = Glu->ucol;
    lusup  = Glu->lusup;
    
    // 压缩ucol数组，去除碎片化
    dfrom = ucol;
    dto = (float *)((char*)lusup + xlusup[ndim] * dword);
    copy_mem_float(xusub[ndim], dfrom, dto);
    ucol = dto;

    // 压缩lsub数组，去除碎片化
    ifrom = lsub;
    ito = (int_t *) ((char*)ucol + xusub[ndim] * iword);
    copy_mem_int(xlsub[ndim], ifrom, ito);
    lsub = ito;
    
    // 压缩usub数组，去除碎片化
    ifrom = usub;
    ito = (int_t *) ((char*)lsub + xlsub[ndim] * iword);
    # 将长度为 xusub[ndim] 的整数数组从 ifrom 复制到 ito
    copy_mem_int(xusub[ndim], ifrom, ito);
    # 将 ito 赋给 usub
    usub = ito;
    
    # 计算 last 指针，指向 usub 之后 xusub[ndim] 个元素之后的位置
    last = (char*)usub + xusub[ndim] * iword;
    # 计算 fragment 指针，指向 Glu->stack.array 栈顶减去 last 的位置
    fragment = (char*) (((char*)Glu->stack.array + Glu->stack.top1) - last);
    # 减少 Glu->stack.used 的值，减去 fragment 的大小
    Glu->stack.used -= (long int) fragment;
    # 减少 Glu->stack.top1 的值，减去 fragment 的大小
    Glu->stack.top1 -= (long int) fragment;

    # 设置 Glu 结构体的 ucol 字段为 ucol
    Glu->ucol = ucol;
    # 设置 Glu 结构体的 lsub 字段为 lsub
    Glu->lsub = lsub;
    # 设置 Glu 结构体的 usub 字段为 usub
    Glu->usub = usub;
#if ( DEBUGlevel>=1 )
    // 如果调试级别大于等于1，打印当前片段的信息
    printf("sStackCompress: fragment %lld\n", (long long) fragment);
    /* for (last = 0; last < ndim; ++last)
    print_lu_col("After compress:", last, 0);*/
#endif    
    
}

/*! \brief Allocate storage for original matrix A
 */
void
sallocateA(int n, int_t nnz, float **a, int_t **asub, int_t **xa)
{
    // 分配存储空间给矩阵 A 的原始数据
    *a    = (float *) floatMalloc(nnz);
    // 分配存储空间给列下标数组 asub
    *asub = (int_t *) intMalloc(nnz);
    // 分配存储空间给行指针数组 xa
    *xa   = (int_t *) intMalloc(n+1);
}


float *floatMalloc(size_t n)
{
    // 分配 n 个 float 类型的内存空间
    float *buf;
    buf = (float *) SUPERLU_MALLOC(n * (size_t) sizeof(float)); 
    // 检查分配是否成功
    if ( !buf ) {
    ABORT("SUPERLU_MALLOC failed for buf in floatMalloc()\n");
    }
    return (buf);
}

float *floatCalloc(size_t n)
{
    // 分配 n 个 float 类型的内存空间并初始化为 0
    float *buf;
    register size_t i;
    float zero = 0.0;
    buf = (float *) SUPERLU_MALLOC(n * (size_t) sizeof(float));
    // 检查分配是否成功
    if ( !buf ) {
    ABORT("SUPERLU_MALLOC failed for buf in floatCalloc()\n");
    }
    // 初始化分配的内存为 0
    for (i = 0; i < n; ++i) buf[i] = zero;
    return (buf);
}


int_t smemory_usage(const int_t nzlmax, const int_t nzumax,
          const int_t nzlumax, const int n)
{
    // 计算内存使用量的估计
    register int iword, liword, dword;

    iword   = sizeof(int);      // int 类型所占字节数
    liword  = sizeof(int_t);    // int_t 类型所占字节数
    dword   = sizeof(float);    // float 类型所占字节数
    
    // 返回内存使用量的估计值
    return (10 * n * iword +
        nzlmax * liword + nzumax * (liword + dword) + nzlumax * dword);
}
```