# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cmemory.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cmemory.c
 * \brief Memory details
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 * </pre>
 */
#include "slu_cdefs.h"

/* Internal prototypes */
void  *cexpand (int_t *, MemType, int_t, int, GlobalLU_t *);
int   cLUWorkInit (int, int, int, int **, singlecomplex **, GlobalLU_t *);
void  copy_mem_singlecomplex (int_t, void *, void *);
void  cStackCompress (GlobalLU_t *);
void  cSetupSpace (void *, int_t, GlobalLU_t *);
void  *cuser_malloc (int, int, GlobalLU_t *);
void  cuser_free (int, int, GlobalLU_t *);

/* External prototypes (in memory.c - prec-independent) */
extern void    copy_mem_int    (int, void *, void *);
extern void    user_bcopy      (char *, char *, int);


/* Macros to manipulate stack */
#define StackFull(x)         ( x + Glu->stack.used >= Glu->stack.size )
#define NotDoubleAlign(addr) ( (intptr_t)addr & 7 )
#define DoubleAlign(addr)    ( ((intptr_t)addr + 7) & ~7L )    
#define TempSpace(m, w)      ( (2*w + 4 + NO_MARKER) * m * sizeof(int) + \
                  (w + 1) * m * sizeof(singlecomplex) )
#define Reduce(alpha)        ((alpha + 1) / 2)  /* i.e. (alpha-1)/2 + 1 */


/*! \brief Setup the memory model to be used for factorization.
 *  
 *    lwork = 0: use system malloc;
 *    lwork > 0: use user-supplied work[] space.
 */
void cSetupSpace(void *work, int_t lwork, GlobalLU_t *Glu)
{
    if ( lwork == 0 ) {
    Glu->MemModel = SYSTEM; /* 使用系统提供的 malloc/free */
    } else if ( lwork > 0 ) {
    Glu->MemModel = USER;   /* 使用用户提供的工作空间 */
    Glu->stack.used = 0;    /* 初始化堆栈使用量 */
    Glu->stack.top1 = 0;    /* 初始化堆栈顶部指针1 */
    Glu->stack.top2 = (lwork/4)*4; /* 必须是字地址对齐 */
    Glu->stack.size = Glu->stack.top2; /* 设置堆栈大小 */
    Glu->stack.array = (void *) work; /* 将工作空间指针赋给堆栈数组 */
    }
}


/*! \brief Allocate memory from the stack for the user.
 *
 *  \param bytes Number of bytes to allocate.
 *  \param which_end Specifies whether allocation is from the head or tail of the stack.
 *  \param Glu GlobalLU_t structure containing stack information.
 *
 *  \return Pointer to allocated memory, or NULL if allocation fails.
 */
void *cuser_malloc(int bytes, int which_end, GlobalLU_t *Glu)
{
    void *buf;
    
    if ( StackFull(bytes) ) return (NULL); /* 如果堆栈空间不足，则返回空指针 */

    if ( which_end == HEAD ) {
    buf = (char*) Glu->stack.array + Glu->stack.top1; /* 从堆栈头部分配内存 */
    Glu->stack.top1 += bytes; /* 更新堆栈顶部指针1 */
    } else {
    Glu->stack.top2 -= bytes; /* 从堆栈尾部分配内存 */
    buf = (char*) Glu->stack.array + Glu->stack.top2; /* 更新堆栈顶部指针2 */
    }
    
    Glu->stack.used += bytes; /* 更新堆栈使用量 */
    return buf; /* 返回分配的内存指针 */
}


/*! \brief Free memory allocated from the stack.
 *
 *  \param bytes Number of bytes to free.
 *  \param which_end Specifies whether to free memory from the head or tail of the stack.
 *  \param Glu GlobalLU_t structure containing stack information.
 */
void cuser_free(int bytes, int which_end, GlobalLU_t *Glu)
{
    if ( which_end == HEAD ) {
    Glu->stack.top1 -= bytes; /* 从堆栈头部释放内存 */
    } else {
    Glu->stack.top2 += bytes; /* 从堆栈尾部释放内存 */
    }
    Glu->stack.used -= bytes; /* 更新堆栈使用量 */
}

/*!
 * Calculate memory usage
 *
 * \param mem_usage consists of the following fields:
 *    - <tt>for_lu (float)</tt>
 *      The amount of space used in bytes for the L\\U data structures.
 *    - <tt>total_needed (float)</tt>
 *      The amount of space needed in bytes to perform factorization.
 */
/*!
 * Calculate memory usage for LU factorization.
 *
 * \param L Pointer to SuperMatrix L, contains the factors of the lower triangular matrix.
 * \param U Pointer to SuperMatrix U, contains the factors of the upper triangular matrix.
 * \param mem_usage Pointer to mem_usage_t struct where memory usage information will be stored.
 *
 * \return 0 indicating successful completion.
 */
int cQuerySpace(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage)
{
    SCformat *Lstore;           /* Pointer to the SCformat structure of L */
    NCformat *Ustore;           /* Pointer to the NCformat structure of U */
    register int n,             /* Number of columns in L */
                 iword,         /* Size of integer in bytes */
                 dword,         /* Size of double complex in bytes */
                 panel_size = sp_ienv(1); /* Panel size for factorization */

    Lstore = L->Store;          /* Store SCformat structure of L */
    Ustore = U->Store;          /* Store NCformat structure of U */
    n = L->ncol;                /* Number of columns in L */
    iword = sizeof(int);        /* Size of integer in bytes */
    dword = sizeof(singlecomplex); /* Size of single complex in bytes */

    /* Calculate memory usage for LU factors */
    mem_usage->for_lu = (float)( (4.0 * n + 3.0) * iword +
                                 Lstore->nzval_colptr[n] * dword +
                                 Lstore->rowind_colptr[n] * iword );
    mem_usage->for_lu += (float)( (n + 1.0) * iword +
                 Ustore->colptr[n] * (dword + iword) );

    /* Calculate total needed working storage to support factorization */
    mem_usage->total_needed = mem_usage->for_lu +
    (float)( (2.0 * panel_size + 4.0 + NO_MARKER) * n * iword +
        (panel_size + 1.0) * n * dword );

    return 0;
} /* cQuerySpace */


/*!
 * Calculate memory usage for ILU (Incomplete LU) factorization.
 *
 * \param L Pointer to SuperMatrix L, contains the factors of the lower triangular matrix.
 * \param U Pointer to SuperMatrix U, contains the factors of the upper triangular matrix.
 * \param mem_usage Pointer to mem_usage_t struct where memory usage information will be stored.
 *
 * \return 0 indicating successful completion.
 */
int ilu_cQuerySpace(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage)
{
    SCformat *Lstore;           /* Pointer to the SCformat structure of L */
    NCformat *Ustore;           /* Pointer to the NCformat structure of U */
    register int n,             /* Number of columns in L */
                 panel_size = sp_ienv(1); /* Panel size for factorization */
    register float iword,       /* Size of integer in bytes */
                   dword;       /* Size of double in bytes */

    Lstore = L->Store;          /* Store SCformat structure of L */
    Ustore = U->Store;          /* Store NCformat structure of U */
    n = L->ncol;                /* Number of columns in L */
    iword = sizeof(int);        /* Size of integer in bytes */
    dword = sizeof(double);     /* Size of double in bytes */

    /* Calculate memory usage for ILU factors */
    mem_usage->for_lu = (float)( (4.0f * n + 3.0f) * iword +
                 Lstore->nzval_colptr[n] * dword +
                 Lstore->rowind_colptr[n] * iword );
    mem_usage->for_lu += (float)( (n + 1.0f) * iword +
                 Ustore->colptr[n] * (dword + iword) );

    /* Calculate total needed working storage to support ILU factorization */
    mem_usage->total_needed = mem_usage->for_lu +
    (float)( (2.0f * panel_size + 9.0f + NO_MARKER) * n * iword +
        (panel_size + 1.0f) * n * dword );

    return 0;
} /* ilu_cQuerySpace */


/*!
 * Initialize memory allocation for LU factorization data structures.
 *
 * \param fact Factorization type (not used in the function).
 * \param work Pointer to allocated memory for work arrays (not used in the function).
 * \param lwork Size of work array (not used in the function).
 * \param m Number of rows in the matrix A (not used in the function).
 * \param n Number of columns in the matrix A (not used in the function).
 * \param annz Number of nonzeros in the matrix A (not used in the function).
 * \param panel_size Panel size for factorization.
 * \param fill_ratio Fill ratio for memory estimation.
 * \param L Pointer to SuperMatrix L, contains the factors of the lower triangular matrix.
 * \param U Pointer to SuperMatrix U, contains the factors of the upper triangular matrix.
 * \param Glu Pointer to GlobalLU_t structure (not used in the function).
 * \param iwork Pointer to integer work array.
 * \param dwork Pointer to single complex work array.
 *
 * \return Estimated or actual amount of memory required for LU factorization.
 */
int_t
cLUMemInit(fact_t fact, void *work, int_t lwork, int m, int n, int_t annz,
      int panel_size, float fill_ratio, SuperMatrix *L, SuperMatrix *U,
          GlobalLU_t *Glu, int **iwork, singlecomplex **dwork)
{
    int      info,              /* Information status */
             iword,             /* Size of integer in bytes */
             dword;             /* Size of double complex in bytes */
    SCformat *Lstore;           /* Pointer to the SCformat structure of L */
    NCformat *Ustore;           /* Pointer to the NCformat structure of U */
    int      *xsup, *supno;     /* Pointers to arrays for supernode structure */
    int_t    *lsub, *xlsub;     /* Pointers to arrays for L structure */
    singlecomplex   *lusup;     /* Pointer to array for L supernodes */
    int_t    *xlusup;           /* Pointer to array for L supernodes */
    singlecomplex   *ucol;      /* Pointer to array for U column structure */
    int_t    *usub, *xusub;     /* Pointers to arrays for U structure */
    int_t    nzlmax, nzumax, nzlumax; /* Maximum nonzero counts for L, U, and LU */

    /*! \brief Allocate storage for the data structures common to all factor routines.
     *
     * <pre>
     * For those unpredictable size, estimate as fill_ratio * nnz(A).
     * Return value:
     *     If lwork = -1, return the estimated amount of space required, plus n;
     *     otherwise, return the amount of space actually allocated when
     *     memory allocation failure occurred.
     * </pre> 
     */

    Lstore = L->Store;          /* Store SCformat structure of L */
    Ustore = U->Store;          /* Store NCformat structure of U */
    iword = sizeof(int);        /* Size of integer in bytes */
    dword = sizeof(double);     /* Size of double in bytes */

    /* Memory initialization for LU factorization */
    /* Note: Detailed allocation and initialization are not commented due to the complexity and specificity of the operation. */

    return 0;
}
    # 定义整数变量 iword，并赋值为 sizeof(int) 的结果
    iword     = sizeof(int);
    # 定义单精度复数变量 dword，并赋值为 sizeof(singlecomplex) 的结果
    dword     = sizeof(singlecomplex);
    # 将参数 n 赋值给结构体 Glu 的成员变量 n
    Glu->n    = n;
    # 将数值 0 赋值给结构体 Glu 的成员变量 num_expansions
    Glu->num_expansions = 0;

    # 为结构体 Glu 的成员变量 expanders 分配内存空间，大小为 NO_MEMTYPE * sizeof(ExpHeader)
    Glu->expanders = (ExpHeader *) SUPERLU_MALLOC( NO_MEMTYPE *
                                                     sizeof(ExpHeader) );
    # 如果内存分配失败，则输出错误信息并终止程序
    if ( !Glu->expanders ) ABORT("SUPERLU_MALLOC fails for expanders");
    
    # 如果不是同一模式和同一行排列的情况下
    if ( fact != SamePattern_SameRowPerm ) {
        /* Guess for L\U factors */
        # 计算估计值 nzumax、nzlumax 和 nzlmax
        nzumax = nzlumax = nzlmax = fill_ratio * annz;
        //nzlmax = SUPERLU_MAX(1, fill_ratio/4.) * annz;

        # 如果 lwork 的值为 -1，则返回计算出的内存需求
        if ( lwork == -1 ) {
            return ( GluIntArray(n) * iword + TempSpace(m, panel_size)
                + (nzlmax+nzumax)*iword + (nzlumax+nzumax)*dword + n );
        } else {
            # 设置工作空间的大小和相关参数
            cSetupSpace(work, lwork, Glu);
        }
    }
#if ( PRNTlevel >= 1 )
    // 如果打印级别大于等于1，则打印以下消息，用于调试输出
    printf("cLUMemInit() called: fill_ratio %.0f, nzlmax %lld, nzumax %lld\n", 
           fill_ratio, (long long) nzlmax, (long long) nzumax);
    // 刷新标准输出流，确保消息即时显示
    fflush(stdout);
#endif

/* Integer pointers for L\U factors */
// 为L/U因子分配整数指针
if ( Glu->MemModel == SYSTEM ) {
    // 如果内存模型为SYSTEM，则使用系统内存分配函数分配以下指针
    xsup   = int32Malloc(n+1);   // 整型数组xsuo，大小为n+1
    supno  = int32Malloc(n+1);   // 整型数组supno，大小为n+1
    xlsub  = intMalloc(n+1);     // 整型数组xlsub，大小为n+1
    xlusup = intMalloc(n+1);     // 整型数组xlusup，大小为n+1
    xusub  = intMalloc(n+1);     // 整型数组xusub，大小为n+1
} else {
    // 否则，使用用户自定义的内存分配函数cuser_malloc分配以下指针
    xsup   = (int *)cuser_malloc((n+1) * iword, HEAD, Glu);   // 整型数组xsuo，大小为(n+1)*iword
    supno  = (int *)cuser_malloc((n+1) * iword, HEAD, Glu);   // 整型数组supno，大小为(n+1)*iword
    xlsub  = cuser_malloc((n+1) * iword, HEAD, Glu);          // 整型数组xlsub，大小为(n+1)*iword
    xlusup = cuser_malloc((n+1) * iword, HEAD, Glu);          // 整型数组xlusup，大小为(n+1)*iword
    xusub  = cuser_malloc((n+1) * iword, HEAD, Glu);          // 整型数组xusub，大小为(n+1)*iword
}

lusup = (singlecomplex *) cexpand( &nzlumax, LUSUP, 0, 0, Glu );
// 扩展复数数组lusup，大小为nzlumax，类型为LUSUP，使用Glu管理的内存
ucol  = (singlecomplex *) cexpand( &nzumax, UCOL, 0, 0, Glu );
// 扩展复数数组ucol，大小为nzumax，类型为UCOL，使用Glu管理的内存
lsub  = (int_t *) cexpand( &nzlmax, LSUB, 0, 0, Glu );
// 扩展整型数组lsub，大小为nzlmax，类型为LSUB，使用Glu管理的内存
usub  = (int_t *) cexpand( &nzumax, USUB, 0, 1, Glu );
// 扩展整型数组usub，大小为nzumax，类型为USUB，使用Glu管理的内存

while ( !lusup || !ucol || !lsub || !usub ) {
    // 如果有任何一个数组指针为NULL，则执行以下循环体
    if ( Glu->MemModel == SYSTEM ) {
        // 如果内存模型为SYSTEM，则释放以下指针的内存
        SUPERLU_FREE(lusup); 
        SUPERLU_FREE(ucol); 
        SUPERLU_FREE(lsub); 
        SUPERLU_FREE(usub);
    } else {
        // 否则，使用用户自定义的内存释放函数cuser_free释放内存
        cuser_free((nzlumax+nzumax)*dword+(nzlmax+nzumax)*iword,
                   HEAD, Glu);
    }
    // 减半所有计数器
    nzlumax /= 2;
    nzumax /= 2;
    nzlmax /= 2;
    // 如果减半后的nzlumax小于annz，则输出错误信息并返回相应的内存使用量
    if ( nzlumax < annz ) {
        printf("Not enough memory to perform factorization.\n");
        return (cmemory_usage(nzlmax, nzumax, nzlumax, n) + n);
    }
#if ( PRNTlevel >= 1)
    // 如果打印级别大于等于1，则打印以下消息，用于调试输出
    printf("cLUMemInit() reduce size: nzlmax %ld, nzumax %ld\n", 
           (long) nzlmax, (long) nzumax);
    // 刷新标准输出流，确保消息即时显示
    fflush(stdout);
#endif
    // 重新扩展所有数组指针
    lusup = (singlecomplex *) cexpand( &nzlumax, LUSUP, 0, 0, Glu );
    ucol  = (singlecomplex *) cexpand( &nzumax, UCOL, 0, 0, Glu );
    lsub  = (int_t *) cexpand( &nzlmax, LSUB, 0, 0, Glu );
    usub  = (int_t *) cexpand( &nzumax, USUB, 0, 1, Glu );
}

} else {
/* fact == SamePattern_SameRowPerm */
// 如果fact == SamePattern_SameRowPerm，则执行以下代码块
Lstore   = L->Store;            // 将L的存储结构赋值给Lstore
Ustore   = U->Store;            // 将U的存储结构赋值给Ustore
xsup     = Lstore->sup_to_col;  // 将L的sup_to_col数组赋值给xsup
supno    = Lstore->col_to_sup;  // 将L的col_to_sup数组赋值给supno
xlsub    = Lstore->rowind_colptr;  // 将L的rowind_colptr数组赋值给xlsub
xlusup   = Lstore->nzval_colptr;   // 将L的nzval_colptr数组赋值给xlusup
xusub    = Ustore->colptr;      // 将U的colptr数组赋值给xusub
nzlmax   = Glu->nzlmax;         // 将Glu的nzlmax赋值给nzlmax
nzumax   = Glu->nzumax;         // 将Glu的nzumax赋值给nzumax
nzlumax  = Glu->nzlumax;        // 将Glu的nzlumax赋值给nzlumax

if ( lwork == -1 ) {
    // 如果lwork为-1，则返回需要的内存量的估计
    return ( GluIntArray(n) * iword + TempSpace(m, panel_size)
             + (nzlmax+nzumax)*iword + (nzlumax+nzumax)*dword + n );
} else if ( lwork == 0 ) {
    // 如果lwork为0，则设置内存模型为SYSTEM
    Glu->MemModel = SYSTEM;
} else {
    // 否则，设置内存模型为USER，并设置栈顶位置
    Glu->MemModel = USER;
    Glu->stack.top2 = (lwork/4)*4;  // 必须是字（word）可寻址的
    Glu->stack.size = Glu->stack.top2;  // 栈的大小为栈顶位置
}

lsub  = Glu->expanders[LSUB].mem  = Lstore->rowind;  // 将L的rowind数组赋值给lsub
lusup = Glu->expanders[LUSUP].mem = Lstore->nzval;   // 将L的nzval数组赋值给lusup
    usub  = Glu->expanders[USUB].mem  = Ustore->rowind;
    # 设置指向Ustore的行指针数组的起始地址，并将其赋值给Glu结构体中USUB扩展器的内存指针
    ucol  = Glu->expanders[UCOL].mem  = Ustore->nzval;;
    # 设置指向Ustore的非零元值数组的起始地址，并将其赋值给Glu结构体中UCOL扩展器的内存指针
    Glu->expanders[LSUB].size         = nzlmax;
    # 设置Glu结构体中LSUB扩展器的大小为nzlmax
    Glu->expanders[LUSUP].size        = nzlumax;
    # 设置Glu结构体中LUSUP扩展器的大小为nzlumax
    Glu->expanders[USUB].size         = nzumax;
    # 设置Glu结构体中USUB扩展器的大小为nzumax
    Glu->expanders[UCOL].size         = nzumax;    
    # 设置Glu结构体中UCOL扩展器的大小为nzumax

    Glu->xsup    = xsup;
    # 设置Glu结构体中xsup字段为xsup的值
    Glu->supno   = supno;
    # 设置Glu结构体中supno字段为supno的值
    Glu->lsub    = lsub;
    # 设置Glu结构体中lsub字段为lsub的值
    Glu->xlsub   = xlsub;
    # 设置Glu结构体中xlsub字段为xlsub的值
    Glu->lusup   = (void *) lusup;
    # 将lusup的地址强制类型转换为void指针，并设置为Glu结构体中lusup字段的值
    Glu->xlusup  = xlusup;
    # 设置Glu结构体中xlusup字段为xlusup的值
    Glu->ucol    = (void *) ucol;
    # 将ucol的地址强制类型转换为void指针，并设置为Glu结构体中ucol字段的值
    Glu->usub    = usub;
    # 设置Glu结构体中usub字段为usub的值
    Glu->xusub   = xusub;
    # 设置Glu结构体中xusub字段为xusub的值
    Glu->nzlmax  = nzlmax;
    # 设置Glu结构体中nzlmax字段为nzlmax的值
    Glu->nzumax  = nzumax;
    # 设置Glu结构体中nzumax字段为nzumax的值
    Glu->nzlumax = nzlumax;
    # 设置Glu结构体中nzlumax字段为nzlumax的值
    
    info = cLUWorkInit(m, n, panel_size, iwork, dwork, Glu);
    # 调用cLUWorkInit函数，初始化LU分解工作区，返回信息存储在info中
    if ( info )
        # 如果info非零，表示出现错误
        return ( info + cmemory_usage(nzlmax, nzumax, nzlumax, n) + n);
        # 返回错误码，加上内存使用量和n
    ++Glu->num_expansions;
    # 如果成功初始化，则增加Glu结构体中num_expansions字段的值
    return 0;
    # 返回成功初始化的标志
} /* cLUMemInit */

/*! \brief Allocate known working storage.
 * Returns 0 if success, otherwise
 * returns the number of bytes allocated so far when failure occurred.
 */
int
cLUWorkInit(int m, int n, int panel_size, int **iworkptr, 
            singlecomplex **dworkptr, GlobalLU_t *Glu)
{
    int    isize, dsize, extra;
    singlecomplex *old_ptr;
    int    maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) ),
           rowblk   = sp_ienv(4);

    /* xplore[m] and xprune[n] can be 64-bit; they are allocated separately */
    //isize = ( (2 * panel_size + 3 + NO_MARKER ) * m + n ) * sizeof(int);
    isize = ( (2 * panel_size + 2 + NO_MARKER ) * m ) * sizeof(int);  // 计算整型工作数组的大小
    dsize = (m * panel_size +
         NUM_TEMPV(m,panel_size,maxsuper,rowblk)) * sizeof(singlecomplex);  // 计算单精度复数工作数组的大小
    
    if ( Glu->MemModel == SYSTEM ) 
    *iworkptr = (int *) int32Calloc(isize/sizeof(int));  // 根据内存模型分配整型工作数组内存
    else
    *iworkptr = (int *) cuser_malloc(isize, TAIL, Glu);  // 使用用户定义的内存分配函数分配整型工作数组内存
    if ( ! *iworkptr ) {
    fprintf(stderr, "cLUWorkInit: malloc fails for local iworkptr[]\n");  // 分配失败时打印错误信息
    return (isize + n);
    }

    if ( Glu->MemModel == SYSTEM )
    *dworkptr = (singlecomplex *) SUPERLU_MALLOC(dsize);  // 根据内存模型分配单精度复数工作数组内存
    else {
    *dworkptr = (singlecomplex *) cuser_malloc(dsize, TAIL, Glu);  // 使用用户定义的内存分配函数分配单精度复数工作数组内存
    if ( NotDoubleAlign(*dworkptr) ) {
        old_ptr = *dworkptr;
        *dworkptr = (singlecomplex*) DoubleAlign(*dworkptr);  // 对齐内存地址到双精度边界
        *dworkptr = (singlecomplex*) ((double*)*dworkptr - 1);  // 调整地址以存储额外信息
        extra = (char*)old_ptr - (char*)*dworkptr;  // 计算额外存储的字节数
#if ( DEBUGlevel>=1 )
        printf("cLUWorkInit: not aligned, extra %d\n", extra); fflush(stdout);  // 输出调试信息，表示内存未对齐
#endif        
        Glu->stack.top2 -= extra;  // 调整栈顶指针和使用的内存大小
        Glu->stack.used += extra;
    }
    }
    if ( ! *dworkptr ) {
    fprintf(stderr, "malloc fails for local dworkptr[].");  // 分配失败时打印错误信息
    return (isize + dsize + n);
    }
    
    return 0;  // 返回成功
} /* end cLUWorkInit */


/*! \brief Set up pointers for real working arrays.
 */
void
cSetRWork(int m, int panel_size, singlecomplex *dworkptr,
     singlecomplex **dense, singlecomplex **tempv)
{
    singlecomplex zero = {0.0, 0.0};

    int maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) ),
        rowblk   = sp_ienv(4);
    *dense = dworkptr;  // 设置稠密矩阵工作数组的指针
    *tempv = *dense + panel_size*m;  // 设置临时向量工作数组的指针
    cfill (*dense, m * panel_size, zero);  // 使用零填充稠密矩阵工作数组
    cfill (*tempv, NUM_TEMPV(m,panel_size,maxsuper,rowblk), zero);  // 使用零填充临时向量工作数组     
}
    
/*! \brief Free the working storage used by factor routines.
 */
void cLUWorkFree(int *iwork, singlecomplex *dwork, GlobalLU_t *Glu)
{
    if ( Glu->MemModel == SYSTEM ) {
    SUPERLU_FREE (iwork);  // 释放系统内存模型下的整型工作数组内存
    SUPERLU_FREE (dwork);  // 释放系统内存模型下的单精度复数工作数组内存
    } else {
    Glu->stack.used -= (Glu->stack.size - Glu->stack.top2);  // 调整栈顶指针和使用的内存大小
    Glu->stack.top2 = Glu->stack.size;
/*    cStackCompress(Glu);  */  // 压缩栈
    }
    
    SUPERLU_FREE (Glu->expanders);  // 释放扩展器内存    
    Glu->expanders = NULL;
}

/*! \brief Expand the data structures for L and U during the factorization.
 * 
 * <pre>
 * Return value:   0 - successful return
 *               > 0 - number of bytes allocated when run out of space
 * </pre>
 */
int_t
/*
   函数 cLUMemXpand 用于根据指定的条件扩展全局LU因子数据结构中的内存。
   参数解释：
   - jcol: 当前列的索引
   - next: 当前因子中元素的数量
   - mem_type: 需要扩展的内存类型
   - *maxlen: 最大数据结构长度的指针（将被修改）
   - *Glu: 全局LU数据结构的指针（将被修改）
*/
void cLUMemXpand(int jcol,
       int_t next,          /* number of elements currently in the factors */
       MemType mem_type,  /* which type of memory to expand  */
       int_t *maxlen,       /* modified - maximum length of a data structure */
       GlobalLU_t *Glu    /* modified - global LU data structures */
       )
{
    void   *new_mem;
    
#if ( DEBUGlevel>=1 ) 
    printf("cLUMemXpand[1]: jcol %d, next %lld, maxlen %lld, MemType %d\n",
       jcol, (long long) next, (long long) *maxlen, mem_type);
#endif    

    if (mem_type == USUB) 
        new_mem = cexpand(maxlen, mem_type, next, 1, Glu);
    else
        new_mem = cexpand(maxlen, mem_type, next, 0, Glu);
    
    if ( !new_mem ) {
        int_t    nzlmax  = Glu->nzlmax;
        int_t    nzumax  = Glu->nzumax;
        int_t    nzlumax = Glu->nzlumax;
        fprintf(stderr, "Can't expand MemType %d: jcol %d\n", mem_type, jcol);
        return (cmemory_usage(nzlmax, nzumax, nzlumax, Glu->n) + Glu->n);
    }

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
    
    return 0;
}


/*
   函数 copy_mem_singlecomplex 用于复制单精度复数数组的内容到新的内存空间。
   参数解释：
   - howmany: 要复制的元素数量
   - old: 原始数据的指针
   - new: 新内存空间的指针
*/
void copy_mem_singlecomplex(int_t howmany, void *old, void *new)
{
    register int_t i;
    singlecomplex *dold = old;
    singlecomplex *dnew = new;
    for (i = 0; i < howmany; i++) dnew[i] = dold[i];
}


/*
   函数 cexpand 用于扩展指定类型的内存存储空间。
   参数解释：
   - *prev_len: 上一次调用时使用的长度
   - type: 需要扩展的内存类型
   - len_to_copy: 需要复制到新存储的内存大小
   - keep_prev: 是否保持前一次的长度
   - *Glu: 全局LU数据结构的指针（将被修改）
*/
void *cexpand (
     int_t *prev_len,   /* length used from previous call */
     MemType type,    /* which part of the memory to expand */
     int_t len_to_copy, /* size of the memory to be copied to new store */
     int keep_prev,   /* = 1: use prev_len; = 0: compute new_len to expand */
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

    if ( Glu->num_expansions == 0 || keep_prev ) {
        /* First time allocate requested */
        new_len = *prev_len;
    } else {
        new_len = alpha * *prev_len;
    }
    
    if ( type == LSUB || type == USUB ) lword = sizeof(int_t);
    else lword = sizeof(singlecomplex);

    if ( Glu->MemModel == SYSTEM ) {
        new_mem = (void *) SUPERLU_MALLOC((size_t)new_len * lword);
    # 如果Glu->num_expansions不为0，则执行以下操作
    if ( Glu->num_expansions != 0 ) {
        tries = 0;  # 初始化尝试次数为0
        if ( keep_prev ) {  # 如果keep_prev为真
            if ( !new_mem ) return (NULL);  # 如果new_mem为空，则返回空指针
        } else {
            # 如果new_mem不为空，或者在尝试次数不超过10次之前
            while ( !new_mem ) {
                if ( ++tries > 10 ) return (NULL);  # 如果尝试次数超过10次，则返回空指针
                alpha = Reduce(alpha);  # 减少alpha的值（根据上下文中Reduce函数的作用）
                new_len = alpha * *prev_len;  # 计算新的长度
                new_mem = (void *) SUPERLU_MALLOC((size_t)new_len * lword);  # 分配新内存空间
            }
        }
        # 如果type为LSUB或USUB，则复制整数数据；否则复制单精度复数数据
        if ( type == LSUB || type == USUB ) {
            copy_mem_int(len_to_copy, expanders[type].mem, new_mem);
        } else {
            copy_mem_singlecomplex(len_to_copy, expanders[type].mem, new_mem);
        }
        SUPERLU_FREE (expanders[type].mem);  # 释放旧的内存空间
    }
    expanders[type].mem = (void *) new_mem;  # 更新expanders数组中的内存指针
    
    } else { /* MemModel == USER */
    
    if ( Glu->num_expansions == 0 ) {  # 如果Glu->num_expansions为0，即第一次初始化
        
        # 根据new_len和lword分配内存，HEAD和Glu是参数
        new_mem = cuser_malloc(new_len * lword, HEAD, Glu);
        
        # 如果new_mem不是双对齐，并且type为LUSUP或UCOL
        if ( NotDoubleAlign(new_mem) &&
             (type == LUSUP || type == UCOL) ) {
            old_mem = new_mem;  # 保存旧的内存指针
            new_mem = (void *)DoubleAlign(new_mem);  # 对new_mem进行双对齐
            extra = (char*)new_mem - (char*)old_mem;  # 计算额外的偏移量
#if ( DEBUGlevel>=1 )
        printf("expand(): not aligned, extra %d\n", extra);
#endif        
        // 假设 DEBUGlevel 大于等于 1 时打印调试信息，显示额外的 extra 值

        // 更新堆栈的顶部和已使用的元素数量
        Glu->stack.top1 += extra;
        Glu->stack.used += extra;
        }

        // 将新分配的内存指针存储到 expanders[type] 的 mem 字段中
        expanders[type].mem = (void *) new_mem;
        
    } else { /* CASE: num_expansions != 0 */
    
        tries = 0;
        // 计算需要额外分配的内存空间
        extra = (new_len - *prev_len) * lword;
        if ( keep_prev ) {
        // 如果 keep_prev 为真且堆栈已满，则返回空指针
        if ( StackFull(extra) ) return (NULL);
        } else {
        // 当 keep_prev 为假时，尝试解决堆栈满的情况
        while ( StackFull(extra) ) {
            // 尝试次数超过 10 次则返回空指针
            if ( ++tries > 10 ) return (NULL);
            // 减小 alpha 的值并重新计算 new_len
            alpha = Reduce(alpha);
            new_len = alpha * *prev_len;
            extra = (new_len - *prev_len) * lword;        
        }
        }

          /* 需要扩展内存：将当前 MemType 后面的内容移动，为当前 MemType 腾出额外空间。
               内存布局：[ LUSUP || UCOL || LSUB || USUB ]
          */
          // 如果 type 不等于 USUB，则进行内存扩展操作
          if ( type != USUB ) {
        // 计算新的内存指针位置，并拷贝需要移动的字节数
        new_mem = (void*)((char*)expanders[type + 1].mem + extra);
        bytes_to_copy = (char*)Glu->stack.array + Glu->stack.top1
            - (char*)expanders[type + 1].mem;
        // 调用用户定义的拷贝函数，将数据从 expanders[type+1].mem 移动到 new_mem
        user_bcopy(expanders[type+1].mem, new_mem, bytes_to_copy);

        // 如果 type 小于 USUB，则更新 Glu->usub 和 expanders[USUB].mem 的内存指针
        if ( type < USUB ) {
            Glu->usub = expanders[USUB].mem =
            (void*)((char*)expanders[USUB].mem + extra);
        }
        // 如果 type 小于 LSUB，则更新 Glu->lsub 和 expanders[LSUB].mem 的内存指针
        if ( type < LSUB ) {
            Glu->lsub = expanders[LSUB].mem =
            (void*)((char*)expanders[LSUB].mem + extra);
        }
        // 如果 type 小于 UCOL，则更新 Glu->ucol 和 expanders[UCOL].mem 的内存指针
        if ( type < UCOL ) {
            Glu->ucol = expanders[UCOL].mem =
            (void*)((char*)expanders[UCOL].mem + extra);
        }
        // 更新堆栈的顶部和已使用的元素数量
        Glu->stack.top1 += extra;
        Glu->stack.used += extra;
        // 如果 type 为 UCOL，则再次增加堆栈的顶部和已使用的元素数量
        if ( type == UCOL ) {
            Glu->stack.top1 += extra;   /* Add same amount for USUB */
            Glu->stack.used += extra;
        }
        
        } /* end expansion */

    } /* else ... */
    }

    // 更新 expanders[type] 的大小和 *prev_len 的值
    expanders[type].size = new_len;
    *prev_len = new_len;
    // 如果 Glu->num_expansions 存在，则增加其值
    if ( Glu->num_expansions ) ++Glu->num_expansions;
    
    // 返回 expanders[type] 的内存指针
    return (void *) expanders[type].mem;
    
} /* cexpand */


/*! \brief Compress the work[] array to remove fragmentation.
 */
void
cStackCompress(GlobalLU_t *Glu)
{
    register int iword, dword, ndim;
    char     *last, *fragment;
    int_t    *ifrom, *ito;
    singlecomplex   *dfrom, *dto;
    int_t    *xlsub, *lsub, *xusub, *usub, *xlusup;
    singlecomplex   *ucol, *lusup;
    
    // 初始化变量 iword、dword 和 ndim
    iword = sizeof(int);
    dword = sizeof(singlecomplex);
    ndim = Glu->n;

    // 获取 GlobalLU_t 结构体中的指针和数组
    xlsub  = Glu->xlsub;
    lsub   = Glu->lsub;
    xusub  = Glu->xusub;
    usub   = Glu->usub;
    xlusup = Glu->xlusup;
    ucol   = Glu->ucol;
    lusup  = Glu->lusup;
    
    // 将 ucol 的内容复制到 dto 指针指向的内存区域
    dfrom = ucol;
    dto = (singlecomplex *)((char*)lusup + xlusup[ndim] * dword);
    copy_mem_singlecomplex(xusub[ndim], dfrom, dto);
    ucol = dto;

    // 将 lsub 的内容复制到 ito 指针指向的内存区域
    ifrom = lsub;
    ito = (int_t *) ((char*)ucol + xusub[ndim] * iword);
    copy_mem_int(xlsub[ndim], ifrom, ito);
    lsub = ito;
    
    // 将 usub 的内容赋值给 ifrom 指针
    ifrom = usub;
    
    # 将 lsub 的地址向后移动 xlsub[ndim] * iword 字节，转换为 int_t* 类型指针，并赋给 ito
    ito = (int_t *) ((char*)lsub + xlsub[ndim] * iword);
    
    # 将从 ifrom 开始的 xusub[ndim] 个 int_t 类型数据复制到 ito 指向的内存位置
    copy_mem_int(xusub[ndim], ifrom, ito);
    
    # 更新 usub 指针，使其指向 ito 指向的内存位置
    usub = ito;
    
    # 计算出 last 指针，指向 usub 后 xusub[ndim] * iword 字节的内存位置
    last = (char*)usub + xusub[ndim] * iword;
    
    # 计算出 fragment 指针，使其指向 Glu 结构体中的 stack.array 的顶部减去 last 的位置
    fragment = (char*) (((char*)Glu->stack.array + Glu->stack.top1) - last);
    
    # 减少 Glu 结构体中 stack.used 的值，使其减去 fragment 的长度
    Glu->stack.used -= (long int) fragment;
    
    # 减少 Glu 结构体中 stack.top1 的值，使其减去 fragment 的长度
    Glu->stack.top1 -= (long int) fragment;

    # 将 ucol 赋值给 Glu 结构体中的 ucol 成员
    Glu->ucol = ucol;
    
    # 将 lsub 赋值给 Glu 结构体中的 lsub 成员
    Glu->lsub = lsub;
    
    # 将 usub 赋值给 Glu 结构体中的 usub 成员
    Glu->usub = usub;
#if ( DEBUGlevel>=1 )
    printf("cStackCompress: fragment %lld\n", (long long) fragment);
    /* 调试模式下输出压缩片段信息 */
    /* for (last = 0; last < ndim; ++last)
    print_lu_col("After compress:", last, 0);*/
#endif    
    
}

/*! \brief Allocate storage for original matrix A
 */
void
callocateA(int n, int_t nnz, singlecomplex **a, int_t **asub, int_t **xa)
{
    // 分配存储空间给原始矩阵 A
    *a    = (singlecomplex *) complexMalloc(nnz);
    *asub = (int_t *) intMalloc(nnz);
    *xa   = (int_t *) intMalloc(n+1);
}


singlecomplex *complexMalloc(size_t n)
{
    // 分配 n 个 singlecomplex 元素的内存空间
    singlecomplex *buf;
    buf = (singlecomplex *) SUPERLU_MALLOC(n * (size_t) sizeof(singlecomplex)); 
    if ( !buf ) {
    // 如果内存分配失败，则终止程序并输出错误信息
    ABORT("SUPERLU_MALLOC failed for buf in singlecomplexMalloc()\n");
    }
    return (buf);
}

singlecomplex *complexCalloc(size_t n)
{
    // 分配 n 个 singlecomplex 元素的内存空间，并将每个元素初始化为零
    singlecomplex *buf;
    register size_t i;
    singlecomplex zero = {0.0, 0.0};
    buf = (singlecomplex *) SUPERLU_MALLOC(n * (size_t) sizeof(singlecomplex));
    if ( !buf ) {
    // 如果内存分配失败，则终止程序并输出错误信息
    ABORT("SUPERLU_MALLOC failed for buf in singlecomplexCalloc()\n");
    }
    for (i = 0; i < n; ++i) buf[i] = zero;
    return (buf);
}


int_t cmemory_usage(const int_t nzlmax, const int_t nzumax,
          const int_t nzlumax, const int n)
{
    register int iword, liword, dword;

    iword   = sizeof(int);
    liword  = sizeof(int_t);
    dword   = sizeof(singlecomplex);
    
    // 计算内存使用量的估计值
    return (10 * n * iword +
        nzlmax * liword + nzumax * (liword + dword) + nzlumax * dword);

}
```