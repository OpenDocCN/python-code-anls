# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zmemory.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zmemory.c
 * \brief Memory details
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 * </pre>
 */
#include "slu_zdefs.h"


/* Internal prototypes */
void  *zexpand (int_t *, MemType, int_t, int, GlobalLU_t *);
int   zLUWorkInit (int, int, int, int **, doublecomplex **, GlobalLU_t *);
void  copy_mem_doublecomplex (int_t, void *, void *);
void  zStackCompress (GlobalLU_t *);
void  zSetupSpace (void *, int_t, GlobalLU_t *);
void  *zuser_malloc (int, int, GlobalLU_t *);
void  zuser_free (int, int, GlobalLU_t *);

/* External prototypes (in memory.c - prec-independent) */
extern void    copy_mem_int    (int, void *, void *);
extern void    user_bcopy      (char *, char *, int);


/* Macros to manipulate stack */
#define StackFull(x)         ( x + Glu->stack.used >= Glu->stack.size )
#define NotDoubleAlign(addr) ( (intptr_t)addr & 7 )
#define DoubleAlign(addr)    ( ((intptr_t)addr + 7) & ~7L )    
#define TempSpace(m, w)      ( (2*w + 4 + NO_MARKER) * m * sizeof(int) + \
                  (w + 1) * m * sizeof(doublecomplex) )
#define Reduce(alpha)        ((alpha + 1) / 2)  /* i.e. (alpha-1)/2 + 1 */


/*! \brief Setup the memory model to be used for factorization.
 *  
 *    lwork = 0: use system malloc;
 *    lwork > 0: use user-supplied work[] space.
 */
void zSetupSpace(void *work, int_t lwork, GlobalLU_t *Glu)
{
    if ( lwork == 0 ) {
        Glu->MemModel = SYSTEM; /* 设置内存模型为系统提供的 malloc/free */
    } else if ( lwork > 0 ) {
        Glu->MemModel = USER;   /* 设置内存模型为用户提供的空间 */
        Glu->stack.used = 0;
        Glu->stack.top1 = 0;
        Glu->stack.top2 = (lwork/4)*4; /* 必须是字地址对齐 */
        Glu->stack.size = Glu->stack.top2;
        Glu->stack.array = (void *) work;
    }
}



void *zuser_malloc(int bytes, int which_end, GlobalLU_t *Glu)
{
    void *buf;
    
    if ( StackFull(bytes) ) return (NULL); /* 如果栈已满，则返回空指针 */

    if ( which_end == HEAD ) {
        buf = (char*) Glu->stack.array + Glu->stack.top1; /* 头部分配内存 */
        Glu->stack.top1 += bytes;
    } else {
        Glu->stack.top2 -= bytes; /* 尾部分配内存 */
        buf = (char*) Glu->stack.array + Glu->stack.top2;
    }
    
    Glu->stack.used += bytes;
    return buf;
}


void zuser_free(int bytes, int which_end, GlobalLU_t *Glu)
{
    if ( which_end == HEAD ) {
        Glu->stack.top1 -= bytes; /* 释放头部内存 */
    } else {
        Glu->stack.top2 += bytes; /* 释放尾部内存 */
    }
    Glu->stack.used -= bytes;
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
/*! \brief Query the space requirements for the LU factorization of a sparse matrix.
 *
 * \param L The input matrix L in SuperLU's data structure format.
 * \param U The input matrix U in SuperLU's data structure format.
 * \param mem_usage Pointer to a structure to hold memory usage information.
 * \return Always returns 0 indicating successful completion.
 */
int zQuerySpace(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage)
{
    SCformat *Lstore;  // Pointer to the compressed column storage of L
    NCformat *Ustore;  // Pointer to the compressed column storage of U
    register int n, iword, dword, panel_size = sp_ienv(1);  // Declare variables n, iword, dword, panel_size

    Lstore = L->Store;  // Initialize Lstore with L's compressed column storage
    Ustore = U->Store;  // Initialize Ustore with U's compressed column storage
    n = L->ncol;  // Number of columns in L (assumed to be square)
    iword = sizeof(int);  // Size of an integer in bytes
    dword = sizeof(doublecomplex);  // Size of a double complex number in bytes

    /* For LU factors */
    mem_usage->for_lu = (float)( (4.0*n + 3.0) * iword +
                                 Lstore->nzval_colptr[n] * dword +
                                 Lstore->rowind_colptr[n] * iword );
    mem_usage->for_lu += (float)( (n + 1.0) * iword +
                 Ustore->colptr[n] * (dword + iword) );

    /* Working storage to support factorization */
    mem_usage->total_needed = mem_usage->for_lu +
    (float)( (2.0 * panel_size + 4.0 + NO_MARKER) * n * iword +
        (panel_size + 1.0) * n * dword );

    return 0;
} /* zQuerySpace */


/*!
 * \brief Calculate memory usage for an ILU (Incomplete LU) factorization of a sparse matrix.
 *
 * \param L The input matrix L in SuperLU's data structure format.
 * \param U The input matrix U in SuperLU's data structure format.
 * \param mem_usage Pointer to a structure to hold memory usage information.
 * \return Always returns 0 indicating successful completion.
 */
int ilu_zQuerySpace(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage)
{
    SCformat *Lstore;  // Pointer to the compressed column storage of L
    NCformat *Ustore;  // Pointer to the compressed column storage of U
    register int n, panel_size = sp_ienv(1);  // Declare variables n, panel_size
    register float iword, dword;  // Floating-point variables for sizes

    Lstore = L->Store;  // Initialize Lstore with L's compressed column storage
    Ustore = U->Store;  // Initialize Ustore with U's compressed column storage
    n = L->ncol;  // Number of columns in L (assumed to be square)
    iword = sizeof(int);  // Size of an integer in bytes
    dword = sizeof(double);  // Size of a double in bytes

    /* For LU factors */
    mem_usage->for_lu = (float)( (4.0f * n + 3.0f) * iword +
                 Lstore->nzval_colptr[n] * dword +
                 Lstore->rowind_colptr[n] * iword );
    mem_usage->for_lu += (float)( (n + 1.0f) * iword +
                 Ustore->colptr[n] * (dword + iword) );

    /* Working storage to support factorization.
       ILU needs 5*n more integers than LU */
    mem_usage->total_needed = mem_usage->for_lu +
    (float)( (2.0f * panel_size + 9.0f + NO_MARKER) * n * iword +
        (panel_size + 1.0f) * n * dword );

    return 0;
} /* ilu_zQuerySpace */


/*! \brief Initialize memory allocation for SuperLU's factorization routines.
 *
 * This function estimates memory requirements and allocates memory accordingly.
 *
 * \param fact The factorization strategy to use.
 * \param work Pointer to the workspace memory.
 * \param lwork The size of the workspace memory.
 * \param m Number of rows in the matrix A.
 * \param n Number of columns in the matrix A.
 * \param annz Number of nonzeros in the matrix A.
 * \param panel_size Size of the panel (used internally for memory calculations).
 * \param fill_ratio Estimated fill ratio of the matrix A.
 * \param L The output matrix L in SuperLU's data structure format.
 * \param U The output matrix U in SuperLU's data structure format.
 * \param Glu Pointer to the global LU data structure.
 * \param iwork Pointer to integer workspace array.
 * \param dwork Pointer to double complex workspace array.
 * \return If lwork = -1, returns the estimated amount of space required, plus n;
 *         otherwise, returns the amount of space actually allocated when
 *         memory allocation failure occurred.
 */
int_t
zLUMemInit(fact_t fact, void *work, int_t lwork, int m, int n, int_t annz,
      int panel_size, double fill_ratio, SuperMatrix *L, SuperMatrix *U,
          GlobalLU_t *Glu, int **iwork, doublecomplex **dwork)
{
    int      info, iword, dword;  // Declare variables info, iword, dword
    SCformat *Lstore;  // Pointer to the compressed column storage of L
    NCformat *Ustore;  // Pointer to the compressed column storage of U
    int      *xsup, *supno;  // Pointers to integer arrays xsup and supno
    int_t    *lsub, *xlsub;  // Pointers to integer arrays lsub and xlsub
    doublecomplex   *lusup;  // Pointer to double complex array lusup
    int_t    *xlusup;  // Pointer to integer array xlusup
    doublecomplex   *ucol;  // Pointer to double complex array ucol
    int_t    *usub, *xusub;  // Pointers to integer arrays usub and xusub
    int_t    nzlmax, nzumax, nzlumax;  // Maximum nonzero counts for L, U, and LU
    # 计算 int 类型的大小并赋值给变量 iword
    iword     = sizeof(int);
    # 计算 doublecomplex 类型的大小并赋值给变量 dword
    dword     = sizeof(doublecomplex);
    # 将输入的 n 赋值给 Glu 结构体的成员变量 n
    Glu->n    = n;
    # 将 0 赋值给 Glu 结构体的成员变量 num_expansions
    Glu->num_expansions = 0;

    # 分配内存给 Glu 结构体的 expanders 成员，大小为 NO_MEMTYPE 个 ExpHeader 结构体的大小
    Glu->expanders = (ExpHeader *) SUPERLU_MALLOC( NO_MEMTYPE *
                                                     sizeof(ExpHeader) );
    # 检查内存分配是否成功，如果失败则终止程序并输出错误信息
    if ( !Glu->expanders ) ABORT("SUPERLU_MALLOC fails for expanders");
    
    # 如果 fact 不等于 SamePattern_SameRowPerm
    if ( fact != SamePattern_SameRowPerm ) {
        /* Guess for L\U factors */
        # 计算估计的 nzumax、nzlumax 和 nzlmax 的值
        nzumax = nzlumax = nzlmax = fill_ratio * annz;
        //nzlmax = SUPERLU_MAX(1, fill_ratio/4.) * annz;

        # 如果 lwork 等于 -1，则返回所需的空间大小
        if ( lwork == -1 ) {
            return ( GluIntArray(n) * iword + TempSpace(m, panel_size)
                + (nzlmax+nzumax)*iword + (nzlumax+nzumax)*dword + n );
        } else {
            # 调用 zSetupSpace 函数为工作空间配置空间
            zSetupSpace(work, lwork, Glu);
        }
    }
#if ( PRNTlevel >= 1 )
    // 如果 PRNTlevel 大于等于 1，则输出 zLUMemInit() 被调用的信息，包括 fill_ratio、nzlmax 和 nzumax 的值
    printf("zLUMemInit() called: fill_ratio %.0f, nzlmax %lld, nzumax %lld\n", 
           fill_ratio, (long long) nzlmax, (long long) nzumax);
    // 刷新标准输出流
    fflush(stdout);
#endif

/* Integer pointers for L\U factors */
// 根据 Glu->MemModel 的值选择分配整型指针数组的内存空间
if ( Glu->MemModel == SYSTEM ) {
    xsup   = int32Malloc(n+1);   // 分配 n+1 个 int32 类型的内存空间
    supno  = int32Malloc(n+1);   // 分配 n+1 个 int32 类型的内存空间
    xlsub  = intMalloc(n+1);     // 分配 n+1 个 int 类型的内存空间
    xlusup = intMalloc(n+1);     // 分配 n+1 个 int 类型的内存空间
    xusub  = intMalloc(n+1);     // 分配 n+1 个 int 类型的内存空间
} else {
    xsup   = (int *)zuser_malloc((n+1) * iword, HEAD, Glu);    // 使用用户定义的内存分配函数分配内存空间
    supno  = (int *)zuser_malloc((n+1) * iword, HEAD, Glu);    // 使用用户定义的内存分配函数分配内存空间
    xlsub  = zuser_malloc((n+1) * iword, HEAD, Glu);           // 使用用户定义的内存分配函数分配内存空间
    xlusup = zuser_malloc((n+1) * iword, HEAD, Glu);           // 使用用户定义的内存分配函数分配内存空间
    xusub  = zuser_malloc((n+1) * iword, HEAD, Glu);           // 使用用户定义的内存分配函数分配内存空间
}

lusup = (doublecomplex *) zexpand( &nzlumax, LUSUP, 0, 0, Glu );   // 扩展 lusup 数组的内存空间
ucol  = (doublecomplex *) zexpand( &nzumax, UCOL, 0, 0, Glu );    // 扩展 ucol 数组的内存空间
lsub  = (int_t *) zexpand( &nzlmax, LSUB, 0, 0, Glu );            // 扩展 lsub 数组的内存空间
usub  = (int_t *) zexpand( &nzumax, USUB, 0, 1, Glu );            // 扩展 usub 数组的内存空间

while ( !lusup || !ucol || !lsub || !usub ) {
    // 如果数组空间分配失败
    if ( Glu->MemModel == SYSTEM ) {
        SUPERLU_FREE(lusup);    // 释放 lusup 数组内存空间
        SUPERLU_FREE(ucol);     // 释放 ucol 数组内存空间
        SUPERLU_FREE(lsub);     // 释放 lsub 数组内存空间
        SUPERLU_FREE(usub);     // 释放 usub 数组内存空间
    } else {
        zuser_free((nzlumax+nzumax)*dword+(nzlmax+nzumax)*iword, HEAD, Glu);   // 使用用户定义的内存释放函数释放内存空间
    }
    // 将数组的最大容量减半
    nzlumax /= 2;
    nzumax /= 2;
    nzlmax /= 2;
    // 如果减半后的数组容量小于 annz
    if ( nzlumax < annz ) {
        // 输出内存不足以执行因子分解的信息，并返回内存使用情况和 n
        printf("Not enough memory to perform factorization.\n");
        return (zmemory_usage(nzlmax, nzumax, nzlumax, n) + n);
    }
#if ( PRNTlevel >= 1 )
    // 如果 PRNTlevel 大于等于 1，则输出减小数组大小后的信息
    printf("zLUMemInit() reduce size: nzlmax %ld, nzumax %ld\n", 
           (long) nzlmax, (long) nzumax);
    fflush(stdout);   // 刷新标准输出流
#endif
    // 重新扩展数组的内存空间
    lusup = (doublecomplex *) zexpand( &nzlumax, LUSUP, 0, 0, Glu );
    ucol  = (doublecomplex *) zexpand( &nzumax, UCOL, 0, 0, Glu );
    lsub  = (int_t *) zexpand( &nzlmax, LSUB, 0, 0, Glu );
    usub  = (int_t *) zexpand( &nzumax, USUB, 0, 1, Glu );
}

} else {
    /* fact == SamePattern_SameRowPerm */
    // 如果 fact 等于 SamePattern_SameRowPerm
    Lstore   = L->Store;        // 获取 L 的存储结构体
    Ustore   = U->Store;        // 获取 U 的存储结构体
    xsup     = Lstore->sup_to_col;   // 获取 L 的超节点到列索引映射数组
    supno    = Lstore->col_to_sup;   // 获取 L 的列索引到超节点映射数组
    xlsub    = Lstore->rowind_colptr;    // 获取 L 的行索引到列指针映射数组
    xlusup   = Lstore->nzval_colptr;     // 获取 L 的非零元值到列指针映射数组
    xusub    = Ustore->colptr;           // 获取 U 的列指针数组
    nzlmax   = Glu->nzlmax;    // 获取 Glu 中保存的上次因子分解的最大 nzlmax
    nzumax   = Glu->nzumax;    // 获取 Glu 中保存的上次因子分解的最大 nzumax
    nzlumax  = Glu->nzlumax;   // 获取 Glu 中保存的上次因子分解的最大 nzlumax

    if ( lwork == -1 ) {
        // 如果 lwork 等于 -1，则返回所需内存空间的估计值
        return ( GluIntArray(n) * iword + TempSpace(m, panel_size)
                 + (nzlmax+nzumax)*iword + (nzlumax+nzumax)*dword + n );
    } else if ( lwork == 0 ) {
        Glu->MemModel = SYSTEM;   // 设置 Glu 的内存模型为 SYSTEM
    } else {
        Glu->MemModel = USER;     // 设置 Glu 的内存模型为 USER
        Glu->stack.top2 = (lwork/4)*4;   // 设置 Glu 的栈顶指针为 lwork 的四舍五入到最近的 4 的倍数
        Glu->stack.size = Glu->stack.top2;   // 设置 Glu 的栈大小为栈顶指针的值
    }

    lsub  = Glu->expanders[LSUB].mem  = Lstore->rowind;   // 设置 Glu 中 LSUB 扩展器的内存空间
    lusup = Glu->expanders[LUSUP].mem = Lstore->nzval;    // 设置 Glu 中 LUSUP 扩展器的内存空间
    usub  = Glu->expanders[USUB].mem  = Ustore->rowind;
    # 设置 Glu 结构体中 USUB 扩展器的内存指针为 Ustore 的行索引

    ucol  = Glu->expanders[UCOL].mem  = Ustore->nzval;;
    # 设置 Glu 结构体中 UCOL 扩展器的内存指针为 Ustore 的非零值数组

    Glu->expanders[LSUB].size         = nzlmax;
    # 设置 Glu 结构体中 LSUB 扩展器的大小为 nzlmax

    Glu->expanders[LUSUP].size        = nzlumax;
    # 设置 Glu 结构体中 LUSUP 扩展器的大小为 nzlumax

    Glu->expanders[USUB].size         = nzumax;
    # 设置 Glu 结构体中 USUB 扩展器的大小为 nzumax

    Glu->expanders[UCOL].size         = nzumax;
    # 设置 Glu 结构体中 UCOL 扩展器的大小为 nzumax
    }

    Glu->xsup    = xsup;
    # 设置 Glu 结构体中的 xsup 成员

    Glu->supno   = supno;
    # 设置 Glu 结构体中的 supno 成员

    Glu->lsub    = lsub;
    # 设置 Glu 结构体中的 lsub 成员

    Glu->xlsub   = xlsub;
    # 设置 Glu 结构体中的 xlsub 成员

    Glu->lusup   = (void *) lusup;
    # 设置 Glu 结构体中的 lusup 成员为 lusup 指针的强制类型转换

    Glu->xlusup  = xlusup;
    # 设置 Glu 结构体中的 xlusup 成员

    Glu->ucol    = (void *) ucol;
    # 设置 Glu 结构体中的 ucol 成员为 ucol 指针的强制类型转换

    Glu->usub    = usub;
    # 设置 Glu 结构体中的 usub 成员

    Glu->xusub   = xusub;
    # 设置 Glu 结构体中的 xusub 成员

    Glu->nzlmax  = nzlmax;
    # 设置 Glu 结构体中的 nzlmax 成员

    Glu->nzumax  = nzumax;
    # 设置 Glu 结构体中的 nzumax 成员

    Glu->nzlumax = nzlumax;
    # 设置 Glu 结构体中的 nzlumax 成员
    
    info = zLUWorkInit(m, n, panel_size, iwork, dwork, Glu);
    # 调用 zLUWorkInit 初始化 LU 分解所需的工作空间和数据结构

    if ( info )
        # 如果初始化返回非零值，表示出现错误，计算返回信息和内存使用，并返回
        return ( info + zmemory_usage(nzlmax, nzumax, nzlumax, n) + n);
    
    ++Glu->num_expansions;
    # 增加 Glu 结构体中 num_expansions 成员的值

    return 0;
    # 返回 0，表示初始化成功
/*! \brief 初始化LU分解的工作存储空间。
 * 如果成功返回0，否则返回发生失败时已分配的字节数。
 */
int
zLUWorkInit(int m, int n, int panel_size, int **iworkptr, 
            doublecomplex **dworkptr, GlobalLU_t *Glu)
{
    int    isize, dsize, extra;
    doublecomplex *old_ptr;
    int    maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) ),
           rowblk   = sp_ienv(4);

    /* xplore[m] and xprune[n] can be 64-bit; they are allocated separately */
    //isize = ( (2 * panel_size + 3 + NO_MARKER ) * m + n ) * sizeof(int);
    isize = ( (2 * panel_size + 2 + NO_MARKER ) * m ) * sizeof(int);
    dsize = (m * panel_size +
         NUM_TEMPV(m,panel_size,maxsuper,rowblk)) * sizeof(doublecomplex);
    
    if ( Glu->MemModel == SYSTEM ) 
        *iworkptr = (int *) int32Calloc(isize/sizeof(int));
    else
        *iworkptr = (int *) zuser_malloc(isize, TAIL, Glu);
    if ( ! *iworkptr ) {
        fprintf(stderr, "zLUWorkInit: malloc fails for local iworkptr[]\n");
        return (isize + n);
    }

    if ( Glu->MemModel == SYSTEM )
        *dworkptr = (doublecomplex *) SUPERLU_MALLOC(dsize);
    else {
        *dworkptr = (doublecomplex *) zuser_malloc(dsize, TAIL, Glu);
        if ( NotDoubleAlign(*dworkptr) ) {
            old_ptr = *dworkptr;
            *dworkptr = (doublecomplex*) DoubleAlign(*dworkptr);
            *dworkptr = (doublecomplex*) ((double*)*dworkptr - 1);
            extra = (char*)old_ptr - (char*)*dworkptr;
    #if ( DEBUGlevel>=1 )
            printf("zLUWorkInit: not aligned, extra %d\n", extra); fflush(stdout);
    #endif        
            Glu->stack.top2 -= extra;
            Glu->stack.used += extra;
        }
    }
    if ( ! *dworkptr ) {
        fprintf(stderr, "malloc fails for local dworkptr[].");
        return (isize + dsize + n);
    }
    
    return 0;
} /* end zLUWorkInit */


/*! \brief 为实部工作数组设置指针。
 */
void
zSetRWork(int m, int panel_size, doublecomplex *dworkptr,
     doublecomplex **dense, doublecomplex **tempv)
{
    doublecomplex zero = {0.0, 0.0};

    int maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) ),
        rowblk   = sp_ienv(4);
    *dense = dworkptr;
    *tempv = *dense + panel_size*m;
    zfill (*dense, m * panel_size, zero);
    zfill (*tempv, NUM_TEMPV(m,panel_size,maxsuper,rowblk), zero);     
}
    
/*! \brief 释放因子例程使用的工作存储空间。
 */
void zLUWorkFree(int *iwork, doublecomplex *dwork, GlobalLU_t *Glu)
{
    if ( Glu->MemModel == SYSTEM ) {
        SUPERLU_FREE (iwork);
        SUPERLU_FREE (dwork);
    } else {
        Glu->stack.used -= (Glu->stack.size - Glu->stack.top2);
        Glu->stack.top2 = Glu->stack.size;
    /*    zStackCompress(Glu);  */
    }
    
    SUPERLU_FREE (Glu->expanders);    
    Glu->expanders = NULL;
}

/*! \brief 在因子化过程中扩展L和U的数据结构。
 * 
 * <pre>
 * 返回值:   0 - 成功返回
 *         > 0 - 当空间不足时已分配的字节数
 * </pre>
 */
int_t
/*
 * 函数 zLUMemXpand：根据给定的参数扩展全局LU数据结构的内存
 */
zLUMemXpand(int jcol,
       int_t next,          /* number of elements currently in the factors */
       MemType mem_type,  /* which type of memory to expand  */
       int_t *maxlen,       /* modified - maximum length of a data structure */
       GlobalLU_t *Glu    /* modified - global LU data structures */
       )
{
    void   *new_mem;
    
#if ( DEBUGlevel>=1 ) 
    printf("zLUMemXpand[1]: jcol %d, next %lld, maxlen %lld, MemType %d\n",
       jcol, (long long) next, (long long) *maxlen, mem_type);
#endif    

    /*
     * 根据 mem_type 调用 zexpand 函数来扩展内存
     */
    if (mem_type == USUB) 
        new_mem = zexpand(maxlen, mem_type, next, 1, Glu);
    else
        new_mem = zexpand(maxlen, mem_type, next, 0, Glu);
    
    /*
     * 如果内存分配失败，则打印错误信息并返回当前内存使用情况
     */
    if ( !new_mem ) {
        int_t    nzlmax  = Glu->nzlmax;
        int_t    nzumax  = Glu->nzumax;
        int_t    nzlumax = Glu->nzlumax;
        fprintf(stderr, "Can't expand MemType %d: jcol %d\n", mem_type, jcol);
        return (zmemory_usage(nzlmax, nzumax, nzlumax, Glu->n) + Glu->n);
    }

    /*
     * 根据 mem_type 更新全局 LU 数据结构中的相应内存和最大长度
     */
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
 * 函数 copy_mem_doublecomplex：复制双精度复数类型的内存数据
 */
void
copy_mem_doublecomplex(int_t howmany, void *old, void *new)
{
    register int_t i;
    doublecomplex *dold = old;
    doublecomplex *dnew = new;
    for (i = 0; i < howmany; i++) dnew[i] = dold[i];
}

/*! \brief Expand the existing storage to accommodate more fill-ins.
 *  函数 zexpand：扩展现有存储空间以容纳更多的填充内容
 */
void
*zexpand (
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

    /*
     * 根据 keep_prev 决定是否使用 prev_len，计算新的长度 new_len
     */
    if ( Glu->num_expansions == 0 || keep_prev ) {
        /* 第一次请求分配内存 */
        new_len = *prev_len;
    } else {
        new_len = alpha * *prev_len;
    }
    
    /*
     * 根据 mem_type 确定每个元素的字节数 lword
     */
    if ( type == LSUB || type == USUB ) lword = sizeof(int_t);
    else lword = sizeof(doublecomplex);

    /*
     * 根据 Glu->MemModel 分配新的内存空间
     */
    if ( Glu->MemModel == SYSTEM ) {
        new_mem = (void *) SUPERLU_MALLOC((size_t)new_len * lword);
        /* 其他内存模型的处理 */
    }

    /* 更多处理逻辑可以根据具体需求扩展 */

    return new_mem;
}
    # 如果 Glu->num_expansions 不等于 0，则执行以下操作
    if ( Glu->num_expansions != 0 ) {
        tries = 0;  # 初始化尝试次数为 0
        if ( keep_prev ) {  # 如果 keep_prev 为真
        if ( !new_mem ) return (NULL);  # 如果 new_mem 为 NULL，则返回 NULL
        } else {
        while ( !new_mem ) {  # 当 new_mem 为 NULL 时循环执行
            if ( ++tries > 10 ) return (NULL);  # 如果尝试次数超过 10 次，则返回 NULL
            alpha = Reduce(alpha);  # 对 alpha 进行 Reduce 操作
            new_len = alpha * *prev_len;  # 计算新的长度 new_len
            new_mem = (void *) SUPERLU_MALLOC((size_t)new_len * lword);  # 分配新的内存空间
        }
        }
        if ( type == LSUB || type == USUB ) {  # 如果 type 为 LSUB 或 USUB
        copy_mem_int(len_to_copy, expanders[type].mem, new_mem);  # 复制整数类型数据到新的内存空间
        } else {
        copy_mem_doublecomplex(len_to_copy, expanders[type].mem, new_mem);  # 复制复数类型数据到新的内存空间
        }
        SUPERLU_FREE (expanders[type].mem);  # 释放原来的内存空间
    }
    expanders[type].mem = (void *) new_mem;  # 更新 expanders[type].mem 为新的内存空间
    
    } else { /* MemModel == USER */  # 否则，如果 MemModel == USER
    
    if ( Glu->num_expansions == 0 ) { /* First time initialization */  # 如果 Glu->num_expansions == 0，表示首次初始化
    
        new_mem = zuser_malloc(new_len * lword, HEAD, Glu);  # 调用 zuser_malloc 分配新的内存空间
        if ( NotDoubleAlign(new_mem) &&  # 如果新分配的内存不是双精度对齐，并且
        (type == LUSUP || type == UCOL) ) {  # type 是 LUSUP 或 UCOL
        old_mem = new_mem;  # 将原始内存地址保存到 old_mem
        new_mem = (void *)DoubleAlign(new_mem);  # 对新内存进行双精度对齐
        extra = (char*)new_mem - (char*)old_mem;  # 计算额外偏移量
        }
#if ( DEBUGlevel>=1 )
        printf("expand(): not aligned, extra %d\n", extra);
#endif
        // 如果 DEBUGlevel 大于等于 1，则打印额外的调试信息，显示 extra 的值

        Glu->stack.top1 += extra;
        Glu->stack.used += extra;
        // 更新 Glu 结构体中的 stack.top1 和 stack.used，增加额外的空间量 extra

        }
        
        expanders[type].mem = (void *) new_mem;
        // 将 expanders 数组中指定类型 type 的 mem 成员更新为 new_mem 的地址
        
    } else { /* CASE: num_expansions != 0 */
    
        tries = 0;
        // 尝试次数初始化为 0
        extra = (new_len - *prev_len) * lword;
        // 计算需要额外空间的字节数，根据新长度与前一个长度的差异乘以每个元素的字节数 lword
        if ( keep_prev ) {
        if ( StackFull(extra) ) return (NULL);
        // 如果 keep_prev 为真，并且栈已满，则返回空指针
        } else {
        while ( StackFull(extra) ) {
            if ( ++tries > 10 ) return (NULL);
            // 如果栈已满，尝试次数超过 10 次，则返回空指针
            alpha = Reduce(alpha);
            new_len = alpha * *prev_len;
            extra = (new_len - *prev_len) * lword;        
        }
        }

          /* Need to expand the memory: moving the content after the current MemType
               to make extra room for the current MemType.
                   Memory layout: [ LUSUP || UCOL || LSUB || USUB ]
          */
          // 需要扩展内存：移动当前 MemType 后面的内容，为当前 MemType 留出额外的空间
          // 内存布局：[ LUSUP || UCOL || LSUB || USUB ]
          if ( type != USUB ) {
        new_mem = (void*)((char*)expanders[type + 1].mem + extra);
        // 计算新的内存位置，使得 type+1 类型的 mem 在 type 类型的基础上增加 extra 字节的偏移量
        bytes_to_copy = (char*)Glu->stack.array + Glu->stack.top1
            - (char*)expanders[type + 1].mem;
        // 计算需要复制的字节数，从 Glu 结构体的 stack.array 开始复制，长度为 Glu->stack.top1 到 expanders[type + 1].mem 之间的距离
        user_bcopy(expanders[type+1].mem, new_mem, bytes_to_copy);
        // 调用用户自定义的拷贝函数，将 expanders[type+1].mem 的内容复制到 new_mem

        if ( type < USUB ) {
            Glu->usub = expanders[USUB].mem =
            (void*)((char*)expanders[USUB].mem + extra);
        // 如果 type 小于 USUB，则更新 Glu 结构体中 USUB 类型的 mem 成员的地址为原地址加上 extra 的偏移量
        }
        if ( type < LSUB ) {
            Glu->lsub = expanders[LSUB].mem =
            (void*)((char*)expanders[LSUB].mem + extra);
        // 如果 type 小于 LSUB，则更新 Glu 结构体中 LSUB 类型的 mem 成员的地址为原地址加上 extra 的偏移量
        }
        if ( type < UCOL ) {
            Glu->ucol = expanders[UCOL].mem =
            (void*)((char*)expanders[UCOL].mem + extra);
        // 如果 type 小于 UCOL，则更新 Glu 结构体中 UCOL 类型的 mem 成员的地址为原地址加上 extra 的偏移量
        }
        Glu->stack.top1 += extra;
        Glu->stack.used += extra;
        // 更新 Glu 结构体中的 stack.top1 和 stack.used，增加额外的空间量 extra
        if ( type == UCOL ) {
            Glu->stack.top1 += extra;   /* Add same amount for USUB */
            Glu->stack.used += extra;
        // 如果 type 等于 UCOL，则再次增加相同的额外空间量给 USUB
        }
        
        } /* end expansion */

    } /* else ... */
    }

    expanders[type].size = new_len;
    // 更新 expanders 数组中指定类型 type 的 size 成员为新长度 new_len
    *prev_len = new_len;
    // 更新 prev_len 指向的值为 new_len
    if ( Glu->num_expansions ) ++Glu->num_expansions;
    // 如果 Glu 结构体中的 num_expansions 不为零，则增加其值

    return (void *) expanders[type].mem;
    // 返回指向 expanders 数组中指定类型 type 的 mem 成员的指针
    
} /* zexpand */


/*! \brief Compress the work[] array to remove fragmentation.
 */
void
zStackCompress(GlobalLU_t *Glu)
{
    register int iword, dword, ndim;
    char     *last, *fragment;
    int_t    *ifrom, *ito;
    doublecomplex   *dfrom, *dto;
    int_t    *xlsub, *lsub, *xusub, *usub, *xlusup;
    doublecomplex   *ucol, *lusup;
    
    iword = sizeof(int);
    dword = sizeof(doublecomplex);
    ndim = Glu->n;

    xlsub  = Glu->xlsub;
    lsub   = Glu->lsub;
    xusub  = Glu->xusub;
    usub   = Glu->usub;
    xlusup = Glu->xlusup;
    ucol   = Glu->ucol;
    lusup  = Glu->lusup;
    
    dfrom = ucol;
    // 将 ucol 指针赋值给 dfrom
    dto = (doublecomplex *)((char*)lusup + xlusup[ndim] * dword);
    // 计算 dto 指针，指向 lusup 数组之后的内存，长度为 xlusup[ndim] 个 doublecomplex 元素
    copy_mem_doublecomplex(xusub[ndim], dfrom, dto);
    // 调用用户自定义的拷贝函数，将 dfrom 指向的数据拷贝到 dto 指向的位置
    ucol = dto;

    ifrom = lsub;
    // 将 lsub 指针赋值给 ifrom
    ito = (int_t *) ((char*)ucol + xusub[ndim] * iword);
    // 计算 ito 指针，指向 ucol 数组之后的内存，长度为 xusub[ndim] 个 int_t 元素
    copy_mem_int(xlsub[ndim], ifrom, ito);
    // 调用用户自定义的拷贝函数，将 ifrom 指向的数据拷贝到 ito 指向的位置
    lsub = ito;
    
    ifrom = usub;
    // 将 usub 指针赋值给 ifrom


这样注释了每一行代码，确保了每个操作的目的和效果都清晰可见。
    # 将 lsub 的地址偏移 xlsub[ndim] * iword 处的整数类型指针赋给 ito
    ito = (int_t *) ((char*)lsub + xlsub[ndim] * iword);

    # 将从 ifrom 开始的 xusub[ndim] 个整数复制到 ito 指向的内存位置
    copy_mem_int(xusub[ndim], ifrom, ito);

    # 将 ito 指向的地址赋给 usub
    usub = ito;
    
    # 计算出 usub 指向的最后一个元素之后的地址，并将其赋给 last
    last = (char*)usub + xusub[ndim] * iword;

    # 计算从 Glu->stack.array 的顶部到 last 之间的内存片段，并将其赋给 fragment
    fragment = (char*) (((char*)Glu->stack.array + Glu->stack.top1) - last);

    # 减少 Glu->stack.used 的值，减去 fragment 所占的字节数
    Glu->stack.used -= (long int) fragment;

    # 减少 Glu->stack.top1 的值，减去 fragment 所占的字节数
    Glu->stack.top1 -= (long int) fragment;

    # 将 ucol 赋给 Glu->ucol
    Glu->ucol = ucol;

    # 将 lsub 赋给 Glu->lsub
    Glu->lsub = lsub;

    # 将 usub 赋给 Glu->usub
    Glu->usub = usub;
#if ( DEBUGlevel>=1 )
    // 如果调试级别大于等于1，打印碎片号码
    printf("zStackCompress: fragment %lld\n", (long long) fragment);
    /* for (last = 0; last < ndim; ++last)
    print_lu_col("After compress:", last, 0);*/
#endif    
    
}

/*! \brief Allocate storage for original matrix A
 */
void
zallocateA(int n, int_t nnz, doublecomplex **a, int_t **asub, int_t **xa)
{
    // 分配 nnz 个 doublecomplex 类型的内存空间给指针 *a
    *a    = (doublecomplex *) doublecomplexMalloc(nnz);
    // 分配 nnz 个 int_t 类型的内存空间给指针 *asub
    *asub = (int_t *) intMalloc(nnz);
    // 分配 (n+1) 个 int_t 类型的内存空间给指针 *xa
    *xa   = (int_t *) intMalloc(n+1);
}


doublecomplex *doublecomplexMalloc(size_t n)
{
    // 分配 n 个 doublecomplex 结构的内存空间给 buf
    doublecomplex *buf;
    buf = (doublecomplex *) SUPERLU_MALLOC(n * (size_t) sizeof(doublecomplex)); 
    if ( !buf ) {
        // 如果分配失败，终止程序并打印错误信息
        ABORT("SUPERLU_MALLOC failed for buf in doublecomplexMalloc()\n");
    }
    // 返回分配的内存空间的起始地址
    return (buf);
}

doublecomplex *doublecomplexCalloc(size_t n)
{
    // 分配 n 个 doublecomplex 结构的内存空间给 buf
    doublecomplex *buf;
    register size_t i;
    // 设置 zero 为 {0.0, 0.0}
    doublecomplex zero = {0.0, 0.0};
    buf = (doublecomplex *) SUPERLU_MALLOC(n * (size_t) sizeof(doublecomplex));
    if ( !buf ) {
        // 如果分配失败，终止程序并打印错误信息
        ABORT("SUPERLU_MALLOC failed for buf in doublecomplexCalloc()\n");
    }
    // 将分配的内存空间初始化为 zero
    for (i = 0; i < n; ++i) buf[i] = zero;
    // 返回分配的内存空间的起始地址
    return (buf);
}


int_t zmemory_usage(const int_t nzlmax, const int_t nzumax,
          const int_t nzlumax, const int n)
{
    register int iword, liword, dword;

    // 计算内存使用量
    iword   = sizeof(int);
    liword  = sizeof(int_t);
    dword   = sizeof(doublecomplex);
    
    return (10 * n * iword +
        nzlmax * liword + nzumax * (liword + dword) + nzlumax * dword);
}
```