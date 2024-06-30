# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dmemory.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dmemory.c
 * \brief Memory details
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 * </pre>
 */
#include "slu_ddefs.h"


/* Internal prototypes */
void  *dexpand (int_t *, MemType, int_t, int, GlobalLU_t *);
int   dLUWorkInit (int, int, int, int **, double **, GlobalLU_t *);
void  copy_mem_double (int_t, void *, void *);
void  dStackCompress (GlobalLU_t *);
void  dSetupSpace (void *, int_t, GlobalLU_t *);
void  *duser_malloc (int, int, GlobalLU_t *);
void  duser_free (int, int, GlobalLU_t *);

/* External prototypes (in memory.c - prec-independent) */
extern void    copy_mem_int    (int, void *, void *);
extern void    user_bcopy      (char *, char *, int);


/* Macros to manipulate stack */
#define StackFull(x)         ( x + Glu->stack.used >= Glu->stack.size )
#define NotDoubleAlign(addr) ( (intptr_t)addr & 7 )
#define DoubleAlign(addr)    ( ((intptr_t)addr + 7) & ~7L )    
#define TempSpace(m, w)      ( (2*w + 4 + NO_MARKER) * m * sizeof(int) + \
                  (w + 1) * m * sizeof(double) )
#define Reduce(alpha)        ((alpha + 1) / 2)  /* i.e. (alpha-1)/2 + 1 */




/*! \brief Setup the memory model to be used for factorization.
 *  
 *    lwork = 0: use system malloc;
 *    lwork > 0: use user-supplied work[] space.
 */
void dSetupSpace(void *work, int_t lwork, GlobalLU_t *Glu)
{
    if ( lwork == 0 ) {
    Glu->MemModel = SYSTEM; /* 设置为使用系统的 malloc/free */
    } else if ( lwork > 0 ) {
    Glu->MemModel = USER;   /* 设置为使用用户提供的工作空间 */
    Glu->stack.used = 0;
    Glu->stack.top1 = 0;
    Glu->stack.top2 = (lwork/4)*4; /* 必须是字地址可访问的 */
    Glu->stack.size = Glu->stack.top2;
    Glu->stack.array = (void *) work;
    }
}



void *duser_malloc(int bytes, int which_end, GlobalLU_t *Glu)
{
    void *buf;
    
    if ( StackFull(bytes) ) return (NULL);

    if ( which_end == HEAD ) {
    buf = (char*) Glu->stack.array + Glu->stack.top1;
    Glu->stack.top1 += bytes;
    } else {
    Glu->stack.top2 -= bytes;
    buf = (char*) Glu->stack.array + Glu->stack.top2;
    }
    
    Glu->stack.used += bytes;
    return buf;
}


void duser_free(int bytes, int which_end, GlobalLU_t *Glu)
{
    if ( which_end == HEAD ) {
    Glu->stack.top1 -= bytes;
    } else {
    Glu->stack.top2 += bytes;
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
int_t
dLUMemInit(fact_t fact, void *work, int_t lwork, int m, int n, int_t annz,
      int panel_size, double fill_ratio, SuperMatrix *L, SuperMatrix *U,
          GlobalLU_t *Glu, int **iwork, double **dwork)
{
    int      info, iword, dword;
    SCformat *Lstore;
    NCformat *Ustore;
    int      *xsup, *supno;
    int_t    *lsub, *xlsub;
    double   *lusup;
    int_t    *xlusup;
    double   *ucol;
    int_t    *usub, *xusub;
    int_t    nzlmax, nzumax, nzlumax;
    
    // Size of an integer in bytes
    iword     = sizeof(int);
    // Size of a double in bytes
    dword     = sizeof(double);

    // Pointer to the compressed column storage of matrix L
    Lstore = L->Store;
    // Pointer to the compressed column storage of matrix U
    Ustore = U->Store;

    // Initialize the size of workspace needed for L\\U factorization

    /* For LU factors */
    // Calculate memory usage for LU factors in L
    // 4*n + 3 is an estimate for the integer storage
    // Lstore->nzval_colptr[n] * dword is for the double storage in nzval
    // Lstore->rowind_colptr[n] * iword is for the integer storage in rowind
    mem_usage->for_lu = (float)( (4.0 * n + 3.0) * iword +
                 Lstore->nzval_colptr[n] * dword +
                 Lstore->rowind_colptr[n] * iword );
    // Add memory usage for LU factors in U
    mem_usage->for_lu += (float)( (n + 1.0) * iword +
                 Ustore->colptr[n] * (dword + iword) );

    // Working storage to support factorization.
    // ILU needs 5*n more integers than LU
    mem_usage->total_needed = mem_usage->for_lu +
    (float)( (2.0 * panel_size + 9.0 + NO_MARKER) * n * iword +
        (panel_size + 1.0) * n * dword );

    return 0;
} /* dLUMemInit */
    // 定义一个变量，存储 double 类型的大小（字节数）
    dword     = sizeof(double);
    // 将参数 n 赋值给 Glu 结构体的成员变量 n
    Glu->n    = n;
    // 将数值 0 赋值给 Glu 结构体的成员变量 num_expansions
    Glu->num_expansions = 0;

    // 分配存储空间以容纳 ExpHeader 结构体数组，其大小为 NO_MEMTYPE 乘以 ExpHeader 结构体的大小
    Glu->expanders = (ExpHeader *) SUPERLU_MALLOC( NO_MEMTYPE *
                                                     sizeof(ExpHeader) );
    // 检查分配是否成功，若失败则输出错误信息并终止程序
    if ( !Glu->expanders ) ABORT("SUPERLU_MALLOC fails for expanders");
    
    // 若传入的参数 fact 不等于 SamePattern_SameRowPerm
    if ( fact != SamePattern_SameRowPerm ) {
        /* Guess for L\U factors */
        // 通过填充比率 fill_ratio 对估算的非零元素个数进行初始化
        nzumax = nzlumax = nzlmax = fill_ratio * annz;
        // 对 nzlmax 进行进一步的计算，但注释中的代码被注释掉了，实际上未被执行
        
        // 若传入的参数 lwork 为 -1，返回所需的内存空间大小
        if ( lwork == -1 ) {
            // 返回的内存大小由多个部分组成，这里使用 GluIntArray(n) * iword + TempSpace(m, panel_size) + (nzlmax+nzumax)*iword + (nzlumax+nzumax)*dword + n 计算
            return ( GluIntArray(n) * iword + TempSpace(m, panel_size)
                + (nzlmax+nzumax)*iword + (nzlumax+nzumax)*dword + n );
        } else {
            // 调用 dSetupSpace 函数，为工作空间分配内存并进行设置
            dSetupSpace(work, lwork, Glu);
        }
    }
#if ( PRNTlevel >= 1 )
    // 如果打印级别大于等于1，则输出以下信息
    printf("dLUMemInit() called: fill_ratio %.0f, nzlmax %lld, nzumax %lld\n", 
           fill_ratio, (long long) nzlmax, (long long) nzumax);
    // 刷新标准输出流
    fflush(stdout);
#endif

/* Integer pointers for L\U factors */
// 为 L/U 因子分配整数指针空间
if ( Glu->MemModel == SYSTEM ) {
    xsup   = int32Malloc(n+1);    // 分配 n+1 个 int32 的内存空间给 xsup
    supno  = int32Malloc(n+1);    // 分配 n+1 个 int32 的内存空间给 supno
    xlsub  = intMalloc(n+1);      // 分配 n+1 个 int 的内存空间给 xlsub
    xlusup = intMalloc(n+1);      // 分配 n+1 个 int 的内存空间给 xlusup
    xusub  = intMalloc(n+1);      // 分配 n+1 个 int 的内存空间给 xusub
} else {
    xsup   = (int *)duser_malloc((n+1) * iword, HEAD, Glu);  // 使用 duser_malloc 分配内存给 xsup
    supno  = (int *)duser_malloc((n+1) * iword, HEAD, Glu);  // 使用 duser_malloc 分配内存给 supno
    xlsub  = duser_malloc((n+1) * iword, HEAD, Glu);         // 使用 duser_malloc 分配内存给 xlsub
    xlusup = duser_malloc((n+1) * iword, HEAD, Glu);         // 使用 duser_malloc 分配内存给 xlusup
    xusub  = duser_malloc((n+1) * iword, HEAD, Glu);         // 使用 duser_malloc 分配内存给 xusub
}

lusup = (double *) dexpand( &nzlumax, LUSUP, 0, 0, Glu );    // 调用 dexpand 函数扩展 LUSUP 内存空间
ucol  = (double *) dexpand( &nzumax, UCOL, 0, 0, Glu );     // 调用 dexpand 函数扩展 UCOL 内存空间
lsub  = (int_t *) dexpand( &nzlmax, LSUB, 0, 0, Glu );      // 调用 dexpand 函数扩展 LSUB 内存空间
usub  = (int_t *) dexpand( &nzumax, USUB, 0, 1, Glu );      // 调用 dexpand 函数扩展 USUB 内存空间

while ( !lusup || !ucol || !lsub || !usub ) {
    // 若有任何一个指针为 NULL
    if ( Glu->MemModel == SYSTEM ) {
        // 如果内存模型是 SYSTEM
        SUPERLU_FREE(lusup);   // 释放 lusup 指向的内存
        SUPERLU_FREE(ucol);    // 释放 ucol 指向的内存
        SUPERLU_FREE(lsub);    // 释放 lsub 指向的内存
        SUPERLU_FREE(usub);    // 释放 usub 指向的内存
    } else {
        // 如果内存模型不是 SYSTEM
        duser_free((nzlumax+nzumax)*dword+(nzlmax+nzumax)*iword,
                            HEAD, Glu);   // 调用 duser_free 释放内存
    }
    // 减半内存大小
    nzlumax /= 2;
    nzumax /= 2;
    nzlmax /= 2;
    // 如果减半后的大小小于 annz
    if ( nzlumax < annz ) {
        printf("Not enough memory to perform factorization.\n");  // 输出内存不足的信息
        return (dmemory_usage(nzlmax, nzumax, nzlumax, n) + n);   // 返回内存使用量
    }
#if ( PRNTlevel >= 1)
    // 如果打印级别大于等于1，则输出以下信息
    printf("dLUMemInit() reduce size: nzlmax %ld, nzumax %ld\n", 
           (long) nzlmax, (long) nzumax);
    // 刷新标准输出流
    fflush(stdout);
#endif
    // 重新扩展 lusup, ucol, lsub, usub 的内存空间
    lusup = (double *) dexpand( &nzlumax, LUSUP, 0, 0, Glu );
    ucol  = (double *) dexpand( &nzumax, UCOL, 0, 0, Glu );
    lsub  = (int_t *) dexpand( &nzlmax, LSUB, 0, 0, Glu );
    usub  = (int_t *) dexpand( &nzumax, USUB, 0, 1, Glu );
}

} else {
/* fact == SamePattern_SameRowPerm */
// 若 fact == SamePattern_SameRowPerm
Lstore   = L->Store;           // 获取 L 矩阵的存储结构
Ustore   = U->Store;           // 获取 U 矩阵的存储结构
xsup     = Lstore->sup_to_col; // 指向 L 矩阵的超节点到列的映射
supno    = Lstore->col_to_sup; // 指向 L 矩阵的列到超节点的映射
xlsub    = Lstore->rowind_colptr;  // 指向 L 矩阵的行索引指针
xlusup   = Lstore->nzval_colptr;   // 指向 L 矩阵的非零值指针
xusub    = Ustore->colptr;     // 指向 U 矩阵的列指针
nzlmax   = Glu->nzlmax;        // 从先前的因子分解中获取最大值
nzumax   = Glu->nzumax;        // 从先前的因子分解中获取最大值
nzlumax  = Glu->nzlumax;       // 从先前的因子分解中获取最大值

if ( lwork == -1 ) {
    // 如果 lwork 等于 -1，则返回内存需求
    return ( GluIntArray(n) * iword + TempSpace(m, panel_size)
        + (nzlmax+nzumax)*iword + (nzlumax+nzumax)*dword + n );
} else if ( lwork == 0 ) {
    // 如果 lwork 等于 0，则设置内存模型为 SYSTEM
    Glu->MemModel = SYSTEM;
} else {
    // 否则，设置内存模型为 USER
    Glu->MemModel = USER;
    // 设置栈顶到最近的 4 的倍数（以字为单位）
    Glu->stack.top2 = (lwork/4)*4; /* must be word-addressable */
    Glu->stack.size = Glu->stack.top2; // 设置栈的大小
}

lsub  = Glu->expanders[LSUB].mem  = Lstore->rowind;  // 设置 LSUB 内存
lusup = Glu->expanders[LUSUP].mem = Lstore->nzval;   // 设置 LUSUP 内存
usub  = Glu->expanders[USUB].mem  = Ustore->rowind;  // 设置 USUB 内存
    ucol  = Glu->expanders[UCOL].mem  = Ustore->nzval;;
    # 将Ustore结构体中的nzval成员赋值给ucol，并将其赋给Glu结构体中expanders数组中UCOL对应的成员变量mem。

    Glu->expanders[LSUB].size         = nzlmax;
    # 设置Glu结构体中expanders数组中LSUB对应的成员变量size为nzlmax。

    Glu->expanders[LUSUP].size        = nzlumax;
    # 设置Glu结构体中expanders数组中LUSUP对应的成员变量size为nzlumax。

    Glu->expanders[USUB].size         = nzumax;
    # 设置Glu结构体中expanders数组中USUB对应的成员变量size为nzumax。

    Glu->expanders[UCOL].size         = nzumax;    
    # 设置Glu结构体中expanders数组中UCOL对应的成员变量size为nzumax。

    }

    Glu->xsup    = xsup;
    # 将xsup赋值给Glu结构体中的xsup成员变量。

    Glu->supno   = supno;
    # 将supno赋值给Glu结构体中的supno成员变量。

    Glu->lsub    = lsub;
    # 将lsub赋值给Glu结构体中的lsub成员变量。

    Glu->xlsub   = xlsub;
    # 将xlsub赋值给Glu结构体中的xlsub成员变量。

    Glu->lusup   = (void *) lusup;
    # 将lusup强制转换为void指针类型，并赋值给Glu结构体中的lusup成员变量。

    Glu->xlusup  = xlusup;
    # 将xlusup赋值给Glu结构体中的xlusup成员变量。

    Glu->ucol    = (void *) ucol;
    # 将ucol强制转换为void指针类型，并赋值给Glu结构体中的ucol成员变量。

    Glu->usub    = usub;
    # 将usub赋值给Glu结构体中的usub成员变量。

    Glu->xusub   = xusub;
    # 将xusub赋值给Glu结构体中的xusub成员变量。

    Glu->nzlmax  = nzlmax;
    # 将nzlmax赋值给Glu结构体中的nzlmax成员变量。

    Glu->nzumax  = nzumax;
    # 将nzumax赋值给Glu结构体中的nzumax成员变量。

    Glu->nzlumax = nzlumax;
    # 将nzlumax赋值给Glu结构体中的nzlumax成员变量。
    
    info = dLUWorkInit(m, n, panel_size, iwork, dwork, Glu);
    # 调用dLUWorkInit函数，初始化LU分解所需的工作空间，返回初始化信息。

    if ( info )
        return ( info + dmemory_usage(nzlmax, nzumax, nzlumax, n) + n);
    # 如果初始化信息非零，返回info与dmemory_usage函数计算结果的和加上n。

    ++Glu->num_expansions;
    # Glu结构体中num_expansions成员变量加1，表示扩展次数增加。

    return 0;
    # 返回0，表示函数执行成功。
} /* dLUMemInit */

/*! \brief Allocate known working storage.
 * Returns 0 if success, otherwise
 * returns the number of bytes allocated so far when failure occurred.
 */
int
dLUWorkInit(int m, int n, int panel_size, int **iworkptr, 
            double **dworkptr, GlobalLU_t *Glu)
{
    int    isize, dsize, extra;
    double *old_ptr;
    int    maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) ),
           rowblk   = sp_ienv(4);

    /* xplore[m] and xprune[n] can be 64-bit; they are allocated separately */
    //isize = ( (2 * panel_size + 3 + NO_MARKER ) * m + n ) * sizeof(int);
    isize = ( (2 * panel_size + 2 + NO_MARKER ) * m ) * sizeof(int);
    // Calculate size of double array needed
    dsize = (m * panel_size +
         NUM_TEMPV(m,panel_size,maxsuper,rowblk)) * sizeof(double);
    
    // Allocate memory for integer workspace based on memory model
    if ( Glu->MemModel == SYSTEM ) 
        *iworkptr = (int *) int32Calloc(isize/sizeof(int));
    else
        *iworkptr = (int *) duser_malloc(isize, TAIL, Glu);

    // Check if allocation was successful
    if ( ! *iworkptr ) {
        fprintf(stderr, "dLUWorkInit: malloc fails for local iworkptr[]\n");
        return (isize + n);
    }

    // Allocate memory for double workspace based on memory model
    if ( Glu->MemModel == SYSTEM )
        *dworkptr = (double *) SUPERLU_MALLOC(dsize);
    else {
        *dworkptr = (double *) duser_malloc(dsize, TAIL, Glu);

        // Handle alignment issues for non-system memory model
        if ( NotDoubleAlign(*dworkptr) ) {
            old_ptr = *dworkptr;
            *dworkptr = (double*) DoubleAlign(*dworkptr);
            *dworkptr = (double*) ((double*)*dworkptr - 1);
            extra = (char*)old_ptr - (char*)*dworkptr;
#if ( DEBUGlevel>=1 )
            printf("dLUWorkInit: not aligned, extra %d\n", extra); fflush(stdout);
#endif        
            Glu->stack.top2 -= extra;
            Glu->stack.used += extra;
        }
    }

    // Check if allocation was successful
    if ( ! *dworkptr ) {
        fprintf(stderr, "malloc fails for local dworkptr[].");
        return (isize + dsize + n);
    }
    
    return 0;
} /* end dLUWorkInit */


/*! \brief Set up pointers for real working arrays.
 */
void
dSetRWork(int m, int panel_size, double *dworkptr,
     double **dense, double **tempv)
{
    double zero = 0.0;

    int maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) ),
        rowblk   = sp_ienv(4);

    // Assign dense and tempv pointers
    *dense = dworkptr;
    *tempv = *dense + panel_size*m;

    // Fill dense and tempv arrays with zeros
    dfill (*dense, m * panel_size, zero);
    dfill (*tempv, NUM_TEMPV(m,panel_size,maxsuper,rowblk), zero);     
}
    
/*! \brief Free the working storage used by factor routines.
 */
void dLUWorkFree(int *iwork, double *dwork, GlobalLU_t *Glu)
{
    // Free memory based on memory model
    if ( Glu->MemModel == SYSTEM ) {
        SUPERLU_FREE (iwork);
        SUPERLU_FREE (dwork);
    } else {
        // Compress stack for non-system memory model
        Glu->stack.used -= (Glu->stack.size - Glu->stack.top2);
        Glu->stack.top2 = Glu->stack.size;
        /*    dStackCompress(Glu);  */
    }
    
    // Free expanders array
    SUPERLU_FREE (Glu->expanders);    
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
# 扩展LU因子存储的内存空间以容纳更多的填充项
dLUMemXpand(int jcol,
       int_t next,          /* 当前因子中的元素数量 */
       MemType mem_type,    /* 扩展哪种类型的内存 */
       int_t *maxlen,       /* 修改后的 - 数据结构的最大长度 */
       GlobalLU_t *Glu      /* 修改后的 - 全局LU数据结构 */
       )
{
    void   *new_mem;
    
#if ( DEBUGlevel>=1 ) 
    printf("dLUMemXpand[1]: jcol %d, next %lld, maxlen %lld, MemType %d\n",
       jcol, (long long) next, (long long) *maxlen, mem_type);
#endif    

    if (mem_type == USUB) 
        new_mem = dexpand(maxlen, mem_type, next, 1, Glu);
    else
        new_mem = dexpand(maxlen, mem_type, next, 0, Glu);
    
    if ( !new_mem ) {
        int_t    nzlmax  = Glu->nzlmax;
        int_t    nzumax  = Glu->nzumax;
        int_t    nzlumax = Glu->nzlumax;
        fprintf(stderr, "无法扩展内存类型 %d: jcol %d\n", mem_type, jcol);
        return (dmemory_usage(nzlmax, nzumax, nzlumax, Glu->n) + Glu->n);
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

# 复制双精度内存数据
void
copy_mem_double(int_t howmany, void *old, void *new)
{
    register int_t i;
    double *dold = old;
    double *dnew = new;
    for (i = 0; i < howmany; i++) dnew[i] = dold[i];
}

/*! \brief 扩展现有存储空间以容纳更多的填充项。
 */
void
*dexpand (
     int_t *prev_len,     /* 上次调用使用的长度 */
     MemType type,        /* 要扩展的内存部分 */
     int_t len_to_copy,   /* 要复制到新存储的内存大小 */
     int keep_prev,       /* = 1: 使用 prev_len; = 0: 计算新的长度以扩展 */
     GlobalLU_t *Glu      /* 修改后的 - 全局LU数据结构 */
    )
{
    float    EXPAND = 1.5;
    float    alpha;
    void     *new_mem, *old_mem;
    int_t    new_len, bytes_to_copy;
    int      tries, lword, extra;
    ExpHeader *expanders = Glu->expanders; /* 包含四种类型内存的数组 */

    alpha = EXPAND;

    if ( Glu->num_expansions == 0 || keep_prev ) {
        /* 首次分配请求 */
        new_len = *prev_len;
    } else {
        new_len = alpha * *prev_len;
    }
    
    if ( type == LSUB || type == USUB ) lword = sizeof(int_t);
    else lword = sizeof(double);

    if ( Glu->MemModel == SYSTEM ) {
        new_mem = (void *) SUPERLU_MALLOC((size_t)new_len * lword);
    # 如果 Glu 结构体中的 num_expansions 不等于 0，则执行以下代码块
    if ( Glu->num_expansions != 0 ) {
        tries = 0;  # 尝试次数初始化为 0
        # 如果 keep_prev 为真，则直接返回 NULL
        if ( keep_prev ) {
            if ( !new_mem ) return (NULL);
        } else {
            # 否则，在 new_mem 不为真的情况下，进行循环尝试
            while ( !new_mem ) {
                # 尝试次数加一，如果超过 10 次，则返回 NULL
                if ( ++tries > 10 ) return (NULL);
                alpha = Reduce(alpha);  # 调用 Reduce 函数计算 alpha
                new_len = alpha * *prev_len;  # 计算新的长度 new_len
                # 分配新的内存空间，要求大小为 new_len * lword
                new_mem = (void *) SUPERLU_MALLOC((size_t)new_len * lword);
            }
        }
        # 如果 type 是 LSUB 或 USUB，则复制长度为 len_to_copy 的整数内存块到新内存
        if ( type == LSUB || type == USUB ) {
            copy_mem_int(len_to_copy, expanders[type].mem, new_mem);
        } else {
            # 否则，复制长度为 len_to_copy 的双精度浮点数内存块到新内存
            copy_mem_double(len_to_copy, expanders[type].mem, new_mem);
        }
        SUPERLU_FREE (expanders[type].mem);  # 释放旧内存块
    }
    expanders[type].mem = (void *) new_mem;  # 将 expanders[type] 的 mem 指向新内存块
    
    } else { /* MemModel == USER */
    
    # 如果 Glu 结构体中的 num_expansions 等于 0，则进行以下操作，表示首次初始化
    if ( Glu->num_expansions == 0 ) { /* First time initialization */
    
        # 调用 duser_malloc 函数分配新内存，大小为 new_len * lword，使用 HEAD 参数
        new_mem = duser_malloc(new_len * lword, HEAD, Glu);
        # 如果 new_mem 不是双精度对齐的，并且 type 是 LUSUP 或 UCOL，则进行双精度对齐操作
        if ( NotDoubleAlign(new_mem) &&
        (type == LUSUP || type == UCOL) ) {
            old_mem = new_mem;  # 保存旧的内存地址
            new_mem = (void *)DoubleAlign(new_mem);  # 调用 DoubleAlign 函数进行双精度对齐
            extra = (char*)new_mem - (char*)old_mem;  # 计算额外的字节偏移量
#if ( DEBUGlevel>=1 )
        printf("expand(): not aligned, extra %d\n", extra);
#endif        
        // 假设调试级别高于等于1时，打印调试信息，显示额外的内存空间
        Glu->stack.top1 += extra;
        // 堆栈顶部位置增加额外的空间
        Glu->stack.used += extra;
        // 堆栈使用的空间增加额外的空间
        }
        
        expanders[type].mem = (void *) new_mem;
        // 将扩展后的内存分配给对应类型的内存指针
        
    } else { /* CASE: num_expansions != 0 */
    
        tries = 0;
        // 尝试次数初始化为0
        extra = (new_len - *prev_len) * lword;
        // 计算额外需要的内存空间
        
        if ( keep_prev ) {
        if ( StackFull(extra) ) return (NULL);
        // 如果保持之前状态，并且堆栈已满，则返回空指针
        } else {
        while ( StackFull(extra) ) {
            // 当堆栈已满时循环
            if ( ++tries > 10 ) return (NULL);
            // 如果尝试次数超过10次，则返回空指针
            alpha = Reduce(alpha);
            // 减少 alpha 的值
            new_len = alpha * *prev_len;
            // 根据新的 alpha 值计算新的长度
            extra = (new_len - *prev_len) * lword;        
        }
        }

          /* Need to expand the memory: moving the content after the current MemType
               to make extra room for the current MemType.
                   Memory layout: [ LUSUP || UCOL || LSUB || USUB ]
          */
          // 需要扩展内存：将当前 MemType 后面的内容移动，为当前 MemType 留出额外的空间
          if ( type != USUB ) {
        new_mem = (void*)((char*)expanders[type + 1].mem + extra);
        // 计算新内存的位置
        bytes_to_copy = (char*)Glu->stack.array + Glu->stack.top1
            - (char*)expanders[type + 1].mem;
        // 计算需要复制的字节数
        user_bcopy(expanders[type+1].mem, new_mem, bytes_to_copy);
        // 使用自定义的内存拷贝函数复制内存

        if ( type < USUB ) {
            Glu->usub = expanders[USUB].mem =
            (void*)((char*)expanders[USUB].mem + extra);
        // 更新 USUB 类型的内存指针
        }
        if ( type < LSUB ) {
            Glu->lsub = expanders[LSUB].mem =
            (void*)((char*)expanders[LSUB].mem + extra);
        // 更新 LSUB 类型的内存指针
        }
        if ( type < UCOL ) {
            Glu->ucol = expanders[UCOL].mem =
            (void*)((char*)expanders[UCOL].mem + extra);
        // 更新 UCOL 类型的内存指针
        }
        Glu->stack.top1 += extra;
        // 堆栈顶部位置增加额外的空间
        Glu->stack.used += extra;
        // 堆栈使用的空间增加额外的空间
        if ( type == UCOL ) {
            Glu->stack.top1 += extra;   /* Add same amount for USUB */
            Glu->stack.used += extra;
        // 如果类型是 UCOL，则为 USUB 也添加相同的额外空间
        }
        
        } /* end expansion */

    } /* else ... */
    }

    expanders[type].size = new_len;
    // 更新 expanders 数组中对应类型的大小
    *prev_len = new_len;
    // 更新前一个长度
    if ( Glu->num_expansions ) ++Glu->num_expansions;
    // 如果存在扩展操作，则增加扩展次数
    
    return (void *) expanders[type].mem;
    // 返回对应类型的内存指针
    
} /* dexpand */


/*! \brief Compress the work[] array to remove fragmentation.
 */
void
dStackCompress(GlobalLU_t *Glu)
{
    register int iword, dword, ndim;
    char     *last, *fragment;
    int_t    *ifrom, *ito;
    double   *dfrom, *dto;
    int_t    *xlsub, *lsub, *xusub, *usub, *xlusup;
    double   *ucol, *lusup;
    
    iword = sizeof(int);
    // 计算 int 类型的字节数
    dword = sizeof(double);
    // 计算 double 类型的字节数
    ndim = Glu->n;
    // 获取 GlobalLU_t 结构体中的维度信息

    xlsub  = Glu->xlsub;
    lsub   = Glu->lsub;
    xusub  = Glu->xusub;
    usub   = Glu->usub;
    xlusup = Glu->xlusup;
    ucol   = Glu->ucol;
    lusup  = Glu->lusup;
    // 获取 GlobalLU_t 结构体中的各个成员的指针或地址
    
    dfrom = ucol;
    // 将 ucol 指针赋值给 dfrom
    dto = (double *)((char*)lusup + xlusup[ndim] * dword);
    // 计算 dto 的地址
    copy_mem_double(xusub[ndim], dfrom, dto);
    // 调用函数复制 double 类型的内存内容
    ucol = dto;

    ifrom = lsub;
    // 将 lsub 指针赋值给 ifrom
    ito = (int_t *) ((char*)ucol + xusub[ndim] * iword);
    // 计算 ito 的地址
    copy_mem_int(xlsub[ndim], ifrom, ito);
    // 调用函数复制 int_t 类型的内存内容
    lsub = ito;
    
    ifrom = usub;
    // 将 usub 指针赋值给 ifrom
    # 将 lsub 的地址加上 xlsub[ndim] * iword 的偏移量，然后转换为 int_t 类型指针，赋给 ito
    ito = (int_t *) ((char*)lsub + xlsub[ndim] * iword);
    
    # 调用函数 copy_mem_int，将 xusub[ndim] 个数据从 ifrom 复制到 ito 指向的内存区域
    copy_mem_int(xusub[ndim], ifrom, ito);
    
    # 将 ito 的地址赋给 usub，即更新 usub 指向的内存区域
    usub = ito;
    
    # 计算 usub 后面的内存区域的起始地址，使得 last 指向该地址
    last = (char*)usub + xusub[ndim] * iword;
    
    # 计算从 Glu->stack.array 的顶部到 last 之间的距离，更新 stack 的 used 和 top1 属性
    fragment = (char*) (((char*)Glu->stack.array + Glu->stack.top1) - last);
    Glu->stack.used -= (long int) fragment;
    Glu->stack.top1 -= (long int) fragment;
    
    # 将 ucol 赋给 Glu 结构体的 ucol 成员变量
    Glu->ucol = ucol;
    
    # 将 lsub 赋给 Glu 结构体的 lsub 成员变量
    Glu->lsub = lsub;
    
    # 将 usub 赋给 Glu 结构体的 usub 成员变量
    Glu->usub = usub;
#if ( DEBUGlevel>=1 )
    // 如果 DEBUGlevel 大于等于 1，打印片段号
    printf("dStackCompress: fragment %lld\n", (long long) fragment);
    /* for (last = 0; last < ndim; ++last)
    print_lu_col("After compress:", last, 0);*/
#endif    
    
}

/*! \brief Allocate storage for original matrix A
 */
void
dallocateA(int n, int_t nnz, double **a, int_t **asub, int_t **xa)
{
    // 分配空间存储原始矩阵 A
    *a    = (double *) doubleMalloc(nnz);
    *asub = (int_t *) intMalloc(nnz);
    *xa   = (int_t *) intMalloc(n+1);
}


double *doubleMalloc(size_t n)
{
    // 分配 n 个 double 类型的内存空间
    double *buf;
    buf = (double *) SUPERLU_MALLOC(n * (size_t) sizeof(double)); 
    if ( !buf ) {
    // 如果分配失败，输出错误信息并终止程序
    ABORT("SUPERLU_MALLOC failed for buf in doubleMalloc()\n");
    }
    return (buf);
}

double *doubleCalloc(size_t n)
{
    // 分配 n 个 double 类型的内存空间，并初始化为 0.0
    double *buf;
    register size_t i;
    double zero = 0.0;
    buf = (double *) SUPERLU_MALLOC(n * (size_t) sizeof(double));
    if ( !buf ) {
    // 如果分配失败，输出错误信息并终止程序
    ABORT("SUPERLU_MALLOC failed for buf in doubleCalloc()\n");
    }
    for (i = 0; i < n; ++i) buf[i] = zero;
    return (buf);
}


int_t dmemory_usage(const int_t nzlmax, const int_t nzumax,
          const int_t nzlumax, const int n)
{
    // 计算内存使用量的估计
    register int iword, liword, dword;

    iword   = sizeof(int);
    liword  = sizeof(int_t);
    dword   = sizeof(double);
    
    return (10 * n * iword +
        nzlmax * liword + nzumax * (liword + dword) + nzlumax * dword);
}
```