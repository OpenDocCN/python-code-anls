# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\memory.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file memory.c
 * \brief Precision-independent memory-related routines
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 * </pre>
 */
/** Precision-independent memory-related routines.
    (Shared by [sdcz]memory.c) **/

#include "slu_ddefs.h"

#if ( DEBUGlevel>=1 )           /* Debug malloc/free. */
int_t superlu_malloc_total = 0;

#define PAD_FACTOR  2
#define DWORD  (sizeof(double)) /* Be sure it's no smaller than double. */
/* size_t is usually defined as 'unsigned long' */

/*! \brief Allocate memory with debugging information.
 *
 *  \param size Size of memory block to allocate.
 *  \return Pointer to allocated memory.
 */
void *superlu_malloc(size_t size)
{
    char *buf;

    buf = (char *) malloc(size + DWORD);
    if ( !buf ) {
        printf("superlu_malloc fails: malloc_total %.0f MB, size %lld\n",
               superlu_malloc_total*1e-6, (long long)size);
        ABORT("superlu_malloc: out of memory");
    }

    ((size_t *) buf)[0] = size;
#if 0
    superlu_malloc_total += size + DWORD;
#else
    superlu_malloc_total += size;
#endif
    return (void *) (buf + DWORD);
}

/*! \brief Free memory allocated by superlu_malloc.
 *
 *  \param addr Pointer to memory block to free.
 */
void superlu_free(void *addr)
{
    char *p = ((char *) addr) - DWORD;

    if ( !addr )
        ABORT("superlu_free: tried to free NULL pointer");

    if ( !p )
        ABORT("superlu_free: tried to free NULL+DWORD pointer");

    { 
        int_t n = ((size_t *) p)[0];
        
        if ( !n )
            ABORT("superlu_free: tried to free a freed pointer");
        *((size_t *) p) = 0; /* Set to zero to detect duplicate free's. */
#if 0    
        superlu_malloc_total -= (n + DWORD);
#else
        superlu_malloc_total -= n;
#endif

        if ( superlu_malloc_total < 0 )
            ABORT("superlu_malloc_total went negative!");
        
        /*free (addr);*/
        free (p);
    }
}

#else   /* production mode */

/*! \brief Allocate memory in production mode.
 *
 *  \param size Size of memory block to allocate.
 *  \return Pointer to allocated memory.
 */
void *superlu_malloc(size_t size)
{
    void *buf;
    buf = (void *) malloc(size);
    return (buf);
}

/*! \brief Free memory in production mode.
 *
 *  \param addr Pointer to memory block to free.
 */
void superlu_free(void *addr)
{
    free (addr);
}

#endif

/*! \brief Set up pointers for integer working arrays.
 *
 *  \param m Number of rows in the matrix.
 *  \param n Number of columns in the matrix.
 *  \param panel_size Size of the panel.
 *  \param iworkptr Pointer to the integer working array.
 *  \param segrep Pointer to segment representation array.
 *  \param parent Pointer to parent array.
 *  \param xplore Pointer to explore array.
 *  \param repfnz Pointer to repfnz array.
 *  \param panel_lsub Pointer to panel_lsub array.
 *  \param xprune Pointer to prune array.
 *  \param marker Pointer to marker array.
 */
void
SetIWork(int m, int n, int panel_size, int *iworkptr, int **segrep,
     int **parent, int_t **xplore, int **repfnz, int **panel_lsub,
     int_t **xprune, int **marker)
{
    *segrep = iworkptr;
    *parent = iworkptr + m;
    //    *xplore = *parent + m;
    *repfnz = *parent + m;
    *panel_lsub = *repfnz + panel_size * m;
    //    *xprune = *panel_lsub + panel_size * m;
    // *marker = *xprune + n;
    *marker = *panel_lsub + panel_size * m;
    
    ifill (*repfnz, m * panel_size, EMPTY);
    ifill (*panel_lsub, m * panel_size, EMPTY);
    
    *xplore = intMalloc(m); /* can be 64 bit */
    *xprune = intMalloc(n);
}

/*! \brief Copy integer memory from old to new.
 *
 *  \param howmany Number of elements to copy.
 *  \param old Pointer to old memory block.
 *  \param new Pointer to new memory block.
 */
void
copy_mem_int(int_t howmany, void *old, void *new)
{
    // Implementation of copying memory from old to new
}
    # 声明一个整型变量 i 用于循环计数
    register int_t i;
    # 创建指针 iold 指向数组 old 的首地址
    int_t *iold = old;
    # 创建指针 inew 指向数组 new 的首地址
    int_t *inew = new;
    # 循环，将数组 old 中的元素逐个复制给数组 new 对应位置的元素
    for (i = 0; i < howmany; i++) inew[i] = iold[i];
}

// 将源地址到目标地址的字节内容反向复制，长度为指定字节数
void
user_bcopy(char *src, char *dest, int bytes)
{
    char *s_ptr, *d_ptr;

    // 指向源和目标的指针，分别指向最后一个字节
    s_ptr = src + bytes - 1;
    d_ptr = dest + bytes - 1;
    // 逐个字节复制，从末尾向前复制
    for (; d_ptr >= dest; --s_ptr, --d_ptr ) *d_ptr = *s_ptr;
}

// 分配指定数量的 int 类型内存，返回指针
int *int32Malloc(int n)
{
    int *buf;
    buf = (int *) SUPERLU_MALLOC((size_t) n * sizeof(int));
    if ( !buf ) {
    ABORT("SUPERLU_MALLOC fails for buf in int32Malloc()");
    }
    return (buf);
}

// 分配指定数量的 int_t 类型内存，返回指针
int_t *intMalloc(int_t n)
{
    int_t *buf;
    buf = (int_t *) SUPERLU_MALLOC((size_t) n * sizeof(int_t));
    if ( !buf ) {
    ABORT("SUPERLU_MALLOC fails for buf in intMalloc()");
    }
    return (buf);
}

// 分配指定数量的 int 类型内存并初始化为零，返回指针
int *int32Calloc(int n)
{
    int *buf;
    register int i;
    buf = (int *) SUPERLU_MALLOC(n * sizeof(int));
    if ( !buf ) {
    ABORT("SUPERLU_MALLOC fails for buf in intCalloc()");
    }
    // 初始化分配的内存为零
    for (i = 0; i < n; ++i) buf[i] = 0;
    return (buf);
}

// 分配指定数量的 int_t 类型内存并初始化为零，返回指针
int_t *intCalloc(int_t n)
{
    int_t *buf;
    register int_t i;
    buf = (int_t *) SUPERLU_MALLOC(n * sizeof(int_t));
    if ( !buf ) {
    ABORT("SUPERLU_MALLOC fails for buf in intCalloc()");
    }
    // 初始化分配的内存为零
    for (i = 0; i < n; ++i) buf[i] = 0;
    return (buf);
}
```