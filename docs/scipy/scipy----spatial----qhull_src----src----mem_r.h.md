# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\mem_r.h`

```
/*
   mem_r.h
     prototypes for memory management functions

   see qh-mem_r.htm, mem_r.c and qset_r.h

   for error handling, writes message and calls
     qh_errexit(qhT *qh, qhmem_ERRmem, NULL, NULL) if insufficient memory
       and
     qh_errexit(qhT *qh, qhmem_ERRqhull, NULL, NULL) otherwise

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/mem_r.h#5 $$Change: 2698 $
   $DateTime: 2019/06/24 14:52:34 $$Author: bbarber $
*/

#ifndef qhDEFmem
#define qhDEFmem 1

#include <stdio.h>

#ifndef DEFsetT
#define DEFsetT 1
typedef struct setT setT;          /* defined in qset_r.h */
#endif

#ifndef DEFqhT
#define DEFqhT 1
typedef struct qhT qhT;          /* defined in libqhull_r.h */
#endif

/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="NOmem">-</a>

  qh_NOmem
    turn off quick-fit memory allocation

  notes:
    mem_r.c implements Quickfit memory allocation for about 20% time
    savings.  If it fails on your machine, try to locate the
    problem, and send the answer to qhull@qhull.org.  If this can
    not be done, define qh_NOmem to use malloc/free instead.

    #define qh_NOmem
*/

/*-<a                             href="qh-mem_r.htm#TOC"
>-------------------------------</a><a name="TRACEshort">-</a>

qh_TRACEshort
Trace short and quick memory allocations at T5

*/
#define qh_TRACEshort

/*-------------------------------------------
    to avoid bus errors, memory allocation must consider alignment requirements.
    malloc() automatically takes care of alignment.   Since mem_r.c manages
    its own memory, we need to explicitly specify alignment in
    qh_meminitbuffers().

    A safe choice is sizeof(double).  sizeof(float) may be used if doubles
    do not occur in data structures and pointers are the same size.  Be careful
    of machines (e.g., DEC Alpha) with large pointers.  If gcc is available,
    use __alignof__(double) or fmax_(__alignof__(float), __alignof__(void *)).

   see <a href="user_r.h#MEMalign">qh_MEMalign</a> in user_r.h for qhull's alignment
*/

#define qhmem_ERRmem 4    /* matches qh_ERRmem in libqhull_r.h */
#define qhmem_ERRqhull 5  /* matches qh_ERRqhull in libqhull_r.h */

/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="ptr_intT">-</a>

  ptr_intT
    for casting a void * to an integer-type that holds a pointer
    Used for integer expressions (e.g., computing qh_gethash() in poly_r.c)

  notes:
    WARN64 -- these notes indicate 64-bit issues
    On 64-bit machines, a pointer may be larger than an 'int'.
    qh_meminit()/mem_r.c checks that 'ptr_intT' holds a 'void*'
    ptr_intT is typically a signed value, but not necessarily so
    size_t is typically unsigned, but should match the parameter type

*/
    # Qhull 使用 int 而非 size_t，除了像 malloc、qsort、qh_malloc 等系统调用外。
    # 这与 Qt 的约定相符，且更易于处理。
/*
#if (defined(__MINGW64__)) && defined(_WIN64)
typedef long long ptr_intT;
#elif defined(_MSC_VER) && defined(_WIN64)
typedef long long ptr_intT;
#else
typedef long ptr_intT;
#endif
*/

/*
   -<a                             href="qh-mem_r.htm#TOC"
     >--------------------------------</a><a name="qhmemT">-</a>
   
   qhmemT
     global memory structure for mem_r.c

   notes:
     users should ignore qhmem except for writing extensions
     qhmem is allocated in mem_r.c

     qhmem could be swapable like qh and qhstat, but then
     multiple qh's and qhmem's would need to keep in synch.
     A swapable qhmem would also waste memory buffers.  As long
     as memory operations are atomic, there is no problem with
     multiple qh structures being active at the same time.
     If you need separate address spaces, you can swap the
     contents of qh->qhmem.
*/
typedef struct qhmemT qhmemT;



/*
   This block of code defines a typedef based on conditional compilation directives for different compilers and platforms.

   #if (defined(__MINGW64__)) && defined(_WIN64)
   - If the current platform is MinGW-w64 and the target is 64-bit Windows, define ptr_intT as long long.
   
   #elif defined(_MSC_VER) && defined(_WIN64)
   - Otherwise, if the current compiler is MSVC and the target is 64-bit Windows, define ptr_intT as long long.
   
   #else
   - In all other cases (assuming a different compiler or a different platform), define ptr_intT as long.
   
   #endif

   The typedef defines ptr_intT as the type used for pointer integers in subsequent code. It ensures compatibility and correct size of pointer integers across different platforms and compilers.
*/
struct qhmemT {               /* global memory management variables */
  int      BUFsize;           /* size of memory allocation buffer */
  int      BUFinit;           /* initial size of memory allocation buffer */
  int      TABLEsize;         /* actual number of sizes in free list table */
  int      NUMsizes;          /* maximum number of sizes in free list table */
  int      LASTsize;          /* last size in free list table */
  int      ALIGNmask;         /* worst-case alignment, must be 2^n-1 */
  void   **freelists;          /* free list table, linked by offset 0 */
  int     *sizetable;         /* size of each freelist */
  int     *indextable;        /* size->index table */
  void    *curbuffer;         /* current buffer, linked by offset 0 */
  void    *freemem;           /*   free memory in curbuffer */
  int      freesize;          /*   size of freemem in bytes */
  setT    *tempstack;         /* stack of temporary memory, managed by users */
  FILE    *ferr;              /* file for reporting errors when 'qh' may be undefined */
  int      IStracing;         /* =5 if tracing memory allocations */
  int      cntquick;          /* count of quick allocations */
                              /* Note: removing statistics doesn't effect speed */
  int      cntshort;          /* count of short allocations */
  int      cntlong;           /* count of long allocations */
  int      freeshort;         /* count of short memfrees */
  int      freelong;          /* count of long memfrees */
  int      totbuffer;         /* total short memory buffers minus buffer links */
  int      totdropped;        /* total dropped memory at end of short memory buffers (e.g., freesize) */
  int      totfree;           /* total size of free, short memory on freelists */
  int      totlong;           /* total size of long memory in use */
  int      maxlong;           /*   maximum totlong */
  int      totshort;          /* total size of short memory in use */
  int      totunused;         /* total unused short memory (estimated, short size - request size of first allocations) */
  int      cntlarger;         /* count of setlarger's */
  int      totlarger;         /* total copied by setlarger */
};


/*==================== -macros ====================*/

/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="memalloc_">-</a>

  qh_memalloc_(qh, insize, freelistp, object, type)
    returns object of size bytes
        assumes size<=qh->qhmem.LASTsize and void **freelistp is a temp
*/

#if defined qh_NOmem
#define qh_memalloc_(qh, insize, freelistp, object, type) {\
  (void)freelistp; /* Avoid warnings */ \
  object= (type *)qh_memalloc(qh, insize); }
#elif defined qh_TRACEshort
#define qh_memalloc_(qh, insize, freelistp, object, type) {\
  (void)freelistp; /* Avoid warnings */ \
  object= (type *)qh_memalloc(qh, insize); }
#else /* !qh_NOmem */


注释：

struct qhmemT {               /* 全局内存管理变量 */
  int      BUFsize;           /* 内存分配缓冲区大小 */
  int      BUFinit;           /* 内存分配缓冲区初始大小 */
  int      TABLEsize;         /* 空闲列表表中实际大小数目 */
  int      NUMsizes;          /* 空闲列表表中的最大大小数目 */
  int      LASTsize;          /* 空闲列表表中的最后一个大小 */
  int      ALIGNmask;         /* 最坏情况对齐，必须是2^n-1 */
  void   **freelists;          /* 空闲列表表，通过偏移量0链接 */
  int     *sizetable;         /* 每个空闲列表的大小 */
  int     *indextable;        /* 大小到索引的转换表 */
  void    *curbuffer;         /* 当前缓冲区，通过偏移量0链接 */
  void    *freemem;           /* 当前缓冲区中的空闲内存 */
  int      freesize;          /* freemem的字节大小 */
  setT    *tempstack;         /* 临时内存栈，由用户管理 */
  FILE    *ferr;              /* 当'qh'可能未定义时用于报告错误的文件 */
  int      IStracing;         /* 如果跟踪内存分配则为5 */
  int      cntquick;          /* 快速分配的计数 */
                              /* 注意：移除统计信息不影响速度 */
  int      cntshort;          /* 短期分配的计数 */
  int      cntlong;           /* 长期分配的计数 */
  int      freeshort;         /* 短期内存释放的计数 */
  int      freelong;          /* 长期内存释放的计数 */
  int      totbuffer;         /* 总的短期内存缓冲区大小，减去缓冲区链接大小 */
  int      totdropped;        /* 短期内存缓冲区末尾丢弃的内存总量（例如freesize） */
  int      totfree;           /* 空闲短期内存总大小 */
  int      totlong;           /* 使用中的长期内存总大小 */
  int      maxlong;           /* 最大的长期内存总大小 */
  int      totshort;          /* 使用中的短期内存总大小 */
  int      totunused;         /* 总的未使用的短期内存（估计值，短期大小 - 第一次分配的请求大小） */
  int      cntlarger;         /* setlarger的调用次数 */
  int      totlarger;         /* 被setlarger复制的总量 */
};


/*==================== -macros ====================*/

/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="memalloc_">-</a>

  qh_memalloc_(qh, insize, freelistp, object, type)
    返回大小为insize字节的对象
        假设size<=qh->qhmem.LASTsize，并且void **freelistp是临时变量
*/

#if defined qh_NOmem
#define qh_memalloc_(qh, insize, freelistp, object, type) {\
  (void)freelistp; /* 避免警告 */ \
  object= (type *)qh_memalloc(qh, insize); }
#elif defined qh_TRACEshort
#define qh_memalloc_(qh, insize, freelistp, object, type) {\
  (void)freelistp; /* 避免警告 */ \
  object= (type *)qh_memalloc(qh, insize); }
#else /* !qh_NOmem */


注释部分已按要求对每行代码进行了逐行注释，解释了变量的含义和宏的作用。
/* 定义宏 qh_memalloc_

   qh_memalloc_(qh, insize, freelistp, object, type) {\
   为快速分配内存对象而设计的宏定义

   freelistp= qh->qhmem.freelists + qh->qhmem.indextable[insize];\
   设置指向可用对象列表的指针

   if ((object= (type *)*freelistp)) {\
   如果列表中有可用对象，则执行以下操作：

   qh->qhmem.totshort += qh->qhmem.sizetable[qh->qhmem.indextable[insize]]; \
   增加已分配的短期内存总量

   qh->qhmem.totfree -= qh->qhmem.sizetable[qh->qhmem.indextable[insize]]; \
   减少可用的空闲内存总量

   qh->qhmem.cntquick++;  \
   增加快速分配计数器

   *freelistp= *((void **)*freelistp);\
   更新可用对象列表指针

   }else object= (type *)qh_memalloc(qh, insize);}

   否则，调用通用的内存分配函数进行对象分配
*/
#endif

/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="memfree_">-</a>

  qh_memfree_(qh, object, insize, freelistp)
    free up an object

  notes:
    object may be NULL
    assumes size<=qh->qhmem.LASTsize and void **freelistp is a temp
*/
#if defined qh_NOmem
/* 定义宏 qh_memfree_

   qh_memfree_(qh, object, insize, freelistp) {\
   完成对象释放的宏定义

   (void)freelistp; /* 避免警告 */ \
   忽略 freelistp 的使用，以避免编译器警告

   qh_memfree(qh, object, insize); }
   调用通用的内存释放函数
*/
#elif defined qh_TRACEshort
/* 定义宏 qh_memfree_

   qh_memfree_(qh, object, insize, freelistp) {\
   完成对象释放的宏定义

   (void)freelistp; /* 避免警告 */ \
   忽略 freelistp 的使用，以避免编译器警告

   qh_memfree(qh, object, insize); }
   调用通用的内存释放函数
*/
#else /* !qh_NOmem */

/* 定义宏 qh_memfree_

   qh_memfree_(qh, object, insize, freelistp) {\
   完成对象释放的宏定义

   if (object) { \
   如果对象不为空，则执行以下操作：

   qh->qhmem.freeshort++;\
   增加短期释放计数器

   freelistp= qh->qhmem.freelists + qh->qhmem.indextable[insize];\
   设置指向可用对象列表的指针

   qh->qhmem.totshort -= qh->qhmem.sizetable[qh->qhmem.indextable[insize]]; \
   减少已分配的短期内存总量

   qh->qhmem.totfree += qh->qhmem.sizetable[qh->qhmem.indextable[insize]]; \
   增加可用的空闲内存总量

   *((void **)object)= *freelistp;\
   更新可用对象列表指针

   *freelistp= object;}}
   将对象添加回可用对象列表
*/
#endif

/*=============== prototypes in alphabetical order ============*/

#ifdef __cplusplus
extern "C" {
#endif

void *qh_memalloc(qhT *qh, int insize);
void qh_memcheck(qhT *qh);
void qh_memfree(qhT *qh, void *object, int insize);
void qh_memfreeshort(qhT *qh, int *curlong, int *totlong);
void qh_meminit(qhT *qh, FILE *ferr);
void qh_meminitbuffers(qhT *qh, int tracelevel, int alignment, int numsizes,
                        int bufsize, int bufinit);
void qh_memsetup(qhT *qh);
void qh_memsize(qhT *qh, int size);
void qh_memstatistics(qhT *qh, FILE *fp);
void qh_memtotal(qhT *qh, int *totlong, int *curlong, int *totshort, int *curshort, int *maxlong, int *totbuffer);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFmem */


注释完毕。
```