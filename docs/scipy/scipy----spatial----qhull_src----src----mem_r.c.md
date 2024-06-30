# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\mem_r.c`

```
/*<html><pre>  -<a                             href="qh-mem_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

  mem_r.c
    memory management routines for qhull

  See libqhull/mem.c for a standalone program.

  To initialize memory:

    qh_meminit(qh, stderr);
    qh_meminitbuffers(qh, qh->IStracing, qh_MEMalign, 7, qh_MEMbufsize,qh_MEMinitbuf);
    qh_memsize(qh, (int)sizeof(facetT));
    qh_memsize(qh, (int)sizeof(facetT));
    ...
    qh_memsetup(qh);

  To free up all memory buffers:
    qh_memfreeshort(qh, &curlong, &totlong);

  if qh_NOmem,
    malloc/free is used instead of mem_r.c

  notes:
    uses Quickfit algorithm (freelists for commonly allocated sizes)
    assumes small sizes for freelists (it discards the tail of memory buffers)

  see:
    qh-mem_r.htm and mem_r.h
    global_r.c (qh_initbuffers) for an example of using mem_r.c

  Copyright (c) 1993-2019 The Geometry Center.
  $Id: //main/2019/qhull/src/libqhull_r/mem_r.c#6 $$Change: 2711 $
  $DateTime: 2019/06/27 22:34:56 $$Author: bbarber $
*/

#include "libqhull_r.h"  /* includes user_r.h and mem_r.h */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef qh_NOmem

/*============= internal functions ==============*/

static int qh_intcompare(const void *i, const void *j);

/*========== functions in alphabetical order ======== */

/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="intcompare">-</a>

  qh_intcompare( i, j )
    used by qsort and bsearch to compare two integers
*/
static int qh_intcompare(const void *i, const void *j) {
  return(*((const int *)i) - *((const int *)j));
} /* intcompare */


/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="memalloc">-</a>

  qh_memalloc(qh, insize )
    returns object of insize bytes
    qhmem is the global memory structure

  returns:
    pointer to allocated memory
    errors if insufficient memory

  notes:
    use explicit type conversion to avoid type warnings on some compilers
    actual object may be larger than insize
    use qh_memalloc_() for inline code for quick allocations
    logs allocations if 'T5'
    caller is responsible for freeing the memory.
    short memory is freed on shutdown by qh_memfreeshort unless qh_NOmem

  design:
    if size < qh->qhmem.LASTsize
      if qh->qhmem.freelists[size] non-empty
        return first object on freelist
      else
        round up request to size of qh->qhmem.freelists[size]
        allocate new allocation buffer if necessary
        allocate object from allocation buffer
    else
      allocate object with qh_malloc() in user_r.c
*/
void *qh_memalloc(qhT *qh, int insize) {
  // 指向空闲列表指针和新分配的内存块
  void **freelistp, *newbuffer;
  // 索引、大小、计数器
  int idx, size, n;
  // 输出大小、缓冲区大小
  int outsize, bufsize;
  // 分配的对象指针
  void *object;

  // 如果请求大小为负数，输出错误信息并退出
  if (insize<0) {
      qh_fprintf(qh, qh->qhmem.ferr, 6235, "qhull error (qh_memalloc): negative request size (%d).  Did int overflow due to high-D?\n", insize); /* WARN64 */
      qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
  }
  // 如果请求大小在允许范围内
  if (insize>=0 && insize <= qh->qhmem.LASTsize) {
    // 根据请求大小获取索引
    idx= qh->qhmem.indextable[insize];
    // 获取输出大小
    outsize= qh->qhmem.sizetable[idx];
    // 增加已分配内存统计量
    qh->qhmem.totshort += outsize;
    // 获取空闲列表指针
    freelistp= qh->qhmem.freelists+idx;
    // 如果空闲列表中有空闲对象
    if ((object= *freelistp)) {
      // 快速分配计数增加
      qh->qhmem.cntquick++;
      // 减少空闲内存统计量
      qh->qhmem.totfree -= outsize;
      // 将空闲列表更新为下一个对象
      *freelistp= *((void **)*freelistp);  /* replace freelist with next object */
#ifdef qh_TRACEshort
      // 如果启用了短内存跟踪，输出分配信息
      n= qh->qhmem.cntshort+qh->qhmem.cntquick+qh->qhmem.freeshort;
      if (qh->qhmem.IStracing >= 5)
          qh_fprintf(qh, qh->qhmem.ferr, 8141, "qh_mem %p n %8d alloc quick: %d bytes (tot %d cnt %d)\n", object, n, outsize, qh->qhmem.totshort, qh->qhmem.cntshort+qh->qhmem.cntquick-qh->qhmem.freeshort);
#endif
      // 返回分配的对象指针
      return(object);
    }else {
      // 空闲列表中没有可用对象
      qh->qhmem.cntshort++;
      // 如果需要的输出大小超过了空闲内存块大小
      if (outsize > qh->qhmem.freesize) {
        // 增加丢弃的内存统计量
        qh->qhmem.totdropped += qh->qhmem.freesize;
        // 如果当前缓冲区为空
        if (!qh->qhmem.curbuffer)
          bufsize= qh->qhmem.BUFinit;
        else
          bufsize= qh->qhmem.BUFsize;
        // 分配新的内存缓冲区
        if (!(newbuffer= qh_malloc((size_t)bufsize))) {
          qh_fprintf(qh, qh->qhmem.ferr, 6080, "qhull error (qh_memalloc): insufficient memory to allocate short memory buffer (%d bytes)\n", bufsize);
          qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
        }
        // 将新缓冲区加入到当前缓冲区列表中
        *((void **)newbuffer)= qh->qhmem.curbuffer;  /* prepend newbuffer to curbuffer list.  newbuffer!=0 by QH6080 */
        qh->qhmem.curbuffer= newbuffer;
        // 计算新内存块的空闲空间
        size= ((int)sizeof(void **) + qh->qhmem.ALIGNmask) & ~qh->qhmem.ALIGNmask;
        qh->qhmem.freemem= (void *)((char *)newbuffer+size);
        qh->qhmem.freesize= bufsize - size;
        // 增加总缓冲区统计量
        qh->qhmem.totbuffer += bufsize - size; /* easier to check */
        /* Periodically test totbuffer.  It matches at beginning and exit of every call */
        // 定期检查总缓冲区统计量是否匹配
        n= qh->qhmem.totshort + qh->qhmem.totfree + qh->qhmem.totdropped + qh->qhmem.freesize - outsize;
        if (qh->qhmem.totbuffer != n) {
            qh_fprintf(qh, qh->qhmem.ferr, 6212, "qhull internal error (qh_memalloc): short totbuffer %d != totshort+totfree... %d\n", qh->qhmem.totbuffer, n);
            qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
        }
      }
      // 分配新的对象内存块
      object= qh->qhmem.freemem;
      qh->qhmem.freemem= (void *)((char *)qh->qhmem.freemem + outsize);
      qh->qhmem.freesize -= outsize;
      // 增加未使用内存统计量
      qh->qhmem.totunused += outsize - insize;
#ifdef qh_TRACEshort
      // 如果启用了短跟踪，计算当前内存使用情况
      n= qh->qhmem.cntshort+qh->qhmem.cntquick+qh->qhmem.freeshort;
      // 如果跟踪级别高于等于5，输出内存分配信息到错误流中
      if (qh->qhmem.IStracing >= 5)
          qh_fprintf(qh, qh->qhmem.ferr, 8140, "qh_mem %p n %8d alloc short: %d bytes (tot %d cnt %d)\n", object, n, outsize, qh->qhmem.totshort, qh->qhmem.cntshort+qh->qhmem.cntquick-qh->qhmem.freeshort);
#endif
      // 返回分配的对象指针
      return object;
    }
  }else {                     /* 长分配 */
    // 如果未初始化索引表，则输出错误信息并终止程序
    if (!qh->qhmem.indextable) {
      qh_fprintf(qh, qh->qhmem.ferr, 6081, "qhull internal error (qh_memalloc): qhmem has not been initialized.\n");
      qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
    }
    // 设置输出大小为输入大小，并增加长分配计数
    outsize= insize;
    qh->qhmem.cntlong++;
    qh->qhmem.totlong += outsize;
    // 更新最大长分配大小
    if (qh->qhmem.maxlong < qh->qhmem.totlong)
      qh->qhmem.maxlong= qh->qhmem.totlong;
    // 分配内存，如果分配失败则输出错误信息并终止程序
    if (!(object= qh_malloc((size_t)outsize))) {
      qh_fprintf(qh, qh->qhmem.ferr, 6082, "qhull error (qh_memalloc): insufficient memory to allocate %d bytes\n", outsize);
      qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
    }
    // 如果跟踪级别高于等于5，输出长分配信息到错误流中
    if (qh->qhmem.IStracing >= 5)
      qh_fprintf(qh, qh->qhmem.ferr, 8057, "qh_mem %p n %8d alloc long: %d bytes (tot %d cnt %d)\n", object, qh->qhmem.cntlong+qh->qhmem.freelong, outsize, qh->qhmem.totlong, qh->qhmem.cntlong-qh->qhmem.freelong);
  }
  // 返回分配的对象指针
  return(object);
} /* memalloc */


/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="memcheck">-</a>

  qh_memcheck(qh)
*/
void qh_memcheck(qhT *qh) {
  int i, count, totfree= 0;
  void *object;

  // 如果 qh 为空指针，则输出错误信息并终止程序
  if (!qh) {
    qh_fprintf_stderr(6243, "qhull internal error (qh_memcheck): qh is 0.  It does not point to a qhT\n");
    qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
  }
  // 检查 qh->qhmem 的状态，如果异常则输出错误信息并终止程序
  if (qh->qhmem.ferr == 0 || qh->qhmem.IStracing < 0 || qh->qhmem.IStracing > 10 || (((qh->qhmem.ALIGNmask+1) & qh->qhmem.ALIGNmask) != 0)) {
    qh_fprintf_stderr(6244, "qhull internal error (qh_memcheck): either qh->qhmem is overwritten or qh->qhmem is not initialized.  Call qh_meminit or qh_new_qhull before calling qh_mem routines.  ferr 0x%x, IsTracing %d, ALIGNmask 0x%x\n", 
          qh->qhmem.ferr, qh->qhmem.IStracing, qh->qhmem.ALIGNmask);
    qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
  }
  // 如果启用跟踪，输出内存检查信息到错误流中
  if (qh->qhmem.IStracing != 0)
    qh_fprintf(qh, qh->qhmem.ferr, 8143, "qh_memcheck: check size of freelists on qh->qhmem\nqh_memcheck: A segmentation fault indicates an overwrite of qh->qhmem\n");
  // 遍历自由列表，计算总空闲内存大小
  for (i=0; i < qh->qhmem.TABLEsize; i++) {
    count=0;
    for (object= qh->qhmem.freelists[i]; object; object= *((void **)object))
      count++;
    totfree += qh->qhmem.sizetable[i] * count;
  }
  // 如果计算出的总空闲内存大小与记录的不一致，则输出错误信息并终止程序
  if (totfree != qh->qhmem.totfree) {
    qh_fprintf(qh, qh->qhmem.ferr, 6211, "qhull internal error (qh_memcheck): totfree %d not equal to freelist total %d\n", qh->qhmem.totfree, totfree);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  // 如果启用跟踪，输出内存检查信息到错误流中
  if (qh->qhmem.IStracing != 0)
    // 继续上一个函数的输出
    // 使用 qh_fprintf 函数向 qh->qhmem.ferr 文件中写入格式化输出，将字符串常量和变量 totfree 作为参数传递
    qh_fprintf(qh, qh->qhmem.ferr, 8144, "qh_memcheck: total size of freelists totfree is the same as qh->qhmem.totfree\n", totfree);
/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="memfree">-</a>

  qh_memfree(qh, object, insize )
    free up an object of size bytes
    size is insize from qh_memalloc

  notes:
    object may be NULL
    type checking warns if using (void **)object
    use qh_memfree_() for quick free's of small objects

  design:
    if size <= qh->qhmem.LASTsize
      append object to corresponding freelist
    else
      call qh_free(object)
*/
void qh_memfree(qhT *qh, void *object, int insize) {
  void **freelistp;
  int idx, outsize;

  // 如果对象为空，则直接返回，不进行释放操作
  if (!object)
    return;
  
  // 如果对象大小不超过上次分配的大小
  if (insize <= qh->qhmem.LASTsize) {
    qh->qhmem.freeshort++;
    idx= qh->qhmem.indextable[insize];
    outsize= qh->qhmem.sizetable[idx];
    qh->qhmem.totfree += outsize;
    qh->qhmem.totshort -= outsize;
    freelistp= qh->qhmem.freelists + idx;
    *((void **)object)= *freelistp;
    *freelistp= object;
    
    // 如果启用了短内存追踪，则输出追踪信息
#ifdef qh_TRACEshort
    idx= qh->qhmem.cntshort+qh->qhmem.cntquick+qh->qhmem.freeshort;
    if (qh->qhmem.IStracing >= 5)
        qh_fprintf(qh, qh->qhmem.ferr, 8142, "qh_mem %p n %8d free short: %d bytes (tot %d cnt %d)\n", object, idx, outsize, qh->qhmem.totshort, qh->qhmem.cntshort+qh->qhmem.cntquick-qh->qhmem.freeshort);
#endif
  } else {
    // 对于大对象，直接释放，并输出相关追踪信息
    qh->qhmem.freelong++;
    qh->qhmem.totlong -= insize;
    if (qh->qhmem.IStracing >= 5)
      qh_fprintf(qh, qh->qhmem.ferr, 8058, "qh_mem %p n %8d free long: %d bytes (tot %d cnt %d)\n", object, qh->qhmem.cntlong+qh->qhmem.freelong, insize, qh->qhmem.totlong, qh->qhmem.cntlong-qh->qhmem.freelong);
    qh_free(object);
  }
} /* memfree */


/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="memfreeshort">-</a>

  qh_memfreeshort(qh, curlong, totlong )
    frees up all short and qhmem memory allocations

  returns:
    number and size of current long allocations

  notes:
    if qh_NOmem (qh_malloc() for all allocations),
       short objects (e.g., facetT) are not recovered.
       use qh_freeqhull(qh, qh_ALL) instead.

  see:
    qh_freeqhull(qh, allMem)
    qh_memtotal(qh, curlong, totlong, curshort, totshort, maxlong, totbuffer);
*/
void qh_memfreeshort(qhT *qh, int *curlong, int *totlong) {
  void *buffer, *nextbuffer;
  FILE *ferr;

  // 计算当前长内存的使用情况
  *curlong= qh->qhmem.cntlong - qh->qhmem.freelong;
  *totlong= qh->qhmem.totlong;
  
  // 释放当前缓冲区中的所有内存块
  for (buffer=qh->qhmem.curbuffer; buffer; buffer= nextbuffer) {
    nextbuffer= *((void **) buffer);
    qh_free(buffer);
  }
  
  // 将当前缓冲区置为空
  qh->qhmem.curbuffer= NULL;
  
  // 如果有上次分配的大小，释放相关的数据结构
  if (qh->qhmem.LASTsize) {
    qh_free(qh->qhmem.indextable);
    qh_free(qh->qhmem.freelists);
    qh_free(qh->qhmem.sizetable);
  }
  
  // 将 qhmem 结构体的所有字段清零
  ferr= qh->qhmem.ferr;
  memset((char *)&qh->qhmem, 0, sizeof(qh->qhmem));  /* every field is 0, FALSE, NULL */
  qh->qhmem.ferr= ferr;
} /* memfreeshort */


/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="meminit">-</a>

  qh_meminit(qh, ferr )
    # 初始化 qhmem 和测试 void* 的大小
    # 如果失败，不会抛出错误，而是调用 qh_exit
    initialize qhmem and test sizeof(void *)
    Does not throw errors.  qh_exit on failure
/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="meminit">-</a>

  qh_meminit(qh, ferr )
    initialize qhmem with default values
    sets ferr to stderr if ferr is NULL

      qhT *qh       pointer to the global qhull structure
      FILE *ferr    file for error messages, or NULL

      qh->qhmem     structure containing memory management settings
      memset((char *)&qh->qhmem, 0, sizeof(qh->qhmem));  /* every field is 0, FALSE, NULL */
        // Initialize qhmem structure to all zeros, indicating no allocations yet
      if (ferr)
        qh->qhmem.ferr= ferr;
        // Set ferr pointer for error messages to provided ferr, or stderr if ferr is NULL
      else
        qh->qhmem.ferr= stderr;
        // If ferr is NULL, set error output to stderr
      if (sizeof(void *) < sizeof(int)) {
        qh_fprintf(qh, qh->qhmem.ferr, 6083, "qhull internal error (qh_meminit): sizeof(void *) %d < sizeof(int) %d.  qset_r.c will not work\n", (int)sizeof(void*), (int)sizeof(int));
        qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
        // Check if void pointer size is less than integer size, output error and exit if true
      }
      if (sizeof(void *) > sizeof(ptr_intT)) {
        qh_fprintf(qh, qh->qhmem.ferr, 6084, "qhull internal error (qh_meminit): sizeof(void *) %d > sizeof(ptr_intT) %d. Change ptr_intT in mem_r.h to 'long long'\n", (int)sizeof(void*), (int)sizeof(ptr_intT));
        qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
        // Check if void pointer size is greater than ptr_intT size, output error and exit if true
      }
      qh_memcheck(qh);
        // Perform memory check for internal consistency
*/ /* meminit */

/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="meminitbuffers">-</a>

  qh_meminitbuffers(qh, tracelevel, alignment, numsizes, bufsize, bufinit )
    initialize qhmem with detailed memory settings
    if tracelevel >= 5, trace memory allocations

      qhT *qh           pointer to the global qhull structure
      int tracelevel    level of tracing for memory allocations
      int alignment     desired address alignment for memory allocations
      int numsizes      number of freelists
      int bufsize       size of additional memory buffers for short allocations
      int bufinit       size of initial memory buffer for short allocations

      qh->qhmem         structure containing memory management settings
      qh->qhmem.IStracing= tracelevel;
        // Set trace level for memory allocation tracing
      qh->qhmem.NUMsizes= numsizes;
        // Set number of freelists for memory management
      qh->qhmem.BUFsize= bufsize;
        // Set size of additional memory buffers for short allocations
      qh->qhmem.BUFinit= bufinit;
        // Set initial size of memory buffer for short allocations
      qh->qhmem.ALIGNmask= alignment-1;
        // Calculate alignment mask based on desired alignment

      if (qh->qhmem.ALIGNmask & ~qh->qhmem.ALIGNmask) {
        qh_fprintf(qh, qh->qhmem.ferr, 6085, "qhull internal error (qh_meminit): memory alignment %d is not a power of 2\n", alignment);
        qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
        // Check if alignment mask is not a power of two, output error and exit if true
      }
      qh->qhmem.sizetable= (int *) calloc((size_t)numsizes, sizeof(int));
        // Allocate memory for size table based on number of freelists
      qh->qhmem.freelists= (void **) calloc((size_t)numsizes, sizeof(void *));
        // Allocate memory for freelists based on number of freelists

      if (!qh->qhmem.sizetable || !qh->qhmem.freelists) {
        qh_fprintf(qh, qh->qhmem.ferr, 6086, "qhull error (qh_meminit): insufficient memory\n");
        qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
        // Output error and exit if memory allocation fails
      }
      if (qh->qhmem.IStracing >= 1)
        qh_fprintf(qh, qh->qhmem.ferr, 8059, "qh_meminitbuffers: memory initialized with alignment %d\n", alignment);
*/ /* meminitbuffers */

/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="memsetup">-</a>

  qh_memsetup(qh)
    set up memory after running memsize()

      qhT *qh       pointer to the global qhull structure

      qh->qhmem     structure containing memory management settings
      qsort(qh->qhmem.sizetable, (size_t)qh->qhmem.TABLEsize, sizeof(int), qh_intcompare);
        // Sort the size table for efficient memory allocation management
      qh->qhmem.LASTsize= qh->qhmem.sizetable[qh->qhmem.TABLEsize-1];
        // Determine the largest size from the sorted size table

      if (qh->qhmem.LASTsize >= qh->qhmem.BUFsize || qh->qhmem.LASTsize >= qh->qhmem.BUFinit) {
        // Check if the largest size exceeds buffer size settings
    // 使用 qh_fprintf 函数向 qh->qhmem.ferr 文件流打印错误信息，格式化输出最大内存大小 qh->qhmem.LASTsize、缓冲区大小 qh->qhmem.BUFsize 和初始缓冲区大小 qh->qhmem.BUFinit
    qh_fprintf(qh, qh->qhmem.ferr, 6087, "qhull error (qh_memsetup): largest mem size %d is >= buffer size %d or initial buffer size %d\n",
                qh->qhmem.LASTsize, qh->qhmem.BUFsize, qh->qhmem.BUFinit);
    // 使用 qh_errexit 函数终止程序，并指定错误类型为 qhmem_ERRmem，不带附加消息和文件名
    qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
    // 如果条件成立，分配内存给 qh->qhmem.indextable，分配大小为 (qh->qhmem.LASTsize+1) * sizeof(int) 字节
    if (!(qh->qhmem.indextable= (int *)qh_malloc((size_t)(qh->qhmem.LASTsize+1) * sizeof(int)))) {
        // 使用 qh_fprintf 函数向 qh->qhmem.ferr 文件流打印内存分配错误信息
        qh_fprintf(qh, qh->qhmem.ferr, 6088, "qhull error (qh_memsetup): insufficient memory\n");
        // 使用 qh_errexit 函数终止程序，并指定错误类型为 qhmem_ERRmem，不带附加消息和文件名
        qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
    }
    // 初始化 qh->qhmem.indextable 数组，使其递增填充
    for (k=qh->qhmem.LASTsize+1; k--; )
        qh->qhmem.indextable[k]= k;
    // 初始化 i 为 0
    i= 0;
    // 填充 qh->qhmem.indextable 数组，根据 qh->qhmem.sizetable 数组的值将索引值映射到更小的连续索引值
    for (k=0; k <= qh->qhmem.LASTsize; k++) {
        if (qh->qhmem.indextable[k] <= qh->qhmem.sizetable[i])
          qh->qhmem.indextable[k]= i;
        else
          qh->qhmem.indextable[k]= ++i;
    }
/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="memsize">-</a>

  qh_memsize(qh, size )
    define a free list for this size
*/
void qh_memsize(qhT *qh, int size) {
  int k;

  // 检查是否已经进行了内存设置，如果是，则报错并退出程序
  if (qh->qhmem.LASTsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6089, "qhull internal error (qh_memsize): qh_memsize called after qh_memsetup\n");
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  // 将请求的内存大小按照对齐要求进行对齐
  size= (size + qh->qhmem.ALIGNmask) & ~qh->qhmem.ALIGNmask;
  // 如果跟踪级别大于等于3，则打印快速内存分配的信息
  if (qh->qhmem.IStracing >= 3)
    qh_fprintf(qh, qh->qhmem.ferr, 3078, "qh_memsize: quick memory of %d bytes\n", size);
  // 遍历大小表，检查是否已经存在该大小的内存块，如果存在则直接返回
  for (k=qh->qhmem.TABLEsize; k--; ) {
    if (qh->qhmem.sizetable[k] == size)
      return;
  }
  // 如果大小表还有空间，将新的内存块大小添加到大小表中
  if (qh->qhmem.TABLEsize < qh->qhmem.NUMsizes)
    qh->qhmem.sizetable[qh->qhmem.TABLEsize++]= size;
  else
    // 如果大小表已满，则发出警告
    qh_fprintf(qh, qh->qhmem.ferr, 7060, "qhull warning (qh_memsize): free list table has room for only %d sizes\n", qh->qhmem.NUMsizes);
} /* memsize */


/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="memstatistics">-</a>

  qh_memstatistics(qh, fp )
    print out memory statistics

    Verifies that qh->qhmem.totfree == sum of freelists
*/
void qh_memstatistics(qhT *qh, FILE *fp) {
  int i;
  int count;
  void *object;

  // 检查内存状态，确保内存检查通过
  qh_memcheck(qh);
  // 打印内存统计信息
  qh_fprintf(qh, fp, 9278, "\nmemory statistics:\n\
%7d quick allocations\n\
%7d short allocations\n\
%7d long allocations\n\
%7d short frees\n\
%7d long frees\n\
%7d bytes of short memory in use\n\
%7d bytes of short memory in freelists\n\
%7d bytes of dropped short memory\n\
%7d bytes of unused short memory (estimated)\n\
%7d bytes of long memory allocated (max, except for input)\n\
%7d bytes of long memory in use (in %d pieces)\n\
%7d bytes of short memory buffers (minus links)\n\
%7d bytes per short memory buffer (initially %d bytes)\n",
           qh->qhmem.cntquick, qh->qhmem.cntshort, qh->qhmem.cntlong,
           qh->qhmem.freeshort, qh->qhmem.freelong,
           qh->qhmem.totshort, qh->qhmem.totfree,
           qh->qhmem.totdropped + qh->qhmem.freesize, qh->qhmem.totunused,
           qh->qhmem.maxlong, qh->qhmem.totlong, qh->qhmem.cntlong - qh->qhmem.freelong,
           qh->qhmem.totbuffer, qh->qhmem.BUFsize, qh->qhmem.BUFinit);
  // 如果存在更大的内存分配请求，则打印相应信息
  if (qh->qhmem.cntlarger) {
    qh_fprintf(qh, fp, 9279, "%7d calls to qh_setlarger\n%7.2g     average copy size\n",
           qh->qhmem.cntlarger, ((double)qh->qhmem.totlarger)/(double)qh->qhmem.cntlarger);
    qh_fprintf(qh, fp, 9280, "  freelists(bytes->count):");
  }
  // 遍历大小表，打印每个大小对应的空闲列表长度
  for (i=0; i < qh->qhmem.TABLEsize; i++) {
    count=0;
    for (object= qh->qhmem.freelists[i]; object; object= *((void **)object))
      count++;
    qh_fprintf(qh, fp, 9281, " %d->%d", qh->qhmem.sizetable[i], count);
  }
  // 打印空行，结束内存统计信息打印
  qh_fprintf(qh, fp, 9282, "\n\n");
} /* memstatistics */


/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="NOmem">-</a>

  qh_NOmem
*/
    turn off quick-fit memory allocation
    # 关闭快速适配内存分配功能

  notes:
    uses qh_malloc() and qh_free() instead
    # 注意：
    # 使用 qh_malloc() 和 qh_free() 替代
#else /* qh_NOmem */

/* 分配内存的函数，返回指向分配内存的指针 */
void *qh_memalloc(qhT *qh, int insize) {
  void *object;

  // 使用 qh_malloc 分配指定大小的内存
  if (!(object= qh_malloc((size_t)insize))) {
    // 若内存分配失败，输出错误信息并退出程序
    qh_fprintf(qh, qh->qhmem.ferr, 6090, "qhull error (qh_memalloc): insufficient memory\n");
    qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
  }
  // 统计分配的长内存的数量和总量
  qh->qhmem.cntlong++;
  qh->qhmem.totlong += insize;
  // 更新最大长内存量
  if (qh->qhmem.maxlong < qh->qhmem.totlong)
      qh->qhmem.maxlong= qh->qhmem.totlong;
  // 若开启了追踪，则输出分配信息
  if (qh->qhmem.IStracing >= 5)
    qh_fprintf(qh, qh->qhmem.ferr, 8060, "qh_mem %p n %8d alloc long: %d bytes (tot %d cnt %d)\n", object, qh->qhmem.cntlong+qh->qhmem.freelong, insize, qh->qhmem.totlong, qh->qhmem.cntlong-qh->qhmem.freelong);
  // 返回分配的内存指针
  return object;
}

/* 内存检查函数，无操作 */
void qh_memcheck(qhT *qh) {
}

/* 释放内存的函数 */
void qh_memfree(qhT *qh, void *object, int insize) {

  // 若对象为空指针，则直接返回
  if (!object)
    return;
  // 使用 qh_free 释放指定对象的内存
  qh_free(object);
  // 统计释放的长内存的数量和总量
  qh->qhmem.freelong++;
  qh->qhmem.totlong -= insize;
  // 若开启了追踪，则输出释放信息
  if (qh->qhmem.IStracing >= 5)
    qh_fprintf(qh, qh->qhmem.ferr, 8061, "qh_mem %p n %8d free long: %d bytes (tot %d cnt %d)\n", object, qh->qhmem.cntlong+qh->qhmem.freelong, insize, qh->qhmem.totlong, qh->qhmem.cntlong-qh->qhmem.freelong);
}

/* 释放所有短内存的函数 */
void qh_memfreeshort(qhT *qh, int *curlong, int *totlong) {
  // 将总长内存和当前长内存数量赋值给传入的指针变量
  *totlong= qh->qhmem.totlong;
  *curlong= qh->qhmem.cntlong - qh->qhmem.freelong;
  // 清空 qh->qhmem 结构体的所有字段
  memset((char *)&qh->qhmem, 0, sizeof(qh->qhmem));  /* every field is 0, FALSE, NULL */
}

/* 初始化 qh->qhmem 结构体的函数 */
void qh_meminit(qhT *qh, FILE *ferr) {

  // 将 qh->qhmem 结构体的所有字段清零
  memset((char *)&qh->qhmem, 0, sizeof(qh->qhmem));  /* every field is 0, FALSE, NULL */
  // 若传入了错误输出文件指针，则使用该指针；否则使用标准错误输出
  if (ferr)
      qh->qhmem.ferr= ferr;
  else
      qh->qhmem.ferr= stderr;
  // 检查 void* 的大小是否小于 int 的大小，若是则输出错误信息并退出程序
  if (sizeof(void *) < sizeof(int)) {
    qh_fprintf(qh, qh->qhmem.ferr, 6091, "qhull internal error (qh_meminit): sizeof(void *) %d < sizeof(int) %d.  qset_r.c will not work\n", (int)sizeof(void*), (int)sizeof(int));
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
}

/* 初始化 qh->qhmem 结构体的缓冲区相关字段 */
void qh_meminitbuffers(qhT *qh, int tracelevel, int alignment, int numsizes, int bufsize, int bufinit) {

  // 设置追踪级别
  qh->qhmem.IStracing= tracelevel;
}

/* 内存设置函数，无操作 */
void qh_memsetup(qhT *qh) {
}

/* 设置内存大小的函数，无操作 */
void qh_memsize(qhT *qh, int size) {
}

/* 输出内存统计信息的函数 */
void qh_memstatistics(qhT *qh, FILE *fp) {

  // 输出长内存的统计信息到指定文件指针
  qh_fprintf(qh, fp, 9409, "\nmemory statistics:\n\
%7d long allocations\n\
%7d long frees\n\
%7d bytes of long memory allocated (max, except for input)\n\
%7d bytes of long memory in use (in %d pieces)\n",
           qh->qhmem.cntlong,
           qh->qhmem.freelong,
           qh->qhmem.maxlong, qh->qhmem.totlong, qh->qhmem.cntlong - qh->qhmem.freelong);
}

#endif /* qh_NOmem */
    Does not error (for deprecated UsingLibQhull.cpp in libqhullpcpp)
/*
void qh_memtotal(qhT *qh, int *totlong, int *curlong, int *totshort, int *curshort, int *maxlong, int *totbuffer) {
    // 将总长期内存量赋给 totlong 指针所指的变量
    *totlong= qh->qhmem.totlong;
    // 计算当前长期内存使用量：总长期内存块数减去当前长期空闲内存块数
    *curlong= qh->qhmem.cntlong - qh->qhmem.freelong;
    // 将总短期内存量赋给 totshort 指针所指的变量
    *totshort= qh->qhmem.totshort;
    // 计算当前短期内存使用量：总短期内存块数加上当前快速内存块数减去当前短期空闲内存块数
    *curshort= qh->qhmem.cntshort + qh->qhmem.cntquick - qh->qhmem.freeshort;
    // 将最大长期内存块数赋给 maxlong 指针所指的变量
    *maxlong= qh->qhmem.maxlong;
    // 将总缓冲区量赋给 totbuffer 指针所指的变量
    *totbuffer= qh->qhmem.totbuffer;
} // memtotlong
*/
```