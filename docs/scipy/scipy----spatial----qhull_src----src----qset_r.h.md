# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\qset_r.h`

```
/*
   <html><pre>  -<a                             href="qh-set_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   qset_r.h
     header file for qset_r.c that implements set

   see qh-set_r.htm and qset_r.c

   only uses mem_r.c, malloc/free

   for error handling, writes message and calls
      qh_errexit(qhT *qh, qhmem_ERRqhull, NULL, NULL);

   set operations satisfy the following properties:
    - sets have a max size, the actual size (if different) is stored at the end
    - every set is NULL terminated
    - sets may be sorted or unsorted, the caller must distinguish this

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/qset_r.h#3 $$Change: 2700 $
   $DateTime: 2019/06/25 05:52:18 $$Author: bbarber $
*/

#ifndef qhDEFset
#define qhDEFset 1

#include <stdio.h>

/*================= -structures- ===============*/

#ifndef DEFsetT
#define DEFsetT 1
typedef struct setT setT;   /* a set is a sorted or unsorted array of pointers */
#endif

#ifndef DEFqhT
#define DEFqhT 1
typedef struct qhT qhT;          /* defined in libqhull_r.h */
#endif

/* [jan'15] Decided not to use countT.  Most sets are small.  The code uses signed tests */

/*-<a                                      href="qh-set_r.htm#TOC"
>----------------------------------------</a><a name="setT">-</a>

setT
  a set or list of pointers with maximum size and actual size.

variations:
  unsorted, unique   -- a list of unique pointers with NULL terminator
                           user guarantees uniqueness
  sorted             -- a sorted list of unique pointers with NULL terminator
                           qset_r.c guarantees uniqueness
  unsorted           -- a list of pointers terminated with NULL
  indexed            -- an array of pointers with NULL elements

structure for set of n elements:

        --------------
        |  maxsize
        --------------
        |  e[0] - a pointer, may be NULL for indexed sets
        --------------
        |  e[1]

        --------------
        |  ...
        --------------
        |  e[n-1]
        --------------
        |  e[n] = NULL
        --------------
        |  ...
        --------------
        |  e[maxsize] - n+1 or NULL (determines actual size of set)
        --------------

*/

/*-- setelemT -- internal type to allow both pointers and indices
*/
typedef union setelemT setelemT;
union setelemT {
  void    *p;
  int   i;         /* integer used for e[maxSize] */
};

struct setT {
  int maxsize;          /* maximum number of elements (except NULL) */
  setelemT e[1];        /* array of pointers, tail is NULL */
                        /* last slot (unless NULL) is actual size+1
                           e[maxsize]==NULL or e[e[maxsize]-1]==NULL */
                        /* this may generate a warning since e[] contains
                           maxsize elements */
};

/*=========== -constants- =========================*/
/*
   定义宏 SETelemsize，用于计算集合元素的大小（字节数）
   根据 setelemT 的大小计算得出
*/
#define SETelemsize ((int)sizeof(setelemT))


/*=========== -macros- =========================*/


/*
   宏 FOREACHsetelement_

   定义 FOREACH 迭代器，用于遍历集合中的元素

   参数说明：
     type: 元素类型
     set: 集合指针
     variable: 迭代变量名

   声明：
     假设 *variable 和 **variablep 已经声明
     "variable)" 中不允许有空格（DEC Alpha cc 编译器）

   每次迭代：
     variable 设置为当前集合元素
     variablep 设置为 variable 的下一个元素

   若要重复一个元素：
     variablep--; /* 重复 * /

   循环结束时：
     variable 在循环结束时为 NULL

   示例：
     #define FOREACHfacet_(facets) FOREACHsetelement_(facetT, facets, facet)

   注意：
     如果需要索引或包含 NULL 元素，请使用 FOREACHsetelement_i_()
     假设 set 在迭代过程中不被修改

   警告：
     嵌套循环不能使用相同的变量（需定义另一个 FOREACH）
     如果嵌套在另一个 FOREACH 内部，则需要使用大括号

     这包括中间的块，例如 FOREACH...{ if () FOREACH...} )
*/
#define FOREACHsetelement_(type, set, variable) \
        if (((variable= NULL), set)) for (\
          variable##p= (type **)&((set)->e[0].p); \
          (variable= *variable##p++);)


/*
   宏 FOREACHsetelement_i_

   定义带索引的 FOREACH 迭代器，用于遍历集合中的元素

   参数说明：
     qh: 全局数据结构指针
     type: 元素类型
     set: 集合指针
     variable: 迭代变量名

   声明：
     type *variable, variable_n, variable_i;

   每次迭代：
     variable 设置为当前集合元素（可能为 NULL）
     variable_i 设置为当前索引，variable_n 设置为 qh_setsize()

   若要重复一个元素：
     variable_i--; variable_n-- 用于重复已删除的元素

   循环结束时：
     variable==NULL 且 variable_i==variable_n

   示例：
     #define FOREACHfacet_i_(qh, facets) FOREACHsetelement_i_(qh, facetT, facets, facet)

   警告：
     嵌套循环不能使用相同的变量（需定义另一个 FOREACH）
     如果嵌套在另一个 FOREACH 内部，则需要使用大括号

     这包括中间的块，例如 FOREACH...{ if () FOREACH...} )
*/
#define FOREACHsetelement_i_(qh, type, set, variable) \
        if (((variable= NULL), set)) for (\
          variable##_i= 0, variable= (type *)((set)->e[0].p), \
                   variable##_n= qh_setsize(qh, set);\
          variable##_i < variable##_n;\
          variable= (type *)((set)->e[++variable##_i].p) )
/*-<a                                    href="qh-set_r.htm#TOC"
  >--------------------------------------</a><a name="FOREACHsetelementreverse_">-</a>

   FOREACHsetelementreverse_(qh, type, set, variable)-
     define FOREACH iterator in reverse order

   declare:
     assumes *variable and **variablep are declared
     also declare 'int variabletemp'

   each iteration:
     variable is set element

   to repeat an element:
     variabletemp++; / *repeat* /

   at exit:
     variable is NULL

   example:
     #define FOREACHvertexreverse_(vertices) FOREACHsetelementreverse_(vertexT, vertices, vertex)

   notes:
     use FOREACHsetelementreverse12_() to reverse first two elements
     WARNING: needs braces if nested inside another FOREACH
*/
#define FOREACHsetelementreverse_(qh, type, set, variable) \
        if (((variable= NULL), set)) for (\
           variable##temp= qh_setsize(qh, set)-1, variable= qh_setlast(qh, set);\
           variable; variable= \
           ((--variable##temp >= 0) ? SETelemt_(set, variable##temp, type) : NULL))

/*-<a                                 href="qh-set_r.htm#TOC"
  >-----------------------------------</a><a name="FOREACHsetelementreverse12_">-</a>

   FOREACHsetelementreverse12_(type, set, variable)-
     define FOREACH iterator with e[1] and e[0] reversed

   declare:
     assumes *variable and **variablep are declared

   each iteration:
     variable is set element
     variablep is one after variable.

   to repeat an element:
     variablep--; / *repeat* /

   at exit:
     variable is NULL at end of loop

   example
     #define FOREACHvertexreverse12_(vertices) FOREACHsetelementreverse12_(vertexT, vertices, vertex)

   notes:
     WARNING: needs braces if nested inside another FOREACH
*/
#define FOREACHsetelementreverse12_(type, set, variable) \
        if (((variable= NULL), set)) for (\
          variable##p= (type **)&((set)->e[1].p); \
          (variable= *variable##p); \
          variable##p == ((type **)&((set)->e[0].p))?variable##p += 2: \
              (variable##p == ((type **)&((set)->e[1].p))?variable##p--:variable##p++))

/*-<a                                 href="qh-set_r.htm#TOC"
  >-----------------------------------</a><a name="FOREACHelem_">-</a>

   FOREACHelem_( set )-
     iterate elements in a set

   declare:
     void *elem, *elemp;

   each iteration:
     elem is set element
     elemp is one beyond

   to repeat an element:
     elemp--; / *repeat* /

   at exit:
     elem == NULL at end of loop

   example:
     FOREACHelem_(set) {

   notes:
     assumes set is not modified
     WARNING: needs braces if nested inside another FOREACH
*/
#define FOREACHelem_(set) FOREACHsetelement_(void, set, elem)
/*-<a                                 href="qh-set_r.htm#TOC"
  >-----------------------------------</a><a name="FOREACHset_">-</a>

   FOREACHset_( set )-
     iterate a set of sets

   declare:
     setT *set, **setp;

   each iteration:
     set is set element
     setp is one beyond

   to repeat an element:
     setp--; / *repeat* /

   at exit:
     set == NULL at end of loop

   example
     FOREACHset_(sets) {

   notes:
     WARNING: needs braces if nested inside another FOREACH
*/
#define FOREACHset_(sets) FOREACHsetelement_(setT, sets, set)

/*-<a                                       href="qh-set_r.htm#TOC"
  >-----------------------------------------</a><a name="SETindex_">-</a>

   SETindex_( set, elem )
     return index of elem in set

   notes:
     for use with FOREACH iteration
     WARN64 -- Maximum set size is 2G

   example:
     i= SETindex_(ridges, ridge)
*/
#define SETindex_(set, elem) ((int)((void **)elem##p - (void **)&(set)->e[1].p))

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETref_">-</a>

   SETref_( elem )
     l.h.s. for modifying the current element in a FOREACH iteration

   example:
     SETref_(ridge)= anotherridge;
*/
#define SETref_(elem) (elem##p[-1])

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETelem_">-</a>

   SETelem_(set, n)
     return the n'th element of set

   notes:
      assumes that n is valid [0..size] and that set is defined
      use SETelemt_() for type cast
*/
#define SETelem_(set, n)           ((set)->e[n].p)

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETelemt_">-</a>

   SETelemt_(set, n, type)
     return the n'th element of set as a type

   notes:
      assumes that n is valid [0..size] and that set is defined
*/
#define SETelemt_(set, n, type)    ((type *)((set)->e[n].p))

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETelemaddr_">-</a>

   SETelemaddr_(set, n, type)
     return address of the n'th element of a set

   notes:
      assumes that n is valid [0..size] and set is defined
*/
#define SETelemaddr_(set, n, type) ((type **)(&((set)->e[n].p)))

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETfirst_">-</a>

   SETfirst_(set)
     return first element of set

*/
#define SETfirst_(set)             ((set)->e[0].p)

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETfirstt_">-</a>

   SETfirstt_(set, type)
     return first element of set as a type

*/
#define SETfirstt_(set, type)      ((type *)((set)->e[0].p))
/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETsecond_">-</a>

   SETsecond_(set)
     return second element of set

*/
#define SETsecond_(set)            ((set)->e[1].p)
/* 宏定义 SETsecond_(set)，返回集合中第二个元素的指针 */

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETsecondt_">-</a>

   SETsecondt_(set, type)
     return second element of set as a type
*/
#define SETsecondt_(set, type)     ((type *)((set)->e[1].p))
/* 宏定义 SETsecondt_(set, type)，将集合中第二个元素视为指定类型返回 */

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETaddr_">-</a>

   SETaddr_(set, type)
       return address of set's elements
*/
#define SETaddr_(set,type)         ((type **)(&((set)->e[0].p)))
/* 宏定义 SETaddr_(set, type)，返回集合元素的地址 */

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETreturnsize_">-</a>

   SETreturnsize_(set, size)
     return size of a set

   notes:
      set must be defined
      use qh_setsize(qhT *qh, set) unless speed is critical
*/
#define SETreturnsize_(set, size) (((size)= ((set)->e[(set)->maxsize].i))?(--(size)):((size)= (set)->maxsize))
/* 宏定义 SETreturnsize_(set, size)，返回集合的大小 */

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETempty_">-</a>

   SETempty_(set)
     return true(1) if set is empty (i.e., FOREACHsetelement_ is empty)

   notes:
      set may be NULL
      qh_setsize may be non-zero if first element is NULL
*/
#define SETempty_(set)            (!set || (SETfirst_(set) ? 0 : 1))
/* 宏定义 SETempty_(set)，如果集合为空则返回true(1)，否则返回false(0) */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="SETsizeaddr_">-</a>

  SETsizeaddr_(set)
    return pointer to 'actual size+1' of set (set CANNOT be NULL!!)
    Its type is setelemT* for strict aliasing
    All SETelemaddr_ must be cast to setelemT


  notes:
    *SETsizeaddr==NULL or e[*SETsizeaddr-1].p==NULL
*/
#define SETsizeaddr_(set) (&((set)->e[(set)->maxsize]))
/* 宏定义 SETsizeaddr_(set)，返回指向集合 '实际大小+1' 的指针 */

/*-<a                                     href="qh-set_r.htm#TOC"
  >---------------------------------------</a><a name="SETtruncate_">-</a>

   SETtruncate_(set, size)
     truncate set to size

   see:
     qh_settruncate()

*/
#define SETtruncate_(set, size) {set->e[set->maxsize].i= size+1; /* maybe overwritten */ \
      set->e[size].p= NULL;}
/* 宏定义 SETtruncate_(set, size)，将集合截断至指定大小 */
/* 删除集合中的最后一个元素并返回其指针 */
void *qh_setdellast(setT *set);

/* 删除集合中第 nth 个元素并返回其指针 */
void *qh_setdelnth(qhT *qh, setT *set, int nth);

/* 删除排序后集合中第 nth 个元素并返回其指针 */
void *qh_setdelnthsorted(qhT *qh, setT *set, int nth);

/* 从集合中删除指定元素并插入新元素，返回被删除的元素的指针 */
void *qh_setdelsorted(setT *set, void *newelem);

/* 复制集合并返回新的集合指针，elemsize 表示每个元素的大小 */
setT *qh_setduplicate(qhT *qh, setT *set, int elemsize);

/* 返回指向集合尾部元素指针的指针 */
void **qh_setendpointer(setT *set);

/* 判断两个集合是否相等，返回 1 表示相等，0 表示不相等 */
int qh_setequal(setT *setA, setT *setB);

/* 判断两个集合在忽略指定元素后是否相等，返回 1 表示相等，0 表示不相等 */
int qh_setequal_except(setT *setA, void *skipelemA, setT *setB, void *skipelemB);

/* 判断两个集合在忽略指定位置元素后是否相等，返回 1 表示相等，0 表示不相等 */
int qh_setequal_skip(setT *setA, int skipA, setT *setB, int skipB);

/* 释放集合及其元素占用的内存 */
void qh_setfree(qhT *qh, setT **set);

/* 释放集合及其元素占用的内存，elemsize 表示每个元素的大小 */
void qh_setfree2(qhT *qh, setT **setp, int elemsize);

/* 释放长期使用的集合及其元素占用的内存 */
void qh_setfreelong(qhT *qh, setT **set);

/* 判断元素是否在集合中，返回 1 表示存在，0 表示不存在 */
int qh_setin(setT *set, void *setelem);

/* 返回集合中元素的索引位置，从 0 开始 */
int qh_setindex(setT *set, void *setelem);

/* 扩展集合的大小以容纳更多元素 */
void qh_setlarger(qhT *qh, setT **setp);

/* 快速计算扩展集合的新大小 */
int qh_setlarger_quick(qhT *qh, int setsize, int *newsize);

/* 返回集合中最后一个元素的指针 */
void *qh_setlast(setT *set);

/* 创建新的集合并返回其指针，size 表示集合的初始大小 */
setT *qh_setnew(qhT *qh, int size);

/* 创建新的集合，删除排序后第 nth 个元素并返回其指针，prepend 表示是否从开头删除 */
setT *qh_setnew_delnthsorted(qhT *qh, setT *set, int size, int nth, int prepend);

/* 打印集合的内容到指定文件流中 */
void qh_setprint(qhT *qh, FILE *fp, const char* string, setT *set);

/* 替换集合中的旧元素为新元素 */
void qh_setreplace(qhT *qh, setT *set, void *oldelem, void *newelem);

/* 返回集合中元素的数量 */
int qh_setsize(qhT *qh, setT *set);

/* 创建临时集合并返回其指针，setsize 表示集合的初始大小 */
setT *qh_settemp(qhT *qh, int setsize);

/* 释放临时集合及其元素占用的内存 */
void qh_settempfree(qhT *qh, setT **set);

/* 释放所有临时集合及其元素占用的内存 */
void qh_settempfree_all(qhT *qh);

/* 弹出并返回栈顶的临时集合 */
setT *qh_settemppop(qhT *qh);

/* 将集合推入临时集合栈 */
void qh_settemppush(qhT *qh, setT *set);

/* 截断集合的大小至指定大小 */
void qh_settruncate(qhT *qh, setT *set, int size);

/* 使集合中的元素唯一化，返回 1 表示有修改，0 表示无修改 */
int qh_setunique(qhT *qh, setT **set, void *elem);

/* 将集合中指定索引位置的元素置零 */
void qh_setzero(qhT *qh, setT *set, int idx, int size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFset */
```