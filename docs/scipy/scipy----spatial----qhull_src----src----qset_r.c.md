# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\qset_r.c`

```
/*
   qset_r.c
   implements set manipulations needed for quickhull

   see qh-set_r.htm and qset_r.h

   Be careful of strict aliasing (two pointers of different types
   that reference the same location).  The last slot of a set is
   either the actual size of the set plus 1, or the NULL terminator
   of the set (i.e., setelemT).

   Only reference qh for qhmem or qhstat.  Otherwise the matching code in qset.c will bring in qhT

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/qset_r.c#7 $$Change: 2711 $
   $DateTime: 2019/06/27 22:34:56 $$Author: bbarber $
*/

#include "libqhull_r.h" /* for qhT and QHULL_CRTDBG */
#include "qset_r.h"
#include "mem_r.h"
#include <stdio.h>
#include <string.h>
/*** uncomment here and qhull_ra.h
     if string.h does not define memcpy()
#include <memory.h>
*/

#ifndef qhDEFlibqhull
typedef struct ridgeT ridgeT;
typedef struct facetT facetT;
void    qh_errexit(qhT *qh, int exitcode, facetT *, ridgeT *);
void    qh_fprintf(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... );
#  ifdef _MSC_VER  /* Microsoft Visual C++ -- warning level 4 */
#  pragma warning( disable : 4127)  /* conditional expression is constant */
#  pragma warning( disable : 4706)  /* assignment within conditional function */
#  endif
#endif

/*=============== internal macros ===========================*/

/*============ functions in alphabetical order ===================*/

/*-<a                             href="qh-set_r.htm#TOC"
  >--------------------------------<a name="setaddnth">-</a>

  qh_setaddnth(qh, setp, nth, newelem )
    adds newelem as n'th element of sorted or unsorted *setp

  notes:
    *setp and newelem must be defined
    *setp may be a temp set
    nth=0 is first element
    errors if nth is out of bounds

  design:
    expand *setp if empty or full
    move tail of *setp up one
    insert newelem
*/
void qh_setaddnth(qhT *qh, setT **setp, int nth, void *newelem) {
  int oldsize, i;
  setelemT *sizep;          /* avoid strict aliasing */
  setelemT *oldp, *newp;

  if (!*setp || (sizep= SETsizeaddr_(*setp))->i==0) {
    qh_setlarger(qh, setp);  // 扩展集合 *setp 的大小
    sizep= SETsizeaddr_(*setp);
  }
  oldsize= sizep->i - 1;
  if (nth < 0 || nth > oldsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6171, "qhull internal error (qh_setaddnth): nth %d is out-of-bounds for set:\n", nth);
    qh_setprint(qh, qh->qhmem.ferr, "", *setp);  // 打印出错信息和集合内容到 qhmem.ferr
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);  // 退出程序
  }
  sizep->i++;
  oldp= (setelemT *)SETelemaddr_(*setp, oldsize, void);   /* NULL */
  newp= oldp+1;
  for (i=oldsize-nth+1; i--; )  /* move at least NULL  */
    (newp--)->p= (oldp--)->p;       /* may overwrite *sizep */
  newp->p= newelem;  // 插入新元素到第 nth 位置
} /* setaddnth */


/*-<a                              href="qh-set_r.htm#TOC"
  >--------------------------------<a name="setaddsorted">-</a>

  setaddsorted( setp, newelem )
*/
    # 将 newelem 添加到已排序集合 *setp 中
    
    # 注意事项：
    #   *setp 和 newelem 必须已定义
    #   *setp 可能是临时集合
    #   如果 newelem 已经存在于集合中，则不执行任何操作
    
    # 设计：
    #   找到 newelem 在 *setp 中的插入位置
    #   将 newelem 插入到合适的位置
    adds an newelem into sorted *setp
/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setappend">-</a>

  qh_setappend(qh, setp, newelem )
    append newelem to *setp

  notes:
    *setp may be a temp set
    *setp and newelem may be NULL

  design:
    expand *setp if empty or full
    append newelem to *setp
*/
void qh_setappend(qhT *qh, setT **setp, void *newelem) {
  setelemT *sizep;  /* Avoid strict aliasing.  Writing to *endp may overwrite *sizep */
  setelemT *endp;
  int count;

  if (!newelem)
    return;  // 如果 newelem 为空指针，则直接返回

  // 如果 *setp 为空或者已满，扩展 *setp
  if (!*setp || (sizep= SETsizeaddr_(*setp))->i==0) {
    qh_setlarger(qh, setp);  // 调用 qh_setlarger 扩展集合
    sizep= SETsizeaddr_(*setp);
  }
  
  count= (sizep->i)++ - 1;  // 计算当前集合元素数量，并更新计数器
  endp= (setelemT *)SETelemaddr_(*setp, count, void);  // 获取最后一个元素的地址
  (endp++)->p= newelem;  // 将 newelem 添加到集合末尾
  endp->p= NULL;  // 最后一个元素的下一个置为空指针，表示集合末尾
} /* setappend */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setappend_set">-</a>

  qh_setappend_set(qh, setp, setA )
    appends setA to *setp

  notes:
    *setp can not be a temp set
    *setp and setA may be NULL

  design:
    setup for copy
    expand *setp if it is too small
    append all elements of setA to *setp
*/
void qh_setappend_set(qhT *qh, setT **setp, setT *setA) {
  int sizeA, size;
  setT *oldset;
  setelemT *sizep;

  if (!setA)
    return;  // 如果 setA 为空，则直接返回

  SETreturnsize_(setA, sizeA);  // 获取 setA 的大小
  if (!*setp)
    *setp= qh_setnew(qh, sizeA);  // 如果 *setp 为空，则创建一个新集合
  sizep= SETsizeaddr_(*setp);
  if (!(size= sizep->i))
    size= (*setp)->maxsize;
  else
    size--;

  // 如果当前集合空间不足以容纳 setA，进行扩展
  if (size + sizeA > (*setp)->maxsize) {
    oldset= *setp;
    *setp= qh_setcopy(qh, oldset, sizeA);  // 复制并扩展集合
    qh_setfree(qh, &oldset);  // 释放旧集合的内存
    sizep= SETsizeaddr_(*setp);
  }

  if (sizeA > 0) {
    sizep->i= size+sizeA+1;   /* memcpy may overwrite */
    // 将 setA 中的所有元素追加到 *setp 中
    memcpy((char *)&((*setp)->e[size].p), (char *)&(setA->e[0].p), (size_t)(sizeA+1) * SETelemsize);
  }
} /* setappend_set */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setappend2ndlast">-</a>

  qh_setappend2ndlast(qh, setp, newelem )
    makes newelem the next to the last element in *setp

  notes:
    *setp must have at least one element
    newelem must be defined
    *setp may be a temp set

  design:
    expand *setp if empty or full
    move last element of *setp up one
    insert newelem
*/
void qh_setappend2ndlast(qhT *qh, setT **setp, void *newelem) {
    setelemT *sizep;  /* Avoid strict aliasing.  Writing to *endp may overwrite *sizep */
    setelemT *endp, *lastp;
    int count;

    // 如果 *setp 为空或者已满，扩展 *setp
    if (!*setp || (sizep= SETsizeaddr_(*setp))->i==0) {
        qh_setlarger(qh, setp);  // 调用 qh_setlarger 扩展集合
        sizep= SETsizeaddr_(*setp);
    }
    /* (endp) is the last element
       (lastp) is the 2nd last element
    */
}
    count= (sizep->i)++ - 1;
    endp= (setelemT *)SETelemaddr_(*setp, count, void); /* 获取集合中第 count 个元素的地址 */
    lastp= endp-1;    /* 将 endp 的前一个元素的地址赋给 lastp */
    *(endp++)= *lastp;    /* 将 lastp 指向的元素复制给 endp 指向的位置，然后增加 endp 的指针 */
    endp->p= NULL;    /* 将 endp 所指向的元素的指针字段置为 NULL，可能覆盖 *sizep */
    lastp->p= newelem;    /* 将 lastp 所指向的元素的指针字段设置为 newelem */
/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setappend2ndlast">-</a>

  setappend2ndlast(qh, set, newelem )
    append newelem as the second last element in an unsorted set

  returns:
    none

  notes:
    set may be NULL
    newelem must not be NULL
    set may be unsorted

  design:
    if set is NULL, return immediately
    append newelem before the last element in set
*/
void setappend2ndlast(qhT *qh, setT *set, void *newelem ) {
  int size;
  void **elemp;

  if (!set)
    return;
  SETreturnsize_(set, size);
  elemp= SETelemt_(set, size-1);
  *elemp= newelem;
} /* setappend2ndlast */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setcheck">-</a>

  qh_setcheck(qh, set, typename, id )
    check set for validity
    report errors with typename and id

  design:
    checks that maxsize, actual size, and NULL terminator agree
*/
void qh_setcheck(qhT *qh, setT *set, const char *tname, unsigned int id) {
  int maxsize, size;
  int waserr= 0;

  if (!set)
    return;
  SETreturnsize_(set, size);
  maxsize= set->maxsize;
  if (size > maxsize || !maxsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6172, "qhull internal error (qh_setcheck): actual size %d of %s%d is greater than max size %d\n",
             size, tname, id, maxsize);
    waserr= 1;
  }else if (set->e[size].p) {
    qh_fprintf(qh, qh->qhmem.ferr, 6173, "qhull internal error (qh_setcheck): %s%d(size %d max %d) is not null terminated.\n",
             tname, id, size-1, maxsize);
    waserr= 1;
  }
  if (waserr) {
    qh_setprint(qh, qh->qhmem.ferr, "ERRONEOUS", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
} /* setcheck */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setcompact">-</a>

  qh_setcompact(qh, set )
    remove internal NULLs from an unsorted set

  returns:
    updated set

  notes:
    set may be NULL
    it would be faster to swap tail of set into holes, like qh_setdel

  design:
    setup pointers into set
    skip NULLs while copying elements to start of set
    update the actual size
*/
void qh_setcompact(qhT *qh, setT *set) {
  int size;
  void **destp, **elemp, **endp, **firstp;

  if (!set)
    return;
  SETreturnsize_(set, size);
  destp= elemp= firstp= SETaddr_(set, void);
  endp= destp + size;
  while (1) {
    if (!(*destp++= *elemp++)) {
      destp--;
      if (elemp > endp)
        break;
    }
  }
  qh_settruncate(qh, set, (int)(destp-firstp));   /* WARN64 */
} /* setcompact */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setcopy">-</a>

  qh_setcopy(qh, set, extra )
    make a copy of a sorted or unsorted set with extra slots

  returns:
    new set

  design:
    create a newset with extra slots
    copy the elements to the newset
*/
setT *qh_setcopy(qhT *qh, setT *set, int extra) {
  setT *newset;
  int size;

  if (extra < 0)
    extra= 0;
  SETreturnsize_(set, size);
  newset= qh_setnew(qh, size+extra);
  SETsizeaddr_(newset)->i= size+1;    /* memcpy may overwrite */
  memcpy((char *)&(newset->e[0].p), (char *)&(set->e[0].p), (size_t)(size+1) * SETelemsize);
  return(newset);
} /* setcopy */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdel">-</a>

  qh_setdel(set, oldelem )
    delete oldelem from an unsorted set

  returns:
    returns oldelem if found
    returns NULL otherwise

  notes:
    set may be NULL
    oldelem must not be NULL;
*/
    # 创建一个函数用于从集合中删除指定元素 oldelem
    def delete_one_copy(setobj, oldelem):
        # 查找集合中第一个匹配 oldelem 的元素的位置
        loc = setobj.index(oldelem)
        # 如果集合已满（实际大小等于最大容量），更新实际大小
        if len(setobj) == setobj.maxsize:
            setobj.actualsize -= 1
        # 将集合中最后一个元素移动到 oldelem 所在位置
        setobj[loc] = setobj[-1]
/*
 * 删除集合中指定元素，返回删除的元素指针，若集合为空或元素不存在则返回NULL
 */
void *qh_setdel(setT *set, void *oldelem) {
  setelemT *sizep;
  setelemT *elemp;
  setelemT *lastp;

  if (!set)
    return NULL;
  // 获取集合元素的地址，类型转换为setelemT*
  elemp= (setelemT *)SETaddr_(set, void);
  // 查找要删除的元素，直到找到或者遍历完所有元素
  while (elemp->p != oldelem && elemp->p)
    elemp++;
  // 如果找到要删除的元素
  if (elemp->p) {
    // 获取集合大小的指针
    sizep= SETsizeaddr_(set);
    // 如果集合原先是满的
    if (!(sizep->i)--)         /*  if was a full set */
      sizep->i= set->maxsize;  /*     *sizep= (maxsize-1)+ 1 */
    // 获取集合最后一个元素的地址
    lastp= (setelemT *)SETelemaddr_(set, sizep->i-1, void);
    // 将要删除的元素替换为最后一个元素的内容
    elemp->p= lastp->p;      /* may overwrite itself */
    lastp->p= NULL;
    return oldelem;
  }
  // 没有找到要删除的元素，返回NULL
  return NULL;
} /* setdel */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdellast">-</a>

  qh_setdellast( set )
    return last element of set or NULL

  notes:
    deletes element from set
    set may be NULL

  design:
    return NULL if empty
    if full set
      delete last element and set actual size
    else
      delete last element and update actual size
*/
/*
 * 删除集合中的最后一个元素，并返回该元素的指针，若集合为空则返回NULL
 */
void *qh_setdellast(setT *set) {
  int setsize;  /* actually, actual_size + 1 */
  int maxsize;
  setelemT *sizep;
  void *returnvalue;

  if (!set || !(set->e[0].p))
    return NULL;
  // 获取集合大小的指针
  sizep= SETsizeaddr_(set);
  // 如果集合非空
  if ((setsize= sizep->i)) {
    // 返回集合倒数第二个元素的指针
    returnvalue= set->e[setsize - 2].p;
    // 清空集合倒数第二个位置的元素
    set->e[setsize - 2].p= NULL;
    sizep->i--;
  }else {
    // 如果集合为空，返回集合最后一个元素的指针
    maxsize= set->maxsize;
    returnvalue= set->e[maxsize - 1].p;
    // 清空集合最后一个位置的元素
    set->e[maxsize - 1].p= NULL;
    sizep->i= maxsize;
  }
  return returnvalue;
} /* setdellast */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdelnth">-</a>

  qh_setdelnth(qh, set, nth )
    deletes nth element from unsorted set
    0 is first element

  returns:
    returns the element (needs type conversion)

  notes:
    errors if nth invalid

  design:
    setup points and check nth
    delete nth element and overwrite with last element
*/
/*
 * 从无序集合中删除第n个元素（从0开始计数），并返回删除的元素指针，若n无效则报错
 */
void *qh_setdelnth(qhT *qh, setT *set, int nth) {
  void *elem;
  setelemT *sizep;
  setelemT *elemp, *lastp;

  sizep= SETsizeaddr_(set);
  // 如果集合为空
  if ((sizep->i--)==0)         /*  if was a full set */
    sizep->i= set->maxsize;    /*    *sizep= (maxsize-1)+ 1 */
  // 如果n无效，报错
  if (nth < 0 || nth >= sizep->i) {
    qh_fprintf(qh, qh->qhmem.ferr, 6174, "qhull internal error (qh_setdelnth): nth %d is out-of-bounds for set:\n", nth);
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  // 获取第n个元素的指针
  elemp= (setelemT *)SETelemaddr_(set, nth, void); /* nth valid by QH6174 */
  // 获取集合最后一个元素的指针
  lastp= (setelemT *)SETelemaddr_(set, sizep->i-1, void);
  // 获取第n个元素的内容，并将其替换为最后一个元素的内容
  elem= elemp->p;
  elemp->p= lastp->p;      /* may overwrite itself */
  lastp->p= NULL;
  return elem;
} /* setdelnth */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdelnthsorted">-</a>

  qh_setdelnthsorted(qh, set, nth )
    deletes nth element from sorted set

  returns:
    returns the element (use type conversion)

  notes:
    errors if nth invalid

  see also:
*/
    # 定义函数 setnew_delnthsorted，用于更新并删除排序后的数据集中的第 n 个元素
    
    design:
      # 设置点并检查第 n 个元素
      setup points and check nth
      # 复制剩余元素向下移动一个位置
      copy remaining elements down one
      # 更新实际大小
      update actual size
/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdelnthsorted">-</a>

  qh_setdelnthsorted( qh, set, nth )
    Deletes nth element from a sorted set

  returns:
    Returns the deleted element

  notes:
    If nth is out of bounds, it prints an error message and terminates
    Set must be sorted in ascending order of elements

  design:
    Check if nth is within valid range
    Retrieve the nth element
    Shift subsequent elements to fill the gap left by deletion
    Adjust the actual size of the set
*/
void *qh_setdelnthsorted(qhT *qh, setT *set, int nth) {
  void *elem;
  setelemT *sizep;
  setelemT *newp, *oldp;

  sizep= SETsizeaddr_(set);
  // Check if nth is out of bounds or exceeds the maximum size
  if (nth < 0 || (sizep->i && nth >= sizep->i-1) || nth >= set->maxsize) {
    // Print error message and set details if nth is out of bounds
    qh_fprintf(qh, qh->qhmem.ferr, 6175, "qhull internal error (qh_setdelnthsorted): nth %d is out-of-bounds for set:\n", nth);
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  // Retrieve the nth element
  newp= (setelemT *)SETelemaddr_(set, nth, void);
  elem= newp->p;
  oldp= newp+1;
  // Shift subsequent elements to fill the gap
  while (((newp++)->p= (oldp++)->p))
    ; /* copy remaining elements and NULL */
  // Adjust size if it was a full set
  if ((sizep->i--)==0)
    sizep->i= set->maxsize;
  return elem;
} /* setdelnthsorted */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdelsorted">-</a>

  qh_setdelsorted( set, oldelem )
    Deletes oldelem from sorted set

  returns:
    Returns oldelem if it was deleted; otherwise, NULL

  notes:
    Set may be NULL

  design:
    Locate oldelem in set
    Copy remaining elements down one to fill the gap
    Update actual size of the set
*/
void *qh_setdelsorted(setT *set, void *oldelem) {
  setelemT *sizep;
  setelemT *newp, *oldp;

  if (!set)
    return NULL;
  // Locate oldelem in set
  newp= (setelemT *)SETaddr_(set, void);
  while(newp->p != oldelem && newp->p)
    newp++;
  if (newp->p) {
    // Copy remaining elements
    oldp= newp+1;
    while (((newp++)->p= (oldp++)->p))
      ; /* copy remaining elements */
    sizep= SETsizeaddr_(set);
    // Adjust size if it was a full set
    if ((sizep->i--)==0)
      sizep->i= set->maxsize;
    return oldelem;
  }
  return NULL;
} /* setdelsorted */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setduplicate">-</a>

  qh_setduplicate(qh, set, elemsize )
    Duplicates a set of elemsize elements

  returns:
    Returns a new duplicated set

  notes:
    Uses qh_setcopy if retaining old elements

  design:
    Create a new set
    For each element of the old set, create a copy and append to the new set
*/
setT *qh_setduplicate(qhT *qh, setT *set, int elemsize) {
  void          *elem, **elemp, *newElem;
  setT          *newSet;
  int           size;

  // If set is empty, return NULL
  if (!(size= qh_setsize(qh, set)))
    return NULL;
  // Create a new set with the same size
  newSet= qh_setnew(qh, size);
  // Iterate through each element in the old set
  FOREACHelem_(set) {
    // Allocate memory for a new element and copy from the old element
    newElem= qh_memalloc(qh, elemsize);
    memcpy(newElem, elem, (size_t)elemsize);
    // Append the new element to the new set
    qh_setappend(qh, &newSet, newElem);
  }
  return newSet;
} /* setduplicate */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setendpointer">-</a>

  qh_setendpointer( set )
    Returns pointer to NULL terminator of a set's elements
    set can not be NULL

  returns:
    Returns a pointer to the NULL terminator of the set's elements

  notes:
    Assumes set is not NULL

*/
void **qh_setendpointer(setT *set) {

  setelemT *sizep= SETsizeaddr_(set);
  int n= sizep->i;
  // Return pointer to the last element's NULL terminator
  return (n ? &set->e[n-1].p : &sizep->p);
} /* qh_setendpointer */
/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setequal">-</a>

  qh_setequal( setA, setB )
    returns 1 if two sorted sets are equal, otherwise returns 0

  notes:
    either set may be NULL

  design:
    check size of each set
    setup pointers
    compare elements of each set
*/
int qh_setequal(setT *setA, setT *setB) {
  void **elemAp, **elemBp;
  int sizeA= 0, sizeB= 0;

  // 如果 setA 不为空，获取其大小
  if (setA) {
    SETreturnsize_(setA, sizeA);
  }
  // 如果 setB 不为空，获取其大小
  if (setB) {
    SETreturnsize_(setB, sizeB);
  }
  // 如果两个集合大小不同，返回 0
  if (sizeA != sizeB)
    return 0;
  // 如果两个集合大小为 0，返回 1
  if (!sizeA)
    return 1;
  // 获取集合 A 和集合 B 的元素指针
  elemAp= SETaddr_(setA, void);
  elemBp= SETaddr_(setB, void);
  // 比较集合 A 和集合 B 的元素是否相同
  if (!memcmp((char *)elemAp, (char *)elemBp, (size_t)(sizeA * SETelemsize)))
    return 1;
  return 0;
} /* setequal */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setequal_except">-</a>

  qh_setequal_except( setA, skipelemA, setB, skipelemB )
    returns 1 if sorted setA and setB are equal except for skipelemA & B

  returns:
    false if either skipelemA or skipelemB are missing

  notes:
    neither set may be NULL

    if skipelemB is NULL,
      can skip any one element of setB

  design:
    setup pointers
    search for skipelemA, skipelemB, and mismatches
    check results
*/
int qh_setequal_except(setT *setA, void *skipelemA, setT *setB, void *skipelemB) {
  void **elemA, **elemB;
  int skip=0;

  // 获取集合 A 和集合 B 的元素指针
  elemA= SETaddr_(setA, void);
  elemB= SETaddr_(setB, void);
  while (1) {
    // 如果 elemA 指向 skipelemA，跳过该元素
    if (*elemA == skipelemA) {
      skip++;
      elemA++;
    }
    // 如果 skipelemB 不为空，且 elemB 指向 skipelemB，跳过该元素
    if (skipelemB) {
      if (*elemB == skipelemB) {
        skip++;
        elemB++;
      }
    // 如果 skipelemB 为空，跳过集合 B 的一个任意元素
    } else if (*elemA != *elemB) {
      skip++;
      // 如果 skipelemB 还未赋值，赋值为当前 elemB 指向的元素并跳过该元素
      if (!(skipelemB= *elemB++))
        return 0;
    }
    // 如果 elemA 指向空（集合末尾），退出循环
    if (!*elemA)
      break;
    // 比较集合 A 和集合 B 的当前元素是否相同
    if (*elemA++ != *elemB++)
      return 0;
  }
  // 如果跳过的元素不是两个，或者集合 B 还有剩余元素，返回 0
  if (skip != 2 || *elemB)
    return 0;
  return 1;
} /* setequal_except */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setequal_skip">-</a>

  qh_setequal_skip( setA, skipA, setB, skipB )
    returns 1 if sorted setA and setB are equal except for elements skipA & B

  returns:
    false if different size

  notes:
    neither set may be NULL

  design:
    setup pointers
    search for mismatches while skipping skipA and skipB
*/
int qh_setequal_skip(setT *setA, int skipA, setT *setB, int skipB) {
  void **elemA, **elemB, **skipAp, **skipBp;

  // 获取集合 A 和集合 B 的元素指针，以及要跳过的元素指针
  elemA= SETaddr_(setA, void);
  elemB= SETaddr_(setB, void);
  skipAp= SETelemaddr_(setA, skipA, void);
  skipBp= SETelemaddr_(setB, skipB, void);
  while (1) {
    // 如果 elemA 指向 skipAp，跳过该元素
    if (elemA == skipAp)
      elemA++;
    // 如果 elemB 指向 skipBp，跳过该元素
    if (elemB == skipBp)
      elemB++;
    // 如果 elemA 指向空（集合末尾），退出循环
    if (!*elemA)
      break;
    // 比较集合 A 和集合 B 的当前元素是否相同
    if (*elemA++ != *elemB++)
      return 0;
  }
  // 如果集合 B 还有剩余元素，返回 0
  if (*elemB)
    return 0;
  return 1;
} /* setequal_skip */
/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setfree">-</a>

  qh_setfree(qh, setp )
    frees the space occupied by a sorted or unsorted set

  returns:
    sets setp to NULL

  notes:
    set may be NULL

  design:
    free array
    free set
*/
void qh_setfree(qhT *qh, setT **setp) {
  int size;
  void **freelistp;  /* used if !qh_NOmem by qh_memfree_() */

  // 检查是否需要释放内存，*setp 不为 NULL 时执行
  if (*setp) {
    // 计算要释放的内存大小
    size = (int)sizeof(setT) + ((*setp)->maxsize) * SETelemsize;
    // 根据内存大小选择合适的释放方法
    if (size <= qh->qhmem.LASTsize) {
      qh_memfree_(qh, *setp, size, freelistp);  // 使用带有自定义内存管理的释放方法
    } else {
      qh_memfree(qh, *setp, size);  // 直接释放内存
    }
    *setp = NULL;  // 将指针设置为 NULL
  }
} /* setfree */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setfree2">-</a>

  qh_setfree2(qh, setp, elemsize )
    frees the space occupied by a set and its elements

  notes:
    set may be NULL

  design:
    free each element
    free set
*/
void qh_setfree2(qhT *qh, setT **setp, int elemsize) {
  void *elem, **elemp;

  // 遍历每个元素并释放内存
  FOREACHelem_(*setp)
    qh_memfree(qh, elem, elemsize);  // 使用自定义内存释放方法释放元素内存
  qh_setfree(qh, setp);  // 调用普通的集合释放函数
} /* setfree2 */



/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setfreelong">-</a>

  qh_setfreelong(qh, setp )
    frees a set only if it's in long memory

  returns:
    sets setp to NULL if it is freed

  notes:
    set may be NULL

  design:
    if set is large
      free it
*/
void qh_setfreelong(qhT *qh, setT **setp) {
  int size;

  // 检查集合是否存在并且大小是否大于上限，如果是则释放内存
  if (*setp) {
    size = (int)sizeof(setT) + ((*setp)->maxsize) * SETelemsize;
    if (size > qh->qhmem.LASTsize) {
      qh_memfree(qh, *setp, size);  // 使用自定义内存释放函数释放内存
      *setp = NULL;  // 将指针设置为 NULL
    }
  }
} /* setfreelong */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setin">-</a>

  qh_setin( set, setelem )
    returns 1 if setelem is in a set, 0 otherwise

  notes:
    set may be NULL or unsorted

  design:
    scans set for setelem
*/
int qh_setin(setT *set, void *setelem) {
  void *elem, **elemp;

  // 遍历集合查找元素是否存在，如果存在返回1，否则返回0
  FOREACHelem_(set) {
    if (elem == setelem)
      return 1;  // 找到元素，返回1
  }
  return 0;  // 遍历完集合未找到元素，返回0
} /* setin */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setindex">-</a>

  qh_setindex(set, atelem )
    returns the index of atelem in set.
    returns -1, if not in set or maxsize wrong

  notes:
    set may be NULL and may contain nulls.
    NOerrors returned (qh_pointid, QhullPoint::id)

  design:
    checks maxsize
    scans set for atelem
*/
int qh_setindex(setT *set, void *atelem) {
  void **elem;
  int size, i;

  // 如果集合为空，返回-1
  if (!set)
    return -1;
  SETreturnsize_(set, size);  // 获取集合的大小
  if (size > set->maxsize)  // 检查集合大小是否超过最大限制
    return -1;  // 返回-1表示未找到或者最大大小错误
  elem = SETaddr_(set, void);  // 获取集合起始地址
  for (i = 0; i < size; i++) {
    if (*elem++ == atelem)  // 遍历集合查找元素
      return i;  // 找到元素返回索引
  }
  return -1;  // 遍历完集合未找到元素，返回-1
} /* setindex */
/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setlarger">-</a>

  qh_setlarger(qh, oldsetp )
    returns a larger set that contains all elements of *oldsetp

  notes:
    if long memory,
      the new set is 2x larger
    if qhmem.LASTsize is between 1.5x and 2x
      the new set is qhmem.LASTsize
    otherwise use quick memory,
      the new set is 2x larger, rounded up to next qh_memsize
       
    if temp set, updates qh->qhmem.tempstack

  design:
    creates a new set
    copies the old set to the new set
    updates pointers in tempstack
    deletes the old set
*/
void qh_setlarger(qhT *qh, setT **oldsetp) {
  // 初始化变量和指针
  int setsize= 1, newsize;
  setT *newset, *set, **setp, *oldset;
  setelemT *sizep;
  setelemT *newp, *oldp;

  if (*oldsetp) {
    // 获取旧集合的指针和大小
    oldset= *oldsetp;
    SETreturnsize_(oldset, setsize);  // 获取旧集合的大小
    qh->qhmem.cntlarger++;  // 记录扩展集合次数
    qh->qhmem.totlarger += setsize+1;  // 记录扩展集合的总大小
    // 根据当前内存状态确定新集合的大小
    qh_setlarger_quick(qh, setsize, &newsize);
    // 创建新集合
    newset= qh_setnew(qh, newsize);
    // 复制旧集合的数据到新集合
    oldp= (setelemT *)SETaddr_(oldset, void);
    newp= (setelemT *)SETaddr_(newset, void);
    memcpy((char *)newp, (char *)oldp, (size_t)(setsize+1) * SETelemsize);
    // 更新临时栈中指向旧集合的指针
    sizep= SETsizeaddr_(newset);
    sizep->i= setsize+1;
    FOREACHset_((setT *)qh->qhmem.tempstack) {
      if (set == oldset)
        *(setp-1)= newset;
    }
    // 释放旧集合的内存
    qh_setfree(qh, oldsetp);
  }else
    // 如果旧集合为空，则创建一个初始大小为3的新集合
    newset= qh_setnew(qh, 3);
  // 将新集合的指针返回给调用者
  *oldsetp= newset;
} /* setlarger */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setlarger_quick">-</a>

  qh_setlarger_quick(qh, setsize, newsize )
    determine newsize for setsize
    returns True if newsize fits in quick memory

  design:
    if 2x fits into quick memory
      return True, 2x
    if x+4 does not fit into quick memory
      return False, 2x
    if x+x/3 fits into quick memory
      return True, the last quick set
    otherwise
      return False, 2x
*/
int qh_setlarger_quick(qhT *qh, int setsize, int *newsize) {
    // 计算新集合的大小为旧集合大小的两倍
    *newsize= 2 * setsize;
    // 计算快速内存中能容纳的最大集合大小
    int lastquickset;
    lastquickset= (qh->qhmem.LASTsize - (int)sizeof(setT)) / SETelemsize; /* matches size computation in qh_setnew */
    // 根据快速内存的情况确定是否使用快速内存
    if (*newsize <= lastquickset)
      return 1;
    if (setsize + 4 > lastquickset)
      return 0;
    if (setsize + setsize/3 <= lastquickset) {
      *newsize= lastquickset;
      return 1;
    }
    return 0;
} /* setlarger_quick */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setlast">-</a>

  qh_setlast( set )
    return last element of set or NULL (use type conversion)

  notes:
    set may be NULL

  design:
    return last element
*/
void *qh_setlast(setT *set) {
  // 初始化变量
  int size;

  if (set) {
    // 如果集合不为空，获取集合的大小
    size= SETsizeaddr_(set)->i;
    // 根据集合大小返回最后一个元素或NULL
    if (!size)
      return SETelem_(set, set->maxsize - 1);
    else if (size > 1)
      return SETelem_(set, size - 2);
  }
  // 如果集合为空，返回NULL
  return NULL;
} /* setlast */
/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setnew">-</a>

  qh_setnew(qh, setsize )
    creates and allocates space for a set

  notes:
    setsize means the number of elements (!including the NULL terminator)
    use qh_settemp/qh_setfreetemp if set is temporary

  design:
    allocate memory for set
    roundup memory if small set
    initialize as empty set
*/
// 创建并分配空间用于一个集合，参数包括集合的大小
setT *qh_setnew(qhT *qh, int setsize) {
  setT *set;
  int sizereceived; /* used if !qh_NOmem */
  int size;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */

  if (!setsize)
    setsize++;
  size= (int)sizeof(setT) + setsize * SETelemsize; /* setT includes NULL terminator, see qh.LASTquickset */
  // 如果集合大小在有效范围内，使用快速内存分配函数
  if (size>0 && size <= qh->qhmem.LASTsize) {
    qh_memalloc_(qh, size, freelistp, set, setT);
#ifndef qh_NOmem
    sizereceived= qh->qhmem.sizetable[ qh->qhmem.indextable[size]];
    if (sizereceived > size)
      setsize += (sizereceived - size)/SETelemsize;
#endif
  }else
    // 否则使用标准内存分配函数
    set= (setT *)qh_memalloc(qh, size);
  // 初始化集合的最大大小和第一个元素为NULL
  set->maxsize= setsize;
  set->e[setsize].i= 1;
  set->e[0].p= NULL;
  return(set);
} /* setnew */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setnew_delnthsorted">-</a>

  qh_setnew_delnthsorted(qh, set, size, nth, prepend )
    creates a sorted set not containing nth element
    if prepend, the first prepend elements are undefined

  notes:
    set must be defined
    checks nth
    see also: setdelnthsorted

  design:
    create new set
    setup pointers and allocate room for prepend'ed entries
    append head of old set to new set
    append tail of old set to new set
*/
// 创建一个不包含第n个元素的排序集合，如果有prepend参数，则前prepend个元素未定义
setT *qh_setnew_delnthsorted(qhT *qh, setT *set, int size, int nth, int prepend) {
  setT *newset;
  void **oldp, **newp;
  int tailsize= size - nth -1, newsize;

  // 检查第n个元素是否超出集合范围，如果是则输出错误信息并退出程序
  if (tailsize < 0) {
    qh_fprintf(qh, qh->qhmem.ferr, 6176, "qhull internal error (qh_setnew_delnthsorted): nth %d is out-of-bounds for set:\n", nth);
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  newsize= size-1 + prepend;
  // 创建新的集合
  newset= qh_setnew(qh, newsize);
  newset->e[newset->maxsize].i= newsize+1;  /* may be overwritten */
  oldp= SETaddr_(set, void);
  newp= SETaddr_(newset, void) + prepend;
  switch (nth) {
  case 0:
    break;
  case 1:
    *(newp++)= *oldp++;
    break;
  case 2:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  case 3:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  case 4:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  default:
    // 复制剩余元素到新集合
    memcpy((char *)newp, (char *)oldp, (size_t)nth * SETelemsize);
    newp += nth;
    oldp += nth;
    break;
  }
  oldp++;
  switch (tailsize) {
  case 0:
    break;
  case 1:
    *(newp++)= *oldp++;
    break;
  case 2:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  case 3:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  default:
    // 复制剩余尾部元素到新集合
    memcpy((char *)newp, (char *)oldp, (size_t)tailsize * SETelemsize);
    break;
  }
  return newset;
}
    *(newp++)= *oldp++;

将指针 oldp 指向的值复制给 newp 指向的位置，并递增 oldp 和 newp 的指针位置。


    *(newp++)= *oldp++;

再次将指针 oldp 指向的值复制给 newp 指向的位置，并递增 oldp 和 newp 的指针位置。


    *(newp++)= *oldp++;

再次将指针 oldp 指向的值复制给 newp 指向的位置，并递增 oldp 和 newp 的指针位置。


    break;

结束当前 switch 语句的执行。


  case 4:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;

如果 switch 的值为 4，则依次将 oldp 指向的值复制给 newp 指向的位置，递增 oldp 和 newp 的指针位置，然后结束 case 4 的执行。


  default:
    memcpy((char *)newp, (char *)oldp, (size_t)tailsize * SETelemsize);
    newp += tailsize;

对于其他情况，使用 memcpy 将 oldp 指向的内存块复制到 newp 指向的位置，复制长度为 tailsize * SETelemsize 字节，然后将 newp 指针移动到已复制数据的末尾。


  }
  *newp= NULL;

结束 switch 结构后，在 newp 指向的位置设置 NULL 值，表示数据复制结束。


  return(newset);

返回指向新数据集的指针 newset。
/* <a href="qh-set_r.htm#TOC">导航到qh-set_r.htm的链接</a> */
/* 设置打印函数，将集合元素打印到文件流fp，使用字符串string标识 */

void qh_setprint(qhT *qh, FILE *fp, const char* string, setT *set) {
    int size, k;

    /* 如果集合为空，则输出错误信息 */
    if (!set)
        qh_fprintf(qh, fp, 9346, "%s set is null\n", string);
    else {
        /* 获取集合的大小 */
        SETreturnsize_(set, size);
        /* 打印集合信息，包括集合地址、最大大小、当前大小和元素列表 */
        qh_fprintf(qh, fp, 9347, "%s set=%p maxsize=%d size=%d elems=",
                   string, set, set->maxsize, size);
        if (size > set->maxsize)
            size= set->maxsize+1;
        /* 打印集合中元素的地址 */
        for (k=0; k < size; k++)
            qh_fprintf(qh, fp, 9348, " %p", set->e[k].p);
        qh_fprintf(qh, fp, 9349, "\n");
    }
} /* setprint */

/* <a href="qh-set_r.htm#TOC">导航到qh-set_r.htm的链接</a> */
/* 替换集合中的旧元素为新元素 */

void qh_setreplace(qhT *qh, setT *set, void *oldelem, void *newelem) {
    void **elemp;

    elemp= SETaddr_(set, void);
    /* 查找并替换集合中的旧元素 */
    while (*elemp != oldelem && *elemp)
        elemp++;
    if (*elemp)
        *elemp= newelem;
    else {
        /* 如果未找到旧元素，则输出错误信息，并打印集合内容 */
        qh_fprintf(qh, qh->qhmem.ferr, 6177, "qhull internal error (qh_setreplace): elem %p not found in set\n",
                   oldelem);
        qh_setprint(qh, qh->qhmem.ferr, "", set);
        qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
    }
} /* setreplace */

/* <a href="qh-set_r.htm#TOC">导航到qh-set_r.htm的链接</a> */
/* 返回集合的大小 */

int qh_setsize(qhT *qh, setT *set) {
    int size;
    setelemT *sizep;

    if (!set)
        return(0);
    sizep= SETsizeaddr_(set);
    /* 获取集合的实际大小并检查是否超过最大大小 */
    if ((size= sizep->i)) {
        size--;
        if (size > set->maxsize) {
            qh_fprintf(qh, qh->qhmem.ferr, 6178, "qhull internal error (qh_setsize): current set size %d is greater than maximum size %d\n",
                       size, set->maxsize);
            qh_setprint(qh, qh->qhmem.ferr, "set: ", set);
            qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
        }
    }else
        size= set->maxsize;
    return size;
} /* setsize */

/* <a href="qh-set_r.htm#TOC">导航到qh-set_r.htm的链接</a> */
/* 返回一个大小为setsize的临时栈集合 */

void qh_settemp(qhT *qh, setsize ) {
    /* 分配一个临时集合并将其加入到qh->qhmem.tempstack中 */
    return set;
} /* settemp */
/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settempfree">-</a>

  qh_settempfree(qh, set )
    释放位于qh->qhmem.tempstack顶部的临时集合

  notes:
    如果set为空，则不执行任何操作
    如果set不是由先前的qh_settemp分配的，则会报错

  to locate errors:
    使用'T2'来查找源，并找到不匹配的qh_settemp

  design:
    检查qh->qhmem.tempstack的顶部
    释放它
*/
void qh_settempfree(qhT *qh, setT **set) {
  setT *stackedset;

  if (!*set)
    return;
  stackedset= qh_settemppop(qh);
  if (stackedset != *set) {
    qh_settemppush(qh, stackedset);
    qh_fprintf(qh, qh->qhmem.ferr, 6179, "qhull internal error (qh_settempfree): set %p(size %d) was not last temporary allocated(depth %d, set %p, size %d)\n",
             *set, qh_setsize(qh, *set), qh_setsize(qh, qh->qhmem.tempstack)+1,
             stackedset, qh_setsize(qh, stackedset));
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  qh_setfree(qh, set);
} /* settempfree */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settempfree_all">-</a>

  qh_settempfree_all(qh)
    释放qh->qhmem.tempstack中的所有临时集合

  design:
    对于tempstack中的每个集合
      释放集合
    释放qh->qhmem.tempstack
*/
void qh_settempfree_all(qhT *qh) {
  setT *set, **setp;

  FOREACHset_(qh->qhmem.tempstack)
    qh_setfree(qh, &set);
  qh_setfree(qh, &qh->qhmem.tempstack);
} /* settempfree_all */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settemppop">-</a>

  qh_settemppop(qh)
    弹出并返回qh->qhmem.tempstack中的临时集合

  notes:
    返回的集合是永久的

  design:
    弹出并检查qh->qhmem.tempstack的顶部
*/
setT *qh_settemppop(qhT *qh) {
  setT *stackedset;

  stackedset= (setT *)qh_setdellast(qh->qhmem.tempstack);
  if (!stackedset) {
    qh_fprintf(qh, qh->qhmem.ferr, 6180, "qhull internal error (qh_settemppop): pop from empty temporary stack\n");
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  if (qh->qhmem.IStracing >= 5)
    qh_fprintf(qh, qh->qhmem.ferr, 8124, "qh_settemppop: depth %d temp set %p of %d elements\n",
       qh_setsize(qh, qh->qhmem.tempstack)+1, stackedset, qh_setsize(qh, stackedset));
  return stackedset;
} /* settemppop */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settemppush">-</a>

  qh_settemppush(qh, set )
    将临时集合推送到qh->qhmem.tempstack（使其成为临时的）

  notes:
    用于跟踪的settemp()的复制版本

  design:
    将临时集合推送到qh->qhmem.tempstack
*/
    # 将 "append set to tempstack" 添加到 tempstack 中
/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settruncate">-</a>

  qh_settruncate(qh, set, size )
    truncate set to size elements

  notes:
    set must be defined

  see:
    SETtruncate_

  design:
    check size
    update actual size of set
*/
void qh_settruncate(qhT *qh, setT *set, int size) {
  // 检查要截断的大小是否有效，不能小于0或大于集合的最大容量
  if (size < 0 || size > set->maxsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6181, "qhull internal error (qh_settruncate): size %d out of bounds for set:\n", size);
    // 打印出错信息和当前集合内容
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    // 引发程序终止
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  // 更新集合的实际大小到指定的大小
  set->e[set->maxsize].i= size+1;   /* maybe overwritten */
  // 将超过指定大小的元素置空
  set->e[size].p= NULL;
} /* settruncate */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setunique">-</a>

  qh_setunique(qh, set, elem )
    add elem to unsorted set unless it is already in set

  notes:
    returns 1 if it is appended

  design:
    if elem not in set
      append elem to set
*/
int qh_setunique(qhT *qh, setT **set, void *elem) {
  // 如果元素不在集合中，则将其添加到集合中
  if (!qh_setin(*set, elem)) {
    qh_setappend(qh, set, elem);
    // 返回1表示已经追加了新元素
    return 1;
  }
  // 返回0表示元素已存在于集合中
  return 0;
} /* setunique */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setzero">-</a>

  qh_setzero(qh, set, index, size )
    zero elements from index on
    set actual size of set to size

  notes:
    set must be defined
    the set becomes an indexed set (can not use FOREACH...)

  see also:
    qh_settruncate

  design:
    check index and size
    update actual size
    zero elements starting at e[index]
*/
void qh_setzero(qhT *qh, setT *set, int idx, int size) {
  int count;

  // 检查索引和大小是否有效，索引必须在0到size之间，size不能超过集合的最大容量
  if (idx < 0 || idx >= size || size > set->maxsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6182, "qhull internal error (qh_setzero): index %d or size %d out of bounds for set:\n", idx, size);
    // 打印出错信息和当前集合内容
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    // 引发程序终止
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  // 更新集合的实际大小到指定的大小
  set->e[set->maxsize].i=  size+1;  /* may be overwritten */
  // 计算要清零的元素数量
  count= size - idx + 1;   /* +1 for NULL terminator */
  // 使用memset函数将从索引开始的元素清零
  memset((char *)SETelemaddr_(set, idx, void), 0, (size_t)count * SETelemsize);
} /* setzero */
```