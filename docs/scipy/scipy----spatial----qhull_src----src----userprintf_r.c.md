# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\userprintf_r.c`

```
/*<html><pre>  -<a                             href="qh-user_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

  userprintf_r.c
  user redefinable function -- qh_fprintf

  see README.txt  see COPYING.txt for copyright information.

  If you recompile and load this file, then userprintf_r.o will not be loaded
  from qhull_r.a or qhull_r.lib

  See libqhull_r.h for data structures, macros, and user-callable functions.
  See user_r.c for qhull-related, redefinable functions
  see user_r.h for user-definable constants
  See usermem_r.c for qh_exit(), qh_free(), and qh_malloc()
  see Qhull.cpp and RboxPoints.cpp for examples.

  qh_printf is a good location for debugging traps, checked on each log line

  Please report any errors that you fix to qhull@qhull.org
*/

#include "libqhull_r.h"
#include "poly_r.h" /* for qh.tracefacet */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="qh_fprintf">-</a>

  qh_fprintf(qh, fp, msgcode, format, list of args )
    print arguments to *fp according to format
    Use qh_fprintf_rbox() for rboxlib_r.c

  notes:
    sets qh.last_errcode if msgcode is error 6000..6999
    same as fprintf()
    fgets() is not trapped like fprintf()
    exit qh_fprintf via qh_errexit()
    may be called for errors in qh_initstatistics and qh_meminit
*/

void qh_fprintf(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... ) {
  va_list args;
  facetT *neighbor, **neighborp;

  // 检查文件指针是否为空，如果为空则输出错误信息并退出
  if (!fp) {
    if(!qh){
      qh_fprintf_stderr(6241, "qhull internal error (userprintf_r.c): fp and qh not defined for qh_fprintf '%s'\n", fmt);
      qh->last_errcode= 6241;
      qh_exit(qh_ERRqhull);  /* can not use qh_errexit() */
    }
    // 输出错误信息到标准错误输出，并设置错误代码，最后退出
    qh_fprintf_stderr(6028, "qhull internal error (userprintf_r.c): fp is 0.  Wrong qh_fprintf was called.\n");
    qh->last_errcode= 6028;
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  // 根据消息代码和输出选项，选择输出格式
  if ((qh && qh->ANNOTATEoutput) || msgcode < MSG_TRACE4) {
    fprintf(fp, "[QH%.4d]", msgcode);
  }else if (msgcode >= MSG_ERROR && msgcode < MSG_STDERR ) {
    fprintf(fp, "QH%.4d ", msgcode);
  }
  // 使用可变参数列表打印格式化输出到文件流
  va_start(args, fmt);
  vfprintf(fp, fmt, args);
  va_end(args);
    
  // 如果存在 qh 结构体，根据消息代码设置最后的错误代码
  if (qh) {
    if (msgcode >= MSG_ERROR && msgcode < MSG_WARNING)
      qh->last_errcode= msgcode;
    /* Place debugging traps here. Use with trace option 'Tn' 
       Set qh.tracefacet_id, qh.traceridge_id, and/or qh.tracevertex_id in global_r.c
    */
  }
}
    # 如果条件为假，则跳过测试以避免调试陷阱
    if (False) { /* in production skip test for debugging traps */
      # 如果跟踪面存在且已经被测试过
      if (qh->tracefacet && qh->tracefacet->tested) {
        # 检查跟踪面的邻居数是否小于凸壳维度
        if (qh_setsize(qh, qh->tracefacet->neighbors) < qh->hull_dim)
          # 如果条件成立，调用错误处理函数并退出
          qh_errexit(qh, qh_ERRdebug, qh->tracefacet, qh->traceridge);
        # 遍历跟踪面的每个邻居
        FOREACHneighbor_(qh->tracefacet) {
          # 如果邻居不是重复边或合并边，并且可见
          if (neighbor != qh_DUPLICATEridge && neighbor != qh_MERGEridge && neighbor->visible)
            # 调用错误处理函数并退出，传递跟踪面和邻居作为参数
            qh_errexit2(qh, qh_ERRdebug, qh->tracefacet, neighbor);
        }
      } 
      # 如果跟踪边存在且其顶点ID为特定值
      if (qh->traceridge && qh->traceridge->top->id == 234342223) {
        # 调用错误处理函数并退出，传递跟踪面和跟踪边作为参数
        qh_errexit(qh, qh_ERRdebug, qh->tracefacet, qh->traceridge);
      }
      # 如果跟踪顶点存在且其邻居数大于特定值
      if (qh->tracevertex && qh_setsize(qh, qh->tracevertex->neighbors)>3434334) {
        # 调用错误处理函数并退出，传递跟踪面和跟踪边作为参数
        qh_errexit(qh, qh_ERRdebug, qh->tracefacet, qh->traceridge);
      }
    }
    # 如果 FLUSHprint 标志为真，则刷新输出流
    if (qh->FLUSHprint)
      fflush(fp);
  }
} /* qh_fprintf */
```