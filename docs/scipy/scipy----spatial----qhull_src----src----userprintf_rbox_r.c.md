# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\userprintf_rbox_r.c`

```
/*
   <html><pre>  -<a href="qh-user_r.htm">-------------------------------</a><a name="TOP">-</a>

   userprintf_rbox_r.c
   user redefinable function -- qh_fprintf_rbox

   see README.txt  see COPYING.txt for copyright information.

   If you recompile and load this file, then userprintf_rbox_r.o will not be loaded
   from qhull.a or qhull.lib

   See libqhull_r.h for data structures, macros, and user-callable functions.
   See user_r.c for qhull-related, redefinable functions
   see user_r.h for user-definable constants
   See usermem_r.c for qh_exit(), qh_free(), and qh_malloc()
   see Qhull.cpp and RboxPoints.cpp for examples.

   Please report any errors that you fix to qhull@qhull.org
*/

#include "libqhull_r.h"  // 引入 libqhull_r.h 头文件

#include <stdarg.h>  // 引入标准参数处理头文件
#include <stdio.h>   // 引入标准输入输出头文件
#include <stdlib.h>  // 引入标准库头文件

/*-<a href="qh-user_r.htm#TOC">-------------------------------</a><a name="qh_fprintf_rbox">-</a>

   qh_fprintf_rbox(qh, fp, msgcode, format, list of args )
     print arguments to *fp according to format
     Use qh_fprintf_rbox() for rboxlib_r.c

   notes:
     same as fprintf()
     fgets() is not trapped like fprintf()
     exit qh_fprintf_rbox via qh_errexit_rbox()
*/

void qh_fprintf_rbox(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... ) {
    va_list args;  // 定义变量参数列表

    if (!fp) {
      // 如果文件指针为 NULL，则输出错误信息到标准错误流
      qh_fprintf_stderr(6231, "qhull internal error (userprintf_rbox_r.c): fp is 0.  Wrong qh_fprintf_rbox called.\n");
      // 通过 qh_errexit_rbox 退出程序
      qh_errexit_rbox(qh, qh_ERRqhull);
    }
    if (msgcode >= MSG_ERROR && msgcode < MSG_STDERR)
      // 如果消息代码在错误范围内，则输出特定格式的消息代码到文件流
      fprintf(fp, "QH%.4d ", msgcode);
    // 开始处理变量参数列表
    va_start(args, fmt);
    // 将格式化字符串及其参数列表输出到指定文件流
    vfprintf(fp, fmt, args);
    // 结束变量参数列表的处理
    va_end(args);
} /* qh_fprintf_rbox */
```