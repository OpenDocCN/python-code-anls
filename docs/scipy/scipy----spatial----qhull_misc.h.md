# `D:\src\scipysrc\scipy\scipy\spatial\qhull_misc.h`

```
/*
 * Handle qh_new_qhull_scipy entry point.
 * 处理 qh_new_qhull_scipy 入口点。
 */
#ifndef QHULL_MISC_H_
#define QHULL_MISC_H_

/* for CBLAS_INT only*/
#include "npy_cblas.h"
// 仅用于 CBLAS_INT

#define qhull_misc_lib_check() QHULL_LIB_CHECK
// 定义宏 qhull_misc_lib_check()，用于检查 Qhull 库

#include "qhull_src/src/libqhull_r.h"
// 引入 Qhull 库的头文件 libqhull_r.h

int qh_new_qhull_scipy(qhT *qh, int dim, int numpoints, coordT *points, boolT ismalloc,
                       char *qhull_cmd, FILE *outfile, FILE *errfile, coordT* feaspoint);
// 函数声明，定义 qh_new_qhull_scipy 函数，接受 Qhull 结构体指针 qh、维度 dim、点数 numpoints、坐标数组 points、是否为 malloc 分配的内存 ismalloc、Qhull 命令字符串 qhull_cmd、输出文件 outfile、错误文件 errfile、可行点 feaspoint 作为参数

#endif /* QHULL_MISC_H_ */
// 结束头文件宏定义
```