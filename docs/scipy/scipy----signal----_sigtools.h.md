# `D:\src\scipysrc\scipy\scipy\signal\_sigtools.h`

```
#ifndef _SCIPY_PRIVATE_SIGNAL__SIGTOOLS_H_
#define _SCIPY_PRIVATE_SIGNAL__SIGTOOLS_H_

#include "Python.h"

// 定义常量掩码
#define BOUNDARY_MASK 12
#define OUTSIZE_MASK 3
#define FLIP_MASK  16
#define TYPE_MASK  (32+64+128+256+512)
#define TYPE_SHIFT 5

// 定义滤波器的边界模式常量
#define FULL  2
#define SAME  1
#define VALID 0

// 定义卷积的填充模式常量
#define CIRCULAR 8
#define REFLECT  4
#define PAD      0

// 最大支持的类型数
#define MAXTYPES 21


/* 通用结构体，用于在子程序间传递数据
   用于通用函数而不是使用特定于 Python 的结构体，以便这些函数可以轻松地
   在其他脚本语言中使用 */
   
// 通用指针结构体
typedef struct {
  char *data;     // 数据指针
  int elsize;     // 元素大小
} Generic_ptr;

// 通用向量结构体
typedef struct {
  char *data;         // 数据指针
  npy_intp numels;    // 元素数量
  int elsize;         // 元素大小
  char *zero;         // 零的表示指针
} Generic_Vector;

// 通用数组结构体
typedef struct {
  char *data;         // 数据指针
  int  nd;            // 数组维度
  npy_intp  *dimensions;  // 维度数组
  int  elsize;        // 元素大小
  npy_intp  *strides; // 步长数组
  char *zero;         // 零的表示指针
} Generic_Array;

// 多元加法函数指针类型定义
typedef void (MultAddFunction) (char *, npy_intp, char *, npy_intp, char *,
                                npy_intp *, npy_intp *, int, npy_intp, int,
                                npy_intp *, npy_intp *, npy_uintp *);

// 函数声明
PyObject*
scipy_signal__sigtools_linear_filter(PyObject * NPY_UNUSED(dummy), PyObject * args);

PyObject*
scipy_signal__sigtools_correlateND(PyObject *NPY_UNUSED(dummy), PyObject *args);

void
scipy_signal__sigtools_linear_filter_module_init(void);

// 静态函数声明（被注释掉的函数）
/*
static int index_out_of_bounds(int *, int *, int );
static long compute_offsets (unsigned long *, long *, int *, int *, int *, int *, int);
static int increment(int *, int, int *);
static void convolveND(Generic_Array *, Generic_Array *, Generic_Array *, MultAddFunction *, int);
static void RawFilter(Generic_Vector, Generic_Vector, Generic_Array, Generic_Array, Generic_Array *, Generic_Array *, BasicFilterFunction *, int);
*/

#endif
```