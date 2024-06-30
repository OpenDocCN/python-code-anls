# `D:\src\scipysrc\scipy\scipy\optimize\_direct\direct-internal.h`

```
#ifndef DIRECT_INTERNAL_H
#define DIRECT_INTERNAL_H

#include "../_directmodule.h"  // 引入外部模块的头文件

#include <stdio.h>   // C 标准输入输出库
#include <stdlib.h>  // C 标准库函数，包含动态内存分配等
#include <math.h>    // C 数学库

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

typedef int integer;         // 定义整型别名 integer
typedef double doublereal;   // 定义双精度浮点型别名 doublereal

#define ASRT(c) if (!(c)) { fprintf(stderr, "DIRECT assertion failure at " __FILE__ ":%d -- " #c "\n", __LINE__); exit(EXIT_FAILURE); }  // 定义断言宏，如果条件不满足，打印错误信息并退出程序

#define MIN(a,b) ((a) < (b) ? (a) : (b))  // 定义取两者最小值的宏
#define MAX(a,b) ((a) > (b) ? (a) : (b))  // 定义取两者最大值的宏

/* DIRsubrout.c */

extern void direct_dirheader_(
     FILE *logfile, integer *version,
     doublereal *x, PyObject *x_seq, integer *n, doublereal *eps, integer *maxf, integer *
     maxt, doublereal *l, doublereal *u, integer *algmethod, integer *
     maxfunc, const integer *maxdeep, doublereal *fglobal, doublereal *fglper,
     integer *ierror, doublereal *epsfix, integer *iepschange, doublereal *
     volper, doublereal *sigmaper);
     // 声明函数 direct_dirheader_，接受一系列参数，用于执行特定的任务

extern PyObject* direct_dirinit_(
     doublereal *f, PyObject* fcn, doublereal *c__,
     integer *length, integer *actdeep, integer *point, integer *anchor,
     integer *free, FILE *logfile, integer *arrayi,
     integer *maxi, integer *list2, doublereal *w, doublereal *x, PyObject* x_seq,
     doublereal *l, doublereal *u, doublereal *minf, integer *minpos,
     doublereal *thirds, doublereal *levels, integer *maxfunc, const integer *
     maxdeep, integer *n, integer *maxor, doublereal *fmax, integer *
     ifeasiblef, integer *iinfeasible, integer *ierror, PyObject *args,
     integer jones, int *force_stop);
     // 声明函数 direct_dirinit_，返回一个 PyObject 指针，并接受一系列参数

extern void direct_dirinitlist_(
     integer *anchor, integer *free, integer *
     point, doublereal *f, integer *maxfunc, const integer *maxdeep);
     // 声明函数 direct_dirinitlist_，用于初始化列表

extern void direct_dirpreprc_(doublereal *u, doublereal *l, integer *n,
                  doublereal *xs1, doublereal *xs2, integer *oops);
     // 声明函数 direct_dirpreprc_，执行预处理操作

extern void direct_dirchoose_(
     integer *anchor, integer *s, integer *actdeep,
     doublereal *f, doublereal *minf, doublereal epsrel, doublereal epsabs, doublereal *thirds,
     integer *maxpos, integer *length, integer *maxfunc, const integer *maxdeep,
     const integer *maxdiv, integer *n, FILE *logfile,
     integer *cheat, doublereal *kmax, integer *ifeasiblef, integer jones);
     // 声明函数 direct_dirchoose_，选择操作

extern void direct_dirdoubleinsert_(
     integer *anchor, integer *s, integer *maxpos, integer *point,
     doublereal *f, const integer *maxdeep, integer *maxfunc,
     const integer *maxdiv, integer *ierror);
     // 声明函数 direct_dirdoubleinsert_，双重插入操作

extern integer direct_dirgetmaxdeep_(integer *pos, integer *length, integer *maxfunc,
                  integer *n);
     // 声明函数 direct_dirgetmaxdeep_，获取最大深度

extern void direct_dirget_i__(
     integer *length, integer *pos, integer *arrayi, integer *maxi,
     integer *n, integer *maxfunc);
     // 声明函数 direct_dirget_i__，获取整型数组

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* DIRECT_INTERNAL_H */
/* 
   外部函数声明，调用 DIRect 算法的不同部分的特定函数。
   每个函数的参数和返回类型在接口说明中有详细定义。
*/

extern void direct_dirsamplepoints_(
     doublereal *c__, integer *arrayi,
     doublereal *delta, integer *sample, integer *start, integer *length,
     FILE *logfile, doublereal *f, integer *free,
     integer *maxi, integer *point, doublereal *x, doublereal *l,
     doublereal *minf, integer *minpos, doublereal *u, integer *n,
     integer *maxfunc, const integer *maxdeep, integer *oops);

/*
   外部函数声明，实现 DIRect 算法的采样点分配和计算。
   参数详细说明：
   - c__: 采样点的坐标数组
   - arrayi: 数组索引
   - delta: 步长
   - sample: 采样数
   - start: 起始位置
   - length: 数据长度
   - logfile: 日志文件
   - f: 函数值
   - free: 自由变量
   - maxi: 最大值
   - point: 点
   - x: 变量数组
   - l: 最小值
   - minf: 最小函数值
   - minpos: 最小位置
   - u: 最大值
   - n: 数量
   - maxfunc: 最大函数
   - maxdeep: 最大深度
   - oops: 错误标志
*/

extern void direct_dirdivide_(
     integer *new__, integer *currentlength,
     integer *length, integer *point, integer *arrayi, integer *sample,
     integer *list2, doublereal *w, integer *maxi, doublereal *f,
     integer *maxfunc, const integer *maxdeep, integer *n);

/*
   外部函数声明，实现 DIRect 算法的分割操作。
   参数详细说明：
   - new__: 新的位置
   - currentlength: 当前长度
   - length: 长度
   - point: 点
   - arrayi: 数组索引
   - sample: 采样数
   - list2: 列表2
   - w: 权重
   - maxi: 最大值
   - f: 函数值
   - maxfunc: 最大函数
   - maxdeep: 最大深度
   - n: 数量
*/

extern void direct_dirinsertlist_(
     integer *new__, integer *anchor, integer *point, doublereal *f,
     integer *maxi, integer *length, integer *maxfunc,
     const integer *maxdeep, integer *n, integer *samp, integer jones);

/*
   外部函数声明，实现 DIRect 算法的列表插入操作。
   参数详细说明：
   - new__: 新的位置
   - anchor: 锚点
   - point: 点
   - f: 函数值
   - maxi: 最大值
   - length: 长度
   - maxfunc: 最大函数
   - maxdeep: 最大深度
   - n: 数量
   - samp: 样本
   - jones: 琼斯
*/

extern void direct_dirreplaceinf_(
     integer *free, integer *freeold,
     doublereal *f, doublereal *c__, doublereal *thirds, integer *length,
     integer *anchor, integer *point, doublereal *c1, doublereal *c2,
     integer *maxfunc, const integer *maxdeep, integer *maxdim, integer *n,
     FILE *logfile, doublereal *fmax, integer jones);

/*
   外部函数声明，实现 DIRect 算法的替换操作。
   参数详细说明：
   - free: 自由变量
   - freeold: 旧的自由变量
   - f: 函数值
   - c__: 常数
   - thirds: 三分之一
   - length: 长度
   - anchor: 锚点
   - point: 点
   - c1: 常数1
   - c2: 常数2
   - maxfunc: 最大函数
   - maxdeep: 最大深度
   - maxdim: 最大维度
   - n: 数量
   - logfile: 日志文件
   - fmax: 最大函数值
   - jones: 琼斯
*/

extern void direct_dirsummary_(
     FILE *logfile, doublereal *x, doublereal *l, doublereal *u,
     integer *n, doublereal *minf, doublereal *fglobal,
     integer *numfunc, integer *ierror);

/*
   外部函数声明，实现 DIRect 算法的总结操作。
   参数详细说明：
   - logfile: 日志文件
   - x: 变量数组
   - l: 最小值
   - u: 最大值
   - n: 数量
   - minf: 最小函数值
   - fglobal: 全局函数值
   - numfunc: 函数数量
   - ierror: 错误标志
*/

extern integer direct_dirgetlevel_(
     integer *pos, integer *length,
     integer *maxfunc, integer *n, integer jones);

/*
   外部函数声明，获取 DIRect 算法的级别。
   参数详细说明：
   - pos: 位置
   - length: 长度
   - maxfunc: 最大函数
   - n: 数量
   - jones: 琼斯
*/

extern PyObject* direct_dirinfcn_(
     PyObject* fcn, doublereal *x, PyObject *x_seq, doublereal *c1,
     doublereal *c2, integer *n, doublereal *f, integer *flag__,
     PyObject* args);

/*
   外部函数声明，实现 DIRect 算法的输入函数。
   参数详细说明：
   - fcn: 函数
   - x: 变量数组
   - x_seq: 变量序列
   - c1: 常数1
   - c2: 常数2
   - n: 数量
   - f: 函数值
   - flag__: 标志
   - args: 参数
*/

/* DIRserial.c / DIRparallel.c */

extern PyObject* direct_dirsamplef_(
     doublereal *c__, integer *arrayi, doublereal
     *delta, integer *sample, integer *new__, integer *length,
     FILE *logfile, doublereal *f, integer *free, integer *maxi,
     integer *point, PyObject* fcn, doublereal *x, PyObject* x_seq, doublereal *l, doublereal *
     minf, integer *minpos, doublereal *u, integer *n, integer *maxfunc,
     const integer *maxdeep, integer *oops, doublereal *fmax, integer *
     ifeasiblef, integer *iinfesiblef, PyObject* args, int *force_stop);

/*
   外部函数声明，实现 DIRect 算法的采样函数。
   参数详细说明：
   - c__: 采样点的坐标数组
   - arrayi: 数组索引
   - delta: 步长
   - sample: 采样数
   - new__: 新的位置
   - length: 长度
   - logfile: 日志文件
   - f: 函数值
   - free: 自由变量
   - maxi: 最大值
   - point: 点
   - fcn: 函数
   - x: 变量数组
   - x_seq: 变量序列
   - l: 最小值
   - minf: 最小函数值
   - minpos: 最小位置
   - u: 最大值
   - n: 数量
   - maxfunc: 最大函数
   - maxdeep: 最大深度
   - oops: 错误标志
   - fmax: 最大函数值
   - ifeasiblef: 可行函数
   - iinfesiblef: 不可行函数
   - args: 参数
   - force_stop: 强制停止
*/

/* DIRect.c */

extern PyObject* direct_direct_(
     PyObject* fcn, doublereal *x, PyObject *x_seq, integer *n, doublereal *eps, doublereal epsabs,
     integer *maxf, integer *maxt,
     int *force_stop, doublereal *minf, doublereal
```