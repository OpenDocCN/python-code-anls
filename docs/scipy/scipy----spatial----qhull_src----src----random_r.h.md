# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\random_r.h`

```
/*
   <html><pre>  -<a                             href="qh-geom_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

  random_r.h
    header file for random and utility routines

   see qh-geom_r.htm and random_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/random_r.h#2 $$Change: 2666 $
   $DateTime: 2019/05/30 10:11:25 $$Author: bbarber $
*/

#ifndef qhDEFrandom
#define qhDEFrandom 1

#include "libqhull_r.h"

/*============= prototypes in alphabetical order ======= */

#ifdef __cplusplus
extern "C" {
#endif

/* 函数：qh_argv_to_command
   描述：将命令行参数转换为单个字符串命令
   参数：argc - 参数数量，argv - 参数数组，command - 存放命令的缓冲区，max_size - 缓冲区最大尺寸
   返回：整数，表示生成的命令的长度
*/
int     qh_argv_to_command(int argc, char *argv[], char* command, int max_size);

/* 函数：qh_argv_to_command_size
   描述：计算将命令行参数转换为单个字符串命令所需的缓冲区大小
   参数：argc - 参数数量，argv - 参数数组
   返回：整数，表示生成的命令的长度
*/
int     qh_argv_to_command_size(int argc, char *argv[]);

/* 函数：qh_rand
   描述：生成一个随机数
   参数：qh - qhT类型的结构体指针
   返回：整数，表示生成的随机数
*/
int     qh_rand(qhT *qh);

/* 函数：qh_srand
   描述：初始化随机数种子
   参数：qh - qhT类型的结构体指针，seed - 种子值
   返回：无
*/
void    qh_srand(qhT *qh, int seed);

/* 函数：qh_randomfactor
   描述：计算随机因子
   参数：qh - qhT类型的结构体指针，scale - 缩放因子，offset - 偏移量
   返回：realT类型的数值，表示计算得到的随机因子
*/
realT   qh_randomfactor(qhT *qh, realT scale, realT offset);

/* 函数：qh_randommatrix
   描述：生成随机矩阵
   参数：qh - qhT类型的结构体指针，buffer - 缓冲区，dim - 矩阵维度，row - 矩阵的行
   返回：无
*/
void    qh_randommatrix(qhT *qh, realT *buffer, int dim, realT **row);

/* 函数：qh_strtol
   描述：将字符串转换为长整型数值
   参数：s - 输入的字符串，endp - 结束指针
   返回：转换后的长整型数值
*/
int     qh_strtol(const char *s, char **endp);

/* 函数：qh_strtod
   描述：将字符串转换为双精度浮点数值
   参数：s - 输入的字符串，endp - 结束指针
   返回：转换后的双精度浮点数值
*/
double  qh_strtod(const char *s, char **endp);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFrandom */
*/
```