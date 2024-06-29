# `.\numpy\tools\swig\test\Flat.cxx`

```py
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "Flat.h"

// 宏定义 TEST_FUNCS(TYPE, SNAME) 定义了一组函数，用于处理给定类型 TYPE 的 1D 数组，
// 函数名称采用 SNAMEProcess(TYPE * array, int size) 的格式。
// 这些函数用于测试 numpy 接口，处理：
// 
//  * 具有固定元素数量的多维度原地数组
//
#define TEST_FUNCS(TYPE, SNAME) \
\
// 实现处理函数 SNAME ## Process，用于处理 TYPE 类型的数组，大小为 size
void SNAME ## Process(TYPE * array, int size) {          \
  // 遍历数组，对每个元素执行加一操作
  for (int i=0; i<size; ++i) array[i] += 1;              \
}

// 以下展开宏 TEST_FUNCS(TYPE, SNAME) 以生成具体的处理函数

// 生成 signed char 类型的处理函数 scharProcess
TEST_FUNCS(signed char       , schar    )
// 生成 unsigned char 类型的处理函数 ucharProcess
TEST_FUNCS(unsigned char     , uchar    )
// 生成 short 类型的处理函数 shortProcess
TEST_FUNCS(short             , short    )
// 生成 unsigned short 类型的处理函数 ushortProcess
TEST_FUNCS(unsigned short    , ushort   )
// 生成 int 类型的处理函数 intProcess
TEST_FUNCS(int               , int      )
// 生成 unsigned int 类型的处理函数 uintProcess
TEST_FUNCS(unsigned int      , uint     )
// 生成 long 类型的处理函数 longProcess
TEST_FUNCS(long              , long     )
// 生成 unsigned long 类型的处理函数 ulongProcess
TEST_FUNCS(unsigned long     , ulong    )
// 生成 long long 类型的处理函数 longLongProcess
TEST_FUNCS(long long         , longLong )
// 生成 unsigned long long 类型的处理函数 ulongLongProcess
TEST_FUNCS(unsigned long long, ulongLong)
// 生成 float 类型的处理函数 floatProcess
TEST_FUNCS(float             , float    )
// 生成 double 类型的处理函数 doubleProcess
TEST_FUNCS(double            , double   )
```