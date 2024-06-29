# `.\numpy\tools\swig\test\Fortran.cxx`

```py
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "Fortran.h"

// 宏定义：TEST_FUNCS(TYPE, SNAME)
// 生成一个函数，用于获取矩阵中第二个元素的值，根据传入的类型和名称生成函数
#define TEST_FUNCS(TYPE, SNAME) \
\
// 函数定义：TYPE SNAME ## SecondElement(TYPE * matrix, int rows, int cols)
// 返回矩阵中第二个元素的值，根据传入的类型和名称生成函数
TYPE SNAME ## SecondElement(TYPE * matrix, int rows, int cols) {      \
  // 从矩阵中取出第二个元素的值
  TYPE result = matrix[1];                                \
  // 返回获取到的值
  return result;                                          \
}                                                         \

// 生成各种类型的 SecondElement 函数
TEST_FUNCS(signed char       , schar    )
TEST_FUNCS(unsigned char     , uchar    )
TEST_FUNCS(short             , short    )
TEST_FUNCS(unsigned short    , ushort   )
TEST_FUNCS(int               , int      )
TEST_FUNCS(unsigned int      , uint     )
TEST_FUNCS(long              , long     )
TEST_FUNCS(unsigned long     , ulong    )
TEST_FUNCS(long long         , longLong )
TEST_FUNCS(unsigned long long, ulongLong)
TEST_FUNCS(float             , float    )
TEST_FUNCS(double            , double   )


这段代码定义了一个宏 `TEST_FUNCS`，用于生成多个函数，这些函数根据不同的数据类型和名称，从传入的数组中获取第二个元素的值，并返回。
```