# `.\numpy\tools\swig\test\Fortran.h`

```
#ifndef FORTRAN_H
#define FORTRAN_H

// 定义一个宏 TEST_FUNC_PROTOS，用于生成函数原型声明
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
// 声明一个函数原型，函数名由 SNAME 和 "SecondElement" 组成，参数为指向 TYPE 类型数组的指针、行数和列数
TYPE SNAME ## SecondElement(TYPE * matrix, int rows, int cols); \

// 使用宏 TEST_FUNC_PROTOS 分别生成以下类型的函数原型声明
// 以下各行对应的注释格式相同，仅注明了宏的作用和生成的函数原型的具体参数
TEST_FUNC_PROTOS(signed char       , schar    )
TEST_FUNC_PROTOS(unsigned char     , uchar    )
TEST_FUNC_PROTOS(short             , short    )
TEST_FUNC_PROTOS(unsigned short    , ushort   )
TEST_FUNC_PROTOS(int               , int      )
TEST_FUNC_PROTOS(unsigned int      , uint     )
TEST_FUNC_PROTOS(long              , long     )
TEST_FUNC_PROTOS(unsigned long     , ulong    )
TEST_FUNC_PROTOS(long long         , longLong )
TEST_FUNC_PROTOS(unsigned long long, ulongLong)
TEST_FUNC_PROTOS(float             , float    )
TEST_FUNC_PROTOS(double            , double   )

#endif
```