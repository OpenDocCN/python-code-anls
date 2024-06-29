# `.\numpy\tools\swig\test\Flat.h`

```
#ifndef FLAT_H
#define FLAT_H

// 宏定义开始：定义了一组函数原型，用于处理不同类型数组
//
//     void SNAMEProcess(TYPE * array,  int size);
//
// 对于任意指定的类型 TYPE（例如：signed char, unsigned int, long long 等）和简短名称 SNAME
// （例如：schar, uint, longLong 等）。该宏会根据给定的 TYPE/SNAME 组合进行扩展，生成用于测试
// numpy 接口的函数。
//
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
void SNAME ## Process(TYPE * array, int size); \

// 以下是针对不同类型和简短名称的宏扩展示例：
// 扩展为 signed char 对应的处理函数原型
TEST_FUNC_PROTOS(signed char       , schar    )
// 扩展为 unsigned char 对应的处理函数原型
TEST_FUNC_PROTOS(unsigned char     , uchar    )
// 扩展为 short 对应的处理函数原型
TEST_FUNC_PROTOS(short             , short    )
// 扩展为 unsigned short 对应的处理函数原型
TEST_FUNC_PROTOS(unsigned short    , ushort   )
// 扩展为 int 对应的处理函数原型
TEST_FUNC_PROTOS(int               , int      )
// 扩展为 unsigned int 对应的处理函数原型
TEST_FUNC_PROTOS(unsigned int      , uint     )
// 扩展为 long 对应的处理函数原型
TEST_FUNC_PROTOS(long              , long     )
// 扩展为 unsigned long 对应的处理函数原型
TEST_FUNC_PROTOS(unsigned long     , ulong    )
// 扩展为 long long 对应的处理函数原型
TEST_FUNC_PROTOS(long long         , longLong )
// 扩展为 unsigned long long 对应的处理函数原型
TEST_FUNC_PROTOS(unsigned long long, ulongLong)
// 扩展为 float 对应的处理函数原型
TEST_FUNC_PROTOS(float             , float    )
// 扩展为 double 对应的处理函数原型
TEST_FUNC_PROTOS(double            , double   )

#endif
```