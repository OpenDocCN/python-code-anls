# `.\numpy\tools\swig\test\Vector.h`

```
#ifndef VECTOR_H
#define VECTOR_H

// 定义了一系列与一维数组操作相关的函数原型的宏
//
//     TYPE SNAMELength( TYPE vector[3]);
//     TYPE SNAMEProd(   TYPE * series, int size);
//     TYPE SNAMESum(    int size, TYPE * series);
//     void SNAMEReverse(TYPE array[3]);
//     void SNAMEOnes(   TYPE * array,  int size);
//     void SNAMEZeros(  int size, TYPE * array);
//     void SNAMEEOSplit(TYPE vector[3], TYPE even[3], TYPE odd[3]);
//     void SNAMETwos(   TYPE * twoVec, int size);
//     void SNAMEThrees( int size, TYPE * threeVec);
//
// 对于指定的类型 TYPE（例如：short、unsigned int、long long 等），使用简称 SNAME
// （例如：short、uint、longLong 等）定义这些函数原型。宏会根据给定的 TYPE/SNAME 组合
// 进行扩展。生成的函数用于测试 numpy 接口，分别为：
//
//  * 针对固定长度的一维输入数组
//  * 针对一维输入数组
//  * 针对一维输入数组，数据在最后
//  * 针对固定长度的一维原地操作数组
//  * 针对一维原地操作数组
//  * 针对一维原地操作数组，数据在最后
//  * 针对固定长度的一维输出数组
//  * 针对一维输出数组
//  * 针对一维输出数组，数据在最后
//
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
TYPE SNAME ## Length( TYPE vector[3]); \
TYPE SNAME ## Prod(   TYPE * series, int size); \
TYPE SNAME ## Sum(    int size, TYPE * series); \
void SNAME ## Reverse(TYPE array[3]); \
void SNAME ## Ones(   TYPE * array,  int size); \
void SNAME ## Zeros(  int size, TYPE * array); \
void SNAME ## EOSplit(TYPE vector[3], TYPE even[3], TYPE odd[3]); \
void SNAME ## Twos(   TYPE * twoVec, int size); \
void SNAME ## Threes( int size, TYPE * threeVec); \

// 使用宏扩展，生成不同类型的函数原型
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