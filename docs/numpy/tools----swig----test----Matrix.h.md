# `.\numpy\tools\swig\test\Matrix.h`

```
// 如果未定义过 MATRIX_H，则开始定义 MATRIX_H
#ifndef MATRIX_H
#define MATRIX_H

// 以下宏定义了一组函数原型，这些函数用于处理二维数组，其形式为
//
//     TYPE SNAMEDet(    TYPE matrix[2][2]);
//     TYPE SNAMEMax(    TYPE * matrix, int rows, int cols);
//     TYPE SNAMEMin(    int rows, int cols, TYPE * matrix);
//     void SNAMEScale(  TYPE array[3][3], TYPE val);
//     void SNAMEFloor(  TYPE * array, int rows, int cols, TYPE floor);
//     void SNAMECeil(   int rows, int cols, TYPE * array, TYPE ceil );
//     void SNAMELUSplit(TYPE matrix[3][3], TYPE lower[3][3], TYPE upper[3][3]);
//
// 对于任何指定的类型 TYPE（例如：short、unsigned int、long long 等），以及给定的短名称 SNAME（例如：schar、uint、longLong 等）。
// 然后，根据给定的 TYPE/SNAME 对展开宏。生成的函数用于测试 numpy 接口，分别用于：
//
//  * 二维输入数组，硬编码长度
//  * 二维输入数组
//  * 二维输入数组，数据在最后
//  * 二维原位数组，硬编码长度
//  * 二维原位数组
//  * 二维原位数组，数据在最后
//  * 二维输出参数数组，硬编码长度
//
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
TYPE SNAME ## Det(    TYPE matrix[2][2]); \
TYPE SNAME ## Max(    TYPE * matrix, int rows, int cols); \
TYPE SNAME ## Min(    int rows, int cols, TYPE * matrix); \
void SNAME ## Scale(  TYPE array[3][3], TYPE val); \
void SNAME ## Floor(  TYPE * array, int rows, int cols, TYPE floor); \
void SNAME ## Ceil(   int rows, int cols, TYPE * array, TYPE ceil ); \
void SNAME ## LUSplit(TYPE matrix[3][3], TYPE lower[3][3], TYPE upper[3][3]);

// 使用宏 TEST_FUNC_PROTOS 展开各种类型和对应的短名称，定义相关的函数原型
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

// 结束宏定义部分
#endif
```