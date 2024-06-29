# `.\numpy\tools\swig\test\Tensor.h`

```py
// 如果未定义 TENSOR_H，则进入条件编译，避免重复包含
#ifndef TENSOR_H
// 定义 TENSOR_H，确保此头文件只被包含一次
#define TENSOR_H

// 下面的宏定义了一系列与 3D 数组相关的函数原型，这些函数根据指定的类型 TYPE（例如：short、unsigned int、long long 等）
// 和简称 SNAME（例如：short、uint、longLong 等）来生成
// 这些函数用于测试与 numpy 接口相关的功能，涵盖了不同数据类型和操作方式的测试需求，具体包括：
//
// TYPE SNAME ## Norm(   TYPE tensor[2][2][2]);              // 对 3D 输入数组进行范数计算
// TYPE SNAME ## Max(    TYPE * tensor, int slices, int rows, int cols);  // 计算 3D 输入数组的最大值
// TYPE SNAME ## Min(    int slices, int rows, int cols, TYPE * tensor);   // 计算 3D 输入数组的最小值
// void SNAME ## Scale(  TYPE array[3][3][3], TYPE val);     // 对 3D 原位数组进行按值缩放
// void SNAME ## Floor(  TYPE * array, int slices, int rows, int cols, TYPE floor);  // 对 3D 原位数组进行下取整
// void SNAME ## Ceil(   int slices, int rows, int cols, TYPE * array, TYPE ceil );  // 对 3D 原位数组进行上取整
// void SNAME ## LUSplit(TYPE tensor[2][2][2], TYPE lower[2][2][2], TYPE upper[2][2][2]);  // 对 3D 输出数组进行 LU 分解

#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
TYPE SNAME ## Norm(   TYPE tensor[2][2][2]); \
TYPE SNAME ## Max(    TYPE * tensor, int slices, int rows, int cols); \
TYPE SNAME ## Min(    int slices, int rows, int cols, TYPE * tensor); \
void SNAME ## Scale(  TYPE array[3][3][3], TYPE val); \
void SNAME ## Floor(  TYPE * array, int slices, int rows, int cols, TYPE floor); \
void SNAME ## Ceil(   int slices, int rows, int cols, TYPE * array, TYPE ceil ); \
void SNAME ## LUSplit(TYPE tensor[2][2][2], TYPE lower[2][2][2], TYPE upper[2][2][2]);

// 根据各种类型和简称，展开测试函数的宏定义
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

// 结束条件编译指令
#endif
```