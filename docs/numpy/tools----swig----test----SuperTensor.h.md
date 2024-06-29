# `.\numpy\tools\swig\test\SuperTensor.h`

```py
// 定义条件编译宏，防止头文件重复包含
#ifndef SUPERTENSOR_H
#define SUPERTENSOR_H

// 宏定义说明：
// 下面的宏定义了一系列函数原型，用于处理4维数组，形式如下：
//
//     TYPE SNAMENorm(TYPE supertensor[2][2][2][2]);
//     TYPE SNAMEMax(TYPE *supertensor, int cubes, int slices, int rows, int cols);
//     TYPE SNAMEMin(int cubes, int slices, int rows, int cols, TYPE *supertensor);
//     void SNAMEScale(TYPE array[3][3][3][3], TYPE val);
//     void SNAMEFloor(TYPE *array, int cubes, int slices, int rows, int cols, TYPE floor);
//     void SNAMECeil(int cubes, int slices, int rows, int cols, TYPE *array, TYPE ceil);
//     void SNAMELUSplit(TYPE in[3][3][3][3], TYPE lower[3][3][3][3], TYPE upper[3][3][3][3]);
//
// 其中TYPE可以是任意指定的类型（如：short, unsigned int, long long等），SNAME为类型的简称（如：short, uint, longLong等）。
// 这些宏根据给定的TYPE/SNAME对扩展为特定的函数原型。这些函数用于测试numpy接口，分别用于：
//
//  * 处理4维输入数组，长度硬编码
//  * 处理4维输入数组
//  * 处理4维输入数组，数据最后
//  * 处理4维原地数组，长度硬编码
//  * 处理4维原地数组
//  * 处理4维原地数组，数据最后
//  * 处理4维输出数组，长度硬编码
//
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
TYPE SNAME##Norm(TYPE supertensor[2][2][2][2]); \
TYPE SNAME##Max(TYPE *supertensor, int cubes, int slices, int rows, int cols); \
TYPE SNAME##Min(int cubes, int slices, int rows, int cols, TYPE *supertensor); \
void SNAME##Scale(TYPE array[3][3][3][3], TYPE val); \
void SNAME##Floor(TYPE *array, int cubes, int slices, int rows, int cols, TYPE floor); \
void SNAME##Ceil(int cubes, int slices, int rows, int cols, TYPE *array, TYPE ceil); \
void SNAME##LUSplit(TYPE supertensor[2][2][2][2], TYPE lower[2][2][2][2], TYPE upper[2][2][2][2]);

// 对各种类型的TYPE/SNAME进行宏扩展，生成相应的函数原型
TEST_FUNC_PROTOS(signed char, schar)
TEST_FUNC_PROTOS(unsigned char, uchar)
TEST_FUNC_PROTOS(short, short)
TEST_FUNC_PROTOS(unsigned short, ushort)
TEST_FUNC_PROTOS(int, int)
TEST_FUNC_PROTOS(unsigned int, uint)
TEST_FUNC_PROTOS(long, long)
TEST_FUNC_PROTOS(unsigned long, ulong)
TEST_FUNC_PROTOS(long long, longLong)
TEST_FUNC_PROTOS(unsigned long long, ulongLong)
TEST_FUNC_PROTOS(float, float)
TEST_FUNC_PROTOS(double, double)

// 结束条件编译宏的定义
#endif
```