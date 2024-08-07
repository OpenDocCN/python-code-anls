# `.\numpy\numpy\_core\src\common\simd\simd_utils.h`

```py
#ifndef _NPY_SIMD_UTILS_H
#define _NPY_SIMD_UTILS_H

// 定义用于创建长度为2的向量的宏，CAST是类型转换宏，I0和I1是初始值，__VA_ARGS__是可变参数列表
#define NPYV__SET_2(CAST, I0, I1, ...) (CAST)(I0), (CAST)(I1)

// 定义用于创建长度为4的向量的宏，CAST是类型转换宏，I0到I3是初始值，__VA_ARGS__是可变参数列表
#define NPYV__SET_4(CAST, I0, I1, I2, I3, ...) \
    (CAST)(I0), (CAST)(I1), (CAST)(I2), (CAST)(I3)

// 定义用于创建长度为8的向量的宏，CAST是类型转换宏，I0到I7是初始值，__VA_ARGS__是可变参数列表
#define NPYV__SET_8(CAST, I0, I1, I2, I3, I4, I5, I6, I7, ...) \
    (CAST)(I0), (CAST)(I1), (CAST)(I2), (CAST)(I3), (CAST)(I4), (CAST)(I5), (CAST)(I6), (CAST)(I7)

// 定义用于创建长度为16的向量的宏，CAST是类型转换宏，I0到I15是初始值，__VA_ARGS__是可变参数列表
#define NPYV__SET_16(CAST, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, ...) \
    NPYV__SET_8(CAST, I0, I1, I2, I3, I4, I5, I6, I7), \
    NPYV__SET_8(CAST, I8, I9, I10, I11, I12, I13, I14, I15)

// 定义用于创建长度为32的向量的宏，CAST是类型转换宏，I0到I31是初始值，__VA_ARGS__是可变参数列表
#define NPYV__SET_32(CAST, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, \
    I16, I17, I18, I19, I20, I21, I22, I23, I24, I25, I26, I27, I28, I29, I30, I31, ...) \
    \
    NPYV__SET_16(CAST, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15), \
    NPYV__SET_16(CAST, I16, I17, I18, I19, I20, I21, I22, I23, I24, I25, I26, I27, I28, I29, I30, I31)

// 定义用于创建长度为64的向量的宏，CAST是类型转换宏，I0到I63是初始值，__VA_ARGS__是可变参数列表
#define NPYV__SET_64(CAST, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, \
    I16, I17, I18, I19, I20, I21, I22, I23, I24, I25, I26, I27, I28, I29, I30, I31, \
    I32, I33, I34, I35, I36, I37, I38, I39, I40, I41, I42, I43, I44, I45, I46, I47, \
    I48, I49, I50, I51, I52, I53, I54, I55, I56, I57, I58, I59, I60, I61, I62, I63, ...) \
    \
    NPYV__SET_32(CAST, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, \
    I16, I17, I18, I19, I20, I21, I22, I23, I24, I25, I26, I27, I28, I29, I30, I31), \
    NPYV__SET_32(CAST, I32, I33, I34, I35, I36, I37, I38, I39, I40, I41, I42, I43, I44, I45, I46, I47, \
    I48, I49, I50, I51, I52, I53, I54, I55, I56, I57, I58, I59, I60, I61, I62, I63)

// 定义用于创建长度为2的向量的宏，CAST是类型转换宏，F是填充值，__VA_ARGS__是可变参数列表
#define NPYV__SET_FILL_2(CAST, F, ...) NPY_EXPAND(NPYV__SET_2(CAST, __VA_ARGS__, F, F))

// 定义用于创建长度为4的向量的宏，CAST是类型转换宏，F是填充值，__VA_ARGS__是可变参数列表
#define NPYV__SET_FILL_4(CAST, F, ...) NPY_EXPAND(NPYV__SET_4(CAST, __VA_ARGS__, F, F, F, F))

// 定义用于创建长度为8的向量的宏，CAST是类型转换宏，F是填充值，__VA_ARGS__是可变参数列表
#define NPYV__SET_FILL_8(CAST, F, ...) NPY_EXPAND(NPYV__SET_8(CAST, __VA_ARGS__, F, F, F, F, F, F, F, F))

// 定义用于创建长度为16的向量的宏，CAST是类型转换宏，F是填充值，__VA_ARGS__是可变参数列表
#define NPYV__SET_FILL_16(CAST, F, ...) NPY_EXPAND(NPYV__SET_16(CAST, __VA_ARGS__, \
    F, F, F, F, F, F, F, F, F, F, F, F, F, F, F))

// 定义用于创建长度为32的向量的宏，CAST是类型转换宏，F是填充值，__VA_ARGS__是可变参数列表
#define NPYV__SET_FILL_32(CAST, F, ...) NPY_EXPAND(NPYV__SET_32(CAST, __VA_ARGS__, \
    F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F))

// 定义用于创建长度为64的向量的宏，CAST是类型转换宏，F是填充值，__VA_ARGS__是可变参数列表
#define NPYV__SET_FILL_64(CAST, F, ...) NPY_EXPAND(NPYV__SET_64(CAST, __VA_ARGS__, \
    F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, \
    F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F))

#endif // _NPY_SIMD_UTILS_H
```