# `.\numpy\tools\swig\test\Matrix.cxx`

```
// 引入标准库头文件
#include <stdlib.h>
#include <math.h>
// 引入输入输出流库头文件
#include <iostream>
// 引入自定义矩阵头文件
#include "Matrix.h"

// 下面的宏定义了一组针对 2D 数组的函数，这些函数接受特定类型 TYPE 的参数，并使用简称 SNAME
// 宏展开如下，对于给定的类型 TYPE 和简称 SNAME：
//
//     TYPE SNAME ## Det(TYPE matrix[2][2]) { ... }
//     TYPE SNAME ## Max(TYPE * matrix, int rows, int cols) { ... }
//     TYPE SNAME ## Min(int rows, int cols, TYPE * matrix) { ... }
//     void SNAME ## Scale(TYPE array[3][3], TYPE val) { ... }
//
// 这些函数分别用于处理：
//  * 2x2 矩阵的行列式计算
//  * 任意行列数的二维数组中的最大值计算
//  * 任意行列数的二维数组中的最小值计算
//  * 3x3 矩阵的每个元素与给定值的乘积计算
#define TEST_FUNCS(TYPE, SNAME) \
\
TYPE SNAME ## Det(TYPE matrix[2][2]) {                          \
  return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]; \
}                                                               \
\
TYPE SNAME ## Max(TYPE * matrix, int rows, int cols) {      \
  int i, j, index;                                        \
  TYPE result = matrix[0];                                \
  // 遍历二维数组找出最大值
  for (j=0; j<cols; ++j) {                                \
    for (i=0; i<rows; ++i) {                              \
      index = j*rows + i;                                 \
      if (matrix[index] > result) result = matrix[index]; \
    }                                                     \
  }                                                       \
  return result;                                          \
}                                                         \
\
TYPE SNAME ## Min(int rows, int cols, TYPE * matrix) {    \
  int i, j, index;                                        \
  TYPE result = matrix[0];                                \
  // 遍历二维数组找出最小值
  for (j=0; j<cols; ++j) {                                \
    for (i=0; i<rows; ++i) {                              \
      index = j*rows + i;                                 \
      if (matrix[index] < result) result = matrix[index]; \
    }                                                     \
  }                                                       \
  return result;                                          \
}                                                         \
\
void SNAME ## Scale(TYPE array[3][3], TYPE val) { \
  // 遍历 3x3 矩阵，每个元素乘以给定的值
  for (int i=0; i<3; ++i)                         \
    for (int j=0; j<3; ++j)                       \
      array[i][j] *= val;                         \
}                                                 \
\
# 定义函数 SNAME ## Floor，用于将数组中小于给定值 floor 的元素替换为 floor
void SNAME ## Floor(TYPE * array, int rows, int cols, TYPE floor) { \
  # 定义变量 i、j 和数组索引 index，循环遍历数组
  int i, j, index;                                                  \
  # 遍历列
  for (j=0; j<cols; ++j) {                                          \
    # 遍历行
    for (i=0; i<rows; ++i) {                                        \
      # 计算当前元素在一维数组中的索引
      index = j*rows + i;                                           \
      # 如果数组中的元素小于 floor，则将其赋值为 floor
      if (array[index] < floor) array[index] = floor;               \
    }                                                               \
  }                                                                 \
}                                                                   \
\
# 定义函数 SNAME ## Ceil，用于将数组中大于给定值 ceil 的元素替换为 ceil
void SNAME ## Ceil(int rows, int cols, TYPE * array, TYPE ceil) { \
  # 定义变量 i、j 和数组索引 index，循环遍历数组
  int i, j, index;                                                \
  # 遍历列
  for (j=0; j<cols; ++j) {                                        \
    # 遍历行
    for (i=0; i<rows; ++i) {                                      \
      # 计算当前元素在一维数组中的索引
      index = j*rows + i;                                         \
      # 如果数组中的元素大于 ceil，则将其赋值为 ceil
      if (array[index] > ceil) array[index] = ceil;               \
    }                                                             \
  }                                                               \
}                                  \
\
# 定义函数 SNAME ## LUSplit，将3x3矩阵分解为下三角矩阵和上三角矩阵
void SNAME ## LUSplit(TYPE matrix[3][3], TYPE lower[3][3], TYPE upper[3][3]) { \
  # 循环遍历矩阵的行和列
  for (int i=0; i<3; ++i) {                               \
    for (int j=0; j<3; ++j) {                               \
      # 如果行数大于等于列数，则当前位置属于下三角矩阵
      if (i >= j) {                                    \
    # 将原始矩阵中的元素复制到下三角矩阵中
    lower[i][j] = matrix[i][j];                           \
    # 上三角矩阵对应位置置零
    upper[i][j] = 0;                                \
      } else {                                        \
    # 下三角矩阵对应位置置零
    lower[i][j] = 0;                                \
    # 将原始矩阵中的元素复制到上三角矩阵中
    upper[i][j] = matrix[i][j];                           \
      }                                            \
    }                                            \
  }                                            \
}

# 定义测试宏 TEST_FUNCS，用于生成一系列函数的测试函数
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
```