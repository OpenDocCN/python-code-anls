# `.\numpy\tools\swig\test\SuperTensor.cxx`

```
// 包含标准库头文件
#include <stdlib.h>
#include <math.h>
// 包含输入输出流库的头文件
#include <iostream>
// 包含自定义头文件 "SuperTensor.h"
#include "SuperTensor.h"

// 宏定义，生成一组函数，用于处理特定类型的四维数组
#define TEST_FUNCS(TYPE, SNAME) \
\
// 计算四维数组的范数，返回结果
TYPE SNAME ## Norm(TYPE supertensor[2][2][2][2]) {
  // 定义并初始化结果变量
  double result = 0;
  // 循环遍历四维数组的所有元素
  for (int l=0; l<2; ++l)
    for (int k=0; k<2; ++k)
      for (int j=0; j<2; ++j)
        for (int i=0; i<2; ++i)
          // 计算每个元素的平方，并累加到结果中
          result += supertensor[l][k][j][i] * supertensor[l][k][j][i];
  // 对结果进行平方根运算并返回，转换为指定类型
  return (TYPE)sqrt(result/16);
}

// 查找四维数组中的最大值，并返回
TYPE SNAME ## Max(TYPE * supertensor, int cubes, int slices, int rows, int cols) {
  // 定义变量保存结果，初始为数组第一个元素
  TYPE result = supertensor[0];
  // 多层循环遍历四维数组的每个元素
  for (int l=0; l<cubes; ++l) {
    for (int k=0; k<slices; ++k) {
      for (int j=0; j<rows; ++j) {
        for (int i=0; i<cols; ++i) {
          // 计算当前元素在一维数组中的索引
          int index = l*slices*rows*cols + k*rows*cols + j*cols + i;
          // 如果当前元素大于保存的最大值，则更新最大值
          if (supertensor[index] > result) result = supertensor[index];
        }
      }
    }
  }
  // 返回找到的最大值
  return result;
}

// 查找四维数组中的最小值，并返回
TYPE SNAME ## Min(int cubes, int slices, int rows, int cols, TYPE * supertensor) {
  // 定义变量保存结果，初始为数组第一个元素
  TYPE result = supertensor[0];
  // 多层循环遍历四维数组的每个元素
  for (int l=0; l<cubes; ++l) {
    for (int k=0; k<slices; ++k) {
      for (int j=0; j<rows; ++j) {
        for (int i=0; i<cols; ++i) {
          // 计算当前元素在一维数组中的索引
          int index = l*slices*rows*cols + k*rows*cols + j*cols + i;
          // 如果当前元素小于保存的最小值，则更新最小值
          if (supertensor[index] < result) result = supertensor[index];
        }
      }
    }
  }
  // 返回找到的最小值
  return result;
}

// 将四维数组中的每个元素乘以自身
void SNAME ## Scale(TYPE supertensor[3][3][3][3]) {
  // 多层循环遍历四维数组的每个元素
  for (int l=0; l<3; ++l)
    for (int k=0; k<3; ++k)
      for (int j=0; j<3; ++j)
        for (int i=0; i<3; ++i)
          // 将当前元素乘以自身
          supertensor[l][k][j][i] *= supertensor[l][k][j][i];
}

// 将四维数组中的每个元素向下取整到指定值
void SNAME ## Floor(TYPE * array, int cubes, int slices, int rows, int cols, TYPE floor) {
  // 多层循环遍历四维数组的每个元素
  for (int l=0; l<cubes; ++l) {
    for (int k=0; k<slices; ++k) {
      for (int j=0; j<rows; ++j) {
        for (int i=0; i<cols; ++i) {
          // 计算当前元素在一维数组中的索引
          int index = l*slices*rows*cols + k*rows*cols + j*cols + i;
          // 如果当前元素小于指定的 floor 值，则将其设为 floor
          if (array[index] < floor) array[index] = floor;
        }
      }
    }
  }
}

// 将四维数组中的每个元素向上取整到指定值
void SNAME ## Ceil(int slices, int cubes, int slices, int rows, int cols, TYPE * array, TYPE ceil) {
  // 多层循环遍历四维数组的每个元素
  for (int l=0; l<cubes; ++l) {
    for (int k=0; k<slices; ++k) {
      for (int j=0; j<rows; ++j) {
        for (int i=0; i<cols; ++i) {
          // 计算当前元素在一维数组中的索引
          int index = l*slices*rows*cols + k*rows*cols + j*cols + i;
          // 如果当前元素大于指定的 ceil 值，则将其设为 ceil
          if (array[index] > ceil) array[index] = ceil;
        }
      }
    }
  }
}

// 将输入的四维数组分解为下三角矩阵和上三角矩阵
void SNAME ## LUSplit(TYPE in[2][2][2][2], TYPE lower[2][2][2][2], TYPE upper[2][2][2][2]) {
  // 多层循环遍历四维数组的每个元素
  for (int l=0; l<2; ++l)
    for (int k=0; k<2; ++k)
      for (int j=0; j<2; ++j)
        for (int i=0; i<2; ++i) {
          // 将输入数组的元素分别存入下三角和上三角矩阵
          if (j > i)
            lower[l][k][j][i] = 0;
          else if (j < i)
            upper[l][k][j][i] = 0;
          else {
            lower[l][k][j][i] = in[l][k][j][i];
            upper[l][k][j][i] = in[l][k][j][i];
          }
        }
}
    for (k=0; k<slices; ++k) {                    \
      // 遍历超级张量的第一维（切片维度）
      for (j=0; j<rows; ++j) {                    \
        // 在当前切片中，遍历第二维（行维度）
        for (i=0; i<cols; ++i) {                    \
          // 在当前行中，遍历第三维（列维度）
          // 计算超级张量中元素的索引
          index = l*slices*rows*cols + k*rows*cols + j*cols + i;            \
          // 检查超级张量中的当前元素是否小于结果值，更新结果值
          if (supertensor[index] < result) result = supertensor[index];    \
        }                       \
      }                                \
    }                                \
  }                                \
  // 返回最终的结果值
  return result;                        \
}                                \
\
void SNAME ## Scale(TYPE array[3][3][3][3], TYPE val) { \
  // 多维数组的缩放函数，将每个元素乘以指定的值
  for (int l=0; l<3; ++l)                 \
    for (int k=0; k<3; ++k)                 \
      for (int j=0; j<3; ++j)                 \
        for (int i=0; i<3; ++i)                 \
          array[l][k][j][i] *= val;                 \
}                             \
\
void SNAME ## Floor(TYPE * array, int cubes, int slices, int rows, int cols, TYPE floor) { \
  // 对一维数组中的元素执行下界截断操作，将小于给定下界的元素设为下界值
  int i, j, k, l, index;                                 \
  for (l=0; l<cubes; ++l) {                 \
    for (k=0; k<slices; ++k) {                             \
      for (j=0; j<rows; ++j) {                             \
        for (i=0; i<cols; ++i) {                             \
          index = l*slices*rows*cols + k*rows*cols + j*cols + i;            \
          if (array[index] < floor) array[index] = floor;                 \
        }                                         \
      }                                         \
    }                                         \
  }                                         \
}                                         \
\
void SNAME ## Ceil(int cubes, int slices, int rows, int cols, TYPE * array, TYPE ceil) { \
  // 对一维数组中的元素执行上界截断操作，将大于给定上界的元素设为上界值
  int i, j, k, l, index;                               \
  for (l=0; l<cubes; ++l) {                           \
    for (k=0; k<slices; ++k) {                           \
      for (j=0; j<rows; ++j) {                           \
        for (i=0; i<cols; ++i) {                           \
          index = l*slices*rows*cols + k*rows*cols + j*cols + i;            \
          if (array[index] > ceil) array[index] = ceil;               \
        }                                   \
      }                                       \
    }                                       \
  }                                       \
}                                       \
\
void SNAME ## LUSplit(TYPE supertensor[2][2][2][2], TYPE lower[2][2][2][2], \
              TYPE upper[2][2][2][2]) {             \
  // 将四维超张量分解为下三角矩阵和上三角矩阵
  int sum;                             \
  for (int l=0; l<2; ++l) {                     \
    for (int k=0; k<2; ++k) {                     \
      for (int j=0; j<2; ++j) {                     \
        for (int i=0; i<2; ++i) {                     \
          sum = i + j + k + l;                     \
          if (sum < 2) {                         \
            lower[l][k][j][i] = supertensor[l][k][j][i];             \
            upper[l][k][j][i] = 0;                     \
          } else {                         \
            upper[l][k][j][i] = supertensor[l][k][j][i];             \
            lower[l][k][j][i] = 0;                     \
          }                             \
        }                           \
      }                                 \
    }                                 \
  }                                 \
}

TEST_FUNCS(signed char       , schar    )
TEST_FUNCS(unsigned char     , uchar    )
# 定义一个宏 TEST_FUNCS，用于生成测试函数和它们的类型名
TEST_FUNCS(short             , short    )
# 生成测试函数及其类型名，类型名为 short
TEST_FUNCS(unsigned short    , ushort   )
# 生成测试函数及其类型名，类型名为 unsigned short
TEST_FUNCS(int               , int      )
# 生成测试函数及其类型名，类型名为 int
TEST_FUNCS(unsigned int      , uint     )
# 生成测试函数及其类型名，类型名为 unsigned int
TEST_FUNCS(long              , long     )
# 生成测试函数及其类型名，类型名为 long
TEST_FUNCS(unsigned long     , ulong    )
# 生成测试函数及其类型名，类型名为 unsigned long
TEST_FUNCS(long long         , longLong )
# 生成测试函数及其类型名，类型名为 long long
TEST_FUNCS(unsigned long long, ulongLong)
# 生成测试函数及其类型名，类型名为 unsigned long long
TEST_FUNCS(float             , float    )
# 生成测试函数及其类型名，类型名为 float
TEST_FUNCS(double            , double   )
# 生成测试函数及其类型名，类型名为 double
```