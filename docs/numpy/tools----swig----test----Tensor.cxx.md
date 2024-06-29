# `.\numpy\tools\swig\test\Tensor.cxx`

```py
// 包含标准库头文件 <stdlib.h> 和 <math.h>
#include <stdlib.h>
#include <math.h>
// 包含输入输出流库头文件 <iostream>
#include <iostream>
// 包含自定义头文件 "Tensor.h"
#include "Tensor.h"

// 定义宏 TEST_FUNCS(TYPE, SNAME)，生成一系列函数用于处理三维数组
#define TEST_FUNCS(TYPE, SNAME) \
\
// 计算三维数组的范数，返回类型为 TYPE
TYPE SNAME ## Norm(TYPE tensor[2][2][2]) {         
  // 初始化结果变量为 0
  double result = 0;                     
  // 遍历三维数组的每个元素并计算平方和
  for (int k=0; k<2; ++k)                 
    for (int j=0; j<2; ++j)                 
      for (int i=0; i<2; ++i)                 
        result += tensor[k][j][i] * tensor[k][j][i]; 
  // 返回平方和的平方根除以 8 的结果，强制转换为指定类型 TYPE
  return (TYPE)sqrt(result/8);                 
}                             

// 计算三维数组的最大值，返回类型为 TYPE
TYPE SNAME ## Max(TYPE * tensor, int slices, int rows, int cols) { 
  // 声明变量 i, j, k, index
  int i, j, k, index;                        
  // 初始化结果变量为第一个元素的值
  TYPE result = tensor[0];                    
  // 嵌套循环遍历三维数组的每个元素
  for (k=0; k<slices; ++k) {                    
    for (j=0; j<rows; ++j) {                    
      for (i=0; i<cols; ++i) {                    
        // 计算当前元素在一维数组中的索引
        index = k*rows*cols + j*cols + i;            
        // 如果当前元素大于结果变量，更新结果变量的值
        if (tensor[index] > result) result = tensor[index];    
      }                                
    }                                
  }                                
  // 返回最大值
  return result;                        
}                                

// 计算三维数组的最小值，返回类型为 TYPE
TYPE SNAME ## Min(int slices, int rows, int cols, TYPE * tensor) {    
  // 声明变量 i, j, k, index
  int i, j, k, index;                        
  // 初始化结果变量为第一个元素的值
  TYPE result = tensor[0];                    
  // 嵌套循环遍历三维数组的每个元素
  for (k=0; k<slices; ++k) {                    
    for (j=0; j<rows; ++j) {                    
      for (i=0; i<cols; ++i) {                    
        // 计算当前元素在一维数组中的索引
        index = k*rows*cols + j*cols + i;            
        // 如果当前元素小于结果变量，更新结果变量的值
        if (tensor[index] < result) result = tensor[index];    
      }                                
    }                                
  }                                
  // 返回最小值
  return result;                        
}                                
void scharScale(signed char array[3][3][3], signed char val) { \
  // 遍历三维数组，对每个元素乘以给定的值
  for (int k=0; k<3; ++k)                 \
    for (int j=0; j<3; ++j)                 \
      for (int i=0; i<3; ++i)                 \
        array[k][j][i] *= val;                 \
}                             \
\
void scharFloor(signed char * array, int slices, int rows, int cols, signed char floor) { \
  int i, j, k, index;                                 \
  // 遍历一维数组，将小于指定下限的元素设为下限值
  for (k=0; k<slices; ++k) {                             \
    for (j=0; j<rows; ++j) {                             \
      for (i=0; i<cols; ++i) {                             \
        index = k*rows*cols + j*cols + i;                     \
        if (array[index] < floor) array[index] = floor;                 \
      }                                         \
    }                                         \
  }                                         \
}                                         \
\
void scharCeil(int slices, int rows, int cols, signed char * array, signed char ceil) { \
  int i, j, k, index;                               \
  // 遍历一维数组，将大于指定上限的元素设为上限值
  for (k=0; k<slices; ++k) {                           \
    for (j=0; j<rows; ++j) {                           \
      for (i=0; i<cols; ++i) {                           \
        index = k*rows*cols + j*cols + i;                           \
        if (array[index] > ceil) array[index] = ceil;               \
      }                                       \
    }                                       \
  }                                       \
}                                       \
\
void scharLUSplit(signed char tensor[2][2][2], signed char lower[2][2][2], \
              signed char upper[2][2][2]) {             \
  int sum;                             \
  // 遍历三维数组，根据索引和小于2的条件分割为上下三角部分
  for (int k=0; k<2; ++k) {                     \
    for (int j=0; j<2; ++j) {                     \
      for (int i=0; i<2; ++i) {                     \
        sum = i + j + k;                     \
        if (sum < 2) {                         \
          lower[k][j][i] = tensor[k][j][i];             \
          upper[k][j][i] = 0;                     \
        } else {                         \
          upper[k][j][i] = tensor[k][j][i];             \
          lower[k][j][i] = 0;                     \
        }                             \
      }                                 \
    }                                 \
  }                                 \
}
```