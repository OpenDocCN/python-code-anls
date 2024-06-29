# `.\numpy\tools\swig\test\Vector.cxx`

```
// 包含标准库头文件：stdlib.h（包含通用函数）、math.h（包含数学函数）、iostream（输入输出流）
#include <stdlib.h>
#include <math.h>
#include <iostream>
// 引入自定义头文件 "Vector.h"
#include "Vector.h"

// 定义宏 TEST_FUNCS，生成一系列函数，用于处理给定类型的一维数组
#define TEST_FUNCS(TYPE, SNAME) \
\
// 计算向量长度，接受长度为3的向量数组作为参数
TYPE SNAME ## Length(TYPE vector[3]) {                   \
  double result = 0;                                     \
  // 计算向量的模长
  for (int i=0; i<3; ++i) result += vector[i]*vector[i]; \
  return (TYPE)sqrt(result);                        \
}                                                        \
\
// 计算数组元素的乘积，接受指向数组的指针和数组大小作为参数
TYPE SNAME ## Prod(TYPE * series, int size) {     \
  TYPE result = 1;                                \
  // 计算数组元素的乘积
  for (int i=0; i<size; ++i) result *= series[i]; \
  return result;                                  \
}                                                 \
\
// 计算数组元素的总和，接受数组大小和指向数组的指针作为参数
TYPE SNAME ## Sum(int size, TYPE * series) {      \
  TYPE result = 0;                                \
  // 计算数组元素的总和
  for (int i=0; i<size; ++i) result += series[i]; \
  return result;                                  \
}                                                 \
\
// 将数组中的元素反转，接受长度为3的数组作为参数
void SNAME ## Reverse(TYPE array[3]) { \
  TYPE temp = array[0];               \
  array[0] = array[2];                 \
  array[2] = temp;                     \
}                                      \
\
// 将数组中的所有元素设置为1，接受指向数组的指针和数组大小作为参数
void SNAME ## Ones(TYPE * array, int size) { \
  // 将数组中所有元素设置为1
  for (int i=0; i<size; ++i) array[i] = 1;   \
}                                            \
\
// 将数组中的所有元素设置为0，接受数组大小和指向数组的指针作为参数
void SNAME ## Zeros(int size, TYPE * array) { \
  // 将数组中所有元素设置为0
  for (int i=0; i<size; ++i) array[i] = 0;    \
}                                             \
\
// 将一个长度为3的向量数组拆分为奇数位和偶数位数组，接受源向量、偶数位数组和奇数位数组作为参数
void SNAME ## EOSplit(TYPE vector[3], TYPE even[3], TYPE odd[3]) { \
  for (int i=0; i<3; ++i) {                       \
    // 将偶数位元素复制到偶数位数组
    if (i % 2 == 0) {                           \
      even[i] = vector[i];                       \
      // 将奇数位元素设置为0
      odd[ i] = 0;                           \
    } else {                               \
      even[i] = 0;                           \  # 如果当前索引处的元素为偶数，则将 even 数组对应位置置为 0
      odd[ i] = vector[i];                       \  # 如果当前索引处的元素为奇数，则将 odd 数组对应位置设置为 vector 数组当前位置的值
    }                                   \  # 结束 if-else 分支
  }                                   \  # 结束 for 循环
# 定义一个以给定类型和名称为前缀的函数，将给定数组中的所有元素设置为2
void SNAME ## Twos(TYPE* twoVec, int size) {
  # 循环遍历数组，将每个元素设置为2
  for (int i=0; i<size; ++i) twoVec[i] = 2;
}

# 定义一个以给定名称为前缀的函数，将给定数组中的所有元素设置为3
void SNAME ## Threes(int size, TYPE* threeVec) {
  # 循环遍历数组，将每个元素设置为3
  for (int i=0; i<size; ++i) threeVec[i] = 3;
}

# 以下是为不同数据类型定义测试函数的宏展开
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


每个函数和宏的作用已经通过注释清楚地描述了出来，不改变代码的结构和缩进。
```