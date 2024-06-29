# `.\numpy\numpy\_core\include\numpy\random\bitgen.h`

```
#ifndef NUMPY_CORE_INCLUDE_NUMPY_RANDOM_BITGEN_H_
#define NUMPY_CORE_INCLUDE_NUMPY_RANDOM_BITGEN_H_

#pragma once
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/* Must match the declaration in numpy/random/<any>.pxd */

// 定义了一个名为 bitgen 的结构体类型，用于封装随机数生成器的状态和相关函数指针
typedef struct bitgen {
  void *state;                       // 指向随机数生成器状态的指针
  uint64_t (*next_uint64)(void *st); // 函数指针，用于生成下一个 uint64_t 类型的随机数
  uint32_t (*next_uint32)(void *st); // 函数指针，用于生成下一个 uint32_t 类型的随机数
  double (*next_double)(void *st);   // 函数指针，用于生成下一个 double 类型的随机数
  uint64_t (*next_raw)(void *st);    // 函数指针，用于生成下一个原始数据类型（uint64_t）的随机数
} bitgen_t;

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_RANDOM_BITGEN_H_ */
```