# `.\numpy\numpy\random\src\pcg64\pcg64-benchmark.c`

```py
/*
 * 以下是一个 C 语言程序，用于基准测试 PCG64 随机数生成器的性能。
 * 在不同的编译器和系统上进行了两次编译测试，分别用了不同的编译选项。
 */

#include "../splitmix64/splitmix64.h"    // 引入 splitmix64 随机数生成器的头文件
#include "pcg64.h"                        // 引入 pcg64 随机数生成器的头文件
#include <inttypes.h>                     // 包含了 int64_t 等整型的定义
#include <stdio.h>                        // 标准输入输出库
#include <time.h>                         // 包含了时钟函数 clock()

#define N 1000000000                      // 循环迭代次数

int main() {
  pcg64_random_t rng;                    // 定义 pcg64 随机数生成器结构体
  uint64_t sum = 0, count = 0;            // 初始化总和与计数
  uint64_t seed = 0xDEADBEAF;             // 种子值
  int i;

#if __SIZEOF_INT128__ && !defined(PCG_FORCE_EMULATED_128BIT_MATH)
  // 如果支持原生的 128 位整数并且没有强制启用模拟的 128 位数学运算
  rng.state = (__uint128_t)splitmix64_next(&seed) << 64;    // 使用 splitmix64 生成高位状态
  rng.state |= splitmix64_next(&seed);                      // 生成低位状态
  rng.inc = (__uint128_t)1;                                 // 设置增量为 1
#else
  // 否则使用分离的 64 位状态
  rng.state.high = splitmix64_next(&seed);                   // 使用 splitmix64 生成高位状态
  rng.state.low = splitmix64_next(&seed);                    // 生成低位状态
  rng.inc.high = 0;                                         // 设置增量高位为 0
  rng.inc.low = 1;                                          // 设置增量低位为 1
#endif

  clock_t begin = clock();                // 记录开始时钟时间
  for (i = 0; i < N; i++) {
    sum += pcg64_random_r(&rng);          // 调用 pcg64 生成随机数并累加到总和
    count++;                              // 增加计数
  }
  clock_t end = clock();                  // 记录结束时钟时间
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;  // 计算运行时间
  printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, count);   // 输出总和与计数
  printf("%" PRIu64 " randoms per second\n",
         (uint64_t)(N / time_spent) / 1000000 * 1000000);      // 输出每秒生成的随机数数量
}
```