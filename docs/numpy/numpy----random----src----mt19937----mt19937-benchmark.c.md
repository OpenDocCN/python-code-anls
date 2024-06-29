# `.\numpy\numpy\random\src\mt19937\mt19937-benchmark.c`

```py
/*
 * 编译命令行示例（Windows下）：
 * cl mt19937-benchmark.c mt19937.c /Ox
 * 测量执行时间命令（Windows下）：
 * Measure-Command { .\mt19937-benchmark.exe }
 *
 * 编译命令行示例（Linux下）：
 * gcc mt19937-benchmark.c mt19937.c -O3 -o mt19937-benchmark
 * 运行可执行文件时间（Linux下）：
 * time ./mt19937-benchmark
 */
#include "mt19937.h"      // 引入自定义的Mersenne Twister实现头文件
#include <inttypes.h>     // 提供符合C99标准的整数类型
#include <stdio.h>        // 标准输入输出库
#include <time.h>         // 时间库，用于计时

#define Q 1000000000      // 定义循环次数

int main() {
  int i;
  uint32_t seed = 0x0;    // 设置随机数生成器的种子
  uint64_t sum = 0, count = 0;  // 初始化总和与计数变量
  mt19937_state state;    // 定义Mersenne Twister状态变量
  mt19937_seed(&state, seed);  // 初始化Mersenne Twister随机数生成器
  clock_t begin = clock();  // 记录开始时间
  for (i = 0; i < Q; i++) {
    sum += mt19937_next64(&state);  // 生成64位随机数并累加到总和
    count++;              // 计数器加一
  }
  clock_t end = clock();  // 记录结束时间
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;  // 计算运行时间
  printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, count);  // 输出总和和计数
  printf("%" PRIu64 " randoms per second\n",
         (uint64_t)(Q / time_spent) / 1000000 * 1000000);  // 输出每秒生成的随机数数量
}
```