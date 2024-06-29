# `.\numpy\numpy\random\src\philox\philox-benchmark.c`

```
/*
 * Simple benchmark command
 *
 *  cl philox-benchmark.c /Ox
 *      使用 Microsoft Visual Studio 编译器 cl 编译 philox-benchmark.c 文件，使用 /Ox 开启最高级别优化
 *
 *  gcc philox-benchmark.c -O3 -o philox-benchmark
 *      使用 GCC 编译器编译 philox-benchmark.c 文件，使用 -O3 开启最高级别优化，生成可执行文件 philox-benchmark
 *
 * Requires the Random123 directory containing header files to be located in the
 * same directory (not included).
 *      需要在同一目录下包含 Random123 目录，其中包含所需的头文件（未包含在本示例中）。
 */
#include "Random123/philox.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#define N 1000000000

int main() {
  philox4x64_ctr_t ctr = {{0, 0, 0, 0}}; // 初始化 Philox 64位计数器
  philox4x64_key_t key = {{0, 0xDEADBEAF}}; // 初始化 Philox 64位密钥
  philox4x64_ctr_t out; // 用于存储 Philox 算法输出的计数器
  uint64_t count = 0, sum = 0; // 初始化统计计数器 count 和累加和 sum
  int i, j;
  clock_t begin = clock(); // 记录开始时间
  for (i = 0; i < N / 4UL; i++) { // 执行 N 次循环（每次生成 4 个随机数）
    ctr.v[0]++; // 增加 Philox 计数器的第一个元素
    out = philox4x64_R(philox4x64_rounds, ctr, key); // 使用 Philox 算法生成随机数
    for (j = 0; j < 4; j++) { // 对生成的 4 个随机数进行累加
      sum += out.v[j]; // 将生成的随机数加入到累加和中
      count++; // 增加计数器 count
    }
  }
  clock_t end = clock(); // 记录结束时间
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // 计算运行时间
  printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, count); // 输出累加和和计数器 count 的值
  printf("%" PRIu64 " randoms per second\n",
         (uint64_t)(N / time_spent) / 1000000 * 1000000); // 输出每秒生成的随机数数量
}
```