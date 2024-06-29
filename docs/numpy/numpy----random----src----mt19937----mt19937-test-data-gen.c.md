# `.\numpy\numpy\random\src\mt19937\mt19937-test-data-gen.c`

```
/*
 * 生成测试用的 CSV 文件
 *
 * cl mt19937-test-data-gen.c randomkit.c
 *   -IC:\Anaconda\Lib\site-packages\numpy\_core\include -IC:\Anaconda\include
 *   Advapi32.lib Kernel32.lib C:\Anaconda\libs\python36.lib  -DRK_NO_WINCRYPT=1
 *
 */
#include "randomkit.h"     // 包含随机数生成器的头文件
#include <inttypes.h>      // 提供整数格式转换宏
#include <stdio.h>         // 标准输入输出库

#define N 1000             // 定义生成随机数的数量

int main() {
  uint64_t sum = 0;        // 用于计算随机数总和的变量
  uint32_t seed = 0xDEADBEAF;  // 随机数种子
  int i;                   // 循环变量
  rk_state state;          // 随机数生成器状态结构体
  rk_seed(seed, &state);   // 初始化随机数生成器状态
  uint64_t store[N];       // 存储生成的随机数数组

  // 生成随机数并存储到数组中
  for (i = 0; i < N; i++) {
    store[i] = (uint64_t)rk_random(&state);
  }

  FILE *fp;
  fp = fopen("mt19937-testset-1.csv", "w");  // 打开第一个 CSV 文件进行写操作
  if (fp == NULL) {
    printf("Couldn't open file\n");  // 打开文件失败时输出错误信息
    return -1;  // 返回错误状态码
  }
  fprintf(fp, "seed, 0x%" PRIx32 "\n", seed);  // 将种子写入文件
  // 将随机数和其对应的索引写入文件
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);  // 打印最后一个随机数的值
    }
  }
  fclose(fp);  // 关闭文件

  seed = 0;  // 重置种子
  rk_seed(seed, &state);  // 使用新种子重新初始化随机数生成器状态
  // 重新生成随机数并存储到数组中
  for (i = 0; i < N; i++) {
    store[i] = (uint64_t)rk_random(&state);
  }
  fp = fopen("mt19937-testset-2.csv", "w");  // 打开第二个 CSV 文件进行写操作
  if (fp == NULL) {
    printf("Couldn't open file\n");  // 打开文件失败时输出错误信息
    return -1;  // 返回错误状态码
  }
  fprintf(fp, "seed, 0x%" PRIx32 "\n", seed);  // 将新种子写入文件
  // 将新生成的随机数和其对应的索引写入文件
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);  // 打印最后一个随机数的值
    }
  }
  fclose(fp);  // 关闭文件
}
```