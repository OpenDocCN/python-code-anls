# `.\numpy\numpy\random\src\pcg64\pcg64-test-data-gen.c`

```py
/*
 * 生成测试用的 CSV 文件
 *
 * 仅适用于 GCC 编译器
 *
 * 编译命令：
 * gcc  pcg64-test-data-gen.c pcg64.orig.c ../splitmix64/splitmix64.c -o pgc64-test-data-gen
 */

#include "pcg64.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  pcg64_random_t rng;  // 定义 pcg64 随机数生成器结构体
  uint64_t state, seed = 0xDEADBEAF;  // 定义状态变量和种子值
  state = seed;  // 将种子值赋给状态变量
  __uint128_t temp, s, inc;  // 定义用于存储状态和增量的 128 位整数变量
  int i;  // 循环变量
  uint64_t store[N];  // 存储生成随机数的数组

  s = (__uint128_t)seed;  // 将种子转换为 128 位整数类型
  inc = (__uint128_t)0;   // 设置增量为 0
  pcg64_srandom_r(&rng, s, inc);  // 初始化随机数生成器
  printf("0x%" PRIx64, (uint64_t)(rng.state >> 64));  // 打印状态的高 64 位十六进制值
  printf("%" PRIx64 "\n", (uint64_t)rng.state);  // 打印状态的低 64 位十六进制值
  printf("0x%" PRIx64, (uint64_t)(rng.inc >> 64));  // 打印增量的高 64 位十六进制值
  printf("%" PRIx64 "\n", (uint64_t)rng.inc);  // 打印增量的低 64 位十六进制值

  for (i = 0; i < N; i++) {
    store[i] = pcg64_random_r(&rng);  // 生成随机数并存储到数组中
  }

  FILE *fp;  // 文件指针
  fp = fopen("pcg64-testset-1.csv", "w");  // 打开文件准备写入
  if (fp == NULL) {  // 检查文件是否成功打开
    printf("Couldn't open file\n");  // 打开失败时输出错误信息
    return -1;  // 返回错误状态码
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);  // 将种子值写入文件
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);  // 将索引和对应的随机数写入文件
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);  // 打印第 1000 个数据的索引和随机数
    }
  }
  fclose(fp);  // 关闭文件

  state = seed = 0;  // 重置状态和种子
  s = (__uint128_t)seed;  // 将种子转换为 128 位整数类型
  i = (__uint128_t)0;  // 将循环变量转换为 128 位整数类型
  pcg64_srandom_r(&rng, s, i);  // 重新初始化随机数生成器
  printf("0x%" PRIx64, (uint64_t)(rng.state >> 64));  // 打印状态的高 64 位十六进制值
  printf("%" PRIx64 "\n", (uint64_t)rng.state);  // 打印状态的低 64 位十六进制值
  printf("0x%" PRIx64, (uint64_t)(rng.inc >> 64));  // 打印增量的高 64 位十六进制值
  printf("%" PRIx64 "\n", (uint64_t)rng.inc);  // 打印增量的低 64 位十六进制值

  for (i = 0; i < N; i++) {
    store[i] = pcg64_random_r(&rng);  // 生成随机数并存储到数组中
  }
  fp = fopen("pcg64-testset-2.csv", "w");  // 打开文件准备写入
  if (fp == NULL) {  // 检查文件是否成功打开
    printf("Couldn't open file\n");  // 打开失败时输出错误信息
    return -1;  // 返回错误状态码
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);  // 将种子值写入文件
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);  // 将索引和对应的随机数写入文件
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);  // 打印第 1000 个数据的索引和随机数
    }
  }
  fclose(fp);  // 关闭文件
}
```