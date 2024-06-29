# `.\numpy\numpy\random\src\philox\philox-test-data-gen.c`

```py
/*
 * 生成测试用的 CSV 文件
 *
 *  cl philox-test-data-gen.c /Ox
 *  philox-test-data-gen.exe
 *
 *  gcc philox-test-data-gen.c -o philox-test-data-gen
 *  ./philox-test-data-gen
 *
 * 需要 Random123 目录包含头文件与该文件位于同一目录中（未包含在此处）。
 *
 */

#include "../splitmix64/splitmix64.h"  // 包含 splitmix64 头文件
#include "Random123/philox.h"  // 包含 philox 头文件
#include <inttypes.h>  // 包含格式化输出宏定义
#include <stdio.h>  // 标准输入输出库

#define N 1000  // 定义常量 N 为 1000

int main() {
  philox4x64_ctr_t ctr = {{0, 0, 0, 0}};  // 初始化 philox4x64 计数器 ctr
  philox4x64_key_t key = {{0, 0}};  // 初始化 philox4x64 密钥 key
  uint64_t state, seed = 0xDEADBEAF;  // 定义变量 state 和种子 seed，初始化 seed 为 0xDEADBEAF
  philox4x64_ctr_t out;  // 定义输出结构体 out
  uint64_t store[N];  // 定义存储数组 store，大小为 N
  state = seed;  // 将 state 初始化为 seed 的值
  int i, j;  // 定义循环变量 i 和 j

  // 生成两个 64 位随机数作为密钥的初始值
  for (i = 0; i < 2; i++) {
    key.v[i] = splitmix64_next(&state);
  }

  // 使用 Philox 算法生成 N 个随机数，并写入 store 数组
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;  // 计数器递增
    out = philox4x64_R(philox4x64_rounds, ctr, key);  // 执行 Philox 算法
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];  // 将生成的随机数存入 store 数组
    }
  }

  // 打开并写入第一个 CSV 文件
  FILE *fp;
  fp = fopen("philox-testset-1.csv", "w");
  if (fp == NULL) {  // 检查文件是否成功打开
    printf("无法打开文件\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);  // 写入种子值到文件
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);  // 写入索引和随机数到文件
    if (i == 999) {  // 如果索引为 999，也打印到控制台
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);  // 关闭文件

  // 重置计数器和状态，并重新生成种子和密钥
  ctr.v[0] = 0;
  state = seed = 0;
  for (i = 0; i < 2; i++) {
    key.v[i] = splitmix64_next(&state);
  }

  // 再次使用 Philox 算法生成 N 个随机数，并写入 store 数组
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;  // 计数器递增
    out = philox4x64_R(philox4x64_rounds, ctr, key);  // 执行 Philox 算法
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];  // 将生成的随机数存入 store 数组
    }
  }

  // 打开并写入第二个 CSV 文件
  fp = fopen("philox-testset-2.csv", "w");
  if (fp == NULL) {  // 检查文件是否成功打开
    printf("无法打开文件\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);  // 写入种子值到文件
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);  // 写入索引和随机数到文件
    if (i == 999) {  // 如果索引为 999，也打印到控制台
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);  // 关闭文件
}
```