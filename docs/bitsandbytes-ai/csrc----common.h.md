# `bitsandbytes\csrc\common.h`

```py
// 包含二进制搜索头文件
#include <BinSearch.h>

// 如果未定义 common，则定义 common
#ifndef common
#define common

// 使用 BinSearch 命名空间
using namespace BinSearch;

// 定义块大小为 16384
#define BLOCK_SIZE 16384

// 定义结构体 quantize_block_args，包含量化块所需的参数
struct quantize_block_args {
    BinAlgo<Scalar, float, Direct2> *bin_searcher; // 二进制搜索器指针
    float *code; // 代码数组指针
    float *A; // A 数组指针
    float *absmax; // absmax 数组指针
    unsigned char *out; // 输出数组指针
    long long block_end; // 块结束位置
    long long block_idx; // 块索引
    long long threadidx; // 线程索引
    long long blocksize; // 块大小
};

// 声明量化块函数，接受参数并返回 void 指针
void *quantize_block(void *arguments);

// 结束 common 定义
#endif
```