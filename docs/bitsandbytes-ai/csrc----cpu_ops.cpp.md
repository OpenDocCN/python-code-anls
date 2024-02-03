# `bitsandbytes\csrc\cpu_ops.cpp`

```py
// 包含二分搜索头文件
#include <BinSearch.h>
// 根据操作系统选择线程库
#ifdef _WIN32
#include <thread>
#else
#include <pthread.h>
#endif
// 包含通用头文件
#include <common.h>

// 使用 BinSearch 命名空间
using namespace BinSearch;

// CPU 上的反量化函数
void dequantize_cpu(float *code, unsigned char *A, float *absmax, float *out, long long blocksize, long long n) {
    // 遍历每个块
    for (long long block_idx = 0; block_idx < n; block_idx += blocksize) {
        // 计算有效项数
        long long valid_items = n - block_idx >= blocksize ? blocksize : n - block_idx;
        long long block_end = block_idx + valid_items;
        // 计算块的结束位置
        for (long long i = block_idx; i < block_end; i++)
            // 执行反量化操作
            out[i] = code[A[i]] * absmax[block_idx / blocksize];
    }
}

// CPU 上的量化函数
void quantize_cpu(float *code, float *A, float *absmax, unsigned char *out, long long blocksize, long long n)
{
    // 将默认 code 设置为 -1.0f，避免二分搜索算法中的错误
    code[0] = -1.0f;

    // 计算块数
    long long num_blocks = n / blocksize;
    num_blocks += n % blocksize == 0 ? 0 : 1;

    // 定义 code 元素个数
    const uint32 elements_code = 256;
    // 创建二分搜索对象
    BinAlgo<Scalar, float, Direct2> bin_searcher(code, elements_code);

    // 定义线程波大小
    int thread_wave_size = 256;
    // 将线程分成波，每波 256 个线程
    // 在 Linux 上，最大限制在 16k 到 64k 之间（当运行 BLOOM-176B 时，批处理大小较大时会达到这个限制）
    for(long long offset = 0; offset < num_blocks; offset+=thread_wave_size)
    {
        // 计算有效块数
        long long valid_chunks = num_blocks - offset >= thread_wave_size ? thread_wave_size : num_blocks - offset;
#ifdef _WIN32
        // 在 Windows 上分配线程数组
        std::thread *threads = (std::thread *) malloc(sizeof(std::thread) * valid_chunks);
#else
        // 在非 Windows 上分配线程数组
        pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * valid_chunks);
#endif

      // 为存储量化块参数的数组分配内存空间
      struct quantize_block_args **args = (quantize_block_args **) malloc(valid_chunks * sizeof(quantize_block_args *));

      // 为每个有效块的参数结构体分配内存空间
      for(long long i = 0; i < valid_chunks; i++)
          args[i] = (quantize_block_args *) malloc(sizeof(quantize_block_args));

      // 初始化已处理的块数为0，遍历每个块
      int chunks_processed = 0;
      for(long long block_idx = offset*blocksize; block_idx < n; block_idx += blocksize)
      {
          // 计算当前块的有效项数和结束位置
          long long valid_items = n - block_idx >= blocksize ? blocksize : n - block_idx;
          long long block_end = block_idx + valid_items;

          // 获取当前块的参数结构体
          struct quantize_block_args *arg = args[chunks_processed];
          arg->bin_searcher = &bin_searcher;
          arg->code = code;
          arg->A = A;
          arg->absmax = absmax;
          arg->out = out;
          arg->block_end = block_end;
          arg->block_idx = block_idx;
          arg->threadidx = block_idx / blocksize;
          arg->blocksize = blocksize;

          // 根据操作系统选择线程创建方式
#ifdef _WIN32
          new (&threads[chunks_processed]) std::thread(quantize_block, arg);
#else
          pthread_create(&threads[chunks_processed], NULL, &quantize_block, (void *) arg);
#endif
          chunks_processed += 1;
          if(chunks_processed == valid_chunks){ break; }
      }

      // 等待所有线程执行完毕
      for (int i = 0; i < valid_chunks; i++)
      {
#ifdef _WIN32
          threads[i].join();
#else
          int err = pthread_join(threads[i], NULL);
#endif
      }
      // 释放线程数组的内存空间
      free(threads);
      // 释放每个块参数结构体的内存空间
      for (int i = 0; i < valid_chunks; i++)
          free(args[i]);
      // 释放块参数结构体数组的内存空间
      free(args);

    }

}
```