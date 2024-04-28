# `.\transformers\kernels\yoso\common_cuda.h`

```
# 定义每个 CUDA 线程块中的最大线程数
#define MAX_THREADS_PER_BLOCK 1024
# 定义每个 CUDA 线程块中的最佳线程数
#define OPTIMAL_THREADS_PER_BLOCK 256
# 定义每个 CUDA 线程束的大小
#define WARP_SIZE 32
# 定义在 X 方向上的最大块数
#define MAX_NUM_BLOCK_X 2147483647
# 定义在 Y 方向上的最大块数
#define MAX_NUM_BLOCK_Y 65535
# 定义在 Z 方向上的最大块数
#define MAX_NUM_BLOCK_Z 65535
# 定义每个 CUDA 线程块中的最大共享内存
#define MAX_SHARED_MEM_PER_BLOCK 48000
# 定义全掩码
#define FULL_MASK 0xffffffff
```