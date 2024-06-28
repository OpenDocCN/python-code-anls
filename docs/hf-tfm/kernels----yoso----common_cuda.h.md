# `.\kernels\yoso\common_cuda.h`

```
# 定义每个线程块中的最大线程数
#define MAX_THREADS_PER_BLOCK 1024

# 定义优化后的推荐线程块中的线程数
#define OPTIMAL_THREADS_PER_BLOCK 256

# 定义线程束（warp）的大小
#define WARP_SIZE 32

# 定义在X方向上的最大线程块数量
#define MAX_NUM_BLOCK_X 2147483647

# 定义在Y方向上的最大线程块数量
#define MAX_NUM_BLOCK_Y 65535

# 定义在Z方向上的最大线程块数量
#define MAX_NUM_BLOCK_Z 65535

# 定义每个线程块可用的最大共享内存量
#define MAX_SHARED_MEM_PER_BLOCK 48000

# 定义一个掩码，包含所有位都是1
#define FULL_MASK 0xffffffff
```