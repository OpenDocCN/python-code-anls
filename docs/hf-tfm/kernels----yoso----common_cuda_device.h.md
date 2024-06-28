# `.\kernels\yoso\common_cuda_device.h`

```py
# 包含公共头文件，这里假设 "common.h" 包含了项目中的通用定义和声明
#include "common.h"

# 定义一个模板函数 set_insert，用于向集合中插入元素
template<typename T>
__device__ int set_insert(T *set, int set_size, T value) {
  # 计算值在集合中的插入位置
  int slot = value % set_size;
  int start_slot = slot;
  # 循环尝试插入值，直到成功或者集合已满
  while (true) {
    # 使用原子操作 CAS（Compare and Swap），尝试在集合中插入值
    T prev = atomicCAS(&set[slot], EMPTY_VALUE, value);
    # 如果插入成功或者集合中已经存在相同值，则返回插入位置
    if (prev == EMPTY_VALUE || prev == value) {
      return slot;
    }
    # 如果插入失败，则尝试下一个位置
    slot = (slot + 1) % set_size;
    # 如果回到起始位置，表示集合已满
    if (slot == start_slot) {
      return -1;
    }
  }
  return -1;
}

# 定义一个模板函数 set_lookup，用于在集合中查找元素的位置
template<typename T>
__device__ int set_lookup(T *set, int set_size, T value) {
  # 计算值在集合中的起始位置
  int slot = value % set_size;
  int start_slot = slot;
  # 循环查找值，直到找到或者集合遍历完毕
  while (true) {
    # 如果当前位置的值等于要查找的值，则返回该位置
    if (set[slot] == value) {
      return slot;
    }
    # 否则尝试下一个位置
    slot = (slot + 1) % set_size;
    # 如果回到起始位置，表示值不在集合中
    if (slot == start_slot) {
      return -1;
    }
  }
  return -1;
}

# 定义一个模板函数 init_buffer，用于初始化缓冲区
template<typename T>
__device__ void init_buffer(T init_value, T *buffer, int buffer_size, int num_threads, int thread_id) {
  # 同步所有线程，确保前面的操作已完成
  __syncthreads();
  # 循环初始化缓冲区
  for (int i = 0; i < buffer_size; i = i + num_threads) {
    int offset_idx = i + thread_id;
    # 如果当前线程需要处理有效的索引，则初始化缓冲区值
    if (offset_idx < buffer_size) {
      buffer[offset_idx] = init_value;
    }
  }
  # 再次同步所有线程，确保所有初始化操作已完成
  __syncthreads();
}

# 定义一个模板函数 copy_data，用于从源地址复制数据到目标地址
template<typename T>
__device__ void copy_data(T *src_pt, T *dist_pt, int data_length, int num_threads, int thread_id) {
  # 同步所有线程，确保前面的操作已完成
  __syncthreads();
  # 循环复制数据
  for (int i = 0; i < data_length; i = i + num_threads) {
    int offset_idx = i + thread_id;
    # 如果当前线程需要处理有效的索引，则复制数据
    if (offset_idx < data_length) {
      dist_pt[offset_idx] = src_pt[offset_idx];
    }
  }
  # 再次同步所有线程，确保所有复制操作已完成
  __syncthreads();
}

# 定义一个模板函数 init_buffer_nonblocking，用于非阻塞方式初始化缓冲区
template<typename T>
__device__ void init_buffer_nonblocking(T init_value, T *buffer, int buffer_size, int num_threads, int thread_id) {
  # 循环初始化缓冲区，无需同步线程
  for (int i = 0; i < buffer_size; i = i + num_threads) {
    int offset_idx = i + thread_id;
    # 如果当前线程需要处理有效的索引，则初始化缓冲区值
    if (offset_idx < buffer_size) {
      buffer[offset_idx] = init_value;
    }
  }
}

# 定义一个模板函数 copy_data_nonblocking，用于非阻塞方式从源地址复制数据到目标地址
template<typename T>
__device__ void copy_data_nonblocking(T *src_pt, T *dist_pt, int data_length, int num_threads, int thread_id) {
  # 循环复制数据，无需同步线程
  for (int i = 0; i < data_length; i = i + num_threads) {
    int offset_idx = i + thread_id;
    # 如果当前线程需要处理有效的索引，则复制数据
    if (offset_idx < data_length) {
      dist_pt[offset_idx] = src_pt[offset_idx];
    }
  }
}
```