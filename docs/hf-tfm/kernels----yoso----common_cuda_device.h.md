# `.\transformers\kernels\yoso\common_cuda_device.h`

```py
// 在设备上执行插入操作，将值插入到集合中
template<typename T>
__device__ int set_insert(T *set, int set_size, T value) {
  // 计算值在集合中的插槽位置
  int slot = value % set_size;
  int start_slot = slot;
  // 循环直到找到合适的插槽位置
  while (true) {
    // 原子操作，尝试将值插入到插槽中
    T prev = atomicCAS(&set[slot], EMPTY_VALUE, value);
    // 如果插槽为空或者已经存在相同值，则插入成功
    if (prev == EMPTY_VALUE || prev == value) {
      return slot;
    }
    // 如果插槽被占用，继续查找下一个插槽
    slot = (slot + 1) % set_size;
    // 如果回到起始插槽位置，表示集合已满
    if (slot == start_slot) {
      return -1;
    }
  }
  return -1;
}

// 在设备上执行查找操作，查找集合中是否存在指定值
template<typename T>
__device__ int set_lookup(T *set, int set_size, T value) {
  // 计算值在集合中的插槽位置
  int slot = value % set_size;
  int start_slot = slot;
  // 循环直到找到值或者集合已满
  while (true) {
    // 如果当前插槽中的值等于要查找的值，则返回插槽位置
    if (set[slot] == value) {
      return slot;
    }
    // 继续查找下一个插槽
    slot = (slot + 1) % set_size;
    // 如果回到起始插槽位置，表示集合已满
    if (slot == start_slot) {
      return -1;
    }
  }
  return -1;
}

// 在设备上初始化缓冲区，使用同步线程块
template<typename T>
__device__ void init_buffer(T init_value, T *buffer, int buffer_size, int num_threads, int thread_id) {
  __syncthreads();
  // 循环初始化缓冲区
  for (int i = 0; i < buffer_size; i = i + num_threads) {
    int offset_idx = i + thread_id;
    // 如果当前索引在缓冲区范围内，则初始化为指定值
    if (offset_idx < buffer_size) {
      buffer[offset_idx] = init_value;
    }
  }
  __syncthreads();
}

// 在设备上复制数据，使用同步线程块
template<typename T>
__device__ void copy_data(T *src_pt, T *dist_pt, int data_length, int num_threads, int thread_id) {
  __syncthreads();
  // 循环复制数据
  for (int i = 0; i < data_length; i = i + num_threads) {
    int offset_idx = i + thread_id;
    // 如果当前索引在数据长度范围内，则复制数据
    if (offset_idx < data_length) {
      dist_pt[offset_idx] = src_pt[offset_idx];
    }
  }
  __syncthreads();
}

// 在设备上初始化缓冲区，不使用同步线程块
template<typename T>
__device__ void init_buffer_nonblocking(T init_value, T *buffer, int buffer_size, int num_threads, int thread_id) {
  // 循环初始化缓冲区
  for (int i = 0; i < buffer_size; i = i + num_threads) {
    int offset_idx = i + thread_id;
    // 如果当前索引在缓冲区范围内，则初始化为指定值
    if (offset_idx < buffer_size) {
      buffer[offset_idx] = init_value;
    }
  }
}

// 在设备上复制数据，不使用同步线程块
template<typename T>
__device__ void copy_data_nonblocking(T *src_pt, T *dist_pt, int data_length, int num_threads, int thread_id) {
  // 循环复制数据
  for (int i = 0; i < data_length; i = i + num_threads) {
    int offset_idx = i + thread_id;
    // 如果当前索引在数据长度范围内，则复制数据
    if (offset_idx < data_length) {
      dist_pt[offset_idx] = src_pt[offset_idx];
    }
  }
}
```