# `.\pytorch\aten\src\ATen\cuda\CachingHostAllocator.cpp`

```
namespace at::cuda {

// CUDA 主机分配器的缓存实现，继承自 CUDA 流和事件池的缓存主机分配器实现
struct CUDACachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<CUDAStream, EventPool::Event> {
 private:
  // 重写父类方法，分配主机内存
  void allocate_host_memory(size_t size, void** ptr) override {
    // 由于统一地址映射，任何设备分配的页锁定内存都可以直接被其他设备使用，
    // 因此在分配时获取任何现有的主要上下文设备索引。
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::cuda::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      // 在主要上下文设备上进行设备保护
      device_guard.reset_device(
          at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }


这段代码实现了一个针对 CUDA 的主机内存分配器的缓存实现。在分配主机内存时，通过获取主要上下文的设备索引，确保分配的内存可以在任何设备上使用，因为假设存在统一的地址映射。
  if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
          pinned_use_cuda_host_register()) {
    // 如果设置为使用 CUDA 主机注册，则使用 allocWithCudaHostRegister 分配内存
    allocWithCudaHostRegister(ptr, size);
  } else {
    // 否则，使用 cudaHostAlloc 来分配固定内存（驱动中的全局锁）
    C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
  }
}

void free_block(Block* block) override {
  if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
          pinned_use_cuda_host_register()) {
    // 如果设置为使用 CUDA 主机注册，则取消注册内存并释放指针
    void* ptr = block->ptr_;
    AT_CUDA_CHECK(cudaHostUnregister(ptr));
    free(ptr);
  } else {
    // 否则，使用 cudaFreeHost 释放固定内存
    AT_CUDA_CHECK(cudaFreeHost(block->ptr_));
  }
}

void record_stream(
    std::optional<std::vector<EventPool::Event>>& events,
    CUDAStream stream) override {
  // 创建事件并记录到指定的 CUDA 流中，然后将事件移动到事件向量中
  auto event = create_event_internal(stream.device_index());
  event->record(stream);
  events->push_back(std::move(event));
}

bool query_event(EventPool::Event& event) override {
  // 查询 CUDA 事件的状态，如果事件尚未完成，则返回 false
  cudaError_t err = cudaEventQuery(*event);
  if (err == cudaErrorNotReady) {
    (void)cudaGetLastError(); // 清除 CUDA 错误
    return false;
  } else if (err != cudaSuccess) {
    C10_CUDA_CHECK(err);
  }
  return true;
}

EventPool::Event create_event_internal(DeviceIndex idx) {
  // 创建一个 CUDA 事件并返回，利用静态的事件池来避免关闭问题
  static auto* event_pool = new EventPool();
  return event_pool->get(idx);
}

TaskThreadPool* getThreadPool() {
  // 获取任务线程池的单例对象，使用配置中指定的最大注册线程数
  static TaskThreadPool* pool = new TaskThreadPool(
      c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
          pinned_max_register_threads());
  return pool;
}

void mapPagesForRegister(
    const void* ptr,
    size_t size,
    size_t i,
    size_t numThreads,
    size_t pageSize) {
  uintptr_t start = (uintptr_t)ptr + (size * i / numThreads);
  uintptr_t end = (uintptr_t)start + (size / numThreads);
  if (i == (numThreads - 1)) {
    end = (uintptr_t)ptr + size;
  }

  // 预先访问/映射页面，通过设置页面的第一个字节
  uintptr_t alignedStart =
      (((uintptr_t)start + pageSize - 1) & ~(pageSize - 1));
  for (uintptr_t p = alignedStart; p < ((uintptr_t)end); p += pageSize) {
    memset((void*)p, 0, 1);
  }
}

void registerPages(const void* ptr, size_t size) {
  // 使用 cudaHostRegister 注册指定大小的内存
  AT_CUDA_CHECK(
      cudaHostRegister((void*)ptr, (size_t)size, cudaHostRegisterDefault));

  // 检查主机和设备指针是否匹配，如果不匹配，则给出警告并退出
  void* devptr = nullptr;
  AT_CUDA_CHECK(cudaHostGetDevicePointer(&devptr, (void*)ptr, 0));
  TORCH_CHECK(
      (void*)devptr == (void*)ptr,
      "Host and device pointer dont match with cudaHostRegister. "
      "Please dont use this feature by setting "
      "PYTORCH_CUDA_ALLOC_CONF=use_cuda_host_register:False (default)",
      "");
}

void allocWithCudaHostRegister(void** ptr, size_t roundSize) {
  // 此处进行常规分配，预先访问/映射页面，然后执行...
    // 使用 cudaHostRegister 函数将内存页锁定，以便最小化 cuda 全局锁的开销。
    *ptr = malloc(roundSize);

    // 并行映射/注册内存页，以减少总体时间消耗
    size_t pageSize = (1 << 12); // 4kB 页面大小
    size_t numMapThreads = c10::cuda::CUDACachingAllocator::
        CUDAAllocatorConfig::pinned_num_register_threads();
    if ((numMapThreads > 1) && (roundSize >= (pageSize * numMapThreads))) {
      // 使用线程池并行映射页面
      auto* pool = getThreadPool();
      std::vector<std::promise<void>> promises;
      std::vector<std::future<void>> futures;
      promises.reserve(numMapThreads);
      futures.reserve(numMapThreads);

      for (size_t i = 0; i < numMapThreads; i++) {
        promises.emplace_back();
        futures.push_back(promises[i].get_future());
        auto task = [this,
                     i,
                     ptr,
                     roundSize,
                     numMapThreads,
                     pageSize,
                     &promises]() mutable {
          // 调用函数映射页以进行注册
          mapPagesForRegister(
              *ptr,
              roundSize,
              i, // 线程任务 ID
              numMapThreads,
              pageSize);
          // 映射页面完成后设置 promise
          promises[i].set_value();
        };
        pool->run(task);
      }
      // 等待所有映射任务完成
      for (auto& future : futures) {
        future.wait();
      }
    } else {
      // 在同一个线程中映射页面
      mapPagesForRegister(*ptr, roundSize, 0, 1, pageSize);
    }

    // 使用 cudaHostRegister 函数注册映射的页面
    registerPages(*ptr, roundSize);
};

void raw_local_deleter(void* ptr);

// 定义一个名为CUDACachingHostAllocator的结构体，继承自CachingHostAllocatorInterface<CUDACachingHostAllocatorImpl>
struct CUDACachingHostAllocator final
    : public CachingHostAllocatorInterface<CUDACachingHostAllocatorImpl> {
  
  // 重写allocate方法，用于分配内存
  at::DataPtr allocate(size_t size) override {
    // 调用impl_的allocate方法来获取指针和上下文
    auto ptr_and_ctx = impl_->allocate(size);
    // 返回一个DataPtr对象，其中包含分配的指针、上下文、删除器和设备类型（CPU）
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }
};

// 创建一个名为caching_host_allocator的CUDACachingHostAllocator对象
CUDACachingHostAllocator caching_host_allocator;

// 定义一个静态内联函数，返回caching_host_allocator引用
static inline CUDACachingHostAllocator& getCUDACachingHostAllocator() {
  return caching_host_allocator;
}

// 删除器函数的实现，调用getCUDACachingHostAllocator().free(ptr)释放内存
void raw_local_deleter(void* ptr) {
  getCUDACachingHostAllocator().free(ptr);
}

// 匿名命名空间结束

} // anonymous namespace

// 记录事件的函数，调用getCUDACachingHostAllocator().record_event(ptr, ctx, stream)来记录事件
bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    at::cuda::CUDAStream stream) {
  return getCUDACachingHostAllocator().record_event(ptr, ctx, stream);
}

// 清空缓存的函数，调用getCUDACachingHostAllocator().empty_cache()来释放缓存的固定内存分配
void CachingHostAllocator_emptyCache() {
  getCUDACachingHostAllocator().empty_cache();
}

// 获取缓存主机分配器的函数，返回getCUDACachingHostAllocator()的地址
at::Allocator* getCachingHostAllocator() {
  return &getCUDACachingHostAllocator();
}

// 命名空间结束

} // namespace at::cuda
```