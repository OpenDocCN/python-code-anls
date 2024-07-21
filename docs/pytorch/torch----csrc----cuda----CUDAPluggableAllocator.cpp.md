# `.\pytorch\torch\csrc\cuda\CUDAPluggableAllocator.cpp`

```
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

// 命名空间 torch::cuda::CUDAPluggableAllocator 的定义和实现
namespace torch::cuda::CUDAPluggableAllocator {

// 设备计数器，初始化为 0
int device_count = 0;

// 自定义的原始内存释放函数声明
void custom_raw_deleter(void* ptr);

// _AllocationMetadata 结构的默认构造函数实现
_AllocationMetadata::_AllocationMetadata()
    : size(0), device_idx(-1), stream{} {}

// _AllocationMetadata 结构的带参数构造函数实现
_AllocationMetadata::_AllocationMetadata(
    size_t size,
    c10::DeviceIndex device_idx,
    cudaStream_t stream)
    : size(size), device_idx(device_idx), stream(stream) {}

// CUDAPluggableAllocator 的构造函数实现，接受两个分配和释放函数的函数指针
// 这样可以在不需要链接 libtorch 的情况下使用基于函数指针的自定义分配器
// 并且可以从 Python 中调用
CUDAPluggableAllocator::CUDAPluggableAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn)
    : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) {}

// CUDAPluggableAllocator 的复制构造函数实现，复制另一个 CUDAPluggableAllocator 对象的所有函数指针
CUDAPluggableAllocator::CUDAPluggableAllocator(CUDAPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      begin_allocate_to_pool_fn_(other.begin_allocate_to_pool_fn_),
      end_allocate_to_pool_fn_(other.end_allocate_to_pool_fn_),
      relase_pool_fn_(other.relase_pool_fn_) {}

// 设置初始化函数的成员函数实现
void CUDAPluggableAllocator::set_init_fn(std::function<void(int)> init_fn) {
  init_fn_ = std::move(init_fn);
}

// 设置重置函数的成员函数实现
void CUDAPluggableAllocator::set_reset_fn(std::function<void()> reset_fn) {
  reset_fn_ = std::move(reset_fn);
}

// 设置内存分配函数的成员函数实现
void CUDAPluggableAllocator::set_memory_fraction_fn(
    std::function<void(double, int)> memory_fraction_fn) {
  memory_fraction_fn_ = std::move(memory_fraction_fn);
}

// 设置基础分配函数的成员函数实现
void CUDAPluggableAllocator::set_base_alloc_fn(
    std::function<void*(void*, size_t*)> base_alloc_fn) {
  base_alloc_fn_ = std::move(base_alloc_fn);
}

// 设置记录流函数的成员函数实现
void CUDAPluggableAllocator::set_record_stream_fn(
    std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn) {
  record_stream_fn_ = std::move(record_stream_fn);
}

// 设置开始分配到池的函数的成员函数实现
void CUDAPluggableAllocator::set_begin_allocate_to_pool(
    std::function<
        void(int, c10::cuda::MempoolId_t, std::function<bool(cudaStream_t)>)>
        capture_begin_fn) {
  begin_allocate_to_pool_fn_ = std::move(capture_begin_fn);
}

// 设置结束分配到池的函数的成员函数实现
void CUDAPluggableAllocator::set_end_allocate_to_pool_fn(
    std::function<void(int, c10::cuda::MempoolId_t)> capture_about_to_end_fn) {
  end_allocate_to_pool_fn_ = std::move(capture_about_to_end_fn);
}

// 设置释放池的函数的成员函数实现
void CUDAPluggableAllocator::set_release_pool(
    std::function<void(int, c10::cuda::MempoolId_t)> capture_destroy_fn) {
  relase_pool_fn_ = std::move(capture_destroy_fn);
}

// 分配内存的成员函数实现，通过调用 alloc_fn_ 函数指针实现
void* CUDAPluggableAllocator::malloc(
    size_t size,
    int device_idx,
    cudaStream_t stream) {
  
    c10::DeviceIndex device,
    cudaStream_t stream) {
  // 调用分配器函数 alloc_fn_，分配指定大小的内存块在指定设备上的指定流中
  void* r = alloc_fn_(size, device, stream);
  {
    // 使用互斥锁锁定 allocator_mutex_，以确保在多线程环境中 allocation_metadata_ 的安全访问
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    // 将分配的内存块的元数据（大小、设备索引、流）添加到 allocation_metadata_ 中
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
  }
  // 返回分配的内存块的指针
  return r;
}
}

// 分配函数，用于在 CUDA 设备上分配内存
c10::DataPtr CUDAPluggableAllocator::allocate(size_t size) {
  // 初始化设备索引
  c10::DeviceIndex device = -1;
  // 获取当前 CUDA 设备
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  // 获取当前 CUDA 流
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  // 调用自定义的分配函数
  void* r = this->malloc(size, device, stream);
  // 创建数据指针对象
  c10::DataPtr data_ptr = {
      r, r, raw_deleter(), c10::Device(c10::DeviceType::CUDA, device)};
  // 返回数据指针
  return data_ptr;
}

// 返回原始删除函数指针
c10::DeleterFnPtr CUDAPluggableAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

// 分配给定大小的内存
void* CUDAPluggableAllocator::raw_alloc(size_t nbytes) {
  // 初始化设备索引
  c10::DeviceIndex device = -1;
  // 获取当前 CUDA 设备
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  // 获取当前 CUDA 流
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  // 调用自定义的分配函数
  return malloc(nbytes, device, stream);
}

// 分配给定大小的内存，并指定 CUDA 流
void* CUDAPluggableAllocator::raw_alloc_with_stream(
    size_t nbytes,
    cudaStream_t stream) {
  // 初始化设备索引
  c10::DeviceIndex device = -1;
  // 获取当前 CUDA 设备
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  // 调用自定义的分配函数
  return malloc(nbytes, device, stream);
}

// 释放给定指针的内存
void CUDAPluggableAllocator::raw_delete(void* ptr) {
  // 初始化 CUDA 流和设备索引
  cudaStream_t stream{};
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    // 使用互斥锁保护的区块
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    // 检查要释放的指针是否在分配元数据中
    TORCH_CHECK(
        allocation_metadata_.count(ptr),
        "Trying to free a pointer not allocated here");
    // 获取指针的分配元数据
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    stream = metadata.stream;
    // 从分配元数据中删除该指针的信息
    allocation_metadata_.erase(ptr);
  }
  // 调用自定义的释放函数
  free_fn_(ptr, size, device_idx, stream);
}

// 初始化分配器
void CUDAPluggableAllocator::init(int device_count) {
  // 如果初始化函数存在，则调用它
  if (init_fn_) {
    init_fn_(device_count);
  }
  // 标记已初始化
  initialized_ = true;
}

// 返回分配器是否已初始化
bool CUDAPluggableAllocator::initialized() {
  return initialized_;
}

// 设置指定设备的内存分配比例
void CUDAPluggableAllocator::setMemoryFraction(
    double fraction,
    c10::DeviceIndex device) {
  // 如果内存分配函数存在，则调用它
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

// 清空缓存
void CUDAPluggableAllocator::emptyCache() {
  // 如果重置函数存在，则调用它
  if (reset_fn_) {
    return reset_fn_();
  }
}

// 获取指定设备的缓存信息
void CUDAPluggableAllocator::cacheInfo(
    c10::DeviceIndex device,
    size_t* largestBlock) {
  // 抛出异常，因为当前不支持缓存信息功能
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

// 获取给定指针基础分配的起始位置和大小
void* CUDAPluggableAllocator::getBaseAllocation(void* ptr, size_t* size) {
  // 如果基础分配函数存在，则调用它
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  } else {
    return ptr;
  }
}

// 记录数据指针与流的关系
void CUDAPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    streamType stream) {
  // 如果记录流函数存在，则调用它
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

// 获取指定设备的设备统计信息
c10::cuda::CUDACachingAllocator::DeviceStats CUDAPluggableAllocator::
    getDeviceStats(c10::DeviceIndex device) {
  // 抛出异常，因为当前不支持设备统计信息功能
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}
// 重置设备上的累积统计数据（未实现）
void CUDAPluggableAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  // 抛出错误，指示CUDAPluggableAllocator尚不支持此功能
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

// 重置设备上的峰值统计数据（未实现）
void CUDAPluggableAllocator::resetPeakStats(c10::DeviceIndex device) {
  // 抛出错误，指示CUDAPluggableAllocator尚不支持此功能
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

// 获取设备上的内存快照信息（未实现）
c10::cuda::CUDACachingAllocator::SnapshotInfo CUDAPluggableAllocator::
    snapshot() {
  // 抛出错误，指示CUDAPluggableAllocator尚不支持此功能
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support snapshot. "
      "If you need it, please file an issue describing your use case.");
}

// 获取与给定IPC句柄相关联的设备指针（未实现）
std::shared_ptr<void> CUDAPluggableAllocator::getIpcDevPtr(std::string handle) {
  // 抛出错误，指示CUDAPluggableAllocator尚不支持此功能
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// 开始分配到内存池中（如果定义了回调函数则调用）
void CUDAPluggableAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id,
    std::function<bool(cudaStream_t)> filter) {
  // 如果定义了回调函数，则调用以开始分配到内存池中
  if (begin_allocate_to_pool_fn_) {
    begin_allocate_to_pool_fn_(device, mempool_id, std::move(filter));
  }
}

// 结束分配到内存池中（如果定义了回调函数则调用）
void CUDAPluggableAllocator::endAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  // 如果定义了回调函数，则调用以结束分配到内存池中
  if (end_allocate_to_pool_fn_) {
    end_allocate_to_pool_fn_(device, mempool_id);
  }
}

// 释放内存池（如果定义了回调函数则调用）
void CUDAPluggableAllocator::releasePool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  // 如果定义了回调函数，则调用以释放内存池
  if (relase_pool_fn_) {
    relase_pool_fn_(device, mempool_id);
  }
}

// 记录历史分配信息（未实现）
void CUDAPluggableAllocator::recordHistory(
    bool enabled,
    c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    c10::cuda::CUDACachingAllocator::RecordContext when) {
  // 抛出错误，指示CUDAPluggableAllocator尚不支持此功能
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

// 附加内存耗尽观察者（未实现）
void CUDAPluggableAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  // 抛出错误，指示CUDAPluggableAllocator尚不支持此功能
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

// 附加分配器追踪跟踪器（未实现）
void CUDAPluggableAllocator::attachAllocatorTraceTracker(
    c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) {
  // 抛出错误，指示CUDAPluggableAllocator不支持此功能
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not support attachAllocatorTraceTracker. "
      "attachAllocatorTraceTracker is only used inside Pytorch.");
}

// 获取设备的检查点状态（未实现）
std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
CUDAPluggableAllocator::getCheckpointState(
    c10::DeviceIndex device,
    # 使用 TORCH_CHECK 断言，验证条件为 false，如果条件为真，程序继续执行；否则抛出异常并显示指定的错误信息
    TORCH_CHECK(
        false,
        "CUDAPluggableAllocator does not yet support getCheckpointState. "
        "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::CheckpointDelta CUDAPluggableAllocator::
    setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) {
  // 检查并抛出异常，因为当前 CUDAPluggableAllocator 不支持设置 checkpoint pool state
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support setCheckpointPoolState. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  // 设置当前设备为 dev
  c10::cuda::CUDAGuard device_guard(dev);
  // 尝试启用设备之间的 peer access
  cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    // 如果已经启用了 peer access，则忽略并清除错误
    (void)cudaGetLastError();
  } else {
    // 抛出错误如果启用 peer access 失败
    C10_CUDA_CHECK(err);
  }
}

cudaError_t CUDAPluggableAllocator::memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cudaStream_t stream,
    bool p2p_enabled) {
  // 使用异步方式在设备之间执行内存拷贝操作
  return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

std::string CUDAPluggableAllocator::name() {
  // 返回当前分配器的名称
  return "pluggable";
}

void CUDAPluggableAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  // 使用 cudaMemcpy 函数在设备之间同步拷贝数据
  C10_CUDA_CHECK(
      cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
    current_custom_allocator;

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
getCurrentAllocator() {
  // 返回当前的自定义 CUDA 分配器
  return current_custom_allocator;
}

// TODO: add more functions in the argument
std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn) {
  // 创建一个新的 CUDAPluggableAllocator 实例，并初始化
  std::shared_ptr<CUDAPluggableAllocator> allocator(
      new CUDAPluggableAllocator(std::move(alloc_fn), std::move(free_fn)));
  allocator->init(device_count);  // 初始化分配器
  return allocator;  // 返回新创建的分配器
}

void changeCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator) {
  // 检查当前分配器是否已初始化，如果是，则抛出异常
  TORCH_CHECK(
      !c10::cuda::CUDACachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  // 将全局 CUDA 分配器替换为传入的 allocator
  c10::cuda::CUDACachingAllocator::allocator.store(allocator.get());
  // 更新当前自定义分配器为传入的 allocator
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ptr) {
  // 调用当前自定义分配器的 raw_delete 方法来释放 ptr
  current_custom_allocator->raw_delete(ptr);
}

} // namespace torch::cuda::CUDAPluggableAllocator
```