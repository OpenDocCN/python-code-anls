# `.\pytorch\aten\src\ATen\xpu\CachingHostAllocator.cpp`

```
namespace at::xpu {
namespace {

constexpr size_t kHostAlignment = 512;  // 定义主机内存对齐的大小为 512 字节

using Block = HostBlock<XPUStream>;  // 使用 HostBlock<XPUStream> 别名 Block

// XPUCachingHostAllocatorImpl 类继承自 CachingHostAllocatorImpl<XPUStream, XPUEvent>
struct XPUCachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<XPUStream, XPUEvent> {

  // 重写接口函数 allocate_host_memory
  void allocate_host_memory(size_t size, void** ptr) override {
    // 使用 sycl::aligned_alloc_host 分配主机内存，指定对齐方式和设备上下文
    *ptr = sycl::aligned_alloc_host(
        kHostAlignment, size, c10::xpu::get_device_context());
  }

  // 重写接口函数 free_block
  void free_block(Block* block) override {
    // 使用 sycl::free 释放 block 指向的内存，传入设备上下文
    sycl::free(block->ptr_, c10::xpu::get_device_context());
  }

  // 重写接口函数 record_stream
  void record_stream(
      std::optional<std::vector<XPUEvent>>& events,
      XPUStream stream) override {
    // 创建 XPUEvent 对象，并记录流信息
    XPUEvent event;
    event.record(stream);
    // 将 event 添加到 events 向量中
    events->push_back(std::move(event));
  }

  // 重写接口函数 query_event
  bool query_event(XPUEvent& event) override {
    // 查询事件是否完成
    return event.query();
  }
};

// XPUCachingHostAllocator 类继承自 CachingHostAllocatorInterface<XPUCachingHostAllocatorImpl>
struct XPUCachingHostAllocator final
    : public CachingHostAllocatorInterface<XPUCachingHostAllocatorImpl> {

  // 重写接口函数 allocate
  at::DataPtr allocate(size_t size) override {
    // 调用 impl_->allocate(size) 分配内存，并返回 DataPtr 对象
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }
};

// 静态实例 caching_host_allocator，类型为 XPUCachingHostAllocator
static XPUCachingHostAllocator caching_host_allocator;

// 内联函数 getXPUCachingHostAllocator 返回 caching_host_allocator 实例的引用
static inline XPUCachingHostAllocator& getXPUCachingHostAllocator() {
  return caching_host_allocator;
}

// 原始本地删除器函数 raw_local_deleter
void raw_local_deleter(void* ptr) {
  // 调用 getXPUCachingHostAllocator().free(ptr) 释放内存
  getXPUCachingHostAllocator().free(ptr);
}

} // anonymous namespace

// 公共函数 CachingHostAllocator_recordEvent
bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::xpu::XPUStream stream) {
  // 调用 getXPUCachingHostAllocator().record_event(ptr, ctx, stream) 记录事件
  return getXPUCachingHostAllocator().record_event(ptr, ctx, stream);
}

// 公共函数 CachingHostAllocator_emptyCache
void CachingHostAllocator_emptyCache() {
  // 调用 getXPUCachingHostAllocator().empty_cache() 清空缓存
  getXPUCachingHostAllocator().empty_cache();
}

// getCachingHostAllocator 函数返回 caching_host_allocator 实例的地址
at::Allocator* getCachingHostAllocator() {
  return &getXPUCachingHostAllocator();
}

} // namespace at::xpu
```