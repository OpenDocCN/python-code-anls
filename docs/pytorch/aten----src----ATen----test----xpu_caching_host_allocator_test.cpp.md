# `.\pytorch\aten\src\ATen\test\xpu_caching_host_allocator_test.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <ATen/ATen.h>  // 包含 PyTorch ATen 库的头文件
#include <ATen/TensorIndexing.h>  // 包含 ATen 的张量索引头文件
#include <ATen/xpu/CachingHostAllocator.h>  // 包含 ATen 的主机内存缓存分配器头文件
#include <ATen/xpu/XPUContext.h>  // 包含 ATen 的 XPU 上下文头文件
#include <ATen/xpu/XPUEvent.h>  // 包含 ATen 的 XPU 事件头文件
#include <c10/core/ScalarType.h>  // 包含 c10 库的标量类型头文件
#include <c10/xpu/XPUStream.h>  // 包含 c10 的 XPU 流头文件

constexpr int64_t N = 100;  // 定义常量 N，值为 100

TEST(CachingHostAllocatorTest, testPinnedAliasSlice) {  // 定义测试案例 CachingHostAllocatorTest.testPinnedAliasSlice
  if (!at::xpu::is_available()) {  // 如果 XPU 不可用，则返回
    return;
  }

  // 检查标准的 pinned tensor 是否可以正确记录
  auto pinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  // TODO: 当 op `pin_memory` 在 XPU 上支持时取消注释此行
  // ASSERT_TRUE(pinned_tensor.is_pinned());
  ASSERT_TRUE(at::xpu::CachingHostAllocator_recordEvent(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream()));

  // 检查通过 from_blob 构造的张量是否可以正确记录（通过共享的 data_ptr）
  auto alias_tensor = at::from_blob(
      pinned_tensor.data_ptr(), pinned_tensor.sizes(), pinned_tensor.options());
  // ASSERT_TRUE(alias_tensor.is_pinned());

  ASSERT_FALSE(
      alias_tensor.storage().data_ptr().get_context() ==
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_EQ(alias_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::xpu::CachingHostAllocator_recordEvent(
      alias_tensor.data_ptr(),
      alias_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream()));

  // 检查通过切片构造的张量是否可以正确记录（通过共享的 context）
  auto slice_tensor =
      pinned_tensor.index({at::indexing::Slice(1, at::indexing::None, 2)});
  ASSERT_EQ(
      slice_tensor.storage().data_ptr().get_context(),
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_NE(slice_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::xpu::CachingHostAllocator_recordEvent(
      slice_tensor.data_ptr(),
      slice_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream()));

  // 检查既没有匹配的 context 也没有匹配的 data_ptr 的张量无法记录
  auto alias_slice_tensor = at::from_blob(
      slice_tensor.data_ptr(), slice_tensor.sizes(), slice_tensor.options());
  // ASSERT_TRUE(alias_slice_tensor.is_pinned());
  ASSERT_FALSE(at::xpu::CachingHostAllocator_recordEvent(
      alias_slice_tensor.data_ptr(),
      alias_slice_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream()));
  ASSERT_NE(
      alias_slice_tensor.storage().data_ptr().get(),
      slice_tensor.storage().data_ptr().get());
}

TEST(CachingHostAllocatorTest, testRawAllocation) {  // 定义测试案例 CachingHostAllocatorTest.testRawAllocation
  if (!at::xpu::is_available()) {  // 如果 XPU 不可用，则返回
    return;
  }

  auto data_ptr = at::xpu::getCachingHostAllocator()->allocate(N);  // 分配大小为 N 的内存

  class UserDataDeleter {  // 定义一个内部类 UserDataDeleter
   public:
    explicit UserDataDeleter(std::unique_ptr<void, c10::DeleterFnPtr> ptr)  // 构造函数，接受一个唯一指针和删除函数指针
        : ptr_(std::move(ptr)) {}  // 初始化成员变量 ptr_

   private:
    std::unique_ptr<void, c10::DeleterFnPtr> ptr_;  // 唯一指针成员变量
  };
}
    std::unique_ptr<void, c10::DeleterFnPtr> ptr_;

# 定义了一个名为 ptr_ 的智能指针，用于管理一个无具体类型的指针，其删除器类型为 c10::DeleterFnPtr。

  };

# 结构体定义结束。

  auto* user_data_deleter = new UserDataDeleter(data_ptr.move_context());

# 创建了一个名为 user_data_deleter 的指针，指向一个通过 data_ptr.move_context() 创建的 UserDataDeleter 对象。

  struct IOBuf {
    explicit IOBuf(void* buf, void* ctx, std::function<void(void*)> deleter)
        : buf_(buf), ctx_(ctx), deleter_(std::move(deleter)) {}

# 定义了一个名为 IOBuf 的结构体，包含了一个构造函数，接受 buf、ctx 和 deleter 作为参数，并初始化成员变量 buf_、ctx_ 和 deleter_。

    void* buf_;
    void* ctx_;
    std::function<void(void*)> deleter_;

# IOBuf 结构体包含了三个公共成员变量，分别是 buf_ 和 ctx_，均为 void* 类型，以及 deleter_，为 std::function<void(void*)> 类型。

    ~IOBuf() {
      deleter_(ctx_);
    }

# 定义了 IOBuf 结构体的析构函数，用于释放资源，调用 deleter_ 函数，并传递 ctx_ 作为参数。

  };

# 结构体定义结束。

  auto iobuf =
      std::make_unique<IOBuf>(data_ptr.get(), user_data_deleter, [](void* ctx) {
        delete static_cast<UserDataDeleter*>(ctx);
      });

# 创建了一个名为 iobuf 的 std::unique_ptr，用于管理一个 IOBuf 对象，通过 std::make_unique 初始化，并传入 data_ptr.get()、user_data_deleter 和一个 lambda 函数作为参数。

  auto pinned_tensor =
      at::for_blob(iobuf->buf_, {N})
          .context(
              iobuf.release(),
              [](void* ctx) { delete static_cast<IOBuf*>(ctx); })
          .make_tensor();

# 创建了一个名为 pinned_tensor 的变量，通过 at::for_blob 创建一个张量，使用 iobuf->buf_ 作为数据源，设置了一个大小为 {N} 的张量。

  // ASSERT_TRUE(pinned_tensor.is_pinned());

# 断言确认 pinned_tensor 是否已被固定在内存中，但是被注释掉了，不会被执行。

  ASSERT_TRUE(at::xpu::CachingHostAllocator_recordEvent(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream()));

# 断言确认调用 at::xpu::CachingHostAllocator_recordEvent 函数，传入 pinned_tensor 的数据指针、存储的数据指针上下文和当前的 XPU 流。
}

TEST(CachingHostAllocatorTest, testUnknownTensor) {
  // 检查是否支持 XPU，如果不支持则直接返回
  if (!at::xpu::is_available()) {
    return;
  }

  // 创建一个未固定内存的空张量
  auto unpinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(false));

  // 断言：记录未固定内存张量的事件是否返回 false
  ASSERT_FALSE(at::xpu::CachingHostAllocator_recordEvent(
      unpinned_tensor.data_ptr(),
      unpinned_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream()));
}

TEST(CachingHostAllocatorTest, testEmptyCache) {
  // 检查是否支持 XPU，如果不支持则直接返回
  if (!at::xpu::is_available()) {
    return;
  }

  // 初始化指针为 nullptr
  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    // 创建一个固定内存的空张量
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
    // 断言：记录固定内存张量的事件是否返回 true
    ASSERT_TRUE(at::xpu::CachingHostAllocator_recordEvent(
        ptr, ctx, at::xpu::getCurrentXPUStream()));
  }

  {
    // 创建一个固定内存的空张量
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    // 在当前 XPU 流上同步
    at::xpu::syncStreamsOnDevice();
  }

  // 清空缓存
  at::xpu::CachingHostAllocator_emptyCache();
  // 断言：记录缓存清空后，固定内存张量的事件是否返回 false
  ASSERT_FALSE(at::xpu::CachingHostAllocator_recordEvent(
      ptr, ctx, at::xpu::getCurrentXPUStream()));
}

TEST(CachingHostAllocatorTest, testReuse) {
  // 检查是否支持 XPU，如果不支持则直接返回
  if (!at::xpu::is_available()) {
    return;
  }

  // 初始化指针为 nullptr
  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    // 创建一个固定内存的空张量
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
  }
  // 确保重用分配
  {
    // 创建一个固定内存的空张量
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    // 断言：检查分配的指针和上下文是否相同
    ASSERT_EQ(ptr, pinned_tensor.data_ptr());
    ASSERT_EQ(ctx, pinned_tensor.storage().data_ptr().get_context());
  }
}

// 初始化 Google 测试框架并运行所有测试用例
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```