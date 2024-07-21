# `.\pytorch\aten\src\ATen\test\cuda_caching_host_allocator_test.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 ATen 库的头文件
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>

// 定义常量 N 为 100
constexpr int64_t N = 100;

// 定义测试用例 CachingHostAllocatorTest.pinned_alias_slice
TEST(CachingHostAllocatorTest, pinned_alias_slice) {
  // 如果 CUDA 不可用，则返回
  if (!at::cuda::is_available()) {
    return;
  }

  // 检查创建标准 pinned 内存的张量是否被正确记录
  auto pinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  ASSERT_TRUE(pinned_tensor.is_pinned());
  ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));

  // 检查使用 from_blob 构造的张量是否正确记录（通过共享的 data_ptr）
  auto alias_tensor = at::from_blob(
      pinned_tensor.data_ptr(), pinned_tensor.sizes(), pinned_tensor.options());
  ASSERT_TRUE(alias_tensor.is_pinned());
  ASSERT_FALSE(
      alias_tensor.storage().data_ptr().get_context() ==
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_EQ(alias_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
      alias_tensor.data_ptr(),
      alias_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));

  // 检查使用切片构造的张量是否正确记录（通过共享的 context）
  auto slice_tensor =
      pinned_tensor.index({at::indexing::Slice(1, at::indexing::None, 2)});
  ASSERT_EQ(
      slice_tensor.storage().data_ptr().get_context(),
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_NE(slice_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
      slice_tensor.data_ptr(),
      slice_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));

  // 检查既没有匹配的 context 也没有匹配的 data_ptr 的张量是否不能被记录
  auto alias_slice_tensor = at::from_blob(
      slice_tensor.data_ptr(), slice_tensor.sizes(), slice_tensor.options());
  ASSERT_TRUE(alias_slice_tensor.is_pinned());
  ASSERT_FALSE(at::cuda::CachingHostAllocator_recordEvent(
      alias_slice_tensor.data_ptr(),
      alias_slice_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));
  ASSERT_NE(
      alias_slice_tensor.storage().data_ptr().get(),
      slice_tensor.storage().data_ptr().get());
}

// 定义测试用例 CachingHostAllocatorTest.check_raw_allocation
TEST(CachingHostAllocatorTest, check_raw_allocation) {
  // 如果 CUDA 不可用，则返回
  if (!at::cuda::is_available()) {
    return;
  }

  // 分配原始内存并检查
  auto data_ptr = at::cuda::getCachingHostAllocator()->allocate(N);

  // 定义 UserDataDeleter 类
  class UserDataDeleter {
   public:
    // 构造函数，接受一个带有自定义释放函数指针的唯一指针
    explicit UserDataDeleter(std::unique_ptr<void, c10::DeleterFnPtr> ptr)
        : ptr_(std::move(ptr)) {}

   private:
    // 私有成员变量，用于存储唯一指针
    std::unique_ptr<void, c10::DeleterFnPtr> ptr_;
  };
}
    // 使用 unique_ptr 管理指向 void 类型数据的指针，并指定自定义的删除器函数
    std::unique_ptr<void, c10::DeleterFnPtr> ptr_;

  };
  
  // 创建 UserDataDeleter 对象的指针，用于管理 data_ptr 的上下文
  auto* user_data_deleter = new UserDataDeleter(data_ptr.move_context());

  // 定义 IOBuf 结构体，用于管理 buf、ctx 和 deleter 函数
  struct IOBuf {
    explicit IOBuf(void* buf, void* ctx, std::function<void(void*)> deleter)
        : buf_(buf), ctx_(ctx), deleter_(std::move(deleter)) {}
    void* buf_;
    void* ctx_;
    std::function<void(void*)> deleter_;
    
    // 析构函数，调用 deleter 函数释放 ctx
    ~IOBuf() {
      deleter_(ctx_);
    }
  };
  
  // 创建 IOBuf 对象的 unique_ptr，管理 data_ptr 的 buf 和 user_data_deleter
  auto iobuf =
      std::make_unique<IOBuf>(data_ptr.get(), user_data_deleter, [](void* ctx) {
        delete static_cast<UserDataDeleter*>(ctx);
      });
      
  // 根据 IOBuf 创建固定在 blob 上的 Tensor
  auto pinned_tensor =
      at::for_blob(iobuf->buf_, {N})
          .context(
              iobuf.release(),  // 释放 IOBuf 对象的所有权
              [](void* ctx) { delete static_cast<IOBuf*>(ctx); })  // 删除 IOBuf 对象
          .make_tensor();
  
  // 断言 Tensor 是固定在内存中的
  ASSERT_TRUE(pinned_tensor.is_pinned());
  
  // 断言使用 CUDA 主机分配器记录事件成功
  ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));
}

// 定义单元测试 CachingHostAllocatorTest.check_unknown_tensor
TEST(CachingHostAllocatorTest, check_unknown_tensor) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) {
    return;
  }

  // 创建一个非固定内存的空张量
  auto unpinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(false));

  // 断言：验证未固定内存张量的事件记录函数返回 false
  ASSERT_FALSE(at::cuda::CachingHostAllocator_recordEvent(
      unpinned_tensor.data_ptr(),
      unpinned_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));
}

// 定义单元测试 CachingHostAllocatorTest.check_empty_cache
TEST(CachingHostAllocatorTest, check_empty_cache) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) {
    return;
  }

  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    // 创建一个固定内存的空张量
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    // 获取张量的数据指针和上下文
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
    // 断言：验证固定内存张量的事件记录函数返回 true
    ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
        ptr, ctx, at::cuda::getCurrentCUDAStream()));
  }

  // 清空缓存
  at::cuda::CachingHostAllocator_emptyCache();
  // 断言：验证清空缓存后，事件记录函数返回 false
  ASSERT_FALSE(at::cuda::CachingHostAllocator_recordEvent(
      ptr, ctx, at::cuda::getCurrentCUDAStream()));
}

// 定义单元测试 CachingHostAllocatorTest.check_reuse
TEST(CachingHostAllocatorTest, check_reuse) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) {
    return;
  }

  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    // 创建一个固定内存的空张量，并获取其数据指针和上下文
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
  }

  // 确保重用内存分配
  {
    // 再次创建一个固定内存的空张量
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    // 断言：验证新创建的张量的数据指针与上下文与之前相同
    ASSERT_EQ(ptr, pinned_tensor.data_ptr());
    ASSERT_EQ(ctx, pinned_tensor.storage().data_ptr().get_context());
  }
}

// 主函数入口，初始化 Google 测试并运行所有测试
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  // 设置随机种子
  at::manual_seed(42);
  // 运行所有测试，并返回结果
  return RUN_ALL_TESTS();
}
```