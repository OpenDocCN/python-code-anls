# `.\pytorch\aten\src\ATen\test\xla_tensor_test.cpp`

```
#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <ATen/test/allocator_clone_test.h>

using namespace at;

// 定义一个函数 XLAFree，用于释放内存，参数为指针 ptr
void XLAFree(void *ptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(ptr);  // 使用 free 函数释放内存
}

// 定义一个函数 XLAMalloc，用于分配内存，返回分配的指针，参数为 size
void* XLAMalloc(ptrdiff_t size) {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  return malloc(size);  // 使用 malloc 函数分配内存并返回指针
}

// 定义结构 XLAAllocator，继承自 at::Allocator 类
struct XLAAllocator final : public at::Allocator {
  // 实现 allocate 方法，用 XLAMalloc 分配内存，返回 DataPtr
  at::DataPtr allocate(size_t size) override {
    auto* ptr = XLAMalloc(size);  // 调用 XLAMalloc 分配内存
    return {ptr, ptr, &XLAFree, at::DeviceType::XLA};  // 返回 DataPtr 对象
  }
  // 实现 raw_deleter 方法，返回 XLAFree 函数指针
  at::DeleterFnPtr raw_deleter() const override {
    return &XLAFree;  // 返回 XLAFree 函数指针
  }
  // 实现 copy_data 方法，调用 default_copy_data 进行数据复制
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);  // 调用 default_copy_data 复制数据
  }
};

// 定义测试案例 XlaTensorTest.TestNoStorage
TEST(XlaTensorTest, TestNoStorage) {
  XLAAllocator allocator;  // 创建 XLAAllocator 实例
  // 创建 TensorImpl 对象 tensor_impl，传入 DispatchKey、数据类型和设备信息
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      DispatchKey::XLA,
      caffe2::TypeMeta::Make<float>(),
      at::Device(DeviceType::XLA, 0));
  // 创建 Tensor 对象 t，并使用 tensor_impl 初始化
  at::Tensor t(std::move(tensor_impl));
  // 断言 t 的设备类型与指定的设备类型相同
  ASSERT_TRUE(t.device() == at::Device(DeviceType::XLA, 0));
}

// 定义测试案例 XlaTensorTest.test_allocator_clone
TEST(XlaTensorTest, test_allocator_clone) {
  // 如果当前环境不支持 XLA，直接返回
  if (!at::hasXLA()) {
    return;
  }
  XLAAllocator allocator;  // 创建 XLAAllocator 实例
  // 调用 test_allocator_clone 函数，传入 allocator 进行测试
  test_allocator_clone(&allocator);
}
```