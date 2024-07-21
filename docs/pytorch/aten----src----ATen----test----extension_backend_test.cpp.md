# `.\pytorch\aten\src\ATen\test\extension_backend_test.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include <torch/csrc/jit/runtime/operator.h>

// NB. These tests use the MAIA dispatch key to test backend dispatching
// machinery, but these tests are not specific to MAIA at all. The MAIA
// backend is fully out-of-tree, so it's safe to use this key for
// in-tree tests.

using namespace at;

// 静态整型变量，用于测试目的
static int test_int;

// 重写了 `aten::empty.memory_format` 操作的实现
Tensor empty_override(SymIntArrayRef size, std::optional<ScalarType> dtype, std::optional<Layout> layout,
                      std::optional<Device> device, std::optional<bool> pin_memory, std::optional<MemoryFormat> optional_memory_format) {
  // 设置测试整数为1
  test_int = 1;
  // 创建一个新的未初始化的张量实例，使用 MAIA 分发键
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          Storage::use_byte_size_t(),  // 使用字节大小的存储
          0,                           // 存储的字节大小为0，即未分配存储空间
          at::DataPtr(nullptr, Device(DeviceType::MAIA, 1)),  // 数据指针为nullptr，设备类型为 MAIA，索引为1
          nullptr,                     // 空的分析器
          false),                      // 不共享存储
      DispatchKey::MAIA,               // 使用 MAIA 分发键
      caffe2::TypeMeta::Make<float>()); // 数据类型为 float
  // 返回创建的张量实例
  return Tensor(std::move(tensor_impl));
}

// 重写了 `aten::add.Tensor` 操作的实现
Tensor add_override(const Tensor & a, const Tensor & b , const Scalar& c) {
  // 创建一个新的空张量，尺寸为 {5, 5}，使用 MAIA 分发键
  auto out = empty({5, 5}, at::kMAIA);  // 不返回自身
  // 设置测试整数为2
  test_int = 2;
  // 返回创建的张量
  return out;
}

// 重写了 `aten::empty_strided` 操作的实现
Tensor empty_strided_override(
  IntArrayRef size,
  IntArrayRef stride,
  std::optional<c10::ScalarType> dtype,
  std::optional<c10::Layout> layout,
  std::optional<c10::Device> device,
  std::optional<bool> pin_memory) {

  // 调用 `empty_override` 函数来实现 `empty_strided` 操作
  return empty_override(fromIntArrayRefSlow(size), dtype, layout, device, pin_memory, c10::nullopt);
}

// 注册了新的操作实现到 ATen 库，使用 MAIA 分发键
TORCH_LIBRARY_IMPL(aten, MAIA, m) {
  m.impl("aten::empty.memory_format",  empty_override);   // 注册 `aten::empty.memory_format` 操作的实现
  m.impl("aten::empty_strided",        empty_strided_override);  // 注册 `aten::empty_strided` 操作的实现
  m.impl("aten::add.Tensor",           add_override);     // 注册 `aten::add.Tensor` 操作的实现
}

// 测试用例，验证操作的注册是否正确
TEST(BackendExtensionTest, TestRegisterOp) {
  // 创建一个空张量 `a`，使用 MAIA 分发键
  Tensor a = empty({5, 5}, at::kMAIA);
  // 断言张量 `a` 的设备类型为 MAIA
  ASSERT_EQ(a.device().type(), at::kMAIA);
  // 断言张量 `a` 的设备索引为1
  ASSERT_EQ(a.device().index(), 1);
  // 断言张量 `a` 的数据类型为 float
  ASSERT_EQ(a.dtype(), caffe2::TypeMeta::Make<float>());
  // 断言测试整数为1
  ASSERT_EQ(test_int, 1);

  // 创建一个与 `a` 类型相同的空张量 `b`
  Tensor b = empty_like(a, at::kMAIA);
  // 断言张量 `b` 的设备类型为 MAIA
  ASSERT_EQ(b.device().type(), at::kMAIA);
  // 断言张量 `b` 的设备索引为1
  ASSERT_EQ(b.device().index(), 1);
  // 断言张量 `b` 的数据类型为 float
  ASSERT_EQ(b.dtype(), caffe2::TypeMeta::Make<float>());

  // 调用 `add` 操作
  add(a, b);
  // 断言测试整数为2
  ASSERT_EQ(test_int, 2);

  // 确保非 MAIA 操作仍能正常工作
  // 创建一个空张量 `d`，使用 CPU 设备
  Tensor d = empty({5, 5}, at::kCPU);
  // 断言张量 `d` 的设备类型为 CPU
  ASSERT_EQ(d.device().type(), at::kCPU);
}
```