# `.\pytorch\test\inductor\extension_backends\cpp\extension_device.cpp`

```py
// 包含 C10 库中的 CPU 内存分配和 Allocator 头文件
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>

// 包含 Torch 库中的 Device 相关头文件
#include <torch/csrc/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

// 包含 ATen 库中的 CPU 循环、调度存根、调整大小、空张量等头文件
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

// 静态变量，用于操作计数
static uint64_t op_counter = 0;
// 上次保存的值的静态变量
static uint64_t last_saved_value = 0;

// 注册守卫的命名空间
namespace at {
namespace detail {

// 注册一个空操作设备守卫实现，用于 PrivateUse1 设备类型
C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);

}} // namespace at::detail

// 基本的虚拟加法函数，返回一个空张量
at::Tensor custom_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  // 操作计数加一
  op_counter += 1;
  // 由于这个自定义设备只是用于测试，没有实现内核功能
  return at::empty(self.sizes(), self.options());
}

// 基本的虚拟乘法函数，返回一个空张量
at::Tensor custom_mul_Tensor(const at::Tensor & self, const at::Tensor & other) {
  // 操作计数加一
  op_counter += 1;
  // 由于这个自定义设备只是用于测试，没有实现内核功能
  return at::empty(self.sizes(), self.options());
}

// 基本的虚拟相等函数，只支持 CPU
at::Tensor custom_to_device(
    const at::Tensor & self,
    at::Device device,
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    std::optional<at::MemoryFormat> memory_format) {
  // 检查张量是否在 CPU 上或者目标设备是 PrivateUse1，否则抛出错误
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(device.is_cpu() || device.type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  // 一些基本使用情况的虚拟断言：输入张量与目标类型和布局一致，都是连续的
  TORCH_CHECK(self.scalar_type() == dtype);
  TORCH_CHECK(self.is_contiguous());

  // 操作计数加一
  op_counter += 1;
  // 如果目标设备不是 CPU，返回一个空张量
  if (device != at::DeviceType::CPU) {
    return at::empty(self.sizes(), self.options());
  }

  // 在 CPU 上创建一个新的张量 out，复制数据到目标张量
  auto out = at::empty(self.sizes(), dtype, self.options().layout(), device, false, memory_format);
  memcpy(out.mutable_data_ptr(), self.mutable_data_ptr(), self.nbytes());
  // 由于这个自定义设备只是用于测试，没有实现内核功能
  return out;
}

// 一个用于自定义设备的虚拟分配器，实际上使用 CPU
struct DummyCustomAllocator final : at::Allocator {
  DummyCustomAllocator() = default;
  // 分配内存的实现，返回数据指针
  at::DataPtr allocate(size_t nbytes) override {
    void* data = c10::alloc_cpu(nbytes);
    // 返回一个包含数据指针、删除函数、设备信息的 DataPtr
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  // 返回删除函数指针，用于释放内存
  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  // 复制数据的实现，使用默认的复制函数
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }

  // 删除函数的静态实现，调用 C10 的 CPU 释放函数
  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    c10::free_cpu(ptr);
  }
};
// 注册我们的虚拟分配器
static DummyCustomAllocator global_custom_alloc;
// 使用全局虚拟分配器注册 PrivateUse1 设备类型的分配器
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

// 自定义函数，用于将张量中的所有元素设置为标量值
at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  // 检查张量的设备类型必须是 PrivateUse1
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows dummy device.");
  // 检查张量必须是连续的
  TORCH_CHECK(self.is_contiguous());
  // 检查张量的标量类型必须是 Float
  TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);

  // 增加操作计数器
  op_counter += 1;
  // 获取张量数据的指针，并将所有元素设置为标量值
  auto _data = static_cast<float*>(self.mutable_data_ptr());
  for (size_t idx = 0; idx < self.numel(); idx++) {
    _data[idx] = value.toFloat();
  }

  return self;
}

// 自定义函数，用于从自定义设备到 CPU 或者从 CPU 到自定义设备的拷贝
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  // 检查源张量必须是 CPU 或 PrivateUse1 设备类型
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  // 检查目标张量必须是 CPU 或 PrivateUse1 设备类型
  TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");

  // 对于基本用例的一些虚拟断言：输入的大小、数据类型必须相同，并且都必须是连续的
  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());
  TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

  // 增加操作计数器
  op_counter += 1;
  // 使用 std::memcpy 进行内存拷贝
  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
  return dst;
}

// 自定义函数，根据指定的大小、数据类型等参数创建空张量
at::Tensor custom_empty_memory_format(at::IntArrayRef size,
                                      std::optional<at::ScalarType> dtype,
                                      std::optional<at::Layout> layout,
                                      std::optional<at::Device> device,
                                      std::optional<bool> pin_memory,
                                      std::optional<at::MemoryFormat> memory_format) {
  // 创建一个私有的 DispatchKeySet，包含 PrivateUse1
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  // 调用通用的 empty_generic 函数创建张量
  return at::detail::empty_generic(size,
                                   &global_custom_alloc,
                                   private_use_ks,
                                   c10::dtype_or_default(dtype),
                                   memory_format);
}

// 自定义函数，根据指定的大小、步长、数据类型等参数创建空张量
at::Tensor custom_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<at::ScalarType> dtype_opt, std::optional<at::Layout> layout_opt, std::optional<at::Device> device_opt, std::optional<bool> pin_memory_opt) {
  // 增加操作计数器
  op_counter += 1;

  // 创建一个私有的 DispatchKeySet，包含 PrivateUse1
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  // 获取数据类型或者默认数据类型
  auto dtype = c10::dtype_or_default(dtype_opt);
  // 调用通用的 empty_strided_generic 函数创建张量
  return  at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, dtype);
}

// 这个宏承担了大部分的工作
// 使用 TORCH_LIBRARY_IMPL，你可以为你的后端注册自定义内核
// 对于开放注册，我们将所有的内核注册到 PrivateUse1 DispatchKey
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // 注册自定义的 add.Tensor 操作到 PrivateUse1 设备类型上
  m.impl("add.Tensor", &custom_add_Tensor);
  // 注册自定义的 mul.Tensor 操作到 PrivateUse1 设备类型上
  m.impl("mul.Tensor", &custom_mul_Tensor);
  // 注册自定义的 to.Device 操作到 PrivateUse1 设备类型上
  m.impl("to.Device", &custom_to_device);
  // 注册自定义的 fill_.Scalar 操作到 PrivateUse1 设备类型上
  m.impl("fill_.Scalar", &custom_fill__scalar);
  // 注册自定义的 _copy_from 操作到 PrivateUse1 设备类型上
  m.impl("_copy_from", &custom__copy_from);
  // 注册自定义的 empty.memory_format 操作到 PrivateUse1 设备类型上
  m.impl("empty.memory_format", &custom_empty_memory_format);
  // 注册自定义的 empty_strided 操作到 PrivateUse1 设备类型上
  m.impl("empty_strided", &custom_empty_strided);
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
// 获取自定义设备 PrivateUse1 的对象
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

// 检查是否调用了自定义操作
bool custom_op_called() {
  bool called = false;
  // 如果操作计数器大于上次保存的值，则认为调用了自定义操作
  if (op_counter > last_saved_value) {
    called = true;
    last_saved_value = op_counter;
  }
  return called;
}

// PrivateGeneratorImpl 类继承自 at::CPUGeneratorImpl 类
class PrivateGeneratorImpl : public at::CPUGeneratorImpl {
public:
  // 构造函数，使用指定的设备索引创建 PrivateUse1 设备对象和 DispatchKeySet
  PrivateGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  // 虚析构函数
  ~PrivateGeneratorImpl() override = default;
};

// 注册 PrivateUse1 设备的生成器
at::Generator make_generator_privateuse1(c10::DeviceIndex device_index) {
  return at::make_generator<PrivateGeneratorImpl>(device_index);
}

// 注册 PrivateUse1 设备的生成器函数
void register_generator() {
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
// 使用 pybind11 在 Python 中公开与我们自定义后端对应的自定义设备对象函数 "extension_name.custom_device()"
// 该函数在 C++ 中实现，映射到 PrivateUse1 设备类型。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 在 Python 中注册 custom_device 函数，返回自定义设备对象
    m.def("custom_device", &get_custom_device, "get custom device object");
    // 在 Python 中注册 custom_op_called 函数，检查是否调用了自定义操作
    m.def("custom_op_called", &custom_op_called, "check if our custom function was called");
    // 在 Python 中注册 register_generator 函数，注册用于自定义设备的生成器
    m.def("register_generator", &register_generator, "register generator for custom device");
}
```