# `.\pytorch\test\cpp_extensions\rng_extension.cpp`

```py
// 引入 Torch C++ 扩展所需的头文件
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <memory>

// 使用 Torch 的命名空间
using namespace at;

// 静态变量，记录 TestCPUGenerator 实例的数量
static size_t instance_count = 0;

// 定义 TestCPUGenerator 结构体，继承自 c10::GeneratorImpl
struct TestCPUGenerator : public c10::GeneratorImpl {
  // 构造函数，初始化基类和成员变量 value_
  TestCPUGenerator(uint64_t value) : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(DispatchKey::CustomRNGKeyId)}, value_(value) {
    // 每创建一个实例，增加实例计数
    ++instance_count;
  }
  
  // 析构函数，减少实例计数
  ~TestCPUGenerator() {
    --instance_count;
  }
  
  // 返回 32 位随机数的方法
  uint32_t random() { return static_cast<uint32_t>(value_); }
  
  // 返回 64 位随机数的方法
  uint64_t random64() { return value_; }
  
  // 设置当前种子的方法，抛出未实现异常
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  
  // 设置偏移量的方法，抛出未实现异常
  void set_offset(uint64_t offset) override { throw std::runtime_error("not implemented"); }
  
  // 获取偏移量的方法，抛出未实现异常
  uint64_t get_offset() const override { throw std::runtime_error("not implemented"); }
  
  // 获取当前种子的方法，抛出未实现异常
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  
  // 获取种子的方法，抛出未实现异常
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  
  // 设置状态的方法，抛出未实现异常
  void set_state(const c10::TensorImpl& new_state) override { throw std::runtime_error("not implemented"); }
  
  // 获取状态的方法，抛出未实现异常
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override { throw std::runtime_error("not implemented"); }
  
  // 克隆实例的方法，抛出未实现异常
  TestCPUGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }

  // 静态方法，返回设备类型为 CPU
  static DeviceType device_type() { return DeviceType::CPU; }

  // 成员变量，存储生成器的值
  uint64_t value_;
};

// 修改张量自身的随机值，调用模板函数 random_impl，使用 TestCPUGenerator 生成随机数
Tensor& random_(Tensor& self, std::optional<Generator> generator) {
  return at::native::templates::random_impl<native::templates::cpu::RandomKernel, TestCPUGenerator>(self, generator);
}

// 修改张量范围内的随机值，调用模板函数 random_from_to_impl，使用 TestCPUGenerator 生成随机数
Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, std::optional<Generator> generator) {
  return at::native::templates::random_from_to_impl<native::templates::cpu::RandomFromToKernel, TestCPUGenerator>(self, from, to, generator);
}

// 修改张量到指定上限的随机值，调用 random_from_to 函数
Tensor& random_to(Tensor& self, int64_t to, std::optional<Generator> generator) {
  return random_from_to(self, 0, to, generator);
}

// 创建 TestCPUGenerator 的生成器
Generator createTestCPUGenerator(uint64_t value) {
  return at::make_generator<TestCPUGenerator>(value);
}

// 返回传入生成器的函数，即返回其本身
Generator identity(Generator g) {
  return g;
}

// 返回 TestCPUGenerator 实例的数量
size_t getInstanceCount() {
  return instance_count;
}

// 实现 Torch 库中的 aten 命名空间，使用 CustomRNGKeyId
TORCH_LIBRARY_IMPL(aten, CustomRNGKeyId, m) {
  // 实现 aten::random_.from 方法，调用 random_from_to 函数
  m.impl("aten::random_.from",                 random_from_to);
  // 实现 aten::random_.to 方法，调用 random_to 函数
  m.impl("aten::random_.to",                   random_to);
  // 实现 aten::random_ 方法，调用 random_ 函数
  m.impl("aten::random_",                      random_);
}

// 绑定扩展模块的函数，命名为 TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 绑定 createTestCPUGenerator 函数
  m.def("createTestCPUGenerator", &createTestCPUGenerator);
  // 绑定 getInstanceCount 函数
  m.def("getInstanceCount", &getInstanceCount);
  // 绑定 identity 函数
  m.def("identity", &identity);
}
```