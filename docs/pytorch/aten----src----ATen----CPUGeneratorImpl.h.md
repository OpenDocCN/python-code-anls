# `.\pytorch\aten\src\ATen\CPUGeneratorImpl.h`

```
#pragma once
// 预处理指令，确保此头文件只被编译一次

#include <ATen/core/Generator.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/util/Optional.h>
// 包含必要的头文件

namespace at {

struct TORCH_API CPUGeneratorImpl : public c10::GeneratorImpl {
  // CPUGeneratorImpl 结构体，继承自 c10::GeneratorImpl

  // Constructors
  CPUGeneratorImpl(uint64_t seed_in = default_rng_seed_val);
  // 构造函数，初始化 CPUGeneratorImpl 对象，可选地使用指定的种子值

  ~CPUGeneratorImpl() override = default;
  // 析构函数，用于对象销毁时的清理工作，采用默认实现

  // CPUGeneratorImpl methods
  std::shared_ptr<CPUGeneratorImpl> clone() const;
  // 克隆当前生成器对象，返回其共享指针

  void set_current_seed(uint64_t seed) override;
  // 设置当前种子值

  void set_offset(uint64_t offset) override;
  // 设置偏移量

  uint64_t get_offset() const override;
  // 获取当前偏移量

  uint64_t current_seed() const override;
  // 获取当前种子值

  uint64_t seed() override;
  // 生成一个新的种子值

  void set_state(const c10::TensorImpl& new_state) override;
  // 设置状态，使用新的张量实现对象

  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  // 获取当前状态的张量实现对象

  static c10::DeviceType device_type();
  // 静态方法，返回设备类型

  uint32_t random();
  // 生成一个 32 位随机数

  uint64_t random64();
  // 生成一个 64 位随机数

  std::optional<float> next_float_normal_sample();
  // 获取下一个浮点型正态分布样本，可能返回空值

  std::optional<double> next_double_normal_sample();
  // 获取下一个双精度浮点型正态分布样本，可能返回空值

  void set_next_float_normal_sample(std::optional<float> randn);
  // 设置下一个浮点型正态分布样本值

  void set_next_double_normal_sample(std::optional<double> randn);
  // 设置下一个双精度浮点型正态分布样本值

  at::mt19937 engine();
  // 获取随机数生成引擎对象

  void set_engine(at::mt19937 engine);
  // 设置随机数生成引擎对象

 private:
  CPUGeneratorImpl* clone_impl() const override;
  // 私有方法，实现对象的克隆

  at::mt19937 engine_;
  // 随机数生成引擎对象

  std::optional<float> next_float_normal_sample_;
  // 可选的下一个浮点型正态分布样本值

  std::optional<double> next_double_normal_sample_;
  // 可选的下一个双精度浮点型正态分布样本值
};

namespace detail {

TORCH_API const Generator& getDefaultCPUGenerator();
// 声明获取默认 CPU 生成器的函数

TORCH_API Generator
createCPUGenerator(uint64_t seed_val = default_rng_seed_val);
// 声明创建 CPU 生成器的函数，可选地使用指定的种子值

} // namespace detail

} // namespace at
// 命名空间 at 结束
```