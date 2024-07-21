# `.\pytorch\aten\src\ATen\xpu\XPUGeneratorImpl.h`

```py
#pragma once

#include <ATen/core/Generator.h>  // 包含 ATen 库中的 Generator 头文件

namespace at {

struct TORCH_XPU_API XPUGeneratorImpl : public GeneratorImpl {
  // Constructors
  XPUGeneratorImpl(DeviceIndex device_index = -1);  // 构造函数，可以指定设备索引，默认为 -1
  ~XPUGeneratorImpl() override = default;  // 默认析构函数

  // XPUGeneratorImpl methods
  std::shared_ptr<XPUGeneratorImpl> clone() const;  // 克隆当前生成器实例的方法
  void set_current_seed(uint64_t seed) override;  // 设置当前种子值的方法
  void set_offset(uint64_t offset) override;  // 设置偏移量的方法
  uint64_t get_offset() const override;  // 获取当前偏移量的方法
  uint64_t current_seed() const override;  // 获取当前种子值的方法
  uint64_t seed() override;  // 获取默认种子值的方法
  void set_state(const c10::TensorImpl& new_state) override;  // 设置生成器状态的方法
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;  // 获取生成器状态的方法
  void set_philox_offset_per_thread(uint64_t offset);  // 设置每线程 Philox 算法的偏移量的方法
  uint64_t philox_offset_per_thread() const;  // 获取每线程 Philox 算法的偏移量的方法
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);  // 获取 Philox 引擎输入参数的方法
  static c10::DeviceType device_type();  // 获取设备类型的静态方法

 private:
  XPUGeneratorImpl* clone_impl() const override;  // 克隆实现的私有方法
  uint64_t seed_ = default_rng_seed_val;  // 默认种子值的成员变量
  uint64_t philox_offset_per_thread_ = 0;  // 每线程 Philox 算法的偏移量的成员变量
};

namespace xpu::detail {

TORCH_XPU_API const Generator& getDefaultXPUGenerator(DeviceIndex device = -1);  // 获取默认 XPUGenerator 的函数声明

TORCH_XPU_API Generator createXPUGenerator(DeviceIndex device = -1);  // 创建 XPUGenerator 的函数声明

} // namespace xpu::detail
} // namespace at
```