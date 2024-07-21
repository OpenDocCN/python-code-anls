# `.\pytorch\aten\src\ATen\mps\MPSGeneratorImpl.h`

```py
//  Copyright © 2022 Apple Inc. 

#pragma once

// 包含 ATen 库的随机数生成器和 Philox 引擎相关的头文件
#include <ATen/core/Generator.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/util/Optional.h>

// 命名空间 at 内的命名空间 mps::detail
namespace at {
namespace mps::detail {

// 定义 Philox 状态数组大小为 7
static const uint32_t PHILOX_STATE_N = 7;
// 定义 RNG 数据的 POD 类型，包括状态数组和种子
struct rng_data_pod {
  std::array<uint32_t, PHILOX_STATE_N> state{1}; // 初始化 Philox 状态数组
  uint64_t seed = default_rng_seed_val; // 初始化种子值为默认种子值
};

// 声明获取默认 MPS 生成器和创建 MPS 生成器的函数
TORCH_API const Generator& getDefaultMPSGenerator();
TORCH_API Generator createMPSGenerator(uint64_t seed_val = default_rng_seed_val);

} // namespace mps::detail

// 定义 MPSGeneratorImpl 类，继承自 c10::GeneratorImpl
struct TORCH_API MPSGeneratorImpl : public c10::GeneratorImpl {
  // 构造函数，可选初始化种子值为默认种子值
  MPSGeneratorImpl(uint64_t seed_in = default_rng_seed_val);
  ~MPSGeneratorImpl() override = default; // 默认析构函数

  // 克隆当前生成器对象的方法
  std::shared_ptr<MPSGeneratorImpl> clone() const;
  // 设置当前种子值的方法
  void set_current_seed(uint64_t seed) override;
  // 设置偏移量的方法
  void set_offset(uint64_t offset) override;
  // 获取偏移量的方法
  uint64_t get_offset() const override;
  // 获取当前种子值的方法
  uint64_t current_seed() const override;
  // 获取种子值的方法
  uint64_t seed() override;
  // 设置生成器状态的方法，使用新状态张量实现
  void set_state(const c10::TensorImpl& new_state) override;
  // 获取生成器状态的方法，返回状态张量的引用
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  // 更新 Philox 计数器的方法
  void update_philox_counters();

  // 设置 Philox 引擎的方法
  void set_engine(at::Philox4_32 engine) { engine_ = engine; };
  // 获取 Philox 引擎的方法
  at::Philox4_32 engine() { return engine_; };
  // 获取 RNG 数据的状态数组指针的方法
  uint32_t* state_data() { return data_.state.data(); }
  // 静态方法，返回设备类型为 MPS
  static DeviceType device_type() { return DeviceType::MPS; };

private:
  // 成员变量，包含 RNG 数据和 Philox 引擎
  mps::detail::rng_data_pod data_;
  at::Philox4_32 engine_;

  // 克隆生成器对象的实际实现方法
  MPSGeneratorImpl* clone_impl() const override;
};

} // namespace at
```