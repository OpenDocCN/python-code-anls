# `.\pytorch\aten\src\ATen\core\GeneratorForPrivateuseone.cpp`

```py
#include <mutex>
#include <ATen/core/GeneratorForPrivateuseone.h>

namespace at {

// 定义静态互斥量，用于保护全局的生成器函数对象
static std::mutex _generator_mutex_lock;

// 返回一个可选的生成器函数类型的引用，用于私有使用1调度键
std::optional<GeneratorFuncType>& GetGeneratorPrivate() {
  // 定义静态的可选生成器函数类型对象，初始为无值状态
  static std::optional<GeneratorFuncType> generator_privateuse1 = c10::nullopt;
  return generator_privateuse1;
}

// 生成器注册类的构造函数，用于注册私有使用1调度键的生成器函数
_GeneratorRegister::_GeneratorRegister(const GeneratorFuncType& func) {
  // 获取生成器函数对象的互斥量锁
  std::lock_guard<std::mutex> lock(_generator_mutex_lock);
  // 检查是否已经存在注册的生成器函数对象
  TORCH_CHECK(
      !GetGeneratorPrivate().has_value(),
      "Only can register a generator to the PrivateUse1 dispatch key once!");

  // 获取当前的生成器函数对象并注册新的生成器函数
  auto& m_generator = GetGeneratorPrivate();
  m_generator = func;
}

// 获取私有使用1调度键的生成器对象
at::Generator GetGeneratorForPrivateuse1(c10::DeviceIndex device_index) {
  // 检查是否已经注册了私有使用1调度键的生成器函数对象
  TORCH_CHECK(
      GetGeneratorPrivate().has_value(),
      "Please register a generator to the PrivateUse1 dispatch key, \
      using the REGISTER_GENERATOR_PRIVATEUSE1 macro.");

  // 返回私有使用1调度键的生成器函数对象，并传入设备索引参数
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return GetGeneratorPrivate().value()(device_index);
}

} // namespace at
```