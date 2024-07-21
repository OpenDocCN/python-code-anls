# `.\pytorch\aten\src\ATen\core\DeprecatedTypePropertiesRegistry.cpp`

```
#include <ATen/core/DeprecatedTypePropertiesRegistry.h>

#include <ATen/core/DeprecatedTypeProperties.h>
#include <c10/util/irange.h>

namespace at {

// 定义了一个析构器，用于释放 DeprecatedTypeProperties 类型的对象
void DeprecatedTypePropertiesDeleter::operator()(DeprecatedTypeProperties * ptr) {
  delete ptr;
}

// DeprecatedTypePropertiesRegistry 类的构造函数
DeprecatedTypePropertiesRegistry::DeprecatedTypePropertiesRegistry() {
  // 遍历所有 Backend 和 ScalarType 的组合，初始化 registry 成员
  for (const auto b : c10::irange(static_cast<int>(Backend::NumOptions))) {
    for (const auto s : c10::irange(static_cast<int>(ScalarType::NumOptions))) {
      // 使用 std::make_unique 创建一个 DeprecatedTypeProperties 对象，并放入 registry
      registry[b][s] = std::make_unique<DeprecatedTypeProperties>(
              static_cast<Backend>(b),
              static_cast<ScalarType>(s));
    }
  }
}

// 返回指定 Backend 和 ScalarType 对应的 DeprecatedTypeProperties 引用
DeprecatedTypeProperties& DeprecatedTypePropertiesRegistry::getDeprecatedTypeProperties(
    Backend p, ScalarType s) const {
  return *registry[static_cast<int>(p)][static_cast<int>(s)];
}

// 返回全局唯一的 DeprecatedTypePropertiesRegistry 对象的引用
// 注意：如果在具有静态生命周期对象的析构函数中调用 globalContext() 可能会产生不良后果。
DeprecatedTypePropertiesRegistry & globalDeprecatedTypePropertiesRegistry() {
  static DeprecatedTypePropertiesRegistry singleton;  // 声明并初始化静态的单例对象
  return singleton;  // 返回该静态对象的引用
}

}  // namespace at
```