# `.\pytorch\aten\src\ATen\core\DeprecatedTypePropertiesRegistry.h`

```py
#pragma once

// 为了保留 BC（兼容性），我们使 DeprecatedTypeProperties 实例像 Type 一样唯一。

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <memory>

namespace at {

// 声明 DeprecatedTypeProperties 类
class DeprecatedTypeProperties;

// 定义用于释放 DeprecatedTypeProperties 实例的删除器结构体
struct TORCH_API DeprecatedTypePropertiesDeleter {
  void operator()(DeprecatedTypeProperties * ptr);
};

// DeprecatedTypePropertiesRegistry 类声明
class TORCH_API DeprecatedTypePropertiesRegistry {
 public:
  // 构造函数，用于初始化 DeprecatedTypePropertiesRegistry
  DeprecatedTypePropertiesRegistry();

  // 获取给定 Backend 和 ScalarType 的 DeprecatedTypeProperties 实例的引用
  DeprecatedTypeProperties& getDeprecatedTypeProperties(Backend p, ScalarType s) const;

 private:
  // NOLINTNEXTLINE(*c-array*)
  // 用于存储 DeprecatedTypeProperties 实例的二维数组，大小为 Backend 和 ScalarType 的选项数
  std::unique_ptr<DeprecatedTypeProperties> registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
};

// 返回全局的 DeprecatedTypePropertiesRegistry 实例的引用
TORCH_API DeprecatedTypePropertiesRegistry& globalDeprecatedTypePropertiesRegistry();

} // namespace at
```