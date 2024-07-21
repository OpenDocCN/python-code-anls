# `.\pytorch\torch\csrc\jit\backends\coreml\objc\PTMCoreMLTensorSpec.h`

```
// 包含 C10 库中的 ScalarType 类定义
#include <c10/core/ScalarType.h>
// 导入 nlohmann/json 库，用于 JSON 数据处理
#import <nlohmann/json.hpp>

// 包含标准库中的 string 头文件
#include <string>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {
// Mobile 命名空间
namespace mobile {
// CoreML 命名空间
namespace coreml {

// 定义 TensorSpec 结构体，包含名称和数据类型
struct TensorSpec {
  std::string name = "";                   // 张量名称，默认为空字符串
  c10::ScalarType dtype = c10::ScalarType::Float;  // 数据类型，默认为 Float
};

// 定义静态内联函数，根据给定的类型字符串返回对应的 ScalarType 枚举值
static inline c10::ScalarType scalar_type(const std::string& type_string) {
  if (type_string == "0") {
    return c10::ScalarType::Float;         // 类型字符串为 "0"，返回 Float
  } else if (type_string == "1") {
    return c10::ScalarType::Double;        // 类型字符串为 "1"，返回 Double
  } else if (type_string == "2") {
    return c10::ScalarType::Int;           // 类型字符串为 "2"，返回 Int
  } else if (type_string == "3") {
    return c10::ScalarType::Long;          // 类型字符串为 "3"，返回 Long
  }
  return c10::ScalarType::Undefined;       // 默认情况下返回 Undefined
}

} // namespace coreml
} // namespace mobile
} // namespace jit
} // namespace torch
```