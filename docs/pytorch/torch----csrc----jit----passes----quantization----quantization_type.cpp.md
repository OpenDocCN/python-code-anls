# `.\pytorch\torch\csrc\jit\passes\quantization\quantization_type.cpp`

```py
#include <torch/csrc/jit/passes/quantization/quantization_type.h>  // 包含量化类型定义的头文件

namespace torch {
namespace jit {

std::ostream& operator<<(std::ostream& os, QuantType t) {  // 定义输出流操作符重载函数，输出量化类型
  switch (t) {
    case QuantType::DYNAMIC:  // 如果是动态量化类型
      os << "dynamic";  // 输出 "dynamic"
      break;
    case QuantType::STATIC:  // 如果是静态量化类型
      os << "static";  // 输出 "static"
      break;
    default:
      os.setstate(std::ios_base::failbit);  // 其它情况下，设置输出流状态为失败
  }
  return os;  // 返回输出流
}

} // namespace jit
} // namespace torch
```