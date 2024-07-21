# `.\pytorch\aten\src\ATen\core\CheckMemoryFormat.h`

```py
#include <c10/core/TensorOptions.h>  // 包含 c10 库中的 TensorOptions 头文件

namespace c10::impl {

inline std::optional<MemoryFormat>
check_tensor_options_and_extract_memory_format(
    const TensorOptions& options,  // 输入参数 options，表示张量的选项
    std::optional<MemoryFormat> memory_format) {  // 输入参数 memory_format，表示内存格式的可选参数
    
  TORCH_CHECK(
      options.requires_grad_opt() == c10::nullopt ||  // 检查 options 是否未设置梯度选项
      options.requires_grad_opt().value() == false,   // 检查 options 的 requires_grad 是否为 false
      "Operators taking TensorOptions cannot take a TensorOptions with "
      "options.requires_grad set as true. This isn't implemented yet.");  // 错误信息提示

  TORCH_CHECK(
      !(options.has_memory_format() && memory_format.has_value()),  // 检查是否同时设置了 options 的 memory_format 和显式参数 memory_format
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");  // 错误信息提示

  if (memory_format.has_value()) {  // 如果显式参数 memory_format 有值
    return memory_format;  // 返回显式参数的内存格式
  } else {
    return options.memory_format_opt();  // 否则返回 options 中的内存格式选项
  }
}

} // namespace impl namespace c10  // 结束命名空间 impl 和 c10
```