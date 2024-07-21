# `.\pytorch\torch\csrc\jit\runtime\calculate_necessary_args.h`

```py
// 预处理命令，确保头文件只包含一次
#pragma once

// 包含 Torch 的导出头文件
#include <torch/csrc/Export.h>
// 包含 Torch 的模式匹配头文件
#include <torch/csrc/jit/frontend/schema_matching.h>
// 包含标准库头文件
#include <cstddef>

// Torch 的命名空间
namespace torch::jit {

// 计算需要传递的参数数量
// 如果提供了默认值，可能需要更少的参数
// 返回：{所需参数数量，输出参数数量}
inline std::pair<int64_t, int64_t> CalculateNecessaryArgs(
    const std::vector<Argument>& schema_args,  // 输入的模式参数列表
    at::ArrayRef<Value*> actual_inputs,        // 实际输入的值引用数组
    bool allow_trailing_out_args) {            // 是否允许尾部输出参数
  // 如果模式参数列表为空，直接返回0个需要参数和0个输出参数
  if (schema_args.empty()) {
    return std::make_pair(0, 0);
  }

  // 计算输出参数的数量
  int64_t schema_idx = static_cast<int64_t>(schema_args.size()) - 1;
  if (allow_trailing_out_args) {
    // 如果允许尾部输出参数，则跳过末尾的输出参数
    while (schema_idx >= 0) {
      const auto& current_arg = schema_args.at(schema_idx);
      if (!current_arg.is_out()) {
        break;
      }
      schema_idx--;
    }
  }

  // 计算输出参数的数量
  int64_t num_out = static_cast<int64_t>(schema_args.size()) - schema_idx - 1;

  // 如果实际输入参数比模式参数多，则返回实际输入参数的数量和输出参数的数量
  if (schema_args.size() < actual_inputs.size()) {
    return std::make_pair(actual_inputs.size(), num_out);
  }

  // 如果不允许尾部输出参数，则将索引重置到最后一个元素
  if (!allow_trailing_out_args) {
    schema_idx = schema_args.size() - 1;
  }

  // 跟踪不必要的尾部参数
  while (schema_idx >= 0) {
    // 如果没有默认值，则表示这个参数是必需的
    if (!schema_args.at(schema_idx).default_value().has_value()) {
      return std::make_pair(schema_idx + 1, num_out);
    } else {
      auto schema_value =
          schema_args.at(schema_idx).default_value().value().toIValue();
      auto actual_value = toIValue(actual_inputs[schema_idx]);
      // 如果实际值为nullptr，则参数是必需的
      if (!actual_value.has_value()) {
        return std::make_pair(schema_idx + 1, num_out);
      }
      // 如果实际值与模式参数的默认值不相等，则参数是必需的
      if (schema_value != actual_value.value()) {
        return std::make_pair(schema_idx + 1, num_out);
      }
    }
    schema_idx--;
  }
  // 返回所需参数数量为0，输出参数数量为计算得出的值
  return std::make_pair(0, num_out);
}

} // namespace torch::jit
```