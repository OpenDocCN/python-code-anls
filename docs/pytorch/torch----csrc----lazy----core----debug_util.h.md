# `.\pytorch\torch\csrc\lazy\core\debug_util.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <string>
// 包含标准字符串库，用于处理字符串操作
#include <vector>
// 包含标准向量库，用于处理动态数组操作

#include <torch/csrc/lazy/core/tensor.h>
// 包含 Torch 懒执行模块的张量核心头文件

namespace torch {
namespace lazy {

TORCH_API std::function<std::vector<SourceLocation>()>&
GetPythonFramesFunction();
// 声明一个函数 GetPythonFramesFunction，返回一个函数类型对象，其返回值是存储 SourceLocation 的向量的引用

TORCH_API std::string GetFirstUserFrameInPython();
// 声明一个函数 GetFirstUserFrameInPython，返回一个字符串，表示在 Python 中的第一个用户帧

class TORCH_API DebugUtil {
 public:
  enum GraphFormat {
    kText,
    kDot,
    kBackend,
  };
  // 定义一个枚举类型 GraphFormat，包含三个成员：kText、kDot、kBackend

  static GraphFormat GetDefaultGraphFormat();
  // 声明一个静态成员函数 GetDefaultGraphFormat，返回默认的图形格式类型 GraphFormat

  // 打印当前 Python 帧和 IR 图，其根节点是张量所包含的 IR 值。如果 indices 非空，则选择指定索引的张量来生成图形。
  static std::string GetTensorsGraphInfo(
      c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors,
      const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());
  // 声明一个静态成员函数 GetTensorsGraphInfo，返回一个字符串，展示当前 Python 帧和 IR 图的信息。参数包括张量数组、索引向量和图形格式，默认使用默认的图形格式。

  // 如果环境变量 LTC_SAVE_TENSORS_FILE 设置了合适的输出路径，则保存由 GetTensorsGraphInfo() 返回的报告实例。
  static void SaveTensorsGraphInfo(
      const char* name,
      c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors,
      const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());
  // 声明一个静态成员函数 SaveTensorsGraphInfo，用于保存由 GetTensorsGraphInfo() 生成的张量图信息报告。参数包括文件名、张量数组、索引向量和图形格式，默认使用默认的图形格式。

  static bool ExperimentEnabled(const std::string& name);
  // 声明一个静态成员函数 ExperimentEnabled，检查给定名称的实验是否已启用，返回布尔值。
};

} // namespace lazy
} // namespace torch
```