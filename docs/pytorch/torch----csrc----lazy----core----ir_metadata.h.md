# `.\pytorch\torch\csrc\lazy\core\ir_metadata.h`

```
#pragma once
// 预处理指令，确保头文件仅被包含一次

#include <c10/macros/Macros.h>
// 包含 c10 库的宏定义

#include <string>
// 包含标准字符串库

#include <vector>
// 包含标准向量库

namespace torch {
namespace lazy {
// 命名空间 torch 下的 lazy 命名空间

struct SourceLocation {
  std::string file;
  std::string function;
  int line = -1;
};
// 定义结构体 SourceLocation，包含文件名、函数名和行号信息

TORCH_API void EmitShortFrameInfo(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames);
// 声明函数 EmitShortFrameInfo，将帧信息输出到流中

TORCH_API std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames);
// 声明重载运算符 <<，用于输出帧信息到流中

// 用户定义的元数据基类，可以附加到 IR 节点上
struct TORCH_API UserMetaData {
  virtual ~UserMetaData() = default;
};
// 定义用户元数据的基类，析构函数为默认实现

struct TORCH_API MetaData {
  std::string scope;
  std::vector<SourceLocation> frame_info;
};
// 定义结构体 MetaData，包含作用域和帧信息向量

// TODO(whc) is this going to be used outside of in IR decompositions?
// RAII 数据结构，作为堆栈变量用于进入新的 IR 作用域。IR 作用域名称将出现在 IR 中，
// 有助于标识单个 IR 节点的来源。
struct TORCH_API ScopePusher {
  explicit ScopePusher(const std::string& name);
  ~ScopePusher();

  static void ResetScopes();
};
// 定义结构体 ScopePusher，实现了进入新 IR 作用域的 RAII 数据结构

TORCH_API MetaData GetMetaDataIfDebugging();
// 声明函数 GetMetaDataIfDebugging，用于在调试时获取元数据

} // namespace lazy
} // namespace torch
// 命名空间结束
```