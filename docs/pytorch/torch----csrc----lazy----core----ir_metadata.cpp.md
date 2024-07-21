# `.\pytorch\torch\csrc\lazy\core\ir_metadata.cpp`

```
// 引入 Torch 懒加载模块的必要头文件
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <functional>

// Torch 命名空间开始
namespace torch {
namespace lazy {

// 函数定义：输出给定源位置信息的简短框架信息到流中
void EmitShortFrameInfo(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames) {
  // 如果源位置信息非空
  if (!frames.empty()) {
    // 获取第一个源位置信息
    const SourceLocation& frame = frames.front();
    // 查找文件路径中最后一个 '/' 的位置
    std::string::size_type pos = frame.file.find_last_of('/');
    // 如果未找到 '/'
    if (pos == std::string::npos) {
      pos = 0;
    } else {
      ++pos;
    }
    // 输出简短的框架信息到流中，包括函数名、文件名和行号
    stream << ", location=" << frame.function << "@" << frame.file.substr(pos)
           << ":" << frame.line;
  }
}

// 重载运算符<<，用于向流中输出源位置信息向量的详细框架信息
std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames) {
  // 输出框架信息的标题
  stream << "Frames:\n";
  // 遍历每个源位置信息，输出函数名、文件名和行号
  for (auto& location : frames) {
    stream << "  " << location.function << " (" << location.file << ":"
           << location.line << ")\n";
  }
  return stream;
}

// 匿名命名空间开始，用于定义局部的数据结构和函数

// 结构体：作用域条目，包含作用域名和保存的下一个 ID
struct ScopeEntry {
  std::string name;
  size_t saved_next_id = 1;
};

// 结构体：作用域上下文，包含作用域条目的向量和下一个 ID
struct ScopeContext {
  std::vector<ScopeEntry> scopes;
  size_t next_id = 1;
};

// 定义线程局部变量 g_scope_context，用于存储作用域上下文
thread_local ScopeContext g_scope_context;

// 函数：获取当前作用域的完整名字
std::string GetCurrentScope() {
  std::string scope;
  // 遍历 g_scope_context 中的每个作用域条目，构建完整作用域名字
  for (auto& scope_entry : g_scope_context.scopes) {
    if (scope.empty()) {
      scope = scope_entry.name;
    } else {
      scope += "/" + scope_entry.name;
    }
  }
  return scope;
}

// 函数：压入新的作用域到作用域上下文中
void PushScope(const std::string& name) {
  // 获取当前的下一个 ID
  size_t id = g_scope_context.next_id;
  // 将新作用域条目压入作用域上下文的作用域条目向量中
  g_scope_context.scopes.push_back(
      {c10::str(name, ".", id), g_scope_context.next_id + 1});
  // 重置下一个 ID 为 1
  g_scope_context.next_id = 1;
}

// 函数：弹出当前作用域
void PopScope() {
  // 检查作用域条目向量不为空
  TORCH_CHECK(!g_scope_context.scopes.empty());
  // 恢复下一个 ID，并弹出最后一个作用域条目
  g_scope_context.next_id = g_scope_context.scopes.back().saved_next_id;
  g_scope_context.scopes.pop_back();
}

// 函数：重置作用域上下文
void ResetScopeContext() {
  // 如果作用域条目向量不为空，报错并输出当前作用域的完整名字
  if (!g_scope_context.scopes.empty()) {
    TORCH_CHECK(
        false, "Expecting scope to be empty but it is " + GetCurrentScope());
  }
  // 重置下一个 ID 为 1
  g_scope_context.next_id = 1;
}

} // namespace lazy
} // namespace torch
```