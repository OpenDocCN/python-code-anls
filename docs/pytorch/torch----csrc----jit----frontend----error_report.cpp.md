# `.\pytorch\torch\csrc\jit\frontend\error_report.cpp`

```py
// 包含 Torch 的 JIT 前端错误报告模块的头文件
#include <torch/csrc/jit/frontend/error_report.h>

// 包含 Torch 的 JIT 前端树结构的头文件
#include <torch/csrc/jit/frontend/tree.h>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 对于移动端构建，避免在 thread_local 中存储带析构函数的对象
#ifndef C10_MOBILE
// 定义线程局部变量，用于存储调用堆栈
thread_local std::vector<Call> calls;
#endif // C10_MOBILE

// ErrorReport 类的拷贝构造函数
ErrorReport::ErrorReport(const ErrorReport& e)
    : ss(e.ss.str()),                        // 拷贝错误消息的字符串流
      context(e.context),                    // 拷贝错误上下文
      the_message(e.the_message),            // 拷贝错误消息
      error_stack(e.error_stack.begin(), e.error_stack.end()) {}  // 拷贝错误调用堆栈

// 仅在非移动端构建时定义的构造函数，根据源代码范围初始化错误报告
#ifndef C10_MOBILE
ErrorReport::ErrorReport(SourceRange r)
    : context(std::move(r)),                // 移动构造源代码范围
      error_stack(calls.begin(), calls.end()) {}  // 拷贝调用堆栈信息

// 更新调用堆栈中最后调用的范围信息
void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {
  calls.back().caller_range = range;        // 更新最后一个调用的范围信息
}

// 调用堆栈的构造函数，记录调用信息和范围
ErrorReport::CallStack::CallStack(
    const std::string& name,
    const SourceRange& range) {
  calls.push_back({name, range});           // 将调用信息和范围压入堆栈
}

// 调用堆栈的析构函数，从堆栈中弹出最后一个调用信息
ErrorReport::CallStack::~CallStack() {
  calls.pop_back();                         // 弹出最后一个调用信息
}
#else // defined C10_MOBILE
// 移动端构建时定义的构造函数，仅根据源代码范围初始化错误报告
ErrorReport::ErrorReport(SourceRange r) : context(std::move(r)) {}

// 在移动端构建时，空实现调用堆栈更新函数
void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {}

// 在移动端构建时，空实现调用堆栈构造函数
ErrorReport::CallStack::CallStack(
    const std::string& name,
    const SourceRange& range) {}

// 在移动端构建时，空实现调用堆栈析构函数
ErrorReport::CallStack::~CallStack() {}
#endif // C10_MOBILE

// 静态函数，返回堆栈错误信息的字符串表示
static std::string get_stacked_errors(const std::vector<Call>& error_stack) {
  std::stringstream msg;

  // 如果错误堆栈不为空，则遍历错误堆栈
  if (!error_stack.empty()) {
    for (auto it = error_stack.rbegin(); it != error_stack.rend() - 1; ++it) {
      auto callee = it + 1;

      // 输出调用信息及其调用者信息的字符串表示
      msg << "'" << it->fn_name
          << "' is being compiled since it was called from '" << callee->fn_name
          << "'\n";
      
      // 在消息中突出显示调用者的代码范围
      callee->caller_range.highlight(msg);
    }
  }
  return msg.str();                         // 返回生成的错误信息字符串
}

// 返回当前调用堆栈的字符串表示
std::string ErrorReport::current_call_stack() {
#ifndef C10_MOBILE
  return get_stacked_errors(calls);         // 返回非移动端构建的堆栈错误信息
#else
  AT_ERROR("Call stack not supported on mobile");  // 移动端构建时抛出错误，不支持堆栈信息
#endif // C10_MOBILE
}

// 返回错误报告的字符串表示
const char* ErrorReport::what() const noexcept {
  std::stringstream msg;
  
  // 输出错误消息的字符串流内容
  msg << "\n" << ss.str();
  msg << ":\n";
  
  // 在消息中突出显示错误的源代码范围
  context.highlight(msg);
  
  // 输出堆栈错误信息的字符串表示
  msg << get_stacked_errors(error_stack);
  
  // 将生成的完整错误信息字符串存储在 the_message 中并返回其 C 风格字符串表示
  the_message = msg.str();
  return the_message.c_str();
}

} // namespace torch::jit  // 结束 torch::jit 命名空间
```