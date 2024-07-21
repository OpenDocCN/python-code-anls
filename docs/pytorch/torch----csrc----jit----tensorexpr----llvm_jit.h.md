# `.\pytorch\torch\csrc\jit\tensorexpr\llvm_jit.h`

```py
#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <c10/macros/Macros.h> // 引入C10宏定义
#include <c10/util/Exception.h> // 引入C10异常处理工具
#include <c10/util/Optional.h> // 引入C10的可选类型工具
#include <torch/csrc/Export.h> // 引入Torch导出宏

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override") // 忽略编译器警告推荐覆盖函数
#include <llvm/ExecutionEngine/JITSymbol.h> // 引入LLVM JIT符号定义
C10_DIAGNOSTIC_POP() // 恢复编译器警告设置
#include <llvm/ExecutionEngine/Orc/Core.h> // 引入LLVM Orc核心功能
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h> // 引入LLVM Orc线程安全模块
#include <llvm/Target/TargetMachine.h> // 引入LLVM目标机器定义

#include <memory> // 引入内存管理工具
#include <string> // 引入字符串处理工具

namespace torch {
namespace jit {
namespace tensorexpr {

inline std::string formatError(llvm::Error&& err, const char* msg) {
  static constexpr const char* defaultErrorMsg =
      "Unexpected failure in LLVM JIT"; // 默认的LLVM JIT失败消息
  std::string errorMsg(msg ? msg : defaultErrorMsg); // 根据传入的消息或默认消息创建错误消息字符串
  llvm::raw_string_ostream ss(errorMsg); // 创建一个llvm::raw_string_ostream对象用于错误消息输出
  ss << ": " << err; // 将错误消息和具体错误信息连接
  return ss.str(); // 返回完整的错误消息字符串
}

template <typename T>
T assertSuccess(llvm::Expected<T> valOrErr, const char* msg = nullptr) {
  TORCH_INTERNAL_ASSERT(valOrErr, formatError(valOrErr.takeError(), msg)); // 断言操作成功，否则输出错误消息
  return std::move(*valOrErr); // 返回操作结果
}

inline void assertSuccess(llvm::Error err, const char* msg = nullptr) {
  TORCH_INTERNAL_ASSERT(!err, formatError(std::move(err), msg)); // 断言操作成功，否则输出错误消息
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace llvm {
namespace orc {

class PytorchLLVMJITImpl;

class TORCH_API PytorchLLVMJIT {
 public:
  PytorchLLVMJIT(
      std::optional<std::string> triple,
      std::optional<std::string> cpu,
      std::optional<std::string> attrs); // 构造函数声明，可选参数为三元组、CPU和属性

  ~PytorchLLVMJIT(); // 析构函数声明

  void addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C); // 添加模块方法声明

  JITSymbol findSymbol(const std::string Name); // 查找符号方法声明

  bool hasSymbol(const std::string& Name); // 判断是否有符号方法声明

  TargetMachine& getTargetMachine(); // 获取目标机器方法声明

  const DataLayout& getDataLayout(); // 获取数据布局方法声明

 private:
  std::unique_ptr<PytorchLLVMJITImpl> impl_; // 使用PImpl idiom隐藏JIT结构的no-rtti部分
};

} // end namespace orc
} // end namespace llvm

#endif // ENABLE LLVM
```