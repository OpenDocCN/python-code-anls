# `.\pytorch\torch\csrc\jit\tensorexpr\llvm_codegen.h`

```
#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <torch/csrc/Export.h>

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <c10/util/Optional.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

// LLVM code generation implementation for TensorExpr
class LLVMCodeGen : public CodeGen {
 public:
  // Constructor for LLVM code generation with specified parameters
  explicit LLVMCodeGen(
      StmtPtr stmt,  // Pointer to the statement to be compiled
      const std::vector<BufferArg>& args,  // Arguments passed to the kernel
      at::Device device = at::kCPU,  // Device type (default: CPU)
      const std::string& kernel_func_name = "func",  // Kernel function name
      Dtype dtype = kInt,  // Data type (default: int)
      std::optional<std::string> triple = c10::nullopt,  // LLVM target triple
      std::optional<std::string> cpu = c10::nullopt,  // CPU architecture
      std::optional<std::string> attrs = c10::nullopt);  // Additional attributes

  // Constructor with only the statement pointer (default parameters used)
  explicit LLVMCodeGen(StmtPtr stmt);

  LLVMCodeGen() = delete;  // Default constructor deleted
  ~LLVMCodeGen() override;  // Destructor

  // Clean up memory used during LLVM code generation, except for the kernel
  void cleanup_memory();

  // Execute the generated LLVM kernel with arguments
  TORCH_API void call(const std::vector<CallArg>& args) override;

  // Execute the generated LLVM kernel with raw void pointers as arguments
  TORCH_API void call_raw(const std::vector<void*>& args) override;

  // Execute the generated LLVM kernel with arguments and a specified numel
  TORCH_API void call_with_numel(void** args, int64_t numel) override;

  // Create an empty strided tensor with optional attributes
  at::Tensor empty_strided(
      c10::IntArrayRef size,
      c10::IntArrayRef stride,
      std::optional<c10::ScalarType> dtype_opt,
      std::optional<c10::Layout> layout_opt,
      std::optional<c10::Device> device_opt,
      std::optional<bool> pin_memory_opt) override;

  // Get the generated LLVM code as text with optional attributes
  std::string getCodeText(const std::string& attr = "") override;

  // Template function to retrieve a typed value from the LLVM kernel
  template <typename T>
  T value() {
    return value<T>(nullptr);
  }

  // Template function to retrieve a typed value from the LLVM kernel with arguments
  template <typename T>
  T value(std::vector<void*>& args) {
    return value<T>(args.data());
  }

  // Template function to retrieve a typed value from the LLVM kernel with void pointers
  template <typename T>
  T value(void** args) {
    T (*fp)(void**) = (T(*)(void**))getKernelAddress(callee_.get());
    T rv = fp(args);
    return rv;
  }

 private:
  // Get the address of the kernel function from the LLVM code generation callee
  void* getKernelAddress(LLVMCodeGenCallee* callee);

  std::unique_ptr<LLVMCodeGenCallee> callee_;  // LLVM code generation callee
  std::unique_ptr<LLVMCodeGenImpl> impl_;  // LLVM code generation implementation
};

// Builder class for constructing LLVM code generators
struct TORCH_API LLVMCodeGenBuilder {
  using BufferArg = CodeGen::BufferArg;

  // Constructor initializing with a statement pointer and arguments
  LLVMCodeGenBuilder(StmtPtr stmt, std::vector<BufferArg> args)
      : stmt_(stmt), args_(std::move(args)) {}

  // Set the device type for the LLVM code generator
  LLVMCodeGenBuilder& device(at::Device device) {
    device_ = device;
    return *this;
  }

  // Set the kernel function name for the LLVM code generator
  LLVMCodeGenBuilder& kernelFuncName(std::string name) {
    kernelFuncName_ = std::move(name);
    return *this;
  }

  // Set the data type for the LLVM code generator
  LLVMCodeGenBuilder& dtype(Dtype d) {
    dtype_ = d;
    return *this;
  }

  // Set the LLVM target triple for the LLVM code generator
  LLVMCodeGenBuilder& triple(std::string triple) {
    triple_ = std::move(triple);
    return *this;
  }

  // Set the CPU architecture for the LLVM code generator
  LLVMCodeGenBuilder& cpu(std::string cpu) {
    cpu_ = std::move(cpu);
    return *this;
  }

  // Set additional attributes for the LLVM code generator
  LLVMCodeGenBuilder& attrs(std::string attrs) {
    attrs_ = std::move(attrs);
    return *this;
  }

  StmtPtr stmt_;  // Pointer to the statement to be compiled
  std::vector<BufferArg> args_;  // Arguments passed to the kernel
  at::Device device_ = at::kCPU;  // Default device type (CPU)
  Dtype dtype_ = kInt;  // Default data type (int)
  std::string kernelFuncName_ = "func";  // Default kernel function name
  std::string triple_;  // LLVM target triple
  std::string cpu_;  // CPU architecture
  std::string attrs_;  // Additional attributes
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch

#endif // TORCH_ENABLE_LLVM
    // 返回当前对象的引用，用于支持链式调用
    return *this;
  }

  // 设置属性字符串并返回当前对象的引用，支持链式调用
  LLVMCodeGenBuilder& attrs(std::string attrs) {
    attrs_ = std::move(attrs);
    return *this;
  }

  // 构建并返回一个指向 LLVMCodeGen 对象的 unique_ptr
  std::unique_ptr<LLVMCodeGen> build() {
    return std::make_unique<LLVMCodeGen>(
        stmt_, args_, device_, kernelFuncName_, dtype_, triple_, cpu_, attrs_);
  }

 private:
  // 成员变量声明
  StmtPtr stmt_;
  std::vector<BufferArg> args_;
  at::Device device_ = at::kCPU;
  std::string kernelFuncName_ = "func";
  Dtype dtype_ = kInt;
  std::optional<std::string> triple_ = c10::nullopt;
  std::optional<std::string> cpu_ = c10::nullopt;
  std::optional<std::string> attrs_ = c10::nullopt;
};

// 结束 namespace torch

// 如果定义了 TORCH_ENABLE_LLVM，则包含以下内容
#ifdef TORCH_ENABLE_LLVM

// 引入 LLVM 相关的命名空间
namespace tensorexpr {
namespace jit {

// 提供 LLVM 目标三元组（target triple）的可选字符串
TORCH_API std::optional<std::string>& LLVMTargetTriple();

// 提供 LLVM 目标 CPU 的可选字符串
TORCH_API std::optional<std::string>& LLVMTargetCPU();

// 提供 LLVM 目标属性的可选字符串
TORCH_API std::optional<std::string>& LLVMTargetAttrs();

// 提供 LLVM AOT 工作流的布尔值
TORCH_API bool& LLVMAOTWorkflow();

} // namespace tensorexpr
} // namespace jit
} // namespace torch

// 结束 ifdef TORCH_ENABLE_LLVM
#endif // TORCH_ENABLE_LLVM
```