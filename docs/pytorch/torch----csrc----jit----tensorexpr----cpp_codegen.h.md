# `.\pytorch\torch\csrc\jit\tensorexpr\cpp_codegen.h`

```py
#pragma once
// 包含 Torch 库中的头文件，用于张量表达式的代码生成
#include <torch/csrc/jit/tensorexpr/codegen.h>
// 包含 Torch 库中的头文件，用于将 IR 打印成 C++ 代码
#include <torch/csrc/jit/tensorexpr/ir_printer.h>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {
// TensorExpr 命名空间
namespace tensorexpr {

// C++ 变量名重写器类的前置声明
class CppVarNameRewriter;

// CppPrinter 类，继承自 IRPrinter 类，用于生成 C++ 代码
class TORCH_API CppPrinter : public IRPrinter {
 public:
  // 构造函数，接受一个指向输出流的指针作为参数
  explicit CppPrinter(std::ostream* os);
  // 析构函数，用于清理资源
  ~CppPrinter() override;

  // 打印头部声明部分
  void printPrologue();

  // 使用基类的 visit 函数
  using IRPrinter::visit;

  // 以下是各种类型的表达式的 visit 函数的重载

  // 二元表达式
  void visit(ModPtr) override;
  void visit(MaxPtr) override;
  void visit(MinPtr) override;

  // 条件表达式
  void visit(CompareSelectPtr) override;
  void visit(IfThenElsePtr) override;

  // 张量操作
  void visit(AllocatePtr) override;
  void visit(FreePtr) override;
  void visit(LoadPtr) override;
  void visit(StorePtr) override;

  // 强制类型转换
  void visit(CastPtr) override;
  void visit(BitCastPtr) override;

  // 函数调用
  void visit(IntrinsicsPtr) override;
  void visit(ExternalCallPtr) override;

  // 变量
  void visit(LetPtr) override;
  void visit(VarPtr) override;

  // 向量数据类型
  void visit(RampPtr) override;
  void visit(BroadcastPtr) override;

 private:
  int lane_; // 私有成员变量，用于表示向量的宽度
  std::unordered_map<VarPtr, ExprPtr> vector_vars_; // 私有成员变量，存储变量和表达式的映射关系
};

// CppCodeGen 类，继承自 CodeGen 类，用于生成 C++ 代码
class TORCH_API CppCodeGen : public CodeGen {
 public:
  // 构造函数，接受语句、缓冲区参数、设备和内核函数名称作为参数
  CppCodeGen(
      StmtPtr stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func");

  // 析构函数，用于清理资源
  ~CppCodeGen() override;

  // 调用函数，接受参数的向量并执行调用
  void call(const std::vector<CallArg>& args) override;
  // 原始调用函数，接受指针参数的向量并执行调用
  void call_raw(const std::vector<void*>& args) override;

  // 模板运算符重载函数，接受任意数量的参数并调用 call 函数
  template <typename... Ts>
  void operator()(const Ts&... ts) {
    call(std::vector<CallArg>({CallArg(ts)...}));
  }

  // 获取生成的代码文本，可选参数用于附加属性
  std::string getCodeText(const std::string& attr = "") override {
    return oss_.str();
  }

 private:
  // 初始化函数，用于设置打印器和变量名重写器
  void init();

  // 获取输出流的引用
  std::ostream& os() {
    return printer_->os();
  }

  std::ostringstream oss_; // 字符串流，用于存储生成的代码
  std::unique_ptr<CppPrinter> printer_; // CppPrinter 对象的唯一指针
  std::unique_ptr<CppVarNameRewriter> var_name_rewriter_; // CppVarNameRewriter 对象的唯一指针
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```