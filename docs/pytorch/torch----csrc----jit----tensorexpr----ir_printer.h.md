# `.\pytorch\torch\csrc\jit\tensorexpr\ir_printer.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ostream>
// 引入标准输出流类

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>
// 引入Torch Tensor Expression库的前向声明、IR模块、IR访问器、唯一名称管理器头文件

namespace torch {
namespace jit {
namespace tensorexpr {

class Tensor;
// 声明一个Tensor类，但未提供具体实现

class TORCH_API IRPrinter : public IRVisitor {
 public:
  explicit IRPrinter(std::ostream& os) : printer_os_(this, os) {}
  // IRPrinter类构造函数，接受一个标准输出流作为参数

  void print(ExprHandle);
  void print(Expr&);
  void print(Stmt&);
  // 打印表达式和语句的方法声明

  void visit(AddPtr v) override;
  void visit(SubPtr v) override;
  void visit(MulPtr v) override;
  void visit(DivPtr v) override;
  void visit(ModPtr v) override;
  void visit(MaxPtr v) override;
  void visit(MinPtr v) override;
  void visit(AndPtr v) override;
  void visit(OrPtr v) override;
  void visit(XorPtr v) override;
  void visit(LshiftPtr v) override;
  void visit(RshiftPtr v) override;
  void visit(CompareSelectPtr v) override;
  // IRVisitor中虚函数的重写声明，用于访问不同类型的IR节点

#define IMM_PRINT_VISIT(Type, Name) void visit(Name##ImmPtr v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT);
#undef IMM_PRINT_VISIT
  // 定义宏来生成访问立即数节点的虚函数声明

  void visit(CastPtr v) override;
  void visit(BitCastPtr v) override;
  void visit(VarPtr v) override;
  void visit(BufPtr v) override;
  void visit(RampPtr v) override;
  void visit(LoadPtr v) override;
  void visit(BroadcastPtr v) override;
  void visit(IfThenElsePtr v) override;
  void visit(IntrinsicsPtr v) override;
  void visit(TermPtr v) override;
  void visit(PolynomialPtr v) override;
  void visit(RoundOffPtr v) override;
  void visit(MaxTermPtr v) override;
  void visit(MinTermPtr v) override;
  void visit(ReduceOpPtr v) override;
  // 其他各种IR节点类型的访问函数声明

  void visit(AtomicAddPtr v) override;
  void visit(SyncThreadsPtr v) override;
  void visit(ExternalCallPtr v) override;
  void visit(ExternalCallWithAllocPtr v) override;
  void visit(StorePtr v) override;
  void visit(ForPtr v) override;
  void visit(CondPtr v) override;
  void visit(BlockPtr v) override;
  void visit(AllocatePtr v) override;
  void visit(FreePtr v) override;
  void visit(FreeExtPtr v) override;
  void visit(PlacementAllocatePtr v) override;
  void visit(LetPtr v) override;
  // 其他各种IR节点类型的访问函数声明

  // 子类可以有不同的规则来生成dtype的字符串，例如CUDA需要将int64_t生成为long long。
  virtual std::string dtypeToCppString(const Dtype& dtype);
  // 虚函数，根据数据类型生成对应的C++字符串表示

  std::ostream& os() {
    return printer_os_;
  }
  // 返回打印机输出流对象的方法

  class PrinterStream : public std::ostream {
   public:
    PrinterStream(IRPrinter* printer, std::ostream& os)
        : std::ostream(os.rdbuf()), printer_(printer) {}
    // PrinterStream类构造函数，接受一个IRPrinter指针和一个标准输出流作为参数

    IRPrinter* printer() {
      return printer_;
    }
    // 返回IRPrinter指针的方法

   private:
    IRPrinter* printer_ = nullptr;
    // IRPrinter指针的私有成员变量，默认初始化为空指针
  };

 protected:
  std::string to_string(CompareSelectOperation op);
  // 将CompareSelectOperation枚举值转换为字符串表示的方法声明

  UniqueNameManager* name_manager() {
    // 返回唯一名称管理器的方法
    return &name_manager_;
  }
  void emitIndent();

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  int indent_ = 0;

 private:
  // 声明一个私有成员变量，用于输出打印流
  PrinterStream printer_os_;
  // 声明一个私有成员变量，用于管理唯一名称
  UniqueNameManager name_manager_;
// 定义了一个匿名命名空间，用于声明一些与表达式、语句和张量相关的输出流操作符重载
namespace {

    // 重载操作符 <<，使得能够将 Expr 对象输出到给定的输出流中
    TORCH_API std::ostream& operator<<(std::ostream& stream, const Expr&);

    // 重载操作符 <<，使得能够将 ExprHandle 对象输出到给定的输出流中
    TORCH_API std::ostream& operator<<(std::ostream& stream, const ExprHandle&);

    // 重载操作符 <<，使得能够将 Stmt 对象输出到给定的输出流中
    TORCH_API std::ostream& operator<<(std::ostream& stream, const Stmt&);

    // 重载操作符 <<，使得能够将 Tensor 对象输出到给定的输出流中
    TORCH_API std::ostream& operator<<(std::ostream& stream, const Tensor&);

    // 定义了打印函数 print，用于输出 ExprPtr 智能指针所指向的表达式
    TORCH_API void print(ExprPtr expr);

    // 定义了打印函数 print，用于输出 StmtPtr 智能指针所指向的语句
    TORCH_API void print(StmtPtr stmt);

    // 定义了打印函数 print，用于输出 Tensor 对象的信息
    TORCH_API void print(const Tensor& t);

} // namespace tensorexpr

// 使用 std 命名空间，引入了表达式、语句和张量相关的类型和函数，用于字符串表示的转换
namespace std {

    using torch::jit::tensorexpr::Expr;
    using torch::jit::tensorexpr::ExprPtr;
    using torch::jit::tensorexpr::Stmt;
    using torch::jit::tensorexpr::StmtPtr;
    using torch::jit::tensorexpr::Tensor;

    // 将 ExprPtr 智能指针所指向的表达式转换为字符串
    TORCH_API std::string to_string(ExprPtr expr);

    // 将 StmtPtr 智能指针所指向的语句转换为字符串
    TORCH_API std::string to_string(StmtPtr stmt);

    // 将 Tensor 对象转换为字符串
    TORCH_API std::string to_string(const Tensor& t);

} // namespace std
```