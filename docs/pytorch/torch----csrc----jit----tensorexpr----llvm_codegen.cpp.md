# `.\pytorch\torch\csrc\jit\tensorexpr\llvm_codegen.cpp`

```
// 如果定义了 TORCH_ENABLE_LLVM，编译以下代码块

#include <torch/csrc/jit/tensorexpr/llvm_codegen.h> // 引入 LLVM 代码生成器头文件

#include <ATen/NativeFunctions.h> // 引入 ATen 原生函数头文件
#include <ATen/Parallel.h> // 引入 ATen 并行处理头文件
#include <c10/util/Exception.h> // 引入 c10 异常处理头文件
#include <c10/util/irange.h> // 引入 c10 迭代范围头文件
#include <torch/csrc/jit/tensorexpr/analysis.h> // 引入 Torch Tensor 表达式分析头文件
#include <torch/csrc/jit/tensorexpr/llvm_jit.h> // 引入 Torch LLVM JIT 头文件

// 注意 [llvm::SCEVPredicate 非虚拟析构函数]
// llvm::SCEVPredicate 类有虚拟函数但没有虚拟析构函数
// 参考：https://github.com/llvm/llvm-project/blob/c1a0a213378a458fbea1a5c77b315c7dce08fd05/llvm/include/llvm/Analysis/ScalarEvolution.h#L198
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <llvm/Analysis/TargetTransformInfo.h> // 引入 LLVM 目标转换信息头文件
#pragma GCC diagnostic pop

#include <llvm/Analysis/CGSCCPassManager.h> // 引入 LLVM CGSCC Pass 管理头文件
#include <llvm/Analysis/LoopAnalysisManager.h> // 引入 LLVM 循环分析管理头文件
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h> // 引入 LLVM JIT 目标机器构建器头文件
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h> // 引入 LLVM 线程安全模块头文件
#include <llvm/IR/IRBuilder.h> // 引入 LLVM IR 构建器头文件
#include <llvm/IR/LegacyPassManager.h> // 引入 LLVM 旧 Pass 管理器头文件
#include <llvm/IR/MDBuilder.h> // 引入 LLVM 元数据构建器头文件
#include <llvm/IR/PassManager.h> // 引入 LLVM Pass 管理器头文件
#include <llvm/IR/Verifier.h> // 引入 LLVM 验证器头文件
#include <llvm/MC/MCSubtargetInfo.h> // 引入 LLVM MC 子目标信息头文件
#include <llvm/Pass.h> // 引入 LLVM Pass 头文件

// 参考注意 [llvm::SCEVPredicate 非虚拟析构函数]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <llvm/Passes/PassBuilder.h> // 引入 LLVM Pass 构建器头文件
#pragma GCC diagnostic pop

#if LLVM_VERSION_MAJOR >= 18
#include <llvm/TargetParser/Host.h> // 引入 LLVM 目标解析头文件（版本 >= 18）
#else
#include <llvm/Support/Host.h> // 引入 LLVM 主机支持头文件（版本 < 18）
#endif
#include <llvm/Support/TargetSelect.h> // 引入 LLVM 目标选择头文件
#include <llvm/Transforms/IPO/AlwaysInliner.h> // 引入 LLVM 总是内联优化头文件
#include <llvm/Transforms/Scalar/DCE.h> // 引入 LLVM 标量死代码消除头文件
#include <llvm/Transforms/Vectorize/LoopVectorize.h> // 引入 LLVM 循环向量化头文件
#include <llvm/Transforms/Vectorize/SLPVectorizer.h> // 引入 LLVM SLP 向量化头文件

#if LLVM_VERSION_MAJOR >= 10
#include <llvm/Support/CodeGen.h> // 引入 LLVM 代码生成支持头文件（版本 >= 10）
#else
#include <llvm/Target/TargetMachine.h> // 引入 LLVM 目标机器头文件（版本 < 10）
#endif

#if LLVM_VERSION_MAJOR >= 11
#include <llvm/Support/TypeSize.h> // 引入 LLVM 类型大小头文件（版本 >= 11）
#endif

#if LLVM_VERSION_MAJOR < 15
#include <llvm/Transforms/IPO/PassManagerBuilder.h> // 引入 LLVM IPO Pass 管理器构建器头文件（版本 < 15）
#endif

#include <llvm/Transforms/IPO/AlwaysInliner.h> // 引入 LLVM 总是内联优化头文件
#include <llvm/Transforms/Scalar.h> // 引入 LLVM 标量转换头文件

#include <torch/csrc/jit/tensorexpr/expr.h> // 引入 Torch Tensor 表达式头文件
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h> // 引入 Torch 外部函数注册头文件
#include <torch/csrc/jit/tensorexpr/half_support.h> // 引入 Torch 半精度支持头文件
#include <torch/csrc/jit/tensorexpr/ir.h> // 引入 Torch IR 头文件
#include <torch/csrc/jit/tensorexpr/ir_printer.h> // 引入 Torch IR 打印头文件
#include <torch/csrc/jit/tensorexpr/tensor.h> // 引入 Torch Tensor 头文件
#include <torch/csrc/jit/tensorexpr/types.h> // 引入 Torch 类型头文件

#include <torch/csrc/jit/jit_log.h> // 引入 Torch JIT 日志头文件

#include <memory> // 引入内存管理头文件

using namespace torch::jit::tensorexpr; // 使用 Torch Tensor 表达式命名空间

C10_DEFINE_bool(
    torch_jit_llvm_use_fast_intrinsics,
    false,
    "Use fast (but slightly less accurate) implementations of tanh and sigmoid"); // 定义 Torch JIT 使用快速内部函数的标志

namespace torch::jit::tensorexpr {

// 返回 LLVM 目标三元组的可选值引用
std::optional<std::string>& LLVMTargetTriple() {
  static std::optional<std::string> triple = c10::nullopt;
  return triple;
}

// 返回 LLVM 目标 CPU 的可选值引用
std::optional<std::string>& LLVMTargetCPU() {
  static std::optional<std::string> cpu = c10::nullopt;
  return cpu;
}
// 返回对 LLVM 目标属性的可选字符串引用，初始化为无
std::optional<std::string>& LLVMTargetAttrs() {
  static std::optional<std::string> attrs = c10::nullopt;
  return attrs;
}

// 返回 LLVM AOT 工作流程的布尔引用，初始化为 false
bool& LLVMAOTWorkflow() {
  static bool aot_workflow = false;
  return aot_workflow;
}

// 匿名命名空间，用于封装局部定义的类型和变量，限制其作用域

#if LLVM_VERSION_MAJOR >= 15
// 用于处理不透明指针的地址和类型对的结构体
struct TypedPointer {
  TypedPointer() = default;
  TypedPointer(llvm::Type* t, llvm::Value* a) : type(t), addr(a) {}
  llvm::Type* type = nullptr;  // 类型指针，默认为 nullptr
  llvm::Value* addr = nullptr; // 地址值，默认为 nullptr
};
#endif

// 将比较选择操作和标量类型映射到 LLVM 比较操作的谓词
llvm::CmpInst::Predicate llvm_comparison_predicate(
    CompareSelectOperation compare_op,
    const ScalarType& type) {
  switch (compare_op) {
    case CompareSelectOperation::kEQ:
      return llvm::ICmpInst::ICMP_EQ;  // 返回等于的整数比较操作谓词
    case CompareSelectOperation::kNE:
      return llvm::ICmpInst::ICMP_NE;  // 返回不等于的整数比较操作谓词
    case CompareSelectOperation::kGT:
      return c10::isSignedType(type) ? llvm::ICmpInst::ICMP_SGT
                                     : llvm::ICmpInst::ICMP_UGT;  // 根据类型返回有符号或无符号大于的比较操作谓词
    case CompareSelectOperation::kGE:
      return c10::isSignedType(type) ? llvm::ICmpInst::ICMP_SGE
                                     : llvm::ICmpInst::ICMP_UGE;  // 根据类型返回有符号或无符号大于等于的比较操作谓词
    case CompareSelectOperation::kLT:
      return c10::isSignedType(type) ? llvm::ICmpInst::ICMP_SLT
                                     : llvm::ICmpInst::ICMP_ULT;  // 根据类型返回有符号或无符号小于的比较操作谓词
    case CompareSelectOperation::kLE:
      return c10::isSignedType(type) ? llvm::ICmpInst::ICMP_SLE
                                     : llvm::ICmpInst::ICMP_ULE;  // 根据类型返回有符号或无符号小于等于的比较操作谓词
    default:
      // TODO: 更改为适当的错误报告
      throw std::runtime_error("invalid operator type");  // 抛出运行时错误，表示操作类型无效
  }
}

// 将比较选择操作映射到 LLVM 浮点数比较操作的谓词
llvm::CmpInst::Predicate llvm_fp_comparison_predicate(
    CompareSelectOperation compare_op) {
  switch (compare_op) {
    case CompareSelectOperation::kEQ:
      return llvm::FCmpInst::FCMP_OEQ;  // 返回等于的浮点数比较操作谓词
    case CompareSelectOperation::kNE:
      return llvm::FCmpInst::FCMP_ONE;  // 返回不等于的浮点数比较操作谓词
    case CompareSelectOperation::kGT:
      return llvm::FCmpInst::FCMP_OGT;  // 返回大于的浮点数比较操作谓词
    case CompareSelectOperation::kGE:
      return llvm::FCmpInst::FCMP_OGE;  // 返回大于等于的浮点数比较操作谓词
    case CompareSelectOperation::kLT:
      return llvm::FCmpInst::FCMP_OLT;  // 返回小于的浮点数比较操作谓词
    case CompareSelectOperation::kLE:
      return llvm::FCmpInst::FCMP_OLE;  // 返回小于等于的浮点数比较操作谓词
    default:
      // TODO: 更改为适当的错误报告
      throw std::runtime_error("invalid operator type");  // 抛出运行时错误，表示操作类型无效
  }
}

// 根据不同的 LLVM 版本定义 ElementCount 函数的实现
#if LLVM_VERSION_MAJOR <= 9
// 返回 lanes 参数作为元素计数的整数
int ElementCount(int lanes) {
  return lanes;
}
#else
// 返回 lanes 参数作为元素计数的 ElementCount 对象
llvm::ElementCount ElementCount(int lanes) {
#if LLVM_VERSION_MAJOR <= 11
  return llvm::ElementCount(static_cast<unsigned>(lanes), false);  // 使用指定的 lanes 和标志创建 ElementCount 对象
#elif LLVM_VERSION_MAJOR >= 12
  return llvm::ElementCount::getFixed(lanes);  // 创建一个固定元素计数的 ElementCount 对象
#else
#error Only LLVM versions 8 and above are supported.  // 如果版本不在支持范围内则抛出预处理器错误
#endif
}
#endif

// 根据 LLVM 版本选择不同的函数调用类型别名
#if LLVM_VERSION_MAJOR >= 9
using FunctionCallee = llvm::FunctionCallee;  // 使用 LLVM 9 及以上版本的函数调用类型别名
#elif LLVM_VERSION_MAJOR == 8 && LLVM_VERSION_PATCH == 20181009
struct FunctionCallee {
  FunctionCallee() {}

  // 构造函数，用于初始化成员变量
  FunctionCallee(llvm::Constant* fn)
      : v_(fn), ft_(cast<llvm::Function>(v_)->getFunctionType()) {}

  // 返回成员变量 ft_，即函数类型指针
  llvm::FunctionType* getFunctionType() {
    return ft_;
  }

  // 返回成员变量 v_，即函数调用对象
  llvm::Value* getCallee() {
    return v_;
  }

 private:
  llvm::Value* v_{nullptr}; // 函数调用对象
  llvm::FunctionType* ft_{nullptr}; // 函数类型指针
};

#else
#error Only LLVM versions 8 and above are supported.
#endif
} // namespace

class LLVMCodeGenCallee {
 public:
  // 构造函数，初始化成员变量 jit_ 和 kernelAddress_
  LLVMCodeGenCallee(
      std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit,
      void* kernelAddress)
      : jit_(std::move(jit)), kernelAddress_(kernelAddress) {}

  // 返回成员变量 jit_ 的原始指针
  llvm::orc::PytorchLLVMJIT* getJIT() {
    return jit_.get();
  }

  // 返回成员变量 kernelAddress_，即内核地址
  void* getKernelAddress() {
    return kernelAddress_;
  }

  // 设置成员变量 kernelAddress_ 的值
  void setKernelAddress(void* kernelAddress) {
    kernelAddress_ = kernelAddress;
  }

 private:
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit_; // JIT 编译器对象的智能指针
  void* kernelAddress_; // 内核地址
};

class LLVMCodeGenImpl : public IRVisitor {
 private:
  std::unique_ptr<llvm::LLVMContext> context_; // LLVM 上下文的智能指针
  llvm::IRBuilder<> irb_; // LLVM IR 构建器对象
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit_; // JIT 编译器对象的智能指针
  std::unique_ptr<llvm::Module> module_; // LLVM 模块对象的智能指针
  llvm::Function* fn_; // LLVM 函数对象指针
  llvm::BasicBlock* bb_; // LLVM 基本块对象指针
  llvm::Value* value_{nullptr}; // LLVM 值对象指针
  llvm::JITTargetAddress kernelAddress_; // JIT 内核地址
  std::string kernel_func_name_; // 内核函数名称

#define LLVM_TYPE_DECLARE(_1, Name) llvm::Type* Name##Ty_;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, LLVM_TYPE_DECLARE);
#undef LLVM_TYPE_DECLARE

#if LLVM_VERSION_MAJOR >= 15
  llvm::Type* OpqPtrTy_; // LLVM 不透明指针类型
#else
  llvm::Type* Int8PtrTy_; // LLVM int8 指针类型
#endif
  llvm::Type* VoidTy_; // LLVM 空类型
  std::unordered_map<VarPtr, int> varToArg_; // 变量指针到参数索引的映射
  std::unordered_map<VarPtr, llvm::Value*> varToVal_; // 变量指针到值对象的映射
  std::unordered_set<BufPtr> bufsExtAlloc_; // 外部分配的缓冲区集合
  std::unordered_map<VarPtr, llvm::Value*> bufsExtToFreeVal_; // 外部释放值对象的变量指针映射
  std::unordered_multimap<BufPtr, BufPtr> bufsExtAllocReuse_; // 外部分配重用的缓冲区映射
  std::unordered_map<BlockPtr, std::vector<VarPtr>> scopeToVar_; // 作用域到变量指针列表的映射
  BlockPtr scope_; // 块指针

  std::string llvmCode_; // LLVM 代码字符串
  std::string asmCode_; // 汇编代码字符串

 private:
  // 获取 LLVM 上下文的引用
  llvm::LLVMContext& getContext();
  // 将数据类型转换为 LLVM 类型
  llvm::Type* dtypeToLLVM(Dtype dtype);
  // 将数据类型转换为 LLVM 指针类型
  llvm::Type* dtypeToLLVMPtr(Dtype dtype);
  // 发射包装器函数，参数为 LLVM 类型的向量
  void emitWrapper(const std::vector<llvm::Type*>& params);
  // 发射内核函数，参数为语句指针和 LLVM 类型的向量
  void emitKernel(StmtPtr stmt, const std::vector<llvm::Type*>& params);
  // 将值对象转换为向量，参数为值对象和通道数
  llvm::Value* toVec(llvm::Value* v, int lanes);

  // 枚举类型，表示函数的参数数量
  enum Arity {
    Unary, // 一元函数
    Binary, // 二元函数
  };

  // SIMD 函数调用类型的别名
  using SimdCallee = std::tuple<llvm::FunctionType*, llvm::Value*, bool>;
  // 获取 SIMD 函数，参数为名称、类型、函数参数数量、通道数
  SimdCallee getSimdFunction(
      const std::string& name,
      llvm::Type* type,
      Arity arity,
      int lanes);

  // 将变量指针映射到值对象
  llvm::Value* varToValue(VarPtr var);
  // 替换变量映射，参数为变量指针列表和值对象列表
  void replaceVarMapping(
      const std::vector<VarPtr>& vars,
      const std::vector<llvm::Value*>& vals);

#if LLVM_VERSION_MAJOR >= 15
  // 打包函数参数，参数为 LLVM 值对象的向量
  TypedPointer packFuncArgs(const std::vector<llvm::Value*>& func_args);
  // 解包函数参数，参数为打包的类型指针和参数数量
  std::vector<llvm::Value*> unpackFuncArgs(TypedPointer packed, int arg_count);
#ifdef SOME_CONDITION
  // 如果定义了 SOME_CONDITION，则声明 packFuncArgs 函数，用于打包函数参数
  llvm::Value* packFuncArgs(const std::vector<llvm::Value*>& func_args);
  // 如果定义了 SOME_CONDITION，则声明 unpackFuncArgs 函数，用于解包函数参数
  std::vector<llvm::Value*> unpackFuncArgs(llvm::Value* packed, int arg_count);
#endif

// 声明 processParallelFor 函数，处理并行循环
void processParallelFor(ForPtr v);
// 声明 handleBufReuse 函数，处理缓冲区重用
void handleBufReuse(BufPtr buf, BufPtr buf_to_reuse);

public:
// LLVMCodeGenImpl 类的构造函数，初始化成员变量和设备特性
LLVMCodeGenImpl(
    StmtPtr stmt,
    const std::vector<CodeGen::BufferArg>& args,
    at::Device device,
    Dtype dtype,
    std::string kernel_func_name,
    std::optional<std::string> triple,
    std::optional<std::string> cpu,
    std::optional<std::string> attrs);
// LLVMCodeGenImpl 类的析构函数，默认实现
~LLVMCodeGenImpl() = default;

// 获取内核函数地址的方法
llvm::JITTargetAddress getKernelAddress() const;
// 释放 JIT 编译的方法，返回 PytorchLLVMJIT 对象的唯一指针
std::unique_ptr<llvm::orc::PytorchLLVMJIT> releaseJIT();

// 以下是各种 visit 方法的重写，处理不同类型的 AST 节点
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

// 使用宏定义批量声明 visit 方法，处理不同的标量类型
#define IMM_VISIT_DECLARE(_1, Name) void visit(Name##ImmPtr v) override;
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_VISIT_DECLARE);
#undef IMM_VISIT_DECLARE

// 处理 CastPtr 类型的节点
void visit(CastPtr v) override;
// 处理 BitCastPtr 类型的节点
void visit(BitCastPtr v) override;
// 处理 VarPtr 类型的节点
void visit(VarPtr v) override;
// 处理 RampPtr 类型的节点
void visit(RampPtr v) override;
// 处理 LoadPtr 类型的节点
void visit(LoadPtr v) override;
// 处理 ForPtr 类型的节点
void visit(ForPtr v) override;
// 处理 BlockPtr 类型的节点
void visit(BlockPtr v) override;
// 处理 StorePtr 类型的节点
void visit(StorePtr v) override;
// 处理 BroadcastPtr 类型的节点
void visit(BroadcastPtr v) override;
// 处理 IfThenElsePtr 类型的节点
void visit(IfThenElsePtr v) override;
// 处理 IntrinsicsPtr 类型的节点
void visit(IntrinsicsPtr v) override;
// 处理 AllocatePtr 类型的节点
void visit(AllocatePtr v) override;
// 处理 FreePtr 类型的节点
void visit(FreePtr v) override;
// 处理 FreeExtPtr 类型的节点
void visit(FreeExtPtr v) override;
// 处理 PlacementAllocatePtr 类型的节点
void visit(PlacementAllocatePtr v) override;
// 处理 LetPtr 类型的节点
void visit(LetPtr v) override;
// 处理 CondPtr 类型的节点
void visit(CondPtr v) override;
// 处理 ExternalCallPtr 类型的节点
void visit(ExternalCallPtr v) override;
// 处理 ExternalCallWithAllocPtr 类型的节点
void visit(ExternalCallWithAllocPtr v) override;

// 发出 IsNan 方法的实现
void emitIsNan(IntrinsicsPtr v);

// 发出未掩码加载的方法
llvm::Value* emitUnmaskedLoad(
    llvm::Type* ty,
    llvm::Value* addr,
    llvm::Value* idx);
// 发出掩码加载的方法
llvm::Value* emitMaskedLoad(
    llvm::Type* ty,
    llvm::Value* addr,
    llvm::Value* idx,
    llvm::Value* mask);
// 发出未掩码存储的方法
void emitUnmaskedStore(
    llvm::Type* ty,
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* val);
// 发出掩码存储的方法
void emitMaskedStore(
    llvm::Type* ty,
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* mask,
    llvm::Value* val);

// 优化 LLVM 模块的方法
void optimize(llvm::Module& M);
// 获取 LLVM 代码文本的方法
std::string getLLVMCodeText() {
  return llvmCode_;
}
// 获取汇编代码文本的方法
std::string getASMCodeText() {
  return asmCode_;
}
};

// 结束 torch::jit::tensorexpr 命名空间
} // namespace torch::jit::tensorexpr

// 默认析构函数实现
LLVMCodeGen::~LLVMCodeGen() = default;

// LLVMCodeGen 类的构造函数实现，调用 LLVMCodeGenImpl 的构造函数
LLVMCodeGen::LLVMCodeGen(StmtPtr stmt)
    : LLVMCodeGen(stmt, std::vector<CodeGen::BufferArg>()) {}

// LLVMCodeGen 类的构造函数实现，带有参数列表，调用 LLVMCodeGenImpl 的构造函数
LLVMCodeGen::LLVMCodeGen(
    StmtPtr stmt,
    const std::vector<BufferArg>& args,
    # 构造函数，初始化 CodeGen 类，继承自 stmt, args, device, kernel_func_name 参数
    at::Device device,
    # kernel_func_name 参数，指定要生成的内核函数名称
    const std::string& kernel_func_name,
    # dtype 参数，指定数据类型
    Dtype dtype,
    # triple 参数，可选，指定目标三元组信息
    std::optional<std::string> triple,
    # cpu 参数，可选，指定目标 CPU 信息
    std::optional<std::string> cpu,
    # attrs 参数，可选，指定额外的属性信息
    std::optional<std::string> attrs)
    # 构造函数初始化列表，调用基类构造函数并初始化 impl_ 成员
    : CodeGen(stmt, args, device, kernel_func_name) {
  # 创建 LLVMCodeGenImpl 的实例，并传入构造函数参数进行初始化
  impl_ = std::make_unique<LLVMCodeGenImpl>(
      this->stmt(),
      args,
      device,
      dtype,
      this->kernel_func_name(),
      triple,
      cpu,
      attrs);
  # 创建 LLVMCodeGenCallee 的实例，传入 LLVMCodeGenImpl 实例的 JIT 释放和内核地址作为参数
  callee_ = std::make_unique<LLVMCodeGenCallee>(
      impl_->releaseJIT(), (void*)impl_->getKernelAddress());
{
// 重置impl_指针，释放其所指向的内存资源
void LLVMCodeGen::cleanup_memory() {
  impl_.reset(nullptr);
}

// 调用LLVM生成的代码，传入参数列表args的内存地址
void LLVMCodeGen::call_raw(const std::vector<void*>& args) {
  value<float>(const_cast<void**>(args.data()));
}

// 带有numel参数的调用，args为指向参数数组的指针，numel为数组元素个数（未使用）
void LLVMCodeGen::call_with_numel(void** args, int64_t /* numel */) {
  value<float>(const_cast<void**>(args));
}

// 使用CallArg向量进行调用，确保参数数量与buffer_args相匹配
void LLVMCodeGen::call(const std::vector<CallArg>& args) {
  auto& buf_args = buffer_args();
  if (args.size() != buf_args.size()) {
    throw malformed_input("wrong number of args in call");
  }

  constexpr unsigned nargs = 8;
  c10::SmallVector<void*, nargs> argv;
  argv.resize(buf_args.size());
  for (size_t i = 0, e = buf_args.size(); i < e; i++) {
    auto const& bufferArg = buf_args[i];
    auto const& callArg = args[i];
    argv[i] = argToPtr(bufferArg, callArg);
  }
  // 调用LLVM生成的代码，传入参数数组argv的数据指针
  value<float>(argv.data());
}

// 创建空的分步张量，参数指定了张量的大小、步长以及其他属性
at::Tensor LLVMCodeGen::empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  return at::native::empty_strided_cpu(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// 获取LLVMCodeGenCallee对象的内核地址
void* LLVMCodeGen::getKernelAddress(LLVMCodeGenCallee* callee) {
  return (void*)callee->getKernelAddress();
}

// 获取代码文本，可以选择性地传入attr参数，默认为空字符串
std::string LLVMCodeGen::getCodeText(const std::string& attr /*=""*/) {
  // 断言impl_不为空，确保LLVMCodeGen对象的内存未被清理
  TORCH_INTERNAL_ASSERT(
      impl_.get(),
      "LLVMCodeGen memory has been cleaned up. So, code text is not available at this point");
  if (attr == "asm") {
    return impl_->getASMCodeText();  // 返回ASM格式的代码文本
  } else {
    return impl_->getLLVMCodeText();  // 返回LLVM IR格式的代码文本
  }
}

// 获取内核地址，返回JITTargetAddress类型
llvm::JITTargetAddress LLVMCodeGenImpl::getKernelAddress() const {
  return kernelAddress_;
}

// 释放JIT对象所有权，返回一个unique_ptr指针
std::unique_ptr<llvm::orc::PytorchLLVMJIT> LLVMCodeGenImpl::releaseJIT() {
  return std::move(jit_);
}

namespace {
// 全局互斥锁，保护LLVM的初始化过程，特别是TargetRegistry::lookupTarget函数
static std::mutex llvmInitMutex;
} // namespace

// LLVMCodeGenImpl的构造函数，初始化成员变量并进行一些默认设置
LLVMCodeGenImpl::LLVMCodeGenImpl(
    StmtPtr stmt,
    const std::vector<CodeGen::BufferArg>& args,
    at::Device device,
    Dtype dtype,
    std::string kernel_func_name,
    std::optional<std::string> triple,
    std::optional<std::string> cpu,
    std::optional<std::string> attrs)
    : context_(std::make_unique<llvm::LLVMContext>()),
      irb_(getContext()),
      kernel_func_name_(std::move(kernel_func_name)),
      bufsExtAlloc_(ExternalAllocBufFinder::find(stmt)) {
  // 如果triple未提供，则获取默认的LLVM目标三元组
  if (!triple) {
    triple = LLVMTargetTriple();
  }
  // 如果cpu未提供，则获取默认的LLVM目标CPU类型
  if (!cpu) {
    cpu = LLVMTargetCPU();
  }
  // 如果attrs未提供，则设置为空字符串
  if (!attrs) {
    // 创建 LLVMTargetAttrs 对象，并初始化为默认构造函数生成的对象
    attrs = LLVMTargetAttrs();

  // 手动映射类型到 LLVM 类型。

  // 将上下文中的字节类型设置为 LLVM 的 8 位整数类型
  ByteTy_ = llvm::Type::getInt8Ty(getContext());
  // 将上下文中的字符类型设置为 LLVM 的 8 位整数类型
  CharTy_ = llvm::Type::getInt8Ty(getContext());
  // 将上下文中的短整数类型设置为 LLVM 的 16 位整数类型
  ShortTy_ = llvm::Type::getInt16Ty(getContext());
  // 将上下文中的整数类型设置为 LLVM 的 32 位整数类型
  IntTy_ = llvm::Type::getInt32Ty(getContext());
  // 将上下文中的长整数类型设置为 LLVM 的 64 位整数类型
  LongTy_ = llvm::Type::getInt64Ty(getContext());
  // 将上下文中的半精度浮点数类型设置为 LLVM 的半精度浮点类型
  HalfTy_ = llvm::Type::getHalfTy(getContext());
  // 将上下文中的单精度浮点数类型设置为 LLVM 的单精度浮点类型
  FloatTy_ = llvm::Type::getFloatTy(getContext());
  // 将上下文中的双精度浮点数类型设置为 LLVM 的双精度浮点类型
  DoubleTy_ = llvm::Type::getDoubleTy(getContext());
  // 将上下文中的空类型设置为 LLVM 的空类型
  VoidTy_ = llvm::Type::getVoidTy(getContext());
  // 将布尔类型设置为与 ByteTy_ 相同，即 LLVM 的 8 位整数类型
  BoolTy_ = ByteTy_;
#if LLVM_VERSION_MAJOR >= 15
  // 如果 LLVM 版本大于等于 15，使用未限定的指针类型 OpqPtrTy_
  OpqPtrTy_ = llvm::PointerType::getUnqual(getContext());
#else
  // 如果 LLVM 版本小于 15，使用 int8 指针类型 Int8PtrTy_
  Int8PtrTy_ = llvm::Type::getInt8PtrTy(getContext());
#endif

  {
    // 使用互斥锁保护代码段，确保线程安全性
    std::lock_guard<std::mutex> g(llvmInitMutex);
    // 初始化所有的 LLVM 目标
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    // 创建一个 PytorchLLVMJIT 对象 jit_
    jit_ = std::make_unique<llvm::orc::PytorchLLVMJIT>(triple, cpu, attrs);
  }

  // 创建一个名为 "pytorch" 的新 LLVM 模块 module_
  module_ = std::make_unique<llvm::Module>("pytorch", getContext());
  // 设置模块的数据布局为 JIT 对象 jit_ 的数据布局
  module_->setDataLayout(jit_->getDataLayout());
  // 设置模块的目标三元组为 JIT 对象 jit_ 的目标机器的目标三元组字符串表示
  module_->setTargetTriple(jit_->getTargetMachine().getTargetTriple().str());

  // 支持 float16 运算，通过将表达式输入转换为 float32，然后将结果转换回 float16
  GRAPH_DEBUG("Before HalfRewriter ", *stmt);
  // 创建 HalfRewriter 对象 hsFix 并应用于 stmt
  HalfRewriter hsFix;
  stmt = stmt->accept_mutator(&hsFix);
  GRAPH_DEBUG("After HalfRewriter ", *stmt);

  // 将 Dtype 转换为 LLVM 类型，返回 LLVM 类型指针 retTy
  llvm::Type* retTy = dtypeToLLVM(dtype);
  // 构建参数列表 params
  std::vector<llvm::Type*> params;
  for (const auto i : c10::irange(args.size())) {
    auto const& arg = args[i];
    if (arg.isVar()) {
      params.push_back(dtypeToLLVM(arg.dtype()));
    } else {
      params.push_back(dtypeToLLVMPtr(arg.dtype()));
    }
    // 将参数与参数索引绑定到 varToArg_ 映射中
    varToArg_[arg.var()] = i;
  }
  // 创建 LLVM 函数类型 fntype
  llvm::FunctionType* fntype = llvm::FunctionType::get(retTy, params, false);
  // 创建 LLVM 函数 fn_，并添加到模块 module_ 中，私有链接
  fn_ = llvm::Function::Create(
      fntype, llvm::Function::PrivateLinkage, "pytorch", module_.get());
  // 将函数标记为总是内联
  fn_->addFnAttr(llvm::Attribute::AlwaysInline);
  // 对于参数列表中不是变量的参数，添加 LLVM 参数属性 NoAlias
  for (const auto i : c10::irange(args.size())) {
    if (!args[i].isVar()) {
      fn_->addParamAttr(i, llvm::Attribute::NoAlias);
    }
  }

  // 发出包装器函数的代码
  emitWrapper(params);
  // 发出核心函数 stmt 的代码
  emitKernel(stmt, params);

  // 将模块 module_ 和上下文 context_ 移动到 JIT 对象 jit_ 中
  jit_->addModule(std::move(module_), std::move(context_));
  // 如果不是 LLVM AOT 工作流，查找内核函数符号 kernel_func_name_ 并获取其地址赋值给 kernelAddress_
  if (!LLVMAOTWorkflow()) {
    auto sym = jit_->findSymbol(kernel_func_name_);
    kernelAddress_ = assertSuccess(sym.getAddress());
  }
}

// 获取 LLVM 上下文的引用
llvm::LLVMContext& LLVMCodeGenImpl::getContext() {
  return *context_;
}

// 将 Dtype 转换为相应的 LLVM 类型
llvm::Type* LLVMCodeGenImpl::dtypeToLLVM(Dtype dtype) {
  switch (dtype.scalar_type()) {
#define TYPE_CASE(_1, n) \
  case ScalarType::n:    \
    return n##Ty_;       \
    break;

    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::QInt8:
      return CharTy_;
      break;

    case ScalarType::QUInt8:
      return ByteTy_;
      break;

    case ScalarType::BFloat16:
      return ShortTy_;
      break;

    default:
      // 抛出不支持的 Dtype 异常
      throw unsupported_dtype();
  }
  return nullptr;
}

// 将 Dtype 转换为对应的 LLVM 类型指针
llvm::Type* LLVMCodeGenImpl::dtypeToLLVMPtr(Dtype dtype) {
  return dtypeToLLVM(dtype)->getPointerTo();
}

// 发出包装器函数的代码
void LLVMCodeGenImpl::emitWrapper(const std::vector<llvm::Type*>& params) {
#if LLVM_VERSION_MAJOR >= 15
  // 如果 LLVM 版本大于等于 15，创建一个外部链接的函数 wrapper，其参数为 OpqPtrTy_
  auto wrapper = llvm::Function::Create(
      llvm::FunctionType::get(IntTy_, {OpqPtrTy_}, false),
      llvm::Function::ExternalLinkage,
      kernel_func_name_,
      module_.get());
#else
  // 定义一个指向 void 类型的指针类型
  auto voidPtrTy = llvm::Type::getInt8PtrTy(getContext());
  // 获取指向指针类型的指针类型
  auto voidPtrPtrTy = voidPtrTy->getPointerTo();
  // 创建一个外部链接的 LLVM 函数，接受一个指向指针的参数
  auto wrapper = llvm::Function::Create(
      llvm::FunctionType::get(IntTy_, {voidPtrPtrTy}, false),
      llvm::Function::ExternalLinkage,
      kernel_func_name_,
      module_.get());
#endif

// 创建一个基本块，用于包装函数
auto wrapBB = llvm::BasicBlock::Create(getContext(), "wrapBB", wrapper);
irb_.SetInsertPoint(wrapBB);
// 创建一个小型向量，用于存储包装后的参数
llvm::SmallVector<llvm::Value*, 6> wrappedArgs;
// 遍历参数列表
for (const auto i : c10::irange(params.size())) {
#if LLVM_VERSION_MAJOR >= 15
  // 创建一个指向指针的 GEP 指令
  auto argp = irb_.CreateGEP(
      OpqPtrTy_,
      wrapper->arg_begin(),
      llvm::ConstantInt::getSigned(IntTy_, i));
  // 如果参数是指针类型，则创建指针转换和加载指令
  if (params[i]->isPointerTy()) {
    auto arg =
        irb_.CreatePointerCast(irb_.CreateLoad(OpqPtrTy_, argp), params[i]);
    wrappedArgs.push_back(arg);
  } else {
    // 否则创建指针转换和加载指令
    auto p =
        irb_.CreatePointerCast(irb_.CreateLoad(OpqPtrTy_, argp), OpqPtrTy_);
    auto arg = irb_.CreateLoad(params[i], p);
    wrappedArgs.push_back(arg);
  }
#else
  // 创建一个指向 void 的 GEP 指令
  auto argp = irb_.CreateGEP(
      voidPtrTy,
      wrapper->arg_begin(),
      llvm::ConstantInt::getSigned(IntTy_, i));
  // 如果参数是指针类型，则创建指针转换和加载指令
  if (params[i]->isPointerTy()) {
    auto arg = irb_.CreatePointerCast(
        irb_.CreateLoad(argp->getType()->getPointerElementType(), argp),
        params[i]);
    wrappedArgs.push_back(arg);
  } else {
    // 否则创建指针转换和加载指令
    auto p = irb_.CreatePointerCast(
        irb_.CreateLoad(argp->getType()->getPointerElementType(), argp),
        params[i]->getPointerTo());
    auto arg = irb_.CreateLoad(p->getType()->getPointerElementType(), p);
    wrappedArgs.push_back(arg);
  }
#endif
}
// 创建一个函数调用指令，并返回
auto cc = irb_.CreateCall(fn_, wrappedArgs);
irb_.CreateRet(cc);
}
  // 使用语句接受变异器 intrinsics_expander 处理后的结果
  stmt = stmt->accept_mutator(&intrinsics_expander);
}

// 编译内核代码
stmt->accept(this);

// 如果内核为空，设置默认返回值为整数常量 0
if (value_ == nullptr) {
  value_ = llvm::ConstantInt::get(IntTy_, 0);
}

// 创建函数返回指令，返回值为 value_
irb_.CreateRet(value_);

// 准备打印图形调试信息之前的 LLVM 模块
llvm::SmallVector<char, 0> asmBuffer;
llvm::raw_svector_ostream asmStream(asmBuffer);
if (GRAPH_DEBUG_ENABLED) {
  // 将模块内容打印到 asmStream 中
  module_->print(asmStream, nullptr);
}
GRAPH_DEBUG(
    "\nLLVM module before optimizations\n\n", asmStream.str().str(), "\n");

// 验证函数的正确性，如果验证失败则抛出异常
if (llvm::verifyFunction(*fn_, &llvm::outs())) {
  throw std::runtime_error("Function verification failed");
}

// 对模块进行优化
optimize(*module_);

// 清空 asmBuffer，准备打印优化后的 LLVM 模块
asmBuffer.clear();
module_->print(asmStream, nullptr);
llvmCode_ = asmStream.str().str();
GRAPH_DEBUG(
    "\nLLVM module after optimizations\n\n", asmStream.str().str(), "\n");

// 准备打印图形调试信息之后的 LLVM 模块
asmBuffer.clear();
llvm::legacy::PassManager PM;
// 将 JIT 的目标机器添加到 PassManager 中，用于生成输出文件
jit_->getTargetMachine().addPassesToEmitFile(
    PM,
    asmStream,
    nullptr,
#if LLVM_VERSION_MAJOR >= 18
      llvm::CodeGenFileType::AssemblyFile);
#elif LLVM_VERSION_MAJOR >= 10
      llvm::CodeGenFileType::CGFT_AssemblyFile);
#else
      llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
#endif



// 根据 LLVM 版本选择不同的代码生成文件类型
#if LLVM_VERSION_MAJOR >= 18
      llvm::CodeGenFileType::AssemblyFile);
#elif LLVM_VERSION_MAJOR >= 10
      llvm::CodeGenFileType::CGFT_AssemblyFile);
#else
      llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
#endif



  PM.run(*module_);



  // 运行 LLVM Pass Manager 来优化和转换模块
  PM.run(*module_);



  asmCode_ = asmStream.str().str();



  // 将汇编代码流转换为字符串并存储在 asmCode_ 变量中
  asmCode_ = asmStream.str().str();



  GRAPH_DEBUG("\nLLVM generated assembly code\n\n", asmCode_, "\n");
}



  // 在调试模式下输出生成的 LLVM 汇编代码
  GRAPH_DEBUG("\nLLVM generated assembly code\n\n", asmCode_, "\n");
}



// TODO: The binary ops are copypasta.

void LLVMCodeGenImpl::visit(AddPtr v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFAdd(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateAdd(lhs, rhs);
  } else {
    throw malformed_input("llvm_codegen: bad type in Add", v);
  }
}



// TODO: The binary ops are copypasta.

void LLVMCodeGenImpl::visit(AddPtr v) {
  // 访问加法表达式左操作数
  v->lhs()->accept(this);
  auto lhs = this->value_;
  // 检查左操作数是否为浮点数或浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问加法表达式右操作数
  v->rhs()->accept(this);
  auto rhs = this->value_;
  // 检查右操作数是否为浮点数或浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 根据操作数类型选择相应的 LLVM IR 指令生成方式
  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFAdd(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateAdd(lhs, rhs);
  } else {
    // 抛出异常，因为加法操作要求两个操作数类型必须一致
    throw malformed_input("llvm_codegen: bad type in Add", v);
  }
}



void LLVMCodeGenImpl::visit(SubPtr v) {
  // 访问减法表达式左操作数
  v->lhs()->accept(this);
  auto lhs = this->value_;
  // 检查左操作数是否为浮点数或浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问减法表达式右操作数
  v->rhs()->accept(this);
  auto rhs = this->value_;
  // 检查右操作数是否为浮点数或浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 根据操作数类型选择相应的 LLVM IR 指令生成方式
  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFSub(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateSub(lhs, rhs);
  } else {
    // 抛出异常，因为减法操作要求两个操作数类型必须一致
    throw malformed_input("llvm_codegen: bad type in Sub", v);
  }
}



void LLVMCodeGenImpl::visit(MulPtr v) {
  // 访问乘法表达式左操作数
  v->lhs()->accept(this);
  auto lhs = this->value_;
  // 检查左操作数是否为浮点数或浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问乘法表达式右操作数
  v->rhs()->accept(this);
  auto rhs = this->value_;
  // 检查右操作数是否为浮点数或浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 根据操作数类型选择相应的 LLVM IR 指令生成方式
  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFMul(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateMul(lhs, rhs);
  } else {
    // 抛出异常，因为乘法操作要求两个操作数类型必须一致
    throw malformed_input("llvm_codegen: bad type in Mul", v);
  }
}



void LLVMCodeGenImpl::visit(DivPtr v) {
  // 访问除法表达式左操作数
  v->lhs()->accept(this);
  auto lhs = this->value_;
  // 检查左操作数是否为浮点数或浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问除法表达式右操作数
  v->rhs()->accept(this);
  auto rhs = this->value_;
  // 检查右操作数是否为浮点数或浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 根据操作数类型选择相应的 LLVM IR 指令生成方式
  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFDiv(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateSDiv(lhs, rhs);
  } else {
    // 抛出异常，因为除法操作要求两个操作数类型必须一致
    throw malformed_input("llvm_codegen: bad type in Div", v);
  }
}



void LLVMCodeGenImpl::visit(AndPtr v) {
  // 访问按位与表达式左操作数
  v->lhs()->accept(this);
  auto lhs = this->value_;
  // 检查左操作数是否既不是浮点数也不是浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问按位与表达式右操作数
  v->rhs()->accept(this);
  auto rhs = this->value_;
  // 检查右操作数是否既不是浮点数也不是浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 按位与操作要求两个操作数必须都是整数类型
  if (!lfp && !rfp) {
    value_ = irb_.CreateAnd(lhs, rhs);
  } else {
    // 抛出异常，因为按位与操作要求两个操作数必须都是整数类型
    throw malformed_input("llvm_codegen: bad type in And", v);
  }
}
void LLVMCodeGenImpl::visit(OrPtr v) {
  // 访问 Or 操作的左操作数，并进行处理
  v->lhs()->accept(this);
  // 获取左操作数的 LLVM IR 表示
  auto lhs = this->value_;
  // 检查左操作数是否为浮点数或浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问 Or 操作的右操作数，并进行处理
  v->rhs()->accept(this);
  // 获取右操作数的 LLVM IR 表示
  auto rhs = this->value_;
  // 检查右操作数是否为浮点数或浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 如果左右操作数都不是浮点数或浮点向量类型，则创建 Or 指令
  if (!lfp && !rfp) {
    value_ = irb_.CreateOr(lhs, rhs);
  } else {
    // 如果其中一个操作数是浮点数或浮点向量类型，则抛出异常
    throw malformed_input("llvm_codegen: bad type in Or", v);
  }
}

void LLVMCodeGenImpl::visit(XorPtr v) {
  // 访问 Xor 操作的左操作数，并进行处理
  v->lhs()->accept(this);
  // 获取左操作数的 LLVM IR 表示
  auto lhs = this->value_;
  // 检查左操作数是否为浮点数或浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问 Xor 操作的右操作数，并进行处理
  v->rhs()->accept(this);
  // 获取右操作数的 LLVM IR 表示
  auto rhs = this->value_;
  // 检查右操作数是否为浮点数或浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 如果左右操作数都不是浮点数或浮点向量类型，则创建 Xor 指令
  if (!lfp && !rfp) {
    value_ = irb_.CreateXor(lhs, rhs);
  } else {
    // 如果其中一个操作数是浮点数或浮点向量类型，则抛出异常
    throw malformed_input("llvm_codegen: bad type in Xor", v);
  }
}

void LLVMCodeGenImpl::visit(LshiftPtr v) {
  // 访问左移操作的左操作数，并进行处理
  v->lhs()->accept(this);
  // 获取左操作数的 LLVM IR 表示
  auto lhs = this->value_;
  // 检查左操作数是否为浮点数或浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问左移操作的右操作数，并进行处理
  v->rhs()->accept(this);
  // 获取右操作数的 LLVM IR 表示
  auto rhs = this->value_;
  // 检查右操作数是否为浮点数或浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 如果左右操作数都不是浮点数或浮点向量类型，则创建左移指令
  if (!lfp && !rfp) {
    value_ = irb_.CreateShl(lhs, rhs);
  } else {
    // 如果其中一个操作数是浮点数或浮点向量类型，则抛出异常
    throw malformed_input("llvm_codegen: bad type in Lshift", v);
  }
}

void LLVMCodeGenImpl::visit(RshiftPtr v) {
  // 访问右移操作的左操作数，并进行处理
  v->lhs()->accept(this);
  // 获取左操作数的 LLVM IR 表示
  auto lhs = this->value_;
  // 检查左操作数是否为浮点数或浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问右移操作的右操作数，并进行处理
  v->rhs()->accept(this);
  // 获取右操作数的 LLVM IR 表示
  auto rhs = this->value_;
  // 检查右操作数是否为浮点数或浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 如果左右操作数都不是浮点数或浮点向量类型，则根据有符号或无符号类型创建右移指令
  if (!lfp && !rfp) {
    if (v->lhs()->dtype().is_signed()) {
      value_ = irb_.CreateAShr(lhs, rhs);
    } else {
      value_ = irb_.CreateLShr(lhs, rhs);
    }
  } else {
    // 如果其中一个操作数是浮点数或浮点向量类型，则抛出异常
    throw malformed_input("llvm_codegen: bad type in Rshift", v);
  }
}

void LLVMCodeGenImpl::visit(ModPtr v) {
  // 访问取模操作的左操作数，并进行处理
  v->lhs()->accept(this);
  // 获取左操作数的 LLVM IR 表示
  auto lhs = this->value_;
  // 检查左操作数是否为浮点数或浮点向量类型
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  // 访问取模操作的右操作数，并进行处理
  v->rhs()->accept(this);
  // 获取右操作数的 LLVM IR 表示
  auto rhs = this->value_;
  // 检查右操作数是否为浮点数或浮点向量类型
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // 如果左右操作数都不是浮点数或浮点向量类型，则创建取模指令
  if (!lfp && !rfp) {
    value_ = irb_.CreateSRem(lhs, rhs);
  } else {
    // 如果其中一个操作数是浮点数或浮点向量类型，则抛出异常
    throw malformed_input("llvm_codegen: bad type in Mod", v);
  }
}

void LLVMCodeGenImpl::visit(MaxPtr v) {
  // 访问求最大值操作的左操作数，并进行处理
  v->lhs()->accept(this);
  // 获取左操作数的 LLVM IR 表示
  auto lhs = this->value_;
  // 访问求最大值操作的右操作数，并进行处理
  v->rhs()->accept(this);
  // 获取右操作数的 LLVM IR 表示
  auto rhs = this->value_;

  // 如果数据类型为整数类型
  if (v->dtype().is_integral()) {
    // 创建整数比较，根据符号类型选择有符号或无符号比较
    auto icmp = v->dtype().is_signed() ? irb_.CreateICmpSGT(lhs, rhs)
                                       : irb_.CreateICmpUGT(lhs, rhs);
    // 根据比较结果选择返回较大值或较小值
    value_ = irb_.CreateSelect(icmp, lhs, rhs);
    return;
  }

  // 如果数据类型为浮点数类型
  value_ = irb_.CreateSelect(
      // 创建浮点数比较，检查左操作数是否为 NaN
      irb_.CreateFCmp(
          llvm::FCmpInst::FCMP_UNO,
          lhs,
          llvm::ConstantFP::get(lhs->getType(), 0.0)),
      // 如果左操作数为 NaN，则返回左操作数
      lhs,
      // 否则根据大小关系返回较大值或较小值
      irb_.CreateSelect(
          irb_.CreateFCmp(llvm::FCmpInst::FCMP_OGT, lhs, rhs), lhs, rhs));
}

void LLVMCodeGenImpl::visit(MinPtr v) {
  // 访问求最小值操作的左操作数，并进行处理
  v->lhs()->accept(this);
  // 获取左操作数的 LLVM IR 表示
  auto lhs = this->value_;
  // 访问求最小值操作的右操作数，并进行处理
  v->rhs()->accept(this);
  // 获取右操作数的 LLVM IR 表示
  auto rhs = this->value_;
  
  // 如果数据类型为整数类型
  if (v->dtype().is_integral()) {
    // 根据大小关系选择返回较小值或较大值
    auto icmp = v->dtype().is_signed() ? irb_.CreateICmpSLT(lhs, rhs)
                                       : irb_.CreateICmpULT(lhs, rhs);
    value_ = ir
    // 根据变量类型是否有符号，创建相应的比较指令（有符号或无符号整数比较）
    auto icmp = v->dtype().is_signed() ? irb_.CreateICmpSLT(lhs, rhs)
                                       : irb_.CreateICmpULT(lhs, rhs);
    // 根据比较结果选择其中一个操作数作为结果值
    value_ = irb_.CreateSelect(icmp, lhs, rhs);
    // 函数返回
    return;
  }

  // 根据浮点数 lhs 是否未定义（NaN），选择返回 lhs 或者根据 lhs 与 rhs 的大小关系选择一个值
  value_ = irb_.CreateSelect(
      irb_.CreateFCmp(
          llvm::FCmpInst::FCMP_UNO,  // 比较 lhs 是否未定义（NaN）
          lhs,
          llvm::ConstantFP::get(lhs->getType(), 0.0)),  // 创建一个与 lhs 类型相匹配的常数 0.0
      lhs,  // 如果 lhs 未定义，则返回 lhs
      irb_.CreateSelect(
          irb_.CreateFCmp(llvm::FCmpInst::FCMP_OLT, lhs, rhs), lhs, rhs));  // 否则，根据 lhs 和 rhs 的大小关系选择一个值作为结果
// 定义 LLVMCodeGenImpl 类中的 visit 方法，用于处理 CompareSelectPtr 类型的节点
void LLVMCodeGenImpl::visit(CompareSelectPtr v) {
  // 生成无偏倾向的选择逻辑
  auto genUnbiased = [this, v]() -> llvm::Value* {
    // 处理左操作数并获取其值
    v->lhs()->accept(this);
    auto lhs = this->value_;
    // 处理右操作数并获取其值
    v->rhs()->accept(this);
    auto rhs = this->value_;
    // 处理第一个返回值并获取其值
    v->ret_val1()->accept(this);
    auto retval1 = this->value_;
    // 处理第二个返回值并获取其值
    v->ret_val2()->accept(this);
    auto retval2 = this->value_;

    // 获取操作数类型
    auto type_used = v->lhs()->dtype().scalar_type();

    llvm::Value* cmp_;
    // 根据操作数类型和比较操作类型创建比较指令
    CompareSelectOperation cmp_op_ = v->compare_select_op();
    if (c10::isIntegralType(type_used, true)) {
      cmp_ = irb_.CreateICmp(llvm_comparison_predicate(cmp_op_, type_used), lhs, rhs);
    } else if (c10::isFloatingType(type_used)) {
      cmp_ = irb_.CreateFCmp(llvm_fp_comparison_predicate(cmp_op_), lhs, rhs);
    } else {
      throw std::runtime_error("invalid type for CompareSelect");
    }

    // 根据比较结果选择返回值
    return irb_.CreateSelect(cmp_, retval1, retval2);
  };

  // 生成有偏倾向的选择逻辑
  auto genBiased = [this, v]() -> llvm::Value* {
    // 处理左操作数并获取其值
    v->lhs()->accept(this);
    auto lhs = this->value_;
    // 处理右操作数并获取其值
    v->rhs()->accept(this);
    auto rhs = this->value_;

    // 获取操作数类型和比较操作类型
    auto cmp_type = v->lhs()->dtype().scalar_type();
    auto cmp_op = v->compare_select_op();
    llvm::Value* cmp;

    // 根据操作数类型和比较操作类型创建比较指令
    if (c10::isIntegralType(cmp_type, true)) {
      cmp = irb_.CreateICmp(llvm_comparison_predicate(cmp_op, cmp_type), lhs, rhs);
    } else if (c10::isFloatingType(cmp_type)) {
      cmp = irb_.CreateFCmp(llvm_fp_comparison_predicate(cmp_op), lhs, rhs);
    } else {
      throw std::runtime_error("invalid type for CompareSelect");
    }

    // 处理多个子元素的情况
    auto lanes = v->lhs()->dtype().lanes();
    if (lanes > 1) {
      // 创建位掩码用于比较
      auto maskType = llvm::Type::getIntNTy(getContext(), lanes);
      auto zero = llvm::ConstantInt::get(maskType, 0);
      auto mask = irb_.CreateBitOrPointerCast(cmp, maskType);
      cmp = irb_.CreateICmpNE(mask, zero);
    }

    // 创建条件分支指令
    auto then_block = llvm::BasicBlock::Create(getContext(), "then", fn_);
    auto else_block = llvm::BasicBlock::Create(getContext(), "else", fn_);
    auto end_block = llvm::BasicBlock::Create(getContext(), "block", fn_);
    constexpr int32_t total_weight = 100000;
    auto true_weight = v->bias() == kLikely ? total_weight : 0;
    auto false_weight = total_weight - true_weight;
    irb_.CreateCondBr(cmp, then_block, else_block,
        llvm::MDBuilder(getContext()).createBranchWeights(true_weight, false_weight));

    // 设置当前插入点为 then 分支
    irb_.SetInsertPoint(then_block);
    // 处理第一个返回值并获取其值
    v->ret_val1()->accept(this);
    llvm::Value* then_val = value_;
    then_block = irb_.GetInsertBlock();
    irb_.CreateBr(end_block);

    // 设置当前插入点为 else 分支
    irb_.SetInsertPoint(else_block);
    // 处理第二个返回值并获取其值
    v->ret_val2()->accept(this);
    llvm::Value* else_val = value_;
    else_block = irb_.GetInsertBlock();
    irb_.CreateBr(end_block);

    // 设置当前插入点为结束分支
    irb_.SetInsertPoint(end_block);
    // 创建 PHI 节点用于合并两个分支的返回值
    llvm::PHINode* phi = irb_.CreatePHI(then_val->getType(), 2);
    phi->addIncoming(then_val, then_block);
    phi->addIncoming(else_val, else_block);
    return phi;
  };
    return phi;
  };



    // 返回 phi 变量的值作为函数的返回结果
    return phi;
  };



  value_ = v->bias() == kUnbiased ? genUnbiased() : genBiased();



  // 根据条件判断 v 对象的偏置状态，选择调用相应的生成函数
  // 如果 v 的偏置为 kUnbiased，则调用 genUnbiased() 函数
  // 否则调用 genBiased() 函数
  value_ = v->bias() == kUnbiased ? genUnbiased() : genBiased();
}

template <typename T>
// 当 T 是整数类型时，返回一个 llvm::Value* 类型，根据指定类型和值创建常量整数
typename std::enable_if<std::is_integral<T>::value, llvm::Value*>::type
getFromType(llvm::Type* type, T value) {
  return llvm::ConstantInt::get(type, value, std::is_signed<T>::value);
}

template <typename T>
// 当 T 是浮点数类型时，返回一个 llvm::Value* 类型，根据指定类型和值创建常量浮点数
typename std::enable_if<std::is_floating_point<T>::value, llvm::Value*>::type
getFromType(llvm::Type* type, T value) {
  return llvm::ConstantFP::get(type, value);
}

#define IMM_VISIT_DECLARE(Type, Name)                  \
  // 定义 LLVMCodeGenImpl 类中处理 Name##ImmPtr 类型的 visit 方法
  void LLVMCodeGenImpl::visit(Name##ImmPtr v) {        \
    // 设置当前值为根据 Type 类型和 v 的值创建的常量值
    value_ = getFromType<Type>(Name##Ty_, v->value()); \
  }
AT_FORALL_SCALAR_TYPES(IMM_VISIT_DECLARE);
#undef IMM_VISIT_DECLARE

// 处理 HalfImmPtr 类型的 visit 方法
void LLVMCodeGenImpl::visit(HalfImmPtr v) {
  // 设置当前值为根据 HalfTy_ 类型和 v 的值创建的常量浮点数
  value_ = llvm::ConstantFP::get(HalfTy_, v->value());
}

// 处理 BFloat16ImmPtr 类型的 visit 方法
void LLVMCodeGenImpl::visit(BFloat16ImmPtr v) {
  // 设置当前值为根据 ShortTy_ 类型和 v 的值的 x 成员创建的常量整数
  value_ = llvm::ConstantInt::get(ShortTy_, v->value().x);
}

// 处理 BoolImmPtr 类型的 visit 方法
void LLVMCodeGenImpl::visit(BoolImmPtr v) {
  // 设置当前值为根据 BoolTy_ 类型和 v 的值创建的常量整数
  value_ = llvm::ConstantInt::get(BoolTy_, v->value());
}

// 将 LLVM 类型 type 转换为具有指定数量 lanes 的矢量类型
static llvm::Type* llvmTypeToVec(llvm::Type* type, int lanes) {
  if (lanes > 1) {
    return llvm::VectorType::get(type, ElementCount(lanes));
  } else {
    return type;
  }
}

// 处理 CastPtr 类型的 visit 方法
void LLVMCodeGenImpl::visit(CastPtr v) {
  // 递归调用 src_value() 的 accept 方法，设置当前值
  v->src_value()->accept(this);

  auto dst_type = v->dtype().scalar_type();
  auto src_type = v->src_value()->dtype().scalar_type();
  bool is_to_bf16 = (dst_type == c10::kBFloat16);
  bool is_to_float = (dst_type == c10::kFloat);
  bool is_from_bf16 = (src_type == c10::kBFloat16);
  bool is_from_float = (src_type == c10::kFloat);

  bool cast_from_bf16_to_fp32 = is_from_bf16 && is_to_float;
  bool cast_from_fp32_to_bf16 = is_from_float && is_to_bf16;
  bool non_bf16_cast = (!is_to_bf16) && (!is_from_bf16);
  bool valid_bf16_cast = cast_from_bf16_to_fp32 || cast_from_fp32_to_bf16;

  // 检查是否为有效的 BF16 类型转换，否则抛出异常
  TORCH_CHECK(
      valid_bf16_cast || non_bf16_cast,
      "Cast is not implemented for the conversion between ",
      src_type,
      " and ",
      dst_type,
      ".");

  // 获取目标类型的 LLVM 类型
  llvm::Type* dstType =
      llvmTypeToVec(dtypeToLLVM(v->dtype()), v->dtype().lanes());
  // 获取源类型的 LLVM 类型
  llvm::Type* srcType = dtypeToLLVM(v->src_value()->dtype());

  // 如果源类型和目标类型相同，不进行任何操作
  if (srcType == dstType) {
    // do nothing.
    return;
  }

  // 判断是否为无符号类型
  bool destUnsigned = v->dtype().scalar_type() == ScalarType::Byte ||
      v->dtype().scalar_type() == ScalarType::QUInt8 ||
      v->dtype().scalar_type() == ScalarType::Bool;
  bool srcUnsigned =
      v->src_value()->dtype().scalar_type() == ScalarType::Byte ||
      v->src_value()->dtype().scalar_type() == ScalarType::QUInt8 ||
      v->src_value()->dtype().scalar_type() == ScalarType::Bool;

  // 标量类型的类型转换
  if (is_from_bf16) {
    // 将 BF16 值左移 16 位，然后位转换为 FP32
    //   FP32_VAL = BF16_VAL << 16
    auto lans = v->dtype().lanes();
    value_ = irb_.CreateZExt(value_, llvmTypeToVec(IntTy_, lans));
    auto vec_shl_val = toVec(llvm::ConstantInt::get(IntTy_, 16), lans);
    value_ = irb_.CreateShl(value_, vec_shl_val);
  if (is_to_bf16) {
    // 如果需要转换为BF16格式
    // Convert the FP32 value by RNE(Rounding to Nearest Even). Algorithm is as
    // follows:
    //   STEP1: U32_VAL = BITCAST(F32_VAL)
    //   将FP32值转换为无符号32位整数U32_VAL
    //   STEP2: U32_VAL_TMP = U32_VAL >> 16
    //   将U32_VAL右移16位得到U32_VAL_TMP
    //   STEP3: U32_VAL_TMP = U32_VAL_TMP & 1
    //   将U32_VAL_TMP与1进行按位与运算
    //   STEP4: ROUNDING_BIAS = U32_VAL_TMP + UINT32(0x7FFF)
    //   设置舍入偏置为U32_VAL_TMP加上0x7FFF
    //   STEP5: U32_VAL_TMP = U32_VAL + ROUNDING_BIAS
    //   将U32_VAL_TMP与舍入偏置相加得到新的U32_VAL_TMP
    //   STEP6: BF16_VAL = static_cast<UINT16>(U32_VAL_TMP >> 16)
    //   将U32_VAL_TMP右移16位并截断为16位无符号整数，得到BF16值
    auto lans = v->src_value()->dtype().lanes();
    auto shift_len = llvm::ConstantInt::get(IntTy_, 16);
    auto one = llvm::ConstantInt::get(ShortTy_, 1);
    auto rounding_bias = llvm::ConstantInt::get(ShortTy_, 0x7FFF);
    auto bf16_nan = llvm::ConstantInt::get(ShortTy_, 0xFFFF);

    // 判断是否为NaN
    auto mask = irb_.CreateFCmpOEQ(value_, value_);
    // STEP1: U32_VAL = BITCAST(F32_VAL)
    auto fp32_i32_value =
        irb_.CreateBitOrPointerCast(value_, llvmTypeToVec(IntTy_, lans));
    // STEP2: U32_VAL_TMP = (U32_VAL >> 16)
    value_ = irb_.CreateLShr(fp32_i32_value, toVec(shift_len, lans));
    value_ = irb_.CreateTrunc(value_, llvmTypeToVec(ShortTy_, lans));
    // STEP3: U32_VAL_TMP = U32_VAL_TMP & 1
    value_ = irb_.CreateAnd(value_, toVec(one, lans));
    // STEP4: ROUNDING_BIAS = U32_VAL_TMP + UINT32(0x7FFF)
    value_ = irb_.CreateAdd(value_, toVec(rounding_bias, lans));
    value_ = irb_.CreateZExt(value_, llvmTypeToVec(IntTy_, lans));
    // STEP5: U32_VAL_TMP = U32_VAL + ROUNDING_BIAS
    value_ = irb_.CreateAdd(value_, fp32_i32_value);
    // STEP6: BF16_VAL = static_cast<UINT16>(U32_VAL_TMP >> 16)
    value_ = irb_.CreateLShr(value_, toVec(shift_len, lans));
    value_ = irb_.CreateTrunc(value_, llvmTypeToVec(ShortTy_, lans));
    value_ = irb_.CreateBitOrPointerCast(value_, llvmTypeToVec(ShortTy_, lans));
    // 如果值为NaN，则返回BF16的NaN值
    value_ = irb_.CreateSelect(mask, value_, toVec(bf16_nan, lans));
    return;
  }

  if (srcType->isFPOrFPVectorTy()) {
    if (dstType->isFPOrFPVectorTy()) {
      // 如果源类型和目标类型都是浮点数或浮点向量类型
      // as with eager, convert from Double -> Half by Converting to Float then
      // Half. TODO: __truncdfhf2
      // 类似于急切模式，通过先转换为浮点数再转换为半精度浮点数，将Double类型转换为Half类型
      if (v->dtype().scalar_type() == ScalarType::Half &&
          v->src_value()->dtype().scalar_type() == ScalarType::Double) {
        value_ = irb_.CreateFPCast(
            value_, llvmTypeToVec(FloatTy_, v->dtype().lanes()));
      }
      // 将值转换为目标类型
      value_ = irb_.CreateFPCast(value_, dstType);
      // 立即返回
      return;
    }
  }
    } else if (dstType->isIntOrIntVectorTy()) {
      // 如果目标类型是整数或整数向量，执行以下操作
      // 严格从浮点数到整数类型的转换通常不会产生正确的结果
      // 如果输入的浮点数不为0，则设置一个位为真
      if (v->dtype().scalar_type() == ScalarType::Bool) {
        // 创建一个与输入类型相同的浮点常数0
        llvm::Value* zero =
            toVec(llvm::ConstantFP::get(srcType, 0.), v->dtype().lanes());
        // 比较当前值与零的不等性，生成一个布尔值
        value_ = irb_.CreateFCmp(llvm::FCmpInst::FCMP_UNE, value_, zero);
        // 将布尔值与整数1进行比较，得到一个整数值
        value_ = irb_.CreateICmpEQ(
            value_, llvm::ConstantInt::get(value_->getType(), 1));
        // 将生成的整数值转换为目标整数类型
        value_ = irb_.CreateIntCast(value_, dstType, !destUnsigned);
        return;
      }

      // 如果目标类型是无符号整数，将当前值从浮点数转换为无符号整数类型
      if (destUnsigned) {
        value_ = irb_.CreateFPToUI(value_, dstType);
      } else {
        // 否则，将当前值从浮点数转换为有符号整数类型
        value_ = irb_.CreateFPToSI(value_, dstType);
      }
    } else {
      // 如果目标类型既不是浮点数也不是整数类型，则抛出未实现的降低错误
      throw unimplemented_lowering(v);
    }
    return;
  }

  // 如果源类型既不是整数也不是整数向量类型，则抛出未实现的降低错误
  if (!srcType->isIntOrIntVectorTy()) {
    throw unimplemented_lowering(v);
  }
  // 如果目标类型是浮点数或浮点数向量类型
  if (dstType->isFPOrFPVectorTy()) {
    // 如果源类型是无符号整数，将当前值从无符号整数转换为浮点数类型
    if (srcUnsigned) {
      value_ = irb_.CreateUIToFP(value_, dstType);
    } else {
      // 否则，将当前值从有符号整数转换为浮点数类型
      value_ = irb_.CreateSIToFP(value_, dstType);
    }
  } else if (dstType->isIntOrIntVectorTy()) {
    // 如果源类型是布尔类型
    // 确保布尔真值精确为一，因为我们通过零扩展int8进行从布尔到整数的转换
    if (v->dtype().scalar_type() == ScalarType::Bool) {
      // 创建一个与源类型相同的整数常数0
      llvm::Value* zero =
          toVec(llvm::ConstantInt::get(srcType, 0), v->dtype().lanes());
      // 比较当前值与零的不等性，得到一个布尔值
      value_ = irb_.CreateICmpNE(value_, zero);
    }
    // 将当前值转换为目标整数类型，根据目标类型是否为无符号选择是否零扩展
    value_ = irb_.CreateIntCast(value_, dstType, !destUnsigned);
  } else {
    // 如果目标类型既不是浮点数也不是整数类型，则抛出未实现的降低错误
    throw unimplemented_lowering(v);
  }
}

void LLVMCodeGenImpl::visit(BitCastPtr v) {
  // 访问源值，并递归地接受访问器
  v->src_value()->accept(this);

  // 将目标类型转换为 LLVM 类型
  llvm::Type* dstType = dtypeToLLVM(v->dtype());
  // 如果数据类型的通道数大于1，创建一个 LLVM 向量类型
  if (v->dtype().lanes() > 1) {
    dstType = llvm::VectorType::get(dstType, ElementCount(v->dtype().lanes()));
  }
  // 将源类型转换为 LLVM 类型
  llvm::Type* srcType = dtypeToLLVM(v->src_value()->dtype());

  // 如果源类型与目标类型相同，无需操作，直接返回
  if (srcType == dstType) {
    // 什么也不做。
    return;
  }

  // 检查是否可以进行位或指针转换
  TORCH_CHECK(llvm::CastInst::isBitCastable(
      srcType->getScalarType(), dstType->getScalarType()));
  // 创建位或指针转换指令
  value_ = irb_.CreateBitOrPointerCast(value_, dstType);
}

void LLVMCodeGenImpl::visit(VarPtr v) {
  // 将变量转换为 LLVM 值
  value_ = varToValue(v);
}

llvm::Value* LLVMCodeGenImpl::varToValue(VarPtr v) {
  // 变量可能同时存在于 varToVal_ 和 varToArgs 中，varToVal_ 优先。
  if (varToVal_.count(v)) {
    return varToVal_.at(v);
  } else if (varToArg_.count(v)) {
    // 如果在 varToArg_ 中找到变量，则返回相应的函数参数
    auto idx = varToArg_.at(v);
    auto arg = fn_->arg_begin() + idx;
    return arg;
  }
  // 如果找不到对应的值，则返回空指针
  return nullptr;
}

void LLVMCodeGenImpl::replaceVarMapping(
    const std::vector<VarPtr>& vars,
    const std::vector<llvm::Value*>& vals) {
  // 确保变量和值的数量相等
  TORCH_CHECK(vars.size() == vals.size());
  // 替换变量到值的映射关系
  for (const auto i : c10::irange(vars.size())) {
    VarPtr var = vars[i];
    llvm::Value* val = vals[i];
    if (val) {
      varToVal_[var] = val;
    } else {
      varToVal_.erase(var);
    }
  }
}

void LLVMCodeGenImpl::visit(RampPtr v) {
  // 访问基础值并接受访问器
  v->base()->accept(this);
  auto base = this->value_;
  // 访问步长并接受访问器
  v->stride()->accept(this);
  auto stride = this->value_;
  int lanes = v->lanes();

  // 如果步长是常量整数
  if (llvm::ConstantInt* const_stride =
          llvm::dyn_cast<llvm::ConstantInt>(stride)) {
    // 创建常量向量偏移量
    std::vector<llvm::Constant*> vals = {
        llvm::ConstantInt::get(base->getType(), 0)};
    for (int i = 1; i < lanes; ++i) {
      vals.push_back(llvm::ConstantExpr::getAdd(vals.back(), const_stride));
    }

    // 创建向量值和偏移量的加法操作
    llvm::Value* offsets = llvm::ConstantVector::get(vals);
    llvm::Value* splat = irb_.CreateVectorSplat(lanes, base);
    value_ = irb_.CreateAdd(splat, offsets);
    return;
  }

  // 否则，根据数据类型的标量类型创建向量类型
  llvm::Type* vecType = nullptr;
  auto element_count = ElementCount(lanes);
  switch (v->dtype().scalar_type()) {
#define TYPE_CASE(_1, Name)                                    \
  case ScalarType::Name:                                       \
    vecType = llvm::VectorType::get(Name##Ty_, element_count); \
    break;
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::QInt8:
      vecType = llvm::VectorType::get(CharTy_, element_count);
      break;
    case ScalarType::QUInt8:
      vecType = llvm::VectorType::get(ByteTy_, element_count);
      break;
    case ScalarType::BFloat16:
      vecType = llvm::VectorType::get(ShortTy_, element_count);
      break;
    default:
      // 如果数据类型无效，抛出运行时错误
      throw std::runtime_error("invalid dtype in Ramp");
  }

  // 创建未定义的向量值
  value_ = llvm::UndefValue::get(vecType);
  // 循环插入基础值和步长
  for (int i = 0; i < lanes; ++i) {
    value_ = irb_.CreateInsertElement(value_, base, i);
    base = irb_.CreateAdd(base, stride);
  }
}
llvm::Value* LLVMCodeGenImpl::emitUnmaskedLoad(
    llvm::Type* ty,
    llvm::Value* base,
    llvm::Value* idx) {
#if LLVM_VERSION_MAJOR >= 15
  // 创建基于指针索引的地址计算
  auto addr = irb_.CreateGEP(ty, base, idx);
  // 创建加载指令并返回加载的值
  return irb_.CreateLoad(ty, addr);
#else
  // 根据基类型获取标量类型的指针元素类型，并创建基于其的地址计算
  auto addr = irb_.CreateGEP(
      base->getType()->getScalarType()->getPointerElementType(), base, idx);
  // 根据地址的指针元素类型创建加载指令，并返回加载的值
  return irb_.CreateLoad(addr->getType()->getPointerElementType(), addr);
#endif
}

llvm::Value* LLVMCodeGenImpl::emitMaskedLoad(
    llvm::Type* ty,
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* mask) {
  // 为掩码加载创建基本块结构
  auto preheader = irb_.GetInsertBlock();
  auto condblock = llvm::BasicBlock::Create(getContext(), "cond", fn_);
  auto tailblock = llvm::BasicBlock::Create(getContext(), "tail", fn_);

  // 测试掩码条件
  auto cond = irb_.CreateICmpEQ(mask, llvm::ConstantInt::get(IntTy_, 1));
  // 根据条件跳转到相应的基本块
  irb_.CreateCondBr(cond, condblock, tailblock);

  // 在条件块中执行加载操作
  irb_.SetInsertPoint(condblock);

#if LLVM_VERSION_MAJOR >= 15
  // 创建基于指针索引的地址计算
  auto addr = irb_.CreateGEP(ty, base, idx);
  // 创建加载指令
  auto load = irb_.CreateLoad(ty, addr);
#else
  // 根据基类型获取标量类型的指针元素类型，并创建基于其的地址计算
  auto addr = irb_.CreateGEP(
      base->getType()->getScalarType()->getPointerElementType(), base, idx);
  // 根据地址的指针元素类型创建加载指令
  auto load = irb_.CreateLoad(addr->getType()->getPointerElementType(), addr);
#endif

  // 在条件块中创建跳转指令到尾块
  irb_.CreateBr(tailblock);

  // 合并掩码加载和非掩码加载的控制流
  irb_.SetInsertPoint(tailblock);
  // 创建 PHI 节点以便合并两条路径上的值
  auto phi = irb_.CreatePHI(load->getType(), 2);
  // 添加非掩码路径上的未定义值
  phi->addIncoming(llvm::UndefValue::get(load->getType()), preheader);
  // 添加掩码路径上的加载值
  phi->addIncoming(load, condblock);

  return phi;
}

void LLVMCodeGenImpl::visit(LoadPtr v) {
  // 处理单个元素加载情况
  if (v->dtype().lanes() == 1) {
    // 访问基地址，并存储其值
    v->base_handle()->accept(this);
    auto base = this->value_;
    // 访问扁平化索引，并存储其值
    v->flat_index()->accept(this);
    auto idx = this->value_;
    // 调用无掩码加载函数，存储结果
    value_ = emitUnmaskedLoad(dtypeToLLVM(v->dtype()), base, idx);
    return;
  }

  llvm::Type* loadType = nullptr;

  // 计算向量化加载类型
  auto element_count = ElementCount(v->dtype().lanes());
  switch (v->dtype().scalar_type()) {
#define TYPE_CASE(_1, Name)                                     \
  case ScalarType::Name:                                        \
    // 根据数据类型创建对应的向量类型
    loadType = llvm::VectorType::get(Name##Ty_, element_count); \
    break;
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::QInt8:
      loadType = llvm::VectorType::get(CharTy_, element_count);
      break;
    case ScalarType::QUInt8:
      loadType = llvm::VectorType::get(ByteTy_, element_count);
      break;
    case ScalarType::BFloat16:
      loadType = llvm::VectorType::get(ShortTy_, element_count);
      break;
    default:
      // 抛出异常，指示加载中的无效数据类型
      throw std::runtime_error("invalid dtype in Load");
  }

  // 处理连续且无掩码加载的情况
  auto idx_ramp = to<Ramp>(v->flat_index());
  if (idx_ramp) {
    // 获取步长的整数值
    auto stride_imm = intValue(idx_ramp->stride());
    // 如果步长为1，则执行简单的无掩码加载
    if (stride_imm == 1) {
      // 访问基地址，并存储其值
      v->base_handle()->accept(this);
      auto base = this->value_;
      // 访问扁平化索引，并存储其值
      v->flat_index()->accept(this);
      auto idx = this->value_;
      // 调用无掩码加载函数，存储结果
      value_ = emitUnmaskedLoad(loadType, base, idx);
      return;
    }
  }

  // 处理非连续或有掩码的加载情况
  // 略...
}
    // 检查 stride_imm 是否非空且其值为1
    if (stride_imm && *stride_imm == 1) {
      // 访问 v 对象的 base_handle() 方法，并接受其访问
      v->base_handle()->accept(this);
      // 将当前对象的值赋给 base
      auto base = this->value_;
      // 访问 idx_ramp 对象的 base() 方法，并接受其访问
      idx_ramp->base()->accept(this);
      // 将当前对象的值赋给 first_idx
      auto first_idx = this->value_;
#if LLVM_VERSION_MAJOR >= 15
// 如果 LLVM 主版本号大于或等于 15，使用 CreateGEP 创建地址指针
auto addr = irb_.CreateGEP(dtypeToLLVM(v->dtype()), base, first_idx);
#else
// 如果 LLVM 主版本号小于 15，使用 CreateGEP 创建地址指针，需额外指定基类型的指针元素类型
auto addr = irb_.CreateGEP(
    base->getType()->getScalarType()->getPointerElementType(),
    base,
    first_idx);
#endif

// 创建地址指针的位或指针转换
auto vaddr = irb_.CreateBitOrPointerCast(
    addr, llvm::PointerType::get(loadType, 0));

#if LLVM_VERSION_MAJOR >= 12
// 如果 LLVM 主版本号大于或等于 12，使用指定对齐方式的加载操作
value_ = irb_.CreateAlignedLoad(loadType, vaddr, llvm::MaybeAlign(4));
#else
// 如果 LLVM 主版本号小于 12，使用指定对齐方式的加载操作，对齐值为 4
value_ = irb_.CreateAlignedLoad(loadType, vaddr, 4);
#endif

return;
}

// Fallback to a scalar implementation
v->base_handle()->accept(this);
auto base = this->value_;
v->flat_index()->accept(this);
auto idx = this->value_;

llvm::Value* load = llvm::UndefValue::get(loadType);
for (int i = 0; i < v->dtype().lanes(); ++i) {
  auto sub_idx = irb_.CreateExtractElement(idx, i);
  llvm::Value* sub_load = nullptr;
  // 发出未掩码加载的操作
  sub_load = emitUnmaskedLoad(dtypeToLLVM(v->dtype()), base, sub_idx);
  // 创建插入元素操作
  load = irb_.CreateInsertElement(load, sub_load, i);
}

value_ = load;
}

#if LLVM_VERSION_MAJOR >= 15
// 将参数打包成一个聚合结构以进行转发
TypedPointer LLVMCodeGenImpl::packFuncArgs(
    const std::vector<llvm::Value*>& func_args) {
  if (func_args.empty()) {
    llvm::PointerType* VoidPtrType = llvm::PointerType::getUnqual(getContext());
    // 返回空指针常量作为 TypedPointer
    return TypedPointer(
        VoidPtrType, llvm::ConstantPointerNull::get(VoidPtrType));
  }
  std::vector<llvm::Type*> arg_types(func_args.size());
  for (const auto i : c10::irange(func_args.size())) {
    // 填充参数类型数组
    arg_types[i] = func_args[i]->getType();
  }
  // 创建结构体类型
  llvm::StructType* packed_type = llvm::StructType::create(arg_types);
  llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
  llvm::Value* one = llvm::ConstantInt::get(IntTy_, 1);
  // 分配结构体类型的内存空间
  llvm::Value* packed = irb_.CreateAlloca(packed_type, one);
  for (const auto i : c10::irange(func_args.size())) {
    // 创建基于范围的指针运算，用于访问结构体成员
    llvm::Value* dst_ptr = irb_.CreateInBoundsGEP(
        packed_type, packed, {zero, llvm::ConstantInt::get(IntTy_, i)});
    // 创建存储操作，将参数值存入结构体成员位置
    irb_.CreateStore(func_args[i], dst_ptr);
  }
  // 返回 TypedPointer 对象
  return TypedPointer(packed_type, packed);
}

// 解包聚合结构到单独的参数
std::vector<llvm::Value*> LLVMCodeGenImpl::unpackFuncArgs(
    TypedPointer packed,
    int arg_count) {
  // TODO: 从 packed 中提取 arg_count
  std::vector<llvm::Value*> func_args(arg_count);
  llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
  for (const auto i : c10::irange(arg_count)) {
    // 获取结构体元素类型
    llvm::Type* feild_type = packed.type->getStructElementType(i);
    // 创建基于范围的指针运算，用于访问结构体成员
    llvm::Value* feild_addr = irb_.CreateInBoundsGEP(
        packed.type, packed.addr, {zero, llvm::ConstantInt::get(IntTy_, i)});
    // 创建加载操作，从结构体成员位置加载值
    func_args[i] = irb_.CreateLoad(feild_type, feild_addr);
  }
  // 返回解包后的参数向量
  return func_args;
}
#else
// 如果 LLVM 主版本号小于 15，实现参数打包成一个聚合结构的函数
llvm::Value* LLVMCodeGenImpl::packFuncArgs(
    const std::vector<llvm::Value*>& func_args) {
  if (func_args.empty()) {
    // 如果参数为空，返回空指针常量
    llvm::PointerType* VoidPtrType = llvm::PointerType::getUnqual(getContext());
    return llvm::ConstantPointerNull::get(VoidPtrType);

    // 如果参数为空，返回空指针常量
    llvm::PointerType* VoidPtrType = llvm::PointerType::getUnqual(getContext());
    return llvm::ConstantPointerNull::get(VoidPtrType);
  }
  // 创建参数类型向量
  std::vector<llvm::Type*> arg_types(func_args.size());
  for (const auto i : c10::irange(func_args.size())) {
    // 填充参数类型数组
    arg_types[i] = func_args[i]->getType();
  }
  // 创建结构体类型
  llvm::StructType* packed_type = llvm::StructType::create(arg_types);
  llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
  llvm::Value* one = llvm::ConstantInt::get(IntTy_, 1);
  // 分配结构体类型的内存空间
  llvm::Value* packed = irb_.CreateAlloca(packed_type, one);
  for (const auto i : c10::irange(func_args.size())) {
    // 创建基于范围的指针运算，用于访问结构体成员
    llvm::Value* dst_ptr = irb_.CreateInBoundsGEP(
        packed_type, packed, {zero, llvm::ConstantInt::get(IntTy_, i)});
    // 创建存储操作，将参数值存入结构体成员位置
    irb_.CreateStore(func_args[i], dst_ptr);
  }
  // 返回结构体的指针作为函数返回值
  return packed;
}
#endif
    # 创建一个指向 void 指针类型的指针类型对象 VoidPtrType
    llvm::PointerType* VoidPtrType = llvm::Type::getInt8PtrTy(getContext());
    
    # 创建一个空指针常量 NullPtr，其类型为 VoidPtrType
    llvm::Constant* NullPtr = llvm::ConstantPointerNull::get(VoidPtrType);
    
    # 返回空指针常量 NullPtr
    return NullPtr;
    }
    
    # 创建一个大小为 func_args.size() 的 llvm::Type 类型的向量 arg_types
    std::vector<llvm::Type*> arg_types(func_args.size());
    
    # 遍历 func_args 容器中的每一个元素
    for (const auto i : c10::irange(func_args.size())) {
        # 将 func_args[i] 的类型存入 arg_types[i] 中
        arg_types[i] = func_args[i]->getType();
    }
    
    # 创建一个结构体类型 packed_type，其成员类型由 arg_types 决定
    llvm::StructType* packed_type = llvm::StructType::create(arg_types);
    
    # 创建一个表示整数类型 IntTy_ 中值为 0 的常量 zero
    llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
    
    # 创建一个表示整数类型 IntTy_ 中值为 1 的常量 one
    llvm::Value* one = llvm::ConstantInt::get(IntTy_, 1);
    
    # 在当前插入点 irb_ 上创建一个大小为 packed_type 的内存分配操作 packed
    llvm::Value* packed = irb_.CreateAlloca(packed_type, one);
    
    # 遍历 func_args 容器中的每一个元素
    for (const auto i : c10::irange(func_args.size())) {
        # 创建一个指向 packed 中第 i 个元素的指针 dst_ptr
        llvm::Value* dst_ptr = irb_.CreateInBoundsGEP(
            packed_type, packed, {zero, llvm::ConstantInt::get(IntTy_, i)});
        
        # 将 func_args[i] 的值存储到 dst_ptr 指向的位置
        irb_.CreateStore(func_args[i], dst_ptr);
    }
    
    # 返回 packed 结构体类型的值
    return packed;
// Unpack the aggregate struct into individual arguments.
std::vector<llvm::Value*> LLVMCodeGenImpl::unpackFuncArgs(
    llvm::Value* packed,
    int arg_count) {
  // TODO: extract arg_count from packed.
  // 创建一个存储函数参数的向量，大小为arg_count
  std::vector<llvm::Value*> func_args(arg_count);
  // 创建一个常量0
  llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
  // 遍历arg_count次，进行参数解包
  for (const auto i : c10::irange(arg_count)) {
    // 获取packed指针的元素类型
    llvm::Type* packed_type = packed->getType()->getPointerElementType();
    // 创建一个指向目标位置的指针，使用CreateInBoundsGEP函数
    llvm::Value* dst_ptr = irb_.CreateInBoundsGEP(
        packed_type, packed, {zero, llvm::ConstantInt::get(IntTy_, i)});
    // 将目标位置的值加载到func_args[i]中
    func_args[i] =
        irb_.CreateLoad(dst_ptr->getType()->getPointerElementType(), dst_ptr);
  }
  // 返回解包后的函数参数向量
  return func_args;
}
#endif

// Lower the parallel for-loop.
// * Move the body into its own closure.
// * Identify var across the boundary into arguments and forward them.
// * Send the closure and range to the dispatcher for execution.
void LLVMCodeGenImpl::processParallelFor(ForPtr v) {
  // Create "start" and "stop" values.
  // 访问并获取循环起始值和结束值
  v->start()->accept(this);
  auto start = this->value_;
  v->stop()->accept(this);
  auto stop = this->value_;

  // The Vars that need to be forward in the body closure.
  // 存储需要在闭包中进行传递的变量
  std::vector<VarPtr> body_arg_vars;
  // Corresponding Value* that was used in the old body for the caller.
  // 用于调用者的旧体中使用的对应Value*
  std::vector<llvm::Value*> body_caller_vals;
  // Corresponding Value* that will be used in the new body closure.
  // 将在新闭包体中使用的对应Value*
  std::vector<llvm::Value*> body_closure_args;

  // Identify the VarPtr used in the body, and generated outside.
  // 查找在循环体中使用的VarPtr及其在外部生成的变量
  VarFinder var_finder;
  v->body()->accept(&var_finder);
  auto& vars = var_finder.vars();
  for (auto& var : vars) {
    // 如果可以找到var对应的llvm::Value*，则将其添加到相应的列表中
    if (llvm::Value* value = varToValue(var)) {
      body_arg_vars.push_back(var);
      body_caller_vals.push_back(value);
    }
  }

  // Pack the arguments in an automatic variable for forwarding.
  // 将参数打包到自动变量中以进行传递
#if LLVM_VERSION_MAJOR >= 15
  TypedPointer packData = packFuncArgs(body_caller_vals);
  llvm::Value* packed_caller_args = packData.addr;
#else
  llvm::Value* packed_caller_args = packFuncArgs(body_caller_vals);
#endif
  // 记录进入新函数之前的插入点
  llvm::BasicBlock* old_insert_block = irb_.GetInsertBlock();

  // Create the new body closure code.
  // 创建新的闭包体代码
#if LLVM_VERSION_MAJOR >= 15
  auto func_type =
      llvm::FunctionType::get(VoidTy_, {LongTy_, OpqPtrTy_}, false);
#else
  auto func_type =
      llvm::FunctionType::get(VoidTy_, {LongTy_, Int8PtrTy_}, false);
#endif

  llvm::Function* func = llvm::Function::Create(
      func_type, llvm::Function::PrivateLinkage, "func", module_.get());
  auto func_body = llvm::BasicBlock::Create(getContext(), "func_body", func);
  irb_.SetInsertPoint(func_body);
  auto args = func->arg_begin();
  llvm::Value* index = args++;
  llvm::Value* packed_func_args_raw = args++;
  llvm::Value* packed_func_args = irb_.CreatePointerCast(
      packed_func_args_raw, packed_caller_args->getType());

  // Unpack the arguments from the opaque buffer.
  // 从不透明缓冲区中解包参数
  if (v->var()->dtype().scalar_type() != c10::kLong) {
    // 调用 irb_ 对象的 CreateIntCast 方法，将 index 转换为 LLVM 中对应变量类型的整数类型
    index = irb_.CreateIntCast(
        index, dtypeToLLVM(v->var()->dtype()), v->var()->dtype().is_signed());
  }
#if LLVM_VERSION_MAJOR >= 15
  // 根据 LLVM 主版本号判断使用不同的参数解包方式
  body_closure_args =
      unpackFuncArgs({packData.type, packed_func_args}, body_arg_vars.size());
#else
  // 使用旧版本的参数解包方式
  body_closure_args = unpackFuncArgs(packed_func_args, body_arg_vars.size());
#endif

  // 设置代码生成器的变量到值的映射，将变量 v 的值映射到 index
  varToVal_[v->var()] = index;

  // 替换变量映射，将 body_arg_vars 替换为 body_closure_args
  replaceVarMapping(body_arg_vars, body_closure_args);

  // 保存旧的函数指针，并将当前函数指针更新为新的函数
  llvm::Function* old_fn = fn_;
  fn_ = func;

  // 如果 v 的 body 存在，遍历并接受其访问
  if (v->body()) {
    v->body()->accept(this);
  }

  // 恢复回之前保存的旧函数指针
  fn_ = old_fn;

  // 在当前插入点创建一个空的返回语句
  irb_.CreateRet(nullptr);

  // 替换变量映射，将 body_arg_vars 替换为 body_caller_vals
  replaceVarMapping(body_arg_vars, body_caller_vals);

  // 从 varToVal_ 中移除变量 v 的映射
  varToVal_.erase(v->var());

  // 设置插入点回到原始的代码块，并生成调用者代码
  irb_.SetInsertPoint(old_insert_block);

#if LLVM_VERSION_MAJOR >= 15
  // 根据 LLVM 主版本号判断使用不同的指针类型转换
  llvm::Value* packed_caller_args_ptr =
      irb_.CreatePointerCast(packed_caller_args, OpqPtrTy_);
  llvm::Value* func_value = irb_.CreatePointerCast(func, OpqPtrTy_);
  llvm::FunctionType* dispatcher_fntype = llvm::FunctionType::get(
      VoidTy_, {OpqPtrTy_, LongTy_, LongTy_, OpqPtrTy_}, false);
#else
  // 使用旧版本的指针类型转换
  llvm::Value* packed_caller_args_ptr =
      irb_.CreatePointerCast(packed_caller_args, Int8PtrTy_);
  llvm::Value* func_value = irb_.CreatePointerCast(func, Int8PtrTy_);
  llvm::FunctionType* dispatcher_fntype = llvm::FunctionType::get(
      VoidTy_, {Int8PtrTy_, LongTy_, LongTy_, Int8PtrTy_}, false);
#endif

  // 获取或插入名为 "DispatchParallel" 的函数
  FunctionCallee dispatcher_callee =
      module_->getOrInsertFunction("DispatchParallel", dispatcher_fntype);
  llvm::Function* dispatcher =
      llvm::cast<llvm::Function>(dispatcher_callee.getCallee());

  // 为调度函数添加无异常属性
  dispatcher->addFnAttr(llvm::Attribute::NoUnwind);

  // 将 start 和 stop 转换为长整型，并创建调用指令调用调度函数
  start = irb_.CreateIntCast(start, LongTy_, true);
  stop = irb_.CreateIntCast(stop, LongTy_, true);
  irb_.CreateCall(
      dispatcher, {func_value, start, stop, packed_caller_args_ptr});

  // 将 value_ 设置为整数常量 0
  value_ = llvm::ConstantInt::get(IntTy_, 0);
}
  // 抛出异常，指示在此之前变量不应该存在
  throw std::runtime_error("var should not exist before");
}

// 创建循环体和退出块。
auto body = llvm::BasicBlock::Create(getContext(), "body", fn_);
auto exit = llvm::BasicBlock::Create(getContext(), "exit", fn_);

// 创建停止条件。
auto cond = irb_.CreateICmpSLT(idx, stop);
irb_.CreateCondBr(cond, body, exit);

// 生成循环体的代码。
irb_.SetInsertPoint(body);
if (v->body()) {
  // 如果存在循环体，则生成其代码。
  v->body()->accept(this);
}
// 如果生成了嵌套控制流，则可能会更改"body"块。
body = irb_.GetInsertBlock();

// 增加索引变量并跳转回循环测试。
auto inc = irb_.CreateAdd(idx, llvm::ConstantInt::getSigned(start->getType(), 1));
irb_.CreateBr(condBlock);
idx->addIncoming(inc, body);

// 退出循环。
irb_.SetInsertPoint(exit);

// 从变量到值的映射中移除变量。
varToVal_.erase(v->var());
// 将值设为常量整数0。
value_ = llvm::ConstantInt::get(IntTy_, 0);
}

// 处理 BlockPtr 类型的访问，包括在作用域内执行每个语句以及清理作用域中的变量
void LLVMCodeGenImpl::visit(BlockPtr v) {
  // 保存当前作用域，将作用域设置为当前 BlockPtr
  BlockPtr last = scope_;
  scope_ = v;

  // 遍历当前 BlockPtr 中的每个语句，并让其接受当前访问者的访问
  for (StmtPtr s : *v) {
    s->accept(this);
  }

  // 恢复之前保存的作用域
  scope_ = last;

  // 查找当前 BlockPtr 对应的变量集合，清理掉对应的变量值
  auto it = scopeToVar_.find(v);
  if (it != scopeToVar_.end()) {
    for (VarPtr e : it->second) {
      // 尝试从 varToVal_ 中删除变量 e 的值，如果失败则抛出运行时错误
      if (varToVal_.erase(e) != 1) {
        throw std::runtime_error("erasing var that doesn't exist");
      }
    }
  }
}

// 发出未掩码存储指令
void LLVMCodeGenImpl::emitUnmaskedStore(
    llvm::Type* ty,
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* val) {
#if LLVM_VERSION_MAJOR >= 15
  // 创建基于类型、基址和索引的指针地址
  auto addr = irb_.CreateGEP(ty, base, idx);
#else
  // 为低版本 LLVM 创建指针地址，根据基址的标量类型获取指针元素类型
  auto addr = irb_.CreateGEP(
      base->getType()->getScalarType()->getPointerElementType(), base, idx);
#endif

  // 创建存储指令，将值 val 存储到地址 addr
  irb_.CreateStore(val, addr);
}

// 发出掩码存储指令
void LLVMCodeGenImpl::emitMaskedStore(
    llvm::Type* ty,
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* mask,
    llvm::Value* val) {
  // 创建用于掩码存储的基本块结构
  auto condblock = llvm::BasicBlock::Create(getContext(), "cond", fn_);
  auto tailblock = llvm::BasicBlock::Create(getContext(), "tail", fn_);

  // 测试掩码条件
  auto cond = irb_.CreateICmpEQ(mask, llvm::ConstantInt::get(IntTy_, 1));
  // 根据条件跳转到相应的基本块
  irb_.CreateCondBr(cond, condblock, tailblock);

  // 在条件块中设置插入点
  irb_.SetInsertPoint(condblock);

#if LLVM_VERSION_MAJOR >= 15
  // 创建基于类型、基址和索引的指针地址
  auto addr = irb_.CreateGEP(ty, base, idx);
#else
  // 为低版本 LLVM 创建指针地址，根据基址的标量类型获取指针元素类型
  auto addr = irb_.CreateGEP(
      base->getType()->getScalarType()->getPointerElementType(), base, idx);
#endif

  // 创建存储指令，将值 val 存储到地址 addr
  irb_.CreateStore(val, addr);
  // 在条件块中创建无条件跳转到尾块的指令
  irb_.CreateBr(tailblock);

  // 设置尾块的插入点，合并掩码和未掩码的控制流边缘
  irb_.SetInsertPoint(tailblock);
}

// 处理 StorePtr 类型的访问
void LLVMCodeGenImpl::visit(StorePtr v) {
  // 如果存储值的数据类型的通道数为 1
  if (v->value()->dtype().lanes() == 1) {
    // 接受基址处理
    v->base_handle()->accept(this);
    auto base = this->value_;
    // 接受扁平索引处理
    v->flat_index()->accept(this);
    auto idx = this->value_;
    // 接受值处理
    v->value()->accept(this);
    auto val = this->value_;

    // 发出未掩码存储指令
    emitUnmaskedStore(dtypeToLLVM(v->value()->dtype()), base, idx, val);
    // 设置返回值为常数整数 0
    value_ = llvm::ConstantInt::get(IntTy_, 0);
    return;
  }

  // 接受基址处理
  v->base_handle()->accept(this);
  auto base = this->value_;
  // 接受值处理
  v->value()->accept(this);
  auto val = this->value_;

  // 处理存储是连续的且未掩码的情况
  auto idx_ramp = to<Ramp>(v->flat_index());
  if (idx_ramp) {
    // 获取 Ramp 类型的对象
    auto stride_imm = intValue(idx_ramp->stride());
    if (stride_imm && *stride_imm == 1) {
      // 接受基址处理
      idx_ramp->base()->accept(this);
      auto first_idx = value_;

#if LLVM_VERSION_MAJOR >= 15
      // 创建基于类型、基址和第一个索引的指针地址
      auto addr =
          irb_.CreateGEP(dtypeToLLVM(v->value()->dtype()), base, first_idx);
#else
      // 为低版本 LLVM 创建指针地址，根据基址的标量类型获取指针元素类型
      auto addr = irb_.CreateGEP(
          base->getType()->getScalarType()->getPointerElementType(),
          base,
          first_idx);
#endif

      // 创建位或指针转换，将地址转换为相应类型的指针
      auto vaddr = irb_.CreateBitOrPointerCast(
          addr, llvm::PointerType::get(val->getType(), 0));

#if LLVM_VERSION_MAJOR >= 13
      // 创建按对齐要求对齐存储指令
      irb_.CreateAlignedStore(val, vaddr, llvm::MaybeAlign(4));
#else
      // 创建按固定对齐 4 字节存储指令
      irb_.CreateAlignedStore(val, vaddr, 4);
#endif
// 终止预处理指令，结束条件分支，将值设为整数常量 0
#endif
      value_ = llvm::ConstantInt::get(IntTy_, 0);
      return;
    }
  }

  // 接受 v 的 flat_index，并将结果存储到 idx
  v->flat_index()->accept(this);
  auto idx = this->value_;

  // 回退到标量实现
  for (int i = 0; i < v->value()->dtype().lanes(); ++i) {
    // 从 idx 和 val 中提取第 i 个元素
    auto sub_idx = irb_.CreateExtractElement(idx, i);
    auto sub_val = irb_.CreateExtractElement(val, i);
    // 调用 emitUnmaskedStore 函数，将数据 sub_val 存储到 base 偏移 sub_idx 的位置
    emitUnmaskedStore(dtypeToLLVM(v->value()->dtype()), base, sub_idx, sub_val);
  }

  // 将值设为整数常量 0
  value_ = llvm::ConstantInt::get(IntTy_, 0);
}

// 访问 BroadcastPtr 类型的节点
void LLVMCodeGenImpl::visit(BroadcastPtr v) {
  // 接受 v 的 value，并将结果存储到 value_
  v->value()->accept(this);
  int lanes = v->lanes();
  // 使用 value_ 创建一个宽度为 lanes 的向量
  value_ = irb_.CreateVectorSplat(lanes, value_);
}

// 访问 IfThenElsePtr 类型的节点
void LLVMCodeGenImpl::visit(IfThenElsePtr v) {
  // 接受 v 的 condition，并将结果存储到 value_
  v->condition()->accept(this);
  llvm::Value* condition = value_;
  // 创建一个比较指令，比较 condition 和整数常量 0 的不等关系
  llvm::Value* c = irb_.CreateICmpNE(
      condition, llvm::ConstantInt::get(condition->getType(), 0));

  // 创建三个基本块：then_block、else_block 和 end_block
  auto then_block = llvm::BasicBlock::Create(getContext(), "then", fn_);
  auto else_block = llvm::BasicBlock::Create(getContext(), "else", fn_);
  auto end_block = llvm::BasicBlock::Create(getContext(), "block", fn_);
  // 根据比较结果条件 c，选择跳转到 then_block 或 else_block
  irb_.CreateCondBr(c, then_block, else_block);

  // 处理 then_block
  irb_.SetInsertPoint(then_block);
  // 接受 v 的 true_value，并将结果存储到 value_
  v->true_value()->accept(this);
  llvm::Value* then_val = value_;
  then_block = irb_.GetInsertBlock();
  irb_.CreateBr(end_block);

  // 处理 else_block
  irb_.SetInsertPoint(else_block);
  // 接受 v 的 false_value，并将结果存储到 value_
  v->false_value()->accept(this);
  llvm::Value* else_val = value_;
  else_block = irb_.GetInsertBlock();
  irb_.CreateBr(end_block);

  // 处理 end_block
  irb_.SetInsertPoint(end_block);
  // 创建 PHI 节点，合并 then_val 和 else_val 的值
  llvm::PHINode* phi = irb_.CreatePHI(then_val->getType(), 2);
  phi->addIncoming(then_val, then_block);
  phi->addIncoming(else_val, else_block);
  value_ = phi;
}

// 静态函数，为 LLVM 函数对象应用数学函数的属性
static void applyMathFunctionAttributes(llvm::Function* f) {
  f->addFnAttr(llvm::Attribute::ReadNone);  // 函数没有副作用，不读取内存
  f->addFnAttr(llvm::Attribute::NoUnwind);  // 函数不会抛出异常
  // 添加此属性在 LLVM 9.0.1 中会导致某些数学函数错误地转换为尾调用
  // f->addFnAttr(llvm::Attribute::Speculatable);
#if LLVM_VERSION_MAJOR >= 9
  f->addFnAttr(llvm::Attribute::NoFree);    // 函数不会释放内存
  f->addFnAttr(llvm::Attribute::WillReturn);  // 函数总是返回结果
#endif
}

// 将给定的 LLVM 值 v 转换为一个宽度为 lanes 的向量
llvm::Value* LLVMCodeGenImpl::toVec(llvm::Value* v, int lanes) {
  if (lanes > 1) {
    return irb_.CreateVectorSplat(lanes, v);  // 创建向量，所有元素都是 v
  } else {
    return v;  // 返回原始值
  }
}

// 访问 IntrinsicsPtr 类型的节点，处理 IsNan 操作
void LLVMCodeGenImpl::emitIsNan(IntrinsicsPtr v) {
  // 接受 v 的第一个参数，并将结果存储到 value_
  v->param(0)->accept(this);
  llvm::Type* dstType = dtypeToLLVM(v->dtype());
  if (!v->param(0)->dtype().is_floating_point()) {
    // 如果参数不是浮点数类型，则将 value_ 转换为长度为 v->dtype().lanes() 的向量
    value_ = toVec(llvm::ConstantInt::get(dstType, 0), v->dtype().lanes());
  } else {
    // 否则，参数为浮点数类型
    TORCH_INTERNAL_ASSERT(
        v->dtype().scalar_type() == ScalarType::Int,
        buildErrorMessage(
            "Unexpected non-Int dtype of Intrinsics' result value in the fuser."));
    // 创建一个比较指令，检查 value_ 是否为 NaN
    auto is_nan = irb_.CreateFCmpUNO(
        value_, llvm::ConstantFP::get(value_->getType(), 0.));
    if (v->dtype().lanes() > 1) {
      dstType =
          llvm::VectorType::get(dstType, ElementCount(v->dtype().lanes()));
    }
    // 如果需要的话，将 dstType 设置为长度为 v->dtype().lanes() 的向量类型
    // (此处代码截断，继续...)
    # 使用 IRBuilder 对象调用 CreateIntCast 方法，将 is_nan 转换为目标类型 dstType
    value_ = irb_.CreateIntCast(is_nan, dstType, /*isSigned*/ false);
  }
}

static bool wantSleef(const std::string& name) {
  // 检查是否应该使用 Sleef 库而非标准库函数
  static std::unordered_set<std::string> noSleef = {
      "sqrt",
      "ceil",
      "trunc",
      "fabs",
      "floor",
      "sqrtf",
      "ceilf",
      "truncf",
      "fabsf",
      "floorf",
  };
  // 如果函数名在不使用 Sleef 的集合中，则返回 true，否则返回 false
  return noSleef.find(name) == noSleef.end();
}

LLVMCodeGenImpl::SimdCallee LLVMCodeGenImpl::getSimdFunction(
    const std::string& basename,
    llvm::Type* basetype,
    Arity arity,
    int lanes) {
  std::string name;
  llvm::Type* type;
  bool useSimd;

  // 获取目标机器支持的特性字符串
  auto const& featureString = jit_->getTargetMachine().getTargetFeatureString();
  // 检查目标机器是否支持 AVX 指令集
  bool hasAVX = featureString.find("+avx") != llvm::StringRef::npos;
  // 根据基本类型选择后缀
  std::string typeSuffix = basetype == DoubleTy_ ? "d" : "";
  // 构造 Sleef 函数名
  std::string sleefName =
      "Sleef_" + basename + typeSuffix + std::to_string(lanes);
  // 判断是否应该使用 Sleef 函数，且目标机器支持 AVX，并且 JIT 编译器有这个符号
  if (wantSleef(basename) && hasAVX && jit_->hasSymbol(sleefName)) {
    name = std::move(sleefName);
    // 创建矢量类型
    type = llvm::VectorType::get(basetype, ElementCount(lanes));
    useSimd = true;
  } else {
    name = basename;
    type = basetype;
    useSimd = false;
  }

  // 根据函数名和类型获取要调用的函数
  llvm::FunctionType* fntype;
  switch (arity) {
    case Unary:
      fntype = llvm::FunctionType::get(type, {type}, false);
      break;
    case Binary:
      fntype = llvm::FunctionType::get(type, {type, type}, false);
      break;
  }
  // 获取或插入模块中的函数声明
  FunctionCallee callee = module_->getOrInsertFunction(name, fntype, {});
  // 应用数学函数的属性
  applyMathFunctionAttributes(llvm::cast<llvm::Function>(callee.getCallee()));
  // 返回 SimdCallee 结构体
  return SimdCallee{callee.getFunctionType(), callee.getCallee(), useSimd};
}

void LLVMCodeGenImpl::visit(IntrinsicsPtr v) {
  llvm::FunctionType* call_ty = nullptr;
  llvm::Value* call_fn = nullptr;
  bool call_simd_sleef = false;

  // 如果是 kIsNan 操作，直接调用 emitIsNan 函数并返回
  if (v->op_type() == kIsNan) {
    return emitIsNan(v);
  }

  // 如果数据类型是 float
  if (v->dtype().scalar_type() == ScalarType::Float) {
    // 根据操作类型进行处理
    switch (v->op_type()) {
      case kRsqrt: {
        // 访问参数，并调用 sqrt 内置函数
        v->params().front()->accept(this);
        value_ = irb_.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, value_);
        // 创建常数向量，并将其除以 value_
        llvm::Value* constant =
            toVec(llvm::ConstantFP::get(FloatTy_, 1.0), v->dtype().lanes());
        value_ = irb_.CreateFDiv(constant, value_);
        return;
      } break;

#define SIMD_UNARY_MATH_CASE(enum, name, type)                  \
  case enum: {                                                  \
    std::tie(call_ty, call_fn, call_simd_sleef) =               \
        getSimdFunction(name, type, Unary, v->dtype().lanes()); \

// 调用 getSimdFunction 函数获取 SIMD 函数的类型、函数名和 SIMD Sleef 版本

  } break;

// 结束 switch-case 结构

        SIMD_UNARY_MATH_CASE(kLog10, "log10f", FloatTy_)

// 定义一个宏，处理 kLog10 情况，调用 log10f 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kLog, "logf", FloatTy_)

// 定义一个宏，处理 kLog 情况，调用 logf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kLog1p, "log1pf", FloatTy_)

// 定义一个宏，处理 kLog1p 情况，调用 log1pf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kLog2, "log2f", FloatTy_)

// 定义一个宏，处理 kLog2 情况，调用 log2f 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kExp, "expf", FloatTy_)

// 定义一个宏，处理 kExp 情况，调用 expf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kCos, "cosf", FloatTy_)

// 定义一个宏，处理 kCos 情况，调用 cosf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kSin, "sinf", FloatTy_)

// 定义一个宏，处理 kSin 情况，调用 sinf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kSqrt, "sqrtf", FloatTy_)

// 定义一个宏，处理 kSqrt 情况，调用 sqrtf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kAbs, "fabsf", FloatTy_)

// 定义一个宏，处理 kAbs 情况，调用 fabsf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kFloor, "floorf", FloatTy_)

// 定义一个宏，处理 kFloor 情况，调用 floorf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kCeil, "ceilf", FloatTy_)

// 定义一个宏，处理 kCeil 情况，调用 ceilf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kTrunc, "truncf", FloatTy_)

// 定义一个宏，处理 kTrunc 情况，调用 truncf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kRound, "nearbyint", FloatTy_)

// 定义一个宏，处理 kRound 情况，调用 nearbyint 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kErf, "erff", FloatTy_)

// 定义一个宏，处理 kErf 情况，调用 erff 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kErfc, "erfcf", FloatTy_)

// 定义一个宏，处理 kErfc 情况，调用 erfcf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kTan, "tanf", FloatTy_)

// 定义一个宏，处理 kTan 情况，调用 tanf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kAcos, "acosf", FloatTy_)

// 定义一个宏，处理 kAcos 情况，调用 acosf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kAsin, "asinf", FloatTy_)

// 定义一个宏，处理 kAsin 情况，调用 asinf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kAtan, "atanf", FloatTy_)

// 定义一个宏，处理 kAtan 情况，调用 atanf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kCosh, "coshf", FloatTy_)

// 定义一个宏，处理 kCosh 情况，调用 coshf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kSinh, "sinhf", FloatTy_)

// 定义一个宏，处理 kSinh 情况，调用 sinhf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kTanh, "tanhf", FloatTy_)

// 定义一个宏，处理 kTanh 情况，调用 tanhf 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kExpm1, "expm1f", FloatTy_)

// 定义一个宏，处理 kExpm1 情况，调用 expm1f 函数，操作 float 类型

        SIMD_UNARY_MATH_CASE(kLgamma, "lgammaf", FloatTy_)

// 定义一个宏，处理 kLgamma 情况，调用 lgammaf 函数，操作 float 类型
#undef SIMD_UNARY_MATH_CASE

#define SIMD_BINARY_MATH_CASE(enum, name, type)                  \
  case enum: {                                                   \
    // 获取指定函数名、类型和操作数类型的 SIMD 函数
    std::tie(call_ty, call_fn, call_simd_sleef) =                \
        getSimdFunction(name, type, Binary, v->dtype().lanes()); \
  } break;
        // 处理 kAtan2 操作
        SIMD_BINARY_MATH_CASE(kAtan2, "atan2f", FloatTy_)
        // 处理 kPow 操作
        SIMD_BINARY_MATH_CASE(kPow, "powf", FloatTy_)
        // 处理 kFmod 操作
        SIMD_BINARY_MATH_CASE(kFmod, "fmodf", FloatTy_)
#undef SIMD_BINARY_MATH_CASE

      case kRemainder: {
        // 获取或插入 LLVM 中的 remainderf 函数
        FunctionCallee callee = module_->getOrInsertFunction(
            "remainderf",
            llvm::FunctionType::get(FloatTy_, {FloatTy_, FloatTy_}, false),
            {});
        call_ty = callee.getFunctionType();
        call_fn = callee.getCallee();
        // 应用数学函数属性于调用的函数
        applyMathFunctionAttributes(llvm::cast<llvm::Function>(call_fn));
      } break;

      default: {
        // 抛出未实现的操作异常
        throw unimplemented_lowering(v);
      } break;
    }

  } else if (v->dtype().scalar_type() == ScalarType::Double) {
    switch (v->op_type()) {
#define SIMD_UNARY_MATH_CASE(enum, name, type)                  \
  case enum: {                                                  \
    // 获取指定函数名、类型和操作数类型的 SIMD 函数
    std::tie(call_ty, call_fn, call_simd_sleef) =               \
        getSimdFunction(name, type, Unary, v->dtype().lanes()); \
  } break;
      // 处理 kLog10 操作
      SIMD_UNARY_MATH_CASE(kLog10, "log10", DoubleTy_)
      // 处理 kLog 操作
      SIMD_UNARY_MATH_CASE(kLog, "log", DoubleTy_)
      // 处理 kLog1p 操作
      SIMD_UNARY_MATH_CASE(kLog1p, "log1p", DoubleTy_)
      // 处理 kLog2 操作
      SIMD_UNARY_MATH_CASE(kLog2, "log2", DoubleTy_)
      // 处理 kExp 操作
      SIMD_UNARY_MATH_CASE(kExp, "exp", DoubleTy_)
      // 处理 kCos 操作
      SIMD_UNARY_MATH_CASE(kCos, "cos", DoubleTy_)
      // 处理 kSin 操作
      SIMD_UNARY_MATH_CASE(kSin, "sin", DoubleTy_)
      // 处理 kSqrt 操作
      SIMD_UNARY_MATH_CASE(kSqrt, "sqrt", DoubleTy_)
      // 处理 kAbs 操作
      SIMD_UNARY_MATH_CASE(kAbs, "fabs", DoubleTy_)
      // 处理 kFloor 操作
      SIMD_UNARY_MATH_CASE(kFloor, "floor", DoubleTy_)
      // 处理 kCeil 操作
      SIMD_UNARY_MATH_CASE(kCeil, "ceil", DoubleTy_)
      // 处理 kTrunc 操作
      SIMD_UNARY_MATH_CASE(kTrunc, "trunc", DoubleTy_)
      // 处理 kRound 操作
      SIMD_UNARY_MATH_CASE(kRound, "nearbyint", DoubleTy_)
      // 处理 kErf 操作
      SIMD_UNARY_MATH_CASE(kErf, "erf", DoubleTy_)
      // 处理 kErfc 操作
      SIMD_UNARY_MATH_CASE(kErfc, "erfc", DoubleTy_)
      // 处理 kTan 操作
      SIMD_UNARY_MATH_CASE(kTan, "tan", DoubleTy_)
      // 处理 kAcos 操作
      SIMD_UNARY_MATH_CASE(kAcos, "acos", DoubleTy_)
      // 处理 kAsin 操作
      SIMD_UNARY_MATH_CASE(kAsin, "asin", DoubleTy_)
      // 处理 kAtan 操作
      SIMD_UNARY_MATH_CASE(kAtan, "atan", DoubleTy_)
      // 处理 kCosh 操作
      SIMD_UNARY_MATH_CASE(kCosh, "cosh", DoubleTy_)
      // 处理 kSinh 操作
      SIMD_UNARY_MATH_CASE(kSinh, "sinh", DoubleTy_)
      // 处理 kTanh 操作
      SIMD_UNARY_MATH_CASE(kTanh, "tanh", DoubleTy_)
      // 处理 kExpm1 操作
      SIMD_UNARY_MATH_CASE(kExpm1, "expm1", DoubleTy_)
      // 处理 kLgamma 操作
      SIMD_UNARY_MATH_CASE(kLgamma, "lgamma", DoubleTy_)
#undef SIMD_UNARY_MATH_CASE

      // 处理 kRsqrt 操作
      case kRsqrt: {
        // 访问第一个参数并生成相应的 LLVM IR
        v->params().front()->accept(this);
        // 创建对应的 sqrt 的 LLVM 内置函数调用
        value_ = irb_.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, value_);
        // 创建常数值为 1.0 的 LLVM 浮点常数
        llvm::Value* constant = llvm::ConstantFP::get(DoubleTy_, 1.0);
        // 如果数据类型的 lanes 大于 1，则将常数值向量化
        if (v->dtype().lanes() > 1) {
          constant = irb_.CreateVectorSplat(v->dtype().lanes(), constant);
        }
        // 创建常数值除以 value_ 的 LLVM IR
        value_ = irb_.CreateFDiv(constant, value_);
        // 返回操作结束
        return;
      } break;

#define SIMD_BINARY_MATH_CASE(enum, name, type)                  \
  // 处理二元数学操作的宏定义，根据操作类型获取相应的 SIMD 函数信息
  case enum: {                                                   \
    std::tie(call_ty, call_fn, call_simd_sleef) =                \
        getSimdFunction(name, type, Binary, v->dtype().lanes()); \
  } break;
        // 使用 SIMD_BINARY_MATH_CASE 宏定义处理不同的二元数学操作
        SIMD_BINARY_MATH_CASE(kAtan2, "atan2", DoubleTy_)
        SIMD_BINARY_MATH_CASE(kPow, "pow", DoubleTy_)
        SIMD_BINARY_MATH_CASE(kFmod, "fmod", DoubleTy_)
#undef SIMD_BINARY_MATH_CASE

      // 处理 kRemainder 操作
      case kRemainder: {
        // 获取或插入 LLVM 函数 "remainder" 的函数声明
        FunctionCallee callee = module_->getOrInsertFunction(
            "remainder",
            llvm::FunctionType::get(DoubleTy_, {DoubleTy_, DoubleTy_}, false),
            {});
        // 获取函数调用类型和函数指针
        call_ty = callee.getFunctionType();
        call_fn = callee.getCallee();
        // 应用数学函数的属性到 LLVM 函数
        applyMathFunctionAttributes(llvm::cast<llvm::Function>(call_fn));
      } break;

      // 默认情况，抛出未实现的降级异常
      default: {
        throw unimplemented_lowering(v);
      } break;
    }
  } else if (v->dtype().is_integral() && v->op_type() == kAbs) {
    // 对于整数输入且操作类型为 kAbs，处理绝对值运算
    // abs 是 pytorch eager 中为整数输入定义的唯一内置函数
    v->params().front()->accept(this);
    // 如果数据类型未签名，则直接返回
    if (!v->dtype().is_signed()) {
      return;
    }
    // 创建 LLVM 常数值为 0
    auto zero = llvm::ConstantInt::get(value_->getType(), 0);
    // 创建 value_ 的相反数
    auto neg_value = irb_.CreateSub(zero, value_);
    // 创建比较操作，判断 value_ 是否大于零
    auto icmp = irb_.CreateICmpSGT(value_, zero);
    // 根据比较结果选择 value_ 或 neg_value
    value_ = irb_.CreateSelect(icmp, value_, neg_value);
    // 返回操作结束
    return;
  } else {
    // 抛出错误，表示在 fuser 的 LLVM 代码生成中未实现的内置操作
    TORCH_INTERNAL_ASSERT(
        false,
        buildErrorMessage(
            std::string("Unimplemented lowering for intrinsic '") +
            std::to_string(v->op_type()) + "' for input of dtype " +
            std::to_string(v->dtype().scalar_dtype()) +
            " in LLVM codegen of the fuser."));
  }

  // 准备参数列表
  std::vector<llvm::Value*> params;
  // 访问所有参数并生成相应的 LLVM IR
  for (auto& p : v->params()) {
    p->accept(this);
    params.push_back(value_);
  }

  // 如果数据类型的 lanes 等于 1 或者使用 SIMD Sleef，创建简单的函数调用
  if (v->dtype().lanes() == 1 || call_simd_sleef == true) {
    value_ = irb_.CreateCall(call_ty, call_fn, params);
  } else {
    // 否则，处理向量化操作
    llvm::Type* vecType = params[0]->getType();
    value_ = llvm::UndefValue::get(vecType);
    // 遍历每个 lane
    for (int i = 0; i < v->dtype().lanes(); ++i) {
      std::vector<llvm::Value*> call_operands;
      // 对于每个参数，生成相应 lane 的 LLVM IR
      for (auto p : params) {
        call_operands.push_back(irb_.CreateExtractElement(p, i));
      }

      // 创建函数调用并插入结果到向量中
      llvm::Value* val = irb_.CreateCall(call_ty, call_fn, call_operands);
      value_ = irb_.CreateInsertElement(value_, val, i);
    }
  }
}
    // 处理缓冲重用，更新变量到值的映射关系
    void LLVMCodeGenImpl::handleBufReuse(BufPtr buf, BufPtr buf_to_reuse) {
      // 获取要重用的缓冲区的基本句柄对应的 LLVM 值
      llvm::Value* ptr = varToVal_.at(buf_to_reuse->base_handle());
      // 如果要重用的缓冲区的数据类型与当前缓冲区的数据类型不同，进行类型转换
      if (buf_to_reuse->dtype().scalar_type() != buf->dtype().scalar_type()) {
        ptr = irb_.CreatePointerCast(ptr, dtypeToLLVMPtr(buf->dtype()));
      }
      // 将当前缓冲区的基本句柄映射到更新后的 LLVM 值
      varToVal_[buf->base_handle()] = ptr;
    }

    // 访问外部调用节点
    void LLVMCodeGenImpl::visit(ExternalCallPtr v) {
      // 获取 NNC 函数注册表的引用
      auto& func_registry = getNNCFunctionRegistry();
      // 如果函数名不在注册表中，抛出未实现的下降异常
      if (!func_registry.count(v->func_name())) {
        throw unimplemented_lowering(v);
      }

      // 准备传递给外部函数的缓冲区向量，包括输出缓冲区和参数缓冲区
      std::vector<BufPtr> bufs(v->buf_args());
      bufs.insert(bufs.begin(), v->buf());

      int64_t bufs_num = bufs.size();
      int64_t args_num = v->args().size();

      // 计算维度数组的大小，它包括所有缓冲区的维度连接在一起
      int64_t dims_num = 0;
      for (BufPtr b : bufs) {
        dims_num += b->dims().size();
      }

      // 根据 LLVM 版本选择合适的类型创建缓冲区指针数组
#if LLVM_VERSION_MAJOR >= 15
      llvm::Value* buf_ptrs = irb_.CreateAlloca(
          OpqPtrTy_, llvm::ConstantInt::getSigned(IntTy_, bufs_num));
#else
      llvm::Value* buf_ptrs = irb_.CreateAlloca(
          Int8PtrTy_, llvm::ConstantInt::getSigned(IntTy_, bufs_num));
#endif
      // 创建数组来存储缓冲区的秩
      llvm::Value* buf_ranks = irb_.CreateAlloca(
          LongTy_, llvm::ConstantInt::getSigned(IntTy_, bufs_num));
      // 创建数组来存储缓冲区的维度
      llvm::Value* buf_dims = irb_.CreateAlloca(
          LongTy_, llvm::ConstantInt::getSigned(IntTy_, dims_num));
      // 创建数组来存储缓冲区的步长
      llvm::Value* buf_strides = irb_.CreateAlloca(
          LongTy_, llvm::ConstantInt::getSigned(IntTy_, dims_num));
      // 创建数组来存储缓冲区的数据类型
      llvm::Value* buf_dtypes = irb_.CreateAlloca(
          ByteTy_, llvm::ConstantInt::getSigned(IntTy_, bufs_num));
      // 创建数组来存储额外参数
      llvm::Value* extra_args = irb_.CreateAlloca(
          LongTy_, llvm::ConstantInt::getSigned(IntTy_, args_num));

      int i = 0;
      int dim_idx = 0;
      int stride_idx = 0;
      for (BufPtr b : bufs) {
        // 存储缓冲区指针的值
        b->base_handle()->accept(this);
        auto buf_ptr = this->value_;
        // 根据 LLVM 版本创建适当的指针类型和 GEP 指令
#if LLVM_VERSION_MAJOR >= 15
        auto gep = irb_.CreateInBoundsGEP(
            OpqPtrTy_, buf_ptrs, llvm::ConstantInt::getSigned(IntTy_, i));
        auto buf_void_ptr = irb_.CreatePointerCast(buf_ptr, OpqPtrTy_);
#else
        auto gep = irb_.CreateInBoundsGEP(
            Int8PtrTy_, buf_ptrs, llvm::ConstantInt::getSigned(IntTy_, i));
        auto buf_void_ptr = irb_.CreatePointerCast(buf_ptr, Int8PtrTy_);
#endif
        irb_.CreateStore(buf_void_ptr, gep);

        // 存储缓冲区的数据类型
        gep = irb_.CreateInBoundsGEP(
            ByteTy_, buf_dtypes, llvm::ConstantInt::getSigned(IntTy_, i));
        irb_.CreateStore(
            llvm::ConstantInt::getSigned(ByteTy_, (int8_t)b->dtype().scalar_type()),
            gep);

        // 存储缓冲区的秩
        gep = irb_.CreateInBoundsGEP(
            LongTy_, buf_ranks, llvm::ConstantInt::getSigned(IntTy_, i));
        irb_.CreateStore(
            llvm::ConstantInt::getSigned(LongTy_, b->dims().size()), gep);

        // 存储缓冲区的维度
    // 对于每个维度，创建相应的内存偏移指令
    for (const auto dim : c10::irange(b->dims().size())) {
      // 创建一个指向缓冲区维度数组中指定维度的指针
      gep = irb_.CreateInBoundsGEP(
          LongTy_, buf_dims, llvm::ConstantInt::getSigned(IntTy_, dim_idx));
      // 访问缓冲区的维度表达式并保存其值
      b->dims()[dim]->accept(this);
      auto dim_val = this->value_;
      // 将维度值转换为长整型并存储到之前创建的指针指向的位置
      irb_.CreateStore(irb_.CreateZExt(dim_val, LongTy_), gep);
      dim_idx++;
    }

    // 存储缓冲区的步长
    for (const auto dim : c10::irange(b->dims().size())) {
      // 创建一个指向缓冲区步长数组中指定步长的指针
      gep = irb_.CreateInBoundsGEP(
          LongTy_,
          buf_strides,
          llvm::ConstantInt::getSigned(IntTy_, stride_idx));
      // 访问缓冲区的步长表达式并保存其值
      b->strides()[dim]->accept(this);
      auto stride_val = this->value_;
      // 将步长值转换为长整型并存储到之前创建的指针指向的位置
      irb_.CreateStore(irb_.CreateZExt(stride_val, LongTy_), gep);
      stride_idx++;
    }

    // 重置循环计数器 i
    i++;

  }

  // 将循环计数器 i 重置为 0，用于遍历函数的参数列表
  i = 0;
  // 遍历函数的参数列表，并将每个参数的值存储到额外参数数组中
  for (ExprPtr arg : v->args()) {
    // 创建一个指向额外参数数组中指定位置的指针
    auto gep = irb_.CreateInBoundsGEP(
        LongTy_, extra_args, llvm::ConstantInt::getSigned(IntTy_, i));
    // 访问参数表达式并保存其值
    arg->accept(this);
    // 将参数值转换为长整型或位转换，并存储到之前创建的指针指向的位置
    irb_.CreateStore(irb_.CreateZExtOrBitCast(this->value_, LongTy_), gep);
    i++;
  }

  // 生成函数调用的名称
  std::string fname = v->func_name();
#if LLVM_VERSION_MAJOR >= 15
  // 如果 LLVM 版本大于等于 15，使用新的函数类型创建函数调用描述符
  FunctionCallee callee = module_->getOrInsertFunction(
      fname, // 函数名
      llvm::FunctionType::get(
          llvm::Type::getVoidTy(getContext()), // 返回类型为 void
          {LongTy_,       // int64_t bufs_num
           OpqPtrTy_,    // void** buf_data
           OpqPtrTy_,    // int64_t* buf_ranks
           OpqPtrTy_,    // int64_t* buf_dims
           OpqPtrTy_,    // int64_t* buf_strides
           OpqPtrTy_,    // int64_t* buf_dtypes
           LongTy_,      // int64_t args_num
           OpqPtrTy_},   // int64_t* extra_args
          false));       // 不是变长参数
#else
  // 如果 LLVM 版本小于 15，使用较旧的函数类型创建函数调用描述符
  FunctionCallee callee = module_->getOrInsertFunction(
      fname, // 函数名
      llvm::FunctionType::get(
          llvm::Type::getVoidTy(getContext()), // 返回类型为 void
          {LongTy_,                      // int64_t bufs_num
           Int8PtrTy_->getPointerTo(),  // void** buf_data
           LongTy_->getPointerTo(),     // int64_t* buf_ranks
           LongTy_->getPointerTo(),     // int64_t* buf_dims
           LongTy_->getPointerTo(),     // int64_t* buf_strides
           ByteTy_->getPointerTo(),     // int64_t* buf_dtypes
           LongTy_,                      // int64_t args_num
           LongTy_->getPointerTo()},    // int64_t* extra_args
          false));                      // 不是变长参数
#endif

// 获取函数调用的类型
auto call_ty = callee.getFunctionType();
// 获取函数调用的函数指针
auto call_fn = callee.getCallee();
// 给函数指定 'NoUnwind' 属性，表示在调用期间不会抛出异常
llvm::cast<llvm::Function>(call_fn)->addFnAttr(llvm::Attribute::NoUnwind);

// 创建 LLVM IR 中的函数调用指令
irb_.CreateCall(
    call_ty, // 函数调用类型
    call_fn, // 函数指针
    {llvm::ConstantInt::getSigned(LongTy_, bufs_num), // 实参列表
     buf_ptrs,
     buf_ranks,
     buf_dims,
     buf_strides,
     buf_dtypes,
     llvm::ConstantInt::getSigned(LongTy_, args_num),
     extra_args});

// 将 value_ 设置为整数常量 0
value_ = llvm::ConstantInt::get(IntTy_, 0);
}

void LLVMCodeGenImpl::visit(ExternalCallWithAllocPtr v) {
  // 获取函数注册表
  auto& func_registry = getNNCFunctionRegistry();
  // 如果函数名在注册表中找不到，抛出未实现的异常
  if (!func_registry.count(v->func_name())) {
    throw unimplemented_lowering(v);
  }

  // 获取输出缓冲区和输入缓冲区
  const auto& bufs_out = v->buf_out_args();
  const auto& bufs_in = v->buf_args();

  // 获取输入缓冲区和输出缓冲区的大小，以及参数的数量
  const auto bufs_in_size = bufs_in.size();
  const auto bufs_out_size = bufs_out.size();
  const auto args_num = v->args().size();

  // 计算 dims 数组的大小，该数组由所有输入缓冲区的维度拼接而成
  size_t dims_num = 0;
  for (const auto& b : bufs_in) {
    dims_num += b->dims().size();
  }

  // 根据 LLVM 版本创建不同类型的缓冲区指针
#if LLVM_VERSION_MAJOR >= 15
  llvm::Value* buf_ptrs = irb_.CreateAlloca(
      OpqPtrTy_, // void** 类型
      llvm::ConstantInt::getSigned(IntTy_, bufs_in_size + 2 * bufs_out_size));
#else
  llvm::Value* buf_ptrs = irb_.CreateAlloca(
      Int8PtrTy_, // void** 类型
      llvm::ConstantInt::getSigned(IntTy_, bufs_in_size + 2 * bufs_out_size));
#endif
#endif
  // 创建一个用于存储 buf_ranks 的 LLVM 变量，类型为指向长整型的指针，大小为 bufs_in_size
  llvm::Value* buf_ranks = irb_.CreateAlloca(
      LongTy_, llvm::ConstantInt::getSigned(IntTy_, bufs_in_size));
  // 创建一个用于存储 buf_dims 的 LLVM 变量，类型为指向长整型的指针，大小为 dims_num
  llvm::Value* buf_dims = irb_.CreateAlloca(
      LongTy_, llvm::ConstantInt::getSigned(IntTy_, dims_num));
  // 创建一个用于存储 buf_strides 的 LLVM 变量，类型为指向长整型的指针，大小为 dims_num
  llvm::Value* buf_strides = irb_.CreateAlloca(
      LongTy_, llvm::ConstantInt::getSigned(IntTy_, dims_num));
  // 创建一个用于存储 buf_dtypes 的 LLVM 变量，类型为指向字节型的指针，大小为 bufs_in_size
  llvm::Value* buf_dtypes = irb_.CreateAlloca(
      ByteTy_, llvm::ConstantInt::getSigned(IntTy_, bufs_in_size));
  // 创建一个用于存储 extra_args 的 LLVM 变量，类型为指向长整型的指针，大小为 args_num
  llvm::Value* extra_args = irb_.CreateAlloca(
      LongTy_, llvm::ConstantInt::getSigned(IntTy_, args_num));

  int i = 0;
  int dim_idx = 0;
  int stride_idx = 0;
  // 遍历 bufs_in 中的每个缓冲区 b
  for (const auto& b : bufs_in) {
    // 存储 buf 指针的值
    b->base_handle()->accept(this);
    auto buf_ptr = this->value_;

    // 根据 LLVM 版本不同选择创建 GEP 操作
#if LLVM_VERSION_MAJOR >= 15
    // 创建 InBoundsGEP 操作，用于 buf_ptrs 数组中的偏移位置
    llvm::Value* gep = irb_.CreateInBoundsGEP(
        OpqPtrTy_,
        buf_ptrs,
        // @lint-ignore CLANGTIDY
        llvm::ConstantInt::getSigned(IntTy_, bufs_out_size + i));
    auto buf_void_ptr = irb_.CreatePointerCast(buf_ptr, OpqPtrTy_);
#else
    // 创建 InBoundsGEP 操作，用于 buf_ptrs 数组中的偏移位置
    llvm::Value* gep = irb_.CreateInBoundsGEP(
        Int8PtrTy_,
        buf_ptrs,
        // @lint-ignore CLANGTIDY
        llvm::ConstantInt::getSigned(IntTy_, bufs_out_size + i));
    auto buf_void_ptr = irb_.CreatePointerCast(buf_ptr, Int8PtrTy_);
#endif

    // 将 buf_void_ptr 存储到 gep 指向的位置
    irb_.CreateStore(buf_void_ptr, gep);

    // 存储 buf 的数据类型
    gep = irb_.CreateInBoundsGEP(
        ByteTy_, buf_dtypes, llvm::ConstantInt::getSigned(IntTy_, i));
    irb_.CreateStore(
        llvm::ConstantInt::getSigned(ByteTy_, (int8_t)b->dtype().scalar_type()),
        gep);

    // 存储 buf 的秩（rank）
    gep = irb_.CreateInBoundsGEP(
        LongTy_, buf_ranks, llvm::ConstantInt::getSigned(IntTy_, i));
    irb_.CreateStore(
        llvm::ConstantInt::getSigned(LongTy_, b->dims().size()), gep);

    // 存储 buf 的维度（dims）
    for (const auto dim : c10::irange(b->dims().size())) {
      gep = irb_.CreateInBoundsGEP(
          LongTy_, buf_dims, llvm::ConstantInt::getSigned(IntTy_, dim_idx));
      b->dims()[dim]->accept(this);
      auto dim_val = this->value_;
      irb_.CreateStore(irb_.CreateZExt(dim_val, LongTy_), gep);
      dim_idx++;
    }

    // 存储 buf 的步长（strides）
    for (const auto dim : c10::irange(b->dims().size())) {
      gep = irb_.CreateInBoundsGEP(
          LongTy_,
          buf_strides,
          llvm::ConstantInt::getSigned(IntTy_, stride_idx));
      b->strides()[dim]->accept(this);
      auto stride_val = this->value_;
      irb_.CreateStore(irb_.CreateZExt(stride_val, LongTy_), gep);
      stride_idx++;
    }

    i++;
  }

  // 重置计数器 i，并存储 v 的每个参数到 extra_args 中
  i = 0;
  for (const ExprPtr& arg : v->args()) {
    auto gep = irb_.CreateInBoundsGEP(
        LongTy_, extra_args, llvm::ConstantInt::getSigned(IntTy_, i));
    arg->accept(this);
    irb_.CreateStore(irb_.CreateZExtOrBitCast(this->value_, LongTy_), gep);
    i++;
  }
    i++;
  }



  // 增加变量 i 的值，实现自增操作
  i++;



  // Generate the call itself
  // 生成函数调用的名称，并将其存储在字符串 fname 中
  std::string fname = v->func_name();


这些注释将每行代码的作用解释清楚，符合规定的格式和要求。
#if LLVM_VERSION_MAJOR >= 15
  // 如果 LLVM 版本大于等于 15，则插入带有指定参数类型的函数定义到模块中
  FunctionCallee callee = module_->getOrInsertFunction(
      fname,
      llvm::FunctionType::get(
          llvm::Type::getVoidTy(getContext()), // 返回类型为 void
          {LongTy_, // int64_t bufs_in_size
           OpqPtrTy_, // void** buf_data
           OpqPtrTy_, // int64_t* buf_ranks
           OpqPtrTy_, // int64_t* buf_dims
           OpqPtrTy_, // int64_t* buf_strides
           OpqPtrTy_, // int64_t* buf_dtypes
           LongTy_, // int64_t args_num
           OpqPtrTy_}, // int64_t* extra_args
          false)); // 不是可变参数
#else
  // 如果 LLVM 版本小于 15，则插入带有不同参数类型的函数定义到模块中
  FunctionCallee callee = module_->getOrInsertFunction(
      fname,
      llvm::FunctionType::get(
          llvm::Type::getVoidTy(getContext()), // 返回类型为 void
          {LongTy_, // int64_t bufs_in_size
           Int8PtrTy_->getPointerTo(), // void** buf_data
           LongTy_->getPointerTo(), // int64_t* buf_ranks
           LongTy_->getPointerTo(), // int64_t* buf_dims
           LongTy_->getPointerTo(), // int64_t* buf_strides
           ByteTy_->getPointerTo(), // int64_t* buf_dtypes
           LongTy_, // int64_t args_num
           LongTy_->getPointerTo()}, // int64_t* extra_args
          false)); // 不是可变参数
#endif

// 获取函数的类型和函数指针
auto call_ty = callee.getFunctionType();
auto call_fn = callee.getCallee();
// 为函数添加无异常属性
llvm::cast<llvm::Function>(call_fn)->addFnAttr(llvm::Attribute::NoUnwind);

// 创建函数调用指令
irb_.CreateCall(
    call_ty,
    call_fn,
    // @lint-ignore CLANGTIDY
    {llvm::ConstantInt::getSigned(LongTy_, bufs_in_size),
     buf_ptrs,
     buf_ranks,
     buf_dims,
     buf_strides,
     buf_dtypes,
     // @lint-ignore CLANGTIDY
     llvm::ConstantInt::getSigned(LongTy_, args_num),
     extra_args});

// 遍历 bufs_out 数组的大小，处理每个输出缓冲区
// @lint-ignore CLANGTIDY
for (const auto i : c10::irange(bufs_out_size)) {
  const auto& buf_out = bufs_out[i];
#if LLVM_VERSION_MAJOR >= 15
  // 如果 LLVM 版本大于等于 15，则创建指向输出缓冲区的指针
  auto gep = irb_.CreateInBoundsGEP(
      OpqPtrTy_, buf_ptrs, llvm::ConstantInt::getSigned(IntTy_, i));
  llvm::Value* ptr = irb_.CreatePointerCast(
      irb_.CreateLoad(OpqPtrTy_, gep), dtypeToLLVMPtr(buf_out->dtype()));
#else
  // 如果 LLVM 版本小于 15，则创建指向输出缓冲区的指针
  auto gep = irb_.CreateInBoundsGEP(
      Int8PtrTy_, buf_ptrs, llvm::ConstantInt::getSigned(IntTy_, i));
  llvm::Value* ptr = irb_.CreatePointerCast(
      irb_.CreateLoad(Int8PtrTy_, gep), dtypeToLLVMPtr(buf_out->dtype()));
#endif

  // 将缓冲区输出的基础句柄映射到对应的指针值
  varToVal_[buf_out->base_handle()] = ptr;

  // 处理缓冲区的扩展分配或重用情况
  for (auto it = bufsExtAllocReuse_.find(buf_out);
       it != bufsExtAllocReuse_.end();
       it++) {
    auto buf = it->second;
    handleBufReuse(buf, buf_out);
  }
  bufsExtAllocReuse_.erase(buf_out);

#if LLVM_VERSION_MAJOR >= 15
  // 如果 LLVM 版本大于等于 15，则创建指向释放的输出缓冲区的指针
  gep = irb_.CreateInBoundsGEP(
      OpqPtrTy_,
      buf_ptrs,
      // @lint-ignore CLANGTIDY
      llvm::ConstantInt::getSigned(IntTy_, bufs_out_size + bufs_in_size + i));
  bufsExtToFreeVal_[buf_out->base_handle()] = irb_.CreateLoad(OpqPtrTy_, gep);
#else
    # 使用 LLVM IR 构造函数 CreateInBoundsGEP 创建一个指针指向偏移量处的内存位置
    gep = irb_.CreateInBoundsGEP(
        Int8PtrTy_,  # 指针类型为 Int8 指针
        buf_ptrs,    # 基础指针数组
        // @lint-ignore CLANGTIDY  # 忽略 CLANGTIDY 的静态分析警告
        llvm::ConstantInt::getSigned(IntTy_, bufs_out_size + bufs_in_size + i)  # 创建一个有符号整数常量
    )
    # 将 buf_out 对象的基础句柄作为键，将偏移位置 gep 的加载结果存储到 bufsExtToFreeVal_ 字典中
    bufsExtToFreeVal_[buf_out->base_handle()] = irb_.CreateLoad(Int8PtrTy_, gep);
#endif
  }
#if LLVM_VERSION_MAJOR >= 15
  // 如果 LLVM 主版本号大于等于 15，则使用 module_ 的 getOrInsertFunction 方法插入名为 "nnc_aten_free" 的函数声明
  FunctionCallee callee = module_->getOrInsertFunction(
      "nnc_aten_free",
      llvm::FunctionType::get(
          llvm::Type::getVoidTy(getContext()), // 返回类型为 void
          {
              LongTy_, // 参数类型为 int64_t bufs_num
              OpqPtrTy_, // 参数类型为 void** ptrs
          },
          false)); // 不支持可变参数

#else
  // 如果 LLVM 主版本号小于 15，则使用 module_ 的 getOrInsertFunction 方法插入名为 "nnc_aten_free" 的函数声明
  FunctionCallee callee = module_->getOrInsertFunction(
      "nnc_aten_free",
      llvm::FunctionType::get(
          llvm::Type::getVoidTy(getContext()), // 返回类型为 void
          {
              LongTy_, // 参数类型为 int64_t bufs_num
              Int8PtrTy_->getPointerTo(), // 参数类型为 void** ptrs
          },
          false)); // 不支持可变参数
#endif

  // 获取函数的类型和指针
  auto call_ty = callee.getFunctionType();
  auto call_fn = callee.getCallee();
  // 将函数标记为不抛出异常
  llvm::cast<llvm::Function>(call_fn)->addFnAttr(llvm::Attribute::NoUnwind);

  // 创建函数调用指令，调用名为 "nnc_aten_free" 的函数
  irb_.CreateCall(
      call_ty,
      call_fn,
      {llvm::ConstantInt::getSigned(LongTy_, bufs_num), ptrs});

  // 将 value_ 设置为常数 0
  value_ = llvm::ConstantInt::get(IntTy_, 0);
}
#ifdef TORCH_ENABLE_LLVM
// 如果 TORCH_ENABLE_LLVM 被定义，则使用新的 PassManager 架构进行优化

void LLVMCodeGenImpl::optimize(llvm::Module& M) {
  // 获取 JIT 对象的目标机器
  auto& TM = jit_->getTargetMachine();

  // 创建分析管理器
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // 创建新的 Pass 管理器构建器
  // 可以根据需要进行自定义，例如指定 TargetMachine 或不同的调试选项
  llvm::PassBuilder PB(&TM);

  // 注册 Pass 构建器的回调函数
  TM.registerPassBuilderCallbacks(PB);

  // 使用 Pass 构建器注册所有基本分析到对应的管理器中
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);

  // 交叉注册各个管理器的代理
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // 构建模块级默认优化管线
  llvm::ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

  // 构建函数简化管线
  llvm::FunctionPassManager FPM = PB.buildFunctionSimplificationPipeline(
      llvm::OptimizationLevel::O3, llvm::ThinOrFullLTOPhase::None);

  // 注册目标机器的 IR 分析到函数分析管理器
  FAM.registerPass([&] { return TM.getTargetIRAnalysis(); });

  // 添加循环向量化 Pass 和 SLP 向量化 Pass 到函数简化管线
  FPM.addPass(llvm::LoopVectorizePass());
  FPM.addPass(llvm::SLPVectorizerPass());

  // 添加死代码消除 Pass 到函数简化管线
  FPM.addPass(llvm::DCEPass());

  // 添加始终内联 Pass 到模块级管线
  MPM.addPass(llvm::AlwaysInlinerPass());

  // 运行模块级管线
  MPM.run(M, MAM);

  // 对每个函数进行迭代，运行函数简化管线
  for (auto& FF : M) {
    if (!FF.empty()) {
      FPM.run(FF, FAM);
    }
  }
}

#else // "Old" PassManager
// 如果 TORCH_ENABLE_LLVM 未定义，则使用旧的 PassManager 架构进行优化

void LLVMCodeGenImpl::optimize(llvm::Module& M) {
  // 创建函数 Pass 管理器和 Pass 管理器
  llvm::legacy::FunctionPassManager FPM(&M);
  llvm::legacy::PassManager PM;

  // 添加目标机器的内部分析 Pass
  auto& TM = jit_->getTargetMachine();
  PM.add(llvm::createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
  FPM.add(llvm::createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));

  // 创建 Pass 管理器构建器
  llvm::PassManagerBuilder PMB;
  PMB.OptLevel = 3;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;

  // 调整 Pass 管理器根据目标机器的特性
  TM.adjustPassManager(PMB);

  // 填充函数 Pass 管理器和 Pass 管理器
  PMB.populateFunctionPassManager(FPM);
  PMB.populateModulePassManager(PM);

  // 执行函数 Pass 管理器的初始化
  FPM.doInitialization();

  // 添加死代码消除 Pass 到 Pass 管理器
  PM.add(llvm::createDeadCodeEliminationPass());

  // 添加始终内联 Pass 到 Pass 管理器
  PM.add(llvm::createAlwaysInlinerLegacyPass());

  // 运行 Pass 管理器
  PM.run(M);

  // 对每个函数运行函数 Pass 管理器
  for (auto& FF : M) {
    FPM.run(FF);
  }

  // 执行函数 Pass 管理器的终结
  FPM.doFinalization();
}

#endif

// 注册 LLVMCodeGen 类到 LLVM 代码生成器
RegisterCodeGen<LLVMCodeGen> llvm_codegen_reg("llvm_codegen");

#endif // TORCH_ENABLE_LLVM
```