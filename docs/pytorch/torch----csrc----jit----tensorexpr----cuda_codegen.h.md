# `.\pytorch\torch\csrc\jit\tensorexpr\cuda_codegen.h`

```
#pragma once

#include <unordered_map>  // 引入无序映射
#include <unordered_set>  // 引入无序集合

#include <ATen/ATen.h>  // 引入 PyTorch 的 ATen 库
#include <ATen/cuda/CUDAContext.h>  // 引入 CUDA 上下文相关头文件
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>  // 引入 CUDA NVRTC 相关头文件
#include <c10/cuda/CUDACachingAllocator.h>  // 引入 CUDA 缓存分配器头文件
#include <c10/cuda/CUDAGuard.h>  // 引入 CUDA guard 头文件
#include <torch/csrc/jit/resource_guard.h>  // 引入 PyTorch JIT 资源保护头文件
#include <torch/csrc/jit/tensorexpr/codegen.h>  // 引入 PyTorch TensorExpr 代码生成头文件
#include <torch/csrc/jit/tensorexpr/eval.h>  // 引入 PyTorch TensorExpr 评估头文件
#include <torch/csrc/jit/tensorexpr/ir.h>  // 引入 PyTorch TensorExpr IR 头文件
#include <torch/csrc/jit/tensorexpr/ir_printer.h>  // 引入 PyTorch TensorExpr IR 打印头文件
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>  // 引入 PyTorch TensorExpr IR 访问者头文件
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>  // 引入 PyTorch TensorExpr LLVM 代码生成头文件
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>  // 引入 PyTorch TensorExpr 唯一名称管理器头文件

namespace torch {
namespace jit {
namespace tensorexpr {

// 分析适用于 CUDA 后端的给定程序的类。
class CudaAnalysis : public IRVisitor {
 public:
  CudaAnalysis() {
    // 初始化 GPU 块和线程的默认维度为 1
    gpu_block_extents_ = {alloc<IntImm>(1), alloc<IntImm>(1), alloc<IntImm>(1)};
    gpu_thread_extents_ = {
        alloc<IntImm>(1), alloc<IntImm>(1), alloc<IntImm>(1)};
  }

  // 检查给定的缓冲区是否是存储目标
  bool is_buf_store_target(BufPtr buf) const {
    return store_targets_.count(buf) > 0;
  }

  // 返回线程局部缓冲区集合的引用
  const std::unordered_set<VarPtr>& thread_local_bufs() const {
    return thread_local_bufs_;
  }

  // 返回跨块缓冲区集合的引用
  const std::unordered_set<VarPtr>& cross_block_bufs() const {
    return cross_block_bufs_;
  }

  // 返回 GPU 块的维度向量的引用
  const std::vector<ExprPtr>& gpu_block_extents() const {
    return gpu_block_extents_;
  }

  // 返回 GPU 线程的维度向量的引用
  const std::vector<ExprPtr>& gpu_thread_extents() const {
    return gpu_thread_extents_;
  }

 private:
  // 访问存储语句节点时调用的方法
  void visit(StorePtr v) override {
    store_targets_.insert(v->buf());
  }

  // 以下是需要实现的其他访问方法，略去具体实现

  std::unordered_set<BufPtr> store_targets_;  // 存储目标缓冲区集合
  std::unordered_set<VarPtr> thread_local_bufs_;  // 线程局部缓冲区集合
  std::unordered_set<VarPtr> cross_block_bufs_;  // 跨块缓冲区集合

  std::vector<ExprPtr> gpu_block_extents_;  // GPU 块维度向量
  std::vector<ExprPtr> gpu_thread_extents_;  // GPU 线程维度向量
};

// 将绑定循环选项替换为 CUDA 元变量，并掩盖应该执行的语句块的范围
class GPUMetaVarRewriter : public IRMutator {
 public:
  // 构造函数，使用给定的 CUDA 分析器初始化
  explicit GPUMetaVarRewriter(const CudaAnalysis* cuda_analysis)
      : cuda_analysis_(cuda_analysis) {
    // 初始化 GPU 块和线程的变量名称和类型
    gpu_block_vars_ = {
        alloc<Var>("blockIdx.x", kInt),
        alloc<Var>("blockIdx.y", kInt),
        alloc<Var>("blockIdx.z", kInt)};
    gpu_thread_vars_ = {
        alloc<Var>("threadIdx.x", kInt),
        alloc<Var>("threadIdx.y", kInt),
        alloc<Var>("threadIdx.z", kInt)};

    // 初始化当前块的可达性为默认值 1
    current_block_reach_ = {
        alloc<IntImm>(1), alloc<IntImm>(1), alloc<IntImm>(1)};


这段代码需要结合后续的代码继续完成，但这里只展示了注释部分。
    current_thread_reach_ = {
        alloc<IntImm>(1), alloc<IntImm>(1), alloc<IntImm>(1)};


    // 初始化 current_thread_reach_，分配三个 IntImm 类型的对象，每个值为 1
    current_thread_reach_ = {
        alloc<IntImm>(1), alloc<IntImm>(1), alloc<IntImm>(1)};
  }

  StmtPtr mutate(ForPtr v) override;
  StmtPtr mutate(BlockPtr v) override;

  const std::vector<VarPtr>& gpu_block_vars() const {
    // 返回存储 GPU block 变量的向量引用
    return gpu_block_vars_;
  }

  const std::vector<VarPtr>& gpu_thread_vars() const {
    // 返回存储 GPU thread 变量的向量引用
    return gpu_thread_vars_;
  }

  const std::vector<ExprPtr>& gpu_block_extents() const {
    // 返回 CUDA 分析对象中的 GPU block 尺寸向量引用
    return cuda_analysis_->gpu_block_extents();
  }

  const std::vector<ExprPtr>& gpu_thread_extents() const {
    // 返回 CUDA 分析对象中的 GPU thread 尺寸向量引用
    return cuda_analysis_->gpu_thread_extents();
  }

 private:
  // When processing a block, stores the contents of each sub-segment.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  class Segment {
   public:
    void reset(bool mask) {
      // 重置 Segment 对象，清空语句向量并设置掩码值
      stmts_.clear();
      mask_ = mask;
    }

    bool empty() const {
      // 检查 Segment 是否为空，即语句向量是否为空
      return stmts_.empty();
    }

    std::vector<StmtPtr>& stmts() {
      // 返回语句向量的引用
      return stmts_;
    }
    bool mask() {
      // 返回掩码值
      return mask_;
    }

   private:
    std::vector<StmtPtr> stmts_; // 存储语句的向量
    bool mask_{true}; // 控制 Segment 的掩码值，默认为 true
  };

  // Returns true if the current execution scope is equivalent to the launch
  // parameters.
  bool isFullExtent();

  std::vector<VarPtr> gpu_block_vars_; // 存储 GPU block 变量的向量
  std::vector<VarPtr> gpu_thread_vars_; // 存储 GPU thread 变量的向量

  std::vector<ExprPtr> current_block_reach_; // 存储当前块的 reach（未明确用途）
  std::vector<ExprPtr> current_thread_reach_; // 存储当前线程的 reach（未明确用途）

  const CudaAnalysis* cuda_analysis_; // 指向 CUDA 分析对象的指针
};

// 重写 IRPrinter 类，以生成 Cuda C 代码的类。
class CudaPrinter : public IRPrinter {
 public:
  explicit CudaPrinter(
      std::ostream* os,
      const CudaAnalysis* cuda_analysis,
      bool has_random)
      : IRPrinter(*os), cuda_analysis_(cuda_analysis) {
    // 如果需要随机函数，分配一个名为 "rand" 的变量
    if (has_random) {
      rand_func_ = alloc<Var>("rand", kHandle);
    }
  }

  // 下面是对不同类型的表达式进行访问并打印的方法的重写
  void visit(CastPtr v) override;
  void visit(IntrinsicsPtr v) override;
  void visit(ForPtr v) override;

  void visit(LoadPtr v) override;
  void visit(StorePtr v) override;
  void visit(AtomicAddPtr v) override;
  void visit(MaxPtr v) override;
  void visit(MinPtr v) override;
  void visit(IfThenElsePtr v) override;
  void visit(BlockPtr v) override;
  void visit(AllocatePtr v) override;
  void visit(FreePtr v) override;
  void visit(LetPtr v) override;

  void visit(ExternalCallPtr v) override;

  // 返回随机函数变量的方法
  VarPtr rand_func() const {
    return rand_func_;
  }

  // 将 Dtype 类型转换为 C++ 字符串的方法的重写
  std::string dtypeToCppString(const Dtype& dtype) override;

  // 继承 IRPrinter 类的成员和方法
  using IRPrinter::name_manager;
  using IRPrinter::visit;

 private:
  // 随机函数的变量指针
  VarPtr rand_func_;
  // Cuda 分析对象的指针
  const CudaAnalysis* cuda_analysis_;

  // 打印平坦分配的方法
  void print_flat_alloc(AllocatePtr alloc);
};

// 从缓冲区和张量输入构造 Cuda C 代码，并在提供实际参数时调用内核
class TORCH_CUDA_CU_API CudaCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CudaCodeGen(StmtPtr stmt, Ts... ts)
      : CodeGen(
            stmt,
            std::vector<BufferArg>({BufferArg(ts)...}),
            at::Device(at::kCUDA, at::cuda::current_device())) {
    // 初始化
    Initialize();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CudaCodeGen(
      StmtPtr stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::Device(at::kCUDA, at::cuda::current_device()),
      const std::string& kernel_func_name = "func")
      : CodeGen(stmt, buffer_args, device, kernel_func_name) {
    // 初始化
    Initialize();
  }

  // 析构函数，释放资源
  ~CudaCodeGen() override;

  // 调用内核的方法，接受参数列表
  void call(const std::vector<CallArg>& args) override;
  // 使用原始指针调用内核的方法，接受参数列表
  void call_raw(const std::vector<void*>& args) override;
  // 使用元素数量调用内核的方法，接受参数列表
  void call_with_numel(void** args, int64_t numel) override;

  // 操作符重载，直接调用 call 方法
  template <typename... Ts>
  void operator()(const Ts&... ts) {
    call(std::vector<CallArg>({CallArg(ts)...}));
  }

  // 创建一个具有指定属性的空跨步张量
  at::Tensor empty_strided(
      c10::IntArrayRef size,
      c10::IntArrayRef stride,
      std::optional<c10::ScalarType> dtype_opt,
      std::optional<c10::Layout> layout_opt,
      std::optional<c10::Device> device_opt,
      std::optional<bool> pin_memory_opt) override;

  // 返回 Cuda 分析对象的 GPU 块大小的引用
  const std::vector<ExprPtr>& gpu_block_extents() const {
    return cuda_analysis_->gpu_block_extents();
  }

  // 返回 Cuda 分析对象的 GPU 线程大小的引用
  const std::vector<ExprPtr>& gpu_thread_extents() const {
    return cuda_analysis_->gpu_thread_extents();
  }

  // 获取生成的 Cuda 代码的文本表示
  std::string getCodeText(const std::string& attr = "") override {
  // 返回私有成员变量 oss_ 的字符串表示
  return oss_.str();
}

private:
  // 初始化函数，用于初始化对象的各个成员变量
  void Initialize();

  // 编译给定代码到 NVRTC，生成指定函数名的 CUDA 函数
  void CompileToNVRTC(const std::string& code, const std::string& func_name);

  // 获取名称管理器指针，如果打印器为空则抛出运行时错误
  UniqueNameManager* name_manager() {
    if (!printer_) {
      throw std::runtime_error("Null IRPrinter is not expected");
    }
    return printer_->name_manager();
  }

  // 获取输出流的引用
  std::ostream& os() {
    return printer_->os();
  }

  // 字符串流对象，用于临时存储字符串数据
  std::ostringstream oss_;

  // 独占指针，用于 CUDA 打印器对象
  std::unique_ptr<CudaPrinter> printer_;

  // 独占指针，用于 CUDA 分析对象
  std::unique_ptr<CudaAnalysis> cuda_analysis_;

  // 独占指针，用于 GPU 元变量重写器对象
  std::unique_ptr<GPUMetaVarRewriter> metavar_rewriter_;

  // 存储已使用函数名的无序集合
  std::unordered_set<std::string> taken_func_names;

  // 互斥量，用于多线程下的评估操作的锁
  std::mutex eval_lock_;

  // CUDA 函数句柄
  CUfunction function_;

  // 标记是否包含随机数生成
  bool has_random_ = false;

  // 线程块大小，默认为 -1
  int thread_block_size_ = -1;

  // 参数在扩展中的位置标记向量
  std::vector<bool> arg_pos_in_extents_;
#ifdef TORCH_ENABLE_LLVM
  // 如果定义了 TORCH_ENABLE_LLVM 宏，则使用 LLVMCodeGen 作为模板参数创建 ExprEval 对象的向量
  std::vector<ExprEval<LLVMCodeGen>> block_extents_eval_;
  std::vector<ExprEval<LLVMCodeGen>> thread_extents_eval_;
#else
  // 如果未定义 TORCH_ENABLE_LLVM 宏，则使用 SimpleIREvaluator 作为模板参数创建 ExprEval 对象的向量
  std::vector<ExprEval<SimpleIREvaluator>> block_extents_eval_;
  std::vector<ExprEval<SimpleIREvaluator>> thread_extents_eval_;
#endif

// 返回一个唯一的函数名，该函数名由给定的前缀 func_prefix 构成
std::string GetUniqueFuncName(const std::string& func_prefix);
};

// 结束 namespace tensorexpr
} // namespace jit
// 结束 namespace torch
} // namespace torch
```