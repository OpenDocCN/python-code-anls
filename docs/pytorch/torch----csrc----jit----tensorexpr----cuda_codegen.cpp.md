# `.\pytorch\torch\csrc\jit\tensorexpr\cuda_codegen.cpp`

```py
// 引入 Torch 的 CUDA Tensor 表达式相关头文件
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#include <torch/csrc/jit/tensorexpr/half_support.h>

// 引入 ATen 的 CUDA 上下文、生成器实现、CUDA JIT 工具等相关头文件
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/irange.h>

// 引入 Torch 的 CUDA 融合内核、资源字符串、日志等相关头文件
#include <torch/csrc/jit/codegen/fuser/cuda/fused_kernel.h>
#include <torch/csrc/jit/codegen/fuser/cuda/resource_strings.h>
#include <torch/csrc/jit/jit_log.h>

// 引入 Torch 的 Tensor 表达式分析、CUDA 随机数生成、求值、异常处理、IR 简化、寄存器化等相关头文件
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/cuda_random.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/registerizer.h>

// 根据宏定义选择引入 ATen 的原生函数或特定操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_strided_native.h>
#endif

// 引入标准库中的无序映射容器
#include <unordered_map>

// Torch JIT TensorExpr 命名空间
namespace torch::jit::tensorexpr {

// RAII 包装类，用于管理变量及其名称的查找表中的变量和名称对
// TODO: 将此类移动到更多共享的位置
class ScopedVarName {
 public:
  ScopedVarName(VarNameMap* mapping, VarPtr var, const std::string& name)
      : mapping_(mapping), var_(var) {
    // 检查是否存在重复的变量条目，若有则抛出运行时错误
    auto iter = mapping->find(var);
    if (iter != mapping->end()) {
      throw std::runtime_error("Duplicate var entry: " + var->name_hint());
    }
    // 将变量及其名称插入映射表中
    mapping->insert(std::make_pair(var, name));
  }

  // 构造函数重载，利用唯一名称管理器插入变量及其名称到映射表中
  ScopedVarName(UniqueNameManager* manager, VarPtr var, const std::string& name)
      : ScopedVarName(&manager->unique_name_mapping_, var, name) {}

  // 禁用拷贝构造函数和赋值运算符
  ScopedVarName(const ScopedVarName&) = delete;
  ScopedVarName& operator=(const ScopedVarName&) = delete;

  // 析构函数，用于从映射表中移除变量
  ~ScopedVarName() noexcept(false) {
    mapping_->erase(var_);
  }

 private:
  VarNameMap* mapping_ = nullptr; // 变量名称映射指针
  VarPtr var_ = nullptr; // 变量指针
};

// 判断表达式是否为零的静态函数
static bool is_zero(ExprPtr expr) {
  // 获取表达式的整数值，并判断其是否为零
  auto v = intValue(expr);
  return v && *v == 0;
}

// 返回全局 CUDA 上下文中的 NVRTC 实例的引用
static const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

// 将 Dtype 类型转换为对应的 C++ 字符串表示
std::string CudaPrinter::dtypeToCppString(const Dtype& dtype) {
  // 根据标量类型选择相应的 C++ 字符串表示
  switch (dtype.scalar_type()) {
    case ScalarType::Bool:
      return "bool";
    case ScalarType::Half:
      return "half";
    case ScalarType::BFloat16:
      return "__nv_bfloat16";
    case ScalarType::Char:
      return "char";
    case ScalarType::Byte:
      return "unsigned char";
    case ScalarType::Short:
      return "short";
    case ScalarType::Long:
      return "long long";
    default:
      return dtype.ToCppString();
  }
}

// 访问 FreePtr 对象的方法，如果缓冲区变量不在本地线程或跨块缓冲区集合中则抛出异常
void CudaAnalysis::visit(FreePtr v) {
  if (thread_local_bufs_.count(v->buffer_var()) == 0 &&
      cross_block_bufs_.count(v->buffer_var()) == 0) {
    throw std::runtime_error("Global free not supported yet");
  }
}

// 访问 AllocatePtr 对象的方法，用于分析分配操作的上下文
void CudaAnalysis::visit(AllocatePtr v) {
  StmtPtr p = v->get_parent();
  while (p) {
    ForPtr for_v = to<For>(p); // 尝试将父语句转换为 For 循环对象
    if (for_v) { // 如果成功转换为 For 循环对象
      // 分配操作的上下文检查完成，退出循环
      break;
    }
    p = p->get_parent(); // 获取当前语句的父语句，继续检查上级语境
  }
  if (!p) { // 如果未找到 For 循环作为 AllocatePtr 的父语句
    throw std::runtime_error("Allocate without loop nest not supported yet");
  }
}
    // 如果存在循环变量
    if (for_v) {
      // 如果循环变量的循环选项是 GPU 块索引
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (for_v->loop_options().is_gpu_block_index()) {
        // TODO: 如果在比这更高级别的地方有线程索引，则这样做是不正确的
        // 将变量的缓冲区插入跨块缓冲区集合中
        cross_block_bufs_.insert(v->buffer_var());
        // 返回
        return;
      } else if (for_v->loop_options().is_gpu_thread_index()) {
        // 将变量的缓冲区插入线程本地缓冲区集合中
        thread_local_bufs_.insert(v->buffer_var());
        // 返回
        return;
      }
    }
    // 获取父节点
    p = p->get_parent();
  }
  // 抛出异常，表示全局分配尚未支持
  throw std::runtime_error("Global alloc not supported yet");
void CudaAnalysis::visit(PlacementAllocatePtr v) {
  // 抛出异常，暂不支持内存重用
  throw std::runtime_error("Memory reuse not supported yet");
}

void CudaAnalysis::visit(ForPtr v) {
  // 首先递归访问循环体
  v->body()->accept(this);

  // 获取循环的选项信息
  const LoopOptions& loop_options = v->loop_options();
  
  // 如果循环是 GPU 块索引
  if (loop_options.is_gpu_block_index()) {
    int gpu_block_index = loop_options.gpu_block_index();
    // 检查 GPU 块索引是否超过了三维限制
    if (gpu_block_index >= 3) {
      throw std::runtime_error("support only 3D gpu_block_index");
    }
    
    ExprPtr prev = nullptr;
    
    // 检查 GPU 块大小是否需要扩展
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (gpu_block_extents_.size() <= static_cast<size_t>(gpu_block_index)) {
      gpu_block_extents_.resize(gpu_block_index + 1);
    } else {
      prev = gpu_block_extents_[gpu_block_index];
    }
    
    // 检查循环起始位置是否为零
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_block_index: " +
          std::to_string(v->start()));
    }

    // 根据条件更新 GPU 块的大小信息
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (prev == nullptr) {
      gpu_block_extents_[gpu_block_index] = v->stop();
    } else if (prev->isConstant() && immediateEquals(prev, 1)) {
      // 如果前一个大小是常数 1，则即使停止符号化，它也是最大的
      gpu_block_extents_[gpu_block_index] = v->stop();
    } else {
      gpu_block_extents_[gpu_block_index] =
          IRSimplifier::simplify(alloc<Max>(prev, v->stop(), true));
    }
  } else if (loop_options.is_gpu_thread_index()) {
    // 如果循环是 GPU 线程索引
    int gpu_thread_index = loop_options.gpu_thread_index();
    // 检查 GPU 线程索引是否超过了三维限制
    if (gpu_thread_index >= 3) {
      throw std::runtime_error("support only 3D gpu_thread_index");
    }
    
    ExprPtr prev = nullptr;
    
    // 检查 GPU 线程大小是否需要扩展
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (gpu_thread_extents_.size() <= static_cast<size_t>(gpu_thread_index)) {
      gpu_thread_extents_.resize(gpu_thread_index + 1);
    } else {
      prev = gpu_thread_extents_[gpu_thread_index];
    }
    
    // 检查循环起始位置是否为零
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_thread_index: " +
          std::to_string(v->start()));
    }

    // 根据条件更新 GPU 线程的大小信息
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (prev == nullptr) {
      gpu_thread_extents_[gpu_thread_index] = v->stop();
    } else if (prev->isConstant() && immediateEquals(prev, 1)) {
      // 如果前一个大小是常数 1，则即使停止符号化，它也是最大的
      gpu_thread_extents_[gpu_thread_index] = v->stop();
    } else {
      gpu_thread_extents_[gpu_thread_index] =
          IRSimplifier::simplify(alloc<Max>(prev, v->stop(), true));
    }
  }
}

void CudaPrinter::print_flat_alloc(AllocatePtr alloc) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprPtr> dims = alloc->dims();
  
  // TODO: 这部分应与存储平铺器合并
  
  int64_t flat_size = 1;
  // 计算分配的扁平化大小
  for (auto dim : dims) {
    auto dim_i = intValue(dim);
    if (dim_i) {
      flat_size *= *dim_i;
    }
    } else {
      // 如果不是整数维度，则抛出运行时错误
      throw std::runtime_error("Only integer dimensions are supported for now");
    }
  }
  // 输出分配器的数据类型作为 C++ 字符串表示，以及其缓冲区变量名和其扁平化大小
  os() << dtypeToCppString(alloc->dtype()) << " " << (*alloc->buffer_var())
       << "[" << flat_size << "];" << std::endl;
}

void CudaPrinter::visit(AllocatePtr v) {
  // TODO: handle dynamic shapes here.
  // 检查是否为跨块缓冲区，如果是，则输出 __shared__ 和分配语句
  if (cuda_analysis_->cross_block_bufs().count(v->buffer_var()) != 0) {
    emitIndent(); // 输出当前缩进
    os() << "__shared__ "; // 输出 __shared__ 关键字
    print_flat_alloc(v); // 打印分配语句
    return; // 结束函数
  }

  // 检查是否为线程局部缓冲区，如果是，则直接打印分配语句
  if (cuda_analysis_->thread_local_bufs().count(v->buffer_var()) != 0) {
    emitIndent(); // 输出当前缩进
    print_flat_alloc(v); // 打印分配语句
    return; // 结束函数
  }

  // 如果既不是跨块缓冲区也不是线程局部缓冲区，则抛出运行时错误
  throw std::runtime_error("Encountered Alloc not local to block or thread");
}

void CudaPrinter::visit(FreePtr v) {
  // do nothing
  // 什么也不做，即空函数
}

void CudaPrinter::visit(ForPtr v) {
  IRPrinter::visit(v); // 调用父类 IRPrinter 的 visit 方法
}

void CudaPrinter::visit(CastPtr v) {
  // 根据数据类型选择相应的转换函数
  std::string castFn = v->dtype().scalar_type() == ScalarType::Half
      ? "__float2half"
      : v->dtype().scalar_type() == ScalarType::BFloat16 ? "__float2bfloat16"
      : v->src_value()->dtype().scalar_type() == ScalarType::Half
      ? "__half2float"
      : v->src_value()->dtype().scalar_type() == ScalarType::BFloat16
      ? "__bfloat162float"
      : ("(" + dtypeToCppString(v->dtype()) + ")");
  os() << castFn << "("; // 输出转换函数及其参数的起始部分
  v->src_value()->accept(this); // 访问源值表达式
  os() << ")"; // 输出转换函数及其参数的结束部分
}

void CudaPrinter::visit(IntrinsicsPtr v) {
  if (v->op_type() == IntrinsicsOp::kRand) {
    // 对于随机数操作，输出相应的 CUDA 函数调用
    os() << "Uint32ToFloat(" << *rand_func_ << "())";
    return; // 结束函数
  }

  std::string func_name = v->func_name();

  // 获取表达式结果类型
  ScalarType returnType = v->param(0)->dtype().scalar_type();
  for (int i = 1; i < v->nparams(); ++i) {
    returnType = promoteTypes(returnType, v->param(i)->dtype().scalar_type());
  }

  // 如果返回类型是半精度或单精度浮点数，则追加 'f' 到函数名
  if (returnType == ScalarType::Half || returnType == ScalarType::Float) {
    func_name = func_name + "f";
  }

  // 对于取绝对值操作且返回类型不是整数型，追加 'f' 到函数名
  if (v->op_type() == IntrinsicsOp::kAbs &&
      !c10::isIntegralType(returnType, true)) {
    func_name = "f" + func_name;
  }

  // 对于判断是否为 NaN 的操作，将函数名设置为 "isnan"
  if (v->op_type() == IntrinsicsOp::kIsNan) {
    func_name = "isnan";
  }

  // 输出最终的函数名及其参数列表
  os() << func_name << "(";
  for (const auto i : c10::irange(v->nparams())) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->param(i);
  }
  os() << ")";
}

void CudaPrinter::visit(ExternalCallPtr v) {
  // 抛出未实现的降级异常
  throw unimplemented_lowering(v);
}

void CudaPrinter::visit(LoadPtr v) {
  // TODO: find a better metric in using ldg or not. Support different dtypes.
  // 检测是否是简单的基础地址加载，若是则直接输出基础地址
  if (v->indices().empty()) {
    os() << *v->base_handle();
    return; // 结束函数
  }

  // 对于布尔型、半精度或 BFloat16 类型，直接输出基础地址及索引
  if (v->dtype().scalar_type() == ScalarType::Bool ||
      v->dtype().scalar_type() == ScalarType::Half ||
      v->dtype().scalar_type() == ScalarType::BFloat16) {
    os() << *v->base_handle() << "[" << *v->flat_index() << "]";
    return; // 结束函数
  }

  // 如果是写入目标缓冲区，则使用 CUDA 的 ldg 加载函数
  if (cuda_analysis_->is_buf_store_target(v->buf())) {
    os() << *v->base_handle(); // 输出基础地址
    // 省略 ldg 的实现，因为未提供完整代码
  }
}
    os() << *v->base_handle() << "[" << *v->flat_index() << "]";
    return;
  }
  os() << "__ldg(" << *v->base_handle() << " + " << *v->flat_index() << ")";
}

// TODO: maybe this should be a more shared location?
// TODO: investigate how "ExprPtr" can be implicitly converted to "ExprHandle"
// as a bool.
// 定义静态函数 CheckEqual，用于比较两个表达式指针是否相等
static bool CheckEqual(ExprPtr lhs, ExprPtr rhs) {
  // The fast path. Checks if the pointers are the same.
  // 快速路径。检查指针是否相同。
  if (lhs == rhs) {
    return true;
  }
  // 使用 Sub::make 函数创建差值表达式 diff
  ExprHandle diff = Sub::make(ExprHandle(lhs), ExprHandle(rhs));
  // 对差值表达式进行简化
  ExprHandle diff_s = IRSimplifier::simplify(diff);
  // 检查简化后的表达式是否等于零
  return immediateEquals(diff_s.node(), 0);
}

// 定义 AtomicAddFuser 类，继承自 IRMutator
class AtomicAddFuser : public IRMutator {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，接受 thread_local_bufs 和 metavars 作为参数
  AtomicAddFuser(
      const std::unordered_set<VarPtr>& thread_local_bufs,
      const GPUMetaVarRewriter& metavars)
      : thread_local_bufs_(thread_local_bufs) {
    // 处理 GPU 块维度变量
    const std::vector<ExprPtr>& block_extents = metavars.gpu_block_extents();
    const std::vector<VarPtr>& block_vars = metavars.gpu_block_vars();
    for (size_t i = 0; i < block_extents.size(); ++i) {
      // 创建 MetaVarExtent 对象 extent，并根据是否为常量值为 1 进行标记
      MetaVarExtent extent{block_extents[i], false};
      if (extent.expr->isConstant() && immediateEquals(extent.expr, 1)) {
        extent.trivial = true;
      } else {
        nontrivial_metavars_.insert(block_vars[i]);
      }
      metavars_[block_vars[i]] = extent;
    }

    // 处理 GPU 线程维度变量
    const std::vector<ExprPtr>& thread_extents = metavars.gpu_thread_extents();
    const std::vector<VarPtr>& thread_vars = metavars.gpu_thread_vars();
    for (size_t i = 0; i < thread_extents.size(); ++i) {
      // 创建 MetaVarExtent 对象 extent，并根据是否为常量值为 1 进行标记
      MetaVarExtent extent{thread_extents[i], false};
      if (extent.expr->isConstant() && immediateEquals(extent.expr, 1)) {
        extent.trivial = true;
      } else {
        nontrivial_metavars_.insert(thread_vars[i]);
      }
      metavars_[thread_vars[i]] = extent;
    }
  }

  // 重写 mutate 函数，处理 StorePtr 类型的节点
  StmtPtr mutate(StorePtr v) override {
    // 获取存储操作的缓冲区指针
    BufPtr buf = v->buf();

    // 如果是线程局部缓冲区，则不需要原子操作
    if (thread_local_bufs_.count(buf->base_handle()) != 0) {
      return v;
    }

    // 获取值的数据类型
    ScalarType dtype = v->value()->dtype().scalar_type();
    // 如果数据类型不是 Float 或 Double，则不处理
    if (dtype != ScalarType::Float && dtype != ScalarType::Double) {
      return v;
    }
    // 尝试将值转换为 AddPtr 类型
    AddPtr add_v = to<Add>(v->value());
    if (!add_v) {
      return v;
    }
    // 尝试将左操作数转换为 LoadPtr 类型
    LoadPtr load_v = to<Load>(add_v->lhs());
    if (!load_v) {
      return v;
    }
    // 检查存储节点的基本句柄是否与加载节点的基本句柄相同
    if (v->base_handle() != load_v->base_handle()) {
      return v;
    }
    // 检查存储节点和加载节点的索引是否为空，并且索引是否相同
    if (v->indices().empty() && load_v->indices().empty()) {
      return v;
    }
    // 检查索引是否相等
    bool index_equal = CheckEqual(v->flat_index(), load_v->flat_index());
    if (!index_equal) {
      return v;
    }

    // TODO: this checks that the metavars occur directly as an index, but this
    // is pessimistic, blockIdx.x + 1 is fine too if there is no overlapping.
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 创建待查找的变量集合，初始化为非平凡 metavars_
    std::unordered_set<VarPtr> vars_to_find = nontrivial_metavars_;
    // 遍历存储节点的索引表达式
    for (ExprPtr e : v->indices()) {
      // 如果索引表达式可以转换为 VarPtr 类型，则从待查找变量集合中删除
      if (VarPtr v = to<Var>(e)) {
        vars_to_find.erase(v);
      }
    }
    if (vars_to_find.empty()) {
      // 如果 vars_to_find 集合为空
      // 说明所有元变量都已处理完毕，直接返回当前表达式 v
      return v;
    }

    // 如果 vars_to_find 集合不为空，则说明还有未处理的元变量

    // 创建并返回一个 AtomicAdd 对象，用于向 buf 中添加 v->indices() 和 add_v->rhs() 的计算结果
    return alloc<AtomicAdd>(buf, v->indices(), add_v->rhs());
  }

 private:
  // 线程局部缓存的变量集合
  const std::unordered_set<VarPtr>& thread_local_bufs_;
  // 元变量的映射，每个元变量关联到一个 MetaVarExtent 结构
  struct MetaVarExtent {
    ExprPtr expr{nullptr};  // 元变量的表达式
    bool trivial{false};    // 元变量是否为平凡（trivial）
  };
  // 元变量到其扩展信息的映射
  std::unordered_map<VarPtr, MetaVarExtent> metavars_;
  // 非平凡元变量的集合
  std::unordered_set<VarPtr> nontrivial_metavars_;
// void类型函数CudaPrinter::visit的具体实现，用于处理StorePtr类型的指针v
void CudaPrinter::visit(StorePtr v) {
  // 输出当前缩进
  emitIndent();
  // 如果v的索引为空
  if (v->indices().empty()) {
    // 输出v的基础句柄和赋值符号
    os() << *v->base_handle() << " = ";
  } else {
    // 否则输出v的基础句柄、方括号以及扁平化索引，并附加赋值符号
    os() << *v->base_handle() << "[" << *v->flat_index() << "] = ";
  }
  // 输出v的值，并结束当前语句行
  os() << *v->value() << ";";
  os() << std::endl;
}

// void类型函数CudaPrinter::visit的具体实现，用于处理AtomicAddPtr类型的指针v
void CudaPrinter::visit(AtomicAddPtr v) {
  // 输出当前缩进
  emitIndent();
  // 如果v的基础句柄在CUDA分析的线程局部缓冲中
  if (cuda_analysis_->thread_local_bufs().count(v->base_handle()) > 0) {
    // 输出基础句柄、方括号、扁平化索引以及递增赋值操作
    os() << *v->base_handle() << "[" << *v->flat_index()
         << "] += " << *v->value() << ";";
  } else {
    // 否则输出原子增加操作的字符串表示
    os() << "atomicAdd(&" << *v->base_handle() << "[" << *v->flat_index() << "]"
         << ", " << *v->value() << ");";
  }
  // 结束当前语句行
  os() << std::endl;
}

// void类型函数CudaPrinter::visit的具体实现，用于处理MaxPtr类型的指针v
void CudaPrinter::visit(MaxPtr v) {
  // 如果v的数据类型为整数型
  if (v->dtype().is_integral()) {
    // 输出max函数
    os() << "max(";
  } else {
    // 否则输出maximum函数
    os() << "maximum(";
  }
  // 访问v的左操作数，并输出逗号
  v->lhs()->accept(this);
  os() << ",";
  // 访问v的右操作数，并输出右括号
  v->rhs()->accept(this);
  os() << ")";
}

// void类型函数CudaPrinter::visit的具体实现，用于处理MinPtr类型的指针v
void CudaPrinter::visit(MinPtr v) {
  // 如果v的数据类型为整数型
  if (v->dtype().is_integral()) {
    // 输出min函数
    os() << "min(";
  } else {
    // 否则输出minimum函数
    os() << "minimum(";
  }
  // 访问v的左操作数，并输出逗号
  v->lhs()->accept(this);
  os() << ",";
  // 访问v的右操作数，并输出右括号
  v->rhs()->accept(this);
  os() << ")";
}

// void类型函数CudaPrinter::visit的具体实现，用于处理IfThenElsePtr类型的指针v
void CudaPrinter::visit(IfThenElsePtr v) {
  // 输出条件表达式的起始括号
  os() << "((";
  // 访问条件部分，并输出条件判断符号
  v->condition()->accept(this);
  os() << ") ? ";
  // 访问true分支，并输出冒号分隔符
  v->true_value()->accept(this);
  os() << " : ";
  // 访问false分支
  v->false_value()->accept(this);
  os() << ")";
}

// void类型函数CudaPrinter::visit的具体实现，用于处理BlockPtr类型的指针v
void CudaPrinter::visit(BlockPtr v) {
  // 输出块的起始大括号，并换行
  os() << "{" << std::endl;
  // 增加缩进计数
  indent_++;

  // 遍历块中的语句列表
  for (StmtPtr s : v->stmts()) {
    // 访问每个语句
    s->accept(this);
  }

  // 减少缩进计数
  indent_--;
  // 输出缩进，并输出块的结束大括号
  emitIndent();
  os() << "}";
}

// void类型函数CudaPrinter::visit的具体实现，用于处理LetPtr类型的指针v
void CudaPrinter::visit(LetPtr v) {
  // 输出当前缩进
  emitIndent();
  // 输出变量类型的C++字符串表示和变量名，并赋值符号
  os() << dtypeToCppString(v->var()->dtype());
  os() << " " << *v->var() << " = ";
  // 访问变量的值，并输出结束分号
  v->value()->accept(this);
  os() << ";" << std::endl;
}

// 类PrioritizeLoad的构造函数，忽略特定类型的成员初始化警告
class PrioritizeLoad : public IRMutator {
 public:
  // 重写LoadPtr类型的mutate函数
  ExprPtr mutate(LoadPtr v) override {
    // 查看变量声明的具体细节
    if (nested_if_then_else_ > 0) {
      // 如果嵌套的条件语句数大于0，则继续使用基类的mutate函数
      return IRMutator::mutate(v);
    }
    if (nested_let_) {
      // 如果存在嵌套的let语句，则继续使用基类的mutate函数
      return IRMutator::mutate(v);
    }
    if (thread_local_bufs_.count(v->base_handle()) > 0) {
      // 如果在线程局部缓冲中存在v的基础句柄，则继续使用基类的mutate函数
      return IRMutator::mutate(v);
    }
    if (v->indices().size() == 0) {
      // 如果v的索引大小为0，则继续使用基类的mutate函数
      return IRMutator::mutate(v);
    }
    if (nested_store_) {
      // 如果存在嵌套的存储语句
      if (v->base_handle() == nested_store_->buf()->base_handle() &&
          v->indices().size() == nested_store_->indices().size()) {
        // 检查索引是否相同
        bool same = true;
        for (const auto i : c10::irange(v->indices().size())) {
          if (!exprEquals(v->indices()[i], nested_store_->indices()[i])) {
            same = false;
            break;
          }
        }
        if (same) {
          // 如果索引相同，则继续使用基类的mutate函数
          return IRMutator::mutate(v);
        }
      } else if (nested_store_->indices().empty()) {
        // 如果嵌套存储的索引为空，则继续使用基类的mutate函数
        return IRMutator::mutate(v);
      }
    }

    // 返回当前加载列表的最后一个加载内存变量，并分配新的变量
    MemLoadList& load_list = load_stack_.back();
    VarPtr load_new_var = alloc<Var>("v", v->dtype());
    // 省略部分处理...
  ExprPtr new_value = IRMutator::mutate(v);
  // 对给定的表达式节点 v 进行变异处理，返回变异后的新表达式节点 new_value
  load_list.push_back(std::make_pair(load_new_var, new_value));
  // 将 load_new_var 和 new_value 组成的 pair 放入 load_list 中

  return load_new_var;
}

ExprPtr mutate(CastPtr v) override {
  LoadPtr src_load = to<Load>(v->src_value());
  // 尝试将 v 的源值强制转换为 LoadPtr 类型的指针 src_load
  ExprPtr new_src = v->src_value()->accept_mutator(this);
  // 对 v 的源值进行变异处理，返回变异后的新表达式节点 new_src
  VarPtr new_var = to<Var>(new_src);
  // 将 new_src 尝试转换为 VarPtr 类型的指针 new_var
  if (!src_load || !new_var) {
    // 如果源值不是 LoadPtr 类型或者 new_src 不是 VarPtr 类型，则执行以下操作
    return alloc<Cast>(v->dtype(), new_src);
    // 分配并返回一个新的 CastPtr 节点，强制转换类型为 v 的 dtype
  }

  // We just did the prioritize load, let's fold in the Cast.
  MemLoadList& load_list = load_stack_.back();
  // 获取 load_stack_ 中的最后一个 MemLoadList 对象的引用
  assert(!load_list.empty());
  auto pair = load_list.back();
  // 获取 load_list 中最后一个元素所组成的 pair
  assert(pair.first == new_var);
  // 断言 pair 的第一个元素为 new_var
  load_list.pop_back();
  // 弹出 load_list 中的最后一个元素

  new_var = alloc<Var>("v", v->dtype());
  // 分配一个名为 "v"，类型为 v 的 dtype 的新的 VarPtr 节点
  ExprPtr new_value = alloc<Cast>(v->dtype(), pair.second);
  // 分配一个将 pair 的第二个元素进行强制转换为 v 的 dtype 的新 CastPtr 节点
  load_list.push_back(std::make_pair(new_var, new_value));
  // 将 new_var 和 new_value 组成的 pair 放入 load_list 中
  return new_var;
}

StmtPtr mutate(StorePtr v) override {
  StorePtr last = nested_store_;
  // 将当前的 nested_store_ 赋值给 last
  nested_store_ = v;
  // 将 v 赋值给 nested_store_
  StmtPtr s = IRMutator::mutate(v);
  // 对 v 进行变异处理，返回变异后的新语句节点 s
  nested_store_ = last;
  // 将 last 赋值给 nested_store_
  return s;
}

StmtPtr mutate(LetPtr v) override {
  nested_let_ = true;
  // 将 nested_let_ 标志设置为 true
  StmtPtr s = IRMutator::mutate(v);
  // 对 v 进行变异处理，返回变异后的新语句节点 s
  nested_let_ = false;
  // 将 nested_let_ 标志设置为 false
  return s;
}

StmtPtr mutate(BlockPtr v) override {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::list<StmtPtr> stmts = v->stmts();
  // 获取 BlockPtr v 中的语句列表 stmts
  for (StmtPtr stmt : stmts) {
    // 遍历语句列表 stmts 中的每个语句 stmt
    PushList();
    // 将一个空的 MemLoadList 放入 load_stack_ 中
    StmtPtr stmt_new = stmt->accept_mutator(this);
    // 对当前的语句 stmt 进行变异处理，返回变异后的新语句节点 stmt_new

    AddMemLoadsFromList(v, stmt);
    // 将 load_stack_ 中的加载项添加到 BlockPtr v 的末尾
    PopList();
    // 从 load_stack_ 中弹出最后一个加载项

    if (stmt_new == stmt) {
      continue;
    }
    // 如果 stmt_new 和 stmt 相同，则继续下一个循环
    v->replace_stmt(stmt, stmt_new);
    // 用 stmt_new 替换 BlockPtr v 中的 stmt
  }
  return v;
}

ExprPtr mutate(IfThenElsePtr v) override {
  nested_if_then_else_++;
  // 增加 nested_if_then_else_ 计数器
  ExprPtr new_v = IRMutator::mutate(v);
  // 对 v 进行变异处理，返回变异后的新表达式节点 new_v
  nested_if_then_else_--;
  // 减少 nested_if_then_else_ 计数器
  return new_v;
}
};

// 定义 CudaCodeGen 类中的 GetUniqueFuncName 方法，返回一个唯一的函数名
std::string CudaCodeGen::GetUniqueFuncName(const std::string& func_prefix) {
  // 初始化计数器为 0
  int64_t counter = 0;
  // 初始函数名为给定的前缀
  std::string name = func_prefix;
  // 当函数名已被占用时，添加计数器作为后缀，直到找到一个唯一的函数名
  while (taken_func_names.count(name)) {
    name = func_prefix + "_" + std::to_string(counter++);
  }

  // 将新生成的唯一函数名添加到已占用函数名集合中
  taken_func_names.insert(name);
  // 返回唯一的函数名
  return name;
}

// 定义 GPUMetaVarRewriter 类中的 isFullExtent 方法，检查当前是否是完整的扩展
bool GPUMetaVarRewriter::isFullExtent() {
  {
    // 获取 GPU 块的边界范围
    auto& extents = cuda_analysis_->gpu_block_extents();
    // 检查当前块的范围是否与 GPU 块的边界范围相等
    for (int i = 0; i < 3; ++i) {
      if (!exprEquals(current_block_reach_[i], extents[i])) {
        return false;
      }
    }
  }

  {
    // 获取 GPU 线程的边界范围
    auto& extents = cuda_analysis_->gpu_thread_extents();
    // 检查当前线程的范围是否与 GPU 线程的边界范围相等
    for (int i = 0; i < 3; ++i) {
      if (!exprEquals(current_thread_reach_[i], extents[i])) {
        return false;
      }
    }
  }

  // 如果所有范围都匹配，则返回 true，表示当前是完整的扩展
  return true;
}

// 定义 GPUMetaVarRewriter 类中的 mutate 方法，用于修改 For 循环语句
StmtPtr GPUMetaVarRewriter::mutate(ForPtr v) {
  // 获取 For 循环的主体语句
  StmtPtr body = v->body();
  // 初始化旧的达到（reach）表达式为 nullptr
  ExprPtr old_reach = nullptr;
  // 获取循环的选项设置
  const LoopOptions& loop_options = v->loop_options();

  // 如果循环是 GPU 块索引
  if (loop_options.is_gpu_block_index()) {
    // 获取 GPU 块索引
    int gpu_block_index = loop_options.gpu_block_index();
    // 检查 GPU 块索引是否超过了3维
    if (gpu_block_index >= 3) {
      throw std::runtime_error("support only 3D gpu_block_index");
    }
    // 获取旧的达到表达式
    old_reach = current_block_reach_[gpu_block_index];

    // 如果旧的达到是常数并且等于1，则将当前块的达到设置为循环的停止条件
    if (old_reach->isConstant() && immediateEquals(old_reach, 1)) {
      current_block_reach_[gpu_block_index] = v->stop();
    } else {
      // 否则，使用 IRSimplifier 简化达到表达式
      current_block_reach_[gpu_block_index] =
          IRSimplifier::simplify(alloc<Max>(old_reach, v->stop(), true));
    }

    // 获取 GPU 块变量
    VarPtr metaVar = gpu_block_vars_[gpu_block_index];
    // 用 GPU 块变量替换循环变量，更新循环体
    body = Substitute(Stmt::clone(body), {{v->var(), metaVar}});
  } else if (loop_options.is_gpu_thread_index()) {
    // 如果循环是 GPU 线程索引
    int gpu_thread_index = loop_options.gpu_thread_index();
    // 检查 GPU 线程索引是否超过了3维
    if (gpu_thread_index >= 3) {
      throw std::runtime_error("support only 3D gpu_thread_index");
    }
    // 获取旧的达到表达式
    old_reach = current_thread_reach_[gpu_thread_index];

    // 如果旧的达到是常数并且等于1，则将当前线程的达到设置为循环的停止条件
    if (old_reach->isConstant() && immediateEquals(old_reach, 1)) {
      current_thread_reach_[gpu_thread_index] = v->stop();
    } else {
      // 否则，使用 IRSimplifier 简化达到表达式
      current_thread_reach_[gpu_thread_index] =
          IRSimplifier::simplify(alloc<Max>(old_reach, v->stop(), true));
    }

    // 获取 GPU 线程变量
    VarPtr metaVar = gpu_thread_vars_[gpu_thread_index];
    // 用 GPU 线程变量替换循环变量，更新循环体
    body = Substitute(Stmt::clone(body), {{v->var(), metaVar}});
  }

  // 递归进入循环体块
  body = Stmt::clone(body->accept_mutator(this));

  // 弹出内部达到（reach）表达式的栈
  if (loop_options.is_gpu_block_index()) {
    // 如果是 GPU 块索引，恢复旧的块达到表达式
    current_block_reach_[loop_options.gpu_block_index()] = old_reach;
    return body;
  } else if (loop_options.is_gpu_thread_index()) {
    // 如果是 GPU 线程索引，恢复旧的线程达到表达式
    current_thread_reach_[loop_options.gpu_thread_index()] = old_reach;
    return body;
  }
    // 返回当前函数的结果 body
    return body;
  }

  // 使用新的 body 克隆给定对象 v，并返回结果
  return v->cloneWithNewBody(body);
}

StmtPtr GPUMetaVarRewriter::mutate(BlockPtr v) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<Segment> innerSegments;
  Segment current;

  auto pushAndReset = [&](bool mask) {
    // 如果当前段落不为空，则将其添加到内部段落列表中
    if (!current.empty()) {
      innerSegments.push_back(current);
    }
    // 重置当前段落，并设置是否需要掩码
    current.reset(mask);
  };

  // 这里我们将块的内容切分为具有相同启动范围的段落。段落由不是循环的所有语句组成，
  // 而循环语句本身则是一个段落。某些操作（如线程和内存操作）永远不应被掩码，因此也有自己的段落。
  for (StmtPtr stmt : *v) {
    // 使用变异器处理当前语句，如果变异后语句与原语句相同，则克隆一个新实例
    StmtPtr stmt_new = stmt->accept_mutator(this);
    if (stmt == stmt_new) {
      stmt_new = Stmt::clone(stmt_new);
    }

    // 同样，分配和释放操作不应被掩码
    if (to<Allocate>(stmt) || to<Free>(stmt)) {
      pushAndReset(false);
    }

    // 如果当前语句是循环，则是一个段落边界
    if (ForPtr f = to<For>(stmt)) {
      pushAndReset(false);
    }

    // 将变异后的语句添加到当前段落中
    current.stmts().push_back(stmt_new);
    // 如果当前段落不需要掩码，则在其远侧也是一个段落边界
    if (!current.mask()) {
      pushAndReset(true);
    }
  }

  // 如果当前段落不为空，则将其添加到内部段落列表中
  if (!current.empty()) {
    innerSegments.push_back(current);
  }

  // 如果在所有维度上都是最大范围，则在此级别不需要掩码
  if (isFullExtent()) {
    // 展开内部段落
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<StmtPtr> stmts;
    for (auto& v : innerSegments) {
      for (auto s : v.stmts()) {
        stmts.push_back(s);
      }
    }

    return alloc<Block>(stmts);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<StmtPtr> stmts;
  for (auto& segment : innerSegments) {
    bool need_sync = false;
    // 我们不会掩码循环，它们将掩码它们的内容。
    if (!segment.mask()) {
      // 断言段落语句的数量为1，并返回该语句
      TORCH_INTERNAL_ASSERT(segment.stmts().size() == 1, buildErrorMessage());
      stmts.push_back(segment.stmts()[0]);
      continue;
    }

    // 如果到达这里，由于不是完整的范围且直接子块不是循环，我们必须进行掩码。
    StmtPtr inner = alloc<Block>(segment.stmts());
    // 对块内线程进行处理。
    auto& thread_extents = cuda_analysis_->gpu_thread_extents();
    for (size_t i = 0; i < gpu_thread_vars_.size(); ++i) {
      if (!exprEquals(current_thread_reach_[i], thread_extents[i])) {
        need_sync = true;
        // 根据当前维度掩码内部块。
        inner = alloc<Cond>(
            alloc<CompareSelect>(
                gpu_thread_vars_[i],
                current_thread_reach_[i],
                CompareSelectOperation::kLT),
            inner,
            nullptr);
      }
    }
    auto& block_extents = cuda_analysis_->gpu_block_extents();
    // 遍历 GPU 块变量数组
    for (size_t i = 0; i < gpu_block_vars_.size(); ++i) {
      // 检查当前块的到达情况是否与块扩展相等
      if (!exprEquals(current_block_reach_[i], block_extents[i])) {
        // 对当前维度进行掩码处理
        inner = alloc<Cond>(
            // 分配条件节点，比较 GPU 块变量与当前块到达情况，判断是否小于
            alloc<CompareSelect>(
                gpu_block_vars_[i],
                current_block_reach_[i],
                CompareSelectOperation::kLT),
            // 将结果作为条件内部的子节点
            inner,
            nullptr);
      }
    }

    // 如果需要同步
    if (need_sync) {
      // 将同步线程的操作节点加入到语句列表中
      stmts.push_back(alloc<SyncThreads>());
    }
    // 将内部节点加入到语句列表中
    stmts.push_back(inner);
    // 如果需要同步
    if (need_sync) {
      // 将同步线程的操作节点再次加入到语句列表中
      stmts.push_back(alloc<SyncThreads>());
    }
  }

  // 返回一个块节点，包含所有语句
  return alloc<Block>(stmts);
}

static std::ostream& operator<<(
    std::ostream& out,
    const std::vector<ExprPtr>& exprs) {
  // 重载输出流操作符，用于打印表达式向量
  size_t i = 0;
  for (auto expr : exprs) {
    // 遍历表达式向量
    if (i++ > 0) {
      // 如果不是第一个表达式，输出逗号和空格
      out << ", ";
    }
    // 输出当前表达式的内容
    out << *expr;
  }
  return out;
}

static const char* device_resource_string = R"(
#define NAN __int_as_float(0x7fffffff)
#define POS_INFINITY __int_as_float(0x7f800000)
#define NEG_INFINITY __int_as_float(0xff800000)

)";

static const char* shared_resource_string = R"(
template<typename T>
__device__ T maximum(T a, T b) {
  return isnan(a) ? a : (a > b ? a : b);
}

template<typename T>
__device__ T minimum(T a, T b) {
  return isnan(a) ? a : (a < b ? a : b);
}

)";

void CudaCodeGen::Initialize() {
  // TODO: 处理多个核函数。
  // TODO: 处理动态维度。
  // TODO: 调用 nvrtc。
  // TODO: 将 HasRand 合并到 CudaAnalysis 中。
  GenericIntrinsicsExpander intrinsics_expander;
  apply_mutator(&intrinsics_expander);

  HasRand has_rand_func(stmt());
  has_random_ = has_rand_func.has_rand();
  cuda_analysis_ = std::make_unique<CudaAnalysis>();
  printer_ =
      std::make_unique<CudaPrinter>(&oss_, cuda_analysis_.get(), has_random_);
  metavar_rewriter_ =
      std::make_unique<GPUMetaVarRewriter>(cuda_analysis_.get());

  // 检查语句是否使用了 Half 类型，如果是则添加 half_support_literal。
  StmtPtr stmt_v = stmt();
  HalfChecker halfChecker(buffer_args());
  stmt_v->accept(&halfChecker);

  // 将设备资源字符串和共享资源字符串输出到流中
  os() << device_resource_string << shared_resource_string;

  if (has_random_) {
    // 如果使用了随机数生成，添加随机数字符串
    os() << philox_random_string << std::endl;
  }

  if (halfChecker.hasHalf()) {
    // 如果使用了 Half 类型，添加 half_support_literal
    os() << fuser::cuda::half_support_literal << std::endl;
  }
  if (halfChecker.hasBFloat16()) {
    // 如果使用了 BFloat16 类型，添加 bfloat16_support_literal
    os() << fuser::cuda::bfloat16_support_literal << std::endl;
  }

  std::string func_name = GetUniqueFuncName(kernel_func_name());
  os() << "extern \"C\" __global__" << std::endl;
#if defined(USE_ROCM)
  // ROCm 使用 256 作为默认的 flat work group size 上限
  os() << "__attribute__((amdgpu_flat_work_group_size(1, 1024)))" << std::endl;
#endif
  // 输出函数名称及其参数列表
  os() << "void " << func_name << "(";
  // 遍历缓冲区参数列表
  const std::vector<BufferArg> buffer_args = this->buffer_args();
  for (size_t i = 0; i < buffer_args.size(); i++) {
    if (i > 0) {
      // 如果不是第一个参数，输出逗号和空格
      os() << ", ";
    }
    // 输出当前缓冲区参数的变量和数据类型
    const BufferArg& buffer_arg = buffer_args[i];
    VarPtr var = buffer_arg.var();
    Dtype dtype = buffer_arg.dtype();
  os() << printer_->dtypeToCppString(dtype)
         << (buffer_arg.isVar() ? " " : "* ")
         << name_manager()->get_unique_name(var);
  }

// 输出类型转换的C++字符串表示，根据缓冲区参数类型决定是否输出"* "，并获取唯一的变量名。


  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  VarPtr rand_seed;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  VarPtr rand_offset;

// 声明两个变量指针rand_seed和rand_offset，忽略初始化变量的警告。


  if (has_random_) {
    // TODO: switch to kUint64 when it is available.
    rand_seed = alloc<Var>("rand_seed", kInt);
    rand_offset = alloc<Var>("rand_offset", kInt);
    std::string uint64_str = "unsigned long long";
    os() << ", " << uint64_str << " " << *rand_seed << ", " << uint64_str << " "
         << *rand_offset;
  }

// 如果有随机数需求，分配名为"rand_seed"和"rand_offset"的整型变量，并输出它们的C++声明语句。


  os() << ") {";
  os() << std::endl;

// 输出函数结尾的左大括号，换行。


  if (has_random_) {
    VarPtr idx = alloc<Var>("idx", kInt);
    os() << "int " << *idx << " = blockIdx.x*blockDim.x + threadIdx.x;"
         << std::endl;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    VarPtr rand_func = printer_->rand_func();
    os() << "Philox " << *rand_func << "(" << *rand_seed << ", " << *idx << ", "
         << *rand_offset << ");" << std::endl;
    os() << std::endl;
  }

// 如果有随机数需求，在CUDA环境下生成一个线程索引idx，并调用名为"rand_func"的随机数生成函数Philox。


  stmt_v->accept(cuda_analysis_.get());

// 使用CUDA分析器处理语句块stmt_v。


  stmt_v = stmt_v->accept_mutator(metavar_rewriter_.get());

// 使用变量重写器处理语句块stmt_v，并将结果重新分配给stmt_v。


  AtomicAddFuser atomic_add_fuser(
      cuda_analysis_->thread_local_bufs(), *metavar_rewriter_.get());
  stmt_v = stmt_v->accept_mutator(&atomic_add_fuser);

// 使用原子加法融合器处理CUDA分析器的线程局部缓冲区，并将结果重新分配给stmt_v。


  stmt_v = registerize(stmt_v);

// 使用寄存器分配器处理语句块stmt_v，并将结果重新分配给stmt_v。


  PrioritizeLoad prioritize_load;
  stmt_v = stmt_v->accept_mutator(&prioritize_load);

// 使用加载优先级处理器处理语句块stmt_v，并将结果重新分配给stmt_v。


  // The registerizer might insert half-type scalars, we don't want this.
  HalfRewriter hsFix;
  stmt_v = stmt_v->accept_mutator(&hsFix);

// 寄存器分配器可能会插入半类型标量，我们不希望这样。使用半类型重写器处理语句块stmt_v，并将结果重新分配给stmt_v。


  stmt_v = IRSimplifier::simplify(stmt_v);
  set_stmt(stmt_v);

// 使用IR简化器简化语句块stmt_v，并将结果设置为当前语句块。


  stmt_v->accept(printer_.get());
  os() << std::endl;
  os() << "}";

// 使用打印器打印语句块stmt_v，并输出右大括号，换行。


  // Check that all block extents had been set.
  const std::vector<ExprPtr>& gpu_block_extents =
      metavar_rewriter_->gpu_block_extents();
  for (size_t i = 0; i < gpu_block_extents.size(); i++) {
    if (!gpu_block_extents[i]) {
      throw std::runtime_error("Missing gpu_block_index: " + std::to_string(i));
    }
  }

// 检查所有块范围是否已设置。获取CUDA变量重写器的GPU块范围表达式向量，并逐一检查其是否为空，若为空则抛出运行时错误。


  // Precompute block and thread extents for call_with_numel().  If
  // precomputation can't be done (block/thread extents aren't
  // constant), then disallow call_with_numel.
  auto block_extents = metavar_rewriter_->gpu_block_extents();
  auto thread_extents = metavar_rewriter_->gpu_thread_extents();
  bool canCallWithNumel =
      !has_random_ && block_extents.size() > 0 && thread_extents.size() > 0;
  for (size_t i = 1; i < block_extents.size() && canCallWithNumel; i++) {
    canCallWithNumel = canCallWithNumel && block_extents[i]->isConstant() &&
        immediateAs<int>(block_extents[i]) == 1;
  }
  for (size_t i = 1; i < thread_extents.size() && canCallWithNumel; i++) {
    canCallWithNumel = canCallWithNumel && thread_extents[i]->isConstant() &&
        immediateAs<int>(thread_extents[i]) == 1;
  }
  if (canCallWithNumel && thread_extents[0]->isConstant()) {

// 为call_with_numel()预计算块和线程范围。如果无法预计算（块/线程范围不是常量），则禁止调用call_with_numel。获取CUDA变量重写器的GPU块和线程范围表达式向量，检查它们是否都是常量，如果可以进行调用，并且线程范围的第一个元素是常量。
    // 假设 block_extents[0] 的值等于 output.numel() / thread_block_size_
    thread_block_size_ = immediateAs<int>(thread_extents[0]);
  } else {
    // 禁用 call_with_numel
    thread_block_size_ = -1;
  }

  // 为块的维度构建基于 LLVM 的求值表达式
  block_extents_eval_.reserve(block_extents.size());
  std::vector<BufferArg> extents_buffer_args;

  // 从 bufferArgs 中提取在线程和块维度中使用的参数，并将其用于下面的 `ExprEval`。
  // 如果不进行这一步，bufferArgs 可能包含不被 LLVM 处理的任意类型，可能导致错误。
  std::unordered_set<VarPtr> vars_in_extents;
  for (const auto& be : block_extents) {
    auto v = VarFinder::find(be);
    vars_in_extents.insert(v.begin(), v.end());
  }
  for (const auto& te : thread_extents) {
    auto v = VarFinder::find(te);
    vars_in_extents.insert(v.begin(), v.end());
  }
  for (const size_t i : c10::irange(buffer_args.size())) {
    if (vars_in_extents.count(buffer_args[i].var())) {
      extents_buffer_args.push_back(buffer_args[i]);
      arg_pos_in_extents_.push_back(true);
    } else {
      arg_pos_in_extents_.push_back(false);
    }
  }
  for (const auto& be : block_extents) {
#ifdef TORCH_ENABLE_LLVM
    // 如果编译时开启了LLVM支持，使用LLVM代码生成器对表达式进行求值，并将结果存入block_extents_eval_
    block_extents_eval_.emplace_back(
        ExprEval<LLVMCodeGen>(ExprHandle(be), extents_buffer_args));
#else
    // 如果编译时未开启LLVM支持，使用简单的IR求值器对表达式进行求值，并将结果存入block_extents_eval_
    block_extents_eval_.emplace_back(
        ExprEval<SimpleIREvaluator>(ExprHandle(be), extents_buffer_args));
#endif
  }

  // 预留空间以容纳thread_extents的大小
  thread_extents_eval_.reserve(thread_extents.size());
  // 遍历thread_extents中的每个表达式
  for (const auto& te : thread_extents) {
#ifdef TORCH_ENABLE_LLVM
    // 如果编译时开启了LLVM支持，使用LLVM代码生成器对表达式进行求值，并将结果存入thread_extents_eval_
    thread_extents_eval_.emplace_back(
        ExprEval<LLVMCodeGen>(ExprHandle(te), extents_buffer_args));
#else
    // 如果编译时未开启LLVM支持，使用简单的IR求值器对表达式进行求值，并将结果存入thread_extents_eval_
    thread_extents_eval_.emplace_back(
        ExprEval<SimpleIREvaluator>(ExprHandle(te), extents_buffer_args));
#endif
  }

  // 打印CUDA内核相关的调试信息，包括内核代码、gpu_block_extents和gpu_thread_extents
  GRAPH_DEBUG(
      "Fused TE CUDA kernel:\n",
      oss_.str(),
      "\n",
      "gpu_block_extents: (",
      metavar_rewriter_->gpu_block_extents(),
      ")\n",
      "gpu_thread_extents: (",
      metavar_rewriter_->gpu_thread_extents(),
      ")");

  // 编译内核代码字符串oss_.str()到NVRTC，并使用func_name作为内核函数名
  CompileToNVRTC(oss_.str(), func_name);
}

void CudaCodeGen::call_with_numel(void** args, int64_t numel) {
  if (C10_UNLIKELY(numel == 0)) {
    return;
  }
  if (C10_UNLIKELY(thread_block_size_ <= 0)) {
    // 如果线程块大小小于等于0，则抛出异常
    TORCH_INTERNAL_ASSERT(
        thread_block_size_ >= 0,
        "call_with_numel() requires a precomputed thread block size");
  }

  auto const& buffer_args = this->buffer_args();
  // 计算gpu_block_extents和gpu_thread_extents
  size_t gpu_block_extents =
      (numel + thread_block_size_ - 1) / thread_block_size_;
  size_t gpu_thread_extents = thread_block_size_;

  // 在CUDA中，需要为缓冲区传递指针的指针，因此需要处理非标量参数
  // 参考：https://stackoverflow.com/questions/34388712/cannot-understand-how-jcuda-culaunchkernel-work
  std::vector<void*> ptr_to_args(buffer_args.size());
  for (size_t i = 0; i < buffer_args.size(); i++) {
    ptr_to_args[i] =
        // NOLINTNEXTLINE: const_cast
        buffer_args[i].isVar() ? args[i] : const_cast<void**>(&args[i]);
  }

  // 设置当前CUDA设备为this对象所表示的设备
  const auto device = this->device().index();
  const auto prior_device = at::cuda::current_device();
  if (prior_device != device) {
    at::cuda::set_device(device);
  }

  // 获取当前CUDA流，并初始化CUDA上下文
  auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::jit::initializeCudaContext();
  // 使用NVRTC编译器启动CUDA内核函数function_
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      function_,
      gpu_block_extents,
      1,
      1,
      gpu_thread_extents,
      1,
      1,
      0,
      stream,
      ptr_to_args.data(),
      nullptr));

  // 恢复之前的CUDA设备状态
  if (prior_device != device) {
    at::cuda::set_device(prior_device);
  }
}

void CudaCodeGen::call_raw(const std::vector<void*>& raw_args) {
  auto const& buffer_args = this->buffer_args();

  // TODO: move as much of this into the constructors.
  // 获取gpu_block_extents和gpu_thread_extents的表达式列表
  const std::vector<ExprPtr>& gpu_block_extents =
      metavar_rewriter_->gpu_block_extents();
  const std::vector<ExprPtr>& gpu_thread_extents =
      metavar_rewriter_->gpu_thread_extents();
  if (gpu_block_extents.size() > 3 || gpu_thread_extents.size() > 3) {
    // 如果块或线程范围大于3D，则抛出异常
    throw malformed_input(
        "cuda_codegen: block or thread extent greater than 3D");
  }

  // 创建一个包含3个元素且每个元素都初始化为1的整数向量，用于表示GPU块的范围
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int> gpu_block_extents_v(3, 1);
  // 创建一个包含3个元素且每个元素都初始化为1的整数向量，用于表示GPU线程的范围
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int> gpu_thread_extents_v(3, 1);

  // 计算所有块和线程范围的值
  // TODO: 最终，对这些计算进行代码生成，并将其作为模块的一部分。
  std::vector<void*> extent_args;
  size_t raw_args_size = raw_args.size();
  extent_args.reserve(raw_args_size);
  for (size_t i = 0; i < raw_args_size; ++i) {
    // 如果参数在范围内，则将其添加到extent_args中
    if (arg_pos_in_extents_[i]) {
      extent_args.push_back(raw_args[i]);
    }
  }
  // 对GPU块的每个维度进行评估
  for (size_t i = 0; i < gpu_block_extents.size(); i++) {
    // 如果GPU块的某个维度是常数，则将其立即转换为int64_t类型
    if (gpu_block_extents[i]->isConstant()) {
      gpu_block_extents_v[i] = immediateAs<int64_t>(gpu_block_extents[i]);
      continue;
    }
    {
      // block_extents_eval_的调用不是线程安全的，因此使用互斥锁进行保护
      std::lock_guard<std::mutex> guard(eval_lock_);
      gpu_block_extents_v[i] =
          block_extents_eval_[i].value<int64_t>(extent_args);
    }
  }
  // 对GPU线程的每个维度进行评估
  for (size_t i = 0; i < gpu_thread_extents.size(); i++) {
    // 如果GPU线程的某个维度是常数，则将其立即转换为int64_t类型
    if (gpu_thread_extents[i]->isConstant()) {
      gpu_thread_extents_v[i] = immediateAs<int64_t>(gpu_thread_extents[i]);
      continue;
    }
    {
      std::lock_guard<std::mutex> guard(eval_lock_);
      gpu_thread_extents_v[i] =
          thread_extents_eval_[i].value<int64_t>(extent_args);
    }
  }

  // 如果GPU块的任何维度为0，则跳过启动内核的步骤
  for (int extent : gpu_block_extents_v) {
    if (extent == 0) {
      return;
    }
  }

  // 计算指针参数的数量
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int ptr_count = buffer_args.size();
  // 如果内核中包含随机调用，则为随机种子和偏移添加两个额外的参数
  if (has_random_) {
    ptr_count += 2;
  }
  // 创建一个指针向量，用于存储指向参数的指针
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<void*> ptr_to_args(ptr_count);

  // 在CUDA中，对于非标量参数，需要传递指针的指针，因此需要处理buffer_args
  for (size_t i = 0; i < buffer_args.size(); i++) {
    // 如果buffer_args是变量，则直接将raw_args[i]作为指针传递
    // 否则，需要将raw_args[i]的指针进行类型转换后传递
    ptr_to_args[i] =
        buffer_args[i].isVar() ? raw_args[i] : const_cast<void**>(&raw_args[i]);
  }

  // 如果存在随机数生成，则初始化随机种子和偏移
  if (has_random_) {
    uint64_t rand_seed = uint64_t(-1);
    uint64_t rand_offset = uint64_t(-1);
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    // TODO: 这是一个临时解决方案。当numel可用时，切换为numel。
    int64_t total_elements_per_thread = (1LL << 28);
    {
      // 使用 std::lock_guard 对 gen.mutex() 进行加锁，确保线程安全
      std::lock_guard<std::mutex> lock(gen.mutex());
      // 检查 gen 是否为 at::CUDAGeneratorImpl 类型，获取 Philox 引擎的输入参数
      auto philox_engine_inputs =
          at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
              total_elements_per_thread);
      // 获取随机种子和偏移量
      rand_seed = philox_engine_inputs.first;
      rand_offset = philox_engine_inputs.second;
    }
    
    // 将 rand_seed 和 rand_offset 存储到 ptr_to_args 数组中
    ptr_to_args[buffer_args.size()] = &rand_seed;
    ptr_to_args[buffer_args.size() + 1] = &rand_offset;
    
    // 获取当前 CUDA 设备，并确保与 this 对象所指定的设备一致
    auto prior_device = at::cuda::current_device();
    if (prior_device != this->device().index()) {
      at::cuda::set_device(this->device().index());
    }
    
    // 启动 CUDA 核函数
    // 获取当前 CUDA 流
    auto stream = at::cuda::getCurrentCUDAStream();
    // 初始化 CUDA 上下文
    at::cuda::jit::initializeCudaContext();
    // 调用 CUDA 驱动程序接口启动核函数
    AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
        function_,
        gpu_block_extents_v[0],
        gpu_block_extents_v[1],
        gpu_block_extents_v[2],
        gpu_thread_extents_v[0],
        gpu_thread_extents_v[1],
        gpu_thread_extents_v[2],
        0,
        stream,
        ptr_to_args.data(),
        nullptr));
    
    // 如果之前的 CUDA 设备不是 this 对象所指定的设备，则恢复之前的设备
    if (prior_device != this->device().index()) {
      at::cuda::set_device(prior_device);
    }
// 结束 CudaCodeGen 类的定义

void CudaCodeGen::call(const std::vector<CallArg>& args) {
  // 检查传入的参数个数是否与预期的缓冲区参数个数相同，如果不同则抛出异常
  if (args.size() != buffer_args().size()) {
    throw malformed_input("cuda_codegen: wrong number of args in call");
  }

  // 获取当前对象的缓冲区参数列表的引用
  auto const& buffer_args = this->buffer_args();
  // 创建一个空的指针数组，大小与缓冲区参数列表相同
  std::vector<void*> raw_args(buffer_args.size());
  // 遍历缓冲区参数列表
  for (size_t i = 0; i < buffer_args.size(); i++) {
    // 获取当前缓冲区参数和对应的调用参数
    auto const& bufferArg = buffer_args[i];
    auto const& callArg = args[i];
    // 将调用参数转换为指针，并存储在 raw_args 数组中
    raw_args[i] = argToPtr(bufferArg, callArg);
  }
  // 调用 call_raw 方法，传入 raw_args 数组
  call_raw(raw_args);
}

// 创建一个新的空张量，使用指定的大小、步幅、数据类型等参数
at::Tensor CudaCodeGen::empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  // 在代码块执行期间，设置当前设备为 device_opt 指定的设备
  c10::DeviceGuard device_guard(device_opt.value());
  // 调用 ATen 库中的 CUDA 特定函数创建一个空的张量
  return at::native::empty_strided_cuda(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// 编译给定的 CUDA 代码到 NVRTC 中
void CudaCodeGen::CompileToNVRTC(
    const std::string& code,
    const std::string& func_name) {
  // 初始化 CUDA 上下文，确保 CUDA 环境处于正确状态
  at::cuda::jit::initializeCudaContext();
  // 记录当前设备的 ID，因为某些情况下 at::DeviceGuard 可能无法正常工作
  auto prior_device = at::cuda::current_device();
  // 如果当前设备 ID 不等于对象的设备 ID，则切换设备为对象指定的设备
  if (prior_device != this->device().index()) {
    at::cuda::set_device(this->device().index());
  }
  // 获取当前设备的 CUDA 属性
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // 初始化 major 和 minor 变量，并设置 compile_to_sass 为 false
  int major, minor;
  bool compile_to_sass = false;
  // 调用 fuser::cuda::codegenOutputQuery 函数获取 CUDA 属性的主版本、次版本和编译到 sass 的状态
  fuser::cuda::codegenOutputQuery(prop, major, minor, compile_to_sass);

  // 创建一个新的 NVRTC 程序对象，用于编译传入的 CUDA 代码字符串
  nvrtcProgram program;
  // 调用 NVRTC 库的 nvrtcCreateProgram 函数创建程序对象
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));

#if defined(USE_ROCM)
  // 如果使用 ROCm 平台，则设置编译参数为 {"--std=c++17", "-hip-pch"}
  std::vector<const char*> args = {"--std=c++17"};
  args.push_back("-hip-pch");
#else
  // 根据 CUDA 版本选择适当的编译选项
  const std::string compute = std::string("--gpu-architecture=") +
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11010
      // CUDA 11.1 允许直接使用 SASS (sm_) 替代 PTX (compute_)，提供更好的向后兼容性
      // 对于未来设备，如果 compile_to_sass 为 false，则回退到使用 PTX
      (compile_to_sass ? "sm_" : "compute_") +
#else
      "compute_" +
#endif
      std::to_string(major) + std::to_string(minor);
  // 设置编译参数数组，包括 "--std=c++17"、compute 变量和 "-default-device"
  const std::vector<const char*> args = {
      "--std=c++17", compute.c_str(), "-default-device"};
#endif

// 使用 nvrtc 编译给定的程序
auto result = nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
if (result != NVRTC_SUCCESS) {
    // 获取编译日志的大小
    size_t logsize;
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetProgramLogSize(program, &logsize));
    // 分配足够大小的缓冲区来存储编译日志
    std::vector<char> log(logsize);
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetProgramLog(program, log.data()));
    // 将编译日志转换为字符串流
    std::stringstream cu;
    cu << log.data() << std::endl;
    cu << "nvrtc compilation failed: " << std::endl;
    cu << code << std::endl;
    // 抛出运行时错误，包含编译日志和源代码
    throw std::runtime_error(cu.str());
}

// 注册资源守护，确保程序对象被销毁
ResourceGuard holdProgram(
    [&] { AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });
AT_CUDA_NVRTC_CHECK(result);

// 获取 PTX 大小
size_t ptx_size;
std::vector<char> ptx;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11010
// 根据编译目标选择获取大小的函数
auto getSize = compile_to_sass
    ? at::globalContext().getNVRTC().nvrtcGetCUBINSize
    : at::globalContext().getNVRTC().nvrtcGetPTXSize;
auto getFunc = compile_to_sass ? at::globalContext().getNVRTC().nvrtcGetCUBIN
                               : at::globalContext().getNVRTC().nvrtcGetPTX;
#else
auto getSize = at::globalContext().getNVRTC().nvrtcGetPTXSize;
auto getFunc = at::globalContext().getNVRTC().nvrtcGetPTX;
#endif

// 获取 PTX 数据
AT_CUDA_NVRTC_CHECK(getSize(program, &ptx_size));
ptx.resize(ptx_size);
AT_CUDA_NVRTC_CHECK(getFunc(program, ptx.data()));

// 加载 PTX 数据为 CUDA 模块
CUmodule module;
AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&module, ptx.data()));
AT_CUDA_DRIVER_CHECK(
    nvrtc().cuModuleGetFunction(&function_, module, func_name.c_str()));

// 如果之前的设备与当前设备不同，设置回之前的设备
if (prior_device != this->device().index()) {
    at::cuda::set_device(prior_device);
}
}

// CUDA 代码生成器的析构函数，默认
CudaCodeGen::~CudaCodeGen() = default;

// 注册 CUDA 代码生成器
RegisterCodeGen<CudaCodeGen> cuda_codegen_reg("cuda_codegen");

} // namespace torch::jit::tensorexpr
```