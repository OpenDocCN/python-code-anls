# `.\pytorch\torch\csrc\jit\runtime\static\te_wrapper.cpp`

```
// 包含 Torch 的静态张量表达式相关头文件
#include <torch/csrc/jit/runtime/static/te_wrapper.h>

// 包含 Torch 的 CPU 函数实现头文件
#include <ATen/CPUFunctions.h>
// 包含 Torch 的 IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT 日志头文件
#include <torch/csrc/jit/jit_log.h>
// 包含 Torch 的静态运行时实现头文件
#include <torch/csrc/jit/runtime/static/impl.h>
// 包含 Torch 的张量表达式相关头文件
#include <torch/csrc/jit/tensorexpr/expr.h>
// 包含 Torch 的张量表达式操作相关头文件
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

// Torch JIT 命名空间
namespace torch::jit {

// 使用 Torch 张量表达式命名空间
using namespace torch::jit::tensorexpr;

// 默认使用 AVX-512 向量的宽度；这对于 AVX2 也能正常工作。一些操作受益于多个 AVX 端口的使用，因此它们被向量化为两倍于这个常量的宽度。
// 一个例外是 logit，因为它包含 FP 除法，这是单端口的。
static constexpr int kVectorWidth = 16;

#ifdef TORCH_ENABLE_LLVM

// TEWrapper 类的方法，用于更新 LLVMCodeGen 对象
void TEWrapper::update(std::unique_ptr<LLVMCodeGen>&& cg_) {
  cg = std::move(cg_);
}

// TEWrapper 类的方法，调用 LLVMCodeGen 对象的原始调用方法
void TEWrapper::call(const std::vector<void*>& args) {
  cg->call_raw(args);
}

// 优化点对点操作的静态函数，将循环嵌套分裂和向量化
static void optimizePointwise(LoopNest* ln, Tensor target, int width) {
  // 获取目标张量的循环语句列表
  std::vector<ForPtr> loops = ln->getLoopStmtsFor(target);
  ForPtr inner, tail;
  // 检查是否为点对点操作创建了循环
  TORCH_CHECK(loops.size() > 0, "No loops created for pointwise op");
  // 使用给定宽度分裂循环并留下尾部
  ln->splitWithTail(loops[0], width, &inner, &tail);
  // 对内部循环进行向量化
  ln->vectorize(inner);
}

// 包装 TE 计算的函数，返回更新后的 TEWrapper 对象
static std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    Tensor out,
    std::vector<CodeGen::BufferArg> args,
    int width = kVectorWidth) {
  // 创建循环嵌套对象，并优化点对点操作
  LoopNest ln({out});
  optimizePointwise(&ln, out, width);
  ln.prepareForCodegen();
  // 简化生成的 IR 语句
  StmtPtr s = ln.root_stmt();
  s = IRSimplifier::simplify(s);
  // 将输出张量作为第一个参数插入参数列表
  args.insert(args.begin(), out);
  // 创建新的 LLVMCodeGen 对象并更新 TEWrapper 对象
  auto cg = std::make_unique<LLVMCodeGen>(s, args);
  cg->cleanup_memory();
  wrap->update(std::move(cg));
  return wrap;
}

// 包装 TE 计算的函数，返回更新后的 TEWrapper 对象（重载版本）
static std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    LoopNest* ln,
    std::vector<CodeGen::BufferArg> args) {
  // 创建新的 LLVMCodeGen 对象并更新 TEWrapper 对象
  auto cg = std::make_unique<LLVMCodeGen>(ln->root_stmt(), args);
  wrap->update(std::move(cg));
  return wrap;
}

#else

// 当未启用 LLVM 时，TEWrapper 类的方法，断言失败无效调用
void TEWrapper::call(const std::vector<void*>& args) {
  DCHECK(0 && "Invalid call");
}

// 当未启用 LLVM 时，返回未更新的 TEWrapper 对象的包装 TE 计算函数
static std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    Tensor out,
    std::vector<CodeGen::BufferArg> args,
    int width = kVectorWidth) {
  return wrap;
}

// 当未启用 LLVM 时，返回未更新的 TEWrapper 对象的包装 TE 计算函数（重载版本）
static std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    LoopNest* ln,
    std::vector<CodeGen::BufferArg> args) {
  return wrap;
}

#endif

// 匿名命名空间，定义用于并发访问 NN 缓存的互斥量
namespace {

std::mutex& getNNCCacheMutex() {
  static std::mutex nncCacheMutex;
  return nncCacheMutex;
}

// 获取 NN 缓存的全局静态映射
c10::FastMap<NodeKind, std::shared_ptr<TEWrapper>>& getNNCCache() {
  static c10::FastMap<NodeKind, std::shared_ptr<TEWrapper>> nncCache;
  return nncCache;
}

// 查询 NN 缓存中特定 NodeKind 的 TEWrapper 对象
std::shared_ptr<TEWrapper> lookupNNCCache(NodeKind kind) {
  std::lock_guard<std::mutex> lock(getNNCCacheMutex());
  auto it = getNNCCache().find(kind);
  if (it != getNNCCache().end()) {
    return it->second;
  }
  return nullptr;
}

} // anonymous namespace

} // namespace torch::jit
void updateNNCCache(NodeKind kind, std::shared_ptr<TEWrapper> code) {
  // 使用互斥锁保护对 NN 缓存的更新操作，确保线程安全
  std::lock_guard<std::mutex> lock(getNNCCacheMutex());
  // 将给定种类的节点代码更新到 NN 缓存中
  getNNCCache()[kind] = code;
}

} // namespace

// 创建并返回一个表示除法操作的 TEWrapper 对象
std::shared_ptr<TEWrapper> createDiv() {
  auto wrap = lookupNNCCache(aten::div);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();

  auto dim = VarHandle("dim", kInt);
  auto mode = VarHandle("mode", kInt);
  BufHandle A("A", {dim}, kFloat);
  BufHandle B("B", {dim}, kFloat);

  using axis = const VarHandle&;
  // 创建张量 C，表示针对每个维度 dim 的除法操作
  Tensor C = Compute("C", {dim}, [&](axis x) {
    auto true_div_result = A.load(x) / B.load(x);

    auto mode_default = IntImm::make(0);
    auto mode_trunc = IntImm::make(1);
    auto mode_floor = IntImm::make(2);

    // 根据 mode 参数选择不同的处理方式，返回处理后的结果
    // 用于处理除法结果的条件选择逻辑
    return CompareSelect::make(
        mode,
        mode_default,
        true_div_result,
        CompareSelect::make(
            mode,
            mode_trunc,
            trunc(true_div_result),
            floor(true_div_result),
            kEQ),
        kEQ);
  });

  // 将创建的 TEWrapper 对象和张量 C 封装，并更新到 NN 缓存中
  wrap = wrapTECompute(wrap, C, {A, B, mode, dim});

  updateNNCCache(aten::div, wrap);
  return wrap;
}

// 创建并返回一个表示逻辑函数操作的 TEWrapper 对象
std::shared_ptr<TEWrapper> createLogit() {
  auto wrap = lookupNNCCache(aten::logit);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  auto C = VarHandle("C", kFloat);
  BufHandle A("A", {N}, kFloat);
  // 创建张量 B，表示逻辑函数操作
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto A_elem = [&]() {
      auto elem = A.load(i);
      auto one = FloatImm::make(1.0f);
      const auto& min = C;
      auto max = one - C;
      // 使用条件选择逻辑处理 A_elem
      elem = CompareSelect::make(elem, min, min, elem, kLT);
      return CompareSelect::make(elem, max, max, elem, kGT);
    }();
    // 返回经过对数函数处理后的 A_elem
    return log_vml(A_elem / (FloatImm::make(1.0f) - A_elem));
  });
  // 将创建的 TEWrapper 对象和张量 B 封装，并更新到 NN 缓存中
  wrap = wrapTECompute(wrap, B, {A, N, C});
  updateNNCCache(aten::logit, wrap);
  return wrap;
}

// 创建并返回一个表示 ReLU 激活函数操作的 TEWrapper 对象
std::shared_ptr<TEWrapper> createRelu() {
  auto wrap = lookupNNCCache(aten::relu);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  // 创建张量 B，表示 ReLU 激活函数操作
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto zero = FloatImm::make(0.f);
    auto a = A.load(i);
    // 使用条件选择逻辑处理 ReLU 操作
    return CompareSelect::make(a, zero, zero, a, kLT);
  });
  // 将创建的 TEWrapper 对象和张量 B 封装，并更新到 NN 缓存中
  wrap = wrapTECompute(wrap, B, {A, N});
  updateNNCCache(aten::relu, wrap);
  return wrap;
}

// 创建并返回一个表示双曲正切函数操作的 TEWrapper 对象
std::shared_ptr<TEWrapper> createTanh() {
  auto wrap = lookupNNCCache(aten::tanh);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  // 创建张量 B，表示双曲正切函数操作
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto a = A.load(i);
    // 使用快速双曲正切函数处理操作
    return fast_tanh(a);
  });
  // 将创建的 TEWrapper 对象和张量 B 封装，并更新到 NN 缓存中
  wrap = wrapTECompute(wrap, B, {A, N});
  updateNNCCache(aten::tanh, wrap);
  return wrap;
}

// 创建并返回一个表示 sigmoid 函数操作的 TEWrapper 对象
std::shared_ptr<TEWrapper> createSigmoid() {
  auto wrap = lookupNNCCache(aten::sigmoid);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  // 创建张量 B，表示 sigmoid 函数操作
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto a = A.load(i);
    // 返回经过 sigmoid 函数处理后的结果
    return sigmoid(a);
  });
  // 将创建的 TEWrapper 对象和张量 B 封装，并更新到 NN 缓存中
  wrap = wrapTECompute(wrap, B, {A, N});
  updateNNCCache(aten::sigmoid, wrap);
  return wrap;
}
    return wrap;
  }
  // 创建一个名为wrap的std::shared_ptr指针，指向TEWrapper类型的对象
  wrap = std::make_shared<TEWrapper>();
  // 创建一个名为N的VarHandle对象，表示一个整数变量
  auto N = VarHandle("N", kInt);
  // 创建一个名为A的BufHandle对象，表示一个一维浮点数数组，大小为N
  BufHandle A("A", {N}, kFloat);
  // 创建一个名为B的Tensor对象，表示一个一维数组，大小为N，使用A中的每个元素通过fast_sigmoid函数计算得到
  Tensor B = Compute(
      "B", {N}, [&](const VarHandle& i) { return fast_sigmoid(A.load(i)); });
  // 调用wrapTECompute函数，将wrap、B、{A, N}作为参数，返回更新后的wrap
  wrap = wrapTECompute(wrap, B, {A, N});
  // 调用updateNNCCache函数，将aten::sigmoid和wrap作为参数，更新神经网络计算缓存
  updateNNCCache(aten::sigmoid, wrap);
  // 返回wrap指针，即TEWrapper对象的共享指针
  return wrap;
}

// 创建并返回一个 TEWrapper 的 shared_ptr，用于表示 clamp 操作
std::shared_ptr<TEWrapper> createClamp() {
  // 静态变量，代表 aten::clamp 操作的符号
  static auto clamp_symbol = c10::Symbol::fromQualString("aten::clamp");
  // 查找符号对应的 TEWrapper，如果已存在则返回
  auto wrap = lookupNNCCache(clamp_symbol);
  if (wrap) {
    return wrap;
  }
  // 如果 TEWrapper 不存在，则创建一个新的 TEWrapper 对象
  wrap = std::make_shared<TEWrapper>();
  // 定义变量句柄 N、min_handle 和 max_handle
  auto N = VarHandle("N", kInt);
  auto min_handle = VarHandle("min", kFloat);
  auto max_handle = VarHandle("max", kFloat);

  // 定义缓冲区句柄 A，表示一个 float 类型的数组
  BufHandle A("A", {N}, kFloat);
  
  // 创建张量 result，代表 aten_clamp 运算的结果
  Tensor result = Compute("aten_clamp", {N}, [&](const VarHandle& i) {
    auto a = A.load(i); // 加载 A 中的数据
    return tensorexpr::clamp(min_handle, max_handle, a); // 执行 clamp 操作
  });

  // 包装 TEWrapper，将 result 与相关的句柄传递给 wrapTECompute 函数
  wrap = wrapTECompute(wrap, result, {A, min_handle, max_handle, N});
  
  // 更新符号对应的 TEWrapper 缓存
  updateNNCCache(clamp_symbol, wrap);
  
  // 返回创建的 TEWrapper 对象
  return wrap;
}

// 创建并返回一个 TEWrapper 的 shared_ptr，用于表示 clamp_nan_to_num 操作
std::shared_ptr<TEWrapper> createClampNanToNum() {
  // 静态变量，代表 static_runtime::clamp_nan_to_num 操作的符号
  static auto symbol = c10::Symbol::fromQualString("static_runtime::clamp_nan_to_num");
  // 查找符号对应的 TEWrapper，如果已存在则返回
  auto wrap = lookupNNCCache(symbol);
  if (wrap) {
    return wrap;
  }
  // 如果 TEWrapper 不存在，则创建一个新的 TEWrapper 对象
  wrap = std::make_shared<TEWrapper>();
  // 定义变量句柄 N、min_handle、max_handle 和 nan_replace_val
  auto N = VarHandle("N", kInt);
  auto min_handle = VarHandle("min", kFloat);
  auto max_handle = VarHandle("max", kFloat);
  auto nan_replace_val = VarHandle("nan_replace_val", kFloat);

  // 定义缓冲区句柄 A，表示一个 float 类型的数组
  BufHandle A("A", {N}, kFloat);
  
  // 创建张量 result，代表 aten_clamp 运算的结果
  Tensor result = Compute("aten_clamp", {N}, [&](const VarHandle& i) {
    auto a = A.load(i); // 加载 A 中的数据
    auto clamp = tensorexpr::clamp(min_handle, max_handle, a); // 执行 clamp 操作
    auto is_nan = tensorexpr::isnan(clamp); // 检测 clamp 中的 NaN 值
    auto nans_replaced = tensorexpr::CompareSelect::make(is_nan, 1, nan_replace_val, clamp, kEQ); // 替换 NaN 值
    return nans_replaced; // 返回替换后的张量
  });

  // 包装 TEWrapper，将 result 与相关的句柄传递给 wrapTECompute 函数
  wrap = wrapTECompute(wrap, result, {A, min_handle, max_handle, nan_replace_val, N});
  
  // 更新符号对应的 TEWrapper 缓存
  updateNNCCache(symbol, wrap);
  
  // 返回创建的 TEWrapper 对象
  return wrap;
}

// 创建并返回一个 TEWrapper 的 shared_ptr，用于表示 signed_log1p 操作
std::shared_ptr<TEWrapper> createSignedLog1p() {
  // 静态变量，代表 static_runtime::signed_log1p 操作的符号
  static auto signed_log1p_symbol = c10::Symbol::fromQualString("static_runtime::signed_log1p");
  // 查找符号对应的 TEWrapper，如果已存在则返回
  auto wrap = lookupNNCCache(signed_log1p_symbol);
  if (wrap) {
    return wrap;
  }
  // 如果 TEWrapper 不存在，则创建一个新的 TEWrapper 对象
  wrap = std::make_shared<TEWrapper>();
  // 定义变量句柄 N
  auto N = VarHandle("N", kInt);
  
  // 定义缓冲区句柄 A，表示一个 float 类型的数组
  BufHandle A("A", {N}, kFloat);
  
  // 创建张量 abs_result，代表 aten_abs 运算的结果
  Tensor abs_result = Compute("aten_abs", {N}, [&](const VarHandle& i) {
    return tensorexpr::abs(A.load(i)); // 计算 A 中每个元素的绝对值
  });
  
  // 创建张量 log1p_result，代表 aten_log1p 运算的结果
  Tensor log1p_result = Compute("aten_log1p", {N}, [&](const VarHandle& i) {
    return log1p(abs_result.load(i)); // 对 abs_result 中每个元素执行 log1p 函数
  });
  
  // 创建张量 sign，表示计算 A 中每个元素的符号
  Tensor sign = computeSign({A}, {N});
  
  // 创建张量 output，代表 aten_mul 运算的结果
  Tensor output = Compute("aten_mul", {N}, [&](const VarHandle& i) {
    return sign.load(i) * log1p_result.load(i); // 计算 sign 与 log1p_result 的乘积
  });
  
  // 创建 LoopNest 对象 ln，用于表示一组张量计算操作
  LoopNest ln({output}, {abs_result, log1p_result, sign, output});
  
  // 在 ln 中内联中间缓冲区
  ln.inlineIntermediateBufs(true);
  
  // 为代码生成做准备
  ln.prepareForCodegen();
  
  // 简化 ln 中的计算
  ln.simplify();
  
  // 对内部循环进行矢量化优化
  ln.vectorizeInnerLoops();
  
  // 简化 ln 中的计算
  ln.simplify();
  
  // 打印调试信息，输出最终的计算语句
  GRAPH_DEBUG("Final stmt: ", *ln.root_stmt());
  
  // 包装 TEWrapper，将 ln 与相关的张量传递给 wrapTECompute 函数
  wrap = wrapTECompute(wrap, &ln, {output, A, N});
  
  // 更新符号对应的 TEWrapper 缓存
  updateNNCCache(signed_log1p_symbol, wrap);
  
  // 返回创建的 TEWrapper 对象
  return wrap;
}

// torch::jit 命名空间的结束
} // namespace torch::jit
```