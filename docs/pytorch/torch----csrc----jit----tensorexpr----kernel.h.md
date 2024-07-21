# `.\pytorch\torch\csrc\jit\tensorexpr\kernel.h`

```py
#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// 哈希函数对象，用于对存储大小为一对的对象进行哈希计算
struct SmallSizeTPairHash {
 public:
  std::size_t operator()(const std::pair<size_t, size_t>& x) const {
    // 计算哈希值，将第一个元素乘以 128 再加上第二个元素
    return x.first * 128 + x.second;
  }
};

// 返回 true 如果 TE 融合器支持这个 Conv2d 节点
bool conv2dIsSupportedJit(const Node* node);
// 返回 true 如果 TE 融合器支持带有 MKLDNN 预打包卷积的 Conv2d
bool mkldnnPrepackedConvIsSupportedJit(const Node* node);
// 返回 true 如果 TE _convolution 节点是 Conv2d
bool isConv2d(const Node* node);
// 返回 true 如果 TE 融合器支持这个 Matmul 节点
bool matmulIsSupported(const Node* node);

// 缓冲区大小计算模板函数，返回存储尺寸的 vector
template <typename T>
inline std::vector<int64_t> bufferSizes(const T& t) {
  std::vector<int64_t> sizes;
  for (size_t i = 0; i < t->ndim(); i++) {
    sizes.push_back(*intValue(t->dim(i)));  // 获取每个维度的大小并存储到 sizes 中
  }
  return sizes;
}

// 获取值的维度
std::vector<ExprHandle> valueShape(const ArgValue& v);

// 如果 v 是张量，则广播它以匹配轴的形状，否则直接返回常量
ExprHandle tensorOrConstant(
    const ArgValue& v,
    const std::vector<ExprHandle>& axes);

// 标准化并检查索引，确保在列表大小范围内
int64_t normalizeAndCheckIndex(int64_t idx, int64_t list_size);

// 广播操作，返回广播后的表达式句柄
ExprHandle broadcast(BufHandle b, const std::vector<ExprHandle>& axes);

// 创建常量表达式句柄
ExprHandle constant(const ArgValue& v);

// 计算要广播的索引
std::vector<ExprHandle> computeIndicesToBroadcast(
    const std::vector<ExprHandle>& outputAxes,
    const std::vector<ExprHandle>& inputSizes);

// 获取 ArgValue 对象的名称作为字符串
inline std::string getArgValueName(const ArgValue& a) {
  if (std::holds_alternative<tensorexpr::BufHandle>(a)) {
    return "BufHandle";
  } else if (std::holds_alternative<tensorexpr::VarHandle>(a)) {
    return "VarHandle";
  } else if (std::holds_alternative<double>(a)) {
    return "double";
  } else if (std::holds_alternative<int64_t>(a)) {
    return "int64_t";
  } else if (std::holds_alternative<bool>(a)) {
    return "bool";
  } else if (std::holds_alternative<BufList>(a)) {
    return "BufList";
  } else if (std::holds_alternative<DoubleList>(a)) {
    return "DoubleList";
  } else if (std::holds_alternative<IntList>(a)) {
    return "IntList";
  } else if (std::holds_alternative<ArgNone>(a)) {
    return "None";
  } else {
    throw std::runtime_error("ArgValue type not handled in string conversion");
  }
}

// 将 ArgValue 向量转换为指定类型的向量
template <class T>
std::vector<T> convertVecArgValue(const std::vector<ArgValue>& v) {
  std::vector<T> res;
  for (auto& x : v) {
    auto val = std::get_if<T>(&x);  // 尝试获取指定类型的值
    if (val) {
      res.push_back(*val);  // 如果成功，将其添加到结果向量中
    }
  }
  return res;
}

// 这里省略了代码块的结束大括号 '}'
    } else {
      throw std::runtime_error(
          "vector type not homogeneous - found " + getArgValueName(x) +
          ", expected " + getArgValueName(v[0]));
    }
  }
  return res;



    } else {
      // 如果发现向量类型不是同质的，抛出运行时错误异常
      throw std::runtime_error(
          "vector type not homogeneous - found " + getArgValueName(x) +
          ", expected " + getArgValueName(v[0]));
    }
  }
  // 返回结果变量 res
  return res;
}

class TORCH_API TensorExprKernel {
  struct ConstantDescr {
    BufPtr buf;
    // 只有 ptr 和 node 中的一个会被使用
    // 1) ptr 用于常量张量
    // 2) node 用于常量自定义类对象
    void* ptr = nullptr;
    Node* node = nullptr;
  };

 public:
  // 构造函数参数:
  //  * subgraph
  //      - 需要编译的图形
  //  * kernel_func_name
  //      - 生成的内核函数名称
  //  * custom_lowerings
  //      - 表示一组操作的自定义降级定义的映射
  //  * symbolic_shape_inputs
  //      - 表示输入张量的符号维度的符号图输入列表
  //  * pre_alloc
  //      - 控制缓冲区的预分配的标志
  explicit TensorExprKernel(
      const std::shared_ptr<Graph>& subgraph,
      const std::string& kernel_func_name,
      std::unordered_map<c10::Symbol, NNCLoweringFunction> custom_lowerings = {},
      std::vector<int64_t> symbolic_shape_inputs = {},
      bool pre_alloc = false,
      std::unordered_map<
          const torch::jit::Value*,
          std::vector<torch::jit::StrideInput>> symbolic_strides = {});

  explicit TensorExprKernel(
      const std::shared_ptr<Graph>& subgraph,
      std::unordered_map<c10::Symbol, NNCLoweringFunction> custom_lowerings = {},
      std::vector<int64_t> symbolic_shape_inputs = {},
      bool pre_alloc = false,
      std::unordered_map<
          const torch::jit::Value*,
          std::vector<torch::jit::StrideInput>> symbolic_strides = {})
      : TensorExprKernel(
            subgraph,
            SubgraphUtils::generateNameForGraph(subgraph),
            custom_lowerings,
            symbolic_shape_inputs,
            pre_alloc,
            symbolic_strides) {}

  void run(Stack& stack) const;
  void runFast(
      const std::vector<void*>& inputs,
      const std::vector<void*>& outputs) const;
  // stack 的预期格式:
  //  ... <outputs> <inputs>
  // 即，输出的 IValues 必须在输入的 IValues 之下在 stack 中
  void runWithAllocatedOutputs(Stack& stack) const;

  void fallback(Stack& stack) const {
    InterpreterState(code_).run(stack);
  }
  void recompile();

  StmtPtr getCodeGenStmt();

  std::string getCodeText(const std::string& attr = "") {
    return codegen_->getCodeText(attr);
  }

  const std::shared_ptr<Graph> graph() {
    return graph_;
  }

  const std::vector<ConstantDescr>& getConstantDescriptors() const {
    return constants_;
  }

  const std::vector<CodeGen::BufferArg>& getBufferArgs() const {
    return bufferArgs_;
  }

  const std::string& getKernelName() const {
    return (codegen_ ? codegen_->kernel_func_name() : kernel_func_name_);
  }

  const std::vector<int64_t>& getSymbolicShapeInputs() const {
    return symbolic_shape_inputs_;
  }

 private:
  enum BackendType {
    kUninitialized,
    kSimpleIREval,
    kLLVMCodeGen,
    kCudaCodeGen,
    kBlockCodeGen,
  };

  enum MemoryLayoutPolicy {
    kContiguous,  // 内存布局策略：连续存储
    kChannelsLastNdContiguous,  // 内存布局策略：通道最后Nd连续
  };

  void compile();  // 编译函数声明
  void genInputDebugNames();  // 生成输入调试名称函数声明
  void runKernel(Stack& stack) const;  // 运行内核函数声明，接受堆栈作为参数

  std::vector<ExprHandle> sizesForValue(const torch::jit::Value* v);  // 返回给定值对应的尺寸列表

  // 这些函数广播形状并存储 `hasBroadcast_` 变量。
  std::vector<ExprHandle> broadcastShapesMut(
      const std::vector<ExprHandle>& a,
      const std::vector<ExprHandle>& b);  // 广播形状并返回变异后的形状列表
  std::vector<ExprHandle> broadcastShapesMut(
      std::vector<std::vector<ExprHandle>> shapes);  // 广播形状并返回变异后的形状列表

  ArgValue toArg(const torch::jit::Value* v) const;  // 将 Torch 值转换为参数值
  ExprHandle constant(const torch::jit::Value* v);  // 返回 Torch 值对应的常量表达式

  Tensor computeValue(const torch::jit::Value* v);  // 计算 Torch 值对应的张量值

  void bindConstant(const torch::jit::Value* v);  // 绑定常量值

  StmtPtr transformLoops(BackendType backendType, StmtPtr st);  // 转换循环语句，根据后端类型和语句指针返回转换后的语句

  std::string getCodeGenName(BackendType backendType);  // 返回给定后端类型的代码生成器名称

  void getStaticOutputSizesAndStrides(
      const at::ArrayRef<IValue>& inputs,
      std::vector<std::vector<int64_t>>* static_sizes,
      std::vector<std::vector<int64_t>>* static_strides) const;
      // 获取静态输出的尺寸和步长，存储在提供的向量中

  std::vector<CodeGen::CallArg> prepareRunArgs(
      const at::ArrayRef<IValue>& inputs,
      std::vector<at::Tensor>& outputs) const;  // 准备运行时参数，返回调用参数列表

  BackendType inferBackendTypeFromDevice(at::Device device);  // 根据设备推断后端类型

  Tensor bindInput(const torch::jit::Value* input);  // 绑定输入张量
  BlockPtr bindAllInputs();  // 绑定所有输入块

  // 推断在 NNC 融合组中传播的内存布局策略。
  // 内存布局策略可以是 `kContiguous` 或 `kChannelsLastNdContiguous`。
  //    `kContiguous`: 始终将非连续的输入张量和内部缓冲转换为连续的。
  //    `kChannelsLastNdContiguous`: 始终将输入张量和内部缓冲转换为通道最后的连续。
  // 目前，规则很简单。
  //    如果 NNC 融合组的所有输入和输出张量都是通道最后连续的，则策略是 `kChannelsLastNdContiguous`。
  //    否则，策略始终是 `kContiguous`。
  void deduceMemoryLayoutPolicy();  // 推断内存布局策略函数声明

  Tensor convertSymbolicOutputToCorrectStrides(torch::jit::Value* v);  // 将符号输出转换为正确的步长张量
  Tensor convertStaticShapeOutputToCorrectStrides(torch::jit::Value* v);  // 将静态形状输出转换为正确的步长张量
  Tensor convertSymbolicOutputToCorrectStrides(
      const std::vector<ExprHandle>& sizes,
      const std::vector<size_t>& sorted_stride_indices_descending,
      const std::vector<ExprPtr>& strides,
      BufPtr& buf);  // 将符号输出转换为正确的步长张量，使用给定的尺寸、步长索引和缓冲区指针

  NNCLoweringFunction getCustomLoweringFor(c10::Symbol op) const;  // 获取给定操作的自定义降低功能
  std::unordered_map<c10::Symbol, NNCLoweringFunction> getCustomLowerings()
      const {  // 获取所有自定义降低功能的映射表
  // 返回 custom_lowerings_ 变量
  return custom_lowerings_;
}

// 在编译时为中间缓冲区分配内存。
// 具体地，我们预先为静态大小的中间缓冲区分配内存，并以管理 JIT 常量张量的方式管理这些缓冲区：
// 将缓冲区参数推入堆栈，以便 NNC IR 可以在运行时访问它们。
std::vector<BufPtr> preAllocIntermediateBufs(
    const std::vector<BufPtr>& interm_bufs);

// 定义 UnpackedTensorOptions 结构体，用于描述解压缩的张量选项
struct UnpackedTensorOptions {
  // 可选的标量类型
  std::optional<c10::ScalarType> dtype;
  // 可选的布局方式
  std::optional<c10::Layout> layout;
  // 可选的设备类型
  std::optional<c10::Device> device;
  // 可选的固定内存标志
  std::optional<bool> pinned_memory;
// 引入 Torch API 的定义
};

// 获取 CUDA 并行循环的层级数
TORCH_API int& getTECudaPointwiseLoopLevels();
// 获取 CUDA 并行循环的块数
TORCH_API int& getTECudaPointwiseBlockCount();
// 获取 CUDA 并行循环的块大小
TORCH_API int& getTECudaPointwiseBlockSize();
// 获取是否生成块级别代码的标志
TORCH_API bool& getTEGenerateBlockCode();
// 获取在 CPU 上是否必须使用 LLVM 的标志
TORCH_API bool& getTEMustUseLLVMOnCPU();
// 获取是否允许降级的标志
TORCH_API bool fallbackAllowed();
// 设置是否允许降级的标志
TORCH_API bool setFallbackAllowed(bool value);
// 获取是否优化条件语句的标志
TORCH_API bool& getCatWoConditionals();
// 获取是否优化条件表达式的标志
TORCH_API bool& getOptConditionals();

// 选择设备类型作为首选设备
TORCH_API std::optional<at::Device> pickDeviceType(
    const at::ArrayRef<torch::jit::Value*>& inputs);

// 检查给定值是否是连续的内存布局
bool isContiguous(
    const torch::jit::Value* v,
    at::MemoryFormat memory_format = at::MemoryFormat::Contiguous);

// 结束 'tensorexpr' 命名空间
} // namespace tensorexpr
// 结束 'jit' 命名空间
} // namespace jit
// 结束 'torch' 命名空间
} // namespace torch
```