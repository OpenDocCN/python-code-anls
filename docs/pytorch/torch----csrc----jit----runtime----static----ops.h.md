# `.\pytorch\torch\csrc\jit\runtime\static\ops.h`

```
#pragma once
// 预处理指令，确保此头文件仅被编译一次

#include <ATen/Utils.h>
// 包含 ATen 库的实用工具函数

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 的中间表示(IR)相关头文件

#include <torch/csrc/jit/runtime/static/impl.h>
// 包含 Torch JIT 静态运行时实现的头文件

namespace at::native {
// ATen 命名空间下的 native 命名空间

at::Tensor& reshape_copy_out(
    at::Tensor& out,
    const at::Tensor& self,
    const at::DimVector& proposed_shape,
    bool infer_size = true);
// 函数声明：reshape_copy_out，用于重新塑形 Tensor，可能会复制数据

at::Tensor& to_copy_out(
    Tensor& out,
    const Tensor& self,
    bool non_blocking,
    bool copy_strides,
    std::optional<MemoryFormat> memory_format);
// 函数声明：to_copy_out，用于将 Tensor 复制到指定的输出 Tensor 中
} // namespace at::native
// 结束 ATen native 命名空间的定义

namespace torch::jit {
// Torch JIT 命名空间

using SROpFunctor = SROperator (*)(Node* n);
// 定义 SROpFunctor 类型别名，表示一个接受 Node* 参数并返回 SROperator 的函数指针

struct SROperatorFunctor {
  virtual SROperator Generate(Node*) {
    SROperator out;
    return out;
  }
  // 虚拟函数 Generate，基类默认实现返回空的 SROperator

  virtual ~SROperatorFunctor() = default;
  // 虚析构函数，保证正确的派生类对象销毁

};

TORCH_DECLARE_REGISTRY(SROperatorRegistry, SROperatorFunctor);
// 声明 SROperatorRegistry 注册表，用于注册 SROperatorFunctor 类型的工厂函数

#define REGISTER_OPERATOR_FUNCTOR(name, id, ...)             \
  struct SROperatorFunctor_##id : public SROperatorFunctor { \
    const SROpFunctor fn = __VA_ARGS__;                      \
    SROperator Generate(Node* n) override {                  \
      return fn(n);                                          \
    }                                                        \
  };                                                         \
  C10_REGISTER_CLASS(SROperatorRegistry, name, SROperatorFunctor_##id);
// 宏定义 REGISTER_OPERATOR_FUNCTOR，用于注册操作符的函数对象生成器

TORCH_DECLARE_REGISTRY(SRNativeOperatorRegistry, SROperatorFunctor);
// 声明 SRNativeOperatorRegistry 注册表，用于注册 SROperatorFunctor 类型的本地操作符工厂函数

#define REGISTER_NATIVE_OPERATOR_FUNCTOR(name, id, ...)            \
  struct SRNativeOperatorFunctor_##id : public SROperatorFunctor { \
    const SROpFunctor fn = __VA_ARGS__;                            \
    SROperator Generate(Node* n) override {                        \
      return fn(n);                                                \
    }                                                              \
  };                                                               \
  C10_REGISTER_CLASS(                                              \
      SRNativeOperatorRegistry, name, SRNativeOperatorFunctor_##id);
// 宏定义 REGISTER_NATIVE_OPERATOR_FUNCTOR，用于注册本地操作符的函数对象生成器

inline at::Tensor create_empty_from(const at::Tensor& t) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      t.device(),
      c10::nullopt,
      c10::nullopt);
}
// 内联函数 create_empty_from，创建一个空的 Tensor，与给定 Tensor 具有相同的数据类型等属性

inline at::Tensor create_empty_from(
    at::IntArrayRef sizes,
    const at::Tensor& t) {
  return at::detail::empty_cpu(
      sizes,
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      t.device(),
      c10::nullopt,
      c10::nullopt);
}
// 内联函数 create_empty_from，创建一个空的 Tensor，具有指定大小和与给定 Tensor 相同的属性

inline at::Tensor create_empty(c10::ScalarType dtype) {
  return at::detail::empty_cpu(
      {0}, dtype, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
}
// 内联函数 create_empty，创建一个空的 Tensor，指定数据类型，其余属性使用默认值

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::ScalarType dtype) {
  return at::detail::empty_cpu(
      {0}, dtype, t.layout(), t.device(), c10::nullopt, c10::nullopt);
}
// 内联函数 create_empty_from，创建一个空的 Tensor，指定数据类型，与给定 Tensor 其他属性相同
inline at::Tensor create_empty_from(const at::Tensor& t, c10::Layout layout) {
    // 使用 empty_cpu 函数创建一个形状为空的张量，具体参数如下：
    // - 形状为 {0}
    // - 数据类型为 t 的数据类型
    // - 布局为指定的 layout
    // - 设备与输入张量 t 相同
    // - 不使用任何可选参数
    // 返回创建的空张量
    return at::detail::empty_cpu(
        {0},
        c10::typeMetaToScalarType(t.dtype()),
        layout,
        t.device(),
        c10::nullopt,
        c10::nullopt);
}

inline at::Tensor create_empty_from(const at::Tensor& t, c10::Device device) {
    // 使用 empty_cpu 函数创建一个形状为空的张量，具体参数如下：
    // - 形状为 {0}
    // - 数据类型为 t 的数据类型
    // - 布局与输入张量 t 相同
    // - 设备为指定的 device
    // - 不使用任何可选参数
    // 返回创建的空张量
    return at::detail::empty_cpu(
        {0},
        c10::typeMetaToScalarType(t.dtype()),
        t.layout(),
        device,
        c10::nullopt,
        c10::nullopt);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::MemoryFormat memory_format) {
    // 使用 empty_cpu 函数创建一个形状为空的张量，具体参数如下：
    // - 形状为 {0}
    // - 数据类型为 t 的数据类型
    // - 布局与输入张量 t 相同
    // - 设备与输入张量 t 相同
    // - 内存格式为指定的 memory_format
    // 返回创建的空张量
    return at::detail::empty_cpu(
        {0},
        c10::typeMetaToScalarType(t.dtype()),
        t.layout(),
        t.device(),
        c10::nullopt,
        memory_format);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::ScalarType dtype,
    c10::MemoryFormat memory_format) {
    // 使用 empty_cpu 函数创建一个形状为空的张量，具体参数如下：
    // - 形状为 {0}
    // - 数据类型为指定的 dtype
    // - 布局与输入张量 t 相同
    // - 设备与输入张量 t 相同
    // - 内存格式为指定的 memory_format
    // 返回创建的空张量
    return at::detail::empty_cpu(
        {0}, dtype, t.layout(), t.device(), c10::nullopt, memory_format);
}

inline bool checkResizedDataPtr(at::Tensor& t) {
    // 记录当前张量 t 的数据指针
    auto const prev_data_ptr = t.data_ptr();
    // 调整张量 t 的大小为 {0}
    t.resize_({0});
    // 检查调整大小后张量 t 的数据指针是否保持不变，并返回结果
    return prev_data_ptr == t.data_ptr();
}

inline void fastResizeToZero(at::Tensor& t) {
    // 使用 unsafeGetTensorImpl 函数获取张量 t 的实现，将其大小设置为 {0} 并确保是连续的
    t.unsafeGetTensorImpl()->set_sizes_contiguous({0});
    // 在调试模式下，验证调整大小后张量 t 的数据指针是否保持不变
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(checkResizedDataPtr(t));
}

// 检查是否在 Static Runtime 中已注册指定操作的 out 变体
bool opIsRegistered(const c10::Symbol& op_name);

// 检查 Static Runtime 是否可以本地运行指定的操作
// jit 解释器中实现的 prim 操作在 Static Runtime 中作为本地操作实现
bool nativeOpIsRegistered(const c10::Symbol& op_name);

// 检查是否可以重用节点 n 的输入和输出
bool canReuseInputsOutputs(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant);

// 检查节点 n 是否为可优化容器类型
bool isOptimizableContainerType(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant);

// 获取节点 n 的 out-of-place 操作
SROperator getOutOfPlaceOperation(Node* n);

// 获取节点 n 的本地操作
SROperator getNativeOperation(Node* n);

// 检查节点 n 是否具有可变长度参数
bool hasVarArgs(Node* n);

inline std::string PrintNode(const Node* node) {
    // 使用流输出节点 node 的信息，并将其转换为字符串返回
    std::ostringstream ss;
    node->print(ss, 0, nullptr, false);
    return ss.str();
}

inline void LogAndDumpSchema(const Node* node) {
    // 记录和输出节点的架构信息
    VLOG(1) << "Found schema mismatch for: " << node->schema();
}

inline bool sr_schema_check(torch::jit::Node*) {
    // 默认实现，始终返回 true，表示架构检查通过
    return true;
}

template <typename Schema, typename... Schemas>
bool sr_schema_check(
    torch::jit::Node* node,
    Schema&& first,
    Schemas&&... rest) {
    // 检查节点 node 是否与任一提供的架构匹配，如果不匹配则输出架构信息并返回 false
    auto is_match = node->matches(first) || sr_schema_check(node, rest...);
    if (!is_match) {
        torch::jit::LogAndDumpSchema(node);
    }
    return is_match;
}

// 检查节点 node 的类型是否与给定的 node_kind 匹配
bool sr_schema_check_kind(torch::jit::Node* node, c10::Symbol node_kind);

} // namespace torch::jit
```