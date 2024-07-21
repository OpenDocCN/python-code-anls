# `.\pytorch\torch\csrc\jit\mobile\nnc\context.h`

```
#pragma once



// 指令编译器一次包含头文件
#include <memory>
#include <string>
#include <utility>
#include <vector>

// 引入 ATen 库的 IValue 头文件
#include <ATen/core/ivalue.h>
// 引入 C10 库的 ScalarType 头文件
#include <c10/core/ScalarType.h>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {
// 移动端命名空间
namespace mobile {
// NNC 命名空间
namespace nnc {

// 指定输入张量的需求规范
// TODO: 支持具有动态形状的输入张量（PR #54982）
struct TORCH_API InputSpec {
  InputSpec() = default;

  // 从 IValue 反序列化规范
  explicit InputSpec(const c10::IValue& value);

  // 将规范序列化为 IValue
  C10_NODISCARD c10::IValue serialize() const;

  // 检查输入张量是否符合规范
  C10_NODISCARD bool validate(const at::Tensor& input) const;

  // 尺寸列表
  std::vector<int64_t> sizes_;
  // 数据类型
  c10::ScalarType dtype_{c10::ScalarType::Undefined};
};

// 指定输出张量的大小/数据类型等，以预先分配输出
// TODO: 支持内核动态分配输出张量的情况
struct TORCH_API OutputSpec {
  OutputSpec() = default;

  // 从 IValue 反序列化规范
  explicit OutputSpec(const c10::IValue& value);

  // 将规范序列化为 IValue
  C10_NODISCARD c10::IValue serialize() const;

  // 根据规范分配一个输出张量
  C10_NODISCARD at::Tensor allocate() const;

  // 尺寸列表
  std::vector<int64_t> sizes_;
  // 数据类型
  c10::ScalarType dtype_{c10::ScalarType::Undefined};
  // 可选的量化比例
  std::optional<double> qscale_;
  // 可选的量化零点
  std::optional<int64_t> qzero_;
};

// 保存执行期间所需的临时缓冲区/状态
struct TORCH_API ExecutionState {
  ExecutionState() = default;
  ExecutionState(const ExecutionState&) = delete;
  ExecutionState(ExecutionState&&) = default;
  ExecutionState& operator=(const ExecutionState&) = delete;
  ExecutionState& operator=(ExecutionState&&) = default;

  // NNC 内核需要的预分配缓冲区
  std::vector<c10::DataPtr> preallocations_;

  // NNC 内核期望以下参数布局：
  //   输入张量 1
  //   ...
  //   输入张量 INPUT_NUM
  //   输出张量 1
  //   ...
  //   输出张量 OUTPUT_NUM
  //   参数张量 1
  //   ...
  //   参数张量 PARAM_NUM
  //   临时缓冲区 1
  //   ...
  //   临时缓冲区 BUFFER_NUM
  std::vector<void*> arguments_;
};

// 指定如何在初始化时分配临时缓冲区
struct TORCH_API MemoryPlan {
  MemoryPlan() = default;

  explicit MemoryPlan(const c10::IValue& value);

  C10_NODISCARD c10::IValue serialize() const;

  void allocate(ExecutionState* state) const;

  // 缓冲区尺寸列表
  std::vector<int64_t> buffer_sizes_;
};

// 符号形状在输入维度中的位置
struct TORCH_API SymbolicShapePosition {
  SymbolicShapePosition() = default;
  SymbolicShapePosition(int64_t input_idx, int64_t dim_idx)
      : input_idx_(input_idx), dim_idx_(dim_idx) {}

  // 输入索引
  int64_t input_idx_;
  // 维度索引
  int64_t dim_idx_;
};

// 表示已编译的 NNC 函数，与 `Method`（例如 `forward`）具有一一对应关系，类似于 torch::jit::mobile::Function.
class TORCH_API Function {
 public:
  explicit Function() = default;

  // 从由 'serialize()' 方法生成的 IValue 反序列化得到 Function 对象
  explicit Function(const c10::IValue& value);

  // 将 Function 对象序列化成 IValue
  c10::IValue serialize() const;

  // 执行编译后的 NNC 函数
  c10::impl::GenericList run(const c10::impl::GenericList& inputs) const;

  // 函数的名称，与模型代码中指定的名称对应
  c10::QualifiedName name() const {
    return name_;
  }

  // 设置函数的名称
  void set_name(const c10::QualifiedName& name) {
    name_ = name;
  }

  // 生成的 NNC 内核的唯一标识符
  const std::string& nnc_kernel_id() const {
    return nnc_kernel_id_;
  }

  // 设置 NNC 内核的唯一标识符
  void set_nnc_kernel_id(const std::string& name) {
    nnc_kernel_id_ = name;
  }

  // 要传递给生成的 NNC 内核的参数（例如权重/偏置张量）
  const c10::impl::GenericList& parameters() const {
    return parameters_;
  }

  // 设置要传递给生成的 NNC 内核的参数
  void set_parameters(const c10::impl::GenericList& parameters) {
    parameters_ = parameters;
  }

  // 输入规格列表
  const std::vector<InputSpec>& input_specs() const {
    return input_specs_;
  }

  // 设置输入规格列表
  void set_input_specs(const std::vector<InputSpec>& input_specs) {
    input_specs_ = input_specs;
  }

  // 输出规格列表
  const std::vector<OutputSpec>& output_specs() const {
    return output_specs_;
  }

  // 设置输出规格列表
  void set_output_specs(const std::vector<OutputSpec>& output_specs) {
    output_specs_ = output_specs;
  }

  // 内存计划
  const MemoryPlan& memory_plan() const {
    return memory_plan_;
  }

  // 设置内存计划
  void set_memory_plan(const MemoryPlan& memory_plan) {
    memory_plan_ = memory_plan;
  }

  // 符号形状位置列表
  const std::vector<SymbolicShapePosition>& sym_shape_positions() const {
    return sym_shape_positions_;
  }

  // 设置符号形状位置列表
  void set_sym_shape_positions(
      const std::vector<SymbolicShapePosition>& sym_shape_pos) {
    sym_shape_positions_ = sym_shape_pos;
  }

 private:
  // 初始化执行状态
  void init_execution_state() const;

  c10::QualifiedName name_;  // 函数名称
  std::string nnc_kernel_id_;  // NNC 内核的唯一标识符
  c10::impl::GenericList parameters_{at::AnyType::get()};  // 参数列表，默认为任意类型
  std::vector<InputSpec> input_specs_;  // 输入规格列表
  std::vector<OutputSpec> output_specs_;  // 输出规格列表
  std::vector<SymbolicShapePosition> sym_shape_positions_;  // 符号形状位置列表
  MemoryPlan memory_plan_;  // 内存计划
  mutable std::unique_ptr<ExecutionState> execution_state_;  // 可变的执行状态的唯一指针
};

// CompilationUnit 包含一组编译后的 NNC 函数，与 'Module' 对应的数量一一对应。
// 它类似于 torch::jit::mobile::CompilationUnit。
// Torch API 中的编译单元类定义
class TORCH_API CompilationUnit {
 public:
  // 默认构造函数
  CompilationUnit() = default;
  // 禁用复制构造函数
  CompilationUnit(const CompilationUnit&) = delete;
  // 移动构造函数
  CompilationUnit(CompilationUnit&&) = default;
  // 禁用赋值运算符重载
  CompilationUnit& operator=(const CompilationUnit&) = delete;
  // 移动赋值运算符重载
  CompilationUnit& operator=(CompilationUnit&&) = default;

  // 从由 'serialize()' 方法生成的 IValue 反序列化
  explicit CompilationUnit(const c10::IValue& value);

  // 将所有注册的函数序列化为 IValue。此 IValue 将预先保存在主机上的编译的 TorchScript 模型文件中，并在目标设备上运行时进行反序列化。
  C10_NODISCARD c10::IValue serialize() const;

  // 执行已注册的函数。
  C10_NODISCARD c10::impl::GenericList run(
      const c10::QualifiedName& function_name,
      const c10::impl::GenericList& inputs) const;

  // 向编译单元注册一个函数。
  void register_function(std::unique_ptr<Function> fn);

 private:
  // 根据限定名查找函数
  C10_NODISCARD Function* find_function(const c10::QualifiedName& qn) const;

  // 存储注册函数的哈希映射，限定名到函数的唯一指针
  std::unordered_map<c10::QualifiedName, std::unique_ptr<Function>> functions_;
};

// namespace nnc
// namespace mobile
// namespace jit
// namespace torch
// Torch API 的命名空间结尾
```