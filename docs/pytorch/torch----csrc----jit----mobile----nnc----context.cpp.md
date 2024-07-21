# `.\pytorch\torch\csrc\jit\mobile\nnc\context.cpp`

```
#include <torch/csrc/jit/mobile/nnc/context.h>

#include <ATen/Functions.h>
#include <ATen/core/functional.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/mobile/nnc/registry.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

// 定义了 NNCompiler 文件格式的版本号
constexpr int64_t kProducedNNCFileFormatVersion = 0x1L;

namespace {

// 辅助函数：创建包含给定 IValue 的元组
c10::IValue Tup(std::initializer_list<c10::IValue> ivalues) {
  return c10::ivalue::Tuple::create(ivalues);
}

// 辅助函数：创建包含给定移动语义 IValue 的元组
c10::IValue Tup(std::vector<c10::IValue>&& ivalues) {
  return c10::ivalue::Tuple::create(ivalues);
}

} // namespace

// 构造函数：从 IValue 创建输入规格
InputSpec::InputSpec(const c10::IValue& value) {
  auto dict = value.toGenericDict();
  // 从字典中获取并设置 sizes
  sizes_ = dict.at("sizes").toIntVector();
  // 从字典中获取并设置 dtype
  dtype_ = dict.at("dtype").toScalarType();
}

// 序列化函数：将输入规格序列化为 IValue
c10::IValue InputSpec::serialize() const {
  // 创建空字典
  c10::Dict<c10::IValue, c10::IValue> dict(
      at::StringType::get(), at::AnyType::get());
  // 插入 sizes 到字典
  dict.insert("sizes", sizes_);
  // 插入 dtype 到字典
  dict.insert("dtype", dtype_);
  return dict;
}

// 验证函数：验证输入张量是否符合规格
bool InputSpec::validate(const at::Tensor& input) const {
  // 检查 sizes 是否匹配和 dtype 是否匹配
  if (sizes_.size() != input.sizes().size() || input.scalar_type() != dtype_) {
    return false;
  }
  auto spec_sizes = sizes_;
  for (const auto i : c10::irange(spec_sizes.size())) {
    // 如果 spec_sizes[i] 为 0，则表示该维度是动态的
    if (spec_sizes[i] != 0 && spec_sizes[i] != input.sizes()[i]) {
      return false;
    }
  }
  return true;
}

// 构造函数：从 IValue 创建输出规格
OutputSpec::OutputSpec(const c10::IValue& value) {
  auto dict = value.toGenericDict();
  // 从字典中获取并设置 sizes
  sizes_ = dict.at("sizes").toIntVector();
  // 从字典中获取并设置 dtype
  dtype_ = dict.at("dtype").toScalarType();
  // 如果字典包含 qscale，则从字典中获取并设置 qscale
  if (dict.contains("qscale")) {
    qscale_ = dict.at("qscale").toDouble();
  }
  // 如果字典包含 qzero，则从字典中获取并设置 qzero
  if (dict.contains("qzero")) {
    qzero_ = dict.at("qzero").toInt();
  }
}

// 序列化函数：将输出规格序列化为 IValue
c10::IValue OutputSpec::serialize() const {
  // 创建空字典
  c10::Dict<c10::IValue, c10::IValue> dict(
      at::StringType::get(), at::AnyType::get());
  // 插入 sizes 到字典
  dict.insert("sizes", sizes_);
  // 插入 dtype 到字典
  dict.insert("dtype", dtype_);
  // 如果 qscale 存在，则插入 qscale 到字典
  if (qscale_) {
    dict.insert("qscale", *qscale_);
  }
  // 如果 qzero 存在，则插入 qzero 到字典
  if (qzero_) {
    dict.insert("qzero", *qzero_);
  }
  return dict;
}

// 分配函数：根据规格分配输出张量
at::Tensor OutputSpec::allocate() const {
  // 如果是量化整数类型
  if (isQIntType(dtype_)) {
    TORCH_CHECK(
        qscale_ && qzero_,
        "Quantized output tensor must have qscale_ and qzero_");
    // 使用给定的 qscale 和 qzero 创建量化整数张量
    return at::_empty_affine_quantized(
        sizes_,
        at::TensorOptions()
            .dtype(dtype_)
            .layout(at::kStrided)
            .device(at::kCPU)
            .requires_grad(false),
        *qscale_,
        *qzero_);
  }
  // 否则，创建普通张量
  return at::empty(
      sizes_,
      at::TensorOptions()
          .dtype(dtype_)
          .layout(at::kStrided)
          .device(at::kCPU)
          .requires_grad(false));
}

// 构造函数：从 IValue 创建内存计划
MemoryPlan::MemoryPlan(const c10::IValue& value) {
  auto dict = value.toGenericDict();
  // 从字典中获取并设置 buffer_sizes
  buffer_sizes_ = dict.at("buffer_sizes").toIntVector();
}
c10::IValue MemoryPlan::serialize() const {
  // 创建一个空的字典，键和值的类型均为任意类型
  c10::Dict<c10::IValue, c10::IValue> dict(
      at::StringType::get(), at::AnyType::get());
  // 将 buffer_sizes_ 插入到字典中作为 "buffer_sizes" 键的值
  dict.insert("buffer_sizes", buffer_sizes_);
  // 返回创建的字典作为序列化结果
  return dict;
}

void MemoryPlan::allocate(ExecutionState* state) const {
  // 获取 state 对象中的 preallocations_，并清空其中的内容
  auto& allocations = state->preallocations_;
  allocations.clear();
  // 预留足够的空间以容纳所有 buffer_sizes_ 的大小
  allocations.reserve(buffer_sizes_.size());
  // 遍历 buffer_sizes_ 中的每一个大小
  for (int64_t buffer_size : buffer_sizes_) {
    // 使用 CPU 分配器分配指定大小的内存，并将其包装为 DataPtr 对象
    at::DataPtr buffer = c10::GetCPUAllocator()->allocate(buffer_size);
    // 将分配的内存 buffer 添加到 allocations 中
    allocations.emplace_back(std::move(buffer));
  }
}

Function::Function(const c10::IValue& value) {
  // 将传入的 IValue 对象转换为通用字典 dict
  auto dict = value.toGenericDict();
  // 从 dict 中读取 "name" 键的字符串值，并转换为 QualifiedName 对象赋给 name_
  name_ = c10::QualifiedName(dict.at("name").toStringRef());
  // 从 dict 中读取 "nnc_kernel_id" 键的字符串值赋给 nnc_kernel_id_
  nnc_kernel_id_ = dict.at("nnc_kernel_id").toStringRef();
  // 从 dict 中读取 "parameters" 键的列表值赋给 parameters_
  parameters_ = dict.at("parameters").toList();

  // input_specs_
  // 从 dict 中读取 "input_specs" 键的元组值，遍历其中的每个元素并添加到 input_specs_ 中
  for (const auto& input_value :
       dict.at("input_specs").toTupleRef().elements()) {
    input_specs_.emplace_back(input_value);
  }

  // output_specs_
  // 从 dict 中读取 "output_specs" 键的元组值，遍历其中的每个元素并添加到 output_specs_ 中
  for (const auto& output_value :
       dict.at("output_specs").toTupleRef().elements()) {
    output_specs_.emplace_back(output_value);
  }

  // memory_plan_
  // 从 dict 中读取 "memory_plan" 键的值，并用其构造 MemoryPlan 对象赋给 memory_plan_
  memory_plan_ = MemoryPlan(dict.at("memory_plan"));

  // symbolic shape positions
  // 从 dict 中读取 "sym_shape_pos" 键的元组值，遍历其中的每个元素并添加到 sym_shape_positions_ 中
  for (const auto& sym_shape_pos :
       dict.at("sym_shape_pos").toTupleRef().elements()) {
    auto sym_shape_elements = sym_shape_pos.toTupleRef().elements();
    // 将元组中的两个元素转换为整数并作为 pair 添加到 sym_shape_positions_ 中
    sym_shape_positions_.emplace_back(
        sym_shape_elements[0].toInt(), sym_shape_elements[1].toInt());
  }
}

c10::IValue Function::serialize() const {
  // 创建一个空的字典，键和值的类型均为任意类型
  c10::Dict<c10::IValue, c10::IValue> dict(
      at::StringType::get(), at::AnyType::get());

  // 将 name_ 的 QualifiedName 字符串表示形式插入字典作为 "name" 键的值
  dict.insert("name", name_.qualifiedName());
  // 将 nnc_kernel_id_ 插入字典作为 "nnc_kernel_id" 键的值
  dict.insert("nnc_kernel_id", nnc_kernel_id_);
  
  // TODO: 应该使用 Module 而不是每个 Method 单独序列化 parameters。
  // 并且如果可以在同一个模型文件中序列化原始模型和编译模型，参数应该共享。
  // 将 parameters_ 插入字典作为 "parameters" 键的值
  dict.insert("parameters", parameters_);

  // input_specs_
  std::vector<c10::IValue> input_specs;
  input_specs.reserve(input_specs_.size());
  // 遍历 input_specs_ 中的每个元素，将其序列化后添加到 input_specs 中
  for (const auto& input_spec : input_specs_) {
    input_specs.emplace_back(input_spec.serialize());
  }
  // 将序列化后的 input_specs 放入字典中作为 "input_specs" 键的值
  dict.insert("input_specs", Tup(std::move(input_specs)));

  // output_specs_
  std::vector<c10::IValue> output_specs;
  output_specs.reserve(output_specs_.size());
  // 遍历 output_specs_ 中的每个元素，将其序列化后添加到 output_specs 中
  for (const auto& output_spec : output_specs_) {
    output_specs.emplace_back(output_spec.serialize());
  }
  // 将序列化后的 output_specs 放入字典中作为 "output_specs" 键的值
  dict.insert("output_specs", Tup(std::move(output_specs)));

  // memory_plan_
  // 将 memory_plan_ 对象序列化后作为 "memory_plan" 键的值插入字典中
  dict.insert("memory_plan", memory_plan_.serialize());

  // sym_shape_positions_
  std::vector<c10::IValue> sym_shape_pos_vec;
  sym_shape_pos_vec.reserve(sym_shape_positions_.size());
  // 遍历 sym_shape_positions_ 中的每个 pair，将其序列化后添加到 sym_shape_pos_vec 中
  for (const auto& sym_shape_pos : sym_shape_positions_) {
    sym_shape_pos_vec.emplace_back(Tup({sym_shape_pos.first, sym_shape_pos.second}));
  }
  // 将序列化后的 sym_shape_pos_vec 放入字典中作为 "sym_shape_pos" 键的值
  dict.insert("sym_shape_pos", Tup(std::move(sym_shape_pos_vec)));

  // 返回创建的字典作为序列化结果
  return dict;
}
    # 将包含元组的 Tup 对象添加到 sym_shape_pos_vec 向量中
    sym_shape_pos_vec.emplace_back(
        Tup({sym_shape_pos.input_idx_, sym_shape_pos.dim_idx_}));
  }
  # 将键为 "sym_shape_pos"，值为包含 sym_shape_pos_vec 的 Tup 对象，插入到 dict 中
  dict.insert("sym_shape_pos", Tup(std::move(sym_shape_pos_vec)));

  # 返回包含 "sym_shape_pos" 键值对的 dict 字典对象
  return dict;
}

// 初始化执行状态函数，确保执行状态对象有效
void Function::init_execution_state() const {
  // 如果执行状态对象已经存在，则直接返回
  if (execution_state_.get() != nullptr) {
    return;
  }

  // 创建新的执行状态对象
  ExecutionState state;
  // 使用内存计划对象为执行状态分配内存
  memory_plan_.allocate(&state);

  // 参数向量包含五个部分：输入、符号形状、输出、参数和缓冲区
  auto input_args = input_specs_.size();
  auto sym_shape_args = sym_shape_positions_.size();
  auto output_args = output_specs_.size();
  auto param_args = parameters_.size();
  auto buffer_args = state.preallocations_.size();

  auto& arguments = state.arguments_;
  arguments.reserve(
      input_args + sym_shape_args + output_args + param_args + buffer_args);

  // 预留空的槽位，在执行时填充输入/输出指针
  arguments.resize(input_args + sym_shape_args + output_args);

  // 将参数填充为未类型化的原始指针
  // 参数的底层存储应由 `parameters_` 持有，而 `execution_state_` 使用时 `parameters_` 应保持有效
  for (const auto& param : parameters_) {
    const c10::IValue& ivalue = (c10::IValue)param;
    if (ivalue.isTensor()) {
      arguments.emplace_back(ivalue.toTensor().data_ptr());
    } else if (torch::isCustomClass(ivalue)) {
      arguments.emplace_back(ivalue.toObjectRef().getSlot(0).toCapsule().get());
    } else {
      TORCH_CHECK(false, "Invalid parameter: ", ivalue);
    }
  }

  // 填充预分配缓冲区指针
  for (const auto& preallocation : state.preallocations_) {
    arguments.emplace_back(preallocation.get());
  }

  // 使用移动语义创建唯一的执行状态对象
  execution_state_ = std::make_unique<ExecutionState>(std::move(state));
}

// 运行函数，接收输入并返回运行结果
c10::impl::GenericList Function::run(
    const c10::impl::GenericList& inputs) const {
  // 检查是否存在指定的 NNC 核心函数
  TORCH_CHECK(
      registry::has_nnc_kernel(nnc_kernel_id_),
      "Cannot find NNC kernel: ",
      nnc_kernel_id_);

  // 初始化执行状态
  init_execution_state();

  // 获取执行状态中的参数向量引用
  std::vector<void*>& args = execution_state_->arguments_;

  // 填充输入张量
  TORCH_CHECK(
      input_specs_.size() == inputs.size(),
      "Input size doesn't match the spec, expect: ",
      input_specs_.size(),
      " actual: ",
      inputs.size());
  std::vector<int64_t> scalar_values;
  int offset = 0;
  for (const auto i : c10::irange(inputs.size())) {
    const c10::IValue& input = inputs[i];
    const auto& spec = input_specs_[i];
    const auto& input_tensor = input.toTensor();
    // 验证输入张量是否有效
    TORCH_CHECK(spec.validate(input_tensor), "Invalid input at pos: ", i);
    args[i] = input_tensor.data_ptr();
  }
  offset += inputs.size();

  // 预留空间以存储符号形状参数
  scalar_values.reserve(sym_shape_positions_.size());
  for (const auto i : c10::irange(sym_shape_positions_.size())) {
    const auto& sym_shape_pos = sym_shape_positions_[i];
    const c10::IValue& input = inputs[sym_shape_pos.input_idx_];
    // 获取符号形状的维度大小并存储在 scalar_values 中
    auto dim = input.toTensor().size(sym_shape_pos.dim_idx_);
    scalar_values.push_back(dim);
    args[i + offset] = &scalar_values[scalar_values.size() - 1];

# 将指向 `scalar_values` 中最后一个标量值的指针赋给 `args` 数组中的第 `i + offset` 个位置。


  }
  offset += sym_shape_positions_.size();

# 更新 `offset` 的值，使其增加等于 `sym_shape_positions_` 的大小。


  // Preallocate and fill in output tensors.
  c10::List<at::Tensor> outputs;
  outputs.reserve(output_specs_.size());

# 创建一个 `c10::List` 类型的列表 `outputs`，用于存储输出张量，并预留足够的空间以容纳 `output_specs_` 的大小。


  for (const auto i : c10::irange(output_specs_.size())) {
    at::Tensor output = output_specs_[i].allocate();
    outputs.emplace_back(output);
    args[i + offset] = output.data_ptr();
  }

# 遍历 `output_specs_` 列表中的每个索引 `i`，分配一个张量 `output` 并添加到 `outputs` 列表中，同时将 `output` 的数据指针赋给 `args` 数组中的第 `i + offset` 个位置。


  // TODO: check consistency, e.g.: code version, input shape and compiled
  // shape, etc.

# TODO：检查一致性，例如代码版本、输入形状和编译后的形状等。


  auto kernel = registry::get_nnc_kernel(nnc_kernel_id_);
  kernel->execute(args.data());

# 获取 `nnc_kernel_id_` 对应的 NNCompiler 内核，并执行使用 `args` 数组中的数据进行的操作。


  return c10::impl::toList(outputs);

# 将 `outputs` 转换为 `c10::List` 类型，并返回作为输出结果。
}

CompilationUnit::CompilationUnit(const c10::IValue& value) {
  // 从给定的IValue中获取元组引用的根元素
  const auto& root = value.toTupleRef().elements();
  // 获取根元素中的第二个元素，即函数列表
  const auto& functions = root[1].toTupleRef().elements();
  // 遍历函数列表中的每一个函数，并注册到CompilationUnit中
  for (const auto& function : functions) {
    register_function(std::make_unique<Function>(function));
  }
}

c10::IValue CompilationUnit::serialize() const {
  // 序列化所有函数，将函数列表转换为IValue列表
  auto functions =
      c10::fmap(functions_, [](decltype(functions_)::const_reference func) {
        return func.second->serialize();
      });
  // 构造一个元组，包含NNC文件格式版本和函数列表
  return Tup({kProducedNNCFileFormatVersion, Tup(std::move(functions))});
}

c10::impl::GenericList CompilationUnit::run(
    const c10::QualifiedName& name,
    const c10::impl::GenericList& inputs) const {
  // 根据函数名称查找函数对象
  Function* func = find_function(name);
  // 检查找到的函数对象是否为空，如果为空则抛出异常
  TORCH_CHECK(
      func != nullptr, "Function '", name.qualifiedName(), "' is not defined.");
  // 调用函数对象的run方法，执行函数，并返回结果
  return func->run(inputs);
}

void CompilationUnit::register_function(std::unique_ptr<Function> fn) {
  // 检查函数是否已经定义，如果已经定义则抛出异常
  TORCH_CHECK(
      0 == functions_.count(fn->name()),
      "method '",
      fn->name().qualifiedName(),
      "' already defined.");
  // 获取函数的名称，并将函数对象插入到functions_映射中
  const auto& name = fn->name();
  functions_.emplace(name, std::move(fn));
}

Function* CompilationUnit::find_function(const c10::QualifiedName& name) const {
  // 根据函数名称在functions_映射中查找函数对象
  auto it = functions_.find(name);
  // 如果未找到，则返回nullptr
  if (it == functions_.end()) {
    return nullptr;
  }
  // 返回找到的函数对象的指针
  return it->second.get();
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
```