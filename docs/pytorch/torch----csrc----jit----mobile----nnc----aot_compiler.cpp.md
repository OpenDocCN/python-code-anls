# `.\pytorch\torch\csrc\jit\mobile\nnc\aot_compiler.cpp`

```py
// 包含头文件，用于移动端 AOT 编译器的前向声明
#include <torch/csrc/jit/mobile/nnc/aot_compiler.h>

// 包含 ATen 库的功能和原生功能的头文件
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

// 包含用于后端处理的头文件
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>

// 包含 JIT 的中间表示和日志的头文件
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>

// 包含 JIT 的各种优化 passes 的头文件
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>

// 包含 JIT 运行时的跟踪头文件
#include <torch/csrc/jit/runtime/jit_trace.h>

// 包含 TensorExpr 库的图优化和 IR 头文件
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

// 包含标准库的文件流操作头文件
#include <fstream>

// 使用 torch::jit 命名空间
using namespace torch::jit;

// 使用 torch::jit::tensorexpr 命名空间
using namespace torch::jit::tensorexpr;

// 定义 torch 命名空间
namespace torch {
// 定义 JIT 命名空间
namespace jit {
// 定义移动端命名空间
namespace mobile {
// 定义 NNC 命名空间
namespace nnc {

// TODO(mvz): 暂时禁用移动端构建中的 NNC 后端。
/*
static std::vector<int64_t> getConstSizes(const BufPtr b) {
  std::vector<int64_t> r;
  for (const auto& dim : b->dims()) {
    LongImmPtr imm_dim = to<LongImm>(dim);
    // TODO: 断言其实际为立即数
    int64_t s = imm_dim->value();
    r.push_back(s);
  }
  return r;
}
*/

// 从原始图的输入构造输入规范向量
static std::vector<mobile::nnc::InputSpec> toInputSpecs(
    const std::shared_ptr<tensorexpr::TensorExprKernel>& kernel) {
  // 获取图对象
  const std::shared_ptr<Graph>& g = kernel->graph();
  // 创建空的输入规范向量
  std::vector<mobile::nnc::InputSpec> specs;

  // 图的输入包括用于符号形状的标量值，这些不需要输入规范。
  // 这些标量值在图输入中的最后位置。
  auto num_inputs =
      g->inputs().size() - kernel->getSymbolicShapeInputs().size();

  // 遍历图的输入
  for (const auto i : c10::irange(num_inputs)) {
    auto v = g->inputs()[i];
    const auto& t = v->type();
    mobile::nnc::InputSpec spec;
    // 检查输入类型是否为 TensorType
    TORCH_CHECK(t->kind() == TypeKind::TensorType, "Unsupported input type");
    const auto& tt = t->cast<TensorType>();
    spec.sizes_ = {};
    // 获取张量的大小并存入 spec 中
    auto sizes_vec = *tt->sizes().sizes();
    for (auto s : sizes_vec) {
      spec.sizes_.push_back(s ? *s : 0);
    }
    // 获取张量的数据类型并存入 spec 中
    spec.dtype_ = *tt->scalarType();
    specs.emplace_back(std::move(spec));
  }
  return specs;
}

// 在输入形状中定位符号形状。
//
// 对于每个符号形状，我们尝试找到可以从中提取该符号形状的输入以及在该输入中的维度索引。
// 例如，如果我们有以下情况：
// graph(%x : Float(SS(-1), 10), %y : Long(20, SS(-2), %ss_1 : int, %ss_2 : int)
// 那么我们需要找到两个符号形状 SS(-1) 和 SS(-2) 的位置。
//
// SS(-2). The first one corresponds to the first dimension of the first input,
// the second one corresponds to the second dimension of the second input,
// so we will return {{0, 0}, {1, 1}}.
//
// If a symbolic shape cannot be found among dimensions of inputs, we
// will throw an error (this situation is possible when symbolic shape
// corresponds to the size of an intermediate - we don't support this
// case here yet).
//
// If a symbolic shape can be found in several different positions, we
// return the first one we find (TODO: maybe we should return all and
// verify that they all match at runtime).
static std::vector<SymbolicShapePosition> findSymbolicShapePositions(
    std::shared_ptr<tensorexpr::TensorExprKernel> kernel) {
  std::vector<SymbolicShapePosition> res;
  // Iterate over each symbolic index in the kernel's symbolic shape inputs
  for (int64_t sym_idx : kernel->getSymbolicShapeInputs()) {
    bool found = false;
    // Iterate over each input in the computational graph
    for (int64_t input_idx : c10::irange(kernel->graph()->inputs().size())) {
      auto input = kernel->graph()->inputs()[input_idx];

      // Skip inputs that are not tensors
      if (!input->type()->cast<TensorType>()) {
        continue;
      }
      auto tt = input->type()->expect<TensorType>();

      // Skip tensors without symbolic sizes
      if (!tt->symbolic_sizes().sizes()) {
        continue;
      }
      // Retrieve the vector of ShapeSymbols representing symbolic sizes
      std::vector<at::ShapeSymbol> shape_vec = *tt->symbolic_sizes().sizes();
      // Iterate over each dimension index in the shape vector
      for (int64_t dim_idx : c10::irange(shape_vec.size())) {
        // If the current dimension matches the symbolic index, record its position
        if (shape_vec[dim_idx].value() == sym_idx) {
          res.emplace_back(input_idx, dim_idx);
          found = true;
          break;
        }
      }
      // If a match is found for the current symbolic index, stop searching further
      if (found) {
        break;
      }
    }
    // Ensure that a symbolic shape was found for the current sym_idx
    TORCH_CHECK(
        found, "Could not locate a symbolic shape among input tensor shapes");
  }
  // Return the list of symbolic shape positions found
  return res;
}

static std::unique_ptr<Function> compileMethod(
    std::shared_ptr<tensorexpr::TensorExprKernel> kernel,
    const std::string& method_name,
    const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<at::ScalarType>& types) {
  auto func = std::make_unique<Function>();
  func->set_name(method_name);
  func->set_input_specs(toInputSpecs(kernel));

  auto params = c10::impl::GenericList(c10::AnyType::get());
  auto const_descriptors = kernel->getConstantDescriptors();
  // Iterate over each constant descriptor associated with the kernel
  for (const auto& cd : const_descriptors) {
    auto sizes = getConstSizes(cd.buf);
    if (!cd.node) {
      // Handle cases where constant tensor sizes are empty (scalar tensors)
      at::Tensor const_tensor = !sizes.empty()
          ? at::from_blob(cd.ptr, sizes).clone()
          : at::native::wrapped_scalar_tensor(*static_cast<double*>(cd.ptr));
      params.push_back(const_tensor);
    } else {
      // Convert the constant node's output to IValue and add to parameters
      params.emplace_back(toIValue(cd.node->output()));
    }
    // Add the parameters to the function
    func->set_params(std::move(params));
  }

  // Return the compiled function
  return func;
}
  }
}
// 调用函数对象的 set_parameters 方法，设置函数的参数
func->set_parameters(params);

MemoryPlan plan;
plan.buffer_sizes_ = {}; // temp_sizes_;
// TODO: 实现预分配优化并填充 temp_sizes
// 将 MemoryPlan 对象传递给函数对象，设置函数的内存计划
func->set_memory_plan(plan);

int64_t n_inputs = kernel->graph()->inputs().size();
int64_t n_outputs = kernel->graph()->outputs().size();
std::vector<OutputSpec> out_spec;
// 遍历输出规范，为每个输出设置大小和数据类型等信息
for (int64_t idx = n_inputs; idx < n_inputs + n_outputs; idx++) {
  const auto& ba = kernel->getBufferArgs()[idx];
  OutputSpec output;
  output.sizes_ = getConstSizes(ba.buf());
  // TODO: 断言输出是缓冲区而不是标量
  output.dtype_ = ba.buf()->dtype().scalar_type();
  if (isQIntType(output.dtype_)) {
    // 仅支持静态的量化比例和零点
    output.qscale_ =
        to<DoubleImm>(torch::jit::tensorexpr::IRSimplifier::simplify(
                          ba.buf()->qscale()))
            ->value();
    output.qzero_ =
        to<LongImm>(
            torch::jit::tensorexpr::IRSimplifier::simplify(ba.buf()->qzero()))
            ->value();
  }
  out_spec.push_back(output);
}
// 将输出规范设置到函数对象中
func->set_output_specs(out_spec);
// 设置函数对象的符号形状位置
func->set_sym_shape_positions(findSymbolicShapePositions(kernel));

// 返回设置好的函数对象
return func;
}

// 将指定方法编译为AOT（Ahead Of Time）代码，并返回生成的函数对象和编译后的汇编代码
static std::pair<std::unique_ptr<Function>, const std::string> aotCompile(
    const std::string& method_name,                    // 方法名称
    std::shared_ptr<Graph>& g,                         // 图对象的共享指针
    const std::vector<std::vector<int64_t>>& sizes,    // 输入张量的大小
    const std::vector<at::ScalarType>& types,          // 输入张量的数据类型
    const std::string& kernel_func_name,               // 内核函数名称
    const std::vector<int64_t>& symbolic_ind) {        // 符号索引
  GRAPH_DEBUG("Input sizes ", sizes);                  // 输出调试信息：输入大小
  GRAPH_DEBUG("Input types ", types);                  // 输出调试信息：输入类型
  GRAPH_DEBUG("Method name ", method_name);            // 输出调试信息：方法名称
  GRAPH_DEBUG("Kernel func name ", kernel_func_name);  // 输出调试信息：内核函数名称
  GRAPH_DEBUG("Symbolic indices ", symbolic_ind);      // 输出调试信息：符号索引

  std::shared_ptr<tensorexpr::TensorExprKernel> kernel;  // TensorExprKernel 对象的共享指针
  std::vector<torch::jit::StrideInput> stride_desc = {   // 步长描述数组初始化
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<                                  // 符号步长映射
      const torch::jit::Value*,                        // 键类型为 torch::jit::Value*
      std::vector<torch::jit::StrideInput>>            // 值类型为 std::vector<torch::jit::StrideInput>
      symbolic_strides;
  if (!symbolic_ind.empty()) {                         // 如果符号索引非空
    for (auto i : g->inputs()) {                       // 遍历图对象的输入
      symbolic_strides[i] = stride_desc;               // 设置输入的符号步长
    }
    for (auto o : g->outputs()) {                      // 遍历图对象的输出
      symbolic_strides[o] = stride_desc;               // 设置输出的符号步长
    }
  }
  // 创建 TensorExprKernel 对象，并初始化其属性
  kernel = std::make_shared<tensorexpr::TensorExprKernel>(TensorExprKernel(
      g, kernel_func_name, {}, symbolic_ind, false, symbolic_strides));

  const std::string compiled_assembly = kernel->getCodeText();  // 获取编译后的汇编代码
  auto func = compileMethod(kernel, method_name, sizes, types); // 编译方法
  return std::make_pair(std::move(func), compiled_assembly);    // 返回函数对象和编译后的汇编代码
}

// 将 LLVM 汇编代码写入指定文件
static void writeOutputLlvmAssembly(
    const std::string& asm_code,                // LLVM 汇编代码
    const std::string& output_llvm_file_name) { // 输出文件名
  std::ofstream output(output_llvm_file_name);  // 打开输出文件流
  output << asm_code;                           // 将汇编代码写入文件
  GRAPH_DEBUG(                                  // 输出调试信息：已保存编译后的 LLVM 汇编代码
      "The compiled llvm assembly code was saved to ", output_llvm_file_name);
}

// 使用指定分隔符分割字符串，并返回分割后的子串列表
static std::vector<std::string> split(
    char separator,                            // 分隔符
    const std::string& string,                 // 输入字符串
    bool ignore_empty = true) {                // 是否忽略空子串，默认为 true
  std::vector<std::string> pieces;             // 存储分割后的子串列表
  std::stringstream ss(string);                // 字符串流对象
  std::string item;                            // 存储每个子串的临时变量
  while (getline(ss, item, separator)) {        // 使用 getline() 函数按分隔符分割字符串
    if (!ignore_empty || !item.empty()) {       // 如果不忽略空子串或者当前子串不为空
      pieces.push_back(std::move(item));        // 将子串移动到列表中
    }
  }
  return pieces;                                // 返回分割后的子串列表
}

// 解析输入形状字符串，返回形状的二维数组
static std::vector<std::vector<int64_t>> parseInputShapes(
    const std::string& input_dims_s) {          // 输入形状字符串
  std::vector<std::string> input_dims_list = split(';', input_dims_s);  // 使用 ';' 分割输入形状字符串
  std::vector<std::vector<int64_t>> inputs;     // 存储解析后的形状二维数组
  for (const auto& input_dims_item : input_dims_list) {  // 遍历每个形状字符串
    auto input_dims_str = split(',', input_dims_item);   // 使用 ',' 分割当前形状字符串
    std::vector<int64_t> input_dims;           // 存储当前解析后的形状
    input_dims.reserve(input_dims_str.size()); // 预留空间以优化性能
    for (const auto& s : input_dims_str) {     // 遍历当前形状的每个维度
      input_dims.push_back(std::stoi(s));      // 将每个维度的字符串转换为整数并存储
    }
    inputs.push_back(input_dims);              // 将当前形状的整数数组存储到结果中
  }
  return inputs;                               // 返回解析后的形状二维数组
}

// 解析输入类型字符串，返回对应的 ScalarType 数组
static std::vector<at::ScalarType> parseInputTypes(
    const std::string& input_types_str) {      // 输入类型字符串
  std::vector<std::string> inputTypes = split(';', input_types_str);  // 使用 ';' 分割输入类型字符串
  std::vector<at::ScalarType> scalarTypes;     // 存储解析后的 ScalarType 数组
  for (const auto& inputType : inputTypes) {   // 遍历每个输入类型字符串
    at::ScalarType scalarType;                 // 存储当前解析后的 ScalarType
    if (inputType == "float") {                // 如果当前类型为 "float"
      scalarType = at::ScalarType::Float;      // 设置为 Float 类型

      scalarType = at::ScalarType::Float;      // 设置为 Float 类型
    // 可以根据需要添加其他常见的类型，如 int、double、bool 等
    // 注意：确保所有类型字符串与 Torch 的 ScalarType 枚举值匹配
    // 添加更多类型时，需要在此处添加相应的 if 分支来解析
    }
    // 将解析后的 ScalarType 添加到结果数组中
    scalarTypes.push_back(scalarType);
  }
  return scalarTypes;                          // 返回解析后的 ScalarType 数组
}
    } else if (inputType == "uint8") {
      // 如果输入类型为 "uint8"，则标量类型设为 Byte
      scalarType = at::ScalarType::Byte;
    } else if (inputType == "int64") {
      // 如果输入类型为 "int64"，则标量类型设为 Long
      scalarType = at::ScalarType::Long;
    } else {
      // 如果输入类型既不是 "uint8" 也不是 "int64"，抛出异常并显示不支持的输入类型
      CAFFE_THROW("Unsupported input type: ", inputType);
    }
    // 将确定的标量类型添加到标量类型数组中
    scalarTypes.push_back(scalarType);
  }
  // 返回所有输入类型对应的标量类型数组
  return scalarTypes;
// 静态方法：解析输入的内存格式字符串，返回内存格式枚举向量
static std::vector<at::MemoryFormat> parseInputMemoryFormats(
    const std::string& input_memory_format_str) {
  // 使用分号分割输入的内存格式字符串
  std::vector<std::string> memFormatsStr = split(';', input_memory_format_str);
  // 存储解析后的内存格式枚举向量
  std::vector<at::MemoryFormat> memFormats;
  // 遍历每个内存格式字符串
  for (const auto& memFormatStr : memFormatsStr) {
    at::MemoryFormat memFormat;
    // 根据字符串值设置对应的内存格式枚举
    if (memFormatStr == "contiguous") {
      memFormat = at::MemoryFormat::Contiguous;
    } else if (memFormatStr == "channels_last") {
      memFormat = at::MemoryFormat::ChannelsLast;
    } else {
      // 如果不支持的内存格式字符串，则抛出异常
      CAFFE_THROW("Unsupported memory format: ", memFormatStr);
    }
    // 将解析后的内存格式枚举添加到向量中
    memFormats.push_back(memFormat);
  }
  // 返回解析后的内存格式枚举向量
  return memFormats;
}

// 静态方法：解析输入的动态形状字符串，返回整数向量
static std::vector<int64_t> parseInputDynamicShapes(
    const std::string& dynamic_dims_s) {
  // 使用逗号分割输入的动态形状字符串
  std::vector<std::string> dynamic_dims_list = split(',', dynamic_dims_s);
  // 存储解析后的整数向量
  std::vector<int64_t> dynamic_dims;
  // 预先分配动态形状向量的空间
  dynamic_dims.reserve(dynamic_dims_list.size());
  // 遍历每个动态形状字符串
  for (const auto& dim : dynamic_dims_list) {
    // 将每个字符串转换为整数并添加到动态形状向量中
    dynamic_dims.push_back(std::stoi(dim));
  }
  // 返回解析后的动态形状整数向量
  return dynamic_dims;
}

// 静态方法：生成NNC内核ID字符串
static std::string getNncKernelId(
    const std::string& model_name,
    const std::string& model_version,
    const std::string& method_name) {
  // TODO: 计算版本令牌（暂未实现）
  const std::string version_token = "VERTOKEN";
  // 构建NNC内核ID字符串，包含模型名、模型版本、方法名和版本令牌
  return model_name + ":" + model_version + ":" + method_name + ":" +
      version_token;
}

// 静态方法：生成NNC内核函数名字符串
static std::string getNncKernelFuncName(
    const std::string& model_name,
    const std::string& model_version,
    const std::string& method_name) {
  // 构建NNC内核函数名字符串，包含固定前缀和模型相关信息
  return "nnc_" + model_name + "_" + model_version + "_" + method_name;
}

// 预处理图形并返回处理后的图形及其符号值（如果指定了动态输入形状）
static std::pair<std::shared_ptr<Graph>, std::vector<int64_t>>
preprocessGraphPasses(
    std::shared_ptr<Graph>& graph,
    const std::vector<std::optional<at::Tensor>>& example_inputs,
    const std::vector<int64_t>& dynamic_sizes) {
  // 调试输出：预处理图形前的图形状态
  GRAPH_DEBUG("Before preprocessing graph passes: ", *graph);
  // 移除图中的张量变异
  torch::jit::RemoveTensorMutation(graph);
  // 消除死代码
  torch::jit::EliminateDeadCode(graph->block());
  // 移除未使用的self参数
  graph = torch::jit::tensorexpr::removeUnusedSelfArgument(graph);

  // 标注输入形状
  torch::jit::tensorexpr::annotateInputShapes(graph, example_inputs);
  // 优化冻结图形
  torch::jit::OptimizeFrozenGraph(graph, true);
  // 在图上传播形状
  torch::jit::PropagateShapesOnGraph(graph);
  // 进行逐点优化
  torch::jit::PeepholeOptimize(graph, false);
  // 常数传播
  torch::jit::ConstantPropagation(graph);
  // 再次传播形状
  torch::jit::PropagateShapesOnGraph(graph);
  // 再次进行逐点优化
  torch::jit::PeepholeOptimize(graph, false);
  // 再次进行常数传播

  // 移除未使用的self参数（不明确的命名空间）
  tensorexpr::removeUnusedSelfArgument(graph);

  // 准备示例输入的IValue值
  std::vector<at::IValue> example_values;
  example_values.reserve(example_inputs.size());
  // 遍历每个示例输入
  for (auto example_input : example_inputs) {
    // 将 example_input 指向的值作为 vector 的最后一个元素添加到 example_values 中
    example_values.emplace_back(*example_input);
  }
  // 使用 example_values 对图进行跟踪，生成新的 TraceGraph 对象
  graph = TraceGraph(graph, example_values);
  // TODO: 当 TraceGraph 能够捕获输入形状时，移除 annotateInputShapes pass
  // 输入形状的注释 pass 暂时保留，待 TraceGraph 能够捕获输入形状后移除

  // 从 example_inputs 中标注输入形状到 TraceGraph
  tensorexpr::annotateInputShapes(graph, example_inputs);

  // 移除图中的列表变异
  RemoveListMutation(graph);
  // 移除图中张量的变异
  RemoveTensorMutation(graph);
  // 消除图中的死代码
  EliminateDeadCode(graph);
  // 将图中的所有元组降级
  LowerAllTuples(graph);

  // 将图中的形状符号化，使用 dynamic_sizes
  auto sym_val =
      torch::jit::tensorexpr::makeShapesSymbolic(graph, dynamic_sizes);

  // 调试输出，显示图经过预处理后的状态
  GRAPH_DEBUG("After preprocessing graph passes: ", *graph);
  // 返回包含图和符号化值的 pair 对象
  return std::make_pair(graph, sym_val);
static std::vector<std::optional<at::Tensor>> generateExampleInputs(
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<at::ScalarType>& inputTypes,
    const std::vector<at::MemoryFormat>& inputMemoryFormats) {
  // 创建一个空的可选张量向量，用于存储示例输入
  std::vector<std::optional<at::Tensor>> example_inputs;
  // 预留足够的空间以容纳输入形状的数量
  example_inputs.reserve(inputShapes.size());
  // 对输入形状列表进行迭代
  for (const auto i : c10::irange(inputShapes.size())) {
    // 获取当前输入的数据类型和内存格式
    const auto dtype = at::dtype(inputTypes[i]);
    const auto memory_format = inputMemoryFormats[i];
    // 生成随机数据张量，并确保其满足指定的数据类型和内存格式要求，然后加入到示例输入向量中
    example_inputs.emplace_back(
        at::rand(inputShapes[i]).to(dtype).contiguous(memory_format));
  }
  // 返回生成的示例输入向量
  return example_inputs;
}

static c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& compile_spec,
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {
  // 创建一个空的 NNCompiler 编译单元
  torch::jit::mobile::nnc::CompilationUnit cu;
  // 对编译规范中的每个键值对进行迭代处理
  for (const auto& kv : compile_spec) {
    // 输出调试信息，显示当前处理的键和对应的值
    GRAPH_DEBUG("Key: ", kv.key());
    GRAPH_DEBUG("Value: ", kv.value());
    // 从键中获取方法名字符串
    std::string method_name = *(kv.key().toString());
    GRAPH_DEBUG("Method name: ", method_name);
    // 将值解析为泛型字典
    auto method_spec = kv.value().toGenericDict();
    // 从方法规范中获取模型名称、模型版本和汇编文件名
    std::string model_name = *method_spec.at("model_name").toString();
    std::string model_version = *method_spec.at("model_version").toString();
    std::string asmfile_name = *method_spec.at("asmfile").toString();
    GRAPH_DEBUG("Model name: ", model_name);
    GRAPH_DEBUG("Model version: ", model_version);
    GRAPH_DEBUG("Asm file name: ", asmfile_name);

    // 获取模块中指定方法的函数对象
    auto method = mod.get_method(method_name);
    // 复制方法的图形表示
    auto graph = toGraphFunction(method.function()).graph()->copy();

    // 解析输入形状、类型和动态形状
    auto sizes = parseInputShapes(*method_spec.at("sizes").toString());
    auto types = parseInputTypes(*method_spec.at("types").toString());
    auto dynamic_sizes =
        parseInputDynamicShapes(*method_spec.at("dynamic_sizes").toString());

    // 解析内存格式字符串，如果为空则使用默认的连续内存格式
    std::string memory_formats_str = method_spec.contains("memory_formats")
        ? (*method_spec.at("memory_formats").toString()).string()
        : "";
    auto memory_formats = memory_formats_str.empty()
        ? std::vector<at::MemoryFormat>(
              sizes.size(), at::MemoryFormat::Contiguous)
        : parseInputMemoryFormats(memory_formats_str);

    // 生成示例输入
    auto example_inputs = generateExampleInputs(sizes, types, memory_formats);
    // 对图进行预处理，以便后续编译
    auto preprocessed =
        preprocessGraphPasses(graph, example_inputs, dynamic_sizes);

    // 获取 NNCompiler 内核函数的名称
    auto kernel_func_name =
        getNncKernelFuncName(model_name, model_version, method_name);
    // 获取处理后的图形和符号值
    auto processed_graph = preprocessed.first;
    auto sym_values = preprocessed.second;
    // 编译处理后的图形为 AOT 形式
    auto compiled = torch::jit::mobile::nnc::aotCompile(
        method_name,
        processed_graph,
        sizes,
        types,
        kernel_func_name,
        sym_values);
    // 将编译结果写入 LLVM 汇编文件
    writeOutputLlvmAssembly(compiled.second, asmfile_name);
    // 获取编译后的函数对象，并设置其 NNCompiler 内核 ID
    auto func = std::move(compiled.first);
    func->set_nnc_kernel_id(
        getNncKernelId(model_name, model_version, method_name));
    cu.register_function(std::move(func));
  }
  return cu.serialize();



    # 将 func 移动到 cu 对象中注册的函数列表中
    cu.register_function(std::move(func));
  }
  # 序列化 cu 对象并返回序列化后的结果
  return cu.serialize();
// 结束命名空间 torch
} // namespace torch

// 结束命名空间 jit
} // namespace jit

// 结束命名空间 mobile
} // namespace mobile

// 结束命名空间 nnc
} // namespace nnc

// 静态变量 reg，用于注册后端预处理函数，将 "nnc" 与 preprocess 关联起来
static auto reg = torch::jit::backend_preprocess_register("nnc", preprocess);
```