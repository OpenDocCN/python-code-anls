# `.\pytorch\torch\csrc\jit\passes\quantization\helper.cpp`

```
# 包含 Torch 库中的量化辅助函数头文件
#include <torch/csrc/jit/passes/quantization/helper.h>

# 包含 Torch 中的函数实现接口头文件
#include <torch/csrc/jit/api/function_impl.h>
# 包含 Torch 中的图重写辅助函数头文件
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>

# 引入标准库中的实用工具
#include <utility>

# 定义 torch 命名空间下的 jit 命名空间
namespace torch {
namespace jit {

# 使用 graph_rewrite_helper 命名空间中的函数 getFuncName
using graph_rewrite_helper::getFuncName;

# 定义结构体 FuncArg，用于表示函数参数
struct FuncArg {
  std::string func_name;  // 函数名
  int arg_index;          // 参数索引
};

# 定义类型别名 AtenFuncArgs 和 CallFuncArgs，分别表示 Atene 和 Call 函数参数的向量
using AtenFuncArgs = std::vector<FuncArg>;
using CallFuncArgs = std::vector<FuncArg>;

# 定义静态量化可用的 Call 函数列表
std::vector<std::string> _static_quantizable_call_funcs = {
    "conv2d",
    "linear",
    "batch_norm",
    "hardswish",
    "elu",
    "celu",
    "layer_norm",
    "group_norm",
    "instance_norm",
    "embedding_bag",
};

# 定义静态量化可用的 Atene 函数列表
std::vector<std::string> _static_quantizable_aten_funcs = {
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "linear",
    "hardswish",
    "hardswish_",
    "elu",
    "elu_",
    "celu",
    "celu_",
    "batch_norm",
    "layer_norm",
    "group_norm",
    "instance_norm",
    "embedding_bag",
};

# 定义动态量化可用的 Call 函数列表
std::vector<std::string> _dynamic_quantizable_call_funcs = {
    "linear",
};

# 定义动态量化可用的 Atene 函数列表
std::vector<std::string> _dynamic_quantizable_aten_funcs = {
    "linear",
};

# 定义仅权重量化静态 Atene 函数列表
std::vector<std::string> _static_weight_only_quant_aten_funcs = {
    "embedding_bag",
};

# 定义仅权重量化静态 Call 函数列表
std::vector<std::string> _static_weight_only_quant_call_funcs = {
    "embedding_bag",
};

# 下面是 prim::CallFunctions 的列表，这些函数不需要观察，并且只有一个输入张量
# 例如：`prim::CallFunction(%dropout, %input_tensor, ...)`
# 因此我们将观察到的属性从 %input_tensor 传播到 `prim::CallFunction` 的输出
# 这些操作不会在张量值上执行计算，操作仅依赖于张量的形状
std::vector<std::string> _single_input_general_shape_call_funcs = {
    "_max_pool1d",
    "_max_pool2d",
    "_max_pool3d",
    "dropout",
    "relu",
};

# 类似于 prim::CallFunctions，这里有一些 Atene 操作，它们不需要观察并且只有一个输入张量
# 这些操作也不会在张量值上执行计算，操作仅依赖于张量的形状
# 例如：`aten::flatten(%input_tensor, ...)`
std::vector<std::string> _single_input_general_shape_aten_funcs = {
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "flatten",
    "max",
    "min",
    "dropout",
    "reshape",
    // Non-inplace resize is deprecated
    "resize_",
    "chunk",
    "view",
    "transpose",
    "contiguous",
    "permute",
    "repeat",
    "repeat_interleave",
    "relu",
    "relu_",
    "squeeze",
    "squeeze_",
    "unsqueeze",
    "unsqueeze_",
    "detach",
    "detach_",
    "stack",
    "__getitem__",
};

# 这些是 prim::CallFunctions 的列表，适用于不需要观察且只有一个输入张量的操作
# 这些操作会在张量值上执行计算
# TODO: [需要验证] 看起来我们可以量化只调用 Atene 函数的简单功能函数
// 存储一组字符串，这些字符串表示不需要观察器的操作函数名称，这些函数仅接受单个输入张量并在其值上执行计算
std::vector<std::string> _single_input_general_value_call_funcs = {
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "interpolate",
    "upsample",
    "upsample_bilinear",
    "upsample_nearest",
    "hardtanh",
    "leaky_relu",
};

// 存储一组字符串，这些字符串表示不需要观察器的 ATen 函数名称，这些函数仅接受单个输入张量并在其值上执行计算。
// 此外，这些操作函数的名称是 ATen（PyTorch 的张量库）函数的一部分。
std::vector<std::string> _single_input_general_value_aten_funcs = {
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "mean",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "upsample_linear1d",
    "upsample_bilinear2d",
    "upsample_trilinear3d",
    "upsample_bicubic2d",
    "clamp",
    // "clamp_",  // 当量化的 `clamp_` 就绪时启用
    "hardtanh",
    "hardtanh_",
    "leaky_relu",
    "leaky_relu_",
};

// 存储一组字符串，这些字符串表示需要观察器的操作函数名称，这些操作函数的名称是 ATen（PyTorch 的张量库）函数的一部分。
std::vector<std::string> _clamp_funcs = {
    "hardtanh",
    "hardtanh_",
    "clamp",
    // "clamp_",  // 当量化的 `clamp_` 就绪时启用
};

// 定义常量，表示无符号八位整数的比例尺度，用于量化参数
const float _asym_scale = 1.0f / 256.0f;
// 定义常量，表示无符号八位整数的零点偏移，用于量化参数
const int _asym_zero_point = 0;
// 定义常量，表示对称量化的比例尺度，用于量化参数
const float _sym_scale = 2.0f / 256.0f;
// 定义常量，表示对称量化的零点偏移，用于量化参数
const int _sym_zero_point = 128;

// 表示一种量化参数的元组，用于 ops 的范围为 0 到 1
// 例如：aten/src/ATen/native/quantized/cpu/qsigmoid.cpp
std::tuple<c10::QScheme, QParamVector> _per_tensor_asym_qparam =
    std::make_tuple(
        c10::kPerTensorAffine,
        QParamVector(
            {std::make_pair(".scale", IValue(_asym_scale)),
             std::make_pair(".zero_point", IValue(_asym_zero_point)),
             std::make_pair(".scalar_type", IValue(c10::kQUInt8))}));

// 表示一种量化参数的元组，用于 ops 的范围为 -1 到 1
// 例如：aten/src/ATen/native/quantized/cpu/qtanh.cpp
std::tuple<c10::QScheme, QParamVector> _per_tensor_sym_qparam = std::make_tuple(
    c10::kPerTensorAffine,
    QParamVector(
        {std::make_pair(".scale", IValue(_sym_scale)),
         std::make_pair(".zero_point", IValue(_sym_zero_point)),
         std::make_pair(".scalar_type", IValue(c10::kQUInt8))}));

// 将 ATen 操作符符号映射到量化参数的无序映射
std::unordered_map<NodeKind, std::tuple<c10::QScheme, QParamVector>>
    _fixed_qparams_map = {
        {Symbol::aten("hardsigmoid"), _per_tensor_asym_qparam},
        {Symbol::aten("hardsigmoid_"), _per_tensor_asym_qparam},
        {Symbol::aten("sigmoid"), _per_tensor_asym_qparam},
        {Symbol::aten("sigmoid_"), _per_tensor_asym_qparam},
        {Symbol::aten("tanh"), _per_tensor_sym_qparam},
        {Symbol::aten("tanh_"), _per_tensor_sym_qparam},
};

// 用于不需要为所有输入张量插入观察器的特殊检查的操作符列表。
// 对于此列表中的每个操作符，观察器基于输入被插入。
// 在指定的索引处观察输入的 Atan 函数参数
AtenFuncArgs _observe_inputs_aten_func = {};

// 在调用函数中观察输入的 CallFunc 参数，例如 {"batch_norm", 1}
CallFuncArgs _observe_inputs_call_func = {{"batch_norm", 1}};

// 获取张量信息的 Atan 函数列表
std::vector<std::string> _tensor_info_funcs = {"size", "len", "dim", "numel"};

// 根据输入张量决定输出是否量化的 Atan 函数列表
std::vector<std::string> _propagate_quant_single_input_ops = {"cat"};

// 对于二元操作如 `aten::add`，规则略有不同：
// 如果两个输入都是张量，则只有当两个输入都量化时才量化输出；
// 如果第二个输入是标量，则只检查第一个输入来决定是否量化输出。
std::vector<std::string> _propagate_quant_binary_ops = {
    "add",
    "add_",
    "mul",
    "mul_"};

// 检查 `use` 是否是名称为 `func_name` 的 Atan 函数，并且值 `v` 是否是函数的第 n 个参数（如果提供了）
bool matchAtenFuncToUse(
    const Use& use,
    const std::string& func_name,
    std::optional<int> n) {
  Node* node = use.user;
  return node->kind() == Symbol::aten(func_name) &&
      (!n.has_value() || static_cast<size_t>(n.value()) == use.offset);
}

// 检查 `use` 是否是调用函数模式的函数，并且函数名称为 `func_name`，值 `v` 是否是函数的第 n 个参数（如果提供了）
bool matchCallFuncToUse(
    const Use& use,
    const std::string& func_name,
    std::optional<int> n) {
  Node* node = use.user;
  return node->kind() == prim::CallFunction &&
      getFuncName(node->inputs()[0]) == func_name &&
      (!n.has_value() || static_cast<size_t>(n.value()) == use.offset);
}

// 检查值 `v` 的任何使用是否匹配 Atan 函数调用或 CallFunction 模式
static bool matchArgPattern(
    Value* v,
    const AtenFuncArgs& aten_func_args,
    const CallFuncArgs& call_func_args) {
  for (const Use& u : v->uses()) {
    for (const auto& func_arg : aten_func_args) {
      if (matchAtenFuncToUse(u, func_arg.func_name, func_arg.arg_index)) {
        return true;
      }
    }

    for (const auto& func_arg : call_func_args) {
      if (matchCallFuncToUse(u, func_arg.func_name, func_arg.arg_index)) {
        return true;
      }
    }
  }
  return false;
}

// 判断值 `v` 是否为权重
bool isWeight(Value* v) {
  bool result = matchArgPattern(
      v,
      // 对于如下 Atan 函数，确定权重：conv1d, conv2d, conv3d, conv_transpose1d,
      // conv_transpose2d, linear, embedding_bag
      AtenFuncArgs(
          {{"conv1d", 1},
           {"conv2d", 1},
           {"conv3d", 1},
           {"conv_transpose1d", 1},
           {"conv_transpose2d", 1},
           {"linear", 1},
           {"embedding_bag", 0}}),
      // 对于如下 CallFunction，确定权重：linear, embedding_bag
      CallFuncArgs({{"linear", 2}, {"embedding_bag", 2}}));
  return result;
}
// 检查给定值是否符合卷积或线性层函数的模式
bool isBiasOfConvOrLinear(Value* v) {
  // 调用 matchArgPattern 函数，匹配函数名和参数模式，判断是否匹配卷积或线性层的 bias 参数
  bool result = matchArgPattern(
      v,
      AtenFuncArgs(
          {{"conv1d", 2},
           {"conv2d", 2},
           {"conv3d", 2},
           {"conv_transpose1d", 2},
           {"conv_transpose2d", 2},
           {"linear", 2}}),
      CallFuncArgs({{"linear", 3}}));
  return result;
}

// 检查给定值是否符合 embedding_bag 函数的非输入模式
bool isEmbeddingBagNonInput(Value* v) {
  // 调用 matchArgPattern 函数，匹配函数名和参数模式，判断是否匹配 embedding_bag 函数的非输入模式
  bool result = matchArgPattern(
      v,
      AtenFuncArgs({{"embedding_bag", 2}, {"embedding_bag", 6}}),
      CallFuncArgs({}));
  return result;
}

// 获取给定值的 clamp 函数标量输入的使用情况
std::optional<Use> getClampScalarInputUse(Value* v) {
  // 遍历给定值的使用情况，匹配 _clamp_funcs 中的 clamp 函数，并返回匹配到的使用情况
  for (const auto& use : v->uses()) {
    for (const auto& aten_func : _clamp_funcs) {
      if (matchAtenFuncToUse(use, aten_func, 1) ||
          matchAtenFuncToUse(use, aten_func, 2)) {
        return use;
      }
    }
  }
  return c10::nullopt;
}

// 在模块中克隆指定方法，并命名为新的方法名
void cloneMethod(
    Module& module,
    const std::string& orig_method_name,
    const std::string& new_method_name) {
  // 获取原始方法的函数对象和图表达式，进行复制操作
  const Function& method = module.get_method(orig_method_name).function();
  auto graph = toGraphFunction(method).graph()->copy();
  const auto& schema = method.getSchema();
  const auto this_method_name =
      c10::QualifiedName(*module.type()->name(), new_method_name);
  // 创建新的函数对象并添加到模块中
  auto copied = module._ivalue()->compilation_unit()->create_function(
      this_method_name, std::move(graph));
  module.type()->addMethod(copied);
  copied->setSchema(schema);
}

// 获取给定值的传递输入节点
std::vector<Value*> getPassThroughInputs(Value* v) {
  Node* n = v->node();
  if (isSingleInputGeneralCallFunction(n)) {
    // 如果节点是通用调用函数且只有一个输入，返回该输入
    return {n->input(1)};
  } else if (
      isSingleInputGeneralAtenFunction(n) ||
      (n->kind() == Symbol::aten("sort") && v->offset() == 0)) {
    // 如果节点是通用 ATen 函数或 sort 函数的第一个输入，返回第一个输入
    return {n->input(0)};
  } else if (n->kind() == prim::If && n->outputs().size() == 1) {
    // 如果节点是 if 语句且只有一个输出，返回所有子块的第一个输出
    std::vector<Value*> inputs;
    for (Block* subblock : n->blocks()) {
      if (alwaysRaisesException(subblock)) {
        continue;
      }
      auto* output = subblock->outputs()[0];
      inputs.push_back(output);
    }
    return inputs;
  } else if (n->kind() == prim::ListUnpack || n->kind() == prim::TupleUnpack) {
    // 如果节点是列表解包或元组解包，只在类型为 Tensor 时传播
    if (v->type()->isSubtypeOf(*TensorType::get())) {
      return {n->input(0)};
    } else {
      return {};
    }
  } else if (
      n->kind() == prim::ListConstruct &&
      v->type()->isSubtypeOf(*ListType::ofTensors())) {
    // 如果节点是列表构造且类型为 Tensor 列表，返回所有输入
    std::vector<Value*> inputs;
    for (auto* v : n->inputs()) {
      inputs.push_back(v);
    }
    return inputs;
  } else if (n->kind() == prim::TupleConstruct) {
    // 如果节点是元组构造，只返回类型为 Tensor 的输入
    std::vector<Value*> inputs;
    for (auto* input : n->inputs()) {
      if (input->type()->isSubtypeOf(*TensorType::get())) {
        inputs.push_back(input);
      }
    }
    return inputs;
  } else if (n->kind() == Symbol::aten("append")) {
    // 如果节点是 append 函数，返回所有输入
    std::vector<Value*> inputs;
    for (auto* input : n->inputs()) {
      inputs.push_back(input);
    }
    return inputs;
  }

  // 默认情况下返回空的输入向量
  return {};
}
    # 接受一个常量引用的字符串向量 func_names 作为参数，并返回一个 NodeKind 类型的向量
    const std::vector<NodeKind> & func_names) {
        # 创建一个空的 NodeKind 类型的向量 symbols，用于存储转换后的结果
        std::vector<NodeKind> symbols;
        # 使用 std::transform 函数，对 func_names 中的每个元素执行转换操作：
        # - func_names.begin() 开始迭代器
        # - func_names.end() 结束迭代器
        # - std::back_inserter(symbols) 将结果插入到 symbols 的尾部
        # - Symbol::aten 是转换函数，用于将每个字符串转换为 NodeKind 类型
        std::transform(
            func_names.begin(),
            func_names.end(),
            std::back_inserter(symbols),
            Symbol::aten);
        # 返回转换后的 symbols 向量
        return symbols;
    }
static bool isAtenFunc(Node* n, const std::vector<NodeKind>& aten_funcs) {
  // 检查给定节点的类型是否在 ATen 函数列表中
  return std::find(aten_funcs.begin(), aten_funcs.end(), n->kind()) !=
      aten_funcs.end();
}

static bool isAtenFunc(Node* n, const std::vector<std::string>& aten_funcs) {
  // 将字符串类型的 ATen 函数转换为符号类型，然后检查给定节点是否在符号列表中
  const auto& symbols = toAtenSymbol(aten_funcs);
  return isAtenFunc(n, symbols);
}

// TODO: factor out isCallFunc
static bool isFunctionNode(
    Node* n,
    const std::vector<std::string>& call_funcs,
    const std::vector<std::string>& aten_funcs) {
  // 检查节点是否为函数节点，可能是 ATen 函数或者特定的调用函数
  bool is_func_node = isAtenFunc(n, aten_funcs);
  if (n->kind() == prim::CallFunction) {
    auto func_name = getFuncName(n->inputs()[0]);
    // 如果是调用函数节点，检查调用的函数名是否在调用函数列表中
    is_func_node |=
        std::find(call_funcs.begin(), call_funcs.end(), func_name) !=
        call_funcs.end();
  }
  return is_func_node;
}

bool isSingleInputGeneralShapeAtenFunction(Node* n) {
  // 检查节点是否为单输入通用形状 ATen 函数
  return isAtenFunc(n, _single_input_general_shape_aten_funcs);
}

bool isSingleInputGeneralValueAtenFunction(Node* n) {
  // 检查节点是否为单输入通用值 ATen 函数或者是具有标量输入的二元操作
  return isAtenFunc(n, _single_input_general_value_aten_funcs) ||
      isBinaryOpWithScalarInput(n);
}

bool isSingleInputGeneralCallFunction(Node* n) {
  static std::vector<std::string> single_input_general_call_funcs;
  // 将通用形状和通用值的调用函数列表合并
  std::copy(
      _single_input_general_shape_call_funcs.begin(),
      _single_input_general_shape_call_funcs.end(),
      std::back_inserter(single_input_general_call_funcs));
  std::copy(
      _single_input_general_value_call_funcs.begin(),
      _single_input_general_value_call_funcs.end(),
      std::back_inserter(single_input_general_call_funcs));
  // 检查节点是否为单输入通用调用函数
  return isFunctionNode(
      n,
      /* call_funcs = */ single_input_general_call_funcs,
      /* aten_funcs = */ {});
}

bool isSingleInputGeneralAtenFunction(Node* n) {
  static std::vector<NodeKind> fixed_qparams_aten_funcs;
  // 转换固定量化参数映射为节点类型列表
  std::transform(
      _fixed_qparams_map.begin(),
      _fixed_qparams_map.end(),
      std::back_inserter(fixed_qparams_aten_funcs),
      [](auto pair) { return pair.first; });

  // 检查节点是否为单输入通用 ATen 函数，包括通用值、通用形状和固定量化参数 ATen 函数
  return isSingleInputGeneralValueAtenFunction(n) ||
      isSingleInputGeneralShapeAtenFunction(n) ||
      isAtenFunc(n, fixed_qparams_aten_funcs);
}

bool isClamp(Node* n) {
  // 检查节点是否为 Clamp 函数
  return isAtenFunc(n, _clamp_funcs);
}

bool isTensorInfoNode(Node* n) {
  // 检查节点是否为 Tensor 信息节点
  return isAtenFunc(n, _tensor_info_funcs);
}

bool isPropagateQuantSingleInputOp(Node* n) {
  // 检查节点是否为传播量化的单输入操作节点
  return isAtenFunc(n, _propagate_quant_single_input_ops);
}

bool isPropagateQuantBinaryOp(Node* n) {
  // 检查节点是否为传播量化的二元操作节点
  return isAtenFunc(n, _propagate_quant_binary_ops);
}

bool isPropagateQuantOp(Node* n) {
  // 检查节点是否为传播量化的操作节点（包括单输入和二元操作）
  return isPropagateQuantSingleInputOp(n) || isPropagateQuantBinaryOp(n);
}

bool isBinaryOpWithScalarInput(Node* n) {
  // 检查节点是否为具有标量输入的二元操作
  return isPropagateQuantBinaryOp(n) && isScalar(n->input(1));
}
// 获取节点固定的量化参数，返回一个可选的元组，包含量化方案和量化参数向量
std::optional<std::tuple<c10::QScheme, QParamVector>> getFixedQParams(Node* n) {
  // 静态变量，存储固定量化参数函数的节点类型
  static std::vector<NodeKind> fixed_qparam_funcs;
  // 从_fixed_qparams_map的键中提取节点类型，填充到fixed_qparam_funcs中
  std::transform(
      _fixed_qparams_map.begin(),
      _fixed_qparams_map.end(),
      std::back_inserter(fixed_qparam_funcs),
      [](const auto& pair) { return pair.first; });
  // 检查节点是否为Aten函数，并返回其对应的固定量化参数（如果有）
  if (isAtenFunc(n, fixed_qparam_funcs)) {
    return _fixed_qparams_map.at(n->kind());
  }
  // 没有找到匹配的固定量化参数，返回空值
  return c10::nullopt;
}

// 检查节点是否为用户定义的调用函数
bool userDefinedCallFunction(Node* n) {
  return n->kind() == prim::CallFunction &&
      !isSingleInputGeneralCallFunction(n) &&
      !isFunctionNode(n, _static_quantizable_call_funcs, {});
}

// 检查节点是否为仅包含权重的静态量化操作
bool isWeightOnlyStaticQuantOp(Node* n) {
  return isFunctionNode(
      n,
      _static_weight_only_quant_call_funcs,
      _static_weight_only_quant_aten_funcs);
}

// 检查节点是否可以量化，根据量化类型（动态或静态）
bool nodeQuantizable(Node* n, QuantType quant_type) {
  bool is_dynamic = quant_type == QuantType::DYNAMIC;
  return isFunctionNode(
      n,
      /* call_funcs = */
      is_dynamic ? _dynamic_quantizable_call_funcs
                 : _static_quantizable_call_funcs,
      /* aten_funcs = */
      is_dynamic ? _dynamic_quantizable_aten_funcs
                 : _static_quantizable_aten_funcs);
}

// 检查使用是否可量化，根据量化类型（静态或动态）
bool useQuantizable(const Use& use, QuantType quant_type) {
  if (quant_type == QuantType::STATIC) {
    // 遍历静态观察输入的Aten函数，检查使用是否匹配
    for (const auto& func_input : _observe_inputs_aten_func) {
      if (matchAtenFuncToUse(use, func_input.func_name, c10::nullopt)) {
        return use.offset == static_cast<size_t>(func_input.arg_index);
      }
    }
    // 遍历静态观察输入的调用函数，检查使用是否匹配
    for (const auto& func_input : _observe_inputs_call_func) {
      if (matchCallFuncToUse(use, func_input.func_name, c10::nullopt)) {
        return use.offset == static_cast<size_t>(func_input.arg_index);
      }
    }
  }
  // 调用nodeQuantizable检查使用的用户节点是否可量化
  return nodeQuantizable(use.user, quant_type);
}

// 获取调用函数节点的图形对象
std::shared_ptr<Graph> getCallFunctionGraph(Node* n) {
  auto* func_node = n->input(0)->node();
  auto func = func_node->output()->type()->expectRef<FunctionType>().function();
  auto graphFunc = tryToGraphFunction(*func);
  // 检查图形函数是否有效，若无效则抛出错误
  TORCH_CHECK(graphFunc, "Quantization only works for graph function");
  return graphFunc->graph();
}

// 块辅助函数
// 检查块是否总是引发异常
bool alwaysRaisesException(Block* block) {
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::RaiseException) {
      return true;
    }
    if (n->kind() == prim::If) {
      bool exception = true;
      for (Block* b : n->blocks()) {
        exception &= alwaysRaisesException(b);
      }
      if (exception) {
        return true;
      }
    }
  }
  return false;
}

// 检查图中的值是否为标量值
bool isScalar(Value* v) {
  auto iv = toIValue(v);
  return v->type()->isSubtypeOf(*NumberType::get()) ||
      (v->type()->isSubtypeOf(*TensorType::get()) && iv && iv->isTensor() &&
       iv->toTensor().dim() == 0);
}
// 判断给定的值是否是图的输入
bool hitGraphInput(Value* value) {
  // 获取值所属的图对象
  Graph* graph = value->owningGraph();
  // 获取图的输入列表
  const auto& inputs = graph->inputs();
  // 判断给定的值是否在图的输入列表中
  return std::find(inputs.begin(), inputs.end(), value) != inputs.end();
}

// 根据表示模块实例的值，追溯 GetAttr 节点并记录路径中所有属性名
// 假设 'self.sub.basic_block.conv1',
// Input1: conv1 的值实例
// Input2: self 的值实例
// Output: ['sub', 'basic_block', 'conv1']
std::vector<std::string> getModuleAccessPath(Value* instance, Value* self) {
  std::vector<std::string> path;
  // 用于迭代回溯 GetAttr 调用
  Value* iter = instance;
  // 追溯实例以恢复子模块的路径
  while (!hitGraphInput(iter) && iter->node()->kind() == prim::GetAttr) {
    Node* get_attr = iter->node();
    // 记录 GetAttr 的名称
    path.push_back(get_attr->s(attr::name));
    // 回溯 GetAttr 链
    iter = get_attr->inputs()[0];
  }
  // 检查是否回溯到了 self
  TORCH_CHECK(
      iter == self,
      "Can't handle the access pattern of GetAttr "
      " in getModuleAccessPath, traced back to:",
      iter->debugName(),
      " which is not self:",
      self->debugName());
  // 反转路径顺序以得到正确的模块访问路径
  std::reverse(path.begin(), path.end());
  return path;
}

// 假设 self.foo.bar.conv1,
// Input1: self 的模块实例
// Input2: ['foo', 'bar', 'conv1']
// Output: conv1 的模块实例
Module findChildModule(
    const Module& module,
    const std::vector<std::string>& path) {
  Module m = module;
  // 根据路径访问子模块
  for (const auto& p : path) {
    m = m.attr(p).toModule();
  }
  return m;
}

// 获取被调用的模块
Module getInvokedModule(Module& module, Node* n, Value* self) {
  auto* instance = n->inputs()[0];
  auto path = getModuleAccessPath(instance, self);
  return findChildModule(module, path);
}

// 获取被调用的模块，返回一个可选的模块实例
std::optional<Module> getInvokedModuleOpt(
    const Module& module,
    Node* n,
    Value* self) {
  auto* instance = n->inputs()[0];
  auto path = getModuleAccessPath(instance, self);
  Module m = module;
  // 根据路径迭代访问子模块，如果路径不存在，则返回空
  for (const auto& p : path) {
    if (m.attr(p).isModule()) {
      m = m.attr(p).toModule();
    } else {
      return c10::nullopt;
    }
  }
  return m;
}

// ==================== 匹配过滤函数 ==============

// 判断匹配中给定的值是否是整数常量，并且其值等于指定的整数
bool is_int_constant(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap,
    const std::string& vname,
    int value) {
  const auto& match_vmap = match.values_map;
  // 获取匹配中给定名称对应的值，转换为整数类型，检查是否符合指定值
  auto v = toIValue(match_vmap.at(vmap.at(vname)));
  return v && v->isInt() && v->toInt() == value;
}

// 判断匹配中给定的值是否是函数类型，并且其函数名等于指定的函数名
static bool is_functional(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap,
    const std::string& vname,
    const std::string& functional) {
  const auto& match_vmap = match.values_map;
  // 获取匹配中给定名称对应的值，检查其类型是否为函数类型，并且函数名是否匹配
  Value* v = match_vmap.at(vmap.at(vname));
  return v->type()->cast<FunctionType>() && getFuncName(v) == functional;
}
# 移除 Torch 引擎生成的名称修饰，返回原始名称
std::string removeTorchMangle(const std::string& orig_name) {
  // 定义用于匹配 Torch 引擎生成名称的正则表达式
  static std::regex mangle_re("\\.___torch_mangle_\\d+");
  // 替换原始名称中匹配的 Torch 引擎生成名称为 ""
  auto qualified_name = std::regex_replace(orig_name, mangle_re, "");
  // 返回处理后的原始名称
  return qualified_name;
}

// 获取给定值的模块名称，如果存在的话
std::optional<std::string> getModuleName(Value* value) {
  // 尝试将值转换为 ClassType 类型
  auto type = value->type()->cast<ClassType>();
  // 如果类型有效且有名称
  if (type && type->name()) {
    // 返回移除 Torch 引擎生成名称后的限定名称
    return removeTorchMangle(type->name()->qualifiedName());
  }
  // 否则返回空的可选字符串
  return c10::nullopt;
}

// 检查给定匹配是否为指定的模块
static bool is_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap,
    const std::string& vname,
    const std::string& module_qualified_name) {
  // 获取匹配中的值映射
  const auto& match_vmap = match.values_map;
  // 获取与给定名称相关联的值
  Value* v = match_vmap.at(vmap.at(vname));
  // 获取值对应的模块名称
  auto module_name = getModuleName(v);
  // 如果成功获取模块名称
  if (module_name.has_value()) {
    // 比较模块名称与给定的模块限定名称是否相等
    return module_name.value() == module_qualified_name;
  }
  // 如果未获取到模块名称，则返回 false
  return false;
};

// 检查匹配是否为 alpha 参数为 1 的整数常量
bool aten_add_alpha_is_one(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 调用 is_int_constant 函数检查匹配是否为整数常量 alpha 为 1
  return is_int_constant(match, vmap, "alpha", 1);
}

// 检查匹配是否为功能性 ReLU 函数
bool is_functional_relu(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 调用 is_functional 函数检查匹配是否为功能性模块 relu
  return is_functional(match, vmap, "relu", "relu");
}

// 检查匹配是否为 ReLU 激活函数模块
bool is_relu_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 调用 is_module 函数检查匹配是否为 ReLU 模块
  return is_module(
      match, vmap, "relu", "__torch__.torch.nn.modules.activation.ReLU");
}

// 检查匹配是否为线性层模块
bool is_linear_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 调用 is_module 函数检查匹配是否为线性层模块
  return is_module(
      match, vmap, "linear", "__torch__.torch.nn.modules.linear.Linear");
}

// 检查匹配是否为 1D 卷积层模块
bool is_conv1d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 调用 is_module 函数检查匹配是否为 1D 卷积层模块
  return is_module(
      match, vmap, "conv", "__torch__.torch.nn.modules.conv.Conv1d");
}

// 检查匹配是否为 2D 卷积层模块
bool is_conv2d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 调用 is_module 函数检查匹配是否为 2D 卷积层模块
  return is_module(
      match, vmap, "conv", "__torch__.torch.nn.modules.conv.Conv2d");
}

// 检查匹配是否为 3D 卷积层模块
bool is_conv3d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 调用 is_module 函数检查匹配是否为 3D 卷积层模块
  return is_module(
      match, vmap, "conv", "__torch__.torch.nn.modules.conv.Conv3d");
}

// 检查匹配是否为 1D 转置卷积层模块
bool is_conv_transpose1d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 调用 is_module 函数检查匹配是否为 1D 转置卷积层模块
  return is_module(
      match, vmap, "conv", "__torch__.torch.nn.modules.conv.ConvTranspose1d");
}

// 检查匹配是否为 2D 转置卷积层模块
bool is_conv_transpose2d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 调用 is_module 函数检查匹配是否为 2D 转置卷积层模块
  return is_module(
      match, vmap, "conv", "__torch__.torch.nn.modules.conv.ConvTranspose2d");
}

// 检查匹配是否为 2D 批量归一化模块
bool is_batchnorm2d_module(
    const Match& match,
    // 定义函数 `is_module` 用于检查给定模块是否在指定的模块映射 `vmap` 中，并且满足特定的条件。
    // 检查是否是标准的批归一化模块 "__torch__.torch.nn.modules.batchnorm.BatchNorm2d"
    bool regnorm = is_module(
        match,                              // 使用传入的匹配对象 `match` 进行模块检查
        vmap,                               // 使用传入的模块映射 `vmap`
        "batchnorm",                        // 指定要检查的模块名称为 "batchnorm"
        "__torch__.torch.nn.modules.batchnorm.BatchNorm2d");  // 指定标准的批归一化模块名称

    // 检查是否是简单同步批归一化模块 "__torch__.mobile_cv.arch.layers.batch_norm.NaiveSyncBatchNorm"
    bool naivenorm = is_module(
        match,                              // 使用传入的匹配对象 `match` 进行模块检查
        vmap,                               // 使用传入的模块映射 `vmap`
        "batchnorm",                        // 指定要检查的模块名称为 "batchnorm"
        "__torch__.mobile_cv.arch.layers.batch_norm.NaiveSyncBatchNorm");  // 指定简单同步批归一化模块名称

    // 返回两个模块是否任一匹配的逻辑或结果
    return (regnorm || naivenorm);
}

bool is_batchnorm3d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 检查匹配对象是否表示批量归一化三维模块
  return is_module(
      match,
      vmap,
      "batchnorm",  // 模块名称为 "batchnorm"
      "__torch__.torch.nn.modules.batchnorm.BatchNorm3d");  // 预期的模块类型
}

} // namespace jit
} // namespace torch
```