# `.\pytorch\torch\csrc\jit\passes\onnx\shape_type_inference.cpp`

```
// 包含 Torch 库中的 ONNX 形状推断头文件
#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>

// 包含 C++ 标准库头文件
#include <c10/util/irange.h>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <unordered_set>
#include <utility>

// Torch JIT 的日志记录功能
#include <torch/csrc/jit/jit_log.h>

// Torch JIT 中用于 ONNX 的常量折叠和映射头文件
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <torch/csrc/jit/passes/onnx/constant_map.h>

// Torch JIT 中用于修复 ONNX 控制流的功能
#include <torch/csrc/jit/passes/onnx/fixup_onnx_controlflow.h>

// Torch JIT 中的 ONNX 辅助函数
#include <torch/csrc/jit/passes/onnx/helper.h>

// Torch JIT 中用于标量类型分析的头文件
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>

// Torch Python 参数展平功能
#include <torch/csrc/jit/python/python_arg_flatten.h>

// Torch 导出功能的序列化和 ONNX 导出头文件
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/onnx.h>

// Torch 实用工具中的 Python 字符串处理功能
#include <torch/csrc/utils/python_strings.h>

// Torch ONNX 诊断功能的头文件
#include <torch/csrc/onnx/diagnostics/diagnostics.h>

// ONNX 库中的形状推断实现
#include <onnx/shape_inference/implementation.h>

namespace torch {
namespace jit {

// 内联函数，用于检查 Python 对象是否为 None
inline bool PyNone_Check(PyObject* o) {
  return o == Py_None;
}

// 合并推断类型的函数，返回合并后的类型及是否使用推断类型的布尔值
std::pair<TypePtr, bool> MergeInferredType(
    TypePtr existing_type,
    TypePtr inferred_type) {
  auto new_list_type = inferred_type->cast<ListType>();
  auto use_inferred_type = false;
  if (new_list_type) {
    return std::make_pair(inferred_type, true);
  }
  auto new_tensor_type = inferred_type->cast<TensorType>();
  auto old_tensor_type = existing_type->cast<TensorType>();

  if (new_tensor_type && old_tensor_type) {
    if (!old_tensor_type->device()) {
      // 如果旧的张量类型没有设备信息，则直接返回推断类型
      // 这通常表示无效的张量类型（可能是空的）
      return std::make_pair(new_tensor_type, true);
    }
    auto type = old_tensor_type;
    if (new_tensor_type->dim()) {
      type = type->withSymbolicShapes(new_tensor_type->symbolic_sizes());
      use_inferred_type = true;
    }
    if (new_tensor_type->scalarType().has_value()) {
      type = type->withScalarType(new_tensor_type->scalarType());
      use_inferred_type = true;
    }
    return std::make_pair(type, use_inferred_type);
  }

  if (old_tensor_type) {
    // 如果只有旧的张量类型，则返回旧的类型并指示不使用推断类型
    return std::make_pair(existing_type, false);
  }

  auto old_list_type = existing_type->cast<ListType>();
  if (new_tensor_type && old_list_type) {
    if (new_tensor_type->sizes().isComplete()) {
      // 如果推断出的张量类型的尺寸是完整的，则返回推断类型
      return std::make_pair(inferred_type, true);
    }
    // 否则返回旧的类型并指示不使用推断类型
    return std::make_pair(existing_type, false);
  }

  // 默认情况下返回推断类型并指示使用推断类型
  return std::make_pair(inferred_type, true);
}

// 合并推断类型并设置映射的函数
void MergeInferredTypeAndSetMap(
    Value* dest_v,
    TypePtr existing_type,
    TypePtr inferred_type) {
  auto [mergedType, inferred] = MergeInferredType(existing_type, inferred_type);
  dest_v->setType(mergedType);
  ConstantValueMap::SetUseInferredType(dest_v->debugName(), inferred);
}

namespace {
namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;
namespace diagnostics = ::torch::onnx::diagnostics;

// SymbolDimMap 是 Torch 到 ONNX 形状查找表的数据结构
// 用于导出函数时返回形状信息
// 在ONNX维度到形状符号的转换函数中，根据给定的维度对象返回对应的形状符号。
// 如果维度具有具体的维度值(dim_value)，则返回静态大小的形状符号。
c10::ShapeSymbol ONNXDimToShapeSymbol(
    const onnx::TensorShapeProto_Dimension& dim,
    SymbolDimMap& symbol_dim_map,
    DimSymbolMap& dim_symbol_map) {
  if (dim.has_dim_value()) {
    return c10::ShapeSymbol::fromStaticSize(dim.dim_value());
  }
  
  // 如果维度具有参数(dim_param)，则尝试查找是否已经存在对应的符号。
  std::optional<c10::ShapeSymbol> sym = c10::nullopt;
  if (dim.has_dim_param()) {
    // 如果此参数已知，则分配相同的符号。
    GRAPH_UPDATE("Got dim_param:", dim.dim_param());
    auto maybe_symbol = dim_symbol_map.find(dim.dim_param());
    if (maybe_symbol != dim_symbol_map.end()) {
      sym = maybe_symbol->second;
    }
  }
  
  // 如果未找到符号，则创建一个新的符号，并更新符号到参数的映射关系。
  if (!sym) {
    sym = c10::ShapeSymbol::newSymbol();
    // 如果dim.dim_param()为空，则无需跟踪，因为不会有重复。
    symbol_dim_map[sym.value()] = dim.dim_param();
    dim_symbol_map[dim.dim_param()] = sym.value();
  }
  
  return sym.value();
}

// 从ONNX的Tensor类型信息创建Torch的TensorTypePtr。
TensorTypePtr TorchTensorTypeFromONNX(
    const onnx::TypeProto_Tensor& onnx_tensor_type,
    SymbolDimMap& symbol_dim_map,
    DimSymbolMap& dim_symbol_map) {
  std::optional<at::ScalarType> scalar_type;
  if (onnx_tensor_type.has_elem_type()) {
    scalar_type = ONNXTypeToATenType(onnx_tensor_type.elem_type());
  }

  // 创建一个初始的TensorTypePtr对象，表示空的形状和尺寸。
  auto v_type = TensorType::create(
      scalar_type,
      at::kCPU,
      c10::SymbolicShape(),
      c10::VaryingShape<c10::Stride>{},
      {});

  // 如果ONNX的Tensor类型有形状信息，则处理形状信息。
  if (onnx_tensor_type.has_shape()) {
    std::vector<c10::ShapeSymbol> sizes;
    const auto& onnx_shape = onnx_tensor_type.shape();

    // 遍历每个维度，将ONNX的维度对象转换为对应的形状符号。
    for (const auto i : c10::irange(onnx_shape.dim_size())) {
      sizes.emplace_back(ONNXDimToShapeSymbol(
          onnx_shape.dim(i), symbol_dim_map, dim_symbol_map));
    }

    // 根据获取的形状符号，更新TensorTypePtr对象。
    v_type = TensorType::create(scalar_type, at::kCPU, sizes.size(), {});
    v_type = v_type->withSymbolicShapes(c10::SymbolicShape(sizes));

    // 如果所有尺寸都是静态的，则基于尺寸信息填充步长。
    if (v_type->sizes().concrete_sizes().has_value()) {
      v_type = v_type->contiguous();
    }
  }

  return v_type;
}

// 从ONNX的Sequence类型信息创建Torch的ListTypePtr。
ListTypePtr TorchListTypeFromONNX(
    const onnx::TypeProto_Sequence& onnx_sequence_type,
    SymbolDimMap& symbol_dim_map,
    DimSymbolMap& dim_symbol_map) {
  if (onnx_sequence_type.has_elem_type()) {
    const auto& onnx_seq_elem_type = onnx_sequence_type.elem_type();
    if (onnx_seq_elem_type.has_tensor_type()) {
      const auto& onnx_tensor_type = onnx_seq_elem_type.tensor_type();
      // 根据ONNX的Tensor类型创建相应的TensorTypePtr对象。
      const auto v_tensor_type = TorchTensorTypeFromONNX(
          onnx_tensor_type, symbol_dim_map, dim_symbol_map);
      auto v_type = ListType::create(v_tensor_type);
      return v_type;
    }
  }
  
  // 如果未找到有效的Tensor类型信息，则返回nullptr。
  return nullptr;
}
    // 如果 p_info 没有指定类型，则直接返回，不进行后续处理
    if (!p_info.has_type()) {
        return;
    }
    
    // 获取 p_info 中的类型信息
    const auto& p_type = p_info.type();
    
    // 如果类型是张量类型（tensor_type）
    if (p_type.has_tensor_type()) {
        // 根据 ONNX 的张量类型转换为 Torch 的张量类型，并返回结果
        const auto torch_tensor_type = TorchTensorTypeFromONNX(
            p_type.tensor_type(), symbol_dim_map, dim_symbol_map);
        
        // 如果成功转换得到了 Torch 张量类型，则将推断的类型与当前节点的类型合并并更新映射
        if (torch_tensor_type) {
            MergeInferredTypeAndSetMap(v, v->type(), torch_tensor_type);
        }
    } 
    // 如果类型是序列类型（sequence_type）
    else if (p_type.has_sequence_type()) {
        // 根据 ONNX 的序列类型转换为 Torch 的列表类型，并返回结果
        const auto torch_list_type = TorchListTypeFromONNX(
            p_type.sequence_type(), symbol_dim_map, dim_symbol_map);
        
        // 如果成功转换得到了 Torch 列表类型，则将推断的类型与当前节点的类型合并并更新映射
        if (torch_list_type) {
            MergeInferredTypeAndSetMap(v, v->type(), torch_list_type);
        }
    }
}

bool IsValidONNXControlflowNode(const Node* n) {
  // 检查节点是否是循环或条件控制流节点
  auto node_kind = n->kind();
  if (node_kind == ::c10::onnx::Loop || node_kind == ::c10::onnx::If) {
    // 如果节点的子块为空，则跳过该节点
    if (n->blocks().empty()) {
      return false;
    }
  }

  return true;
}

bool IsValidONNXNode(const Node* n) {
  // 检查节点是否是有效的ONNX节点
  auto node_kind = n->kind();

  if (!node_kind.is_onnx()) {
    // 如果节点类型不是ONNX类型，则跳过
    return false;
  }

  if (!IsValidONNXControlflowNode(n)) {
    // 如果节点是控制流节点但无效，则跳过
    return false;
  }

  // 递归检查节点的所有子块和子节点是否有效
  for (auto b : n->blocks()) {
    for (auto b_n : b->nodes()) {
      if (!IsValidONNXNode(b_n)) {
        return false;
      }
    }
  }

  return true;
}

bool CustomSettype(Node* node) {
  // 判断非ONNX节点是否有用户自定义的setType
  auto all_output_has_type = [](Value* output) {
    if (auto output_type = output->type()->cast<TensorType>()) {
      if (auto sizes = output_type->symbolic_sizes().sizes()) {
        // 检查张量类型的符号尺寸是否存在静态尺寸
        return std::any_of(std::begin(*sizes), std::end(*sizes), [](auto size) {
          return size.is_static();
        });
      }
    }
    return false;
  };

  // 对节点的所有输出进行检查，如果任何一个输出有静态尺寸，则返回true
  return std::all_of(
      node->outputs().begin(), node->outputs().end(), all_output_has_type);
}

Value* CloneValueFromListConstruct(
    Value* v,
    std::shared_ptr<Graph> n_graph,
    int opset_version) {
  auto lc_node = v->node();
  TORCH_INTERNAL_ASSERT(lc_node->kind() == ::c10::prim::ListConstruct);
  // 在特定情况下，将prim::ListConstruct节点转换为onnx::Concat节点
  // 如果元素类型为Int，则进行转换，并在图中插入onnx::Concat节点
  TypePtr elem = v->type()->castRaw<ListType>()->getElementType();
  std::optional<at::ScalarType> scalar_type = c10::nullopt;
  if (elem->cast<IntType>()) {
    scalar_type = at::kLong;
    if (isValidToTransformToONNXConcatNode(v->node())) {
      auto concat_node = transformToONNXConcatNode(
          n_graph.get(), v->node(), true, opset_version);
      return concat_node->output();
    }
  } else if (elem->cast<FloatType>()) {
    scalar_type = at::kFloat;
  } else if (elem->cast<BoolType>()) {
    scalar_type = at::kBool;
  } else if (auto t_type = elem->cast<TensorType>()) {
    scalar_type = t_type->scalarType();
  }

  auto input = n_graph->addInput();
  if (scalar_type) {
    # 使用 TensorType::create 方法创建一个张量类型对象 v_type
    auto v_type = TensorType::create(
        scalar_type.value(),  # 设置张量的标量类型，使用 scalar_type 的值
        at::kCPU,              # 设置张量在 CPU 上运行
        c10::SymbolicShape(),  # 使用符号形状，表示形状未知或动态变化
        c10::VaryingShape<c10::Stride>{},  # 使用变化的形状和步长，表示形状和步长可能变化
        {}                     # 空的额外参数，这里没有额外参数传递
    );
    # 将 input 的类型设置为 v_type，即将 input 张量的类型设定为 v_type 所表示的类型
    input->setType(v_type);
  }
  # 返回设置类型后的 input 张量
  return input;
}

// 克隆节点 n 到新图中。
Node* CloneNodeToGraph(
    Node* n,
    std::shared_ptr<Graph> n_graph,
    const ParamMap& params_dict,
    int opset_version) {
  // 使用节点 n 创建克隆节点，并将其插入到新图中。
  auto clone_node = n_graph->createClone(
      n, [&n_graph, &params_dict, opset_version](Value* v) {
        auto v_n = v->node();
        switch (v_n->kind()) {
          case ::c10::prim::Constant:
          case ::c10::onnx::Constant: {
            // 如果输入是常量，则克隆该常量。
            auto constant_n = n_graph->insertNode(
                n_graph->createClone(v_n, [](Value* v) { return v; }));
            return constant_n->output();
          }
          case ::c10::prim::ListConstruct: {
            // 如果输入是列表构造，则从列表构造中克隆值。
            return CloneValueFromListConstruct(v, n_graph, opset_version);
          }
          case ::c10::prim::PackPadded: {
            auto input = n_graph->addInput();
            if (v == v_n->output(0)) {
              // 仅需要对第一个输出进行此解决方法。
              // 在“peephole” pass 中，用户节点被修改以消耗输入。
              input->copyMetadata(v_n->input(0));
            } else {
              input->copyMetadata(v);
            }
            return input;
          }
          default: {
            // 尝试查找输入值并将其插入图中。
            // 如果输入值未知，则在新图中将其设置为图输入，并复制元数据，例如数据类型和形状。
            ::std::optional<at::Tensor> val = ::c10::nullopt;
            auto v0 = params_dict.find(v->debugName());
            if (v0 != params_dict.end()) {
              val = v0->second.toTensor();
            } else {
              val = ConstantValueMap::GetValue(v->debugName());
            }

            if (val.has_value()) {
              return n_graph
                  ->insertNode(n_graph->create(::c10::onnx::Constant)
                                   ->t_(attr::value, val.value()))
                  ->output();
            }
            auto input = n_graph->addInput();
            input->copyMetadata(v);
            return input;
          }
        }
      });
  return clone_node;
}

// 检查类型是否有效（张量、序列或可选类型）。
bool HasValidType(TypePtr type, std::string name) {
  if (auto t_type = type->cast<TensorType>()) {
    if (!t_type->scalarType().has_value()) {
      GRAPH_UPDATE("Input ", name, " is missing tensor datatype.");
      return false;
    }
  } else if (auto s_type = type->cast<ListType>()) {
    auto e_type = s_type->getElementType();
    return HasValidType(e_type, name);
  } else if (auto o_type = type->cast<OptionalType>()) {
    auto e_type = o_type->getElementType();
    return HasValidType(e_type, name);
  }
  return true;
}

// 检查图是否适用于推断。
bool IsGraphValidForInference(std::shared_ptr<Graph> graph) {
  // 验证每个输入是否具有类型（张量、序列或可选）和标量类型。
  // 这是 ONNX 图输入的要求。
  for (auto in : graph->inputs()) {
    return HasValidType(in->type(), in->debugName());
  }
  // 如果条件不满足，直接返回 true
  return true;
// 将图形转换为 ONNX 协议的模型表示
void ConvertGraphToONNXProto(
    std::shared_ptr<Graph> graph,                           // 输入：图形对象的共享指针
    std::shared_ptr<onnx::ModelProto>& model_proto,         // 输出：ONNX 模型的共享指针
    SymbolDimMap& symbol_dim_map,                           // 输入/输出：符号维度映射
    DimSymbolMap& dim_symbol_map,                           // 输出：维度符号映射
    int opset_version) {                                    // 输入：ONNX 协议版本号

  RawDataExportMap export_map;                              // 导出数据映射
  bool val_use_external_data_format;                        // 是否使用外部数据格式
  SymbolDimMap new_symbol_dim_map;                          // 新符号维度映射
  NodeNameMap node_names;                                   // 节点名称映射

  // 调用 export_onnx 函数导出 ONNX 模型
  std::tie(
      model_proto,                                          // 输出：ONNX 模型的共享指针
      export_map,                                           // 输出：导出数据映射
      new_symbol_dim_map,                                   // 输出：新符号维度映射
      val_use_external_data_format,                         // 输出：是否使用外部数据格式
      node_names) =                                         // 输出：节点名称映射
      export_onnx(
          graph,                                            // 输入：图形对象的共享指针
          {},                                               // 输入：空的参数
          opset_version,                                    // 输入：ONNX 协议版本号
          {},                                               // 输入：空的输入参数
          false,                                            // 输入：false，不使用 PyTorch 扩展类型
          onnx_torch::OperatorExportTypes::ONNX,            // 输入：ONNX 操作导出类型
          true,                                             // 输入：true，执行类型检查
          true,                                             // 输入：true，导出节点
          {},                                               // 输入：空的输入参数
          true,                                             // 输入：true，执行正向传递
          false,                                            // 输入：false，不使用混合精度
          std::string());                                   // 输入：空字符串作为模型名称

  // 将新符号维度映射插入到符号维度映射中
  symbol_dim_map.insert(new_symbol_dim_map.begin(), new_symbol_dim_map.end());

  // 将新符号维度映射的反向插入到维度符号映射中
  for (const auto& pair : new_symbol_dim_map) {
    dim_symbol_map[pair.second] = pair.first;
  }

  // 清除模型的每个输出的类型信息
  for (int i = 0; i < model_proto->graph().output_size(); ++i) {
    model_proto->mutable_graph()->mutable_output(i)->clear_type();
  }
}

// 计算常量折叠
std::optional<at::Tensor> ComputeConstantFolding(Node* n,    // 输入：节点指针
                                                 int opset_version) {  // 输入：ONNX 协议版本号
  if (n->inputs().empty()) {
    return c10::nullopt;                                   // 如果节点没有输入，返回空的 std::optional
  }
  std::vector<at::Tensor> inputTensorValues;                // 输入张量值向量
  for (auto i : c10::irange(n->inputs().size())) {          // 遍历节点的输入
    if (TensorTypePtr input_type = n->input(i)->type()->cast<TensorType>()) {  // 检查输入是否为张量类型
      if (!ConstantValueMap::HasValue(n->input(i)->debugName())) {
        return c10::nullopt;                               // 如果输入的常量值映射中没有对应值，返回空的 std::optional
      }
      auto tensor_value =
          ConstantValueMap::GetValue(n->input(i)->debugName()).value();  // 获取输入的常量张量值
      inputTensorValues.emplace_back(tensor_value);         // 将张量值添加到输入张量值向量中
    }
  }
  if (inputTensorValues.size() < n->inputs().size()) {
    return c10::nullopt;                                   // 如果未收集到所有输入的张量值，返回空的 std::optional
  }
  try {
    return onnx_constant_fold::runTorchBackendForOnnx(
        n,                                                  // 输入：节点指针
        inputTensorValues,                                  // 输入：输入张量值向量
        opset_version);                                     // 输入：ONNX 协议版本号
  } catch (const std::exception& ex) {
    auto ex_str = std::string(ex.what());
    ex_str = ex_str.substr(0, ex_str.find('\n'));
    TORCH_WARN("Constant folding in symbolic shape inference fails: ", ex_str);
    return c10::nullopt;                                   // 如果常量折叠过程中出现异常，返回空的 std::optional
  }
}

// 类似于上面的函数，但用于符号形状的计算
std::optional<::c10::SymbolicShape> ComputeShapeFromReshape(
    Node* n,                                                // 输入：节点指针
    const c10::SymbolicShape& input_shape,                  // 输入：输入符号形状
    const c10::SymbolicShape& shape,                        // 输入：目标形状
    int opset_version) {                                    // 输入：ONNX 协议版本号
  std::vector<c10::ShapeSymbol> input_shape_vector =
      input_shape.sizes().value();                          // 获取输入符号形状的大小向量
  std::vector<c10::ShapeSymbol> shape_vector = shape.sizes().value();  // 获取目标形状的大小向量
  TORCH_INTERNAL_ASSERT(
      !input_shape_vector.empty() || !shape_vector.empty(),
      "Reshape node should have at least one input size > 0 when constant folding.");  // 断言：重塑节点应至少有一个输入大小 > 0
  if (shape_vector.empty()) {
    return input_shape;                                    // 如果目标形状为空，返回输入符号形状
  }
  if (input_shape_vector.empty()) {
    // 实现缺失，下一部分在继续
  return shape;
}

// Lambda function to check if a c10::ShapeSymbol ss is zero
auto is_zero = [](c10::ShapeSymbol& ss) { return ss.value() == 0; };

// Find the first occurrence of a zero in shape_vector
auto it_0 = std::find_if(shape_vector.begin(), shape_vector.end(), is_zero);

// Check if shape_vector contains at least one zero
bool shape_has_zero = it_0 != shape_vector.end();

// Find the position of the first occurrence of -1 in shape_vector
int minus_one_pos = -1;
for (auto i : c10::irange(shape_vector.size())) {
  if (shape_vector[i].value() == -1) {
    minus_one_pos = i;
    break;
  }
}

// Initialize allowzero based on opset_version and node attribute "allowzero"
int allowzero = 0;
if (opset_version >= 14 && n->hasAttributeS("allowzero")) {
  allowzero = n->i(attr::allowzero);
}

// Check a condition that must not occur according to certain rules
TORCH_CHECK(
    !(shape_has_zero && allowzero == 1 && minus_one_pos != -1),
    "0 and -1 cannot both be present in `Shape` input of `Reshape` node, when `allowzero=1`.");

// Return the original shape if certain conditions are met
if (minus_one_pos == -1 && (!shape_has_zero || allowzero)) {
  return shape;
}

// Initialize variables for final shape computation
std::vector<c10::ShapeSymbol> final_shape;
uint64_t shape_ratio = 1;
std::unordered_map<int64_t, int64_t> sym_map;

// Iterate through input_shape_vector to calculate shape_ratio and populate sym_map
for (const c10::ShapeSymbol& input_shape : input_shape_vector) {
  // Handle cases where input_shape is static and not zero-sized
  if (input_shape.is_static() && input_shape.static_size() != 0) {
    if (shape_ratio >=
        std::numeric_limits<uint64_t>::max() / input_shape.static_size()) {
      TORCH_WARN(
          "ComputeShapeFromReshape(), shape_ratio overflows, skip shape inference.");
      return c10::nullopt;
    } else {
      shape_ratio *= static_cast<uint64_t>(input_shape.static_size());
    }
  } else {
    // Handle cases where input_shape is symbolic
    auto value = input_shape.value();
    sym_map.emplace(value, 0).first->second += 1;
  }
}

// Determine the size of shape_vector and iterate through it
int shape_size = static_cast<int>(shape_vector.size());
for (const int i : c10::irange(shape_size)) {
  // Skip the position with -1 in shape_vector
  if (i == minus_one_pos) {
    continue;
  }
  c10::ShapeSymbol& target_shape = shape_vector[i];
  
  // Replace target_shape with input_shape_vector[i] if it was zero
  if (target_shape.value() == 0) {
    target_shape = input_shape_vector[i];
  }

  // Adjust shape_ratio and sym_map based on whether target_shape is static or symbolic
  if (target_shape.is_static()) {
    shape_ratio /= static_cast<uint64_t>(target_shape.static_size());
  } else {
    auto value = target_shape.value();
    if (sym_map.find(value) == sym_map.end()) {
      return c10::nullopt;
    }
    sym_map[value]--;
    if (sym_map[value] == 0) {
      sym_map.erase(value);
    }
  }
}

// Check if there are remaining unmatched symbols in sym_map
if (!sym_map.empty()) {
  return c10::nullopt;
}

// Ensure that minus_one_pos is valid at this point
TORCH_INTERNAL_ASSERT(
    minus_one_pos != -1,
    "There are no examples for shape_has_zero = true && minus_one_pos == -1.");

// Populate final_shape with computed shapes up to minus_one_pos
for (const auto i : c10::irange(minus_one_pos)) {
  c10::ShapeSymbol cur_shape(
      shape_vector[i].value() == 0 ? input_shape_vector[i] : shape_vector[i]);
  final_shape.push_back(cur_shape);
}

// Add a shape symbol representing shape_ratio to final_shape if necessary
if (minus_one_pos != -1) {
  final_shape.push_back(
      c10::ShapeSymbol::fromStaticSize(static_cast<int64_t>(shape_ratio)));
}

// Continue populating final_shape for indices greater than minus_one_pos
for (auto i = minus_one_pos + 1; i < shape_size; i++) {
    # 根据条件判断创建一个 ShapeSymbol 对象 cur_shape，选择 input_shape_vector[i] 或 shape_vector[i] 作为参数
    c10::ShapeSymbol cur_shape(
        shape_vector[i].value() == 0 ? input_shape_vector[i] : shape_vector[i]);
    
    # 将 cur_shape 添加到 final_shape 的末尾
    final_shape.push_back(cur_shape);
  }
  
  # 使用 final_shape 创建一个 SymbolicShape 对象 final_shape_0
  c10::SymbolicShape final_shape_0(final_shape);
  
  # 返回创建的 SymbolicShape 对象 final_shape_0
  return final_shape_0;
}

// 从输入形状和重塑向量计算符号形状，如果重塑向量包含负数则返回空
std::optional<::c10::SymbolicShape> ComputeShapeFromExpand(
    const std::vector<::c10::ShapeSymbol>& input_shape,
    const std::vector<int64_t>& reshape) {
  // 检查重塑向量中是否有负数，如果有则返回空
  for (const auto& it : reshape) {
    if (it < 0) {
      return c10::nullopt;
    }
  }
  std::vector<::c10::ShapeSymbol> final_shape;
  // 根据输入形状和重塑向量的大小确定最终形状的大小
  if (input_shape.size() >= reshape.size()) {
    final_shape = input_shape;
  } else {
    // 如果输入形状比重塑向量小，根据重塑向量创建最终形状
    for (auto v : reshape) {
      final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(v));
    }
  }
  auto min_size = std::min(input_shape.size(), reshape.size());
  // 根据最小的大小进行形状更新
  for (const auto i : c10::irange(min_size)) {
    auto idx = final_shape.size() - i - 1;
    auto input_shape_idx = input_shape.size() - i - 1;
    auto reshape_idx = reshape.size() - i - 1;
    if (input_shape[input_shape_idx].is_static()) {
      auto input_shape_value = input_shape[input_shape_idx].static_size();
      auto reshape_value = reshape[reshape_idx];
      // 断言输入形状值与重塑值相等或者其中一个为1
      TORCH_INTERNAL_ASSERT(
          input_shape_value == reshape_value || input_shape_value == 1 ||
              reshape_value == 1,
          "ONNX Expand input shape constraint not satisfied.");
      // 更新最终形状中的静态大小
      final_shape[idx] = ::c10::ShapeSymbol::fromStaticSize(
          std::max(input_shape_value, reshape_value));

    } else {
      // 如果输入形状是符号化的，将最终形状设为新符号
      final_shape[idx] = ::c10::ShapeSymbol::newSymbol();
    }
  }
  // 创建符号形状对象并返回
  ::c10::SymbolicShape shape(final_shape);
  return shape;
}

// 从输入形状和重复向量计算符号形状
std::optional<::c10::SymbolicShape> ComputeShapeFromTile(
    const std::vector<::c10::ShapeSymbol>& input_shape,
    const std::vector<int64_t>& reshape) {
  // 断言输入形状与重塑向量的大小相等
  TORCH_INTERNAL_ASSERT(
      input_shape.size() == reshape.size(),
      "ONNX Tile input shapes do not match.");
  // 检查重塑向量中是否有负数，如果有则返回空
  for (const auto& it : reshape) {
    if (it < 0) {
      return c10::nullopt;
    }
  }
  std::vector<::c10::ShapeSymbol> final_shape;
  final_shape.reserve(input_shape.size());
  // 根据输入形状和重塑向量计算最终形状
  for (const auto i : c10::irange(input_shape.size())) {
    if (input_shape[i].is_static()) {
      // 如果输入形状是静态的，根据重塑向量计算静态大小
      final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(
          input_shape[i].static_size() * reshape[i]));
    } else {
      // 如果输入形状是符号化的，将最终形状设为新符号
      final_shape.emplace_back(::c10::ShapeSymbol::newSymbol());
    }
  }
  // 创建符号形状对象并返回
  ::c10::SymbolicShape shape(final_shape);
  return shape;
}

// 更新值的秩
void UpdateRank(Value* value, size_t rank) {
  // 使用常量值映射设置值的秩
  ConstantValueMap::SetRank(value->debugName(), rank);
  // 如果值的类型是张量类型，更新其符号形状
  if (TensorTypePtr value_type = value->type()->cast<TensorType>()) {
    std::optional<size_t> rank_opt = rank;
    auto shape = ::c10::SymbolicShape(rank_opt);
    value->setType(value_type->withSymbolicShapes(shape));
  }
}

// 从向量更新值的形状
void UpdateShapeFromVector(
    Value* value,
    const std::vector<int64_t>& shape_size) {
  // 创建符号形状对象
  ::c10::SymbolicShape shape(shape_size);
  // 使用常量值映射设置值的形状
  ConstantValueMap::SetShape(value->debugName(), shape);
  // 如果形状大小为空，更新值的秩为0并返回
  if (shape_size.empty()) {
    UpdateRank(value, 0);
    return;
  }
  // 设置值的秩为形状大小的大小
  ConstantValueMap::SetRank(value->debugName(), shape_size.size());
  // 如果值的类型是张量类型，更新其符号形状
  if (TensorTypePtr value_type = value->type()->cast<TensorType>()) {
    value->setType(value_type->withSymbolicShapes(shape));
  }
}
}

void UpdateShape(Value* value, const ::c10::SymbolicShape& shape) {
    // 更新常量值映射中的形状，使用值的调试名称作为键
    ConstantValueMap::SetShape(value->debugName(), shape);

    // 如果形状的秩（rank）有值
    if (shape.rank().has_value()) {
        auto rank = shape.rank().value();

        // 如果形状的秩为0
        if (rank == 0) {
            // 更新值的秩为0，并返回
            UpdateRank(value, 0);
            return;
        }

        // 设置常量值映射中的秩
        ConstantValueMap::SetRank(value->debugName(), rank);

        // 如果值的类型为张量类型指针
        if (TensorTypePtr value_type = value->type()->cast<TensorType>()) {
            // 使用符号形状更新值的类型
            value->setType(value_type->withSymbolicShapes(shape));
        }
    }
}

void UpdateShapeConstantValueMap(
    const Value* value,
    const ::c10::SymbolicShape& shape) {
    // 设置常量值映射中的形状，使用值的调试名称作为键
    ConstantValueMap::SetShape(value->debugName(), shape);

    // 如果形状的秩（rank）有值
    if (shape.rank().has_value()) {
        auto rank = shape.rank().value();
        
        // 设置常量值映射中的秩
        ConstantValueMap::SetRank(value->debugName(), rank);
    }
}

std::optional<std::vector<int64_t>> GetValueFromListConstructNode(
    Node* lc_node) {
    // 初始化形状大小向量
    std::vector<int64_t> shape_size;

    // 遍历列表构造节点的输入
    for (const auto& input : lc_node->inputs()) {
        // 如果输入是张量类型并且在常量值映射中有值
        if (input->type()->cast<TensorType>() &&
            ConstantValueMap::HasValue(input->debugName())) {
            
            // 获取输入的常量值
            auto lc_value = ConstantValueMap::GetValue(input->debugName()).value();

            // 如果常量值的维度为0
            if (lc_value.dim() == 0) {
                // 提取并添加到形状大小向量中
                int64_t lc_value_0 = lc_value.item<int64_t>();
                shape_size.emplace_back(lc_value_0);
            }
        }
    }

    // 如果列表构造节点的输入数量与形状大小向量的大小相同，则返回形状大小向量；否则返回空
    return lc_node->inputs().size() == shape_size.size()
        ? std::optional<std::vector<int64_t>>(shape_size)
        : c10::nullopt;
}

void SetShapeValueFromListConstructNode(Node* lc_node) {
    // 初始化形状符号向量
    std::vector<c10::ShapeSymbol> shape_size;

    // 遍历列表构造节点的输入
    for (const auto& input : lc_node->inputs()) {
        // 如果输入是张量类型指针
        if (TensorTypePtr shape_type = input->type()->cast<TensorType>()) {
            // 如果在常量值映射中有值
            if (ConstantValueMap::HasValue(input->debugName())) {
                // 获取输入的常量值
                auto lc_value = ConstantValueMap::GetValue(input->debugName()).value();

                // 如果常量值的维度为0
                if (lc_value.dim() == 0) {
                    // 提取并添加到形状符号向量中
                    int64_t lc_value_0 = lc_value.item<int64_t>();
                    shape_size.emplace_back(c10::ShapeSymbol::fromStaticSize(lc_value_0));
                }
            }
            // 如果在常量值映射中有形状值
            else if (ConstantValueMap::HasShapeValue(input->debugName())) {
                // 获取输入的形状值
                auto lc_value = ConstantValueMap::GetShapeValue(input->debugName()).value();

                // 如果形状值的秩为1
                if (lc_value.rank() == 1U) {
                    // 添加形状值的第一个元素到形状符号向量中
                    shape_size.emplace_back(lc_value.at(0));
                }
            }
        }
    }

    // 如果列表构造节点的输入数量与形状符号向量的大小相同
    if (lc_node->inputs().size() == shape_size.size()) {
        // 创建符号形状对象
        c10::SymbolicShape final_shape(shape_size);

        // 设置常量值映射中的形状值，使用列表构造节点的输出的调试名称作为键
        ConstantValueMap::SetShapeValue(
            lc_node->output()->debugName(), final_shape);
    }
}

std::vector<::c10::ShapeSymbol> Broadcast(
    const std::vector<::c10::ShapeSymbol>& input_shape_value_0,
    // 计算输入形状的维度信息，生成合成的最终形状
    const std::vector<::c10::ShapeSymbol>& input_shape_value_1) {
      // 计算输入形状的维度数量
      size_t rank_0 = input_shape_value_0.size();
      size_t rank_1 = input_shape_value_1.size();
      // 确定最大和最小的维度数量
      size_t rank_max = std::max(rank_0, rank_1);
      size_t rank_min = std::min(rank_0, rank_1);
      // 创建用于存储最终形状的向量
      std::vector<::c10::ShapeSymbol> final_shape;
      final_shape.reserve(rank_max);
      // 使用 ::c10::ShapeSymbol::newSymbol 生成 rank_max 个形状符号对象
      std::generate_n(
          std::back_inserter(final_shape), rank_max, ::c10::ShapeSymbol::newSymbol);
      // 逐维度比较输入形状信息
      for (auto idx : c10::irange(rank_min)) {
        // 获取当前维度的形状符号对象
        const c10::ShapeSymbol& ss_shape_0 = input_shape_value_0[rank_0 - 1 - idx];
        const c10::ShapeSymbol& ss_shape_1 = input_shape_value_1[rank_1 - 1 - idx];
        // 检查当前维度是否是静态维度
        bool is_static_0 = ss_shape_0.is_static();
        bool is_static_1 = ss_shape_1.is_static();
        // 确定在 final_shape 中的索引位置
        size_t shape_idx = rank_max - 1 - idx;
        // 如果两个维度都是静态的
        if (is_static_0 && is_static_1) {
          // 获取静态大小
          int64_t static_0_sz = ss_shape_0.static_size();
          int64_t static_1_sz = ss_shape_1.static_size();
          // 处理0维张量的特殊情况
          if (std::min(static_0_sz, static_1_sz) == 0) {
            final_shape[shape_idx] = ::c10::ShapeSymbol::fromStaticSize(
                std::min(static_0_sz, static_1_sz));
          } else {
            final_shape[shape_idx] = ::c10::ShapeSymbol::fromStaticSize(
                std::max(static_0_sz, static_1_sz));
          }
        } else if (!is_static_0 && !is_static_1) {
          // 如果两个维度都不是静态的，且形状符号值相等
          if (ss_shape_0.value() == ss_shape_1.value()) {
            final_shape[shape_idx] = ss_shape_0;
          }
        }
      }
      // 根据输入形状的维度数量不同，填充剩余的 final_shape
      if (rank_0 < rank_1) {
        for (size_t idx = rank_min; idx < rank_max; idx++) {
          size_t shape_idx = rank_max - 1 - idx;
          final_shape[shape_idx] = input_shape_value_1[shape_idx];
        }
      } else {
        for (size_t idx = rank_min; idx < rank_max; idx++) {
          size_t shape_idx = rank_max - 1 - idx;
          final_shape[shape_idx] = input_shape_value_0[shape_idx];
        }
      }
      // 返回合成的最终形状
      return final_shape;
    }
// 处理广播节点的函数，要求节点的输入数为2
void ProcessBroadcastNode(Node* n) {
  // 断言节点的输入数为2
  TORCH_INTERNAL_ASSERT(n->inputs().size() == 2);
  
  // 检查输入节点的形状是否在常量值映射中
  if (ConstantValueMap::HasShape(n->input(0)->debugName()) &&
      ConstantValueMap::HasShape(n->input(1)->debugName())) {
    
    // 获取输入节点的形状
    auto input_shape_0 = ConstantValueMap::GetShape(n->input(0)->debugName());
    auto input_shape_value_0 = input_shape_0.value().sizes().value();
    
    auto input_shape_1 = ConstantValueMap::GetShape(n->input(1)->debugName());
    auto input_shape_value_1 = input_shape_1.value().sizes().value();
    
    // 计算广播后的最终形状
    auto final_shape = Broadcast(input_shape_value_0, input_shape_value_1);
    
    // 更新节点输出的符号形状
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }
}

// 处理拼接节点的形状函数
void ProcessShapeForConcatNode(Node* n) {
  // 获取拼接轴的值
  int axis = n->i(attr::axis);
  
  // 检查输入节点的秩是否在常量值映射中
  if (ConstantValueMap::HasRank(n->input(0)->debugName())) {
    // 获取输入节点的秩
    auto rank = ConstantValueMap::GetRank(n->input(0)->debugName()).value();
    
    size_t axis_adjust = 0;
    // 根据轴的正负调整轴值
    if (axis >= 0) {
      axis_adjust = static_cast<size_t>(axis);
    } else {
      axis_adjust = static_cast<size_t>(axis + static_cast<int>(rank));
    }
    
    // 存储最终形状的向量
    std::vector<::c10::ShapeSymbol> final_shape;
    final_shape.reserve(rank);
    
    // 遍历节点的秩，构建最终形状
    for (auto idx : c10::irange(rank)) {
      if (idx == axis_adjust) {
        auto flag = true;
        int64_t size_total = 0;
        
        // 遍历节点的输入，检查每个输入的形状
        for (auto input_idx : c10::irange(n->inputs().size())) {
          if (ConstantValueMap::HasShape(n->input(input_idx)->debugName())) {
            auto input_shape =
                ConstantValueMap::GetShape(n->input(input_idx)->debugName());
            auto input_shape_value = input_shape.value().sizes();
            auto shape_symbol = input_shape_value.value()[idx];
            
            // 如果形状为静态值，累加静态大小
            if (shape_symbol.is_static()) {
              size_total += shape_symbol.static_size();
            } else {
              flag = false;
              break;
            }
          }
        }
        
        // 如果所有输入的形状都是静态的，则将静态大小添加到最终形状中
        if (flag) {
          final_shape.emplace_back(
              ::c10::ShapeSymbol::fromStaticSize(size_total));
        } else {
          final_shape.emplace_back(::c10::ShapeSymbol::newSymbol());
        }
      } else {
        auto flag = false;
        
        // 遍历节点的输入，检查每个输入的形状
        for (auto input_idx : c10::irange(n->inputs().size())) {
          if (ConstantValueMap::HasShape(n->input(input_idx)->debugName())) {
            auto input_shape =
                ConstantValueMap::GetShape(n->input(input_idx)->debugName());
            auto input_shape_value = input_shape.value().sizes();
            auto shape_symbol = input_shape_value.value()[idx];
            
            // 如果形状为静态值，则将静态大小添加到最终形状中
            if (shape_symbol.is_static()) {
              final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(
                  shape_symbol.static_size()));
              flag = true;
              break;
            }
          }
        }
        
        // 如果没有找到静态形状，则添加新符号到最终形状中
        if (!flag) {
          final_shape.emplace_back(::c10::ShapeSymbol::newSymbol());
        }
      }
    }
    
    // 更新节点输出的符号形状
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }
}
// 处理连接节点的形状值，根据输入节点的常量值或者形状值进行计算
void ProcessShapeValueForConcatNode(Node* n) {
  // 获取输入节点的数量，即连接节点的输入数量
  auto rank = n->inputs().size();
  // 创建一个空的形状符号向量
  std::vector<c10::ShapeSymbol> shape_size;
  // 遍历连接节点的每个输入节点
  for (const auto& input : n->inputs()) {
    // 检查常量值映射中是否存在当前输入节点的值
    if (ConstantValueMap::HasValue(input->debugName())) {
      // 获取当前输入节点的常量值
      auto concat_value = ConstantValueMap::GetValue(input->debugName()).value();
      // 如果常量值是一维且长度为1
      if (concat_value.dim() == 1 && concat_value.size(0) == 1) {
        // 提取第一个元素作为 int64_t 类型
        auto concat_value_0 = concat_value[0].item<int64_t>();
        // 将其转换为形状符号并添加到形状符号向量中
        shape_size.emplace_back(c10::ShapeSymbol::fromStaticSize(concat_value_0));
      }
    } else if (ConstantValueMap::HasShapeValue(input->debugName())) {
      // 检查形状值映射中是否存在当前输入节点的形状值
      auto concat_value = ConstantValueMap::GetShapeValue(input->debugName()).value();
      // 如果形状值的秩为1
      if (concat_value.rank() == 1U) {
        // 添加形状值的第一个元素到形状符号向量中
        shape_size.emplace_back(concat_value.at(0));
      }
    }
  }
  // 如果连接节点的输入数量与形状符号向量的大小相等
  if (rank == shape_size.size()) {
    // 创建最终的符号形状对象
    c10::SymbolicShape final_shape(shape_size);
    // 将最终的符号形状值设置为输出节点的形状
    ConstantValueMap::SetShapeValue(n->output(0)->debugName(), final_shape);
  }
}

// 处理连接节点，依次处理其形状和形状值
void ProcessConcatNode(Node* n) {
  // 处理连接节点的形状
  ProcessShapeForConcatNode(n);
  // 处理连接节点的形状值
  ProcessShapeValueForConcatNode(n);
}

// 处理矩阵乘法节点
void ProcessMatMulNode(Node* n) {
  // 如果输入节点的形状在常量值映射中存在
  if (ConstantValueMap::HasShape(n->input(0)->debugName()) &&
      ConstantValueMap::HasShape(n->input(1)->debugName())) {
    // 获取第一个输入节点和第二个输入节点的形状
    auto input_shape_0 = ConstantValueMap::GetShape(n->input(0)->debugName()).value();
    auto input_shape_value_0 = input_shape_0.sizes().value();
    auto input_shape_1 = ConstantValueMap::GetShape(n->input(1)->debugName()).value();
    auto input_shape_value_1 = input_shape_1.sizes().value();
    // 计算第一个输入节点和第二个输入节点的秩
    size_t rank_0 = input_shape_value_0.size();
    size_t rank_1 = input_shape_value_1.size();
    // 处理秩为1的输入，按照类似 numpy.matmul 的方式处理
    auto is_rank_0_1 = false;
    if (rank_0 == 1) {
      // 在第一个输入节点的形状值的开头插入一个静态大小为1的形状符号
      input_shape_value_0.insert(
          input_shape_value_0.begin(), ::c10::ShapeSymbol::fromStaticSize(1));
      rank_0 = 2;
      is_rank_0_1 = true;
    }
    auto is_rank_1_1 = false;
    if (rank_1 == 1) {
      // 在第二个输入节点的形状值的末尾添加一个静态大小为1的形状符号
      input_shape_value_1.emplace_back(::c10::ShapeSymbol::fromStaticSize(1));
      rank_1 = 2;
      is_rank_1_1 = true;
    }
    // 根据 PyTorch 的文档，广播逻辑仅适用于批处理维度，不适用于矩阵维度
    // 因此在广播之前，需要移除最后两个维度（即矩阵维度）
    auto final_shape = Broadcast(
        std::vector<::c10::ShapeSymbol>(
            input_shape_value_0.begin(), input_shape_value_0.end() - 2),
        std::vector<::c10::ShapeSymbol>(
            input_shape_value_1.begin(), input_shape_value_1.end() - 2));
    // 将最后两个维度添加回去，除非它们本来就不存在并且是由此函数插入的
    // 然后应用 [n,k]X[k,m]=[n,m]，其中 n=input_shape_value_0[rank_0 - 2], m=input_shape_value_1[rank_1 - 1]
    # 如果不是 rank 0 的第一个维度，则将其添加到最终形状中
    if (!is_rank_0_1) {
      final_shape.emplace_back(input_shape_value_0[rank_0 - 2]);
    }
    # 如果不是 rank 1 的第一个维度，则将其添加到最终形状中
    if (!is_rank_1_1) {
      final_shape.emplace_back(input_shape_value_1[rank_1 - 1]);
    }
    # 更新节点 n 的输出形状为最终形状的符号形状
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
void ProcessReduceNode(Node* n) {
  // 检查节点输入的第一个输入的调试名称是否具有常量值映射的形状信息
  if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
    // 获取第一个输入节点的形状信息
    auto input_shape_0 = ConstantValueMap::GetShape(n->input(0)->debugName());
    // 获取第一个输入节点的形状大小
    auto input_shape_value_0 = input_shape_0.value().sizes();
    // 获取形状的秩（rank）
    size_t rank_0 = input_shape_value_0.value().size();
    // 存储最终形状的符号数组
    std::vector<::c10::ShapeSymbol> final_shape;
    // 创建一个整数向量，存储轴的索引
    std::vector<int64_t> axes_vector(rank_0);
    // 如果节点具有名为 "axes" 的属性，则使用该属性的值作为轴向量
    if (n->hasAttributeS("axes")) {
      axes_vector = n->is(attr::axes);
    } 
    // 如果节点有多于一个输入，且没有 "axes" 属性，则尝试从第二个输入获取轴向量
    else if (n->inputs().size() > 1) {
      axes_vector =
          ConstantValueMap::GetValueInto1DInt64Vector(n->input(1)->debugName());
    } 
    // 否则，默认情况下使用从 0 到 rank_0 的索引作为轴向量
    else {
      std::iota(axes_vector.begin(), axes_vector.end(), 0);
    }

    // 处理负索引，将其转换为正索引
    for (auto idx : c10::irange(axes_vector.size())) {
      if (axes_vector[idx] < 0) {
        axes_vector[idx] += rank_0;
      }
    }

    // 预留最终形状的空间
    final_shape.reserve(rank_0);

    // 获取保持维度（keepdims）属性，默认为 1
    int64_t keepdims = 1;
    if (n->hasAttributeS("keepdims")) {
      keepdims = n->i(attr::keepdims);
    }

    // 根据轴向量和保持维度的值构建最终形状
    for (auto idx : c10::irange(rank_0)) {
      auto it = std::find(axes_vector.begin(), axes_vector.end(), idx);
      if (it != axes_vector.end()) {
        // 如果轴向量中包含当前索引，并且 keepdims 不为 0，则添加静态大小为 1 的符号
        if (keepdims != 0) {
          final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(1));
        }
      } else {
        // 否则，添加输入形状中对应索引的值作为最终形状的符号
        final_shape.emplace_back(input_shape_value_0.value()[idx]);
      }
    }

    // 更新节点输出的形状
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }
}

void ProcessReshapeNode(Node* n, int opset_version) {
  // 获取输入和形状的调试名称
  const auto& input_name = n->input(0)->debugName();
  const auto& shape_name = n->input(1)->debugName();

  // 当形状输入的值是静态已知时，计算输出形状
  if (ConstantValueMap::HasValue(shape_name)) {
    // 获取静态形状值
    auto static_shape_value =
        ConstantValueMap::GetValueInto1DInt64Vector(shape_name);
    // 获取输入的符号形状
    auto symbolic_input_shape = ConstantValueMap::GetShape(input_name);
    // 如果输入的符号形状存在且静态形状值非空，则计算最终形状
    if (symbolic_input_shape && !static_shape_value.empty()) {
      auto final_shape = ComputeShapeFromReshape(
          n,
          symbolic_input_shape.value(),
          c10::SymbolicShape(static_shape_value),
          opset_version);
      // 如果计算得到最终形状，则更新节点输出的形状并返回
      if (final_shape) {
        UpdateShape(n->output(), final_shape.value());
        return;
      }
    }
  }

  // 当形状输入的值是符号已知时，计算输出形状
  if (ConstantValueMap::HasShapeValue(shape_name) &&
      ConstantValueMap::HasShape(input_name)) {
    // 获取输入的符号形状和符号形状值
    auto symbolic_input_shape = ConstantValueMap::GetShape(input_name).value();
    auto symbolic_shape_value =
        ConstantValueMap::GetShapeValue(shape_name).value();
    // 计算最终形状
    auto final_shape = ComputeShapeFromReshape(
        n, symbolic_input_shape, symbolic_shape_value, opset_version);
    // 如果计算得到最终形状，则更新节点输出的形状并返回
    if (final_shape.has_value()) {
      UpdateShape(n->output(), final_shape.value());
      return;
    }
  }

  // 当只知道新形状的形状时，分配输出的秩（rank）
  if (ConstantValueMap::HasShape(shape_name)) {
    // 获取输出形状的 rank（维度数量），如果成功获取到
    auto output_rank = ConstantValueMap::GetShapeInto1DInt64Vector(shape_name);
    if (output_rank.has_value()) {
      // 断言输出的 rank（维度数量）为 1
      TORCH_INTERNAL_ASSERT(output_rank.value().size() == 1);
      // 更新节点 n 的输出 rank（维度数量）
      UpdateRank(n->output(), output_rank.value()[0]);
      // 函数结束，返回
      return;
    }
  }

  // 对于 ListConstruct，在 ProcessConstantValueMap 的开头已经处理，这里无需进一步处理。
  // 如果输入节点 1 的类型为 TensorType
  if (TensorTypePtr shape_type = n->input(1)->type()->cast<TensorType>()) {
    // 如果可能，设置重塑（Reshape）的输出 rank（维度数量）
    // 根据形状推断，我们有以下情况：
    // %4236 : Float(*, device=cpu) = onnx::Transpose[perm=[0]](%4235)
    // %4237 : Long(2, strides=[1], device=cpu) = onnx::Concat[axis=0](%4232)
    // %4238 : FloatTensor(device=cpu) = onnx::Reshape(%4236, %4237)
    // 我们可以得到一个具有已知 rank 的 SymbolicShape：
    // %4238 : Float(*, *, strides=[2480, 1], requires_grad=0, device=cpu) =
    // onnx::Reshape(%4236, %4237)
    auto shape_type_dim = shape_type->dim();
    if (shape_type_dim.has_value()) {
      auto shape_type_size = shape_type->sizes()[0];
      if (shape_type_size.has_value()) {
        // 获取 shape_type_size 的值作为 rank
        size_t rank = shape_type_size.value();
        // 更新节点 n 的输出 rank（维度数量）
        UpdateRank(n->output(), rank);
      }
    }
  }
// 计算切片操作后的形状，返回一个 SymbolicShape 对象
c10::SymbolicShape ComputeShapeForSlice(
    const std::vector<c10::ShapeSymbol>& input_shape, // 输入张量的形状符号列表
    const std::vector<int64_t>& start_vector,         // 切片开始索引列表
    const std::vector<int64_t>& end_vector,           // 切片结束索引列表
    const std::vector<int64_t>& axes_vector,          // 切片轴列表
    const std::vector<int64_t>& step_vector) {        // 切片步长列表
  TORCH_INTERNAL_ASSERT(axes_vector.size() <= input_shape.size()); // 断言：轴数不超过输入形状的维度数
  TORCH_INTERNAL_ASSERT(axes_vector.size() == start_vector.size()); // 断言：轴数与切片开始索引列表长度相同
  TORCH_INTERNAL_ASSERT(axes_vector.size() == end_vector.size());   // 断言：轴数与切片结束索引列表长度相同
  TORCH_INTERNAL_ASSERT(axes_vector.size() == step_vector.size());  // 断言：轴数与切片步长列表长度相同

  std::vector<c10::ShapeSymbol> final_shape; // 创建最终的形状符号列表
  final_shape = input_shape; // 将输入形状复制给最终形状列表

  for (const auto idx : c10::irange(axes_vector.size())) { // 对于每一个切片轴
    auto axis = axes_vector[idx]; // 获取当前轴
    TORCH_INTERNAL_ASSERT(axis >= 0); // 断言：轴的索引值非负

    if (!input_shape[axis].is_static()) { // 如果输入形状的当前轴不是静态的
      final_shape[axis] = c10::ShapeSymbol::newSymbol(); // 将最终形状列表的当前轴设为新符号
      continue; // 继续处理下一个轴
    }

    auto input_shape_axis_value = input_shape[axis].static_size(); // 获取输入形状当前轴的静态大小
    auto cur_start = start_vector[idx]; // 获取当前轴的切片开始索引
    auto cur_end = end_vector[idx];     // 获取当前轴的切片结束索引
    auto cur_step = step_vector[idx];   // 获取当前轴的切片步长

    if (cur_start < -input_shape_axis_value) { // 处理切片开始索引小于负的静态轴大小情况
      cur_start = 0; // 将切片开始索引设为0
    } else if (cur_start < 0) { // 处理切片开始索引小于0但大于负的静态轴大小情况
      cur_start = input_shape_axis_value + cur_start; // 调整切片开始索引
    } else if (cur_start > input_shape_axis_value - 1) { // 处理切片开始索引超过静态轴大小的情况
      cur_start = input_shape_axis_value; // 将切片开始索引设为静态轴大小
    }

    if (cur_end < -input_shape_axis_value) { // 处理切片结束索引小于负的静态轴大小情况
      cur_end = -1; // 将切片结束索引设为-1
    } else if (cur_end < 0) { // 处理切片结束索引小于0但大于负的静态轴大小情况
      cur_end = input_shape_axis_value + cur_end; // 调整切片结束索引
    } else if (cur_end > input_shape_axis_value - 1) { // 处理切片结束索引超过静态轴大小的情况
      cur_end = input_shape_axis_value; // 将切片结束索引设为静态轴大小
    }

    TORCH_INTERNAL_ASSERT(cur_step != 0); // 断言：切片步长不为0

    if (cur_step > 0) { // 处理切片步长大于0的情况
      final_shape[axis] = c10::ShapeSymbol::fromStaticSize(
          (cur_end - cur_start - 1) / cur_step + 1); // 计算静态大小并设置最终形状列表的当前轴
    } else { // 处理切片步长小于0的情况
      final_shape[axis] = c10::ShapeSymbol::fromStaticSize(
          (cur_start - cur_end - 1) / (-cur_step) + 1); // 计算静态大小并设置最终形状列表的当前轴
    }
  }

  return c10::SymbolicShape(final_shape); // 返回包含最终形状的 SymbolicShape 对象
}

// 处理切片节点的函数
void ProcessSliceNode(Node* n, int opset_version) {
  bool valid = ConstantValueMap::HasShape(n->input(0)->debugName()); // 检查输入张量的形状是否已知

  // 对于 opset 版本大于等于 10，只有在 'axes' 已知的情况下才能推断形状
  if (opset_version >= 10) {
    if (n->inputs().size() > 3) { // 如果节点的输入数量大于3
      valid = valid && ConstantValueMap::HasValue(n->input(3)->debugName()); // 检查 'axes' 是否具有已知值
    }
  }

  if (!valid) { // 如果形状未知
    if (ConstantValueMap::HasRank(n->input(0)->debugName())) { // 检查输入张量的秩是否已知
      auto rank = ConstantValueMap::GetRank(n->input(0)->debugName()).value(); // 获取输入张量的秩
      UpdateRank(n->output(), rank); // 更新节点输出的秩信息
    }
    return; // 返回，不继续处理
  } else { // 如果形状已知
    auto shape_size_0 = ConstantValueMap::GetShape(n->input(0)->debugName()).value(); // 获取输入张量的形状
    // 继续处理...
    // 检查输入张量的维度是否已知，并处理切片操作
    if (shape_size_0.rank().has_value()) {
      // 获取输入张量的形状大小
      auto input0_shape_value = shape_size_0.sizes().value();

      // 初始化存储切片开始、结束和步长的向量
      std::vector<int64_t> start_vector;
      std::vector<int64_t> end_vector;
      std::vector<int64_t> step_vector;

      // 初始化包含所有轴索引的向量
      std::vector<int64_t> axes_vector(input0_shape_value.size(), 0);
      for (const auto i : c10::irange(input0_shape_value.size())) {
        axes_vector[i] = i;
      }

      // 根据操作集版本和输入数量处理轴向量
      if (opset_version >= 10 && n->inputs().size() > 3) {
        // 从第四个输入中获取轴向量
        axes_vector = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input(3)->debugName());
      } else if (opset_version < 10 && n->hasAttributeS("axes")) {
        // 从节点的 'axes' 属性获取轴向量
        axes_vector = n->is(attr::axes);
      }

      // 处理轴向量中的负数索引，转换为正数索引
      for (auto& axis : axes_vector) {
        if (axis < 0) {
          axis += input0_shape_value.size();
        }
      }

      // 根据操作集版本处理切片的开始和结束向量
      if (opset_version < 10) {
        // 从节点的 'starts' 和 'ends' 属性获取开始和结束向量
        start_vector = n->is(attr::starts);
        end_vector = n->is(attr::ends);
      } else {
        // 检查切片的开始、结束或步长是否未知，若是，则将所有指定轴标记为未知
        std::vector<uint64_t> indices = {1U, 2U, 4U};
        bool start_end_step_known =
            std::all_of(indices.begin(), indices.end(), [&n](auto i) {
              return (i >= n->inputs().size()) ||
                  ConstantValueMap::HasValue(n->input(i)->debugName());
            });
        if (!start_end_step_known) {
          // 将未知的轴标记为符号形状
          auto final_shape = input0_shape_value;
          for (const auto axis : axes_vector) {
            final_shape[axis] = c10::ShapeSymbol::newSymbol();
          }
          // 更新节点的输出形状为符号形状
          UpdateShape(n->output(), final_shape);
          return;
        }

        // 从输入中获取切片的开始、结束和步长向量
        start_vector = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input(1)->debugName());
        end_vector = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input(2)->debugName());
        if (n->inputs().size() > 4) {
          step_vector = ConstantValueMap::GetValueInto1DInt64Vector(
              n->input(4)->debugName());
        }
      }

      // 如果步长向量为空，则将其初始化为全一向量
      if (step_vector.empty()) {
        step_vector = std::vector<int64_t>(axes_vector.size(), 1);
      }

      // 计算切片操作后的最终形状
      auto final_shape = ComputeShapeForSlice(
          input0_shape_value,
          start_vector,
          end_vector,
          axes_vector,
          step_vector);
      
      // 更新节点的输出形状为计算得到的最终形状
      UpdateShape(n->output(), final_shape);
    }
  }
}

// 处理不可变节点的函数，接受一个指向节点的指针作为参数
void ProcessUnchangeNode(Node* n) {
  // 检查输入节点的形状是否在常量值映射中
  if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
    // 获取输入节点的形状大小
    auto shape_size_0 = ConstantValueMap::GetShape(n->input(0)->debugName()).value();
    // 更新输出节点的形状
    UpdateShape(n->output(), shape_size_0);
  }
}

// 处理时间序列节点的函数，接受一个指向节点的指针作为参数
void ProcessTimeSeriesNode(Node* n) {
  // 获取输入节点0和节点1的形状
  auto input0_shape = ConstantValueMap::GetShape(n->input(0)->debugName());
  auto input1_shape = ConstantValueMap::GetShape(n->input(1)->debugName());
  // 如果输入节点的形状不完整，则返回
  if (!(input0_shape.has_value() && input1_shape.has_value())) {
    return;
  }
  // 获取输入节点0和节点1的形状值
  auto input0_shape_value = input0_shape.value().sizes();
  auto input1_shape_value = input1_shape.value().sizes();
  // 定义形状符号变量
  c10::ShapeSymbol seq_length;
  c10::ShapeSymbol num_directions;
  c10::ShapeSymbol batch_size;
  c10::ShapeSymbol hidden_size;
  // 如果输入节点0的形状值可用，则获取序列长度和批量大小
  if (input0_shape_value.has_value()) {
    seq_length = input0_shape_value.value()[0];
    batch_size = input0_shape_value.value()[1];
  }

  // 如果输入节点1的形状值可用，则获取方向数
  if (input1_shape_value.has_value()) {
    num_directions = input1_shape_value.value()[0];
    // 如果输入节点1的第二个维度是静态的，则计算隐藏大小
    if (input1_shape_value.value()[1].is_static()) {
      auto input1_value = input1_shape_value.value()[1].static_size();
      switch (n->kind()) {
        case ::c10::onnx::RNN:
          hidden_size = c10::ShapeSymbol::fromStaticSize(input1_value);
          break;
        case ::c10::onnx::LSTM:
          hidden_size = c10::ShapeSymbol::fromStaticSize(input1_value / 4);
          break;
        case ::c10::onnx::GRU:
          hidden_size = c10::ShapeSymbol::fromStaticSize(input1_value / 3);
          break;
        default:
          throw std::runtime_error(
              std::string() + "This is not a valid TimeSeries Node with type " +
              n->kind().toDisplayString());
      }
    } else {
      hidden_size = c10::ShapeSymbol::newSymbol();
    }
  }

  // 如果节点有多个输出，则更新输出的形状
  if (n->outputs().size() > 1) {
    std::vector<c10::ShapeSymbol> final_shape = {
        seq_length, num_directions, batch_size, hidden_size};
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }

  // 对于输出索引为2和3的情况，更新其形状
  for (const auto idx : c10::irange(2U, 4U)) {
    if (n->outputs().size() > idx) {
      std::vector<c10::ShapeSymbol> final_shape = {
          num_directions, batch_size, hidden_size};
      UpdateShape(n->output(idx - 1), c10::SymbolicShape(final_shape));
    }
  }
}

// 处理展开节点的函数，接受一个指向节点的指针作为参数
void ProcessUnsqueezeNode(Node* n) {
  // 获取输出节点的类型
  TensorTypePtr output_type = n->output(0)->type()->cast<TensorType>();
  // 如果输出类型为空，则返回
  if (output_type == nullptr) {
    return;
  }
  // 如果输出的维度为1，并且输入节点在常量值映射中有形状值，则设置输出节点的形状值
  if (output_type->dim().has_value() && output_type->dim().value() == 1 &&
      ConstantValueMap::HasShapeValue(n->input(0)->debugName())) {
    auto shape_value =
        ConstantValueMap::GetShapeValue(n->input(0)->debugName()).value();
    // 当标量表示形状时，在展开时与形状值相同
    ConstantValueMap::SetShapeValue(n->output()->debugName(), shape_value);
  }
}

// 作为对ONNX形状推断的补充，此函数利用常量值映射提供了额外的功能
// 执行常量计算。如果节点是常量节点并且值为张量，则将其值存入常量映射中。
void ComputeConstant(Node* n, int opset_version) {
  // 检查节点是否为常量节点
  if (n->kind() == ::c10::onnx::Constant) {
    // 检查节点值是否为张量
    if (n->kindOf(attr::value) == AttributeKind::t) {
      // 获取节点的常量张量值
      at::Tensor const_val = n->t(attr::value);
      // 创建张量的副本
      at::Tensor const_val_copy = at::empty(const_val.sizes(), const_val.options());
      // 将值复制到副本中
      const_val_copy.copy_(const_val);
      // 将常量张量副本存入常量值映射中，键为节点输出的调试名称
      ConstantValueMap::SetValue(n->output()->debugName(), const_val_copy);
    }
    return;
  }

  // 初始化变量
  auto only_rank_available = false;
  size_t rank = 0;

  // 常量折叠
  auto const_fold_val = ComputeConstantFolding(n, opset_version);
  if (const_fold_val.has_value()) {
    // 创建常量折叠值的张量副本
    at::Tensor const_fold_val_copy = at::empty(const_fold_val.value().sizes(), const_fold_val.value().options());
    // 将值复制到副本中
    const_fold_val_copy.copy_(const_fold_val.value());
    // 将常量折叠值的张量副本存入常量值映射中，键为节点输出的调试名称
    ConstantValueMap::SetValue(n->output()->debugName(), const_fold_val_copy);
    // 根据张量的形状更新节点输出的形状
    UpdateShapeFromVector(n->output(), const_fold_val_copy.sizes().vec());
    return;
  }

  // 根据节点类型进行不同处理
  switch (n->kind()) {
    // 处理加法、除法、等于、大于等操作等节点，进行广播处理
    case ::c10::onnx::Add:
    case ::c10::onnx::Div:
    case ::c10::onnx::Equal:
    case ::c10::onnx::Greater:
    case ::c10::onnx::GreaterOrEqual:
    case ::c10::onnx::Less:
    case ::c10::onnx::LessOrEqual:
    case ::c10::onnx::Mod:
    case ::c10::onnx::Mul:
    case ::c10::onnx::Pow:
    case ::c10::onnx::Sub: {
      // 处理广播节点
      ProcessBroadcastNode(n);
      break;
    }
    // 处理获取形状操作
    case ::c10::onnx::Shape: {
      // 获取输入节点的形状并转换为一维的 int64 向量
      auto input_shape = ConstantValueMap::GetShapeInto1DInt64Vector(n->input()->debugName());
      if (input_shape.has_value()) {
        auto shape_value = input_shape.value();
        // 创建张量的选项
        auto options = c10::TensorOptions().dtype(at::kLong).device(at::kCPU);
        auto shape_value_size = static_cast<int64_t>(shape_value.size());
        // 从形状值数据创建张量，并将其转换到 CPU 设备
        auto f = at::from_blob(shape_value.data(), {shape_value_size}, at::kLong).to(at::kCPU);
        // 需要在这里进行复制操作
        at::Tensor f_copy = at::empty({shape_value_size}, options);
        f_copy.copy_(f);
        // 将形状值的张量副本存入常量值映射中，键为节点输出的调试名称
        ConstantValueMap::SetValue(n->output()->debugName(), f_copy);
        // 创建一个包含静态大小的形状符号向量
        std::vector<::c10::ShapeSymbol> final_shape_vector(1, c10::ShapeSymbol::fromStaticSize(shape_value_size));
        ::c10::SymbolicShape final_shape(final_shape_vector);
        // 更新节点输出的形状
        UpdateShape(n->output(), final_shape);
      }
      break;
    }
    // 处理重塑操作
    case ::c10::onnx::Reshape: {
      // 处理重塑节点
      ProcessReshapeNode(n, opset_version);
      break;
    }
    case ::c10::onnx::Transpose: {
      // 如果节点有 "perm" 属性
      if (n->hasAttributeS("perm")) {
        // 获取排列 perm_v，并确定其维度 rank
        auto perm_v = n->is(attr::perm);
        rank = perm_v.size();
        
        // 检查是否为默认的排列方式
        auto is_default_perm = false;
        if (rank == 2 && perm_v[0] == 1 && perm_v[1] == 0) {
          is_default_perm = true;
        }
        
        // 标志：形状是否已更新
        auto shape_updated = false;
        
        // 如果输入节点的形状信息是已知的
        if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
          // 获取输入节点的形状大小
          auto shape_size_0 =
              ConstantValueMap::GetShape(n->input(0)->debugName())
                  .value()
                  .sizes();
          
          // 如果形状大小可用
          if (shape_size_0.has_value()) {
            auto shape_vector_0 = shape_size_0.value();
            
            // 创建最终的形状向量
            std::vector<::c10::ShapeSymbol> final_shape_vector(
                shape_vector_0.size(), ::c10::ShapeSymbol());
            
            // 根据是否是默认排列方式来更新最终形状向量
            if (is_default_perm) {
              std::reverse_copy(
                  std::begin(shape_vector_0),
                  std::end(shape_vector_0),
                  std::begin(final_shape_vector));
            } else {
              for (const auto i : c10::irange(shape_vector_0.size())) {
                final_shape_vector[i] = shape_vector_0[perm_v[i]];
              }
            }
            
            // 创建符号化的最终形状
            ::c10::SymbolicShape final_shape(final_shape_vector);
            
            // 更新节点输出的形状
            UpdateShape(n->output(), final_shape);
            shape_updated = true;
          }
        }
        
        // 如果形状未更新
        if (!shape_updated) {
          // 如果不是默认排列方式，则只有 rank 可用
          if (!is_default_perm) {
            only_rank_available = true;
          } else if (ConstantValueMap::HasRank(n->input(0)->debugName())) {
            // 否则，如果输入节点的 rank 已知，则只有 rank 可用
            rank = ConstantValueMap::GetRank(n->input(0)->debugName()).value();
            only_rank_available = true;
          }
        }
      }
      // 结束处理 Transpose 节点
      break;
    }
    case ::c10::onnx::Concat: {
      // 处理 Concat 节点
      ProcessConcatNode(n);
      break;
    }
    case ::c10::onnx::ConstantOfShape: {
      // 如果节点的输入有已知的常量值
      if (ConstantValueMap::HasValue(n->input()->debugName())) {
        // 获取常量值作为形状的临时向量
        auto shape_temp = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input()->debugName());
        
        // 根据临时向量更新节点输出的形状
        UpdateShapeFromVector(n->output(), shape_temp);
        
        // 如果临时向量不为空
        if (!shape_temp.empty()) {
          // 如果节点有 "value" 属性
          if (n->hasAttributeS("value")) {
            // 获取属性值，并重复到与形状相同的尺寸
            auto value = n->t(attr::value).repeat(shape_temp);
            ConstantValueMap::SetValue(n->output()->debugName(), value);
          } else {
            // 否则，创建一个全为 0.0 的张量，并重复到与形状相同的尺寸
            auto options =
                c10::TensorOptions().dtype(at::kFloat).device(at::kCPU);
            auto value = at::full({1}, 0.0, options).repeat(shape_temp);
            ConstantValueMap::SetValue(n->output()->debugName(), value);
          }
        }
      }
      // 结束处理 ConstantOfShape 节点
      break;
    }
    case ::c10::onnx::Expand: {
      // 检查是否已知输入张量的形状
      if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
        // 获取输入张量的形状大小
        auto input0_shape_size =
            ConstantValueMap::GetShape(n->input(0)->debugName())
                .value()
                .sizes();
        // 如果形状大小可用
        if (input0_shape_size.has_value()) {
          // 获取输入张量的形状值
          auto input0_shape_value = input0_shape_size.value();
          // 检查是否已知输入的扩展形状
          if (ConstantValueMap::HasValue(n->input(1)->debugName())) {
            // 当 `shape` 的值在静态情况下已知时，
            // 可以计算输出的形状
            auto shape_temp = ConstantValueMap::GetValueInto1DInt64Vector(
                n->input(1)->debugName());
            // 计算最终的输出形状
            auto final_shape =
                ComputeShapeFromExpand(input0_shape_value, shape_temp);
            // 如果最终形状可用
            if (final_shape.has_value()) {
              // 更新节点输出的形状
              UpdateShape(n->output(), final_shape.value());
            }
          } else if (
              auto expand_shape =
                  ConstantValueMap::GetShapeInto1DInt64VectorWithOneUnknown(
                      n->input(1)->debugName())) {
            // 当 `shape` 的形状在静态情况下已知时，
            // 可以计算输出的秩（rank）
            TORCH_INTERNAL_ASSERT(
                expand_shape.value().size() == 1,
                "`Shape` input to `Expand` should be a 1-D tensor. Instead got rank ",
                expand_shape.value().size());
            // 如果扩展形状的秩大于 0
            if (expand_shape.value()[0] > 0) {
              // 创建包含形状符号的最终形状向量
              std::vector<c10::ShapeSymbol> final_shape;
              std::generate_n(
                  std::back_inserter(final_shape),
                  expand_shape.value()[0],
                  ::c10::ShapeSymbol::newSymbol);
              // 更新节点输出的符号形状
              UpdateShape(n->output(), c10::SymbolicShape(final_shape));
            }
          }
        }
      }
      // 结束对 Expand 操作的处理
      break;
    }
    case ::c10::onnx::NonZero: {
      // 如果输入节点是常量且已知其形状
      if (ConstantValueMap::HasRank(n->input()->debugName())) {
        // 获取输入节点的秩（rank）
        auto rank = ConstantValueMap::GetRank(n->input()->debugName()).value();
        // 创建一个包含秩信息的形状符号向量
        std::vector<c10::ShapeSymbol> dims;
        dims.emplace_back(
            c10::ShapeSymbol::fromStaticSize(static_cast<int64_t>(rank)));
        
        // 获取输入节点
        auto input_node = n->input()->node();
        // 如果输入节点是 ConstantOfShape 类型
        if (input_node->kind() == ::c10::onnx::ConstantOfShape) {
          // 检查是否有 'value' 属性
          if (input_node->hasAttributeS("value")) {
            // 获取 'value' 属性的值并转换为 Float 类型的张量
            auto value =
                input_node->t(attr::value).toType(at::ScalarType::Float);
            auto value_a = value.accessor<float, 1>();
            // 如果值的长度为 1 并且绝对值大于 1e-6
            if (value_a.size(0) == 1 && std::abs(value_a[0]) > 1e-6) {
              // 如果输入节点的形状已知
              if (ConstantValueMap::HasShape(n->input()->debugName())) {
                // 获取输入节点的形状信息
                auto shape_size_0 =
                    ConstantValueMap::GetShape(n->input()->debugName()).value();
                // 如果形状是完整的
                if (shape_size_0.isComplete()) {
                  // 获取形状向量并计算元素个数
                  auto shape_vector_0 = shape_size_0.sizes().value();
                  int64_t num_elements = 1;
                  for (auto cur_dim : shape_vector_0) {
                    num_elements *= cur_dim.static_size();
                  }
                  // 添加元素个数作为静态大小的形状符号
                  dims.emplace_back(c10::ShapeSymbol::fromStaticSize(
                      static_cast<int64_t>(num_elements)));
                }
              }
            }
          }
        }
        
        // 如果 dims 中只有一个元素
        if (dims.size() == 1) {
          // 添加一个新的符号作为形状
          dims.emplace_back(c10::ShapeSymbol::newSymbol());
        }
        
        // 创建符号形状对象
        c10::SymbolicShape shape_v(dims);
        // 更新节点输出的形状
        UpdateShape(n->output(), shape_v);
      }
      break;
    }
    case ::c10::onnx::MatMul: {
      // 处理矩阵乘法节点
      ProcessMatMulNode(n);
      break;
    }
    case ::c10::onnx::ReduceMean:
    case ::c10::onnx::ReduceProd: {
      // 处理求平均或求乘积节点
      ProcessReduceNode(n);
      break;
    }
    case ::c10::onnx::RNN:
    case ::c10::onnx::LSTM:
    case ::c10::onnx::GRU: {
      // 处理时间序列节点（RNN、LSTM、GRU）
      ProcessTimeSeriesNode(n);
      break;
    }
    case ::c10::onnx::Size: {
      // 如果输入节点的形状已知
      if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
        // 获取输入节点的形状信息
        auto input0_shape_size =
            ConstantValueMap::GetShape(n->input(0)->debugName())
                .value()
                .sizes();
        // 如果形状信息存在
        if (input0_shape_size.has_value()) {
          // 获取形状值
          auto input0_shape_value = input0_shape_size.value();
          int64_t total_size = 1;
          auto is_full_static = true;
          // 遍历形状值的维度
          for (const auto i : c10::irange(input0_shape_value.size())) {
            // 如果维度是静态的
            if (input0_shape_value[i].is_static()) {
              // 计算总大小
              total_size *= input0_shape_value[i].static_size();
            } else {
              // 如果有非静态维度，则不是完全静态的形状
              is_full_static = false;
              break;
            }
          }
          // 如果形状是完全静态的
          if (is_full_static) {
            // 将总大小转换为张量并保存到常量值映射中
            auto f_final = onnx_constant_fold::IntToTensor(total_size);
            ConstantValueMap::SetValue(n->output(0)->debugName(), f_final);
          }
        }
      }
      break;
    }
    case ::c10::onnx::Slice: {
      // 处理切片节点
      ProcessSliceNode(n, opset_version);
      break;
    }
    case ::c10::onnx::Cast:
    case ::c10::onnx::Relu:
    case ::c10::onnx::Softmax: {
      // 处理无需修改的节点，调用 ProcessUnchangeNode 函数
      ProcessUnchangeNode(n);
      break;
    }
    case ::c10::onnx::Tile: {
      // 检查输入节点0的形状是否在常量映射中
      if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
        // 获取输入节点0的形状大小
        auto input0_shape_size =
            ConstantValueMap::GetShape(n->input(0)->debugName())
                .value()
                .sizes();
        if (input0_shape_size.has_value()) {
          auto input0_shape_value = input0_shape_size.value();
          // 检查输入节点1的值是否在常量映射中
          if (ConstantValueMap::HasValue(n->input(1)->debugName())) {
            // 获取输入节点1的值作为一维整数向量
            auto shape_temp = ConstantValueMap::GetValueInto1DInt64Vector(
                n->input(1)->debugName());
            // 计算 Tile 操作后的最终形状
            auto final_shape =
                ComputeShapeFromTile(input0_shape_value, shape_temp);
            if (final_shape.has_value()) {
              // 更新节点的输出形状
              UpdateShape(n->output(), final_shape.value());
            }
          }
        }
      }
      break;
    }
    case ::c10::onnx::Unsqueeze: {
      // 处理 Unsqueeze 节点，调用 ProcessUnsqueezeNode 函数
      ProcessUnsqueezeNode(n);
      break;
    }
    default: {
      // 默认情况下不做任何操作
      break;
    }
  }
  // 如果节点有多个输出或者输出节点的形状在常量映射中存在，则返回
  if (n->outputs().size() > 1 ||
      ConstantValueMap::HasShape(n->output(0)->debugName())) {
    return;
  }
  // 如果仅支持获取秩（rank），则更新节点的输出秩（rank）
  if (only_rank_available) {
    UpdateRank(n->output(), rank);
  }
}

// 检查列表构造节点的第一个元素是否为整数类型
bool IsListConstructIntType(const Value* v) {
  // 检查节点是否为列表构造节点
  if (v->node()->kind() == prim::ListConstruct) {
    // 获取列表的类型信息
    auto listType = v->node()->output()->type();
    // 获取列表中元素的类型信息
    auto containedType = listType->containedTypes().at(0);
    // 如果列表中第一个元素是整数类型，则返回 true
    if (containedType == IntType::get()) {
      return true;
    }
  }
  // 如果不是列表构造节点或者第一个元素不是整数类型，则返回 false
  return false;
}

// 检查图中所有输入是否都是静态的，并允许缓存返回值
// 由于这会遍历图中的所有输入（包括权重），对于大型图形可能开销较大。
// 由于在导出时对每个节点都调用此函数且输入保持不变，可以通过缓存来减少导出时间。
bool AllGraphInputsStaticWithCaching(const Graph* g) {
  // 尝试从常量值映射中获取图中所有输入是否静态的信息
  auto maybe_is_static = ConstantValueMap::GetAllGraphInputsStatic();
  if (maybe_is_static.has_value()) {
    // 如果已经有缓存的结果，则直接返回缓存的值
    return maybe_is_static.value();
  } else {
    // 否则，计算图中所有输入是否静态，并将结果缓存起来
    bool ret = AllGraphInputsStatic(g);
    ConstantValueMap::SetAllGraphInputsStatic(ret);
    return ret;
  }
}

// 处理常量值映射中的节点信息
void ProcessConstantValueMap(Node* n, int opset_version) {
  // 在执行 ONNX 形状推断之前，更新节点的形状可靠性
  UpdateReliable(n);

  // 使用缓存来确定图中所有输入是否静态
  auto static_input_shape = AllGraphInputsStaticWithCaching(n->owningGraph());
  
  // 遍历节点的所有输出
  for (auto i : c10::irange(n->outputs().size())) {
    // 检查输出是否为张量类型
    if (TensorTypePtr output_type = n->output(i)->type()->cast<TensorType>()) {
      // 如果输出具有维度信息
      if (output_type->dim().has_value()) {
        // 获取张量的维度数并存入常量值映射中
        size_t rank = static_cast<size_t>(output_type->dim().value());
        ConstantValueMap::SetRank(n->output(i)->debugName(), rank);
        // 获取张量的符号化大小并更新形状信息
        auto shape = output_type->symbolic_sizes();
        if (shape.isComplete()) {
          UpdateShape(n->output(i), shape);
        }
      }
    }
  }

  // 更新常量值映射中节点的输入信息，包括处理 ListConstruct 的情况
  for (auto i : c10::irange(n->inputs().size())) {
    // 如果输入节点指向的类型可以转换为 TensorTypePtr
    if (TensorTypePtr input_type = n->input(i)->type()->cast<TensorType>()) {
      // 如果输入节点的维度信息是可用的
      if (input_type->dim().has_value()) {
        // 获取输入节点的维度作为 rank
        size_t rank = static_cast<size_t>(input_type->dim().value());
        // 将输入节点的名称与其 rank 映射存储到 ConstantValueMap 中
        ConstantValueMap::SetRank(n->input(i)->debugName(), rank);
        
        // 仅当输入节点为 onnx 类型或者静态输入形状被启用时更新形状
        if (n->input(i)->node()->kind().is_onnx() || static_input_shape) {
          // 获取输入节点的符号化大小信息作为 shape
          auto shape = input_type->symbolic_sizes();
          // 如果 ConstantValueMap 中没有记录该节点的形状，则更新形状
          if (!ConstantValueMap::HasShape(n->input(i)->debugName())) {
            UpdateShape(n->input(i), shape);
          }
        }
      }
    } else if (IsListConstructIntType(n->input(i))) {
      // 如果输入节点是列表构造整数类型
      auto lc_node = n->input(i)->node();
      // 获取列表构造节点的输入数量作为 rank
      auto rank = lc_node->inputs().size();
      // 从列表构造节点中获取整数向量作为可选值
      auto lc_vector_optional = GetValueFromListConstructNode(lc_node);
      
      // 如果成功获取到列表构造节点中的整数向量
      if (lc_vector_optional.has_value()) {
        auto lc_vector = lc_vector_optional.value();
        // 创建长整型张量的选项，并将 lc_vector 数据转换为张量 f
        auto options = c10::TensorOptions().dtype(at::kLong).device(at::kCPU);
        auto lc_vector_size = static_cast<int64_t>(lc_vector.size());
        auto f = at::from_blob(lc_vector.data(), {lc_vector_size}, at::kLong)
                     .to(at::kCPU);
        // 需要进行复制操作
        at::Tensor f_copy = at::empty({lc_vector_size}, options);
        f_copy.copy_(f);
        // 将 f 的副本存储到 ConstantValueMap 中
        ConstantValueMap::SetValue(n->input(i)->debugName(), f_copy);
        // 根据向量更新输入节点的形状
        UpdateShapeFromVector(n->input(i), {lc_vector_size});
      } else {
        // 根据 rank 更新输入节点的形状
        UpdateShapeFromVector(n->input(i), {static_cast<int64_t>(rank)});
      }
      // 从列表构造节点设置形状值
      SetShapeValueFromListConstructNode(lc_node);
    }
  }
  // 额外逻辑：更新图和 ConstantValueMap 中的常量信息
  ComputeConstant(n, opset_version);
// 定义一个特殊的后处理函数，用于处理特定节点类型的后续操作
void SpecialPostProcess(Node* n) {
  switch (n->kind()) {  // 根据节点的类型进行分支判断
    case ::c10::onnx::Cast: {  // 如果节点类型是 onnx::Cast
      // 当输入到 onnx::Cast 的输入张量形状不完整时，ONNX 形状推断无法分配输出张量的形状。
      // 例如，缺少形状、秩、数据类型等。此后处理函数设置输出张量的正确数据类型，
      // 因为数据类型信息存储在 Cast 属性中。
      TensorTypePtr t_type = n->output()->type()->cast<TensorType>();
      if (nullptr != t_type && !t_type->scalarType().has_value()) {
        auto onnx_dtype = n->i(attr::to);  // 获取 Cast 操作的目标数据类型
        auto aten_dtype = ONNXTypeToATenType(onnx_dtype);  // 将 ONNX 数据类型转换为 ATen 数据类型
        n->output()->setType(t_type->withScalarType(aten_dtype));  // 设置输出类型的数据类型
      }
      break;
    }
    case ::c10::onnx::ConstantOfShape: {  // 如果节点类型是 onnx::ConstantOfShape
      // 当输入 `shape` 不是常量时，ONNX 形状推断无法传播输出张量的形状。
      // 这是一个临时解决方案，当部分信息可用时，例如，了解输出张量的秩或了解符号形状。
      // 一旦我们有了适当的符号传播，这个解决方案将不再需要。
      auto shape_node = n->input(0)->node();  // 获取输入 `shape` 节点
      if (shape_node->kind() == ::c10::onnx::Shape) {
        // Shape -> ConstantOfShape
        auto orig_type = shape_node->input()->type()->cast<TensorType>();  // 原始输入的张量类型
        auto v_type = n->output()->type()->cast<TensorType>();  // 输出张量的类型
        if (v_type && !v_type->sizes().concrete_sizes()) {
          if (orig_type && orig_type->dim()) {
            // 分配原始 onnx::Shape 输入的符号形状。
            v_type = v_type->withSymbolicShapes(orig_type->symbolic_sizes());
            n->output()->setType(v_type);
          } else if (shape_node->input()->node()->kind() == ::c10::prim::ListConstruct) {
            // 分配原始 onnx::Shape 输入的秩。
            v_type = v_type->withSizes({static_cast<int64_t>(
                shape_node->input()->node()->inputs().size())});
            n->output()->setType(v_type);
          }
        }
      } else if (shape_node->kind() == ::c10::prim::ListConstruct) {
        // ListConstruct -> ConstantOfShape
        auto v_type = n->output()->type()->cast<TensorType>();  // 输出张量的类型
        if (v_type && !v_type->sizes().concrete_sizes()) {
          auto value = n->t(attr::value);  // 获取属性 `value` 的值
          v_type = v_type->withScalarType(value.scalar_type());  // 设置输出张量的数据类型
          std::vector<c10::ShapeSymbol> sizes(
              shape_node->inputs().size(), c10::ShapeSymbol::newSymbol());  // 创建符号形状的向量
          v_type = v_type->withSymbolicShapes(c10::SymbolicShape(sizes));  // 设置符号形状
          n->output()->setType(v_type);
        }
      }
      break;
    }
    case ::c10::onnx::If: {  // 如果节点类型是 onnx::If
      if (!IsValidONNXControlflowNode(n)) {  // 如果节点不是有效的 ONNX 控制流节点，跳过处理
        break;
      }
      FixupONNXControlflowNodeOutputs(n);  // 修复 ONNX 控制流节点的输出
      break;
    }
  }
}
    case ::c10::onnx::Loop: {
      // 检查是否是有效的 ONNX 控制流节点，如果无效则跳出循环
      if (!IsValidONNXControlflowNode(n)) {
        break;
      }
      // 修正 ONNX 控制流节点的输出
      FixupONNXControlflowNodeOutputs(n);
      // 结束当前 case 分支
      break;
    }
// 更新输出类型根据 ONNX 协议
void UpdateOutputTypeByONNXProto(
    Node* n,  // 当前节点指针
    Node* clone_node,  // 克隆节点指针
    const onnx::ModelProto& model_proto,  // ONNX 模型的协议
    SymbolDimMap& symbol_dim_map,  // 符号维度映射
    DimSymbolMap& dim_symbol_map) {  // 维度符号映射

  // 从 value_info 中获取数据并更新原始图
  const auto updateNodeOutputsByONNXValueInfo =
      [&](const onnx::ValueInfoProto& v_info) {
        for (size_t i = 0; i < n->outputs().size(); ++i) {
          if (clone_node->output(i)->debugName() == v_info.name()) {
            UpdateTorchValueByOnnxValueInfo(
                n->output(i), v_info, symbol_dim_map, dim_symbol_map);  // 根据 ONNX value_info 更新 Torch 值
          }
        }
      };

  // 检查图的输出以获取推断形状
  for (const auto i : c10::irange(graph_proto.output_size())) {
    updateNodeOutputsByONNXValueInfo(graph_proto.output(i));  // 更新节点输出根据 ONNX value_info
  }

  // 检查 value_infos 以获取推断形状
  for (const auto i : c10::irange(graph_proto.value_info_size())) {
    updateNodeOutputsByONNXValueInfo(graph_proto.value_info(i));  // 更新节点输出根据 ONNX value_info
  }
}

// 从父级复制块输入元数据
void FetchBlockInputMetadataFromParent(Block* b) {
  auto n = b->owningNode();  // 获取拥有该块的节点
  if (nullptr != n && n->kind() == ::c10::onnx::Loop) {
    // 将节点的输入元数据复制到子图的输入
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      b->inputs().at(i)->setType(n->inputs().at(i)->type());  // 设置块的输入类型为节点的输入类型
    }
  }
}

// 移除已处理的输入
void RemoveProcessedInputs(const Node* n) {
  // 在进行形状推断后，从 ConstantValueMap 中移除存储的中间张量，以减少内存使用
  // 这只会移除不再被任何其他节点需要的张量

  // 返回节点是否已经进行了形状推断处理
  const auto isNodeProcessed = [](const Node* node) {
    const auto& outputs = node->outputs();
    return std::any_of(outputs.begin(), outputs.end(), [](const Value* output) {
      // 假设形状推断至少可以确定输出的秩
      // 如果该假设错误，一些中间张量只有在整个图的形状推断完成后才会被删除
      return ConstantValueMap::HasRank(output->debugName());
    });
  };

  // 如果所有使用该输入的消费节点都已经被处理，则输入值不再需要
  const auto isValueNoLongerNeeded = [isNodeProcessed](const Value* input) {
    const auto& uses = input->uses();
    return std::all_of(
        uses.begin(), uses.end(), [isNodeProcessed](const Use& use) {
          return isNodeProcessed(use.user);
        });
  };

  for (const auto* input : n->inputs()) {
    if (ConstantValueMap::HasValue(input->debugName()) &&
        isValueNoLongerNeeded(input)) {
      ConstantValueMap::EraseValue(input->debugName());  // 从 ConstantValueMap 中删除输入值
    }
  }
}

// ONNX 形状类型推断
void ONNXShapeTypeInference(
    Block* b,  // 当前块指针
    const ParamMap& params_dict,  // 参数映射字典
    int opset_version) {  // 操作集版本号
  FetchBlockInputMetadataFromParent(b);  // 从父级获取块输入元数据
  auto valsToParamsMap = buildValueToParamsMap(b, params_dict);  // 构建值到参数的映射
  for (auto const& it : valsToParamsMap) {
    // 迭代值到参数映射
    // 获取当前迭代器指向的键和值
    auto key = it.first;
    auto value = it.second;
    // 检查当前键所对应节点的类型是否为 prim::Param
    if (key->node()->kind() == prim::Param) {
      // 如果值的第二部分是张量类型，则将其作为常量值存入 ConstantValueMap
      if (value.second.isTensor()) {
        ConstantValueMap::SetValue(value.first, value.second.toTensor());
      }
    } else if (key->node()->kind() == ::c10::onnx::Constant) {
      // 如果键对应节点的类型是 ::c10::onnx::Constant
      // 获取该节点的 value 属性作为常量值
      at::Tensor const_val = key->node()->t(attr::value);
      // 创建一个与 const_val 具有相同大小和选项的新张量
      at::Tensor const_val_copy =
          at::empty(const_val.sizes(), const_val.options());
      // 将 const_val 的数据复制到 const_val_copy 中
      const_val_copy.copy_(const_val);
      // 将复制后的常量值存入 ConstantValueMap
      ConstantValueMap::SetValue(value.first, const_val_copy);
    } else {
      // 如果键对应节点的类型不是 prim::Param 也不是 ::c10::onnx::Constant，则抛出异常
      throw std::runtime_error(
          "ONNXShapeTypeInference - Unsupported kind of constant node found.");
    }
  }
  // 遍历当前块的每个节点
  for (auto n : b->nodes()) {
    // 遍历当前节点的每个子块，对每个子块递归进行 ONNXShapeTypeInference
    for (auto subblock : n->blocks()) {
      ONNXShapeTypeInference(subblock, params_dict, opset_version);
    }
    // 对当前节点进行 ONNXShapeTypeInference
    ONNXShapeTypeInference(n, params_dict, opset_version);
    // 移除已处理的输入
    RemoveProcessedInputs(n);
  }
} // namespace

// 对于某些操作符，存在一些与形状推断无关的输入。
// 例如，LSTM 的第四个输入 (sequence_lens) 是可选的，
// 可以通过其他必需的输入完成形状推断。
// 当我们进行可靠性计算时，不需要这个输入是可靠的。
static std::unordered_map<std::string, std::unordered_set<int64_t>>
    non_required_shape_inference_idx_map = {{"onnx::LSTM", {4}}};

// 检查图中所有输入是否都是静态的（即形状信息已知且完整）
bool AllGraphInputsStatic(const Graph* g) {
  for (auto n : g->inputs()) {
    // 如果输入是 Tensor 类型
    if (TensorTypePtr input_type = n->type()->cast<TensorType>()) {
      // 如果输入有维度信息
      if (input_type->dim()) {
        auto shape = input_type->symbolic_sizes();
        // 如果常量值映射中没有该节点的形状信息，则更新
        if (!ConstantValueMap::HasShape(n->debugName())) {
          UpdateShapeConstantValueMap(n, shape);
        }
      }
    }
  }
  // 再次检查所有输入
  for (auto n : g->inputs()) {
    // 有些输入可能不是 Tensor 类型，比如 quantized.LinearPackedParamsBase，
    // 所以这里只检查 Tensor 类型的输入
    if (n->type()->cast<TensorType>() && !n->isCompleteTensor()) {
      return false;
    }
  }
  return true;
}

// 判断节点的所有输入是否可靠或静态
std::pair<bool, bool> AreInputsReliableOrStatic(Node* n) {
  auto reliable = true;
  auto complete = true;
  auto input_size = n->inputs().size();
  std::unordered_set<int64_t> non_required_idx = {};
  // 如果节点在非必需的形状推断索引映射中
  if (non_required_shape_inference_idx_map.find(n->kind().toDisplayString()) !=
      non_required_shape_inference_idx_map.end()) {
    non_required_idx =
        non_required_shape_inference_idx_map[n->kind().toDisplayString()];
  }
  for (auto idx : c10::irange(input_size)) {
    // 如果索引在非必需索引集合中，则跳过
    if (!non_required_idx.empty() &&
        non_required_idx.find(idx) != non_required_idx.end()) {
      continue;
    }
    auto input = n->inputs()[idx];
    // 总是将 None 视为可靠且完整的，因为它表示 ONNX 中未指定的可选输入
    if (input->node()->mustBeNone()) {
      continue;
    }
    // 检查输入的类型是否可靠
    reliable &=
        ConstantValueMap::GetTypeReliable(input->debugName()).value_or(false);
    if (auto pt = input->type()->cast<TensorType>()) {
      // 如果 Tensor 的大小不完整，则标记为不完整
      if (!pt->sizes().isComplete()) {
        complete = false;
      }
    }
  }
  return std::make_pair(reliable, complete);
}

// 这里没有必要注明 ONNX 类型，但我们需要它
// 用于某些遗留测试，当 onnx_shape_inference=False 时。
static std::unordered_set<std::string> nodeTypeReliableForTracer = {
    "prim::ListConstruct",
    "onnx::Cast",
    "onnx::Constant",
    "onnx::Relu",
    "com.microsoft::Gelu",
    "aten::ATen"};

// 更新输出值的可靠性
void UpdateReliable(
    torch::jit::Value* output,
    const std::pair<bool, bool>& inferred_type_reliable,
  // 检查是否已推断出类型
  auto inferred =
      ConstantValueMap::GetUseInferredType(output->debugName()).value_or(false);
  // 检查节点类型是否可靠用于追踪
  auto isTypeReliableForTracer =
      nodeTypeReliableForTracer.find(
          output->node()->kind().toDisplayString()) !=
      nodeTypeReliableForTracer.end();
  // 如果未推断出类型并且节点类型不可靠，并且不是 ONNX 类型，并且不需要类型警告
  if (!inferred && !isTypeReliableForTracer &&
      !output->node()->kind().is_onnx() && no_type_warning) {
    // 发出警告，说明缺少节点形状推断信息
    TORCH_WARN(
        "The shape inference of ",
        output->node()->kind().toDisplayString(),
        " type is missing, so it may result in wrong shape inference for the exported graph. ",
        "Please consider adding it in symbolic function.");
    // 实验性质，无输出到标准输出或标准错误流
    diagnostics::Diagnose(
        diagnostics::Rule::kNodeMissingOnnxShapeInference,
        diagnostics::Level::kWarning,
        {{"op_name", output->node()->kind().toDisplayString()}});
  }
  // 初始化可靠性为假
  auto reliable = false;
  // 如果已推断出类型，则从推断类型的可靠性中获取值
  if (inferred) {
    reliable = inferred_type_reliable.first;
  } else {
    // 否则，如果第二个推断类型可靠并且节点类型可靠，则设置为可靠
    if (inferred_type_reliable.second && isTypeReliableForTracer) {
      reliable = true;
    }
  }
  // 假设追踪器可以正确估计秩（rank），则 Shape 的输出张量应始终是可靠的
  if (output->node()->kind() == ::c10::onnx::Shape) {
    reliable = true;
  }
  // 设置输出的类型是否可靠
  ConstantValueMap::SetTypeReliable(output->debugName(), reliable);
  // 如果不可靠
  if (!reliable) {
    // 如果输出张量的类型可以转换为 TensorType
    if (auto output_tensor_type = output->type()->cast<TensorType>()) {
      // 更新输出的类型为带有符号形状的 TensorType
      output->setType(output_tensor_type->withSymbolicShapes(
          ::c10::SymbolicShape(output_tensor_type->dim())));
    }
  }
}

// 更新节点及其输出的可靠性信息
void UpdateReliable(Node* n) {
  // 检查节点的输入是否可靠或静态
  auto input_reliable = AreInputsReliableOrStatic(n);
  // 遍历节点的每一个输出，并更新其可靠性信息
  for (auto output : n->outputs()) {
    UpdateReliable(output, input_reliable);
  }
}

// 设置图的输入类型为可靠（例如，形状是否静态）
// 由于导出过程中输入不会改变，通过标记为已计算，跳过后续计算以节省时间
void SetGraphInputTypeReliable(const Graph* g) {
  // 如果尚未计算所有图输入的可靠性
  if (!ConstantValueMap::GetAllGraphInputsReliableComputed()) {
    // 遍历图的每一个输入
    for (auto graph_input : g->inputs()) {
      // 如果输入的类型尚未被标记为可靠，则标记为可靠
      if (!ConstantValueMap::HasTypeReliable(graph_input->debugName())) {
        ConstantValueMap::SetTypeReliable(graph_input->debugName(), true);
      }
    }
    // 标记所有图输入的可靠性为已计算
    ConstantValueMap::SetAllGraphInputsReliableComputed(true);
  }
}

// 对ONNX节点进行形状推断
void ONNXShapeTypeInference(
    Node* n,
    const ParamMap& params_dict,
    int opset_version) {
  // 存储Torch到ONNX输入输出名称的映射关系
  std::unordered_map<std::string, std::string> torch_to_onnx_input;
  std::unordered_map<std::string, std::string> torch_to_onnx_output;
  // 原始推断形状数据
  auto& original_shape_data = ConstantValueMap::GetInferredShapeData();
  // 推断的形状数据映射
  ShapeDataMap inferred_shape_data;
  // 符号维度映射
  auto& symbol_dim_map = ConstantValueMap::GetSymbolDimMap();
  // 维度符号映射
  auto& dim_symbol_map = ConstantValueMap::GetDimSymbolMap();

  // 设置图的输入类型为可靠
  SetGraphInputTypeReliable(n->owningGraph());

  // 输出运行ONNX形状推断的信息
  GRAPH_UPDATE(
      "Running ONNX shape inference for node: ", n->kind().toDisplayString());

  // 如果是有效的ONNX节点
  if (IsValidONNXNode(n)) {
    // 创建一个仅包含单个节点n的图
    auto n_graph = std::make_shared<Graph>();
    // 克隆节点到新图中，并进行相关参数的转换和版本设置
    auto clone_node = CloneNodeToGraph(n, n_graph, params_dict, opset_version);
    n_graph->insertNode(clone_node);

    // 将克隆节点的所有输出注册为图的输出
    for (auto output : clone_node->outputs()) {
      n_graph->registerOutput(output);
    }

    // 将原始PyTorch图的输入/输出名称映射到临时ONNX图的输入/输出名称以进行形状推断
    for (size_t i = 0; i < clone_node->inputs().size(); ++i) {
      torch_to_onnx_input[n->input(i)->debugName()] =
          clone_node->input(i)->debugName();
    }

    for (size_t i = 0; i < clone_node->outputs().size(); ++i) {
      torch_to_onnx_output[n->output(i)->debugName()] =
          clone_node->output(i)->debugName();
    }

    // 使用ONNX图的名称更新推断形状数据，仅复制所需的输入
    for (auto input : n->inputs()) {
      const auto maybe_shape = original_shape_data.find(input->debugName());
      if (maybe_shape != original_shape_data.end()) {
        const auto onnx_output_name =
            torch_to_onnx_input.find(input->debugName());
        if (onnx_output_name != torch_to_onnx_input.end()) {
          inferred_shape_data[onnx_output_name->second] = maybe_shape->second;
        }
      }
    }

    // 使用标量类型分析，无需进行低精度转换
    // 对给定的 n_graph 进行标量类型分析，不进行推理，使用指定的 opset 版本
    ScalarTypeAnalysisForONNX(n_graph, false, opset_version);

    // 输出原始的 Torch 图形表示，用于调试目的
    GRAPH_DEBUG("Original torch graph: ", n->owningGraph()->toString());

    // 输出克隆后的 Torch 图形表示，用于运行形状推断
    GRAPH_DEBUG(
        "Cloned torch graph to run shape inference: ", n_graph->toString());

    // 检查图形是否可以用于推断
    if (IsGraphValidForInference(n_graph)) {
      // TODO: 某些操作在 Peephole pass 阶段发生转换，这里的转换对这些操作是不完全的
      //       例如：ListConstruct, ListUnpack 等
      std::shared_ptr<onnx::ModelProto> model_proto;

      // 将图形转换为 ONNX Proto 格式，用于进行形状推断
      ConvertGraphToONNXProto(
          n_graph, model_proto, symbol_dim_map, dim_symbol_map, opset_version);
      GRAPH_DEBUG(
          "ONNX graph to run shape inference: ", prettyPrint(*model_proto));

      // 执行形状推断
      try {
        // TODO(#79208): 启用更多操作符以支持数据传播
        switch (n->kind()) {
          case ::c10::onnx::Shape:
          case ::c10::onnx::Gather: {
            auto* schema_registry = onnx::OpSchemaRegistry::Instance();
            onnx::ShapeInferenceOptions options{
                /*check_type=*/false,
                /*error_mode=*/false,
                /*enable_data_propagation=*/true};
            onnx::shape_inference::InferShapes(
                *model_proto, schema_registry, options, &inferred_shape_data);
            break;
          }
          default: {
            onnx::shape_inference::InferShapes(*model_proto);
            break;
          }
        }
        // 根据 ONNX Proto 更新节点输出的类型
        UpdateOutputTypeByONNXProto(
            n, clone_node, *model_proto, symbol_dim_map, dim_symbol_map);
      } catch (std::runtime_error& ex) {
        // TODO: 一旦有更统一的警告系统，将此处包括为警告信息
        GRAPH_DEBUG(
            "ONNX shape inference fails with: ",
            ex.what(),
            " on graph: ",
            n_graph->toString());
        // 捕获异常，如果不是形状推断错误或类型推断错误，则继续抛出异常
        const char shape_err[] = "ShapeInferenceError";
        const char type_err[] = "TypeInferenceError";
        if ((strstr(ex.what(), shape_err) == nullptr) &&
            (strstr(ex.what(), type_err) == nullptr)) {
          throw;
        }
      }
      // 输出经过形状推断后的 ONNX 图形表示
      GRAPH_DEBUG(
          "ONNX graph after shape inference: ", prettyPrint(*model_proto));
    }
  } else if (CustomSettype(n)) {
    // 如果节点不是 ONNX 标准的，则遍历每个输出以检查是否都具有形状
    // 即使操作不来自 ONNX，只要所有输出都具有形状，这应该是可靠的
    for (auto node_output : n->outputs()) {
      // 如果自定义的 setType 输出被正确设置，它们将在稍后的更新函数中更新为推断类型
      ConstantValueMap::SetUseInferredType(node_output->debugName(), true);
  }
}

SpecialPostProcess(n);
// 从 ONNX 形状推断中获取数据传播结果
for (const auto& output : n->outputs()) {
  // 查找推断的形状数据中是否包含当前输出节点的数据
  const auto inferred_shape_pair =
      inferred_shape_data.find(torch_to_onnx_output[output->debugName()]);
  // 如果找到了对应的推断形状数据
  if (inferred_shape_pair != inferred_shape_data.end()) {
    // 获取推断的形状数据
    const auto& inferred_shape = inferred_shape_pair->second;
    // 获取形状的维度数
    int rank = inferred_shape.dim_size();
    // 创建维度符号的向量
    std::vector<::c10::ShapeSymbol> final_shape(rank);
    // 将 ONNX 的维度转换为形状符号
    for (int i = 0; i < rank; ++i) {
      final_shape[i] = ONNXDimToShapeSymbol(
          inferred_shape.dim(i), symbol_dim_map, dim_symbol_map);
    }
    // 创建符号形状对象
    c10::SymbolicShape shape_value(final_shape);
    // 将数据传播结果存储到 shapeValueMap 中
    ConstantValueMap::SetShapeValue(output->debugName(), shape_value);
    // 在 PyTorch 图中使用原始名称，而不是在中间 ONNX 图中使用的临时名称
    // 将此信息添加回 original_shape_data 中
    original_shape_data[output->debugName()] = inferred_shape;
  }
}

// 如果节点是有效的 ONNX 节点
if (IsValidONNXNode(n)) {
  // 处理常量值映射
  ProcessConstantValueMap(n, opset_version);
  // 如果节点的类型不是 ListConstruct
  if (n->kind() != prim::ListConstruct) {
    // 遍历节点的输入
    for (auto input : n->inputs()) {
      // 如果输入节点的类型是 ListConstruct
      if (input->node()->kind() == prim::ListConstruct) {
        // 更新可靠性
        UpdateReliable(input, AreInputsReliableOrStatic(input->node()));
      }
    }
  }
}
// 更新节点的可靠性
UpdateReliable(n);

// 对于没有 ComputeConstant 逻辑的节点类型，可能具有可靠的形状，但其形状不在 ConstantValueMap 中。因此，我们需要这个逻辑来更新 ConstantValueMap。
// 更新 ConstantValueMap 中的形状常量，如果是可靠的
for (auto node_output : n->outputs()) {
  UpdateShapeConstantIfReliable(node_output);
}

// 记录 Torch 图在形状推断后的状态
GRAPH_DEBUG(
    "Torch graph after shape inference:", n->owningGraph()->toString());
}

// 定义函数 `ONNXSetDynamicInputShape`，用于设置动态输入的形状信息
void ONNXSetDynamicInputShape(
    std::shared_ptr<Graph>& graph,  // 图形对象的共享指针
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,  // 动态轴的映射，从输入名称到轴的映射
    const std::vector<std::string>& input_names) {  // 输入名称的向量
  GRAPH_UPDATE("ONNX set dynamic input shape.");  // 更新图形信息，设置动态输入形状
  GRAPH_UPDATE("dynamic axes tensor names:", [&]() {  // 更新图形信息，输出动态轴的张量名称
    std::vector<std::string> res(dynamic_axes.size());
    std::transform(
        dynamic_axes.begin(), dynamic_axes.end(), res.begin(), [](auto pair) {
          return pair.first;  // 返回每个动态轴的名称
        });
    return res;
  }());

  std::map<std::string, ::c10::ShapeSymbol> name_to_sym;  // 创建名称到形状符号的映射

  // 遍历输入名称列表
  for (const auto i : c10::irange(input_names.size())) {
    const auto& input_name = input_names[i];  // 获取当前输入名称
    // 检查是否存在动态轴的定义
    if (dynamic_axes.find(input_name) != dynamic_axes.end()) {
      auto axes_names = dynamic_axes.find(input_name)->second;  // 获取当前输入的动态轴名称映射
      TORCH_INTERNAL_ASSERT(i < graph->inputs().size());  // 内部断言，确保输入索引在图形输入范围内
      auto input_tensor_type = graph->inputs()[i]->type()->cast<TensorType>();  // 获取输入的张量类型
      if (!input_tensor_type) {
        continue;  // 如果输入类型不是张量类型，则继续下一次循环
      }

      auto shape_ref = input_tensor_type->symbolic_sizes().sizes();  // 获取输入张量的符号化大小
      TORCH_CHECK(
          shape_ref.has_value(), "Input tensor shape should have value.");  // 检查输入张量形状是否有值
      auto shape = shape_ref.value();  // 获取实际的张量形状

      // 遍历动态轴名称映射
      for (const auto& pair : axes_names) {
        const auto axis = pair.first;  // 获取轴的索引
        const auto name = pair.second;  // 获取轴的名称
        // 如果名称到符号的映射中不存在当前名称，则创建新的符号
        if (name_to_sym.find(name) == name_to_sym.end()) {
          name_to_sym[name] = ::c10::ShapeSymbol::newSymbol();  // 创建新的符号
        }
        // 检查动态形状轴是否小于当前形状的维度数
        TORCH_CHECK(
            axis < static_cast<int64_t>(shape.size()),
            "Dynamic shape axis should be no more than the shape dimension for ",
            name);
        shape[axis] = name_to_sym[name];  // 更新形状中的符号
      }

      // 更新图形输入的类型，包含符号化形状信息
      graph->inputs()[i]->setType(
          input_tensor_type->withSymbolicShapes(::c10::SymbolicShape(shape)));
    }
  }
}

// 检查节点是否有序列类型的输出
bool HasSequenceTypeOutput(Node* node) {
  if (node->kind() == ::c10::onnx::SplitToSequence ||
      node->kind() == ::c10::onnx::SequenceInsert ||
      node->kind() == ::c10::onnx::SequenceEmpty ||
      node->kind() == ::c10::onnx::SequenceErase ||
      node->kind() == ::c10::onnx::SequenceConstruct ||
      node->kind() == ::c10::onnx::Loop || node->kind() == ::c10::onnx::If)
    return true;  // 如果节点的种类是序列相关操作，则返回true
  return false;  // 否则返回false
}

// 更新输出对象的类型和形状信息
void ONNXUpdateTypeFromTensor(
    Value* graph_output,  // 图形输出的值对象
    const at::Tensor& output,  // 输出的Tensor对象
    bool onnx_shape_inference) {  // 是否进行ONNX形状推断
  if (onnx_shape_inference) {
    MergeInferredTypeAndSetMap(
        graph_output, TensorType::create(output), graph_output->type());  // 合并推断类型并设置映射
  } else {
    graph_output->inferTypeFrom(output);  // 从输出推断类型
  }
}

// 递归地查看 `output_obj` 中的元素，并将形状/类型信息分配到扁平化的图形输出中
// `outputs_index` 用于指向当前扁平化图形输出的索引。函数结束时返回更新后的 `outputs_index`
size_t ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,  // 图形对象的共享指针
    size_t outputs_index,  // 扁平化图形输出的索引
    PyObject* output_obj,  // Python对象，用于输出
    bool onnx_shape_inference,  // 是否进行ONNX形状推断
    bool is_script,
    int opset_version) {
  // 定义一个 lambda 函数 index_check，用于检查输出索引是否超出图的输出范围
  auto index_check = [&]() {
    TORCH_INTERNAL_ASSERT(
        outputs_index <= graph->outputs().size(),
        "Incorrect number of elements provided as example outputs.");
  };

  // 调用 index_check 函数，确保输出索引在有效范围内
  index_check();

  // 如果输出对象是 THPVariable 类型
  if (THPVariable_Check(output_obj)) {
    // 将 THPVariable 转换为 at::Tensor
    const at::Tensor& var = THPVariable_Unpack(output_obj);
    // 根据 Tensor 更新对应图输出节点的类型
    ONNXUpdateTypeFromTensor(
        graph->outputs().at(outputs_index), var, onnx_shape_inference);
    // 增加输出索引
    outputs_index++;
  } else if (PyTuple_Check(output_obj)) {  // 如果输出对象是 PyTuple 类型
    size_t tuple_len = PyTuple_GET_SIZE(output_obj);
    // 遍历 PyTuple 的每个元素
    for (const auto i : c10::irange(tuple_len)) {
      // 递归调用 ONNXAssignOutputShape 处理每个 PyTuple 元素
      outputs_index = ONNXAssignOutputShape(
          graph,
          outputs_index,
          PyTuple_GET_ITEM(output_obj, i),
          onnx_shape_inference,
          is_script,
          opset_version);
    }
  } else if (PyList_Check(output_obj)) {  // 如果输出对象是 PyList 类型
    const auto list_len = PyList_GET_SIZE(output_obj);
    // 如果输出索引对应的图输出节点具有序列类型
    if (HasSequenceTypeOutput(graph->outputs().at(outputs_index)->node())) {
      auto output_type = graph->outputs().at(outputs_index)->type();
      // 检查输出类型是否为 ListType
      TORCH_CHECK(
          output_type->cast<ListType>(),
          "Expected a sequence type, but received a non-iterable type in graph output index ",
          outputs_index);
      // 如果列表长度大于 0
      if (list_len > 0) {
        auto list_elem = PyList_GET_ITEM(output_obj, 0);
        // 断言第一个元素为 THPVariable 类型
        TORCH_INTERNAL_ASSERT(THPVariable_Check(list_elem));
        auto& var = THPVariable_Unpack(list_elem);
        // 遍历列表的剩余元素，检查类型是否一致
        for (const auto i : c10::irange(1, list_len)) {
          list_elem = PyList_GET_ITEM(output_obj, i);
          TORCH_INTERNAL_ASSERT(THPVariable_Check(list_elem));
          auto& new_var = THPVariable_Unpack(list_elem);
          TORCH_CHECK(
              var.scalar_type() == new_var.scalar_type(),
              "Unsupported sequence with mixed element types in model outputs. "
              "ONNX supports only sequences of elements of the same data type.");
        }
        // 获取列表元素的类型，并更新为与第一个元素相同的标量类型
        auto elem_type = graph->outputs()
                             .at(outputs_index)
                             ->type()
                             ->castRaw<ListType>()
                             ->getElementType()
                             ->cast<TensorType>();
        elem_type = elem_type->withScalarType(var.scalar_type());
        auto graph_output = graph->outputs().at(outputs_index);
        // 合并推断的类型并设置映射
        MergeInferredTypeAndSetMap(
            graph_output, graph_output->type(), ListType::create(elem_type));
      } else {
        // 如果列表为空，则保持输出节点类型不变
        graph->outputs()
            .at(outputs_index)
            ->setType(graph->outputs().at(outputs_index)->type());
      }
      // 增加输出索引
      outputs_index++;
    }
    // 如果输出索引对应的图输出节点没有序列类型，不执行任何操作
  }
  } else {
    // 当 Torch 输出为列表类型，但 ONNX 节点不是序列类型，如 prim::ListConstruct

    // 遍历输出列表中的每个元素
    for (const auto i : c10::irange(list_len)) {
      // 调用 ONNXAssignOutputShape 函数为每个元素分配输出形状
      outputs_index = ONNXAssignOutputShape(
          graph,
          outputs_index,
          PyList_GET_ITEM(output_obj, i),
          onnx_shape_inference,
          is_script,
          opset_version);
    }
  }
} else if (PyDict_Check(output_obj)) {
  // 对于字典数据类型的支持在 ONNX 中有限
  // 字典值会被展开，而键不会被保留

  // 获取字典的键值对列表
  auto* items = PyDict_Items(output_obj);
  auto unrolled_dict = py::reinterpret_borrow<py::list>(items);
  TORCH_INTERNAL_ASSERT(PyList_Check(unrolled_dict.ptr()));

  // 遍历展开后的字典列表
  for (const auto i : c10::irange(unrolled_dict.size())) {
    // 调用 ONNXAssignOutputShape 函数为每个元素分配输出形状
    outputs_index = ONNXAssignOutputShape(
        graph,
        outputs_index,
        PyList_GET_ITEM(unrolled_dict.ptr(), i),
        onnx_shape_inference,
        is_script,
        opset_version);
  }
  Py_DECREF(items);
} else if (THPUtils_checkString(output_obj)) {
  // 忽略字符串类型，因为在 ONNX 中不支持作为输出
} else if (PyNone_Check(output_obj)) {
  // 对于 None 类型的处理：
  // - 在跟踪（tracing）中，因为在 IR 图中未捕获为输出，所以忽略 None
  // - 在脚本化（scripting）中，如果在 IR 图中观察到固定的 `None` 节点，也忽略它，
  //   因为它不携带数据/信息。此外，静态的 `None` 在 ONNX IR 中也不支持。
  //   否则，输出应该是 `Optional` 类型，并应转换为 ONNX 的 `Optional`。

  // 更多上下文：
  // 原因：在跟踪过程中，我们在 torch/jit/_trace.py 中的 ONNXTracedModule.forward 中展开了输出。
  // 这意味着跟踪的 IR 图中省略了 None 输出。
  // 但是这里传入的输出是未展开的，这意味着它们包含 None 对象。理想情况下，我们应该消除这种差异。
  if (is_script && outputs_index < graph->outputs().size()) {
    // 如果在脚本化模式下且输出索引小于图中的输出数目
    if (graph->outputs().at(outputs_index)->node()->mustBeNone()) {
      if (opset_version >= 15) {
        // 如果 ONNX 版本 >= 15，则替换图中的 None 输出为 Optional
        ReplaceGraphOutputNoneWithOptional(graph, outputs_index);
        outputs_index++;
      } else {
        // 否则，直接从图中删除输出
        graph->eraseOutput(outputs_index);
      }
    } else {
      outputs_index++;
    }
  }
} else {
  // 如果输出对象的类型不被支持，抛出运行时错误
  std::string msg =
      ("Model output has unsupported type. See "
       "https://pytorch.org/docs/stable/onnx.html#types. Got type: ");
  msg += THPUtils_typename(output_obj);
  throw std::runtime_error(msg);
}

// 索引检查
index_check();

// 返回输出索引
return outputs_index;
} // namespace jit
} // namespace torch

// 创建一个 OptionalNode，用于代表 Optional 类型的节点，将其插入到图的返回节点之前
Node* ONNXOptionalNodeForNone(std::shared_ptr<Graph>& graph) {
  // 创建一个浮点类型的张量类型
  TypePtr elem_type = TensorType::get()->withScalarType(at::ScalarType::Float);
  // 在图中创建一个 Optional 类型的节点
  Node* opt_node = graph->create(::c10::onnx::Optional, 1);
  // 设置节点的类型属性为 elem_type
  opt_node->ty_(Symbol::attr("type"), elem_type);
  // 设置节点的输出为 Optional 类型的 elem_type
  opt_node->output()->setType(OptionalType::create(elem_type));
  return opt_node;
}

// 将图中指定输出索引处的 None 类型替换为 Optional 类型
void ReplaceGraphOutputNoneWithOptional(
    std::shared_ptr<Graph>& graph,
    size_t outputs_index) {
  // 创建一个表示 Optional 类型的节点
  Node* opt_node = ONNXOptionalNodeForNone(graph);
  // 将 Optional 节点插入到图的返回节点之前
  opt_node->insertBefore(graph->return_node());
  // 获取图中指定索引的输出值
  Value* graph_output = graph->outputs().at(outputs_index);
  // 替换该输出值的使用为 Optional 类型节点的输出
  // 只有最后一个值会被替换为 Optional 类型，因为 Optional 类型只影响输出前的值
  graph_output->replaceAllUsesAfterNodeWith(opt_node, opt_node->output());
  // 如果图中的输出值不是 None 类型，则将其作为输入连接到 Optional 节点，并复制元数据
  if (!graph_output->type()->cast<NoneType>()) {
    opt_node->addInput(graph_output);
    opt_node->copyMetadata(graph_output->node());
  }
}

// 根据输出、描述符等信息为图中的输出分配形状
void ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,
    at::ArrayRef<at::Tensor> outputs,
    const python::IODescriptor& desc,
    bool onnx_shape_inference,
    bool is_script,
    int opset_version) {
  size_t outputs_index = 0;
  // 将 Python 对象转换为 PyObj
  PyObject* py_obj = unflatten(outputs, desc);
  // 检查 PyObj 是否为元组类型
  TORCH_INTERNAL_ASSERT(PyTuple_Check(py_obj));

  // 调用另一个函数来为输出分配形状
  outputs_index = ONNXAssignOutputShape(
      graph,
      outputs_index,
      py_obj,
      onnx_shape_inference,
      is_script,
      opset_version);

  // 断言确保提供的示例输出数量与图的输出数量相匹配
  TORCH_INTERNAL_ASSERT(
      outputs_index == graph->outputs().size(),
      "Incorrect number of elements provided as example outputs.");

  // 释放 Python 对象
  Py_DECREF(py_obj);
  // 在执行形状推断后，输出图的状态信息
  GRAPH_DUMP("After ONNXAssignOutputShape", graph);
}

// 执行 ONNX 形状类型推断
void ONNXShapeTypeInference(
    std::shared_ptr<Graph>& graph,
    const ParamMap& params_dict,
    int opset_version) {
  // 清除常量值映射
  ConstantValueMap::ClearMaps();
  // 设置图输入类型为可靠类型
  SetGraphInputTypeReliable(graph.get());
  // 执行 ONNX 形状类型推断
  ONNXShapeTypeInference(graph->block(), params_dict, opset_version);
  // 再次清除常量值映射
  ConstantValueMap::ClearMaps();
}

// 如果值的形状是可靠的，则更新形状常量映射
void UpdateShapeConstantIfReliable(torch::jit::Value* node_output) {
  // 检查常量值映射中是否有可靠的类型
  if (ConstantValueMap::HasTypeReliable(node_output->debugName())) {
    // 获取节点输出的可靠状态
    auto reliable = ConstantValueMap::GetTypeReliable(node_output->debugName())
                        .value_or(false);
    // 如果是可靠的且没有形状，则更新形状常量映射
    if (reliable && !ConstantValueMap::HasShape(node_output->debugName())) {
      // 如果输出张量类型具有维度信息，则更新形状常量值映射
      if (auto output_tensor_type = node_output->type()->cast<TensorType>()) {
        if (output_tensor_type->dim()) {
          auto symbolic_sizes = output_tensor_type->symbolic_sizes();
          UpdateShapeConstantValueMap(node_output, symbolic_sizes);
        }
      }
    }
  }
}
```