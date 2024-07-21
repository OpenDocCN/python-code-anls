# `.\pytorch\torch\csrc\jit\serialization\onnx.cpp`

```
// 包含 C++ 标准库头文件
#include <c10/util/irange.h>
// 包含 Torch 序列化到 ONNX 的相关头文件
#include <torch/csrc/jit/serialization/onnx.h>
// 包含 ONNX 的核心头文件
#include <torch/csrc/onnx/onnx.h>

// 包含 C++ 标准库中用于字符串处理的头文件
#include <sstream>
#include <string>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 声明一个嵌套的匿名命名空间，使用 ONNX 的命名空间别名
namespace {
namespace onnx = ::ONNX_NAMESPACE;

// 用于缩进的字符和缩进级别乘数的常量定义
constexpr char indent_char = ' ';
constexpr size_t indent_multiplier = 2;

// 根据给定的缩进级别，返回对应的缩进字符串
std::string idt(size_t indent) {
  return std::string(indent * indent_multiplier, indent_char);
}

// 根据给定的缩进级别，返回对应的换行符加缩进字符串
std::string nlidt(size_t indent) {
  return std::string("\n") + idt(indent);
}

// 打印输出 ONNX 的 TensorProto 对象到给定的流中
void dump(const onnx::TensorProto& tensor, std::ostream& stream) {
  stream << "TensorProto shape: [";
  // 遍历 TensorProto 的维度信息并打印
  for (const auto i : c10::irange(tensor.dims_size())) {
    stream << tensor.dims(i) << (i == tensor.dims_size() - 1 ? "" : " ");
  }
  stream << "]";
}

// 打印输出 ONNX 的 TensorShapeProto 对象到给定的流中
void dump(const onnx::TensorShapeProto& shape, std::ostream& stream) {
  // 遍历 TensorShapeProto 的维度信息并打印
  for (const auto i : c10::irange(shape.dim_size())) {
    auto& dim = shape.dim(i);
    if (dim.has_dim_value()) {
      stream << dim.dim_value();
    } else {
      stream << "?";
    }
    stream << (i == shape.dim_size() - 1 ? "" : " ");
  }
}

// 打印输出 ONNX 的 TypeProto_Tensor 对象到给定的流中
void dump(const onnx::TypeProto_Tensor& tensor_type, std::ostream& stream) {
  stream << "Tensor dtype: ";
  // 打印 Tensor 类型的元素类型，如果有的话
  if (tensor_type.has_elem_type()) {
    stream << tensor_type.elem_type();
  } else {
    stream << "None.";
  }
  stream << ", ";
  stream << "Tensor dims: ";
  // 打印 Tensor 的维度信息，如果有的话
  if (tensor_type.has_shape()) {
    dump(tensor_type.shape(), stream);
  } else {
    stream << "None.";
  }
}

// 打印输出 ONNX 的 TypeProto_Optional 对象到给定的流中
void dump(const onnx::TypeProto_Optional& optional_type, std::ostream& stream) {
  stream << "Optional<";
  // 打印 Optional 类型的元素类型，如果有的话
  if (optional_type.has_elem_type()) {
    dump(optional_type.elem_type(), stream);
  } else {
    stream << "None";
  }
  stream << ">";
}

// 打印输出 ONNX 的 TypeProto_Sequence 对象到给定的流中
void dump(const onnx::TypeProto_Sequence& sequence_type, std::ostream& stream) {
  stream << "Sequence<";
  // 打印 Sequence 类型的元素类型，如果有的话
  if (sequence_type.has_elem_type()) {
    dump(sequence_type.elem_type(), stream);
  } else {
    stream << "None";
  }
  stream << ">";
}

// 根据给定的 TypeProto 类型分派打印输出到对应的流中
void dump(const onnx::TypeProto& type, std::ostream& stream) {
  if (type.has_tensor_type()) {
    dump(type.tensor_type(), stream);
  } else if (type.has_sequence_type()) {
    dump(type.sequence_type(), stream);
  } else if (type.has_optional_type()) {
    dump(type.optional_type(), stream);
  } else {
    stream << "None";
  }
}

// 打印输出 ONNX 的 ValueInfoProto 对象到给定的流中
void dump(const onnx::ValueInfoProto& value_info, std::ostream& stream) {
  stream << "{name: \"" << value_info.name() << "\", type:";
  dump(value_info.type(), stream);
  stream << "}";
}

// 前向声明，用于打印输出 ONNX 的 GraphProto 对象到给定的流中
void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent);

// 打印输出 ONNX 的 AttributeProto 对象到给定的流中
void dump(
    const onnx::AttributeProto& attr,
    std::ostream& stream,
    size_t indent) {
  stream << "{ name: '" << attr.name() << "', type: ";
  // 根据 AttributeProto 的类型不同选择打印输出不同的值
  if (attr.has_f()) {
    stream << "float, value: " << attr.f();
  } else if (attr.has_i()) {
    stream << "int, value: " << attr.i();
  } else if (attr.has_s()) {
    // 如果属性是字符串类型，则输出字符串类型及其数值
    stream << "string, value: '" << attr.s() << "'";
  } else if (attr.has_g()) {
    // 如果属性是图类型，则输出图类型及其详细内容
    stream << "graph, value:\n";
    dump(attr.g(), stream, indent + 1);
    // 输出当前缩进的换行符
    stream << nlidt(indent);
  } else if (attr.has_t()) {
    // 如果属性是张量类型，则输出张量类型及其详细内容
    stream << "tensor, value:";
    dump(attr.t(), stream);
  } else if (attr.floats_size()) {
    // 如果属性是浮点数数组，则输出浮点数数组类型及其数值
    stream << "floats, values: [";
    // 遍历并输出每个浮点数及其之间的空格
    for (const auto i : c10::irange(attr.floats_size())) {
      stream << attr.floats(i) << (i == attr.floats_size() - 1 ? "" : " ");
    }
    stream << "]";
  } else if (attr.ints_size()) {
    // 如果属性是整数数组，则输出整数数组类型及其数值
    stream << "ints, values: [";
    // 遍历并输出每个整数及其之间的空格
    for (const auto i : c10::irange(attr.ints_size())) {
      stream << attr.ints(i) << (i == attr.ints_size() - 1 ? "" : " ");
    }
    stream << "]";
  } else if (attr.strings_size()) {
    // 如果属性是字符串数组，则输出字符串数组类型及其数值
    stream << "strings, values: [";
    // 遍历并输出每个字符串及其之间的空格
    for (const auto i : c10::irange(attr.strings_size())) {
      stream << "'" << attr.strings(i) << "'"
             << (i == attr.strings_size() - 1 ? "" : " ");
    }
    stream << "]";
  } else if (attr.tensors_size()) {
    // 如果属性是张量数组，则输出张量数组类型及其每个张量的详细内容
    stream << "tensors, values: [";
    // 遍历并递归调用 dump 函数输出每个张量的详细内容
    for (auto& t : attr.tensors()) {
      dump(t, stream);
    }
    stream << "]";
  } else if (attr.graphs_size()) {
    // 如果属性是图数组，则输出图数组类型及其每个图的详细内容
    stream << "graphs, values: [";
    // 遍历并递归调用 dump 函数输出每个图的详细内容
    for (auto& g : attr.graphs()) {
      dump(g, stream, indent + 1);
    }
    stream << "]";
  } else {
    // 如果属性类型未知，则输出 "UNKNOWN"
    stream << "UNKNOWN";
  }
  // 输出大括号闭合当前属性的表示
  stream << "}";
void dump(const onnx::NodeProto& node, std::ostream& stream, size_t indent) {
  // 输出节点的类型和输入列表
  stream << "Node {type: \"" << node.op_type() << "\", inputs: [";
  // 遍历节点的输入列表
  for (const auto i : c10::irange(node.input_size())) {
    stream << node.input(i) << (i == node.input_size() - 1 ? "" : ",");
  }
  // 输出节点的输出列表
  stream << "], outputs: [";
  for (const auto i : c10::irange(node.output_size())) {
    stream << node.output(i) << (i == node.output_size() - 1 ? "" : ",");
  }
  // 输出节点的属性列表
  stream << "], attributes: [";
  for (const auto i : c10::irange(node.attribute_size())) {
    // 递归调用 dump 函数输出每个属性
    dump(node.attribute(i), stream, indent + 1);
    stream << (i == node.attribute_size() - 1 ? "" : ",");
  }
  stream << "]}";
}

void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent) {
  // 输出图的名称
  stream << idt(indent) << "GraphProto {" << nlidt(indent + 1) << "name: \""
         << graph.name() << "\"" << nlidt(indent + 1) << "inputs: [";
  // 遍历图的输入列表，调用 dump 函数输出每个输入
  for (const auto i : c10::irange(graph.input_size())) {
    dump(graph.input(i), stream);
    stream << (i == graph.input_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "outputs: [";
  // 遍历图的输出列表，调用 dump 函数输出每个输出
  for (const auto i : c10::irange(graph.output_size())) {
    dump(graph.output(i), stream);
    stream << (i == graph.output_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "value_infos: [";
  // 遍历图的 value_info 列表，调用 dump 函数输出每个 value_info
  for (const auto i : c10::irange(graph.value_info_size())) {
    dump(graph.value_info(i), stream);
    stream << (i == graph.value_info_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "initializers: [";
  // 遍历图的 initializer 列表，调用 dump 函数输出每个 initializer
  for (const auto i : c10::irange(graph.initializer_size())) {
    dump(graph.initializer(i), stream);
    stream << (i == graph.initializer_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "nodes: [" << nlidt(indent + 2);
  // 遍历图的节点列表，调用 dump 函数输出每个节点
  for (const auto i : c10::irange(graph.node_size())) {
    dump(graph.node(i), stream, indent + 2);
    if (i != graph.node_size() - 1) {
      stream << "," << nlidt(indent + 2);
    }
  }
  stream << nlidt(indent + 1) << "]\n" << idt(indent) << "}\n";
}

void dump(
    const onnx::OperatorSetIdProto& operator_set_id,
    std::ostream& stream) {
  // 输出 OperatorSetIdProto 对象的 domain 和 version
  stream << "OperatorSetIdProto { domain: " << operator_set_id.domain()
         << ", version: " << operator_set_id.version() << "}";
}

void dump(const onnx::ModelProto& model, std::ostream& stream, size_t indent) {
  // 输出 ModelProto 对象的生产者名称、领域和文档字符串
  stream << idt(indent) << "ModelProto {" << nlidt(indent + 1)
         << "producer_name: \"" << model.producer_name() << "\""
         << nlidt(indent + 1) << "domain: \"" << model.domain() << "\""
         << nlidt(indent + 1) << "doc_string: \"" << model.doc_string() << "\"";
  // 如果 ModelProto 包含图，则调用 dump 函数输出图
  if (model.has_graph()) {
    stream << nlidt(indent + 1) << "graph:\n";
    dump(model.graph(), stream, indent + 2);
  }
  // 如果 ModelProto 包含操作集导入信息，则输出操作集导入列表
  if (model.opset_import_size()) {
    stream << idt(indent + 1) << "opset_import: [";
    for (auto& opset_imp : model.opset_import()) {
      dump(opset_imp, stream);
    }
    stream << "],\n";
  }
  stream << idt(indent) << "}\n";
}
} // 结束命名空间

// 定义函数，将 ONNX 模型 Proto 对象格式化为字符串输出
std::string prettyPrint(const ::ONNX_NAMESPACE::ModelProto& model) {
  // 创建字符串流对象
  std::ostringstream ss;
  // 调用 dump 函数，将模型数据转储到字符串流中
  dump(model, ss, 0);
  // 返回字符串流中的内容作为字符串
  return ss.str();
}

} // 结束命名空间 torch::jit
```