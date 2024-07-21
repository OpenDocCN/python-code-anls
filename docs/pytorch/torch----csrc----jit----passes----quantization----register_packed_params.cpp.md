# `.\pytorch\torch\csrc\jit\passes\quantization\register_packed_params.cpp`

```
// 包含标准库头文件：堆栈
#include <stack>

// 包含 PyTorch 头文件
#include <ATen/ATen.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/quantization/register_packed_params.h>

// 定义 torch 命名空间下的 jit 命名空间
namespace torch {
namespace jit {

// 实现内部匿名命名空间，用于私有函数和局部变量
namespace {

// 检查节点是否为预包装节点
bool isPrepackNode(Node* n) {
  return (
      n->kind() == Symbol::fromQualString("quantized::linear_prepack") ||
      n->kind() == Symbol::fromQualString("quantized::conv1d_prepack") ||
      n->kind() == Symbol::fromQualString("quantized::conv2d_prepack") ||
      n->kind() == Symbol::fromQualString("quantized::conv3d_prepack") ||
      n->kind() ==
          Symbol::fromQualString("quantized::conv_transpose1d_prepack") ||
      n->kind() ==
          Symbol::fromQualString("quantized::conv_transpose2d_prepack"));
}

// 查找量化前权重的值和名称
std::pair<Value*, std::string> findFPWeight(Node* prepack_node) {
  TORCH_CHECK(isPrepackNode(prepack_node));
  Node* n = nullptr;
  n = prepack_node->input(0)->node();
  bool is_quantize_node =
      (n->kind() == Symbol::fromQualString("aten::quantize_per_tensor") ||
       n->kind() == Symbol::fromQualString("aten::quantize_per_channel"));
  TORCH_CHECK(
      is_quantize_node,
      "Input to prepack node must be output of weight quantization.");
  // 量化节点的第一个输入是 FP32 权重
  n = n->input(0)->node();
  bool is_getattr_node = (n->kind() == prim::GetAttr);
  if (is_getattr_node) {
    return {n->input(0), n->s(attr::name)};
  }
  return {nullptr, "AttributeDoesNotExist"};
}

} // namespace

// 拼接路径字符串向量的函数
std::string joinPaths(const std::vector<std::string>& paths) {
  std::string path;
  for (const auto& p : paths) {
    path.append(p).append(".");
  }
  return path;
}

// 注册预打包参数的函数，必须在常量折叠之后运行
std::unordered_set<std::string> RegisterPrePackParams(
    Module& m,
    const std::string& method_name,
    const PrePackParamFilterFn& is_packed_param,
    const std::string& attr_prefix) {
  int64_t uid = 0; // 使用方法名来创建唯一标识符
  auto graph = m.get_method(method_name).graph();
  std::stack<Block*> blocks_to_visit;
  std::unordered_set<Node*> nodes_to_delete;
  blocks_to_visit.push(graph->block());
  std::string attr_name_base =
      attr_prefix + "_" + method_name + "_ondevice_ptq_packed_weight_";
  std::unordered_set<std::string> packed_param_names;

  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    // 这里需要填充代码，但是当前的 while 循环体为空
    }
  }
  return packed_param_names;
}

} // namespace jit
} // namespace torch
```