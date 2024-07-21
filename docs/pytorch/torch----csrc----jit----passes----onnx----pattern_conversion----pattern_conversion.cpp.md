# `.\pytorch\torch\csrc\jit\passes\onnx\pattern_conversion\pattern_conversion.cpp`

```py
// 包含头文件：c10/util/irange.h，提供范围迭代功能
#include <c10/util/irange.h>
// 包含头文件：torch/csrc/jit/passes/dead_code_elimination.h，定义了消除死代码的优化
#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 包含头文件：torch/csrc/jit/passes/erase_number_types.h，定义了消除数字类型的优化
#include <torch/csrc/jit/passes/erase_number_types.h>
// 包含头文件：torch/csrc/jit/passes/onnx.h，提供ONNX相关功能
#include <torch/csrc/jit/passes/onnx.h>
// 包含头文件：torch/csrc/jit/passes/onnx/pattern_conversion/common.h，提供ONNX模式转换的通用功能
#include <torch/csrc/jit/passes/onnx/pattern_conversion/common.h>
// 包含头文件：torch/csrc/jit/passes/onnx/pattern_conversion/pattern_conversion.h，提供ONNX模式转换功能
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_conversion.h>

// 包含头文件：ATen/ScalarOps.h，提供标量操作功能
#include <ATen/ScalarOps.h>

// 包含头文件：iostream，提供输入输出流功能
#include <iostream>

// 命名空间：torch::jit，定义了PyTorch JIT编译器的功能
namespace torch {
namespace jit {

// 命名空间：匿名命名空间，定义了用于将inplace index_put转换为ONNX的功能
namespace {

// 创建指定维度的尺寸值节点
Value* CreateSizeOfDim(Value* input, int64_t dim, Node* insertBefore) {
  auto graph = input->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto size = graph->insert(aten::size, {input, dim});
  return size;
}

// 将select操作转换为索引操作
Value* ConvertSelectToIndex(Value* index, Node* insertBefore) {
  auto graph = insertBefore->owningGraph();
  WithInsertPoint guard(insertBefore);
  return graph->insert(aten::unsqueeze, {index, 0});
}

// 将slice操作转换为索引操作
Value* ConvertSliceToIndex(Node* slice, Value* size, Node* insertBefore) {
  auto graph = slice->owningGraph();
  WithInsertPoint guard(insertBefore);
  TORCH_INTERNAL_ASSERT((slice->inputs()).size() == 5);
  auto start = slice->inputs()[2];
  auto end = slice->inputs()[3];
  auto step = slice->inputs()[4];
  auto index =
      graph->insert(aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
  auto sliced_index_n = graph->create(
      aten::slice,
      {index,
       graph->insertConstant(
           scalar_to_tensor(at::Scalar(0)), c10::nullopt, slice->scope()),
       start,
       end,
       step});

  sliced_index_n->copyMetadata(insertBefore);
  auto sliced_index = sliced_index_n->insertBefore(insertBefore)->output();
  return sliced_index;
}

// 结构体：转换后的索引结构
struct ConvertedIndex {
  ConvertedIndex(Value* index, c10::Symbol orig_node_kind)
      : index(index), orig_node_kind(orig_node_kind) {}

  Value* index = nullptr;
  c10::Symbol orig_node_kind;
};

// 哈希映射：维度到转换后的索引结构的映射
std::unordered_map<int64_t, ConvertedIndex> MergeSliceAndSelectToIndices(
    Graph* graph,
    Node* index_put_node,
    const std::vector<Node*>& slice_and_select_nodes,
    Value* orig_data,
    const py::dict& env) {
  std::unordered_map<int64_t, ConvertedIndex> dim_index_map;

  // 遍历已获取的slice和select节点，并将它们转换为索引张量
  // 记录当前slice或select节点应用的维度
  int64_t cur_dim = 0;
  int64_t dim_offset = 0;
  const auto orig_tensor_indices = index_put_node->input(1)->node()->inputs();
  for (auto it = slice_and_select_nodes.rbegin();
       it != slice_and_select_nodes.rend();
       ++it) {
    auto node = *it;
    // select操作不保留维度，
    // 这会为后续的slice和select节点创建偏移量
    // 注意：不能依赖于get(attr::dim)，因为操作不再匹配模式
    // 获取节点的第二个输入，即表示维度的节点，获取其值并转换为 int64_t 类型
    int64_t dim = node->inputs().at(1)->node()->t(attr::value).item().toLong();

    // 如果维度值小于 0，则需要进行自动推断
    if (dim < 0) {
      // 获取原始数据在环境中的 Python 值
      auto py_value = env[py::cast(orig_data)];
      // 将 Python 值转换为 TorchScript 值
      Value* value = py_value.cast<Value*>();
      // 获取 TorchScript 值的类型，并期望其为 TensorType
      auto input_type = value->type()->expect<TensorType>();
      // 如果输入类型包含维度信息，则计算维度的有效值
      if (input_type->dim().has_value()) {
        auto rank = static_cast<int64_t>(input_type->dim().value());
        // 计算原始张量的有效维度索引，减去由选择操作符创建的偏移量
        dim = dim + rank - dim_offset;
      } else {
        // 如果无法确定输入的维度，则输出错误信息
        std::cerr
            << "Error: Cannot export ellipsis indexing for input "
            << "of unknown rank. Check https://pytorch.org/docs/stable/onnx.html#indexing"
            << "for details.";
      }
    }
    // 将计算后的维度值加上偏移量
    dim = dim + dim_offset;

    // 处理直到当前维度小于计算得到的维度的所有维度
    while (cur_dim < dim) {
      // 处理跳过的维度，这些维度由...或张量索引创建
      if (cur_dim - dim_offset >= (int64_t)orig_tensor_indices.size() ||
          index_put_node->input(1)
              ->node()
              ->input(cur_dim - dim_offset)
              ->node()
              ->mustBeNone()) {
        // 创建当前维度大小的张量
        auto size = CreateSizeOfDim(orig_data, cur_dim, index_put_node);
        // 在当前节点位置插入 arange 操作，创建索引张量
        WithInsertPoint guard(index_put_node);
        auto index_tensor = graph->insert(
            aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
        // 将当前维度及其对应的索引信息添加到映射中
        dim_index_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cur_dim),
            std::forward_as_tuple(index_tensor, aten::slice));
      } else if (cur_dim - dim_offset < (int64_t)orig_tensor_indices.size()) {
        // 将原始张量的索引添加到映射中
        dim_index_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cur_dim),
            std::forward_as_tuple(
                orig_tensor_indices[cur_dim - dim_offset], aten::index));
      }
      cur_dim++;
    }

    // 断言当前维度与计算得到的维度相等
    TORCH_INTERNAL_ASSERT(cur_dim == dim);

    // 如果节点的类型是 aten::slice
    if (node->kind() == aten::slice) {
      // 创建当前维度大小的张量
      auto size = CreateSizeOfDim(orig_data, dim, index_put_node);
      // 将 slice 操作转换为索引张量
      auto index_tensor = ConvertSliceToIndex(node, size, index_put_node);
      // 将当前维度及其对应的索引信息添加到映射中
      dim_index_map.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(dim),
          std::forward_as_tuple(index_tensor, aten::slice));
    } 
    // 如果节点的类型是 aten::select
    else if (node->kind() == aten::select) {
      // 将 select 操作转换为索引张量
      auto index_tensor = ConvertSelectToIndex(node->input(2), index_put_node);
      // 将当前维度及其对应的索引信息添加到映射中
      dim_index_map.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(dim),
          std::forward_as_tuple(index_tensor, aten::select));
      // 增加维度偏移量
      dim_offset++;
    }
  } else {
    // 如果节点类型不是aten::slice或aten::select，则引发错误
    TORCH_CHECK(
        false,
        node->kind().toDisplayString(),
        " Expected aten::slice or aten::select.");
  }

  // 增加当前维度计数
  cur_dim++;
}

// 将剩余的原始张量索引映射到当前维度
while (cur_dim - dim_offset < (int64_t)orig_tensor_indices.size()) {
  dim_index_map.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(cur_dim),
      std::forward_as_tuple(
          orig_tensor_indices[cur_dim - dim_offset], aten::index));
  // 增加当前维度计数
  cur_dim++;
}

// 检查每个维度是否有对应的索引张量
TORCH_INTERNAL_ASSERT((int64_t)dim_index_map.size() == cur_dim);
// 返回维度索引映射
return dim_index_map;
// 将切片和选择操作符转换为张量索引。
// 根据它们的轴重塑张量索引。
// 例如：               x[1:3, 0, ind1, ind2] = y
// 切片索引形状：       [2,   1, 1 ]
// 选择索引形状：      [     1, 1 ]
// ind1 形状：          [        _ ]
// ind2 形状：          [        _ ]
// 这里的 _ 是 ind1 和 ind2 的原始大小。
// ind1 和 ind2 都是一维张量，因为目前我们只支持一维张量索引。
std::vector<Value*> ReshapeToAdvancedIndexingFormat(
    Graph* graph,                                  // 图对象，用于创建新节点
    Node* index_put_node,                          // 索引放置节点，当前正在处理的节点
    std::unordered_map<int64_t, ConvertedIndex>& dim_index_map) {
  std::vector<Value*> indices;                    // 存储转换后的索引值的向量

  size_t min_index_dim = dim_index_map.size();    // 最小索引维度的大小
  size_t max_index_dim = 0;                       // 最大索引维度的大小
  size_t tensor_ind_count = 0;                    // 张量索引的数量
  for (const auto i : c10::irange(dim_index_map.size())) {
    auto index_i = dim_index_map.find(i);         // 查找索引 i 的信息
    TORCH_INTERNAL_ASSERT(index_i != dim_index_map.end());  // 确保找到了对应的索引信息
    if (index_i->second.orig_node_kind == aten::index) {  // 如果是索引操作
      if (i < min_index_dim)
        min_index_dim = i;                        // 更新最小索引维度
      if (i > max_index_dim)
        max_index_dim = i;                        // 更新最大索引维度
      tensor_ind_count++;                         // 增加张量索引计数
    }
  }

  // 检查索引维度是否连续
  if (((max_index_dim - min_index_dim + 1) != tensor_ind_count) &&
      tensor_ind_count != 0) {
    TORCH_CHECK(
        false,
        "Only consecutive 1-d tensor indices are supported in exporting aten::index_put to ONNX.",
        "Check https://pytorch.org/docs/stable/onnx.html#indexing for details");
  }

  size_t tensor_ind_offset = tensor_ind_count == 0 ? 0 : tensor_ind_count - 1;  // 张量索引的偏移量
  WithInsertPoint guard(index_put_node);          // 设置插入点为当前节点之前
  for (const auto i : c10::irange(dim_index_map.size())) {
    size_t ind_size = 0;                          // 索引的大小
    auto index_i = dim_index_map.find(i);         // 查找索引 i 的信息
    TORCH_INTERNAL_ASSERT(index_i != dim_index_map.end());  // 确保找到了对应的索引信息
    Value* index = index_i->second.index;         // 获取索引的值
    switch (index_i->second.orig_node_kind) {
      case aten::select:
      case aten::slice: {
        if (i < min_index_dim) {
          ind_size = dim_index_map.size() - tensor_ind_offset - i;  // 计算切片和选择操作的索引大小
        } else {
          ind_size = dim_index_map.size() - i;    // 计算切片和选择操作的索引大小
        }
        break;
      }

      case aten::index: {
        ind_size = dim_index_map.size() - tensor_ind_offset - min_index_dim;  // 计算索引的大小
        break;
      }
      default:
        TORCH_CHECK(
            false, "Unexpected node kind ", index_i->second.orig_node_kind);  // 非预期的节点类型错误
    }

    // 根据索引大小决定是否需要插入 view 操作节点
    if (ind_size != 1) {
      std::vector<int64_t> view_shape(ind_size, 1);  // 创建视图形状向量
      view_shape[0] = -1;                         // 设置视图形状的第一个维度
      auto unsqueezed_index = graph->insert(aten::view, {index, view_shape});  // 插入视图操作节点
      indices.emplace_back(unsqueezed_index);     // 将新节点加入索引向量
    } else {
      indices.emplace_back(index);                // 将索引加入索引向量
    }
  }

  return indices;                                 // 返回转换后的索引向量
}

// 追溯与索引放置节点相关联的所有切片和选择节点，并将它们转换为相关的索引。
// 例如，对于 x[1:3, 0] = update 的 IR
//    ...
//    %8 : Float(2, 4) = aten::slice(%0, %4, %5, %6, %7)
//    ...
//    %11 : Float(2) = aten::select(%8, %9, %10)
//    ...
//    %13 : Tensor?[] = prim::ListConstruct()
//    ...
// 检查节点是否为 ONNX 占位符并且是 "index_put" 或 "index_put_" 操作
std::vector<Value*> ConvertIndexPutToONNX(
    Block* new_block,
    Node* old_node,
    py::dict& env,
    py::set& values_in_env) {
  // 如果节点不是 "onnx::Placeholder" 或者名称不是 "index_put" 或 "index_put_"，则直接返回空向量
  if (old_node->kind() != Symbol::fromQualString("onnx::Placeholder") ||
      (old_node->s(attr::name) != "index_put" &&
       old_node->s(attr::name) != "index_put_")) {
    // 返回一个空字典
    return {};
  }

  // 断言：旧节点的块数量为1
  TORCH_INTERNAL_ASSERT(old_node->blocks().size() == 1);
  // 获取旧节点所属的图
  auto old_graph = old_node->owningGraph();
  // 获取子块，这里假设旧节点的块列表中只有一个块
  auto subblock = old_node->blocks()[0];
  // 获取子块中最后一个节点的前一个节点
  auto index_put_node = subblock->nodes().back()->prev();

  // 查找与索引操作符相关联的切片和选择操作符
  // 例如，x[1:3, 0] = y 会生成一个切片操作符(1:3)和一个选择操作符(0)
  std::vector<Node*> slice_and_select_nodes =
      IndexingPatternFinder::FetchSliceAndSelect(index_put_node);
  // 获取最后一个节点，如果找不到切片和选择节点，则为索引操作节点本身
  Node* last_node = !slice_and_select_nodes.empty()
      ? slice_and_select_nodes.back()
      : index_put_node;
  // 更新内部块的输入，源自外部的数据
  last_node->replaceInput(0, old_node->input(0));
  // 获取原始数据作为输入
  Value* orig_data = last_node->input(0);

  // 将切片和选择操作符转换为索引
  std::unordered_map<int64_t, ConvertedIndex> dim_index_map =
      MergeSliceAndSelectToIndices(
          old_graph, index_put_node, slice_and_select_nodes, orig_data, env);

  // 将索引重塑为高级索引格式
  std::vector<Value*> indices =
      ReshapeToAdvancedIndexingFormat(old_graph, index_put_node, dim_index_map);

  // 创建带有转换索引的新的index_put节点
  const auto list_indices =
      old_graph->createList(OptionalType::ofTensor(), indices)
          ->insertBefore(index_put_node)
          ->output();
  auto new_index_put_node = old_graph->create(
      aten::index_put,
      {orig_data,
       list_indices,
       index_put_node->input(2),
       index_put_node->input(3)});
  new_index_put_node->insertBefore(index_put_node);
  new_index_put_node->copyMetadata(index_put_node);
  auto new_index_put = new_index_put_node->output();
  new_index_put->copyMetadata(index_put_node->output());
  index_put_node->output()->replaceAllUsesWith(new_index_put);

  // 将aten类型转换为onnx类型
  EraseNumberTypesOnBlock(subblock);
  EliminateDeadCode(
      subblock,
      true,
      DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);

  // 将刚创建的所有新aten节点转换为onnx节点
  // 新的onnx节点会追加到new_block的末尾
  for (auto at_n : subblock->nodes()) {
    if (at_n == subblock->param_node() || at_n == subblock->return_node()) {
      continue;
    }

    NodeToONNX(
        at_n,
        new_block,
        torch::onnx::OperatorExportTypes::ONNX,
        env,
        values_in_env);
  }

  // 找到与索引_put的aten输出对应的onnx输出
  std::vector<Value*> outs;
  for (auto o : subblock->return_node()->inputs()) {
    auto py_value = env[py::cast(o)];
    Value* value = py_value.cast<Value*>();
    outs.emplace_back(value);
  }
  // 返回输出值向量
  return outs;
} // 结束命名空间 jit

} // 结束命名空间 torch

// 将子块中的模式从其它形式转换为标准形式的函数
std::vector<Value*> ConvertPatternFromSubblock(
    Block* new_block,                    // 新块
    Node* old_node,                     // 旧节点
    py::dict& env,                      // 环境字典
    py::set& values_in_env) {           // 环境中的值集合

  std::vector<Value*> res;              // 结果向量

  // 如果旧节点不是 "onnx::Placeholder" 类型，则直接返回空结果
  if (old_node->kind() != Symbol::fromQualString("onnx::Placeholder")) {
    return res;
  }

  // 获取操作名称
  auto op_name = old_node->s(attr::name);

  // 如果操作名称是 "index_put" 或 "index_put_"，则调用相应的转换函数
  if (op_name == "index_put" || op_name == "index_put_") {
    res = ConvertIndexPutToONNX(new_block, old_node, env, values_in_env);
  }

  return res;  // 返回结果向量
}
```