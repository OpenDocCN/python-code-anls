# `.\pytorch\torch\csrc\lazy\core\ir_dump_util.cpp`

```py
// 引入 Torch 框架中的头文件，用于IR（Intermediate Representation）的工具函数
#include <torch/csrc/lazy/core/ir_dump_util.h>

// 引入 C10 库中的 Optional 类型
#include <c10/util/Optional.h>
// 引入 C10 库中的 irange 函数，用于遍历范围
#include <c10/util/irange.h>
// 引入 Torch 框架中的后端接口定义
#include <torch/csrc/lazy/backend/backend_interface.h>
// 引入 Torch 框架中的 lowering_context 头文件
#include <torch/csrc/lazy/backend/lowering_context.h>
// 引入 Torch 框架中的 IR（Intermediate Representation）工具函数
#include <torch/csrc/lazy/core/ir_util.h>

// 引入正则表达式标准库
#include <regex>
// 引入字符串流标准库
#include <sstream>
// 引入无序映射（哈希表）标准库
#include <unordered_map>

// Torch 框架的命名空间
namespace torch {
// Torch 框架中 lazy 模块的命名空间
namespace lazy {
// 匿名命名空间，局部定义的结构体和函数不会污染全局命名空间
namespace {

// 定义一个类型别名 NodeIdMap，用于映射 Node 指针到整数
using NodeIdMap = std::unordered_map<const Node*, size_t>;

// 定义属性标签结构体，包含属性名、属性值和位置信息
struct AttrTag {
  std::string name;               // 属性名
  std::string value;              // 属性值
  std::string::size_type pos = 0; // 位置信息，默认为 0
};

// 跳过标签分隔符的函数
std::string::size_type SkipTagSeparator(
    const std::string& node_string, // 节点字符串
    std::string::size_type pos) {   // 当前位置
  return node_string.compare(pos, 2, ", ") == 0 ? pos + 2 : pos;
}

// 解析属性标签的函数
std::optional<AttrTag> ParseAttrTag(
    const std::string& node_string,      // 节点字符串
    std::string::size_type pos) {        // 当前位置
  // 使用正则表达式匹配属性标签
  // @lint-ignore-every CLANGTIDY facebook-hte-StdRegexIsAwful
  const std::regex tag_regex("^([a-zA-Z0-9_]+)="); // 匹配属性名
  std::smatch match;
  // @lint-ignore-every CLANGTIDY facebook-hte-StdRegexIsAwful
  if (!std::regex_search(
          node_string.begin() + pos, node_string.end(), match, tag_regex)) {
    return c10::nullopt; // 如果未找到匹配，返回空值
  }

  // 获取属性值的起始位置
  std::string::size_type vpos = match[1].second - node_string.begin() + 1;
  char nested_open = -1; // 嵌套结构的开放符号，默认为 -1 表示无
  char nested_close = -1; // 嵌套结构的关闭符号，默认为 -1 表示无
  size_t nest_count = 1; // 嵌套计数器，默认为 1
  AttrTag tag;
  tag.name = match[1].str(); // 设置属性名
  for (pos = vpos; pos < node_string.size(); ++pos) {
    if (nested_open < 0) {
      // 跳过标签分隔符
      if (SkipTagSeparator(node_string, pos) != pos) {
        break;
      }
      // 根据不同的开放符号设置嵌套结构
      switch (node_string[pos]) {
        case '(':
          nested_open = node_string[pos];
          nested_close = ')';
          break;
        case '[':
          nested_open = node_string[pos];
          nested_close = ']';
          break;
        case '{':
          nested_open = node_string[pos];
          nested_close = '}';
          break;
      }
    } else if (node_string[pos] == nested_close) {
      --nest_count; // 减少嵌套计数
      if (nest_count == 0) {
        nest_count = 1;
        nested_open = nested_close = -1; // 重置嵌套符号
      }
    } else if (node_string[pos] == nested_open) {
      ++nest_count; // 增加嵌套计数
    }
  }
  tag.value = node_string.substr(vpos, pos - vpos); // 获取属性值
  tag.pos = pos; // 设置位置信息
  return tag; // 返回解析后的属性标签
}

// 生成节点 ID 映射的函数
NodeIdMap GenerateIdMap(c10::ArrayRef<const Node*> post_order) {
  NodeIdMap id_map; // 创建节点 ID 映射
  for (auto node : post_order) {
    TORCH_CHECK(id_map.emplace(node, id_map.size()).second, node->ToString()); // 插入节点及其 ID 到映射中
  }
  return id_map; // 返回节点 ID 映射
}

// 获取根节点 ID 映射的函数
std::unordered_map<const Node*, size_t> GetRootsIds(
    c10::ArrayRef<const Node*> roots) { // 根节点数组
  std::unordered_map<const Node*, size_t> roots_ids; // 创建根节点 ID 映射
  for (const auto i : c10::irange(roots.size())) {
    roots_ids[roots[i]] = i; // 将根节点及其 ID 存入映射中
  }
  return roots_ids; // 返回根节点 ID 映射
}

// 获取根节点的 ID 的可选函数
std::optional<size_t> GetRootNodeId(
    const Node* node, // 节点指针
    const std::unordered_map<const Node*, size_t>& roots_ids) { // 根节点 ID 映射
  auto it = roots_ids.find(node); // 在映射中查找节点
  if (it == roots_ids.end()) {
    return c10::nullopt; // 如果未找到，返回空值
  }
  return it->second; // 返回节点的 ID
}
// 获取给定节点的标签列表
std::vector<AttrTag> GetNodeTags(const Node* node) {
  // 将节点转换为字符串形式
  std::string node_string = node->ToString();
  // 获取节点操作符的字符串表示
  std::string op_string = node->op().ToString();
  // 在节点字符串中查找操作符的位置
  std::string::size_type pos = node_string.find(op_string);
  // 如果找不到操作符位置，抛出错误，显示节点和操作符信息
  TORCH_CHECK(pos != std::string::npos, node_string, " : ", op_string);
  // 将位置移动到操作符的末尾
  pos += op_string.size();
  // 存储标签的向量
  std::vector<AttrTag> tags;
  // 无限循环，直到没有更多标签可解析
  for (;;) {
    // 跳过标签分隔符后的位置
    pos = SkipTagSeparator(node_string, pos);
    // 解析下一个标签
    auto tag = ParseAttrTag(node_string, pos);
    // 如果解析失败，退出循环
    if (!tag) {
      break;
    }
    // 更新位置为解析后的位置，将标签添加到向量中
    pos = tag->pos;
    tags.push_back(std::move(*tag));
  }
  // 返回解析得到的标签向量
  return tags;
}

// 生成节点的 DOT 图标签
std::string GenerateDotNodeLabel(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  // 设置最大值大小为64
  static const size_t kMaxValueSize = 64;
  // 创建流以构建标签
  std::stringstream ss;
  // 添加操作符和形状信息到标签中
  ss << node->op() << "\\n" << node->shape();
  // 添加节点的标签信息到标签中
  for (auto& tag : GetNodeTags(node)) {
    ss << "\\n" << tag.name << "=";
    // 如果值的大小小于最大值大小，则添加完整值；否则添加部分值和省略号
    if (tag.value.size() < kMaxValueSize) {
      ss << tag.value;
    } else {
      ss << tag.value.substr(0, kMaxValueSize) << "...";
    }
  }
  // 获取节点的根节点 ID，并添加到标签中
  auto opt_root_id = GetRootNodeId(node, roots_ids);
  if (opt_root_id) {
    ss << "\\nROOT=" << *opt_root_id;
  }
  // 返回构建的标签字符串
  return ss.str();
}

// 生成节点的 DOT 图节点规格
std::string GenerateDotNodeSpec(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  // 创建流以构建节点规格
  std::stringstream ss;
  // 添加标签到节点规格中
  ss << "label=\"" << GenerateDotNodeLabel(node, roots_ids) << "\"";
  // 返回构建的节点规格字符串
  return ss.str();
}

// 生成文本节点的 DOT 图节点规格
std::string GenerateTextNodeSpec(const Node* node, const NodeIdMap& id_map) {
  // 创建流以构建节点规格
  std::stringstream ss;
  // 添加形状和操作信息到节点规格中
  ss << node->shapes() << " " << node->op() << "(";
  // 计算节点操作数数量
  size_t count = 0;
  // 遍历节点操作数
  for (auto& output : node->operands()) {
    // 如果不是第一个操作数，添加逗号分隔符
    if (count > 0) {
      ss << ", ";
    }
    // 添加节点操作数的 ID 到节点规格中
    ss << "%" << id_map.at(output.node);
    // 如果节点操作数的输出数量大于1，添加索引号
    if (output.node->num_outputs() > 1) {
      ss << "." << output.index;
    }
    ++count;
  }
  // 添加节点的标签信息到节点规格中
  for (auto& tag : GetNodeTags(node)) {
    ss << ", " << tag.name << "=" << tag.value;
  }
  // 返回构建的节点规格字符串
  return ss.str();
}

} // namespace

// 转换节点数组到 DOT 图字符串
std::string DumpUtil::ToDot(c10::ArrayRef<const Node*> nodes) {
  // 计算节点的后序遍历顺序
  auto post_order = Util::ComputePostOrder(nodes);
  // 将后序遍历顺序转换为 DOT 图字符串并返回
  return PostOrderToDot(post_order, nodes);
}

// 将后序遍历顺序节点数组转换为 DOT 图字符串
std::string DumpUtil::PostOrderToDot(
    c10::ArrayRef<const Node*> post_order,
    c10::ArrayRef<const Node*> roots) {
  // 获取根节点的 ID 映射
  std::unordered_map<const Node*, size_t> roots_ids = GetRootsIds(roots);
  // 生成节点 ID 映射
  NodeIdMap id_map = GenerateIdMap(post_order);
  // 创建流以构建 DOT 图
  std::stringstream ss;
  // 添加 DOT 图的头部信息
  ss << "digraph G {\n";
  // 遍历后序遍历顺序的节点，并添加节点信息到 DOT 图中
  for (auto node : post_order) {
    ss << "  node" << id_map.at(node) << " ["
       << GenerateDotNodeSpec(node, roots_ids) << "]\n";
  }
  // 逆序遍历后序遍历顺序的节点，并添加连接信息到 DOT 图中
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    const Node* node = *it;
    size_t id = id_map.at(node);
    // 省略部分连接信息的添加，因为未提供后续代码
  }
  // 返回构建的 DOT 图字符串
  return ss.str();
}
    // 对于节点的操作数数量范围进行循环遍历
    for (const auto i : c10::irange(node->operands().size())) {
      // 获取当前操作数的输出
      const Output& output = node->operand(i);
      // 构建节点之间的连接关系字符串，连接当前节点和操作数节点
      ss << "  node" << id_map.at(output.node) << " -> node" << id;
      // 如果当前节点有多个操作数
      if (node->operands().size() > 1) {
        // 添加标签，显示操作数索引
        ss << " [label=\"i=" << i;
        // 如果输出节点有多个输出
        if (output.node->num_outputs() > 1) {
          // 添加输出索引到标签
          ss << ",o=" << output.index;
        }
        ss << "\"]\n";
      } else {
        // 如果当前节点只有一个操作数
        if (output.node->num_outputs() > 1) {
          // 添加输出索引到标签
          ss << " [label=\"o=" << output.index << "\"]";
        }
        ss << "\n";
      }
    }
  }
  // 添加 Graphviz 文件尾部信息
  ss << "}\n";
  // 返回拼接好的 Graphviz DOT 格式字符串
  return ss.str();
} // 关闭命名空间 torch

} // 关闭命名空间 lazy

std::string DumpUtil::ToText(c10::ArrayRef<const Node*> nodes) {
  // 计算节点的后序遍历顺序
  auto post_order = Util::ComputePostOrder(nodes);
  // 将后序遍历顺序转换为文本表示
  return PostOrderToText(post_order, nodes);
}

std::string DumpUtil::PostOrderToText(
    c10::ArrayRef<const Node*> post_order,
    c10::ArrayRef<const Node*> roots) {
  // 获取根节点的 ID 映射
  std::unordered_map<const Node*, size_t> roots_ids = GetRootsIds(roots);
  // 生成节点到其唯一 ID 的映射
  NodeIdMap id_map = GenerateIdMap(post_order);
  // 创建字符串流对象
  std::stringstream ss;
  // 添加 IR 开始标记
  ss << "IR {\n";
  // 遍历后序遍历顺序中的每个节点
  for (auto node : post_order) {
    // 获取节点的根节点 ID（如果有的话）
    auto opt_root_id = GetRootNodeId(node, roots_ids);
    // 将节点转换为文本规范表示，并添加到字符串流中
    ss << "  %" << id_map.at(node) << " = "
       << GenerateTextNodeSpec(node, id_map);
    // 如果存在根节点 ID，则添加到字符串流中
    if (opt_root_id) {
      ss << ", ROOT=" << *opt_root_id;
    }
    // 添加节点类型信息到字符串流中
    ss << ", NodeType=" << typeid(*node).name();
    ss << "\n";
  }
  // 添加 IR 结束标记
  ss << "}\n";
  // 返回字符串流的内容作为字符串
  return ss.str();
}

std::string DumpUtil::ToBackend(
    c10::ArrayRef<Value> values,
    const BackendDevice& device) {
  // 创建降低上下文对象，用于将 IR 值降低到指定后端
  auto lowering_ctx = LoweringContext::Create("IrToBackend", device);
  // 将每个 IR 值添加到降低上下文中
  for (auto& ir_value : values) {
    lowering_ctx->AddResult(ir_value);
  }
  // 构建计算图表示
  auto computation = lowering_ctx->Build();
  // 获取后端对象，并获取计算后端的文本表示
  return getBackend()->GetComputationBackendText(computation);
}
```