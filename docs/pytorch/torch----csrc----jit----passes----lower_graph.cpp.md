# `.\pytorch\torch\csrc\jit\passes\lower_graph.cpp`

```
// 包含 Torch 库中的头文件 lower_graph.h，提供了图降级操作的功能
#include <torch/csrc/jit/passes/lower_graph.h>

// 包含 Torch 的对象 API 头文件以及前端错误报告头文件
#include <torch/csrc/jit/api/object.h>
#include <torch/csrc/jit/frontend/error_report.h>

// 包含 Torch 的内联函数头文件，以及自定义类相关头文件
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/custom_class.h>

// 包含 C++ 标准库中的无序映射 unordered_map 头文件
#include <unordered_map>

// Torch 的命名空间声明
namespace torch {
namespace jit {

// 定义结构体 Slot，用于表示一个对象的指定偏移处的槽位信息
struct Slot {
  c10::intrusive_ptr<c10::ivalue::Object> obj; // 指向 IValue 对象的智能指针
  size_t offset; // 槽位的偏移量

  // 定义相等运算符，用于比较两个 Slot 结构是否相等
  bool operator==(const Slot& other) const {
    return (this->obj == other.obj && this->offset == other.offset);
  }
};

// lower_graph 函数的声明，用于将模块图进行降级处理
// 移除第一个模块参数，并替换其参数/属性访问为额外的输入 Slot，用于 ONNX 导出
static std::pair<std::shared_ptr<Graph>, std::vector<Slot>> lower_graph(
    const ModulePtr& self, // 模块指针，表示当前模块对象
    Graph& g_, // 要处理的图对象的引用
    size_t self_offset = 0); // 第一个模块参数的偏移量，默认为 0

// lower_graph 函数的实现开始
static std::pair<std::shared_ptr<Graph>, std::vector<Slot>> lower_graph(
    const ModulePtr& self,
    Graph& g_,
    size_t self_offset /* = 0 */) {
  std::shared_ptr<Graph> g = g_.copy(); // 复制输入的图对象 g_
  
  // 内联处理图 g 中的方法/函数调用，以减少函数调用的层次
  Inline(*g);

  std::vector<Slot> extra_ivalues; // 存储额外输入 Slot 的向量

  // 定义 Slot 的哈希函数结构 SlotHash
  struct SlotHash {
    std::size_t operator()(const Slot& slot) const {
      auto obj_hash = std::hash<c10::ivalue::Object*>{}(slot.obj.get());
      auto offset_hash = std::hash<size_t>{}(slot.offset);
      return c10::hash_combine(obj_hash, offset_hash); // 合并对象指针和偏移量的哈希值
    }
  };

  // 使用 SlotHash 定义无序映射，将 Slot 映射到偏移量的索引
  std::unordered_map<Slot, size_t, SlotHash> slot_to_offset;

  // 定义结构体 ToScan，用于存储需要扫描的模块及其节点信息
  struct ToScan {
    ModulePtr mod; // 模块指针
    Node* n; // 节点指针
    size_t offset; // 偏移量
  };

  std::vector<ToScan> to_scan; // 需要扫描的模块信息向量
  std::vector<Node*> to_clean; // 应在最后清除的节点向量，表示不再需要的节点

  // 定义 lambda 函数 getOrAddSlot，用于获取或添加 Slot 对应的输入值
  auto getOrAddSlot = [&](const Slot& slot) -> Value* {
    auto it = slot_to_offset.find(slot); // 查找 Slot 是否已存在
    if (it != slot_to_offset.end()) { // 如果找到
      size_t ivalues_start = g->inputs().size() - extra_ivalues.size();
      return g->inputs().at(ivalues_start + it->second); // 返回对应输入值
    }
    extra_ivalues.emplace_back(slot); // 否则将 Slot 添加到 extra_ivalues 中
    slot_to_offset[slot] = extra_ivalues.size() - 1; // 记录新 Slot 的偏移量
    return g->addInput()->setType(slot.obj->getSlot(slot.offset).type()); // 添加输入节点
  };

  auto self_value = g->inputs().at(self_offset); // 获取 self 参数的输入值

  // 遍历 self_value 的使用情况，将需要扫描的节点加入 to_scan 中
  for (Use use : self_value->uses()) {
    to_scan.emplace_back(ToScan{self, use.user, use.offset});
  }

  // 开始处理需要扫描的节点
  while (!to_scan.empty()) {
    auto e = to_scan.back();
    to_scan.pop_back();

    // 如果当前节点是 prim::fork，需要递归降级其中的子图
    if (e.n->kind() == prim::fork) {
      auto subgraph = e.n->g(attr::Subgraph); // 获取 fork 节点的子图
      std::vector<Slot> new_slots;
      std::tie(subgraph, new_slots) = lower_graph(e.mod, *subgraph, e.offset); // 递归降级子图
      e.n->g_(attr::Subgraph, subgraph); // 更新原 fork 节点的子图
      for (const Slot& slot : new_slots) {
        e.n->addInput(getOrAddSlot(slot)); // 为 fork 节点添加输入 Slot
      }
      e.n->removeInput(e.offset); // 移除原 fork 节点中的输入
      continue;
    }
    
    // 如果当前节点是 prim::PythonOp，则抛出错误报告，不支持导出 Python 方法
    if (e.n->kind() == prim::PythonOp) {
      throw ErrorReport(e.n->sourceRange()) << "Couldn't export Python method.";
    }
    // 检查节点是否是属性访问节点，如果不是则抛出错误报告
    if (e.n->kind() != prim::GetAttr) {
      throw ErrorReport(e.n->sourceRange())
          << "temporary: the only valid use of a module is looking up an "
             "attribute but found "
          << *e.n;
    }
    // 获取属性名对应的槽位索引
    size_t slot_idx = e.mod->type()->getAttributeSlot(e.n->s(attr::name));
    // 获取属性值
    auto iv = e.mod->getSlot(slot_idx);
    // 如果输出节点的类型可以转换为 ClassTypePtr
    if (ClassTypePtr c = e.n->output()->type()->cast<ClassType>()) {
      // 如果该类是一个模块
      if (c->is_module()) {
        // 遍历输出节点的所有使用
        for (Use use : e.n->output()->uses()) {
          // 将需要扫描的信息加入列表中
          to_scan.emplace_back(ToScan{iv.toObject(), use.user, use.offset});
        }
        // 将当前节点加入需要清理的列表中
        to_clean.emplace_back(e.n);
        // 继续处理下一个节点
        continue;
      }
    }
    // 用模块中指定槽位的值替换输出节点的所有使用
    e.n->output()->replaceAllUsesWith(getOrAddSlot({e.mod, slot_idx}));
    // 销毁当前节点
    e.n->destroy();
  }

  // 清理所有需要销毁的节点
  while (!to_clean.empty()) {
    Node* n = to_clean.back();
    // 断言当前节点没有被使用
    AT_ASSERT(!n->hasUses());
    // 销毁当前节点
    n->destroy();
    // 弹出已处理节点
    to_clean.pop_back();
  }
  // 断言 self_value 没有被使用
  AT_ASSERT(!self_value->hasUses());
  // 从图中擦除输入 self_offset
  g->eraseInput(self_offset);

  // 返回图和额外的值对
  return std::make_pair(std::move(g), std::move(extra_ivalues));
} // 结束 namespace jit

} // 结束 namespace torch

static std::vector<IValue> loadTensors(const std::vector<Slot>& slots) {
  // 初始化一个空的结果向量
  std::vector<IValue> result;
  // 预留空间以容纳所有 slots 的数据
  result.reserve(slots.size());
  // 遍历每个 Slot
  for (const Slot& slot : slots) {
    // 获取 Slot 中的对象
    auto obj = slot.obj->getSlot(slot.offset);
    // 如果对象是 Tensor 类型
    if (obj.isTensor()) {
      // 将 Tensor 添加到结果向量中
      result.emplace_back(obj.toTensor());
    } else {
      // 如果对象不是 Tensor 类型，则解包量化打包的 Tensor
      auto type = obj.type();
      // 检查对象的类型是否是量化打包参数的基类之一
      TORCH_CHECK(
          (type ==
           getCustomClass(
               "__torch__.torch.classes.quantized.Conv2dPackedParamsBase")) ||
              (type ==
               getCustomClass(
                   "__torch__.torch.classes.quantized.Conv3dPackedParamsBase")) ||
              (type ==
               getCustomClass(
                   "__torch__.torch.classes.quantized.LinearPackedParamsBase")),
          "Unknown type ",
          type->repr_str(),
          " encountered in graph lowering. This type is not supported in ONNX export.");
      // 将解包后的对象的状态添加到结果向量中
      result.emplace_back(
          script::Object(obj.toObject()).run_method("__getstate__"));
    }
  }
  // 返回包含所有解包后数据的结果向量
  return result;
}

std::pair<std::shared_ptr<Graph>, std::vector<IValue>> LowerGraph(
    Graph& graph,
    const ModulePtr& self) {
  // 调用 lower_graph 函数降低图形
  auto result = lower_graph(self, graph);
  // 返回包含降低后图形和解包后数据的 pair
  return std::make_pair(result.first, loadTensors(result.second));
}
```