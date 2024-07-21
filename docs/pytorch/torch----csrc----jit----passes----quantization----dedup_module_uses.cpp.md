# `.\pytorch\torch\csrc\jit\passes\quantization\dedup_module_uses.cpp`

```py
// 包含 Torch 的量化相关头文件
#include <torch/csrc/jit/passes/quantization/dedup_module_uses.h>

// 包含 Torch 的 JIT 日志相关头文件
#include <torch/csrc/jit/jit_log.h>
// 包含 Torch 的量化辅助函数头文件
#include <torch/csrc/jit/passes/quantization/helper.h>

// 标准库头文件，包含 stack 容器
#include <stack>

// Torch 的命名空间
namespace torch {
namespace jit {
// 匿名命名空间，用于实现私有类和函数
namespace {
// 模块使用去重类
class ModuleUseDeduper {
 public:
  // 构造函数，初始化模块引用
  ModuleUseDeduper(Module& module) : module_(module) {}
  
  // 执行模块使用去重操作的入口函数
  void dedup() {
    // 遍历模块中的每个方法
    for (auto& method : module_.get_methods()) {
      // 获取方法的计算图
      const auto& graph = method.graph();
      // 查找计算图中的模块使用情况
      findModuleUses(graph.get());
    }
    // 执行模块使用去重操作
    dedupModuleUses();
  }

 private:
  // 查找计算图中的模块使用情况，记录信息以备后续去重操作使用
  void findModuleUses(Graph* graph) {
    // 打印调试信息，显示正在查找模块使用情况的计算图
    GRAPH_DUMP("Finding module uses for ", graph);

    // 使用堆栈来实现深度优先搜索遍历计算图的块
    std::stack<Block*> blocks_to_visit;
    blocks_to_visit.push(graph->block());
    // 获取计算图的输入，通常第一个输入是 self
    Value* self = graph->inputs()[0];
    while (!blocks_to_visit.empty()) {
      Block* b = blocks_to_visit.top();
      blocks_to_visit.pop();
      // 遍历当前块的每个节点
      for (Node* n : b->nodes()) {
        // 对于节点中的每个子块，将其加入堆栈以待遍历
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
        }
        // 如果节点不是 prim::CallMethod 类型，则跳过
        if (n->kind() != prim::CallMethod) {
          continue;
        }
        // 获取调用的实例
        Value* instance = n->inputs()[0];
        // 获取模块访问路径，返回一个向上追溯 prim::GetAttr 的路径
        auto path = getModuleAccessPath(instance, self);

        // 如果路径为空，表示在 self 上调用方法，不需要去重 self 的使用
        if (path.empty()) {
          continue;
        }
        // 将实例和其对应的路径存入映射表中
        value_to_path_map_[instance] = path;
        // 查找子模块并插入到唯一模块集合中
        auto m = findChildModule(module_, path);
        // 如果无法将模块插入唯一模块集合中，说明之前已经有该模块的使用，需要重写该使用
        if (!unique_modules_.insert(m._ivalue()).second) {
          uses_to_rewrite_.push_back(instance);
          // 打印调试信息，显示需要重写的使用
          GRAPH_DEBUG("Found use to rewrite: ", instance->debugName());
        }
      }
    }
  }

  // 执行模块使用去重操作，根据之前记录的信息进行操作
  void dedupModuleUses() {
    // 遍历需要重写的使用列表
    for (Value* v : uses_to_rewrite_) {
      // 获取该值对应的路径
      const auto& path = value_to_path_map_.at(v);
      // 查找对应的子模块
      const auto& m = findChildModule(module_, path);
      // 将子模块的克隆添加到父模块中
      const auto& child_name = addChildModule(module_, m, path);
      // 断言，确认该值的节点类型为 prim::GetAttr
      TORCH_INTERNAL_ASSERT(v->node()->kind() == prim::GetAttr);
      // 修改 GetAttr 调用中的名称
      auto original_name = v->node()->s(attr::name);
      v->node()->s_(attr::name, child_name);
      // 更新调试信息，显示模块使用去重的变化
      GRAPH_UPDATE(
          "Module use dedup: changing use of original module ",
          original_name,
          " to ",
          child_name);
    }
  }

  // 模块引用
  Module& module_;
  // 存储值到路径映射的容器
  std::unordered_map<Value*, std::vector<std::string>> value_to_path_map_;
  // 唯一模块集合
  std::unordered_set<IValue> unique_modules_;
  // 需要重写的使用列表
  std::vector<Value*> uses_to_rewrite_;
};

// 匿名命名空间结束
} // namespace

// Torch JIT 模块使用去重的入口函数，接收模块引用并执行去重逻辑
void DeduplicateModuleUses(Module& module) {
  // 创建模块使用去重器实例并执行去重操作
  ModuleUseDeduper deduper(module);
  deduper.dedup();
}

// JIT 命名空间结束
} // namespace jit
// Torch 命名空间结束
} // namespace torch
  }

  // 向父模块中添加子模块
  std::string addChildModule(
      Module& module,
      const Module& child_module,
      const std::vector<std::string>& path) {
    // 断言路径不为空，至少包含一个元素
    TORCH_INTERNAL_ASSERT(
        !path.empty(), "path must have at least one element.");
    // 找到路径对应的叶子子模块的父模块
    auto parent_of_leaf = findChildModule(
        module, std::vector<std::string>(path.begin(), path.end() - 1));

    // 子模块的原始名称
    const std::string& original_name = path[path.size() - 1];
    int uid = 0;
    // 生成唯一的子模块名称，以确保不重复
    std::string child_name = original_name + "_" + std::to_string(uid++);
    while (parent_of_leaf.hasattr(child_name)) {
      child_name = original_name + "_" + std::to_string(uid++);
    }
    // 在父模块中注册深拷贝的子模块，并返回子模块的名称
    parent_of_leaf.register_module(child_name, child_module.deepcopy());
    return child_name;
  }

  // 主模块实例
  Module module_;
  // 映射，从模块实例的值到子模块名称路径列表，从顶层模块开始，例如 ["sub1", "sub2", "relu"]
  // 同时也是调用 `getModuleAccessPath` 的值的缓存
  std::unordered_map<Value*, std::vector<std::string>> value_to_path_map_;
  // 用于存储被使用在图中的唯一模块集合
  std::unordered_set<ModulePtr> unique_modules_;
  // 表示需要重写为克隆模块实例使用的值的集合
  std::vector<Value*> uses_to_rewrite_;
}; // 结束匿名命名空间

} // 结束命名空间

// 对模块中的使用进行去重
void DedupModuleUses(Module& module) {
  // 创建模块使用去重器对象
  ModuleUseDeduper d(module);
  // 执行去重操作
  d.dedup();
}

} // 结束命名空间 jit
} // 结束命名空间 torch
```