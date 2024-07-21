# `.\pytorch\torch\csrc\jit\passes\quantization\insert_observers.cpp`

```py
// 引入必要的头文件
#include <c10/util/irange.h>
#include <torch/csrc/jit/passes/quantization/insert_observers.h>

#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/inline_fork_wait.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

#include <memory>  // 引入内存管理相关头文件
#include <stack>   // 引入栈数据结构头文件
#include <string>  // 引入字符串处理头文件
#include <utility> // 引入实用工具头文件

namespace torch {  // 定义命名空间 torch
namespace jit {    // 定义命名空间 jit，嵌套在 torch 内部

using ModuleQConfigMap = std::unordered_map<ModulePtr, std::optional<QConfig>>;  // 定义 Module 到可选 QConfig 的映射类型别名

namespace {  // 匿名命名空间，限定在当前编译单元内部

struct OptionalQConfigHash {  // 定义一个哈希函数结构 OptionalQConfigHash
  inline size_t operator()(const std::optional<QConfig>& qconfig_opt) const {  // 定义哈希函数调用操作符
    if (qconfig_opt.has_value()) {  // 如果 qconfig_opt 有值
      const auto& m1 = std::get<0>(*qconfig_opt);  // 获取 qconfig_opt 的第一个元素
      const auto& m2 = std::get<1>(*qconfig_opt);  // 获取 qconfig_opt 的第二个元素
      constexpr int CONST = 7;  // 定义一个常量 CONST 为 7
      return std::hash<Module>()(m1) + CONST * std::hash<Module>()(m2);  // 返回哈希值
    }
    return 0;  // 若 qconfig_opt 无值，则返回 0
  }
};
using QConfigTypePtrMap =  // 定义一个哈希映射，将可选 QConfig 映射到类型指针 TypePtr
    std::unordered_map<std::optional<QConfig>, TypePtr, OptionalQConfigHash>;
using NameModuleVector = std::vector<std::pair<std::string, Module>>;  // 定义名称与模块的向量类型别名
using OptionalModuleVector = std::vector<std::optional<Module>>;  // 定义可选模块的向量类型别名
using ModuleMethodVector = std::vector<std::pair<Module, std::string>>;  // 定义模块与字符串对的向量类型别名
using graph_rewrite_helper::PatternInfo;  // 使用 graph_rewrite_helper 命名空间中的 PatternInfo
using graph_rewrite_helper::replaceConvolutionWithAtenConv;  // 使用 graph_rewrite_helper 命名空间中的 replaceConvolutionWithAtenConv 函数

// 帮助函数
void fillQConfigMap(
    const Module& module,
    const QConfigDict& qconfig_dict,
    ModuleQConfigMap& map,
    const std::string& key = "",
    const std::optional<QConfig>& parent_qconfig = c10::nullopt) {
  std::optional<QConfig> qconfig;  // 定义一个可选的 QConfig 对象
  if (qconfig_dict.find(key) != qconfig_dict.end()) {  // 如果 qconfig_dict 中存在指定的键 key
    GRAPH_DEBUG("Got module config for key:", key);  // 记录调试信息，获取了指定键的模块配置
    qconfig = qconfig_dict.at(key);  // 获取 qconfig_dict 中键为 key 的值
  } else {
    GRAPH_DEBUG("Inheriting qconfig from parent module:", key);  // 记录调试信息，从父模块继承 qconfig
    qconfig = parent_qconfig;  // 从父 QConfig 继承当前 QConfig
  }
  map[module._ivalue()] = qconfig;  // 将模块的 _ivalue 与 QConfig 映射存入 map 中

  for (const NameModule& s : module.named_children()) {  // 遍历模块的命名子模块列表
    std::string child_key;  // 定义子模块的键名
    if (key.empty()) {  // 如果当前键名为空
      child_key = s.name;  // 子模块的键名为其名称 s.name
    } else {
      child_key = key + "." + s.name;  // 否则，构造子模块的键名为 key.s.name
    }
    fillQConfigMap(s.value._ivalue(), qconfig_dict, map, child_key, qconfig);  // 递归调用填充 QConfig 映射
  }
}

Module getObserverModuleFor(Value* v, const QConfig& qconfig) {
  return isWeight(v) ? std::get<1>(qconfig) : std::get<0>(qconfig);  // 根据值 v 判断是否为权重，返回相应的观察器模块
}

// 辅助类
  /** 根据模块的 QConfig 映射克隆模块，用于处理这样一种情况：
   *  有两个模块实例共享相同的 ClassType，但配置了不同的 QConfig
   *  代码从 https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp 复制并修改
   *  inplace 选项表示是否深拷贝张量
   *  如果 inplace 为 true，则克隆的模块将与原始模型共享张量，而不是深拷贝它们
   */
  Module clone(
      const Module& module,
      const ModuleQConfigMap& module_qconfig_map,
      bool inplace = false) {
    std::unordered_map<TypePtr, QConfigTypePtrMap> type_remap; // 创建类型指针到 QConfig 映射的哈希表
    IValue::HashIdentityIValueMap memo; // 创建 IValue 的哈希映射 memo
    // 调用 clone_impl 函数进行实际的克隆操作
    return clone_impl(
        module, module_qconfig_map, type_remap, inplace, std::move(memo));
  }

 private:
  Module clone_impl(
      const Module& module,
      const ModuleQConfigMap& module_qconfig_map,
      std::unordered_map<TypePtr, QConfigTypePtrMap>& type_remap,
      bool inplace,
      IValue::HashIdentityIValueMap memo) {
    auto qconfig = module_qconfig_map.at(module._ivalue()); // 获取模块的 QConfig
    auto type = module.type(); // 获取模块的类型
    // 创建一个新的 _ivalue 在相同的编译单元中
    // 因为现在我们有共享的 ClassType，所以在克隆过程中需要保留共享的 ClassType
    // 因此，我们首先使用 type 和 qconfig 检查是否已经克隆了该类型，如果是，则创建一个具有克隆的 ClassType 的新模块，
    // 如果不是，则创建一个新的模块和一个新的 ClassType。
    bool type_already_cloned = type_remap.find(type) != type_remap.end() &&
        type_remap.at(type).find(qconfig) != type_remap.at(type).end();
    Module r;
    if (type_already_cloned) {
      // 如果之前已经克隆过类类型，将重用它
      Module new_module(
          module._ivalue()->compilation_unit(),
          type_remap.at(type).at(qconfig)->cast<ClassType>());
      r = new_module;
    } else {
      Module new_module(
          *type->name(), module._ivalue()->compilation_unit(), true);
      r = new_module;
      // 将新的类型映射到 QConfig 映射表中
      type_remap[type][module_qconfig_map.at(module._ivalue())] = r.type();
    }
    // 复制槽位。如果槽位是模块，则递归克隆它。
    size_t N = type->numAttributes();
    // 遍历从 0 到 N-1 的范围，使用 auto 关键字依次获取每个值
    for (const auto i : c10::irange(N)) {
      // 从模块的 IValue 获取指定索引 i 的值
      IValue s = module._ivalue()->getSlot(i);
      // 获取类型的属性名字
      std::string attr_name = type->getAttributeName(i);
      // 获取类型的属性类型
      TypePtr attr_type = type->getAttribute(i);
      // 检查属性类型是否为模块类型
      if (attr_type->is_module()) {
        // 将 IValue 转换为 Module 类型的原始对象
        const Module& orig = Module(s.toObject());
        // 克隆模块对象 orig，使用给定的参数和选项
        Module cloned =
            clone_impl(orig, module_qconfig_map, type_remap, inplace, memo);

        // 注意：为什么需要手动在对象上设置属性而不是使用 register_module？
        // 因为属性可能是模块接口类型，并且仍然持有一个模块对象。
        // register_module 无法正确为该属性设置类型，因此必须手动执行此操作。
        // 如果属性是接口类型，在同一编译单元中，新克隆实例将共享类型，
        // 因为它仅包含函数模式列表。
        r.type()->addOrCheckAttribute(
            attr_name,
            attr_type->cast<ClassType>() ? cloned.type() : attr_type);
        // 设置新克隆模块的属性值
        r._ivalue()->setAttr(attr_name, cloned._ivalue());
      } else {
        // 如果不是模块类型，则根据 inplace 参数选择深度复制或浅复制 IValue
        r.register_attribute(
            type->getAttributeName(i),
            type->getAttribute(i),
            inplace ? s : s.deepcopy(memo),
            type->is_parameter(i),
            type->is_buffer(i));
      }
    }

    // 如果类型尚未被克隆过，则克隆其中的常量和方法
    if (!type_already_cloned) {
      // 遍历类型的所有常量，添加到克隆后的类型中
      for (size_t i = 0; i < type->numConstants(); ++i) {
        r.type()->addConstant(type->getConstantName(i), type->getConstant(i));
      }
      // 克隆类型的方法，并重新映射类型到克隆后的类型
      for (auto& fn : type->methods()) {
        clone_method(module, r, *fn, module_qconfig_map, type_remap);
      }
      // 执行 __setstate__(__getstate__()) 方法初始化自定义类成员
      if (auto setstate_method = r.find_method("__setstate__")) {
        auto getstate_method = r.find_method("__getstate__");
        TORCH_INTERNAL_ASSERT(getstate_method, "expect __getstate__");
        auto state = (*getstate_method)(Stack{});
        (*setstate_method)(Stack{std::move(state)});
      }
    }
    // 返回克隆后的模块对象 r
    return r;
  }

  // 重新映射类型，修改指定代码块中的 self 值和模块对象
  void remapTypes(
      Block* block,
      Value* self,
      const Module& source,
      Module& target,
      const ModuleQConfigMap& module_qconfig_map,
      const std::function<TypePtr(TypePtr, std::optional<QConfig>)>&
          type_remap_fn) {
    // remapTypes 不支持将模块作为方法参数传入的情况，需要在函数外进行 %self 的重映射
    // 因为这种情况下需要更全面的分析来决定模块的 QConfig
    // 所以该方法只处理特定的 remap 操作，不支持模块作为参数
    // 遍历每个输入块的索引，从第二个开始（第一个是self），检查是否有输入参数是Object类型，如果是则抛出错误
    for (size_t i = 1; i < block->inputs().size(); ++i) {
      TORCH_CHECK(
          !block->inputs()[i]->type()->cast<ClassType>(),
          "We don't support quantizing methods that has Object as arguments");
    }
    // 遍历块中的每个节点
    for (Node* node : block->nodes()) {
      // 如果节点是调用方法或获取属性
      if (node->kind() == prim::CallMethod || node->kind() == prim::GetAttr) {
        // 获取节点的第一个输入，通常是实例对象
        Value* instance = node->inputs()[0];
        // 获取调用模块的Optional信息
        auto child_opt = getInvokedModuleOpt(source, node, self);
        // 如果存在有效值
        if (child_opt.has_value()) {
          // 获取模块对应的量化配置
          auto qconfig = module_qconfig_map.at(child_opt->_ivalue());
          // 重新映射实例对象的类型
          instance->setType(type_remap_fn(instance->type(), qconfig));
        }
      }
      // 不对输出进行重新映射，在CallMethod中进行模块类型的重新映射，不支持返回自方法或函数的模块类型重新映射
      // 遍历节点的子块，递归地对子块进行类型映射
      for (Block* sub_block : node->blocks()) {
        remapTypes(
            sub_block, self, source, target, module_qconfig_map, type_remap_fn);
      }
      // 遍历节点的属性名称
      for (Symbol name : node->attributeNames()) {
        // 如果属性的类型是单个图
        if (node->kindOf(name) == AttributeKind::g) {
          // 对该图进行类型映射
          remapTypes(
              node->g(name).get(),
              source,
              target,
              module_qconfig_map,
              type_remap_fn);
        } 
        // 如果属性的类型是图列表
        else if (node->kindOf(name) == AttributeKind::gs) {
          // 对每个图进行类型映射
          for (const auto& g : node->gs(name)) {
            remapTypes(
                g.get(), source, target, module_qconfig_map, type_remap_fn);
          }
        }
      }
    }
  }

  // 为图中的所有节点进行类型映射
  void remapTypes(
      Graph* graph,
      const Module& source,
      Module& target,
      const ModuleQConfigMap& module_qconfig_map,
      const std::function<TypePtr(TypePtr, std::optional<QConfig>)>&
          type_remap_fn) {
    // 调用重载的remapTypes函数，从图的第一个输入开始映射类型
    remapTypes(
        graph->block(),
        graph->inputs()[0],
        source,
        target,
        module_qconfig_map,
        type_remap_fn);
  }

  // 克隆方法到目标模块，包括类型映射和self的重映射
  void clone_method(
      const Module& source,
      Module& target,
      const Function& method,
      const ModuleQConfigMap& module_qconfig_map,
      const std::unordered_map<TypePtr, QConfigTypePtrMap>& type_remap) {
    // 定义类型重映射的lambda函数
    auto type_remap_fn = [&](TypePtr type_ptr,
                             const std::optional<QConfig>& qconfig) {
      // 如果类型存在于映射中
      if (type_remap.find(type_ptr) != type_remap.end()) {
        const auto& qconfig_map = type_remap.at(type_ptr);
        // 如果配置存在于类型映射中，则返回映射后的类型
        if (qconfig_map.find(qconfig) != qconfig_map.end()) {
          return qconfig_map.at(qconfig);
        }
      }
      // 否则直接返回原始类型
      return type_ptr;
    };
    // 将方法转换为图函数，并复制其图
    auto graph = toGraphFunction(method).graph()->copy();
    // 对图进行类型映射
    remapTypes(graph.get(), source, target, module_qconfig_map, type_remap_fn);
    // 重新映射self参数的类型
    graph->inputs()[0]->setType(target.type());
    // 只支持%self作为函数参数中的Module类型
    auto schema_type_remap_fn = [&](TypePtr type_ptr) {
      return type_remap_fn(
          std::move(type_ptr), module_qconfig_map.at(source._ivalue()));
    };
    // 闭合匿名代码块，其中包含函数体的定义和实现

    auto schema =
        method.getSchema().cloneWithRemappedTypes(schema_type_remap_fn);
    // 从 method 获取其 schema，并通过指定的类型重映射函数来克隆 schema

    const auto this_method_name =
        c10::QualifiedName(*target.type()->name(), method.name());
    // 构造当前方法的限定名称，由目标类型的名称和方法名称组成

    auto copied = target._ivalue()->compilation_unit()->create_function(
        this_method_name, std::move(graph));
    // 在目标对象的 IValue 上的编译单元中创建一个新函数，使用当前方法的名称和移动语义传递的图形数据

    target.type()->addMethod(copied);
    // 将新创建的函数添加到目标对象的类型中

    copied->setSchema(std::move(schema));
    // 设置新创建函数的 schema，使用移动语义传递 schema
  }
};

/**
 * Helper class for inserting observers into a neural network module.
 * This class assists in preprocessing, graph analysis, and observer insertion.
 */
class InsertObserversHelper {
 public:
  /**
   * Constructor for InsertObserversHelper.
   * Initializes with a map of module configurations and the quantization type.
   */
  explicit InsertObserversHelper(
      const ModuleQConfigMap& map,
      QuantType quant_type)
      : module_qconfig_map_(map), quant_type_(quant_type) {}

  // TODO: replace (module, method_name) with graph?
  /**
   * Preprocesses the module to clean up the graph from tracing.
   * @param module The module to preprocess.
   * @param method_name The name of the method to preprocess.
   */
  void preprocess(Module& module, const std::string& method_name);

  /**
   * Fills the boundary value map between caller input/output and callee input/output.
   * This map helps navigate through the graph to find observers for specific values.
   * @param module The module to analyze.
   * @param method_name The name of the method to fill boundary values.
   */
  void fillBoundaryValueMap(Module& module, const std::string& method_name);

  /**
   * Analyzes the module's graph and records necessary information for observer insertion.
   * @param module The module to analyze.
   * @param method_name The name of the method to analyze.
   */
  void analyze(Module& module, const std::string& method_name);

  /**
   * Removes activation observers from the module.
   */
  void removeActivationObservers();

  /**
   * Recursively inserts observers into the module's method.
   * Observers are inserted in the order of node execution to avoid duplication.
   * @param module The module to insert observers into.
   * @param method_name The name of the method to insert observers.
   * @param is_entry_point Indicates if the method is the forward method of the top-level module.
   * @param graph_observed_values Set of values already observed in the graph.
   * @return A tuple containing vectors of optional observer modules for input and output,
   *         and a vector of indexes indicating whether the output value is observed.
   */
  std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>>
  insertObservers(
      Module& module,
      const std::string& method_name,
      bool is_entry_point = false,
      std::unordered_set<Value*> graph_observed_values =
          std::unordered_set<Value*>());

  /**
   * Sets whether to insert reset observer method and the associated method name.
   * @param insert_reset_observer_method Whether to insert reset observer method.
   * @param method_name The name of the method to insert reset observer.
   */
  void setInsertResetObserverMethod(
      bool insert_reset_observer_method,
      const std::string& method_name) {
    insert_reset_observer_method_ = insert_reset_observer_method;
  
    // Implementation details of setting the insert reset observer method
    // for a specific method in the module.
  }

  // Other private members and methods can be defined here if needed.

 private:
  ModuleQConfigMap module_qconfig_map_;  // Map of module configurations
  QuantType quant_type_;  // Type of quantization
  bool insert_reset_observer_method_;  // Flag to insert reset observer method
};
    reset_observer_method_name_ = "reset_observers_" + method_name;
  }

 private:
  // 在给定的方法名后加上固定前缀，用于重置观察器方法的名称
  std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>>
  insertObserversFor(
      Block* block,
      script::Module& module,
      // 这是一个引用，因为在一个块中插入值的观察器时，它也可能在另一个块中被观察到，
      // 我们不希望为同一个值插入多个观察器
      std::unordered_set<Value*>& block_observed_values,
      bool is_entry_point = false,
      bool is_user_defined_function = false);

  // 将值 v 记录为“准备观察”，通过将其存储在 values_to_observe 中进行记录。
  // 如果 v 是延迟观察模式的一部分，则根据延迟规则记录 v 的后代。
  // 观察器将在稍后的阶段通过读取此函数创建的状态来插入。
  void recordObserved(
      Value* v,
      const Module& observer_module,
      std::unordered_map<Value*, Module>& values_to_observe,
      std::unordered_set<Value*>& block_observed_values);

  // 获取调用了指定方法名的模块方法列表
  ModuleMethodVector getInvokedMethods(
      Module& module,
      const std::string& method_name);

  // 检查值 v 是否需要量化，根据提供的量化配置信息 qconfig 进行判断
  bool valueNeedsToBeQuantized(Value* v, const QConfig& qconfig);

  // 检查值 v 是否已被观察，根据块观察值集合进行判断
  bool isObserved(
      Value* v,
      const std::unordered_set<Value*>& block_observed_values) {
    return block_observed_values.count(v) || observed_values_.count(v);
  }

  // 填充从值到相应观察器模块的映射，这个映射在 insertObservers 中用于实际插入观察器到模块中
  void fillValueObserverMap(Module& module, const std::string& method_name);

  // 克隆观察器模块并将其添加到原始模块中，同时插入调用观察器前向函数的调用
  void insertObserverFor(
      Value* v,
      Module& module,
      const Module& observer_module,
      NameModuleVector& observer_name_and_modules);

  // 向模块中插入观察器重置最小最大值的方法，使用观察器名称和模块向量
  void insertObserverResetMinMax(
      Module& module,
      const NameModuleVector& observer_name_and_modules);

  // 根据 fillBoundaryValueMap 和 fillValueObserverMap 创建的状态，
  // 返回一个配置好的用于值的观察器（如果需要的话）
  std::optional<Module> getObserverFor(Value* v);

  // 根据 fillPassThroughValueMap 创建的状态，传播应该从输入到输出传递的观察属性
  void propagateObservedProperty(
      Value* output,
      std::unordered_set<Value*>& block_observed_values);

  // 对于 cat/add/mul，只有在它们的输入被观察时才会观察它们的输出
  bool shouldObserve(
      Node* n,
      const std::unordered_set<Value*>& block_observed_values,
      QuantType quant_type) {
    // 检查节点输出的使用是否可以量化，例如 cat 后跟线性操作
    for (Value* v : n->outputs()) {
      for (const auto& use : v->uses()) {
        if (useQuantizable(use, quant_type)) {
          return true;
        }
      }
    }
    // 如果节点 n 是单输入量化传播操作，则检查其第一个输入是否被观察
    if (isPropagateQuantSingleInputOp(n)) {
      // 返回第一个输入是否被观察，使用给定的块观察数值
      return isObserved(n->input(0), block_observed_values);
    } else if (isPropagateQuantBinaryOp(n)) {
      // 如果节点 n 是二输入量化传播操作，则检查两个输入都应为张量并且被观察。
      // 这里没有做的一个检查是 !isScalar(isObserved(n->input(1), block_observed_values)
      // 确保输入 1 不是标量，因为对于当前规则，加法/乘法的标量张量输入不会被观察，我们可以在这里省略这个检查。
      return isObserved(n->input(0), block_observed_values) &&
          isObserved(n->input(1), block_observed_values);
    }
    // 默认情况下，返回 true，表示应插入观察器方法
    return true;
  }

  void delayObservingValuesInPattern(Graph& graph, const PatternInfo& pattern);

  // 找到并标记已知的模式，如 conv-relu（和其他模式），在这些模式中我们不应该在中间插入观察器
  void addValuesToDelayObservation(
      const Module& module,
      const std::string& method_name);

  // 填充从值到可以传递观察属性的值列表的映射
  void fillPassThroughValueMap(const std::shared_ptr<Graph>& graph);

  // 插入重置观察器方法
  bool insertResetObserverMethod() {
// 定义模式 nn.Linear + nn.ReLU
const PatternInfo nn_linear_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%input, %linear, %relu):
    %first_output = prim::CallMethod[name="forward"](%linear, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
    {is_linear_module, is_relu_module});

// 定义模式 nn.Linear + F.relu
const PatternInfo nn_linear_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%input, %linear, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%linear, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    {is_linear_module, is_functional_relu});

// 定义模式 nn.Linear + aten::relu
const PatternInfo nn_linear_aten_relu = PatternInfo::parse_from_str(
    R"(
graph(%input, %linear, %relu):
    %first_output = prim::CallMethod[name="forward"](%linear, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
    {is_linear_module});

// 定义模式 nn.Linear + aten::relu_
const PatternInfo nn_linear_aten_relu_ = PatternInfo::parse_from_str(
    R"(
graph(%input, %linear, %relu):
    %first_output = prim::CallMethod[name="forward"](%linear, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
    {is_linear_module});

// 定义模式 aten::linear + nn.ReLU
const PatternInfo aten_linear_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%input, %weight, %bias, %relu):
    %first_output = aten::linear(%input, %weight, %bias)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
    {is_relu_module});

// 定义模式 aten::linear + F.relu
const PatternInfo aten_linear_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%input, %weight, %bias, %relu, %inplace):
    %first_output = aten::linear(%input, %weight, %bias)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    {is_functional_relu});

// 定义模式 aten::linear + aten::relu
const PatternInfo aten_linear_aten_relu = PatternInfo::parse_from_str(
    R"(
graph(%input, %weight, %bias):
    %first_output = aten::linear(%input, %weight, %bias)
    %second_output = aten::relu(%first_output)
    return (%second_output) )");

// 定义模式 aten::linear + aten::relu_
const PatternInfo aten_linear_aten_relu_ = PatternInfo::parse_from_str(
    R"(
graph(%input, %weight, %bias):
    %first_output = aten::linear(%input, %weight, %bias)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )");

// 定义模式 nn.Conv1d + F.relu
const PatternInfo nn_conv1d_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    {is_conv1d_module, is_functional_relu});

// 定义模式 nn.Conv1d + nn.ReLU
const PatternInfo nn_conv1d_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv, %relu):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
    {is_conv1d_module, is_relu_module});
    // 调用名为 "forward" 的方法，并使用 %conv 和 %input 作为参数，将结果保存到 %first_output 中
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    // 调用名为匹配正则表达式 "forward\d*" 的方法，并使用 %relu 和 %first_output 作为参数，将结果保存到 %second_output 中
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    // 返回 %second_output 的值，表示函数执行的结果
    return (%second_output) )",
      // 使用 is_conv1d_module 和 is_relu_module 作为条件，构建一个模式信息对象
      {is_conv1d_module, is_relu_module});

  // 从字符串中解析出模式信息对象，该字符串可能包含模式匹配的规则
  const PatternInfo nn_conv1d_aten_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output)


// 定义一个图模式，用于描述一个包含 relu 激活函数的 Conv1D 模块
const PatternInfo nn_conv1d_aten_relu_ = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
    {is_conv1d_module});

// 定义一个图模式，用于描述一个包含 functional relu 激活函数的 Conv2D 模块
const PatternInfo nn_conv2d_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    {is_conv2d_module, is_functional_relu});

// 定义一个图模式，用于描述一个包含 nn.Module relu 激活函数的 Conv2D 模块
const PatternInfo nn_conv2d_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv, %relu):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
    {is_conv2d_module, is_relu_module});

// 定义一个图模式，用于描述一个包含 aten::relu 激活函数的 Conv2D 模块
const PatternInfo nn_conv2d_aten_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
    {is_conv2d_module});

// 定义一个图模式，用于描述一个包含 aten::relu_ 激活函数的 Conv2D 模块
const PatternInfo nn_conv2d_aten_relu_ = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
    {is_conv2d_module});

// 定义一个图模式，用于描述一个包含 functional relu 激活函数的 Conv3D 模块
const PatternInfo nn_conv3d_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    {is_conv3d_module, is_functional_relu});

// 定义一个图模式，用于描述一个包含 nn.Module relu 激活函数的 Conv3D 模块
const PatternInfo nn_conv3d_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv, %relu):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
    {is_conv3d_module, is_relu_module});

// 定义一个图模式，用于描述一个包含 aten::relu 激活函数的 Conv3D 模块
const PatternInfo nn_conv3d_aten_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %conv, %input):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
    {is_conv3d_module});

// 定义一个图模式，用于描述一个包含 aten::relu_ 激活函数的 Conv3D 模块
const PatternInfo nn_conv3d_aten_relu_ = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
      {is_conv3d_module});


    // 返回第二个输出，并传递 is_conv3d_module 作为参数
    return (%second_output) )",
      {is_conv3d_module});


  const PatternInfo add_nn_relu = PatternInfo::parse_from_str(
      R"(


  // 创建一个名为 add_nn_relu 的 PatternInfo 对象，从给定的字符串解析模式信息
  const PatternInfo add_nn_relu = PatternInfo::parse_from_str(
      R"(
const PatternInfo add_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %a, %b, %alpha, %relu, %inplace):
    %first_output = aten::add(%a, %b, %alpha)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    {aten_add_alpha_is_one, is_functional_relu});



// 定义模式信息对象 add_f_relu，用于解析特定的图模式字符串
const PatternInfo add_f_relu = PatternInfo::parse_from_str(
    // 定义包含占位符的图模式，用于匹配特定的计算图
    R"(
graph(%self, %a, %b, %alpha, %relu, %inplace):
    %first_output = aten::add(%a, %b, %alpha)  // 使用aten库的add函数进行张量相加操作，结果存储在first_output中
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)  // 调用指定的函数（通过relu参数传递）对first_output进行处理，结果存储在second_output中
    return (%second_output) )",  // 返回second_output作为结果
    {aten_add_alpha_is_one, is_functional_relu});  // 指定用于匹配该模式的附加条件，如alpha参数为1，relu是一个功能性的激活函数



const PatternInfo inplace_add_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %a, %b, %alpha, %relu):
    %first_output = aten::add_(%a, %b, %alpha)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
    {aten_add_alpha_is_one, is_relu_module});



// 定义模式信息对象 inplace_add_nn_relu，用于解析特定的图模式字符串
const PatternInfo inplace_add_nn_relu = PatternInfo::parse_from_str(
    // 定义包含占位符的图模式，用于匹配特定的计算图
    R"(
graph(%self, %a, %b, %alpha, %relu):
    %first_output = aten::add_(%a, %b, %alpha)  // 使用aten库的inplace版本的add_函数进行张量相加操作，结果存储在first_output中
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)  // 调用指定的模块的forward方法处理first_output，结果存储在second_output中
    return (%second_output) )",  // 返回second_output作为结果
    {aten_add_alpha_is_one, is_relu_module});  // 指定用于匹配该模式的附加条件，如alpha参数为1，relu是一个模块化的激活函数


（以下代码类似地进行注释，按照相同的格式进行解释）


const PatternInfo add_aten_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %alpha):
    %first_output = aten::add(%a, %b, %alpha)
    %second_output = aten::relu(%first_output)
    return (%second_output) )");

const PatternInfo add_aten_relu_ = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %alpha):
    %first_output = aten::add(%a, %b, %alpha)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )");

const PatternInfo inplace_add_aten_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %alpha):
    %first_output = aten::add_(%a, %b, %alpha)
    %second_output = aten::relu(%first_output)
    return (%second_output) )");

const PatternInfo inplace_add_aten_relu_ = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %alpha):
    %first_output = aten::add_(%a, %b, %alpha)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )");

const PatternInfo nn_bn2d_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %batchnorm, %relu):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
    {is_batchnorm2d_module, is_relu_module});

const PatternInfo nn_bn2d_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %batchnorm, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    {is_batchnorm2d_module, is_functional_relu});

const PatternInfo nn_bn2d_aten_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %batchnorm):
    // 省略部分代码



// 定义模式信息对象 nn_bn2d_aten_relu，用于解析特定的图模式字符串
const PatternInfo nn_bn2d_aten_relu = PatternInfo::parse_from_str(
    // 定义包含占位符的图模式，用于匹配特定的计算图
    R"(
graph(%self, %input, %batchnorm):
    // 省略部分代码
    // 调用 batchnorm 对象的 forward 方法，并将结果存储在 first_output 中
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    // 对 first_output 执行 ReLU 激活函数，将结果存储在 second_output 中
    %second_output = aten::relu(%first_output)
    // 返回经过 ReLU 激活后的输出作为结果
    return (%second_output) )",
      // 定义模式信息，用于匹配 nn.BatchNorm2d 模块的 aten::relu 模式
      {is_batchnorm2d_module});
  
  // 解析字符串形式的模式信息，用于匹配 nn.BatchNorm2d 模块的 aten::relu 模式
  const PatternInfo nn_bn2d_aten_relu_ = PatternInfo::parse_from_str(
      R"(
// 定义了一个名为 nn_bn2d_nn_relu 的模式信息对象，用于匹配特定的图模式
const PatternInfo nn_bn2d_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %batchnorm):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
    // 用于匹配是 BatchNorm2D 模块的断言函数
    {is_batchnorm2d_module});

// 定义了一个名为 nn_bn3d_nn_relu 的模式信息对象，用于匹配特定的图模式
const PatternInfo nn_bn3d_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %batchnorm, %relu):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
    // 用于匹配是 BatchNorm3D 模块和 ReLU 模块的断言函数
    {is_batchnorm3d_module, is_relu_module});

// 定义了一个名为 nn_bn3d_f_relu 的模式信息对象，用于匹配特定的图模式
const PatternInfo nn_bn3d_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %batchnorm, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    // 用于匹配是 BatchNorm3D 模块和 functional ReLU 的断言函数
    {is_batchnorm3d_module, is_functional_relu});

// 定义了一个名为 nn_bn3d_aten_relu 的模式信息对象，用于匹配特定的图模式
const PatternInfo nn_bn3d_aten_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %batchnorm):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
    // 用于匹配是 BatchNorm3D 模块的断言函数
    {is_batchnorm3d_module});

// 定义了一个名为 nn_bn3d_aten_relu_ 的模式信息对象，用于匹配特定的图模式
const PatternInfo nn_bn3d_aten_relu_ = PatternInfo::parse_from_str(
    R"(
graph(%self, %input, %batchnorm):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
    // 用于匹配是 BatchNorm3D 模块的断言函数
    {is_batchnorm3d_module});

// 定义了一个名为 mul_nn_relu 的模式信息对象，用于匹配特定的图模式
const PatternInfo mul_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %a, %b, %relu):
    %first_output = aten::mul(%a, %b)
    %second_output = prim::CallMethod[name="forward"](%relu, %first_output)
    return (%second_output) )",
    // 用于匹配是 ReLU 模块的断言函数
    {is_relu_module});

// 定义了一个名为 mul_f_relu 的模式信息对象，用于匹配特定的图模式
const PatternInfo mul_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %a, %b, %relu, %inplace):
    %first_output = aten::mul(%a, %b)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    // 用于匹配是 functional ReLU 的断言函数
    {is_functional_relu});

// 定义了一个名为 inplace_mul_nn_relu 的模式信息对象，用于匹配特定的图模式
const PatternInfo inplace_mul_nn_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %a, %b, %relu):
    %first_output = aten::mul_(%a, %b)
    %second_output = prim::CallMethod[name="forward"](%relu, %first_output)
    return (%second_output) )",
    // 用于匹配是 ReLU 模块的断言函数
    {is_relu_module});

// 定义了一个名为 inplace_mul_f_relu 的模式信息对象，用于匹配特定的图模式
const PatternInfo inplace_mul_f_relu = PatternInfo::parse_from_str(
    R"(
graph(%self, %a, %b, %relu, %inplace):
    %first_output = aten::mul_(%a, %b)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
    // 用于匹配是 functional ReLU 的断言函数
    {is_functional_relu});

// 定义了一个名为 mul_aten_relu 的模式信息对象，用于匹配特定的图模式
const PatternInfo mul_aten_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
    %first_output = aten::mul(%a, %b)
    %second_output = aten::relu(%first_output)
    return (%second_output) )");

// 定义了一个名为 mul_aten_relu_ 的模式信息对象，用于匹配特定的图模式
const PatternInfo mul_aten_relu_ = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
    %first_output = aten::mul(%a, %b)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )");
// 定义一个名为 graph 的函数，接受三个参数 %self, %a, %b
graph(%self, %a, %b):
     // 执行 %a 和 %b 的元素级乘法，将结果存储在 %first_output 中
     %first_output = aten::mul(%a, %b)
     // 对 %first_output 执行原位操作的 ReLU 函数，将结果存储在 %second_output 中
     %second_output = aten::relu_(%first_output)
     // 返回 %second_output
     return (%second_output) )");

// 解析一个名为 inplace_mul_aten_relu 的模式信息
const PatternInfo inplace_mul_aten_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     // 执行 %a 和 %b 的原位乘法，将结果存储在 %first_output 中
     %first_output = aten::mul_(%a, %b)
     // 对 %first_output 执行 ReLU 函数，将结果存储在 %second_output 中
     %second_output = aten::relu(%first_output)
     // 返回 %second_output
     return (%second_output) )");

// 解析一个名为 inplace_mul_aten_relu_ 的模式信息
const PatternInfo inplace_mul_aten_relu_ = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     // 执行 %a 和 %b 的原位乘法，将结果存储在 %first_output 中
     %first_output = aten::mul_(%a, %b)
     // 对 %first_output 执行原位操作的 ReLU 函数，将结果存储在 %second_output 中
     %second_output = aten::relu_(%first_output)
     // 返回 %second_output
     return (%second_output) )");

// 延迟处理模式的向量，包含各种线性操作和激活函数的组合
const std::vector<std::reference_wrapper<const PatternInfo>> delay_patterns =
    {
        nn_linear_f_relu,      nn_linear_nn_relu,
        nn_linear_aten_relu,   nn_linear_aten_relu_,
        aten_linear_f_relu,    aten_linear_nn_relu,
        aten_linear_aten_relu, aten_linear_aten_relu_,

        nn_conv1d_f_relu,      nn_conv1d_nn_relu,
        nn_conv1d_aten_relu,   nn_conv1d_aten_relu_,
        nn_conv2d_f_relu,      nn_conv2d_nn_relu,
        nn_conv2d_aten_relu,   nn_conv2d_aten_relu_,
        nn_conv3d_f_relu,      nn_conv3d_nn_relu,
        nn_conv3d_aten_relu,   nn_conv3d_aten_relu_,

        add_nn_relu,           add_f_relu,
        inplace_add_nn_relu,   inplace_add_f_relu,
        add_aten_relu,         add_aten_relu_,
        inplace_add_aten_relu, inplace_add_aten_relu_,

        nn_bn2d_nn_relu,       nn_bn2d_f_relu,
        nn_bn2d_aten_relu,     nn_bn2d_aten_relu_,
        nn_bn3d_nn_relu,       nn_bn3d_f_relu,
        nn_bn3d_aten_relu,     nn_bn3d_aten_relu_,

        mul_nn_relu,           mul_f_relu,
        inplace_mul_nn_relu,   inplace_mul_f_relu,
        mul_aten_relu,         mul_aten_relu_,
        inplace_mul_aten_relu, inplace_mul_aten_relu_,
    };

// 是否插入重置观察器方法的标志，默认为 false
bool insert_reset_observer_method_{false};
// 重置观察器方法的名称
std::string reset_observer_method_name_;
};

// 获取调用的方法列表的帮助函数
ModuleMethodVector InsertObserversHelper::getInvokedMethods(
    Module& module,
    const std::string& method_name) {
  ModuleMethodVector invoked_methods;
  // 获取指定名称的方法对象
  Method method = module.get_method(method_name);
  // 获取方法的计算图
  auto graph = method.graph();

  // 使用栈来遍历图中的每一个块
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    // 遍历块中的每一个节点
    for (Node* n : b->nodes()) {
      // 跳过观察器节点
      if (observer_nodes_.count(n)) {
        continue;
      }
      // 如果节点是调用方法的节点
      if (n->kind() == prim::CallMethod) {
        // 获取被调用模块的可选项
        auto m_opt = getInvokedModuleOpt(module, n, graph->inputs()[0]);
        // 如果模块可选项有值，则将调用的模块信息添加到列表中
        if (m_opt.has_value()) {
          invoked_methods.emplace_back(*m_opt, n->s(attr::name));
        }
      }

      // 将节点的子块加入到访问列表中
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  // 返回调用的方法列表
  return invoked_methods;
}

// 为给定值插入观察器的帮助函数
void InsertObserversHelper::insertObserverFor(
    Value* v,
    Module& module,
    const Module& observer_module,
  // 检查是否已经存在与观察值对应的观察者，如果存在则返回
  if (observed_values_.count(v)) {
    return;
  }
  // 打印调试信息，显示正在为特定节点插入观察者
  GRAPH_DEBUG("Inserting observer for:", v->debugName());
  // 深拷贝观察者模块以便后续使用
  Module observer = observer_module.deepcopy();
  // 生成观察者模块的唯一名称
  std::string observer_name = "_observer_" + std::to_string(uid_++);
  // 确保生成的观察者名称在当前模块中是唯一的
  while (module.hasattr(observer_name)) {
    observer_name = "_observer_" + std::to_string(uid_++);
  }
  // 在当前模块中注册观察者模块
  module.register_module(observer_name, observer);
  // 将观察者名称和观察者模块添加到观察者名称与模块的向量中
  observer_name_and_modules.emplace_back(observer_name, observer);

  // 获取节点所属的计算图
  auto* g = v->owningGraph();
  // 在计算图中获取观察者模块的实例
  Node* observer_instance =
      g->createGetAttr(g->inputs()[0], observer_name)->insertAfter(v->node());
  // 设置观察者实例节点的调试名称
  observer_instance->output()->setDebugName(observer_name);

  {
    // 在观察者实例节点的下一个插入点范围内执行以下操作
    WithInsertPoint guard(observer_instance->next());
    // 匹配观察者的forward方法的参数类型和数量
    MatchedSchema forward_matched_schema = matchSchema(
        observer.get_method("forward").function().getSchema(),
        v->node()->sourceRange(),
        *g,
        {observer_instance->output(), v},
        {});
    // 在计算图中插入调用观察者forward方法的节点
    Node* call = g->insertMethodCall("forward", forward_matched_schema)->node();
    // 复制调用节点的元数据信息
    call->output()->copyMetadata(v);

    // 用观察者forward方法的输出替换原始节点v的所有用途
    v->replaceAllUsesWith(call->output());
    // 由于上面的替换还涉及到`call`的输入，因此将其正确切换回原始值
    call->replaceInput(1, v);
    // 将调用节点添加到观察者节点集合中
    observer_nodes_.emplace(call);
    // 将调用节点的输出值添加到观察值集合中
    observed_values_.insert(call->output());
  }
void InsertObserversHelper::insertObserverResetMinMax(
    Module& module,
    const NameModuleVector& observer_name_and_modules) {
  // 如果观察器名称和模块的向量为空，则直接返回
  if (observer_name_and_modules.empty()) {
    return;
  }
  // 查找重置观察器方法是否存在于模块中
  auto reset_min_max_opt = module.find_method(reset_observer_method_name_);
  // 如果找不到重置观察器方法，则创建一个新的图形对象
  if (!reset_min_max_opt.has_value()) {
    std::shared_ptr<Graph> reset_observer_graph = std::make_shared<Graph>();
    // 为图形对象添加一个输入节点，表示模块自身
    Value* module_value = reset_observer_graph->addInput("self");
    // 创建一个空节点作为输出节点
    Node* output_node = reset_observer_graph->createNone();
    // 向图形对象插入输出节点
    reset_observer_graph->insertNode(output_node);
    // 将输出节点注册为图形对象的输出
    reset_observer_graph->registerOutput(output_node->output());
    // 设置输入模块的类型信息
    module_value->setType(module._ivalue()->type());
    // 构建重置观察器方法的限定名
    const auto method_name = c10::QualifiedName(
        *(module.type()->name()), reset_observer_method_name_);
    // 在编译单元中创建新的函数对象，表示重置观察器方法
    auto reset_observer_fn =
        module._ivalue()->compilation_unit()->create_function(
            method_name, std::move(reset_observer_graph));
    // 定义 self 和 none 参数
    auto self_arg = c10::Argument("self", module.type());
    auto output_arg = c10::Argument("none", output_node->output()->type());
    // 创建函数模式，并设置其模式
    auto schema = c10::FunctionSchema(
        reset_observer_method_name_,
        "",
        {std::move(self_arg)},
        {std::move(output_arg)});
    reset_observer_fn->setSchema(std::move(schema));
    // 向模块类型中添加该函数
    module.type()->addMethod(reset_observer_fn);
  }
  // 获取重置观察器方法的图形对象
  auto reset_min_max_graph =
      module.get_method(reset_observer_method_name_).graph();
  // 获取重置观察器方法的输入 self
  Value* self = reset_min_max_graph->inputs()[0];

  // 遍历观察器名称和模块的向量
  for (const auto& pair : observer_name_and_modules) {
    const auto& observer_name = pair.first;
    const auto& observer = pair.second;
    // 获取观察器的值
    Value* observer_value =
        reset_min_max_graph->insertGetAttr(self, observer_name);
    // 匹配观察器的重置最小值和最大值方法的模式
    MatchedSchema reset_minmax_schema = matchSchema(
        observer.get_method("reset_min_max_vals").function().getSchema(),
        observer_value->node()->sourceRange(),
        *reset_min_max_graph,
        {observer_value},
        {});
    // 插入方法调用以重置最小值和最大值
    reset_min_max_graph->insertMethodCall(
        "reset_min_max_vals", reset_minmax_schema);
  }
}

void InsertObserversHelper::delayObservingValuesInPattern(
    Graph& graph,
    const PatternInfo& pattern) {
  // 获取模式图形对象和值映射
  const Graph& pattern_graph = *pattern.pattern_graph;
  const std::unordered_map<std::string, Value*>& vmap = pattern.vmap;

  // 查找模式图形对象在主图形对象中的匹配项
  const auto& matches = findPatternMatches(pattern_graph, graph);
  // 遍历所有匹配项
  for (const auto& match : matches) {
    // 如果不是所有的模式过滤器都通过，则继续下一个匹配项
    if (!std::all_of(
            pattern.filters.begin(),
            pattern.filters.end(),
            [&](const MatchFilter& f) { return f(match, vmap); })) {
      continue;
    }
    // 获取匹配项中的第一个输出值和第二个输出值
    auto first_output = match.values_map.at(vmap.at("first_output"));
    auto second_output = match.values_map.at(vmap.at("second_output"));
    // 输出延迟观察的调试信息，指示延迟观察的值的变化
    GRAPH_DEBUG(
        "Delay observation for value in function pattern:",
        first_output->debugName(),
        " to ",
        second_output->debugName());
    // 将第一个输出值与第二个输出值映射起来，延迟观察
    delay_observation_map_[first_output] = second_output;
  }
}
// 将延迟观察模式中的值添加到延迟观察中
void InsertObserversHelper::addValuesToDelayObservation(
    const Module& module,
    const std::string& method_name) {
  // 获取指定方法名的方法对象
  Method method = module.get_method(method_name);
  // 获取方法对象的计算图
  auto graph = method.graph();

  // 遍历延迟观察模式列表
  for (const auto& pattern : delay_patterns) {
    // 对方法图中的每个延迟观察模式应用延迟观察值操作
    delayObservingValuesInPattern(*graph, pattern);
  }
}

// 填充透传值映射
void InsertObserversHelper::fillPassThroughValueMap(
    const std::shared_ptr<Graph>& graph) {
  // 初始化待访问块的栈，从计算图的主块开始
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());

  // 当待访问块栈不为空时进行循环
  while (!blocks_to_visit.empty()) {
    // 弹出栈顶块
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();

    // 遍历块中的每个节点
    for (Node* n : b->nodes()) {
      // 如果节点是用户定义的函数调用
      if (userDefinedCallFunction(n)) {
        // 获取函数调用节点的计算图并加入待访问块栈
        auto g = getCallFunctionGraph(n);
        blocks_to_visit.push(g->block());
      }

      // 遍历节点的每个输出
      for (auto* output : n->outputs()) {
        // 获取透传输入并填充透传值映射
        for (auto* input : getPassThroughInputs(output)) {
          pass_through_value_map_[output].push_back(input);
        }
      }

      // 遍历节点的子块并加入待访问块栈
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
}

// 填充边界值映射
void InsertObserversHelper::fillBoundaryValueMap(
    Module& module,
    const std::string& method_name) {
  // 获取调用方法的列表并递归填充边界值映射
  for (auto& invoked_method : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_method);
    const auto& invoked_method_name = std::get<1>(invoked_method);
    fillBoundaryValueMap(invoked_module, invoked_method_name);
  }

  // 获取指定方法名的方法对象的计算图
  auto graph = module.get_method(method_name).graph();
  // 初始化待访问块的栈，从计算图的主块开始
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  // 获取计算图的输入并指定self指针
  auto* self = graph->inputs()[0];

  // 当待访问块栈不为空时进行循环
  while (!blocks_to_visit.empty()) {
    // 弹出栈顶块
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();

    // 继续填充边界值映射的实现代码...
    // 遍历基本块中的每个节点
    for (Node* n : b->nodes()) {
      // 检查节点类型是否为 prim::CallMethod 或者用户自定义的调用函数
      if (n->kind() == prim::CallMethod || userDefinedCallFunction(n)) {
        std::shared_ptr<Graph> g;
        // 调用者节点的输入偏移量，因为 CallFunction 的第一个输入是函数节点，
        // 而 CallFunction 的图从实际输入开始
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        size_t input_offset;
        if (n->kind() == prim::CallMethod) {
          // 获取被调用模块的可选项
          auto m_opt = getInvokedModuleOpt(module, n, self);
          if (!m_opt.has_value()) {
            continue;  // 如果没有获取到模块，继续下一个节点
          }
          auto m = *m_opt;
          // 获取调用方法的图
          g = m.get_method(n->s(attr::name)).graph();
          input_offset = 0;
        } else {
          // 获取 CallFunction 的图
          g = getCallFunctionGraph(n);
          input_offset = 1;
        }
        // 将调用节点值到被调用图中值的映射添加到边界值映射中
        for (auto i = 0U; i < g->outputs().size(); ++i) {
          auto* return_val = g->outputs()[i];
          GRAPH_DEBUG(
              "Boundary Map[return]:",
              n->output(i)->debugName(),
              " -> ",
              return_val->debugName());
          boundary_value_map_[n->output(i)].insert(return_val);
        }
        // 将调用者输入到调用图输入的映射添加到边界值映射中
        for (auto i = 0U; i < g->inputs().size(); ++i) {
          auto caller_input_index = i + input_offset;
          auto* caller_input = n->input(caller_input_index);
          auto* input_val = g->inputs()[i];
          GRAPH_DEBUG(
              "Boundary Map[input]:",
              caller_input->debugName(),
              " -> ",
              input_val->debugName());
          boundary_value_map_[caller_input].insert(input_val);
        }
      } else if (n->kind() == prim::If) {
        // 如果节点类型为 prim::If
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);  // 将子块添加到待访问列表中
          for (Value* v : n->outputs()) {
            Value* subblock_output = subblock->outputs()[v->offset()];
            GRAPH_DEBUG(
                "Boundary Map[if_output]:",
                v->debugName(),
                " -> ",
                subblock_output->debugName());
            boundary_value_map_[v].insert(subblock_output);  // 将输出值映射添加到边界值映射中
          }
        }
      } else {
        // 其他情况，遍历节点的子块并添加到待访问列表中
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
        }
      }
    }
}

void InsertObserversHelper::preprocess(
    Module& module,
    const std::string& method_name) {
  // 对子模块进行预处理，因为预处理会改变图结构，可能影响像fillBoundaryValueMap这样的后续步骤
  for (auto& invoked_method : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_method);
    const auto& invoked_method_name = std::get<1>(invoked_method);
    preprocess(invoked_module, invoked_method_name);
  }

  // 获取指定方法名的方法对象
  Method method = module.get_method(method_name);
  // 获取方法的计算图
  auto graph = method.graph();
  // 内联处理fork-wait调用
  InlineForkWait(graph);
  // 将分解的线性操作融合为aten::linear操作
  FuseLinear(graph);
  // 替换卷积操作为aten::conv操作
  replaceConvolutionWithAtenConv(graph);
  // 移除图中的列表突变
  RemoveListMutation(graph);
}

void InsertObserversHelper::analyze(
    Module& module,
    const std::string& method_name) {
  // 对子模块进行分析，与预处理类似，确保内部状态被正确填充以便后续insertObservers使用
  for (auto& invoked_method : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_method);
    const auto& invoked_method_name = std::get<1>(invoked_method);
    analyze(invoked_module, invoked_method_name);
  }

  // 将值添加到延迟观察列表中，稍后在insertObservers中使用
  addValuesToDelayObservation(module, method_name);
  // 填充值观察器映射，稍后在insertObservers中使用
  fillValueObserverMap(module, method_name);
  // 获取指定方法名的方法对象
  Method method = module.get_method(method_name);
  // 获取方法的计算图
  auto graph = method.graph();
  // 填充通过值映射表，稍后在insertObservers中使用
  fillPassThroughValueMap(graph);
}

bool InsertObserversHelper::valueNeedsToBeQuantized(
    Value* v,
    const QConfig& qconfig) {
  // 判断值是否需要量化观察
  if (isBiasOfConvOrLinear(v) ||
      !(v->type()->isSubtypeOf(*TensorType::get()) ||
        v->type()->isSubtypeOf(*ListType::ofTensors())) ||
      isEmbeddingBagNonInput(v)) {
    return false;
  }
  // 对于静态量化，只在可量化函数的输入处插入观察器
  if (quant_type_ == QuantType::STATIC) {
    // 检查生产者节点是否是静态量化的权重操作
    if (!isWeightOnlyStaticQuantOp(v->node()) &&
        (nodeQuantizable(v->node()) || isPropagateQuantOp(v->node()))) {
      return true;
    }
  }
  // 对于动态量化，检查观察器模块的数据类型
  if (quant_type_ == QuantType::DYNAMIC) {
    // 获取值的观察器模块
    Module observer_module = getObserverModuleFor(v, qconfig);
    auto scalar_type = observer_module.attr("dtype");
    // 对于非权重的Fp16类型输入，不插入观察器
    if (scalar_type == at::ScalarType::Half && !isWeight(v)) {
      return false;
    }
  }
  // 检查节点输入值是否可量化
  for (const auto& use : v->uses()) {
    if (useQuantizable(use, quant_type_)) {
      return true;
    }
  }
  return false;
}

void InsertObserversHelper::removeActivationObservers() {
  // 准备存储待移除的观察器值映射
  std::vector<std::unordered_map<Value*, Module>::iterator>
      values_to_be_removed;
  // 遍历所有观察器值映射
  for (auto it = observer_for_value_.begin(); it != observer_for_value_.end();
       it++) {
    // 如果值不是权重，则将其添加到待移除列表中
    if (!isWeight(it->first)) {
      values_to_be_removed.push_back(it);
    }
  }
  // 遍历要移除的值列表，并从值观察者映射中移除这些值的观察者
  for (auto it : values_to_be_removed) {
    observer_for_value_.erase(it);
  }
}

// 填充值观察器映射表
void InsertObserversHelper::fillValueObserverMap(
    Module& module,
    const std::string& method_name) {
  // 获取方法对象
  Method method = module.get_method(method_name);
  // 获取方法的计算图
  auto graph = method.graph();

  // 如果计算图已经被访问过，则直接返回
  if (visited_graph_of_observer_map_.count(graph.get())) {
    return;
  }
  visited_graph_of_observer_map_.insert(graph.get());

  // 初始化要访问的块栈
  std::stack<Block*> blocks_to_visit;
  // 获取模块对应的配置
  auto qconfig_opt = module_qconfig_map_.at(module._ivalue());
  if (!qconfig_opt) {
    return;
  }
  auto qconfig = *qconfig_opt;
  // 遍历计算图的输入值
  for (auto* v : graph->inputs()) {
    // 如果值需要量化，则记录观察器
    if (valueNeedsToBeQuantized(v, qconfig)) {
      GRAPH_DEBUG("Recording observer for ", v->debugName());
      GRAPH_DUMP("In graph:", v->owningGraph());
      observer_for_value_[v] = getObserverModuleFor(v, qconfig);
    }
  }

  // 将计算图的根块加入栈中
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    // 遍历块中的节点
    for (Node* n : b->nodes()) {
      // 遍历节点的输出值
      for (Value* v : n->outputs()) {
        // 如果值需要量化，则记录观察器
        if (valueNeedsToBeQuantized(v, qconfig)) {
          GRAPH_DEBUG("Recording observer for ", v->debugName());
          GRAPH_DUMP("In graph:", v->owningGraph());
          observer_for_value_[v] = getObserverModuleFor(v, qconfig);
        }
      }

      // 遍历节点的子块
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
}

// 获取值对应的观察器
std::optional<Module> InsertObserversHelper::getObserverFor(Value* v) {
  if (observer_for_value_.count(v)) {
    auto observer = observer_for_value_.at(v);
    GRAPH_DEBUG("Got observer module config for:", v->debugName());
    return observer;
  }
  std::optional<Module> result;
  if (boundary_value_map_.count(v)) {
    for (Value* next : boundary_value_map_.at(v)) {
      GRAPH_DEBUG(
          "Going through boundary map:",
          v->debugName(),
          " --> ",
          next->debugName());
      GRAPH_DUMP("From graph:", v->owningGraph());
      GRAPH_DUMP("To graph:", next->owningGraph());
      auto observer_opt = getObserverFor(next);
      if (observer_opt) {
        // 需要确保所有值都配置了相同的观察器
        if (result) {
          TORCH_CHECK(
              *observer_opt == *result,
              "Expecting all values in the graph only configured with one observer");
        } else {
          result = observer_opt;
        }
      }
    }
  }
  GRAPH_DEBUG(
      "Observer module config for ", v->debugName(), ":", result.has_value());
  return result;
}

// 插入观察器
std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>>
InsertObserversHelper::insertObservers(
    Module& module,
    const std::string& method_name,
    bool is_entry_point,
    std::unordered_set<Value*> graph_observed_values) {
  auto graph = module.get_method(method_name).graph();
  return insertObserversFor(
      graph->block(), module, graph_observed_values, is_entry_point);
}

// 记录已观察的值
void InsertObserversHelper::recordObserved(
    Value* v,
    const Module& observer_module,
    // 将要观察的值初始化为参数 v
    Value* to_observe = v;
    // 如果延迟观察映射中包含 v，则更新为延迟观察映射中的值
    if (delay_observation_map_.count(v)) {
        to_observe = delay_observation_map_.at(v);
    }
    // 将待观察的值映射到观察者模块，并存入值到观察者模块的字典中
    values_to_observe[to_observe] = observer_module;
    // 将待观察的值插入到当前块已观察值的集合中
    block_observed_values.insert(to_observe);
  }



// 结束了前面的 if 语句块，这里是 InsertObserversHelper 类的成员函数的结尾
std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>>
InsertObserversHelper::insertObserversFor(
    Block* block,
    script::Module& module,
    std::unordered_set<Value*>& block_observed_values,
    bool is_entry_point,
    bool is_user_defined_function) {
  // input/output values, used to skip inserting observers
  // for input and output of the block and the owning graph,
  // we have to insert the observers at call site because
  // the graph itself can be shared
  std::unordered_set<Value*> inputs_outputs;
  // list of observer modules for input values
  std::vector<std::optional<Module>> block_input_observers;
  // list of observer modules for output values
  std::vector<std::optional<Module>> block_output_observers;

  // if the current block is the block for entry point graph(the forward graph
  // of the top level module), we can insert observers in the block directly
  if (!is_entry_point) {
    auto* graph = block->owningGraph();
    // graph inputs/outputs
    for (auto list : {graph->inputs(), graph->outputs()}) {
      for (auto* v : list) {
        inputs_outputs.insert(v);
      }
    }
    // block outputs
    for (auto* v : block->outputs()) {
      inputs_outputs.insert(v);
    }

    for (auto* v : block->inputs()) {
      block_input_observers.emplace_back(getObserverFor(v));
    }

    for (auto* v : block->outputs()) {
      // we need explicitly skip the values that are already observed
      // this might happen in subblocks for `if` since
      // these subblock has access to all values before the `if` node
      if (!isObserved(v, block_observed_values)) {
        block_output_observers.emplace_back(getObserverFor(v));
      } else {
        block_output_observers.emplace_back(c10::nullopt);
      }
    }
  }

  // This means the block is been processed before, we just
  // need to attach observer modules and construct the information
  // needed by call site here
  bool visited = block_observer_map_.count(block);
  if (visited) {
    // instance clone of observer module and setAttr
    for (const auto& observer_attrs : block_observer_map_.at(block)) {
      const auto& name = std::get<0>(observer_attrs);
      const auto& observer = std::get<1>(observer_attrs);
      module._ivalue()->setAttr(name, observer.deepcopy()._ivalue());
    }
  }
  }
  // NB: 为什么即使访问过了，我们仍然需要处理图形？
  // 原因在于 `block_observed_values` 可能会根据方法调用的位置发生变化，
  // 并且观察到的输出（返回结果的第三个项目）
  // 可能会根据这一点发生变化，因此对于每个图形，我们都需要通过整个插入观察器的过程，
  // 插入到这个块中的观察器不会改变，但我们返回给调用者的信息将根据 `block_observed_values` 的情况改变

  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(block);
  auto* self = block->owningGraph()->inputs()[0];
  // 首先构建从值到模块的映射，稍后插入观察器，
  // 这是为了避免插入的观察器与决定插入观察器的分析干扰，
  // 我们仅为不是图形的输入/输出的 "中间值" 插入观察器
  std::unordered_map<Value*, Module> values_to_observe;

  for (auto* v : block->inputs()) {
    if (!inputs_outputs.count(v) && !values_to_observe.count(v)) {
      if (auto observer_opt = getObserverFor(v)) {
        recordObserved(
            v, *observer_opt, values_to_observe, block_observed_values);
      }
    }
  }
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    }
  }
  std::vector<size_t> output_idxs;
  for (auto i = 0U; i < block->outputs().size(); ++i) {
    if (isObserved(block->outputs()[i], block_observed_values)) {
      output_idxs.push_back(i);
    }
  }
  if (!visited) {
    NameModuleVector observer_name_and_modules;
    for (const auto& item : values_to_observe) {
      auto* v = item.first;
      auto observer = item.second;
      TORCH_CHECK(
          !is_user_defined_function,
          "Inserting observers for user defined functions is not "
          "supported right now");
      insertObserverFor(v, module, observer, observer_name_and_modules);
    }
    if (insertResetObserverMethod()) {
      insertObserverResetMinMax(module, observer_name_and_modules);
    }
    block_observer_map_[block] = observer_name_and_modules;
  }
  return std::make_tuple(
      block_input_observers, block_output_observers, output_idxs);
}

// InsertObserversHelper 类的方法，用于传播观察属性
void InsertObserversHelper::propagateObservedProperty(
    Value* output,
    std::unordered_set<Value*>& block_observed_values) {
  // 检查 pass_through_value_map_ 是否包含 output
  if (pass_through_value_map_.count(output)) {
    // 由于向量始终非空，因此不会返回初始值
    bool all_observed = true;
    // 遍历 pass_through_value_map_ 中 output 对应的值列表
    for (Value* v : pass_through_value_map_.at(output)) {
      // 检查 observed_values_ 或 block_observed_values 中是否包含 v
      all_observed &=
          observed_values_.count(v) || block_observed_values.count(v);
    }
    // 如果所有值都被观察到
    if (all_observed) {
      GRAPH_DEBUG("Pass through observed property in node:", *output->node());
      // 传播观察属性至所有不需要观察的操作
      block_observed_values.insert(output);
    }
  }
}

// 匿名命名空间结束

// 将观察者插入到模块中的函数
Module InsertObservers(
    Module& input_module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace,
    QuantType quant_type) {
  // 在克隆前填充 QConfig 映射
  ModuleQConfigMap map_before_clone;
  fillQConfigMap(input_module, qconfig_dict, map_before_clone);
  // 创建模块克隆助手对象
  ModuleCloneHelper mh;
  // 克隆输入模块
  Module module = mh.clone(input_module, map_before_clone, inplace);
  // 交换功能线性模块
  SwapFunctionalLinear(module);
  // 创建模块 QConfig 映射
  ModuleQConfigMap module_qconfig_map;
  // 由于克隆后类型已更改，需重新填充 qconfig 映射
  fillQConfigMap(module, qconfig_dict, module_qconfig_map);
  // 调试信息，输出量化类型
  GRAPH_DEBUG("Quant type:", quant_type);
  // 创建 InsertObserversHelper 对象
  InsertObserversHelper helper(module_qconfig_map, quant_type);
  // 预处理模块
  helper.preprocess(module, method_name);
  // 填充边界值映射
  helper.fillBoundaryValueMap(module, method_name);
  // 分析函数需在填充边界值映射后运行，以便跟踪调用链
  helper.analyze(module, method_name);
  // 插入观察者到模块中
  helper.insertObservers(module, method_name, /* is_entry_point */ true);
  // 返回处理后的模块
  return module;
}

// 对 OnDevicePTQ 进行模块插入观察者的函数
Module InsertObserversForOnDevicePTQ(
    Module& input_module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace,
    // 创建一个空的 ModuleQConfigMap 对象，用于存储克隆前的量化配置信息
    ModuleQConfigMap map_before_clone;
    // 调用 fillQConfigMap 函数，将输入模块 input_module 的量化配置填充到 map_before_clone 中
    fillQConfigMap(input_module, qconfig_dict, map_before_clone);
    // 创建 ModuleCloneHelper 对象
    ModuleCloneHelper mh;
    // 使用 ModuleCloneHelper 对象克隆输入模块 input_module，得到克隆后的模块 cloned_module
    Module cloned_module = mh.clone(input_module, map_before_clone, inplace);
    // 获取克隆后模块中指定方法的图形对象
    std::shared_ptr<Graph> g = cloned_module.get_method(method_name).graph();
    // 在图形对象 g 上执行 SwapFunctionalLinear 操作
    SwapFunctionalLinear(g);
    // 构建观察方法名，命名规则为 "observe_" 加上 method_name
    std::string observer_method_name = "observe_" + method_name;
    // 克隆模块的指定方法，生成一个观察方法 observer_method_name
    cloneMethod(cloned_module, method_name, observer_method_name);
    // 创建 ModuleQConfigMap 对象，用于存储模块的量化配置信息
    ModuleQConfigMap module_qconfig_map;
    // 由于克隆后类型已更改，需要重新填充量化配置映射表 module_qconfig_map
    fillQConfigMap(cloned_module, qconfig_dict, module_qconfig_map);
    // 输出调试信息，显示量化类型 quant_type 的值
    GRAPH_DEBUG("Quant type:", quant_type);
    // 创建 InsertObserversHelper 对象，用于插入观察器
    InsertObserversHelper helper(module_qconfig_map, quant_type);
    // 在量化前处理模块，包括插入观察器和处理边界值映射
    helper.preprocess(cloned_module, observer_method_name);
    // 填充边界值映射表，用于分析需要运行的 analyze 函数
    helper.fillBoundaryValueMap(cloned_module, observer_method_name);
    // 分析模块，以准备进行量化操作
    helper.analyze(cloned_module, observer_method_name);
    // 如果量化类型是动态的，则移除激活观察器
    if (quant_type == QuantType::DYNAMIC) {
      helper.removeActivationObservers();
    }
    // 设置插入重置观察器方法为真，并插入观察器到克隆模块中的指定方法
    helper.setInsertResetObserverMethod(true, method_name);
    // 在克隆模块中插入观察器，标记为入口点
    helper.insertObservers(
        cloned_module, observer_method_name, /* is_entry_point */ true);
    // 返回经过量化处理后的克隆模块
    return cloned_module;
} // 关闭 torch 命名空间
} // 关闭 jit 命名空间
} // 关闭命名空间
```