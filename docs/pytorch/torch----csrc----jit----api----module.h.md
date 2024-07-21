# `.\pytorch\torch\csrc\jit\api\module.h`

```
#pragma once
// 包含异常处理工具类头文件
#include <c10/util/Exception.h>
// 包含 PyTorch 变量相关头文件
#include <torch/csrc/autograd/variable.h>
// 包含 PyTorch 对象相关头文件
#include <torch/csrc/jit/api/object.h>
// 包含 PyTorch 源码范围相关头文件
#include <torch/csrc/jit/frontend/source_range.h>
// 包含 PyTorch IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 PyTorch 命名值相关头文件
#include <torch/csrc/jit/ir/named_value.h>
// 包含 PyTorch 运行时参数规范头文件
#include <torch/csrc/jit/runtime/argument_spec.h>
// 包含 PyTorch 图执行器相关头文件
#include <torch/csrc/jit/runtime/graph_executor.h>

// 包含 PyTorch 导出相关头文件
#include <torch/csrc/Export.h>
// 包含 PyTorch 有序字典头文件
#include <torch/csrc/api/include/torch/ordered_dict.h>
// 包含 PyTorch 编译单元头文件
#include <torch/csrc/jit/api/compilation_unit.h>

// 包含 ATen 函数模式头文件
#include <ATen/core/function_schema.h>
// 包含 ATen 资格名称头文件
#include <ATen/core/qualified_name.h>
// 包含 C10 数组引用头文件
#include <c10/util/ArrayRef.h>
// 包含 C10 可选类型头文件
#include <c10/util/Optional.h>
// 包含 C10 范围迭代头文件
#include <c10/util/irange.h>

// 包含标准库函数头文件
#include <functional>
// 包含内存管理头文件
#include <memory>
// 包含互斥锁头文件
#include <mutex>
// 包含输出流头文件
#include <ostream>
// 包含字符串处理头文件
#include <string>
// 包含无序映射头文件
#include <unordered_map>
// 包含无序集合头文件
#include <unordered_set>
// 包含实用工具头文件
#include <utility>
// 包含向量容器头文件
#include <vector>

// 本文件定义了帮助将 Python 风格模块及其方法展开为不包含任何函数调用的平面图的类。

namespace torch::jit {

// 使用 c10 命名空间中的 Argument 类型
using ::c10::Argument;
// 使用 c10 命名空间中的 FunctionSchema 类型
using ::c10::FunctionSchema;
// 使用 c10 命名空间中的 QualifiedName 类型
using ::c10::QualifiedName;

// Map 类型，用于存储文件名到内容的映射关系
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

// ModulePtr 类型，指向 c10::ivalue::Object 对象的智能指针
using ModulePtr = c10::intrusive_ptr<c10::ivalue::Object>;

// Module 结构体的前向声明
struct Module;

// 模板类 slot_list_impl 的特化，使用 ModulePolicy 策略
template <typename T>
struct slot_list_impl;

// 命名结构体，包含名称和模块的映射关系
template <typename T>
struct Named {
  std::string name; // 名称字符串
  T value;          // 泛型值
};

// 使用 Named 结构体的模板特化，存储模块对象及其名称的映射关系
using NameModule = Named<Module>;
// 使用 Named 结构体的模板特化，存储 IValue 值及其名称的映射关系
using NameValue = Named<IValue>;
// 使用 Named 结构体的模板特化，存储 Tensor 及其名称的映射关系
using NameTensor = Named<at::Tensor>;

// detail 命名空间中的结构体和策略类的前向声明
namespace detail {
struct TORCH_API ModulePolicy;
struct TORCH_API ParameterPolicy;
struct TORCH_API AttributePolicy;
struct TORCH_API BufferPolicy;
template <typename P>
struct NamedPolicy;
} // namespace detail

// 使用 slot_list_impl 类的特化，存储 ModulePolicy 策略的模块列表
using module_list = slot_list_impl<detail::ModulePolicy>;
// 使用 slot_list_impl 类的特化，存储具名的 ModulePolicy 策略的模块列表
using named_module_list =
    slot_list_impl<detail::NamedPolicy<detail::ModulePolicy>>;

// 使用 slot_list_impl 类的特化，存储 ParameterPolicy 策略的参数列表
using parameter_list = slot_list_impl<detail::ParameterPolicy>;
// 使用 slot_list_impl 类的特化，存储具名的 ParameterPolicy 策略的参数列表
using named_parameter_list =
    slot_list_impl<detail::NamedPolicy<detail::ParameterPolicy>>;

// 使用 slot_list_impl 类的特化，存储 AttributePolicy 策略的属性列表
using attribute_list = slot_list_impl<detail::AttributePolicy>;
// 使用 slot_list_impl 类的特化，存储具名的 AttributePolicy 策略的属性列表
using named_attribute_list =
    slot_list_impl<detail::NamedPolicy<detail::AttributePolicy>>;

// 使用 slot_list_impl 类的特化，存储 BufferPolicy 策略的缓冲区列表
using buffer_list = slot_list_impl<detail::BufferPolicy>;
// 使用 slot_list_impl 类的特化，存储具名的 BufferPolicy 策略的缓冲区列表
using named_buffer_list =
    slot_list_impl<detail::NamedPolicy<detail::BufferPolicy>>;

// ModuleLookup 类型，函数签名为接收字符串向量并返回 Module 对象的函数指针
using ModuleLookup = std::function<Module(const std::vector<std::string>&)>;
// 定义一个 Module 结构体，继承自 Object 类
struct TORCH_API Module : public Object {
  // 显式构造函数，根据类名创建 Module 对象
  explicit Module(c10::QualifiedName class_name);
  // 构造函数，使用 CompilationUnit 和 ClassTypePtr 创建 Module 对象
  Module(std::shared_ptr<CompilationUnit> cu, const c10::ClassTypePtr& type);
  // 默认构造函数
  Module() = default;
  // 拷贝构造函数
  Module(const Module&) = default;
  // 拷贝赋值运算符重载
  Module& operator=(const Module&) = default;
  // 移动构造函数
  Module(Module&&) noexcept = default;
  // 移动赋值运算符重载
  Module& operator=(Module&&) noexcept = default;
  // 构造函数，根据类名、CompilationUnit 和 shouldMangle 创建 Module 对象
  Module(
      c10::QualifiedName,
      std::shared_ptr<CompilationUnit> cu,
      bool shouldMangle = false);
  // 构造函数，根据 ModulePtr 创建 Module 对象
  Module(ModulePtr module_value) : Object(std::move(module_value)) {}
  // 默认析构函数
  ~Module() = default;

  // 设置优化状态（已弃用）
  void set_optimized(bool o) {
    TORCH_WARN(
        "Module::set_optimized() is deprecated and has no effect. "
        "Please use setGraphExecutorOptimize()");
  }

  // 查询优化状态（已弃用）
  bool is_optimized() const {
    TORCH_WARN(
        "Module::is_optimized() is deprecated and always returns true. "
        "Please use getGraphExecutorOptimize()");
    return true;
  }

  // 执行模型前向传播
  IValue forward(std::vector<IValue> inputs, const Kwargs& kwargs = Kwargs()) {
    return get_method("forward")(std::move(inputs), kwargs);
  }

  // 注册 buffer，用于脚本模块中的 Tensor 属性（未注册为参数）
  void register_buffer(const std::string& name, at::Tensor v) {
    bool is_param = false;
    bool is_buffer = true;
    std::lock_guard<std::mutex> lock(*register_mutex_);
    type()->addOrCheckAttribute(name, TensorType::get(), is_param, is_buffer);
    _ivalue()->setAttr(name, std::move(v));
  }

  // 注册 parameter，可以选择是否作为 buffer
  void register_parameter(
      const std::string& name,
      at::Tensor v,
      bool is_buffer) {
    std::lock_guard<std::mutex> lock(*register_mutex_);
    type()->addOrCheckAttribute(name, TensorType::get(), !is_buffer, is_buffer);
    _ivalue()->setAttr(name, std::move(v));
  }

  // 注册属性，可以指定是否为参数或缓冲区
  void register_attribute(
      const std::string& name,
      const TypePtr& t,
      IValue v,
      bool is_param = false,
      bool is_buffer = false) {
    type()->addOrCheckAttribute(name, t, is_param, is_buffer);
    _ivalue()->setAttr(name, std::move(v));
  }

  // 注册子模块
  void register_module(const std::string& name, const Module& module) {
    type()->addOrCheckAttribute(name, module.type());
  // 设置对象的属性，属性名由参数 name 指定，属性值为 module._ivalue()
  _ivalue()->setAttr(name, module._ivalue());
}

// 应用函数 fn 到当前模块及其子模块
void apply(const std::function<void(Module&)>& fn);

// 返回当前模块的缓冲区列表，可选择递归包含子模块的缓冲区
buffer_list buffers(bool recurse = true) const;
// 返回当前模块的命名缓冲区列表，可选择递归包含子模块的缓冲区
named_buffer_list named_buffers(bool recurse = true) const;

// 返回当前模块的直接子模块列表
module_list children() const; // direct modules
// 返回当前模块的命名直接子模块列表
named_module_list named_children() const;
// 返回当前模块及其所有子模块的列表，包括当前模块本身
module_list modules() const; // all modules, including this one, recursively
// 返回当前模块及其所有子模块的命名列表，包括当前模块本身
named_module_list named_modules() const;

// 返回当前模块及其子模块中所有参与梯度优化的参数列表
parameter_list parameters(bool recurse = true) const;
// 返回当前模块及其子模块中所有命名的参与梯度优化的参数列表
named_parameter_list named_parameters(bool recurse = true) const;

// 返回当前模块及其子模块的所有成员列表，类似于 Python 中的 dir(obj)
attribute_list attributes(bool recurse = true) const;
// 返回当前模块及其子模块的所有命名成员列表，类似于 Python 中的 dir(obj)
named_attribute_list named_attributes(bool recurse = true) const;

// 打印当前模块的信息，包括方法体、属性值和参数值
void dump(
    bool print_method_bodies,
    bool print_attr_values,
    bool print_param_values) const;

// 返回当前模块的信息字符串，包括方法体、属性值和参数值
std::string dump_to_str(
    bool print_method_bodies,
    bool print_attr_values,
    bool print_param_values) const;

/// 启用“训练”模式。
void train(bool on = true);
/// 调用 train(false) 来启用“评估”模式。
/// 不要重写此方法，请重写 train() 方法。
void eval() {
  train(/*on=*/false);
}
/// 如果模块处于训练模式，则返回 true。
bool is_training() const {
    // 返回 `training` 属性的布尔值。假设 `attr` 是一个函数或方法，接受一个字符串和布尔值作为参数。
    return attr("training", true).toBool();
    
    
    
    /// 递归地将所有参数转换为指定的 `dtype` 和 `device`。
    ///
    /// 如果 `non_blocking` 为 true，并且源位于固定内存且目标在 GPU 上或反之，则异步执行复制操作。
    /// 否则，此参数不起作用。
    void to(at::Device device, at::ScalarType dtype, bool non_blocking = false);
    
    
    
    /// 递归地将所有参数转换为指定的 `dtype`。
    ///
    /// 如果 `non_blocking` 为 true，并且源位于固定内存且目标在 GPU 上或反之，则异步执行复制操作。
    /// 否则，此参数不起作用。
    void to(at::ScalarType dtype, bool non_blocking = false);
    
    
    
    /// 递归地将所有参数移动到指定的设备。
    ///
    /// 如果 `non_blocking` 为 true，并且源位于固定内存且目标在 GPU 上或反之，则异步执行复制操作。
    /// 否则，此参数不起作用。
    void to(at::Device device, bool non_blocking = false);
    
    
    
    // 将模型保存到输出流 `out` 中，并可选地保存附加文件。
    void save(
        std::ostream& out,
        const ExtraFilesMap& extra_files = ExtraFilesMap()) const;
    
    
    
    // 将模型保存到文件 `filename` 中，并可选地保存附加文件。
    void save(
        const std::string& filename,
        const ExtraFilesMap& extra_files = ExtraFilesMap()) const;
    
    
    
    // 为移动设备保存模型到输出流 `out` 中，可选地保存附加文件和调试信息，以及指定是否使用 flatbuffer。
    void _save_for_mobile(
        std::ostream& out,
        const ExtraFilesMap& extra_files = ExtraFilesMap(),
        bool save_mobile_debug_info = false,
        bool use_flatbuffer = false) const;
    
    
    
    // 为移动设备保存模型到文件 `filename` 中，可选地保存附加文件和调试信息，以及指定是否使用 flatbuffer。
    void _save_for_mobile(
        const std::string& filename,
        const ExtraFilesMap& extra_files = ExtraFilesMap(),
        bool save_mobile_debug_info = false,
        bool use_flatbuffer = false) const;
    
    
    
    // 复制当前模型并返回新的模型实例。
    Module copy() const;
    
    
    
    // 深度复制当前模型并返回新的模型实例，可以选择指定目标设备。
    Module deepcopy(std::optional<at::Device> device = c10::nullopt) const;
    
    
    
    // 克隆当前模型和其底层 `ClassType`，返回一个新的实例，类型相同但数据相同，如果 `inplace` 为 true，则原地克隆。
    Module clone(bool inplace = false) const;
    
    
    
    // 克隆当前模型和其底层 `ClassType`，返回一个新的实例，类型相同但数据相同，如果 `inplace` 为 true，则原地克隆。
    // 还允许调用者指定一组方法和属性名称，以避免克隆。
    Module clone(
        bool inplace,
        const std::unordered_set<std::string>& ignored_method,
        const std::unordered_set<std::string>& ignored_attributes) const;
    
    
    
    // 克隆给定模型 `orig` 中的方法 `name`。
    void clone_method(const Module& orig, const std::string& name);
    
    
    
    // 执行模型，并返回对应的 `IValue` 结果。
    IValue operator()(std::vector<IValue> inputs);
    
    
    
    // 创建一个指定名称的类实例，并传递参数 `args`，返回对应的 `IValue`。
    template <typename... Types>
    IValue create_class(const c10::QualifiedName& name, Types&&... args) const {
  // 调用 create_class 函数，创建一个类对象并返回
  return create_class(name, {IValue(std::forward<Types>(args))...});
}

// 根据类名创建一个类对象，参数为一个堆栈
IValue create_class(const c10::QualifiedName& name, Stack stack) const;

// 比较操作符重载，用于判断两个 Module 对象是否相等
inline bool operator==(const Module& y) const noexcept {
  // 判断两个 Module 对象的内部状态是否相等
  return _ivalue() == y._ivalue();
}

// 设置需要在对象销毁时一同删除的内存
void set_delete_memory(std::shared_ptr<char> delete_mem) {
  mem_to_delete_ = std::move(delete_mem);
}

// 一组函数，用于在 torch.jit.save 和 torch.jit.load 过程中维护输入形状。
// 仅支持张量及其列表/字典，因为追踪仅支持这些类型。
void store_traced_inputs(std::string func_name, std::vector<IValue> inputs) {
  if (inputs.size() == 0) {
    return;
  }
  auto c10_inputs = c10::impl::GenericList(AnyType::get());
  for (IValue& value : inputs) {
    // 不检查此值是否可追踪类型，因为此前已在堆栈的较高位置检查过，
    // 修改此行为需要更大的重构。
    c10_inputs.emplace_back(std::move(value));
  }
  // 插入或更新函数名及其对应的追踪输入列表
  traced_inputs_.insert_or_assign(func_name, c10_inputs);
}

// 返回模块已追踪的所有输入，以字典形式返回
c10::Dict<std::string, c10::impl::GenericList> retrieve_traced_inputs() const {
  return traced_inputs_;
}

private:
// 克隆模块的私有实现，用于在克隆过程中处理类型重映射、忽略方法及属性等
Module clone_impl(
    std::unordered_map<TypePtr, TypePtr>& type_remap,
    bool inplace,
    IValue::HashIdentityIValueMap memo,
    const std::unordered_set<std::string>& ignored_methods,
    const std::unordered_set<std::string>& ignored_attributes) const;

// 克隆方法的私有实现，用于复制模块的特定方法
void clone_method(
    const Module& orig,
    const Function& method,
    const std::unordered_map<TypePtr, TypePtr>& type_remap);

// 获取给定基础名称的方法的限定名称
c10::QualifiedName getNameForMethod(std::string basename) const {
  return QualifiedName(*type()->name(), std::move(basename));
}

// 将对象转换为特定的实现，可以指定设备、数据类型及非阻塞模式
void to_impl(
    const std::optional<at::Device>& device,
    const std::optional<at::ScalarType>& dtype,
    bool non_blocking);

// 模块销毁时一同删除的额外内存句柄
std::shared_ptr<char> mem_to_delete_;

// 函数名到其追踪输入列表的映射
c10::Dict<std::string, c10::impl::GenericList> traced_inputs_;

// 保证注册缓冲区或参数时线程安全的互斥锁
std::shared_ptr<std::mutex> register_mutex_ = std::make_shared<std::mutex>();
// 结束匿名命名空间的声明
};

// torch.jit.freeze 的 C++ 等效 API。具体细节请参阅相应的文档。
TORCH_API Module freeze(
    const Module& module,
    const std::optional<std::vector<std::string>>& preserved_attrs =
        c10::nullopt,
    bool optimize_numerics = true);

// torch.jit.optimize_for_inference 的 C++ 等效 API。具体细节请参阅相应的文档。
TORCH_API Module optimize_for_inference(
    Module& module,
    const std::vector<std::string>& other_methods = {});

// 枚举类型 FusionBehavior，定义了融合行为的类型：STATIC（静态）和 DYNAMIC（动态）
enum class FusionBehavior { STATIC, DYNAMIC };

// FusionStrategy 类型的别名，用于定义融合操作的特化类型和数量
using FusionStrategy = std::vector<std::pair<FusionBehavior, size_t>>;
// clang-format off
/*
设置融合过程中可能发生的特化类型和数量。

用法：提供一系列类型为 (type, depth) 的对，其中 type 是 STATIC 或 DYNAMIC，
depth 是一个整数。

行为 - 静态 vs 动态：
    在 STATIC 融合中，融合操作被编译为具有固定的输入形状。形状根据一些初始的性能分析运行来确定。
    在 DYNAMIC 融合中，融合操作被编译为具有可变的输入形状，因此可能存在多个形状。

在两种情况下，还会基于新的步进行为、设备或数据类型进行重新编译。

行为 - 回退函数 & 深度：
    当输入不符合专门编译的操作所需的格式时，将运行回退函数。回退函数根据观察到的张量形状递归地编译和特化。
    由于编译可能很慢，提供了 "depth" 参数来限制可以编译的特化数量，在放弃重新编译并回退到完全未融合、未特化的实现之前。

(type, depth) 对列表控制特化的类型和数量。例如：[(STATIC, 2), (DYNAMIC, 2)] 表示前两个特化将使用静态融合，接下来的两个特化将使用动态融合，
而任何不满足这四个选项的输入将运行未融合的实现。

注：随着更多的融合后端的添加，可能会有更多细粒度的特定融合器的 API。
*/
// clang-format on
TORCH_API FusionStrategy getFusionStrategy();
// 返回之前的策略
TORCH_API FusionStrategy setFusionStrategy(FusionStrategy& fusion_strategy);

namespace detail {

// SlotCursor 结构体，用于追踪 Module 的槽位游标
struct TORCH_API SlotCursor {
  Module module_;
  int64_t i_; // 槽位偏移，-1 表示模块本身
};

} // namespace detail

// 此迭代器允许对 Module 的成员（可选地递归）进行枚举。它执行模块的深度优先前序遍历。
// Policy 模板参数确定应包括对象的哪些槽位。例如，在迭代参数时，返回参数张量，但跳过模块、缓冲区和其他属性。
// 有关 Policy 对象的 API，请参阅 ModulePolicy 的注释。
template <typename Policy>
// 定义结构体 slot_iterator_impl，实现迭代器接口用于遍历模块的属性槽
struct slot_iterator_impl {
  using SlotCursor = detail::SlotCursor; // 使用 SlotCursor 类型别名
  using value_type = typename Policy::value_type; // 定义 value_type 为 Policy 的值类型

  // 构造函数，初始化迭代器
  slot_iterator_impl(
      Module root, // 根模块
      bool recurse, // 是否进行深度优先搜索
      bool return_module) // 是否包含根模块自身作为首个访问对象（用于 modules() 函数）
      : cursors_({SlotCursor{std::move(root), return_module ? -1 : 0}}), // 初始化 cursors_，传入根模块和起始索引
        recurse_(recurse) { // 初始化 recurse_

    // 将迭代器推进到第一个有效元素（或者如果为空，则到末尾）
    while_not_valid_next();
  }

  // 默认构造函数，空 cursors_ 表示迭代结束
  slot_iterator_impl() : recurse_(false) {}

  // 解引用操作符，返回当前迭代元素的值
  value_type operator*() const {
    return Policy::create(cursors_, cur());
  }

  // 成员访问操作符，返回当前迭代元素的值
  value_type operator->() const {
    return **this;
  }

  // 前缀递增操作符，使迭代器指向下一个有效元素
  slot_iterator_impl& operator++() {
    next_valid();
    return *this;
  }

  // 后缀递增操作符，返回迭代之前的副本
  slot_iterator_impl operator++(int) {
    // 这非常耗费资源，是否应该删除它，以防止人们使用它而不是前缀递增操作符？
    slot_iterator_impl old = *this;
    ++(*this);
    return old;
  }

 private:
  // return_module() 是一个特殊情况，它返回根模块自身，而不是其子模块，
  // 因为我们正在迭代 modules()，这包括根模块本身。
  bool return_module() const {
    return top().i_ == -1;
  }

  // 返回当前堆栈顶部的 SlotCursor 引用
  const SlotCursor& top() const {
    return cursors_.back();
  }

  // 返回当前堆栈顶部的 SlotCursor 引用（可修改版本）
  SlotCursor& top() {
    return cursors_.back();
  }

  // 返回当前迭代器指向的值
  IValue cur() const {
    return return_module() ? top().module_._ivalue()
                           : top().module_._ivalue()->getSlot(top().i_);
  }

  // 前进到深度优先遍历模块槽的下一个槽位。
  // 此函数不保证下一个槽位是迭代的有效元素，这由 valid() 函数保证。
  // 不变量: !cursors_.empty()
  void next() {
    // 如果我们刚刚返回了模块本身，则将 i_ 增加到 0，现在我们位于模块的第一个槽位。
    if (return_module()) {
      ++top().i_;
      return;
    }
    // 上次遍历操作超出了模块槽的数量，继续在父级中进行迭代。
    if (top().i_ >= int64_t(top().module_._ivalue()->type()->numAttributes())) {
      cursors_.pop_back();
      if (!cursors_.empty()) {
        ++top().i_;
      }
      return;
    }
    // 如果当前对象是一个模块，我们必须扫描它进行递归遍历。通过添加一个新的 SlotCursor 来追踪遍历。
    if (recurse_ &&
        top().module_._ivalue()->type()->getAttribute(top().i_)->is_module()) {
      cursors_.emplace_back(SlotCursor{cur().toModule(), 0});
      return;
    }
    // 常见情况: 前进到下一个槽位。

    ++top().i_;
    return;
  }
}
    ++top().i_;

# 增加当前迭代器的索引，使其指向下一个位置

  // is the current position of the iterator a valid one?
  // otherwise, we have to continue advancing.
  bool valid() const {
    return top().i_ <
        int64_t(top().module_._ivalue()->type()->numAttributes()) &&
        Policy::valid(
               top().module_._ivalue()->type(),
               top().i_,
               top().module_._ivalue()->getSlot(top().i_));
  }

// 检查当前迭代器的位置是否有效，如果无效则需要继续前进。

  void while_not_valid_next() {
    // advance iteration until we are either at the end (cursors_.empty())
    // or in a valid state. return_module() is a special case,
    // and is always considered valid, regardless of Policy, because it is
    // it is only true when we are iterating modules.
    while (!cursors_.empty() && !return_module() && !valid()) {
      next();
    }
  }

// 不断前进迭代，直到到达末尾（cursors_.empty()）或者处于有效状态。
// return_module() 是一个特殊情况，无论 Policy 如何，它始终被认为是有效的，
// 因为它只在迭代模块时为真。

  void next_valid() {
    // avoid crashing if this is empty
    if (cursors_.empty()) {
      return;
    }
    // advance to next element, which is maybe not valid
    next();
    while_not_valid_next();
  }

// 如果为空则避免崩溃。
// 前进到下一个元素，该元素可能无效。
// 继续前进直到找到有效的下一个元素。

  std::vector<SlotCursor> cursors_;
  bool recurse_;

  friend inline bool operator!=(
      const slot_iterator_impl<Policy>& a,
      const slot_iterator_impl<Policy>& b) {
    // we are finished iteration when we have no more iteration SlotCursors.
    // end is always an empty iterator with no cursors.
    return (a.cursors_.empty() != b.cursors_.empty());
  }

// 当没有更多迭代的 SlotCursors 时，迭代结束。
// end 总是一个没有任何 cursor 的空迭代器。
};

// This type represents lists of parameters, attributes, and
// submodules contained in the module. It is abstract because
// they are not stored directly in std::vectors but inside the
// module's IValue object itself.
template <typename Policy>
struct slot_list_impl {
  using iterator = slot_iterator_impl<Policy>;
  using const_iterator = slot_iterator_impl<Policy>;
  using value_type = typename iterator::value_type;

  // Returns an iterator pointing to the beginning of the slot list
  slot_iterator_impl<Policy> begin() const {
    return slot_iterator_impl<Policy>(module_, recurse_, return_module_);
  }

  // Returns an iterator pointing to the end of the slot list
  slot_iterator_impl<Policy> end() const {
    return slot_iterator_impl<Policy>();
  }

  // Returns the size of the slot list
  size_t size() const {
    if (!size_) {
      size_ = size_t(0);
      for ([[maybe_unused]] const value_type& _ : *(this)) {
        ++*size_;
      }
    }
    return *size_;
  }

  // Constructor for slot_list_impl, initializes with a Module object and options
  slot_list_impl(Module module, bool recurse, bool return_module)
      : module_(std::move(module)),
        recurse_(recurse),
        return_module_(return_module),
        size_(c10::nullopt) {
    if (!recurse && !return_module && Policy::all_slots) {
      size_ = module_.num_slots();
    }
  }

 private:
  Module module_;               // The module associated with this slot list
  bool recurse_;                // Whether to recurse into submodules
  bool return_module_;          // Whether to return the module itself
  mutable std::optional<size_t> size_;  // Cached size of the slot list
  friend struct Module;
};

namespace detail {

// Policy for iterating over module slots (submodules)
struct TORCH_API ModulePolicy {
  using value_type = Module;    // The type of value returned by iterators

  // Creates a Module object from an IValue and slot cursors
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    return Module(std::move(v).toObject());
  }

  // Checks if the slot at index i in typ is a module
  static bool valid(const ClassTypePtr& typ, size_t i, const IValue& v) {
    return typ->getAttribute(i)->is_module();
  }

  // Determines whether to return all slots (submodules) or not
  static CONSTEXPR_EXCEPT_WIN_CUDA bool all_slots = false;
};

// Policy for iterating over parameters
struct TORCH_API ParameterPolicy {
  using value_type = at::Tensor;  // The type of value returned by iterators

  // Creates a Tensor object from an IValue and slot cursors
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    return std::move(v).toTensor();
  }

  // Checks if the slot at index i in typ is a parameter of type Tensor
  static bool valid(const ClassTypePtr& typ, size_t i, const IValue& v) {
    return typ->is_parameter(i) && v.isTensor();
  }

  // Determines whether to return all slots (parameters) or not
  static CONSTEXPR_EXCEPT_WIN_CUDA bool all_slots = false;
};

// Policy for iterating over buffers
struct TORCH_API BufferPolicy {
  using value_type = at::Tensor;  // The type of value returned by iterators

  // Creates a Tensor object from an IValue and slot cursors
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    return std::move(v).toTensor();
  }

  // Checks if the slot at index i in typ is a buffer of type Tensor
  static bool valid(const ClassTypePtr& typ, size_t i, const IValue& v) {
    return typ->is_parameter(i) && v.isTensor();
  }

  // Determines whether to return all slots (buffers) or not
  static CONSTEXPR_EXCEPT_WIN_CUDA bool all_slots = false;
};

} // namespace detail
    // 返回条件：检查 typ 对象的第 i 个属性是否是 TensorType 的子类型，并且该属性是一个缓冲区
    return typ->getAttribute(i)->isSubtypeOf(*TensorType::get()) &&
        typ->is_buffer(i);
  }
  // 定义静态常量 all_slots，并初始化为 false
  static CONSTEXPR_EXCEPT_WIN_CUDA bool all_slots = false;
};

// 结构体定义：AttributePolicy
struct TORCH_API AttributePolicy {
  // value_type 被定义为 IValue 类型
  using value_type = IValue;
  
  // 创建函数：根据指定的 cursors 和 v 创建一个 value_type 对象并返回
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    return v;
  }
  
  // 验证函数：检查给定的 typ 类型、索引 i 和值 v 是否有效，始终返回 true
  static bool valid(const ClassTypePtr& typ, size_t i, const IValue& v) {
    return true;
  }
  
  // 常量表达式：除了 Windows 和 CUDA 以外的平台都为 true
  static CONSTEXPR_EXCEPT_WIN_CUDA bool all_slots = true;
};

// 模板类定义：NamedPolicy 模板化于 Policy
template <typename Policy>
struct NamedPolicy {
  // value_type 被定义为 Named<typename Policy::value_type> 类型
  using value_type = Named<typename Policy::value_type>;
  
  // 创建函数：根据指定的 cursors 和 v 创建一个 value_type 对象并返回
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    std::string name;
    if (cursors.size() == 1) {
      // 如果 cursors 只有一个元素，根据条件设置 name
      name = (cursors.back().i_ == -1) ? "" : nameFragment(cursors.back());
    } else {
      // 如果 cursors 有多个元素，生成完整的 name
      std::ostringstream ss;
      for (const auto i : c10::irange(cursors.size())) {
        if (i > 0) {
          ss << ".";
        }
        ss << nameFragment(cursors[i]);
      }
      name = ss.str();
    }
    // 返回一个包含 name 和 Policy::create(cursors, std::move(v)) 的 Named 对象
    return value_type{std::move(name), Policy::create(cursors, std::move(v))};
  }
  
  // 验证函数：检查给定的 t 类型、索引 i 和值 v 是否有效，调用 Policy::valid 进行检查
  static bool valid(const ClassTypePtr& t, size_t i, const IValue& v) {
    return Policy::valid(t, i, v);
  }
  
  // 常量表达式：继承自 Policy::all_slots，表示所有的插槽都有效
  static constexpr bool all_slots = Policy::all_slots;

 private:
  // 静态私有函数：返回 detail::SlotCursor 对象 f 的模块类型的属性名称
  static std::string nameFragment(const detail::SlotCursor& f) {
    return f.module_.type()->getAttributeName(f.i_);
  }
};

// detail 命名空间结束
} // namespace detail

// 获取内联模式开关的全局函数声明
TORCH_API bool& getInlineEverythingMode();

// script 命名空间开始
namespace script {
// 以前我们有一个已删除的 `script::` 命名空间，这个别名是为了向后兼容公共 API
// 新的代码不应该使用这个类型别名
using Module = ::torch::jit::Module;
using ExtraFilesMap = ::torch::jit::ExtraFilesMap;
} // namespace script

// torch::jit 命名空间结束
} // namespace torch::jit
```