# `.\pytorch\torch\csrc\jit\frontend\tracer.h`

```py
#pragma once
// 防止头文件被多次包含

#include <ATen/core/Dimname.h>
// 引入 ATen 库中的 Dimname 头文件
#include <ATen/core/class_type.h>
// 引入 ATen 库中的 class_type 头文件
#include <ATen/core/jit_type.h>
// 引入 ATen 库中的 jit_type 头文件
#include <ATen/core/stack.h>
// 引入 ATen 库中的 stack 头文件
#include <ATen/core/symbol.h>
// 引入 ATen 库中的 symbol 头文件
#include <c10/util/Exception.h>
// 引入 c10 库中的 Exception 头文件
#include <torch/csrc/Export.h>
// 引入 torch 库中的 Export 头文件

#include <torch/csrc/jit/frontend/source_range.h>
// 引入 torch 库中的 source_range 头文件
#include <torch/csrc/utils/variadic.h>
// 引入 torch 库中的 variadic 头文件

#include <cstdint>
// 引入标准库的 cstdint 头文件，包含整数类型定义
#include <memory>
// 引入标准库的 memory 头文件，包含智能指针和内存管理相关功能
#include <mutex>
// 引入标准库的 mutex 头文件，包含互斥锁功能
#include <unordered_map>
// 引入标准库的 unordered_map 头文件，包含无序映射容器
#include <vector>
// 引入标准库的 vector 头文件，包含动态数组功能

namespace torch::jit {
// 声明 torch::jit 命名空间

struct Node;
// 声明 Node 结构体
struct Value;
// 声明 Value 结构体
struct Graph;
// 声明 Graph 结构体
struct Module;
// 声明 Module 结构体

namespace tracer {
// 声明 tracer 命名空间

using ::c10::ivalue::Shared;
// 使用 c10 库中 ivalue 命名空间的 Shared

using ::c10::IValue;
// 使用 c10 库中的 IValue 类
using ::c10::ivalue::Future;
// 使用 c10 库中 ivalue 命名空间的 Future

using ::c10::ArrayRef;
// 使用 c10 库中的 ArrayRef 类
using ::c10::TupleType;
// 使用 c10 库中的 TupleType 类
using ::c10::TupleTypePtr;
// 使用 c10 库中的 TupleTypePtr 类
using ::c10::ivalue::ConstantString;
// 使用 c10 库中 ivalue 命名空间的 ConstantString

using torch::autograd::Variable;
// 使用 torch 库中 autograd 命名空间的 Variable
using variable_list = std::vector<Variable>;
// 定义 variable_list 别名为 std::vector<Variable>

TORCH_API std::atomic<bool>& getTracerStateWarnMode();
// 声明 TORCH_API 的 getTracerStateWarnMode 函数，返回 std::atomic<bool>&

struct TORCH_API TracingState
    : public std::enable_shared_from_this<TracingState> {
  // 声明 TracingState 结构体，继承自 enable_shared_from_this<TracingState>

  TracingState();
  // 声明构造函数 TracingState()

  ~TracingState();
  // 声明析构函数 ~TracingState()

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Graph> graph;
  // 声明 graph 成员变量，类型为 std::shared_ptr<Graph>

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool warn = getTracerStateWarnMode();
  // 声明 warn 成员变量，类型为 bool，初始化为 getTracerStateWarnMode() 返回值

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool strict = true;
  // 声明 strict 成员变量，类型为 bool，初始化为 true

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool force_outplace = false;
  // 声明 force_outplace 成员变量，类型为 bool，初始化为 false

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::function<std::string(const Variable& var)> lookup_var_name_fn =
      [](const Variable& var) { return ""; };
  // 声明 lookup_var_name_fn 成员变量，类型为 std::function<std::string(const Variable& var)>，
  // 初始化为 lambda 函数，返回空字符串

  void enterFrame() {
    env_stack.emplace_back();
  }
  // 声明 enterFrame() 方法，向 env_stack 中添加一个新帧

  void leaveFrame() {
    env_stack.pop_back();
  }
  // 声明 leaveFrame() 方法，从 env_stack 中移除当前帧

  void setValue(const IValue& v, Value* value);
  // 声明 setValue() 方法，设置指定 IValue 对应的 Value

  void delValue(const IValue& var);
  // 声明 delValue() 方法，删除指定的 IValue

  Value* getValue(const IValue& var);
  // 声明 getValue() 方法，获取指定的 IValue 对应的 Value

  Value* getOutput(const IValue& var, size_t i);
  // 声明 getOutput() 方法，获取指定的 IValue 对应的输出 Value 的第 i 个值

  bool hasValue(const IValue& var) const;
  // 声明 hasValue() 方法，检查是否存在指定的 IValue

  Node* createNode(c10::Symbol op_name, size_t num_outputs);
  // 声明 createNode() 方法，创建一个具有指定操作名和输出数量的 Node

  void insertNode(Node* node);
  // 声明 insertNode() 方法，插入一个 Node

 private:
  using WeakIValue = at::WeakIValue;
  // 使用 at 命名空间的 WeakIValue 类

  struct WeakIValueHasher {
    size_t operator()(const WeakIValue& t) const {
      return t.hash();
    }
  };
  // 声明 WeakIValueHasher 结构体，实现 WeakIValue 的哈希函数

  struct WeakIValueEq {
    bool operator()(const WeakIValue& t1, const WeakIValue& t2) const {
      return t1.isSameIdentity(t2);
    }
  };
  // 声明 WeakIValueEq 结构体，实现 WeakIValue 的相等性比较函数

  using Frame =
      std::unordered_map<WeakIValue, Value*, WeakIValueHasher, WeakIValueEq>;
  // 使用 unordered_map 定义 Frame 类型，键为 WeakIValue，值为 Value*

  std::vector<Frame> env_stack;
  // 声明 env_stack 成员变量，类型为 vector<Frame>，用于存储环境帧
};

// This is meant to be used as a thread local place, where we can store extra
// info that gets lost when we call into ATen from Python bindings. One example
// for when this happens is when we get an IntArrayRef argument with e.g. sizes
// for view. When tracing, those might be tensors, which let us encode extra
// data dependencies, but once they get to the ATen call where we actually have
// the tracing logic, they get converted into a raw IntArrayRef, and we loose
// 注释：此注释描述了 TracingState 类的设计目的，用于存储在从 Python 绑定调用 ATen 时丢失的额外信息。
// 例如，当我们获取 IntArrayRef 参数（例如 view 的大小）时。在跟踪过程中，这些可能是张量，
// 允许我们编码额外的数据依赖关系，但一旦它们到达实际具有跟踪逻辑的 ATen 调用时，
// 它们会转换为原始的 IntArrayRef，并且我们会失去这些额外信息。
// 所有信息的暂存位置，防止泄露。NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// ArgumentStash 结构体，用于存储函数参数的信息
struct ArgumentStash {
  // IntArrayRefTrace 结构体，继承自 std::vector<Value*>，用于跟踪 IntArrayRef 类型的参数
  struct IntArrayRefTrace : std::vector<Value*> {
    IntArrayRefTrace(int size) : std::vector<Value*>(size, nullptr) {}
  };

  // 静态方法，检查 stash 中的 intlists 是否为空
  static bool empty() {
    return stash.intlists.empty();
  }

  // TORCH_API 标记的静态方法，用于暂存 IntArrayRef 类型参数的元素信息
  TORCH_API static void stashIntArrayRefElem(
      const std::string& arg_name,
      size_t size,
      size_t idx,
      const Variable& var);

  // 静态方法，检查是否存在特定名称的 IntArrayRef 类型参数
  static bool hasIntArrayRef(const std::string& arg_name) {
    return stash.intlists.count(arg_name) > 0;
  }

  // 静态方法，从 stash 中弹出特定名称的 IntArrayRefTrace 对象
  static IntArrayRefTrace popIntArrayRef(const std::string& arg_name) {
    auto info = std::move(stash.intlists.at(arg_name));
    stash.intlists.erase(arg_name);
    return info;
  }

  // Value stashing: 使用这些方法来暂存图中对应的普通 Value* 类型参数，不需要特殊处理，如 IntArrayRefs
  TORCH_API static void stashValue(
      const std::string& arg_name,
      size_t idx,
      const Variable& var,
      const c10::TypePtr& type = nullptr);

  // 静态方法，检查是否存在特定名称的 Value 类型参数
  static bool hasValue(const std::string& arg_name) {
    return stash.values.count(arg_name) > 0;
  }

  // 静态方法，从 stash 中弹出特定名称的 Value 对象
  static Value* popValue(const std::string& arg_name) {
    auto info = stash.values.at(arg_name);
    stash.values.erase(arg_name);
    return info;
  }

 private:
  // 使用线程局部存储的 stash 对象，存储 IntArrayRefTrace 和 Value 对象
  static thread_local ArgumentStash stash;
  std::unordered_map<std::string, IntArrayRefTrace> intlists;
  std::unordered_map<std::string, Value*> values;
};

// 获取当前的跟踪状态，如果跟踪被禁用则返回 nullptr
TORCH_API const std::shared_ptr<TracingState>& getTracingState();
// 设置当前的跟踪状态
TORCH_API void setTracingState(std::shared_ptr<TracingState> state);

// 判断是否处于跟踪状态
inline bool isTracing() {
  return static_cast<bool>(getTracingState());
}

// 警告函数指针类型定义
using warn_fn_type = void (*)(const std::string& msg);

// 外部声明的警告消息常量
TORCH_API extern const char* WARN_PYTHON_DATAFLOW;
TORCH_API extern const char* WARN_CONSTRUCTOR;
TORCH_API extern const char* WARN_RESIZE;
TORCH_API extern const char* STRICT_TRACER_MSG;

// 警告函数，如果跟踪状态存在并启用，则发出警告
TORCH_API void _do_warn(const char* _reason, const char* _kind);
inline void warn(const char* _reason, const char* _kind = nullptr) {
  if (const auto& state = getTracingState()) {
    if (!state->warn)
      return;
    _do_warn(_reason, _kind);
  }
}

// 设置警告处理函数
TORCH_API void setWarn(warn_fn_type fn);

// NoWarn 结构体，用于临时禁用警告
struct TORCH_API NoWarn {
  NoWarn() : state(getTracingState()) {
    if (state) {
      prev = state->warn;
      state->warn = false;
    }
  }
  ~NoWarn() {
    if (state) {
      state->warn = prev;
    }
  }
  std::shared_ptr<TracingState> state;
  bool prev{false};
};

// WithNestedTracingFrame 结构体，用于在嵌套跟踪框架内进行跟踪
struct WithNestedTracingFrame {
  WithNestedTracingFrame() {
    getTracingState()->enterFrame();
  }

  ~WithNestedTracingFrame() {
    getTracingState()->leaveFrame();
  }
};

// 记录节点的源代码位置
TORCH_API void recordSourceLocation(Node* n);
// 设置记录节点源代码位置的函数
TORCH_API void setRecordSourceLocation(void (*v)(Node*));
// 声明一个返回值为 std::vector<StackEntry> 类型的函数 pythonCallstack
TORCH_API std::vector<StackEntry> pythonCallstack();

// 声明一个不返回任何值的函数，接受一个返回 std::vector<StackEntry> 的函数指针作为参数
TORCH_API void setPythonCallstack(std::vector<StackEntry> (*v)());

// 将一个 IValue 对象与一个值节点关联起来，以便后续涉及该变量的操作知道在 IR 中引用的节点
TORCH_API void setValueTrace(const IValue& v, Value* value);

// 删除与给定 IValue 相关的值节点追踪
TORCH_API void delValueTrace(const IValue& var);

// 暂停追踪的函数，返回一个无参无返回值的函数对象
TORCH_API std::function<void()> pauseTracing();

// 获取与给定 IValue 相关联的值节点
TORCH_API Value* getValueTrace(const IValue& var);

// 在给定输入和函数后，执行跟踪操作，并返回一个包含共享指针和堆栈的 std::pair
TORCH_API std::pair<std::shared_ptr<TracingState>, Stack> trace(
    Stack inputs,
    const std::function<Stack(Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool strict = true,
    bool force_outplace = false,
    Module* self = nullptr,
    const std::vector<std::string>& argument_names = {});

// 放弃当前跟踪状态的函数
TORCH_API void abandon();

// 下面的函数用作 addInputs 函数的中间步骤和模板递归终止的重载
// 将一个 int64_t 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, int64_t value);

// 将一个 c10::SymInt 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, c10::SymInt value);

// 将一个 std::optional<int64_t> 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, std::optional<int64_t> value);

// 将一个 bool 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, bool value);

// 将一个 std::optional<bool> 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, const std::optional<bool>& value);

// 将一个 double 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, double value);

// 将一个 std::optional<double> 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, const std::optional<double>& value);

// 将一个 at::Scalar 对象作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, const at::Scalar& value);

// 将一个 std::optional<at::Scalar> 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, const std::optional<at::Scalar>& value);

// 将一个 at::Tensor 对象作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, const at::Tensor& value);

// 将一个 std::optional<at::Tensor> 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, const std::optional<at::Tensor>& value);

// 将一个 ArrayRef<int64_t> 对象作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, ArrayRef<int64_t> value);

// 将一个 c10::SymIntArrayRef 对象作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, c10::SymIntArrayRef value);

// 将一个 std::optional<c10::SymInt> 值作为输入添加到给定的节点中
TORCH_API void addInputs(Node* n, const char* name, std::optional<c10::SymInt> value);

// 将一个 std::optional<ArrayRef<int64_t>> 值作为输入添加到给定的节点中
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<ArrayRef<int64_t>>& value);

// 将一个 at::OptionalIntArrayRef 对象作为输入添加到给定的节点中
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const at::OptionalIntArrayRef& opt_value);

// 将一个 at::OptionalSymIntArrayRef 对象作为输入添加到给定的节点中
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const at::OptionalSymIntArrayRef& opt_value);

// 将一个 ArrayRef<at::Tensor> 对象作为输入添加到给定的节点中
TORCH_API void addInputs(
    Node* n,
    const char* name,
    ArrayRef<at::Tensor> value,
    bool allow_undefined = false);

// 将一个 std::vector<at::Tensor> 对象作为输入添加到给定的节点中
TORCH_API void addInputs(
    Node* n,
    const char* name,
    std::vector<at::Tensor> value,
    bool allow_undefined = false);

// 将一个 at::ITensorListRef 对象作为输入添加到给定的节点中
TORCH_API void addInputs(
    Node* n,
    const char* name,
    at::ITensorListRef value,
    bool allow_undefined = false);
    // 定义一个指向 Node 类型的指针 n，
    // 该指针指向一个节点对象，用于处理树或链表结构中的节点
    Node* n,
    // 定义一个指向常量字符的指针 name，
    // 用于表示参数的名称或标识符，通常用于字符串的传递
    const char* name,
    // 定义一个 List 对象，该 List 的元素是 std::optional<at::Tensor> 类型的引用，
    // 表示一个可能为空的张量的列表
    const List<std::optional<at::Tensor>>& value);
    // value 是一个输入参数，表示包含张量（Tensor）的列表引用，
    // 使用了 C++ 中的模板和引用来实现参数传递
TORCH_API void addInputs(
    Node* n,
    const char* name,
    ArrayRef<c10::intrusive_ptr<c10::ivalue::Object>> value,
    const c10::ClassTypePtr& class_type);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为 `ArrayRef` 类型的对象指针数组，
// 并指定对象的类类型为 `class_type`。

TORCH_API void addInputs(Node* n, const char* name, ArrayRef<double> value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为 `ArrayRef` 类型的双精度浮点数数组。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<ArrayRef<double>>& value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为可选的 `ArrayRef` 类型的双精度浮点数数组。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::string_view value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为 `c10::string_view` 类型的字符串视图。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<c10::string_view>& value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为可选的 `c10::string_view` 类型的字符串视图。

TORCH_API void addInputs(Node* n, const char* name, at::Device value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为 `at::Device` 类型的设备。

TORCH_API void addInputs(Node* n, const char* name, c10::Stream stream);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为 `c10::Stream` 类型的流对象。

TORCH_API void addInputs(Node* n, const char* name, at::Layout value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为 `at::Layout` 类型的张量布局。

TORCH_API void addInputs(Node* n, const char* name, at::ScalarType value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为 `at::ScalarType` 类型的张量标量类型。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::ScalarType>& value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为可选的 `at::ScalarType` 类型的张量标量类型。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Device>& value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为可选的 `at::Device` 类型的设备。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Layout>& value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为可选的 `at::Layout` 类型的张量布局。

TORCH_API void addInputs(Node* n, const char* name, at::MemoryFormat value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为 `at::MemoryFormat` 类型的内存格式。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    std::optional<at::DimnameList> value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为可选的 `at::DimnameList` 类型的维度名称列表。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::MemoryFormat>& value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为可选的 `at::MemoryFormat` 类型的内存格式。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Generator>& value);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为可选的 `at::Generator` 类型的生成器对象。

inline void addInputs(
    Node* n,
    const char* name,
    const std::vector<bool>& value) {
  AT_ERROR("Tracing a list of bool type is currently not supported!");
}
// 内联函数：添加输入参数到节点 `n` 中，参数名为 `name`，值为 `std::vector<bool>` 类型的布尔数组，
// 抛出错误，表示当前不支持跟踪布尔类型的数组。

template <typename T>
void addInputs(Node* n, const char* name, ArrayRef<T> value) {
  AT_ERROR("Tracing a list of arbitrary type is currently not supported!");
}
// 模板函数：添加输入参数到节点 `n` 中，参数名为 `name`，值为 `ArrayRef` 类型的任意类型数组，
// 抛出错误，表示当前不支持跟踪任意类型的数组。

template <typename K, typename V>
void addInputs(
    Node* n,
    const char* name,
    const std::unordered_map<K, V>& value) {
  AT_ERROR("Tracing a dict of arbitrary types is currently not supported!");
}
// 模板函数：添加输入参数到节点 `n` 中，参数名为 `name`，值为 `std::unordered_map` 类型的任意键值对，
// 抛出错误，表示当前不支持跟踪任意类型的字典。

template <size_t N>
void addInputs(Node* n, const char* name, std::array<bool, N> value) {
  throw std::runtime_error(
      "Found an unsupported argument type in the JIT tracer. File a bug report.");
}
// 模板函数：添加输入参数到节点 `n` 中，参数名为 `name`，值为 `std::array<bool, N>` 类型的布尔数组，
// 抛出运行时错误，表示发现了不支持的参数类型，建议提交错误报告。

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::intrusive_ptr<c10::ivalue::Object>& obj);
// 添加输入参数到节点 `n` 中，参数名为 `name`，值为 `c10::intrusive_ptr<c10::ivalue::Object>` 类型的对象指针。

TORCH_API void ensureUniqueIfOutOfPlaced(
    const char* name,
    const at::Tensor& tensor);
// 确保在不合适的情况下保持唯一性，参数名为 `name`，值为 `at::Tensor` 类型的张量。

TORCH_API void ensureUniqueIfOutOfPlaced(
    const char* name,
    const std::optional<at::Tensor>& tensor);
// 确保在不合适的情况下保持唯一性，参数名为 `name`，值为可选的 `at::Tensor` 类型的张量。
    # 定义一个类型别名 `typename`，使用 `std::enable_if_t` 条件模板来限制 T 类型：
    # - T 不能转换为 `at::TensorList` 类型
    # - T 不能转换为 `c10::List<at::Tensor>` 类型
    # - T 不能转换为 `at::Tensor` 类型
    # - T 不能转换为 `c10::intrusive_ptr<c10::ivalue::Object>` 类型
    typename = std::enable_if_t<
        (!std::is_convertible_v<std::decay_t<T>, at::TensorList> &&
         !std::is_convertible_v<std::decay_t<T>, c10::List<at::Tensor>> &&
         !std::is_convertible_v<std::decay_t<T>, at::Tensor> &&
         !std::is_convertible_v<
             std::decay_t<T>,
             c10::intrusive_ptr<c10::ivalue::Object>>)>>
// 定义了一系列函数和方法，用于 TorchScript 的 JIT 追踪器
namespace torch::jit {

// 当传入的参数类型 T 不支持时，抛出错误信息
void addOutput(Node* node, T&&) {
  AT_ERROR(
      "Found an unsupported argument type ",
      c10::demangle_type<T>(),
      " in the JIT tracer. File a bug report.");
}

// 向节点添加输出张量
TORCH_API void addOutput(Node* node, const at::Tensor& tensor);

// 设置值的输出为指定张量
TORCH_API void setOutput(Value* value, const at::Tensor& output);

// 向节点添加输出张量列表
TORCH_API void addOutput(Node* node, const std::vector<at::Tensor>& list);

// 向节点添加输出张量列表
TORCH_API void addOutput(Node* node, const c10::List<at::Tensor>& list);

// 向节点添加输出为 IValue 对象的指针
TORCH_API void addOutput(
    Node* node,
    const c10::intrusive_ptr<c10::ivalue::Object>& output);

// 获取变量在指定维度上的大小
TORCH_API autograd::Variable getSizeOf(
    const autograd::Variable& var,
    int64_t dim);

// 获取变量中元素的总数
TORCH_API autograd::Variable getNumelOf(const autograd::Variable& var);

} // namespace tracer
} // namespace torch::jit
```