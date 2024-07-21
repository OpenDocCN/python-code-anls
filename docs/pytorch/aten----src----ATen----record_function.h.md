# `.\pytorch\aten\src\ATen\record_function.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/ivalue.h>
// 引入 ATen 库的 IValue 类定义

#include <ATen/core/operator_name.h>
// 引入 ATen 库的 operator_name 定义

#include <c10/macros/Export.h>
// 引入 c10 库的导出宏定义

#include <c10/util/Optional.h>
// 引入 c10 库的 Optional 类定义

#include <c10/util/SmallVector.h>
// 引入 c10 库的 SmallVector 类定义

#include <array>
// 引入标准库的 array 类

#include <functional>
// 引入标准库的 functional 头文件

#include <memory>
// 引入标准库的 memory 头文件

#include <variant>
// 引入标准库的 variant 头文件

namespace c10 {
class TORCH_API OperatorHandle;
}
// 命名空间 c10，包含 OperatorHandle 类定义

namespace at {

// Function name to record NCCL metadata
// 用于记录 NCCL 元数据的函数名称常量
extern TORCH_API const std::string kParamCommsCallName;

// Kind of record function scope;
// 记录函数作用域的枚举类型
enum class C10_API_ENUM RecordScope : uint8_t {
  // c10/ATen ops, autograd nodes
  FUNCTION = 0,
  // 从自动求导调用的函数/节点
  BACKWARD_FUNCTION,
  // TorchScript 函数，方法
  TORCHSCRIPT_FUNCTION,
  // Kernel 函数数据类型标签
  KERNEL_FUNCTION_DTYPE,
  // Torchbind 自定义类
  CUSTOM_CLASS,
  // 通用构建特性
  BUILD_FEATURE,
  // Kernel 函数数据类型标签
  LITE_INTERPRETER,
  // 用户定义的作用域 (例如，使用 record_function())
  USER_SCOPE,
  // 静态运行时的作用域，一个专用的 TorchScript 解释器
  STATIC_RUNTIME_OP,
  STATIC_RUNTIME_MODEL,
  NUM_SCOPES, // 必须是列表中的最后一个
};

} // namespace at

namespace std {
template <>
struct hash<at::RecordScope> {
  size_t operator()(const at::RecordScope& sc) const {
    return static_cast<std::size_t>(sc);
  }
};
} // namespace std

namespace at {

struct TORCH_API StringView {
  // 字符串视图结构体
  StringView() : StringView(nullptr) {}
  explicit StringView(const char* str_ptr)
      : owned_str_ptr_(nullptr), str_ptr_(str_ptr) {}
  explicit StringView(std::string str)
      : owned_str_ptr_(std::make_shared<std::string>(std::move(str))),
        str_ptr_(owned_str_ptr_->c_str()) {}

  const char* str() const {
    return str_ptr_;
  }

  friend std::ostream& operator<<(std::ostream& os, const StringView& dt) {
    os << dt.str();
    return os;
  }

  friend bool operator==(const StringView& lhs, const StringView& rhs) {
    return strcmp(lhs.str(), rhs.str()) == 0;
  }

  friend bool operator!=(const StringView& lhs, const StringView& rhs) {
    return !(lhs == rhs);
  }

 private:
  std::shared_ptr<std::string> owned_str_ptr_;
  const char* str_ptr_;
};

// Soft limit on the number of callbacks to use;
// 使用回调函数的软限制数量
constexpr std::size_t kSoftLimitCallbacks = 4;

// An abstract base class for various observer contexts that can be attached to
// the RecordFunction.
// 可以附加到 RecordFunction 的各种观察者上下文的抽象基类
struct ObserverContext {
  virtual ~ObserverContext() = default;

 protected:
  ObserverContext() = default;
};

// CallbackHandles 是一个 SmallVector，存储 uint64_t 类型的回调句柄，使用软限制数量 kSoftLimitCallbacks
typedef c10::SmallVector<uint64_t, kSoftLimitCallbacks> CallbackHandles;
// ObserverContextList 是一个 SmallVector，存储 unique_ptr 指向 ObserverContext 的指针，使用软限制数量 kSoftLimitCallbacks
typedef c10::SmallVector<std::unique_ptr<ObserverContext>, kSoftLimitCallbacks> ObserverContextList;
// RecordFunctionHandle 是一个 uint64_t 类型的句柄
typedef uint64_t RecordFunctionHandle;
// RecordFunction 结构体的前向声明
struct RecordFunction;

//
// PyTorch callbacks/observers API:
//
// PyTorch 回调函数/观察者 API：
/**
 * RecordFunctionCallback represents a pair of callbacks to be used with
 * RecordFunction, members:
 *   start, end - the callbacks to run when entering and exiting the scope;
 *     optionally, the start callback may return an ObserverContext which will
 *     be passed to the end callback, use appropriate constructor accordingly.
 *   needs_inputs - whether the callbacks need the inputs passed from the
 *     observed function/range; NOTE: passing the inputs incurs an additional
 *     overhead; sampling_probability - if not 1.0, then the callback is
 *     probabilistically sampled to run; NOTE: start and end callbacks always run as
 *     a pair and are sampled together; scopes - types of scopes to execute the
 *     callbacks on (see RecordScope); passing empty set means the callbacks will be
 *     executed for all possible scope types should_run - optional function that
 *     returns whether this callback should run; overwrites the effect of setting
 *     sampling_probability
 */
class TORCH_API RecordFunctionCallback {
 public:
  using StartCallback =
      std::unique_ptr<ObserverContext> (*)(const RecordFunction&);
  using EndCallback = void (*)(const RecordFunction&, ObserverContext*);

  // This interface supports observers that require passing an ObserverContext
  // between start and end callbacks.
  explicit RecordFunctionCallback(
      StartCallback start,
      EndCallback end = nullptr)
      : start_(start), end_(end) {
    // Initialize scopes to execute callbacks for all scope types by default
    scopes_.fill(true);
  }

  // Set whether the callbacks need to access inputs from the observed function/range
  RecordFunctionCallback& needsInputs(bool needs_inputs) {
    needs_inputs_ = needs_inputs;
    return *this;
  }

  // Set whether the callbacks need to access outputs from the observed function/range
  RecordFunctionCallback& needsOutputs(bool needs_outputs) {
    needs_outputs_ = needs_outputs;
    return *this;
  }

  // Set whether the callbacks need to access identifiers from the observed function/range
  RecordFunctionCallback& needsIds(bool needs_ids) {
    needs_ids_ = needs_ids;
    return *this;
  }

  // Set the probability of sampling the callback to run
  RecordFunctionCallback& samplingProb(double sampling_prob) {
    TORCH_CHECK(
        sampling_prob >= 0.0 && sampling_prob <= 1.0,
        "Invalid sampling probability");
    sampling_prob_ = sampling_prob;
    return *this;
  }

  // Set specific scopes where the callbacks should be executed
  RecordFunctionCallback& scopes(
      const std::unordered_set<RecordScope, std::hash<RecordScope>>& scopes) {
    if (!scopes.empty()) {
      scopes_.fill(false);  // Reset all scopes to false initially
      for (auto sc : scopes) {
        scopes_[static_cast<size_t>(sc)] = true;  // Enable specific scopes
      }
    } else {
      scopes_.fill(true);  // If no specific scopes provided, execute for all
    }
    return *this;
  }

  // Check if callbacks need inputs
  bool needsInputs() const {
    return needs_inputs_;
  }

  // Check if callbacks need outputs
  bool needsOutputs() const {
    return needs_outputs_;
  }

  // Check if callbacks need identifiers
  bool needsIds() const {
    return needs_ids_;
  }

  // Get the sampling probability of the callback
  double samplingProb() const {
    return sampling_prob_;
  }

  // Check if a specific scope type is enabled for callback execution
  bool checkScope(RecordScope sc) const {
    return scopes_[(size_t)sc];
  }

  // Get the start callback function
  StartCallback start() const {
    return start_;
  }

  // Get the end callback function
  EndCallback end() const {
    // 返回成员变量 end_
    return end_;
  }

 private:
  // 成员变量，用于存储开始回调函数
  StartCallback start_;
  // 成员变量，用于存储结束回调函数
  EndCallback end_;
  // 双精度浮点型成员变量，采样概率，默认为1.0
  double sampling_prob_ = 1.0;
  // 布尔类型的固定大小数组，表示不同记录范围的状态，默认全为false
  std::array<bool, static_cast<size_t>(RecordScope::NUM_SCOPES)> scopes_ = {};
  // 布尔类型成员变量，指示是否需要输入数据，默认为false
  bool needs_inputs_ = false;
  // 布尔类型成员变量，指示是否需要输出数据，默认为false
  bool needs_outputs_ = false;
  // 布尔类型成员变量，指示是否需要标识符，默认为false
  bool needs_ids_ = false;
// Notes:
//  - two types of callbacks are provided: thread local and global
//     - thread local callbacks are added/removed only for the given thread
//       and are stored locally for each thread and separately from the list
//       of the global callbacks
//     - global callbacks are stored in a single per process list and are
//       invoked by every RecordFunction, in addition to the thread local
//       callbacks specific to the given thread
//  - we allow the added callbacks to be sampled, by specifying a sampling
//    probability for each callback pair, if the start callback is
//    not picked to run, the corresponding end callback won't be called
//  - a typical use case for the global callbacks is passive monitoring
//    in the background (e.g. fleet-wide monitoring), without focusing on
//    the specific piece of code
//  - in contrast, thread local callbacks are enabled locally, on demand,
//    for the specific piece of code (range) and are not sampled
//  - a typical use case for thread local callbacks is profiler and code
//    execution tracer
//  - note, thread local callbacks are automatically propagated with
//    ThreadLocalState across JIT continuations and async tasks (at::launch)

// Definition of CallbackHandle type as an unsigned 64-bit integer
typedef uint64_t CallbackHandle;

// Constant for an invalid callback handle, initialized to zero
constexpr CallbackHandle INVALID_CALLBACK_HANDLE{0};

// Struct defining an entry for thread-local function callbacks
struct RecordFunctionCallbacksEntry {
  // Constructor initializing with a callback function and handle
  RecordFunctionCallbacksEntry(RecordFunctionCallback cb, CallbackHandle h)
      : callback_(cb), handle_(h) {}

  // Member variables
  RecordFunctionCallback callback_;  // Function callback
  bool enabled_{true};               // Flag indicating if the callback is enabled
  CallbackHandle handle_;            // Handle associated with the callback
};

// Alias for a vector of RecordFunctionCallbacksEntry, representing
// pairs of callbacks and unique identifiers
using RecordFunctionCallbacks = std::vector<RecordFunctionCallbacksEntry>;

// Struct representing step callbacks generated by callback managers
struct StepCallbacks {
  // Default constructor
  StepCallbacks() = default;
  
  // Constructor initializing with thread ID and record scope
  StepCallbacks(uint64_t thread_id, RecordScope scope)
      : thread_id_{thread_id}, scope_{scope} {}

  // Method to check if callbacks are empty
  bool empty() const {
    return callbacks_.empty();
  }

  // Struct defining start and end callback pairs
  struct StartEndPair {
    RecordFunctionCallback::StartCallback start_;  // Start callback function
    RecordFunctionCallback::EndCallback end_;      // End callback function
  };

  // Using a small vector to store StartEndPair instances, with a soft limit on size
  using StartEndPairs = c10::SmallVector<StartEndPair, kSoftLimitCallbacks>;

  // Member variables
  StartEndPairs callbacks_;     // Collection of start and end callback pairs
  uint64_t thread_id_{0};       // Thread ID associated with the callbacks
  RecordScope scope_{RecordScope::FUNCTION}; // Record scope of the callbacks
  bool needs_inputs_{false};    // Flag indicating if inputs are needed
  bool needs_outputs_{false};   // Flag indicating if outputs are needed
  bool needs_ids_{false};       // Flag indicating if IDs are needed
};
// 定义一个名为 RecordFunction 的结构体，用于记录函数执行的相关信息
struct TORCH_API RecordFunction {

  // 显式默认构造函数，创建一个记录函数作用域的实例：
  //   scope - 此函数跟踪的记录范围
  //   pre_sampled - 标识此 RecordFunction 是否已经使用 kLowProb 概率进行了预采样
  explicit RecordFunction(RecordScope scope = RecordScope::FUNCTION);

  // 带有步骤回调函数的构造函数
  explicit RecordFunction(StepCallbacks&& step_callbacks);

  // 在函数执行前调用的模板函数，记录函数参数和序列号：
  //   fn - 要执行的函数
  //   args - 函数参数的引用数组
  //   current_sequence_nr - 当前序列号，默认为 -1
  template <typename F>
  void before(
      F fn,
      c10::ArrayRef<const c10::IValue> args,
      int64_t current_sequence_nr = -1) {
    if (!isActive()) {
      return;
    }
    inputs_ = args; // 记录函数参数
    before(fn, current_sequence_nr); // 调用重载的 before 函数
  }

  // 重载的 before 函数，接受参数为 std::vector<IValue> 指针
  template <typename F>
  void before(
      F fn,
      const std::vector<IValue>* args,
      int64_t current_sequence_nr = -1) {
    before(
        std::move(fn),
        c10::ArrayRef<const c10::IValue>(args->data(), args->size()),
        current_sequence_nr);
  }

  // 重载的 before 函数，接受参数为 std::vector<IValue> 指针和关键字参数 map 的指针
  template <typename F>
  void before(
      F fn,
      const std::vector<IValue>* args,
      const std::unordered_map<std::string, IValue>* kwargs,
      int64_t current_sequence_nr = -1) {
    if (!isActive()) {
      return;
    }
    kwinputs_ = *kwargs; // 记录关键字参数
    before(std::move(fn), args, current_sequence_nr); // 调用重载的 before 函数
  }

  // 虚析构函数，调用结束回调函数
  virtual ~RecordFunction();

  // 删除拷贝构造函数和赋值运算符，禁止对象的拷贝和赋值
  RecordFunction(const RecordFunction&) = delete;
  RecordFunction& operator=(const RecordFunction&) = delete;

  // 返回函数名
  const char* name() const;

  // 返回序列号
  int64_t seqNr() const {
    return sequence_nr_;
  }

  // 返回函数输入参数数组的引用
  c10::ArrayRef<const IValue> inputs() const {
#ifndef NDEBUG
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        inputs_valid_, "Called inputs() outside RecordFunction start callback");
#endif
    return inputs_;
  }

  // 返回关键字参数的 map
  std::unordered_map<std::string, IValue> kwinputs() const {
#ifndef NDEBUG
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        inputs_valid_,
        "Called kwinputs() outside RecordFunction start callback");
#endif
    return kwinputs_;
  }

  // 返回函数输出参数数组的引用
  const std::vector<c10::IValue>& outputs() const {
    return outputs_;
  }

  // 设置函数输出参数数组
  void setOutputs(std::vector<c10::IValue>&& outputs) {
    outputs_ = std::move(outputs);
  }

  // 设置函数输出参数数组
  void setOutputs(c10::ArrayRef<c10::IValue> outputs) {
    outputs_ = outputs.vec();
  }

  // 返回输入参数的数量
  size_t num_inputs() const;

  // 返回输出参数的数量
  size_t num_outputs() const;

  // 获取此 RecordFunction 运行开始回调的线程 ID，用于编写线程安全的结束回调函数
  uint64_t threadId() const {
    return step_callbacks_.thread_id_;
  }

  // 对于反向函数 - 对应前向函数的线程 ID，否则为零；
  // 用于将反向函数与前向函数相关联
  uint64_t forwardThreadId() const {
    return fwd_thread_id_;
  }

  // 设置前向函数的线程 ID
  void setForwardThreadId(uint64_t thread_id) {
    fwd_thread_id_ = thread_id;
  }

  // 返回记录函数的作用域
  RecordScope scope() const {
  // 返回 step_callbacks_ 的 scope_ 属性
  return step_callbacks_.scope_;
}

// 返回当前线程的逻辑线程 ID
static uint64_t currentThreadId();

// 内部函数，不直接使用；用于 Python 上下文管理器

// 初始化 RecordFunction 成员并调用 start 回调函数
using schema_ref_t = std::reference_wrapper<const c10::FunctionSchema>;
void before(const char* name, int64_t sequence_nr = -1);
void before(std::string name, int64_t sequence_nr = -1);
void before(schema_ref_t schema, int64_t sequence_nr = -1);

// 设置用于分布式 profiling 的节点 ID
static void setDefaultNodeId(int64_t defaultNodeId);
// 获取用于分布式 profiling 的节点 ID
static int64_t getDefaultNodeId();

// 调用 end 回调函数。调用 end() 后，访问器将不再提供有用的结果。
void end();

// 内部使用，用于强制异步事件以进行分布式事件 profiling。
void _setAsync();

// 返回此 RecordFunction 是否对应于异步事件。
bool isAsync() const;

// 返回此 RecordFunction 是否对应于 NCCL 元数据收集。
bool isNcclMeta() const {
  return is_nccl_meta_;
}

// 内部使用，用于指示用于 Static Runtime 执行的输出变体。
void _setStaticRuntimeOutVariant();
bool isStaticRuntimeOutVariant() const;

// 返回 handle_ 属性的 RecordFunctionHandle。
RecordFunctionHandle handle() const {
  return handle_;
}

// 返回操作符名称的可选值。
std::optional<OperatorName> operator_name() const;

// 返回函数模式的可选副本，可能开销较大。
std::optional<FunctionSchema> operator_schema() const;

// 设置 handle_ 的值。
void setHandle(RecordFunctionHandle handle) {
  handle_ = handle;
}

// 返回此 RecordFunction 是否运行任何回调。
bool isActive() const {
  return !step_callbacks_.empty();
}

// 返回此 RecordFunction 是否需要输入。
bool needsInputs() const {
  return step_callbacks_.needs_inputs_;
}

// 返回此 RecordFunction 是否需要输出。
bool needsOutputs() const {
  return step_callbacks_.needs_outputs_;
}

// 返回 debug_handle_ 的值。
int64_t debugHandle() const {
  return debug_handle_;
}

// 设置 debug_handle_ 的值。
void setDebugHandle(int64_t debug_handle) {
  debug_handle_ = debug_handle;
}

// 使输入无效。
void invalidateInputs() {
#ifndef NDEBUG
    // 如果处于调试模式，标记输入不合法
    inputs_valid_ = false;
#endif
  }

 private:
  // 运行开始回调函数
  void runStartCallbacks();

  // 步骤回调函数集合
  StepCallbacks step_callbacks_;

  // 当 RecordFunction 可能处于活动状态但我们选择不使用观察器时（例如操作符未被观察），此布尔标志用于检查是否调用了开始回调
  bool called_start_callbacks_ = false;

#ifndef NDEBUG
  // 如果处于调试模式，标记输入不合法
  bool inputs_valid_ = false;
#endif

  // 存储具有事件元数据的各种 ObserverContext 对象，用于回调
  ObserverContextList ctx_;

  // 函数名称或模式引用
  std::variant<std::string, schema_ref_t> fn_;

  // 序列号，默认为 -1
  int64_t sequence_nr_ = -1;
  // 输入数据的常量引用数组
  c10::ArrayRef<const IValue> inputs_;
  // 关键字输入的无序映射
  std::unordered_map<std::string, IValue> kwinputs_;
  // 输出值的向量
  std::vector<c10::IValue> outputs_;

  // 用于反向函数的前向函数线程 ID
  uint64_t fwd_thread_id_ = 0;

  // 此 RecordFunction 的唯一标识符，用于跟踪范围的开始和结束
  RecordFunctionHandle handle_{0};

  // 此 record_function 是否对应异步事件。异步事件可能在不同线程完成，或者遵循类似 future 的使用模式。
  bool is_async_{false};

  // 调试句柄，用于惰性注释模块层次结构和调用堆栈
  // 在移动运行时特别有用，可以使用调试信息惰性符号化生成的调试句柄
  int64_t debug_handle_{-1};

  // 是否用于 Static Runtime 的输出变体运行
  bool is_static_runtime_out_variant_{false};

  // 是否用于 NCCL 元数据收集
  bool is_nccl_meta_{false};
};

// 获取给定作用域的步骤回调函数
TORCH_API StepCallbacks getStepCallbacks(RecordScope scope);

// 获取给定作用域的步骤回调函数，除非为空
TORCH_API std::optional<StepCallbacks> getStepCallbacksUnlessEmpty(
    RecordScope scope);

namespace detail {
// 在给定范围内使用 RecordFunction 记录函数，带有输入参数、函数和其他参数
template <typename Inputs, typename F, typename... Args>
void record_function_with_scope(
    RecordFunction& guard,
    F fn,
    const Inputs& inputs,
    Args&&... args) {
  if (guard.needsInputs()) {
    guard.before(
        fn,
        c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()),
        std::forward<Args>(args)...);
  } else {
    guard.before(fn, std::forward<Args>(args)...);
  }
}

// 在给定范围内使用 RecordFunction 记录函数，带有调试句柄、输入参数、函数和其他参数
template <typename Inputs, typename F, typename... Args>
void record_function_with_scope_and_debug_handle(
    RecordFunction& guard,
    F fn,
    int64_t debug_handle,
    const Inputs& inputs,
    Args&&... args) {
  guard.setDebugHandle(debug_handle);
  if (guard.needsInputs()) {
    guard.before(
        fn,
        c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()),
        std::forward<Args>(args)...);
  } else {
    guard.before(fn, std::forward<Args>(args)...);
  }
}

// 在给定范围内使用 RecordFunction 记录函数，带有输入参数、函数和其他参数
template <typename F, typename... Args>
void record_function_with_scope(
    RecordFunction& guard,
    F fn,
    c10::ArrayRef<const c10::IValue> inputs,
    // 定义一个模板函数，接受函数对象和参数包 Args&&... args
    // 返回通过 record_function_with_scope 函数模板实例化得到的结果
    // 使用 c10::ArrayRef 包装类型为 const c10::IValue 的数组作为第一个模板参数
    // guard 是记录函数的上下文，fn 是要记录的函数对象，inputs 是函数的输入参数
    // std::forward<Args>(args)... 将参数包展开并转发给 record_function_with_scope
  return record_function_with_scope<
      c10::ArrayRef<const c10::IValue>,
      F,
      Args...>(guard, std::move(fn), inputs, std::forward<Args>(args)...);
} // 结束 detail 命名空间

// 定义一个模板函数，记录函数调用时的范围和调试句柄
template <typename F, typename... Args>
void record_function_with_scope_and_debug_handle(
    RecordFunction& guard,           // 记录函数对象的引用
    F fn,                           // 要记录的函数对象
    int64_t debug_handle,           // 调试句柄
    c10::ArrayRef<const c10::IValue> inputs,  // 输入参数的引用数组
    Args&&... args) {               // 可变模板参数包
  return record_function_with_scope_and_debug_handle<
      c10::ArrayRef<const c10::IValue>,  // 返回类型是输入参数的引用数组
      F,                                // 函数对象类型
      Args...>(                         // 可变模板参数包
      guard, std::move(fn), debug_handle, inputs, std::forward<Args>(args)...);
}

} // 结束 namespace detail

// 可选参数 - 函数的序列号
#define RECORD_FUNCTION_WITH_SCOPE(scope, fn, inputs, ...) \
  at::RecordFunction guard(scope);                         \
  if (guard.isActive()) {                                  \
    ::at::detail::record_function_with_scope(              \
        guard, fn, inputs, ##__VA_ARGS__);                 \
  }

#define RECORD_FUNCTION_WITH_SCOPE_INPUTS_OUTPUTS( \
    scope, fn, inputs, outputs, ...)               \
  at::RecordFunction guard(scope);                 \
  if (guard.isActive()) {                          \
    if (guard.needsInputs()) {                     \
      guard.before(fn, inputs, ##__VA_ARGS__);     \
    } else {                                       \
      guard.before(fn, ##__VA_ARGS__);             \
    }                                              \
    if (guard.needsOutputs()) {                    \
      guard.setOutputs(outputs);                   \
    }                                              \
  }

#define RECORD_FUNCTION(fn, inputs, ...) \
  RECORD_FUNCTION_WITH_SCOPE(            \
      at::RecordScope::FUNCTION, fn, inputs, ##__VA_ARGS__)

#define RECORD_TORCHSCRIPT_FUNCTION(mn, inputs) \
  RECORD_FUNCTION_WITH_SCOPE(at::RecordScope::TORCHSCRIPT_FUNCTION, mn, inputs)

#define RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(fn, inputs, outputs, ...) \
  RECORD_FUNCTION_WITH_SCOPE_INPUTS_OUTPUTS(                          \
      at::RecordScope::FUNCTION, fn, inputs, outputs, ##__VA_ARGS__)

// 在 C++ 中自定义用户范围，类似于 Python 中的 'with record_function("..."):'
#define RECORD_USER_SCOPE(fn) \
  RECORD_FUNCTION_WITH_SCOPE( \
      at::RecordScope::USER_SCOPE, fn, c10::ArrayRef<const c10::IValue>{})

// RECORD_USER_SCOPE 带有输入参数
#define RECORD_USER_SCOPE_WITH_INPUTS(fn, inputs) \
  RECORD_FUNCTION_WITH_SCOPE(at::RecordScope::USER_SCOPE, fn, inputs)

// 辅助宏，传递用于后处理事件的调试句柄和输入参数
#define RECORD_WITH_SCOPE_DEBUG_HANDLE_AND_INPUTS(             \
    scope, fn, debug_handle, inputs, ...)                      \
  at::RecordFunction guard(scope);                             \
  if (guard.isActive()) {                                      \
    ::at::detail::record_function_with_scope_and_debug_handle( \
        guard, fn, debug_handle, inputs, ##__VA_ARGS__);       \
  }

// 辅助宏，记录具有调试句柄的 LITE INTERPETER 范围事件与输入参数
#define RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS( \
    fn, debug_handle, inputs)                           \
  RECORD_WITH_SCOPE_DEBUG_HANDLE_AND_INPUTS(            \
      at::RecordScope::LITE_INTERPRETER, fn, debug_handle, inputs)



    # 将参数 fn, debug_handle, inputs 传递给宏 RECORD_WITH_SCOPE_DEBUG_HANDLE_AND_INPUTS
    # 在宏中使用宏参数 at::RecordScope::LITE_INTERPRETER 作为记录作用域类型
    # fn: 函数名
    # debug_handle: 调试句柄
    # inputs: 输入参数
    RECORD_WITH_SCOPE_DEBUG_HANDLE_AND_INPUTS(
        at::RecordScope::LITE_INTERPRETER, fn, debug_handle, inputs)
// 定义宏 RECORD_OUTPUTS，用于在 kernel 启动后记录输出以绑定到生成它们的操作
#define RECORD_OUTPUTS(outputs)                                    \
  if (guard.needsOutputs()) {                                      \
    guard.setOutputs(                                              \
        std::vector<c10::IValue>(outputs.begin(), outputs.end())); \
  }

/**
 * addThreadLocalCallback 添加一个线程本地回调以与 RecordFunction 一起运行，
 * 返回一个句柄，用于与 removeThreadLocalCallback 结合使用
 */
TORCH_API CallbackHandle addThreadLocalCallback(RecordFunctionCallback cb);

/**
 * hasThreadLocalCallbacks 返回是否已注册了 addThreadLocalCallback 的回调函数
 */
TORCH_API bool hasThreadLocalCallbacks();

/**
 * clearThreadLocalCallbacks 移除所有线程本地回调
 */
TORCH_API void clearThreadLocalCallbacks();

/**
 * addGlobalCallback 添加一个全局回调以与 RecordFunction 一起运行：
 * 仅在程序初始化期间调用
 */
TORCH_API CallbackHandle addGlobalCallback(RecordFunctionCallback cb);

/**
 * removeCallback 根据 addThreadLocalCallback 或 addGlobalCallback 返回的句柄移除回调函数；
 * 不能同时运行其他代码
 */
TORCH_API void removeCallback(CallbackHandle handle);

/**
 * 禁止给定的回调函数执行。如果句柄无效，则不执行任何操作。
 */
TORCH_API void disableCallback(CallbackHandle handle);

/**
 * 允许先前使用 disableCallback 禁用的回调函数再次执行。如果句柄无效，则不执行任何操作。
 */
TORCH_API void reenableCallback(CallbackHandle handle);

/**
 * hasGlobalCallbacks 返回是否已注册使用 pushGlobalCallback 的全局回调
 */
TORCH_API bool hasGlobalCallbacks();

/**
 * clearGlobalCallbacks 移除所有全局回调
 */
TORCH_API void clearGlobalCallbacks();

// 适用于线程本地和全局回调
TORCH_API bool hasCallbacks();
TORCH_API void clearCallbacks();

/**
 * enableRecordFunction 在线程本地启用 RecordFunction
 */
TORCH_API void enableRecordFunction(bool enable = true);

/**
 * isRecordFunctionEnabled 返回线程本地 RecordFunction 是否已启用
 */
TORCH_API bool isRecordFunctionEnabled();

/**
 * RecordFunctionGuard 类用于在其范围内启用 RecordFunction
 */
class TORCH_API RecordFunctionGuard {
 public:
  explicit RecordFunctionGuard(bool is_enabled = true)
      : prev_value_(isRecordFunctionEnabled()) {
    enableRecordFunction(is_enabled);
  }

  virtual ~RecordFunctionGuard() {
    enableRecordFunction(prev_value_);
  }

 private:
  bool prev_value_ = false;
};

/**
 * DisableRecordFunctionGuard 类继承自 RecordFunctionGuard，用于在其范围内禁用 RecordFunction
 */
class TORCH_API DisableRecordFunctionGuard : public RecordFunctionGuard {
 public:
  DisableRecordFunctionGuard() : RecordFunctionGuard(false) {}
  ~DisableRecordFunctionGuard() override = default;
};
struct TORCH_API RecordFunctionTLS {
  // 定义了一个名为 RecordFunctionTLS 的结构体

  // 线程本地的回调函数向量，存储了回调函数与唯一标识的配对，
  // 必须按照句柄递增的顺序排序
  RecordFunctionCallbacks sorted_tls_callbacks_;

  // 标志位，用于指示是否启用线程本地记录函数
  bool tls_record_function_enabled_ = true;
};

// 获取记录函数线程本地存储的全局函数，返回一个常量引用
TORCH_API const RecordFunctionTLS& get_record_function_tls_();

// 设置记录函数线程本地存储的全局函数，传入一个 RecordFunctionTLS 类型的引用参数
TORCH_API void set_record_function_tls_(const RecordFunctionTLS& tls);

// 用于测试目的设置记录函数的种子值，传入一个 uint32_t 类型的种子参数
TORCH_API void set_record_function_seed_for_testing(uint32_t seed);

} // namespace at


这段代码定义了一个名为 `RecordFunctionTLS` 的结构体，该结构体包含了线程本地的回调函数向量和一个标志位，用于控制是否启用记录函数的功能。此外，还包括了几个函数接口用于获取和设置线程本地存储的全局函数以及用于测试目的设置记录函数的种子值。
```