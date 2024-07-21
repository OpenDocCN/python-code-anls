# `.\pytorch\aten\src\ATen\core\dispatch\Dispatcher.cpp`

```py
// 包含 ATen 库的调度器和 Python 操作注册的头文件
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/PythonOpRegistrationTrampoline.h>

// 包含用于时间操作、列表、字符串流和实用工具的标准库头文件
#include <chrono>
#include <list>
#include <sstream>
#include <utility>

// 在 FBCODE_CAFFE2 宏定义下，包含静态跟踪点库的头文件
#ifdef FBCODE_CAFFE2
#include <c10/util/static_tracepoint.h>
#endif

// 定义在 c10 命名空间中
namespace c10 {

// 在 FBCODE_CAFFE2 宏定义下，定义操作开始和结束的信号量
#ifdef FBCODE_CAFFE2
TORCH_SDT_DEFINE_SEMAPHORE(operator_start)
TORCH_SDT_DEFINE_SEMAPHORE(operator_end)
#endif

// 返回环境变量 TORCH_SHOW_DISPATCH_TRACE 是否设置的布尔值
bool show_dispatch_trace() {
    static char const* temp = getenv("TORCH_SHOW_DISPATCH_TRACE");
    return temp != nullptr;
}

// 定义线程局部变量 dispatch_trace_nesting_value_
static thread_local int64_t dispatch_trace_nesting_value_;

// dispatch_trace_nesting_value_ 自增和自减的函数定义
void dispatch_trace_nesting_incr() { ++dispatch_trace_nesting_value_; }
void dispatch_trace_nesting_decr() { --dispatch_trace_nesting_value_; }

// 返回 dispatch_trace_nesting_value_ 的函数定义
int64_t dispatch_trace_nesting_value() { return dispatch_trace_nesting_value_; }

// 定义在 detail 命名空间中的注册监听器列表类
namespace detail {

class RegistrationListenerList final {
public:
  // 添加监听器的函数，返回用于删除监听器的函数
  std::function<void()> addListener(std::unique_ptr<OpRegistrationListener> listener) {
    listeners_.push_back(std::move(listener));
    auto delete_it = --listeners_.end();
    return [this, delete_it] {
        listeners_.erase(delete_it);
    };
  }

  // 调用注册操作时通知所有监听器的函数
  void callOnOperatorRegistered(const OperatorHandle& op) {
    for (auto& listener : listeners_) {
      listener->onOperatorRegistered(op);
    }
  }

  // 调用注销操作时通知所有监听器的函数
  void callOnOperatorDeregistered(const OperatorHandle& op) {
    for (auto& listener : listeners_) {
      listener->onOperatorDeregistered(op);
    }
  }
private:
  std::list<std::unique_ptr<OpRegistrationListener>> listeners_;
};

} // namespace detail

// OpRegistrationListener 析构函数的默认定义
OpRegistrationListener::~OpRegistrationListener()= default;

// Dispatcher 类的构造函数的定义
Dispatcher::Dispatcher()
: operators_()
, operatorLookupTable_()
, backendFallbackKernels_()
, listeners_(std::make_unique<detail::RegistrationListenerList>())
, cond_var_()
, guard_(std::make_shared<Guard>())
{}

// Dispatcher 类的析构函数的定义
Dispatcher::~Dispatcher() {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  guard_->alive.store(false);
}

// 返回 Dispatcher 类的实例的函数定义
C10_EXPORT Dispatcher& Dispatcher::realSingleton() {
  static Dispatcher _singleton;
  return _singleton;
}

// 根据重载名查找操作的函数定义
std::optional<OperatorHandle> Dispatcher::findOp(const OperatorName& overload_name) {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::optional<OperatorHandle> {
    auto found = operatorLookupTable.find(overload_name);
    if (found == operatorLookupTable.end()) {
      return c10::nullopt;
    }
    return found->second;
  });
}

// 注意事项: 如果添加了更多的 waitFor* 实现，必须在相关的注册调用中添加适当的 notify_all() 调用

// 等待定义的函数模式的实现
void Dispatcher::waitForDef(const FunctionSchema& schema) {
  using namespace std::chrono_literals;
  std::unique_lock<std::mutex> lock(guard_->mutex);
  bool r = cond_var_.wait_for(lock, 2s, [&]{
    return findOp(schema.operator_name()) != c10::nullopt;
  });
  TORCH_INTERNAL_ASSERT(r,
    "Expected main interpreter to define ", schema.operator_name(),
    ", but this didn't happen within timeout.  Are you trying to load "
    "different models in the same torchdeploy/multipy instance?  You "
    "must warmup each interpreter identically, e.g., import all "
    "the same dependencies.");
}

// 等待指定操作符名称的实现，支持可选的分发键
void Dispatcher::waitForImpl(const OperatorName& op_name, std::optional<c10::DispatchKey> maybe_dk) {
  // 引入命名空间，使代码更清晰
  using namespace std::chrono_literals;
  // 获取互斥锁，确保线程安全
  std::unique_lock<std::mutex> lock(guard_->mutex);
  // 获取或设置分发键，默认为CompositeImplicitAutograd
  auto dk = maybe_dk.value_or(DispatchKey::CompositeImplicitAutograd);
  // 查找或注册操作符名称
  auto op = findOrRegisterName_(op_name);
  // 等待条件变量，最长等待2秒，直到操作符支持指定的分发键
  bool r = cond_var_.wait_for(lock, 2s, [&]{
    // 注意：对于重载的情况，这种做法略有不足，但重载本身就是个特殊情况
    return op.hasKernelForDispatchKey(dk);
  });
  // 内部断言，确保等待成功
  TORCH_INTERNAL_ASSERT(r,
    "Expected main interpreter to implement ", dk, " for ", op_name,
    ", but this didn't happen within timeout.  Are you trying to load "
    "different models in the same torchdeploy/multipy instance?  You "
    "must warmup each interpreter identically, e.g., import all "
    "the same dependencies.");
}

// 查找给定重载名称的操作符模式，若找到返回其句柄，否则返回空
std::optional<OperatorHandle> Dispatcher::findSchema(const OperatorName& overload_name) {
  // 查找操作符名称在操作符表中的位置
  auto it = findOp(overload_name);
  if (it.has_value()) {
    // 如果找到了操作符并且其具有模式(schema)
    if (it->hasSchema()) {
      return it;
    } else {
      return c10::nullopt;
    }
  } else {
    return it;
  }
}

// 查找给定名称和重载名称的操作符模式，若找不到则抛出异常
OperatorHandle Dispatcher::findSchemaOrThrow(const char* name, const char* overload_name) {
  // 查找操作符的模式
  auto it = findSchema({name, overload_name});
  if (!it.has_value()) {
    // 如果找不到模式，则检查是否找到了任何操作符实现
    auto it2 = findOp({name, overload_name});
    if (!it2.has_value()) {
      // 抛出错误，指出找不到指定名称和重载名称的模式
      TORCH_CHECK(false, "Could not find schema for ", name, ".", overload_name);
    } else {
      // 抛出错误，指出找到了实现但忘记定义操作符的模式
      TORCH_CHECK(false, "Could not find schema for ", name, ".", overload_name,
        " but we found an implementation; did you forget to def() the operator?");
    }
  }
  // 返回找到的操作符句柄
  return it.value();
}

// 获取所有操作符名称的列表
const std::vector<OperatorName> Dispatcher::getAllOpNames() {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::vector<OperatorName> {
    std::vector<OperatorName> allOpNames;
    // 遍历操作符表，将所有操作符名称添加到列表中
    for (const auto& op : operatorLookupTable) {
        allOpNames.push_back(op.first);
    }
    return allOpNames;
  });
}

// 向操作符表中查找或注册给定名称的操作符，返回操作符句柄
// 调用者需在完成后负责注销注册
OperatorHandle Dispatcher::findOrRegisterName_(const OperatorName& op_name) {
  // 查找操作符名称是否已经在操作符表中
  const auto found = findOp(op_name);
  if (found != c10::nullopt) {
    // 如果找到，则返回已存在的操作符句柄
    return *found;
  }

  // 如果没有找到，则注册新的操作符名称
  operators_.emplace_back(OperatorName(op_name));
  OperatorHandle handle(--operators_.end());
  // 在操作符查找表中注册新的操作符名称和句柄
  operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
    operatorLookupTable.emplace(op_name, handle);
  });

  // 返回新注册的操作符句柄
  return handle;
}

// 显式定义析构函数以解决Windows构建中的链接错误
// Windows构建中，PyTorch库不生成析构符号，导致下游项目的链接失败
// 参考：https://github.com/pytorch/pytorch/issues/70032
OperatorHandle::~OperatorHandle() = default;
// 注册一个库命名空间，确保互斥访问
RegistrationHandleRAII Dispatcher::registerLibrary(std::string ns, std::string debug) {
  // 使用互斥锁保护临界区域
  std::lock_guard<std::mutex> lock(guard_->mutex);
  // 查找是否已经存在相同命名空间的注册
  auto found = libraries_.find(ns);
  // 检查是否找到已注册的命名空间，如果找到则抛出错误信息
  TORCH_CHECK(
    found == libraries_.end(),
    "Only a single TORCH_LIBRARY can be used to register the namespace ", ns,
    "; please put all of your definitions in a single TORCH_LIBRARY block.  "
    "If you were trying to specify implementations, consider using TORCH_LIBRARY_IMPL "
    "(which can be duplicated).  If you really intended to define operators for a "
    "single namespace in a distributed way, you can use TORCH_LIBRARY_FRAGMENT to "
    "explicitly indicate this.  "
    "Previous registration of TORCH_LIBRARY was ",
    found->second, "; latest registration was ", debug
  );
  // 将命名空间及其调试信息添加到注册表中
  libraries_.emplace(ns, std::move(debug));
  // 返回注册句柄对象，用于取消注册
  return RegistrationHandleRAII([guard = this->guard_, this, ns] {
    // 使用互斥锁保护临界区域
    std::lock_guard<std::mutex> lock(guard->mutex);
    // 如果不再存活，直接返回
    if (!guard->alive.load()) {
      return;
    }
    // 取消注册该命名空间下的库
    deregisterLibrary_(ns);
  });
}

void Dispatcher::deregisterLibrary_(const std::string& ns) {
  // 需要使用锁以避免并发写入
  libraries_.erase(ns);
}

// 注册一个函数定义及其相关信息
RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug, std::vector<at::Tag> tags) {
  // 需要使用锁以避免并发写入
  std::lock_guard<std::mutex> lock(guard_->mutex);

  // 获取操作符名称并查找或注册它
  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);

  // 检查是否已经注册过相同名称和重载名的操作符
  TORCH_CHECK(op.operatorDef_->def_count == 0, "Tried to register an operator (", schema, ") with the same name and overload name multiple times.",
                                                    " Each overload's schema should only be registered with a single call to def().",
                                                    " Duplicate registration: ", debug, ". Original registration: ", op.operatorDef_->op.debug());
  // 向操作符注册表注册模式及其调试信息和标签
  op.operatorDef_->op.registerSchema(std::move(schema), std::move(debug), std::move(tags));
  // 调用操作符注册监听器
  listeners_->callOnOperatorRegistered(op);

  // 注意：在错误检查之后不要增加计数
  // 增加操作符定义计数
  ++op.operatorDef_->def_count;
  // 增加操作符定义和实现计数
  ++op.operatorDef_->def_and_impl_count;

  // 通知所有等待条件变量
  cond_var_.notify_all();

  // 返回注册句柄对象，用于取消注册
  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name] {
    // 需要使用锁以避免并发写入
    std::lock_guard<std::mutex> lock(guard->mutex);
    // 如果不再存活，直接返回
    if (!guard->alive.load()) {
      return;
    }
    // 取消注册该操作符定义
    deregisterDef_(op, op_name);
  });
}

void Dispatcher::deregisterDef_(
    const OperatorHandle& op,
    const OperatorName& op_name) {
  // 断言操作符的名称与操作符名相符
  TORCH_INTERNAL_ASSERT(op.schema().operator_name() == op_name);

  // 减少操作符定义计数并在没有引用时取消注册
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_count > 0);
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);

  // 减少操作符定义计数
  --op.operatorDef_->def_count;
  // 减少操作符定义和实现计数
  --op.operatorDef_->def_and_impl_count;
  // 如果定义计数为零，则取消注册该操作符
  if (0 == op.operatorDef_->def_count) {
    // 在移除操作符之前调用监听器，确保调度器在操作符被移除时仍然有效
    // TODO: 检查监听器是否依赖于 prepareForDeregistration() 方法
    // 不变条件：确保操作符被注销前，监听器被通知
    listeners_->callOnOperatorDeregistered(op);
    // 调用操作符的 deregisterSchema() 方法，从操作符定义中取消注册模式
    op.operatorDef_->op.deregisterSchema();
  }

  // 调用 cleanup 函数清理操作符相关资源
  cleanup(op, op_name);
}

namespace {

// OperatorName 到 (Python 模块名称, 描述) 的映射
using PythonModuleMapType = std::unordered_map<at::OperatorName, std::pair<const char*, const char*>>;

// 返回 Python 模块的单例映射
PythonModuleMapType& pythonModulesSingleton() {
  static PythonModuleMapType _data;
  return _data;
}

}

// 获取给定 OperatorName 对应的 Python 模块信息（如果存在）
std::optional<std::pair<const char*, const char*>> Dispatcher::getPyStub(OperatorName op_name) {
  // 使用互斥锁保护访问单例映射
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto found = pythonModulesSingleton().find(op_name);
  if (found == pythonModulesSingleton().end()) {
    return c10::nullopt;
  }
  return found->second;
}

// 注册 Python 模块的注册信息，并返回注册的句柄对象
RegistrationHandleRAII Dispatcher::registerPythonModule(
  const OperatorName& op_name,
  const char* pymodule,
  const char* context
) {
  // 使用互斥锁保护访问单例映射
  std::lock_guard<std::mutex> lock(guard_->mutex);
  
  // 检查是否已存在同名的 Python 模块注册信息
  auto found = pythonModulesSingleton().find(op_name);
  if (found != pythonModulesSingleton().end()) {
    // 发出警告，说明即将覆盖现有的 Python 模块注册信息
    TORCH_WARN(
        "Tried to register an python registration stub (pystub) for ", op_name, " ",
        "that specifies the Python module ", pymodule, " "
        "but there already was a pystub that specifies the Python module ",
        found->second.first, ". We will override the existing pystub.");
  }
  
  // 将新的 Python 模块注册信息存入单例映射中
  pythonModulesSingleton()[op_name] = std::make_pair(pymodule, context);
  
  // 返回注册信息的句柄对象，用于后续管理
  return RegistrationHandleRAII([guard = this->guard_, op_name] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    // 如果资源已不再有效，则不执行任何操作
    if (!guard->alive.load()) {
      return;
    }
    // 在析构时从单例映射中删除注册信息
    pythonModulesSingleton().erase(op_name);
  });
}

// 如果给定的 OperatorName 存在对应的 Python 模块信息，则抛出错误
void Dispatcher::throwIfHasPythonModule(OperatorName op_name) {
  // 使用互斥锁保护访问单例映射
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto elt = pythonModulesSingleton().find(op_name);
  if (elt == pythonModulesSingleton().end()) {
    return;
  }
  
  // 获取相关的 Python 模块名称和描述
  const char* pymodule = elt->second.first;
  const char* context = elt->second.second;
  
  // 获取 Python 解释器实例，并检查是否可用
  auto* interpreter = at::impl::PythonOpRegistrationTrampoline::getInterpreter();
  TORCH_CHECK(
      interpreter != nullptr,
      op_name,
      ": while attempting to run this operator with Meta Tensors: "
      "Either there is no meta kernel for this operator, or it is located "
      "in the python module ", pymodule, " which is not available "
      "because Python isn't available.")
  
  // 如果 Python 模块不可用，抛出错误
  (*interpreter)->throw_abstract_impl_not_imported_error(toString(op_name), pymodule, context);
}

// 注册 OperatorName 对应的具体实现信息
RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  std::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  // 使用互斥锁保护访问
  std::lock_guard<std::mutex> lock(guard_->mutex);

  // 查找或注册 OperatorName 对应的名称
  auto op = findOrRegisterName_(op_name);

  // 调用操作的注册内核方法，注册具体的实现信息
  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  );

  // 返回注册信息的句柄对象，用于后续管理
  return handle;
}
    std::move(cpp_signature),
    // 使用 std::move 转移 cpp_signature 到构造函数中作为参数，cpp_signature 可能是某个类型的对象或值

    std::move(inferred_function_schema),
    // 使用 std::move 转移 inferred_function_schema 到构造函数中作为参数，inferred_function_schema 可能是某个类型的对象或值

    std::move(debug)
    // 使用 std::move 转移 debug 到构造函数中作为参数，debug 可能是某个类型的对象或值
  );

  ++op.operatorDef_->def_and_impl_count;
  // 增加 op 对象的 def_and_impl_count 属性值

  cond_var_.notify_all();
  // 唤醒所有等待 cond_var_ 的线程

  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name, dispatch_key, handle] {
    // 返回一个 RegistrationHandleRAII 对象，使用 lambda 表达式进行构造

    std::lock_guard<std::mutex> lock(guard->mutex);
    // 在当前作用域内使用 guard->mutex 创建一个互斥锁

    if (!guard->alive.load()) {
      return;
    }
    // 如果 guard 的 alive 标志位为 false，直接返回，不执行后续代码

    deregisterImpl_(op, op_name, dispatch_key, handle);
    // 否则调用 deregisterImpl_ 方法，传递 op, op_name, dispatch_key, handle 作为参数
  });
  // lambda 表达式结束
}

// 实现 Dispatcher 类中的 deregisterImpl_ 函数，用于注销操作符实现
void Dispatcher::deregisterImpl_(const OperatorHandle& op, const OperatorName& op_name, std::optional<DispatchKey> dispatch_key, impl::OperatorEntry::AnnotatedKernelContainerIterator handle) {
  // 调用操作符的 deregisterKernel_ 函数，注销指定调度键的内核实现
  op.operatorDef_->op.deregisterKernel_(*this, dispatch_key, handle);

  // 断言操作符的名称与给定的操作符名相同
  TORCH_INTERNAL_ASSERT(op.operator_name() == op_name);

  // 断言操作符定义和实现计数大于零，减少定义和实现计数
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);
  --op.operatorDef_->def_and_impl_count;

  // 调用 cleanup 函数，清理操作符相关资源
  cleanup(op, op_name);
}

// 实现 Dispatcher 类中的 registerName 函数，注册操作符名称
RegistrationHandleRAII Dispatcher::registerName(OperatorName op_name) {
  // 使用互斥锁保护共享数据
  std::lock_guard<std::mutex> lock(guard_->mutex);
  // 查找或注册给定操作符名的操作符
  auto op = findOrRegisterName_(op_name);
  // 增加操作符定义和实现计数
  ++op.operatorDef_->def_and_impl_count;

  // 返回注册处理的 RAII 包装对象
  return RegistrationHandleRAII(
      [guard = this->guard_, this, op, op_name] {
        std::lock_guard<std::mutex> lock(guard->mutex);
        // 如果 guard 不再有效，直接返回
        if (!guard->alive.load()) {
          return;
        }
        // 注销操作符名称
        deregisterName_(op, op_name);
      }
  );
}

// 实现 Dispatcher 类中的 deregisterName_ 函数，注销操作符名称
void Dispatcher::deregisterName_(
    const OperatorHandle& op,
    const OperatorName& op_name) {
  // 断言操作符的名称与给定的操作符名相同
  TORCH_INTERNAL_ASSERT(op.operator_name() == op_name);
  // 断言操作符定义和实现计数大于零，减少定义和实现计数
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);
  --op.operatorDef_->def_and_impl_count;
  // 调用 cleanup 函数，清理操作符相关资源
  cleanup(op, op_name);
}

// 实现 Dispatcher 类中的 cleanup 函数，清理不再使用的操作符资源
// 如果操作符定义和实现计数为零，则删除相应的操作符条目
void Dispatcher::cleanup(const OperatorHandle& op, const OperatorName& op_name) {
  if (0 == op.operatorDef_->def_and_impl_count) {
    // 注意：OperatorHandle 存储 operatorIterator_ 是为了加速此调用！
    // 从 operators_ 中删除操作符迭代器
    operators_.erase(op.operatorIterator_);
    // 写操作操作符查找表，删除指定操作符名的条目
    operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
      operatorLookupTable.erase(op_name);
    });
  }
}

// 实现 Dispatcher 类中的 registerFallback 函数，注册后备内核函数
RegistrationHandleRAII Dispatcher::registerFallback(DispatchKey dispatchKey, KernelFunction kernel, std::string debug) {
  // 使用互斥锁保护共享数据
  std::lock_guard<std::mutex> lock(guard_->mutex);

  // 获取给定调度键的调度表索引
  auto idx = getDispatchTableIndexForDispatchKey(dispatchKey);
  // 检查索引有效性
  TORCH_CHECK(idx >= 0 && static_cast<uint64_t>(idx) < backendFallbackKernels_.size(), "idx=", idx);
  // 检查是否已经注册过相同调度键的后备内核函数
  TORCH_CHECK(
    !backendFallbackKernels_[idx].kernel.isValid(),
    "Tried to register multiple backend fallbacks for the same dispatch key ", dispatchKey, "; previous registration ",
    backendFallbackKernels_[idx].debug, ", new registration ", debug
  );

  // 注册后备内核函数，推测函数模式总是为 nullptr，因为后备函数不能被拆箱
  backendFallbackKernels_[idx] = impl::AnnotatedKernel(std::move(kernel), nullptr, std::move(debug));

  // 更新所有操作符的后备内核函数
  for (auto& op : operators_) {
    op.op.updateFallback(*this, dispatchKey);
  }

  // 返回注册处理的 RAII 包装对象
  return RegistrationHandleRAII([guard = this->guard_, this, dispatchKey] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    // 如果 guard 不再有效，直接返回
    if (!guard->alive.load()) {
      return;
    }
    // 注销后备内核函数
    deregisterFallback_(dispatchKey);
  });
}
// 根据给定的 dispatchKey 找到对应的调度表索引
void Dispatcher::deregisterFallback_(DispatchKey dispatchKey) {
  auto idx = getDispatchTableIndexForDispatchKey(dispatchKey);
  // 清空对应索引处的后备内核
  backendFallbackKernels_[idx] = {};

  // 遍历所有操作符并更新其后备处理逻辑
  for (auto& op : operators_) {
    op.op.updateFallback(*this, dispatchKey);
  }
}

// 添加一个注册监听器，返回一个 RegistrationHandleRAII 对象来管理监听器的生命周期
RegistrationHandleRAII Dispatcher::addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener) {
  // 获取锁保证线程安全
  std::lock_guard<std::mutex> lock(guard_->mutex);

  // 遍历操作符列表，对于每个已定义的操作符，调用监听器的注册回调
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    if (iter->def_count > 0) {
      listener->onOperatorRegistered(OperatorHandle(iter));
    }
  }

  // 将监听器添加到监听器管理对象中，并返回一个 RegistrationHandleRAII 对象
  auto removeListener = listeners_->addListener(std::move(listener));
  return RegistrationHandleRAII([guard = this->guard_, this, removeListener] {
      std::lock_guard<std::mutex> lock(guard_->mutex);
      // 如果 guard 已经被销毁，直接返回
      if (!guard->alive.load()) {
        return;
      }
      // 否则移除监听器
      removeListener();
  });
}

// 检查调度器中的不变量
void Dispatcher::checkInvariants() const {
  // 遍历所有操作符并检查其不变量
  for (const auto& op : operators_) {
    op.op.checkInvariants();
  }
}

// 查找所有具有悬空实现的操作符，并返回其操作符句柄列表
std::vector<OperatorHandle> Dispatcher::findDanglingImpls() const {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::vector<OperatorHandle> {
    std::vector<OperatorHandle> opsWithDanglingImpls;
    // 遍历操作符查找表，找到所有没有有效实现的操作符，并将其操作符句柄加入列表
    for (const auto& op : operatorLookupTable) {
      if (!op.second.hasSchema()) {
        opsWithDanglingImpls.push_back(op.second);
      }
    }
    return opsWithDanglingImpls;
  });
}

// 获取指定调度键对应的所有注册操作符的名称列表
std::vector<OperatorName> Dispatcher::getRegistrationsForDispatchKey(std::optional<DispatchKey> k) const {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::vector<OperatorName> {
    std::vector<OperatorName> op_names;
    // 遍历操作符查找表，如果没有指定 DispatchKey 或者当前操作符支持指定的 DispatchKey，则将操作符名称加入列表
    for (const auto& op : operatorLookupTable) {
      if (!k || op.second.hasKernelForDispatchKey(*k)) {
          op_names.push_back(op.first);
      }
    }
    return op_names;
  });
}
// 返回正在运行记录函数的序列号，根据给定的调度键和调度键集合
int64_t Dispatcher::sequenceNumberForRunningRecordFunction(DispatchKey dispatchKey, DispatchKeySet dispatchKeySet) {
  int64_t seq_num = -1;
  
  // 在 Autograd 情况下设置序列号，以将前向范围与相应的 Autograd 节点关联起来

  // 注意：这会为 Autograd 键和仍包含 Autograd 键的非 Autograd 键记录序列号。
  // 这意味着如果在 Autograd 上下文中发生两个不同事件，并且调度键集合仍然包含 Autograd 键，
  // 那么可能会收集相同的序列号两次。
  // 不过，这种情况通常不会发生：通常第一次调用会通过调用路径（call() 或 callBoxed()）进入分发器，
  // 而后续的重新调度会通过 redispatch() 或 redispatchBoxed() 进行。
  // `call` 具有性能分析工具的仪器化，而 `redispatch` 没有。
  // 因此，通常情况下，如果调度键包含 Autograd，则第一次调用() 会收集序列号，而后续重新调度则不会。
  
  bool dispatchHasAutograd = !(dispatchKeySet & autograd_dispatch_keyset).empty();

  // 如果存在 Autograd 并且 GradMode 已启用，则获取当前的序列号
  if (dispatchHasAutograd && at::GradMode::is_enabled()) {
    seq_num = at::sequence_number::peek();
  }
  
  return seq_num;
}

// 运行记录函数，在函数运行之前调用 guard.before() 方法，记录函数的模式引用和序列号
void Dispatcher::runRecordFunction(at::RecordFunction& guard, at::RecordFunction::schema_ref_t schema_ref, DispatchKey dispatchKey, DispatchKeySet dispatchKeySet, c10::ArrayRef<const c10::IValue> args) {
  guard.before(schema_ref, args, sequenceNumberForRunningRecordFunction(dispatchKey, dispatchKeySet));
}

// 运行记录函数，在函数运行之前调用 guard.before() 方法，记录函数的模式引用和序列号（无参数版本）
void Dispatcher::runRecordFunction(at::RecordFunction& guard, at::RecordFunction::schema_ref_t schema_ref, DispatchKey dispatchKey, DispatchKeySet dispatchKeySet) {
  guard.before(schema_ref, sequenceNumberForRunningRecordFunction(dispatchKey, dispatchKeySet));
}

#ifdef FBCODE_CAFFE2
// 返回是否对操作符事件进行性能分析
bool Dispatcher::profilingOperatorEvents() {
  return TORCH_SDT_IS_ENABLED(operator_start) || TORCH_SDT_IS_ENABLED(operator_end);
}

// 触发操作开始的 USDT 事件
C10_NOINLINE void Dispatcher::fireOpStartUSDT(at::RecordFunction::schema_ref_t schema_ref) {
  if (TORCH_SDT_IS_ENABLED(operator_start)) {
    TORCH_SDT_WITH_SEMAPHORE(operator_start, schema_ref.get().name().c_str());
  }
}

// 触发操作结束的 USDT 事件
C10_NOINLINE void Dispatcher::fireOpEndUSDT(at::RecordFunction::schema_ref_t schema_ref) {
  if (TORCH_SDT_IS_ENABLED(operator_end)) {
    TORCH_SDT_WITH_SEMAPHORE(operator_end, schema_ref.get().name().c_str());
  }
}
#endif
```