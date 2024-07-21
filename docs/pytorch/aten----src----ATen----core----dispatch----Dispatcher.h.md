# `.\pytorch\aten\src\ATen\core\dispatch\Dispatcher.h`

```
    // 指令，确保头文件仅被包含一次
#pragma once

    // 包含 ATen 库中的一些头文件
#include <ATen/SequenceNumber.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/boxing/impl/boxing.h>
#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/dispatch/CppSignature.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/LeftRight.h>
#include <list>
#include <mutex>
#include <condition_variable>
#include <type_traits>
#include <c10/core/SafePyObject.h>

    // 包含 ATen 库中的一些辅助头文件
#include <ATen/core/grad_mode.h>
#include <ATen/core/enum_tag.h>

    // 在调试模式下，包含标准输出流的头文件
#ifndef NDEBUG
#include <iostream>
#endif

    // 声明命名空间 c10
namespace c10 {

    // TORCH_API 宏标记的函数声明，用于 Torch 的 API
    TORCH_API bool show_dispatch_trace();
    TORCH_API void dispatch_trace_nesting_incr();
    TORCH_API void dispatch_trace_nesting_decr();
    TORCH_API int64_t dispatch_trace_nesting_value();

    // 定义一个用于管理调度追踪嵌套层级的类
    struct DispatchTraceNestingGuard {
        DispatchTraceNestingGuard() { dispatch_trace_nesting_incr(); }
        ~DispatchTraceNestingGuard() { dispatch_trace_nesting_decr(); }
    };

    // OperatorHandle 类的前置声明
    class TORCH_API OperatorHandle;
    template<class FuncType> class TypedOperatorHandle;

    /**
     * 操作符注册监听器接口，用于在操作符注册或注销时接收通知。
     */
    class TORCH_API OpRegistrationListener {
    public:
        virtual ~OpRegistrationListener();

        // 纯虚函数，当操作符注册时调用
        virtual void onOperatorRegistered(const OperatorHandle& op) = 0;
        
        // 纯虚函数，当操作符注销时调用
        virtual void onOperatorDeregistered(const OperatorHandle& op) = 0;
    };

    // detail 命名空间中 RegistrationListenerList 类的前置声明
    namespace detail {
        class RegistrationListenerList;
    }

    // SchemaRegistrationHandleRAII 类的前置声明
    class SchemaRegistrationHandleRAII;

    /**
     * 动态调度器的顶层调度接口。
     * 大多数终端用户不应直接使用此类；如果要注册操作符，请查看 op_registration。
     */
    class TORCH_API Dispatcher final {
    private:
        // 友元类，允许直接访问后端回退信息
        friend class impl::OperatorEntry;

        // OperatorDef 结构体的定义，表示一个操作符的定义
        struct OperatorDef final {
            explicit OperatorDef(OperatorName&& op_name)
            : op(std::move(op_name)) {}

            // 操作符的实际实现入口
            impl::OperatorEntry op;

            // 下面两个变量表示针对此操作符的 RegistrationHandleRAII 实例数目
            // def_count 表示仅有 def() 注册的数目，在新世界中应始终为 1，但旧的注册方式可能多次注册模式，导致此数增加
            // def_and_impl_count 表示 def() 和 impl() 注册的总数。当最后一个 def() 注销时，必须立即调用注销监听器，
            // 但不能真正删除操作符句柄，因为还有其他未处理的 RAII 析构器尝试析构，它们必须在这种情况下仍然有一个有效的操作符句柄
            size_t def_count = 0;
    // 定义一个名为 def_and_impl_count 的 size_t 类型变量并初始化为 0
    size_t def_and_impl_count = 0;
  };

  // 声明 OperatorHandle 类为友元类
  friend class OperatorHandle;

  // 声明 TypedOperatorHandle 模板类为友元类，模板参数为任意类型
  template<class> friend class TypedOperatorHandle;

  // 定义一个名为 Guard 的结构体
  struct Guard final {
    // Guard 结构体的构造函数，初始化 alive 为 true，mutex 为默认构造的互斥锁对象
    Guard() : alive(true), mutex() {}
    // 原子布尔类型的 alive 成员变量，用于表示 Guard 的存活状态
    std::atomic<bool> alive;
    // 互斥锁类型的 mutex 成员变量，用于在多线程环境中保护 Guard 对象的访问
    std::mutex mutex;
  };
public:
  // 析构函数
  ~Dispatcher();

  // Implementation note: this class abstracts over the fact that we have per-operator
  // dispatch tables.  This could be easily adjusted to have a single global hash
  // table.
  // 获取实际的单例 Dispatcher 对象
  static Dispatcher& realSingleton();

  // 返回单例 Dispatcher 对象，内联函数，用于避免函数调用开销
  C10_ALWAYS_INLINE static Dispatcher& singleton() {
#if !defined C10_MOBILE
    // 在此内联实现以避免函数调用开销。不能直接内联 realSingleton，
    // 因为函数内的静态变量会在包含和使用此头文件的所有 DSO 中复制，
    // 导致多个单例实例。
    static Dispatcher& s = realSingleton();
    return s;
#else
    // 对于 C10_MOBILE，不应内联具有静态成员的静态函数，
    // 因为生成的代码会调用 __cxa_guard_acquire 和 __cxa_guard_release，
    // 这些函数帮助实现静态变量 s 的初始化（非移动设备情况下）。
    // 当在每个后端的所有操作符存根中复制此额外代码时，会生成大量额外的代码。
    return realSingleton();
#endif // !defined C10_MOBILE
  }

private:
  // 构造函数
  Dispatcher();

  // 根据 DispatchKey 和 DispatchKeySet 返回运行记录函数的序列号
  static int64_t sequenceNumberForRunningRecordFunction(DispatchKey dispatchKey, DispatchKeySet dispatchKeySet);

  // 运行记录函数，记录函数的模式参考和 DispatchKey 及 DispatchKeySet
  static void runRecordFunction(at::RecordFunction& guard, at::RecordFunction::schema_ref_t schema_ref, DispatchKey dispatchKey, DispatchKeySet dispatchKeySet);

  // 运行记录函数，记录函数的模式参考、DispatchKey、DispatchKeySet 和参数列表
  static void runRecordFunction(at::RecordFunction& guard, at::RecordFunction::schema_ref_t schema_ref, DispatchKey dispatchKey, DispatchKeySet dispatchKeySet, c10::ArrayRef<const c10::IValue> args);

#ifdef FBCODE_CAFFE2
  // 判断操作符事件是否正在进行中
  static bool profilingOperatorEvents();

  // 发出操作开始的 USDT 事件，记录函数的模式参考
  static void fireOpStartUSDT(at::RecordFunction::schema_ref_t schema_ref);

  // 发出操作结束的 USDT 事件，记录函数的模式参考
  static void fireOpEndUSDT(at::RecordFunction::schema_ref_t schema_ref);
#endif // FBCODE_CAFFE2

  // 查找或注册函数模式的操作符句柄
  OperatorHandle findOrRegisterSchema_(FunctionSchema&& schema);

  // 查找或注册操作符名称的操作符句柄
  OperatorHandle findOrRegisterName_(const OperatorName& op_name);

  // 注销操作符定义
  void deregisterDef_(const OperatorHandle& op, const OperatorName& op_name);

  // 注销实现
  void deregisterImpl_(
    const OperatorHandle& op,
    const OperatorName& op_name,
    std::optional<DispatchKey> dispatch_key,
    impl::OperatorEntry::AnnotatedKernelContainerIterator kernel_handle);

  // 注销名称
  void deregisterName_(const OperatorHandle& op, const OperatorName& op_name);

  // 注销回退
  void deregisterFallback_(DispatchKey dispatchKey);

  // 注销库
  void deregisterLibrary_(const std::string& ns);

  // 清理操作符及其名称
  void cleanup(const OperatorHandle& op, const OperatorName& op_name);

  // 检查模式兼容性
  void checkSchemaCompatibility(const OperatorHandle& op, const FunctionSchema& schema, const std::string& debug);

  // 操作符定义列表
  std::list<OperatorDef> operators_;

#if !defined(C10_MOBILE)
  // 操作符查找表
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
#endif // !defined(C10_MOBILE)
#else
  RWSafeLeftRightWrapper<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
#endif
// operatorLookupTable_ 变量声明，条件编译指令 #else 后定义的类型为 RWSafeLeftRightWrapper，
// 用于安全读写操作的包装器，内部存储了一个从 OperatorName 到 OperatorHandle 的 flat_hash_map

// Map from namespace to debug string (saying, e.g., where the library was defined)
ska::flat_hash_map<std::string, std::string> libraries_;
// libraries_ 变量声明，用于存储从命名空间到调试字符串的映射，描述了库的定义位置等信息

std::array<impl::AnnotatedKernel, num_runtime_entries> backendFallbackKernels_;
// backendFallbackKernels_ 变量声明，定义了一个固定大小的数组，存储了 AnnotatedKernel 结构体的实例，
// 数组长度为 num_runtime_entries，用于存储后端回退的内核信息

std::unique_ptr<detail::RegistrationListenerList> listeners_;
// listeners_ 变量声明，是一个独占指针，指向 RegistrationListenerList 的实例，
// 用于管理注册监听器列表的生命周期和访问

// This condition variable gets notified whenever we add a new def/impl to the
// dispatch table.  This is primarily used by multipy/torchdeploy, when
// we have multiple interpreters trying to register to the dispatch table.
// In this situation, whenever the non-primary interpreter would have tried
// to register to the dispatch table, instead it will check to see if the
// expected registration has already been made, and if it hasn't, wait on
// this condition variable to see if it was just racing with the primary
// interpreter.
//
// We expect it to be rare for there to be any waiters on this condition
// variable.  This is mostly just to help give better diagnostics if
// something goes horribly wrong
std::condition_variable cond_var_;
// cond_var_ 变量声明，条件变量，用于在向调度表中添加新定义/实现时通知等待的线程，
// 主要用于在多个解释器尝试注册到调度表时，确保注册的同步性和正确性

// Protect concurrent access to the dispatcher.  We store this in a
// `shared_ptr` as we return callbacks that call back into dispatcher methods,
// and we need to be able to handle and guard against the event when the
// `Dispatcher` has been destroyed before the callbacks fire.
std::shared_ptr<Guard> guard_;
// guard_ 变量声明，使用 shared_ptr 保护对调度程序的并发访问，
// 主要用于确保在返回调度程序方法的回调时，能正确处理和保护 Dispatcher 销毁前的情况
};
    // 调用 operatorDef_->op 对象的 op.hasComputedKernelForDispatchKey 方法，检查指定的调度键是否有计算内核
    return operatorDef_->op.hasComputedKernelForDispatchKey(k);
  }

  // 返回 operatorDef_->op 对象的 dumpComputedTable 方法的结果，获取计算表的字符串表示
  std::string dumpComputedTable() const {
    return operatorDef_->op.dumpComputedTable();
  }

  // 调用 operatorDef_->op 对象的 checkInvariants 方法，检查操作的不变性条件
  void checkInvariants() const {
    return operatorDef_->op.checkInvariants();
  }

  // 返回 operatorDef_->op 对象的 getTags 方法的结果，获取操作的标签数组引用
  c10::ArrayRef<at::Tag> getTags() const {
    return operatorDef_->op.getTags();
  }

  // 调用 operatorDef_->op 对象的 setReportErrorCallback_ 方法，设置报错回调函数
  void setReportErrorCallback_(std::unique_ptr<c10::SafePyObject> callback) {
    operatorDef_->op.setReportErrorCallback_(std::move(callback));
  }

  // 检查 operatorDef_->op 对象是否具有指定的标签 tag，通过遍历标签数组来确定
  bool hasTag(const at::Tag& tag) const {
    for(const auto& tag_: getTags()) {
      if (tag == tag_) {
        return true;
      }
    }
    return false;
  }

  // 返回 operatorDef_->op 对象的模板方法 typed() 的结果，获取类型化的操作句柄
  template<class FuncType>
  TypedOperatorHandle<FuncType> typed() const {
    // 注意事项: 此处的断言并不完全准确，可以在操作符注册 C++ 签名之前获取 typed() 操作句柄，
    // 并且检查会认为一切正常（此时可以传入类型不正确的内核）。对于核心库中的所有内容，
    // 这种情况不会发生，因为所有静态注册将在获取 typed() 句柄时完成。
    // 返回类型化的操作句柄 FuncType
#if !defined C10_MOBILE
    // 如果不是在移动设备上，则进行以下操作
    operatorDef_->op.assertSignatureIsCorrect<FuncType>();
    // 断言操作符的签名是否正确，针对 FuncType 类型的操作符
    if (fn_has_symint<FuncType>::value) {
      // 如果 FuncType 类型的操作符包含符号整数
      operatorDef_->op.assertSignatureIsCorrect<typename fn_remove_symint<FuncType>::type>();
      // 断言操作符的签名是否正确，针对移除了符号整数的 FuncType 类型的操作符
    }
#endif
    // 返回一个 TypedOperatorHandle 对象，该对象包含 operatorIterator_ 指向的操作符句柄
    return TypedOperatorHandle<FuncType>(operatorIterator_);
  }

  // 调用包装在堆栈上的操作符，通过 Dispatcher 单例来实现
  void callBoxed(Stack* stack) const {
    c10::Dispatcher::singleton().callBoxed(*this, stack);
  }

  // 重载函数，用于直接传递堆栈的引用来调用 callBoxed(Stack* stack) 函数
  void callBoxed(Stack& stack) const {
    callBoxed(&stack);
  }

  // 根据 dispatch key 调用包装在堆栈上的操作符
  void callBoxedForDispatchKey(DispatchKey dk, Stack& stack) const {
    c10::Dispatcher::singleton().callBoxedForDispatchKey(*this, dk, &stack);
  }

  // 根据 dispatch key 集合重新分发包装在堆栈上的操作符
  void redispatchBoxed(DispatchKeySet ks, Stack* stack) const {
    c10::Dispatcher::singleton().redispatchBoxed(*this, ks, stack);
  }

  // 获取 Python 操作符对象，使用指定的慢速访问器
  template <typename F>
  PyObject* getPythonOp(c10::impl::PyInterpreter* self_interpreter, F slow_accessor) const {
    return operatorDef_->op.getPythonOp(self_interpreter, slow_accessor);
  }

  // 判断两个 OperatorHandle 对象是否相等
  bool operator==(const OperatorHandle& other) const {
    return operatorDef_ == other.operatorDef_;
  }

  // 判断两个 OperatorHandle 对象是否不相等
  bool operator!=(const OperatorHandle& other) const {
    return operatorDef_ != other.operatorDef_;
  }

private:
  // 显式构造函数，初始化 OperatorHandle 对象
  explicit OperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
  : operatorDef_(&*operatorIterator), operatorIterator_(operatorIterator)  {}
  friend class Dispatcher;
  template<class> friend class TypedOperatorHandle;

  // 直接指向 OperatorDef 的指针，虽然已经有了迭代器，但是这样做可以在关键调度路径上节省一条指令
  // 因为迭代器实际上是指向 std::list 节点的指针，在 libstdc++ 实现中，元素在节点中的偏移量为 16 字节
  // 这是因为节点结构中先出现了 prev/next 指针，所以需要一条 add 指令将迭代器转换为 OperatorDef*
  Dispatcher::OperatorDef* operatorDef_;

  // 我们需要存储这个迭代器以便于使 Dispatcher::cleanup() 方法变得快速
  // 它会在程序终止时频繁运行（并且可能在库卸载时）
  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
};

/**
 * 这是一个对注册在调度器中的操作符模式的句柄。
 * 它持有与 OperatorHandle 相同的信息，但其模板参数是操作符参数类型，
 * 允许以无盒方式调用操作符。
 */
template<class FuncType>
class TypedOperatorHandle final {
  static_assert(guts::false_t<FuncType>(), "FuncType in OperatorHandle::typed<FuncType> was not a valid function type");
  // 编译期错误，如果 FuncType 不是有效的函数类型，则会触发静态断言
};
template<class Return, class... Args>
class TypedOperatorHandle<Return (Args...)> final : public OperatorHandle {
  // 模板特化，继承自 OperatorHandle，针对 Return (Args...) 类型的操作符
// TypedOperatorHandle 类的移动构造函数，默认不抛出异常
TypedOperatorHandle(TypedOperatorHandle&&) noexcept = default;

// TypedOperatorHandle 类的移动赋值运算符，默认不抛出异常
TypedOperatorHandle& operator=(TypedOperatorHandle&&) noexcept = default;

// TypedOperatorHandle 类的拷贝构造函数，默认生成
TypedOperatorHandle(const TypedOperatorHandle&) = default;

// TypedOperatorHandle 类的拷贝赋值运算符，默认生成
TypedOperatorHandle& operator=(const TypedOperatorHandle&) = default;

// 调用具有 Args 参数的成员函数 call，使用 Dispatcher 单例进行调度
// Args 不使用右值引用的原因详见 [Note: Argument forwarding in the dispatcher]
C10_ALWAYS_INLINE Return call(Args... args) const {
  return c10::Dispatcher::singleton().call<Return, Args...>(*this, std::forward<Args>(args)...);
}

// 调用具有 Args 参数的 redispatch 函数，使用 Dispatcher 单例进行重调度
// Args 不使用右值引用的原因详见 [Note: Argument forwarding in the dispatcher]
C10_ALWAYS_INLINE Return redispatch(DispatchKeySet currentDispatchKeySet, Args... args) const {
  return c10::Dispatcher::singleton().redispatch<Return, Args...>(*this, currentDispatchKeySet, std::forward<Args>(args)...);
}

// 显式构造函数，接受一个操作符定义列表迭代器作为参数，用于初始化 OperatorHandle 基类
explicit TypedOperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
: OperatorHandle(operatorIterator) {}
friend class OperatorHandle;
};

namespace detail {
// 用于捕获 Dispatcher 的非装箱内核调用的返回值，用于记录函数
template <typename ReturnType>
struct CaptureKernelCall {
  // 构造函数，调用 kernel 并捕获返回值到 output_
  template <typename F, typename... Args>
  CaptureKernelCall(
      const F& kernel,
      const TypedOperatorHandle<ReturnType(Args...)>& op,
      const DispatchKeySet& dispatchKeySet,
      Args&&... args)
      : output_{kernel.template call<ReturnType, Args...>(
            op,
            dispatchKeySet,
            std::forward<Args>(args)...)} {}

  // 获取输出值并封装到 Stack 中
  Stack getOutputs() {
    Stack stack;
    impl::push_outputs<ReturnType, false>::copy(output_, &stack);
    return stack;
  }

  // 移动语义返回 output_，避免不必要的拷贝
  ReturnType release() && {
    return std::move(output_);
  }

 private:
  ReturnType output_;
};

// 特化模板，处理 kernel 返回类型为 at::Tensor& 的情况，返回其引用
template <>
inline at::Tensor& CaptureKernelCall<at::Tensor&>::release() && {
  return output_;
}

// 特化模板，处理 kernel 返回类型为 void 的情况
template <>
struct CaptureKernelCall<void> {
  // 构造函数，处理 kernel 返回类型为 void 的情况
  template <typename F, typename... Args>
  CaptureKernelCall(
      const F& kernel,
      const TypedOperatorHandle<void(Args...)>& op,
      const DispatchKeySet& dispatchKeySet,
      Args&&... args) {
    // 调用模板函数 call，用于调度操作 op，传入参数 dispatchKeySet 和 args
    kernel.template call<void, Args...>(
        op, dispatchKeySet, std::forward<Args>(args)...);
  }
  
  // 返回一个空的 Stack 对象
  Stack getOutputs() {
    return Stack();
  }
  
  // 移动语义的成员函数 release，用于移动构造和销毁对象
  void release() && {}
};

} // namespace detail

// See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
// 定义了一个内联函数，用于通过调度键慢速执行路径调用操作符
template<class Return, class... Args>
inline Return Dispatcher::callWithDispatchKeySlowPath(const TypedOperatorHandle<Return(Args...)>& op, at::StepCallbacks& stepCallbacks, DispatchKeySet dispatchKeySet, const KernelFunction& kernel, Args... args) {
  // 如果回调函数需要输入，我们将参数进行封装并传递给保护器
  at::RecordFunction guard(std::move(stepCallbacks));
  // 断言操作符已被观察，仅用于调试
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(op.operatorDef_->op.isObserved());
  // 获取最高优先级的调度键
  auto dispatchKey = dispatchKeySet.highestPriorityTypeId();
  // 获取操作符的函数模式
  auto& schema = op.schema();
  // 创建函数模式的常量引用
  auto schema_ref = std::reference_wrapper<const FunctionSchema>(schema);
  // 计算被封箱参数的数量
  constexpr auto num_boxed_args = impl::boxed_size<Args...>();
  // 如果参数数量不为零，执行以下内容
  if constexpr (num_boxed_args != 0) {
    // 如果保护器需要输入
    if (guard.needsInputs()) {
      // 使用对齐存储分配封箱参数的空间
      impl::IValueAlignedStorage boxedArgs[num_boxed_args];
      // 仅用于调试目的；在调试构建中，可以删除
      int lastArgIdx = 0;
      // 将参数封箱到栈中
      impl::boxArgsToStack(boxedArgs, lastArgIdx, args...);
      // 断言最后一个参数索引等于封箱参数数量
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(lastArgIdx == num_boxed_args);
      // 执行记录函数的运行，传递封箱参数的数组引用
      runRecordFunction(guard, schema_ref, dispatchKey, dispatchKeySet, c10::ArrayRef<const c10::IValue>(reinterpret_cast<IValue *>(boxedArgs), num_boxed_args));
      // 对封箱参数数组进行析构
      for (size_t ii = 0; ii < num_boxed_args; ++ii) {
        reinterpret_cast<IValue *>(&boxedArgs[ii])->~IValue();
      }
    } else {
      // 如果保护器不需要输入，执行记录函数的运行
      runRecordFunction(guard, schema_ref, dispatchKey, dispatchKeySet);
    }
  } else {
    // 如果没有封箱参数，执行记录函数的运行
    runRecordFunction(guard, schema_ref, dispatchKey, dispatchKeySet);
  }

  // 如果保护器需要输出
  if (C10_UNLIKELY(guard.needsOutputs())) {
    // 调用内核并临时捕获输出以传递给记录函数
    detail::CaptureKernelCall<Return> captureKernelCall(
        kernel, op, dispatchKeySet, std::forward<Args>(args)...);
    // 设置保护器的输出
    guard.setOutputs(captureKernelCall.getOutputs());
    // 释放捕获的输出以返回给调用方
    return std::move(captureKernelCall).release();
  }

  // 在执行内核时保持保护器活动
  return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
}

// See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
// 定义了一个内联函数模板，用于说明为什么 Args 不使用 &&
template<class Return, class... Args>
// 定义一个内联函数 `call`，返回类型为 `Return`，接受一个操作符句柄 `op` 和参数 `Args...`
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  detail::unused_arg_(args...);  // 用于解决在 gcc 5 中未使用参数的误报警告
  // 从操作符定义中提取调度键集合 `dispatchKeySet`
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  // 如果启用调度跟踪，则输出调用信息到标准错误流
  if (show_dispatch_trace()) {
      auto nesting_value = dispatch_trace_nesting_value();
      for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
      std::cerr << "[call] op=[" << op.operator_name() << "], key=[" << toString(dispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
  }
#endif
  // 根据调度键集合 `dispatchKeySet` 查找相应的内核函数 `kernel`
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  // 如果禁用单操作符性能分析，检查是否有步骤回调，并且操作符已被观察
  auto step_callbacks = at::getStepCallbacksUnlessEmpty(at::RecordScope::FUNCTION);
  if (C10_UNLIKELY(step_callbacks.has_value() && op.operatorDef_->op.isObserved())) {
    // 使用慢路径调用 `callWithDispatchKeySlowPath` 处理调用
    return callWithDispatchKeySlowPath<Return, Args...>(op, *step_callbacks, dispatchKeySet, kernel, std::forward<Args>(args)...);
  }
#endif  // PYTORCH_DISABLE_PER_OP_PROFILING

#ifdef FBCODE_CAFFE2
  // 如果使用了 FBCODE_CAFFE2，并且操作符事件性能分析已启用
  if(profilingOperatorEvents()) {
    // 使用 RAII 结构 `FireOpRAII` 来触发操作符开始和结束事件
    struct FireOpRAII {
       FireOpRAII(at::RecordFunction::schema_ref_t schema_ref) : schema_ref_(schema_ref) {
           fireOpStartUSDT(schema_ref);
        }
       ~FireOpRAII() { fireOpEndUSDT(schema_ref_); }
       at::RecordFunction::schema_ref_t schema_ref_;
    } event(op.schema());
    // 调用内核函数 `kernel` 处理操作，返回结果
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
  } else {
    // 否则，直接调用内核函数 `kernel` 处理操作，返回结果
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
  }
#else
    // 如果未定义 FBCODE_CAFFE2，则直接调用内核函数 `kernel` 处理操作，返回结果
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
#endif // FBCODE_CAFFE2
}

// 重新分发函数 `redispatch`，不使用 RecordFunction 进行记录
// 参考 [Note: Argument forwarding in the dispatcher]
template<class Return, class... Args>
inline Return Dispatcher::redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const {
  detail::unused_arg_(args...);  // 用于解决在 gcc 5 中未使用参数的误报警告
  // 不使用 RecordFunction 记录的调度跟踪
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  // 如果启用调度跟踪，则输出重新分发信息到标准错误流
  if (show_dispatch_trace()) {
      auto nesting_value = dispatch_trace_nesting_value();
      for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
      std::cerr << "[redispatch] op=[" << op.operator_name() << "], key=[" << toString(currentDispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
  }
#endif
  // 根据当前的调度键集合 `currentDispatchKeySet` 查找相应的内核函数 `kernel`，并调用处理操作，返回结果
  const KernelFunction& kernel = op.operatorDef_->op.lookup(currentDispatchKeySet);
  return kernel.template call<Return, Args...>(op, currentDispatchKeySet, std::forward<Args>(args)...);
}
// 调用一个操作符的包装版本，使用给定的操作符句柄和堆栈参数
inline void Dispatcher::callBoxed(const OperatorHandle& op, Stack* stack) const {
  // 注意：这里不需要互斥锁，因为对列表的写操作不会破坏迭代器。
  // 获取操作符句柄对应的操作符定义
  const auto& entry = op.operatorDef_->op;
  // 从操作符定义中提取调度关键字集合，以获取包装后的调度关键字
  auto dispatchKeySet = entry.dispatchKeyExtractor().getDispatchKeySetBoxed(stack);
#ifndef NDEBUG
  // 调度追踪调试守卫，用于调试目的
  DispatchTraceNestingGuard debug_guard;
  // 如果启用了调度追踪，则输出调度信息到标准错误流
  if (show_dispatch_trace()) {
      auto nesting_value = dispatch_trace_nesting_value();
      for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
      std::cerr << "[callBoxed] op=[" << op.operator_name() << "], key=[" << toString(dispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
  }
#endif
  // 根据调度关键字集合查找对应的内核函数
  const auto& kernel = entry.lookup(dispatchKeySet);
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  // 如果没有禁用每个操作的性能分析，并且操作已被观察，则执行性能分析
  auto step_callbacks = at::getStepCallbacksUnlessEmpty(at::RecordScope::FUNCTION);
  if (C10_UNLIKELY(step_callbacks.has_value() && entry.isObserved())) {
    // 创建记录函数守卫，以记录函数调用和参数
    at::RecordFunction guard(std::move(*step_callbacks));
    auto dispatchKey = dispatchKeySet.highestPriorityTypeId();
    auto& schema = op.schema();
    auto schema_ref = std::reference_wrapper<const FunctionSchema>(schema);
    // 如果需要输入，则传递输入参数；否则，只记录函数调用
    guard.needsInputs() ? runRecordFunction(guard, schema_ref, dispatchKey, dispatchKeySet, c10::ArrayRef<const c10::IValue>(stack->data(), stack->size()))
                        : runRecordFunction(guard, schema_ref, dispatchKey, dispatchKeySet);

    // 在执行内核函数时保持守卫有效
    kernel.callBoxed(op, dispatchKeySet, stack);

    // 如果需要输出，则设置输出到堆栈
    if (C10_UNLIKELY(guard.needsOutputs())) {
      guard.setOutputs(*stack);
    }
    return;
  }
#endif  // PYTORCH_DISABLE_PER_OP_PROFILING
  // 直接调用内核函数，传递操作符句柄、调度关键字集合和堆栈
  kernel.callBoxed(op, dispatchKeySet, stack);
}

// 注意：这不算是真正的调度器跳转，因此没有记录器
// 根据给定的调度关键字调用操作符的包装版本
inline void Dispatcher::callBoxedForDispatchKey(const OperatorHandle& op, DispatchKey dk, Stack* stack) const {
  // 注意：这里不需要互斥锁，因为对列表的写操作不会破坏迭代器。
  const auto& entry = op.operatorDef_->op;
  // 计算包装后的调度关键字集合，即使在这里我们需要传递给内部内核
  auto dispatchKeySet = entry.dispatchKeyExtractor().getDispatchKeySetBoxed(stack);
  // 根据调度关键字查找相应的内核函数
  const auto& kernel = ([&]() {
    if (op.hasKernelForDispatchKey(dk)) {
      return entry.kernelForDispatchKey(dk);
    } else {
      auto idx = getDispatchTableIndexForDispatchKey(dk);
      TORCH_INTERNAL_ASSERT(idx >= 0);
      return backendFallbackKernels_[idx].kernel;
    }
  })();
  // 调用内核函数，传递操作符句柄、调度关键字集合和堆栈
  kernel.callBoxed(op, dispatchKeySet, stack);
}

// 根据给定的调度关键字集合重新调度操作符的包装版本
inline void Dispatcher::redispatchBoxed(const OperatorHandle& op, DispatchKeySet dispatchKeySet, Stack* stack) const {
  // 注意：这里不需要互斥锁，因为对列表的写操作不会破坏迭代器。
  const auto& entry = op.operatorDef_->op;
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  // 如果处于调试模式下，创建 DispatchTraceNestingGuard 对象，用于跟踪调用深度
  if (show_dispatch_trace()) {
      // 如果需要显示调度跟踪信息
      auto nesting_value = dispatch_trace_nesting_value();
      // 获取当前调度跟踪深度
      for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
      // 根据深度在标准错误流中打印相应的缩进空格
      std::cerr << "[redispatchBoxed] op=[" << op.operator_name() << "], key=[" << toString(dispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
      // 打印调度信息，包括操作名和最高优先级类型的字符串表示
  }
#endif

const auto& kernel = entry.lookup(dispatchKeySet);
// 查找与给定 dispatchKeySet 相关的内核实例

return kernel.callBoxed(op, dispatchKeySet, stack);
// 调用找到的内核实例的 callBoxed 方法，传入操作 op、dispatchKeySet 和堆栈信息 stack
}

} // namespace c10

namespace std {

template <>
struct hash<c10::OperatorHandle> {
  size_t operator()(const c10::OperatorHandle& op) const noexcept {
    // 为 c10::OperatorHandle 类型定义哈希函数
    return std::hash<void*>{}(static_cast<void*>(op.operatorDef_));
    // 使用默认的指针哈希函数，哈希 OperatorHandle 的 operatorDef_ 指针
  }
};

} // namespace std
```