# `.\pytorch\aten\src\ATen\core\dispatch\OperatorEntry.h`

```py
// 预处理指令：只包含一次的头文件，确保头文件只被包含一次
#pragma once

// 包含C++标准库头文件
#include <list>
#include <array>

// 包含ATen库的头文件
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/dispatch/CppSignature.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <ATen/core/enum_tag.h>

// 包含c10库的头文件
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/Optional.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/PyHandleCache.h>
#include <c10/core/SafePyObject.h>

// 在移动平台上定义宏，指示每个调度键只有一个内核
#ifdef C10_MOBILE
#define C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
#endif

// c10命名空间
namespace c10 {

// 前置声明，Dispatcher类在此命名空间中声明
class Dispatcher;

// c10::impl命名空间，包含内部实现细节
namespace impl {

// 表示从用户注册到我们的内核的数据结构。
// AnnotatedKernel相比KernelFunction包含了一些额外的元数据，用于提供良好的错误消息。
struct AnnotatedKernel final {
  AnnotatedKernel(KernelFunction k, std::unique_ptr<FunctionSchema> s, std::string d)
    : kernel(std::move(k))
    , inferred_function_schema(std::move(s))
    , debug(std::move(d))
    {}
  AnnotatedKernel() = default;
  KernelFunction kernel;
  std::unique_ptr<FunctionSchema> inferred_function_schema;
  // 一个小的调试字符串，用于帮助我们识别相关的内核。
  // 最重要的是它记录了进行注册的TORCH_LIBRARY块。
  std::string debug;
};

// 表示运算符模式的数据结构，包含指定此模式注册位置的元数据
struct AnnotatedSchema final {
  AnnotatedSchema(FunctionSchema s, std::string d)
    : schema(std::move(s))
    , debug(std::move(d))
    {}
  FunctionSchema schema;
  std::string debug;
};

// 内部数据结构，记录特定运算符的信息。
// 它不是公共API的一部分；通常用户将与OperatorHandle交互而不是OperatorEntry。
//
// 对OperatorEntry的并发写入由全局的Dispatcher锁保护
// （这很重要，因为OperatorEntry中的某些方法访问调度程序状态）
class TORCH_API OperatorEntry final {
public:
  explicit OperatorEntry(OperatorName&& operator_name);

  OperatorEntry(const OperatorEntry&) = delete;
  OperatorEntry(OperatorEntry&&) noexcept = delete;
  OperatorEntry& operator=(const OperatorEntry&) = delete;
  OperatorEntry& operator=(OperatorEntry&&) noexcept = delete;

  // 返回运算符的函数模式
  const FunctionSchema& schema() const {
    TORCH_INTERNAL_ASSERT(schema_.has_value(), "Tried to access the schema for ", name_, " which doesn't have a schema registered yet");
    return schema_->schema;
  }

  // 返回调试信息
  const std::string& debug() const {
    TORCH_INTERNAL_ASSERT(schema_.has_value());
    return schema_->debug;
  }

  // 判断是否存在函数模式
  bool hasSchema() const {
    // 返回当前 schema_ 是否有值的布尔结果
    return schema_.has_value();
    }
    
    bool isObserved() const {
    // 返回 is_observed_ 成员变量的布尔值
    return is_observed_;
    }
    
    // 对于某些操作符，即使没有 schema，也可能会分配 OperatorEntry。
    // 当接收到 schema 注册时，我们事后会注册一个 schema。
    //
    // 注意：registerSchema 和 deregisterSchema 不是幂等的操作；
    // 如果尝试注册已存在或不存在的 schema，会导致错误。
    // （注册计数在 Dispatcher 中的 OperatorHandle 中处理）
    void registerSchema(FunctionSchema&&, std::string&& debug, std::vector<at::Tag> tags = {});
    void deregisterSchema();
    
    const OperatorName& operator_name() const {
    // 返回 name_ 成员变量的引用，即操作符的名称
    return name_;
    }
#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  // 使用数组来容纳注释的内核，每个调度键只有一个内核
  using AnnotatedKernelContainer = std::array<AnnotatedKernel, 1>;
#else
  // 使用列表来容纳注释的内核，允许每个调度键有多个内核
  using AnnotatedKernelContainer = std::list<AnnotatedKernel>;
#endif

// 定义迭代器类型，用于遍历注释的内核容器
using AnnotatedKernelContainerIterator = AnnotatedKernelContainer::iterator;

// Why are kernels and fallback asymmetric?  It has to do with ownership.
// Kernels and the computed dispatch tables for them are canonically
// owned by OperatorEntry, but backend fallbacks are specified once
// and apply for all operators, so they should be owned by Dispatcher.
// However, the registration of a backend fallback affects the
// state of the computed dispatch table, so when a backend fallback
// is updated, we need to update the operator tables too.  Thus,
// registerKernel is the mechanism by which we give kernels to
// operator entry to own (and update dispatch table), but we only
// need a non-owning mechanism to update fallback.
// 内核和后备的不对称性是由所有权决定的。内核及其计算得到的调度表一般由OperatorEntry拥有，
// 但后备只需一次指定，并适用于所有运算符，因此应由Dispatcher拥有。然而，后备的注册会影响计算得到的调度表的状态，
// 因此当后备更新时，我们也需要更新运算符表。因此，registerKernel 是我们将内核提供给运算符条目进行所有权（并更新调度表）的机制，
// 但我们只需要一个非所有权机制来更新后备。

// Precondition: Dispatcher::mutex_ is held
// Postcondition: caller is responsible for disposing of the kernel
// 前提条件：必须持有 Dispatcher::mutex_
// 后置条件：调用者负责释放内核资源
AnnotatedKernelContainerIterator registerKernel(
  const Dispatcher& dispatcher,
  std::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
);

// Precondition: Dispatcher::mutex_ is held
// 前提条件：必须持有 Dispatcher::mutex_
void deregisterKernel_(
  const Dispatcher& dispatcher,
  std::optional<DispatchKey> dispatch_key,
  AnnotatedKernelContainerIterator kernel
);

// Precondition: Dispatcher::mutex_ is held
// 前提条件：必须持有 Dispatcher::mutex_
void updateFallback(
  const Dispatcher& dispatcher,
  DispatchKey dispatch_key
);

// Precondition: Dispatcher::mutex_ is held
// 前提条件：必须持有 Dispatcher::mutex_
void updateSchemaAliasAnalysis(AliasAnalysisKind a) {
  // 断言 schema_ 必须有值
  TORCH_INTERNAL_ASSERT(schema_.has_value());
  // 更新 schema 的别名分析策略
  schema_->schema.setAliasAnalysis(a);
}

// 返回计算得到的调度表的字符串表示
std::string dumpComputedTable() const;

// 返回状态的字符串表示
std::string dumpState() const;

// 检查不变量
void checkInvariants() const;

// 返回调度键提取器的常量引用
const DispatchKeyExtractor& dispatchKeyExtractor() const { return dispatchKeyExtractor_; }

// Asserts that the given FuncType is correct for calling this operator in an unboxed way.
// 断言给定的 FuncType 在无盒调用此运算符时是正确的。
template<class FuncType>
inline void assertSignatureIsCorrect() {
  assertSignatureIsCorrect(CppSignature::make<FuncType>(), fn_has_symint<FuncType>::value);
}

// 检查调用签名是否正确
void assertSignatureIsCorrect(const CppSignature& call_signature, bool has_symint) const;

// 报告错误，指定调度键
[[noreturn]] void reportError(DispatchKey dispatchKey) const;

// 查找与给定调度键集合对应的内核函数
const KernelFunction& lookup(DispatchKeySet ks) const {
  const auto idx = ks.getDispatchTableIndexForDispatchKeySet();
  if (C10_UNLIKELY(idx == -1)) {
    // 如果索引为 -1，则报告错误，使用最高优先级的类型 ID
    reportError(ks.highestPriorityTypeId());
  }
  // 返回对应索引的内核函数
  const auto& kernel = dispatchTable_[idx];
  // 一个有效的内核总是有盒子内核，也可能有无盒子内核。然而，我们通常在 at:: API 中进行无盒调用，
  // 其中内核 1) 很可能是有效的，并且 2) ...
  // （这里的注释未完全提供）
    // 检查 kernel 对象是否是有效的未装箱内核。
    // 在常见情况下，首先检查未装箱内核可以避免完全访问装箱内核。
    if (C10_UNLIKELY(!kernel.isValidUnboxed())) {
      // 如果未装箱内核无效，则检查装箱内核是否有效。
      if (!kernel.isValid()) {
        // 报告错误，使用类型标识符 ks 的最高优先级。
        reportError(ks.highestPriorityTypeId());
      }
    }
    // 返回 kernel 对象。
    return kernel;
  }

  // 返回包含所有分发键的列表的字符串表示。
  std::string listAllDispatchKeys() const;

  // 返回 kernel_ 是否具有 ks 中任何键的条目。
  //
  // 不变量: 传入的分发键集合中不包含别名键。
  // 注意 [分发键集合中没有别名键]
  // 别名键应使用 `hasKernelForDispatchKey` 进行检查。
  // 别名键不应该包含在 DispatchKeySet 中，因为它们可以有大于 63 的值（导致溢出）。
  bool hasKernelForAnyDispatchKey(DispatchKeySet ks) const;

  // 返回 kernel_ 是否具有特定键的条目。
  bool hasKernelForDispatchKey(DispatchKey k) const;

  // 检索特定键处的内核条目。与 hasKernelForDispatchKey 对称。
  // 要获取 AnnotatedKernel，请参阅 getKernelForDispatchKey（私有）。
  const KernelFunction& kernelForDispatchKey(DispatchKey k) const;

  // 返回 "计算表" 是否具有特定键的条目。
  bool hasComputedKernelForDispatchKey(DispatchKey k) const;

  // 返回在注册时添加的所有操作符标签。
  const std::vector<at::Tag>& getTags() const;

  // 设置报告错误回调函数。
  void setReportErrorCallback_(std::unique_ptr<c10::SafePyObject> callback);

  // 获取 Python 操作的 PyObject 指针。
  template <typename F>
  PyObject* getPythonOp(PyInterpreter* self_interpreter, F slow_accessor) const {
    return py_cache_.ptr_or(self_interpreter, slow_accessor);
  }
// 操作符名称
OperatorName name_;
// 可选的带注解的模式
std::optional<AnnotatedSchema> schema_;
#ifndef C10_MOBILE
  // 标签向量，仅在非移动平台编译时存在
  std::vector<at::Tag> tags_;
#endif
// 分发表，存储运行时条目数量大小的内核函数数组
std::array<KernelFunction, c10::num_runtime_entries> dispatchTable_;
// 分发键提取器
DispatchKeyExtractor dispatchKeyExtractor_;
// 用于快速访问 torch.ops.ns.op.overload 对象的指针
c10::PyHandleCache py_cache_;

// kernels_ 存储对应分发键的所有注册内核函数，catchAllKernels_ 存储全捕获内核函数
// 如果加载了覆盖已有内核函数的操作符库，这两种内核函数都会在列表中，但只有新的内核函数会在 dispatchTable 中
// 如果其中任何内核函数被移除（例如库被卸载），我们会从列表中删除该内核函数，并在必要时更新 dispatchTable
// 列表中的内核函数按注册时间降序排序，新注册的排在旧注册的前面
// 我们没有将 dispatchTable 和 kernels_ 合并成一个哈希映射，因为 kernels_ 是一个更大的数据结构，
// 访问频率较低，而 dispatchTable 经常被访问，应保持较小以适应 CPU 缓存
// 不变性：
// - dispatchTable[dispatch_key] == kernels_[dispatch_key].front()
// - 当且仅当 kernels_[dispatch_key] 不存在时，dispatchTable[dispatch_key] 也不存在
// - 如果 kernels_[dispatch_key] 存在，则其中有元素，永不为空列表
//
// 为什么这么做？
// -----
// 主要为了支持 Jupyter 笔记本，其中可能多次执行注册内核的单元格，后续执行应覆盖早期执行的内容。
// 注意，当函数模式在执行之间发生更改时，这种方法仍会失败，但只要函数模式未更改，它就能正常工作。
// 更好的解决方案是，在 Jupyter 单元格重新执行时卸载旧的扩展库，然后在此处只允许一个内核，即如果已经注册内核，则报错，
// 但这需要大量工作来实现，目前不是高优先级任务。
ska::flat_hash_map<DispatchKey,
#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
                   // 在移动平台上，不需要担心 Jupyter 笔记本
                   std::array<AnnotatedKernel, 1>
#else
                   std::list<AnnotatedKernel>
#endif
> kernels_;
// 结束当前的预处理指令块
#endif
// 存储内核的列表
> kernels_;

// 返回缺失内核的注解版本
const AnnotatedKernel& missingKernel() const;

// 返回模糊的自动微分其他内核的注解版本
const AnnotatedKernel& ambiguousAutogradOtherKernel() const;

// cpp_signature_ 存储函数签名（如果有的话），表示至少有一个内核是通过提供未打包的 C++ 内核函数创建的方式来知道函数签名的。
// 如果设置了此值，将用于检查未来的内核注册是否匹配，并且在未打包的函数调用中，用于根据已知的函数签名验证其参数。
struct CppSignatureWithDebug {
  CppSignature signature;
  std::string debug;
  std::optional<DispatchKey> dispatch_key;
};
// 可选的 C++ 函数签名与调试信息
std::optional<CppSignatureWithDebug> cpp_signature_;
// 可选的符号化 C++ 函数签名与调试信息
std::optional<CppSignatureWithDebug> sym_cpp_signature_;

// OperatorEntry::reportError 的 Python 自定义错误处理器
std::unique_ptr<c10::SafePyObject> report_error_callback_;

// 此运算符是否需要通过 RecordFunction 进行观察
const bool is_observed_;

// 报告签名错误的方法，使用调用签名和已保存签名的 C++ 签名版本
[[noreturn]] void reportSignatureError(const CppSignature& call_signature, const CppSignatureWithDebug& saved_signature) const;

// 计算并返回指定调度键的调度表条目的内核函数
const KernelFunction& computeDispatchTableEntry(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const;

// 计算并返回指定调度键的调度表条目的内核函数及其调试信息
std::pair<const AnnotatedKernel&, const char*> computeDispatchTableEntryWithDebug(
  const c10::Dispatcher& dispatcher, DispatchKey dispatch_key
) const;

// 此函数重新确保 dispatchTable 包含给定运行时调度键的 kernels 列表的第一个元素。
void updateDispatchTableEntry_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key);

// 与上述类似，但还处理别名调度键。
void updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key);

// 与上述类似，但用于调度表中的所有条目。
void updateDispatchTableFull_(const c10::Dispatcher& dispatcher);

// 获取给定调度键的 AnnotatedKernel 指针，该指针指向 kernels_ 中指定位置的元素。
const AnnotatedKernel* getKernelForDispatchKey(DispatchKey dispatch_key) const;
```