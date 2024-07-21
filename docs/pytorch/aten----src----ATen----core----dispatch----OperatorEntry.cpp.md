# `.\pytorch\aten\src\ATen\core\dispatch\OperatorEntry.cpp`

```py
// 包含 ATen 库中的头文件，用于操作符注册和调度
#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/dispatch/ObservedOperators.h>

// 在 c10 命名空间内定义实现细节
namespace c10 {
namespace impl {

// 匿名命名空间，用于实现一些辅助函数或局部变量
namespace {
  // 将 DispatchKey 类型的可选值转换为字符串表示
  std::string toString(std::optional<DispatchKey> k) {
    if (k.has_value()) {
      return toString(*k);
    } else {
      return "(catch all)";
    }
  }
}

// OperatorEntry 类的构造函数定义
OperatorEntry::OperatorEntry(OperatorName&& operator_name)
: name_(std::move(operator_name))  // 使用移动语义初始化 operator_name
, schema_()  // 初始化为空函数模式（schema）
#ifndef C10_MOBILE
, tags_()  // 移动语义初始化标签（tags），在非移动端编译时启用
#endif
, dispatchTable_()  // 初始化调度表为空
, dispatchKeyExtractor_(DispatchKeyExtractor::makeUninitialized())  // 初始化调度键提取器
, kernels_()  // 初始化内核列表为空
, cpp_signature_()  // 初始化 C++ 签名为空
, sym_cpp_signature_()  // 初始化符号化 C++ 签名为空
, is_observed_(ObservedOperators::isObserved(name_))  // 检查操作符是否被观察
{
  // 从全局调度器中获取所有的后端回退信息并更新调度表
  updateDispatchTableFull_(c10::Dispatcher::singleton());
}

// 匿名命名空间中的函数，用于检查操作符的函数模式
namespace {
  void checkSchema(const OperatorName& name, const FunctionSchema& from_def_, const std::string& from_def_debug, const KernelFunction& kernel, const FunctionSchema& inferred_, const std::string& inferred_debug) {
    // TODO: figure out if we can just directly save real schema at def time
    // 克隆传入的函数模式以保留真实的类型信息
    FunctionSchema from_def = from_def_.cloneWithRealTypes(kernel.isValidSymUnboxed());
    FunctionSchema inferred = inferred_.cloneWithRealTypes();
    // 查找两个函数模式的差异
    std::optional<std::string> schema_difference = findSchemaDifferences(from_def, inferred);
    // 如果存在差异，则抛出异常
    if (schema_difference.has_value()) {
      TORCH_CHECK(false,
        "Inferred operator schema for a C++ kernel function doesn't match the expected function schema.\n"
        "  operator: ", toString(name), "\n",
        "  expected schema: ", toString(from_def), "\n",
        "    ", from_def_debug, "\n",
        "  inferred schema: ", toString(inferred), "\n",
        "    ", inferred_debug, "\n",
        "  reason: ", *schema_difference);
    }
  }
} // anonymous namespace

// 返回缺失内核的静态函数
const AnnotatedKernel& OperatorEntry::missingKernel() const {
  static AnnotatedKernel kernel;
  return kernel;
}

// 返回模糊的 Autograd 其他内核的静态函数
const AnnotatedKernel& OperatorEntry::ambiguousAutogradOtherKernel() const {
  static AnnotatedKernel kernel(
    c10::KernelFunction::makeAmbiguousAutogradOther(), nullptr, "ambiguous_autogradother");
  return kernel;
}

// 断言调用签名的正确性
void OperatorEntry::assertSignatureIsCorrect(const CppSignature& call_signature, bool has_symint) const {
  if (has_symint) {
    // 如果有符号整数输入，检查符号化 C++ 签名是否匹配
    if (C10_UNLIKELY(sym_cpp_signature_.has_value() && (call_signature != sym_cpp_signature_->signature))) {
      reportSignatureError(call_signature, *sym_cpp_signature_);
    }
  } else {
    // 否则检查普通 C++ 签名是否匹配
    if (C10_UNLIKELY(cpp_signature_.has_value() && (call_signature != cpp_signature_->signature))) {
      reportSignatureError(call_signature, *cpp_signature_);
    }
  }
}

// 注册函数模式的方法，包括调试信息和标签
void OperatorEntry::registerSchema(FunctionSchema&& schema, std::string&& debug, std::vector<at::Tag> tags) {
  // 确保函数模式尚未注册
  TORCH_INTERNAL_ASSERT(!schema_.has_value());
  // 对每个内核遍历
  for (const auto& kernel : kernels_) {
    for (const auto &j : kernel.second) {
        // 遍历 kernel 结构体中的每个元素 j
        if (j.inferred_function_schema != nullptr) {
            // 如果 j 中的 inferred_function_schema 不为空指针，则执行下面的函数检查
            checkSchema(name_, schema, debug, j.kernel, *j.inferred_function_schema, j.debug);
        }
    }
  }
  // NB: don't register schema until after we've checked everything!
  // 注意：在检查完所有内容之后再注册 schema！
  dispatchKeyExtractor_.registerSchema(schema);
  // 将 schema 移动到 AnnotatedSchema 中，并且将 debug 也移动进去
  schema_ = AnnotatedSchema(std::move(schema), std::move(debug));
  // 如果不是在 C10_MOBILE 环境下，将 tags 移动进来
  #ifndef C10_MOBILE
    tags_ = std::move(tags);
  #endif
// 在注销模式时，确保 schema_ 已经被赋值，然后将其置为空值
void OperatorEntry::deregisterSchema() {
  TORCH_INTERNAL_ASSERT(schema_.has_value());  // 断言 schema_ 不为空
  schema_ = c10::nullopt;  // 将 schema_ 置为空值
  dispatchKeyExtractor_.deregisterSchema();  // 调用 dispatchKeyExtractor_ 的注销模式方法
}

// 注册一个新的内核函数，并返回一个迭代器用于访问 AnnotatedKernelContainer
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  std::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  // 如果 cpp_signature 有值
  if (cpp_signature.has_value()) {
    // 根据 kernel 的有效性选择要操作的 cpp_signature_ 或 sym_cpp_signature_
    auto& local_cpp_signature = kernel.isValidSymUnboxed() ? sym_cpp_signature_ : cpp_signature_;
    // 如果 local_cpp_signature 已经有值，则检查两个签名是否匹配
    if (local_cpp_signature.has_value()) {
      TORCH_CHECK(*cpp_signature == local_cpp_signature->signature,
        "\nMismatch in kernel C++ signatures\n",
        "  operator: ", (this->schema_.has_value() ? toString(this->schema_->schema) : toString(name_)), "\n",
        "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
        "  kernel 1: ", local_cpp_signature->signature.name(), "\n",
        "    dispatch key: ", toString(local_cpp_signature->dispatch_key), "\n",
        "    ", local_cpp_signature->debug, "\n",
        "  kernel 2: ", cpp_signature->name(), "\n",
        "    dispatch key: ", toString(dispatch_key), "\n",
        "    ", debug, "\n"
      );
    } else {
      // 否则，将 cpp_signature 值赋给 local_cpp_signature
      local_cpp_signature = CppSignatureWithDebug { *cpp_signature, debug, dispatch_key };
    }
  }

  // 如果 schema_ 和 inferred_function_schema 都有值，则检查注册的内核函数与函数模式的一致性
  if (schema_ && inferred_function_schema) {
    checkSchema(name_, schema_->schema, schema_->debug, kernel, *inferred_function_schema, debug);
  }

  // 将内核函数添加到 kernels_ 列表中，并根据 dispatch_key 的有无选择存储位置
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];

#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  // 如果启用了单一 dispatch key 对应一个内核函数的模式，则检查 k[0].kernel 是否有效
  if (k[0].kernel.isValid()) {
#else
  // 否则，检查 k 是否为空
  if (!k.empty()) {
#endif
    // 如果 k 不为空，则可能输出警告信息，特别处理 Meta key 以覆盖 C++ 元函数与 Python 元函数的情况
    // for some ops
    if (dispatch_key != DispatchKey::Meta) {
      // 如果 dispatch_key 不是 DispatchKey::Meta，则执行以下逻辑
      // 输出一次性警告，说明有其他操作符可能已被覆盖
      TORCH_WARN_ONCE("Warning only once for all operators,  other operators may also be overrided.\n",
            "  Overriding a previously registered kernel for the same operator and the same dispatch key\n",
            "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",
            "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
            "  dispatch key: ", toString(dispatch_key), "\n",
            "  previous kernel: ", (cpp_signature_.has_value() ? cpp_signature_->debug : (sym_cpp_signature_.has_value() ? sym_cpp_signature_->debug : "no debug info")), "\n",
            "       new kernel: ", debug
      );
    }
#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  // 如果定义了单一调度键对应一个内核的宏，则直接移动内核、推断函数架构和调试信息到 k[0]
  k[0].kernel = std::move(kernel);
  k[0].inferred_function_schema = std::move(inferred_function_schema);
  k[0].debug = std::move(debug);
#else
  // 否则，在容器的前端插入内核、推断函数架构和调试信息
  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
#endif
  // 插入的位置迭代器
  AnnotatedKernelContainerIterator inserted = k.begin();
  // 更新调度表，确保调度表指向最新的内核
  if (dispatch_key.has_value()) {
    // 如果有指定的调度键，则更新部分调度表
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    // 否则更新整个调度表
    updateDispatchTableFull_(dispatcher);
  }
  // 返回插入的位置迭代器
  return inserted;
}

void OperatorEntry::deregisterKernel_(
  const c10::Dispatcher& dispatcher,
  std::optional<DispatchKey> dispatch_key,
  AnnotatedKernelContainerIterator kernel
) {
  // 将捕捉所有的注销重定向到 CompositeImplicitAutograd
  DispatchKey dk = dispatch_key.has_value() ? *dispatch_key : DispatchKey::CompositeImplicitAutograd;
  // 查找指定调度键对应的内核列表
  auto found = kernels_.find(dk);
  // 如果找不到对应的调度键，抛出异常
  TORCH_INTERNAL_ASSERT(found != kernels_.end(), "Tried to deregister a kernel for dispatch key ", toString(dispatch_key), " but there are no kernels registered for this dispatch key. The operator is ", toString(name_));
  auto& k = found->second;
#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  // 如果定义了单一调度键对应一个内核的宏，则直接从 map 中删除数组
  // 此处无需执行其他操作
#else
  // 否则，从内核列表中删除指定的内核迭代器
  k.erase(kernel);
#endif
  // 如果删除后内核列表为空，则从 map 中删除该调度键对应的条目
  if (k.empty()) {
    kernels_.erase(found);
  }
  // 更新调度表
  updateDispatchTable_(dispatcher, dk);
}

void OperatorEntry::updateFallback(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  // 更新调度表的指定调度键条目
  updateDispatchTable_(dispatcher, dispatch_key);
}

const KernelFunction& OperatorEntry::computeDispatchTableEntry(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const {
  // 计算调度表中指定调度键的条目，返回内核函数
  return computeDispatchTableEntryWithDebug(dispatcher, dispatch_key).first.kernel;
}

bool OperatorEntry::hasKernelForAnyDispatchKey(DispatchKeySet ks) const {
  // 断言未定义调度键不存在于内核 map 中
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end());
  // 遍历所有的内核条目
  for (auto& kv : kernels_) {
    // 如果调度键不是别名且存在于给定的调度键集合中，则返回 true
    if (!isAliasDispatchKey(kv.first) && ks.has(kv.first)) return true;
  }
  // 否则返回 false
  return false;
}

bool OperatorEntry::hasKernelForDispatchKey(DispatchKey k) const {
  // 断言未定义调度键不存在于内核 map 中
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end());
  // 查找指定调度键对应的内核列表
  auto it = kernels_.find(k);
  // 如果找到，则返回该调度键对应的内核列表是否不为空
  if (it == kernels_.end()) return false;
  return !it->second.empty();
}

const KernelFunction& OperatorEntry::kernelForDispatchKey(DispatchKey k) const {
  // 查找指定调度键对应的内核列表
  auto it = kernels_.find(k);
  // 如果找不到对应的内核列表，抛出异常
  TORCH_CHECK(it != kernels_.end() && !it->second.empty(), "no kernel for ", k, " on ", name_);
  // 否则返回该调度键对应的第一个内核函数
  auto jt = it->second.begin();
  TORCH_INTERNAL_ASSERT(jt->kernel.isValid())
  return jt->kernel;
}
// 检查是否针对给定的调度键计算了运行时内核
bool OperatorEntry::hasComputedKernelForDispatchKey(DispatchKey k) const {
  // 检查给定的调度键是否是别名键，如果是，则抛出错误信息
  TORCH_CHECK(!isAliasDispatchKey(k), "Alias keys do not have runtime kernel registrations.");
  // 获取给定调度键对应的调度表索引
  const auto dispatch_ix = getDispatchTableIndexForDispatchKey(k);
  // 断言调度表索引在有效范围内
  TORCH_INTERNAL_ASSERT(dispatch_ix >= 0 && dispatch_ix < c10::num_runtime_entries, toString(k), dispatch_ix);
  // 返回调度表中对应索引的内核是否有效
  return dispatchTable_[dispatch_ix].isValid();
}

// 获取给定调度键对应的内核
const AnnotatedKernel* OperatorEntry::getKernelForDispatchKey(DispatchKey dispatch_key) const{
  // 查找给定调度键在内核映射中的位置
  auto kern_it = kernels_.find(dispatch_key);
  // 如果找到了对应的内核条目
  if (kern_it != kernels_.end()) {
    // 断言找到的内核列表不为空
    TORCH_INTERNAL_ASSERT(!kern_it->second.empty());
    // 断言找到的第一个内核条目的内核对象有效
    TORCH_INTERNAL_ASSERT(kern_it->second.front().kernel.isValid());
    // 返回找到的第一个内核条目的地址
    return &kern_it->second.front();
  }
  // 如果未找到对应的内核，返回空指针
  return nullptr;
}

// 获取操作符条目的标签
const std::vector<at::Tag>& OperatorEntry::getTags() const {
  // 如果定义了 C10_MOBILE 宏
  #if defined C10_MOBILE
    // 抛出错误，移动平台上不保存标签信息
    TORCH_CHECK(false, "tags are not saved for Mobile");
  #else
    // 返回保存的标签向量引用
    return tags_;
  #endif
}
  // 检查是否有针对 DispatchKey::CompositeExplicitAutograd 的内核
  hasKernelForDispatchKey(DispatchKey::CompositeExplicitAutograd);

  // 2.3. 如果可用，使用 CompositeImplicitAutograd 内核。对于自动求导键，仅在没有直接注册到其对应后端键或 CompositeExplicitAutograd 时才使用 CompositeImplicitAutograd 内核。
  //      对于 AutogradOther，如果已注册到任何后端，则返回 ambiguousAutogradOtherKernel()。
  //      有关 Undefined 的特殊处理，请参见注释 [Undefined in dispatchTable_]。

  // 如果 dispatch key 包含在 CompositeImplicitAutogradNestedTensor 中，
  // 则将其注册到 nested-tensor 内核，而不是 regular-tensor CompositeImplicitAutograd 内核。
  // 我们不打算更改 Undefined 的行为，
  // 因此此 nested-tensor 分支需要 `dispatch_key != DispatchKey::Undefined`
  // 以便让原始的 CompositeImplicitAutograd 处理 Undefined。
  // 请参见注释: [Disjoint AliasKeyset] 此别名键的顺序不重要
  if (dispatch_key != DispatchKey::Undefined && isIncludedInAlias(dispatch_key, DispatchKey::CompositeImplicitAutogradNestedTensor)) {
    if (auto nested_registration = getKernelForDispatchKey(DispatchKey::CompositeImplicitAutogradNestedTensor)) {
      return {*nested_registration, "nested kernel"};
    }
  }

  // 如果 dispatch key 是 Undefined 或者包含在 CompositeImplicitAutograd 中，
  // 则执行以下操作
  if (dispatch_key == DispatchKey::Undefined || isIncludedInAlias(dispatch_key, DispatchKey::CompositeImplicitAutograd)) {
    // 获取 CompositeImplicitAutograd 的内核注册
    if (auto math_registration = getKernelForDispatchKey(DispatchKey::CompositeImplicitAutograd)) {
      // 如果 dispatch key 是 AutogradOther，并且有任何后端的内核注册，
      // 则返回 ambiguousAutogradOtherKernel()
      if (dispatch_key == DispatchKey::AutogradOther
          && hasKernelForAnyDispatchKey(c10::autogradother_backends)) {
        return {ambiguousAutogradOtherKernel(), "ambiguous autogradother"};
      } else if (!has_backend_kernel) {
        // 如果没有后端的内核注册，则返回 math_registration
        return {*math_registration, "math kernel"};
      }
    }
  }

  // 2.4. 对于自动求导后端键，如果可用，使用 DispatchKey::Autograd 的内核
  if (isIncludedInAlias(dispatch_key, DispatchKey::Autograd)) {
    if (auto autograd_registration = getKernelForDispatchKey(DispatchKey::Autograd)) {
      return {*autograd_registration, "autograd kernel"};
    }
  }

  // 2.5. 对于批处理后端键，如果可用，使用 DispatchKey::FuncTorchBatchedDecomposition 的内核
  // 请参见注释: [Disjoint AliasKeyset] 此别名键的顺序不重要
  if (isIncludedInAlias(dispatch_key, DispatchKey::FuncTorchBatchedDecomposition)) {
    if (auto batched_registration = getKernelForDispatchKey(DispatchKey::FuncTorchBatchedDecomposition)) {
      return {*batched_registration, "batched kernel"};
    }
  }

  // 3. 后端回退
  // 获取 dispatch key 的调度表索引
  auto dispatch_ix = getDispatchTableIndexForDispatchKey(dispatch_key);
  // 如果索引小于 0，表示未注册后端回退
  if (dispatch_ix < 0) {
    return {missingKernel(), "backend fallback not registered on mobile"};
  }
  // 如果指定的后端回退内核有效，则返回其内核
  if (dispatcher.backendFallbackKernels_[dispatch_ix].kernel.isValid()) {
    return {dispatcher.backendFallbackKernels_[dispatch_ix], "backend fallback"};
  }

  // 4. Default to error
  // 如果以上条件都不满足，则默认返回一个包含错误内核和字符串"missing"的元组
  return {missingKernel(), "missing"};
// 同步给定调度键的调度表条目，与调度程序中的内核注册状态当前同步。
// 注意，这不是完整的更新，因为调度键之间存在关系（例如，运行时键及其关联的自动求导键，或别名键及其关联的键集）。
// 此函数应被视为 updateDispatchTable_ 的私有辅助函数。
void OperatorEntry::updateDispatchTableEntry_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  // 获取给定调度键在调度表中的索引
  const auto dispatch_ix = getDispatchTableIndexForDispatchKey(dispatch_key);
  // 如果索引为 -1，表示未找到对应的调度表条目，直接返回
  if (C10_UNLIKELY(dispatch_ix == -1)) {
    return;
  }
  // 计算并更新调度表中的条目
  dispatchTable_[dispatch_ix] = computeDispatchTableEntry(dispatcher, dispatch_key);
  // 设置操作符在给定调度键上是否有 fallthrough 特性
  dispatchKeyExtractor_.setOperatorHasFallthroughForKey(dispatch_key, dispatchTable_[dispatch_ix].isFallthrough());
}

// 同步给定调度键及其关联键集的调度表条目，与调度程序中的内核注册状态当前同步。
// 当一个内核已注册到一个调度键后，调用此函数将同步调度程序状态。
// 参见例如 registerKernel()
void OperatorEntry::updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  // 处理 Undefined 调度键的情况，因为它不是运行时键，但在 dispatchTable_ 中有一个条目。
  // 参见 Note [Undefined in dispatchTable_]
  if (dispatch_key == DispatchKey::Undefined) {
    updateDispatchTableEntry_(dispatcher, dispatch_key);
    return;
  }
  // 对于给定调度键的所有运行时调度键，逐个更新调度表条目
  for (auto k : c10::getRuntimeDispatchKeySet(dispatch_key)) {
    updateDispatchTableEntry_(dispatcher, k);
  }
  // 将注册到 CompositeImplicitAutograd、CompositeExplicitAutograd 和 CompositeExplicitAutogradNonFunctional 的调度表条目填充到 Undefined。
  // 不能在上面执行此操作，因为 Undefined 无法在 DispatchKeySet 中表示。
  if (dispatch_key == DispatchKey::CompositeImplicitAutograd
   || dispatch_key == DispatchKey::CompositeExplicitAutograd
   || dispatch_key == DispatchKey::CompositeExplicitAutogradNonFunctional) {
    updateDispatchTableEntry_(dispatcher, DispatchKey::Undefined);
  }
  // 注册到后端键可能会影响其自动求导后端键上的计算条目，参见 Note [Refresh Runtime Autograd entries in dispatchTable_]
  // 理论上，我们只需检查给定的运行时键是否具有 "dense" 功能，例如 DispatchKey::CPU（由 DispatchKey::Dense 和 BackendComponent::CPUBit 组成）。
  // 但是，有些后端应该包含在此集合中，但它们没有密集键集。
  // 例如 DispatchKey::Meta、DispatchKey::MAIA。
  if (c10::isBackendDispatchKey(dispatch_key)) {
    // 获取后端键对应的自动求导键
    DispatchKey autograd_key = getAutogradKeyFromBackend(toBackendComponent(dispatch_key));
    // 更新对应自动求导键的调度表条目
    updateDispatchTableEntry_(dispatcher, autograd_key);
  }
}
// 在运行时，根据当前内核注册状态更新调度器中的分发键。
// 注意，我们使用 updateDispatchTable_() 函数执行按键更新，
// 即使该函数可以处理无序更新和别名键更新，我们目前并未发送这些内容给它。这是有意为之 - 
// 当前设计更适合通过单一的按键更新机制处理所有更新，而不是通过多个假设不同不变量的变化。
void OperatorEntry::updateDispatchTableFull_(const c10::Dispatcher& dispatcher) {
  // 注意 [Undefined in dispatchTable_]
  // DispatchKey Undefined 在运行时中使用：
  // (1) 它提供了一个地方来指定当没有分发键时应运行的功能，例如没有张量输入或空的张量列表参数的操作。
  // (2) 它可以让我们去掉分发热路径中的显式错误检查代码，因此当没有分发键可用时，我们会滑入未定义处理程序，然后会引发错误消息。
  // 在旧的 catchAll 的世界中，将内核注册到 Undefined 的唯一方法是将其注册到 catchAll。删除 catchAllKernel_ 后，Undefined 现在可以从 CompositeExplicitAutograd 或 CompositeImplicitAutograd 别名键中获取内核，以便我们不破坏支持。理想情况下，isIncludedInAlias(Undefined, CompositeImplicitAutograd) 应返回 true，但它返回 false 是因为 Undefined 无法表示为 DispatchKeySet。
  updateDispatchTable_(dispatcher, DispatchKey::Undefined);
  for (auto k : DispatchKeySet(DispatchKeySet::FULL)) {
    updateDispatchTable_(dispatcher, k);
  }
}

// 检查不变量是否满足
void OperatorEntry::checkInvariants() const {
  if (schema_) {
    TORCH_INTERNAL_ASSERT(schema_->schema.operator_name() == name_, dumpState());
    dispatchKeyExtractor().checkInvariants(schema_->schema);
  }
  // 检查 Undefined 分发键是否未在 kernels_ 中出现
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end(), dumpState());
  // 检查每个分发键是否至少有一个内核注册
  for (const auto& kv : kernels_) {
    TORCH_INTERNAL_ASSERT(!kv.second.empty(), dumpState());
  }
  // 检查 DispatchKeySet(FULL) 中的每个分发键
  for (auto k : DispatchKeySet(DispatchKeySet::FULL)) {
    // 计算给定分发键的预期分发表项
    auto expected_k = computeDispatchTableEntry(c10::Dispatcher::singleton(), k);
    auto idx = getDispatchTableIndexForDispatchKey(k);
    if (C10_UNLIKELY(idx == -1)) {
      continue;
    }
    // 断言预期的分发表项与计算的分发表项相等
    TORCH_INTERNAL_ASSERT(expected_k._equalsBoxedAndUnboxed(dispatchTable_[idx]),
      "Canonical state\n~~~~~~~~~~~\n", dumpState(), "\n\n"
      "Computed table:\n~~~~~~~~~~~\n", dumpComputedTable());
  }
}

// 返回所有分发键的字符串表示
std::string OperatorEntry::listAllDispatchKeys() const {
  std::ostringstream str;
  str << "[";

  bool has_kernels = false;
  for (auto k : DispatchKeySet(DispatchKeySet::FULL)) {
    auto iter = getDispatchTableIndexForDispatchKey(k);
    if (iter == -1 || !dispatchTable_[iter].isValid()) {
      continue;
    }
    if (has_kernels) {
      str << ", ";
    }
    // 将分发键添加到输出字符串中
    str << k;
    // 检查是否有内核，这里有可能是一个变量，但是应该是小写的true，而不是首字母大写的True。
    has_kernels = true;
    // 向字符串流添加一个右方括号，以结束JSON数组的表示。
    str << "]";
    // 返回字符串流的内容作为最终的字符串表示。
    return str.str();
// 定义 OperatorEntry 类中的 reportSignatureError 方法，报告操作符签名错误
void OperatorEntry::reportSignatureError(const CppSignature& call_signature, const CppSignatureWithDebug& saved_signature) const {
  // 使用 TORCH_CHECK 来断言条件 false，如果条件为真，则报错并打印以下信息
  TORCH_CHECK(false,
        "\nTried to access or call an operator with a wrong signature.\n",  // 尝试使用错误签名访问或调用操作符
        "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",  // 操作符名称或者模式的字符串表示
        "    ", (schema_.has_value() ? schema_->debug : "unknown debug info"), "\n",  // 操作符的调试信息或者未知的调试信息
        "  correct signature:  ", saved_signature.signature.name(), "\n",  // 正确的操作符签名名称
        "    ", saved_signature.debug, "\n",  // 正确签名的调试信息
        "  accessed/called as: ", call_signature.name(), "\n",  // 实际访问或调用的操作符签名名称
        "This likely happened in a call to OperatorHandle::typed<Return (Args...)>(). ",  // 这可能发生在调用 OperatorHandle::typed<Return (Args...)>() 中
        "Please make sure that the function signature matches the signature in the operator registration call."  // 请确保函数签名与操作符注册调用中的签名匹配
  );
};

// 静态函数，用于处理分发键字符串的后处理
static std::string post_process_dispatch_key_str(std::string dispatch_key) {
  const std::string substr = "PrivateUse1";
  // 如果 dispatch_key 以 "PrivateUse1" 结尾，则进行以下处理
  if (substr.size() <= dispatch_key.size() && std::equal(substr.rbegin(), substr.rend(), dispatch_key.rbegin())) {
    // 获取 PrivateUse1 的后端名称
    auto privateuse1_backend = get_privateuse1_backend();
    // 如果后端名称不是 "privateuseone"
    if (privateuse1_backend != "privateuseone") {
      // 移除末尾的 "*PrivateUse1"
      dispatch_key.erase(dispatch_key.length() - substr.length());
      // 添加注册的后端名称到 dispatch_key
      // AutogradPrivateUse1 -> AutogradFoo
      auto backend_name = c10::get_privateuse1_backend();
      dispatch_key = dispatch_key + backend_name;
    }
  }
  // 返回处理后的 dispatch_key
  return dispatch_key;
}

// OperatorEntry 类中的 reportError 方法，报告分发键 dispatchKey 的错误
void OperatorEntry::reportError(DispatchKey dispatchKey) const {
  // 检查不变量是否成立
  checkInvariants();

  // 如果 report_error_callback_ 不为空指针
  if (report_error_callback_ != nullptr) {
    // 调用 report_error_callback_ 的 pyinterpreter()->reportErrorCallback 方法，并传递 dispatchKey
    report_error_callback_->pyinterpreter()->reportErrorCallback(report_error_callback_->ptr(&report_error_callback_->pyinterpreter()), dispatchKey);
    // reportErrorCallback 应该已经引发了一个错误，否则断言失败
    TORCH_INTERNAL_ASSERT(false);
  }

  // 如果 dispatchKey 是 DispatchKey::Undefined
  if (dispatchKey == DispatchKey::Undefined) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
        "There were no tensor arguments to this function (e.g., you passed an "
        "empty list of Tensors), but no fallback function is registered for schema ", name_,
        ".  This usually means that this function requires a non-empty list of Tensors, "
        "or that you (the operator writer) forgot to register a fallback function.  "
        "Available functions are ", listAllDispatchKeys(), ".\n\n", dumpComputedTable())
  }


# 检查未实现错误，如果条件为 false，则输出错误消息，说明缺少张量参数
TORCH_CHECK_NOT_IMPLEMENTED(false,
      "There were no tensor arguments to this function (e.g., you passed an "
      "empty list of Tensors), but no fallback function is registered for schema ", name_,
      ".  This usually means that this function requires a non-empty list of Tensors, "
      "or that you (the operator writer) forgot to register a fallback function.  "
      "Available functions are ", listAllDispatchKeys(), ".\n\n", dumpComputedTable())
}




  TORCH_CHECK_NOT_IMPLEMENTED(false, "Could not run '", name_, "' with arguments",
          " from the '", post_process_dispatch_key_str(toString(dispatchKey)), "' backend. This could be because "
          "the operator doesn't exist for this backend, or was omitted during ",
          "the selective/custom build process (if using custom build). If you are a ",
          "Facebook employee using PyTorch on mobile, please visit ",
          "https://fburl.com/ptmfixes for possible resolutions. '",
          name_, "' is only available for these backends: ",
          listAllDispatchKeys(), ".\n\n", dumpComputedTable());


# 检查未实现错误，如果条件为 false，则输出错误消息，说明无法使用指定参数运行该函数
TORCH_CHECK_NOT_IMPLEMENTED(false, "Could not run '", name_, "' with arguments",
        " from the '", post_process_dispatch_key_str(toString(dispatchKey)), "' backend. This could be because "
        "the operator doesn't exist for this backend, or was omitted during ",
        "the selective/custom build process (if using custom build). If you are a ",
        "Facebook employee using PyTorch on mobile, please visit ",
        "https://fburl.com/ptmfixes for possible resolutions. '",
        name_, "' is only available for these backends: ",
        listAllDispatchKeys(), ".\n\n", dumpComputedTable());
}

// INSPECTING DISPATCHER STATE
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The dumper functions purposely do not check invariants, as you might be using
// them to debug situations where the invariants are violated.

// Inspect what the computed dispatch table would be (e.g., what
// updateDispatchTableFull_ would update the dispatch table to be)
// 返回一个描述计算得到的调度表的字符串
std::string OperatorEntry::dumpComputedTable() const {
  std::ostringstream oss;
  // 需要单独处理 Undefined，因为它是一个运行时的键，不能被表示为 DispatchKeySet。
  std::vector<DispatchKey> runtime_keys = {DispatchKey::Undefined};
  // 遍历 DispatchKeySet(DispatchKeySet::FULL)，获取所有的调度键并加入到 runtime_keys 中
  for (auto k : DispatchKeySet(DispatchKeySet::FULL)) runtime_keys.push_back(k);

  // 遍历 runtime_keys，计算每个调度键对应的调度表项
  for (auto k : runtime_keys) {
    auto kernel_prov = computeDispatchTableEntryWithDebug(c10::Dispatcher::singleton(), k);
    // 如果内核有效，将内核信息写入 oss
    if (kernel_prov.first.kernel.isValid()) {
      oss << toString(k) << ": "
          << (kernel_prov.first.kernel.isFallthrough() ? "fallthrough " : "")
          << kernel_prov.first.debug << " [" << kernel_prov.second << "]\n";
    }
  }
  return oss.str(); // 返回 oss 中的字符串表示
}

void OperatorEntry::setReportErrorCallback_(std::unique_ptr<c10::SafePyObject> callback) {
  report_error_callback_ = std::move(callback);
}

// Inspect the "canonical" information in OperatorEntry.  This only prints out
// *non-derived* information including kernels registered to alias dispatch keys;
// i.e., what the source of truth says about the operator.  This dumping function
// is appropriate for expect tests.
// This WON'T report backend fallbacks.
// 返回一个描述 OperatorEntry 中“规范”信息的字符串。仅打印非派生信息，包括注册到别名调度键的内核。
// 适用于期望测试的转储函数。
std::string OperatorEntry::dumpState() const {
  std::ostringstream oss;
  oss << "name: " << name_ << "\n"; // 输出操作符名称

  // 如果有 schema，则输出相关信息；否则标记为 (none)
  if (schema_) {
    oss << "schema: " << schema_->schema << "\n";
    oss << "debug: " << schema_->debug << "\n";
    oss << "alias analysis kind: " << toString(schema_->schema.aliasAnalysis())
        << (schema_->schema.isDefaultAliasAnalysisKind() ? " (default)" : "") << "\n";
  } else {
    oss << "schema: (none)\n";
  }

  // 打印内核信息的 Lambda 函数
  auto print_kernel = [&](const char* k_desc, const AnnotatedKernelContainer& jts, bool is_alias_key=false) {
    int64_t i = 0;
    // 遍历内核容器 jts，输出内核调试信息
    for (const auto& jt : jts) {
      oss << k_desc
          << (is_alias_key ? "[alias]" :  "")
          << (i > 0 ? " (inactive)" : "")
          << ": "
          << jt.debug << " :: "
          << (jt.inferred_function_schema ? toString(*jt.inferred_function_schema) : "(none)")
          << " [ " << jt.kernel.dumpState() << "]\n";
      i++;
    }
  };

  // 遍历调度键，输出对应的内核信息
  for (uint8_t i = 0; i <= static_cast<uint8_t>(DispatchKey::EndOfAliasKeys); i++) {
    auto k = static_cast<DispatchKey>(i);
    auto it = kernels_.find(k);
    // 如果找到对应的内核容器，调用 print_kernel 输出信息
    if (it != kernels_.end()) {
      print_kernel(toString(k), it->second, c10::isAliasDispatchKey(k));
    }
  }
  return oss.str(); // 返回 oss 中的字符串表示
}

}
}
```