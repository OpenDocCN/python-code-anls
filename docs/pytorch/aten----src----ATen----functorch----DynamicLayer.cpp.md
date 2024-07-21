# `.\pytorch\aten\src\ATen\functorch\DynamicLayer.cpp`

```py
// functorch stores some TLS. Inside the TLS is the stack of transforms.
// Unfortunately, since functorch isn't a part of libtorch, we have
// a level of indirection. FuncTorchTLSBase is the interface that lives in libtorch,
// while FuncTorchTLS implements all the methods and stores data.
//
// TODO: after functorch C++ code is moved into PyTorch, we can get rid of
// this layer of indirection.
class FuncTorchTLS : public FuncTorchTLSBase {
 public:
  // 默认构造函数
  FuncTorchTLS() = default;

  // 实现接口，创建 FuncTorchTLSBase 的深拷贝
  std::unique_ptr<FuncTorchTLSBase> deepcopy() const override {
    // 创建 FuncTorchTLSBase 的新实例并返回
    auto result = std::make_unique<FuncTorchTLS>();
    return result;
  }
};
  }

  // 检查是否支持单层自动微分函数
  int64_t checkSupportsSingleLevelAutogradFunction() const override {
    // 如果 dynamicLayerStack 是空的或者允许使用单层自动微分函数，则通过；否则抛出错误
    TORCH_INTERNAL_ASSERT(dynamicLayerStack.empty() || getSingleLevelAutogradFunctionAllowed(),
        "functorch functions (vmap, grad, vjp, etc.) incorrectly used with ",
        "torch.autograd.function._SingleLevelFunction. ",
        "This is not expected, please file a bug.");
    return 0;
  }

  // 检查是否支持 C++ Autograd 函数
  void checkSupportsCppAutogradFunction() const override {
    // 如果 dynamicLayerStack 不为空，则抛出错误，禁止使用 C++ Autograd 函数与 functorch 转换一起使用
    TORCH_CHECK(
        dynamicLayerStack.empty(),
        "cannot use C++ torch::autograd::Function with functorch transforms (vmap, grad, vjp, etc)");
  }

  // 检查是否支持 inplace_requires_grad
  void checkSupportsInplaceRequiresGrad() const override {
    // 如果 dynamicLayerStack 是空的或者允许 inplace_requires_grad，则通过；否则抛出错误
    TORCH_CHECK(dynamicLayerStack.empty() || allow_inplace_requires_grad_,
        "You are attempting to call Tensor.requires_grad_() (or perhaps using ",
        "torch.autograd.functional.* APIs) inside of a function being transformed ",
        "by a functorch transform. ",
        "This is unsupported, please attempt to use the functorch transforms ",
        "(e.g. grad, vjp, jacrev, jacfwd, hessian) or call requires_grad_() "
        "outside of a function being transformed instead.");
  }

  // 检查是否支持 retain_grad
  void checkSupportsRetainGrad() const override {
    // 如果 dynamicLayerStack 不为空，则抛出错误，禁止在 functorch 转换中使用 retain_grad
    TORCH_CHECK(dynamicLayerStack.empty(),
        "You are attempting to call Tensor.retain_grad() ",
        "inside of a function being transformed ",
        "by a functorch transform. ",
        "This is unsupported, please attempt to use the functorch transforms ",
        "(e.g. grad, vjp, jacrev, jacfwd, hessian) or call retain_grad() "
        "outside of a function being transformed instead.");
  }

  // 动态图层堆栈
  std::vector<DynamicLayer> dynamicLayerStack;
  bool allow_inplace_requires_grad_ = false;  // 是否允许 inplace_requires_grad
  bool allow_single_level_autograd_function_ = false;  // 是否允许单层自动微分函数
};

// 获取 FuncTorchTLS 对象的原始指针
static FuncTorchTLS* getRawFunctorchTLS() {
  // 获取 functorchTLS 的状态引用
  auto& state = functorchTLSAccessor();
  // 如果状态为空指针，则创建一个新的 FuncTorchTLS 对象
  if (state == nullptr) {
    state = std::make_unique<FuncTorchTLS>();
  }
  // 使用原始指针是安全的，因为 state 保持了指针的有效性
  FuncTorchTLSBase* raw_state = state.get();
  // 将原始指针转换为 FuncTorchTLS 指针
  FuncTorchTLS* result = static_cast<FuncTorchTLS*>(raw_state);
  return result;
}

// 设置是否允许原地操作并需要梯度
void setInplaceRequiresGradAllowed(bool allowed) {
  auto* functorch_tls = getRawFunctorchTLS();
  functorch_tls->allow_inplace_requires_grad_ = allowed;
}

// 获取是否允许原地操作并需要梯度
bool getInplaceRequiresGradAllowed() {
  auto* functorch_tls = getRawFunctorchTLS();
  return functorch_tls->allow_inplace_requires_grad_;
}

// 设置是否允许单层自动求导函数
void setSingleLevelAutogradFunctionAllowed(bool allowed) {
  auto* functorch_tls = getRawFunctorchTLS();
  functorch_tls->allow_single_level_autograd_function_ = allowed;
}

// 获取是否允许单层自动求导函数
bool getSingleLevelAutogradFunctionAllowed() {
  auto* functorch_tls = getRawFunctorchTLS();
  return functorch_tls->allow_single_level_autograd_function_;
}

// 获取动态图层栈的引用
static std::vector<DynamicLayer>& dynamicLayerStackAccessor() {
  return getRawFunctorchTLS()->dynamicLayerStack;
}

// 获取给定级别的生命周期句柄
const std::shared_ptr<bool>& getLifeHandleForLevel(int64_t level) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  // 断言确保动态图层栈的大小符合要求
  TORCH_INTERNAL_ASSERT(
      (int64_t)dynamicLayerStack.size() >= level && level >= 1,
      "If you're trying to construct a tensor with the current level (",
      level,
      ") then the interpreter for that level must be on the DynamicLayerStack ");

  // 获取指定级别的动态图层对象
  auto& dynamic_layer = dynamicLayerStack[level - 1];
  // 返回动态图层对象的生命周期句柄
  return dynamic_layer.interpreter().is_alive_ptr();
}

// 可选函数：返回当前动态图层对象，如果栈为空则返回空
optional<DynamicLayer> maybeCurrentDynamicLayer() {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  if (dynamicLayerStack.empty()) {
    return {};
  }
  return dynamicLayerStack.back();
}

// 保存本地调度键集的构造和析构器
struct SaveLocalDispatchKeySet {
 public:
  // 构造函数：保存当前动态图层的本地调度键集
  SaveLocalDispatchKeySet() {
    auto& dynamicLayerStack = dynamicLayerStackAccessor();
    TORCH_INTERNAL_ASSERT(!dynamicLayerStack.empty());
    auto& layer = dynamicLayerStack.back();
    auto tmp = c10::impl::tls_local_dispatch_key_set();
    layer.interpreter().saveLocalDispatchKeySet(tmp);
  }
  // 析构函数：清除保存的本地调度键集
  ~SaveLocalDispatchKeySet() {
    auto& dynamicLayerStack = dynamicLayerStackAccessor();
    TORCH_INTERNAL_ASSERT(!dynamicLayerStack.empty());
    auto& layer = dynamicLayerStack.back();
    auto tmp = layer.interpreter().getSavedLocalDispatchKeySet();
    layer.interpreter().clearSavedLocalDispatchKeySet();
    c10::impl::_force_tls_local_dispatch_key_set(tmp);
  }
  // 禁用复制构造函数和赋值运算符
  SaveLocalDispatchKeySet(const SaveLocalDispatchKeySet&) = delete;
  SaveLocalDispatchKeySet& operator=(const SaveLocalDispatchKeySet&) = delete;
};

// 获取动态图层栈
const std::vector<DynamicLayer>& getDynamicLayerStack() {
  return dynamicLayerStackAccessor();
}

// 设置动态图层栈
void setDynamicLayerStack(const std::vector<DynamicLayer>& stack) {
  dynamicLayerStackAccessor() = stack;
}
// 从动态图层堆栈中弹出顶部动态图层并返回，确保堆栈不为空
DynamicLayer popDynamicLayer() {
  // 获取动态图层堆栈的引用
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  // 断言堆栈不为空
  TORCH_INTERNAL_ASSERT(!dynamicLayerStack.empty());
  // 获取堆栈顶部的动态图层
  auto result = dynamicLayerStack.back();
  // 弹出堆栈顶部的动态图层
  dynamicLayerStack.pop_back();

  // 如果堆栈为空
  if (dynamicLayerStack.empty()) {
    // 如果定义了显示分发跟踪，并且启用了显示分发跟踪
#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
    if (c10::show_dispatch_trace_enabled()) {
      // 打印信息表明动态图层已关闭
      std::cout << "DynamicLayer off" << std::endl;
    }
#endif
    // 设置前后键是否包含在内为 false
    setDynamicLayerFrontBackKeysIncluded(false);
  }

  // 返回弹出的动态图层
  return result;
}

// 推送给定的动态图层到动态图层堆栈中，并返回新的图层 ID
int64_t pushDynamicLayer(DynamicLayer&& dynamic_layer) {
  // 获取动态图层堆栈的引用
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  // 计算新的图层 ID
  int64_t layerId = 1 + dynamicLayerStack.size();
  // 断言新图层的 ID 与其本身的 ID 相等
  TORCH_INTERNAL_ASSERT(layerId == dynamic_layer.layerId());
  // 将动态图层推送到堆栈中
  dynamicLayerStack.emplace_back(std::move(dynamic_layer));

  // 如果是第一个图层
  if (layerId == 1) {
    // 设置前后键包含在内为 true
    setDynamicLayerFrontBackKeysIncluded(true);
    // 如果定义了显示分发跟踪，并且启用了显示分发跟踪
#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
    if (c10::show_dispatch_trace_enabled()) {
      // 打印信息表明动态图层已开启
      std::cout << "DynamicLayer on" << std::endl;
    }
#endif
  }

  // 返回推送的动态图层的 ID
  return layerId;
}

// 初始化并推送新的动态图层到堆栈中，并返回新的图层 ID
int64_t initAndPushDynamicLayer(
    TransformType transform_type,
    optional<c10::SymInt> batch_size,
    optional<RandomnessType> randomness,
    optional<bool> prev_grad_mode,
    optional<bool> prev_fwd_grad_mode,
    optional<bool> functionalize_add_back_views) {
  // 获取动态图层堆栈的引用
  const auto& dynamicLayerStack = dynamicLayerStackAccessor();
  // 计算新的图层 ID
  const auto layerId = 1 + dynamicLayerStack.size();
  // 创建新的动态图层对象
  DynamicLayer new_layer(transform_type, layerId, std::move(batch_size), randomness, prev_grad_mode, prev_fwd_grad_mode, functionalize_add_back_views);
  // 设置解释器状态为存活
  // 注意：调用此函数应在持有 GIL 时，以避免竞争条件
  new_layer.interpreter().set_is_alive(true);
  // 将新的动态图层推送到堆栈中
  pushDynamicLayer(std::move(new_layer));

  // 如果是梯度变换类型
  if (transform_type == TransformType::Grad) {
    // 断言是否定义了前一个梯度模式
    TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
  }
  // 如果是 Jvp 变换类型
  if (transform_type == TransformType::Jvp) {
    // 断言是否定义了前一个前向梯度模式
    TORCH_INTERNAL_ASSERT(prev_fwd_grad_mode.has_value());
  }
  // 返回推送的动态图层的 ID
  return layerId;
}

// 从动态图层堆栈中弹出顶部的动态图层并删除其元数据，确保堆栈不为空
DynamicLayer popDynamicLayerAndDeleteMetadata() {
  // 弹出顶部的动态图层
  auto result = popDynamicLayer();

  // 设置解释器状态为非存活
  // 注意：调用此函数应在持有 GIL 时，以避免竞争条件
  result.interpreter().set_is_alive(false);
  // 返回弹出的动态图层
  return result;
}

// 检查给定张量是否是死亡张量包装
bool isDeadTensorWrapper(const Tensor& tensor) {
  // 尝试获取张量的包装器
  auto* wrapped = maybeGetTensorWrapper(tensor);
  // 如果未获取到包装器，则返回 false
  if (!wrapped) {
    return false;
  }
  // 返回包装器是否非存活
  return !wrapped->is_alive();
}

// 如果给定张量是死亡张量包装，则返回其包装的值，否则返回原始张量
Tensor unwrapIfDead(const Tensor& tensor) {
  // 尝试获取张量的包装器
  auto* wrapped = maybeGetTensorWrapper(tensor);
  // 如果未获取到包装器，则直接返回原始张量
  if (!wrapped) {
    return tensor;
  }
  // 如果包装器存活，则返回原始张量
  if (wrapped->is_alive()) {
    return tensor;
  }
  // 否则返回包装器的值
  return wrapped->value();
}

// 对给定的参数列表中的张量进行原地操作，使用给定的函数
void foreachTensorInplace(std::vector<IValue>& args, int64_t begin, int64_t end,
    std::function<Tensor(const Tensor&)> func) {
   // 定义一个带布尔值的函数，用于对张量进行操作
   auto func_with_bool = [&](const Tensor& tensor, bool unused) { return func(tensor); };
   // 调用带标志位的原地遍历函数
   foreachTensorInplaceWithFlag(args, begin, end, std::bitset<64>(), func_with_bool);
}

// 对给定的参数列表中的张量进行原地操作，并使用给定的函数和标志位
void foreachTensorInplaceWithFlag(std::vector<IValue>& args, int64_t begin, int64_t end,
    std::bitset<64> flags,
    std::function<Tensor(const Tensor&, bool)> func) {
  // 循环遍历指定范围内的参数列表
  for (int64_t i = begin; i < end; ++i) {
    // 获取当前参数的引用
    auto& ivalue = args[i];
    // 检查是否是张量类型
    if (ivalue.isTensor()) {
      // 转换为张量对象
      auto& tensor = ivalue.toTensor();
      // 调用给定的函数处理张量
      func(tensor, flags[i]);
    }
  }
}
    const std::bitset<64> use_flag_relative, const std::function<Tensor(const Tensor&, bool)>& func){
  // 确保 begin 和 end 非负
  TORCH_INTERNAL_ASSERT(begin >= 0);
  TORCH_INTERNAL_ASSERT(end >= 0);
  // 确保 begin 小于等于 end
  TORCH_INTERNAL_ASSERT(begin <= end);
  // 循环遍历索引范围 [begin, end)，以 relative_idx 作为相对索引
  for (int64_t relative_idx = 0; relative_idx < end - begin; relative_idx++) {
    // 根据相对索引确定是否使用对应标志位
    const bool flag = use_flag_relative[relative_idx] == 1;

    // 计算真实索引
    const auto idx = relative_idx + begin;
    auto ivalue = args[idx];

    // 如果 ivalue 是列表类型
    if (ivalue.isList()) {
      bool modified = false;
      // 拷贝列表以便修改
      auto list = ivalue.toList().copy();
      // 遍历列表中的每个元素
      for (const auto list_idx : c10::irange(0, list.size())) {
        const auto& elt = list.get(list_idx);
        // 如果列表元素是 Tensor 类型，则应用 func 函数进行转换
        if (elt.isTensor()) {
          list.set(list_idx, func(elt.toTensor(), flag));
          modified = true;
        }
      }
      // 如果列表有修改，则更新参数列表中的值
      if (modified) {
        args[idx] = list;
      }
      continue;
    }

    // 如果 ivalue 是 Tensor 列表类型
    if (ivalue.isTensorList()) {
      auto list = ivalue.toTensorList();
      // 遍历 Tensor 列表中的每个 Tensor，并应用 func 函数进行转换
      for (const auto list_idx : c10::irange(0, list.size())) {
        list[list_idx] = func(list[list_idx], flag);
      }
      args[idx] = list;
    }

    // 确保 ivalue 不是 GenericDict 类型，因为当前操作不支持 GenericDict
    TORCH_INTERNAL_ASSERT(!ivalue.isGenericDict(), "No operators can accept GenericDict");

    // 如果 ivalue 不是 Tensor 类型，则继续下一个循环
    if (!ivalue.isTensor()) {
      continue;
    }

    // 将 ivalue 转换为 Tensor
    Tensor value = ivalue.toTensor();
    // 根据 func 函数和标志位进行 Tensor 的替换
    Tensor replacement = func(value, flag);
    args[idx] = std::move(replacement);

    // 进行健全性检查，确保替换后的 Tensor 仍然是定义好的
    if (ivalue.toTensor().defined()) {
      TORCH_INTERNAL_ASSERT(args[idx].toTensor().defined());
    }
  }
}

// 重载流输出操作符，用于打印 DynamicLayer 对象的 ID 和键值
std::ostream& operator<< (std::ostream& os, const DynamicLayer& layer) {
  os << layer.layerId() << ":" << layer.key();
  return os;
}

// 重载流输出操作符，用于打印 DynamicLayer 对象的向量
std::ostream& operator<< (std::ostream& os, const std::vector<DynamicLayer>& dls) {
  os << "DynamicLayerStack[ ";
  for (const auto& layer : dls) {
    os << layer << " ";
  }
  os << "]";
  return os;
}

// 检查函数是否是原地操作的辅助函数
bool isInplaceOp(const FunctionSchema& schema) {
  // 如果函数不可变或返回值不止一个，则不是原地操作
  if (!schema.is_mutable() || schema.returns().size() != 1) {
    return false;
  }
  // 检查第一个参数是否被写入
  const auto& first_arg_alias_info = schema.arguments().begin()->alias_info();
  if (!first_arg_alias_info || !first_arg_alias_info->isWrite()) {
    return false;
  }
  // 检查其它参数是否有别名
  for (auto it = schema.arguments().begin() + 1; it != schema.arguments().end(); ++it) {
    const auto& alias_info = it->alias_info();
    if (alias_info) {
      return false;
    }
  }
  // 检查第一个张量是否被返回（即输出具有 (a!) 标记）
  const auto& return_alias_info = schema.returns()[0].alias_info();
  return return_alias_info && return_alias_info->isWrite();
}

// 查找被别名输出的可选索引
std::optional<size_t> findAliasedOutput(const FunctionSchema& schema, const int64_t immutable_input_idx) {
  for (size_t res_idx = 0; res_idx != schema.returns().size(); ++res_idx) {
    // 检查输入是否可能别名输出
    if (schema.may_contain_alias(SchemaArgument(SchemaArgType::input, immutable_input_idx), SchemaArgument(SchemaArgType::output, res_idx))) {
      return res_idx; // 对于当前在 native_functions 中的每个输入，最多一个输出被别名（张量列表计为一个输出）
    }
  }
  return nullopt;
}

#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
// 打印本地 TLS 的辅助函数
static void dump_local_tls() {
  auto tls = c10::impl::tls_local_dispatch_key_set();
  std::cout << "[Local Include] " << tls.included_ << std::endl;
  std::cout << "[Local Exclude] " << tls.excluded_ << std::endl;
}
#endif

// 不包含顶层的结构体
struct WithoutTop {
  WithoutTop();
  ~WithoutTop();
  DynamicLayer layer_;
};

// WithoutTop 的构造函数，从动态层中弹出一个动态层对象
WithoutTop::WithoutTop(): layer_(popDynamicLayer()) {}

// WithoutTop 的析构函数，在析构时将动态层对象推回
WithoutTop::~WithoutTop() {
  pushDynamicLayer(std::move(layer_));
}

// NOTE: [functorch 前端和后端键的回退]
//
// 请先阅读 NOTE: [functorch 解释器堆栈] 以获取一些上下文。
// 以下文档还提供了一些视觉效果：
// https://docs.google.com/document/d/14qyaa3xIjmVxYiMLlIlQErunYgR_uR1WupsKMZlnGY4/edit
//
// functorch 的“变换堆栈”实现如下：
// - 每个变换与 PyTorch 调度器中的一个或多个调度键相关联。例如，vmap -> {FuncTorchBatched, FuncTorchVmapMode}，
//   Autograd -> {Autograd{Backend}, ADInplaceOrView}
// - 每当 functorch 变换处于活动状态时，FuncTorchDynamicLayer{Front, Back}Mode 键都会添加到调度器的本地调度键集中。
//
// DynamicLayerFrontMode 负责：
// 1. 选择位于堆栈顶部的变换并抓取其解释器
// 2. 调用 interpreter.process() 方法，它执行以下操作：
//    2a. 启用/禁用一系列调度键，确保只有属于该转换的调度键是启用的。
//    2b. 重新调度操作。
//
// 最终，DynamicLayerBackMode 捕获从转换中的重新调度。
// DynamicLayerBackMode 负责：
// - 重定向回 DynamicLayerFrontMode

static void dynamicLayerFrontFallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  TORCH_INTERNAL_ASSERT(!dynamicLayerStack.empty());
#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
  // 如果启用了显示调度跟踪，输出动态层栈信息和本地 TLS (Thread Local Storage) 内容。
  if (c10::show_dispatch_trace_enabled()) {
    std::cout << dynamicLayerStack << std::endl;
    dump_local_tls();
  }
#endif
  // 保存当前的 LocalDispatchKeySet（保存到当前的 DynamicLayer）。
  // 在当前作用域结束时，该 LocalDispatchKeySet 会被恢复。
  // 当前 DynamicLayer 调度到下一个（内部的） DynamicLayer 时，
  // 也会临时恢复保存的 LocalDispatchKeySet。
  SaveLocalDispatchKeySet guard;

  // 解开逃逸的 GradWrappers
  auto num_args = op.schema().arguments().size();
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(), unwrapIfDead);

  auto& layer = dynamicLayerStack.back();
  // 调用当前 DynamicLayer 的 interpreter 的 process 方法。
  layer.interpreter().process(op, stack);
}

// 创建一个 RAII 对象，用于恢复 LocalDispatchKeySet。
static c10::impl::ForceDispatchKeyGuard
restoreLocalDispatchKeySetRAII(const c10::impl::LocalDispatchKeySet& key_set) {
  return c10::impl::ForceDispatchKeyGuard(key_set);
}

// 动态层的回退函数，处理带有 grad 特殊情况的情况。
static void dynamicLayerBack(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case) {
  auto restore_guard = restoreLocalDispatchKeySetRAII(
      dynamicLayerStackAccessor().back().interpreter().getSavedLocalDispatchKeySet());
  WithoutTop guard;

  // WithoutTop 存储弹出的 DynamicLayer 对象。
  guard.layer_.interpreter().sendToNextInterpreter(op, stack, grad_special_case);
}

// 处理带有 grad 特殊情况的动态层回退函数。
static void dynamicLayerBackGradSpecialCase(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  return dynamicLayerBack(op, stack, true);
}

// 动态层的普通回退函数。
static void dynamicLayerBackFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  return dynamicLayerBack(op, stack, false);
}

// Torch 库的实现，用于注册 DynamicLayerFrontMode 的回退函数。
TORCH_LIBRARY_IMPL(_, FuncTorchDynamicLayerFrontMode, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerFrontFallback>());
}

// Torch 库的实现，用于注册 DynamicLayerBackMode 的回退函数。
TORCH_LIBRARY_IMPL(_, FuncTorchDynamicLayerBackMode, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerBackFallback>());
}
#define SPECIAL_GRAD_CASE(op) \  // 定义一个宏，用于注册特定操作的梯度处理函数
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&dynamicLayerBackGradSpecialCase>());

TORCH_LIBRARY_IMPL(aten, FuncTorchDynamicLayerBackMode, m) {
  // lift_fresh: 必须是新分配的对象，并且应该被包装。用户不能访问输入版本
  // alias: 这对于复合隐式实例归一化是必要的（running_mean/var 被设置为包装值）
  //        它不是面向用户的函数，但更容易出现可能的错误
  SPECIAL_GRAD_CASE(lift_fresh);  // 注册 lift_fresh 操作的特殊梯度处理函数
  SPECIAL_GRAD_CASE(alias);       // 注册 alias 操作的特殊梯度处理函数
}

} // namespace at::functorch
```