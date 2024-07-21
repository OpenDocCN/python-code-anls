# `.\pytorch\torch\csrc\autograd\VariableTypeManual.cpp`

```
// 包含 ATen 库中的头文件和声明的函数

#include <ATen/RedispatchFunctions.h>
#include <ATen/TracerMode.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/FunctionsManual.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/generated/ViewFuncs.h>
#include <torch/library.h>

// 使用标准库中的工具
#include <utility>

// 使用 ATen 和 torch 的命名空间
using namespace at;
using namespace torch::autograd::generated;

// 声明在 torch::autograd::generated 命名空间中的函数和类型
using torch::autograd::as_view;
using torch::autograd::CreationMeta;

// 定义 torch::autograd::VariableType 命名空间
namespace torch {
namespace autograd {
namespace VariableType {

// 返回给定后端所有已弃用类型的属性的列表
static std::vector<at::DeprecatedTypeProperties*> allTypesForBackends(
    at::ArrayRef<at::Backend> backends) {
  std::vector<DeprecatedTypeProperties*> res;
  res.reserve(backends.size());
  for (auto p : backends) {
    for (const auto s :
         c10::irange(static_cast<int64_t>(ScalarType::NumOptions))) {
      auto& type = getDeprecatedTypeProperties(
          static_cast<Backend>(p), static_cast<ScalarType>(s));
      res.emplace_back(&type);
    }
  }
  return res;
}

// 返回所有 CPU 后端的已弃用类型属性列表
std::vector<at::DeprecatedTypeProperties*> allCPUTypes() {
  return allTypesForBackends({Backend::CPU, Backend::SparseCPU});
}

// 返回所有 CUDA 后端的已弃用类型属性列表
std::vector<at::DeprecatedTypeProperties*> allCUDATypes() {
  at::globalContext().lazyInitCUDA();
  return allTypesForBackends({Backend::CUDA, Backend::SparseCUDA});
}

// 返回所有 XPU 后端的已弃用类型属性列表
std::vector<at::DeprecatedTypeProperties*> allXPUTypes() {
  return allTypesForBackends({Backend::XPU, Backend::SparseXPU});
}

// 返回所有 PrivateUse1 后端的已弃用类型属性列表
std::vector<at::DeprecatedTypeProperties*> allPrivateUser1Types() {
  at::globalContext().lazyInitPrivateUse1();
  return allTypesForBackends(
      {Backend::PrivateUse1, Backend::SparsePrivateUse1});
}

// 匿名命名空间下的函数，用于检查类型转换是否正确
namespace {
const Variable& checked_cast_variable(
    const Tensor& t,
    const char* name,
    int pos) {
  if (!t.defined()) {
    // 如果 Tensor 未定义，抛出错误
    AT_ERROR(
        "Expected a proper Tensor but got None (or an undefined Tensor in C++) ",
        "for argument #",
        pos,
        " '",
        name,
        "'");
  }
  return t;
}

Variable& checked_cast_variable(Tensor& t, const char* name, int pos) {
  if (!t.defined()) {
    // 如果 Tensor 未定义，抛出错误
    AT_ERROR(
        "Expected a proper Tensor but got None (or an undefined Tensor in C++) ",
        "for argument #",
        pos,
        " '",
        name,
        "'");
  }
  return t;
}
} // namespace

// 检查并返回传入 Tensor 的常量引用版本
const Tensor& unpack(const Tensor& t, const char* name, int pos) {
  return checked_cast_variable(t, name, pos);
}

// 检查并返回传入 Tensor 的引用版本
Tensor& unpack(Tensor& t, const char* name, int pos) {
  return checked_cast_variable(t, name, pos);
}

// 检查并返回传入 Tensor 的可选版本
Tensor unpack_opt(const Tensor& t, const char* name, int pos) {
  if (!t.defined()) {
    // 如果 Tensor 未定义，返回一个未定义的 Tensor
    return Tensor();
  }
  return unpack(t, name, pos);
}

// 检查并返回传入 ITensorListRef 的解包操作
std::vector<at::Tensor> unpack(
    const at::ITensorListRef& tl,
    const char* name,
    int pos) {
    // 实现未提供，未完待续...

    // 该函数的具体实现未在提供的代码段中，故未提供注释
    return {};
}
    // 定义一个函数，返回类型为 std::vector<at::Tensor>
    // 函数名和参数列表未提供，应该在代码上下文中查找或完善
    int pos) {
        // 创建一个空的 std::vector<at::Tensor> 对象 ret
        std::vector<at::Tensor> ret;
        // 预留 tl.size() 个元素的空间，以避免在 push_back 时重复分配空间
        ret.reserve(tl.size());
        // 遍历 tl 中的每个元素 t
        for (const auto& t : tl) {
            // 将每个元素 t 添加到 ret 的末尾
            ret.push_back(t);
        }
        // 返回填充好的 std::vector<at::Tensor> 对象 ret
        return ret;
    }
}

namespace {

// Taken from codegened version
// 计算前向传播的原始版本，根据指定的键集和级别操作输入张量
Tensor _fw_primal(c10::DispatchKeySet ks, const Tensor& self, int64_t level) {
  // 解包输入张量 self
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Identity> grad_fn;
  // 检查是否需要计算梯度
  if (compute_requires_grad(self)) {
    // 创建 Identity 梯度函数对象
    grad_fn = std::make_shared<Identity>();
    // 设置下一步梯度函数依赖关系
    grad_fn->set_next_edges(collect_next_edges(self));
  }

  // 执行结果计算，并确保在 Autograd 以下的分发环境中
  auto result = ([&]() {
    at::AutoDispatchBelowAutograd guard;
    // 重新分发至 _fw_primal 函数，根据键集和自动梯度键集 ks
    return at::redispatch::_fw_primal(
        ks & c10::after_autograd_keyset, self_, level);
  })();

  // 如果存在梯度函数，设置历史记录
  if (grad_fn) {
    set_history(flatten_tensor_args(result), grad_fn);
  }

  // 如果定义了前向梯度，检查给定的级别是否有效
  if (isFwGradDefined(self)) {
    // 修改自原始代码生成的部分
    // 明确忽略给定级别的前向梯度
    TORCH_CHECK(level == 0, "Invalid level given to _fw_primal");
    // 结束自原始代码生成的部分
  }

  // 返回计算结果
  return result;
}

// NB: We need a manual variable type kernel so that set_fw_grad properly
// detects that _make_dual is not a forward-differentiable view
//
// This function can be used to create a dual Tensor that holds a tangent to
// compute forward mode gradients. Note that the dual Tensor's primal is a view
// of the given primal and the given tangent is used as-is. This function is
// backward differentiable.
// 创建一个双重张量，其中持有一个切线用于计算前向模式梯度
Tensor _make_dual(
    c10::DispatchKeySet ks,
    const Tensor& primal,
    const Tensor& tangent,
    int64_t level) {
  // 检查是否不支持在具有相同级别的前向梯度上创建双重张量
  TORCH_CHECK(
      !primal._fw_grad(level).defined(),
      "Making a dual Tensor based on a Tensor that "
      "already has a forward gradient at the same level ",
      level,
      " is not supported.");
  // 解包输入的原始张量和切线张量
  auto& primal_ = unpack(primal, "primal", 0);
  auto& tangent_ = unpack(tangent, "tangent", 0);
  std::shared_ptr<ViewBackward0> grad_fn;
  // 检查是否需要计算梯度
  if (compute_requires_grad(primal_)) {
    // 创建 ViewBackward0 梯度函数对象
    grad_fn = std::make_shared<ViewBackward0>();
    // 设置自身符号尺寸作为梯度函数对象的一部分
    grad_fn->self_sym_sizes = primal_.sym_sizes().vec();
    // 设置下一步梯度函数依赖关系
    grad_fn->set_next_edges(collect_next_edges(primal_));
  }

  // 执行结果计算，并确保在 Autograd 以下的分发环境中
  auto result = ([&]() {
    at::AutoDispatchBelowAutograd guard;
    // 重新分发至 _make_dual 函数，根据键集和自动梯度键集 ks
    return at::redispatch::_make_dual(
        ks & c10::after_autograd_keyset, primal_, tangent_, level);
  })();

  // 如果存在梯度函数，设置历史记录
  if (grad_fn) {
    set_history(flatten_tensor_args(result), grad_fn);
  }

  // 检查给定的级别是否有效
  TORCH_CHECK(level == 0, "Invalid level given to _make_dual");

  // 设置切线的前向梯度，不是原地操作
  result._set_fw_grad(tangent_, level, /* is_inplace_op */ false);

  // 返回计算结果
  return result;
}

// We don't have an outplace copy, so this can't be generated automatically
// 我们没有外部复制操作，因此无法自动生成
// 执行张量的复制操作，支持梯度计算和异步非阻塞
Tensor& copy_(
    c10::DispatchKeySet ks,
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  // 解包输入的自身张量和源张量
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  std::shared_ptr<CopyBackwards> grad_fn;
  // 检查是否需要计算梯度，并且自身张量类型支持差分计算
  auto requires_grad = compute_requires_grad(self, src);
  requires_grad &= isDifferentiableType(self.scalar_type());
  // 检查是否是原地操作，并且在计算梯度情况下
  check_inplace(self, requires_grad);
  // 如果需要计算梯度，则创建 CopyBackwards 梯度函数对象
  if (requires_grad) {
    grad_fn = std::make_shared<CopyBackwards>();
  }
    // 设置当前梯度函数节点的下一条边，收集自动微分的下一条边信息
    grad_fn->set_next_edges(collect_next_edges(self, src));
    // 将当前梯度函数节点的源张量选项设置为源张量的选项
    grad_fn->src_options = src.options();
  }
  {
    // 进入自动微分以下的自动调度模式
    at::AutoDispatchBelowAutograd mode;
    // 重新分发复制操作，使用非阻塞方式，支持自动微分后的键集
    at::redispatch::copy_(
        ks & c10::after_autograd_keyset, self_, src_, non_blocking);
  }
  // 将梯度函数附加到张量历史记录中
  rebase_history(self, std::move(grad_fn));

  // 如果自身和源张量都是可微分类型，并且其中至少一个定义了前向梯度
  if (isDifferentiableType(self.scalar_type()) &&
      (isFwGradDefined(self) || isFwGradDefined(src))) {
    // 获取自身和源张量的非可选的前向梯度
    auto self_fw_grad = generated::details::toNonOptFwGrad(self);
    auto src_fw_grad = generated::details::toNonOptFwGrad(src);
    Tensor new_fw_grad;
    // 如果自身有前向梯度
    if (self_fw_grad.defined()) {
      // 如果源张量也有前向梯度，则将其复制到新前向梯度中
      if (src_fw_grad.defined()) {
        new_fw_grad = self_fw_grad.copy_(src_fw_grad);
      } else {
        // 否则用零填充新前向梯度
        new_fw_grad = self_fw_grad.fill_(0);
      }
    } else {
      // 如果自身没有前向梯度，则根据尺寸情况广播或克隆源张量的前向梯度
      if (!self.is_same_size(src_fw_grad)) {
        new_fw_grad = src_fw_grad.broadcast_to(self.sizes());
      } else {
        new_fw_grad = src_fw_grad.clone();
      }
    }
    // 设置自身的前向梯度，设置级别为0，是原地操作
    self._set_fw_grad(new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
  }

  // 返回更新后的自身张量
  return self;
}

const Tensor& resize_(
    c10::DispatchKeySet ks,                                     // 定义函数 resize_，接受调度键集合和自身张量作为参数
    const Tensor& self,                                         // 常量引用类型的自身张量参数
    SymIntArrayRef size,                                        // 使用 SymIntArrayRef 类型的 size 参数
    std::optional<MemoryFormat> optional_memory_format) {       // 可选的内存格式参数
  auto& self_ = unpack(self, "self", 0);                        // 解包自身张量，并命名为 self_
  if (self.requires_grad()) {                                   // 检查自身张量是否需要梯度，若是则报错
    AT_ERROR("cannot resize variables that require grad");
  }
  {
    at::AutoDispatchBelowAutograd mode;                         // 进入自动微分以下的自动调度模式
    at::redispatch::resize__symint(                             // 调用 redispatch 中的 resize__symint 函数
        ks & c10::after_autograd_keyset, self_, size, optional_memory_format);
  }

  if (self._fw_grad(/* level */ 0).defined()) {                 // 如果自身张量的前向梯度已定义，则报错
    AT_ERROR("cannot resize variables that has a forward grad");
  }

  return self;                                                  // 返回自身张量的常量引用
}

const Tensor& resize_as_(
    c10::DispatchKeySet ks,                                     // 定义函数 resize_as_，接受调度键集合和两个张量作为参数
    const Tensor& self,                                         // 常量引用类型的自身张量参数
    const Tensor& the_template,                                 // 常量引用类型的模板张量参数
    std::optional<MemoryFormat> optional_memory_format) {       // 可选的内存格式参数
  auto& self_ = unpack(self, "self", 0);                        // 解包自身张量，并命名为 self_
  auto& the_template_ = unpack(the_template, "the_template", 1);// 解包模板张量，并命名为 the_template_
  if (self.requires_grad()) {                                   // 检查自身张量是否需要梯度，若是则报错
    AT_ERROR("cannot resize variables that require grad");
  }
  {
    at::AutoDispatchBelowAutograd mode;                         // 进入自动微分以下的自动调度模式
    at::redispatch::resize_as_(                                 // 调用 redispatch 中的 resize_as_ 函数
        ks & c10::after_autograd_keyset,
        self_,
        the_template_,
        optional_memory_format);
  }

  // Handle fw grad
  if (self._fw_grad(/* level */ 0).defined()) {                 // 如果自身张量的前向梯度已定义，则报错
    AT_ERROR("cannot resize variables that has a forward grad");
  }

  return self;                                                  // 返回自身张量的常量引用
}

Tensor detach(c10::DispatchKeySet ks, const Tensor& self) {      // 定义函数 detach，接受调度键集合和自身张量作为参数
  auto& self_ = unpack(self, "self", 0);                         // 解包自身张量，并命名为 self_
  RECORD_FUNCTION("detach", std::vector<c10::IValue>({self}));  // 记录函数执行，标记为 detach
  auto result = ([&]() {                                         // 使用 Lambda 表达式执行以下代码块
    at::AutoDispatchBelowAutograd guard;                         // 进入自动微分以下的自动调度保护区域
    return at::redispatch::detach(                               // 调用 redispatch 中的 detach 函数
        ks & c10::after_autograd_keyset, self_);
  })();
  namedinference::propagate_names(result, self);                 // 根据结果和自身张量，推断名称

  // Detach the forward grads by not setting anything on the result
  // 通过不在结果上设置任何东西来分离前向梯度

  return result;                                                 // 返回结果张量
}

Tensor& detach_(c10::DispatchKeySet ks, Tensor& self) {          // 定义函数 detach_，接受调度键集合和自身张量作为参数
  RECORD_FUNCTION("detach_", std::vector<c10::IValue>({self})); // 记录函数执行，标记为 detach_
  if (self.is_view()) {                                          // 如果自身张量是视图，则执行以下操作
    // See NOTE [ View + Inplace detection ]
    // 参见注释 [ 视图 + 原地检测 ]
    AT_ERROR(
        "Can't detach views in-place. Use detach() instead. "
        "If you are using DistributedDataParallel (DDP) for training, "
        "and gradient_as_bucket_view is set as True, gradients are "
        "views of DDP buckets, and hence detach_() cannot be called "
        "on these gradients. To fix this error, please refer to the "
        "Optimizer.zero_grad() function in torch/optim/optimizer.py "
        "as the solution.");
  }


// 抛出错误消息，指出不能原地执行视图分离操作，建议使用 detach() 方法代替。
// 如果在使用 DistributedDataParallel (DDP) 进行训练，并且设置了 gradient_as_bucket_view 为 True，
// 梯度将是 DDP 桶的视图，因此不能对这些梯度调用 detach_()。
// 要解决此错误，请参考 torch/optim/optimizer.py 中的 Optimizer.zero_grad() 函数。

// 返回控制权给调用者。



  // I think the choice here is conservative.  In principle, doing
  // an in-place detach should give us the ability to just clear
  // the autograd meta.  But this function ONLY resets requires_grad,
  // grad_fn and output_nr; there's other metadata like debug name
  // and hooks which aren't cleared.  Is this function supposed to
  // clear those too? I'm not too sure, so I'm leaving it be for now.
  auto autograd_meta = impl::materialize_autograd_meta(self);
  autograd_meta->set_requires_grad(false, self.unsafeGetTensorImpl());
  autograd_meta->grad_fn_.reset();
  autograd_meta->output_nr_ = 0;
  autograd_meta->fw_grad_.reset();

  return self;


// 我认为这里的选择比较保守。原则上，进行原地分离应该能够清除自动求导元数据。
// 但是这个函数只重置了 requires_grad、grad_fn 和 output_nr 这些元数据；
// 其他像调试名称和钩子等元数据并没有被清除。不确定这个函数是否应该清除那些元数据，
// 所以暂时保持不变。

// 实例化自动求导元数据
auto autograd_meta = impl::materialize_autograd_meta(self);
// 设置 requires_grad 为 false，并传入当前张量的底层实现
autograd_meta->set_requires_grad(false, self.unsafeGetTensorImpl());
// 重置 grad_fn
autograd_meta->grad_fn_.reset();
// 将 output_nr 设置为 0
autograd_meta->output_nr_ = 0;
// 重置 fw_grad
autograd_meta->fw_grad_.reset();

// 返回处理后的张量
return self;
} // 结束 TORCH_LIBRARY_IMPL(aten, Autograd, m) 块

// 下面的注册列表中的操作被注册为：
//   (1) CompositeImplicitAutograd 内核
//   (2) Autograd 内核
//   (3) CompositeExplicitAutograd 内核以及额外的 Autograd 内核
// 原因是 (3) 中的操作还使用了 dispatch （例如注册 CPU/CUDA/QuantizedCPU 内核），
// 这些操作会跳过 CompositeImplicitAutograd 内核以便 Autograd 使用，因此我们同时
// 将它们注册到 CompositeExplicitAutograd 和 Autograd 内核中。详细信息请参见：
// https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword
// 不变条件：
// - 在下面注册到 CompositeImplicitAutograd 或 CompositeExplicitAutograd 的操作必须匹配
//   在 tools/autograd/gen_variable_type.py 中设置的 `MANUAL_BACKEND` 集合，并且在
//   native_functions.yaml 中具有 manual_kernel_registration=True。
// - 在下面注册到 DispatchKey::Autograd 的操作必须包含在 tools/autograd/gen_variable_type.py
//   中的 `MANUAL_AUTOGRAD` 集合中。

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  // 注册 "resize_" 操作，使用 Autograd DispatchKey，并调用 VariableType::resize_ 函数
  m.impl(
      "resize_",
      torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::resize_)));
  // 注册 "resize_as_" 操作，使用 Autograd DispatchKey，并调用 VariableType::resize_as_ 函数
  m.impl(
      "resize_as_",
      torch::dispatch(
          DispatchKey::Autograd, TORCH_FN(VariableType::resize_as_)));
  // 注册 "detach" 操作，使用 Autograd DispatchKey，并调用 VariableType::detach 函数
  m.impl(
      "detach",
      torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::detach)));
  // 注册 "detach_" 操作，使用 Autograd DispatchKey，并调用 VariableType::detach_ 函数
  m.impl(
      "detach_",
      torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::detach_)));
  // 注册 "copy_" 操作，使用 Autograd DispatchKey，并调用 VariableType::copy_ 函数
  m.impl(
      "copy_",
      torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::copy_)));
  // 注册 "_fw_primal" 操作，使用 Autograd DispatchKey，并调用 VariableType::_fw_primal 函数
  m.impl(
      "_fw_primal",
      torch::dispatch(
          DispatchKey::Autograd, TORCH_FN(VariableType::_fw_primal)));
  // 注册 "_make_dual" 操作，使用 Autograd DispatchKey，并调用 VariableType::_make_dual 函数
  m.impl(
      "_make_dual",
      torch::dispatch(
          DispatchKey::Autograd, TORCH_FN(VariableType::_make_dual)));
}

} // 结束 namespace

} // 结束 namespace VariableType

} // 结束 namespace autograd

namespace ADInplaceOrView {

// 定义 CREATION_META_DEFINITION 宏
#define CREATION_META_DEFINITION                            \
  InferenceMode::is_enabled()                               \
      ? CreationMeta::INFERENCE_MODE                        \
      : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT \
                                    : CreationMeta::NO_GRAD_MODE)

// 实现 Tensor 类的 copy_ 操作
static Tensor& copy_(
    c10::DispatchKeySet ks,
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  {
    // 在 ADInplaceOrView 下面的自动分发保护块
    at::AutoDispatchBelowADInplaceOrView guard;
    // 重新调度 copy_ 操作，使用 ks 与 c10::after_ADInplaceOrView_keyset 相交后的 DispatchKeySet
    at::redispatch::copy_(
        ks & c10::after_ADInplaceOrView_keyset, self, src, non_blocking);
  }
  // 增加 Tensor 版本号
  torch::autograd::increment_version(self);
  // 返回修改后的 Tensor 引用
  return self;
}

// 实现 Tensor 类的 resize_ 操作
static const Tensor& resize_(
    c10::DispatchKeySet ks,
    const Tensor& self,
    SymIntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  // 保存原始大小以验证是否真正改变了 self 的大小
  auto org_size = self.sym_sizes().vec();
  {
    // 在 ADInplaceOrView 下面的自动分发保护块
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::resize__symint(
        ks & c10::after_ADInplaceOrView_keyset,
        self,
        size,
        optional_memory_format);

调用 `at::redispatch::resize__symint` 函数，对张量 `self` 进行重置大小操作。参数包括 `ks` 和 `optional_memory_format`，其中 `ks` 是一个引用，用于可能在操作后更新的键集合。


  }

结束 `resize__symint` 函数的代码块。


  // If `self` was resized, increment the version.
  if (org_size != size) {

如果张量 `self` 被重新调整大小，则执行以下操作：


    torch::autograd::increment_version(self);

通过调用 `torch::autograd::increment_version` 函数增加张量 `self` 的版本号。


  }

结束条件语句块。


  return self;

返回更新后的张量 `self`。
// 静态函数，将 self 调整为与 the_template 相同大小的张量
static const Tensor& resize_as_(
    c10::DispatchKeySet ks,                     // 指定的调度键集合
    const Tensor& self,                         // 要调整大小的张量
    const Tensor& the_template,                 // 作为模板的张量
    std::optional<MemoryFormat> optional_memory_format) {  // 可选的内存格式

  // 保存原始的大小以验证是否实际调整了 self 的大小
  auto org_size = self.sym_sizes().vec();

  {
    // 进入自动分派区域，确保在 ADInplaceOrView 之下进行操作或视图
    at::AutoDispatchBelowADInplaceOrView guard;

    // 调用 redispatch 的 resize_as_ 函数，使用指定的键集和内存格式
    at::redispatch::resize_as_(
        ks & c10::after_ADInplaceOrView_keyset,  // 使用位与操作确保在 ADInplaceOrView 之后进行调度
        self,                                   // 要调整大小的张量
        the_template,                           // 作为模板的张量
        optional_memory_format);                // 可选的内存格式
  }

  // 如果 self 被调整大小，则增加版本号
  if (org_size != the_template.sym_sizes()) {
    torch::autograd::increment_version(self);
  }

  // 返回调整大小后的张量 self
  return self;
}

// 分离张量的视图，返回分离后的张量
static Tensor detach(c10::DispatchKeySet ks, const Tensor& self) {
  // 使用 lambda 表达式在自动分派区域内调用 detach 函数
  auto out = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::detach::redispatch(
        ks & c10::after_ADInplaceOrView_keyset,  // 使用位与操作确保在 ADInplaceOrView 之后进行调度
        self);                                  // 要分离的张量
  })();

  // 创建视图并返回其结果
  // 注意: detach() 不能作为普通视图操作符，因为代码生成器会生成 allow_tensor_metadata_change = True
  auto result = as_view(
      /* base */ self,
      /* output */ out,
      /* is_bw_differentiable */ false,
      /* is_fw_differentiable */ false,
      /* view_func */ nullptr,
      /* rev_view_func */ nullptr,
      /* creation_meta */ CreationMeta::DEFAULT,
      /* allow_tensor_metadata_change=*/false);

  return result;
}

// 前向原始视图函数，返回原始视图的张量
static Tensor _fw_primal(
    c10::DispatchKeySet ks,  // 指定的调度键集合
    const Tensor& self,      // 要创建原始视图的张量
    int64_t level) {         // 级别参数

  // 使用 lambda 表达式在自动分派区域内调用 alias 函数
  auto tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::alias(self);
  })();

  // 初始化用于反向视图的函数和逆函数
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func = nullptr;

  // 如果张量不支持作为 stride 视图，则创建 ViewViewFunc
  if (!self.unsafeGetTensorImpl()->support_as_strided()) {
    func = std::make_unique<ViewViewFunc>(self.sym_sizes());
    rev_func = [=](const at::Tensor& input_view) {
      TORCH_INTERNAL_ASSERT(
          false,
          "Reverse view_func for _fw_primal() is not currently supported");
      return Tensor();
    };
  }

  // 创建视图并返回其结果
  auto result = as_view(
      /* base */ self,
      /* output */ tmp,
      /* is_bw_differentiable */ true,
      /* is_fw_differentiable */ false,
      /* view_func */ std::move(func),
      /* rev_view_func */ std::move(rev_func),
      /* creation_meta */ CREATION_META_DEFINITION);

  return result;
}

// 创建双重视图的函数，返回创建的双重视图的张量
// 注意: 这不会进一步重新分派
static Tensor _make_dual(
    c10::DispatchKeySet ks,       // 指定的调度键集合
    const Tensor& primal,         // 原始张量
    const Tensor& tangent,        // 切向张量
    int64_t level) {              // 级别参数

  // 使用 lambda 表达式在自动分派区域内调用 alias 函数
  auto tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::alias(primal);
  })();

  // 初始化用于反向视图的函数和逆函数
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func = nullptr;

  // 如果原始张量不支持作为 stride 视图，则创建 ViewViewFunc
  if (!primal.unsafeGetTensorImpl()->support_as_strided()) {
    func = std::make_unique<ViewViewFunc>(primal.sym_sizes());
  }

  // 创建视图并返回其结果
  auto result = as_view(
      /* base */ primal,
      /* output */ tmp,
      /* is_bw_differentiable */ true,
      /* is_fw_differentiable */ false,
      /* view_func */ std::move(func),
      /* rev_view_func */ std::move(rev_func),
      /* creation_meta */ CREATION_META_DEFINITION);

  return result;
}
    // 创建一个新的 ViewViewFunc 对象，使用 primal.sym_sizes() 作为参数
    func = std::make_unique<ViewViewFunc>(primal.sym_sizes());
    
    // 创建一个 lambda 函数 rev_func，接受一个名为 input_view 的常量引用参数
    rev_func = [=](const at::Tensor& input_view) {
      // 内部断言，如果条件为假则触发错误信息
      TORCH_INTERNAL_ASSERT(
          false,
          "Reverse view_func for _make_dual() is not currently supported");
      // 返回一个空的 Tensor 对象
      return Tensor();
    };
  }
  
  // 调用 as_view 函数，传入多个参数来创建一个视图对象 result
  auto result = as_view(
      /* base */ primal,
      /* output */ tmp,
      /* is_bw_differentiable */ true,
      /* is_fw_differentiable */ false,
      /* view_func */ std::move(func),
      /* rev_view_func */ std::move(rev_func),
      /* creation_meta */ CREATION_META_DEFINITION);

  // 返回创建的视图对象 result
  return result;
} // 结束 TORCH_LIBRARY_IMPL 命名空间的定义

namespace {  // 匿名命名空间，限定作用域在当前文件内部

TORCH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {  // 定义 TORCH_LIBRARY_IMPL 宏，注册 aten 库的实现
  m.impl(  // 注册 "copy_" 实现
      "copy_",  // 函数名
      torch::dispatch(  // 使用 torch::dispatch 函数进行注册
          DispatchKey::ADInplaceOrView,  // 指定调度键为 ADInplaceOrView
          TORCH_FN(ADInplaceOrView::copy_)));  // 指定实现函数为 ADInplaceOrView::copy_

  m.impl(  // 注册 "detach" 实现，以下类似
      "detach",
      torch::dispatch(
          DispatchKey::ADInplaceOrView,
          TORCH_FN(ADInplaceOrView::detach)));

  m.impl(
      "resize_",
      torch::dispatch(
          DispatchKey::ADInplaceOrView,
          TORCH_FN(ADInplaceOrView::resize_)));

  m.impl(
      "resize_as_",
      torch::dispatch(
          DispatchKey::ADInplaceOrView,
          TORCH_FN(ADInplaceOrView::resize_as_)));

  m.impl(
      "_fw_primal",
      torch::dispatch(
          DispatchKey::ADInplaceOrView,
          TORCH_FN(ADInplaceOrView::_fw_primal)));

  m.impl(
      "_make_dual",
      torch::dispatch(
          DispatchKey::ADInplaceOrView,
          TORCH_FN(ADInplaceOrView::_make_dual)));
}
} // 结束匿名命名空间

} // 结束命名空间 ADInplaceOrView
} // 结束命名空间 torch
```