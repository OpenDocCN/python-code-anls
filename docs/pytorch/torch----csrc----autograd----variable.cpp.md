# `.\pytorch\torch\csrc\autograd\variable.cpp`

```py
// 引入 Torch 的自动求导变量头文件
#include <torch/csrc/autograd/variable.h>

// 引入 Torch 自动求导模块的相关头文件
#include <torch/csrc/autograd/InferenceMode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/csrc/autograd/generated/ViewFuncs.h>
#include <torch/csrc/autograd/utils/error_messages.h>

// 引入 ATen 库的头文件
#include <ATen/ATen.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/MemoryOverlap.h>
#include <c10/util/Exception.h>

// 引入标准库头文件
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Torch 命名空间
namespace torch {
namespace autograd {

// 返回一个 ViewFunc 对象，其对应的视图与给定张量的形状、步长和存储偏移匹配
// 注意：在移动设备上，可能没有 as_strided() 操作，因此生成的 AsStridedViewFunc 可能不可用
static std::unique_ptr<ViewFunc> create_view_func_matching(const Variable& t) {
#ifdef AS_STRIDED_VIEW_FUNC_AVAILABLE
  // 使用给定张量的符号大小、符号步长和符号存储偏移创建一个 AsStridedViewFunc
  return std::make_unique<torch::autograd::generated::AsStridedViewFunc>(
      t.sym_sizes(), t.sym_strides(), t.sym_storage_offset());
#else
  // 返回一个 ErroringViewFunc，表示 as_strided() 不可用
  return std::make_unique<ErroringViewFunc>("as_strided() not available");
#endif
}

// DifferentiableViewMeta 的构造函数，用于管理可微视图的元信息
DifferentiableViewMeta::DifferentiableViewMeta(
    at::TensorImpl* self_impl,
    std::optional<ViewInfo> backward_info,
    std::optional<ViewInfo> forward_info,
    bool shared_view_info,
    CreationMeta creation_meta)
    : AutogradMeta(self_impl),
      backward_info_(std::move(backward_info)),
      forward_info_(std::move(forward_info)),
      shared_view_info_(shared_view_info),
      creation_meta_(creation_meta) {
  // 标记为视图
  is_view_ = true;
  // 如果存在向后视图信息，则设置版本计数器
  if (backward_info_.has_value()) {
    self_impl->set_version_counter(
        impl::version_counter(backward_info_.value().base_));
    attr_version_ = self_impl->version_counter().current_version();
    // 内部断言：向后视图信息的基础张量与当前张量不同
    TORCH_INTERNAL_ASSERT(
        backward_info_.value().base_.unsafeGetTensorImpl() != self_impl);
  }
  // 如果共享视图信息为真
  if (shared_view_info_) {
    // 内部断言：共享视图信息要求有向后视图信息
    TORCH_INTERNAL_ASSERT(
        backward_info_.has_value(),
        "Shared view info require a backward view info.");
    // 内部断言：共享视图信息要求向前视图信息为空
    TORCH_INTERNAL_ASSERT(
        !forward_info_.has_value(),
        "Shared view info require forward view info to be empty")
  }
}

// 链接此视图信息与基础和张量之间的新视图操作
ViewInfo ViewInfo::chain(
    const Variable& base,
    const Variable& tensor,
    std::unique_ptr<ViewFunc> view_func,
  // 使用给定的 view_func 来设置视图函数
  // 当 as_strided 不被支持或者视图函数改变了元数据但未被 as_strided 记录时，在反向传播中用来恢复视图
  // 详见 "View + Inplace update on base tensor" 和 "View + Inplace update on view tensor"
  // 了解更多关于在反向传播中如何使用这个函数的细节。

  // 如果 view_func 不为空
  if (view_func) {
    // 如果当前视图和其父视图都有一个 view_func
    if (view_fn_) {
      // 创建一个 ChainedViewFunc 对象，用于链式调用视图函数
      view_func = std::make_unique<ChainedViewFunc>(
          view_fn_->clone_and_set(), std::move(view_func));

      // 假设 view_fn_ 和 rev_view_fn_ 总是一起存在或者都未设置
      auto prev_rev_fn = rev_view_fn_;
      // 创建一个 lambda 函数，用于反向传播时应用视图函数的逆操作
      rev_view_func = [=](const at::Tensor& root_view) {
        auto temp = rev_view_func(root_view);
        return prev_rev_fn(temp);
      };
    } else {
      // 如果当前视图有一个 view_func，但其父视图没有
      // 如果基础张量支持 as_strided()
      if (base.unsafeGetTensorImpl()->support_as_strided()) {
        // 创建一个匹配基础张量视图函数的对象
        auto match_base_view_func = create_view_func_matching(base);
        // 创建一个 ChainedViewFunc 对象，用于链式调用视图函数
        view_func = std::make_unique<ChainedViewFunc>(
            std::move(match_base_view_func), std::move(view_func));

        // 假设 view_fn_ 和 rev_view_fn_ 总是一起存在或者都未设置
        const auto& root_base = base._base();
        auto root_base_size = root_base.sym_sizes().vec();
        auto root_base_stride = root_base.sym_strides().vec();
        auto root_base_storage_offset = root_base.sym_storage_offset();
        // 创建一个 lambda 函数，用于反向传播时应用视图函数的逆操作
        rev_view_func = [=](const at::Tensor& root_view) {
          auto temp = rev_view_func(root_view);
          return temp.as_strided_symint(
              root_base_size, root_base_stride, root_base_storage_offset);
        };
      } else {
        // 这种情况应该比较少见：父视图既没有 view_func() 又不支持 as_strided()；
        // 没有明显的方法将两个视图链在一起。
        auto error_msg =
            ("Attempted to chain views when the parent view has no view_func() and "
             "does not support as_strided(). This is not supported.");
        // 创建一个 ErroringViewFunc 对象，用于报错处理
        view_func = std::make_unique<ErroringViewFunc>(error_msg);
        // 创建一个 lambda 函数，用于反向传播时报错
        rev_view_func = [=](const at::Tensor& root_view) {
          TORCH_CHECK(false, error_msg);
          return root_view;
        };
      }
    }
  } else if (view_fn_) {
    // 如果当前视图没有 view_func，但其父视图有
    // 创建一个匹配张量视图函数的对象
    auto match_tensor_view_func = create_view_func_matching(tensor);
    // 创建一个 ChainedViewFunc 对象，用于链式调用视图函数
    view_func = std::make_unique<ChainedViewFunc>(
        view_fn_->clone_and_set(), std::move(match_tensor_view_func));

    // 假设 view_fn_ 和 rev_view_fn_ 总是一起存在或者都未设置
    auto prev_rev_view_fn = rev_view_fn_;
    auto base_size = base.sym_sizes().vec();
    auto base_stride = base.sym_strides().vec();
    # 获取基础张量的存储偏移量
    auto base_storage_offset = base.sym_storage_offset();
    
    # 定义一个 lambda 函数 rev_view_func，用于生成反向视图
    rev_view_func = [=](const at::Tensor& root_view) {
      # 使用基础大小、步长和存储偏移量，创建一个 strided 张量 temp
      auto temp = root_view.as_strided_symint(
          base_size, base_stride, base_storage_offset);
      # 调用之前定义的反向视图函数 prev_rev_view_fn，并返回结果
      return prev_rev_view_fn(temp);
    };
  }
  
  # 返回一个 ViewInfo 对象，其中包含基础张量 base_，以及移动了的 view_func 和 rev_view_func
  return ViewInfo(base_, std::move(view_func), std::move(rev_view_func));
} // 结束匿名命名空间

namespace {

at::Tensor singleton_undefined_tensor; // 声明一个全局的未定义张量对象

struct ConcreteAutogradMetaFactory : public c10::impl::AutogradMetaFactory {
  // 实现 AutogradMetaFactory 接口，用于创建 AutogradMeta 对象
  std::unique_ptr<c10::AutogradMetaInterface> make() const override {
    return std::make_unique<AutogradMeta>();
  }
  // 返回全局的未定义张量对象
  const at::Tensor& undefined_tensor() const override {
    return singleton_undefined_tensor;
  }
};

ConcreteAutogradMetaFactory meta_factory; // 创建 ConcreteAutogradMetaFactory 的实例

// 将 meta_factory 注册为 AutogradMetaFactory 的实现
static c10::impl::AutogradMetaFactoryRegisterer meta_factory_registerer(
    &meta_factory);

} // 结束匿名命名空间 "impl"

namespace impl {

// 根据自身的实现创建 AutogradMeta 对象
AutogradMeta* materialize_autograd_meta(const at::TensorBase& self) {
  // 检查张量是否已定义，若未定义则报错
  TORCH_CHECK(
      self.defined(),
      "cannot call materialize_autograd_meta() on undefined tensor");
  auto p = self.unsafeGetTensorImpl(); // 获取张量的底层实现指针
  if (!p->autograd_meta()) {
    p->set_autograd_meta(std::make_unique<AutogradMeta>()); // 若无 AutogradMeta，则创建新的对象
  }
  return get_autograd_meta(self); // 返回张量的 AutogradMeta 指针
}

// 更新张量的 hooks 当新的 grad_fn 被设置时
static void update_tensor_hooks_on_new_gradfn(
    const at::TensorBase& self,
    const std::shared_ptr<torch::autograd::Node>& old_fn,
    const std::shared_ptr<torch::autograd::Node>& new_fn) {
  // 当张量的 grad_fn 被更新时调用此函数。
  // 假设 new_fn 尚未有自己的 hooks。

  // 这个函数执行两件事：
  // (1) 当 grad_fn 被更新时，重置列表以避免新 hooks 误注册到旧的 grad_fn 上。
  //     注意，旧的 cpp_hooks_list_ 仍由旧的 grad_fn 保持活动状态，因此注册到旧版本张量的旧 hooks 仍将保持活跃。
  // (2) 如果有 retains_grad hook 注册了，将其从旧的 cpp_hooks_list_ 移动到新的列表中。

  const auto& meta = impl::get_autograd_meta(self); // 获取张量的 AutogradMeta
  TORCH_INTERNAL_ASSERT(meta);
  TORCH_INTERNAL_ASSERT(new_fn);
  meta->cpp_hooks_list_ = nullptr; // 重置 cpp_hooks_list_
  const c10::impl::PyInterpreter* interp =
      self.unsafeGetTensorImpl()->pyobj_slot()->pyobj_interpreter();
  if (interp) {
    (*interp)->reset_backward_hooks(self.unsafeGetTensorImpl()); // 重置反向传播 hooks
  }
  if (self.retains_grad()) {
    TORCH_INTERNAL_ASSERT(old_fn);
    auto out = old_fn->pop_retains_grad_hook(self.output_nr()); // 弹出旧的 retains_grad hook
    TORCH_INTERNAL_ASSERT(out != nullptr);
    new_fn->add_retains_grad_hook(std::move(out), self.output_nr()); // 添加到新的 grad_fn
  }
}

// 在 Variable 上重新基于历史记录
void rebase_history(const Variable& self, Edge gradient_edge) {
  TORCH_INTERNAL_ASSERT(gradient_edge.function != nullptr);
  const auto& meta = impl::get_autograd_meta(self); // 获取 Variable 的 AutogradMeta
  auto old_fn = meta != nullptr ? meta->grad_fn_ : nullptr; // 获取旧的 grad_fn
  auto diff_view_meta = get_view_autograd_meta(self);
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    // 查看 View + Inplace 检测的 NOTE
    auto creation_meta = diff_view_meta->get_creation_meta();
    // 不在此处使用 handle_view_on_rebase，因为应在此之前调用 check_inplace，并且要么抛出错误
    // 要么执行 handle_view_on_rebase。
    TORCH_INTERNAL_ASSERT(creation_meta == CreationMeta::DEFAULT);
    TORCH_INTERNAL_ASSERT(gradient_edge.input_nr == 0);
    // 断言梯度边的函数指针不为空
    TORCH_INTERNAL_ASSERT(gradient_edge.function);
    // 检查函数修改视图时必须返回单个变量
    TORCH_CHECK(
        gradient_edge.function->num_inputs() == 1,
        "Functions which modify views in-place must return a single Variable");
    // 获取反向视图信息
    const auto& view_info = diff_view_meta->get_backward_view();
    // 设置输出编号为梯度边的输入编号
    diff_view_meta->output_nr_ = gradient_edge.input_nr;
    // 创建一个共享的 CopySlices 对象，用于拷贝视图数据
    auto copy_slices = std::make_shared<CopySlices>(
        view_info.base_,
        at::TensorGeometry(self),
        // 如果视图信息包含视图函数，克隆并设置视图函数
        view_info.has_view_fn() ? view_info.view_fn().clone_and_set() : nullptr,
        std::move(gradient_edge.function));
    // 如果 self 需要梯度
    if (self.requires_grad()) {
      // 如果之前 self 没有需要梯度，没有钩子可以移动
      torch::autograd::impl::update_tensor_hooks_on_new_gradfn(
          view_info.base_, view_info.base_.grad_fn(), copy_slices);
    }
    // 设置 self 的梯度边为 copy_slices
    set_gradient_edge(view_info.base_, {std::move(copy_slices), 0});
    // 触发更新视图的 grad_fn
    self.grad_fn(); // trigger an update to the view's grad_fn
    // 函数结束，返回
    return;
  }

  // 设置 self 的梯度边为 gradient_edge
  set_gradient_edge(self, std::move(gradient_edge));
  // 传递 self 和其旧的 grad_fn，以避免重入调用 grad_fn
  torch::autograd::impl::update_tensor_hooks_on_new_gradfn(
      self, old_fn, self.grad_fn());
}

// 创建一个 C++ 钩子函数，用于在张量操作前执行特定操作
void create_cpp_hook(const at::TensorBase& self, bool is_retains_grad_hook) {
  // 获取张量的梯度函数
  const auto& fn = self.grad_fn();
  // 获取张量的自动微分元数据，并初始化 C++ 钩子列表
  std::shared_ptr<hooks_list>& list =
      materialize_autograd_meta(self)->cpp_hooks_list_;
  list.reset(new hooks_list());
  // 创建一个新的 C++ 函数钩子，并添加到钩子列表中
  auto hook_ptr =
      std::make_unique<CppFunctionTensorPreHook>(list, self.output_nr());
  // 注意：如果 fn 不存在，可能可以只更新 hooks_，但当前不会影响结果，保持此方式不变
  // 清除当前的钩子
  clear_hooks(self);
  // 添加新的钩子到列表中
  add_hook(self, std::make_unique<CppFunctionTensorPreHook>(list, 0));
  // 如果存在梯度函数，将钩子添加到梯度函数的前置钩子列表中
  if (fn) {
    fn->add_tensor_pre_hook(std::move(hook_ptr));
  }
}

// 设置梯度累加器
void set_grad_accumulator(
    const Variable& self,
    std::weak_ptr<Node> grad_accumulator) {
  // 获取自动微分元数据，并设置梯度累加器
  materialize_autograd_meta(self)->grad_accumulator_ =
      std::move(grad_accumulator);
}

// 尝试获取梯度累加器
std::shared_ptr<Node> try_get_grad_accumulator(const Variable& self) {
  // 如果存在自动微分元数据，返回梯度累加器
  if (get_autograd_meta(self)) {
    return get_autograd_meta(self)->grad_accumulator_.lock();
  } else {
    return nullptr;
  }
}

// 获取梯度累加器
std::shared_ptr<Node> grad_accumulator(const Variable& self) {
  // 获取自动微分元数据
  auto autograd_meta = get_autograd_meta(self);
  // 如果元数据不存在，返回空指针
  if (!autograd_meta) {
    return nullptr;
  }
  // 如果存在梯度函数，抛出逻辑错误，因为应该只在叶子变量上调用 grad_accumulator()
  if (autograd_meta->grad_fn_) {
    throw std::logic_error(
        "grad_accumulator() should be only called on leaf Variables");
  }
  // 如果不需要梯度，返回空指针
  if (!autograd_meta->requires_grad_) {
    return nullptr;
  }

  // 使用互斥锁保护下面的操作
  std::lock_guard<std::mutex> lock(autograd_meta->mutex_);

  // 尝试获取梯度累加器
  auto result = autograd_meta->grad_accumulator_.lock();
  if (result)
    return result;

  // 如果不存在梯度累加器，创建一个新的累积梯度对象，并设置为梯度累加器
  c10::raw::intrusive_ptr::incref(self.unsafeGetTensorImpl());
  auto intrusive_from_this =
      c10::intrusive_ptr<at::TensorImpl>::reclaim(self.unsafeGetTensorImpl());
  result = std::make_shared<AccumulateGrad>(
      Variable(std::move(intrusive_from_this)));
  autograd_meta->grad_accumulator_ = result;
  return result;
}

// 获取梯度边缘
Edge gradient_edge(const Variable& self) {
  // 如果存在梯度函数，则返回该梯度函数的边缘
  if (const auto& gradient = self.grad_fn()) {
    return Edge(gradient, self.output_nr());
  } else {
    // 否则返回该变量的梯度累加器的边缘
    return Edge(grad_accumulator(self), 0);
  }
}
// 设置梯度的边缘信息，将给定的梯度函数和输入编号移动到自变量的自动求导元信息中
void set_gradient_edge(const Variable& self, Edge edge) {
  // 获取自变量的自动求导元信息
  auto* meta = materialize_autograd_meta(self);
  // 设置梯度函数为给定的 edge.function
  meta->grad_fn_ = std::move(edge.function);
  // 设置输出编号为给定的 edge.input_nr
  meta->output_nr_ = edge.input_nr;

  // 对于视图（views），确保只有在必要时才覆盖这个新的 grad_fn_
  // 这个逻辑仅对自定义自动求导函数（custom autograd Functions）相关，其中在退出自定义函数前可能对给定张量进行多次操作。
  auto diff_view_meta = get_view_autograd_meta(self);
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    // 设置视图的版本号与 self 的版本号相同
    diff_view_meta->set_attr_version(self._version());
  }
}

// 返回自变量的梯度函数指针，如果不存在则返回 nullptr
Node* grad_fn_unsafe(const Variable& self) {
  if (get_autograd_meta(self)) {
    return get_autograd_meta(self)->grad_fn_.get();
  } else {
    return nullptr;
  }
}

// 设置自变量的版本计数器
void set_version_counter(
    const Variable& self,
    const c10::VariableVersion& version_counter) {
  // 检查自变量是否已定义
  TORCH_CHECK(
      self.defined(), "cannot call set_version_counter() on undefined tensor");
  // 设置自变量的版本计数器
  self.unsafeGetTensorImpl()->set_version_counter(version_counter);
}

// 增加自变量的版本号
void bump_version(const Variable& self) {
  // 检查自变量是否已定义
  TORCH_CHECK(self.defined(), "cannot call bump_version() on undefined tensor");
  // 增加自变量的版本号
  self.unsafeGetTensorImpl()->bump_version();
}

// 返回自变量的版本计数器的引用
const c10::VariableVersion& version_counter(const Variable& self) {
  // 检查自变量是否已定义
  TORCH_CHECK(
      self.defined(), "cannot call version_counter() on undefined tensor");
  // 返回自变量的版本计数器
  return self.unsafeGetTensorImpl()->version_counter();
}

// 添加钩子函数到自变量的自动求导元信息中
void add_hook(
    const at::TensorBase& self,
    std::unique_ptr<FunctionPreHook> hook) {
  // 获取自变量的自动求导元信息
  AutogradMeta* meta = materialize_autograd_meta(self);
  // 断言钩子列表为空
  TORCH_INTERNAL_ASSERT(meta->hooks_.empty());
  // 将钩子函数移动到钩子列表中
  meta->hooks_.push_back(std::move(hook));
}

// 返回自变量的钩子函数列表的引用
std::vector<std::unique_ptr<FunctionPreHook>>& hooks(const Variable& self) {
  // 断言自变量存在自动求导元信息
  TORCH_INTERNAL_ASSERT(get_autograd_meta(self));
  // 返回自变量的钩子函数列表
  return get_autograd_meta(self)->hooks_;
}

// 清空自变量的钩子函数列表
void clear_hooks(const at::TensorBase& self) {
  // 清空自变量的钩子函数列表
  materialize_autograd_meta(self)->hooks_.clear();
}

// 设置自变量的后积累梯度钩子函数
void set_post_acc_grad_hooks(
    const at::TensorBase& self,
    std::unique_ptr<PostAccumulateGradHook> dict) {
  // 获取自变量的自动求导元信息
  AutogradMeta* meta = materialize_autograd_meta(self);
  // 设置后积累梯度钩子函数
  meta->post_acc_grad_hooks_ = std::move(dict);
}

// 返回自变量的后积累梯度钩子函数的引用
std::unique_ptr<PostAccumulateGradHook>& post_acc_grad_hooks(
    const Variable& self) {
  // 断言自变量存在自动求导元信息
  TORCH_INTERNAL_ASSERT(get_autograd_meta(self));
  // 返回自变量的后积累梯度钩子函数
  return get_autograd_meta(self)->post_acc_grad_hooks_;
}

// 设置自变量的名称
void set_name(const Variable& self, const std::string& name) {
  // 设置自变量的名称属性
  materialize_autograd_meta(self)->name_ = name;
}
// 返回给定张量的自动微分元信息，如果张量未定义则返回 nullptr
AutogradMeta* get_autograd_meta(const at::TensorBase& self) {
  // 检查张量是否已定义，若未定义则抛出错误信息
  TORCH_CHECK(
      self.defined(), "cannot call get_autograd_meta() on undefined tensor");
  // 获取张量底层实现的自动微分元数据并强制转换为 AutogradMeta 指针返回
  return static_cast<AutogradMeta*>(
      self.unsafeGetTensorImpl()->autograd_meta());
}

// 返回视图张量的自动微分元信息，如果张量不是视图则返回 nullptr
DifferentiableViewMeta* get_view_autograd_meta(const at::TensorBase& self) {
  // 获取张量的自动微分元信息
  AutogradMeta* meta = get_autograd_meta(self);
  // 检查是否存在元信息且该张量为视图，如果是则返回不同的视图元信息指针，否则返回 nullptr
  if (meta && meta->is_view_) {
    return static_cast<DifferentiableViewMeta*>(meta);
  } else {
    return nullptr;
  }
}

} // namespace impl

using at::Tensor;

// 创建 VariableHooks 对象实例 variableHooks
VariableHooks variableHooks;
// 注册 VariableHooks 对象的实例，用于管理张量的钩子函数
at::impl::VariableHooksRegisterer registerVariableHooks(&variableHooks);

// 返回给定张量的变量数据，如果张量未定义则抛出错误信息
at::TensorBase VariableHooks::variable_data(const at::TensorBase& self) const {
  // 检查张量是否已定义，若未定义则抛出错误信息
  TORCH_CHECK(
      self.defined(), "cannot call variable_data() on undefined tensor");
  // 创建张量底层实现的浅拷贝并分离，不允许改变张量元数据
  auto self_impl_copy = self.unsafeGetTensorImpl()->shallow_copy_and_detach(
      /*version_counter=*/0,
      /*allow_tensor_metadata_change=*/false);
  // 清除浅拷贝的自动微分元信息
  self_impl_copy->set_autograd_meta(nullptr);
  // 返回新的张量对象，其底层实现为 self_impl_copy
  return at::Tensor(self_impl_copy);
}

// 返回给定张量的数据，如果张量未定义则抛出错误信息
at::TensorBase VariableHooks::tensor_data(const at::TensorBase& self) const {
  // 检查张量是否已定义，若未定义则抛出错误信息
  TORCH_CHECK(self.defined(), "cannot call tensor_data() on undefined tensor");
  // 创建张量底层实现的浅拷贝并分离，允许改变张量元数据
  auto self_impl_copy = self.unsafeGetTensorImpl()->shallow_copy_and_detach(
      /*version_counter=*/self.unsafeGetTensorImpl()->version_counter(),
      /*allow_tensor_metadata_change=*/
      self.unsafeGetTensorImpl()->allow_tensor_metadata_change());
  // 返回新的张量对象，其底层实现为 self_impl_copy
  return at::Tensor(self_impl_copy);
}

// 判断给定张量是否为叶子节点
bool VariableHooks::is_leaf(const at::TensorBase& self) const {
  // 获取张量的自动微分元信息
  if (impl::get_autograd_meta(self)) {
    // 检查自动微分元信息中的梯度函数指针是否为空，为空则视为叶子节点
    return impl::get_autograd_meta(self)->grad_fn_ == nullptr;
  } else {
    // 若不存在自动微分元信息，则视为叶子节点
    return true;
  }
}

// 返回给定张量的输出编号
int64_t VariableHooks::output_nr(const at::TensorBase& self) const {
  // 获取张量的自动微分元信息
  if (impl::get_autograd_meta(self)) {
    // 返回自动微分元信息中的输出编号
    return impl::get_autograd_meta(self)->output_nr_;
  } else {
    // 若不存在自动微分元信息，则默认输出编号为 0
    return 0;
  }
}

// 设置给定张量的数据
void VariableHooks::set_data(
    const at::TensorBase& self_base,
    const at::TensorBase& new_data) {
  // 获取 `self_base` 的可选引用并将其解引用为 `self` 引用的常量 Tensor
  at::OptionalTensorRef self_ref(self_base);
  const Tensor& self = *self_ref;

  // 获取 `new_data_base` 的可选引用并将其解引用为 `new_data` 引用的常量 Tensor
  at::OptionalTensorRef new_data_ref(new_data_base);
  const Tensor& new_data = *new_data_ref;

  // `var.set_data(new_data)` 浅复制 `new_data` 中所有非自动微分 TensorImpl 字段到 `var` 中。
  // 要求 `new_data` 和 `var` 具有兼容的张量类型。
  TORCH_CHECK(
      _has_compatible_shallow_copy_type(self, new_data),
      "Attempted to call `variable.set_data(tensor)`, but `variable` and `tensor` have incompatible tensor type.");

  // 如果 `self` 需要梯度，`new_data` 的数据类型必须是浮点数或复数类型。
  TORCH_CHECK(
      !self.requires_grad() ||
          isDifferentiableType(at::typeMetaToScalarType(new_data.dtype())),
      "data set to a tensor that requires gradients must be floating point or complex dtype");

  // 如果自动微分元数据已过期，则重置梯度累加器
  AutogradMeta* autograd_meta = impl::get_autograd_meta(self);
  if (autograd_meta) {
    // 使用互斥锁保护自动微分元数据的访问
    std::lock_guard<std::mutex> lock(autograd_meta->mutex_);
    auto prior_accumulator = autograd_meta->grad_accumulator_.lock();
    if (prior_accumulator) {
      // 获取先前梯度累加器的设备和新数据的设备
      const auto prior_device = prior_accumulator->input_metadata(0).device();
      const auto new_device = new_data.device();

      // 如果新数据与 `self` 的选项不兼容或设备不同，则重置梯度累加器
      if (!new_data.options().type_equal(self.options()) ||
          prior_device != new_device) {
        autograd_meta->grad_accumulator_.reset();
      }
    }
  }

  // 当使用 `set_data(...)` 替换 `Variable` 的张量数据时，版本计数器不共享。
  // 原始 `Variable` 的版本始终保留。详见 NOTE [ Version Counter Sharing ]。
  //
  // `var.set_data(new_data)` 总是忽略 `var` 的 `allow_tensor_metadata_change_`，
  // 因为用户需要此 API 来更改张量的元数据，而不考虑其 `allow_tensor_metadata_change_` 值，
  // 用户负责确保这是他们想要的行为。
  self.unsafeGetTensorImpl()->shallow_copy_from(new_data.getIntrusivePtr());
}

// 返回变量的数据作为 TensorBase 类型
at::TensorBase VariableHooks::data(const at::TensorBase& self) const {
  return self.variable_data();
}

// 返回变量的版本号作为 int64_t 类型
int64_t VariableHooks::_version(const at::TensorBase& self) const {
  return self.unsafeGetTensorImpl()->version_counter().current_version();
}

// 保留变量的梯度信息
void VariableHooks::retain_grad(const at::TensorBase& self) const {
  TORCH_CHECK(
      self.requires_grad(),
      "can't retain_grad on Tensor that has requires_grad=False");

  // 临时修复以改进 functorch 的用户体验
  const auto& functorch_tls = at::functorch::functorchTLSAccessor();
  if (functorch_tls) {
    functorch_tls->checkSupportsRetainGrad();
  }

  // 如果是叶子节点，则无操作
  if (self.is_leaf()) {
    return;
  }
  
  // 如果已经保留了梯度信息，则无操作
  if (impl::get_autograd_meta(self)->retains_grad_) {
    return;
  }
  
  // 创建弱引用指向自身的 TensorImpl
  c10::weak_intrusive_ptr<c10::TensorImpl> weak_self(self.getIntrusivePtr());

  // 定义保留梯度的钩子函数
  auto retain_grad_hook = [weak_self](const at::TensorBase& grad_base) {
    at::Tensor grad{grad_base};
    if (!weak_self.expired() && grad.defined()) {
      auto var = weak_self.lock();
      if (!var->grad().defined()) {
        if (grad.is_sparse()) {
          var->mutable_grad() = grad.clone();
        } else {
          var->mutable_grad() = grad.clone(at::MemoryFormat::Contiguous);
        }
      } else {
        var->mutable_grad() = var->grad() + grad;
      }
    }
    return at::TensorBase{};
  };

  // 获取梯度函数并添加保留梯度的钩子
  const auto& fn = self.grad_fn();
  fn->add_retains_grad_hook(
      std::make_unique<CppFunctionSingleTensorPreHook>(
          std::move(retain_grad_hook), self.output_nr()),
      self.output_nr());
  
  // 设置自动求导元信息中的保留梯度标志
  impl::get_autograd_meta(self)->retains_grad_ = true;
}

// 检查变量是否保留梯度
bool VariableHooks::retains_grad(const at::TensorBase& self) const {
  if (impl::get_autograd_meta(self)) {
    return impl::get_autograd_meta(self)->retains_grad_;
  } else {
    return false;
  }
}

// 执行反向传播
void VariableHooks::_backward(
    const Tensor& self,
    at::TensorList inputs,
    const std::optional<Tensor>& gradient,
    std::optional<bool> keep_graph,
    bool create_graph) const {
  // TODO torch::autograd::backward 应该直接接受 std::optional<Tensor> gradient，而不是我们在此解包成 Tensor _gradient
  Tensor _gradient = gradient.has_value() ? *gradient : Tensor();
  std::vector<torch::autograd::Variable> input_vars(
      inputs.begin(), inputs.end());
  torch::autograd::backward(
      {self}, {std::move(_gradient)}, keep_graph, create_graph, input_vars);
}

// 设置变量是否需要梯度
void VariableHooks::requires_grad_(
    const at::TensorBase& self,
    bool _requires_grad) const {
  if (!self.is_leaf() && !_requires_grad) {
    throw std::runtime_error(
        autograd::utils::requires_grad_leaf_error(_requires_grad));
  }
  self.set_requires_grad(_requires_grad);
}

// 检查变量是否是视图
bool VariableHooks::is_view(const at::TensorBase& self) const {
  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(self);
  if (diff_view_meta) {
    // 如果 diff_view_meta 指针有效，则调用其指向对象的 has_bw_view() 方法，并返回结果
    return diff_view_meta->has_bw_view();
    // 如果 diff_view_meta 指针无效，则返回 false
    } else {
        return false;
    }
}

const at::TensorBase& VariableHooks::base(const at::TensorBase& self) const {
  // 获取与给定张量关联的视图自动微分元信息
  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(self);
  // 如果存在视图自动微分元信息
  if (diff_view_meta) {
    // 检查是否存在反向视图
    TORCH_CHECK(
        diff_view_meta->has_bw_view(),
        "Can't get base of non-backward view Tensor");
    // 返回反向视图的基本张量引用
    return diff_view_meta->get_backward_view().base_;
  } else {
    // 如果不存在视图自动微分元信息，抛出运行时错误
    throw std::runtime_error("Can't get base of non-view Tensor");
  }
}

namespace {
std::string singleton_string;
}

const std::string& VariableHooks::name(const at::TensorBase& self) const {
  // 检查张量是否已定义
  TORCH_CHECK(
      self.defined(), "cannot call variable_data() on undefined tensor");
  // 如果存在自动微分元信息
  if (torch::autograd::impl::get_autograd_meta(self)) {
    // 返回自动微分元信息中的名称
    return torch::autograd::impl::get_autograd_meta(self)->name_;
  } else {
    // 否则返回单例字符串引用
    return singleton_string;
  }
}

namespace {
std::shared_ptr<torch::autograd::Node> singleton_shared_ptr;
}

const std::shared_ptr<torch::autograd::Node>& VariableHooks::grad_fn(
    const at::TensorBase& self) const {
  // 获取与给定张量关联的视图自动微分元信息
  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(self);
  // 如果存在视图自动微分元信息并且有反向视图
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    // 见 NOTE [ View + Inplace detection ]
    // 使用互斥锁锁定，以防止多线程访问
    std::lock_guard<std::mutex> lock(diff_view_meta->mutex_);
    // 获取反向视图信息
    auto& view_info = diff_view_meta->get_backward_view();
    // 如果不存在梯度函数并且基本张量不需要梯度
    if (!diff_view_meta->grad_fn_ && !view_info.base_.requires_grad()) {
      // 返回梯度函数
      return diff_view_meta->grad_fn_;
    }
    // 获取当前版本号
    auto current_version = self._version();
    // 保存旧的梯度函数
    auto old_fn = diff_view_meta->grad_fn_;
  }
  // 如果存在自动微分元信息，返回其中的梯度函数
  if (torch::autograd::impl::get_autograd_meta(self)) {
    return torch::autograd::impl::get_autograd_meta(self)->grad_fn_;
  } else {
    // 否则返回单例共享指针
    return singleton_shared_ptr;
  }
}

void VariableHooks::remove_hook(const at::TensorBase& self, unsigned pos)
    const {
  // 获取张量关联的自动微分元信息并实例化钩子列表
  auto& list =
      torch::autograd::impl::materialize_autograd_meta(self)->cpp_hooks_list_;
  // 检查钩子列表是否存在，并且位置是否有效
  TORCH_CHECK(
      list && pos < list->size(), "Invalid index, no hook at position ", pos);
  // 将指定位置的钩子设置为 nullptr，从而忽略该钩子
  // Hook will be ignored
  (*list)[pos] = nullptr;
}

unsigned VariableHooks::_register_hook(
    const at::TensorBase& self,
    std::function<at::TensorBase(const at::TensorBase&)> hook) const {
  // 检查张量是否需要梯度
  TORCH_CHECK(
      self.requires_grad(),
      "cannot register a hook on a variable that "
      "doesn't require gradient");
  // 获取自动微分元信息并创建钩子列表（如果不存在的话）
  auto& list = torch::autograd::impl::get_autograd_meta(self)->cpp_hooks_list_;
  if (!list) {
    // 创建 C++ 钩子
    torch::autograd::impl::create_cpp_hook(
        self, /*is_retains_grad_hooks=*/false);
  }
  // 获取当前钩子列表的大小作为索引
  unsigned idx = list->size();
  // 将新的钩子函数添加到列表末尾
  list->push_back(hook);
  return idx;
}

void handle_view_on_rebase(
    DifferentiableViewMeta* diff_view_meta,
    bool indirect) {
  /// See NOTE [ View + Inplace detection ] for justification of the logic below
  // 获取视图的创建元信息
  auto creation_meta = diff_view_meta->get_creation_meta();
  // 如果创建元信息不是默认的情况
  if (creation_meta != CreationMeta::DEFAULT) {
    // 获取视图的梯度函数指针
    auto grad_fn = diff_view_meta->grad_fn_.get();
    std::string msg;
    std::string modified_obj;
    // 创建错误消息的标题。
    // 根据是否间接修改设置修改对象的描述。
    if (indirect) {
      modified_obj = "its base or another view of its base has been";
    } else {
      modified_obj = "is being";
    }

    // 根据创建元数据的不同情况设置前缀。
    if (creation_meta == CreationMeta::INFERENCE_MODE ||
        creation_meta == CreationMeta::NO_GRAD_MODE || !grad_fn) {
      std::string prefix;
      // 如果存在梯度函数，设置包含输出编号和梯度函数名称的前缀。
      if (grad_fn) {
        prefix = c10::str(
            "Output ",
            diff_view_meta->output_nr_,
            " of ",
            grad_fn->name(),
            " is a view of a view which was created in");
      } else {
        prefix = "A view was created in";
      }
      // 根据创建元数据不同情况设置消息内容。
      if (creation_meta == CreationMeta::INFERENCE_MODE) {
        msg = c10::str(
            prefix,
            " inference mode and ",
            modified_obj,
            " modified inplace in normal mode.");
      } else {
        // 创建元数据不一定是 CreationMeta::NO_GRAD_MODE
        // 例如 CreationMeta::IN_CUSTOM_FUNCTION 也是可能的，但是如果没有 grad_fn，说明视图是在无梯度模式下创建的。
        msg = c10::str(
            prefix,
            " no_grad mode and ",
            modified_obj,
            " modified inplace with grad mode enabled.");
      }
    } else {
      // 默认情况下设置消息内容。
      msg = c10::str(
          "Output ",
          diff_view_meta->output_nr_,
          " of ",
          grad_fn->name(),
          " is a view and ",
          modified_obj,
          " modified inplace.");
    }

    // 根据创建元数据为 MULTI_OUTPUT_NODE 的情况设置消息内容。
    if (creation_meta == CreationMeta::MULTI_OUTPUT_NODE) {
      msg = c10::str(
          msg,
          " This view is the output of a function that returns multiple views. Such functions do not"
          " allow the output views to be modified inplace. You should replace the inplace operation by an"
          " out-of-place one.");
    } else if (creation_meta == CreationMeta::NO_GRAD_MODE) {
      // 根据创建元数据为 NO_GRAD_MODE 的情况设置消息内容。
      msg = c10::str(
          msg,
          " Given that this use case is ambiguous and error-prone, it is forbidden."
          " You can clarify your code by moving both the view and the inplace either both"
          " inside the no_grad block (if you don't want the inplace to be tracked) or both outside (if you want"
          " the inplace to be tracked).");
    } else if (creation_meta == CreationMeta::INFERENCE_MODE) {
      // 根据创建元数据为 INFERENCE_MODE 的情况设置消息内容。
      msg = c10::str(
          msg,
          " Given that this use case is ambiguous and error-prone, it is forbidden."
          " You can clarify your code by moving both the view and the inplace either both"
          " inside the inference_mode block (if you don't want the inplace to be tracked) or both outside (if you want"
          " the inplace to be tracked).");
    // 如果创建元信息为 IN_CUSTOM_FUNCTION
    } else if (creation_meta == CreationMeta::IN_CUSTOM_FUNCTION) {
      // 拼接错误信息，说明视图是在自定义函数内部创建的，或者因为某个输入被原样返回，
      // 而 autograd 逻辑处理视图+就地操作将会覆盖与自定义函数相关联的自定义反向传播，
      // 导致梯度不正确。这种行为是禁止的。您可以通过克隆自定义函数的输出来修复此问题。
      msg = c10::str(
          msg,
          " This view was created inside a custom Function (or because an input was returned as-is) and the"
          " autograd logic to handle view+inplace would override the custom backward associated with the custom"
          " Function, leading to incorrect gradients. This behavior is forbidden. You can fix this by"
          " cloning the output of the custom Function.");
    } else {
      // 如果不是以上任何情况，则抛出内部断言错误，说明创建元信息状态无效
      TORCH_INTERNAL_ASSERT(false, "Invalid CreationMeta state");
    }

    // 对条件进行检查，如果为 false，则抛出错误信息 msg
    TORCH_CHECK(false, msg);
  }
}

// 获取 ChainedViewFunc 中第一个视图函数的符号整数列表
std::vector<c10::SymInt> ChainedViewFunc::get_symints() const {
  // 调用第一个视图函数的 get_symints 方法获取符号整数列表
  auto symints = first->get_symints();
  // 调用第二个视图函数的 get_symints 方法获取符号整数列表
  auto second_symints = second->get_symints();
  // 扩展 symints 向量的容量以容纳第二个视图函数的符号整数
  symints.reserve(symints.size() + second_symints.size());
  // 将 second_symints 向量中的符号整数移动到 symints 向量的末尾
  symints.insert(
      symints.end(),
      std::make_move_iterator(second_symints.begin()),
      std::make_move_iterator(second_symints.end()));
  // 返回包含两个视图函数所有符号整数的向量
  return symints;
}

// 获取 ChainedViewFunc 中第一个视图函数的张量列表
std::vector<at::Tensor> ChainedViewFunc::get_tensors() const {
  // 调用第一个视图函数的 get_tensors 方法获取张量列表
  auto tensors = first->get_tensors();
  // 调用第二个视图函数的 get_tensors 方法获取张量列表
  auto second_tensors = second->get_tensors();
  // 扩展 tensors 向量的容量以容纳第二个视图函数的张量
  tensors.reserve(tensors.size() + second_tensors.size());
  // 将 second_tensors 向量中的张量移动到 tensors 向量的末尾
  tensors.insert(
      tensors.end(),
      std::make_move_iterator(second_tensors.begin()),
      std::make_move_iterator(second_tensors.end()));
  // 返回包含两个视图函数所有张量的向量
  return tensors;
}

// ChainedViewFunc 的函数调用运算符重载，按照第一个和第二个视图函数的顺序对输入进行处理
at::Tensor ChainedViewFunc::operator()(const at::Tensor& input_base) const {
  // 先调用第一个视图函数，再调用第二个视图函数，处理输入张量
  return (*second)((*first)(input_base));
}

// 克隆 ChainedViewFunc 对象并设置符号整数和张量
std::unique_ptr<ViewFunc> ChainedViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  // 分别存储第一个和第二个视图函数的符号整数
  std::optional<std::vector<c10::SymInt>> first_symints;
  std::optional<std::vector<c10::SymInt>> second_symints;
  // 如果有符号整数值传入
  if (symints.has_value()) {
    // 断言传入的符号整数数量等于当前 ChainedViewFunc 的符号整数数量
    TORCH_INTERNAL_ASSERT(symints->size() == num_symints());
    // 拷贝第一个视图函数的符号整数
    first_symints = std::vector<c10::SymInt>(
        symints->begin(), symints->begin() + first->num_symints());
    // 拷贝第二个视图函数的符号整数
    second_symints = std::vector<c10::SymInt>(
        symints->begin() + first->num_symints(), symints->end());
  }

  // 分别存储第一个和第二个视图函数的张量
  std::optional<std::vector<at::Tensor>> first_tensors;
  std::optional<std::vector<at::Tensor>> second_tensors;
  // 如果有张量值传入
  if (tensors.has_value()) {
    // 断言传入的张量数量等于当前 ChainedViewFunc 的张量数量
    TORCH_INTERNAL_ASSERT(tensors->size() == num_tensors());
    // 拷贝第一个视图函数的张量
    first_tensors = std::vector<at::Tensor>(
        tensors->begin(), tensors->begin() + first->num_tensors());
    // 拷贝第二个视图函数的张量
    second_tensors = std::vector<at::Tensor>(
        tensors->begin() + first->num_tensors(), tensors->end());
  }

  // 创建一个新的 ChainedViewFunc 对象，设置第一个和第二个视图函数的符号整数和张量
  return std::make_unique<ChainedViewFunc>(
      first->clone_and_set(first_symints, first_tensors),
      second->clone_and_set(second_symints, second_tensors));
}

// 命名空间 autograd 结束
} // namespace autograd
// 命名空间 torch 结束
} // namespace torch
```