# `.\pytorch\aten\src\ATen\FunctionalStorageImpl.h`

```
#pragma once

#include <ATen/Tensor.h>

namespace at::functionalization {

// See Note [Functionalization Pass In Core]

// ViewMeta is a class used by the functionalization pass to navigate between
// a base tensor and a view tensor.
// For example, if I call `b = a.view1(...)`
// the functionalization pass will generate and store a ViewMeta on b that looks
// like:
//
// ViewMeta(
//   [<captures>](const Tensor& base, int64_t mutated_view_idx) {
//     return base.view1(...);
//   },
//   [<captures>](const at::Tensor& base, const at::Tensor& mutated_view,
//   int64_t mutated_view_idx) -> at::Tensor {
//     return at::functionalization::impl::view1_inverse(base, mutated_view,
//     ...);
//   }
//
// The forward_fn lambda describes how to replay view1 on a tensor.
//
// The reverse_fn lambda describes how, given a tensor that is already a view,
// how to get the corresponding base tensor. See Note [Functionalization Pass:
// View Inverses] for details.
struct ViewMeta {
  // Constructor to initialize a ViewMeta object.
  ViewMeta(
      std::function<Tensor(const Tensor&, int64_t)> forward,
      std::function<Tensor(const Tensor&, const Tensor&, int64_t)> reverse,
      bool has_symbolic_inputs,
      bool is_multi_output = false,
      bool is_as_strided = false,
      int64_t out_idx = 0)
      : forward_fn(std::move(forward)),  // Stores the forward replay function
        reverse_fn(std::move(reverse)),  // Stores the reverse function
        out_index(out_idx),              // Stores the output index
        is_multi_output(is_multi_output),  // Indicates if this is a multi-output view
        is_as_strided(is_as_strided),    // Indicates if this is an as_strided view
        has_symbolic_inputs(has_symbolic_inputs) {}  // Indicates if the view operation has symbolic inputs

  // Lambda function to replay the forward view operation on a tensor.
  std::function<Tensor(const Tensor&, int64_t)> forward_fn;

  // Lambda function to get the base tensor from a mutated view tensor.
  std::function<Tensor(const Tensor&, const Tensor&, int64_t)> reverse_fn;

  // See Note [out_idx in ViewMeta]
  int64_t out_index;  // Stores the output index

  // Indicates if this is a multi-output view
  bool is_multi_output;

  // Indicates if this is an as_strided view
  bool is_as_strided;

  // Indicates if the view operation has any symbolic inputs
  bool has_symbolic_inputs;

  // Returns a copy of the current ViewMeta, if out_idx matches the current
  // out_index. Otherwise, returns a new ViewMeta with the same forward/reverse
  // functions, but a new out index.
  ViewMeta to_out_idx(int64_t out_idx);  // Method to return a new ViewMeta object with a different output index
};

// FunctionalStorageImpl is a subclass of StorageImpl used by the
// functionalization pass. It has no underlying data (similar to meta storage).
// It also knows how to reflect mutations to tensors in the absence of a valid
// data pointer.
//
// A storage represents the state shared by (potentially multiple) views of the
// same tensor. For example, in the following code:
//
// b = a.view1(...)
// c = b.view2(...)
// b.add_(1)
// --> storage.add_update(b, {view1_meta})
//
// The call to add_(1) will result in a call to alias.add_update(b,
// {view1_meta}), queueing up the mutation from b onto the alias. Later, suppose
// c is used in an expression (e.g. you try to print c, or pass it to an
// operator). Doing so will involve "syncing" c. First we apply any pending
// updates to the alias, and then we regenerate c by replaying its views off of
// the updated alias. E.g:
//
// print(str(c))
// --> c.sync_()
//     --> alias.apply_updates() // after this, the alias will be updated to
//     reflect the mutation to b
// FunctionalStorageImpl 结构体，继承自 c10::StorageImpl，用于实现可变存储的功能
struct TORCH_API FunctionalStorageImpl : public c10::StorageImpl {
 public:
  // Update 结构体，表示更新操作的数据结构
  struct Update {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    // 更新后的张量
    const at::Tensor new_val;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    // 视图元数据的向量
    const std::vector<ViewMeta> view_metas;
  };

  // 使用给定张量初始化 FunctionalStorageImpl 对象
  explicit FunctionalStorageImpl(const Tensor& value);

  // 添加更新操作
  void add_update(
      const Tensor& updated_val,
      const std::vector<ViewMeta>& view_metas);
  
  // 应用所有更新操作并返回是否成功
  bool apply_updates();

  // 返回基础张量的引用
  const Tensor& base() {
    return base_;
  }

  // 返回代数的大小
  size_t generation() const {
    return generation_;
  }

  // 冻结对象，标记为不可变
  void freeze() {
    frozen_ = true;
  }

  // 获取存储大小，根据参数决定返回原始或当前的存储大小
  c10::SymInt get_storage_size(bool before) {
    if (before) {
      return original_storage_size_;
    } else {
      return curr_storage_size_;
    }
  }

  // 默认析构函数
  ~FunctionalStorageImpl() override = default;

  // 标记发生了突变
  void mark_mutation() {
    mutation_counter_++;
  }

  // 在无梯度或推断模式下标记发生了突变
  void mark_mutation_during_no_grad_or_inference_mode() {
    mutation_counter_during_no_grad_or_inference_mode_++;
  }

  // 标记从自动求导隐藏的突变
  void mark_mutation_hidden_from_autograd() {
    mutation_counter_hidden_from_autograd_++;
  }

  // 检查所有突变是否都在无梯度或推断模式下
  bool are_all_mutations_under_no_grad_or_inference_mode() const {
    auto non_autograd_mutations =
        mutation_counter_during_no_grad_or_inference_mode_ +
        mutation_counter_hidden_from_autograd_;
    // <= 是因为两个计数器都会被增加，如果在无梯度下执行了变异操作
    return mutation_counter_ <= non_autograd_mutations;
  }

  // 检查所有突变是否都从自动求导隐藏
  bool are_all_mutations_hidden_from_autograd() const {
    // 在无梯度/推断模式下的突变技术上不会对自动求导隐藏
    return mutation_counter_ <= mutation_counter_hidden_from_autograd_;
  }

  // 标记诱导器存储调整的变化
  void mark_inductor_storage_resize(c10::SymInt new_size) {
    inductor_storage_resized_ = true;
    curr_storage_size_ = new_size;
  }

  // 检查诱导器存储是否已调整大小
  bool was_inductor_storage_resized() {
    // 返回 inductor_storage_resized_ 的当前值
      return inductor_storage_resized_;
    }
    
    private:
    // NB: base_ 应始终指向当前功能化层下方的张量。
    // 这主要是为了避免引用循环。例如，对于 `b = a.view(...)`，a 和 b 都是 FunctionalTensorWrapper，
    // base_ 应指向 a 的值而不是 a 本身，因为 a.storage_ 和 b.storage_ 都包含一个 FunctionStorageImpl，
    // 其中包含一个 Tensor `base_`。参见注释 [Functionalization: Walualias Removal] 中的图表进行直观了解。
    at::Tensor base_;
    
    // 更新操作的列表
    std::vector<Update> updates_;
    
    // generation_ 每当将变异排队到别名时会递增。
    // 它用于确定张量是否“最新”，或者是否需要从别名重新生成。
    size_t generation_ = 0;
    
    // 如果被冻结，则此存储上不允许进一步的变异。
    // 一旦被冻结，存储就不能解冻。
    bool frozen_ = false;
    
    // 这些变异计数器在 FunctionalTensorWrapper 经历变异时会增加。
    // 当变异处于 no_grad 状态下，或来自 triton 内核时，还会增加相应的 during_no_grad 或 hidden_from_autograd 计数器。
    // 为什么需要单独检测这两种情况，而不是普通的输入变异？
    // (1) 普通的输入变异可能会变异 autograd 元数据，如 .grad_fn，
    //     在这种情况下，需要在编译图之外重播它们。
    // (2) no_grad 输入变异通常可以安全地保留在图中（并编译），
    //     但它们会增加张量的版本计数，因此需要在 torch.compile 中标记输入为 dirty。
    // (3) 完全隐藏于 autograd 的变异（例如来自 triton 内核）不会变异任何 autograd 状态，可以完全保留在图中。
    uint64_t mutation_counter_during_no_grad_or_inference_mode_ = 0;
    uint64_t mutation_counter_ = 0;
    uint64_t mutation_counter_hidden_from_autograd_ = 0;
    
    // 用于判断：
    // (1) 图输入是否发生了任何存储大小调整
    // (2) 原始/当前存储大小告诉我们这些调整是否导致了无操作
    bool inductor_storage_resized_ = false;
    c10::SymInt original_storage_size_;
    c10::SymInt curr_storage_size_;
};

} // namespace at::functionalization
```