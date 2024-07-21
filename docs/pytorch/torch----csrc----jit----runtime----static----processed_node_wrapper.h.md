# `.\pytorch\torch\csrc\jit\runtime\static\processed_node_wrapper.h`

```
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/jit/runtime/static/impl.h>

// 定义命名空间 torch::jit
namespace torch::jit {

// 下面的类通过 CRTP 实现 ProcessedNodeInputWrapper 和 ProcessedNodeOutputWrapper 的代码重用
template <typename DerivedWrapper>
class ProcessedNodeWrapperBase {
 public:
  // 内部迭代器类 ProcessedNodeWrapperBaseIter
  class ProcessedNodeWrapperBaseIter {
   public:
    // 迭代器类型定义
    using iterator_category = std::forward_iterator_tag;  // 迭代器类型标签为前向迭代器
    using value_type = at::Tensor;  // 迭代器所指对象类型为 at::Tensor
    using difference_type = size_t;  // 迭代器之间的距离类型为 size_t
    using pointer = const at::Tensor*;  // 指针类型为指向 const at::Tensor 的指针
    using reference = const at::Tensor&;  // 引用类型为 const at::Tensor 的引用

    // 默认构造函数
    ProcessedNodeWrapperBaseIter() = default;

    // 初始化构造函数，接受容器指针和起始索引
    ProcessedNodeWrapperBaseIter(
        const DerivedWrapper* container,
        size_t start_idx)
        : container_(container), idx_(start_idx) {}

    // 前缀递增运算符重载
    ProcessedNodeWrapperBaseIter& operator++() {
      TORCH_DCHECK_NE(idx_, container_->size());  // 使用 TORCH_DCHECK_NE 进行索引边界检查
      ++idx_;
      return *this;
    }

    // 后缀递增运算符重载
    ProcessedNodeWrapperBaseIter operator++(int) {
      ProcessedNodeWrapperBaseIter old = *this;
      ++(*this);
      return old;
    }

    // 解引用运算符重载，返回当前迭代器位置的元素引用
    reference operator*() const {
      TORCH_CHECK(container_ != nullptr);  // 使用 TORCH_CHECK 检查容器非空
      return (*container_)[idx_];
    }

    // 成员访问运算符重载，返回指向当前迭代器位置的指针
    pointer operator->() const {
      TORCH_CHECK(container_ != nullptr);  // 使用 TORCH_CHECK 检查容器非空
      return &(*container_)[idx_];
    }

    // 相等运算符重载，比较两个迭代器是否相等
    friend bool operator==(
        ProcessedNodeWrapperBaseIter lhs,
        ProcessedNodeWrapperBaseIter rhs) {
      TORCH_DCHECK_EQ(lhs.container_, rhs.container_);  // 使用 TORCH_DCHECK_EQ 检查容器相等
      return lhs.idx_ == rhs.idx_;
    }

    // 不等运算符重载，比较两个迭代器是否不相等
    friend bool operator!=(
        ProcessedNodeWrapperBaseIter lhs,
        ProcessedNodeWrapperBaseIter rhs) {
      return !(lhs == rhs);
    }

   private:
    const DerivedWrapper* container_ = nullptr;  // 容器指针
    size_t idx_ = 0;  // 当前索引
  };

  // 定义迭代器类型别名，都是 const 版本，模仿 at::ArrayRef 的行为
  using iterator = ProcessedNodeWrapperBaseIter;
  using const_iterator = ProcessedNodeWrapperBaseIter;
  using size_type = size_t;  // 大小类型为 size_t
  using value_type = at::Tensor;  // 值类型为 at::Tensor

  // 构造函数，接受一个 ProcessedNode 对象的引用
  explicit ProcessedNodeWrapperBase(ProcessedNode& pnode) : pnode_(pnode) {}

  // 返回首迭代器，从容器的开头开始
  iterator begin() {
    return ProcessedNodeWrapperBaseIter(static_cast<DerivedWrapper*>(this), 0);
  }

  // 返回尾迭代器，容器大小处的迭代器
  iterator end() {
    return ProcessedNodeWrapperBaseIter(
        static_cast<DerivedWrapper*>(this),
        static_cast<DerivedWrapper*>(this)->size());
  }

  // 返回首迭代器，const 版本
  const_iterator begin() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this), 0);
  }

  // 返回尾迭代器，const 版本
  const_iterator end() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this),
        static_cast<const DerivedWrapper*>(this)->size());
  }

  // 返回首迭代器，const 版本
  const_iterator cbegin() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this), 0);
  }

  // 返回尾迭代器，const 版本
  const_iterator cend() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this),
        static_cast<const DerivedWrapper*>(this)->size());
  }

  // 判断容器是否为空
  bool empty() const {
    return static_cast<const DerivedWrapper*>(this)->size() == 0;
  }



# 返回一个布尔值，判断 DerivedWrapper 派生类对象的 size 是否为 0
    return static_cast<const DerivedWrapper*>(this)->size() == 0;



 protected:
  ProcessedNode& pnode_;



# 保护访问修饰符下的成员变量 pnode_，它是一个引用，指向 ProcessedNode 对象
  ProcessedNode& pnode_;
};

// ProcessedNodeWrapperBase允许我们直接在期望IValues容器的上下文中使用ProcessedNode。
// 这种技巧对于避免性能敏感的本地操作中的引用计数增加非常有用。
// 例如，假设我们有一个操作，它以张量列表作为参数，并且我们已经将该操作转换为静态运行时的可变参数变体。
// 要使用PyTorch库的操作实现，我们必须将可变参数打包成列表：
//   std::vector<Tensor> tensor_list;
//   tensor_list.reserve(pnode->num_outputs());
//   for (const auto i : c10::irange(pnode->num_inputs())
//     tensor_list.push_back(pnode->Input(i).toTensor());
//   op_impl(tensor_list);
// 使用ProcessedNodeWrapperBase，我们可以避免这一轮引用计数增加。
// 我们只需要将`op_impl`转换为模板，并传递ProcessedNodeInputWrapper(*pnode)!
class ProcessedNodeInputWrapper
    : public ProcessedNodeWrapperBase<ProcessedNodeInputWrapper> {
 public:
  // 最后`back_elements_ignored`个元素不被考虑。
  // 对于前`front_elements_ignored`个元素也是如此。
  // 这对于只有前N个元素是张量的操作非常有用（N < inputs.size()）。
  // 例如，VarStack的最后一个参数是整数维度。
  explicit ProcessedNodeInputWrapper(
      ProcessedNode& pnode,
      size_t front_elements_ignored = 0,
      size_t back_elements_ignored = 1)
      : ProcessedNodeWrapperBase<ProcessedNodeInputWrapper>(pnode),
        front_elements_ignored_(front_elements_ignored),
        back_elements_ignored_(back_elements_ignored) {
    TORCH_CHECK(front_elements_ignored_ <= pnode_.num_inputs());
    TORCH_CHECK(
        back_elements_ignored_ <=
        pnode_.num_inputs() - front_elements_ignored_);
  }

  // 返回有效元素的数量
  size_t size() const {
    return pnode_.num_inputs() - back_elements_ignored_ -
        front_elements_ignored_;
  }

  // 访问指定索引处的元素
  const at::Tensor& operator[](size_t idx) const {
    TORCH_CHECK(idx < size());
    return pnode_.Input(front_elements_ignored_ + idx).toTensor();
  }

  // 返回第一个元素
  const at::Tensor& front() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access front() of empty ProcessedNodeInputWrapper");
    return pnode_.Input(front_elements_ignored_).toTensor();
  }

  // 返回最后一个元素
  const at::Tensor& back() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access back() of empty ProcessedNodeInputWrapper");
    return pnode_.Input(pnode_.num_inputs() - back_elements_ignored_ - 1)
        .toTensor();
  }

 private:
  size_t front_elements_ignored_;  // 忽略的前部元素数目
  size_t back_elements_ignored_;   // 忽略的后部元素数目
};

// 类似于ProcessedNodeInputWrapper，但包装输出并允许写入。
class ProcessedNodeOutputWrapper
    : public ProcessedNodeWrapperBase<ProcessedNodeOutputWrapper> {
 public:
  using ProcessedNodeWrapperBase<
      ProcessedNodeOutputWrapper>::ProcessedNodeWrapperBase;

  // 返回有效元素的数量
  size_t size() const {
    // 返回处理节点输出的数量
    return pnode_.num_outputs();
  }

  // 重载索引运算符，返回给定索引处的处理节点输出张量引用
  at::Tensor& operator[](size_t idx) const {
    // 检查索引是否有效
    TORCH_CHECK(idx < size());
    // 返回索引对应的处理节点输出张量
    return pnode_.Output(idx).toTensor();
  }

  // 返回第一个处理节点输出张量的引用
  at::Tensor& front() const {
    // 检查是否处理节点输出为空
    TORCH_CHECK(
        !empty(),
        "Attempted to access front() of empty ProcessedNodeOutputWrapper");
    // 返回第一个处理节点输出张量的引用
    return pnode_.Output(0).toTensor();
  }

  // 返回最后一个处理节点输出张量的引用
  at::Tensor& back() const {
    // 检查是否处理节点输出为空
    TORCH_CHECK(
        !empty(),
        "Attempted to access back() of empty ProcessedNodeOutputWrapper");
    // 返回最后一个处理节点输出张量的引用
    return pnode_.Output(size() - 1).toTensor();
  }
};

} // namespace torch::jit
```