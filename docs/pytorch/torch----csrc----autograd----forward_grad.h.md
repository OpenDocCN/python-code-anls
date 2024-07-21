# `.\pytorch\torch\csrc\autograd\forward_grad.h`

```py
#pragma once

#include <ATen/core/Tensor.h>
#include <unordered_set>

namespace torch::autograd {

// [ Using ForwardGrad ]
// ForwardGrad needs to be a shared_ptr to satisfy constraints of its inner
// design. But this shared_ptr must be uniquely associated with the object that
// stores it (as of writing, either AutogradMeta or SavedVariable). This object
// is called the "owning object" in the discussions below. This owning object
// must call `ForwardGrad::clear()` when it is destroyed to ensure that the
// ForwardGrad is properly de-allocated.

struct ForwardGrad;

// This file contains two classes that are used to store forward AD gradients
// and ensure that they are scoped properly. Because forward AD runs
// concurrently with the evaluation of the function, we need a mechanism to
// separate different forward AD invocations and be able to compute the right
// gradients. We model such invocations as levels here. The particular scoping
// issue mentioned above has two main drivers:
//   - Ensure that we can conveniently use forward AD within a high level API
//   without
//     leaking the forward AD states outside.
//   - Ensure that we can keep the level that we expose to the user API simple
//   (an integer
//     that represents the nesting depth) while avoiding confusions when the
//     level index is re-used.

// The important external APIs from this file are:
//   - ForwardADLevel::get_next_idx() that can be used to enter a new level and
//   get its index
//   - ForwardADLevel::release_idx() that can be used to exit a given level.
//   - ForwardGrad() can be used to store a given forward gradient that will
//   handle the level
//     tracking automatically.

// The basic implementation strategy is as follows:
// Every tensor has a ForwardGrad, maintaining a map from levels to tangents.
// ForwardGrad is responsible for registering itself to the appropriate
// ForwardADLevel when a new tangent is added to it via ForwardGrad::set_value
// and to un-register itself from this same level if that tangent is removed via
// ForwardGrad::reset. The ForwardADLevel is created when a new level is entered
// via ForwardADLevel::get_next_idx. A reference to the new ForwardADLevel is
// stored into a global (for the whole process) vector that ensure it can be
// accessed via ForwardADLevel::get_by_idx. This reference is deleted when the
// index is released by the user when calling ForwardADLevel::release_idx. When
// it is destructed, the ForwardADLevel is responsible for clearing all the
// tangents for its level stored in all the ForwardGrad that registered with it.
//
// This process-wide level design, compared to a thread local one, allows us to
// use very simple user facing handle for the level (an int) while enabling
// cross-thread forward AD. The only required synchronization for the user is
// when entering and exiting the levels. Some discussion on alternative design
// is in https://github.com/pytorch/pytorch/pull/49097#discussion_r543716453 and
// can be refined in the future.
// 参见 https://github.com/pytorch/pytorch/pull/49097#discussion_r543716453，未来可能会进行优化。

// Correctness of concurrency:
// Each class uses its own lock when reading or modifying internal storages.
// This allows in particular to safely remove tangents from ForwardGrad when the
// ForwardADLevel is being exited. We ensure no deadlock by ensuring that a
// methods never calls into another class's method while the local class's lock
// is held except in one single case: calling from ForwardADLevel's destructor
// into ForwardGrad::reset with update_level=false.
// 并发正确性：
// 每个类在读取或修改内部存储时都使用自己的锁。
// 这确保了在退出 ForwardADLevel 时从 ForwardGrad 安全地移除切线。我们通过确保一个方法在持有本地类锁时永远不会调用另一个类的方法来确保没有死锁，除了一种情况：从 ForwardADLevel 的析构函数调用 ForwardGrad::reset（update_level=false）。

// The lifetime of these objects is as follows:
// The ForwardADLevel can be in three states:
//      - Initialized: where one of its reference is held by the global vector
//      and there may be more
//        references held by temporary variables in ForwardGrad's methods.
//      - About to be destructed: where "release_idx" has been called and the
//      only reason for the
//        ForwardADLevel not to be destructed right away is that some methods in
//        ForwardGrad have owning reference to it. This is done so that a
//        ForwardADLevel can never be destructed when a ForwardGrad is
//        registered with it and in the process of adding something to its
//        internal state.
//      - Being destructed: Here the ForwardADLevel is not referenced anymore
//      and can be safely reset
//        all of the ForwardGrad. Note that we can have more than one reset
//        being called here (which is ok) but we are guaranteed that there is at
//        least one.
// The ForwardGrad is simpler as there is no intermediary state and no special
// destructor for. The logic to unregister it from the different ForwardADLevel
// is done when the owning object (AutogradMeta or SavedVariable) is being
// destroyed.
// 这些对象的生命周期如下：
// ForwardADLevel 可以处于三种状态：
//      - 初始化状态：其中全局向量持有其一个引用，ForwardGrad 方法的临时变量可能还持有更多引用。
//      - 准备销毁状态：调用了 "release_idx"，但 ForwardADLevel 不会立即被销毁的唯一原因是 ForwardGrad 的某些方法持有对其的引用。这样做是为了确保在 ForwardGrad 注册并在向其内部状态添加内容时，不会销毁 ForwardADLevel。
//      - 正在销毁状态：在此状态下，ForwardADLevel 不再被引用，可以安全地重置所有 ForwardGrad。请注意，可能会在此处调用多个重置操作（这是可以的），但我们保证至少会有一个。
// ForwardGrad 更简单，因为没有中间状态，也没有特殊的析构函数。取消注册它与不同 ForwardADLevel 的逻辑是在拥有它的对象（AutogradMeta 或 SavedVariable）被销毁时完成的。

// Other considered design:
// To avoid having the ForwardGrad::clear, we considered storing weak_ptr inside
// the ForwardADLevel. While this would work, it would mean that the set inside
// the ForwardADLevel would only grow unless we do an expensive linear scan to
// remove all the dangling weak pointers. Hence this approach was not used.
// 其他考虑的设计：
// 为了避免使用 ForwardGrad::clear，我们考虑在 ForwardADLevel 内部存储 weak_ptr。虽然这样做是可行的，但这意味着除非我们执行昂贵的线性扫描来移除所有悬空的 weak 指针，否则 ForwardADLevel 内部的集合只会增长。因此，没有采用这种方法。

// Data structures in this file are optimized for this maximum number of levels.
// The number of levels corresponds to the degree of the gradient being
// computed using forward AD and we don't expect more than second order
// gradients to be common.
#define EXPECTED_MAX_LEVEL 2
// 此文件中的数据结构针对最大级别进行了优化。
// 级别的数量对应使用前向自动微分计算的梯度的阶数，我们不期望常见的梯度超过二阶。

struct TORCH_API ForwardADLevel {
  ForwardADLevel(uint64_t idx) : idx_(idx) {}
  ~ForwardADLevel();

  static uint64_t get_next_idx();
  static void release_idx(uint64_t idx);
  static std::shared_ptr<ForwardADLevel> get_by_idx(uint64_t idx);
  static std::shared_ptr<ForwardADLevel> try_get_by_idx(uint64_t idx);

  void erase(const std::shared_ptr<ForwardGrad>& grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    grads_.erase(grad);
  }

  void insert(const std::shared_ptr<ForwardGrad>& grad) {
    // 使用 std::lock_guard 对 mutex_ 进行自动加锁，确保互斥访问
    std::lock_guard<std::mutex> lock(mutex_);
    // 将 grad 插入到 grads_ 中，利用 unordered_set 确保唯一性
    grads_.insert(grad);
  }

 private:
  // 存储共享指针 ForwardGrad 的无序集合
  std::unordered_set<std::shared_ptr<ForwardGrad>> grads_;
  // 线程互斥量，用于保护 grads_ 的并发访问
  std::mutex mutex_;
  // 索引号，用于标识 grads_ 中元素的顺序或唯一性
  uint64_t idx_;
};

// 结构体 `ForwardGrad`，实现了 `std::enable_shared_from_this` 接口
struct TORCH_API ForwardGrad : std::enable_shared_from_this<ForwardGrad> {
  ForwardGrad() = default;

  // 清空函数，用于在 `AutogradMeta` 或 `SavedVariable` 销毁时调用
  // 确保：
  //   - 此 `ForwardGrad` 的唯一（可能的）其他引用是它所注册的不同级别
  //   - 现在开始，没有其他线程会调用 `set_value` 或 `value`
  //   - 在此函数执行期间，此 `ForwardGrad` 所注册的任何 `ForwardADLevel` 可能会随时调用 `reset`
  void clear() {
    c10::SmallVector<uint64_t, EXPECTED_MAX_LEVEL> levels_idx;

    {
      // 使用互斥锁保护临界区域
      std::lock_guard<std::mutex> lock(mutex_);
      // 收集所有级别的索引
      for (auto& c : content_) {
        levels_idx.push_back(c.first);
      }
    }

    // 遍历收集到的级别索引
    for (auto l_idx : levels_idx) {
      // 在此处使用 "try" 版本，因为另一个线程可能在我们到达之前删除了此级别
      // 这是一个拥有的引用，因为我们希望保持级别活动，直到成功注销自己
      auto level = ForwardADLevel::try_get_by_idx(l_idx);
      if (level) {
        level->erase(shared_from_this());
      }
    }
  }

  // 设置值函数，将张量值与给定级别关联
  void set_value(const at::Tensor& value, uint64_t level) {
    // 拥有的引用，确保在更新内部状态时不会销毁 `forward_level`
    auto forward_level = ForwardADLevel::get_by_idx(level);
    forward_level->insert(shared_from_this());

    // 使用互斥锁保护临界区域
    std::lock_guard<std::mutex> lock(mutex_);
    // 将值与级别插入到内容映射中
    content_.insert({level, value});
  }

  // 重置函数，删除给定级别的梯度信息
  // 使用 `update_level` 标志来禁止通知级别关于此重置的情况
  // 此标志主要由 `ForwardADLevel` 析构函数使用
  void reset(uint64_t level, bool update_level = true) {
    if (update_level) {
      ForwardADLevel::get_by_idx(level)->erase(shared_from_this());
    }

    // 使用独占锁保护临界区域
    std::unique_lock<std::mutex> lock(mutex_);
    // 查找要重置的级别
    const auto& it = content_.find(level);
    // 断言确保重置存在的级别
    TORCH_INTERNAL_ASSERT(
        it != content_.end(), "Resetting a non-existent level.");
    // 在释放锁之前保持张量存活，这是因为可能在 `ForwardADLevel` 析构函数中调用此函数
    auto t = (*it).second;
    // 从内容映射中删除级别
    content_.erase(level);
    lock.unlock();
  }

  // 返回给定级别的值张量的常量引用
  const at::Tensor& value(uint64_t level) const;

  // 检查给定级别是否存在于内容映射中
  bool contains(uint64_t level) {
    // 使用互斥锁保护临界区域
    std::lock_guard<std::mutex> lock(mutex_);
    return content_.count(level) > 0;
  }

  // 检查内容映射是否为空
  bool empty() const {
    return content_.empty();
  }

  // 返回未定义梯度的张量的常量引用
  static const at::Tensor& undef_grad();

 private:
  // TODO(albanD): 替换此处的 `unordered_map` 为 `SmallVector`
  // 内容映射，将级别映射到张量
  std::unordered_map<uint64_t, at::Tensor> content_;
  // 互斥锁，用于保护内容映射的并发访问
  mutable std::mutex mutex_;
};

// 命名空间结束标记 `torch::autograd`
} // namespace torch::autograd
```