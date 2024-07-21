# `.\pytorch\torch\csrc\autograd\forward_grad.cpp`

```py
// 引入 Torch 库中的 forward_grad.h 头文件

#include <torch/csrc/autograd/forward_grad.h>

// 定义命名空间 torch::autograd
namespace torch {
namespace autograd {

// 匿名命名空间，讨论在 forward_grad.h 中全局变量而非线程局部变量的原因
namespace {
// 用于保护所有前向自动微分级别的互斥锁
std::mutex all_forward_levels_mutex_;

// 存储所有前向自动微分级别的共享指针向量
std::vector<std::shared_ptr<ForwardADLevel>> all_forward_levels_;

// 单例未定义张量，用于表示未定义梯度
const static at::Tensor singleton_undefined_tensor;
} // namespace

// 获取下一个可用的前向自动微分级别的索引
uint64_t ForwardADLevel::get_next_idx() {
  std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
  auto next_idx = all_forward_levels_.size();
  TORCH_CHECK(
      next_idx == 0, "Nested forward mode AD is not supported at the moment");
  // 创建并添加一个新的前向自动微分级别对象到 all_forward_levels_ 中
  all_forward_levels_.push_back(std::make_shared<ForwardADLevel>(next_idx));
  return next_idx;
}

// 释放给定索引的前向自动微分级别
void ForwardADLevel::release_idx(uint64_t idx) {
  std::unique_lock<std::mutex> lock(all_forward_levels_mutex_);
  TORCH_CHECK(
      idx + 1 == all_forward_levels_.size(),
      "Exiting a forward AD level that is not the "
      "last that was created is not support. Ensure they are released in the reverse "
      "order they were created.");
  TORCH_INTERNAL_ASSERT(!all_forward_levels_.empty());
  // 移除最后一个前向自动微分级别对象
  auto lvl = all_forward_levels_.back();
  all_forward_levels_.pop_back();
  lock.unlock();
}

// 根据索引获取前向自动微分级别对象的共享指针
std::shared_ptr<ForwardADLevel> ForwardADLevel::get_by_idx(uint64_t idx) {
  std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
  TORCH_CHECK(
      idx < all_forward_levels_.size(),
      "Trying to access a forward AD level with an invalid index. "
      "This index was either not created or is already deleted.");
  return all_forward_levels_[idx];
}

// 尝试根据索引获取前向自动微分级别对象的共享指针，如果索引无效则返回 nullptr
std::shared_ptr<ForwardADLevel> ForwardADLevel::try_get_by_idx(uint64_t idx) {
  std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
  if (idx < all_forward_levels_.size()) {
    return all_forward_levels_[idx];
  } else {
    return nullptr;
  }
}

// 前向自动微分级别对象的析构函数，负责清理对象内部的梯度列表
ForwardADLevel::~ForwardADLevel() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = grads_.begin();
  while (it != grads_.end()) {
    // 警告：这将锁定 *it 的互斥锁
    // 这是可以接受的，因为这个函数是唯一一个调用另一个类方法的函数
    (*it)->reset(idx_, /* update_level */ false);
    it = grads_.erase(it);
  }
}

// 获取给定级别的前向梯度对象的值
const at::Tensor& ForwardGrad::value(uint64_t level) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& it = content_.find(level);
  // 如果未找到该级别的值，则返回单例未定义张量
  return it == content_.end() ? singleton_undefined_tensor : (*it).second;
}

// 返回单例未定义张量，用于表示未定义梯度
const at::Tensor& ForwardGrad::undef_grad() {
  return singleton_undefined_tensor;
}

} // namespace autograd
} // namespace torch
```