# `.\pytorch\torch\csrc\autograd\anomaly_mode.cpp`

```py
#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>
#include <mutex>

namespace torch {
namespace autograd {

// 是否启用异常检测模式的标志
bool AnomalyMode::_enabled = false;
// 是否检查 NaN 值的标志
bool AnomalyMode::_check_nan = true;

namespace {
// 获取异常检测互斥锁对象
std::mutex& get_anomaly_guard_lock() {
  static std::mutex anomaly_guard_lock{};
  return anomaly_guard_lock;
}

// 获取异常计数器对象
uint32_t& get_anomaly_counter() {
  static uint32_t counter = 0;
  return counter;
}
} // namespace

// 异常检测守护类的构造函数
DetectAnomalyGuard::DetectAnomalyGuard(bool check_nan) {
  // 发出警告信息，提醒仅在调试时启用此模式，因为不同的测试会减慢程序执行速度
  TORCH_WARN_ONCE(
      "This mode should be enabled only for debugging as the different tests will slow down your program execution.");
  // 获取异常检测互斥锁对象的锁保护
  std::lock_guard<std::mutex> lock(get_anomaly_guard_lock());
  // 获取当前异常计数器的引用并递增
  uint32_t& counter = get_anomaly_counter();
  counter++;
  // 保存当前的 NaN 检查状态，并设置异常检测模式为启用状态
  this->prev_check_nan_ = AnomalyMode::should_check_nan();
  AnomalyMode::set_enabled(true, check_nan);
}

// 异常检测守护类的析构函数
DetectAnomalyGuard::~DetectAnomalyGuard() {
  // 获取异常检测互斥锁对象的锁保护
  std::lock_guard<std::mutex> lock(get_anomaly_guard_lock());
  // 获取当前异常计数器的引用并递减
  uint32_t& counter = get_anomaly_counter();
  counter--;
  // 恢复之前的 NaN 检查状态，并根据异常计数器判断是否保持异常检测模式为启用状态
  AnomalyMode::set_enabled(counter > 0, this->prev_check_nan_);
}

// 异常元数据类的析构函数的默认实现
AnomalyMetadata::~AnomalyMetadata() = default;

// 存储当前调用栈信息
void AnomalyMetadata::store_stack() {
  traceback_ = c10::get_backtrace(/* frames_to_skip */ 1);
}

// 打印异常调用栈信息
void AnomalyMetadata::print_stack(const std::string& current_node_name) {
  // 发出警告信息，指示在当前节点名称中检测到错误，打印引发错误的前向调用的调用栈信息
  TORCH_WARN(
      "Error detected in ",
      current_node_name,
      ". ",
      "Traceback of forward call that caused the error:\n",
      traceback_);

  auto& cur_parent = parent_;
  // 如果元数据中没有 "parent_"，则停止打印调用栈，表明此节点为根节点
  while (cur_parent) {
    auto parent_metadata = cur_parent->metadata();
    // 发出警告信息，指示前一个计算由父节点引起，打印导致前一个计算的前向调用的调用栈信息
    TORCH_WARN(
        "\n\n",
        "Previous calculation was induced by ",
        cur_parent->name(),
        ". "
        "Traceback of forward call that induced the previous calculation:\n",
        parent_metadata->traceback_);
    // 获取父节点的父节点，如果当前节点为根节点，则父节点为空
    cur_parent = parent_metadata->parent_;
  }
}

// 分配父节点给当前异常元数据对象
void AnomalyMetadata::assign_parent(const std::shared_ptr<Node>& parent_node) {
  parent_ = parent_node;
}

} // namespace autograd
} // namespace torch
```