# `.\pytorch\torch\csrc\autograd\anomaly_mode.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/Export.h>
// 包含 Torch 库的导出头文件

#include <memory>
// 包含标准库中的内存管理功能

#include <string>
// 包含标准库中的字符串处理功能

namespace torch::autograd {

// 前向声明，声明来自 function.h 的 Node 结构体
struct Node;

struct TORCH_API AnomalyMode {
  // 返回当前异常检测模式是否启用
  static bool is_enabled() {
    return _enabled;
  }
  // 返回当前异常检测模式是否需要检查 NaN（不是数字）
  static bool should_check_nan() {
    return _check_nan;
  }
  // 设置异常检测模式的状态
  static void set_enabled(bool enabled, bool check_nan = true) {
    _enabled = enabled;
    _check_nan = check_nan;
  }

 private:
  static bool _enabled;
  static bool _check_nan;
};

/// 用于启用异常检测模式的 RAII（资源获取即初始化）守卫类
///
/// 异常检测模式在调试反向传播中非常有用，例如检测张量是否意外地被修改或在反向传播中出现 NaN。
///
/// 一旦有一个这样的守卫，异常模式就是全局启用的，影响所有的计算和线程。但它也会带来显著的性能损失。
///
/// 示例：
/// @code
/// auto x = torch::tensor({1.}, torch::requires_grad());
/// {
///   torch::autograd::DetectAnomalyGuard detect_anomaly;
///   auto x = torch::tensor({5.0}, torch::requires_grad());
///   auto y = x * x;
///   auto z = y * y;
///   y += 1;
///   z.backward();
/// }
/// @endcode
class TORCH_API DetectAnomalyGuard {
 public:
  // 构造函数，启用异常检测守卫并可选择是否检查 NaN
  DetectAnomalyGuard(bool check_nan = true);
  // 析构函数，用于退出异常检测守卫时的清理工作

 private:
  bool prev_check_nan_; // 保存进入异常检测守卫前的检查 NaN 的状态
};

// AnomalyMetadata 结构的声明
struct TORCH_API AnomalyMetadata {
  virtual ~AnomalyMetadata(); // 虚析构函数，用于多态类型的销毁
  // 存储当前调用栈信息
  virtual void store_stack();
  // 打印当前调用栈信息，传入当前节点名称作为参数
  virtual void print_stack(const std::string& current_node_name);
  // 指定父节点，传入父节点的共享指针作为参数
  virtual void assign_parent(const std::shared_ptr<Node>& parent_node);

 private:
  std::string traceback_; // 存储跟踪信息的字符串
  std::shared_ptr<Node> parent_; // 存储父节点的共享指针
};

} // namespace torch::autograd
```