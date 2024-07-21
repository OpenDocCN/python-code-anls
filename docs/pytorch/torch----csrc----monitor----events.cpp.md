# `.\pytorch\torch\csrc\monitor\events.cpp`

```py
// 引入 Torch 监视器事件头文件

#include <torch/csrc/monitor/events.h>

// 引入标准库头文件
#include <algorithm>
#include <mutex>
#include <vector>

// 定义命名空间 torch::monitor
namespace torch {
namespace monitor {

// 匿名命名空间，包含 EventHandlers 类的实现细节
namespace {
// EventHandlers 类，管理事件处理器
class EventHandlers {
 public:
  // 注册事件处理器，添加到 handlers_ 中
  void registerEventHandler(std::shared_ptr<EventHandler> handler) noexcept {
    std::unique_lock<std::mutex> lock(mu_);
    handlers_.emplace_back(std::move(handler));
  }

  // 注销事件处理器，从 handlers_ 中移除
  void unregisterEventHandler(
      const std::shared_ptr<EventHandler>& handler) noexcept {
    std::unique_lock<std::mutex> lock(mu_);
    auto it = std::find(handlers_.begin(), handlers_.end(), handler);
    handlers_.erase(it);
  }

  // 记录事件，依次调用每个 handler 的 handle 方法处理事件 e
  void logEvent(const Event& e) {
    std::unique_lock<std::mutex> lock(mu_);
    for (auto& handler : handlers_) {
      handler->handle(e);
    }
  }

  // 获取静态单例 EventHandlers 对象
  static EventHandlers& get() noexcept {
    static EventHandlers ehs;
    return ehs;
  }

 private:
  std::mutex mu_{}; // 互斥量，保护 handlers_ 的并发访问
  std::vector<std::shared_ptr<EventHandler>> handlers_{}; // 存储事件处理器的容器
};
} // namespace

// 对外接口，记录事件 e 的全局函数，委托给 EventHandlers::get().logEvent(e)
void logEvent(const Event& e) {
  EventHandlers::get().logEvent(e);
}

// 对外接口，注册事件处理器 p 的全局函数，委托给 EventHandlers::get().registerEventHandler(p)
void registerEventHandler(std::shared_ptr<EventHandler> p) {
  EventHandlers::get().registerEventHandler(std::move(p));
}

// 对外接口，注销事件处理器 p 的全局函数，委托给 EventHandlers::get().unregisterEventHandler(p)
void unregisterEventHandler(const std::shared_ptr<EventHandler>& p) {
  EventHandlers::get().unregisterEventHandler(p);
}

} // namespace monitor
} // namespace torch


这段代码实现了一个事件监视器的功能，通过 `EventHandlers` 类管理注册的事件处理器，提供了注册、注销和记录事件的接口函数，并使用互斥量确保线程安全访问 `handlers_` 容器。
```