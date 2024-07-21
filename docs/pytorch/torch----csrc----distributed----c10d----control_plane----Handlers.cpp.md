# `.\pytorch\torch\csrc\distributed\c10d\control_plane\Handlers.cpp`

```
// 包含 Torch 分布式控制平面的处理程序头文件
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

// 包含格式化输出库的头文件
#include <fmt/format.h>

// 包含互斥量和共享互斥量的头文件
#include <mutex>
#include <shared_mutex>

// 包含标准异常处理的头文件
#include <stdexcept>

// c10d 命名空间
namespace c10d {
namespace control_plane {

// 私有静态命名空间，用于隐藏内部实现细节
namespace {

// 处理程序注册表类
class HandlerRegistry {
 public:
  // 注册处理程序函数，将处理程序与名称关联
  void registerHandler(const std::string& name, HandlerFunc f) {
    std::unique_lock<std::shared_mutex> lock(handlersMutex_);

    // 如果已经存在同名的处理程序，则抛出运行时异常
    if (handlers_.find(name) != handlers_.end()) {
      throw std::runtime_error(
          fmt::format("Handler {} already registered", name));
    }

    // 将处理程序函数关联到名称
    handlers_[name] = f;
  }

  // 获取指定名称的处理程序函数
  HandlerFunc getHandler(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(handlersMutex_);

    // 查找指定名称的处理程序
    auto it = handlers_.find(name);
    if (it == handlers_.end()) {
      throw std::runtime_error(fmt::format("Failed to find handler {}", name));
    }
    return handlers_[name];
  }

  // 获取所有注册的处理程序名称列表
  std::vector<std::string> getHandlerNames() {
    std::shared_lock<std::shared_mutex> lock(handlersMutex_);

    std::vector<std::string> names;
    // 遍历所有注册的处理程序，将名称添加到列表中
    for (const auto& [name, _] : handlers_) {
      names.push_back(name);
    }
    return names;
  }

 private:
  std::shared_mutex handlersMutex_{}; // 用于保护处理程序注册表的共享互斥量
  std::unordered_map<std::string, HandlerFunc> handlers_{}; // 存储处理程序名称及其对应的函数
};

// 获取处理程序注册表的实例
HandlerRegistry& getHandlerRegistry() {
  static HandlerRegistry registry;
  return registry;
}

// 在静态初始化期间自动注册 "ping" 处理程序
RegisterHandler pingHandler{"ping", [](const Request&, Response& res) {
                              res.setContent("pong", "text/plain");
                            }};

} // namespace

// 注册新的处理程序函数到处理程序注册表中
void registerHandler(const std::string& name, HandlerFunc f) {
  return getHandlerRegistry().registerHandler(name, f);
}

// 获取指定名称的处理程序函数
HandlerFunc getHandler(const std::string& name) {
  return getHandlerRegistry().getHandler(name);
}

// 获取所有已注册的处理程序名称列表
std::vector<std::string> getHandlerNames() {
  return getHandlerRegistry().getHandlerNames();
}

} // namespace control_plane
} // namespace c10d
```