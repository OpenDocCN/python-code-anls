# `.\pytorch\torch\csrc\utils\python_raii.h`

```py
// 包含必要的头文件
#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>
#include <tuple>

// 定义命名空间 torch::impl
namespace torch::impl {

// RAIIContextManager 模板结构，用于管理资源的 RAII 实现
template <typename GuardT, typename... Args>
struct RAIIContextManager {
  // 构造函数，接受参数并初始化成员变量
  explicit RAIIContextManager(Args&&... args)
      : args_(std::forward<Args>(args)...) {}

  // 进入上下文的方法
  void enter() {
    // 使用 lambda 表达式 emplace 来初始化 guard_
    auto emplace = [&](Args... args) {
      guard_.emplace(std::forward<Args>(args)...);
    };
    // 将参数 args_ 应用到 emplace 函数
    std::apply(std::move(emplace), args_);
  }

  // 退出上下文的方法
  void exit() {
    // 清空 guard_，释放资源
    guard_ = c10::nullopt;
  }

 private:
  std::optional<GuardT> guard_; // 使用 std::optional 来保存资源的可选对象
  std::tuple<Args...> args_;    // 使用 std::tuple 来保存参数列表
};

// 将 C++ 的 RAII guard 转换为 Python 上下文管理器
template <typename GuardT, typename... GuardArgs>
void py_context_manager(const py::module& m, const char* name) {
  using ContextManagerT = RAIIContextManager<GuardT, GuardArgs...>;
  // 在 Python 中注册 ContextManagerT 类
  py::class_<ContextManagerT>(m, name)
      .def(py::init<GuardArgs...>()) // 定义初始化函数
      .def("__enter__", [](ContextManagerT& guard) { guard.enter(); }) // 定义进入上下文方法
      .def(
          "__exit__",
          [](ContextManagerT& guard,
             const py::object& exc_type,
             const py::object& exc_value,
             const py::object& traceback) { guard.exit(); }); // 定义退出上下文方法
}

// DeprecatedRAIIContextManager 模板结构，用于兼容老版本的 RAII 实现
template <typename GuardT, typename... Args>
struct DeprecatedRAIIContextManager {
  // 构造函数，接受参数并初始化 guard_
  explicit DeprecatedRAIIContextManager(Args&&... args) {
    guard_.emplace(std::forward<Args>(args)...);
  }

  // 进入上下文的方法（空实现）
  void enter() {}

  // 退出上下文的方法，清空 guard_，释放资源
  void exit() {
    guard_ = c10::nullopt;
  }

 private:
  std::optional<GuardT> guard_; // 使用 std::optional 来保存资源的可选对象
  std::tuple<Args...> args_;    // 使用 std::tuple 来保存参数列表
};

// 定义：Python RAII guard 是一个在 Python 中初始化时获取资源，在删除时释放资源的对象。
//
// 该 API 将 C++ 的 RAII guard 转换为一个既可以作为 Python 上下文管理器，也可以作为 Python RAII guard 使用的对象。
//
// 如果您要绑定新的 RAII guard 到 Python，优先使用 py_context_manager，因为 Python 中没有关于对象何时被删除的明确保证。
template <typename GuardT, typename... GuardArgs>
void py_context_manager_DEPRECATED(const py::module& m, const char* name) {
  using ContextManagerT = DeprecatedRAIIContextManager<GuardT, GuardArgs...>;
  // 在 Python 中注册 ContextManagerT 类
  py::class_<ContextManagerT>(m, name)
      .def(py::init<GuardArgs...>()) // 定义初始化函数
      .def("__enter__", [](ContextManagerT& guard) { guard.enter(); }) // 定义进入上下文方法（空实现）
      .def(
          "__exit__",
          [](ContextManagerT& guard,
             const py::object& exc_type,
             const py::object& exc_value,
             const py::object& traceback) { guard.exit(); }); // 定义退出上下文方法
}

} // namespace torch::impl
```