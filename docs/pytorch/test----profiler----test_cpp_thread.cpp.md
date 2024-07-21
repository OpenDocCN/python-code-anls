# `.\pytorch\test\profiler\test_cpp_thread.cpp`

```
// 包含 Torch 的自动微分和分析器的头文件
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/torch.h>
// 包含 C++ 标准库中的字符串处理功能
#include <string>

// 使用 Torch 的 profiler 命名空间中的声明
using namespace torch::autograd::profiler;

// 打印输出带有蓝色的文本
void blueprint(const std::string& text) {
  printf("\33[94m%s\33[0m\n", text.c_str());
}

/**
 * 模拟 C++ 训练引擎调用 Python 控制分析器的方式。
 */
class ProfilerEventHandler
    : public std::enable_shared_from_this<ProfilerEventHandler> {
 public:
  // 共享指针用于跟踪全局的事件处理器
  static std::shared_ptr<ProfilerEventHandler> Handler;
  // 注册全局的事件处理器
  static void Register(const std::shared_ptr<ProfilerEventHandler>& handler) {
    Handler = handler;
  }

 public:
  // 虚析构函数
  virtual ~ProfilerEventHandler() {}
  // 迭代开始时的事件处理，纯虚函数
  virtual void onIterationStart(int) {}
  // 模拟训练过程，纯虚函数
  virtual void emulateTraining(int, int) {}
};
std::shared_ptr<ProfilerEventHandler> ProfilerEventHandler::Handler;

// 事件处理器的跳板类，继承自 ProfilerEventHandler
class ProfilerEventHandlerTrampoline : public ProfilerEventHandler {
 public:
  // 重写迭代开始事件处理
  virtual void onIterationStart(int iteration) override {
    PYBIND11_OVERRIDE(void, ProfilerEventHandler, onIterationStart, iteration);
  }
  // 重写模拟训练事件处理
  virtual void emulateTraining(int iteration, int thread_id) override {
    PYBIND11_OVERRIDE(
        void, ProfilerEventHandler, emulateTraining, iteration, thread_id);
  }
};

/**
 * C++ 训练引擎的入口点函数。
 */
void start_threads(int thread_count, int iteration_count, bool attach) {
  // 打印输出开始启动 C++ 线程
  blueprint("start_cpp_threads called");

  // 静态原子整数用于线程间的同步
  static std::atomic<int> barrier = 0;
  barrier = 0;
  // 线程局部变量，用于记录主线程是否启用分析器
  thread_local bool enabled_in_main_thread = false;

  // 存储线程对象的向量
  std::vector<std::thread> threads;
  // 循环创建并启动多个线程
  for (int id = 0; id < thread_count; id++) {
    // 打印输出正在启动的线程编号
    blueprint("starting thread " + std::to_string(id));
    // 向线程向量中添加线程对象
    threads.emplace_back([thread_count, iteration_count, id, attach]() {
      // 循环执行指定次数的迭代
      for (int iteration = 0; iteration < iteration_count; iteration++) {
        // 如果是主线程
        if (id == 0) {
          // 调用全局事件处理器的迭代开始处理函数
          ProfilerEventHandler::Handler->onIterationStart(iteration);
        }

        // 此屏障确保当主线程启用分析器时，所有子线程也将被启用分析器
        ++barrier;
        while (barrier % thread_count) {
          std::this_thread::yield();
        }

        // 如果是子线程且需要附加分析器
        if (id > 0 && attach) {
          // 检查主线程中分析器是否启用
          bool enabled = isProfilerEnabledInMainThread();
          // 如果当前状态与主线程不同，则在子线程中启用或禁用分析器
          if (enabled != enabled_in_main_thread) {
            if (enabled) {
              enableProfilerInChildThread();
            } else {
              disableProfilerInChildThread();
            }
            enabled_in_main_thread = enabled;
          }
        }

        // 调用全局事件处理器的模拟训练处理函数
        ProfilerEventHandler::Handler->emulateTraining(iteration, id);
      }
    });
  }

  // 等待所有线程完成
  for (auto& t : threads) {
    t.join();
  }
}
// 创建一个 Python 模块，命名为 "profiler_test_cpp_thread_lib"
PYBIND11_MODULE(profiler_test_cpp_thread_lib, m) {
  // 定义一个 Python 类 ProfilerEventHandler，绑定到 C++ 类 ProfilerEventHandler
  // 使用 std::shared_ptr<ProfilerEventHandler> 进行对象管理
  py::class_<
      ProfilerEventHandler,                              // Python 类的名称
      ProfilerEventHandlerTrampoline,                    // Python 类的基类
      std::shared_ptr<ProfilerEventHandler>>(m, "ProfilerEventHandler")
      .def(py::init<>())                                 // 定义构造函数 __init__
      .def_static("Register", &ProfilerEventHandler::Register)  // 定义静态方法 Register
      .def(
          "onIterationStart",                            // 定义成员方法 onIterationStart
          &ProfilerEventHandler::onIterationStart,
          py::call_guard<py::gil_scoped_release>())      // 设置 GIL 保护，释放 GIL
      .def(
          "emulateTraining",                             // 定义成员方法 emulateTraining
          &ProfilerEventHandler::emulateTraining,
          py::call_guard<py::gil_scoped_release>());     // 设置 GIL 保护，释放 GIL

  // 定义一个 Python 函数 start_threads，绑定到 C++ 函数 start_threads
  m.def(
      "start_threads",
      &start_threads,
      py::call_guard<py::gil_scoped_release>());         // 设置 GIL 保护，释放 GIL
};
```