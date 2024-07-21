# `.\pytorch\torch\csrc\profiler\kineto_client_interface.cpp`

```
#ifdef USE_KINETO
#include <ATen/Context.h>
#include <libkineto.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <chrono>
#include <thread>

// Ondemand tracing is not supported on Apple or edge platform
#if defined(__APPLE__) || defined(EDGE_PROFILER_USE_KINETO)
#define ENABLE_GLOBAL_OBSERVER (0)
#else
#define ENABLE_GLOBAL_OBSERVER (1)
#endif

namespace torch {

namespace profiler::impl {

namespace {

using namespace torch::autograd::profiler;

// 定义 LibKinetoClient 类，实现 libkineto::ClientInterface 接口
class LibKinetoClient : public libkineto::ClientInterface {
 public:
  // 初始化函数，暂无实际实现
  void init() override {}

  // 准备函数，设置性能分析选项
  void prepare(
      bool report_input_shapes = false,
      bool profile_memory = false,
      bool with_stack = false,
      bool with_flops = false,
      bool with_modules = false) override {
    reportInputShapes_ = report_input_shapes;
    profileMemory_ = profile_memory;
    withStack_ = with_stack;
    withFlops_ = with_flops;
    withModules_ = with_modules;
  }

  // 启动函数，配置并启动性能分析器
  void start() override {
    // 配置性能分析器的参数
    ProfilerConfig cfg{
        ProfilerState::KINETO_ONDEMAND,
        /*report_input_shapes=*/reportInputShapes_,
        /*profile_memory=*/profileMemory_,
        /*with_stack=*/withStack_,
        /*with_flops=*/withFlops_,
        /*with_modules=*/withModules_};
    std::set<ActivityType> activities{ActivityType::CPU};
    std::unordered_set<at::RecordScope> scopes;
    scopes.insert(at::RecordScope::FUNCTION);
    scopes.insert(at::RecordScope::USER_SCOPE);
    scopes.insert(at::RecordScope::BACKWARD_FUNCTION);
    // 启用性能分析器
    enableProfiler(cfg, activities, scopes);
  }

  // 停止函数，停止性能分析器
  void stop() override {
    (void)disableProfiler();
  }

 private:
  // 临时禁用形状收集，直到重新为按需情况重新实现该功能
  bool reportInputShapes_{false};
  bool profileMemory_{false};
  bool withStack_{false};
  bool withFlops_{false};
  bool withModules_{false};
};

} // namespace

} // namespace profiler::impl

// 如果启用全局观察器，定义相关的实现
#if ENABLE_GLOBAL_OBSERVER
namespace {

// 获取初始化延迟时间的函数
int get_init_delay() {
  const char* delay_c = std::getenv("KINETO_DAEMON_INIT_DELAY_S");
  if (!delay_c) {
    return -1;
  }
  std::string delay_s{delay_c};
  try {
    return std::stoi(delay_s);
  } catch (const std::invalid_argument& _) {
    return -1;
  }
}

// 注册 LibKinetoClient 类的实例作为 Kineto 客户端
struct RegisterLibKinetoClient {
  RegisterLibKinetoClient() {
    static profiler::impl::LibKinetoClient client;
    libkineto::api().registerClient(&client);

    // Kineto 初始化函数
    auto kineto_init = []() {
      libkineto_init(
          /*cpuOnly=*/!(at::hasCUDA() || at::hasXPU() || at::hasMTIA()),
          /*logOnError=*/true);
      libkineto::api().suppressLogMessages();
    };

    // 如果设置了使用守护进程的环境变量，根据延迟时间启动初始化
    if (std::getenv("KINETO_USE_DAEMON") != nullptr) {
      int init_delay_s = get_init_delay();
      if (init_delay_s > 0) {
        std::thread t([init_delay_s, kineto_init]() {
          std::this_thread::sleep_for(std::chrono::seconds(init_delay_s));
          kineto_init();
        });
        t.detach();
      } else {
        kineto_init();
      }
    }
  }
} register_libkineto_client;

} // namespace
#endif
} // namespace torch
#endif // USE_KINETO
```