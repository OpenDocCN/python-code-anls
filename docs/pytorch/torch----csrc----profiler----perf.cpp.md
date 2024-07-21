# `.\pytorch\torch\csrc\profiler\perf.cpp`

```
// 引入需要的头文件：无序映射和无序集合
#include <unordered_map>
#include <unordered_set>

// 引入 Torch 的性能分析模块的内部头文件
#include <torch/csrc/profiler/perf-inl.h>
#include <torch/csrc/profiler/perf.h>

// 定义命名空间 torch::profiler::impl::linux_perf
namespace torch::profiler::impl::linux_perf {

// 如果是 Android 或者 Linux 系统
#if defined(__ANDROID__) || defined(__linux__)

/*
 * PerfEvent
 * ---------
 */

/*
 * perf_event_open 的系统调用封装函数，用于调用 perf_event_open(2)
 */
inline long perf_event_open(
    struct perf_event_attr* hw_event,
    pid_t pid,
    int cpu,
    int group_fd,
    unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// TODO: 与 profiler/events.h 中的 Kineto 层抽象事件进行同步

// 静态的无序映射 EventTable，将事件名称映射到对应的 perf_type_id 和事件类型
static const std::unordered_map<
    std::string,
    std::pair<perf_type_id, /* perf event type */ uint32_t>>
    EventTable{
        {"cycles",
         std::make_pair(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES)},
        {"instructions",
         std::make_pair(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS)},

        // 非标准测试事件
        {"pagefaults",
         std::make_pair(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS)},
        {"backend-stall-cycles",
         std::make_pair(
             PERF_TYPE_HARDWARE,
             PERF_COUNT_HW_STALLED_CYCLES_BACKEND)},
        {"frontend-stall-cycles",
         std::make_pair(
             PERF_TYPE_HARDWARE,
             PERF_COUNT_HW_STALLED_CYCLES_FRONTEND)}};

// PerfEvent 类的析构函数
PerfEvent::~PerfEvent() {
  // 如果文件描述符大于 -1，则关闭文件描述符
  if (fd_ > -1) {
    close(fd_);
  }
  fd_ = -1; // 标记为毒药值，表示对象已经被销毁
}

// PerfEvent 类的初始化函数
void PerfEvent::Init() {
  // 检查事件名称是否为空
  TORCH_CHECK(!name_.empty(), "Invalid profiler event name");

  // 在 EventTable 中查找给定名称的事件
  auto const it = EventTable.find(name_);
  // 如果未找到，则抛出异常
  if (it == EventTable.end()) {
    TORCH_CHECK(false, "Unsupported profiler event name: ", name_);
  }

  // 初始化 perf_event_attr 结构体，并清零
  struct perf_event_attr attr {};
  memset(&attr, 0, sizeof(attr));

  // 设置 perf_event_attr 结构体的字段
  attr.size = sizeof(perf_event_attr);
  attr.type = it->second.first; // 设置事件类型
  attr.config = it->second.second; // 设置事件配置
  attr.disabled = 1; // 初始时禁用事件
  attr.inherit = 1; // 子进程继承该事件
  attr.exclude_kernel = 1; // 排除内核空间
  attr.exclude_hv = 1; // 排除虚拟化层

  /*
   * 如果性能计数器过载且正在进行复用，可以用这些字段计算估算的总时间
   */
  attr.read_format =
      PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

  pid_t pid = getpid(); // 获取当前进程的 PID
  int cpu = -1; // 指定所有 CPU
  int group_fd = -1; // 未使用分组文件描述符
  unsigned long flags = 0; // 无特殊标志

  // 调用 perf_event_open 函数，获取事件的文件描述符
  fd_ = static_cast<int>(perf_event_open(&attr, pid, cpu, group_fd, flags));
  // 如果打开失败，则抛出异常
  if (fd_ == -1) {
    TORCH_CHECK(
        false, "perf_event_open() failed, error: ", std::strerror(errno));
  }
  // 重置事件的状态
  Reset();
}
/*
 * PerfEvent
 * ---------
 */

/*
 * 读取性能计数器的当前值。
 * 从文件描述符中读取 PerfCounter 结构体，确保完整读取，否则抛出错误。
 * 检查性能计数器的时间启用和运行时间是否相等，否则抛出未处理的多路复用错误。
 * 返回性能计数器的值。
 */
uint64_t PerfEvent::ReadCounter() const {
  PerfCounter counter{};
  long n = read(fd_, &counter, sizeof(PerfCounter));
  TORCH_CHECK(
      n == sizeof(counter),
      "Read failed for Perf event fd, event : ",
      name_,
      ", error: ",
      std::strerror(errno));
  TORCH_CHECK(
      counter.time_enabled == counter.time_running,
      "Hardware performance counter time multiplexing is not handled yet",
      ", name: ",
      name_,
      ", enabled: ",
      counter.time_enabled,
      ", running: ",
      counter.time_running);
  return counter.value;
}

#else /* __ANDROID__ || __linux__ */
/*
 * 未支持平台的 Shim 类 - 总是返回计数器值为 0
 */
PerfEvent::~PerfEvent(){};

void PerfEvent::Init(){};

uint64_t PerfEvent::ReadCounter() const {
  return 0;
};

#endif /* __ANDROID__ || __linux__ */

/*
 * PerfProfiler
 * ------------
 */

/*
 * 配置性能分析器，设置要监视的事件名称。
 * 检查事件名称数量不超过最大允许值 MAX_EVENTS。
 * 检查是否有重复的事件名称。
 * 初始化每个事件，并加入到 events_ 中。
 * TODO: 在此处重置 pthreadpool，以确保能够附加到新的子线程。
 */
void PerfProfiler::Configure(std::vector<std::string>& event_names) {
  TORCH_CHECK(
      event_names.size() <= MAX_EVENTS,
      "Too many events to configure, configured: ",
      event_names.size(),
      ", max allowed:",
      MAX_EVENTS);
  std::unordered_set<std::string> s(event_names.begin(), event_names.end());
  TORCH_CHECK(
      s.size() == event_names.size(), "Duplicate event names are not allowed!")
  for (auto name : event_names) {
    events_.emplace_back(name);
    events_.back().Init();
  }

  // TODO
  // Reset pthreadpool here to make sure we can attach to new children
  // threads
}

/*
 * 启用性能分析器，开始计数。
 * 如果已经有起始值存在，则先停止计数。
 * 获取每个事件的当前计数器值，并存入 start_values_。
 * 开始计数。
 */
void PerfProfiler::Enable() {
  if (!start_values_.empty()) {
    StopCounting();
  }

  start_values_.emplace(events_.size(), 0);

  auto& sv = start_values_.top();
  for (unsigned i = 0; i < events_.size(); ++i) {
    sv[i] = events_[i].ReadCounter();
  }
  StartCounting();
}

/*
 * 停用性能分析器，停止计数。
 * 检查要输出的性能计数器容器与事件数量是否匹配。
 * 检查 PerfProfiler 是否已经启用，否则抛出错误。
 * 将停用事件连接到开始计数时的事件，计算每个事件的增量，并存入 vals 中。
 * 弹出 start_values_ 的顶部值。
 * 如果还有父级存在，则重新启用计数。
 */
void PerfProfiler::Disable(perf_counters_t& vals) {
  StopCounting();
  TORCH_CHECK(
      vals.size() == events_.size(),
      "Can not fit all perf counters in the supplied container");
  TORCH_CHECK(
      !start_values_.empty(), "PerfProfiler must be enabled before disabling");

  /* Always connecting this disable event to the last enable event i.e. using
   * whatever is on the top of the start counter value stack. */
  perf_counters_t& sv = start_values_.top();
  for (unsigned i = 0; i < events_.size(); ++i) {
    vals[i] = CalcDelta(sv[i], events_[i].ReadCounter());
  }
  start_values_.pop();

  // Restore it for a parent
  if (!start_values_.empty()) {
    StartCounting();
  }
}
```