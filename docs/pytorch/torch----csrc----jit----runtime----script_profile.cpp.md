# `.\pytorch\torch\csrc\jit\runtime\script_profile.cpp`

```
#include <torch/csrc/jit/runtime/script_profile.h>

#include <atomic>  // 引入原子操作支持
#include <chrono>  // 引入时间操作支持
#include <mutex>   // 引入互斥锁支持
#include <unordered_set>  // 引入无序集合支持

#include <c10/util/Exception.h>   // 引入异常处理支持
#include <c10/util/intrusive_ptr.h>  // 引入引用计数指针支持
#include <torch/csrc/jit/api/function_impl.h>  // 引入 Torch JIT API 的函数实现支持

namespace torch::jit {

namespace {

class ProfilesRegistry {
 public:
  // 检查注册表是否为空
  bool empty() {
    return empty_.load(std::memory_order_relaxed);  // 使用轻量级的内存顺序加载操作
  }

  // 添加一个脚本性能分析对象到注册表中
  void addProfile(ScriptProfile& p) {
    std::lock_guard<std::mutex> g(mutex_);  // 加锁当前线程，保护共享资源
    enabledProfiles_.emplace(&p);  // 将脚本性能分析对象添加到已启用的集合中
    empty_.store(false, std::memory_order_relaxed);  // 更新注册表非空状态
  }

  // 从注册表中移除一个脚本性能分析对象
  void removeProfile(ScriptProfile& p) {
    std::lock_guard<std::mutex> g(mutex_);  // 加锁当前线程，保护共享资源
    enabledProfiles_.erase(&p);  // 从已启用的集合中移除指定的脚本性能分析对象
    if (enabledProfiles_.empty()) {
      empty_.store(true, std::memory_order_relaxed);  // 若已启用的集合为空，则更新注册表为空状态
    }
  }

  // 向所有启用的脚本性能分析对象发送一个数据点
  void send(std::unique_ptr<profiling::Datapoint> datapoint) {
    auto shared = std::shared_ptr<profiling::Datapoint>(std::move(datapoint));  // 转移所有权并创建共享指针
    std::lock_guard<std::mutex> g(mutex_);  // 加锁当前线程，保护共享资源
    for (auto* p : enabledProfiles_) {
      p->addDatapoint(shared);  // 向每个启用的脚本性能分析对象添加数据点
    }
  }

 private:
  std::atomic<bool> empty_{true};  // 原子布尔值，表示注册表是否为空
  std::mutex mutex_;  // 互斥锁，用于保护对注册表的并发访问
  std::unordered_set<ScriptProfile*> enabledProfiles_;  // 存储已启用的脚本性能分析对象的集合
};

// 获取全局的性能分析注册表的实例
ProfilesRegistry& getProfilesRegistry() {
  static auto registry = std::ref(*new ProfilesRegistry{});  // 使用静态局部变量保证唯一实例
  return registry;  // 返回注册表的引用
}
// 初始化 Torch 绑定，定义和注册各个类的方法
auto initBindings() {
  // 定义并注册 profiling::SourceRef 类
  torch::class_<SourceRef>("profiling", "SourceRef")
      .def(
          "starting_lineno",
          // Lambda 函数，返回 SourceRef 对象的起始行号
          [](const c10::intrusive_ptr<SourceRef>& self) {
            return static_cast<int64_t>((*self)->starting_line_no());
          })
      .def("text", 
          // Lambda 函数，返回 SourceRef 对象的文本内容
          [](const c10::intrusive_ptr<SourceRef>& self) {
            return (*self)->text_str().str();
          });

  // 定义并注册 profiling::InstructionStats 类
  torch::class_<InstructionStats>("profiling", "InstructionStats")
      .def(
          "count",
          // Lambda 函数，返回 InstructionStats 对象的计数值
          [](const c10::intrusive_ptr<InstructionStats>& self) {
            return self->count;
          })
      .def("duration_ns", 
          // Lambda 函数，返回 InstructionStats 对象的持续时间（以纳秒为单位）
          [](const c10::intrusive_ptr<InstructionStats>& self) {
            return static_cast<int64_t>(self->duration.count());
          });

  // 定义并注册 profiling::SourceStats 类
  torch::class_<SourceStats>("profiling", "SourceStats")
      .def(
          "source",
          // Lambda 函数，返回 SourceStats 对象的源引用
          [](const c10::intrusive_ptr<SourceStats>& self) {
            return c10::make_intrusive<SourceRef>(self->getSourceRef());
          })
      .def("line_map", 
          // 返回 SourceStats 对象的行映射
          &SourceStats::getLineMap);

  // 定义并注册 profiling::_ScriptProfile 类
  torch::class_<ScriptProfile>("profiling", "_ScriptProfile")
      .def(torch::init<>())
      .def("enable", 
          // 启用脚本分析器
          &ScriptProfile::enable)
      .def("disable", 
          // 禁用脚本分析器
          &ScriptProfile::disable)
      .def("_dump_stats", 
          // Lambda 函数，导出分析数据统计
          [](const c10::intrusive_ptr<ScriptProfile>& self) {
            const auto& stats = self->dumpStats();
            c10::List<c10::intrusive_ptr<SourceStats>> ret;
            // 遍历统计数据，生成返回的 SourceStats 列表
            for (const auto& source : stats) {
              SourceStats::LineMap lineMap;
              for (const auto& line : source.second) {
                lineMap.insert(
                    line.first, c10::make_intrusive<InstructionStats>(line.second));
              }
              ret.push_back(c10::make_intrusive<SourceStats>(
                  source.first, std::move(lineMap)));
            }
            return ret;
          });
  return nullptr;
}

// 初始化 Torch 绑定
const auto C10_UNUSED torchBindInitializer = initBindings();

} // namespace

// 命名空间 profiling 中的定义

namespace profiling {

// InstructionSpan 类的构造函数，接收 Node 对象作为参数
InstructionSpan::InstructionSpan(Node& node) {
  // 创建 Datapoint 对象，并使用节点的源范围初始化
  datapoint_ = std::make_unique<Datapoint>(node.sourceRange());
}

// InstructionSpan 类的析构函数
InstructionSpan::~InstructionSpan() {
  // 设置 Datapoint 对象的结束时间为当前时间
  datapoint_->end = std::chrono::steady_clock::now();
  // 将 Datapoint 对象发送到 profiles 注册表中
  getProfilesRegistry().send(std::move(datapoint_));
}

// 检查是否正在进行分析
bool isProfilingOngoing() {
  // 返回 profiles 注册表是否为空
  return !getProfilesRegistry().empty();
}

} // namespace profiling

// 启用脚本分析器
void ScriptProfile::enable() {
  if (!std::exchange(enabled_, true)) {
    // 如果尚未启用，则添加此脚本分析器到 profiles 注册表中
    getProfilesRegistry().addProfile(*this);
  }
}

// 禁用脚本分析器
void ScriptProfile::disable() {
  if (std::exchange(enabled_, false)) {
    // 如果已启用，则从 profiles 注册表中移除此脚本分析器
    getProfilesRegistry().removeProfile(*this);
  }
}

// 添加 Datapoint 到脚本分析器
void ScriptProfile::addDatapoint(
    std::shared_ptr<profiling::Datapoint> datapoint) {
  // 检查脚本分析器是否已启用，否则抛出异常
  TORCH_CHECK(enabled_, "Cannot only add datapoint to disabled profilers.");
  // 将 Datapoint 对象添加到 datapoints_ 列表中
  datapoints_.push_back(std::move(datapoint));
}

// 获取脚本分析器的统计数据
const ScriptProfile::SourceMap& ScriptProfile::dumpStats() {
  // 检查脚本分析器是否已禁用，否则抛出异常
  TORCH_CHECK(!enabled_, "Only disabled profilers are allowed to dump stats.");

  // 遍历所有的 Datapoint 对象
  for (const auto& datapoint : datapoints_) {
    // 如果 datapoint 的源范围可用
    if (const auto& source = datapoint->sourceRange.source()) {
        // 如果可以获取到文件行列信息
        if (auto fileLineCol = datapoint->sourceRange.file_line_col()) {
            // 在 sourceMap_ 中查找源的条目
            auto it = sourceMap_.find(*source.get());
            // 如果找不到对应的条目，则插入新条目
            if (it == sourceMap_.end()) {
                it = sourceMap_.emplace(SourceRef{source}, LineMap{}).first;
            }
            // 获取对应源的行映射，并更新统计信息
            auto& stats = it->second[std::get<1>(*fileLineCol)];
            stats.count++;  // 增加计数
            stats.duration += datapoint->end - datapoint->start;  // 累加持续时间
        }
    }
  }
  // 清空 datapoints_ 列表
  datapoints_.clear();

  // 返回更新后的 sourceMap_
  return sourceMap_;
}

// ScriptProfile 析构函数的定义
ScriptProfile::~ScriptProfile() {
    // 如果 enabled_ 为 true，则从 profiles 注册表中移除当前的 profile 对象
    if (enabled_) {
        getProfilesRegistry().removeProfile(*this);
    }
}

// 命名空间结束标志，命名空间为 torch::jit
} // namespace torch::jit
```