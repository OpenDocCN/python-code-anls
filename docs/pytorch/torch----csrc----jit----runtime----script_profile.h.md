# `.\pytorch\torch\csrc\jit\runtime\script_profile.h`

```
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次

#include <chrono>
// 包含时间处理头文件chrono，用于处理时间点和持续时间

#include <map>
// 包含容器头文件map，用于定义键-值对的关联容器

#include <string>
// 包含字符串处理头文件string，用于字符串操作

#include <ATen/core/ivalue.h>
// 包含ATen库的头文件ivalue.h，提供与Torch张量相关的值类型

#include <c10/macros/Macros.h>
// 包含c10库的头文件Macros.h，提供一些预定义的宏

#include <torch/csrc/jit/frontend/source_ref.h>
// 包含Torch库的头文件source_ref.h，提供源码的引用信息

#include <torch/csrc/jit/ir/ir.h>
// 包含Torch库的头文件ir.h，提供表示Torch JIT中间表示的数据结构

namespace torch::jit {
namespace profiling {

struct Datapoint {
  using Timepoint = std::chrono::time_point<std::chrono::steady_clock>;
  // 定义时间点类型Timepoint为steady_clock的时间点类型
  SourceRange sourceRange;
  // 定义源码范围sourceRange，用于表示源码中的位置信息
  Timepoint start;
  // 记录操作开始时间点start
  Timepoint end;
  // 记录操作结束时间点end

  explicit Datapoint(SourceRange sr)
      : sourceRange(std::move(sr)), start(std::chrono::steady_clock::now()) {}
  // 构造函数，初始化sourceRange和start，start记录当前时间点
};

class TORCH_API InstructionSpan {
 public:
  explicit InstructionSpan(Node&);
  // 显式构造函数，接受一个Node引用作为参数
  ~InstructionSpan();
  // 析构函数，用于结束指令跨度

  InstructionSpan(InstructionSpan&&) = delete;
  // 删除移动构造函数
  InstructionSpan& operator=(InstructionSpan&&) = delete;
  // 删除移动赋值运算符

 private:
  std::unique_ptr<Datapoint> datapoint_;
  // 使用智能指针std::unique_ptr管理Datapoint对象的所有权
};

bool TORCH_API isProfilingOngoing();
// 声明函数isProfilingOngoing，用于检查是否正在进行性能分析

} // namespace profiling

struct TORCH_API InstructionStats : public CustomClassHolder {
  int64_t count{0};
  // 记录指令执行次数的计数器，默认初始化为0
  std::chrono::nanoseconds duration{0};
  // 记录指令执行总时间的持续时间，默认初始化为0纳秒
};

class TORCH_API SourceStats : public CustomClassHolder {
 public:
  using LineMap = c10::Dict<int64_t, c10::intrusive_ptr<InstructionStats>>;
  // 使用c10库中的Dict定义LineMap，键为行号，值为指向InstructionStats的智能指针

  SourceStats(SourceRef source, LineMap lineMap)
      : source_(std::move(source)), lineMap_(std::move(lineMap)) {}
  // 构造函数，初始化source_和lineMap_

  const SourceRef& getSourceRef() const {
    return source_;
  }
  // 返回源码引用对象source_

  const LineMap& getLineMap() const {
    return lineMap_;
  }
  // 返回行号映射表lineMap_

 private:
  SourceRef source_;
  // 源码引用对象source_
  LineMap lineMap_;
  // 行号映射表lineMap_
};

/**
 * ScriptProfile is an underlying C++ implementation for TorchScript profiling.
 * The profiling section is specified by calling enable() and disable():
 *
 * ...
 * scriptProfile.enable();
 * ...
 * (scripts)
 * ...
 * scriptProfile.disable();
 * ...
 *
 * NOTE: you cannot attach the profiler while the script is running.
 *
 * To retrieve collected runtime data, users may call dumpStats() and do
 * arbitrary filtering on the data they want. Note that dumpStats() should
 * not be called inside a profiling section.
 * In general, stats are aggregated per source function body, and then by line
 * number.
 */
class TORCH_API ScriptProfile : public CustomClassHolder {
  // 按函数源码ID和行号聚合数据点的数据结构
  using LineMap = std::map<int64_t, InstructionStats>;
  // 使用std::map定义LineMap，键为行号，值为InstructionStats对象

  using SourceMap = std::map<SourceRef, LineMap, std::less<>>;
  // 使用std::map定义SourceMap，键为SourceRef对象，值为LineMap对象，按SourceRef的比较器排序

 public:
  void enable();
  // 启用性能分析

  void disable();
  // 禁用性能分析

  const SourceMap& dumpStats();
  // 导出已收集的运行时数据

  void addDatapoint(std::shared_ptr<profiling::Datapoint>);
  // 添加数据点到分析器中

  ~ScriptProfile() override;
  // 虚析构函数，清理资源

 private:
  bool enabled_{false};
  // 标记性能分析是否启用，默认为false
  std::vector<std::shared_ptr<profiling::Datapoint>> datapoints_;
  // 存储数据点的容器，使用shared_ptr管理资源
  SourceMap sourceMap_;
  // 源码映射表，按源码引用SourceRef聚合数据
};

} // namespace torch::jit
```