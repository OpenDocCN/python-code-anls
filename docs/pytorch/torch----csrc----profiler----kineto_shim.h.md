# `.\pytorch\torch\csrc\profiler\kineto_shim.h`

```py
#pragma once

#include <memory>
#include <string>

// Skip Kineto dependency on mobile unless explicitly asked for.
// When is it explicitly asked for?
//   KinetoEdgeCPUProfiler uses KinetoProfiler for cpu
//   event profiling. This has a dependency on cpu only libkineto
#if defined(USE_KINETO) && defined(C10_MOBILE) && \
    !defined(EDGE_PROFILER_USE_KINETO)
#undef USE_KINETO
#endif

#include <ActivityType.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/profiler/api.h>

#ifdef USE_KINETO
// Forward declarations so we don't have to include `libkineto.h` in a header.
namespace libkineto {
class GenericTraceActivity;
struct CpuTraceBuffer;
class ActivityTraceInterface;
} // namespace libkineto
#endif

namespace torch {
namespace profiler {

#ifdef USE_KINETO
// Indicates whether Kineto profiling is available or not
constexpr bool kKinetoAvailable{true};
#else
// Indicates Kineto profiling is not available
constexpr bool kKinetoAvailable{false};
#endif

namespace impl::kineto {

// ----------------------------------------------------------------------------
// -- Interface (Does not require Kineto) -------------------------------------
// ----------------------------------------------------------------------------

// Structure representing a device and resource identifier
struct DeviceAndResource {
  int32_t device;    // Device identifier
  int32_t resource;  // Resource identifier
};
// Returns the Kineto device and resource identifiers
const DeviceAndResource kineto_ids();

#ifdef USE_KINETO
// Typedefs for Kineto tracing components
using trace_t = libkineto::CpuTraceBuffer;
using interface_trace_t = libkineto::ActivityTraceInterface;
using activity_t = libkineto::GenericTraceActivity;
#else
// Dummy structures used when Kineto is not available
struct DummyTraceBuffer {};
struct DummyTraceInterface {};

using trace_t = DummyTraceBuffer;
using interface_trace_t = DummyTraceBuffer;
struct activity_t;
#endif // USE_KINETO

// Adds metadata to a Kineto activity
void addMetadata(
    activity_t* activity,
    const std::string& key,
    const std::string& value);

// Wrapper class for Kineto CPU trace buffer
struct TraceWrapper {
  TraceWrapper(const int64_t start_time, const std::string& name);  // Constructor
  TraceWrapper(TraceWrapper&&) = default;  // Move constructor
  TraceWrapper(const TraceWrapper&) = delete;  // Deleted copy constructor
  ~TraceWrapper();  // Destructor

  // Adds a CPU activity to the trace
  activity_t* addCPUActivity(
      const std::string& name,
      const libkineto::ActivityType type,
      const DeviceAndResource device_and_resource,
      const uint64_t correlation_id,
      const int64_t start_time,
      const int64_t end_time);

  // Transfers CPU trace data up to the specified end time
  void transferCpuTrace(int64_t end_time);

  explicit operator bool() const;  // Conversion operator to bool

  std::unique_ptr<trace_t>& get() {  // Getter for the CPU trace buffer
    return cpu_trace_;
  }

 private:
  std::unique_ptr<trace_t> cpu_trace_;  // Unique pointer to the CPU trace buffer
};

// Wrapper class for Kineto activity trace interface
struct ActivityTraceWrapper {
  explicit ActivityTraceWrapper(std::unique_ptr<interface_trace_t>&& trace);  // Constructor with move semantics
  ActivityTraceWrapper() = default;  // Default constructor
  ActivityTraceWrapper(ActivityTraceWrapper&&) = default;  // Move constructor
  ActivityTraceWrapper(const ActivityTraceWrapper&) = delete;  // Deleted copy constructor
  explicit operator bool() const;  // Conversion operator to bool
  void save(const std::string& path);  // Saves the activity trace to a specified path

  const std::unique_ptr<interface_trace_t>& get() {  // Getter for the activity trace interface
    return trace_;
  }

 private:
  std::unique_ptr<interface_trace_t> trace_;  // Unique pointer to the activity trace interface
#ifdef USE_KINETO
  bool saved_ = false; // 标记Kineto的保存操作是否已执行，保存操作是具有破坏性的
#endif
};

using ActivitySet = std::set<torch::autograd::profiler::ActivityType>;
void prepareTrace(
    const bool cpuOnly,
    const ActivitySet& activities,
    const torch::profiler::impl::ExperimentalConfig& config);
void startTrace();
ActivityTraceWrapper stopTrace();
void pushCorrelationId(uint64_t correlation_id);
void pushUserCorrelationId(uint64_t correlation_id);
void popCorrelationId();
void popUserCorrelationId();
void recordThreadInfo();
bool collectivesProfilerExists();

void logInvariantViolation(
    const std::string& assertion,
    const std::string& error,
    const std::string& profile_id,
    const std::string& group_profile_id);

} // namespace impl::kineto

} // namespace profiler

namespace autograd::profiler {
c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type);

// 向当前活动的分析器中添加元数据的JSON表示
TORCH_API void addMetadataJson(
    const std::string& key,
    const std::string& value);

// 启动分析器的一个步骤
TORCH_API void profilerStep();

} // namespace autograd::profiler

} // namespace torch
```