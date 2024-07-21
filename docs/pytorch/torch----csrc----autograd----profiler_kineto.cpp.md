# `.\pytorch\torch\csrc\autograd\profiler_kineto.cpp`

```py
#include <cstring>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/csrc/autograd/profiler_kineto.h>

#include <c10/macros/Export.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/overloaded.h>

#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/events.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#include <torch/csrc/profiler/perf.h>
#include <torch/csrc/profiler/standalone/itt_observer.h>
#include <torch/csrc/profiler/standalone/nvtx_observer.h>
#include <torch/csrc/profiler/standalone/privateuse1_observer.h>
#include <torch/csrc/profiler/util.h>

#include <ATen/Context.h>

#include <stdexcept>
#include <utility>

#ifdef USE_KINETO
#include <ApproximateClock.h>
#include <libkineto.h>
#include <time_since_epoch.h>

#ifndef _MSC_VER
// TODO: TO be removed, once this properly works from libkineto
// Literal copy-n-paste from third_party/kineto/libkineto/src/WeakSymbols.cpp
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library
// when cuPTI is linked statically For more details see
// https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type();
__attribute__((weak)) int acc_get_device_type() {
  throw std::runtime_error(
      "Dummy implementation of acc_get_device_type is not supposed to be called!");
}
} // extern "C"
#endif // _MSC_VER
#endif // USE_KINETO

namespace torch {
namespace autograd::profiler {

namespace {
// 获取当前时间的纳秒表示，根据是否使用 Kineto 进行选择性返回
inline int64_t getTimeNs() {
#ifdef USE_KINETO
  return libkineto::timeSinceEpoch(std::chrono::system_clock::now());
#else
  return c10::getTime();
#endif // USE_KINETO
}

// 使用的命名空间别名，以简化长命名的使用
using torch::profiler::impl::ActiveProfilerType;
using torch::profiler::impl::EventType;
using torch::profiler::impl::ExtraFields;
using torch::profiler::impl::get_record_concrete_inputs_enabled;
using torch::profiler::impl::ivalueListToStr;
using torch::profiler::impl::op_input_t;
using torch::profiler::impl::ProfilerStateBase;
using torch::profiler::impl::PyExtraFieldsBase;
using torch::profiler::impl::Result;
using torch::profiler::impl::shape;
using torch::profiler::impl::shapesToStr;
using torch::profiler::impl::stacksToStr;
using torch::profiler::impl::strListToStr;
using torch::profiler::impl::TensorMetadata;
using torch::profiler::impl::variantShapesToStr;

// 用于存储操作参数的结构体
struct OpArgData {
  bool hasData;
  std::vector<shape> shapes;
  std::vector<std::string> dtypes;
  std::vector<c10::IValue> concreteInputs;
  std::vector<std::vector<int64_t>> shapesForKinetoEvent;
  std::vector<shape> strides;
};

// 解析操作参数数据，生成 OpArgData 结构体
auto parseArgData(
    const std::vector<op_input_t>& input_shapes,
    const std::vector<op_input_t>& concreteInputs) {
  if (input_shapes.empty()) {
    // 返回一个 OpArgData 结构，表示操作的输入参数信息，初始状态设定为 false 和空值
    return OpArgData{false, {}, {}, {}, {}, {}};
  }

  // 创建用于存储输入张量形状、步长、数据类型和具体输入值的向量
  std::vector<shape> shapes(input_shapes.size());
  std::vector<shape> strides(input_shapes.size());
  std::vector<std::vector<int64_t>> shapesForKinetoEvent(input_shapes.size());
  std::vector<std::string> dtypes(input_shapes.size());
  std::vector<c10::IValue> concrete_inputs_list;

  // 遍历输入张量的元数据列表，填充上述向量
  for (const auto& i : c10::irange(input_shapes.size())) {
    std::visit(
        // 根据输入类型的不同，执行不同的操作
        c10::overloaded(
            // 如果是单个张量的元数据
            [&](const TensorMetadata& t) {
              shapes[i] = t.sizes_;
              shapesForKinetoEvent[i] = t.sizes_;
              dtypes[i] = std::string(scalarTypeToTypeMeta(t.dtype_).name());
              strides[i] = t.strides_;
            },
            // 如果是张量列表的元数据
            [&](const std::vector<TensorMetadata>& l) {
              std::vector<std::vector<int64_t>> shape;
              shape.reserve(l.size());
              std::vector<std::vector<int64_t>> stride;
              stride.reserve(l.size());
              // 遍历张量列表中的每个张量元数据
              for (const auto& t : l) {
                shape.emplace_back(t.sizes_);
                stride.emplace_back(t.strides_);
              }
              shapes[i] = shape;
              strides[i] = stride;
              dtypes[i] = "TensorList";
            },
            // 如果是标量的 c10::IValue
            [&](const c10::IValue&) { dtypes[i] = "Scalar"; },
            // 默认情况，什么也不做
            [&](const auto&) {}),
        // 对应当前索引的输入元数据
        input_shapes[i]);
  }

  // 如果记录了具体的输入值并且输入形状与具体输入的大小相同且不为空
  if (input_shapes.size() == concreteInputs.size() && !concreteInputs.empty()) {
    concrete_inputs_list.resize(input_shapes.size());

    // 遍历输入形状列表
    for (const auto& i : c10::irange(input_shapes.size())) {
      std::visit(
          // 根据具体输入值的类型执行不同操作
          c10::overloaded(
              // 如果是标量的 c10::IValue
              [&](const c10::IValue& val) { concrete_inputs_list[i] = val; },
              // 默认情况，什么也不做
              [&](const auto&) {}),
          // 对应当前索引的输入形状
          input_shapes[i]);
      std::visit(
          // 根据具体输入的类型执行不同操作
          c10::overloaded(
              // 如果是标量的 c10::IValue
              [&](const c10::IValue& val) {
                concrete_inputs_list[i] = val;
                dtypes[i] = "ScalarList";
              },
              // 默认情况，什么也不做
              [&](const auto&) {}),
          // 对应当前索引的具体输入
          concreteInputs[i]);
    }
  }

  // 返回 OpArgData 结构，包含填充后的各项数据：成功标志、形状、数据类型、具体输入列表、Kineto 事件形状、步长
  return OpArgData{
      true,
      shapes,
      dtypes,
      concrete_inputs_list,
      shapesForKinetoEvent,
      strides};
}

struct MetadataBase {
  /* implicit */ MetadataBase(const std::shared_ptr<Result>& result)
      : kinetoActivity_{result->kineto_activity_} {
    // 构造函数，初始化 kinetoActivity_ 成员变量为 result 的 kineto_activity_
    if (std::holds_alternative<ExtraFields<EventType::Kineto>>(
            result->extra_fields_)) {
      // 如果 result 的 extra_fields_ 包含 Kineto 类型的附加字段
      // 执行安全断言，确保没有 kinetoActivity_
      if (!(SOFT_ASSERT(!hasKinetoActivity()))) {
        // 如果有 kinetoActivity_，则将其置为 nullptr
        result->kineto_activity_ = nullptr;
      }
      // 更新 kinetoActivity_ 为 result 的 kineto_activity_
      kinetoActivity_ = result->kineto_activity_;
    }
  }

  void addMetadata(const std::string& key, const std::string& value) {
    // 添加元数据到 kinetoActivity_
    // 前提条件是 kinetoActivity_ 存在且 value 不为空
    if (kinetoActivity_ && !value.empty() && value != "\"\"") {
      torch::profiler::impl::kineto::addMetadata(
          // 强制类型转换，传递 const 指针到 kinetoActivity_
          const_cast<torch::profiler::impl::kineto::activity_t*>(
              kinetoActivity_),
          key,
          value);
    }
  }

  bool hasKinetoActivity() const {
    // 检查 kinetoActivity_ 是否存在
    return kinetoActivity_ != nullptr;
  }

 private:
  const torch::profiler::impl::kineto::activity_t* kinetoActivity_{nullptr};
};

struct AddTensorboardFields : public MetadataBase {
  AddTensorboardFields(
      const std::shared_ptr<Result>& result,
      KinetoEvent& kineto_event)
      : MetadataBase(result) {
    // 继承 MetadataBase 的构造函数
    result->visit(*this);
    // 获取模块层次结构，并添加到元数据中
    const auto module_hierarchy = kineto_event.moduleHierarchy();
    addMetadata("Module Hierarchy", stacksToStr(module_hierarchy.vec(), "."));
    // 获取调用堆栈，并添加到元数据中
    addMetadata("Call stack", stacksToStr(kineto_event.stack().vec(), ";"));

    // 访问 result 的附加字段，如果是 PyExtraFieldsBase 类型，则执行以下操作
    result->visit_if_base<PyExtraFieldsBase>([&, this](const auto& i) -> void {
      // 添加 Python 对象的 ID 到元数据中
      this->addMetadata("Python id", std::to_string(i.id_));

      std::optional<std::string> parent_id;
      std::shared_ptr<Result> parent = result->parent_.lock();
      // 获取 Python 对象的父级 ID，并添加到元数据中
      while (parent && !parent_id.has_value()) {
        parent->visit_if_base<PyExtraFieldsBase>(
            [&](const auto& j) { parent_id = std::to_string(j.id_); });
        parent = parent->parent_.lock();
      }
      this->addMetadata("Python parent id", parent_id.value_or("null"));
    });
  }

  // 处理 PyCall 类型的附加字段
  void operator()(const ExtraFields<EventType::PyCall>& py_call) {
    if (py_call.module_.has_value()) {
      // 如果 PyCall 附加字段包含模块 ID，则添加到元数据中
      addMetadata("Python module id", std::to_string(py_call.module_->id_));
    }
  }

  // 默认模板操作符，不执行任何操作
  template <typename T>
  void operator()(const T&) {}
};

struct AddGenericMetadata : public MetadataBase {
  AddGenericMetadata(
      std::shared_ptr<Result>& result,
      const torch::profiler::impl::ProfilerConfig* config)
      : MetadataBase(result), config_(config) {
    // 继承 MetadataBase 的构造函数
    result->visit(*this);
    // 访问 result 的附加字段，根据配置添加通用的元数据信息
    // 如果配置中设置了详细输出标志
    if (config->experimental_config.verbose) {
      // 遍历并访问每个 PyExtraFieldsBase 实例
      result->visit_if_base<PyExtraFieldsBase>(
          [&, this](const auto& i) -> void {
            // 添加 Python 线程信息到元数据中
            this->addMetadata("Python thread", std::to_string(i.python_tid_));
          });
    }
  }

  // 处理 TorchOp 类型的事件
  void operator()(ExtraFields<EventType::TorchOp>& op_event) {
    // 解析事件的参数数据
    const auto arg_data =
        parseArgData(op_event.inputs_, op_event.concrete_inputs_);

    // 如果参数数据有效
    if (arg_data.hasData) {
      // 如果启用记录具体输入
      if (get_record_concrete_inputs_enabled()) {
        // 添加输入维度信息到元数据中
        addMetadata("Input Dims", variantShapesToStr(arg_data.shapes));
        // 添加输入步幅信息到元数据中
        addMetadata("Input Strides", variantShapesToStr(arg_data.strides));
      } else {
        // 添加 Kineto 事件的输入维度信息到元数据中
        addMetadata("Input Dims", shapesToStr(arg_data.shapesForKinetoEvent));
      }
      // 添加输入类型信息到元数据中
      addMetadata("Input type", strListToStr(arg_data.dtypes));
      // 如果具体输入不为空
      if (!arg_data.concreteInputs.empty()) {
        // 添加具体输入信息到元数据中
        addMetadata(
            "Concrete Inputs", ivalueListToStr(arg_data.concreteInputs));
      }
    }

    // 添加额外元数据（如果有的话）
    for (const auto& [key, val] : op_event.extra_meta_) {
      addMetadata(key, val);
    }

    // 如果配置存在且性能事件列表非空
    if (config_ && !config_->experimental_config.performance_events.empty()) {
      // 获取性能事件名称列表
      auto& event_names = config_->experimental_config.performance_events;
      // 遍历性能事件计数器
      for (const auto i : c10::irange(op_event.perf_event_counters_->size())) {
        // 添加性能事件计数信息到元数据中
        addMetadata(
            event_names[i],
            std::to_string((*op_event.perf_event_counters_)[i]));
      }
    }

    // 如果事件有序列号（例如在训练期间）
    if (op_event.sequence_number_ >= 0) {
      // 添加前向操作线程 ID 到元数据中
      addMetadata("Fwd thread id", std::to_string(op_event.forward_tid_));
      // 添加序列号信息到元数据中
      addMetadata("Sequence number", std::to_string(op_event.sequence_number_));
    }
    // 添加记录函数 ID 到元数据中
    addMetadata(
        "Record function id", std::to_string(op_event.record_function_id_));
  }

  // 处理 Backend 类型的事件
  void operator()(ExtraFields<EventType::Backend>& backend_event) {
    // 如果后端信息不为空
    if (!backend_event.backend_.empty()) {
      // 添加后端信息到元数据中
      addMetadata("Backend", "\"" + backend_event.backend_ + "\"");
    }
  }

  // 处理 Allocation 类型的事件
  void operator()(const ExtraFields<EventType::Allocation>& alloc) {
    // 添加设备类型到元数据中
    addMetadata("Device Type", std::to_string((int8_t)alloc.device_type_));
    // 添加设备 ID 到元数据中
    addMetadata("Device Id", std::to_string(alloc.device_index_));
    // 添加地址信息到元数据中
    addMetadata("Addr", std::to_string(reinterpret_cast<intptr_t>(alloc.ptr_)));
    // 添加分配字节数到元数据中
    addMetadata("Bytes", std::to_string(alloc.alloc_size_));
    // 添加总分配字节数到元数据中
    addMetadata("Total Allocated", std::to_string(alloc.total_allocated_));
    // 添加总保留字节数到元数据中
    addMetadata("Total Reserved", std::to_string(alloc.total_reserved_));
  }

  // 处理 OutOfMemory 类型的事件
  void operator()(const ExtraFields<EventType::OutOfMemory>& alloc) {
    // 添加设备类型到元数据中
    addMetadata("Device Type", std::to_string((int8_t)alloc.device_type_));
    // 添加设备 ID 到元数据中
    addMetadata("Device Id", std::to_string(alloc.device_index_));
    // 添加分配字节数到元数据中
    addMetadata("Bytes", std::to_string(alloc.alloc_size_));
    // 添加总分配字节数到元数据中
    addMetadata("Total Allocated", std::to_string(alloc.total_allocated_));
  # 将字符串形式的总保留量添加到元数据中
  addMetadata("Total Reserved", std::to_string(alloc.total_reserved_));
}

template <typename T>
void operator()(const T&) {}

private:
/* To get names of the performance events */
# 用于获取性能事件的名称
const torch::profiler::impl::ProfilerConfig* config_;
};

// 定义结构体 KinetoThreadLocalState，继承自 ProfilerStateBase
struct KinetoThreadLocalState : public ProfilerStateBase {
  // 构造函数，接受配置和活动类型集合作为参数
  explicit KinetoThreadLocalState(
      const ProfilerConfig& config,
      std::set<torch::profiler::impl::ActivityType> activities)
      : ProfilerStateBase(config), // 调用基类 ProfilerStateBase 的构造函数
        startTime(getTimeNs()), // 初始化 startTime 为当前时间的纳秒表示
        recordQueue(config, std::move(activities)) {} // 初始化 recordQueue，使用给定配置和活动类型集合

  // 虚析构函数，使用默认实现
  ~KinetoThreadLocalState() override = default;

  // 静态成员函数 get，返回 KinetoThreadLocalState 指针
  static KinetoThreadLocalState* get(bool global) {
    // 调用基类的静态成员函数 get，并断言返回状态为空或为 KINETO 类型
    auto* state = ProfilerStateBase::get(/*global=*/global);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        state == nullptr ||
        state->profilerType() == ActiveProfilerType::KINETO);
    return static_cast<KinetoThreadLocalState*>(state);
  }

  // 重写基类的 profilerType 函数，返回活动的分析器类型为 KINETO
  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::KINETO;
  }

  // 报告 Vulkan 事件给分析器的方法
  void reportVulkanEventToProfiler(torch::profiler::impl::vulkan_id_t id) {
    // 如果未禁用分析且配置允许内存分析，则将 Vulkan 事件放入记录队列
    if (!config_.disabled()) {
      recordQueue.getSubqueue()->emplace_vulkan_event(
          c10::getApproximateTime(), id);
    }
  }

  // 报告内存使用情况给分析器的方法
  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      size_t total_allocated,
      size_t total_reserved,
      c10::Device device) override {
    // 如果配置允许内存分析且未禁用分析，则将内存分配事件放入记录队列
    if (config_.profile_memory && !config_.disabled()) {
      recordQueue.getSubqueue()->emplace_allocation_event(
          c10::getApproximateTime(),
          ptr,
          alloc_size,
          total_allocated,
          total_reserved,
          device.type(),
          device.index());
    }
  }

  // 报告内存耗尽情况给分析器的方法
  void reportOutOfMemory(
      int64_t alloc_size,
      size_t total_allocated,
      size_t total_reserved,
      c10::Device device) override {
    // 如果配置允许内存分析且未禁用分析，则将内存耗尽事件放入记录队列
    if (config_.profile_memory && !config_.disabled()) {
      recordQueue.getSubqueue()->emplace_ooms_event(
          c10::getApproximateTime(),
          alloc_size,
          total_allocated,
          total_reserved,
          device.type(),
          device.index());
    }
  }

  // 设置事件后处理回调函数的方法
  void setEventPostProcessingCallback(post_process_t&& cb) {
    eventPostProcessCb = std::move(cb);
  }

  // 完成跟踪并返回活动迹线对象的方法
  std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>
  finalizeTrace() {
    auto end_time = getTimeNs(); // 获取结束时间
    recordQueue.stop(); // 停止记录队列

    std::lock_guard<std::mutex> guard(state_mutex_); // 使用互斥锁保护状态
    auto converter = clockConverter.makeConverter(); // 创建时间转换器
#ifdef USE_KINETO
    libkineto::get_time_converter() = converter; // 设置时间转换器（特定条件下）
#endif
    auto records_and_trace =
        recordQueue.getRecords(std::move(converter), startTime, end_time); // 获取记录和迹线

    materializeOpEvents(records_and_trace.first); // 实现操作事件

    // 删除 kinetoEvents 中的 Python 函数事件，通过 stacks 属性暴露它们
    kinetoEvents.erase(
        std::remove_if(
            kinetoEvents.begin(),
            kinetoEvents.end(),
            [](const auto& i) { return i.isPythonFunction(); }),
        kinetoEvents.end());

    return std::move(records_and_trace.second); // 返回活动迹线对象的唯一指针
  }

  // 模板成员函数，调用回调函数
  template <typename T>
  void invokeCallback(T& t) {
    // 实现调用指定类型的回调函数对象

    // 实现调用指定类型的回调函数对象
    // 如果事件后处理回调存在，则调用该回调，并传递调试句柄、JIT 栈和JIT 模块
    if (eventPostProcessCb) {
      eventPostProcessCb(t.debug_handle_, t.jit_stack_, t.jit_modules_);
    }
  }

  // 将事件列表中满足条件的事件添加到事件树和 Kineto 事件列表中
  void materializeOpEvents(std::vector<std::shared_ptr<Result>>& events) {
    // 遍历事件列表中的每一个事件
    for (auto& e : events) {
      // 如果事件的父节点已过期且设备类型为 CPU
      if (e->parent_.expired() && e->deviceType() == c10::DeviceType::CPU) {
        // 将事件添加到事件树中
        eventTree.push_back(e);
      }

      // 如果事件已完成处理
      if (e->finished_) {
        // 根据事件类型不同，调用相应的回调函数
        e->visit(c10::overloaded(
            [this](ExtraFields<EventType::TorchOp>& i) { invokeCallback(i); },
            [this](ExtraFields<EventType::Backend>& i) { invokeCallback(i); },
            [](auto&) {}));

        // 将事件及其详细配置信息加入 Kineto 事件列表中
        kinetoEvents.emplace_back(e, config_.experimental_config.verbose);
        // 为事件添加 TensorBoard 字段
        AddTensorboardFields add_tb(e, kinetoEvents.back());
        // 为事件添加通用元数据
        AddGenericMetadata add_generic(e, &config_);

        // 在后处理完成后，不再安全使用活动信息
        e->kineto_activity_ = nullptr;
      }
    }
  }

  // 记录开始时间的时间戳
  uint64_t startTime;
  // 时间转换器，用于将近似时钟时间转换为 Unix 时间
  c10::ApproximateClockToUnixTimeConverter clockConverter;
  // Torch 分析器的记录队列
  torch::profiler::impl::RecordQueue recordQueue;
  // 存储 Kineto 事件的列表
  std::vector<KinetoEvent> kinetoEvents;
  // 存储实验性事件树节点的列表
  std::vector<experimental_event_t> eventTree;
  // 可选项，用于启用事件后处理的回调函数
  post_process_t eventPostProcessCb;
};

// 模板函数：在函数进入时执行的回调函数，返回一个 ObserverContext 对象的 unique_ptr
// 如果 use_global_state_ptr 为 true，则使用全局状态指针
template <bool use_global_state_ptr = false>
std::unique_ptr<at::ObserverContext> onFunctionEnter(
    const at::RecordFunction& fn) {
  // 获取当前线程的 KinetoThreadLocalState 指针
  auto state_ptr = KinetoThreadLocalState::get(use_global_state_ptr);
  // 如果状态指针为空，则返回空指针
  if (!state_ptr) {
    return nullptr;
  }
  // 调用 recordQueue 的 begin_op 方法开始记录函数操作
  return state_ptr->recordQueue.getSubqueue()->begin_op(fn);
}

// 模板函数：在函数退出时执行的回调函数
// @lint-ignore CLANGTIDY clang-diagnostic-unused-parameter
// 如果 use_global_state_ptr 为 true，则使用全局状态指针
template <bool use_global_state_ptr = false>
void onFunctionExit(
    const at::RecordFunction& fn,
    at::ObserverContext* ctx_ptr) {
  // 获取当前线程的 KinetoThreadLocalState 指针
  auto state_ptr = KinetoThreadLocalState::get(use_global_state_ptr);
  // 如果状态指针为空，则直接返回
  if (!state_ptr) {
    return;
  }
  // 获取状态对象的配置信息
  const auto& config = state_ptr->config();
  // 将 ObserverContext 指针转换为 KinetoObserverContext 指针
  auto* kineto_ctx_ptr =
      static_cast<torch::profiler::impl::KinetoObserverContext*>(ctx_ptr);
  // 断言转换后的指针不为空
  TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);
  // 设置事件的结束时间为近似时间
  kineto_ctx_ptr->event_->end_time_ = c10::getApproximateTime();
  // 如果配置中包含性能事件，则禁用性能分析器
  if (!config.experimental_config.performance_events.empty()) {
    state_ptr->recordQueue.getSubqueue()->disable_perf_profiler(
        *kineto_ctx_ptr->event_->counters_);
  }
  // 设置事件的基本字段的结束线程 ID
  kineto_ctx_ptr->event_->basic_fields_.end_tid_ =
      at::RecordFunction::currentThreadId();
  // 根据配置状态执行相应的处理逻辑
  if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
    try {
      // 获取 fallback 指针并记录 CUDA 事件
      auto fallback = kineto_ctx_ptr->fallback_;
      TORCH_INTERNAL_ASSERT(fallback != nullptr);
      torch::profiler::impl::cudaStubs()->record(
          nullptr, &fallback->device_event_end_, nullptr);
    } catch (const std::exception& e) {
      // 记录 CUDA 事件失败时输出警告信息
      LOG(WARNING) << "Failed to record CUDA event. " << e.what();
    }
  } else if (config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK) {
    // 获取 fallback 指针并记录私有使用事件 1
    auto fallback = kineto_ctx_ptr->fallback_;
    TORCH_INTERNAL_ASSERT(fallback != nullptr);
    torch::profiler::impl::privateuse1Stubs()->record(
        nullptr, &fallback->device_event_end_, nullptr);
  }

  // 根据函数作用域不同，弹出对应的关联 ID
  if (fn.scope() == at::RecordScope::USER_SCOPE) {
    torch::profiler::impl::kineto::popUserCorrelationId();
  } else {
    torch::profiler::impl::kineto::popCorrelationId();
  }
}

// 模板函数：推入性能分析回调函数
// 如果 use_global_callback 为 true，则使用全局回调
template <bool use_global_callback = false>
void pushProfilingCallbacks(const std::unordered_set<at::RecordScope>& scopes) {
  // 获取注册状态指针
  auto registration_state_ptr =
      KinetoThreadLocalState::get(use_global_callback);
  // 断言注册状态指针不为空
  TORCH_INTERNAL_ASSERT(registration_state_ptr, "Expected profiler state set");
  // 创建记录函数回调对象
  auto recordFunctionCallback =
      at::RecordFunctionCallback(
          onFunctionEnter<use_global_callback>,
          onFunctionExit<use_global_callback>)
          .needsInputs(registration_state_ptr->config().report_input_shapes)
          .scopes(scopes);

  // 如果 use_global_callback 为 true，则将全局回调添加到全局回调列表中
  if constexpr (use_global_callback) {
    registration_state_ptr->setCallbackHandle(
        at::addGlobalCallback(recordFunctionCallback));
  } else {
    // 否则将线程本地回调添加到线程本地回调列表中
    registration_state_ptr->setCallbackHandle(
        at::addThreadLocalCallback(recordFunctionCallback));
  }
}

// 结构体：性能分析状态信息
struct ProfilerStateInfo {
  std::shared_ptr<KinetoThreadLocalState> state_ptr;  // KinetoThreadLocalState 的共享指针
  std::unordered_set<at::RecordScope> scopes;         // 记录函数作用域的无序集合
};
// 声明一个空的共享指针，指向 ProfilerStateInfo 类型的对象，初始化为 nullptr
std::shared_ptr<ProfilerStateInfo> profiler_state_info_ptr{nullptr};

} // namespace

// 向活跃的 Kineto 分析器报告后端事件
void reportBackendEventToActiveKinetoProfiler(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const at::RecordScope scope,
    const std::string& event_name,
    const std::string& backend_name) {
  // 检查全局的 KinetoThreadLocalState 是否为空，不支持后处理回调
  TORCH_INTERNAL_ASSERT(
      KinetoThreadLocalState::get(/*global=*/true) == nullptr,
      "On-demand profiling does not support post processing callback");

  // 获取当前线程的 KinetoThreadLocalState
  auto state_ptr = KinetoThreadLocalState::get(/*global=*/false);
  // 如果 state_ptr 为空，直接返回
  if (!state_ptr) {
    return;
  }

  // 将后端事件推送到记录队列中
  state_ptr->recordQueue.getSubqueue()->emplace_backend_event(
      start_time_us,
      end_time_us,
      debug_handle,
      scope,
      event_name,
      backend_name);

  /* no support for input shapes now?
  if (config.report_input_shapes) {
    ctx_ptr->shapes = inputSizes(fn);
    ctx_ptr->dtypes = inputTypes(fn);
  }
  */
}

// 准备启动分析器
void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  // 如果配置状态是 NVTX 或 ITT，则直接返回
  if (config.state == ProfilerState::NVTX ||
      config.state == ProfilerState::ITT) {
    return;
  }
  // 检查配置状态是否是 Kineto 相关的状态
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
          config.state == ProfilerState::KINETO_GPU_FALLBACK ||
          config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK,
      "Supported only in Kineto profiler");
  
  // 准备启动跟踪器
  torch::profiler::impl::kineto::prepareTrace(
      /*cpuOnly=*/!(
          at::hasCUDA() || at::hasXPU() || at::hasMTIA() ||
          c10::get_privateuse1_backend() != "privateuseone"),
      activities,
      config.experimental_config);

  // 如果配置中包含性能事件
  if (!config.experimental_config.performance_events.empty()) {
    /* For now only CPU activity is supported */
    // 检查是否包含 CPU 活动类型
    TORCH_CHECK(
        activities.count(torch::autograd::profiler::ActivityType::CPU),
        "Cannot run cpu hardware profiler without CPU activities, please only use CPU activity type");
    /*
     * 发出警告并将非标准事件传递给后端
     * 如果事件不被支持，后端可以中止操作
     * TODO 如果至少有一个有效事件，我们应该优雅地丢弃无效事件吗？
     */
    auto is_standard_event = [](const std::string& event) -> bool {
      for (auto e : torch::profiler::ProfilerPerfEvents) {
        if (!std::strcmp(event.c_str(), e)) {
          return true;
        }
      }
      return false;
    };

    // 遍历性能事件列表，检查是否为非标准事件，并发出相应警告
    for (const auto& e : config.experimental_config.performance_events) {
      if (!is_standard_event(e)) {
        TORCH_WARN("Forwarding a non-standard CPU performance event : ", e);
      }
    }
  }
}

// 启用具有事件后处理的分析器
void enableProfilerWithEventPostProcess(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    post_process_t&& cb,
    // 检查是否配置状态不支持使用 NVTX 进行后处理回调
    TORCH_CHECK(
        config.state != ProfilerState::NVTX,
        "NVTX does not support post processing callback.");

    // 检查是否配置状态不支持使用 ITT 进行后处理回调
    TORCH_CHECK(
        config.state != ProfilerState::ITT,
        "ITT does not support post processing callback.");

    // 断言在全局范围内获取 KinetoThreadLocalState 的结果为空指针
    TORCH_INTERNAL_ASSERT(
        KinetoThreadLocalState::get(/*global=*/true) == nullptr,
        "On-demand profiling does not support post processing callback");

    // 启用分析器，配置为给定的配置、活动和作用域
    enableProfiler(config, activities, scopes);

    // 获取 KinetoThreadLocalState 的指针，并设置事件后处理回调函数
    auto state_ptr = KinetoThreadLocalState::get(config.global());
    state_ptr->setEventPostProcessingCallback(std::move(cb));
}

void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes) {
  // 检查是否包含 CPU 活动类型
  const auto has_cpu = activities.count(ActivityType::CPU);
  // 检查全局状态下是否已经存在 KinetoThreadLocalState
  TORCH_CHECK(
      KinetoThreadLocalState::get(/*global=*/config.global()) == nullptr,
      "Profiler is already enabled",
      (config.global() ? "." : " on this thread."));

  // 根据配置的状态选择相应的回调函数
  if (config.state == ProfilerState::NVTX) {
    // 使用 NVTX 进行性能分析，配置回调函数
    torch::profiler::impl::pushNVTXCallbacks(config, scopes);
    return;
  } else if (config.state == ProfilerState::ITT) {
    // 使用 ITT 进行性能分析，配置回调函数
    torch::profiler::impl::pushITTCallbacks(config, scopes);
    return;
  } else if (config.state == ProfilerState::PRIVATEUSE1) {
    // 使用 PRIVATEUSE1 进行性能分析，配置回调函数
    torch::profiler::impl::pushPRIVATEUSE1CallbacksStub(config, scopes);
    return;
  }

  // 检查是否为 Kineto 或者 Kineto GPU 回退状态，或者全局状态
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK ||
      config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK ||
      config.global());
  TORCH_CHECK(!activities.empty(), "No activities specified.");
  // 如果未包含 CPU 活动类型，则全局状态必须为 false
  TORCH_INTERNAL_ASSERT(
      has_cpu || !config.global(),
      "Ondemand profiling must enable CPU tracing");

  // 创建 KinetoThreadLocalState 对象，并推入线程本地状态栈
  auto state_ptr = std::make_shared<KinetoThreadLocalState>(config, activities);
  KinetoThreadLocalState::push(state_ptr);

  // 如果包含 CPU 活动类型，则根据全局状态选择相应的性能分析回调函数
  if (has_cpu) {
    config.global() ? pushProfilingCallbacks</*global=*/true>(scopes)
                    : pushProfilingCallbacks</*global=*/false>(scopes);
  }

  // 如果不是全局状态，则启动 Kineto 的跟踪
  if (!config.global()) {
    torch::profiler::impl::kineto::startTrace();
  }

  // 如果包含 CPU 活动类型，则创建 ProfilerStateInfo 对象
  if (has_cpu) {
    auto state_info_ptr = std::make_shared<ProfilerStateInfo>();
    state_info_ptr->state_ptr = state_ptr;
    state_info_ptr->scopes = scopes;
    profiler_state_info_ptr = state_info_ptr;
  }
}

// 检查是否在主线程中启用了性能分析器
bool isProfilerEnabledInMainThread() {
  return profiler_state_info_ptr != nullptr;
}

// 在子线程中启用性能分析器
void enableProfilerInChildThread() {
  auto state_info_ptr = profiler_state_info_ptr;
  // 检查主线程中是否已经启用了性能分析器
  TORCH_CHECK(state_info_ptr, "Profiler is not enabled in main thread.");
  // 检查当前线程是否已经启用了性能分析器
  TORCH_CHECK(
      KinetoThreadLocalState::get(/*global=*/false) == nullptr,
      "Profiler is already enabled in this thread.");

  // 推入主线程的性能分析器状态到当前线程
  KinetoThreadLocalState::push(state_info_ptr->state_ptr);
  // 根据全局状态选择相应的性能分析回调函数
  pushProfilingCallbacks</*global=*/false>(state_info_ptr->scopes);
}

// 在子线程中禁用性能分析器
void disableProfilerInChildThread() {
  // 从栈中弹出 Kineto 的性能分析状态
  auto state_ptr = ProfilerStateBase::pop();
  // 检查当前线程是否正在运行 Kineto 的性能分析
  TORCH_CHECK(
      state_ptr,
      "Can't disable Kineto profiler when it's not running in this thread");
  // 移除当前线程的性能分析回调函数
  state_ptr->removeCallback();
}
std::unique_ptr<ProfilerResult> disableProfiler() {
  // 释放 profiler_state_info_ptr 以通知子线程停止 profiling
  profiler_state_info_ptr = nullptr;

  // 弹出最新的 ProfilerStateBase 对象
  auto state_ptr = ProfilerStateBase::pop();
  // 获取状态配置信息
  const auto& config = state_ptr->config();
  // 检查状态是否有效，只有在特定的 profiling 状态下才能禁用 Kineto profiler
  TORCH_CHECK(
      state_ptr &&
          (config.state == ProfilerState::KINETO ||
           config.state == ProfilerState::KINETO_GPU_FALLBACK ||
           config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK ||
           config.state == ProfilerState::KINETO_ONDEMAND ||
           config.state == ProfilerState::NVTX ||
           config.state == ProfilerState::ITT ||
           config.state == ProfilerState::PRIVATEUSE1),
      "Can't disable Kineto profiler when it's not running");

  // 移除回调函数
  state_ptr->removeCallback();

  // 如果是全局状态，通过 std::static_pointer_cast 调用 finalizeTrace()
  if (state_ptr->config().global()) {
    (void)std::static_pointer_cast<KinetoThreadLocalState>(state_ptr)
        ->finalizeTrace();
    return std::make_unique<ProfilerResult>();
  }

  // 对于 NVTX、PRIVATEUSE1、KINETO、KINETO_GPU_FALLBACK、KINETO_PRIVATEUSE1_FALLBACK 共享的处理
  std::unique_ptr<ProfilerResult> result;
  if (state_ptr->config().state == ProfilerState::NVTX ||
      state_ptr->config().state == ProfilerState::PRIVATEUSE1) {
    result = std::make_unique<ProfilerResult>();
  }

  // 如果是 KINETO 系列状态，创建 ProfilerResult 对象
  if (config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK ||
      config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK) {
    auto kineto_state_ptr =
        std::static_pointer_cast<KinetoThreadLocalState>(state_ptr);
    auto trace = kineto_state_ptr->finalizeTrace();
    result = std::make_unique<ProfilerResult>(
        kineto_state_ptr->startTime,
        std::move(kineto_state_ptr->kinetoEvents),
        std::move(trace),
        std::move(kineto_state_ptr->eventTree));
  }

  return result;
}

KinetoEvent::KinetoEvent(
    const std::shared_ptr<const torch::profiler::impl::Result>& result,
    const bool verbose)
    : result_{result} {
  TORCH_INTERNAL_ASSERT(result != nullptr);

  // 如果 verbose 为 true，则填充 Python 栈信息
  if (verbose) {
    auto parent = result_->parent_.lock();
    while (parent != nullptr) {
      parent->visit_if_base<PyExtraFieldsBase>(
          [&](const auto&) { python_stack_.push_back(parent->name()); });
      parent = parent->parent_.lock();
    }
  }

  // 访问 result 中的 TorchOp 类型的额外字段，解析参数数据
  result->visit_if_base<ExtraFields<EventType::TorchOp>>([&](const auto& op) {
    auto arg_data = parseArgData(op.inputs_, op.concrete_inputs_);
    shapes_ = std::move(arg_data.shapesForKinetoEvent);
    dtypes_ = std::move(arg_data.dtypes);
    concrete_inputs_ = std::move(arg_data.concreteInputs);
  });
}

bool KinetoEvent::isPythonFunction() const {
  // 检查结果中是否包含 Python 函数信息
  bool out{false};
  result_->visit_if_base<PyExtraFieldsBase>([&](const auto&) { out = true; });
  return out;
}

bool KinetoEvent::hasShapes() const {
  // 检查是否有形状信息
  return !shapes_.empty();
}
// 返回事件的形状数组的引用
const c10::ArrayRef<std::vector<int64_t>> KinetoEvent::shapes() const {
  return shapes_;
}

// 检查事件是否具有数据类型
bool KinetoEvent::hasTypes() const {
  return !dtypes_.empty();
}

// 返回事件的数据类型数组的引用
const c10::ArrayRef<std::string> KinetoEvent::dtypes() const {
  return dtypes_;
}

// 检查事件是否具有具体输入
bool KinetoEvent::hasConcreteInputs() const {
  return !concrete_inputs_.empty();
}

// 返回事件的具体输入数组的引用
const c10::ArrayRef<c10::IValue> KinetoEvent::concreteInputs() const {
  return concrete_inputs_;
}

// 返回事件的堆栈数组的引用
const c10::ArrayRef<std::string> KinetoEvent::stack() const {
  // 获取事件的额外字段
  auto const& extra_fields = result_->extra_fields_;
  // 如果额外字段是Torch操作类型，则返回其JIT堆栈或Python堆栈
  if (auto p = std::get_if<ExtraFields<EventType::TorchOp>>(&extra_fields)) {
    return !p->jit_stack_.empty() ? p->jit_stack_ : python_stack_;
  }
  // 如果额外字段是后端类型，则返回其JIT堆栈或Python堆栈
  if (auto p = std::get_if<ExtraFields<EventType::Backend>>(&extra_fields)) {
    return !p->jit_stack_.empty() ? p->jit_stack_ : python_stack_;
  }
  // 否则返回Python堆栈
  return python_stack_;
}

// 返回事件的模块层次结构数组的引用
const c10::ArrayRef<std::string> KinetoEvent::moduleHierarchy() const {
  // 获取事件的额外字段
  auto const& extra_fields = result_->extra_fields_;
  // 如果额外字段是Torch操作类型，则返回其JIT模块列表
  if (auto p = std::get_if<ExtraFields<EventType::TorchOp>>(&extra_fields)) {
    return p->jit_modules_;
  }
  // 如果额外字段是后端类型，则返回其JIT模块列表
  if (auto p = std::get_if<ExtraFields<EventType::Backend>>(&extra_fields)) {
    return p->jit_modules_;
  }
  // 否则返回空数组
  return {};
}

// 返回事件的结束时间戳（纳秒）
uint64_t KinetoEvent::endNs() const {
  return result_->endTimeNS();
}

// 返回事件的持续时间（纳秒）
uint64_t KinetoEvent::durationNs() const {
  return (result_->endTimeNS() - result_->start_time_ns_);
}

// 返回事件的调试句柄
int64_t KinetoEvent::debugHandle() const {
  return result_->visit(c10::overloaded(
      // 如果是Torch操作类型，返回其调试句柄
      [](const ExtraFields<EventType::TorchOp>& i) { return i.debug_handle_; },
      // 如果是后端类型，返回其调试句柄
      [](const ExtraFields<EventType::Backend>& i) { return i.debug_handle_; },
      // 否则返回-1
      [](const auto&) -> int64_t { return -1; }));
}

// 返回事件的设备索引
int KinetoEvent::deviceIndex() const {
  return result_->visit(c10::overloaded(
      // 如果是分配类型事件，返回其设备索引转换为整数
      [](const ExtraFields<EventType::Allocation>& i) {
        return static_cast<int>(i.device_index_);
      },
      // 如果是内存不足类型事件，返回其设备索引转换为整数
      [](const ExtraFields<EventType::OutOfMemory>& i) {
        return static_cast<int>(i.device_index_);
      },
      // 否则返回事件的kineto_info_字段的设备索引转换为整数
      [&](const auto&) {
        return static_cast<int>(result_->kineto_info_.device);
      }));
}

// 检查事件是否具有堆栈信息
bool KinetoEvent::hasStack() const {
  return !stack().empty();
}

// 返回CUDA事件的经过时间（微秒）
int64_t KinetoEvent::cudaElapsedUs() const {
  auto cuda_event_start = fallbackStart();
  auto cuda_event_end = fallbackEnd();
  // 如果CUDA事件的开始或结束时间为空，则返回-1
  if (!cuda_event_start || !cuda_event_end) {
    return -1;
  }
  try {
    // 测量两个CUDA事件之间的时间，并返回结果（转换为微秒）
    return (int64_t)torch::profiler::impl::cudaStubs()->elapsed(
        &cuda_event_start, &cuda_event_end);
  } catch (std::exception& e) {
    // 捕获异常并记录警告信息
    LOG(WARNING) << "Failed to measure time between two CUDA events. "
                 << e.what();
  }
  // 发生异常或错误时返回-1
  return -1;
}

// 返回私有用途1事件的经过时间（微秒）
int64_t KinetoEvent::privateuse1ElapsedUs() const {
  auto privateuse1_event_start = fallbackStart();
  auto privateuse1_event_end = fallbackEnd();
  // 如果私有用途1事件的开始或结束时间为空，则返回-1
  if (!privateuse1_event_start || !privateuse1_event_end) {
    return -1;
  }
  // TODO: 未完成的代码块，需根据实际情况进一步补充
}
    // 如果条件不满足，返回 -1
    return -1;
  }
  // 调用私有使用的第一个存根函数来获取时间间隔，并返回其结果
  return (int64_t)torch::profiler::impl::privateuse1Stubs()->elapsed(
      &privateuse1_event_start, &privateuse1_event_end);
  // 在任何其他情况下，返回 -1（这行代码实际上不会被执行到）
  return -1;
}

// 从 `result_` 中获取性能事件计数器的值，并存入向量 `in` 中
void KinetoEvent::getPerfEventCounters(std::vector<uint64_t>& in) const {
  return result_->visit(c10::overloaded(
      // 处理 TorchOp 类型的事件
      [&in](const ExtraFields<EventType::TorchOp>& e) -> void {
        const size_t n = e.perf_event_counters_->size();
        // 这种情况应该很少见
        if (in.size() < n) {
          // 如果输入向量 `in` 的大小小于 `n`，则扩展向量并初始化为 0
          in.resize(n, 0);
        }
        // 将性能事件计数器的值复制到 `in` 向量中
        for (size_t i = 0; i < n; ++i) {
          in[i] = (*e.perf_event_counters_)[i];
        }
      },
      // 处理其他类型的事件，直接返回
      [](const auto&) -> void { return; }));
}

// 定义一个宏，用于从 `result_` 中获取特定方法的结果并转发
#define FORWARD_FROM_RESULT(method_name, result_expr)                        \
  decltype(std::declval<KinetoEvent>().method_name())                        \
  KinetoEvent::method_name() const {                                         \
    return static_cast<decltype(std::declval<KinetoEvent>().method_name())>( \
        result_->result_expr);                                               \
  }

// 下面一系列宏使用 `FORWARD_FROM_RESULT` 宏，分别获取 `KinetoEvent` 类的不同方法的结果并转发

// 大多数 `KinetoEvent` 中的字段只对单个事件类型有意义（通常是 TorchOp 类型）。
// 对于其他类型，它们只返回默认值。此宏提供了表达此行为的简洁方式。
#define TYPED_ATTR_WITH_DEFAULT(                                       \
    event_type, method_name, expression, default_value)                \
  decltype(std::declval<KinetoEvent>().method_name())                  \
  KinetoEvent::method_name() const {                                   \
    using out_t = decltype(std::declval<KinetoEvent>().method_name()); \
    // 根据事件类型 `event_type` 使用 `result_` 访问器，返回对应的值或默认值
    return result_->visit(c10::overloaded(                             \
        [](const ExtraFields<EventType::event_type>& e) -> out_t {     \
          return expression;                                           \
        },                                                             \
        [](const auto&) -> out_t { return default_value; }));          \
  }

// 定义一个简化版的 `TYPED_ATTR_WITH_DEFAULT` 宏，不提供默认值
#define TYPED_ATTR(event_type, method_name, expression) \
  TYPED_ATTR_WITH_DEFAULT(event_type, method_name, expression, {})

// 下面一系列宏使用 `TYPED_ATTR_WITH_DEFAULT` 宏，分别获取 `KinetoEvent` 类不同方法的结果并转发
    !e.extra_args_.empty()
        ? torch::profiler::impl::computeFlops(e.name_, e.extra_args_)
        : 0)


注释：


// 如果额外参数 e.extra_args_ 不为空，则调用 computeFlops 函数计算浮点运算量，并返回结果
!e.extra_args_.empty()
    ? torch::profiler::impl::computeFlops(e.name_, e.extra_args_)
    // 如果额外参数为空，则返回 0
    : 0)


这段代码是一个条件表达式，根据 `e.extra_args_.empty()` 是否为空来决定执行不同的操作：
- 如果 `e.extra_args_.empty()` 返回 `false`（即不为空），则调用 `torch::profiler::impl::computeFlops(e.name_, e.extra_args_)` 来计算某个事件 `e` 的浮点运算量。
- 如果 `e.extra_args_.empty()` 返回 `true`（即为空），则返回 `0`。
// 定义宏TYPED_ATTR，用于声明带类型的属性，并将其与Backend类的backend成员变量关联
TYPED_ATTR(Backend, backend, e.backend_)
// 定义宏TYPED_ATTR，用于声明带类型的属性，并将其与Allocation类的nBytes成员变量关联
TYPED_ATTR(Allocation, nBytes, e.alloc_size_)
// 定义宏TYPED_ATTR_WITH_DEFAULT，用于声明带默认值的带类型属性，并将其与Kineto类的linkedCorrelationId成员变量关联
TYPED_ATTR(Kineto, linkedCorrelationId, [&]() {
  // 获取弱引用所指向的对象（linked_activity_），如果存在则返回其关联的correlationID，否则返回0
  const auto linked = e.linked_activity_.lock();
  return linked ? linked->correlationID() : 0;
}())
// 取消宏定义TYPED_ATTR
#undef TYPED_ATTR
// 取消宏定义TYPED_ATTR_WITH_DEFAULT

// 构造函数：ProfilerResult类的构造函数，接受起始时间、事件列表、活动跟踪包装器和事件树作为参数
ProfilerResult::ProfilerResult(
    uint64_t start_time,
    std::vector<KinetoEvent> events,
    std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>&&
        trace,
    std::vector<experimental_event_t>&& event_tree)
    : trace_start_ns_(start_time), // 初始化跟踪开始时间
      events_(std::move(events)), // 初始化事件列表
      trace_(std::move(trace)),   // 移动赋值活动跟踪包装器
      event_tree_(std::move(event_tree)) {} // 移动赋值事件树
// 默认构造函数定义
ProfilerResult::ProfilerResult() = default;
// 默认析构函数定义
ProfilerResult::~ProfilerResult() = default;

// 方法：保存ProfilerResult对象的活动跟踪信息到指定路径
void ProfilerResult::save(const std::string& path) {
  trace_->save(path); // 调用活动跟踪包装器的保存方法
}

} // namespace autograd::profiler

// 命名空间：定义profiler::impl命名空间
namespace profiler::impl {
// 函数：将Vulkan事件报告给分析器
void _reportVulkanEventToProfiler(vulkan_id_t id) {
  // 获取当前线程的KinetoThreadLocalState对象的指针（不使用全局对象）
  auto state_ptr = ::torch::autograd::profiler::KinetoThreadLocalState::get(
      /*global=*/false);
  // 如果state_ptr不为空，则调用其reportVulkanEventToProfiler方法，将Vulkan事件id报告给分析器
  if (state_ptr) {
    state_ptr->reportVulkanEventToProfiler(id);
  }
}
} // namespace profiler::impl

} // namespace torch
```