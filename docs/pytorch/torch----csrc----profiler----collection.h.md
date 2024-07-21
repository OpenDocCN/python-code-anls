# `.\pytorch\torch\csrc\profiler\collection.h`

```py
#pragma once
// 预处理指令：确保此头文件只被编译一次

#include <cstdint>
// 包含标准整数类型（如 uint8_t）

#include <memory>
// 包含智能指针的头文件

#include <mutex>
// 包含互斥量的头文件

#include <type_traits>
// 提供类型特性的头文件

#include <utility>
// 包含通用工具函数的头文件

#include <variant>
// 提供多态变量 std::variant 的支持

#include <ATen/Context.h>
// 包含 ATen 库的上下文信息头文件

#include <c10/core/Device.h>
// 包含 c10 库的设备信息头文件

#include <c10/core/TensorImpl.h>
// 包含 c10 库的张量实现头文件

#include <c10/macros/Macros.h>
// 包含 c10 库的宏定义头文件

#include <c10/util/ApproximateClock.h>
// 包含 c10 库的近似时钟头文件

#include <c10/util/flat_hash_map.h>
// 包含 c10 库的平面哈希映射头文件

#include <c10/util/strong_type.h>
// 包含 c10 库的强类型头文件

#include <torch/csrc/profiler/containers.h>
// 包含 Torch Profiler 模块的容器头文件

#include <torch/csrc/profiler/data_flow.h>
// 包含 Torch Profiler 模块的数据流头文件

#include <torch/csrc/profiler/events.h>
// 包含 Torch Profiler 模块的事件头文件

#include <torch/csrc/profiler/kineto_shim.h>
// 包含 Torch Profiler 模块的 Kineto Shim 头文件

#include <torch/csrc/profiler/orchestration/python_tracer.h>
// 包含 Torch Profiler 模块的 Python Tracer 头文件

#include <torch/csrc/profiler/perf.h>
// 包含 Torch Profiler 模块的性能头文件

#include <torch/csrc/profiler/stubs/base.h>
// 包含 Torch Profiler 模块的存根基类头文件

#include <torch/csrc/profiler/util.h>
// 包含 Torch Profiler 模块的实用工具函数头文件

#include <torch/csrc/utils/python_stub.h>
// 包含 Torch 实用工具函数的 Python 存根头文件

namespace torch::profiler::impl {

enum class EventType : uint8_t {
  TorchOp = 0,
  Backend,
  Vulkan,
  Allocation,
  OutOfMemory,
  PyCall,
  PyCCall,
  Kineto
};
// 枚举类型 EventType 定义：包含 Torch Profiler 的事件类型

// ============================================================================
// == Value (Tensor, Scalar) summary ==========================================
// ============================================================================

struct TORCH_API RawTensorMetadataBase {
  RawTensorMetadataBase() = default;
  // 默认构造函数：构造 RawTensorMetadataBase 对象

  explicit RawTensorMetadataBase(const at::Tensor& t);
  // 显式构造函数：从给定的 Tensor 对象 t 构造 RawTensorMetadataBase 对象

  StorageImplData data_;
  // 存储实现数据

  c10::ScalarType dtype_{c10::ScalarType::Undefined};
  // 标量类型，默认为未定义

  c10::Layout layout_{c10::Layout::Strided};
  // 布局类型，默认为步进布局

  uint32_t dim_{0};
  // 维度数，默认为 0
};

// Collected during profiling.
struct TORCH_API RawTensorMetadata : RawTensorMetadataBase {
  RawTensorMetadata() = default;
  // 默认构造函数：构造 RawTensorMetadata 对象

  RawTensorMetadata(const RawTensorMetadata&) = default;
  // 复制构造函数：从另一个 RawTensorMetadata 对象复制构造

  RawTensorMetadata(RawTensorMetadata&&) noexcept = default;
  // 移动构造函数：从另一个 RawTensorMetadata 对象移动构造

  RawTensorMetadata& operator=(const RawTensorMetadata&) = default;
  // 复制赋值运算符：复制另一个 RawTensorMetadata 对象

  RawTensorMetadata& operator=(RawTensorMetadata&&) noexcept = default;
  // 移动赋值运算符：移动另一个 RawTensorMetadata 对象

  explicit RawTensorMetadata(const at::Tensor& t);
  // 显式构造函数：从给定的 Tensor 对象 t 构造 RawTensorMetadata 对象

  // Wrap `weak_self_` in `std::optional` and split device into components to
  // keep struct default constructable. (which the std::array initializer needs)
  std::optional<WeakTensor> weak_self_;
  // 可选类型：弱引用张量对象

  c10::DeviceType device_type_{c10::DeviceType::CPU};
  // 设备类型，默认为 CPU

  c10::DeviceIndex device_index_{-1};
  // 设备索引，默认为 -1
};

// Used during post processing.
struct TORCH_API TensorMetadata : public RawTensorMetadataBase {
  TensorMetadata(
      const RawTensorMetadata& r,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides);
  // 构造函数：从给定的 RawTensorMetadata 对象 r、尺寸 sizes 和步长 strides 构造 TensorMetadata 对象

  TensorImplAddress impl() const {
    return weak_self_.get();
  }
  // 方法：返回张量实现的地址

  WeakTensor weak_self_;
  // 弱引用张量对象

  c10::Device device_;
  // 设备对象

  std::vector<int64_t> sizes_;
  // 尺寸向量

  std::vector<int64_t> strides_;
  // 步长向量

  // Set during `calculateUniqueTensorIDs`.
  std::optional<TensorID> id_;
  // 可选类型：张量 ID

  std::optional<AllocationID> allocation_id_;
  // 可选类型：分配 ID
};

using op_input_t = std::variant<
    TensorMetadata,
    std::vector<TensorMetadata>,
    c10::IValue,
    c10::nullopt_t>;
// 多态变量类型 op_input_t 定义：可以是 TensorMetadata 对象、张量元数据向量、c10::IValue 或空值

// ============================================================================
// == ExtraFields =============================================================
// ============================================================================
// 模板特化，定义了 EventType 为 TorchOp 时的额外字段结构体 ExtraFields
template <>
struct ExtraFields<EventType::TorchOp> : TorchOpBasicFields {
  // 构造函数，初始化 ExtraFields 的各个成员变量
  ExtraFields(
      TorchOpBasicFields&& f,                   // 基础字段的右值引用
      uint64_t correlation_id,                  // 相关性 ID
      c10::time_t end_time_ns,                  // 结束时间（纳秒）
      std::vector<op_input_t>&& inputs,         // 输入参数的右值引用
      std::vector<op_input_t>&& concrete_inputs, // 具体输入参数的右值引用
      jit_stack_t&& jit_stack,                  // JIT 栈的右值引用
      jit_modules_t&& jit_modules,              // JIT 模块的右值引用
      extra_args_t&& extra_args,                // 额外参数的右值引用
      extra_meta_t&& extra_meta,                // 额外元数据的右值引用
      FallbackPair&& device_fallback,           // 设备回退的右值引用
      bool allow_tf32_cublas,                   // 是否允许 TF32 CUBLAS
      std::unique_ptr<perf_counters_t>&& perf_event_counters) // 性能计数器的右值引用
      : TorchOpBasicFields(std::move(f)),       // 调用基类 TorchOpBasicFields 的移动构造函数
        correlation_id_{correlation_id},        // 初始化相关性 ID
        end_time_ns_{end_time_ns},              // 初始化结束时间（纳秒）
        inputs_{std::move(inputs)},             // 移动赋值输入参数
        concrete_inputs_{std::move(concrete_inputs)}, // 移动赋值具体输入参数
        jit_stack_{std::move(jit_stack)},       // 移动赋值 JIT 栈
        jit_modules_{std::move(jit_modules)},   // 移动赋值 JIT 模块
        extra_args_{std::move(extra_args)},     // 移动赋值额外参数
        extra_meta_{std::move(extra_meta)},     // 移动赋值额外元数据
        device_fallback_{std::move(device_fallback)}, // 移动赋值设备回退
        allow_tf32_cublas_{allow_tf32_cublas},  // 初始化是否允许 TF32 CUBLAS
        perf_event_counters_{std::move(perf_event_counters)} {} // 移动赋值性能计数器

  uint64_t correlation_id_;                    // 相关性 ID
  c10::time_t end_time_ns_;                    // 结束时间（纳秒）
  std::vector<op_input_t> inputs_;             // 输入参数列表
  std::vector<op_input_t> concrete_inputs_;    // 具体输入参数列表
  jit_stack_t jit_stack_;                      // JIT 栈
  jit_modules_t jit_modules_;                  // JIT 模块
  extra_args_t extra_args_;                    // 额外参数
  extra_meta_t extra_meta_;                    // 额外元数据
  FallbackPair device_fallback_;               // 设备回退
  bool allow_tf32_cublas_;                     // 是否允许 TF32 CUBLAS
  std::unique_ptr<perf_counters_t> perf_event_counters_; // 性能计数器的唯一指针
};

// ============================================================================
// 模板特化，定义了 EventType 为 Backend 时的额外字段结构体 ExtraFields
template <>
struct ExtraFields<EventType::Backend> {
  int64_t start_time_us_;                      // 开始时间（微秒）
  int64_t end_time_us_;                        // 结束时间（微秒）
  int64_t debug_handle_;                       // 调试句柄
  at::RecordScope scope_;                      // 记录范围
  std::string name_;                           // 名称
  std::string backend_;                        // 后端
  jit_stack_t jit_stack_;                      // JIT 栈
  jit_modules_t jit_modules_;                  // JIT 模块
};

// ============================================================================
// 模板特化，定义了 EventType 为 Vulkan 时的额外字段结构体 ExtraFields
template <>
struct ExtraFields<EventType::Vulkan> {
  using raw_event_t = std::pair<c10::approx_time_t, vulkan_id_t>; // 原始事件类型
  std::string name_;                           // 名称
  int64_t duration_ns_{0};                     // 持续时间（纳秒），默认为 0
  // 在构建事件树时，我们希望报告 Vulkan 事件的持续时间为 0，以免其结束时间超过其父 CPU 操作的结束时间
  bool in_tree_building_{false};               // 在构建事件树时的标志
};
// 定义一个结构体，表示原始分配的信息
struct RawAllocation {
  c10::approx_time_t start_time_; // 起始时间
  void* ptr_; // 指向分配内存的指针
  int64_t alloc_size_; // 分配的内存大小
  size_t total_allocated_; // 总共已分配的内存大小
  size_t total_reserved_; // 总共保留的内存大小
  c10::DeviceType device_type_; // 设备类型
  c10::DeviceIndex device_index_; // 设备索引
};

// 用于性能优化，确保 RawAllocation 是 POD（平凡数据类型）
static_assert(c10::is_pod_v<RawAllocation>, "Non-POD member of RawAllocation.");

// 对于 EventType::Allocation，定义额外的字段，继承自 RawAllocation
template <>
struct ExtraFields<EventType::Allocation> : RawAllocation {
  ExtraFields(const RawAllocation& allocation) : RawAllocation(allocation) {}

  c10::Device device() const {
    return {device_type_, device_index_}; // 返回设备类型和索引
  }

  std::optional<TensorID> id_; // 可选的张量 ID
  std::optional<AllocationID> allocation_id_; // 可选的分配 ID
};

// 对于 EventType::OutOfMemory，定义额外的字段
template <>
struct ExtraFields<EventType::OutOfMemory> {
  c10::approx_time_t start_time_; // 起始时间
  int64_t alloc_size_; // 分配的内存大小
  size_t total_allocated_; // 总共已分配的内存大小
  size_t total_reserved_; // 总共保留的内存大小
  c10::DeviceType device_type_; // 设备类型
  c10::DeviceIndex device_index_; // 设备索引
};

// 用于性能优化，确保 ExtraFields<EventType::OutOfMemory> 是 POD
static_assert(
    c10::is_pod_v<ExtraFields<EventType::OutOfMemory>>,
    "Non-POD member of ExtraFields<EventType::OutOfMemory>.");

// 表示 Python 框架的状态信息
struct PyFrameState {
  int line_no_; // 行号
  at::StringView filename_; // 文件名视图
  at::StringView funcname_; // 函数名视图
};

// 定义强类型模板 strong_t，用于标记不同的 Python 模块和方法
template <typename T, typename Tag>
using strong_t = strong::
    type<T, Tag, strong::regular, strong::convertible_to<T>, strong::hashable>;

using PyModuleSelf = strong_t<PyObject*, struct PyModuleSelf_>; // Python 模块自身的强类型
using PyModuleCls = strong_t<PyObject*, struct PyModuleCls_>; // Python 模块类的强类型
using PyMethod = strong_t</*PyMethodDef*/ void*, struct PyMethod_>; // Python 方法的强类型
using PyOptimizerSelf = strong_t<PyObject*, struct PyOptSelf_>; // Python 优化器自身的强类型
using PyOptimizerCls = strong_t<PyObject*, struct PyOptimizer_>; // Python 优化器类的强类型

// 表示神经网络模块的信息
struct NNModuleInfo {
  // 参数信息结构体
  struct ParameterInfo {
    std::string name_; // 参数名
    TensorMetadata metadata_; // 张量元数据
    std::optional<TensorMetadata> grad_metadata_; // 可选的梯度张量元数据
  };

  PyModuleSelf self_; // Python 模块自身
  PyModuleCls cls_; // Python 模块类
  at::StringView cls_name_; // 类名视图

  std::vector<ParameterInfo> parameters_; // 参数信息数组
  // 表示 `self_` 是 `cls_` 的第 k 个实例观察到
  size_t id_{std::numeric_limits<size_t>::max()}; // 实例 ID，默认为最大值
};

// 表示优化器信息
struct OptimizerInfo {
  // 参数信息结构体
  struct ParameterInfo {
    TensorMetadata metadata_; // 张量元数据
    std::optional<TensorMetadata> grad_metadata_; // 可选的梯度张量元数据
    std::vector<std::pair<std::string, TensorMetadata>> state_; // 状态信息数组
  };

  PyOptimizerSelf self_; // Python 优化器自身
  PyOptimizerCls cls_; // Python 优化器类
  at::StringView cls_name_; // 类名视图

  std::vector<ParameterInfo> parameters_; // 参数信息数组
};

// 表示 Python 额外字段的基类
struct PyExtraFieldsBase {
  PyExtraFieldsBase(
      c10::time_t end_time_ns, // 结束时间（纳秒）
      size_t python_tid, // Python 线程 ID
      PyFrameState caller) // 调用者的框架状态
      : end_time_ns_{end_time_ns},
        python_tid_{python_tid},
        caller_{std::move(caller)} {}

  c10::time_t end_time_ns_; // 结束时间（纳秒）
  size_t python_tid_; // Python 线程 ID
  PyFrameState caller_; // 调用者的框架状态

  // 第 k 个 Python 事件观察到（用于 TensorBoard）
  size_t id_{std::numeric_limits<size_t>::max()}; // 事件 ID，默认为最大值
};

// 对于 EventType::PyCall，定义额外的字段，继承自 PyExtraFieldsBase
template <>
struct ExtraFields<EventType::PyCall> : public PyExtraFieldsBase {
  struct args_t {
    PyFrameState frame_state_; // 框架状态
    std::optional<NNModuleInfo> module_info_; // 可选的神经网络模块信息
    // 声明一个可选的优化器信息对象，可能为空
    std::optional<OptimizerInfo> optimizer_info_;
  };

  // ExtraFields 类的构造函数，接收多个参数，用于初始化成员变量
  ExtraFields(
      c10::time_t end_time_ns,           // 结束时间的纳秒表示
      size_t python_tid,                 // Python 线程 ID
      PyFrameState caller,               // 调用者的帧状态对象
      args_t args)                       // 构造函数参数
      : PyExtraFieldsBase(end_time_ns, python_tid, std::move(caller)),  // 调用基类构造函数初始化基类成员变量
        callsite_{std::move(args.frame_state_)},  // 使用参数中的帧状态初始化 callsite_
        module_{std::move(args.module_info_)},    // 使用参数中的模块信息初始化 module_
        optimizer_{std::move(args.optimizer_info_)} {}  // 使用参数中的优化器信息初始化 optimizer_

  PyFrameState callsite_;              // 调用位置的帧状态对象
  std::optional<NNModuleInfo> module_; // 可选的神经网络模块信息对象，可能为空
  std::optional<OptimizerInfo> optimizer_;  // 可选的优化器信息对象，可能为空
};

template <>
struct ExtraFields<EventType::PyCCall> : public PyExtraFieldsBase {
  using args_t = at::StringView;

  // 构造函数，初始化 PyCCall 类型的附加字段
  ExtraFields(
      c10::time_t end_time_ns,  // 结束时间戳（纳秒）
      size_t python_tid,        // Python 线程 ID
      PyFrameState caller,      // 调用者的 Python 帧状态
      args_t args)              // 函数名的字符串视图
      : PyExtraFieldsBase(end_time_ns, python_tid, std::move(caller)),
        function_name_{std::move(args)} {}  // 初始化函数名字段

  at::StringView function_name_;  // 函数名字符串视图
};

template <>
struct ExtraFields<EventType::Kineto> {
  // Mirrors `libkineto::GenericTraceActivity::Flow`. This information is used
  // during post processing to properly embed Kineto events into the broader
  // profiler tree structure. End users are not generally expected to use these
  // fields directly, but they are available for debugging.
  
  // 定义 Flow 结构体，用于嵌入 Kineto 事件到更广泛的分析器树结构中
  struct Flow {
    uint32_t id{0};        // ID
    uint32_t type{0};      // 类型
    uint32_t start{0};     // 起始时间
  };

  std::string name_;                  // 名称
  int64_t duration_ns_{0};            // 持续时间（纳秒）
  uint64_t correlation_id_{0};        // 相关 ID
  libkineto::ActivityType activity_type_;  // 活动类型
  Flow flow;                          // Flow 结构体实例
  std::weak_ptr<Result> linked_activity_{};  // 弱引用指向 Result 实例
};

struct TORCH_API Result : public std::enable_shared_from_this<Result> {
  // 创建 Result 实例的静态工厂方法
  template <typename... Args>
  [[nodiscard]] static std::shared_ptr<Result> create(Args... args) {
    return std::shared_ptr<Result>(new Result(std::forward<Args>(args)...));
  }

  // 访问 extra_fields_ 的泛型访问方法
  template <typename T>
  decltype(auto) visit(T&& visitor) {
    return std::visit(std::forward<T>(visitor), extra_fields_);
  }

  // 访问 extra_fields_ 的泛型访问方法（const 版本）
  template <typename T>
  decltype(auto) visit(T&& visitor) const {
    return std::visit(std::forward<T>(visitor), extra_fields_);
  }

  // 如果 extra_fields_ 是 T 类型的基类，则访问对应的函数
  template <typename T, typename Fn>
  void visit_if_base(Fn&& fn) const {
    visit([&](const auto& extra_fields) {
      using extra_fields_t = typename std::remove_cv_t<
          typename std::remove_reference_t<decltype(extra_fields)>>;

      if constexpr (std::is_base_of_v<T, extra_fields_t>) {
        fn(extra_fields);
      }
    });
  }

  // 返回事件类型标签
  EventType tag() const {
    // 返回一个空的 lambda 函数，用于访问传入的空 vector
    return visit([](const auto& i) { return deduceTag(i); });
  }

  // 返回事件名称
  std::string name() const;
  
  // 返回 Kineto 活动类型
  libkineto::ActivityType kinetoType() const;
  
  // 返回关联 ID
  uint64_t correlationID() const;
  
  // 返回结束时间（纳秒）
  int64_t endTimeNS() const;
  
  // 返回结束线程 ID
  uint64_t endTID() const;
  
  // 返回设备类型
  c10::DeviceType deviceType() const;

  // 开始时间（纳秒）
  int64_t start_time_ns_;
  
  // 开始线程 ID
  uint64_t start_tid_;
  
  // Kineto 相关信息，包含设备和资源
  kineto::DeviceAndResource kineto_info_;
  
  // 存储不同类型事件的扩展字段，使用 std::variant
  std::variant<
      ExtraFields<EventType::TorchOp>,
      ExtraFields<EventType::Backend>,
      ExtraFields<EventType::Vulkan>,
      ExtraFields<EventType::Allocation>,
      ExtraFields<EventType::OutOfMemory>,
      ExtraFields<EventType::PyCall>,
      ExtraFields<EventType::PyCCall>,
      ExtraFields<EventType::Kineto>>
      extra_fields_;
  
  // 弱引用指向父结果的 shared_ptr
  std::weak_ptr<Result> parent_;
  
  // 子结果的 shared_ptr 数组
  std::vector<std::shared_ptr<Result>> children_;
  
  // 标记结果是否已完成，默认为 false
  bool finished_{false};

  // 指向 torch::profiler::impl::kineto::activity_t 的指针，默认为 nullptr
  const torch::profiler::impl::kineto::activity_t* kineto_activity_{nullptr};

 private:
  // 私有构造函数模板，用于初始化各个成员变量
  template <EventType E>
  Result(
      int64_t start_time_ns,
      uint64_t start_tid,
      kineto::DeviceAndResource kineto_info,
      ExtraFields<E>&& extra_fields)
      : start_time_ns_{start_time_ns},
        start_tid_{start_tid},
        kineto_info_{kineto_info},
        extra_fields_{std::move(extra_fields)} {}

  // 模板函数，根据 ExtraFields 的类型推断并返回对应的 EventType
  template <EventType E>
  static EventType deduceTag(const ExtraFields<E>&) {
    return E;
  }
};

// KinetoObserverContext 结构体，继承自 at::ObserverContext
struct KinetoObserverContext : public at::ObserverContext {
  // 定义 Event 结构体，用于存储事件相关信息
  struct Event {
    TorchOpBasicFields basic_fields_; // Torch 操作的基本字段
    c10::approx_time_t start_time_; // 开始时间戳

    // 在退出回调中设置的结束时间戳，默认为最小值
    c10::approx_time_t end_time_{std::numeric_limits<c10::approx_time_t>::min()};

    bool allow_tf32_cublas_; // 是否允许 TF32 的 cuBLAS 加速
    std::unique_ptr<perf_counters_t> counters_; // 性能计数器的唯一指针
  };

  // 构造函数，接受 Event 指针作为参数
  explicit KinetoObserverContext(Event* event) : event_{event} {}

  Event* event_; // 指向事件的指针
  FallbackPair* fallback_{nullptr}; // 回退对（FallbackPair）的指针，默认为 nullptr
};

// 默认的 IO_ENCODER_DEFAULT_BLOCK_SIZE 值为 1024
constexpr int IO_ENCODER_DEFAULT_BLOCK_SIZE = 1024;

// SCALAR_LIST_LENGTH_LIMIT 常量定义为 30，用于限制标量列表的长度
constexpr int SCALAR_LIST_LENGTH_LIMIT = 30;

// InputOutputEncoder 类
// 存储每个操作事件的形状、数据类型和具体值，以连续的 AppendOnlyList 形式存储，
// 避免在每个操作上创建形状和数据类型的向量。这些向量可以在后处理期间创建。
// 将数据分为两个类别：输入形状和具体输入。
class InputOutputEncoder final {
 public:
  // 将值压入编码器
  void push(c10::ArrayRef<const c10::IValue> values);

  // 用于后处理期间解包编码数据的方法
  // 每个方法返回一个“供应商” lambda，该 lambda 不带参数；
  // 调用一次 lambda 将返回一个参数列表，表示一个操作的输入。
  // 数据分为两个流：“输入形状”和“具体输入”。
  // 注意："auto" 只能工作因为这些方法只在 collection.cpp 中使用，那里已经实现。
  auto getInputShapeGenerator();
  auto getConcreteInputGenerator();

  // 检查是否支持标量列表的长度限制
  bool isSupportedScalarList(const c10::IValue& list_candidate);

  // 清除所有数据
  void clear();

  // 枚举类型 Tag 定义
  enum class Tag {
    Tensor = 0, // 张量
    UndefinedTensor, // 未定义的张量
    TensorListBegin, // 张量列表的起始（TODO：泛化到其他列表）
    ScalarList, // 标量列表
    Scalar, // 标量
    Other, // 其他
    TERMINATOR // 终结符
  };

  // 枚举类型 IOType 定义
  enum class IOType { Shapes, ConcreteInputs, None }; // 输入输出类型：形状、具体输入、无

 private:
  // 将张量推入编码器的私有方法
  void push(const at::Tensor& t);

  // getInputShapeGenerator 和 getConcreteInputGenerator 的实现细节
  auto getIValueGenerator(const IOType& io_type);

  // 各种 AppendOnlyList 类型成员变量
  AppendOnlyList<Tag, IO_ENCODER_DEFAULT_BLOCK_SIZE> tags_;
  AppendOnlyList<RawTensorMetadata, IO_ENCODER_DEFAULT_BLOCK_SIZE> tensor_metadata_;
  AppendOnlyList<int64_t, IO_ENCODER_DEFAULT_BLOCK_SIZE> tensor_sizes_strides_;
  AppendOnlyList<c10::IValue, IO_ENCODER_DEFAULT_BLOCK_SIZE> ivalues_;
};

// 定义 perf_profiler_t 类型为 linux_perf 的性能分析器
using perf_profiler_t = torch::profiler::impl::linux_perf::PerfProfiler;

// ThreadLocalSubqueue 类
class TORCH_API ThreadLocalSubqueue {
 public:
  // 构造函数，接受线程 ID 和配置对象作为参数
  ThreadLocalSubqueue(const uint64_t tid, ProfilerConfig config);

  // 开始操作的方法，返回一个唯一的 KinetoObserverContext 对象指针
  std::unique_ptr<KinetoObserverContext> begin_op(const at::RecordFunction& fn);

  // 向 backend_events_ 添加事件的方法
  template <class... Args>
  void emplace_backend_event(Args&&... args) {
    backend_events_.emplace_back(std::forward<Args>(args)...);
  }

  // 向 vulkan_events_ 添加事件的方法
  template <class... Args>
  void emplace_vulkan_event(Args&&... args) {
    vulkan_events_.emplace_back(std::forward<Args>(args)...);
  }

  // 向 allocation_events_ 添加事件的方法
  template <class... Args>
  void emplace_allocation_event(Args&&... args) {
    allocation_events_.emplace_back(std::forward<Args>(args)...);
  }
  
  // 省略了部分代码...
    // 将传入参数 args 转发到 allocations_ 的末尾
    allocations_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  // 将传入参数 args 转发到 ooms_ 的末尾
  void emplace_ooms_event(Args&&... args) {
    ooms_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  // 将传入参数 args 转发到 py_calls_ 的末尾
  void emplace_py_call(Args&&... args) {
    py_calls_.emplace_back(std::forward<Args>(args)...);
  }

  // 返回 tid_ 的值
  uint64_t tid() const {
    return tid_;
  }

  // 返回 kineto_info_ 的引用
  const kineto::DeviceAndResource& kineto_info() const {
    return kineto_info_;
  }

  // 调用 perf_profiler_ 的 Disable 方法来禁用性能分析器
  inline void disable_perf_profiler(perf_counters_t& counters) const {
    perf_profiler_->Disable(counters);
  }

 private:
  uint64_t tid_;  // 线程 ID
  ProfilerConfig config_;  // 分析器配置
  kineto::DeviceAndResource kineto_info_;  // 设备和资源信息
  std::unique_ptr<perf_profiler_t> perf_profiler_;  // 独占性能分析器指针

  friend class RecordQueue;
  // 见 `containers.h` 中的块大小基准
  static constexpr size_t BlockSize = 512;

  struct TorchOpStorage {
    // 注意：这是一个破坏性操作。
    void materialize(
        std::vector<std::shared_ptr<Result>>& out,
        const std::function<c10::time_t(c10::approx_time_t)>& time_converter,
        const uint64_t tid,
        const kineto::DeviceAndResource& kineto_info);

    template <typename T, size_t ChunkSize>
    // EventBlock 类继承自 std::array<T, ChunkSize>
    class EventBlock : public std::array<T, ChunkSize> {
     public:
      EventBlock();  // 构造函数
      // 返回指针 ptr 对应的关联 ID
      uint64_t correlation_id(const T* ptr) const;

     private:
      uint64_t id_start_;  // 起始 ID
    };

    using event_t = KinetoObserverContext::Event;
    // OpList 类继承自 AppendOnlyList<event_t, BlockSize, EventBlock>
    class OpList : public AppendOnlyList<event_t, BlockSize, EventBlock> {
     public:
      template <class... Args>
      // 将传入参数 args 转发到 OpList 的末尾，并返回插入的事件和其 ID
      std::pair<event_t*, uint64_t> emplace_back(Args&&... args);
      // 返回迭代器 e 对应事件的关联 ID
      static uint64_t correlationID(const OpList::Iterator& e);
    } op_events_;

    // 报告输入输出形状
    InputOutputEncoder inputs_outputs_;

    // 带有堆栈的 JIT
    AppendOnlyList<jit_stack_t, BlockSize> jit_stack_;

    // 带有模块的 JIT
    AppendOnlyList<jit_modules_t, BlockSize> jit_modules_;

    // 带有 FLOPS 的附加参数
    AppendOnlyList<extra_args_t, BlockSize> extra_args_;

    // 报告额外元数据，如集体通信元数据
    AppendOnlyList<extra_meta_t, BlockSize> extra_meta_;

    // ProfilerState::KINETO_GPU_FALLBACK 或 ProfilerState::KINETO_PRIVATEUSE1_FALLBACK
    AppendOnlyList<FallbackPair, BlockSize> device_fallback_;
  } torch_ops_;

  // 报告后端事件到活动 Kineto 分析器
  AppendOnlyList<ExtraFields<EventType::Backend>, BlockSize> backend_events_;

  // 报告 Vulkan 事件到分析器
  AppendOnlyList<ExtraFields<EventType::Vulkan>::raw_event_t, BlockSize>
      vulkan_events_;

  // 报告内存使用情况
  AppendOnlyList<RawAllocation, BlockSize> allocations_;

  // 报告内存溢出事件
  AppendOnlyList<ExtraFields<EventType::OutOfMemory>, BlockSize> ooms_;

  // 带有堆栈的 Python 调用
  AppendOnlyList<
      std::pair<python_tracer::TraceKey, c10::approx_time_t>,
      BlockSize>
      py_calls_;
};

// RecordQueue 类的实现，用于管理和处理记录队列
class TORCH_API RecordQueue {
 public:
  // 构造函数，接受 ProfilerConfig 和活动类型集合作为参数
  RecordQueue(ProfilerConfig config, std::set<ActivityType> activities);

  // 返回是否跟踪 Python
  bool tracePython() const;

  // 获取当前线程的子队列
  ThreadLocalSubqueue* getSubqueue();

  // 停止记录队列的操作
  void stop();

  // 获取记录，同时返回结果和活动追踪包装器
  // 参数包括时间转换函数、起始时间和结束时间
  std::pair<
      std::vector<std::shared_ptr<Result>>,
      std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>>
  getRecords(
      std::function<c10::time_t(c10::approx_time_t)> time_converter,
      uint64_t start_time_ns,
      uint64_t end_time_ns);

 private:
  uint32_t id_;  // 队列 ID
  ProfilerConfig config_;  // 分析器配置
  std::set<ActivityType> activities_;  // 活动类型集合
  ska::flat_hash_map<uint64_t, std::unique_ptr<ThreadLocalSubqueue>>
      sub_queues_;  // 线程本地子队列映射
  std::mutex sub_queue_mutex_;  // 子队列操作互斥锁
  std::unique_ptr<python_tracer::PythonTracerBase> python_tracer_;  // Python 跟踪器的唯一指针
};

// 获取具体输入记录是否启用的函数
TORCH_API bool get_record_concrete_inputs_enabled();

// 设置具体输入记录是否启用的函数
TORCH_API void set_record_concrete_inputs_enabled_fn(std::function<bool()>);

// 设置具体输入记录是否启用的值
TORCH_API void set_record_concrete_inputs_enabled_val(bool);

// 获取前向和后向传播记录是否启用的函数
TORCH_API bool get_fwd_bwd_enabled();

// 设置前向和后向传播记录是否启用的函数
TORCH_API void set_fwd_bwd_enabled_fn(std::function<bool()>);

// 设置前向和后向传播记录是否启用的值
TORCH_API void set_fwd_bwd_enabled_val(bool);

// 获取 CUDA 同步记录是否启用的函数
TORCH_API bool get_cuda_sync_enabled();

// 设置 CUDA 同步记录是否启用的函数
TORCH_API void set_cuda_sync_enabled_fn(std::function<bool()>);

// 设置 CUDA 同步记录是否启用的值

TORCH_API void set_cuda_sync_enabled_val(bool);

} // namespace torch::profiler::impl
```