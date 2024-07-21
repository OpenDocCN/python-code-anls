# `.\pytorch\aten\src\ATen\mps\MPSProfiler.h`

```
// 版权声明，指明此代码的版权归属于苹果公司，2022年
#pragma once

// 引入 ATen 库中的 Tensor 类定义
#include <ATen/Tensor.h>
// 引入 ATen 库中的 MPSStream 类定义
#include <ATen/mps/MPSStream.h>
// 引入 ATen 库中的 MPSAllocatorInterface 接口定义
#include <ATen/mps/MPSAllocatorInterface.h>

// 引入操作系统的标志点和日志库
#include <os/signpost.h>
#include <os/log.h>

// 引入标准库头文件
#include <atomic>               // 原子操作支持
#include <ctime>                // 时间操作支持
#include <sstream>              // 字符串流支持
#include <string>               // 字符串支持
#include <unordered_map>        // 无序映射支持
#include <utility>              // 实用工具支持

// at::mps 命名空间下的 Profiler 命名空间
namespace at::mps {

// Profiler 命名空间下的结构体 BaseInfo
namespace Profiler {

// BaseInfo 结构体
struct BaseInfo {
  // profiling info types，用于描述性能分析信息的类型
  enum class Type {
    GRAPH,          // 图形相关的性能信息
    KERNEL,         // 内核相关的性能信息
    COPY,           // 数据复制相关的性能信息
    CPU_FALLBACK,   // CPU 回退相关的性能信息
  };

  // 构造函数，初始化 profiling info 的类型、ID 和句柄
  BaseInfo(Type infoType, uint64_t Id, const uintptr_t Handle) :
      type(infoType), profileId(Id), handle(Handle) { }
  virtual ~BaseInfo() = default;

  // profiling info 的类型
  Type type;
  // 执行实例的唯一 profile ID
  uint64_t profileId;
  // 由 os_signpost 生成的事件和间隔的 signpost ID
  os_signpost_id_t eventSignpostId = 0, intervalSignpostId = 0;
  // 累积的 GPU 时间（毫秒）
  std::atomic<double> totalGpuTime{0.0};
  // 累积的调度时间（毫秒）
  std::atomic<double> totalSchedulingTime{0.0};
  // 操作或复制是否已完成的标志
  std::atomic_bool completed{false};
  // 用于标识 profile info 实例的句柄（通常是指针）
  const uintptr_t handle;

  // 虚函数，返回描述对象的字符串表示形式
  virtual const std::string toString(double gpuTime = 0, double schedulingTime = 0) const;

  // 构建张量的字符串表示形式（格式为：Device:ScalarType[tensor.sizes()]）
  static std::string buildTensorString(const Tensor& tensor, bool includeBufferId = false) {
    if (tensor.defined()) {
      std::stringstream tensorStr;
      auto deviceType = tensor.device().type();
      tensorStr << c10::DeviceTypeName(deviceType);
      // 根据标志位 INCLUDE_BUFFER_ID 决定是否包含缓冲区 ID
      if (includeBufferId && deviceType == at::kMPS) {
        // 从 tensor 的存储中获取 Metal 缓冲区
        id<MTLBuffer> buffer = __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
        tensorStr << "(buf#" << (getIMPSAllocator()->getBufferId(buffer))
                  << ":" << buffer.retainCount << ")";
      }
      tensorStr << ":"
                << tensor.scalar_type() << tensor.sizes();
      return tensorStr.str();
    } else {
      return "undefined";
    }
  }

  // 获取当前时间的静态方法
  static uint64_t getTime() {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
  }
};
// 定义一个继承自BaseInfo的结构体OperationInfo，用于表示操作的信息
struct OperationInfo : BaseInfo {
  // 构造函数，初始化操作信息，根据IsGraph确定类型，设置Id和Handle，初始化strKey
  OperationInfo(const void* Handle, bool IsGraph, uint64_t Id, const std::string& StrKey) :
      BaseInfo(IsGraph ? Type::GRAPH : Type::KERNEL, Id, uintptr_t(Handle)), strKey(StrKey) { }

  // 记录操作运行的次数
  uint64_t runCount = 0;
  // 操作关键字的字符串表示
  std::string strKey;

  // 重写父类虚函数，返回操作信息的字符串表示
  const std::string toString(double gpuTime = 0, double schedulingTime = 0) const override;

  // 静态方法，构建表示内核的字符串
  // kernelName为内核名称，tensors为张量列表，includeBufferId指定是否包含缓冲区ID
  static std::string buildKernelString(const std::string& kernelName,
                                       const TensorList& tensors,
                                       bool includeBufferId = false) {
    std::stringstream kernelStr;
    kernelStr << kernelName;
    // 遍历张量列表，构建字符串表示
    for (const Tensor& tensor: tensors) {
      kernelStr << ":" << BaseInfo::buildTensorString(tensor, includeBufferId);
    }
    return kernelStr.str();
  }
};

// 定义一个继承自BaseInfo的结构体CpuFbInfo，用于表示CPU回退的信息
struct CpuFbInfo : BaseInfo {
  // 构造函数，初始化CPU回退信息，设置Id和OpName
  CpuFbInfo(uint64_t Id, const std::string& OpName) :
      BaseInfo(Type::CPU_FALLBACK, Id, 0), opName(OpName) { }

  // 记录操作运行的次数
  uint64_t runCount = 0;
  // 记录当前和总的拷贝开销，以字节计
  size_t currentCopyOverhead = 0;
  size_t totalCopyOverhead = 0;
  std::string opName;
  std::string strKey;
  // 记录操作开始时间
  uint64_t startTime = 0;

  // 重写父类虚函数，返回CPU回退信息的字符串表示
  const std::string toString(double gpuTime = 0, double schedulingTime = 0) const override;

  // 更新拷贝开销信息，tensors为张量列表
  void updateCopyOverhead(const TensorList& tensors) {
    currentCopyOverhead = 0;
    // 遍历张量列表，累加拷贝开销
    for (const Tensor& tensor: tensors) {
      if (tensor.defined()) {
        currentCopyOverhead += tensor.nbytes();
      }
    }
    totalCopyOverhead += currentCopyOverhead;
  }
};

// 定义一个继承自BaseInfo的结构体CopyInfo，用于表示拷贝操作的信息
struct CopyInfo : BaseInfo {
  enum class Kind {
    MPS_TO_MPS,  // MPS到MPS的拷贝类型
    MPS_TO_CPU,  // MPS到CPU的拷贝类型
    CPU_TO_MPS,  // CPU到MPS的拷贝类型
  };

  // 构造函数，初始化拷贝信息，设置类型kind、长度length、Id和其他属性
  CopyInfo(const void* Handle, size_t Length, uint64_t Id, bool IsNonBlocking, bool UsesBlitter) :
           BaseInfo(Type::COPY, Id, uintptr_t(Handle)), kind(Kind::MPS_TO_MPS),
           length(Length), isNonBlocking(IsNonBlocking), usesBlitter(UsesBlitter) { }

  Kind kind;              // 拷贝类型
  size_t length;          // 拷贝长度
  bool isNonBlocking;     // 是否为非阻塞拷贝
  bool usesBlitter;       // 是否使用blitter进行拷贝
  std::string srcStrKey;  // 源操作关键字字符串
  std::string dstStrKey;  // 目标操作关键字字符串
  // 对于不使用blitter的拷贝，记录CPU时间
  uint64_t startTime = 0;

  // 重写父类虚函数，返回拷贝信息的字符串表示
  const std::string toString(double gpuTime = 0, double schedulingTime = 0) const override;

  // 静态方法，构建表示张量的字符串
  static std::string buildTensorString(const void* buffer, const OptionalTensorRef tensor, bool includeBufferId = false);

  // 静态方法，判断给定的buffer和tensor是否在MPS设备上存储
  static bool isStorageOnMPS(const void* buffer, const OptionalTensorRef tensor) {
    if (tensor.has_value()) {
      return tensor->device().type() == at::kMPS;
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(buffer);
    // 如果输入buffer不在MPS设备上，getUnalignedBufferSize()返回-1
    return getIMPSAllocator()->getUnalignedBufferSize(buffer) >= 0;
  }

  // 静态方法，获取拷贝的类型
  static Kind getCopyKind(const void* srcBuffer, const void* dstBuffer,
                          const OptionalTensorRef srcTensor, const OptionalTensorRef dstTensor) {
    // 检查源缓冲区和张量是否在 MPS 上
    const bool isSrcOnMPS = isStorageOnMPS(srcBuffer, srcTensor);
    // 检查目标缓冲区和张量是否在 MPS 上
    const bool isDstOnMPS = isStorageOnMPS(dstBuffer, dstTensor);
    // 在调试模式下，确保至少一个操作数在 MPS 上
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isSrcOnMPS || isDstOnMPS);
    // 如果源在 MPS 上且目标不在 MPS 上，返回 MPS_TO_CPU
    if (isSrcOnMPS && !isDstOnMPS) {
      return Kind::MPS_TO_CPU;
    } else if (!isSrcOnMPS && isDstOnMPS) { // 如果源不在 MPS 上且目标在 MPS 上，返回 CPU_TO_MPS
      return Kind::CPU_TO_MPS;
    }
    // 默认情况下，源和目标都在 MPS 上，返回 MPS_TO_MPS
    return Kind::MPS_TO_MPS;
}
};

// 定义一个结构体 CopyStat，继承自 CopyInfo
struct CopyStat : CopyInfo {
  // 显式构造函数，接受一个 std::string 类型的参数 CopyKindStr
  explicit CopyStat(std::string CopyKindStr) :
          // 调用基类 CopyInfo 的构造函数，传递参数 nullptr, 0, 0, false, false
          CopyInfo(nullptr, 0, 0, false, false), kindStr(std::move(CopyKindStr)) {}
  
  // 总拷贝次数
  size_t totalCount = 0;
  // 标量拷贝次数（小于 sizeof(int64)）
  size_t scalarsCount = 0;
  // 阻塞式拷贝次数（需要同步到 GPU）
  size_t blockingCount = 0;
  // 使用 memcpy() 而不是 Metal Blit Encoder 的拷贝次数
  size_t memcpyCount = 0;
  // 标量拷贝的 GPU 累计时间（毫秒）
  std::atomic<double> scalarsGpuTime{0.0};
  // 拷贝类型的字符串描述
  std::string kindStr;
};

// MPSProfiler 类的定义
class MPSProfiler {
public:
  // Profiler 选项的枚举，低 16 位用于选项设置
  enum ProfileOptions : uint32_t {
    OPTIONS_NONE = 0,
    // 跟踪所有标志点事件类型使用事件
    ALL_SIGNPOST_EVENTS    = (1 << 0),
    // 跟踪所有标志点事件类型使用时间间隔
    ALL_SIGNPOST_INTERVALS = (1 << 1),
    // 每次提交后等待命令缓冲区执行完毕
    WAIT_UNTIL_COMPLETED   = (1 << 2),
    // 对于基于间隔的标志点，包括图形/内核/拷贝执行的调度部分
    INCLUDE_SCHEDULE_INTERVAL = (1 << 3),

    // 如果需要单独跟踪标志点类型，则使用以下选项（很少需要）
    // 使用时间间隔跟踪标志点
    USE_INTERVALS = (1 << 4),
    // 通过事件发射跟踪标志点
    USE_EVENTS    = (1 << 5),
    // 用于健全性检查（当添加新选项时更改此值）
    OPTIONS_COUNT = (USE_EVENTS << 1) - 1,
  };

  // 标志点类型的枚举，使用高 16 位用于事件类型
  enum SignpostTypes : uint32_t {
    SIGNPOST_NONE = 0,
    // 跟踪 PyTorch 操作执行的标志点
    RUN_OPERATION = (1 << 16),
    // 跟踪 Blitter 拷贝的标志点
    BLIT_COPY     = (1 << 17),
    // 跟踪回退到 CPU 的操作的标志点
    CPU_FALLBACK  = (1 << 18),
    // 用于健全性检查（当添加新类型时更改此值）
    SIGNPOST_COUNT = (CPU_FALLBACK << 1) - 1,
  };

  enum LogOptions : uint32_t {
    LOG_NONE = 0,

    // 执行期间的信息日志选项
    // -------------------------------------
    // 打印操作信息（id/key/run_count）
    OPERATION_INFO      = (1 << 0),
    // 打印拷贝信息（src/dst tensors/buffers, size 等）
    COPY_INFO           = (1 << 1),
    // 打印 CPU 回退信息（id/runCount/opName/copyOverhead）
    CPU_FALLBACK_INFO   = (1 << 2),

    // 进程终止时的性能统计日志选项
    // ------------------------------------------------------------
    // 定义各种统计选项的位掩码，用于打印统计信息
    // 在进程终止前打印所有统计信息（操作统计、复制统计、CPU后备统计），方便不手动组合这些统计位标志
    ALL_STATS           = (1 << 3),

    // 打印操作统计信息（如GPU运行时间、运行次数等）在进程终止前
    OPERATION_STATS     = (1 << 4),

    // 打印复制统计信息（如GPU时间、复制类型、大小等）在进程终止前
    COPY_STATS          = (1 << 5),

    // 打印CPU后备统计信息（如CPU时间、运行时间、MPS<->CPU复制的大小等）在进程终止前
    CPU_FALLBACK_STATS  = (1 << 6),

    // 日志信息时的元数据格式选项
    // ---------------------------------------------

    // 如果启用，将GPU运行时间包含在元数据中（例如来自Metal命令缓冲区的GPUEndTime-GPUStartTime）（例如，[GPU=0.324 ms]）
    INCLUDE_GPU_TIME    = (1 << 7),

    // 如果启用，将GPU调度时间单独包含在元数据中（例如来自Metal命令缓冲区的KernelEndTime-KernelStartTime）（例如，[GPU=0.324 ms, KRNL=0.036 ms]）
    INCLUDE_KERNEL_TIME = (1 << 8),

    // 如果启用，将分配在MPSAllocator上的张量的唯一缓冲区ID包含在元数据中。这对于（与EV“PYTORCH_DEBUG_MPS_ALLOCATOR”一起）标识参与各种操作的缓冲区非常有用。
    INCLUDE_BUFFER_ID   = (1 << 9),

    // 用于检查健全性（当添加新选项时更改此值）
    LOG_COUNT = (INCLUDE_BUFFER_ID << 1) - 1,
  };  


// 定义 LOG_COUNT 常量，其值为 (INCLUDE_BUFFER_ID << 1) - 1
// 这里使用位移操作符将 INCLUDE_BUFFER_ID 左移一位，并减去 1 来计算 LOG_COUNT 的值



  explicit MPSProfiler();
  ~MPSProfiler();


// 显式声明 MPSProfiler 的构造函数和析构函数
// MPSProfiler 类的构造函数和析构函数的声明



  // the handle is either "MPSGraph*" or "id<MTLComputePipelineState>" for Metal Kernels
  // the beginProfile*() functions return a profileId which is unique per graph/kernel/copy
  uint64_t beginProfileKernel(const void* handle, const std::string& strKey, bool isGraph);
  uint64_t beginProfileKernel(const void* handle, const std::string& kernelName, const TensorList& tensors);
  uint64_t beginProfileCopy(const void* srcBuffer, const void* dstBuffer,
                            const OptionalTensorRef srcTensor,
                            const OptionalTensorRef dstTensor,
                            size_t length, bool isNonBlocking, bool usesBlitter = true);
  uint64_t beginProfileCPUFallback(const std::string& opName, const TensorList& tensors);
  void beginProfileGPUInterval(const void* handle);


// 下面几个函数用于开始对不同类型操作进行性能分析
// beginProfileKernel() 函数接受一个 handle 和一个描述键以及一个布尔值，返回一个唯一的 profileId
// beginProfileKernel() 函数接受一个 handle、一个内核名称和张量列表，返回一个唯一的 profileId
// beginProfileCopy() 函数接受源缓冲区、目标缓冲区、可选的源张量和目标张量等参数，返回一个唯一的 profileId
// beginProfileCPUFallback() 函数接受一个操作名称和张量列表，返回一个唯一的 profileId
// beginProfileGPUInterval() 函数接受一个 handle，用于开始 GPU 的时间间隔性能分析



  void endProfileCopy(uint64_t profileId, SyncType syncType);
  void endProfileKernel(const void* handle, SyncType syncType = SyncType::NONE);
  void endProfileCPUFallback(const std::string& opName);


// 下面几个函数用于结束不同类型操作的性能分析
// endProfileCopy() 函数接受一个 profileId 和同步类型参数，用于结束拷贝操作的性能分析
// endProfileKernel() 函数接受一个 handle 和可选的同步类型参数，用于结束内核操作的性能分析
// endProfileCPUFallback() 函数接受一个操作名称，用于结束 CPU 回退操作的性能分析



  // these are used to hook into Python bindings for torch.mps.profiler module.
  // this enables generating OS Signpost traces from MPSProfiler on-demand
  // during runtime (instead of environment variables).
  // The "mode" could be either "interval", "event", or both "interval,event"
  // for interval-based and/or event-based signpost tracing.
  void StartTrace(const std::string& mode, bool waitUntilCompleted);
  void StopTrace();


// 下面几个函数用于与 Python 绑定进行连接，以便生成 OS Signpost 跟踪信息
// StartTrace() 函数接受一个模式字符串和一个布尔值，用于开始跟踪操作
// StopTrace() 函数用于停止跟踪操作
// 这些函数使得可以在运行时根据需求生成 MPSProfiler 的 OS Signpost 跟踪信息，而不是依赖环境变量



  // Abstractions for GPU trace capturing
  bool isCaptureEnabled() const;
  bool isCapturing() const;
  void startCapture(const std::string& name, MPSStream* stream = nullptr);
  void stopCapture(MPSStream* stream = nullptr);


// 下面几个函数提供了 GPU 跟踪捕获的抽象接口
// isCaptureEnabled() 函数用于检查 GPU 跟踪是否启用
// isCapturing() 函数用于检查当前是否正在进行 GPU 跟踪捕获
// startCapture() 函数用于开始指定名称和流的 GPU 跟踪捕获
// stopCapture() 函数用于停止指定流的 GPU 跟踪捕获



  // convenience functions to indicate whether signpost tracing or
  // logging are enabled for the SignpostTypes
  bool isOperationProfilingEnabled() const {
    return (m_signpost_types & SignpostTypes::RUN_OPERATION) ||
           (m_log_options & (LogOptions::OPERATION_INFO | LogOptions::OPERATION_STATS));
  }
  bool isCopyProfilingEnabled() const {
    return (m_signpost_types & SignpostTypes::BLIT_COPY) ||
           (m_log_options & (LogOptions::COPY_INFO | LogOptions::COPY_STATS));
  }
  bool isCPUFallbackProfilingEnabled() const {
    return (m_signpost_types & SignpostTypes::CPU_FALLBACK) ||
           (m_log_options & (LogOptions::CPU_FALLBACK_INFO | LogOptions::CPU_FALLBACK_STATS));
  }
  bool isSignpostTracingEnabled() const {


// 下面几个函数用于检查不同类型的性能分析是否启用
// isOperationProfilingEnabled() 函数检查运行操作的性能分析是否启用
// isCopyProfilingEnabled() 函数检查拷贝操作的性能分析是否启用
// isCPUFallbackProfilingEnabled() 函数检查 CPU 回退操作的性能分析是否启用
// isSignpostTracingEnabled() 函数检查 Signpost 跟踪是否启用
};

} // namespace Profiler

// 获取 MPSProfiler 的实例引用
Profiler::MPSProfiler& getMPSProfiler();

// 结束 at::mps 命名空间
} // namespace at::mps
```