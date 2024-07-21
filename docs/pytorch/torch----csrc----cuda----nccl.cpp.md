# `.\pytorch\torch\csrc\cuda\nccl.cpp`

```
#include <ATen/core/functional.h>
#include <torch/csrc/cuda/device_set.h>
#include <torch/csrc/cuda/nccl.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>

#include <nccl.h>

#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>

#if !defined(USE_ROCM) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 14)))
#define NCCL_HAS_COMM_NONBLOCKING 1
#endif

// 将 torch::cuda::nccl::ncclComm_t* 转换为 ncclComm_t*
ncclComm_t* to_nccl_comm(torch::cuda::nccl::ncclComm_t* var) {
  return reinterpret_cast<ncclComm_t*>(var);
}

// 将 torch::cuda::nccl::ncclComm_t 转换为 ncclComm_t
ncclComm_t to_nccl_comm(torch::cuda::nccl::ncclComm_t var) {
  return reinterpret_cast<ncclComm_t>(var);
}

// 将 torch::cuda::nccl::ncclUniqueId* 转换为 ncclUniqueId*
ncclUniqueId* to_nccl_unique_id(torch::cuda::nccl::ncclUniqueId* var) {
  return reinterpret_cast<ncclUniqueId*>(var);
}

// 将 torch::cuda::nccl::ncclResult 转换为 ncclResult_t
ncclResult_t to_nccl_result(torch::cuda::nccl::ncclResult var) {
  switch (var) {
    case torch::cuda::nccl::ncclResult::Success:
      return ncclResult_t::ncclSuccess;
    case torch::cuda::nccl::ncclResult::UnhandledCudaError:
      return ncclResult_t::ncclUnhandledCudaError;
    case torch::cuda::nccl::ncclResult::SystemError:
      return ncclResult_t::ncclSystemError;
    case torch::cuda::nccl::ncclResult::InternalError:
      return ncclResult_t::ncclInternalError;
    case torch::cuda::nccl::ncclResult::InvalidArgument:
      return ncclResult_t::ncclInvalidArgument;
    case torch::cuda::nccl::ncclResult::InvalidUsage:
      return ncclResult_t::ncclInvalidUsage;
    case torch::cuda::nccl::ncclResult::RemoteError:
      return ncclResult_t::ncclRemoteError;
#ifdef NCCL_HAS_COMM_NONBLOCKING
    case torch::cuda::nccl::ncclResult::InProgress:
      return ncclResult_t::ncclInProgress;
#endif
    case torch::cuda::nccl::ncclResult::NumResults:
      return ncclResult_t::ncclNumResults;
    default:
      throw std::runtime_error("Unconvertible NCCL type");
  }
}

// 将 ncclResult_t 转换为 torch::cuda::nccl::ncclResult
torch::cuda::nccl::ncclResult from_nccl_result(ncclResult_t var) {
  switch (var) {
    case ncclSuccess:
      return torch::cuda::nccl::ncclResult::Success;
    case ncclUnhandledCudaError:
      return torch::cuda::nccl::ncclResult::UnhandledCudaError;
    case ncclSystemError:
      return torch::cuda::nccl::ncclResult::SystemError;
    case ncclInternalError:
      return torch::cuda::nccl::ncclResult::InternalError;
    case ncclInvalidArgument:
      return torch::cuda::nccl::ncclResult::InvalidArgument;
    case ncclInvalidUsage:
      return torch::cuda::nccl::ncclResult::InvalidUsage;
    case ncclRemoteError:
      return torch::cuda::nccl::ncclResult::RemoteError;
#ifdef NCCL_HAS_COMM_NONBLOCKING
    case ncclInProgress:
      return torch::cuda::nccl::ncclResult::InProgress;
#endif
    case ncclNumResults:
      return torch::cuda::nccl::ncclResult::NumResults;
    default:
      throw std::runtime_error("Unconvertible NCCL type");
  }
}
// 根据 PyTorch 标量类型转换为对应的 NCCL 数据类型
ncclDataType_t to_nccl_data_type(c10::ScalarType type) {
  switch (type) {
    case at::kFloat:
      return ncclDataType_t::ncclFloat;
    case at::kHalf:
      return ncclDataType_t::ncclHalf;
    case at::kDouble:
      return ncclDataType_t::ncclDouble;
    case at::kLong:
      return ncclDataType_t::ncclInt64;
    case at::kInt:
      return ncclDataType_t::ncclInt;
    case at::kChar:
      return ncclDataType_t::ncclChar;
    case at::kByte:
      return ncclDataType_t::ncclUint8;
    case at::kBool:
      return ncclDataType_t::ncclUint8;
#if HAS_NCCL_BF16_DATATYPE
    case at::kBFloat16:
      return ncclDataType_t::ncclBfloat16;
#endif
    default:
      TORCH_CHECK(false, "Unconvertible NCCL type ", type);
  }
}

// 根据 PyTorch Tensor 类型获取对应的 NCCL 数据类型
ncclDataType_t to_nccl_data_type(const at::Tensor& t) {
  // 检查是否在 CUDA 设备上
  if (!t.is_cuda()) {
    TORCH_CHECK(
        false,
        "NCCL only supports CUDA tensors, but got a tensor on ",
        t.device());
  }
  // 调用标量类型转换函数，并返回结果
  return to_nccl_data_type(t.scalar_type());
}

// 将整数转换为对应的 NCCL reduction 操作类型
ncclRedOp_t to_nccl_red_op(int var) {
  return (ncclRedOp_t)(var);
}

namespace torch::cuda::nccl {

using namespace at;

namespace detail {

// 检查是否启用了非阻塞通信模式
static inline void NCCL_CHECK(ncclResult_t result) {
  NCCL_CHECK(from_nccl_result(result));
}

// 检查是否环境变量中设置了非阻塞通信
bool nccl_use_nonblocking() {
  static bool nccl_use_nonblocking_ =
      c10::utils::check_env("TORCH_NCCL_USE_COMM_NONBLOCKING") == true;
  if (nccl_use_nonblocking_) {
    TORCH_WARN("Using experimental non-blocking NCCL communicator.");
  }
  return nccl_use_nonblocking_;
}

// 解析环境变量中的非阻塞通信超时设置
static int _parse_nccl_nonblocking_timeout() {
  const char* val = getenv("TORCH_NCCL_NONBLOCKING_TIMEOUT");
  int timeout = -1;
  if (val) {
    const std::string config(val);
    timeout = std::stoi(config);
    if (!nccl_use_nonblocking() && timeout > 0) {
      TORCH_WARN(
          "TORCH_NCCL_NONBLOCKING_TIMEOUT has no effect when TORCH_NCCL_USE_COMM_NONBLOCKING is false.");
      timeout = -1;
    }
  }
  return timeout;
}

// 获取非阻塞通信超时时间
static int nccl_nonblocking_timeout() {
  static int timeout = _parse_nccl_nonblocking_timeout();
  return timeout;
}

// 检查 NCCL 操作是否超时，并处理结果
static inline void NCCL_CHECK_TIMEOUT(ncclResult status, ncclComm_t comm) {
#ifdef NCCL_HAS_COMM_NONBLOCKING
  ncclResult_t result = to_nccl_result(status);
  auto startTimepoint = std::chrono::steady_clock::now();
  while (result == ncclInProgress) {
    if (nccl_nonblocking_timeout() > 0) {
      auto currentTimepoint = std::chrono::steady_clock::now();
      auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                             currentTimepoint - startTimepoint)
                             .count();
      if (timeElapsed > nccl_nonblocking_timeout()) {
        throw std::runtime_error("NCCL timeout.");
      }
    }
    ncclCommGetAsyncError(to_nccl_comm(comm), &result);
  }
  if (result != ncclSuccess) {
    throw_nccl_error(from_nccl_result(result));
  }
#endif
}
#ifdef NCCL_HAS_COMM_NONBLOCKING
  // 检查是否支持非阻塞通信
  ncclResult_t result = to_nccl_result(status);
  // 将传入的状态转换为NCCL结果类型
  auto startTimepoint = std::chrono::steady_clock::now();
  // 记录开始时间点

  // 如果通信正在进行中
  if (result == ncclInProgress) {
    // 遍历所有的通信对象
    for (const auto i : c10::irange(comms.size())) {
      // 在结果变为非成功前不断尝试
      do {
        // 如果非阻塞超时时间大于0
        if (nccl_nonblocking_timeout() > 0) {
          // 获取当前时间点
          auto currentTimepoint = std::chrono::steady_clock::now();
          // 计算经过的秒数
          auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                 currentTimepoint - startTimepoint)
                                 .count();
          // 如果超过了非阻塞超时时间，则抛出运行时错误
          if (timeElapsed > nccl_nonblocking_timeout()) {
            throw std::runtime_error("NCCL timeout.");
          }
        }
        // 获取异步错误状态
        ncclCommGetAsyncError(to_nccl_comm(comms[i]), &result);
      } while (result == ncclInProgress);
      // 如果结果不是成功，则跳出循环处理失败情况
      if (result != ncclSuccess) {
        break; /* fall through to failed case */
      }
    }
  }
  // 如果结果不是成功，则抛出相应的NCCL错误
  if (result != ncclSuccess) {
    throw_nccl_error(from_nccl_result(result));
  }
#else
  // 如果不支持非阻塞通信，则断言失败，并显示错误信息
  TORCH_INTERNAL_ASSERT(
      false, "NCCL COMM NONBLOCKING USED WITH UNSUPPORTED NCCL VERSION.");
#endif
}

static inline void NCCL_CHECK_TIMEOUT(
    ncclResult_t result,
    std::vector<ncclComm_t>& comms) {
  // 调用重载函数，将传入的NCCL结果类型转换为NCCL错误类型处理
  NCCL_CHECK_TIMEOUT(from_nccl_result(result), comms);
}

void throw_nccl_error(torch::cuda::nccl::ncclResult status) {
  // 构造NCCL错误信息，并抛出运行时错误
  std::ostringstream err;
  err << "NCCL Error " << static_cast<int>(status) << ": "
      << ncclGetErrorString(to_nccl_result(status));
  throw std::runtime_error(err.str());
}

struct NcclCommList {
  std::unique_ptr<ncclComm_t[]> comms;
  int ndevices;
  // 构造函数，初始化NCCL通信对象列表
  NcclCommList(const std::vector<int>& devices)
      : comms(new ncclComm_t[devices.size()]), ndevices(devices.size()) {
    NCCL_CHECK(ncclCommInitAll(
        to_nccl_comm(comms.get()), devices.size(), devices.data()));
  }
  // 移动构造函数
  NcclCommList(NcclCommList&& foo) = default;
  // 析构函数，释放NCCL通信对象列表资源
  ~NcclCommList() {
    if (comms) {
      // 遍历所有设备，销毁通信对象
      for (const auto i : c10::irange(ndevices)) {
        int dummy_var;
        // 检查CUDA设备状态，避免在CUDA驱动程序已从进程卸载后继续销毁通信对象
        if (C10_CUDA_ERROR_HANDLED(cudaGetDevice(&dummy_var)) != cudaSuccess) {
          /* there are cases when this destructor is called after the
           CUDA driver is already unloaded from the process.
           In these cases, skip ncclCommDestroy */
          return;
        }
        // 销毁通信对象
        comm_destroy(comms[i]);
      }
    }
  }
  // 返回NCCL通信对象列表的引用
  ArrayRef<ncclComm_t> ref() const {
    return ArrayRef<ncclComm_t>(comms.get(), ndevices);
  }
};

using device_list = std::vector<int>;
// 访问此对象必须由THC的CudaFreeMutex进行保护
static std::unordered_map<device_list, NcclCommList, c10::hash<device_list>>
    _communicators;
// 检查张量是否满足要求，包括 CUDA 和密集张量，张量类型一致性和连续性
static inline void check_tensor(
    const at::Tensor& input,
    const at::optional<at::Tensor>& output,
    int input_multiplier,
    int output_multiplier,
    int64_t ref_numel,
    ScalarType ref_dtype) {
  
  auto check_one = [&](const at::Tensor& tensor) {
    // 检查张量是否在 CUDA 上且不是稀疏张量
    if (!tensor.is_cuda() || tensor.is_sparse()) {
      throw std::runtime_error(
          "input and output elements have to be cuda dense Tensors");
    }

    // 检查张量是否与参考数据类型一致
    if (ref_dtype != tensor.scalar_type()) {
      throw std::runtime_error(
          "all inputs and outputs must be of the same Tensor dtype");
    }

    // 检查张量是否是连续的
    if (!tensor.is_contiguous()) {
      throw std::runtime_error("all inputs and outputs have to be contiguous");
    }
  };

  // 对输入张量进行检查
  check_one(input);

  // 所有输入张量必须具有相同数量的元素
  if (input.numel() != ref_numel) {
    throw std::runtime_error(
        "all inputs must have the same number of elements");
  }

  if (output) {
    // 对输出张量进行检查
    check_one(*output);

    // 输入和输出必须位于相同的设备上
    if (input.get_device() != output->get_device()) {
      throw std::runtime_error("input and output must be on the same device");
    }

    // 输出的元素数量乘以输出乘数必须等于输入元素数量乘以输入乘数
    if (output->numel() * output_multiplier != ref_numel * input_multiplier) {
      throw std::runtime_error(
          "output must be of size input_size * size_multiplier");
    }
  }
}

// 检查输入张量列表和输出张量列表是否满足要求
void check_inputs(
    TensorList inputs,
    TensorList outputs,
    int input_multiplier,
    int output_multiplier) {
  
  // 输入张量列表和输出张量列表的长度必须相等
  size_t len = inputs.size();

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  if (len != outputs.size()) {
    // 抛出异常，输入和输出张量列表长度不一致
    std::stringstream err;
    err << "inputs and outputs sequences have to be of the same length, but got input of length "
        << len << " and output of length " << outputs.size();
    throw std::runtime_error(err.str());
  }

  // 用于存储设备的集合
  device_set devices;
  // 参考元素数量和数据类型来自于第一个输入张量
  int64_t numel = inputs[0].numel();
  auto dtype = inputs[0].scalar_type();

  for (const auto i : c10::irange(len)) {
    auto input = inputs[i];
    auto output = outputs[i];

    // 检查当前输入和输出张量是否符合要求
    check_tensor(
        input, output, input_multiplier, output_multiplier, numel, dtype);

    auto input_device = input.get_device();
    // 输入张量必须位于唯一的设备上
    if (devices.test(input_device)) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.set(input_device);
  }
}

// 检查输入张量列表和单个输出张量是否满足要求
void check_inputs(
    TensorList inputs,
    const at::Tensor& output,
    int root,
    int input_multiplier,
    int output_multiplier) {
  
  auto len = inputs.size();

  if (len <= 0) {
    // 抛出异常，输入张量列表不能为空
    throw std::runtime_error("input sequence can't be empty");
  }
}
    // 如果输入序列为空，抛出运行时错误
    throw std::runtime_error("input sequence can't be empty");
  }

  // 创建设备集合对象
  device_set devices;
  // 获取第一个输入张量的元素数量
  int64_t numel = inputs[0].numel();
  // 获取第一个输入张量的数据类型
  auto dtype = inputs[0].scalar_type();

  // 遍历输入张量序列的每个索引 i
  for (const auto i : c10::irange(len)) {
    // 获取当前索引 i 处的输入张量
    auto input = inputs[i];

    // 检查当前张量是否满足特定条件
    check_tensor(
        input,
        // 如果当前索引 i 等于根索引的类型，将输出张量作为可选项传递
        i == static_cast<std::remove_cv_t<decltype(i)>>(root)
            ? at::optional<at::Tensor>{output}
            : at::nullopt,
        input_multiplier,
        output_multiplier,
        numel,
        dtype);

    // 获取当前输入张量所在的设备
    auto input_device = input.get_device();
    // 检查输入张量是否在不同的设备上
    // 如果已经有相同设备的输入张量，则抛出运行时错误
    if (devices.test(input_device)) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    // 将当前输入张量所在设备添加到设备集合中
    devices.set(input_device);
  }
}

} // namespace detail

AutoNcclGroup::AutoNcclGroup() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR < 2)
  // 如果 NCCL 主版本号小于 2，需要先锁定 CUDA 的内存释放互斥锁
  (c10::cuda::getFreeMutex())->lock();
#endif
  comm_nonblocking_ = false;  // 初始化通信非阻塞为 false
  comm_ = nullptr;  // 初始化通信句柄为 nullptr
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
  // 如果 NCCL 主版本号大于等于 2，则调用 ncclGroupStart() 开始一个 NCCL 通信组
  detail::NCCL_CHECK(ncclGroupStart());
#endif
}

AutoNcclGroup::AutoNcclGroup(ncclComm_t comm, bool comm_nonblocking) {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR < 2)
  // 如果 NCCL 主版本号小于 2，需要先锁定 CUDA 的内存释放互斥锁
  (c10::cuda::getFreeMutex())->lock();
#endif
  comm_ = comm;  // 设置传入的通信句柄
  comm_nonblocking_ = comm_nonblocking;  // 设置是否为非阻塞通信
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
  // 如果 NCCL 主版本号大于等于 2，则调用 ncclGroupStart() 开始一个 NCCL 通信组
  detail::NCCL_CHECK(ncclGroupStart());
#endif
}

AutoNcclGroup::~AutoNcclGroup() noexcept(false) {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
  // 如果是非阻塞通信且通信句柄不为 nullptr，则调用 ncclGroupEnd() 结束 NCCL 通信组并检查超时
  if (comm_nonblocking_ && comm_ != nullptr) {
    detail::NCCL_CHECK_TIMEOUT(ncclGroupEnd(), comm_);
  } else {
    // 否则，调用 ncclGroupEnd() 结束 NCCL 通信组
    detail::NCCL_CHECK(ncclGroupEnd());
  }
#endif
#if defined(NCCL_MAJOR) && (NCCL_MAJOR < 2)
  // 如果 NCCL 主版本号小于 2，解锁 CUDA 的内存释放互斥锁
  (c10::cuda::getFreeMutex())->unlock();
#endif
}

bool is_available(TensorList tensors) {
#ifdef USE_NCCL
  // 检查是否支持 NCCL，如果不支持则直接返回 false
  device_set devices;
  for (auto& tensor : tensors) {
    // 检查张量是否在 GPU 上且不是稀疏张量
    if (!tensor.is_cuda() || tensor.is_sparse())
      return false;
    // 检查张量是否是连续的
    if (!tensor.is_contiguous())
      return false;
    auto device = tensor.get_device();
    // 检查是否有重复的设备
    if (devices[device])
      return false;
    devices[device] = true;
  }
  return true;  // 所有张量满足条件，返回 true
#else
  return false;  // 不支持 NCCL，直接返回 false
#endif
}

std::uint64_t version() {
#if defined(NCCL_MAJOR)
  // 如果定义了 NCCL_MAJOR，构造 NCCL 的版本号
  constexpr std::uint64_t ver = (((uint64_t)NCCL_MAJOR) << 32) |
      (((uint64_t)NCCL_MINOR) << 16) | ((uint64_t)NCCL_PATCH);
  return ver;  // 返回构造的版本号
#elif defined(USE_NCCL)
  // 如果定义了 USE_NCCL，返回主版本号为 1 的版本号
  return ((uint64_t)1) << 32;
#else
  return 0;  // 否则返回 0
#endif
}

const char* version_suffix() {
#if defined(NCCL_SUFFIX)
  return NCCL_SUFFIX;  // 如果定义了 NCCL_SUFFIX，则返回版本后缀
#else
  return "";  // 否则返回空字符串
#endif
}

void get_unique_id(ncclUniqueId& id) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  // 获取唯一标识符并转换为 NCCL 格式，然后检查操作是否成功
  NCCL_CHECK(ncclGetUniqueId(to_nccl_unique_id(&id)));
#else
  AT_ERROR("PyTorch built without NCCL support");  // 如果不支持 NCCL，则抛出错误
#endif
}

ncclComm_t comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  ncclComm_t comm;
  ncclUniqueId id = comm_id;
  // 初始化 NCCL 通信组，并检查操作是否成功
  NCCL_CHECK(ncclCommInitRank(
      to_nccl_comm(&comm), nranks, *(to_nccl_unique_id(&id)), rank));
  return comm;  // 返回初始化的通信句柄
#else
  return nullptr;  // 如果不支持 NCCL，则返回空指针
#endif
}

void comm_destroy(ncclComm_t comm) {
  /*
   * TODO(T30279827) Temporarily disable calling ncclCommDestroy
   * Calling ncclCommDestroy while program exiting is undefined
   * according to Nvidia, and lead to segfault in NCCL 2
   * (whether it is called before or after the CUDA runtime destructor).
   * Temporarily disable it in destructor to avoid segfault.
   * Following up with Nvidia for long term solution.
   */
  return;  // 暂时禁用调用 ncclCommDestroy，以避免潜在的段错误

#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  // 销毁 NCCL 通信句柄，并检查操作是否成功
  NCCL_CHECK(ncclCommDestroy(to_nccl_comm(comm)));
#endif
}
namespace {
// NCCL changed the numerical type used for count between NCCL1 and NCCL2.
// So we use the following struct, which gets the type of the second argument
// of T, if T is a function type, with ncclBcast, to get that type statically
// and programmatically.

// 模板类 GetSecondArgType 用于获取函数类型 T 的第二个参数类型
template <typename T>
struct GetSecondArgType;

// 部分特化，用于提取函数类型 R(Arg0, Arg1, Args...) 的第二个参数类型
template <typename R, typename Arg0, typename Arg1, typename... Args>
struct GetSecondArgType<R(Arg0, Arg1, Args...)> {
  typedef typename std::decay<Arg1>::type type;
};

// 获取 ncclBcast 函数的第二个参数类型，并将其最大值设为 count_max
constexpr auto count_max =
    std::numeric_limits<GetSecondArgType<decltype(ncclBcast)>::type>::max();

// Since NCCL 2.12.10, NCCL supports send/recv 0 byte:
// https://github.com/NVIDIA/nccl/issues/696. The issue of skipping send/recv
// is that it can cause deadlock when a rank send and recv 0 bytes so it's
// completely skipping the collective, causing mismatch across ranks
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR > 13)))
// 对于支持 0 字节传输的情况，始终返回 true
template <typename T>
constexpr bool _nccl_should_send_recv(C10_UNUSED T _unused_) {
  return true;
}
#else
// old NCCL uses 0 byte message for synchronization
// Avoid send/recv when message size is zero
// 在消息大小为零时，避免发送/接收操作
template <typename T>
inline bool _nccl_should_send_recv(T value) {
  return value != 0;
}
#endif
} // namespace

// 返回 count_max 的值
size_t get_max_count() {
  return count_max;
}

// 使用 NCCL 进行广播操作
void broadcast(
    TensorList tensors,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  // 引入 torch::cuda::nccl::detail 命名空间
  using namespace torch::cuda::nccl::detail;
  // 检查输入张量的有效性
  check_inputs(tensors, tensors, 1, 1);
  // 获取第一个张量的 NCCL 数据类型
  auto data_type = to_nccl_data_type(tensors[0]);
  // 获取第一个张量的元素数量
  int64_t numel = tensors[0].numel();

  // 根据用户输入或默认情况获取通信器列表
  const auto comms = user_comms.empty() ? get_communicators(tensors)
                                        : ArrayRef<ncclComm_t>(user_comms);

  // 自动管理 NCCL 组的生命周期
  AutoNcclGroup nccl_group_guard;
  // 可选的 CUDA 设备守卫
  at::cuda::OptionalCUDAGuard device_guard;
  // 遍历所有张量
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; i++) {
    auto device = tensors[i].get_device();
    device_guard.set_index(device);
    // 默认使用当前流
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();
    // 检查张量的元素数量是否超过 NCCL 支持的最大限制
    TORCH_CHECK(
        static_cast<uint64_t>(numel) <= static_cast<uint64_t>(count_max),
        "Broadcast tensor has ",
        numel,
        " elements, which exceeds the "
        "maximum NCCL supports (",
        count_max,
        ")");
    // 获取当前通信器
    ncclComm_t comm = comms[i];
    // 执行广播操作
    NCCL_CHECK(ncclBcast(
        tensors[i].data_ptr(),
        numel,
        data_type,
        0,
        to_nccl_comm(comm),
        stream));
  }
#else
  // 如果未使用 NCCL，抛出错误
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

// 使用 NCCL 执行归约操作
void reduce(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& output,
    int32_t root,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  // 如果定义了 USE_NCCL 宏，则使用 torch::cuda::nccl::detail 命名空间
  using namespace torch::cuda::nccl::detail;
  // 检查 root 参数是否有效，即在 inputs 的范围内
  TORCH_CHECK(
      root >= 0 && static_cast<size_t>(root) < inputs.size(), "invalid root");

  // 检查输入参数的有效性
  check_inputs(inputs, output, root, 1, 1);
  // 获取输入张量的个数
  const auto len = inputs.size();

  // 将输入张量的数据类型转换为 NCCL 支持的数据类型
  auto data_type = to_nccl_data_type(inputs[0]);

  // 获取第一个输入张量的元素总数
  const auto count = inputs[0].numel();
  // 获取通信器的引用，如果用户未指定，则使用默认的通信器
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  // 自动管理 NCCL 通信组的生命周期
  AutoNcclGroup nccl_group_guard;
  // 可选的 CUDA 设备上下文管理器
  at::cuda::OptionalCUDAGuard device_guard;
  // 遍历所有输入张量
  for (const auto i : c10::irange(len)) {
    // 获取当前张量所在的 CUDA 设备索引
    auto device = inputs[i].device().index();
    // 设置当前设备上下文
    device_guard.set_index(device);
    // 默认使用当前流，如果未指定流，则使用当前 CUDA 流
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    // 获取当前通信器
    ncclComm_t comm = comms_ref[i];
    // 执行 NCCL 的 reduce 操作，将结果写入输出张量中
    NCCL_CHECK(ncclReduce(
        inputs[i].data_ptr(),
        static_cast<std::remove_cv_t<decltype(i)>>(root) == i
            ? output.data_ptr()
            : nullptr,
        count,
        data_type,
        to_nccl_red_op(op),
        root,
        to_nccl_comm(comm),
        stream));
  }
#else
  // 如果没有定义 USE_NCCL 宏，则抛出错误信息
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
  // 调用 reduce 函数，将输出结果写入到 inputs[root] 中
  reduce(inputs, /*output=*/inputs[root], root, op, streams, user_comms);
}

void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  // 如果定义了 USE_NCCL 宏，则使用 torch::cuda::nccl::detail 命名空间
  using namespace torch::cuda::nccl::detail;
  // 检查输入参数的有效性
  check_inputs(inputs, outputs, 1, 1);
  // 获取输入张量的个数
  const auto len = inputs.size();

  // 将输入张量的数据类型转换为 NCCL 支持的数据类型
  auto data_type = to_nccl_data_type(inputs[0]);

  // 获取第一个输入张量的元素总数
  const auto count = inputs[0].numel();
  // 获取通信器的引用，如果用户未指定，则使用默认的通信器
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  // 自动管理 NCCL 通信组的生命周期
  AutoNcclGroup nccl_group_guard;
  // 可选的 CUDA 设备上下文管理器
  at::cuda::OptionalCUDAGuard device_guard;
  // 遍历所有输入张量
  for (const auto i : c10::irange(len)) {
    // 获取当前张量所在的 CUDA 设备索引
    auto device = inputs[i].device().index();
    // 设置当前设备上下文
    device_guard.set_index(device);
    // 默认使用当前流，如果未指定流，则使用当前 CUDA 流
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    // 获取当前通信器
    ncclComm_t comm = comms_ref[i];
    // 执行 NCCL 的 all_reduce 操作，将结果写入到 outputs[i] 中
    NCCL_CHECK(ncclAllReduce(
        inputs[i].data_ptr(),
        outputs[i].data_ptr(),
        count,
        data_type,
        to_nccl_red_op(op),
        to_nccl_comm(comm),
        stream));
  }
#else
  // 如果没有定义 USE_NCCL 宏，则抛出错误信息
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  // 使用NCCL命名空间中的细节命名空间，提供CUDA通信相关的细节
  using namespace torch::cuda::nccl::detail;
  // 获取输入张量的数量
  const auto len = inputs.size();
  // 检查输入和输出张量的有效性，要求输入数量为len，输出数量为1
  check_inputs(inputs, outputs, 1, len);

  // 将第一个输入张量的数据类型转换为NCCL数据类型
  auto data_type = to_nccl_data_type(inputs[0]);

  // 计算每个输入张量的元素数量
  const auto count = inputs[0].numel() / len;

  // 获取通信子组列表的引用，若用户未指定则使用默认方法获取通信子组
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  // 自动NCCL组管理器，用于管理NCCL通信的生命周期
  AutoNcclGroup nccl_group_guard;
  // 可选的CUDA设备保护器，用于设置和保持当前CUDA设备的索引
  at::cuda::OptionalCUDAGuard device_guard;
  // 遍历输入张量的每一个索引
  for (const auto i : c10::irange(len)) {
    // 获取当前输入张量所在的CUDA设备索引
    auto device = inputs[i].device().index();
    // 设置当前CUDA设备索引
    device_guard.set_index(device);
    // 默认使用当前设备上的CUDA流
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    // 获取当前通信子组
    ncclComm_t comm = comms_ref[i];
    // 执行NCCL Reduce-Scatter操作，将输入张量数据归约并分散到输出张量上
    NCCL_CHECK(ncclReduceScatter(
        inputs[i].data_ptr(),
        outputs[i].data_ptr(),
        count,
        data_type,
        to_nccl_red_op(op),
        to_nccl_comm(comm),
        stream));
  }
#else
  // 如果未使用NCCL支持，则抛出错误
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  // 使用NCCL命名空间中的细节命名空间，提供CUDA通信相关的细节
  using namespace torch::cuda::nccl::detail;
  // 获取输入张量的数量
  const auto len = inputs.size();
  // 检查输入和输出张量的有效性，要求输入和输出张量数量一致，均为len
  check_inputs(inputs, outputs, len, 1);

  // 将第一个输入张量的数据类型转换为NCCL数据类型
  auto data_type = to_nccl_data_type(inputs[0]);

  // 计算每个输入张量的元素总数
  const auto count = inputs[0].numel();
  
  // 获取通信子组列表的引用，若用户未指定则使用默认方法获取通信子组
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  // 自动NCCL组管理器，用于管理NCCL通信的生命周期
  AutoNcclGroup nccl_group_guard;
  // 可选的CUDA设备保护器，用于设置和保持当前CUDA设备的索引
  at::cuda::OptionalCUDAGuard device_guard;
  // 遍历输入张量的每一个索引
  for (const auto i : c10::irange(len)) {
    // 获取当前输入张量所在的CUDA设备索引
    auto device = inputs[i].device().index();
    // 设置当前CUDA设备索引
    device_guard.set_index(device);
    // 默认使用当前设备上的CUDA流
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    // 获取当前通信子组
    ncclComm_t comm = comms_ref[i];

    // 根据NCCL的版本不同，执行相应的NCCL All-Gather操作
    #if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
    NCCL_CHECK(ncclAllGather(
        inputs[i].data_ptr(),
        outputs[i].data_ptr(),
        count,
        data_type,
        to_nccl_comm(comm),
        stream));
    #else
    NCCL_CHECK(ncclAllGather(
        inputs[i].data_ptr(),
        count,
        data_type,
        outputs[i].data_ptr(),
        to_nccl_comm(comm),
        stream));
    #endif
  }
#else
  // 如果未使用NCCL支持，则抛出错误
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all2all_single_equal_split(
    at::Tensor& input,
    at::Tensor& output,
    int size,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream) {
#ifdef USE_NCCL
  // 当NCCL的主版本号大于等于2时，执行以下代码块
  #if defined(NCCL_MAJOR) && \
      (NCCL_MAJOR >= 2)
  // 执行NCCL All-to-All操作，将输入张量按等分规则分发到输出张量上
#else
  // 当NCCL的主版本号小于2时，执行以下代码块
#endif
#else
  // 如果未使用NCCL支持，则抛出错误
  AT_ERROR("PyTorch built without NCCL support");
#endif
}
    // 检查 NCCL 主版本号是否大于 2 或者主版本号等于 2 且次版本号大于等于 7
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  // 引入 torch::cuda::nccl::detail 命名空间
  using namespace torch::cuda::nccl::detail;

  // 声明整型变量 numranks
  int numranks;
  // 将输入张量转换为 NCCL 数据类型
  auto type = to_nccl_data_type(input);
  // 计算输入张量元素个数除以 size 的结果，得到 count
  size_t count = input.numel() / size;
  // 计算输入张量字节数除以 size 的结果，得到 rankdiff
  size_t rankdiff = input.nbytes() / size;
  // 将输入张量的常量数据指针转换为 const char* 类型，作为发送缓冲区
  const auto* sendbuff = reinterpret_cast<const char*>(input.const_data_ptr());
  // 将输出张量的数据指针转换为 char* 类型，作为接收缓冲区
  auto* recvbuff = reinterpret_cast<char*>(output.data_ptr());
  // 将通信组对象转换为 NCCL 通信对象
  auto comm = to_nccl_comm(_comm);
#if defined(USE_ROCM)
  // 使用 ROCm 特定的 allToAll 函数进行通信
  NCCL_CHECK(ncclAllToAll(sendbuff, recvbuff, count, type, comm, stream));
#else
  // 获取通信组中的节点数目
  NCCL_CHECK(ncclCommCount(comm, &numranks));
  // 开始一个通信组，准备执行多个通信操作
  NCCL_CHECK(ncclGroupStart());
  // 遍历所有节点
  for (const auto r : c10::irange(numranks)) {
    // 根据应用程序逻辑判断是否需要发送和接收数据
    if (_nccl_should_send_recv(count)) {
      // 发送数据给节点 r
      NCCL_CHECK(
          ncclSend(sendbuff + r * rankdiff, count, type, r, comm, stream));
      // 接收来自节点 r 的数据
      NCCL_CHECK(
          ncclRecv(recvbuff + r * rankdiff, count, type, r, comm, stream));
    }
  }
  // 结束当前通信组的操作
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclGroupEnd());
#else
  // 在超时限制下结束通信组的操作
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif
#endif
#else
  // 当 NCCL 版本低于 2.7.0 时，报错提示
  AT_ERROR("all2all is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  // 当 PyTorch 编译时未启用 NCCL 支持时，报错提示
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType _type,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  // 使用 torch::cuda::nccl::detail 命名空间
  using namespace torch::cuda::nccl::detail;

  // 将 Torch 数据类型转换为对应的 NCCL 数据类型
  auto type = to_nccl_data_type(_type);
  // 将 Torch 通信对象转换为对应的 NCCL 通信对象
  auto comm = to_nccl_comm(_comm);
  int numranks;
  // 获取通信组中的节点数目
  NCCL_CHECK(ncclCommCount(comm, &numranks));
  // 开始一个通信组，准备执行多个通信操作
  NCCL_CHECK(ncclGroupStart());
  // 遍历所有节点
  for (const auto r : c10::irange(numranks)) {
    // 根据应用程序逻辑判断是否需要发送数据
    if (_nccl_should_send_recv(sendcounts[r])) {
      // 发送数据给节点 r
      NCCL_CHECK(ncclSend(
          ((char*)sendbuff) + senddispls[r] * size,
          sendcounts[r],
          type,
          r,
          comm,
          stream));
    }
    // 根据应用程序逻辑判断是否需要接收数据
    if (_nccl_should_send_recv(recvcounts[r])) {
      // 接收来自节点 r 的数据
      NCCL_CHECK(ncclRecv(
          ((char*)recvbuff) + recvdispls[r] * size,
          recvcounts[r],
          type,
          r,
          comm,
          stream));
    }
  }
  // 结束当前通信组的操作
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclGroupEnd());
#else
  // 在超时限制下结束通信组的操作
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif
#else
  // 当 NCCL 版本低于 2.7.0 时，报错提示
  AT_ERROR("all2all is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  // 当 PyTorch 编译时未启用 NCCL 支持时，报错提示
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  // 使用 torch::cuda::nccl::detail 命名空间
  using namespace torch::cuda::nccl::detail;
  // 将 Torch 通信对象转换为对应的 NCCL 通信对象
  auto comm = to_nccl_comm(_comm);

  // 开始一个通信组，准备执行多个通信操作
  NCCL_CHECK(ncclGroupStart());
  // 遍历所有输出张量
  for (const auto r : c10::irange(outputTensors.size())) {
    // 获取当前输入和输出张量
    at::Tensor& input = inputTensors[r];
    at::Tensor& output = outputTensors[r];

    // 根据应用程序逻辑判断是否需要发送数据
    if (_nccl_should_send_recv(input.numel())) {
      // 发送数据
      NCCL_CHECK(ncclSend(
          input.data_ptr(),
          input.numel(),
          to_nccl_data_type(input),
          r,
          comm,
          stream.stream()));
    }
    // 根据应用程序逻辑判断是否需要接收数据
    # 检查是否应该执行发送和接收操作，根据输出张量的元素数量判断
    if (_nccl_should_send_recv(output.numel())) {
      # 调用 NCCL 接收函数，接收数据到输出张量的数据指针中
      NCCL_CHECK(ncclRecv(
          output.data_ptr(),                   # 输出张量的数据指针
          output.numel(),                      # 输出张量的元素数量
          to_nccl_data_type(output),           # 将输出张量的数据类型转换为对应的 NCCL 数据类型
          r,                                   # 接收方的设备标识符
          comm,                                # NCCL 通信组
          stream.stream()));                   # 使用的 CUDA 流
    }
  }
#ifndef`
#ifndef NCCL_HAS_COMM_NONBLOCKING
  // 如果没有非阻塞通信支持，调用 ncclGroupEnd()，结束 NCCL 通信组
  NCCL_CHECK(ncclGroupEnd());
#else
  // 如果有非阻塞通信支持，调用 ncclGroupEnd()，并设定超时参数 _comm
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif
#else
  // 如果 NCCL 版本小于 2.7.0，抛出错误，提示不支持 all2all
  AT_ERROR("all2all is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  // 如果 PyTorch 未编译 NCCL 支持，抛出错误
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

// 函数定义，负责发送数据
void send(
    const at::Tensor& input,            // 输入张量
    ncclComm_t comm,                    // NCCL 通信句柄
    at::cuda::CUDAStream stream,        // CUDA 流
    int dst) {                          // 目标节点
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail; // 使用 NCCL detail 命名空间

#ifndef NCCL_HAS_COMM_NONBLOCKING
  // 如果没有非阻塞通信支持，调用 ncclSend 发送数据
  NCCL_CHECK(ncclSend(
      input.data_ptr(),                 // 数据指针
      input.numel(),                    // 数据元素数量
      to_nccl_data_type(input),          // 数据类型
      dst,                               // 目标节点
      to_nccl_comm(comm),                // NCCL 通信对象
      stream.stream()));                 // CUDA 流
#else
  // 如果有非阻塞通信支持，调用 ncclSend，设置超时参数 comm
  NCCL_CHECK_TIMEOUT(
      ncclSend(
          input.data_ptr(),             // 数据指针
          input.numel(),                // 数据元素数量
          to_nccl_data_type(input),      // 数据类型
          dst,                           // 目标节点
          to_nccl_comm(comm),            // NCCL 通信对象
          stream.stream()),              // CUDA 流
      comm);
#endif
#else
  // 如果 NCCL 版本小于 2.7.0，抛出错误，提示不支持发送
  AT_ERROR("Send is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  // 如果未启用 NCCL，抛出错误
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

// 函数定义，负责接收数据
void recv(
    at::Tensor& output,                 // 输出张量
    ncclComm_t comm,                    // NCCL 通信句柄
    at::cuda::CUDAStream stream,        // CUDA 流
    int src) {                          // 源节点
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail; // 使用 NCCL detail 命名空间

#ifndef NCCL_HAS_COMM_NONBLOCKING
  // 如果没有非阻塞通信支持，调用 ncclRecv 接收数据
  NCCL_CHECK(ncclRecv(
      output.data_ptr(),                // 数据指针
      output.numel(),                   // 数据元素数量
      to_nccl_data_type(output),         // 数据类型
      src,                               // 源节点
      to_nccl_comm(comm),                // NCCL 通信对象
      stream.stream()));                 // CUDA 流
#else
  // 如果有非阻塞通信支持，调用 ncclRecv，设置超时参数 comm
  NCCL_CHECK_TIMEOUT(
      ncclRecv(
          output.data_ptr(),            // 数据指针
          output.numel(),               // 数据元素数量
          to_nccl_data_type(output),     // 数据类型
          src,                           // 源节点
          to_nccl_comm(comm),            // NCCL 通信对象
          stream.stream()),              // CUDA 流
      comm);
#endif
#else
  // 如果 NCCL 版本小于 2.7.0，抛出错误，提示不支持接收
  AT_ERROR("Recv is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  // 如果未启用 NCCL，抛出错误
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

// 函数定义，负责数据的收集操作
void gather(
    const at::Tensor& inputs,            // 输入张量
    std::vector<at::Tensor>& outputs,    // 输出张量列表
    ncclComm_t _comm,                    // NCCL 通信句柄
    at::cuda::CUDAStream& stream,        // CUDA 流
    int32_t root) {                      // 根节点
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail; // 使用 NCCL detail 命名空间

  auto comm = to_nccl_comm(_comm);          // 转换 NCCL 通信句柄
  int numranks, cur_rank;                   // 定义通信的节点数和当前节点编号
  NCCL_CHECK(ncclCommCount(comm, &numranks));  // 获取节点总数
  NCCL_CHECK(ncclCommUserRank(comm, &cur_rank)); // 获取当前节点编号

  size_t count = inputs.numel();            // 数据元素数量
  auto type = to_nccl_data_type(inputs);     // 数据类型
  const auto* sendbuff = reinterpret_cast<const char*>(inputs.const_data_ptr());  // 输入数据指针

  NCCL_CHECK(ncclGroupStart());             // 开始 NCCL 通信组

  if (cur_rank == root) {                   // 如果当前节点是根节点
    for (const auto r : c10::irange(numranks)) { // 遍历所有节点
      if (r != root) {                       // 如果节点不是根节点
        auto* recvbuff = reinterpret_cast<char*>(outputs[r].data_ptr());  // 输出数据指针
        NCCL_CHECK(ncclRecv(recvbuff, count, type, r, comm, stream));  // 接收数据
      } else {
        // 根节点直接复制输入数据到输出张量
        outputs[r].copy_(inputs);
      }
    }
  } else {
    // 其他节点进行数据发送
    NCCL_CHECK(ncclSend(sendbuff, count, type, root, comm, stream)); // 发送数据
  }
}
#else
  // 如果 NCCL 版本小于 2.7.0，抛出错误，提示不支持 gather
  AT_ERROR("Gather is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  // 如果未启用 NCCL，抛出错误
  AT_ERROR("PyTorch built without NCCL support");
#endif
}
    # 使用 NCCL 库的函数 `ncclSend` 发送数据
    NCCL_CHECK(ncclSend(sendbuff, count, type, root, comm, stream));
    # `sendbuff`: 待发送的数据缓冲区
    # `count`: 数据元素的数量
    # `type`: 数据类型
    # `root`: 根节点的标识
    # `comm`: NCCL 通信子（communication handle）
    # `stream`: CUDA 流（用于异步操作）
    # 函数调用后会检查返回值，确保发送操作成功完成
#ifndef NCCL_HAS_COMM_NONBLOCKING
  // 如果未定义 NCCL_HAS_COMM_NONBLOCKING 宏，则执行以下代码块
  NCCL_CHECK(ncclGroupEnd());
#else
  // 如果定义了 NCCL_HAS_COMM_NONBLOCKING 宏，则执行以下代码块
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif

#else
  // 如果未定义 NCCL 宏，则报错，提示需要 NCCL 支持版本大于等于 2.7.0
  AT_ERROR("gather is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  // 如果 PyTorch 编译时没有 NCCL 支持，则报错
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream,
    int32_t root) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  // 使用 NCCL 进行 scatter 操作，确保引用正确的命名空间
  using namespace torch::cuda::nccl::detail;

  // 将传入的 _comm 转换为 NCCL comm 对象
  auto comm = to_nccl_comm(_comm);
  int numranks, cur_rank;
#ifndef NCCL_HAS_COMM_NONBLOCKING
  // 如果未定义 NCCL_HAS_COMM_NONBLOCKING 宏，则获取通信组中的进程数和当前进程的排名
  NCCL_CHECK(ncclCommCount(comm, &numranks));
  NCCL_CHECK(ncclCommUserRank(comm, &cur_rank));
#else
  // 如果定义了 NCCL_HAS_COMM_NONBLOCKING 宏，则使用超时版本获取通信组信息
  NCCL_CHECK_TIMEOUT(ncclCommCount(comm, &numranks), _comm);
  NCCL_CHECK_TIMEOUT(ncclCommUserRank(comm, &cur_rank), _comm);
#endif
  // 在 NCCL 通信组中开始一个新的通信步骤
  NCCL_CHECK(ncclGroupStart());
  if (cur_rank == root) {
    // 如果当前进程是根进程，则执行以下代码块
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        // 对于非根进程，发送数据到指定的进程 r
        size_t send_count = inputs[r].numel();
        auto send_type = to_nccl_data_type(inputs[r]);
        const auto* sendbuff =
            reinterpret_cast<const char*>(inputs[r].const_data_ptr());
        NCCL_CHECK(ncclSend(sendbuff, send_count, send_type, r, comm, stream));
      } else {
        // 对于根进程，直接将数据复制到输出张量中
        outputs.copy_(inputs[r]);
      }
    }
  } else {
    // 如果当前进程不是根进程，则接收来自根进程的数据
    size_t recv_count = outputs.numel();
    auto recv_type = to_nccl_data_type(outputs);
    auto* recvbuff = reinterpret_cast<char*>(outputs.data_ptr());
    NCCL_CHECK(ncclRecv(recvbuff, recv_count, recv_type, root, comm, stream));
  }
#ifndef NCCL_HAS_COMM_NONBLOCKING
  // 如果未定义 NCCL_HAS_COMM_NONBLOCKING 宏，则结束通信步骤
  NCCL_CHECK(ncclGroupEnd());
#else
  // 如果定义了 NCCL_HAS_COMM_NONBLOCKING 宏，则使用超时版本结束通信步骤
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif
#else
  // 如果 NCCL 版本小于 2.7.0，则报错
  AT_ERROR("scatter is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  // 如果 PyTorch 编译时没有 NCCL 支持，则报错
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

} // namespace torch::cuda::nccl
```