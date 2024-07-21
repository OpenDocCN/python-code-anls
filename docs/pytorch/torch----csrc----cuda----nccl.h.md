# `.\pytorch\torch\csrc\cuda\nccl.h`

```py
#pragma once
// 预处理指令：指示编译器在编译此头文件时只包含一次，防止重复包含

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <vector>

// NCCL BFloat16 仅在 CUDA 11+ 和 NCCL 版本 2.10+，或者 HIP 3.1+ 时启用
#if defined(__CUDA_BF16_TYPES_EXIST__)
#define HAS_NCCL_BF16_DATATYPE \
  ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
#elif defined(USE_ROCM) && (TORCH_HIP_VERSION >= 301)
#define HAS_NCCL_BF16_DATATYPE 1
#else
#define HAS_NCCL_BF16_DATATYPE 0
#endif

namespace torch::cuda::nccl {

/* 以下内容从 <nccl.h> 复制并在 torch::cuda::nccl 命名空间中重新定义 */
/* pytorch 应该只在 pytorch 范围内使用以下定义 */

/* 用于在 nccl.cpp 中重新解释为 ncclComm 的不透明通信句柄 */
typedef void* ncclComm_t;

/** 在 torch 范围内重新定义 nccl unique ID，应与 native nccl 实现相同 */
#define NCCL_UNIQUE_ID_BYTES 128
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
typedef struct {
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

/* 错误类型 */
enum class ncclResult {
  Success = 0,                // 成功
  UnhandledCudaError = 1,     // 未处理的 CUDA 错误
  SystemError = 2,            // 系统错误
  InternalError = 3,          // 内部错误
  InvalidArgument = 4,        // 无效参数
  InvalidUsage = 5,           // 无效用法
  RemoteError = 6,            // 远程错误
  InProgress = 7,             // 进行中
  NumResults = 8              // 结果数
};

/* 归约操作选择器 */
enum class ncclRedOp {
  Sum = 0,    // 求和
  Prod = 1,   // 乘积
  Max = 2,    // 最大值
  Min = 3,    // 最小值
  NumOps = 4  // 操作数
};

/* 数据类型 */
enum class ncclDataType {
  Int8 = 0,
  Char = 0,
  Uint8 = 1,
  Int32 = 2,
  Int = 2,
  Uint32 = 3,
  Int64 = 4,
  Uint64 = 5,
  Float16 = 6,
  Half = 6,
  Float32 = 7,
  Float = 7,
  Float64 = 8,
  Double = 8,
  Bfloat16 = 9,
  NumTypes = 10
};

// RAII 辅助类，管理 NCCL 组 API 和 CUDA 释放互斥锁
// 析构函数允许抛出异常，因为此辅助类仅管理组和锁的生命周期
struct AutoNcclGroup {
  AutoNcclGroup();  // 构造函数
  AutoNcclGroup(ncclComm_t comm, bool comm_nonblocking);  // 构造函数
  ~AutoNcclGroup() noexcept(false);  // 析构函数
  ncclComm_t comm_;  // NCCL 通信句柄
  bool comm_nonblocking_;  // 是否非阻塞通信
};

// 注意：此处仅用于 python_nccl.cpp 可能使用这些辅助函数
// 不要在这些文件之外使用它们
namespace detail {

TORCH_CUDA_CPP_API void throw_nccl_error(ncclResult status);  // 抛出 NCCL 错误

inline void NCCL_CHECK(ncclResult status) {  // 检查 NCCL 操作结果
  if (status != ncclResult::Success) {
    throw_nccl_error(status);
  }
}

TORCH_CUDA_CPP_API at::ArrayRef<ncclComm_t> get_communicators(
    at::TensorList inputs);  // 获取通信句柄数组的引用

TORCH_CUDA_CPP_API void check_inputs(
    at::TensorList inputs,
    at::TensorList outputs,
    int input_multiplier,
    int output_multiplier);  // 检查输入和输出张量列表

TORCH_CUDA_CPP_API void check_inputs(
    at::TensorList inputs,
    const at::Tensor& output,
    int root,
    int input_multiplier,
    int output_multiplier);  // 检查输入和输出张量列表

} // namespace detail

using comm_list = std::vector<ncclComm_t>;  // 通信句柄列表
using stream_list = std::vector<std::optional<at::cuda::CUDAStream>>;  // CUDA 流列表

TORCH_CUDA_CPP_API std::uint64_t version();  // 获取版本号

} // namespace torch::cuda::nccl
TORCH_CUDA_CPP_API const char* version_suffix();
// 声明一个函数 version_suffix，返回一个指向常量字符的指针，用于获取版本后缀

bool is_available(at::TensorList tensors);
// 声明一个函数 is_available，接受一个张量列表作为参数，返回一个布尔值，表示CUDA是否可用

TORCH_CUDA_CPP_API void get_unique_id(ncclUniqueId& id);
// 声明一个函数 get_unique_id，接受一个 ncclUniqueId 的引用作为参数，用于获取唯一标识符

TORCH_CUDA_CPP_API ncclComm_t
comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank);
// 声明一个函数 comm_init_rank，接受三个参数：nranks（总的通信等级数）、comm_id（通信唯一标识符）、rank（当前通信等级的排名），返回一个 ncclComm_t 类型，表示初始化通信

TORCH_CUDA_CPP_API void comm_destroy(ncclComm_t comm);
// 声明一个函数 comm_destroy，接受一个 ncclComm_t 类型的参数 comm，用于销毁通信对象

TORCH_CUDA_CPP_API void broadcast(
    at::TensorList tensors,
    const stream_list& streams = {},
    const comm_list& user_comms = {});
/*
声明一个函数 broadcast，接受三个参数：
- tensors：张量列表
- streams：流列表，默认为空
- user_comms：通信列表，默认为空
用于在CUDA环境下进行广播操作
*/

size_t get_max_count();
// 声明一个函数 get_max_count，返回一个 size_t 类型的值，表示获取最大计数

TORCH_CUDA_CPP_API void reduce(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& output,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});
/*
声明一个函数 reduce，接受六个参数：
- inputs：输入张量的向量
- output：输出张量
- root：根的索引，默认为0
- op：归约操作的类型，默认为求和
- streams：流列表，默认为空
- user_comms：通信列表，默认为空
用于在CUDA环境下进行归约操作
*/

TORCH_CUDA_CPP_API void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});
/*
重载的 reduce 函数，接受五个参数：
- inputs：输入张量的向量
- root：根的索引，默认为0
- op：归约操作的类型，默认为求和
- streams：流列表，默认为空
- user_comms：通信列表，默认为空
用于在CUDA环境下进行归约操作
*/

TORCH_CUDA_CPP_API void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});
/*
声明一个函数 all_reduce，接受五个参数：
- inputs：输入张量的向量
- outputs：输出张量的向量
- op：归约操作的类型，默认为求和
- streams：流列表，默认为空
- user_comms：通信列表，默认为空
用于在CUDA环境下进行全局归约操作
*/

TORCH_CUDA_CPP_API void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});
/*
声明一个函数 reduce_scatter，接受五个参数：
- inputs：输入张量的向量
- outputs：输出张量的向量
- op：归约操作的类型，默认为求和
- streams：流列表，默认为空
- user_comms：通信列表，默认为空
用于在CUDA环境下进行归约分发操作
*/

TORCH_CUDA_CPP_API void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream,
    int32_t root = 0);
/*
声明一个函数 scatter，接受五个参数：
- inputs：输入张量的向量
- outputs：输出张量
- comm：通信对象
- stream：CUDA流
- root：根的索引，默认为0
用于在CUDA环境下进行分发操作
*/

TORCH_CUDA_CPP_API void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams = {},
    const comm_list& user_comms = {});
/*
声明一个函数 all_gather，接受四个参数：
- inputs：输入张量的向量
- outputs：输出张量的向量
- streams：流列表，默认为空
- user_comms：通信列表，默认为空
用于在CUDA环境下进行全局收集操作
*/

TORCH_CUDA_CPP_API void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream,
    int32_t root = 0);
/*
声明一个函数 gather，接受五个参数：
- inputs：输入张量
- outputs：输出张量的向量
- comm：通信对象
- stream：CUDA流
- root：根的索引，默认为0
用于在CUDA环境下进行收集操作
*/

TORCH_CUDA_CPP_API void all2all_single_equal_split(
    at::Tensor& input,
    at::Tensor& output,
    int size,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream);
/*
声明一个函数 all2all_single_equal_split，接受六个参数：
- input：输入张量
- output：输出张量
- size：大小
- comm：通信对象
- stream：CUDA流
用于在CUDA环境下进行等分 all-to-all 操作
*/

TORCH_CUDA_CPP_API void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType type,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream);
/*
声明一个函数 all2all_single_unequal_split，接受十个参数：
- sendbuff：发送缓冲区
- sendcounts：发送计数数组
- senddispls：发送位移数组
- recvbuff：接收缓冲区
- recvcounts：接收计数数组
- recvdispls：接收位移数组
- size：大小
- type：标量类型
- comm：通信对象
- stream：CUDA流
用于在CUDA环境下进行非等分 all-to-all 操作
*/

TORCH_CUDA_CPP_API void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream);
/*
声明一个函数 all2all，接受四个参数：
- outputTensors：输出张量的向量
- inputTensors：输入张量的向量
- _comm：通信对象
- stream：CUDA流
用于在CUDA环境下进行 all-to-all 操作
*/

TORCH_CUDA_CPP_API void send(
    const at::Tensor& input,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int dst);
/*
声明一个函数 send，接受四个参数：
- input：输入张量
- comm：通信对象
- stream：CUDA流
- dst：目标索引
用于在CUDA环境下发送张量
*/

TORCH_CUDA_CPP_API void recv(
    at::Tensor& output,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int src);
/*
声明一个函数 recv，接受四个参数：
- output：输出张量
- comm：通信对象
- stream：CUDA流
- src：源索引
用于在CUDA环境下接收张量
*/

} // namespace torch::cuda::nccl
// 结束命名空间 torch::cuda::nccl
```