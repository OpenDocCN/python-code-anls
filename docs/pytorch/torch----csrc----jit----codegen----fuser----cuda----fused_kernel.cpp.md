# `.\pytorch\torch\csrc\jit\codegen\fuser\cuda\fused_kernel.cpp`

```py
// 包含 CUDA 融合内核相关头文件
#include <torch/csrc/jit/codegen/fuser/cuda/fused_kernel.h>

// 包含 CUDA 编译器相关头文件
#include <torch/csrc/jit/codegen/fuser/compiler.h>

// 包含 ATen 库的核心头文件
#include <ATen/ATen.h>

// 包含 ATen CUDA 上下文相关头文件
#include <ATen/cuda/CUDAContext.h>

// 包含 ATen CUDA 生成器实现相关头文件
#include <ATen/cuda/CUDAGeneratorImpl.h>

// 包含 ATen NVRTC Stub 头文件
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

// 包含 ATen CUDA JIT 实用工具相关头文件
#include <ATen/native/cuda/jit_utils.h>

// 包含 c10 CUDA Guard 相关头文件
#include <c10/cuda/CUDAGuard.h>

// 包含 Torch JIT 资源保护相关头文件
#include <torch/csrc/jit/resource_guard.h>

// 包含 CUDA 运行时相关头文件
#include <cuda_runtime.h>

// 包含 C++ 标准库头文件
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

// 定义命名空间 torch::jit::fuser::cuda 中的所有内容
namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// 返回全局 CUDA 上下文中的 NVRTC 对象
const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

// 查询代码生成输出的架构和目标
void codegenOutputQuery(
    const cudaDeviceProp* const prop,  // CUDA 设备属性指针
    int& major,                        // 主版本号
    int& minor,                        // 次版本号
    bool& compile_to_sass) {           // 是否编译为 SASS
#ifdef USE_ROCM
  // 检查 NVRTC 版本并设置编译标志
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&major, &minor));
  compile_to_sass = false;
#else
  // 定义 CUDA 版本类型
  using CudaVersion = std::pair<int, int>;
  CudaVersion nvrtc_version;

  // 获取 NVRTC 版本信息
  AT_CUDA_NVRTC_CHECK(
      nvrtc().nvrtcVersion(&nvrtc_version.first, &nvrtc_version.second));

  // 检查 NVRTC 版本是否支持当前设备
  TORCH_CHECK(
      nvrtc_version.first >= 6,
      "NVRTC versions less than 6 are not supported. Is: ",
      nvrtc_version.first);

  // 根据设备支持的最大驱动版本设置最大设备版本
  const CudaVersion dev_version = CudaVersion(prop->major, prop->minor);
  CudaVersion max_dev_version;

  // 根据 NVRTC 版本设置支持的最大设备版本
  if (nvrtc_version.first <= 7) {
    max_dev_version = CudaVersion(5, 0);
  } else if (nvrtc_version.first <= 8) {
    max_dev_version = CudaVersion(6, 0);
  } else if (nvrtc_version.first <= 9) {
    max_dev_version = CudaVersion(7, 2);
  } else if (nvrtc_version.first <= 10) {
    max_dev_version = CudaVersion(7, 5);
  } else if (nvrtc_version == CudaVersion(11, 0)) {
    max_dev_version = CudaVersion(8, 0);
  } else if (nvrtc_version.first == 11 && nvrtc_version.second < 8) {
    max_dev_version = CudaVersion(8, 6);
  } else {
    max_dev_version = dev_version;
  }

  // 根据设备和最大设备版本确定编译目标是否为 SASS
  if (dev_version > max_dev_version) {
    major = max_dev_version.first;
    minor = max_dev_version.second;
    compile_to_sass = false;
  } else {
    major = dev_version.first;
    minor = dev_version.second;
    compile_to_sass = true;
  }
#endif
}

// 构造函数：编译指定的内核并存储运行所需的元数据
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
FusedKernelCUDA::FusedKernelCUDA(
    at::DeviceIndex device,              // 设备索引
    std::string name,                   // 内核名称
    std::string code,                   // 内核代码
    std::vector<TensorDesc> input_desc,  // 输入张量描述
    std::vector<TensorDesc> output_desc) // 输出张量描述
    // 定义构造函数，接受多个参数：名称（name）、代码（code）、输入描述（input_desc）、
    // 输出描述（output_desc）、分块描述（chunk_desc）、连接描述（concat_desc）、是否随机（has_random）
    : FusedKernel(
          std::move(name),               // 使用 std::move 将名称移动到构造函数中
          std::move(code),               // 使用 std::move 将代码移动到构造函数中
          std::move(input_desc),         // 使用 std::move 将输入描述移动到构造函数中
          std::move(output_desc),        // 使用 std::move 将输出描述移动到构造函数中
          std::move(chunk_desc),         // 使用 std::move 将分块描述移动到构造函数中
          std::move(concat_desc),        // 使用 std::move 将连接描述移动到构造函数中
          has_random),                   // 将是否随机传递给基类构造函数，并完成初始化
    device_(device) {                    // 初始化成员变量 device_
    
    // 初始化 CUDA 上下文（如果需要）
    at::cuda::jit::initializeCudaContext();
    
    // 注意：因为在某些情况下 at::DeviceGuard 无法正常工作，所以这里使用了 hacked at::DeviceGuard
    // 获取当前 CUDA 设备并设置为构造函数传入的设备 ID
    const auto prior_device = at::cuda::current_device();
    at::cuda::set_device(device_);
    
    // 获取设备和 NVRTC 属性（用于编译架构和资源占用计算）
    prop_ = at::cuda::getCurrentDeviceProperties();
    
    // 定义变量用于存储主版本号、次版本号和编译到 SASS（CUDA 汇编）的标志
    int major, minor;
    bool compile_to_sass = false;
    
    // 调用函数获取代码生成的输出信息
    codegenOutputQuery(prop_, major, minor, compile_to_sass);
    
    // 创建 NVRTC 程序
    nvrtcProgram program;
    // 使用 nvrtc().nvrtcCreateProgram 函数创建 NVRTC 程序，传递代码字符串和必要的参数
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
        &program, code_.c_str(), nullptr, 0, nullptr, nullptr));
#if defined(USE_ROCM)
  // 如果定义了 USE_ROCM，则使用 HIP 的预编译头选项
  std::vector<const char*> args = {"--std=c++17"};
  // 添加 HIP 的预编译头选项到参数列表中
  args.push_back("-hip-pch");
#else
  // 如果未定义 USE_ROCM，则设置 CUDA 的计算能力字符串
  const std::string compute = std::string("--gpu-architecture=") +
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11010
      // CUDA 11.1 允许直接使用 SASS（sm_）而不是 PTX（compute_）
      // 这提供了更好的向后兼容性，以便在旧驱动程序上工作，
      // （因为旧驱动程序不一定能识别新工具包生成的 PTX）；
      // 同时，为了向前兼容性（未来设备可能不支持 `compile_to_sass==false`），
      // 因为 SASS 不一定兼容，我们回退到使用 PTX。
      (compile_to_sass ? "sm_" : "compute_") +
#else
      "compute_" +
#endif
      std::to_string(major) + std::to_string(minor);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 构建参数列表，包括标准版本、计算能力字符串和默认设备选项
  const std::vector<const char*> args = {
      "--std=c++17", compute.c_str(), "-default-device"};
#endif

  // 编译程序并获取结果
  const auto result =
      nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
  if (result != NVRTC_SUCCESS) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 获取编译日志的大小
    size_t logsize;
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetProgramLogSize(program, &logsize));
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 申请存储编译日志的缓冲区
    std::vector<char> log(logsize);
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetProgramLog(program, log.data()));
    // 将日志内容转为字符串流并抛出运行时错误
    std::stringstream cu;
    cu << log.data();
    throw std::runtime_error(cu.str());
  }

  // 在程序结束时销毁编译程序的资源
  ResourceGuard holdProgram(
      [&] { AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });
  // 检查编译结果状态
  AT_CUDA_NVRTC_CHECK(result);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 获取生成的 PTX 或 CUBIN 的大小
  size_t ptx_size;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11010
  // 根据 compile_to_sass 确定是生成 SASS 还是 PTX，使用不同的 API
  const auto getSize = compile_to_sass
      ? at::globalContext().getNVRTC().nvrtcGetCUBINSize
      : at::globalContext().getNVRTC().nvrtcGetPTXSize;
  const auto getFunc = compile_to_sass
      ? at::globalContext().getNVRTC().nvrtcGetCUBIN
      : at::globalContext().getNVRTC().nvrtcGetPTX;
#else
  // 使用 PTX 的获取大小和获取函数的 API
  const auto getSize = at::globalContext().getNVRTC().nvrtcGetPTXSize;
  const auto getFunc = at::globalContext().getNVRTC().nvrtcGetPTX;
#endif

  // 获取生成的 PTX 或 CUBIN 的大小并分配内存
  AT_CUDA_NVRTC_CHECK(getSize(program, &ptx_size));
  ptx_.resize(ptx_size);

  // 获取生成的 PTX 或 CUBIN 的内容
  AT_CUDA_NVRTC_CHECK(getFunc(program, ptx_.data()));

  // 加载生成的模块数据到 CUDA 模块中
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&module_, ptx_.data()));
  // 获取 CUDA 函数句柄
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuModuleGetFunction(&function_, module_, name_.c_str()));

  // 计算每个多处理器的最大块数
  AT_CUDA_DRIVER_CHECK(nvrtc().cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks_, function_, 128, 0));
  maxBlocks_ *= prop_->multiProcessorCount;

  // 恢复设备状态
  at::cuda::set_device(prior_device);
}

// 计算向上取整的除法结果
static int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

// 启动 CUDA 的原始融合内核
void FusedKernelCUDA::launch_raw(
    const uint32_t numel,
    std::vector<void*>& arguments) const {
  // 设置当前设备为指定的 CUDA 设备
  at::cuda::CUDAGuard{device_};
  // 使用 hack 后的 at::DeviceGuard（参见上文注释）
  const auto prior_device = at::cuda::current_device();
  // 将当前 CUDA 设备切换为指定的 device_
  at::cuda::set_device(device_);

  // 计算需要的线程块数，确保不超过设定的最大块数
  const auto nBlocks = std::min(maxBlocks_, ceilDiv(numel, kBlockSize));

  // 如果需要添加随机状态参数
  // 注意：这里定义 philox_engine_inputs 是为了确保其生命周期延续到 kernel 启动
  std::pair<uint64_t, uint64_t> philox_engine_inputs;
  if (has_random_) {
    // 计算随机数生成器的偏移量
    const auto rand_offset =
        4 * (std::ceil(numel / (4.0 * kBlockSize * nBlocks)) + 1);
    // 获取当前 CUDA 默认的生成器
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    {
      // 见注释 [Acquire lock when using random generators]，在使用随机生成器时需要获取锁
      std::lock_guard<std::mutex> lock(gen.mutex());
      // 生成随机数引擎的输入参数
      philox_engine_inputs =
          at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
              rand_offset);
    }
    // 将生成的随机数引擎参数添加到 arguments 中
    arguments.push_back(&philox_engine_inputs.first);
    arguments.push_back(&philox_engine_inputs.second);
  }

  // 在当前流上启动 kernel
  auto stream = at::cuda::getCurrentCUDAStream();
  // 使用 NVRTC 接口启动 CUDA kernel
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      function_,
      nBlocks,
      1,
      1,
      kBlockSize,
      1,
      1,
      0,
      stream,
      arguments.data(),
      nullptr));

  // 恢复先前的 CUDA 设备（参见上文 at::DeviceGuard 的注释）
  at::cuda::set_device(prior_device);
}

// 析构函数，用于释放 CUDA 融合内核模块
FusedKernelCUDA::~FusedKernelCUDA() {
  // 使用 NVRTC API 卸载 CUDA 模块
  nvrtc().cuModuleUnload(module_);
}

// 静态函数，创建 CUDA 融合内核的共享指针
static std::shared_ptr<FusedKernel> createFusionKernel(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random) {
  return std::make_shared<FusedKernelCUDA>(
      static_cast<at::DeviceIndex>(device),
      std::move(name),
      std::move(code),
      std::move(input_desc),
      std::move(output_desc),
      std::move(chunk_desc),
      std::move(concat_desc),
      has_random);
}

// 注册 CUDA 设备的融合后端
RegisterFusionBackend reg(DeviceType::CUDA, createFusionKernel);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
```