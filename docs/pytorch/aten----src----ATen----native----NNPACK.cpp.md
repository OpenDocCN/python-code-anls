# `.\pytorch\aten\src\ATen\native\NNPACK.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#include <c10/util/CallOnce.h>

#include <thread>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nnpack_available_native.h>
#include <ATen/ops/_nnpack_spatial_convolution_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#if !AT_NNPACK_ENABLED()

namespace at::native {

// 如果没有启用NNPACK支持，则抛出运行时错误
at::Tensor _nnpack_spatial_convolution(
    const Tensor& input,
    const Tensor& weight, const std::optional<Tensor>& bias_opt,
    const IntArrayRef padding,
    const IntArrayRef stride) {
  throw std::runtime_error(
      "nnpack_spatial_convolution: ATen not compiled with NNPACK support");
}

// 如果没有启用NNPACK支持，则返回false
bool _nnpack_available() {
  return false;
}

} // namespace at::native

#else

#include <nnpack.h>

#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

namespace at::native {

// 初始化NNPACK，使用静态变量确保只初始化一次
static bool init_nnpack() {
  static c10::once_flag once_;
  static bool nnpack_successfully_initialized_ = false;

  c10::call_once(once_, []() {
    const nnp_status nnpack_status = nnp_initialize();
    nnpack_successfully_initialized_ = (nnp_status_success == nnpack_status);

    // 处理初始化NNPACK失败的情况
    if (nnpack_status != nnp_status_success) {
      if (nnpack_status == nnp_status_out_of_memory) {
        LOG(WARNING) << "Could not initialize NNPACK! Reason: Out of memory.";
      } else if (nnpack_status == nnp_status_unsupported_hardware) {
        LOG(WARNING) << "Could not initialize NNPACK! Reason: Unsupported hardware.";
      } else {
        LOG(WARNING) << "Could not initialize NNPACK! Reason: Unknown error!";
      }
    }
  });

  return nnpack_successfully_initialized_;
}

// 返回NNPACK的线程池对象
static pthreadpool_t nnpack_threadpool() {
#ifdef C10_MOBILE
  return caffe2::pthreadpool_();
#else
  static pthreadpool_t nnpack_threadpool_ = nullptr;
  static bool called_nnpack_threadpool_ = false;

  if (!called_nnpack_threadpool_) {
    called_nnpack_threadpool_ = true;

#ifdef INTRA_OP_PARALLEL
    const uint32_t threads = at::get_num_threads();
#else
    const uint32_t threads = std::thread::hardware_concurrency();
#endif

    // 创建NNPACK的线程池
    nnpack_threadpool_ = pthreadpool_create(threads);
    if (!nnpack_threadpool_) {
      LOG(WARNING) << "Failed to initialize pthreadpool! Running NNPACK in single-threaded mode.";
    }
  }

  return nnpack_threadpool_;
#endif
}

// 检查NNPACK是否可用
bool _nnpack_available() {
  return init_nnpack();
}

namespace {
// 用于存储NNPACK的工作空间
struct Workspace {
  void* buffer = nullptr;
  size_t size = 0;

  // 释放工作空间的内存
  void deallocate() {
    if (buffer) {
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      std::free(buffer);
      buffer = nullptr;
    }
  }

  // 分配工作空间的内存
  void allocate() {
    deallocate();

    // NNPACK有内存对齐的要求
    constexpr size_t nnpack_memory_alignment_boundary = 64;

    // 在此分配NNPACK所需的内存
    // 使用 posix_memalign 分配内存，并将结果存储在 buffer 中
    auto res = posix_memalign(&buffer, nnpack_memory_alignment_boundary, size);
    // 检查 posix_memalign 的返回值，如果不为 0，说明分配失败
    if (res != 0) {
      // 抛出错误信息，包括 posix_memalign 失败的原因和错误码
      TORCH_CHECK(false, "posix_memalign failed:", strerror(errno), " (", errno, ")");
    }
    // 返回，结束函数
    return;
  }

  // Workspace 类的析构函数
  ~Workspace() {
    // 调用 deallocate 方法释放内存
    deallocate();
  }


这些注释按照要求详细解释了每行代码的功能和作用，确保了代码逻辑和操作的清晰性。
// } 是命名空间的结束符号
};
// 命名空间的结束

// 在多线程同时运行Conv时，使用thread_local关键字确保Workspace的线程安全性
static thread_local Workspace workspace;

// 定义NNPack的空间卷积函数，接受输入、权重、可选的偏置、填充和步幅作为参数
Tensor _nnpack_spatial_convolution(
    const Tensor& input,
    const Tensor& weight, const std::optional<Tensor>& bias_opt,
    const IntArrayRef padding,
    const IntArrayRef stride) {
  // 使用c10::borrow_from_optional_tensor函数从可选的Tensor中获取bias，返回MaybeOwned类型的bias_maybe_owned
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 解引用MaybeOwned，得到bias Tensor
  const Tensor& bias = *bias_maybe_owned;

  // 创建一个空的Tensor作为输出，其形状由输入、权重、填充和步幅决定
  at::Tensor output = at::empty(
      conv_output_size(input.sizes(), weight.sizes(), padding, stride),
      input.options());

  // 确保输入Tensor的维度是4
  if (input.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolutionOutput expects 4D input Tensor N,C,H,W");
  }
  // 确保权重Tensor的维度是4
  if (weight.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolutionOutput expects 4D weight Tensor oC,iC,kH,kW");
  }
  // 确保输出Tensor的维度是4
  if (output.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolutionOutput expects 4D output Tensor N,oC,oH,oW");
  }

  // 检查输入通道数与权重Tensor的输入通道数是否匹配
  if (input.size(1) != weight.size(1)) {
    std::stringstream err;
    err << "Mismatch between number of input channels in input Tensor ("
        << input.size(1) << ") and weight Tensor (" << weight.size(1)
        << ") in NNPack convolutionOutput";
    throw std::runtime_error(err.str());
  }
  // 检查权重Tensor的输出通道数与输出Tensor的通道数是否匹配
  if (weight.size(0) != output.size(1)) {
    std::stringstream err;
    err << "Mismatch between number of output channels in weight Tensor ("
        << weight.size(0) << ") and output Tensor (" << output.size(1)
        << ") in NNPack convolutionOutput";
    throw std::runtime_error(err.str());
  }
  // 检查输入Tensor的批量大小与输出Tensor的批量大小是否匹配
  if (input.size(0) != output.size(0)) {
    std::stringstream err;
    err << "Mismatch between batch size in input Tensor (" << input.size(0)
        << ") and output Tensor (" << output.size(0)
        << ") in NNPack convolutionOutput";
    throw std::runtime_error(err.str());
  }

  // 确保所有Tensor都是float类型的Tensor，且存储在CPU上
  if (input.device().type() != kCPU || input.scalar_type() != kFloat ||
      weight.device().type() != kCPU || weight.scalar_type() != kFloat ||
      output.device().type() != kCPU || output.scalar_type() != kFloat ||
      (bias.defined() && (bias.device().type() != kCPU || bias.scalar_type() != kFloat))) {
    // 抛出运行时错误，指示 NNPack 卷积输出中的张量类型不匹配
    throw std::runtime_error(
        "Mismatched Tensor types in NNPack convolutionOutput");
  }

  // 自动选择 NNpack 卷积算法
  const auto algorithm = nnp_convolution_algorithm_auto;
  // 获取输入张量的通道数
  const size_t input_channels = input.size(1);
  // 获取输出张量的通道数
  const size_t output_channels = weight.size(0);
  // 定义输入大小结构体
  const struct nnp_size input_size = {
      .width = (size_t)input.size(3),
      .height = (size_t)input.size(2),
  };
  // 定义输入填充结构体
  const struct nnp_padding input_padding = {
      .top = (size_t)padding[0],
      .right = (size_t)padding[1],
      .bottom = (size_t)padding[0],
      .left = (size_t)padding[1],
  };
  // 定义卷积核大小结构体
  const struct nnp_size kernel_size = {
      .width = (size_t)weight.size(3),
      .height = (size_t)weight.size(2),
  };
  // 定义输出大小结构体
  const struct nnp_size output_size = {
      .width = (size_t)output.size(3),
      .height = (size_t)output.size(2),
  };
  // 定义输出子采样大小结构体
  const nnp_size output_subsample = {
      .width = static_cast<std::size_t>(stride[1]),
      .height = static_cast<std::size_t>(stride[0]),
  };

  // 获取连续存储的输入张量
  const auto input_ = input.contiguous();
  // 获取连续存储的权重张量
  const auto weight_ = weight.contiguous();
  // 如果没有定义偏置张量，则创建一个用零填充的偏置张量
  const auto bias_ = bias.defined() ? bias.contiguous() : at::zeros({weight.size(0)}, input.options());

  // 定义计算函数，用于执行卷积计算
  const auto compute = [&](const size_t batch_size) -> nnp_status {
    // 如果批次大小为1或者输出子采样宽度或高度不为1，则执行以下逻辑
    if ((batch_size == 1) || (output_subsample.width != 1) || (output_subsample.height != 1)) {
      // 计算每个批次的输入数据大小
      const size_t input_size_per_batch = input_channels * input_size.width * input_size.height;
      // 计算每个批次的输出数据大小
      const size_t output_size_per_batch = output_channels * output_size.width * output_size.height;

      // 遍历批次范围内的每个批次
      for (const auto batch : c10::irange(0u, batch_size)) {
        // 调用 NNpack 推理函数进行卷积计算
        const nnp_status status = nnp_convolution_inference(
            algorithm,
            nnp_convolution_transform_strategy_compute,
            input_channels,
            output_channels,
            input_size,
            input_padding,
            kernel_size,
            output_subsample,
            input_.data_ptr<float>() + batch * input_size_per_batch,
            weight_.data_ptr<float>(),
            bias_.data_ptr<float>(),
            output.data_ptr<float>() + batch * output_size_per_batch,
            workspace.buffer,
            &workspace.size,
            nnp_activation_identity,
            nullptr,
            nnpack_threadpool(),
            nullptr );

        // 如果计算失败，则返回错误状态
        if (nnp_status_success != status) {
          return status;
        }
      }

      // 所有批次计算成功，返回成功状态
      return nnp_status_success;
    }
    else {
      // 调用 NNpack 输出函数进行卷积计算
      return nnp_convolution_output(
        algorithm,
        batch_size,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        input_.data_ptr<float>(),
        weight_.data_ptr<float>(),
        bias_.data_ptr<float>(),
        output.data_ptr<float>(),
        workspace.buffer,
        &workspace.size,
        nnp_activation_identity,
        nullptr,
        nnpack_threadpool(),
        nullptr );
  }
};

const size_t batch_size = input.size(0);

auto size_and_allocate_ws = [&]() {
  // 运行一次以获取内存工作空间缓冲区的大小
  const auto status = compute(batch_size);
  if (status != nnp_status_success) {
    throw std::runtime_error("NNPACK SpatialConvolution_updateOutput failed");
  }
  workspace.allocate();  // 分配工作空间
};

// 如果尚未创建工作空间，则进行分配
if (workspace.buffer == nullptr) {
  size_and_allocate_ws();
}

// 尝试使用新创建的或现有的工作空间运行
auto status = compute(batch_size);

if (status == nnp_status_insufficient_buffer) {
  // 需要重新分配工作空间
  workspace.deallocate();  // 释放工作空间
  size_and_allocate_ws();  // 重新分配工作空间

  // 再次尝试运行
  status = compute(batch_size);
}

if (status != nnp_status_success) {
  throw std::runtime_error("NNPACK SpatialConvolution_updateOutput failed");
}

return output;
}

} // namespace at::native

#endif // AT_NNPACK_ENABLED
```