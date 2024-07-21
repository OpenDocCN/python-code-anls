# `.\pytorch\aten\src\ATen\native\cudnn\LossCTC.cpp`

```py
namespace at {
namespace native {

// 声明是否仅在操作符方法中使用断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h> // 包含 ATen 的配置文件
#include <ATen/core/Tensor.h> // 包含 ATen 的 Tensor 类定义
#include <ATen/cuda/CUDAConfig.h> // 包含 ATen 的 CUDA 配置
#include <ATen/cuda/CUDAGraphsUtils.cuh> // 包含 ATen CUDA 图形工具

#if AT_CUDNN_ENABLED()
#include <ATen/cudnn/Descriptors.h> // 如果启用了 cuDNN，则包含 cuDNN 描述符
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h> // 如果未启用 AT_PER_OPERATOR_HEADERS，包含 ATen 函数定义
#include <ATen/NativeFunctions.h> // 如果未启用 AT_PER_OPERATOR_HEADERS，包含 ATen 原生函数定义
#else
#include <ATen/ops/_cudnn_ctc_loss.h> // 如果启用 AT_PER_OPERATOR_HEADERS，包含 cuDNN CTC loss 相关头文件
#include <ATen/ops/_cudnn_ctc_loss_native.h> // 如果启用 AT_PER_OPERATOR_HEADERS，包含 cuDNN CTC loss 原生实现头文件
#include <ATen/ops/_use_cudnn_ctc_loss.h> // 如果启用 AT_PER_OPERATOR_HEADERS，包含使用 cuDNN CTC loss 的头文件
#include <ATen/ops/_use_cudnn_ctc_loss_native.h> // 如果启用 AT_PER_OPERATOR_HEADERS，包含使用 cuDNN CTC loss 原生实现的头文件
#include <ATen/ops/empty.h> // 如果启用 AT_PER_OPERATOR_HEADERS，包含 empty 相关头文件
#include <ATen/ops/empty_like.h> // 如果启用 AT_PER_OPERATOR_HEADERS，包含 empty_like 相关头文件
#endif

#if (!AT_CUDNN_ENABLED())

namespace at {
namespace native {

// 如果未启用 cuDNN，定义以下函数

bool _use_cudnn_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK) {
  return false; // 返回 false，表示不使用 cuDNN CTC loss
}

bool _use_cudnn_ctc_loss_tensor(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK) {
  return false; // 返回 false，表示不使用 cuDNN CTC loss（接受张量形式输入）
}

std::tuple<Tensor, Tensor> _cudnn_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK,
    bool deterministic,
    bool zero_infinity) {
  AT_ERROR("cudnn_ctc_loss: ATen not compiled with cuDNN >= 7 support"); // 报错信息，表示 ATen 未使用 cuDNN 版本 >= 7 的支持
}

std::tuple<Tensor, Tensor> _cudnn_ctc_loss_tensor(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK,
    bool deterministic,
    bool zero_infinity) {
  AT_ERROR("cudnn_ctc_loss: ATen not compiled with cuDNN >= 8 support"); // 报错信息，表示 ATen 未使用 cuDNN 版本 >= 8 的支持
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED

#include <ATen/cudnn/Descriptors.h> // 如果启用了 cuDNN，包含 cuDNN 描述符
#include <ATen/cudnn/Types.h> // 如果启用了 cuDNN，包含 cuDNN 类型
#include <ATen/cudnn/Utils.h> // 如果启用了 cuDNN，包含 cuDNN 工具

#include <ATen/TensorUtils.h> // 包含 ATen 张量工具
#include <c10/util/irange.h> // 包含 c10 中的 irange 工具

namespace at {
namespace native {

namespace {
// 缓存：标记之前是否未通过目标长度检查
static bool tensor_failed_target_lengths_check = false;
} // namespace

bool _use_cudnn_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK) {
  auto& ctx = at::globalContext(); // 获取全局上下文

  bool use_cudnn = ctx.userEnabledCuDNN() && (BLANK == 0) &&
      (targets.dim() == 1) && (log_probs.scalar_type() == at::kFloat) &&
      (targets.scalar_type() == at::kInt) &&
      (targets.device().type() == at::kCPU) && (targets.is_contiguous()) &&
      (log_probs.device().type() == at::kCUDA) && (log_probs.dim() == 3);

  if (use_cudnn) {
    // 我们不知道 input_lengths 和 target_lengths 的大小是否相同
    // （它们应该相同，但我们还没有检查）
    int64_t max_input_length = log_probs.size(0);
    for (const auto input_length : input_lengths) {
      use_cudnn = use_cudnn && ((input_length == max_input_length) ? 1 : 0); // 检查 input_lengths 中每个值是否等于 max_input_length
    }
    for (const auto b : c10::irange(target_lengths.size())) {
        // 遍历目标长度数组，使用范围迭代器，其中 c10::irange 是一个范围生成器
        // 检查目标长度是否小于 256，因为文档指出小于 256 的长度是合法的，但是我们观察到当目标长度大于输入长度时会出现非法内存访问问题
        use_cudnn = use_cudnn && (target_lengths[b] < 256) &&
            (target_lengths[b] <= input_lengths[b]);
        // 更新 use_cudnn 变量，检查是否继续使用 CuDNN，条件是目标长度小于 256 并且小于等于对应的输入长度
    }
  }
  return use_cudnn;
}
// 结束函数定义

bool _use_cudnn_ctc_loss_tensor(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK) {
  // 获取当前全局上下文
  auto& ctx = at::globalContext();

  // 判断是否可以使用 CuDNN 加速的条件
  bool use_cudnn = ctx.userEnabledCuDNN() && (BLANK == 0) &&
      (targets.dim() == 1) && (log_probs.scalar_type() == at::kFloat) &&
      (targets.scalar_type() == at::kInt) &&
      (log_probs.device().type() == at::kCUDA) && (targets.is_contiguous()) &&
      (log_probs.dim() == 3) && (input_lengths.scalar_type() == at::kInt) &&
      (target_lengths.scalar_type() == at::kInt);

  // 如果当前 CUDA 流不处于捕获状态
  if (at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None) {
    // 将目标长度转移到 CPU 设备，并保证连续性
    Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
    IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
    // 遍历目标长度数组
    for (const auto b : c10::irange(tl.size())) {
      // 将输入长度和目标长度转移到 CPU 设备，并保证连续性
      Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
      Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
      IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
      IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
      // 更新 use_cudnn 变量，检查目标长度是否符合条件
      use_cudnn = use_cudnn && (tl[b] < 256) && (tl[b] <= il[b]);
      // 如果条件不符合，设置标志并跳出循环
      if (!use_cudnn) {
        tensor_failed_target_lengths_check = true;
        break;
      }
    }
  } else {
    // 在 CUDA 流捕获状态下，更新 use_cudnn 变量，并进行相关警告
    use_cudnn = use_cudnn && !tensor_failed_target_lengths_check;
    if (tensor_failed_target_lengths_check) {
      TORCH_WARN(
          "cuDNN max target length restriction < 256 cannot be checked during graph capture,"
          " but target length >= 256 was observed previously e.g., during warmup, so we"
          " presume it is unsafe to dispatch to cuDNN ctc_loss.");
    }
  }

  // 返回是否可以使用 CuDNN 加速的结果
  return use_cudnn;
}

std::tuple<Tensor, Tensor> _cudnn_ctc_loss(
    const Tensor& log_probs_t,
    const Tensor& targets_t,
    IntArrayRef input_lengths_,
    IntArrayRef target_lengths_,
    int64_t BLANK,
    bool deterministic,
    // 忽略 zero_infinity 参数，仅用于向后兼容
    (void)zero_infinity;
    // 定义常量 c，用于调试目的
    const CheckedFrom c = "cudnn_ctc_loss";
    // 定义 log_probs 和 targets 张量参数
    const TensorArg log_probs{log_probs_t, "log_probs", 1};
    const TensorArg targets{targets_t, "targets", 2};
    // 检查 log_probs 张量的维度是否为 3
    checkDim(c, log_probs, 3);
    // 检查 log_probs 张量的数据类型是否为 kFloat
    checkScalarType(c, log_probs, kFloat);
    // 检查 targets 张量的维度是否为 1
    checkDim(c, targets, 1);
    // 检查 targets 张量的数据类型是否为 kInt
    checkScalarType(c, targets, kInt);
    // 检查 targets 张量是否为连续存储
    checkContiguous(c, targets);
    // 检查 log_probs 张量是否在 CUDA 后端
    checkBackend(c, {*log_probs}, Backend::CUDA);
    // 检查 targets 张量是否在 CPU 后端
    checkBackend(c, {*targets}, Backend::CPU);
    // 获取 log_probs 张量的批量大小
    const auto batch_size = log_probs->size(1);
    // 检查 input_lengths_ 和 target_lengths_ 的大小是否与批量大小相同
    TORCH_CHECK(
        static_cast<int64_t>(input_lengths_.size()) == batch_size,
        "input_lengths needs to have size to match batch_size");
    TORCH_CHECK(
        static_cast<int64_t>(target_lengths_.size()) == batch_size,
        "target_lengths needs to have size to match batch_size");
    
    // 将 input_lengths_ 和 target_lengths_ 转换为 std::vector<int>
    std::vector<int> input_lengths(input_lengths_.begin(), input_lengths_.end());
    std::vector<int> target_lengths(
        target_lengths_.begin(), target_lengths_.end());
    
    // 检查 BLANK 是否为 0，对于 cudnn_ctc_loss 必须是标签 0
    TORCH_CHECK(BLANK == 0, "blank must be label 0 for cudnn_ctc_loss");
    
    // 获取 CuDNN 的句柄
    const auto handle = getCudnnHandle();
    
    // 根据 deterministic 确定使用的 CuDNN CTC Loss 算法
    const cudnnCTCLossAlgo_t algo =
        (deterministic ? CUDNN_CTC_LOSS_ALGO_DETERMINISTIC
                       : CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC);
    
    // 创建 CTCLossDescriptor 对象
    CTCLossDescriptor ctc_loss_desc;
    
    // 设置 CuDNN CTC Loss 描述符的属性
    ctc_loss_desc.setEx(
        CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN);
    
    // 创建 log_probs_t 张量的描述符
    TensorDescriptor log_probs_desc{log_probs_t};
    
    // 根据 log_probs_t 张量创建与其形状相同的空张量 grad
    Tensor grad = at::empty_like(log_probs_t, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    // 创建 grad 张量的描述符
    TensorDescriptor grad_desc{grad};
    
    // 计算所需的 workspace 大小
    size_t workspace_size;
    AT_CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize(
        handle,
        log_probs_desc.desc(),
        grad_desc.desc(),
        targets->data_ptr<int>(),
        target_lengths.data(),
        input_lengths.data(),
        algo,
        ctc_loss_desc.desc(),
        &workspace_size));
    
    // 创建指定大小的 workspace 张量，数据类型为 kByte
    Tensor workspace =
        at::empty(workspace_size, log_probs->options().dtype(kByte));
    // 创建与 log_probs_t 形状相同的 costs 张量
    Tensor costs = at::empty({log_probs->size(1)}, log_probs->options());
    
    // 执行 CuDNN CTC Loss 操作
    AT_CUDNN_CHECK(cudnnCTCLoss(
        handle,
        log_probs_desc.desc(),
        log_probs_t.data_ptr(),
        targets->data_ptr<int>(),
        target_lengths.data(),
        input_lengths.data(),
        costs.data_ptr(),
        grad_desc.desc(),
        grad.data_ptr(),
        algo,
        ctc_loss_desc.desc(),
        workspace.data_ptr(),
        workspace_size));
    
    // 返回 costs 和 grad 张量作为结果
    return std::make_tuple(costs, grad);
} // 结束 namespace native

} // 结束 namespace at

#endif



std::tuple<Tensor, Tensor> _cudnn_ctc_loss_tensor(
    const Tensor& log_probs_t,
    const Tensor& targets_t,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK,
    bool deterministic,
    bool zero_infinity) {
  Tensor targets_t_ = targets_t;
  if (targets_t.device().type() == at::kCPU) {
    targets_t_ = targets_t.to(Device(at::kCUDA));
  }
  const CheckedFrom c = "cudnn_ctc_loss";
  const TensorArg log_probs{log_probs_t, "log_probs", 1};
  const TensorArg targets{targets_t_, "targets", 2};
  checkDim(c, log_probs, 3); // 检查 log_probs 张量的维度是否为 3
  checkScalarType(c, log_probs, kFloat); // 检查 log_probs 张量的标量类型是否为 float
  checkDim(c, targets, 1); // 检查 targets 张量的维度是否为 1
  checkScalarType(c, targets, kInt); // 检查 targets 张量的标量类型是否为 int
  checkContiguous(c, targets); // 检查 targets 张量是否是连续的
  checkBackend(c, {*log_probs}, Backend::CUDA); // 检查 log_probs 张量的后端是否为 CUDA
  checkBackend(c, {*targets}, Backend::CUDA); // 检查 targets 张量的后端是否为 CUDA
  const auto batch_size = log_probs->size(1); // 获取 log_probs 张量的第二个维度大小作为批处理大小
  int64_t input_lengths_size =
      input_lengths.sizes().size() ? input_lengths.size(0) : 1; // 获取 input_lengths 张量的大小，若未提供则默认为 1
  int64_t target_lengths_size =
      target_lengths.sizes().size() ? target_lengths.size(0) : 1; // 获取 target_lengths 张量的大小，若未提供则默认为 1
  TORCH_CHECK(
      input_lengths_size == batch_size,
      "input_lengths needs to have size to match batch_size"); // 检查 input_lengths 张量的大小是否与批处理大小相匹配
  TORCH_CHECK(
      target_lengths_size == batch_size,
      "target_lengths needs to have size to match batch_size"); // 检查 target_lengths 张量的大小是否与批处理大小相匹配

  TORCH_CHECK(BLANK == 0, "blank must be label 0 for cudnn_ctc_loss"); // 检查 BLANK 是否为 0，因为 cudnn_ctc_loss 要求空白标签为 0
  // 在分发函数中检查：
  // 断言 cudnnCTCLoss 的其他条件：所有标签长度 <= 256，所有输入长度 = log_probs.size(0)

  const auto handle = getCudnnHandle(); // 获取 cuDNN 的句柄

  const cudnnCTCLossAlgo_t algo =
      (deterministic ? CUDNN_CTC_LOSS_ALGO_DETERMINISTIC
                     : CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC); // 根据 deterministic 参数选择 cudnnCTCLoss 的算法

  CTCLossDescriptor ctc_loss_desc; // 创建 CTCLossDescriptor 对象

  ctc_loss_desc.set_v8_v9(
      CUDNN_DATA_FLOAT,
      CUDNN_LOSS_NORMALIZATION_SOFTMAX,
      CUDNN_PROPAGATE_NAN,
      255); // 配置 CTCLoss 的描述符，设置数据类型为浮点型，损失归一化为 softmax，传播 NaN，标签的最大长度为 255
  TensorDescriptor log_probs_desc{log_probs_t}; // 创建 log_probs 张量的描述符对象
  Tensor grad = at::empty_like(log_probs_t, LEGACY_CONTIGUOUS_MEMORY_FORMAT); // 创建与 log_probs_t 相同大小和内存格式的空张量 grad
  TensorDescriptor grad_desc{grad}; // 创建 grad 张量的描述符对象

  size_t workspace_size; // 定义工作空间的大小变量
  AT_CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize_v8(
      handle,
      algo,
      ctc_loss_desc.desc(),
      log_probs_desc.desc(),
      grad_desc.desc(),
      &workspace_size)); // 获取 cudnnCTCLoss 所需的工作空间大小
  Tensor workspace =
      at::empty(workspace_size, log_probs->options().dtype(kByte)); // 创建指定大小和数据类型的工作空间张量
  Tensor costs = at::empty({log_probs->size(1)}, log_probs->options()); // 创建与批处理大小相等的成本张量

  AT_CUDNN_CHECK(cudnnCTCLoss_v8(
      handle,
      algo,
      ctc_loss_desc.desc(),
      log_probs_desc.desc(),
      log_probs_t.data_ptr(),
      targets_t_.data_ptr<int>(),
      target_lengths.data_ptr<int>(),
      input_lengths.data_ptr<int>(),
      costs.data_ptr(),
      grad_desc.desc(),
      grad.data_ptr(),
      workspace_size,
      workspace.data_ptr()

          )); // 调用 cudnnCTCLoss_v8 计算 CTC 损失，并计算梯度
  return std::make_tuple(costs, grad); // 返回损失和梯度的元组
}
```