# `.\pytorch\aten\src\ATen\native\quantized\cpu\ChannelShuffle.cpp`

```py
// 定义预处理指令，仅限于方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量操作的头文件
#include <ATen/core/Tensor.h>
// 包含封装内核函数的头文件
#include <ATen/core/boxing/KernelFunction.h>
// 包含量化 QNNPACK 初始化的头文件
#include <ATen/native/quantized/cpu/init_qnnpack.h>
// 包含 QNNPACK 工具函数的头文件
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
// 包含线程池实用工具的头文件
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

// 如果未定义每个操作符的头文件，则包含通用的 Native 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
// 否则，包含特定操作符的头文件
#else
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/channel_shuffle_native.h>
#endif

// 命名空间声明，定义在 at 命名空间下的 native 子命名空间
namespace at {
namespace native {

// 如果使用 PyTorch QNNPACK，则声明在匿名命名空间下的量化通道重排实现函数
#ifdef USE_PYTORCH_QNNPACK
namespace {
Tensor quantized_channel_shuffle_impl(
    const Tensor& self,   // 输入张量 self
    int64_t groups) {     // 分组数目 groups

  // 断言：分组数目必须大于 0
  TORCH_CHECK(
      groups > 0,
      "Number of groups to divide channels in must be positive.",
      " Value of groups:", groups);

  // 断言：输入张量必须是四维的
  TORCH_CHECK(
      self.dim() == 4,
      "channel_shuffle expects 4D input, but got input with sizes ",
      self.sizes());

  // 断言：输入张量的标量类型必须是 kQUInt8（无符号量化八位整数）
  TORCH_CHECK(
      self.scalar_type() == kQUInt8,
      "Quantized channel shuffle works only on ",
      toString(c10::kQUInt8),
      " but got ", self.scalar_type());

  // 将输入张量转换为 ChannelsLast 的内存格式
  const Tensor self_nhwc = self.contiguous(MemoryFormat::ChannelsLast);

  // 创建一个与 self_nhwc 具有相同尺寸和量化参数的空张量 qy
  Tensor qy = at::native::empty_affine_quantized(
      self_nhwc.sizes(),                      // 尺寸与 self_nhwc 相同
      kQUInt8,                                // 量化类型为 kQUInt8
      c10::nullopt /* layout */,              // 无特定布局要求
      kCPU,                                   // CPU 上运行
      c10::nullopt /* pin_memory */,          // 非固定内存
      self_nhwc.q_scale(),                    // 输入张量的量化比例
      self_nhwc.q_zero_point(),               // 输入张量的量化零点
      MemoryFormat::ChannelsLast);            // 输出张量的内存格式为 ChannelsLast

  // 如果分组数为 1，直接复制 self_nhwc 到 qy
  // 这是退化情况，仅进行复制操作
  if (groups == 1) {
    qy.copy_(self_nhwc);


这段代码还未完成，继续上述代码的注释。
    // 返回qy的连续版本，使用self建议的内存格式
    return qy.contiguous(self.suggest_memory_format());
  }

  // 获取张量self的通道数
  int64_t channels = self.size(1);
  // 检查通道数必须大于0
  TORCH_CHECK(channels > 0,
             "Number of channels must be positive, got:", channels);
  // 检查通道数必须能够被groups整除
  TORCH_CHECK((channels % groups) == 0,
             "Number of channels must be divisible gy groups. Got ",
             channels, " channels and ", groups, " groups.");

  // 初始化QNNPACK库
  initQNNPACK();

  // 定义QNNPACK操作符的指针
  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  // 创建QNNPACK的ChannelShuffle操作符
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_channel_shuffle_nc_x8(
      groups /* groups */,
      channels / groups /* group channels */,
      0 /* flags */,
      &qnnpack_operator);
  // 断言操作符创建成功
  TORCH_INTERNAL_ASSERT(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK ChannelShuffle operator");

  // 使用智能指针管理QNNPACK操作符的生命周期
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);

  // 设置QNNPACK ChannelShuffle操作符
  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_channel_shuffle_nc_x8(
      qnnpack_uniq_ptr.get(),
      self_nhwc.numel() / channels /* batch size */,
      (uint8_t*)self_nhwc.data_ptr<c10::quint8>() /* self data */,
      channels /* self stride */,
      (uint8_t*)qy.data_ptr<c10::quint8>() /* qy data */,
      channels /* qy stride */);
  // 断言设置操作成功
  TORCH_INTERNAL_ASSERT(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK ChannelShuffle operator");

  // 获取pthreadpool线程池
  pthreadpool_t threadpool = caffe2::pthreadpool_();
  // 运行QNNPACK ChannelShuffle操作符
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
  // 断言运行操作成功
  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK ChannelShuffle operator");

  // 返回qy的连续版本，使用self建议的内存格式
  return qy.contiguous(self.suggest_memory_format());
// at::native functions for the native_functions.yaml
// 定义一个名为 channel_shuffle_quantized_cpu 的函数，用于通道重排列操作
Tensor channel_shuffle_quantized_cpu(
    // 输入参数为一个张量 self 和一个整数 groups，groups 表示分组数
    const Tensor& self,
    int64_t groups) {
#ifdef USE_PYTORCH_QNNPACK
  // 如果使用了 QNNPACK，调用 quantized_channel_shuffle_impl 进行量化通道重排列操作
  return quantized_channel_shuffle_impl(self, groups);
#endif
  // 如果 QNNPACK 不可用，则回退到非量化路径，调用 at::native::channel_shuffle 进行通道重排列操作
  return at::native::channel_shuffle(self, groups);
}

// Keep the registry in the anonymous namespace.
// 将注册信息放在匿名命名空间中
namespace {
// 定义一个名为 QChannelShuffle 的类，继承自 c10::OperatorKernel
class QChannelShuffle final : public c10::OperatorKernel {
 public:
  // 重载操作符 ()，接受一个张量 qx 和一个整数 groups，调用 channel_shuffle_quantized_cpu 函数
  Tensor operator()(Tensor qx, int64_t groups) {
    return channel_shuffle_quantized_cpu(qx, groups);
  }
};

} // namespace
// 结束匿名命名空间

// 结束 native 命名空间
// 结束 at 命名空间
// 结束整个文件的命名空间 at
```