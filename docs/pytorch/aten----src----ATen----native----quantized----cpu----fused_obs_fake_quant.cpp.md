# `.\pytorch\aten\src\ATen\native\quantized\cpu\fused_obs_fake_quant.cpp`

```py
// 定义宏，仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入张量操作的头文件
#include <ATen/core/Tensor.h>
// 引入范围工具
#include <c10/util/irange.h>
// 引入数学函数
#include <cmath>
// 引入元组功能
#include <tuple>
// 引入向量容器
#include <vector>

// 如果未定义每个操作符的头文件
#ifndef AT_PER_OPERATOR_HEADERS
// 引入 ATen 库的功能函数和本地函数
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// 引入特定的操作头文件
#include <ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.h>
#include <ATen/ops/_fused_moving_avg_obs_fq_helper.h>
#include <ATen/ops/_fused_moving_avg_obs_fq_helper_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask.h>
#include <ATen/ops/fused_moving_avg_obs_fake_quant_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like.h>
#endif

// 如果定义了使用 FBGEMM
#ifdef USE_FBGEMM
// 引入 FBGEMM 的量化工具
#include <fbgemm/QuantUtils.h>
#endif
// 引入 CPU 上的量化工具函数
#include <ATen/native/quantized/cpu/QuantUtils.h>

// 命名空间定义开始
namespace {
// 计算移动平均的函数
void calculate_moving_average(
    const at::Tensor& x,           // 输入张量 x
    at::Tensor& running_min,       // 运行时最小值张量
    at::Tensor& running_max,       // 运行时最大值张量
    float averaging_const,         // 平均常数
    bool per_row_fake_quant,       // 是否按行进行伪量化
    int ch_axis                    // 通道轴索引
) {
  at::Tensor x_min, x_max;
  // 如果按行伪量化
  if (per_row_fake_quant) {
    // 检查通道轴索引是否为 0
    TORCH_CHECK(
        ch_axis == 0,
        "Per-channel FakeQuant in fused_moving_avg_obs_fake_quant is only supported on axis == 0");
    // 获取每行的最小和最大值
    std::tie(x_min, x_max) = at::aminmax(x, 1);
  } else {
    // 获取整体的最小和最大值
    std::tie(x_min, x_max) = at::aminmax(x);
  }
  // 获取当前最小值和最大值的指针
  const float* min_curr_val = x_min.const_data_ptr<float>();
  const float* max_curr_val = x_max.const_data_ptr<float>();
  // 移动平均最小值和最大值观察器
  float* running_min_val = running_min.data_ptr<float>();
  float* running_max_val = running_max.data_ptr<float>();
  // 循环处理每个元素
  for (const auto i : c10::irange(x_min.numel())) {
    // 更新运行时最小值和最大值
    running_min_val[i] = std::isinf(running_min_val[i]) ? min_curr_val[i]
                                                        : running_min_val[i] +
                                                          averaging_const * (min_curr_val[i] - running_min_val[i]);
    running_max_val[i] = std::isinf(running_max_val[i]) ? max_curr_val[i]
                                                        : running_max_val[i] +
                                                          averaging_const * (max_curr_val[i] - running_max_val[i]);
  }
  // 函数返回
  return;
}

// 选择伪量化参数的函数
std::tuple<at::Tensor, at::Tensor> choose_qparams_fake_quant(
    const at::Tensor& x,                 // 输入张量 x
    const at::Tensor& inp_running_min,   // 输入运行时最小值
    const at::Tensor& inp_running_max,   // 输入运行时最大值
    at::Tensor& scale,                   // 伸缩因子张量
    at::Tensor& zero_point,              // 零点张量
    bool per_row_fake_quant,             // 是否按行进行伪量化
    bool symmetric_quant,                // 是否对称量化
    int qmin,                            // 量化的最小值
    int qmax,                            // 量化的最大值
    int ch_axis                          // 通道轴索引
) {
  std::tuple<at::Tensor, at::Tensor> fake_quant_out;
  at::Tensor x_min, x_max;
  // 如果按行伪量化
  if (per_row_fake_quant) {
    // 获取输入运行时最小值和最大值的数据指针
    float* x_min_data = inp_running_min.data_ptr<float>();
    float* x_max_data = inp_running_max.data_ptr<float>();
    // 循环处理每个元素
    for (const auto i : c10::irange(inp_running_min.numel())) {
#ifdef USE_FBGEMM
      // 如果定义了 USE_FBGEMM，使用 FBGEMM 库计算张量的量化参数
      fbgemm::TensorQuantizationParams x_qparams{};
      // 选择量化参数，根据最小值和最大值、量化范围、是否对称量化和是否强制为二的幂
      x_qparams = fbgemm::ChooseQuantizationParams(
          x_min_data[i],
          x_max_data[i],
          qmin,
          qmax,
          symmetric_quant, // 保留稀疏性
          false // 强制为二的幂
      );
      // 将计算得到的量化参数的缩放因子和零点保存到数组中
      scale[i] = x_qparams.scale;
      zero_point[i] = x_qparams.zero_point;
#else
      // 如果未定义 USE_FBGEMM，使用 quant_utils 库计算张量的量化参数
      quant_utils::TensorQuantizationParams x_qparams{};
      // 选择量化参数，根据最小值和最大值、量化范围、是否对称量化和是否强制为二的幂
      x_qparams = quant_utils::ChooseQuantizationParams(
          x_min_data[i],
          x_max_data[i],
          qmin,
          qmax,
          symmetric_quant, // 保留稀疏性
          false // 强制为二的幂
      );
      // 将计算得到的量化参数的缩放因子和零点保存到数组中
      scale[i] = x_qparams.scale;
      zero_point[i] = x_qparams.zero_point;
#endif
    }
    // 使用 per-channel 仿真量化的方法对张量进行量化
    fake_quant_out = at::fake_quantize_per_channel_affine_cachemask(
        x, scale, zero_point, ch_axis, qmin, qmax);
  } else {
#ifdef USE_FBGEMM
    // 如果定义了 USE_FBGEMM，使用 FBGEMM 库计算张量的量化参数
    fbgemm::TensorQuantizationParams x_qparams{};
    // 计算量化参数，使用运行时的最小和最大值、量化范围、是否对称量化和是否强制为二的幂
    x_qparams = fbgemm::ChooseQuantizationParams(
        inp_running_min.item().toFloat(),
        inp_running_max.item().toFloat(),
        qmin,
        qmax,
        symmetric_quant, // 保留稀疏性
        false // 强制为二的幂
    );

    // 将计算得到的量化参数的缩放因子和零点保存到数组中
    scale[0] = x_qparams.scale;
    zero_point[0] = x_qparams.zero_point;
#else
    // 如果未定义 USE_FBGEMM，使用 quant_utils 库计算张量的量化参数
    quant_utils::TensorQuantizationParams x_qparams{};
    // 计算量化参数，使用运行时的最小和最大值、量化范围、是否对称量化和是否强制为二的幂
    x_qparams = quant_utils::ChooseQuantizationParams(
        inp_running_min.item().toFloat(),
        inp_running_max.item().toFloat(),
        qmin,
        qmax,
        symmetric_quant, // 保留稀疏性
        false // 强制为二的幂
    );
    // 将计算得到的量化参数的缩放因子和零点保存到数组中
    scale[0] = x_qparams.scale;
    zero_point[0] = x_qparams.zero_point;
#endif
    // 创建一个张量，所有元素为 1，类型为 long
    auto fake_quant_enabled = at::ones(1, x.options().dtype(at::kLong));
    // 使用 tensor-wise 仿真量化方法对张量进行量化
    fake_quant_out =
        at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
            x, scale, zero_point, fake_quant_enabled, qmin, qmax);
  }
  // 返回量化后的张量
  return fake_quant_out;
}
} // namespace

namespace at {
namespace native {

// 函数 fused_moving_avg_obs_fake_quant_cpu，用于融合移动平均、观察器和仿真量化的 CPU 实现
std::tuple<at::Tensor, at::Tensor> fused_moving_avg_obs_fake_quant_cpu(
    const at::Tensor& self, // 输入张量
    const at::Tensor& observer_on, // 观察器状态张量
    const at::Tensor& fake_quant_on, // 仿真量化状态张量
    at::Tensor& running_min, // 运行时最小值张量
    at::Tensor& running_max, // 运行时最大值张量
    at::Tensor& scale, // 量化缩放因子张量
    at::Tensor& zero_point, // 量化零点张量
    const double averaging_const, // 平均常数
    const int64_t quant_min, // 量化最小值
    const int64_t quant_max, // 量化最大值
    const int64_t ch_axis, // 通道轴索引
    bool per_row_fake_quant, // 是否使用逐行仿真量化
    bool symmetric_quant) { // 是否对称量化
  // 检查通道轴索引是否有效
  TORCH_CHECK(ch_axis < self.dim(), "Error in fused_moving_avg_obs_fake_quant_cpu: ch_axis must be < self.dim()");
  // 获取观察器状态的整数值
  auto observe = observer_on.item().toInt();
  // 计算需要进行量化的维度大小，
  // 对于 per-channel 量化，默认使用轴 0，因为目前仅用于权重量化。
  if (per_row_fake_quant) {
    // 如果进行逐行仿真量化，使用原始张量作为仿真输出
    at::Tensor y = self;
    // 如果张量不是二维的，则重新排列以适应二维
    if (self.dim() != 2) {
      auto res = DimVector(self.sizes());  // 创建一个和当前张量尺寸相同的向量 res
      std::iota(res.begin(), res.end(), 0);  // 将 res 初始化为 [0, 1, 2, ..., n-1]
      res[ch_axis] = 0;  // 将指定的通道轴调整到第一个维度
      res[0] = ch_axis;  // 将原本的第一个维度移到指定的通道轴位置

      y = self.permute(res);  // 根据重新排列的维度对张量进行重排列操作
      y = y.flatten(1);  // 将张量展平为二维，第二维的大小根据参数指定
    }
    int64_t size = self.size(ch_axis);  // 获取指定通道轴的大小
    if (running_min.numel() == 0) {
      float inf = std::numeric_limits<float>::infinity();  // 设置一个浮点数的无穷大
      running_min.resize_(size).fill_(inf);  // 将 running_min 初始化为指定大小并填充为无穷大
      running_max.resize_(size).fill_(-inf);  // 将 running_max 初始化为指定大小并填充为负无穷大
      scale.resize_(size);  // 初始化 scale 为指定大小的向量
      zero_point.resize_(size);  // 初始化 zero_point 为指定大小的向量
    }
    if (observe) {
      // 如果需要观察，则计算移动平均值
      calculate_moving_average(
          y,
          running_min,
          running_max,
          averaging_const,
          per_row_fake_quant,
          ch_axis);
    }
  } else {
    if (observe) {
      // 如果需要观察且张量是二维的，则计算移动平均值
      calculate_moving_average(
          self,
          running_min,
          running_max,
          averaging_const,
          per_row_fake_quant,
          ch_axis);
    }
  }
  // 获取是否进行伪量化的标志位
  auto fake_quant = fake_quant_on.item().toInt();
  if (fake_quant) {
    // 如果需要进行伪量化，则调用函数选择量化参数和伪量化
    return choose_qparams_fake_quant(
        self,
        running_min,
        running_max,
        scale,
        zero_point,
        per_row_fake_quant,
        symmetric_quant,
        quant_min,
        quant_max,
        ch_axis);
  }
  // 创建一个与 self 张量相同尺寸和格式的掩码
  auto mask = at::ones_like(self, at::kBool, MemoryFormat::Preserve);
  // 返回一个包含 self 张量克隆和掩码的元组
  return std::make_tuple(self.clone(), mask);
}
// 结束函数定义的位置

at::Tensor fused_moving_avg_obs_fake_quant(
    const at::Tensor& self,  // 输入张量 self
    const at::Tensor& observer_on,  // 观察器开关
    const at::Tensor& fake_quant_on,  // 伪量化器开关
    at::Tensor& running_min,  // 运行时最小值张量
    at::Tensor& running_max,  // 运行时最大值张量
    at::Tensor& scale,  // 比例张量
    at::Tensor& zero_point,  // 零点张量
    const double averaging_const,  // 平均常数
    const int64_t quant_min,  // 量化最小值
    const int64_t quant_max,  // 量化最大值
    const int64_t ch_axis,  // 通道轴
    bool per_row_fake_quant,  // 是否逐行伪量化
    bool symmetric_quant) {  // 是否对称量化

  // 如果输入张量 self 的符号元素数为 0，则返回其克隆
  if (self.sym_numel() == 0) {
    return self.clone();
  }

  // 调用 _fused_moving_avg_obs_fq_helper 函数进行融合移动平均观察伪量化的辅助操作
  const auto res = at::_fused_moving_avg_obs_fq_helper(
      self,
      observer_on,
      fake_quant_on,
      running_min,
      running_max,
      scale,
      zero_point,
      averaging_const,
      quant_min,
      quant_max,
      ch_axis,
      per_row_fake_quant,
      symmetric_quant);

  // 返回结果元组的第一个元素作为函数的返回值
  return std::get<0>(res);
}
} // namespace native
} // namespace at
```