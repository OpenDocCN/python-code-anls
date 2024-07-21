# `.\pytorch\aten\src\ATen\native\quantized\cpu\Pooling.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，用于限制仅包含方法操作符

#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <ATen/native/Pool.h>
#include <ATen/native/MaxPooling.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
// 引入各种头文件，用于定义 Tensor、上下文、分发、并行处理、库、量化操作等

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/quantized_max_pool1d.h>
#include <ATen/ops/quantized_max_pool1d_native.h>
#include <ATen/ops/quantized_max_pool2d.h>
#include <ATen/ops/quantized_max_pool2d_native.h>
#include <ATen/ops/quantized_max_pool3d_native.h>
#endif
// 根据宏定义选择性地引入操作头文件，用于不同的操作类型

#include <algorithm>
#include <vector>
// 引入算法和向量相关的标准库

namespace at {
namespace native {

DEFINE_DISPATCH(qmaxpool_2d_nhwc_stub);
DEFINE_DISPATCH(qmaxpool_3d_nthwc_stub);
// 定义分发函数，用于量化最大池化操作的 NHWC 和 NTHWC 版本

namespace {

/* Computes the spatial 2D max pooling with dilation.

Argument description in the argument list.
*/
template <typename T>
void spatial_dilated_max_pooling(
    const T* iData,
    int64_t iC, // input/output channels
    int64_t iH,
    int64_t iW, // input sizes
    int64_t oH,
    int64_t oW, // output sizes
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sH,
    int64_t sW, // strides
    int64_t pH,
    int64_t pW, // padding
    int64_t dH,
    int64_t dW, // dilation
    T* oData) { // output arrays (data and max-index)
  // 并行处理每个输入通道的数据
  at::parallel_for(0, iC, 0, [&](int64_t start, int64_t end) {
    for (const auto p : c10::irange(start, end)) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t row, col;
      const T* i_p = iData + p * iW * iH;
      for (row = 0; row < oH; ++row) {
        for (col = 0; col < oW; ++col) {
          // 计算池化窗口的起始和结束位置
          int64_t h_start = row * sH - pH;
          int64_t w_start = col * sW - pW;
          int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
          int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);
          while (h_start < 0)
            h_start += dH;
          while (w_start < 0)
            w_start += dW;

          // 定义局部指针和最大值
          T* o_p = oData + p * oW * oH + row * oW + col;
          auto max_val = std::numeric_limits<typename T::underlying>::lowest();
          int64_t tcntr = 0; // center point
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t x, y;
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              tcntr = y * iW + x;
              auto val = (i_p + tcntr)->val_;
              if (val > max_val) {
                max_val = val;
              }
            }
          }
          *o_p = T(max_val); // 输出最大值
        }
      }
    }
  });
}
// 定义了一个模板函数，用于三维空间中的膨胀最大池化操作
template <typename T>
void spatial_dilated_max_pooling3d(
    const T* qxd, // 输入数据指针，类型为T
    int64_t nbatch, // 批次大小
    int64_t iC, // 输入/输出通道数
    int64_t iT, // 输入时间维度大小
    int64_t iH, // 输入高度维度大小
    int64_t iW, // 输入宽度维度大小
    int64_t oT, // 输出时间维度大小
    int64_t oH, // 输出高度维度大小
    int64_t oW, // 输出宽度维度大小
    int64_t kT, // 卷积核时间维度大小
    int64_t kH, // 卷积核高度维度大小
    int64_t kW, // 卷积核宽度维度大小
    int64_t sT, // 时间维度步长
    int64_t sH, // 高度维度步长
    int64_t sW, // 宽度维度步长
    int64_t pT, // 时间维度填充大小
    int64_t pH, // 高度维度填充大小
    int64_t pW, // 宽度维度填充大小
    int64_t dT, // 时间维度膨胀大小
    int64_t dH, // 高度维度膨胀大小
    int64_t dW, // 宽度维度膨胀大小
    T* qyd) { // 输出数据指针，类型为T
  // TODO: 进一步优化性能，参考 @mingfeima 建议，对 NCTH 进行并行处理，并缓存 W 的输出索引。
  // 处理每个批次
  int64_t oC = iC;
  int64_t parallel_dim = nbatch * iC;
  // 使用 ATen 提供的并行工具对指定范围内的元素进行并行处理
  at::parallel_for(0, parallel_dim, 0, [&](int64_t start, int64_t end) {
    for (const auto p : c10::irange(start, end)) {
      // 计算批次和通道索引
      int64_t batch_idx = p / iC;
      int64_t channel_idx = p - batch_idx * iC;

      // 输入和输出数据指针
      auto* iData = qxd + batch_idx * iC * iT * iH * iW;
      auto* oData = qyd + batch_idx * oC * oT * oH * oW;

      // 处理每个通道
      int64_t time, row, col;
      const T* i_p = iData + channel_idx * iT * iW * iH;
      for (time = 0; time < oT; ++time) {
        for (row = 0; row < oH; ++row) {
          for (col = 0; col < oW; ++col) {
            // 处理每个输出元素
            int64_t t_start = time * sT - pT;
            int64_t h_start = row * sH - pH;
            int64_t w_start = col * sW - pW;
            int64_t t_end = std::min(t_start + (kT - 1) * dT + 1, iT);
            int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
            int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);

            while (t_start < 0)
              t_start += dT;
            while (h_start < 0)
              h_start += dH;
            while (w_start < 0)
              w_start += dW;

            // 本地指针
            T* o_p = oData + channel_idx * oT * oH * oW  + time * oH * oW  + row * oW + col;

            // 本地最大值
            auto max_val = std::numeric_limits<typename T::underlying>::lowest();
            int64_t tcntr = 0; // 中心点
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            int64_t t, x, y;
            for (t = t_start; t < t_end; t += dT) {
              for (y = h_start; y < h_end; y += dH) {
                for (x = w_start; x < w_end; x += dW) {
                  tcntr = t * iH * iW + y * iW + x;
                  auto val = (i_p + tcntr)->val_;
                  if (val > max_val) {
                    max_val = val;
                  }
                }
              }
            }
            *o_p = T(max_val); // 输出。
          }
        }
      }
    }
  });
}

// 定义了一个模板函数，用于二维最大池化操作
template <typename Q>
Tensor q_maxpool_2d(
    Tensor qx, // 输入张量（量化）
    int64_t kH, // 卷积核高度维度大小
    int64_t kW, // 卷积核宽度维度大小
    int64_t sH, // 高度维度步长
    int64_t sW, // 宽度维度步长
    int64_t pH, // 高度维度填充大小
    int64_t pW, // 宽度维度填充大小
    int64_t dH, // 高度维度膨胀大小
    int64_t dW,
    bool ceil_mode) { // dilation


  // 检查输入维度
  TORCH_CHECK(kH > 0 && kW > 0, "kernel_size should be greater than zero.");
  TORCH_CHECK(sH > 0 && sW > 0, "strides should be greater than zero.");
  TORCH_CHECK(
      dH > 0 && dW > 0,
      "dilation should be greater than zero. "
      "Got (",
      dH,
      ", ",
      dW,
      ")");

  // 获取输入张量的维度信息
  int ndim = qx.dim();
  TORCH_CHECK(
      ndim == 3 || ndim == 4, "Expecting the input tensor of rank 3 or 4.");
  int dimc = 0;
  int dimh = 1;
  int dimw = 2;
  int nbatch = 1;
  if (ndim == 4) { // 包含批次维度
    ++dimc;
    ++dimh;
    ++dimw;
    nbatch = qx.size(0);
  }

  // 检查输入张量是否有效
  int64_t iC = qx.size(dimc);
  int64_t iH = qx.size(dimh);
  int64_t iW = qx.size(dimw);
  TORCH_CHECK(iC > 0 && iH > 0 && iW > 0, "input dimensions must be non-zero.");
  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "non-empty 3D or 4D input tensor is expected.");
  TORCH_CHECK(
      kH / 2 >= pH && kW / 2 >= pW,
      "padding should be smaller than half of kernel_size.");

  // 检查输出维度
  int64_t oC = iC;
  int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
  int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);
  TORCH_CHECK(oH > 0 && oW > 0,
              "Given input size: (",
              iC, "x", iH, "x", iW,
              "). Calculated output size: (",
              oC, "x", oH, "x", oW,
              "). Output size is too small.");

  // 创建输出大小的向量
  std::vector<int64_t> oSizes;
  if (ndim == 3) {
    oSizes = {oC, oH, oW};
  } else {
    oSizes = {nbatch, oC, oH, oW};
  }

  if (qx.is_contiguous(c10::MemoryFormat::ChannelsLast)) {
    // 处理通道最后内存布局的快速路径情况
    // 在这种情况下，我们可以保持内存中的数据布局
    // 同时使用更适合矢量化的循环嵌套
    Tensor qy;
    if constexpr(std::is_same_v<Q, uint8_t>) {
      // 如果输入是 uint8_t 类型，则使用 at::empty 创建输出张量
      qy = at::empty(
        oSizes,
        qx.options()
          .device(c10::kCPU)
          .dtype(qx.scalar_type())
          .memory_format(c10::MemoryFormat::ChannelsLast));
    } else {
      // 否则，使用 at::_empty_affine_quantized 创建量化后的输出张量
      qy = at::_empty_affine_quantized(
          oSizes,
          qx.options()
            .dtype(toQIntType(qx.scalar_type()))
            .memory_format(qx.suggest_memory_format()),
          qx.q_scale(),
          qx.q_zero_point(),
          c10::nullopt);
    }
    // 调用 qmaxpool_2d_nhwc_stub 执行最大池化操作
    qmaxpool_2d_nhwc_stub(qx.device().type(), qx, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
    return qy;
  } else {
    Tensor qy;
    # 如果模板参数 Q 不是 uint8_t 类型
    if constexpr(!std::is_same_v<Q, uint8_t>) {
      # 创建一个空的仿射量化张量 qy
      qy = at::_empty_affine_quantized(
              oSizes,
              qx.options().dtype(toQIntType(qx.scalar_type())),  # 根据 qx 的数据类型转换为对应的量化整数类型
              qx.q_scale(),  # 获取 qx 的量化比例因子
              qx.q_zero_point());  # 获取 qx 的零点偏移量
      # 强制 qx 连续化
      auto qx_contig = qx.contiguous();
      # 获取 qx 数据的指针 qxd 和 qy 数据的指针 qyd
      auto qxd = qx_contig.data_ptr<Q>();
      auto qyd = qy.data_ptr<Q>();
      # 如果是三维数据或者批次大小为1
      if (ndim == 3 || nbatch == 1) {
        # 设置输入数据和输出数据的指针
        auto* iData = qxd;
        auto* oData = qyd;
        # 执行空间扩张最大池化操作
        spatial_dilated_max_pooling<Q>(
            iData,
            iC,
            iH,
            iW,
            oH,
            oW,
            kH,
            kW,
            sH,
            sW,
            pH,
            pW,
            dH,
            dW,
            oData);
      } else {
        # 对于每个批次并行执行操作
        at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
          for (const auto p : c10::irange(start, end)) {
            # 计算当前批次的输入数据和输出数据的指针
            auto* iData = qxd + p * iC * iW * iH;
            auto* oData = qyd + p * oC * oW * oH;
            # 执行空间扩张最大池化操作
            spatial_dilated_max_pooling<Q>(
                iData,
                iC,
                iH,
                iW,
                oH,
                oW,
                kH,
                kW,
                sH,
                sW,
                pH,
                pW,
                dH,
                dW,
                oData);
          }
        });
      }
    } else {
      # 如果 qx 是 uint8 并且是连续内存格式
      // 如果 qx 是 uint8 并且是连续内存格式
      // 使用 channels_last 实现，并将 qy 转换回连续格式
      qy = at::empty(
        oSizes,
        qx.options()
          .device(c10::kCPU)
          .dtype(qx.scalar_type())
          .memory_format(c10::MemoryFormat::ChannelsLast));
      # 强制 qx 转换为 channels_last 的连续格式
      auto qx_nhwc = qx.contiguous(c10::MemoryFormat::ChannelsLast);
      # 调用 channels_last 的最大池化函数
      qmaxpool_2d_nhwc_stub(qx_nhwc.device().type(), qx_nhwc, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
      # 强制 qy 转换为连续格式
      qy = qy.contiguous();
    }
    # 返回处理后的 qy 张量
    return qy;
  // 定义模板函数q_maxpool_3d，接受一个量化输入Tensor qx和一系列池化操作的参数
  Tensor q_maxpool_3d(
      Tensor qx, // 输入Tensor（量化）
      int64_t kT, // 池化核的时间维度大小
      int64_t kH, // 池化核的高度维度大小
      int64_t kW, // 池化核的宽度维度大小
      int64_t sT, // 池化操作的时间步长
      int64_t sH, // 池化操作的高度步长
      int64_t sW, // 池化操作的宽度步长
      int64_t pT, // 池化核的时间维度填充大小
      int64_t pH, // 池化核的高度维度填充大小
      int64_t pW, // 池化核的宽度维度填充大小
      int64_t dT, // 池化核的时间维度膨胀大小
      int64_t dH, // 池化核的高度维度膨胀大小
      int64_t dW, // 池化核的宽度维度膨胀大小
      bool ceil_mode) { // 是否使用向上取整模式
    // 检查输入的池化核大小应大于零
    TORCH_CHECK(kT > 0 && kH > 0 && kW > 0, "kernel_size should be greater than zero.");
    // 检查输入的步长大小应大于零
    TORCH_CHECK(sT > 0 && sH > 0 && sW > 0, "strides should be greater than zero.");
    // 检查输入的膨胀大小应大于零，并输出相应的错误消息
    TORCH_CHECK(
        dT && dH > 0 && dW > 0,
        "dilation should be greater than zero. "
        "Got (",
        dT,
        ", ",
        dH,
        ", ",
        dW,
        ")");
    // 获取输入Tensor的维度数，并进行维度数为5的检查
    int ndim = qx.dim();
    TORCH_CHECK(ndim == 5, "Expecting the input tensor of rank 5.");

    // 定义tensor中的每个维度的索引值
    // act: n, c, t, h, w
    int dimc = 1;
    int dimt = 2;
    int dimh = 3;
    int dimw = 4;
    // 获取批处理大小
    int nbatch = qx.size(0);
    // 检查输入的维度是否为有效的非零值
    int64_t iC = qx.size(dimc);
    int64_t iT = qx.size(dimt);
    int64_t iH = qx.size(dimh);
    int64_t iW = qx.size(dimw);
    TORCH_CHECK(iC > 0 && iT > 0 && iH > 0 && iW > 0, "input dimensions must be non-zero.");
    // 检查填充大小是否小于池化核大小的一半
    TORCH_CHECK(
        kT / 2 >= pT && kH / 2 >= pH && kW / 2 >= pW,
        "padding should be smaller than half of kernel_size.");

    // 检查输出维度
    int64_t oC = iC;
    int64_t oT = pooling_output_shape(iT, kT, pT, sT, dT, ceil_mode);
    int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
    int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);
    // 检查输出维度是否有效
    TORCH_CHECK(oT > 0 && oH > 0 && oW > 0,
                "Given input size: (",
                iC, "t", iT , "x", iH, "x", iW,
                "). Calculated output size: (",
                oC, "t", oT , "x", oH, "x", oW,
                "). Output size is too small.");

    // 创建输出tensor的大小数组
    std::vector<int64_t> oSizes = {nbatch, oC, oT, oH, oW};

    // 检查是否是ChannelsLast3d内存格式，以进行快速通道末尾情况处理
    if (qx.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
      // 在ChannelsLast3d情况下的快速路径
      // 在此情况下，可以保持内存中的数据布局
      // 同时使用更适合矢量化的循环嵌套
      // 创建一个与qx形状和量化类型相匹配的空输出tensor
      Tensor qy = at::_empty_affine_quantized(
          oSizes,
          qx.options()
            .dtype(toQIntType(qx.scalar_type()))
            .memory_format(qx.suggest_memory_format()),
          qx.q_scale(),
          qx.q_zero_point(),
          c10::nullopt);
      // 调用池化的C++扩展函数qmaxpool_3d_nthwc_stub，进行具体的池化操作
      qmaxpool_3d_nthwc_stub(qx.device().type(), qx, iC, iT, iH, iW, oT, oH, oW, kT, kH, kW, sT, sH, sW, pT, pH, pW, dT, dH, dW, qy);
      // 返回量化后的输出tensor
      return qy;
    } else {
      // 创建一个与qx形状和量化类型相匹配的空输出tensor
      Tensor qy = at::_empty_affine_quantized(
        oSizes,
        qx.options().dtype(toQIntType(qx.scalar_type())),
        qx.q_scale(),
        qx.q_zero_point());
      // 确保输入tensor是连续的
      auto qx_contig = qx.contiguous();
      // 获取输入和输出tensor的数据指针
      auto qxd = qx_contig.data_ptr<Q>();
      auto qyd = qy.data_ptr<Q>();
      
      // 继续填写余下的函数体
    # 调用空间扩展的三维最大池化函数，对输入进行池化操作并计算输出
    spatial_dilated_max_pooling3d<Q>(
        qxd,    # 输入张量 qxd
        nbatch, # 批大小
        iC,     # 输入通道数
        iT,     # 输入时间维度大小
        iH,     # 输入高度维度大小
        iW,     # 输入宽度维度大小
        oT,     # 输出时间维度大小
        oH,     # 输出高度维度大小
        oW,     # 输出宽度维度大小
        kT,     # 池化核在时间维度上的大小
        kH,     # 池化核在高度维度上的大小
        kW,     # 池化核在宽度维度上的大小
        sT,     # 时间维度上的步长大小
        sH,     # 高度维度上的步长大小
        sW,     # 宽度维度上的步长大小
        pT,     # 时间维度上的填充大小
        pH,     # 高度维度上的填充大小
        pW,     # 宽度维度上的填充大小
        dT,     # 时间维度上的扩张大小
        dH,     # 高度维度上的扩张大小
        dW,     # 宽度维度上的扩张大小
        qyd     # 输出张量 qyd
    );
    
    # 返回经过空间扩展的三维最大池化处理后的输出张量 qy
    return qy;
} // 结束匿名命名空间

} // 结束命名空间

// 检查 max_pool2d 的参数有效性
Tensor quantized_max_pool2d(
    const Tensor& qx,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // 调用函数检查 max_pool2d 的参数
  check_maxpool2d_params(
      kernel_size,
      stride,
      padding,
      dilation);
  // 如果未指定 stride，则将其设置为 kernel_size
  if (stride.empty()) {
    stride = kernel_size;
  }
#ifdef USE_PYTORCH_QNNPACK
  // 如果使用 QNNPACK 引擎且输入类型为 kQUInt8 且非 ceil_mode，则调用 QNNPACK 提供的 maxpool2d 实现
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK && qx.scalar_type() == kQUInt8 && !ceil_mode) {
    return qnnpack_maxpool2d(qx, kernel_size, stride, padding, dilation, ceil_mode);
  }
#endif
  // 定义输出张量 qy
  Tensor qy;
  // 根据 qx 的类型分发至 q_maxpool_2d 进行计算
  AT_DISPATCH_QINT_TYPES_AND(ScalarType::Byte, qx.scalar_type(), "max_pool2d", [&]() {
    qy = q_maxpool_2d<scalar_t>(
        qx,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        ceil_mode);
  });
  return qy; // 返回计算结果
}

// 检查 max_pool3d 的参数有效性
Tensor quantized_max_pool3d(
    const Tensor& qx,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // 调用函数检查 max_pool3d 的参数
  check_maxpool3d_params(
      kernel_size,
      stride,
      padding,
      dilation);
  // 如果未指定 stride，则将其设置为 kernel_size
  if (stride.empty()) {
    stride = kernel_size;
  }
#ifdef USE_PYTORCH_QNNPACK
  // 如果使用 QNNPACK 引擎，则抛出错误，因为 QNNPACK 不支持 quantized_max_pool3d
  TORCH_CHECK(at::globalContext().qEngine() != at::QEngine::QNNPACK,
              "QNNPACK backend doesn't support of quantized_max_pool3d");
#endif
  // 定义输出张量 qy
  Tensor qy;
  // 根据 qx 的类型分发至 q_maxpool_3d 进行计算
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool3d", [&]() {
  // 调用 q_maxpool_3d 函数进行三维最大池化操作，并将结果保存在 qy 中
  qy = q_maxpool_3d<scalar_t>(
      qx,  // 输入张量 qx，进行池化操作
      kernel_size[0],  // 池化核大小的第一个维度
      kernel_size[1],  // 池化核大小的第二个维度
      kernel_size[2],  // 池化核大小的第三个维度
      stride[0],       // 池化操作的步长的第一个维度
      stride[1],       // 池化操作的步长的第二个维度
      stride[2],       // 池化操作的步长的第三个维度
      padding[0],      // 池化操作的填充的第一个维度
      padding[1],      // 池化操作的填充的第二个维度
      padding[2],      // 池化操作的填充的第三个维度
      dilation[0],     // 池化操作的膨胀系数的第一个维度
      dilation[1],     // 池化操作的膨胀系数的第二个维度
      dilation[2],     // 池化操作的膨胀系数的第三个维度
      ceil_mode);      // 池化操作是否使用 ceil 模式
});
// 返回 qy 作为函数的结果
return qy;
}

// Quantized max_pool1d is a special case of the max_pool2d, with one of the
// dimensions and kernels removed.
// 定义一个函数 quantized_max_pool1d，实现对一维张量的量化最大池化操作
Tensor quantized_max_pool1d(
    const Tensor& qx,  // 输入的量化张量 qx
    IntArrayRef kernel_size,  // 池化核大小
    IntArrayRef stride,  // 池化步长
    IntArrayRef padding,  // 池化填充
    IntArrayRef dilation,  // 池化膨胀
    bool ceil_mode) {  // 是否使用 ceil mode 进行池化
  check_max_pool1d(qx, kernel_size, stride, padding, dilation, ceil_mode);  // 检查池化参数的有效性
  // 根据 qx 的维度确定要挤压的维度，使得 qx 变为 (C, 1, L) 或 (N, C, 1, L)
  const int32_t kSqueezeDim = qx.dim() - 1;
  const auto qx_unsqueeze = qx.unsqueeze(kSqueezeDim);  // 在 kSqueezeDim 维度上对 qx 进行unsqueeze操作
  if (stride.empty()) {
    stride = kernel_size;  // 如果步长未指定，则默认与核大小相同
  }
  auto qy = at::quantized_max_pool2d(
    qx.unsqueeze(kSqueezeDim),  // 在 kSqueezeDim 维度上对 qx 进行unsqueeze操作
    {1, kernel_size[0]},  // 池化核大小，在第一维设为 1，第二维设为 kernel_size[0]
    {1, stride[0]},  // 池化步长，在第一维设为 1，第二维设为 stride[0]
    {0, padding[0]},  // 池化填充，在第一维设为 0，第二维设为 padding[0]
    {1, dilation[0]},  // 池化膨胀，在第一维设为 1，第二维设为 dilation[0]
    ceil_mode);  // 是否使用 ceil mode 进行池化
  qy = qy.squeeze(kSqueezeDim);  // 在 kSqueezeDim 维度上对 qy 进行squeeze操作
  return qy;  // 返回池化结果 qy
}

// Keep the registry in the anonymous namespace.
namespace {
// 模板类 QMaxPool_arr_args，处理不同维度的量化最大池化操作
template <uint32_t kSpatialDim>
class QMaxPool_arr_args final {
 public:
  // 静态方法 run，根据空间维度 kSpatialDim 执行量化最大池化操作
  static Tensor run(
      const Tensor& qx,  // 输入的量化张量 qx
      std::vector<int64_t> kernel_size,  // 池化核大小
      std::vector<int64_t> stride,  // 池化步长
      std::vector<int64_t> padding,  // 池化填充
      std::vector<int64_t> dilation,  // 池化膨胀
      bool ceil_mode) {  // 是否使用 ceil mode 进行池化
    // 如果 qx 不是量化的并且 kSpatialDim 等于 2 且 qx 的标量类型是 Byte 类型
    if (!qx.is_quantized() && kSpatialDim == 2 && qx.scalar_type() == c10::ScalarType::Byte){
      // 调用 ATen 的原生量化最大池化函数
      return at::native::quantized_max_pool2d(qx, kernel_size, stride, padding,
                                      dilation, ceil_mode);
    }
    // 如果 kSpatialDim 等于 1，调用 ATen 的一维量化最大池化函数
    if (kSpatialDim == 1) {
      return at::quantized_max_pool1d(qx, kernel_size, stride, padding,
                                      dilation, ceil_mode);
    } else if (kSpatialDim == 2) {  // 如果 kSpatialDim 等于 2，调用 ATen 的二维量化最大池化函数
      return at::quantized_max_pool2d(qx, kernel_size, stride, padding,
                                      dilation, ceil_mode);
    }
    // 如果 kSpatialDim 不是 1 或者 2，抛出错误信息
    TORCH_CHECK(false, "MaxPool", kSpatialDim, "D is not supported.");
  }
};

// 在匿名命名空间中注册量化最大池化操作的实现
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // 将 quantized::max_pool1d 的实现指定为 QMaxPool_arr_args<1>::run
  m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool1d"), TORCH_FN(QMaxPool_arr_args<1>::run));
  // 将 quantized::max_pool2d 的实现指定为 QMaxPool_arr_args<2>::run
  m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool2d"), TORCH_FN(QMaxPool_arr_args<2>::run));
}

// 在匿名命名空间中注册量化最大池化操作的实现（CPU 版本）
TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  // 将 quantized::max_pool2d 的实现指定为 QMaxPool_arr_args<2>::run
  m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool2d"), TORCH_FN(QMaxPool_arr_args<2>::run));
}

} // namespace
} // namespace native
} // namespace at
```