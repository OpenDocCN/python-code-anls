# `.\pytorch\aten\src\ATen\native\GridSampler.cpp`

```
// 定义编译器宏以限制仅用于方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入相关头文件和命名空间
#include <ATen/native/GridSampler.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/GridSamplerKernel.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 根据编译宏条件选择是否包含完整的 ATen 函数库头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_grid_sampler_2d_cpu_fallback_backward_native.h>
#include <ATen/ops/_grid_sampler_2d_cpu_fallback_native.h>
#include <ATen/ops/cudnn_grid_sampler.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/grid_sampler_2d.h>
#include <ATen/ops/grid_sampler_2d_backward_native.h>
#include <ATen/ops/grid_sampler_2d_native.h>
#include <ATen/ops/grid_sampler_3d.h>
#include <ATen/ops/grid_sampler_3d_backward_native.h>
#include <ATen/ops/grid_sampler_3d_native.h>
#include <ATen/ops/grid_sampler_native.h>
#include <ATen/ops/zeros_like.h>
#endif

// 命名空间 at::native 开始
namespace at::native {

// 使用 at::native::detail 中定义的 GridSamplerInterpolation 和 GridSamplerPadding
using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

// 匿名命名空间，用于封装内部实现细节，提供局部性和限定作用域
namespace {

  // 模板函数，实现 CPU 版本的 3D 网格采样
  template<typename scalar_t>
  Tensor grid_sampler_3d_cpu_impl(const Tensor& input, const Tensor& grid,
                                  GridSamplerInterpolation interpolation_mode,
                                  GridSamplerPadding padding_mode,
                                  bool align_corners) {
    // 查看注释 [ grid_sampler Native Functions ]
    // 如果该函数被调用而非 grid_sampler，添加检查
    check_grid_sampler_common(input, grid);

    // 检查 3D 网格采样的特定条件
    check_grid_sampler_3d(
      input, grid, static_cast<int64_t>(interpolation_mode));

    // 获取输入张量的尺寸
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_D = input.size(2);
    int64_t inp_H = input.size(3);
    int64_t inp_W = input.size(4);

    // 获取输出张量的尺寸
    int64_t out_D = grid.size(1);
    int64_t out_H = grid.size(2);
    int64_t out_W = grid.size(3);

    // 创建一个与输入相同类型的空输出张量
    auto output = at::empty({N, C, out_D, out_H, out_W}, input.options());

    // 如果输出张量为空，直接返回空张量
    if (output.numel() == 0) {
        return output;
    }

    // 获取输入张量的步长
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sD = input.stride(2);
    int64_t inp_sH = input.stride(3);
    int64_t inp_sW = input.stride(4);

    // 获取网格张量的步长
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sD = grid.stride(1);
    int64_t grid_sH = grid.stride(2);
    int64_t grid_sW = grid.stride(3);
    int64_t grid_sCoor = grid.stride(4);

    // 获取输出张量的步长
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sD = output.stride(2);
    int64_t out_sH = output.stride(3);
    int64_t out_sW = output.stride(4);

    // 获取输入、输出和网格数据指针
    const scalar_t *inp_ptr = input.const_data_ptr<scalar_t>();
    scalar_t *out_ptr = output.data_ptr<scalar_t>();
    const scalar_t *grid_ptr = grid.const_data_ptr<scalar_t>();
    // 循环遍历每个输出像素
    });
    // 返回输出张量
    return output;
  }

  template<typename scalar_t>
  std::tuple<Tensor, Tensor>
  grid_sampler_3d_backward_cpu_impl(const Tensor& grad_output,
                                    const Tensor& input, const Tensor& grid,
                                    GridSamplerInterpolation interpolation_mode,
                                    GridSamplerPadding padding_mode,
                                    bool align_corners, std::array<bool,2> output_mask) {
    // 见注释 [ grid_sampler Native Functions ]，这里添加检查以防止直接调用而非 grid_sampler。
    check_grid_sampler_common(input, grid);
    // 检查 3D 网格采样参数
    check_grid_sampler_3d(
      input, grid, static_cast<int64_t>(interpolation_mode));

    // 确定是否需要计算输入的梯度
    auto input_requires_grad = output_mask[0];
    // 创建梯度输入张量
    Tensor grad_input = ([&]() {
      // 如果需要计算输入的梯度，则返回与输入相同形状的零张量
      if (input_requires_grad) {
        return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      } else {
        // 否则返回空张量
        return Tensor();
      }
    })();
    // 创建梯度网格张量
    auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    // 如果输入或网格张量的元素数为零，则将梯度网格张量置零并返回
    if (grid.numel() == 0 || input.numel() == 0) {
      grad_grid.zero_();
      return std::make_tuple(grad_input, grad_grid);
    }
    // 如果插值模式为最近邻，则将梯度网格张量置零
    if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      grad_grid.zero_();
    }
    // 获取输入和输出的各维度大小
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_D = input.size(2);
    int64_t inp_H = input.size(3);
    int64_t inp_W = input.size(4);
    int64_t out_D = grid.size(1);
    int64_t out_H = grid.size(2);
    int64_t out_W = grid.size(3);
    // 获取输入张量的步幅
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sD = input.stride(2);
    int64_t inp_sH = input.stride(3);
    int64_t inp_sW = input.stride(4);
    // 获取网格张量的步幅
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sD = grid.stride(1);
    int64_t grid_sH = grid.stride(2);
    int64_t grid_sW = grid.stride(3);
    int64_t grid_sCoor = grid.stride(4);
    // 获取梯度输出张量的步幅
    int64_t gOut_sN = grad_output.stride(0);
    int64_t gOut_sC = grad_output.stride(1);
    int64_t gOut_sD = grad_output.stride(2);
    int64_t gOut_sH = grad_output.stride(3);
    int64_t gOut_sW = grad_output.stride(4);
    // 初始化输入梯度张量的步幅为零
    int64_t gInp_sN = 0;
    int64_t gInp_sC = 0;
    int64_t gInp_sD = 0;
    int64_t gInp_sH = 0;
    int64_t gInp_sW = 0;
    // 如果需要计算输入的梯度，则获取输入梯度张量的步幅
    if (input_requires_grad) {
      gInp_sN = grad_input.stride(0);
      gInp_sC = grad_input.stride(1);
      gInp_sD = grad_input.stride(2);
      gInp_sH = grad_input.stride(3);
      gInp_sW = grad_input.stride(4);
    }
    // 获取梯度网格张量的步幅
    int64_t gGrid_sN = grad_grid.stride(0);
    int64_t gGrid_sW = grad_grid.stride(3);
    // 获取输入、网格和梯度输出张量的数据指针
    const scalar_t *inp_ptr = input.const_data_ptr<scalar_t>();
    const scalar_t *grid_ptr = grid.const_data_ptr<scalar_t>();
    const scalar_t *gOut_ptr = grad_output.const_data_ptr<scalar_t>();
    scalar_t *gInp_ptr = nullptr;
    // 如果输入需要梯度计算
    if (input_requires_grad) {
      // 获取可变的梯度输入数据指针
      gInp_ptr = grad_input.mutable_data_ptr<scalar_t>();
    }
    // 获取梯度网格数据指针
    scalar_t *gGrid_ptr = grad_grid.data_ptr<scalar_t>();
    // 循环遍历每个输出像素
    // 注意：这里可能缺少循环体的具体实现，需要根据上下文推断
    });
    // 返回一个包含梯度输入和梯度网格的元组
    return std::make_tuple(grad_input, grad_grid);
  }
}  // namespace

// 定义静态函数 `_grid_sampler_2d_cpu_quantized`，接受输入、网格、插值模式、填充模式和对齐标志作为参数
static Tensor _grid_sampler_2d_cpu_quantized(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode_,
    int64_t padding_mode_,
    bool align_corners) {
  
  // 查看 NOTE [ grid_sampler Native Functions ]，确保在调用该函数之前已添加了通用的网格采样检查
  check_grid_sampler_common(input, grid);
  // 检查输入和网格是否适合 2D 网格采样
  check_grid_sampler_2d(input, grid);

  // 将插值模式转换为枚举类型 GridSamplerInterpolation
  auto interpolation_mode =
      static_cast<GridSamplerInterpolation>(interpolation_mode_);
  
  /* 使用量化值进行线性插值的支持，基于我们可以在不重新缩放的情况下对量化值执行线性插值的事实 */
  // 只支持双线性插值
  TORCH_CHECK(
      interpolation_mode == GridSamplerInterpolation::Bilinear,
      "_grid_sampler_2d_cpu_quantized(): only bilinear interpolation supported")
  
  // 将填充模式转换为枚举类型 GridSamplerPadding
  auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);

  // 获取输入张量的维度信息
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t inp_H = input.size(2);
  int64_t inp_W = input.size(3);
  
  // 获取网格张量的输出高度和宽度
  int64_t out_H = grid.size(1);
  int64_t out_W = grid.size(2);
  
  // 获取输入张量的量化零点
  uint8_t zero_point = input.q_zero_point();
  
  // 创建一个新的量化张量作为输出，使用与输入相同的量化参数
  auto output = at::_empty_affine_quantized(
      {N, C, out_H, out_W},
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      input.q_scale(),
      zero_point);
  
  // 获取输入张量在各个维度上的步长信息
  int64_t inp_sN = input.stride(0);
  int64_t inp_sC = input.stride(1);
  int64_t inp_sH = input.stride(2);
  int64_t inp_sW = input.stride(3);
  
  // 获取网格张量在各个维度上的步长信息
  int64_t grid_sN = grid.stride(0);
  int64_t grid_sH = grid.stride(1);
  int64_t grid_sW = grid.stride(2);
  int64_t grid_sCoor = grid.stride(3);
  
  // 获取输出张量在各个维度上的步长信息
  int64_t out_sN = output.stride(0);
  int64_t out_sC = output.stride(1);
  int64_t out_sH = output.stride(2);
  int64_t out_sW = output.stride(3);
  
  // 获取输入张量和输出张量的数据指针，并将网格张量的数据指针转换为浮点型
  uint8_t* inp_ptr = (uint8_t*)input.data_ptr<quint8>();
  uint8_t* out_ptr = (uint8_t*)output.data_ptr<quint8>();
  float* grid_ptr = grid.data_ptr<float>();
  
  // 使用并行处理，对每个样本进行操作
  at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
    ```cpp`
        // 遍历范围 [start, end) 内的每个索引 n
        for (const auto n : c10::irange(start, end)) {
          // 计算当前 n 对应的 grid 和 inp 的指针偏移量
          float* grid_ptr_N = grid_ptr + n * grid_sN;
          uint8_t* inp_ptr_N = inp_ptr + n * inp_sN;
          
          // 遍历输出张量的高度和宽度
          for (const auto h : c10::irange(out_H)) {
            for (const auto w : c10::irange(out_W)) {
              // 计算 grid 中的 x, y 坐标值
              float* grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
              float x = *grid_ptr_NHW;
              float y = grid_ptr_NHW[grid_sCoor];
    
              // 计算输入张量中 x, y 对应的源索引 ix, iy
              float ix = grid_sampler_compute_source_index(
                  x, inp_W, padding_mode, align_corners);
              float iy = grid_sampler_compute_source_index(
                  y, inp_H, padding_mode, align_corners);
    
              // 获取 (x, y) 周围像素的值，用于双线性插值
              int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
              int64_t iy_nw = static_cast<int64_t>(std::floor(iy));
    
              int64_t ix_ne = ix_nw + 1;
              int64_t iy_ne = iy_nw;
    
              int64_t ix_sw = ix_nw;
              int64_t iy_sw = iy_nw + 1;
    
              int64_t ix_se = ix_nw + 1;
              int64_t iy_se = iy_nw + 1;
    
              // 计算双线性插值的权重
              float nw = (ix_se - ix) * (iy_se - iy);
              float ne = (ix - ix_sw) * (iy_sw - iy);
              float sw = (ix_ne - ix) * (iy - iy_ne);
              float se = (ix - ix_nw) * (iy - iy_nw);
    
              // 计算双线性加权像素值并设置输出像素
              uint8_t* inp_ptr_NC = inp_ptr_N;
              uint8_t* out_ptr_NCHW =
                  out_ptr + n * out_sN + h * out_sH + w * out_sW;
              for (int64_t c = 0; c < C;
                   ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                float res = 0;
                // 考虑输入是否在边界内，进行双线性插值计算
                res += within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)
                    ? inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw
                    : zero_point * nw;
                res += within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)
                    ? inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne
                    : zero_point * ne;
                res += within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)
                    ? inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw
                    : zero_point * sw;
                res += within_bounds_2d(iy_se, ix_se, inp_H, inp_W)
                    ? inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se
                    : zero_point * se;
                // 将结果四舍五入并存入输出指针中
                *out_ptr_NCHW = std::nearbyint(res);
              }
            }
          }
        }
}
// 定义 CPU 环境下的二维网格采样回退函数，返回类型为 Tensor，参数包括输入 input、网格 grid、插值模式 interpolation_mode_、填充模式 padding_mode_、以及是否对齐角点 align_corners
Tensor _grid_sampler_2d_cpu_fallback(const Tensor& input, const Tensor& grid,
                                     int64_t interpolation_mode_,
                                     int64_t padding_mode_,
                                     bool align_corners) {
  // 查看注释 [ grid_sampler Native Functions ]。
  // 添加检查以防此函数被调用而非 grid_sampler。
  check_grid_sampler_common(input, grid);  // 检查公共的 grid_sampler 参数
  check_grid_sampler_2d(input, grid);      // 检查二维 grid_sampler 参数

  auto interpolation_mode = static_cast<GridSamplerInterpolation>(interpolation_mode_);  // 将插值模式转换为枚举类型
  auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);                    // 将填充模式转换为枚举类型
  using scalar_t = float;  // 定义标量类型为 float

  // 获取输入张量的尺寸信息
  int64_t N = input.size(0);    // 批次大小
  int64_t C = input.size(1);    // 通道数
  int64_t inp_H = input.size(2); // 输入高度
  int64_t inp_W = input.size(3); // 输入宽度
  int64_t out_H = grid.size(1);  // 输出高度
  int64_t out_W = grid.size(2);  // 输出宽度

  // 创建一个与输入相同大小的空输出张量
  auto output = at::empty({N, C, out_H, out_W}, input.options());

  // 如果输出张量为空，则直接返回空的输出张量
  if (output.numel() == 0) {
      return output;
  }

  // 获取输入张量的步长信息
  int64_t inp_sN = input.stride(0);
  int64_t inp_sC = input.stride(1);
  int64_t inp_sH = input.stride(2);
  int64_t inp_sW = input.stride(3);

  // 获取网格张量的步长信息
  int64_t grid_sN = grid.stride(0);
  int64_t grid_sH = grid.stride(1);
  int64_t grid_sW = grid.stride(2);
  int64_t grid_sCoor = grid.stride(3);

  // 获取输出张量的步长信息
  int64_t out_sN = output.stride(0);
  int64_t out_sC = output.stride(1);
  int64_t out_sH = output.stride(2);
  int64_t out_sW = output.stride(3);

  // 获取输入、输出、网格张量的指针信息
  const scalar_t *inp_ptr = input.const_data_ptr<scalar_t>();
  scalar_t *out_ptr = output.data_ptr<scalar_t>();
  const scalar_t *grid_ptr = grid.const_data_ptr<scalar_t>();

  // 循环遍历每个输出像素
  at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
    // 实际操作代码将在此处添加
  });

  // 返回输出张量
  return output;
}

// 定义 CPU 环境下的二维网格采样回退函数的反向传播，返回类型为包含梯度张量的元组，参数包括梯度输出 grad_output、输入 input、网格 grid、插值模式 interpolation_mode_、填充模式 padding_mode_、以及是否对齐角点 align_corners
std::tuple<Tensor, Tensor>
_grid_sampler_2d_cpu_fallback_backward(const Tensor& grad_output,
                                       const Tensor& input, const Tensor& grid,
                                       int64_t interpolation_mode_,
                                       int64_t padding_mode_,
                                       bool align_corners) {
  // 查看注释 [ grid_sampler Native Functions ]。
  // 添加检查以防此函数被调用而非 grid_sampler。
  check_grid_sampler_common(input, grid);  // 检查公共的 grid_sampler 参数
  check_grid_sampler_2d(input, grid);      // 检查二维 grid_sampler 参数

  // 将插值模式和填充模式转换为枚举类型
  const auto interpolation_mode = static_cast<GridSamplerInterpolation>(interpolation_mode_);
  const auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);
  using scalar_t = float;  // 定义标量类型为 float

  // 创建与输入张量相同形状的零张量作为梯度输入
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 创建与网格张量相同形状的空张量作为网格梯度
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // 如果网格或输入张量的元素数为零，将网格梯度置零并返回零梯度输入和网格梯度的元组
  if (grid.numel() == 0 || input.numel() == 0) {
    grad_grid.zero_();
    return std::make_tuple(grad_input, grad_grid);
  }

  // 如果插值模式为最近邻插值，则在下面的循环中不填充 grad_grid
  if (interpolation_mode == GridSamplerInterpolation::Nearest) {
    // 将梯度 grad_grid 初始化为零
    grad_grid.zero_();
  }
  // 获取输入张量的维度信息
  int64_t N = input.size(0);         // 输入张量的批大小
  int64_t C = input.size(1);         // 输入张量的通道数
  int64_t inp_H = input.size(2);     // 输入张量的高度
  int64_t inp_W = input.size(3);     // 输入张量的宽度
  // 获取输出网格 grid 的维度信息
  int64_t out_H = grid.size(1);      // 输出网格的高度
  int64_t out_W = grid.size(2);      // 输出网格的宽度
  // 获取输入张量的步长信息
  int64_t inp_sN = input.stride(0);  // 输入张量在批维度上的步长
  int64_t inp_sC = input.stride(1);  // 输入张量在通道维度上的步长
  int64_t inp_sH = input.stride(2);  // 输入张量在高度维度上的步长
  int64_t inp_sW = input.stride(3);  // 输入张量在宽度维度上的步长
  // 获取输出网格 grid 的步长信息
  int64_t grid_sN = grid.stride(0);  // 输出网格在批维度上的步长
  int64_t grid_sH = grid.stride(1);  // 输出网格在高度维度上的步长
  int64_t grid_sW = grid.stride(2);  // 输出网格在宽度维度上的步长
  int64_t grid_sCoor = grid.stride(3);  // 输出网格在坐标维度上的步长
  // 获取梯度 grad_output 的步长信息
  int64_t gOut_sN = grad_output.stride(0);  // 梯度 grad_output 在批维度上的步长
  int64_t gOut_sC = grad_output.stride(1);  // 梯度 grad_output 在通道维度上的步长
  int64_t gOut_sH = grad_output.stride(2);  // 梯度 grad_output 在高度维度上的步长
  int64_t gOut_sW = grad_output.stride(3);  // 梯度 grad_output 在宽度维度上的步长
  // 获取梯度 grad_input 的步长信息
  int64_t gInp_sN = grad_input.stride(0);   // 梯度 grad_input 在批维度上的步长
  int64_t gInp_sC = grad_input.stride(1);   // 梯度 grad_input 在通道维度上的步长
  int64_t gInp_sH = grad_input.stride(2);   // 梯度 grad_input 在高度维度上的步长
  int64_t gInp_sW = grad_input.stride(3);   // 梯度 grad_input 在宽度维度上的步长
  // 获取梯度 grad_grid 的步长信息
  int64_t gGrid_sN = grad_grid.stride(0);   // 梯度 grad_grid 在批维度上的步长
  int64_t gGrid_sW = grad_grid.stride(2);   // 梯度 grad_grid 在宽度维度上的步长
  // 获取输入张量 input 的数据指针
  const scalar_t *inp_ptr = input.const_data_ptr<scalar_t>();  // 输入张量的常量数据指针
  // 获取输出网格 grid 的数据指针
  const scalar_t *grid_ptr = grid.const_data_ptr<scalar_t>();  // 输出网格的常量数据指针
  // 获取梯度 grad_output 的数据指针
  const scalar_t *gOut_ptr = grad_output.const_data_ptr<scalar_t>();  // 梯度 grad_output 的常量数据指针
  // 获取可变梯度 grad_input 的数据指针
  scalar_t *gInp_ptr = grad_input.mutable_data_ptr<scalar_t>();  // 梯度 grad_input 的可变数据指针
  // 获取梯度 grad_grid 的数据指针
  scalar_t *gGrid_ptr = grad_grid.data_ptr<scalar_t>();  // 梯度 grad_grid 的数据指针
  // 并行循环处理每个输出像素
  at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
    }
  });
  // 返回梯度计算结果的元组，包括 grad_input 和 grad_grid
  return std::make_tuple(grad_input, grad_grid);
}

Tensor grid_sampler_2d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode,
                           bool align_corners) {
    // 查看注释 [ grid_sampler Native Functions ]。
    // 在此处添加检查，以防此函数被调用而不是 grid_sampler。
    check_grid_sampler_common(input, grid);
    check_grid_sampler_2d(input, grid);

    // 如果输入张量的数据类型是 kQUInt8，则调用对应的量化版本的函数
    if (input.scalar_type() == kQUInt8) {
        return native::_grid_sampler_2d_cpu_quantized(
            input, grid, interpolation_mode, padding_mode, align_corners);
    }

    // AVX gather 指令使用有符号 32 位偏移来收集浮点值。
    // 检查可能的溢出情况，并回退到标量实现
    if (input.scalar_type() != kDouble) {
        TORCH_CHECK(input.scalar_type() == kFloat,
                    "grid_sampler_2d_cpu not implemented for ", input.scalar_type());
        auto sizes = input.sizes();
        auto strides = input.strides();
        const auto grid_sW = grid.strides()[2];
        
        // 注意：收集偏移仅用于输入张量的 H、W 维度
        //       或者用于对网格张量进行分段访问
        auto max_gather_offset = std::max(
            (sizes[2] - 1) * strides[2] + (sizes[3] - 1) * strides[3],
            grid_sW * (vec::Vectorized<float>::size() - 1));

        // 如果最大收集偏移超过了 int32_t 的最大值，则回退到标量实现
        if (max_gather_offset > std::numeric_limits<int32_t>::max()) {
            return native::_grid_sampler_2d_cpu_fallback(
                input, grid, interpolation_mode, padding_mode, align_corners);
        }
    }

    auto in_size = input.sizes();
    auto grid_size = grid.sizes();
    // 创建一个空张量作为输出，形状为 [in_size[0], in_size[1], grid_size[1], grid_size[2]]
    auto output = at::empty(
        {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
    // 调用 CPU 上的 grid_sampler_2d_cpu_kernel 函数处理
    grid_sampler_2d_cpu_kernel(
        kCPU, output, input, grid, interpolation_mode, padding_mode, align_corners);
    return output;
}

// 定义 grid_sampler_2d_cpu_kernel 的分发器
DEFINE_DISPATCH(grid_sampler_2d_cpu_kernel);


Tensor grid_sampler_3d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode,
                           bool align_corners) {
    // 查看注释 [ grid_sampler Native Functions ]。
    // 在此处添加检查，以防此函数被调用而不是 grid_sampler。
    check_grid_sampler_common(input, grid);
    check_grid_sampler_3d(input, grid, interpolation_mode);

    // 使用 AT_DISPATCH_FLOATING_TYPES 宏根据输入张量的数据类型进行分发
    return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler3d_cpu", [&] {
        return grid_sampler_3d_cpu_impl<scalar_t>(
            input, grid, static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode), align_corners);
    });
}

// 定义返回两个张量的元组
std::tuple<Tensor, Tensor>
// 定义一个名为 grid_sampler_2d_backward_cpu 的函数，用于计算二维网格采样的反向传播
const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
                             std::array<bool,2> output_mask) {
  // 查看注释 [ grid_sampler Native Functions ]。
  // 在此添加检查以防这个函数被调用而不是 grid_sampler。
  check_grid_sampler_common(input, grid);  // 检查输入和网格的共同点
  check_grid_sampler_2d(input, grid);      // 检查二维网格采样器

  // AVX gather 指令使用带符号的32位偏移来收集浮点值。
  // 检查可能的溢出情况，并回退到标量实现
  if (input.scalar_type() != kDouble) {
    TORCH_CHECK(input.scalar_type() == kFloat,
                "grid_sampler_2d_backward_cpu not implemented for ", input.scalar_type());
    auto isizes = input.sizes();        // 获取输入张量的尺寸
    auto istrides = input.strides();    // 获取输入张量的步幅
    auto gsizes = grad_output.sizes();  // 获取梯度输出张量的尺寸
    auto gstrides = grad_output.strides();  // 获取梯度输出张量的步幅
    const auto grid_sW = grid.strides()[2];  // 获取网格张量在宽度维度上的步幅
    // 注意：收集偏移仅用于高度和宽度维度
    auto max_gather_offset = std::max(
      std::max(
        (isizes[2] - 1) * istrides[2] + (isizes[3] - 1) * istrides[3],  // 计算输入张量的最大收集偏移量
        (gsizes[2] - 1) * gstrides[2] + (gsizes[3] - 1) * gstrides[3]),  // 计算梯度输出张量的最大收集偏移量
      grid_sW * (vec::Vectorized<float>::size() - 1));  // 计算网格张量的最大收集偏移量

    // 如果最大收集偏移超过32位带符号整数的最大值，则回退到标量实现
    if (max_gather_offset > std::numeric_limits<int32_t>::max()) {
      return native::_grid_sampler_2d_cpu_fallback_backward(
        grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
    }
  }

  auto input_requires_grad = output_mask[0];  // 获取是否需要计算输入梯度的标志
  Tensor grad_input = ([&]() {
    // 根据需要是否计算输入梯度，返回相应的张量
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);  // 返回与输入张量相同形状的零张量
    } else {
      return Tensor();  // 返回空张量
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);  // 创建一个与网格张量相同形状的空张量
  grid_sampler_2d_backward_cpu_kernel(
      kCPU, grad_input, grad_grid, grad_output, input, grid,
      interpolation_mode, padding_mode, align_corners, output_mask);  // 调用二维网格采样反向传播的核心函数
  return std::make_tuple(std::move(grad_input), std::move(grad_grid));  // 返回梯度输入和梯度网格的元组
}

DEFINE_DISPATCH(grid_sampler_2d_backward_cpu_kernel);  // 定义二维网格采样反向传播的 CPU 核心函数的分派

// 定义一个名为 grid_sampler_3d_backward_cpu 的函数，用于计算三维网格采样的反向传播
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
                             std::array<bool,2> output_mask) {
  // 查看注释 [ grid_sampler Native Functions ]。
  // 在此添加检查以防这个函数被调用而不是 grid_sampler。
  check_grid_sampler_common(input, grid);  // 检查输入和网格的共同点
  check_grid_sampler_3d(input, grid, interpolation_mode);  // 检查三维网格采样器及插值模式

  // 使用 AT_DISPATCH_FLOATING_TYPES 宏分派到相应的浮点类型，调用具体的三维网格采样反向传播实现
  return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_3d_backward_cpu", [&] {
    return grid_sampler_3d_backward_cpu_impl<scalar_t>(
      grad_output, input, grid,
      static_cast<GridSamplerInterpolation>(interpolation_mode),
      static_cast<GridSamplerPadding>(padding_mode),
      align_corners, output_mask);  // 调用具体浮点类型的三维网格采样反向传播实现
  });
}
// 查看注释 [ grid_sampler Native Functions ] 的说明。
Tensor grid_sampler(
  // 输入张量，即要进行采样的数据
  const Tensor& input,
  // 网格张量，用于指定采样点的位置
  const Tensor& grid,
  // 插值模式，指定在采样过程中如何插值
  int64_t interpolation_mode,
  // 填充模式，指定在采样区域外部如何填充
  int64_t padding_mode,
  // 是否对齐角点，影响插值过程中的边界行为
  bool align_corners
) {
  // 如果满足使用 cudnn 加速的条件，则调用 cudnn_grid_sampler 函数
  if (cond_cudnn_grid_sampler(input, grid) &&
      static_cast<GridSamplerInterpolation>(interpolation_mode) ==
        GridSamplerInterpolation::Bilinear &&
      static_cast<GridSamplerPadding>(padding_mode) ==
        GridSamplerPadding::Zeros &&
      align_corners) {
    return cudnn_grid_sampler(input, grid);
  }

  // 如果输入张量是四维的，则调用 2D 网格采样函数
  if (input.dim() == 4) {
    return at::grid_sampler_2d(
      input, grid, interpolation_mode, padding_mode, align_corners);
  } else {
    // 否则，假定输入张量是三维的，调用 3D 网格采样函数
    return at::grid_sampler_3d(
      input, grid, interpolation_mode, padding_mode, align_corners);
  }
}

}  // namespace at::native
```