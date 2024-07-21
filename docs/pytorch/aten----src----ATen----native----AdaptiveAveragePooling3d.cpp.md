# `.\pytorch\aten\src\ATen\native\AdaptiveAveragePooling3d.cpp`

```py
// 定义编译宏，仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量操作的头文件
#include <ATen/core/Tensor.h>
// 包含分发调度功能的头文件
#include <ATen/Dispatch.h>
// 包含并行计算功能的头文件
#include <ATen/Parallel.h>
// 包含范围遍历的实用工具头文件
#include <c10/util/irange.h>

// 包含自适应池化的相关实现
#include <ATen/native/AdaptivePooling.h>

// 如果未定义每个运算符的头文件，则包含一般的张量操作和原生函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含特定的自适应平均池化三维操作的头文件
#else
#include <ATen/ops/_adaptive_avg_pool3d.h>
#include <ATen/ops/_adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool3d_native.h>
#include <ATen/ops/adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/adaptive_avg_pool3d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#endif

// 定义 at::native 命名空间
namespace at::native {

// 匿名命名空间，用于局部函数的实现
namespace {

// 模板函数，用于计算自适应平均池化三维输出帧
template <typename scalar_t>
static void adaptive_avg_pool3d_out_frame(
    const scalar_t* input_p,      // 输入数据指针
    scalar_t* output_p,           // 输出数据指针
    int64_t sizeD,                // 输入张量的 D 维大小
    int64_t isizeT, int64_t isizeH, int64_t isizeW,   // 输入张量的尺寸
    int64_t osizeT, int64_t osizeH, int64_t osizeW,   // 输出张量的尺寸
    int64_t istrideD,             // 输入张量的 D 维步长
    int64_t istrideT, int64_t istrideH, int64_t istrideW) {  // 输入张量的步长
  // 使用并行计算，对 D 维进行遍历
  at::parallel_for(0, sizeD, 1, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      /* 循环遍历输出 */
      for (const auto ot : c10::irange(osizeT)) {
        // 计算 T 维的起始和结束索引，以及范围大小
        int istartT = start_index(ot, osizeT, isizeT);
        int iendT = end_index(ot, osizeT, isizeT);
        int kT = iendT - istartT;

        // 循环遍历 H 维
        for (const auto oh : c10::irange(osizeH)) {
          // 计算 H 维的起始和结束索引，以及范围大小
          int istartH = start_index(oh, osizeH, isizeH);
          int iendH = end_index(oh, osizeH, isizeH);
          int kH = iendH - istartH;

          // 循环遍历 W 维
          for (const auto ow : c10::irange(osizeW)) {
            // 计算 W 维的起始和结束索引，以及范围大小
            int istartW = start_index(ow, osizeW, isizeW);
            int iendW = end_index(ow, osizeW, isizeW);
            int kW = iendW - istartW;

            /* 本地指针 */
            // 计算输入数据的局部指针和输出数据的局部指针
            const scalar_t* ip = input_p + d * istrideD + istartT * istrideT +
                istartH * istrideH + istartW * istrideW;
            scalar_t* op = output_p + d * osizeT * osizeH * osizeW +
                ot * osizeH * osizeW + oh * osizeW + ow;

            /* 计算局部平均值 */
            scalar_t sum = 0;
            for (const auto it : c10::irange(kT)) {
              for (const auto ih : c10::irange(kH)) {
                for (const auto iw : c10::irange(kW)) {
                  scalar_t val =
                      *(ip + it * istrideT + ih * istrideH + iw * istrideW);
                  sum += val;
                }
              }
            }

            /* 设置输出为局部平均值 */
            *op = sum / kT / kH / kW;
          }
        }
      }
    }
  });
}

// CPU 版本的自适应平均池化三维模板函数
void adaptive_avg_pool3d_out_cpu_template(
    Tensor& output,               // 输出张量
    Tensor const& input,         // 输入张量
    IntArrayRef output_size) {    // 输出尺寸数组
  // 检查输出尺寸是否为三维
  TORCH_CHECK(output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");

  // 对输入张量的维度进行遍历
  for (const auto i : c10::irange(1, input.ndimension())) {
    // 检查输入张量在给定维度 `i` 上的尺寸是否大于零，否则抛出异常
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool3d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  // 检查输入张量的维度是否为 4 或 5，否则抛出异常
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "adaptive_avg_pool3d(): Expected 4D or 5D tensor, but got ",
      input.sizes());
  // 检查输出张量的数据类型是否与输入张量的数据类型相同，否则抛出异常
  TORCH_CHECK(input.dtype() == output.dtype(),
      "expected dtype ", input.dtype(), " for `output` but got dtype ", output.dtype());

  /* sizes */
  // 获取输入张量在最后四个维度上的尺寸
  int64_t sizeD = input.size(-4);
  int64_t isizeT = input.size(-3);
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  /* strides */
  // 获取输入张量在最后四个维度上的步幅
  int64_t istrideD = input.stride(-4);
  int64_t istrideT = input.stride(-3);
  int64_t istrideH = input.stride(-2);
  int64_t istrideW = input.stride(-1);
  /* output sizes */
  // 获取输出张量的尺寸
  auto osizeT = output_size[0];
  auto osizeH = output_size[1];
  auto osizeW = output_size[2];

  if (input.ndimension() == 4) {
    // 如果输入张量的维度为 4，调整输出张量的尺寸为 (sizeD, osizeT, osizeH, osizeW)
    output.resize_({sizeD, osizeT, osizeH, osizeW});

    // 使用模板函数 AT_DISPATCH_FLOATING_TYPES_AND2 处理浮点类型和特定类型的输入数据
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
        input.scalar_type(), "adaptive_avg_pool3d_cpu", [&] {
          // 获取输入张量和输出张量的数据指针，调用 adaptive_avg_pool3d_out_frame 处理数据
          auto input_data = input.const_data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          adaptive_avg_pool3d_out_frame<scalar_t>(
              input_data,
              output_data,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW,
              istrideD,
              istrideT,
              istrideH,
              istrideW);
        });
  } else {
    // 如果输入张量的维度为 5，调整输出张量的尺寸为 (input.size(-5), sizeD, osizeT, osizeH, osizeW)
    output.resize_({input.size(-5), sizeD, osizeT, osizeH, osizeW});
    int64_t n = input.size(0);

    // 使用模板函数 AT_DISPATCH_FLOATING_TYPES_AND2 处理浮点类型和特定类型的输入数据
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
        input.scalar_type(), "adaptive_avg_pool3d_cpu", [&] {
          // 获取输入张量和输出张量的数据指针，使用并行处理方法 at::parallel_for 处理数据
          auto input_data = input.const_data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          at::parallel_for(0, n, 1, [&](int64_t start, int64_t end) {
            for (const auto b : c10::irange(start, end)) {
              adaptive_avg_pool3d_out_frame<scalar_t>(
                  input_data + b * input.stride(0),
                  output_data + b * sizeD * osizeT * osizeH * osizeW,
                  sizeD,
                  isizeT,
                  isizeH,
                  isizeW,
                  osizeT,
                  osizeH,
                  osizeW,
                  istrideD,
                  istrideT,
                  istrideH,
                  istrideW);
            }
          });
    });
  }
}

template <typename scalar_t>
static void adaptive_avg_pool3d_backward_out_frame(
    scalar_t* gradInput_p,
    const scalar_t* gradOutput_p,
    int64_t sizeD,
    int64_t isizeT,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeT,
    int64_t osizeH,
    int64_t osizeW) {
  // 使用并行计算，对每个维度进行遍历
  at::parallel_for(0, sizeD, 1, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      // 计算当前维度下的梯度输入和梯度输出指针位置
      scalar_t* gradInput_p_d = gradInput_p + d * isizeT * isizeW * isizeH;
      const scalar_t* gradOutput_p_d = gradOutput_p + d * osizeT * osizeW * osizeH;

      /* calculate average */
      // 对输出尺寸的每个维度进行遍历，计算起始和结束索引，以及对应的步长
      for (const auto ot : c10::irange(osizeT)) {
        int istartT = start_index(ot, osizeT, isizeT);
        int iendT = end_index(ot, osizeT, isizeT);
        int kT = iendT - istartT;

        for (const auto oh : c10::irange(osizeH)) {
          int istartH = start_index(oh, osizeH, isizeH);
          int iendH = end_index(oh, osizeH, isizeH);
          int kH = iendH - istartH;

          for (const auto ow : c10::irange(osizeW)) {
            int istartW = start_index(ow, osizeW, isizeW);
            int iendW = end_index(ow, osizeW, isizeW);
            int kW = iendW - istartW;

            // 计算梯度增量，根据输出尺寸的每个维度大小进行归一化
            scalar_t grad_delta =
                gradOutput_p_d[ot * osizeH * osizeW + oh * osizeW + ow] / kT /
                kH / kW;

            // 更新梯度输入
            for (const auto it : c10::irange(istartT, iendT)) {
              for (const auto ih : c10::irange(istartH, iendH)) {
                for (const auto iw : c10::irange(istartW, iendW)) {
                  /* update gradient */
                  gradInput_p_d[it * isizeH * isizeW + ih * isizeW + iw] +=
                      grad_delta;
                }
              }
            }
          }
        }
      }
    }
  });
}

Tensor& adaptive_avg_pool3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  /* get contiguous gradOutput */
  // 获取连续的梯度输出张量
  auto gradOutput = gradOutput_.contiguous();

  // 检查梯度输出是否为空，用于自适应平均池化的反向传播
  adaptive_pool_empty_output_check(gradOutput_, "adaptive_avg_pool3d_backward");

  /* sizes */
  // 获取张量的尺寸信息
  int64_t sizeD = input.size(-4);
  int64_t isizeT = input.size(-3);
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeT = gradOutput.size(-3);
  int64_t osizeH = gradOutput.size(-2);
  int64_t osizeW = gradOutput.size(-1);

  /* backprop */
  // 如果输入张量的维度为4
  if (input.ndimension() == 4) {
    // 如果输入张量的维度大于1，则使用并行化方式处理反向自适应三维平均池化的梯度计算
    if (input.dim() > 1) {
        int64_t n = input.size(0);

        AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
            input.scalar_type(), "adaptive_avg_pool3d_backward_cpu", [&] {
              /* 获取原始指针 */
              scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
              const scalar_t* gradOutput_data = gradOutput.const_data_ptr<scalar_t>();

              // 使用并行化处理每个批次的计算
              at::parallel_for(0, n, 1, [&](int64_t start, int64_t end) {
                // 遍历每个批次中的数据，并调用适应性三维平均池化的反向操作
                for (const auto b : c10::irange(start, end)) {
                  adaptive_avg_pool3d_backward_out_frame<scalar_t>(
                      gradInput_data + b * sizeD * isizeT * isizeH * isizeW,
                      gradOutput_data + b * sizeD * osizeT * osizeH * osizeW,
                      sizeD,
                      isizeT,
                      isizeH,
                      isizeW,
                      osizeT,
                      osizeH,
                      osizeW);
                }
              });
        });
    } else {
        // 单批次情况下，直接调用适应性三维平均池化的反向操作
        int64_t n = input.size(0);

        AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
            input.scalar_type(), "adaptive_avg_pool3d_backward_cpu", [&] {
              /* 获取原始指针 */
              scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
              const scalar_t* gradOutput_data = gradOutput.const_data_ptr<scalar_t>();

              adaptive_avg_pool3d_backward_out_frame<scalar_t>(
                  gradInput_data,
                  gradOutput_data,
                  sizeD,
                  isizeT,
                  isizeH,
                  isizeW,
                  osizeT,
                  osizeH,
                  osizeW);
        });
    }
    // 返回计算得到的梯度输入
    return gradInput;
} // namespace

} // namespace

// 对 CPU 上的输入进行三维自适应平均池化操作的实现
Tensor& adaptive_avg_pool3d_out_cpu(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  // 调用模板函数执行实际的自适应平均池化操作
  adaptive_avg_pool3d_out_cpu_template(output, input, output_size);
  // 返回输出张量
  return output;
}

// 在 CPU 上执行三维自适应平均池化操作
Tensor adaptive_avg_pool3d_cpu(Tensor const& input, IntArrayRef output_size) {
  // 创建一个空张量作为输出，并使用与输入相同的选项
  auto output = at::empty({0}, input.options());
  // 调用模板函数执行实际的自适应平均池化操作
  adaptive_avg_pool3d_out_cpu_template(output, input, output_size);
  // 返回输出张量
  return output;
}

// 对 CPU 上的输入进行符号整数组成的三维自适应平均池化操作
Tensor adaptive_avg_pool3d_symint(Tensor const& input, SymIntArrayRef output_size) {
  // 检查输出尺寸是否为三维
  TORCH_CHECK(output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");
  // 检查输出尺寸中每个元素是否大于等于零
  TORCH_CHECK(
        (output_size[0] >= 0 && output_size[1] >= 0 && output_size[2] >= 0),
        "adaptive_avg_pool3d: elements of output_size must be greater than or equal to 0 ",
        "but received {", output_size[0], ", ", output_size[1], ",", output_size[2], "}");

  // 如果输出尺寸为 (1,1,1)，并且输入张量不是 XPU 张量
  if (output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1 && !input.is_xpu()) {
    // 在这种情况下，自适应池化仅计算高度和宽度上的均值，可以更高效地完成
    // 计算输入张量在高度和宽度上的均值，保持维度
    Tensor out = input.mean({-1, -2, -3}, /* keepdim = */ true);
    // 如果输入张量的内存格式建议为 ChannelsLast3d
    if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d) {
      // 断言张量维度为 5，因为维度为 4 时不会得到 ChannelsLast
      const auto n = input.sym_size(0);
      const auto c = input.sym_size(1);
      // 将输出张量重塑为 ChannelsLast3d 的内存格式
      out.as_strided__symint({n, c, 1, 1, 1}, {c, 1, c, c, c});
    }
    // 返回处理后的输出张量
    return out;
  } else {
    // 对符号整数组成的输入执行实际的自适应平均池化操作
    return _adaptive_avg_pool3d_symint(input, output_size);
  }
}

// 对 CPU 上的输入进行三维自适应平均池化操作的反向传播
Tensor& adaptive_avg_pool3d_backward_out_cpu(const Tensor& gradOutput_,
    const Tensor& input,
    Tensor& gradInput) {
  // 将梯度输入张量的尺寸调整为与输入张量相同，并清零
  gradInput.resize_as_(input).zero_();
  // 调用模板函数执行自适应平均池化反向传播操作
  adaptive_avg_pool3d_backward_out_cpu_template(gradInput, gradOutput_, input);
  // 返回梯度输入张量
  return gradInput;
}

// 对 CPU 上的输入进行三维自适应平均池化操作的反向传播
Tensor adaptive_avg_pool3d_backward_cpu(const Tensor& gradOutput_,
    const Tensor& input) {
  // 创建一个与输入张量相同尺寸和内存格式的零张量作为梯度输入张量
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用模板函数执行自适应平均池化反向传播操作
  adaptive_avg_pool3d_backward_out_cpu_template(gradInput, gradOutput_, input);
  // 返回梯度输入张量
  return gradInput;
}

} // namespace at::native
```