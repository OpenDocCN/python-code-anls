# `.\pytorch\aten\src\ATen\native\PadNd.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/PadNd.h>
#include <ATen/core/Tensor.h>

#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_pad_circular.h>
#include <ATen/ops/_pad_circular_native.h>
#include <ATen/ops/_pad_enum_native.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/constant_pad_nd_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/pad_native.h>
#include <ATen/ops/reflection_pad1d.h>
#include <ATen/ops/reflection_pad2d.h>
#include <ATen/ops/reflection_pad3d.h>
#include <ATen/ops/replication_pad1d.h>
#include <ATen/ops/replication_pad2d.h>
#include <ATen/ops/replication_pad3d.h>
#endif

namespace at::native {

// 实现常数填充多维张量的函数
Tensor constant_pad_nd(const Tensor& self, IntArrayRef pad, const Scalar& value) {
    // 检查填充尺寸的长度是否为偶数
    TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ",
             pad.size());

    // 获取输入张量的尺寸信息和维度数量
    auto input_sizes = self.sizes();
    auto l_inp = self.dim();

    // 计算填充数量和输入维度的差值
    auto l_pad = pad.size() / 2;
    auto l_diff = l_inp - l_pad;

    // 检查填充数量是否符合规范
    TORCH_CHECK(l_inp >= (int64_t)l_pad, "Length of pad should be no more than twice the number of "
             "dimensions of the input. Pad length is ", pad.size(), "while the input has ",
             l_inp, "dimensions.");

    // 新形状的尺寸容器
    std::vector<int64_t> new_shape;

    // 是否所有填充都是非正数的标志
    bool all_pads_non_positive = true;

    // 复制输入张量的副本
    auto c_input = self;

    // 遍历维度差值到输入维度之间的维度
    for (const auto i : c10::irange(l_diff, l_inp)) {
        auto pad_idx = 2 * (l_inp - i - 1);
        // 如果负向填充小于0，则对当前维度进行缩小
        if (pad[pad_idx] < 0) {
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.size(i) + pad[pad_idx]);
        } else if (pad[pad_idx] != 0) {
            all_pads_non_positive = false;
        }
        // 如果正向填充小于0，则对当前维度进行缩小
        if (pad[pad_idx + 1] < 0) {
            c_input = c_input.narrow(i, 0, c_input.size(i) + pad[pad_idx + 1]);
        } else if (pad[pad_idx + 1] != 0) {
            all_pads_non_positive = false;
        }
    }

    // 如果所有填充都是非正数，则优化返回副本
    if (all_pads_non_positive) {
        return c_input.clone();
    }

    // 计算新形状的维度
    for (size_t i = 0; i < (size_t)l_diff; i ++) {
        new_shape.emplace_back(input_sizes[i]);
    }

    // 对每个填充维度进行处理
    for (const auto i : c10::irange((size_t)l_pad)) {
        auto pad_idx = pad.size() - ((i + 1) * 2);
        // 计算新维度，包括填充的正负方向
        auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
        // 检查新维度是否大于0
        TORCH_CHECK(new_dim > 0, "The input size ", input_sizes[l_diff + i], ", plus negative padding ",
                 pad[pad_idx], " and ", pad[pad_idx + 1], " resulted in a negative output size, "
                 "which is invalid. Check dimension ", l_diff + i, " of your input.");
        new_shape.emplace_back(new_dim);
    }

    // 创建输出张量
    at::Tensor output;
    // 建议的内存格式
    const auto memory_format = self.suggest_memory_format();
    // 检查当前张量是否是量化的
    if (self.is_quantized()) {
        // 获取量化方案
        const auto qscheme = self.qscheme();
        // 检查量化方案是否为每张量仿射或每张量对称
        TORCH_CHECK(qscheme == kPerTensorAffine || qscheme == kPerTensorSymmetric,
                    "Only per-tensor padding is supported.");
        // 使用仿射量化参数创建一个新的仿射量化张量
        output = at::_empty_affine_quantized(
            new_shape, self.options().memory_format(memory_format),
            self.q_scale(), self.q_zero_point(), c10::nullopt);
    } else {
        // 创建一个未量化的新张量
        output = at::empty(new_shape, self.options().memory_format(memory_format));
    }
    // 用指定的值填充输出张量
    output.fill_(value);

    // 将 c_output 初始化为 output
    auto c_output = output;
    // 遍历范围在 [l_diff, l_inp) 之间的索引 i
    for (const auto i : c10::irange(l_diff, l_inp)) {
        // 计算当前填充索引
        auto pad_idx = 2 * (l_inp - i - 1);
        // 如果前向填充大于 0，则对 c_output 在维度 i 上进行裁剪
        if (pad[pad_idx] > 0) {
            c_output = c_output.narrow(i, pad[pad_idx], c_output.size(i) - pad[pad_idx]);
        }
        // 如果后向填充大于 0，则对 c_output 在维度 i 上进行裁剪
        if (pad[pad_idx + 1] > 0) {
            c_output = c_output.narrow(i, 0, c_output.size(i) - pad[pad_idx + 1]);
        }
    }
    // 将 c_input 的值拷贝到 c_output 中
    c_output.copy_(c_input);
    // 返回填充后的输出张量
    return output;
}

Tensor _pad_circular_symint(const Tensor &self, c10::SymIntArrayRef padding) {
  const auto in_shape = self.sym_sizes();  // 获取输入张量的符号尺寸信息
  const auto self_ndim = static_cast<int64_t>(in_shape.size());  // 获取输入张量的维度数

  // 计算被填充的维度数目
  const auto ndim_padded = padding.size() / 2;
  // 计算前面未被填充的维度数目（对于无批量维度的情况为1，否则为2）
  const auto ndim_nonpadded = self_ndim - ndim_padded;

  TORCH_CHECK(ndim_nonpadded == 1 || ndim_nonpadded == 2,
              "Invalid padding size, expected 1 or 2 non-padded dimensions, ",
              "which would be equivalent to padding of length ",
              (self_ndim - 1) * 2,
              " or ",
              (self_ndim - 2) * 2,
              " respectively but got ",
              padding.size());  // 检查填充大小是否有效

  c10::SymDimVector out_shape(in_shape.size());
  for (const auto i: c10::irange(ndim_nonpadded)) {
    out_shape[i] = in_shape[i];
  }

  // 获取填充后张量的形状
  for (const auto i : c10::irange(ndim_padded)) {
    const auto& pad_l = padding[2 * (ndim_padded - i - 1) + 0];
    const auto& pad_r = padding[2 * (ndim_padded - i - 1) + 1];
    const auto& size = in_shape[ndim_nonpadded + i];
    out_shape[ndim_nonpadded + i] = size + pad_l + pad_r;

    TORCH_CHECK(
        pad_l <= size && pad_r <= size,
        "Padding value causes wrapping around more than once.");  // 检查填充值是否会导致超过一圈的包围
    TORCH_CHECK(
        out_shape[ndim_nonpadded + i] >= 0,
        "Negative padding value is resulting in an empty dimension");  // 检查负填充值是否导致空维度
  }

  auto out = self.new_empty_symint(out_shape, self.options());  // 创建新的填充后的符号整数张量

  // 将原始数组放入填充后的数组中
  Tensor out_slice = out;
  Tensor in_slice = self;
  const SymInt zero = 0;
  for (const auto i : c10::irange(ndim_padded)) {
    const auto dim = ndim_padded - i + ndim_nonpadded - 1;
    const auto& pad_l = padding[2*i + 0];
    const auto& pad_r = padding[2*i + 1];
    out_slice = out_slice.slice_symint(dim, std::max(pad_l, zero), out_shape[dim] - std::max(pad_r, zero));
    in_slice = in_slice.slice_symint(dim, std::max(-pad_l, zero), in_shape[dim] - std::max(-pad_r, zero));
  }
  out_slice.copy_(in_slice);

  // 以下步骤首先填充张量的起始（左侧），然后填充张量的末端（右侧）。
  // 注意：当 ndim_padded > 1 时，角落处将被写入多次。
  //
  // 仅在填充值大于0时需要额外复制。
  for (const auto i : c10::irange(ndim_padded)) {
    const auto dim = ndim_padded - i + ndim_nonpadded - 1;
    const auto& pad_l = padding[2*i + 0];
    const auto& pad_r = padding[2*i + 1];

    if (pad_l > 0) {
      out_slice = out.slice_symint(dim, 0, pad_l);
      in_slice = out.slice_symint(dim,
                           out_shape[dim] - pad_l - std::max(pad_r, zero),
                           out_shape[dim] - std::max(pad_r, zero));
      out_slice.copy_(in_slice);
    }
    # 如果 pad_r 大于 0，则执行以下操作
    if (pad_r > 0) {
      # 对输出张量进行切片，保留最后 pad_r 个元素
      out_slice = out.slice_symint(dim, out_shape[dim] - pad_r, out_shape[dim]);
      # 对输出张量进行切片，保留从 pad_l 和零中的最大值开始，到最大值加上 pad_r 的元素
      in_slice = out.slice_symint(dim, std::max(pad_l, zero), std::max(pad_l, zero) + pad_r);
      # 将 in_slice 的值复制到 out_slice 中
      out_slice.copy_(in_slice);
    }
  }

  # 返回修改后的输出张量 out
  return out;
}

Tensor _pad_enum_symint(const Tensor &self, c10::SymIntArrayRef pad, int64_t mode_int, std::optional<double> value) {
  const auto input_dim = self.dim();
  // 检查填充长度是否为偶数
  TORCH_CHECK(pad.size() % 2 == 0, "Padding length must be divisible by 2");
  // 检查填充长度是否不超过输入维度的两倍
  TORCH_CHECK(static_cast<int64_t>(pad.size()) <= input_dim * 2,
              "Padding length should be less than or equal to two times the input dimension but got padding length ", pad.size(), " and input of dimension ", input_dim);
  // 将整数模式转换为枚举类型
  auto mode = static_cast<at::padding_mode>(mode_int);

  // 如果填充模式是常数填充
  if (mode == at::padding_mode::constant) {
    // 调用常数填充函数，并返回结果
    return at::constant_pad_nd_symint(self, pad, value.value_or(0.0));
  }
  // 如果值参数存在且不为零，则抛出错误
  TORCH_CHECK(!value.has_value() || *value == 0,
              "Padding mode \"", padding_mode_string(mode),
              "\" doesn't take in value argument");

  // 根据填充长度和输入维度选择相应的填充方式
  if (pad.size() == 2 && (input_dim == 2 || input_dim == 3)) {
    switch (mode) {
      // 对于反射填充模式，调用一维/二维反射填充函数
      case at::padding_mode::reflect: return at::reflection_pad1d_symint(self, pad);
      // 对于复制填充模式，调用一维/二维复制填充函数
      case at::padding_mode::replicate: return at::replication_pad1d_symint(self, pad);
      // 对于循环填充模式，调用环绕填充函数
      case at::padding_mode::circular: return at::_pad_circular_symint(self, pad);
      default: {}
    }
  } else if(pad.size() == 4 && (input_dim == 3 || input_dim == 4)) {
    switch (mode) {
      // 对于反射填充模式，调用二维/三维反射填充函数
      case at::padding_mode::reflect: return at::reflection_pad2d_symint(self, pad);
      // 对于复制填充模式，调用二维/三维复制填充函数
      case at::padding_mode::replicate: return at::replication_pad2d_symint(self, pad);
      // 对于循环填充模式，调用环绕填充函数
      case at::padding_mode::circular: return at::_pad_circular_symint(self, pad);
      default: {}
    }
  } else if (pad.size() == 6 && (input_dim == 4 || input_dim == 5)) {
    switch (mode) {
      // 对于反射填充模式，调用三维/四维反射填充函数
      case at::padding_mode::reflect: return at::reflection_pad3d_symint(self, pad);
      // 对于复制填充模式，调用三维/四维复制填充函数
      case at::padding_mode::replicate: return at::replication_pad3d_symint(self, pad);
      // 对于循环填充模式，调用环绕填充函数
      case at::padding_mode::circular: return at::_pad_circular_symint(self, pad);
      default: {}
    }
  }
  // 若填充长度和输入维度不满足以上条件，则抛出未实现错误
  C10_THROW_ERROR(NotImplementedError,
      "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now");
}

Tensor pad_symint(const Tensor &self, c10::SymIntArrayRef pad, c10::string_view mode, std::optional<double> value) {
  const auto mode_enum = [&] {
    // 根据给定的填充模式字符串选择相应的枚举值
    if (mode == "reflect") {
      return at::padding_mode::reflect;
    } else if (mode == "constant") {
      return at::padding_mode::constant;
    } else if (mode == "replicate") {
      return at::padding_mode::replicate;
    } else if (mode == "circular") {
      return at::padding_mode::circular;
    }
    // 若填充模式未识别，则抛出未实现错误
    C10_THROW_ERROR(NotImplementedError,
                    c10::str("Unrecognised padding mode ", mode));
  }();
  // 调用内部函数来进行填充操作，返回结果张量
  return at::native::_pad_enum_symint(self, pad, static_cast<int64_t>(mode_enum), value);
}

}  // namespace at::native
```