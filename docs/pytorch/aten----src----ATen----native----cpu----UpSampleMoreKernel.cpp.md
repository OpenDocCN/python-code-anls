# `.\pytorch\aten\src\ATen\native\cpu\UpSampleMoreKernel.cpp`

```py
// 定义宏以启用仅支持方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含标准库头文件
#include <vector>

// 包含 ATen 库的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/UpSample.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>
#include <ATen/cpu/vec/vec.h>

// ATen 的 native 命名空间
namespace at::native {

// 匿名命名空间，用于限定函数的作用域
namespace {

// 定义 scale_t 类型为可选的双精度浮点数的向量
using scale_t = std::vector<std::optional<double>>;

// nearest_channels_last_acc 函数模板，处理非降低浮点数类型的情况
template <typename acc_t, typename scalar_t,
          typename scalar_nonconst_t = std::remove_const_t<scalar_t>,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_nonconst_t> || !std::is_same<acc_t, float>::value, int> = 0>
void inline nearest_channels_last_acc(acc_t* gin, scalar_t* gout, int64_t size) {
  // 断言 acc_t 类型与 scalar_t 类型一致，对于 CPU 上的 float 或 double 类型的 Upsample 反向操作
  TORCH_CHECK((std::is_same<acc_t, scalar_nonconst_t>::value),
              "acc data type of Upsample backward should be same as scalar_t for float or double on CPU.")
  // 使用 Vectorized 类型 Vec 加速计算
  using Vec = Vectorized<acc_t>;
  // 初始化 d 为 0，遍历数据大小为 size 的数组
  int64_t d = 0;
  // 循环处理每个 Vectorized 的元素
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    // 加载并处理数据向量化后的元素
    Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d);
    // 存储处理后的数据
    gin_vec.store(gin + d);
  }
  // 处理剩余的数据元素
  for (; d < size; d++) {
    gin[d] += gout[d];
  }
}

// nearest_channels_last_acc 函数模板，处理降低浮点数类型的情况
template <typename acc_t, typename scalar_t,
          typename scalar_nonconst_t = std::remove_const_t<scalar_t>,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_nonconst_t> && std::is_same<acc_t, float>::value, int> = 0>
void inline nearest_channels_last_acc(acc_t* gin, scalar_t* gout, int64_t size) {
  // 使用 bVec 和 fVec 分别表示向量化的标量类型和浮点数类型
  using bVec = Vectorized<scalar_nonconst_t>;
  using fVec = Vectorized<float>;
  // 初始化 d 为 0，遍历数据大小为 size 的数组
  int64_t d = 0;
  // 循环处理每个 Vectorized 的元素
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    // 加载并处理数据向量化后的元素
    bVec gout_bvec = bVec::loadu(gout + d);
    auto [gout_fvec0, gout_fvec1] = convert_to_float<scalar_nonconst_t>(gout_bvec);
    fVec gin_fvec0 = fVec::loadu(gin + d) + gout_fvec0;
    fVec gin_fvec1 = fVec::loadu(gin + d + fVec::size()) + gout_fvec1;
    gin_fvec0.store(gin + d);
    gin_fvec1.store(gin + d + fVec::size());
  }
  // 处理剩余的数据元素
  for (; d < size; d++) {
    gin[d] += gout[d];
  }
}

// linear_channels_last_acc 函数模板，处理非降低浮点数类型的情况
template <typename acc_t, typename scalar_t,
          typename scalar_nonconst_t = std::remove_const_t<scalar_t>,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_nonconst_t> || !std::is_same<acc_t, float>::value, int> = 0>
void inline linear_channels_last_acc(acc_t* gin, const scalar_t* gout, acc_t w, int64_t size) {
  // 断言 acc_t 类型与 scalar_t 类型一致，对于 CPU 上的 float 或 double 类型的 Upsample 反向操作
  TORCH_CHECK((std::is_same<acc_t, scalar_nonconst_t>::value),
              "acc data type of Upsample backward should be same as scalar_t for float or double on CPU.")
  // 使用 Vectorized 类型 Vec 加速计算
  using Vec = Vectorized<acc_t>;
  // 初始化 d 为 0，遍历数据大小为 size 的数组
  int64_t d = 0;
  // 循环处理每个 Vectorized 的元素
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    // 加载并处理数据向量化后的元素
    Vec gin_vec = Vec::loadu(gin + d) + Vec(w) * Vec::loadu(gout + d);
    // 存储处理后的数据
    gin_vec.store(gin + d);
  }
  // 处理剩余的数据元素
  for (; d < size; d++) {
    gin[d] += w * gout[d];
  }
}

} // namespace
} // namespace at::native
# 定义了一个模板函数 `linear_channels_last_acc`，用于在最后一个维度上累加梯度。
# 函数接受四个模板参数：`acc_t` 为累加类型，`scalar_t` 为标量类型，`scalar_nonconst_t` 为去除常量属性的标量类型，`std::enable_if_t<>` 用于条件编译。
# 如果 `scalar_nonconst_t` 是缩减的浮点数且 `acc_t` 是 `float` 类型，则启用这个函数，否则编译器会选择其他重载。

void inline linear_channels_last_acc(acc_t* gin, const scalar_t* gout, acc_t w, int64_t size) {
  using bVec = Vectorized<scalar_nonconst_t>;  # 使用别名 `bVec` 表示 `Vectorized<scalar_nonconst_t>` 类型
  using fVec = Vectorized<float>;              # 使用别名 `fVec` 表示 `Vectorized<float>` 类型
  int64_t d = 0;                              # 初始化循环变量 `d` 为 0
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    # 对每个向量化的步长，加载 `gout` 中的数据到 `gout_bvec`
    bVec gout_bvec = bVec::loadu(gout + d);
    # 将 `gout_bvec` 转换为 `float` 向量，存储在 `gout_fvec0` 和 `gout_fvec1`
    auto [gout_fvec0, gout_fvec1] = convert_to_float<scalar_nonconst_t>(gout_bvec);
    # 加载 `gin` 中的数据到 `gin_fvec0` 和 `gin_fvec1`，并加上 `w` 乘以对应的 `gout` 向量
    fVec gin_fvec0 = fVec::loadu(gin + d) + fVec(w) * gout_fvec0;
    fVec gin_fvec1 = fVec::loadu(gin + d + fVec::size()) + fVec(w) * gout_fvec1;
    # 将计算结果存储回 `gin` 中对应的位置
    gin_fvec0.store(gin + d);
    gin_fvec1.store(gin + d + fVec::size());
  }
  # 处理余下的非向量化部分，直到 `size`
  for (; d < size; d++) {
    gin[d] += w * gout[d];
  }
}

# 定义了一个模板函数 `cpu_upsample_nearest_backward`，用于最近邻插值的反向传播。
# 函数接受三个模板参数：`scalar_t` 为标量类型，`scale_type` 为比例类型，`nearest_idx_fn` 为最近邻索引函数类型。
void cpu_upsample_nearest_backward(
    const Tensor& grad_input_,    # 接受 `grad_input_` 引用参数，表示梯度输入张量
    const Tensor& grad_output_,   # 接受 `grad_output_` 引用参数，表示梯度输出张量
    const scale_type& scales) {   # 接受 `scales` 引用参数，表示缩放比例

  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());  # 检查梯度输入和输出张量的数据类型是否一致

  auto grad_output = grad_output_.contiguous();  # 使 `grad_output` 张量连续化
  auto grad_input = grad_input_.contiguous();    # 使 `grad_input` 张量连续化

  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();  # 获取 `grad_output` 的常量数据指针
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();  # 获取 `grad_input` 的可变数据指针
  auto input_sizes = grad_input.sizes().vec();    # 获取 `grad_input` 的大小并转换为向量
  auto output_sizes = grad_output.sizes().vec();  # 获取 `grad_output` 的大小并转换为向量
  auto ndim = input_sizes.size();                 # 获取 `grad_input` 的维度数

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];  # 计算通道数，将 `nbatch` 和 `channels` 视为一个维度
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;      # 如果维度数为 5，则设置输入深度，否则为 1
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;    # 如果维度数为 5，则设置输出深度，否则为 1
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;    # 如果维度数大于等于 4，则设置输入高度，否则为 1
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;  # 如果维度数大于等于 4，则设置输出高度，否则为 1
  int64_t input_width = input_sizes[ndim - 1];    # 设置输入宽度
  int64_t output_width = output_sizes[ndim - 1];  # 设置输出宽度

  int64_t output_slice_size = output_depth * output_height * output_width;  # 计算输出切片大小
  int64_t input_slice_size = input_depth * input_height * input_width;      # 计算输入切片大小

  using opmath_t = at::opmath_type<scalar_t>;  # 使用 `at::opmath_type<scalar_t>` 别名 `opmath_t`
  auto loop1d = [&](int64_t begin, int64_t end) {  # 定义一个 lambda 函数 `loop1d`，用于处理 1 维循环
    opmath_t* acc_data_ptr = nullptr;   # 初始化累加数据指针为空
    std::unique_ptr<opmath_t[]> buffer_data;  # 定义一个独占指针 `buffer_data`，用于缓存数据
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
      buffer_data = std::make_unique<opmath_t[]>(input_slice_size);  # 如果 `scalar_t` 和 `opmath_t` 类型不同，则分配缓存空间
      acc_data_ptr = buffer_data.get();  # 设置累加数据指针为缓存数据的指针
      memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);  # 使用 `memset` 将累加数据初始化为 0
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);  # 否则，将累加数据指针设置为 `grad_input_data` 的强制转换指针
    }
  auto loop2d = [&](int64_t begin, int64_t end) {
    // 初始化累加数据指针和缓冲区数据指针
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    // 如果 scalar_t 和 opmath_t 类型不同，创建缓冲区并初始化为零
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      // 否则直接使用 grad_input_data 作为累加数据指针
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    // 循环遍历每个通道 c
    for (const auto c : c10::irange(begin, end)) {
      // 计算输入偏移量
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      
      // 遍历输出高度的每一行
      for (const auto oh : c10::irange(output_height)) {
        // 计算输入高度索引
        int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[0]);
        
        // 遍历输出宽度的每一个位置
        for (const auto ow : c10::irange(output_width)) {
          // 计算输入宽度索引
          int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[1]);
          
          // 计算输出偏移量
          int64_t output_offset = c * output_slice_size + oh * output_width + ow;
          
          // 累加梯度到对应的输入位置
          acc_data_ptr[input_offset + ih * input_width + iw] += grad_output_data[output_offset];
        }
      }

      // 如果 scalar_t 和 opmath_t 类型不同，应用梯度到 grad_input_data
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + c * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    // 初始化累加数据指针和缓冲区数据指针
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    // 如果 scalar_t 和 opmath_t 类型不同，创建缓冲区并初始化为零
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      // 否则直接使用 grad_input_data 作为累加数据指针
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }
    // 遍
  // 检查 `grad_input_` 和 `grad_output_` 的数据类型是否一致
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  // 获取输出张量的维度数
  auto ndim = grad_output_.ndimension();
  // 检查张量的维度是否在4到5之间，NHWC格式仅支持这些维度
  TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

  // 根据张量维度确定使用的存储格式，4维使用ChannelsLast，5维使用ChannelsLast3d
  auto channels_last_memory_format = ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  // 使grad_output_和grad_input_张量按照指定的存储格式连续化
  auto grad_output = grad_output_.contiguous(channels_last_memory_format);
  auto grad_input = grad_input_.contiguous(channels_last_memory_format);

  // 获取grad_output和grad_input的数据指针
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // 获取输入和输出张量的尺寸
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();

  // 分别获取批次数、通道数、深度、高度和宽度等尺寸信息
  int64_t num_batches =  input_sizes[0];
  int64_t channels =  input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];
  int64_t input_slice_size = input_depth * input_height * input_width * channels;

  // 定义操作类型opmath_t
  using opmath_t = at::opmath_type<scalar_t>;

  // 定义2D循环函数，用于计算梯度
  auto loop2d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    // 如果scalar_t和opmath_t不同，创建临时缓冲区buffer_data，初始化acc_data_ptr并清零
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      // 如果scalar_t和opmath_t相同，则直接使用grad_input_data
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }
    for (const auto n : c10::irange(begin, end)) {
      // 计算当前输入数据在缓冲区的起始偏移量
      int64_t input_offset = buffer_data.get() == nullptr ? n * input_slice_size : 0;
      for (const auto oh : c10::irange(output_height)) {
        // 计算最近邻索引转换函数应用后的输入高度索引
        int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[0]);
        for (const auto ow : c10::irange(output_width)) {
          // 计算最近邻索引转换函数应用后的输入宽度索引
          int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[1]);
          // 计算梯度输出指针位置
          const scalar_t* grad_output_ptr = grad_output_data +
              (n * output_height * output_width + oh * output_width + ow) * channels;
          // 计算缓冲区指针位置
          opmath_t* buffer_ptr = acc_data_ptr + input_offset + (ih * input_width + iw) * channels;
          // 调用最近邻插值计算累加结果
          nearest_channels_last_acc(buffer_ptr, grad_output_ptr, channels);
        }
      }
      // 如果类型不同，更新梯度输入数据
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + n * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }

  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    // 如果类型不同，创建缓冲区数据
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        // 初始化缓冲区数据
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      // 否则，使用梯度输入数据作为累加数据指针
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    for (const auto n : c10::irange(begin, end)) {
      // 计算当前输入数据在缓冲区的起始偏移量
      int64_t input_offset = buffer_data.get() == nullptr ? n * input_slice_size : 0;
      for (int64_t od = 0; od < output_depth; od++) {
        // 计算最近邻索引转换函数应用后的输入深度索引
        int64_t id = nearest_idx_fn(od, input_depth, output_depth, scales[0]);
        for (int64_t oh = 0; oh < output_height; oh++) {
          // 计算最近邻索引转换函数应用后的输入高度索引
          int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[1]);
          for (int64_t ow = 0; ow < output_width; ow++) {
            // 计算最近邻索引转换函数应用后的输入宽度索引
            int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[2]);
            // 计算梯度输出指针位置
            const scalar_t* grad_output_ptr = grad_output_data +
                (n * output_depth * output_height * output_width +
                od * output_height * output_width + oh * output_width + ow) * channels;
            // 计算缓冲区指针位置
            opmath_t* buffer_ptr = acc_data_ptr + input_offset + (id * input_height * input_width + ih * input_width + iw) * channels;
            // 调用最近邻插值计算累加结果
            nearest_channels_last_acc(buffer_ptr, grad_output_ptr, channels);
          }
        }
      }
      // 如果类型不同，更新梯度输入数据
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + n * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }

  };

  if (ndim == 4) {
    // 对于4维情况，使用并行处理进行2D最近邻上采样
    at::parallel_for(0, num_batches, 0, loop2d);
  } else {
    // 对于5维情况，使用并行处理进行3D最近邻上采样
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, num_batches, 0, loop3d);
  }

  // 如果梯度输入不是按照通道优先的内存格式，复制梯度输入数据到相应格式
  if (!grad_input_.is_contiguous(channels_last_memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void upsample_nearest1d_backward_kernel_impl(
    const Tensor& grad_input,  // 输入梯度张量
    const Tensor& grad_output,  // 输出梯度张量
    std::optional<double> scales_w) {  // 可选的缩放因子 scales_w
  // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和额外的类型（kBFloat16, kHalf）
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "upsample_nearest1d_backward", [&] {
    // 调用 CPU 上的 nearest1d 反向插值函数，处理梯度输入和输出
    cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_idx>(grad_input, grad_output, {scales_w});
  });
}

void _upsample_nearest_exact1d_backward_kernel_impl(
    const Tensor& grad_input,  // 输入梯度张量
    const Tensor& grad_output,  // 输出梯度张量
    std::optional<double> scales_w) {  // 可选的缩放因子 scales_w
  // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和额外的类型（kBFloat16, kHalf）
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "_upsample_nearest_exact1d_backward", [&] {
    // 调用 CPU 上的 nearest_exact1d 反向插值函数，处理梯度输入和输出
    cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_exact_idx>(grad_input, grad_output, {scales_w});
  });
}

void upsample_nearest2d_backward_kernel_impl(
    const Tensor& grad_input,  // 输入梯度张量
    const Tensor& grad_output,  // 输出梯度张量
    std::optional<double> scales_h,  // 可选的缩放因子 scales_h
    std::optional<double> scales_w) {  // 可选的缩放因子 scales_w
  // 检查输出梯度是否在 ChannelsLast 内存格式下连续存储
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和额外的类型（kBFloat16, kHalf）
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "upsample_nearest2d_backward_cl", [&] {
      // 调用 CPU 上的 channels-last nearest2d 反向插值函数，处理梯度输入和输出
      cpu_upsample_nearest_backward_channels_last<scalar_t, scale_t, nearest_idx>(grad_input, grad_output, {scales_h, scales_w});
    });
  } else {
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和额外的类型（kBFloat16, kHalf）
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "upsample_nearest2d_backward", [&] {
      // 调用 CPU 上的 nearest2d 反向插值函数，处理梯度输入和输出
      cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_idx>(grad_input, grad_output, {scales_h, scales_w});
    });
  }
}

void _upsample_nearest_exact2d_backward_kernel_impl(
    const Tensor& grad_input,  // 输入梯度张量
    const Tensor& grad_output,  // 输出梯度张量
    std::optional<double> scales_h,  // 可选的缩放因子 scales_h
    std::optional<double> scales_w) {  // 可选的缩放因子 scales_w
  // 检查输出梯度是否在 ChannelsLast 内存格式下连续存储
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和额外的类型（kBFloat16, kHalf）
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "_upsample_nearest_exact2d_backward_cl", [&] {
      // 调用 CPU 上的 channels-last nearest_exact2d 反向插值函数，处理梯度输入和输出
      cpu_upsample_nearest_backward_channels_last<scalar_t, scale_t, nearest_exact_idx>(grad_input, grad_output, {scales_h, scales_w});
    });
  } else {
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和额外的类型（kBFloat16, kHalf）
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "_upsample_nearest_exact2d_backward", [&] {
      // 调用 CPU 上的 nearest_exact2d 反向插值函数，处理梯度输入和输出
      cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_exact_idx>(grad_input, grad_output, {scales_h, scales_w});
    });
  }
}

void upsample_nearest3d_backward_kernel_impl(
    const Tensor& grad_input,  // 输入梯度张量
    const Tensor& grad_output,  // 输出梯度张量
    std::optional<double> scales_d,  // 可选的缩放因子 scales_d
    std::optional<double> scales_h,  // 可选的缩放因子 scales_h
    std::optional<double> scales_w) {  // 可选的缩放因子 scales_w
  // 检查输出梯度是否在 ChannelsLast3d 内存格式下连续存储
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和额外的类型（kBFloat16, kHalf）
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "_upsample_nearest3d_backward_cl", [&] {
      // 调用 CPU 上的 channels-last3d nearest3d 反向插值函数，处理梯度输入和输出
      cpu_upsample_nearest_backward_channels_last<scalar_t, scale_t, nearest_idx>(grad_input, grad_output, {scales_d, scales_h, scales_w});
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "upsample_nearest3d_backward", [&] {
      // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏来处理浮点类型，并包括 kBFloat16 和 kHalf 类型
      // grad_output.scalar_type() 返回梯度输出的数据类型
      // "upsample_nearest3d_backward" 是函数名称字符串
      // [&] 表示使用 Lambda 表达式捕获外部所有变量的引用
      cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_idx>(grad_input, grad_output, {scales_d, scales_h, scales_w});
      // 调用 cpu_upsample_nearest_backward 函数，传递参数 grad_input, grad_output 和 {scales_d, scales_h, scales_w}
    });
  }
}

// 定义了一个函数 `_upsample_nearest_exact3d_backward_kernel_impl`，用于实现3D最近邻上采样的反向传播
void _upsample_nearest_exact3d_backward_kernel_impl(
    const Tensor& grad_input,  // 输入的梯度张量
    const Tensor& grad_output,  // 输出的梯度张量
    std::optional<double> scales_d,  // 深度（D方向）上的缩放比例（可选）
    std::optional<double> scales_h,  // 高度（H方向）上的缩放比例（可选）
    std::optional<double> scales_w) {  // 宽度（W方向）上的缩放比例（可选）
  
  // 如果输出梯度张量按照 ChannelsLast3d 内存格式连续
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    // 根据梯度张量的数据类型选择不同的函数进行最近邻上采样的反向传播
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "_upsample_nearest_exact3d_backward_cl", [&] {
      // 调用特定的 CPU 函数处理 ChannelsLast3d 内存格式的反向传播
      cpu_upsample_nearest_backward_channels_last<scalar_t, scale_t, nearest_exact_idx>(grad_input, grad_output, {scales_d, scales_h, scales_w});
    });
  } else {
    // 否则，选择通常的最近邻上采样的反向传播函数
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "_upsample_nearest_exact3d_backward", [&] {
      // 调用通常的 CPU 函数处理一般情况下的反向传播
      cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_exact_idx>(grad_input, grad_output, {scales_d, scales_h, scales_w});
    });
  }
}

// 定义了一个模板函数 `cpu_upsample_linear_backward`，用于线性插值上采样的反向传播
template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_backward(
    const Tensor& grad_input_,  // 输入的梯度张量
    const Tensor& grad_output_,  // 输出的梯度张量
    bool align_corners,  // 是否对齐角点
    const scale_type& scales) {  // 缩放比例

  // 检查输入梯度和输出梯度的数据类型是否一致
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  // 对输入和输出梯度张量进行连续性处理
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  // 获取梯度张量的数据指针和大小信息
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();
  auto ndim = input_sizes.size();

  // 将批次数和通道数视为一个维度处理
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  // 计算每个切片的大小
  int64_t input_slice_size = input_depth * input_height * input_width;
  int64_t output_slice_size = output_depth * output_height * output_width;
  using opmath_t = at::opmath_type<scalar_t>;

  // 定义一个 lambda 函数处理一维循环
  auto loop1d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    
    // 如果 scalar_t 和 opmath_t 不同，分配并初始化缓冲区
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      // 否则，直接使用 grad_input_data
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    // 计算宽度方向上的缩放比例
    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[0]);

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t iw0, iw1;
    opmath_t w0lambda, w1lambda;
    // 遍历从 `begin` 到 `end` 之间的索引，每次迭代都使用变量 `c`
    for (const auto c : c10::irange(begin, end)) {
      // 根据条件设置输入偏移量：如果 `buffer_data` 为空，则使用 `c * input_slice_size`；否则为 0
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      // 遍历输出宽度的每个元素
      for (const auto ow : c10::irange(output_width)) {
        // 计算源索引和插值权重 lambda，用于水平方向
        compute_source_index_and_lambda(
            iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
        // 获取梯度输出值
        opmath_t grad_output_value = grad_output_data[c * output_slice_size + ow];
        // 更新累加器中的数据：在输入偏移量 `input_offset + iw0` 处加上 `w0lambda * grad_output_value`
        acc_data_ptr[input_offset + iw0] += w0lambda * grad_output_value; /* i0 */
        // 更新累加器中的数据：在输入偏移量 `input_offset + iw1` 处加上 `w1lambda * grad_output_value`
        acc_data_ptr[input_offset + iw1] += w1lambda * grad_output_value; /* i1*/
      }
      // 如果 `scalar_t` 不等于 `opmath_t`，执行条件为真的代码块
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        // 获取梯度输入数据的指针 `gin`，指向 `grad_input_data` 的第 `c * input_slice_size` 位置
        auto gin = grad_input_data + c * input_slice_size;
        // 应用梯度输入到累加器数据 `acc_data_ptr` 中
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  // 定义 lambda 函数 `loop2d`，处理二维循环操作
  auto loop2d = [&](int64_t begin, int64_t end) {
    // 初始化累加器数据指针和缓冲数据
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    // 如果 `scalar_t` 不等于 `opmath_t`，执行条件为真的代码块
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        // 创建大小为 `input_slice_size` 的唯一指针数组 `buffer_data`
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        // 将 `acc_data_ptr` 指向 `buffer_data` 的数据指针
        acc_data_ptr = buffer_data.get();
        // 使用 `memset` 初始化 `acc_data_ptr`，大小为 `sizeof(opmath_t) * input_slice_size`
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      // 否则，将 `acc_data_ptr` 解释为 `grad_input_data` 的 `opmath_t` 指针
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    // 计算高度和宽度的缩放比例
    const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[0]);
    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[1]);

    // 初始化变量 `ih0, ih1, iw0, iw1, h0lambda, h1lambda, w0lambda, w1lambda`
    int64_t ih0, ih1, iw0, iw1;
    opmath_t h0lambda, h1lambda, w0lambda, w1lambda;
    // 遍历从 `begin` 到 `end` 之间的索引，每次迭代都使用变量 `c`
    for (const auto c : c10::irange(begin, end)) {
      // 根据条件设置输入偏移量：如果 `buffer_data` 为空，则使用 `c * input_slice_size`；否则为 0
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      // 遍历输出高度的每个元素
      for (const auto oh : c10::irange(output_height)) {
        // 计算源索引和插值权重 lambda，用于垂直方向
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        // 遍历输出宽度的每个元素
        for (const auto ow : c10::irange(output_width)) {
          // 计算源索引和插值权重 lambda，用于水平方向
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
          // 获取梯度输出值
          opmath_t grad_output_value = grad_output_data[c * output_slice_size + oh * output_width + ow];
          // 更新累加器中的数据：在输入偏移量 `input_offset + ih0 * input_width + iw0` 处加上 `h0lambda * w0lambda * grad_output_value`
          acc_data_ptr[input_offset + ih0 * input_width + iw0] += h0lambda * w0lambda * grad_output_value; /* i00 */
          // 更新累加器中的数据：在输入偏移量 `input_offset + ih0 * input_width + iw1` 处加上 `h0lambda * w1lambda * grad_output_value`
          acc_data_ptr[input_offset + ih0 * input_width + iw1] += h0lambda * w1lambda * grad_output_value; /* i01 */
          // 更新累加器中的数据：在输入偏移量 `input_offset + ih1 * input_width + iw0` 处加上 `h1lambda * w0lambda * grad_output_value`
          acc_data_ptr[input_offset + ih1 * input_width + iw0] += h1lambda * w0lambda * grad_output_value; /* i10 */
          // 更新累加器中的数据：在输入偏移量 `input_offset + ih1 * input_width + iw1` 处加上 `h1lambda * w1lambda * grad_output_value`
          acc_data_ptr[input_offset + ih1 * input_width + iw1] += h1lambda * w1lambda * grad_output_value; /* i11 */
        }
      }
      // 如果 `scalar_t` 不等于 `opmath_t`，执行条件为真的代码块
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        // 获取梯度输入数据的指针 `gin`，指向 `grad_input_data` 的第 `c * input_slice_size` 位置
        auto gin = grad_input_data + c * input_slice_size;
        // 应用梯度输入到累加器数据 `acc_data_ptr` 中
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  // 定义了一个 lambda 函数 loop3d，用于处理 3 维循环的区间 [begin, end)
  auto loop3d = [&](int64_t begin, int64_t end) {
    // 初始化累加数据指针为 nullptr
    opmath_t* acc_data_ptr = nullptr;
    // 声明一个 unique_ptr 用于管理 opmath_t 类型的动态数组 buffer_data
    std::unique_ptr<opmath_t[]> buffer_data;
    // 如果 scalar_t 类型与 opmath_t 类型不同
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        // 创建大小为 input_slice_size 的 opmath_t 类型动态数组 buffer_data
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        // 将 acc_data_ptr 指向 buffer_data 的起始位置
        acc_data_ptr = buffer_data.get();
        // 使用 memset 将 acc_data_ptr 指向的内存区域初始化为 0
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      // 否则，将 acc_data_ptr 解释为 grad_input_data 的 opmath_t 类型指针
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    // 计算深度、高度、宽度的缩放比例
    const opmath_t depth_scale = area_pixel_compute_scale<opmath_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[1]);
    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[2]);

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 声明变量用于索引和插值
    int64_t id0, id1, ih0, ih1, iw0, iw1;
    opmath_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    for (const auto c : c10::irange(begin, end)) {
      // 计算当前输入数据的偏移量，如果数据缓冲为空则为零，否则为常量乘以输入切片大小
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      for (const auto od : c10::irange(output_depth)) {
        // 计算源索引和插值参数，用于深度维度的插值计算
        compute_source_index_and_lambda(
            id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
        for (const auto oh : c10::irange(output_height)) {
          // 计算源索引和插值参数，用于高度维度的插值计算
          compute_source_index_and_lambda(
              ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
          for (const auto ow : c10::irange(output_width)) {
            // 计算源索引和插值参数，用于宽度维度的插值计算
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
            // 计算梯度输出值的索引，更新累加数据指针中对应位置的值
            opmath_t grad_output_value = grad_output_data[c * output_slice_size +
                od *  output_height * output_width + oh * output_width + ow];
            acc_data_ptr[input_offset + id0 * input_height * input_width + ih0 * input_width + iw0] += d0lambda * h0lambda * w0lambda * grad_output_value; /* i000 */
            acc_data_ptr[input_offset + id0 * input_height * input_width + ih0 * input_width + iw1] += d0lambda * h0lambda * w1lambda * grad_output_value; /* i001 */
            acc_data_ptr[input_offset + id0 * input_height * input_width + ih1 * input_width + iw0] += d0lambda * h1lambda * w0lambda * grad_output_value; /* i010 */
            acc_data_ptr[input_offset + id0 * input_height * input_width + ih1 * input_width + iw1] += d0lambda * h1lambda * w1lambda * grad_output_value; /* i011 */
            acc_data_ptr[input_offset + id1 * input_height * input_width + ih0 * input_width + iw0] += d1lambda * h0lambda * w0lambda * grad_output_value; /* i100 */
            acc_data_ptr[input_offset + id1 * input_height * input_width + ih0 * input_width + iw1] += d1lambda * h0lambda * w1lambda * grad_output_value; /* i101 */
            acc_data_ptr[input_offset + id1 * input_height * input_width + ih1 * input_width + iw0] += d1lambda * h1lambda * w0lambda * grad_output_value; /* i110 */
            acc_data_ptr[input_offset + id1 * input_height * input_width + ih1 * input_width + iw1] += d1lambda * h1lambda * w1lambda * grad_output_value; /* i111 */
          }
        }
      }
      // 如果数据类型不同，对梯度输入应用更新函数
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + c * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  if (ndim == 3) {
    // 线性插值一维上采样
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 2, loop1d);
  } else if (ndim == 4) {
    // 双线性插值二维上采样
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // 三线性插值三维上采样
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
  }

  // 如果梯度输入不是连续的，则断言错误
  if (!grad_input_.is_contiguous()) {
    # 使用 PyTorch 中的张量方法 `copy_()` 复制梯度输入 `grad_input` 到 `grad_input_` 中
    grad_input_.copy_(grad_input);
    # 结束函数或代码块
    }
    // 结束上一个lambda函数定义和其使用的大括号
    }
    
    // 模板函数定义，用于计算通道最后格式的线性上采样的反向传播
template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_backward_channels_last(
    // 梯度输入张量
    const Tensor& grad_input_,
    // 梯度输出张量
    const Tensor& grad_output_,
    // 是否对齐角点
    bool align_corners,
    // 缩放因子类型
    const scale_type& scales) {
  // 检查梯度输入和输出的数据类型是否相同
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  // 获取梯度输出的维度数
  auto ndim = grad_output_.ndimension();
  // 检查通道最后格式是否支持具有4或5维的张量
  TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

  // 根据张量维度确定内存格式为通道最后的二维或三维
  auto channels_last_memory_format = ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  // 将梯度输出张量按照指定的内存格式进行连续化
  auto grad_output = grad_output_.contiguous(channels_last_memory_format);
  // 将梯度输入张量按照指定的内存格式进行连续化
  auto grad_input = grad_input_.contiguous(channels_last_memory_format);

  // 获取梯度输出数据的指针
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  // 获取梯度输入数据的指针
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // 获取梯度输入和输出的大小
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();

  // 获取批次数、通道数、输入深度和输出深度、输入高度和输出高度、输入宽度和输出宽度
  int64_t num_batches =  input_sizes[0];
  int64_t channels =  input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];
  int64_t input_slice_size = input_depth * input_height * input_width * channels;
  // 使用opmath_t类型进行数学操作
  using opmath_t = at::opmath_type<scalar_t>;

  // 定义二维循环lambda函数
  auto loop2d = [&](int64_t begin, int64_t end) {
    // 累加数据指针初始化为nullptr
    opmath_t* acc_data_ptr = nullptr;
    // 唯一指针缓冲区数据
    std::unique_ptr<opmath_t[]> buffer_data;
    // 如果scalar_t类型和opmath_t类型不同，分配缓冲区并初始化
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      // 否则使用grad_input_data的opmath_t类型
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    // 计算高度和宽度的缩放比例
    const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[0]);
    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[1]);

    // 定义输入索引函数，根据给定参数返回适当偏移
    auto input_indexr = [=](int64_t n, int64_t h, int64_t w, int64_t offset){
      return acc_data_ptr + offset + (h * input_width + w) * channels;
    };

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 初始化ih0, ih1, iw0, iw1和h0lambda, h1lambda, w0lambda, w1lambda
    int64_t ih0, ih1, iw0, iw1;
    opmath_t h0lambda, h1lambda, w0lambda, w1lambda;
    for (const auto n : c10::irange(begin, end)) {
      // 计算输入数据的偏移量，若缓冲区数据为空则根据输入切片大小计算偏移量
      int64_t input_offset = buffer_data.get() == nullptr ? n * input_slice_size : 0;
      // 遍历输出图像的高度
      for (const auto oh : c10::irange(output_height)) {
        // 计算垂直方向的源索引和插值权重
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        // 遍历输出图像的宽度
        for (const auto ow : c10::irange(output_width)) {
          // 计算水平方向的源索引和插值权重
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
          // 计算梯度输出指针位置
          const scalar_t* grad_output_ptr = grad_output_data +
              (n * output_height * output_width + oh * output_width + ow) * channels;
          // 线性插值累加，处理 channels_last 布局的输入数据
          linear_channels_last_acc(input_indexr(n, ih0, iw0, input_offset), grad_output_ptr, h0lambda * w0lambda, channels); /* i00 */
          linear_channels_last_acc(input_indexr(n, ih0, iw1, input_offset), grad_output_ptr, h0lambda * w1lambda, channels); /* i01 */
          linear_channels_last_acc(input_indexr(n, ih1, iw0, input_offset), grad_output_ptr, h1lambda * w0lambda, channels); /* i10 */
          linear_channels_last_acc(input_indexr(n, ih1, iw1, input_offset), grad_output_ptr, h1lambda * w1lambda, channels); /* i11 */
        }
      }
      // 如果 scalar_t 类型与 opmath_t 类型不同，应用梯度到输入数据
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + n * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  // 定义 3D 循环处理函数
  auto loop3d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    // 若 scalar_t 类型与 opmath_t 类型不同，创建缓冲区以存储梯度数据
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        // 初始化缓冲区数据为 0
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      // 否则，直接使用梯度输入数据指针
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    // 计算深度、高度和宽度的缩放比例
    const opmath_t depth_scale = area_pixel_compute_scale<opmath_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[1]);
    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[2]);

    // 定义输入索引函数，用于计算 3D 输入数据的索引位置
    auto input_indexr = [=](int64_t n, int64_t d, int64_t h, int64_t w, int64_t offset) {
      return acc_data_ptr + offset + (d * input_height * input_width + h * input_width + w) * channels;
    };

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 定义深度和高度的索引、以及插值权重
    int64_t id0, id1, ih0, ih1, iw0, iw1;
    opmath_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    // 遍历输入数据的索引范围 [begin, end)，其中 begin 和 end 是索引的起始和结束位置
    for (const auto n : c10::irange(begin, end)) {
        // 计算当前输入偏移量，如果缓冲区数据为空，则使用默认值；否则，根据当前索引计算偏移量
        int64_t input_offset = buffer_data.get() == nullptr ? n * input_slice_size : 0;
        // 遍历输出的深度维度
        for (const auto od : c10::irange(output_depth)) {
            // 计算深度维度的源索引和插值系数
            compute_source_index_and_lambda(
                id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
            // 遍历输出的高度维度
            for (const auto oh : c10::irange(output_height)) {
                // 计算高度维度的源索引和插值系数
                compute_source_index_and_lambda(
                    ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
                // 遍历输出的宽度维度
                for (const auto ow : c10::irange(output_width)) {
                    // 计算宽度维度的源索引和插值系数
                    compute_source_index_and_lambda(
                        iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
                    // 计算当前梯度输出的指针位置
                    const scalar_t* grad_output_ptr = grad_output_data + (n * output_depth * output_height * output_width +
                        od * output_height * output_width + oh * output_width + ow) * channels;
                    // 对线性插值的通道优先的累加计算，处理不同的输入索引和插值系数
                    linear_channels_last_acc(input_indexr(n, id0, ih0, iw0, input_offset), grad_output_ptr, d0lambda * h0lambda * w0lambda, channels); /* i000 */
                    linear_channels_last_acc(input_indexr(n, id0, ih0, iw1, input_offset), grad_output_ptr, d0lambda * h0lambda * w1lambda, channels); /* i001 */
                    linear_channels_last_acc(input_indexr(n, id0, ih1, iw0, input_offset), grad_output_ptr, d0lambda * h1lambda * w0lambda, channels); /* i010 */
                    linear_channels_last_acc(input_indexr(n, id0, ih1, iw1, input_offset), grad_output_ptr, d0lambda * h1lambda * w1lambda, channels); /* i011 */
                    linear_channels_last_acc(input_indexr(n, id1, ih0, iw0, input_offset), grad_output_ptr, d1lambda * h0lambda * w0lambda, channels); /* i100 */
                    linear_channels_last_acc(input_indexr(n, id1, ih0, iw1, input_offset), grad_output_ptr, d1lambda * h0lambda * w1lambda, channels); /* i101 */
                    linear_channels_last_acc(input_indexr(n, id1, ih1, iw0, input_offset), grad_output_ptr, d1lambda * h1lambda * w0lambda, channels); /* i110 */
                    linear_channels_last_acc(input_indexr(n, id1, ih1, iw1, input_offset), grad_output_ptr, d1lambda * h1lambda * w1lambda, channels); /* i111 */
                }
            }
        }
        // 如果类型不是 opmath_t，对当前输入切片应用梯度输入
        if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
            auto gin = grad_input_data + n * input_slice_size;
            apply_grad_input(acc_data_ptr, gin, input_slice_size);
        }
    }
  };

  // 如果输入的维度是 4，使用双线性插值的方式进行上采样
  if (ndim == 4) {
    // 并行处理每个批次的数据，使用 loop2d 函数
    at::parallel_for(0, num_batches, 0, loop2d);
  } else {
    // 如果输入的维度是 5，使用三线性插值的方式进行上采样
    TORCH_INTERNAL_ASSERT(ndim == 5);
    // 并行处理每个批次的数据，使用 loop3d 函数
    at::parallel_for(0, num_batches, 0, loop3d);
  }

  // 如果梯度输入不是按照通道优先的内存格式连续存储，将梯度输入复制到按照通道优先的内存格式中
  if (!grad_input_.is_contiguous(channels_last_memory_format)) {
    grad_input_.copy_(grad_input);
  }
void upsample_linear1d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_w) {
  // 使用AT_DISPATCH_FLOATING_TYPES_AND2宏，以grad_output的数据类型为参数，生成一个CPU函数调度器，
  // 函数名为"upsample_linear1d_backward"，lambda函数中执行实际的反向线性上采样操作。
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "upsample_linear1d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_w});
  });
}

void upsample_bilinear2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  // 检查grad_output是否按ChannelsLast内存格式连续存储
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    // 使用AT_DISPATCH_FLOATING_TYPES_AND2宏，以grad_output的数据类型为参数，生成一个CPU函数调度器，
    // 函数名为"upsample_bilinear2d_backward_channels_last"，lambda函数中执行通道最后内存格式的反向双线性上采样操作。
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "upsample_bilinear2d_backward_channels_last", [&] {
      cpu_upsample_linear_backward_channels_last<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_h, scales_w});
    });
  } else {
    // 使用AT_DISPATCH_FLOATING_TYPES_AND2宏，以grad_output的数据类型为参数，生成一个CPU函数调度器，
    // 函数名为"upsample_bilinear2d_backward"，lambda函数中执行一般情况下的反向双线性上采样操作。
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "upsample_bilinear2d_backward", [&] {
      cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_h, scales_w});
    });
  }
}

void upsample_trilinear3d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  // 检查grad_output是否按ChannelsLast3d内存格式连续存储
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    // 使用AT_DISPATCH_FLOATING_TYPES_AND2宏，以grad_output的数据类型为参数，生成一个CPU函数调度器，
    // 函数名为"upsample_trilinear3d_backward_channels_last"，lambda函数中执行通道最后3D内存格式的反向三线性上采样操作。
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "upsample_trilinear3d_backward_channels_last", [&] {
      cpu_upsample_linear_backward_channels_last<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_d, scales_h, scales_w});
    });
  } else {
    // 使用AT_DISPATCH_FLOATING_TYPES_AND2宏，以grad_output的数据类型为参数，生成一个CPU函数调度器，
    // 函数名为"upsample_trilinear3d_backward"，lambda函数中执行一般情况下的反向三线性上采样操作。
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, grad_output.scalar_type(), "upsample_trilinear3d_backward", [&] {
      cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_d, scales_h, scales_w});
    });
  }
}
REGISTER_DISPATCH(upsample_trilinear3d_backward_kernel, &upsample_trilinear3d_backward_kernel_impl);



// 注册一个分发函数，将名为 upsample_trilinear3d_backward_kernel 的函数指针与实现函数 upsample_trilinear3d_backward_kernel_impl 绑定起来
REGISTER_DISPATCH(upsample_trilinear3d_backward_kernel, &upsample_trilinear3d_backward_kernel_impl);



} // namespace at::native



// 结束命名空间 at::native
```