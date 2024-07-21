# `.\pytorch\aten\src\ATen\native\cpu\GridSamplerKernel.cpp`

```py
// 定义宏 TORCH_ASSERT_NO_OPERATORS，用于避免编译器中操作符的定义
#define TORCH_ASSERT_NO_OPERATORS
// 包含 GridSampler 相关的头文件
#include <ATen/native/GridSampler.h>
// 包含 GridSamplerKernel 相关的头文件
#include <ATen/native/cpu/GridSamplerKernel.h>
// 包含 TensorBase 类的头文件
#include <ATen/core/TensorBase.h>
// 包含 Dispatch 头文件，用于调度功能
#include <ATen/Dispatch.h>
// 包含 Parallel 头文件，用于并行处理
#include <ATen/Parallel.h>
// 包含 TensorGeometry 头文件，定义了张量的几何属性
#include <ATen/TensorGeometry.h>
// 包含 TensorIterator 头文件，用于处理张量的迭代
#include <ATen/TensorIterator.h>
// 包含 CPU 向量化操作相关的头文件
#include <ATen/cpu/vec/vec.h>
// 包含 c10 的 irange.h 头文件，定义了整数范围的工具函数
#include <c10/util/irange.h>

// 包含标准库的 algorithm 头文件，用于算法操作
#include <algorithm>
// 包含标准库的 cstring 头文件，用于字符串操作
#include <cstring>

// 命名空间 at::native 内部的匿名命名空间
namespace at::native {

// 嵌套命名空间，定义了 GridSampler 的具体实现细节
namespace {

// 使用 at::native::detail 中的 GridSamplerInterpolation 和 GridSamplerPadding
using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;
using namespace at::vec;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ComputeLocation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 结构体，用于计算从网格值到插值位置的映射，并应用填充机制（如反射）。
// 参见 NOTE [ Grid Sample CPU Kernels ] 以获取详细信息。

// 模板结构体，根据是否对齐角点进行插值位置的计算
template<typename scalar_t, bool align_corners>
struct ComputeLocationBase;

// 当 align_corners 为 true 时的特化结构体，用于插值位置计算
template<typename scalar_t>
struct ComputeLocationBase<scalar_t, /*align_corners=*/true> {
  using Vec = Vectorized<scalar_t>;

  // 值被截断到 0 到 max_val 之间
  const scalar_t max_val;
  // 非标准化的缩放因子
  const scalar_t scaling_factor;
  // 反射参数：反射的坐标落在 [low, low+span] 范围内（包括边界）
  const scalar_t low; // 当 align_corners=False 时使用
  const scalar_t twice_span;
  // 如果反射跨度为空，所有反射坐标设置为 0
  const bool empty;

  // 构造函数，初始化各成员变量
  ComputeLocationBase(int64_t size)
    : max_val(static_cast<scalar_t>(size - 1))
    , scaling_factor(static_cast<scalar_t>(size - 1) / 2)
    , low(static_cast<scalar_t>(0))
    , twice_span(static_cast<scalar_t>(size - 1) * 2)
    , empty(size <= 1) {}

  // 标准化方法，将输入向量非标准化
  inline Vec unnormalize(const Vec &in) const {
    return (in + Vec(1)) * Vec(scaling_factor);
  }

  // 坐标裁剪方法，将输入向量的坐标限制在合法范围内
  inline Vec clip_coordinates(const Vec &in) const {
    // 反转 clamp_min 操作数的顺序，以便将 NaN 截断为零
    return clamp_max(Vec(max_val), clamp_min(Vec(0), in));
  }

  // 与 clip_coordinates 相同，同时返回梯度乘数
  inline std::pair<Vec, Vec> clip_coordinates_get_grad(const Vec &in) const {
    using int_t = int_same_size_t<scalar_t>;
    auto bounded_lo = maximum(in, Vec(0));
    // 整数类型的相等比较非常快速，因为它只查看位。强制转换也是免费的。
    // 因此，我们使用以下模式，而不是比较 + blendv。
    // 注意，对于梯度计算很重要的是，边界被视为超出边界。
    auto in_bound_lo = cast<scalar_t>(cast<int_t>(bounded_lo) != cast<int_t>(Vec(0)));
    auto res = minimum(bounded_lo, Vec(max_val));
    auto in_bound_hi = cast<scalar_t>(cast<int_t>(res) != cast<int_t>(Vec(max_val)));
    return std::make_pair(res, in_bound_lo & in_bound_hi);
  }

  // 反射坐标方法，根据空状态判断是否执行反射
  inline Vec reflect_coordinates(const Vec &in) const {
    if (empty) {
      return Vec(0);
    }
    Vec twice_span_vec(twice_span);
    auto abs_in = in.abs();
    auto fdouble_flips = abs_in / twice_span_vec;
    // 将 fdouble_flips 截断为整数部分
    auto double_flips = fdouble_flips.trunc();
    
    // 计算超出部分 extra，用以确定是否需要额外的翻转
    auto extra = abs_in - double_flips * twice_span_vec;
    
    // 检查 extra 是否大于 max_val，以确定是否需要进行另一次翻转。
    // 下面的比较操作实现了这一功能，并返回正确的翻转后的值。
    return minimum(extra, twice_span_vec - extra);
  }



  // 与 reflect_coordinates 相同，但同时返回梯度乘数
  inline std::pair<Vec, Vec> reflect_coordinates_get_grad(const Vec &in) const {
    if (empty) {
      // 如果空，则返回零向量对
      return std::make_pair(Vec(0), Vec(0));
    }
    
    Vec twice_span_vec(twice_span);
    
    // 计算输入向量的负值标志和绝对值
    auto neg_in = in < Vec(0);
    auto abs_in = in.abs();
    
    // 计算 fdouble_flips，即 abs_in 除以 twice_span_vec 的结果
    auto fdouble_flips = abs_in / twice_span_vec;
    
    // 将 fdouble_flips 截断为整数部分
    auto double_flips = fdouble_flips.trunc();

    // 计算超出部分 extra 和反射后的 extra
    auto extra = abs_in - double_flips * twice_span_vec;
    auto reflected_extra = twice_span_vec - extra;
    
    // 判断是否需要进行额外的翻转
    auto one_more_flip = extra > reflected_extra;

    // 根据条件进行值的混合，返回翻转后的坐标和梯度乘数
    return std::make_pair(
      Vec::blendv(extra, reflected_extra, one_more_flip),
      Vec::blendv(Vec(1), Vec(-1), one_more_flip ^ neg_in)
    );
  }
};

template<typename scalar_t>
struct ComputeLocationBase<scalar_t, /*align_corners=*/false> {
  using Vec = Vectorized<scalar_t>;

  // values are clipped to between 0 and max_val
  const scalar_t max_val;
  // unnormalization scaling factor
  const scalar_t scaling_factor;
  // reflection parameters: reflected coordinates land in [low, low+span] inclusive
  const scalar_t low;
  const scalar_t twice_span;
  // if the reflecting span is empty, all reflected coords are set to 0
  const bool empty; // only used when align_corners=True

  // 构造函数，初始化对象的成员变量
  ComputeLocationBase(int64_t size)
    : max_val(static_cast<scalar_t>(size - 1))
    , scaling_factor(static_cast<scalar_t>(size) / 2)
    , low(static_cast<scalar_t>(-0.5))
    , twice_span(static_cast<scalar_t>(size) * 2)
    , empty(size <= 0) {}

  // 对输入向量进行反归一化操作
  inline Vec unnormalize(const Vec &in) const {
    return (in + Vec(1)) * Vec(scaling_factor) - Vec(0.5);
  }

  // 对输入向量进行坐标剪裁操作
  inline Vec clip_coordinates(const Vec &in) const {
    // 反转 clamp_min 操作数的顺序，以便将 NaN 剪裁为零
    return clamp_max(Vec(max_val), clamp_min(Vec(0), in));
  }

  // 与 clip_coordinates 相同，但还返回梯度乘数
  inline std::pair<Vec, Vec> clip_coordinates_get_grad(const Vec &in) const {
    using int_t = int_same_size_t<scalar_t>;
    auto bounded_lo = maximum(in, Vec(0));
    // 整数类型的相等性比较非常快，因为它只检查比特位。类型转换也是免费的。因此，我们使用以下模式，而不是比较 + blendv。
    // 注意，在梯度计算中，边界被认为是越界的很重要。
    auto in_bound_lo = cast<scalar_t>(cast<int_t>(bounded_lo) != cast<int_t>(Vec(0)));
    auto res = minimum(bounded_lo, Vec(max_val));
    auto in_bound_hi = cast<scalar_t>(cast<int_t>(res) != cast<int_t>(Vec(max_val)));
    return std::make_pair(res, in_bound_lo & in_bound_hi);
  }

  // 对输入向量进行反射坐标操作
  inline Vec reflect_coordinates(const Vec &in) const {
    Vec twice_span_vec(twice_span), low_vec(low);
    // 由于反射围绕 low 和 low+span，先在反射之前减去 low，然后在最后加回去。
    auto abs_in = (in - low_vec).abs();
    auto fdouble_flips = abs_in / twice_span_vec;
    auto double_flips = fdouble_flips.trunc();
    auto extra = abs_in - double_flips * twice_span_vec;
    // 现在我们需要测试 extra > max_val，以判断是否需要进行额外的翻转。以下比较完成这一操作，并返回正确的翻转值。
    return minimum(extra, twice_span_vec - extra) + low_vec;
  }

  // 与 reflect_coordinates 相同，但还返回梯度乘数
  inline std::pair<Vec, Vec> reflect_coordinates_get_grad(const Vec &in) const {
    Vec twice_span_vec(twice_span), low_vec(low);
    Vec in_minus_low = in - low_vec;
    auto neg_in = in_minus_low < Vec(0);
    auto abs_in = in_minus_low.abs();
    auto fdouble_flips = abs_in / twice_span_vec;
    // 将 fdouble_flips 截断为整数部分
    auto double_flips = fdouble_flips.trunc();

    // 计算额外部分，即输入减去整数部分乘以两倍的向量
    auto extra = abs_in - double_flips * twice_span_vec;

    // 计算反射后的额外部分，即两倍的向量减去额外部分
    auto reflected_extra = twice_span_vec - extra;

    // 判断是否需要再翻转一次，比较额外部分与反射后的额外部分大小
    auto one_more_flip = extra > reflected_extra;

    // 返回值为一对向量：
    // 1. 使用条件混合向量操作，根据 one_more_flip 选择 extra 或 reflected_extra，再加上低向量 low_vec
    // 2. 使用条件混合向量操作，根据 one_more_flip 异或 neg_in，选择返回 Vec(1) 或 Vec(-1)
    return std::make_pair(
      Vec::blendv(extra, reflected_extra, one_more_flip) + low_vec,
      Vec::blendv(Vec(1), Vec(-1), one_more_flip ^ neg_in)
    );
};

// 模板特化：处理在 padding 为 Zeros 时的位置计算
template<typename scalar_t, bool align_corners>
struct ComputeLocation<scalar_t, GridSamplerPadding::Zeros, align_corners>
  : ComputeLocationBase<scalar_t, align_corners> {
  using Vec = Vectorized<scalar_t>;

  // 继承基类的方法
  using ComputeLocationBase<scalar_t, align_corners>::unnormalize;
  using ComputeLocationBase<scalar_t, align_corners>::scaling_factor;

  // 构造函数继承基类的构造函数
  using ComputeLocationBase<scalar_t, align_corners>::ComputeLocationBase;

  // 应用 unnormalize 函数到输入向量
  inline Vec apply(const Vec &in) const {
    return unnormalize(in);
  }

  // 直接返回输入向量，不进行额外计算
  inline Vec compute_coordinates(const Vec &in) const {
    return in;
  }

  // 应用 unnormalize 函数到输入向量，并返回结果及缩放因子的向量
  inline std::pair<Vec, Vec> apply_get_grad(const Vec &in) const {
    return std::make_pair(unnormalize(in), Vec(scaling_factor));
  }
};

// 模板特化：处理在 padding 为 Border 时的位置计算
template<typename scalar_t, bool align_corners>
struct ComputeLocation<scalar_t, GridSamplerPadding::Border, align_corners>
  : ComputeLocationBase<scalar_t, align_corners> {
  using Vec = Vectorized<scalar_t>;

  // 继承基类的方法
  using ComputeLocationBase<scalar_t, align_corners>::unnormalize;
  using ComputeLocationBase<scalar_t, align_corners>::clip_coordinates;
  using ComputeLocationBase<scalar_t, align_corners>::clip_coordinates_get_grad;
  using ComputeLocationBase<scalar_t, align_corners>::scaling_factor;

  // 构造函数继承基类的构造函数
  using ComputeLocationBase<scalar_t, align_corners>::ComputeLocationBase;

  // 应用 unnormalize 函数到输入向量，再通过 clip_coordinates 函数处理
  inline Vec apply(const Vec &in) const {
    return clip_coordinates(unnormalize(in));
  }

  // 直接通过 clip_coordinates 函数处理输入向量
  inline Vec compute_coordinates(const Vec &in) const {
    return clip_coordinates(in);
  }

  // 应用 unnormalize 函数到输入向量，并返回结果及缩放因子的向量
  inline std::pair<Vec, Vec> apply_get_grad(const Vec &in) const {
    auto [res, grad_clip] = clip_coordinates_get_grad(unnormalize(in));
    return std::make_pair(res, grad_clip & Vec(scaling_factor));
  }
};

// 模板特化：处理在 padding 为 Reflection 时的位置计算
template<typename scalar_t, bool align_corners>
struct ComputeLocation<scalar_t, GridSamplerPadding::Reflection, align_corners>
  : ComputeLocationBase<scalar_t, align_corners> {
  using Vec = Vectorized<scalar_t>;

  // 继承基类的方法
  using ComputeLocationBase<scalar_t, align_corners>::unnormalize;
  using ComputeLocationBase<scalar_t, align_corners>::clip_coordinates;
  using ComputeLocationBase<scalar_t, align_corners>::clip_coordinates_get_grad;
  using ComputeLocationBase<scalar_t, align_corners>::reflect_coordinates;
  using ComputeLocationBase<scalar_t, align_corners>::reflect_coordinates_get_grad;
  using ComputeLocationBase<scalar_t, align_corners>::scaling_factor;

  // 构造函数继承基类的构造函数
  using ComputeLocationBase<scalar_t, align_corners>::ComputeLocationBase;

  // 应用 unnormalize 函数到输入向量，然后通过 reflect_coordinates 和 clip_coordinates 函数处理
  inline Vec apply(const Vec &in) const {
    auto res = reflect_coordinates(unnormalize(in));
    res = clip_coordinates(res);
    return res;
  }

  // 应用 reflect_coordinates 和 clip_coordinates 函数处理输入向量
  inline Vec compute_coordinates(const Vec &in) const {
    auto res = reflect_coordinates(in);
    res = clip_coordinates(res);
    return res;
  }

  // 应用 unnormalize 函数到输入向量，然后通过 reflect_coordinates_get_grad 函数处理，并返回结果及缩放因子的向量
  inline std::pair<Vec, Vec> apply_get_grad(const Vec &in) const {
    auto [res, grad_refl] = reflect_coordinates_get_grad(unnormalize(in));
    // 使用位与运算符 & 处理 grad_clip 和 Vec(scaling_factor) 的逻辑与
    return std::make_pair(res, grad_refl & Vec(scaling_factor));
  }
};
    // 定义两个向量对象：grad_clip 和 grad，并初始化 grad 为 grad_refl 乘以 scaling_factor
    Vec grad_clip, grad(scaling_factor);
    // 将 grad 重新赋值为 grad_refl 与 grad 的点乘结果
    grad = grad_refl * grad;
    // 调用 clip_coordinates_get_grad 函数，获取 res 和 grad_clip 的值，并将其赋给 res 和 grad_clip
    std::tie(res, grad_clip) = clip_coordinates_get_grad(res);
    // 将 grad 更新为 grad_clip 与 grad 的按位与结果
    grad = grad_clip & grad;
    // 返回一个 std::pair 对象，包含 res 和 grad 的值
    return std::make_pair(res, grad);
}
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ApplyGridSample ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 用于应用网格采样的结构体（从输入读取，插值并写入输出）。
// 详见 NOTE [ Grid Sample CPU Kernels ]。

template<typename scalar_t>
static inline void
mask_scatter_add(const scalar_t *src, scalar_t* base_addr,
                 const int_same_size_t<scalar_t> *offsets,
                 const int_same_size_t<scalar_t> *mask, int64_t len) {
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  // 遍历长度为 len 的数组，根据 mask 判断是否应用 scatter add 操作
  for (const auto i : c10::irange(len)) {
    if (mask[i] & 0x01) {
      base_addr[offsets[i]] += src[i];  // 如果 mask[i] 的最低位为 1，则将 src[i] 加到 base_addr[offsets[i]] 上
    }
  }
}

template<typename scalar_t, int spatial_dim,
         GridSamplerInterpolation interp,
         GridSamplerPadding padding,
         bool align_corners>
struct ApplyGridSample;

template<typename scalar_t, GridSamplerPadding padding, bool align_corners>
struct ApplyGridSample<scalar_t, 2, GridSamplerInterpolation::Bilinear,
                       padding, align_corners> {
  using Vec = Vectorized<scalar_t>;
  using integer_t = int_same_size_t<scalar_t>;
  using iVec = Vectorized<integer_t>;

  const int64_t inp_H;
  const int64_t inp_W;
  const int64_t inp_sH;
  const int64_t inp_sW;
  const int64_t C;
  const int64_t inp_sC;
  const ComputeLocation<scalar_t, padding, align_corners> compute_H;
  const ComputeLocation<scalar_t, padding, align_corners> compute_W;
  const bool must_in_bound = padding != GridSamplerPadding::Zeros;

  // 构造函数，初始化各种成员变量
  ApplyGridSample(const TensorAccessor<const scalar_t, 4>& input)
    : inp_H(input.size(2))        // 输入张量的高度
    , inp_W(input.size(3))        // 输入张量的宽度
    , inp_sH(input.stride(2))     // 输入张量的高度步长
    , inp_sW(input.stride(3))     // 输入张量的宽度步长
    , C(input.size(1))            // 输入张量的通道数
    , inp_sC(input.stride(1))     // 输入张量的通道步长
    , compute_H(input.size(2))    // 计算高度的方式
    , compute_W(input.size(3)) {} // 计算宽度的方式

  // 计算插值参数的函数，返回多个 Vec 和 iVec 类型的数据
  inline std::tuple<
    Vec, Vec, Vec, Vec,       // 4个方向的距离
    Vec, Vec, Vec, Vec,       // 4个角落的插值权重
    Vec, Vec, Vec, Vec,       // 是否在边界内的掩码
    iVec, iVec                // y_n 和 x_w 的整数部分
  >
  compute_interp_params(const Vec& x, const Vec& y) const {
    // 获取 (x, y) 周围的 NE、NW、SE、SW 四个像素值
    // 假设我们得到了精确的整数表示，并直接使用 scalar_t
    // 如果不是整数，权重将会是垃圾数据。
    auto x_w = x.floor();   // x 的向下取整
    auto y_n = y.floor();   // y 的向下取整

    // 计算到每个边的距离
    auto w = x - x_w;       // 到西边的距离
    auto e = Vec(1) - w;    // 到东边的距离
    auto n = y - y_n;       // 到北边的距离
    auto s = Vec(1) - n;    // 到南边的距离

    // 计算每个邻居的插值权重
    // 例如，对于 NW 角落，权重是 `到南边的距离 * 到东边的距离`。
    auto nw = s * e;
    auto ne = s * w;
    auto sw = n * e;
    auto se = n * w;

    auto i_x_w = convert_to_int_of_same_size(x_w);  // 将 x_w 转换为相同大小的整数类型
    auto i_y_n = convert_to_int_of_same_size(y_n);  // 将 y_n 转换为相同大小的整数类型
    auto i_x_e = i_x_w + iVec(1);                   // x_w 向东移动一个单位
    auto i_y_s = i_y_n + iVec(1);                   // y_n 向南移动一个单位

    // 使用整数比较，因为在 AVX2 上比浮点数比较要快
    // （在 skylake 上的延迟分别为 1 个周期和 4 个周期）
    // 使用必要的条件判断来生成掩码，确保索引在合法范围内
    auto w_mask = must_in_bound ? iVec(-1)  // 如果必须在边界内，则全为1（真）
                                : (i_x_w > iVec(-1)) & (i_x_w < iVec(inp_W));
    auto n_mask = must_in_bound ? iVec(-1)  // 如果必须在边界内，则全为1（真）
                                : (i_y_n > iVec(-1)) & (i_y_n < iVec(inp_H));
    auto e_mask = must_in_bound ? (i_x_e < iVec(inp_W))  // 如果必须在边界内，则判断是否小于宽度
                                : (i_x_e > iVec(-1)) & (i_x_e < iVec(inp_W));
    auto s_mask = must_in_bound ? (i_y_s < iVec(inp_H))  // 如果必须在边界内，则判断是否小于高度
                                : (i_y_s > iVec(-1)) & (i_y_s < iVec(inp_H));
    auto nw_mask = cast<scalar_t>(must_in_bound ? iVec(-1) : (w_mask & n_mask));  // 如果必须在边界内，则使用西北方向掩码
    auto ne_mask = cast<scalar_t>(e_mask & n_mask);  // 使用东北方向掩码
    auto sw_mask = cast<scalar_t>(w_mask & s_mask);  // 使用西南方向掩码
    auto se_mask = cast<scalar_t>(e_mask & s_mask);  // 使用东南方向掩码

    // 返回元组，包含各方向的插值参数和掩码，以及计算得到的索引偏移值
    return std::make_tuple(
      n, s, w, e,       // 返回北、南、西、东方向的插值参数
      nw, ne, sw, se,   // 返回西北、东北、西南、东南方向的插值参数
      nw_mask, ne_mask, sw_mask, se_mask,  // 返回西北、东北、西南、东南方向的掩码
      i_y_n, i_x_w);    // 返回计算得到的北方向和西方向的索引偏移值
  }

  // 前向插值函数，计算输出切片中的值
  inline void forward(TensorAccessor<scalar_t, 3>& out_slice,
                      const TensorAccessor<const scalar_t, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    // 计算在宽度方向上的加权值
    auto x = compute_W.apply(grid_x);
    // 计算在高度方向上的加权值
    auto y = compute_H.apply(grid_y);

    // 计算插值参数
    auto interp_params = compute_interp_params(x, y);

    // 提取各个方向上的插值参数和掩码
    auto nw = std::get<4>(interp_params);
    auto ne = std::get<5>(interp_params);
    auto sw = std::get<6>(interp_params);
    auto se = std::get<7>(interp_params);

    auto nw_mask = std::get<8>(interp_params);
    auto ne_mask = std::get<9>(interp_params);
    auto sw_mask = std::get<10>(interp_params);
    auto se_mask = std::get<11>(interp_params);

    // 提取计算得到的北方向和西方向的索引偏移值
    auto i_y_n = std::get<12>(interp_params);
    auto i_x_w = std::get<13>(interp_params);

    // 计算各个区域的偏移值
    auto i_nw_offset = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
    auto i_ne_offset = i_nw_offset + iVec(inp_sW);
    auto i_sw_offset = i_nw_offset + iVec(inp_sH);
    auto i_se_offset = i_sw_offset + iVec(inp_sW);

    // 在非 Microsoft Visual Studio 编译器和非最小尺寸编译下，启用循环展开优化
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto c : c10::irange(C)) {
      auto inp_slice_C_ptr = inp_slice[c].data();

      // mask_gather zeros out the mask, so we need to make copies
      // 复制掩码以避免在 mask_gather 中被清零
      Vec nw_mask_copy = nw_mask;
      Vec ne_mask_copy = ne_mask;
      Vec sw_mask_copy = sw_mask;
      Vec se_mask_copy = se_mask;
      // 使用 mask_gather 从输入切片中按掩码收集数据
      auto nw_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_nw_offset, nw_mask_copy);
      auto ne_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_ne_offset, ne_mask_copy);
      auto sw_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_sw_offset, sw_mask_copy);
      auto se_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_se_offset, se_mask_copy);

      // 计算插值结果并存储到输出切片中的指定位置
      auto interpolated = (nw_val * nw) + (ne_val * ne) + (sw_val * sw) + (se_val * se);
      interpolated.store(out_slice[c].data() + offset, len);
    }
  }

  template<bool input_requires_grad>
  inline void backward(TensorAccessor<scalar_t, 3>* gInp_slice_ptr,
                       TensorAccessor<scalar_t, 3>& gGrid_slice,
                       const TensorAccessor<const scalar_t, 3>& gOut_slice,
                       const TensorAccessor<const scalar_t, 3>& inp_slice,
                       int64_t offset, const Vec& grid_x, const Vec& grid_y,
                       int64_t len) const {
    auto [x, gx_mult] = compute_W.apply_get_grad(grid_x);
    auto [y, gy_mult] = compute_H.apply_get_grad(grid_y);

    auto [
      n, s, w, e, nw, ne, sw, se, nw_mask, ne_mask, sw_mask, se_mask,
      i_y_n, i_x_w] = compute_interp_params(x, y);

    auto i_nw_offset = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
    auto i_ne_offset = i_nw_offset + iVec(inp_sW);
    auto i_sw_offset = i_nw_offset + iVec(inp_sH);
    auto i_se_offset = i_sw_offset + iVec(inp_sW);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_nw_mask_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_ne_mask_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_sw_mask_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_se_mask_arr[iVec::size()];
    // 将掩码向量存储到临时数组以备后续使用
    nw_mask.store(i_nw_mask_arr);
    ne_mask.store(i_ne_mask_arr);
    sw_mask.store(i_sw_mask_arr);
    se_mask.store(i_se_mask_arr);

    // i_gInp_*_offset_arr and gInp_corner_arr variables below are unnecessary
    // when input_requires_grad is false (they are only used within the
    // if-blocks), but required to make the code well-formed.

    // When reading input values, we used mask_gather. Unfortunately, there is
    // no mask_scatter_add (the backward of mask_gather) in Intel intrinsics.
    // So we store the necessary vectors to temporary arrays and use the helper
    // mask_scatter_add defined above.
    // 在读取输入值时，我们使用了 mask_gather。不幸的是，Intel 的指令集中没有 mask_scatter_add（mask_gather 的反向操作）。
    // 因此，我们将必要的向量存储到临时数组中，并使用上面定义的 mask_scatter_add 辅助函数。
    // 定义四个整型数组，每个数组长度为 iVec::size()，用于存储偏移量
    integer_t i_gInp_nw_offset_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_gInp_ne_offset_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_gInp_sw_offset_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_gInp_se_offset_arr[iVec::size()];

    // 如果需要计算输入的梯度
    if (input_requires_grad) {
      // 计算四个角的偏移量
      auto i_gInp_nw_offset = i_y_n * iVec(inp_W) + i_x_w;
      auto i_gInp_ne_offset = i_gInp_nw_offset + iVec(1);
      auto i_gInp_sw_offset = i_gInp_nw_offset + iVec(inp_W);
      auto i_gInp_se_offset = i_gInp_sw_offset + iVec(1);

      // 将计算得到的偏移量存储到对应的数组中
      i_gInp_nw_offset.store(i_gInp_nw_offset_arr);
      i_gInp_ne_offset.store(i_gInp_ne_offset_arr);
      i_gInp_sw_offset.store(i_gInp_sw_offset_arr);
      i_gInp_se_offset.store(i_gInp_se_offset_arr);
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    // 定义标量数组 gInp_corner_arr，长度为 Vec::size()
    scalar_t gInp_corner_arr[Vec::size()];

    // 初始化 gx 和 gy 为零向量
    auto gx = Vec(0), gy = Vec(0);

    // 根据条件编译选项，对 C 个通道进行循环
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto c : c10::irange(C)) {
      // 获取当前通道 c 的输入切片和输出梯度切片的指针
      auto inp_slice_C_ptr = inp_slice[c].data();
      auto gOut = Vec::loadu(gOut_slice[c].data() + offset, len);

      // 如果需要计算输入的梯度
      if (input_requires_grad) {
        // 获取当前通道 c 的输入梯度切片的指针
        TORCH_INTERNAL_ASSERT(gInp_slice_ptr);
        auto gInp_slice_C_ptr = (*gInp_slice_ptr)[c].data();

        // 计算四个角的梯度并存储到 gInp_corner_arr 中，然后使用 mask_scatter_add 函数累加到输入梯度切片中
        (nw * gOut).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_nw_offset_arr, i_nw_mask_arr, len);
        (ne * gOut).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_ne_offset_arr, i_ne_mask_arr, len);
        (sw * gOut).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_sw_offset_arr, i_sw_mask_arr, len);
        (se * gOut).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_se_offset_arr, i_se_mask_arr, len);
      }

      // mask_gather 函数根据指定的偏移量和掩码从输入切片中收集数据，然后使用 s、n、e、w 权重计算 gx 和 gy
      Vec nw_mask_copy = nw_mask;
      Vec ne_mask_copy = ne_mask;
      Vec sw_mask_copy = sw_mask;
      Vec se_mask_copy = se_mask;
      auto nw_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_nw_offset, nw_mask_copy);
      auto ne_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_ne_offset, ne_mask_copy);
      auto sw_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_sw_offset, sw_mask_copy);
      auto se_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_se_offset, se_mask_copy);

      gx = gx + ((ne_val - nw_val) * s + (se_val - sw_val) * n) * gOut;
      gy = gy + ((sw_val - nw_val) * e + (se_val - ne_val) * w) * gOut;
    }

    // 对 gx 和 gy 分别乘以其倍数系数 gx_mult 和 gy_mult
    gx = gx * gx_mult;
    gy = gy * gy_mult;

    // 定义步长为 Vec::size()
    constexpr int64_t step = Vec::size();

    // 将 gx 和 gy 交错排列，存储到 interleaved_gGrid 中
    auto interleaved_gGrid = interleave2(gx, gy);
    // 获取指向 gGrid_slice 数组中特定偏移量位置的指针，偏移量乘以2是因为 gGrid_ptr 是指向二维数组的指针
    auto gGrid_ptr = gGrid_slice.data() + offset * 2;
    // 将 gGrid_ptr 指向的数据存储到 interleaved_gGrid 的第一个元素中，
    // 存储长度为 std::min(len * 2, step)，确保不超过给定的长度 step
    std::get<0>(interleaved_gGrid).store(gGrid_ptr,
                                         std::min(len * 2, step));
    // 将 gGrid_ptr + step 指向的数据存储到 interleaved_gGrid 的第二个元素中，
    // 存储长度为 std::max(static_cast<int64_t>(0), len * 2 - step)，保证不小于0
    std::get<1>(interleaved_gGrid).store(gGrid_ptr + step,
                                         std::max(static_cast<int64_t>(0), len * 2 - step));
  }
};

// 定义特化模板结构体，用于处理二维情况下最近邻插值的网格采样
template<typename scalar_t, GridSamplerPadding padding, bool align_corners>
struct ApplyGridSample<scalar_t, 2, GridSamplerInterpolation::Nearest,
                       padding, align_corners> {
  using Vec = Vectorized<scalar_t>;       // 使用 Vectorized 类型来处理 scalar_t 的向量化操作
  using integer_t = int_same_size_t<scalar_t>;  // 使用 int_same_size_t<scalar_t> 作为整数类型
  using iVec = Vectorized<integer_t>;     // 使用 Vectorized 类型来处理 integer_t 的向量化操作

  const int64_t inp_H;    // 输入张量的高度
  const int64_t inp_W;    // 输入张量的宽度
  const int64_t inp_sH;   // 输入张量在高度方向上的步长
  const int64_t inp_sW;   // 输入张量在宽度方向上的步长
  const int64_t C;        // 输入张量的通道数
  const int64_t inp_sC;   // 输入张量在通道数方向上的步长
  const ComputeLocation<scalar_t, padding, align_corners> compute_H;  // 计算高度的位置对象
  const ComputeLocation<scalar_t, padding, align_corners> compute_W;  // 计算宽度的位置对象
  const bool must_in_bound = padding != GridSamplerPadding::Zeros;     // 是否需要进行边界内判断

  // 构造函数，初始化各个成员变量
  ApplyGridSample(const TensorAccessor<const scalar_t, 4>& input)
    : inp_H(input.size(2))
    , inp_W(input.size(3))
    , inp_sH(input.stride(2))
    , inp_sW(input.stride(3))
    , C(input.size(1))
    , inp_sC(input.stride(1))
    , compute_H(input.size(2))
    , compute_W(input.size(3)) {}

  // 前向传播函数，计算输出张量的一个切片
  inline void forward(TensorAccessor<scalar_t, 3>& out_slice,
                      const TensorAccessor<const scalar_t, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    auto x = compute_W.apply(grid_x);   // 计算网格在宽度方向上的位置
    auto y = compute_H.apply(grid_y);   // 计算网格在高度方向上的位置

    auto x_nearest = x.round();         // 将宽度方向上的位置四舍五入到最近的整数
    auto y_nearest = y.round();         // 将高度方向上的位置四舍五入到最近的整数

    auto i_x_nearest = convert_to_int_of_same_size(x_nearest);  // 将宽度方向上的位置转换为相同大小的整数类型
    auto i_y_nearest = convert_to_int_of_same_size(y_nearest);  // 将高度方向上的位置转换为相同大小的整数类型

    auto i_mask = must_in_bound ? iVec(-1)   // 根据是否需要边界内判断，创建一个掩码向量
                                : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                  (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));
    auto mask = cast<scalar_t>(i_mask);     // 将整数掩码向量转换为标量类型的掩码

    auto i_offset = i_y_nearest * iVec(inp_sH) + i_x_nearest * iVec(inp_sW);  // 计算偏移量向量

    auto out_ptr = out_slice.data() + offset;      // 输出张量的指针偏移
    auto out_sC = out_slice.stride(0);             // 输出张量在通道维度上的步长
    auto inp_slice_ptr = inp_slice.data();         // 输入张量的指针

    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    // 遍历通道维度上的每个通道
    for (int64_t c = 0; c < C; ++c, out_ptr += out_sC, inp_slice_ptr += inp_sC) {
      // mask_gather 函数根据掩码从输入张量中提取数据，需要注意需要传入一个掩码副本
      auto mask_copy = mask;
      auto inp_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_ptr, i_offset, mask_copy);
      // 将提取的数据存储到输出张量中
      inp_val.store(static_cast<void*>(out_ptr), len);
    }
  }

  // 反向传播函数，计算梯度
  template<bool input_requires_grad>
  inline void backward(TensorAccessor<scalar_t, 3>* gInp_slice_ptr,
                       TensorAccessor<scalar_t, 3>& gGrid_slice,
                       const TensorAccessor<const scalar_t, 3>& gOut_slice,
                       const TensorAccessor<const scalar_t, 3>& /*inp_slice*/,
                       int64_t offset, const Vec& grid_x, const Vec& grid_y,
                       int64_t len) const {
    // 如果需要输入梯度
    if (input_requires_grad) {
      // 计算宽度方向的双线性插值权重
      auto x = compute_W.apply(grid_x);
      // 计算高度方向的双线性插值权重
      auto y = compute_H.apply(grid_y);

      // 将宽度方向的插值结果四舍五入到最近整数
      auto x_nearest = x.round();
      // 将高度方向的插值结果四舍五入到最近整数
      auto y_nearest = y.round();

      // 将四舍五入后的整数值转换为相同大小的整数向量
      auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
      auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

      // 如果必须在边界内，则初始化一个所有元素为-1的整数向量；否则，根据条件判断得到一个掩码向量
      auto i_mask = must_in_bound ? iVec(-1)
                                  : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                    (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));

      // 计算 gInp 在 gInp_tensor 中的偏移量
      auto i_gInp_offset = i_y_nearest * iVec(inp_W) + i_x_nearest;  // gInp is contiguous

      // 创建一个整数数组来存储掩码向量的值
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      integer_t mask_arr[iVec::size()];
      i_mask.store(mask_arr);

      // 创建一个整数数组来存储 gInp_offset 向量的值
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      integer_t gInp_offset_arr[iVec::size()];
      i_gInp_offset.store(gInp_offset_arr);

      // 循环处理每个通道 c
      #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (const auto c : c10::irange(C)) {
        // 将 gInp 的偏移量加到 gOut_slice 的数据中，同时考虑掩码数组和长度 len
        mask_scatter_add(gOut_slice[c].data() + offset, (*gInp_slice_ptr)[c].data(),
                        gInp_offset_arr, mask_arr, len);
      }
    }

    // 在 Nearest 模式下，网格 gGrid 的梯度为零
    auto gGrid_ptr = gGrid_slice.data() + offset * 2;
    // 将 gGrid_ptr 指向的内存块设置为零，大小为 len * 2 个 scalar_t 类型的元素
    std::memset(gGrid_ptr, 0, sizeof(scalar_t) * len * 2);
};

// 使用双三次卷积算法。基于
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template<typename scalar_t, GridSamplerPadding padding, bool align_corners>
struct ApplyGridSample<scalar_t, 2, GridSamplerInterpolation::Bicubic,
                       padding, align_corners> {
  using Vec = Vectorized<scalar_t>;  // 定义标量类型的向量化操作
  using integer_t = int_same_size_t<scalar_t>;  // 使用与标量类型相同大小的整数类型
  using iVec = Vectorized<integer_t>;  // 整数类型的向量化操作

  const int64_t inp_H;  // 输入张量的高度
  const int64_t inp_W;  // 输入张量的宽度
  const int64_t inp_sH;  // 输入张量的高度步幅
  const int64_t inp_sW;  // 输入张量的宽度步幅
  const int64_t C;  // 输入张量的通道数
  const int64_t inp_sC;  // 输入张量的通道步幅
  const ComputeLocation<scalar_t, padding, align_corners> compute_H;  // 高度计算位置
  const ComputeLocation<scalar_t, padding, align_corners> compute_W;  // 宽度计算位置
  const bool must_in_bound = padding != GridSamplerPadding::Zeros;  // 是否需要边界内操作

  // 用于双三次卷积的常数
  // 可能是 -0.5 或 -0.75，在 UpSampleBicubic2d.h 中使用相同的值
  const Vec A = Vec(-0.75);

  ApplyGridSample(const TensorAccessor<const scalar_t, 4>& input)
    : inp_H(input.size(2))  // 初始化输入张量的高度
    , inp_W(input.size(3))  // 初始化输入张量的宽度
    , inp_sH(input.stride(2))  // 初始化输入张量的高度步幅
    , inp_sW(input.stride(3))  // 初始化输入张量的宽度步幅
    , C(input.size(1))  // 初始化输入张量的通道数
    , inp_sC(input.stride(1))  // 初始化输入张量的通道步幅
    , compute_H(input.size(2))  // 初始化高度计算位置
    , compute_W(input.size(3)) {}  // 初始化宽度计算位置

  // 计算双三次卷积系数
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  inline void get_cubic_coefficients(Vec (&coeffs)[4], const Vec& tx) const {
    Vec x;
    x = tx + Vec(1);  // 1 < x = |-1 - tx| < 2
    coeffs[0] = ((A * x - Vec(5) * A) * x + Vec(8) * A) * x - Vec(4) * A;
    x = tx;           // x = |0 - tx| <= 1
    coeffs[1] = ((A + Vec(2)) * x - (A + Vec(3))) * x * x + Vec(1);
    x = Vec(1) - tx;  // x = |1 - tx| <= 1
    coeffs[2] = ((A + Vec(2)) * x - (A + Vec(3))) * x * x + Vec(1);
    x = Vec(2) - tx;  // 1 < x = |2 - tx| < 2
    coeffs[3] = ((A * x - Vec(5) * A) * x + Vec(8) * A) * x - Vec(4) * A;
  }

  // 计算双三次卷积的导数，即 `d coeff / d x`
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  inline void get_cubic_coefficients_grad(Vec (&coeffs)[4], const Vec& tx) const {
    Vec x;
    x = Vec(-1) - tx; // 1 < x = |-1 - tx| < 2
    coeffs[0] = (Vec(-3) * A * x - Vec(10) * A ) * x - Vec(8) * A;
    x = Vec(0) - tx;  // x = |0 - tx| <= 1
    coeffs[1] = (Vec(-3) * (A + Vec(2)) * x - Vec(2) * (A + Vec(3))) * x;
    x = Vec(1) - tx;  // x = |1 - tx| <= 1
    coeffs[2] = (Vec(3) * (A + Vec(2)) * x - Vec(2) * (A + Vec(3))) * x;
    x = Vec(2) - tx;  // 1 < x = |2 - tx| < 2
    coeffs[3] = (Vec(3) * A * x - Vec(10) * A) * x + Vec(8) * A;
  }

  // 获取有界值，即在边界内操作的数据值
  inline Vec get_value_bounded(const scalar_t* data, const Vec& x, const Vec& y) const {
    auto ix = convert_to_int_of_same_size(compute_W.compute_coordinates(x));  // 计算宽度方向的整数坐标
    auto iy = convert_to_int_of_same_size(compute_H.compute_coordinates(y));  // 计算高度方向的整数坐标

    auto mask_x = must_in_bound ? iVec(-1) : (ix > iVec(-1)) & (ix < iVec(inp_W));
    // 如果需要在边界内操作，则设置掩码为-1；否则根据 ix 是否在有效范围内设置掩码
    // 根据 must_in_bound 变量确定是否使用边界检查条件来生成掩码，若为真则为 -1，否则为 (iy > iVec(-1)) & (iy < iVec(inp_H))
    auto mask_y = must_in_bound ? iVec(-1) : (iy > iVec(-1)) & (iy < iVec(inp_H));
    // 根据 mask_x 和 mask_y 生成最终的掩码
    auto mask = cast<scalar_t>(mask_x & mask_y);

    // 计算在数据中的偏移量，基于输入图像的高度和宽度的偏移
    auto offset = iy * iVec(inp_sH) + ix * iVec(inp_sW);

    // 使用 mask_gather 函数从数据中根据掩码和偏移量收集数据
    auto val = mask_gather<sizeof(scalar_t)>(Vec(0), data, offset, mask);
    // 返回收集到的值
    return val;
  }

  // 向数据中添加值，限制在边界内
  inline void add_value_bounded(scalar_t* data, int64_t len, const Vec& x, const Vec&y,
                               const Vec& delta) const {

    // 计算 x 和 y 的整数坐标
    auto ix = convert_to_int_of_same_size(compute_W.compute_coordinates(x));
    auto iy = convert_to_int_of_same_size(compute_H.compute_coordinates(y));

    // 根据 must_in_bound 变量确定是否使用边界检查条件来生成掩码，若为真则为 -1，否则为 (ix > iVec(-1)) & (ix < iVec(inp_W))
    auto mask_x = must_in_bound ? iVec(-1) : (ix > iVec(-1)) & (ix < iVec(inp_W));
    // 根据 must_in_bound 变量确定是否使用边界检查条件来生成掩码，若为真则为 -1，否则为 (iy > iVec(-1)) & (iy < iVec(inp_H))
    auto mask_y = must_in_bound ? iVec(-1) : (iy > iVec(-1)) & (iy < iVec(inp_H));
    // 根据 mask_x 和 mask_y 生成最终的掩码
    auto mask = cast<scalar_t>(mask_x & mask_y);

    // 计算在数据中的偏移量，基于输入图像的高度和宽度的偏移
    auto i_gInp_offset = iy * iVec(inp_W) + ix;
    // 将 i_gInp_offset 转存到数组中
    integer_t i_gInp_offset_arr[iVec::size()];
    i_gInp_offset.store(i_gInp_offset_arr);

    // 将 mask 转存到数组中
    integer_t mask_arr[iVec::size()];
    mask.store(mask_arr);

    // 将 delta 存储到数组 gInp_corner_arr 中
    scalar_t gInp_corner_arr[Vec::size()];
    delta.store(gInp_corner_arr);

    // 使用 mask_scatter_add 函数将 gInp_corner_arr 中的数据根据 mask_arr 和 i_gInp_offset_arr 添加到 data 中
    mask_scatter_add(gInp_corner_arr, data, i_gInp_offset_arr, mask_arr, len);
  }

  // 执行前向传播，计算输出切片
  inline void forward(TensorAccessor<scalar_t, 3>& out_slice,
                      const TensorAccessor<const scalar_t, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {

    // 根据 grid_x 和 grid_y 计算未标准化的 x 和 y 坐标
    auto x = compute_W.unnormalize(grid_x);
    auto y = compute_H.unnormalize(grid_y);

    // 计算 x 和 y 的地板值，即最大整数不大于 x 和 y 的值
    auto ix = x.floor();
    auto iy = y.floor();

    // 创建用于存储三次插值系数的数组 coeff_x 和 coeff_y
    Vec coeff_x[4];
    Vec coeff_y[4];
    // 获取 x 和 y 的三次插值系数
    get_cubic_coefficients(coeff_x, x - ix);
    get_cubic_coefficients(coeff_y, y - iy);

    // 根据特定条件展开循环（pragma unroll），优化编译器指令以提高性能
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto c : c10::irange(C)) {
      auto inp_slice_C_ptr = inp_slice[c].data();

      // 在 x 方向进行插值，计算四个点的插值结果
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      Vec interp_x[4];
      for (const auto i : c10::irange(4)) {
        interp_x[i] =
          coeff_x[0] * get_value_bounded(inp_slice_C_ptr, ix - Vec(1), iy + Vec(-1 + i)) +
          coeff_x[1] * get_value_bounded(inp_slice_C_ptr, ix + Vec(0), iy + Vec(-1 + i)) +
          coeff_x[2] * get_value_bounded(inp_slice_C_ptr, ix + Vec(1), iy + Vec(-1 + i)) +
          coeff_x[3] * get_value_bounded(inp_slice_C_ptr, ix + Vec(2), iy + Vec(-1 + i));
      }

      // 在 y 方向进行插值，结合 x 方向的插值结果得到最终插值结果
      auto interpolated = coeff_y[0] * interp_x[0] + coeff_y[1] * interp_x[1] +
                          coeff_y[2] * interp_x[2] + coeff_y[3] * interp_x[3];
      // 将插值结果存储到输出张量的指定位置
      interpolated.store(out_slice[c].data() + offset, len);
    }
  }

  template<bool input_requires_grad>
  inline void backward(TensorAccessor<scalar_t, 3>* gInp_slice_ptr,
                      TensorAccessor<scalar_t, 3>& gGrid_slice,
                      const TensorAccessor<const scalar_t, 3>& gOut_slice,
                      const TensorAccessor<const scalar_t, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    // 计算 grid_x 和 grid_y 的逆标准化值
    Vec x = compute_W.unnormalize(grid_x);
    Vec y = compute_H.unnormalize(grid_y);
    // 计算 grid_x 和 grid_y 的缩放因子
    Vec gx_mult = Vec(compute_W.scaling_factor);
    Vec gy_mult = Vec(compute_H.scaling_factor);

    // 计算 x 和 y 的下取整值
    auto ix = x.floor();
    auto iy = y.floor();

    // 计算 x 和 y 方向的三次插值系数
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_x[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_y[4];
    get_cubic_coefficients(coeff_x, x - ix);
    get_cubic_coefficients(coeff_y, y - iy);

    // 计算 x 和 y 方向的三次插值系数的梯度
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_x_grad[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_y_grad[4];
    get_cubic_coefficients_grad(coeff_x_grad, x - ix);
    get_cubic_coefficients_grad(coeff_y_grad, y - iy);

    // 初始化 gx 和 gy 的值
    auto gx = Vec(0), gy = Vec(0);
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    # 遍历输入张量的通道维度
    for (const auto c : c10::irange(C)) {
        # 获取当前通道的输入切片指针
        auto inp_slice_C_ptr = inp_slice[c].data();
        # 从偏移量开始加载输出切片的数据向量
        auto gOut = Vec::loadu(gOut_slice[c].data() + offset, len);
    
        # 遍历 4x4 的区域
        for (const auto i : c10::irange(4)) {
            for (const auto j : c10::irange(4)) {
                # 计算在输入张量上的偏移位置 xx 和 yy
                auto xx = ix + Vec(-1 + i);
                auto yy = iy + Vec(-1 + j);
    
                # 如果需要计算输入张量的梯度
                if (input_requires_grad) {
                    # 获取当前通道的梯度输入切片指针
                    auto gInp_slice_C_ptr = (*gInp_slice_ptr)[c].data();
                    # 添加受限制的值到输入张量梯度切片
                    add_value_bounded(gInp_slice_C_ptr, len, xx, yy, gOut * coeff_x[i] * coeff_y[j]);
                }
    
                # 获取在输入张量上受限制的值
                auto val = get_value_bounded(inp_slice_C_ptr, xx, yy);
                # 计算 gx 和 gy 的梯度值
                gx = gx - val * gOut * coeff_x_grad[i] * coeff_y[j];
                gy = gy - val * gOut * coeff_y_grad[j] * coeff_x[i];
            }
        }
    }
    
    # 对 gx 和 gy 应用预先设定的倍数
    gx = gx * gx_mult;
    gy = gy * gy_mult;
    
    # 定义步长为向量的大小
    constexpr int64_t step = Vec::size();
    # 交错存储 gx 和 gy 到 gGrid_slice 的指定偏移位置
    auto interleaved_gGrid = interleave2(gx, gy);
    auto gGrid_ptr = gGrid_slice.data() + offset * 2;
    # 存储到 gGrid_ptr，限制长度为 step 的一半
    std::get<0>(interleaved_gGrid).store(gGrid_ptr,
                                         std::min(len * 2, step));
    # 存储到 gGrid_ptr + step，限制长度为 len * 2 减去 step 的最大值
    std::get<1>(interleaved_gGrid).store(gGrid_ptr + step,
                                         std::max(static_cast<int64_t>(0), len * 2 - step));
};

// ~~~~~~~~~~~~~~~~~~ grid_sample_2d_grid_slice_iterator ~~~~~~~~~~~~~~~~~~~~~~
// Function to apply a vectorized function on a grid slice tensor (without batch
// dimension).
// See NOTE [ Grid Sample CPU Kernels ] for details.

template<typename scalar_t, typename ApplyFn>
static inline void grid_sample_2d_grid_slice_iterator(
    const TensorAccessor<const scalar_t, 3>& grid_slice, const ApplyFn &apply_fn) {
  // 获取输出网格片张量的尺寸信息
  int64_t out_H = grid_slice.size(0);   // 输出网格片的高度
  int64_t out_W = grid_slice.size(1);   // 输出网格片的宽度
  int64_t grid_sH = grid_slice.stride(0);   // 网格片在高度维度上的步长
  int64_t grid_sW = grid_slice.stride(1);   // 网格片在宽度维度上的步长
  int64_t grid_sCoor = grid_slice.stride(2);   // 网格片在坐标维度上的步长
  auto grid_ptr = grid_slice.data();   // 获取网格片的数据指针

  using Vec = Vectorized<scalar_t>;
  using iVec = Vectorized<int_same_size_t<scalar_t>>;
  constexpr int64_t step = Vec::size();

  // 遍历每个输出像素点的网格
  // 我们考虑以下三种情况（在切片掉批量维度后）。
  // 详细讨论见每个 if-case 下的说明。

  if (at::geometry_is_contiguous({out_H, out_W, 2}, {grid_sH, grid_sW, grid_sCoor})) {
    // 情况 1:
    // 网格是连续的。
    // 策略: 同时加载两个向量，例如 {x0, y0, x1, y1}, {x2, y2, x3, y3}，然后使用
    //      at::vec::deinterleave2 来获取 x 和 y 向量。
    auto total_size = out_H * out_W;
    for (int64_t spatial_offset = 0; spatial_offset < total_size; spatial_offset += step) {
      auto grid_offset = spatial_offset * 2;
      auto len = std::min(step, total_size - spatial_offset);
      auto vec1 = Vec::loadu(grid_ptr + grid_offset,
                             std::min(step, len * 2));
      auto vec2 = Vec::loadu(grid_ptr + grid_offset + step,
                             std::max(static_cast<int64_t>(0), len * 2 - step));
      auto vec_xy_pair = deinterleave2(vec1, vec2);

      auto x = std::get<0>(vec_xy_pair);
      auto y = std::get<1>(vec_xy_pair);

      // 确保 x 和 y 是有效的网格采样位置
      if (len < step) {
        x = Vec::set(Vec(0), x, len);
        y = Vec::set(Vec(0), y, len);
      }
      // 应用函数到 x 和 y 上，以及空间偏移和长度参数
      apply_fn(x, y, spatial_offset, len);
    }
  } else if (grid_sW == 1 || out_W == 1) {
    // 情况 2:
    // W 维度是连续的。
    // 这种情况很常见，例如网格来自形状为 [N, 2, H, W] 的卷积网络输出。
    // 策略: 将其划分为两个连续的片段，每个片段的形状为 [H, W]，每个包含 x 和 y 向量。
    //      因此我们从每个中顺序加载一个向量以获取 x 和 y 向量。

    // 在连续的 W 维度（或者展平的 H x W）上应用的函数
    // 定义一个 Lambda 函数 line_fn，用于处理网格上的行操作，支持 SIMD 并行处理
    auto line_fn = [&](const scalar_t *grid_ptr_x, const scalar_t *grid_ptr_y,
                       int64_t out_base_offset, int64_t total_size) {
      // 循环处理网格数据，每次处理 step 个元素
      for (int64_t i = 0; i < total_size; i += step) {
        // 计算当前迭代的有效长度
        auto len = std::min(step, total_size - i);
        // 从 grid_ptr_x 和 grid_ptr_y 中加载数据到 SIMD 向量 x 和 y
        auto x = Vec::loadu(grid_ptr_x + i, len);
        auto y = Vec::loadu(grid_ptr_y + i, len);
        // 如果当前长度 len 小于 step，则将超出边界的部分设置为零向量
        if (len < step) {
          x = Vec::set(Vec(0), x, len);
          y = Vec::set(Vec(0), y, len);
        }
        // 应用外部传入的 apply_fn 函数处理当前 SIMD 向量 x 和 y
        apply_fn(x, y, out_base_offset + i, len);
      }
    };

    // 检查输出张量是否在内存中连续存储，如果是，则仅调用一次 line_fn 处理整个输出
    if (at::geometry_is_contiguous({out_H, out_W}, {grid_sH, grid_sW})) {
      // 如果 [H, W] 连续，则只需调用一次 line_fn 处理全部数据
      line_fn(grid_ptr, grid_ptr + grid_sCoor, 0, out_H * out_W);
    } else {
      // 如果仅 [W] 连续，则需要逐行调用 line_fn 处理网格的每一行
      auto grid_ptr_NH = grid_ptr;
      for (const auto h : c10::irange(out_H)) {
        line_fn(grid_ptr_NH, grid_ptr_NH + grid_sCoor, h * out_W, out_W);
        grid_ptr_NH += grid_sH;
      }
    }
  } else {
    // Case 3:
    // General case.
    // 一般情况下，使用 for 循环遍历 H，对于每个 W 切片，使用 at::vec::gather 加载 x 和 y 向量
    int64_t spatial_offset = 0;
    const int64_t i_offset_delta = grid_sW * step;

    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto h : c10::irange(out_H)) {
      // 计算当前行的起始位置指针
      auto grid_ptr_x = grid_ptr + h * grid_sH;
      auto grid_ptr_y = grid_ptr_x + grid_sCoor;
      // 创建偏移量数组 i_offsets，用于 gather 操作
      auto i_offsets = iVec::arange(0, grid_sW);
      #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      // 遍历当前行的每个 W 切片
      for (int64_t w = 0; w < out_W; w += step) {
        // 计算当前迭代的有效长度
        auto len = std::min(step, out_W - w);
        // 如果当前长度 len 小于 step，则将超出边界的部分设置为零偏移量
        if (len < step) {
          i_offsets = iVec::set(iVec(0), i_offsets, len);
        }
        // 使用 gather 函数加载 grid_ptr_x 和 grid_ptr_y 中的数据到 SIMD 向量，并应用 apply_fn 处理
        apply_fn(vec::gather<sizeof(scalar_t)>(grid_ptr_x, i_offsets),
                 vec::gather<sizeof(scalar_t)>(grid_ptr_y, i_offsets),
                 spatial_offset, len);

        // 更新 grid_ptr_x 和 grid_ptr_y 的指针位置
        grid_ptr_x += i_offset_delta;
        grid_ptr_y += i_offset_delta;
        // 更新空间偏移量
        spatial_offset += len;
      }
    }
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~ Grid Sample Kernels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 使用上面定义的结构体和函数来计算网格采样的前向和后向操作。
// 详细信息请参见注释 [ Grid Sample CPU Kernels ]。

void grid_sampler_2d_cpu_kernel_impl(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  // 获取输入张量的批次大小、网格的高度和宽度
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto spatial_size = H * W;
  // 计算并设置并行处理的粒度
  auto grain_size = spatial_size == 0 ? (N + 1)
                                      : at::divup(at::internal::GRAIN_SIZE, spatial_size * 4 /* 2d * 2 tensors*/);
  // 如果输出张量为空，直接返回
  if (output.numel() == 0) {
         return;
  }

#define HANDLE_CASE(interp, padding, align_corners)                            \
  case padding: {                                                              \
    // 应用 Grid Sample 算法到输入数据
    ApplyGridSample<scalar_t, 2, interp, padding, align_corners>               \
    grid_sample(inp_acc);                                                      \
    // 并行处理每个批次中的数据
    parallel_for(0, N, grain_size, [&](int64_t begin, int64_t end) {           \
      for (const auto n : c10::irange(begin, end)) {                           \
        auto out_slice = out_acc[n];                                           \
        auto inp_slice = inp_acc[n];                                           \
        // 迭代网格的每个切片并执行网格采样操作
        grid_sample_2d_grid_slice_iterator(                                    \
          grid_acc[n],                                                         \
          [&](const Vectorized<scalar_t>& grid_x, const Vectorized<scalar_t>& grid_y,  \
              int64_t spatial_offset, int64_t len) {                           \
            grid_sample.forward(out_slice, inp_slice, spatial_offset,          \
                                grid_x, grid_y, len);                          \
          });                                                                  \
        }                                                                      \
      });                                                                      \
    return;                                                                    \
  }

#define HANDLE_INTERP(interp, align_corners)                                   \
  case interp: {                                                               \
    switch (static_cast<GridSamplerPadding>(padding_mode)) {                   \
      // 处理不同的填充模式，每种模式下调用相应的处理 CASE 宏
      HANDLE_CASE(interp, GridSamplerPadding::Zeros, align_corners);           \
      HANDLE_CASE(interp, GridSamplerPadding::Border, align_corners);          \
      HANDLE_CASE(interp, GridSamplerPadding::Reflection, align_corners);      \
    }                                                                          \
    return;                                                                    \
  }

  // 根据输入张量的数据类型调度具体的 Grid Sample 实现
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_2d_cpu_kernel_impl", [&] {
    // 获取输出张量的访问器，允许以标量类型访问四维数据
    auto out_acc = output.accessor<scalar_t, 4>();
    // 获取输入张量的常量访问器，允许以标量类型访问四维数据
    auto inp_acc = input.accessor<const scalar_t, 4>();
    // 获取网格张量的常量访问器，允许以标量类型访问四维数据
    auto grid_acc = grid.accessor<const scalar_t, 4>();
    // 如果设置了 align_corners 标志，则执行以下操作
    if (align_corners) {
      // 根据插值模式的枚举值，选择相应的插值方式处理
      switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
        // 处理双线性插值，使用 align_corners 的方式
        HANDLE_INTERP(GridSamplerInterpolation::Bilinear, true);
        // 处理最近邻插值，使用 align_corners 的方式
        HANDLE_INTERP(GridSamplerInterpolation::Nearest, true);
        // 处理双三次插值，使用 align_corners 的方式
        HANDLE_INTERP(GridSamplerInterpolation::Bicubic, true);
      }
    } else {
      // 如果未设置 align_corners 标志，则执行以下操作
      // 根据插值模式的枚举值，选择相应的插值方式处理
      switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
        // 处理双线性插值，不使用 align_corners 的方式
        HANDLE_INTERP(GridSamplerInterpolation::Bilinear, false);
        // 处理最近邻插值，不使用 align_corners 的方式
        HANDLE_INTERP(GridSamplerInterpolation::Nearest, false);
        // 处理双三次插值，不使用 align_corners 的方式
        HANDLE_INTERP(GridSamplerInterpolation::Bicubic, false);
      }
    }
  });
#undef HANDLE_CASE
#undef HANDLE_INTERP
}

void grid_sampler_2d_backward_cpu_kernel_impl(
    const TensorBase &grad_input,
    const TensorBase &grad_grid,
    const TensorBase &grad_output_,
    const TensorBase &input,
    const TensorBase &grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool,2> output_mask) {
  if (grad_output_.numel() == 0) {
    // 如果 grad_output 的元素数量为 0，则将 grad_grid 清零并返回
    grad_grid.zero_();
    return;
  }
  // grad_output 应该大部分时间是连续的。确保它是连续的可以极大简化这段代码。
  auto grad_output = grad_output_.contiguous();

  // 如果不需要计算 `input` 的梯度，我们跳过该计算 -- 不需要创建梯度张量可以显著提高性能。
  auto input_requires_grad = output_mask[0];

  auto N = input.size(0);
  auto spatial_size = grid.size(1) * grid.size(2);
  auto grain_size = spatial_size == 0 ? (N + 1)
                                      : at::divup(at::internal::GRAIN_SIZE, spatial_size * 10 /* 2d * 5 tensors*/);

#define GINP_SLICE_PTR_true auto gInp_slice = gInp_acc[n]; auto gInp_slice_ptr = &gInp_slice;
#define GINP_SLICE_PTR_false TensorAccessor<scalar_t, 3>* gInp_slice_ptr = nullptr;
#define GINP_SLICE_PTR(input_requires_grad) GINP_SLICE_PTR_##input_requires_grad

#define HANDLE_CASE(interp, padding, align_corners, input_requires_grad)         \
  case padding: {                                                                \
    // 处理不同的填充模式，应用对应的网格采样函数
    ApplyGridSample<scalar_t, 2, interp, padding, align_corners>                 \
    grid_sample(inp_acc);                                                        \
    // 并行循环，分割任务范围从 begin 到 end，grain_size 是每个任务的最小单位大小
    parallel_for(0, N, grain_size, [&](int64_t begin, int64_t end) {             \
      // 遍历从 begin 到 end 的范围
      for (const auto n : c10::irange(begin, end)) {                             \
        // 定义需要梯度的输入的指针
        GINP_SLICE_PTR(input_requires_grad)
        // 获取当前索引对应的梯度网格和输出梯度切片
        auto gGrid_slice = gGrid_acc[n];
        auto gOut_slice = gOut_acc[n];
        auto inp_slice = inp_acc[n];
        // 对当前网格进行二维插值处理
        grid_sample_2d_grid_slice_iterator(                                      \
          grid_acc[n],                                                           \
          // lambda 函数处理每个网格点的插值过程
          [&](const Vectorized<scalar_t>& grid_x, const Vectorized<scalar_t>& grid_y,    \
              int64_t spatial_offset, int64_t len) {                             \
            // 调用 grid_sample 的反向传播方法，根据输入是否需要梯度处理
            grid_sample.backward<input_requires_grad>(gInp_slice_ptr, gGrid_slice,       \
                                                      gOut_slice, inp_slice,     \
                                                      spatial_offset, grid_x,    \
                                                      grid_y, len);              \
          });                                                                    \
      }                                                                          \
    });                                                                          \
    // 返回空，函数结束
    return;                                                                      
    }
#define HANDLE_INTERP(interp, align_corners, input_requires_grad)           \
  case interp: {                                                            \  // 处理不同的插值方式，interp为插值方式，align_corners为是否对齐角点，input_requires_grad表示是否需要计算输入的梯度
    switch (static_cast<GridSamplerPadding>(padding_mode)) {                \  // 根据padding_mode选择不同的填充方式
      HANDLE_CASE(interp, GridSamplerPadding::Zeros, align_corners, input_requires_grad);      \  // 处理特定插值方式和填充方式的情况，例如Zeros填充方式
      HANDLE_CASE(interp, GridSamplerPadding::Border, align_corners, input_requires_grad);     \  // 处理特定插值方式和填充方式的情况，例如Border填充方式
      HANDLE_CASE(interp, GridSamplerPadding::Reflection, align_corners, input_requires_grad); \  // 处理特定插值方式和填充方式的情况，例如Reflection填充方式
    }                                                                       \  // 结束padding_mode的switch语句
    return;                                                                 \  // 返回结果
  }                                                                         \  // 结束interp的case语句

AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_2d_backward_cpu_kernel_impl", [&] {  \  // 根据input的数据类型分发函数调用，执行lambda表达式
    auto gGrid_acc = grad_grid.accessor<scalar_t, 4>();                     \  // 创建grad_grid的访问器，访问四维张量
    auto inp_acc = input.accessor<const scalar_t, 4>();                     \  // 创建input的常量访问器，访问四维张量
    auto grid_acc = grid.accessor<const scalar_t, 4>();                     \  // 创建grid的常量访问器，访问四维张量
    auto gOut_acc = grad_output.accessor<const scalar_t, 4>();              \  // 创建grad_output的常量访问器，访问四维张量
    if (input_requires_grad) {                                              \  // 如果需要计算输入的梯度
      auto gInp_acc = grad_input.accessor<scalar_t, 4>();                   \  // 创建grad_input的访问器，访问四维张量
      if (align_corners) {                                                  \  // 如果对齐角点
        switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) { \  // 根据插值模式选择不同的插值方式
          HANDLE_INTERP(GridSamplerInterpolation::Bilinear, true, true);    \  // 处理Bilinear插值的情况，对齐角点，需要计算输入的梯度
          HANDLE_INTERP(GridSamplerInterpolation::Nearest, true, true);     \  // 处理Nearest插值的情况，对齐角点，需要计算输入的梯度
          HANDLE_INTERP(GridSamplerInterpolation::Bicubic, true, true);     \  // 处理Bicubic插值的情况，对齐角点，需要计算输入的梯度
        }
      } else {
        switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) { \  // 不对齐角点的情况下，根据插值模式选择不同的插值方式
          HANDLE_INTERP(GridSamplerInterpolation::Bilinear, false, true);   \  // 处理Bilinear插值的情况，不对齐角点，需要计算输入的梯度
          HANDLE_INTERP(GridSamplerInterpolation::Nearest, false, true);    \  // 处理Nearest插值的情况，不对齐角点，需要计算输入的梯度
          HANDLE_INTERP(GridSamplerInterpolation::Bicubic, false, true);    \  // 处理Bicubic插值的情况，不对齐角点，需要计算输入的梯度
        }
      }
    } else {
      if (align_corners) {                                                  \  // 如果对齐角点
        switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) { \  // 根据插值模式选择不同的插值方式
          HANDLE_INTERP(GridSamplerInterpolation::Bilinear, true, false);   \  // 处理Bilinear插值的情况，对齐角点，不需要计算输入的梯度
          HANDLE_INTERP(GridSamplerInterpolation::Nearest, true, false);    \  // 处理Nearest插值的情况，对齐角点，不需要计算输入的梯度
          HANDLE_INTERP(GridSamplerInterpolation::Bicubic, true, false);    \  // 处理Bicubic插值的情况，对齐角点，不需要计算输入的梯度
        }
      } else {
        switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) { \  // 不对齐角点的情况下，根据插值模式选择不同的插值方式
          HANDLE_INTERP(GridSamplerInterpolation::Bilinear, false, false);  \  // 处理Bilinear插值的情况，不对齐角点，不需要计算输入的梯度
          HANDLE_INTERP(GridSamplerInterpolation::Nearest, false, false);   \  // 处理Nearest插值的情况，不对齐角点，不需要计算输入的梯度
          HANDLE_INTERP(GridSamplerInterpolation::Bicubic, false, false);   \  // 处理Bicubic插值的情况，不对齐角点，不需要计算输入的梯度
        }
      }
    }
  });
#undef HANDLE_CASE                                                             \  // 取消定义HANDLE_CASE宏
#undef HANDLE_INTERP                                                           \  // 取消定义HANDLE_INTERP宏
}

}

REGISTER_DISPATCH(grid_sampler_2d_cpu_kernel, &grid_sampler_2d_cpu_kernel_impl); \  // 注册grid_sampler_2d_cpu_kernel函数调用，使用grid_sampler_2d_cpu_kernel_impl实现
REGISTER_DISPATCH(grid_sampler_2d_backward_cpu_kernel, &grid_sampler_2d_backward_cpu_kernel_impl); \  // 注册grid_sampler_2d_backward_cpu_kernel函数调用，使用grid_sampler_2d_backward_cpu_kernel_impl实现

}  // namespace at::native                                                     \  // 结束at::native命名空间
```