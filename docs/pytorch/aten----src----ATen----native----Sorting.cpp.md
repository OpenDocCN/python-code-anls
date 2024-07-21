# `.\pytorch\aten\src\ATen\native\Sorting.cpp`

```py
// 定义编译时仅支持方法操作符的宏，仅在此处起作用
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中必要的头文件，用于张量操作和计算
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NumericUtils.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/Sorting.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/irange.h>

// 根据不同情况选择是否包含完整操作符的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/argsort_native.h>
#include <ATen/ops/broadcast_tensors.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/kthvalue.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/median.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/msort_native.h>
#include <ATen/ops/nanmedian.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/nanquantile_native.h>
#include <ATen/ops/quantile_native.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/sort_native.h>
#include <ATen/ops/topk_native.h>
#endif

#include <utility>

// 定义命名空间 at::meta，用于 ATen 的元信息操作
namespace at::meta {

// 使用命名空间 at::native
using namespace ::at::native;

// 定义 TORCH_META_FUNC 宏处理 topk 函数元信息
TORCH_META_FUNC(topk)
(const Tensor& self, int64_t k, int64_t dim_, bool largest, bool sorted) {
  // 对 dim 进行处理，确保在有效范围内
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);
  TORCH_CHECK(k >= 0 && k <= sliceSize, "k not in range for dimension");

  // 构建输出的大小，将选择的维度设置为大小为 k
  DimVector topKSize(self.sizes().vec());
  if (!topKSize.empty()) {
    topKSize[dim] = k;
  }
  // 设置第一个输出张量的大小和选项
  set_output_raw_strided(0, topKSize, {}, self.options());
  // 设置第二个输出张量的大小、选项和数据类型
  set_output_raw_strided(1, topKSize, {}, self.options().dtype(at::kLong));
}

// 定义 TORCH_META_FUNC2 宏处理 sort 函数元信息，支持稳定排序
TORCH_META_FUNC2(sort, stable)
(const Tensor& self, std::optional<bool> stable, int64_t dim, bool descending) {
  // 对 dim 进行处理，确保在有效范围内
  maybe_wrap_dim(dim, self.dim());

  // 解决问题：https://github.com/pytorch/pytorch/issues/65863
  // 应当使用密集的步长，以免分配过多内存
  // 使用 self 的步长，或从中推断出密集步长
  std::vector<int64_t> strides = (self.is_non_overlapping_and_dense())
      ? self.strides().vec()
      : at::infer_dense_strides(self.sizes(), self.strides());

  // 设置第一个输出张量的大小、步长、选项和数据类型
  set_output_raw_strided(0, self.sizes(), strides, self.options(), {});
  // 设置第二个输出张量的大小、步长、选项和数据类型为长整型
  set_output_raw_strided(1, self.sizes(), strides, self.options().dtype(kLong), {});
}

} // namespace at::meta

// 定义命名空间 at::native，用于 ATen 的本地操作
namespace at::native {

// 定义 sort_stub 的分发器
DEFINE_DISPATCH(sort_stub);

} // namespace at::native
QUANTILE_INTERPOLATION_MODE get_quantile_interpolation_mode(
    const c10::string_view interpolation) {
  // 根据插值方式字符串确定并返回对应的插值模式枚举值
  if (interpolation == "linear") {
    return QUANTILE_INTERPOLATION_MODE::LINEAR;
  } else if (interpolation == "lower") {
    return QUANTILE_INTERPOLATION_MODE::LOWER;
  } else if (interpolation == "higher") {
    return QUANTILE_INTERPOLATION_MODE::HIGHER;
  } else if (interpolation == "midpoint") {
    return QUANTILE_INTERPOLATION_MODE::MIDPOINT;
  } else if (interpolation == "nearest") {
    return QUANTILE_INTERPOLATION_MODE::NEAREST;
  } else {
    // 如果插值方式未知，则默认返回 NEAREST 插值模式
    return QUANTILE_INTERPOLATION_MODE::NEAREST;
  }
}
    TORCH_CHECK(
        false,
        // 检查条件，如果为 false，则抛出错误
        "quantile() interpolation must be one of linear, lower, higher, midpoint or nearest, but got ",
        interpolation);
  }
}

// 检查输入张量 self 是否非空
// 检查输入张量 q 是否为标量或者1维张量
// 检查输入张量 self 是否为 float 或 double 类型
// 检查输入张量 self 和 q 是否具有相同的数据类型
// 检查输入张量 self 和 q 是否在相同设备上
void quantile_checks(const Tensor& self, const Tensor& q) {
  TORCH_CHECK(self.numel() > 0, "quantile() input tensor must be non-empty");
  TORCH_CHECK(q.dim() <= 1, "quantile() q must be a scalar or 1D tensor");
  TORCH_CHECK(
      self.scalar_type() == kFloat || self.scalar_type() == kDouble,
      "quantile() input tensor must be either float or double dtype");
  TORCH_CHECK(
      self.scalar_type() == q.scalar_type(),
      "quantile() q tensor must be same dtype as the input tensor");
  TORCH_CHECK(
      self.device() == q.device(),
      "quantile() q tensor must be on the same device as the input tensor");
}

// 计算 quantile() 的输出形状
std::vector<int64_t> quantile_output_shape(
    const optional<int64_t> original_dim,
    const Tensor& self,
    const Tensor& q,
    const bool keepdim,
    int64_t wrapped_dim) {
  // 初始化输出形状向量
  std::vector<int64_t> out_shape;
  
  // 如果有原始维度信息且输入张量 self 的维度大于 0
  if (original_dim && self.dim() > 0) {
    // 将输出形状初始化为 self 的大小向量
    out_shape = self.sizes().vec();
    // 如果 keepdim 为真，则在 wrapped_dim 维度上将大小设为 1，否则移除 wrapped_dim 维度
    if (keepdim) {
      out_shape[wrapped_dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + wrapped_dim);
    }
  } else if (keepdim) {
    // 如果 keepdim 为真，则将输出形状初始化为全为 1 的向量，长度为 self 的维度数
    out_shape = std::vector<int64_t>(self.dim(), 1);
  }
  
  // 如果 q 是1维张量，则在输出形状的开头插入 q 的元素数
  if (q.dim() > 0) {
    out_shape.insert(out_shape.begin(), q.numel());
  }

  return out_shape;
}

// 计算 quantile() 的实际运算结果
Tensor quantile_compute(
    const Tensor& self,
    const Tensor& q,
    const optional<int64_t> orginal_dim,
    const bool keepdim,
    const QUANTILE_INTERPOLATION_MODE& interpolation,
    const bool ignore_nan,
    int64_t wrapped_dim,
    std::vector<int64_t> out_shape) {
  // 当在 CPU 上运行时，检查所有 q 值是否在 [0, 1] 范围内
  // 避免将加速器与 CPU 同步，以提高效率
  if (self.device().is_cpu()) {
    auto all_q_in_range = q.ge(0).logical_and_(q.le(1)).all();
    TORCH_CHECK(at::is_scalar_tensor_true(all_q_in_range),
                "quantile() q values must be in the range [0, 1]");
  }

  // 根据情况对输入张量进行扁平化或者维度转换，并对其进行排序以便有效地查询第 k 个值
  Tensor sorted;
  if (!orginal_dim) {
    sorted = std::get<0>(self.flatten().sort());
  } else if (wrapped_dim == self.dim() - 1) {
    sorted = std::get<0>(self.sort());
  } else {
    sorted = std::get<0>(self.unsqueeze(-1).transpose(wrapped_dim, -1).sort());
  }

  // 将 q 视为 1 维张量以便进行下面的计算
  if (q.dim() == 0) {


这些注释为每行代码提供了详细的解释，确保代码功能和逻辑清晰可理解。
    // 在 out_shape 的开头插入输出张量的元素数量
    out_shape.insert(out_shape.begin(), q.numel());
  }

  // 将输入视为 reduced_size + 要减少的维度的大小
  std::vector<int64_t> in_shape(out_shape.size());
  std::copy(out_shape.begin() + 1, out_shape.end(), in_shape.begin());
  in_shape[in_shape.size() - 1] = sorted.size(-1);
  sorted = sorted.view(in_shape);

  // 确保从 int64_t 转换为 double 不会溢出
  TORCH_CHECK(
      sorted.size(-1) <= std::pow(2, 24),
      "quantile() 输入张量过大");

  // 将 q 从 [0, 1] 转换为 ranks 在 [0, reduction_size) 的范围内
  Tensor ranks;
  if (ignore_nan) {
    // 对于 nanquantile，基于非 NaN 值的数量计算 ranks
    // 如果所有值都是 NaN，则将 rank 设置为 0，以便计算的分位数为 NaN
    ranks = q * (sorted.isnan().logical_not_().sum(-1, true) - 1);
    // 对于 Composite Compliance，
    // 如果 `ranks` 是 `CCT`，但其 tangent 是一个普通 Tensor，
    // 在计算 jvp 时，我们会在普通 Tensor 上调用 `masked_fill_`，
    // 使用 `masked_fill` 替代。
    if (isTensorSubclassLike(ranks) && ranks._fw_grad(/*level=*/0).defined()) {
      ranks = ranks.masked_fill(ranks < 0, 0);
    } else {
      ranks.masked_fill_(ranks < 0, 0);
    }
  } else {
    // 对于 quantile，基于 reduction size 计算 ranks
    // 如果存在 NaN，则将 rank 设置为最后一个索引，以便计算的分位数为 NaN
    int64_t last_index = sorted.size(-1) - 1;
    std::vector<Tensor> tl =
        at::broadcast_tensors({q * last_index, sorted.isnan().any(-1, true)});
    ranks = at::masked_fill(tl[0], tl[1], last_index);
  }

  // 根据插值模式调整 ranks
  if (interpolation == QUANTILE_INTERPOLATION_MODE::LOWER) {
    ranks.floor_();
  } else if (interpolation == QUANTILE_INTERPOLATION_MODE::HIGHER) {
    ranks.ceil_();
  } else if (interpolation == QUANTILE_INTERPOLATION_MODE::NEAREST) {
    ranks.round_();
  }

  // 将 ranks 转换为 kLong 类型
  Tensor ranks_below = ranks.toType(kLong);
  // 根据 ranks_below 从 sorted 中聚合对应的值
  Tensor values_below = sorted.gather(-1, ranks_below);

  // 仅在线性和中点模式下需要实际插值
  if (interpolation == QUANTILE_INTERPOLATION_MODE::LINEAR ||
      interpolation == QUANTILE_INTERPOLATION_MODE::MIDPOINT) {
    // 计算线性和中点模式的权重
    Tensor weights = interpolation == QUANTILE_INTERPOLATION_MODE::MIDPOINT
        ? at::full_like(ranks, 0.5)
        : ranks - ranks_below;

    // 插值计算分位数，并存储在 values_below 中
    Tensor ranks_above = ranks.ceil_().toType(kLong);
    Tensor values_above = sorted.gather(-1, ranks_above);
    // 对于 Composite Compliance，
    // 如果 `values_below`、`values_above` 或 `weights` 是 CCT，
    // 或者 `value_above` 和 `weights` 的 tangents 是 CCT，
    // 但如果 `value_below` 的 tangent 是普通 Tensor，
    // 在计算 jvp 时，我们将会将 `CCT` 复制到普通 Tensor 中。
    // 因此，我们使用 `lerp` 的非就地变体。
    # 检查是否任何一个张量子类似于 C++ 的类
    auto is_primal_cct =
        areAnyTensorSubclassLike({values_below, values_above, weights});
    # 检查是否任何一个张量子类似于 C++ 的类，并且是切线 C++ 的类
    auto is_tangent_cct = areAnyTensorSubclassLike(
        {values_above._fw_grad(/*level=*/0), weights._fw_grad(/*level=*/0)});
    # 如果是原始 C++ 的类或者切线 C++ 的类，并且 values_below 的梯度在 level 0 存在，并且其不是张量子类似于 C++ 的类
    if ((is_primal_cct || is_tangent_cct) &&
        values_below._fw_grad(/*level=*/0).defined() &&
        !isTensorSubclassLike(values_below._fw_grad(/*level=*/0))) {
      # 在 values_below 上执行线性插值操作，结果存回 values_below
      values_below = values_below.lerp(values_above, weights);
    } else {
      # 在 values_below 上原地执行线性插值操作，结果存回 values_below
      values_below.lerp_(values_above, weights);
    }
  }

  # 如果 q 的维度为 0
  if (q.dim() == 0) {
    # 如果 q 是标量，则移除最后一个维度以匹配输出形状
    values_below.squeeze_(-1);
  } else {
    # 如果 q 不是标量，则将分位数移动到第一个维度以匹配输出形状
    values_below.unsqueeze_(0).transpose_(0, -1).squeeze_(-1);
  }

  # 返回 values_below
  return values_below;
}

} // namespace

// 定义 quantile_out_impl 函数，计算输入张量 self 的分位数，并将结果写入 out 张量中
void quantile_out_impl(
    Tensor& out, // 输出张量，存储计算得到的分位数结果
    const Tensor& self, // 输入张量，计算其分位数
    const Tensor& q, // 分位数的列表
    const optional<int64_t> original_dim, // 原始维度的可选参数
    const bool keepdim, // 是否保持维度
    const QUANTILE_INTERPOLATION_MODE& interpolation, // 分位数插值模式
    const bool ignore_nan) { // 是否忽略 NaN 值

  quantile_checks(self, q); // 执行分位数计算前的检查

  // 检查输出张量的数据类型必须与输入张量相同
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      "quantile() out tensor must be same dtype as the input tensor");

  // 检查输出张量必须与输入张量在同一设备上
  TORCH_CHECK(
      self.device() == out.device(),
      "quantile() out tensor must be on the same device as the input tensor");

  // 根据原始维度参数获取包装后的维度索引
  int64_t wrapped_dim = at::maybe_wrap_dim(original_dim.value_or(0), self.dim());

  // 计算输出张量的形状
  auto out_shape = quantile_output_shape(original_dim, self, q, keepdim, wrapped_dim);

  // 调整输出张量的大小
  resize_output(out, out_shape);

  // 执行分位数计算，并将结果拷贝到输出张量中
  auto quantile = quantile_compute(
      self, q, original_dim, keepdim, interpolation, ignore_nan, wrapped_dim, std::move(out_shape));
  out.copy_(quantile);
}

// 定义 quantile_impl 函数，计算输入张量 self 的分位数，并返回结果张量
Tensor quantile_impl(
    const Tensor& self, // 输入张量，计算其分位数
    const Tensor& q, // 分位数的列表
    const optional<int64_t> original_dim, // 原始维度的可选参数
    const bool keepdim, // 是否保持维度
    const QUANTILE_INTERPOLATION_MODE& interpolation, // 分位数插值模式
    const bool ignore_nan) { // 是否忽略 NaN 值

  quantile_checks(self, q); // 执行分位数计算前的检查

  // 根据原始维度参数获取包装后的维度索引
  int64_t wrapped_dim = at::maybe_wrap_dim(original_dim.value_or(0), self.dim());

  // 计算输出张量的形状
  auto out_shape = quantile_output_shape(original_dim, self, q, keepdim, wrapped_dim);

  // 执行分位数计算，并返回结果张量
  return quantile_compute(
      self, q, original_dim, keepdim, interpolation, ignore_nan, wrapped_dim, std::move(out_shape));
}

// 定义 kthvalue_out_impl_cpu 函数，计算输入张量 self 在指定维度 dim 上的第 k 小值，并返回值和索引张量
std::tuple<Tensor&, Tensor&> kthvalue_out_impl_cpu(
    Tensor& values, // 输出张量，存储计算得到的第 k 小值
    Tensor& indices, // 输出张量，存储计算得到的第 k 小值对应的索引
    const Tensor& self, // 输入张量，进行 kth value 计算
    int64_t k, // 第 k 小值的位置
    int64_t dim_, // 计算 kth value 的维度
    bool keepdim) { // 是否保持维度

  // 包装维度索引，确保在输入张量维度范围内
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  // 如果输入张量是标量，则 slicesize 为 1；否则为指定维度的大小
  int64_t slicesize = self.dim() == 0 ? 1 : self.size(dim);

  // 检查 k 值是否在指定维度的范围内
  TORCH_CHECK(k >= 1 && k <= slicesize,
              "kthvalue(): selected number k out of range for dimension ", dim);

  // 检查输入张量与输出张量不重叠
  at::assert_no_overlap(self, values);

  // 分配或调整输出张量的大小
  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim_, keepdim);

  // 如果输入张量是标量且只含有一个元素，则直接复制到 values 张量，indices 张量置零后返回
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }

  // 克隆连续内存格式的输入张量为临时张量 tmp_values
  auto tmp_values = self.clone(at::MemoryFormat::Contiguous);

  // 创建空的 tmp_indices 张量，其形状与输入张量相同，数据类型为 kLong
  auto tmp_indices = at::empty(self.sizes(), self.options().dtype(kLong));

  // 获取 tmp_values 和 tmp_indices 在指定维度上的步长
  auto tmp_values_stride = tmp_values.strides()[dim];
  auto tmp_indices_stride = tmp_indices.strides()[dim];

  // 获取输入张量的形状
  auto sizes = self.sizes();

  // 检查 indices 张量的数据类型必须为 kLong
  TORCH_CHECK(indices.scalar_type() == kLong);

  // 配置张量迭代器，为计算 kth value 和索引做准备
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(sizes, /*squash_dims=*/dim)
    .add_output(tmp_values)
    .add_output(tmp_indices)
    .add_output(values)
    .add_output(indices)
    .build();

  // 根据输入张量的数据类型执行特化的 kthvalue_cpu 函数
  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "kthvalue_cpu", [&] {
    // 定义一个 lambda 函数 loop，用于迭代处理传入的数据数组
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      // 遍历 n 次，处理每个数据点
      for (const auto i : c10::irange(n)) {
        // 创建一个 TensorAccessor 对象 tmp_values，用于访问当前数据点的值
        TensorAccessor<scalar_t, 1> tmp_values(
            reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
            &sizes[dim], &tmp_values_stride);
        // 创建一个 TensorAccessor 对象 tmp_indices，用于访问当前数据点的索引
        TensorAccessor<int64_t, 1> tmp_indices(
            reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
            &sizes[dim], &tmp_indices_stride);
        // 获取当前数据点的模式值和模式索引的指针
        auto mode_value = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        auto mode_index = reinterpret_cast<int64_t*>(data[3] + i * strides[3]);

        // 将 tmp_indices 数组初始化为从 0 开始的递增序列
        for (const auto j : c10::irange(tmp_indices.size(0))) {
          tmp_indices[j] = j;
        }

        // 使用快速选择算法对 tmp_values 和 tmp_indices 进行排序
        // 要求 NaN（非数值）在排序时位于顶部，以保持与 numpy 的兼容性
        quick_select_template(
          tmp_values,
          k - 1,
          // 比较函数，确保 NaN 在排序时被视为最大值
          [](scalar_t x, scalar_t y) -> bool {
            return (
              (_isnan<scalar_t>(x) && !_isnan<scalar_t>(y)) || (x > y));
          },
          // 交换函数，用于交换两个元素在 tmp_values 和 tmp_indices 中的位置
          [&](int64_t i, int64_t j) {
            std::swap(tmp_values[i], tmp_values[j]);
            std::swap(tmp_indices[i], tmp_indices[j]);
          });
        
        // 将第 k-1 个排序后的值赋给 mode_value
        *mode_value = tmp_values[k - 1];
        // 将第 k-1 个排序后的索引赋给 mode_index
        *mode_index = tmp_indices[k - 1];
      }
    };

    // 计算每个数据点的处理粒度 grain_size，确保不小于内部定义的 GRAIN_SIZE 除以 sizes[dim] 和 1 的较大值
    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    // 使用迭代器 iter 对每个数据点应用 loop 函数
    iter.for_each(loop, /*grain_size=*/grain_size);
  });

  // 如果 keepdim 为 false，对 values 和 indices 在指定维度 dim 上进行压缩（去除大小为 1 的维度）
  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
  // 返回 values 和 indices 的 tuple，作为结果
  return std::forward_as_tuple(values, indices);
}

// Computes both the median and its index along dimension dim of the input
// 计算输入张量沿指定维度 dim 的中位数及其索引
std::tuple<Tensor&, Tensor&> median_with_indices_impl(
    // 输出的值和索引张量的引用
    Tensor& values,
    Tensor& indices,
    // 输入张量的常量引用
    const Tensor& self,
    // 操作的维度
    int64_t dim,
    // 是否保持维度
    bool keepdim,
    // 是否忽略 NaN 值
    bool ignore_nan) {
  // 确保维度值在有效范围内
  dim = at::maybe_wrap_dim(dim, self.dim());

  // 如果输入张量的维度大于 0，则获取指定维度的大小
  int64_t size = self.dim() > 0 ? self.size(dim) : 1;
  // 检查维度的有效性
  zero_numel_check_dims(self, dim, "median()");

  // 检查输出张量的设备类型是否与输入张量一致
  checkDeviceType("median", {values, indices}, self.device().type());
  // 检查索引张量的标量类型是否为长整型
  checkScalarType("median", {indices, "indices", 1}, kLong);
  // 检查值张量的标量类型是否与输入张量一致
  checkSameType("median", {values, "values", 0}, {self, "self", 2});

  // 根据输入张量的形状调整输出张量的形状
  std::vector<int64_t> out_shape = self.sizes().vec();
  if (self.dim() > 0) {
    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  // 调整输出张量的大小
  resize_output(values, out_shape);
  resize_output(indices, out_shape);

  // 确保所有需要的张量维度一致
  Tensor in = self.dim() > 0 ? self : self.unsqueeze(0);
  Tensor vals = keepdim && self.dim() > 0 ? values : values.unsqueeze(dim);
  Tensor inds = keepdim && self.dim() > 0 ? indices : indices.unsqueeze(dim);

  // 如果指定维度的步长大于 1，则需要调整张量以便进行操作
  if (in.stride(dim) > 1) {
    in = in.unsqueeze(-1).transpose_(dim, -1).squeeze_(dim).contiguous();
    vals = vals.unsqueeze(-1).transpose_(dim, -1).squeeze_(dim);
    inds = inds.unsqueeze(-1).transpose_(dim, -1).squeeze_(dim);
    dim = in.dim() - 1;
  }

  // 配置张量迭代器以执行操作
  auto sizes = in.sizes();
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(sizes, /*squash_dims=*/dim)
    .add_output(vals)
    .add_output(inds)
    .add_const_input(in)
    .build();

  // 根据输入张量的标量类型选择相应的操作
  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, in.scalar_type(), "median_out", [&] {
    // 定义 lambda 函数 loop，用于并行迭代处理输入数据的每个元素
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      // 遍历范围 [0, n)
      for (const auto i : c10::irange(n)) {
        // 计算当前元素在各个数据数组中的指针位置
        auto valp = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto indp = reinterpret_cast<int64_t*>(data[1] + i * strides[1]);
        auto ip = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);

        // 对于 torch.median，搜索是否存在 NaN 值，如果存在则返回
        if (!ignore_nan) {
          // 查找第一个 NaN 值的位置
          const scalar_t* nanp = std::find_if(ip, ip + size, _isnan<scalar_t>);
          if (nanp != ip + size) {
            // 将找到的 NaN 值赋给输出的值和索引，并继续处理下一个元素
            *valp = *nanp;
            *indp = nanp - ip;
            continue;
          }
        }

        // 创建一个存储索引的向量，用于在中值周围间接分区输入
        std::vector<int64_t> idx(size);
        auto first = idx.begin();
        auto last = idx.end();
        // 填充索引向量，从 0 开始递增
        std::iota(first, last, 0);

        // 使用索引向量在不修改原始输入张量的情况下间接围绕中值分区输入
        auto nth = first;
        if (!ignore_nan) {
          // 如果没有 NaN 值，找到中值的索引
          nth += (size - 1) / 2;
          // 根据 ip 中的值进行分区，以找到中值的索引
          std::nth_element(first, nth, last, [&ip](int64_t i, int64_t j) {
            return ip[i] < ip[j] || (ip[i] == ip[j] && i < j);
          });
        } else {
          // 对于 torch.nanmedian，仅计算非 NaN 值的中值
          int64_t num_nan = std::count_if(ip, ip + size, _isnan<scalar_t>);
          nth += (size - num_nan - 1) / 2;
          // 根据 ip 中的值进行分区，找到非 NaN 值的中值索引
          std::nth_element(first, nth, last, [&ip](int64_t i, int64_t j) {
            return ip[i] < ip[j] || (ip[i] == ip[j] && i < j) ||
                (_isnan(ip[j]) && !_isnan(ip[i]));
          });
        }

        // 将找到的中值赋给输出的值和索引
        *valp = ip[*nth];
        *indp = *nth;
      }
    };
    // 计算并行操作的粒度，为内部常量 GRAIN_SIZE 除以 sizes[dim] 和 1 中的较大者
    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    // 使用并行迭代器 iter，对每个元素应用 loop 函数，使用给定的粒度进行并行操作
    iter.for_each(loop, /*grain_size=*/grain_size);
  });

  // 返回包含 values 和 indices 的元组
  return std::forward_as_tuple(values, indices);
// 计算输入张量中所有值的中位数
Tensor median_impl(const Tensor& self, bool ignore_nan) {
    // 禁止名称保护，确保没有命名冲突
    NoNamesGuard guard;
    // 获取张量中元素的总数
    const int64_t size = self.numel();

    // 对于空张量，返回 NaN
    if (size <= 0) {
        return at::full({}, std::numeric_limits<float>::quiet_NaN()).to(self.options());
    }

    // 克隆输入张量以便围绕中位数对其进行分区
    Tensor in = self.clone();
    // 创建一个空张量用于存储输出
    Tensor out = at::empty({}, self.options());

    // 根据张量元素类型分发执行具体的中位数计算函数
    AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, in.scalar_type(), "median_cpu", [&] {
        // 获取输出张量的数据指针
        scalar_t* op = out.data_ptr<scalar_t>();
        // 获取输入张量的数据指针和最后一个元素的下一个位置的指针
        scalar_t* first = in.data_ptr<scalar_t>();
        scalar_t* last = first + size;

        // 对于 torch.median，如果存在 NaN 值，则返回 NaN
        if (!ignore_nan && std::any_of(first, last, _isnan<scalar_t>)) {
            *op = std::numeric_limits<scalar_t>::quiet_NaN();
            return;
        }

        // 定义指向中位数的指针，并根据是否忽略 NaN 值来选择计算方式
        scalar_t* median = first;
        if (!ignore_nan) {
            // 如果没有 NaN 值，则选择此分支计算中位数
            median += (size - 1) / 2;
            std::nth_element(first, median, last);
        } else {
            // 对于 torch.nanmedian，仅计算非 NaN 值的中位数
            int64_t num_nan = std::count_if(first, last, _isnan<scalar_t>);
            median += (size - num_nan - 1) / 2;
            std::nth_element(first, median, last, [](scalar_t a, scalar_t b) {
                return a < b || (_isnan(b) && !_isnan(a));
            });
        }

        // 将计算得到的中位数存储到输出张量中
        *op = *median;
    });

    // 返回输出张量
    return out;
}

// namespace 结束
} // namespace

// 用于计算 quantile 的输出版本，将结果存储在指定的输出张量中
Tensor& quantile_out(
    const Tensor& self,
    const Tensor& q,
    optional<int64_t> dim,
    bool keepdim,
    const c10::string_view interpolation,
    Tensor& out) {
  // 调用 quantile_out_impl 函数进行实际计算
  quantile_out_impl(
      out,
      self,
      q,
      dim,
      keepdim,
      get_quantile_interpolation_mode(interpolation),
      /*ignore_nan=*/false);
  // 返回输出张量的引用
  return out;
}

// 用于计算 quantile 的输出版本，将结果存储在指定的输出张量中
Tensor& quantile_out(
    const Tensor& self,
    double q,
    optional<int64_t> dim,
    bool keepdim,
    const c10::string_view interpolation,
    Tensor& out) {
  // 检查 q 是否在合理范围内
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
  // 调用 native::quantile_out 函数进行实际计算
  return at::native::quantile_out(
      self,
      at::scalar_tensor(q, self.options()),
      dim,
      keepdim,
      interpolation,
      out);
}

// 计算张量的分位数
Tensor quantile(
    const Tensor& self,
    const Tensor& q,
    optional<int64_t> dim,
    bool keepdim,
    const c10::string_view interpolation) {
  // 调用 quantile_impl 函数进行实际计算
  return quantile_impl(
      self,
      q,
      dim,
      keepdim,
      get_quantile_interpolation_mode(interpolation),
      /*ignore_nan=*/false);
}

// 计算张量的分位数
Tensor quantile(
    const Tensor& self,
    double q,
    optional<int64_t> dim,
    bool keepdim,
    const c10::string_view interpolation) {
  // 检查 q 是否在合理范围内
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
  // 调用 native::quantile 函数进行实际计算
  return at::native::quantile(
      self, at::scalar_tensor(q, self.options()), dim, keepdim, interpolation);
}

// 计算张量中的 NaN 值的分位数
Tensor& nanquantile_out(
    const Tensor& self,
    const Tensor& q,
    // 调用量化分位数计算的实现函数quantile_out_impl，并将计算结果存储在输出张量out中
    optional<int64_t> dim,
    // 指定在哪个维度上进行分位数计算
    bool keepdim,
    // 指定是否保持输出张量的维度
    const c10::string_view interpolation,
    // 插值方法，以字符串视图的形式传入
    Tensor& out) {
  // 调用量化分位数计算的实现函数quantile_out_impl，传入参数self（输入张量）、q（分位数）、dim（维度）、keepdim（是否保持维度）、interpolation（插值方法）、ignore_nan（是否忽略NaN值）
  quantile_out_impl(
      out,
      self,
      q,
      dim,
      keepdim,
      get_quantile_interpolation_mode(interpolation),
      /*ignore_nan=*/true);
  // 返回计算后的输出张量out
  return out;
}
}

Tensor& nanquantile_out(
    const Tensor& self,  // 输入张量
    double q,            // 分位数
    optional<int64_t> dim,  // 沿着哪个维度计算
    bool keepdim,        // 是否保持输出张量的维度
    const c10::string_view interpolation,  // 插值方式
    Tensor& out) {       // 输出张量的引用
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);  // 检查分位数 q 是否在 [0, 1] 范围内
  return at::native::nanquantile_out(
      self,
      at::scalar_tensor(q, self.options()),  // 创建一个标量张量表示分位数 q
      dim,
      keepdim,
      interpolation,
      out);  // 调用原生的 nanquantile_out 函数
}

Tensor nanquantile(
    const Tensor& self,  // 输入张量
    const Tensor& q,     // 分位数张量
    optional<int64_t> dim,  // 沿着哪个维度计算
    bool keepdim,
    const c10::string_view interpolation) {  // 插值方式
  return quantile_impl(
      self,
      q,
      dim,
      keepdim,
      get_quantile_interpolation_mode(interpolation),  // 获取插值模式
      /*ignore_nan=*/true);  // 忽略 NaN 值
}

Tensor nanquantile(
    const Tensor& self,  // 输入张量
    double q,            // 分位数
    optional<int64_t> dim,  // 沿着哪个维度计算
    bool keepdim,
    const c10::string_view interpolation) {  // 插值方式
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);  // 检查分位数 q 是否在 [0, 1] 范围内
  return at::native::nanquantile(
      self, at::scalar_tensor(q, self.options()),  // 创建一个标量张量表示分位数 q
      dim, keepdim, interpolation);  // 调用原生的 nanquantile 函数
}

std::tuple<Tensor&, Tensor&> kthvalue_out_cpu(
    const Tensor& self,   // 输入张量
    int64_t k,            // 第 k 小的值
    int64_t dim,          // 沿着哪个维度计算
    bool keepdim,
    Tensor& values,       // 存储第 k 小值的张量
    Tensor& indices) {    // 存储第 k 小值的索引的张量
  auto result = [&]() {
    NoNamesGuard guard;  // 临时禁用命名推断
    return kthvalue_out_impl_cpu(values, indices, self, k, dim, keepdim);  // 调用 CPU 上的 kthvalue_out_impl_cpu 函数
  }();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);  // 传播约简操作的命名
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);  // 传播约简操作的命名
  return result;  // 返回计算结果
}

std::tuple<Tensor&, Tensor&> kthvalue_out(
    const Tensor& self,   // 输入张量
    int64_t k,            // 第 k 小的值
    Dimname dim,          // 沿着哪个维度计算的维度名
    bool keepdim,
    Tensor& values,       // 存储第 k 小值的张量
    Tensor& indices) {    // 存储第 k 小值的索引的张量
  return at::kthvalue_out(
      values, indices, self, k, dimname_to_position(self, dim), keepdim);  // 调用 kthvalue_out 函数，将维度名转换为位置
}

std::tuple<Tensor, Tensor> kthvalue(
    const Tensor& self,   // 输入张量
    int64_t k,            // 第 k 小的值
    int64_t dim,          // 沿着哪个维度计算
    bool keepdim) {
  Tensor values = at::empty({0}, self.options());  // 创建一个空张量用于存储第 k 小值
  Tensor indices = at::empty({0}, self.options().dtype(kLong));  // 创建一个空张量用于存储第 k 小值的索引
  at::kthvalue_out(values, indices, self, k, dim, keepdim);  // 调用 kthvalue_out 函数计算第 k 小值及其索引
  return std::make_tuple(values, indices);  // 返回第 k 小值张量及其索引张量的元组
}

std::tuple<Tensor, Tensor> kthvalue(
    const Tensor& self,   // 输入张量
    int64_t k,            // 第 k 小的值
    Dimname dim,          // 沿着哪个维度计算的维度名
    bool keepdim) {
  return at::kthvalue(self, k, dimname_to_position(self, dim), keepdim);  // 调用 kthvalue 函数，将维度名转换为位置
}

TORCH_IMPL_FUNC(topk_out_cpu)
   (const Tensor& self,   // 输入张量
    int64_t k,            // 返回的最大或最小的 k 个值
    int64_t dim_,         // 沿着哪个维度计算
    bool largest,         // 是否返回最大的 k 个值
    bool sorted,          // 是否返回排序后的结果
    const Tensor& values, // 存储最大或最小的 k 个值的张量
    const Tensor& indices) {  // 存储最大或最小的 k 个值的索引的张量
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);  // 根据输入张量的维度数，可能包装维度
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),  // 检查 k 是否在合理范围内
      "selected index k out of range");

  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);  // 如果输入张量是标量，直接将值复制到 values 张量中
    indices.zero_();     // 索引张量清零
  } else {
    topk_stub(kCPU, values, indices, self, k, dim, largest, sorted);  // 调用 topk_stub 函数计算最大或最小的 k 个值及其索引
  }
}

std::tuple<Tensor&, Tensor&> median_out_cpu(
    const Tensor& self,   // 输入张量
    int64_t dim,          // 沿着哪个维度计算
    bool keepdim,
    # 定义一个函数，接受两个引用参数：values 和 indices
    Tensor& values,
    Tensor& indices) {
      # 使用 lambda 表达式定义一个闭包，执行以下操作
      auto result = [&]() {
        # 创建一个 NoNamesGuard 对象，确保在闭包执行期间不会记录操作名称
        NoNamesGuard guard;
        # 调用 median_with_indices_impl 函数，计算在给定的 self 张量上，
        # 沿着指定的 dim 维度进行中位数计算，同时保留维度信息，不忽略 NaN 值
        return median_with_indices_impl(
            values, indices, self, dim, keepdim, /*ignore_nan=*/false);
      }();
      # 根据 values 张量的操作结果，传播与 reduction 相关的操作名称
      namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
      # 根据 indices 张量的操作结果，传播与 reduction 相关的操作名称
      namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
      # 返回闭包中计算得到的结果
      return result;
    }
}

// 通过引用返回中位数计算结果
std::tuple<Tensor&, Tensor&> median_out(
    const Tensor& self,   // 输入张量
    Dimname dim,          // 维度名
    bool keepdim,         // 是否保持维度
    Tensor& values,       // 输出的中位数值
    Tensor& indices) {    // 输出的中位数索引
  // 调用 ATen 中的 median_out 函数，计算中位数，并存储结果到 values 和 indices 中
  return at::median_out(
      values, indices, self, dimname_to_position(self, dim), keepdim);
}

// 计算张量在指定维度上的中位数，返回值和索引
std::tuple<Tensor, Tensor> median(
    const Tensor& self,   // 输入张量
    int64_t dim,          // 维度
    bool keepdim) {       // 是否保持维度
  // 创建空的 values 和 indices 张量
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  // 调用 median_out 函数填充 values 和 indices，并返回它们的元组
  at::median_out(values, indices, self, dim, keepdim);
  return std::make_tuple(values, indices);
}

// 计算张量在指定维度上的中位数，使用 Dimname
std::tuple<Tensor, Tensor> median(
    const Tensor& self,   // 输入张量
    Dimname dim,          // 维度名
    bool keepdim) {       // 是否保持维度
  // 调用 ATen 中的 median 函数，通过 dimname_to_position 转换维度名为位置
  return at::median(self, dimname_to_position(self, dim), keepdim);
}

// 在 CPU 上计算张量的中位数，忽略 NaN 值
Tensor median_cpu(const Tensor& self) {
  // 调用 median_impl 函数，设置 ignore_nan 为 false
  return median_impl(self, /*ignore_nan=*/false);
}

// 通过引用返回带 NaN 处理的中位数计算结果
std::tuple<Tensor&, Tensor&> nanmedian_out_cpu(
    const Tensor& self,   // 输入张量
    int64_t dim,          // 维度
    bool keepdim,         // 是否保持维度
    Tensor& values,       // 输出的中位数值
    Tensor& indices) {    // 输出的中位数索引
  // 使用 lambda 表达式调用 median_with_indices_impl 函数，忽略 NaN 值
  auto result = [&]() {
    NoNamesGuard guard;   // 临时禁用命名
    return median_with_indices_impl(
        values, indices, self, dim, keepdim, /*ignore_nan=*/true);
  }();
  // 根据降维操作传播输出张量的命名信息
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  // 根据降维操作传播输出索引张量的命名信息
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
  return result;          // 返回中位数值和索引
}

// 通过引用返回带 NaN 处理的中位数计算结果，使用 Dimname
std::tuple<Tensor&, Tensor&> nanmedian_out(
    const Tensor& self,   // 输入张量
    Dimname dim,          // 维度名
    bool keepdim,         // 是否保持维度
    Tensor& values,       // 输出的中位数值
    Tensor& indices) {    // 输出的中位数索引
  // 调用 ATen 中的 nanmedian_out 函数，通过 dimname_to_position 转换维度名为位置
  return at::nanmedian_out(
      values, indices, self, dimname_to_position(self, dim), keepdim);
}

// 计算带 NaN 处理的张量在指定维度上的中位数，返回值和索引
std::tuple<Tensor, Tensor> nanmedian(
    const Tensor& self,   // 输入张量
    int64_t dim,          // 维度
    bool keepdim) {       // 是否保持维度
  // 创建空的 values 和 indices 张量
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  // 调用 nanmedian_out 函数填充 values 和 indices，并返回它们的元组
  at::nanmedian_out(values, indices, self, dim, keepdim);
  return std::make_tuple(values, indices);
}

// 计算带 NaN 处理的张量在指定维度上的中位数，使用 Dimname
std::tuple<Tensor, Tensor> nanmedian(
    const Tensor& self,   // 输入张量
    Dimname dim,          // 维度名
    bool keepdim) {       // 是否保持维度
  // 调用 ATen 中的 nanmedian 函数，通过 dimname_to_position 转换维度名为位置
  return at::nanmedian(self, dimname_to_position(self, dim), keepdim);
}

// 在 CPU 上计算带 NaN 处理的张量的中位数
Tensor nanmedian_cpu(const Tensor& self) {
  // 调用 median_impl 函数，设置 ignore_nan 为 true
  return median_impl(self, /*ignore_nan=*/true);
}

// 实现稳定排序的函数，输出结果到 values 和 indices
TORCH_IMPL_FUNC(sort_stable_out)
(const Tensor& self,         // 输入张量
 std::optional<bool> stable, // 是否稳定排序的选项
 int64_t dim,                // 排序的维度
 bool descending,            // 是否降序排序
 const Tensor& values,       // 排序后的值
 const Tensor& indices) {    // 排序后的索引
  values.copy_(self);        // 将 self 的内容复制到 values
  // 检查 self 是否是标量
  if (self.dim() == 0 && self.numel() == 1) {
    indices.zero_();         // 如果是标量，将 indices 清零
  } else {
    dim = maybe_wrap_dim(dim, self.dim());  // 处理维度的包装
    // 调用 sort_stub 函数执行排序操作
    sort_stub(self.device().type(), self, values, indices, dim, descending, stable.value_or(false));
  }
}

// 通过引用返回排序结果的函数
std::tuple<Tensor&, Tensor&> sort_out(
    const Tensor& self,    // 输入张量
    int64_t dim,           // 排序的维度
    bool descending,       // 是否降序排序
    Tensor& values,        // 输出的排序后的值
    Tensor& indices) {     // 输出的排序后的索引
  // 调用 ATen 中的 sort_out 函数，执行排序操作
  return at::sort_out(values, indices, self, false, dim, descending);
}

// 计算张量在指定维度上的排序结果，返回排序后的值和索引
std::tuple<Tensor, Tensor> sort(
    const Tensor& self,    // 输入张量
    int64_t dim,           // 排序的维度
    bool descending) {     // 是否降序排序
  // 调用 ATen 中的 sort 函数，执行排序操作
  return at::sort(self, false, dim, descending);
}
// 对输入张量 self 进行排序，结果存入 values 中，并返回 values 的引用
Tensor& msort_out(const Tensor& self, Tensor& values) {
  // 创建一个空的长整型张量 indices，用于存储排序后的索引
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  // 调用 PyTorch 的排序函数，将 self 按照 dim=0 升序排序，将结果存入 values 和 indices
  at::sort_out(values, indices, self, 0, false);
  // 返回排序后的 values 引用
  return values;
}

// 对输入张量 self 进行排序，返回排序后的结果
Tensor msort(const Tensor& self) {
  // 调用 PyTorch 的排序函数，按照 dim=0 升序排序 self，并返回排序后的第一个返回值（排序后的张量）
  return std::get<0>(at::sort(self, 0, false));
}

// 对输入张量 self 按照指定维度 dim 进行排序，返回排序后的索引
Tensor argsort(const Tensor& self, int64_t dim, bool descending) {
  // 调用 PyTorch 的排序函数，按照指定维度 dim 和排序顺序 descending 对 self 进行排序，
  // 返回排序后的第二个返回值（排序后的索引）
  return std::get<1>(at::sort(self, dim, descending));
}

// 对输入张量 self 按照指定维度 dim 进行排序，返回排序后的索引，支持稳定排序和降序选项
Tensor argsort(const Tensor& self, bool stable, int64_t dim, bool descending) {
  // 调用 PyTorch 的排序函数，按照稳定性 stable、指定维度 dim 和排序顺序 descending 对 self 进行排序，
  // 返回排序后的第二个返回值（排序后的索引）
  return std::get<1>(at::sort(self, stable, dim, descending));
}

// 对输入张量 self 按照指定维度 dim 进行排序，并将排序结果输出到指定的张量 out 中，支持稳定排序和降序选项
Tensor& argsort_out(const Tensor& self, bool stable, int64_t dim, bool descending, Tensor& out) {
  // 创建一个空的与 self 类型相同的张量 values，用于存储排序后的值
  auto values = at::empty({0}, self.options());
  // 调用 PyTorch 的排序输出函数，将 self 按照稳定性 stable、指定维度 dim 和排序顺序 descending 排序，
  // 将结果存入 values 和 out 中
  at::sort_outf(self, stable, dim, descending, values, out);
  // 返回排序后的 out 引用
  return out;
}

// 命名空间结束符
} // namespace at::native
```