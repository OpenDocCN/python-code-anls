# `.\pytorch\aten\src\ATen\native\Bucketization.cpp`

```
// 定义编译选项，仅允许方法操作符使用
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含张量操作的头文件
#include <ATen/core/Tensor.h>
// 包含分发机制的头文件
#include <ATen/Dispatch.h>
// 包含并行处理的头文件
#include <ATen/Parallel.h>
// 包含桶化工具函数的头文件
#include <ATen/native/BucketizationUtils.h>
// 包含调整大小函数的头文件
#include <ATen/native/Resize.h>
// 包含范围迭代器的头文件
#include <c10/util/irange.h>

// 如果未定义每个操作符的头文件，则包含整体操作函数的头文件，否则包含各自的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/bucketize_native.h>
#include <ATen/ops/searchsorted_native.h>
#endif

/* 实现类似 NumPy 的 searchsorted 和类似 TensorFlow 的 bucketize 函数在 CPU 上运行
 *
 * - torch.searchsorted(sorted_sequence, values, right=False, side=None, out_int32=False, sorter=None)
 *   sorted_sequence - N*D 或 1D（适用于所有值）张量，包含最后一个维度中的排序序列
 *   values          - N*D 张量或标量（当 sorted_sequence 是 1D 时）包含搜索值
 *   right           - 如果为 False 对应下界，为 True 对应上界
 *   side            - （推荐使用 right）如果为 'left' 对应下界，为 'right' 对应上界
 *   out_int32       - 如果为 False 输出张量类型为 int64_t，如果为 True 输出张量类型为 int（通常是 32 位）
 *   sorter          - 如果提供，则 sorted_sequence 可能未排序，并且排序顺序由此张量给出
 *
 * - torch.bucketize(values, boundaries, right=False, out_int32=False)
 *   values     - N*D 张量或标量，包含搜索值
 *   boundaries - 1D 张量，包含排序序列
 *   right      - 如果为 False 对应下界，为 True 对应上界
 *   out_int32  - 如果为 False 输出张量类型为 int64_t，如果为 True 输出张量类型为 int（通常是 32 位）
 *
 * - 约束条件在 searchsorted_pre_check() 中定义
 */

namespace at::native {

namespace {

// 用于确保 searchsorted_cpu_contiguous 函数可以并行运行的最小大小（粒度）
constexpr int64_t SEARCHSORTED_GRAIN_SIZE = 200;

// 自定义的 lower_bound 函数，确保 'nan'、'inf' 等的下界是边界的末端，
// 并且可以正确处理 sorter 参数
// 这里不能使用 std::lower_bound，因为它的自定义比较器需要严格弱排序，
// 而自定义比较器要求两个参数具有相同的类型，在比较输入值 input_t 和 sorter 中的索引值时会出问题
template<typename input_t>
int64_t cus_lower_bound(int64_t start, int64_t end, const input_t val, const input_t* bd, const int64_t* sort) {
  // sorter 给出了 ND 张量的相对顺序，因此我们需要保存并将未更新的 start 作为偏移量添加
  // 例如，3x3 张量的第二行从元素 3 开始，但是 sorter 的第二行仅包含 0、1 或 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);  // 中间值的计算
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];  // 根据 sort 参数选择性读取 bd 的值
    if (!(mid_val >= val)) {
      start = mid + 1;  // 如果 mid_val 不大于 val，则更新 start
    }
    else {
      end = mid;  // 否则更新 end
    }
  }
  return start;  // 返回找到的下界位置
}
// 自定义的上界查找函数，用于确保可以正确处理排序器参数
// 这里无法使用 std::upper_bound，因为其自定义比较器要求两个参数具有相同类型，
// 而在比较输入类型 input_t 的 val 和排序器中 int64 类型的索引值时，类型不匹配
template<typename input_t>
int64_t cus_upper_bound(int64_t start, int64_t end, const input_t val, const input_t* bd, const int64_t* sort) {
  // sorter 提供了 ND 张量的相对排序，因此我们需要保存并将未更新的起始位置作为偏移量添加
  // 例如，3x3 张量的第二行从元素3开始，但是排序器的第二行只包含0、1或2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
void searchsorted_cpu_contiguous(Tensor& result, const Tensor& input, const Tensor& boundaries, const bool& right, const Tensor& sorter) {
  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // 输入和边界张量的最内层维度大小
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t *data_in = input.const_data_ptr<input_t>();
  const input_t *data_bd = boundaries.const_data_ptr<input_t>();
  const int64_t *data_st = sorter.defined() ? sorter.const_data_ptr<int64_t>() : nullptr;
  output_t *data_out = result.data_ptr<output_t>();

  bool is_1d_boundaries = boundaries.dim() == 1;
  at::parallel_for(0, numel_in, SEARCHSORTED_GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      // 如果边界张量是一维的，则始终搜索整个边界张量
      int64_t start_bd = is_1d_boundaries ? 0 : i / idim_in * idim_bd;
      int64_t end_bd = start_bd + idim_bd;

      int64_t pos = !right ?
        cus_lower_bound(start_bd, end_bd, data_in[i], data_bd, data_st) - start_bd :
        cus_upper_bound(start_bd, end_bd, data_in[i], data_bd, data_st) - start_bd;

      // 可能在这里进行类型转换
      data_out[i] = pos;
    }
  });
}

void dispatch(Tensor& result, const Tensor& input, const Tensor& boundaries, bool out_int32, bool right, const Tensor& sorter) {
  if (!out_int32) {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        input.scalar_type(),
        "searchsorted_out_cpu",
        [&] {
          searchsorted_cpu_contiguous<scalar_t, int64_t>(
              result, input, boundaries, right, sorter);
        });
  }
  else {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,                              # 调度并处理所有数据类型和 Half 类型以及 BFloat16 类型
        ScalarType::BFloat16,
        input.scalar_type(),                          # 获取输入张量的数据类型
        "searchsorted_out_cpu",                       # 标识 CPU 端的 searchsorted_out 函数
        [&] {                                          # Lambda 函数开始
          searchsorted_cpu_contiguous<scalar_t, int>(  # 调用 searchsorted_cpu_contiguous 函数，处理连续的输入张量和整数边界
              result, input, boundaries, right, sorter);
        });
  }
}

}

# 在 CPU 上执行搜索排序（in-place 操作），将结果存储在 result 中
Tensor& searchsorted_out_cpu(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const std::optional<c10::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt,
    Tensor& result) {
  # 处理可选张量的包装器，确保 sorter_opt 不为空时得到有效的 sorter 引用
  c10::MaybeOwned<Tensor> sorter_maybe_owned = at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  # 执行预检查，确保参数合法性和准备工作
  searchsorted_pre_check(sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  # 调整输出 result 的大小以匹配输入 self 的大小
  resize_output(result, self.sizes());

  # 确定是否使用 right 参数的值，若 side_opt 存在，则使用其值；否则使用 right 参数
  bool is_right = side_opt ? *side_opt == "right" : right;

  # 如果输入 self 的元素数量为 0，则直接返回 result
  if (self.numel() == 0) {
    return result;
  }

  # 将 out 初始化为 result，如果 result 不是连续的，则将 out 设置为 result 的连续版本
  Tensor out = result;
  if (!result.is_contiguous()) {
    out = result.contiguous();
  }

  # 如果 sorted_sequence 和 self 都是连续的，并且数据类型相同，并且 sorter 是连续的，则分发调度函数
  if (sorted_sequence.is_contiguous() && self.is_contiguous() && sorted_sequence.dtype() == self.dtype() && sorter.is_contiguous()) {
    dispatch(out, self, sorted_sequence, out_int32, is_right, sorter);
  }
  else {
    # 如果有非连续的输入张量，则对其进行裁剪，以准备传递给调度函数的最终输入
    Tensor trimmed_input;
    Tensor trimmed_boundaries;
    Tensor trimmed_sorter;
    searchsorted_maybe_trim_input_tensors(trimmed_input, trimmed_boundaries, trimmed_sorter, self, sorted_sequence, sorter);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_boundaries = trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
    const Tensor& final_sorter = trimmed_sorter.defined() ? trimmed_sorter : sorter;
    dispatch(out, final_input, final_boundaries, out_int32, is_right, final_sorter);
  }

  # 如果 result 是非连续的，则将结果从复制的版本复制回原始的 result 张量
  if (!result.is_contiguous()) {
    result.copy_(out);
  }
  # 返回结果张量 result
  return result;
}

# 在 CPU 上执行搜索排序（in-place 操作）的重载函数，处理标量 self 的情况
Tensor& searchsorted_out_cpu(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const std::optional<c10::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt,
    Tensor& result) {
  # 将标量 self 转换为张量 scalar_tensor
  const Tensor& scalar_tensor = searchsorted_scalar_tensor(self, sorted_sequence.device());
  # 调用前面定义的函数处理张量版本的 searchsorted 操作
  return searchsorted_out_cpu(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter_opt, result);
}
// 在 CPU 上执行二分搜索，用于在已排序的序列中搜索元素的位置
Tensor searchsorted_cpu(
      const Tensor& sorted_sequence,                 // 排好序的序列
      const Tensor& self,                           // 待搜索的张量
      bool out_int32,                              // 是否输出结果为 Int32 类型
      bool right,                                  // 是否搜索右侧的位置
      const std::optional<c10::string_view> side_opt,  // 可选参数，指定搜索的方向
      const std::optional<Tensor>& sorter_opt) {    // 可选参数，用于排序的张量
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;  // 根据 out_int32 确定结果的数据类型
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);  // 创建张量的选项
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);  // 创建一个空的张量作为结果
  at::native::searchsorted_out_cpu(sorted_sequence, self, out_int32, right, side_opt, sorter_opt, result);  // 调用底层搜索函数
  return result;  // 返回搜索结果张量
}

// 在 CPU 上执行标量的 bucketize 操作，将标量 self 放入 boundaries 中对应的桶中
Tensor searchsorted_cpu(
    const Tensor& sorted_sequence,                    // 排好序的序列
    const Scalar& self,                              // 待搜索的标量
    bool out_int32,                                 // 是否输出结果为 Int32 类型
    bool right,                                     // 是否搜索右侧的位置
    const std::optional<c10::string_view> side_opt,  // 可选参数，指定搜索的方向
    const std::optional<Tensor>& sorter_opt) {       // 可选参数，用于排序的张量
  const Tensor& scalar_tensor = searchsorted_scalar_tensor(self, sorted_sequence.device());  // 将标量 self 转换为张量
  return searchsorted_cpu(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter_opt);  // 调用前一个函数进行搜索
}

// 在 CPU 上执行 bucketize 操作，将 self 放入 boundaries 中对应的桶中，结果保存在 result 中
Tensor& bucketize_out_cpu(const Tensor& self,             // 待处理的张量
                          const Tensor& boundaries,       // 边界张量
                          bool out_int32,                // 是否输出结果为 Int32 类型
                          bool right,                    // 是否将元素放入右侧的桶中
                          Tensor& result) {              // 存放结果的张量引用
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");  // 检查边界张量的维度
  at::native::searchsorted_out_cpu(boundaries, self, out_int32, right, nullopt, nullopt, result);  // 调用底层搜索函数
  return result;  // 返回处理后的结果张量
}

// 在 CPU 上执行 bucketize 操作，将 self 放入 boundaries 中对应的桶中，创建并返回结果张量
Tensor bucketize_cpu(const Tensor& self,             // 待处理的张量
                     const Tensor& boundaries,       // 边界张量
                     bool out_int32,                // 是否输出结果为 Int32 类型
                     bool right) {                  // 是否将元素放入右侧的桶中
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;  // 根据 out_int32 确定结果的数据类型
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);  // 创建张量的选项
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);  // 创建一个空的张量作为结果
  at::native::bucketize_out_cpu(self, boundaries, out_int32, right, result);  // 调用底层 bucketize 函数
  return result;  // 返回处理后的结果张量
}

// 在 CPU 上执行标量的 bucketize 操作，将标量 self 放入 boundaries 中对应的桶中，创建并返回结果张量
Tensor bucketize_cpu(const Scalar& self,             // 待处理的标量
                     const Tensor& boundaries,       // 边界张量
                     bool out_int32,                // 是否输出结果为 Int32 类型
                     bool right) {                  // 是否将元素放入右侧的桶中
  return bucketize_cpu(searchsorted_scalar_tensor(self, boundaries.device()), boundaries, out_int32, right);  // 调用前一个函数进行 bucketize 操作
}
```