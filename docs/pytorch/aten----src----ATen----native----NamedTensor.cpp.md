# `.\pytorch\aten\src\ATen\native\NamedTensor.cpp`

```py
// 定义宏 `TORCH_ASSERT_ONLY_METHOD_OPERATORS`，用于指定仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中的 Tensor 类和 NamedTensorUtils 头文件
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>

// 根据编译器宏选择是否包含特定的操作函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/align_as_native.h>
#include <ATen/ops/align_tensors_native.h>
#include <ATen/ops/align_to_native.h>
#include <ATen/ops/gather_native.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_copy_native.h>
#include <ATen/ops/index_fill.h>
#include <ATen/ops/index_fill_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/refine_names_native.h>
#include <ATen/ops/rename_native.h>
#include <ATen/ops/scatter_add_native.h>
#include <ATen/ops/scatter_native.h>
#include <ATen/ops/sort_native.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/squeeze_native.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

// 包含 C++ 标准库中的 bitset 头文件
#include <bitset>

// ATen 库的命名空间
namespace at::native {

// 函数：将当前张量 `self` 重命名为 `names` 中指定的命名
Tensor& rename_(Tensor& self, optional<DimnameList> names) {
  // 调用 ATen 库中的函数，原地设置张量的命名
  at::internal_set_names_inplace(self, names);
  // 返回更新后的自身张量
  return self;
}

// 函数：返回一个新的张量，其命名由 `self` 张量的 `names` 和 `names` 参数指定
Tensor rename(const Tensor& self, optional<DimnameList> names) {
  // 创建 `self` 的别名张量
  auto result = self.alias();
  // 调用 ATen 库中的函数，设置别名张量的命名
  at::internal_set_names_inplace(result, names);
  // 返回具有新命名的别名张量
  return result;
}

// 静态函数：报告移动未命名维度错误，给出相关的 `names` 和 `other` 参数
static void report_moving_unnamed_dim_error(
    DimnameList names, DimnameList other, bool is_aligning_two_tensors) {
  // 如果正在对齐两个张量，则报告对齐时可能改变未命名维度绝对位置的错误
  if (is_aligning_two_tensors) {
    TORCH_CHECK(false,
        "Aligning Tensor", names, " and Tensor", other,
        " would change the absolute position from the right of an unnamed dimension. ",
        "Please name unnamed dimensions to avoid ambiguity.");
  } else {
    TORCH_CHECK(false,
        "Aligning Tensor", names, " to `names` ", other,
        " would change the absolute position from the right of an unnamed dimension. ",
        "Please name unnamed dimensions to avoid ambiguity.");
  }
}

// 静态函数：报告不是子序列错误，给出相关的 `names` 和 `other` 参数
static void report_not_a_subsequence_error(
    DimnameList names, DimnameList other, bool is_aligning_two_tensors) {
  // 如果正在对齐两个张量，则报告无法将它们对齐为子序列的错误
  if (is_aligning_two_tensors) {
    auto shorter = names.size() > other.size() ? other : names;
    auto longer = names.size() > other.size() ? names : other;
    TORCH_CHECK(false,
        "Could not align Tensor", shorter, " and Tensor", longer,
        " because ", shorter, " is not a subsequence of ", longer, ". ");
  } else {
    TORCH_CHECK(false,
        "Could not align Tensor", names, " to `names` ", other,
        " because ", names, " is not a subsequence of `names`.");
  }
}

// 辅助函数：计算张量 `t` 在对齐到 `aligned_names` 后的结果大小
// 根据 Note [Alignment rules] 强制执行对齐规则
static std::vector<int64_t> aligned_size(
    IntArrayRef tensor_sizes,
    DimnameList tensor_names,
    DimnameList aligned_names,
    // 继续下一个函数签名
    // 创建一个大小与 aligned_names 相同的整数向量，初始值都为 1
    std::vector<int64_t> expanded_sizes(aligned_names.size(), 1);
    // 计算 tensor_sizes 的维度数，并将其转换为 ptrdiff_t 类型
    ptrdiff_t dim = (ptrdiff_t)tensor_sizes.size() - 1;
    // 计算 aligned_names 的元素个数，并将其转换为 ptrdiff_t 类型
    ptrdiff_t idx = (ptrdiff_t)aligned_names.size() - 1;
    // 从右向左遍历 tensor_names 和 aligned_names
    for (; idx >= 0 && dim >= 0; --idx) {
        // 如果 tensor_names[dim] 不等于 aligned_names[idx]，则继续下一轮循环
        if (tensor_names[dim] != aligned_names[idx]) {
            continue;
        }
        // 如果 tensor_names[dim] 是通配符（wildcard），并且 tensor_sizes.size() - dim
        // 不等于 aligned_names.size() - idx，报告移动未命名维度错误
        // 这违反了对齐规则的条件 2
        if (tensor_names[dim].isWildcard() &&
            tensor_sizes.size() - dim != aligned_names.size() - idx) {
            report_moving_unnamed_dim_error(
                tensor_names, aligned_names, /*is_aligning_two_tensors=*/false);
        }
        // 将 expanded_sizes[idx] 设置为 tensor_sizes[dim]，表示对齐维度的大小
        expanded_sizes[idx] = tensor_sizes[dim];
        // 减少 dim，继续向前匹配下一个维度
        --dim;
    }
    // 如果 dim 不等于 -1，即 tensor_names 不是 aligned_names 的子序列，报告不是子序列错误
    report_not_a_subsequence_error(
        tensor_names, aligned_names, /*is_aligning_two_tensors=*/false);
    // 返回存储了扩展大小的 expanded_sizes 向量
    return expanded_sizes;
}

// 定义函数 `refine_names`，用于调整张量的命名维度
Tensor refine_names(const Tensor& self, DimnameList names) {
  // 获取输入张量的当前命名
  const auto self_names = self.names();
  // 检查输入张量和目标命名列表的维度数量是否相同，否则抛出错误
  TORCH_CHECK(self_names.size() == names.size(),
      "refine_names: cannot coerce Tensor", self_names, " to Tensor", names,
      " because they have a different number of dims (",
      self_names.size(), " and ", names.size(), " respectively).");
  // 检查命名列表是否有效
  check_names_valid_for(self, names);

  // 遍历张量的每一个命名维度
  for (const auto idx : c10::irange(self_names.size())) {
    const auto& self_name = self_names[idx];
    const auto& out_name = names[idx];
    // 如果当前命名与目标命名相同或者是通配符，则继续下一次循环
    if (self_name == out_name || self_name.isWildcard()) {
      continue;
    }
    // 如果目标命名是通配符，抛出错误
    if (out_name.isWildcard()) {
      TORCH_CHECK(false,
          "refine_names: cannot coerce Tensor", self_names, " to Tensor", names,
          " because ", self_name, " is more specific than ", out_name, " at index ",
          idx);
    }
    // 如果当前命名与目标命名不同，抛出错误
    TORCH_CHECK(false,
        "refine_names: cannot coerce Tensor", self_names, " to Tensor", names,
        " because ", self_name, " is different from ", out_name, " at index ",
        idx);
    // 如果以上错误检查都未通过，则触发内部断言错误
    TORCH_INTERNAL_ASSERT(false); // done handling errors
  }

  // 创建输入张量的别名，并在别名上设置新的命名
  auto result = self.alias();
  internal_set_names_inplace(result, names);
  // 返回带有新命名的结果张量
  return result;
}

// [Alignment rules]
// 对张量进行命名对齐，以满足以下规则：
// 1) 检查张量的命名是否是 `names` 的子序列（不一定连续）。
// 2) 对张量的命名进行与 `names` 的对齐不能改变未命名维度的绝对位置。
//
// `is_aligning_two_tensors` 用于调整错误消息，以更好地匹配以下情况：
// 1) tensor.align_to(names)  (is_aligning_two_tensors=false)
// 2) torch.align_tensors([tensor, other])  (is_aligning_two_tensors=true)
static Tensor align(const Tensor& tensor, DimnameList names, bool is_aligning_two_tensors) {
  // 计算调整后张量的扩展尺寸
  std::vector<int64_t> expanded_sizes = aligned_size(
        tensor.sizes(),
        tensor.names(),
        names,
        is_aligning_two_tensors);
  // 对张量进行重命名并调整视图以匹配新的命名
  auto result = tensor.rename(nullopt).view(expanded_sizes);
  // 在结果张量上设置新的命名
  at::internal_set_names_inplace(result, names);
  // 返回带有新命名的结果张量
  return result;
}

// 计算未设置的位数
static int64_t countUnset(std::bitset<kMaxNamedTensorDim> set, int64_t up_to_idx) {
  int64_t result = 0;
  // 统计位集合中未设置的位数
  for (const auto i : c10::irange(up_to_idx)) {
    if (!set.test(i)) result++;
  }
  // 返回未设置的位数
  return result;
}

// 处理带有省略号的 `tensor.align_to(*order)` 的情况
//
// 示例：对于张量 tensor: Tensor[N, C, H, W]，考虑 `tensor.align_to('W', ..., 'N')`
// 我们将 `...` 扩展为 "所有未提及的维度，按照它们在原始张量中出现的顺序"。
//
// `order` 参数在 Python 调用时不包含省略号名称。这是因为省略号在当前的 C++ 中不是有效的命名。
// 未来的工作应该在使省略号成为有效的名称上进行改进。
//
// `ellipsis_idx` 表示省略号在 Python 调用中的位置。在我们的示例中，`tensor.align_to('W', ..., 'N')`，
// order = ['W', 'N']，ellipsis_idx = 1。
// 根据给定的参数对输入张量进行重新排列，并返回新的张量
Tensor align_to(const Tensor& tensor, DimnameList order, int64_t ellipsis_idx) {
  // 获取输入张量的命名维度列表、尺寸和步长
  const auto tensor_names = tensor.names();
  const auto tensor_sizes = tensor.sizes();
  const auto tensor_strides = tensor.strides();
  const auto tensor_dim = tensor.sizes().size();  // 获取张量的维度数
  constexpr int64_t not_found = -1;

  // General strategy.
  //
  // Step 1: We compute the following 3 things:
  // 1. How many names the ellipsis should expand to
  // 2. Which names in `tensor.names` are not mentioned in `order`.
  // 3. Where names in `order` occur in tensor, if at all.
  //
  // Step 2: Compute the new sizes/strides/names.
  // First, determine the ndim of the output tensor (this is not obvious)
  // by counting the number of names in `tensor` that are not in `order`.
  // Next, fill in output sizes/strides/names by using `order` and knowledge
  // of which dimensions in `tensor` are unmentioned in `order`.

  // 用于记录 `order` 中的每个名称是否在 `tensor_names` 中存在的位集合
  std::bitset<kMaxNamedTensorDim> order_has_tensor_name;

  // tensor_idx_for[i] = j 表示 `order` 中第 i 个名称在 `tensor_names` 中的索引是 j
  std::vector<int64_t> tensor_idx_for(order.size(), not_found);

  // 遍历 `order` 中的每个名称，确定其在 `tensor_names` 中的位置
  for (const auto order_idx : c10::irange(order.size())) {
    const auto name = order[order_idx];
    TORCH_CHECK(name.isBasic(),
        "align_to: the desired order of dimensions cannot contain a None name, got ",
        order);
    auto it = std::find(tensor_names.begin(), tensor_names.end(), name);
    if (it == tensor_names.end()) {
      continue;  // 如果名称不在 `tensor_names` 中，继续下一个名称
    }
    auto idx_in_tensor = std::distance(tensor_names.begin(), it);
    tensor_idx_for[order_idx] = idx_in_tensor;
    order_has_tensor_name.set(idx_in_tensor);  // 在位集合中标记该索引位置存在
  }

  // 计算 ellipsis 应扩展的名称数量，并得到输出张量的维度
  const auto num_ellipsis_names = countUnset(order_has_tensor_name, tensor_dim);
  const auto out_dim = num_ellipsis_names + order.size();

  // Step 2: Now that we know the size of the output tensor, we can use the
  // metadata obtained from Step 1 to fill in the new sizes/strides/names
  // 创建新的尺寸、步长和名称数组，大小为输出张量的维度
  std::vector<int64_t> new_sizes(out_dim, 1);
  std::vector<int64_t> new_strides(out_dim, 0);
  std::vector<Dimname> new_names(out_dim, Dimname::wildcard());

  // lambda 函数，用于设置新的尺寸、步长和名称
  auto setNewSizesStridesNamesFor = [&](int64_t out_dim, int64_t tensor_dim) {
    new_sizes[out_dim] = tensor_sizes[tensor_dim];
    new_strides[out_dim] = tensor_strides[tensor_dim];
    new_names[out_dim] = tensor_names[tensor_dim];
  };

  // 填充非 ellipsis 维度
  for (const auto order_idx : c10::irange(static_cast<int64_t>(order.size()))) {
    auto out_idx = order_idx;
    if (order_idx >= ellipsis_idx) {
      out_idx = order_idx + num_ellipsis_names;
    }
    const auto tensor_idx = tensor_idx_for[order_idx];
    if (tensor_idx == not_found) {
      // 添加一个新的尺寸为 1 的维度
      new_names[out_idx] = order[order_idx];
      continue;
    }
    setNewSizesStridesNamesFor(out_idx, tensor_idx);
  }

  // 填充 ellipsis 维度
  for (const auto tensor_idx : c10::irange(tensor_dim)) {
    # 如果当前张量索引满足 order_has_tensor_name 的测试条件，则跳过当前循环
    if (order_has_tensor_name.test(tensor_idx)) {
      continue;
    }
    # 为当前椭圆索引和张量索引设置新的大小、步长和名称
    setNewSizesStridesNamesFor(ellipsis_idx, tensor_idx);
    # 增加椭圆索引，为下一个张量做准备
    ellipsis_idx++;
  }

  # 检查输出维度和新名称列表是否有效
  check_names_valid_for(out_dim, new_names);

  # 定义一个 Tensor 对象 result
  Tensor result;
  {
    # 创建一个 NoNamesGuard 对象 guard，保护无名称状态
    NoNamesGuard guard;
    # 使用给定的新大小和新步长创建 result 张量
    result = tensor.as_strided(new_sizes, new_strides);
  }
  # 在 result 张量中就地设置名称，使用移动语义转移 new_names，不验证名称的有效性
  internal_set_names_inplace(result, std::move(new_names), /*validate_names=*/false);
  # 返回处理后的结果张量 result
  return result;
// 返回一个与给定张量对齐的新张量，按照指定的维度名称列表对其重新排序
Tensor align_to(const Tensor& tensor, DimnameList names) {
  // 获取给定张量的维度名称、大小和步长信息
  auto tensor_names = tensor.names();
  auto tensor_sizes = tensor.sizes();
  auto tensor_strides = tensor.strides();
  // 创建新的大小和步长向量，初始化为适当的默认值
  std::vector<int64_t> new_sizes(names.size(), 1);
  std::vector<int64_t> new_strides(names.size(), 0);

  // 遍历给定张量的每一个维度
  for (const auto idx : c10::irange(tensor_names.size())) {
    const auto& dim = tensor_names[idx];
    // 检查所有输入维度是否都有名称
    TORCH_CHECK(dim.isBasic(),
        "align_to: All input dims must be named. Found unnamed dim at index ",
        idx, " of Tensor", tensor_names);
    // 在目标名称列表中查找当前维度
    auto it = std::find(names.begin(), names.end(), dim);
    TORCH_CHECK(it != names.end(),
        "align_to: Cannot find dim ", dim, " from Tensor", names,
        " in desired alignment ", names, ".");
    // 计算当前维度在目标名称列表中的索引位置
    int64_t new_idx = std::distance(names.begin(), it);
    // 更新新的大小和步长向量
    new_sizes[new_idx] = tensor_sizes[idx];
    new_strides[new_idx] = tensor_strides[idx];
  }

  // 使用重新排序后的大小和步长创建新的张量
  Tensor result;
  {
    NoNamesGuard guard;
    result = tensor.as_strided(new_sizes, new_strides);
  }
  // 将目标名称列表应用到新的张量上
  internal_set_names_inplace(result, names);
  // 返回重新对齐后的张量
  return result;
}

// 返回与给定张量对齐的另一个张量
Tensor align_as(const Tensor& tensor, const Tensor& other) {
  return native::align_to(tensor, other.names());
}

// 将给定张量列表中的每个张量与指定的维度名称列表对齐，并返回对齐后的结果列表
static std::vector<Tensor> align_tensors_to(TensorList tensors, DimnameList names) {
  std::vector<Tensor> result;
  result.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    // 将每个张量与给定的名称列表对齐，并添加到结果列表中
    result.emplace_back(align(tensor, names, /*is_aligning_two_tensors=*/true));
  }
  return result;
}

// 将给定张量列表中的每个张量与最长维度的名称对齐，并返回对齐后的结果列表
std::vector<Tensor> align_tensors(TensorList tensors) {
  // 找到具有最大维度的张量
  auto longest_dim = std::max_element(
      tensors.begin(), tensors.end(),
      [](const Tensor& a, const Tensor& b) {
        return a.dim() < b.dim();
      });
  // 使用最长维度的名称对齐给定的张量列表，并返回结果
  return align_tensors_to(tensors, longest_dim->names());
}

// 这里是一些杂项的 Dimname 重载，它们没有具体的归属，也许我们应该将它们移动到这里或自动生成，因为它们看起来很相似。
// 这些函数都会调用未实现的 Dimname 版本，并报告未实现的维度名称重载
Tensor gather(const Tensor& self, Dimname dim, const Tensor& index, bool sparse_grad) {
  reportNYIDimnameOverload("gather");
}
Tensor& gather_out(const Tensor& self, Dimname dim, const Tensor& index, bool sparse_grad, Tensor& result) {
  reportNYIDimnameOverload("gather");
}
Tensor index_add(const Tensor& self, Dimname dim, const Tensor& index, const Tensor& source, const Scalar &alpha) {
  reportNYIDimnameOverload("index_add");
}
Tensor index_fill(const Tensor& self, Dimname dim, const Tensor& index, const Scalar& source) {
  return at::index_fill(self, dimname_to_position(self, dim), index, source);
}
Tensor& index_fill_(Tensor& self, Dimname dim, const Tensor& index, const Scalar& source) {
  return self.index_fill_(dimname_to_position(self, dim), index, source);
}
Tensor index_fill(const Tensor& self, Dimname dim, const Tensor& index, const Tensor& source) {
  return at::index_fill(self, dimname_to_position(self, dim), index, source);
}
Tensor& index_fill_(Tensor& self
```