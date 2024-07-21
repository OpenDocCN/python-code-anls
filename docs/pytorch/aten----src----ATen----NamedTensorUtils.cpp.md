# `.\pytorch\aten\src\ATen\NamedTensorUtils.cpp`

```
// 引入 ATen 库中的命名张量工具头文件
#include <ATen/NamedTensorUtils.h>
// 引入 ATen 库中的张量命名头文件
#include <ATen/TensorNames.h>
// 引入 ATen 库中的多维度包装工具头文件
#include <ATen/WrapDimUtilsMulti.h>
// 引入 C10 库中的整数范围工具头文件
#include <c10/util/irange.h>

// 引入标准库中的位集合头文件
#include <bitset>
// 引入标准库中的字符串流头文件
#include <sstream>

// 定义命名空间 at
namespace at {

// 对于具有名称 ('N', 'C', 'H', 'W') 的张量，返回其字符串表示 "Tensor['N', 'C', 'H', 'W']"
static std::string toDimnameRepr(const Tensor& tensor) {
  std::ostringstream os;
  os << "Tensor" << tensor.names();
  return os.str();
}

// 根据给定的张量和维度名 dim，返回该维度名在张量中的位置索引
int64_t dimname_to_position(const Tensor& tensor, Dimname dim) {
  // 检查维度名的类型不能为通配符
  TORCH_CHECK(dim.type() != NameType::WILDCARD,
      "Please look up dimensions by name, got: name = None.");
  // 检查张量是否具有命名维度
  TORCH_CHECK(tensor.has_names(),
      "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");
  // 获取张量的所有维度名
  const auto names = tensor.names();

  // 在张量的维度名列表中查找指定的 dim
  const auto it = std::find(names.begin(), names.end(), dim);
  // 如果未找到，则抛出异常
  TORCH_CHECK(it != names.end(),
      "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");

  // 返回找到的维度名在列表中的索引位置
  return std::distance(names.begin(), it);
}

// 根据给定的张量和维度名列表 dims，返回每个维度名在张量中的位置索引的向量
std::vector<int64_t> dimnames_to_positions(const Tensor& tensor, DimnameList dims) {
  std::vector<int64_t> result;
  result.reserve(dims.size());
  // 遍历维度名列表 dims，对每个维度名调用 dimname_to_position 获取其索引，并存入 result 向量
  for (const auto& name : dims) {
    result.push_back(dimname_to_position(tensor, name));
  }
  return result;
}

// 报告维度名位置错误的静态方法
static void report_positional_error(
    const Dimname& name,
    const Dimname& other_name,
    DimnameList names,
    DimnameList other_names,
    const char* action) {
  // TODO(zou3519): 可以通过检查是否可以对齐名称并建议解决方法来改进消息
  TORCH_CHECK(false,
      "Error when attempting to ", action, " dims ", names, " and dims ",
      other_names, ": dim ", name, " and dim ", other_name, " are at the same position "
      "from the right but do not match.")
}

// 检查维度名是否错位的静态方法
static void check_for_misalignment(
    const Dimname& name,
    DimnameList names,
    DimnameList other_names,
    const char* action) {
  // 如果维度名是通配符，则直接返回
  if (name.isWildcard()) {
    return;
  }
  // 在 other_names 中查找当前维度名 name
  auto it = std::find(other_names.begin(), other_names.end(), name);
  // TODO(zou3519): 可以通过检查是否可以对齐名称并建议解决方法来改进消息
  TORCH_CHECK(it == other_names.end(),
      "Misaligned dims when attempting to ", action, " dims ", names, " and dims ",
      other_names, ": dim ", name, " appears in a different position from the right "
      "across both lists.");
}

// 从右侧统一两个维度名列表的静态方法
// 假设：一个 DimnameList 可以没有重复的完整名称，除了通配符
std::vector<Dimname> unify_from_right(
    DimnameList names,
    DimnameList other_names,
    const char* action) {
  // 获取通配符 Dimname
  const auto wildcard = Dimname::wildcard();
  // 计算两个列表的最大大小
  const auto size = std::max(names.size(), other_names.size());
  // 创建结果向量，并初始化为通配符
  auto result = std::vector<Dimname>(size, wildcard);

  // 反向迭代 names 和 other_names，从右向左处理
  auto names_it = names.rbegin();
  auto other_it = other_names.rbegin();
  auto result_it = result.rbegin();
  while (names_it != names.rend() || other_it != other_names.rend()) {
    // 如果 names 迭代器已经到达末尾，则使用通配符
    const auto& name = names_it == names.rend() ? wildcard : *names_it;

    // 更新 result 向量的当前位置迭代器
    *result_it = name;

    // 移动到下一个位置
    ++names_it;
    ++other_it;
    ++result_it;
  }

  // 返回统一后的结果向量
  return result;
}

// 命名空间 at 结束
}
    // 使用条件运算符根据条件选择 other_it 或 wildcard 的引用，并赋值给 other_name
    const auto& other_name = other_it == other_names.rend() ? wildcard : *other_it;

    // 步骤1：检查名称是否匹配
    // 尝试将 name 和 other_name 统一，如果失败则报告位置错误
    const auto maybeName = name.unify(other_name);
    if (!maybeName) {
      report_positional_error(name, other_name, names, other_names, action);
    }
    // 将 maybeName 的值存入 result_it 指向的位置
    *result_it = *maybeName;

    // 步骤2：检查名称是否对齐
    // 如果 name 或 other_name 不是基本名称，进行以下检查
    if (!name.isBasic() || !other_name.isBasic()) {
      // 设 N 为 names 和 other_names 的最大长度
      // 设 K 为 names 和 other_names 中特殊名称的数量
      // 此搜索（包括外部循环）的复杂度为 O(N*K)，但通常维度数较小
      check_for_misalignment(name, names, other_names, action);
      check_for_misalignment(other_name, other_names, names, action);
    }

    // 如果 names_it 迭代器未到达 rend，则向前移动
    if (names_it != names.rend()) {
      ++names_it;
    }
    // 如果 other_it 迭代器未到达 rend，则向前移动
    if (other_it != other_names.rend()) {
      ++other_it;
    }
    // 向前移动 result_it 迭代器
    ++result_it;
  }
  // 返回结果
  return result;
}

namespace namedinference {

// 返回一个位集，标识在指定维度列表中未排除的索引位置
static std::bitset<dim_bitset_size>
compute_included_idxs(IntArrayRef excluded_idxs, int64_t ndims) {
  auto result = dim_list_to_bitset(excluded_idxs, ndims); // 将排除的索引列表转换为位集
  result.flip(); // 反转位集，得到包含的索引位置
  return result; // 返回计算得到的位集
}

// 断言两个 DimnameList 是否相等，否则抛出错误信息
static void assert_names_equal(DimnameList a, DimnameList b) {
  TORCH_CHECK(a == b, // 使用 TORCH_CHECK 断言 a 和 b 是否相等
      "Name mismatch: specified out tensor with names ", a,
      " are not the same as the computed output names ", b,
      ". Please rename the out tensor's dims with `Tensor.rename`."); // 错误信息提示
}

// 如果可能的命名列表非空且存在，则传播命名到结果张量
const Tensor& propagate_names_if_present_and_nonempty(const Tensor& result,
    std::optional<DimnameList> maybe_names,
    bool validate_names) {
  auto maybe_name_list = maybe_names.value_or(at::ArrayRef<Dimname>{}); // 获取可能的命名列表，若为空则使用空列表
  propagate_names_if_nonempty(result.unsafeGetTensorImpl(), maybe_name_list, validate_names); // 调用非空命名传播函数
  return result; // 返回结果张量
}

// 如果可能的命名列表非空且存在，则传播命名到结果张量
const Tensor& propagate_names_if_nonempty(const Tensor& result,
    DimnameList maybe_names,
    bool validate_names) {
  propagate_names_if_nonempty(result.unsafeGetTensorImpl(), maybe_names, validate_names); // 调用非空命名传播函数
  return result; // 返回结果张量
}

// 如果可能的命名列表非空且存在，则传播命名到结果张量实现
TensorImpl* propagate_names_if_nonempty(TensorImpl* result,
    DimnameList maybe_names,
    bool validate_names) {
  if (maybe_names.empty()) { // 如果可能的命名列表为空
    return result; // 直接返回结果张量实现
  }
  return propagate_names(result, maybe_names, validate_names); // 调用命名传播函数
}

// 传播命名到结果张量
const Tensor& propagate_names(const Tensor& result, DimnameList names, bool validate_names) {
  propagate_names(result.unsafeGetTensorImpl(), names, validate_names); // 调用命名传播函数
  return result; // 返回结果张量
}

// 传播命名到结果张量实现
TensorImpl* propagate_names(TensorImpl* result, DimnameList names, bool validate_names) {
  if (result->dim() > 0) { // 如果结果张量维度大于0
    TORCH_INTERNAL_ASSERT( // 内部断言，确保条件成立，否则抛出错误信息
        !names.empty(), // 确保命名列表非空
        "propagate_names: passed in empty names to propagate to result with",
        " shape ", result->sizes(), ". Empty names means that name inference did",
        "not occur; use `propagate_names_if_nonempty` instead of `propagate_names`.");
  }
  if (!impl::has_names(result)) { // 如果结果张量实现没有命名
    impl::internal_set_names_inplace(result, names, validate_names); // 在原地设置命名
  } else {
    assert_names_equal(impl::get_names(result), names); // 断言实际命名与给定命名相等
  }
  return result; // 返回结果张量实现
}

// 传播命名到结果张量，除了指定的排除索引
void propagate_names_except(const Tensor& result, const Tensor& src, IntArrayRef excluded_idxs) {
  if (!result.has_names() && !src.has_names()) { // 如果结果张量和源张量都没有命名
    return; // 直接返回
  }
  const auto src_names = src.names(); // 获取源张量的命名
  const auto result_dim = static_cast<int64_t>(result.dim()); // 结果张量的维度
  const auto src_dim = static_cast<int64_t>(src_names.size()); // 源张量的维度
  const auto excluded_dim = static_cast<int64_t>(excluded_idxs.size()); // 排除索引的数量
  TORCH_INTERNAL_ASSERT(src_dim - excluded_dim == result_dim); // 内部断言，确保排除后的维度与结果张量维度一致

  // 快速路径
  if (excluded_idxs.size() == 1) { // 如果排除索引只有一个
    std::vector<Dimname> outnames = src_names.vec(); // 复制源张量的命名列表
    outnames.erase(outnames.begin() + maybe_wrap_dim(excluded_idxs[0], src_dim)); // 删除指定索引的命名
    propagate_names(result, outnames); // 传播处理后的命名到结果张量
    return; // 结束处理
  }

  std::vector<Dimname> outnames; // 输出命名列表
  outnames.reserve(result_dim); // 预留空间
  auto included_idxs = compute_included_idxs(excluded_idxs, src_dim); // 计算包含的索引集合
  for (const auto dim : c10::irange(src_dim)) { // 遍历源张量的维度范围
    # 如果 included_idxs 列表中的当前维度的值为真（非零），则执行以下操作
    if (included_idxs[dim]) {
      # 将当前维度对应的源名称 src_names[dim] 添加到 outnames 列表末尾
      outnames.push_back(src_names[dim]);
    }
  }
  # 调用 propagate_names 函数，传递 result 和 outnames 作为参数
  propagate_names(result, outnames);
}

void propagate_names_for_reduction(const Tensor& result, const Tensor& src, IntArrayRef reduced_dims, bool keepdim) {
  // 如果 keepdim 为 true，则传播 result 和 src 的命名信息并返回
  if (keepdim) {
    propagate_names(result, src);
    return;
  }
  // 如果 reduced_dims 为空，表示进行完全缩减，无需传播命名信息
  // 实际上是进行完全缩减操作
  if (reduced_dims.empty()) {
    return;
  }
  // 传播 result 和 src 的命名信息，但不包括指定的缩减维度
  propagate_names_except(result, src, reduced_dims);
}

// 传播 result 张量和 src 张量的命名信息
void propagate_names(const Tensor& result, const Tensor& src) {
  propagate_names(result.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
}

// 传播 result 张量实现和 src 张量实现的命名信息
void propagate_names(TensorImpl* result, TensorImpl* src) {
  // 如果 result 和 src 是同一个张量实现，则无需传播命名信息
  if (result == src) {
    return;
  }
  // 如果 result 和 src 都没有命名信息，则无需传播
  if (!impl::has_names(result) && !impl::has_names(src)) {
    return;
  }
  // 传播 src 张量实现的命名信息到 result 张量实现
  propagate_names(result, impl::get_names(src));
}

// 计算在 squeeze 操作中要移除的维度的输出命名
std::vector<Dimname> compute_squeeze_outnames(const Tensor& tensor) {
  // 如果张量没有命名信息，则返回空向量
  if (!tensor.has_names()) {
    return {};
  }
  // 准备输出向量
  std::vector<Dimname> outnames;
  auto tensor_names = tensor.names();
  // 遍历张量的维度
  for (const auto d : c10::irange(tensor.dim())) {
    // 如果维度大小不为 1，则保留该维度的命名
    if (tensor.sym_sizes()[d] != 1) {
      outnames.push_back(tensor_names[d]);
    }
  }
  return outnames;
}

// 计算在 squeeze 操作中要移除的指定维度的输出命名
std::vector<Dimname> compute_squeeze_outnames(const Tensor& tensor, std::bitset<dim_bitset_size> dims) {
  // 如果张量没有命名信息，则返回空向量
  if (!tensor.has_names()) {
    return {};
  }
  // 准备输出向量
  std::vector<Dimname> outnames;
  auto tensor_names = tensor.names();
  // 遍历张量的维度
  for (const auto d : c10::irange(tensor.dim())) {
    // 如果维度不在 dims 中或者维度大小不为 1，则保留该维度的命名
    if (!dims.test(d) || tensor.sym_sizes()[d] != 1) {
      outnames.push_back(tensor_names[d]);
    }
  }
  return outnames;
}

// 计算在 diagonal 操作中要保留的维度输出命名
std::vector<Dimname> compute_diagonal_outnames(
    const Tensor& tensor,
    int64_t dim1,
    int64_t dim2) {
  // 如果张量没有命名信息，则返回空向量
  if (!tensor.has_names()) {
    return {};
  }
  // 准备输出向量
  std::vector<Dimname> outnames;
  auto tensor_names = tensor.names();
  // 遍历张量的维度
  for (const auto d : c10::irange(tensor.dim())) {
    // 如果维度不是 dim1 或 dim2，则保留该维度的命名
    if (d == dim1 || d == dim2) {
      continue;
    }
    outnames.push_back(tensor_names[d]);
  }
  // 添加通配符维度作为最后一个维度的命名
  outnames.push_back(Dimname::wildcard());
  return outnames;
}

// 检查特征维度名称是否都不同，防止输出张量中出现重复名称
static void check_feature_names_are_distinct(
    DimnameList self_names,
    DimnameList other_names,
    const DimnameList& outnames) {
  // 如果输入张量的特征维度少于 2 个，则无需检查
  if (self_names.size() < 2 || other_names.size() < 2) {
    return;
  }
  // 获取倒数第二个和最后一个特征维度名称
  auto feature0 = outnames[outnames.size() - 2];
  auto feature1 = outnames[outnames.size() - 1];
  // 检查最后两个特征维度名称是否相同或者有通配符
  TORCH_CHECK(
    feature0 == Dimname::wildcard() || feature0 != feature1,
    "Matrix multiplying Tensor", self_names,
    " with Tensor", other_names,
    " would produce output tensor with duplicate names ",
    outnames,
    ". Please rename the input tensors with `Tensor.rename` to prevent this.");
}

// 计算张量名称列表中的批处理维度数目
static int64_t num_batch_dims(DimnameList names) {
  // 如果名称列表的大小小于等于 2，则批处理维度数为 0
  if (names.size() <= 2) {
    return 0;
  }
  // 否则批处理维度数为列表大小减去 2
  return static_cast<int64_t>(names.size() - 2);
}

// 计算矩阵乘法操作的输出命名
static std::vector<Dimname> compute_matmul_outnames(
    DimnameList self_names,
  // 检查self_names和other_names都不为空，因为它们至少需要是1维的数组
  TORCH_CHECK(!self_names.empty() && !other_names.empty(),
      "both arguments to matmul need to be at least 1D, but they are ",
      self_names.size(), "D and ", other_names.size(), "D");

  // matmul函数执行self和other之间的批量矩阵乘法，它们可以是以下之一：
  // - 矩阵的批次（如果维度 > 2）
  // - 矩阵（如果维度 == 2）
  // - 向量（如果维度 == 1）
  //
  // 计算输出名称时，我们统一批次维度，因为这些可以进行广播以得到输出的批次维度。
  //
  // 然后，我们附加一些名称，这些名称等于没有批次维度的matmul结果。这些名称通过删除合并掉的维度的名称来计算。
  // 我们总是将第一个张量的最后一个维度与第二个张量的第一个特征维度进行合并。

  // 获取输出的批次维度名称
  auto wrapped_self_names = TensorNames(self_names, 0, num_batch_dims(self_names));
  const auto wrapped_other_names = TensorNames(other_names, 0, num_batch_dims(other_names));
  auto& working_names = wrapped_self_names.unifyFromRightInplace(wrapped_other_names, "matmul");

  // 添加每个单独（非批量）matmul的结果。
  // 如果self或other中有维度为1，则它们是向量。向量在matmul期间完全被合并，因此我们不从它们中获取任何名称。
  if (self_names.size() >= 2) {
    working_names.append(TensorName(self_names, -2));
  }
  if (other_names.size() >= 2) {
    working_names.append(TensorName(other_names, -1));
  }
  auto result = working_names.toDimnameVec();

  // 检查特征名称是否不同
  check_feature_names_are_distinct(self_names, other_names, result);
  // 返回最终的结果名称
  return result;
}

// 根据 mat、vec 和 bias 的命名情况传播命名
std::vector<Dimname> propagate_names_for_addmv(
    const Tensor& mat,
    const Tensor& vec,
    const Tensor& bias) {
  // 如果 mat、vec 和 bias 都没有命名，则返回空的 Dimname 向量
  if (!mat.has_names() &&
      !vec.has_names() && !bias.has_names()) {
    return std::vector<Dimname>{};
  }
  // 计算矩阵乘法后的输出命名
  auto mv_outnames = compute_matmul_outnames(mat.names(), vec.names());
  // 将输出命名与 bias 的命名进行统一，并返回结果
  return unify_from_right(mv_outnames, bias.names());
}

// 根据 m1、m2 和 bias 的命名情况传播命名
std::vector<Dimname> propagate_names_for_addmm(
    const Tensor& m1,
    const Tensor& m2,
    const Tensor& bias) {
  // 如果 m1、m2 和 bias 都没有命名，则返回空的 Dimname 向量
  if (!m1.has_names() && !m2.has_names() &&
      !bias.has_names()) {
    return std::vector<Dimname>{};
  }
  // 计算矩阵乘法后的输出命名
  auto mm_outnames = compute_matmul_outnames(m1.names(), m2.names());
  // 将输出命名与 bias 的命名进行统一，并返回结果
  return unify_from_right(mm_outnames, bias.names());
}

// 检查 vec1 和 vec2 的命名情况是否支持点乘
void check_names_for_dot(
    TensorImpl* vec1,
    TensorImpl* vec2) {
  // 如果 vec1 和 vec2 都没有命名，则直接返回
  if (!impl::has_names(vec1) && !impl::has_names(vec2)) {
    return;
  }
  // 计算点乘后的输出命名
  compute_matmul_outnames(impl::get_names(vec1), impl::get_names(vec2));
}

// 根据 result 和 self 的命名情况传播命名，用于扩展操作
void propagate_names_for_expand(const Tensor& result, const Tensor& self) {
  // 如果 self 没有命名，则直接返回
  if (!self.has_names()) {
    return;
  }
  // 获取 result 和 self 的维度
  auto result_dim = result.dim();
  // 如果 self 的维度与 result 的维度相同，则直接传播命名并返回
  if (self.dim() == result_dim) {
    propagate_names(result, self);
    return;
  }
  // 否则，创建一个新的输出命名向量，将 self 的命名拷贝到正确的位置
  std::vector<Dimname> outnames(result_dim, Dimname::wildcard());
  std::copy(
      self.opt_names()->begin(),
      self.opt_names()->end(),
      outnames.begin() + result_dim - self.dim());
  // 将 result 和计算得到的 outnames 进行命名传播
  propagate_names(result, outnames);
}

// 计算两个 Tensor 在广播操作后的输出命名
std::vector<Dimname> compute_broadcast_outnames(
    const Tensor& self,
    const Tensor& other) {
  // 如果 self 和 other 都没有命名，则返回空的 Dimname 向量
  if (!self.has_names() && !other.has_names()) {
    return {};
  }
  // 将 self 和 other 的命名进行右侧对齐，并返回结果
  return unify_from_right(self.names(), other.names());
}

// 将 tensor 广播到 reference_tensor 的命名情况，并返回输出命名
std::vector<Dimname> broadcast_to_outnames(
    const Tensor& tensor,
    const Tensor& reference_tensor,
    const char* op_name) {
  // 如果 tensor 和 reference_tensor 都没有命名，则返回空的 Dimname 向量
  if (!tensor.has_names() && !reference_tensor.has_names()) {
    return {};
  }
  // 获取 reference_tensor 和 tensor 的命名
  auto reference_names = reference_tensor.names();
  auto tensor_names = tensor.names();
  // 检查 tensor 的维度是否小于等于 reference_tensor 的维度
  TORCH_CHECK(
      reference_names.size() >= tensor_names.size(),
      op_name, ": attempted to broadcast Tensor", tensor_names, " to Tensor",
      reference_names, " but the number of dims (", tensor_names.size(),
      ") must be less than or equal to the number of dims in the tensor (",
      reference_names.size(), ")");
  // 将 reference_tensor 和 tensor 的命名进行右侧对齐，并返回结果
  return unify_from_right(reference_names, tensor_names);
}

// 计算 MaterializedITensorListRef 中 Tensor 进行 concatenate 后的输出命名
std::vector<Dimname> compute_cat_outnames(const MaterializedITensorListRef& tensors) {
  // 如果 tensors 中没有 Tensor 带有命名，则返回空的 Dimname 向量
  if (!at::has_names(tensors)) {
    return {};
  }
  // 初始化结果向量
  std::vector<Dimname> result;
  // 遍历 tensors 中的每个 Tensor
  for (const Tensor& tensor : tensors) {
    const auto tensor_names = tensor.names();
    // 检查是否有零维度的 Tensor，不能进行 concatenate 操作
    TORCH_CHECK(!tensor_names.empty(), "zero-dimensional tensor cannot be concatenated");
    // 检查结果是否为空或张量名称数与结果数相同，否则抛出异常
    TORCH_CHECK(result.empty() || tensor_names.size() == result.size(),
        "Tensors must have same number of dimensions: got ", result.size(),
        " and ", tensor_names.size());
    // 使用右侧对齐方式，将结果与张量名称进行统一（合并），方法为 "cat"
    result = unify_from_right(result, tensor_names, "cat");
  }
  // 返回处理后的结果
  return result;
} // namespace namedinference
} // namespace at

std::vector<Dimname> compute_matmul_outnames(
    const Tensor& self,
    const Tensor& other) {
  // 如果 self 和 other 都没有命名，则返回空向量
  if (!self.has_names() && !other.has_names()) {
    return {};
  }
  // 调用另一个函数来计算矩阵乘法的输出维度名称
  return compute_matmul_outnames(self.names(), other.names());
}

std::vector<Dimname> compute_cdist_outnames(
    const Tensor& self,
    const Tensor& other) {
  // 如果 self 和 other 都没有命名，则返回空向量
  if (!self.has_names() && !other.has_names()) {
    return {};
  }
  // 获取 self 和 other 的命名
  const auto self_names = self.names();
  const auto other_names = other.names();

  // 创建 self 的命名对象，并标识其作为批次的一部分
  auto self_batch = TensorNames(self_names, 0, num_batch_dims(self_names));
  // 创建 other 的命名对象，并标识其作为批次的一部分
  const auto other_batch = TensorNames(other_names, 0, num_batch_dims(other_names));

  // 在 self_batch 中从右边就地合并 other_batch 的命名，用 "cdist" 标记
  auto& result = self_batch.unifyFromRightInplace(other_batch, "cdist");

  // 添加关于 cdist 函数处理方式的详细说明
  // cdist 将 self 和 other 视为 M x D 和 N x D 张量的批次。
  // 它计算 `self` 中每个 M 向量（大小为 D）与 `other` 中每个 N 向量之间的成对距离，
  // 返回一个 M x N 的距离值批次。我们传播大小为 M（在 self 中）和大小为 N（在 other 中）
  // 的维度的名称，它们都是倒数第二个维度。
  result.append(TensorName(self_names, -2));
  result.append(TensorName(other_names, -2));
  result.checkUnique("cdist");

  // 将结果转换为 Dimname 向量并返回
  return result.toDimnameVec();
}

std::vector<Dimname> compute_bmm_outnames(
    const Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  // 如果 result、self 和 other 都没有命名，则返回空向量
  if (!result.has_names() && !self.has_names() && !other.has_names()) {
    return {};
  }
  // 调用另一个函数来计算矩阵乘法的输出维度名称
  return compute_matmul_outnames(self.names(), other.names());
}

std::vector<Dimname> compute_baddbmm_outnames(
    const Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const Tensor& bias) {
  // 如果 result、self、other 和 bias 都没有命名，则返回空向量
  if (!result.has_names() && !self.has_names()
    && !other.has_names() && !bias.has_names()) {
    return {};
  }
  // 计算矩阵乘法的输出维度名称
  auto bmm_names = compute_matmul_outnames(self.names(), other.names());
  // 在 bias 的命名中从右边合并 bmm_names 的命名
  auto baddbmm_names = unify_from_right(bias.names(), bmm_names);
  return baddbmm_names;
}

bool are_names_equal(TensorImpl* self, TensorImpl* other) {
  // 如果 self 和 other 都没有命名，则它们被认为是相等的
  if (!impl::has_names(self) && !impl::has_names(other)) {
    return true;
  }
  // 比较 self 和 other 的命名是否相等
  return impl::get_names(self) == impl::get_names(other);
}
```