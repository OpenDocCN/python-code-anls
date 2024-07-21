# `.\pytorch\aten\src\ATen\native\Linear.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>
#include <c10/core/SymInt.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/TensorSubclassLikeUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_trilinear.h>
#include <ATen/ops/_trilinear_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/bilinear_native.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/einsum_native.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mkldnn_linear.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/tensordot_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <cctype>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace at::native {

// 解析环境变量 "TORCH_LINEAR_FLATTEN_3D"
static inline bool parseLinearFlatten3d() {
  // 静态变量，保存解析结果，初始为未初始化状态
  static int value = -1;
  // 如果尚未解析
  if (value == -1) {
    // 获取环境变量 TORCH_LINEAR_FLATTEN_3D 的值
    const char* env_str = std::getenv("TORCH_LINEAR_FLATTEN_3D");
    // 如果环境变量存在且为 "1"，则设置解析结果为 true，否则为 false
    if (env_str != nullptr && strcmp(env_str, "1") == 0) {
      value = 1;
    } else {
      value = 0;
    }
  }
  // 返回解析结果
  return bool(value);
}

// `_flatten_nd_linear` 在执行线性操作之前，将输入张量除最后一维外全部展平
static inline Tensor _flatten_nd_linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // 获取输入张量的尺寸
    const auto input_sizes = input.sym_sizes();
    // 计算展平后的维度大小，除了最后一维
    c10::SymInt flattened_dim = 1;
    for (int64_t i = 0, ndim = input_sizes.size(); i < ndim - 1; ++i) {
      flattened_dim = flattened_dim * input_sizes[i];
    }
    // 对输入张量进行形状重塑，展平除最后一维外的所有维度
    auto inp_reshape = input.reshape_symint({flattened_dim, input_sizes.at(input_sizes.size() -1)});
    // 执行矩阵乘法操作：bias + inp_reshape * weight.t()
    const auto result = at::addmm(bias, inp_reshape, weight.t());
    // 构造新的尺寸，保留原始尺寸除最后一维外的所有维度，并添加结果张量的最后一维尺寸
    auto new_size = input_sizes.slice(0, input_sizes.size() - 1);
    c10::SymDimVector sizes_vec(new_size.begin(), new_size.end());
    sizes_vec.push_back(result.sym_size(1));
    // 返回具有新尺寸的结果张量
    return result.view_symint(sizes_vec);
}


Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  // 检查输入张量和权重张量的维度是否大于0
  const auto input_dim = input.dim();
  const auto weight_dim = weight.dim();
  TORCH_CHECK(input_dim != 0 && weight_dim != 0,
              "both arguments to linear need to be at least 1D, but they are ",
              input_dim, "D and ", weight_dim, "D");

  // See [Note: hacky wrapper removal for optional tensor]
  // 如果提供了偏置张量，则使用 borrow 方式获取其引用
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    // 创建一个可能拥有的 Tensor，类型为 MaybeOwned<Tensor>，并使用 Tensor 的 owned 构造函数，在其原位（in_place）创建。
    c10::MaybeOwned<Tensor>::owned(std::in_place);
    
    // 如果输入张量 input 是 MKL-DNN 张量
    if (input.is_mkldnn()) {
        // 调用 ATen 库中的 mkldnn_linear 函数，使用输入 input、权重 weight 和偏置 *bias 进行计算，并返回结果
        return at::mkldnn_linear(input, weight, *bias);
    }
#if defined(C10_MOBILE)
  // 如果定义了 C10_MOBILE 宏
  if (xnnpack::use_linear(input, weight, *bias)) {
    // 使用 XNNPACK 加速线性运算
    return xnnpack::linear(input, weight, *bias);
  }
#endif

  // 如果输入数据维度为 2 并且偏置已定义
  if (input_dim == 2 && bias->defined()) {
    // 使用融合操作略微更快
    return at::addmm(*bias, input, weight.t());
  }

  // 如果偏置已定义并且不是 XLA 张量
  if (bias->defined() && !input.is_xla()) {
    // 对于连续的 3D 输入，也命中融合路径，如果不使用 XLA 后端
    // 在 XLA 上，重塑/扁平化会影响性能
    if (input.is_contiguous() && input_dim == 3) {
      // 执行扁平化操作并进行线性运算
      return _flatten_nd_linear(input, weight, *bias);
    } else if (input.is_contiguous() && input.layout() == c10::kStrided && weight.layout() == c10::kStrided && bias->dim() == 1) {
      // 对于布局为 Strided 的连续输入和权重，以及一维偏置，执行扁平化操作并进行线性运算
      return _flatten_nd_linear(input, weight, *bias);
    } else if (parseLinearFlatten3d() && input_dim == 3) {
      // 如果用户通过环境变量强制进行扁平化
      const Tensor input_cont = input.contiguous();
      return _flatten_nd_linear(input_cont, weight, *bias);
    }
  }

  // 计算输入与权重转置矩阵的矩阵乘法
  auto output = at::matmul(input, weight.t());

  // 如果偏置已定义
  if (bias->defined()) {
    // 为了兼容复合使用 `add` 的版本，使用非就地操作
    if (isTensorSubclassLike(*bias) ||
        bias->_fw_grad(/*level*/ 0).defined()) {
      // 如果偏置是张量子类或具有定义的梯度
      output = at::add(output, *bias);
    } else {
      // 否则就地添加偏置
      output.add_(*bias);
    }
  }

  // 返回输出张量
  return output;
}

Tensor& linear_out(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt, Tensor& output) {
  // 确保输入不是 MKLDNN 张量，因为 linear 不支持 MKLDNN 张量的输出
  TORCH_CHECK(!input.is_mkldnn(), "linear doesn't support out for MKLDNN tensors");

  // 获取偏置张量的包装
  // 如果偏置可选值有值，则借用它；否则创建一个新的包装
  auto bias = bias_opt.has_value()
              ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
              : c10::MaybeOwned<Tensor>::owned(std::in_place);

  // 如果输入数据的维度为 2 并且偏置已定义
  if (input.dim() == 2 && bias->defined()) {
    // 使用融合操作略微更快地计算输出
    return at::addmm_out(output, *bias, input, weight.t());
  }

  // 执行矩阵乘法，并将结果存储到输出张量中
  output = at::matmul_out(output, input, weight.t());

  // 如果偏置已定义，则将其加到输出张量中
  if (bias->defined()) {
    output.add_(*bias);
  }

  // 返回输出张量的引用
  return output;
}

// sumproduct_pair 函数计算 `(left*right).sum(sumdims)`，通过排列和批量矩阵乘法来实现
// 其主要目的是为 einsum 提供成对的归约
static Tensor sumproduct_pair(const Tensor& left_, const Tensor& right_, IntArrayRef sum_dims_, bool keepdim) {
  // 假设张量已经被预先展开（以便所有维度匹配 - 广播后）
  // 但不做维度顺序的其他假设
  TORCH_CHECK(left_.dim()==right_.dim(), "number of dimensions must match");

  if (sum_dims_.empty())
    // 返回两个张量的按元素乘法结果
    return at::mul(left_, right_);
  // 获取左侧张量的维度数量
  int64_t dim = left_.dim();
  // 将需要保留的维度转换为位集合，以备后续使用
  auto sum_dims = at::dim_list_to_bitset(sum_dims_, dim);
  // 用于存储三个不同类别的维度索引：lro包含在左、右和输出中的维度，lo包含在左和输出中的维度，ro包含在右和输出中的维度
  std::vector<int64_t> lro, lo, ro;
  // 初始化符号整数用于跟踪各个类别维度的大小，以及sum_size用于乘积维度的尺寸
  SymInt lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
  // 复制左右两个张量，避免直接修改原始张量
  Tensor left = left_;
  Tensor right = right_;
  // 遍历张量的维度范围
  for (const auto i : c10::irange(dim)) {
    // 检查当前维度是否需要求和
    auto sl = left.sym_size(i)!=1;
    auto sr = right.sym_size(i)!=1;
    if (sum_dims[i]) { // 如果是需要进行乘法后求和的维度
      if (sl && sr) {  // 如果该维度在左右两个张量中都非平凡（非大小为1）
        // 检查这些维度是否大小匹配
        TORCH_CHECK(left.sym_size(i)==right.sym_size(i), "non-broadcast dimensions must match");
        // 计算这些维度的乘积大小
        sum_size *= left.sym_size(i);
      } else if (sl) { // 如果该维度只在左侧张量中非平凡，对左侧张量进行求和
        left = left.sum(i, true);
      } else if (sr) { // 如果该维度只在右侧张量中非平凡，对右侧张量进行求和
        right = right.sum(i, true);
      }
    } else if (sl && sr) { // 处理将成为输出的维度
      // 检查这些维度在左右张量中的大小是否匹配
      TORCH_CHECK(left.sym_size(i)==right.sym_size(i), "non-broadcast dimensions must match");
      // 将该维度索引添加到lro中，并计算lro的大小
      lro.push_back(i);
      lro_size *= left.sym_size(i);
    } else if (sl) { // 处理只在左侧张量中出现的维度
      // 将该维度索引添加到lo中，并计算lo的大小
      lo.push_back(i);
      lo_size *= left.sym_size(i);
    } else { // 处理只在右侧张量中出现的维度
      // 将该维度索引添加到ro中，并计算ro的大小
      ro.push_back(i);
      ro_size *= right.sym_size(i);
      }
    }
  }
    }
  }
  // 我们现在处理以下排列方式 / 形状。
  // 流程是排列输入 -> 重塑输入 -> 批量矩阵乘法 -> 重塑(view)输出 -> 排列输出
  // 输出形状为："lro, lo, 1-用于求和的维度, ro"，与原始形状维度相同
  // 左侧："lro, lo, summed" 通过 lpermutation 和三个展平的维度进行排列
  // 右侧："lro, summed, ro" 通过 rpermutation 和三个展平的维度进行排列
  // 然后排列后的输出是 bmm(left, right) 的视图
  // 最后，opermutation 将排列还原为原始维度顺序
  auto out_num_dim = lro.size() + lo.size() + sum_dims_.size() + ro.size();
  std::vector<SymInt> out_size;
  out_size.reserve(out_num_dim);
  for (auto& d : lro) out_size.push_back(left.sym_size(d));
  for (auto& d : lo) out_size.push_back(left.sym_size(d));
  for (auto& d : sum_dims_) { out_size.emplace_back(1); (void)(d); }; // 避免未使用 d 的警告
  for (auto& d : ro) out_size.push_back(right.sym_size(d));

  std::vector<int64_t> lpermutation(lro);
  lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
  lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

  std::vector<int64_t> rpermutation(lro);
  rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
  rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

  std::vector<int64_t> opermutation(out_num_dim, -1);
  {
    int64_t i = 0;

    // 设置 opermutation 数组，以便将排列还原到原始维度顺序
    for (auto it = lro.cbegin(); it != lro.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = lo.cbegin(); it != lo.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = sum_dims_.cbegin(); it != sum_dims_.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = ro.cbegin(); it != ro.cend(); i++, it++) {
      opermutation[*it] = i;
    }
  }

  // 现在可以执行上述操作
  left = left.permute(lpermutation).reshape_symint({lro_size, std::move(lo_size), sum_size});
  right = right.permute(rpermutation).reshape_symint({std::move(lro_size), std::move(sum_size), std::move(ro_size)});
  Tensor result = at::bmm(left, right);
  result = result.view_symint(out_size).permute(opermutation);

  // 最后，如果需要，挤压掉求和维度
  if (! keepdim) {
    auto sizes = result.sizes().vec();
    for (auto i = dim-1; i>=0; i--) {
      if (sum_dims[i]) {
        sizes.erase(sizes.begin() + i);
      }
    }
    result = result.view(sizes);
  }
  return result;
}

// 结束当前函数或代码块，此处应是一个函数的结尾或者代码块的结尾

// 计算 einsum 的过程大致可以分为三个部分：
// 1. 解析方程以提取每个输入操作数和输出的标签
// 2. 对输入操作数进行展开以及重新排列以对齐它们
// 3. 通过乘法和求和约简维度来计算结果。最后一部分通过 bmm 函数实现。
// 如果指定了路径，则按指定路径的顺序约简，否则默认从左到右约简。
// 路径是一个列表，格式与 opt-einsum 相同：https://optimized-einsum.readthedocs.io/en/stable/path_finding.html#format-of-the-path
Tensor einsum(c10::string_view equation, TensorList operands, at::OptionalIntArrayRef path) {
  TORCH_CHECK(!operands.empty(), "einsum(): must provide at least one operand");
  const auto num_ops = operands.size();

  // 如果指定了路径，检查路径的大小是否符合预期
  if (path.has_value()) {
    const auto path_size = num_ops == 1 ? 1 : (num_ops - 1) * 2;
    TORCH_CHECK(
        path->size() == path_size,
        "einsum(): expected contraction path given in path parameter to have size ",
        path_size,
        " but got ",
        path->size());
  }

  // 标签必须在 [A-Za-z] 范围内
  constexpr uint8_t NUM_OF_LETTERS = 'z' - 'a' + 1;
  constexpr uint8_t TOTAL_LABELS = NUM_OF_LETTERS * 2;

  // 用于识别省略号（"..."）的代码
  constexpr uint8_t ELLIPSIS = TOTAL_LABELS;

  // 将 [A-Za-z] 范围内的标签转换为 [0, TOTAL_LABELS) 范围内的下标
  auto label_to_subscript = [=](unsigned char label) -> uint8_t {
    return std::isupper(label) ? label - 'A' : label - 'a' + NUM_OF_LETTERS;
  };

  // 将 [0, TOTAL_LABELS) 范围内的下标转换为 [A-Za-z] 范围内的标签
  auto subscript_to_label = [=](uint8_t s) -> unsigned char {
    return s < NUM_OF_LETTERS ? s + 'A' : s + 'a' - NUM_OF_LETTERS;
  };

  // 找到方程中箭头（->）的位置，将方程分割为左侧和右侧
  const auto arrow_pos = equation.find("->");
  const auto lhs = equation.substr(0, arrow_pos);

  // 将输入操作数的标签转换为 [0, 52) 范围内的索引，并存储在 op_labels 中，如果存在省略号，也存储省略号
  std::vector<std::vector<uint8_t>> op_labels(num_ops);
  bool ell_in_input = false;
  std::size_t curr_op = 0;
  for (std::size_t i = 0; i < lhs.length(); ++i) {
    const unsigned char label = lhs[i];
    // 将标签转换为对应的下标
    const auto idx = label_to_subscript(label);
    // 存储每个操作数的标签及省略号（如果存在）
    op_labels[curr_op].push_back(idx);
    if (idx == ELLIPSIS) {
      ell_in_input = true;
    }
    // 如果遇到逗号，则当前操作数结束，进入下一个操作数
    if (label == ',') {
      ++curr_op;
    }
  }
    switch (label) {
      case ' ':
        // 忽略空格
        break;

      case '.':
        TORCH_CHECK(
            // 检查是否已经存在一个省略号
            !ell_in_input,
            "einsum(): found \'.\' for operand ",
            curr_op,
            " for which an ellipsis was already found");
        TORCH_CHECK(
            // 确保这是一个有效的省略号
            i + 2 < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.',
            "einsum(): found \'.\' for operand ",
            curr_op,
            " that is not part of any ellipsis");
        // 将省略号标记添加到当前操作数的标签列表中
        op_labels[curr_op].push_back(ELLIPSIS);
        ell_in_input = true;
        break;

      case ',':
        // 移动到下一个操作数
        ++curr_op;
        TORCH_CHECK(
            curr_op < num_ops,
            "einsum(): fewer operands were provided than specified in the equation");
        ell_in_input = false;
        break;

      default:
        // 解析标签
        TORCH_CHECK(
            std::isalpha(label),
            "einsum(): invalid subscript given at index ",
            i,
            " in the equation string, subscripts must be in [a-zA-Z]");
        // 将标签转换为对应的下标并添加到当前操作数的标签列表中
        op_labels[curr_op].push_back(label_to_subscript(label));
    }
  }

  TORCH_CHECK(
      curr_op == num_ops - 1,
      "einsum(): more operands were provided than specified in the equation");

  std::vector<int64_t> label_count(TOTAL_LABELS, 0);

  // 每个标签的频率计数，以及被省略号覆盖的最大维度数
  int64_t ell_num_dim = 0;

  // 计算标签的频率和被省略号覆盖的维度数
  // 在解析标签后进行此操作，以提高可读性和简化计算覆盖的维度数
  for(const auto i : c10::irange(num_ops)) {
    const auto& operand = operands[i];
    const auto labels = op_labels[i];
    const auto ndims = operand.dim();
    int64_t nlabels = static_cast<int64_t>(labels.size());
    bool has_ellipsis = false;

    for (const auto& label : labels) {
      if (label == ELLIPSIS) {
        --nlabels;
        has_ellipsis = true;
        // 更新被省略号覆盖的最大维度数
        ell_num_dim = std::max(ell_num_dim, ndims - nlabels);
      } else {
        // 增加标签计数
        ++label_count[label];
      }
    }
    // 检查 einsum() 函数的输入方程是否符合指定的维度要求
    TORCH_CHECK(
        has_ellipsis ? nlabels <= ndims : nlabels == ndims,
        "einsum(): 方程中的下标数 (",
        nlabels,
        has_ellipsis ? ") 超过了维度数 ("
                     : ") 与维度数不匹配 (",
        ndims,
        ")，操作数为 ",
        i,
        has_ellipsis ? "" : " 且未给出省略号");

  }

  // 我们希望将每个输入张量的维度对齐到形状 out_dims + sum_dims。为此，创建一个标签到索引的映射，用于排列后的形状。
  std::vector<int64_t> label_perm_index(TOTAL_LABELS, -1);

  // 当前在排列后形状中的索引
  int64_t perm_index = 0;

  // 省略号维度在排列后形状中的起始索引
  int64_t ell_index = 0;
  bool ell_in_output = false;

  if (arrow_pos == std::string::npos) {
    // 隐式输出为省略号 (...) + 只出现一次的标签
    perm_index = ell_num_dim;
    // ell_in_output 用于防止后续减少省略号维度
    ell_in_output = true;
    for (const auto label : c10::irange(TOTAL_LABELS)) {
      if (label_count[label] == 1) {
        label_perm_index[label] = perm_index++;
      }
    }
  } else {
    // 解析显式输出
    const auto rhs = equation.substr(arrow_pos + 2);
    for (std::size_t i = 0; i < rhs.length(); ++i) {
      const unsigned char label = rhs[i];
      switch (label) {
        case ' ':
          // 忽略空格
          break;

        case '.':
          TORCH_CHECK(
              // 输出中只能有一个省略号
              !ell_in_output,
              "einsum(): 输出中找到了 \'.\'，但已经存在一个省略号 (...)");
          TORCH_CHECK(
              // 确保省略号正确
              i + 2 < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.',
              "einsum(): 输出中找到了不属于任何省略号 (...) 的 \'.\'");
          ell_index = perm_index;
          perm_index += ell_num_dim;
          ell_in_output = true;
          break;

        default:
          TORCH_CHECK(
              std::isalpha(label),
              "einsum(): 在方程字符串的索引 ",
              lhs.size() + 2 + i,
              " 处给出了无效的下标，下标必须在 [a-zA-Z] 内");
          const auto index = label_to_subscript(label);
          TORCH_CHECK(
              // 确保标签至少出现在某些输入操作数中，并且在输出中最多只能出现一次
              label_count[index] > 0 && label_perm_index[index] == -1,
              "einsum(): 输出下标 ",
              label,
              label_perm_index[index] > -1
                  ? " 在输出中出现了多次"
                  : " 在方程中没有出现在任何输入操作数中");
          label_perm_index[index] = perm_index++;
      }
  }
}

// 保存输出维度的数量，在添加压缩维度（需要求和的维度）之前
const int64_t out_num_dim = perm_index;

// 如果省略号不在输出中，将其添加到压缩维度
if (!ell_in_output) {
  ell_index = perm_index;
  perm_index += ell_num_dim;
}

// 添加压缩标签（输出中不存在的标签）
for (const auto label : c10::irange(TOTAL_LABELS)) {
  // 如果标签出现次数大于零且在排列索引中为-1，则将其索引加入排列索引中
  if (label_count[label] > 0 && label_perm_index[label] == -1) {
    label_perm_index[label] = perm_index++;
  }
}

// 接下来：我们检查大小，对重复标签取对角线，展开缺失的维度，以便所有操作数具有相同的维度
// 并根据上面计算的索引重新排列操作数以对齐维度。
// 同时，我们还计算每个用于标识可以压缩的维度的标签的维度大小不为1的操作数数量。
std::vector<SymInt> label_size(TOTAL_LABELS, 1);
std::vector<SymInt> ell_sizes(ell_num_dim, 1);
std::vector<uint64_t> dim_counts(perm_index, 0);
std::deque<Tensor> ops;
for (const auto i : irange(num_ops)) {
  auto op = operands[i];
  std::vector<int64_t> permutation(perm_index, -1);
  std::int64_t dim = 0;
    // 对于操作符 i 的每个标签 s 进行迭代
    for (const auto s : op_labels[i]) {
      // 如果标签 s 是省略号（ELLIPSIS）
      if (s == ELLIPSIS) {
        // 计算省略号覆盖的每个维度
        const auto ndim = operands[i].ndimension() - (static_cast<int64_t>(op_labels[i].size()) - 1);
        // 遍历覆盖的维度并更新 ellipsis 的大小和排列
        for (auto j = ell_num_dim - ndim; j < ell_num_dim; ++j) {
          // 如果操作符 op 的维度 dim 的符号大小不为1，则更新 ellipsis 的大小
          if (op.sym_size(dim) != 1) {
            // 更新 ellipsis 的大小并增加对应维度的计数
            TORCH_CHECK(
                ell_sizes[j] == 1 || ell_sizes[j] == op.sym_size(dim),
                "einsum(): dimension ",
                dim,
                " covered by ellipsis in operand ",
                i,
                " has size ",
                op.size(dim),
                " which does not broadcast with previously seen ellipsis with size ",
                ell_sizes[j],
                " for the respective dimension");
            ell_sizes[j] = op.sym_size(dim);
            ++dim_counts[ell_index + j];
          }
          // 更新 permutation 数组以匹配 ellipsis 的维度排列
          permutation[ell_index + j] = dim++;
        }
      } else if (permutation[label_perm_index[s]] == -1) {
        // 如果标签 s 在 permutation 数组中对应位置为 -1，则更新标签的维度排列
        if (op.sym_size(dim) != 1) {
          // 更新 subscript 的大小并增加对应维度的计数
          TORCH_CHECK(
              label_size[s] == 1 || label_size[s] == op.sym_size(dim),
              "einsum(): subscript ",
              subscript_to_label(s),
              " has size ",
              op.sym_size(dim),
              " for operand ",
              i,
              " which does not broadcast with previously seen size ",
              label_size[s]);
          label_size[s] = op.sym_size(dim);
          ++dim_counts[label_perm_index[s]];
        }
        // 更新 permutation 数组以匹配标签 s 的维度排列
        permutation[label_perm_index[s]] = dim++;
      } else {
        // 如果标签重复，则取对角线
        const auto prev_dim = permutation[label_perm_index[s]];
        // 检查重复的标签维度大小是否匹配，若不匹配则报错
        TORCH_CHECK(
          op.sym_size(dim) == op.sym_size(prev_dim),
            "einsum(): subscript ",
            subscript_to_label(s),
            " is repeated for operand ",
            i,
            " but the sizes don't match, ",
            op.sym_size(dim),
            " != ",
            op.sym_size(prev_dim));
        // 对 op 取对角线操作，并移动维度以匹配前一个维度排列
        op = op.diagonal(0, prev_dim, dim).movedim(-1, prev_dim);
      }
    }

    // 添加缺失标签的维度
    for (auto& val : permutation) {
      if (val == -1) {
        // 如果 permutation 中存在值为 -1 的元素，则扩展 op 的维度并更新 permutation
        op = op.unsqueeze(dim);
        val = dim++;
      }
    }
    // 将操作添加到 ops 中，使用 permutation 进行排列
    ops.emplace_back(op.permute(permutation));
  }

  // 设置 contract_path 的默认值为空向量或提供的值
  const auto contract_path = path.value_or(std::vector<int64_t>{});
  // 设置 contract_path 的迭代器
  auto it = contract_path.begin();

  // 进行张量收缩操作
  while (ops.size() > 1) {
    int64_t i = 0;
    int64_t j = 1;

    // 如果 path 中有值，则使用其值更新 i 和 j
    if (path.has_value()) {
      i = *it++;
      j = *it++;
      // 如果 j < i，则交换 i 和 j 的值
      if (j < i) {
        std::swap(i, j);
      }

      // 检查 i 和 j 是否有效，并且不同且在 ops 的索引范围内
      TORCH_CHECK(
          i != j && i >= 0 && j < static_cast<int64_t>(ops.size()),
          "einsum(): invalid contraction (",
          i,
          ", ",
          j,
          i == j ? ") cannot contract an operand with itself"
                 : ") operand index is out of bounds");
    }

    // 执行张量收缩操作，并更新 ops 数组
    auto a = ops[i];
    // 从 ops 中获取第 j 个操作符并保存在 b 中
    auto b = ops[j];
    // 从 ops 中删除第 j 个操作符
    ops.erase(ops.begin() + j);
    // 从 ops 中删除第 i 个操作符
    ops.erase(ops.begin() + i);

    // 收集可以立即进行求和的维度
    std::vector<int64_t> sum_dims;
    // 用于存储 a 可以进行求和的维度
    SmallVector<int64_t, 5> a_dims_to_sum;
    // 用于存储 b 可以进行求和的维度
    SmallVector<int64_t, 5> b_dims_to_sum;
    // 遍历从 out_num_dim 到 perm_index 之间的每个维度
    for (auto dim = out_num_dim; dim < perm_index; ++dim) {
      // 如果 a 和 b 在当前维度上的符号大小均不为1
      if (a.sym_size(dim) != 1 && b.sym_size(dim) != 1) {
        // 如果该维度计数减到1，将其添加到可求和的维度列表中，并将计数置为0
        if (--dim_counts[dim] == 1) {
          sum_dims.push_back(dim);
          dim_counts[dim] = 0;
        }
      } else if (dim_counts[dim] == 1) {
        // 如果该维度计数为1且 a 在该维度上的符号大小不为1，将其添加到 a 可求和的维度列表中，并将计数置为0
        if (a.sym_size(dim) != 1) {
          a_dims_to_sum.push_back(dim);
          dim_counts[dim] = 0;
        // 如果该维度计数为1且 b 在该维度上的符号大小不为1，将其添加到 b 可求和的维度列表中，并将计数置为0
        } else if (b.sym_size(dim) != 1) {
          b_dims_to_sum.push_back(dim);
          dim_counts[dim] = 0;
        }
      }
    }

    // 如果存在需要求和的 a 的维度，则对其进行求和
    if (!a_dims_to_sum.empty()) {
      a = a.sum(a_dims_to_sum, true);
    }
    // 如果存在需要求和的 b 的维度，则对其进行求和
    if (!b_dims_to_sum.empty()) {
      b = b.sum(b_dims_to_sum, true);
    }

    // 如果 path 有值，则将 sumproduct_pair(a, b, sum_dims, true) 添加到 ops 的末尾
    // 否则将其添加到 ops 的开头
    if (path.has_value()) {
      ops.emplace_back(sumproduct_pair(a, b, sum_dims, true));
    } else {
      ops.emplace_front(sumproduct_pair(a, b, sum_dims, true));
    }
  }

  // 对缩并维度进行求和
  if (perm_index - out_num_dim > 0) {
    // 如果存在要缩并的操作符，我们已经在前面的循环中执行了这些操作，现在所有要求和的维度都为1
    // 注意：为了更快的性能，使用 view 而不是 squeeze（或 sum）
    if (num_ops > 1) {
      // 获取 ops[0] 的符号大小的向量
      auto sizes = ops[0].sym_sizes().vec();
      // 从 perm_index - 1 开始，逆向删除大小向量中的维度，直到 out_num_dim
      for (auto dim = perm_index - 1; dim >= out_num_dim; --dim) {
        sizes.erase(sizes.begin() + dim);
      }
      // 返回 ops[0] 的 view_symint 结果，使用新的大小向量
      return ops[0].view_symint(sizes);
    } else {
      // 创建一个包含从 out_num_dim 到 perm_index 的维度的求和维度列表
      std::vector<int64_t> sum_dims(perm_index - out_num_dim);
      std::iota(sum_dims.begin(), sum_dims.end(), out_num_dim);
      // 返回 ops[0] 在 sum_dims 上的求和结果
      return ops[0].sum(sum_dims);
    }
  }

  // 返回 ops[0]
  return ops[0];
}

// _trilinear computes a trilinear einstein sum with an unrolled dimension
// the result is `(i1.unsqueeze(expand1)*i2.unsqueeze(expand2)*i2.unsqueeze(expand3)).sum(sumdim)`
// the computation is unrolled in the unroll_dim dimension
// its main purpose is to unify the computations in bilinear and bilinear_backward
Tensor _trilinear(const Tensor& i1_, const Tensor& i2_, const Tensor& i3_,
                  IntArrayRef expand1_, IntArrayRef expand2_, IntArrayRef expand3_,
                  IntArrayRef sumdim_, int64_t unroll_dim) {
  // Calculate the total dimensionality of the tensors involved
  int64_t total_dim = i1_.dim()+expand1_.size();
  // Check if unroll_dim is within valid bounds
  TORCH_CHECK((unroll_dim >= 0) && (unroll_dim < total_dim), "unroll_dim must be in [0,", total_dim-1, "]");
  // Convert expand1_, expand2_, and expand3_ to bitsets representing dimensions
  auto expand1 = at::dim_list_to_bitset(expand1_, total_dim);
  auto expand2 = at::dim_list_to_bitset(expand2_, total_dim);
  auto expand3 = at::dim_list_to_bitset(expand3_, total_dim);
  // Convert sumdim_ to a bitset representing sum dimensions
  auto sumdim  = at::dim_list_to_bitset(sumdim_,  total_dim);
  
  // Copy input tensors i1_, i2_, and i3_ to local variables i1, i2, and i3
  Tensor i1 = i1_;
  Tensor i2 = i2_;
  Tensor i3 = i3_;
  
  // Initialize vectors and variables for output size and sum dimensions
  std::vector<c10::SymInt> output_size;
  std::vector<int64_t> sum_dims_12, sum_dims_23;
  int64_t unroll_size = -1;
  
  // Iterate over each dimension to process expansion and determine output size
  for (const auto i : c10::irange(total_dim)) {
    c10::SymInt s = 0;
    // Handle expansion for tensor i1
    if (expand1[i]) {
      i1 = i1.unsqueeze(i);
    } else  {
      s = i1.sym_size(i);
    }
    // Handle expansion for tensor i2
    if (expand2[i]) {
      i2 = i2.unsqueeze(i);
    } else  {
      s = i2.sym_size(i);
    }
    // Handle expansion for tensor i3 and determine sum dimensions
    if (expand3[i]) {
      i3 = i3.unsqueeze(i);
      if (sumdim[i] && (i != unroll_dim))
        sum_dims_12.push_back(i);
    } else  {
      s = i3.sym_size(i);
      if (sumdim[i] && (i != unroll_dim))
        sum_dims_23.push_back(i);
    }
    // Collect sizes for output tensor dimensions
    output_size.push_back(sumdim[i] ? 1 : s);
    // Record size of unrolled dimension
    if (i == unroll_dim)
      unroll_size = s.guard_int(__FILE__, __LINE__);
  }
  
  // Determine slicing factors for unrolled dimension
  int64_t slicemul1 = (expand1[unroll_dim] ? 0 : 1);
  int64_t slicemul2 = (expand2[unroll_dim] ? 0 : 1);
  int64_t slicemul3 = (expand3[unroll_dim] ? 0 : 1);
  
  // Initialize output tensor with symbolic integers based on computed output size
  auto output = at::zeros_symint(output_size, i1.options());
  
  // Perform trilinear computation if none of the input tensors have zero elements
  if (i1.sym_numel() != 0 && i2.sym_numel() != 0 && i3.sym_numel() != 0) {
    if (! sumdim[unroll_dim]) {
      // Loop over unrolled dimension to compute trilinear sum
      for (const auto k : c10::irange(unroll_size)) {
        // Perform first stage of trilinear computation
        Tensor buf = at::native::sumproduct_pair(i1.narrow(unroll_dim, k * slicemul1, 1),
                                                 i2.narrow(unroll_dim, k * slicemul2, 1),
                                                 sum_dims_12, true);
        // Perform second stage of trilinear computation
        buf = at::native::sumproduct_pair(buf, i3.narrow(unroll_dim, k * slicemul3, 1), sum_dims_23, true);
        // Add computed buffer to appropriate slice of output tensor
        output.narrow(unroll_dim, k, 1).add_(buf);
      }
    }
    else {
      // 对于每个循环中的索引 k，执行以下操作
      for (const auto k : c10::irange(unroll_size)) {
        // 从张量 i1 中选择指定维度上的切片，并计算其与张量 i2 同样维度上的切片的内积，结果保存在 buf 中
        Tensor buf = at::native::sumproduct_pair(i1.narrow(unroll_dim, k*slicemul1, 1),
                                                 i2.narrow(unroll_dim, k*slicemul2, 1), sum_dims_12, true);
        // 将 buf 与张量 i3 同样维度上的切片的内积进行计算，结果仍保存在 buf 中
        buf = at::native::sumproduct_pair(buf, i3.narrow(unroll_dim, k*slicemul3, 1), sum_dims_23, true);
        // 将 buf 的结果添加到输出张量中
        output.add_(buf);
      }
    }
  }
  // 反向遍历输出张量的维度
  for (int64_t i = output.dim()-1; i >= 0; i--)
    // 如果 sumdim 数组中索引 i 处的值为真，则在输出张量中挤压（去除）维度 i
    if (sumdim[i])
      output.squeeze_(i);
  // 返回处理后的输出张量
  return output;
// 实现了双线性插值操作，用于计算两个输入张量的双线性插值结果
Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  // 从可选的张量中获取权重张量，这里是为了处理可选张量的封装问题
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 解引用获取偏置张量
  const Tensor& bias = *bias_maybe_owned;
  
  // 检查所有张量必须具有相同的数据类型
  if (bias.defined()) {
    TORCH_CHECK(
        input1.dtype() == input2.dtype() && input1.dtype() == weight.dtype() &&
            input1.dtype() == bias.dtype(),
        "All tensors must have the same dtype, got input1: ",
        input1.dtype(),
        ", input2: ",
        input2.dtype(),
        ", weight: ",
        weight.dtype(),
        ", bias: ",
        bias.dtype());
  } else {
    TORCH_CHECK(
        input1.dtype() == input2.dtype() && input1.dtype() == weight.dtype(),
        "All tensors must have the same dtype, got input1: ",
        input1.dtype(),
        ", input2: ",
        input2.dtype(),
        ", weight: ",
        weight.dtype());
  }
  
  // 检查输入张量的维度必须相同
  TORCH_CHECK(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got ", input1.dim(), " and ", input2.dim());
  // 检查输入张量在每个维度上的大小必须相同
  for (const auto i : c10::irange(input1.dim() - 1)) {
    TORCH_CHECK(input1.sym_size(i) == input2.sym_size(i),
              "bilinear(): input batch dimensions do not match at dim ", i, ": got ", input1.sym_size(i), " and ", input2.sym_size(i));
  }
  // 检查输入张量的最后一个维度与权重张量的第二个维度大小必须相同
  TORCH_CHECK(input1.sym_size(input1.dim() - 1) == weight.sym_size(1),
            "bilinear(): input1 size does not match weight size: got ",
            input1.sym_size(input1.dim() - 1), " but expected ", weight.sym_size(1));
  // 检查输入张量的第二个维度与权重张量的第三个维度大小必须相同
  TORCH_CHECK(input2.sym_size(input2.dim() - 1) == weight.sym_size(2),
            "bilinear(): input2 size does not match weight size: got ",
            input2.sym_size(input2.dim() - 1), " but expected ", weight.sym_size(2));
  // 如果定义了偏置张量，则检查其第一个维度与权重张量的第一个维度大小必须相同
  TORCH_CHECK(!bias.defined() || bias.sym_size(0) == weight.sym_size(0),
            "bilinear(): bias size does not match weight size: got ",
            bias.sym_size(0), " but expected ", weight.sym_size(0));

  // 创建用于输出尺寸的符号整数向量
  std::vector<c10::SymInt> output_size;
  auto size1 = input1.sym_sizes();
  // 将输入张量1的符号整数大小（除最后一个维度）添加到输出尺寸中
  output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
  // 添加权重张量的第一个维度大小到输出尺寸中
  output_size.push_back(weight.sym_size(0));
  
  // 将输入张量1展平为符号整数形状
  auto input1_flattened = input1.reshape_symint({-1, input1.sym_size(-1)});
  // 将输入张量2展平为符号整数形状
  auto input2_flattened = input2.reshape_symint({-1, input2.sym_size(-1)});
  
  // 执行三线性插值操作，并将结果重塑为符号整数形状
  Tensor output = at::_trilinear(input1_flattened, weight, input2_flattened, {1,3}, {0}, {1,2}, {2,3}).reshape_symint(output_size);
  
  // 如果定义了偏置张量，则将偏置张量加到输出上
  if (bias.defined()) {
    output = output + bias;
  }
  
  // 返回插值结果张量
  return output;
}
Tensor tensordot(const Tensor& input1, const Tensor& input2, IntArrayRef dims1, IntArrayRef dims2) {
  // 检查输入维度列表长度是否相等
  TORCH_CHECK(dims1.size() == dims2.size(), "both dimension lists should have same length");
  // 检查输入张量的数据类型是否相同
  TORCH_CHECK(input1.scalar_type() == input2.scalar_type(), "both inputs should have same dtype");
  // 初始化合约维度的总大小
  SymInt csize = 1;  // total size of the contracted dimensions
  // 将输入张量赋值给局部变量 t1 和 t2
  Tensor t1 = input1;
  Tensor t2 = input2;
  // 遍历合约维度列表
  for (const auto i : c10::irange(dims1.size())) {
    // 获取输入张量在给定维度上的符号大小
    SymInt s1 = input1.sym_size(dims1[i]);
    SymInt s2 = input2.sym_size(dims2[i]);
    // 如果第二个张量在当前维度上的大小为1，则可以直接进行求和（广播）
    if (s2 == 1) {
      t1 = t1.sum(dims1[i], true, t1.scalar_type());
    } else if (s1 == 1) {
      // 如果第一个张量在当前维度上的大小为1，则可以直接进行求和（广播）
      t2 = t2.sum(dims2[i], true, t2.scalar_type());
    } else {
      // 否则检查两个合约维度的大小是否相同
      TORCH_CHECK(s1 == s2, "contracted dimensions need to match, but first has size ", s1, " in dim ", dims1[i],
               " and second has size ", s2, " in dim ", dims2[i]);
      // 计算合约维度的总大小
      csize *= s1;
    }
  }

  // 将维度列表转换为位集合，并获取输入张量的排列和结果的尺寸
  auto cdims1 = at::dim_list_to_bitset(dims1, input1.dim());
  auto cdims2 = at::dim_list_to_bitset(dims2, input2.dim());
  std::vector<int64_t> p1, p2;  // p1, p2: input permutations
  std::vector<SymInt> rsizes;  // rsizes: sizes of the result
  p1.reserve(input1.dim());
  p2.reserve(input2.dim());
  rsizes.reserve(input1.dim() + input2.dim() - (int64_t) dims1.size());
  SymInt size1 = 1; // number of non-contracted elements in input1
  SymInt size2 = 1; // number of non-contracted elements in input2

  // 填充排列并计算大小
  for (const auto i : c10::irange(input1.dim())) {
    if (! cdims1[i]) {
      p1.emplace_back(i);
      size1 *= t1.sym_size(i);
      rsizes.emplace_back(t1.sym_size(i));
    }
  }
  // 将合约维度添加到排列中
  for (const auto x : dims1) {
    p1.emplace_back(x);
  }
  // 将第二组合约维度添加到排列中
  for (const auto x : dims2) {
    p2.emplace_back(x);
  }
  // 填充排列并计算大小
  for (const auto i : c10::irange(input2.dim())) {
    if (! cdims2[i]) {
      p2.emplace_back(i);
      size2 *= t2.sym_size(i);
      rsizes.emplace_back(t2.sym_size(i));
    }
  }
  // 对输入张量进行排列和重新形状以进行矩阵乘法
  t1 = t1.permute(p1).reshape_symint({size1, csize});
  t2 = t2.permute(p2).reshape_symint({csize, size2});
  // 执行矩阵乘法并将结果重新形状为目标尺寸
  return at::mm(t1, t2).reshape_symint(rsizes);
}

Tensor &tensordot_out(const Tensor& input1, const Tensor& input2, IntArrayRef dims1, IntArrayRef dims2, Tensor& result) {
  // 调用内部实现的 tensordot 函数计算结果
  Tensor result_tmp = at::native::tensordot(input1, input2, dims1, dims2);
  // 获取临时结果张量的数据类型
  auto result_dtype = result_tmp.scalar_type();
  // 获取输出张量的数据类型和设备信息
  auto output_tensor_dtype = result.scalar_type();
  auto output_device = result.device();
  auto input1_device = input1.device();
  auto input2_device = input2.device();
  // 检查输入和输出张量是否在同一设备上
  TORCH_CHECK(
    (output_device == input1_device) && (input1_device == input2_device),
    "tensordot: Expected the output and input tensors to be on the "
    "same device, but got the output tensor on ", output_device,
  ", input tensor a on ", input1_device, ", and input tensor b on ", input2_device);
// 打印调试信息，显示输入张量 a 和 b 分别位于 input1_device 和 input2_device 上

// 检查计算结果的数据类型是否与输出张量的数据类型相同
// 因为 tensordot 不支持数据类型的自动提升
TORCH_CHECK(
  result_dtype == output_tensor_dtype, "tensordot",
  ": Expected the output tensor to have dtype ", result_dtype,
  ", but got an output tensor with dtype ", output_tensor_dtype);
// 使用 TORCH_CHECK 断言确保 result 的数据类型与 result_dtype 相同，否则输出错误信息

// 调整输出张量 result 的大小以匹配计算结果
at::native::resize_output(result, result_tmp.sizes());
// 将计算结果 result_tmp 的数据复制到输出张量 result
result.copy_(result_tmp);
// 返回最终的结果张量 result
return result;
}

}  // namespace at::native
```