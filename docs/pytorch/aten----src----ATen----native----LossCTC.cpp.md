# `.\pytorch\aten\src\ATen\native\LossCTC.cpp`

```py
// 版权声明及许可信息
// 本代码版权归MathInf GmbH, Thomas Viehmann所有，使用BSD-3-Clause许可协议
// 这是Connectionist Temporal Loss的CPU实现
// 主要遵循Graves的方法
// 1. 参考文献Graves et al.: http://www.cs.toronto.edu/~graves/icml_2006.pdf
// 我们使用上述链接中的方程，但注意[1]使用基于1的索引，我们使用基于0的索引。
// Graves et al.称概率为y，我们使用log_probs（也称为inputs）
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>            // 引入ATen张量的核心头文件
#include <ATen/Dispatch.h>               // 引入ATen分发相关头文件
#include <ATen/Parallel.h>               // 引入ATen并行计算相关头文件
#include <ATen/TensorIterator.h>         // 引入ATen张量迭代器相关头文件
#include <ATen/TensorOperators.h>        // 引入ATen张量操作相关头文件
#include <ATen/native/Fill.h>            // 引入ATen填充相关头文件
#include <c10/util/irange.h>             // 引入C10的范围操作头文件
#include <ATen/TensorSubclassLikeUtils.h>// 引入ATen张量子类相关工具头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>              // 引入ATen函数头文件
#include <ATen/NativeFunctions.h>        // 引入ATen本地函数头文件
#else
#include <ATen/ops/_ctc_loss.h>                   // 引入ATen CTC损失函数相关头文件
#include <ATen/ops/_ctc_loss_backward.h>          // 引入ATen CTC损失反向传播函数相关头文件
#include <ATen/ops/_ctc_loss_backward_native.h>   // 引入ATen CTC损失本地反向传播函数相关头文件
#include <ATen/ops/_ctc_loss_native.h>            // 引入ATen CTC损失本地函数相关头文件
#include <ATen/ops/_cudnn_ctc_loss.h>             // 引入ATen CTC损失CUDA库相关头文件
#include <ATen/ops/_use_cudnn_ctc_loss.h>         // 引入ATen 使用CUDA库CTC损失函数相关头文件
#include <ATen/ops/ctc_loss_native.h>             // 引入ATen CTC损失本地函数相关头文件
#include <ATen/ops/empty.h>                       // 引入ATen 空张量函数相关头文件
#include <ATen/ops/empty_like.h>                  // 引入ATen 类似空张量函数相关头文件
#include <ATen/ops/full_like.h>                   // 引入ATen 类似满张量函数相关头文件
#include <ATen/ops/tensor.h>                      // 引入ATen 张量操作相关头文件
#include <ATen/ops/where.h>                       // 引入ATen where函数相关头文件
#include <ATen/ops/zeros.h>                       // 引入ATen 全零张量函数相关头文件
#endif

#include <type_traits>                // 引入类型特性相关头文件
#include <utility>                    // 引入实用工具相关头文件

namespace at::native {

namespace {

// 此处的ad-hoc函数将targets（在[1]中表示为l）转换为增强目标（在[1]中表示为l'）
// 注意：此处不执行任何边界检查
template<typename target_t>
static inline int64_t get_target_prime(target_t* target, int64_t offset, int64_t stride, int64_t idx, int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;  // 如果idx为偶数，返回BLANK
  } else {
    return target[offset + stride * (idx / 2)];  // 如果idx为奇数，返回增强目标值
  }
}

template<typename scalar_t, ScalarType target_scalar_type>
// 分配 CTC 损失函数计算过程中所需的输出对象：log_probs, targets, input_lengths, target_lengths
std::tuple<Tensor, Tensor, size_t, std::vector<int64_t>> ctc_loss_allocate_outputs(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK) {
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)

  // 创建一个检查来源对象
  CheckedFrom c = "ctc_loss_allocate_outputs";
  // 定义 log_probs 的 TensorArg 对象
  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  // 定义 targets 的 TensorArg 对象
  auto targets_arg = TensorArg(targets, "targets", 2);
  // 检查 targets 的标量类型是否符合要求
  checkScalarType(c, targets_arg, target_scalar_type);
  // 检查 log_probs 的维度是否为 3
  checkDim(c, log_probs_arg, 3);
  // 检查 targets 的维度范围是否在 1 到 3 之间
  checkDimRange(c, targets_arg, 1, 3);

  // 获取 log_probs 的批大小（batch_size）和标签数（num_labels）
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  // 检查 BLANK 是否在标签范围内
  TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  // 检查 input_lengths 的大小是否与 batch_size 相等
  TORCH_CHECK((int64_t) input_lengths.size() == batch_size, "input_lengths must be of size batch_size");
  // 检查 target_lengths 的大小是否与 batch_size 相等
  TORCH_CHECK((int64_t) target_lengths.size() == batch_size, "target_lengths must be of size batch_size");

  // 定义变量 tg_target_stride 和 tg_batch_offsets，初始化 max_target_length 为 0
  size_t tg_target_stride;
  int64_t max_target_length = 0;
  std::vector<int64_t> tg_batch_offsets(batch_size);

  // 如果 targets 的维度为 1，则表示是拼接的目标值
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    // 遍历 batch_size
    for (const auto i : c10::irange(batch_size)) {
      // 检查目标长度是否为非负数
      TORCH_CHECK(target_lengths[i] >= 0,
                  "Expected target_lengths to have value at least ", 0, ", but got value ", target_lengths[i],
                  " (while checking arguments for ", c, ")");
      // 设置 tg_batch_offsets[i] 的值，并更新 pos
      tg_batch_offsets[i] = pos;
      pos += target_lengths[i];
      // 更新 max_target_length
      if (max_target_length < target_lengths[i])
         max_target_length = target_lengths[i];
    }
    // 设置 tg_target_stride 为 targets 的步幅
    tg_target_stride = targets.stride(0);
    // 检查 targets 的大小是否为 pos
    checkSize(c, targets_arg, 0, pos);
  }
  else { // batch x max_target_length
    // targets 的维度为 2，即 batch x max_target_length
    int64_t tg_batch_stride = targets.stride(0);
    // 遍历 batch_size
    for (const auto i : c10::irange(batch_size)) {
      // 检查目标长度是否为非负数
      TORCH_CHECK(target_lengths[i] >= 0,
                  "Expected target_lengths to have value at least ", 0, ", but got value ", target_lengths[i],
                  " (while checking arguments for ", c, ")");
      // 设置 tg_batch_offsets[i] 的值
      tg_batch_offsets[i] = i * tg_batch_stride;
      // 更新 max_target_length
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    // 设置 tg_target_stride 为 targets 的第二维度步幅
    tg_target_stride = targets.stride(1);
    // 检查 targets 第二维度的大小是否至少为 max_target_length
    checkSize(c, targets_arg, 0, batch_size);
    TORCH_CHECK(targets.size(1) >= max_target_length,
             "Expected tensor to have size at least ", max_target_length, " at dimension 1, but got size ", targets.size(1), " for ", targets_arg,
             " (while checking arguments for ", c, ")");
  }

  // 获取 log_probs 的最大输入长度
  int64_t max_input_length = log_probs.size(0);
  // 遍历 batch_size
  for (const auto b : c10::irange(batch_size)) {
    // 检查输入长度是否为非负数
    TORCH_CHECK(input_lengths[b] >= 0,
             "Expected input_lengths to have value at least ", 0, ", but got value ", input_lengths[b],
             " (while checking arguments for ", c, ")");
    // 检查输入长度是否小于等于最大输入长度，如果不是，则抛出错误信息
    TORCH_CHECK(input_lengths[b] <= max_input_length,
                "Expected input_lengths to have value at most ", max_input_length, ", but got value ", input_lengths[b],
                " (while checking arguments for ", c, ")");
  }

  // 创建一个空的张量 log_alpha，其形状为 {batch_size, log_probs.size(0), 2*max_target_length+1}，使用 log_probs 的选项
  Tensor log_alpha = at::empty({batch_size, log_probs.size(0), 2*max_target_length+1}, log_probs.options());
  // 创建一个空的张量 neg_log_likelihood，其形状为 {batch_size}，使用 log_probs 的选项
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  // 返回一个元组，包含 neg_log_likelihood、log_alpha、tg_target_stride 和 tg_batch_offsets
  return std::make_tuple(neg_log_likelihood, log_alpha, tg_target_stride, tg_batch_offsets);
// 结束模板函数 ctc_loss_cpu_template，处理 CTC 损失计算的核心部分。
// log_probs: 输入数据的对数概率，维度为 input_len x batch_size x num_labels
// targets [int64]: 目标序列，维度为 batch_size x target_length 或 sum(target_lengths)
// input_lengths: 输入序列的长度
// target_lengths: 目标序列的长度
// BLANK: CTC 中表示空字符的索引

template<typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor> ctc_loss_cpu_template(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK) {
  // 声明负对数似然和对数 alpha
  Tensor neg_log_likelihood, log_alpha;
  size_t tg_target_stride;
  std::vector<int64_t> tg_batch_offsets;

  // 根据目标数据类型选择不同的实现方式
  if (targets.scalar_type() == kLong) {
    // 使用 long 类型的实现方式
    std::tie(neg_log_likelihood, log_alpha, tg_target_stride, tg_batch_offsets) =
        ctc_loss_allocate_outputs<scalar_t, kLong>(
            log_probs, targets, input_lengths, target_lengths, BLANK);
  } else {
    // 使用 int 类型的实现方式
    std::tie(neg_log_likelihood, log_alpha, tg_target_stride, tg_batch_offsets) =
        ctc_loss_allocate_outputs<scalar_t, kInt>(
            log_probs, targets, input_lengths, target_lengths, BLANK);
  }

  // 获取 batch_size
  int64_t batch_size = log_probs.size(1);
  // 对 log_probs 进行维度置换，维度变为 batch_size x input_len x num_labels
  auto lpp  = log_probs.permute({1,0,2});
  auto log_probs_a_global = lpp.accessor<const scalar_t, 3>();
  auto log_alpha_a_global = log_alpha.accessor<scalar_t, 3>();
  auto targets_data = targets.const_data_ptr<target_t>();
  auto neg_log_likelihood_a = neg_log_likelihood.accessor<scalar_t, 1>();

  // 开始 alpha 的计算，处理第一行的 alpha_1
  // 首先将第一行的 log_alpha 填充为负无穷
  log_alpha.narrow(1, 0, 1).fill_(neginf);
  // 使用并行计算进行迭代处理每个 batch 中的数据
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    // 这里进行具体的 alpha 计算
    // ...
  });

  // 返回计算结果，包括负对数似然和计算出的 alpha
  return std::make_tuple(neg_log_likelihood, log_alpha);
}

// 开始模板函数的反向传播部分，处理 CTC 损失计算的反向传播算法的后半部分
// a) 计算 beta，类似于前向传播中的 alpha 计算部分
// b) 收集每个激活的字符对应的梯度，并封装梯度
template<typename scalar_t, ScalarType target_scalar_type>
// 定义函数 ctc_loss_backward_cpu_template，计算 CTC 损失的反向传播
Tensor ctc_loss_backward_cpu_template(const Tensor& grad_out, const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths,
                                      const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK, bool zero_infinity) {
  // 定义常量 neginf，表示负无穷
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  // 使用 std::conditional 条件类型选择，target_t 表示目标数据类型，根据 target_scalar_type 的值确定
  using target_t = typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  // 获取 log_probs 张量的维度信息
  int64_t max_input_length = log_probs.size(0);
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  // 创建一个与 log_probs 大小相同的张量 grad，并用 neginf 填充，内存布局为 LEGACY_CONTIGUOUS_MEMORY_FORMAT
  Tensor grad = at::full_like(log_probs, neginf, LEGACY_CONTIGUOUS_MEMORY_FORMAT); // at this point, this is log of empty sum

  // 管理性配置。在此处我们不做太多检查，假定前向传播已经完成了检查。
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t tg_target_stride;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t max_target_length;
  // 创建一个整型向量 tg_batch_offsets，大小为 batch_size
  std::vector<int64_t> tg_batch_offsets(batch_size);

  // 如果 targets 张量的维度为 1，表示目标是连接在一起的
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    max_target_length = 0;
    // 遍历 batch_size 范围内的每个索引 i
    for (const auto i : c10::irange(batch_size)) {
      tg_batch_offsets[i] = pos;
      pos += target_lengths[i];
      // 更新 max_target_length，记录最大的目标长度
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    // 获取 targets 张量的步长 tg_target_stride
    tg_target_stride = targets.stride(0);
  }
  else { // batch x max_target_length
    // 维度为 2 的情况，目标以 batch x max_target_length 的形式给出
    int64_t tg_batch_stride = targets.stride(0);
    // 计算每个 batch 的偏移量，存储在 tg_batch_offsets 中
    for (const auto i : c10::irange(batch_size)) {
      tg_batch_offsets[i] = i * tg_batch_stride;
    }
    // 获取 targets 张量的步长 tg_target_stride 和最大长度 max_target_length
    tg_target_stride = targets.stride(1);
    max_target_length = targets.size(1);
  }

  // 创建一个与 log_alpha 张量大小相同的空张量 log_beta，内存布局为 LEGACY_CONTIGUOUS_MEMORY_FORMAT
  Tensor log_beta = at::empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);  // could be optimized to use only 2 rows
  // 对 log_probs 进行维度变换，调整顺序为 {1, 0, 2}
  auto lpp  = log_probs.permute({1,0,2});
  // 获取 lpp 张量的访问器，类型为 const scalar_t，三维，顺序为 {batch_size, max_input_length, num_labels}
  auto log_probs_a_global = lpp.accessor<const scalar_t, 3>();
  // 获取 log_alpha 张量的访问器，类型为 const scalar_t，三维
  auto log_alpha_a_global = log_alpha.accessor<const scalar_t, 3>();
  // 获取 log_beta 张量的访问器，类型为 scalar_t，三维
  auto log_beta_a_global = log_beta.accessor<scalar_t, 3>();
  // 对 grad 进行维度变换，调整顺序为 {1, 0, 2}
  auto gp = grad.permute({1,0,2});
  // 获取 gp 张量的访问器，类型为 scalar_t，三维
  auto grad_a_global = gp.accessor<scalar_t, 3>();
  // 获取 targets 数据的指针，类型为 target_t
  auto targets_data = targets.const_data_ptr<target_t>();
  // 获取 grad_out 张量的访问器，类型为 const scalar_t，一维
  auto grad_out_a = grad_out.accessor<const scalar_t, 1>();

  // 创建函数 create_fill_iterator，用于填充张量，并返回填充迭代器
  auto create_fill_iterator = [](const Tensor& tensor, IntArrayRef squash_dims) {
    return TensorIteratorConfig()
        .set_check_mem_overlap(false)  // Fill is idempotent, so overlap is okay
        .check_all_same_dtype(false)
        .add_output(tensor)
        .resize_outputs(false)
        .declare_static_shape(tensor.sizes(), squash_dims)
        .build();
  };
  // 创建 fill_iter 迭代器，用于填充 grad 张量，维度 squash_dims 为 1
  const auto fill_iter = create_fill_iterator(grad, /*squash_dims=*/1);
  // 创建 fill_1d_iter 迭代器，用于填充 grad 张量，维度 squash_dims 为 {0, 1}
  const auto fill_1d_iter = create_fill_iterator(grad, /*squash_dims=*/{0, 1});
  // 创建 fill_log_beta_1d_iter 迭代器，用于填充 log_beta 张量，维度 squash_dims 为 {0, 1}
  const auto fill_log_beta_1d_iter = create_fill_iterator(log_beta, /*squash_dims=*/{0, 1});

  // 并行处理 batch_size 范围内的索引，使用 lambda 表达式作为并行处理的函数体
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    // 创建本地 fill_iter_local 迭代器，用于填充操作
    TensorIterator fill_iter_local(fill_iter);
    # 创建一个名为 fill_1d_iter_local 的 TensorIterator 对象，使用 fill_1d_iter 初始化
    TensorIterator fill_1d_iter_local(fill_1d_iter);
    # 创建一个名为 fill_log_beta_1d_iter_local 的 TensorIterator 对象，使用 fill_log_beta_1d_iter 初始化
    TensorIterator fill_log_beta_1d_iter_local(fill_log_beta_1d_iter);
    
    # 结束 lambda 表达式的定义，lambda 表达式之后的代码将在其调用时执行
    }
    
    # 返回变量 grad 的值作为函数的结果
    return grad;
}

} // namespace



std::tuple<Tensor, Tensor> ctc_loss_meta(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, bool zero_infinity) {
  (void)zero_infinity; // only used for backwards
  // 根据 log_probs 的数据类型分发计算 CTC 损失的元数据，返回负对数似然和对数 alpha 值
  return AT_DISPATCH_FLOATING_TYPES(
      log_probs.scalar_type(), "ctc_loss_meta", [&] {
        Tensor neg_log_likelihood, log_alpha;
        if (targets.scalar_type() == kLong) {
          std::tie(neg_log_likelihood, log_alpha, std::ignore, std::ignore) =  ctc_loss_allocate_outputs<scalar_t, kLong>(
              log_probs, targets, input_lengths, target_lengths, BLANK);
        } else {
          std::tie(neg_log_likelihood, log_alpha, std::ignore, std::ignore) = ctc_loss_allocate_outputs<scalar_t, kInt>(
              log_probs, targets, input_lengths, target_lengths, BLANK);
        }
        return std::make_tuple(neg_log_likelihood, log_alpha);
      });
}



std::tuple<Tensor, Tensor> ctc_loss_cpu(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, bool zero_infinity) {
  (void)zero_infinity; // only used for backwards
  // 根据 log_probs 的数据类型分发计算 CTC 损失（CPU 版本），返回负对数似然和对数 alpha 值
  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_cpu", [&] {
      if (targets.scalar_type() == kLong) {
        return ctc_loss_cpu_template<scalar_t, kLong>(log_probs, targets, input_lengths, target_lengths, BLANK);
      } else {
        return ctc_loss_cpu_template<scalar_t, kInt>(log_probs, targets, input_lengths, target_lengths, BLANK);
      }
  });
}



std::tuple<Tensor, Tensor> ctc_loss_tensor(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK, bool zero_infinity) {
  TORCH_CHECK(isIntegralType(input_lengths.scalar_type(), /*includeBool=*/false), "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), /*includeBool=*/false), "target_lengths must be integral");

  // 将输入长度和目标长度转换为长整型张量，保证在 CPU 上连续存储
  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  // 创建整数数组引用，用于后续计算 CTC 损失
  IntArrayRef il(ilc.const_data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.const_data_ptr<int64_t>(), tlc.numel());

  // 调用 PyTorch 的 C++ 实现计算 CTC 损失，返回负对数似然和对数 alpha 值
  return at::_ctc_loss(log_probs, targets, il, tl, BLANK, zero_infinity);
}
// 计算 CTC 损失函数的梯度，针对 CPU 实现
Tensor ctc_loss_backward_cpu(const Tensor& grad, const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths,
                             const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK, bool zero_infinity) {
  // 使用 AT_DISPATCH_FLOATING_TYPES 宏根据 log_probs 的数据类型分发函数调用
  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_backward_cpu", [&] {
      // 根据 targets 的数据类型选择合适的模板函数进行调用
      if (targets.scalar_type() == kLong) {
        // 调用具体的模板函数处理长整型数据类型的情况
        return ctc_loss_backward_cpu_template<scalar_t,kLong>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
      } else {
        // 调用具体的模板函数处理整型数据类型的情况
        return ctc_loss_backward_cpu_template<scalar_t,kInt>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
      }
  });
}

// 计算 CTC 损失函数的梯度，支持张量输入
Tensor ctc_loss_backward_tensor(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t BLANK,
    bool zero_infinity) {
  // 检查 input_lengths 和 target_lengths 的数据类型必须为整型
  TORCH_CHECK(
      isIntegralType(input_lengths.scalar_type(), /*includeBool=*/false),
      "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), /*includeBool=*/false), "target_lengths must be integral");

  // 将 input_lengths 和 target_lengths 转换为长整型张量，并确保在 CPU 上连续存储
  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  // 创建 IntArrayRef 引用以便在后续调用中使用
  IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
  // 调用 ATen 的 C++ 后端函数计算 CTC 损失的梯度
  return at::_ctc_loss_backward(grad, log_probs, targets, il, tl, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
}

// 用于获取经过截断处理的目标长度的张量
Tensor get_clamped_target_length(
    IntArrayRef target_lengths,
    const TensorOptions& options) {
  // 创建一个张量并对其进行截断处理，保证长度不小于1
  return at::tensor(target_lengths, options).clamp_min(1);
}

// 用于获取经过截断处理的目标长度的张量（重载函数，输入参数为张量而非数组引用）
Tensor get_clamped_target_length(
    const Tensor & target_lengths,
    const TensorOptions& options) {
  // 对目标长度张量进行截断处理，保证长度不小于1
  return target_lengths.clamp_min(1);
}

// CTC 损失函数实现的模板函数，用于选择不同的实现方式（CPU 或者 CUDA），并处理梯度和损失的返回
template <typename LengthsType>
Tensor ctc_loss_impl(const Tensor& log_probs_, const Tensor& targets, LengthsType input_lengths, LengthsType target_lengths, int64_t BLANK, int64_t reduction, bool zero_infinity) {
  // 检查 log_probs_ 的维度是否为3，判断是否为批处理模式
  auto is_batched = log_probs_.dim() == 3;
  // 如果不是批处理模式，对 log_probs_ 进行unsqueeze操作以支持批处理
  Tensor log_probs = is_batched ? log_probs_ : log_probs_.unsqueeze(1);
  // 判断是否可以使用 cuDNN 进行 CTC 损失的计算
  bool use_cudnn =
      (log_probs.device().type() == at::kCUDA) &&
      at::_use_cudnn_ctc_loss(
          log_probs, targets, input_lengths, target_lengths, BLANK);

  Tensor res;
  if (use_cudnn) {
    // 对 cuDNN 的非确定性 CTC 损失进行禁用，因为结果不一致
    // TODO: Implement non-deterministic CTC loss on cuDNN
    // 如果使用 CuDNN 实现的 CTC loss，参考链接: https://github.com/pytorch/pytorch/issues/21680
    res = std::get<0>(at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, BLANK, /*deterministic=*/true, zero_infinity));
  } else {
    // 如果目标张量在 CPU 上（CuDNN 需要），将其移动到当前 log_probs 的设备上，以便用户可以使用 GPU 加速
    res = std::get<0>(at::_ctc_loss(
        log_probs,
        targets.to(log_probs.device(), kLong),
        input_lengths,
        target_lengths,
        BLANK,
        zero_infinity));
    // 如果 zero_infinity 标志为 true，则将结果中无穷大的元素替换为零
    if (zero_infinity) {
      res = at::where(res == Scalar(std::numeric_limits<double>::infinity()), at::zeros({}, res.options()), res);
    }
  }
  // 根据 reduction 参数进行结果处理
  if (reduction == at::Reduction::Mean) {
    // 获取修正后的目标长度张量
    auto target_lengths_t = get_clamped_target_length(target_lengths, res.options());
    // 返回平均值
    return (res / target_lengths_t).mean();
  } else if (reduction == at::Reduction::Sum) {
    // 返回总和
    return res.sum();
  }
  // 如果是批处理模式，返回结果移除第一个维度后的张量
  return is_batched ? std::move(res) : res.squeeze(0);
} // namespace



} // namespace

这行表示前一个命名空间的结束。


Tensor ctc_loss(const Tensor& log_probs_, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, int64_t reduction, bool zero_infinity) {
  return ctc_loss_impl(log_probs_, targets, input_lengths, target_lengths, BLANK, reduction, zero_infinity);
}

定义了一个接受引用参数的函数 `ctc_loss`，用于计算CTC（Connectionist Temporal Classification）损失。


Tensor ctc_loss(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK, int64_t reduction, bool zero_infinity) {
  // we don't want to convert to IntArrayRef if we can dispatch to cuDNN (this allows graph-capturable ctc_loss)
  bool use_cudnn =
      (log_probs.device().type() == at::kCUDA) &&
      at::_use_cudnn_ctc_loss(
          log_probs, targets, input_lengths, target_lengths, BLANK);
  if (at::areAnyTensorSubclassLike(
          {log_probs, targets, input_lengths, target_lengths}) || use_cudnn) {
    // Composite Compliant path for TensorSubclasses
    return ctc_loss_impl(log_probs, targets, input_lengths, target_lengths, BLANK, reduction, zero_infinity);
  }
  // Fast path (which accesses data_ptr) and less operator dispatches for
  // regular tensors
  TORCH_CHECK(isIntegralType(input_lengths.scalar_type(), /*includeBool=*/false), "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), /*includeBool=*/false), "target_lengths must be integral");

  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  IntArrayRef il(ilc.const_data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.const_data_ptr<int64_t>(), tlc.numel());
  return at::native::ctc_loss(log_probs, targets, il, tl, BLANK, reduction, zero_infinity);
}

定义了另一个重载的 `ctc_loss` 函数，接受 `Tensor` 类型的参数，并根据条件选择不同的执行路径。


} // at::native

结束了 `at::native` 命名空间的定义。
```