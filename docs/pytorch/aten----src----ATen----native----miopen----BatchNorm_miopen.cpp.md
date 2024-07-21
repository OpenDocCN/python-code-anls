# `.\pytorch\aten\src\ATen\native\miopen\BatchNorm_miopen.cpp`

```
// 定义预处理器宏，仅限于 Torch 断言和方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 库中的 Tensor 类
#include <ATen/core/Tensor.h>
// 引入 ATen 库的配置文件
#include <ATen/Config.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则引入以下标准函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，引入特定操作的头文件
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/miopen_batch_norm_native.h>
#include <ATen/ops/miopen_batch_norm_backward_native.h>
#endif

// 待办事项：完全移除对 AT_ROCM_ENABLED 的条件编译，不要将此文件包含在 CPU 构建中
#include <ATen/cuda/CUDAConfig.h>

// 如果未启用 ROCm，进入条件编译
#if !AT_ROCM_ENABLED()

// 进入 at::native 命名空间
namespace at { namespace native {

// 实现 MIOpen 批量归一化函数，当未启用 MIOpen 支持时报错
std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool training, double exponential_average_factor, double epsilon) {
  AT_ERROR("miopen_batch_norm: ATen not compiled with MIOpen support");
}

// 实现 MIOpen 批量归一化反向传播函数，当未启用 MIOpen 支持时报错
std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    double epsilon) {
  AT_ERROR("miopen_batch_norm_backward: ATen not compiled with MIOpen support");
}

}}  // namespace at::native

// 若启用 ROCm，进入此分支
#else // AT_ROCM_ENABLED

// 引入 MIOpen 相关的描述符、类型和工具头文件
#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>

// 引入 TensorUtils.h 头文件
#include <ATen/TensorUtils.h>

// 进入 at::native 命名空间
namespace at { namespace native {

// 匿名命名空间，定义 expandScale 函数
namespace {

// 根据给定的 Tensor 和维度 dim 扩展 scale
Tensor expandScale(const Tensor& t, int64_t dim) {
  // 创建初始大小为 [1, t.numel()] 的 size 向量
  std::vector<int64_t> size{ 1, t.numel() };
  // 若 size 大小小于 dim，则添加维度 1 直到大小为 dim
  while (static_cast<int64_t>(size.size()) < dim) {
    size.emplace_back(1);
  }
  // 将 Tensor t 重新视图为指定 size 的 Tensor
  return t.view(size);
}

}  // namespace

// 实现 MIOpen 批量归一化函数
std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt, const std::optional<Tensor>& running_mean_t_opt, const std::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  // 处理 bias 和 running_mean/running_var 可选值
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = c10::value_or_else(running_mean_t_opt, [] {return Tensor();});
  const Tensor& running_var_t = c10::value_or_else(running_var_t_opt, [] {return Tensor();});

  // 定义输入 TensorArg 和 CheckedFrom，用于参数校验
  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 };
  CheckedFrom c = "miopen_batch_norm";

  // 检查所有定义的参数是否已定义
  checkAllDefined(c, {input, weight, bias});
  // 若非训练模式，执行以下代码块
  if (!training) {
  # 检查 running_mean 和 running_var 是否都已定义
  checkAllDefined(c, {running_mean, running_var});

  # 检查 input, weight, bias, running_mean, running_var 是否都在同一个 GPU 上
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});

  # 如果 input 的数据类型不是半精度 (Half)，则检查 input 和 weight 的数据类型是否相同
  if (input->scalar_type() != ScalarType::Half) {
    checkAllSameType(c, {input, weight});
  }

  # 检查 weight, bias, running_mean, running_var 是否都具有相同的数据类型
  checkAllSameType(c, {weight, bias, running_mean, running_var});

  # 检查 weight, bias, running_mean, running_var 是否都是连续的张量
  checkAllContiguous(c, {weight, bias, running_mean, running_var});

  # 检查 input 张量是否按照其推荐的内存格式是连续的
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));

  # 检查 input 张量的维度范围是否在 2 到 6 之间（不包括 6）
  checkDimRange(c, input, 2, 6 /* exclusive */);

  # 获取 input 张量的第二个维度的大小，通常表示特征数
  auto num_features = input->size(1);

  # 遍历 weight, bias, running_mean, running_var 张量，并检查每个张量的元素数量是否等于 num_features
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  # 根据 input 张量的维度确定 miopenBatchNormMode_t 模式
  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  # 创建一个与 input 张量相同尺寸和选项的空张量 output_t
  auto output_t = at::empty(input->sizes(), input->options());
  TensorArg output{ output_t, "output", 0 };

  # 获取 MIOpen 的句柄
  auto handle = getMiopenHandle();

  # 获取 input 张量的数据类型
  auto dataType = getMiopenDataType(*input);

  # 创建 input 张量的描述符 idesc，维度为 4
  TensorDescriptor idesc{ *input, 4 };  // input descriptor

  # 创建 weight, bias, running_mean 等张量的描述符 wdesc，通过 expandScale 函数扩展到与 input 张量相同的维度，维度为 4
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.

  # 创建常数张量 one 和 zero，数据类型为 dataType，分别表示值 1 和 0
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  # 创建空的 save_mean 和 save_var 张量
  Tensor save_mean, save_var;

  # 如果是训练模式
  if (training) {
    # 获取 input 张量第二维度的大小
    int64_t num_features = input_t.size(1);

    # 创建与 weight_t 相同选项的 save_mean 和 save_var 张量，维度为 { num_features }
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());

    # 调用 miopenBatchNormalizationForwardTraining 函数执行批量归一化前向训练
    MIOPEN_CHECK(miopenBatchNormalizationForwardTraining(
      handle, mode, &one, &zero,
      idesc.desc(), input->const_data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      // NOTE: MIOpen docs say that the bnScale and bnBias args are only inputs,
      // not outputs. However, unfortunately the function signature only takes
      // non-const pointers, presumably by accident
      const_cast<void*>(weight->const_data_ptr()),
      const_cast<void*>(bias->const_data_ptr()),
      exponential_average_factor,
      at::maybe_data_ptr(running_mean),
      at::maybe_data_ptr(running_var),
      epsilon,
      save_mean.mutable_data_ptr(),
      save_var.mutable_data_ptr()));
  } else {
    # 创建空的 save_mean 和 save_var 张量，维度为 {0}
    save_mean = at::empty({0}, weight_t.options());
    save_var = at::empty({0}, weight_t.options());
    MIOPEN_CHECK(miopenBatchNormalizationForwardInference(
      handle, mode, &one, &zero,
      idesc.desc(), input->const_data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      // MIOpen文档指出，bnScale和bnBias参数仅为输入，而非输出。
      // 然而，不幸的是，函数签名只接受非const指针，可能是意外的。
      const_cast<void*>(weight->const_data_ptr()),
      const_cast<void*>(bias->const_data_ptr()),
      running_mean->data_ptr(),
      running_var->data_ptr(),
      epsilon));
  }



  // save_mean和save_var可能未定义
  // 如果这造成问题，我们可以将它们初始化为空的正确类型的张量
  return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
// 返回类型为包含三个 Tensor 的元组，表示 Miopen 批归一化的反向传播结果
std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input_t,                   // 输入 Tensor
    const Tensor& grad_output_t,             // 梯度输出 Tensor
    const Tensor& weight_t,                  // 权重 Tensor

    // 以下是可选参数，用于双向传播时访问，但当前未使用
    const optional<Tensor>& running_mean_opt,  // 可选的运行均值 Tensor
    const optional<Tensor>& running_var_opt,   // 可选的运行方差 Tensor
    const optional<Tensor>& save_mean_t_opt,   // 可选的保存均值 Tensor
    const optional<Tensor>& save_var_t_opt,    // 可选的保存方差 Tensor
    double epsilon) {                         // epsilon 参数用于数值稳定性

  // 根据是否存在可选参数，选择合适的 Tensor 或者创建一个空的 Tensor
  const Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return Tensor(); });
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });
  const Tensor& save_mean_t =
      c10::value_or_else(save_mean_t_opt, [] { return Tensor(); });
  const Tensor& save_var_t =
      c10::value_or_else(save_var_t_opt, [] { return Tensor(); });

  // 定义 TensorArg 对象，用于参数有效性检查
  TensorArg input{ input_t, "input", 1 },
            grad_output{ grad_output_t, "grad_output", 2 },
            weight{ weight_t, "weight", 3 },
            save_mean{ save_mean_t, "save_mean", 4 },
            save_var{ save_var_t, "save_var", 5 };
  CheckedFrom c = "miopen_batch_norm_backward";

  // 执行参数有效性检查
  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});

  // 根据输入的标量类型进行额外的类型检查
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  checkAllContiguous(c, {input, grad_output, save_mean, save_var});

  // 检查输入 Tensor 的维度范围
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkSameSize(c, input, grad_output);

  // 获取输入 Tensor 的特征数
  auto num_features = input->size(1);

  // 对 weight, save_mean, save_var 等 Tensor 进行元素数量检查
  for (auto t : {weight, save_mean, save_var}) {
    checkNumel(c, t, num_features);
  }

  // 根据输入 Tensor 的维度确定 Miopen 批归一化模式
  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  // 创建用于存储梯度的 Tensor
  auto grad_input_t  = at::empty(input->sizes(), input->options());
  auto grad_weight_t = at::empty(weight->sizes(), weight->options());
  auto grad_bias_t   = at::empty(weight->sizes(), weight->options());

  // 获取 Miopen 句柄和数据类型
  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);

  // 定义输入和权重 Tensor 的描述符
  TensorDescriptor idesc{ *input, 4 };  // 输入、输出和梯度输出的描述符
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // 权重、偏置、保存均值等的描述符

  // 定义常量值
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  // 调用 Miopen 批归一化反向传播函数
  MIOPEN_CHECK(miopenBatchNormalizationBackward(
    handle, mode, &one, &zero, &one, &zero,
    idesc.desc(), input->const_data_ptr(),
    idesc.desc(), grad_output->const_data_ptr(),
    idesc.desc(), grad_input_t.data_ptr(),
    wdesc.desc(), weight->const_data_ptr(),
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->const_data_ptr(),
  // 返回一个包含三个张量的元组，分别是梯度输入张量grad_input_t、梯度权重张量grad_weight_t、梯度偏置张量grad_bias_t
  return std::tuple<Tensor, Tensor, Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}}  // namespace native

#endif


注释：


// 结束了 native 命名空间的定义

#endif
// 结束了头文件的条件编译指令


这段代码是 C++ 中的结尾部分，其中：

- `}` 结束了 `native` 命名空间的定义。
- `}}` 结束了整个 `namespace native` 的定义，这里 `}}` 表示结束两层的命名空间。
- `// namespace native` 是单行注释，说明前面的 `}}` 是结束 `native` 命名空间的。
- `#endif` 是条件编译指令的结束，用于结束条件编译块，通常与 `#ifdef` 或 `#ifndef` 配对使用。
```