# `.\pytorch\aten\src\ATen\native\cpu\FusedAdamKernel.cpp`

```py
// 定义宏，仅允许使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含头文件，引入需要的依赖
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedAdam.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>

// 进入 at::native 命名空间
namespace at::native {

// 匿名命名空间，用于封装内部实现
namespace {

// 模板函数，根据 scalar_t 类型和 adam_mode 进行特化
template <typename scalar_t, typename opmath_t, ADAM_MODE adam_mode>
typename std::enable_if<
    std::is_same<scalar_t, Half>::value || std::is_same<scalar_t, BFloat16>::value,
    void>::
    type inline adam_math(
  // 函数参数列表开始
  scalar_t* param_ptr,                      // 参数指针
  scalar_t* exp_avg_ptr,                   // 指数加权平均参数的指针
  scalar_t* exp_avg_sq_ptr,                // 指数加权平方平均参数的指针
  scalar_t* grad_ptr,                      // 梯度指针
  scalar_t* max_exp_avg_sq_ptr,            // 最大指数加权平方平均参数的指针
  double lr,                               // 学习率
  double bias_correction1,                 // 偏置校正1
  double bias_correction2,                 // 偏置校正2
  double exp_avg_grad_coefficient,         // 指数加权梯度系数
  double exp_avg_sq_grad_coefficient,      // 指数加权平方梯度系数
  double bias_correction2_sqrt,            // 偏置校正2的平方根
  double eps,                              // 微小值
  double weight_decay,                     // 权重衰减
  double beta2,                            // Beta2 参数
  bool amsgrad,                            // 是否启用 AMSGrad
  bool maximize,                           // 是否最大化
  const float* grad_scale_ptr,             // 梯度缩放因子指针
  int64_t size                             // 大小
){
  // 计算步长大小
  double step_size = lr / bias_correction1;

  // 使用 at::vec::Vectorized 封装标量类型 scalar_t 和 opmath_t
  using lpVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<opmath_t>;

  // 初始化向量化操作所需的向量和标量
  lpVec grad_vec_to_store;
  int64_t d = 0;
  fVec param_vec1, param_vec2;
  fVec grad_vec1, grad_vec2;
  fVec exp_avg_vec1, exp_avg_vec2;
  fVec exp_avg_sq_vec1, exp_avg_sq_vec2;
  fVec max_exp_avg_sq_vec1, max_exp_avg_sq_vec2;

  // 进行向量化操作，处理每个元素直到剩余元素小于向量大小
  for (; d < size - (size % lpVec::size()); d += lpVec::size()) {
    // 加载参数并转换为浮点数向量
    lpVec param_lpvec = lpVec::loadu(param_ptr + d);
    std::tie(param_vec1, param_vec2) = vec::convert_to_float<scalar_t>(param_lpvec);

    // 加载梯度并转换为浮点数向量
    lpVec grad_lpvec = lpVec::loadu(grad_ptr + d);
    std::tie(grad_vec1, grad_vec2) = vec::convert_to_float<scalar_t>(grad_lpvec);

    // 如果提供了梯度缩放因子，则进行缩放
    if (grad_scale_ptr) {
      grad_vec1 = grad_vec1 / fVec(float(*grad_scale_ptr));
      grad_vec2 = grad_vec2 / fVec(float(*grad_scale_ptr));
      grad_vec_to_store = vec::convert_from_float<scalar_t>(grad_vec1, grad_vec2);
      grad_vec_to_store.store(grad_ptr + d);
    }

    // 如果需要最大化梯度，则取反
    if (maximize) {
      grad_vec1 = grad_vec1 * fVec(opmath_t(-1.0));
      grad_vec2 = grad_vec2 * fVec(opmath_t(-1.0));
    }

    // 如果有权重衰减，则根据不同的 adam_mode 进行处理
    if (weight_decay != 0.f) {
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_vec1 += param_vec1 * fVec(opmath_t(weight_decay));
        grad_vec2 += param_vec2 * fVec(opmath_t(weight_decay));
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_vec1 = param_vec1 * fVec(opmath_t(1 - lr * weight_decay));
        param_vec2 = param_vec2 * fVec(opmath_t(1 - lr * weight_decay));
      }
    }

    // 加载指数加权平均值并转换为浮点数向量
    lpVec exp_avg_lpvec = lpVec::loadu(exp_avg_ptr + d);
    std::tie(exp_avg_vec1, exp_avg_vec2) = vec::convert_to_float<scalar_t>(exp_avg_lpvec);

    // 计算 exp_avg.lerp_(grad, 1 - beta1) 的逻辑
    const fVec lerp_weight = fVec(opmath_t(exp_avg_grad_coefficient));
    auto mask = lerp_weight.abs() < fVec(0.5);
    auto coeff = fVec::blendv(lerp_weight - fVec(1), lerp_weight, mask);
    auto base1 = fVec::blendv(grad_vec1, exp_avg_vec1, mask);


此处为代码的一部分，稍后将继续添加注释。
    // 计算指数加权平均向量1
    exp_avg_vec1 = vec::fmadd(coeff, grad_vec1 - exp_avg_vec1, base1);

    // 使用掩码选择更新基准向量2
    auto base2 = fVec::blendv(grad_vec2, exp_avg_vec2, mask);
    // 计算指数加权平均向量2
    exp_avg_vec2 = vec::fmadd(coeff, grad_vec2 - exp_avg_vec2, base2);

    // 加载指数加权平方平均向量
    lpVec exp_avg_sq_lpvec = lpVec::loadu(exp_avg_sq_ptr + d);
    // 将加载的向量转换为浮点数类型
    std::tie(exp_avg_sq_vec1, exp_avg_sq_vec2) = vec::convert_to_float<scalar_t>(exp_avg_sq_lpvec);
    // 更新指数加权平方平均向量1
    exp_avg_sq_vec1 = exp_avg_sq_vec1 * fVec(opmath_t(beta2)) +
        fVec(opmath_t(exp_avg_sq_grad_coefficient)) * grad_vec1 * grad_vec1;
    // 更新指数加权平方平均向量2
    exp_avg_sq_vec2 = exp_avg_sq_vec2 * fVec(opmath_t(beta2)) +
        fVec(opmath_t(exp_avg_sq_grad_coefficient)) * grad_vec2 * grad_vec2;

    // 将浮点数类型的向量转换回原始类型并存储到内存
    vec::convert_from_float<scalar_t>(exp_avg_vec1, exp_avg_vec2).store(exp_avg_ptr + d);
    // 将浮点数类型的向量转换回原始类型并存储到内存
    vec::convert_from_float<scalar_t>(exp_avg_sq_vec1, exp_avg_sq_vec2).store(exp_avg_sq_ptr + d);

    // 初始化分母向量1和向量2
    fVec denom_vec1, denom_vec2;
    // 如果使用AMSGrad算法
    if (amsgrad) {
      // 加载最大指数加权平方平均向量
      lpVec max_exp_avg_sq_lpvec = lpVec::loadu(max_exp_avg_sq_ptr + d);
      // 将加载的向量转换为浮点数类型
      std::tie(max_exp_avg_sq_vec1, max_exp_avg_sq_vec2) = vec::convert_to_float<scalar_t>(max_exp_avg_sq_lpvec);
      // 更新最大指数加权平方平均向量1
      max_exp_avg_sq_vec1 = maximum(max_exp_avg_sq_vec1, exp_avg_sq_vec1);
      // 更新最大指数加权平方平均向量2
      max_exp_avg_sq_vec2 = maximum(max_exp_avg_sq_vec2, exp_avg_sq_vec2);
      // 将浮点数类型的向量转换回原始类型并存储到内存
      vec::convert_from_float<scalar_t>(max_exp_avg_sq_vec1, max_exp_avg_sq_vec2).store(max_exp_avg_sq_ptr + d);
      // 计算分母向量1
      denom_vec1 =
          (max_exp_avg_sq_vec1.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
      // 计算分母向量2
      denom_vec2 =
          (max_exp_avg_sq_vec2.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
    } else {
      // 计算分母向量1
      denom_vec1 =
          (exp_avg_sq_vec1.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
      // 计算分母向量2
      denom_vec2 =
          (exp_avg_sq_vec2.sqrt() / fVec(opmath_t(bias_correction2_sqrt))) + fVec(opmath_t(eps));
    }

    // 更新参数向量1
    param_vec1 = param_vec1 + fVec(opmath_t(-step_size)) * exp_avg_vec1 / denom_vec1;
    // 更新参数向量2
    param_vec2 = param_vec2 + fVec(opmath_t(-step_size)) * exp_avg_vec2 / denom_vec2;
    // 将浮点数类型的向量转换回原始类型并存储到内存
    vec::convert_from_float<scalar_t>(param_vec1, param_vec2).store(param_ptr + d);
  }

  // 处理剩余的参数值
  scalar_t grad_val_to_store;
  for (; d < size; d++) {
    // 获取梯度和参数值
    opmath_t grad_val = grad_ptr[d];
    opmath_t param_val = param_ptr[d];
    // 如果有梯度缩放，则进行缩放
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / float(*grad_scale_ptr);
      grad_val_to_store = scalar_t(grad_val);
      grad_ptr[d] = grad_val_to_store;
    }
    // 如果需要最大化，则取相反数
    if (maximize) grad_val = -grad_val;
    // 如果有权重衰减
    if (weight_decay != 0.f){
      // 根据Adam模式应用权重衰减
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_val += param_val * opmath_t(weight_decay);
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_val = param_val * opmath_t(1 - lr * weight_decay);
      }
    }
    // 计算指数加权移动平均
    opmath_t exp_avg_var = exp_avg_ptr[d];
    // 检查指数加权梯度系数的权重是否小于0.5
    auto is_lerp_weight_small = std::abs(opmath_t(exp_avg_grad_coefficient)) < opmath_t(0.5);
    # 如果当前的 LERP 权重较小
    if (is_lerp_weight_small) {
        # 更新指数加权平均方差 exp_avg_var
        exp_avg_var = exp_avg_var + opmath_t(exp_avg_grad_coefficient) * (grad_val - exp_avg_var);
    } else {
        # 更新指数加权平均方差 exp_avg_var，考虑衰减系数
        exp_avg_var = grad_val - (grad_val - exp_avg_var) * (opmath_t(1) - opmath_t(exp_avg_grad_coefficient));
    }
    # 将 exp_avg_var 转换为标量类型，并存储到 exp_avg_ptr 数组中的索引 d 处
    exp_avg_ptr[d] = scalar_t(exp_avg_var);
    
    # 获取当前指数加权平均平方变量 exp_avg_sq_var 的值
    opmath_t exp_avg_sq_var = exp_avg_sq_ptr[d];
    # 更新指数加权平均平方变量 exp_avg_sq_var，乘以 beta2
    exp_avg_sq_var = exp_avg_sq_var * opmath_t(beta2);
    # 添加梯度的平方乘以 exp_avg_sq_grad_coefficient 到 exp_avg_sq_var
    exp_avg_sq_var = exp_avg_sq_var +
        opmath_t(exp_avg_sq_grad_coefficient) * grad_val * grad_val;
    # 将更新后的 exp_avg_sq_var 转换为标量类型，并存储到 exp_avg_sq_ptr 数组中的索引 d 处
    exp_avg_sq_ptr[d] = scalar_t(exp_avg_sq_var);
    
    # 定义变量 demon_val
    opmath_t demon_val;
    # 如果启用了 AMSGrad
    if (amsgrad) {
        # 获取当前的 max_exp_avg_sq_var 值
        opmath_t max_exp_avg_sq_var = max_exp_avg_sq_ptr[d];
        # 更新 max_exp_avg_sq_var，取当前值与 exp_avg_sq_var 的较大者
        max_exp_avg_sq_var = std::max(max_exp_avg_sq_var, exp_avg_sq_var);
        # 将更新后的 max_exp_avg_sq_var 转换为标量类型，并存储到 max_exp_avg_sq_ptr 数组中的索引 d 处
        max_exp_avg_sq_ptr[d] =
            scalar_t(max_exp_avg_sq_var);
        # 计算 demon_val，包括对 max_exp_avg_sq_var 的平方根、偏置校正项 bias_correction2_sqrt 和小值 eps 的操作
        demon_val =
            std::sqrt(max_exp_avg_sq_var) / opmath_t(bias_correction2_sqrt) + opmath_t(eps);
    } else {
        # 计算 demon_val，包括对 exp_avg_sq_var 的平方根、偏置校正项 bias_correction2_sqrt 和小值 eps 的操作
        demon_val = std::sqrt(exp_avg_sq_var) / opmath_t(bias_correction2_sqrt) + opmath_t(eps);
    }
    # 更新 param_ptr 数组中索引 d 处的参数值，考虑学习率 step_size、exp_avg_var 和 demon_val 的比率
    param_ptr[d] = param_val - opmath_t(step_size) * exp_avg_var / demon_val;
// 结束函数模板定义
template <typename scalar_t, typename opmath_t, ADAM_MODE adam_mode>
// 当 scalar_t 是 float 或 double 时，函数返回类型为 void
typename std::enable_if<
    std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value,
    void>::
    type inline adam_math(
  // 参数: 指向参数的指针
  scalar_t* param_ptr,
  // 参数: 指向 exp_avg 的指针
  scalar_t* exp_avg_ptr,
  // 参数: 指向 exp_avg_sq 的指针
  scalar_t* exp_avg_sq_ptr,
  // 参数: 指向梯度的指针
  scalar_t* grad_ptr,
  // 参数: 指向 max_exp_avg_sq 的指针
  scalar_t* max_exp_avg_sq_ptr,
  // 学习率 lr
  double lr,
  // 偏差修正项1
  double bias_correction1,
  // 偏差修正项2
  double bias_correction2,
  // exp_avg 的梯度系数
  double exp_avg_grad_coefficient,
  // exp_avg_sq 的梯度系数
  double exp_avg_sq_grad_coefficient,
  // 偏差修正项2的平方根
  double bias_correction2_sqrt,
  // eps
  double eps,
  // 权重衰减
  double weight_decay,
  // beta2
  double beta2,
  // 是否使用 amsgrad
  bool amsgrad,
  // 是否最大化
  bool maximize,
  // 梯度缩放的指针
  const float* grad_scale_ptr,
  // 尺寸大小
  int64_t size
){
  // 计算步长
  double step_size = lr / bias_correction1;
  // 使用 at::vec::Vectorized 定义 Vec 类型
  using Vec = at::vec::Vectorized<scalar_t>;
  // 用于存储梯度向量化后的结果
  Vec grad_vec_to_store;
  // 循环处理每一个向量化的元素
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    // 加载参数向量
    Vec param_vec = Vec::loadu(param_ptr + d);
    // 加载梯度向量
    Vec grad_vec = Vec::loadu(grad_ptr + d);
    // 如果有梯度缩放指针，则对梯度进行缩放
    if (grad_scale_ptr) {
      grad_vec = grad_vec / Vec(scalar_t(*grad_scale_ptr));
      grad_vec_to_store = grad_vec;
      grad_vec_to_store.store(grad_ptr + d);
    }
    // 如果最大化标志为真，则取相反数
    if (maximize) grad_vec = grad_vec * Vec(scalar_t(-1.0));
    // 如果有权重衰减，则根据不同的 adam_mode 进行处理
    if (weight_decay != 0.f){
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad_vec += param_vec * Vec(scalar_t(weight_decay));
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param_vec = param_vec * Vec(scalar_t(1 - lr * weight_decay));
      }
    }
    // 加载 exp_avg 向量
    Vec exp_avg_vec = Vec::loadu(exp_avg_ptr + d);
    // 计算 exp_avg 的更新：exp_avg.lerp_(grad, 1 - beta1)
    const Vec lerp_weight = Vec(scalar_t(exp_avg_grad_coefficient));
    auto mask = lerp_weight.abs() < Vec(0.5);
    auto coeff = Vec::blendv(lerp_weight - Vec(1), lerp_weight, mask);
    auto base = Vec::blendv(grad_vec, exp_avg_vec, mask);
    exp_avg_vec = vec::fmadd(coeff, grad_vec - exp_avg_vec, base);

    // 加载 exp_avg_sq 向量，并更新它
    Vec exp_avg_sq_vec = Vec::loadu(exp_avg_sq_ptr + d) * Vec(scalar_t(beta2)) +
        Vec(scalar_t(exp_avg_sq_grad_coefficient)) * grad_vec * grad_vec;
    exp_avg_vec.store(exp_avg_ptr + d);
    exp_avg_sq_vec.store(exp_avg_sq_ptr + d);

    // 计算 denom 向量
    Vec denom_vec;
    if (amsgrad) {
      // 如果使用 amsgrad，则取 max_exp_avg_sq 和 exp_avg_sq 的最大值
      Vec max_exp_avg_sq_vec =
          maximum(Vec::loadu(max_exp_avg_sq_ptr + d), exp_avg_sq_vec);
      max_exp_avg_sq_vec.store(max_exp_avg_sq_ptr + d);
      denom_vec =
          (max_exp_avg_sq_vec.sqrt() / Vec(scalar_t(bias_correction2_sqrt))) + Vec(scalar_t(eps));
    } else {
      // 否则直接使用 exp_avg_sq
      denom_vec =
          (exp_avg_sq_vec.sqrt() / Vec(scalar_t(bias_correction2_sqrt))) + Vec(scalar_t(eps));
    }
    // 更新参数向量
    param_vec = param_vec + Vec(scalar_t(-step_size)) * exp_avg_vec / denom_vec;
    param_vec.store(param_ptr + d);
  }
  // 处理剩余的不足一个向量化尺寸的元素
  scalar_t grad_val_to_store;
  for (; d < size; d++) {
    // 加载单个元素的梯度值
    scalar_t grad_val = grad_ptr[d];
    // 如果有梯度缩放指针，则对梯度进行缩放
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / scalar_t(*grad_scale_ptr);
      grad_val_to_store = grad_val;
      grad_ptr[d] = grad_val_to_store;
    }
    // 如果最大化标志为真，则取相反数
    if (maximize) grad_val = -grad_val;
    // 省略的部分，未完全添加
    // 如果权重衰减不为零
    if (weight_decay != 0.f){
      // 根据不同的Adam模式进行处理
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        // 原始Adam模式下，加入权重衰减的梯度更新
        grad_val += param_ptr[d] * scalar_t(weight_decay);
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        // AdamW模式下，更新参数考虑了权重衰减
        param_ptr[d] = param_ptr[d] * scalar_t(1 - lr * weight_decay);
      }
    }
    
    // 使用指数加权平均进行参数更新
    // 判断是否lerp权重很小
    auto is_lerp_weight_small = std::abs(scalar_t(exp_avg_grad_coefficient)) < scalar_t(0.5);
    if (is_lerp_weight_small) {
      // 如果lerp权重很小，则使用简化的lerp更新方式
      exp_avg_ptr[d] = exp_avg_ptr[d] + scalar_t(exp_avg_grad_coefficient) * (grad_val - exp_avg_ptr[d]);
    } else {
      // 否则，使用标准的lerp更新方式
      exp_avg_ptr[d] = grad_val - (grad_val - exp_avg_ptr[d]) * (scalar_t(1) - scalar_t(exp_avg_grad_coefficient));
    }
    
    // 更新第二矩估计值
    exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] * scalar_t(beta2);
    exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] +
        scalar_t(exp_avg_sq_grad_coefficient) * grad_val * grad_val;
    
    // 计算分母的值
    scalar_t demon_val;
    if (amsgrad) {
      // 如果使用AMSGrad，更新最大第二矩估计值
      max_exp_avg_sq_ptr[d] =
          std::max(max_exp_avg_sq_ptr[d], exp_avg_sq_ptr[d]);
      demon_val =
          std::sqrt(max_exp_avg_sq_ptr[d]) / scalar_t(bias_correction2_sqrt) + scalar_t(eps);
    } else {
      // 否则，直接使用第二矩估计值
      demon_val = std::sqrt(exp_avg_sq_ptr[d]) / scalar_t(bias_correction2_sqrt) + scalar_t(eps);
    }
    
    // 更新参数
    param_ptr[d] = param_ptr[d] - scalar_t(step_size) * exp_avg_ptr[d] / demon_val;
}

// 定义模板函数，实现 fused Adam 算法的步骤
template <typename scalar_t, ADAM_MODE adam_mode>
void adam_fused_step_impl(
    // 参数1: 待更新的参数张量
    const at::Tensor& param,
    // 参数2: 梯度张量
    const at::Tensor& grad,
    // 参数3: 指数移动平均张量
    const at::Tensor& exp_avg,
    // 参数4: 指数移动平方平均张量
    const at::Tensor& exp_avg_sq,
    // 参数5: 可选的最大指数移动平方平均张量（如果启用了 AMSGrad）
    const at::Tensor& max_exp_avg_sq,
    // 参数6: 状态步数张量
    const at::Tensor& state_step,
    // 参数7: 学习率
    const double lr,
    // 参数8: Adam 算法的 beta1 参数
    const double beta1,
    // 参数9: Adam 算法的 beta2 参数
    const double beta2,
    // 参数10: 权重衰减
    const double weight_decay,
    // 参数11: 用于数值稳定性的 epsilon 参数
    const double eps,
    // 参数12: 是否使用 AMSGrad
    const bool amsgrad,
    // 参数13: 是否最大化优化目标
    const bool maximize,
    // 参数14: 梯度缩放因子指针
    const float* grad_scale_ptr) {

  // 使用 opmath_type 定义操作类型为 scalar_t
  using opmath_t = at::opmath_type<scalar_t>;

  // 获取当前步数
  double step = state_step.item<float>();

  // 获取各张量的数据指针
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* exp_avg_data = exp_avg.data_ptr<scalar_t>();
  scalar_t* exp_avg_sq_data = exp_avg_sq.data_ptr<scalar_t>();
  scalar_t* max_exp_avg_sq_data = amsgrad ? max_exp_avg_sq.data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_data = grad.data_ptr<scalar_t>();

  // 计算偏置修正项
  // 这里需要使用 double 类型以与非融合 Adam 一致
  double bias_correction1 = 1 - std::pow(beta1, step);
  double bias_correction2 = 1 - std::pow(beta2, step);

  // 计算指数移动平均系数和指数移动平方平均系数
  double exp_avg_grad_coefficient = 1 - beta1;
  double exp_avg_sq_grad_coefficient = 1 - beta2;

  // 计算二阶偏置修正项的平方根
  double bias_correction2_sqrt = std::sqrt(bias_correction2);

  // 定义缓存行大小和缓存行对齐的任务单元大小
  constexpr size_t cache_line_size = 64;
  constexpr int64_t cache_line_aligned_task_unit = cache_line_size / sizeof(scalar_t);

  // 计算需要处理的任务单元数量
  size_t num_units = divup(param.numel(), cache_line_aligned_task_unit);

  // 定义 lambda 函数执行 Adam 算法的数学操作
  auto adam_fn = [&](int64_t begin, int64_t end) {
        // 计算本地指针
        begin *= cache_line_aligned_task_unit;
        end = std::min(end * cache_line_aligned_task_unit, param.numel());
        scalar_t* param_ptr = param_data + begin;
        scalar_t* exp_avg_ptr = exp_avg_data + begin;
        scalar_t* exp_avg_sq_ptr = exp_avg_sq_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* max_exp_avg_sq_ptr = amsgrad ? max_exp_avg_sq_data + begin : nullptr;

        // 计算本地任务大小
        const int64_t size = end - begin;

        // 调用 adam_math 函数执行 Adam 算法的数学操作
        adam_math<scalar_t, opmath_t, adam_mode>(
          param_ptr,
          exp_avg_ptr,
          exp_avg_sq_ptr,
          grad_ptr,
          max_exp_avg_sq_ptr,
          lr,
          bias_correction1,
          bias_correction2,
          exp_avg_grad_coefficient,
          exp_avg_sq_grad_coefficient,
          bias_correction2_sqrt,
          eps,
          weight_decay,
          beta2,
          amsgrad,
          maximize,
          grad_scale_ptr,
          size
        );
      };

  // 使用并行化函数 parallel_for 执行任务单元的并行处理
  at::parallel_for(
      0, num_units, 0, adam_fn);
}

// 定义融合 Adam 算法的核心函数
void fused_adam_kernel(
    // 参数1: 待更新的参数张量
    const at::Tensor& param,
    // 参数2: 梯度张量
    const at::Tensor& grad,
    // 参数3: 指数移动平均张量
    const at::Tensor& exp_avg,
    // 参数4: 指数移动平方平均张量
    const at::Tensor& exp_avg_sq,
    // 参数5: 可选的最大指数移动平方平均张量（如果启用了 AMSGrad）
    const at::Tensor& max_exp_avg_sq,
    // 参数6: 状态步数张量
    const at::Tensor& state_step,
    // 参数7: 学习率
    const double lr,
    // 参数8: Adam 算法的 beta1 参数
    const double beta1,
    // 参数9: Adam 算法的 beta2 参数
    const double beta2,
    // 参数10: 权重衰减
    const double weight_decay,
    // 参数11: 用于数值稳定性的 epsilon 参数
    const double eps,
    // 参数12: 是否使用 AMSGrad
    const bool amsgrad,
    // 参数13: 是否最大化优化目标
    const bool maximize,
    // 参数14: 梯度缩放因子指针
    const float* grad_scale_ptr,
    // 参数15: 梯度缩放因子指针
    )
    // 定义一个常量 ADAM_MODE，表示 Adam 优化器的模式
    const ADAM_MODE adam_mode
      ) {
      // 创建一个连续存储的张量 grad_contiguous，用于存放梯度数据
      Tensor grad_contiguous = grad.contiguous();
      // 使用模板函数 AT_DISPATCH_FLOATING_TYPES_AND2，针对指定的浮点类型进行分发
      AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, param.scalar_type(), "fused_adam_kernel", [&] {
        // 根据 adam_mode 的值选择不同的 Adam 优化器模式
        if(adam_mode == ADAM_MODE::ORIGINAL){
          // 调用特定模式下的 Adam 优化器实现函数，传递相应的参数
          adam_fused_step_impl<scalar_t, ADAM_MODE::ORIGINAL>(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale_ptr);
        } else {
          // 调用另一种模式下的 Adam 优化器实现函数，传递相应的参数
          adam_fused_step_impl<scalar_t, ADAM_MODE::ADAMW>(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale_ptr);
        }
    
      });
}

}
// 注册融合 Adam 算子的调度函数指针
REGISTER_DISPATCH(fused_adam_stub, &fused_adam_kernel);
} // namespace at::native
```