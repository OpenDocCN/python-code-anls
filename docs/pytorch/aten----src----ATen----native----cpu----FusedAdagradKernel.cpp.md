# `.\pytorch\aten\src\ATen\native\cpu\FusedAdagradKernel.cpp`

```py
// 定义宏，用于限制仅声明方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量相关头文件
#include <ATen/core/Tensor.h>
// 包含并行计算支持
#include <ATen/Parallel.h>
// 包含数学运算类型
#include <ATen/OpMathType.h>
// 包含调度存根
#include <ATen/native/DispatchStub.h>
// 包含融合的Adagrad优化器实现
#include <ATen/native/FusedAdagrad.h>
// 包含调度器
#include <ATen/Dispatch.h>
// 包含向量化支持
#include <ATen/cpu/vec/vec.h>
// 包含向量化功能
#include <ATen/cpu/vec/functional.h>

// 开始at::native命名空间
namespace at::native {

// 匿名命名空间，局部函数和变量作用域
namespace {

// 模板函数，针对Half和BFloat16类型执行Adagrad优化
template <typename scalar_t, typename opmath_t>
typename std::enable_if<
    std::is_same<scalar_t, Half>::value || std::is_same<scalar_t, BFloat16>::value,
    void>::
    type inline adagrad_math(
  scalar_t* param_ptr,            // 参数指针
  scalar_t* grad_ptr,             // 梯度指针
  scalar_t* state_sum_ptr,        // 状态和指针
  const double clr,               // 学习率
  const double eps,               // 平滑项
  const double weight_decay,      // 权重衰减
  const bool maximize,            // 是否最大化
  const float* grad_scale_ptr,    // 梯度缩放指针
  int64_t size                    // 尺寸大小
){
  // 使用at::vec::Vectorized加载Half或BFloat16类型数据
  using lpVec = at::vec::Vectorized<scalar_t>;
  // 使用at::vec::Vectorized加载opmath_t类型数据
  using fVec = at::vec::Vectorized<opmath_t>;
  
  // 声明向量存储变量
  lpVec grad_vec_to_store;
  fVec param_vec1, param_vec2;
  fVec grad_vec1, grad_vec2;
  fVec state_sum_vec1, state_sum_vec2;
  
  // 遍历数据，处理向量化数据
  int64_t d = 0;
  for (; d < size - (size % lpVec::size()); d += lpVec::size()) {
    // 加载参数数据到lpVec
    lpVec param_lpvec = lpVec::loadu(param_ptr + d);
    // 将参数向量转换为浮点数向量
    std::tie(param_vec1, param_vec2) = vec::convert_to_float<scalar_t>(param_lpvec);
    
    // 加载梯度数据到lpVec
    lpVec grad_lpvec = lpVec::loadu(grad_ptr + d);
    // 将梯度向量转换为浮点数向量
    std::tie(grad_vec1, grad_vec2) = vec::convert_to_float<scalar_t>(grad_lpvec);
    
    // 如果有梯度缩放指针，则缩放梯度向量
    if (grad_scale_ptr) {
      grad_vec1 = grad_vec1 / fVec(float(*grad_scale_ptr));
      grad_vec2 = grad_vec2 / fVec(float(*grad_scale_ptr));
      // 将转换后的浮点数向量转回原始类型，存储到梯度指针
      grad_vec_to_store = vec::convert_from_float<scalar_t>(grad_vec1, grad_vec2);
      grad_vec_to_store.store(grad_ptr + d);
    }
    
    // 如果最大化标志位为真，则取反梯度向量
    if (maximize){
      grad_vec1 = grad_vec1 * fVec(opmath_t(-1.0));
      grad_vec2 = grad_vec2 * fVec(opmath_t(-1.0));
    }
    
    // 如果有权重衰减，则加上衰减后的参数向量
    if (weight_decay != 0.0){
      grad_vec1 += param_vec1 * fVec(scalar_t(weight_decay));
      grad_vec2 += param_vec2 * fVec(scalar_t(weight_decay));
    }
    
    // 加载状态和数据到浮点数向量
    std::tie(state_sum_vec1, state_sum_vec2) = vec::convert_to_float<scalar_t>(lpVec::loadu(state_sum_ptr + d));
    // 更新状态和数据，加上梯度平方和
    state_sum_vec1 += grad_vec1 * grad_vec1;
    state_sum_vec2 += grad_vec2 * grad_vec2;
    // 将浮点数向量转回原始类型，存储状态和数据
    vec::convert_from_float<scalar_t>(state_sum_vec1, state_sum_vec2).store(state_sum_ptr + d);

    // 计算标准差向量
    fVec std_vec1 = state_sum_vec1.sqrt() + fVec(scalar_t(eps));
    fVec std_vec2 = state_sum_vec2.sqrt() + fVec(scalar_t(eps));
    // 更新参数向量
    param_vec1 = param_vec1 - fVec(scalar_t(clr)) * grad_vec1 / std_vec1;
    param_vec2 = param_vec2 - fVec(scalar_t(clr)) * grad_vec2 / std_vec2;
    // 将浮点数向量转回原始类型，存储参数数据
    vec::convert_from_float<scalar_t>(param_vec1, param_vec2).store(param_ptr + d);
  }
  
  // 处理剩余的数据，非向量化处理
  scalar_t grad_val_to_store;
  for (; d < size; d++) {
    // 获取单个梯度值和参数值
    opmath_t grad_val = grad_ptr[d];
    opmath_t param_val = param_ptr[d];
    
    // 如果有梯度缩放指针，则缩放梯度值
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / opmath_t(*grad_scale_ptr);
      grad_val_to_store = grad_val;
      grad_ptr[d] = grad_val_to_store;
    }
    
    // 如果最大化标志位为真，则取反梯度值
    if (maximize) grad_val = -grad_val;
    
    // 如果有权重衰减，则加上衰减后的参数值
    if (weight_decay != 0.0){
      grad_val += param_val * opmath_t(weight_decay);
    }
    // 继续处理未向量化的参数数据
    # 获取指针 state_sum_ptr[d] 所指向的值，将 grad_val 的平方加到该值上
    opmath_t state_sum_val = state_sum_ptr[d];
    state_sum_val += grad_val * grad_val;
    # 更新 state_sum_ptr[d] 指向的值为新计算的 state_sum_val
    state_sum_ptr[d] = state_sum_val;
    # 计算标准差 std_val，即 state_sum_val 的平方根加上一个小的常数 eps
    opmath_t std_val = std::sqrt(state_sum_val) + opmath_t(eps);
    # 更新 param_ptr[d] 指向的值，使用参数 param_val 减去学习率 clr 乘以 grad_val 再除以 std_val
    param_val -= opmath_t(clr) * grad_val / std_val;
    # 更新 param_ptr[d] 指向的值为新计算的 param_val
    param_ptr[d] = param_val;
}
// 结束函数 adagrad_math 的实现

template <typename scalar_t, typename opmath_t>
typename std::enable_if<
    std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value,
    void>::
    type inline adagrad_math(
  scalar_t* param_ptr,
  scalar_t* grad_ptr,
  scalar_t* state_sum_ptr,
  const double clr,
  const double eps,
  const double weight_decay,
  const bool maximize,
  const float* grad_scale_ptr,
  int64_t size
){
  using Vec = at::vec::Vectorized<scalar_t>;
  // 使用 Vectorized 类型 Vec 加速向量操作

  Vec grad_vec_to_store;
  // 声明向量 grad_vec_to_store 用于存储梯度向量

  int64_t d = 0;
  // 初始化循环变量 d

  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    // 对于向量化大小的范围内循环

    Vec param_vec = Vec::loadu(param_ptr + d);
    // 从 param_ptr 加载向量化参数向量 param_vec

    Vec grad_vec = Vec::loadu(grad_ptr + d);
    // 从 grad_ptr 加载向量化梯度向量 grad_vec

    if (grad_scale_ptr) {
      // 如果存在梯度缩放指针 grad_scale_ptr

      grad_vec = grad_vec / Vec(scalar_t(*grad_scale_ptr));
      // 对梯度向量进行缩放

      grad_vec_to_store = grad_vec;
      // 将缩放后的梯度向量存储在 grad_vec_to_store 中

      grad_vec_to_store.store(grad_ptr + d);
      // 将存储的梯度向量写回 grad_ptr
    }

    if (maximize) grad_vec = grad_vec * Vec(scalar_t(-1.0));
    // 如果需要最大化，则对梯度向量进行取反操作

    if (weight_decay != 0.0){
      grad_vec += param_vec * Vec(scalar_t(weight_decay));
      // 如果权重衰减不为零，则将权重衰减项加到梯度向量上
    }

    Vec sum_vec = Vec::loadu(state_sum_ptr + d) + grad_vec * grad_vec;
    // 加载状态和向量，并计算梯度的平方和

    sum_vec.store(state_sum_ptr + d);
    // 将更新后的状态和向量写回 state_sum_ptr

    Vec std_vec = sum_vec.sqrt() + Vec(scalar_t(eps));
    // 计算标准差向量，加上平滑项 eps

    param_vec = param_vec - Vec(scalar_t(clr)) * grad_vec / std_vec;
    // 更新参数向量

    param_vec.store(param_ptr + d);
    // 将更新后的参数向量写回 param_ptr
  }

  scalar_t grad_val_to_store;
  // 声明标量 grad_val_to_store，用于存储梯度值

  for (; d < size; d++) {
    // 对于剩余的单个元素的范围进行循环

    scalar_t grad_val = grad_ptr[d];
    // 加载梯度值

    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / scalar_t(*grad_scale_ptr);
      // 如果存在梯度缩放指针，则进行梯度值缩放操作

      grad_val_to_store = grad_val;
      // 将缩放后的梯度值存储在 grad_val_to_store 中

      grad_ptr[d] = grad_val_to_store;
      // 将存储的梯度值写回 grad_ptr
    }

    if (maximize) grad_val = -grad_val;
    // 如果需要最大化，则对梯度值进行取反操作

    if (weight_decay != 0.0){
      grad_val += param_ptr[d] * scalar_t(weight_decay);
      // 如果权重衰减不为零，则将权重衰减项加到梯度值上
    }

    state_sum_ptr[d] += grad_val * grad_val;
    // 更新状态和向量的平方和

    scalar_t std_val = std::sqrt(state_sum_ptr[d]) + scalar_t(eps);
    // 计算标准差值，加上平滑项 eps

    param_ptr[d] -= scalar_t(clr) * grad_val / std_val;
    // 更新参数值
  }
}

template <typename scalar_t>
void adagrad_fused_step_impl(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& state_sum,
    const at::Tensor& state_step,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const float* grad_scale_ptr) {
```  

// 定义一个函数，该函数使用了模板类型opmath_t，它是scalar_t类型的at::opmath_type
using opmath_t = at::opmath_type<scalar_t>;
scalar_t* param_data = param.data_ptr<scalar_t>();  // 获取param张量的数据指针，类型为scalar_t
scalar_t* grad_data = grad.data_ptr<scalar_t>();    // 获取grad张量的数据指针，类型为scalar_t
scalar_t* state_sum_data = state_sum.data_ptr<scalar_t>();  // 获取state_sum张量的数据指针，类型为scalar_t
double step = state_step.item<float>();  // 获取state_step张量的浮点数值作为步长step
double clr = lr / (1.0 + (step - 1.0) * lr_decay);  // 计算学习率clr，根据步长step和lr_decay调整lr

constexpr size_t cache_line_size = 64;  // 定义缓存行大小为64字节
constexpr int64_t cache_line_aligned_task_unit = cache_line_size / sizeof(scalar_t);  // 计算缓存行对齐的任务单位大小，以scalar_t为单位
size_t num_units = divup(param.numel(), cache_line_aligned_task_unit);  // 计算并行任务单元数，保证每个任务单元在缓存行上对齐


auto adagrad_fn = [&](int64_t begin, int64_t end) {
    // local pointers
    begin *= cache_line_aligned_task_unit;  // 将任务单元索引转换为数据索引
    end = std::min(end * cache_line_aligned_task_unit, param.numel());  // 计算结束索引，不超过param张量的元素数
    scalar_t* param_ptr = param_data + begin;  // 获取param数据的部分指针，以缓存行对齐的方式
    scalar_t* grad_ptr = grad_data + begin;    // 获取grad数据的部分指针，以缓存行对齐的方式
    scalar_t* state_sum_ptr = state_sum_data + begin;  // 获取state_sum数据的部分指针，以缓存行对齐的方式

    const int64_t size = end - begin;  // 计算处理的数据块大小
    adagrad_math<scalar_t, opmath_t>(
      param_ptr,
      grad_ptr,
      state_sum_ptr,
      clr,
      eps,
      weight_decay,
      maximize,
      grad_scale_ptr,
      size
    );  // 调用adagrad_math函数处理数据块
  };


at::parallel_for(
    0, num_units, 0, adagrad_fn);  // 并行执行adagrad_fn函数，处理num_units个任务单元
}

// 定义一个 C++ 函数，执行融合的 Adagrad 算法
void fused_adagrad_kernel(
    const at::Tensor& param,                 // 输入参数：权重张量
    const at::Tensor& grad,                  // 输入参数：梯度张量
    const at::Tensor& state_sum,             // 输入参数：状态总和张量
    const at::Tensor& state_step,            // 输入参数：状态步数张量
    const double lr,                         // 输入参数：学习率
    const double lr_decay,                   // 输入参数：学习率衰减
    const double weight_decay,               // 输入参数：权重衰减
    const double eps,                        // 输入参数：平滑项 epsilon
    const bool maximize,                     // 输入参数：是否最大化优化目标
    const float* grad_scale_ptr              // 输入参数：梯度缩放因子指针
  ) {
  // 使梯度张量连续化
  Tensor grad_contiguous = grad.contiguous();
  // 根据参数的数据类型调度相应类型的 Adagrad 实现
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, param.scalar_type(), "fused_adagrad_kernel", [&] {
    // 调用具体的 Adagrad 实现函数
    adagrad_fused_step_impl<scalar_t>(
      param,
      grad,
      state_sum,
      state_step,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale_ptr);
  });
}

}

// 注册 fused_adagrad_kernel 函数到分发机制
REGISTER_DISPATCH(fused_adagrad_stub, &fused_adagrad_kernel);
} // namespace at::native
```