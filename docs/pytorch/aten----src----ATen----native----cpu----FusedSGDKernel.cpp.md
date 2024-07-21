# `.\pytorch\aten\src\ATen\native\cpu\FusedSGDKernel.cpp`

```py
// 定义宏以仅包含方法操作符的 Torch 断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 库中的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedSGD.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>

// ATen 库的 native 命名空间
namespace at::native {

// 匿名命名空间内的函数模板定义
namespace {

// 根据 scalar_t 和 opmath_t 类型的不同，启用不同的函数模板
template <typename scalar_t, typename opmath_t>
typename std::enable_if<
    std::is_same<scalar_t, Half>::value || std::is_same<scalar_t, BFloat16>::value,
    void>::
    type inline sgd_math(
  scalar_t* param_ptr,                // 参数的指针，类型为 scalar_t
  scalar_t* grad_ptr,                 // 梯度的指针，类型为 scalar_t
  scalar_t* momentum_buf_ptr,         // 动量缓冲区的指针，类型为 scalar_t
  const double weight_decay,          // 权重衰减因子
  const double momentum,              // 动量参数
  const double lr,                    // 学习率
  const double dampening,             // 阻尼参数
  const bool nesterov,                // 是否使用 Nesterov 动量
  const bool maximize,                // 是否最大化目标（最大化优化）
  const bool is_first_step,           // 是否第一步
  const float* grad_scale_ptr,        // 梯度缩放的指针
  int64_t size                        // 数据大小
){
  using lpVec = at::vec::Vectorized<scalar_t>; // 使用 ATen 的向量化类型 lpVec
  using fVec = at::vec::Vectorized<opmath_t>;  // 使用 ATen 的向量化类型 fVec

  lpVec grad_vec_to_store;          // 用于存储梯度向量的向量化变量
  fVec param_vec1, param_vec2;      // 参数向量
  fVec grad_vec1, grad_vec2;        // 梯度向量
  fVec momentum_buffer_vec1, momentum_buffer_vec2;  // 动量缓冲区向量

  int64_t d = 0;  // 循环索引初始化为 0
  for (; d < size - (size % lpVec::size()); d += lpVec::size()) {
    // 加载参数的向量化表示
    lpVec param_lpvec = lpVec::loadu(param_ptr + d);
    // 转换参数向量为 float 类型并存储在 param_vec1 和 param_vec2 中
    std::tie(param_vec1, param_vec2) = vec::convert_to_float<scalar_t>(param_lpvec);

    // 加载梯度的向量化表示
    lpVec grad_lpvec = lpVec::loadu(grad_ptr + d);
    // 转换梯度向量为 float 类型并存储在 grad_vec1 和 grad_vec2 中
    std::tie(grad_vec1, grad_vec2) = vec::convert_to_float<scalar_t>(grad_lpvec);

    // 如果提供了梯度缩放因子，则将梯度向量除以它
    if (grad_scale_ptr) {
      grad_vec1 = grad_vec1 / fVec(float(*grad_scale_ptr));
      grad_vec2 = grad_vec2 / fVec(float(*grad_scale_ptr));
      // 将转换后的梯度向量重新存储回原梯度数组
      grad_vec_to_store = vec::convert_from_float<scalar_t>(grad_vec1, grad_vec2);
      grad_vec_to_store.store(grad_ptr + d);
    }

    // 如果需要最大化目标，则将梯度向量取反
    if (maximize){
      grad_vec1 = grad_vec1 * fVec(opmath_t(-1.0));
      grad_vec2 = grad_vec2 * fVec(opmath_t(-1.0));
    }

    // 如果有权重衰减，则在梯度向量上应用权重衰减
    if (weight_decay != 0.0){
      grad_vec1 = vec::fmadd(param_vec1, fVec(scalar_t(weight_decay)), grad_vec1);
      grad_vec2 = vec::fmadd(param_vec2, fVec(scalar_t(weight_decay)), grad_vec2);
    }

    // 如果有动量，则根据是否第一步来应用动量
    if (momentum != 0.0) {
      fVec momentum_vec1, momentum_vec2;
      if (is_first_step) {
        momentum_vec1 = grad_vec1;
        momentum_vec2 = grad_vec2;
      } else {
        // 加载动量缓冲区的向量化表示并应用动量公式
        momentum_vec1 = fVec::loadu(momentum_buf_ptr + d) * fVec(scalar_t(momentum));
        momentum_vec2 = fVec::loadu(momentum_buf_ptr + d + fVec::size()) * fVec(scalar_t(momentum));
        momentum_vec1 = vec::fmadd(fVec(scalar_t(1 - dampening)), grad_vec1, momentum_vec1);
        momentum_vec2 = vec::fmadd(fVec(scalar_t(1 - dampening)), grad_vec2, momentum_vec2);
      }
      // 将转换后的动量向量存储回动量缓冲区
      vec::convert_from_float<scalar_t>(momentum_vec1, momentum_vec2).store(momentum_buf_ptr + d);

      // 如果使用 Nesterov 动量，则在梯度向量上应用 Nesterov 更新
      if (nesterov) {
        grad_vec1 = vec::fmadd(momentum_vec1, fVec(scalar_t(momentum)), grad_vec1);
        grad_vec2 = vec::fmadd(momentum_vec2, fVec(scalar_t(momentum)), grad_vec2);
      } else {
        grad_vec1 = momentum_vec1;
        grad_vec2 = momentum_vec2;
      }
    }


这段代码实现了一种优化算法（SGD），使用了 ATen 库中的向量化功能来加速数值计算，特别适用于处理半精度数据（如 Half 和 BFloat16 类型）。
  }
  // 结束第二层循环

  // 定义存储梯度值的变量
  scalar_t grad_val_to_store;

  // 开始遍历参数空间
  for (; d < size; d++) {
    // 获取梯度值和参数值
    opmath_t grad_val = grad_ptr[d];
    opmath_t param_val = param_ptr[d];

    // 如果存在梯度缩放因子，则重新计算梯度值
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / opmath_t(*grad_scale_ptr);
      grad_val_to_store = grad_val;
      grad_ptr[d] = grad_val_to_store;
    }

    // 如果需要最大化目标函数，则取负梯度值
    if (maximize) grad_val = -grad_val;

    // 如果存在权重衰减，则应用于梯度值
    if (weight_decay != 0.0){
      grad_val += param_val * opmath_t(weight_decay);
    }

    // 如果存在动量参数，则计算动量更新
    if (momentum != 0.0) {
      // 获取动量缓冲变量
      opmath_t momentum_buf_var = momentum_buf_ptr[d];

      // 如果是第一步，则直接将梯度值赋给动量缓冲变量
      if (is_first_step) {
        momentum_buf_var = grad_val;
      } else {
        // 否则按动量更新规则更新动量缓冲变量
        momentum_buf_var = momentum_buf_var * opmath_t(momentum) +
            grad_val * opmath_t(1 - dampening);
      }

      // 更新动量缓冲变量
      momentum_buf_ptr[d] = momentum_buf_var;

      // 如果使用 Nesterov 动量，则更新梯度值
      if (nesterov) {
        grad_val += momentum_buf_var * opmath_t(momentum);
      } else {
        grad_val = momentum_buf_var;
      }
    }

    // 更新参数值
    param_ptr[d] = param_val - grad_val * opmath_t(lr);
  }
  // 结束遍历参数空间
}
// 结束模板函数定义

template <typename scalar_t, typename opmath_t>
typename std::enable_if<
    std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value,
    void>::
    type inline sgd_math(
  scalar_t* param_ptr,  // 参数指针，指向需要更新的参数数组
  scalar_t* grad_ptr,   // 梯度指针，指向当前参数的梯度数组
  scalar_t* momentum_buf_ptr,  // 动量缓冲指针，用于存储动量的缓冲数组
  const double weight_decay,   // 权重衰减（L2 正则化）参数
  const double momentum,       // 动量参数
  const double lr,             // 学习率
  const double dampening,      // 动量的阻尼项
  const bool nesterov,         // 是否使用 Nesterov 动量
  const bool maximize,         // 是否最大化优化目标
  const bool is_first_step,    // 是否为第一步迭代
  const float* grad_scale_ptr, // 梯度缩放因子指针
  int64_t size                 // 数组的大小
){
  using Vec = at::vec::Vectorized<scalar_t>;  // 使用模板中的矢量化类型 Vec

  Vec grad_vec_to_store;  // 用于存储处理后的梯度矢量
  int64_t d = 0;  // 初始化循环计数器 d

  // 对每个向量化的块执行更新
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec param_vec = Vec::loadu(param_ptr + d);  // 加载参数向量
    Vec grad_vec = Vec::loadu(grad_ptr + d);    // 加载梯度向量

    // 如果存在梯度缩放因子，则对梯度向量进行缩放
    if (grad_scale_ptr) {
      grad_vec = grad_vec / Vec(scalar_t(*grad_scale_ptr));  // 缩放梯度向量
      grad_vec_to_store = grad_vec;  // 存储缩放后的梯度向量
      grad_vec_to_store.store(grad_ptr + d);  // 将缩放后的梯度向量存回原数组
    }

    // 如果指定最大化优化目标，则对梯度向量取反
    if (maximize) grad_vec = grad_vec * Vec(scalar_t(-1.0));

    // 如果设置了权重衰减，则计算带衰减的梯度
    if (weight_decay != 0.0){
      grad_vec = vec::fmadd(param_vec, Vec(scalar_t(weight_decay)), grad_vec);
    }

    // 如果设置了动量，则更新动量缓冲区
    if (momentum != 0.0) {
      Vec momentum_vec;
      if (is_first_step) {
        momentum_vec = grad_vec;  // 如果是第一步迭代，直接使用梯度向量
      } else {
        momentum_vec =
            Vec::loadu(momentum_buf_ptr + d) * Vec(scalar_t(momentum));  // 加载当前动量向量并乘以动量因子
        momentum_vec = vec::fmadd(Vec(scalar_t(1 - dampening)), grad_vec, momentum_vec);  // 计算带阻尼的新动量向量
      }
      momentum_vec.store(momentum_buf_ptr + d);  // 将更新后的动量向量存入缓冲区
      if (nesterov) {
        grad_vec =  vec::fmadd(momentum_vec, Vec(scalar_t(momentum)), grad_vec);  // 如果使用 Nesterov 动量，则更新梯度向量
      } else {
        grad_vec = momentum_vec;  // 否则直接使用动量向量作为更新值
      }
    }

    param_vec += grad_vec * Vec(scalar_t(-lr));  // 更新参数向量
    param_vec.store(param_ptr + d);  // 将更新后的参数向量存回原数组
  }

  scalar_t grad_val_to_store;  // 用于存储处理后的梯度值

  // 处理剩余的不完整向量化块或标量更新
  for (; d < size; d++) {
    scalar_t grad_val = grad_ptr[d];  // 加载当前位置的梯度值

    // 如果存在梯度缩放因子，则对梯度值进行缩放
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / scalar_t(*grad_scale_ptr);  // 缩放梯度值
      grad_val_to_store = grad_val;  // 存储缩放后的梯度值
      grad_ptr[d] = grad_val_to_store;  // 将缩放后的梯度值存回原数组
    }

    // 如果指定最大化优化目标，则对梯度值取反
    if (maximize) grad_val = -grad_val;

    // 如果设置了权重衰减，则计算带衰减的梯度值
    if (weight_decay != 0.0){
      grad_val += param_ptr[d] * scalar_t(weight_decay);
    }

    // 如果设置了动量，则更新动量缓冲区
    if (momentum != 0.0) {
      if (is_first_step) {
        momentum_buf_ptr[d] = grad_val;  // 如果是第一步迭代，直接使用梯度值
      } else {
        momentum_buf_ptr[d] = momentum_buf_ptr[d] * scalar_t(momentum) +
            grad_val * scalar_t(1 - dampening);  // 计算带阻尼的新动量值
      }
      if (nesterov) {
        grad_val += momentum_buf_ptr[d] * scalar_t(momentum);  // 如果使用 Nesterov 动量，则更新梯度值
      } else {
        grad_val = momentum_buf_ptr[d];  // 否则直接使用动量值作为更新值
      }
    }

    param_ptr[d] -= grad_val * scalar_t(lr);  // 更新参数值
  }
}

template <typename scalar_t>
void sgd_fused_step_impl(
    const at::Tensor& param,  // 参数张量
    const at::Tensor& grad,   // 梯度张量
    const at::Tensor& momentum_buffer,  // 动量缓冲张量
    const double weight_decay,   // 权重衰减（L2 正则化）参数
    const double momentum,       // 动量参数
    const double lr,             // 学习率
    const double dampening,      // 动量的阻尼项
    const bool nesterov,         // 是否使用 Nesterov 动量
    const bool maximize,         // 是否最大化优化目标
    const bool is_first_step,    // 是否为第一步迭代
    const float* grad_scale_ptr) {
  // 定义类型别名 opmath_t 为 scalar_t 类型的 at::opmath_type
  using opmath_t = at::opmath_type<scalar_t>;
  // 获取参数张量 param 的数据指针
  scalar_t* param_data = param.data_ptr<scalar_t>();
  // 获取梯度张量 grad 的数据指针
  scalar_t* grad_data = grad.data_ptr<scalar_t>();
  // 判断是否有动量缓存
  bool has_momentum_buffer = momentum != 0.0;
  // 获取动量缓存张量 momentum_buffer 的数据指针，如果没有动量缓存则设为 nullptr
  scalar_t* momentum_buffer_data = has_momentum_buffer ? momentum_buffer.data_ptr<scalar_t>() : nullptr;

  // 定义缓存行大小为 64 字节
  constexpr size_t cache_line_size = 64;
  // 计算缓存行对齐的任务单元大小，以 scalar_t 类型为单位
  constexpr int64_t cache_line_aligned_task_unit = cache_line_size / sizeof(scalar_t);
  // 计算参数张量 param 的单元数
  size_t num_units = divup(param.numel(), cache_line_aligned_task_unit);

  // 定义 SGD 更新函数 sgd_fn，使用 lambda 表达式
  auto sgd_fn = [&](int64_t begin, int64_t end) {
        // 转换为参数在数组中的起始索引
        begin *= cache_line_aligned_task_unit;
        // 计算结束位置的最小值，以参数张量的元素数为界
        end = std::min(end * cache_line_aligned_task_unit, param.numel());
        // 获取当前任务单元的参数、梯度、动量缓存的指针
        scalar_t* param_ptr = param_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* momentum_buffer_ptr = has_momentum_buffer ? momentum_buffer_data + begin : nullptr;

        // 计算当前任务单元的大小
        const int64_t size = end - begin;
        // 调用 SGD 更新函数 sgd_math，更新参数
        sgd_math<scalar_t, opmath_t>(
          param_ptr,
          grad_ptr,
          momentum_buffer_ptr,
          weight_decay,
          momentum,
          lr,
          dampening,
          nesterov,
          maximize,
          is_first_step,
          grad_scale_ptr,
          size
        );
      };
  // 使用 ATen 提供的并行函数 parallel_for，按任务单元并行执行 SGD 更新
  at::parallel_for(
      0, num_units, 0, sgd_fn);
}

// 定义一个名为 fused_sgd_kernel 的函数，用于执行融合的 SGD 更新操作
void fused_sgd_kernel(
    const at::Tensor& param,                      // 参数张量
    const at::Tensor& grad,                       // 梯度张量
    const at::Tensor& momentum_buffer,            // 动量缓存张量
    const double weight_decay,                    // 权重衰减参数
    const double momentum,                        // 动量参数
    const double lr,                              // 学习率
    const double dampening,                       // 阻尼参数
    const bool nesterov,                          // 是否使用 Nesterov 动量
    const bool maximize,                          // 是否最大化优化目标
    const bool is_first_step,                     // 是否第一步更新
    const float* grad_scale_ptr                   // 梯度缩放比例指针
  ) {
  // 创建一个连续的梯度张量
  Tensor grad_contiguous = grad.contiguous();
  // 根据参数的数据类型，调度具体的 SGD 更新实现函数
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, param.scalar_type(), "fused_sgd_kernel", [&] {
    // 调用具体的融合 SGD 步骤实现函数，传入参数
    sgd_fused_step_impl<scalar_t>(
      param,
      grad,
      momentum_buffer,
      weight_decay,
      momentum,
      lr,
      dampening,
      nesterov,
      maximize,
      is_first_step,
      grad_scale_ptr);
  });
}

}

// 注册并分发 fused_sgd_kernel 函数的调度器
REGISTER_DISPATCH(fused_sgd_stub, &fused_sgd_kernel);
} // namespace at::native
```