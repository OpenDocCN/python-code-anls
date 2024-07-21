# `.\pytorch\aten\src\ATen\native\cpu\MaxPooling.cpp`

```
// 定义编译选项，仅使用 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的分发和并行处理相关头文件
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

// 包含 ATen 核心张量相关头文件
#include <ATen/core/Tensor.h>

// 包含 ATen CPU 向量化相关头文件
#include <ATen/cpu/vec/vec.h>

// 包含 ATen 最大池化操作的头文件
#include <ATen/native/MaxPooling.h>

// 包含 c10 工具库中的 irange 函数头文件
#include <c10/util/irange.h>

// 进入 at::native 命名空间
namespace at::native {

// 匿名命名空间，内部函数和数据结构的作用域仅限于当前文件
namespace {

// max_pool1d_kernel 函数模板，实现一维最大池化操作
template <typename scalar_t>
inline void max_pool1d_kernel(
    scalar_t* C10_RESTRICT op,            // 输出张量的数据指针
    const scalar_t* C10_RESTRICT ip,      // 输入张量的数据指针
    const PoolingParams1D& p) {           // 一维池化参数结构体的引用
  // 遍历池化核的每一个元素
  for (const auto kj : c10::irange(p.KW)) {
    // 计算有效输出起始和结束位置
    int64_t oj = p.valid_output_start(kj);
    int64_t oe = p.valid_output_end(kj);
    // 计算输入张量的索引位置
    int64_t ij = p.index(kj, oj);
    // 遍历输出张量的每一个位置
    for (; oj < oe; ++oj, ij += p.SJ) {
      // 获取当前输入张量的值
      scalar_t val = ip[ij];
      // 判断是否更新最大值，处理 NaN 值情况
      bool update_max = std::isnan(val) || op[oj] < val;
      // 更新输出张量的值为最大值
      op[oj] = update_max ? val : op[oj];
    }
  }
}

// max_pool1d_impl 函数，实现一维最大池化操作的具体实现
void max_pool1d_impl(
    Tensor& output,                       // 输出张量的引用
    const Tensor& input,                  // 输入张量的常量引用
    const PoolingParams1D& p) {           // 一维池化参数结构体的引用
  // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏处理浮点类型和额外类型 Half、BFloat16
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      input.scalar_type(),                // 输入张量的标量类型
      "max_pool1d_impl",                  // 函数名称字符串
      [&] {                               // Lambda 表达式开始
        // 将输入张量转换为连续内存的张量
        const Tensor in = input.contiguous();
        // 获取输出张量的数据指针，并设置为当前类型的指针
        scalar_t* const OP = output.data_ptr<scalar_t>();
        // 获取输入张量的常量数据指针，并设置为当前类型的指针
        const scalar_t* const IP = in.const_data_ptr<scalar_t>();

        // 设置填充值，用于初始化输出张量
        scalar_t FILL = std::numeric_limits<scalar_t>::has_infinity
            ? -std::numeric_limits<scalar_t>::infinity()
            : std::numeric_limits<scalar_t>::lowest();

        // 并行处理池化操作，范围为 [0, p.NB * p.NC)，lambda 函数定义并行任务
        at::parallel_for(0, p.NB * p.NC, 0, [&](int64_t begin, int64_t end) {
          // 遍历并行任务的范围
          for (const auto it : c10::irange(begin, end)) {
            // 获取当前输出张量的指针位置
            scalar_t* op = OP + it * p.OW;
            // 获取当前输入张量的指针位置
            const scalar_t* ip = IP + it * p.IW;
            // 使用填充值初始化输出张量的值
            std::fill_n(op, p.OW, FILL);
            // 调用最大池化内核函数处理当前块的池化操作
            max_pool1d_kernel(op, ip, p);
          }
        });
      });                                 // Lambda 表达式结束
}

} // namespace

// 使用 REGISTER_DISPATCH 宏注册 max_pool1d_stub 分发函数
REGISTER_DISPATCH(max_pool1d_stub, &max_pool1d_impl);

// 退出 at::native 命名空间
} // namespace at::native
```