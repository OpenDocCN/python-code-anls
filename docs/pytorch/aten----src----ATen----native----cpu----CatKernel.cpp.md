# `.\pytorch\aten\src\ATen\native\cpu\CatKernel.cpp`

```py
// 定义预处理器宏，用于只包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库的 Tensor 类定义
#include <ATen/core/Tensor.h>

// 包含 ATen 库的调度功能
#include <ATen/Dispatch.h>
// 包含 ATen 库的 CPU 环境下的 CatKernel 头文件
#include <ATen/native/cpu/CatKernel.h>
// 包含 ATen 库的 CPU 环境下的向量化功能头文件
#include <ATen/cpu/vec/functional.h>
// 包含 ATen 库的 CPU 环境下的向量化支持头文件
#include <ATen/cpu/vec/vec.h>
// 包含 C10 实用工具库的 irange 功能
#include <c10/util/irange.h>

// 进入 ATen 的 native 命名空间
namespace at::native {

// 匿名命名空间，用于隐藏实现细节
namespace {

// 输入元数据结构体，描述张量的数据指针和内部大小
struct InputMeta {
  const void* data_ptr;  // 数据指针，指向张量的数据
  int64_t inner_size;    // 内部大小，指定张量在给定维度上的大小乘以内部大小

  // 构造函数，根据张量和维度信息初始化
  InputMeta(const Tensor& t, int64_t dim, int64_t inner)
    : data_ptr(t.const_data_ptr())  // 初始化数据指针
    , inner_size(t.sizes()[dim] * inner) {}  // 计算并初始化内部大小
};

// 模板函数，用于串行连接多个张量到结果张量
template <typename scalar_t>
void cat_serial_kernel_impl(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim) {
  // 内部断言，检查维度是否在合法范围内
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dim >= 0 && dim < result.dim(), "dim out of range in cat_serial_kernel_impl");

  // 计算结果张量的外部大小
  int64_t outer = result.numel() / (result.sizes()[dim] * result.strides()[dim]);
  // 获取结果张量的数据指针，并转换为指定类型的指针
  scalar_t* result_data = result.data_ptr<scalar_t>();
  // 计算输入张量的数量
  int64_t ninputs = static_cast<int64_t>(tensors.size());
  // 创建输入元数据结构体的向量，并预留空间
  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  
  // 遍历所有输入张量，填充输入元数据结构体向量
  for (const Tensor& tensor : tensors) {
    inputs.emplace_back(tensor, dim, result.strides()[dim]);
  }

  // 使用向量化功能的别名
  using Vec = vec::Vectorized<scalar_t>;
  // 初始化结果张量的指针
  scalar_t* result_ptr = result_data;

  // 双重循环，遍历外部和输入张量
  for (const auto i : c10::irange(outer)) {
    for (const auto j : c10::irange(ninputs)) {
      // 获取当前输入张量的内部大小
      int64_t local_inner = inputs[j].inner_size;
      // 获取当前输入张量的数据指针，考虑偏移量
      const scalar_t* input_ptr = (const scalar_t*)(inputs[j].data_ptr) + i * local_inner;
      // 初始化循环变量 d
      int64_t d = 0;
      // 循环处理每个向量大小的块
      for (; d < local_inner - (local_inner % Vec::size()); d += Vec::size()) {
        // 加载输入向量并存储到结果指针
        Vec in_vec = Vec::loadu(input_ptr + d);
        in_vec.store(result_ptr + d);
      }
      // 如果不是 Microsoft 编译器，并且未定义为最小尺寸编译，则展开内部循环
      #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      // 处理剩余的元素，直到内部大小
      for (; d < local_inner; d++) {
        result_ptr[d] = input_ptr[d];
      }
      // 更新结果指针，准备处理下一个输入张量
      result_ptr += local_inner;
    }
  }
}

// 主函数，调用具体的模板函数处理串行连接操作
void cat_serial_kernel(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim) {
  // 使用 AT_DISPATCH 宏，处理浮点类型和其他特定类型的连接操作
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, result.scalar_type(), "cat_serial_kernel", [&]() {
    cat_serial_kernel_impl<scalar_t>(result, tensors, dim);
  });
}

} // anonymous namespace

// 注册调度分发函数，将串行连接的核心函数注册为 cat_serial_stub
REGISTER_DISPATCH(cat_serial_stub, &cat_serial_kernel);

} // at::native
```