# `.\pytorch\aten\src\ATen\native\Fill.cpp`

```
// 定义宏，仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入必要的头文件
#include <ATen/native/Fill.h>
#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

// 根据不同的宏定义条件引入不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/fill_diagonal_native.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zero_native.h>
#endif

// 命名空间 at::native 开始
namespace at::native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 填充函数，填充输出张量 self 为指定标量值 value
Tensor& fill_out(Tensor& self, const Scalar& value) {
  // 如果 self 在 CPU 上且只有一个元素，调用 detail::scalar_fill 函数
  if (self.device() == at::kCPU && self.numel() == 1) {
    return at::detail::scalar_fill(self, value);
  }
  // 配置张量迭代器，设置内存重叠检查为 false，因为填充是幂等的，所以重叠是可以接受的
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .add_output(self)
    .resize_outputs(false)
    .build();
  // 调用填充的具体实现函数 fill_stub
  fill_stub(iter.device_type(), iter, value);
  return self;
}

// 量化填充函数，用标量值 value 填充输出张量 self
static Tensor& fill_out_quantized(Tensor& self, const Scalar& value) {
  // 创建全为 1 的张量，转换为 kFloat 类型，并乘以标量值 value
  at::Tensor out = at::ones(self.sizes()).to(kFloat) * value;
  // 将 out 转移到 self 的设备上，并按照推荐的内存格式转换
  out = out.to(self.device()).to(self.suggest_memory_format());
  // 使用 copy_ 方法将 out 的值复制到 self 中
  self.copy_(out);
  return self;
}

// 原地填充函数，用标量值 value 填充输出张量 self
Tensor& fill_(Tensor& self, const Scalar& value) {
  return fill_out(self, value);
}

// 原地量化填充函数，用标量值 value 填充输出张量 self
Tensor& fill_quantized_(Tensor& self, const Scalar& value) {
  return fill_out_quantized(self, value);
}

// 原地填充函数，用张量 value 填充输出张量 self
Tensor& fill_(Tensor& self, const Tensor& value) {
  // 检查 value 张量的维度是否为 0
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  // 如果 self 和 value 不在同一设备上，使用标量 value 填充 self
  if (self.device() != value.device()){
    return fill_out(self, value.item());
  }
  // 检查 value 是否是 self 的视图，若是则克隆一份避免过早地覆盖 self
  if(self.is_alias_of(value)) {
    self.copy_(value.clone());
  } else{
    self.copy_(value);
  }
  return self;
}

// 原地量化填充函数，用张量 value 填充输出张量 self
Tensor& fill_quantized_(Tensor& self, const Tensor& value) {
  // 检查 value 张量的维度是否为 0
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  // 使用标量值 value.item() 填充输出张量 self
  return fill_out_quantized(self, value.item());
}

// 元数据填充函数，直接返回 self，不做任何操作
Tensor& fill_meta_(Tensor& self, const Scalar& value) {
  return self;
}

// 元数据填充函数，直接返回 self，不做任何操作
Tensor& fill_meta_(Tensor& self, const Tensor& value) {
  // 检查 value 张量的维度是否为 0
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return self;
}

// 填充函数，返回一个新的张量，将输入张量 self 按照标量值 value 填充
Tensor fill(const Tensor& self, const Scalar& value) {
  // 创建一个与 self 相同形状的空张量，并使用 fill_ 函数填充
  return at::empty_like(self).fill_(value);
}

// 填充函数，返回一个新的张量，将输入张量 self 按照张量 value 填充
Tensor fill(const Tensor& self, const Tensor& value) {
  // 创建一个与 self 相同形状的空张量，并使用 fill_ 函数填充
  return at::empty_like(self).fill_(value);
}

// 定义填充操作的分发函数 fill_stub
DEFINE_DISPATCH(fill_stub);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill_diagonal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 填充张量的对角线元素为指定值，可能会进行环绕操作
Tensor& fill_diagonal_(Tensor& self, const Scalar& fill_value, bool wrap) {
  // 获取张量的维度数
  int64_t nDims = self.dim();
  // 检查张量维度是否大于等于2，否则抛出错误
  TORCH_CHECK(nDims >= 2, "dimensions must larger than 1");

  // 获取张量的高度和宽度
  int64_t height = self.size(0);
  int64_t width = self.size(1);

  // 如果张量维度大于2，检查除第一维度外的其他维度是否与第一维度大小相同
  if (nDims > 2) {
    int64_t dim1 = height;
    for (const auto i : c10::irange(1, nDims)) {
      if (self.size(i) != dim1) {
        AT_ERROR("all dimensions of input must be of equal length");
      }
    }
  }

  // 获取张量数据的存储偏移量
  int64_t storage_offset = self.storage_offset();
  // 初始化大小和步长向量
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  // 计算对角线元素个数（取最小值）
  int64_t size = std::min(height, width);

  // 计算总步长
  int64_t stride = 0;
  for (const auto i : c10::irange(nDims)) {
    stride += self.stride(i);
  }
  strides.push_back(stride);
  sizes.push_back(size);

  // 创建主对角线视图
  auto main_diag = self.as_strided(sizes, strides, storage_offset);
  // 用指定值填充主对角线
  main_diag.fill_(fill_value);

  // 如果需要环绕填充并且张量为二维且高度大于宽度加1
  if (wrap && nDims == 2 && height > width + 1) {
    // 初始化环绕视图大小向量
    std::vector<int64_t> wrap_sizes;

    // 计算环绕步长和环绕元素个数
    int64_t step = width + 1;
    int64_t wrap_size = ((self.numel() + step - 1) / step) - size;
    wrap_sizes.push_back(wrap_size);

    // 计算环绕视图的偏移量
    int64_t offset = self.stride(0) * (width + 1);

    // 创建环绕视图
    auto wrap_diag = self.as_strided(wrap_sizes, strides, storage_offset + offset);
    // 用指定值填充环绕视图
    wrap_diag.fill_(fill_value);
  }

  // 返回修改后的张量自身
  return self;
}

// 在 CPU 上将张量的所有元素置为零
static Tensor& zero_cpu_(Tensor &self, int64_t nelements) {
  // 获取张量数据指针
  void* ptr = self.data_ptr();
  // 如果指针为空，用零填充张量并返回
  if (nullptr == ptr) {
    return self.fill_(0);
  }
  // 计算需要填充的字节数
  int64_t size_bytes = nelements * self.dtype().itemsize();
  // 如果字节数大于0，使用 memset 函数将内存清零
  if (size_bytes > 0) {
    std::memset(ptr, 0, size_bytes);
  }
  // 返回修改后的张量自身
  return self;
}

// 根据张量的设备和特性，将张量的所有元素置为零
Tensor& zero_(Tensor &self) {
  // 计算张量的元素总数
  int64_t nelements = c10::multiply_integers(self.sizes());
  // 如果张量在 CPU 上且非重叠且稠密，并且元素数量小于指定阈值，调用 zero_cpu_ 函数
  if (self.device() == at::kCPU &&
      self.is_non_overlapping_and_dense() &&
      nelements < internal::GRAIN_SIZE) {
    return zero_cpu_(self, nelements);
  }
  // 否则，用零填充张量并返回
  return self.fill_(0);
}

// 返回原始张量本身，用于元数据操作
Tensor& zero_meta_(Tensor& self) {
  return self;
}
```