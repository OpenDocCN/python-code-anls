# `.\pytorch\aten\src\ATen\native\UpSample.cpp`

```py
// 定义宏，仅在断言方法运算符中使用
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含必要的头文件
#include <ATen/native/UpSample.h>
#include <c10/util/irange.h>
#include <c10/util/TypeCast.h>

// 定义命名空间：at::native::upsample
namespace at::native::upsample {

// 定义公共的 TORCH_API 函数，计算输出大小
TORCH_API c10::SmallVector<int64_t, 3> compute_output_size(
    c10::IntArrayRef input_size,  // 完整的输入张量大小
    at::OptionalIntArrayRef output_size,  // 可选的输出大小
    std::optional<c10::ArrayRef<double>> scale_factors) {  // 可选的缩放因子数组

  // 计算空间维度数量
  const auto spatial_dimensions = static_cast<int64_t>(input_size.size()) - 2;

  // 如果指定了输出大小
  if (output_size) {
    // 检查不能同时指定输出大小和缩放因子
    TORCH_CHECK(!scale_factors, "Must specify exactly one of output_size and scale_factors");
    // 检查输出大小的维度与空间维度数量是否匹配
    TORCH_CHECK(static_cast<int64_t>(output_size->size()) == spatial_dimensions);
    // 返回输出大小作为向量
    return {output_size->data(), output_size->data() + output_size->size()};
  }

  // 如果指定了缩放因子
  if (scale_factors) {
    // 检查不能同时指定输出大小和缩放因子
    TORCH_CHECK(!output_size, "Must specify exactly one of output_size and scale_factors");
    // 检查缩放因子的维度与空间维度数量是否匹配
    TORCH_CHECK(static_cast<int64_t>(scale_factors->size()) == spatial_dimensions);
    
    // 创建返回的大小向量
    c10::SmallVector<int64_t, 3> ret;
    // 遍历空间维度
    for (const auto i : c10::irange(spatial_dimensions)) {
      // 计算每个维度的输出大小
      const double odim = static_cast<double>(input_size[i+2]) * scale_factors.value()[i];
      // 将计算得到的大小转换为 int64_t 类型并添加到返回向量中
      ret.push_back(c10::checked_convert<int64_t>(odim, "int64_t"));
    }
    // 返回计算出的输出大小向量
    return ret;
  }

  // 如果既未指定输出大小也未指定缩放因子，则抛出错误
  TORCH_CHECK(false, "Must specify exactly one of output_size and scale_factors");
}

} // namespace at::native::upsample
```