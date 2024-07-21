# `.\pytorch\aten\src\ATen\native\vulkan\ops\Factory.h`

```py
#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 声明一个函数 _empty_affine_quantized，返回一个 Tensor
Tensor _empty_affine_quantized(
    // 参数1：Tensor 的大小，作为 IntArrayRef 传入
    const IntArrayRef sizes,
    // 参数2：数据类型，可选的标量类型
    const std::optional<ScalarType> dtype,
    // 参数3：布局，可选的张量布局类型
    const std::optional<c10::Layout> layout,
    // 参数4：设备，可选的设备类型
    const std::optional<Device> device,
    // 参数5：是否使用锁页内存，可选的布尔值
    const std::optional<bool> pin_memory,
    // 参数6：量化比例，作为双精度浮点数
    const double scale,
    // 参数7：零点，作为整数值
    const int64_t zero_point,
    // 参数8：内存格式，可选的内存格式
    const optional<MemoryFormat> memory_format);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```