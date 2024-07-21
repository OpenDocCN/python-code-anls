# `.\pytorch\torch\csrc\utils\out_types.cpp`

```py
// 包含 Torch 的输出类型头文件
#include <torch/csrc/utils/out_types.h>

// 命名空间 torch::utils 下定义的函数和类
namespace torch::utils {

// 用于 Python 绑定代码生成，确保任何 TensorOptions 参数与输出张量的选项一致
void check_out_type_matches(
    const at::Tensor& result,                              // 输出结果张量
    std::optional<at::ScalarType> scalarType,              // 可选的标量类型
    bool scalarType_is_none,                               // 标量类型是否为 None
    std::optional<at::Layout> layout,                      // 可选的布局
    std::optional<at::Device> device,                      // 可选的设备
    bool device_is_none) {                                 // 设备是否为 None
  // 如果标量类型为 None 且布局和设备均未指定，则为常见情况，直接返回
  if (scalarType_is_none && !layout && device_is_none) {
    return;
  }
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  // 如果标量类型不为 None 且结果张量的标量类型与给定的标量类型不匹配，则抛出错误
  if (!scalarType_is_none && result.scalar_type() != scalarType.value()) {
    AT_ERROR(
        "dtype ",
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        *scalarType,
        " does not match dtype of out parameter (",
        result.scalar_type(),
        ")");
  }
  // 如果指定了布局且结果张量的布局与给定的布局不匹配，则抛出错误
  if (layout && result.layout() != *layout) {
    AT_ERROR(
        "layout ",
        *layout,
        " does not match layout of out parameter (",
        result.layout(),
        ")");
  }
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  // 如果设备类型不为 None 且结果张量的设备类型与给定的设备类型不匹配，则抛出错误
  if (!device_is_none && result.device().type() != device.value().type()) {
    AT_ERROR(
        "device type ",
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        device->type(),
        " does not match device type of out parameter (",
        result.device().type(),
        ")");
  }
}

} // namespace torch::utils
```