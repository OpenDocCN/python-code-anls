# `.\pytorch\torch\csrc\lazy\core\tensor_util.cpp`

```py
// 包含 Torch 框架的懒加载模块中的头文件
#include <torch/csrc/lazy/core/tensor_util.h>

// 包含 C10 库中的 BFloat16 类型定义
#include <c10/util/BFloat16.h>
// 包含 C10 库中的 Half 类型定义
#include <c10/util/Half.h>
// 包含 C10 库中的复数类型定义
#include <c10/util/complex.h>
// 包含 C10 库中的整数范围迭代器定义
#include <c10/util/irange.h>

// 包含 Torch 框架的懒加载模块中后端设备管理的头文件
#include <torch/csrc/lazy/backend/backend_device.h>
// 包含 Torch 框架的懒加载模块中后端接口定义的头文件
#include <torch/csrc/lazy/backend/backend_interface.h>
// 包含 Torch 框架的懒加载模块中配置管理的头文件
#include <torch/csrc/lazy/core/config.h>
// 包含 Torch 框架的懒加载模块中辅助函数的头文件
#include <torch/csrc/lazy/core/helpers.h>

// 包含标准库中的算法函数
#include <algorithm>
// 包含标准库中的字符串处理函数
#include <cstring>
// 包含标准库中的函数对象
#include <functional>
// 包含标准库中的链表容器
#include <list>
// 包含标准库中的数值计算函数
#include <numeric>
// 包含标准库中的多线程支持
#include <thread>

// 定义 Torch 框架中懒加载模块的命名空间
namespace torch {
namespace lazy {

// 计算给定大小的数组的步幅
std::vector<int64_t> ComputeArrayStrides(c10::ArrayRef<int64_t> sizes) {
  // 创建步幅数组，初始化为1
  std::vector<int64_t> strides(sizes.size(), 1);
  // 根据数组大小反向计算步幅
  for (int64_t i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * sizes[i - 1];
  }
  return strides;
}

// 将后端数据句柄转换为张量
std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<BackendDataPtr> data_handles,
    at::ScalarType dest_element_type) {
  // 创建张量数组
  std::vector<at::Tensor> tensors;
  // 遍历数据句柄数组
  for (const auto& handle : data_handles) {
    // 调用后端接口，从计算数据句柄创建张量，并加入数组
    tensors.push_back(
        getBackend()->MakeTensorFromComputationData(handle, dest_element_type));
  }
  return tensors;
}

// 将张量转换为后端数据句柄
BackendDataPtr TensorToDataHandle(
    const at::Tensor& tensor,
    const BackendDevice& device) {
  // 调用后端接口，从张量创建计算数据句柄
  return getBackend()->MakeComputationDataFromTensor(
      tensor, Shape(tensor.scalar_type(), tensor.sizes()), device);
}

// 创建张量数据的后端数据句柄数组
std::vector<BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<BackendDevice>& devices) {
  // 检查张量和设备数组的大小是否相同
  TORCH_CHECK(tensors.size() == devices.size());
  // 创建后端数据句柄数组
  std::vector<BackendDataPtr> result;
  result.reserve(tensors.size());
  // 遍历张量和设备数组，将每个张量转换为后端数据句柄并加入数组
  for (const auto i : c10::irange(tensors.size())) {
    result.push_back(TensorToDataHandle(tensors[i], devices[i]));
  }
  return result;
}

// 判断是否为特殊标量值
bool IsSpecialScalar(const at::Scalar& value) {
  // 检查是否启用处理特殊标量的标志，并且值是整数或浮点数
  if (FLAGS_torch_lazy_handle_special_scalars &&
      (value.isIntegral(false) || value.isFloatingPoint())) {
    // 如果启用处理所有数字为特殊标量的标志，则返回 true
    if (FLAGS_torch_lazy_all_numbers_special_scalars) {
      return true;
    }
    // 将标量值转换为双精度浮点数
    double scalar_value = value.toDouble();
    // 如果标量值等于 0 或绝对值等于 1，则返回 true
    return scalar_value == 0.0 || std::fabs(scalar_value) == 1.0;
  }
  return false;
}

} // namespace lazy
} // namespace torch
```