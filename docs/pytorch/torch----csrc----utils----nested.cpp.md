# `.\pytorch\torch\csrc\utils\nested.cpp`

```py
// 包含 ATen 库中的头文件
#include <ATen/ATen.h>
// 包含 NestedTensorImpl 类的头文件
#include <ATen/NestedTensorImpl.h>
// 包含 ScalarType 类型定义的头文件
#include <c10/core/ScalarType.h>
// 包含与 Python 交互的头文件
#include <torch/csrc/python_headers.h>
// 包含处理嵌套结构的实用函数的头文件
#include <torch/csrc/utils/nested.h>
// 包含 Pybind 库的头文件
#include <torch/csrc/utils/pybind.h>
// 包含创建张量的头文件
#include <torch/csrc/utils/tensor_new.h>
// 包含 PyTorch 核心库的头文件
#include <torch/torch.h>
// 包含处理异常的头文件
#include <stdexcept>
// 包含标准向量的头文件
#include <vector>

// 定义 torch::utils 命名空间
namespace torch::utils {

// 注意：这里的 device_idx 不是 DeviceIndex，而是 PythonArgs 中的索引
static c10::TensorOptions typeIdWithDefault(
    PythonArgs& r,              // Python 参数对象的引用
    int device_idx,             // 设备索引
    c10::DispatchKey dispatch_key) {  // 分发键
  auto options = dispatchKeyToTensorOptions(dispatch_key);  // 根据分发键获取张量选项
  if (!r.isNone(device_idx)) {  // 如果 Python 参数中的设备索引不为空
    options = options.device(r.device(device_idx));  // 设置张量选项的设备
  }
  return options;  // 返回更新后的张量选项
}

// 创建嵌套张量的构造函数
at::Tensor nested_tensor_ctor(
    c10::DispatchKey dispatch_key,      // 分发键
    at::ScalarType scalar_type,         // 标量类型
    torch::PythonArgs& r) {             // Python 参数对象的引用
  TORCH_CHECK(r.idx == 0, "nested_tensor(): invalid arguments");  // 检查参数的有效性

  PyObject* data = r.pyobject(0);  // 获取第一个参数作为 Python 对象
  // 检查数据是否为列表：只接受 List[Tensor] 和 List[List...[Scalar]]
  TORCH_CHECK_TYPE(
      PyList_Check(data),
      "Only lists (List[Tensor] and List[List...[Scalar]]) are accepted in nested_tensor");

  auto dtype_val = r.scalartypeWithDefault(1, scalar_type);  // 获取标量类型参数，如果未提供则使用默认值
  auto tensor_options = typeIdWithDefault(r, 2, dispatch_key);  // 获取张量选项，可能包括设备信息
  bool pin_memory = r.toBool(3);  // 检查是否需要固定内存
  bool args_requires_grad = r.toBool(4);  // 检查是否需要梯度

  TORCH_CHECK(
      PyList_Size(data) >= 0,
      "Something went really wrong and your list has negative size");  // 检查列表大小是否非负

  // 检查是否处理张量列表
  std::vector<at::Tensor> new_list(PyList_Size(data));
  for (const auto i : c10::irange(PyList_Size(data))) {
    PyObject* elem = PyList_GetItem(data, i);  // 获取列表中的每个元素
    if (THPVariable_Check(elem)) {  // 如果是 Torch 张量变量
      new_list[i] = THPVariable_Unpack(PyList_GetItem(data, i)).detach();  // 解包 Torch 变量并分离梯度
      TORCH_CHECK(
          !new_list[i].is_nested(),
          "We do not accept nested tensors as input to nested tensors");  // 拒绝嵌套张量作为输入
      TORCH_CHECK(
          new_list[i].layout() == kStrided,
          "We do not accept non-strided layouts as input to nested tensors");  // 拒绝非步进布局的张量
    } else {
      PythonArgs elem_r(r);  // 使用当前 Python 参数对象初始化新的 PythonArgs 对象
      std::array<PyObject*, 6> elem_args = {
          elem,             // 数据
          r.args[1],        // 标量类型
          nullptr,          // 设备 (cpu)
          nullptr,          // 不固定内存
          r.args[4],        // 需要梯度
          nullptr           // 名称
      };
      elem_r.args = elem_args.data();  // 设置元素的参数
      new_list[i] = tensor_ctor(dispatch_key, scalar_type, elem_r);  // 使用构造函数创建张量
    }
  }

  at::ScalarType final_dtype = dtype_val;  // 最终标量类型
  if (r.isNone(1) && !new_list.empty()) {  // 如果标量类型未提供且列表不为空
    final_dtype = c10::typeMetaToScalarType(new_list[0].dtype());  // 使用第一个张量的类型作为最终类型
  }
  at::Device final_device = tensor_options.device();  // 最终设备
  if (r.isNone(2) && !new_list.empty()) {  // 如果设备未提供且列表不为空
    final_device = new_list[0].device();  // 使用第一个张量的设备作为最终设备
  }
  auto out = at::_nested_tensor_from_tensor_list(
      new_list, final_dtype, c10::nullopt, final_device, pin_memory);  // 创建嵌套张量
  out.requires_grad_(args_requires_grad);  // 设置是否需要梯度
  return out;  // 返回创建的嵌套张量
}

} // namespace torch::utils
```