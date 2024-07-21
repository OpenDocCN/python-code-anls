# `.\pytorch\torch\csrc\utils\tensor_layouts.cpp`

```py
// 包含头文件：ATen 布局定义
#include <ATen/Layout.h>
// 包含头文件：C10 标量类型定义
#include <c10/core/ScalarType.h>
// 包含头文件：Torch 动态类型定义
#include <torch/csrc/DynamicTypes.h>
// 包含头文件：Torch 异常处理定义
#include <torch/csrc/Exceptions.h>
// 包含头文件：Torch 布局定义
#include <torch/csrc/Layout.h>
// 包含头文件：Python C API 头文件
#include <torch/csrc/python_headers.h>
// 包含头文件：Torch 对象指针工具定义
#include <torch/csrc/utils/object_ptr.h>
// 包含头文件：Torch 张量布局定义
#include <torch/csrc/utils/tensor_layouts.h>

// 命名空间定义：torch::utils
namespace torch::utils {

// 宏定义：注册布局对象到 Python 模块
#define REGISTER_LAYOUT(layout, LAYOUT)                                     \
  // 创建对应布局的 Python 对象，并绑定到 torch 模块上
  PyObject* layout##_layout =                                               \
      THPLayout_New(at::Layout::LAYOUT, "torch." #layout);                  \
  // 增加对象的引用计数
  Py_INCREF(layout##_layout);                                               \
  // 将布局对象添加到 torch 模块
  if (PyModule_AddObject(torch_module, "" #layout, layout##_layout) != 0) { \
    throw python_error();                                                   \
  }                                                                         \
  // 注册布局对象到 C++ 中的数据结构
  registerLayoutObject((THPLayout*)layout##_layout, at::Layout::LAYOUT);

// 函数定义：初始化 Torch 的各种布局对象
void initializeLayouts() {
  // 导入 torch Python 模块
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  // 如果导入失败，抛出 Python 错误
  if (!torch_module)
    throw python_error();

  // 创建并注册 Strided 布局对象
  PyObject* strided_layout =
      THPLayout_New(at::Layout::Strided, "torch.strided");
  Py_INCREF(strided_layout);
  if (PyModule_AddObject(torch_module, "strided", strided_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)strided_layout, at::Layout::Strided);

  // 创建并注册 Sparse COO 布局对象
  PyObject* sparse_coo_layout =
      THPLayout_New(at::Layout::Sparse, "torch.sparse_coo");
  Py_INCREF(sparse_coo_layout);
  if (PyModule_AddObject(torch_module, "sparse_coo", sparse_coo_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)sparse_coo_layout, at::Layout::Sparse);

  // 使用宏注册 Sparse CSR、CSC、BSR、BSC 布局对象
  REGISTER_LAYOUT(sparse_csr, SparseCsr)
  REGISTER_LAYOUT(sparse_csc, SparseCsc)
  REGISTER_LAYOUT(sparse_bsr, SparseBsr)
  REGISTER_LAYOUT(sparse_bsc, SparseBsc)

  // 创建并注册 MKLDNN 布局对象
  PyObject* mkldnn_layout = THPLayout_New(at::Layout::Mkldnn, "torch._mkldnn");
  Py_INCREF(mkldnn_layout);
  if (PyModule_AddObject(torch_module, "_mkldnn", mkldnn_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)mkldnn_layout, at::Layout::Mkldnn);

  // 使用宏注册 Jagged 布局对象
  REGISTER_LAYOUT(jagged, Jagged);
}

} // namespace torch::utils
```