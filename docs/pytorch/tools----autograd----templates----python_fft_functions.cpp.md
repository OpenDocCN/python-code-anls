# `.\pytorch\tools\autograd\templates\python_fft_functions.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于仅在方法操作符中进行断言

// 包含 Torch 的设备相关头文件
#include "torch/csrc/Device.h"
// 包含 Torch 的动态类型相关头文件
#include "torch/csrc/DynamicTypes.h"
// 包含 Torch 的异常处理相关头文件
#include "torch/csrc/Exceptions.h"
// 包含 Torch 的自动微分 FFT 函数的 Python 接口头文件
#include "torch/csrc/autograd/python_fft_functions.h"
// 包含 Torch 的自动微分生成的 Python 返回类型头文件
#include "torch/csrc/autograd/generated/python_return_types.h"
// 包含 Torch 的自动微分 Python 变量相关头文件
#include "torch/csrc/autograd/python_variable.h"
// 包含 Torch 的自动微分工具函数，用于封装输出
#include "torch/csrc/autograd/utils/wrap_outputs.h"
// 包含 Torch 的自动微分 Python 参数解析工具头文件
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
// 包含 Torch 的自动微分生成的变量工厂函数头文件
#include "torch/csrc/autograd/generated/variable_factories.h"
// 包含 Torch 的工具函数，用于输出类型
#include "torch/csrc/utils/out_types.h"
// 包含 Torch 的 Python C 函数辅助工具头文件
#include "torch/csrc/utils/pycfunction_helpers.h"
// 包含 Torch 的 Python 参数解析器头文件
#include "torch/csrc/utils/python_arg_parser.h"
// 包含 Torch 的结构序列工具头文件
#include "torch/csrc/utils/structseq.h"
// 包含 Torch 的设备延迟初始化工具头文件
#include "torch/csrc/utils/device_lazy_init.h"

// 包含 ATen 的张量核心头文件
#include <ATen/core/Tensor.h>

// 如果未定义每个操作符的头文件，则包含 ATen 的函数集合头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 否则，使用预定义的操作符头文件列表
#else
$ops_headers
#endif

// 使用 ATen 命名空间中的一些常用类型
using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;

// 使用 Torch 的工具函数，检查输出类型是否匹配
using torch::utils::check_out_type_matches;
// 使用 Torch 自动微分工具函数的命名空间
using namespace torch::autograd::utils;

// Torch 自动微分命名空间开始
namespace torch::autograd {

// 以下是生成的前向声明

${py_forwards}

// FFT 函数列表，用于 Python 方法定义
static PyMethodDef fft_functions[] = {
  ${py_method_defs}
  {NULL}  // 方法列表以 NULL 结束
};

// THPFFTVariableFunctionsModule 定义为 NULL
static PyObject* THPFFTVariableFunctionsModule = NULL;

// 初始化 FFT 函数，将其添加到 Python 模块中
void initFFTFunctions(PyObject* module) {
  // 定义 PyModuleDef 结构体
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._fft",  // 模块名为 torch._C._fft
     NULL,
     -1,
     fft_functions  // 使用之前定义的 FFT 函数列表
  };
  // 创建 FFT 模块对象
  PyObject* fft = PyModule_Create(&def);
  // 将创建的 FFT 模块赋值给全局变量 THPFFTVariableFunctionsModule
  THPFFTVariableFunctionsModule = fft;
  // 如果创建模块失败，抛出 Python 异常
  if (!fft) {
    throw python_error();
  }
  // 将 FFT 模块添加到给定的 Python 模块中，并释放 FFT 模块的引用
  // 注意，PyModule_AddObject 会“偷走” fft 的引用计数
  if (PyModule_AddObject(module, "_fft", fft) != 0) {
    throw python_error();
  }
}

// 以下是生成的方法定义

${py_methods}

} // namespace torch::autograd
// Torch 自动微分命名空间结束
```