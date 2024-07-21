# `.\pytorch\torch\csrc\jit\backends\coreml\cpp\preprocess.cpp`

```py
// 包含 pybind11 库，用于 Python 和 C++ 之间的互操作
#include <pybind11/pybind11.h>
// 包含 Torch 的 JIT 后端相关的头文件
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
// 包含 Torch 的 Python 绑定工具函数
#include <torch/csrc/jit/python/pybind_utils.h>
// 包含 Torch 的 Python 工具函数
#include <torch/csrc/utils/pybind.h>
// 包含 Torch 的 Script 模块
#include <torch/script.h>

namespace py = pybind11;

namespace {

// 定义一个静态函数 preprocess，用于预处理 Torch 模块
c10::IValue preprocess(
    const torch::jit::Module& mod,  // 输入参数：Torch 模块
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,  // 输入参数：方法编译规范
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {  // 输入参数：调试句柄生成器

  // 导入 torch.backends._coreml.preprocess 模块并获取 preprocess 方法对象
  py::object pyModule = py::module_::import("torch.backends._coreml.preprocess");
  py::object pyMethod = pyModule.attr("preprocess");

  // 调用 Python 的 preprocess 方法，并将 Torch 模块和方法编译规范作为参数
  py::dict modelDict = pyMethod(mod, torch::jit::toPyObject(method_compile_spec));

  // 创建一个空的 C++ 字典，用于存储模型数据
  c10::Dict<std::string, std::string> modelData;

  // 遍历 Python 字典 modelDict 的项，并将其转换为 C++ 的字符串键值对，存储到 modelData 中
  for (auto item : modelDict) {
    modelData.insert(
        item.first.cast<std::string>(), item.second.cast<std::string>());
  }

  // 返回处理后的模型数据
  return modelData;
}

// 注册 preprocess 函数为 Torch 的 JIT 后端预处理函数，关联到 "coreml" 后端
static auto pre_reg = torch::jit::backend_preprocess_register("coreml", preprocess);

} // namespace
```