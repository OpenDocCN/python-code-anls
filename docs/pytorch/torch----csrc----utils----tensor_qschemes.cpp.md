# `.\pytorch\torch\csrc\utils\tensor_qschemes.cpp`

```
// 包含头文件，定义了与张量量化方案相关的实用工具
#include <torch/csrc/utils/tensor_qschemes.h>

// 包含核心量化方案相关的头文件
#include <c10/core/QScheme.h>
// 包含用于迭代范围的工具函数
#include <c10/util/irange.h>
// 包含动态类型定义相关的头文件
#include <torch/csrc/DynamicTypes.h>
// 包含异常处理相关的头文件
#include <torch/csrc/Exceptions.h>
// 包含量化方案定义相关的头文件
#include <torch/csrc/QScheme.h>

// 包含Python C API相关的头文件
#include <torch/csrc/python_headers.h>
// 包含对象智能指针相关的工具函数
#include <torch/csrc/utils/object_ptr.h>

// 定义 torch::utils 命名空间
namespace torch::utils {

// 声明静态数组，用于存储所有编译时数量的量化方案的 Python 对象
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::array<PyObject*, at::COMPILE_TIME_NUM_QSCHEMES> thp_qscheme_array;

// 初始化量化方案
void initializeQSchemes() {
  // 导入 torch 模块
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();  // 抛出 Python 异常
  }

  // 遍历所有编译时数量的量化方案
  for (const auto i : c10::irange(at::COMPILE_TIME_NUM_QSCHEMES)) {
    auto qscheme = static_cast<at::QScheme>(i);  // 转换为量化方案枚举类型
    // 创建量化方案的 Python 对象
    PyObject* qscheme_obj = THPQScheme_New(qscheme, toString(qscheme));
    // 将 Python 对象存储到静态数组中
    thp_qscheme_array[static_cast<int>(qscheme)] = qscheme_obj;
    Py_INCREF(qscheme_obj);  // 增加 Python 对象的引用计数
    // 将量化方案对象添加到 torch 模块中
    if (PyModule_AddObject(
            torch_module, toString(qscheme).c_str(), qscheme_obj) != 0) {
      throw python_error();  // 添加失败时抛出异常
    }
  }
}

// 根据量化方案枚举获取对应的 Python 对象
PyObject* getTHPQScheme(at::QScheme qscheme) {
  auto qscheme_ = thp_qscheme_array[static_cast<int>(qscheme)];
  if (!qscheme_) {
    throw std::invalid_argument("unsupported QScheme");  // 如果对象为空，则抛出无效参数异常
  }
  return qscheme_;  // 返回量化方案的 Python 对象
}

} // namespace torch::utils
```