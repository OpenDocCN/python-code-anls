# `.\pytorch\torch\csrc\jit\python\python_ivalue.h`

```
#pragma once
// 用于确保头文件只包含一次

#include <ATen/core/ivalue.h>
// 引入 ATen 库的 IValue 头文件

#include <pybind11/pybind11.h>
// 引入 pybind11 库的主头文件

#include <torch/csrc/jit/python/pybind_utils.h>
// 引入 PyTorch JIT 的 Python 绑定工具头文件

#include <torch/csrc/python_headers.h>
// 引入 PyTorch Python 头文件

#include <torch/csrc/utils/pybind.h>
// 引入 PyTorch 的 PyBind 工具函数头文件

namespace py = pybind11;
// 命名空间别名，简化 pybind11 的命名空间引用

namespace c10::ivalue {

// 定义一个具体的 PyObjectHolder，用于保存 py::object
struct C10_EXPORT ConcretePyObjectHolder final : PyObjectHolder {
 public:
  // 创建 ConcretePyObjectHolder 实例的静态方法，接受 py::object 作为参数
  static c10::intrusive_ptr<PyObjectHolder> create(py::object py_obj) {
    return c10::make_intrusive<ConcretePyObjectHolder>(std::move(py_obj));
  }

  // 创建 ConcretePyObjectHolder 实例的静态方法，接受 py::handle 作为参数
  static c10::intrusive_ptr<PyObjectHolder> create(const py::handle& handle) {
    py::gil_scoped_acquire ag;
    return c10::make_intrusive<ConcretePyObjectHolder>(
        handle.cast<py::object>());
  }

  // 覆盖父类的虚函数，返回持有的 PyObject 指针
  PyObject* getPyObject() override {
    return py_obj_.ptr();
  }

  // 覆盖父类的虚函数，尝试推断对象的类型信息
  InferredType tryToInferType() override {
    pybind11::gil_scoped_acquire ag;
    return torch::jit::tryToInferType(py_obj_);
  }

  // 覆盖父类的虚函数，将对象转换为对应的 IValue
  IValue toIValue(const TypePtr& type, std::optional<int32_t> N = c10::nullopt)
      override {
    pybind11::gil_scoped_acquire ag;
    return torch::jit::toIValue(py_obj_, type, N);
  }

  // 覆盖父类的虚函数，返回对象的字符串表示
  std::string toStr() override {
    pybind11::gil_scoped_acquire ag;
    return py::str(py_obj_);
  }

  // 覆盖父类的虚函数，从对象中提取张量数据
  std::vector<at::Tensor> extractTensors() override {
    try {
      pybind11::gil_scoped_acquire ag;
      static py::object& extractorFn = *new py::object(
          py::module::import("torch._jit_internal").attr("_extract_tensors"));
      return extractorFn(py_obj_).cast<std::vector<at::Tensor>>();
    } catch (py::error_already_set& e) {
      auto err = std::runtime_error(
          c10::str("Cannot extract tensors from value: ", e.what()));
      {
        pybind11::gil_scoped_acquire ag;
        e.restore();
        PyErr_Clear();
      }
      throw err;
    }
  }

  // 销毁 py::object 的注释
  // 注释 [Destructing py::object]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // (1) 为什么 py_obj_ = py::none(); 不起作用？因为我们也需要在销毁指向 None 的
  //     py::object 时获得 GIL。这是因为它对 None 解引用。
  //
  // (2) 为什么我们需要显式调用 dec_ref()？因为在销毁时，py::object 的 nullptr，
  //     实际上不执行任何操作，因为它调用 Py_XDECREF(NULL)。
  //     https://docs.python.org/3/c-api/refcounting.html#c.Py_XDECREF
  ~ConcretePyObjectHolder() override {
    pybind11::gil_scoped_acquire ag;
    py_obj_.dec_ref();
    // 显式将 PyObject* 设置为 nullptr，以防止 py::object 的析构函数再次对 PyObject 进行 decref
    // 将指针设置为 nullptr，确保在初始化时指向空值
    py_obj_.ptr() = nullptr;
  }

  // 显式构造函数，避免错误的隐式转换和拷贝初始化
  // 使用 std::move 将 py_obj 转移，确保高效的对象所有权转移
  explicit ConcretePyObjectHolder(py::object py_obj)
      : py_obj_(std::move(py_obj)) {}

 private:
  // Python 对象的持有者，使用 py::object 进行封装
  py::object py_obj_;
};

// 结束 c10::ivalue 命名空间
} // namespace c10::ivalue
```