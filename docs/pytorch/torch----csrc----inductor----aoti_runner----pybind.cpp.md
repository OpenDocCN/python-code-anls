# `.\pytorch\torch\csrc\inductor\aoti_runner\pybind.cpp`

```py
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <torch/csrc/utils/pybind.h>

namespace torch::inductor {

// 初始化 AOTI 运行时绑定
void initAOTIRunnerBindings(PyObject* module) {
  // 将传入的 Python 模块对象转换为 py::module 类型
  auto rootModule = py::handle(module).cast<py::module>();
  // 定义名为 "_aoti" 的子模块
  auto m = rootModule.def_submodule("_aoti");

  // 定义 AOTIModelContainerRunnerCpu 类的 Python 绑定
  py::class_<AOTIModelContainerRunnerCpu>(m, "AOTIModelContainerRunnerCpu")
      .def(py::init<const std::string&, int>())  // 构造函数绑定
      .def("run", &AOTIModelContainerRunnerCpu::run)  // 成员函数 run 绑定
      .def("get_call_spec", &AOTIModelContainerRunnerCpu::get_call_spec)  // 成员函数 get_call_spec 绑定
      .def(
          "get_constant_names_to_original_fqns",  // 获取常量名到原始全限定名的映射
          &AOTIModelContainerRunnerCpu::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",  // 获取常量名到数据类型的映射
          &AOTIModelContainerRunnerCpu::getConstantNamesToDtypes);

  // 如果定义了 USE_CUDA，定义 AOTIModelContainerRunnerCuda 类的 Python 绑定
#ifdef USE_CUDA
  py::class_<AOTIModelContainerRunnerCuda>(m, "AOTIModelContainerRunnerCuda")
      .def(py::init<const std::string&, int>())  // 构造函数绑定
      .def(py::init<const std::string&, int, const std::string&>())  // 构造函数绑定
      .def(py::init<
           const std::string&,
           int,
           const std::string&,
           const std::string&>())  // 构造函数绑定
      .def("run", &AOTIModelContainerRunnerCuda::run)  // 成员函数 run 绑定
      .def("get_call_spec", &AOTIModelContainerRunnerCuda::get_call_spec)  // 成员函数 get_call_spec 绑定
      .def(
          "get_constant_names_to_original_fqns",  // 获取常量名到原始全限定名的映射
          &AOTIModelContainerRunnerCuda::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",  // 获取常量名到数据类型的映射
          &AOTIModelContainerRunnerCuda::getConstantNamesToDtypes);
#endif

  // 定义 unsafe_alloc_void_ptrs_from_tensors 函数，接受一个 tensor 向量作为参数
  m.def(
      "unsafe_alloc_void_ptrs_from_tensors",
      [](std::vector<at::Tensor>& tensors) {
        // 调用 torch::aot_inductor::unsafe_alloc_new_handles_from_tensors 函数
        std::vector<AtenTensorHandle> handles =
            torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(tensors);
        // 将 AtenTensorHandle 指针转换为 void* 向量，并返回结果
        std::vector<void*> result(
            reinterpret_cast<void**>(handles.data()),
            reinterpret_cast<void**>(handles.data()) + handles.size());
        return result;
      });

  // 定义 unsafe_alloc_void_ptr_from_tensor 函数，接受一个 tensor 作为参数
  m.def("unsafe_alloc_void_ptr_from_tensor", [](at::Tensor& tensor) {
    // 调用 torch::aot_inductor::new_tensor_handle 函数，返回新分配的 tensor handle
    return reinterpret_cast<void*>(
        torch::aot_inductor::new_tensor_handle(std::move(tensor)));
  });

  // 定义 alloc_tensors_by_stealing_from_void_ptrs 函数，接受一个 void* 向量作为参数
  m.def(
      "alloc_tensors_by_stealing_from_void_ptrs",
      [](std::vector<void*>& raw_handles) {
        // 调用 torch::aot_inductor::alloc_tensors_by_stealing_from_handles 函数
        return torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
            reinterpret_cast<AtenTensorHandle*>(raw_handles.data()),
            raw_handles.size());
      });

  // 定义 alloc_tensor_by_stealing_from_void_ptr 函数，接受一个 void* 参数
  m.def("alloc_tensor_by_stealing_from_void_ptr", [](void* raw_handle) {
    // 调用 torch::aot_inductor::tensor_handle_to_tensor_pointer 函数，返回对应的 tensor 指针
    return *torch::aot_inductor::tensor_handle_to_tensor_pointer(
        reinterpret_cast<AtenTensorHandle>(raw_handle));
  });
}
} // namespace torch::inductor
```