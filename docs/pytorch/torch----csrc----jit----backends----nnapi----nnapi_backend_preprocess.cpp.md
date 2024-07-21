# `.\pytorch\torch\csrc\jit\backends\nnapi\nnapi_backend_preprocess.cpp`

```py
// 包含 PyBind11 库的头文件
#include <pybind11/pybind11.h>
// 包含 Torch 的后端相关头文件
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

// 将模型转换为 Android NNAPI 后端，并对其进行序列化以用于移动设备
// 返回一个包含预处理项的字典：
//    "shape_compute_module": torch::jit::Module,
//    "ser_model": at::Tensor,
//    "weights": List[torch.Tensor],
//    "inp_mem_fmts": List[int],
//    "out_mem_fmts": List[int]
//
// method_compile_spec 应包含一个 Tensor 或 Tensor List，其中包含多个输入参数：
// 形状、数据类型、量化和维度顺序（NHWC/NCHW）
// 对于输入形状，运行时/加载时使用 0 表示灵活输入
//
// 编译规范应包括以下格式：
// {"forward": {"inputs": at::Tensor}}
// 或 {"forward": {"inputs": c10::List<at::Tensor>}}
// 示例输入 Tensor：
// torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
//
// 未来，预处理将接受专用对象
c10::IValue preprocess(
    const torch::jit::Module& mod,  // 输入的 Torch 模块
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,  // 方法编译规范字典
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {  // 后端调试句柄生成器

  // 获取全局解释器锁，以调用 Python 函数处理模块到 Android NNAPI 后端的转换
  py::gil_scoped_acquire gil;
  py::object pyModule = py::module_::import("torch.backends._nnapi.prepare");
  py::object pyMethod = pyModule.attr("process_for_nnapi");

  // 将 C++ 模块包装为 RecursiveScriptModule
  auto wrapped_mod =
      py::module::import("torch.jit._recursive").attr("wrap_cpp_module")(mod);
  wrapped_mod.attr("eval")();

  // 检查 method_compile_spec 是否包含必要的键和 Tensor/TensorList 输入
  c10::IValue inp;
  std::string error = "";
  if (!method_compile_spec.contains("forward")) {
    error = R"(method_compile_spec does not contain the "forward" key.)";
  } else {
    auto innerDict = method_compile_spec.at("forward");
    if (!innerDict.isGenericDict() ||
        !innerDict.toGenericDict().contains("inputs")) {
      error =
          R"(method_compile_spec does not contain a dictionary with an "inputs" key, under its "forward" key.)";
    } else {
      inp = innerDict.toGenericDict().at("inputs");
      if (!inp.isTensor() && !inp.isTensorList()) {
        error =
            R"(method_compile_spec does not contain either a Tensor or TensorList, under its "inputs" key.)";
      }
    }
  }
  // 如果存在错误，则...
  if (!error.empty()) {
    throw std::runtime_error(
        error +
        "\nmethod_compile_spec should contain a Tensor or Tensor List which bundles input parameters:"
        " shape, dtype, quantization, and dimorder."
        "\nFor input shapes, use 0 for run/load time flexible input."
        "\nmethod_compile_spec must use the following format:"
        "\n{\"forward\": {\"inputs\": at::Tensor}} OR {\"forward\": {\"inputs\": c10::List<at::Tensor>}}");
  }


// 如果方法编译规范不符合预期，抛出运行时错误，包含错误信息和建议的正确格式说明
throw std::runtime_error(
    error +
    "\nmethod_compile_spec should contain a Tensor or Tensor List which bundles input parameters:"
    " shape, dtype, quantization, and dimorder."
    "\nFor input shapes, use 0 for run/load time flexible input."
    "\nmethod_compile_spec must use the following format:"
    "\n{\"forward\": {\"inputs\": at::Tensor}} OR {\"forward\": {\"inputs\": c10::List<at::Tensor>}}");
}



  // Convert input to a Tensor or a python list of Tensors
  py::list nnapi_processed;
  if (inp.isTensor()) {
    nnapi_processed = pyMethod(wrapped_mod, inp.toTensor());
  } else {
    py::list pyInp;
    for (at::Tensor inpElem : inp.toTensorList()) {
      pyInp.append(inpElem);
    }
    nnapi_processed = pyMethod(wrapped_mod, pyInp);
  }


  // 将输入转换为 Tensor 或 Python 的 Tensor 列表
  py::list nnapi_processed;
  if (inp.isTensor()) {
    // 如果输入是 Tensor，使用 inp.toTensor() 调用 pyMethod 处理
    nnapi_processed = pyMethod(wrapped_mod, inp.toTensor());
  } else {
    // 如果输入是 Tensor 列表，逐个处理并添加到 pyInp 列表中，然后调用 pyMethod 处理
    py::list pyInp;
    for (at::Tensor inpElem : inp.toTensorList()) {
      pyInp.append(inpElem);
    }
    nnapi_processed = pyMethod(wrapped_mod, pyInp);
  }



  // Cast and insert processed items into dict
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  dict.insert("ser_model", py::cast<at::Tensor>(nnapi_processed[1]));


  // 将处理后的项目转换并插入到字典中
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  // 将 nnapi_processed 中的第二个元素转换为 at::Tensor 类型，并插入字典
  dict.insert("ser_model", py::cast<at::Tensor>(nnapi_processed[1]));



  // Serialize shape_compute_module for mobile
  auto shape_compute_module =
      py::cast<torch::jit::Module>(nnapi_processed[0].attr("_c"));
  std::stringstream ss;
  shape_compute_module._save_for_mobile(ss);
  dict.insert("shape_compute_module", ss.str());


  // 为移动端序列化 shape_compute_module
  auto shape_compute_module =
      py::cast<torch::jit::Module>(nnapi_processed[0].attr("_c"));
  // 创建一个 stringstream 对象 ss，将 shape_compute_module 序列化并存入 ss
  std::stringstream ss;
  shape_compute_module._save_for_mobile(ss);
  // 将序列化后的字符串插入到字典中的 "shape_compute_module" 键
  dict.insert("shape_compute_module", ss.str());



  // transform Python lists to C++ c10::List
  c10::List<at::Tensor> weights(
      py::cast<std::vector<at::Tensor>>(nnapi_processed[2]));
  for (auto i = 0U; i < weights.size(); i++) {
    weights.set(i, weights.get(i).contiguous());
  }
  c10::List<int64_t> inp_mem_fmts(
      py::cast<std::vector<int64_t>>(nnapi_processed[3]));
  c10::List<int64_t> out_mem_fmts(
      py::cast<std::vector<int64_t>>(nnapi_processed[4]));
  dict.insert("weights", weights);
  dict.insert("inp_mem_fmts", inp_mem_fmts);
  dict.insert("out_mem_fmts", out_mem_fmts);


  // 将 Python 的列表转换为 C++ 的 c10::List
  c10::List<at::Tensor> weights(
      py::cast<std::vector<at::Tensor>>(nnapi_processed[2]));
  // 对 weights 中的每个 Tensor 进行连续内存处理
  for (auto i = 0U; i < weights.size(); i++) {
    weights.set(i, weights.get(i).contiguous());
  }
  // 将 nnapi_processed 中的第三到第五个元素转换为对应的 c10::List 类型，并插入字典中
  c10::List<int64_t> inp_mem_fmts(
      py::cast<std::vector<int64_t>>(nnapi_processed[3]));
  c10::List<int64_t> out_mem_fmts(
      py::cast<std::vector<int64_t>>(nnapi_processed[4]));
  dict.insert("weights", weights);
  dict.insert("inp_mem_fmts", inp_mem_fmts);
  dict.insert("out_mem_fmts", out_mem_fmts);



  return dict;


  // 返回填充好数据的字典对象
  return dict;
}



// 这是一个闭合的大括号，用于结束某个代码块或函数定义。

constexpr auto backend_name = "nnapi";



// 定义了一个常量表达式的变量 `backend_name`，其值为字符串 "nnapi"。
// `constexpr` 表示在编译时计算该变量的值，并且其值在程序执行期间不会改变。



static auto pre_reg =
    torch::jit::backend_preprocess_register(backend_name, preprocess);



// 使用 `torch::jit::backend_preprocess_register` 注册一个后端预处理函数 `preprocess`，
// 该函数将与名称为 `backend_name` 的后端关联起来。
// `static auto` 表示该变量仅在当前编译单元内可见，不会被其他编译单元引用。
// `pre_reg` 可能是一个静态变量，用于在程序初始化时注册后端预处理函数。
```