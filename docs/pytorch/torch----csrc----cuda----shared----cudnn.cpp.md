# `.\pytorch\torch\csrc\cuda\shared\cudnn.cpp`

```
// 如果定义了 USE_CUDNN 或者 USE_ROCM，则包含以下头文件和命名空间定义
#if defined(USE_CUDNN) || defined(USE_ROCM)
#include <torch/csrc/utils/pybind.h> // 包含 pybind.h 头文件

#include <array> // 包含 array 头文件
#include <tuple> // 包含 tuple 头文件

namespace {
using version_tuple = std::tuple<size_t, size_t, size_t>; // 定义一个元组类型 version_tuple
}

#ifdef USE_CUDNN
#include <cudnn.h> // 如果定义了 USE_CUDNN，则包含 cudnn.h 头文件

namespace {

// 获取编译时的 cuDNN 版本号作为元组返回
version_tuple getCompileVersion() {
  return version_tuple(CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
}

// 获取运行时的 cuDNN 版本号作为元组返回
version_tuple getRuntimeVersion() {
#ifndef USE_STATIC_CUDNN
  int major, minor, patch;
  cudnnGetProperty(MAJOR_VERSION, &major);
  cudnnGetProperty(MINOR_VERSION, &minor);
  cudnnGetProperty(PATCH_LEVEL, &patch);
  return version_tuple((size_t)major, (size_t)minor, (size_t)patch);
#else
  return getCompileVersion(); // 如果使用静态链接，则返回编译时的版本号
#endif
}

// 获取 cuDNN 的版本号作为整数返回
size_t getVersionInt() {
#ifndef USE_STATIC_CUDNN
  return cudnnGetVersion();
#else
  return CUDNN_VERSION; // 如果使用静态链接，则返回编译时的版本号
#endif
}

} // namespace
#elif defined(USE_ROCM)
#include <miopen/miopen.h> // 如果定义了 USE_ROCM，则包含 miopen.h 头文件
#include <miopen/version.h> // 包含 miopen 版本信息头文件

namespace {

// 获取编译时的 MIOpen 版本号作为元组返回
version_tuple getCompileVersion() {
  return version_tuple(
      MIOPEN_VERSION_MAJOR, MIOPEN_VERSION_MINOR, MIOPEN_VERSION_PATCH);
}

// 获取运行时的 MIOpen 版本号作为元组返回
version_tuple getRuntimeVersion() {
  // MIOpen 在 2.3.0 之前不包含运行时版本信息
#if (MIOPEN_VERSION_MAJOR > 2) || \
    (MIOPEN_VERSION_MAJOR == 2 && MIOPEN_VERSION_MINOR > 2)
  size_t major, minor, patch;
  miopenGetVersion(&major, &minor, &patch);
  return version_tuple(major, minor, patch);
#else
  return getCompileVersion(); // 如果版本较低，则返回编译时的版本号
#endif
}

// 获取 MIOpen 的版本号作为整数返回
size_t getVersionInt() {
  // MIOpen 的版本号是 MAJOR*1000000 + MINOR*1000 + PATCH
  auto [major, minor, patch] = getRuntimeVersion();
  return major * 1000000 + minor * 1000 + patch;
}

} // namespace
#endif

namespace torch::cuda::shared {

// 初始化 cudnn 绑定函数，将其加入到 Python 模块中
void initCudnnBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudnn = m.def_submodule("_cudnn", "libcudnn.so bindings");

  // 定义 cudnnRNNMode_t 的枚举类型及其对应的值
  py::enum_<cudnnRNNMode_t>(cudnn, "RNNMode")
      .value("rnn_relu", CUDNN_RNN_RELU)
      .value("rnn_tanh", CUDNN_RNN_TANH)
      .value("lstm", CUDNN_LSTM)
      .value("gru", CUDNN_GRU);

  // 根据使用的库（cuDNN 还是 MIOpen）设置 cudnn 是否为 CUDA
#ifdef USE_CUDNN
  cudnn.attr("is_cuda") = true;
#else
  cudnn.attr("is_cuda") = false;
#endif

  // 将获取版本信息的函数加入 cudnn 模块中
  cudnn.def("getRuntimeVersion", getRuntimeVersion);
  cudnn.def("getCompileVersion", getCompileVersion);
  cudnn.def("getVersionInt", getVersionInt);
}

} // namespace torch::cuda::shared
#endif
```