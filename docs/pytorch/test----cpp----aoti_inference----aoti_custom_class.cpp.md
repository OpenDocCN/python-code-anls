# `.\pytorch\test\cpp\aoti_inference\aoti_custom_class.cpp`

```
// 引入标准异常库
#include <stdexcept>

// 引入模型容器运行时 CPU 实现头文件
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

// 如果使用 CUDA，引入模型容器运行时 CUDA 实现头文件
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif

// 引入自定义类 MyAOTIClass 的头文件
#include "aoti_custom_class.h"

// 定义命名空间 torch::aot_inductor
namespace torch::aot_inductor {

// 静态变量 registerMyAOTIClass，注册 MyAOTIClass 类
static auto registerMyAOTIClass =
    torch::class_<MyAOTIClass>("aoti", "MyAOTIClass")
        // 定义构造函数接受两个字符串参数
        .def(torch::init<std::string, std::string>())
        // 定义 forward 方法绑定到 MyAOTIClass 的 forward 方法
        .def("forward", &MyAOTIClass::forward)
        // 定义 pickle 方法，用于序列化和反序列化对象
        .def_pickle(
            // 序列化函数，将 MyAOTIClass 对象转换为参数向量
            [](const c10::intrusive_ptr<MyAOTIClass>& self)
                -> std::vector<std::string> {
              std::vector<std::string> v;
              v.push_back(self->lib_path());  // 存储模型路径
              v.push_back(self->device());    // 存储设备类型
              return v;
            },
            // 反序列化函数，根据参数向量创建 MyAOTIClass 对象
            [](std::vector<std::string> params) {
              return c10::make_intrusive<MyAOTIClass>(params[0], params[1]);
            });

// MyAOTIClass 的构造函数定义，接受模型路径和设备类型两个参数
MyAOTIClass::MyAOTIClass(
    const std::string& model_path,
    const std::string& device)
    : lib_path_(model_path), device_(device) {
  // 根据设备类型选择模型容器运行时实现
  if (device_ == "cuda") {
    // 使用 CUDA 实现创建 AOTIModelContainerRunnerCuda 对象
    runner_ = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_path.c_str());
  } else if (device_ == "cpu") {
    // 使用 CPU 实现创建 AOTIModelContainerRunnerCpu 对象
    runner_ = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        model_path.c_str());
  } else {
    // 抛出运行时异常，表示设备类型无效
    throw std::runtime_error("invalid device: " + device);
  }
}

// 定义 MyAOTIClass 的 forward 方法，接受输入张量并返回张量数组
std::vector<torch::Tensor> MyAOTIClass::forward(
    std::vector<torch::Tensor> inputs) {
  // 调用模型容器运行时对象的 run 方法进行前向推断
  return runner_->run(inputs);
}

} // namespace torch::aot_inductor


这段代码是一个 C++ 的命名空间 torch::aot_inductor 下的类 MyAOTIClass 的定义及相关操作，包括注册类、构造函数定义、前向推断方法定义和异常处理。
```