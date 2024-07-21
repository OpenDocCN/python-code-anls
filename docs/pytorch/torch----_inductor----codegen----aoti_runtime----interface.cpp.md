# `.\pytorch\torch\_inductor\codegen\aoti_runtime\interface.cpp`

```py
// 包含需要的头文件
#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/thread_local.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// 定义宏，用于捕获异常并返回错误代码
#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                 \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTI_RUNTIME_FAILURE;                             \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTI_RUNTIME_FAILURE;                             \
  }                                                          \
  return AOTI_RUNTIME_SUCCESS;

// 定义宏，用于检查向量大小是否符合预期
#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor 使用 at::addmm_out，不支持需要梯度的参数。
// 因此，我们在运行 API 时强制启用 no_grad 上下文。
// RAII 线程局部的守卫，构造时禁用梯度模式，析构时恢复原始模式。
struct AOTINoGradGuard {
  AOTINoGradGuard() : prev_mode(aoti_torch_grad_mode_is_enabled()) {
    aoti_torch_grad_mode_set_enabled(false);
  }
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  bool prev_mode;
};

// 定义 C 语言风格的外部函数接口
extern "C" {

// 创建 AOTInductorModelContainer，使用指定的设备
AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
}

// 创建 AOTInductorModelContainer，使用指定的设备和 cubin 目录
AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {
  // 检查模型数量是否为正数
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0" << std::endl;
    return AOTI_RUNTIME_FAILURE;
  }
  // 尝试创建容器，捕获可能的异常并转换为错误代码
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 创建可选的 cubin 目录路径
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    // 创建一个指向 torch::aot_inductor::AOTInductorModelContainer 类型对象的指针 container，
    // 通过 new 运算符在堆上分配内存并初始化对象，参数包括 num_models（模型数量）、
    // device_str 的字符串表示（设备信息字符串）、cubin_dir_opt（CUBIN 目录选项）。
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    
    // 将 container 的指针地址强制转换为 AOTInductorModelContainerHandle 类型，
    // 并将其赋值给 container_handle 指向的对象。
    *container_handle = reinterpret_cast<AOTInductorModelContainerHandle>(container);
}

// 删除 AOTInductorModelContainer 的实例，并将异常转换为错误代码
AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将容器句柄转换为 AOTInductorModelContainer* 类型
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    // 删除容器对象
    delete container;
  });
}

// 运行 AOTInductorModelContainer 中的模型，并处理输入和输出的张量句柄
AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // 输入的 AtentTensorHandle 数组；句柄被窃取，数组本身被借用
    size_t num_inputs,
    AtenTensorHandle* output_handles, // 用于写入输出的 AtentTensorHandle 数组；句柄将被调用者窃取，数组本身被借用
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  // 将容器句柄转换为 AOTInductorModelContainer* 类型
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 检查输入和输出张量的数量是否与容器中定义的匹配
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  // 将流句柄转换为 DeviceStreamType 类型
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 使用 AOTINoGradGuard，运行容器中的模型
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

// 获取 AOTInductorModelContainer 中的常数数量，并将异常转换为错误代码
AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  // 将容器句柄转换为 AOTInductorModelContainer* 类型
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

// 获取 AOTInductorModelContainer 中特定索引处常数的名称，并将异常转换为错误代码
AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  // 将容器句柄转换为 AOTInductorModelContainer* 类型
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

// 获取 AOTInductorModelContainer 中特定索引处常数的原始全限定名，并将异常转换为错误代码
AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  // 将容器句柄转换为 AOTInductorModelContainer* 类型
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

// 获取 AOTInductorModelContainer 中特定索引处常数是否来自折叠的信息，并将异常转换为错误代码
AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  // 将容器句柄转换为 AOTInductorModelContainer* 类型
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

// 获取 AOTInductorModelContainer 中特定索引处常数的数据类型，并将异常转换为错误代码
AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  // 将传入的容器句柄转换为 AOTInductorModelContainer 指针类型
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 调用容器对象的 constant_dtype 方法，将结果赋值给传入的 dtype 指针所指向的位置
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  // 将容器句柄重新解释为模型容器指针
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 将常量映射句柄重新解释为字符串到张量句柄的无序映射指针
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  // 将异常转换为错误代码，并执行常量缓冲区更新操作
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  // 调用常量缓冲区更新函数，使用未激活的标志位并验证完全更新
  return AOTInductorModelContainerUpdateConstantBuffer(container_handle,
          constant_map_handle,
          /*use_inactive*/ true,
          /*validate_full_update*/ true);
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  // 将容器句柄重新解释为模型容器指针
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 将流句柄重新解释为设备流类型
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  // 将异常转换为错误代码，并执行常量折叠运算
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 禁用梯度运算并执行常量折叠运算
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  // 将容器句柄重新解释为模型容器指针
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 将异常转换为错误代码，并执行常量缓冲区交换操作
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  // 将容器句柄重新解释为模型容器指针
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 将异常转换为错误代码，并获取输入数量
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  // 将容器句柄重新解释为模型容器指针
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 将异常转换为错误代码，并获取指定输入索引的输入名称
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  // 将容器句柄重新解释为模型容器指针
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 将异常转换为错误代码，并获取输出数量
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}
// 获取 AOTInductorModelContainer 的输出名称
AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  // 将容器句柄转换为 AOTInductorModelContainer 对象指针
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 将异常转换为错误代码，并获取输出名称
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

// 获取 AOTInductorModelContainer 的调用规范
AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  // 将容器句柄转换为 AOTInductorModelContainer 对象指针
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // 将异常转换为错误代码，并获取输入和输出规范
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

// 创建 AOTInductorModel 对象
AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      // 创建常量映射和常量数组的共享指针
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto constant_array = std::make_shared<std::vector<torch::aot_inductor::ConstantHandle>>();
      // 将常量映射的原始指针转换为 unordered_map
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      // 创建 AOTInductorModel 对象并设置设备为 CPU
      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          "cpu", // device_str is hardcoded, as AOTInductorModelCreate is only use for CPU models
          ""
      );

      // 如果输入映射存在，将其添加到常量映射中；否则加载默认常量
      if (input_map) {
        for (auto const& kv : *input_map) {
          constant_map->emplace(kv.first, kv.second);
        }
      } else {
        model->load_constants();
      }

      // 将 AOTInductorModel 对象的指针转换为句柄
      *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
    })}
}

// 运行 AOTInductorModel 对象
AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  // 将模型句柄转换为 AOTInductorModel 对象指针
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  // 将异常转换为错误代码，并调用模型运行方法
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

// 删除 AOTInductorModel 对象
AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      // 将模型句柄转换为 AOTInductorModel 对象指针，并释放其内存
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}
}

// 获取 AOTInductorModel 的输出数量
AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      // 将模型句柄转换为 AOTInductorModel 对象指针，并获取其输出数量
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

// 更新 AOTInductorModel 的常量映射
AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    // 将传入的模型句柄转换为 AOTInductorModel 指针
    auto model =
        reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
    // 将异常转换为错误代码，并执行以下操作
    CONVERT_EXCEPTION_TO_ERROR_CODE({
        // 创建一个指向 ConstantMap 的 shared_ptr
        auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
        // 将 constant_map_handle 转换为指向未排序的字符串到 AtenTensorHandle 映射的指针
        auto input_map =
            reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
                constant_map_handle);

        // 遍历输入映射中的每对键值对，并将其插入 constant_map 中
        for (auto const& kv : *input_map) {
            constant_map->emplace(kv.first, kv.second);
        }

        // 调用模型对象的 update_constants_map 方法，传递 constant_map 的所有权
        model->update_constants_map(std::move(constant_map));
    })
}

} // extern "C"


注释：


} 
} // extern "C"


这段代码示例中的注释解释了两个重要的内容：

1. `}`：表示代码块的结束，这里是结束一个代码块的大括号。
2. `// extern "C"`：这是C++语言中的外部链接说明符，用于声明一个函数或变量是按照C语言的方式进行编译和链接的。
```