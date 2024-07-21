# `.\pytorch\test\cpp\aoti_inference\test.cpp`

```py
#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <torch/script.h>
#include <torch/torch.h>

#define STR_VALUE(x) #x
#define STRINGIZE(x) STR_VALUE(x)

namespace {

// 测试函数，用于测试 AOTI 模型
void test_aoti(const std::string& device, bool use_runtime_constant_folding) {
  // 禁用梯度计算上下文
  torch::NoGradGuard no_grad;

  // 构造数据路径，将当前二进制目录下的"data.pt"文件路径转换为字符串
  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  // 加载 Torch 脚本模块"data.pt"并赋值给data_loader
  torch::jit::script::Module data_loader = torch::jit::load(data_path);

  // 根据设备和是否使用运行时常量折叠选择后缀名
  std::string suffix = use_runtime_constant_folding
      ? device + "_use_runtime_constant_folding"
      : device;

  // 构造模型路径属性名和输入输出属性名
  std::string path_attr = "model_so_path_" + suffix;
  std::string inputs_attr = "inputs_" + suffix;
  std::string outputs_attr = "outputs_" + suffix;

  // 从data_loader模块中获取模型路径，并转换为字符串引用
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  // 从data_loader模块中获取输入张量列表并转换为向量
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  // 从data_loader模块中获取参考输出张量列表并转换为向量
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  // 根据设备类型选择合适的模型运行器
  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  if (device == "cuda") {
    // 如果设备是cuda，则使用AOTIModelContainerRunnerCuda模型容器运行器
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_so_path);
  } else if (device == "cpu") {
    // 如果设备是cpu，则使用AOTIModelContainerRunnerCpu模型容器运行器
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        model_so_path);
  } else {
    // 如果设备不是cuda或cpu，抛出测试断言失败异常
    testing::AssertionFailure() << "unsupported device: " << device;
  }

  // 运行模型容器的run方法，得到实际输出张量列表
  auto actual_output_tensors = runner->run(input_tensors);
  // 断言实际输出张量与参考输出张量的元素之间的近似性
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

// 测试函数，用于测试AOTI脚本模型
void test_aoti_script(const std::string& device) {
  // 禁用梯度计算上下文
  torch::NoGradGuard no_grad;

  // 构造脚本模型文件名，例如"script_model_cuda.pt"或"script_model_cpu.pt"
  std::string script_model = "script_model_" + device + ".pt";
  // 构造模型路径，将当前二进制目录下的"script_model_{device}.pt"文件路径转换为字符串
  std::string model_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / script_model.c_str())
           .string();
  // 加载 Torch 脚本模型
  torch::jit::script::Module model = torch::jit::load(model_path);

  // 构造样本数据文件路径，将当前二进制目录下的"script_data.pt"文件路径转换为字符串
  std::string sample_data_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "script_data.pt")
           .string();
  // 加载 Torch 脚本模块"script_data.pt"
  torch::jit::script::Module sample_data = torch::jit::load(sample_data_path);

  // 构造输入和输出属性名
  std::string inputs_attr = "inputs_" + device;
  std::string outputs_attr = "outputs_" + device;

  // 从sample_data模块中获取输入张量列表并转换为向量
  const auto& inputs = sample_data.attr(inputs_attr.c_str()).toList().vec();
  // 从sample_data模块中获取参考输出张量列表并转换为向量
  const auto& ref_output_tensors =
      sample_data.attr(outputs_attr.c_str()).toTensorVector();

  // 对模型进行前向推断，得到输出张量列表
  auto outputs = model.forward(inputs).toTuple()->elements();

  // 断言输出张量列表的长度与参考输出张量列表的长度相等
  ASSERT_EQ(outputs.size(), ref_output_tensors.size());

  // 遍历每个输出张量，断言其与对应的参考输出张量之间的元素近似性
  for (size_t i = 0; i < ref_output_tensors.size(); i++) {
    ASSERT_TRUE(torch::allclose(outputs[i].toTensor(), ref_output_tensors[i]));
  }
}

void test_aoti_constants_update(
    const std::string& device,
  // 设置不计算梯度
  torch::NoGradGuard no_grad;

  // 构建数据路径
  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();

  // 加载模型数据
  torch::jit::script::Module data_loader = torch::jit::load(data_path);

  // 根据是否使用运行时常量折叠来确定后缀
  std::string suffix = use_runtime_constant_folding
      ? device + "_use_runtime_constant_folding"
      : device;

  // 构建属性名
  std::string path_attr = "model_so_path_" + suffix;
  std::string inputs_attr = "inputs_" + suffix;
  std::string outputs_attr = "outputs_" + suffix;
  std::string weights_attr = "w_pre_" + suffix;
  std::string add_attr = "w_add_" + suffix;

  // 获取模型路径、输入、输出、权重和添加值
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();
  const auto& weight_tensors =
      data_loader.attr(weights_attr.c_str()).toTensor();
  const auto& add_tensors = data_loader.attr(add_attr.c_str()).toTensor();

  // 创建三个映射，用于存储不同类型的权重
  torch::inductor::TensorConstantMap missing_map, rand_map, real_map;
  missing_map.emplace("L__self___w_pre", new at::Tensor(at::randn({4, 4})));
  rand_map.emplace("L__self___w_pre", new at::Tensor(at::randn({10})));
  rand_map.emplace("L__self___w_add", new at::Tensor(at::randn({10})));
  real_map.emplace("L__self___w_pre", new at::Tensor(weight_tensors));
  real_map.emplace("L__self___w_add", new at::Tensor(add_tensors));

  // 创建模型容器运行器
  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  if (device == "cuda") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_so_path);
  } else if (device == "cpu") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        model_so_path);
  } else {
    // 抛出异常，不支持的设备
    testing::AssertionFailure() << "unsupported device: " << device;
  }

  // 加载模型并验证结果
  auto actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // 更新缺失映射，预期抛出异常
  EXPECT_THROW(
      runner->update_constant_buffer(missing_map, false, true),
      std::runtime_error);

  // 更新随机权重到缓冲区 #1
  runner->update_constant_buffer(missing_map, false, false);
  actual_output_tensors = runner->run(input_tensors);

  if (use_runtime_constant_folding) {
    // 在这一刻，此更新应用于原始权重
    // 被消耗的权重被“折叠”，因此不会产生影响
    ASSERT_TRUE(
        torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
    // 运行常量折叠
    runner->run_const_fold(/* use_inactive = */ false);
    // 使用 runner 对象运行输入张量，得到实际输出张量
    actual_output_tensors = runner->run(input_tensors);
  }
  // 断言：参考输出张量的第一个与实际输出张量的第一个不全相等
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // 使用真实映射更新常量缓冲区，不进行运行时常量折叠，然后再次运行输入张量
  runner->update_constant_buffer(real_map, false, false);
  actual_output_tensors = runner->run(input_tensors);
  // 如果使用运行时常量折叠，运行 runner 对象的常量折叠方法（不使用非活动常量）
  if (use_runtime_constant_folding) {
    runner->run_const_fold(/* use_inactive = */ false);
  }
  // 再次运行输入张量，得到实际输出张量
  actual_output_tensors = runner->run(input_tensors);
  // 断言：参考输出张量的第一个与实际输出张量的第一个全相等
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // 使用完全随机映射更新常量缓冲区，不进行运行时常量折叠，然后再次运行输入张量
  runner->update_constant_buffer(rand_map, false, false);
  // 如果使用运行时常量折叠，运行 runner 对象的常量折叠方法（不使用非活动常量）
  if (use_runtime_constant_folding) {
    runner->run_const_fold(/* use_inactive = */ false);
  }
  // 再次运行输入张量，得到实际输出张量
  actual_output_tensors = runner->run(input_tensors);
  // 断言：参考输出张量的第一个与实际输出张量的第一个不全相等
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
  // 定义测试函数，用于测试 AOTI 模型的双缓冲特性
void test_aoti_double_buffering(
    const std::string& device,                         // 设备名称，指定运行模型的硬件设备
    bool use_runtime_constant_folding) {                // 是否使用运行时常量折叠优化标志

  torch::NoGradGuard no_grad;                           // 禁用梯度计算上下文管理器，确保在测试中不会进行梯度计算

  // 构建数据文件路径
  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();

  // 加载 PyTorch JIT 模型
  torch::jit::script::Module data_loader = torch::jit::load(data_path);

  // 根据设备和运行时常量折叠标志确定后缀
  std::string suffix = use_runtime_constant_folding
      ? device + "_use_runtime_constant_folding"         // 使用运行时常量折叠的后缀
      : device;                                          // 不使用运行时常量折叠的后缀

  // 构建模型动态属性字符串
  std::string path_attr = "model_so_path_" + suffix;     // 模型动态库路径属性
  std::string inputs_attr = "inputs_" + suffix;          // 输入数据属性
  std::string outputs_attr = "outputs_" + suffix;        // 输出数据属性
  std::string weights_attr = "w_pre_" + suffix;          // 权重数据属性
  std::string add_attr = "w_add_" + suffix;              // 加法数据属性

  // 获取模型动态属性值
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();  // 输入张量列表

  // 获取参考输出张量列表
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  // 获取权重张量
  const auto& weight_tensors =
      data_loader.attr(weights_attr.c_str()).toTensor();

  // 获取加法张量
  const auto& add_tensors = data_loader.attr(add_attr.c_str()).toTensor();

  // 创建随机张量映射和真实张量映射
  torch::inductor::TensorConstantMap rand_map, real_map;
  rand_map.emplace("L__self___w_pre", new at::Tensor(at::randn({4, 4})));  // 随机权重映射
  rand_map.emplace("L__self___w_add", new at::Tensor(at::randn({4, 4})));  // 随机加法映射
  real_map.emplace("L__self___w_pre", new at::Tensor(weight_tensors));     // 真实权重映射
  real_map.emplace("L__self___w_add", new at::Tensor(add_tensors));        // 真实加法映射

  // 创建 AOTI 模型容器运行器的唯一指针
  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;

  // 根据设备类型选择相应的 AOTI 模型容器运行器
  if (device == "cuda") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_so_path.c_str());                          // CUDA 设备的模型容器运行器
  } else if (device == "cpu") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        model_so_path.c_str());                          // CPU 设备的模型容器运行器
  } else {
    testing::AssertionFailure() << "unsupported device: " << device;  // 不支持的设备类型，测试失败
  }

  // 默认情况下，缓冲区 #1 装载烧录的权重，验证正确结果
  auto actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));  // 断言实际输出张量与参考输出张量一致

  // 更新缓冲区 #2 的权重并激活，依然应该产生正确结果，因为它是真实的常量映射
  runner->update_inactive_constant_buffer(real_map);      // 更新非活动常量缓冲区
  if (use_runtime_constant_folding) {
    runner->run_const_fold(/* use_inactive = */ true);    // 运行常量折叠，使用非活动缓冲区
  }
  runner->swap_constant_buffer();                        // 交换常量缓冲区
  actual_output_tensors = runner->run(input_tensors);     // 运行模型
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));  // 断言实际输出张量与参考输出张量一致

  // 更新随机权重到缓冲区 #1，但暂时不交换权重
  runner->update_inactive_constant_buffer(rand_map);      // 更新非活动常量缓冲区
  if (use_runtime_constant_folding) {
    // 调用 runner 对象的常量折叠方法，使用非活跃模式
    runner->run_const_fold(/* use_inactive = */ true);
  }
  // 运行模型，获取实际输出张量
  actual_output_tensors = runner->run(input_tensors);
  // 断言：验证参考输出张量和实际输出张量的所有元素是否接近
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // 交换并激活权重至缓冲区 #1。这些是随机权重，可能导致错误的结果。
  runner->swap_constant_buffer();
  // 再次运行模型，获取更新后的实际输出张量
  actual_output_tensors = runner->run(input_tensors);
  // 断言：验证参考输出张量和实际输出张量的所有元素是否不接近
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // 交换回缓冲区 #2，这里存储的是真实的常量数据。
  runner->swap_constant_buffer();
  // 再次运行模型，获取最终的实际输出张量
  actual_output_tensors = runner->run(input_tensors);
  // 断言：验证参考输出张量和实际输出张量的所有元素是否接近
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
void test_aoti_double_buffering_with_tensor_constants() {
  // 禁用梯度计算上下文管理器，确保不计算梯度
  torch::NoGradGuard no_grad;

  // 获取数据路径，使用 CMAKE_CURRENT_BINARY_DIR 和文件名拼接而成
  std::string data_path = (std::filesystem::path(
                               STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) /
                               "data_with_tensor_constants.pt")
                               .string();

  // 加载 TorchScript 模型数据
  torch::jit::script::Module data_loader = torch::jit::load(data_path);

  // 定义模型数据中的属性名称
  std::string path_attr = "model_so_path";
  std::string inputs_attr = "inputs";
  std::string w_attr = "w";
  std::string outputs_attr = "outputs";

  // 从模型数据中获取模型文件路径
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();

  // 从模型数据中获取输入张量列表，并转换为向量
  auto input_tensors = data_loader.attr(inputs_attr.c_str()).toTensorList().vec();

  // 从模型数据中获取权重张量 w
  const auto& w_tensors = data_loader.attr(w_attr.c_str()).toTensor();

  // 从模型数据中获取参考输出张量列表，并转换为向量
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  // 创建实际张量常量映射
  torch::inductor::TensorConstantMap real_map;
  real_map.emplace("L__self___w", new at::Tensor(w_tensors));

  // 创建 AOTIModelContainerRunner 对象，使用 CUDA 运行时
  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path.c_str());

  // 默认情况下，使用预烧的权重加载缓冲区 #1，并验证结果正确性
  auto actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // 更新不活动常量缓冲区 #2 中的权重，并激活该缓冲区，预期结果仍然正确
  runner->update_inactive_constant_buffer(real_map);
  runner->swap_constant_buffer();
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}
```