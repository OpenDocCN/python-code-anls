# `.\pytorch\aten\src\ATen\nnapi\nnapi_bind.cpp`

```
// 包含必要的头文件：utility（实用工具）、vector（向量容器）
#include <utility>
#include <vector>

// 包含 ATen（PyTorch C++ 前端库）相关头文件
#include <ATen/ATen.h>
#include <ATen/nnapi/nnapi_bind.h>
#include <ATen/nnapi/nnapi_wrapper.h>
#include <ATen/nnapi/nnapi_model_loader.h>
#include <c10/util/irange.h>

// 声明命名空间 torch::nnapi::bind
namespace torch {
namespace nnapi {
namespace bind {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
// 声明 nnapi 和 check_nnapi 为非常量全局变量
nnapi_wrapper* nnapi;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
nnapi_wrapper* check_nnapi;

// 静态函数 load_platform_library 的实现
static void load_platform_library() {
  // 使用 Lambda 表达式确保该函数只运行一次
  static int run_once = [](){
    // 调用 nnapi_wrapper_load 加载 NNAPI 库，并进行断言检查
    nnapi_wrapper_load(&nnapi, &check_nnapi);
    CAFFE_ENFORCE(nnapi);  // 断言 nnapi 非空
    CAFFE_ENFORCE(nnapi->Model_free);  // 断言 nnapi 的 Model_free 方法非空
    CAFFE_ENFORCE(nnapi->Compilation_free);  // 断言 nnapi 的 Compilation_free 方法非空
    CAFFE_ENFORCE(nnapi->Execution_free);  // 断言 nnapi 的 Execution_free 方法非空
    return 0;
  }();
  (void)run_once;  // 防止编译器报未使用变量的警告
}

// NnapiCompilation 类的初始化函数 init
void NnapiCompilation::init(
    at::Tensor serialized_model_tensor,  // 序列化模型的 Tensor
    std::vector<at::Tensor> parameter_buffers  // 参数缓冲区的向量
) {
  // 调用 init2 函数，移动参数并设置默认偏好
  init2(
    std::move(serialized_model_tensor),
    std::move(parameter_buffers),
    ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,  // 编译偏好设置为 Prefer Sustained Speed
    false  // 不放松从 float32 到 float16 的转换
  );
}

// NnapiCompilation 类的初始化函数 init2
void NnapiCompilation::init2(
    at::Tensor serialized_model_tensor,  // 序列化模型的 Tensor
    const std::vector<at::Tensor>& parameter_buffers,  // 参数缓冲区的向量
    int64_t compilation_preference,  // 编译偏好
    bool relax_f32_to_f16  // 是否放松从 float32 到 float16 的转换
  ) {
  // 检查是否已经初始化过 NnapiCompilation
  TORCH_CHECK(!model_, "Attempted to re-initialize NnapiCompilation.");

  // 加载 NNAPI 平台库
  load_platform_library();

  // 准备参数缓冲区
  std::vector<const void*> buffers;
  buffers.reserve(parameter_buffers.size());
  std::vector<int32_t> buffer_sizes;
  buffer_sizes.reserve(parameter_buffers.size());
  for (auto& t : parameter_buffers) {
    TORCH_CHECK(t.is_contiguous());  // 确保 Tensor 是连续的
    buffers.push_back(t.data_ptr());  // 将 Tensor 的数据指针加入缓冲区
    buffer_sizes.push_back(t.nbytes());  // 将 Tensor 的字节数加入缓冲区大小
  }

  // 检查序列化模型 Tensor 是否是连续的
  TORCH_CHECK(serialized_model_tensor.is_contiguous());
  // 根据 Tensor 的标量类型确定序列化模型指针的类型
  uint8_t* ser_model_ptr =
    serialized_model_tensor.scalar_type() == at::ScalarType::Byte
      ? serialized_model_tensor.data_ptr<uint8_t>()
      : reinterpret_cast<uint8_t*>(serialized_model_tensor.data_ptr<int32_t>());
  c10::ArrayRef<uint8_t> ser_model = {
    ser_model_ptr,
    serialized_model_tensor.nbytes()
  };
  TORCH_CHECK(!ser_model.empty());  // 确保序列化模型数据非空

  ANeuralNetworksModel* model{};
  // 调用 nnapi 的 Model_create 函数创建模型，并进行断言检查
  check_nnapi->Model_create(&model);
  CAFFE_ENFORCE(model);

  // 使用 model_.reset 重置 model_ 指针，管理新创建的模型
  model_.reset(model);

  // 调用 ::caffe2::nnapi::load_nnapi_model 加载 NNAPI 模型
  int load_result = ::caffe2::nnapi::load_nnapi_model(
      nnapi,
      model_.get(),
      ser_model.data(),
      ser_model.size(),
      buffers.size(),
      buffers.data(),
      buffer_sizes.data(),
      0,
      nullptr,
      nullptr,
      &num_inputs_,
      &num_outputs_,
      nullptr);
  CAFFE_ENFORCE(load_result == 0);  // 断言加载结果为 0，表示成功

  // 如果设置了 relax_f32_to_f16，执行以下操作
  if (relax_f32_to_f16) {
    check_nnapi->Model_relaxComputationFloat32toFloat16(model_.get(), true);

调用NNAPI模型对象的方法，将计算精度从Float32放宽到Float16。


  }
  check_nnapi->Model_finish(model_.get());

完成当前NNAPI模型对象的操作。


  ANeuralNetworksCompilation* compilation{};
  // 创建一个NNAPI编译对象，将模型绑定到编译对象上
  check_nnapi->Compilation_create(model_.get(), &compilation);
  // TODO: 将此部分设置为可配置项
  // 设置编译选项，指定编译的偏好设置为给定的编译优先级
  check_nnapi->Compilation_setPreference(compilation, static_cast<int32_t>(compilation_preference));
  // 完成编译操作
  check_nnapi->Compilation_finish(compilation);
  // 将编译对象封装到智能指针中，确保资源管理
  compilation_.reset(compilation);
}

// 实现 NnapiCompilation 类的 run 方法，接受输入和输出张量数组
void NnapiCompilation::run(
    std::vector<at::Tensor> inputs,
    std::vector<at::Tensor> outputs) {
  // 声明指向 ANeuralNetworksExecution 结构的指针
  ANeuralNetworksExecution* execution;
  // 使用 NNAPI 提供的函数创建执行对象，并将其赋给 execution
  check_nnapi->Execution_create(compilation_.get(), &execution);
  // 使用智能指针管理执行对象，确保在函数退出时执行对象被正确释放
  ExecutionPtr execution_unique_ptr(execution);

  // 检查输入张量数量是否与预期的输入数量相符
  TORCH_CHECK((int32_t)inputs.size() == num_inputs_);
  // 检查输出张量数量是否与预期的输出数量相符
  TORCH_CHECK((int32_t)outputs.size() == num_outputs_);

  // 遍历输入张量数组
  for (const auto i : c10::irange(inputs.size())) {
    auto& t = inputs[i];
    // TODO: 检查张量是否连续并且数据类型是否匹配
    // 获取张量的操作类型和维度信息
    ANeuralNetworksOperandType op_type;
    std::vector<uint32_t> dim;
    get_operand_type(t, &op_type, &dim);
    // 将输入张量设置到 NNAPI 执行对象中
    check_nnapi->Execution_setInput(
        execution,
        i,
        &op_type,
        t.data_ptr(),
        t.nbytes());
  }

  // 遍历输出张量数组
  for (const auto i : c10::irange(outputs.size())) {
    auto& t = outputs[i];
    // TODO: 检查张量是否连续并且数据类型是否匹配
    // 将输出张量设置到 NNAPI 执行对象中
    check_nnapi->Execution_setOutput(
        execution,
        i,
        nullptr,
        t.data_ptr(),
        t.nbytes());
  }

  // 执行 NNAPI 执行对象中的计算
  check_nnapi->Execution_compute(execution);

  // 遍历输出张量数组，根据执行结果调整张量的维度
  // TODO: 对于固定大小的输出可能可以跳过此步骤？
  for (const auto i : c10::irange(outputs.size())) {
    auto& t = outputs[i];
    // 获取输出张量的维度信息
    uint32_t rank;
    check_nnapi->Execution_getOutputOperandRank(execution, i, &rank);
    std::vector<uint32_t> dims(rank);
    check_nnapi->Execution_getOutputOperandDimensions(execution, i, dims.data());
    std::vector<int64_t> long_dims(dims.begin(), dims.end());
    // TODO: 可能需要检查只有批处理维度被修改？
    // 调整张量的大小以匹配输出维度
    t.resize_(long_dims);
  }
}

// 获取张量的操作类型信息
void NnapiCompilation::get_operand_type(const at::Tensor& t, ANeuralNetworksOperandType* operand, std::vector<uint32_t>* dims) {
  operand->dimensionCount = t.dim();
  // 检查维度是否与张量的实际维度匹配，以防溢出
  TORCH_CHECK(operand->dimensionCount == t.dim()); // 检查是否溢出
  dims->resize(t.dim());
  operand->dimensions = dims->data();
  // 遍历张量的维度，将其赋值给操作类型的维度数组
  for (const auto i : c10::irange(dims->size())) {
    (*dims)[i] = t.sizes()[i];
    // 再次检查维度是否与张量的实际维度匹配，以防溢出
    TORCH_CHECK((*dims)[i] == t.sizes()[i]); // 检查是否溢出
  }
  // 根据张量的数据类型设置操作类型的类型字段
  if (t.scalar_type() == c10::kFloat) {
    operand->type = ANEURALNETWORKS_TENSOR_FLOAT32;
    operand->scale = 0;
    operand->zeroPoint = 0;
    return;
  }
  if (t.scalar_type() == c10::kQUInt8) {
    TORCH_CHECK(t.is_quantized());
    operand->type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
    operand->scale = t.q_scale();
    operand->zeroPoint = t.q_zero_point();
    return;
  }
  if (t.scalar_type() == c10::kInt) {
    operand->type = ANEURALNETWORKS_TENSOR_INT32;
    operand->scale = 0;
    operand->zeroPoint = 0;
    return;
  }
  if (t.scalar_type() == c10::kShort) {
    TORCH_WARN(
      "NNAPI qint16 inputs to model are only supported for ",
      "testing with fixed scale, zero_point. Please change your ",
      "inputs if you see this in production");
    // 设置操作数的数据类型为 ANEURALNETWORKS_TENSOR_QUANT16_ASYMM
    operand->type = ANEURALNETWORKS_TENSOR_QUANT16_ASYMM;
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    // 设置操作数的缩放因子为 0.125（注意：此处存在类型转换警告）
    operand->scale = 0.125;
    // 设置操作数的零点为 0
    operand->zeroPoint = 0;
    // 函数执行完毕，返回
    return;
  }

  // TODO: Support more dtypes.
  // 抛出异常，指明不支持的数据类型，将具体数据类型转换为整数并转化为字符串形式作为异常信息
  CAFFE_THROW("Bad dtype: " + std::to_string(static_cast<int8_t>(t.scalar_type())));
}

} // namespace bind
} // namespace nnapi
} // namespace torch
```