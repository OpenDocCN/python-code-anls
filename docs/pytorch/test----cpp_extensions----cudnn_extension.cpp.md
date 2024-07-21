# `.\pytorch\test\cpp_extensions\cudnn_extension.cpp`

```py
/*
 * CuDNN ReLU extension. Simple function but contains the general structure of
 * most CuDNN extensions:
 * 1) Check arguments. torch::check* functions provide a standard way to
 * validate input and provide pretty errors.
 * 2) Create descriptors. Most CuDNN functions require creating and setting a variety of descriptors.
 * 3) Apply the CuDNN function.
 * 4) Destroy your descriptors.
 * 5) Return something (optional).
 */

#include <torch/extension.h>

#include <ATen/cuda/Exceptions.h> // for CUDNN_CHECK
#include <ATen/cudnn/Descriptors.h> // for TensorDescriptor
#include <ATen/cudnn/Handle.h> // for getCudnnHandle

// Name of function in python module and name used for error messages by
// torch::check* functions.
const char* cudnn_relu_name = "cudnn_relu";

// Check arguments to cudnn_relu
void cudnn_relu_check(
    const torch::Tensor& inputs,
    const torch::Tensor& outputs) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_inputs(inputs, "inputs", 0);
  torch::TensorArg arg_outputs(outputs, "outputs", 1);
  
  // Check arguments. No need to return anything. These functions with throw an
  // error if they fail. Messages are populated using information from
  // TensorArgs.
  torch::checkContiguous(cudnn_relu_name, arg_inputs);  // Check if inputs tensor is contiguous
  torch::checkScalarType(cudnn_relu_name, arg_inputs, torch::kFloat);  // Check if inputs tensor has float scalar type
  torch::checkBackend(cudnn_relu_name, arg_inputs.tensor, torch::Backend::CUDA);  // Check if inputs tensor is on CUDA backend
  torch::checkContiguous(cudnn_relu_name, arg_outputs);  // Check if outputs tensor is contiguous
  torch::checkScalarType(cudnn_relu_name, arg_outputs, torch::kFloat);  // Check if outputs tensor has float scalar type
  torch::checkBackend(cudnn_relu_name, arg_outputs.tensor, torch::Backend::CUDA);  // Check if outputs tensor is on CUDA backend
  torch::checkSameSize(cudnn_relu_name, arg_inputs, arg_outputs);  // Check if inputs and outputs tensors have the same size
}
// 定义名为 cudnn_relu 的函数，实现 CuDNN 的 ReLU 激活函数操作
void cudnn_relu(const torch::Tensor& inputs, const torch::Tensor& outputs) {
  // 大多数 CuDNN 扩展遵循相似的模式。
  // Step 1: 检查输入。如果输入无效，此处会抛出错误，因此这里不需要检查返回码。
  cudnn_relu_check(inputs, outputs);
  
  // Step 2: 创建描述符
  cudnnHandle_t cuDnn = torch::native::getCudnnHandle();
  // 注意：4 是张量描述符的最小维度。输入和输出大小、类型都相同且连续，因此一个描述符就足够了。
  torch::native::TensorDescriptor input_tensor_desc(inputs, 4);
  
  cudnnActivationDescriptor_t activationDesc;
  // 注意：始终使用 CUDNN_CHECK 检查 cudnn 函数的返回值
  AT_CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
  AT_CUDNN_CHECK(cudnnSetActivationDescriptor(
      activationDesc,
      /*mode=*/CUDNN_ACTIVATION_RELU,
      /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
      /*coef=*/1.));
  
  // Step 3: 应用 CuDNN 函数
  float alpha = 1.;
  float beta = 0.;
  AT_CUDNN_CHECK(cudnnActivationForward(
      cuDnn,
      activationDesc,
      &alpha,
      input_tensor_desc.desc(),
      inputs.data_ptr(),
      &beta,
      input_tensor_desc.desc(), // 输出描述符与输入相同
      outputs.data_ptr()));
  
  // Step 4: 销毁描述符
  AT_CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
  
  // Step 5: 返回结果（可选）
}

// 创建 pybind11 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 使用与检查函数相同的名称，以便错误消息更合理
  m.def(cudnn_relu_name, &cudnn_relu, "CuDNN ReLU");
}
```