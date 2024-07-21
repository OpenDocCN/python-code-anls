# `.\pytorch\aten\src\ATen\native\RNN.h`

```
#pragma once
// 使用预处理器指令#pragma once，确保头文件只被编译一次

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
// 包含 ATen 库的头文件，用于张量操作和分发机制

namespace at::native {
// 定义命名空间 at::native

using lstm_fn = void(*)(Tensor&, Tensor&, Tensor&, const Tensor&, TensorList, TensorList, bool, int64_t, double, bool, bool, bool);
using rnn_fn = void(*)(Tensor&, Tensor&, const Tensor&, const Tensor&, TensorList, bool, int64_t, double, bool, bool, bool);
using lstm_packed_fn = void(*)(Tensor&, Tensor&, Tensor&, const Tensor&, const Tensor&, TensorList, TensorList, bool, int64_t, double, bool, bool);
using rnn_packed_fn = void(*)(Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&, TensorList, bool, int64_t, double, bool, bool);

// 声明不同的 LSTM 和 RNN 分发函数类型

DECLARE_DISPATCH(lstm_fn, lstm_cudnn_stub);
DECLARE_DISPATCH(lstm_fn, lstm_miopen_stub);
DECLARE_DISPATCH(lstm_fn, lstm_mkldnn_stub);
DECLARE_DISPATCH(rnn_fn, gru_cudnn_stub);
DECLARE_DISPATCH(rnn_fn, gru_miopen_stub);
DECLARE_DISPATCH(rnn_fn, rnn_tanh_cudnn_stub);
DECLARE_DISPATCH(rnn_fn, rnn_tanh_miopen_stub);
DECLARE_DISPATCH(rnn_fn, rnn_relu_cudnn_stub);
DECLARE_DISPATCH(rnn_fn, rnn_relu_miopen_stub);
DECLARE_DISPATCH(lstm_packed_fn, lstm_packed_cudnn_stub);
DECLARE_DISPATCH(lstm_packed_fn, lstm_packed_miopen_stub);
DECLARE_DISPATCH(rnn_packed_fn, gru_packed_cudnn_stub);
DECLARE_DISPATCH(rnn_packed_fn, gru_packed_miopen_stub);
DECLARE_DISPATCH(rnn_packed_fn, rnn_tanh_packed_cudnn_stub);
DECLARE_DISPATCH(rnn_packed_fn, rnn_tanh_packed_miopen_stub);
DECLARE_DISPATCH(rnn_packed_fn, rnn_relu_packed_cudnn_stub);
DECLARE_DISPATCH(rnn_packed_fn, rnn_relu_packed_miopen_stub);

// 声明各种 LSTM 和 RNN 的分发函数，使用不同的后端加速库实现

inline void check_attributes(const Tensor& input, const TensorList& params, const TensorList& hiddens, bool check_dtype=false) {
  // 定义检查张量属性的函数，检查输入张量、参数张量和隐藏状态张量的设备和数据类型是否匹配

  auto input_device = input.device();
  auto input_dtype = input.scalar_type();

  auto check_tensors = [&](const std::string& name, const Tensor& t) {
    // 定义检查单个张量的闭包函数
    if (!t.defined()) return;
    auto t_device = t.device();
    TORCH_CHECK(input_device == t_device,
             "Input and ", name, " tensors are not at the same device, found input tensor at ",
             input_device, " and ", name, " tensor at ", t_device);
    if (check_dtype) {
      auto t_dtype = t.scalar_type();
      TORCH_CHECK(input_dtype == t_dtype,
               "Input and ", name, " tensors are not the same dtype, found input tensor with ",
               input_dtype, " and ", name, " tensor with ", t_dtype);
    }
  };

  // 循环检查所有隐藏状态张量和参数张量
  for (const auto& h : hiddens) check_tensors("hidden", h);
  for (const auto& p : params) check_tensors("parameter", p);
}

} // namespace at::native
// 结束命名空间 at::native
```