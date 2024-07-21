# `.\pytorch\aten\src\ATen\native\TypeProperties.h`

```py
#pragma once


#include <ATen/core/Tensor.h>
#include <ATen/core/IListRef.h>


namespace at::native {


struct ResultTypeState {
  c10::ScalarType dimResult = ScalarType::Undefined;
  c10::ScalarType wrappedResult = ScalarType::Undefined;
  c10::ScalarType zeroResult = ScalarType::Undefined;
};


// 声明一个函数 update_result_type_state，用于更新结果类型的状态，接受一个张量和当前状态作为参数
TORCH_API ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state);

// 声明一个函数 update_result_type_state，用于更新结果类型的状态，接受一个标量和当前状态作为参数
TORCH_API ResultTypeState update_result_type_state(const Scalar& scalar, const ResultTypeState& in_state);

// 声明一个函数 result_type，根据给定的状态返回相应的标量类型
TORCH_API ScalarType result_type(const ResultTypeState& state);

// 声明一个函数 result_type，根据张量列表返回相应的结果类型
TORCH_API ScalarType result_type(ITensorListRef tensors);

} // namespace at::native
```