# `.\pytorch\aten\src\ATen\native\mps\TensorFactory.h`

```py
// 定义一个宏来根据不同的MPS类型分发处理函数
#define AT_DISPATCH_MPS_TYPES(TYPE, NAME, ...)                          \
  // 调用一个宏来根据给定的类型进行分发处理
  AT_DISPATCH_SWITCH(                                                   \
      TYPE, NAME,                                                       \
      // 对于浮点型进行分发处理
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)              \
      // 对于半精度浮点型进行分发处理
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)               \
      // 对于长整型进行分发处理
      AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)               \
      // 对于整型进行分发处理
      AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)                \
      // 对于短整型进行分发处理
      AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)              \
      // 对于字符型进行分发处理
      AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)               \
      // 对于字节型进行分发处理
      AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__))
```