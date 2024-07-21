# `.\pytorch\aten\src\ATen\native\Padding.h`

```py
#pragma once

声明指令，确保头文件只被编译一次，防止重复包含。


#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

包含两个头文件，分别是ATen张量操作的核心头文件和分发存根的头文件。


namespace at::native {

进入`at::native`命名空间。


using padding_fn = void (*)(const Tensor&, const Tensor&, IntArrayRef);

定义一个函数指针类型`padding_fn`，指向一个接受两个`Tensor`对象和一个整数数组引用作为参数并返回`void`的函数。


// reflection padding
DECLARE_DISPATCH(padding_fn, reflection_pad1d_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad1d_backward_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad2d_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad2d_backward_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad3d_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad3d_backward_kernel);

// replication padding
DECLARE_DISPATCH(padding_fn, replication_pad1d_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad1d_backward_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad2d_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad2d_backward_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad3d_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad3d_backward_kernel);

声明了多个分发函数，用于不同类型的填充操作（反射填充和复制填充）的前向和反向内核。


namespace padding {

进入`padding`命名空间。


template <int dim>
inline void check_valid_input(const Tensor& input, IntArrayRef padding) {

定义模板函数`check_valid_input`，接受一个整数模板参数`dim`，以及一个`Tensor`对象`input`和一个整数数组引用`padding`作为参数。


  TORCH_CHECK(padding.size() == 2 * dim,
      "padding size is expected to be ", 2 * dim,
      ", but got: ", padding.size());

检查填充数组的大小是否为`2 * dim`，如果不是则抛出错误。


  int input_dim = input.dim();

获取输入张量的维度数。


  bool is_batch_mode = input_dim == (dim + 2);

判断是否处于批处理模式，即输入张量维度数是否为`dim + 2`。


  bool valid_batch_mode = is_batch_mode;
  bool valid_non_batch_mode = !is_batch_mode;

初始化两个布尔变量，用于检查批处理模式和非批处理模式的有效性。


  if (is_batch_mode) {
    // allow batch size of 0-dim.
    for (const auto d : c10::irange(1, input_dim)) {
      valid_batch_mode = valid_batch_mode && input.size(d) != 0;
    }
  } else {
    for (const auto d : c10::irange(0, input_dim)) {
      valid_non_batch_mode = valid_non_batch_mode && input.size(d) != 0;
    }
  }

根据是否处于批处理模式，分别检查输入张量的大小是否为零，并更新有效性布尔变量。


  // allow empty batch size but not other dimensions.
  TORCH_CHECK(valid_batch_mode || valid_non_batch_mode,
      "Expected ", dim + 1, "D or ", dim + 2,
      "D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());

最终检查批处理模式或非批处理模式是否有效，如果不是则抛出错误，说明预期的输入张量维度结构。


} // namespace padding

结束`padding`命名空间。


} // at::native

结束`at::native`命名空间。
```