# `.\pytorch\aten\src\ATen\native\mkldnn\Copy.cpp`

```
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/copy_native.h>
#endif


// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含 NativeFunctions.h 文件；否则，包含 copy_native.h 文件。



#if !AT_MKLDNN_ENABLED()


// 如果未启用 AT_MKLDNN_ENABLED 宏，则进入此条件判断块。



namespace at {
namespace native {


// 命名空间 at 和 native 的开始。



Tensor& copy_mkldnn_(Tensor& self, const Tensor& src, bool non_blocking) {


// 定义函数 copy_mkldnn_，接收三个参数：self（目标张量）、src（源张量）、non_blocking（是否非阻塞）。



TORCH_CHECK(false, "copy_mkldnn_: ATen not compiled with MKLDNN support");


// 使用 TORCH_CHECK 宏检查条件，如果为 false，则输出错误信息 "copy_mkldnn_: ATen not compiled with MKLDNN support"。



} // namespace native
} // namespace at


// 命名空间 native 和 at 的结束。



#else // AT_MKLDNN_ENABLED


// 如果定义了 AT_MKLDNN_ENABLED 宏，则进入此条件判断块。



TORCH_CHECK(
    self.sizes() == src.sizes(),
    "copy_mkldnn_: only support same size tensor.");


// 使用 TORCH_CHECK 宏检查条件，确保 self 和 src 张量具有相同的尺寸，否则输出错误信息 "copy_mkldnn_: only support same size tensor."。



TORCH_CHECK(
    self.is_mkldnn() && src.is_mkldnn(),
    "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! Found self type = ",
    self.toString(),
    " and src type = ",
    src.toString());


// 使用 TORCH_CHECK 宏检查条件，确保 self 和 src 张量均为 MKLDNN 布局，否则输出错误信息 "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! Found self type = " + self.toString() + " and src type = " + src.toString()。



ideep::tensor& x = itensor_from_mkldnn(src);
ideep::tensor& y = itensor_from_mkldnn(self);


// 创建 ideep::tensor 的引用 x 和 y，分别从 src 和 self 的 MKLDNN 张量中获取。



ideep::direct_copy::compute(x, y);


// 调用 ideep::direct_copy::compute 函数，将 x 中的数据直接复制到 y 中。



return self;


// 返回修改后的 self 张量的引用。



#endif // AT_MKLDNN_ENABLED


// AT_MKLDNN_ENABLED 宏条件编译块的结束。



#include <ATen/native/mkldnn/MKLDNNCommon.h>


// 包含 MKLDNNCommon.h 文件，提供 MKLDNN 相关的通用功能。



namespace at {
namespace native {


// 命名空间 at 和 native 的开始。



Tensor& copy_mkldnn_(Tensor& self, const Tensor& src, bool non_blocking) {


// 定义函数 copy_mkldnn_，接收三个参数：self（目标张量）、src（源张量）、non_blocking（是否非阻塞）。



TORCH_CHECK(
    self.sizes() == src.sizes(),
    "copy_mkldnn_: only support same size tensor.");


// 使用 TORCH_CHECK 宏检查条件，确保 self 和 src 张量具有相同的尺寸，否则输出错误信息 "copy_mkldnn_: only support same size tensor."。



TORCH_CHECK(
    self.is_mkldnn() && src.is_mkldnn(),
    "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! Found self type = ",
    self.toString(),
    " and src type = ",
    src.toString());


// 使用 TORCH_CHECK 宏检查条件，确保 self 和 src 张量均为 MKLDNN 布局，否则输出错误信息 "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! Found self type = " + self.toString() + " and src type = " + src.toString()。



ideep::tensor& x = itensor_from_mkldnn(src);
ideep::tensor& y = itensor_from_mkldnn(self);


// 创建 ideep::tensor 的引用 x 和 y，分别从 src 和 self 的 MKLDNN 张量中获取。



ideep::direct_copy::compute(x, y);


// 调用 ideep::direct_copy::compute 函数，将 x 中的数据直接复制到 y 中。



return self;


// 返回修改后的 self 张量的引用。



} // namespace native
} // namespace at


// 命名空间 native 和 at 的结束。



#endif // AT_MKLDNN_ENABLED


// AT_MKLDNN_ENABLED 宏条件编译块的结束。
```