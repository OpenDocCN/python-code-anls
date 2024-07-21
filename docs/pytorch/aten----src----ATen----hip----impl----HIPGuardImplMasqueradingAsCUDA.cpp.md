# `.\pytorch\aten\src\ATen\hip\impl\HIPGuardImplMasqueradingAsCUDA.cpp`

```py
// 在这里导入 ATen 的头文件，其中包含了一些 HIPGuardImplMasqueradingAsCUDA 的实现
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA>

// 这是一个巨大的破解。如果加载了 ATen_hip，即使在运行时从未使用 ATen_hip，也会破坏您的 Caffe2 CUDA 代码。
//
// 如果将 ATen_hip 静态链接到整个库中（与 ATen_cuda 同时链接，如 libomnibus），那么此库与常规的 ATen_cuda 的加载顺序是不确定的，
// 您将无法确定是使用哪个版本（会很明显，因为您的所有代码都会失败）。
//
// 一旦 PyTorch 完全支持 HIP，不再假装 CUDA 是 HIP，就可以移除这个破解。
C10_REGISTER_GUARD_IMPL(CUDA, at::cuda::HIPGuardImplMasqueradingAsCUDA);
```