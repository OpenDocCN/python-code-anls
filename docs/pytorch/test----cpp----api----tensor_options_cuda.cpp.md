# `.\pytorch\test\cpp\api\tensor_options_cuda.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 的头文件

#include <torch/torch.h>  // 包含 PyTorch 的头文件

#include <torch/cuda.h>   // 包含 PyTorch CUDA 相关的头文件

// NB: This file is compiled even in CPU build (for some reason), so
// make sure you don't include any CUDA only headers.
// 注意：即使在 CPU 构建中，此文件也会被编译（某些原因），因此确保不包含任何仅限于 CUDA 的头文件。

using namespace at;  // 使用 PyTorch 命名空间

// TODO: This might be generally helpful aliases elsewhere.
// 定义一个在其他地方可能有用的别名
at::Device CPUDevice() {
  return at::Device(at::kCPU);  // 返回一个 CPU 设备对象
}
at::Device CUDADevice(DeviceIndex index) {
  return at::Device(at::kCUDA, index);  // 返回指定 CUDA 设备对象
}

// A macro so we don't lose location information when an assertion fails.
// 定义一个宏，用于在断言失败时不丢失位置信息
#define REQUIRE_OPTIONS(device_, index_, type_, layout_)                  \
  ASSERT_EQ(options.device().type(), Device((device_), (index_)).type()); \
  ASSERT_TRUE(                                                            \
      options.device().index() == Device((device_), (index_)).index());   \
  ASSERT_EQ(typeMetaToScalarType(options.dtype()), (type_));              \
  ASSERT_TRUE(options.layout() == (layout_))

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  ASSERT_EQ(tensor.device().type(), Device((device_), (index_)).type());   \
  ASSERT_EQ(tensor.device().index(), Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.scalar_type(), (type_));                                \
  ASSERT_TRUE(tensor.options().layout() == (layout_))

TEST(TensorOptionsTest, ConstructsWellFromCUDATypes_CUDA) {
  auto options = CUDA(kFloat).options();  // 创建一个 CUDA 类型为 kFloat 的选项对象
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kStrided);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配

  options = CUDA(kInt).options();  // 创建一个 CUDA 类型为 kInt 的选项对象
  REQUIRE_OPTIONS(kCUDA, -1, kInt, kStrided);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配

  options = getDeprecatedTypeProperties(Backend::SparseCUDA, kFloat).options();  // 获取过时的 SparseCUDA kFloat 类型的属性，并创建选项对象
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kSparse);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配

  options = getDeprecatedTypeProperties(Backend::SparseCUDA, kByte).options();  // 获取过时的 SparseCUDA kByte 类型的属性，并创建选项对象
  REQUIRE_OPTIONS(kCUDA, -1, kByte, kSparse);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配

  // NOLINTNEXTLINE(bugprone-argument-comment,cppcoreguidelines-avoid-magic-numbers)
  options = CUDA(kFloat).options(/*device=*/5);  // 创建一个 CUDA 类型为 kFloat 的选项对象，指定设备为 5
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kStrided);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配

  options =
      // NOLINTNEXTLINE(bugprone-argument-comment,cppcoreguidelines-avoid-magic-numbers)
      getDeprecatedTypeProperties(Backend::SparseCUDA, kFloat)
          .options(/*device=*/5);  // 获取过时的 SparseCUDA kFloat 类型的属性，并创建选项对象，指定设备为 5
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kSparse);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配
}

TEST(TensorOptionsTest, ConstructsWellFromCUDATensors_MultiCUDA) {
  auto options = empty(5, device(kCUDA).dtype(kDouble)).options();  // 创建一个大小为 5 的双精度张量，指定为 CUDA 设备，创建选项对象
  REQUIRE_OPTIONS(kCUDA, 0, kDouble, kStrided);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配

  options = empty(5, getDeprecatedTypeProperties(Backend::SparseCUDA, kByte))
                .options();  // 创建一个大小为 5 的稀疏字节张量，指定为 CUDA 设备，创建选项对象
  REQUIRE_OPTIONS(kCUDA, 0, kByte, kSparse);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配

  if (torch::cuda::device_count() > 1) {
    Tensor tensor;
    {
      DeviceGuard guard(CUDADevice(1));  // 在 CUDA 设备 1 上创建设备保护
      tensor = empty(5, device(kCUDA));   // 创建一个大小为 5 的 CUDA 设备张量
    }
    options = tensor.options();  // 获取张量的选项对象
    REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配

    {
      DeviceGuard guard(CUDADevice(1));  // 在 CUDA 设备 1 上创建设备保护
      tensor = empty(5, device(kCUDA).layout(kSparse));  // 创建一个大小为 5 的稀疏布局 CUDA 设备张量
    }
    options = tensor.options();  // 获取张量的选项对象
    REQUIRE_OPTIONS(kCUDA, 1, kFloat, kSparse);  // 断言选项对象的属性与预期的 CUDA 设备、类型、布局匹配
  }
}
```