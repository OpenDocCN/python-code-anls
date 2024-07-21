# `.\pytorch\aten\src\ATen\test\cuda_apply_test.cpp`

```
/*
   包含 Google Test 框架的头文件
*/
#include <gtest/gtest.h>

/*
   包含 CUDA 的头文件
*/
#include <cuda.h>
#include <cuda_runtime.h>

/*
   包含 ATen 库中的 CUDA 相关头文件
*/
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAContext.h>

/*
   自定义宏，用于在测试中比较两个值是否相等
*/
#define ASSERT_EQ_CUDA(X, Y) \
  {                          \
    bool _isEQ = X == Y;     \
    ASSERT_TRUE(_isEQ);      \
  }

/*
   关于张量索引和操作应用的测试
*/
#ifndef _WIN32

// 测试用例: "2D Contiguous"，将二维连续张量压缩为一维连续张量
TEST(ApplyTest, Contiguous2D) {
  if (!at::cuda::is_available()) return;  // 如果 CUDA 不可用，则跳过测试
  int sizes[] = {4, 4};
  int strides[] = {4, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 2, sizes, strides};
  ti.collapseDims();  // 压缩张量的维度
  ASSERT_EQ_CUDA(ti.dims, 1);  // 断言压缩后的维度为1
  ASSERT_EQ_CUDA(ti.sizes[0], (4 * 4));  // 断言压缩后的大小为原来的大小
}

// 测试用例: "3D Contiguous"，将三维连续张量压缩为一维连续张量
TEST(ApplyTest, Contiguous3D) {
  if (!at::cuda::is_available()) return;  // 如果 CUDA 不可用，则跳过测试
  int sizes[] = {6, 3, 7};
  int strides[] = {3 * 7, 7, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ti.collapseDims();  // 压缩张量的维度
  ASSERT_EQ_CUDA(ti.dims, 1);  // 断言压缩后的维度为1
  ASSERT_EQ_CUDA(ti.sizes[0], (6 * 3 * 7));  // 断言压缩后的大小为原来的大小
}

// 测试用例: "3D Partial Collapse"，将三维非连续张量压缩为二维张量
TEST(ApplyTest, PartialCollapse3D) {
  if (!at::cuda::is_available()) return;  // 如果 CUDA 不可用，则跳过测试
  int sizes[] = {4, 3, 2};
  int strides[] = {3 * 3, 3, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ti.collapseDims();  // 压缩张量的维度
  ASSERT_EQ_CUDA(ti.dims, 2);  // 断言压缩后的维度为2
  ASSERT_EQ_CUDA(ti.sizes[0], (4 * 3));  // 断言压缩后的第一个维度大小
  ASSERT_EQ_CUDA(ti.sizes[1], 2);  // 断言压缩后的第二个维度大小
}

// 测试用例: "StridedCollapse2D"，将二维跳跃连续张量压缩为一维跳跃连续张量
TEST(ApplyTest, StridedCollapse2D) {
  if (!at::cuda::is_available()) return;  // 如果 CUDA 不可用，则跳过测试
  int sizes[] = {3, 2};
  int strides[] = {2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 2, sizes, strides};
  ti.collapseDims();  // 压缩张量的维度
  ASSERT_EQ_CUDA(ti.dims, 1);  // 断言压缩后的维度为1
  ASSERT_EQ_CUDA(ti.sizes[0], (3 * 2));  // 断言压缩后的大小为原来的大小
  ASSERT_EQ_CUDA(ti.strides[0], 2);  // 断言压缩后的步长为原来的步长
}

// 测试用例: "PartialStridedCollapse4D"，将四维张量部分压缩为二维张量
TEST(ApplyTest, PartialStridedCollapse4D) {
  if (!at::cuda::is_available()) return;  // 如果 CUDA 不可用，则跳过测试
  int sizes[] = {3, 6, 5, 2};
  int strides[] = {6 * 22, 22, 2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
  ti.collapseDims();  // 压缩张量的维度
  ASSERT_EQ_CUDA(ti.dims, 2);  // 断言压缩后的维度为2
  ASSERT_EQ_CUDA(ti.sizes[0], (3 * 6));  // 断言压缩后的第一个维度大小
  ASSERT_EQ_CUDA(ti.strides[0], 22);  // 断言压缩后的第一个维度步长
  ASSERT_EQ_CUDA(ti.sizes[1], (5 * 2));  // 断言压缩后的第二个维度大小
  ASSERT_EQ_CUDA(ti.strides[1], 2);  // 断言压缩后的第二个维度步长
}

// 测试用例: "CollapsesZerosAndOnes"，将五维张量压缩为一维张量
TEST(ApplyTest, CollapsesZerosAndOnes) {
  if (!at::cuda::is_available()) return;  // 如果 CUDA 不可用，则跳过测试
  int sizes[] = {1, 10, 1, 5, 4};
  int strides[] = {4, 0, 16, 0, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 5, sizes, strides};
  ti.collapseDims();  // 压缩张量的维度
  ASSERT_EQ_CUDA(ti.dims, 2);  // 断言压缩后的维度为2
  ASSERT_EQ_CUDA(ti.sizes[0], (10 * 5));  // 断言压缩后的第一个维度大小
  ASSERT_EQ_CUDA(ti.strides[0], 0);  // 断言压缩后的第一个维度步长
  ASSERT_EQ_CUDA(ti.sizes[1], 4);  // 断言压缩后的第二个维度大小
  ASSERT_EQ_CUDA(ti.strides[1], 1);  // 断言压缩后的第二个维度步长
}
// 定义一个测试用例，验证将4维张量折叠到3维张量的行为
TEST(ApplyTest, CollapseToPointTensor) {
  // 如果CUDA不可用，则跳过测试
  if (!at::cuda::is_available()) return;
  // 定义张量的尺寸和步长数组
  int sizes[] = {1, 1, 1};
  int strides[] = {17, 12, 3};
  // 创建一个TensorInfo对象，表示一个void类型的张量，使用指定的尺寸和步长
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  // 断言折叠维度操作返回0
  ASSERT_EQ_CUDA(ti.collapseDims(), 0);
  // 断言TensorInfo对象的维度现在为1
  ASSERT_EQ_CUDA(ti.dims, 1);
  // 断言张量的第一个维度大小为1
  ASSERT_EQ_CUDA(ti.sizes[0], 1);
  // 断言张量的第一个维度步长为1
  ASSERT_EQ_CUDA(ti.strides[0], 1);
}

// 定义一个测试用例，验证将4维张量折叠到3维张量的行为，排除连续性的维度
TEST(ApplyTest, ExcludingInContiguous4D) {
  // 如果CUDA不可用，则跳过测试
  if (!at::cuda::is_available()) return;
  // 定义张量的尺寸和步长数组
  int sizes[] = {3, 6, 5, 2};
  int strides[] = {6 * 22, 22, 2 * 2, 2};
  // 创建一个TensorInfo对象，表示一个void类型的张量，使用指定的尺寸和步长
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
  // 断言折叠维度操作返回1
  ASSERT_EQ_CUDA(ti.collapseDims(1), 1);
  // 断言TensorInfo对象的维度现在为3
  ASSERT_EQ_CUDA(ti.dims, 3);
  // 断言张量的第一个维度大小为3
  ASSERT_EQ_CUDA(ti.sizes[0], 3);
  // 断言张量的第一个维度步长为6 * 22
  ASSERT_EQ_CUDA(ti.strides[0], (6 * 22));
  // 断言张量的第二个维度大小为6
  ASSERT_EQ_CUDA(ti.sizes[1], 6);
  // 断言张量的第二个维度步长为22
  ASSERT_EQ_CUDA(ti.strides[1], 22);
  // 断言张量的第三个维度大小为5 * 2
  ASSERT_EQ_CUDA(ti.sizes[2], (5 * 2));
  // 断言张量的第三个维度步长为2
  ASSERT_EQ_CUDA(ti.strides[2], 2);
}

// 定义一个测试用例，验证将4维张量折叠到3维张量的行为，移动排除的维度
TEST(ApplyTest, RovingExclusion) {
  // 如果CUDA不可用，则跳过测试
  if (!at::cuda::is_available()) return;
  // 定义张量的尺寸和步长数组
  int sizes[] = {3, 6, 5, 2};
  int strides[] = {6 * 22, 22, 2 * 2, 2};
  // 创建一个TensorInfo对象，表示一个void类型的张量，使用指定的尺寸和步长
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
  // 断言折叠维度操作返回1
  ASSERT_EQ_CUDA(ti.collapseDims(2), 1);
  // 断言TensorInfo对象的维度现在为3
  ASSERT_EQ_CUDA(ti.dims, 3);
  // 断言张量的第一个维度大小为3 * 6
  ASSERT_EQ_CUDA(ti.sizes[0], (3 * 6));
  // 断言张量的第一个维度步长为22
  ASSERT_EQ_CUDA(ti.strides[0], 22);
  // 断言张量的第二个维度大小为5
  ASSERT_EQ_CUDA(ti.sizes[1], 5);
  // 断言张量的第二个维度步长为4
  ASSERT_EQ_CUDA(ti.strides[1], 4);
  // 断言张量的第三个维度大小为2
  ASSERT_EQ_CUDA(ti.sizes[2], 2);
  // 断言张量的第三个维度步长为2
  ASSERT_EQ_CUDA(ti.strides[2], 2);
}

// 定义一个测试用例，验证尝试排除一个不存在的维度时的行为
TEST(ApplyTest, InvalidExclusion) {
  // 如果CUDA不可用，则跳过测试
  if (!at::cuda::is_available()) return;
  // 定义张量的尺寸和步长数组
  int sizes[] = {1, 1, 1};
  int strides[] = {17, 12, 3};
  // 创建一个TensorInfo对象，表示一个void类型的张量，使用指定的尺寸和步长
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  // 断言在尝试排除一个不存在的维度时抛出异常
  ASSERT_ANY_THROW(ti.collapseDims(5));
}
```