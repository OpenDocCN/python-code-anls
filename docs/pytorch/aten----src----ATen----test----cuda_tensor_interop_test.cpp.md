# `.\pytorch\aten\src\ATen\test\cuda_tensor_interop_test.cpp`

```
// 包含 Google Test 的头文件
#include <gtest/gtest.h>

// 包含 PyTorch 的 ATen 头文件
#include <ATen/ATen.h>
// 包含 PyTorch 的 CUDA 上下文管理头文件
#include <ATen/cuda/CUDAContext.h>

// 包含 Caffe2 的初始化头文件
#include <caffe2/core/init.h>
// 包含 Caffe2 的操作符定义头文件
#include <caffe2/core/operator.h>
// 包含 Caffe2 的 CUDA 上下文头文件
#include <caffe2/core/context_gpu.h>
// 包含 Caffe2 的数学计算头文件
#include <caffe2/utils/math.h>

// 定义一个模板函数 cuda_get，用于从 CUDA 地址 addr 处获取数据并返回
template<typename T>
T cuda_get(T* addr) {
  T result;
  // 使用 CUDA 强制执行函数从地址 addr 处复制 sizeof(T) 大小的数据到 result 中
  CUDA_ENFORCE(cudaMemcpy(&result, addr, sizeof(T), cudaMemcpyDefault));
  return result;
}

// 定义一个模板函数 cuda_set，用于将 value 设置到 CUDA 地址 addr 处
template<typename T>
void cuda_set(T* addr, T value) {
  // 使用 CUDA 强制执行函数将 value 的值复制到地址 addr 处，数据大小为 sizeof(T)
  CUDA_ENFORCE(cudaMemcpy(addr, &value, sizeof(T), cudaMemcpyDefault));
}

// 测试用例，测试 Caffe2 到 PyTorch 的简单数据传输（Legacy 版本）
TEST(CUDACaffe2ToPytorch, SimpleLegacy) {
  // 如果 CUDA 不可用，则直接返回
  if (!at::cuda::is_available()) return;
  
  // 创建一个 Caffe2 的 Tensor 对象，并设定在 CUDA 上
  caffe2::Tensor c2_tensor(caffe2::CUDA);
  // 调整 Tensor 大小为 4x4
  c2_tensor.Resize(4, 4);
  // 获取 Tensor 数据的可变指针
  auto data = c2_tensor.mutable_data<int64_t>();
  {
    // 创建 CUDA 上下文
    caffe2::CUDAContext context;
    // 使用数学库设置 Tensor 中的数据为 777
    caffe2::math::Set<int64_t>(16, 777, data, &context);
  }
  // 将 Caffe2 的 Tensor 转换为 PyTorch 的 Tensor
  at::Tensor at_tensor(c2_tensor);
  // 断言转换后的 Tensor 在 CUDA 上
  ASSERT_TRUE(at_tensor.is_cuda());

  // 获取在 CPU 上的 Tensor 数据指针
  auto at_cpu = at_tensor.cpu();
  auto it = at_cpu.data_ptr<int64_t>();
  // 遍历数据并断言每个元素都为 777
  for (const auto i : c10::irange(16)) {
    ASSERT_EQ(it[i], 777);
  }
}

// 测试用例，测试 Caffe2 到 PyTorch 的简单数据传输
TEST(CUDACaffe2ToPytorch, Simple) {
  // 如果 CUDA 不可用，则直接返回
  if (!at::cuda::is_available()) return;
  
  // 创建一个空的 Caffe2 Tensor，设定在 CUDA 上，并指定大小为 4x4
  caffe2::Tensor c2_tensor =
      caffe2::empty({4, 4}, at::dtype<int64_t>().device(caffe2::CUDA));
  // 获取 Tensor 数据的可变指针
  auto data = c2_tensor.mutable_data<int64_t>();
  {
    // 创建 CUDA 上下文
    caffe2::CUDAContext context;
    // 使用数学库设置 Tensor 中的数据为 777
    caffe2::math::Set<int64_t>(16, 777, data, &context);
  }
  // 将 Caffe2 的 Tensor 转换为 PyTorch 的 Tensor
  at::Tensor at_tensor(c2_tensor);
  // 断言转换后的 Tensor 在 CUDA 上
  ASSERT_TRUE(at_tensor.is_cuda());

  // 获取在 CPU 上的 Tensor 数据指针
  auto at_cpu = at_tensor.cpu();
  auto it = at_cpu.data_ptr<int64_t>();
  // 遍历数据并断言每个元素都为 777
  for (const auto i : c10::irange(16)) {
    ASSERT_EQ(it[i], 777);
  }
}

// 测试用例，测试 Caffe2 到 PyTorch 的操作
TEST(CUDACaffe2ToPytorch, Op) {
  // 如果 CUDA 不可用，则直接返回
  if (!at::cuda::is_available()) return;
  
  // 创建一个空的 Caffe2 Tensor，设定在 CUDA 上，并指定大小为 3x3
  caffe2::Tensor c2_tensor =
      caffe2::empty({3, 3}, at::dtype<int64_t>().device(caffe2::CUDA));
  // 获取 Tensor 数据的可变指针
  auto data = c2_tensor.mutable_data<int64_t>();
  {
    // 创建 CUDA 上下文
    caffe2::CUDAContext context;
    // 使用数学库设置 Tensor 中的数据为 111
    caffe2::math::Set<int64_t>(9, 111, data, &context);
  }
  // 将 Caffe2 的 Tensor 转换为 PyTorch 的 Tensor
  at::Tensor at_tensor(c2_tensor);
  // 断言转换后的 Tensor 在 CUDA 上

  // 断言所有元素的和为 999
  ASSERT_EQ(at::sum(at_tensor).item<int64_t>(), 999);
}

// 测试用例，测试 PyTorch 到 Caffe2 的操作
TEST(CUDAPytorchToCaffe2, Op) {
  // 如果 CUDA 不可用，则直接返回
  if (!at::cuda::is_available()) return;
  
  // 创建一个 Caffe2 的 Workspace 和 NetDef
  caffe2::Workspace workspace;
  caffe2::NetDef net;

  // 创建三个在 CUDA 上的 PyTorch Tensor，均为大小为 5x5 的全 1 Tensor
  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));
  auto at_tensor_b = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));
  auto at_tensor_c = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));

  // 在 Workspace 中创建名为 "a" 和 "b" 的 Blob，并将对应的 PyTorch Tensor 转换为 Caffe2 的 Tensor
  auto* c2_tensor_a = BlobSetTensor(workspace.CreateBlob("a"), caffe2::Tensor(at_tensor_a));
  auto* c2_tensor_b = BlobSetTensor(workspace.CreateBlob("b"), caffe2::Tensor(at_tensor_b));
  (void)c2_tensor_a;
  (void)c2_tensor_b;

  // 测试别名
  {
    // 创建一个 Caffe2 的 Tensor，从现有的 PyTorch Tensor at_tensor_c 中生成别名
    caffe2::Tensor c2_tensor_from_aten(at_tensor_c);
    // 在 Workspace 中创建名为 "c" 的 Blob，并将别名设置为其内容
    BlobSetTensor(workspace.CreateBlob("c"), c2_tensor_from_aten.Alias());
  }

  {
    // 向 NetDef 中添加一个类型为 "Sum" 的操作
    auto op = net.add_op();
    op->set_type("Sum");
    // 添加输入为 "a", "b", "c"，输出为 "d"
    op->add_input("a");
    op->add_input("b");
    op->add_input("c");
    op->add_output("d");
    // 设置操作的设备选项为 CUDA，并将其设置为可变选项
    op->mutable_device_option()->set_device_type(caffe2::PROTO_CUDA);
  }

  // 在工作空间中运行一次神经网络 net
  workspace.RunNetOnce(net);

  // 获取工作空间中名为 "d" 的 Blob，并将其转换为 caffe2::Tensor 类型的引用
  const auto& result = workspace.GetBlob("d")->Get<caffe2::Tensor>();

  // 断言结果的设备类型为 CUDA
  ASSERT_EQ(result.GetDeviceType(), caffe2::CUDA);

  // 获取结果的数据指针，并遍历前 25 个元素
  auto data = result.data<float>();
  for (const auto i : c10::irange(25)) {
    // 断言 CUDA 设备上的数据值为 3.0
    ASSERT_EQ(cuda_get(data + i), 3.0);
  }

  // 将 caffe2::Tensor 转换为 at::Tensor 类型
  at::Tensor at_result(result);

  // 断言 at::Tensor 对象在 CUDA 设备上
  ASSERT_TRUE(at_result.is_cuda());

  // 断言 at::Tensor 对象所有元素值之和为 75.0
  ASSERT_EQ(at::sum(at_result).item<float>(), 75);
}

TEST(CUDAPytorchToCaffe2, SharedStorageWrite) {
  // 检查当前环境是否支持 CUDA，如果不支持则退出测试
  if (!at::cuda::is_available()) return;

  // 创建一个大小为 5x5 的全一张量，并放置在 CUDA 设备上
  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));
  
  // 将 at_tensor_a 重塑为大小为 25 的张量
  auto at_tensor_b = at_tensor_a.view({25});

  // 使用 at_tensor_a 创建一个 Caffe2 张量 c2_tensor_a
  caffe2::Tensor c2_tensor_a(at_tensor_a);
  
  // 使用 at_tensor_b 创建一个 Caffe2 张量 c2_tensor_b
  caffe2::Tensor c2_tensor_b(at_tensor_b);

  // 在 c2_tensor_a 中的数据上进行更改，该更改在所有引用中可见
  cuda_set<float>(c2_tensor_a.mutable_data<float>() + 1, 123);
  
  // 断言 c2_tensor_b 中相应位置的数据与之前设置的值相等
  ASSERT_EQ(cuda_get(c2_tensor_b.mutable_data<float>() + 1), 123);
  
  // 断言 PyTorch 张量 at_tensor_a 中相应位置的数据是否与设置的值相等
  ASSERT_EQ(at_tensor_a[0][1].item().to<float>(), 123);
  
  // 断言 PyTorch 张量 at_tensor_b 中相应位置的数据是否与设置的值相等
  ASSERT_EQ(at_tensor_b[1].item().to<float>(), 123);
}

TEST(CUDAPytorchToCaffe2, MutualResizes) {
  // 检查当前环境是否支持 CUDA，如果不支持则退出测试
  if (!at::cuda::is_available()) return;

  // 创建一个大小为 5x5 的全一张量，并放置在 CUDA 设备上
  auto at_tensor = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));

  // 使用 PyTorch 张量 at_tensor 创建一个 Caffe2 张量 c2_tensor
  caffe2::Tensor c2_tensor(at_tensor);

  // 在 c2_tensor 中的数据上进行更改，该更改在所有引用中可见
  cuda_set<float>(c2_tensor.mutable_data<float>(), 123);
  
  // 断言 PyTorch 张量 at_tensor 中相应位置的数据是否与设置的值相等
  ASSERT_EQ(at_tensor[0][0].item().to<float>(), 123);

  // 将 PyTorch 张量 at_tensor 的大小调整为较小的尺寸，存储空间保持不变
  at_tensor.resize_({4, 4});
  
  // 在 c2_tensor 中的数据上进行更改，确保数据仍然可以正确访问
  cuda_set<float>(c2_tensor.mutable_data<float>() + 1, 234);
  
  // 断言 PyTorch 张量 at_tensor 中相应位置的数据是否与设置的值相等
  ASSERT_EQ(at_tensor[0][1].item().to<float>(), 234);

  // 将 PyTorch 张量 at_tensor 的大小调整为较大的尺寸，存储空间保持不变
  at_tensor.resize_({6, 6});
  
  // 在 c2_tensor 中的数据上进行更改，确保数据仍然可以正确访问
  cuda_set<float>(c2_tensor.mutable_data<float>() + 2, 345);
  
  // 断言 PyTorch 张量 at_tensor 中相应位置的数据是否与设置的值相等
  ASSERT_EQ(at_tensor[0][2].item().to<float>(), 345);
  
  // 断言 c2_tensor 的尺寸是否正确变为 6x6
  ASSERT_EQ(c2_tensor.sizes()[0], 6);
  ASSERT_EQ(c2_tensor.sizes()[1], 6);

  // 调整 Caffe2 张量 c2_tensor 的大小，语义上不保留数据，但 TensorImpl 仍然是共享的
  c2_tensor.Resize(7, 7);
  
  // 在 c2_tensor 中的数据上进行更改，确保数据仍然可以正确访问
  cuda_set<float>(c2_tensor.mutable_data<float>() + 3, 456);
  
  // 断言 PyTorch 张量 at_tensor 中相应位置的数据是否与设置的值相等
  ASSERT_EQ(at_tensor[0][3].item().to<float>(), 456);
  
  // 断言 PyTorch 张量 at_tensor 的尺寸是否正确变为 7x7
  ASSERT_EQ(at_tensor.sizes()[0], 7);
  ASSERT_EQ(at_tensor.sizes()[1], 7);
}
```