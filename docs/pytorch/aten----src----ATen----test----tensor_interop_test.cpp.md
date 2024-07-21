# `.\pytorch\aten\src\ATen\test\tensor_interop_test.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/ATen.h>  // 引入 PyTorch C++ 前端的头文件
#include <c10/util/irange.h>  // 引入 Caffe2 和 PyTorch 公用的迭代器工具
#include <caffe2/core/init.h>  // 引入 Caffe2 核心初始化头文件
#include <caffe2/core/operator.h>  // 引入 Caffe2 核心操作符头文件

TEST(Caffe2ToPytorch, SimpleLegacy) {  // 定义 Google Test 单元测试，测试 Caffe2 到 PyTorch 的转换（旧式API）
  caffe2::Tensor c2_tensor(caffe2::CPU);  // 创建一个 Caffe2 张量在 CPU 上
  c2_tensor.Resize(4, 4);  // 调整张量大小为 4x4
  auto data = c2_tensor.mutable_data<int64_t>();  // 获取张量的可变数据指针

  // 填充张量数据为 0 到 15
  for (const auto i : c10::irange(16)) {
    data[i] = i;
  }

  at::Tensor at_tensor(c2_tensor);  // 将 Caffe2 张量转换为 PyTorch 张量

  auto it = at_tensor.data_ptr<int64_t>();  // 获取 PyTorch 张量的数据指针
  // 验证 PyTorch 张量数据与预期一致
  for (const auto i : c10::irange(16)) {
    ASSERT_EQ(it[i], i);
  }
}

TEST(Caffe2ToPytorch, Simple) {  // 定义 Google Test 单元测试，测试 Caffe2 到 PyTorch 的转换（新式API）
  caffe2::Tensor c2_tensor = caffe2::empty({4, 4}, at::kLong);  // 创建一个空的 4x4 长整型 Caffe2 张量
  auto data = c2_tensor.mutable_data<int64_t>();  // 获取张量的可变数据指针

  // 填充张量数据为 0 到 15
  for (const auto i : c10::irange(16)) {
    data[i] = i;
  }

  at::Tensor at_tensor(c2_tensor);  // 将 Caffe2 张量转换为 PyTorch 张量

  auto it = at_tensor.data_ptr<int64_t>();  // 获取 PyTorch 张量的数据指针
  // 验证 PyTorch 张量数据与预期一致
  for (const auto i : c10::irange(16)) {
    ASSERT_EQ(it[i], i);
  }
}

TEST(Caffe2ToPytorch, ExternalData) {  // 定义 Google Test 单元测试，测试使用外部数据的情况
  caffe2::Tensor c2_tensor = caffe2::empty({4, 4}, at::kLong);  // 创建一个空的 4x4 长整型 Caffe2 张量
  int64_t buf[16];  // 声明一个包含 16 个长整型的缓冲区

  // 填充缓冲区数据为 0 到 15
  for (const auto i : c10::irange(16)) {
    buf[i] = i;
  }

  c2_tensor.ShareExternalPointer(buf, 16 * sizeof(int64_t));  // 将外部缓冲区数据与 Caffe2 张量共享

  at::Tensor at_tensor(c2_tensor);  // 将 Caffe2 张量转换为 PyTorch 张量
  at_tensor.permute({1, 0});  // 执行张量的维度置换操作
  at_tensor.permute({1, 0});  // 再次执行张量的维度置换操作

  auto it = at_tensor.data_ptr<int64_t>();  // 获取 PyTorch 张量的数据指针
  // 验证 PyTorch 张量数据与预期一致
  for (const auto i : c10::irange(16)) {
    ASSERT_EQ(it[i], i);
  }

  ASSERT_FALSE(at_tensor.storage().resizable());  // 验证 PyTorch 张量的存储是否可调整大小
  ASSERT_ANY_THROW(at_tensor.resize_({7,7}));  // 验证尝试调整 PyTorch 张量大小时是否抛出异常
}

TEST(Caffe2ToPytorch, Op) {  // 定义 Google Test 单元测试，测试 Caffe2 操作
  caffe2::Tensor c2_tensor(caffe2::CPU);  // 创建一个 Caffe2 张量在 CPU 上
  c2_tensor.Resize(3, 3);  // 调整张量大小为 3x3
  auto data = c2_tensor.mutable_data<int64_t>();  // 获取张量的可变数据指针

  // 填充张量数据为 0 到 8
  for (const auto i : c10::irange(9)) {
    data[i] = i;
  }

  at::Tensor at_tensor(c2_tensor);  // 将 Caffe2 张量转换为 PyTorch 张量

  // 验证 PyTorch 张量所有元素之和为 36
  ASSERT_EQ(at::sum(at_tensor).item<int64_t>(), 36);
}

TEST(Caffe2ToPytorch, PartiallyInitialized) {  // 定义 Google Test 单元测试，测试部分初始化的情况
  // 测试部分初始化的张量是否能够被捕获
  {
    // 没有数据类型和存储的张量，预期会抛出异常
    caffe2::Tensor c2_tensor(caffe2::CPU);
    ASSERT_ANY_THROW(at::Tensor at_tensor(c2_tensor));
  }
  {
    // 有存储但没有数据类型的张量，预期会抛出异常
    caffe2::Tensor c2_tensor(caffe2::CPU);
    c2_tensor.Resize(4,4);
    ASSERT_ANY_THROW(at::Tensor at_tensor(c2_tensor));
  }
  {
    // 有数据类型但没有存储的张量，预期会抛出异常
    caffe2::Tensor c2_tensor(caffe2::CPU);
    c2_tensor.Resize(4, 4);
    c2_tensor.mutable_data<int64_t>();
    ASSERT_ANY_THROW(at::Tensor at_tensor(c2_tensor));
  }
}
    // 创建一个名为 c2_tensor 的 caffe2::Tensor 对象，使用 CPU 设备
    caffe2::Tensor c2_tensor(caffe2::CPU);
    // 调整张量大小为 4x4
    c2_tensor.Resize(4,4);
    // 获取可修改的数据指针，数据类型为 float
    c2_tensor.mutable_data<float>();
    // 释放张量的内存
    c2_tensor.FreeMemory();
    // 使用 ASSERT_ANY_THROW 宏来检测是否抛出异常，此处检测是否能够从 c2_tensor 创建 at::Tensor 对象
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto) 是对静态代码分析工具的指导，告诉它忽略特定的警告
    ASSERT_ANY_THROW(at::Tensor at_tensor(c2_tensor));
  }
}

// 定义单元测试 Caffe2ToPytorch.MutualResizes
TEST(Caffe2ToPytorch, MutualResizes) {
  // 创建一个大小为 5x5 的空白 Caffe2 张量
  caffe2::Tensor c2_tensor = caffe2::empty({5, 5}, at::kFloat);
  // 获取可变的数据指针，用于填充数据
  auto data = c2_tensor.mutable_data<float>();
  // 使用循环将所有元素初始化为 0
  for (const auto i : c10::irange(25)) {
    data[i] = 0;
  }

  // 将 Caffe2 张量转换为 PyTorch 张量
  at::Tensor at_tensor(c2_tensor);

  // 修改后可以在两个张量之间看到变化
  at_tensor[0][0] = 123;
  ASSERT_EQ(c2_tensor.mutable_data<float>()[0], 123);

  // 在较小的方向上重新调整 PyTorch 张量的大小 - 存储空间保持不变
  at_tensor.resize_({4, 4});
  c2_tensor.mutable_data<float>()[1] = 234;
  ASSERT_EQ(at_tensor[0][1].item().to<float>(), 234);

  // 在较大的方向上重新调整 PyTorch 张量的大小 - 存储空间保持不变
  at_tensor.resize_({6, 6});
  c2_tensor.mutable_data<float>()[2] = 345;
  ASSERT_EQ(at_tensor[0][2].item().to<float>(), 345);
  ASSERT_EQ(c2_tensor.sizes()[0], 6);
  ASSERT_EQ(c2_tensor.sizes()[1], 6);

  // 调整 Caffe2 张量的大小 - 语义是不保留数据，但 TensorImpl 仍然共享
  c2_tensor.Resize(7, 7);
  c2_tensor.mutable_data<float>()[3] = 456;
  ASSERT_EQ(at_tensor[0][3].item().to<float>(), 456);
  ASSERT_EQ(at_tensor.sizes()[0], 7);
  ASSERT_EQ(at_tensor.sizes()[1], 7);
}

// 定义单元测试 PytorchToCaffe2.Op
TEST(PytorchToCaffe2, Op) {
  // 创建一个 Caffe2 工作空间和网络定义
  caffe2::Workspace workspace;
  caffe2::NetDef net;

  // 创建三个大小为 5x5 的全为 1 的 PyTorch 张量
  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat));
  auto at_tensor_b = at::ones({5, 5}, at::dtype(at::kFloat));
  auto at_tensor_c = at::ones({5, 5}, at::dtype(at::kFloat));

  // 将 PyTorch 张量 at_tensor_a 转换为 Caffe2 张量，并存储在工作空间中
  auto* c2_tensor_a = BlobSetTensor(workspace.CreateBlob("a"), caffe2::Tensor(at_tensor_a));
  // 将 PyTorch 张量 at_tensor_b 转换为 Caffe2 张量，并存储在工作空间中
  auto* c2_tensor_b = BlobSetTensor(workspace.CreateBlob("b"), caffe2::Tensor(at_tensor_b));

  // 测试别名操作，将 PyTorch 张量 at_tensor_c 转换为 Caffe2 张量并存储在工作空间中
  {
    caffe2::Tensor c2_tensor_from_aten(at_tensor_c);
    BlobSetTensor(workspace.CreateBlob("c"), c2_tensor_from_aten.Alias());
  }

  // 在网络定义中添加 Sum 操作，并指定输入输出
  {
    auto op = net.add_op();
    op->set_type("Sum");
    op->add_input("a");
    op->add_input("b");
    op->add_input("c");
    op->add_output("d");
  }

  // 在工作空间中执行网络
  workspace.RunNetOnce(net);

  // 从工作空间获取可变的结果张量，并将其存储为 PyTorch 张量
  auto result = XBlobGetMutableTensor(workspace.CreateBlob("d"), {5, 5}, at::kCPU);

  // 遍历结果张量中的数据，确保所有元素为 3.0
  auto it = result.data<float>();
  for (const auto i : c10::irange(25)) {
    ASSERT_EQ(it[i], 3.0);
  }

  // 将结果张量转换为 PyTorch 张量，并验证其总和为 75
  at::Tensor at_result(result);
  ASSERT_EQ(at::sum(at_result).item<float>(), 75);
}

// 定义单元测试 PytorchToCaffe2.SharedStorageRead
TEST(PytorchToCaffe2, SharedStorageRead) {
  // 创建一个 Caffe2 工作空间和网络定义
  caffe2::Workspace workspace;
  caffe2::NetDef net;

  // 创建一个全为 1 的大小为 5x5 的 PyTorch 张量 at_tensor_a
  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat));
  // 创建一个与 at_tensor_a 共享存储的 PyTorch 张量 at_tensor_b
  auto at_tensor_b = at_tensor_a.view({5, 5});

  // 将 PyTorch 张量 at_tensor_a 转换为 Caffe2 张量，并存储在工作空间中
  auto* c2_tensor_a = BlobSetTensor(workspace.CreateBlob("a"), caffe2::Tensor(at_tensor_a));
  // 将 PyTorch 张量 at_tensor_b 转换为 Caffe2 张量，并存储在工作空间中
  auto* c2_tensor_b = BlobSetTensor(workspace.CreateBlob("b"), caffe2::Tensor(at_tensor_b));

  // 在网络定义中添加 Add 操作，并指定输入输出
  {
    auto op = net.add_op();
    op->set_type("Add");
    op->add_input("a");
    op->add_input("b");


这样，每行代码都被详细地解释了其作用和功能。
    // 向操作 op 添加输出端 "c"
    op->add_output("c");
  }

  // 在工作空间中执行一次网络运行
  workspace.RunNetOnce(net);

  // 从工作空间创建名为 "c" 的 Blob，并获取可变的大小为 {5, 5} 的张量，存储在 CPU 上
  auto result = XBlobGetMutableTensor(workspace.CreateBlob("c"), {5, 5}, at::kCPU);
  // 获取结果张量的数据指针
  auto it = result.data<float>();
  // 对结果张量中的每个元素进行迭代
  for (const auto i : c10::irange(25)) {
    // 断言每个元素的值等于 2.0
    ASSERT_EQ(it[i], 2.0);
  }
  // 将可变张量 result 转换为 PyTorch 的 Tensor 对象
  at::Tensor at_result(result);
  // 断言张量所有元素之和等于 50
  ASSERT_EQ(at::sum(at_result).item<float>(), 50);
}

TEST(PytorchToCaffe2, SharedStorageWrite) {
  // 创建一个大小为5x5的全1的PyTorch张量，数据类型为float
  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat));
  // 重新视图化张量，将其大小变为25
  auto at_tensor_b = at_tensor_a.view({25});

  // 使用PyTorch张量创建一个Caffe2张量
  caffe2::Tensor c2_tensor_a(at_tensor_a);
  caffe2::Tensor c2_tensor_b(at_tensor_b);

  // 修改Caffe2张量的数据，改动在所有地方都可见
  c2_tensor_a.mutable_data<float>()[1] = 123;
  // 断言：验证修改在另一个张量上也可见
  ASSERT_EQ(c2_tensor_b.mutable_data<float>()[1], 123);
  // 断言：验证PyTorch张量上的修改
  ASSERT_EQ(at_tensor_a[0][1].item().to<float>(), 123);
  ASSERT_EQ(at_tensor_b[1].item().to<float>(), 123);
}

TEST(PytorchToCaffe2, MutualResizes) {
  // 创建一个大小为5x5的全1的PyTorch张量，数据类型为float
  auto at_tensor = at::ones({5, 5}, at::dtype(at::kFloat));

  // 使用PyTorch张量创建一个Caffe2张量
  caffe2::Tensor c2_tensor(at_tensor);

  // 修改Caffe2张量的数据，改动在PyTorch张量上也可见
  c2_tensor.mutable_data<float>()[0] = 123;
  ASSERT_EQ(at_tensor[0][0].item().to<float>(), 123);

  // 在较小的方向上调整PyTorch张量的大小 - 存储被保留
  at_tensor.resize_({4, 4});
  // 继续修改Caffe2张量的数据，验证在PyTorch张量上的影响
  c2_tensor.mutable_data<float>()[1] = 234;
  ASSERT_EQ(at_tensor[0][1].item().to<float>(), 234);

  // 在较大的方向上调整PyTorch张量的大小 - 存储被保留
  at_tensor.resize_({6, 6});
  // 继续修改Caffe2张量的数据，验证在PyTorch张量上的影响
  c2_tensor.mutable_data<float>()[2] = 345;
  ASSERT_EQ(at_tensor[0][2].item().to<float>(), 345);
  // 断言：验证Caffe2张量的大小与预期相符
  ASSERT_EQ(c2_tensor.sizes()[0], 6);
  ASSERT_EQ(c2_tensor.sizes()[1], 6);

  // 调整Caffe2张量的大小 - 语义上不保留数据，但TensorImpl仍然共享
  c2_tensor.Resize(7, 7);
  // 继续修改Caffe2张量的数据，验证在PyTorch张量上的影响
  c2_tensor.mutable_data<float>()[3] = 456;
  ASSERT_EQ(at_tensor[0][3].item().to<float>(), 456);
  // 断言：验证PyTorch张量的大小与预期相符
  ASSERT_EQ(at_tensor.sizes()[0], 7);
  ASSERT_EQ(at_tensor.sizes()[1], 7);
}

TEST(PytorchToCaffe2, Strided) {
  // 创建一个大小为5x5的全1的PyTorch张量，数据类型为float，并转置
  auto at_tensor = at::ones({5, 5}, at::dtype(at::kFloat)).t();
  // 断言：尝试直接使用不连续的PyTorch张量创建Caffe2张量会抛出异常
  ASSERT_ANY_THROW(caffe2::Tensor c2_tensor(at_tensor));
  // 但是，调用contiguous方法后可以正常创建
  caffe2::Tensor c2_tensor(at_tensor.contiguous());
  // 断言：验证Caffe2张量中的数据与预期一致
  for (const auto i : c10::irange(25)) {
    ASSERT_EQ(c2_tensor.data<float>()[i], 1.0);
  }
}

TEST(PytorchToCaffe2, InplaceStrided) {
  // 创建一个大小为2x5的全0的PyTorch张量，数据类型为float
  auto at_tensor = at::zeros({2, 5}, at::dtype(at::kFloat));
  // 使用PyTorch张量创建一个Caffe2张量
  caffe2::Tensor c2_tensor(at_tensor);
  // 断言：验证Caffe2张量的大小与预期相符
  ASSERT_EQ(c2_tensor.sizes()[0], 2);
  ASSERT_EQ(c2_tensor.sizes()[1], 5);

  // 修改Caffe2张量的数据，验证在PyTorch张量上的影响
  c2_tensor.mutable_data<float>()[1] = 234;
  ASSERT_EQ(at_tensor[0][1].item().to<float>(), 234);

  // 对PyTorch张量进行转置操作
  at_tensor.t_();
  // 断言：验证Caffe2张量的大小在PyTorch张量转置后发生变化
  ASSERT_EQ(c2_tensor.sizes()[0], 5);
  ASSERT_EQ(c2_tensor.sizes()[1], 2);
  // 这是一个错误的情况，然而，在每次数据访问时检查is_contiguous会很昂贵。
  // 我们依赖用户不要做疯狂的事情。
  ASSERT_EQ(at_tensor[1][0].item().to<float>(), 234);
  ASSERT_EQ(c2_tensor.data<float>()[1], 234);
}

TEST(PytorchToCaffe2, NonRegularTensor) {
  // 创建一个空的2x3的PyTorch张量，数据类型为float，布局为稀疏
  at::Tensor at_tensor =
      at::empty({2, 3}, at::dtype<float>().layout(at::kSparse));
  // 断言：验证PyTorch张量确实是稀疏张量
  ASSERT_TRUE(at_tensor.is_sparse());
  // 断言：尝试直接使用稀疏的PyTorch张量创建Caffe2张量会抛出异常
  ASSERT_ANY_THROW(caffe2::Tensor c2_tensor(at_tensor));
}
// 在测试框架中定义一个测试用例 Caffe2ToPytorch.NonPOD
TEST(Caffe2ToPytorch, NonPOD) {
  // 创建一个空的 Caffe2 张量，数据类型为 std::string
  caffe2::Tensor c2_tensor = caffe2::empty({1}, at::dtype<std::string>());
  // 获取可变的指向数据的指针
  auto data = c2_tensor.mutable_data<std::string>();
  // 将字符串 "test" 存入数据指针所指向的位置
  *data = "test";
  // 断言在下一行代码抛出异常，忽略特定的静态分析警告
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(at::Tensor at_tensor(c2_tensor));
}

// 在测试框架中定义一个测试用例 Caffe2ToPytorch.Nullptr
TEST(Caffe2ToPytorch, Nullptr) {
  // 创建一个未定义的 Caffe2 张量
  caffe2::Tensor c2_tensor;
  // 断言该张量未被定义
  ASSERT_FALSE(c2_tensor.defined());
  // 将未定义的 Caffe2 张量转换为 PyTorch 张量
  at::Tensor at_tensor(c2_tensor);
  // 断言生成的 PyTorch 张量也是未定义的
  ASSERT_FALSE(at_tensor.defined());
}

// 在测试框架中定义一个测试用例 PytorchToCaffe2.Nullptr
TEST(PytorchToCaffe2, Nullptr) {
  // 创建一个未定义的 PyTorch 张量
  at::Tensor at_tensor;
  // 断言该张量未被定义
  ASSERT_FALSE(at_tensor.defined());
  // 将未定义的 PyTorch 张量转换为 Caffe2 张量
  caffe2::Tensor c2_tensor(at_tensor);
  // 断言生成的 Caffe2 张量也是未定义的
  ASSERT_FALSE(c2_tensor.defined());
}
```