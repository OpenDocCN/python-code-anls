# `.\pytorch\aten\src\ATen\test\legacy_vmap_test.cpp`

```py
#include <gtest/gtest.h> // 引入 Google Test 框架的头文件

#include <ATen/ATen.h> // 引入 ATen 库的头文件
#include <ATen/LegacyBatchedTensorImpl.h> // 引入 ATen 中与批处理张量相关的旧版本实现的头文件
#include <ATen/LegacyVmapTransforms.h> // 引入 ATen 中与 Vmap 变换相关的旧版本实现的头文件
#include <c10/util/irange.h> // 引入 c10 库中的 irange 函数的头文件

using namespace at; // 使用 ATen 命名空间

namespace {

TEST(VmapTest, TestBatchedTensor) { // 定义测试用例 VmapTest.TestBatchedTensor
  {
    // NOLINTNEXTLINE(bugprone-argument-comment)
    Tensor x = addBatchDim(ones({2, 3, 4}), /*lvl=*/1, /*dim=*/1); // 添加批处理维度到张量 x
    std::vector<int64_t> expected_size = {2, 4}; // 预期张量 x 的尺寸
    ASSERT_EQ(x.sizes(), expected_size); // 断言张量 x 的尺寸符合预期
    ASSERT_EQ(x.dim(), 2); // 断言张量 x 的维度为 2
    ASSERT_EQ(x.numel(), 8); // 断言张量 x 的元素数量为 8
    ASSERT_EQ(x.is_contiguous(), false); // 断言张量 x 不是连续的
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(x.storage(), c10::Error); // 断言访问张量 x 的存储抛出异常
    ASSERT_EQ(x.storage_offset(), 0); // 断言张量 x 的存储偏移为 0
  }
  {
    // Test multiple batch dims
    // NOLINTNEXTLINE(bugprone-argument-comment)
    Tensor x = addBatchDim(ones({2, 3, 4}), /*lvl=*/1, /*dim=*/1); // 添加批处理维度到张量 x
    // NOLINTNEXTLINE(bugprone-argument-comment)
    x = addBatchDim(x, /*lvl=*/2, /*dim=*/1); // 继续添加批处理维度到张量 x
    std::vector<int64_t> expected_size = {2}; // 预期张量 x 的尺寸
    ASSERT_EQ(x.sizes(), expected_size); // 断言张量 x 的尺寸符合预期
    ASSERT_EQ(x.dim(), 1); // 断言张量 x 的维度为 1
    ASSERT_EQ(x.numel(), 2); // 断言张量 x 的元素数量为 2
  }
  {
    // Test vmap tensor dimensionality limit

    // Should not throw
    std::vector<int64_t> sizes(kVmapMaxTensorDims, 1); // 创建具有 kVmapMaxTensorDims 维度的大小向量
    // NOLINTNEXTLINE(bugprone-argument-comment)
    Tensor x = addBatchDim(ones(sizes), /*lvl=*/1, /*dim=*/1); // 添加批处理维度到张量 x

    // Should throw
    std::vector<int64_t> too_many_sizes(kVmapMaxTensorDims + 1, 1); // 创建超过限制维度的大小向量
    auto big_dim_tensor = ones(too_many_sizes); // 创建具有超过限制维度的张量
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto,bugprone-argument-comment)
    ASSERT_THROW(addBatchDim(big_dim_tensor, /*lvl=*/1, /*dim=*/1), c10::Error); // 断言添加批处理维度到超过限制的张量抛出异常
  }
  {
    // Create a "scalar" BatchedTensor. Should not crash.
    Tensor tensor = addBatchDim(ones({3}), /*lvl*/1, /*dim*/0); // 创建一个“标量”批处理张量
  }
}

// returns {{lvl=0,dim=0}, {lvl=1,dim=1}, ..., {lvl=kVmapNumLevels-1,dim=kVmapNumLevels-1}};
static BatchDims maxBatchDimsAtFront() { // 定义返回批处理维度的函数 maxBatchDimsAtFront
  BatchDims result; // 创建批处理维度的结果变量
  for (const auto lvl : c10::irange(kVmapNumLevels)) { // 遍历批处理级别范围
    result.emplace_back(lvl, /*dim=*/lvl); // 将每个级别和对应维度添加到结果中
  }
  return result; // 返回结果
}

TEST(VmapTest, TestBatchedTensorMaxLevel) { // 定义测试用例 VmapTest.TestBatchedTensorMaxLevel
  {
    // Should not throw
    auto tensor = ones({2, 3, 4}); // 创建一个张量
    makeBatched(ones({2, 3, 4}), {{/*lvl*/kVmapNumLevels - 1, /*dim*/0}}); // 创建一个具有最大级别的批处理张量
  }
  {
    auto tensor = ones({2, 3, 4}); // 创建一个张量
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(
        makeBatched(ones({2, 3, 4}), {{/*lvl*/kVmapNumLevels, /*dim*/0}}),
        c10::Error); // 断言创建超出最大级别的批处理张量抛出异常
  }
  {
    auto tensor = ones({2, 3, 4}); // 创建一个张量
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(
        makeBatched(ones({2, 3, 4}), {{/*lvl*/kVmapNumLevels + 5, /*dim*/0}}),
        c10::Error); // 断言创建超出最大级别的批处理张量抛出异常
  }
  {
    // create a BatchedTensor with kVmapNumLevels levels.
    // Should not throw
    auto tensor = ones(std::vector<int64_t>(kVmapNumLevels, 1)); // 创建一个具有最大级别的批处理张量
    makeBatched(tensor, maxBatchDimsAtFront()); // 使用最大级别的批处理维度创建批处理张量
  }
  {
    // create a BatchedTensor with kVmapNumLevels+1 levels.
    // 创建一个所有元素为1的张量，其形状为 kVmapNumLevels + 1 维度的向量
    auto tensor = ones(std::vector<int64_t>(kVmapNumLevels + 1, 1));
    // 调用 maxBatchDimsAtFront 函数获取批处理维度信息
    auto batch_dims = maxBatchDimsAtFront();
    // 向 batch_dims 向量中添加一个元素，元素为 std::pair<int, int>，表示级别为 kVmapNumLevels，维度为 kVmapNumLevels
    batch_dims.emplace_back(/*lvl*/kVmapNumLevels, /*dim*/kVmapNumLevels);
    // 使用 ASSERT_THROW 宏来确保 makeBatched 函数在执行时会抛出 c10::Error 异常
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(makeBatched(tensor, batch_dims), c10::Error);
  }
TEST(VmapTest, TestBatchedTensorActualDim) {
  {
    // No batch dims
    // 创建一个没有批次维度的张量
    Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {});
    // 获取可能存在批次维度的实现对象指针
    auto* batched = maybeGetBatchedImpl(tensor);
    // 断言实际维度与预期维度相等
    ASSERT_EQ(batched->actualDim(0), 0);
    ASSERT_EQ(batched->actualDim(1), 1);
    ASSERT_EQ(batched->actualDim(3), 3);

    // Test wrap around
    // 测试负索引包装
    ASSERT_EQ(batched->actualDim(-1), 3);
    ASSERT_EQ(batched->actualDim(-4), 0);
    // 使用 NOLINTNEXTLINE 标记，避免某些 Lint 工具报告
    ASSERT_THROW(batched->actualDim(-5), c10::Error);
    ASSERT_THROW(batched->actualDim(4), c10::Error);

    // test wrap_dim = False
    // 测试 wrap_dim = False 的情况
    ASSERT_THROW(batched->actualDim(-1, /*wrap_dim*/false), c10::Error);
    ASSERT_THROW(batched->actualDim(-4, /*wrap_dim*/false), c10::Error);
  }
  {
    // Single batch dim at front
    // 创建一个在前面的单个批次维度的张量
    Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {{/*lvl*/1, /*dim*/0}});
    auto* batched = maybeGetBatchedImpl(tensor);
    ASSERT_EQ(batched->actualDim(0), 1);
    ASSERT_EQ(batched->actualDim(2), 3);
    ASSERT_EQ(batched->actualDim(-1), 3);
    ASSERT_THROW(batched->actualDim(3), c10::Error);
  }
  {
    // Single batch dim in middle
    // 创建一个在中间的单个批次维度的张量
    Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {{/*lvl*/1, /*dim*/1}});
    auto* batched = maybeGetBatchedImpl(tensor);
    ASSERT_EQ(batched->actualDim(0), 0);
    ASSERT_EQ(batched->actualDim(1), 2);
    ASSERT_EQ(batched->actualDim(2), 3);
  }
  {
    // Single batch dim at end
    // 创建一个在末尾的单个批次维度的张量
    Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {{/*lvl*/1, /*dim*/1}});
    auto* batched = maybeGetBatchedImpl(tensor);
    ASSERT_EQ(batched->actualDim(0), 0);
    ASSERT_EQ(batched->actualDim(2), 3);
    ASSERT_EQ(batched->actualDim(-1), 3);
  }
  {
    // Multiple (2) batch dims at front
    // 创建一个在前面有多个（2个）批次维度的张量
    Tensor tensor = makeBatched(
        ones({2, 3, 5, 7}),
        {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/1}});
    auto* batched = maybeGetBatchedImpl(tensor);
    ASSERT_EQ(batched->actualDim(0), 2);
    ASSERT_EQ(batched->actualDim(1), 3);
  }
  {
    // Multiple (2) batch dims, misc places
    // 创建一个在不同位置有多个（2个）批次维度的张量
    Tensor tensor = makeBatched(
        ones({2, 3, 5, 7}),
        {{/*lvl*/1, /*dim*/1}, {/*lvl*/2, /*dim*/3}});
    auto* batched = maybeGetBatchedImpl(tensor);
    ASSERT_EQ(batched->actualDim(0), 0);
    ASSERT_EQ(batched->actualDim(1), 2);
    ASSERT_EQ(batched->actualDim(-1), 2);
    ASSERT_EQ(batched->actualDim(-2), 0);
  }
  {
    // ActualDim on kVmapMaxTensorDims sized underlying tensor
    // 在 kVmapMaxTensorDims 大小的底层张量上使用 ActualDim
    auto tensor = ones({});
    for (C10_UNUSED const auto i : c10::irange(kVmapMaxTensorDims)) {
      tensor = tensor.unsqueeze(0);
    }
    ASSERT_EQ(tensor.dim(), kVmapMaxTensorDims);

    // 添加批次维度
    auto batched = addBatchDim(tensor, /*lvl*/1, /*dim*/0);
    # 获取可能存在的批处理实现的指针，并分配给 auto 类型的指针变量 batched_impl
    auto* batched_impl = maybeGetBatchedImpl(batched);
    # 使用断言确保批处理实现的倒数第二维度与 kVmapMaxTensorDims - 1 相等
    ASSERT_EQ(
        batched_impl->actualDim(kVmapMaxTensorDims - 2),
        kVmapMaxTensorDims - 1);
    # 使用断言确保批处理实现的倒数第一维度与 kVmapMaxTensorDims - 1 相等
    ASSERT_EQ(
        batched_impl->actualDim(-1),
        kVmapMaxTensorDims - 1);
  }
}
// 定义一个名为 VmapTest 的测试案例，用于测试 MultiBatchVmapTransform 类的功能
TEST(VmapTest, TestMultiBatchVmapTransform) {
  {
    // 创建一个形状为 [2, 3, 5] 的常规张量 tensor
    auto tensor = ones({2, 3, 5});
    // 使用 MultiBatchVmapTransform 类的 logicalToPhysical 方法，预期抛出 c10::Error 异常
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(MultiBatchVmapTransform::logicalToPhysical(tensor), c10::Error);
  }
  {
    // 创建一个形状为 [2, 3, 5] 的张量 tensor，并将其包装成 BatchedTensor，批次维度已在最前面
    auto tensor = ones({2, 3, 5});
    // 定义批次维度为 {{1, 0}, {3, 1}}
    BatchDims bdims = {{/*lvl*/1, /*dim*/0}, {/*lvl*/3, /*dim*/1}};
    // 通过 makeBatched 函数将 tensor 和 bdims 结合成 BatchedTensor 对象 batched
    auto batched = makeBatched(tensor, bdims);

    // 调用 MultiBatchVmapTransform 类的 logicalToPhysical 方法，预期返回结果与原始 tensor 相同
    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    ASSERT_TRUE(result.tensor().is_same(tensor));
  }
  {
    // 创建一个形状为 [2, 3, 5] 的张量 tensor，并将其包装成 BatchedTensor，单个批次维度不在最前面
    auto tensor = ones({2, 3, 5});
    // 定义批次维度为 {{1, 1}}
    BatchDims bdims = {{/*lvl*/1, /*dim*/1}};
    // 通过 makeBatched 函数将 tensor 和 bdims 结合成 BatchedTensor 对象 batched
    auto batched = makeBatched(tensor, bdims);

    // 调用 MultiBatchVmapTransform 类的 logicalToPhysical 方法，预期结果张量与 tensor 的维度交换后相同
    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    ASSERT_EQ(result.tensor().data_ptr(), tensor.data_ptr());
    ASSERT_TRUE(at::allclose(result.tensor(), tensor.permute({1, 0, 2})));
  }
  {
    // 创建一个形状为 [2, 3, 5] 的张量 tensor，并将其包装成 BatchedTensor，多个批次维度不在最前面
    auto tensor = ones({2, 3, 5});
    // 定义批次维度为 {{1, 1}, {2, 2}, {3, 0}}
    BatchDims bdims = {{/*lvl*/1, /*dim*/1}, {/*lvl*/2, /*dim*/2}, {/*lvl*/3, /*dim*/0}};
    // 通过 makeBatched 函数将 tensor 和 bdims 结合成 BatchedTensor 对象 batched
    auto batched = makeBatched(tensor, bdims);

    // 调用 MultiBatchVmapTransform 类的 logicalToPhysical 方法，预期结果张量与 tensor 的维度交换后相同
    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    ASSERT_EQ(result.tensor().data_ptr(), tensor.data_ptr());
    ASSERT_TRUE(at::allclose(result.tensor(), tensor.permute({1, 2, 0})));
  }
  {
    // 边界情况：具有 kVmapNumLevels 个级别的张量，批次维度已在最前面

    // 创建一个大小为 kVmapNumLevels 的整数向量 sizes，初始全部为 1
    auto sizes = std::vector<int64_t>(kVmapNumLevels, 1);
    sizes[0] = 2;  // 第一个维度设为 2
    sizes[2] = 3;  // 第三个维度设为 3
    sizes[5] = 7;  // 第六个维度设为 7

    // 获取具有最大批次维度的前置 bdims
    auto batch_dims = maxBatchDimsAtFront();
    // 创建一个形状为 sizes 的张量 tensor
    auto tensor = ones(sizes);

    // 通过 makeBatched 函数将 tensor 和 batch_dims 结合成 BatchedTensor 对象 batched
    auto batched = makeBatched(tensor, batch_dims);
    // 调用 MultiBatchVmapTransform 类的 logicalToPhysical 方法，预期结果与原始 tensor 相同
    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    ASSERT_TRUE(result.tensor().is_same(tensor));
  }
  {
    // 边界情况：具有 kVmapNumLevels 个级别的张量，批次维度不在最前面

    // 创建一个大小为 kVmapNumLevels 的整数向量 sizes，初始全部为 1
    auto sizes = std::vector<int64_t>(kVmapNumLevels, 1);
    sizes[1] = 3;  // 第二个维度设为 3
    sizes[2] = 2;  // 第三个维度设为 2
    sizes[5] = 7;  // 第六个维度设为 7
    sizes[kVmapNumLevels - 1] = 5;  // 最后一个维度设为 5

    // 目标是重新排列 sizes，使最终大小为 [2, 3, 5, 7, 1, 1, 1, 1, ...]
    auto expected_result_sizes = std::vector<int64_t>(kVmapNumLevels, 1);
    expected_result_sizes[0] = 2;
    expected_result_sizes[1] = 3;
    expected_result_sizes[2] = 5;
    expected_result_sizes[3] = 7;

    // 定义批次维度为 {{0, 2}, {1, 1}, {2, 63}, {3, 5}, {4, 0}, {5, 3}, {6, 4}}
    BatchDims batch_dims = {
      {0, 2}, {1, 1}, {2, kVmapNumLevels - 1}, {3, 5}, {4, 0}, {5, 3}, {6, 4}
    };
    // 对于范围在 [7, kVmapNumLevels) 内的所有级别，将它们的维度加入 batch_dims
    for (const auto level : c10::irange(7, kVmapNumLevels)) {
      batch_dims.emplace_back(level, /*dim=*/level - 1);
    }
    // 创建一个形状为 sizes 的张量 tensor
    auto tensor = ones(sizes);
    # 使用 makeBatched 函数对给定的张量 tensor 进行分批处理，使用 batch_dims 参数指定批处理的维度
    auto batched = makeBatched(tensor, batch_dims);
    # 使用 MultiBatchVmapTransform 类的 logicalToPhysical 方法，将分批处理后的数据 batched 转换为物理存储表示
    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    # 使用断言 ASSERT_EQ 检查转换后的结果 result 的数据指针是否与原始张量 tensor 的数据指针相同
    ASSERT_EQ(result.tensor().data_ptr(), tensor.data_ptr());
    # 使用断言 ASSERT_EQ 检查转换后的结果 result 的张量尺寸是否与期望的结果尺寸 expected_result_sizes 相同
    ASSERT_EQ(result.tensor().sizes(), expected_result_sizes);
  }
TEST(VmapTest, TestVmapPhysicalViewGetPhysicalDim) {
  // 创建一个 VmapPhysicalView 对象，使用 ones 函数创建的张量作为物理视图数据，levels 参数为 1 和 4
  VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 1 | 4);

  // 测试正向维度索引
  ASSERT_EQ(physical_view.getPhysicalDim(0), 2);  // 检查索引为 0 的物理维度是否为 2
  ASSERT_EQ(physical_view.getPhysicalDim(1), 3);  // 检查索引为 1 的物理维度是否为 3
  ASSERT_EQ(physical_view.getPhysicalDim(2), 4);  // 检查索引为 2 的物理维度是否为 4
  // 使用 NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto) 禁止 LINT 检查
  ASSERT_THROW(physical_view.getPhysicalDim(3), c10::Error);  // 检查索引为 3 的物理维度是否会抛出错误

  // 测试负向维度索引（测试循环维度的行为）
  ASSERT_EQ(physical_view.getPhysicalDim(-1), 4);  // 检查负索引为 -1 的物理维度是否为 4
  ASSERT_EQ(physical_view.getPhysicalDim(-2), 3);  // 检查负索引为 -2 的物理维度是否为 3
  ASSERT_EQ(physical_view.getPhysicalDim(-3), 2);  // 检查负索引为 -3 的物理维度是否为 2
  // 使用 NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto) 禁止 LINT 检查
  ASSERT_THROW(physical_view.getPhysicalDim(-4), c10::Error);  // 检查负索引为 -4 的物理维度是否会抛出错误
}

TEST(VmapTest, TestVmapPhysicalViewGetPhysicalDims) {
  // 创建一个 VmapPhysicalView 对象，使用 ones 函数创建的张量作为物理视图数据，levels 参数为 2、8 和 16
  VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 2 | 8 | 16);

  // 测试正向和负向维度列表的映射
  ASSERT_EQ(
      physical_view.getPhysicalDims({0, 1, -1, -2}),
      VmapDimVector({3, 4, 4, 3}));  // 检查给定的维度列表映射到物理视图的维度向量是否符合预期

  // 使用 NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto) 禁止 LINT 检查
  ASSERT_THROW(physical_view.getPhysicalDims({2, 0}), c10::Error);  // 检查给定的维度列表是否会抛出错误
  // 使用 NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto) 禁止 LINT 检查
  ASSERT_THROW(physical_view.getPhysicalDims({0, -3}), c10::Error);  // 检查给定的维度列表是否会抛出错误
}

static void checkBatchDimsEqual(BatchDimsRef bdims, BatchDimsRef expected_bdims) {
  // 检查两个批次维度引用的大小是否相等，并逐个比较每个元素的维度和级别
  ASSERT_EQ(bdims.size(), expected_bdims.size());
  for (const auto idx : c10::irange(bdims.size())) {
    ASSERT_EQ(bdims[idx].dim(), expected_bdims[idx].dim());
    ASSERT_EQ(bdims[idx].level(), expected_bdims[idx].level());
  }
}

TEST(VmapTest, TestVmapPhysicalViewNewLogicalFromPhysical) {
  {
    // 简单情况：单一级别的物理视图
    VmapPhysicalView physical_view(ones({2, 3, 4}), /*levels = {2}*/4);
    Tensor physical = ones({2, 6, 7});

    auto result = physical_view.getPhysicalToLogicalMap().apply(physical);
    auto* batched = maybeGetBatchedImpl(result);
    ASSERT_TRUE(batched != nullptr);
    ASSERT_TRUE(batched->value().is_same(physical));
    // 检查批次维度是否与预期一致
    checkBatchDimsEqual(batched->bdims(), {{2, 0}});
  }
  {
    // 多个级别的物理视图
    VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), /*levels = {1, 3, 4}*/2 | 8 | 16);
    Tensor physical = ones({2, 3, 4, 7});

    auto result = physical_view.getPhysicalToLogicalMap().apply(physical);
    auto* batched = maybeGetBatchedImpl(result);
    ASSERT_TRUE(batched != nullptr);
    ASSERT_TRUE(batched->value().is_same(physical));
    // 检查批次维度是否与预期一致
    checkBatchDimsEqual(batched->bdims(), {{1, 0}, {3, 1}, {4, 2}});
  }
  {
    // 逻辑维度为空的情况
    VmapPhysicalView physical_view(ones({2}), /*levels = {2}*/4);
    Tensor physical = ones({2});

    auto result = physical_view.getPhysicalToLogicalMap().apply(physical);
    auto* batched = maybeGetBatchedImpl(result);
    ASSERT_TRUE(batched != nullptr);
    ASSERT_TRUE(batched->value().is_same(physical));
    // 检查批次维度是否与预期一致
    checkBatchDimsEqual(batched->bdims(), {{2, 0}});
  }
}
// 定义测试用例 VmapTest.TestBatchedTensorSum，验证批处理规则对张量求和的影响
TEST(VmapTest, TestBatchedTensorSum) {
  {
    // 创建形状为 {2, 3, 5, 7} 的随机张量 x
    Tensor x = at::randn({2, 3, 5, 7});

    // 将张量 x 进行批处理转换，批级别为 1，维度为 0
    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/0}});
    // 对批处理后的张量进行求和操作，沿第 0 维
    Tensor batched_out = batched_x.sum(0);
    // 获取批处理后的实现并取值
    const auto& out = maybeGetBatchedImpl(batched_out)->value();

    // 断言：批处理后的求和结果应与原始张量 x 沿第 1 维的求和结果相近
    ASSERT_TRUE(at::allclose(out, x.sum(1)));
  }
  {
    // 创建形状为 {2, 3} 的随机张量 x
    Tensor x = at::randn({2, 3});

    // 将张量 x 进行批处理转换，批级别为 1，维度为 1
    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/1}});
    // 对批处理后的张量进行求和操作，沿最后一维
    Tensor batched_out = batched_x.sum(-1);
    // 获取批处理后的实现并取值
    const auto& out = maybeGetBatchedImpl(batched_out)->value();

    // 断言：批处理后的求和结果应与原始张量 x 沿第 0 维的求和结果相近
    ASSERT_TRUE(at::allclose(out, x.sum(0)));
  }
  {
    // 创建形状为 {2, 3, 5, 7} 的随机张量 x
    Tensor x = at::randn({2, 3, 5, 7});

    // 将张量 x 进行批处理转换，批级别为 1，维度为 1
    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/1}});
    // 对批处理后的张量进行求和操作，沿维度列表 {0, 1}
    Tensor batched_out = batched_x.sum(std::vector<int64_t>{0, 1});
    // 获取批处理后的实现并取值
    const auto& out = maybeGetBatchedImpl(batched_out)->value();

    // 断言：批处理后的求和结果应与原始张量 x 沿维度列表 {0, 2} 的求和结果相近
    ASSERT_TRUE(at::allclose(out, x.sum(std::vector<int64_t>{0, 2})));
  }
  {
    // 创建形状为 {2, 3, 5, 7} 的随机张量 x
    Tensor x = at::randn({2, 3, 5, 7});

    // 将张量 x 进行批处理转换，批级别为 1，维度为 0 和 2
    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/1}});
    // 对批处理后的张量进行求和操作，沿维度列表 {0, 1}
    Tensor batched_out = batched_x.sum(std::vector<int64_t>{0, 1});
    // 获取批处理后的实现并取值
    const auto& out = maybeGetBatchedImpl(batched_out)->value();

    // 断言：批处理后的求和结果应与原始张量 x 沿维度列表 {2, 3} 的求和结果相近
    ASSERT_TRUE(at::allclose(out, x.sum(std::vector<int64_t>{2, 3})));
  }
}

// 辅助函数，检查广播 Vmap 转换是否正确
static void checkBroadcastingVmapTransform(TensorList inputs, TensorList expected_outputs) {
  // 使用 BroadcastingVmapTransform 类将逻辑输入转换为物理输出
  auto outputs = BroadcastingVmapTransform::logicalToPhysical(inputs);
  // 断言：输出张量列表的大小应与预期输出列表的大小相等
  ASSERT_EQ(outputs.size(), expected_outputs.size());
  // 遍历输出张量列表的索引
  for (const auto idx : c10::irange(outputs.size())) {
    // 获取输出张量的引用
    const auto& output = outputs[idx].tensor();
    // 断言：输出张量的数据指针应与预期输出张量的数据指针相等
    ASSERT_EQ(output.data_ptr(), expected_outputs[idx].data_ptr());
    // 断言：输出张量应与预期输出张量在数值上相近
    ASSERT_TRUE(at::allclose(output, expected_outputs[idx]));
  }
}

// 定义测试用例 VmapTest.TestBroadcastingVmapTransformBatchedBatched，验证批处理和广播转换的正确性
TEST(VmapTest, TestBroadcastingVmapTransformBatchedBatched) {
  {
    // 检查批处理维度是否移动到前面
    int64_t B0 = 5, B1 = 7;
    // 创建形状为 {2, B0, 3, B1} 的随机张量 x
    Tensor x = at::randn({2, B0, 3, B1});
    // 创建形状为 {B1, 2, 3, B0} 的随机张量 y
    Tensor y = at::randn({B1, 2, 3, B0});
    // 将张量 x 和 y 进行批处理转换
    Tensor batched_x = makeBatched(x, {{0, 1}, {1, 3}});
    Tensor batched_y = makeBatched(y, {{0, 3}, {1, 0}});

    // 检查广播 Vmap 转换是否正确
    checkBroadcastingVmapTransform(
        {batched_x, batched_y},
        // 期望的输出为 x 和 y 沿指定维度的置换结果
        {x.permute({1, 3, 0, 2}), y.permute({3, 0, 1, 2})});
  }
  {
    // 检查批处理维度是否对齐（即是否添加额外的 1 维）
    int64_t B0 = 5, B1 = 7, B2 = 9;
    // 创建形状为 {B0, B2, 2, 3} 的随机张量 x
    Tensor x = at::randn({B0, B2, 2, 3});
    // 创建形状为 {B0, B1, 2, 3} 的随机张量 y
    Tensor y = at::randn({B0, B1, 2, 3});
    // 将张量 x 和 y 进行批处理转换
    Tensor batched_x = makeBatched(x, {{0, 0}, {2, 1}});
    Tensor batched_y = makeBatched(y, {{0, 0}, {1, 1}});

    // 检查广播 Vmap 转换是否正确
    checkBroadcastingVmapTransform(
        {batched_x, batched_y},
        // 期望的输出为在 x 和 y 上添加维度 1 的结果
        {x.unsqueeze(1), y.unsqueeze(2)});
  }
}
    // 检查"example"是否用大小为1的额外维度填充。
    int64_t B0 = 5;
    // 创建形状为[B0, 3]的随机张量x
    Tensor x = at::randn({B0, 3});
    // 创建形状为[B0, 2, 3]的随机张量y
    Tensor y = at::randn({B0, 2, 3});
    // 对张量x进行批处理，使用makeBatched函数，填充方式为{{0, 0}}
    Tensor batched_x = makeBatched(x, {{0, 0}});
    // 对张量y进行批处理，使用makeBatched函数，填充方式为{{0, 0}}
    Tensor batched_y = makeBatched(y, {{0, 0}});

    // 调用检查广播vmap变换的函数，验证batched_x和batched_y与x.unsqueeze(1)和y之间的广播行为
    checkBroadcastingVmapTransform(
        {batched_x, batched_y},
        {x.unsqueeze(1), y});
  }
  {
    // 检查批次维度是否移动到前面，批次维度是否对齐，并确保示例正确填充。
    int64_t B0 = 5, B1 = 7, B2 = 11, B3 = 13;
    // 创建形状为[2, B0, 3, B2]的随机张量x
    Tensor x = at::randn({2, B0, 3, B2});
    // 创建形状为[B3, 3, B1]的随机张量y
    Tensor y = at::randn({B3, 3, B1});
    // 对张量x进行批处理，使用makeBatched函数，填充方式为{{0, 1}, {2, 3}}
    Tensor batched_x = makeBatched(x, {{0, 1}, {2, 3}});
    // 对张量y进行批处理，使用makeBatched函数，填充方式为{{1, 2}, {3, 0}}
    Tensor batched_y = makeBatched(y, {{1, 2}, {3, 0}});

    // 调用检查广播vmap变换的函数，验证batched_x和batched_y与通过排列和重塑得到的张量之间的广播行为
    checkBroadcastingVmapTransform(
        {batched_x, batched_y},
        {
          x.permute({1, 3, 0, 2}).view({B0, 1, B2, 1, 2, 3}),
          y.permute({2, 0, 1}).view({1, B1, 1, B3, 1, 3}),
        });
  }
  {
    // 边界情况：处理BatchedTensor "scalar"
    int64_t B0 = 5, B2 = 11;
    // 创建形状为[B0]的随机张量x
    Tensor x = at::randn({B0});
    // 创建形状为[B0, B2]的随机张量y
    Tensor y = at::randn({B0, B2});
    // 对张量x进行批处理，使用makeBatched函数，填充方式为{{0, 0}}
    Tensor batched_x = makeBatched(x, {{0, 0}});
    // 对张量y进行批处理，使用makeBatched函数，填充方式为{{0, 0}, {1, 1}}
    Tensor batched_y = makeBatched(y, {{0, 0}, {1, 1}});

    // 调用检查广播vmap变换的函数，验证batched_x和batched_y与x.view({B0, 1})和y之间的广播行为
    checkBroadcastingVmapTransform({batched_x, batched_y}, {x.view({B0, 1}), y});
    // 再次调用检查广播vmap变换的函数，验证batched_y和batched_x与y和x.view({B0, 1})之间的广播行为
    checkBroadcastingVmapTransform({batched_y, batched_x}, {y, x.view({B0, 1})});
  }
  {
    // 边界情况：只有一个张量是"batchedtensor scalar"
    int64_t B0 = 5, B2 = 11;
    // 创建形状为[B0]的随机张量x
    Tensor x = at::randn({B0});
    // 创建形状为[B0, B2, 2]的随机张量y
    Tensor y = at::randn({B0, B2, 2});
    // 对张量x进行批处理，使用makeBatched函数，填充方式为{{0, 0}}
    Tensor batched_x = makeBatched(x, {{0, 0}});
    // 对张量y进行批处理，使用makeBatched函数，填充方式为{{0, 0}, {1, 1}}
    Tensor batched_y = makeBatched(y, {{0, 0}, {1, 1}});

    // 调用检查广播vmap变换的函数，验证batched_x和batched_y与x.view({B0, 1, 1})和y之间的广播行为
    checkBroadcastingVmapTransform({batched_x, batched_y}, {x.view({B0, 1, 1}), y});
    // 再次调用检查广播vmap变换的函数，验证batched_y和batched_x与y和x.view({B0, 1, 1})之间的广播行为
    checkBroadcastingVmapTransform({batched_y, batched_x}, {y, x.view({B0, 1, 1})});
  }
}

TEST(VmapTest, TestBroadcastingVmapTransformBatchedUnbatched) {
  {
    // Check same example size
    // 定义两个维度 B0 和 B1
    int64_t B0 = 5, B1 = 7;
    // 创建一个形状为 {2, B0, 3, B1} 的随机张量 x
    Tensor x = at::randn({2, B0, 3, B1});
    // 创建一个形状为 {2, 3} 的随机张量 y
    Tensor y = at::randn({2, 3});
    // 将张量 x 变成批处理形式，batched_x 是一个经过处理的张量
    Tensor batched_x = makeBatched(x, {{0, 1}, {1, 3}});

    // 检查广播和 Vmap 转换
    checkBroadcastingVmapTransform(
        {batched_x, y},
        {x.permute({1, 3, 0, 2}), y.view({1, 1, 2, 3})});
    // 再次检查广播和 Vmap 转换，交换了输入顺序
    checkBroadcastingVmapTransform(
        {y, batched_x},
        {y.view({1, 1, 2, 3}), x.permute({1, 3, 0, 2})});
  }
  {
    // BatchedTensor has higher example dim than non-batched-tensor
    // 定义两个维度 B0 和 B1
    int64_t B0 = 5, B1 = 7;
    // 创建一个形状为 {B0, B1, 2, 3} 的随机张量 x
    Tensor x = at::randn({B0, B1, 2, 3});
    // 创建一个形状为 {3} 的随机张量 y
    Tensor y = at::randn({3});
    // 将张量 x 变成批处理形式，batched_x 是一个经过处理的张量
    Tensor batched_x = makeBatched(x, {{0, 0}, {1, 1}});

    // 检查广播和 Vmap 转换
    checkBroadcastingVmapTransform(
        {batched_x, y}, {x, y.view({1, 1, 1, 3})});
    // 再次检查广播和 Vmap 转换，交换了输入顺序
    checkBroadcastingVmapTransform(
        {y, batched_x}, {y.view({1, 1, 1, 3}), x});
  }
  {
    // BatchedTensor has lower example dim than non-batched-tensor
    // 定义两个维度 B0 和 B1
    int64_t B0 = 5, B1 = 7;
    // 创建一个形状为 {B0, B1, 3} 的随机张量 x
    Tensor x = at::randn({B0, B1, 3});
    // 创建一个形状为 {2, 3} 的随机张量 y
    Tensor y = at::randn({2, 3});
    // 将张量 x 变成批处理形式，batched_x 是一个经过处理的张量
    Tensor batched_x = makeBatched(x, {{0, 0}, {1, 1}});

    // 检查广播和 Vmap 转换
    checkBroadcastingVmapTransform(
        {batched_x, y}, {x.view({B0, B1, 1, 3}), y.view({1, 1, 2, 3})});
    // 再次检查广播和 Vmap 转换，交换了输入顺序
    checkBroadcastingVmapTransform(
        {y, batched_x}, {y.view({1, 1, 2, 3}), x.view({B0, B1, 1, 3})});
  }
  {
    // Scalar handling
    // 定义两个维度 B0 和 B1
    int64_t B0 = 5, B1 = 7;
    // 创建一个形状为 {B0, B1} 的随机张量 x
    Tensor x = at::randn({B0, B1});
    // 创建一个形状为 {} 的随机张量 y（标量）
    Tensor y = at::randn({});
    // 将张量 x 变成批处理形式，batched_x 是一个经过处理的张量
    Tensor batched_x = makeBatched(x, {{0, 0}, {1, 1}});

    // 检查广播和 Vmap 转换
    checkBroadcastingVmapTransform({batched_x, y}, {x, y.view({1, 1})});
    // 再次检查广播和 Vmap 转换，交换了输入顺序
    checkBroadcastingVmapTransform({y, batched_x}, {y.view({1, 1}), x});
  }
}

TEST(VmapTest, TestBroadcastingVmapTransformMaxLevels) {
  {
    // inputs have all 64 levels
    // 创建形状为 kVmapNumLevels 的张量 x 和 y，并进行随机初始化
    auto x = randn(std::vector<int64_t>(kVmapNumLevels, 1));
    auto y = randn(std::vector<int64_t>(kVmapNumLevels, 1));
    // 将张量 x 和 y 变成批处理形式，batched_x 和 batched_y 是经过处理的张量
    auto batched_x = makeBatched(x, maxBatchDimsAtFront());
    auto batched_y = makeBatched(y, maxBatchDimsAtFront());

    // 检查广播和 Vmap 转换
    checkBroadcastingVmapTransform({batched_x, batched_y}, {x, y});
  }
  {
    // inputs don't have all 64 levels, but results do.
    // 定义一个分割点 split
    int64_t split = 19;
    // 创建形状为 split 的张量 x 和 (kVmapNumLevels - split) 的张量 y，并进行随机初始化
    auto x = randn(std::vector<int64_t>(split, 1));
    auto y = randn(std::vector<int64_t>(kVmapNumLevels - split, 1));

    // 获取最大批处理维度的向量 tmp
    auto tmp = maxBatchDimsAtFront();
    // 构建 x_bdims
    BatchDims x_bdims(tmp.begin(), tmp.begin() + split);

    // 构建 y_bdims
    int64_t dim = 0;
    auto y_bdims_vector = fmap(
        ArrayRef<BatchDim>(tmp.begin() + split, tmp.end()),
        [&](const BatchDim& bdim) -> BatchDim {
          return { bdim.level(), dim++ };
        });
    BatchDims y_bdims(y_bdims_vector.begin(), y_bdims_vector.end());

    // 将张量 x 和 y 变成批处理形式，batched_x 和 batched_y 是经过处理的张量
    auto batched_x = makeBatched(x, x_bdims);
    auto batched_y = makeBatched(y, y_bdims);

    // 期望的结果形状是 kVmapNumLevels
    auto expected_size = std::vector<int64_t>(kVmapNumLevels, 1);
    checkBroadcastingVmapTransform(
        {batched_x, batched_y},
        {x.view(expected_size), y.view(expected_size)});
  }



# 调用名为 checkBroadcastingVmapTransform 的函数，检查 Vmap 转换是否可行
checkBroadcastingVmapTransform(
    # 第一个参数是一个包含 batched_x 和 batched_y 的元组，表示需要进行 Vmap 转换的批量数据
    {batched_x, batched_y},
    # 第二个参数是一个包含 x.view(expected_size) 和 y.view(expected_size) 的元组，表示视图操作后的数据
    {x.view(expected_size), y.view(expected_size)});
}
// Basic test for BatchedTensor::mul.
TEST(VmapTest, TestBatchedTensorMul) {
  {
    // batched * batched
    Tensor x = at::randn({2, 3});  // 创建一个大小为2x3的随机张量x
    Tensor y = at::randn({2, 3});  // 创建一个大小为2x3的随机张量y

    Tensor Bx = addBatchDim(x, /*lvl*/1, /*dim*/0);  // 在维度0上为张量x添加批次维度
    Tensor By = addBatchDim(y, /*lvl*/1, /*dim*/0);  // 在维度0上为张量y添加批次维度
    Tensor Bout = Bx * By;  // 对添加了批次维度的张量进行逐元素乘法运算

    const auto& out = maybeGetBatchedImpl(Bout)->value();  // 获取可能的批次实现，并获取其值
    std::vector<int64_t> expected_size = {2, 3};
    ASSERT_EQ(out.sizes(), expected_size);  // 断言输出张量的尺寸符合预期
    ASSERT_TRUE(at::allclose(out, x * y));  // 断言输出张量与未添加批次维度的张量乘积一致
  }
  {
    // batched * unbatched
    Tensor x = at::randn({2, 3});  // 创建一个大小为2x3的随机张量x
    Tensor y = at::randn({3});     // 创建一个大小为3的随机张量y

    Tensor Bx = addBatchDim(x, /*lvl*/1, /*dim*/0);  // 在维度0上为张量x添加批次维度
    Tensor Bout = Bx * y;  // 对添加了批次维度的张量x与未添加批次维度的张量y进行逐元素乘法运算
    const auto& out = maybeGetBatchedImpl(Bout)->value();  // 获取可能的批次实现，并获取其值
    std::vector<int64_t> expected_size = {2, 3};
    ASSERT_EQ(out.sizes(), expected_size);  // 断言输出张量的尺寸符合预期
    ASSERT_TRUE(at::allclose(out, x * y));  // 断言输出张量与未添加批次维度的张量乘积一致
  }
  {
    // batched (level 1) * batched (level 2)
    Tensor x = at::randn({2, 3});   // 创建一个大小为2x3的随机张量x
    Tensor y = at::randn({5, 3});   // 创建一个大小为5x3的随机张量y

    Tensor Bx = addBatchDim(x, /*lvl*/1, /*dim*/0);  // 在维度0上为张量x添加批次维度（级别1）
    Tensor By = addBatchDim(y, /*lvl*/2, /*dim*/0);  // 在维度0上为张量y添加批次维度（级别2）
    Tensor Bout = Bx * By;  // 对添加了批次维度的张量进行逐元素乘法运算

    // 我们得到一个双重包装的 BatchTensor...
    const auto& out = maybeGetBatchedImpl(Bout)->value();  // 获取可能的批次实现，并获取其值
    std::vector<int64_t> expected_size = {2, 5, 3};
    ASSERT_EQ(out.sizes(), expected_size);  // 断言输出张量的尺寸符合预期
    ASSERT_TRUE(at::allclose(out, x.unsqueeze(1) * y));  // 断言输出张量与未添加批次维度的张量乘积一致
  }
  {
    // batched (level 2, 3, 4) * batched (level 3, 1, 2)
    Tensor x = at::randn({3, 5, 7});   // 创建一个大小为3x5x7的随机张量x
    Tensor y = at::randn({5, 2, 3});   // 创建一个大小为5x2x3的随机张量y

    // 每个 BatchDim 都以 {dim, level} 格式构造。
    Tensor Bx = makeBatched(x, {{2, 0}, {3, 1}, {4, 2}});  // 根据指定的 {dim, level} 构造批次张量Bx
    Tensor By = makeBatched(y, {{1, 1}, {2, 2}, {3, 0}});  // 根据指定的 {dim, level} 构造批次张量By
    Tensor Bout = Bx * By;  // 对构造的批次张量进行逐元素乘法运算

    const auto& out = maybeGetBatchedImpl(Bout)->value();  // 获取可能的批次实现，并获取其值

    // 批次规则按照它们的级别顺序对维度进行对齐。
    // 这里所选的尺寸恰好按照级别的顺序排列。
    std::vector<int64_t> expected_size = {2, 3, 5, 7};
    ASSERT_EQ(out.sizes(), expected_size);  // 断言输出张量的尺寸符合预期
    ASSERT_TRUE(at::allclose(out, x * y.permute({1, 2, 0}).unsqueeze(3)));  // 断言输出张量与未添加批次维度的张量乘积一致
  }
}

// test for BatchedTensor::size(int).
TEST(VmapTest, TestBatchedTensorSize) {
  {
    // Single batch dim at front
    Tensor x = at::randn({3, 5, 7});  // 创建一个大小为3x5x7的随机张量x
    Tensor Bx = makeBatched(x, {{0, 0}});  // 在第0维度添加单一的批次维度

    ASSERT_EQ(Bx.size(0), 5);  // 断言批次张量在第0维度的大小为5
    ASSERT_EQ(Bx.size(1), 7);  // 断言批次张量在第1维度的大小为7
    ASSERT_EQ(Bx.size(-1), 7); // 使用负索引断言批次张量在最后一维度的大小为7
    ASSERT_EQ(Bx.size(-2), 5); // 使用负索引断言批次张量在倒数第二维度的大小为5
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(Bx.size(2), c10::Error);  // 断言访问第2维度将抛出异常
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(Bx.size(-3), c10::Error);  // 断言访问倒数第三维度将抛出异常
  }
  {
    // multiple batch dims not at front
    Tensor x = at::randn({2, 3, 5, 7, 11});  // 创建一个大小为2x3x5x7x11的随机张量x
    Tensor Bx = makeBatched(x, {{0, 3}, {1, 1}});  // 在维度3和1上添加多个批次维度

    ASSERT_EQ(Bx.size(0), 2);   // 断言批次张量在第0维度的大小为2
    ASSERT_EQ(Bx.size(1), 5);   // 断言批次张量在第1维度的大小为5
    ASSERT_EQ(Bx.size(2), 11);  // 断言批次张量在第2维度的大小为11
    ASSERT_EQ(Bx.size(-1), 11); // 使用负索引断言批次张量在最后一维度的大小为11
    ASSERT_EQ(Bx.size(-2), 5);  // 使用负索引断言批次张量在倒数第二维度的大小为5
    // 使用 ASSERT_THROW 宏检查 Bx 张量的维度大小，预期为 4 维，如果不是则抛出 c10::Error 异常
    ASSERT_THROW(Bx.size(3), c10::Error);
    
    // 使用 ASSERT_THROW 宏检查 Bx 张量的负索引维度大小，预期为倒数第4维，如果不是则抛出 c10::Error 异常
    ASSERT_THROW(Bx.size(-4), c10::Error);
}

// 测试 VmapPhysicalView 类的 getPhysicalShape 方法
TEST(VmapTest, TestVmapPhysicalViewGetPhysicalShape) {
  {
    // 创建 VmapPhysicalView 对象，使用 ones 函数生成的张量作为数据，参数为 1 | 4
    VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 1 | 4);
    // 断言获取的物理形状与预期相符，参数为空列表时返回 VmapDimVector({2, 3})
    ASSERT_EQ(physical_view.getPhysicalShape({}), VmapDimVector({2, 3}));
    // 断言获取的物理形状与预期相符，参数为 {7} 时返回 VmapDimVector({2, 3, 7})
    ASSERT_EQ(physical_view.getPhysicalShape({7}), VmapDimVector({2, 3, 7}));
    // 断言获取的物理形状与预期相符，参数为 {7, 11, 13} 时返回 VmapDimVector({2, 3, 7, 11, 13})
    ASSERT_EQ(physical_view.getPhysicalShape({7, 11, 13}), VmapDimVector({2, 3, 7, 11, 13}));
    // 断言获取的物理形状与预期相符，参数为 {7, 11, 13, 17} 时返回 VmapDimVector({2, 3, 7, 11, 13, 17})
    ASSERT_EQ(physical_view.getPhysicalShape({7, 11, 13, 17}), VmapDimVector({2, 3, 7, 11, 13, 17}));
  }
  {
    // 创建另一个 VmapPhysicalView 对象，使用 ones 函数生成的张量作为数据，参数为 2
    VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 2);
    // 断言获取的物理形状与预期相符，参数为空列表时返回 VmapDimVector({2})
    ASSERT_EQ(physical_view.getPhysicalShape({}), VmapDimVector({2}));
    // 断言获取的物理形状与预期相符，参数为 {7} 时返回 VmapDimVector({2, 7})
    ASSERT_EQ(physical_view.getPhysicalShape({7}), VmapDimVector({2, 7}));
  }
}

// 测试 BatchedTensor::expand 的基本功能
TEST(VmapTest, TestBatchedTensorExpand) {
  {
    // 扩展大小过小的情况
    auto tensor = at::randn({2, 3, 5});
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    // 断言在扩展大小不合适时抛出异常 c10::Error
    ASSERT_THROW(batched.expand({5}), c10::Error);
  }
  {
    // 扩展大小与逻辑维度相同的情况
    auto tensor = at::randn({2, 1, 5});
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
    // 对扩展后的 BatchedTensor 进行操作
    auto batched_out = batched.expand({3, 5});
    const auto& out = maybeGetBatchedImpl(batched_out)->value();

    // 断言内存地址相同，即操作结果与原始张量共享数据内存
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
    // 断言操作结果与原始张量在形状上相等
    ASSERT_TRUE(at::allclose(out, tensor.expand({2, 3, 5})));
  }
  {
    // 扩展大小与逻辑维度相同但尺寸不正确的情况
    auto tensor = at::randn({2, 1, 5});
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    // 断言在扩展大小不合适时抛出异常 c10::Error
    ASSERT_THROW(batched.expand({1, 25}), c10::Error);
  }
  {
    // 扩展大小比逻辑维度更大的情况
    auto tensor = at::randn({2, 3, 5});
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
    // 对扩展后的 BatchedTensor 进行操作
    auto batched_out = batched.expand({7, 3, 5});
    const auto& out = maybeGetBatchedImpl(batched_out)->value();

    // 断言内存地址相同，即操作结果与原始张量共享数据内存
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
    // 断言操作结果与原始张量按照新形状展开后在值上相等
    ASSERT_TRUE(at::allclose(out, tensor.view({2, 1, 3, 5}).expand({2, 7, 3, 5})));
  }
  {
    // 逻辑维度为 0，扩展大小与逻辑维度相同的情况
    auto tensor = at::randn({2, 3});
    auto batched = makeBatched(tensor, {{0, 0}, {1, 1}});
    // 对扩展后的 BatchedTensor 进行操作
    auto batched_out = batched.expand(c10::IntArrayRef({}));
    const auto& out = maybeGetBatchedImpl(batched_out)->value();

    // 断言内存地址相同，即操作结果与原始张量共享数据内存
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
    // 断言操作结果与原始张量在形状上相等
    ASSERT_TRUE(at::allclose(out, tensor));
  }
  {
    // 创建一个大小为 [2, 3] 的随机张量 tensor
    auto tensor = at::randn({2, 3});
    // 使用 makeBatched 函数将 tensor 包装成一个 batched 张量，扩展维度为 {{0, 0}, {1, 1}}
    auto batched = makeBatched(tensor, {{0, 0}, {1, 1}});
    // 将 batched 张量 batched_out 进行扩展，新的大小为 {5, 7}
    auto batched_out = batched.expand({5, 7});
    // 获取 batched_out 的 batched 实现，并将其值作为常量引用存储在 out 中
    const auto& out = maybeGetBatchedImpl(batched_out)->value();
    // 使用断言确保 out 的数据指针与 tensor 的数据指针相同
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
    // 使用断言检查 out 是否与 tensor.view({2, 3, 1, 1}).expand({2, 3, 5, 7}) 得到的张量在所有元素上接近
    ASSERT_TRUE(at::allclose(out, tensor.view({2, 3, 1, 1}).expand({2, 3, 5, 7})));
// Basic test for BatchedTensor::unsqueeze
TEST(VmapTest, TestBatchedTensorUnsqueeze) {
  {
    // Basic test
    auto tensor = at::randn({2, 3, 5});  // 创建一个形状为[2, 3, 5]的随机张量
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}}); // 使用 makeBatched 函数将张量批处理化，指定批处理级别和维度

    auto batched_out = batched.unsqueeze(0); // 对批处理后的张量在维度0上进行unsqueeze操作
    const auto& out = maybeGetBatchedImpl(batched_out)->value(); // 获取批处理实现后的输出张量的引用
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr()); // 断言输出张量的数据指针与原张量相同
    ASSERT_TRUE(at::allclose(out, tensor.unsqueeze(1))); // 断言输出张量与原张量在维度1上unsqueeze后的结果是否接近
  }
  {
    // Test with multiple levels
    auto tensor = at::randn({2, 3, 5});  // 创建一个形状为[2, 3, 5]的随机张量
    auto batched = makeBatched(tensor, {{0, 0}, {1, 1}}); // 使用 makeBatched 函数将张量批处理化，指定多个批处理级别和维度

    auto batched_out = batched.unsqueeze(0); // 对批处理后的张量在维度0上进行unsqueeze操作
    const auto& out = maybeGetBatchedImpl(batched_out)->value(); // 获取批处理实现后的输出张量的引用
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr()); // 断言输出张量的数据指针与原张量相同
    ASSERT_TRUE(at::allclose(out, tensor.unsqueeze(2))); // 断言输出张量与原张量在维度2上unsqueeze后的结果是否接近
  }
  {
    // Negative dim
    auto tensor = at::randn({2, 3, 5});  // 创建一个形状为[2, 3, 5]的随机张量
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}}); // 使用 makeBatched 函数将张量批处理化，指定批处理级别和维度

    auto batched_out = batched.unsqueeze(-1); // 对批处理后的张量在维度-1上进行unsqueeze操作（即最后一个维度）
    const auto& out = maybeGetBatchedImpl(batched_out)->value(); // 获取批处理实现后的输出张量的引用
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr()); // 断言输出张量的数据指针与原张量相同
    ASSERT_TRUE(at::allclose(out, tensor.unsqueeze(-1))); // 断言输出张量与原张量在最后一个维度unsqueeze后的结果是否接近
  }
}
// Basic test for BatchedTensor::squeeze(dim)
TEST(VmapTest, TestBatchedTensorSqueeze) {
  {
    // Basic test
    auto tensor = at::randn({2, 1, 5});  // 创建一个形状为[2, 1, 5]的随机张量
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}}); // 使用 makeBatched 函数将张量批处理化，指定批处理级别和维度

    auto batched_out = batched.squeeze(0); // 对批处理后的张量在维度0上进行squeeze操作
    const auto& out = maybeGetBatchedImpl(batched_out)->value(); // 获取批处理实现后的输出张量的引用
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr()); // 断言输出张量的数据指针与原张量相同
    ASSERT_TRUE(at::allclose(out, tensor.squeeze(1))); // 断言输出张量与原张量在维度1上squeeze后的结果是否接近
  }
  {
    // Test with multiple levels
    auto tensor = at::randn({2, 3, 1});  // 创建一个形状为[2, 3, 1]的随机张量
    auto batched = makeBatched(tensor, {{0, 0}, {1, 1}}); // 使用 makeBatched 函数将张量批处理化，指定多个批处理级别和维度

    auto batched_out = batched.squeeze(0); // 对批处理后的张量在维度0上进行squeeze操作
    const auto& out = maybeGetBatchedImpl(batched_out)->value(); // 获取批处理实现后的输出张量的引用
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr()); // 断言输出张量的数据指针与原张量相同
    ASSERT_TRUE(at::allclose(out, tensor.squeeze(2))); // 断言输出张量与原张量在维度2上squeeze后的结果是否接近
  }
  {
    // Negative dim
    auto tensor = at::randn({2, 3, 1});  // 创建一个形状为[2, 3, 1]的随机张量
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}}); // 使用 makeBatched 函数将张量批处理化，指定批处理级别和维度

    auto batched_out = batched.squeeze(-1); // 对批处理后的张量在维度-1上进行squeeze操作（即最后一个维度）
    const auto& out = maybeGetBatchedImpl(batched_out)->value(); // 获取批处理实现后的输出张量的引用
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr()); // 断言输出张量的数据指针与原张量相同
    ASSERT_TRUE(at::allclose(out, tensor.squeeze(-1))); // 断言输出张量与原张量在最后一个维度squeeze后的结果是否接近
  }
}
// Basic test for BatchedTensor::transpose
TEST(VmapTest, TestBatchedTensorTranspose) {
  {
    // Basic test
    auto tensor = at::randn({2, 3, 5});  // 创建一个形状为[2, 3, 5]的随机张量
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}}); // 使用 makeBatched 函数将张量批处理化，指定批处理级别和维度

    auto batched_out = batched.transpose(0, 1); // 对批处理后的张量进行转置，交换维度0和1
    const auto& out = maybeGetBatchedImpl(batched_out)->value(); // 获取批处理实现后的输出张量的引用
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr()); // 断言输出张量的数据指针与原张量相同
    ASSERT_TRUE(at::allclose(out, tensor.transpose(1, 2))); // 断言输出张量与原张量在维度1和2上转置后的结果是否接近
  }
  {
    // Test with multiple levels
    auto tensor = at::randn({2, 3, 5, 7, 11});  // 创建一个形状为[2, 3, 5, 7, 11]的随机张量
    auto batched = makeBatched(tensor, {{0, 0}, {1, 1}}); // 使用 makeBatched 函数将张量批处理化，指定多个批处理级别和维度

    // 对批处理后的张量进行转置，交换维度0和1
    auto batched_out = batched.transpose(0, 1);
    const auto& out = maybeGetBatchedImpl(batched_out)->value(); // 获取批处理实现后的输出张量的引用
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr()); // 断言输出张量的数据指针与原张量相同
    ASSERT_TRUE(at::allclose(out, tensor.transpose(1, 2))); // 断言输出张量与原张量在维度1和2上转置后的结果是否接近
  }
}
    {
        // 将 `batched` 张量进行维度转置，交换维度0和2
        auto batched_out = batched.transpose(0, 2);
        // 获取经过批处理处理后的实现，并获取其值
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        // 断言输出张量的数据指针与原始张量的数据指针相同
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        // 断言输出张量与原始张量在维度2和4上的转置是否全部接近
        ASSERT_TRUE(at::allclose(out, tensor.transpose(2, 4)));
      }
      {
        // 负维度
        // 创建一个形状为 {2, 3, 5, 7} 的随机张量
        auto tensor = at::randn({2, 3, 5, 7});  // NOLINT
        // 使用 `makeBatched` 函数将张量进行批处理，并指定级别为0，维度为0
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
    
        // 对批处理后的张量进行转置操作
        auto batched_out = batched.mT();
        // 获取经过批处理处理后的实现，并获取其值
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        // 断言输出张量的数据指针与原始张量的数据指针相同
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        // 断言输出张量与原始张量的转置是否全部接近
        ASSERT_TRUE(at::allclose(out, tensor.mT()));
      }
// Basic test for BatchedTensor::permute
TEST(VmapTest, TestBatchedTensorPermute) {
  {
    // Basic test
    // 创建一个形状为 {2, 3, 5} 的随机张量 tensor
    auto tensor = at::randn({2, 3, 5});  // NOLINT
    // 使用 makeBatched 函数创建一个批处理张量 batched，指定级别和维度为 {0, 0}
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

    // 对 batched 批处理张量进行维度置换操作，将维度 {1, 0} 转换为 {0, 1}
    auto batched_out = batched.permute({1, 0});
    // 获取 batched_out 的批处理实现，然后取其值
    const auto& out = maybeGetBatchedImpl(batched_out)->value();
    // 断言 out 的数据指针与 tensor 相同
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
    // 断言 out 与 tensor 经过 {0, 2, 1} 维度置换后的张量数据接近
    ASSERT_TRUE(at::allclose(out, tensor.permute({0, 2, 1})));
  }
  {
    // Test with multiple levels
    // 创建一个形状为 {2, 3, 5, 7, 11} 的随机张量 tensor
    auto tensor = at::randn({2, 3, 5, 7, 11});  // NOLINT
    // 使用 makeBatched 函数创建一个批处理张量 batched，指定级别和维度为 {{0, 0}, {1, 1}}
    auto batched = makeBatched(tensor, {{0, 0}, {1, 1}});

    // 对 batched 批处理张量进行维度置换操作，将维度 {2, 1, 0} 转换为 {0, 1, 4, 3, 2}
    auto batched_out = batched.permute({2, 1, 0});
    // 获取 batched_out 的批处理实现，然后取其值
    const auto& out = maybeGetBatchedImpl(batched_out)->value();
    // 断言 out 的数据指针与 tensor 相同
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
    // 断言 out 与 tensor 经过 {0, 1, 4, 3, 2} 维度置换后的张量数据接近
    ASSERT_TRUE(at::allclose(out, tensor.permute({0, 1, 4, 3, 2})));
  }
  {
    // Negative dims
    // 创建一个形状为 {2, 3, 5, 7} 的随机张量 tensor
    auto tensor = at::randn({2, 3, 5, 7});  // NOLINT
    // 使用 makeBatched 函数创建一个批处理张量 batched，指定级别和维度为 {0, 0}
    auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

    // 对 batched 批处理张量进行维度置换操作，将维度 {-1, -2, -3} 转换为 {0, -1, -2, -3}
    auto batched_out = batched.permute({-1, -2, -3});
    // 获取 batched_out 的批处理实现，然后取其值
    const auto& out = maybeGetBatchedImpl(batched_out)->value();
    // 断言 out 的数据指针与 tensor 相同
    ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
    // 断言 out 与 tensor 经过 {0, -1, -2, -3} 维度置换后的张量数据接近
    ASSERT_TRUE(at::allclose(out, tensor.permute({0, -1, -2, -3})));
  }
}

// 检查多批次 VmapTransform 的转换
static void checkMultiBatchVmapTransform(TensorList inputs, TensorList expected_outputs) {
  // 将输入 inputs 通过 MultiBatchVmapTransform::logicalToPhysical 转换为 outputs
  auto outputs = MultiBatchVmapTransform::logicalToPhysical(inputs);
  // 断言 outputs 和 expected_outputs 的大小相等
  ASSERT_EQ(outputs.size(), expected_outputs.size());
  // 遍历 outputs 中的每个索引
  for (const auto idx : c10::irange(outputs.size())) {
    // 获取 outputs 中索引为 idx 的张量输出
    const auto& output = outputs[idx].tensor();
    // 断言 output 的数据指针与 expected_outputs 中索引为 idx 的张量数据指针相同
    ASSERT_EQ(output.data_ptr(), expected_outputs[idx].data_ptr());
    // 断言 output 的大小与 expected_outputs 中索引为 idx 的张量大小相同
    ASSERT_EQ(output.sizes(), expected_outputs[idx].sizes());
    // 断言 output 与 expected_outputs 中索引为 idx 的张量数据接近
    ASSERT_TRUE(at::allclose(output, expected_outputs[idx]));
  }
}

// 测试 MultiBatchVmapTransform 的批处理批处理操作
TEST(VmapTest, TestMultiBatchVmapTransformBatchedBatched) {
  {
    // 检查批处理维度是否移动到前面
    int64_t B0 = 5, B1 = 7;
    // 创建形状为 {2, B0, 3, B1} 的随机张量 x
    Tensor x = at::randn({2, B0, 3, B1});
    // 创建形状为 {B1, 2, 3, B0} 的随机张量 y
    Tensor y = at::randn({B1, 2, 3, B0});
    // 使用 makeBatched 函数创建批处理张量 batched_x 和 batched_y，分别指定级别和维度
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/1}, {/*lvl*/1, /*dim*/3}});
    Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/3}, {/*lvl*/1, /*dim*/0}});

    // 调用 checkMultiBatchVmapTransform 函数检查批处理 VmapTransform
    checkMultiBatchVmapTransform(
        {batched_x, batched_y},
        {at::movedim(x, {1, 3}, {0, 1}), at::movedim(y, {0, 3}, {1, 0})});
  }
  {
    // 检查批处理维度是否变为广播，并且存在于所有返回值中
    int64_t B0 = 5, B1 = 7, B2 = 9;
    // 创建形状为 {B0, B2, 2, 3} 的随机张量 x
    Tensor x = at::randn({B0, B2, 2, 3});
    // 创建形状为 {B0, B1, 2, 3} 的随机张量 y
    Tensor y = at::randn({B0, B1, 2, 3});
    // 使用 makeBatched 函数创建批处理张量 batched_x 和 batched_y，分别指定级别和维度
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}, {/*lvl*/2, /*dim*/1}});
    Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

    // 调用 checkMultiBatchVmapTransform 函数检查批处理 VmapTransform
    checkMultiBatchVmapTransform(
        {batched_x, batched_y},
        {x.unsqueeze(1).expand({B0, B1, B2, 2, 3}), y.unsqueeze(2).expand({B0, B1, B2, 2, 3})});
  }
  {
    // 在不同逻辑维度的张量上进行操作的检查
    int64_t B0 = 5;
    // 创建形状为 {B0, 3} 的随机张量 x
    Tensor x = at::randn({B0, 3});
    // 创建形状为 {B0, 2, 3} 的随机张量 y
    Tensor y = at::randn({B0, 2, 3});
    // 使用 makeBatched 函数对张量 x 进行批处理操作，将其维度0级别批处理化
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}});
    // 使用 makeBatched 函数对张量 y 进行批处理操作，将其维度0级别批处理化
    Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/0}});

    // 检查多批次 Vmap 转换的一致性，传入批处理后的张量和原始张量 x、y
    checkMultiBatchVmapTransform({batched_x, batched_y}, {x, y});
  }
  {
    // 更复杂的示例，涉及两个张量
    int64_t B0 = 5, B1 = 7, B2 = 11, B3 = 13;
    // 创建形状为 {2, B0, 3, B2} 的随机张量 x
    Tensor x = at::randn({2, B0, 3, B2});
    // 创建形状为 {B3, 3, B1} 的随机张量 y
    Tensor y = at::randn({B3, 3, B1});
    // 使用 makeBatched 函数对张量 x 进行批处理操作，指定多个级别和维度的批处理化方式
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/1}, {/*lvl*/2, /*dim*/3}});
    // 使用 makeBatched 函数对张量 y 进行批处理操作，指定多个级别和维度的批处理化方式
    Tensor batched_y = makeBatched(y, {{/*lvl*/1, /*dim*/2}, {/*lvl*/3, /*dim*/0}});

    // 检查多批次 Vmap 转换的一致性，传入批处理后的张量和经过重新排列和扩展后的 x、y
    checkMultiBatchVmapTransform(
        {batched_x, batched_y},
        {
          x.permute({1, 3, 0, 2}).view({B0, 1, B2, 1, 2, 3}).expand({B0, B1, B2, B3, 2, 3}),
          y.permute({2, 0, 1}).view({1, B1, 1, B3, 3}).expand({B0, B1, B2, B3, 3}),
        });
  }
  {
    // 边界情况：处理批处理张量中的 "标量" 张量
    int64_t B0 = 5, B2 = 11;
    // 创建形状为 {B0} 的随机张量 x
    Tensor x = at::randn({B0});
    // 创建形状为 {B0, B2} 的随机张量 y
    Tensor y = at::randn({B0, B2});
    // 使用 makeBatched 函数对张量 x 进行批处理操作，将其维度0级别批处理化
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}});
    // 使用 makeBatched 函数对张量 y 进行批处理操作，指定多个级别和维度的批处理化方式
    Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

    // 检查多批次 Vmap 转换的一致性，传入批处理后的张量和经过变形和扩展后的 x、y
    checkMultiBatchVmapTransform({batched_x, batched_y}, {x.view({B0, 1}).expand({B0, B2}), y});
    // 检查多批次 Vmap 转换的一致性，传入批处理后的张量和经过变形和扩展后的 y、x
    checkMultiBatchVmapTransform({batched_y, batched_x}, {y, x.view({B0, 1}).expand({B0, B2})});
  }
  {
    // 边界情况：只有一个张量是 "批处理张量标量"
    int64_t B0 = 5, B2 = 11;
    // 创建形状为 {B0} 的随机张量 x
    Tensor x = at::randn({B0});
    // 创建形状为 {B0, B2, 2} 的随机张量 y
    Tensor y = at::randn({B0, B2, 2});
    // 使用 makeBatched 函数对张量 x 进行批处理操作，将其维度0级别批处理化
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}});
    // 使用 makeBatched 函数对张量 y 进行批处理操作，指定多个级别和维度的批处理化方式
    Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

    // 检查多批次 Vmap 转换的一致性，传入批处理后的张量和经过变形和扩展后的 x、y
    checkMultiBatchVmapTransform({batched_x, batched_y}, {x.view({B0, 1}).expand({B0, B2}), y});
    // 检查多批次 Vmap 转换的一致性，传入批处理后的张量和经过变形和扩展后的 y、x
    checkMultiBatchVmapTransform({batched_y, batched_x}, {y, x.view({B0, 1}).expand({B0, B2})});
  }
}

TEST(VmapTest, TestMultiBatchVmapTransformBatchedUnbatched) {
  {
    // 检查相同的示例大小
    int64_t B0 = 5, B1 = 7;
    // 创建大小为 {2, B0, 3, B1} 的随机张量 x
    Tensor x = at::randn({2, B0, 3, B1});
    // 创建大小为 {2, 3} 的随机张量 y
    Tensor y = at::randn({2, 3});
    // 使用 makeBatched 函数将张量 x 转换为批处理形式的 batched_x
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/1}, {/*lvl*/1, /*dim*/3}});

    // 检查多批处理 Vmap 转换，验证两个张量的转换结果是否符合预期
    checkMultiBatchVmapTransform(
        {batched_x, y},
        {at::movedim(x, {1, 3}, {0, 1}), y.view({1, 1, 2, 3}).expand({B0, B1, 2, 3})});
    checkMultiBatchVmapTransform(
        {y, batched_x},
        {y.view({1, 1, 2, 3}).expand({B0, B1, 2, 3}), at::movedim(x, {1, 3}, {0, 1})});
  }
  {
    // BatchedTensor 的示例维度高于非批处理张量
    int64_t B0 = 5, B1 = 7;
    // 创建大小为 {B0, B1, 2, 3} 的随机张量 x
    Tensor x = at::randn({B0, B1, 2, 3});
    // 创建大小为 {3} 的随机张量 y
    Tensor y = at::randn({3});
    // 使用 makeBatched 函数将张量 x 转换为批处理形式的 batched_x
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

    // 检查多批处理 Vmap 转换，验证两个张量的转换结果是否符合预期
    checkMultiBatchVmapTransform(
        {batched_x, y}, {x, y.view({1, 1, 3}).expand({B0, B1, 3})});
    checkMultiBatchVmapTransform(
        {y, batched_x}, {y.view({1, 1, 3}).expand({B0, B1, 3}), x});
  }
  {
    // BatchedTensor 的示例维度低于非批处理张量
    int64_t B0 = 5, B1 = 7;
    // 创建大小为 {B0, B1, 3} 的随机张量 x
    Tensor x = at::randn({B0, B1, 3});
    // 创建大小为 {2, 3} 的随机张量 y
    Tensor y = at::randn({2, 3});
    // 使用 makeBatched 函数将张量 x 转换为批处理形式的 batched_x
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

    // 检查多批处理 Vmap 转换，验证两个张量的转换结果是否符合预期
    checkMultiBatchVmapTransform(
        {batched_x, y}, {x.view({B0, B1, 3}), y.view({1, 1, 2, 3}).expand({B0, B1, 2, 3})});
    checkMultiBatchVmapTransform(
        {y, batched_x}, {y.view({1, 1, 2, 3}).expand({B0, B1, 2, 3}), x.view({B0, B1, 3})});
  }
  {
    // 标量处理
    int64_t B0 = 5, B1 = 7;
    // 创建大小为 {B0, B1} 的随机张量 x
    Tensor x = at::randn({B0, B1});
    // 创建大小为 {} 的随机张量 y（标量）
    Tensor y = at::randn({});
    // 使用 makeBatched 函数将张量 x 转换为批处理形式的 batched_x
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

    // 检查多批处理 Vmap 转换，验证两个张量的转换结果是否符合预期
    checkMultiBatchVmapTransform({batched_x, y}, {x, y.view({1, 1}).expand({B0, B1})});
    checkMultiBatchVmapTransform({y, batched_x}, {y.view({1, 1}).expand({B0, B1}), x});
  }
}

TEST(VmapTest, TestMultiBatchVmapTransformMaxLevels) {
  {
    // 输入具有所有 64 级别
    auto x = randn(std::vector<int64_t>(kVmapNumLevels, 1));
    auto y = randn(std::vector<int64_t>(kVmapNumLevels, 1));
    auto batched_x = makeBatched(x, maxBatchDimsAtFront());
    auto batched_y = makeBatched(y, maxBatchDimsAtFront());

    // 检查多批处理 Vmap 转换，验证两个张量的转换结果是否符合预期
    checkMultiBatchVmapTransform({batched_x, batched_y}, {x, y});
  }
  {
    // 输入不具有所有 64 级别，但结果具有。
    int64_t split = 19;
    auto x = randn(std::vector<int64_t>(split, 1));
    auto y = randn(std::vector<int64_t>(kVmapNumLevels - split, 1));

    auto tmp = maxBatchDimsAtFront();
    BatchDims x_bdims(tmp.begin(), tmp.begin() + split);

    // 构建 y_bdims。
    int64_t dim = 0;
    auto y_bdims_vector = fmap(
        ArrayRef<BatchDim>(tmp.begin() + split, tmp.end()),
        [&](const BatchDim& bdim) -> BatchDim {
          return { bdim.level(), dim++ };
        });
    BatchDims y_bdims(y_bdims_vector.begin(), y_bdims_vector.end());
    # 使用 makeBatched 函数将 x 和 x_bdims 批处理，得到批处理后的张量 batched_x
    auto batched_x = makeBatched(x, x_bdims);
    # 使用 makeBatched 函数将 y 和 y_bdims 批处理，得到批处理后的张量 batched_y
    auto batched_y = makeBatched(y, y_bdims);

    # 创建一个期望大小为 kVmapNumLevels 的整数向量，每个元素初始化为 1
    auto expected_size = std::vector<int64_t>(kVmapNumLevels, 1);
    # 调用 checkMultiBatchVmapTransform 函数，检查多批次 Vmap 变换
    # 参数是两个元组，每个元组包含批处理后的张量和期望的视图大小张量
    checkMultiBatchVmapTransform(
        {batched_x, batched_y},    # 第一个元组包含 batched_x 和 batched_y
        {x.view(expected_size), y.view(expected_size)});  # 第二个元组包含 x 和 y 的期望视图大小
  }
// 定义测试用例 VmapTest 的 TestMultiBatchVmapTransformMultipleTensors 测试函数
TEST(VmapTest, TestMultiBatchVmapTransformMultipleTensors) {
  // 在此测试用例中进行三个张量的测试（全部批处理）
  {
    // 定义三个批次大小
    int64_t B0 = 5, B1 = 7, B2 = 9;
    // 创建形状为 {2, B0, 3, B1} 的随机张量 x
    Tensor x = at::randn({2, B0, 3, B1});
    // 创建形状为 {B1, 4} 的随机张量 y
    Tensor y = at::randn({B1, 4});
    // 创建形状为 {2, B2} 的随机张量 z
    Tensor z = at::randn({2, B2});
    // 使用 makeBatched 函数将张量 x 批处理
    Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/1}, {/*lvl*/1, /*dim*/3}});
    // 使用 makeBatched 函数将张量 y 批处理
    Tensor batched_y = makeBatched(y, {{/*lvl*/1, /*dim*/0}});
    // 使用 makeBatched 函数将张量 z 批处理
    Tensor batched_z = makeBatched(z, {{/*lvl*/2, /*dim*/1}});

    // 调用 checkMultiBatchVmapTransform 函数进行多批次 vmap 转换的检查
    checkMultiBatchVmapTransform(
        {batched_x, batched_y, batched_z},  // 输入批处理后的张量列表
        {
          // 期望输出的张量列表
          at::movedim(x, {1, 3}, {0, 1}).view({B0, B1, 1, 2, 3}).expand({B0, B1, B2, 2, 3}),
          y.view({1, B1, 1, 4}).expand({B0, B1, B2, 4}),
          z.t().view({1, 1, B2, 2}).expand({B0, B1, B2, 2}),
        });
  }

  // 在此测试用例中进行三个张量的测试，包含批处理和非批处理
  {
    // 定义三个批次大小
    int64_t B0 = 5, B1 = 7, B2 = 9;
    // 创建形状为 {2, 3} 的随机张量 x
    Tensor x = at::randn({2, 3});
    // 创建形状为 {4, B0} 的随机张量 y
    Tensor y = at::randn({4, B0});
    // 创建形状为 {B1, 2, B2} 的随机张量 z
    Tensor z = at::randn({B1, 2, B2});
    // 使用 makeBatched 函数将张量 y 批处理
    Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/1}});
    // 使用 makeBatched 函数将张量 z 批处理
    Tensor batched_z = makeBatched(z, {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/2}});

    // 调用 checkMultiBatchVmapTransform 函数进行多批次 vmap 转换的检查
    checkMultiBatchVmapTransform(
        {x, batched_y, batched_z},  // 输入批处理后的张量列表
        {
          // 期望输出的张量列表
          x.view({1, 1, 1, 2, 3}).expand({B0, B1, B2, 2, 3}),
          y.t().view({B0, 1, 1, 4}).expand({B0, B1, B2, 4}),
          z.permute({0, 2, 1}).view({1, B1, B2, 2}).expand({B0, B1, B2, 2}),
        });
  }
}
```