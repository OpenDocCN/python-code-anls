# `.\pytorch\c10\test\core\impl\SizesAndStrides_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <c10/core/impl/SizesAndStrides.h>  // 引入 SizesAndStrides 类的头文件
#include <c10/util/irange.h>  // 引入 c10 命名空间下的 irange 函数

#ifdef __clang__
#pragma clang diagnostic ignored "-Wself-assign-overloaded"  // 忽略 clang 编译器的自我赋值警告
#endif

using namespace c10;  // 使用 c10 命名空间
using namespace c10::impl;  // 使用 c10::impl 命名空间

// NOLINTBEGIN(*conversion*, *multiplication*)  // 开始 NOLINT 静态检查标记，禁止指定类型的警告

static void checkData(
    const SizesAndStrides& sz,
    IntArrayRef sizes,
    IntArrayRef strides) {
  EXPECT_EQ(sizes.size(), strides.size())  // 检查 sizes 和 strides 的大小是否相等
      << "bad test case: size() of sizes and strides don't match";  // 如果不相等，输出错误信息

  EXPECT_EQ(sz.size(), sizes.size());  // 检查 SizesAndStrides 对象的尺寸和 sizes 的大小是否相等

  int idx = 0;
  for (auto x : sizes) {
    EXPECT_EQ(sz.size_at_unchecked(idx), x) << "index: " << idx;  // 检查未经检查的索引处的尺寸
    EXPECT_EQ(sz.size_at(idx), x) << "index: " << idx;  // 检查指定索引处的尺寸
    EXPECT_EQ(sz.sizes_data()[idx], x) << "index: " << idx;  // 检查尺寸数据数组中指定索引处的值
    EXPECT_EQ(*(sz.sizes_begin() + idx), x) << "index: " << idx;  // 检查从指定起始位置开始的尺寸数组的值
    idx++;
  }

  EXPECT_EQ(sz.sizes_arrayref(), sizes);  // 检查 SizesAndStrides 对象的尺寸数组是否与 sizes 相等

  idx = 0;
  for (auto x : strides) {
    EXPECT_EQ(sz.stride_at_unchecked(idx), x) << "index: " << idx;  // 检查未经检查的索引处的跨度
    EXPECT_EQ(sz.stride_at(idx), x) << "index: " << idx;  // 检查指定索引处的跨度
    EXPECT_EQ(sz.strides_data()[idx], x) << "index: " << idx;  // 检查跨度数据数组中指定索引处的值
    EXPECT_EQ(*(sz.strides_begin() + idx), x) << "index: " << idx;  // 检查从指定起始位置开始的跨度数组的值
    idx++;
  }

  EXPECT_EQ(sz.strides_arrayref(), strides);  // 检查 SizesAndStrides 对象的跨度数组是否与 strides 相等
}

TEST(SizesAndStridesTest, DefaultConstructor) {
  SizesAndStrides sz;  // 创建 SizesAndStrides 对象，默认构造函数
  checkData(sz, {0}, {1});  // 检查默认构造的 SizesAndStrides 对象的数据
  // Can't test size_at() out of bounds because it just asserts for now.
}

TEST(SizesAndStridesTest, SetSizes) {
  SizesAndStrides sz;  // 创建 SizesAndStrides 对象
  sz.set_sizes({5, 6, 7, 8});  // 设置 SizesAndStrides 对象的尺寸
  checkData(sz, {5, 6, 7, 8}, {1, 0, 0, 0});  // 检查设置尺寸后的 SizesAndStrides 对象的数据
}

TEST(SizesAndStridesTest, Resize) {
  SizesAndStrides sz;  // 创建 SizesAndStrides 对象

  sz.resize(2);  // 调整 SizesAndStrides 对象的尺寸

  // Small to small growing.
  checkData(sz, {0, 0}, {1, 0});  // 检查调整后的 SizesAndStrides 对象的数据

  // Small to small growing, again.
  sz.resize(5);  // 再次调整 SizesAndStrides 对象的尺寸
  checkData(sz, {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0});  // 检查再次调整后的 SizesAndStrides 对象的数据

  for (const auto ii : c10::irange(sz.size())) {
    sz.size_at_unchecked(ii) = ii + 1;  // 设置未检查索引处的尺寸
  // 设置数组 `sz` 的步长为 `2 * (ii + 1)`
  sz.stride_at_unchecked(ii) = 2 * (ii + 1);
}

// 验证函数 `checkData`，用于检查 `sz` 的大小和步长是否符合预期
checkData(sz, {1, 2, 3, 4, 5}, {2, 4, 6, 8, 10});

// 将大小调整为4，验证 `sz` 的大小和步长
sz.resize(4);
checkData(sz, {1, 2, 3, 4}, {2, 4, 6, 8});

// 再次将大小调整为4，验证 `sz` 的大小和步长，确保没有改变
sz.resize(4);
checkData(sz, {1, 2, 3, 4}, {2, 4, 6, 8});

// 将大小调整为5，验证 `sz` 的大小和步长，并确保新数据为零
sz.resize(5);
checkData(sz, {1, 2, 3, 4, 0}, {2, 4, 6, 8, 0});

// 将大小调整为6，验证 `sz` 的大小和步长
sz.resize(6);
checkData(sz, {1, 2, 3, 4, 0, 0}, {2, 4, 6, 8, 0, 0});

// 设置 `sz` 在索引5的大小为6，步长为12，并验证 `sz` 的大小和步长
sz.size_at_unchecked(5) = 6;
sz.stride_at_unchecked(5) = 12;
checkData(sz, {1, 2, 3, 4, 0, 6}, {2, 4, 6, 8, 0, 12});

// 将大小调整为7，验证 `sz` 的大小和步长
sz.resize(7);
checkData(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

// 再次将大小调整为7，验证 `sz` 的大小和步长，确保没有改变
sz.resize(7);
checkData(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

// 设置 `sz` 在索引6的大小为11，步长为22，并验证 `sz` 的大小和步长
sz.size_at_unchecked(6) = 11;
sz.stride_at_unchecked(6) = 22;
checkData(sz, {1, 2, 3, 4, 0, 6, 11}, {2, 4, 6, 8, 0, 12, 22});

// 将大小调整为6，验证 `sz` 的大小和步长
sz.resize(6);
checkData(sz, {1, 2, 3, 4, 0, 6}, {2, 4, 6, 8, 0, 12});

// 将大小调整为7，验证 `sz` 的大小和步长，并确保新数据为零
sz.resize(7);
checkData(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

// 最后，将大小从大调整为小，设置 `sz` 的大小和步长，并验证
for (const auto ii : c10::irange(sz.size())) {
  // 设置 `sz` 在索引 `ii` 处的大小为 `ii - 1`
  sz.size_at_unchecked(ii) = ii - 1;
  // 设置 `sz` 在索引 `ii` 处的步长为 `2 * (ii - 1)`
  sz.stride_at_unchecked(ii) = 2 * (ii - 1);
}

// 验证 `sz` 的大小和步长，确保符合预期
checkData(sz, {-1, 0, 1, 2, 3, 4, 5}, {-2, 0, 2, 4, 6, 8, 10});

// 将大小调整为5，验证 `sz` 的大小和步长
sz.resize(5);
checkData(sz, {-1, 0, 1, 2, 3}, {-2, 0, 2, 4, 6});
TEST(SizesAndStridesTest, SetAtIndex) {
  // 创建 SizesAndStrides 对象实例
  SizesAndStrides sz;

  // 设置实例大小为 5
  sz.resize(5);
  // 设置索引为 4 处的大小为 42
  sz.size_at(4) = 42;
  // 设置索引为 4 处的步幅为 23
  sz.stride_at(4) = 23;

  // 检查数据是否符合预期
  checkData(sz, {0, 0, 0, 0, 42}, {1, 0, 0, 0, 23});

  // 调整实例大小为 6
  sz.resize(6);
  // 设置索引为 5 处的大小为 43
  sz.size_at(5) = 43;
  // 设置索引为 5 处的步幅为 24
  sz.stride_at(5) = 24;

  // 再次检查数据是否符合预期
  checkData(sz, {0, 0, 0, 0, 42, 43}, {1, 0, 0, 0, 23, 24});
}

TEST(SizesAndStridesTest, SetAtIterator) {
  // 创建 SizesAndStrides 对象实例
  SizesAndStrides sz;

  // 设置实例大小为 5
  sz.resize(5);
  // 通过迭代器设置索引为 4 处的大小为 42
  *(sz.sizes_begin() + 4) = 42;
  // 通过迭代器设置索引为 4 处的步幅为 23
  *(sz.strides_begin() + 4) = 23;

  // 检查数据是否符合预期
  checkData(sz, {0, 0, 0, 0, 42}, {1, 0, 0, 0, 23});

  // 调整实例大小为 6
  sz.resize(6);
  // 通过迭代器设置索引为 5 处的大小为 43
  *(sz.sizes_begin() + 5) = 43;
  // 通过迭代器设置索引为 5 处的步幅为 24
  *(sz.strides_begin() + 5) = 24;

  // 再次检查数据是否符合预期
  checkData(sz, {0, 0, 0, 0, 42, 43}, {1, 0, 0, 0, 23, 24});
}

TEST(SizesAndStridesTest, SetViaData) {
  // 创建 SizesAndStrides 对象实例
  SizesAndStrides sz;

  // 设置实例大小为 5
  sz.resize(5);
  // 通过数据指针设置索引为 4 处的大小为 42
  *(sz.sizes_data() + 4) = 42;
  // 通过数据指针设置索引为 4 处的步幅为 23
  *(sz.strides_data() + 4) = 23;

  // 检查数据是否符合预期
  checkData(sz, {0, 0, 0, 0, 42}, {1, 0, 0, 0, 23});

  // 调整实例大小为 6
  sz.resize(6);
  // 通过数据指针设置索引为 5 处的大小为 43
  *(sz.sizes_data() + 5) = 43;
  // 通过数据指针设置索引为 5 处的步幅为 24
  *(sz.strides_data() + 5) = 24;

  // 再次检查数据是否符合预期
  checkData(sz, {0, 0, 0, 0, 42, 43}, {1, 0, 0, 0, 23, 24});
}

static SizesAndStrides makeSmall(int offset = 0) {
  // 创建 SizesAndStrides 对象实例
  SizesAndStrides small;
  // 设置实例大小为 3
  small.resize(3);
  // 使用无需检查的方法设置大小和步幅
  for (const auto ii : c10::irange(small.size())) {
    small.size_at_unchecked(ii) = ii + 1 + offset;
    small.stride_at_unchecked(ii) = 2 * (ii + 1 + offset);
  }

  // 返回设置好的实例
  return small;
}

static SizesAndStrides makeBig(int offset = 0) {
  // 创建 SizesAndStrides 对象实例
  SizesAndStrides big;
  // 设置实例大小为 8
  big.resize(8);
  // 使用无需检查的方法设置大小和步幅
  for (const auto ii : c10::irange(big.size())) {
    big.size_at_unchecked(ii) = ii - 1 + offset;
    big.stride_at_unchecked(ii) = 2 * (ii - 1 + offset);
  }

  // 返回设置好的实例
  return big;
}

static void checkSmall(const SizesAndStrides& sm, int offset = 0) {
  // 创建大小和步幅的期望向量
  std::vector<int64_t> sizes(3), strides(3);
  // 使用循环设置期望值
  for (const auto ii : c10::irange(3)) {
    sizes[ii] = ii + 1 + offset;
    strides[ii] = 2 * (ii + 1 + offset);
  }
  // 检查实际数据与期望值是否一致
  checkData(sm, sizes, strides);
}

static void checkBig(const SizesAndStrides& big, int offset = 0) {
  // 创建大小和步幅的期望向量
  std::vector<int64_t> sizes(8), strides(8);
  // 使用循环设置期望值
  for (const auto ii : c10::irange(8)) {
    sizes[ii] = ii - 1 + offset;
    strides[ii] = 2 * (ii - 1 + offset);
  }
  // 检查实际数据与期望值是否一致
  checkData(big, sizes, strides);
}

TEST(SizesAndStridesTest, MoveConstructor) {
  // 创建空的 SizesAndStrides 对象实例
  SizesAndStrides empty;

  // 使用移动构造函数创建移动后的空实例
  SizesAndStrides movedEmpty(std::move(empty));

  // 使用断言检查空实例大小是否为 0
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(empty.size(), 0);
  // 使用断言检查移动后的空实例大小是否为 1
  EXPECT_EQ(movedEmpty.size(), 1);
  // 检查移动后的空实例数据是否符合预期
  checkData(movedEmpty, {0}, {1});

  // 创建小实例并检查数据
  SizesAndStrides small = makeSmall();
  checkSmall(small);

  // 使用移动构造函数创建移动后的小实例
  SizesAndStrides movedSmall(std::move(small));
  // 检查移动后的小实例数据是否符合预期
  checkSmall(movedSmall);
  // 使用断言检查小实例是否已被清空
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(small.size(), 0);

  // 创建大实例并检查数据
  SizesAndStrides big = makeBig();
  checkBig(big);

  // 使用移动构造函数创建移动后的大实例
  SizesAndStrides movedBig(std::move(big));
  // 检查移动后的大实例数据是否符合预期
  checkBig(movedBig);
  // 使用断言检查大实例是否已被清空
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(big.size(), 0);
}
TEST(SizesAndStridesTest, CopyConstructor) {
  SizesAndStrides empty;

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  // 创建一个空的 SizesAndStrides 对象 copiedEmpty，通过拷贝构造函数复制 empty
  SizesAndStrides copiedEmpty(empty);

  // 验证 empty 和 copiedEmpty 的 size() 方法返回值为 1
  EXPECT_EQ(empty.size(), 1);
  EXPECT_EQ(copiedEmpty.size(), 1);
  // 验证 empty 和 copiedEmpty 的数据内容为 {0} 和 {1}
  checkData(empty, {0}, {1});
  checkData(copiedEmpty, {0}, {1});

  // 创建一个 small SizesAndStrides 对象，并验证其内容
  SizesAndStrides small = makeSmall();
  checkSmall(small);

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  // 创建一个拷贝 small 对象的 SizesAndStrides 对象 copiedSmall
  SizesAndStrides copiedSmall(small);
  // 验证 copiedSmall 与 small 的内容相同
  checkSmall(copiedSmall);
  checkSmall(small);

  // 创建一个 big SizesAndStrides 对象，并验证其内容
  SizesAndStrides big = makeBig();
  checkBig(big);

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  // 创建一个拷贝 big 对象的 SizesAndStrides 对象 copiedBig
  SizesAndStrides copiedBig(big);
  // 验证 copiedBig 与 big 的内容相同
  checkBig(big);
  checkBig(copiedBig);
}

TEST(SizesAndStridesTest, CopyAssignmentSmallToSmall) {
  // 创建一个 small SizesAndStrides 对象 smallTarget
  SizesAndStrides smallTarget = makeSmall();
  // 创建一个带有额外参数的 small SizesAndStrides 对象 smallCopyFrom
  SizesAndStrides smallCopyFrom = makeSmall(1);

  // 验证 smallTarget 和 smallCopyFrom 的内容
  checkSmall(smallTarget);
  checkSmall(smallCopyFrom, 1);

  // 将 smallCopyFrom 拷贝给 smallTarget
  smallTarget = smallCopyFrom;

  // 验证拷贝后 smallTarget 的内容
  checkSmall(smallTarget, 1);
  checkSmall(smallCopyFrom, 1);
}

TEST(SizesAndStridesTest, MoveAssignmentSmallToSmall) {
  // 创建一个 small SizesAndStrides 对象 smallTarget
  SizesAndStrides smallTarget = makeSmall();
  // 创建一个带有额外参数的 small SizesAndStrides 对象 smallMoveFrom
  SizesAndStrides smallMoveFrom = makeSmall(1);

  // 验证 smallTarget 和 smallMoveFrom 的内容
  checkSmall(smallTarget);
  checkSmall(smallMoveFrom, 1);

  // 使用移动语义将 smallMoveFrom 移动给 smallTarget
  smallTarget = std::move(smallMoveFrom);

  // 验证移动后 smallTarget 的内容，以及 smallMoveFrom 是否有效
  checkSmall(smallTarget, 1);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(smallMoveFrom.size(), 0);
}

TEST(SizesAndStridesTest, CopyAssignmentSmallToBig) {
  // 创建一个 big SizesAndStrides 对象 bigTarget
  SizesAndStrides bigTarget = makeBig();
  // 创建一个 small SizesAndStrides 对象 smallCopyFrom
  SizesAndStrides smallCopyFrom = makeSmall();

  // 验证 bigTarget 和 smallCopyFrom 的内容
  checkBig(bigTarget);
  checkSmall(smallCopyFrom);

  // 将 smallCopyFrom 拷贝给 bigTarget
  bigTarget = smallCopyFrom;

  // 验证拷贝后 bigTarget 和 smallCopyFrom 的内容
  checkSmall(bigTarget);
  checkSmall(smallCopyFrom);
}

TEST(SizesAndStridesTest, MoveAssignmentSmallToBig) {
  // 创建一个 big SizesAndStrides 对象 bigTarget
  SizesAndStrides bigTarget = makeBig();
  // 创建一个 small SizesAndStrides 对象 smallMoveFrom
  SizesAndStrides smallMoveFrom = makeSmall();

  // 验证 bigTarget 和 smallMoveFrom 的内容
  checkBig(bigTarget);
  checkSmall(smallMoveFrom);

  // 使用移动语义将 smallMoveFrom 移动给 bigTarget
  bigTarget = std::move(smallMoveFrom);

  // 验证移动后 bigTarget 的内容，以及 smallMoveFrom 是否有效
  checkSmall(bigTarget);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(smallMoveFrom.size(), 0);
}

TEST(SizesAndStridesTest, CopyAssignmentBigToBig) {
  // 创建一个 big SizesAndStrides 对象 bigTarget
  SizesAndStrides bigTarget = makeBig();
  // 创建一个带有额外参数的 big SizesAndStrides 对象 bigCopyFrom
  SizesAndStrides bigCopyFrom = makeBig(1);

  // 验证 bigTarget 和 bigCopyFrom 的内容
  checkBig(bigTarget);
  checkBig(bigCopyFrom, 1);

  // 将 bigCopyFrom 拷贝给 bigTarget
  bigTarget = bigCopyFrom;

  // 验证拷贝后 bigTarget 和 bigCopyFrom 的内容
  checkBig(bigTarget, 1);
  checkBig(bigCopyFrom, 1);
}

TEST(SizesAndStridesTest, MoveAssignmentBigToBig) {
  // 创建一个 big SizesAndStrides 对象 bigTarget
  SizesAndStrides bigTarget = makeBig();
  // 创建一个带有额外参数的 big SizesAndStrides 对象 bigMoveFrom
  SizesAndStrides bigMoveFrom = makeBig(1);

  // 验证 bigTarget 和 bigMoveFrom 的内容
  checkBig(bigTarget);
  checkBig(bigMoveFrom, 1);

  // 使用移动语义将 bigMoveFrom 移动给 bigTarget
  bigTarget = std::move(bigMoveFrom);

  // 验证移动后 bigTarget 的内容，以及 bigMoveFrom 是否有效
  checkBig(bigTarget, 1);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(bigMoveFrom.size(), 0);
}
// 在 SizesAndStridesTest 测试集中，测试复制赋值操作：从大对象复制到小对象
TEST(SizesAndStridesTest, CopyAssignmentBigToSmall) {
  // 创建一个小对象作为目标
  SizesAndStrides smallTarget = makeSmall();
  // 创建一个大对象作为源
  SizesAndStrides bigCopyFrom = makeBig();

  // 检查小对象的状态
  checkSmall(smallTarget);
  // 检查大对象的状态
  checkBig(bigCopyFrom);

  // 执行复制赋值操作：将大对象的内容复制给小对象
  smallTarget = bigCopyFrom;

  // 检查复制后的小对象的状态，应与大对象相同
  checkBig(smallTarget);
  // 大对象的状态应该不受影响
  checkBig(bigCopyFrom);
}

// 在 SizesAndStridesTest 测试集中，测试移动赋值操作：从大对象移动到小对象
TEST(SizesAndStridesTest, MoveAssignmentBigToSmall) {
  // 创建一个小对象作为目标
  SizesAndStrides smallTarget = makeSmall();
  // 创建一个大对象作为移动源
  SizesAndStrides bigMoveFrom = makeBig();

  // 检查小对象的状态
  checkSmall(smallTarget);
  // 检查大对象的状态
  checkBig(bigMoveFrom);

  // 执行移动赋值操作：使用 std::move 将大对象的内容移动给小对象
  smallTarget = std::move(bigMoveFrom);

  // 检查移动后小对象的状态，应与大对象相同
  checkBig(smallTarget);
  // 使用 NOLINTNEXTLINE 标记，验证移动后大对象的大小为零
  EXPECT_EQ(bigMoveFrom.size(), 0);
}

// 在 SizesAndStridesTest 测试集中，测试自我复制赋值操作
TEST(SizesAndStridesTest, CopyAssignmentSelf) {
  // 创建一个小对象和一个大对象
  SizesAndStrides small = makeSmall();
  SizesAndStrides big = makeBig();

  // 检查小对象的状态
  checkSmall(small);
  // 检查大对象的状态
  checkBig(big);

  // 使用 NOLINTNEXTLINE 标记，执行自我复制赋值操作，但实际上不会改变对象的状态
  small = small;
  // 检查自我复制后小对象的状态，应与复制前相同
  checkSmall(small);

  // 使用 NOLINTNEXTLINE 标记，执行自我复制赋值操作，但实际上不会改变对象的状态
  big = big;
  // 检查自我复制后大对象的状态，应与复制前相同
  checkBig(big);
}

// 静态函数，用于避免由于 -Wall -Wself-move 引起的问题
static void selfMove(SizesAndStrides& x, SizesAndStrides& y) {
  // 执行移动赋值操作
  x = std::move(y);
}

// 在 SizesAndStridesTest 测试集中，测试自我移动赋值操作
TEST(SizesAndStridesTest, MoveAssignmentSelf) {
  // 创建一个小对象和一个大对象
  SizesAndStrides small = makeSmall();
  SizesAndStrides big = makeBig();

  // 检查小对象的状态
  checkSmall(small);
  // 检查大对象的状态
  checkBig(big);

  // 使用自定义函数 selfMove 执行自我移动赋值操作，但实际上不会改变对象的状态
  selfMove(small, small);
  // 检查自我移动后小对象的状态，应与移动前相同
  checkSmall(small);

  // 使用自定义函数 selfMove 执行自我移动赋值操作，但实际上不会改变对象的状态
  selfMove(big, big);
  // 检查自我移动后大对象的状态，应与移动前相同
  checkBig(big);
}
// 使用 NOLINTEND 标记结束禁用 lint 检查，排除 (*conversion*, *multiplication*) 类型的警告
```