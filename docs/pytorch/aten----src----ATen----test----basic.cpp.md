# `.\pytorch\aten\src\ATen\test\basic.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/ATen.h>  // 引入 ATen 库的头文件
#include <ATen/core/Reduction.h>  // 引入 ATen 库的 Redution 相关头文件
#include <torch/cuda.h>  // 引入 PyTorch CUDA 相关头文件
#include <ATen/test/test_assert.h>  // 引入 ATen 测试断言相关头文件
#include <c10/util/irange.h>  // 引入 c10 库的 irange 头文件
#include <c10/util/CallOnce.h>  // 引入 c10 库的 CallOnce 头文件

// 仅用于 TH 兼容性测试...
struct THFloatTensor;  // 声明 THFloatTensor 结构体，用于 Torch 兼容性测试

#include <iostream>  // 引入输入输出流库的头文件
#include <chrono>  // 引入时间库的头文件
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <string.h>  // 引入字符串处理库的头文件（NOLINTNEXTLINE 指示忽略特定的 Lint 警告）
#include <sstream>  // 引入字符串流库的头文件
#include <thread>  // 引入线程库的头文件
#include <mutex>  // 引入互斥量库的头文件

#define ASSERT_EQ_RESOLVED(X, Y) \  // 定义断言宏，判断 X 是否等于 Y
  {                              \
    bool isEQ = X == Y;          \
    ASSERT_TRUE(isEQ);           \
  }

using namespace at;  // 使用 at 命名空间

void TestResize(DeprecatedTypeProperties& type) {  // 定义测试函数 TestResize，参数为 DeprecatedTypeProperties 类型的引用
  auto a = at::empty({0}, type.options());  // 创建一个空张量 a
  a.resize_({3, 4});  // 调整张量 a 的大小为 3x4
  ASSERT_EQ_RESOLVED(a.numel(), 12);  // 断言：a 的元素数量应为 12
  a.resize_({5, 7});  // 再次调整张量 a 的大小为 5x7
  ASSERT_EQ_RESOLVED(a.numel(), 35);  // 断言：a 的元素数量应为 35
}

void TestOnesAndDot(DeprecatedTypeProperties& type) {  // 定义测试函数 TestOnesAndDot，参数为 DeprecatedTypeProperties 类型的引用
  Tensor b0 = ones({1, 1}, type);  // 创建一个元素全为 1 的张量 b0
  ASSERT_EQ_RESOLVED((b0 + b0).sum().item<double>(), 2);  // 断言：两个 b0 相加后的和应为 2

  Tensor b1 = ones({1, 2}, type);  // 创建一个元素全为 1 的张量 b1
  ASSERT_EQ_RESOLVED((b1 + b1).sum().item<double>(), 4);  // 断言：两个 b1 相加后的和应为 4

  Tensor b = ones({3, 4}, type);  // 创建一个元素全为 1 的张量 b，大小为 3x4
  ASSERT_EQ_RESOLVED((b + b).sum().item<double>(), 24);  // 断言：两个 b 相加后的和应为 24
  ASSERT_EQ_RESOLVED(b.numel(), 12);  // 断言：b 的元素数量应为 12
  if (type.backend() != Backend::CPU || type.scalarType() != kHalf) {  // 如果张量类型不是半精度 CPU 张量
    ASSERT_EQ_RESOLVED(b.view(-1).dot(b.view(-1)).item<double>(), 12);  // 断言：b 重塑后的向量点积应为 12
  }
}

void TestSort(DeprecatedTypeProperties& type) {  // 定义测试函数 TestSort，参数为 DeprecatedTypeProperties 类型的引用
  Tensor b = rand({3, 4}, type);  // 创建一个随机张量 b，大小为 3x4

  auto z = b.sort(1);  // 对张量 b 按行排序
  auto z_sorted = std::get<0>(z);  // 获取排序后的张量

  bool isLT = z_sorted[0][0].item<float>() < z_sorted[0][1].item<float>();  // 检查排序后的第一行第一个元素是否小于第一个元素
  ASSERT_TRUE(isLT);  // 断言：应为真
}

void TestRandperm(DeprecatedTypeProperties& type) {  // 定义测试函数 TestRandperm，参数为 DeprecatedTypeProperties 类型的引用
  if (type.backend() != Backend::CUDA) {  // 如果不是 CUDA 后端
    Tensor b = randperm(15, type);  // 创建一个大小为 15 的随机排列张量 b
    auto [rv, ri] = sort(b, 0);  // 对张量 b 进行排序，并获取结果张量 rv 和索引张量 ri
    bool isLE = (rv[0].item<float>() <= rv[1].item<float>());  // 检查排序后的第一个和第二个元素是否满足小于等于关系
    ASSERT_TRUE(isLE);  // 断言：应为真
  }
}

void SendContext() {  // 定义函数 SendContext，用于发送上下文信息
  std::stringstream ss;  // 创建一个字符串流 ss
  ss << "context: " << std::hex << (int64_t)&globalContext() << std::endl;  // 将全局上下文的地址以十六进制输出到字符串流 ss
}

void TestAdd(DeprecatedTypeProperties& type) {  // 定义测试函数 TestAdd，参数为 DeprecatedTypeProperties 类型的引用
  Tensor a = rand({3, 4}, type);  // 创建一个大小为 3x4 的随机张量 a
  Tensor b = rand({3, 4}, type);  // 创建一个大小为 3x4 的随机张量 b
  Tensor c = add(a, add(a, b));  // 计算张量 a、a 和 b 的和，并将结果赋给张量 c
  // TODO:0-dim Tensor d(3.f);
  Scalar d = 3.f;  // 创建一个标量 d，值为 3.0
  if (type.backend() == Backend::CPU && type.scalarType() == kHalf) {  // 如果是半精度 CPU 张量
      ASSERT_TRUE(add(c, d).allclose(a + a + b + d, 1e-2));  // 断言：张量 c 加上 d 应与 a、a、b 加上 d 的结果在一定误差范围内相等
  } else {  // 其他情况
      ASSERT_TRUE(add(c, d).allclose(a + a + b + d));  // 断言：张量 c 加上 d 应与 a、a、b 加上 d 的结果相等
  }
}

void TestZeros(DeprecatedTypeProperties& type) {  // 定义测试函数 TestZeros，参数为 DeprecatedTypeProperties 类型的引用
  auto begin = std::chrono::high_resolution_clock::now();  // 记录开始时间点
  Tensor a = zeros({1024, 1024}, type);  // 创建一个大小为 1024x1024 的零张量 a
  for (C10_UNUSED const auto i : c10::irange(1, 1000)) {  // 迭代 999 次（未使用循环变量）
    a = zeros({128, 128}, type);  // 重新创建一个大小为 128x128 的零张量 a
  }
  auto end = std::chrono::high_resolution_clock::now();  // 记录结束时间点
  std::cout << std::dec << "   "  // 输出时间间隔（毫秒）
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - begin)
                   .count()
            << " ms" << std::endl;

   std::srand(std::time(nullptr));  // 根据当前时间设置随机数种子
   ASSERT_EQ(norm(a).item<double>(), 0.0);  // 断言：张量 a 的范数应为 0
}
// 测试函数：对给定类型的张量执行大量加法操作
void TestLoadsOfAdds(DeprecatedTypeProperties& type) {
  // 记录开始时间
  auto begin = std::chrono::high_resolution_clock::now();
  // 创建大小为 (3, 4) 的全一张量 d
  Tensor d = ones({3, 4}, type);
  // 创建大小为 (3, 4) 的全零张量 r
  Tensor r = zeros({3, 4}, type);
  // 执行 1000 次加法操作，将结果保存到 r 中
  for (C10_UNUSED const auto i : c10::irange(1000)) {
    add_out(r, r, d);
  }
  // 记录结束时间
  auto end = std::chrono::high_resolution_clock::now();
  // 输出执行时间
  std::cout << std::dec << "   "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - begin)
                   .count()
            << " ms" << std::endl;
  // 检验预期结果
  ASSERT_EQ_RESOLVED(norm(1000 * d).item<double>(), norm(r).item<double>());
}

// 测试函数：对给定类型的张量执行大量加法操作（使用复制方式）
void TestLoadOfAddsWithCopy(DeprecatedTypeProperties& type) {
  // 记录开始时间
  auto begin = std::chrono::high_resolution_clock::now();
  // 创建大小为 (3, 4) 的全一张量 d
  Tensor d = ones({3, 4}, type);
  // 创建大小为 (3, 4) 的全零张量 r
  Tensor r = zeros({3, 4}, type);
  // 执行 1000 次加法操作，将结果复制到 r 中
  for (C10_UNUSED const auto i : c10::irange(1000)) {
    r = add(r, d);
  }
  // 记录结束时间
  auto end = std::chrono::high_resolution_clock::now();
  // 输出执行时间
  std::cout << std::dec << "   "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - begin)
                   .count()
            << " ms" << std::endl;
  // 检验预期结果
  ASSERT_EQ_RESOLVED(norm(1000 * d).item<double>(), norm(r).item<double>());
}

// 测试函数：检查张量是否连续
void TestIsContiguous(DeprecatedTypeProperties& type) {
  // 创建大小为 (3, 4) 的随机张量 a
  Tensor a = rand({3, 4}, type);
  // 断言张量 a 是否连续
  ASSERT_TRUE(a.is_contiguous());
  // 对张量 a 执行维度转置
  a = a.transpose(0, 1);
  // 断言张量 a 是否不连续
  ASSERT_FALSE(a.is_contiguous());
}

// 测试函数：测试张量维度置换
void TestPermute(DeprecatedTypeProperties& type) {
  // 创建大小为 (3, 4, 5) 的随机张量 a
  Tensor a = rand({3, 4, 5}, type);
  // 对张量 a 执行维度置换操作，新张量 b 的维度应为 (4, 5, 3)
  Tensor b = a.permute({1, 2, 0});
  // 断言张量 b 的维度是否为 (4, 5, 3)
  ASSERT_TRUE(b.sizes().equals({4, 5, 3}));
  // 断言张量 b 的步长是否为 (5, 1, 20)
  ASSERT_TRUE(b.strides().equals({5, 1, 20}));
}

// 测试函数：测试矩阵乘法
void TestMm(DeprecatedTypeProperties& type) {
  // 如果张量类型的后端不是 CPU 或标量类型不是 kHalf，则跳过测试
  if (type.backend() != Backend::CPU || type.scalarType() != kHalf) {
    // 创建大小为 (3, 4) 的随机张量 a 和大小为 (4) 的随机张量 b
    Tensor a = rand({3, 4}, type);
    Tensor b = rand({4}, type);
    // 执行矩阵向量乘法操作，结果保存在张量 c 中
    Tensor c = mv(a, b);
    // 断言张量 c 是否等于 addmv 函数的结果
    ASSERT_TRUE(c.equal(addmv(zeros({3}, type), a, b, 0, 1)));
  }
}

// 测试函数：测试张量压缩（去除维度为 1 的维度）
void TestSqueeze(DeprecatedTypeProperties& type) {
  // 创建大小为 (2, 1) 的随机张量 a
  Tensor a = rand({2, 1}, type);
  // 执行张量压缩操作，新张量 b 应当维度为 1
  Tensor b = squeeze(a);
  // 断言张量 b 的维度是否为 1
  ASSERT_EQ_RESOLVED(b.dim(), 1);
  // 创建大小为 (1) 的随机张量 a
  a = rand({1}, type);
  // 再次执行张量压缩操作，压缩后应与 a 相等
  b = squeeze(a);
  // TODO 0-dim squeeze
  // 断言张量 a 和 b 是否相等
  ASSERT_TRUE(a[0].equal(b));
}

// 测试函数：测试张量复制
void TestCopy(DeprecatedTypeProperties& type) {
  // 创建大小为 (4, 3) 的全零张量 a 和大小相同的随机张量 e
  Tensor a = zeros({4, 3}, type);
  Tensor e = rand({4, 3}, type);
  // 将张量 e 的值复制到张量 a 中
  a.copy_(e);
  // 断言张量 a 和 e 是否相等
  ASSERT_TRUE(a.equal(e));
}

// 测试函数：测试张量广播复制
void TestCopyBroadcasting(DeprecatedTypeProperties& type) {
  // 创建大小为 (4, 3) 的全零张量 a 和大小为 (3) 的随机张量 e
  Tensor a = zeros({4, 3}, type);
  Tensor e = rand({3}, type);
  // 将张量 e 的值广播复制到张量 a 中
  a.copy_(e);
  // 对每一行执行断言：张量 a[i] 是否与张量 e 相等
  for (const auto i : c10::irange(4)) {
    ASSERT_TRUE(a[i].equal(e));
  }
}

// 测试函数：测试计算张量的绝对值
void TestAbsValue(DeprecatedTypeProperties& type) {
  // 计算给定类型中 -3 的绝对值，并保存到张量 r 中
  Tensor r = at::abs(at::scalar_tensor(-3, type.options()));
  // 断言张量 r 的值是否为 3
  ASSERT_EQ_RESOLVED(r.item<int32_t>(), 3);
}
/*
   TODO(zach): 运算符重载
#if 0
{
std::cout << "eq (value):" << std::endl;
Tensor a = Tensor(10.f);
std::cout << (a == 11_i64) << " -- should be 0" << std::endl;
std::cout << (a == 10_i64) << " -- should be 1" << std::endl;
std::cout << (a == 10.) << " -- should be 1" << std::endl;
}
#endif
*/
void TestAddingAValueWithScalar(DeprecatedTypeProperties& type) {
  // 创建一个形状为 {4, 3} 的随机张量 a
  Tensor a = rand({4, 3}, type);
  // 断言：ones({4, 3}, type) 加上张量 a 的结果等于 add(a, 1)
  ASSERT_TRUE((ones({4, 3}, type) + a).equal(add(a, 1)));
}

void TestSelect(DeprecatedTypeProperties& type) {
  // 创建一个形状为 {3, 7} 的随机张量 a
  Tensor a = rand({3, 7}, type);
  // 在第 1 维度上选择索引为 3 的元素，得到 a_13
  auto a_13 = select(a, 1, 3);
  // 在 a_13 的第 1 维度上选择索引为 2 的元素，得到 a_13_02
  auto a_13_02 = select(select(a, 1, 3), 0, 2);
  // 断言：a 的第 0 行第 3 列的元素等于 a_13 的第 0 行
  ASSERT_TRUE(a[0][3].equal(a_13[0]));
  // 断言：a 的第 2 行第 3 列的元素等于 a_13_02 的第 0 行
  ASSERT_TRUE(a[2][3].equal(a_13_02));
}

void TestZeroDim(DeprecatedTypeProperties& type) {
  // 创建一个标量值为 4 的零维张量 a
  Tensor a = at::scalar_tensor(4, type.options()); // rand(type, {1});

  // 创建一个形状为 {3, 4} 的随机张量 b
  Tensor b = rand({3, 4}, type);
  // 断言：a 与自身相加后的维度为 0
  ASSERT_EQ_RESOLVED((a + a).dim(), 0);
  // 断言：1 加上 a 后的维度为 0
  ASSERT_EQ_RESOLVED((1 + a).dim(), 0);
  // 断言：b 加上 a 后的维度为 2
  ASSERT_EQ_RESOLVED((b + a).dim(), 2);
  // 断言：a 加上 b 后的维度为 2
  ASSERT_EQ_RESOLVED((a + b).dim(), 2);
  
  // 创建一个形状为 {3, 4} 的随机张量 c
  auto c = rand({3, 4}, type);
  // 断言：c 的第 1 行第 2 列的元素的维度为 0
  ASSERT_EQ_RESOLVED(c[1][2].dim(), 0);

  // 创建一个形状为 {3, 4} 的随机张量 f
  auto f = rand({3, 4}, type);
  // 将 f 的第 2 行设为形状为 {4}、元素全为 0 的张量
  f[2] = zeros({4}, type);
  // 将 f 的第 1 行第 0 列的元素设为 -1
  f[1][0] = -1;
  // 断言：f 的第 2 行第 0 列的元素值为 0
  ASSERT_EQ_RESOLVED(f[2][0].item<double>(), 0);
}

void TestToCFloat() {
  // 创建一个形状为 {3, 4} 的全零张量 a
  Tensor a = zeros({3, 4});
  // 创建一个形状为 {3, 7} 的全一张量 b
  Tensor b = ones({3, 7});
  // 将张量 a 和 b 沿第 1 维度拼接成张量 c
  Tensor c = cat({a, b}, 1);
  // 断言：张量 c 在第 1 维度上的尺寸为 11
  ASSERT_EQ_RESOLVED(c.size(1), 11);

  // 创建一个标量值为随机数的张量 e
  Tensor e = rand({});
  // 断言：张量 e 的数据指针转为 float 后的值等于其元素和的值（单一元素张量）
  ASSERT_EQ_RESOLVED(*e.data_ptr<float>(), e.sum().item<float>());
}

void TestToString() {
  // 创建一个形状为 {3, 7} 的全一张量 b，每个元素乘以 0.0000001f
  Tensor b = ones({3, 7}) * .0000001f;
  // 创建一个字符串流 s
  std::stringstream s;
  // 将张量 b 输出到字符串流 s，并换行
  s << b << "\n";
  // 期望输出的字符串
  std::string expect = "1e-07 *";
  // 断言：字符串流 s 的前几个字符与期望的字符串相等
  ASSERT_EQ_RESOLVED(s.str().substr(0, expect.size()), expect);
}

void TestIndexingByScalar() {
  // 创建一个从 0 到 9 的整数张量 tensor
  Tensor tensor = arange(0, 10, kInt);
  // 创建一个值为 1 的零维张量 one
  Tensor one = ones({}, kInt);
  // 用范围循环索引 tensor 的每个元素 i
  for (const auto i : c10::irange(tensor.numel())) {
    // 断言：tensor 的第 i 个元素等于 one 乘以 i
    ASSERT_TRUE(tensor[i].equal(one * i));
  }
  // 同上，使用 size_t 类型索引 tensor 的每个元素 i
  for (size_t i = 0; i < static_cast<uint64_t>(tensor.numel()); ++i) {
    // 断言：tensor 的第 i 个元素等于 one 乘以 i
    ASSERT_TRUE(tensor[i].equal(one * static_cast<int64_t>(i)));
  }
  // 再次使用范围循环索引 tensor 的每个元素 i
  for (const auto i : c10::irange(tensor.numel())) {
    // 断言：tensor 的第 i 个元素等于 one 乘以 i
    ASSERT_TRUE(tensor[i].equal(one * i));
  }
  // 使用 int16_t 类型索引 tensor 的每个元素 i
  // NOLINTNEXTLINE(bugprone-too-small-loop-variable)
  for (int16_t i = 0; i < tensor.numel(); ++i) {
    // 断言：tensor 的第 i 个元素等于 one 乘以 i
    ASSERT_TRUE(tensor[i].equal(one * i));
  }
  // 使用 int8_t 类型索引 tensor 的每个元素 i
  // NOLINTNEXTLINE(bugprone-too-small-loop-variable)
  for (int8_t i = 0; i < tensor.numel(); ++i) {
    // 断言：tensor 的第 i 个元素等于 one 乘以 i
    ASSERT_TRUE(tensor[i].equal(one * i));
  }
  // 断言：使用 Scalar(3.14) 索引 tensor 会抛出异常
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(tensor[Scalar(3.14)].equal(one));
}

void TestIndexingByZerodimTensor() {
  // 创建一个从 0 到 9 的整数张量 tensor
  Tensor tensor = arange(0, 10, kInt);
  // 创建一个值为 1 的零维张量 one
  Tensor one = ones({}, kInt);
  // 用范围循环索引 tensor 的每个元素 i
  for (const auto i : c10::irange(tensor.numel())) {
    ASSERT_TRUE(tensor[one * i].equal(one * i));
  }
  // 抛出异常，提示只能使用整数标量来索引张量
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(tensor[ones({}) * 3.14].equal(one));
  // 抛出异常，提示只能使用已定义的张量来索引
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(tensor[Tensor()].equal(one));
  // 抛出异常，提示只能使用零维标量（即标量）来索引
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(tensor[ones({2, 3, 4}, kInt)].equal(one));
void TestIndexingMixedDevice(DeprecatedTypeProperties& type) {
  // 创建一个大小为20x20的随机张量
  Tensor tensor = randn({20, 20}, type);
  // 创建一个从0到9的整数张量，并将其转移到CPU上
  Tensor index = arange(10, kLong).cpu();
  // 使用索引张量在原始张量上进行索引操作
  Tensor result = tensor.index({index});
  // 断言结果张量的第一个元素与原始张量的第一个元素相等
  ASSERT_TRUE(result[0].equal(tensor[0]));
}

void TestDispatch() {
  // 创建两个大小为20x20的随机张量
  Tensor tensor = randn({20, 20});
  Tensor other = randn({20, 20});
  // 对第一个张量进行一系列操作：relu激活函数 -> 均方误差损失计算 -> 平均值降维
  auto result = tensor.m(relu).m(mse_loss, other, at::Reduction::Mean);
  // 断言结果张量与经过相同操作的张量的结果相近
  ASSERT_TRUE(result.allclose(mse_loss(relu(tensor), other)));
}

void TestNegativeDim(DeprecatedTypeProperties& type) {
  // 检查在创建张量时使用负数维度是否会抛出异常
  ASSERT_ANY_THROW(empty({5, -5, 5}, type.options()));
  ASSERT_ANY_THROW(empty({5, -5, -5}, type.options()));
  // 创建一个大小为5x5的空张量
  Tensor tensor = empty({5, 5}, type.options());
  // 检查在调整张量形状时使用负数维度是否会抛出异常
  ASSERT_ANY_THROW(tensor.reshape({-5, -5}));
}

void TestView(DeprecatedTypeProperties& type) {
  // 测试张量视图路径与变量视图路径的区别
  // 更多详情请参考：https://github.com/pytorch/pytorch/pull/23452
  // 创建一个大小为3x4的随机张量
  Tensor tensor = randn({3, 4}, type);;
  // 创建一个视图张量，与原始张量共享数据但形状不同
  Tensor viewed = tensor.view({3, 4});
  // 改变原始张量的形状为6x2
  tensor.resize_({6, 2});
  // 断言张量的尺寸已经改变为6x2
  ASSERT_TRUE(tensor.sizes().equals({6, 2}));
  // 断言视图张量的尺寸仍然是3x4，不受原始张量形状变化的影响
  ASSERT_TRUE(viewed.sizes().equals({3, 4}));
}

void TestIntArrayRefExpansion(DeprecatedTypeProperties& type) {
  // 如果不是CPU后端或者不是半精度类型，则跳过测试
  if (type.backend() != Backend::CPU || type.scalarType() != kHalf) {
    // 对不同维度的输入进行最大池化和平均池化操作，扩展参数列表
    max_pool2d(randn({3, 3, 3, 3}, type.options()), 2, 1, 1, 1);
    max_pool3d(randn({3, 3, 3, 3, 3}, type.options()), 2, 1, 1, 1);
    avg_pool2d(randn({3, 3, 3, 3}, type.options()), 2, 1, 1);
    avg_pool3d(randn({3, 3, 3, 3, 3}, type.options()), 2, 1, 1);
  }
}

void test(DeprecatedTypeProperties& type) {
  // 依次执行多个测试函数
  TestResize(type);
  TestOnesAndDot(type);

  TestSort(type);
  TestRandperm(type);
  TestAdd(type);
  TestZeros(type);
  TestLoadsOfAdds(type);
  TestLoadOfAddsWithCopy(type);
  TestIsContiguous(type);
  TestPermute(type);
  TestMm(type);
  TestSqueeze(type);
  TestCopy(type);
  TestCopyBroadcasting(type);
  TestAbsValue(type);
  TestAddingAValueWithScalar(type);
  TestSelect(type);
  TestZeroDim(type);
  TestToCFloat();
  TestToString();
  TestIndexingByScalar();
  TestIndexingByZerodimTensor();
  TestIndexingMixedDevice(type);
  TestDispatch();
  TestNegativeDim(type);
  TestView(type);
  TestIntArrayRefExpansion(type);
}

TEST(BasicTest, BasicTestCPU) {
  // 设置随机种子
  manual_seed(123);

  // 使用CPU和单精度浮点类型执行测试
  test(CPU(kFloat));
}

TEST(BasicTest, BasicTestHalfCPU) {
  // 设置随机种子
  manual_seed(234);

  // 使用CPU和半精度浮点类型执行测试
  test(CPU(kHalf));
}

TEST(BasicTest, BasicTestCUDA) {
  // 设置随机种子
  manual_seed(123);

  // 如果支持CUDA，则使用CUDA和单精度浮点类型执行测试
  if (at::hasCUDA()) {
    test(CUDA(kFloat));
  }
}
TEST(BasicTest, FactoryMethodsTest) {
  // 创建一个大小为4的空张量，默认为float类型，CPU上的连续布局，不需要梯度，未固定在内存中
  at::Tensor tensor0 = at::empty({4});
  ASSERT_EQ(tensor0.dtype(), at::kFloat); // 断言张量的数据类型为float
  ASSERT_EQ(tensor0.layout(), at::kStrided); // 断言张量的布局为连续布局
  ASSERT_EQ(tensor0.device(), at::kCPU); // 断言张量在CPU上
  ASSERT_FALSE(tensor0.requires_grad()); // 断言张量不需要梯度
  ASSERT_FALSE(tensor0.is_pinned()); // 断言张量未固定在内存中

  // 测试将requires_grad设置为false
  tensor0 = at::empty({4}, at::TensorOptions().requires_grad(false));
  ASSERT_EQ(tensor0.dtype(), at::kFloat); // 断言张量的数据类型为float
  ASSERT_EQ(tensor0.layout(), at::kStrided); // 断言张量的布局为连续布局
  ASSERT_EQ(tensor0.device(), at::kCPU); // 断言张量在CPU上
  ASSERT_FALSE(tensor0.requires_grad()); // 断言张量不需要梯度
  ASSERT_FALSE(tensor0.is_pinned()); // 断言张量未固定在内存中

  // 测试将requires_grad设置为true
  // 这是一个bug。requires_grad被设置为TRUE，但是这个功能尚未实现。
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_ANY_THROW(at::empty({4}, at::TensorOptions().requires_grad(true)));

  // 测试设置dtype
  at::Tensor tensor1 = at::empty({4}, at::TensorOptions().dtype(at::kHalf));
  ASSERT_EQ(tensor1.dtype(), at::kHalf); // 断言张量的数据类型为half
  ASSERT_EQ(tensor1.layout(), at::kStrided); // 断言张量的布局为连续布局
  ASSERT_EQ(tensor1.device(), at::kCPU); // 断言张量在CPU上
  ASSERT_FALSE(tensor1.requires_grad()); // 断言张量不需要梯度
  ASSERT_FALSE(tensor1.is_pinned()); // 断言张量未固定在内存中

  // CPU稀疏张量测试以避免需要CUDA来捕捉简单的错误。
  // 稀疏张量不适用于静态CPU分发。
#ifndef ATEN_CPU_STATIC_DISPATCH
  tensor1 = at::empty({4}, at::TensorOptions().dtype(at::kHalf).layout(at::kSparse));
  ASSERT_EQ(tensor1.dtype(), at::kHalf); // 断言张量的数据类型为half
  ASSERT_EQ(tensor1.layout(), at::kSparse); // 断言张量的布局为稀疏布局
  ASSERT_EQ(tensor1.device(), at::kCPU); // 断言张量在CPU上
  ASSERT_FALSE(tensor1.requires_grad()); // 断言张量不需要梯度
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_FALSE(tensor1.is_pinned()); // 断言张量未固定在内存中
#endif // ATEN_CPU_STATIC_DISPATCH

  if (torch::cuda::is_available()) {
    // 测试设置固定内存
    tensor1 = at::empty({4}, at::TensorOptions().pinned_memory(true));
    ASSERT_EQ(tensor1.dtype(), at::kFloat); // 断言张量的数据类型为float
    ASSERT_EQ(tensor1.layout(), at::kStrided); // 断言张量的布局为连续布局
    ASSERT_EQ(tensor1.device(), at::kCPU); // 断言张量在CPU上
    ASSERT_EQ(tensor1.requires_grad(), false); // 断言张量不需要梯度
    ASSERT_FALSE(tensor1.device().is_cuda()); // 断言张量不在CUDA上
    ASSERT_TRUE(tensor1.is_pinned()); // 断言张量固定在内存中

    // 测试设置设备
    tensor1 = at::empty({4}, at::TensorOptions().device(at::kCUDA));
    ASSERT_EQ(tensor1.dtype(), at::kFloat); // 断言张量的数据类型为float
    ASSERT_EQ(tensor1.layout(), at::kStrided); // 断言张量的布局为连续布局
    ASSERT_TRUE(tensor1.device().is_cuda()); // 断言张量在CUDA上
    ASSERT_FALSE(tensor1.requires_grad()); // 断言张量不需要梯度
    ASSERT_FALSE(tensor1.is_pinned()); // 断言张量未固定在内存中

    // 测试设置所有选项
    tensor1 = at::empty({4}, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA).layout(at::kSparse).requires_grad(false));
    ASSERT_EQ(tensor1.dtype(), at::kHalf); // 断言张量的数据类型为half
    ASSERT_EQ(tensor1.layout(), at::kSparse); // 断言张量的布局为稀疏布局
    ASSERT_TRUE(tensor1.device().is_cuda()); // 断言张量在CUDA上
    ASSERT_THROWS(tensor1.nbytes()); // 断言抛出异常，测试未实现的功能

    // 这是一个bug
    // 问题https://github.com/pytorch/pytorch/issues/30405
    ASSERT_FALSE(tensor1.requires_grad()); // 断言张量不需要梯度
    ASSERT_FALSE(tensor1.is_pinned());
  }

  // Test _like variants
  // 检查是否CUDA可用，解决GitHub问题28093
  if (torch::cuda::is_available()) {
    // 创建一个空的双精度张量作为原型
    at::Tensor proto = at::empty({1}, at::kDouble);
    // 使用 proto 张量的属性创建一个新的 CUDA 张量
    tensor0 = at::empty_like(proto, at::kCUDA);
    // 断言新张量的数据类型为双精度
    ASSERT_EQ(tensor0.dtype(), at::kDouble);
    // 断言新张量的布局为步进布局（strided）
    ASSERT_EQ(tensor0.layout(), at::kStrided);
    // 断言新张量位于 CUDA 设备上
    ASSERT_TRUE(tensor0.device().is_cuda());
    // 断言新张量不需要梯度
    ASSERT_FALSE(tensor0.requires_grad());
    // 断言新张量不是固定在内存中的
    ASSERT_FALSE(tensor0.is_pinned());
  }
}

TEST(BasicTest, BasicStdTestCPU) {
  c10::once_flag flag1, flag2;

  auto simple_do_once = [&]()
  {
      // lambda 函数，确保这段代码只会执行一次
      c10::call_once(flag1, [](){ std::cout << "Simple example: called once\n"; });
  };

  auto may_throw_function = [&](bool do_throw)
  {
    if (do_throw) {
      // 如果抛出异常，则输出一条消息
      std::cout << "throw: call_once will retry\n"; // this may appear more than once
      // 抛出自定义异常
      TORCH_CHECK(false, "throw exception");
    }
    // 如果没有抛出异常，则输出一条消息
    std::cout << "Didn't throw, call_once will not attempt again\n"; // guaranteed once
  };

  auto do_once = [&](bool do_throw)
  {
    try {
      // 使用 c10::call_once 保证 may_throw_function 只被调用一次
      c10::call_once(flag2, may_throw_function, do_throw);
    }
    catch (...) {
      // 捕获所有异常
    }
  };

  // 创建多个线程执行 simple_do_once
  std::thread st1(simple_do_once);
  std::thread st2(simple_do_once);
  std::thread st3(simple_do_once);
  std::thread st4(simple_do_once);
  // 等待所有线程执行完毕
  st1.join();
  st2.join();
  st3.join();
  st4.join();

  // 创建多个线程执行 do_once，参数为 true 或 false
  std::thread t1(do_once, true);
  std::thread t2(do_once, true);
  std::thread t3(do_once, false);
  std::thread t4(do_once, true);
  // 等待所有线程执行完毕
  t1.join();
  t2.join();
  t3.join();
  t4.join();
}
```