# `.\pytorch\aten\src\ATen\test\scalar_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <iostream>  // 引入标准输入输出流库
#include <random>  // 引入随机数生成库
#include <c10/core/SymInt.h>  // 引入 c10 库的 SymInt 头文件

// 在使用 MSVC 编译器时，定义常量如 M_PI 和 C 关键字的头文件
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <ATen/ATen.h>  // 引入 PyTorch ATen 库
#include <ATen/Dispatch.h>  // 引入 PyTorch ATen 的分发机制

// 在本文件中故意测试自赋值/移动操作，忽略相关警告
#ifndef _MSC_VER
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wself-move"
#endif

#ifdef __clang__
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif

using std::cout;  // 使用标准命名空间中的 cout 对象输出
using namespace at;  // 使用 at 命名空间

template<typename scalar_type>
struct Foo {
  static void apply(Tensor a, Tensor b) {
    scalar_type s = 1;  // 定义并初始化类型为 scalar_type 的变量 s
    std::stringstream ss;  // 创建字符串流对象 ss
    ss << "hello, dispatch: " << a.toString() << s << "\n";  // 将信息写入字符串流 ss
    auto data = (scalar_type*)a.data_ptr();  // 获取 Tensor a 的数据指针并转换为 scalar_type 类型
    (void)data;  // 防止未使用 data 变量的编译器警告
  }
};

template<>
struct Foo<Half> {
  static void apply(Tensor a, Tensor b) {}  // 当 scalar_type 为 Half 时的特化，空实现
};

void test_overflow() {
  auto s1 = Scalar(M_PI);  // 使用 M_PI 定义一个标量 s1
  ASSERT_EQ(s1.toFloat(), static_cast<float>(M_PI));  // 断言 s1 转为 float 后的值与 M_PI 相等
  s1.toHalf();  // 将 s1 转为 Half 类型

  s1 = Scalar(100000);  // 将 s1 赋值为整数标量 100000
  ASSERT_EQ(s1.toFloat(), 100000.0);  // 断言 s1 转为 float 后的值为 100000.0
  ASSERT_EQ(s1.toInt(), 100000);  // 断言 s1 转为整数后的值为 100000

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(s1.toHalf(), std::runtime_error);  // 断言转为 Half 类型时抛出运行时错误

  s1 = Scalar(NAN);  // 将 s1 赋值为 NaN
  ASSERT_TRUE(std::isnan(s1.toFloat()));  // 断言 s1 转为 float 后是 NaN
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(s1.toInt(), std::runtime_error);  // 断言转为整数时抛出运行时错误

  s1 = Scalar(INFINITY);  // 将 s1 赋值为无穷大
  ASSERT_TRUE(std::isinf(s1.toFloat()));  // 断言 s1 转为 float 后是无穷大
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(s1.toInt(), std::runtime_error);  // 断言转为整数时抛出运行时错误
}

TEST(TestScalar, TestScalar) {
  manual_seed(123);  // 设置随机数种子为 123

  Scalar what = 257;  // 定义标量 what，并初始化为 257
  Scalar bar = 3.0;  // 定义标量 bar，并初始化为 3.0
  Half h = bar.toHalf();  // 将 bar 转为 Half 类型，并赋给 h
  Scalar h2 = h;  // 将 h 转为标量 h2
  cout << "H2: " << h2.toDouble() << " " << what.toFloat() << " "  // 输出 H2 的各种转换结果
       << bar.toDouble() << " " << what.isIntegral(false) << "\n";

  auto gen = at::detail::getDefaultCPUGenerator();  // 获取默认的 CPU 生成器
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());  // 使用随机数生成器时获取锁保护
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_NO_THROW(gen.set_current_seed(std::random_device()()));  // 断言设置当前种子时不抛出异常
  }

  if (at::hasCUDA()) {  // 如果系统支持 CUDA
    auto t2 = zeros({4, 4}, at::kCUDA);  // 创建一个在 CUDA 上的全零 Tensor
    // 打印变量 t2 的地址
    cout << &t2 << "\n";
    }
    
    // 创建一个大小为 4x4 的张量 t，元素全为 1
    auto t = ones({4, 4});
    
    // 创建一个大小为 4x4 的零张量 wha2，加上张量 t 的元素，并计算所有元素的和
    auto wha2 = zeros({4, 4}).add(t).sum();
    
    // 断言 wha2 的元素值等于 16.0
    ASSERT_EQ(wha2.item<double>(), 16.0);
    
    // 断言张量 t 的第一个维度大小为 4
    ASSERT_EQ(t.sizes()[0], 4);
    // 断言张量 t 的第二个维度大小为 4
    ASSERT_EQ(t.sizes()[1], 4);
    // 断言张量 t 的第一个维度步长为 4
    ASSERT_EQ(t.strides()[0], 4);
    // 断言张量 t 的第二个维度步长为 1
    ASSERT_EQ(t.strides()[1], 1);
    
    // 创建浮点类型的张量选项
    TensorOptions options = dtype(kFloat);
    
    // 创建大小为 [1, 10] 的随机张量 x
    Tensor x = randn({1, 10}, options);
    // 创建大小为 [1, 20] 的随机张量 prev_h
    Tensor prev_h = randn({1, 20}, options);
    // 创建大小为 [20, 20] 的随机张量 W_h
    Tensor W_h = randn({20, 20}, options);
    // 创建大小为 [20, 10] 的随机张量 W_x
    Tensor W_x = randn({20, 10}, options);
    
    // 计算矩阵乘法 W_x * x.t()，结果存储在 i2h 中
    Tensor i2h = at::mm(W_x, x.t());
    // 计算矩阵乘法 W_h * prev_h.t()，结果存储在 h2h 中
    Tensor h2h = at::mm(W_h, prev_h.t());
    // 计算 i2h 和 h2h 的元素级加法，并存储结果在 next_h 中
    Tensor next_h = i2h.add(h2h);
    // 对 next_h 中的每个元素应用双曲正切函数
    next_h = next_h.tanh();
    
    // 检查是否抛出异常，预期是抛出异常，因为尝试对空张量调用 item() 方法
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_ANY_THROW(Tensor{}.item());
    
    // 调用测试溢出的函数
    test_overflow();
    
    // 如果当前环境支持 CUDA
    if (at::hasCUDA()) {
      // 将 next_h 转移到 CUDA 设备上，并确保转换后的张量在 CPU 上与原始张量相等
      auto r = next_h.to(at::Device(kCUDA), kFloat, /*non_blocking=*/ false, /*copy=*/ true);
      ASSERT_TRUE(r.to(at::Device(kCPU), kFloat, /*non_blocking=*/ false, /*copy=*/ true).equal(next_h));
    }
    
    // 检查是否抛出异常，预期是不会抛出异常，因为随机生成了一个大小为 [10, 10, 2] 的张量
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_NO_THROW(randn({10, 10, 2}, options));
    
    // 使用 Scalar.toTensor() 函数测试不同数据类型的标量转换为张量后的标量类型
    ASSERT_EQ(scalar_to_tensor(bar).scalar_type(), kDouble);
    ASSERT_EQ(scalar_to_tensor(what).scalar_type(), kLong);
    ASSERT_EQ(scalar_to_tensor(ones({}).item()).scalar_type(), kDouble);
    
    // 如果张量 x 的标量类型不是半精度浮点类型
    if (x.scalar_type() != ScalarType::Half) {
      // 使用 AT_DISPATCH_ALL_TYPES 宏遍历所有张量的数据类型
      AT_DISPATCH_ALL_TYPES(x.scalar_type(), "foo", [&] {
        // 将字符串 "hello, dispatch"、张量 x 的字符串表示、标量 s 的值转换为字符串，并写入 stringstream ss
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
        ASSERT_NO_THROW(
            ss << "hello, dispatch" << x.toString() << s << "\n");
        auto data = (scalar_t*)x.data_ptr();
        (void)data;
      });
    }
    
    // 测试直接的 C 标量类型转换
    {
      auto x = ones({1, 2}, options);
      // 检查是否抛出异常，预期是抛出异常，因为尝试将张量 x 转换为 float 类型的标量
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
      ASSERT_ANY_THROW(x.item<float>());
    }
    
    // 创建一个大小为 [1] 的浮点类型张量 float_one
    auto float_one = ones({}, options);
    // 断言 float_one 的元素值等于 1
    ASSERT_EQ(float_one.item<float>(), 1);
    // 断言 float_one 的元素值等于 1，并转换为 int32_t 类型
    ASSERT_EQ(float_one.item<int32_t>(), 1);
    // 断言 float_one 的元素值等于 1，并转换为半精度浮点类型
    ASSERT_EQ(float_one.item<at::Half>(), 1);
}

// 定义名为 TEST 的宏，用于声明单元测试 TestScalar 的测试用例 TestConj
TEST(TestScalar, TestConj) {
  // 创建整数标量 int_scalar，并赋值为 257
  Scalar int_scalar = 257;
  // 创建浮点数标量 float_scalar，并赋值为 3.0
  Scalar float_scalar = 3.0;
  // 创建复数标量 complex_scalar，并赋值为复数值 (2.3, 3.5)
  Scalar complex_scalar = c10::complex<double>(2.3, 3.5);

  // 断言整数标量的共轭后转换为整数为 257
  ASSERT_EQ(int_scalar.conj().toInt(), 257);
  // 断言浮点数标量的共轭后转换为双精度浮点数为 3.0
  ASSERT_EQ(float_scalar.conj().toDouble(), 3.0);
  // 断言复数标量的共轭后转换为双精度复数为 (2.3, -3.5)
  ASSERT_EQ(complex_scalar.conj().toComplexDouble(), c10::complex<double>(2.3, -3.5));
}

// 定义名为 TEST 的宏，用于声明单元测试 TestScalar 的测试用例 TestEqual
TEST(TestScalar, TestEqual) {
  // 断言浮点数标量 1.0 不等于布尔值 false
  ASSERT_FALSE(Scalar(1.0).equal(false));
  // 断言浮点数标量 1.0 不等于布尔值 true
  ASSERT_FALSE(Scalar(1.0).equal(true));
  // 断言布尔值 true 不等于浮点数标量 1.0
  ASSERT_FALSE(Scalar(true).equal(1.0));
  // 断言布尔值 true 等于布尔值 true
  ASSERT_TRUE(Scalar(true).equal(true));

  // 断言复数标量 (2.0, 5.0) 等于复数 (2.0, 5.0)
  ASSERT_TRUE(Scalar(c10::complex<double>{2.0, 5.0}).equal(c10::complex<double>{2.0, 5.0}));
  // 断言复数标量 (2.0, 0) 等于浮点数 2.0
  ASSERT_TRUE(Scalar(c10::complex<double>{2.0, 0}).equal(2.0));
  // 断言复数标量 (2.0, 0) 等于整数 2
  ASSERT_TRUE(Scalar(c10::complex<double>{2.0, 0}).equal(2));

  // 断言浮点数标量 2.0 等于复数 (2.0, 0.0)
  ASSERT_TRUE(Scalar(2.0).equal(c10::complex<double>{2.0, 0.0}));
  // 断言浮点数标量 2.0 不等于复数 (2.0, 4.0)
  ASSERT_FALSE(Scalar(2.0).equal(c10::complex<double>{2.0, 4.0}));
  // 断言浮点数标量 2.0 不等于浮点数 3.0
  ASSERT_FALSE(Scalar(2.0).equal(3.0));
  // 断言浮点数标量 2.0 等于整数 2
  ASSERT_TRUE(Scalar(2.0).equal(2));

  // 断言整数标量 2 等于复数 (2.0, 0)
  ASSERT_TRUE(Scalar(2).equal(c10::complex<double>{2.0, 0}));
  // 断言整数标量 2 等于整数 2
  ASSERT_TRUE(Scalar(2).equal(2));
  // 断言整数标量 2 等于浮点数 2.0
  ASSERT_TRUE(Scalar(2).equal(2.0));
}

// 定义名为 TEST 的宏，用于声明单元测试 TestScalar 的测试用例 TestFormatting
TEST(TestScalar, TestFormatting) {
  // 定义 lambda 表达式 format，用于将标量 a 转换为字符串
  auto format = [] (Scalar a) {
    // 创建字符串流 str
    std::ostringstream str;
    // 将标量 a 输出到字符串流 str
    str << a;
    // 返回字符串流 str 的字符串表示
    return str.str();
  };
  // 断言整数标量 3 转换为字符串为 "3"
  ASSERT_EQ("3", format(Scalar(3)));
  // 断言浮点数标量 3.1 转换为字符串为 "3.1"
  ASSERT_EQ("3.1", format(Scalar(3.1)));
  // 断言布尔值 true 转换为字符串为 "true"
  ASSERT_EQ("true", format(Scalar(true)));
  // 断言布尔值 false 转换为字符串为 "false"
  ASSERT_EQ("false", format(Scalar(false)));
  // 断言复数标量 (2.0, 3.1) 转换为字符串为 "(2,3.1)"
  ASSERT_EQ("(2,3.1)", format(Scalar(c10::complex<double>(2.0, 3.1))));
  // 断言复数标量 (2.0, 3.1) 转换为字符串为 "(2,3.1)"
  ASSERT_EQ("(2,3.1)", format(Scalar(c10::complex<float>(2.0, 3.1))));
  // 断言整数标量 4 转换为符号整数后再转换为字符串为 "4"
  ASSERT_EQ("4", format(Scalar(Scalar(4).toSymInt())));
}
```