# `.\pytorch\c10\test\util\complex_math_test_common.h`

```
// 引用必要的头文件：c10/util/complex.h 是处理复数的头文件，gtest/gtest.h 是 Google 测试框架的头文件
#include <c10/util/complex.h>
#include <gtest/gtest.h>

// 如果未定义 PI，则定义为精确的圆周率值
#ifndef PI
#define PI 3.141592653589793238463
#endif

// 如果未定义 tol，则定义为测试中使用的数值容差
#ifndef tol
#define tol 1e-6
#endif

// 指数函数测试开始

// 定义测试用例 TestExponential，名称为 IPi
C10_DEFINE_TEST(TestExponential, IPi) {
  // 测试 exp(i*pi) = -1
  {
    // 计算复数 exp(i*pi)，其中 i 是虚数单位，PI 是圆周率
    c10::complex<float> e_i_pi = std::exp(c10::complex<float>(0, float(PI)));
    // 断言实部接近 -1，虚部接近 0，tol 是容差值
    C10_ASSERT_NEAR(e_i_pi.real(), -1, tol);
    C10_ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
  {
    // 同上，使用全局命名空间下的 exp 函数
    c10::complex<float> e_i_pi = ::exp(c10::complex<float>(0, float(PI)));
    C10_ASSERT_NEAR(e_i_pi.real(), -1, tol);
    C10_ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
  {
    // 使用双精度浮点数进行测试
    c10::complex<double> e_i_pi = std::exp(c10::complex<double>(0, PI));
    C10_ASSERT_NEAR(e_i_pi.real(), -1, tol);
    C10_ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
  {
    // 同上，使用全局命名空间下的 exp 函数
    c10::complex<double> e_i_pi = ::exp(c10::complex<double>(0, PI));
    C10_ASSERT_NEAR(e_i_pi.real(), -1, tol);
    C10_ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
}

// 定义测试用例 TestExponential，名称为 EulerFormula
C10_DEFINE_TEST(TestExponential, EulerFormula) {
  // 测试 exp(ix) = cos(x) + i * sin(x)
  {
    // 定义复数 x = 0.1 + 1.2i
    c10::complex<float> x(0.1, 1.2);
    // 计算 exp(x)
    c10::complex<float> e = std::exp(x);
    // 期望的实部和虚部
    float expected_real = std::exp(x.real()) * std::cos(x.imag());
    float expected_imag = std::exp(x.real()) * std::sin(x.imag());
    // 断言计算值接近期望值
    C10_ASSERT_NEAR(e.real(), expected_real, tol);
    C10_ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
  {
    // 同上，使用全局命名空间下的 exp 函数
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> e = ::exp(x);
    float expected_real = ::exp(x.real()) * ::cos(x.imag());
    float expected_imag = ::exp(x.real()) * ::sin(x.imag());
    C10_ASSERT_NEAR(e.real(), expected_real, tol);
    C10_ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
  {
    // 使用双精度浮点数进行测试
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> e = std::exp(x);
    float expected_real = std::exp(x.real()) * std::cos(x.imag());
    float expected_imag = std::exp(x.real()) * std::sin(x.imag());
    C10_ASSERT_NEAR(e.real(), expected_real, tol);
    C10_ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
  {
    // 同上，使用全局命名空间下的 exp 函数
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> e = ::exp(x);
    float expected_real = ::exp(x.real()) * ::cos(x.imag());
    float expected_imag = ::exp(x.real()) * ::sin(x.imag());
    C10_ASSERT_NEAR(e.real(), expected_real, tol);
    C10_ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
}

// 定义测试用例 TestExpm1，名称为 Normal
C10_DEFINE_TEST(TestExpm1, Normal) {
  // 测试 expm1(x) = exp(x) - 1
  {
    // 定义复数 x = 0.1 + 1.2i
    c10::complex<float> x(0.1, 1.2);
    // 计算 expm1(x) 和 exp(x) - 1
    c10::complex<float> l1 = std::expm1(x);
    c10::complex<float> l2 = std::exp(x) - 1.0f;
    // 断言两者实部和虚部接近
    C10_ASSERT_NEAR(l1.real(), l2.real(), tol);
    C10_ASSERT_NEAR(l1.imag(), l2.imag(), tol);
  }
  {
    // 使用双精度浮点数进行测试
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> l1 = std::expm1(x);
    c10::complex<double> l2 = std::exp(x) - 1.0;
    C10_ASSERT_NEAR(l1.real(), l2.real(), tol);
    C10_ASSERT_NEAR(l1.imag(), l2.imag(), tol);
  }
}
C10_DEFINE_TEST(TestExpm1, Small) {
  // expm1(x) = exp(x) - 1
  // expm1(x)提供比exp(x) - 1在x接近0时更高的精度

  {
    c10::complex<float> x(1e-30, 1e-30);
    // 计算复数expm1(x)
    c10::complex<float> l1 = std::expm1(x);
    // 断言实部接近1e-30，使用tol作为容忍度
    C10_ASSERT_NEAR(l1.real(), 1e-30, tol);
    // 断言虚部接近1e-30，使用tol作为容忍度
    C10_ASSERT_NEAR(l1.imag(), 1e-30, tol);
  }

  {
    c10::complex<double> x(1e-100, 1e-100);
    // 计算复数expm1(x)
    c10::complex<double> l1 = std::expm1(x);
    // 断言实部接近1e-30，使用tol作为容忍度
    C10_ASSERT_NEAR(l1.real(), 1e-30, tol);
    // 断言虚部接近1e-30，使用tol作为容忍度
    C10_ASSERT_NEAR(l1.imag(), 1e-30, tol);
  }
}

C10_DEFINE_TEST(TestLog, Definition) {
  // log(x) = log(r) + i*theta
  // log(x)表示对数函数，其中x = r * e^(i*theta)

  {
    c10::complex<float> x(1.2, 3.4);
    // 计算复数log(x)
    c10::complex<float> l = std::log(x);
    // 预期的实部是log(x的模)
    float expected_real = std::log(std::abs(x));
    // 预期的虚部是x的辐角
    float expected_imag = std::arg(x);
    // 断言实部接近预期值，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), expected_real, tol);
    // 断言虚部接近预期值，使用tol作为容忍度
    C10_ASSERT_NEAR(l.imag(), expected_imag, tol);
  }

  {
    c10::complex<float> x(1.2, 3.4);
    // 计算复数log(x)
    c10::complex<float> l = ::log(x);
    // 预期的实部是log(x的模)
    float expected_real = ::log(std::abs(x));
    // 预期的虚部是x的辐角
    float expected_imag = std::arg(x);
    // 断言实部接近预期值，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), expected_real, tol);
    // 断言虚部接近预期值，使用tol作为容忍度
    C10_ASSERT_NEAR(l.imag(), expected_imag, tol);
  }

  {
    c10::complex<double> x(1.2, 3.4);
    // 计算复数log(x)
    c10::complex<double> l = std::log(x);
    // 预期的实部是log(x的模)
    float expected_real = std::log(std::abs(x));
    // 预期的虚部是x的辐角
    float expected_imag = std::arg(x);
    // 断言实部接近预期值，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), expected_real, tol);
    // 断言虚部接近预期值，使用tol作为容忍度
    C10_ASSERT_NEAR(l.imag(), expected_imag, tol);
  }

  {
    c10::complex<double> x(1.2, 3.4);
    // 计算复数log(x)
    c10::complex<double> l = ::log(x);
    // 预期的实部是log(x的模)
    float expected_real = ::log(std::abs(x));
    // 预期的虚部是x的辐角
    float expected_imag = std::arg(x);
    // 断言实部接近预期值，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), expected_real, tol);
    // 断言虚部接近预期值，使用tol作为容忍度
    C10_ASSERT_NEAR(l.imag(), expected_imag, tol);
  }
}

C10_DEFINE_TEST(TestLog10, Rev) {
  // log10(10^x) = x
  // log10函数的反函数是指数函数

  {
    c10::complex<float> x(0.1, 1.2);
    // 计算复数log10(10^x)
    c10::complex<float> l = std::log10(std::pow(float(10), x));
    // 断言实部接近0.1，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), float(0.1), tol);
    // 断言虚部接近1.2，使用tol作为容忍度
    C10_ASSERT_NEAR(l.imag(), float(1.2), tol);
  }

  {
    c10::complex<float> x(0.1, 1.2);
    // 计算复数log10(10^x)
    c10::complex<float> l = ::log10(::pow(float(10), x));
    // 断言实部接近0.1，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), float(0.1), tol);
    // 断言虚部接近1.2，使用tol作为容忍度
    C10_ASSERT_NEAR(l.imag(), float(1.2), tol);
  }

  {
    c10::complex<double> x(0.1, 1.2);
    // 计算复数log10(10^x)
    c10::complex<double> l = std::log10(std::pow(double(10), x));
    // 断言实部接近0.1，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), double(0.1), tol);
    // 断言虚部接近1.2，使用tol作为容忍度
    C10_ASSERT_NEAR(l.imag(), double(1.2), tol);
  }

  {
    c10::complex<double> x(0.1, 1.2);
    // 计算复数log10(10^x)
    c10::complex<double> l = ::log10(::pow(double(10), x));
    // 断言实部接近0.1，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), double(0.1), tol);
    // 断言虚部接近1.2，使用tol作为容忍度
    C10_ASSERT_NEAR(l.imag(), double(1.2), tol);
  }
}

C10_DEFINE_TEST(TestLog2, Rev) {
  // log2(2^x) = x
  // log2函数的反函数是指数函数

  {
    c10::complex<float> x(0.1, 1.2);
    // 计算复数log2(2^x)
    c10::complex<float> l = std::log2(std::pow(float(2), x));
    // 断言实部接近0.1，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), float(0.1), tol);
    // 断言虚部接近1.2，使用tol作为容忍度
    C10_ASSERT_NEAR(l.imag(), float(1.2), tol);
  }

  {
    c10::complex<float> x(0.1, 1.2);
    // 计算复数log2(2^x)
    c10::complex<float> l = ::log2(std::pow(float(2), x));
    // 断言实部接近0.1，使用tol作为容忍度
    C10_ASSERT_NEAR(l.real(), float(0.1), tol);
  {
    // 创建一个复数对象 x，实部为 0.1，虚部为 1.2
    c10::complex<double> x(0.1, 1.2);
    // 计算以 2 为底 x 的幂的对数，并赋值给 l
    c10::complex<double> l = std::log2(std::pow(double(2), x));
    // 断言 l 的实部接近于 0.1，使用给定的容差 tol
    C10_ASSERT_NEAR(l.real(), double(0.1), tol);
    // 断言 l 的虚部接近于 1.2，使用给定的容差 tol
    C10_ASSERT_NEAR(l.imag(), double(1.2), tol);
  }
  {
    // 创建一个复数对象 x，实部为 0.1，虚部为 1.2
    c10::complex<double> x(0.1, 1.2);
    // 计算以 2 为底 x 的幂的对数，并赋值给 l
    c10::complex<double> l = ::log2(std::pow(double(2), x));
    // 断言 l 的实部接近于 0.1，使用给定的容差 tol
    C10_ASSERT_NEAR(l.real(), double(0.1), tol);
    // 断言 l 的虚部接近于 1.2，使用给定的容差 tol
    C10_ASSERT_NEAR(l.imag(), double(1.2), tol);
  }
}

C10_DEFINE_TEST(TestLog1p, Normal) {
  // 定义一个测试用例 TestLog1p，测试 log1p 函数的正常情况
  {
    // 创建一个复数对象 x，实部为 0.1，虚部为 1.2
    c10::complex<float> x(0.1, 1.2);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<float> l1 = std::log1p(x);
    // 使用 std::log(1.0f + x) 计算 log(1 + x)
    c10::complex<float> l2 = std::log(1.0f + x);
    // 断言 l1 的实部接近于 l2 的实部，使用容忍度 tol
    C10_ASSERT_NEAR(l1.real(), l2.real(), tol);
    // 断言 l1 的虚部接近于 l2 的虚部，使用容忍度 tol
    C10_ASSERT_NEAR(l1.imag(), l2.imag(), tol);
  }
  {
    // 创建一个双精度复数对象 x，实部为 0.1，虚部为 1.2
    c10::complex<double> x(0.1, 1.2);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<double> l1 = std::log1p(x);
    // 使用 std::log(1.0 + x) 计算 log(1 + x)
    c10::complex<double> l2 = std::log(1.0 + x);
    // 断言 l1 的实部接近于 l2 的实部，使用容忍度 tol
    C10_ASSERT_NEAR(l1.real(), l2.real(), tol);
    // 断言 l1 的虚部接近于 l2 的虚部，使用容忍度 tol
    C10_ASSERT_NEAR(l1.imag(), l2.imag(), tol);
  }
}

C10_DEFINE_TEST(TestLog1p, Small) {
  // 定义一个测试用例 TestLog1p，测试 log1p 函数在小数值下的行为
  {
    // 创建一个浮点数复数对象 x，实部为 1e-9，虚部为 2e-9
    c10::complex<float> x(1e-9, 2e-9);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<float> l = std::log1p(x);
    // 断言 l 的实部除以 x 的实部接近于 1，使用容忍度 tol
    C10_ASSERT_NEAR(l.real() / x.real(), 1, tol);
    // 断言 l 的虚部除以 x 的虚部接近于 1，使用容忍度 tol
    C10_ASSERT_NEAR(l.imag() / x.imag(), 1, tol);
  }
  {
    // 创建一个双精度浮点数复数对象 x，实部为 1e-100，虚部为 2e-100
    c10::complex<double> x(1e-100, 2e-100);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<double> l = std::log1p(x);
    // 断言 l 的实部除以 x 的实部接近于 1，使用容忍度 tol
    C10_ASSERT_NEAR(l.real() / x.real(), 1, tol);
    // 断言 l 的虚部除以 x 的虚部接近于 1，使用容忍度 tol
    C10_ASSERT_NEAR(l.imag() / x.imag(), 1, tol);
  }
}

C10_DEFINE_TEST(TestLog1p, Extreme) {
  // 定义一个测试用例 TestLog1p，测试 log1p 函数在极端情况下的行为
  {
    // 创建一个浮点数复数对象 x，实部为 -1，虚部为 1e-30
    c10::complex<float> x(-1, 1e-30);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<float> l = std::log1p(x);
    // 断言 l 的实部接近于 -69.07755278982137，使用容忍度 tol
    C10_ASSERT_NEAR(l.real(), -69.07755278982137, tol);
    // 断言 l 的虚部接近于 1.5707963267948966，使用容忍度 tol
    C10_ASSERT_NEAR(l.imag(), 1.5707963267948966, tol);
  }
  {
    // 创建一个浮点数复数对象 x，实部为 -1，虚部为 1e30
    c10::complex<float> x(-1, 1e30);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<float> l = std::log1p(x);
    // 断言 l 的实部接近于 69.07755278982137，使用容忍度 tol
    C10_ASSERT_NEAR(l.real(), 69.07755278982137, tol);
    // 断言 l 的虚部接近于 1.5707963267948966，使用容忍度 tol
    C10_ASSERT_NEAR(l.imag(), 1.5707963267948966, tol);
  }
  {
    // 创建一个浮点数复数对象 x，实部为 1e30，虚部为 1
    c10::complex<float> x(1e30, 1);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<float> l = std::log1p(x);
    // 断言 l 的实部接近于 69.07755278982137，使用容忍度 tol
    C10_ASSERT_NEAR(l.real(), 69.07755278982137, tol);
    // 断言 l 的虚部接近于 1e-30，使用容忍度 tol
    C10_ASSERT_NEAR(l.imag(), 1e-30, tol);
  }
  {
    // 创建一个浮点数复数对象 x，实部为 1e-30，虚部为 1
    c10::complex<float> x(1e-30, 1);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<float> l = std::log1p(x);
    // 断言 l 的实部接近于 0.34657359027997264，使用容忍度 tol
    C10_ASSERT_NEAR(l.real(), 0.34657359027997264, tol);
    // 断言 l 的虚部接近于 0.7853981633974483，使用容忍度 tol
    C10_ASSERT_NEAR(l.imag(), 0.7853981633974483, tol);
  }
  {
    // 创建一个浮点数复数对象 x，实部为 1e30，虚部为 1e30
    c10::complex<float> x(1e30, 1e30);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<float> l = std::log1p(x);
    // 断言 l 的实部接近于 69.42412638010134，使用容忍度 tol
    C10_ASSERT_NEAR(l.real(), 69.42412638010134, tol);
    // 断言 l 的虚部接近于 0.7853981633974483，使用容忍度 tol
    C10_ASSERT_NEAR(l.imag(), 0.7853981633974483, tol);
  }
  {
    // 创建一个浮点数复数对象 x，实部为 1e-38，虚部为 1e-38
    c10::complex<float> x(1e-38, 1e-38);
    // 使用 std::log1p 计算 log1p(x)
    c10::complex<float> l = std::log1p(x);
    // 断言 l 的实部接近于 1e-38，使用容忍度 tol
    C10_ASSERT_NEAR(l.real(), 1e-38, tol);
    // 断言 l 的虚部接近于 1e-38，使用容忍度 tol
    C10_ASSERT_NEAR(l.imag(), 1e-38, tol);
  }
  {
    // 创建一个浮点数复数对象 x，实部为 1e-38，虚部为 2e-30
    c10::complex<float> x(1e-38, 2e-30);
    // 使用 std::log1p 计算 log1p(x)
    c10::
  {
    // 创建一个复数对象 x，实部为 0.0，虚部为 1.0e-250
    c10::complex<double> x(0.0, 1.0e-250);
    // 对 x 进行 log1p 计算，得到结果复数 l
    c10::complex<double> l = std::log1p(x);
    // 断言 l 的实部接近于 0.34657359027997264，使用给定的容差 tol
    C10_ASSERT_NEAR(l.real(), 0.34657359027997264, tol);
    // 断言 l 的虚部接近于 1.0e-250，使用给定的容差 tol
    C10_ASSERT_NEAR(l.imag(), 1.0e-250, tol);
  }
  {
    // 创建一个复数对象 x，实部为 1.0e-250，虚部为 1.0
    c10::complex<double> x(1.0e-250, 1.0);
    // 对 x 进行 log1p 计算，得到结果复数 l
    c10::complex<double> l = std::log1p(x);
    // 断言 l 的实部接近于 0.0，使用给定的容差 tol
    C10_ASSERT_NEAR(l.real(), 0.0, tol);
    // 断言 l 的虚部接近于 1.5707963267948966（即 π/2），使用给定的容差 tol
    C10_ASSERT_NEAR(l.imag(), 1.5707963267948966, tol);
  }
  {
    // 创建一个复数对象 x，实部和虚部均为 1.0e250
    c10::complex<double> x(1.0e250, 1.0e250);
    // 对 x 进行 log1p 计算，得到结果复数 l
    c10::complex<double> l = std::log1p(x);
    // 断言 l 的实部接近于 575.9928468387914，使用给定的容差 tol
    C10_ASSERT_NEAR(l.real(), 575.9928468387914, tol);
    // 断言 l 的虚部接近于 0.7853981633974483（即 π/4），使用给定的容差 tol
    C10_ASSERT_NEAR(l.imag(), 0.7853981633974483, tol);
  }
  {
    // 创建一个复数对象 x，实部和虚部均为 1.0e-250
    c10::complex<double> x(1.0e-250, 1.0e-250);
    // 对 x 进行 log1p 计算，得到结果复数 l
    c10::complex<double> l = std::log1p(x);
    // 断言 l 的实部接近于 1.0e-250，使用给定的容差 tol
    C10_ASSERT_NEAR(l.real(), 1.0e-250, tol);
    // 断言 l 的虚部接近于 1.0e-250，使用给定的容差 tol
    C10_ASSERT_NEAR(l.imag(), 1.0e-250, tol);
  }
  {
    // 创建一个复数对象 x，实部为 1.0e-250，虚部为 2.0e-250
    c10::complex<double> x(1.0e-250, 2.0e-250);
    // 对 x 进行 log1p 计算，得到结果复数 l
    c10::complex<double> l = std::log1p(x);
    // 断言 l 的实部接近于 1.0e-250，使用给定的容差 tol
    C10_ASSERT_NEAR(l.real(), 1.0e-250, tol);
    // 断言 l 的虚部接近于 2.0e-250，使用给定的容差 tol
    C10_ASSERT_NEAR(l.imag(), 2.0e-250, tol);
  }
  {
    // 创建一个复数对象 x，实部为 2.0e-308，虚部为 1.5e-250
    c10::complex<double> x(2.0e-308, 1.5e-250);
    // 对 x 进行 log1p 计算，得到结果复数 l
    c10::complex<double> l = std::log1p(x);
    // 断言 l 的实部接近于 2.0e-308，使用给定的容差 tol
    C10_ASSERT_NEAR(l.real(), 2.0e-308, tol);
    // 断言 l 的虚部接近于 1.5e-308，使用给定的容差 tol
    C10_ASSERT_NEAR(l.imag(), 1.5e-308, tol);
  }
cpp
}

// Power functions

C10_DEFINE_TEST(TestPowSqrt, Equal) {
  // Test for square root calculation using both std::pow and std::sqrt
  {
    // Define a complex number x
    c10::complex<float> x(0.1, 1.2);
    // Calculate x^(1/2) using std::pow
    c10::complex<float> y = std::pow(x, float(0.5));
    // Calculate sqrt(x)
    c10::complex<float> z = std::sqrt(x);
    // Assert that real and imaginary parts are nearly equal within tolerance
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    // Repeat the above test using ::pow and ::sqrt
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = ::pow(x, float(0.5));
    c10::complex<float> z = ::sqrt(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    // Test for double precision using std::pow and std::sqrt
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = std::pow(x, double(0.5));
    c10::complex<double> z = std::sqrt(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    // Repeat the double precision test using ::pow and ::sqrt
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = ::pow(x, double(0.5));
    c10::complex<double> z = ::sqrt(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

C10_DEFINE_TEST(TestPow, Square) {
  // Test for squaring a complex number using std::pow and multiplication
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = std::pow(x, float(2));
    c10::complex<float> z = x * x;
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    // Repeat the square test using ::pow and multiplication
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = ::pow(x, float(2));
    c10::complex<float> z = x * x;
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    // Test for double precision using std::pow and multiplication
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = std::pow(x, double(2));
    c10::complex<double> z = x * x;
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    // Repeat the double precision test using ::pow and multiplication
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = ::pow(x, double(2));
    c10::complex<double> z = x * x;
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

// Trigonometric functions and hyperbolic functions

C10_DEFINE_TEST(TestSinCosSinhCosh, Identity) {
  // Test for sine and cosine identities involving complex numbers
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = std::sin(x);
    // Calculate expected values based on identities
    float expected_real = std::sin(x.real()) * std::cosh(x.imag());
    float expected_imag = std::cos(x.real()) * std::sinh(x.imag());
    // Assert the real and imaginary parts match within tolerance
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    // Repeat the sine test using ::sin
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = ::sin(x);
    float expected_real = ::sin(x.real()) * ::cosh(x.imag());
    float expected_imag = ::cos(x.real()) * ::sinh(x.imag());
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    // Test for cosine using std::cos
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = std::cos(x);
    float expected_real = std::cos(x.real()) * std::cosh(x.imag());
    # 计算预期的虚部值，使用标准库中的正弦和双曲正弦函数
    float expected_imag = -std::sin(x.real()) * std::sinh(x.imag());
    # 断言实部的值与预期值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    # 断言虚部的值与预期的虚部值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    # 创建一个复数对象 x
    c10::complex<float> x(0.1, 1.2);
    # 对 x 求余弦，并将结果赋给 y
    c10::complex<float> y = ::cos(x);
    # 计算预期的实部值，使用标准库中的余弦和双曲余弦函数
    float expected_real = ::cos(x.real()) * ::cosh(x.imag());
    # 计算预期的虚部值，使用标准库中的正弦和双曲正弦函数
    float expected_imag = -::sin(x.real()) * ::sinh(x.imag());
    # 断言实部的值与预期值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    # 断言虚部的值与预期的虚部值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    # 创建一个复数对象 x
    c10::complex<double> x(0.1, 1.2);
    # 对 x 求正弦，并将结果赋给 y
    c10::complex<double> y = std::sin(x);
    # 计算预期的实部值，使用标准库中的正弦和双曲余弦函数
    float expected_real = std::sin(x.real()) * std::cosh(x.imag());
    # 计算预期的虚部值，使用标准库中的余弦和双曲正弦函数
    float expected_imag = std::cos(x.real()) * std::sinh(x.imag());
    # 断言实部的值与预期值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    # 断言虚部的值与预期的虚部值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    # 创建一个复数对象 x
    c10::complex<double> x(0.1, 1.2);
    # 对 x 求正弦，并将结果赋给 y
    c10::complex<double> y = ::sin(x);
    # 计算预期的实部值，使用标准库中的正弦和双曲余弦函数
    float expected_real = ::sin(x.real()) * ::cosh(x.imag());
    # 计算预期的虚部值，使用标准库中的余弦和双曲正弦函数
    float expected_imag = ::cos(x.real()) * ::sinh(x.imag());
    # 断言实部的值与预期值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    # 断言虚部的值与预期的虚部值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    # 创建一个复数对象 x
    c10::complex<double> x(0.1, 1.2);
    # 对 x 求余弦，并将结果赋给 y
    c10::complex<double> y = std::cos(x);
    # 计算预期的实部值，使用标准库中的余弦和双曲余弦函数
    float expected_real = std::cos(x.real()) * std::cosh(x.imag());
    # 计算预期的虚部值，使用标准库中的正弦和双曲正弦函数
    float expected_imag = -std::sin(x.real()) * std::sinh(x.imag());
    # 断言实部的值与预期值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    # 断言虚部的值与预期的虚部值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    # 创建一个复数对象 x
    c10::complex<double> x(0.1, 1.2);
    # 对 x 求余弦，并将结果赋给 y
    c10::complex<double> y = ::cos(x);
    # 计算预期的实部值，使用标准库中的余弦和双曲余弦函数
    float expected_real = ::cos(x.real()) * ::cosh(x.imag());
    # 计算预期的虚部值，使用标准库中的正弦和双曲正弦函数
    float expected_imag = -::sin(x.real()) * ::sinh(x.imag());
    # 断言实部的值与预期值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    # 断言虚部的值与预期的虚部值在给定的容差范围内接近
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
C10_DEFINE_TEST(TestTan, Identity) {
  // 定义测试函数 TestTan，测试 tan 函数的身份性质
  // tan(x) = sin(x) / cos(x)

  {
    // 定义复数 x，并初始化为 (0.1, 1.2)
    c10::complex<float> x(0.1, 1.2);
    // 计算 std::tan(x) 的值
    c10::complex<float> y = std::tan(x);
    // 计算 sin(x) / cos(x) 的值
    c10::complex<float> z = std::sin(x) / std::cos(x);
    // 断言 y 的实部接近于 z 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    // 断言 y 的虚部接近于 z 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }

  {
    // 定义复数 x，并初始化为 (0.1, 1.2)
    c10::complex<float> x(0.1, 1.2);
    // 计算 ::tan(x) 的值
    c10::complex<float> y = ::tan(x);
    // 计算 ::sin(x) / ::cos(x) 的值
    c10::complex<float> z = ::sin(x) / ::cos(x);
    // 断言 y 的实部接近于 z 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    // 断言 y 的虚部接近于 z 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }

  {
    // 定义复数 x，并初始化为 (0.1, 1.2)
    c10::complex<double> x(0.1, 1.2);
    // 计算 std::tan(x) 的值
    c10::complex<double> y = std::tan(x);
    // 计算 std::sin(x) / std::cos(x) 的值
    c10::complex<double> z = std::sin(x) / std::cos(x);
    // 断言 y 的实部接近于 z 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    // 断言 y 的虚部接近于 z 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }

  {
    // 定义复数 x，并初始化为 (0.1, 1.2)
    c10::complex<double> x(0.1, 1.2);
    // 计算 ::tan(x) 的值
    c10::complex<double> y = ::tan(x);
    // 计算 ::sin(x) / ::cos(x) 的值
    c10::complex<double> z = ::sin(x) / ::cos(x);
    // 断言 y 的实部接近于 z 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    // 断言 y 的虚部接近于 z 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

C10_DEFINE_TEST(TestTanh, Identity) {
  // 定义测试函数 TestTanh，测试 tanh 函数的身份性质
  // tanh(x) = sinh(x) / cosh(x)

  {
    // 定义复数 x，并初始化为 (0.1, 1.2)
    c10::complex<float> x(0.1, 1.2);
    // 计算 std::tanh(x) 的值
    c10::complex<float> y = std::tanh(x);
    // 计算 std::sinh(x) / std::cosh(x) 的值
    c10::complex<float> z = std::sinh(x) / std::cosh(x);
    // 断言 y 的实部接近于 z 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    // 断言 y 的虚部接近于 z 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }

  {
    // 定义复数 x，并初始化为 (0.1, 1.2)
    c10::complex<float> x(0.1, 1.2);
    // 计算 ::tanh(x) 的值
    c10::complex<float> y = ::tanh(x);
    // 计算 ::sinh(x) / ::cosh(x) 的值
    c10::complex<float> z = ::sinh(x) / ::cosh(x);
    // 断言 y 的实部接近于 z 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    // 断言 y 的虚部接近于 z 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }

  {
    // 定义复数 x，并初始化为 (0.1, 1.2)
    c10::complex<double> x(0.1, 1.2);
    // 计算 std::tanh(x) 的值
    c10::complex<double> y = std::tanh(x);
    // 计算 std::sinh(x) / std::cosh(x) 的值
    c10::complex<double> z = std::sinh(x) / std::cosh(x);
    // 断言 y 的实部接近于 z 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    // 断言 y 的虚部接近于 z 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }

  {
    // 定义复数 x，并初始化为 (0.1, 1.2)
    c10::complex<double> x(0.1, 1.2);
    // 计算 ::tanh(x) 的值
    c10::complex<double> y = ::tanh(x);
    // 计算 ::sinh(x) / ::cosh(x) 的值
    c10::complex<double> z = ::sinh(x) / ::cosh(x);
    // 断言 y 的实部接近于 z 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    // 断言 y 的虚部接近于 z 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

// Rev trigonometric functions

C10_DEFINE_TEST(TestRevTrigonometric, Rev) {
  // 定义测试函数 TestRevTrigonometric，测试反三角函数的反函数性质
  // asin(sin(x)) = x
  // acos(cos(x)) = x
  // atan(tan(x)) = x

  {
    // 定义复数 x，并初始化为 (0.5, 0.6)
    c10::complex<float> x(0.5, 0.6);
    // 计算 sin(x) 的值
    c10::complex<float> s = std::sin(x);
    // 计算 asin(sin(x)) 的值
    c10::complex<float> ss = std::asin(s);
    // 计算 cos(x) 的值
    c10::complex<float> c = std::cos(x);
    // 计算 acos(cos(x)) 的值
    c10::complex<float> cc = std::acos(c);
    // 计算 tan(x) 的值
    c10::complex<float> t = std::tan(x);
    // 计算 atan(tan(x)) 的值
    c10::complex<float> tt = std::atan(t);
    // 断言 x 的实部接近于 ss 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    // 断言 x 的虚部接近于 ss 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    // 断言 x 的实部接近于 cc 的实部，使用给定的公差 tol
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    // 断言 x 的虚部接近于 cc 的虚部，使用给定的公差 tol
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    // 断言 x 的实部接近
    # 断言 x 的实部近似于 ss 的实部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    # 断言 x 的虚部近似于 ss 的虚部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    # 断言 x 的实部近似于 cc 的实部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    # 断言 x 的虚部近似于 cc 的虚部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    # 断言 x 的实部近似于 tt 的实部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    # 断言 x 的虚部近似于 tt 的虚部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    # 创建复数 x，具有给定的实部和虚部
    c10::complex<double> x(0.5, 0.6);
    # 计算复数 x 的正弦值并赋给 s
    c10::complex<double> s = std::sin(x);
    # 计算复数 s 的反正弦值并赋给 ss
    c10::complex<double> ss = std::asin(s);
    # 计算复数 x 的余弦值并赋给 c
    c10::complex<double> c = std::cos(x);
    # 计算复数 c 的反余弦值并赋给 cc
    c10::complex<double> cc = std::acos(c);
    # 计算复数 x 的正切值并赋给 t
    c10::complex<double> t = std::tan(x);
    # 计算复数 t 的反正切值并赋给 tt
    c10::complex<double> tt = std::atan(t);
    # 断言 x 的实部近似于 ss 的实部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    # 断言 x 的虚部近似于 ss 的虚部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    # 断言 x 的实部近似于 cc 的实部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    # 断言 x 的虚部近似于 cc 的虚部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    # 断言 x 的实部近似于 tt 的实部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    # 断言 x 的虚部近似于 tt 的虚部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    # 创建复数 x，具有给定的实部和虚部
    c10::complex<double> x(0.5, 0.6);
    # 计算复数 x 的正弦值并赋给 s
    c10::complex<double> s = ::sin(x);
    # 计算复数 s 的反正弦值并赋给 ss
    c10::complex<double> ss = ::asin(s);
    # 计算复数 x 的余弦值并赋给 c
    c10::complex<double> c = ::cos(x);
    # 计算复数 c 的反余弦值并赋给 cc
    c10::complex<double> cc = ::acos(c);
    # 计算复数 x 的正切值并赋给 t
    c10::complex<double> t = ::tan(x);
    # 计算复数 t 的反正切值并赋给 tt
    c10::complex<double> tt = ::atan(t);
    # 断言 x 的实部近似于 ss 的实部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    # 断言 x 的虚部近似于 ss 的虚部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    # 断言 x 的实部近似于 cc 的实部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    # 断言 x 的虚部近似于 cc 的虚部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    # 断言 x 的实部近似于 tt 的实部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    # 断言 x 的虚部近似于 tt 的虚部，使用给定的容差 tol
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
// 定义一个测试函数 TestRevHyperbolic，用于测试反双曲函数
C10_DEFINE_TEST(TestRevHyperbolic, Rev) {
  // 测试 asinh(sinh(x)) = x
  // 测试 acosh(cosh(x)) = x
  // 测试 atanh(tanh(x)) = x
  {
    // 定义一个复数 x，实部为 0.5，虚部为 0.6
    c10::complex<float> x(0.5, 0.6);
    // 计算 sinh(x)
    c10::complex<float> s = std::sinh(x);
    // 计算 asinh(sinh(x))
    c10::complex<float> ss = std::asinh(s);
    // 计算 cosh(x)
    c10::complex<float> c = std::cosh(x);
    // 计算 acosh(cosh(x))
    c10::complex<float> cc = std::acosh(c);
    // 计算 tanh(x)
    c10::complex<float> t = std::tanh(x);
    // 计算 atanh(tanh(x))
    c10::complex<float> tt = std::atanh(t);
    // 断言：检查实部和虚部是否与 x 相近，tol 是容差
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    // 同样的测试，但使用全局命名空间下的函数 ::sinh、::asinh、::cosh、::acosh、::tanh、::atanh
    c10::complex<float> x(0.5, 0.6);
    c10::complex<float> s = ::sinh(x);
    c10::complex<float> ss = ::asinh(s);
    c10::complex<float> c = ::cosh(x);
    c10::complex<float> cc = ::acosh(c);
    c10::complex<float> t = ::tanh(x);
    c10::complex<float> tt = ::atanh(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    // 使用双精度复数进行相同的测试
    c10::complex<double> x(0.5, 0.6);
    c10::complex<double> s = std::sinh(x);
    c10::complex<double> ss = std::asinh(s);
    c10::complex<double> c = std::cosh(x);
    c10::complex<double> cc = std::acosh(c);
    c10::complex<double> t = std::tanh(x);
    c10::complex<double> tt = std::atanh(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    // 使用双精度复数和全局命名空间下的函数进行相同的测试
    c10::complex<double> x(0.5, 0.6);
    c10::complex<double> s = ::sinh(x);
    c10::complex<double> ss = ::asinh(s);
    c10::complex<double> c = ::cosh(x);
    c10::complex<double> cc = ::acosh(c);
    c10::complex<double> t = ::tanh(x);
    c10::complex<double> tt = ::atanh(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
}
```