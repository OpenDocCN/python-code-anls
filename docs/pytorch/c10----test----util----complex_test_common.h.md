# `.\pytorch\c10\test\util\complex_test_common.h`

```py
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <c10/util/hash.h>
#include <gtest/gtest.h>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#if (defined(__CUDACC__) || defined(__HIPCC__))
// 如果在 CUDA 或 HIP 编译环境下，定义为 __global__
#define MAYBE_GLOBAL __global__
#else
// 否则为空定义
#define MAYBE_GLOBAL
#endif

// 定义常数 PI
#define PI 3.141592653589793238463

namespace memory {

// 可能的全局函数声明，用于测试大小
MAYBE_GLOBAL void test_size() {
  // 静态断言：c10::complex<float> 的大小应为两个 float 的大小
  static_assert(sizeof(c10::complex<float>) == 2 * sizeof(float), "");
  // 静态断言：c10::complex<double> 的大小应为两个 double 的大小
  static_assert(sizeof(c10::complex<double>) == 2 * sizeof(double), "");
}

// 可能的全局函数声明，用于测试对齐方式
MAYBE_GLOBAL void test_align() {
  // 静态断言：c10::complex<float> 的对齐方式应为两个 float 的大小
  static_assert(alignof(c10::complex<float>) == 2 * sizeof(float), "");
  // 静态断言：c10::complex<double> 的对齐方式应为两个 double 的大小
  static_assert(alignof(c10::complex<double>) == 2 * sizeof(double), "");
}

// 可能的全局函数声明，用于测试是否是标准布局
MAYBE_GLOBAL void test_pod() {
  // 静态断言：c10::complex<float> 应为标准布局
  static_assert(std::is_standard_layout<c10::complex<float>>::value, "");
  // 静态断言：c10::complex<double> 应为标准布局
  static_assert(std::is_standard_layout<c10::complex<double>>::value, "");
}

// 内存模块的测试用例，测试 reinterpret_cast 转换
TEST(TestMemory, ReinterpretCast) {
  {
    // 创建 std::complex<float> 对象 z，值为 (1, 2)
    std::complex<float> z(1, 2);
    // 通过 reinterpret_cast 将其转换为 c10::complex<float>
    c10::complex<float> zz = *reinterpret_cast<c10::complex<float>*>(&z);
    // 断言：转换后的实部应为 1
    ASSERT_EQ(zz.real(), float(1));
    // 断言：转换后的虚部应为 2
    ASSERT_EQ(zz.imag(), float(2));
  }

  {
    // 创建 c10::complex<float> 对象 z，值为 (3, 4)
    c10::complex<float> z(3, 4);
    // 通过 reinterpret_cast 将其转换为 std::complex<float>
    std::complex<float> zz = *reinterpret_cast<std::complex<float>*>(&z);
    // 断言：转换后的实部应为 3
    ASSERT_EQ(zz.real(), float(3));
    // 断言：转换后的虚部应为 4
    ASSERT_EQ(zz.imag(), float(4));
  }

  {
    // 创建 std::complex<double> 对象 z，值为 (1, 2)
    std::complex<double> z(1, 2);
    // 通过 reinterpret_cast 将其转换为 c10::complex<double>
    c10::complex<double> zz = *reinterpret_cast<c10::complex<double>*>(&z);
    // 断言：转换后的实部应为 1
    ASSERT_EQ(zz.real(), double(1));
    // 断言：转换后的虚部应为 2
    ASSERT_EQ(zz.imag(), double(2));
  }

  {
    // 创建 c10::complex<double> 对象 z，值为 (3, 4)
    c10::complex<double> z(3, 4);
    // 通过 reinterpret_cast 将其转换为 std::complex<double>
    std::complex<double> zz = *reinterpret_cast<std::complex<double>*>(&z);
    // 断言：转换后的实部应为 3
    ASSERT_EQ(zz.real(), double(3));
    // 断言：转换后的虚部应为 4
    ASSERT_EQ(zz.imag(), double(4));
  }
}

#if defined(__CUDACC__) || defined(__HIPCC__)
// 在 CUDA 或 HIP 编译环境下的测试用例，测试 Thrust reinterpret_cast 转换
TEST(TestMemory, ThrustReinterpretCast) {
  {
    // 创建 thrust::complex<float> 对象 z，值为 (1, 2)
    thrust::complex<float> z(1, 2);
    // 通过 reinterpret_cast 将其转换为 c10::complex<float>
    c10::complex<float> zz = *reinterpret_cast<c10::complex<float>*>(&z);
    // 断言：转换后的实部应为 1
    ASSERT_EQ(zz.real(), float(1));
    // 断言：转换后的虚部应为 2
    ASSERT_EQ(zz.imag(), float(2));
  }

  {
    // 创建 c10::complex<float> 对象 z，值为 (3, 4)
    c10::complex<float> z(3, 4);
    // 通过 reinterpret_cast 将其转换为 thrust::complex<float>
    thrust::complex<float> zz = *reinterpret_cast<thrust::complex<float>*>(&z);
    // 断言：转换后的实部应为 3
    ASSERT_EQ(zz.real(), float(3));
    // 断言：转换后的虚部应为 4
    ASSERT_EQ(zz.imag(), float(4));
  }

  {
    // 创建 thrust::complex<double> 对象 z，值为 (1, 2)
    thrust::complex<double> z(1, 2);
    // 通过 reinterpret_cast 将其转换为 c10::complex<double>
    c10::complex<double> zz = *reinterpret_cast<c10::complex<double>*>(&z);
    // 断言：转换后的实部应为 1
    ASSERT_EQ(zz.real(), double(1));
    // 断言：转换后的虚部应为 2
    ASSERT_EQ(zz.imag(), double(2));
  }

  {
    // 创建 c10::complex<double> 对象 z，值为 (3, 4)
    c10::complex<double> z(3, 4);
    // 通过 reinterpret_cast 将其转换为 thrust::complex<double>
    thrust::complex<double> zz =
        *reinterpret_cast<thrust::complex<double>*>(&z);
    // 断言：转换后的实部应为 3
    ASSERT_EQ(zz.real(), double(3));
    // 断言：转换后的虚部应为 4
    ASSERT_EQ(zz.imag(), double(4));
  }
}
#endif

} // namespace memory

namespace constructors {

template <typename scalar_t>
C10_HOST_DEVICE void test_construct_from_scalar() {
  // 定义并初始化三个标量值，其中第三个为默认构造的零值
  constexpr scalar_t num1 = scalar_t(1.23);
  constexpr scalar_t num2 = scalar_t(4.56);
  constexpr scalar_t zero = scalar_t();
  // 使用静态断言验证复数对象的实部与给定的 num1 是否相等
  static_assert(c10::complex<scalar_t>(num1, num2).real() == num1, "");
  // 使用静态断言验证复数对象的虚部与给定的 num2 是否相等
  static_assert(c10::complex<scalar_t>(num1, num2).imag() == num2, "");
  // 使用静态断言验证只给定实部时，复数对象的实部与给定的 num1 是否相等
  static_assert(c10::complex<scalar_t>(num1).real() == num1, "");
  // 使用静态断言验证只给定实部时，复数对象的虚部为默认构造的零值
  static_assert(c10::complex<scalar_t>(num1).imag() == zero, "");
  // 使用静态断言验证默认构造的复数对象的实部为默认构造的零值
  static_assert(c10::complex<scalar_t>().real() == zero, "");
  // 使用静态断言验证默认构造的复数对象的虚部为默认构造的零值
  static_assert(c10::complex<scalar_t>().imag() == zero, "");
}

template <typename scalar_t, typename other_t>
C10_HOST_DEVICE void test_construct_from_other() {
  // 定义并初始化两个 other_t 类型的标量值
  constexpr other_t num1 = other_t(1.23);
  constexpr other_t num2 = other_t(4.56);
  // 将 other_t 类型转换为 scalar_t 类型，并使用静态断言验证转换后的复数对象的实部与 num1 相等
  constexpr scalar_t num3 = scalar_t(num1);
  // 将 other_t 类型转换为 scalar_t 类型，并使用静态断言验证转换后的复数对象的虚部与 num2 相等
  constexpr scalar_t num4 = scalar_t(num2);
  static_assert(
      c10::complex<scalar_t>(c10::complex<other_t>(num1, num2)).real() == num3,
      "");
  static_assert(
      c10::complex<scalar_t>(c10::complex<other_t>(num1, num2)).imag() == num4,
      "");
}

MAYBE_GLOBAL void test_convert_constructors() {
  // 调用模板函数 test_construct_from_scalar 以测试 float 和 double 类型的标量
  test_construct_from_scalar<float>();
  test_construct_from_scalar<double>();

  // 使用静态断言验证两个相同类型的复数对象之间可相互转换
  static_assert(
      std::is_convertible<c10::complex<float>, c10::complex<float>>::value, "");
  // 使用静态断言验证不同类型的复数对象之间不可相互转换
  static_assert(
      !std::is_convertible<c10::complex<double>, c10::complex<float>>::value,
      "");
  // 使用静态断言验证从 float 到 double 类型的复数对象转换是可行的
  static_assert(
      std::is_convertible<c10::complex<float>, c10::complex<double>>::value,
      "");
  // 使用静态断言验证从 double 到 double 类型的复数对象转换是可行的
  static_assert(
      std::is_convertible<c10::complex<double>, c10::complex<double>>::value,
      "");

  // 使用静态断言验证同类型复数对象之间的构造是可行的
  static_assert(
      std::is_constructible<c10::complex<float>, c10::complex<float>>::value,
      "");
  // 使用静态断言验证从 float 到 double 类型的复数对象构造是可行的
  static_assert(
      std::is_constructible<c10::complex<double>, c10::complex<float>>::value,
      "");
  // 使用静态断言验证从 double 到 float 类型的复数对象构造是可行的
  static_assert(
      std::is_constructible<c10::complex<float>, c10::complex<double>>::value,
      "");
  // 使用静态断言验证同类型复数对象之间的构造是可行的
  static_assert(
      std::is_constructible<c10::complex<double>, c10::complex<double>>::value,
      "");

  // 调用模板函数 test_construct_from_other 以测试不同的 float 和 double 类型转换
  test_construct_from_other<float, float>();
  test_construct_from_other<float, double>();
  test_construct_from_other<double, float>();
  test_construct_from_other<double, double>();
}

template <typename scalar_t>
C10_HOST_DEVICE void test_construct_from_std() {
  // 定义并初始化两个标量值，使用 std::complex 类型包装
  constexpr scalar_t num1 = scalar_t(1.23);
  constexpr scalar_t num2 = scalar_t(4.56);
  // 使用静态断言验证 std::complex 转换为 c10::complex 后的实部与 num1 相等
  static_assert(
      c10::complex<scalar_t>(std::complex<scalar_t>(num1, num2)).real() == num1,
      "");
  // 使用静态断言验证 std::complex 转换为 c10::complex 后的虚部与 num2 相等
  static_assert(
      c10::complex<scalar_t>(std::complex<scalar_t>(num1, num2)).imag() == num2,
      "");
}

MAYBE_GLOBAL void test_std_conversion() {
  // 调用模板函数 test_construct_from_std 以测试 float 和 double 类型的标量
  test_construct_from_std<float>();
  test_construct_from_std<double>();
}

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename scalar_t>
TEST(TestConstructors, UnorderedMap) {
  // 创建一个无序映射表 m，键和值都是 c10::complex<double> 类型
  std::unordered_map<
      c10::complex<double>,
      c10::complex<double>,
      c10::hash<c10::complex<double>>>
      m;

  // 定义两个键值对
  auto key1 = c10::complex<double>(2.5, 3);
  auto key2 = c10::complex<double>(2, 0);
  auto val1 = c10::complex<double>(2, -3.2);
  auto val2 = c10::complex<double>(0, -3);

  // 将键值对存入映射表 m
  m[key1] = val1;
  m[key2] = val2;

  // 断言映射表中存储的值与预期相等
  ASSERT_EQ(m[key1], val1);
  ASSERT_EQ(m[key2], val2);
}

namespace assignment {

template <typename scalar_t>
constexpr c10::complex<scalar_t> one() {
  // 创建一个 c10::complex<scalar_t> 对象 result，初始化为复数 (3, 4)
  c10::complex<scalar_t> result(3, 4);

  // 将 result 赋值为 scalar_t(1)
  result = scalar_t(1);

  // 返回赋值后的 result 对象
  return result;
}

// 可能是全局函数或设备函数，测试赋实部的情况
MAYBE_GLOBAL void test_assign_real() {
  // 静态断言，验证 one<float>() 返回的实部是否为 float(1)
  static_assert(one<float>().real() == float(1), "");

  // 静态断言，验证 one<float>() 返回的虚部是否为 float()
  static_assert(one<float>().imag() == float(), "");

  // 静态断言，验证 one<double>() 返回的实部是否为 double(1)
  static_assert(one<double>().real() == double(1), "");

  // 静态断言，验证 one<double>() 返回的虚部是否为 double()
  static_assert(one<double>().imag() == double(), "");
}

// 定义一个返回包含两个 c10::complex<double> 和 c10::complex<float> 元组的 constexpr 函数
constexpr std::tuple<c10::complex<double>, c10::complex<float>> one_two() {
  // 定义一个 constexpr 的 c10::complex<float> 对象 src，值为 (1, 2)
  constexpr c10::complex<float> src(1, 2);

  // 定义 c10::complex<double> 和 c10::complex<float> 的返回值
  c10::complex<double> ret0;
  c10::complex<float> ret1;

  // 将 src 赋值给 ret0 和 ret1，然后返回包含它们的元组
  ret0 = ret1 = src;
  return std::make_tuple(ret0, ret1);
}

// 可能是全局函数或设备函数，测试赋其它类型的赋值情况
MAYBE_GLOBAL void test_assign_other() {
  // 调用 one_two() 获得元组
  constexpr auto tup = one_two();

  // 静态断言，验证元组中的 c10::complex<double> 实部是否为 double(1)
  static_assert(std::get<c10::complex<double>>(tup).real() == double(1), "");

  // 静态断言，验证元组中的 c10::complex<double> 虚部是否为 double(2)
  static_assert(std::get<c10::complex<double>>(tup).imag() == double(2), "");

  // 静态断言，验证元组中的 c10::complex<float> 实部是否为 float(1)
  static_assert(std::get<c10::complex<float>>(tup).real() == float(1), "");

  // 静态断言，验证元组中的 c10::complex<float> 虚部是否为 float(2)
  static_assert(std::get<c10::complex<float>>(tup).imag() == float(2), "");
}

// 定义一个返回包含两个 c10::complex<double> 和 c10::complex<float> 元组的 constexpr 函数
constexpr std::tuple<c10::complex<double>, c10::complex<float>> one_two_std() {
  // 定义一个 constexpr 的 std::complex<float> 对象 src，值为 (1, 1)
  constexpr std::complex<float> src(1, 1);

  // 定义 c10::complex<double> 和 c10::complex<float> 的返回值
  c10::complex<double> ret0;
  c10::complex<float> ret1;

  // 将 src 赋值给 ret0 和 ret1，然后返回包含它们的元组
  ret0 = ret1 = src;
  return std::make_tuple(ret0, ret1);
}

// 可能是全局函数或设备函数，测试赋 std::complex 的情况
MAYBE_GLOBAL void test_assign_std() {
  // 调用 one_two() 获得元组
  constexpr auto tup = one_two();

  // 静态断言，验证元组中的 c10::complex<double> 实部是否为 double(1)
  static_assert(std::get<c10::complex<double>>(tup).real() == double(1), "");

  // 静态断言，验证元组中的 c10::complex<double> 虚部是否为 double(2)
  static_assert(std::get<c10::complex<double>>(tup).imag() == double(2), "");

  // 静态断言，验证元组中的 c10::complex<float> 实部是否为 float(1)
  static_assert(std::get<c10::complex<float>>(tup).real() == float(1), "");

  // 静态断言，验证元组中的 c10::complex<float> 虚部是否为 float(2)
  static_assert(std::get<c10::complex<float>>(tup).imag() == float(2), "");
}

// 若定义了 __CUDACC__ 或 __HIPCC__，定义一个可能的主机或设备函数，返回包含两个 c10::complex<double> 和 c10::complex<float> 元组
#if defined(__CUDACC__) || defined(__HIPCC__)
C10_HOST_DEVICE std::tuple<c10::complex<double>, c10::complex<float>>
one_two_thrust() {
  // 定义一个 thrust::complex<float> 对象 src，值为 (1, 2)
  thrust::complex<float> src(1, 2);

  // 定义 c10::complex<double> 和 c10::complex<float> 的返回值
  c10::complex<double> ret0;
  c10::complex<float> ret1;

  // 将 src 赋值给 ret0 和 ret1，然后返回包含它们的元组
  ret0 = ret1 = src;
  return std::make_tuple(ret0, ret1);
}
template <typename scalar_t>
// 定义函数 p，接受一个标量值作为参数，返回一个复数对象
constexpr c10::complex<scalar_t> p(scalar_t value) {
  // 创建一个复数对象，实部和虚部都初始化为标量值 2
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  // 将复数对象加上参数值
  result += value;
  // 返回修改后的结果
  return result;
}

template <typename scalar_t>
// 定义函数 m，接受一个标量值作为参数，返回一个复数对象
constexpr c10::complex<scalar_t> m(scalar_t value) {
  // 创建一个复数对象，实部和虚部都初始化为标量值 2
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  // 将复数对象减去参数值
  result -= value;
  // 返回修改后的结果
  return result;
}

template <typename scalar_t>
// 定义函数 t，接受一个标量值作为参数，返回一个复数对象
constexpr c10::complex<scalar_t> t(scalar_t value) {
  // 创建一个复数对象，实部和虚部都初始化为标量值 2
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  // 将复数对象乘以参数值
  result *= value;
  // 返回修改后的结果
  return result;
}

template <typename scalar_t>
// 定义函数 d，接受一个标量值作为参数，返回一个复数对象
constexpr c10::complex<scalar_t> d(scalar_t value) {
  // 创建一个复数对象，实部和虚部都初始化为标量值 2
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  // 将复数对象除以参数值
  result /= value;
  // 返回修改后的结果
  return result;
}
// 定义一个测试函数，用于测试复数的算术运算赋值操作
C10_HOST_DEVICE void test_arithmetic_assign_scalar() {
  // 使用给定的函数 p，创建一个复数 x，其实部为 1，虚部为 2
  constexpr c10::complex<scalar_t> x = p(scalar_t(1));
  // 静态断言，验证 x 的实部是否为 3
  static_assert(x.real() == scalar_t(3), "");
  // 静态断言，验证 x 的虚部是否为 2
  static_assert(x.imag() == scalar_t(2), "");
  
  // 使用给定的函数 m，创建一个复数 y，其实部为 1，虚部为 2
  constexpr c10::complex<scalar_t> y = m(scalar_t(1));
  // 静态断言，验证 y 的实部是否为 1
  static_assert(y.real() == scalar_t(1), "");
  // 静态断言，验证 y 的虚部是否为 2
  static_assert(y.imag() == scalar_t(2), "");
  
  // 使用给定的函数 t，创建一个复数 z，其实部为 2，虚部为 2
  constexpr c10::complex<scalar_t> z = t(scalar_t(2));
  // 静态断言，验证 z 的实部是否为 4
  static_assert(z.real() == scalar_t(4), "");
  // 静态断言，验证 z 的虚部是否为 4
  static_assert(z.imag() == scalar_t(4), "");
  
  // 使用给定的函数 d，创建一个复数 t，其实部为 2，虚部为 2
  constexpr c10::complex<scalar_t> t = d(scalar_t(2));
  // 静态断言，验证 t 的实部是否为 1
  static_assert(t.real() == scalar_t(1), "");
  // 静态断言，验证 t 的虚部是否为 1
  static_assert(t.imag() == scalar_t(1), "");
}

// 定义一个模板函数 p，用于实现复数的加法赋值操作
template <typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> p(
    scalar_t real,
    scalar_t imag,
    c10::complex<rhs_t> rhs) {
  // 创建一个复数 result，其实部为 real，虚部为 imag
  c10::complex<scalar_t> result(real, imag);
  // 将 result 加上 rhs，并返回结果
  result += rhs;
  return result;
}

// 定义一个模板函数 m，用于实现复数的减法赋值操作
template <typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> m(
    scalar_t real,
    scalar_t imag,
    c10::complex<rhs_t> rhs) {
  // 创建一个复数 result，其实部为 real，虚部为 imag
  c10::complex<scalar_t> result(real, imag);
  // 将 result 减去 rhs，并返回结果
  result -= rhs;
  return result;
}

// 定义一个模板函数 t，用于实现复数的乘法赋值操作
template <typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> t(
    scalar_t real,
    scalar_t imag,
    c10::complex<rhs_t> rhs) {
  // 创建一个复数 result，其实部为 real，虚部为 imag
  c10::complex<scalar_t> result(real, imag);
  // 将 result 乘以 rhs，并返回结果
  result *= rhs;
  return result;
}

// 定义一个模板函数 d，用于实现复数的除法赋值操作
template <typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> d(
    scalar_t real,
    scalar_t imag,
    c10::complex<rhs_t> rhs) {
  // 创建一个复数 result，其实部为 real，虚部为 imag
  c10::complex<scalar_t> result(real, imag);
  // 将 result 除以 rhs，并返回结果
  result /= rhs;
  return result;
}

// 定义一个模板函数，用于测试复数的算术运算赋值操作
template <typename scalar_t>
C10_HOST_DEVICE void test_arithmetic_assign_complex() {
  // 导入 c10 命名空间中的复数字面量
  using namespace c10::complex_literals;
  // 使用给定的函数 p，创建一个复数 x2，其实部为 2，虚部为 2，rhs 为 1.0_if
  constexpr c10::complex<scalar_t> x2 = p(scalar_t(2), scalar_t(2), 1.0_if);
  // 静态断言，验证 x2 的实部是否为 2
  static_assert(x2.real() == scalar_t(2), "");
  // 静态断言，验证 x2 的虚部是否为 3
  static_assert(x2.imag() == scalar_t(3), "");
  
  // 使用给定的函数 p，创建一个复数 x3，其实部为 2，虚部为 2，rhs 为 1.0_id
  constexpr c10::complex<scalar_t> x3 = p(scalar_t(2), scalar_t(2), 1.0_id);
  // 静态断言，验证 x3 的实部是否为 2
  static_assert(x3.real() == scalar_t(2), "");

  // 如果不是在 nvcc 编译器下或者 CUDA 版本 >= 11020，则进行静态断言验证
#if !defined(__CUDACC__) || (defined(CUDA_VERSION) && CUDA_VERSION >= 11020)
  // 静态断言，验证 x3 的虚部是否为 3
  static_assert(x3.imag() == scalar_t(3), "");
#endif

  // 使用给定的函数 m，创建一个复数 y2，其实部为 2，虚部为 2，rhs 为 1.0_if
  constexpr c10::complex<scalar_t> y2 = m(scalar_t(2), scalar_t(2), 1.0_if);
  // 静态断言，验证 y2 的实部是否为 2
  static_assert(y2.real() == scalar_t(2), "");
  // 静态断言，验证 y2 的虚部是否为 1
  static_assert(y2.imag() == scalar_t(1), "");
  
  // 使用给定的函数 m，创建一个复数 y3，其实部为 2，虚部为 2，rhs 为 1.0_id
  constexpr c10::complex<scalar_t> y3 = m(scalar_t(2), scalar_t(2), 1.0_id);
  // 静态断言，验证 y3 的实部是否为 2
  static_assert(y3.real() == scalar_t(2), "");

  // 如果不是在 nvcc 编译器下或者 CUDA 版本 >= 11020，则进行静态断言验证
#if !defined(__CUDACC__) || (defined(CUDA_VERSION) && CUDA_VERSION >= 11020)
  // 静态断言，验证 y3 的虚部是否为 1
  static_assert(y3.imag() == scalar_t(1), "");
#endif
}
#endif

// 定义一个 constexpr 复数 z2，其实部为 1，虚部为 -2，使用 1.0_if 初始化
constexpr c10::complex<scalar_t> z2 = t(scalar_t(1), scalar_t(-2), 1.0_if);
// 静态断言，验证 z2 的实部为 2
static_assert(z2.real() == scalar_t(2), "");
// 静态断言，验证 z2 的虚部为 1
static_assert(z2.imag() == scalar_t(1), "");

// 定义一个 constexpr 复数 z3，其实部为 1，虚部为 -2，使用 1.0_id 初始化
constexpr c10::complex<scalar_t> z3 = t(scalar_t(1), scalar_t(-2), 1.0_id);
// 静态断言，验证 z3 的实部为 2
static_assert(z3.real() == scalar_t(2), "");
// 静态断言，验证 z3 的虚部为 1
static_assert(z3.imag() == scalar_t(1), "");

// 定义一个 constexpr 复数 t2，其实部为 -1，虚部为 2，使用 1.0_if 初始化
constexpr c10::complex<scalar_t> t2 = d(scalar_t(-1), scalar_t(2), 1.0_if);
// 静态断言，验证 t2 的实部为 2
static_assert(t2.real() == scalar_t(2), "");
// 静态断言，验证 t2 的虚部为 1
static_assert(t2.imag() == scalar_t(1), "");

// 定义一个 constexpr 复数 t3，其实部为 -1，虚部为 2，使用 1.0_id 初始化
constexpr c10::complex<scalar_t> t3 = d(scalar_t(-1), scalar_t(2), 1.0_id);
// 静态断言，验证 t3 的实部为 2
static_assert(t3.real() == scalar_t(2), "");
// 静态断言，验证 t3 的虚部为 1
static_assert(t3.imag() == scalar_t(1), "");
}

// 定义 MAYBE_GLOBAL 类型的函数 test_arithmetic_assign
MAYBE_GLOBAL void test_arithmetic_assign() {
  // 调用模板函数 test_arithmetic_assign_scalar，测试 float 类型
  test_arithmetic_assign_scalar<float>();
  // 调用模板函数 test_arithmetic_assign_scalar，测试 double 类型
  test_arithmetic_assign_scalar<double>();
  // 调用模板函数 test_arithmetic_assign_complex，测试 float 类型
  test_arithmetic_assign_complex<float>();
  // 调用模板函数 test_arithmetic_assign_complex，测试 double 类型
  test_arithmetic_assign_complex<double>();
}

} // namespace arithmetic_assign

// 进入命名空间 arithmetic
namespace arithmetic {

// 模板函数 test_arithmetic_，用于测试数学运算
template <typename scalar_t>
C10_HOST_DEVICE void test_arithmetic_() {
  // 静态断言，验证两个复数相等
  static_assert(
      c10::complex<scalar_t>(1, 2) == +c10::complex<scalar_t>(1, 2), "");
  // 静态断言，验证复数的相反数
  static_assert(
      c10::complex<scalar_t>(-1, -2) == -c10::complex<scalar_t>(1, 2), "");

  // 静态断言，验证复数的加法
  static_assert(
      c10::complex<scalar_t>(1, 2) + c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(4, 6),
      "");
  // 静态断言，验证复数与标量的加法
  static_assert(
      c10::complex<scalar_t>(1, 2) + scalar_t(3) ==
          c10::complex<scalar_t>(4, 2),
      "");
  // 静态断言，验证标量与复数的加法
  static_assert(
      scalar_t(3) + c10::complex<scalar_t>(1, 2) ==
          c10::complex<scalar_t>(4, 2),
      "");

  // 静态断言，验证复数的减法
  static_assert(
      c10::complex<scalar_t>(1, 2) - c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(-2, -2),
      "");
  // 静态断言，验证复数与标量的减法
  static_assert(
      c10::complex<scalar_t>(1, 2) - scalar_t(3) ==
          c10::complex<scalar_t>(-2, 2),
      "");
  // 静态断言，验证标量与复数的减法
  static_assert(
      scalar_t(3) - c10::complex<scalar_t>(1, 2) ==
          c10::complex<scalar_t>(2, -2),
      "");

  // 静态断言，验证复数的乘法
  static_assert(
      c10::complex<scalar_t>(1, 2) * c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(-5, 10),
      "");
  // 静态断言，验证复数与标量的乘法
  static_assert(
      c10::complex<scalar_t>(1, 2) * scalar_t(3) ==
          c10::complex<scalar_t>(3, 6),
      "");
  // 静态断言，验证标量与复数的乘法
  static_assert(
      scalar_t(3) * c10::complex<scalar_t>(1, 2) ==
          c10::complex<scalar_t>(3, 6),
      "");

  // 静态断言，验证复数的除法
  static_assert(
      c10::complex<scalar_t>(-5, 10) / c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(1, 2),
      "");
  // 静态断言，验证复数与标量的除法
  static_assert(
      c10::complex<scalar_t>(5, 10) / scalar_t(5) ==
          c10::complex<scalar_t>(1, 2),
      "");
  // 静态断言，验证标量与复数的除法
  static_assert(
      scalar_t(25) / c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(3, -4),
      "");
}

// 定义 MAYBE_GLOBAL 类型的函数 test_arithmetic，用于测试数学运算
MAYBE_GLOBAL void test_arithmetic() {
  // 调用模板函数 test_arithmetic_，测试 float 类型
  test_arithmetic_<float>();
  // 调用模板函数 test_arithmetic_，测试 double 类型
  test_arithmetic_<double>();
}

// 结束命名空间 arithmetic
} // namespace arithmetic
void test_binary_ops_for_int_type_(T real, T img, int_t num) {
  // 创建复数对象 c，使用传入的实部 real 和虚部 img
  c10::complex<T> c(real, img);
  
  // 测试加法操作：c + num 应该等于复数 (real + num, img)
  ASSERT_EQ(c + num, c10::complex<T>(real + num, img));
  
  // 测试加法操作：num + c 应该等于复数 (num + real, img)
  ASSERT_EQ(num + c, c10::complex<T>(num + real, img));
  
  // 测试减法操作：c - num 应该等于复数 (real - num, img)
  ASSERT_EQ(c - num, c10::complex<T>(real - num, img));
  
  // 测试减法操作：num - c 应该等于复数 (num - real, -img)
  ASSERT_EQ(num - c, c10::complex<T>(num - real, -img));
  
  // 测试乘法操作：c * num 应该等于复数 (real * num, img * num)
  ASSERT_EQ(c * num, c10::complex<T>(real * num, img * num));
  
  // 测试乘法操作：num * c 应该等于复数 (num * real, num * img)
  ASSERT_EQ(num * c, c10::complex<T>(num * real, num * img));
  
  // 测试除法操作：c / num 应该等于复数 (real / num, img / num)
  ASSERT_EQ(c / num, c10::complex<T>(real / num, img / num));
  
  // 测试除法操作：num / c 应该等于复数 (num * real / std::norm(c), -num * img / std::norm(c))
  ASSERT_EQ(
      num / c,
      c10::complex<T>(num * real / std::norm(c), -num * img / std::norm(c)));
}

template <typename T>
void test_binary_ops_for_all_int_types_(T real, T img, int8_t i) {
  // 调用 test_binary_ops_for_int_type_ 函数，分别测试不同整型类型的运算
  test_binary_ops_for_int_type_<T, int8_t>(real, img, i);
  test_binary_ops_for_int_type_<T, int16_t>(real, img, i);
  test_binary_ops_for_int_type_<T, int32_t>(real, img, i);
  test_binary_ops_for_int_type_<T, int64_t>(real, img, i);
}

TEST(TestArithmeticIntScalar, All) {
  // 测试不同浮点类型下的整型运算
  test_binary_ops_for_all_int_types_<float>(1.0, 0.1, 1);
  test_binary_ops_for_all_int_types_<double>(-1.3, -0.2, -2);
}

} // namespace arithmetic

namespace equality {

template <typename scalar_t>
C10_HOST_DEVICE void test_equality_() {
  // 测试复数相等性和不等性
  static_assert(
      c10::complex<scalar_t>(1, 2) == c10::complex<scalar_t>(1, 2), "");
  static_assert(c10::complex<scalar_t>(1, 0) == scalar_t(1), "");
  static_assert(scalar_t(1) == c10::complex<scalar_t>(1, 0), "");
  static_assert(
      c10::complex<scalar_t>(1, 2) != c10::complex<scalar_t>(3, 4), "");
  static_assert(c10::complex<scalar_t>(1, 2) != scalar_t(1), "");
  static_assert(scalar_t(1) != c10::complex<scalar_t>(1, 2), "");
}

MAYBE_GLOBAL void test_equality() {
  // 测试浮点类型的复数相等性和不等性
  test_equality_<float>();
  test_equality_<double>();
}

} // namespace equality

namespace io {

template <typename scalar_t>
void test_io_() {
  // 测试复数的输入输出流操作
  std::stringstream ss;
  c10::complex<scalar_t> a(1, 2);
  ss << a;
  ASSERT_EQ(ss.str(), "(1,2)");
  ss.str("(3,4)");
  ss >> a;
  ASSERT_TRUE(a == c10::complex<scalar_t>(3, 4));
}

TEST(TestIO, All) {
  // 测试不同浮点类型的复数输入输出流操作
  test_io_<float>();
  test_io_<double>();
}

} // namespace io

namespace test_std {

template <typename scalar_t>
C10_HOST_DEVICE void test_callable_() {
  // 测试复数的 std 函数调用
  static_assert(std::real(c10::complex<scalar_t>(1, 2)) == scalar_t(1), "");
  static_assert(std::imag(c10::complex<scalar_t>(1, 2)) == scalar_t(2), "");
  std::abs(c10::complex<scalar_t>(1, 2));
  std::arg(c10::complex<scalar_t>(1, 2));
  static_assert(std::norm(c10::complex<scalar_t>(3, 4)) == scalar_t(25), "");
  static_assert(
      std::conj(c10::complex<scalar_t>(3, 4)) == c10::complex<scalar_t>(3, -4),
      "");
  c10::polar(float(1), float(PI / 2));
  c10::polar(double(1), double(PI / 2));
}

MAYBE_GLOBAL void test_callable() {
  // 测试不同浮点类型的复数 std 函数调用
  test_callable_<float>();
  test_callable_<double>();
}

template <typename scalar_t>
void test_values_() {
  // 断言：验证复数的绝对值是否正确
  ASSERT_EQ(std::abs(c10::complex<scalar_t>(3, 4)), scalar_t(5));
  // 断言：验证复数的辐角是否接近 π/2
  ASSERT_LT(std::abs(std::arg(c10::complex<scalar_t>(0, 1)) - PI / 2), 1e-6);
  // 断言：验证极坐标和复数形式的转换是否正确
  ASSERT_LT(
      std::abs(
          c10::polar(scalar_t(1), scalar_t(PI / 2)) -
          c10::complex<scalar_t>(0, 1)),
      1e-6);
}

TEST(TestStd, BasicFunctions) {
  // 调用测试函数模板，验证 float 类型的基本函数
  test_values_<float>();
  // 调用测试函数模板，验证 double 类型的基本函数
  test_values_<double>();
  
  // CSQRT 边界情况：检查通过极坐标形式计算平方根时可能发生的溢出
  ASSERT_LT(
      std::abs(std::sqrt(c10::complex<float>(-1e20, -4988429.2)).real()), 3e-4);
  ASSERT_LT(
      std::abs(std::sqrt(c10::complex<double>(-1e60, -4988429.2)).real()),
      3e-4);
}

} // namespace test_std
```