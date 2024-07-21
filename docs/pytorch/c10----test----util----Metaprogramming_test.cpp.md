# `.\pytorch\c10\test\util\Metaprogramming_test.cpp`

```
// 包含必要的头文件来使用所需的库和工具
#include <c10/test/util/Macros.h>
#include <c10/util/Metaprogramming.h>
#include <gtest/gtest.h>
#include <cstdlib>

// 使用 c10::guts 命名空间以便访问其中的功能
using namespace c10::guts;

// NOLINTBEGIN(modernize*) 注释：禁止 LINT 提示，避免现代化修改的建议

// 匿名命名空间，用于隐藏内部实现细节
namespace {

// 声明一个用于测试函数 traits 的命名空间 test_function_traits
namespace test_function_traits {

// 静态断言，验证函数 traits 对于返回类型为 void(int, float) 的正确性
static_assert(
    std::is_same<void, typename function_traits<void(int, float)>::return_type>::value,
    "");

// 静态断言，验证函数 traits 对于返回类型为 int 的函数 int(int, float) 的正确性
static_assert(
    std::is_same<int, typename function_traits<int(int, float)>::return_type>::value,
    "");

// 静态断言，验证函数 traits 对于参数类型为 int, float 的函数 void(int, float) 的正确性
static_assert(
    std::is_same<typelist::typelist<int, float>,
                 typename function_traits<void(int, float)>::parameter_types>::value,
    "");

// 静态断言，验证函数 traits 对于参数类型为 int, float 的函数 int(int, float) 的正确性
static_assert(
    std::is_same<typelist::typelist<int, float>,
                 typename function_traits<int(int, float)>::parameter_types>::value,
    "");

// 静态断言，验证 make_function_traits_t 对于返回类型为 bool, 参数类型为 int, float 的正确性
static_assert(
    std::is_same<bool,
                 typename make_function_traits_t<bool, typelist::typelist<int, float>>::return_type>::value,
    "");

// 静态断言，验证 make_function_traits_t 对于返回类型为 void, 参数类型为 int, float 的正确性
static_assert(
    std::is_same<void,
                 typename make_function_traits_t<void, typelist::typelist<int, float>>::return_type>::value,
    "");

// 静态断言，验证 make_function_traits_t 对于返回类型为 bool, 参数类型为 int, float 的参数类型的正确性
static_assert(
    std::is_same<typelist::typelist<int, float>,
                 typename make_function_traits_t<bool, typelist::typelist<int, float>>::parameter_types>::value,
    "");

// 静态断言，验证 make_function_traits_t 对于返回类型为 void, 参数类型为 int, float 的参数类型的正确性
static_assert(
    std::is_same<typelist::typelist<int, float>,
                 typename make_function_traits_t<void, typelist::typelist<int, float>>::parameter_types>::value,
    "");

// 静态断言，验证 make_function_traits_t 对于返回类型为 bool, 参数类型为 int, float 的函数类型的正确性
static_assert(
    std::is_same<bool(int, float),
                 typename make_function_traits_t<bool, typelist::typelist<int, float>>::func_type>::value,
    "");

// 静态断言，验证 make_function_traits_t 对于返回类型为 void, 参数类型为 int, float 的函数类型的正确性
static_assert(
    std::is_same<void(int, float),
                 typename make_function_traits_t<void, typelist::typelist<int, float>>::func_type>::value,
    "");

}  // namespace test_function_traits

// 定义一个只可移动的类 MovableOnly
struct MovableOnly {
  // constexpr 构造函数，初始化 val
  constexpr MovableOnly(int val_) : val(val_) { /* no default constructor */ }

  // 删除拷贝构造函数
  MovableOnly(const MovableOnly&) = delete;

  // 默认生成移动构造函数
  MovableOnly(MovableOnly&&) = default;

  // 删除拷贝赋值运算符
  MovableOnly& operator=(const MovableOnly&) = delete;

  // 默认生成移动赋值运算符
  MovableOnly& operator=(MovableOnly&&) = default;

  // 友元函数，比较两个 MovableOnly 对象是否相等
  friend bool operator==(const MovableOnly& lhs, const MovableOnly& rhs) {
    return lhs.val == rhs.val;
  }

 private:
  int val;  // 私有成员变量 val
};

// 模板定义，用于判断类型 T 是否为 MovableOnly 类型
template <class T>
using is_my_movable_only_class =
    std::is_same<MovableOnly, std::remove_cv_t<std::remove_reference_t<T>>>;

// 定义一个计数拷贝操作的结构体 CopyCounting
struct CopyCounting {
  int move_count;   // 移动构造函数调用计数
  int copy_count;   // 拷贝构造函数调用计数

  // 默认构造函数，初始化 move_count 和 copy_count
  CopyCounting() : move_count(0), copy_count(0) {}

  // 拷贝构造函数，初始化 move_count 为 rhs.move_count，copy_count 为 rhs.copy_count + 1
  CopyCounting(const CopyCounting& rhs)
      : move_count(rhs.move_count), copy_count(rhs.copy_count + 1) {}

  // 移动构造函数，使用 noexcept 说明符，初始化 move_count 为 rhs.move_count + 1，copy_count 为 rhs.copy_count
  CopyCounting(CopyCounting&& rhs) noexcept
      : move_count(rhs.move_count + 1), copy_count(rhs.copy_count) {}

  // 拷贝赋值运算符重载函数，赋值 move_count 和 copy_count
  CopyCounting& operator=(const CopyCounting& rhs) {
    move_count = rhs.move_count;
    copy_count = rhs.copy_count + 1;
    // 返回当前对象的引用
    return *this;


这里截止到了 CopyCounting 结构体的部分，根据需要继续完成剩余代码的注释。
    return *this;
  }
  CopyCounting& operator=(CopyCounting&& rhs) noexcept {
    move_count = rhs.move_count + 1;
    copy_count = rhs.copy_count;
    return *this;
  }



    // 返回当前对象的引用，用于赋值操作符的链式调用
    return *this;
  }
  // 移动赋值操作符，将右值引用作为参数，无异常抛出保证
  CopyCounting& operator=(CopyCounting&& rhs) noexcept {
    // 增加移动赋值计数器，rhs对象的move_count加1
    move_count = rhs.move_count + 1;
    // 将rhs对象的copy_count赋值给当前对象的copy_count
    copy_count = rhs.copy_count;
    // 返回当前对象的引用，完成移动赋值操作
    return *this;
  }
};

// 检查模板类型是否为复制计数类的特化
template <class T>
using is_my_copy_counting_class =
    std::is_same<CopyCounting, std::remove_cv_t<std::remove_reference_t<T>>>;

// test_tuple_elements 命名空间，包含元组元素选择的测试用例
namespace test_tuple_elements {
// 注意：不测试空选择，因为某些编译器会在 tuple_elements() 中引发"参数设置但未使用"的警告，
// 这展示了使用这些工具时可能出现的摩擦

// 测试元组元素选择函数 tuple_elements 的子集选择功能
TEST(MetaprogrammingTest, TupleElements_subsetSelection) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_elements(x, std::index_sequence<0, 2>());
  auto z = std::make_tuple(0, 2.0);
  EXPECT_EQ(y, z);
}

// 测试元组元素选择函数 tuple_elements 的重新排序功能
TEST(MetaprogrammingTest, TupleElements_reorderSelection) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_elements(x, std::index_sequence<0, 2, 1>());
  auto z = std::make_tuple(0, 2.0, "HEY");
  EXPECT_EQ(y, z);
}
} // namespace test_tuple_elements

// test_tuple_take 命名空间，包含元组前缀选择的测试用例
namespace test_tuple_take {
// 注意：不测试空前缀，参见上述空选择的说明

// 测试元组前缀选择函数 tuple_take 的非空前缀选择功能
TEST(MetaprogrammingTest, TupleTake_nonemptyPrefix) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_take<decltype(x), 2>(x);
  auto z = std::make_tuple(0, "HEY");
  EXPECT_EQ(y, z);
}

// 测试元组前缀选择函数 tuple_take 的完整前缀选择功能
TEST(MetaprogrammingTest, TupleTake_fullPrefix) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_take<decltype(x), 3>(x);
  EXPECT_EQ(x, y);
}

// 测试元组前缀选择函数 tuple_take 的负数索引功能
TEST(MetaprogrammingTest, TupleTake_negative) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_take<decltype(x), -2>(x);
  auto z = std::make_tuple("HEY", 2.0);
  EXPECT_EQ(y, z);
}
} // namespace test_tuple_take

// test_tuple_slice 命名空间，包含元组切片功能的测试用例
namespace test_tuple_slice {
// 测试元组切片函数 tuple_slice 的中间切片功能
TEST(MetaprogrammingTest, TupleSlice_middle) {
  auto x = std::make_tuple(0, "HEY", 2.0, false);
  auto y = tuple_slice<decltype(x), 1, 2>(x);
  auto z = std::make_tuple("HEY", 2.0);
  EXPECT_EQ(y, z);
}

// 测试元组切片函数 tuple_slice 的完整切片功能
TEST(MetaprogrammingTest, TupleSlice_full) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_slice<decltype(x), 0, 3>(x);
  EXPECT_EQ(x, y);
}
} // namespace test_tuple_slice

// test_tuple_map 命名空间，包含元组映射功能的测试用例
namespace test_tuple_map {
// 测试元组映射函数 tuple_map 的简单映射功能
TEST(MetaprogrammingTest, TupleMap_simple) {
  auto result = tuple_map(
      std::tuple<int32_t, int32_t, int32_t>(3, 4, 5),
      [](int32_t a) -> int16_t { return static_cast<int16_t>(a + 1); });
  static_assert(
      std::is_same<std::tuple<int16_t, int16_t, int16_t>, decltype(result)>::
          value,
      "");
  EXPECT_EQ(4, std::get<0>(result));
  EXPECT_EQ(5, std::get<1>(result));
  EXPECT_EQ(6, std::get<2>(result));
}

// 测试元组映射函数 tuple_map 的映射器接受不同但可转换类型的功能
TEST(MetaprogrammingTest, TupleMap_mapperTakesDifferentButConvertibleType) {
  auto result = tuple_map(
      std::tuple<int32_t, int32_t, int32_t>(3, 4, 5),
      [](int64_t a) -> int16_t { return static_cast<int16_t>(a + 1); });
  static_assert(
      std::is_same<std::tuple<int16_t, int16_t, int16_t>, decltype(result)>::
          value,
      "");
  EXPECT_EQ(4, std::get<0>(result));
  EXPECT_EQ(5, std::get<1>(result));
  EXPECT_EQ(6, std::get<2>(result));
}
} // namespace test_tuple_map
TEST(MetaprogrammingTest, TupleMap_mapperTakesConstRef) {
  // 定义一个元组，并对其进行映射操作，将每个元素加1并转换为int16_t类型
  auto result = tuple_map(
      std::tuple<int32_t, int32_t, int32_t>(3, 4, 5),
      [](const int32_t& a) -> int16_t { return static_cast<int16_t>(a + 1); });
  // 静态断言，验证结果元组类型是否为<std::tuple<int16_t, int16_t, int16_t>>
  static_assert(
      std::is_same<std::tuple<int16_t, int16_t, int16_t>, decltype(result)>::
          value,
      "");
  // 验证映射后的第一个元素是否为4
  EXPECT_EQ(4, std::get<0>(result));
  // 验证映射后的第二个元素是否为5
  EXPECT_EQ(5, std::get<1>(result));
  // 验证映射后的第三个元素是否为6
  EXPECT_EQ(6, std::get<2>(result));
}

TEST(MetaprogrammingTest, TupleMap_mapsToDifferentTypes) {
  // 定义一个结构体 Mapper，包含两个重载运算符，分别用于int32_t和std::string类型的映射
  struct Mapper {
    std::string operator()(int32_t a) const {
      return std::to_string(a);
    }
    int32_t operator()(const std::string& a) const {
      return atoi(a.c_str());
    }
  };
  // 使用 Mapper 对象对元组<std::tuple<int32_t, std::string>>进行映射操作
  auto result = tuple_map(std::tuple<int32_t, std::string>(3, "4"), Mapper());
  // 静态断言，验证结果元组类型是否为<std::tuple<std::string, int32_t>>
  static_assert(
      std::is_same<std::tuple<std::string, int32_t>, decltype(result)>::value,
      "");
  // 验证映射后的第一个元素是否为"3"
  EXPECT_EQ("3", std::get<0>(result));
  // 验证映射后的第二个元素是否为4
  EXPECT_EQ(4, std::get<1>(result));
}

TEST(MetaprogrammingTest, TupleMap_differentiatesLRValueReferences) {
  // 定义一个结构体 Mapper，包含两个重载运算符，分别用于std::string&&和const std::string&的映射
  struct Mapper {
    std::string operator()(std::string&& a) const {
      return "moved";
    }
    std::string operator()(const std::string& a) const {
      return "copied";
    }
  };
  std::string str1, str2;
  // 使用 Mapper 对象对元组<std::tuple<const std::string&, std::string&&>>进行映射操作
  auto result = tuple_map(
      std::tuple<const std::string&, std::string&&>(str1, std::move(str2)),
      Mapper());
  // 静态断言，验证结果元组类型是否为<std::tuple<std::string, std::string>>
  static_assert(
      std::is_same<std::tuple<std::string, std::string>, decltype(result)>::
          value,
      "");
  // 验证映射后的第一个元素是否为"copied"
  EXPECT_EQ("copied", std::get<0>(result));
  // 验证映射后的第二个元素是否为"moved"
  EXPECT_EQ("moved", std::get<1>(result));
}

TEST(MetaprogrammingTest, TupleMap_canWorkWithMovableOnlyType) {
  // 对只支持可移动类型 MovableOnly 的元组<std::tuple<MovableOnly>>进行映射操作
  auto result = tuple_map(
      std::tuple<MovableOnly>(MovableOnly(7)), [](MovableOnly a) { return a; });
  // 静态断言，验证结果元组类型是否为<std::tuple<MovableOnly>>
  static_assert(
      std::is_same<std::tuple<MovableOnly>, decltype(result)>::value, "");
  // 验证映射后的元素是否为 MovableOnly(7)
  EXPECT_EQ(MovableOnly(7), std::get<0>(result));
}

TEST(MetaprogrammingTest, TupleMap_doesntUnecessarilyCopyValues) {
  // 对拷贝计数类型 CopyCounting 的元组<std::tuple<CopyCounting>>进行映射操作
  auto result = tuple_map(
      std::tuple<CopyCounting>(CopyCounting()),
      [](CopyCounting a) { return a; });
  // 静态断言，验证结果元组类型是否为<std::tuple<CopyCounting>>
  static_assert(
      std::is_same<std::tuple<CopyCounting>, decltype(result)>::value, "");
  // 验证映射后的元素的移动计数是否为4
  EXPECT_EQ(4, std::get<0>(result).move_count);
  // 验证映射后的元素的拷贝计数是否为0
  EXPECT_EQ(0, std::get<0>(result).copy_count);
}

TEST(MetaprogrammingTest, TupleMap_doesntUnecessarilyMoveValues) {
  CopyCounting a;
  // 对右值引用类型的元组<std::tuple<CopyCounting&&>>进行映射操作
  auto result = tuple_map(
      std::tuple<CopyCounting&&>(std::move(a)),
      [](CopyCounting&& a) -> CopyCounting&& { return std::move(a); });
  // 静态断言，验证结果元组类型是否为<std::tuple<CopyCounting&&>>
  static_assert(
      std::is_same<std::tuple<CopyCounting&&>, decltype(result)>::value, "");
  // 验证映射后的元素地址是否与原始对象a的地址相同
  EXPECT_EQ(&a, &std::get<0>(result));
  // 验证映射后的元素的移动计数是否为0
  EXPECT_EQ(0, std::get<0>(result).move_count);
  // 验证映射后的元素的拷贝计数是否为0
  EXPECT_EQ(0, std::get<0>(result).copy_count);
}

TEST(MetaprogrammingTest, TupleMap_canBeUsedWithAutoLambdas) {
  struct A final {
    int32_t func() {
      return 5;
    }
  };
  struct B final {
    std::string func() {
      return "5";
    }
  };
    }
  };

这段代码定义了一个匿名的初始化列表，包含了两个元素，分别是类型 A 和 B 的默认构造对象。


  auto result =
      tuple_map(std::make_tuple(A(), B()), [](auto a) { return a.func(); });

创建了一个变量 `result`，使用 `tuple_map` 函数对 `std::tuple` 类型对象 `(A(), B())` 进行映射操作，使用 lambda 表达式调用每个元素的 `func()` 方法并收集结果。


  static_assert(
      std::is_same<std::tuple<int32_t, std::string>, decltype(result)>::value,
      "");

对 `result` 的类型进行静态断言，验证其是否为 `std::tuple<int32_t, std::string>` 类型，即第一个元素是 `int32_t` 类型，第二个元素是 `std::string` 类型。


  EXPECT_EQ(5, std::get<0>(result));
  EXPECT_EQ("5", std::get<1>(result));

使用 Google Test 的 `EXPECT_EQ` 断言，验证 `result` 中第一个元素是否等于 `5`，第二个元素是否等于 `"5"`。
}
} // namespace test_tuple_map

} // namespace
// NOLINTEND(modernize*)
```