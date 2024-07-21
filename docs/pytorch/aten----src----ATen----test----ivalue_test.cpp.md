# `.\pytorch\aten\src\ATen\test\ivalue_test.cpp`

```py
// 包含 ATen 库中的头文件
#include <ATen/ATen.h>
// 包含 ATen 核心库中的字典相关功能
#include <ATen/core/Dict.h>
// 包含 C10 库中的智能指针功能
#include <c10/util/intrusive_ptr.h>
// 包含 C10 库中的范围操作功能
#include <c10/util/irange.h>
// 包含 Google Mock 库中的头文件，用于单元测试
#include <gmock/gmock.h>
// 包含 Google Test 库中的头文件，用于单元测试
#include <gtest/gtest.h>
// 包含 PyTorch 的核心头文件
#include <torch/torch.h>

// 检查元组构造的片段
c10::IValue inspectTupleConstruction() {
  // 创建包含两个字符串的元组
  std::tuple<std::string, std::string> s = std::make_tuple(
      "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
  // 返回包含元组的 IValue 对象
  return c10::IValue(s);
}

// 命名空间 c10 的空实现

// 单元测试：IValue 类型基本存储测试
TEST(IValueTest, BasicStorage) {
  // 创建空的 Storage 对象
  at::Storage emptyStorage;
  // 创建随机数据的非空 Storage 对象
  at::Storage nonemptyStorage(at::rand({3, 4}).storage());
  // 创建空 IValue 对象和非空 IValue 对象
  IValue ivEmpty(emptyStorage);
  IValue ivNonempty(nonemptyStorage);

  // 断言：验证空 IValue 对象是否为 Storage 类型
  ASSERT_TRUE(ivEmpty.isStorage());
  // 断言：验证非空 IValue 对象是否为 Storage 类型
  ASSERT_TRUE(ivNonempty.isStorage());
  // 断言：验证空 IValue 对象转换后的 Storage 对象是否与原始的空 Storage 对象相同
  ASSERT_EQ(emptyStorage.unsafeGetStorageImpl(), ivEmpty.toStorage().unsafeGetStorageImpl());
  // 断言：验证非空 IValue 对象转换后的 Storage 对象是否与原始的非空 Storage 对象相同
  ASSERT_EQ(nonemptyStorage.unsafeGetStorageImpl(), ivNonempty.toStorage().unsafeGetStorageImpl());
}

// 单元测试：复杂字典的 IValue 类型测试
TEST(IValueTest, ComplexDict) {
  // 定义复杂字典的类型
  typedef c10::complex<double> c_type;
  // 创建空的复杂字典对象
  c10::Dict<c_type, c_type> m;
  // 创建两个复杂数并插入字典中
  auto num1 = c_type(2.3, -3.5);
  auto num2 = c_type(0, 5);
  m.insert(num1, 2 * num1);
  m.insert(num2, 2 * num2);
  // 创建包含复杂字典的 IValue 对象
  IValue dict(std::move(m));
  // 从 IValue 对象中获取通用字典
  auto m_ = dict.toGenericDict();
  // 断言：验证从 IValue 中获取的字典是否符合预期
  ASSERT_EQ(m_.at(num1), 2 * num1);
  ASSERT_EQ(m_.at(num2), 2 * num2);
}

// 创建样本 IValue 数组的函数，包含 16 个样本
// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
static std::array<IValue, 16> makeSampleIValues() {
  return {
    IValue(),
    // 创建随机数值的 Tensor
    at::rand({3, 4}),
    // 创建随机数据的 Storage
    at::rand({3, 4}).storage(),
    1.5, // 浮点数值
    // 创建复数
    c10::complex<double>(2.5, -0.5),
    42, // 整数
    true, // 布尔值
    // 创建整数和字符串的元组
    std::make_tuple(23, "hello"),
    "hello", // 字符串
    // 创建 caffe2::Blob 的智能指针
    c10::make_intrusive<caffe2::Blob>(),
    // 创建整数列表
    c10::List<int64_t>({1, 2, 3}),
    // 创建空的字符串字典
    c10::Dict<std::string, std::string>(),
    // 创建 FloatType 的 ivalue::Future 对象的智能指针
    c10::make_intrusive<ivalue::Future>(FloatType::get()),
    // 创建 CPU 设备对象
    c10::Device(c10::DeviceType::CPU, 0),
    // 创建 CPU 上默认流的对象
    c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::CPU, 0)),
    // 创建包含指向 class1 类型的空指针的 ivalue::Object 对象
    c10::make_intrusive<ivalue::Object>(c10::StrongTypePtr(nullptr, ClassType::create("class1", {})), 1),
  };
}

// 创建更多样本 IValue 数组的函数，包含 16 个样本
// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
static std::array<IValue, 16> makeMoreSampleIValues() {
  return {
    IValue(),
    // 创建随机数值的 Tensor
    at::rand({3, 4}),
    // 创建随机数据的 Storage
    at::rand({3, 4}).storage(),
    2.5, // 浮点数值
    // 创建复数
    c10::complex<double>(2.7, -0.3),
    43, // 整数
    false, // 布尔值
    // 创建整数和字符串的元组
    std::make_tuple(1, "goodbye"),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    // 添加字符串 "goodbye" 到静态初始化列表
    "goodbye",
    // 创建一个名为 Blob 的空白对象，并用智能指针进行管理
    c10::make_intrusive<caffe2::Blob>(),
    // 使用 List 初始化列表创建包含整数 {4, 5, 6} 的 C++ List
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    // 忽略下一行的魔术数字警告
    c10::List<int64_t>({4, 5, 6}),
    // 创建一个空的从字符串到字符串的字典对象
    c10::Dict<std::string, std::string>(),
    // 创建一个带有整数类型的 Future 对象
    c10::make_intrusive<ivalue::Future>(IntType::get()),
    // 创建一个 CUDA 设备对象，设备索引为 2
    c10::Device(c10::DeviceType::CUDA, 2),
    // 创建一个 CUDA 设备上默认流的对象，设备索引为 1
    c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::CUDA, 1)),
    // 使用智能指针创建一个包含类类型为 "class2" 的对象，并将其引用计数初始化为 2
    c10::make_intrusive<ivalue::Object>(c10::StrongTypePtr(nullptr, ClassType::create("class2", {})), 2),
  };
// 定义一个宏 EXPECT_IVALUE_EQ，用于比较两个 IValue 对象是否相等，特别处理包含 Tensor 的情况
#define EXPECT_IVALUE_EQ(a, b)                          \
  EXPECT_EQ((a).isTensor(), (b).isTensor());            \  // 检查 a 和 b 是否都是 Tensor 类型或都不是
  if ((a).isTensor()) {                                 \  // 如果 a 是 Tensor 类型
    EXPECT_TRUE((a).toTensor().equal((b).toTensor()));  \  // 则比较它们的 Tensor 数据是否相等
  } else {                                              \  // 否则
    EXPECT_EQ((a), (b));                                \  // 直接比较它们的值
  }

TEST(IValueTest, Swap) {
  // swap() 有三种情况：tensor、intrusive_ptr 或者其他。在所有这三种情况的组合下进行测试。

  auto sampleInputs = makeSampleIValues();   // 创建示例输入
  auto sampleTargets = makeMoreSampleIValues();  // 创建更多示例目标
  for (const auto& input: sampleInputs) {    // 对于每个输入
    for (const auto& target: sampleTargets) {  // 对于每个目标
      IValue a(input);                       // 创建 IValue 对象 a
      IValue b(target);                      // 创建 IValue 对象 b
      EXPECT_IVALUE_EQ(a, input);            // 检查 a 是否等于 input
      EXPECT_IVALUE_EQ(b, target);           // 检查 b 是否等于 target
      a.swap(b);                             // 交换 a 和 b
      EXPECT_IVALUE_EQ(a, target);           // 检查交换后的 a 是否等于 target
      EXPECT_IVALUE_EQ(b, input);            // 检查交换后的 b 是否等于 input
    }
  }
}

TEST(IValueTest, CopyConstruct) {
  auto sampleInputs = makeSampleIValues();   // 创建示例输入
  for (const IValue& v: sampleInputs) {      // 对于每个输入 v
    IValue copy(v);                          // 使用拷贝构造函数创建副本 copy
    EXPECT_IVALUE_EQ(copy, v);               // 检查副本 copy 是否等于 v
  }
}

TEST(IValueTest, MoveConstruct) {
  auto sampleInputs = makeSampleIValues();   // 创建示例输入
  for (const IValue& v: sampleInputs) {      // 对于每个输入 v
    IValue source(v);                        // 创建源对象 source
    IValue target(std::move(source));        // 使用移动构造函数创建目标对象 target
    EXPECT_IVALUE_EQ(target, v);             // 检查目标对象 target 是否等于 v
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_TRUE(source.isNone());            // 检查源对象是否已被置为 None
  }
}

TEST(IValueTest, CopyAssign) {
  auto sampleInputs = makeSampleIValues();   // 创建示例输入
  auto sampleTargets = makeMoreSampleIValues();  // 创建更多示例目标

  for (const IValue& input: sampleInputs) {  // 对于每个输入 input
    for (const IValue& target: sampleTargets) {  // 对于每个目标 target
      IValue copyTo(target);                 // 创建目标对象 copyTo
      IValue copyFrom(input);                // 创建输入对象 copyFrom
      copyTo = copyFrom;                     // 执行赋值操作
      EXPECT_IVALUE_EQ(copyTo, input);       // 检查赋值后的 copyTo 是否等于 input
      EXPECT_IVALUE_EQ(copyFrom, input);     // 检查赋值后的 copyFrom 是否等于 input
      EXPECT_IVALUE_EQ(copyTo, copyFrom);    // 检查两个对象是否相等
    }
  }
}

TEST(IValueTest, MoveAssign) {
  auto sampleInputs = makeSampleIValues();   // 创建示例输入
  auto sampleTargets = makeMoreSampleIValues();  // 创建更多示例目标

  for (const IValue& input: sampleInputs) {  // 对于每个输入 input
    for (const IValue& target: sampleTargets) {  // 对于每个目标 target
      IValue moveTo(target);                 // 创建目标对象 moveTo
      IValue moveFrom(input);                // 创建输入对象 moveFrom
      moveTo = std::move(moveFrom);          // 执行移动赋值操作
      EXPECT_IVALUE_EQ(moveTo, input);       // 检查移动赋值后的 moveTo 是否等于 input
      // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
      EXPECT_TRUE(moveFrom.isNone());        // 检查移动赋值后的 moveFrom 是否被置为 None
    }
  }
}

TEST(IValueTest, Tuple) {
  std::tuple<int64_t, at::Tensor> t = std::make_tuple(123, at::randn({1}));  // 创建一个元组 t
  auto iv = IValue(t);                      // 使用元组 t 创建 IValue 对象 iv
  auto t_ = iv.to<std::tuple<int64_t, at::Tensor>>();  // 将 iv 转换为元组 t_
  ASSERT_EQ(std::get<0>(t_), 123);          // 检查元组 t_ 的第一个元素是否为 123
  ASSERT_EQ(
      std::get<1>(t_).item().to<float>(), std::get<1>(t).item().to<float>());  // 检查元组 t_ 的第二个 Tensor 元素是否相等
}
TEST(IValueTest, unsafeRemoveAttr) {
  // 创建一个编译单元对象
  auto cu = std::make_shared<CompilationUnit>();
  // 创建名为 "foo.bar" 的类类型对象
  auto cls = ClassType::create("foo.bar", cu);
  // 在类类型对象中添加两个属性，类型为 TensorType
  cls->addAttribute("attr1", TensorType::get());
  cls->addAttribute("attr2", TensorType::get());
  // 创建一个包含指定属性数目的对象实例
  auto obj = c10::ivalue::Object::create(
      c10::StrongTypePtr(cu, cls), cls->numAttributes());
  // 移除对象实例中的 "attr1" 属性
  obj->unsafeRemoveAttr("attr1");
  // 断言：类类型对象仍然包含 "attr1" 属性
  ASSERT_TRUE(cls->hasAttribute("attr1"));
  // 断言：类类型对象仍然包含 "attr2" 属性
  ASSERT_TRUE(cls->hasAttribute("attr2"));
  // 断言：对象实例的属性数为 1
  ASSERT_TRUE(obj->slots().size() == 1);
}

TEST(IValueTest, TuplePrint) {
  {
    // 创建包含一个整数元素的元组 IValue 对象
    IValue tp = std::make_tuple(3);

    // 创建字符串流对象 ss
    std::stringstream ss;
    // 将 tp 输出到 ss 中
    ss << tp;
    // 断言：输出字符串应为 "(3,)"
    ASSERT_EQ(ss.str(), "(3,)");
  }

  {
    // 创建包含两个整数元素的元组 IValue 对象
    IValue tp = std::make_tuple(3, 3);
    // 创建字符串流对象 ss
    std::stringstream ss;
    // 将 tp 输出到 ss 中
    ss << tp;
    // 断言：输出字符串应为 "(3, 3)"
    ASSERT_EQ(ss.str(), "(3, 3)");
  }
}

TEST(IValueTest, ComplexIValuePrint) {
  {
    // 创建复数 IValue 对象
    IValue complex(c10::complex<double>(2, -3));
    // 创建字符串流对象 ss
    std::stringstream ss;
    // 将复数对象 complex 输出到 ss 中
    ss << complex;
    // 断言：输出字符串应为 "2.-3.j"
    ASSERT_EQ(ss.str(), "2.-3.j");
  }

  {
    // 创建复数 IValue 对象
    IValue complex(c10::complex<double>(2, 0));
    // 创建字符串流对象 ss
    std::stringstream ss;
    // 将复数对象 complex 输出到 ss 中
    ss << complex;
    // 断言：输出字符串应为 "2.+0.j"
    ASSERT_EQ(ss.str(), "2.+0.j");
  }

  {
    // 创建复数 IValue 对象
    IValue complex(c10::complex<double>(0, 3));
    // 创建字符串流对象 ss
    std::stringstream ss;
    // 将复数对象 complex 输出到 ss 中
    ss << complex;
    // 断言：输出字符串应为 "0.+3.j"
    ASSERT_EQ(ss.str(), "0.+3.j");
  }
}

TEST(IValueTest, Complex) {
  // 创建复数对象 c 和 c_
  auto c = c10::complex<double>(2, 3);
  auto c_ = c10::complex<double>(2, -3);
  // 创建包含复数对象的 IValue 对象 c1, c2, c3
  IValue c1(c), c2(c_), c3{at::Scalar(c)};

  // 断言：c1 是复数类型
  ASSERT_TRUE(c1.isComplexDouble());
  // 断言：c3 是复数类型
  ASSERT_TRUE(c3.isComplexDouble());

  // 断言：c1 的复数值等于 c
  ASSERT_EQ(c, c1.toComplexDouble());
  // 断言：c1 与 c2 不相等
  ASSERT_FALSE(c1 == c2);
  // 断言：c1 与 c3 相等
  ASSERT_TRUE(c1 == c3);

  // 断言：c1 是标量类型
  ASSERT_TRUE(c1.isScalar());
  // 断言：将 c2 转换为标量并与 c_ 比较相等
  ASSERT_TRUE(c2.toScalar().equal(c_));
}

TEST(IValueTest, BasicFuture) {
  // 创建一个整型 Future 对象 f1
  auto f1 = c10::make_intrusive<ivalue::Future>(IntType::get());
  // 断言：f1 未完成
  ASSERT_FALSE(f1->completed());

  // 标记 f1 为完成状态，值为 42
  f1->markCompleted(IValue(42));
  // 断言：f1 已完成
  ASSERT_TRUE(f1->completed());
  // 断言：f1 的值为 42
  ASSERT_EQ(42, f1->value().toInt());
  // 创建 IValue 对象 iv，包含 f1
  IValue iv(f1);
  // 断言：iv 中包含的 Future 对象的值为 42
  ASSERT_EQ(42, iv.toFuture()->value().toInt());
}

TEST(IValueTest, FutureCallbacks) {
  // 创建一个整型 Future 对象 f2
  auto f2 = c10::make_intrusive<ivalue::Future>(IntType::get());
  int calledTimesA = 0;
  int calledTimesB = 0;

  // 添加回调函数到 f2，对其值进行检查
  f2->addCallback([&calledTimesA](ivalue::Future& f2) {
    ASSERT_TRUE(f2.completed());
    ASSERT_EQ(f2.value().toInt(), 43);
    ++calledTimesA;
  });

  // 标记 f2 为完成状态，值为 43
  f2->markCompleted(IValue(43));
  // 断言：回调函数被调用了一次
  ASSERT_EQ(calledTimesA, 1);
  // 断言：回调函数 B 未被调用
  ASSERT_EQ(calledTimesB, 0);

  // 添加另一个回调函数到 f2，对其值进行检查
  f2->addCallback([&calledTimesB](ivalue::Future& f2) {
    ASSERT_TRUE(f2.completed());
    ASSERT_EQ(f2.value().toInt(), 43);
    ++calledTimesB;
  });

  // 断言：回调函数 A 仍然被调用一次
  ASSERT_EQ(calledTimesA, 1);
  // 断言：回调函数 B 被调用一次
  ASSERT_EQ(calledTimesB, 1);
  // 断言：f2 没有错误
  ASSERT_FALSE(f2->hasError());
}

TEST(IValueTest, FutureExceptions) {
  // 创建一个整型 Future 对象 f3
  auto f3 = c10::make_intrusive<ivalue::Future>(IntType::get());
  int calledTimes = 0;

  // 添加异常处理回调函数到 f3
  f3->addCallback([&calledTimes](ivalue::Future& f3) {
    ASSERT_TRUE(f3.completed());
    try {
      (void)f3.value();
    } catch (const std::exception& e) {
      // 如果捕获到 "My Error" 异常，计数加一
      if (std::string(e.what()) == "My Error") {
        ++calledTimes;
      }
    }
  });
    }
  });

将代码块封装在 lambda 表达式中，用于异步操作完成后的回调处理。


  ivalue::Future::FutureError err("My Error");

创建名为 `err` 的 `FutureError` 对象，初始化错误消息为 "My Error"。


  f3->setError(std::make_exception_ptr(err));

将先前创建的 `err` 错误对象封装成异常指针，并设置给 `f3` 对象，表示该 Future 对象发生了错误。


  ASSERT_EQ(calledTimes, 1);

断言 `calledTimes` 的值等于 1，用于验证某个条件是否为真，通常用于单元测试。


  ASSERT_TRUE(f3->hasError());

断言 `f3` 对象是否有错误，期望其返回 true。


  ASSERT_EQ(f3->tryRetrieveErrorMessage(), std::string("My Error"));

断言尝试从 `f3` 对象中检索错误消息，预期返回的字符串是 "My Error"。
}

# 定义测试用例 `IValueTest` 中的 `FutureSetError` 测试
TEST(IValueTest, FutureSetError) {
    # 创建一个带有整型值类型的 Future 对象
    auto f1 = c10::make_intrusive<ivalue::Future>(IntType::get());
    # 设置 Future 对象的错误状态为运行时错误 "foo"
    f1->setError(std::make_exception_ptr(std::runtime_error("foo")));
    try {
        # 尝试再次设置 Future 对象的错误状态为运行时错误 "bar"
        f1->setError(std::make_exception_ptr(std::runtime_error("bar")));
        # 断言应该抛出异常，因为预期会抛出异常
        FAIL() << "Expected to throw";
    } catch (std::exception& e) {
        # 断言捕获的异常信息应包含 "Error already set"
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Error already set"));
        # 断言捕获的异常信息应包含 "foo"
        EXPECT_THAT(e.what(), ::testing::HasSubstr("foo"));
        # 断言捕获的异常信息应包含 "bar"
        EXPECT_THAT(e.what(), ::testing::HasSubstr("bar"));
    }
}

# 定义测试用例 `IValueTest` 中的 `ValueEquality` 测试
TEST(IValueTest, ValueEquality) {
    # 断言两个字符串值相等
    EXPECT_EQ(IValue("asdf"), IValue("asdf"));
    # 断言两个字符串值不相等（大小写敏感）
    EXPECT_NE(IValue("asdf"), IValue("ASDF"));
    # 断言字符串值和整数值不相等
    EXPECT_NE(IValue("2"), IValue(2));
    # 断言两个整数值相等
    EXPECT_EQ(IValue(1), IValue(1));

    # 检查返回一个布尔值的 equals() 变体的结果
    auto res = IValue("asdf").equals("asdf");
    EXPECT_TRUE(res.isBool());
    EXPECT_TRUE(res.toBool());

    res = IValue("asdf").equals(1);
    EXPECT_TRUE(res.isBool());
    EXPECT_FALSE(res.toBool());
}

# 定义测试用例 `IValueTest` 中的 `TensorEquality` 测试
TEST(IValueTest, TensorEquality) {
    # 创建一个零张量并克隆它
    auto rawTensor = torch::zeros({2, 3});
    auto rawTensorCopy = rawTensor.clone();
    auto t = IValue(rawTensor);
    auto tCopy = IValue(rawTensorCopy);

    # 下面的代码应该抛出异常，因为多元素张量的逐元素相等性是模棱两可的。
    auto testEquality = []() {
        return IValue(torch::ones({2, 3})) == IValue(torch::rand({2, 3}));
    };
    # NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    EXPECT_ANY_THROW(testEquality());

    # equals() 应该返回一个全为 `true` 的张量
    IValue eqTensor = t.equals(tCopy);
    EXPECT_TRUE(eqTensor.isTensor());
    auto booleanTrue = torch::ones({2, 3}).to(torch::kBool);
    EXPECT_TRUE(eqTensor.toTensor().equal(booleanTrue));

    # 测试身份检查
    EXPECT_TRUE(t.is(t));
    EXPECT_FALSE(t.is(tCopy));
    # NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    IValue tReference = t;
    EXPECT_TRUE(t.is(tReference));
}

# 定义测试用例 `IValueTest` 中的 `ListEquality` 测试
TEST(IValueTest, ListEquality) {
    # 创建三个包含不同整数列表的 IValue 对象
    IValue c1 = std::vector<int64_t>{0, 1, 2, 3};
    IValue c2 = std::vector<int64_t>{0, 1, 2, 3};
    IValue c3 = std::vector<int64_t>{0, 1, 2, 3, 4};
    # 断言两个相同的列表对象相等
    EXPECT_EQ(c1, c1);
    # 断言两个相同内容的列表对象相等
    EXPECT_EQ(c1, c2);
    # 断言 c1 和 c2 不是同一个对象
    EXPECT_FALSE(c1.is(c2));
    # 断言 c1 和 c3 不相等
    EXPECT_NE(c1, c3);
    # 断言 c2 和 c3 不相等
    EXPECT_NE(c2, c3);
}
TEST(IValueTest, DictEquality) {
  // 创建一个内部字典 innerDict，键和值都是字符串类型
  auto innerDict = c10::Dict<std::string, std::string>();
  innerDict.insert("foo", "bar");

  // 创建字典 d1，其值为内部字典 innerDict，插入了三个键值对
  auto d1 = c10::Dict<std::string, c10::Dict<std::string, std::string>>();
  d1.insert("one", innerDict);
  d1.insert("two", innerDict);
  d1.insert("three", innerDict);
  // 将字典 d1 包装为 IValue 类型
  auto c1 = IValue(d1);

  // 创建字典 d2，与 d1 结构相同，但是内部字典使用了 innerDict 的拷贝
  auto d2 = c10::Dict<std::string, c10::Dict<std::string, std::string>>();
  d2.insert("one", innerDict.copy());
  d2.insert("two", innerDict.copy());
  d2.insert("three", innerDict.copy());
  // 将字典 d2 包装为 IValue 类型
  auto c2 = IValue(d2);

  // 创建字典 d3，与 d2 结构相同，但包含一个额外的键值对
  auto d3 = c10::Dict<std::string, c10::Dict<std::string, std::string>>();
  d3.insert("one", innerDict.copy());
  d3.insert("two", innerDict.copy());
  d3.insert("three", innerDict.copy());
  d3.insert("four", innerDict.copy());
  // 将字典 d3 包装为 IValue 类型
  auto c3 = IValue(d3);

  // 创建字典 d4，与 d2 结构相同，但第三个键值对的值不同于 innerDict
  auto d4 = c10::Dict<std::string, c10::Dict<std::string, std::string>>();
  d4.insert("one", innerDict.copy());
  d4.insert("two", innerDict.copy());
  auto innerDictNotEqual = c10::Dict<std::string, std::string>();
  innerDictNotEqual.insert("bar", "foo");
  d4.insert("three", innerDictNotEqual);
  // 将字典 d4 包装为 IValue 类型
  auto c4 = IValue(d4);

  // 以下是各种字典的相等性测试
  EXPECT_EQ(c1, c1);        // c1 应与自身相等
  EXPECT_EQ(c1, c2);        // c1 与 c2 应相等
  EXPECT_FALSE(c1.is(c2));  // c1 和 c2 不应是同一对象
  EXPECT_NE(c1, c3);        // c1 和 c3 不应相等
  EXPECT_NE(c2, c3);        // c2 和 c3 不应相等
  EXPECT_NE(c1, c4);        // c1 和 c4 不应相等
  EXPECT_NE(c2, c4);        // c2 和 c4 不应相等
}

TEST(IValueTest, DictEqualityDifferentOrder) {
  // 创建字典 d1 和 d2，键值对顺序不同但内容相同
  auto d1 = c10::Dict<std::string, int64_t>();
  d1.insert("one", 1);
  d1.insert("two", 2);
  auto d2 = c10::Dict<std::string, int64_t>();
  d2.insert("two", 2);
  d2.insert("one", 1);

  // 测试两个字典是否相等
  EXPECT_EQ(d1, d2);
}

TEST(IValueTest, ListNestedEquality) {
  // 创建嵌套的整数向量列表
  IValue c1 = std::vector<std::vector<int64_t>>({{0}, {0, 1}, {0, 1, 2}});
  IValue c2 = std::vector<std::vector<int64_t>>({{0}, {0, 1}, {0, 1, 2}});
  IValue c3 = std::vector<std::vector<int64_t>>({{1}, {0, 1}, {0, 1, 2}});
  // 测试向量是否相等
  EXPECT_EQ(c1, c1);
  EXPECT_EQ(c1, c2);
  EXPECT_NE(c1, c3);
  EXPECT_NE(c2, c3);
}

TEST(IValueTest, StreamEquality) {
  // 创建两个设备对象和流对象，将流对象封装为 IValue 类型
  at::Device device1 = at::Device(kCUDA, 0);
  at::Device device2 = at::Device(kCUDA, 1);
  c10::Stream stream1 = c10::Stream(c10::Stream::Default::DEFAULT, device1);
  c10::Stream stream2 = c10::Stream(c10::Stream::Default::DEFAULT, device2);
  IValue lhs(stream1);
  IValue rhs_different(stream2);
  IValue rhs_same(stream1);
  // 测试流对象的相等性
  EXPECT_FALSE(lhs.equals(rhs_different).toBool());
  EXPECT_TRUE(lhs.equals(rhs_same).toBool());
}
TEST(IValueTest, EnumEquality) {
  // 创建一个共享的编译单元
  auto cu = std::make_shared<CompilationUnit>();
  // 创建整数类型的 IValue 对象
  IValue int_ivalue_1(1);
  IValue int_ivalue_2(2);
  // 创建字符串类型的 IValue 对象
  IValue str_ivalue_1("1");
  
  // 创建两个整数枚举类型
  auto int_enum_type1 = EnumType::create(
      "enum_class_1",
      IntType::get(),
      {{"enum_name_1", int_ivalue_1}, {"enum_name_2", int_ivalue_2}},
      cu);
  auto int_enum_type2 = EnumType::create(
      "enum_class_2",
      IntType::get(),
      {{"enum_name_1", int_ivalue_1}, {"enum_name_2", int_ivalue_2}},
      cu);
  // 创建一个字符串枚举类型
  auto string_enum_type = EnumType::create(
      "enum_class_3", StringType::get(), {{"enum_name_1", str_ivalue_1}}, cu);

  // 测试相等性，预期两个 IValue 对象相等
  EXPECT_EQ(
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1)),
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1))
  );

  // 测试不相等性，预期两个不同的整数枚举类型对象不相等
  EXPECT_NE(
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1)),
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type2, "enum_name_1", int_ivalue_1))
  );

  // 测试不相等性，预期两个相同枚举类型但不同枚举值的对象不相等
  EXPECT_NE(
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1)),
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_2", int_ivalue_2))
  );

  // 测试不相等性，预期整数类型和字符串类型枚举对象不相等
  EXPECT_NE(
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1)),
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          string_enum_type, "enum_name_1", str_ivalue_1))
  );
}

TEST(IValueTest, isPtrType) {
  // 创建包含随机数据的 Tensor 类型 IValue 对象
  IValue tensor(at::rand({3, 4}));
  // 创建未定义的 Tensor 类型 IValue 对象
  IValue undefinedTensor((at::Tensor()));
  // 创建整数类型的 IValue 对象
  IValue integer(42);
  // 创建字符串类型的 IValue 对象
  IValue str("hello");

  // 测试是否为指针类型，预期 tensor 是指针类型
  EXPECT_TRUE(tensor.isPtrType());
  // 测试是否为指针类型，预期 undefinedTensor 不是指针类型
  EXPECT_FALSE(undefinedTensor.isPtrType());
  // 测试是否为指针类型，预期 integer 不是指针类型
  EXPECT_FALSE(integer.isPtrType());
  // 测试是否为指针类型，预期 str 是指针类型
  EXPECT_TRUE(str.isPtrType());
}

TEST(IValueTest, isAliasOf) {
  // 创建示例 IValue 对象数组
  auto sampleIValues = makeSampleIValues();
  for (auto& iv: sampleIValues) {
    for (auto& iv2: sampleIValues) {
      // 如果 iv 和 iv2 是同一个对象且是指针类型，预期 iv 是 iv2 的别名
      if (&iv == &iv2 && iv.isPtrType()) {
        EXPECT_TRUE(iv.isAliasOf(iv2));
      } else {
        // 否则，预期 iv 不是 iv2 的别名
        EXPECT_FALSE(iv.isAliasOf(iv2));
      }
    }
  }
}

TEST(IValueTest, internalToPointer) {
  // 创建包含随机数据的 Tensor 类型 IValue 对象
  IValue tensor(at::rand({3, 4}));
  // 创建字符串类型的 IValue 对象
  IValue str("hello");

  // 测试内部指针转换，预期结果是内部指针等于 unsafeToTensorImpl()
  EXPECT_EQ(tensor.internalToPointer(), tensor.unsafeToTensorImpl());
  // 测试内部指针转换，预期结果是内部指针不等于 nullptr
  EXPECT_NE(str.internalToPointer(), nullptr);

  // 创建空字符串类型的 IValue 对象
  IValue nullStr((c10::intrusive_ptr<ivalue::ConstantString>()));
  ASSERT_TRUE(nullStr.isString());
  // 测试内部指针转换，预期结果是内部指针等于 nullptr
  EXPECT_EQ(nullStr.internalToPointer(), nullptr);
}
// 定义测试函数 IValueTest.IdentityComparisonAndHashing
TEST(IValueTest, IdentityComparisonAndHashing) {
  // 创建形状为 [3, 4] 的随机张量 t1 和 t2
  at::Tensor t1 = at::rand({3, 4});
  at::Tensor t2 = at::rand({3, 4});
  // 将张量 t1 和 t2 封装为 IValue 对象 tv1 和 tv2
  IValue tv1(t1), tv2(t2);
  // 重新使用张量 t1 封装为另一个 IValue 对象 tv1b
  IValue tv1b(t1);

  // 检查 tv1 和 tv1b 的哈希值是否相等
  EXPECT_EQ(tv1.hash(), tv1b.hash());
  // 检查 tv1 和 tv2 的哈希值是否不相等
  EXPECT_NE(tv1.hash(), tv2.hash());

  // 检查对象是否引用相同的值
  EXPECT_TRUE(tv1.is(tv1));
  EXPECT_TRUE(tv1.is(tv1b));
  EXPECT_TRUE(tv1b.is(tv1));
  EXPECT_TRUE(tv2.is(tv2));

  // 检查 tv1 和 tv2 是否引用不同的值
  EXPECT_FALSE(tv1.is(tv2));
  EXPECT_FALSE(tv2.is(tv1));

  // 创建未初始化的 IValue 对象 none 和包含未定义张量的 IValue 对象 undefinedTensor
  IValue none;
  IValue undefinedTensor((at::Tensor()));

  // 检查 none 和 undefinedTensor 是否引用相同的值
  EXPECT_TRUE(none.is(undefinedTensor));
  EXPECT_TRUE(undefinedTensor.is(none));

  // 这里是否存在 Bug？我们可能期望有 is b => a.hash() == b.hash()
  EXPECT_NE(none.hash(), undefinedTensor.hash());

  // 创建示例 IValue 对象的向量 sampleIValues 和 sampleIValues2
  auto sampleIValues = makeSampleIValues();
  auto sampleIValues2 = makeSampleIValues();
  // 创建更多示例 IValue 对象的向量 moreSampleIValues
  auto moreSampleIValues = makeMoreSampleIValues();

  // 断言 sampleIValues 和 moreSampleIValues 的大小相等
  ASSERT_EQ(sampleIValues.size(), moreSampleIValues.size());
  // 遍历 sampleIValues 中的每个元素
  for (const auto ii : c10::irange(sampleIValues.size())) {
    // 如果当前 IValue 对象是复杂双精度、Blob、列表、Future、Stream、对象或通用字典，则跳过
    if (sampleIValues[ii].isComplexDouble() ||
        sampleIValues[ii].isBlob() ||
        sampleIValues[ii].isList() ||
        sampleIValues[ii].isFuture() ||
        sampleIValues[ii].isStream() ||
        sampleIValues[ii].isObject() ||
        sampleIValues[ii].isGenericDict()) {
      // 不可哈希的类型，跳过当前循环
      continue;
    }
    // 对于非元组的情况
    if (!sampleIValues[ii].isTuple()) {
      // 常量字符串将具有相同的指针值
      if (sampleIValues[ii].isPtrType() && !sampleIValues[ii].isString()) {
        // 检查在索引 ii 处的两个 IValue 对象的哈希值是否不相等
        EXPECT_NE(sampleIValues[ii].hash(), sampleIValues2[ii].hash())
          << " at index " << ii;
      } else {
        // 否则检查在索引 ii 处的两个 IValue 对象的哈希值是否相等
        EXPECT_EQ(sampleIValues[ii].hash(), sampleIValues2[ii].hash())
          << " at index " << ii;
      }
    }
    // 如果 sampleIValues[ii] 和 moreSampleIValues[ii] 都不是 None
    if (!sampleIValues[ii].isNone() && !moreSampleIValues[ii].isNone()) {
      // 检查在索引 ii 处的两个 IValue 对象的哈希值是否不相等
      EXPECT_NE(sampleIValues[ii].hash(), moreSampleIValues[ii].hash())
        << " at index " << ii;
    }
  }
}

// 稀疏张量不兼容静态 CPU 分发
#ifndef ATEN_CPU_STATIC_DISPATCH
TEST(IValueTest, IdentityAnd
TEST(IValueTest, getSubValues) {
  // 创建整数、浮点数和复数的 IValue 对象
  IValue integer(42), float_(1.5), complex(c10::complex<double>(2, 3));

  // 声明用于存储子值的哈希映射
  IValue::HashAliasedIValues subvalues;

  // 获取整数对象的子值，预期为空
  integer.getSubValues(subvalues);
  EXPECT_TRUE(subvalues.empty());

  // 清空子值映射
  subvalues.clear();

  // 获取浮点数对象的子值，预期为空
  float_.getSubValues(subvalues);
  EXPECT_TRUE(subvalues.empty());

  // 清空子值映射
  subvalues.clear();

  // 获取复数对象的子值，预期为空
  complex.getSubValues(subvalues);
  EXPECT_TRUE(subvalues.empty());

  // 清空子值映射
  subvalues.clear();

  // 创建两个张量，并用它们创建 IValue 对象
  at::Tensor t1(at::rand({3, 4})), t2(at::rand({3, 4}));
  IValue tv1(t1), tv2(t2);

  // 创建包含两个张量的列表，并用列表创建 IValue 对象
  IValue list(std::vector<at::Tensor>{t1, t2});

  // 创建包含 tv1 和 tv2 的元组，并用元组创建 IValue 对象
  IValue tuple(ivalue::Tuple::create({tv1, tv2}));

  // 创建字典，将张量和对应的键插入其中，并用字典创建 IValue 对象
  c10::Dict<int64_t, at::Tensor> m;
  m.insert(1, t1);
  m.insert(2, t2);
  IValue dict(std::move(m));

  // 创建自定义对象类型，并设置两个属性 t1 和 t2，用对象创建 IValue 对象
  auto objType = ClassType::create(nullopt, {});
  objType->addAttribute("t1", tv1.type());
  objType->addAttribute("t2", tv2.type());
  auto o = ivalue::Object::create(StrongTypePtr(nullptr, objType), 2);
  o->setSlot(0, tv1);
  o->setSlot(1, tv2);
  IValue object(o);

  // 获取 tv1 的子值，预期包含一个子值
  tv1.getSubValues(subvalues);
  EXPECT_EQ(subvalues.size(), 1);
  EXPECT_EQ(subvalues.count(tv1), 1);

  // 清空子值映射
  subvalues.clear();

  // 对于列表、元组、字典和对象，获取它们的子值，并预期每个对象包含三个子值
  for (auto& container: {list, tuple, dict, object}) {
    container.getSubValues(subvalues);
    EXPECT_EQ(subvalues.size(), 3);
    EXPECT_EQ(subvalues.count(container), 1);
    EXPECT_EQ(subvalues.count(tv1), 1);
    EXPECT_EQ(subvalues.count(tv2), 1);

    // 清空子值映射，准备处理下一个容器
    subvalues.clear();
  }
}

TEST(IValueTest, ScalarBool) {
  // 创建布尔类型的标量，并用标量创建 IValue 对象
  Scalar expected(true);
  IValue v(expected);
  
  // 将 IValue 对象转换为标量，并验证其为布尔类型且为 true
  Scalar actual = v.toScalar();
  EXPECT_TRUE(actual.isBoolean());
  EXPECT_TRUE(actual.toBool());
}

TEST(IValueTest, ToWeakAndBack) {
  // 创建样本输入，获取其弱引用并验证恢复后是否与原始对象相等
  auto sampleInputs = makeSampleIValues();
  for (const auto& sample: sampleInputs) {
    WeakIValue weak(sample);
    EXPECT_IVALUE_EQ(sample, weak.lock());
  }
}

// Storage and Generator did not set is_intrusive_ptr if they were
// undefined, which led use_count to return 1 instead of 0 for these
// cases.
TEST(IValueTest, UseCountCornerCases) {
  // 创建未定义的 Storage、Generator 和 Tensor 对象，并用它们创建相应的 IValue 对象
  at::Storage undefinedStorage;
  at::Generator undefinedGenerator;
  at::Tensor undefinedTensor;

  IValue ivEmptyStorage(undefinedStorage);
  IValue ivEmptyGenerator(undefinedGenerator);
  IValue ivEmptyTensor(undefinedTensor);

  // 验证各对象的引用计数是否符合预期
  ASSERT_EQ(1, ivEmptyStorage.use_count());
  ASSERT_EQ(1, ivEmptyGenerator.use_count());
  ASSERT_EQ(0, ivEmptyTensor.use_count());
}

// TODO(gmagogsfm): Add type conversion test?

using ivalue::TupleElements;

namespace {
void validateTupleElements(TupleElements& te, c10::ArrayRef<IValue> contents) {
  // 验证元组元素是否与给定内容相符
  EXPECT_EQ(te.empty(), contents.empty());
  EXPECT_EQ(te.size(), contents.size());
  for (const auto idx: c10::irange(contents.size())) {
    EXPECT_IVALUE_EQ(te[idx], contents[idx]);
    EXPECT_IVALUE_EQ(te.at(idx), contents[idx]);
    EXPECT_IVALUE_EQ(*(te.begin() + idx), contents[idx]);
  }
  if (!contents.empty()) {
    // 进一步验证元组元素的详细内容
    // (此处应包含接下来的代码，未提供)
    // 断言最后一个元素的值相等，使用 EXPECT_IVALUE_EQ 进行检查
    EXPECT_IVALUE_EQ(te.back(), contents.back());
  }
  // 移动 te 对象的内部向量到 v 中
  auto v = std::move(te).vec();
  // 断言 v 的大小与 contents 的大小相等
  EXPECT_EQ(v.size(), contents.size());
  // 遍历 contents 的索引范围
  for (const auto idx: c10::irange(contents.size())) {
    // 断言 v[idx] 的值与 contents[idx] 的值相等，使用 EXPECT_IVALUE_EQ 进行检查
    EXPECT_IVALUE_EQ(v[idx], contents[idx]);
  }
} // namespace

// 定义一个测试案例 TupleElementsTest.Basic
TEST(TupleElementsTest, Basic) {
  // 创建一个空的 TupleElements 对象
  TupleElements empty;
  // 验证空对象，并且其内容为空
  validateTupleElements(empty, {});
  // 创建一个包含一个元素的 TupleElements 对象
  TupleElements size1(1);
  // 验证包含一个元素的对象，内容为 {1}
  validateTupleElements(size1, {1});
  // 创建一个包含两个元素的 TupleElements 对象
  TupleElements size2(1, 2);
  // 验证包含两个元素的对象，内容为 {1, 2}
  validateTupleElements(size2, {1, 2});
  // 创建一个包含三个元素的 TupleElements 对象
  TupleElements size3(1, 2, 3);
  // 验证包含三个元素的对象，内容为 {1, 2, 3}

  // 创建一个示例值数组
  auto sampleIValuesArray = makeSampleIValues();
  // 创建一个包含多个元素的 TupleElements 对象
  TupleElements large(std::vector<IValue>(sampleIValuesArray.begin(), sampleIValuesArray.end()));
  // 验证包含多个元素的对象，内容与示例值数组相同
  validateTupleElements(large, sampleIValuesArray);
}

namespace {

// 定义三个工厂函数，返回不同的 TupleElements 对象
std::array<TupleElements(*)(), 3> factories = {
  []() { return TupleElements();},
  []() { return TupleElements(1, 2, 3);},
  []() { return TupleElements(std::vector<IValue>({1, 2, 3, "hello"})); }
};

// 预期的内容数组，与工厂函数返回的对象对应
std::array<std::vector<IValue>, 3> expectedContents = {
  std::vector<IValue>(),
  std::vector<IValue>({1, 2, 3}),
  std::vector<IValue>({1, 2, 3, "hello"}),
};

}

// 定义一个测试案例 TupleElementsTest.Resize
TEST(TupleElementsTest, Resize) {
  // 创建新的内容数组
  std::array<std::vector<IValue>, 3> newContents = {
    std::vector<IValue>(),
    std::vector<IValue>({4, 5, 6}),
    std::vector<IValue>({7, 8, 9, "hello"})
  };

  // 遍历工厂函数数组
  for (auto factory : factories) {
    // 遍历新内容数组
    for (const auto& contents : newContents) {
      // 从工厂函数创建一个 TupleElements 对象
      auto te = factory();
      // 创建内容的副本
      auto contentsCopy = contents;
      // 将副本设置为 TupleElements 对象的内容
      te.setContents(std::move(contentsCopy));
      // 验证 TupleElements 对象的内容是否正确
      validateTupleElements(te, contents);
    }
  }
}

// 定义一个测试案例 TupleElementsTest.CopyAndMoveConstruct
TEST(TupleElementsTest, CopyAndMoveConstruct) {
  // 初始化索引
  int idx = 0;
  // 遍历工厂函数数组
  for (auto fromFactory : factories) {
    // 从工厂函数创建一个对象，并进行移动构造
    auto toMoveFrom = fromFactory();
    TupleElements movedInto(std::move(toMoveFrom));
    // 验证移动构造后的对象内容是否正确
    validateTupleElements(movedInto, expectedContents[idx]);
    // 从工厂函数创建一个对象，并进行拷贝构造
    auto toCopyFrom = fromFactory();
    TupleElements copiedInto(toCopyFrom);
    // 验证拷贝构造后的对象内容是否正确
    validateTupleElements(copiedInto, expectedContents[idx]);
    idx++;
  }
}

// 定义一个测试案例 TupleElementsTest.CopyAndMoveAssign
TEST(TupleElementsTest, CopyAndMoveAssign) {
  // 初始化来源对象的索引
  int fromIdx = 0;
  // 遍历工厂函数数组
  for (auto fromFactory : factories) {
    // 遍历目标对象的工厂函数数组
    for (auto toFactory : factories) {
      // 从来源工厂函数创建一个对象
      auto from = fromFactory();
      // 从目标工厂函数创建一个对象
      auto to = toFactory();
      // 创建来源对象的拷贝
      auto copyFrom = fromFactory();
      // 创建目标对象的拷贝
      auto toCopy = toFactory();
      // 将来源对象移动给目标对象
      to = std::move(from);
      // 验证移动后目标对象的内容是否正确
      validateTupleElements(to, expectedContents[fromIdx]);
      // 将拷贝对象拷贝给目标对象
      toCopy = copyFrom;
      // 验证拷贝后目标对象的内容是否正确
      validateTupleElements(toCopy, expectedContents[fromIdx]);
    }
    fromIdx++;
  }
}

} // namespace c10
```