# `.\pytorch\c10\test\util\small_vector_test.cpp`

```
// 包含标准头文件 <c10/util/ArrayRef.h>，提供数组引用的支持
#include <c10/util/ArrayRef.h>
// 包含小型向量模板类 SmallVector 的头文件
#include <c10/util/SmallVector.h>
// 包含 Google 测试框架的头文件
#include <gtest/gtest.h>
// 包含标准头文件 <cstdarg>，提供变参函数的支持
#include <cstdarg>
// 包含标准头文件 <list>，提供链表的支持
#include <list>

// 禁用 lint 工具的警告 NOLINTBEGIN，避免对数组和转发引用过载的警告
//(*arrays, bugprone-forwarding-reference-overload)
using c10::SmallVector;           // 使用 c10 命名空间中的 SmallVector
using c10::SmallVectorImpl;       // 使用 c10 命名空间中的 SmallVectorImpl

namespace {

/// 一个辅助类，用于计算构造函数和析构函数的调用次数
class Constructable {
 private:
  static int numConstructorCalls;          // 静态变量，记录构造函数调用次数
  static int numMoveConstructorCalls;      // 静态变量，记录移动构造函数调用次数
  static int numCopyConstructorCalls;      // 静态变量，记录拷贝构造函数调用次数
  static int numDestructorCalls;           // 静态变量，记录析构函数调用次数
  static int numAssignmentCalls;           // 静态变量，记录赋值操作调用次数
  static int numMoveAssignmentCalls;       // 静态变量，记录移动赋值操作调用次数
  static int numCopyAssignmentCalls;       // 静态变量，记录拷贝赋值操作调用次数

  bool constructed;           // 标志对象是否已构造
  int value;                  // 存储的整数值

 public:
  // 默认构造函数，设置标志为已构造，并增加构造函数调用计数
  Constructable() : constructed(true), value(0) {
    ++numConstructorCalls;
  }

  // 带参数的构造函数，设置标志为已构造，初始化存储的值，并增加构造函数调用计数
  Constructable(int val) : constructed(true), value(val) {
    ++numConstructorCalls;
  }

  // 拷贝构造函数，设置标志为已构造，从另一个对象拷贝值，并增加构造函数和拷贝构造函数调用计数
  Constructable(const Constructable& src)
      : constructed(true), value(src.value) {
    ++numConstructorCalls;
    ++numCopyConstructorCalls;
  }

  // 移动构造函数，设置标志为已构造，从另一个对象移动值，增加构造函数和移动构造函数调用计数
  Constructable(Constructable&& src) noexcept
      : constructed(true), value(src.value) {
    src.value = 0;
    ++numConstructorCalls;
    ++numMoveConstructorCalls;
  }

  // 析构函数，断言对象已构造，增加析构函数调用计数，并标记对象为未构造
  ~Constructable() {
    EXPECT_TRUE(constructed);
    ++numDestructorCalls;
    constructed = false;
  }

  // 拷贝赋值操作符重载，断言对象已构造，从另一个对象拷贝值，增加赋值操作和拷贝赋值操作调用计数
  Constructable& operator=(const Constructable& src) {
    EXPECT_TRUE(constructed);
    value = src.value;
    ++numAssignmentCalls;
    ++numCopyAssignmentCalls;
    return *this;
  }

  // 移动赋值操作符重载，断言对象已构造，从另一个对象移动值，增加赋值操作和移动赋值操作调用计数
  Constructable& operator=(Constructable&& src) noexcept {
    EXPECT_TRUE(constructed);
    value = src.value;
    src.value = 0;
    ++numAssignmentCalls;
    ++numMoveAssignmentCalls;
    return *this;
  }

  // 获取存储值的绝对值的方法
  int getValue() const {
    return abs(value);
  }

  // 静态方法，重置所有计数器
  static void reset() {
    numConstructorCalls = 0;
    numMoveConstructorCalls = 0;
    numCopyConstructorCalls = 0;
    numDestructorCalls = 0;
    numAssignmentCalls = 0;
    numMoveAssignmentCalls = 0;
    numCopyAssignmentCalls = 0;
  }

  // 静态方法，获取构造函数调用次数
  static int getNumConstructorCalls() {
    return numConstructorCalls;
  }

  // 静态方法，获取移动构造函数调用次数
  static int getNumMoveConstructorCalls() {
    return numMoveConstructorCalls;
  }

  // 静态方法，获取拷贝构造函数调用次数
  static int getNumCopyConstructorCalls() {
    return numCopyConstructorCalls;
  }

  // 静态方法，获取析构函数调用次数
  static int getNumDestructorCalls() {
    return numDestructorCalls;
  }

  // 静态方法，获取赋值操作调用次数
  static int getNumAssignmentCalls() {
    return numAssignmentCalls;
  }

  // 静态方法，获取移动赋值操作调用次数
  static int getNumMoveAssignmentCalls() {
    return numMoveAssignmentCalls;
  }

  // 静态方法，获取拷贝赋值操作调用次数
  static int getNumCopyAssignmentCalls() {

    return numCopyAssignmentCalls;
  }
    return numCopyAssignmentCalls;
  }



    # 返回 numCopyAssignmentCalls 变量的值
    return numCopyAssignmentCalls;
  }



  friend bool operator==(const Constructable& c0, const Constructable& c1) {
    # 定义友元函数，用于比较两个 Constructable 对象是否相等
    return c0.getValue() == c1.getValue();
  }



  friend bool C10_UNUSED
  operator!=(const Constructable& c0, const Constructable& c1) {
    # 定义友元函数，用于比较两个 Constructable 对象是否不相等
    return c0.getValue() != c1.getValue();
  }
};

// 类静态变量定义
int Constructable::numConstructorCalls;
int Constructable::numCopyConstructorCalls;
int Constructable::numMoveConstructorCalls;
int Constructable::numDestructorCalls;
int Constructable::numAssignmentCalls;
int Constructable::numCopyAssignmentCalls;
int Constructable::numMoveAssignmentCalls;

// 不可拷贝结构体
struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(NonCopyable&&) noexcept = default;
  NonCopyable& operator=(NonCopyable&&) noexcept = default;

  NonCopyable(const NonCopyable&) = delete; // 禁用拷贝构造函数
  NonCopyable& operator=(const NonCopyable&) = delete; // 禁用拷贝赋值运算符
};

// 编译测试函数
C10_USED void CompileTest() {
  SmallVector<NonCopyable, 0> V; // 创建一个空的 SmallVector，其中元素类型为 NonCopyable
  V.resize(42); // 调整 SmallVector 的大小为 42
}

// SmallVectorTestBase 类定义
class SmallVectorTestBase : public testing::Test {
 protected:
  void SetUp() override {
    Constructable::reset(); // 在测试开始前重置 Constructable 类的计数
  }

  template <typename VectorT>
  void assertEmpty(VectorT& v) {
    // 大小测试
    EXPECT_EQ(0u, v.size()); // 断言 vector 的大小为 0
    EXPECT_TRUE(v.empty()); // 断言 vector 是否为空

    // 迭代器测试
    EXPECT_TRUE(v.begin() == v.end()); // 断言 vector 的起始迭代器等于结束迭代器
  }

  // 断言 vector 包含特定顺序的值
  template <typename VectorT>
  void assertValuesInOrder(VectorT& v, size_t size, ...) {
    EXPECT_EQ(size, v.size()); // 断言 vector 的大小与预期值相等

    va_list ap;
    va_start(ap, size);
    for (size_t i = 0; i < size; ++i) {
      int value = va_arg(ap, int); // 从可变参数列表中获取下一个 int 值
      EXPECT_EQ(value, v[i].getValue()); // 断言 vector 中第 i 个元素的值与预期值相等
    }

    va_end(ap);
  }

  // 生成一个初始化 vector 的值序列
  template <typename VectorT>
  void makeSequence(VectorT& v, int start, int end) {
    for (int i = start; i <= end; ++i) {
      v.push_back(Constructable(i)); // 向 vector 中添加一个值为 i 的 Constructable 对象
    }
  }
};

// SmallVectorTest 类模板
template <typename VectorT>
class SmallVectorTest : public SmallVectorTestBase {
 protected:
  VectorT theVector; // 测试中使用的主要 vector 对象
  VectorT otherVector; // 测试中使用的次要 vector 对象
};

// SmallVectorTest 类型别名
typedef ::testing::Types<
    SmallVector<Constructable, 0>,
    SmallVector<Constructable, 1>,
    SmallVector<Constructable, 2>,
    SmallVector<Constructable, 4>,
    SmallVector<Constructable, 5>>
    SmallVectorTestTypes;

// 使用 TYPED_TEST_SUITE 定义 SmallVectorTest 的测试套件
TYPED_TEST_SUITE(SmallVectorTest, SmallVectorTestTypes, );

// 构造函数测试
TYPED_TEST(SmallVectorTest, ConstructorNonIterTest) {
  SCOPED_TRACE("ConstructorTest");
  this->theVector = SmallVector<Constructable, 2>(2, 2); // 使用 (2, 2) 构造一个 SmallVector 对象
  this->assertValuesInOrder(this->theVector, 2u, 2, 2); // 断言 theVector 中包含值为 (2, 2) 的两个元素
}

// 构造函数测试
TYPED_TEST(SmallVectorTest, ConstructorIterTest) {
  SCOPED_TRACE("ConstructorTest");
  int arr[] = {1, 2, 3};
  this->theVector =
      SmallVector<Constructable, 4>(std::begin(arr), std::end(arr)); // 使用数组 arr 初始化一个 SmallVector 对象
  this->assertValuesInOrder(this->theVector, 3u, 1, 2, 3); // 断言 theVector 中包含值为 (1, 2, 3) 的三个元素
}

// 空 vector 测试
TYPED_TEST(SmallVectorTest, EmptyVectorTest) {
  SCOPED_TRACE("EmptyVectorTest");
  this->assertEmpty(this->theVector); // 断言 theVector 是空的
  EXPECT_TRUE(this->theVector.rbegin() == this->theVector.rend()); // 断言 theVector 的反向迭代器等于结束迭代器
  EXPECT_EQ(0, Constructable::getNumConstructorCalls()); // 断言 Constructable 的构造函数调用次数为 0
  EXPECT_EQ(0, Constructable::getNumDestructorCalls()); // 断言 Constructable 的析构函数调用次数为 0
}

// 简单的插入和删除测试
// 在 SmallVectorTest 类型的测试套件中，定义了 PushPopTest 测试用例
TYPED_TEST(SmallVectorTest, PushPopTest) {
  // 添加测试跟踪信息
  SCOPED_TRACE("PushPopTest");

  // 判断是否需要扩容，当前向量容量小于 3 则为真
  bool RequiresGrowth = this->theVector.capacity() < 3;

  // 向向量末尾推入一个元素
  this->theVector.push_back(Constructable(1));

  // 尺寸测试
  this->assertValuesInOrder(this->theVector, 1u, 1);
  // 验证向量不为空
  EXPECT_FALSE(this->theVector.begin() == this->theVector.end());
  // 验证向量非空
  EXPECT_FALSE(this->theVector.empty());

  // 再推入一个元素
  this->theVector.push_back(Constructable(2));
  // 验证元素顺序
  this->assertValuesInOrder(this->theVector, 2u, 1, 2);

  // 在开头插入元素。为避免从 this->theVector[1] 使引用失效，预留空间。
  this->theVector.reserve(this->theVector.size() + 1);
  this->theVector.insert(this->theVector.begin(), this->theVector[1]);
  // 验证元素顺序
  this->assertValuesInOrder(this->theVector, 3u, 2, 1, 2);

  // 弹出一个元素
  this->theVector.pop_back();
  // 验证元素顺序
  this->assertValuesInOrder(this->theVector, 2u, 2, 1);

  // 弹出剩余的元素
  this->theVector.pop_back_n(2);
  // 验证向量为空
  this->assertEmpty(this->theVector);

  // 检查构造函数调用次数。每个列表元素应为 2 次调用构造函数，
  // push_back 的参数一次，insert 的参数一次，列表元素本身一次。
  if (!RequiresGrowth) {
    EXPECT_EQ(5, Constructable::getNumConstructorCalls());
    EXPECT_EQ(5, Constructable::getNumDestructorCalls());
  } else {
    // 如果向量需要扩容，只能确保下限
    EXPECT_LE(5, Constructable::getNumConstructorCalls());
    EXPECT_EQ(
        Constructable::getNumConstructorCalls(),
        Constructable::getNumDestructorCalls());
  }
}

// ClearTest 清空测试用例
TYPED_TEST(SmallVectorTest, ClearTest) {
  // 添加测试跟踪信息
  SCOPED_TRACE("ClearTest");

  // 预留至少两个空间
  this->theVector.reserve(2);
  // 生成序列元素 1 到 2
  this->makeSequence(this->theVector, 1, 2);
  // 清空向量
  this->theVector.clear();

  // 验证向量为空
  this->assertEmpty(this->theVector);
  // 验证构造函数调用次数为 4
  EXPECT_EQ(4, Constructable::getNumConstructorCalls());
  // 验证析构函数调用次数为 4
  EXPECT_EQ(4, Constructable::getNumDestructorCalls());
}

// ResizeShrinkTest 缩小尺寸测试用例
TYPED_TEST(SmallVectorTest, ResizeShrinkTest) {
  // 添加测试跟踪信息
  SCOPED_TRACE("ResizeShrinkTest");

  // 预留至少三个空间
  this->theVector.reserve(3);
  // 生成序列元素 1 到 3
  this->makeSequence(this->theVector, 1, 3);
  // 将向量尺寸调整为 1
  this->theVector.resize(1);

  // 验证元素顺序
  this->assertValuesInOrder(this->theVector, 1u, 1);
  // 验证构造函数调用次数为 6
  EXPECT_EQ(6, Constructable::getNumConstructorCalls());
  // 验证析构函数调用次数为 5
  EXPECT_EQ(5, Constructable::getNumDestructorCalls());
}

// ResizeGrowTest 扩大尺寸测试用例
TYPED_TEST(SmallVectorTest, ResizeGrowTest) {
  // 添加测试跟踪信息
  SCOPED_TRACE("ResizeGrowTest");

  // 将向量尺寸调整为 2
  this->theVector.resize(2);

  // 验证构造函数调用次数为 2
  EXPECT_EQ(2, Constructable::getNumConstructorCalls());
  // 验证析构函数调用次数为 0
  EXPECT_EQ(0, Constructable::getNumDestructorCalls());
  // 验证向量尺寸为 2
  EXPECT_EQ(2u, this->theVector.size());
}
// 使用 TypeParam 定义的类型进行测试，测试 SmallVector 的 resize 方法是否正确处理带有元素的情况
TYPED_TEST(SmallVectorTest, ResizeWithElementsTest) {
  // 将向量大小调整为 2
  this->theVector.resize(2);

  // 重置 Constructable 类的调用计数
  Constructable::reset();

  // 将向量大小再次调整为 4
  this->theVector.resize(4);

  // 检查构造函数的调用次数是否为 2 或 4
  size_t Ctors = Constructable::getNumConstructorCalls();
  EXPECT_TRUE(Ctors == 2 || Ctors == 4);

  // 检查移动构造函数的调用次数是否为 0 或 2
  size_t MoveCtors = Constructable::getNumMoveConstructorCalls();
  EXPECT_TRUE(MoveCtors == 0 || MoveCtors == 2);

  // 检查析构函数的调用次数是否为 0 或 2
  size_t Dtors = Constructable::getNumDestructorCalls();
  EXPECT_TRUE(Dtors == 0 || Dtors == 2);
}

// 使用填充值进行 resize 测试
TYPED_TEST(SmallVectorTest, ResizeFillTest) {
  SCOPED_TRACE("ResizeFillTest");

  // 将向量大小调整为 3，并使用 Constructable(77) 进行填充
  this->theVector.resize(3, Constructable(77));

  // 断言向量中的值是否按顺序包含 3 个 77
  this->assertValuesInOrder(this->theVector, 3u, 77, 77, 77);
}

// 对 resize_for_overwrite 方法进行测试
TEST(SmallVectorTest, ResizeForOverwrite) {
  {
    // 使用堆分配的存储空间
    SmallVector<unsigned, 0> V;
    V.push_back(5U);
    V.pop_back();
    V.resize_for_overwrite(V.size() + 1U);

    // 检查向量最后一个元素是否为 5
    EXPECT_EQ(5U, V.back());

    V.pop_back();
    V.resize(V.size() + 1);

    // 检查向量最后一个元素是否为 0
    EXPECT_EQ(0U, V.back());
  }
  {
    // 使用内联存储空间
    SmallVector<unsigned, 2> V;
    V.push_back(5U);
    V.pop_back();
    V.resize_for_overwrite(V.size() + 1U);

    // 检查向量最后一个元素是否为 5
    EXPECT_EQ(5U, V.back());

    V.pop_back();
    V.resize(V.size() + 1);

    // 检查向量最后一个元素是否为 0
    EXPECT_EQ(0U, V.back());
  }
}

// 测试溢出固定大小的情况
TYPED_TEST(SmallVectorTest, OverflowTest) {
  SCOPED_TRACE("OverflowTest");

  // 向向量中推入超过固定大小的元素
  this->makeSequence(this->theVector, 1, 10);

  // 检查向量的大小和值
  EXPECT_EQ(10u, this->theVector.size());
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(i + 1, this->theVector[i].getValue());
  }

  // 将向量大小调整回固定大小
  this->theVector.resize(1);

  // 断言向量中的值是否为 1
  this->assertValuesInOrder(this->theVector, 1u, 1);
}

// 迭代测试
TYPED_TEST(SmallVectorTest, IterationTest) {
  // 使用 makeSequence 方法生成序列
  this->makeSequence(this->theVector, 1, 2);

  // 前向迭代
  typename TypeParam::iterator it = this->theVector.begin();
  EXPECT_TRUE(*it == this->theVector.front());
  EXPECT_TRUE(*it == this->theVector[0]);
  EXPECT_EQ(1, it->getValue());
  ++it;
  EXPECT_TRUE(*it == this->theVector[1]);
  EXPECT_TRUE(*it == this->theVector.back());
  EXPECT_EQ(2, it->getValue());
  ++it;
  EXPECT_TRUE(it == this->theVector.end());
  --it;
  EXPECT_TRUE(*it == this->theVector[1]);
  EXPECT_EQ(2, it->getValue());
  --it;
  EXPECT_TRUE(*it == this->theVector[0]);
  EXPECT_EQ(1, it->getValue());

  // 反向迭代
  typename TypeParam::reverse_iterator rit = this->theVector.rbegin();
  EXPECT_TRUE(*rit == this->theVector[1]);
  EXPECT_EQ(2, rit->getValue());
  ++rit;
  EXPECT_TRUE(*rit == this->theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  ++rit;
  EXPECT_TRUE(rit == this->theVector.rend());
  --rit;
  EXPECT_TRUE(*rit == this->theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  --rit;
  EXPECT_TRUE(*rit == this->theVector[1]);
  EXPECT_EQ(2, rit->getValue());
}
// 定义了一个类型为 TypeParam 的测试用例 SmallVectorTest 中的 SwapTest 测试
TYPED_TEST(SmallVectorTest, SwapTest) {
  // 添加一个跟踪信息 "SwapTest"
  SCOPED_TRACE("SwapTest");

  // 使用 makeSequence 方法生成序列给 this->theVector，包含元素 1, 2
  this->makeSequence(this->theVector, 1, 2);
  // 交换 this->theVector 和 this->otherVector 的内容
  std::swap(this->theVector, this->otherVector);

  // 断言 this->theVector 已经为空
  this->assertEmpty(this->theVector);
  // 断言 this->otherVector 中的值按顺序为 2, 1, 2
  this->assertValuesInOrder(this->otherVector, 2u, 1, 2);
}

// Append test
TYPED_TEST(SmallVectorTest, AppendTest) {
  // 添加一个跟踪信息 "AppendTest"
  SCOPED_TRACE("AppendTest");

  // 使用 makeSequence 方法生成序列给 this->otherVector，包含元素 2, 3
  this->makeSequence(this->otherVector, 2, 3);

  // 向 this->theVector 中添加一个值为 1 的 Constructable 对象
  this->theVector.push_back(Constructable(1));
  // 将 this->otherVector 中的元素追加到 this->theVector 中
  this->theVector.append(this->otherVector.begin(), this->otherVector.end());

  // 断言 this->theVector 中的值按顺序为 1, 2, 3
  this->assertValuesInOrder(this->theVector, 3u, 1, 2, 3);
}

// Append repeated test
TYPED_TEST(SmallVectorTest, AppendRepeatedTest) {
  // 添加一个跟踪信息 "AppendRepeatedTest"
  SCOPED_TRACE("AppendRepeatedTest");

  // 向 this->theVector 中添加一个值为 1 的 Constructable 对象
  this->theVector.push_back(Constructable(1));
  // 向 this->theVector 中追加两个值为 77 的 Constructable 对象
  this->theVector.append(2, Constructable(77));

  // 断言 this->theVector 中的值按顺序为 1, 77, 77
  this->assertValuesInOrder(this->theVector, 3u, 1, 77, 77);
}

// Append test
TYPED_TEST(SmallVectorTest, AppendNonIterTest) {
  // 添加一个跟踪信息 "AppendRepeatedTest"
  SCOPED_TRACE("AppendRepeatedTest");

  // 向 this->theVector 中添加一个值为 1 的 Constructable 对象
  this->theVector.push_back(Constructable(1));
  // 向 this->theVector 中追加两个值为 7 的对象，这里使用的是基础类型 int
  this->theVector.append(2, 7);

  // 断言 this->theVector 中的值按顺序为 1, 7, 7
  this->assertValuesInOrder(this->theVector, 3u, 1, 7, 7);
}

// 定义一个名为 output_iterator 的结构体
struct output_iterator {
  typedef std::output_iterator_tag iterator_category;
  typedef int value_type;
  typedef int difference_type;
  typedef value_type* pointer;
  typedef value_type& reference;
  // 将 output_iterator 转换为 int 类型返回值 2
  operator int() {
    return 2;
  }
  // 将 output_iterator 转换为 Constructable 类型返回值 7
  operator Constructable() {
    return 7;
  }
};

// Append repeated test using non-forward iterator
TYPED_TEST(SmallVectorTest, AppendRepeatedNonForwardIterator) {
  // 添加一个跟踪信息 "AppendRepeatedTest"
  SCOPED_TRACE("AppendRepeatedTest");

  // 向 this->theVector 中添加一个值为 1 的 Constructable 对象
  this->theVector.push_back(Constructable(1));
  // 使用 output_iterator 作为迭代器，将两个 output_iterator 的结果追加到 this->theVector 中
  this->theVector.append(output_iterator(), output_iterator());

  // 断言 this->theVector 中的值按顺序为 1, 7, 7
  this->assertValuesInOrder(this->theVector, 3u, 1, 7, 7);
}

// Append SmallVector test
TYPED_TEST(SmallVectorTest, AppendSmallVector) {
  // 添加一个跟踪信息 "AppendSmallVector"
  SCOPED_TRACE("AppendSmallVector");

  // 创建一个包含值为 7, 7 的 SmallVector<Constructable, 3> 对象 otherVector
  SmallVector<Constructable, 3> otherVector = {7, 7};
  // 向 this->theVector 中添加一个值为 1 的 Constructable 对象
  this->theVector.push_back(Constructable(1));
  // 将 otherVector 中的元素追加到 this->theVector 中
  this->theVector.append(otherVector);

  // 断言 this->theVector 中的值按顺序为 1, 7, 7
  this->assertValuesInOrder(this->theVector, 3u, 1, 7, 7);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignTest) {
  // 添加一个跟踪信息 "AssignTest"
  SCOPED_TRACE("AssignTest");

  // 向 this->theVector 中添加一个值为 1 的 Constructable 对象
  this->theVector.push_back(Constructable(1));
  // 将 this->theVector 中的所有元素替换为两个值为 77 的 Constructable 对象
  this->theVector.assign(2, Constructable(77));

  // 断言 this->theVector 中的值按顺序为 77, 77
  this->assertValuesInOrder(this->theVector, 2u, 77, 77);
}

// Assign range test
TYPED_TEST(SmallVectorTest, AssignRangeTest) {
  // 添加一个跟踪信息 "AssignTest"
  SCOPED_TRACE("AssignTest");

  // 向 this->theVector 中添加一个值为 1 的 Constructable 对象
  this->theVector.push_back(Constructable(1));
  // 创建一个数组 arr 包含值 1, 2, 3，将 arr 的内容赋值给 this->theVector
  int arr[] = {1, 2, 3};
  this->theVector.assign(std::begin(arr), std::end(arr));

  // 断言 this->theVector 中的值按顺序为 1, 2, 3
  this->assertValuesInOrder(this->theVector, 3u, 1, 2, 3);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignNonIterTest) {
  // 添加一个跟踪信息 "AssignTest"
  SCOPED_TRACE("AssignTest");

  // 向 this->theVector 中添加一个值为 1 的 Constructable 对象
  this->theVector.push_back(Constructable(1));
  // 将 this->theVector 中的所有元素替换为两个值为 7 的对象，这里使用的是基础类型 int
  this->theVector.assign(2, 7);

  // 断言 this->theVector 中的值按顺序为 7, 7
  this->assertValuesInOrder(this->theVector, 2u, 7, 7);
}
//`
TYPED_TEST(SmallVectorTest, AssignSmallVector) {
  SCOPED_TRACE("AssignSmallVector");  # 设置一个追踪点，便于调试和日志记录，标记测试的名称

  SmallVector<Constructable, 3> otherVector = {7, 7};  # 创建一个 SmallVector，包含两个元素 7
  this->theVector.push_back(Constructable(1));  # 向 theVector 添加一个 Constructable 对象，初始化为 1
  this->theVector.assign(otherVector);  # 将 otherVector 的内容赋值给 this->theVector
  this->assertValuesInOrder(this->theVector, 2u, 7, 7);  # 验证 this->theVector 中的元素顺序是否为 2 个 7
}

// Move-assign test
TYPED_TEST(SmallVectorTest, MoveAssignTest) {
  SCOPED_TRACE("MoveAssignTest");  # 设置一个追踪点，便于调试和日志记录，标记测试的名称

  this->theVector.reserve(4);  # 为 this->theVector 预留空间，容量为 4
  this->theVector.push_back(Constructable(1));  # 向 theVector 添加一个 Constructable 对象，初始化为 1

  this->otherVector.push_back(Constructable(2));  # 向 otherVector 添加一个 Constructable 对象，初始化为 2
  this->otherVector.push_back(Constructable(3));  # 向 otherVector 添加一个 Constructable 对象，初始化为 3

  this->theVector = std::move(this->otherVector);  # 将 otherVector 的内容移动赋值给 this->theVector

  this->assertValuesInOrder(this->theVector, 2u, 2, 3);  # 验证 this->theVector 中的元素顺序是否为 2 和 3

  this->otherVector.clear();  # 清空 otherVector
  EXPECT_EQ(
      Constructable::getNumConstructorCalls() - 2,
      Constructable::getNumDestructorCalls());  # 检查构造和析构的次数是否正确

  this->theVector.clear();  # 清空 this->theVector
  EXPECT_EQ(
      Constructable::getNumConstructorCalls(),
      Constructable::getNumDestructorCalls());  # 检查构造和析构的次数是否一致
}

// Erase a single element
TYPED_TEST(SmallVectorTest, EraseTest) {
  SCOPED_TRACE("EraseTest");  # 设置一个追踪点，便于调试和日志记录，标记测试的名称

  this->makeSequence(this->theVector, 1, 3);  # 生成一个包含 1 到 3 的序列并赋值给 theVector
  auto& theVector = this->theVector;  # 创建 theVector 的引用别名
  this->theVector.erase(theVector.begin());  # 删除 theVector 中的第一个元素
  this->assertValuesInOrder(this->theVector, 2u, 2, 3);  # 验证 theVector 中的元素顺序是否为 2 和 3
}

// Erase a range of elements
TYPED_TEST(SmallVectorTest, EraseRangeTest) {
  SCOPED_TRACE("EraseRangeTest");  # 设置一个追踪点，便于调试和日志记录，标记测试的名称

  this->makeSequence(this->theVector, 1, 3);  # 生成一个包含 1 到 3 的序列并赋值给 theVector
  auto& theVector = this->theVector;  # 创建 theVector 的引用别名
  this->theVector.erase(theVector.begin(), theVector.begin() + 2);  # 删除 theVector 中从第一个到第二个元素的范围
  this->assertValuesInOrder(this->theVector, 1u, 3);  # 验证 theVector 中的元素顺序是否为 3
}

// Insert a single element.
TYPED_TEST(SmallVectorTest, InsertTest) {
  SCOPED_TRACE("InsertTest");  # 设置一个追踪点，便于调试和日志记录，标记测试的名称

  this->makeSequence(this->theVector, 1, 3);  # 生成一个包含 1 到 3 的序列并赋值给 theVector
  typename TypeParam::iterator I =
      this->theVector.insert(this->theVector.begin() + 1, Constructable(77));  # 在 theVector 的第 1 个位置插入一个 Constructable 对象，初始化为 77
  EXPECT_EQ(this->theVector.begin() + 1, I);  # 验证插入后的迭代器是否正确
  this->assertValuesInOrder(this->theVector, 4u, 1, 77, 2, 3);  # 验证 theVector 中的元素顺序是否为 1、77、2、3
}

// Insert a copy of a single element.
TYPED_TEST(SmallVectorTest, InsertCopy) {
  SCOPED_TRACE("InsertTest");  # 设置一个追踪点，便于调试和日志记录，标记测试的名称

  this->makeSequence(this->theVector, 1, 3);  # 生成一个包含 1 到 3 的序列并赋值给 theVector
  Constructable C(77);  # 创建一个 Constructable 对象，初始化为 77
  typename TypeParam::iterator I =
      this->theVector.insert(this->theVector.begin() + 1, C);  # 在 theVector 的第 1 个位置插入一个 Constructable 对象 C 的副本
  EXPECT_EQ(this->theVector.begin() + 1, I);  # 验证插入后的迭代器是否正确
  this->assertValuesInOrder(this->theVector, 4u, 1, 77, 2, 3);  # 验证 theVector 中的元素顺序是否为 1、77、2、3
}

// Insert repeated elements.
TYPED_TEST(SmallVectorTest, InsertRepeatedTest) {
  // 定义测试名称
  SCOPED_TRACE("InsertRepeatedTest");

  // 创建序列并初始化容器
  this->makeSequence(this->theVector, 1, 4);
  // 重置构造对象的状态
  Constructable::reset();
  // 在容器的第二个位置插入两个值为 Constructable(16) 的对象，并返回插入的位置迭代器
  auto I =
      this->theVector.insert(this->theVector.begin() + 1, 2, Constructable(16));
  // 检查移动构造函数调用次数，应为2或6次
  // FIXME: 这种方式效率低下，不应该在新分配的空间中移动对象，然后再移动它们
  EXPECT_TRUE(
      Constructable::getNumMoveConstructorCalls() == 2 ||
      Constructable::getNumMoveConstructorCalls() == 6);
  // 移动赋值操作次数，用于移动下一个两个对象以腾出空间
  EXPECT_EQ(1, Constructable::getNumMoveAssignmentCalls());
  // 复制赋值构造次数，用于从参数中复制两个新元素
  EXPECT_EQ(2, Constructable::getNumCopyAssignmentCalls());
  // 没有任何拷贝构造函数调用
  EXPECT_EQ(0, Constructable::getNumCopyConstructorCalls());
  // 验证插入操作返回的迭代器位置
  EXPECT_EQ(this->theVector.begin() + 1, I);
  // 验证容器中的值是否按顺序排列
  this->assertValuesInOrder(this->theVector, 6u, 1, 16, 16, 2, 3, 4);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedNonIterTest) {
  // 定义测试名称
  SCOPED_TRACE("InsertRepeatedTest");

  // 创建序列并初始化容器
  this->makeSequence(this->theVector, 1, 4);
  // 重置构造对象的状态
  Constructable::reset();
  // 在容器的第二个位置插入两个值为 7 的对象，并返回插入的位置迭代器
  auto I = this->theVector.insert(this->theVector.begin() + 1, 2, 7);
  // 验证插入操作返回的迭代器位置
  EXPECT_EQ(this->theVector.begin() + 1, I);
  // 验证容器中的值是否按顺序排列
  this->assertValuesInOrder(this->theVector, 6u, 1, 7, 7, 2, 3, 4);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedAtEndTest) {
  // 定义测试名称
  SCOPED_TRACE("InsertRepeatedTest");

  // 创建序列并初始化容器
  this->makeSequence(this->theVector, 1, 4);
  // 重置构造对象的状态
  Constructable::reset();
  // 在容器末尾插入两个值为 Constructable(16) 的对象，并返回插入的位置迭代器
  auto I = this->theVector.insert(this->theVector.end(), 2, Constructable(16));
  // 只将它们复制构造到新分配的空间中
  EXPECT_EQ(2, Constructable::getNumCopyConstructorCalls());
  // 如果需要重新分配，移动所有内容
  EXPECT_TRUE(
      Constructable::getNumMoveConstructorCalls() == 0 ||
      Constructable::getNumMoveConstructorCalls() == 4);
  // 没有任何其他移动或复制赋值操作
  EXPECT_EQ(0, Constructable::getNumCopyAssignmentCalls());
  EXPECT_EQ(0, Constructable::getNumMoveAssignmentCalls());

  // 验证插入操作返回的迭代器位置
  EXPECT_EQ(this->theVector.begin() + 4, I);
  // 验证容器中的值是否按顺序排列
  this->assertValuesInOrder(this->theVector, 6u, 1, 2, 3, 4, 16, 16);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedEmptyTest) {
  // 定义测试名称
  SCOPED_TRACE("InsertRepeatedTest");

  // 创建序列并初始化容器
  this->makeSequence(this->theVector, 10, 15);

  // 空插入测试
  // 在容器末尾插入0个值为 Constructable(42) 的对象，预期返回末尾迭代器位置
  EXPECT_EQ(
      this->theVector.end(),
      this->theVector.insert(this->theVector.end(), 0, Constructable(42)));
  // 在容器的第二个位置插入0个值为 Constructable(42) 的对象，预期返回插入的位置迭代器
  EXPECT_EQ(
      this->theVector.begin() + 1,
      this->theVector.insert(
          this->theVector.begin() + 1, 0, Constructable(42)));
}
// 在 SmallVectorTest 测试类中定义 InsertRangeTest 测试函数
TYPED_TEST(SmallVectorTest, InsertRangeTest) {
  // 设置测试的跟踪信息为 "InsertRangeTest"
  SCOPED_TRACE("InsertRangeTest");

  // 创建包含三个 Constructable 对象的数组 Arr，每个对象的值为 77
  Constructable Arr[3] = {
      Constructable(77), Constructable(77), Constructable(77)};

  // 在 this->theVector 中生成序列 [1, 2, 3]
  this->makeSequence(this->theVector, 1, 3);
  // 重置 Constructable 类的静态成员，用于跟踪构造函数和赋值操作的调用次数
  Constructable::reset();
  // 在 this->theVector 的第二个位置插入 Arr 数组中的三个元素，并返回插入点迭代器
  auto I = this->theVector.insert(this->theVector.begin() + 1, Arr, Arr + 3);
  // 移动构造新分配空间中的前三个元素
  // 可能先将整个序列移动到新的空间中
  // FIXME: 这种方式效率不高，不应该将元素移动到新分配的空间，然后再进行移动操作，应该只有 2 或 3 次移动构造操作
  EXPECT_TRUE(
      Constructable::getNumMoveConstructorCalls() == 2 ||
      Constructable::getNumMoveConstructorCalls() == 5);
  // 将新插入的两个元素复制赋值到已有的空间中
  EXPECT_EQ(2, Constructable::getNumCopyAssignmentCalls());
  // 在新分配的空间中复制构造第三个元素
  EXPECT_EQ(1, Constructable::getNumCopyConstructorCalls());
  // 验证插入操作返回的迭代器位置是否正确
  EXPECT_EQ(this->theVector.begin() + 1, I);
  // 验证 this->theVector 中的值是否按顺序为 [1, 77, 77, 77, 2, 3]
  this->assertValuesInOrder(this->theVector, 6u, 1, 77, 77, 77, 2, 3);
}

// 同上一个测试函数类似，但在尾部插入元素的测试
TYPED_TEST(SmallVectorTest, InsertRangeAtEndTest) {
  SCOPED_TRACE("InsertRangeTest");

  Constructable Arr[3] = {
      Constructable(77), Constructable(77), Constructable(77)};

  // 在 this->theVector 中生成序列 [1, 2, 3]
  this->makeSequence(this->theVector, 1, 3);

  // 在尾部插入元素
  Constructable::reset();
  auto I = this->theVector.insert(this->theVector.end(), Arr, Arr + 3);
  // 在新分配的空间中复制构造三个元素
  EXPECT_EQ(3, Constructable::getNumCopyConstructorCalls());
  // 不复制或移动其它元素
  EXPECT_EQ(0, Constructable::getNumCopyAssignmentCalls());
  // 可能会重新分配内存，导致所有元素都被移动到新的缓冲区
  EXPECT_TRUE(
      Constructable::getNumMoveConstructorCalls() == 0 ||
      Constructable::getNumMoveConstructorCalls() == 3);
  EXPECT_EQ(0, Constructable::getNumMoveAssignmentCalls());
  // 验证插入操作返回的迭代器位置是否正确
  EXPECT_EQ(this->theVector.begin() + 3, I);
  // 验证 this->theVector 中的值是否按顺序为 [1, 2, 3, 77, 77, 77]
  this->assertValuesInOrder(this->theVector, 6u, 1, 2, 3, 77, 77, 77);
}

// 在空范围插入元素的测试
TYPED_TEST(SmallVectorTest, InsertEmptyRangeTest) {
  SCOPED_TRACE("InsertRangeTest");

  // 在 this->theVector 中生成序列 [1, 2, 3]
  this->makeSequence(this->theVector, 1, 3);

  // 空范围插入
  EXPECT_EQ(
      this->theVector.end(),
      this->theVector.insert(
          this->theVector.end(),
          this->theVector.begin(),
          this->theVector.begin()));
  EXPECT_EQ(
      this->theVector.begin() + 1,
      this->theVector.insert(
          this->theVector.begin() + 1,
          this->theVector.begin(),
          this->theVector.begin()));
}

// 比较测试
// 使用 TYPED_TEST 宏定义测试案例 SmallVectorTest，类型为 ComparisonTest
TYPED_TEST(SmallVectorTest, ComparisonTest) {
  // 添加跟踪信息 "ComparisonTest"
  SCOPED_TRACE("ComparisonTest");

  // 在 this->theVector 中创建序列 [1, 2, 3]
  this->makeSequence(this->theVector, 1, 3);
  // 在 this->otherVector 中创建序列 [1, 2, 3]
  this->makeSequence(this->otherVector, 1, 3);

  // 检查 this->theVector 是否等于 this->otherVector
  EXPECT_TRUE(this->theVector == this->otherVector);
  // 检查 this->theVector 是否不等于 this->otherVector
  EXPECT_FALSE(this->theVector != this->otherVector);

  // 清空 this->otherVector
  this->otherVector.clear();
  // 在 this->otherVector 中创建序列 [2, 3, 4]
  this->makeSequence(this->otherVector, 2, 4);

  // 检查 this->theVector 是否不等于 this->otherVector
  EXPECT_FALSE(this->theVector == this->otherVector);
  // 检查 this->theVector 是否等于 this->otherVector
  EXPECT_TRUE(this->theVector != this->otherVector);
}

// 常量向量测试
TYPED_TEST(SmallVectorTest, ConstVectorTest) {
  // 创建常量向量 constVector
  const TypeParam constVector;

  // 检查 constVector 的大小是否为 0
  EXPECT_EQ(0u, constVector.size());
  // 检查 constVector 是否为空
  EXPECT_TRUE(constVector.empty());
  // 检查 constVector 的起始迭代器是否等于结束迭代器
  EXPECT_TRUE(constVector.begin() == constVector.end());
}

// 直接数组访问测试
TYPED_TEST(SmallVectorTest, DirectVectorTest) {
  // 检查 this->theVector 的大小是否为 0
  EXPECT_EQ(0u, this->theVector.size());
  // 预留 this->theVector 的容量至少为 4
  this->theVector.reserve(4);
  // 检查 this->theVector 的容量是否至少为 4
  EXPECT_LE(4u, this->theVector.capacity());
  // 检查 Constructable 类的构造函数调用次数是否为 0
  EXPECT_EQ(0, Constructable::getNumConstructorCalls());
  // 向 this->theVector 中添加元素 1, 2, 3, 4
  this->theVector.push_back(1);
  this->theVector.push_back(2);
  this->theVector.push_back(3);
  this->theVector.push_back(4);
  // 检查 this->theVector 的大小是否为 4
  EXPECT_EQ(4u, this->theVector.size());
  // 检查 Constructable 类的构造函数调用次数是否为 8
  EXPECT_EQ(8, Constructable::getNumConstructorCalls());
  // 检查 this->theVector 中第一个元素的值是否为 1
  EXPECT_EQ(1, this->theVector[0].getValue());
  // 检查 this->theVector 中第二个元素的值是否为 2
  EXPECT_EQ(2, this->theVector[1].getValue());
  // 检查 this->theVector 中第三个元素的值是否为 3
  EXPECT_EQ(3, this->theVector[2].getValue());
  // 检查 this->theVector 中第四个元素的值是否为 4
  EXPECT_EQ(4, this->theVector[3].getValue());
}

// 迭代器测试
TYPED_TEST(SmallVectorTest, IteratorTest) {
  // 创建整数链表 L
  std::list<int> L;
  // 将 L 的元素插入到 this->theVector 的末尾
  this->theVector.insert(this->theVector.end(), L.begin(), L.end());
}

// 模板类 DualSmallVectorsTest 的特化版本，使用 std::pair<VectorT1, VectorT2> 作为参数
template <typename VectorT1, typename VectorT2>
class DualSmallVectorsTest<std::pair<VectorT1, VectorT2>>
    : public SmallVectorTestBase {
 protected:
  VectorT1 theVector;
  VectorT2 otherVector;

  // 返回 SmallVector<T, N> 类型的内置元素个数 N
  template <typename T, unsigned N>
  static unsigned NumBuiltinElts(const SmallVector<T, N>&) {
    return N;
  }
};

// 定义测试类型 DualSmallVectorTestTypes
typedef ::testing::Types<
    // 小模式 -> 小模式
    std::pair<SmallVector<Constructable, 4>, SmallVector<Constructable, 4>>,
    // 小模式 -> 大模式
    std::pair<SmallVector<Constructable, 4>, SmallVector<Constructable, 2>>,
    // 大模式 -> 小模式
    std::pair<SmallVector<Constructable, 2>, SmallVector<Constructable, 4>>,
    // 大模式 -> 大模式
    std::pair<SmallVector<Constructable, 2>, SmallVector<Constructable, 2>>>
    DualSmallVectorTestTypes;

// 定义测试套件 DualSmallVectorsTest，使用 DualSmallVectorTestTypes 中的类型，并且类型为 DualSmallVectorsTest
TYPED_TEST_SUITE(DualSmallVectorsTest, DualSmallVectorTestTypes, );

// MoveAssignment 测试案例
TYPED_TEST(DualSmallVectorsTest, MoveAssignment) {
  // 添加跟踪信息 "MoveAssignTest-DualVectorTypes"
  SCOPED_TRACE("MoveAssignTest-DualVectorTypes");

  // 设置向量 this->theVector 包含四个元素
  for (int I = 0; I < 4; ++I)
    // 将构造函数返回值为 I 的对象加入到 otherVector 中
    this->otherVector.push_back(Constructable(I));

  // 获取 otherVector 的数据指针，并将其保存到 OrigDataPtr 中
  const Constructable* OrigDataPtr = this->otherVector.data();

  // 使用移动语义从 otherVector 移动数据到 theVector
  this->theVector = std::move(
      static_cast<SmallVectorImpl<Constructable>&>(this->otherVector));

  // 确保 theVector 中的值是按照预期顺序的
  this->assertValuesInOrder(this->theVector, 4u, 0, 1, 2, 3);

  // 确保构造函数和析构函数调用次数匹配。在清空 otherVector 后应该有两个活跃对象。
  this->otherVector.clear();
  EXPECT_EQ(
      Constructable::getNumConstructorCalls() - 4,
      Constructable::getNumDestructorCalls());

  // 如果源向量 (otherVector) 是小模式，断言我们只是移动了数据指针。
  EXPECT_TRUE(
      this->NumBuiltinElts(this->otherVector) == 4 ||
      this->theVector.data() == OrigDataPtr);

  // 现在不应该有任何活跃对象了。
  this->theVector.clear();
  EXPECT_EQ(
      Constructable::getNumConstructorCalls(),
      Constructable::getNumDestructorCalls());

  // 在整个过程中不应该有任何拷贝操作发生。
  EXPECT_EQ(Constructable::getNumCopyConstructorCalls(), 0);
}

struct notassignable {
  // 定义一个结构体，包含一个对整型的引用，但不允许赋值
  int& x;
  notassignable(int& x) : x(x) {} // 构造函数，初始化引用成员 x
};

TEST(SmallVectorCustomTest, NoAssignTest) {
  int x = 0; // 初始化整型变量 x 为 0
  SmallVector<notassignable, 2> vec; // 创建一个 SmallVector，存储 notassignable 结构体，最多容纳 2 个元素
  vec.push_back(notassignable(x)); // 向 vec 中添加一个 notassignable 对象，传入 x 的引用
  x = 42; // 修改 x 的值为 42
  EXPECT_EQ(x, vec.pop_back_val().x); // 断言：验证 x 的值与 vec 中弹出的元素的 x 值相等
}

struct MovedFrom {
  bool hasValue; // 布尔类型成员变量，表示对象是否有值
  MovedFrom() : hasValue(true) {} // 默认构造函数，初始化 hasValue 为 true
  MovedFrom(MovedFrom&& m) noexcept : hasValue(m.hasValue) { // 移动构造函数，接受一个右值引用
    m.hasValue = false; // 将原对象的 hasValue 设为 false
  }
  MovedFrom& operator=(MovedFrom&& m) noexcept { // 移动赋值运算符重载，接受一个右值引用
    hasValue = m.hasValue; // 将右值对象的 hasValue 赋值给当前对象的 hasValue
    m.hasValue = false; // 将右值对象的 hasValue 设为 false
    return *this; // 返回当前对象的引用
  }
};

TEST(SmallVectorTest, MidInsert) {
  SmallVector<MovedFrom, 3> v; // 创建一个 SmallVector，存储 MovedFrom 对象，最多容纳 3 个元素
  v.push_back(MovedFrom()); // 向 v 中添加一个 MovedFrom 的临时对象
  v.insert(v.begin(), MovedFrom()); // 在 v 的开头插入一个 MovedFrom 的临时对象
  for (MovedFrom& m : v)
    EXPECT_TRUE(m.hasValue); // 断言：验证 v 中所有 MovedFrom 对象的 hasValue 都为 true
}

enum EmplaceableArgState {
  EAS_Defaulted,
  EAS_Arg,
  EAS_LValue,
  EAS_RValue,
  EAS_Failure
};
template <int I>
struct EmplaceableArg {
  EmplaceableArgState State; // 枚举类型 State，表示 EmplaceableArg 的状态
  EmplaceableArg() : State(EAS_Defaulted) {} // 默认构造函数，初始化 State 为 EAS_Defaulted
  EmplaceableArg(EmplaceableArg&& X) noexcept
      : State(X.State == EAS_Arg ? EAS_RValue : EAS_Failure) {} // 移动构造函数，根据 X 的状态初始化 State
  EmplaceableArg(EmplaceableArg& X)
      : State(X.State == EAS_Arg ? EAS_LValue : EAS_Failure) {} // 拷贝构造函数，根据 X 的状态初始化 State

  explicit EmplaceableArg(bool) : State(EAS_Arg) {} // 显式构造函数，接受一个布尔值参数，初始化 State 为 EAS_Arg

  EmplaceableArg& operator=(EmplaceableArg&&) = delete; // 删除移动赋值运算符重载
  EmplaceableArg& operator=(const EmplaceableArg&) = delete; // 删除拷贝赋值运算符重载
};

enum EmplaceableState { ES_Emplaced, ES_Moved };
struct Emplaceable {
  EmplaceableArg<0> A0; // EmplaceableArg 类型成员 A0
  EmplaceableArg<1> A1; // EmplaceableArg 类型成员 A1
  EmplaceableArg<2> A2; // EmplaceableArg 类型成员 A2
  EmplaceableArg<3> A3; // EmplaceableArg 类型成员 A3
  EmplaceableState State; // 枚举类型 State，表示 Emplaceable 的状态

  Emplaceable() : State(ES_Emplaced) {} // 默认构造函数，初始化 State 为 ES_Emplaced

  template <class A0Ty>
  explicit Emplaceable(A0Ty&& A0)
      : A0(std::forward<A0Ty>(A0)), State(ES_Emplaced) {} // 模板构造函数，接受一个参数 A0，初始化 A0 和 State

  template <class A0Ty, class A1Ty>
  Emplaceable(A0Ty&& A0, A1Ty&& A1)
      : A0(std::forward<A0Ty>(A0)),
        A1(std::forward<A1Ty>(A1)),
        State(ES_Emplaced) {} // 模板构造函数，接受两个参数 A0 和 A1，初始化 A0、A1 和 State

  template <class A0Ty, class A1Ty, class A2Ty>
  Emplaceable(A0Ty&& A0, A1Ty&& A1, A2Ty&& A2)
      : A0(std::forward<A0Ty>(A0)),
        A1(std::forward<A1Ty>(A1)),
        A2(std::forward<A2Ty>(A2)),
        State(ES_Emplaced) {} // 模板构造函数，接受三个参数 A0、A1 和 A2，初始化 A0、A1、A2 和 State

  template <class A0Ty, class A1Ty, class A2Ty, class A3Ty>
  Emplaceable(A0Ty&& A0, A1Ty&& A1, A2Ty&& A2, A3Ty&& A3)
      : A0(std::forward<A0Ty>(A0)),
        A1(std::forward<A1Ty>(A1)),
        A2(std::forward<A2Ty>(A2)),
        A3(std::forward<A3Ty>(A3)),
        State(ES_Emplaced) {} // 模板构造函数，接受四个参数 A0、A1、A2 和 A3，初始化 A0、A1、A2、A3 和 State

  Emplaceable(Emplaceable&&) noexcept : State(ES_Moved) {} // 移动构造函数，将 State 初始化为 ES_Moved
  Emplaceable& operator=(Emplaceable&&) noexcept { // 移动赋值运算符重载，将 State 设置为 ES_Moved
    State = ES_Moved;
    return *this;
  }

  Emplaceable(const Emplaceable&) = delete; // 删除拷贝构造函数
  Emplaceable& operator=(const Emplaceable&) = delete; // 删除拷贝赋值运算符重载
};

TEST(SmallVectorTest, EmplaceBack) {
  EmplaceableArg<0> A0(true); // 创建 EmplaceableArg<0> 对象，传入 true
  EmplaceableArg<1> A1(true); // 创建 EmplaceableArg<1> 对象，传入 true
  EmplaceableArg<2> A2(true); // 创建 EmplaceableArg<2> 对象，传入 true
  EmplaceableArg<3> A3(true); // 创建 EmplaceableArg<3> 对象，传入 true
  {
    SmallVector<Emplaceable, 3> V; // 创建一个 SmallVector，存储 Emplaceable 对象，最多容纳 3 个元素
    Emplaceable& back = V.emplace_back(); // 使用 emplace_back 在 V 中构造一个 Emplaceable 对象，并返回对该对象的引用
    EXPECT_TRUE(&back == &V.back()); // 断言：验证 back 是 V 的最后一个元素的引用
    // 确认容器 V 中元素个数为 1
    EXPECT_TRUE(V.size() == 1);
    // 确认容器 V 的最后一个元素状态为 ES_Emplaced
    EXPECT_TRUE(back.State == ES_Emplaced);
    // 确认容器 V 的最后一个元素的 A0 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A0.State == EAS_Defaulted);
    // 确认容器 V 的最后一个元素的 A1 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A1.State == EAS_Defaulted);
    // 确认容器 V 的最后一个元素的 A2 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    // 确认容器 V 的最后一个元素的 A3 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    // 创建容器 V，最大容量为 3
    SmallVector<Emplaceable, 3> V;
    // 在容器 V 的末尾构造一个 Emplaceable 对象，并将 A0 移动到其中
    Emplaceable& back = V.emplace_back(std::move(A0));
    // 确认容器 V 的最后一个元素与 back 引用的是同一个对象
    EXPECT_TRUE(&back == &V.back());
    // 确认容器 V 中元素个数为 1
    EXPECT_TRUE(V.size() == 1);
    // 确认容器 V 的最后一个元素状态为 ES_Emplaced
    EXPECT_TRUE(back.State == ES_Emplaced);
    // 确认容器 V 的最后一个元素的 A0 成员状态为 EAS_RValue
    EXPECT_TRUE(back.A0.State == EAS_RValue);
    // 确认容器 V 的最后一个元素的 A1 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A1.State == EAS_Defaulted);
    // 确认容器 V 的最后一个元素的 A2 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    // 确认容器 V 的最后一个元素的 A3 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    // 创建容器 V，最大容量为 3
    SmallVector<Emplaceable, 3> V;
    // 在容器 V 的末尾构造一个 Emplaceable 对象，并将 A0 拷贝到其中
    Emplaceable& back = V.emplace_back(A0);
    // 确认容器 V 的最后一个元素与 back 引用的是同一个对象
    EXPECT_TRUE(&back == &V.back());
    // 确认容器 V 中元素个数为 1
    EXPECT_TRUE(V.size() == 1);
    // 确认容器 V 的最后一个元素状态为 ES_Emplaced
    EXPECT_TRUE(back.State == ES_Emplaced);
    // 确认容器 V 的最后一个元素的 A0 成员状态为 EAS_LValue
    EXPECT_TRUE(back.A0.State == EAS_LValue);
    // 确认容器 V 的最后一个元素的 A1 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A1.State == EAS_Defaulted);
    // 确认容器 V 的最后一个元素的 A2 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    // 确认容器 V 的最后一个元素的 A3 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    // 创建容器 V，最大容量为 3
    SmallVector<Emplaceable, 3> V;
    // 在容器 V 的末尾构造一个 Emplaceable 对象，并使用 A0 和 A1 初始化其成员
    Emplaceable& back = V.emplace_back(A0, A1);
    // 确认容器 V 的最后一个元素与 back 引用的是同一个对象
    EXPECT_TRUE(&back == &V.back());
    // 确认容器 V 中元素个数为 1
    EXPECT_TRUE(V.size() == 1);
    // 确认容器 V 的最后一个元素状态为 ES_Emplaced
    EXPECT_TRUE(back.State == ES_Emplaced);
    // 确认容器 V 的最后一个元素的 A0 成员状态为 EAS_LValue
    EXPECT_TRUE(back.A0.State == EAS_LValue);
    // 确认容器 V 的最后一个元素的 A1 成员状态为 EAS_LValue
    EXPECT_TRUE(back.A1.State == EAS_LValue);
    // 确认容器 V 的最后一个元素的 A2 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    // 确认容器 V 的最后一个元素的 A3 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    // 创建容器 V，最大容量为 3
    SmallVector<Emplaceable, 3> V;
    // 在容器 V 的末尾构造一个 Emplaceable 对象，并将 A0 和 A1 移动到其中
    Emplaceable& back = V.emplace_back(std::move(A0), std::move(A1));
    // 确认容器 V 的最后一个元素与 back 引用的是同一个对象
    EXPECT_TRUE(&back == &V.back());
    // 确认容器 V 中元素个数为 1
    EXPECT_TRUE(V.size() == 1);
    // 确认容器 V 的最后一个元素状态为 ES_Emplaced
    EXPECT_TRUE(back.State == ES_Emplaced);
    // 确认容器 V 的最后一个元素的 A0 成员状态为 EAS_RValue
    EXPECT_TRUE(back.A0.State == EAS_RValue);
    // 确认容器 V 的最后一个元素的 A1 成员状态为 EAS_RValue
    EXPECT_TRUE(back.A1.State == EAS_RValue);
    // 确认容器 V 的最后一个元素的 A2 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    // 确认容器 V 的最后一个元素的 A3 成员状态为 EAS_Defaulted
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    // 创建容器 V，最大容量为 3
    SmallVector<Emplaceable, 3> V;
    // 在容器 V 的末尾构造一个 Emplaceable 对象，并将 A0、A1、A2 和 A3 分别移动或拷贝到其中
    // NOLINTNEXTLINE(bugprone-use-after-move)
    Emplaceable& back = V.emplace_back(std::move(A0), A1, std::move(A2), A3);
    // 确认容器 V 的最后一个元素与 back 引用的是同一个对象
    EXPECT_TRUE(&back == &V.back());
    // 确认容器 V 中元素个数为 1
    EXPECT_TRUE(V.size() == 1);
    // 确认容器 V 的最后一个元素状态为 ES_Emplaced
    EXPECT_TRUE(back.State == ES_Emplaced);
    // 确认容器 V 的最后一个元素的 A0 成员状态为 EAS_RValue
    EXPECT_TRUE(back.A0.State == EAS_RValue);
    // 确认容器 V 的最后一个元素的 A1 成员状态为 EAS_LValue
    EXPECT_TRUE(back.A1.State == EAS_LValue);
    // 确认容器 V 的最后一个元素的 A2 成员状态为 EAS_RValue
    EXPECT_TRUE(back.A2.State == EAS_RValue);
    // 确认容器 V 的最后一个元素的 A3 成员状态为 EAS_LValue
    EXPECT_TRUE(back.A3.State == EAS_LValue);
  }
  {
    // 创建容器 V，元素类型为 int，最大容量为 1
    SmallVector<int, 1> V;
    // 在容器 V 的末尾插入一个默认构造的 int 类型对象
    V.emplace_back();
    // 在容器 V 的末尾插入值为 42 的 int 类型对象
    V.emplace_back(42);
    // 确认容器 V 中元素个数为 2
}

// 定义一个名为 SmallVectorTest 的测试集合，测试 SmallVector 类的默认内联元素行为
TEST(SmallVectorTest, DefaultInlinedElements) {
  // 创建一个空的 SmallVector<int> 对象 V
  SmallVector<int> V;
  // 断言 V 是否为空
  EXPECT_TRUE(V.empty());
  // 将整数 7 添加到 V 的末尾
  V.push_back(7);
  // 断言 V 的第一个元素是否为 7
  EXPECT_EQ(V[0], 7);

  // 检查默认的内联元素策略是否允许至少几层嵌套的 SmallVector<T>
  // 这种模式在实际中经常发生，即使每层 SmallVector 的大小超过了预期的最大大小
  SmallVector<SmallVector<SmallVector<int>>> NestedV;
  NestedV.emplace_back().emplace_back().emplace_back(42);
  // 断言 NestedV 的第一个元素的第一个元素的第一个元素是否为 42
  EXPECT_EQ(NestedV[0][0][0], 42);
}

// 定义一个名为 SmallVectorTest 的测试集合，测试 SmallVector 类的初始化列表功能
TEST(SmallVectorTest, InitializerList) {
  // 创建一个最大容量为 2 的 SmallVector<int> 对象 V1，并初始化为空列表
  SmallVector<int, 2> V1 = {};
  // 断言 V1 是否为空
  EXPECT_TRUE(V1.empty());
  // 将 {0, 0} 赋值给 V1
  V1 = {0, 0};
  // 断言 makeArrayRef(V1) 是否与 {0, 0} 相等
  EXPECT_TRUE(makeArrayRef(V1).equals({0, 0}));
  // 将 {-1, -1} 赋值给 V1
  V1 = {-1, -1};
  // 断言 makeArrayRef(V1) 是否与 {-1, -1} 相等
  EXPECT_TRUE(makeArrayRef(V1).equals({-1, -1}));

  // 创建一个最大容量为 2 的 SmallVector<int> 对象 V2，并使用初始化列表 {1, 2, 3, 4} 初始化
  SmallVector<int, 2> V2 = {1, 2, 3, 4};
  // 断言 makeArrayRef(V2) 是否与 {1, 2, 3, 4} 相等
  EXPECT_TRUE(makeArrayRef(V2).equals({1, 2, 3, 4}));
  // 将 {4} 赋值给 V2
  V2.assign({4});
  // 断言 makeArrayRef(V2) 是否与 {4} 相等
  EXPECT_TRUE(makeArrayRef(V2).equals({4}));
  // 将 {3, 2} 添加到 V2 的末尾
  V2.append({3, 2});
  // 断言 makeArrayRef(V2) 是否与 {4, 3, 2} 相等
  EXPECT_TRUE(makeArrayRef(V2).equals({4, 3, 2}));
  // 在 V2 的第一个位置插入整数 5
  V2.insert(V2.begin() + 1, 5);
  // 断言 makeArrayRef(V2) 是否与 {4, 5, 3, 2} 相等
  EXPECT_TRUE(makeArrayRef(V2).equals({4, 5, 3, 2}));
}

// 定义一个模板类 SmallVectorReferenceInvalidationTest，用于测试 SmallVector 类的引用失效情况
template <class VectorT>
class SmallVectorReferenceInvalidationTest : public SmallVectorTestBase {
 protected:
  const char* AssertionMessage =
      "Attempting to reference an element of the vector in an operation \" "
      "\"that invalidates it";

  VectorT V;

  // 返回 SmallVector<T, N> 类型的内建元素数量
  template <typename T, unsigned N>
  static unsigned NumBuiltinElts(const SmallVector<T, N>&) {
    return N;
  }

  // 判断类型 T 是否与 VectorT 的 value_type 相同
  template <class T>
  static bool isValueType() {
    return std::is_same<T, typename VectorT::value_type>::value;
  }

  // 设置测试的初始条件，在小型模式下填充 SmallVector V
  void SetUp() override {
    SmallVectorTestBase::SetUp();
    for (int I = 0, E = NumBuiltinElts(V); I != E; ++I)
      V.emplace_back(I + 1);
  }
};

// 使用 SmallVector<int, 3> 和 SmallVector<Constructable, 3> 两种类型进行引用失效测试
using SmallVectorReferenceInvalidationTestTypes =
    ::testing::Types<SmallVector<int, 3>, SmallVector<Constructable, 3>>;

// 模板化测试集合，针对不同类型的 SmallVector 进行引用失效测试
TYPED_TEST_SUITE(
    SmallVectorReferenceInvalidationTest,
    SmallVectorReferenceInvalidationTestTypes, );

// 定义一个模板化测试用例，测试 SmallVectorReferenceInvalidationTest 类的 push_back 方法
TYPED_TEST(SmallVectorReferenceInvalidationTest, PushBack) {
  // 注意：SetUp 方法向 V 中添加了 [1, 2, ...] 直到小型存储器满
  auto& V = this->V;
  int N = this->NumBuiltinElts(V);

  // 在从小型存储器扩展时，将对最后一个元素的引用 push_back 到 V 中
  V.push_back(V.back());
  // 断言最后一个元素是否为 N
  EXPECT_EQ(N, V.back());

  // 检查旧值是否仍然存在（未被移动）
  EXPECT_EQ(N, V[V.size() - 2]);

  // 再次填充存储器
  V.back() = V.size();
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // 在从大型存储器扩展时，将对最后一个元素的引用 push_back 到 V 中
  V.push_back(V.back());
  // 断言最后一个元素是否为 int(V.size()) - 1
  EXPECT_EQ(int(V.size()) - 1, V.back());
}
TYPED_TEST(SmallVectorReferenceInvalidationTest, PushBackMoved) {
  // 获取对当前测试用例中的 SmallVector 的引用
  auto& V = this->V;
  // 获取当前 SmallVector 中的元素个数
  int N = this->NumBuiltinElts(V);

  // 在从小型存储模式扩展时，将对最后一个元素的引用推入 SmallVector
  V.push_back(std::move(V.back()));
  // 断言最后一个元素是否与之前最后一个元素相同
  EXPECT_EQ(N, V.back());
  if (this->template isValueType<Constructable>()) {
    // 检查值是否被移动而非复制
    EXPECT_EQ(0, V[V.size() - 2]);
  }

  // 再次填充存储
  V.back() = V.size();
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // 在从大型存储模式扩展时，将对最后一个元素的引用推入 SmallVector
  V.push_back(std::move(V.back()));

  // 检查值
  EXPECT_EQ(int(V.size()) - 1, V.back());
  if (this->template isValueType<Constructable>()) {
    // 检查值是否已移出
    EXPECT_EQ(0, V[V.size() - 2]);
  }
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, Resize) {
  // 获取对当前测试用例中的 SmallVector 的引用
  auto& V = this->V;
  // 获取当前 SmallVector 中的元素个数
  (void)V;
  int N = this->NumBuiltinElts(V);
  // 将 SmallVector 调整大小，添加一个新元素，并用最后一个元素填充
  V.resize(N + 1, V.back());
  EXPECT_EQ(N, V.back());

  // 调整大小以添加足够的元素，使得 SmallVector 再次扩展
  // 如果引用失效，则内存检查器应能够捕获到这里的 use-after-free
  V.resize(V.capacity() + 1, V.front());
  EXPECT_EQ(1, V.back());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, Append) {
  // 获取对当前测试用例中的 SmallVector 的引用
  auto& V = this->V;
  (void)V;
  // 向 SmallVector 添加一个元素，该元素是最后一个元素的引用
  V.append(1, V.back());
  int N = this->NumBuiltinElts(V);
  EXPECT_EQ(N, V[N - 1]);

  // 添加足够多的元素，使得 SmallVector 再次扩展，测试在大型模式下的扩展
  //
  // 如果引用失效，则内存检查器应能够捕获到这里的 use-after-free
  V.append(V.capacity() - V.size() + 1, V.front());
  EXPECT_EQ(1, V.back());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, AppendRange) {
  // 获取对当前测试用例中的 SmallVector 的引用
  auto& V = this->V;
  (void)V;
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  // 预期这里会因为死亡测试而终止，检查是否在使用断言消息时发生了死亡
  EXPECT_DEATH(V.append(V.begin(), V.begin() + 1), this->AssertionMessage);

  ASSERT_EQ(3u, this->NumBuiltinElts(V));
  ASSERT_EQ(3u, V.size());
  V.pop_back();
  ASSERT_EQ(2u, V.size());

  // 确认在添加多个元素时检查扩展
  // 如果引用失效，则内存检查器应能够捕获到这里的 use-after-free
  EXPECT_DEATH(V.append(V.begin(), V.end()), this->AssertionMessage);
#endif
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, Assign) {
  // 获取对当前测试用例中的 SmallVector 的引用
  auto& V = this->V;
  (void)V;
  int N = this->NumBuiltinElts(V);
  ASSERT_EQ(unsigned(N), V.size());
  ASSERT_EQ(unsigned(N), V.capacity());

  // 检查在小型模式下收缩分配
  V.assign(1, V.back());
  EXPECT_EQ(1u, V.size());
  EXPECT_EQ(N, V[0]);

  // 检查在小型模式下扩展分配
  ASSERT_LT(V.size(), V.capacity());
  V.assign(V.capacity(), V.back());
  for (int I = 0, E = V.size(); I != E; ++I) {
    EXPECT_EQ(N, V[I]);

    // 重置为 [1, 2, ...].
  V[I] = I + 1;

// 将数组 V 的第 I 个元素赋值为 I + 1。


}

// Check assign that grows to large mode.
ASSERT_EQ(2, V[1]);
V.assign(V.capacity() + 1, V[1]);
for (int I = 0, E = V.size(); I != E; ++I) {
  EXPECT_EQ(2, V[I]);

  // Reset to [1, 2, ...].
  V[I] = I + 1;
}

// 对于数组 V，检查分配操作以扩展到大模式。
// 断言数组 V 的第 1 个元素是否为 2。
// 将数组 V 扩展到容量加一的大小，所有新元素赋值为 V 的第一个元素的值。
// 遍历数组 V 的所有元素，断言每个元素是否为 2。
// 将数组 V 恢复为 [1, 2, ...] 的顺序。


// Check assign that shrinks in large mode.
V.assign(1, V[1]);
EXPECT_EQ(2, V[0]);

// 对于数组 V，检查分配操作以缩小到小模式。
// 将数组 V 重置为包含一个元素，该元素为数组 V 第二个元素的值。
// 断言数组 V 的第一个元素是否为 2。
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, AssignRange) {
  auto& V = this->V;  // 引用当前测试实例的小型向量 V
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  // 在调试模式下，预期程序会因为此操作而终止，使用断言消息作为死亡测试信息
  EXPECT_DEATH(V.assign(V.begin(), V.end()), this->AssertionMessage);
  EXPECT_DEATH(V.assign(V.begin(), V.end() - 1), this->AssertionMessage);
#endif
  // 将 V 重新赋值为其自身的空范围
  V.assign(V.begin(), V.begin());
  // 确认 V 现在为空
  EXPECT_TRUE(V.empty());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, Insert) {
  // 注意：setup 函数向 V 添加 [1, 2, ...] 直到达到小型模式的容量上限
  auto& V = this->V;  // 引用当前测试实例的小型向量 V
  (void)V;

  // 在向量头部插入尾部元素的引用，使其从小型模式扩展到非小型模式
  // 确认值被复制出去（通过移动构造设置为 0）
  V.insert(V.begin(), V.back());
  EXPECT_EQ(int(V.size() - 1), V.front());
  EXPECT_EQ(int(V.size() - 1), V.back());

  // 再次填充向量，直到其容量
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // 再次从大型存储扩展到大型存储
  V.insert(V.begin(), V.back());
  EXPECT_EQ(int(V.size() - 1), V.front());
  EXPECT_EQ(int(V.size() - 1), V.back());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, InsertMoved) {
  // 注意：setup 函数向 V 添加 [1, 2, ...] 直到达到小型模式的容量上限
  auto& V = this->V;  // 引用当前测试实例的小型向量 V
  (void)V;

  // 插入尾部元素的移动引用，使其从小型模式扩展到非小型模式
  // 确认值被复制出去（通过移动构造设置为 0）
  V.insert(V.begin(), std::move(V.back()));
  EXPECT_EQ(int(V.size() - 1), V.front());
  if (this->template isValueType<Constructable>()) {
    // 检查值已经移出
    EXPECT_EQ(0, V.back());
  }

  // 再次填充向量，直到其容量
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // 再次从大型存储扩展到大型存储
  V.insert(V.begin(), std::move(V.back()));
  EXPECT_EQ(int(V.size() - 1), V.front());
  if (this->template isValueType<Constructable>()) {
    // 检查值已经移出
    EXPECT_EQ(0, V.back());
  }
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, InsertN) {
  auto& V = this->V;  // 引用当前测试实例的小型向量 V
  (void)V;

  // 插入尾部元素的引用到第二个位置，覆盖 NumToInsert <= this->end() - I 的情况
  V.insert(V.begin() + 1, 1, V.back());
  int N = this->NumBuiltinElts(V);
  EXPECT_EQ(N, V[1]);

  // 覆盖 NumToInsert > this->end() - I 的情况，插入足够多的元素使得 V 再次增长
  // V.capacity() 将会比所需的元素多，但这是一种简单的方式来覆盖两种情况
  //
  // 如果引用失效导致错误，内存检查工具应该能在此处捕获到 use-after-free 错误
  V.insert(V.begin(), V.capacity(), V.front());
  EXPECT_EQ(1, V.front());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, InsertRange) {
  auto& V = this->V;  // 引用当前测试实例的小型向量 V
  (void)V;
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  // 断言：在插入元素时，预期会发生程序死亡，使用自定义的断言消息
  EXPECT_DEATH(
      V.insert(V.begin(), V.begin(), V.begin() + 1), this->AssertionMessage);

  // 断言：确认插入一个元素后，内置元素的数量为3
  ASSERT_EQ(3u, this->NumBuiltinElts(V));
  // 断言：确认向容器中插入一个元素后，容器大小为3
  ASSERT_EQ(3u, V.size());
  // 弹出容器中最后一个元素
  V.pop_back();
  // 断言：确认弹出一个元素后，容器大小为2
  ASSERT_EQ(2u, V.size());

  // 说明：确认在插入多个元素时，检查是否进行了容量增长
  EXPECT_DEATH(V.insert(V.begin(), V.begin(), V.end()), this->AssertionMessage);
#endif
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, EmplaceBack) {
  // 说明：将当前测试环境的容器引用赋给V
  auto& V = this->V;
  // 获取当前容器内置元素的数量
  int N = this->NumBuiltinElts(V);

  // 说明：在从小型存储扩展时，推送对最后一个元素的引用
  V.emplace_back(V.back());
  // 断言：确认最后一个元素的值与之前相同
  EXPECT_EQ(N, V.back());

  // 说明：检查旧值是否仍然存在（未被移动）
  EXPECT_EQ(N, V[V.size() - 2]);

  // 说明：再次填充容器
  V.back() = V.size();
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // 说明：在从大型存储扩展时，推送对最后一个元素的引用
  V.emplace_back(V.back());
  // 断言：确认最后一个元素的值与容器大小减1相同
  EXPECT_EQ(int(V.size()) - 1, V.back());
}

template <class VectorT>
class SmallVectorInternalReferenceInvalidationTest
    : public SmallVectorTestBase {
 protected:
  const char* AssertionMessage =
      "Attempting to reference an element of the vector in an operation \" "
      "\"that invalidates it";

  VectorT V;

  template <typename T, unsigned N>
  static unsigned NumBuiltinElts(const SmallVector<T, N>&) {
    return N;
  }

  void SetUp() override {
    SmallVectorTestBase::SetUp();

    // 说明：填充小型容器，使得插入操作会移动元素
    for (int I = 0, E = NumBuiltinElts(V); I != E; ++I)
      V.emplace_back(I + 1, I + 1);
  }
};

// Test pairs of the same types from SmallVectorReferenceInvalidationTestTypes.
using SmallVectorInternalReferenceInvalidationTestTypes = ::testing::Types<
    SmallVector<std::pair<int, int>, 3>,
    SmallVector<std::pair<Constructable, Constructable>, 3>>;

TYPED_TEST_SUITE(
    SmallVectorInternalReferenceInvalidationTest,
    SmallVectorInternalReferenceInvalidationTestTypes, );

TYPED_TEST(SmallVectorInternalReferenceInvalidationTest, EmplaceBack) {
  // 说明：在测试设置中，将[1, 2, ...]添加到V中，直到它达到小型存储的容量为止
  auto& V = this->V;
  // 获取当前容器内置元素的数量
  int N = this->NumBuiltinElts(V);

  // 说明：在从小型存储扩展时，推送对最后一个元素的引用
  V.emplace_back(V.back().first, V.back().second);
  // 断言：确认最后一个元素的第一个和第二个值与之前相同
  EXPECT_EQ(N, V.back().first);
  EXPECT_EQ(N, V.back().second);

  // 说明：检查旧值是否仍然存在（未被移动）
  EXPECT_EQ(N, V[V.size() - 2].first);
  EXPECT_EQ(N, V[V.size() - 2].second);

  // 说明：再次填充容器
  V.back().first = V.back().second = V.size();
  while (V.size() < V.capacity())
    # 向向量 V 中添加一个元素，元素的第一个和第二个元素分别为 V 的大小加一
    V.emplace_back(V.size() + 1, V.size() + 1);

  # 当从大存储容量增长时，将对最后一个元素的引用推入向量
  # 此处确保新元素的第一个和第二个元素与 V 中最后一个元素相同
  V.emplace_back(V.back().first, V.back().second);
  # 验证新添加的元素的第一个元素是 V 的大小减一
  EXPECT_EQ(int(V.size()) - 1, V.back().first);
  # 验证新添加的元素的第二个元素是 V 的大小减一
  EXPECT_EQ(int(V.size()) - 1, V.back().second);
}

} // end namespace
// NOLINTEND(*arrays, bugprone-forwarding-reference-overload)
```