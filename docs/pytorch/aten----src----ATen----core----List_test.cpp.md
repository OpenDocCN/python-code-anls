# `.\pytorch\aten\src\ATen\core\List_test.cpp`

```
// 包含 ATen 核心的 List 头文件
#include <ATen/core/List.h>
// 包含 Google 测试框架的头文件
#include <gtest/gtest.h>

// 使用 c10 命名空间
using namespace c10;

// NOLINTBEGIN(performance-move-const-arg, bugprone-use-after-move)
// 定义测试用例 ListTestIValueBasedList，测试空列表调用 empty 方法是否返回 true
TEST(ListTestIValueBasedList, givenEmptyList_whenCallingEmpty_thenReturnsTrue) {
    // 创建空的 List 对象
    List<string> list;
    // 断言空列表调用 empty 方法返回 true
    EXPECT_TRUE(list.empty());
}

// 定义测试用例 ListTestIValueBasedList，测试非空列表调用 empty 方法是否返回 false
TEST(ListTestIValueBasedList, givenNonemptyList_whenCallingEmpty_thenReturnsFalse) {
    // 创建包含一个元素 "3" 的 List 对象
    List<string> list({"3"});
    // 断言非空列表调用 empty 方法返回 false
    EXPECT_FALSE(list.empty());
}

// 定义测试用例 ListTestIValueBasedList，测试空列表调用 size 方法是否返回 0
TEST(ListTestIValueBasedList, givenEmptyList_whenCallingSize_thenReturnsZero) {
    // 创建空的 List 对象
    List<string> list;
    // 断言空列表调用 size 方法返回 0
    EXPECT_EQ(0, list.size());
}

// 定义测试用例 ListTestIValueBasedList，测试非空列表调用 size 方法是否返回元素数量
TEST(ListTestIValueBasedList, givenNonemptyList_whenCallingSize_thenReturnsNumberOfElements) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // 断言非空列表调用 size 方法返回 2
    EXPECT_EQ(2, list.size());
}

// 定义测试用例 ListTestIValueBasedList，测试清空列表后是否为空
TEST(ListTestIValueBasedList, givenNonemptyList_whenCallingClear_thenIsEmpty) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // 清空列表
    list.clear();
    // 断言清空后列表调用 empty 方法返回 true
    EXPECT_TRUE(list.empty());
}

// 定义测试用例 ListTestIValueBasedList，测试通过索引获取元素是否正确
TEST(ListTestIValueBasedList, whenCallingGetWithExistingPosition_thenReturnsElement) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // 断言获取索引 0 处的元素为 "3"
    EXPECT_EQ("3", list.get(0));
    // 断言获取索引 1 处的元素为 "4"
    EXPECT_EQ("4", list.get(1));
}

// 定义测试用例 ListTestIValueBasedList，测试通过不存在的索引获取元素是否抛出异常
TEST(ListTestIValueBasedList, whenCallingGetWithNonExistingPosition_thenThrowsException) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    // 断言获取不存在的索引 2 处的元素抛出 std::out_of_range 异常
    EXPECT_THROW(list.get(2), std::out_of_range);
}

// 定义测试用例 ListTestIValueBasedList，测试通过索引提取元素是否正确
TEST(ListTestIValueBasedList, whenCallingExtractWithExistingPosition_thenReturnsElement) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // 断言提取索引 0 处的元素为 "3"
    EXPECT_EQ("3", list.extract(0));
    // 断言提取索引 1 处的元素为 "4"
    EXPECT_EQ("4", list.extract(1));
}

// 定义测试用例 ListTestIValueBasedList，测试提取后索引处元素是否无效
TEST(ListTestIValueBasedList, whenCallingExtractWithExistingPosition_thenListElementBecomesInvalid) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // 提取索引 0 处的元素 "3"
    list.extract(0);
    // 断言索引 0 处的元素变为空字符串
    EXPECT_EQ("", list.get(0));
}

// 定义测试用例 ListTestIValueBasedList，测试通过不存在的索引提取元素是否抛出异常
TEST(ListTestIValueBasedList, whenCallingExtractWithNonExistingPosition_thenThrowsException) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    // 断言提取不存在的索引 2 处的元素抛出 std::out_of_range 异常
    EXPECT_THROW(list.extract(2), std::out_of_range);
}

// 定义测试用例 ListTestIValueBasedList，测试通过索引设置元素是否正确
TEST(ListTestIValueBasedList, whenCallingCopyingSetWithExistingPosition_thenChangesElement) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // 创建字符串值 "5"
    string value = "5";
    // 设置索引 1 处的元素为 "5"
    list.set(1, value);
    // 断言索引 0 处的元素为 "3"
    EXPECT_EQ("3", list.get(0));
    // 断言索引 1 处的元素为 "5"
    EXPECT_EQ("5", list.get(1));
}

// 定义测试用例 ListTestIValueBasedList，测试通过索引移动设置元素是否正确
TEST(ListTestIValueBasedList, whenCallingMovingSetWithExistingPosition_thenChangesElement) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // 创建字符串值 "5"
    string value = "5";
    // 移动设置索引 1 处的元素为 "5"
    list.set(1, std::move(value));
    // 断言索引 0 处的元素为 "3"
    EXPECT_EQ("3", list.get(0));
    // 断言索引 1 处的元素为 "5"
    EXPECT_EQ("5", list.get(1));
}

// 定义测试用例 ListTestIValueBasedList，测试通过不存在的索引复制设置元素是否抛出异常
TEST(ListTestIValueBasedList, whenCallingCopyingSetWithNonExistingPosition_thenThrowsException) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // 创建字符串值 "5"
    string value = "5";
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    // 断言复制设置不存在的索引 2 处的元素抛出 std::out_of_range 异常
    EXPECT_THROW(list.set(2, value), std::out_of_range);
}

// 定义测试用例 ListTestIValueBasedList，测试通过不存在的索引移动设置元素是否抛出异常
TEST(ListTestIValueBasedList, whenCallingMovingSetWithNonExistingPosition_thenThrowsException) {
    // 创建包含两个元素 "3" 和 "4" 的 List 对象
    List<string> list({"3", "4"});
    // 创建字符串值 "5"
    string value = "5";
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    // 断言移动设置不存在的索引 2 处的元素抛出 std::out_of_range 异常
    EXPECT_THROW(list.set(2, std::move(value)), std::out_of_range);
}
TEST(ListTestIValueBasedList, whenCallingAccessOperatorWithExistingPosition_thenReturnsElement) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4"}
  List<string> list({"3", "4"});
  // 断言访问索引为 0 的元素返回值为 "3"
  EXPECT_EQ("3", static_cast<string>(list[0]));
  // 断言访问索引为 1 的元素返回值为 "4"
  EXPECT_EQ("4", static_cast<string>(list[1]));
}

TEST(ListTestIValueBasedList, whenAssigningToAccessOperatorWithExistingPosition_thenSetsElement) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4", "5"}
  List<string> list({"3", "4", "5"});
  // 将索引为 1 的元素赋值为 "6"
  list[1] = "6";
  // 断言索引为 0 的元素值为 "3"
  EXPECT_EQ("3", list.get(0));
  // 断言索引为 1 的元素值为 "6"
  EXPECT_EQ("6", list.get(1));
  // 断言索引为 2 的元素值为 "5"
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTestIValueBasedList, whenAssigningToAccessOperatorFromAccessOperator_thenSetsElement) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4", "5"}
  List<string> list({"3", "4", "5"});
  // 将索引为 1 的元素赋值为索引为 2 的元素的值 "5"
  list[1] = list[2];
  // 断言索引为 0 的元素值为 "3"
  EXPECT_EQ("3", list.get(0));
  // 断言索引为 1 的元素值为 "5"
  EXPECT_EQ("5", list.get(1));
  // 断言索引为 2 的元素值为 "5"
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTestIValueBasedList, whenSwappingFromAccessOperator_thenSwapsElements) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4", "5"}
  List<string> list({"3", "4", "5"});
  // 交换索引为 1 和索引为 2 的元素
  swap(list[1], list[2]);
  // 断言索引为 0 的元素值为 "3"
  EXPECT_EQ("3", list.get(0));
  // 断言索引为 1 的元素值为 "5"
  EXPECT_EQ("5", list.get(1));
  // 断言索引为 2 的元素值为 "4"
  EXPECT_EQ("4", list.get(2));
}

TEST(ListTestIValueBasedList, whenCallingAccessOperatorWithNonExistingPosition_thenThrowsException) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4"}
  List<string> list({"3", "4"});
  // 期望访问索引为 2 的元素抛出 std::out_of_range 异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(list[2], std::out_of_range);
}

TEST(ListTestIValueBasedList, whenCallingInsertOnIteratorWithLValue_thenInsertsElement) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4", "6"}
  List<string> list({"3", "4", "6"});
  // 创建字符串 v，并在索引 2 处插入 v 的值 "5"
  string v = "5";
  list.insert(list.begin() + 2, v);
  // 断言列表的大小为 4
  EXPECT_EQ(4, list.size());
  // 断言索引为 2 的元素值为 "5"
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTestIValueBasedList, whenCallingInsertOnIteratorWithRValue_thenInsertsElement) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4", "6"}
  List<string> list({"3", "4", "6"});
  // 创建字符串 v，并在索引 2 处插入 v 的右值 "5"
  string v = "5";
  list.insert(list.begin() + 2, std::move(v));
  // 断言列表的大小为 4
  EXPECT_EQ(4, list.size());
  // 断言索引为 2 的元素值为 "5"
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTestIValueBasedList, whenCallingInsertWithLValue_thenReturnsIteratorToNewElement) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4", "6"}
  List<string> list({"3", "4", "6"});
  // 创建字符串 v，并在索引 2 处插入 v 的值 "5"，返回插入位置的迭代器 result
  string v = "5";
  List<string>::iterator result = list.insert(list.begin() + 2, v);
  // 断言返回的迭代器 result 指向索引 2 的位置
  EXPECT_EQ(list.begin() + 2, result);
}

TEST(ListTestIValueBasedList, whenCallingInsertWithRValue_thenReturnsIteratorToNewElement) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4", "6"}
  List<string> list({"3", "4", "6"});
  // 创建字符串 v，并在索引 2 处插入 v 的右值 "5"，返回插入位置的迭代器 result
  string v = "5";
  List<string>::iterator result = list.insert(list.begin() + 2, std::move(v));
  // 断言返回的迭代器 result 指向索引 2 的位置
  EXPECT_EQ(list.begin() + 2, result);
}

TEST(ListTestIValueBasedList, whenCallingEmplaceWithLValue_thenInsertsElement) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4", "6"}
  List<string> list({"3", "4", "6"});
  // 创建字符串 v，并在索引 2 处就地构造 "5"
  string v = "5";
  list.emplace(list.begin() + 2, v);
  // 断言列表的大小为 4
  EXPECT_EQ(4, list.size());
  // 断言索引为 2 的元素值为 "5"
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTestIValueBasedList, whenCallingEmplaceWithRValue_thenInsertsElement) {
  // 创建一个字符串类型的列表，初始化为 {"3", "4", "6"}
  List<string> list({"3", "4", "6"});
  // 创建字符串 v，并在索引 2 处就地构造 v 的右值 "5"
  string v = "5";
  list.emplace(list.begin() + 2, std::move(v));
  // 断言列表的大小为 4
  EXPECT_EQ(4, list.size());
  // 断言索引为 2 的元素值为 "5"
  EXPECT_EQ("5", list.get(2));
}
TEST(ListTestIValueBasedList, whenCallingEmplaceWithConstructorArg_thenInsertsElement) {
  // 创建一个字符串类型的链表，初始值为 {"3", "4", "6"}
  List<string> list({"3", "4", "6"});
  // 在链表的第二个位置（索引为 2）插入新元素 "5"
  list.emplace(list.begin() + 2, "5"); // const char* is a constructor arg to std::string
  // 断言链表大小为 4
  EXPECT_EQ(4, list.size());
  // 断言索引为 2 的元素为 "5"
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTestIValueBasedList, whenCallingPushBackWithLValue_ThenInsertsElement) {
  // 创建一个空的字符串类型链表
  List<string> list;
  // 创建字符串变量 v，并将其值设置为 "5"
  string v = "5";
  // 将字符串变量 v 的值 "5" 加入链表尾部
  list.push_back(v);
  // 断言链表大小为 1
  EXPECT_EQ(1, list.size());
  // 断言索引为 0 的元素为 "5"
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTestIValueBasedList, whenCallingPushBackWithRValue_ThenInsertsElement) {
  // 创建一个空的字符串类型链表
  List<string> list;
  // 创建字符串变量 v，并将其值设置为 "5"
  string v = "5";
  // 将字符串变量 v 的值 "5" 移动到链表尾部
  list.push_back(std::move(v));
  // 断言链表大小为 1
  EXPECT_EQ(1, list.size());
  // 断言索引为 0 的元素为 "5"
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTestIValueBasedList, whenCallingEmplaceBackWithLValue_ThenInsertsElement) {
  // 创建一个空的字符串类型链表
  List<string> list;
  // 创建字符串变量 v，并将其值设置为 "5"
  string v = "5";
  // 使用字符串变量 v 的值 "5" emplace 到链表尾部
  list.emplace_back(v);
  // 断言链表大小为 1
  EXPECT_EQ(1, list.size());
  // 断言索引为 0 的元素为 "5"
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTestIValueBasedList, whenCallingEmplaceBackWithRValue_ThenInsertsElement) {
  // 创建一个空的字符串类型链表
  List<string> list;
  // 创建字符串变量 v，并将其值设置为 "5"
  string v = "5";
  // 使用字符串变量 v 的值 "5" 的移动形式 emplace 到链表尾部
  list.emplace_back(std::move(v));
  // 断言链表大小为 1
  EXPECT_EQ(1, list.size());
  // 断言索引为 0 的元素为 "5"
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTestIValueBasedList, whenCallingEmplaceBackWithConstructorArg_ThenInsertsElement) {
  // 创建一个空的字符串类型链表
  List<string> list;
  // 使用字符串常量 "5" emplace 到链表尾部，const char* 是传递给 std::string 的构造函数参数
  list.emplace_back("5");  // const char* is a constructor arg to std::string
  // 断言链表大小为 1
  EXPECT_EQ(1, list.size());
  // 断言索引为 0 的元素为 "5"
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTestIValueBasedList, givenEmptyList_whenIterating_thenBeginIsEnd) {
  // 创建一个空的字符串类型链表
  List<string> list;
  // 创建一个空的常量字符串类型链表
  const List<string> clist;
  // 断言空链表的 begin() 和 end() 相等
  EXPECT_EQ(list.begin(), list.end());
  // 断言空链表的 begin() 和 end() 相等
  EXPECT_EQ(list.begin(), list.end());
  // 断言空常量链表的 begin() 和 end() 相等
  EXPECT_EQ(clist.begin(), clist.end());
  // 断言空常量链表的 begin() 和 end() 相等
  EXPECT_EQ(clist.begin(), clist.end());
}

TEST(ListTestIValueBasedList, whenIterating_thenFindsElements) {
  // 创建一个包含 {"3", "5"} 的字符串类型链表
  List<string> list({"3", "5"});
  // 声明并初始化用于记录找到元素的布尔变量
  bool found_first = false;
  bool found_second = false;
  // 使用范围 for 循环遍历链表中的每个元素
  for (const auto && iter : list) {
    // 如果当前元素转换为字符串后等于 "3"
    if (static_cast<string>(iter) == "3") {
      // 断言此前未找到 "3"
      EXPECT_FALSE(found_first);
      // 标记已找到 "3"
      found_first = true;
    // 如果当前元素转换为字符串后等于 "5"
    } else if (static_cast<string>(iter) == "5") {
      // 断言此前未找到 "5"
      EXPECT_FALSE(found_second);
      // 标记已找到 "5"
      found_second = true;
    // 如果当前元素既不是 "3" 也不是 "5"
    } else {
      // 失败，不应该出现这种情况
      ADD_FAILURE();
    }
  }
  // 断言已找到 "3" 和 "5"
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(ListTestIValueBasedList, whenIteratingWithForeach_thenFindsElements) {
  // 创建一个包含 {"3", "5"} 的字符串类型链表
  List<string> list({"3", "5"});
  // 声明并初始化用于记录找到元素的布尔变量
  bool found_first = false;
  bool found_second = false;
  // 使用范围 for 循环遍历链表中的每个元素，显式忽略性能警告（NOLINT）
  for (const string& elem : list) {
    // 如果当前元素等于 "3"
    if (elem == "3") {
      // 断言此前未找到 "3"
      EXPECT_FALSE(found_first);
      // 标记已找到 "3"
      found_first = true;
    // 如果当前元素等于 "5"
    } else if (elem == "5") {
      // 断言此前未找到 "5"
      EXPECT_FALSE(found_second);
      // 标记已找到 "5"
      found_second = true;
    // 如果当前元素既不是 "3" 也不是 "5"
    } else {
      // 失败，不应该出现这种情况
      ADD_FAILURE();
    }
  }
  // 断言已找到 "3" 和 "5"
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(ListTestIValueBasedList, givenOneElementList_whenErasing_thenListIsEmpty) {
  // 创建一个包含 {"3"} 的字符串类型链表
  List<string> list({"3"});
  // 移除链表的第一个元素
  list.erase(list.begin());
  // 断言链表为空
  EXPECT_TRUE(list.empty());
}
TEST(ListTestIValueBasedList, givenList_whenErasing_thenReturnsIterator) {
  // 创建一个包含字符串的 List 对象
  List<string> list({"1", "2", "3"});
  // 调用 erase 方法删除第二个元素，并返回指向删除位置之后元素的迭代器
  List<string>::iterator iter = list.erase(list.begin() + 1);
  // 断言删除位置之后的迭代器与返回的迭代器相等
  EXPECT_EQ(list.begin() + 1, iter);
}

TEST(ListTestIValueBasedList, givenList_whenErasingFullRange_thenIsEmpty) {
  // 创建一个包含字符串的 List 对象
  List<string> list({"1", "2", "3"});
  // 调用 erase 方法删除整个范围的元素，即清空列表
  list.erase(list.begin(), list.end());
  // 断言列表为空
  EXPECT_TRUE(list.empty());
}

TEST(ListTestIValueBasedList, whenCallingReserve_thenDoesntCrash) {
  // 创建一个空的 List 对象并调用 reserve 方法来预留存储空间
  List<string> list;
  list.reserve(100);
}

TEST(ListTestIValueBasedList, whenCopyConstructingList_thenAreEqual) {
  // 创建一个包含字符串的 List 对象
  List<string> list1({"3", "4"});

  // 使用拷贝构造函数创建另一个 List 对象
  List<string> list2(list1);

  // 断言两个列表的大小相等
  EXPECT_EQ(2, list2.size());
  // 断言两个列表的第一个元素相同
  EXPECT_EQ("3", list2.get(0));
  // 断言两个列表的第二个元素相同
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTestIValueBasedList, whenCopyAssigningList_thenAreEqual) {
  // 创建一个包含字符串的 List 对象
  List<string> list1({"3", "4"});

  // 使用拷贝赋值运算符将一个列表赋值给另一个列表
  List<string> list2;
  list2 = list1;

  // 断言两个列表的大小相等
  EXPECT_EQ(2, list2.size());
  // 断言两个列表的第一个元素相同
  EXPECT_EQ("3", list2.get(0));
  // 断言两个列表的第二个元素相同
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTestIValueBasedList, whenCopyingList_thenAreEqual) {
  // 创建一个包含字符串的 List 对象
  List<string> list1({"3", "4"});

  // 使用 copy 方法复制一个列表
  List<string> list2 = list1.copy();

  // 断言两个列表的大小相等
  EXPECT_EQ(2, list2.size());
  // 断言两个列表的第一个元素相同
  EXPECT_EQ("3", list2.get(0));
  // 断言两个列表的第二个元素相同
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTestIValueBasedList, whenMoveConstructingList_thenNewIsCorrect) {
  // 创建一个包含字符串的 List 对象
  List<string> list1({"3", "4"});

  // 使用移动构造函数将一个列表移动到另一个列表
  List<string> list2(std::move(list1));

  // 断言新列表的大小正确
  EXPECT_EQ(2, list2.size());
  // 断言新列表的第一个元素正确
  EXPECT_EQ("3", list2.get(0));
  // 断言新列表的第二个元素正确
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTestIValueBasedList, whenMoveAssigningList_thenNewIsCorrect) {
  // 创建一个包含字符串的 List 对象
  List<string> list1({"3", "4"});

  // 使用移动赋值运算符将一个列表移动到另一个列表
  List<string> list2;
  list2 = std::move(list1);

  // 断言新列表的大小正确
  EXPECT_EQ(2, list2.size());
  // 断言新列表的第一个元素正确
  EXPECT_EQ("3", list2.get(0));
  // 断言新列表的第二个元素正确
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTestIValueBasedList, whenMoveConstructingList_thenOldIsUnchanged) {
  // 创建一个包含字符串的 List 对象
  List<string> list1({"3", "4"});

  // 使用移动构造函数将一个列表移动到另一个列表
  List<string> list2(std::move(list1));

  // 断言原始列表的大小仍然正确
  EXPECT_EQ(2, list1.size());
  // 断言原始列表的第一个元素仍然正确
  EXPECT_EQ("3", list1.get(0));
  // 断言原始列表的第二个元素仍然正确
  EXPECT_EQ("4", list1.get(1));
}

TEST(ListTestIValueBasedList, whenMoveAssigningList_thenOldIsUnchanged) {
  // 创建一个包含字符串的 List 对象
  List<string> list1({"3", "4"});

  // 使用移动赋值运算符将一个列表移动到另一个列表
  List<string> list2;
  list2 = std::move(list1);

  // 断言原始列表的大小仍然正确
  EXPECT_EQ(2, list1.size());
  // 断言原始列表的第一个元素仍然正确
  EXPECT_EQ("3", list1.get(0));
  // 断言原始列表的第二个元素仍然正确
  EXPECT_EQ("4", list1.get(1));
}

TEST(ListTestIValueBasedList, givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition) {
  // 创建一个包含字符串的 List 对象
  List<string> list({"3", "4"});

  // 获取第一个元素的迭代器
  List<string>::iterator iter1 = list.begin();
  // 后缀递增迭代器，返回递增前的位置
  List<string>::iterator iter2 = iter1++;
  // 断言递增后的迭代器指向的元素不是第一个元素
  EXPECT_NE("3", static_cast<string>(*iter1));
  // 断言递增前的迭代器指向的元素是第一个元素
  EXPECT_EQ("3", static_cast<string>(*iter2));
}

TEST(ListTestIValueBasedList, givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition) {
  // 创建一个包含字符串的 List 对象
  List<string> list({"3", "4"});

  // 获取第一个元素的迭代器
  List<string>::iterator iter1 = list.begin();
  // 前缀递增迭代器，返回递增后的位置
  List<string>::iterator iter2 = ++iter1;
  // 断言递增后的迭代器指向的元素不是第一个元素
  EXPECT_NE("3", static_cast<string>(*iter1));
  // 断言递增后的迭代器指向的元素也不是第一个元素
  EXPECT_NE("3", static_cast<string>(*iter2));
}
TEST(ListTestIValueBasedList, givenIterator_whenPostfixDecrementing_thenMovesToNextAndReturnsOldPosition) {
  // 创建一个字符串类型的链表，并初始化为 {"3", "4"}
  List<string> list({"3", "4"});

  // 获取指向链表倒数第二个元素的迭代器
  List<string>::iterator iter1 = list.end() - 1;
  // 使用后缀递减操作符递减迭代器，返回旧的位置
  List<string>::iterator iter2 = iter1--;
  // 断言迭代器 iter1 指向的元素不是 "4"
  EXPECT_NE("4", static_cast<string>(*iter1));
  // 断言迭代器 iter2 指向的元素是 "4"
  EXPECT_EQ("4", static_cast<string>(*iter2));
}

TEST(ListTestIValueBasedList, givenIterator_whenPrefixDecrementing_thenMovesToNextAndReturnsNewPosition) {
  // 创建一个字符串类型的链表，并初始化为 {"3", "4"}
  List<string> list({"3", "4"});

  // 获取指向链表倒数第二个元素的迭代器
  List<string>::iterator iter1 = list.end() - 1;
  // 使用前缀递减操作符递减迭代器，返回新的位置
  List<string>::iterator iter2 = --iter1;
  // 断言迭代器 iter1 指向的元素不是 "4"
  EXPECT_NE("4", static_cast<string>(*iter1));
  // 断言迭代器 iter2 指向的元素不是 "4"
  EXPECT_NE("4", static_cast<string>(*iter2));
}

TEST(ListTestIValueBasedList, givenIterator_whenIncreasing_thenMovesToNextAndReturnsNewPosition) {
  // 创建一个字符串类型的链表，并初始化为 {"3", "4", "5"}
  List<string> list({"3", "4", "5"});

  // 获取指向链表第一个元素的迭代器
  List<string>::iterator iter1 = list.begin();
  // 使用递增操作符将迭代器移动两个位置，返回新的位置
  List<string>::iterator iter2 = iter1 += 2;
  // 断言迭代器 iter1 指向的元素是 "5"
  EXPECT_EQ("5", static_cast<string>(*iter1));
  // 断言迭代器 iter2 指向的元素是 "5"
  EXPECT_EQ("5", static_cast<string>(*iter2));
}

TEST(ListTestIValueBasedList, givenIterator_whenDecreasing_thenMovesToNextAndReturnsNewPosition) {
  // 创建一个字符串类型的链表，并初始化为 {"3", "4", "5"}
  List<string> list({"3", "4", "5"});

  // 获取指向链表末尾的迭代器
  List<string>::iterator iter1 = list.end();
  // 使用递减操作符将迭代器向前移动两个位置，返回新的位置
  List<string>::iterator iter2 = iter1 -= 2;
  // 断言迭代器 iter1 指向的元素是 "4"
  EXPECT_EQ("4", static_cast<string>(*iter1));
  // 断言迭代器 iter2 指向的元素是 "4"
  EXPECT_EQ("4", static_cast<string>(*iter2));
}

TEST(ListTestIValueBasedList, givenIterator_whenAdding_thenReturnsNewIterator) {
  // 创建一个字符串类型的链表，并初始化为 {"3", "4", "5"}
  List<string> list({"3", "4", "5"});

  // 获取指向链表第一个元素的迭代器
  List<string>::iterator iter1 = list.begin();
  // 使用加法操作符将迭代器向后移动两个位置，返回新的迭代器
  List<string>::iterator iter2 = iter1 + 2;
  // 断言迭代器 iter1 指向的元素是 "3"
  EXPECT_EQ("3", static_cast<string>(*iter1));
  // 断言迭代器 iter2 指向的元素是 "5"
  EXPECT_EQ("5", static_cast<string>(*iter2));
}

TEST(ListTestIValueBasedList, givenIterator_whenSubtracting_thenReturnsNewIterator) {
  // 创建一个字符串类型的链表，并初始化为 {"3", "4", "5"}
  List<string> list({"3", "4", "5"});

  // 获取指向链表倒数第二个元素的迭代器
  List<string>::iterator iter1 = list.end() - 1;
  // 使用减法操作符将迭代器向前移动两个位置，返回新的迭代器
  List<string>::iterator iter2 = iter1 - 2;
  // 断言迭代器 iter1 指向的元素是 "5"
  EXPECT_EQ("5", static_cast<string>(*iter1));
  // 断言迭代器 iter2 指向的元素是 "3"
  EXPECT_EQ("3", static_cast<string>(*iter2));
}

TEST(ListTestIValueBasedList, givenIterator_whenCalculatingDifference_thenReturnsCorrectNumber) {
  // 创建一个字符串类型的链表，并初始化为 {"3", "4"}
  List<string> list({"3", "4"});
  // 计算链表末尾迭代器和开头迭代器之间的距离
  EXPECT_EQ(2, list.end() - list.begin());
}

TEST(ListTestIValueBasedList, givenEqualIterators_thenAreEqual) {
  // 创建一个字符串类型的链表，并初始化为 {"3", "4"}
  List<string> list({"3", "4"});

  // 获取链表的两个相等的迭代器
  List<string>::iterator iter1 = list.begin();
  List<string>::iterator iter2 = list.begin();
  // 断言 iter1 和 iter2 是相等的
  EXPECT_TRUE(iter1 == iter2);
  EXPECT_FALSE(iter1 != iter2);
}

TEST(ListTestIValueBasedList, givenDifferentIterators_thenAreNotEqual) {
  // 创建一个字符串类型的链表，并初始化为 {"3", "4"}
  List<string> list({"3", "4"});

  // 获取链表的两个不同的迭代器，iter2 指向第二个元素
  List<string>::iterator iter1 = list.begin();
  List<string>::iterator iter2 = list.begin();
  iter2++;

  // 断言 iter1 和 iter2 不相等
  EXPECT_FALSE(iter1 == iter2);
  EXPECT_TRUE(iter1 != iter2);
}

TEST(ListTestIValueBasedList, givenIterator_whenDereferencing_thenPointsToCorrectElement) {
  // 创建一个字符串类型的链表，并初始化为 {"3"}
  List<string> list({"3"});

  // 获取指向链表第一个元素的迭代器
  List<string>::iterator iter = list.begin();
  // 断言迭代器指向的元素是 "3"
  EXPECT_EQ("3", static_cast<string>(*iter));
}
TEST(ListTestIValueBasedList, givenIterator_whenAssigningNewValue_thenChangesValue) {
  // 创建一个包含单个字符串 "3" 的列表
  List<string> list({"3"});

  // 获取列表的迭代器
  List<string>::iterator iter = list.begin();
  
  // 通过迭代器修改列表中第一个元素的值为 "4"
  *iter = "4";

  // 断言修改后列表的第一个元素为 "4"
  EXPECT_EQ("4", list.get(0));
}

TEST(ListTestIValueBasedList, givenIterator_whenAssigningNewValueFromIterator_thenChangesValue) {
  // 创建一个包含两个字符串 "3" 和 "4" 的列表
  List<string> list({"3", "4"});

  // 获取列表的迭代器
  List<string>::iterator iter = list.begin();
  
  // 将第一个元素的值修改为第二个元素的值
  *iter = *(iter + 1);

  // 断言修改后列表的第一个和第二个元素均为 "4"
  EXPECT_EQ("4", list.get(0));
  EXPECT_EQ("4", list.get(1));
}

TEST(ListTestIValueBasedList, givenIterator_whenSwappingValuesFromIterator_thenChangesValue) {
  // 创建一个包含两个字符串 "3" 和 "4" 的列表
  List<string> list({"3", "4"});

  // 获取列表的迭代器
  List<string>::iterator iter = list.begin();
  
  // 交换第一个和第二个元素的值
  swap(*iter, *(iter + 1));

  // 断言交换后列表的第一个元素为 "4"，第二个元素为 "3"
  EXPECT_EQ("4", list.get(0));
  EXPECT_EQ("3", list.get(1));
}

TEST(ListTestIValueBasedList, givenOneElementList_whenCallingPopBack_thenIsEmpty) {
  // 创建一个包含单个字符串 "3" 的列表
  List<string> list({"3"});
  
  // 移除列表的最后一个元素
  list.pop_back();
  
  // 断言列表现在为空
  EXPECT_TRUE(list.empty());
}

TEST(ListTestIValueBasedList, givenEmptyList_whenCallingResize_thenResizesAndSetsEmptyValue) {
  // 创建一个空列表
  List<string> list;
  
  // 调整列表大小为 2，并且新添加的元素使用默认空值（空字符串）
  list.resize(2);
  
  // 断言列表的大小为 2，且新元素均为空字符串
  EXPECT_EQ(2, list.size());
  EXPECT_EQ("", list.get(0));
  EXPECT_EQ("", list.get(1));
}

TEST(ListTestIValueBasedList, givenEmptyList_whenCallingResizeWithValue_thenResizesAndSetsValue) {
  // 创建一个空列表
  List<string> list;
  
  // 调整列表大小为 2，并且新添加的元素均为指定的值 "value"
  list.resize(2, "value");
  
  // 断言列表的大小为 2，且新元素均为 "value"
  EXPECT_EQ(2, list.size());
  EXPECT_EQ("value", list.get(0));
  EXPECT_EQ("value", list.get(1));
}

TEST(ListTestIValueBasedList, isReferenceType) {
  // 创建一个空列表 list1
  List<string> list1;
  
  // 使用拷贝构造函数创建列表 list2，并声明列表 list3
  List<string> list2(list1);
  List<string> list3;
  
  // 向列表 list1 添加一个元素
  list1.push_back("three");
  
  // 断言三个列表的大小均为 1，说明它们共享同一份数据
  EXPECT_EQ(1, list1.size());
  EXPECT_EQ(1, list2.size());
  EXPECT_EQ(1, list3.size());
}

TEST(ListTestIValueBasedList, copyHasSeparateStorage) {
  // 创建一个空列表 list1
  List<string> list1;
  
  // 使用 copy 方法创建 list1 的副本 list2，并声明列表 list3
  List<string> list2(list1.copy());
  List<string> list3;
  
  // 向列表 list1 添加一个元素
  list1.push_back("three");
  
  // 断言 list1 的大小为 1，list2 和 list3 的大小为 0，说明它们拥有独立的存储空间
  EXPECT_EQ(1, list1.size());
  EXPECT_EQ(0, list2.size());
  EXPECT_EQ(0, list3.size());
}

TEST(ListTestIValueBasedList, givenEqualLists_thenIsEqual) {
  // 创建两个包含相同元素的列表
  List<string> list1({"first", "second"});
  List<string> list2({"first", "second"});
  
  // 断言这两个列表相等
  EXPECT_EQ(list1, list2);
}

TEST(ListTestIValueBasedList, givenDifferentLists_thenIsNotEqual) {
  // 创建两个包含不同元素的列表
  List<string> list1({"first", "second"});
  List<string> list2({"first", "not_second"});
  
  // 断言这两个列表不相等
  EXPECT_NE(list1, list2);
}

TEST(ListTestNonIValueBasedList, givenEmptyList_whenCallingEmpty_thenReturnsTrue) {
  // 创建一个空列表
  List<int64_t> list;
  
  // 断言列表为空
  EXPECT_TRUE(list.empty());
}

TEST(ListTestNonIValueBasedList, givenNonemptyList_whenCallingEmpty_thenReturnsFalse) {
  // 创建一个包含一个整数 3 的列表
  List<int64_t> list({3});
  
  // 断言列表不为空
  EXPECT_FALSE(list.empty());
}

TEST(ListTestNonIValueBasedList, givenEmptyList_whenCallingSize_thenReturnsZero) {
  // 创建一个空列表
  List<int64_t> list;
  
  // 断言列表的大小为 0
  EXPECT_EQ(0, list.size());
}

TEST(ListTestNonIValueBasedList, givenNonemptyList_whenCallingSize_thenReturnsNumberOfElements) {
  // 创建一个包含两个整数 3 和 4 的列表
  List<int64_t> list({3, 4});
  
  // 断言列表的大小为 2
  EXPECT_EQ(2, list.size());
}
TEST(ListTestNonIValueBasedList, givenNonemptyList_whenCallingClear_thenIsEmpty) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  // 清空列表
  list.clear();
  // 断言列表为空
  EXPECT_TRUE(list.empty());
}

TEST(ListTestNonIValueBasedList, whenCallingGetWithExistingPosition_thenReturnsElement) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  // 断言获取位置 0 的元素为 3
  EXPECT_EQ(3, list.get(0));
  // 断言获取位置 1 的元素为 4
  EXPECT_EQ(4, list.get(1));
}

TEST(ListTestNonIValueBasedList, whenCallingGetWithNonExistingPosition_thenThrowsException) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // 断言调用获取位置 2 的元素会抛出 std::out_of_range 异常
  EXPECT_THROW(list.get(2), std::out_of_range);
}

TEST(ListTestNonIValueBasedList, whenCallingExtractWithExistingPosition_thenReturnsElement) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  // 断言提取位置 0 的元素为 3
  EXPECT_EQ(3, list.extract(0));
  // 断言提取位置 1 的元素为 4
  EXPECT_EQ(4, list.extract(1));
}

TEST(ListTestNonIValueBasedList, whenCallingExtractWithNonExistingPosition_thenThrowsException) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // 断言调用提取位置 2 的元素会抛出 std::out_of_range 异常
  EXPECT_THROW(list.extract(2), std::out_of_range);
}

TEST(ListTestNonIValueBasedList, whenCallingCopyingSetWithExistingPosition_thenChangesElement) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  int64_t value = 5;
  // 修改位置 1 的元素为 5
  list.set(1, value);
  // 断言获取位置 0 的元素为 3
  EXPECT_EQ(3, list.get(0));
  // 断言获取位置 1 的元素为 5
  EXPECT_EQ(5, list.get(1));
}

TEST(ListTestNonIValueBasedList, whenCallingMovingSetWithExistingPosition_thenChangesElement) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  int64_t value = 5;
  // NOLINTNEXTLINE(performance-move-const-arg)
  // 使用移动语义将值 5 移动到位置 1
  list.set(1, std::move(value));
  // 断言获取位置 0 的元素为 3
  EXPECT_EQ(3, list.get(0));
  // 断言获取位置 1 的元素为 5
  EXPECT_EQ(5, list.get(1));
}

TEST(ListTestNonIValueBasedList, whenCallingCopyingSetWithNonExistingPosition_thenThrowsException) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  int64_t value = 5;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // 断言调用设置位置 2 的元素会抛出 std::out_of_range 异常
  EXPECT_THROW(list.set(2, value), std::out_of_range);
}

TEST(ListTestNonIValueBasedList, whenCallingMovingSetWithNonExistingPosition_thenThrowsException) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  int64_t value = 5;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,performance-move-const-arg,hicpp-avoid-goto)
  // 断言调用设置位置 2 的元素会抛出 std::out_of_range 异常
  EXPECT_THROW(list.set(2, std::move(value)), std::out_of_range);
}

TEST(ListTestNonIValueBasedList, whenCallingAccessOperatorWithExistingPosition_thenReturnsElement) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  // 断言通过访问操作符获取位置 0 的元素为 3
  EXPECT_EQ(3, static_cast<int64_t>(list[0]));
  // 断言通过访问操作符获取位置 1 的元素为 4
  EXPECT_EQ(4, static_cast<int64_t>(list[1]));
}

TEST(ListTestNonIValueBasedList, whenAssigningToAccessOperatorWithExistingPosition_thenSetsElement) {
  // 创建一个整型列表，初始化为 {3, 4, 5}
  List<int64_t> list({3, 4, 5});
  // 将位置 1 的元素设置为 6
  list[1] = 6;
  // 断言获取位置 0 的元素为 3
  EXPECT_EQ(3, list.get(0));
  // 断言获取位置 1 的元素为 6
  EXPECT_EQ(6, list.get(1));
  // 断言获取位置 2 的元素为 5
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTestNonIValueBasedList, whenAssigningToAccessOperatorFromAccessOperator_thenSetsElement) {
  // 创建一个整型列表，初始化为 {3, 4, 5}
  List<int64_t> list({3, 4, 5});
  // 将位置 1 的元素设置为与位置 2 相同的值
  list[1] = list[2];
  // 断言获取位置 0 的元素为 3
  EXPECT_EQ(3, list.get(0));
  // 断言
TEST(ListTestNonIValueBasedList, whenSwappingFromAccessOperator_thenSwapsElements) {
  // 创建一个整型列表，初始化为 {3, 4, 5}
  List<int64_t> list({3, 4, 5});
  // 交换索引为 1 和 2 的元素
  swap(list[1], list[2]);
  // 检查列表中索引为 0 的元素是否为 3
  EXPECT_EQ(3, list.get(0));
  // 检查列表中索引为 1 的元素是否为 5
  EXPECT_EQ(5, list.get(1));
  // 检查列表中索引为 2 的元素是否为 4
  EXPECT_EQ(4, list.get(2));
}

TEST(ListTestNonIValueBasedList, whenCallingAccessOperatorWithNonExistingPosition_thenThrowsException) {
  // 创建一个整型列表，初始化为 {3, 4}
  List<int64_t> list({3, 4});
  // 测试访问不存在的位置（索引 2），期待抛出 std::out_of_range 异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(list[2], std::out_of_range);
}

TEST(ListTestNonIValueBasedList, whenCallingInsertOnIteratorWithLValue_thenInsertsElement) {
  // 创建一个整型列表，初始化为 {3, 4, 6}
  List<int64_t> list({3, 4, 6});
  // 定义整型变量 v 为 5
  int64_t v = 5;
  // 在迭代器指向的位置（索引为 2）插入值为 v 的元素
  list.insert(list.begin() + 2, v);
  // 检查列表的大小是否为 4
  EXPECT_EQ(4, list.size());
  // 检查列表中索引为 2 的元素是否为 5
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTestNonIValueBasedList, whenCallingInsertOnIteratorWithRValue_thenInsertsElement) {
  // 创建一个整型列表，初始化为 {3, 4, 6}
  List<int64_t> list({3, 4, 6});
  // 定义整型变量 v 为 5
  int64_t v = 5;
  // 在迭代器指向的位置（索引为 2）插入移动后的 v 元素
  // NOLINTNEXTLINE(performance-move-const-arg)
  list.insert(list.begin() + 2, std::move(v));
  // 检查列表的大小是否为 4
  EXPECT_EQ(4, list.size());
  // 检查列表中索引为 2 的元素是否为 5
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTestNonIValueBasedList, whenCallingInsertWithLValue_thenReturnsIteratorToNewElement) {
  // 创建一个整型列表，初始化为 {3, 4, 6}
  List<int64_t> list({3, 4, 6});
  // 定义整型变量 v 为 5
  int64_t v = 5;
  // 在迭代器指向的位置（索引为 2）插入值为 v 的元素，并获取返回的迭代器
  List<int64_t>::iterator result = list.insert(list.begin() + 2, v);
  // 检查返回的迭代器是否指向索引为 2 的位置
  EXPECT_EQ(list.begin() + 2, result);
}

TEST(ListTestNonIValueBasedList, whenCallingInsertWithRValue_thenReturnsIteratorToNewElement) {
  // 创建一个整型列表，初始化为 {3, 4, 6}
  List<int64_t> list({3, 4, 6});
  // 定义整型变量 v 为 5
  int64_t v = 5;
  // 在迭代器指向的位置（索引为 2）插入移动后的 v 元素，并获取返回的迭代器
  // NOLINTNEXTLINE(performance-move-const-arg)
  List<int64_t>::iterator result = list.insert(list.begin() + 2, std::move(v));
  // 检查返回的迭代器是否指向索引为 2 的位置
  EXPECT_EQ(list.begin() + 2, result);
}

TEST(ListTestNonIValueBasedList, whenCallingEmplaceWithLValue_thenInsertsElement) {
  // 创建一个整型列表，初始化为 {3, 4, 6}
  List<int64_t> list({3, 4, 6});
  // 定义整型变量 v 为 5
  int64_t v = 5;
  // 在迭代器指向的位置（索引为 2）使用 v 构造新元素
  list.emplace(list.begin() + 2, v);
  // 检查列表的大小是否为 4
  EXPECT_EQ(4, list.size());
  // 检查列表中索引为 2 的元素是否为 5
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTestNonIValueBasedList, whenCallingEmplaceWithRValue_thenInsertsElement) {
  // 创建一个整型列表，初始化为 {3, 4, 6}
  List<int64_t> list({3, 4, 6});
  // 定义整型变量 v 为 5
  int64_t v = 5;
  // 在迭代器指向的位置（索引为 2）使用移动后的 v 构造新元素
  // NOLINTNEXTLINE(performance-move-const-arg)
  list.emplace(list.begin() + 2, std::move(v));
  // 检查列表的大小是否为 4
  EXPECT_EQ(4, list.size());
  // 检查列表中索引为 2 的元素是否为 5
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTestNonIValueBasedList, whenCallingEmplaceWithConstructorArg_thenInsertsElement) {
  // 创建一个整型列表，初始化为 {3, 4, 6}
  List<int64_t> list({3, 4, 6});
  // 在迭代器指向的位置（索引为 2）使用构造参数 5 构造新元素
  list.emplace(list.begin() + 2, 5); // const char* is a constructor arg to std::int64_t
  // 检查列表的大小是否为 4
  EXPECT_EQ(4, list.size());
  // 检查列表中索引为 2 的元素是否为 5
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTestNonIValueBasedList, whenCallingPushBackWithLValue_ThenInsertsElement) {
  // 创建一个空整型列表
  List<int64_t> list;
  // 定义整型变量 v 为 5
  int64_t v = 5;
  // 将值为 v 的元素添加到列表末尾
  list.push_back(v);
  // 检查列表的大小是否为 1
  EXPECT_EQ(1, list.size());
  // 检查列表中索引为 0 的元素是否为 5
  EXPECT_EQ(5, list.get(0));
}

TEST(ListTestNonIValueBasedList, whenCallingPushBackWithRValue_ThenInsertsElement) {
  // 创建一个空整型列表
  List<int64_t> list;
  // 定义整型变量 v 为 5
  int64_t v = 5;
  // 将移动后的 v 元素添加到列表末尾
  // NOLINTNEXTLINE(performance-move-const-arg)
  list.push_back(std::move(v));
  // 检查列表的大小是否为 1
  EXPECT_EQ(1, list.size());
  // 检查列表中索引为 0 的元素是否为 5
  EXPECT_EQ(5, list.get(0));
}
TEST(ListTestNonIValueBasedList, whenCallingEmplaceBackWithLValue_ThenInsertsElement) {
  // 创建一个空的 List 对象，其中元素类型为 int64_t
  List<int64_t> list;
  // 定义整数变量 v 并赋值为 5
  int64_t v = 5;
  // 使用 lvalue v 调用 emplace_back 方法，将元素插入列表末尾
  list.emplace_back(v);
  // 断言列表的大小为 1
  EXPECT_EQ(1, list.size());
  // 断言列表中第一个元素的值为 5
  EXPECT_EQ(5, list.get(0));
}

TEST(ListTestNonIValueBasedList, whenCallingEmplaceBackWithRValue_ThenInsertsElement) {
  // 创建一个空的 List 对象，其中元素类型为 int64_t
  List<int64_t> list;
  // 定义整数变量 v 并赋值为 5
  int64_t v = 5;
  // 使用 rvalue std::move(v) 调用 emplace_back 方法，将元素插入列表末尾
  // NOLINTNEXTLINE(performance-move-const-arg) 标记不检查性能问题
  list.emplace_back(std::move(v));
  // 断言列表的大小为 1
  EXPECT_EQ(1, list.size());
  // 断言列表中第一个元素的值为 5
  EXPECT_EQ(5, list.get(0));
}

TEST(ListTestNonIValueBasedList, whenCallingEmplaceBackWithConstructorArg_ThenInsertsElement) {
  // 创建一个空的 List 对象，其中元素类型为 int64_t
  List<int64_t> list;
  // 使用构造参数 5 调用 emplace_back 方法，将元素插入列表末尾
  list.emplace_back(5);  // const char* is a constructor arg to std::int64_t
  // 断言列表的大小为 1
  EXPECT_EQ(1, list.size());
  // 断言列表中第一个元素的值为 5
  EXPECT_EQ(5, list.get(0));
}

TEST(ListTestNonIValueBasedList, givenEmptyList_whenIterating_thenBeginIsEnd) {
  // 创建一个空的 List 对象，其中元素类型为 int64_t
  List<int64_t> list;
  // 创建一个空的 const List 对象，其中元素类型为 int64_t
  const List<int64_t> clist;
  // 断言空列表的 begin() 和 end() 相等
  EXPECT_EQ(list.begin(), list.end());
  // 断言空列表的 begin() 和 end() 相等
  EXPECT_EQ(list.begin(), list.end());
  // 断言空 const 列表的 begin() 和 end() 相等
  EXPECT_EQ(clist.begin(), clist.end());
  // 断言空 const 列表的 begin() 和 end() 相等
  EXPECT_EQ(clist.begin(), clist.end());
}

TEST(ListTestNonIValueBasedList, whenIterating_thenFindsElements) {
  // 创建一个包含元素 3 和 5 的 List 对象，其中元素类型为 int64_t
  List<int64_t> list({3, 5});
  // 定义两个布尔变量，用于标记是否找到特定元素
  bool found_first = false;
  bool found_second = false;
  // 对列表进行迭代，使用 const auto && iter 作为迭代变量
  for (const auto && iter : list) {
    // 如果迭代元素为 3，则断言未找到第一个元素，然后标记为已找到
    if (static_cast<int64_t>(iter) == 3) {
      EXPECT_FALSE(found_first);
      found_first = true;
    // 如果迭代元素为 5，则断言未找到第二个元素，然后标记为已找到
    } else if (static_cast<int64_t>(iter) == 5) {
      EXPECT_FALSE(found_second);
      found_second = true;
    // 如果迭代元素既不是 3 也不是 5，则标记为失败
    } else {
      ADD_FAILURE();
    }
  }
  // 断言已找到列表中的第一个和第二个元素
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(ListTestNonIValueBasedList, whenIteratingWithForeach_thenFindsElements) {
  // 创建一个包含元素 3 和 5 的 List 对象，其中元素类型为 int64_t
  List<int64_t> list({3, 5});
  // 定义两个布尔变量，用于标记是否找到特定元素
  bool found_first = false;
  bool found_second = false;
  // 对列表进行迭代，使用 const int64_t& elem 作为迭代变量
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop) 标记不检查性能问题
  for (const int64_t& elem : list) {
    // 如果迭代元素为 3，则断言未找到第一个元素，然后标记为已找到
    if (elem == 3) {
      EXPECT_FALSE(found_first);
      found_first = true;
    // 如果迭代元素为 5，则断言未找到第二个元素，然后标记为已找到
    } else if (elem == 5) {
      EXPECT_FALSE(found_second);
      found_second = true;
    // 如果迭代元素既不是 3 也不是 5，则标记为失败
    } else {
      ADD_FAILURE();
    }
  }
  // 断言已找到列表中的第一个和第二个元素
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(ListTestNonIValueBasedList, givenOneElementList_whenErasing_thenListIsEmpty) {
  // 创建一个包含元素 3 的 List 对象，其中元素类型为 int64_t
  List<int64_t> list({3});
  // 移除列表的第一个元素
  list.erase(list.begin());
  // 断言列表为空
  EXPECT_TRUE(list.empty());
}

TEST(ListTestNonIValueBasedList, givenList_whenErasing_thenReturnsIterator) {
  // 创建一个包含元素 1, 2, 3 的 List 对象，其中元素类型为 int64_t
  List<int64_t> list({1, 2, 3});
  // 移除列表的第二个元素，并返回指向下一个元素的迭代器
  List<int64_t>::iterator iter = list.erase(list.begin() + 1);
  // 断言返回的迭代器与列表的 begin() + 1 相等
  EXPECT_EQ(list.begin() + 1, iter);
}

TEST(ListTestNonIValueBasedList, givenList_whenErasingFullRange_thenIsEmpty) {
  // 创建一个包含元素 1, 2, 3 的 List 对象，其中元素类型为 int64_t
  List<int64_t> list({1, 2, 3});
  // 移除列表的所有元素
  list.erase(list.begin(), list.end());
  // 断言列表为空
  EXPECT_TRUE(list.empty());
}

TEST(ListTestNonIValueBasedList, whenCallingReserve_thenDoesntCrash) {
  // 创建一个空的 List 对象，其中元素类型为 int64_t
  List<int64_t> list;
  // 调用 reserve 方法预留空间，预期不会发生崩溃
  list.reserve(100);
}
TEST(ListTestNonIValueBasedList, whenCopyConstructingList_thenAreEqual) {
  // 创建一个 List 对象 list1，其中包含元素 {3, 4}
  List<int64_t> list1({3, 4});

  // 使用复制构造函数，基于 list1 创建 list2
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  List<int64_t> list2(list1);

  // 断言 list2 的大小为 2
  EXPECT_EQ(2, list2.size());
  // 断言 list2 的第一个元素为 3
  EXPECT_EQ(3, list2.get(0));
  // 断言 list2 的第二个元素为 4
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTestNonIValueBasedList, whenCopyAssigningList_thenAreEqual) {
  // 创建一个 List 对象 list1，其中包含元素 {3, 4}
  List<int64_t> list1({3, 4});

  // 使用赋值运算符，将 list1 复制给 list2
  List<int64_t> list2;
  list2 = list1;

  // 断言 list2 的大小为 2
  EXPECT_EQ(2, list2.size());
  // 断言 list2 的第一个元素为 3
  EXPECT_EQ(3, list2.get(0));
  // 断言 list2 的第二个元素为 4
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTestNonIValueBasedList, whenCopyingList_thenAreEqual) {
  // 创建一个 List 对象 list1，其中包含元素 {3, 4}
  List<int64_t> list1({3, 4});

  // 使用 copy 方法复制 list1 到 list2
  List<int64_t> list2 = list1.copy();

  // 断言 list2 的大小为 2
  EXPECT_EQ(2, list2.size());
  // 断言 list2 的第一个元素为 3
  EXPECT_EQ(3, list2.get(0));
  // 断言 list2 的第二个元素为 4
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTestNonIValueBasedList, whenMoveConstructingList_thenNewIsCorrect) {
  // 创建一个 List 对象 list1，其中包含元素 {3, 4}
  List<int64_t> list1({3, 4});

  // 使用移动构造函数，将 list1 的内容移动到 list2
  List<int64_t> list2(std::move(list1));

  // 断言 list2 的大小为 2
  EXPECT_EQ(2, list2.size());
  // 断言 list2 的第一个元素为 3
  EXPECT_EQ(3, list2.get(0));
  // 断言 list2 的第二个元素为 4
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTestNonIValueBasedList, whenMoveAssigningList_thenNewIsCorrect) {
  // 创建一个 List 对象 list1，其中包含元素 {3, 4}
  List<int64_t> list1({3, 4});

  // 使用移动赋值运算符，将 list1 的内容移动到 list2
  List<int64_t> list2;
  list2 = std::move(list1);

  // 断言 list2 的大小为 2
  EXPECT_EQ(2, list2.size());
  // 断言 list2 的第一个元素为 3
  EXPECT_EQ(3, list2.get(0));
  // 断言 list2 的第二个元素为 4
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTestNonIValueBasedList, whenMoveConstructingList_thenOldIsUnchanged) {
  // 创建一个 List 对象 list1，其中包含元素 {3, 4}
  List<int64_t> list1({3, 4});

  // 使用移动构造函数，将 list1 的内容移动到 list2
  List<int64_t> list2(std::move(list1));

  // 断言 list1 的大小为 2，即原始的内容未改变
  EXPECT_EQ(2, list1.size());
  // 断言 list1 的第一个元素为 3
  EXPECT_EQ(3, list1.get(0));
  // 断言 list1 的第二个元素为 4
  EXPECT_EQ(4, list1.get(1));
}

TEST(ListTestNonIValueBasedList, whenMoveAssigningList_thenOldIsUnchanged) {
  // 创建一个 List 对象 list1，其中包含元素 {3, 4}
  List<int64_t> list1({3, 4});

  // 使用移动赋值运算符，将 list1 的内容移动到 list2
  List<int64_t> list2;
  list2 = std::move(list1);

  // 断言 list1 的大小为 2，即原始的内容未改变
  EXPECT_EQ(2, list1.size());
  // 断言 list1 的第一个元素为 3
  EXPECT_EQ(3, list1.get(0));
  // 断言 list1 的第二个元素为 4
  EXPECT_EQ(4, list1.get(1));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition) {
  // 创建一个 List 对象 list，其中包含元素 {3, 4}
  List<int64_t> list({3, 4});

  // 获取 list 的起始迭代器
  List<int64_t>::iterator iter1 = list.begin();
  // 使用后置递增操作符，移动迭代器 iter1 到下一个位置，并将旧位置保存在 iter2 中
  List<int64_t>::iterator iter2 = iter1++;
  // 断言 iter1 指向的元素不是 3
  EXPECT_NE(3, static_cast<int64_t>(*iter1));
  // 断言 iter2 指向的元素是 3
  EXPECT_EQ(3, static_cast<int64_t>(*iter2));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition) {
  // 创建一个 List 对象 list，其中包含元素 {3, 4}
  List<int64_t> list({3, 4});

  // 获取 list 的起始迭代器
  List<int64_t>::iterator iter1 = list.begin();
  // 使用前置递增操作符，先移动 iter1 到下一个位置，然后将新位置保存在 iter2 中
  List<int64_t>::iterator iter2 = ++iter1;
  // 断言 iter1 指向的元素不是 3
  EXPECT_NE(3, static_cast<int64_t>(*iter1));
  // 断言 iter2 指向的元素也不是 3
  EXPECT_NE(3, static_cast<int64_t>(*iter2));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenPostfixDecrementing_thenMovesToNextAndReturnsOldPosition) {
  // 创建一个 List 对象 list，其中包含元素 {3, 4}
  List<int64_t> list({3, 4});

  // 获取 list 的末尾位置的迭代器，然后向前移动一个位置，并将旧位置保存在 iter2 中
  List<int64_t>::iterator iter1 = list.end() - 1;
  List<int64_t>::iterator iter2 = iter1--;
  // 断言 iter1 指向的元素不是 4
  EXPECT_NE(4, static_cast<int64_t>(*iter1));
  // 断言 iter2 指向的元素是 4
  EXPECT_EQ(4, static_cast<int64_t>(*iter2));
}
TEST(ListTestNonIValueBasedList, givenIterator_whenPrefixDecrementing_thenMovesToNextAndReturnsNewPosition) {
  // 创建一个整型列表，包含元素 {3, 4}
  List<int64_t> list({3, 4});

  // 取得指向列表末尾前一个位置的迭代器
  List<int64_t>::iterator iter1 = list.end() - 1;
  // 对迭代器进行前缀递减操作，移动到前一个位置，并返回新的迭代器
  List<int64_t>::iterator iter2 = --iter1;

  // 断言：iter1 指向的值不等于 4
  EXPECT_NE(4, static_cast<int64_t>(*iter1));
  // 断言：iter2 指向的值不等于 4
  EXPECT_NE(4, static_cast<int64_t>(*iter2));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenIncreasing_thenMovesToNextAndReturnsNewPosition) {
  // 创建一个整型列表，包含元素 {3, 4, 5}
  List<int64_t> list({3, 4, 5});

  // 取得指向列表开头的迭代器
  List<int64_t>::iterator iter1 = list.begin();
  // 对迭代器进行增量操作，移动到当前位置后两个位置，并返回新的迭代器
  List<int64_t>::iterator iter2 = iter1 += 2;

  // 断言：iter1 指向的值等于 5
  EXPECT_EQ(5, static_cast<int64_t>(*iter1));
  // 断言：iter2 指向的值等于 5
  EXPECT_EQ(5, static_cast<int64_t>(*iter2));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenDecreasing_thenMovesToNextAndReturnsNewPosition) {
  // 创建一个整型列表，包含元素 {3, 4, 5}
  List<int64_t> list({3, 4, 5});

  // 取得指向列表末尾的迭代器
  List<int64_t>::iterator iter1 = list.end();
  // 对迭代器进行减量操作，移动到当前位置前两个位置，并返回新的迭代器
  List<int64_t>::iterator iter2 = iter1 -= 2;

  // 断言：iter1 指向的值等于 4
  EXPECT_EQ(4, static_cast<int64_t>(*iter1));
  // 断言：iter2 指向的值等于 4
  EXPECT_EQ(4, static_cast<int64_t>(*iter2));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenAdding_thenReturnsNewIterator) {
  // 创建一个整型列表，包含元素 {3, 4, 5}
  List<int64_t> list({3, 4, 5});

  // 取得指向列表开头的迭代器
  List<int64_t>::iterator iter1 = list.begin();
  // 对迭代器进行加法操作，返回当前位置后两个位置的新迭代器
  List<int64_t>::iterator iter2 = iter1 + 2;

  // 断言：iter1 指向的值等于 3
  EXPECT_EQ(3, static_cast<int64_t>(*iter1));
  // 断言：iter2 指向的值等于 5
  EXPECT_EQ(5, static_cast<int64_t>(*iter2));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenSubtracting_thenReturnsNewIterator) {
  // 创建一个整型列表，包含元素 {3, 4, 5}
  List<int64_t> list({3, 4, 5});

  // 取得指向列表末尾前一个位置的迭代器
  List<int64_t>::iterator iter1 = list.end() - 1;
  // 对迭代器进行减法操作，返回当前位置前两个位置的新迭代器
  List<int64_t>::iterator iter2 = iter1 - 2;

  // 断言：iter1 指向的值等于 5
  EXPECT_EQ(5, static_cast<int64_t>(*iter1));
  // 断言：iter2 指向的值等于 3
  EXPECT_EQ(3, static_cast<int64_t>(*iter2));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenCalculatingDifference_thenReturnsCorrectNumber) {
  // 创建一个整型列表，包含元素 {3, 4}
  List<int64_t> list({3, 4});

  // 计算列表末尾迭代器与开头迭代器之间的距离
  EXPECT_EQ(2, list.end() - list.begin());
}

TEST(ListTestNonIValueBasedList, givenEqualIterators_thenAreEqual) {
  // 创建一个整型列表，包含元素 {3, 4}
  List<int64_t> list({3, 4});

  // 取得列表开头的两个相等的迭代器
  List<int64_t>::iterator iter1 = list.begin();
  List<int64_t>::iterator iter2 = list.begin();

  // 断言：iter1 和 iter2 应相等
  EXPECT_TRUE(iter1 == iter2);
  // 断言：iter1 和 iter2 不应不等
  EXPECT_FALSE(iter1 != iter2);
}

TEST(ListTestNonIValueBasedList, givenDifferentIterators_thenAreNotEqual) {
  // 创建一个整型列表，包含元素 {3, 4}
  List<int64_t> list({3, 4});

  // 取得列表开头的两个不相等的迭代器
  List<int64_t>::iterator iter1 = list.begin();
  List<int64_t>::iterator iter2 = list.begin();
  iter2++;  // 移动 iter2 到下一个位置

  // 断言：iter1 和 iter2 不应相等
  EXPECT_FALSE(iter1 == iter2);
  // 断言：iter1 和 iter2 应不等
  EXPECT_TRUE(iter1 != iter2);
}

TEST(ListTestNonIValueBasedList, givenIterator_whenDereferencing_thenPointsToCorrectElement) {
  // 创建一个整型列表，包含元素 {3}
  List<int64_t> list({3});

  // 取得指向列表开头的迭代器
  List<int64_t>::iterator iter = list.begin();

  // 断言：迭代器指向的值应为 3
  EXPECT_EQ(3, static_cast<int64_t>(*iter));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenAssigningNewValue_thenChangesValue) {
  // 创建一个整型列表，包含元素 {3}
  List<int64_t> list({3});

  // 取得指向列表开头的迭代器
  List<int64_t>::iterator iter = list.begin();

  // 将迭代器指向的元素值改为 4
  *iter = 4;

  // 断言：列表中索引为 0 的元素值应为 4
  EXPECT_EQ(4, list.get(0));
}
TEST(ListTestNonIValueBasedList, givenIterator_whenAssigningNewValueFromIterator_thenChangesValue) {
  // 创建包含元素 3 和 4 的整数列表
  List<int64_t> list({3, 4});

  // 获取列表的迭代器并赋值给 iter
  List<int64_t>::iterator iter = list.begin();

  // 将 iter 所指向的元素赋值为迭代器下一个位置的元素值
  *iter = *(iter + 1);

  // 验证列表第一个位置的值是否为 4
  EXPECT_EQ(4, list.get(0));

  // 验证列表第二个位置的值是否为 4
  EXPECT_EQ(4, list.get(1));
}

TEST(ListTestNonIValueBasedList, givenIterator_whenSwappingValuesFromIterator_thenChangesValue) {
  // 创建包含元素 3 和 4 的整数列表
  List<int64_t> list({3, 4});

  // 获取列表的迭代器并赋值给 iter
  List<int64_t>::iterator iter = list.begin();

  // 交换 iter 和其下一个位置的元素值
  swap(*iter, *(iter + 1));

  // 验证列表第一个位置的值是否为 4
  EXPECT_EQ(4, list.get(0));

  // 验证列表第二个位置的值是否为 3
  EXPECT_EQ(3, list.get(1));
}

TEST(ListTestNonIValueBasedList, givenOneElementList_whenCallingPopBack_thenIsEmpty) {
  // 创建只包含元素 3 的整数列表
  List<int64_t> list({3});

  // 弹出列表最后一个元素
  list.pop_back();

  // 验证列表是否为空
  EXPECT_TRUE(list.empty());
}

TEST(ListTestNonIValueBasedList, givenEmptyList_whenCallingResize_thenResizesAndSetsEmptyValue) {
  // 创建空的整数列表
  List<int64_t> list;

  // 调整列表大小为 2，并设置所有元素为默认值 0
  list.resize(2);

  // 验证列表的大小是否为 2
  EXPECT_EQ(2, list.size());

  // 验证列表第一个位置的值是否为 0
  EXPECT_EQ(0, list.get(0));

  // 验证列表第二个位置的值是否为 0
  EXPECT_EQ(0, list.get(1));
}

TEST(ListTestNonIValueBasedList, givenEmptyList_whenCallingResizeWithValue_thenResizesAndSetsValue) {
  // 创建空的整数列表
  List<int64_t> list;

  // 调整列表大小为 2，并设置所有元素为值 5
  list.resize(2, 5);

  // 验证列表的大小是否为 2
  EXPECT_EQ(2, list.size());

  // 验证列表第一个位置的值是否为 5
  EXPECT_EQ(5, list.get(0));

  // 验证列表第二个位置的值是否为 5
  EXPECT_EQ(5, list.get(1));
}

TEST(ListTestNonIValueBasedList, isReferenceType) {
  // 创建空的整数列表 list1
  List<int64_t> list1;

  // 使用 list1 初始化列表 list2
  List<int64_t> list2(list1);

  // 使用 list1 赋值给列表 list3
  List<int64_t> list3;
  list3 = list1;

  // 在列表 list1 后面添加元素 3
  list1.push_back(3);

  // 验证列表 list1 的大小是否为 1
  EXPECT_EQ(1, list1.size());

  // 验证列表 list2 的大小是否为 1
  EXPECT_EQ(1, list2.size());

  // 验证列表 list3 的大小是否为 1
  EXPECT_EQ(1, list3.size());
}

TEST(ListTestNonIValueBasedList, copyHasSeparateStorage) {
  // 创建空的整数列表 list1
  List<int64_t> list1;

  // 使用 list1 的拷贝初始化列表 list2
  List<int64_t> list2(list1.copy());

  // 使用 list1 赋值给列表 list3
  List<int64_t> list3;
  list3 = list1.copy();

  // 在列表 list1 后面添加元素 3
  list1.push_back(3);

  // 验证列表 list1 的大小是否为 1
  EXPECT_EQ(1, list1.size());

  // 验证列表 list2 的大小是否为 0（拷贝时未复制元素）
  EXPECT_EQ(0, list2.size());

  // 验证列表 list3 的大小是否为 0（拷贝时未复制元素）
  EXPECT_EQ(0, list3.size());
}

TEST(ListTestNonIValueBasedList, givenEqualLists_thenIsEqual) {
  // 创建包含元素 1 和 3 的整数列表 list1
  List<int64_t> list1({1, 3});

  // 创建包含元素 1 和 3 的整数列表 list2
  List<int64_t> list2({1, 3});

  // 验证列表 list1 是否等于列表 list2
  EXPECT_EQ(list1, list2);
}

TEST(ListTestNonIValueBasedList, givenDifferentLists_thenIsNotEqual) {
  // 创建包含元素 1 和 3 的整数列表 list1
  List<int64_t> list1({1, 3});

  // 创建包含元素 1 和 2 的整数列表 list2
  List<int64_t> list2({1, 2});

  // 验证列表 list1 是否不等于列表 list2
  EXPECT_NE(list1, list2);
}

TEST(ListTestNonIValueBasedList, isChecksIdentity) {
  // 创建包含元素 1 和 3 的整数列表 list1
  List<int64_t> list1({1, 3});

  // 将 list1 赋值给列表 list2（常量引用）
  const auto list2 = list1;

  // 验证列表 list1 是否和列表 list2 相等
  EXPECT_TRUE(list1.is(list2));
}

TEST(ListTestNonIValueBasedList, sameValueDifferentStorage_thenIsReturnsFalse) {
  // 创建包含元素 1 和 3 的整数列表 list1
  List<int64_t> list1({1, 3});

  // 创建 list1 的拷贝并赋值给列表 list2
  const auto list2 = list1.copy();

  // 验证列表 list1 是否和列表 list2 不相等
  EXPECT_FALSE(list1.is(list2));
}

TEST(ListTest, canAccessStringByReference) {
  // 创建包含字符串 "one" 和 "two" 的字符串列表
  List<std::string> list({"one", "two"});

  // 创建列表的常量引用 listRef
  const auto& listRef = list;

  // 静态断言，验证 listRef[1] 的类型为 const std::string&
  static_assert(std::is_same_v<decltype(listRef[1]), const std::string&>,
                "const List<std::string> access should be by const reference");

  // 获取列表第二个位置的字符串并赋值给 str
  std::string str = list[1];

  // 获取列表第二个位置的常量引用并赋值给 strRef
  const std::string& strRef = listRef[1];

  // 验证 str 的值是否为 "two"
  EXPECT_EQ("two", str);

  // 验证 strRef 的值是否为 "two"
  EXPECT_EQ("two", strRef);
}
TEST(ListTest, canAccessOptionalStringByReference) {
  // 创建包含三个元素的列表，分别为字符串 "one"、"two" 和 c10::nullopt
  List<std::optional<std::string>> list({"one", "two", c10::nullopt});
  // 声明一个指向常量的列表引用
  const auto& listRef = list;
  // 静态断言，验证 listRef[1] 的类型是否为 std::optional<std::reference_wrapper<const std::string>>
  static_assert(
      std::is_same_v<decltype(listRef[1]), std::optional<std::reference_wrapper<const std::string>>>,
      "List<std::optional<std::string>> access should be by const reference");
  // 从列表中取出第二个元素，赋给可选字符串 str1 和 str2
  std::optional<std::string> str1 = list[1];
  std::optional<std::string> str2 = list[2];
  // 从 listRef 中获取第二个元素的引用，decltype(auto) 用于推断类型
  decltype(auto) strRef1 = listRef[1];
  decltype(auto) strRef2 = listRef[2];
  // 使用断言验证 str1 的值为 "two"
  EXPECT_EQ("two", str1.value());
  // 使用断言验证 str2 不存在值
  EXPECT_FALSE(str2.has_value());
  // 使用断言验证 strRef1 的值为 "two"
  EXPECT_EQ("two", strRef1.value().get());
  // 使用断言验证 strRef2 不存在值
  EXPECT_FALSE(strRef2.has_value());
}

TEST(ListTest, canAccessTensorByReference) {
  // 创建一个空的 Tensor 列表
  List<at::Tensor> list;
  // 声明一个指向常量的列表引用
  const auto& listRef = list;
  // 静态断言，验证 listRef[0] 的类型是否为 const at::Tensor&
  static_assert(
      std::is_same_v<decltype(listRef[0]), const at::Tensor&>,
      "List<at::Tensor> access should be by const reference");
}

TEST(ListTest, toTypedList) {
  // 创建一个字符串列表 stringList 包含 "one" 和 "two"
  List<std::string> stringList({"one", "two"});
  // 将 stringList 转换为通用列表 genericList
  auto genericList = impl::toList(std::move(stringList));
  // 验证 genericList 的大小为 2
  EXPECT_EQ(genericList.size(), 2);
  // 使用 toTypedList 将 genericList 转换回字符串列表 stringList
  stringList = c10::impl::toTypedList<std::string>(std::move(genericList));
  // 验证转换后的 stringList 的大小为 2
  EXPECT_EQ(stringList.size(), 2);

  // 重新将 stringList 转换为通用列表 genericList
  genericList = impl::toList(std::move(stringList));
  // 使用 EXPECT_THROW 断言确保将 genericList 转换为 int64_t 类型的列表时抛出异常
  EXPECT_THROW(c10::impl::toTypedList<int64_t>(std::move(genericList)), c10::Error);
}
// NOLINTEND(performance-move-const-arg, bugprone-use-after-move)
```