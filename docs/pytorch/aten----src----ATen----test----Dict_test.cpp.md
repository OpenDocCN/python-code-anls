# `.\pytorch\aten\src\ATen\test\Dict_test.cpp`

```py
#include <ATen/core/Dict.h> // 引入 ATen 库中的字典实现头文件
#include <ATen/ATen.h> // 引入 ATen 库
#include <gtest/gtest.h> // 引入 Google 测试框架头文件
#include <gmock/gmock.h> // 引入 Google Mock 框架头文件
#include <string> // 引入标准库中的字符串定义

using std::string; // 使用标准库中的 string 类
using c10::Dict; // 使用 ATen 中的字典类

#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1.equal(t2)); // 定义一个断言宏，用于比较两个对象是否相等

// 定义测试用例 DictTest，测试字典类的各种功能
TEST(DictTest, givenEmptyDict_whenCallingEmpty_thenReturnsTrue) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    EXPECT_TRUE(dict.empty()); // 断言字典为空
}

TEST(DictTest, givenNonemptyDict_whenCallingEmpty_thenReturnsFalse) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    dict.insert(3, "value"); // 向字典中插入一对键值对
    EXPECT_FALSE(dict.empty()); // 断言字典不为空
}

TEST(DictTest, givenEmptyDict_whenCallingSize_thenReturnsZero) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    EXPECT_EQ(0, dict.size()); // 断言字典大小为 0
}

TEST(DictTest, givenNonemptyDict_whenCallingSize_thenReturnsNumberOfElements) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    dict.insert(3, "value"); // 向字典中插入一对键值对
    dict.insert(4, "value2"); // 再向字典中插入另一对键值对
    EXPECT_EQ(2, dict.size()); // 断言字典大小为 2
}

TEST(DictTest, givenNonemptyDict_whenCallingClear_thenIsEmpty) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    dict.insert(3, "value"); // 向字典中插入一对键值对
    dict.insert(4, "value2"); // 再向字典中插入另一对键值对
    dict.clear(); // 清空字典
    EXPECT_TRUE(dict.empty()); // 断言字典为空
}

TEST(DictTest, whenInsertingNewKey_thenReturnsTrueAndIteratorToNewElement) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    std::pair<Dict<int64_t, string>::iterator, bool> result = dict.insert(3, "value"); // 插入一对键值对，并接收返回值
    EXPECT_TRUE(result.second); // 断言插入成功
    EXPECT_EQ(3, result.first->key()); // 断言插入后的键为 3
    EXPECT_EQ("value", result.first->value()); // 断言插入后的值为 "value"
}

TEST(DictTest, whenInsertingExistingKey_thenReturnsFalseAndIteratorToExistingElement) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    dict.insert(3, "old_value"); // 向字典中插入一对键值对
    std::pair<Dict<int64_t, string>::iterator, bool> result = dict.insert(3, "new_value"); // 插入一个已存在的键，并接收返回值
    EXPECT_FALSE(result.second); // 断言插入失败（因为键已存在）
    EXPECT_EQ(3, result.first->key()); // 断言返回的迭代器的键为 3
    EXPECT_EQ("old_value", result.first->value()); // 断言返回的迭代器的值为 "old_value"
}

TEST(DictTest, whenInsertingExistingKey_thenDoesNotModifyDict) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    dict.insert(3, "old_value"); // 向字典中插入一对键值对
    dict.insert(3, "new_value"); // 再次插入相同的键，但值不同
    EXPECT_EQ(1, dict.size()); // 断言字典大小为 1
    EXPECT_EQ(3, dict.begin()->key()); // 断言字典中第一个元素的键为 3
    EXPECT_EQ("old_value", dict.begin()->value()); // 断言字典中第一个元素的值为 "old_value"
}

TEST(DictTest, whenInsertOrAssigningNewKey_thenReturnsTrueAndIteratorToNewElement) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    std::pair<Dict<int64_t, string>::iterator, bool> result = dict.insert_or_assign(3, "value"); // 插入或更新一对键值对，并接收返回值
    EXPECT_TRUE(result.second); // 断言插入成功
    EXPECT_EQ(3, result.first->key()); // 断言插入后的键为 3
    EXPECT_EQ("value", result.first->value()); // 断言插入后的值为 "value"
}

TEST(DictTest, whenInsertOrAssigningExistingKey_thenReturnsFalseAndIteratorToChangedElement) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    dict.insert(3, "old_value"); // 向字典中插入一对键值对
    std::pair<Dict<int64_t, string>::iterator, bool> result = dict.insert_or_assign(3, "new_value"); // 插入或更新一个已存在的键，并接收返回值
    EXPECT_FALSE(result.second); // 断言更新成功（因为键已存在）
    EXPECT_EQ(3, result.first->key()); // 断言更新后的迭代器的键为 3
    EXPECT_EQ("new_value", result.first->value()); // 断言更新后的迭代器的值为 "new_value"
}

TEST(DictTest, whenInsertOrAssigningExistingKey_thenDoesModifyDict) {
    Dict<int64_t, string> dict; // 创建一个空的键为 int64_t，值为 string 类型的字典
    dict.insert(3, "old_value"); // 向字典中插入一对键值对
    dict.insert_or_assign(3, "new_value"); // 更新已存在的键的值
    EXPECT_EQ(1, dict.size()); // 断言字典大小为 1
    EXPECT_EQ(3, dict.begin()->key()); // 断言字典中第一个元素的键为 3
    EXPECT_EQ("new_value", dict.begin()->value()); // 断言字典中第一个元素的值为 "new_value"
}
TEST(DictTest, givenEmptyDict_whenIterating_thenBeginIsEnd) {
  // 创建一个名为 dict 的空字典对象，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict;
  // 断言字典的起始迭代器等于结束迭代器，即字典为空时迭代起始和结束位置相同
  EXPECT_EQ(dict.begin(), dict.end());
}

TEST(DictTest, givenMutableDict_whenIterating_thenFindsElements) {
  // 创建一个名为 dict 的字典对象，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict;
  // 向字典中插入键值对 (3, "3") 和 (5, "5")
  dict.insert(3, "3");
  dict.insert(5, "5");
  // 声明并初始化两个布尔变量，用于跟踪是否找到了特定的键
  bool found_first = false;
  bool found_second = false;
  // 使用迭代器遍历字典中的元素
  for (Dict<int64_t, string>::iterator iter = dict.begin(); iter != dict.end(); ++iter) {
    // 如果当前迭代器指向的键为 3，则验证其对应的值为 "3"
    if (iter->key() == 3) {
      EXPECT_EQ("3", iter->value());
      // 断言 found_first 变量为 false，表示第一次找到该键
      EXPECT_FALSE(found_first);
      found_first = true;  // 将 found_first 标记为已找到
    } 
    // 如果当前迭代器指向的键为 5，则验证其对应的值为 "5"
    else if (iter->key() == 5) {
      EXPECT_EQ("5", iter->value());
      // 断言 found_second 变量为 false，表示第一次找到该键
      EXPECT_FALSE(found_second);
      found_second = true;  // 将 found_second 标记为已找到
    } else {
      // 如果迭代器指向的键既不是 3 也不是 5，则添加一个失败记录
      ADD_FAILURE();
    }
  }
  // 断言 found_first 和 found_second 都为 true，表示已找到所有期望的键
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(DictTest, givenMutableDict_whenIteratingWithForeach_thenFindsElements) {
  // 创建一个名为 dict 的字典对象，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict;
  // 向字典中插入键值对 (3, "3") 和 (5, "5")
  dict.insert(3, "3");
  dict.insert(5, "5");
  // 声明并初始化两个布尔变量，用于跟踪是否找到了特定的键
  bool found_first = false;
  bool found_second = false;
  // 使用范围-based for 循环遍历字典中的每个元素
  for (const auto& elem : dict) {
    // 如果当前元素的键为 3，则验证其对应的值为 "3"
    if (elem.key() == 3) {
      EXPECT_EQ("3", elem.value());
      // 断言 found_first 变量为 false，表示第一次找到该键
      EXPECT_FALSE(found_first);
      found_first = true;  // 将 found_first 标记为已找到
    } 
    // 如果当前元素的键为 5，则验证其对应的值为 "5"
    else if (elem.key() == 5) {
      EXPECT_EQ("5", elem.value());
      // 断言 found_second 变量为 false，表示第一次找到该键
      EXPECT_FALSE(found_second);
      found_second = true;  // 将 found_second 标记为已找到
    } else {
      // 如果元素的键既不是 3 也不是 5，则添加一个失败记录
      ADD_FAILURE();
    }
  }
  // 断言 found_first 和 found_second 都为 true，表示已找到所有期望的键
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(DictTest, givenConstDict_whenIterating_thenFindsElements) {
  // 创建一个名为 dict_ 的字典对象，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict_;
  // 向 dict_ 中插入键值对 (3, "3") 和 (5, "5")
  dict_.insert(3, "3");
  dict_.insert(5, "5");
  // 创建一个常量引用 dict，指向 dict_
  const Dict<int64_t, string>& dict = dict_;
  // 声明并初始化两个布尔变量，用于跟踪是否找到了特定的键
  bool found_first = false;
  bool found_second = false;
  // 使用迭代器遍历常量字典 dict 中的元素
  for (Dict<int64_t, string>::iterator iter = dict.begin(); iter != dict.end(); ++iter) {
    // 如果当前迭代器指向的键为 3，则验证其对应的值为 "3"
    if (iter->key() == 3) {
      EXPECT_EQ("3", iter->value());
      // 断言 found_first 变量为 false，表示第一次找到该键
      EXPECT_FALSE(found_first);
      found_first = true;  // 将 found_first 标记为已找到
    } 
    // 如果当前迭代器指向的键为 5，则验证其对应的值为 "5"
    else if (iter->key() == 5) {
      EXPECT_EQ("5", iter->value());
      // 断言 found_second 变量为 false，表示第一次找到该键
      EXPECT_FALSE(found_second);
      found_second = true;  // 将 found_second 标记为已找到
    } else {
      // 如果迭代器指向的键既不是 3 也不是 5，则添加一个失败记录
      ADD_FAILURE();
    }
  }
  // 断言 found_first 和 found_second 都为 true，表示已找到所有期望的键
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(DictTest, givenConstDict_whenIteratingWithForeach_thenFindsElements) {
  // 创建一个名为 dict_ 的字典对象，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict_;
  // 向 dict_ 中插入键值对 (3, "3") 和 (5, "5")
  dict_.insert(3, "3");
  dict_.insert(5, "5");
  // 创建一个常量引用 dict，指向 dict_
  const Dict<int64_t, string>& dict = dict_;
  // 声明并初始化两个布尔变量，用于跟踪是否找到了特定的键
  bool found_first = false;
  bool found_second = false;
  // 使用范围-based for 循环遍历常量字典 dict 中的每个元素
  for (const auto& elem : dict) {
    // 如果当前元素的键为 3，则验证其对应的值为 "3"
    if (elem.key() == 3) {
      EXPECT_EQ("3", elem.value());
      // 断言 found_first 变量为 false，表示第一次找到该键
      EXPECT_FALSE(found_first);
      found_first = true;  // 将 found_first 标记为已找到
    } 
    // 如果当前元素的键为 5，则验证其对应的值为 "5"
    else if (elem.key() == 5) {
      EXPECT_EQ("5", elem.value());
      // 断言 found_second 变量为 false，表示第一次找到该键
      EXPECT_FALSE(found_second);
      found_second = true;  // 将 found_second 标记为已找到
    } else {
      // 如果元素的键既不是 3 也不是 5，则添加一个失败记录
      ADD_FAILURE();
    }
  }
  // 断言 found_first 和 found_second 都为 true，表示已找到所有期望的键
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(DictTest, givenIterator_thenCanModifyValue) {
  // 创建一个名为 dict 的字典对象，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict;
  // 向字典中插入键值对 (3, "old_value")
  dict.insert(3
TEST(DictTest, givenOneElementDict_whenErasingByIterator_thenDictIsEmpty) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 向字典插入一个键为 3，值为 "3" 的元素
  dict.insert(3, "3");
  // 通过迭代器删除字典中的第一个元素
  dict.erase(dict.begin());
  // 验证字典是否为空
  EXPECT_TRUE(dict.empty());
}

TEST(DictTest, givenOneElementDict_whenErasingByKey_thenReturnsOneAndDictIsEmpty) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 向字典插入一个键为 3，值为 "3" 的元素
  dict.insert(3, "3");
  // 通过键值 3 删除字典中的元素，并将删除操作的结果存储在 result 变量中
  bool result = dict.erase(3);
  // 验证 result 是否为 1
  EXPECT_EQ(1, result);
  // 验证字典是否为空
  EXPECT_TRUE(dict.empty());
}

TEST(DictTest, givenOneElementDict_whenErasingByNonexistingKey_thenReturnsZeroAndDictIsUnchanged) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 向字典插入一个键为 3，值为 "3" 的元素
  dict.insert(3, "3");
  // 通过键值 4 删除字典中的元素，并将删除操作的结果存储在 result 变量中
  bool result = dict.erase(4);
  // 验证 result 是否为 0
  EXPECT_EQ(0, result);
  // 验证字典的大小是否为 1
  EXPECT_EQ(1, dict.size());
}

TEST(DictTest, whenCallingAtWithExistingKey_thenReturnsCorrectElement) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 向字典插入键值对 3: "3" 和 4: "4"
  dict.insert(3, "3");
  dict.insert(4, "4");
  // 验证通过键值 4 调用 at 方法返回的值是否为 "4"
  EXPECT_EQ("4", dict.at(4));
}

TEST(DictTest, whenCallingAtWithNonExistingKey_thenReturnsCorrectElement) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 向字典插入键值对 3: "3" 和 4: "4"
  dict.insert(3, "3");
  dict.insert(4, "4");
  // 预期调用 at 方法时会抛出 std::out_of_range 异常，因为键值 5 不存在
  EXPECT_THROW(dict.at(5), std::out_of_range);
}

TEST(DictTest, givenMutableDict_whenCallingFindOnExistingKey_thenFindsCorrectElement) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 向字典插入键值对 3: "3" 和 4: "4"
  dict.insert(3, "3");
  dict.insert(4, "4");
  // 在字典中查找键值 3，并将找到的迭代器存储在 found 变量中
  Dict<int64_t, string>::iterator found = dict.find(3);
  // 验证 found 指向的元素的键是否为 3，值是否为 "3"
  EXPECT_EQ(3, found->key());
  EXPECT_EQ("3", found->value());
}

TEST(DictTest, givenMutableDict_whenCallingFindOnNonExistingKey_thenReturnsEnd) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 向字典插入键值对 3: "3" 和 4: "4"
  dict.insert(3, "3");
  dict.insert(4, "4");
  // 在字典中查找键值 5，并将找到的迭代器存储在 found 变量中
  Dict<int64_t, string>::iterator found = dict.find(5);
  // 验证 found 是否等于 dict.end()，即未找到指定键时 find 方法返回的结果
  EXPECT_EQ(dict.end(), found);
}

TEST(DictTest, givenConstDict_whenCallingFindOnExistingKey_thenFindsCorrectElement) {
  // 创建一个名为 dict_ 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict_;
  // 向字典插入键值对 3: "3" 和 4: "4"
  dict_.insert(3, "3");
  dict_.insert(4, "4");
  // 创建一个常量引用 dict，指向 dict_，实现常量字典的引用
  const Dict<int64_t, string>& dict = dict_;
  // 在常量字典中查找键值 3，并将找到的迭代器存储在 found 变量中
  Dict<int64_t, string>::iterator found = dict.find(3);
  // 验证 found 指向的元素的键是否为 3，值是否为 "3"
  EXPECT_EQ(3, found->key());
  EXPECT_EQ("3", found->value());
}

TEST(DictTest, givenConstDict_whenCallingFindOnNonExistingKey_thenReturnsEnd) {
  // 创建一个名为 dict_ 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict_;
  // 向字典插入键值对 3: "3" 和 4: "4"
  dict_.insert(3, "3");
  dict_.insert(4, "4");
  // 创建一个常量引用 dict，指向 dict_，实现常量字典的引用
  const Dict<int64_t, string>& dict = dict_;
  // 在常量字典中查找键值 5，并将找到的迭代器存储在 found 变量中
  Dict<int64_t, string>::iterator found = dict.find(5);
  // 验证 found 是否等于 dict.end()，即未找到指定键时 find 方法返回的结果
  EXPECT_EQ(dict.end(), found);
}

TEST(DictTest, whenCallingContainsWithExistingKey_thenReturnsTrue) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 向字典插入键值对 3: "3" 和 4: "4"
  dict.insert(3, "3");
  dict.insert(4, "4");
  // 验证字典中是否包含键值 3
  EXPECT_TRUE(dict.contains(3));
}

TEST(DictTest, whenCallingContainsWithNonExistingKey_thenReturnsFalse) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 向字典插入键值对 3: "3" 和 4: "4"
  dict.insert(3, "3");
  dict.insert(4, "4");
  // 验证字典中是否包含键值 5
  EXPECT_FALSE(dict.contains(5));
}

TEST(DictTest, whenCallingReserve_thenDoesntCrash) {
  // 创建一个名为 dict 的字典，键为 int64_t 类型，值为 string 类型
  Dict<int64_t, string> dict;
  // 调用 reserve 方法，预留至少能容纳 100 个元素的空间
  dict.reserve(100);
}
TEST(DictTest, whenCopyConstructingDict_thenAreEqual) {
  // 创建一个空的字典 dict1，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict1;
  // 向 dict1 中插入键值对 (3, "3")
  dict1.insert(3, "3");
  // 向 dict1 中插入键值对 (4, "4")
  dict1.insert(4, "4");

  // 使用拷贝构造函数将 dict1 拷贝给 dict2
  Dict<int64_t, string> dict2(dict1);

  // 断言 dict2 的大小为 2
  EXPECT_EQ(2, dict2.size());
  // 断言 dict2 中键 3 对应的值为 "3"
  EXPECT_EQ("3", dict2.at(3));
  // 断言 dict2 中键 4 对应的值为 "4"
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenCopyAssigningDict_thenAreEqual) {
  // 创建一个空的字典 dict1，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict1;
  // 向 dict1 中插入键值对 (3, "3")
  dict1.insert(3, "3");
  // 向 dict1 中插入键值对 (4, "4")
  dict1.insert(4, "4");

  // 使用拷贝赋值运算符将 dict1 赋值给 dict2
  Dict<int64_t, string> dict2;
  dict2 = dict1;

  // 断言 dict2 的大小为 2
  EXPECT_EQ(2, dict2.size());
  // 断言 dict2 中键 3 对应的值为 "3"
  EXPECT_EQ("3", dict2.at(3));
  // 断言 dict2 中键 4 对应的值为 "4"
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenCopyingDict_thenAreEqual) {
  // 创建一个空的字典 dict1，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict1;
  // 向 dict1 中插入键值对 (3, "3")
  dict1.insert(3, "3");
  // 向 dict1 中插入键值对 (4, "4")
  dict1.insert(4, "4");

  // 使用复制方法 copy() 创建 dict1 的副本 dict2
  Dict<int64_t, string> dict2 = dict1.copy();

  // 断言 dict2 的大小为 2
  EXPECT_EQ(2, dict2.size());
  // 断言 dict2 中键 3 对应的值为 "3"
  EXPECT_EQ("3", dict2.at(3));
  // 断言 dict2 中键 4 对应的值为 "4"
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenMoveConstructingDict_thenNewIsCorrect) {
  // 创建一个空的字典 dict1，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict1;
  // 向 dict1 中插入键值对 (3, "3")
  dict1.insert(3, "3");
  // 向 dict1 中插入键值对 (4, "4")
  dict1.insert(4, "4");

  // 使用移动构造函数将 dict1 移动到 dict2
  Dict<int64_t, string> dict2(std::move(dict1));

  // 断言 dict2 的大小为 2
  EXPECT_EQ(2, dict2.size());
  // 断言 dict2 中键 3 对应的值为 "3"
  EXPECT_EQ("3", dict2.at(3));
  // 断言 dict2 中键 4 对应的值为 "4"
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenMoveAssigningDict_thenNewIsCorrect) {
  // 创建一个空的字典 dict1，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict1;
  // 向 dict1 中插入键值对 (3, "3")
  dict1.insert(3, "3");
  // 向 dict1 中插入键值对 (4, "4")
  dict1.insert(4, "4");

  // 使用移动赋值运算符将 dict1 移动给 dict2
  Dict<int64_t, string> dict2;
  dict2 = std::move(dict1);

  // 断言 dict2 的大小为 2
  EXPECT_EQ(2, dict2.size());
  // 断言 dict2 中键 3 对应的值为 "3"
  EXPECT_EQ("3", dict2.at(3));
  // 断言 dict2 中键 4 对应的值为 "4"
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenMoveConstructingDict_thenOldIsUnchanged) {
  // 创建一个空的字典 dict1，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict1;
  // 向 dict1 中插入键值对 (3, "3")
  dict1.insert(3, "3");
  // 向 dict1 中插入键值对 (4, "4")
  dict1.insert(4, "4");

  // 使用移动构造函数将 dict1 移动到 dict2
  Dict<int64_t, string> dict2(std::move(dict1));

  // 断言 dict1 的大小为 2，即原始 dict1 未被改变
  EXPECT_EQ(2, dict1.size());
  // 断言 dict1 中键 3 对应的值为 "3"
  EXPECT_EQ("3", dict1.at(3));
  // 断言 dict1 中键 4 对应的值为 "4"
  EXPECT_EQ("4", dict1.at(4));
}

TEST(DictTest, whenMoveAssigningDict_thenOldIsUnchanged) {
  // 创建一个空的字典 dict1，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict1;
  // 向 dict1 中插入键值对 (3, "3")
  dict1.insert(3, "3");
  // 向 dict1 中插入键值对 (4, "4")
  dict1.insert(4, "4");

  // 使用移动赋值运算符将 dict1 移动给 dict2
  Dict<int64_t, string> dict2;
  dict2 = std::move(dict1);

  // 断言 dict1 的大小为 2，即原始 dict1 未被改变
  EXPECT_EQ(2, dict1.size());
  // 断言 dict1 中键 3 对应的值为 "3"
  EXPECT_EQ("3", dict1.at(3));
  // 断言 dict1 中键 4 对应的值为 "4"
  EXPECT_EQ("4", dict1.at(4));
}

TEST(DictTest, givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition) {
  // 创建一个空的字典 dict，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict;
  // 向 dict 中插入键值对 (3, "3")
  dict.insert(3, "3");
  // 向 dict 中插入键值对 (4, "4")
  dict.insert(4, "4");

  // 获取 dict 的起始迭代器 iter1
  Dict<int64_t, string>::iterator iter1 = dict.begin();
  // 后缀递增迭代器 iter1，将旧位置的迭代器保存到 iter2
  Dict<int64_t, string>::iterator iter2 = iter1++;
  // 断言 iter1 的键不等于起始位置的键
  EXPECT_NE(dict.begin()->key(), iter1->key());
  // 断言 iter2 的键等于起始位置的键
  EXPECT_EQ(dict.begin()->key(), iter2->key());
}

TEST(DictTest, givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition) {
  // 创建一个空的字典 dict，键类型为 int64_t，值类型为 string
  Dict<int64_t, string> dict;
  // 向 dict 中插入键值对 (3, "3")
  dict.insert(3, "3");
  // 向 dict 中插入键值对 (4, "4")
  dict.insert(4, "4");

  // 获取 dict 的起始迭代器 iter1
  Dict<int64_t, string>::iterator iter1 = dict.begin();
  // 前缀递增迭代器
TEST(DictTest, givenEqualIterators_thenAreEqual) {
  // 创建一个 Dict 对象，键是 int64_t 类型，值是 string 类型
  Dict<int64_t, string> dict;
  // 向字典中插入键值对 (3, "3") 和 (4, "4")
  dict.insert(3, "3");
  dict.insert(4, "4");

  // 获取字典的起始迭代器并赋给 iter1 和 iter2
  Dict<int64_t, string>::iterator iter1 = dict.begin();
  Dict<int64_t, string>::iterator iter2 = dict.begin();
  
  // 断言 iter1 和 iter2 相等
  EXPECT_TRUE(iter1 == iter2);
  EXPECT_FALSE(iter1 != iter2);
}

TEST(DictTest, givenDifferentIterators_thenAreNotEqual) {
  // 创建一个 Dict 对象，键是 int64_t 类型，值是 string 类型
  Dict<int64_t, string> dict;
  // 向字典中插入键值对 (3, "3") 和 (4, "4")
  dict.insert(3, "3");
  dict.insert(4, "4");

  // 获取字典的起始迭代器并赋给 iter1 和 iter2
  Dict<int64_t, string>::iterator iter1 = dict.begin();
  Dict<int64_t, string>::iterator iter2 = dict.begin();
  // iter2 向后移动一个位置
  iter2++;

  // 断言 iter1 和 iter2 不相等
  EXPECT_FALSE(iter1 == iter2);
  EXPECT_TRUE(iter1 != iter2);
}

TEST(DictTest, givenIterator_whenDereferencing_thenPointsToCorrectElement) {
  // 创建一个 Dict 对象，键是 int64_t 类型，值是 string 类型
  Dict<int64_t, string> dict;
  // 向字典中插入键值对 (3, "3")
  dict.insert(3, "3");

  // 获取字典的起始迭代器并赋给 iter
  Dict<int64_t, string>::iterator iter = dict.begin();
  
  // 使用迭代器解引用，检查其指向的元素的 key 和 value
  EXPECT_EQ(3, (*iter).key());
  EXPECT_EQ("3", (*iter).value());
  EXPECT_EQ(3, iter->key());
  EXPECT_EQ("3", iter->value());
}

TEST(DictTest, givenIterator_whenWritingToValue_thenChangesValue) {
  // 创建一个 Dict 对象，键是 int64_t 类型，值是 string 类型
  Dict<int64_t, string> dict;
  // 向字典中插入键值对 (3, "3")
  dict.insert(3, "3");

  // 获取字典的起始迭代器并赋给 iter
  Dict<int64_t, string>::iterator iter = dict.begin();

  // 通过迭代器修改其指向的元素的值
  (*iter).setValue("new_value");
  EXPECT_EQ("new_value", dict.begin()->value());

  iter->setValue("new_value_2");
  EXPECT_EQ("new_value_2", dict.begin()->value());
}

TEST(ListTestIValueBasedList, givenIterator_whenWritingToValueFromIterator_thenChangesValue) {
  // 创建一个 Dict 对象，键是 int64_t 类型，值是 string 类型
  Dict<int64_t, string> dict;
  // 向字典中插入键值对 (3, "3"), (4, "4"), (5, "5")
  dict.insert(3, "3");
  dict.insert(4, "4");
  dict.insert(5, "5");

  // 使用 find 方法查找键为 3 和 4 的迭代器，然后交换其值
  (*dict.find(3)).setValue(dict.find(4)->value());
  EXPECT_EQ("4", dict.find(3)->value());

  // 修改键为 3 的迭代器指向元素的值为键为 5 的元素的值
  dict.find(3)->setValue(dict.find(5)->value());
  EXPECT_EQ("5", dict.find(3)->value());
}

TEST(DictTest, isReferenceType) {
  // 创建一个 Dict 对象，键是 int64_t 类型，值是 string 类型
  Dict<int64_t, string> dict1;
  // 通过拷贝构造函数创建 dict2，并通过赋值运算符创建 dict3
  Dict<int64_t, string> dict2(dict1);
  Dict<int64_t, string> dict3;
  dict3 = dict1;

  // 向 dict1 中插入一个元素后，检查 dict1、dict2 和 dict3 的大小是否都为 1
  dict1.insert(3, "three");
  EXPECT_EQ(1, dict1.size());
  EXPECT_EQ(1, dict2.size());
  EXPECT_EQ(1, dict3.size());
}

TEST(DictTest, copyHasSeparateStorage) {
  // 创建一个 Dict 对象，键是 int64_t 类型，值是 string 类型
  Dict<int64_t, string> dict1;
  // 使用 copy 方法拷贝 dict1 并赋给 dict2，并通过赋值运算符创建 dict3
  Dict<int64_t, string> dict2(dict1.copy());
  Dict<int64_t, string> dict3;
  dict3 = dict1.copy();

  // 向 dict1 中插入一个元素后，检查 dict1 中有 1 个元素，而 dict2 和 dict3 中没有元素
  dict1.insert(3, "three");
  EXPECT_EQ(1, dict1.size());
  EXPECT_EQ(0, dict2.size());
  EXPECT_EQ(0, dict3.size());
}

TEST(DictTest, dictTensorAsKey) {
  // 创建一个 Dict 对象，键是 at::Tensor 类型，值是 string 类型
  Dict<at::Tensor, string> dict;
  // 创建两个 at::Tensor 对象作为键，并插入键值对
  at::Tensor key1 = at::tensor(3);
  at::Tensor key2 = at::tensor(4);
  dict.insert(key1, "three");
  dict.insert(key2, "four");

  // 检查字典中元素的数量，查找键为 key1 和不存在的键（at::tensor(3) 和 at::tensor(5)）
  EXPECT_EQ(2, dict.size());

  Dict<at::Tensor, string>::iterator found_key1 = dict.find(key1);
  // 断言找到的 key1 的迭代器指向的键是 key1，值是 "three"
  ASSERT_EQUAL(key1, found_key1->key());
  EXPECT_EQ("three", found_key1->value());

  Dict<at::Tensor, string>::iterator found_nokey1 = dict.find(at::tensor(3));
  Dict<at::Tensor, string>::iterator found_nokey2 = dict.find(at::tensor(5));
  // 断言找不到 at::tensor(3) 和 at::tensor(5) 对应的迭代器
  EXPECT_EQ(dict.end(), found_nokey1);
  EXPECT_EQ(dict.end(), found_nokey2);
}
TEST(DictTest, dictEquality) {
  // 创建一个名为 dict 的字典对象，键为 string 类型，值为 int64_t 类型
  Dict<string, int64_t> dict;
  // 向字典中插入键值对 "one": 1
  dict.insert("one", 1);
  // 向字典中插入键值对 "two": 2
  dict.insert("two", 2);

  // 创建一个名为 dictSameValue 的字典对象，与 dict 相同的键值对内容
  Dict<string, int64_t> dictSameValue;
  dictSameValue.insert("one", 1);
  dictSameValue.insert("two", 2);

  // 创建一个名为 dictNotEqual 的字典对象，包含不同的键值对
  Dict<string, int64_t> dictNotEqual;
  dictNotEqual.insert("foo", 1);
  dictNotEqual.insert("bar", 2);

  // 创建一个名为 dictRef 的字典对象，初始化为 dict 的拷贝
  Dict<string, int64_t> dictRef = dict;

  // 断言 dict 与 dictSameValue 相等
  EXPECT_EQ(dict, dictSameValue);
  // 断言 dict 与 dictNotEqual 不相等
  EXPECT_NE(dict, dictNotEqual);
  // 断言 dictSameValue 与 dictNotEqual 不相等
  EXPECT_NE(dictSameValue, dictNotEqual);
  // 断言 dict 不是 dictSameValue
  EXPECT_FALSE(dict.is(dictSameValue));
  // 断言 dict 是 dictRef
  EXPECT_TRUE(dict.is(dictRef));
}
```