# `.\pytorch\test\cpp\api\ordered_dict.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 Torch 库的头文件
#include <torch/torch.h>

// 定义模板类型 OrderedDict，是 torch::OrderedDict<std::string, T> 的别名
template <typename T>
using OrderedDict = torch::OrderedDict<std::string, T>;

// 测试用例：默认构造后 OrderedDict 应为空
TEST(OrderedDictTest, IsEmptyAfterDefaultConstruction) {
  // 创建一个空的 OrderedDict 对象
  OrderedDict<int> dict;
  // 断言默认的键描述为 "Key"
  ASSERT_EQ(dict.key_description(), "Key");
  // 断言 OrderedDict 对象为空
  ASSERT_TRUE(dict.is_empty());
  // 断言 OrderedDict 对象的大小为 0
  ASSERT_EQ(dict.size(), 0);
}

// 测试用例：插入元素并确认元素数目增加
TEST(OrderedDictTest, InsertAddsElementsWhenTheyAreYetNotPresent) {
  // 创建一个空的 OrderedDict 对象
  OrderedDict<int> dict;
  // 向 OrderedDict 中插入键值对 "a": 1
  dict.insert("a", 1);
  // 向 OrderedDict 中插入键值对 "b": 2
  dict.insert("b", 2);
  // 断言 OrderedDict 对象的大小为 2
  ASSERT_EQ(dict.size(), 2);
}

// 测试用例：获取已存在键的值并进行断言
TEST(OrderedDictTest, GetReturnsValuesWhenTheyArePresent) {
  // 创建一个空的 OrderedDict 对象
  OrderedDict<int> dict;
  // 向 OrderedDict 中插入键值对 "a": 1
  dict.insert("a", 1);
  // 向 OrderedDict 中插入键值对 "b": 2
  dict.insert("b", 2);
  // 断言获取键 "a" 对应的值为 1
  ASSERT_EQ(dict["a"], 1);
  // 断言获取键 "b" 对应的值为 2
  ASSERT_EQ(dict["b"], 2);
}

// 测试用例：获取不存在的键时应抛出异常
TEST(OrderedDictTest, GetThrowsWhenPassedKeysThatAreNotPresent) {
  // 创建一个空的 OrderedDict 对象
  OrderedDict<int> dict;
  // 向 OrderedDict 中插入键值对 "a": 1
  dict.insert("a", 1);
  // 向 OrderedDict 中插入键值对 "b": 2
  dict.insert("b", 2);
  // 断言获取键 "foo" 应抛出异常，异常信息为 "Key 'foo' is not defined"
  ASSERT_THROWS_WITH(dict["foo"], "Key 'foo' is not defined");
  // 断言获取空键 "" 应抛出异常，异常信息为 "Key '' is not defined"
  ASSERT_THROWS_WITH(dict[""], "Key '' is not defined");
}

// 测试用例：从列表初始化 OrderedDict 对象
TEST(OrderedDictTest, CanInitializeFromList) {
  // 使用初始化列表初始化 OrderedDict 对象，键值对为 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言 OrderedDict 对象的大小为 2
  ASSERT_EQ(dict.size(), 2);
  // 断言获取键 "a" 对应的值为 1
  ASSERT_EQ(dict["a"], 1);
  // 断言获取键 "b" 对应的值为 2
  ASSERT_EQ(dict["b"], 2);
}

// 测试用例：插入已存在的键时应抛出异常
TEST(OrderedDictTest, InsertThrowsWhenPassedElementsThatArePresent) {
  // 使用初始化列表初始化 OrderedDict 对象，键值对为 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言向 OrderedDict 中插入已存在的键 "a" 应抛出异常，异常信息为 "Key 'a' already defined"
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Key 'a' already defined");
  // 断言向 OrderedDict 中插入已存在的键 "b" 应抛出异常，异常信息为 "Key 'b' already defined"
  ASSERT_THROWS_WITH(dict.insert("b", 1), "Key 'b' already defined");
}

// 测试用例：获取 OrderedDict 中第一个元素并进行断言
TEST(OrderedDictTest, FrontReturnsTheFirstItem) {
  // 使用初始化列表初始化 OrderedDict 对象，键值对为 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言获取 OrderedDict 中第一个元素的键为 "a"
  ASSERT_EQ(dict.front().key(), "a");
  // 断言获取 OrderedDict 中第一个元素的值为 1
  ASSERT_EQ(dict.front().value(), 1);
}

// 测试用例：获取空 OrderedDict 的第一个元素时应抛出异常
TEST(OrderedDictTest, FrontThrowsWhenEmpty) {
  // 创建一个空的 OrderedDict 对象
  OrderedDict<int> dict;
  // 断言调用空 OrderedDict 的 front() 函数应抛出异常，异常信息为 "Called front() on an empty OrderedDict"
  ASSERT_THROWS_WITH(dict.front(), "Called front() on an empty OrderedDict");
}

// 测试用例：获取 OrderedDict 中最后一个元素并进行断言
TEST(OrderedDictTest, BackReturnsTheLastItem) {
  // 使用初始化列表初始化 OrderedDict 对象，键值对为 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言获取 OrderedDict 中最后一个元素的键为 "b"
  ASSERT_EQ(dict.back().key(), "b");
  // 断言获取 OrderedDict 中最后一个元素的值为 2
  ASSERT_EQ(dict.back().value(), 2);
}

// 测试用例：获取空 OrderedDict 的最后一个元素时应抛出异常
TEST(OrderedDictTest, BackThrowsWhenEmpty) {
  // 创建一个空的 OrderedDict 对象
  OrderedDict<int> dict;
  // 断言调用空 OrderedDict 的 back() 函数应抛出异常，异常信息为 "Called back() on an empty OrderedDict"
  ASSERT_THROWS_WITH(dict.back(), "Called back() on an empty OrderedDict");
}

// 测试用例：查找存在的键时返回值指针并进行断言
TEST(OrderedDictTest, FindReturnsPointersToValuesWhenPresent) {
  // 使用初始化列表初始化 OrderedDict 对象，键值对为 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言查找键 "a" 应返回非空指针，并且其值为 1
  ASSERT_NE(dict.find("a"), nullptr);
  ASSERT_EQ(*dict.find("a"), 1);
  // 断言查找键 "b" 应返回非空指针，并且其值为 2
  ASSERT_NE(dict.find("b"), nullptr);
  ASSERT_EQ(*dict.find("b"), 2);
}

// 测试用例：查找不存在的键时返回空指针
TEST(OrderedDictTest, FindReturnsNullPointersWhenPasesdKeysThatAreNotPresent) {
  // 使用初始化列表初始化 OrderedDict 对象，键值对为 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言查找键 "bar" 应返回空指针
  ASSERT_EQ(dict.find("bar"), nullptr);
  // 断言查找空键 "" 应返回空指针
  ASSERT_EQ(dict.find(""), nullptr);
}

// 测试用例：通过下标操作符获取已存在的键值对，并进行断言
TEST(OrderedDictTest, SubscriptOperatorThrowsWhenPassedKeysThatAreNotPresent) {
  // 使用初始化列表初始化 OrderedDict 对象，键值对为 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言获取键 "a" 对应的值为 1
  ASSERT_EQ(dict["a"], 1);
  // 断言获取键 "b" 对应的值为 2
  ASSERT_EQ(dict["b"], 2);
}

// 测试用例：通过下标操作符以整数作为索引获取元素，并进行断言
TEST(
    OrderedDictTest,
    SubscriptOperatorReturnsItemsPositionallyWhenPassedIntegers) {
  // 使用初始化列表初始化 OrderedDict 对象，键值对为 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1},
TEST(OrderedDictTest, SubscriptOperatorsThrowswhenPassedKeysThatAreNotPresent) {
  // 创建一个有序字典并初始化，包含键值对 {"a", 1} 和 {"b", 2}
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言访问不存在的键 "foo" 会抛出异常，并检查异常消息是否为预期的 "Key 'foo' is not defined"
  ASSERT_THROWS_WITH(dict["foo"], "Key 'foo' is not defined");
  // 断言访问空键 "" 会抛出异常，并检查异常消息是否为预期的 "Key '' is not defined"
  ASSERT_THROWS_WITH(dict[""], "Key '' is not defined");
}

TEST(OrderedDictTest, UpdateInsertsAllItemsFromAnotherOrderedDict) {
  // 创建两个有序字典并初始化
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> dict2 = {{"c", 3}};
  // 将 dict 中的所有项更新到 dict2 中
  dict2.update(dict);
  // 断言 dict2 的大小为 3
  ASSERT_EQ(dict2.size(), 3);
  // 断言 dict2 中存在键 "a"
  ASSERT_NE(dict2.find("a"), nullptr);
  // 断言 dict2 中存在键 "b"
  ASSERT_NE(dict2.find("b"), nullptr);
  // 断言 dict2 中存在键 "c"
  ASSERT_NE(dict2.find("c"), nullptr);
}

TEST(OrderedDictTest, UpdateAlsoChecksForDuplicates) {
  // 创建两个有序字典并初始化，注意其中有重复的键 "a"
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> dict2 = {{"a", 1}};
  // 断言使用 update 方法时会检测到重复定义的键 "a"，并抛出预期的异常 "Key 'a' already defined"
  ASSERT_THROWS_WITH(dict2.update(dict), "Key 'a' already defined");
}

TEST(OrderedDictTest, CanIterateItems) {
  // 创建一个有序字典并初始化
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 获取字典的迭代器
  auto iterator = dict.begin();
  // 断言迭代器不等于字典的末尾迭代器
  ASSERT_NE(iterator, dict.end());
  // 断言迭代器当前指向的键为 "a"
  ASSERT_EQ(iterator->key(), "a");
  // 断言迭代器当前指向的值为 1
  ASSERT_EQ(iterator->value(), 1);
  // 移动迭代器到下一个位置
  ++iterator;
  // 断言迭代器不等于字典的末尾迭代器
  ASSERT_NE(iterator, dict.end());
  // 断言迭代器当前指向的键为 "b"
  ASSERT_EQ(iterator->key(), "b");
  // 断言迭代器当前指向的值为 2
  ASSERT_EQ(iterator->value(), 2);
  // 再次移动迭代器，使其指向字典的末尾
  ++iterator;
  // 断言迭代器等于字典的末尾迭代器
  ASSERT_EQ(iterator, dict.end());
}

TEST(OrderedDictTest, EraseWorks) {
  // 创建一个有序字典并初始化
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}, {"c", 3}};
  // 删除键为 "b" 的项
  dict.erase("b");
  // 断言字典不包含键 "b"
  ASSERT_FALSE(dict.contains("b"));
  // 断言键 "a" 的值为 1
  ASSERT_EQ(dict["a"], 1);
  // 断言键 "c" 的值为 3
  ASSERT_EQ(dict["c"], 3);
  // 再次删除键为 "a" 的项
  dict.erase("a");
  // 断言字典不包含键 "a"
  ASSERT_FALSE(dict.contains("a"));
  // 断言键 "c" 的值仍然为 3
  ASSERT_EQ(dict["c"], 3);
  // 最后删除键为 "c" 的项
  dict.erase("c");
  // 断言字典不包含键 "c"
  ASSERT_FALSE(dict.contains("c"));
  // 断言字典现在为空
  ASSERT_TRUE(dict.is_empty());
}

TEST(OrderedDictTest, ClearMakesTheDictEmpty) {
  // 创建一个有序字典并初始化
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言字典不为空
  ASSERT_FALSE(dict.is_empty());
  // 清空字典
  dict.clear();
  // 断言字典现在为空
  ASSERT_TRUE(dict.is_empty());
}

TEST(OrderedDictTest, CanCopyConstruct) {
  // 创建一个有序字典并初始化
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 使用拷贝构造函数创建一个新字典 copy
  OrderedDict<int> copy = dict;
  // 断言 copy 的大小为 2
  ASSERT_EQ(copy.size(), 2);
  // 断言 copy 的第一个元素的值为 1
  ASSERT_EQ(*copy[0], 1);
  // 断言 copy 的第二个元素的值为 2
  ASSERT_EQ(*copy[1], 2);
}

TEST(OrderedDictTest, CanCopyAssign) {
  // 创建一个有序字典并初始化
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 创建一个有序字典 copy，包含一个键值对 {"c", 1}
  OrderedDict<int> copy = {{"c", 1}};
  // 断言 copy 包含键 "c"
  ASSERT_NE(copy.find("c"), nullptr);
  // 使用拷贝赋值运算符将 dict 的内容复制给 copy
  copy = dict;
  // 断言 copy 的大小为 2
  ASSERT_EQ(copy.size(), 2);
  // 断言 copy 的第一个元素的值为 1
  ASSERT_EQ(*copy[0], 1);
  // 断言 copy 的第二个元素的值为 2
  ASSERT_EQ(*copy[1], 2);
  // 断言 copy 不包含键 "c"
  ASSERT_EQ(copy.find("c"), nullptr);
}

TEST(OrderedDictTest, CanMoveConstruct) {
  // 创建一个有序字典并初始化
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 使用移动构造函数创建一个新字典 copy
  OrderedDict<int> copy = std::move(dict);
  // 断言 copy 的大小为 2
  ASSERT_EQ(copy.size(), 2);
  // 断言 copy 的第一个元素的值为 1
  ASSERT_EQ(*copy[0], 1);
  // 断言 copy 的第二个元素的值为 2
  ASSERT_EQ(*copy[1], 2);
}

TEST(OrderedDictTest, CanMoveAssign) {
  // 创建一个有序字典并初始化
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 创建一个有序字典 copy，包含一个键值对 {"c", 1}
  OrderedDict<int> copy = {{"c", 1}};
  // 断言 copy 包含键 "c"
  ASSERT_NE(copy.find("c"), nullptr);
  // 使用移动赋值运算符将 dict 的内容移动给 copy
  copy = std::move(dict);
  // 断言 copy 的大小为 2
  ASSERT_EQ(copy.size(), 2);
  // 断言 copy 的第一个元素的值为 1
  ASSERT_EQ(*copy[0], 1);
  // 断言 copy 的第二个元素的值为 2
  ASSERT_EQ(*copy[1], 2);
  // 断言 copy 不包含键 "c"
  ASSERT_EQ(copy.find("c"), nullptr);
}

TEST(OrderedDictTest, CanInsertWithBraces) {
  // 创建一个有序字典，并指定值类型为 std::pair<int, int>
  OrderedDict<std::pair<int, int>> dict;
  // 使用花括号语法插入键值对 {"a", {1, 2}}
  dict.insert("a", {1, 2});
  // 断言字典不为空
  ASSERT_FALSE(dict.is_empty());
  // 断言键 "a" 对应的第一个值为 1
// 测试函数：OrderedDictTest，测试错误消息是否包含键的描述信息
TEST(OrderedDictTest, ErrorMessagesIncludeTheKeyDescription) {
  // 创建一个有描述信息 "Penguin" 的有序字典对象
  OrderedDict<int> dict("Penguin");
  // 断言字典的键描述信息为 "Penguin"
  ASSERT_EQ(dict.key_description(), "Penguin");
  // 向字典中插入键值对 "a": 1
  dict.insert("a", 1);
  // 断言字典不为空
  ASSERT_FALSE(dict.is_empty());
  // 断言访问未定义的键 "b" 时抛出异常，异常信息包含 "Penguin 'b' is not defined"
  ASSERT_THROWS_WITH(dict["b"], "Penguin 'b' is not defined");
  // 断言向字典中插入已定义的键 "a": 1 时抛出异常，异常信息包含 "Penguin 'a' already defined"
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Penguin 'a' already defined");
}

// 测试函数：OrderedDictTest，测试 keys 方法返回所有键
TEST(OrderedDictTest, KeysReturnsAllKeys) {
  // 创建一个有序字典对象，包含键值对 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言 keys 方法返回的键列表与 {"a", "b"} 相等
  ASSERT_EQ(dict.keys(), std::vector<std::string>({"a", "b"}));
}

// 测试函数：OrderedDictTest，测试 values 方法返回所有值
TEST(OrderedDictTest, ValuesReturnsAllValues) {
  // 创建一个有序字典对象，包含键值对 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 断言 values 方法返回的值列表与 {1, 2} 相等
  ASSERT_EQ(dict.values(), std::vector<int>({1, 2}));
}

// 测试函数：OrderedDictTest，测试 items 方法返回所有项
TEST(OrderedDictTest, ItemsReturnsAllItems) {
  // 创建一个有序字典对象，包含键值对 "a": 1, "b": 2
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  // 获取字典的所有项
  std::vector<OrderedDict<int>::Item> items = dict.items();
  // 断言项的数量为 2
  ASSERT_EQ(items.size(), 2);
  // 断言第一项的键为 "a"，值为 1
  ASSERT_EQ(items[0].key(), "a");
  ASSERT_EQ(items[0].value(), 1);
  // 断言第二项的键为 "b"，值为 2
  ASSERT_EQ(items[1].key(), "b");
  ASSERT_EQ(items[1].value(), 2);
}


这些注释解释了每个测试函数的目的以及每行代码的作用，确保了代码的每个部分都得到了解释和描述。
```