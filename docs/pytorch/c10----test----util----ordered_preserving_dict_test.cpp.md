# `.\pytorch\c10\test\util\ordered_preserving_dict_test.cpp`

```py
#include <algorithm>  // 包含 STL 算法头文件
#include <unordered_set>  // 包含无序集合头文件
#include <vector>  // 包含向量头文件

#include <c10/macros/Macros.h>  // 包含 C10 宏定义头文件
#include <c10/util/Exception.h>  // 包含 C10 异常处理头文件
#include <c10/util/irange.h>  // 包含 C10 迭代范围头文件
#include <c10/util/order_preserving_flat_hash_map.h>  // 包含 C10 自定义的有序平面哈希映射头文件
#include <gtest/gtest.h>  // 包含 Google 测试框架头文件

namespace {  // 命名空间开始

#define ASSERT_EQUAL_PRIM(t1, t2) ASSERT_TRUE(t1 == t2);  // 定义宏，用于断言两个基本类型相等

using dict_int_int =  // 使用别名定义有序平面哈希映射的键值对类型
    ska_ordered::order_preserving_flat_hash_map<int64_t, int64_t>;

dict_int_int test_dict(dict_int_int& dict) {  // 定义测试函数，接受一个有序哈希映射的引用
  for (const auto i : c10::irange(100)) {  // 循环插入键值对到哈希映射中
    dict[i] = i + 1;
  }

  int64_t entry_i = 0;  // 初始化条目索引
  for (auto entry : dict) {  // 遍历哈希映射中的条目
    TORCH_INTERNAL_ASSERT(  // 内部断言，验证键值对的正确性
        entry.first == entry_i && entry.second == entry_i + 1);
    ++entry_i;  // 更新条目索引
  }

  // 删除特定键
  std::unordered_set<int64_t> erase_set = {0, 2, 9, 71};
  for (auto erase : erase_set) {  // 遍历要删除的键集合
    dict.erase(erase);  // 从哈希映射中删除指定的键
  }

  // 通过迭代器删除键
  auto begin = dict.begin();  // 获取起始迭代器
  for (const auto i : c10::irange(20)) {  // 循环移动迭代器到指定位置
    (void)i; // Suppress unused variable warning
    begin++;
  }

  auto end = begin;  // 复制起始迭代器
  for (const auto i : c10::irange(20)) {  // 循环移动结束迭代器到指定位置
    (void)i; // Suppress unused variable warning
    erase_set.insert(end->first);  // 将迭代器当前键插入到要删除的键集合中
    end++;
  }
  dict.erase(begin, end);  // 通过迭代器范围删除键

  std::vector<int64_t> order;  // 定义保存键的顺序的向量
  for (const auto i : c10::irange(100)) {  // 循环生成键序列
    if (!erase_set.count(i)) {  // 如果键不在要删除的集合中
      order.push_back(i);  // 将键加入顺序向量中
    }
  }

  entry_i = 0;  // 重置条目索引
  for (auto entry : dict) {  // 遍历哈希映射中的条目
    TORCH_INTERNAL_ASSERT(order[entry_i] == entry.first);  // 内部断言，验证顺序的正确性
    TORCH_INTERNAL_ASSERT(dict[order[entry_i]] == entry.second);  // 内部断言，验证值的正确性
    TORCH_INTERNAL_ASSERT(entry.second == order[entry_i] + 1);  // 内部断言，验证值的正确性
    entry_i++;  // 更新条目索引
  }
  TORCH_INTERNAL_ASSERT(dict.size() == order.size());  // 内部断言，验证哈希映射大小与顺序向量大小相同
  return dict;  // 返回操作后的哈希映射
}

TEST(OrderedPreservingDictTest, InsertAndDeleteBasic) {  // 定义测试用例，插入和删除基础操作
  dict_int_int dict;  // 创建有序哈希映射对象
  test_dict(dict);  // 调用测试函数
  dict.clear();  // 清空哈希映射
  test_dict(dict);  // 再次调用测试函数
}

TEST(OrderedPreservingDictTest, InsertExistingDoesntAffectOrder) {  // 定义测试用例，插入已存在键不影响顺序
  dict_int_int dict;  // 创建有序哈希映射对象
  dict[0] = 1;  // 插入键值对
  dict[1] = 2;  // 插入键值对

  TORCH_INTERNAL_ASSERT(dict.begin()->first == 0);  // 内部断言，验证起始键为0
  dict[0] = 1;  // 更新键值对
  TORCH_INTERNAL_ASSERT(dict.begin()->first == 0);  // 内部断言，验证起始键仍为0
  dict[0] = 2;  // 更新键值对
  TORCH_INTERNAL_ASSERT(dict.begin()->first == 0);  // 内部断言，验证起始键仍为0

  dict.erase(0);  // 删除键为0的键值对
  TORCH_INTERNAL_ASSERT(dict.begin()->first == 1);  // 内部断言，验证起始键变为1
}

TEST(OrderedPreservingDictTest, testRefType) {  // 定义测试用例，测试引用类型
  std::shared_ptr<int64_t> t;  // 创建共享指针对象
  using dict_references =  // 使用别名定义有序平面哈希映射的键值对类型
      ska_ordered::order_preserving_flat_hash_map<int64_t, std::shared_ptr<int64_t>>;

  dict_references dict;  // 创建有序哈希映射对象

  auto ptr = std::make_shared<int64_t>(1);  // 创建共享指针，指向整数1
  dict[1] = ptr;  // 插入键值对
  TORCH_INTERNAL_ASSERT(ptr.use_count() == 2);  // 内部断言，验证共享指针的引用计数为2
  dict.erase(1);  // 删除键为1的键值对
  TORCH_INTERNAL_ASSERT(ptr.use_count() == 1);  // 内部断言，验证共享指针的引用计数为1

  dict[2] = ptr;  // 再次插入键值对
  dict.clear();  // 清空哈希映射
  TORCH_INTERNAL_ASSERT(ptr.use_count() == 1);  // 内部断言，验证共享指针的引用计数为1
}

TEST(OrderedPreservingDictTest, DictCollisions) {  // 定义测试用例，测试哈希冲突
  struct BadHash {  // 定义结构体，实现坏的哈希函数
    size_t operator()(const int64_t input) {  // 重载运算符，实现哈希函数
      return input % 2;  // 返回输入的余数
    };
  };

  using bad_hash_dict =  // 使用别名定义有序平面哈希映射的键值对类型，指定坏的哈希函数
      ska_ordered::order_preserving_flat_hash_map<int64_t, int64_t, BadHash>;

  for (auto init_dict_size : {27, 34, 41}) {  // 循环初始化哈希映射的大小
    bad_hash_dict dict;  // 创建有序哈希映射对象
    // 使用范围循环初始化字典 `dict`，将每个索引 `i` 映射到 `i + 1`
    for (const auto i : c10::irange(init_dict_size)) {
      dict[i] = i + 1;
    }

    // 初始化变量 `i` 为 0，遍历字典 `dict` 的每个条目
    int64_t i = 0;
    for (const auto& entry : dict) {
      // 断言每个条目的键为 `i`，值为 `i + 1`
      TORCH_INTERNAL_ASSERT(entry.first == i && entry.second == i + 1);
      ++i;
    }

    // 删除几个条目
    std::unordered_set<int64_t> erase_set = {0, 2, 9};
    for (auto erase : erase_set) {
      // 从字典 `dict` 中删除键为 `erase` 的条目
      dict.erase(erase);
    }

    // 通过迭代器删除几个条目
    auto begin = dict.begin();
    // 移动 `begin` 迭代器，跳过前 10 个条目
    for (const auto j : c10::irange(10)) {
      (void)j; // 抑制未使用变量警告
      begin++;
    }
    auto end = begin;
    // 继续移动 `end` 迭代器，跳过接下来的 7 个条目
    for (const auto j : c10::irange(7)) {
      (void)j; // 抑制未使用变量警告
      // 将 `end` 迭代器当前位置的键插入到 `erase_set` 中
      erase_set.insert(end->first);
      end++;
    }
    // 使用迭代器范围，从 `begin` 到 `end` 删除字典 `dict` 中的条目
    dict.erase(begin, end);

    // 创建一个顺序向量 `order`，存储未删除条目的键的顺序
    std::vector<int64_t> order;
    for (const auto j : c10::irange(init_dict_size)) {
      // 如果 `j` 不在 `erase_set` 中，则将其添加到 `order` 中
      if (!erase_set.count(j)) {
        order.push_back(j);
      }
    }

    // 重新初始化 `i` 为 0，再次遍历字典 `dict` 的每个条目
    i = 0;
    for (auto entry : dict) {
      // 断言字典中每个条目的值等于相应条目在 `order` 中的位置加 1
      TORCH_INTERNAL_ASSERT(dict[entry.first] == entry.second);
      TORCH_INTERNAL_ASSERT(dict[entry.first] == order[i] + 1);
      // 断言 `order` 中的每个元素与相应的字典键相等
      TORCH_INTERNAL_ASSERT(order[i] == entry.first);
      i += 1;
    }
    // 断言字典 `dict` 的大小与向量 `order` 的大小相等
    TORCH_INTERNAL_ASSERT(dict.size() == order.size());
}

// Tests taken from
// https://github.com/Tessil/ordered-map/blob/master/tests/ordered_map_tests.cpp

TEST(OrderedPreservingDictTest, test_range_insert) {
  // 在向量中插入 x 个值，从向量中的第 x-15 个值到最后第5个值范围插入到映射中，检查值

  const int nb_values = 1000;  // 定义值的总数为 1000
  std::vector<std::pair<int, int>> values;  // 创建一个存放整数对的向量
  for (const auto i : c10::irange(nb_values)) {  // 对值的范围进行迭代
    values.emplace_back(i, i + 1);  // 将每个整数对插入到向量中
  }

  dict_int_int map = {{-1, 0}, {-2, 0}};  // 创建一个包含初始键值对的有序平面哈希映射
  map.insert(values.begin() + 10, values.end() - 5);  // 将从第10个到倒数第5个值范围内的值插入映射中

  ASSERT_EQUAL_PRIM(map.size(), 987);  // 断言映射的大小为987

  ASSERT_EQUAL_PRIM(map.at(-1), 0);  // 断言键 -1 的值为 0

  ASSERT_EQUAL_PRIM(map.at(-2), 0);  // 断言键 -2 的值为 0

  for (int i = 10, j = 2; i < nb_values - 5; i++, j++) {  // 对从10到nb_values-6范围内的每个整数进行迭代
    ASSERT_EQUAL_PRIM(map.at(i), i + 1);  // 断言键 i 的值为 i+1
  }
}

TEST(OrderedPreservingDictTest, test_range_erase_all) {
  // 插入 x 个值，删除所有值

  const std::size_t nb_values = 1000;  // 定义值的总数为 1000
  dict_int_int map;  // 创建一个空的有序平面哈希映射
  for (const int64_t i : c10::irange<int64_t>(nb_values)) {  // 对值的范围进行迭代
    map[i] = i + 1;  // 将每个键值对插入到映射中
  }
  auto it = map.erase(map.begin(), map.end());  // 删除映射中所有的键值对
  ASSERT_TRUE(it == map.end());  // 断言迭代器 it 等于映射的结束迭代器
  ASSERT_TRUE(map.empty());  // 断言映射为空
}

TEST(OrderedPreservingDictTest, test_range_erase) {
  // 插入 x 个值，使用迭代器删除除了前10个和最后780个值之外的所有值

  using HMap = ska_ordered::order_preserving_flat_hash_map<std::string, std::int64_t>;  // 使用有序平面哈希映射定义 HMap

  const int64_t nb_values = 1000;  // 定义值的总数为 1000
  HMap map;  // 创建一个空的有序平面哈希映射
  for (const auto i : c10::irange(nb_values)) {  // 对值的范围进行迭代
    map[std::to_string(i)] = i;  // 将每个键值对插入到映射中，键为整数 i 的字符串形式，值为 i
    auto begin = map.begin();  // 获取映射的开始迭代器
    for (int64_t j = 0; j <= i; ++j, begin++) {  // 对从0到i范围内的每个整数进行迭代
      TORCH_INTERNAL_ASSERT(begin->second == j);  // 断言当前迭代器指向的值的第二个元素为 j
    }
  }

  auto it_first = std::next(map.begin(), 10);  // 获取映射中第10个元素的迭代器
  auto it_last = std::next(map.begin(), 220);  // 获取映射中第220个元素的迭代器

  auto it = map.erase(it_first, it_last);  // 使用迭代器范围删除映射中的元素
  ASSERT_EQUAL_PRIM(std::distance(it, map.end()), 780);  // 断言从迭代器 it 到映射结束的距离为780
  ASSERT_EQUAL_PRIM(map.size(), 790);  // 断言映射的大小为 790
  ASSERT_EQUAL_PRIM(std::distance(map.begin(), map.end()), 790);  // 断言映射的元素数量为 790

  for (auto& val : map) {  // 对映射中的每个值进行迭代
    ASSERT_EQUAL_PRIM(map.count(val.first), 1);  // 断言映射中键为 val.first 的数量为 1
  }

  // 检查顺序
  it = map.begin();  // 将迭代器 it 设置为映射的开始迭代器
  for (std::size_t i = 0; i < nb_values; i++) {  // 对值的范围进行迭代
    if (i >= 10 && i < 220) {  // 如果 i 在10到220之间
      continue;  // 跳过当前迭代
    }
    auto exp_it = std::pair<std::string, std::int64_t>(std::to_string(i), i);  // 创建一个期望的键值对
    TORCH_INTERNAL_ASSERT(*it == exp_it);  // 断言迭代器 it 指向的值等于期望的键值对
    ++it;  // 递增迭代器 it
  }
}

TEST(OrderedPreservingDictTest, test_move_constructor_empty) {
  ska_ordered::order_preserving_flat_hash_map<std::string, int64_t> map(0);  // 创建一个空的有序平面哈希映射
  ska_ordered::order_preserving_flat_hash_map<std::string, int64_t> map_move(
      std::move(map));  // 使用移动构造函数创建一个新的映射

  // NOLINTNEXTLINE(bugprone-use-after-move)
  TORCH_INTERNAL_ASSERT(map.empty());  // 断言原始映射为空
  TORCH_INTERNAL_ASSERT(map_move.empty());  // 断言新的映射为空

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  TORCH_INTERNAL_ASSERT(map.find("") == map.end());  // 断言在原始映射中找不到空字符串
  TORCH_INTERNAL_ASSERT(map_move.find("") == map_move.end());  // 断言在新的映射中找不到空字符串
}
TEST(OrderedPreservingDictTest, test_move_operator_empty) {
  // 创建一个初始容量为0的有序保留平面哈希映射
  ska_ordered::order_preserving_flat_hash_map<std::string, int64_t> map(0);
  // 创建一个空的有序保留平面哈希映射
  ska_ordered::order_preserving_flat_hash_map<std::string, int64_t> map_move;
  // 使用移动操作符将map的内容移动到map_move中
  map_move = (std::move(map));

  // 断言移动后map为空
  TORCH_INTERNAL_ASSERT(map.empty());
  // 断言移动后map_move也为空
  TORCH_INTERNAL_ASSERT(map_move.empty());

  // 断言map中不存在空键""
  TORCH_INTERNAL_ASSERT(map.find("") == map.end());
  // 断言map_move中不存在空键""
  TORCH_INTERNAL_ASSERT(map_move.find("") == map_move.end());
}

TEST(OrderedPreservingDictTest, test_reassign_moved_object_move_constructor) {
  // 定义一个键和值类型均为字符串的有序保留平面哈希映射类型HMap
  using HMap = ska_ordered::order_preserving_flat_hash_map<std::string, std::string>;

  // 初始化一个包含三对键值对的HMap对象
  HMap map = {{"Key1", "Value1"}, {"Key2", "Value2"}, {"Key3", "Value3"}};
  // 使用移动构造函数将map的内容移动到map_move中
  HMap map_move(std::move(map));

  // 断言移动后map_move的大小为3
  ASSERT_EQUAL_PRIM(map_move.size(), 3);
  // 断言移动后map为空
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_TRUE(map.empty());

  // 重新分配map的内容
  map = {{"Key4", "Value4"}, {"Key5", "Value5"}};
  // 断言map等于包含两对键值对的HMap对象
  TORCH_INTERNAL_ASSERT(
      map == (HMap({{"Key4", "Value4"}, {"Key5", "Value5"}})));
}

TEST(OrderedPreservingDictTest, test_reassign_moved_object_move_operator) {
  // 定义一个键和值类型均为字符串的有序保留平面哈希映射类型HMap
  using HMap = ska_ordered::order_preserving_flat_hash_map<std::string, std::string>;

  // 初始化一个包含三对键值对的HMap对象
  HMap map = {{"Key1", "Value1"}, {"Key2", "Value2"}, {"Key3", "Value3"}};
  // 使用移动操作符将map的内容移动到map_move中
  HMap map_move = std::move(map);

  // 断言移动后map_move的大小为3
  ASSERT_EQUAL_PRIM(map_move.size(), 3);
  // 断言移动后map为空
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_TRUE(map.empty());

  // 重新分配map的内容
  map = {{"Key4", "Value4"}, {"Key5", "Value5"}};
  // 断言map等于包含两对键值对的HMap对象
  TORCH_INTERNAL_ASSERT(
      map == (HMap({{"Key4", "Value4"}, {"Key5", "Value5"}})));
}

TEST(OrderedPreservingDictTest, test_copy_constructor_and_operator) {
  // 定义一个键和值类型均为字符串的有序保留平面哈希映射类型HMap
  using HMap = ska_ordered::order_preserving_flat_hash_map<std::string, std::string>;

  // 定义要生成的键值对数量
  const std::size_t nb_values = 100;
  // 初始化一个空的HMap对象
  HMap map;
  // 填充map使其包含nb_values个键值对，键和值都为字符串形式
  for (const auto i : c10::irange(nb_values)) {
    map[std::to_string(i)] = std::to_string(i);
  }

  // 使用拷贝构造函数创建map_copy，复制map的内容
  HMap map_copy = map;
  // 使用拷贝构造函数创建map_copy2，复制map的内容
  HMap map_copy2(map);
  // 初始化一个空的HMap对象map_copy3，并添加一对键值对
  HMap map_copy3;
  map_copy3[std::to_string(0)] = std::to_string(0);

  // 使用赋值操作符将map的内容赋值给map_copy3
  map_copy3 = map;

  // 断言map和map_copy相等
  TORCH_INTERNAL_ASSERT(map == map_copy);
  // 清空map
  map.clear();

  // 断言map_copy和map_copy2相等
  TORCH_INTERNAL_ASSERT(map_copy == map_copy2);
  // 断言map_copy和map_copy3相等
  TORCH_INTERNAL_ASSERT(map_copy == map_copy3);
}

TEST(OrderedPreservingDictTest, test_copy_constructor_empty) {
  // 创建一个初始容量为0的有序保留平面哈希映射
  ska_ordered::order_preserving_flat_hash_map<std::string, int> map(0);
  // 使用拷贝构造函数创建map_copy，复制map的内容
  ska_ordered::order_preserving_flat_hash_map<std::string, int> map_copy(map);

  // 断言map为空
  TORCH_INTERNAL_ASSERT(map.empty());
  // 断言map_copy也为空
  TORCH_INTERNAL_ASSERT(map_copy.empty());

  // 断言map中不存在空键""
  TORCH_INTERNAL_ASSERT(map.find("") == map.end());
  // 断言map_copy中不存在空键""
  TORCH_INTERNAL_ASSERT(map_copy.find("") == map_copy.end());
}
TEST(OrderedPreservingDictTest, test_copy_operator_empty) {
  // 创建一个空的有序保留哈希映射对象 map，容量为 0
  ska_ordered::order_preserving_flat_hash_map<std::string, int> map(0);
  // 创建一个有序保留哈希映射对象 map_copy，预分配容量为 16
  ska_ordered::order_preserving_flat_hash_map<std::string, int> map_copy(16);
  // 将 map 的内容复制给 map_copy
  map_copy = map;

  // 断言 map 和 map_copy 都是空的
  TORCH_INTERNAL_ASSERT(map.empty());
  TORCH_INTERNAL_ASSERT(map_copy.empty());

  // 断言在 map 中找不到空字符串 ""
  TORCH_INTERNAL_ASSERT(map.find("") == map.end());
  // 断言在 map_copy 中找不到空字符串 ""
  TORCH_INTERNAL_ASSERT(map_copy.find("") == map_copy.end());
}

/**
 * at
 */
TEST(OrderedPreservingDictTest, test_at) {
  // 创建一个有序保留哈希映射对象 map，包含键值对 {0: 10, -2: 20}
  const ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>
      map = {{0, 10}, {-2, 20}};

  // 使用 at 方法获取已知键 0 对应的值，并断言其为 10
  ASSERT_EQUAL_PRIM(map.at(0), 10);
  // 使用 at 方法获取已知键 -2 对应的值，并断言其为 20
  ASSERT_EQUAL_PRIM(map.at(-2), 20);
  
  // 尝试使用 at 方法获取未知键 1，预期抛出异常，捕获异常并设置标志为 true
  bool thrown = false;
  try {
    map.at(1);
  } catch (...) {
    thrown = true;
  }
  // 断言捕获到异常
  ASSERT_TRUE(thrown);
}

/**
 * equal_range
 */
TEST(OrderedPreservingDictTest, test_equal_range) {
  // 创建一个有序保留哈希映射对象 map，包含键值对 {0: 10, -2: 20}
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map =
      {{0, 10}, {-2, 20}};

  // 使用 equal_range 方法查找键 0，获取到的迭代器对应的范围中应包含一个元素
  auto it_pair = map.equal_range(0);
  ASSERT_EQUAL_PRIM(std::distance(it_pair.first, it_pair.second), 1);
  // 断言范围中第一个元素的值为 10
  ASSERT_EQUAL_PRIM(it_pair.first->second, 10);

  // 使用 equal_range 方法查找键 1，得到的范围应为空，且第一个迭代器等于 map 的末尾迭代器
  it_pair = map.equal_range(1);
  TORCH_INTERNAL_ASSERT(it_pair.first == it_pair.second);
  TORCH_INTERNAL_ASSERT(it_pair.first == map.end());
}

/**
 * operator[]
 */
TEST(OrderedPreservingDictTest, test_access_operator) {
  // 创建一个有序保留哈希映射对象 map，包含键值对 {0: 10, -2: 20}
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map =
      {{0, 10}, {-2, 20}};

  // 使用访问操作符 [] 获取已知键 0 对应的值，并断言其为 10
  ASSERT_EQUAL_PRIM(map[0], 10);
  // 使用访问操作符 [] 获取已知键 -2 对应的值，并断言其为 20
  ASSERT_EQUAL_PRIM(map[-2], 20);
  // 使用访问操作符 [] 获取未知键 2 对应的值，预期其为 std::int64_t() 的默认值（通常为 0）
  ASSERT_EQUAL_PRIM(map[2], std::int64_t());

  // 断言 map 的大小为 3
  ASSERT_EQUAL_PRIM(map.size(), 3);
}

/**
 * swap
 */
TEST(OrderedPreservingDictTest, test_swap) {
  // 创建有序保留哈希映射对象 map，包含键值对 {1: 10, 8: 80, 3: 30}
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map =
      {{1, 10}, {8, 80}, {3, 30}};
  // 创建有序保留哈希映射对象 map2，包含键值对 {4: 40, 5: 50}
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map2 =
      {{4, 40}, {5, 50}};

  // 使用 std::swap 函数交换 map 和 map2 的内容
  using std::swap;
  swap(map, map2);

  // 断言 map 现在包含键值对 {4: 40, 5: 50}
  TORCH_INTERNAL_ASSERT(
      map ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {4, 40}, {5, 50}}));
  // 断言 map2 现在包含键值对 {1: 10, 8: 80, 3: 30}
  TORCH_INTERNAL_ASSERT(
      map2 ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {1, 10}, {8, 80}, {3, 30}}));

  // 向 map 中插入键值对 {6: 60}
  map.insert({6, 60});
  // 向 map2 中插入键值对 {4: 40}，这是已存在的键，应更新其值
  map2.insert({4, 40});

  // 断言 map 现在包含键值对 {4: 40, 5: 50, 6: 60}
  TORCH_INTERNAL_ASSERT(
      map ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {4, 40}, {5, 50}, {6, 60}}));
  // 断言 map2 现在包含键值对 {1: 10, 8: 80, 3: 30, 4: 40}
  TORCH_INTERNAL_ASSERT(
      map2 ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {1, 10}, {8, 80}, {3, 30}, {4, 40}}));
}
TEST(OrderedPreservingDictTest, test_swap_empty) {
  // 创建一个有序保留键顺序的哈希映射，包含三对键值对
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map =
      {{1, 10}, {8, 80}, {3, 30}};
  // 创建一个空的有序保留键顺序的哈希映射
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map2;

  // 导入 C++ 标准库的 swap 函数
  using std::swap;
  // 交换 map 和 map2 的内容
  swap(map, map2);

  // 断言 map 现在应该是一个空的有序保留键顺序的哈希映射
  TORCH_INTERNAL_ASSERT(
      // NOLINTNEXTLINE(readability-container-size-empty)
      map ==
      (ska_ordered::
           order_preserving_flat_hash_map<std::int64_t, std::int64_t>{}));
  // 断言 map2 现在应该包含之前 map 的所有键值对
  TORCH_INTERNAL_ASSERT(
      map2 ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {1, 10}, {8, 80}, {3, 30}}));

  // 向 map 中插入一对键值对
  map.insert({6, 60});
  // 向 map2 中插入一对键值对
  map2.insert({4, 40});

  // 断言 map 现在应该只包含新插入的一对键值对
  TORCH_INTERNAL_ASSERT(
      map ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {6, 60}}));
  // 断言 map2 现在应该包含之前的三对键值对和新插入的一对键值对
  TORCH_INTERNAL_ASSERT(
      map2 ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {1, 10}, {8, 80}, {3, 30}, {4, 40}}));
}

} // namespace
```