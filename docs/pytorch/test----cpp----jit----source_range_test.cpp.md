# `.\pytorch\test\cpp\jit\source_range_test.cpp`

```
// 引入 Google Test 框架的头文件
#include <gtest/gtest.h>
// 引入 Torch 的源代码范围头文件
#include <torch/csrc/jit/frontend/source_range.h>

// 使用 testing 命名空间，简化测试框架中的代码书写
using namespace ::testing;
// 使用 torch::jit 命名空间，访问 Torch 的 JIT 模块相关功能
using namespace ::torch::jit;

// 定义 SourceRangeTest 测试套件中的测试用例 test_find
TEST(SourceRangeTest, test_find) {
    // 创建一个存储共享指针的字符串向量
    std::vector<std::shared_ptr<std::string>> strings;
    // 向字符串向量中添加两个共享指针，分别指向 "hello world" 和 "nihaoma" 字符串
    strings.push_back(std::make_shared<std::string>("hello world"));
    strings.push_back(std::make_shared<std::string>("nihaoma"));

    // 创建包含两个 c10::string_view 的向量，每个视图指向 strings 中对应位置的字符串
    std::vector<c10::string_view> pieces{*strings[0], *strings[1]};

    // 创建 StringCordView 对象 view，传入 pieces 和 strings
    StringCordView view(pieces, strings);

    // 在 view 中查找子串 "rldni"，从索引 0 开始查找
    auto x = view.find("rldni", 0);
    // 断言查找结果 x 等于 8
    EXPECT_EQ(x, 8);
}

// 定义 SourceRangeTest 测试套件中的测试用例 test_substr
TEST(SourceRangeTest, test_substr) {
    // 创建一个存储共享指针的字符串向量
    std::vector<std::shared_ptr<std::string>> strings;
    // 向字符串向量中添加两个共享指针，分别指向 "hello world" 和 "nihaoma" 字符串
    strings.push_back(std::make_shared<std::string>("hello world"));
    strings.push_back(std::make_shared<std::string>("nihaoma"));

    // 创建包含两个 c10::string_view 的向量，每个视图指向 strings 中对应位置的字符串
    std::vector<c10::string_view> pieces{*strings[0], *strings[1]};

    // 创建 StringCordView 对象 view，传入 pieces 和 strings
    StringCordView view(pieces, strings);

    // 使用 substr 方法从索引 4 开始截取长度为 10 的子串，并将其转换为 std::string
    auto x = view.substr(4, 10).str();
    // 断言 x 等于 view 原始字符串在索引 4 开始长度为 10 的子串
    EXPECT_EQ(x, view.str().substr(4, 10));
    // 断言使用 substr 方法截取整个 view 的子串并转换为 std::string 后，结果与 view 的原始字符串相等
    EXPECT_EQ(view.substr(0, view.size()).str(), view.str());
}

// 定义 SourceRangeTest 测试套件中的测试用例 test_iter
TEST(SourceRangeTest, test_iter) {
    // 创建一个存储共享指针的字符串向量
    std::vector<std::shared_ptr<std::string>> strings;
    // 向字符串向量中添加两个共享指针，分别指向 "hello world" 和 "nihaoma" 字符串
    strings.push_back(std::make_shared<std::string>("hello world"));
    strings.push_back(std::make_shared<std::string>("nihaoma"));

    // 创建包含两个 c10::string_view 的向量，每个视图指向 strings 中对应位置的字符串
    std::vector<c10::string_view> pieces{*strings[0], *strings[1]};

    // 创建 StringCordView 对象 view，传入 pieces 和 strings
    StringCordView view(pieces, strings);

    // 获取 view 中位置为 5 的迭代器 iter
    auto iter = view.iter_for_pos(5);
    // 断言 iter 指向的字符为 ' '
    EXPECT_EQ(*iter, ' ');
    // 断言 iter 的 rest_line 方法返回的字符串为 " world"
    EXPECT_EQ(iter.rest_line(), " world");
    // 断言 iter 的 next_iter 方法返回的字符为 'w'
    EXPECT_EQ(*iter.next_iter(), 'w');
    // 断言 iter 的当前位置为 5
    EXPECT_EQ(iter.pos(), 5);

    // 重新赋值 iter，获取 view 中位置为 13 的迭代器 iter
    iter = view.iter_for_pos(13);
    // 断言 iter 的当前位置为 13
    EXPECT_EQ(iter.pos(), 13);
}
```