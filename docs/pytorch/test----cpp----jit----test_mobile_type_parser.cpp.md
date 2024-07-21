# `.\pytorch\test\cpp\jit\test_mobile_type_parser.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>
// 包含测试工具函数的头文件
#include <test/cpp/jit/test_utils.h>

// 包含 ATen 类型定义的头文件
#include <ATen/core/jit_type.h>
// 包含移动端类型解析器的头文件
#include <torch/csrc/jit/mobile/type_parser.h>

// 定义 torch 命名空间
namespace torch {
namespace jit {

// 测试移动端类型解析器中的整数类型解析
TEST(MobileTypeParserTest, Int) {
  // 定义整数类型字符串
  std::string int_ps("int");
  // 解析整数类型字符串为类型指针
  auto int_tp = c10::parseType(int_ps);
  // 断言解析后的类型指针与预期的整数类型指针相等
  EXPECT_EQ(*int_tp, *IntType::get());
}

// 测试移动端类型解析器中嵌套容器类型的注释字符串解析
TEST(MobileTypeParserTest, NestedContainersAnnotationStr) {
  // 定义嵌套容器类型字符串
  std::string tuple_ps(
      "Tuple[str, Optional[float], Dict[str, List[Tensor]], int]");
  // 解析嵌套容器类型字符串为类型指针
  auto tuple_tp = c10::parseType(tuple_ps);
  // 创建预期的类型指针
  std::vector<TypePtr> args = {
      c10::StringType::get(),
      c10::OptionalType::create(c10::FloatType::get()),
      c10::DictType::create(
          StringType::get(), ListType::create(TensorType::get())),
      IntType::get()};
  auto tp = TupleType::create(std::move(args));
  // 断言解析后的类型指针与预期的类型指针相等
  ASSERT_EQ(*tuple_tp, *tp);
}

// 测试移动端类型解析器中 Torch Bind 类的类型解析
TEST(MobileTypeParserTest, TorchBindClass) {
  // 定义 Torch Bind 类的类型字符串
  std::string tuple_ps("__torch__.torch.classes.rnn.CellParamsBase");
  // 解析 Torch Bind 类的类型字符串为类型指针
  auto tuple_tp = c10::parseType(tuple_ps);
  // 获取类型指针的注释字符串
  std::string tuple_tps = tuple_tp->annotation_str();
  // 断言解析后的注释字符串与原始类型字符串相等
  ASSERT_EQ(tuple_ps, tuple_tps);
}

// 测试移动端类型解析器中 Torch Bind 类的列表类型解析
TEST(MobileTypeParserTest, ListOfTorchBindClass) {
  // 定义 Torch Bind 类的列表类型字符串
  std::string tuple_ps("List[__torch__.torch.classes.rnn.CellParamsBase]");
  // 解析 Torch Bind 类的列表类型字符串为类型指针
  auto tuple_tp = c10::parseType(tuple_ps);
  // 断言类型指针是 AnyListType 的子类型
  EXPECT_TRUE(tuple_tp->isSubtypeOf(AnyListType::get()));
  // 断言类型指针包含的第一个类型的注释字符串与 Torch Bind 类相等
  EXPECT_EQ(
      "__torch__.torch.classes.rnn.CellParamsBase",
      tuple_tp->containedType(0)->annotation_str());
}

// 测试移动端类型解析器中带空格的嵌套容器类型的注释字符串解析
TEST(MobileTypeParserTest, NestedContainersAnnotationStrWithSpaces) {
  // 定义带空格的嵌套容器类型字符串
  std::string tuple_space_ps(
      "Tuple[  str, Optional[float], Dict[str, List[Tensor ]]  , int]");
  // 解析带空格的嵌套容器类型字符串为类型指针
  auto tuple_space_tp = c10::parseType(tuple_space_ps);
  // 获取解析后的类型指针的注释字符串
  std::string tuple_space_tps = tuple_space_tp->annotation_str();
  // 断言解析后的注释字符串不包含奇怪的空白符
  ASSERT_TRUE(tuple_space_tps.find("[ ") == std::string::npos);
  ASSERT_TRUE(tuple_space_tps.find(" ]") == std::string::npos);
  ASSERT_TRUE(tuple_space_tps.find(" ,") == std::string::npos);
}

// 测试移动端类型解析器中命名元组类型字符串的解析
TEST(MobileTypeParserTest, NamedTuple) {
  // 定义命名元组类型字符串
  std::string named_tuple_ps(
      "__torch__.base_models.preproc_types.PreprocOutputType["
      "    NamedTuple, ["
      "        [float_features, Tensor],"
      "        [id_list_features, List[Tensor]],"
      "        [label,  Tensor],"
      "        [weight, Tensor],"
      "        [prod_prediction, Tuple[Tensor, Tensor]],"
      "        [id_score_list_features, List[Tensor]],"
      "        [embedding_features, List[Tensor]],"
      "        [teacher_label, Tensor]"
      "        ]"
      "    ]");

  // 解析命名元组类型字符串为类型指针
  c10::TypePtr named_tuple_tp = c10::parseType(named_tuple_ps);
  // 获取解析后类型指针的注释字符串
  std::string named_tuple_annotation_str = named_tuple_tp->annotation_str();
  // 断言解析后的注释字符串与原始类型字符串相等
  ASSERT_EQ(
      named_tuple_annotation_str,
      "__torch__.base_models.preproc_types.PreprocOutputType");
}
TEST(MobileTypeParserTest, DictNestedNamedTupleTypeList) {
  // 定义第一个测试用例中的类型字符串，描述了一个嵌套结构的命名元组类型列表
  std::string type_str_1(
      "__torch__.base_models.preproc_types.PreprocOutputType["
      "  NamedTuple, ["
      "      [float_features, Tensor],"
      "      [id_list_features, List[Tensor]],"
      "      [label,  Tensor],"
      "      [weight, Tensor],"
      "      [prod_prediction, Tuple[Tensor, Tensor]],"
      "      [id_score_list_features, List[Tensor]],"
      "      [embedding_features, List[Tensor]],"
      "      [teacher_label, Tensor]"
      "      ]");
  // 定义第二个测试用例中的类型字符串，描述了一个字典类型映射字符串到预处理输出类型
  std::string type_str_2(
      "Dict[str, __torch__.base_models.preproc_types.PreprocOutputType]");
  // 创建包含以上类型字符串的向量
  std::vector<std::string> type_strs = {type_str_1, type_str_2};
  // 解析类型字符串为类型指针的向量
  std::vector<c10::TypePtr> named_tuple_tps = c10::parseType(type_strs);
  // 断言第二个解析得到的类型指针中的第一个子类型是字符串类型
  EXPECT_EQ(*named_tuple_tps[1]->containedType(0), *c10::StringType::get());
  // 断言第一个解析得到的类型指针等于第二个解析得到的类型指针的第二个子类型
  EXPECT_EQ(*named_tuple_tps[0], *named_tuple_tps[1]->containedType(1));
}

TEST(MobileTypeParserTest, NamedTupleNestedNamedTupleTypeList) {
  // 定义第一个测试用例中的类型字符串，描述了一个嵌套的命名元组类型列表
  std::string type_str_1(
      " __torch__.ccc.xxx ["
      "    NamedTuple, ["
      "      [field_name_c_1, Tensor],"
      "      [field_name_c_2, Tuple[Tensor, Tensor]]"
      "    ]"
      "]");
  // 定义第二个测试用例中的类型字符串，描述了一个嵌套的命名元组类型列表
  std::string type_str_2(
      "__torch__.bbb.xxx ["
      "    NamedTuple,["
      "        [field_name_b, __torch__.ccc.xxx]]"
      "    ]"
      "]");
  // 定义第三个测试用例中的类型字符串，描述了一个嵌套的命名元组类型列表
  std::string type_str_3(
      "__torch__.aaa.xxx["
      "    NamedTuple, ["
      "        [field_name_a, __torch__.bbb.xxx]"
      "    ]"
      "]");
  // 创建包含以上类型字符串的向量
  std::vector<std::string> type_strs = {type_str_1, type_str_2, type_str_3};
  // 解析类型字符串为类型指针的向量
  std::vector<c10::TypePtr> named_tuple_tps = c10::parseType(type_strs);
  // 获取第三个解析得到的类型指针的注解字符串
  std::string named_tuple_annotation_str = named_tuple_tps[2]->annotation_str();
  // 断言第三个解析得到的类型指针的注解字符串等于 "__torch__.aaa.xxx"
  ASSERT_EQ(named_tuple_annotation_str, "__torch__.aaa.xxx");
}

TEST(MobileTypeParserTest, NamedTupleNestedNamedTuple) {
  // 定义一个嵌套的命名元组类型的字符串表示
  std::string named_tuple_ps(
      "__torch__.aaa.xxx["
      "    NamedTuple, ["
      "        [field_name_a, __torch__.bbb.xxx ["
      "            NamedTuple, ["
      "                [field_name_b, __torch__.ccc.xxx ["
      "                    NamedTuple, ["
      "                      [field_name_c_1, Tensor],"
      "                      [field_name_c_2, Tuple[Tensor, Tensor]]"
      "                    ]"
      "                ]"
      "                ]"
      "            ]"
      "        ]"
      "        ]"
      "    ]   "
      "]");
  // 解析类型字符串为类型指针
  c10::TypePtr named_tuple_tp = c10::parseType(named_tuple_ps);
  // 获取解析得到的类型指针的字符串表示
  std::string named_tuple_annotation_str = named_tuple_tp->str();
  // 断言解析得到的类型指针的字符串表示等于 "__torch__.aaa.xxx"
  ASSERT_EQ(named_tuple_annotation_str, "__torch__.aaa.xxx");
}

// 解析异常情况
TEST(MobileTypeParserTest, Empty) {
  // 定义一个空的类型字符串
  std::string empty_ps("");
  // 断言解析空类型字符串会抛出异常
  ASSERT_ANY_THROW(c10::parseType(empty_ps));
}
TEST(MobileTypeParserTest, TypoRaises) {
  // 创建一个包含拼写错误的字符串作为输入
  std::string typo_token("List[tensor]");
  // 使用 ASSERT_ANY_THROW 宏来检查是否抛出异常，期望抛出异常是因为输入字符串有拼写错误
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(typo_token));
}

TEST(MobileTypeParserTest, MismatchBracketRaises) {
  // 创建一个包含不匹配括号的字符串作为输入
  std::string mismatch1("List[Tensor");
  // 使用 ASSERT_ANY_THROW 宏来检查是否抛出异常，期望抛出异常是因为输入字符串的括号不匹配
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch1));
}

TEST(MobileTypeParserTest, MismatchBracketRaises2) {
  // 创建一个包含不匹配括号的字符串作为输入
  std::string mismatch2("List[[Tensor]");
  // 使用 ASSERT_ANY_THROW 宏来检查是否抛出异常，期望抛出异常是因为输入字符串的括号不匹配
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch2));
}

TEST(MobileTypeParserTest, DictWithoutValueRaises) {
  // 创建一个包含缺少值的字典类型字符串作为输入
  std::string mismatch3("Dict[Tensor]");
  // 使用 ASSERT_ANY_THROW 宏来检查是否抛出异常，期望抛出异常是因为输入字符串的字典类型缺少值类型
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch3));
}

TEST(MobileTypeParserTest, ListArgCountMismatchRaises) {
  // 创建一个包含参数数量不匹配的列表类型字符串作为输入
  std::string mismatch4("List[int, str]");
  // 使用 ASSERT_ANY_THROW 宏来检查是否抛出异常，期望抛出异常是因为输入字符串的列表类型参数数量不匹配
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch4));
}

TEST(MobileTypeParserTest, DictArgCountMismatchRaises) {
  // 创建一个包含参数数量不匹配的字典类型字符串作为输入
  std::string trailing_commm("Dict[str,]");
  // 使用 ASSERT_ANY_THROW 宏来检查是否抛出异常，期望抛出异常是因为输入字符串的字典类型参数数量不匹配
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(trailing_commm));
}

TEST(MobileTypeParserTest, ValidTypeWithExtraStuffRaises) {
  // 创建一个包含额外内容的有效类型字符串作为输入
  std::string extra_stuff("int int");
  // 使用 ASSERT_ANY_THROW 宏来检查是否抛出异常，期望抛出异常是因为输入字符串包含了多余的内容
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(extra_stuff));
}

TEST(MobileTypeParserTest, NonIdentifierRaises) {
  // 创建一个包含非标识符内容的字符串作为输入
  std::string non_id("(int)");
  // 使用 ASSERT_ANY_THROW 宏来检查是否抛出异常，期望抛出异常是因为输入字符串不是有效的标识符
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(non_id));
}

TEST(MobileTypeParserTest, DictNestedNamedTupleTypeListRaises) {
  // 创建多个复杂的类型字符串作为输入
  std::string type_str_1(
      "Dict[str, __torch__.base_models.preproc_types.PreprocOutputType]");
  std::string type_str_2(
      "__torch__.base_models.preproc_types.PreprocOutputType["
      "  NamedTuple, ["
      "      [float_features, Tensor],"
      "      [id_list_features, List[Tensor]],"
      "      [label,  Tensor],"
      "      [weight, Tensor],"
      "      [prod_prediction, Tuple[Tensor, Tensor]],"
      "      [id_score_list_features, List[Tensor]],"
      "      [embedding_features, List[Tensor]],"
      "      [teacher_label, Tensor]"
      "      ]");
  // 将多个类型字符串放入 vector 中作为输入
  std::vector<std::string> type_strs = {type_str_1, type_str_2};
  // 定义期望的错误消息字符串
  std::string error_message =
      R"(Can't find definition for the type: __torch__.base_models.preproc_types.PreprocOutputType)";
  // 使用 ASSERT_THROWS_WITH_MESSAGE 宏来检查是否抛出异常，并且检查异常的错误消息是否符合预期
  ASSERT_THROWS_WITH_MESSAGE(c10::parseType(type_strs), error_message);
}

} // namespace jit
} // namespace torch
```