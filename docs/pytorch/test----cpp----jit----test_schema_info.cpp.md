# `.\pytorch\test\cpp\jit\test_schema_info.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件
#include <torch/csrc/autograd/generated/variable_factories.h>  // 引入 Torch 的变量工厂相关头文件
#include <torch/csrc/utils/schema_info.h>  // 引入 Torch 的模式信息工具头文件

namespace torch {
namespace utils {

using c10::SchemaArgType;  // 使用 C10 库中定义的模式参数类型

// 测试用例，验证函数模式中的别名情况
TEST(FunctionSchemaIsAliasingTest, Basic) {
  // 解析给定的函数模式字符串，创建函数模式对象 schema
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(a) self, Tensor(b!) other, Tensor more_other) -> (Tensor(a), Tensor(b!))");
  // 断言第一个输出参数是否具有别名
  ASSERT_TRUE(schema.is_aliasing({SchemaArgType::output, 0}));
  // 断言第二个输出参数是否具有别名
  ASSERT_TRUE(schema.is_aliasing({SchemaArgType::output, 1}));
  // 断言第一个输入参数是否具有别名
  ASSERT_TRUE(schema.is_aliasing({SchemaArgType::input, 0}));
  // 断言第二个输入参数是否具有别名
  ASSERT_TRUE(schema.is_aliasing({SchemaArgType::input, 1}));
  // 断言第三个输入参数是否不具有别名
  ASSERT_FALSE(schema.is_aliasing({SchemaArgType::input, 2}));
}

// 测试用例，验证函数模式中的无效参数情况
TEST(FunctionSchemaIsAliasingTest, InvalidArgument) {
  // 解析给定的函数模式字符串，创建函数模式对象 schema
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 断言访问第四个输入参数是否引发错误
  ASSERT_THROW(schema.is_aliasing({SchemaArgType::input, 4}), c10::Error);
  // 断言访问第四个输出参数是否引发错误
  ASSERT_THROW(schema.is_aliasing({SchemaArgType::output, 4}), c10::Error);
}

// 测试用例，验证函数模式中的可变性情况
TEST(FunctionSchemaIsMutableTest, Basic) {
  // 解析给定的函数模式字符串，创建函数模式对象 schema
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 断言第一个输出参数是否可变
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::output, 0}));
  // 断言第一个输入参数是否可变
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 0}));
  // 断言 self 参数是否可变
  ASSERT_TRUE(schema.is_mutable("self"));
  // 断言第二个输入参数是否不可变
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 1}));
  // 断言 other 参数是否不可变
  ASSERT_FALSE(schema.is_mutable("other"));
  // 断言第三个输入参数是否不可变
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 2}));
  // 断言 alpha 参数是否不可变
  ASSERT_FALSE(schema.is_mutable("alpha"));
}

// 测试用例，验证函数模式中的无效参数情况
TEST(FunctionSchemaIsMutableTest, InvalidArgument) {
  // 解析给定的函数模式字符串，创建函数模式对象 schema
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 断言访问第四个输入参数是否引发错误
  ASSERT_THROW(schema.is_mutable({SchemaArgType::input, 4}), c10::Error);
  // 断言访问名为 "named_argument" 的参数是否引发错误
  ASSERT_THROW(schema.is_mutable("named_argument"), c10::Error);
}

// 测试用例，验证 SchemaInfo 对象中的可变性情况
TEST(SchemaInfoIsMutableTest, Basic) {
  // 创建 SchemaInfo 对象，并解析给定的函数模式字符串
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 断言第一个输入参数是否可变
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 0}));
  // 断言 self 参数是否可变
  ASSERT_TRUE(schema.is_mutable("self"));
  // 断言第二个输入参数是否不可变
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 1}));
  // 断言 other 参数是否不可变
  ASSERT_FALSE(schema.is_mutable("other"));
  // 断言第三个输入参数是否不可变
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 2}));
  // 断言 alpha 参数是否不可变
  ASSERT_FALSE(schema.is_mutable("alpha"));
}

// 测试用例，验证 SchemaInfo 对象中的无效参数情况
TEST(SchemaInfoIsMutableTest, InvalidArgument) {
  // 创建 SchemaInfo 对象，并解析给定的函数模式字符串
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 断言访问第四个输入参数是否引发错误
  ASSERT_THROW(schema.is_mutable({SchemaArgType::input, 4}), c10::Error);
  // 断言访问名为 "named_argument" 的参数是否引发错误
  ASSERT_THROW(schema.is_mutable("named_argument"), c10::Error);
}

}  // namespace utils
}  // namespace torch
TEST(SchemaInfoIsMutableTest, AliasingInputs) {
  // 创建名为 schema 的 SchemaInfo 对象，用于表示特定的函数签名信息
  SchemaInfo schema(
      "aten::test.Tensor(Tensor(a!) self, Tensor(b) other, *, Scalar alpha=1) -> (Tensor(a!), Tensor(b))");
  // 断言第一个输入参数 self 是可变的
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 0}));
  // 断言第一个输出参数也是可变的
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::output, 0}));
  // 断言参数 "self" 是可变的
  ASSERT_TRUE(schema.is_mutable("self"));
  // 断言第二个输入参数 other 是不可变的
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 1}));
  // 断言第二个输出参数不是可变的
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::output, 1}));
  // 断言参数 "other" 不是可变的
  ASSERT_FALSE(schema.is_mutable("other"));
  // 创建一个形状为 [3, 3] 的随机张量 input
  at::Tensor input = at::randn({3, 3});
  // 为参数 "self" 添加输入值 input
  schema.addArgumentValue("self", input);
  // 为参数 "other" 添加输入值 input
  schema.addArgumentValue("other", input);
  // 断言第二个输入参数 other 现在是可变的
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 1}));
  // 断言第二个输出参数也是可变的
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::output, 1}));
  // 断言参数 "other" 现在是可变的
  ASSERT_TRUE(schema.is_mutable("other"));
}

TEST(SchemaInfoIsMutableTest, InstanceNorm) {
  // 创建名为 schema_info 的 SchemaInfo 对象，表示 instance_norm 函数的签名信息
  SchemaInfo schema_info(
      "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor");
  // 断言参数 "running_mean" 是可变的
  ASSERT_TRUE(schema_info.is_mutable("running_mean"));
  // 断言参数 "running_var" 是可变的
  ASSERT_TRUE(schema_info.is_mutable("running_var"));
  // 为参数 "use_input_stats" 添加输入值 false
  schema_info.addArgumentValue("use_input_stats", false);
  // 断言参数 "running_mean" 现在是不可变的
  ASSERT_FALSE(schema_info.is_mutable("running_mean"));
  // 断言参数 "running_var" 现在是不可变的
  ASSERT_FALSE(schema_info.is_mutable("running_var"));
}

TEST(SchemaInfoIsMutableTest, BatchNorm) {
  // 创建名为 schema_info 的 SchemaInfo 对象，表示 batch_norm 函数的签名信息
  SchemaInfo schema_info(
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor");
  // 断言参数 "running_mean" 是可变的
  ASSERT_TRUE(schema_info.is_mutable("running_mean"));
  // 断言参数 "running_var" 是可变的
  ASSERT_TRUE(schema_info.is_mutable("running_var"));
  // 为参数 "training" 添加输入值 false
  schema_info.addArgumentValue("training", false);
  // 断言参数 "running_mean" 现在是不可变的
  ASSERT_FALSE(schema_info.is_mutable("running_mean"));
  // 断言参数 "running_var" 现在是不可变的
  ASSERT_FALSE(schema_info.is_mutable("running_var"));
}

TEST(SchemaInfoIsNonDeterministicTest, Basic) {
  // 创建名为 deterministic_schema_info 的 SchemaInfo 对象，表示 sub_ 函数的签名信息
  SchemaInfo deterministic_schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 创建名为 nondeterministic_schema_info 的 SchemaInfo 对象，表示 bernoulli 函数的签名信息
  SchemaInfo nondeterministic_schema_info(
      "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor");
  // 断言 deterministic_schema_info 表示的函数不是非确定性的
  ASSERT_FALSE(deterministic_schema_info.is_nondeterministic());
  // 断言 nondeterministic_schema_info 表示的函数是非确定性的
  ASSERT_TRUE(nondeterministic_schema_info.is_nondeterministic());
}

TEST(SchemaInfoIsNonDeterministicTest, Dropout) {
  // 创建名为 dropout_schema_info 的 SchemaInfo 对象，表示 dropout 函数的签名信息
  SchemaInfo dropout_schema_info(
      "aten::dropout(Tensor input, float p, bool train) -> Tensor");
  // 断言 dropout_schema_info 表示的函数是非确定性的
  ASSERT_TRUE(dropout_schema_info.is_nondeterministic());
  // 为参数 "train" 添加输入值 false
  dropout_schema_info.addArgumentValue("train", false);
  // 断言 dropout_schema_info 表示的函数现在不是非确定性的
  ASSERT_FALSE(dropout_schema_info.is_nondeterministic());
}
TEST(FunctionSchemaMayAliasTest, Basic) {
  // 解析给定的函数签名，生成函数模式对象
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 检查输入和输出参数是否可能引用同一数据，返回真值断言
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  // 检查输入参数1和输出参数是否可能引用同一数据，返回假值断言
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  // 检查输入参数1和输入参数0是否可能引用同一数据，返回假值断言
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayAliasTest, InvalidArgument) {
  // 解析给定的函数签名，生成函数模式对象
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 检查输入参数15和输出参数0是否可能引用同一数据，预期抛出异常
  ASSERT_THROW(
      schema.may_alias({SchemaArgType::input, 15}, {SchemaArgType::output, 0}),
      c10::Error);
  // 检查输入参数0和输出参数15是否可能引用同一数据，预期抛出异常
  ASSERT_THROW(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 15}),
      c10::Error);
}

TEST(FunctionSchemaMayAliasTest, Wildcard) {
  // 解析给定的函数签名，生成函数模式对象
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(*) self) -> (Tensor(*), Tensor)");
  // 检查输出参数0和输入参数0是否可能引用同一数据，返回真值断言
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  // 检查输出参数1和输入参数0是否可能引用同一数据，返回假值断言
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 1}, {SchemaArgType::input, 0}));
}

TEST(SchemaInfoMayAliasTest, AliasingInputs) {
  // 创建一个函数模式信息对象
  SchemaInfo schema(
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
  // 检查输入参数0和输入参数1是否可能引用同一数据，返回假值断言
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 创建一个形状为[3, 3]的随机张量作为输入值
  at::Tensor input = at::randn({3, 3});
  // 向函数模式信息对象添加参数值 "self"
  schema.addArgumentValue("self", input);
  // 向函数模式信息对象添加参数值 "other"
  schema.addArgumentValue("other", input);
  // 检查输入参数0和输入参数1是否可能引用同一数据，返回真值断言
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
}

TEST(SchemaInfoMayAliasTest, AliasingOutputs) {
  // 创建一个函数模式信息对象
  SchemaInfo schema(
      "aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)");
  // 检查输出参数0和输出参数1是否可能引用同一数据，返回假值断言
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
  // 创建一个形状为[3, 3]的随机张量作为输入值
  at::Tensor input = at::randn({3, 3});
  // 向函数模式信息对象添加参数值 "min"
  schema.addArgumentValue("min", input);
  // 向函数模式信息对象添加参数值 "max"
  schema.addArgumentValue("max", input);
  // 检查输出参数0和输出参数1是否可能引用同一数据，返回真值断言
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
}

TEST(SchemaInfoMayAliasTest, AliasingInputOutput) {
  // 创建一个函数模式信息对象
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 检查输入参数0和输出参数0是否可能引用同一数据，返回真值断言
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  // 检查输入参数1和输出参数0是否可能引用同一数据，返回假值断言
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  // 创建一个形状为[3, 3]的随机张量作为输入值
  at::Tensor input = at::randn({3, 3});
  // 向函数模式信息对象添加参数值 "self"
  schema.addArgumentValue("self", input);
  // 向函数模式信息对象添加参数值 "other"
  schema.addArgumentValue("other", input);
  // 检查输入参数0和输出参数0是否可能引用同一数据，返回真值断言
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  // 检查输入参数1和输出参数0是否可能引用同一数据，返回真值断言
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
}
TEST(SchemaInfoMayAliasTest, MultipleWildcardInputs) {
  // 创建一个名为 `schema` 的 SchemaInfo 对象，传入指定的字符串表示符号化函数的签名
  SchemaInfo schema(
      "aten::test.Tensor(Tensor(a) a, Tensor(*) b, Tensor(*) c) -> (Tensor(a), Tensor(*))");
  // 断言输入参数0与输出参数0可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  // 断言输入参数1与输出参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 1}));
  // 断言输入参数2与输出参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::output, 1}));
  // 断言输入参数0与输入参数1不可能别名
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 断言输入参数0与输入参数2不可能别名
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  // 断言输入参数0与输出参数1不可能别名
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 1}));
  // 断言输入参数1与输出参数0不可能别名
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  // 创建一个形状为 [3, 3] 的随机张量 `input`
  at::Tensor input = at::randn({3, 3});
  // 将张量 `input` 添加到 `schema` 的参数值中，键为 "a"
  schema.addArgumentValue("a", input);
  // 将张量 `input` 添加到 `schema` 的参数值中，键为 "b"
  schema.addArgumentValue("b", input);
  // 断言输入参数0与输出参数0可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  // 断言输入参数1与输出参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 1}));
  // 断言输入参数2与输出参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::output, 1}));
  // 断言输入参数0与输入参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 断言输入参数0与输入参数2可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  // 断言输入参数0与输出参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 1}));
  // 断言输入参数1与输出参数0可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
}

TEST(SchemaInfoMayAliasTest, MultipleNonWildcardInputs) {
  // 创建一个名为 `schema` 的 SchemaInfo 对象，传入指定的字符串表示符号化函数的签名
  SchemaInfo schema(
      "aten::test.Tensor(Tensor(a) a, Tensor(a) b, Tensor(*) c, Tensor(b) d) -> (Tensor(a), Tensor(*))");
  // 断言输入参数0与输入参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 断言输入参数0与输入参数2可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  // 断言输入参数2与输入参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::input, 1}));
  // 断言输入参数2与输出参数0可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::output, 0}));
}

TEST(SchemaInfoMayAliasTest, MultipleNonWildcardOutputs) {
  // 创建一个名为 `schema` 的 SchemaInfo 对象，传入指定的字符串表示符号化函数的签名
  SchemaInfo schema(
      "aten::test.Tensor(Tensor(a) a, Tensor(*) b) -> (Tensor(a), Tensor(a))");
  // 断言输入参数0与输入参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 断言输出参数0与输出参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
  // 断言输出参数0与输入参数1可能别名
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
}

TEST(SchemaInfoMayAliasTest, MismatchingTypes) {
  // 创建一个名为 `schema` 的 SchemaInfo 对象，传入指定的字符串表示符号化函数的签名
  SchemaInfo schema("aten::test.Tensor(Tensor(a) a) -> int(a)");
  // 断言输入参数0与输出参数0不可能别名
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
}
TEST(FunctionSchemaMayContainAliasTest, Basic) {
  // 解析函数的 schema 字符串，创建 FunctionSchema 对象
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  // 断言输入的第一个参数可能与输出的第一个参数存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  // 断言输入的第二个参数不可能与输出的第一个参数存在别名关系
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  // 断言输入的第二个参数不可能与输入的第一个参数存在别名关系
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, Wildcard) {
  // 解析函数的 schema 字符串，创建 FunctionSchema 对象
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(*) self) -> (Tensor[], Tensor)");
  // 断言输出的第一个参数不可能与输入的第一个参数存在别名关系
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  // 断言输出的第一个参数可能与输入的第一个参数存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  // 断言输出的第一个参数可能与输入的第一个参数存在别名关系（忽略是否完全一致）
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}, false));
  // 断言输入的第一个参数不可能与输出的第一个参数存在别名关系（忽略是否完全一致）
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}, false));
  // 断言输出的第二个参数不可能与输入的第一个参数存在别名关系
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 1}, {SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, InputAndOutputContainers) {
  // 解析函数的 schema 字符串，创建 FunctionSchema 对象
  c10::FunctionSchema schema =
      torch::jit::parseSchema("aten::test.Tensor(Tensor[] self) -> Tensor[]");
  // 断言输出的第一个参数不可能与输入的第一个参数存在别名关系
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  // 断言输出的第一个参数可能与输入的第一个参数存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  // 断言输出的第一个参数可能与输入的第一个参数存在别名关系（忽略是否完全一致）
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}, false));
  // 断言输入的第一个参数可能与输出的第一个参数存在别名关系（忽略是否完全一致）
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}, false));
}

TEST(SchemaInfoMayContainAliasTest, ContainAliasInputsEqual) {
  // 创建 SchemaInfo 对象，指定函数的 schema 字符串
  SchemaInfo schema(
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
  // 断言输入的第一个参数不可能与输入的第二个参数存在别名关系
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 创建一个大小为 3x3 的随机 Tensor
  at::Tensor input = at::randn({3, 3});
  // 向 schema 对象添加参数值映射
  schema.addArgumentValue("self", input);
  schema.addArgumentValue("other", input);
  // 断言输入的第一个参数可能与输入的第二个参数存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 断言输入的第一个参数可能与输入的第二个参数存在别名关系（忽略是否完全一致）
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}, false));
  // 断言输入的第二个参数可能与输入的第一个参数存在别名关系（忽略是否完全一致）
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::input, 0}, false));
}
TEST(SchemaInfoMayContainAliasTest, ContainAliasInputsContained) {
  // 创建一个 SchemaInfo 对象，用指定的函数签名初始化
  SchemaInfo schema(
      "aten::test.Tensor(Tensor[] self, Tensor other, *, Scalar alpha=1) -> Tensor");
  // 断言输入参数0不可能与输入参数1存在别名关系
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 创建一个形状为[3, 3]的随机张量作为输入
  at::Tensor input = at::randn({3, 3});
  // 将名为 "self" 的参数添加到 schema 中，值为单个张量列表
  schema.addArgumentValue("self", c10::List<at::Tensor>({input}));
  // 将名为 "other" 的参数添加到 schema 中，值为单个张量
  schema.addArgumentValue("other", input);
  // 断言输入参数0可能与输入参数1存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 再次断言输入参数0可能与输入参数1存在别名关系，但不强制
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}, false));
  // 断言输入参数1不可能与输入参数0存在别名关系，不强制
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::input, 0}, false));
}

TEST(SchemaInfoMayContainAliasTest, ContainAliasOutputs) {
  // 创建一个 SchemaInfo 对象，用指定的函数签名初始化
  SchemaInfo schema(
      "aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)");
  // 断言输出参数0不可能与输出参数1存在别名关系
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
  // 创建一个形状为[3, 3]的随机张量作为输入
  at::Tensor input = at::randn({3, 3});
  // 将名为 "min" 的输出参数添加到 schema 中，值为单个张量
  schema.addArgumentValue("min", input);
  // 将名为 "max" 的输出参数添加到 schema 中，值为单个张量
  schema.addArgumentValue("max", input);
  // 断言输出参数0可能与输出参数1存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
}

TEST(SchemaInfoMayContainAliasTest, ContainAliasInputOutput) {
  // 创建一个 SchemaInfo 对象，用指定的函数签名初始化
  SchemaInfo schema(
      "aten::test.tensor(Tensor(a) self, Tensor[] other) -> Tensor(a)");
  // 断言输出参数0不可能与输入参数1存在别名关系
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
  // 创建一个形状为[3, 3]的随机张量作为输入
  at::Tensor input = at::randn({3, 3});
  // 将名为 "other" 的输入参数添加到 schema 中，值为单个张量列表
  schema.addArgumentValue("other", c10::List<at::Tensor>({input}));
  // 将名为 "self" 的输入参数添加到 schema 中，值为单个张量
  schema.addArgumentValue("self", input);
  // 断言输出参数0可能与输入参数1存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
  // 断言输出参数0不可能与输入参数1存在别名关系，不强制
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}, false));
  // 断言输入参数1可能与输出参数0存在别名关系，不强制
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::output, 0}, false));
}

TEST(SchemaInfoMayContainAliasTest, InputAndOutputContainers) {
  // 创建一个 SchemaInfo 对象，用指定的函数签名初始化
  SchemaInfo schema(
      "aten::test.tensor(Tensor self, Tensor[] other) -> Tensor[]");
  // 断言输出参数0可能与输入参数1存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
  // 断言输出参数0不可能与输入参数0存在别名关系
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  // 断言输入参数0不可能与输入参数1存在别名关系
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 创建一个形状为[3, 3]的随机张量作为输入
  at::Tensor input = at::randn({3, 3});
  // 将名为 "other" 的输入参数添加到 schema 中，值为单个张量列表
  schema.addArgumentValue("other", c10::List<at::Tensor>({input}));
  // 将名为 "self" 的输入参数添加到 schema 中，值为单个张量
  schema.addArgumentValue("self", input);
  // 断言输出参数0可能与输入参数1存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
  // 断言输出参数0可能与输入参数0存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  // 断言输入参数0可能与输入参数1存在别名关系
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
}
TEST(SchemaInfoMayContainAliasTest, Wildcard) {
  // 创建一个 SchemaInfo 对象，传入特定的字符串作为构造参数
  SchemaInfo schema(
      "aten::test.tensor(Tensor a, Tensor[] b, Tensor(*) c) -> Tensor[]");
  // 断言：参数 input 0 和 input 2 不可能有别名
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  // 断言：参数 input 0 和 input 1 不可能有别名
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 断言：参数 input 2 和 input 1 可能有别名
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 2}, {SchemaArgType::input, 1}));
  // 创建一个大小为 (3, 3) 的随机张量 input
  at::Tensor input = at::randn({3, 3});
  // 向 schema 对象添加一个名称为 "b" 的参数，其值为包含 input 的张量列表
  schema.addArgumentValue("b", c10::List<at::Tensor>({input}));
  // 向 schema 对象添加一个名称为 "a" 的参数，其值为 input 张量
  schema.addArgumentValue("a", input);
  // 断言：参数 input 0 和 input 2 可能有别名
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  // 断言：参数 input 0 和 input 1 可能有别名
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  // 断言：参数 input 2 和 input 1 可能有别名
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 2}, {SchemaArgType::input, 1}));
}
```