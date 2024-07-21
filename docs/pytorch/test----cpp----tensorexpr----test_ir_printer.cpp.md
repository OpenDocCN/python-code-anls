# `.\pytorch\test\cpp\tensorexpr\test_ir_printer.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架头文件

#include <stdexcept>  // 引入标准异常处理头文件
#include "test/cpp/tensorexpr/test_base.h"  // 引入测试基础类头文件

#include <torch/csrc/jit/tensorexpr/expr.h>  // 引入表达式类头文件
#include <torch/csrc/jit/tensorexpr/ir.h>  // 引入中间表示(IR)类头文件
#include <torch/csrc/jit/tensorexpr/ir_printer.h>  // 引入IR打印类头文件
#include <torch/csrc/jit/tensorexpr/loopnest.h>  // 引入循环嵌套类头文件
#include <torch/csrc/jit/tensorexpr/tensor.h>  // 引入张量类头文件
#include <torch/csrc/jit/testing/file_check.h>  // 引入文件检查头文件

#include <sstream>  // 引入字符串流头文件

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;  // 使用torch::jit::tensorexpr命名空间

TEST(IRPrinter, BasicValueTest) {
  ExprHandle a = IntImm::make(2), b = IntImm::make(3);  // 创建两个整数常量表达式
  ExprHandle c = Add::make(a, b);  // 创建加法表达式

  std::stringstream ss;  // 创建字符串流对象
  ss << c;  // 将表达式c输出到字符串流
  ASSERT_EQ(ss.str(), "2 + 3");  // 断言字符串流中的内容为"2 + 3"
}

TEST(IRPrinter, BasicValueTest02) {
  ExprHandle a(2.0f);  // 创建浮点数常量表达式
  ExprHandle b(3.0f);  // 创建浮点数常量表达式
  ExprHandle c(4.0f);  // 创建浮点数常量表达式
  ExprHandle d(5.0f);  // 创建浮点数常量表达式
  ExprHandle f = (a + b) - (c + d);  // 创建复杂的浮点数表达式

  std::stringstream ss;  // 创建字符串流对象
  ss << f;  // 将表达式f输出到字符串流
  ASSERT_EQ(ss.str(), "(2.f + 3.f) - (4.f + 5.f)");  // 断言字符串流中的内容正确表示表达式f
}

TEST(IRPrinter, CastTest) {
  VarHandle x("x", kHalf);  // 创建半精度浮点数变量
  VarHandle y("y", kFloat);  // 创建单精度浮点数变量
  ExprHandle body = ExprHandle(2.f) + (Cast::make(kFloat, x) * ExprHandle(3.f) + ExprHandle(4.f) * y);  // 创建包含类型转换的表达式

  std::stringstream ss;  // 创建字符串流对象
  ss << body;  // 将表达式body输出到字符串流
  ASSERT_EQ(ss.str(), "2.f + (float(x) * 3.f + 4.f * y)");  // 断言字符串流中的内容正确表示表达式body
}

TEST(IRPrinter, FunctionName) {
  int M = 4;  // 定义整数常量M
  int N = 20;  // 定义整数常量N

  Tensor producer = Compute(  // 创建生产者张量
      "producer", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {  // 通过lambda表达式定义生产者张量的计算
        return m * n;  // 返回计算结果
      });

  Tensor chunk_0 = Compute(  // 创建第一个分块张量
      "chunk_0", {M, N / 2}, [&](const ExprHandle& m, const ExprHandle& n) {  // 通过lambda表达式定义分块张量的计算
        return producer.load(m, n);  // 返回加载的生产者张量值
      });

  Tensor chunk_1 = Compute(  // 创建第二个分块张量
      "chunk_1", {M, N / 2}, [&](const ExprHandle& m, const ExprHandle& n) {  // 通过lambda表达式定义分块张量的计算
        return producer.load(m, n + ExprHandle(N / 2));  // 返回加载的生产者张量值
      });

  Tensor consumer = Compute(  // 创建消费者张量
      "consumer", {M, N / 2}, [&](const ExprHandle& i, const ExprHandle& j) {  // 通过lambda表达式定义消费者张量的计算
        return i * chunk_1.load(i, j);  // 返回计算结果
      });

  LoopNest l({chunk_0, chunk_1, consumer});  // 创建循环嵌套对象
  auto body = LoopNest::sanitizeNames(l.root_stmt());  // 规范化循环嵌套的根语句

  std::stringstream ss;  // 创建字符串流对象
  ss << *body;  // 将循环嵌套的根语句输出到字符串流

  const std::string& verification_pattern =  // 定义字符串流的验证模式
      R"IR(
 # CHECK:   for (int i_2
 # CHECK:    for (int j_2
 # CHECK:     consumer[i_2, j_2] = i_2 * (chunk_1[i_2, j_2])IR";

  torch::jit::testing::FileCheck().run(verification_pattern, ss.str());  // 运行文件检查，验证输出的IR是否符合模式
}

} // namespace jit
} // namespace torch
```