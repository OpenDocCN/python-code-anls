# `.\pytorch\test\cpp\tensorexpr\test_registerizer.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include "test/cpp/tensorexpr/test_base.h"  // 引入测试基类的头文件

#include "test/cpp/tensorexpr/test_utils.h"  // 引入测试工具的头文件
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"  // 引入 IR 简化器的头文件
#include "torch/csrc/jit/tensorexpr/registerizer.h"  // 引入寄存器分配器的头文件

#include <iostream>  // 引入标准输入输出流的头文件

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;  // 使用 Torch JIT TensorExpr 命名空间

// 可以用本地变量替换简单的标量访问。
TEST(Registerizer, RegisterizerSimple) {
  BufHandle a("A", {1}, kInt);  // 定义名为 A 的缓冲区，大小为 1，类型为 kInt
  VarHandle x("x", kInt);  // 定义名为 x 的变量，类型为 kInt
  StmtPtr stmt = Block::make(  // 创建多个语句组成的块
      {Store::make(a, {0}, 0),  // 将 0 存储到缓冲区 A 的索引 0 处
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {0}), x))}))});  // 循环体内将 A[0] 更新为当前值加上 x

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  stmt = registerize(stmt);  // 调用寄存器分配器对语句进行优化

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;  // 创建输出字符串流对象
  oss << *stmt;  // 将优化后的语句打印到字符串流中

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用 FileCheck 验证输出是否符合预期
}

// 不会替换循环访问。
TEST(Registerizer, RegisterizerLoop) {
  BufHandle a("A", {10}, kInt);  // 定义名为 A 的缓冲区，大小为 10，类型为 kInt
  VarHandle x("x", kInt);  // 定义名为 x 的变量，类型为 kInt
  StmtPtr stmt = Block::make(  // 创建多个语句组成的块
      {Store::make(a, {0}, 0),  // 将 0 存储到缓冲区 A 的索引 0 处
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {x}, Add::make(Load::make(a, {x}), x))}))});  // 循环体内将 A[x] 更新为当前值加上 x

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[x]) + x;
   * }
   */

  // No change.  // 没有变化

  stmt = registerize(stmt);  // 调用寄存器分配器对语句进行优化

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[x]) + x;
   * }
   */

  std::ostringstream oss;  // 创建输出字符串流对象
  oss << *stmt;  // 将优化后的语句打印到字符串流中

  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: int
# CHECK: A[0] = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A_
# CHECK:   A[x] =
# CHECK-NOT: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用 FileCheck 验证输出是否符合预期
}

// 即使加载的是固定标量，也不会替换，因为存储可能会使其无效。
TEST(Registerizer, RegisterizerLoopFixedLoad) {
  BufHandle a("A", {1}, kInt);  // 定义名为 A 的缓冲区，大小为 1，类型为 kInt
  VarHandle x("x", kInt);  // 定义名为 x 的变量，类型为 kInt
  StmtPtr stmt = Block::make(  // 创建多个语句组成的块
      {Store::make(a, {0}, 0),  // 将 0 存储到缓冲区 A 的索引 0 处
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {x}, Add::make(Load::make(a, {0}), x))}))});  // 循环体内将 A[x] 更新为 A[0] 加上 x

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[0]) + x;
   * }
   */

  // No change.  // 没有变化

  stmt = registerize(stmt);  // 调用寄存器分配器对语句进行优化

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[0]) + x;
   * }
   */

  std::ostringstream oss;  // 创建输出字符串流对象
  oss << *stmt;  // 将优化后的语句打印到字符串流中

  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: int
# CHECK: A[0] = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A_
// CHECK:   A[x] =
// CHECK-NOT: A[0] = A_1;)IR";

// 运行测试用例，验证生成的 IR 是否符合期望的字符串模式
torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 我们可以在内部作用域完全发生的访问中进行寄存器分配，即使它们依赖于循环变量。
TEST(Registerizer, RegisterizerLoopInternal) {
  // 创建一个大小为1的整型缓存 "A"
  BufHandle a("A", {1}, kInt);
  // 创建一个整型变量 "x"
  VarHandle x("x", kInt);
  // 创建一个语句，包含一个循环和两个存储操作
  StmtPtr stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make(
          {Store::make(a, {x}, Add::make(Load::make(a, {x}), x)),  // 存储操作，计算 A[x] = (A[x]) + x;
           Store::make(a, {x}, Add::make(Load::make(a, {x}), x))}))});  // 存储操作，计算 A[x] = (A[x]) + x;

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[x]) + x;
   *   A[x] = (A[x]) + x;
   * }
   */

  // 对语句进行寄存器分配处理
  stmt = registerize(stmt);

  // 在字符串流中打印处理后的语句
  std::ostringstream oss;
  oss << *stmt;

  // TODO: 加法操作中项的顺序可能会改变，一般取决于某个哈希值。这导致操作数的随机交换，这不是很好。
  // 理想情况下，我们应该确保某种特定的顺序（最好是原始顺序）。
  /*
   * for (int x = 0; x < 10; x++) {
   *   int A_1 = A[x];
   *   A_1 = x + A_1;
   *   A_1 = x + A_1;
   *   A[x] = A_1;
   * }
   */

  // 定义用于验证的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int x = 0; x < 10; x++)
# CHECK: int A_1 = A[x];
# CHECK:   A_1 = A_1 + x;
# CHECK:   A_1 = A_1 + x;
# CHECK:   A[x] = A_1;
# CHECK: })IR";

  // 运行测试用例，验证生成的 IR 是否符合期望的字符串模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 一个访问可以被同一个表达式中的另一个读取所覆盖。在这种情况下，B[z] 和 B[y] 重叠，并阻止了两个访问的寄存器化。
TEST(Registerizer, RegisterizerLoopInternalLoadOverlap) {
  // 创建大小为10的整型缓存 "A" 和 "B"
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  // 创建整型变量 "x", "y", "z"
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  // 创建一个包含存储操作的语句
  StmtPtr stmt = Block::make({For::make(
      x,
      0,
      10,
      Store::make(a, {x}, Add::make(Load::make(b, {y}), Load::make(b, {z}))))});
  // 对语句进行简化处理
  stmt = IRSimplifier::simplify(stmt);

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (B[y]) + (B[z]);
   * }
   */

  // 在处理前打印语句
  std::ostringstream before;
  before << *stmt;

  // 执行寄存器分配处理，不应有变化
  stmt = registerize(stmt);

  // 在处理后打印语句
  std::ostringstream after;
  after << *stmt;

  // 断言处理前后语句的字符串表示应该相等
  ASSERT_EQ(before.str(), after.str());
}
TEST(Registerizer, RegisterizerLoopInternalRepeated) {
  BufHandle a("A", {1}, kInt);  // 创建名为 "A" 的缓冲区对象，大小为 {1}，类型为 kInt
  VarHandle x("x", kInt);  // 创建名为 "x" 的变量对象，类型为 kInt
  StmtPtr stmt = Block::make(
      {For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {1}), x)),  // 将 A[0] = A[1] + x 的赋值操作添加到块中
                Store::make(a, {0}, Add::make(Load::make(a, {1}), x))})),  // 再次将 A[0] = A[1] + x 的赋值操作添加到块中
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {1}), x)),  // 将 A[0] = A[1] + x 的赋值操作添加到块中
                Store::make(a, {0}, Add::make(Load::make(a, {1}), x))}))  // 再次将 A[0] = A[1] + x 的赋值操作添加到块中
      });

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[0] = x + (A[1]);
   *   A[0] = x + (A[1]);
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[0] = x + (A[1]);
   *   A[0] = x + (A[1]);
   * }
   */

  stmt = registerize(stmt);  // 对 stmt 进行寄存器分配优化

  /*
   * int A_1 = A[1];
   * int A_2 = A[0];
   * for (int x = 0; x < 10; x++) {
   *   A_2 = A_1 + x;
   *   A_2 = A_1 + x;
   * }
   * for (int x = 0; x < 10; x++) {
   *   A_2 = A_1 + x;
   *   A_2 = A_1 + x;
   * }
   * A[0] = A_2;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[1];
# CHECK: int A_2 = A[0];
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   A_2 = A_1 + x;
# CHECK:   A_2 = A_1 + x;
# CHECK: }
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   A_2 = A_1 + x;
# CHECK:   A_2 = A_1 + x;
# CHECK: }
# CHECK-NOT: A[1]
# CHECK: A[0] = A_2;
# CHECK-NOT: A[1]
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Registerizer, RegisterizerLoopInternalRepeatedOverlapLoopVar) {
  BufHandle a("A", {1}, kInt);  // 创建名为 "A" 的缓冲区对象，大小为 {1}，类型为 kInt
  VarHandle x("x", kInt);  // 创建名为 "x" 的变量对象，类型为 kInt
  StmtPtr stmt = Block::make(
      {For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {x}), x)),  // 将 A[0] = A[x] + x 的赋值操作添加到块中
                Store::make(a, {0}, Add::make(Load::make(a, {x}), x))})),  // 再次将 A[0] = A[x] + x 的赋值操作添加到块中
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {x}), x)),  // 将 A[0] = A[x] + x 的赋值操作添加到块中
                Store::make(a, {0}, Add::make(Load::make(a, {x}), x))}))  // 再次将 A[0] = A[x] + x 的赋值操作添加到块中
      });
  stmt = IRSimplifier::simplify(stmt);  // 对 stmt 进行 IR 简化

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[x]) + x;
   *   A[0] = (A[x]) + x;
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[x]) + x;
   *   A[0] = (A[x]) + x;
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);  // 对 stmt 进行寄存器分配优化

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());  // 断言寄存器分配前后代码没有变化
}
TEST(Registerizer, RegisterizerLoopInternalRepeatedOverlapOther) {
  // 创建一个名为 "A" 的缓冲区对象，大小为 {1}，元素类型为整型
  BufHandle a("A", {1}, kInt);
  // 创建名为 "x" 和 "y" 的整型变量对象
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 简化并生成IR语句块
  StmtPtr stmt = IRSimplifier::simplify(Block::make(
      {For::make(
           x,
           0,
           10,
           Block::make(
               {
                   // 将 A[0] = (A[x]) + x; 的存储操作添加到块中
                   Store::make(a, {0}, Add::make(x, Load::make(a, {y}))),
                   // 将 A[0] = (A[x]) + x; 的存储操作再次添加到块中
                   Store::make(a, {0}, Add::make(x, Load::make(a, {y})))
               })),
       For::make(
           x,
           0,
           10,
           Block::make(
               {
                   // 将 A[0] = (A[x]) + x; 的存储操作添加到块中
                   Store::make(a, {0}, Add::make(x, Load::make(a, {y}))),
                   // 将 A[0] = (A[x]) + x; 的存储操作再次添加到块中
                   Store::make(a, {0}, Add::make(x, Load::make(a, {y})))
               }))
      }));

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[x]) + x;
   *   A[0] = (A[x]) + x;
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[x]) + x;
   *   A[0] = (A[x]) + x;
   * }
   */

  // 创建一个输出流对象 before，将 stmt 的内容写入其中
  std::ostringstream before;
  before << *stmt;

  // 对 stmt 进行寄存器分配处理
  stmt = registerize(stmt);

  // 创建一个输出流对象 after，将处理后的 stmt 的内容写入其中
  std::ostringstream after;
  after << *stmt;

  // 断言：处理前后的 IR 表示应该保持一致
  ASSERT_EQ(before.str(), after.str());
}

// 将同一缓冲区的不同项的多次访问寄存器化处理
TEST(Registerizer, RegisterizerMultiVar) {
  // 创建一个名为 "A" 的缓冲区对象，大小为 {2}，元素类型为整型
  BufHandle a("A", {2}, kInt);
  // 创建名为 "x" 的整型变量对象
  VarHandle x("x", kInt);
  // 生成 IR 语句块
  StmtPtr stmt = Block::make({
      // 将 0 存储到 A[0]
      Store::make(a, {0}, 0),
      // 将 0 存储到 A[1]
      Store::make(a, {1}, 0),
      // 创建循环语句，迭代变量为 x，范围是 [0, 10)
      For::make(
          x,
          0,
          10,
          Block::make(
              {
                  // 将 A[0] = (A[0]) + x; 的存储操作添加到块中
                  Store::make(a, {0}, Add::make(Load::make(a, {0}), x)),
                  // 将 A[1] = (A[1]) - x; 的存储操作添加到块中
                  Store::make(a, {1}, Sub::make(Load::make(a, {1}), x))
              })),
  });

  /*
   * A[0] = 0;
   * A[1] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   *   A[1] = (A[1]) - x;
   * }
   */

  // 对 stmt 进行寄存器分配处理
  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * int A_2 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_2 = x + A_2;
   *   A_1 = A_1 - x;
   * }
   * A[1] = A_2;
   * A[0] = A_1;
   */

  // 创建一个输出流对象 oss，将处理后的 stmt 的内容写入其中
  std::ostringstream oss;
  oss << *stmt;

  // 验证输出流 oss 的内容是否符合指定的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: int A_2 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK:   A_2 =
# CHECK: A[1] = A_2
# CHECK: A[0] = A_1;)IR";

  // 使用 FileCheck 工具验证输出流 oss 的内容是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 将有效访问寄存器化处理，同时跳过无效的替换。
TEST(Registerizer, RegisterizerVariableLoad) {
  // 创建一个名为 "A" 的缓冲区，大小为 {1}，元素类型为整数
  BufHandle a("A", {1}, kInt);
  // 创建一个名为 "B" 的缓冲区，大小为 {10}，元素类型为整数
  BufHandle b("B", {10}, kInt);
  // 创建一个名为 "x" 的变量，类型为整数
  VarHandle x("x", kInt);
  // 创建一个名为 "x" 的变量（重复定义），类型为整数
  VarHandle x2("x", kInt);
  // 创建一个语句块，包含三个语句：
  // 1. 将 a[0] 设为 0
  // 2. 对变量 x 进行循环，范围是 0 到 10，每次循环将 b[x] 设为 x 的值
  // 3. 对变量 x2 进行循环，范围是 0 到 10，每次循环将 a[0] 设为 a[0] 加上 b[x2] 的值
  StmtPtr stmt = Block::make(
      {Store::make(a, {0}, 0),
       For::make(x, 0, 10, Store::make(b, {x}, x)),
       For::make(
           x2,
           0,
           10,
           Block::make({Store::make(
               a, {0}, Add::make(Load::make(a, {0}), Load::make(b, {x2})))}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = x;
   * }
   * for (int x_1 = 0; x_1 < 10; x_1++) {
   *   A[0] = (A[0]) + (B[x_1]);
   * }
   */

  // 调用 registerize 函数对语句进行寄存器分配优化
  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = x;
   * }
   * for (int x_1 = 0; x_1 < 10; x_1++) {
   *   A_1 = A_1 + (B[x_1]);
   * }
   * A[0] = A_1;
   */

  // 将 stmt 转换为字符串形式，存入 oss
  std::ostringstream oss;
  oss << *stmt;

  // 准备用于验证输出结果的模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   B[x] = x
# CHECK: for (int x_1 = 0; x_1 < 10; x_1++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0] = A_1;)IR";

  // 使用 FileCheck 运行验证模式，检查输出是否符合预期
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 可以对变量访问进行寄存器分配，只要变量不改变。
TEST(Registerizer, RegisterizerSymbolicIndices) {
  // 创建名为 "i" 的变量，类型为整数
  VarHandle i("i", kInt);
  // 创建名为 "N" 的变量，类型为整数
  VarHandle N("N", kInt);
  // 创建名为 "A" 的缓冲区，大小为 {N}，元素类型为整数
  BufHandle a("A", {N}, kInt);
  // 创建名为 "x" 的变量，类型为整数
  VarHandle x("x", kInt);
  // 创建一个语句块，包含两个语句：
  // 1. 将 a[i] 设为 0
  // 2. 对变量 x 进行循环，范围是 0 到 10，每次循环将 a[i] 设为 a[i] 加上 x 的值
  StmtPtr stmt = Block::make(
      {Store::make(a, {i}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {i}, Add::make(Load::make(a, {i}), x))}))});

  /*
   * A[i] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[i] = (A[i]) + x;
   * }
   */

  // 调用 registerize 函数对语句进行寄存器分配优化
  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[i] = A_1;
   */

  // 将 stmt 转换为字符串形式，存入 oss
  std::ostringstream oss;
  oss << *stmt;

  // 准备用于验证输出结果的模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[i] = A_1;)IR";

  // 使用 FileCheck 运行验证模式，检查输出是否符合预期
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 可以对依赖于多个循环变量的访问进行寄存器分配。
TEST(Registerizer, RegisterizerMultiLoop) {
  // 创建名为 a 的缓冲区，大小为 {1}，类型为 kInt
  BufHandle a("A", {1}, kInt);
  // 创建名为 x 的变量，类型为 kInt
  VarHandle x("x", kInt);
  // 创建名为 y 的变量，类型为 kInt
  VarHandle y("y", kInt);
  // 创建一个语句指针 stmt，表示一个代码块
  StmtPtr stmt = Block::make(
      {Store::make(a, {0}, 0), // 在缓冲区 a 的索引 {0} 处存储值 0
       For::make(
           x,
           0,
           10,
           For::make(
               y,
               0,
               10,
               Block::make({Store::make(
                   a,
                   {0},
                   Mul::make(Add::make(Load::make(a, {0}), x), y))})))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     A[0] = x * y + (A[0]) * y;
   *   }
   * }
   */

  // 对 stmt 进行寄存器分析和替换处理
  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     A_1 = x * y + y * A_1;
   *   }
   * }
   * A[0] = A_1;
   */

  // 创建一个字符串流 oss，并将 stmt 的内容输出到 oss 中
  std::ostringstream oss;
  oss << *stmt;

  // 定义 IR 格式的字符串，用于验证输出结果
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   for (int y = 0; y < 10; y++)
# CHECK-NOT: A[
# CHECK:     A_1 =
# CHECK: A[0] = A_1;)IR";

  // 使用 Torch 的 FileCheck 工具验证输出结果
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize correctly if scalars already exist in the program.
TEST(Registerizer, RegisterizerRepeated) {
  // 创建名为 a 的缓冲区，大小为 {2}，类型为 kInt
  BufHandle a("A", {2}, kInt);
  // 创建名为 x 的变量，类型为 kInt
  VarHandle x("x", kInt);
  // 创建一个语句指针 stmt，表示一个代码块
  StmtPtr stmt = Block::make({
      Store::make(a, {0}, 0), // 在缓冲区 a 的索引 {0} 处存储值 0
      Store::make(a, {1}, 0), // 在缓冲区 a 的索引 {1} 处存储值 0
      For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(a, {0}, Add::make(Load::make(a, {0}), x)), // 更新缓冲区 a 的索引 {0} 处的值
               Store::make(a, {1}, Sub::make(Load::make(a, {1}), x))})), // 更新缓冲区 a 的索引 {1} 处的值
  });

  // 手动进行寄存器分析以确保只替换单个目标
  {
    registerizer::RegisterizerAnalysis analysis;
    stmt->accept(&analysis);
    auto candidates = analysis.getCandidates();
    ASSERT_EQ(candidates.size(), 2);

    candidates.pop_back();
    registerizer::RegisterizerReplacer replacer(candidates);
    stmt = stmt->accept_mutator(&replacer);
  }

  // 重新分析并替换第二个目标
  {
    registerizer::RegisterizerAnalysis analysis;
    stmt->accept(&analysis);
    auto candidates = analysis.getCandidates();
    ASSERT_EQ(candidates.size(), 1);

    registerizer::RegisterizerReplacer replacer(candidates);
    stmt = stmt->accept_mutator(&replacer);
  }

  // 创建一个字符串流 oss，并将 stmt 的内容输出到 oss 中
  std::ostringstream oss;
  oss << *stmt;

  // 定义 IR 格式的字符串，用于验证输出结果
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: int A_1_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK:   A_1_1 =
# CHECK: A[1] = A_1_1;
# CHECK: A[0] = A_1;)IR";

  // 使用 Torch 的 FileCheck 工具验证输出结果
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize the load of A.
TEST(Registerizer, RegisterizerNoLoads) {
  // 创建一个名为 "A" 的缓冲区，大小为 1，类型为整数
  BufHandle a("A", {1}, kInt);
  // 创建一个名为 "x" 的变量，类型为整数
  VarHandle x("x", kInt);
  // 构造一个语句块，包含赋值和 for 循环
  StmtPtr stmt = Block::make(
      {Store::make(a, {0}, 0),  // 将值 0 存储到缓冲区 A 的索引 0 处
       For::make(
           x, 0, 10,  // 循环变量 x 从 0 到 9
           Block::make({Store::make(a, {0}, Add::make(x, 1))}))});  // 循环体内将 x + 1 存储到缓冲区 A 的索引 0 处

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = x + 1;
   * }
   */

  // 将语句块传入 registerize 函数进行注册优化
  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + 1;
   * }
   * A[0] = A_1;
   */

  // 创建一个输出流 oss，将 stmt 的内容写入其中
  std::ostringstream oss;
  oss << *stmt;

  // 定义用于验证的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0] = A_1;)IR";

  // 使用 torch::jit::testing::FileCheck() 运行验证模式，验证 oss 的输出结果
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize the load of A but not the store of B.
TEST(Registerizer, RegisterizerNoRepeatedStores) {
  // 创建缓冲区 A 和 B，分别大小为 1 和 10，类型均为整数
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {10}, kInt);
  // 创建变量 x，类型为整数
  VarHandle x("x", kInt);
  // 构造一个语句块，包含赋值和 for 循环
  StmtPtr stmt = Block::make(
      {Store::make(a, {0}, 0),  // 将值 0 存储到缓冲区 A 的索引 0 处
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(b, {x}, Add::make(Load::make(a, {0}), x))}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = (A[0]) + x;
   * }
   */

  // 将语句块传入 registerize 函数进行注册优化
  stmt = registerize(stmt);

  // TODO: its unnecessary to reorder the initializer of A[0], but it's not
  // actually worse so lets not worry for now.

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = x + A_1;
   * }
   * A[0] = A_1;
   */

  // 创建一个输出流 oss，将 stmt 的内容写入其中
  std::ostringstream oss;
  oss << *stmt;

  // 定义用于验证的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A_
# CHECK:   B[x] =
# CHECK: A[0] = A_1;)IR";

  // 使用 torch::jit::testing::FileCheck() 运行验证模式，验证 oss 的输出结果
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Won't registerize if there are multiple accesses which may overlap.
TEST(Registerizer, RegisterizerMultiVarOverlap) {
  // 创建缓冲区 A，大小为 2，类型为整数
  BufHandle a("A", {2}, kInt);
  // 创建变量 x，类型为整数
  VarHandle x("x", kInt);
  // 构造一个语句块，包含多次存储和 for 循环
  StmtPtr stmt = Block::make({
      Store::make(a, {0}, 0),  // 将值 0 存储到缓冲区 A 的索引 0 处
      Store::make(a, {1}, 0),  // 将值 0 存储到缓冲区 A 的索引 1 处
      For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(a, {x}, Add::make(Load::make(a, {0}), x)),  // 将 A[0] + x 存储到 A[x] 处
               Store::make(a, {x + 1}, Sub::make(Load::make(a, {1}), x))}))  // 将 A[1] - x 存储到 A[x + 1] 处
  });
  // 使用 IRSimplifier::simplify 对语句块进行简化处理
  stmt = IRSimplifier::simplify(stmt);

  // 创建一个输出流 before，将 stmt 的内容写入其中
  std::ostringstream before;
  before << *stmt;

  // 尝试对语句块进行注册优化，但不会有任何改变
  stmt = registerize(stmt);

  // 创建一个输出流 after，将 stmt 的内容写入其中
  std::ostringstream after;
  after << *stmt;

  // 断言 before 和 after 输出结果相同
  ASSERT_EQ(before.str(), after.str());
}
TEST(Registerizer, RegisterizerAllocs) {
  BufHandle a("A", {2}, kInt);  // 创建一个名为 "A" 的缓冲区，大小为 {2}，数据类型为 kInt
  BufHandle c("C", {1}, kInt);  // 创建一个名为 "C" 的缓冲区，大小为 {1}，数据类型为 kInt
  VarHandle x("x", kInt);  // 创建一个名为 "x" 的变量，数据类型为 kInt

  // 创建一个名为 "B" 的缓冲区，大小为 {C[0]}，数据类型为 kInt
  BufHandle b("B", {Load::make(c, {0})}, kInt);

  // 创建一个语句块 stmt，包含以下操作序列
  StmtPtr stmt = Block::make(
      {Allocate::make(b),  // 分配缓冲区 B
       Store::make(a, {0}, Load::make(c, {0})),  // 将 C[0] 的值存储到 A[0]
       Store::make(b, {0}, 0),  // 将 0 存储到 B[0]
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(b, {0}, Add::make(Load::make(b, {0}), x)),  // 更新 B[0] 的值为 B[0] + x
                Store::make(a, {0}, Load::make(c, {0}))})),  // 将 C[0] 的值存储到 A[0]
       Free::make(b)});  // 释放缓冲区 B

  /*
   * Allocate(B, int, {C[0]});
   * A[0] = C[0];
   * B[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[0] = (B[0]) + x;
   *   A[0] = C[0];
   * }
   * Free(B);
   */

  stmt = registerize(stmt);  // 对 stmt 进行寄存器分配优化

  /*
   * int C_1 = C[0];
   * Allocate(B, int, {C_});
   * int A_1 = C_1;
   * int B_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   B_1 = B_1 + x;
   *   A_1 = C_1;
   * }
   * B[0] = B_1;
   * A[0] = A_1;
   * Free(B);
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int C_1 = C[0];
# CHECK: Allocate(B
# CHECK: int A_1 = C_1;
# CHECK: int B_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   B_1 =
# CHECK:   A_1 = C_
# CHECK: B[0] = B_1;
# CHECK: A[0] = A_1;
# CHECK: Free(B)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Registerizer, RegisterizerNoInitializer) {
  BufHandle a("A", {1}, kInt);  // 创建一个名为 "A" 的缓冲区，大小为 {1}，数据类型为 kInt
  VarHandle x("x", kInt);  // 创建一个名为 "x" 的变量，数据类型为 kInt
  // 创建一个语句块 stmt，包含以下操作序列
  StmtPtr stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make({Store::make(a, {0}, Add::make(Load::make(a, {0}), x))}))});

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  stmt = registerize(stmt);  // 对 stmt 进行寄存器分配优化

  /*
   * int A_1 = A[0];
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Registerizer, RegisterizerNoInitializerLoopVar) {
  BufHandle a("A", {1}, kInt);  // 创建一个名为 "A" 的缓冲区，大小为 {1}，数据类型为 kInt
  VarHandle x("x", kInt);  // 创建一个名为 "x" 的变量，数据类型为 kInt
  // 创建一个语句块 stmt，包含以下操作序列
  StmtPtr stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make({Store::make(a, {x}, Add::make(Load::make(a, {x}), x))}))});
  stmt = IRSimplifier::simplify(stmt);  // 简化语句块

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[x]) + x;
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);  // 对 stmt 进行寄存器分配优化

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}
TEST(Registerizer, RegisterizerLoadThenStore) {
  // 定义名为 "A" 的缓冲区，包含一个元素，元素类型为整数
  BufHandle a("A", {1}, kInt);
  // 定义名为 "B" 的缓冲区，包含一个元素，元素类型为整数
  BufHandle b("B", {1}, kInt);
  // 定义名为 "x" 的变量，类型为整数
  VarHandle x("x", kInt);
  // 构造一个语句块，包含一个 for 循环
  StmtPtr stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make(
          {
              // 在缓冲区 B 的索引 0 处存储表达式 (A[0]) + x
              Store::make(b, {0}, Add::make(Load::make(a, {0}), x)),
              // 在缓冲区 A 的索引 0 处存储缓冲区 B 的索引 0 处的值
              Store::make(a, {0}, Load::make(b, {0}))
          }))});

  /*
   * for (int x = 0; x < 10; x++) {
   *   B[0] = (A[0]) + x;
   *   A[0] = B[0];
   * }
   */

  // 对构造的语句进行寄存器化处理
  stmt = registerize(stmt);

  /*
   * int A_1 = A[0];
   * int B_1 = B[0];
   * for (int x = 0; x < 10; x++) {
   *   B_1 = x + A_1;
   *   A_1 = B_1;
   * }
   * B[0] = B_1;
   * A[0] = A_1;
   */

  // 创建一个字符串流对象 oss，并将寄存器化后的语句输出到 oss 中
  std::ostringstream oss;
  oss << *stmt;

  // 定义字符串变量 verification_pattern，用于验证输出是否符合预期格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: int B_1 = B[0];
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: B[
# CHECK:   B_1 =
# CHECK-NOT: A[
# CHECK:   A_1 = B_
# CHECK: B[0] = B_
# CHECK: A[0] = A_1;)IR";

  // 使用 FileCheck 类来运行验证模式，检查输出是否符合模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Registerizer, RegisterizerParallelized) {
  // 定义名为 "A" 的缓冲区，包含一个元素，元素类型为整数
  BufHandle a("A", {1}, kInt);
  // 定义名为 "x" 的变量，类型为整数
  VarHandle x("x", kInt);
  // 创建循环选项对象，并设置 GPU 块索引为 0
  LoopOptions loopOpts;
  loopOpts.set_gpu_block_index(0);
  // 构造一个语句块，包含一个 for 循环和一个存储操作
  StmtPtr stmt = Block::make(
      {
          // 在缓冲区 A 的索引 0 处存储值 0
          Store::make(a, {0}, 0),
          For::make(
              x,
              0,
              10,
              Block::make({Store::make(a, {0}, Add::make(Load::make(a, {0}), x))}),
              loopOpts)
      });

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  // 断言调用寄存器化函数会抛出特定异常信息
  ASSERT_THROWS_WITH(
      registerize(stmt),
      "Registerization must occur after parallelism flattening");
}

// Should be able to registerize this since the scalar would exist before the
// branch.
TEST(Registerizer, RegisterizerConditionAfter) {
  // 定义名为 "A" 的缓冲区，包含五个元素，元素类型为整数
  BufHandle a("A", {5}, kInt);
  // 定义名为 "B" 的缓冲区，包含五个元素，元素类型为整数
  BufHandle b("B", {5}, kInt);
  // 定义名为 "C" 的缓冲区，包含五个元素，元素类型为整数
  BufHandle c("C", {5}, kInt);
  // 定义名为 "x" 的变量，类型为整数
  VarHandle x("x", kInt);

  // 构造一个语句块，包含一个存储操作和条件语句
  StmtPtr stmt = Block::make(
      {
          // 在缓冲区 A 的索引 x 处存储缓冲区 B 的索引 x 处的值
          Store::make(a, {x}, Load::make(b, {x})),
          // 在缓冲区 C 的索引 x 处存储缓冲区 A 的索引 x 处的值
          Store::make(c, {x}, Load::make(a, {x})),
          // 创建条件语句，如果 x < 5，则在缓冲区 A 的索引 x 处存储表达式 (A[x]) + 1
          Cond::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kLT),
              Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
              nullptr)
      });

  /*
   * A[x] = B[x];
   * C[x] = A[x];
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   */

  // 对构造的语句进行寄存器化处理
  stmt = registerize(stmt);

  /*
   * int A_1 = B[x];
   * C[x] = A_1;
   * if (x<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * A[x] = A_1;
   */

  // 创建一个字符串流对象 oss，并将寄存器化后的语句输出到 oss 中
  std::ostringstream oss;
  oss << *stmt;

  // 定义字符串变量 verification_pattern，用于验证输出是否符合预期格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = B[x];
# CHECK: C[x] = A_1;
# CHECK: if (
# CHECK:   A_1 = A_1 + 1;
# CHECK: A[x] = A_1;)IR";

  // 使用 FileCheck 类来运行验证模式，检查输出是否符合模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Registerizer, RegisterizerConditionBefore) {
  // 创建名为 A、B、C 的缓冲区，每个缓冲区包含五个整数，数据类型为整型
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  // 创建名为 x 的变量，数据类型为整型
  VarHandle x("x", kInt);

  // 创建语句块 stmt，包含三个语句
  StmtPtr stmt = Block::make(
      // 条件语句：如果 x 小于 5，则执行 A[x] = (A[x]) + 1，否则执行空语句
      {Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           nullptr),
       // A[x] = B[x]
       Store::make(a, {x}, Load::make(b, {x})),
       // C[x] = A[x]
       Store::make(c, {x}, Load::make(a, {x}))});

  /*
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   * A[x] = B[x];
   * C[x] = A[x];
   */

  // 对语句块 stmt 进行寄存器分配
  stmt = registerize(stmt);

  /*
   * int A_1 = A[x];
   * if (x<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * A_1 = B[x];
   * C[x] = A_1;
   * A[x] = A_1;
   */

  // 将 stmt 转换为字符串形式并输出到 oss 流中
  std::ostringstream oss;
  oss << *stmt;

  // 定义 IR 格式的验证模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: if (
# CHECK:   A_1 = A_1 + 1;
# CHECK: }
# CHECK: A_1 = B[x];
# CHECK: C[x] = A_1;
# CHECK: A[x] = A_1;)IR";

  // 使用 FileCheck 运行验证模式，验证 oss 中的字符串是否符合模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Should be able to registerize this as the combination of the two above rules.
TEST(Registerizer, RegisterizerConditionInside) {
  // 创建名为 A、B、C 的缓冲区，每个缓冲区包含五个整数，数据类型为整型
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  // 创建名为 x 的变量，数据类型为整型
  VarHandle x("x", kInt);

  // 创建语句块 stmt，包含五个语句
  StmtPtr stmt = Block::make(
      // A[x] = B[x]
      {Store::make(a, {x}, Load::make(b, {x})),
       // C[x] = A[x]
       Store::make(c, {x}, Load::make(a, {x})),
       // 条件语句：如果 x 小于 5，则执行 A[x] = (A[x]) + 1，否则执行空语句
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           nullptr),
       // B[x] = A[x]
       Store::make(b, {x}, Load::make(a, {x})),
       // A[x] = C[x]
       Store::make(a, {x}, Load::make(c, {x}))});

  /*
   * A[x] = B[x];
   * C[x] = A[x];
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   * B[x] = A[x];
   * A[x] = C[x];
   */

  // 对语句块 stmt 进行寄存器分配
  stmt = registerize(stmt);

  /*
   * int A_1 = B[x];
   * C[x] = A_1;
   * if (x<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * B[x] = A_1;
   * A_1 = C[x];
   * A[x] = A_1;
   */

  // 将 stmt 转换为字符串形式并输出到 oss 流中
  std::ostringstream oss;
  oss << *stmt;

  // 定义 IR 格式的验证模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = B[x];
# CHECK: C[x] = A_1;
# CHECK: if (
# CHECK:   A_1 = A_1 + 1;
# CHECK: }
# CHECK: B[x] = A_1;
# CHECK: A_1 = C[x];
# CHECK: A[x] = A_1;)IR";

  // 使用 FileCheck 运行验证模式，验证 oss 中的字符串是否符合模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Registerizer, RegisterizerConditionInsideOverlap1) {
  BufHandle a("A", {5}, kInt);  // 创建名为 "A" 的缓冲区，大小为 5，类型为整数
  BufHandle b("B", {5}, kInt);  // 创建名为 "B" 的缓冲区，大小为 5，类型为整数
  BufHandle c("C", {5}, kInt);  // 创建名为 "C" 的缓冲区，大小为 5，类型为整数
  VarHandle x("x", kInt);       // 创建名为 "x" 的变量，类型为整数
  VarHandle y("y", kInt);       // 创建名为 "y" 的变量，类型为整数

  StmtPtr stmt = Block::make(
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      {Store::make(a, {x}, Load::make(b, {x})),  // 将 B[x] 的值存入 A[x]
       Store::make(c, {x}, Load::make(a, {x})),  // 将 A[x] 的值存入 C[x]
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),  // 如果 x < 5
           Block::make({
               Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),  // 执行 A[x] = A[x] + 1
               Store::make(a, {0}, 3),  // 执行 A[0] = 3
               Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),  // 执行 A[x] = A[x] + 1
           }),
           nullptr),  // 否则条件为空
       Store::make(b, {x}, Load::make(a, {x})),  // 将 A[x] 的值存入 B[x]
       Store::make(a, {x}, Load::make(c, {x}))});  // 将 C[x] 的值存入 A[x]

  /*
   * A[x] = B[x];
   * C[x] = A[x];
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   *   A[0] = 3;
   *   A[x] = (A[x]) + 1;
   * }
   * B[x] = A[x];
   * A[x] = C[x];
   */

  // The A[0] store overlaps, A[x] cutting the region that can be registerized
  // into two groups.
  // Each group has 2 loads and 2 stores however, so we could registerize it,
  // but the first group would need to be finalized inside the condition block,
  // the second would need to be initialized inside the condition block. There's
  // no safe place to put these that's visible to the other uses in the group
  // and so neither registerization is possible.

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);  // 进行寄存器分配优化

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// Same as the above, but the access group before the condition (and after the
// condition) are large enough to be registerized without needing the access
// from the loop. Registerization occurs but does not include any accesses in
// the condition, and the first group must be finalized before the Cond, the
// second initialized after it.
TEST(Registerizer, RegisterizerConditionInsideOverlap2) {
  BufHandle a("A", {5}, kInt);  // 创建名为"A"的缓冲区，大小为5，数据类型为整数
  BufHandle b("B", {5}, kInt);  // 创建名为"B"的缓冲区，大小为5，数据类型为整数
  BufHandle c("C", {5}, kInt);  // 创建名为"C"的缓冲区，大小为5，数据类型为整数
  VarHandle x("x", kInt);       // 创建名为"x"的变量，数据类型为整数
  VarHandle y("y", kInt);       // 创建名为"y"的变量，数据类型为整数

  StmtPtr stmt = Block::make(
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      {Store::make(a, {x}, Load::make(b, {x})),  // 将B[x]的值存入A[x]
       Store::make(a, {x}, Load::make(b, {x + 1})),  // 将B[x+1]的值存入A[x]
       Store::make(c, {x}, Load::make(a, {x})),  // 将A[x]的值存入C[x]
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),  // 创建条件语句：如果x < 5
           Block::make({
               Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),  // 将A[x]加1后存回A[x]
               Store::make(a, {0}, 3),  // 将值3存入A[0]
               Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),  // 将A[x]再次加1后存回A[x]
           }),
           nullptr),  // 条件不满足时执行空块
       Store::make(b, {x}, Load::make(a, {x})),  // 将A[x]的值存入B[x]
       Store::make(b, {x + 1}, Load::make(a, {x})),  // 将A[x]的值存入B[x+1]
       Store::make(a, {x}, Load::make(c, {x}))});  // 将C[x]的值存入A[x]

  /*
   * A[x] = B[x];
   * A[x] = B[x + 1];
   * C[x] = A[x];
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   *   A[0] = 3;
   *   A[x] = (A[x]) + 1;
   * }
   * B[x] = A[x];
   * B[x + 1] = A[x];
   * A[x] = C[x];
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = B[x];              // 创建名为A_1的变量，将B[x]的值赋给A_1（初始化）
   * A_1 = B[x + 1];              // 将B[x+1]的值赋给A_1
   * C[x] = A_1;                  // 将A_1的值赋给C[x]
   * A[x] = A_1;                  // 将A_1的值赋给A[x]（最终化）
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;          // 如果x<5，则将A[x]加1
   *   A[0] = 3;                   // 将3赋给A[0]
   *   A[x] = (A[x]) + 1;          // 再次将A[x]加1
   * }
   * int A_2 = A[x];              // 创建名为A_2的变量，将A[x]的值赋给A_2（初始化）
   * B[x] = A_2;                  // 将A_2的值赋给B[x]
   * B[x + 1] = A_2;              // 将A_2的值赋给B[x+1]
   * A_2 = C[x];                  // 将C[x]的值赋给A_2
   * A[x] = A_2;                  // 将A_2的值赋给A[x]（最终化）
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = B[x];
# CHECK: A_1 = B[x + 1];
# CHECK: C[x] = A_1;
# CHECK: A[x] = A_1;
# CHECK: if (
# CHECK-NOT:   A_1 = A_1 + 1;
# CHECK:   A[x] = (A[x]
# CHECK:   A[0] =
# CHECK:   A[x] = (A[x]
# CHECK: }
# CHECK: int A_2 = A[x];
# CHECK: B[x] = A_2;
# CHECK: B[x + 1] = A_2;
# CHECK: A_2 = C[x];
# CHECK: A[x] = A_2;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 当访问在条件块内时，它们对更广泛的程序不可见，因为我们不知道是否会执行该分支，如果不执行，则其中的访问无需有效（例如索引的大小检查）。在这种情况下，访问无法进行寄存器化。
TEST(Registerizer, RegisterizerConditionHidden) {
  // 创建名为 a, b, c 的缓冲区句柄，每个大小为 5，类型为整数
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  // 创建名为 x 的变量句柄，类型为整数
  VarHandle x("x", kInt);

  // 创建一个语句块 stmt，包含两个条件语句
  StmtPtr stmt = Block::make(
      {
          // 第一个条件语句：如果 x 小于 5，则执行 A[x] = A[x] + 1
          Cond::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kLT),
              Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
              nullptr),
          // 第二个条件语句：如果 x 大于 5，则执行 A[x] = A[x] + 1
          Cond::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kGT),
              Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
              nullptr)
      });

  /*
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   * if (x>5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   */

  // 将 stmt 的字符串表示输出到 before 流中
  std::ostringstream before;
  before << *stmt;

  // 对 stmt 进行寄存器分配处理
  stmt = registerize(stmt);

  // 将处理后的 stmt 的字符串表示输出到 after 流中
  std::ostringstream after;
  after << *stmt;

  // 断言处理前后的字符串表示相同
  ASSERT_EQ(before.str(), after.str());
}

TEST(Registerizer, RegisterizerConditionUnhidden) {
  // 创建名为 a, b, c 的缓冲区句柄，每个大小为 5，类型为整数
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  // 创建名为 x 的变量句柄，类型为整数
  VarHandle x("x", kInt);

  // 创建一个语句块 stmt，包含三个语句：两个条件语句和一个非条件语句
  StmtPtr stmt = Block::make(
      {
          // 第一个条件语句：如果 x 小于 5，则执行 A[x] = A[x] + 1
          Cond::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kLT),
              Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
              nullptr),
          // 非条件语句：无条件执行 A[x] = A[x] + 1，用于取消隐藏条件访问
          Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
          // 第二个条件语句：如果 x 大于 5，则执行 A[x] = A[x] + 1
          Cond::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kGT),
              Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
              nullptr)
      });

  /*
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   * A[x] = (A[x]) + 1;            <-- this is doing the unhiding.
   * if (x>5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   */

  // 对 stmt 进行寄存器分配处理
  stmt = registerize(stmt);

  /*
   * int A_1 = A[x];
   * if (x<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * A_1 = A_1 + 1;
   * if (x>5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * A[x] = A_1;
   */

  // 将 stmt 的字符串表示输出到 oss 流中
  std::ostringstream oss;
  oss << *stmt;

  // 验证 oss 流中的字符串符合给定的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: if (x<5
# CHECK:   A_1 = A_1 + 1;
# CHECK: }
# CHECK: A_1 = A_1 + 1;
# CHECK: if (x>5
# CHECK:   A_1 = A_1 + 1;
# CHECK: }
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Registerizer, RegisterizerCondCondition) {
  BufHandle a("A", {5}, kInt);  // 创建名为 "A" 的缓冲区，大小为 {5}，类型为整数
  BufHandle b("B", {5}, kInt);  // 创建名为 "B" 的缓冲区，大小为 {5}，类型为整数
  BufHandle c("C", {5}, kInt);  // 创建名为 "C" 的缓冲区，大小为 {5}，类型为整数
  VarHandle x("x", kInt);       // 创建名为 "x" 的变量，类型为整数

  // 创建语句块，包含三个操作：A[x] = B[x]; C[x] = A[x]; if ((A[x])<5 ? 1 : 0) { C[x] = (C[x]) + 1; }
  StmtPtr stmt = Block::make(
      {Store::make(a, {x}, Load::make(b, {x})),  // A[x] = B[x];
       Store::make(c, {x}, Load::make(a, {x})),  // C[x] = A[x];
       Cond::make(
           CompareSelect::make(
               Load::make(a, {x}), 5, CompareSelectOperation::kLT),  // 条件判断：(A[x] < 5)
           Store::make(c, {x}, Add::make(Load::make(c, {x}), 1)),   // 如果条件成立：C[x] = C[x] + 1;
           nullptr)});  // 如果条件不成立，为空

  /*
   * A[x] = B[x];
   * C[x] = A[x];
   * if ((A[x])<5 ? 1 : 0) {
   *   C[x] = (C[x]) + 1;
   * }
   */

  stmt = registerize(stmt);  // 运行 registerize 函数处理语句块

  /*
   * int A_1 = B[x];
   * int C_1 = A_1;
   * if (A_1<5 ? 1 : 0) {
   *   C_1 = C_1 + 1;
   * }
   * C[x] = C_1;
   */

  std::ostringstream oss;  // 创建字符串输出流对象 oss
  oss << *stmt;  // 将处理后的语句块输出到 oss 中

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = B[x];
# CHECK: int C_1 = A_1;
# CHECK: if (A_1<5
# CHECK:   C_1 = C_1 + 1;
# CHECK: C[x] = C_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用 FileCheck 验证输出是否符合预期模式
}

// Appearing in the condition of a Cond makes it visible to the enclosing scope,
// and so we can registerize internal usages.
TEST(Registerizer, RegisterizerCondConditionUnhidden) {
  BufHandle a("A", {5}, kInt);  // 创建名为 "A" 的缓冲区，大小为 {5}，类型为整数
  BufHandle b("B", {5}, kInt);  // 创建名为 "B" 的缓冲区，大小为 {5}，类型为整数
  BufHandle c("C", {5}, kInt);  // 创建名为 "C" 的缓冲区，大小为 {5}，类型为整数
  VarHandle x("x", kInt);       // 创建名为 "x" 的变量，类型为整数

  // 创建语句块，包含条件判断：if ((A[x]) < 5) { A[x] = (A[x]) + 1; } else { A[x] = (A[x]) + 10; }
  StmtPtr stmt = Block::make({Cond::make(
      CompareSelect::make(Load::make(a, {x}), 5, CompareSelectOperation::kLT),
      Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
      Store::make(a, {x}, Add::make(Load::make(a, {x}), 10)))});

  /*
   * if ((A[x])<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * } else {
   *   A[x] = (A[x]) + 10;
   * }
   */

  stmt = registerize(stmt);  // 运行 registerize 函数处理语句块

  /*
   * int A_1 = A[x];
   * if (A_1<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * } else {
   *   A_1 = A_1 + 10;
   * }
   * A[x] = A_1;
   */

  std::ostringstream oss;  // 创建字符串输出流对象 oss
  oss << *stmt;  // 将处理后的语句块输出到 oss 中

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: if (A_1<5
# CHECK:   A_1 = A_1 + 1;
# CHECK: } else {
# CHECK:   A_1 = A_1 + 10;
# CHECK: }
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用 FileCheck 验证输出是否符合预期模式
}
TEST(Registerizer, RegisterizerIfThenElseHidden) {
  BufHandle a("A", {5}, kInt);  // 创建名为 "A" 的缓冲区，包含5个元素，类型为整数
  BufHandle b("B", {5}, kInt);  // 创建名为 "B" 的缓冲区，包含5个元素，类型为整数
  BufHandle c("C", {5}, kInt);  // 创建名为 "C" 的缓冲区，包含5个元素，类型为整数
  VarHandle x("x", kInt);       // 创建名为 "x" 的变量，类型为整数
  VarHandle y("y", kInt);       // 创建名为 "y" 的变量，类型为整数

  StmtPtr stmt = Block::make(
      {Store::make(
           b,
           {y},
           IfThenElse::make(
               CompareSelect::make(x, 5, CompareSelectOperation::kLT),  // 如果 x < 5
               Add::make(Load::make(a, {x}), 1),                         // 则 B[y] = A[x] + 1
               Add::make(Load::make(a, {x + 1}), 2)))});                // 否则 B[y] = A[x + 1] + 2

  /*
   * B[y] = IfThenElse(x<5 ? 1 : 0, (A[x]) + 1, (A[x + 1]) + 2);
   * B[y + 1] = IfThenElse(x<5 ? 1 : 0, (A[x]) + 1, (A[x + 1]) + 2);
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);  // 对语句进行寄存器化处理

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());  // 断言寄存器化前后语句的字符串表示应保持一致
}

// Conditional unhiding also works for IfThenElse exprs.
TEST(Registerizer, RegisterizerIfThenElseUnhidden) {
  BufHandle a("A", {5}, kInt);  // 创建名为 "A" 的缓冲区，包含5个元素，类型为整数
  BufHandle b("B", {5}, kInt);  // 创建名为 "B" 的缓冲区，包含5个元素，类型为整数
  BufHandle c("C", {5}, kInt);  // 创建名为 "C" 的缓冲区，包含5个元素，类型为整数
  VarHandle x("x", kInt);       // 创建名为 "x" 的变量，类型为整数
  VarHandle y("y", kInt);       // 创建名为 "y" 的变量，类型为整数

  StmtPtr stmt = Block::make({
      Store::make(a, {x}, 0),  // 将值0存储到 A[x]
      Store::make(
          b,
          {y},
          IfThenElse::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kLT),  // 如果 x < 5
              Add::make(Load::make(a, {x}), 1),                         // 则 B[y] = A[x] + 1
              Add::make(Load::make(a, {x + 1}), 2))),                  // 否则 B[y] = A[x + 1] + 2
      Store::make(
          b,
          {y + 1},
          IfThenElse::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kLT),  // 如果 x < 5
              Add::make(Load::make(a, {x}), 1),                         // 则 B[y + 1] = A[x] + 1
              Add::make(Load::make(a, {x + 1}), 2))),                  // 否则 B[y + 1] = A[x + 1] + 2
  });

  /*
   * A[x] = 0;
   * B[y] = IfThenElse(x<5 ? 1 : 0, (A[x]) + 1, (A[x + 1]) + 2);
   * B[y + 1] = IfThenElse(x<5 ? 1 : 0, (A[x]) + 1, (A[x + 1]) + 2);
   */

  stmt = registerize(stmt);  // 对语句进行寄存器化处理

  /*
   * int A_1 = 0;
   * B[y] = IfThenElse(x<5 ? 1 : 0, A_1 + 1, (A[x + 1]) + 2);
   * B[y + 1] = IfThenElse(x<5 ? 1 : 0, A_1 + 1, (A[x + 1]) + 2);
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: B[y] = IfThenElse(x<5 ? 1 : 0, A_1 + 1, (A[x + 1]) + 2);
# CHECK: B[y + 1] = IfThenElse(x<5 ? 1 : 0, A_1 + 1, (A[x + 1]) + 2);
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用FileCheck验证输出字符串符合预期模式
}

// Nested IfThenElse exprs can't promote to higher level scopes.
TEST(Registerizer, RegisterizerIfThenElseNested) {
  BufHandle a("A", {5}, kInt);  // 创建名为 "A" 的缓冲区，大小为 {5}，类型为整数
  BufHandle b("B", {5}, kInt);  // 创建名为 "B" 的缓冲区，大小为 {5}，类型为整数
  BufHandle c("C", {5}, kInt);  // 创建名为 "C" 的缓冲区，大小为 {5}，类型为整数
  BufHandle d("D", {5}, kInt);  // 创建名为 "D" 的缓冲区，大小为 {5}，类型为整数
  VarHandle x("x", kInt);       // 创建名为 "x" 的变量，类型为整数

  // 构造一个语句块，包含一个存储操作，其值为嵌套的 IfThenElse 语句
  StmtPtr stmt = Block::make({Store::make(
      a,
      {x},
      IfThenElse::make(
          CompareSelect::make(x, 3, CompareSelectOperation::kLT),  // 如果 x < 3
          IfThenElse::make(
              CompareSelect::make(x, 2, CompareSelectOperation::kEQ),  // 如果 x == 2
              Load::make(d, {x}),  // 则加载 D[x]
              Load::make(b, {x})),  // 否则加载 B[x]
          IfThenElse::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kEQ),  // 如果 x == 5
              Load::make(c, {x}),  // 则加载 C[x]
              Load::make(d, {x}))))});  // 否则加载 D[x]

  /*
   * A[x] = IfThenElse(x<3 ? 1 : 0,
   *          IfThenElse(x==2 ? 1 : 0, D[x], B[x]),
   *            IfThenElse(x==5 ? 1 : 0, C[x], D[x]));
   */

  std::ostringstream before;
  before << *stmt;

  // 调用 registerize 函数处理 stmt
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// 无法完全注册一个完全包含在 IfThenElse 分支内的访问，因为它不是 Stmt 且不能持有变量定义。
// 我们需要检查不将初始化器/终结器提升到外部块。
TEST(Registerizer, RegisterizerIfThenElseInternal) {
  // 将这些设置为 float 类型，以防它们被简化为单个访问。
  BufHandle a("A", {5}, kFloat);  // 创建名为 "A" 的缓冲区，大小为 {5}，类型为浮点数
  BufHandle b("B", {5}, kFloat);  // 创建名为 "B" 的缓冲区，大小为 {5}，类型为浮点数
  VarHandle x("x", kInt);         // 创建名为 "x" 的变量，类型为整数

  // 构造一个语句块，包含一个存储操作，其值为 IfThenElse 语句
  StmtPtr stmt = Block::make({Store::make(
      a,
      {x},
      IfThenElse::make(
          CompareSelect::make(x, 3, CompareSelectOperation::kLT),  // 如果 x < 3
          Add::make(Load::make(b, {x}), Load::make(b, {x})),  // 则加法：B[x] + B[x]
          Load::make(b, {x})))});  // 否则加载 B[x]

  /*
   * A[x] = IfThenElse(x<3 ? 1 : 0, (B[x]) + (B[x]), B[x]);
   */

  std::ostringstream before;
  before << *stmt;

  // 不进行任何更改
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());

  // 如果这是一个 Cond 而不是 IfThenElse，则可以注册化 True 分支中对 B[x] 的两次访问。

  // 实际上，让我们验证一下。

  stmt = Block::make({Cond::make(
      CompareSelect::make(x, 3, CompareSelectOperation::kLT),  // 如果 x < 3
      Store::make(a, {x}, Add::make(Load::make(b, {x}), Load::make(b, {x}))),  // 则存储：A[x] = B[x] + B[x]
      Store::make(a, {x}, Load::make(b, {x})))});  // 否则存储：A[x] = B[x]

  /*
   * if (x<3 ? 1 : 0) {
   *   A[x] = (B[x]) + (B[x]);
   * } else {
   *   A[x] = B[x];
   * }
   */

  stmt = registerize(stmt);

  /*
   * if (x<3 ? 1 : 0) {
   *   float B_1 = B[x];
   *   A[x] = B_1 + B_1;
   * } else {
   *   A[x] = B[x];
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: int
# CHECK-NOT: float
# CHECK: if (x<3
# CHECK:   float B_1 =
# CHECK:   A[x] = B_1 + B_1
# CHECK: } else {
# CHECK:   A[x] = B[x]
# CHECK: }
# CHECK-NOT: A[x]
  )IR";
}
// 定义一个名为 RegisterizerIfThenElseCondition 的测试用例函数
TEST(Registerizer, RegisterizerIfThenElseCondition) {
  // 创建四个缓冲区和一个变量，分别表示数组 A、B、C，以及变量 x
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  // 构造一个语句块，包含两个 Store 操作和一个 IfThenElse 条件语句
  StmtPtr stmt = Block::make(
      {Store::make(a, {x}, Load::make(a, {x})),  // A[x] = A[x];
       Store::make(
           a,
           {x},
           IfThenElse::make(
               CompareSelect::make(
                   Load::make(a, {x}), 5, CompareSelectOperation::kLT),
               Load::make(b, {0}),  // A[x] = IfThenElse((A[x])<5 ? 1 : 0, B[0], C[0]);
               Load::make(c, {0})))});

  // 对语句进行寄存器化处理
  stmt = registerize(stmt);

  // 创建一个字符串流对象用于存储寄存器化后的语句
  std::ostringstream oss;
  oss << *stmt;

  // 设置验证模式字符串，用于验证寄存器化后的语句是否符合预期格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: A_1 = IfThenElse(A_1<5 ? 1 : 0, B[0], C[0]);
# CHECK: A[x] = A_1;)IR";

  // 运行文件检查工具来验证输出是否符合预期格式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 定义一个名为 RegisterizerIfThenElseConditionUnhidden 的测试用例函数
TEST(Registerizer, RegisterizerIfThenElseConditionUnhidden) {
  // 创建四个缓冲区和一个变量，分别表示数组 A、B、C，以及变量 x
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  // 构造一个语句块，包含一个 Store 操作和一个 IfThenElse 条件语句
  StmtPtr stmt = Block::make({Store::make(
      b,
      {x},
      IfThenElse::make(
          CompareSelect::make(
              Load::make(a, {x}), 5, CompareSelectOperation::kLT),
          Add::make(Load::make(a, {x}), 1),  // B[x] = IfThenElse((A[x])<5 ? 1 : 0, (A[x]) + 1, (A[x]) + 10);
          Add::make(Load::make(a, {x}), 10)))});

  // 对语句进行寄存器化处理
  stmt = registerize(stmt);

  // 创建一个字符串流对象用于存储寄存器化后的语句
  std::ostringstream oss;
  oss << *stmt;

  // 设置验证模式字符串，用于验证寄存器化后的语句是否符合预期格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: B[x] = IfThenElse(A_1<5 ? 1 : 0, A_1 + 1, A_1 + 10);)IR";

  // 运行文件检查工具来验证输出是否符合预期格式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Registerizer, RegisterizerConditionBranchOnly) {
  // 创建一个名为 "A" 的缓冲区对象，大小为 5，类型为整数
  BufHandle a("A", {5}, kInt);
  // 创建一个名为 "x" 的变量对象，类型为整数
  VarHandle x("x", kInt);
  // 构造一个包含条件分支的语句块，循环控制变量为 x，范围是 0 到 10
  StmtPtr stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make({
          // 创建一个条件语句节点，条件为 x < 5
          Cond::make(
              // 创建一个比较选择节点，比较 x 和 5 的大小关系
              CompareSelect::make(x, 5, CompareSelectOperation::kLT),
              // 如果条件成立，执行的存储操作
              Store::make(
                  a,
                  {x},
                  // 创建一个条件表达式节点，根据 x < 5 的条件选择不同的加法操作
                  IfThenElse::make(
                      CompareSelect::make(x, 5, CompareSelectOperation::kLT),
                      // 如果条件成立，执行 A[x] 的加载并加上 x
                      Add::make(Load::make(a, {x}), x),
                      // 如果条件不成立，执行 A[x - 5] 的加载并加上 x
                      Add::make(Load::make(a, {x - 5}), x))),
              // 如果条件不成立，执行的存储操作
              Store::make(
                  a,
                  {x - 5},
                  // 创建一个条件表达式节点，根据 x < 5 的条件选择不同的加法操作
                  IfThenElse::make(
                      CompareSelect::make(x, 5, CompareSelectOperation::kLT),
                      // 如果条件成立，执行 A[x] 的加载并加上 x
                      Add::make(Load::make(a, {x}), x),
                      // 如果条件不成立，执行 A[x - 5] 的加载并加上 x
                      Add::make(Load::make(a, {x - 5}), x)))),
      }))});
  // 对生成的语句进行简化处理
  stmt = IRSimplifier::simplify(stmt);

  // 创建一个流对象，将简化前的语句输出到流中
  std::ostringstream before;
  before << *stmt;

  /* for (int x = 0; x < 10; x++) {
   *   if (x<5 ? 1 : 0) {
   *     A[x] = IfThenElse(x<5 ? 1 : 0, (A[x]) + x, (A[x - 5]) + x);
   *   } else {
   *     A[x - 5] = IfThenElse(x<5 ? 1 : 0, (A[x]) + x, (A[x - 5]) + x);
   *   }
   * }
   */

  // 将语句进行寄存器分配优化，无改变
  stmt = registerize(stmt);

  // 创建一个流对象，将优化后的语句输出到流中
  std::ostringstream after;
  after << *stmt;

  // 断言优化前后的输出流内容一致
  ASSERT_EQ(before.str(), after.str());
}

// 可以对出现在 Cond 条件分支中的 IfThenElse 进行寄存器分配优化
TEST(Registerizer, RegisterizerCondIfThenElse) {
  // 创建三个缓冲区对象 A, B, C，每个大小为 5，类型为整数
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  // 创建一个名为 "x" 的变量对象，类型为整数
  VarHandle x("x", kInt);

  // 构造一个条件语句节点，条件为 IfThenElse((A[x])<5 ? 1 : 0, A[x], B[x]) == x
  StmtPtr stmt = Block::make({Cond::make(
      // 创建一个比较选择节点，比较 A[x] 和 5 的大小关系
      CompareSelect::make(
          // 创建一个条件表达式节点，根据 (A[x])<5 ? 1 : 0 选择 A[x] 或 B[x]
          IfThenElse::make(
              // 创建一个比较选择节点，比较 A[x] 和 5 的大小关系
              CompareSelect::make(
                  Load::make(a, {x}), 5, CompareSelectOperation::kLT),
              Load::make(a, {x}),
              Load::make(b, {x})),
          x,
          CompareSelectOperation::kEQ),
      // 如果条件成立，执行 C[x] = (C[x]) + 1 的存储操作
      Store::make(c, {x}, Add::make(Load::make(c, {x}), 1)),
      nullptr)});

  /*
   * if ((IfThenElse((A[x])<5 ? 1 : 0, A[x], B[x]))==x ? 1 : 0) {
   *   C[x] = (C[x]) + 1;
   * }
   */

  // 对生成的语句进行寄存器分配优化
  stmt = registerize(stmt);

  // 创建一个流对象，将优化后的语句输出到流中
  std::ostringstream oss;
  oss << *stmt;

  // 创建用于验证输出格式的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: if ((IfThenElse(A_1<5 ? 1 : 0, A_1, B[x]
# CHECK:   C[x] = (C[x]) + 1;)IR";

  // 使用 FileCheck 进行验证输出格式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 可以对在存储操作的右手边出现的条件访问进行寄存器分配优化，并将其提升出循环外
TEST(Registerizer, RegisterizerIfThenElseLoop) {
  // 创建名为"A"的缓冲区，大小为5，类型为整型
  BufHandle a("A", {5}, kInt);
  // 创建名为"B"的缓冲区，大小为5，类型为整型
  BufHandle b("B", {5}, kInt);
  // 创建名为"x"的变量，类型为整型
  VarHandle x("x", kInt);
  // 创建名为"y"的变量，类型为整型
  VarHandle y("y", kInt);

  // 构造一个循环语句，遍历y从0到9，内部执行条件赋值语句
  StmtPtr stmt = For::make(
      y,
      0,
      10,
      Store::make(
          a,
          {x},
          // 创建条件语句，如果x < 3，则加载A[x]，否则加载B[y]
          IfThenElse::make(
              CompareSelect::make(x, 3, CompareSelectOperation::kLT),
              Load::make(a, {x}),
              Load::make(b, {y}))));

  /*
   * for (int y = 0; y < 10; y++) {
   *   A[x] = IfThenElse(x<3 ? 1 : 0, A[x], B[y]);
   * }
   */

  // 注册化生成的语句
  stmt = registerize(stmt);

  /*
   * int A_1 = A[x];
   * for (int y = 0; y < 10; y++) {
   *   A_1 = IfThenElse(x<3 ? 1 : 0, A_1, B[y]);
   * }
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: for (
# CHECK:   A_1 = IfThenElse(x<3 ? 1 : 0, A_1, B[y]);
# CHECK: }
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 如果右手边的访问会影响创建的可见性，则无法注册化
TEST(Registerizer, RegisterizerIfThenElseLoopCut) {
  // 创建名为"A"的缓冲区，大小为5，类型为整型
  BufHandle a("A", {5}, kInt);
  // 创建名为"B"的缓冲区，大小为5，类型为整型
  BufHandle b("B", {5}, kInt);
  // 创建名为"x"的变量，类型为整型
  VarHandle x("x", kInt);
  // 创建名为"y"的变量，类型为整型
  VarHandle y("y", kInt);

  // 构造一个块语句，内部包含一个循环语句，遍历y从0到9，内部执行条件赋值语句
  StmtPtr stmt = Block::make({For::make(
      y,
      0,
      10,
      Store::make(
          a,
          {x},
          // 创建条件语句，如果x < 3，则加载A[x]，否则加载A[y]
          IfThenElse::make(
              CompareSelect::make(x, 3, CompareSelectOperation::kLT),
              Load::make(a, {x}),
              Load::make(a, {y}))))});

  /*
   * for (int y = 0; y < 10; y++) {
   *   A[x] = IfThenElse(x<3 ? 1 : 0, A[x], A[y]);
   * }
   */

  // 在没有变化的情况下注册化语句
  std::ostringstream before;
  before << *stmt;

  // 注册化语句后
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// 简单情况，其中一个访问由程序后续的重叠访问截断，我们可以注册化直至重叠处
TEST(Registerizer, RegisterizerPartialAfter) {
  // 创建名为"A"的缓冲区，大小为1，类型为整型
  BufHandle a("A", {1}, kInt);
  // 创建名为"x"的变量，类型为整型
  VarHandle x("x", kInt);
  // 构造一个块语句，内部包含两个循环语句和一个赋值语句
  StmtPtr stmt = Block::make(
      {Store::make(a, {0}, 0),
       // 第一个循环，遍历x从0到9，内部执行累加赋值语句
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {0}), x))})),
       // 第二个循环，遍历x从1到9，内部执行递归赋值语句
       For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1})))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   */

  // 注册化生成的语句
  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = A_1 + x;
   * }
   * A[0] = A_1;
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (
# CHECK:   A_1 = A_1 + x;
# CHECK: }
# CHECK: A[0] = A_1;
# CHECK: for (
# CHECK:   A[x] = A[x - 1];
# CHECK: }
// 检查是否不包含字符串"A)IR"，用于验证测试结果中不应包含的部分
torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 在注册化中，可以处理部分重叠的访问，初始化值必须插入到前一个访问之后。
TEST(Registerizer, RegisterizerPartialBefore) {
  // 创建一个名为"A"的缓冲区，包含一个整数元素
  BufHandle a("A", {1}, kInt);
  // 创建一个名为"x"的变量，类型为整数
  VarHandle x("x", kInt);
  // 创建一个语句块，包含三个子语句：循环、存储和加载操作
  StmtPtr stmt = Block::make(
      {For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1}))),
       Store::make(a, {0}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {0}), x))}))});

  /*
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  // 进行注册化处理
  stmt = registerize(stmt);

  /*
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = A_1 + x;
   * }
   * A[0] = A_1;
   */

  // 将语句输出到字符串流oss中
  std::ostringstream oss;
  oss << *stmt;

  // 定义用于验证输出模式的字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: int
# CHECK: for (
# CHECK:   A[x] = A[x - 1];
# CHECK: }
# CHECK: int A_1 = 0;
# CHECK: for (
# CHECK:   A_1 = A_1 + x;
# CHECK: }
# CHECK: A[0] = A_1;)IR";

  // 使用FileCheck工具对输出oss.str()进行模式匹配验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 前两个测试的组合，访问被两个方向上的重叠访问切分。
TEST(Registerizer, RegisterizerPartialInside) {
  // 创建一个名为"A"的缓冲区，包含一个整数元素
  BufHandle a("A", {1}, kInt);
  // 创建三个整数类型的变量：x1、x2、x3
  VarHandle x1("x1", kInt);
  VarHandle x2("x2", kInt);
  VarHandle x3("x3", kInt);
  // 创建一个语句块，包含四个子语句：存储操作和三个循环
  StmtPtr stmt = Block::make(
      {Store::make(a, {0}, 2),
       For::make(
           x1, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), x1))),
       For::make(x2, 1, 10, Store::make(a, {x2}, Load::make(a, {x2 - 1}))),
       For::make(
           x3, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), x3)))});

  /*
   * A[0] = 2;
   * for (int x1 = 0; x1 < 10; x1++) {
   *   A[0] = (A[0]) + x1;
   * }
   * for (int x2 = 1; x2 < 10; x2++) {
   *   A[x2] = A[x2 - 1];
   * }
   * for (int x3 = 0; x3 < 10; x3++) {
   *   A[0] = (A[0]) + x3;
   * }
   */

  // 进行注册化处理
  stmt = registerize(stmt);

  /*
   * int A_1 = 2;
   * for (int x1 = 0; x1 < 10; x1++) {
   *   A_1 = A_1 + x1;
   * }
   * A[0] = A_1;
   * for (int x2 = 1; x2 < 10; x2++) {
   *   A[x2] = A[x2 - 1];
   * }
   * int A_2 = A[0];
   * for (int x3 = 0; x3 < 10; x3++) {
   *   A_2 = A_2 + x3;
   * }
   * A[0] = A_2;
   */

  // 将语句输出到字符串流oss中
  std::ostringstream oss;
  oss << *stmt;

  // 定义用于验证输出模式的字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 2;
# CHECK: for (
# CHECK:   A_1 = A_1 + x1;
# CHECK: }
# CHECK: A[0] = A_1;
# CHECK: for (
# CHECK:   A[x2] =
# CHECK: }
# CHECK: int A_2 = A[0];
# CHECK: for (
# CHECK:   A_2 = A_2 + x3;
# CHECK: }
# CHECK: A[0] = A_2;)IR";

  // 使用FileCheck工具对输出oss.str()进行模式匹配验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Registerizer, RegisterizerPartialCondition) {
  // 创建一个名为 "A" 的缓冲区句柄，大小为 {1}，类型为 kInt
  BufHandle a("A", {1}, kInt);
  // 创建一个名为 "x" 的变量句柄，类型为 kInt
  VarHandle x("x", kInt);
  // 创建一个语句块，包含多个语句
  StmtPtr stmt = Block::make(
      // 将值 2 存储到缓冲区 "A" 的索引 {0} 处
      {Store::make(a, {0}, 2),
       // 创建一个循环，变量 x 在 [0, 10) 范围内，将计算结果存储到缓冲区 "A" 的索引 {0} 处
       For::make(
           x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), x))),
       // 创建一个条件语句，根据 x 是否小于 5 来决定存储操作
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Store::make(a, {x}, Load::make(a, {x - 1})),
           nullptr),
       // 创建一个循环，类似之前的操作，将计算结果存储到缓冲区 "A" 的索引 {0} 处
       For::make(
           x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), x)))});

  // 对 stmt 进行寄存器化处理
  stmt = registerize(stmt);

  // 创建一个字符串输出流 oss，并将 stmt 的内容写入其中
  std::ostringstream oss;
  oss << *stmt;

  // 定义用于验证输出的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 2;
# CHECK: for (
# CHECK:   A_1 = A_1 + x;
# CHECK: }
# CHECK: A[0] = A_1;
# CHECK: if (
# CHECK:   A[x] =
# CHECK: }
# CHECK: int A_2 = A[0];
# CHECK: for (
# CHECK:   A_2 = A_2 + x;
# CHECK: }
# CHECK: A[0] = A_2;)IR";

  // 使用 FileCheck 运行验证模式，检查 oss 中的输出是否符合预期
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Registerizer, RegisterizerPartialConditionInternalCut) {
  // 创建一个名为 "A" 的缓冲区句柄，大小为 {1}，类型为 kInt
  BufHandle a("A", {1}, kInt);
  // 创建一个名为 "x" 的变量句柄，类型为 kInt
  VarHandle x("x", kInt);
  // 创建一个语句块，包含多个语句
  StmtPtr stmt = Block::make(
      // 将值 1 存储到缓冲区 "A" 的索引 {0} 处
      {Store::make(a, {0}, 1),
       // 将值 3 存储到缓冲区 "A" 的索引 {0} 处
       Store::make(a, {0}, 3),
       // 创建一个条件语句块，根据 x 是否小于 5 来决定存储操作
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make({Store::make(a, {x}, 1), Store::make(a, {x}, 3)}),
           nullptr),
       // 将值 4 存储到缓冲区 "A" 的索引 {0} 处
       Store::make(a, {0}, 4),
       // 将值 6 存储到缓冲区 "A" 的索引 {0} 处
       Store::make(a, {0}, 6)});

  // 对 stmt 进行寄存器化处理
  stmt = registerize(stmt);

  // 创建一个字符串输出流 oss，并将 stmt 的内容写入其中
  std::ostringstream oss;
  oss << *stmt;

  // 定义用于验证输出的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 1;
# CHECK: A_1 = 3
# CHECK: A[0] = A_1;
# CHECK: if (
# CHECK:   int A_2 = 1;
# CHECK:   A_2 = 3;
# CHECK:   A[x] = A_2;
# CHECK: }
# CHECK: int A_3 = 4;
# CHECK: A_3 = 6;
# CHECK: A[0] = A_3;)IR";

  // 使用 FileCheck 运行验证模式，检查 oss 中的输出是否符合预期
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
// First statement in condition closes outer access, but can be registerized
// with later statements.
TEST(Registerizer, RegisterizerPartialConditionInternalStart) {
  BufHandle a("A", {1}, kInt);  // 声明名为 "A" 的缓冲区，大小为 {1}，类型为整数
  VarHandle x("x", kInt);  // 声明名为 "x" 的变量，类型为整数
  StmtPtr stmt = Block::make(  // 创建一个语句块
      {Store::make(a, {0}, 1),  // 将值 1 存储到缓冲区 "A" 的索引 {0}
       Store::make(a, {0}, 3),  // 将值 3 存储到缓冲区 "A" 的索引 {0}
       Cond::make(  // 创建一个条件语句
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),  // 创建一个比较选择节点，比较 x 是否小于 5
           Block::make({Store::make(a, {x}, 1), Store::make(a, {x}, 3)}),  // 如果 x 小于 5，则将值 1 和 3 存储到缓冲区 "A" 的索引 {x}
           nullptr),  // 如果 x 不小于 5，则为空语句块
       Store::make(a, {x}, 4),  // 将值 4 存储到缓冲区 "A" 的索引 {x}
       Store::make(a, {x}, 6)});  // 将值 6 存储到缓冲区 "A" 的索引 {x}

  /*
   * A[0] = 1;
   * A[0] = 3;
   * if (x<5 ? 1 : 0) {
   *   A[x] = 1;
   *   A[x] = 3;
   * }
   * A[x] = 4;
   * A[x] = 6;
   */

  stmt = registerize(stmt);  // 对语句块进行寄存器化处理

  /*
   * int A_1 = 1;
   * A_1 = 3;
   * A[0] = A_1;
   * int A_2 = A[x];    <--- must read from the input here.
   * if (x<5 ? 1 : 0) {
   *   A_2 = 1;
   *   A_2 = 3;
   * }
   * A_2 = 4;
   * A_2 = 6;
   * A[x] = A_2;
   */

  // TODO: I suppose we could refactor with a conditional initializer?

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 1;
# CHECK: A_1 = 3
# CHECK: A[0] = A_1;
# CHECK: int A_2 = A[x];
# CHECK: if (
# CHECK:   A_2 = 1;
# CHECK:   A_2 = 3;
# CHECK: }
# CHECK: A_2 = 4;
# CHECK: A_2 = 6;
# CHECK: A[x] = A_2;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// An access cuts two open overlaps and creates four scalar variables.
TEST(Registerizer, RegisterizerPartialOverlapsTwo) {
  BufHandle a("A", {1}, kInt);  // 声明名为 "A" 的缓冲区，大小为 {1}，类型为整数
  VarHandle x("x", kInt);  // 声明名为 "x" 的变量，类型为整数
  StmtPtr stmt = Block::make(  // 创建一个语句块
      {Store::make(a, {1}, Load::make(a, {0})),  // 将缓冲区 "A" 的索引 {0} 的值加载并存储到索引 {1}
       Store::make(a, {0}, Load::make(a, {1})),  // 将缓冲区 "A" 的索引 {1} 的值加载并存储到索引 {0}
       Store::make(a, {0}, Load::make(a, {1})),  // 将缓冲区 "A" 的索引 {1} 的值加载并存储到索引 {0}
       For::make(x, 1, 10, Store::make(a, {x}, x)),  // 创建一个 for 循环，从 1 到 10，将 x 的值存储到缓冲区 "A" 的索引 {x}
       Store::make(a, {1}, Load::make(a, {0})),  // 将缓冲区 "A" 的索引 {0} 的值加载并存储到索引 {1}
       Store::make(a, {0}, Load::make(a, {1})),  // 将缓冲区 "A" 的索引 {1} 的值加载并存储到索引 {0}
       Store::make(a, {0}, Load::make(a, {1}))});  // 将缓冲区 "A" 的索引 {1} 的值加载并存储到索引 {0}

  /*
   * A[1] = A[0];
   * A[0] = A[1];
   * A[0] = A[1];
   * for (int x = 1; x < 10; x++) {
   *   A[x] = x;
   * }
   * A[1] = A[0];
   * A[0] = A[1];
   * A[0] = A[1];
   */

  stmt = registerize(stmt);  // 对语句块进行寄存器化处理

  /*
   * int A_1 = A[0];
   * int A_2 = A_1;
   * A_1 = A_2;
   * A_1 = A_2;
   * A[1] = A_2;
   * A[0] = A_1;
   * for (int x = 1; x < 10; x++) {
   *   A[x] = x;
   * }
   * int A_3 = A[0];
   * int A_4 = A_3;
   * A_3 = A_4;
   * A_3 = A_4;
   * A[1] = A_4;
   * A[0] = A_3;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: int A_2 = A_1;
# CHECK: A_1 = A_2;
# CHECK: A_1 = A_2;
# CHECK: A[1] = A_2;
# CHECK: A[0] = A_1;
# CHECK: for (
# CHECK:   A[x] = x;
# CHECK: }
# CHECK: int A_3 = A[0];
# CHECK: int A_4 = A_3;
# CHECK: A_3 = A_4;
# CHECK: A_3 = A_4;
# CHECK: A[1] = A_4;
# CHECK: A[0] = A_3;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
// 定义一个名为 Registerizer 的测试类，测试 RegisterizerNestedBlocks 函数
TEST(Registerizer, RegisterizerNestedBlocks) {
  // 创建一个大小为 1 的整型缓冲区变量 A，初始值为 0
  BufHandle a("A", {1}, kInt);
  // 创建一个整型变量 x，未初始化
  VarHandle x("x", kInt);
  // 创建一个语句指针 stmt，表示一个代码块
  StmtPtr stmt = Block::make(
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      {
        // 将 A[0] 的值加 1 后存回 A[0]
        Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
        // 创建一个嵌套的代码块，将 A[0] 的值加 2 后存回 A[0]
        Block::make({Store::make(a, {0}, Add::make(Load::make(a, {0}), 2))}),
        // 创建另一个嵌套的代码块
        Block::make(
            // 将 A[0] 的值加 3 后存回 A[0]
            {Store::make(a, {0}, Add::make(Load::make(a, {0}), 3)),
             // 创建更深层的嵌套代码块，将 A[0] 的值加 4 后存回 A[0]
             Block::make(
                 {Store::make(a, {0}, Add::make(Load::make(a, {0}), 4))})})
      });

  // 对 stmt 进行寄存器化处理
  stmt = registerize(stmt);

  // 创建一个字符串流 oss，用于将 stmt 的内容输出到字符串中
  std::ostringstream oss;
  oss << *stmt;

  // 定义一个字符串 verification_pattern，用于验证输出字符串是否符合预期
  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: A_1 = A_1 + 1;
# CHECK: A_1 = A_1 + 2;
# CHECK: A_1 = A_1 + 3;
# CHECK: A_1 = A_1 + 4;
# CHECK: A[0] = A_1;)IR";

  // 使用 FileCheck 类检查输出字符串 oss 中是否包含 verification_pattern 的内容
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 测试 RegisterizerNestedConditions 函数，处理嵌套条件的情况
TEST(Registerizer, RegisterizerNestedConditions) {
  // 创建一个大小为 1 的整型缓冲区变量 A，初始值为 0
  BufHandle a("A", {1}, kInt);
  // 创建一个整型变量 x，未初始化
  VarHandle x("x", kInt);
  // 创建一个语句指针 stmt，表示一个代码块
  StmtPtr stmt = Block::make({Cond::make(
      // 创建一个比较选择节点，判断 x 是否小于 5
      CompareSelect::make(x, 5, CompareSelectOperation::kLT),
      // 如果 x 小于 5，则执行以下语句块
      Block::make(
          // 将 A[0] 的值加 1 后存回 A[0]
          {Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
           // 创建一个嵌套的条件语句块
           Cond::make(
               // 创建一个比较选择节点，判断 x 是否等于 2
               CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
               // 如果 x 等于 2，则执行以下语句块
               Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
               // 否则，为空语句块
               nullptr)}),
      // 如果 x 不小于 5，则为空语句块
      nullptr)});

  // 对 stmt 进行寄存器化处理
  stmt = registerize(stmt);

  // 创建一个字符串流 oss，用于将 stmt 的内容输出到字符串中
  std::ostringstream oss;
  oss << *stmt;

  // 定义一个字符串 verification_pattern，用于验证输出字符串是否符合预期
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x<5
# CHECK:   int A_1 = A[0];
# CHECK:   A_1 = A_1 + 1;
# CHECK:   if (x==2
# CHECK:     A_1 = A_1 + 1;
# CHECK:   }
# CHECK: A[0] = A_1;
# CHECK: })IR";

  // 使用 FileCheck 类检查输出字符串 oss 中是否包含 verification_pattern 的内容
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Registerizer, RegisterizerNestedConditionsUnhidden) {
  BufHandle a("A", {1}, kInt);  // 创建一个名为"A"的缓冲区，初始大小为1，存储整数类型数据
  VarHandle x("x", kInt);  // 创建一个名为"x"的变量，类型为整数
  StmtPtr stmt = Block::make(  // 创建一个语句块
      {Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),  // A[0] = (A[0]) + 1;
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),  // 如果 x < 5
           Block::make(
               {Store::make(a, {1}, 1),  // A[1] = 1;
                Cond::make(
                    CompareSelect::make(x, 2, CompareSelectOperation::kEQ),  // 如果 x == 2
                    Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),  // A[0] = (A[0]) + 1;
                    nullptr)}),  // 否则为空
           nullptr)});  // 否则为空

  /*
   * A[0] = (A[0]) + 1;
   * if (x<5 ? 1 : 0) {
   *   A[1] = 1;
   *   if (x==2 ? 1 : 0) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  stmt = registerize(stmt);  // 对语句进行寄存器分配处理

  /*
   * int A_1 = A[0];
   * A_1 = A_1 + 1;
   * if (x<5 ? 1 : 0) {
   *   A[1] = 1;
   *   if (x==2 ? 1 : 0) {
   *     A_1 = A_1 + 1;
   *   }
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: A_1 = A_1 + 1;
# CHECK: if (x<5
# CHECK:   A[1] = 1;
# CHECK:   if (x==2
# CHECK:     A_1 = A_1 + 1;
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Registerizer, RegisterizerNestedConditionsHiddenFirst) {
  BufHandle a("A", {1}, kInt);  // 创建一个名为"A"的缓冲区，初始大小为1，存储整数类型数据
  VarHandle x("x", kInt);  // 创建一个名为"x"的变量，类型为整数
  StmtPtr stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 2, CompareSelectOperation::kEQ),  // 如果 x == 2
           Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),  // A[0] = (A[0]) + 1;
           nullptr),  // 否则为空
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),  // 如果 x < 5
           Block::make({Cond::make(
               CompareSelect::make(x, 2, CompareSelectOperation::kEQ),  // 如果 x == 2
               Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),  // A[0] = (A[0]) + 1;
               nullptr)}),  // 否则为空
           nullptr)});  // 否则为空

  /*
   * if (x==2 ? 1 : 0) {
   *   A[0] = (A[0]) + 1;
   * }
   * if (x<5 ? 1 : 0) {
   *   if (x==2 ? 1 : 0) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);  // 对语句进行寄存器分配处理

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  stmt = registerize(stmt);  // 对语句进行寄存器分配处理
}
TEST(Registerizer, RegisterizerNestedConditionsHiddenSecond) {
  // 创建一个名为 "A" 的缓冲区，初始值为 {1}，数据类型为整数
  BufHandle a("A", {1}, kInt);
  // 创建一个名为 "x" 的变量，数据类型为整数
  VarHandle x("x", kInt);
  // 创建一个语句块，包含两个条件语句
  StmtPtr stmt = Block::make(
      { // 第一个条件语句
       Cond::make(
           // 如果 x < 5 则为真，比较选择操作为 kLT
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           // 如果为真，执行内部语句块
           Block::make({ // 内部条件语句
               Cond::make(
                   // 如果 x == 2 则为真，比较选择操作为 kEQ
                   CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
                   // 执行 A[0] = (A[0]) + 1;
                   Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
                   nullptr)}),
           nullptr),
       // 第二个条件语句
       Cond::make(
           // 如果 x == 2 则为真，比较选择操作为 kEQ
           CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
           // 执行 A[0] = (A[0]) + 1;
           Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
           nullptr)});

  /*
   * if (x<5 ? 1 : 0) {
   *   if (x==2 ? 1 : 0) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   * if (x==2 ? 1 : 0) {
   *   A[0] = (A[0]) + 1;
   * }
   */

  // 创建一个字符串流对象 before，将 stmt 的内容输出到 before 中
  std::ostringstream before;
  before << *stmt;

  // 调用 registerize 函数处理 stmt，没有改变
  stmt = registerize(stmt);

  // 创建一个字符串流对象 after，将处理后的 stmt 的内容输出到 after 中
  std::ostringstream after;
  after << *stmt;

  // 断言 before 和 after 的字符串相等
  ASSERT_EQ(before.str(), after.str());

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  // 再次调用 registerize 函数处理 stmt
  stmt = registerize(stmt);
}

// If an access is cut by another access internal to a condition block, it still
// cuts the access.
TEST(Registerizer, RegisterizerNestedConditionsCut) {
  // 创建一个名为 "A" 的缓冲区，初始值为 {1}，数据类型为整数
  BufHandle a("A", {1}, kInt);
  // 创建一个名为 "x" 的变量，数据类型为整数
  VarHandle x("x", kInt);
  // 创建一个语句块，包含一个存储操作和一个条件语句
  StmtPtr stmt = Block::make(
      { // 存储操作：执行 A[0] = (A[0]) + 1;
       Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
       // 条件语句
       Cond::make(
           // 如果 x < 5 则为真，比较选择操作为 kLT
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           // 如果为真，执行内部语句块
           Block::make(
               { // 存储操作：执行 A[x] = 1;
                Store::make(a, {x}, 1),
                // 内部条件语句
                Cond::make(
                    // 如果 x == 2 则为真，比较选择操作为 kEQ
                    CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
                    // 执行 A[0] = (A[0]) + 1;
                    Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
                    nullptr)}),
           nullptr)});

  /*
   * A[0] = (A[0]) + 1;
   * if (x<5 ? 1 : 0) {
   *   A[x] = 1;
   *   if (x==2 ? 1 : 0) {
   *
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  // 创建一个字符串流对象 before，将 stmt 的内容输出到 before 中
  std::ostringstream before;
  before << *stmt;

  // 调用 registerize 函数处理 stmt，没有改变
  stmt = registerize(stmt);

  // 创建一个字符串流对象 after，将处理后的 stmt 的内容输出到 after 中
  std::ostringstream after;
  after << *stmt;

  // 断言 before 和 after 的字符串相等
  ASSERT_EQ(before.str(), after.str());
}
TEST(Registerizer, RegisterizerNestedConditionLoopHidden) {
  // 创建一个名为a的缓冲区，包含10个元素，每个元素为整数类型
  BufHandle a("A", {10}, kInt);
  // 创建一个名为b的缓冲区，包含10个元素，每个元素为整数类型
  BufHandle b("B", {10}, kInt);
  // 创建一个名为x的变量，类型为整数
  VarHandle x("x", kInt);
  // 创建一个语句指针，指向一个由多个语句组成的块
  StmtPtr stmt = Block::make(
      // 创建一个条件语句，判断x是否等于2，如果是则执行A[0] = A[0] + 1
      {Cond::make(
           CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
           Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
           nullptr),
       // 创建一个循环语句，循环变量为x，范围从0到9
       For::make(
           x,
           0,
           10,
           Block::make(
               // 在循环体内部创建一个块，包含两个语句
               {Store::make(b, {x}, 0),
                // 创建一个条件语句，判断x是否等于2，如果是则执行A[0] = A[0] + 1
                Cond::make(
                    CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
                    Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
                    nullptr)}))});

  /*
   * if (x==2 ? 1 : 0) {
   *   A[0] = (A[0]) + 1;
   * }
   * for (int x = 0; x < 10; x++) {
   *   B[x] = 0;     <-- this is only here to prevent Loop/Cond reordering.
   *   if (x==2 ? 1 : 0) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  // 创建一个ostringstream对象before，将stmt的内容写入其中
  std::ostringstream before;
  before << *stmt;

  // 对stmt进行寄存器分配的操作
  stmt = registerize(stmt);

  // 创建一个ostringstream对象after，将处理后的stmt的内容写入其中
  std::ostringstream after;
  after << *stmt;

  // 断言处理前后的字符串表示应该相等
  ASSERT_EQ(before.str(), after.str());
}

// Three loops and four element regions, three of which should be registerized
// at different levels of the IR.
TEST(Registerizer, RegisterizerNestedConditionThreeDeep) {
  BufHandle a("A", {10}, kInt);  // 创建一个名为"A"的缓冲区句柄a，大小为10，数据类型为整型
  BufHandle b("B", {10}, kInt);  // 创建一个名为"B"的缓冲区句柄b，大小为10，数据类型为整型
  VarHandle x("x", kInt);  // 创建一个名为"x"的变量句柄x，数据类型为整型
  StmtPtr stmt = Block::make(  // 创建一个语句块stmt
      {Store::make(a, {4}, 0),  // 向缓冲区a的索引4处存储值0
       Cond::make(  // 创建一个条件语句
           CompareSelect::make(x, 2, CompareSelectOperation::kGT),  // 比较x是否大于2
           Cond::make(  // 如果条件成立，创建嵌套的条件语句
               CompareSelect::make(x, 3, CompareSelectOperation::kGT),  // 比较x是否大于3
               Block::make({  // 如果条件成立，创建一个包含多个语句的块
                   Cond::make(  // 再次嵌套条件语句
                       CompareSelect::make(x, 4, CompareSelectOperation::kGT),  // 比较x是否大于4
                       Block::make({  // 如果条件成立，创建一个包含多个语句的块
                           Store::make(  // 存储操作
                               a, {1}, Add::make(Load::make(a, {1}), 1)),  // 向a的索引1处存储Load(a[1]) + 1
                           Store::make(  // 存储操作
                               a, {2}, Add::make(Load::make(a, {2}), 1)),  // 向a的索引2处存储Load(a[2]) + 1
                           Store::make(  // 存储操作
                               a, {3}, Add::make(Load::make(a, {3}), 1)),  // 向a的索引3处存储Load(a[3]) + 1
                           Store::make(  // 存储操作
                               a, {4}, Add::make(Load::make(a, {4}), 1)),  // 向a的索引4处存储Load(a[4]) + 1
                           Store::make(  // 存储操作
                               a, {1}, Add::make(Load::make(a, {1}), 1)),  // 再次向a的索引1处存储Load(a[1]) + 1
                       }),
                       nullptr),  // 如果条件不成立，使用nullptr
                   Store::make(a, {2}, Add::make(Load::make(a, {2}), 1)),  // 向a的索引2处存储Load(a[2]) + 1
               }),
               nullptr),  // 如果条件不成立，使用nullptr
           nullptr)});  // 如果条件不成立，使用nullptr

  /*
   * A[4] = 0;
   * if (x>2 ? 1 : 0) {
   *   if (x>3 ? 1 : 0) {
   *     if (x>4 ? 1 : 0) {
   *       A[1] = (A[1]) + 1;
   *       A[2] = (A[2]) + 1;
   *       A[3] = (A[3]) + 1;
   *       A[4] = (A[4]) + 1;
   *       A[1] = (A[1]) + 1;
   *     }
   *     A[2] = (A[2]) + 1;
   *   }
   * }
   */

  stmt = registerize(stmt);  // 对stmt进行寄存器分配优化

  /*
   * int A_1 = 0;
   * if (x>2 ? 1 : 0) {
   *   if (x>3 ? 1 : 0) {
   *     int A_3 = A[2];
   *     if (x>4 ? 1 : 0) {
   *       int A_2 = A[1];
   *       A_2 = A_2 + 1;
   *       A_3 = A_3 + 1;
   *       A[3] = (A[3]) + 1;
   *       A_1 = A_1 + 1;
   *       A_2 = A_2 + 1;
   *       A[1] = A_2;
   *     }
   *     A_3 = A_3 + 1;
   *     A[2] = A_3;
   *   }
   * }
   * A[4] = A_1;
   */

  std::ostringstream oss;  // 创建一个ostringstream对象oss，用于将内容输出为字符串
  oss << *stmt;  // 将stmt的内容输出到oss中

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: if (x>2 ? 1 : 0) {
# CHECK:   if (x>3 ? 1 : 0) {
# CHECK:     int A_3 = A[2];
# CHECK:     if (x>4 ? 1 : 0) {
# CHECK:       int A_2 = A[1];
# CHECK:       A_2 = A_2 + 1;
# CHECK:       A_3 = A_3 + 1;
# CHECK:       A[3] = (A[3]) + 1;
# CHECK:       A_1 = A_1 + 1;
# CHECK:       A_2 = A_2 + 1;
# CHECK:       A[1] = A_2;
# CHECK:     }
# CHECK:     A_3 = A_3 + 1;
# CHECK:     A[2] = A_3;
# CHECK:   }
# CHECK: }
# CHECK: A[4] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用FileCheck验证oss中的内容是否符合verification_pattern中的模式
}

// Can replace a simple scalar access with a local variable even when that
// variable is an outer loop var.
TEST(Registerizer, RegisterizerNestedLoopSimple) {
  // 定义一个名为 A 的缓冲区，包含一个元素，类型为整数
  BufHandle a("A", {1}, kInt);
  // 定义两个整型变量 x 和 y
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 创建一个语句块，包含一个嵌套的 for 循环
  StmtPtr stmt = Block::make({For::make(
      // 外层循环：变量 y 从 0 到 9
      y,
      0,
      10,
      // 内层循环：变量 x 从 0 到 9，对缓冲区 A 中的元素进行累加
      For::make(
          x,
          0,
          10,
          Block::make(
              // 存储操作：将 A[y] 更新为 A[y] + x
              {Store::make(a, {y}, Add::make(Load::make(a, {y}), x))})))});

  /*
   * for (int y = 0; y < 10; y++) {
   *   for (int x = 0; x < 10; x++) {
   *     A[y] = (A[y]) + x;
   *   }
   * }
   */

  // 对语句进行寄存器分配处理
  stmt = registerize(stmt);

  /*
   * for (int y = 0; y < 10; y++) {
   *   int A_1 = A[y];
   *   for (int x = 0; x < 10; x++) {
   *     A_1 = A_1 + x;
   *   }
   * A[y] = A_1;
   * }
   */

  // 将结果输出到字符串流中
  std::ostringstream oss;
  oss << *stmt;

  // 准备用于验证的模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int y
# CHECK:   int A_1 = A[y];
# CHECK:   for (int x
# CHECK:     A_1 = A_1 + x;
# CHECK:   }
# CHECK:   A[y] = A_1;
# CHECK: })IR";

  // 使用 FileCheck 运行验证模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Test the positive case of the hiddenAccess split, where an internal
// conditional access can be hoisted up through a loop to match an existing
// access in a higher scope and the two can be registerized.
TEST(Registerizer, RegisterizerHiddenAccessYes) {
  // 定义两个缓冲区 A 和 B，各包含十个元素，类型为整数
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  // 定义两个整型变量 x 和 y
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 创建一个条件语句块，内含条件选择和循环
  StmtPtr stmt = Block::make({Cond::make(
      // 条件选择：如果 x == 2
      CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
      Block::make(
          // 条件成立分支：包含一个存储操作和一个嵌套循环
          {Store::make(a, {0}, 0),
           For::make(
               // 外层循环：变量 x 从 0 到 9
               x,
               0,
               10,
               Block::make(
                   // 内层循环：变量 y 从 0 到 9，对缓冲区 A 中的元素进行累加
                   {Store::make(b, {x}, 0),
                    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
                    Cond::make(
                        // 内部条件选择：如果 x == 3
                        CompareSelect::make(x, 3, CompareSelectOperation::kEQ),
                        For::make(
                            // 内部循环：变量 y 从 0 到 9，对缓冲区 A 中的特定元素进行操作
                            y,
                            0,
                            10,
                            Store::make(
                                a, {0}, Add::make(Load::make(a, {0}), 1))),
                        nullptr)}))}),
      nullptr)});

  /*
   * if (x==2 ? 1 : 0) {
   *   A[0] = 0;
   *   for (int x = 0; x < 10; x++) {
   *     B[x] = 0;
   *     if (x==3 ? 1 : 0) {
   *       for (int y = 0; y < 10; y++) {
   *         A[0] = (A[0]) + 1;
   *       }
   *     }
   *   }
   * }
   */

  // 对语句进行寄存器分配处理
  stmt = registerize(stmt);

  /*
   * if (x==2 ? 1 : 0) {
   *   int A_1 = 0;
   *   for (int x = 0; x < 10; x++) {
   *     B[x] = 0;
   *     if (x==3 ? 1 : 0) {
   *       for (int y = 0; y < 10; y++) {
   *         A_1 = A_1 + 1;
   *       }
   *     }
   *   }
   *   A[0] = A_1;
   * }
   */

  // 将结果输出到字符串流中
  std::ostringstream oss;
  oss << *stmt;

  // 准备用于验证的模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x==2
# CHECK:   int A_1 = 0;
# CHECK:   for (int x
# CHECK:     B[x] = 0;
# CHECK:     if (x==3
# CHECK:       for (int y
// 测试 RegisterizerHiddenAccessNo 的负面情况，即 hiddenAccess 的拆分，
// 其中 hoisted 访问在更高的作用域中从未 unhidden，并且 registerization 发生在较低的作用域中。
TEST(Registerizer, RegisterizerHiddenAccessNo) {
  // 创建名为 A 和 B 的缓冲区对象，分别是大小为 10 的整型数组
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  // 创建名为 x 和 y 的变量对象，均为整型
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 创建一个语句对象 stmt，包含以下内容：
  StmtPtr stmt = Block::make({Cond::make(
      // 如果 x == 2 则为真，生成条件语句块
      CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
      // 创建一个块，包含以下内容：
      Block::make({For::make(
          // for 循环，循环变量 x，范围 0 到 10
          x,
          0,
          10,
          // 创建一个块，包含以下内容：
          Block::make(
              {Store::make(b, {x}, 0),  // 将 B[x] 设为 0
               // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
               Cond::make(
                   // 如果 x == 3 则为真，生成条件语句块
                   CompareSelect::make(x, 3, CompareSelectOperation::kEQ),
                   // 创建一个 for 循环，循环变量 y，范围 0 到 10
                   For::make(
                       y,
                       0,
                       10,
                       // 将 A[0] 设为 A[0] + 1
                       Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),
                   nullptr)}))}),
      nullptr)});

  // 将 stmt 通过 registerize 函数进行处理
  stmt = registerize(stmt);

  // 将 stmt 转换为字符串流 oss
  std::ostringstream oss;
  oss << *stmt;

  // 定义验证模式字符串，用于检查是否正确 registerize
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x==2
# CHECK:   for (int x
# CHECK:     B[x] = 0;
# CHECK:     if (x==3
# CHECK:       int A_1 = A[0];
# CHECK:       for (int y
# CHECK:         A_1 = A_1 + 1;
# CHECK:       }
# CHECK:       A[0] = A_1;
# CHECK:     }
# CHECK:   }
# CHECK: })IR";

  // 运行 FileCheck 验证生成的代码是否符合 verification_pattern
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 在这种情况下，条件访问必须被两个循环 hoisted，这里有两个访问，一个 unhidden，另一个没有。
// A[0] 可以被 registerize，但是 B[0] 不能。
TEST(Registerizer, RegisterizerHiddenAccessMultiLoop) {
  // 创建名为"A"的缓冲区，大小为10，数据类型为整数
  BufHandle a("A", {10}, kInt);
  // 创建名为"B"的缓冲区，大小为10，数据类型为整数
  BufHandle b("B", {10}, kInt);
  // 创建名为"x"的变量，数据类型为整数
  VarHandle x("x", kInt);
  // 创建名为"y"的变量，数据类型为整数
  VarHandle y("y", kInt);
  // 创建语句指针stmt，表示一个代码块，包含条件语句和循环语句
  StmtPtr stmt = Block::make({Cond::make(
      // 创建条件语句，判断x是否等于2
      CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
      // 若条件成立，执行以下代码块
      Block::make(
          {Store::make(a, {0}, 0),  // 将0存储到数组A的第一个元素
           // 创建循环语句，遍历x从0到10
           For::make(
               x,
               0,
               10,
               // 嵌套循环语句，遍历y从0到10
               For::make(
                   y,
                   0,
                   10,
                   // 创建条件语句，判断y是否等于3
                   Block::make({Cond::make(
                       CompareSelect::make(y, 3, CompareSelectOperation::kEQ),
                       // 若条件成立，执行以下代码块
                       Block::make(
                           {Store::make(
                                a, {0}, Add::make(Load::make(a, {0}), 1)),  // A[0]增加1
                            Store::make(
                                b, {0}, Add::make(Load::make(b, {0}), 1))}),  // B[0]增加1
                       nullptr)})))}),  // 条件不成立时的处理为nullptr
      nullptr)});  // 条件不成立时的处理为nullptr

  // 运行registerize函数处理stmt
  stmt = registerize(stmt);

  // 创建字符串流oss，用于存储stmt的文本表示
  std::ostringstream oss;
  oss << *stmt;

  // 定义字符串verification_pattern，包含预期的IR表示
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x==2
# CHECK:   int A_1 = 0;
# CHECK:   for (int x
# CHECK:     for (int y
# CHECK:       if (y==3
# CHECK:         A_1 = A_1 + 1;
# CHECK:         B[0] = (B[0]) + 1;
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK:  A[0] = A_1;
# CHECK: })IR";

  // 使用FileCheck工具验证oss.str()中的IR表示与verification_pattern是否匹配
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// 在两个条件语句内部注册访问，但其直接父级不是条件语句。
TEST(Registerizer, RegisterizerTwoConditionalLoops) {
  BufHandle a("A", {1}, kInt);  // 创建一个名为"A"的缓冲区句柄，大小为{1}，类型为整型
  VarHandle x("x", kInt);  // 创建一个名为"x"的变量句柄，类型为整型
  StmtPtr stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),  // 创建一个条件节点，判断x是否小于5
           For::make(
               x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),  // 如果条件成立，在循环中将A[0]加1
           nullptr),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kGT),  // 创建另一个条件节点，判断x是否大于5
           For::make(
               x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),  // 如果条件成立，在循环中将A[0]加1
           nullptr)});

  /*
   * if (x<5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   * if (x>5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  stmt = registerize(stmt);  // 对语句进行寄存器分配优化

  /*
   * if (x<5 ? 1 : 0) {
   *   int A_1 = A[0];
   *   for (int x = 0; x < 10; x++) {
   *     A_1 = A_1 + 1;
   *   }
   *   A[0] = A_1;
   * }
   * if (x>5 ? 1 : 0) {
   *   int A_2 = A[0];
   *   for (int x = 0; x < 10; x++) {
   *     A_2 = A_2 + 1;
   *   }
   *   A[0] = A_2;
   * }
   */

  std::ostringstream oss;  // 创建一个ostringstream对象，用于构建字符串流
  oss << *stmt;  // 将stmt的内容输出到字符串流中

  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x<5
# CHECK:   int A_1 = A[0];
# CHECK:   for (int x
# CHECK:     A_1 = A_1 + 1;
# CHECK:   }
# CHECK:   A[0] = A_1;
# CHECK: }
# CHECK: if (x>5
# CHECK:   int A_2 = A[0];
# CHECK:   for (int x
# CHECK:     A_2 = A_2 + 1;
# CHECK:   }
# CHECK:   A[0] = A_2;
# CHECK: })IR";  // 定义一个用于验证输出的字符串模式

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用FileCheck验证输出是否符合预期
}

// Accesses are registerized inside two conditions, cut in the middle.
TEST(Registerizer, RegisterizerTwoConditionalLoopsCut) {
  // 创建一个名为 "A" 的缓冲区，包含一个元素，类型为整数
  BufHandle a("A", {1}, kInt);
  // 创建一个整数类型的变量 x
  VarHandle x("x", kInt);
  // 创建一个语句块，包含两个条件语句和一个普通循环
  StmtPtr stmt = Block::make(
      {Cond::make(
           // 创建比较选择节点，判断 x 是否小于 5
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           // 如果条件为真，创建一个循环，对缓冲区 A 中的第一个元素执行加一操作
           For::make(
               x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),
           nullptr),
       // 创建一个普通循环，对缓冲区 A 中的所有元素赋值为 1
       For::make(x, 0, 10, Store::make(a, {x}, 1)),
       // 创建另一个条件语句，判断 x 是否大于 5
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kGT),
           // 如果条件为真，创建一个循环，对缓冲区 A 中的第一个元素执行加一操作
           For::make(
               x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),
           nullptr)});

  /*
   * if (x<5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[x] = 1;
   * }
   * if (x>5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  // 将语句块传入 registerize 函数，进行寄存器分配
  stmt = registerize(stmt);

  /*
   * if (x<5 ? 1 : 0) {
   *   int A_1 = A[0];
   *   for (int x = 0; x < 10; x++) {
   *     A_1 = A_1 + 1;
   *   }
   *   A[0] = A_1;
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[x] = 1;
   * }
   * if (x>5 ? 1 : 0) {
   *   int A_2 = A[0];
   *   for (int x = 0; x < 10; x++) {
   *     A_2 = A_2 + 1;
   *   }
   *   A[0] = A_2;
   * }
   */

  // 创建一个 ostringstream 对象 oss，将 stmt 的内容写入其中
  std::ostringstream oss;
  oss << *stmt;

  // 定义 IR 校验模式字符串，用于验证生成的 IR 是否符合预期
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x<5
# CHECK:   int A_1 = A[0];
# CHECK:   for (int x
# CHECK:     A_1 = A_1 + 1;
# CHECK:   }
# CHECK:   A[0] = A_1;
# CHECK: }
# CHECK: for (int x
# CHECK:  A[x] = 1;
# CHECK: if (x>5
# CHECK:   int A_2 = A[0];
# CHECK:   for (int x
# CHECK:     A_2 = A_2 + 1;
# CHECK:   }
# CHECK:   A[0] = A_2;
# CHECK: })IR";

  // 使用 FileCheck 对象进行 IR 校验，验证生成的 IR 是否符合预期
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// references a Let var in a local scope which cannot be hoisted out of the
// loop.
TEST(Registerizer, RegisterizerLoopLetVar) {
  // 创建一个名为 "A" 的缓冲区，包含十个元素，类型为整数
  BufHandle a("A", {10}, kInt);
  // 创建整数类型的变量 x 和 y
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 创建一个语句块，包含一个循环，循环体内包含一个声明语句和一个存储语句
  StmtPtr stmt = IRSimplifier::simplify(Block::make({For::make(
      x,
      0,
      10,
      Block::make(
          // 循环体内包含一个声明变量 y 和一个存储操作，y 的值为 30
          {Let::make(y, 30),
           Store::make(a, {y}, Add::make(x, Load::make(a, {y})))}))}));

  /*
   * for (int x = 0; x < 10; x++) {
   *   int y = 30;
   *   A[y] = x + (A[y]);
   * }
   */

  // 将语句块传入 registerize 函数，进行寄存器分配
  std::ostringstream before;
  before << *stmt;

  // 没有改变
  stmt = registerize(stmt);

  // 创建一个 ostringstream 对象 after，将处理后的 stmt 的内容写入其中
  std::ostringstream after;
  after << *stmt;

  // 断言处理前后的字符串表示应该相等
  ASSERT_EQ(before.str(), after.str());
}

// references a Let var in an outer scope that does not prevent hoisting the
// initializer.
TEST(Registerizer, RegisterizerLoopLetVarOuter) {
  BufHandle a("A", {10}, kInt);  // 创建一个名为"A"的缓存句柄，大小为10，类型为整数
  VarHandle x("x", kInt);  // 创建一个名为"x"的变量句柄，类型为整数
  VarHandle y("y", kInt);  // 创建一个名为"y"的变量句柄，类型为整数
  StmtPtr stmt = Block::make(  // 创建一个语句块
      {Let::make(y, 30),  // 定义一个变量y，并赋值为30
       For::make(
           x,
           0,
           10,
           Block::make(  // 创建一个for循环语句块
               {Store::make(a, {y}, Add::make(x, Load::make(a, {y})))}))});

  /*
   * int y = 30;
   * for (int x = 0; x < 10; x++) {
   *   A[y] = x + (A[y]);
   * }
   */

  stmt = registerize(stmt);  // 对语句进行寄存器化处理

  /*
   * int y = 30;
   * int A_1 = A[y];
   * for (int x = 0; x < 10; x++) {
   *   A_1 = A_1 + x;
   * }
   * A[y] = A_1;
   */

  std::ostringstream oss;  // 创建一个字符串流对象oss
  oss << *stmt;  // 将寄存器化后的语句写入字符串流oss中

  const std::string& verification_pattern =
      R"IR(
# CHECK: int y = 30;
# CHECK: int A_1 = A[y];
# CHECK: for (int x
# CHECK:   A_1 = A_1 + x;
# CHECK: A[y] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用FileCheck验证输出的字符串流oss与预期模式是否匹配
}

// Okay so the registerizer generally goes after index flattening, but just in
// case. Test multi index registerization.
TEST(Registerizer, RegisterizerMultiDim) {
  BufHandle a("A", {3, 4, 5}, kInt);  // 创建一个名为"A"的三维缓存句柄，大小为{3, 4, 5}，类型为整数
  VarHandle x("x", kInt);  // 创建一个名为"x"的变量句柄，类型为整数
  StmtPtr stmt = Block::make(  // 创建一个语句块
      {Store::make(a, {0, 1, 2}, 0),  // 将A[0, 1, 2]赋值为0
       For::make(
           x,
           0,
           10,
           Block::make({Store::make(  // 创建一个for循环语句块
               a, {0, 1, 2}, Add::make(Load::make(a, {0, 1, 2}), x))}))});

  /*
   * A[0, 1, 2] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0, 1, 2] = (A[0, 1, 2]) + x;
   * }
   */

  stmt = registerize(stmt);  // 对语句进行寄存器化处理

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[0, 1, 2] = A_1;
   */

  std::ostringstream oss;  // 创建一个字符串流对象oss
  oss << *stmt;  // 将寄存器化后的语句写入字符串流oss中

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0, 1, 2] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用FileCheck验证输出的字符串流oss与预期模式是否匹配
}

// Wont registerize if only some dims match, but will still registerize distinct
// elements.
TEST(Registerizer, RegisterizerMultiDimPartial) {
  BufHandle a("A", {3, 4, 5}, kInt);  // 创建一个名为"A"的三维缓存句柄，大小为{3, 4, 5}，类型为整数
  VarHandle x("x", kInt);  // 创建一个名为"x"的变量句柄，类型为整数
  StmtPtr stmt = Block::make(  // 创建一个语句块
      {Store::make(a, {0, 1, 2}, 0),  // 将A[0, 1, 2]赋值为0
       For::make(
           x,
           0,
           10,
           Block::make({Store::make(  // 创建一个for循环语句块
               a, {0, 2, 2}, Add::make(Load::make(a, {0, 1, 4}), x))}))});

  /*
   * A[0, 1, 2] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0, 2, 2] = (A[0, 1, 4]) + x;
   * }
   */

  stmt = registerize(stmt);  // 对语句进行寄存器化处理

  /*
   * A[0, 1, 2] = 0;
   * int A_1 = A[0, 1, 4];
   * int A_2 = A[0, 2, 2];
   * for (int x = 0; x < 10; x++) {
   *   A_2 = A_1 + x;
   * }
   * A[0, 2, 2] = A_2;
   */

  std::ostringstream oss;  // 创建一个字符串流对象oss
  oss << *stmt;  // 将寄存器化后的语句写入字符串流oss中

  const std::string& verification_pattern =
      R"IR(
# CHECK: A[0, 1, 2] = 0;
# CHECK: int A_1 = A[0, 1, 4];
# CHECK: int A_2 = A[0, 2, 2];
# CHECK: for (
# CHECK:   A_2 = A_1 + x;
// 定义一个测试函数，用于检查注册化器的多维重叠情况。
TEST(Registerizer, RegisterizerMultiDimOverlap) {
  // 创建一个名为 A 的缓冲区，维度为 {3, 4, 5}，元素类型为整型。
  BufHandle a("A", {3, 4, 5}, kInt);
  // 定义两个整型变量 x 和 y。
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 创建一个语句块，包含一个存储操作和一个 for 循环。
  StmtPtr stmt = Block::make(
      {Store::make(a, {0, 1, 2}, 0),
       For::make(
           x,
           0,
           10,
           Block::make({Store::make(
               a, {0, x, 2}, Add::make(Load::make(a, {y, 2, 2}), x))}))});
  // 对语句进行简化。
  stmt = IRSimplifier::simplify(stmt);

  // 创建一个流对象，将简化前的语句转换为字符串并存储在 before 中。
  std::ostringstream before;
  before << *stmt;

  // 调用 registerize 函数对语句进行注册化处理。
  stmt = registerize(stmt);

  // 创建一个流对象，将注册化后的语句转换为字符串并存储在 after 中。
  std::ostringstream after;
  after << *stmt;

  // 断言注册化前后的字符串表示相等。
  ASSERT_EQ(before.str(), after.str());
}

// 如果一个维度已知是不同的，它们不会重叠。
TEST(Registerizer, RegisterizerMultiDimPartialOverlap) {
  // 创建一个名为 A 的缓冲区，维度为 {3, 4, 5}，元素类型为整型。
  BufHandle a("A", {3, 4, 5}, kInt);
  // 定义两个整型变量 x 和 y。
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  // 创建一个语句块，包含一个存储操作和一个 for 循环。
  StmtPtr stmt = Block::make(
      {Store::make(a, {0, 1, 2}, 0),
       For::make(
           x,
           0,
           10,
           Block::make({Store::make(
               a, {0, x, 2}, Add::make(Load::make(a, {y, 2, 4}), x))}))});

  // 调用 registerize 函数对语句进行注册化处理。
  stmt = registerize(stmt);

  // 创建一个流对象，将注册化后的语句转换为字符串并存储在 oss 中。
  std::ostringstream oss;
  oss << *stmt;

  // 定义用于验证模式的字符串，用于检查输出是否符合预期格式。
  const std::string& verification_pattern =
      R"IR(
# CHECK: A[0, 1, 2] = 0;
# CHECK: int A_1 = A[y, 2, 4];
# CHECK: for (
# CHECK:   A[0, x, 2] = A_1 + x;
# CHECK: })IR";

  // 使用 Torch 的 FileCheck 工具运行验证模式，检查 oss 中的输出是否匹配预期模式。
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}
TEST(Registerizer, RegisterizerMultiDim3DReduction1) {
  // 创建名为 "A" 的缓冲区，维度为 {10}，数据类型为整数
  BufHandle a("A", {10}, kInt);
  // 创建名为 "B" 的缓冲区，维度为 {10, 10}，数据类型为整数
  BufHandle b("B", {10, 10}, kInt);
  // 创建名为 "C" 的缓冲区，维度为 {10, 10, 10}，数据类型为整数
  BufHandle c("C", {10, 10, 10}, kInt);
  // 创建整数类型的变量 "x"
  VarHandle x("x", kInt);
  // 创建整数类型的变量 "y"
  VarHandle y("y", kInt);
  // 创建整数类型的变量 "z"
  VarHandle z("z", kInt);
  // 构造一个语句对象，包含三层嵌套的循环
  StmtPtr stmt = For::make(
      x,
      0,
      10,
      For::make(
          y,
          0,
          10,
          For::make(
              z,
              0,
              10,
              // 在 C[x, y, z] 处存储 A[x] * B[x, y] + C[x, y, z] 的结果
              Store::make(
                  c,
                  {x, y, z},
                  Add::make(
                      Load::make(c, {x, y, z}),
                      Mul::make(Load::make(b, {x, y}), Load::make(a, {x})))))));

  /*
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     for (int z = 0; z < 10; z++) {
   *       C[x, y, z] = (C[x, y, z]) + (B[x, y]) * (A[x]);
   *     }
   *   }
   * }
   */

  // 对 stmt 进行寄存器分配优化处理
  stmt = registerize(stmt);

  /*
   * for (int x = 0; x < 10; x++) {
   *   int A_1 = A[x];
   *   for (int y = 0; y < 10; y++) {
   *     int B_1 = B[x, y];
   *     for (int z = 0; z < 10; z++) {
   *       C[x, y, z] = A_1 * B_1 + (C[x, y, z]);
   *     }
   *   }
   * }
   */

  // 将语句对象的内容写入到字符串流 oss 中
  std::ostringstream oss;
  oss << *stmt;

  // 验证模式的字符串，用于检查输出结果是否符合预期格式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int x
# CHECK:   int A_1 = A[x];
# CHECK:   for (int y
# CHECK:     int B_1 = B[x, y];
# CHECK:       for (int z
# CHECK:         C[x, y, z] = A_1 * B_1 + (C[x, y, z]);
# CHECK: })IR";

  // 运行文件检查工具，检查输出字符串是否符合验证模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// A 3D reduction with the same smaller dimensionality using different loop
// vars.
TEST(Registerizer, RegisterizerMultiDim3DReduction2) {
  // 创建缓冲区对象，A、B、C，每个大小为10，类型为整数
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  BufHandle c("C", {10}, kInt);
  // 创建循环变量对象 x、y、z，类型为整数
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  // 创建语句对象，三重嵌套循环，对应于 C[x] = (C[x]) + (B[y]) * (A[x]);
  StmtPtr stmt = For::make(
      x,
      0,
      10,
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      For::make(
          y,
          0,
          10,
          For::make(
              z,
              0,
              10,
              Store::make(
                  c,
                  {x},
                  Add::make(
                      Load::make(c, {x}),
                      Mul::make(Load::make(b, {y}), Load::make(a, {x})))))));

  /*
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     for (int z = 0; z < 10; z++) {
   *       C[x] = (C[x]) + (B[y]) * (A[x]);
   *     }
   *   }
   * }
   */

  // 对语句进行寄存器分配优化
  stmt = registerize(stmt);

  /*
   * for (int x = 0; x < 10; x++) {
   *   int A_1 = A[x];
   *   int C_1 = C[x];
   *   for (int y = 0; y < 10; y++) {
   *     int B_1 = B[y];
   *     for (int z = 0; z < 10; z++) {
   *       C_1 = A_1 * B_1 + C_1;
   *     }
   *   }
   *   C[x] = C_1;
   * }
   */

  // 将优化后的语句对象转换为字符串流
  std::ostringstream oss;
  oss << *stmt;

  // 定义验证模式的字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int x
# CHECK:   int A_1 = A[x];
# CHECK:   int C_1 = C[x];
# CHECK:   for (int y
# CHECK:     int B_1 = B[y];
# CHECK:       for (int z
# CHECK:         C_1 = A_1 * B_1 + C_1;
# CHECK:       }
# CHECK:     }
# CHECK:   C[x] = C_1;
# CHECK: })IR";

  // 运行文件检查工具，验证输出是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

} // namespace jit
} // namespace torch
```