# `.\pytorch\test\cpp\tensorexpr\test_aten.cpp`

```
// 引入所需的头文件：算法、字符串流、异常处理、Google 测试框架、C10 宏定义、C10 范围工具、自定义头文件
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <gtest/gtest.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include "test/cpp/tensorexpr/padded_buffer.h"
#include "test/cpp/tensorexpr/test_base.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"

// 命名空间声明：torch::jit
namespace torch {
namespace jit {

// 使用命名空间：torch::jit::tensorexpr
using namespace torch::jit::tensorexpr;

// 定义测试案例：ATen::_cast_Float
TEST(ATen, _cast_Float) {
  // 定义常量：数组大小为 128
  const int kTotalSize = 128;
  // 创建名为 A 的缓冲区句柄，存储整数类型的数据
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  // 创建名为 B 的缓冲区句柄，存储浮点数类型的数据
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  // 创建变量句柄：索引 index，类型为整数
  VarHandle index = VarHandle("index", kInt);
  // 加载缓冲区 A 中索引为 index 的元素，存储为表达式 load_a
  ExprHandle load_a = a_buf.load(index);
  // 将 load_a 转换为浮点数类型，存储为表达式 to_float
  ExprHandle to_float = Cast::make(kFloat, load_a);
  // 创建存储语句：将 to_float 存储到缓冲区 B 的索引为 index 的位置
  StmtPtr store_b = b_buf.store({index}, to_float);
  // 创建循环语句：遍历索引 index 从 0 到 kTotalSize-1，执行 store_b
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建填充缓冲区：a_v，存储整数类型数据，大小为 kTotalSize
  PaddedBuffer<int> a_v(kTotalSize);
  // 创建填充缓冲区：b_v，存储浮点数类型数据，大小为 kTotalSize
  PaddedBuffer<float> b_v(kTotalSize);

  // 填充缓冲区 a_v：将索引 i 存储到 a_v 的第 i 个位置
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
  }

  // 创建简单 IR 评估器：使用 stmt 语句，操作缓冲区 a_buf 和 b_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  // 执行评估器：将 a_v 的数据转换为 b_v 的数据
  ir_eval(a_v, b_v);

  // 断言：验证转换前后数据是否正确
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), static_cast<float>(i));
  }
}

// 定义测试案例：ATen::negInt
TEST(ATen, negInt) {
  // 定义常量：数组大小为 128
  const int kTotalSize = 128;
  // 创建名为 A 的缓冲区句柄，存储整数类型的数据
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  // 创建名为 B 的缓冲区句柄，存储整数类型的数据
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  // 创建变量句柄：索引 index，类型为整数
  VarHandle index = VarHandle("index", kInt);
  // 加载缓冲区 A 中索引为 index 的元素，存储为表达式 load_a
  ExprHandle load_a = a_buf.load(index);
  // 创建表达式：0 减去 load_a，存储为表达式 to_float
  ExprHandle to_float = Sub::make(0, load_a);
  // 创建存储语句：将 to_float 存储到缓冲区 B 的索引为 index 的位置
  StmtPtr store_b = b_buf.store({index}, to_float);
  // 创建循环语句：遍历索引 index 从 0 到 kTotalSize-1，执行 store_b
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建填充缓冲区：a_v，存储整数类型数据，大小为 kTotalSize
  PaddedBuffer<int> a_v(kTotalSize);
  // 创建填充缓冲区：b_v，存储整数类型数据，大小为 kTotalSize
  PaddedBuffer<int> b_v(kTotalSize);

  // 填充缓冲区 a_v：将索引 i 存储到 a_v 的第 i 个位置
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
  }

  // 创建简单 IR 评估器：使用 stmt 语句，操作缓冲区 a_buf 和 b_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  // 执行评估器：将 a_v 的数据转换为 b_v 的数据
  ir_eval(a_v, b_v);

  // 断言：验证转换前后数据是否正确
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), -static_cast<float>(i));
  }
}

// 定义测试案例：ATen::negFloat
TEST(ATen, negFloat) {
  // 定义常量：数组大小为 128
  const int kTotalSize = 128;
  // 创建名为 A 的缓冲区句柄，存储浮点数类型的数据
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为 B 的缓冲区句柄，存储浮点数类型的数据
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  // 创建变量句柄：索引 index，类型为整数
  VarHandle index = VarHandle("index", kInt);
  // 加载缓冲区 A 中索引为 index 的元素，存储为表达式 load_a
  ExprHandle load_a = a_buf.load(index);
  // 创建表达式：0 减去 load_a，存储为表达式 to_float
  ExprHandle to_float = Sub::make(0, load_a);
  // 创建存储语句：将 to_float 存储到缓冲区 B 的索引为 index 的位置
  StmtPtr store_b = b_buf.store({index}, to_float);
  // 创建循环语句：遍历索引 index 从 0 到 kTotalSize-1，执行 store_b
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建填充缓冲区：a_v，存储浮点数类型数据，大小为 kTotalSize
  PaddedBuffer<float> a_v(kTotalSize);
  // 创建填充缓冲区：b_v，存储浮点数类型数据，大小为 kTotalSize
  PaddedBuffer<float> b_v(kTotalSize);

  // 填充缓冲区 a_v：将索引 i 存储到 a_v 的第 i 个位置
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
  }

  // 创建简单 IR 评估器：使用 stmt 语句，操作缓冲区 a_buf 和 b_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  // 执行评估器：将 a_v 的数据转换为 b_v 的数据
  ir_eval(a_v, b_v);

  // 断言：验证转换前后数据是否正确
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b
# 定义名为 TEST 的测试函数，用于测试 ATen 模块中的 addInt 功能
TEST(ATen, addInt) {
  # 定义常量 kTotalSize，表示数组大小为 128
  const int kTotalSize = 128;
  # 创建名为 a_buf 的缓冲区对象，存储整数类型数据
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  # 创建名为 b_buf 的缓冲区对象，存储整数类型数据
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  # 创建名为 c_buf 的缓冲区对象，存储整数类型数据
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);
  # 创建名为 d_buf 的缓冲区对象，存储整数类型数据
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kInt);

  # 创建名为 index 的变量对象，表示循环索引，类型为整数
  VarHandle index = VarHandle("index", kInt);
  # 创建 load_a 表达式，加载 a_buf 中 index 处的值
  ExprHandle load_a = a_buf.load(index);
  # 创建 load_b 表达式，加载 b_buf 中 index 处的值
  ExprHandle load_b = b_buf.load(index);
  # 创建 load_c 表达式，加载 c_buf 中 index 处的值
  ExprHandle load_c = c_buf.load(index);
  # 创建 store_d 语句，将 load_a 加上 load_b 乘以 load_c 的结果存储到 d_buf 的 index 处
  StmtPtr store_d = d_buf.store({index}, load_a + load_b * load_c);
  # 创建 stmt 语句，使用 For 循环遍历索引 index 从 0 到 kTotalSize，执行 store_d 操作
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  # 创建整数类型的 PaddedBuffer 对象 a_v，大小为 kTotalSize
  PaddedBuffer<int> a_v(kTotalSize);
  # 创建整数类型的 PaddedBuffer 对象 b_v，大小为 kTotalSize
  PaddedBuffer<int> b_v(kTotalSize);
  # 创建整数类型的 PaddedBuffer 对象 c_v，大小为 kTotalSize
  PaddedBuffer<int> c_v(kTotalSize);
  # 创建整数类型的 PaddedBuffer 对象 d_v，大小为 kTotalSize
  PaddedBuffer<int> d_v(kTotalSize);

  # 使用循环为 a_v, b_v, c_v 赋值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  # 创建 SimpleIREvaluator 对象 ir_eval，用于评估 stmt 所定义的简单 IR 表达式
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  # 使用 ir_eval 对象执行表达式求值，将结果存储到 d_v 中
  ir_eval(a_v, b_v, c_v, d_v);

  # 验证每个索引处的值是否符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i));
  }
}

# 定义名为 TEST 的测试函数，用于测试 ATen 模块中的 addFloat 功能
TEST(ATen, addFloat) {
  # 定义常量 kTotalSize，表示数组大小为 128
  const int kTotalSize = 128;
  # 创建名为 a_buf 的缓冲区对象，存储浮点数类型数据
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  # 创建名为 b_buf 的缓冲区对象，存储浮点数类型数据
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  # 创建名为 c_buf 的缓冲区对象，存储浮点数类型数据
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);
  # 创建名为 d_buf 的缓冲区对象，存储浮点数类型数据
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kFloat);

  # 创建名为 index 的变量对象，表示循环索引，类型为整数
  VarHandle index = VarHandle("index", kInt);
  # 创建 load_a 表达式，加载 a_buf 中 index 处的值
  ExprHandle load_a = a_buf.load(index);
  # 创建 load_b 表达式，加载 b_buf 中 index 处的值
  ExprHandle load_b = b_buf.load(index);
  # 创建 load_c 表达式，加载 c_buf 中 index 处的值
  ExprHandle load_c = c_buf.load(index);
  # 创建 store_d 语句，将 load_a 加上 load_b 乘以 load_c 的结果存储到 d_buf 的 index 处
  StmtPtr store_d = d_buf.store({index}, load_a + load_b * load_c);
  # 创建 stmt 语句，使用 For 循环遍历索引 index 从 0 到 kTotalSize，执行 store_d 操作
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  # 创建浮点数类型的 PaddedBuffer 对象 a_v，大小为 kTotalSize
  PaddedBuffer<float> a_v(kTotalSize);
  # 创建浮点数类型的 PaddedBuffer 对象 b_v，大小为 kTotalSize
  PaddedBuffer<float> b_v(kTotalSize);
  # 创建浮点数类型的 PaddedBuffer 对象 c_v，大小为 kTotalSize
  PaddedBuffer<float> c_v(kTotalSize);
  # 创建浮点数类型的 PaddedBuffer 对象 d_v，大小为 kTotalSize
  PaddedBuffer<float> d_v(kTotalSize);

  # 使用循环为 a_v, b_v, c_v 赋值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  # 创建 SimpleIREvaluator 对象 ir_eval，用于评估 stmt 所定义的简单 IR 表达式
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  # 使用 ir_eval 对象执行表达式求值，将结果存储到 d_v 中
  ir_eval(a_v, b_v, c_v, d_v);

  # 验证每个索引处的值是否符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i));
  }
}
# 定义名为 TEST 的测试函数，用于测试 ATen 模块中的 subInt 功能
TEST(ATen, subInt) {
  # 声明并初始化常量 kTotalSize，表示缓冲区大小为 128
  const int kTotalSize = 128;
  # 创建名为 a_buf 的整型缓冲区，大小为 kTotalSize
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  # 创建名为 b_buf 的整型缓冲区，大小为 kTotalSize
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  # 创建名为 c_buf 的整型缓冲区，大小为 kTotalSize
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);
  # 创建名为 d_buf 的整型缓冲区，大小为 kTotalSize
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kInt);

  # 声明名为 index 的整型变量
  VarHandle index = VarHandle("index", kInt);
  # 从 a_buf 中加载 index 处的值，存储到 load_a 中
  ExprHandle load_a = a_buf.load(index);
  # 从 b_buf 中加载 index 处的值，存储到 load_b 中
  ExprHandle load_b = b_buf.load(index);
  # 从 c_buf 中加载 index 处的值，存储到 load_c 中
  ExprHandle load_c = c_buf.load(index);
  # 计算 load_a - load_b * load_c 的表达式，并存储到 d_buf 的 index 处
  StmtPtr store_d = d_buf.store({index}, load_a - load_b * load_c);
  # 使用 For 循环创建 stmt 语句，遍历 index 从 0 到 kTotalSize
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  # 创建整型 PaddedBuffer 类型的 a_v，大小为 kTotalSize
  PaddedBuffer<int> a_v(kTotalSize);
  # 创建整型 PaddedBuffer 类型的 b_v，大小为 kTotalSize
  PaddedBuffer<int> b_v(kTotalSize);
  # 创建整型 PaddedBuffer 类型的 c_v，大小为 kTotalSize
  PaddedBuffer<int> c_v(kTotalSize);
  # 创建整型 PaddedBuffer 类型的 d_v，大小为 kTotalSize
  PaddedBuffer<int> d_v(kTotalSize);

  # 循环遍历范围为 kTotalSize 的整数 i
  for (const auto i : c10::irange(kTotalSize)) {
    # 初始化 a_v(i) 为 i
    a_v(i) = i;
    # 初始化 b_v(i) 为 2 * i + 1
    b_v(i) = 2 * i + 1;
    # 初始化 c_v(i) 为 3 * i + 2
    c_v(i) = 3 * i + 2;
  }

  # 创建 SimpleIREvaluator 对象 ir_eval，用于评估 stmt，传入缓冲区 a_buf, b_buf, c_buf, d_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  # 执行 ir_eval，计算并存储结果到 d_v
  ir_eval(a_v, b_v, c_v, d_v);

  # 再次循环遍历范围为 kTotalSize 的整数 i
  for (const auto i : c10::irange(kTotalSize)) {
    # 断言 a_v(i) 的值等于 i
    ASSERT_EQ(a_v(i), i);
    # 断言 b_v(i) 的值等于 2 * i + 1
    ASSERT_EQ(b_v(i), 2 * i + 1);
    # 断言 c_v(i) 的值等于 3 * i + 2
    ASSERT_EQ(c_v(i), 3 * i + 2);
    # 断言 d_v(i) 的值等于 a_v(i) - b_v(i) * c_v(i)
    ASSERT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i));
  }
}

# 定义名为 TEST 的测试函数，用于测试 ATen 模块中的 subFloat 功能
TEST(ATen, subFloat) {
  # 声明并初始化常量 kTotalSize，表示缓冲区大小为 128
  const int kTotalSize = 128;
  # 创建名为 a_buf 的单精度浮点型缓冲区，大小为 kTotalSize
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  # 创建名为 b_buf 的单精度浮点型缓冲区，大小为 kTotalSize
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  # 创建名为 c_buf 的单精度浮点型缓冲区，大小为 kTotalSize
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);
  # 创建名为 d_buf 的单精度浮点型缓冲区，大小为 kTotalSize
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kFloat);

  # 声明名为 index 的整型变量
  VarHandle index = VarHandle("index", kInt);
  # 从 a_buf 中加载 index 处的值，存储到 load_a 中
  ExprHandle load_a = a_buf.load(index);
  # 从 b_buf 中加载 index 处的值，存储到 load_b 中
  ExprHandle load_b = b_buf.load(index);
  # 从 c_buf 中加载 index 处的值，存储到 load_c 中
  ExprHandle load_c = c_buf.load(index);
  # 计算 load_a - load_b * load_c 的表达式，并存储到 d_buf 的 index 处
  StmtPtr store_d = d_buf.store({index}, load_a - load_b * load_c);
  # 使用 For 循环创建 stmt 语句，遍历 index 从 0 到 kTotalSize
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  # 创建单精度浮点型 PaddedBuffer 类型的 a_v，大小为 kTotalSize
  PaddedBuffer<float> a_v(kTotalSize);
  # 创建单精度浮点型 PaddedBuffer 类型的 b_v，大小为 kTotalSize
  PaddedBuffer<float> b_v(kTotalSize);
  # 创建单精度浮点型 PaddedBuffer 类型的 c_v，大小为 kTotalSize
  PaddedBuffer<float> c_v(kTotalSize);
  # 创建单精度浮点型 PaddedBuffer 类型的 d_v，大小为 kTotalSize
  PaddedBuffer<float> d_v(kTotalSize);

  # 循环遍历范围为 kTotalSize 的整数 i
  for (const auto i : c10::irange(kTotalSize)) {
    # 初始化 a_v(i) 为 i
    a_v(i) = i;
    # 初始化 b_v(i) 为 2 * i + 1
    b_v(i) = 2 * i + 1;
    # 初始化 c_v(i) 为 3 * i + 2
    c_v(i) = 3 * i + 2;
  }

  # 创建 SimpleIREvaluator 对象 ir_eval，用于评估 stmt，传入缓冲区 a_buf, b_buf, c_buf, d_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  # 执行 ir_eval，计算并存储结果到 d_v
  ir_eval(a_v, b_v, c_v, d_v);

  # 再次循环遍历范围为 kTotalSize 的整数 i
  for (const auto i : c10::irange(kTotalSize)) {
    # 断言 a_v(i) 的值等于 i
    ASSERT_EQ(a_v(i), i);
    # 断言 b_v(i) 的值等于 2 * i + 1
    ASSERT_EQ(b_v(i), 2 * i + 1);
    # 断言 c_v(i) 的值等于 3 * i + 2
    ASSERT_EQ(c_v(i), 3 * i + 2);
    # 断言 d_v(i) 的值等于 a_v(i) - b_v(i) * c_v(i)
    ASSERT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i));
  }
}
TEST(ATen, lerp) {
  // 定义总大小常量
  const int kTotalSize = 128;
  // 定义四个缓冲区，并分别初始化为 kTotalSize 大小的浮点数缓冲区
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kFloat);

  // 定义整型变量 index
  VarHandle index = VarHandle("index", kInt);
  // 从 a_buf 中加载索引位置的数据
  ExprHandle load_a = a_buf.load(index);
  // 从 b_buf 中加载索引位置的数据
  ExprHandle load_b = b_buf.load(index);
  // 从 c_buf 中加载索引位置的数据
  ExprHandle load_c = c_buf.load(index);
  // 计算并存储到 d_buf 中：load_a + load_c * (load_b - load_a)
  StmtPtr store_d = d_buf.store({index}, load_a + load_c * (load_b - load_a));
  // 构造一个循环语句，遍历索引从 0 到 kTotalSize，执行 store_d
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  // 初始化大小为 kTotalSize 的浮点数填充缓冲区 a_v, b_v, c_v, d_v
  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  // 对每个 i 在 0 到 kTotalSize 的范围内，初始化填充缓冲区的值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  // 构造一个简单的 IR 评估器，用于执行 stmt，并传入 a_buf, b_buf, c_buf, d_buf 作为参数
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  // 执行 IR 评估器，计算结果并存储在相应的填充缓冲区中
  ir_eval(a_v, b_v, c_v, d_v);

  // 验证填充缓冲区的值是否符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + c_v(i) * (b_v(i) - a_v(i)));
  }
}

TEST(ATen, addcmulInt) {
  // 定义总大小常量
  const int kTotalSize = 128;
  // 定义五个缓冲区，并分别初始化为 kTotalSize 大小的整数缓冲区
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kInt);
  BufHandle e_buf("E", {ExprHandle(kTotalSize)}, kInt);

  // 定义整型变量 index
  VarHandle index = VarHandle("index", kInt);
  // 从 a_buf 中加载索引位置的数据
  ExprHandle load_a = a_buf.load(index);
  // 从 b_buf 中加载索引位置的数据
  ExprHandle load_b = b_buf.load(index);
  // 从 c_buf 中加载索引位置的数据
  ExprHandle load_c = c_buf.load(index);
  // 从 d_buf 中加载索引位置的数据
  ExprHandle load_d = d_buf.load(index);
  // 计算并存储到 e_buf 中：load_a + load_b * load_c * load_d
  StmtPtr store_e = e_buf.store({index}, load_a + load_b * load_c * load_d);
  // 构造一个循环语句，遍历索引从 0 到 kTotalSize，执行 store_e
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_e);

  // 初始化大小为 kTotalSize 的整数填充缓冲区 a_v, b_v, c_v, d_v, e_v
  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);
  PaddedBuffer<int> e_v(kTotalSize);

  // 对每个 i 在 0 到 kTotalSize 的范围内，初始化填充缓冲区的值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
    d_v(i) = 5 * i + 3;
  }

  // 构造一个简单的 IR 评估器，用于执行 stmt，并传入 a_buf, b_buf, c_buf, d_buf, e_buf 作为参数
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf, e_buf});
  // 执行 IR 评估器，计算结果并存储在相应的填充缓冲区中
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  // 验证填充缓冲区的值是否符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), 5 * i + 3);
    ASSERT_EQ(e_v(i), a_v(i) + b_v(i) * c_v(i) * d_v(i));
  }
}
TEST(ATen, mulFloat) {
  // 定义常量，表示数组的总大小
  const int kTotalSize = 128;
  // 创建缓冲区对象，用于存储浮点数数组 A、B 和 C
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);

  // 定义变量对象 index，用于循环索引
  VarHandle index = VarHandle("index", kInt);
  // 加载数组 A 和 B 中的数据到表达式对象 load_a 和 load_b 中
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  // 创建存储语句 store_c，将 load_a 和 load_b 相乘的结果存储到数组 C 中
  StmtPtr store_c = c_buf.store({index}, load_a * load_b);
  // 创建循环语句 stmt，用于迭代 index 从 0 到 kTotalSize，并执行 store_c
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  // 创建 PaddedBuffer 对象，用于存储测试数据和结果数据的数组
  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  // 初始化数组 a_v 和 b_v 中的数据
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  // 创建 SimpleIREvaluator 对象，执行 IR 语句 stmt，传入数组 a_buf、b_buf、c_buf 作为参数
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  // 执行 IR 语句，计算并存储结果到数组 c_v 中
  ir_eval(a_v, b_v, c_v);

  // 验证计算结果是否正确
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), a_v(i) * b_v(i));
  }
}
    b_v(i) = 2 * i + 1;

# 将 `b_v` 数组的第 `i` 个元素赋值为 `2 * i + 1`


  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});

# 使用 `stmt` 和 `{a_buf, b_buf, c_buf}` 初始化 `SimpleIREvaluator` 对象 `ir_eval`


  ir_eval(a_v, b_v, c_v);

# 调用 `ir_eval` 对象的方法，传入 `a_v`, `b_v`, `c_v` 作为参数进行求值


  for (const auto i : c10::irange(kTotalSize)) {

# 对于 `kTotalSize` 范围内的每个 `i`，执行以下循环


    ASSERT_EQ(a_v(i), i);

# 断言 `a_v` 数组的第 `i` 个元素等于 `i`


    ASSERT_EQ(b_v(i), 2 * i + 1);

# 断言 `b_v` 数组的第 `i` 个元素等于 `2 * i + 1`


    ASSERT_EQ(c_v(i), a_v(i) * b_v(i));

# 断言 `c_v` 数组的第 `i` 个元素等于 `a_v(i) * b_v(i)`
}

// 定义一个名为 TEST 的测试用例，测试整数除法操作
TEST(ATen, divInt) {
  // 定义常量，缓冲区及其大小
  const int kTotalSize = 128;
  // 定义缓冲区 A、B、C，每个缓冲区包含 kTotalSize 个整数元素
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);

  // 定义变量 index，用于循环索引
  VarHandle index = VarHandle("index", kInt);
  // 加载缓冲区 A 和 B 中的数据到 load_a 和 load_b
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  // 创建一个存储语句 store_c，将 load_a 除以 load_b 的结果存储到缓冲区 C 中
  StmtPtr store_c = c_buf.store({index}, load_a / load_b);
  // 创建一个循环语句 stmt，遍历 index 从 0 到 kTotalSize，并执行 store_c
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  // 创建 PaddedBuffer 对象，用于存储整数类型数据
  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  // 初始化缓冲区 a_v 和 b_v 的数据
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  // 创建 SimpleIREvaluator 对象 ir_eval，评估 stmt，并传入缓冲区 a_buf、b_buf、c_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  // 执行评估，将结果存储到 c_v
  ir_eval(a_v, b_v, c_v);

  // 断言检查，确保 a_v、b_v 和 c_v 中的数据符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), 2 * i + 1);
    ASSERT_EQ(b_v(i), i + 1);
    ASSERT_EQ(c_v(i), a_v(i) / b_v(i));
  }
}

// 定义一个名为 TEST 的测试用例，测试浮点数除法操作
TEST(ATen, divFloat) {
  // 定义常量，缓冲区及其大小
  const int kTotalSize = 128;
  // 定义缓冲区 A、B、C，每个缓冲区包含 kTotalSize 个浮点数元素
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);

  // 定义变量 index，用于循环索引
  VarHandle index = VarHandle("index", kInt);
  // 加载缓冲区 A 和 B 中的数据到 load_a 和 load_b
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  // 创建一个存储语句 store_c，将 load_a 除以 load_b 的结果存储到缓冲区 C 中
  StmtPtr store_c = c_buf.store({index}, load_a / load_b);
  // 创建一个循环语句 stmt，遍历 index 从 0 到 kTotalSize，并执行 store_c
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  // 创建 PaddedBuffer 对象，用于存储浮点数类型数据
  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  // 初始化缓冲区 a_v 和 b_v 的数据
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  // 创建 SimpleIREvaluator 对象 ir_eval，评估 stmt，并传入缓冲区 a_buf、b_buf、c_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  // 执行评估，将结果存储到 c_v
  ir_eval(a_v, b_v, c_v);

  // 断言检查，确保 a_v、b_v 和 c_v 中的数据符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), 2 * i + 1);
    ASSERT_EQ(b_v(i), i + 1);
    ASSERT_EQ(c_v(i), a_v(i) / b_v(i));
  }
}

// 定义一个名为 TEST 的测试用例，测试整数最大值操作
TEST(ATen, maxInt) {
  // 定义常量，缓冲区及其大小
  const int kTotalSize = 128;
  // 定义缓冲区 A、B、C，每个缓冲区包含 kTotalSize 个整数元素
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);

  // 定义变量 index，用于循环索引
  VarHandle index = VarHandle("index", kInt);
  // 加载缓冲区 A 和 B 中的数据到 load_a 和 load_b
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  // 创建一个存储语句 store_c，将 load_a 和 load_b 中的较大值存储到缓冲区 C 中
  StmtPtr store_c = c_buf.store({index}, Max::make(load_a, load_b, true));
  // 创建一个循环语句 stmt，遍历 index 从 0 到 kTotalSize，并执行 store_c
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  // 创建 PaddedBuffer 对象，用于存储整数类型数据
  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  // 初始化缓冲区 a_v 和 b_v 的数据
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  // 创建 SimpleIREvaluator 对象 ir_eval，评估 stmt，并传入缓冲区 a_buf、b_buf、c_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  // 执行评估，将结果存储到 c_v
  ir_eval(a_v, b_v, c_v);

  // 断言检查，确保 a_v、b_v 和 c_v 中的数据符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::max(a_v(i), b_v(i)));
  }
}
// 定义一个名为 maxFloat 的测试用例函数
TEST(ATen, maxFloat) {
  // 定义常量 kTotalSize，表示缓冲区大小为 128
  const int kTotalSize = 128;
  // 创建一个名为 a_buf 的缓冲区，存储浮点数类型数据
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建一个名为 b_buf 的缓冲区，存储浮点数类型数据
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  // 创建一个名为 c_buf 的缓冲区，存储浮点数类型数据
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);

  // 创建一个名为 index 的变量句柄，表示循环索引
  VarHandle index = VarHandle("index", kInt);
  // 加载 a_buf 中 index 处的值，作为表达式 load_a
  ExprHandle load_a = a_buf.load(index);
  // 加载 b_buf 中 index 处的值，作为表达式 load_b
  ExprHandle load_b = b_buf.load(index);
  // 创建一个语句指针 store_c，将 Max 函数应用于 load_a 和 load_b，并将结果存储到 c_buf 的 index 处
  StmtPtr store_c = c_buf.store({index}, Max::make(load_a, load_b, true));
  // 创建一个 for 循环语句 stmt，循环范围从 0 到 kTotalSize，每次执行 store_c
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  // 创建一个名为 a_v 的填充缓冲区，存储浮点数类型数据，大小为 kTotalSize
  PaddedBuffer<float> a_v(kTotalSize);
  // 创建一个名为 b_v 的填充缓冲区，存储浮点数类型数据，大小为 kTotalSize
  PaddedBuffer<float> b_v(kTotalSize);
  // 创建一个名为 c_v 的填充缓冲区，存储浮点数类型数据，大小为 kTotalSize
  PaddedBuffer<float> c_v(kTotalSize);

  // 填充 a_v 和 b_v 缓冲区的数据
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  // 创建一个 SimpleIREvaluator 对象 ir_eval，用于执行 stmt，并传入 a_buf、b_buf、c_buf 作为参数
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  // 执行 ir_eval，计算结果存储到 a_v、b_v、c_v 缓冲区中
  ir_eval(a_v, b_v, c_v);

  // 验证计算结果的正确性
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::fmax(a_v(i), b_v(i)));
  }
}

// 定义一个名为 minInt 的测试用例函数
TEST(ATen, minInt) {
  // 定义常量 kTotalSize，表示缓冲区大小为 128
  const int kTotalSize = 128;
  // 创建一个名为 a_buf 的缓冲区，存储整数类型数据
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  // 创建一个名为 b_buf 的缓冲区，存储整数类型数据
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  // 创建一个名为 c_buf 的缓冲区，存储整数类型数据
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);

  // 创建一个名为 index 的变量句柄，表示循环索引
  VarHandle index = VarHandle("index", kInt);
  // 加载 a_buf 中 index 处的值，作为表达式 load_a
  ExprHandle load_a = a_buf.load(index);
  // 加载 b_buf 中 index 处的值，作为表达式 load_b
  ExprHandle load_b = b_buf.load(index);
  // 创建一个语句指针 store_c，将 Min 函数应用于 load_a 和 load_b，并将结果存储到 c_buf 的 index 处
  StmtPtr store_c = c_buf.store({index}, Min::make(load_a, load_b, true));
  // 创建一个 for 循环语句 stmt，循环范围从 0 到 kTotalSize，每次执行 store_c
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  // 创建一个名为 a_v 的填充缓冲区，存储整数类型数据，大小为 kTotalSize
  PaddedBuffer<int> a_v(kTotalSize);
  // 创建一个名为 b_v 的填充缓冲区，存储整数类型数据，大小为 kTotalSize
  PaddedBuffer<int> b_v(kTotalSize);
  // 创建一个名为 c_v 的填充缓冲区，存储整数类型数据，大小为 kTotalSize
  PaddedBuffer<int> c_v(kTotalSize);

  // 填充 a_v 和 b_v 缓冲区的数据
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  // 创建一个 SimpleIREvaluator 对象 ir_eval，用于执行 stmt，并传入 a_buf、b_buf、c_buf 作为参数
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  // 执行 ir_eval，计算结果存储到 a_v、b_v、c_v 缓冲区中
  ir_eval(a_v, b_v, c_v);

  // 验证计算结果的正确性
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::min(a_v(i), b_v(i)));
  }
}

// 定义一个名为 minFloat 的测试用例函数
TEST(ATen, minFloat) {
  // 定义常量 kTotalSize，表示缓冲区大小为 128
  const int kTotalSize = 128;
  // 创建一个名为 a_buf 的缓冲区，存储浮点数类型数据
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建一个名为 b_buf 的缓冲区，存储浮点数类型数据
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  // 创建一个名为 c_buf 的缓冲区，存储浮点数类型数据
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);

  // 创建一个名为 index 的变量句柄，表示循环索引
  VarHandle index = VarHandle("index", kInt);
  // 加载 a_buf 中 index 处的值，作为表达式 load_a
  ExprHandle load_a = a_buf.load(index);
  // 加载 b_buf 中 index 处的值，作为表达式 load_b
  ExprHandle load_b = b_buf.load(index);
  // 创建一个语句指针 store_c，将 Min 函数应用于 load_a 和 load_b，并将结果存储到 c_buf 的 index 处
  StmtPtr store_c = c_buf.store({index}, Min::make(load_a, load_b, true));
  // 创建一个 for 循环语句 stmt，循环范围从 0 到 kTotalSize，每次执行 store_c
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  // 创建一个名为 a_v 的填充缓冲区，存储浮点数类型数据，大小为 kTotalSize
  PaddedBuffer<float> a_v(kTotalSize);
  // 创建一个名为 b_v 的填充缓冲区，存储浮点数类型数据，大小为 kTotalSize
  PaddedBuffer<float> b_v(kTotalSize);
  // 创建一个名为 c_v 的填充缓冲区，存储浮点数类型数据，大小为 kTotalSize
void __ubsan_ignore_float_divide_by_zero__ testATenreciprocal() {
  // 定义总大小常量为128
  const int kTotalSize = 128;
  // 创建名为"A"的缓冲区，包含kTotalSize个元素，元素类型为float
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为"B"的缓冲区，包含kTotalSize个元素，元素类型为float
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  // 创建名为index的变量句柄，类型为整型
  VarHandle index = VarHandle("index", kInt);
  // 加载a_buf中索引为index的元素值
  ExprHandle load_a = a_buf.load(index);
  // 构造将1.0除以load_a结果存储到b_buf中索引为index位置的语句
  StmtPtr store_b = b_buf.store({index}, FloatImm::make(1.0f) / load_a);
  // 创建一个循环语句，遍历index从0到kTotalSize，每次执行store_b语句
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建一个大小为kTotalSize的float型填充缓冲区a_v
  PaddedBuffer<float> a_v(kTotalSize);
  // 创建一个大小为kTotalSize的float型填充缓冲区b_v
  PaddedBuffer<float> b_v(kTotalSize);

  // 初始化a_v缓冲区，使其值为索引i的值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
  }

  // 创建一个简单的IR评估器，使用stmt语句和a_buf、b_buf缓冲区
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  // 评估IR并将结果存储到a_v和b_v缓冲区中
  ir_eval(a_v, b_v);

  // 验证每个索引i处a_v和b_v的值是否符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 1.0f / i);
  }
}

TEST(ATen, reluInt) {
  // 定义总大小常量为128
  const int kTotalSize = 128;
  // 创建名为"A"的缓冲区，包含kTotalSize个元素，元素类型为整型
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  // 创建名为"B"的缓冲区，包含kTotalSize个元素，元素类型为整型
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  // 创建名为index的变量句柄，类型为整型
  VarHandle index = VarHandle("index", kInt);
  // 加载a_buf中索引为index的元素值
  ExprHandle load_a = a_buf.load(index);
  // 构造将load_a与0比较取最大值后存储到b_buf中索引为index位置的语句
  StmtPtr store_b = b_buf.store({index}, Max::make(load_a, 0, false));
  // 创建一个循环语句，遍历index从0到kTotalSize，每次执行store_b语句
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建一个大小为kTotalSize的整型填充缓冲区a_v
  PaddedBuffer<int> a_v(kTotalSize);
  // 创建一个大小为kTotalSize的整型填充缓冲区b_v
  PaddedBuffer<int> b_v(kTotalSize);

  // 初始化a_v缓冲区，使其值为索引i - 64的值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i - 64;
  }

  // 创建一个简单的IR评估器，使用stmt语句和a_buf、b_buf缓冲区
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  // 评估IR并将结果存储到a_v和b_v缓冲区中
  ir_eval(a_v, b_v);

  // 验证每个索引i处a_v和b_v的值是否符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i - 64);
    ASSERT_EQ(b_v(i), std::max(a_v(i), 0));
  }
}

TEST(ATen, reluFloat) {
  // 定义总大小常量为128
  const int kTotalSize = 128;
  // 创建名为"A"的缓冲区，包含kTotalSize个元素，元素类型为float
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为"B"的缓冲区，包含kTotalSize个元素，元素类型为float
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  // 创建名为index的变量句柄，类型为整型
  VarHandle index = VarHandle("index", kInt);
  // 加载a_buf中索引为index的元素值
  ExprHandle load_a = a_buf.load(index);
  // 构造将load_a与0比较取最大值后存储到b_buf中索引为index位置的语句
  StmtPtr store_b = b_buf.store(
      {index}, Max::make(load_a, 0, false) // relu does not propagate nans
  );
  // 创建一个循环语句，遍历index从0到kTotalSize，每次执行store_b语句
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建一个大小为kTotalSize的float型填充缓冲区a_v
  PaddedBuffer<float> a_v(kTotalSize);
  // 创建一个大小为kTotalSize的float型填充缓冲区b_v
  PaddedBuffer<float> b_v(kTotalSize);

  // 初始化a_v缓冲区，使其值为索引i - 64的值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i - 64;
  }

  // 创建一个简单的IR评估器，使用stmt语句和a_buf、b_buf缓冲区
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  // 评估IR并将结果存储到a_v和b_v缓冲区中
  ir_eval(a_v, b_v);

  // 验证每个索引i处a_v和b_v的值是否符合预期
  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i - 64);
    ASSERT_EQ(b_v(i), std::fmax(a_v(i), 0));
  }
}

TEST(ATen, logFloat) {
  // 定义总大小常量为128
  const int kTotalSize = 128;
  // 创建名为"A"的缓冲区，包含kTotalSize个元素，元素类型为float
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为"B"的缓冲区，包含kTotalSize个元素，元素类型为float
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  // 创建名为index的变量句柄，类型为整型
  VarHandle index = VarHandle("index", kInt);
  // 加载a_buf中索引为index的元素值
  ExprHandle load_a = a_buf.load(index);
  // 构造将load_a的对数值存储到b_buf中索引为index位置的语句
  StmtPtr store_b = b_buf.store({index}, log(load_a));
  // 创建一个循环语句，遍历index从0到kTotalSize，每次执行store_b语句
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建一个大小为kTotalSize的float型填充缓冲区a_v
  PaddedBuffer<float> a_v(kTotalSize);
  // 创建一个大小为kTotalSize
    ASSERT_EQ(b_v(i), std::log(a_v(i)));



// 断言：验证 b_v(i) 的值是否等于 std::log(a_v(i)) 的返回值
ASSERT_EQ(b_v(i), std::log(a_v(i)));


这行代码是一个断言语句，用于在测试中检查条件是否满足。`ASSERT_EQ` 是一个宏或函数，用于比较两个值是否相等。在这里，它比较 `b_v(i)` 和 `std::log(a_v(i))` 的返回值是否相等，`std::log` 是 C++ 标准库中计算自然对数的函数。
}

// 定义一个名为 "fastLogFloat" 的测试用例
TEST(ATen, fastLogFloat) {
  // 声明常量整数 kTotalSize 并赋值为 128
  const int kTotalSize = 128;
  // 创建名为 a_buf 的缓冲区，用于存储浮点数，大小为 kTotalSize
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为 b_buf 的缓冲区，用于存储浮点数，大小为 kTotalSize
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  // 声明名为 index 的变量句柄，类型为整型
  VarHandle index = VarHandle("index", kInt);
  // 从 a_buf 中加载 index 处的值，并将结果赋给 load_a
  ExprHandle load_a = a_buf.load(index);
  // 创建一个语句指针 store_b，用于将 fast_log(load_a) 存储到 b_buf 的 index 处
  StmtPtr store_b = b_buf.store({index}, fast_log(load_a));
  // 创建一个循环语句 stmt，用于遍历 index 从 0 到 kTotalSize，并执行 store_b
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建一个大小为 kTotalSize 的 float 类型填充缓冲区 a_v
  PaddedBuffer<float> a_v(kTotalSize);
  // 创建一个大小为 kTotalSize 的 float 类型填充缓冲区 b_v
  PaddedBuffer<float> b_v(kTotalSize);

  // 遍历 c10::irange(kTotalSize)，为 a_v 中的每个元素赋予标准正态分布的随机值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  // 创建一个 SimpleIREvaluator 对象 ir_eval，用于评估 stmt 的执行结果，依赖 a_buf 和 b_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  // 使用 a_v 和 b_v 运行 ir_eval
  ir_eval(a_v, b_v);

  // 遍历 c10::irange(kTotalSize)，对比 b_v 和 a_v 经过 std::log() 处理的结果
  for (const auto i : c10::irange(kTotalSize)) {
    auto test = b_v(i);
    auto ref = std::log(a_v(i));
    // 如果 ref 是 NaN，则断言 test 也是 NaN
    if (std::isnan(ref)) {
      ASSERT_EQ(std::isnan(test), true);
    } else {
      // 否则，断言 test 与 ref 相等（浮点数相等）
      ASSERT_FLOAT_EQ(test, ref);
    }
  }
}

// 定义一个名为 "fastTanhFloat" 的测试用例
TEST(ATen, fastTanhFloat) {
  // 声明常量整数 kTotalSize 并赋值为 128
  const int kTotalSize = 128;
  // 创建名为 a_buf 的缓冲区，用于存储浮点数，大小为 kTotalSize
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为 b_buf 的缓冲区，用于存储浮点数，大小为 kTotalSize
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  // 声明名为 index 的变量句柄，类型为整型
  VarHandle index = VarHandle("index", kInt);
  // 从 a_buf 中加载 index 处的值，并将结果赋给 load_a
  ExprHandle load_a = a_buf.load(index);
  // 创建一个语句指针 store_b，用于将 fast_tanh(load_a) 存储到 b_buf 的 index 处
  StmtPtr store_b = b_buf.store({index}, fast_tanh(load_a));
  // 创建一个循环语句 stmt，用于遍历 index 从 0 到 kTotalSize，并执行 store_b
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建一个大小为 kTotalSize 的 float 类型填充缓冲区 a_v
  PaddedBuffer<float> a_v(kTotalSize);
  // 创建一个大小为 kTotalSize 的 float 类型填充缓冲区 b_v
  PaddedBuffer<float> b_v(kTotalSize);

  // 遍历 c10::irange(kTotalSize)，为 a_v 中的每个元素赋予标准正态分布的随机值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  // 创建一个 SimpleIREvaluator 对象 ir_eval，用于评估 stmt 的执行结果，依赖 a_buf 和 b_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  // 使用 a_v 和 b_v 运行 ir_eval
  ir_eval(a_v, b_v);

  // 遍历 c10::irange(kTotalSize)，对比 b_v 和 a_v 经过 std::tanh() 处理的结果
  for (const auto i : c10::irange(kTotalSize)) {
    auto test = b_v(i);
    auto ref = std::tanh(a_v(i));
    // 如果 ref 是 NaN，则断言 test 也是 NaN
    if (std::isnan(ref)) {
      ASSERT_EQ(std::isnan(test), true);
    } else {
      // 否则，断言 test 与 ref 相等（浮点数相等），允许误差为 1e-6
      ASSERT_NEAR(test, ref, 1e-6);
    }
  }
}

// 定义一个名为 "fastSigmoidFloat" 的测试用例
TEST(ATen, fastSigmoidFloat) {
  // 声明常量整数 kTotalSize 并赋值为 128
  const int kTotalSize = 128;
  // 创建名为 a_buf 的缓冲区，用于存储浮点数，大小为 kTotalSize
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  // 创建名为 b_buf 的缓冲区，用于存储浮点数，大小为 kTotalSize
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  // 声明名为 index 的变量句柄，类型为整型
  VarHandle index = VarHandle("index", kInt);
  // 从 a_buf 中加载 index 处的值，并将结果赋给 load_a
  ExprHandle load_a = a_buf.load(index);
  // 创建一个语句指针 store_b，用于将 fast_sigmoid(load_a) 存储到 b_buf 的 index 处
  StmtPtr store_b = b_buf.store({index}, fast_sigmoid(load_a));
  // 创建一个循环语句 stmt，用于遍历 index 从 0 到 kTotalSize，并执行 store_b
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  // 创建一个大小为 kTotalSize 的 float 类型填充缓冲区 a_v
  PaddedBuffer<float> a_v(kTotalSize);
  // 创建一个大小为 kTotalSize 的 float 类型填充缓冲区 b_v
  PaddedBuffer<float> b_v(kTotalSize);

  // 遍历 c10::irange(kTotalSize)，为 a_v 中的每个元素赋予标准正态分布的随机值
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  // 创建一个 SimpleIREvaluator 对象 ir_eval，用于评估 stmt 的执行结果，依赖 a_buf 和 b_buf
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  // 使用 a_v 和 b_v 运行 ir_eval
  ir_eval(a_v, b_v);

  // 遍历 c10::irange(kTotalSize)，对比 b_v 和 a_v 经过 std::sigmoid() 处理的结果
  for (const auto i : c10::irange(kTotalSize)) {
    auto test = b_v(i);
    // 创建一个标量张量 t，值为 a_v(i)，然后将其转换为浮点数并赋给 ref
TEST(ATen, erfFloat) {
  const int kTotalSize = 128;  // 定义总大小为128
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);  // 创建名为"A"的缓冲区a_buf，包含kTotalSize个浮点表达式
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);  // 创建名为"B"的缓冲区b_buf，包含kTotalSize个浮点表达式

  VarHandle index = VarHandle("index", kInt);  // 创建整数变量index
  ExprHandle load_a = a_buf.load(index);  // 加载a_buf中索引为index的数据为load_a
  StmtPtr store_b = b_buf.store({index}, erf(load_a));  // 计算erf(load_a)并将结果存储到b_buf的索引为index的位置
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);  // 创建一个循环语句，遍历index从0到kTotalSize-1，执行store_b

  PaddedBuffer<float> a_v(kTotalSize);  // 创建大小为kTotalSize的浮点数填充缓冲区a_v
  PaddedBuffer<float> b_v(kTotalSize);  // 创建大小为kTotalSize的浮点数填充缓冲区b_v

  for (const auto i : c10::irange(kTotalSize)) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    a_v(i) = i / 10.0f;  // 初始化a_v的第i个元素为i除以10.0的浮点数值
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});  // 创建简单IR求值器ir_eval，传入语句stmt以及缓冲区a_buf和b_buf
  ir_eval(a_v, b_v);  // 使用ir_eval对a_v和b_v进行求值

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i / 10.0f);  // 断言a_v的第i个元素等于i除以10.0
    ASSERT_EQ(b_v(i), std::erf(a_v(i)));  // 断言b_v的第i个元素等于a_v的第i个元素的误差函数值
  }
}
    a_v(i) = i / 10.0f;

# 设置数组 `a_v` 的第 `i` 个元素为 `i / 10.0f`


  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});

# 创建 `SimpleIREvaluator` 对象 `ir_eval`，使用给定的语句 `stmt` 和缓冲区 `{a_buf, b_buf}`


  ir_eval(a_v, b_v);

# 使用 `ir_eval` 对象评估数组 `a_v` 和 `b_v`，将计算结果存储在这些数组中


  for (const auto i : c10::irange(kTotalSize)) {

# 遍历范围为 `0` 到 `kTotalSize-1` 的整数 `i`


    ASSERT_EQ(a_v(i), i / 10.0f);

# 断言：数组 `a_v` 的第 `i` 个元素应该等于 `i / 10.0f`


    ASSERT_EQ(b_v(i), std::erf(a_v(i)));

# 断言：数组 `b_v` 的第 `i` 个元素应该等于 `a_v(i)` 的误差函数值 `std::erf(a_v(i))`
TEST(ATen, eqInt) {
  // 定义常量 N 为 128
  constexpr int N = 128;
  // 创建名为 a 的缓冲区，大小为 N，元素类型为整数
  BufHandle a("A", {N}, kInt);
  // 创建名为 b 的缓冲区，大小为 N，元素类型为整数
  BufHandle b("B", {N}, kInt);
  // 创建名为 c 的缓冲区，大小为 N，元素类型为整数
  BufHandle c("C", {N}, kInt);
  // 初始化大小为 N 的整数向量 a_buffer，每个元素为 1
  std::vector<int> a_buffer(N, 1);
  // 初始化大小为 N 的整数向量 b_buffer，每个元素为 1
  std::vector<int> b_buffer(N, 1);
  // 初始化大小为 N 的整数向量 c_buffer，每个元素为 0
  std::vector<int> c_buffer(N, 0);

  // 创建名为 i 的变量，类型为整数
  VarHandle i("i", kInt);
  // 创建一个循环语句，迭代变量 i 从 0 到 N-1
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      // 在循环中，将 c 中的第 i 个元素设为比较 a 的第 i 个元素与 b 的第 i 个元素是否相等的结果
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kEQ)));

  // 创建一个简单的 IR 评估器，用于评估 memcpy_expr
  SimpleIREvaluator ir_eval(memcpy_expr, {a, b, c});
  // 对 a_buffer, b_buffer, c_buffer 进行 IR 评估
  ir_eval(a_buffer, b_buffer, c_buffer);

  // 断言所有 c_buffer 中的元素都为 1
  assertAllEqual(c_buffer, 1);
}
TEST(ATen, leInt) {
  // 声明常量 N 为 128，表示数组大小
  constexpr int N = 128;
  // 创建名为 "A" 的缓冲区，大小为 N，元素类型为 kInt
  BufHandle a("A", {N}, kInt);
  // 创建名为 "B" 的缓冲区，大小为 N，元素类型为 kInt
  BufHandle b("B", {N}, kInt);
  // 创建名为 "C" 的缓冲区，大小为 N，元素类型为 kInt
  BufHandle c("C", {N}, kInt);
  // 初始化大小为 N 的整数向量 a_buffer，每个元素为 5
  std::vector<int> a_buffer(N, 5);
  // 初始化大小为 N 的整数向量 b_buffer，每个元素为 5
  std::vector<int> b_buffer(N, 5);
  // 初始化大小为 N 的整数向量 c_buffer，每个元素为 0
  std::vector<int> c_buffer(N, 0);

  // 声明名为 i 的变量，类型为 kInt
  VarHandle i("i", kInt);
  // 创建一个 For 循环表达式，迭代变量 i 从 0 到 N-1
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      // 将 CompareSelect 表达式结果存储到 c 中，使用 kLE 操作符比较 a[i] 和 b[i]
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLE)));

  // 创建一个简单的 IR 评估器，评估 memcpy_expr 表达式，作用于 a、b、c 缓冲区
  SimpleIREvaluator ir_eval(memcpy_expr, {a, b, c});
  // 执行评估器，将 a_buffer、b_buffer、c_buffer 作为参数传入
  ir_eval(a_buffer, b_buffer, c_buffer);

  // 断言 c_buffer 中所有元素都等于 1
  assertAllEqual(c_buffer, 1);
}

TEST(ATen, ltInt) {
  // 声明常量 N 为 128，表示数组大小
  constexpr int N = 128;
  // 创建名为 "A" 的缓冲区，大小为 N，元素类型为 kInt
  BufHandle a("A", {N}, kInt);
  // 创建名为 "B" 的缓冲区，大小为 N，元素类型为 kInt
  BufHandle b("B", {N}, kInt);
  // 创建名为 "C" 的缓冲区，大小为 N，元素类型为 kInt
  BufHandle c("C", {N}, kInt);
  // 初始化大小为 N 的整数向量 a_buffer，每个元素为 5
  std::vector<int> a_buffer(N, 5);
  // 初始化大小为 N 的整数向量 b_buffer，每个元素为 5
  std::vector<int> b_buffer(N, 5);
  // 初始化大小为 N 的整数向量 c_buffer，每个元素为 1
  std::vector<int> c_buffer(N, 1);

  // 声明名为 i 的变量，类型为 kInt
  VarHandle i("i", kInt);
  // 创建一个 For 循环表达式，迭代变量 i 从 0 到 N-1
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      // 将 CompareSelect 表达式结果存储到 c 中，使用 kLT 操作符比较 a[i] 和 b[i]
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLT)));

  // 创建一个简单的 IR 评估器，评估 memcpy_expr 表达式，作用于 a、b、c 缓冲区
  SimpleIREvaluator ir_eval(memcpy_expr, {a, b, c});
  // 执行评估器，将 a_buffer、b_buffer、c_buffer 作为参数传入
  ir_eval(a_buffer, b_buffer, c_buffer);

  // 断言 c_buffer 中所有元素都等于 0
  assertAllEqual(c_buffer, 0);
}

} // namespace jit
} // namespace torch
```