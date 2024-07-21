# `.\pytorch\test\cpp\tensorexpr\test_expr.cpp`

```
#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <c10/util/irange.h>
#include <test/cpp/tensorexpr/padded_buffer.h>
#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_verifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;

// 测试用例：Expr 类的基本值测试
TEST(Expr, BasicValueTest) {
  // 创建整数常量表达式 a 和 b
  ExprHandle a = IntImm::make(2), b = IntImm::make(3);
  // 创建加法表达式 c = a + b
  ExprHandle c = Add::make(a, b);
  // 创建表达式评估对象，用于评估表达式 c 的值
  SimpleIRExprEval eval(c);
  // 断言表达式 c 的值为 5
  ASSERT_EQ(eval.value<int>(), 5);
}

// 测试用例：Expr 类的基本值测试 02
TEST(Expr, BasicValueTest02) {
  // 创建浮点数常量表达式 a, b, c, d
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  // 创建复合表达式 f = (a + b) - (c + d)
  ExprHandle f = (a + b) - (c + d);
  // 创建表达式评估对象，用于评估表达式 f 的值
  SimpleIRExprEval eval(f);
  // 断言表达式 f 的值为 -4.0
  ASSERT_EQ(eval.value<float>(), -4.0f);
}

// 测试用例：检查是否通道在最后是连续的
TEST(Expr, IsChannelsLastContiguous) {
  // 创建变量数组 vars，包含五个 VarHandle 对象
  std::vector<VarHandle> vars = {
      VarHandle("var1", kLong),
      VarHandle("var2", kLong),
      VarHandle("var3", kLong),
      VarHandle("var4", kLong),
      VarHandle("var5", kLong)};

  // 定义嵌套类型 shapGenInfo，表示形状生成信息的映射
  using shapGenInfo = std::unordered_map<int, std::vector<std::vector<int>>>>;

  // 定义嵌套类型 shapeInfo，表示形状信息的大小和步长
  using shapeInfo =
      std::pair<std::vector<ExprHandle>, std::vector<std::vector<ExprHandle>>>>;

  // 创建整数数组 dims，表示维度大小
  std::vector<int> dims = {3, 4, 5};

  // 创建映射 dims_expr_vec_conf，从整数到表达式向量的映射
  std::unordered_map<int, std::vector<ExprHandle>> dims_expr_vec_conf = {
      {3, std::vector<ExprHandle>(vars.begin(), vars.begin() + 2)},
      {4, std::vector<ExprHandle>(vars.begin(), vars.begin() + 3)},
      {5, std::vector<ExprHandle>(vars.begin(), vars.begin() + 4)},
  };

  // 创建 channels_last_cont_shape_conf，通道在最后连续的形状配置
  shapGenInfo channels_last_cont_shape_conf = {
      {3, {{1, 2, 0}}}, {4, {{1, 3, 2, 0}}}, {5, {{1, 4, 3, 2, 0}}}};
  
  // 创建 channels_last_non_cont_shape_conf，通道在最后非连续的形状配置
  shapGenInfo channels_last_non_cont_shape_conf = {
      {3, {{2, 1, 0}, {1, 0, 2}}},
      {4, {{3, 1, 2, 0}, {1, 2, 3, 0}, {1, 0, 2, 3}}},
      {5, {{4, 3, 2, 1, 0}, {1, 3, 2, 4, 0}, {1, 4, 3, 2, 0}}}
  };

  // 创建 cont_shape_conf，连续形状配置
  shapGenInfo cont_shape_conf = {
      {3, {{0, 1, 2}}}, {4, {{0, 1, 2, 3}}}, {5, {{0, 1, 2, 3, 4}}}
  };

  // 创建 shape_gen_fn，形状生成函数，返回形状信息
  auto shape_gen_fn = [dims_expr_vec_conf](
                          int ndims, shapGenInfo shape_gen_info) -> shapeInfo {
    auto dims_expr_vec = dims_expr_vec_conf.at(ndims);
    std::vector<std::vector<ExprHandle>> strides_expr_vec;
    for (size_t i = 0; i < strides_expr_vec.size(); i++) {
      strides_expr_vec[i].resize(ndims);
    }
    // 返回形状信息：大小和步长
    return std::make_pair(dims_expr_vec, strides_expr_vec);
  };
}

} // namespace jit
} // namespace torch
    // 定义一个 Lambda 函数，用于生成步长
    auto stride_gen_fn = [](int indicator, ExprHandle a, ExprHandle b) {
      // 根据指示符判断是否偶数，选择不同的步长计算方式
      if (indicator % 2 == 0) {
        return a * b;  // 如果是偶数，返回 a * b
      } else {
        return b * a;  // 如果是奇数，返回 b * a
      }
    };

    // 获取形状生成信息中对应维度的步长顺序向量
    auto stride_order_vec = shape_gen_info.at(ndims);
    // 遍历步长表达式向量
    for (size_t i = 0; i < strides_expr_vec.size(); i++) {
      auto stride_order = stride_order_vec[i];

      // 设置当前步长表达式向量的第一个维度的步长为 1
      strides_expr_vec[i][stride_order[0]] = 1;
      // 遍历步长顺序向量的剩余维度
      for (size_t j = 1; j < stride_order.size(); j++) {
        auto cur_dim_idx = stride_order[j];
        auto adjacent_dim_idx = stride_order[j - 1];

        // 使用步长生成函数计算当前维度的步长
        strides_expr_vec[i][cur_dim_idx] = stride_gen_fn(
            i,
            dims_expr_vec[adjacent_dim_idx],
            strides_expr_vec[i][adjacent_dim_idx]);
      }
    }

    // 返回维度表达式向量和步长表达式向量
    return {dims_expr_vec, strides_expr_vec};
  };

  // 定义一个 Lambda 函数，检查是否是 channels-last 连续的
  auto check_channels_last_fn = [](int ndims, BufHandle buf_handle) -> bool {
    // 如果维度数为 3
    if (ndims == 3) {
      return buf_handle.is_channels_last_1d_contiguous();  // 检查是否是 1D channels-last 连续
    } else if (ndims == 4) {
      return buf_handle.is_contiguous(at::MemoryFormat::ChannelsLast);  // 检查是否是 channels-last 连续
    } else {
      return buf_handle.is_contiguous(at::MemoryFormat::ChannelsLast3d);  // 检查是否是 3D channels-last 连续
    }
  };

  // 对每个维度进行 channels-last 连续性检查
  for (size_t i = 0; i < dims.size(); i++) {
    auto shape_info = shape_gen_fn(dims[i], channels_last_cont_shape_conf);
    // 遍历每个形状信息中的形状向量
    for (size_t j = 0; j < shape_info.second.size(); j++) {
      // 创建缓冲区句柄，使用 channels-last 连续的形状信息
      BufHandle buf_handle("a", shape_info.first, shape_info.second[j], kFloat);
      // 断言缓冲区是否是 channels-last 连续
      ASSERT_EQ(check_channels_last_fn(dims[i], buf_handle), true);
    }
  }

  // 对每个维度进行 channels-last 非连续性检查
  for (size_t i = 0; i < dims.size(); i++) {
    auto shape_info = shape_gen_fn(dims[i], channels_last_non_cont_shape_conf);
    // 遍历每个形状信息中的形状向量
    for (size_t j = 0; j < shape_info.second.size(); j++) {
      // 创建缓冲区句柄，使用 channels-last 非连续的形状信息
      BufHandle buf_handle("a", shape_info.first, shape_info.second[j], kFloat);
      // 断言缓冲区是否是 channels-last 非连续
      ASSERT_EQ(check_channels_last_fn(dims[i], buf_handle), false);
    }
  }

  // 对每个维度进行连续性检查
  for (size_t i = 0; i < dims.size(); i++) {
    auto shape_info = shape_gen_fn(dims[i], cont_shape_conf);
    // 遍历每个形状信息中的形状向量
    for (size_t j = 0; j < shape_info.second.size(); j++) {
      // 创建缓冲区句柄，使用连续的形状信息
      BufHandle buf_handle("a", shape_info.first, shape_info.second[j], kFloat);
      // 断言缓冲区是否是连续的
      ASSERT_EQ(buf_handle.is_contiguous(), true);
    }
  }

  // 对每个维度进行非连续性检查
  for (size_t i = 0; i < dims.size(); i++) {
    auto shape_info = shape_gen_fn(dims[i], channels_last_cont_shape_conf);
    // 遍历每个形状信息中的形状向量
    for (size_t j = 0; j < shape_info.second.size(); j++) {
      // 创建缓冲区句柄，使用 channels-last 连续的形状信息
      BufHandle buf_handle("a", shape_info.first, shape_info.second[j], kFloat);
      // 断言缓冲区是否是连续的
      ASSERT_EQ(buf_handle.is_contiguous(), false);
    }
  }
}

# 定义一个名为 Expr 的测试集，包含 LetTest01 测试用例
TEST(Expr, LetTest01) {
  # 创建一个名为 x 的浮点型变量句柄
  VarHandle x("x", kFloat);
  # 创建一个表达式体，计算 2 + (x * 3 + 4)
  ExprHandle body = ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f));
  # 创建一个 SimpleIRExprEval 对象，用于评估表达式
  SimpleIRExprEval eval(body);
  # 绑定变量 x 到值 3
  eval.bindVar(x, ExprHandle(3.f));
  # 断言评估结果与预期相等
  ASSERT_EQ(eval.value<float>(), 2 + (3 * 3 + 4));
}

# 定义一个名为 Expr 的测试集，包含 LetTest02 测试用例
TEST(Expr, LetTest02) {
  # 创建一个名为 x 的浮点型变量句柄
  VarHandle x("x", kFloat);
  # 创建一个名为 y 的浮点型变量句柄
  VarHandle y("y", kFloat);
  # 创建一个表达式体，计算 2 + (x * 3 + 4 * y)
  ExprHandle body =
      ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f) * y);
  # 创建一个 SimpleIRExprEval 对象，用于评估表达式
  SimpleIRExprEval eval(body);
  # 绑定变量 x 到值 3，y 到值 6
  eval.bindVar(x, ExprHandle(3.f));
  eval.bindVar(y, ExprHandle(6.f));
  # 断言评估结果与预期相等
  ASSERT_EQ(eval.value<float>(), 2 + (3 * 3 + 4 * 6));
}

# 定义一个名为 Expr 的测试集，包含 LetStmtTest01 测试用例
TEST(Expr, LetStmtTest01) {
  # 创建一个名为 a 的缓冲区句柄，大小为 1，类型为浮点型
  BufHandle a_buf("a", {1}, kFloat);
  # 创建一个名为 b 的缓冲区句柄，大小为 1，类型为浮点型
  BufHandle b_buf("b", {1}, kFloat);

  # 创建加载缓冲区 a 第一个元素的表达式
  ExprHandle load_a = a_buf.load(0);
  # 创建一个名为 var 的浮点型变量句柄
  VarHandle var = VarHandle("v", kFloat);
  # 创建一个赋值语句，将 load_a 的值赋给 var
  StmtPtr let_store = Let::make(var, load_a);
  # 创建一个存储 var 到缓冲区 b 第一个元素的语句
  StmtPtr store_b = b_buf.store({0}, var);
  # 创建一个语句块，包含 let_store 和 store_b
  BlockPtr block = Block::make({let_store, store_b});

  # 创建一个 SimpleIREvaluator 对象，用于执行语句块
  SimpleIREvaluator eval(block, {a_buf, b_buf});

  # 创建大小为 1 的浮点型填充缓冲区 a_v, b_v 和 b_ref
  PaddedBuffer<float> a_v(1);
  PaddedBuffer<float> b_v(1);
  PaddedBuffer<float> b_ref(1);

  # 设置 a_v 的第一个元素为 23，b_ref 的第一个元素也为 23
  a_v(0) = 23;
  b_ref(0) = a_v(0);
  # 执行语句块，将结果存储在 b_v 中
  eval(a_v, b_v);

  # 断言 b_v 和 b_ref 的值在 1e-5 的容差范围内相等
  ExpectAllNear(b_v, b_ref, 1e-5);
}

# 定义一个名为 Expr 的测试集，包含 IntTest 测试用例
TEST(Expr, IntTest) {
  # 创建一个名为 x 的整型变量句柄
  VarHandle x("x", kInt);
  # 创建一个表达式体，计算 2 + (x * 3 + 4)
  ExprHandle body = ExprHandle(2) + (x * ExprHandle(3) + ExprHandle(4));
  # 创建一个 SimpleIRExprEval 对象，用于评估表达式
  SimpleIRExprEval eval(body);
  # 绑定变量 x 到值 3
  eval.bindVar(x, ExprHandle(3));
  # 断言评估结果与预期相等
  ASSERT_EQ(eval.value<int>(), 2 + (3 * 3 + 4));
}

# 定义一个名为 Expr 的测试集，包含 FloatTest 测试用例
TEST(Expr, FloatTest) {
  # 创建一个名为 x 的浮点型变量句柄
  VarHandle x("x", kFloat);
  # 创建一个表达式体，计算 2 + (x * 3 + 4)
  ExprHandle body = ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f));
  # 创建一个 SimpleIRExprEval 对象，用于评估表达式
  SimpleIRExprEval eval(body);
  # 绑定变量 x 到值 3
  eval.bindVar(x, ExprHandle(3.f));
  # 断言评估结果与预期相等
  ASSERT_EQ(eval.value<float>(), 2 + (3 * 3 + 4));
}

# 定义一个名为 Expr 的测试集，包含 ByteTest 测试用例
TEST(Expr, ByteTest) {
  # 创建一个名为 x 的字节型变量句柄
  VarHandle x("x", kByte);
  # 创建一个表达式体，计算 2 + (x * 3 + 4)
  ExprHandle body = ExprHandle((uint8_t)2) +
      (x * ExprHandle((uint8_t)3) + ExprHandle((uint8_t)4));
  # 创建一个 SimpleIRExprEval 对象，用于评估表达式
  SimpleIRExprEval eval(body);
  # 绑定变量 x 到值 3
  eval.bindVar(x, ExprHandle((uint8_t)3));
  # 断言评估结果与预期相等
  ASSERT_EQ(eval.value<uint8_t>(), 2 + (3 * 3 + 4));
}

# 定义一个名为 Expr 的测试集，包含 CharTest 测试用例
TEST(Expr, CharTest) {
  # 创建一个名为 x 的字符型变量句柄
  VarHandle x("x", kChar);
  # 创建一个表达式体，计算 2 + (x * 3 + 4)
  ExprHandle body = ExprHandle((int8_t)2) +
      (x * ExprHandle((int8_t)3) + ExprHandle((int8_t)4));
  # 创建一个 SimpleIRExprEval 对象，用于评估表达式
  SimpleIRExprEval eval(body);
  # 绑定变量 x 到值 3
  eval.bindVar(x, ExprHandle((int8_t)3));
  # 断言评估结果与预期相等
  ASSERT_EQ(eval.value<int8_t>(), 2 + (3 * 3 + 4));
}

# 定义一个名为 Expr 的测试集，包含 ShortTest 测试用例
TEST(Expr, ShortTest) {
  # 创建一个名为 x 的短整型变量句柄
  VarHandle x("x", kShort);
  # 创建一个表达式体，计算 2 + (x * 3 + 4)
  ExprHandle body = ExprHandle((int16_t)2) +
      (x * ExprHandle((int16_t)3) + ExprHandle((int16_t)4));
  # 创建一个 SimpleIRExprEval 对象，用于评估表达式
  SimpleIRExprEval eval(body);
  # 绑定变量 x 到值 3
  eval.bindVar(x, ExprHandle((int16_t)3));
  # 断言评估结果与预期相等
  ASSERT_EQ(eval.value<int16_t>(), 2 + (3 * 3 + 4));
}

# 定义一个名为 Expr 的测试集，包含 LongTest 测试用例
TEST(Expr, LongTest) {
  # 创建一个名为 x 的
TEST(Expr, VectorAdd01) {
  // 定义向量的大小、向量个数和总大小
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  // 创建三个缓冲区对象，分别表示A、B、C缓冲区，每个缓冲区的元素类型为float
  BufHandle a_buf("A", {kTotalSize}, kFloat);
  BufHandle b_buf("B", {kTotalSize}, kFloat);
  BufHandle c_buf("C", {kTotalSize}, kFloat);

  /*
  构建以下循环体:
    for (const auto index : c10::irange(kVectorCount)) {
      store(c_buf, ramp(index * 8, 1, 8),
            load(a_buf, ramp(index * 8, 1, 8) +
            load(b_buf, ramp(index * 8, 1, 8))))
    }
  */
  // 创建循环索引变量index，类型为int
  VarHandle index = VarHandle("index", kInt);
  // 计算加载A缓冲区的表达式，使用ramp表示索引
  ExprHandle load_a =
      a_buf.load({Ramp::make(index * kVectorSize, 1, kVectorSize)});
  // 计算加载B缓冲区的表达式，同样使用ramp表示索引
  ExprHandle load_b =
      b_buf.load({Ramp::make(index * kVectorSize, 1, kVectorSize)});
  // 计算存储到C缓冲区的表达式，即A缓冲区和B缓冲区对应元素之和
  ExprHandle value = load_a + load_b;
  // 创建存储操作的语句，将计算结果存储到C缓冲区中对应的位置
  StmtPtr store_c =
      c_buf.store({Ramp::make(index * kVectorSize, 1, kVectorSize)}, value);
  // 创建for循环语句，遍历索引index从0到kVectorCount-1，执行存储操作store_c
  StmtPtr stmt = For::make(index, 0, kVectorCount, store_c);

  // 断言验证加载A缓冲区、加载B缓冲区和计算结果的数据类型均为float类型，且每个元素数量为kVectorSize
  ASSERT_EQ(load_a.dtype(), Dtype(kFloat, kVectorSize));
  ASSERT_EQ(load_b.dtype(), Dtype(kFloat, kVectorSize));
  ASSERT_EQ(value.dtype(), Dtype(kFloat, kVectorSize));

  // 创建用于测试的数据缓冲区，分别为A、B、C和参考结果缓冲区
  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> c_ref(kTotalSize);
  // 填充A和B缓冲区的数据
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i * i;
    b_v(i) = i * i * 4;
    // 计算参考结果缓冲区的值，即A缓冲区和B缓冲区对应位置的元素之和
    c_ref(i) = a_v(i) + b_v(i);
  }
  // 创建简单的IR求值器对象，执行stmt语句，绑定A、B、C缓冲区，将计算结果存储到c_v中
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);
  // 断言验证c_v与c_ref的每个元素之间的差异在指定的精度范围内
  ExpectAllNear(c_v, c_ref, 1e-5);
}
TEST(Expr, CompareSelectDtypes) {
  // 对比选择表达式的测试，确保输入表达式的数据类型相同，但返回值的数据类型可以不同
  // 确保 true 和 false 返回值的数据类型相同
  // 构造一个 CompareSelect 表达式，其中输入数据类型与输出数据类型不同，并验证其正确性：
  //   result = ((int)lhs == (int)rhs) ? (float)retval1 : (float)retval2
  constexpr int N = 1024;
  // 创建三个缓冲区：a 是 int 类型，b 是 int 类型，c 是 float 类型
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kFloat);
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 0.0f);
  std::vector<float> c_ref(N, 3.14f);

  VarHandle i("i", kInt);
  // 构造 C[i] = (A[i] == B[i]) ? 3.14f : 2.78f
  // A 和 B 是 int 类型，C 是 float 类型
  auto select_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i),
              b.load(i),
              FloatImm::make(3.14f),
              FloatImm::make(2.78f),
              CompareSelectOperation::kEQ)));

  SimpleIREvaluator ir_eval(select_expr, {a, b, c});
  ir_eval(a_buffer, b_buffer, c_buffer);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1);
  assertAllEqual(b_buffer, 1);
  ExpectAllNear(c_buffer, c_ref, 1e-7);
}

TEST(Expr, IntrinsicsDtypes) {
  // 内置函数数据类型测试
  constexpr int N = 256;
  // 创建两个缓冲区：a 和 b 都是 double 类型
  BufHandle a("A", {N}, kDouble);
  BufHandle b("B", {N}, kDouble);
  std::vector<double> a_buffer(N, -10.0);
  std::vector<double> b_buffer(N, 0.0);
  std::vector<double> b_ref(N, 10.0);

  VarHandle i("i", kInt);
  // 构造 abs 表达式：B[i] = abs(A[i])
  auto abs_expr = For::make(i, 0, N, b.store({i}, tensorexpr::abs(a.load(i))));

  SimpleIREvaluator ir_eval(abs_expr, {a, b});
  ir_eval(a_buffer, b_buffer);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);

  assertAllEqual(a_buffer, -10.0);
  ExpectAllNear(b_buffer, b_ref, 1e-7);
}

TEST(Expr, Substitute01) {
  // 表达式替换测试
  VarPtr x = alloc<Var>("x", kFloat);
  VarPtr y = alloc<Var>("y", kFloat);
  // 构造表达式：e = (x - 1.0f) * (x + y)
  ExprPtr e =
      alloc<Mul>(alloc<Sub>(x, alloc<FloatImm>(1.0f)), alloc<Add>(x, y));

  VarPtr z = alloc<Var>("z", kFloat);
  // 对表达式 e 进行替换：e2 = e[x := z + 5.0f]
  ExprPtr e2 = Substitute(e, {{x, alloc<Add>(z, alloc<FloatImm>(5.0f))}});
  // 期望的替换后表达式 e2_ref
  ExprPtr e2_ref = alloc<Mul>(
      alloc<Sub>(alloc<Add>(z, alloc<FloatImm>(5.0f)), alloc<FloatImm>(1.0f)),
      alloc<Add>(alloc<Add>(z, alloc<FloatImm>(5.0f)), y));
  std::ostringstream oss;
  oss << *e2;
  std::string e2_str = oss.str();

  oss.str("");
  oss << *e2_ref;
  std::string e2_ref_str = oss.str();
  ASSERT_EQ(e2_str, e2_ref_str);
}

TEST(Expr, Math01) {
  // 数学函数测试
  ExprHandle v = sin(ExprHandle(1.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "sin(1.f)");

  SimpleIRExprEval eval(v);
  float v_ref = std::sin(1.0f);
  float res = eval.value<float>();
  ASSERT_NEAR(res, v_ref, 1e-6);
}

TEST(Expr, UnaryMath01) {
  // 单目数学函数测试
  struct TestConfig {
    // 定义一个包含两个成员函数对象的结构体，用于存储测试配置信息
    struct TestConfig {
        std::function<ExprHandle(const ExprHandle&)> func; // 表达式处理函数对象
        std::function<float(float)> ref_func; // 参考函数对象
    };

    // 初始化测试配置数组
    std::vector<TestConfig> test_configs = {
        // 每个元素是一个包含两个 lambda 函数的 TestConfig 结构体，分别用于表达式处理和参考函数
        {[](const ExprHandle& v) { return sin(v); }, [](float v) { return std::sin(v); }},
        {[](const ExprHandle& v) { return sin(v); }, [](float v) { return std::sin(v); }},
        {[](const ExprHandle& v) { return tan(v); }, [](float v) { return std::tan(v); }},
        {[](const ExprHandle& v) { return asin(v); }, [](float v) { return std::asin(v); }},
        {[](const ExprHandle& v) { return acos(v); }, [](float v) { return std::acos(v); }},
        {[](const ExprHandle& v) { return atan(v); }, [](float v) { return std::atan(v); }},
        {[](const ExprHandle& v) { return sinh(v); }, [](float v) { return std::sinh(v); }},
        {[](const ExprHandle& v) { return cosh(v); }, [](float v) { return std::cosh(v); }},
        {[](const ExprHandle& v) { return tanh(v); }, [](float v) { return std::tanh(v); }},
        {[](const ExprHandle& v) { return exp(v); }, [](float v) { return std::exp(v); }},
        {[](const ExprHandle& v) { return tensorexpr::abs(v); }, [](float v) { return std::fabs(v); }},
        {[](const ExprHandle& v) { return log(v); }, [](float v) { return std::log(v); }},
        {[](const ExprHandle& v) { return log2(v); }, [](float v) { return std::log2(v); }},
        {[](const ExprHandle& v) { return log10(v); }, [](float v) { return std::log10(v); }},
        {[](const ExprHandle& v) { return erf(v); }, [](float v) { return std::erf(v); }},
        {[](const ExprHandle& v) { return sqrt(v); }, [](float v) { return std::sqrt(v); }},
        {[](const ExprHandle& v) { return rsqrt(v); }, [](float v) { return 1.0f / std::sqrt(v); }},
        {[](const ExprHandle& v) { return ceil(v); }, [](float v) { return std::ceil(v); }},
        {[](const ExprHandle& v) { return floor(v); }, [](float v) { return std::floor(v); }},
        {[](const ExprHandle& v) { return round(v); }, [](float v) { return std::round(v); }},
        {[](const ExprHandle& v) { return trunc(v); }, [](float v) { return std::trunc(v); }},
    };

    // 遍历测试配置数组，对每个配置进行表达式计算和参考函数比较
    for (const TestConfig& test_config : test_configs) {
        // 设置输入值
        const float input_v = 0.8765f;
        // 通过表达式处理函数计算表达式结果
        ExprHandle v = test_config.func(ExprHandle(input_v));
        // 使用参考函数计算参考值
        float v_ref = test_config.ref_func(input_v);
        // 创建表达式求值对象
        SimpleIRExprEval eval(v);
        // 断言表达式计算结果接近参考值，误差小于 1e-6
        ASSERT_NEAR(eval.value<float>(), v_ref, 1e-6);
    }

    // 循环测试特定浮点数输入的情况，包括 NaN、0、0.5
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    for (float input_v : {std::nan("1"), 0., .5}) {
        // 创建浮点常量表达式
        ExprHandle v = FloatImm::make(input_v);
        // 创建检查是否 NaN 的表达式
        SimpleIRExprEval eval(Intrinsics::make(kIsNan, v));
        // 断言表达式计算结果为 int 的值等于 std::isnan(input_v)
        ASSERT_NEAR(eval.value<int>(), std::isnan(input_v), 0);
    }
TEST(Expr, LogicalOps01) {
  // 定义表达式中使用的变量和常量
  ExprHandle a(23);
  ExprHandle b(11);
  ExprHandle c(0.72f);
  ExprHandle d(0.69f);

  // 定义多个逻辑表达式
  ExprHandle f1 = (a > b) && (c > d);
  ExprHandle f2 = (a > b) && (c < d);
  ExprHandle f3 = (a < b) && (c > d);
  ExprHandle f4 = (a < b) && (c < d);
  ExprHandle f5 = (a < b) || (c > d);
  ExprHandle f6 = (a < b) || (c < d);
  ExprHandle f7 = (a > b) || (c < d);
  ExprHandle f8 = (a > b) || (c > d);

  // 对每个逻辑表达式进行求值
  SimpleIRExprEval eval1(f1);
  SimpleIRExprEval eval2(f2);
  SimpleIRExprEval eval3(f3);
  SimpleIRExprEval eval4(f4);
  SimpleIRExprEval eval5(f5);
  SimpleIRExprEval eval6(f6);
  SimpleIRExprEval eval7(f7);
  SimpleIRExprEval eval8(f8);

  // 断言每个逻辑表达式的求值结果
  ASSERT_EQ(eval1.value<int>(), 1);
  ASSERT_EQ(eval2.value<int>(), 0);
  ASSERT_EQ(eval3.value<int>(), 0);
  ASSERT_EQ(eval4.value<int>(), 0);
  ASSERT_EQ(eval5.value<int>(), 1);
  ASSERT_EQ(eval6.value<int>(), 0);
  ASSERT_EQ(eval7.value<int>(), 1);
  ASSERT_EQ(eval8.value<int>(), 1);
}

TEST(Expr, LogicalOps02) {
  // 定义表达式中使用的变量和常量
  ExprHandle a(23);
  ExprHandle b(11);
  ExprHandle c(0.72f);
  ExprHandle d(0.72f);

  // 定义多个逻辑表达式
  ExprHandle f1 = (a > b) || (c > d);
  ExprHandle f2 = (a > b) && (c <= d);
  ExprHandle f3 = (a > b) && (c > d);
  ExprHandle ff1 = f1 && f2;
  ExprHandle ff2 = f2 || f3;

  // 对每个逻辑表达式进行求值
  SimpleIRExprEval eval1(ff1);
  SimpleIRExprEval eval2(ff2);

  // 断言每个逻辑表达式的求值结果
  ASSERT_EQ(eval1.value<int>(), 1);
  ASSERT_EQ(eval2.value<int>(), 1);
}
TEST(Expr, LogicalOps03) {
  // 创建整数表达式变量 a、b、c、d 分别初始化为 23、11、0.72 和 0.69
  ExprHandle a(23);
  ExprHandle b(11);
  ExprHandle c(0.72f);
  ExprHandle d(0.69f);

  // 创建布尔类型表达式
  ExprHandle bool_f1 = (a > b) && BoolImm::make(true);  // 布尔运算 (a > b) && true
  ExprHandle bool_f2 = (c <= d) || BoolImm::make(true);  // 布尔运算 (c <= d) || true

  // 创建整数类型表达式
  ExprHandle int_f1 = (a > b) && IntImm::make(1);  // 整数运算 (a > b) && 1
  ExprHandle int_f2 = (c <= d) || IntImm::make(1);  // 整数运算 (c <= d) || 1

  // 创建短整数类型表达式
  ExprHandle short_f1 = (a > b) && ShortImm::make(1);  // 短整数运算 (a > b) && 1
  ExprHandle short_f2 = (c <= d) || ShortImm::make(1);  // 短整数运算 (c <= d) || 1

  // 创建长整数类型表达式
  ExprHandle long_f1 = (a > b) && LongImm::make(1);  // 长整数运算 (a > b) && 1
  ExprHandle long_f2 = (c <= d) || LongImm::make(1);  // 长整数运算 (c <= d) || 1

  // 创建字符类型表达式
  ExprHandle char_f1 = (a > b) && CharImm::make(1);  // 字符运算 (a > b) && 1
  ExprHandle char_f2 = (c <= d) || CharImm::make(1);  // 字符运算 (c <= d) || 1

  // 创建字节类型表达式
  ExprHandle byte_f1 = (a > b) && ByteImm::make(1);  // 字节运算 (a > b) && 1
  ExprHandle byte_f2 = (c <= d) || ByteImm::make(1);  // 字节运算 (c <= d) || 1

  // 创建简单整数表达式评估对象并进行评估
  SimpleIRExprEval eval1(bool_f1);
  SimpleIRExprEval eval2(bool_f2);
  SimpleIRExprEval eval3(int_f1);
  SimpleIRExprEval eval4(int_f2);
  SimpleIRExprEval eval5(short_f1);
  SimpleIRExprEval eval6(short_f2);
  SimpleIRExprEval eval7(long_f1);
  SimpleIRExprEval eval8(long_f2);
  SimpleIRExprEval eval9(char_f1);
  SimpleIRExprEval eval10(char_f2);
  SimpleIRExprEval eval11(byte_f1);
  SimpleIRExprEval eval12(byte_f2);

  // 断言评估结果符合预期
  ASSERT_EQ(eval1.value<bool>(), true);
  ASSERT_EQ(eval2.value<bool>(), true);
  ASSERT_EQ(eval3.value<int>(), 1);
  ASSERT_EQ(eval4.value<int>(), 1);
  ASSERT_EQ(eval5.value<int16_t>(), 1);
  ASSERT_EQ(eval6.value<int16_t>(), 1);
  ASSERT_EQ(eval7.value<int64_t>(), 1);
  ASSERT_EQ(eval8.value<int64_t>(), 1);
  ASSERT_EQ(eval9.value<int8_t>(), 1);
  ASSERT_EQ(eval10.value<int8_t>(), 1);
  ASSERT_EQ(eval11.value<uint8_t>(), 1);
  ASSERT_EQ(eval12.value<uint8_t>(), 1);
}

TEST(Expr, BitwiseOps) {
  // 创建整数表达式变量 a、b、c、d 分别初始化为 59、11、101 和 2
  ExprHandle a(59);
  ExprHandle b(11);
  ExprHandle c(101);
  ExprHandle d(2);
  // 创建位运算表达式 f
  ExprHandle f = (((a ^ (b << 1)) & c) >> 2) | d;

  // 创建简单整数表达式评估对象并进行评估
  SimpleIRExprEval eval(f);
  // 断言评估结果符合预期
  ASSERT_EQ(eval.value<int>(), 11);
}

TEST(Expr, DynamicShapeAdd) {
  // 定义一个 lambda 函数 testWithSize，用于测试不同大小的数组
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt);  // 创建整数变量 n
    BufHandle a("a", {n}, kFloat);  // 创建缓冲区变量 a，其大小由 n 决定
    BufHandle b("b", {n}, kFloat);  // 创建缓冲区变量 b，其大小由 n 决定
    BufHandle c("c", {n}, kFloat);  // 创建缓冲区变量 c，其大小由 n 决定
    VarHandle i("i", kInt);  // 创建整数变量 i
    // 创建一个 For 循环语句，循环次数由变量 n 决定，在缓冲区 c 中存储计算结果
    StmtPtr s = For::make(i, 0, n, c.store({i}, a.load(i) + b.load(i)));
    // 创建数组 aData、bData、cData，分别初始化为 1.0、2.0 和 0.0，大小为 size
    std::vector<float> aData(size, 1.0f);
    std::vector<float> bData(size, 2.0f);
    std::vector<float> cData(size, 0.0f);
    // 创建简单 IR 评估器对象并进行评估
    SimpleIREvaluator(s, {a, b, c, n})(aData, bData, cData, size);
    // 断言所有 cData 中的元素接近 3.0，精度为 1e-7
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  // 分别测试数组大小为 1、16 和 37
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

TEST(Expr, OutOfBounds) {
  ExprHandle N(10);  // 创建整数表达式变量 N，初始化为 10
  ExprHandle start(0);  // 创建整数表达式变量 start，初始化为 0
  ExprHandle stop(15);  // 创建整数表达式变量 stop，初始化为 15
  VarHandle i("i", kInt);  // 创建整数变量 i

  BufHandle X("X", {N}, kInt);  // 创建整数缓冲区变量 X，大小由 N 决定

  auto body = Store::make(X, {i}, i);  // 创建一个 Store 操作体，将变量 i 的值存入 X
  auto stmt = For::make(i, start, stop, body);  // 创建一个 For 循环语句，循环范围为 [start, stop)，执行 body

  PaddedBuffer<int> data(20);  // 创建一个大小为 20 的填充整数缓冲区 data

  // 预期执行 SimpleIREvaluator 时会抛出异常
  EXPECT_ANY_THROW(SimpleIREvaluator(stmt, {X})(data));
}
TEST(Expr, OutOfBounds2d) {
  // 定义一个包含两种不同尺寸的二维数组大小选项
  std::vector<std::pair<int, int>> size_options = {{10, 15}, {15, 10}};
  // 遍历每种尺寸选项
  for (auto sizes : size_options) {
    // 创建表达式句柄 N 和 M 分别表示数组的行数和列数
    ExprHandle N(sizes.first);
    ExprHandle M(sizes.second);
    // 定义起始值和内外层循环的终止条件
    ExprHandle start(0);
    ExprHandle stopInner(15);
    ExprHandle stopOuter(15);
    // 定义整型变量 i 和 j 作为循环变量
    VarHandle i("i", kInt);
    VarHandle j("j", kInt);

    // 创建缓冲区 X，表示一个 N 行 M 列的整型数组
    BufHandle X("X", {N, M}, kInt);

    // 创建存储语句体，将 i 存储到数组 X 的位置 (i, j)
    auto body = Store::make(X, {i, j}, i);
    // 创建内层循环语句，遍历 j 从 start 到 stopInner
    auto inner = For::make(j, start, stopInner, body);
    // 创建外层循环语句，遍历 i 从 start 到 stopOuter
    auto stmt = For::make(i, start, stopOuter, inner);

    // 创建一个具有400个元素的整型填充缓冲区
    PaddedBuffer<int> data(400);

    // 期望在评估过程中抛出异常
    EXPECT_ANY_THROW(SimpleIREvaluator(stmt, {X})(data));
  }
}

TEST(Expr, OutOfBounds2dFlattenedIndex) {
  // 定义缓冲区大小为149
  ExprHandle buf_size(149);
  // 定义起始值和内外层循环的终止条件
  ExprHandle start(0);
  ExprHandle stopInner(15);
  ExprHandle stopOuter(10);
  // 定义整型变量 i 和 j 作为循环变量
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);

  // 创建缓冲区 X，表示一个大小为 buf_size 的整型数组
  BufHandle X("X", {buf_size}, kInt);

  // 创建索引表达式 idx，表示将二维坐标 (i, j) 映射到一维索引
  auto idx = Add::make(Mul::make(i, stopInner), j);
  // 创建存储语句体，将 i 存储到数组 X 的位置 idx
  auto body = Store::make(X, {idx}, i);
  // 创建内层循环语句，遍历 j 从 start 到 stopInner
  auto inner = For::make(j, start, stopInner, body);
  // 创建外层循环语句，遍历 i 从 start 到 stopOuter
  auto stmt = For::make(i, start, stopOuter, inner);

  // 创建一个具有400个元素的整型填充缓冲区
  PaddedBuffer<int> data(400);

  // 期望在评估过程中抛出异常
  EXPECT_ANY_THROW(SimpleIREvaluator(stmt, {X})(data));
}

void testCond01() {
  // 定义数组大小 N 为 16
  const int N = 16;
  // 创建具有 N 个元素的浮点数填充缓冲区 a_v
  PaddedBuffer<float> a_v(N);
  // 创建缓冲区 a_buf，表示一个大小为 N 的浮点数数组
  BufHandle a_buf("a", {N}, kFloat);
  // 创建整型变量 index 作为循环变量
  VarHandle index = VarHandle("index", kInt);
  // 创建存储语句 assign_x2，将 index * 2 存储到数组 a_buf 的位置 index
  StmtPtr assign_x2 = a_buf.store({index}, cast<float>(index) * 2);
  // 创建存储语句 assign_x3，将 index * 3 存储到数组 a_buf 的位置 index
  StmtPtr assign_x3 = a_buf.store({index}, cast<float>(index) * 3);
  // 创建偶数条件表达式 even_cond，判断 index 是否为偶数
  ExprHandle even_cond = CompareSelect::make(Mod::make(index, 2), 0, kEQ);
  // 创建条件语句 assign，根据 even_cond 条件选择执行 assign_x2 或 assign_x3
  StmtPtr assign = Cond::make(even_cond, assign_x2, assign_x3);
  // 创建循环语句 for_stmt，遍历 index 从 0 到 N，执行 assign
  StmtPtr for_stmt = For::make(index, 0, N, assign);
  // 使用 SimpleIREvaluator 执行 for_stmt，传入数组 a_buf，并将结果存储到 a_v
  SimpleIREvaluator(for_stmt, {a_buf})(a_v);

  // 创建参考数组 a_ref，用于存储根据条件赋值的预期结果
  PaddedBuffer<float> a_ref(N);
  // 循环计算预期的结果并存储到 a_ref 中
  for (const auto i : c10::irange(N)) {
    if (i % 2 == 0) {
      a_ref(i) = i * 2;
    } else {
      a_ref(i) = i * 3;
    }
  }
  // 验证计算结果 a_v 与预期结果 a_ref 之间的近似度，允许误差为 1e-5
  ExpectAllNear(a_v, a_ref, 1e-5);
}

void testIfThenElse01() {
  // 创建表达式 v，执行条件选择表达式 ifThenElse(1, 1.0f, 2.0f)
  ExprHandle v = ifThenElse(ExprHandle(1), ExprHandle(1.0f), ExprHandle(2.0f));

  // 创建字符串流 oss，用于将表达式 v 转换为字符串
  std::ostringstream oss;
  oss << v;
  // 断言将表达式 v 转换为字符串后与预期结果 "IfThenElse(1, 1.f, 2.f)" 相等
  ASSERT_EQ(oss.str(), "IfThenElse(1, 1.f, 2.f)");

  // 使用 SimpleIRExprEval 对象 eval 计算表达式 v 的值，并断言其为 1.0f
  SimpleIRExprEval eval(v);
  ASSERT_EQ(eval.value<float>(), 1.0f);
}

void testIfThenElse02() {
  // 创建表达式 v，执行条件选择表达式 ifThenElse(0, 1.0f, 2.0f)
  ExprHandle v = ifThenElse(ExprHandle(0), ExprHandle(1.0f), ExprHandle(2.0f));

  // 创建字符串流 oss，用于将表达式 v 转换为字符串
  std::ostringstream oss;
  oss << v;
  // 断言将表达式 v 转换为字符串后与预期结果 "IfThenElse(0, 1.f, 2.f)" 相等
  ASSERT_EQ(oss.str(), "IfThenElse(0, 1.f, 2.f)");

  // 使用 SimpleIRExprEval 对象 eval 计算表达式 v 的值，并断言其为 2.0f
  SimpleIRExprEval eval(v);
  ASSERT_EQ(eval.value<float>(), 2.0f);
}

void testIfThenElse03() {
  // 创建表达式 v，执行条件选择表达式 ifThenElse(false, 1.0f, 2.0f)
  ExprHandle v =
      ifThenElse(BoolImm::make(false), ExprHandle(1.0f), ExprHandle(2.0f));

  // 创建字符串流 oss，用于将表达式 v 转换为字符串
  std::ostringstream oss;
  oss << v;
  // 断言将表达式 v 转换为字符串后与预期结果 "IfThenElse(0, 1.f, 2.f)" 相等
  ASSERT_EQ(oss.str(), "IfThenElse(0, 1.f, 2.f)");

  // 使用 SimpleIRExprEval 对象 eval 计算表达式 v 的值，并
// 定义一个名为 testStmtClone 的函数，用于测试语句的克隆功能
void testStmtClone() {
  // 声明并初始化常量 N 为 16
  const int N = 16;

  // 创建一个名为 a_buf 的缓冲区句柄，大小为 N，数据类型为整数
  BufHandle a_buf("a", {N}, kInt);

  // 创建一个名为 index 的变量句柄，数据类型为整数
  VarHandle index = VarHandle("index", kInt);

  // 生成一个将值 5 存储到 a_buf 中 index 位置的语句，存储结果存储在 body 中
  StmtPtr body = a_buf.store({index}, 5);

  // 生成一个 for 循环语句，从 0 到 N-1，执行体为之前生成的 body
  StmtPtr loop = For::make(index, 0, N, body);

  // 克隆之前创建的循环语句 loop，得到 cloned_loop
  StmtPtr cloned_loop = Stmt::clone(loop);

  // 创建用于存储原始循环结果和克隆循环结果的整数向量
  std::vector<int> orig_loop_results(N);
  std::vector<int> cloned_loop_results(N);

  // 对原始循环进行简单的 IR 评估，将结果存储在 orig_loop_results 中
  SimpleIREvaluator(loop, {a_buf})(orig_loop_results);

  // 对克隆循环进行简单的 IR 评估，将结果存储在 cloned_loop_results 中
  SimpleIREvaluator(cloned_loop, {a_buf})(cloned_loop_results);

  // 断言：检查 orig_loop_results 中的所有元素是否都等于 5
  assertAllEqual(orig_loop_results, 5);

  // 断言：检查 cloned_loop_results 中的所有元素是否都等于 5
  assertAllEqual(cloned_loop_results, 5);

  // 在克隆的循环体中添加另一个将值 33 存储到 a_buf 中 index 位置的语句
  StmtPtr body_addition = a_buf.store({index}, 33);
  
  // 将克隆循环的执行体强制转换为 BlockPtr 类型，并在其末尾附加新的 body_addition 语句
  BlockPtr cloned_body = static_to<Block>(static_to<For>(cloned_loop)->body());
  cloned_body->append_stmt(body_addition);

  // 创建用于存储变异后原始循环结果和克隆循环结果的整数向量
  std::vector<int> orig_loop_results_after_mutation(N);
  std::vector<int> cloned_loop_results_after_mutation(N);

  // 再次对原始循环进行简单的 IR 评估，将结果存储在 orig_loop_results_after_mutation 中
  SimpleIREvaluator(loop, {a_buf})(orig_loop_results_after_mutation);

  // 再次对克隆循环进行简单的 IR 评估，将结果存储在 cloned_loop_results_after_mutation 中
  SimpleIREvaluator(cloned_loop, {a_buf})(cloned_loop_results_after_mutation);

  // 断言：检查 orig_loop_results_after_mutation 中的所有元素是否都等于 5
  assertAllEqual(orig_loop_results_after_mutation, 5);

  // 断言：检查 cloned_loop_results_after_mutation 中的所有元素是否都等于 33
  assertAllEqual(cloned_loop_results_after_mutation, 33);
}

// 命名空间结束：jit
} // namespace jit

// 命名空间结束：torch
} // namespace torch
```