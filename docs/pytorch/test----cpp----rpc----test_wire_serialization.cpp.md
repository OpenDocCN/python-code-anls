# `.\pytorch\test\cpp\rpc\test_wire_serialization.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 Torch 相关的头文件
#include <c10/util/irange.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/torch.h>

// 包含标准库头文件
#include <memory>
#include <string>
#include <vector>

// 使用 Google Test 的 IsSubstring 断言
using ::testing::IsSubstring;

// 定义测试用例 WireSerialize.Base
TEST(WireSerialize, Base) {
  // 定义内部函数 run，接受 payload 和 tensors 作为参数
  auto run = [](const std::string& payload,
                const std::vector<at::Tensor>& tensors) {
    // 初始化字符串 serialized
    std::string serialized;
    {
      // 将 payload 转换为字符向量 mpayload
      std::vector<char> mpayload(payload.begin(), payload.end());
      // 拷贝 tensors 到 mtensors
      std::vector<at::Tensor> mtensors = tensors;
      // 调用 torch::distributed::rpc::wireSerialize 序列化数据
      serialized = torch::distributed::rpc::wireSerialize(
          std::move(mpayload), std::move(mtensors));
    }
    // 调用 torch::distributed::rpc::wireDeserialize 反序列化数据
    auto deser = torch::distributed::rpc::wireDeserialize(
        serialized.data(), serialized.size());
    // 断言序列化前后 payload 大小相等
    EXPECT_EQ(payload.size(), deser.first.size());
    // 断言序列化前后 tensors 数量相等
    EXPECT_EQ(tensors.size(), deser.second.size());
    // 如果 payload 长度大于 0，比较 payload 数据是否相同
    if (payload.size() > 0) {
      EXPECT_TRUE(
          memcmp(deser.first.data(), payload.data(), payload.size()) == 0);
    }
    // 对每个 tensors 的元素进行相等性断言
    for (const auto i : c10::irange(tensors.size())) {
      EXPECT_TRUE(torch::equal(tensors[i], deser.second[i]));
    }
  };
  // 各种不同参数下的测试运行
  run("", {});
  run("hi", {});
  run("", {torch::randn({5, 5})});
  run("hi", {torch::randn({5, 5})});
  run("more", {torch::randn({5, 5}), torch::rand({10, 10})});
}

// 定义测试用例 WireSerialize.RecopySparseTensors
TEST(WireSerialize, RecopySparseTensors) {
  // 定义常量 k1K 为 1024
  constexpr size_t k1K = 1024;
  // 创建大小为 k1K x k1K 的随机张量 main
  at::Tensor main = torch::randn({k1K, k1K});
  // 从 main 中选择中间的一行作为张量 tiny
  at::Tensor tiny = main.select(0, 2);
  // 断言 tiny 的元素数量为 k1K
  EXPECT_EQ(tiny.numel(), k1K);
  // 断言 tiny 存储的字节数除以元素类型大小为 k1K x k1K
  EXPECT_EQ(tiny.storage().nbytes() / tiny.dtype().itemsize(), k1K * k1K);
  // 调用 torch::distributed::rpc::wireSerialize 序列化 tiny
  auto ser = torch::distributed::rpc::wireSerialize({}, {tiny});
  // 调用 torch::distributed::rpc::wireDeserialize 反序列化 ser
  auto deser = torch::distributed::rpc::wireDeserialize(ser.data(), ser.size());
  // 断言 tiny 和 deser 中的第一个元素相等
  EXPECT_TRUE(torch::equal(tiny, deser.second[0]));
  // 断言 ser 大小小于 (tiny.element_size() * k1K) + k1K
  EXPECT_LT(ser.size(), (tiny.element_size() * k1K) + k1K);
}

// 定义测试用例 WireSerialize.CloneSparseTensors
TEST(WireSerialize, CloneSparseTensors) {
  // 定义常量 k1K 为 1024
  constexpr size_t k1K = 1024;
  // 创建大小为 k1K x k1K 的随机张量 big
  at::Tensor big = torch::randn({k1K, k1K});
  // 调用 torch::distributed::rpc::cloneSparseTensors 克隆 big
  auto v1 = torch::distributed::rpc::cloneSparseTensors({big});
  // 断言 v1 中第一个张量的存储与 big 相同
  EXPECT_EQ(v1.get(0).storage(), big.storage()); // Not cloned

  // 从 big 中选择中间的一行作为张量 tiny
  at::Tensor tiny = big.select(0, 2);
  // 调用 torch::distributed::rpc::cloneSparseTensors 克隆 tiny
  auto v2 = torch::distributed::rpc::cloneSparseTensors({tiny});
  // 断言 v2 中第一个张量的存储不同于 tiny 的存储，即已克隆
  EXPECT_NE(&v2.get(0).storage(), &tiny.storage()); // Cloned.
  // 断言 v2 中第一个张量与 tiny 相等
  EXPECT_TRUE(torch::equal(v2.get(0), tiny));

  // 创建一个大小为 2x3 的稀疏张量 sparse
  at::Tensor sparse = at::empty({2, 3}, at::dtype<float>().layout(at::kSparse));
  // 调用 torch::distributed::rpc::cloneSparseTensors 克隆 sparse
  auto v3 = torch::distributed::rpc::cloneSparseTensors({sparse});
  // 无法比较存储，但确认 v3 中第一个张量与 sparse 相同
  EXPECT_TRUE(v3.get(0).is_same(sparse));
}

// 定义测试用例 WireSerialize.Errors
TEST(WireSerialize, Errors) {
  // 定义内部函数 checkMessage，用于测试异常消息
  auto checkMessage = [](auto&& f, const char* msg) {
    try {
      f();  // 调用函数 f
      FAIL();  // 如果没有抛出异常则失败
    } catch (const std::exception& e) {
      // 断言异常消息包含特定的 msg 字符串
      EXPECT_PRED_FORMAT2(IsSubstring, msg, e.what());
    } catch (...) {
      FAIL();  // 捕获其他异常也算失败
    }
  };
    }
  };

// 定义一个空的 lambda 函数，不接受参数，不执行任何操作


  checkMessage(
      []() { (void)torch::distributed::rpc::wireDeserialize("", 0); },
      "failed parse");

// 调用 checkMessage 函数，传入一个 lambda 函数作为参数，该 lambda 函数调用 torch 库中的 wireDeserialize 函数，尝试对空字符串进行反序列化，检查是否解析失败，如果失败则返回消息 "failed parse"


  checkMessage(
      []() { (void)torch::distributed::rpc::wireDeserialize(" ", 1); },
      "failed parse");

// 调用 checkMessage 函数，传入一个 lambda 函数作为参数，该 lambda 函数调用 torch 库中的 wireDeserialize 函数，尝试对包含空格的字符串进行反序列化，检查是否解析失败，如果失败则返回消息 "failed parse"


  auto serialized =
      torch::distributed::rpc::wireSerialize({}, {torch::randn({5, 5})});

// 调用 torch 库中的 wireSerialize 函数，使用空的请求消息和一个随机生成的 5x5 张量作为数据，将其序列化并存储在变量 serialized 中


  checkMessage(
      [&]() {
        (void)torch::distributed::rpc::wireDeserialize(
            serialized.data(), serialized.size() / 2);
      },
      "failed bounds");

// 调用 checkMessage 函数，传入一个 lambda 函数作为参数，该 lambda 函数捕获了所有外部变量（使用 [&](){} 的形式），在 lambda 函数中调用 torch 库中的 wireDeserialize 函数，尝试对 serialized 的前半部分数据进行反序列化，检查是否超出界限，如果超出则返回消息 "failed bounds"
}

// 禁用此测试，因为 JIT Pickler 尚不支持稀疏张量。
TEST(WireSerialize, DISABLED_Sparse) {
  // 创建一个空的稀疏张量 main，大小为 2x3，数据类型为 float，布局为稀疏布局
  at::Tensor main = at::empty({2, 3}, at::dtype<float>().layout(at::kSparse));
  // 将稀疏张量 main 转换为稀疏格式，并进行序列化
  auto ser = torch::distributed::rpc::wireSerialize({}, {main.to(at::kSparse)});
  // 从序列化数据中反序列化，得到 deser 对象
  auto deser = torch::distributed::rpc::wireDeserialize(ser.data(), ser.size());
  // 断言 main 和 deser 中的第一个稀疏张量相等
  EXPECT_TRUE(torch::equal(main, deser.second[0]));
}
```