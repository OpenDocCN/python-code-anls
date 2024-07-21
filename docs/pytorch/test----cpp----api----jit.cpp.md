# `.\pytorch\test\cpp\api\jit.cpp`

```py
// 包含 Google Test 的头文件，用于单元测试
#include <gtest/gtest.h>

// 包含 PyTorch 的 JIT 编译和脚本化头文件
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/types.h>

// 包含标准库的字符串头文件
#include <string>

// 定义 TorchScriptTest 测试套件，测试多个函数的编译与执行
TEST(TorchScriptTest, CanCompileMultipleFunctions) {
  // 编译 TorchScript 模块，定义多个函数
  auto module = torch::jit::compile(R"JIT(
      def test_mul(a, b):
        return a * b
      def test_relu(a, b):
        return torch.relu(a + b)
      def test_while(a, i):
        while bool(i < 10):
          a += a
          i += 1
        return a
      def test_len(a : List[int]):
        return len(a)
    )JIT");

  // 创建张量 a 和 b，值为 1
  auto a = torch::ones(1);
  auto b = torch::ones(1);

  // 断言：调用 test_mul 函数，验证返回值是否为 1
  ASSERT_EQ(1, module->run_method("test_mul", a, b).toTensor().item<int64_t>());

  // 断言：调用 test_relu 函数，验证返回值是否为 2
  ASSERT_EQ(
      2, module->run_method("test_relu", a, b).toTensor().item<int64_t>());

  // 断言：调用 test_while 函数，验证返回值是否为 512
  ASSERT_TRUE(
      0x200 ==
      module->run_method("test_while", a, b).toTensor().item<int64_t>());

  // 创建包含整数列表的 IValue 对象 list
  at::IValue list = c10::List<int64_t>({3, 4});
  // 断言：调用 test_len 函数，验证返回值是否为 2
  ASSERT_EQ(2, module->run_method("test_len", list).toInt());
}

// 定义测试套件 TorchScriptTest，测试嵌套的 IValue 模块参数匹配
TEST(TorchScriptTest, TestNestedIValueModuleArgMatching) {
  // 编译 TorchScript 模块，定义嵌套循环函数 nested_loop
  auto module = torch::jit::compile(R"JIT(
      def nested_loop(a: List[List[Tensor]], b: int):
        return torch.tensor(1.0) + b
    )JIT");

  // 创建整数 b，值为 3
  auto b = 3;

  // 创建包含随机张量的列表 list
  torch::List<torch::Tensor> list({torch::rand({4, 4})});

  // 创建列表的列表 list_of_lists，包含 list
  torch::List<torch::List<torch::Tensor>> list_of_lists;
  list_of_lists.push_back(list);

  // 运行 nested_loop 函数，传入 list_of_lists 和 b
  module->run_method("nested_loop", list_of_lists, b);

  // 创建通用列表 generic_list，包含张量类型的 GenericList
  auto generic_list = c10::impl::GenericList(at::TensorType::get());

  // 创建空的通用列表 empty_generic_list，包含张量类型的列表
  auto empty_generic_list =
      c10::impl::GenericList(at::ListType::create(at::TensorType::get()));

  // 向 empty_generic_list 中添加 generic_list
  empty_generic_list.push_back(generic_list);

  // 创建太多列表的 too_many_lists，包含多重张量类型的列表
  auto too_many_lists = c10::impl::GenericList(
      at::ListType::create(at::ListType::create(at::TensorType::get())));

  // 向 too_many_lists 中添加 empty_generic_list
  too_many_lists.push_back(empty_generic_list);

  // 尝试运行 nested_loop 函数，传入 too_many_lists 和 b，捕获异常
  try {
    module->run_method("nested_loop", too_many_lists, b);
    // 断言：捕获到 c10::Error 异常，检查异常信息
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    // 断言：验证异常信息是否包含特定字符串
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("nested_loop() Expected a value of type 'List[List[Tensor]]'"
                  " for argument 'a' but instead found type "
                  "'List[List[List[Tensor]]]'") == 0);
  };
}

// 定义测试套件 TorchScriptTest，测试字典类型参数匹配
TEST(TorchScriptTest, TestDictArgMatching) {
  // 编译 TorchScript 模块，定义操作字典的函数 dict_op
  auto module = torch::jit::compile(R"JIT(
      def dict_op(a: Dict[str, Tensor], b: str):
        return a[b]
    )JIT");

  // 创建字符串到张量的字典 dict，包含键 "hello" 和张量 torch::ones({2})
  c10::Dict<std::string, at::Tensor> dict;
  dict.insert("hello", torch::ones({2}));

  // 运行 dict_op 函数，传入 dict 和字符串 "hello"
  auto output = module->run_method("dict_op", dict, std::string("hello"));
  // 断言：验证输出张量的第一个元素是否为 1
  ASSERT_EQ(1, output.toTensor()[0].item<int64_t>());
}

// 定义测试套件 TorchScriptTest，测试元组类型参数匹配
TEST(TorchScriptTest, TestTupleArgMatching) {
  // 编译 TorchScript 模块，定义操作元组的函数 tuple_op
  auto module = torch::jit::compile(R"JIT(
      def tuple_op(a: Tuple[List[int]]):
        return a
    )JIT");

  // 创建整数列表 int_list，包含元素 1
  c10::List<int64_t> int_list({1});

  // 创建包含 int_list 的通用元组 tuple_generic_list
  auto tuple_generic_list = c10::ivalue::Tuple::create({int_list});

  // 不会在参数匹配上失败的情况下运行 tuple_op 函数
  module->run_method("tuple_op", tuple_generic_list);
}
TEST(TorchScriptTest, TestOptionalArgMatching) {
  // 编译 TorchScript 模块，定义了一个可选参数为元组的函数 optional_tuple_op
  auto module = torch::jit::compile(R"JIT(
      def optional_tuple_op(a: Optional[Tuple[int, str]]):
        if a is None:
          return 0
        else:
          return a[0]
    )JIT");

  // 创建一个包含整数和字符串的元组作为参数
  auto optional_tuple = c10::ivalue::Tuple::create({2, std::string("hi")});

  // 断言调用 optional_tuple_op 函数后返回值为 2
  ASSERT_EQ(2, module->run_method("optional_tuple_op", optional_tuple).toInt());
  // 断言调用 optional_tuple_op 函数传入空参数后返回值为 0
  ASSERT_EQ(
      0, module->run_method("optional_tuple_op", torch::jit::IValue()).toInt());
}

TEST(TorchScriptTest, TestPickle) {
  // 创建一个包含浮点数值的 torch::IValue 对象
  torch::IValue float_value(2.3);

  // TODO: 当张量被存储在 pickle 中时，删除这部分代码
  // 创建一个用于存储张量的向量表
  std::vector<at::Tensor> tensor_table;
  // 对浮点数值进行 pickle 操作，并获取序列化数据
  auto data = torch::jit::pickle(float_value, &tensor_table);

  // 反序列化 pickle 数据
  torch::IValue ivalue = torch::jit::unpickle(data.data(), data.size());

  // 计算反序列化后的值与原始浮点数值的差异
  double diff = ivalue.toDouble() - float_value.toDouble();
  double eps = 0.0001;
  // 断言差异小于给定的阈值 eps
  ASSERT_TRUE(diff < eps && diff > -eps);
}
```