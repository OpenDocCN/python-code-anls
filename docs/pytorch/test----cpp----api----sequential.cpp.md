# `.\pytorch\test\cpp\api\sequential.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <c10/util/irange.h>  // 引入 C10 库中的工具 irange 头文件
#include <torch/torch.h>  // 引入 PyTorch 的头文件

#include <algorithm>  // 引入算法标准库
#include <memory>  // 引入内存管理标准库中的智能指针
#include <vector>  // 引入向量容器标准库

#include <test/cpp/api/support.h>  // 引入测试辅助函数支持库中的头文件

using namespace torch::nn;  // 使用 PyTorch 的 nn 命名空间
using namespace torch::test;  // 使用 PyTorch 测试框架的命名空间

struct SequentialTest : torch::test::SeedingFixture {};  // 定义测试结构体 SequentialTest，继承自 SeedingFixture 类

TEST_F(SequentialTest, CanContainThings) {
  Sequential sequential(Linear(3, 4), ReLU(), BatchNorm1d(3));  // 创建一个 Sequential 对象，包含 Linear、ReLU 和 BatchNorm1d 层
}

TEST_F(SequentialTest, ConstructsFromSharedPointer) {
  // 定义一个简单的 Module 结构体 M
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
    int forward() {
      return value;
    }
  };

  // 使用 shared_ptr 创建 Sequential 对象，包含三个 M 类型的共享指针
  Sequential sequential(
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3));
  ASSERT_EQ(sequential->size(), 3);  // 断言 sequential 中的模块数量为 3

  // 使用初始化列表和 shared_ptr 创建带有名称的 Sequential 对象
  Sequential sequential_named(
      {{"m1", std::make_shared<M>(1)},
       {std::string("m2"), std::make_shared<M>(2)},
       {"m3", std::make_shared<M>(3)}});
  ASSERT_EQ(sequential->size(), 3);  // 断言 sequential_named 中的模块数量为 3
}

TEST_F(SequentialTest, ConstructsFromConcreteType) {
  static int copy_count;

  // 定义一个简单的 Module 结构体 M，包含拷贝构造函数以统计拷贝次数
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    M(const M& other) : torch::nn::Module(other) {
      copy_count++;
    }
    int value;
    int forward() {
      return value;
    }
  };

  copy_count = 0;
  // 使用具体类型创建 Sequential 对象，包含三个 M 类型对象
  Sequential sequential(M(1), M(2), M(3));
  ASSERT_EQ(sequential->size(), 3);  // 断言 sequential 中的模块数量为 3
  // 断言拷贝构造函数被调用的次数等于 3，每个模块被传递给 std::make_shared<T>() 时会触发一次拷贝
  ASSERT_EQ(copy_count, 3);

  copy_count = 0;
  // 使用初始化列表和具体类型创建带有名称的 Sequential 对象
  Sequential sequential_named(
      {{"m1", M(1)}, {std::string("m2"), M(2)}, {"m3", M(3)}});
  ASSERT_EQ(sequential->size(), 3);  // 断言 sequential_named 中的模块数量为 3
  ASSERT_EQ(copy_count, 3);  // 断言拷贝构造函数被调用的次数等于 3
}

TEST_F(SequentialTest, ConstructsFromModuleHolder) {
  // 定义一个简单的 ModuleImpl 结构体 MImpl
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int forward() {
      return value;
    }
    int value;
  };

  // 定义 ModuleHolder 结构体 M，持有 MImpl 类型
  struct M : torch::nn::ModuleHolder<MImpl> {
    using torch::nn::ModuleHolder<MImpl>::ModuleHolder;
    using torch::nn::ModuleHolder<MImpl>::get;
  };

  // 使用 ModuleHolder<MImpl> 创建 Sequential 对象，包含三个 MImpl 类型对象
  Sequential sequential(M(1), M(2), M(3));
  ASSERT_EQ(sequential->size(), 3);  // 断言 sequential 中的模块数量为 3

  // 使用初始化列表和 ModuleHolder<MImpl> 创建带有名称的 Sequential 对象
  Sequential sequential_named(
      {{"m1", M(1)}, {std::string("m2"), M(2)}, {"m3", M(3)}});
  ASSERT_EQ(sequential->size(), 3);  // 断言 sequential_named 中的模块数量为 3
}

TEST_F(SequentialTest, PushBackAddsAnElement) {
  // 定义一个简单的 Module 结构体 M
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int forward() {
      return value;
    }
    // 定义一个整型变量 value
    int value;
  };

  // 测试未命名子模块
  // 创建一个 Sequential 对象 sequential
  Sequential sequential;
  // 断言 sequential 的大小为 0
  ASSERT_EQ(sequential->size(), 0);
  // 断言 sequential 是否为空
  ASSERT_TRUE(sequential->is_empty());
  // 向 sequential 中添加一个 Linear(3, 4) 的未命名子模块
  sequential->push_back(Linear(3, 4));
  // 断言 sequential 的大小为 1
  ASSERT_EQ(sequential->size(), 1);
  // 向 sequential 中添加一个使用 std::make_shared 创建的 M(1) 的未命名子模块
  sequential->push_back(std::make_shared<M>(1));
  // 断言 sequential 的大小为 2
  ASSERT_EQ(sequential->size(), 2);
  // 向 sequential 中添加一个 M(2) 的未命名子模块
  sequential->push_back(M(2));
  // 断言 sequential 的大小为 3
  ASSERT_EQ(sequential->size(), 3);

  // 混合命名和未命名子模块的测试
  // 创建一个带有命名子模块的 Sequential 对象 sequential_named
  Sequential sequential_named;
  // 断言 sequential_named 的大小为 0
  ASSERT_EQ(sequential_named->size(), 0);
  // 断言 sequential_named 是否为空
  ASSERT_TRUE(sequential_named->is_empty());

  // 向 sequential_named 中添加一个 Linear(3, 4) 的未命名子模块
  sequential_named->push_back(Linear(3, 4));
  // 断言 sequential_named 的大小为 1
  ASSERT_EQ(sequential_named->size(), 1);
  // 断言第一个命名子模块的键值为 "0"
  ASSERT_EQ(sequential_named->named_children()[0].key(), "0");

  // 向 sequential_named 中添加一个命名为 "linear2" 的 Linear(3, 4) 子模块
  sequential_named->push_back(std::string("linear2"), Linear(3, 4));
  // 断言 sequential_named 的大小为 2
  ASSERT_EQ(sequential_named->size(), 2);
  // 断言第二个命名子模块的键值为 "linear2"
  ASSERT_EQ(sequential_named->named_children()[1].key(), "linear2");

  // 向 sequential_named 中添加一个命名为 "shared_m1" 的 std::shared_ptr<M>(1) 子模块
  sequential_named->push_back("shared_m1", std::make_shared<M>(1));
  // 断言 sequential_named 的大小为 3
  ASSERT_EQ(sequential_named->size(), 3);
  // 断言第三个命名子模块的键值为 "shared_m1"
  ASSERT_EQ(sequential_named->named_children()[2].key(), "shared_m1");

  // 向 sequential_named 中添加一个使用 std::make_shared 创建的 M(1) 的未命名子模块
  sequential_named->push_back(std::make_shared<M>(1));
  // 断言 sequential_named 的大小为 4
  ASSERT_EQ(sequential_named->size(), 4);
  // 断言第四个命名子模块的键值为 "3"
  ASSERT_EQ(sequential_named->named_children()[3].key(), "3");

  // 向 sequential_named 中添加一个 M(1) 的未命名子模块
  sequential_named->push_back(M(1));
  // 断言 sequential_named 的大小为 5
  ASSERT_EQ(sequential_named->size(), 5);
  // 断言第五个命名子模块的键值为 "4"
  ASSERT_EQ(sequential_named->named_children()[4].key(), "4");

  // 向 sequential_named 中添加一个命名为 "m2" 的 M(1) 子模块
  sequential_named->push_back(std::string("m2"), M(1));
  // 断言 sequential_named 的大小为 6
  ASSERT_EQ(sequential_named->size(), 6);
  // 断言第六个命名子模块的键值为 "m2"

  // 命名和未命名 AnyModule 的测试
  // 创建一个带有 AnyModule 的 Sequential 对象 sequential_any
  Sequential sequential_any;
  // 创建一个包含 Linear(1, 2) 的 AnyModule a
  auto a = torch::nn::AnyModule(torch::nn::Linear(1, 2));
  // 断言 sequential_any 的大小为 0
  ASSERT_EQ(sequential_any->size(), 0);
  // 断言 sequential_any 是否为空
  ASSERT_TRUE(sequential_any->is_empty());
  // 向 sequential_any 中添加 AnyModule a
  sequential_any->push_back(a);
  // 断言 sequential_any 的大小为 1
  ASSERT_EQ(sequential_any->size(), 1);
  // 断言第一个命名子模块的键值为 "0"
  ASSERT_EQ(sequential_any->named_children()[0].key(), "0");
  // 向 sequential_any 中添加一个命名为 "fc" 的 AnyModule a
  sequential_any->push_back("fc", a);
  // 断言 sequential_any 的大小为 2
  ASSERT_EQ(sequential_any->size(), 2);
  // 断言第二个命名子模块的键值为 "fc"
  ASSERT_EQ(sequential_any->named_children()[1].key(), "fc");
}

TEST_F(SequentialTest, AccessWithAt) {
  // 定义一个内部模块 M，继承自 torch::nn::Module，包含一个值 value
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    // 定义 forward 方法，返回模块的值 value
    int forward() {
      return value;
    }
    int value;
  };
  // 创建包含三个共享指针的模块向量 modules，每个指针指向一个 M 模块实例
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  // 创建一个 Sequential 对象
  Sequential sequential;
  // 将 modules 中的模块依次添加到 Sequential 对象中
  for (auto& module : modules) {
    sequential->push_back(module);
  }
  // 断言 Sequential 对象中模块的数量为 3
  ASSERT_EQ(sequential->size(), 3);

  // 遍历 modules 中的模块索引，验证通过 at 方法获取的模块是否与 modules 中的相同
  for (const auto i : c10::irange(modules.size())) {
    ASSERT_EQ(&sequential->at<M>(i), modules[i].get());
  }

  // 对于超出索引范围的情况，验证 at 方法是否会抛出异常并带有正确的错误消息
  ASSERT_THROWS_WITH(
      sequential->at<M>(modules.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(
      sequential->at<M>(modules.size() + 1000000), "Index out of range");
}

TEST_F(SequentialTest, AccessWithPtr) {
  // 与前一个测试类似，定义 M 模块，创建 modules 向量，并将其添加到 Sequential 对象中
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int forward() {
      return value;
    }
    int value;
  };
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  Sequential sequential;
  for (auto& module : modules) {
    sequential->push_back(module);
  }
  ASSERT_EQ(sequential->size(), 3);

  // 遍历 modules 中的模块索引，通过 ptr 方法和数组索引操作符获取模块，并验证是否与 modules 中的相同
  for (const auto i : c10::irange(modules.size())) {
    ASSERT_EQ(sequential->ptr(i).get(), modules[i].get());
    ASSERT_EQ(sequential[i].get(), modules[i].get());
    ASSERT_EQ(sequential->ptr<M>(i).get(), modules[i].get());
  }

  // 对于超出索引范围的情况，验证 ptr 方法是否会抛出异常并带有正确的错误消息
  ASSERT_THROWS_WITH(sequential->ptr(modules.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(
      sequential->ptr(modules.size() + 1000000), "Index out of range");
}

TEST_F(SequentialTest, CallingForwardOnEmptySequentialIsDisallowed) {
  // 创建一个空的 Sequential 对象 empty
  Sequential empty;
  // 断言调用 empty 对象的 forward<int>() 方法会抛出异常，并且异常消息应为 "Cannot call forward() on an empty Sequential"
  ASSERT_THROWS_WITH(
      empty->forward<int>(), "Cannot call forward() on an empty Sequential");
}

TEST_F(SequentialTest, CallingForwardChainsCorrectly) {
  // 定义 MockModule，继承自 torch::nn::Module，包含一个期望值 expected
  struct MockModule : torch::nn::Module {
    explicit MockModule(int value) : expected(value) {}
    int expected;
    // 定义 forward 方法，验证输入值与期望值相等后返回增加的值
    int forward(int value) {
      assert(value == expected);
      return value + 1;
    }
  };

  // 创建 Sequential 对象 sequential，并初始化三个 MockModule 实例
  Sequential sequential(MockModule{1}, MockModule{2}, MockModule{3});

  // 断言调用 sequential 对象的 forward<int>(1) 方法返回值为 4
  ASSERT_EQ(sequential->forward<int>(1), 4);
}

TEST_F(SequentialTest, CallingForwardWithTheWrongReturnTypeThrows) {
  // 定义 M，继承自 torch::nn::Module，包含 forward 方法返回固定的整数值
  struct M : public torch::nn::Module {
    int forward() {
      return 5;
    }
  };

  // 创建包含 M 模块的 Sequential 对象 sequential
  Sequential sequential(M{});
  // 断言调用 sequential 对象的 forward<int>() 方法返回值为 5
  ASSERT_EQ(sequential->forward<int>(), 5);
  // 断言调用 sequential 对象的 forward<float>() 方法会抛出异常，并且异常消息应为 "The type of the return value is int, but you asked for type float"
  ASSERT_THROWS_WITH(
      sequential->forward<float>(),
      "The type of the return value is int, but you asked for type float");
}

TEST_F(SequentialTest, TheReturnTypeOfForwardDefaultsToTensor) {
  // 定义 M，继承自 torch::nn::Module，包含 forward 方法接收和返回 torch::Tensor
  struct M : public torch::nn::Module {
    torch::Tensor forward(torch::Tensor v) {
      return v;
    }
  };

  // 创建包含 M 模块的 Sequential 对象 sequential
  Sequential sequential(M{});
    }
  };



// 定义一个名为 sequential 的 Sequential 对象，使用默认构造函数初始化
Sequential sequential(M{});



// 创建一个变量 variable，其值为一个 3x3 的张量，要求梯度信息
auto variable = torch::ones({3, 3}, torch::requires_grad());



// 断言：使用 sequential 对象对 variable 进行前向传播的结果与 variable 自身相等
ASSERT_TRUE(sequential->forward(variable).equal(variable));


这些注释解释了每行代码的作用，包括定义对象、创建变量和进行断言检查。
}

TEST_F(SequentialTest, ForwardReturnsTheLastValue) {
  // 设置随机种子为0
  torch::manual_seed(0);
  // 创建一个包含三个线性层的Sequential模型
  Sequential sequential(Linear(10, 3), Linear(3, 5), Linear(5, 100));

  // 生成一个大小为[1000, 10]的随机张量，并要求梯度计算
  auto x = torch::randn({1000, 10}, torch::requires_grad());
  // 将输入张量通过Sequential模型进行前向传播
  auto y = sequential->forward(x);
  // 断言输出张量y的维度为2
  ASSERT_EQ(y.ndimension(), 2);
  // 断言输出张量y的第0维大小为1000
  ASSERT_EQ(y.size(0), 1000);
  // 断言输出张量y的第1维大小为100
  ASSERT_EQ(y.size(1), 100);
}

TEST_F(SequentialTest, SanityCheckForHoldingStandardModules) {
  // 创建一个包含多个标准模块的Sequential模型
  Sequential sequential(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
}

TEST_F(SequentialTest, ExtendPushesModulesFromOtherSequential) {
  // 定义四个自定义Module A, B, C, D
  struct A : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  struct B : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  struct C : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  struct D : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };

  // 创建两个包含不同自定义模块的Sequential对象a和b
  Sequential a(A{}, B{});
  Sequential b(C{}, D{});

  // 将Sequential b中的模块扩展到Sequential a中
  a->extend(*b);

  // 断言扩展后a中模块的数量为4
  ASSERT_EQ(a->size(), 4);
  // 断言a的第0个模块为类型A
  ASSERT_TRUE(a[0]->as<A>());
  // 断言a的第1个模块为类型B
  ASSERT_TRUE(a[1]->as<B>());
  // 断言a的第2个模块为类型C
  ASSERT_TRUE(a[2]->as<C>());
  // 断言a的第3个模块为类型D
  ASSERT_TRUE(a[3]->as<D>());

  // 断言b中模块的数量为2
  ASSERT_EQ(b->size(), 2);
  // 断言b的第0个模块为类型C
  ASSERT_TRUE(b[0]->as<C>());
  // 断言b的第1个模块为类型D()

  // 创建两个A类型的shared_ptr，并将它们扩展到Sequential b中
  std::vector<std::shared_ptr<A>> c = {
      std::make_shared<A>(), std::make_shared<A>()};
  b->extend(c);

  // 断言扩展后b中模块的数量为4
  ASSERT_EQ(b->size(), 4);
  // 断言b的第0个模块为类型C
  ASSERT_TRUE(b[0]->as<C>());
  // 断言b的第1个模块为类型D
  ASSERT_TRUE(b[1]->as<D>());
  // 断言b的第2个模块为类型A
  ASSERT_TRUE(b[2]->as<A>());
  // 断言b的第3个模块为类型A
  ASSERT_TRUE(b[3]->as<A>());
}

TEST_F(SequentialTest, HasReferenceSemantics) {
  // 创建一个包含三个线性层的Sequential模型first
  Sequential first(Linear(2, 3), Linear(4, 4), Linear(4, 5));
  // 创建第二个Sequential模型second，并复制first的内容
  Sequential second(first);

  // 断言first和second指向同一内存地址
  ASSERT_EQ(first.get(), second.get());
  // 断言first和second的模块数量相等
  ASSERT_EQ(first->size(), second->size());
  // 断言first和second每个位置上的模块指针相等
  ASSERT_TRUE(std::equal(
      first->begin(),
      first->end(),
      second->begin(),
      [](const AnyModule& first, const AnyModule& second) {
        return &first == &second;
      }));
}

TEST_F(SequentialTest, IsCloneable) {
  // 创建一个包含Linear、Functional和BatchNorm1d模块的Sequential模型sequential
  Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));
  // 克隆Sequential模型sequential并赋值给clone
  Sequential clone =
      std::dynamic_pointer_cast<SequentialImpl>(sequential->clone());
  
  // 断言sequential和clone的模块数量相等
  ASSERT_EQ(sequential->size(), clone->size());

  // 逐个比较sequential和clone中的模块类型是否相同但不是同一对象
  for (size_t i = 0; i < sequential->size(); ++i) {
    // 断言sequential和clone对应位置上的模块类型相同
    ASSERT_EQ(sequential[i]->name(), clone[i]->name());
    // 断言sequential和clone对应位置上的模块对象不相等
    ASSERT_NE(sequential[i], clone[i]);
  }

  // 验证克隆是否深层次的，即模块参数也被克隆了
  torch::NoGradGuard no_grad;

  auto params1 = sequential->named_parameters();
  auto params2 = clone->named_parameters();
  ASSERT_EQ(params1.size(), params2.size());
  for (auto& param : params1) {
    // 断言参数param和params2[param.key()]不是同一个对象
    ASSERT_FALSE(pointer_equal(param.value(), params2[param.key()]));
    // 断言参数param和params2[param.key()]在同一设备上
    ASSERT_EQ(param->device(), params2[param.key()].device());
    // 断言参数param和params2[param.key()]的值相近
    ASSERT_TRUE(param->allclose(params2[param.key()]));
    // 修改param的值，验证是否独立
    param->add_(2);
  }
  for (auto& param : params1) {
    // 重新验证修改后的param和params2[param.key()]是否仍然不相等
    // 并且验证修改后的param和params2[param.key()]的设备和值是否相近
    ASSERT_FALSE(param->allclose(params2[param.key()]));



// 断言语句，用于检查 param 对应的参数值与 params2 中相同键的参数值是否不全部接近
ASSERT_FALSE(param->allclose(params2[param.key()]));


这段代码是一个断言语句，它用于在程序运行时检查条件是否满足。`ASSERT_FALSE` 是一个宏或函数，通常在调试阶段使用，用于确保某个条件为假。在这里，它断言了一个条件：`param` 对应的参数值与 `params2` 中相同键的参数值不全部接近。`param->allclose(params2[param.key()])` 可能是一个函数或方法调用，用于比较两个参数值是否接近。
TEST_F(SequentialTest, RegistersElementsAsSubmodules) {
  // 创建一个 Sequential 对象并初始化其子模块
  Sequential sequential(Linear(10, 3), Conv2d(1, 2, 3), Dropout2d(0.5));

  // 获取顺序容器中的所有子模块
  auto modules = sequential->children();

  // 断言第一个子模块是 Linear 类型
  ASSERT_TRUE(modules[0]->as<Linear>());
  
  // 断言第二个子模块是 Conv2d 类型
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  
  // 断言第三个子模块是 Dropout2d 类型
  ASSERT_TRUE(modules[2]->as<Dropout2d>());
}

TEST_F(SequentialTest, CloneToDevice_CUDA) {
  // 创建一个 Sequential 对象并初始化其子模块
  Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));
  torch::Device device(torch::kCUDA, 0);
  
  // 将 sequential 克隆到指定的 CUDA 设备上
  Sequential clone =
      std::dynamic_pointer_cast<SequentialImpl>(sequential->clone(device));
  
  // 断言克隆后的参数都在指定的设备上
  for (const auto& p : clone->parameters()) {
    ASSERT_EQ(p.device(), device);
  }
  
  // 断言克隆后的缓冲区都在指定的设备上
  for (const auto& b : clone->buffers()) {
    ASSERT_EQ(b.device(), device);
  }
}

TEST_F(SequentialTest, PrettyPrintSequential) {
  // 创建一个 Sequential 对象并初始化其多个子模块
  Sequential sequential(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
  
  // 断言序列化后的字符串与预期输出相匹配
  ASSERT_EQ(
      c10::str(sequential),
      "torch::nn::Sequential(\n"
      "  (0): torch::nn::Linear(in_features=10, out_features=3, bias=true)\n"
      "  (1): torch::nn::Conv2d(1, 2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (2): torch::nn::Dropout(p=0.5, inplace=false)\n"
      "  (3): torch::nn::BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=true, track_running_stats=true)\n"
      "  (4): torch::nn::Embedding(num_embeddings=4, embedding_dim=10)\n"
      "  (5): torch::nn::LSTM(input_size=4, hidden_size=5, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)\n"
      ")");
  
  // 创建一个带命名子模块的 Sequential 对象
  Sequential sequential_named(
      {{"linear", Linear(10, 3)},
       {"conv2d", Conv2d(1, 2, 3)},
       {"dropout", Dropout(0.5)},
       {"batchnorm2d", BatchNorm2d(5)},
       {"embedding", Embedding(4, 10)},
       {"lstm", LSTM(4, 5)}});
  
  // 断言序列化后的字符串与预期输出相匹配
  ASSERT_EQ(
      c10::str(sequential_named),
      "torch::nn::Sequential(\n"
      "  (linear): torch::nn::Linear(in_features=10, out_features=3, bias=true)\n"
      "  (conv2d): torch::nn::Conv2d(1, 2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (dropout): torch::nn::Dropout(p=0.5, inplace=false)\n"
      "  (batchnorm2d): torch::nn::BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=true, track_running_stats=true)\n"
      "  (embedding): torch::nn::Embedding(num_embeddings=4, embedding_dim=10)\n"
      "  (lstm): torch::nn::LSTM(input_size=4, hidden_size=5, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)\n"
      ")");
}

TEST_F(SequentialTest, ModuleForwardMethodOptionalArg) {
  {
    // 创建一个包含 Identity 和 ConvTranspose1d 子模块的 Sequential 对象
    Sequential sequential(
        Identity(),
        ConvTranspose1d(ConvTranspose1dOptions(3, 2, 3).stride(1).bias(false)));
    
    // 设置 ConvTranspose1d 的权重数据
    std::dynamic_pointer_cast<ConvTranspose1dImpl>(sequential[1])
        ->weight.set_data(torch::arange(18.).reshape({3, 2, 3}));
    
    // 创建输入张量 x
    auto x = torch::arange(30.).reshape({2, 3, 5});
    
    // 对输入张量进行前向传播
    auto y = sequential->forward(x);
    ```
    {
        // 创建期望的张量，包含预先定义的数值
        auto expected = torch::tensor(
            {{{150., 333., 552., 615., 678., 501., 276.},
              {195., 432., 714., 804., 894., 654., 357.}},
             {{420., 918., 1497., 1560., 1623., 1176., 636.},
              {600., 1287., 2064., 2154., 2244., 1599., 852.}}});
        // 断言张量 y 与期望张量 expected 在数值上接近
        ASSERT_TRUE(torch::allclose(y, expected));
      }
      {
        // 创建一个包含特定层的序列模型
        Sequential sequential(
            Identity(),  // 第一层为恒等映射 Identity
            ConvTranspose2d(ConvTranspose2dOptions(3, 2, 3).stride(1).bias(false)));  // 第二层为转置卷积层，设置参数
        // 设置转置卷积层的权重数据为指定的范围数值
        std::dynamic_pointer_cast<ConvTranspose2dImpl>(sequential[1])
            ->weight.set_data(torch::arange(54.).reshape({3, 2, 3, 3}));
        // 创建输入张量 x，数据为指定范围的数值
        auto x = torch::arange(75.).reshape({1, 3, 5, 5});
        // 对输入张量 x 进行前向传播，得到输出张量 y
        auto y = sequential->forward(x);
        // 创建期望的输出张量 expected，包含预先定义的数值
        auto expected = torch::tensor(
            {{{{2250., 4629., 7140., 7311., 7482., 5133., 2640.},
               {4995., 10272., 15837., 16206., 16575., 11364., 5841.},
               {8280., 17019., 26226., 26820., 27414., 18783., 9648.},
               {9225., 18954., 29196., 29790., 30384., 20808., 10683.},
               {10170., 20889., 32166., 32760., 33354., 22833., 11718.},
               {7515., 15420., 23721., 24144., 24567., 16800., 8613.},
               {4140., 8487., 13044., 13269., 13494., 9219., 4722.}},
              {{2925., 6006., 9246., 9498., 9750., 6672., 3423.},
               {6480., 13296., 20454., 20985., 21516., 14712., 7542.},
               {10710., 21960., 33759., 34596., 35433., 24210., 12402.},
               {12060., 24705., 37944., 38781., 39618., 27045., 13842.},
               {13410., 27450., 42129., 42966., 43803., 29880., 15282.},
               {9810., 20064., 30768., 31353., 31938., 21768., 11124.},
               {5355., 10944., 16770., 17076., 17382., 11838., 6045.}}}});
        // 断言张量 y 与期望张量 expected 在数值上接近
        ASSERT_TRUE(torch::allclose(y, expected));
      }
      {
        // 创建一个包含特定层的序列模型
        Sequential sequential(
            Identity(),  // 第一层为恒等映射 Identity
            ConvTranspose3d(ConvTranspose3dOptions(2, 2, 2).stride(1).bias(false)));  // 第二层为三维转置卷积层，设置参数
        // 设置三维转置卷积层的权重数据为指定的范围数值
        std::dynamic_pointer_cast<ConvTranspose3dImpl>(sequential[1])
            ->weight.set_data(torch::arange(32.).reshape({2, 2, 2, 2, 2}));
        // 创建输入张量 x，数据为指定范围的数值
        auto x = torch::arange(16.).reshape({1, 2, 2, 2, 2});
        // 对输入张量 x 进行前向传播，得到输出张量 y
        auto y = sequential->forward(x);
        // 创建期望的输出张量 expected，包含预先定义的数值
        auto expected = torch::tensor(
            {{{{{128., 280., 154.}, {304., 664., 364.}, {184., 400., 218.}},
               {{352., 768., 420.}, {832., 1808., 984.}, {496., 1072., 580.}},
               {{256., 552., 298.}, {592., 1272., 684.}, {344., 736., 394.}}},
              {{{192., 424., 234.}, {464., 1016., 556.}, {280., 608., 330.}},
               {{544., 1184., 644.}, {1280., 2768., 1496.}, {752., 1616., 868.}},
               {{384., 824., 442.}, {880., 1880., 1004.}, {504., 1072., 570.}}}}});
        // 断言张量 y 与期望张量 expected 在数值上接近
        ASSERT_TRUE(torch::allclose(y, expected));
      }
      {
        // 创建一个权重张量，包含预先定义的数值
        auto weight = torch::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
        // 创建一个包含特定层的序列模型
        Sequential sequential(Identity(), EmbeddingBag::from_pretrained(weight));
        // 创建输入张量 x，数据为指定的数值和类型
        auto x = torch::tensor({{1, 0}}, torch::kLong);
        // 对输入张量 x 进行前向传播，得到输出张量 y
        auto y = sequential->forward(x);
        // 创建期望的输出张量 expected，包含预先定义的数值
        auto expected = torch::tensor({2.5000, 3.7000, 4.6500});
        // 断言张量 y 与期望张量 expected 在数值上接近
        ASSERT_TRUE(torch::allclose(y, expected));
      }
      {
        // 设置随机数种子为 0
        torch::manual_seed(0);
    // 设置嵌入维度为8
    int64_t embed_dim = 8;
    // 设置注意力头数为4
    int64_t num_heads = 4;
    // 设置批量大小为8
    int64_t batch_size = 8;
    // 设置源序列长度为3
    int64_t src_len = 3;
    // 设置目标序列长度为1
    int64_t tgt_len = 1;

    // 创建查询张量，维度为(batch_size, tgt_len, embed_dim)，所有元素初始化为1
    auto query = torch::ones({batch_size, tgt_len, embed_dim});
    // 创建键张量，维度为(batch_size, src_len, embed_dim)，所有元素初始化为1
    auto key = torch::ones({batch_size, src_len, embed_dim});
    // 复制键张量作为值张量
    auto value = key;

    // 创建一个多头注意力机制的序列模型
    Sequential sequential(MultiheadAttention(embed_dim, num_heads));
    // 执行序列模型的前向传播，传入转置后的查询、键和值张量，获取输出
    auto output = sequential->forward<std::tuple<torch::Tensor, torch::Tensor>>(
        query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1));

    // 获取注意力输出
    auto attn_output = std::get<0>(output);
    // 期望的注意力输出张量，每个元素都是预先定义的值
    auto attn_output_expected = torch::tensor(
        {{{0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674,
           -0.0056,
           0.1324,
           0.0922,
           0.0160,
           -0.0934,
           -0.1700,
           0.1663}}});
    // 断言实际的注意力输出与期望的输出在给定的误差范围内相等
    ASSERT_TRUE(
        torch::allclose(attn_output, attn_output_expected, 1e-05, 2e-04));

    // 获取注意力权重输出
    auto attn_output_weights = std::get<1>(output);
    // 期望的注意力权重输出张量，每个元素都是预先定义的值
    auto attn_output_weights_expected = torch::tensor(
        {{{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}}});
    // 断言实际的注意力权重输出与期望的输出在给定的误差范围内相等
    ASSERT_TRUE(torch::allclose(
        attn_output_weights, attn_output_weights_expected, 1e-05, 2e-04));
  }
  {
    // 创建索引张量，指定类型为64位整数，内容为{{1, 3, 4}}
    auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
    // 创建输入张量，指定类型为浮点数，内容为{{2, 4, 5}}
    auto x = torch::tensor({{{2, 4, 5}}}, torch::dtype(torch::kFloat));
    // 创建一个最大反池化1D的序列模型
    Sequential sequential(MaxUnpool1d(3));
    // 执行序列模型的前向传播，传入输入张量和索引张量，获取输出
    auto y = sequential->forward(x, indices);
    // 期望的输出张量，每个元素都是预先定义的值
    auto expected =
        torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat);
    // 断言实际的输出与期望的输出在给定的误差范围内相等
    ASSERT_TRUE(torch::allclose(y, expected));
  }
  {
    // 创建索引张量，指定类型为64位整数，内容为{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}}
    auto indices = torch::tensor(
        {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
         {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}}},
        torch::kLong);
    // 创建输入张量，指定类型为浮点数，内容为{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}}
    // 和{{{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}}
    auto x = torch::tensor(
        {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
         {{{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}}},
        torch::dtype(torch::kFloat));
    // 创建一个带有选项的2D最大反池化序列模型，设定步长为2，填充为1
    Sequential sequential(
        MaxUnpool2d(MaxUnpool2dOptions(3).stride(2).padding(1)));
    // 执行序列模型的前向传播，传入输入张量和索引张量，获取输出
    auto y = sequential->forward(x, indices);
  {
    // 设置随机种子为0，以确保结果可重复性
    torch::manual_seed(0);
    // 创建一个包含 Identity 和 RNN 的顺序模型
    Sequential sequential(Identity(), RNN(2, 3));
    // 创建输入张量 x，形状为 [2, 3, 2]，全为1
    auto x = torch::ones({2, 3, 2});
    // 对顺序模型进行前向传播，返回 RNN 的输出
    auto rnn_output =
        sequential->forward<std::tuple<torch::Tensor, torch::Tensor>>(x);
    // 期待的输出张量，形状为 [2, 3, 3]，包含预期的浮点数值
    auto expected_output = torch::tensor(
        {{{-0.0645, -0.7274, 0.4531},
          {-0.0645, -0.7274, 0.4531},
          {-0.0645, -0.7274, 0.4531}},
         {{-0.3970, -0.6950, 0.6009},
          {-0.3970, -0.6950, 0.6009},
          {-0.3970, -0.6950, 0.6009}}});
    // 断言计算的输出与期望的输出在一定误差范围内相等
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output), expected_output, 1e-05, 2e-04));
  }
  {
    // 设置随机种子为0，以确保结果可重复性
    torch::manual_seed(0);
    // 创建一个包含 Identity 和 LSTM 的顺序模型
    Sequential sequential(Identity(), LSTM(2, 3));
    // 创建输入张量 x，形状为 [2, 3, 2]，全为1
    auto x = torch::ones({2, 3, 2});
    // 对顺序模型进行前向传播，返回 LSTM 的输出及状态元组
    auto rnn_output = sequential->forward<
        std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>>(x);
    // 期待的输出张量，形状为 [2, 3, 3]，包含预期的浮点数值
    auto expected_output = torch::tensor(
        {{{-0.2693, -0.1240, 0.0744},
          {-0.2693, -0.1240, 0.0744},
          {-0.2693, -0.1240, 0.0744}},
         {{-0.3889, -0.1919, 0.1183},
          {-0.3889, -0.1919, 0.1183},
          {-0.3889, -0.1919, 0.1183}}});
    // 断言计算的输出与期望的输出在一定误差范围内相等
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output), expected_output, 1e-05, 2e-04));
  }
  {
    // 设置随机种子为0，以确保结果可重复性
    torch::manual_seed(0);
    // 创建一个包含 Identity 和 GRU 的顺序模型
    Sequential sequential(Identity(), GRU(2, 3));
    // 创建输入张量 x，形状为 [2, 3, 2]，全为1
    auto x = torch::ones({2, 3, 2});
    // 对顺序模型进行前向传播，返回 GRU 的输出
    auto rnn_output =
        sequential->forward<std::tuple<torch::Tensor, torch::Tensor>>(x);
    // 期待的输出张量，形状为 [2, 3, 3]，包含预期的浮点数值
    auto expected_output = torch::tensor(
        {{{-0.1134, 0.0467, 0.2336},
          {-0.1134, 0.0467, 0.2336},
          {-0.1134, 0.0467, 0.2336}},
         {{-0.1189, 0.0502, 0.2960},
          {-0.1189, 0.0502, 0.2960},
          {-0.1189, 0.0502, 0.2960}}});
    // 断言计算的输出与期望的输出在一定误差范围内相等
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output), expected_output, 1e-05, 2e-04));
  }
  {
    // 设置随机种子为0，以确保结果可重复性
    torch::manual_seed(0);
    // 创建一个包含 Identity 和 RNNCell 的顺序模型
    Sequential sequential(Identity(), RNNCell(2, 3));
    // 创建输入张量 x，形状为 [2, 2]，全为1
    auto x = torch::ones({2, 2});
    // 对顺序模型进行前向传播，返回 RNNCell 的输出
    auto rnn_output = sequential->forward<torch::Tensor>(x);
  {
    // 设置随机种子为0，以确保可重现性
    torch::manual_seed(0);
    // 创建包含一个Identity层和一个LSTMCell层的序列模型
    Sequential sequential(Identity(), LSTMCell(2, 3));
    // 创建一个2x2的全1张量作为输入
    auto x = torch::ones({2, 2});
    // 对序列模型进行前向传播，返回包含两个张量的元组
    auto rnn_output =
        sequential->forward<std::tuple<torch::Tensor, torch::Tensor>>(x);
    // 预期的输出张量
    auto expected_output =
        torch::tensor({{-0.2693, -0.1240, 0.0744}, {-0.2693, -0.1240, 0.0744}});
    // 使用torch::allclose检查计算输出是否接近预期输出
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output), expected_output, 1e-05, 2e-04));
  }
  {
    // 设置随机种子为0，以确保可重现性
    torch::manual_seed(0);
    // 创建包含一个Identity层和一个GRUCell层的序列模型
    Sequential sequential(Identity(), GRUCell(2, 3));
    // 创建一个2x2的全1张量作为输入
    auto x = torch::ones({2, 2});
    // 对序列模型进行前向传播，返回一个张量作为输出
    auto rnn_output = sequential->forward<torch::Tensor>(x);
    // 预期的输出张量
    auto expected_output =
        torch::tensor({{-0.1134, 0.0467, 0.2336}, {-0.1134, 0.0467, 0.2336}});
    // 使用torch::allclose检查计算输出是否接近预期输出
    ASSERT_TRUE(torch::allclose(rnn_output, expected_output, 1e-05, 2e-04));
  }
}


注释：


# 结束一个代码块或函数定义的语句，表示此处定义的函数或循环体结束
```