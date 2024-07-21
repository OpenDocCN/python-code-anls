# `.\pytorch\test\cpp\api\modulelist.cpp`

```
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

// 使用了 torch::nn 命名空间中的模块和测试支持
using namespace torch::nn;
using namespace torch::test;

// 测试夹具，继承自 SeedingFixture，用于初始化测试环境
struct ModuleListTest : torch::test::SeedingFixture {};

// 测试用例：从 shared pointer 构造 ModuleList
TEST_F(ModuleListTest, ConstructsFromSharedPointer) {
  // 定义一个简单的模块 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  // 使用 std::make_shared 创建三个 M 类型的对象，并传递给 ModuleList 构造函数
  ModuleList list(
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3));
  // 断言 ModuleList 的大小为 3
  ASSERT_EQ(list->size(), 3);
}

// 测试用例：从具体类型构造 ModuleList
TEST_F(ModuleListTest, ConstructsFromConcreteType) {
  static int copy_count;

  // 定义一个复杂的模块 M，包含拷贝构造函数，用于统计拷贝次数
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    // 禁止 clang-tidy 检查，用于拷贝构造函数
    M(const M& other) : torch::nn::Module(other) {
      copy_count++;
    }
    int value;
  };

  // 初始化拷贝次数计数器
  copy_count = 0;
  // 使用 M 类型直接构造 ModuleList
  ModuleList list(M(1), M(2), M(3));
  // 断言 ModuleList 的大小为 3
  ASSERT_EQ(list->size(), 3);
  // 断言拷贝次数为 3 次，因为每个模块在传递给 std::make_shared 时会发生一次拷贝
  ASSERT_EQ(copy_count, 3);
  // 注意：当前实现期望每个模块被拷贝一次，这发生在将模块传递给 `std::make_shared<T>()` 时
  // TODO: 找到一种避免拷贝的方法，然后删除 `M` 的拷贝构造函数
}

// 测试用例：从 ModuleHolder 构造 ModuleList
TEST_F(ModuleListTest, ConstructsFromModuleHolder) {
  // 定义一个简单的模块实现 MImpl，继承自 torch::nn::Module
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int value;
  };

  // 定义一个模块 M，继承自 ModuleHolder，包含 MImpl 类型的模块
  struct M : torch::nn::ModuleHolder<MImpl> {
    using torch::nn::ModuleHolder<MImpl>::ModuleHolder;
    using torch::nn::ModuleHolder<MImpl>::get;
  };

  // 使用 M 类型直接构造 ModuleList
  ModuleList list(M(1), M(2), M(3));
  // 断言 ModuleList 的大小为 3
  ASSERT_EQ(list->size(), 3);
}

// 测试用例：push_back 方法添加元素
TEST_F(ModuleListTest, PushBackAddsAnElement) {
  // 定义一个简单的模块 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  // 初始化空的 ModuleList，断言其大小为 0
  ModuleList list;
  ASSERT_EQ(list->size(), 0);
  // 断言 ModuleList 为空
  ASSERT_TRUE(list->is_empty());
  // 使用 push_back 添加一个 Linear 模块
  list->push_back(Linear(3, 4));
  // 断言 ModuleList 的大小为 1
  ASSERT_EQ(list->size(), 1);
  // 使用 push_back 添加一个通过 std::make_shared 创建的 M 类型模块
  list->push_back(std::make_shared<M>(1));
  // 断言 ModuleList 的大小为 2
  ASSERT_EQ(list->size(), 2);
  // 使用 push_back 直接添加一个 M 类型模块
  list->push_back(M(2));
  // 断言 ModuleList 的大小为 3
  ASSERT_EQ(list->size(), 3);
}

// 测试用例：insert 方法插入元素
TEST_F(ModuleListTest, Insertion) {
  // 定义一个简单的模块实现 MImpl，继承自 torch::nn::Module
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int value;
  };
  // 定义一个 TORCH_MODULE，简化 MImpl 的模块包装
  TORCH_MODULE(M);

  // 初始化空的 ModuleList
  ModuleList list;
  // 使用 push_back 添加一个 MImpl 类型的模块，断言 ModuleList 的大小为 1
  list->push_back(MImpl(1));
  ASSERT_EQ(list->size(), 1);
  // 使用 insert 在索引 0 处插入一个通过 std::make_shared 创建的 MImpl 类型模块
  list->insert(0, std::make_shared<MImpl>(2));
  // 断言 ModuleList 的大小为 2
  ASSERT_EQ(list->size(), 2);
  // 使用 insert 在索引 1 处插入一个 M 类型模块
  list->insert(1, M(3));
  // 断言 ModuleList 的大小为 3
  ASSERT_EQ(list->size(), 3);
  // 使用 insert 在索引 3 处插入一个 M 类型模块
  list->insert(3, M(4));
  // 断言 ModuleList 的大小为 4
  ASSERT_EQ(list->size(), 4);
  // 断言各个位置的值符合预期
  ASSERT_EQ(list->at<MImpl>(0).value, 2);
  ASSERT_EQ(list->at<MImpl>(1).value, 3);
  ASSERT_EQ(list->at<MImpl>(2).value, 1);
  ASSERT_EQ(list->at<MImpl>(3).value, 4);

  // 构造一个期望的无序映射 U，用于验证 named_modules 方法的结果
  std::unordered_map<size_t, size_t> U = {{0, 2}, {1, 3}, {2, 1}, {3, 4}};
  // 遍历 named_modules 的结果，验证每个模块的值
  for (const auto& P : list->named_modules("", false))
    ASSERT_EQ(U[std::stoul(P.key())], P.value()->as<M>()->value);
}

// 测试用例：使用 at 方法访问模块
TEST_F(ModuleListTest, AccessWithAt) {
  // 定义一个简单的模块 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
    // 定义一个整型变量 value
    int value;
  };
  // 创建包含三个 M 类型智能指针的 vector
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  // 创建 ModuleList 实例
  ModuleList list;
  // 将 modules 中的每个智能指针添加到 list 中
  for (auto& module : modules) {
    list->push_back(module);
  }
  // 断言 list 的大小为 3
  ASSERT_EQ(list->size(), 3);

  // 对于给定索引，返回正确的模块
  for (const auto i : c10::irange(modules.size())) {
    // 断言 list 中第 i 个位置的模块指针与 modules 中第 i 个智能指针的原始指针相同
    ASSERT_EQ(&list->at<M>(i), modules[i].get());
  }

  // 对于超出范围的索引，应该抛出异常
  // 断言 list->at<M>(modules.size() + 1) 会抛出带有 "Index out of range" 消息的异常
  ASSERT_THROWS_WITH(list->at<M>(modules.size() + 1), "Index out of range");
  // 断言 list->at<M>(modules.size() + 1000000) 会抛出带有 "Index out of range" 消息的异常
  ASSERT_THROWS_WITH(
      list->at<M>(modules.size() + 1000000), "Index out of range");
}

// 在 ModuleListTest 中定义一个名为 AccessWithPtr 的测试用例
TEST_F(ModuleListTest, AccessWithPtr) {
  // 定义一个结构 M，继承自 torch::nn::Module，包含一个整型成员变量 value
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  // 创建包含三个 std::shared_ptr<M> 的 vector modules
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  // 创建一个 ModuleList 对象 list
  ModuleList list;

  // 将 modules 中的每个 module 添加到 list 中
  for (auto& module : modules) {
    list->push_back(module);
  }

  // 断言 list 的大小为 3
  ASSERT_EQ(list->size(), 3);

  // 返回给定索引的正确 module
  for (const auto i : c10::irange(modules.size())) {
    ASSERT_EQ(list->ptr(i).get(), modules[i].get());
    ASSERT_EQ(list[i].get(), modules[i].get());
    ASSERT_EQ(list->ptr<M>(i).get(), modules[i].get());
  }

  // 对于超出索引范围的索引，应抛出异常
  ASSERT_THROWS_WITH(list->ptr(modules.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(list->ptr(modules.size() + 1000000), "Index out of range");
}

// 在 ModuleListTest 中定义一个名为 SanityCheckForHoldingStandardModules 的测试用例
TEST_F(ModuleListTest, SanityCheckForHoldingStandardModules) {
  // 创建一个包含多个标准模块的 ModuleList 对象 list
  ModuleList list(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
}

// 在 ModuleListTest 中定义一个名为 ExtendPushesModulesFromOtherModuleList 的测试用例
TEST_F(ModuleListTest, ExtendPushesModulesFromOtherModuleList) {
  // 定义结构 A, B, C, D 继承自 torch::nn::Module
  struct A : torch::nn::Module {};
  struct B : torch::nn::Module {};
  struct C : torch::nn::Module {};
  struct D : torch::nn::Module {};

  // 创建包含 A, B 的 ModuleList 对象 a
  ModuleList a(A{}, B{});

  // 创建包含 C, D 的 ModuleList 对象 b
  ModuleList b(C{}, D{});

  // 将 b 中的模块扩展到 a 中
  a->extend(*b);

  // 断言 a 的大小为 4
  ASSERT_EQ(a->size(), 4);
  ASSERT_TRUE(a[0]->as<A>());
  ASSERT_TRUE(a[1]->as<B>());
  ASSERT_TRUE(a[2]->as<C>());
  ASSERT_TRUE(a[3]->as<D>());

  // 断言 b 的大小为 2
  ASSERT_EQ(b->size(), 2);
  ASSERT_TRUE(b[0]->as<C>());
  ASSERT_TRUE(b[1]->as<D>());

  // 创建包含多个 std::shared_ptr<A> 的 vector c
  std::vector<std::shared_ptr<A>> c = {
      std::make_shared<A>(), std::make_shared<A>()};

  // 将 c 中的模块扩展到 b 中
  b->extend(c);

  // 断言 b 的大小为 4
  ASSERT_EQ(b->size(), 4);
  ASSERT_TRUE(b[0]->as<C>());
  ASSERT_TRUE(b[1]->as<D>());
  ASSERT_TRUE(b[2]->as<A>());
  ASSERT_TRUE(b[3]->as<A>());
}

// 在 ModuleListTest 中定义一个名为 HasReferenceSemantics 的测试用例
TEST_F(ModuleListTest, HasReferenceSemantics) {
  // 创建包含三个 Linear 模块的 ModuleList 对象 first
  ModuleList first(Linear(2, 3), Linear(4, 4), Linear(4, 5));

  // 创建 second 作为 first 的拷贝
  ModuleList second(first);

  // 断言 first 和 second 具有相同的指针
  ASSERT_EQ(first.get(), second.get());
  ASSERT_EQ(first->size(), second->size());

  // 断言 first 和 second 的每个模块是相等的
  ASSERT_TRUE(std::equal(
      first->begin(),
      first->end(),
      second->begin(),
      [](const std::shared_ptr<Module>& first,
         const std::shared_ptr<Module>& second) {
        return first.get() == second.get();
      }));
}

// 在 ModuleListTest 中定义一个名为 IsCloneable 的测试用例
TEST_F(ModuleListTest, IsCloneable) {
  // 创建包含三个模块的 ModuleList 对象 list
  ModuleList list(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));

  // 克隆 list 到一个新的 ModuleList 对象 clone
  ModuleList clone = std::dynamic_pointer_cast<ModuleListImpl>(list->clone());

  // 断言 list 和 clone 的大小相等
  ASSERT_EQ(list->size(), clone->size());

  // 断言 list 和 clone 中每个模块的类型相同，但对象不同
  for (size_t i = 0; i < list->size(); ++i) {
    ASSERT_EQ(list[i]->name(), clone[i]->name());
  }
}
    // 确保 list[i] 和 clone[i] 不相等
    ASSERT_NE(list[i], clone[i]);
    }
    
    // 使用 torch::NoGradGuard 禁用梯度计算，进入无梯度计算环境
    
    torch::NoGradGuard no_grad;
    
    // 获取 list 和 clone 的命名参数列表
    auto params1 = list->named_parameters();
    auto params2 = clone->named_parameters();
    
    // 断言两个模型具有相同数量的参数
    ASSERT_EQ(params1.size(), params2.size());
    
    // 遍历第一个模型的命名参数列表
    for (auto& param : params1) {
        // 断言每个参数在两个模型中的指针不相等
        ASSERT_FALSE(pointer_equal(param.value(), params2[param.key()]));
        // 断言每个参数在两个模型中的设备相同
        ASSERT_EQ(param->device(), params2[param.key()].device());
        // 断言每个参数在两个模型中的值近似相等
        ASSERT_TRUE(param->allclose(params2[param.key()]));
        // 修改第一个模型的参数值，添加常数 2
        param->add_(2);
    }
    
    // 再次遍历第一个模型的命名参数列表
    for (auto& param : params1) {
        // 断言修改后的参数在两个模型中不再近似相等
        ASSERT_FALSE(param->allclose(params2[param.key()]));
    }
TEST_F(ModuleListTest, RegistersElementsAsSubmodules) {
  // 创建 ModuleList 对象，并初始化三个子模块：Linear、Conv2d 和 Dropout2d
  ModuleList list(Linear(10, 3), Conv2d(1, 2, 3), Dropout2d(0.5));

  // 获取 ModuleList 中的子模块列表
  auto modules = list->children();
  // 断言第一个子模块是 Linear 类型
  ASSERT_TRUE(modules[0]->as<Linear>());
  // 断言第二个子模块是 Conv2d 类型
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  // 断言第三个子模块是 Dropout2d 类型
  ASSERT_TRUE(modules[2]->as<Dropout2d>());
}

TEST_F(ModuleListTest, NestingIsPossible) {
  // 创建 ModuleList 对象，并嵌套了两层 ModuleList，每层包含两个 Dropout 模块和一个 Dropout2d 模块
  ModuleList list(
      (ModuleList(Dropout(), Dropout())),
      (ModuleList(Dropout(), Dropout()), Dropout()));
}

TEST_F(ModuleListTest, CloneToDevice_CUDA) {
  // 创建 ModuleList 对象，并初始化三个子模块：Linear、Functional(torch::relu) 和 BatchNorm1d
  ModuleList list(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));
  // 定义 CUDA 设备
  torch::Device device(torch::kCUDA, 0);
  // 克隆 ModuleList 到指定设备，并转换为 ModuleListImpl 类型
  ModuleList clone =
      std::dynamic_pointer_cast<ModuleListImpl>(list->clone(device));
  // 遍历克隆后的 ModuleList 中的参数，检查其设备是否为 CUDA 设备
  for (const auto& p : clone->parameters()) {
    ASSERT_EQ(p.device(), device);
  }
  // 遍历克隆后的 ModuleList 中的缓冲区，检查其设备是否为 CUDA 设备
  for (const auto& b : clone->buffers()) {
    ASSERT_EQ(b.device(), device);
  }
}

TEST_F(ModuleListTest, PrettyPrintModuleList) {
  // 创建 ModuleList 对象，并初始化多个子模块：Linear、Conv2d、Dropout、BatchNorm2d、Embedding 和 LSTM
  ModuleList list(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
  // 断言 ModuleList 对象的字符串表示符合预期格式
  ASSERT_EQ(
      c10::str(list),
      "torch::nn::ModuleList(\n"
      "  (0): torch::nn::Linear(in_features=10, out_features=3, bias=true)\n"
      "  (1): torch::nn::Conv2d(1, 2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (2): torch::nn::Dropout(p=0.5, inplace=false)\n"
      "  (3): torch::nn::BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=true, track_running_stats=true)\n"
      "  (4): torch::nn::Embedding(num_embeddings=4, embedding_dim=10)\n"
      "  (5): torch::nn::LSTM(input_size=4, hidden_size=5, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)\n"
      ")");
}

TEST_F(ModuleListTest, RangeBasedForLoop) {
  // 创建 ModuleList 对象，并初始化三个子模块：Linear、BatchNorm1d 和 Dropout
  torch::nn::ModuleList mlist(
      torch::nn::Linear(3, 4),
      torch::nn::BatchNorm1d(4),
      torch::nn::Dropout(0.5));

  // 创建字符串流对象
  std::stringstream buffer;
  // 使用范围遍历输出每个子模块的漂亮打印形式到字符串流
  for (const auto& module : *mlist) {
    module->pretty_print(buffer);
  }
}

TEST_F(ModuleListTest, InvalidAt) {
  // 创建包含一个 Linear 模块的 ModuleList 对象
  torch::nn::ModuleList m(torch::nn::Linear(1, 2));
  // 断言访问索引为 0 的模块并尝试将其转换为 Dropout2dImpl 类型时抛出异常
  ASSERT_THROWS_WITH(
      m->at<torch::nn::Dropout2dImpl>(0), "Unable to cast module");
}
```