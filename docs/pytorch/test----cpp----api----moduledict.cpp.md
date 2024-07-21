# `.\pytorch\test\cpp\api\moduledict.cpp`

```py
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

// 定义一个测试夹具，继承自 SeedingFixture，用于测试 ModuleDict 相关功能
struct ModuleDictTest : torch::test::SeedingFixture {};

// 测试用例：从列表构造 ModuleDict
TEST_F(ModuleDictTest, ConstructsFromList) {
  // 定义一个简单的 Module 结构体，包含一个整数成员变量
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  // 创建一个包含 Module 指针对的列表
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> list = {
      {"module_1", std::make_shared<M>(1)},
      {"module_2", std::make_shared<M>(2)},
      {"module_3", std::make_shared<M>(3)}};
  
  // 用列表初始化 ModuleDict 对象
  ModuleDict dict(list);
  // 断言字典大小为3
  ASSERT_EQ(dict->size(), 3);
}

// 测试用例：从 OrderedDict 构造 ModuleDict
TEST_F(ModuleDictTest, ConstructsFromordereddict) {
  // 定义一个简单的 Module 结构体，包含一个整数成员变量
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  // 创建一个有序字典 OrderedDict，包含 Module 指针对
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"module_1", std::make_shared<M>(1)},
      {"module_2", std::make_shared<M>(2)},
      {"module_3", std::make_shared<M>(3)},
  };

  // 用 OrderedDict 初始化 ModuleDict 对象
  ModuleDict dict(ordereddict);
  // 断言字典大小为3
  ASSERT_EQ(dict->size(), 3);
}

// 测试用例：测试更新、弹出、清空和包含功能
TEST_F(ModuleDictTest, UpdatePopClearContains) {
  // 定义一个简单的 Module 结构体，包含一个整数成员变量
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  // 初始化一个空的 ModuleDict 对象
  ModuleDict dict;
  // 断言字充当为空
  ASSERT_TRUE(dict->empty());

  // 通过列表更新 ModuleDict
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> list1 = {
      {"module_1", std::make_shared<M>(1)}};
  dict->update(list1);
  // 断言字典大小为1，且包含 "module_1"
  ASSERT_EQ(dict->size(), 1);
  ASSERT_TRUE(dict->contains("module_1"));

  // 通过 OrderedDict 更新 ModuleDict
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"module_2", std::make_shared<M>(2)}};
  dict->update(ordereddict);
  // 断言字典大小为2，且包含 "module_2"
  ASSERT_EQ(dict->size(), 2);
  ASSERT_TRUE(dict->contains("module_2"));

  // 通过另一个 ModuleDict 更新 ModuleDict
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> list2 = {
      {"module_3", std::make_shared<M>(3)}};
  ModuleDict updatedict(list2);
  dict->update(*updatedict);
  // 断言字典大小为3，且包含 "module_3"
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(dict->contains("module_3"));

  // 弹出一个键
  dict->pop("module_1");
  // 断言字典大小减少到2
  ASSERT_EQ(dict->size(), 2);

  // 弹出一个不存在的键，预期抛出异常
  ASSERT_THROWS_WITH(dict->pop("module_4"), " 'module_4' is not defined");

  // 清空字典
  dict->clear();
  // 断言字典大小为0
  ASSERT_EQ(dict->size(), 0);
}

// 测试用例：测试更新已存在的键
TEST_F(ModuleDictTest, UpdateExist) {
  // 定义一个简单的 Module 结构体，包含一个整数成员变量
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
    // 定义一个名为 Module 的结构体，包含一个整型成员 value
    struct Module {
        int value;
    };

    // 创建一个包含键值对的 vector list1，每个键值对包括模块名和指向 Module 的共享指针
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> list1 = {
        {"module_1", std::make_shared<M>(1)},  // 使用 make_shared 创建 Module(1)，并将其与 "module_1" 关联
        {"module_2", std::make_shared<M>(2)}   // 使用 make_shared 创建 Module(2)，并将其与 "module_2" 关联
    };

    // 使用 list1 初始化 ModuleDict 对象 dict
    ModuleDict dict(list1);

    // 断言 dict 中 "module_2" 对应的 Module 对象的 value 为 2
    ASSERT_EQ(dict->at<M>("module_2").value, 2);

    // 更新 dict，使用 list2 中的键值对
    // 更新 "module_2" 为 Module(0)，添加 "module_3" 为 Module(3)
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> list2 = {
        {"module_2", std::make_shared<M>(0)},  // 使用 make_shared 创建 Module(0)，更新 "module_2"
        {"module_3", std::make_shared<M>(3)}   // 使用 make_shared 创建 Module(3)，添加 "module_3"
    };
    dict->update(list2);

    // 断言 dict 的大小为 3
    ASSERT_EQ(dict->size(), 3);

    // 断言 dict 中 "module_2" 对应的 Module 对象的 value 为 0
    ASSERT_EQ(dict->at<M>("module_2").value, 0);

    // 使用 ordereddict 更新 dict
    // 更新 "module_3" 为 Module(0)，添加 "module_4" 为 Module(4)
    torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
        {"module_3", std::make_shared<M>(0)},  // 使用 make_shared 创建 Module(0)，更新 "module_3"
        {"module_4", std::make_shared<M>(4)}   // 使用 make_shared 创建 Module(4)，添加 "module_4"
    };
    dict->update(ordereddict);

    // 断言 dict 的大小为 4
    ASSERT_EQ(dict->size(), 4);

    // 断言 dict 中 "module_3" 对应的 Module 对象的 value 为 0
    ASSERT_EQ(dict->at<M>("module_3").value, 0);

    // 使用 dict2 更新 dict
    // 更新 "module_4" 和 "module_1"，各自为 Module(0)
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> list3 = {
        {"module_4", std::make_shared<M>(0)},  // 使用 make_shared 创建 Module(0)，更新 "module_4"
        {"module_1", std::make_shared<M>(0)}   // 使用 make_shared 创建 Module(0)，更新 "module_1"
    };
    ModuleDict dict2(list3);
    dict->update(*dict2);

    // 断言 dict 的大小为 4
    ASSERT_EQ(dict->size(), 4);

    // 断言 dict 中 "module_1" 和 "module_4" 对应的 Module 对象的 value 都为 0
    ASSERT_EQ(dict->at<M>("module_1").value, 0);
    ASSERT_EQ(dict->at<M>("module_4").value, 0);
}

TEST_F(ModuleDictTest, Keys) {
  // 定义一个内部结构 M，继承自 Module，带有一个整型成员变量 value
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  // 创建一个有序字典 ordereddict，键为字符串，值为指向 Module 的 shared_ptr
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"linear", Linear(10, 3).ptr()},   // 添加 Linear 模块的指针，传入参数 10 和 3
      {"conv", Conv2d(1, 2, 3).ptr()},   // 添加 Conv2d 模块的指针，传入参数 1, 2, 3
      {"dropout", Dropout(0.5).ptr()},   // 添加 Dropout 模块的指针，传入参数 0.5
  };

  // 创建 ModuleDict 对象 dict，使用上述有序字典进行初始化
  ModuleDict dict(ordereddict);

  // 获取 dict 中的键集合 keys
  const auto& keys = dict->keys();

  // 定义预期的键集合 expected
  std::vector<std::string> expected{"linear", "conv", "dropout"};

  // 断言 keys 应该与 expected 相等
  ASSERT_EQ(keys, expected);

  // 断言访问不存在的键 "batch" 时应抛出异常
  ASSERT_THROWS_WITH(dict["batch"], " 'batch' is not defined");

  // 对 "linear"、"conv" 和 "dropout" 键对应的模块进行类型检查
  ASSERT_TRUE(dict["linear"]->as<Linear>());
  ASSERT_TRUE(dict["conv"]->as<Conv2d>());
  ASSERT_TRUE(dict["dropout"]->as<Dropout>());
}

TEST_F(ModuleDictTest, Values) {
  // 定义一个内部结构 M，继承自 Module，带有一个整型成员变量 value
  struct M : Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  // 创建一个有序字典 ordereddict，键为字符串，值为指向 M 的 shared_ptr
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"module_1", std::make_shared<M>(1)},   // 添加 M 对象的 shared_ptr，value 为 1
      {"module_2", std::make_shared<M>(2)},   // 添加 M 对象的 shared_ptr，value 为 2
  };

  // 创建 ModuleDict 对象 dict，使用上述有序字典进行初始化
  ModuleDict dict(ordereddict);

  // 获取 dict 中的值集合 values
  const auto& values = dict->values();

  // 获取 ordereddict 中的值集合 expected
  const auto& expected = ordereddict.values();

  // 断言 values 应该与 expected 相等
  ASSERT_EQ(values, expected);

  // 使用自定义的比较器 lambda 函数，断言 dict 和 ordereddict 中的模块指针相等
  ASSERT_TRUE(std::equal(
      dict->begin(),
      dict->end(),
      ordereddict.begin(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.value().get() == rhs.value().get();
      }));
}

TEST_F(ModuleDictTest, SanityCheckForHoldingStandardModules) {
  // 创建一个有序字典 ordereddict，包含标准模块的名称和对应的指针
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"linear", Linear(10, 3).ptr()},          // Linear 模块，参数 10 和 3
      {"conv", Conv2d(1, 2, 3).ptr()},          // Conv2d 模块，参数 1, 2, 3
      {"dropout", Dropout(0.5).ptr()},          // Dropout 模块，参数 0.5
      {"batch", BatchNorm2d(5).ptr()},          // BatchNorm2d 模块，参数 5
      {"embedding", Embedding(4, 10).ptr()},    // Embedding 模块，参数 4, 10
      {"lstm", LSTM(4, 5).ptr()}                // LSTM 模块，参数 4, 5
  };

  // 创建 ModuleDict 对象 dict，使用上述有序字典进行初始化
  ModuleDict dict(ordereddict);
}

TEST_F(ModuleDictTest, HasReferenceSemantics) {
  // 创建一个有序字典 ordereddict，包含 Linear 模块的三个不同实例
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"linear1", Linear(2, 3).ptr()},   // Linear 模块，参数 2, 3
      {"linear2", Linear(3, 4).ptr()},   // Linear 模块，参数 3, 4
      {"linear3", Linear(4, 5).ptr()},   // Linear 模块，参数 4, 5
  };

  // 创建两个 ModuleDict 对象 first 和 second，使用相同的有序字典进行初始化
  ModuleDict first(ordereddict);
  ModuleDict second(ordereddict);

  // 断言 first 和 second 的大小应该相等
  ASSERT_EQ(first->size(), second->size());

  // 使用自定义的比较器 lambda 函数，断言 first 和 second 中的模块指针相等
  ASSERT_TRUE(std::equal(
      first->begin(),
      first->end(),
      second->begin(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.value().get() == rhs.value().get();
      }));
}

void iscloneable_helper(torch::Device device) {
  // 创建一个有序字典 ordereddict，包含 Linear、Functional 和 BatchNorm1d 模块的实例
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"linear", Linear(2, 3).ptr()},                    // Linear 模块，参数 2, 3
      {"relu", Functional(torch::relu).ptr()},          // Functional 模块，使用 torch::relu 函数
      {"batch", BatchNorm1d(3).ptr()},                  // BatchNorm1d 模块，参数 3
  };

  // 创建 ModuleDict 对象 dict，使用上述有序字典进行初始化
  ModuleDict dict(ordereddict);

  // 将 dict 转移到指定设备上
  dict->to(device);

  // 克隆 dict 对象到指定设备，存储到 clone 中
  ModuleDict clone =
      std::dynamic_pointer_cast<ModuleDictImpl>(dict->clone(device));

  // 断言 dict 和 clone 的大小应该相等
  ASSERT_EQ(dict->size(), clone->size());

  // 使用迭代器遍历 dict 和 clone，断言它们的键相同
  for (auto it = dict->begin(), it_c = clone->begin(); it != dict->end();
       ++it, ++it_c) {
    // The key should be same
    ASSERT_EQ(it->key(), it_c->key());
    // The modules should be the same kind (type).
    // 断言两个对象的名称相同
    ASSERT_EQ(it->value()->name(), it_c->value()->name());
    // 断言两个对象不是同一个指针，即它们是不同的对象
    ASSERT_NE(it->value(), it_c->value());
  }

  // 验证克隆是深度的，即模块的参数也被克隆了
  torch::NoGradGuard no_grad;

  // 获取原始字典和克隆字典的命名参数
  auto params1 = dict->named_parameters();
  auto params2 = clone->named_parameters();
  // 断言两个字典中命名参数的数量相同
  ASSERT_EQ(params1.size(), params2.size());
  // 遍历原始字典的命名参数
  for (auto& param : params1) {
    // 断言两个参数对象不是同一个指针
    ASSERT_FALSE(pointer_equal(param.value(), params2[param.key()]));
    // 断言两个参数对象的设备相同
    ASSERT_EQ(param->device(), params2[param.key()].device());
    // 断言两个参数对象的数值在相对误差范围内相等
    ASSERT_TRUE(param->allclose(params2[param.key()]));
    // 修改原始字典中的参数对象
    param->add_(2);
  }
  // 再次遍历原始字典的命名参数
  for (auto& param : params1) {
    // 断言修改后的参数对象与克隆字典中对应的参数对象不再相等
    ASSERT_FALSE(param->allclose(params2[param.key()]));
  }
}

TEST_F(ModuleDictTest, IsCloneable) {
  iscloneable_helper(torch::kCPU);
}

TEST_F(ModuleDictTest, IsCloneable_CUDA) {
  iscloneable_helper({torch::kCUDA, 0});
}

TEST_F(ModuleDictTest, RegistersElementsAsSubmodules) {
  // 创建一个有序字典，包含三个子模块：Linear、Conv2d 和 Dropout
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict1 = {
      {"linear", Linear(10, 3).ptr()},
      {"conv", Conv2d(1, 2, 3).ptr()},
      {"test", Dropout(0.5).ptr()},
  };
  // 使用有序字典初始化 ModuleDict 对象
  ModuleDict dict(ordereddict1);

  // 获取 ModuleDict 中的子模块列表
  auto modules = dict->children();
  // 断言第一个子模块是 Linear 类型
  ASSERT_TRUE(modules[0]->as<Linear>());
  // 断言第二个子模块是 Conv2d 类型
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  // 断言第三个子模块是 Dropout 类型
  ASSERT_TRUE(modules[2]->as<Dropout>());

  // 更新已存在的子模块
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict2 = {
      {"lstm", LSTM(4, 5).ptr()}, {"test", BatchNorm2d(5).ptr()}};
  dict->update(ordereddict2);

  // 再次获取更新后的子模块列表
  modules = dict->children();
  // 断言第一个子模块仍然是 Linear 类型
  ASSERT_TRUE(modules[0]->as<Linear>());
  // 断言第二个子模块仍然是 Conv2d 类型
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  // 断言第三个子模块是 BatchNorm2d 类型，保持原有顺序
  ASSERT_TRUE(modules[2]->as<BatchNorm2d>());
  // 断言第四个子模块是 LSTM 类型
  ASSERT_TRUE(modules[3]->as<LSTM>());
}

TEST_F(ModuleDictTest, CloneToDevice_CUDA) {
  // 创建一个有序字典，包含三个子模块：Linear、Functional 和 BatchNorm1d
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"linear", Linear(2, 3).ptr()},
      {"relu", Functional(torch::relu).ptr()},
      {"batch", BatchNorm1d(3).ptr()},
  };
  // 使用有序字典初始化 ModuleDict 对象
  ModuleDict dict(ordereddict);
  // 指定设备为 CUDA 设备 0
  torch::Device device(torch::kCUDA, 0);
  // 克隆 ModuleDict 到指定设备
  ModuleDict clone =
      std::dynamic_pointer_cast<ModuleDictImpl>(dict->clone(device));
  // 遍历克隆后的参数，断言其设备为指定设备
  for (const auto& p : clone->parameters()) {
    ASSERT_EQ(p.device(), device);
  }
  // 遍历克隆后的缓冲区，断言其设备为指定设备
  for (const auto& b : clone->buffers()) {
    ASSERT_EQ(b.device(), device);
  }
}

TEST_F(ModuleDictTest, PrettyPrintModuleDict) {
  // 创建一个有序字典，包含六个子模块：Linear、Conv2d、Dropout、BatchNorm2d、Embedding 和 LSTM
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"linear", Linear(10, 3).ptr()},
      {"conv", Conv2d(1, 2, 3).ptr()},
      {"dropout", Dropout(0.5).ptr()},
      {"batch", BatchNorm2d(5).ptr()},
      {"embedding", Embedding(4, 10).ptr()},
      {"lstm", LSTM(4, 5).ptr()}};
  // 使用有序字典初始化 ModuleDict 对象
  ModuleDict dict(ordereddict);

  // 断言 ModuleDict 对象的字符串表示与预期相符
  ASSERT_EQ(
      c10::str(dict),
      "torch::nn::ModuleDict(\n"
      "  (linear): torch::nn::Linear(in_features=10, out_features=3, bias=true)\n"
      "  (conv): torch::nn::Conv2d(1, 2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (dropout): torch::nn::Dropout(p=0.5, inplace=false)\n"
      "  (batch): torch::nn::BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=true, track_running_stats=true)\n"
      "  (embedding): torch::nn::Embedding(num_embeddings=4, embedding_dim=10)\n"
      "  (lstm): torch::nn::LSTM(input_size=4, hidden_size=5, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)\n"
      ")");
}

TEST_F(ModuleDictTest, InvalidAt) {
  // 创建一个有序字典，包含一个子模块：Linear
  torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
      {"linear", Linear(10, 3).ptr()}};
  // 使用有序字典初始化 ModuleDict 对象
  ModuleDict dict(ordereddict);
  // 断言在使用非相关类型访问子模块时抛出异常
  ASSERT_THROWS_WITH(
      dict->at<torch::nn::Dropout2dImpl>("linear"), "Unable to cast module");
}
```