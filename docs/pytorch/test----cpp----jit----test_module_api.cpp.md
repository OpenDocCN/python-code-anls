# `.\pytorch\test\cpp\jit\test_module_api.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <test/cpp/jit/test_utils.h>  // 包含用于 JIT 测试的实用函数的头文件

#include <ATen/core/qualified_name.h>  // 包含有关限定名称的头文件
#include <torch/csrc/jit/api/module.h>  // 包含 PyTorch JIT 模块的头文件
#include <torch/csrc/jit/frontend/resolver.h>  // 包含用于前端解析的头文件
#include <torch/csrc/jit/serialization/import.h>  // 包含用于导入模型的头文件
#include <torch/csrc/jit/serialization/import_source.h>  // 包含导入模型源的头文件
#include <torch/csrc/jit/testing/file_check.h>  // 包含用于文件检查的头文件
#include <torch/torch.h>  // 包含 PyTorch 核心库的头文件

namespace torch {
namespace jit {

static constexpr c10::string_view moduleInterfaceSrc = R"JIT(
class OneInterface(ModuleInterface):
    def one(self, x: Tensor, y: Tensor) -> Tensor:
        pass
)JIT";  // 定义一个模块接口的源代码字符串

static const std::vector<std::string> subModuleMethodsSrc = {R"JIT(
def one(self, x: Tensor, y: Tensor) -> Tensor:
    return self.attr * x + y + 1

def forward(self, x: Tensor) -> Tensor:
    return self.attr + x
)JIT"};  // 定义子模块方法的源代码字符串向量

static const std::string parentForward = R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return self.subMod1.one(x, x) + self.subMod2.one(x, x)
)JIT";  // 定义父模块的前向传播函数的源代码字符串

static void import_libs(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& class_name,
    const std::shared_ptr<Source>& src,
    const std::vector<at::IValue>& tensor_table) {
  SourceImporter si(
      cu,
      &tensor_table,
      [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
      /*version=*/2);  // 创建源代码导入器对象，用于导入模块

  si.loadType(QualifiedName(class_name));  // 加载指定限定名称的类型
}

TEST(ModuleAPITest, MethodRunAsync) {
  // Module m("m");
  // m.define(R"(
  //   def forward(self):
  //     r1 = torch.jit.fork(torch.mm, torch.rand(100,100),torch.rand(100,100))
  //     r2 = torch.jit.fork(torch.mm, torch.rand(100,100),torch.rand(100,100))
  //     return r1.wait() + r2.wait()
  // )");
  std::string filePath(__FILE__);  // 获取当前文件的路径
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);  // 提取文件路径中的目录部分
  // borrow model file from TEST(GraphExecutorTest, runAsync_executor)
  testModelFile.append("test_interpreter_async.pt");  // 拼接测试模型文件的路径

  auto m = load(testModelFile);  // 加载指定路径的模型文件

  auto counter = 0;  // 计数器初始化为0，用于计数任务启动次数
  std::mutex mtx;  // 创建互斥量，用于保护计数器的并发访问

  auto launcher = [&](std::function<void()> f) {  // 定义任务启动器
    mtx.lock();  // 加锁
    ++counter;  // 增加计数器
    mtx.unlock();  // 解锁
    at::launch(std::move(f));  // 启动任务
  };

  auto method = m.get_method("forward");  // 获取模型的前向传播方法

  std::vector<IValue> stack;  // 创建空的堆栈，用于传递参数
  auto kwargs = std::unordered_map<std::string, at::IValue>();  // 创建空的关键字参数映射
  auto future = method.run_async(stack, kwargs, launcher);  // 异步运行模型的前向传播方法

  future->wait();  // 等待异步任务完成

  // expect 2 forks and 2 wait callbacks being executed on provided taskLauncher
  // but ivalue::Future would be marked completed and release wait before
  // finishing all callbacks
  ASSERT_GE(counter, 2);  // 断言计数器值至少为2，验证任务启动次数
}
TEST(ModuleAPITest, Clone) {
  // 创建一个 CompilationUnit 的 shared_ptr 对象 cu
  auto cu = std::make_shared<CompilationUnit>();
  
  // 创建一个名为 "child" 的 ClassType，并将其封装在 child 变量中，共享 CompilationUnit cu
  auto child = ClassType::create("child", cu, true);
  
  // 设置属性名 attr_name 为 "attr"，并将 IntType::get() 作为属性类型添加到 child
  auto attr_name = "attr";
  child->addAttribute(attr_name, IntType::get());
  
  // 使用 cu 和 child 创建 Module 对象 c1，并将 IValue(2) 注册为名为 attr_name 的属性值
  Module c1(cu, child);
  auto v1 = IValue(2);
  c1.register_attribute(attr_name, IntType::get(), v1, false);
  
  // 使用 cu 和 child 创建另一个 Module 对象 c2，并将 IValue(3) 注册为名为 attr_name 的属性值
  Module c2(cu, child);
  auto v2 = IValue(3);
  c2.register_attribute(attr_name, IntType::get(), v2, false);
  
  // 创建一个名为 "parent" 的 ClassType，并将其封装在 parent 变量中，共享 CompilationUnit cu
  auto parent = ClassType::create("parent", cu, true);
  
  // 使用 cu 和 parent 创建 Module 对象 p，并将 c1 和 c2 注册为其属性 "c1" 和 "c2"
  Module p(cu, parent);
  p.register_attribute("c1", c1.type(), c1._ivalue(), false);
  p.register_attribute("c2", c2.type(), c2._ivalue(), false);
  
  // 克隆 Module 对象 p，创建新的 Module 对象 p2
  Module p2 = p.clone();
  
  // 断言：检查 p2 的 "c1" 和 "c2" 属性具有相同的类型
  ASSERT_EQ(p2.attr("c1").type(), p2.attr("c2").type());
  
  // 断言：检查 p2 的 "c1" 属性的 attr_name 属性值为 2
  ASSERT_EQ(Module(p2.attr("c1").toObject()).attr(attr_name).toInt(), 2);
  
  // 断言：检查 p2 的 "c2" 属性的 attr_name 属性值为 3
  ASSERT_EQ(Module(p2.attr("c2").toObject()).attr(attr_name).toInt(), 3);
}

TEST(ModuleAPITest, CloneWithModuleInterface) {
  // 创建一个 CompilationUnit 的 shared_ptr 对象 cu
  auto cu = std::make_shared<CompilationUnit>();
  
  // 创建一个名为 "parentMod" 的 Module 对象 parentMod
  Module parentMod("parentMod", cu);
  
  // 创建两个子 Module 对象 subMod1 和 subMod2，共享 CompilationUnit cu
  Module subMod1("subMod1", cu);
  Module subMod2("subMod2", cu);
  
  // 导入模块接口 "__torch__.OneInterface" 并设置常量表 constantTable
  std::vector<at::IValue> constantTable;
  import_libs(
      cu,
      "__torch__.OneInterface",
      std::make_shared<Source>(moduleInterfaceSrc),
      constantTable);
  
  // 将值为 2 的 IValue 注册为 subMod1 的 "attr" 属性
  auto v1 = IValue(2);
  subMod1.register_attribute("attr", IntType::get(), v1, false);
  
  // 将值为 4 的 IValue 注册为 subMod2 的 "attr" 属性
  auto v2 = IValue(4);
  subMod2.register_attribute("attr", IntType::get(), v2, false);
  
  // 为两个子模块定义 subModuleMethodsSrc 中指定的方法
  for (const std::string& method : subModuleMethodsSrc) {
    subMod1.define(method, nativeResolver());
    subMod2.define(method, nativeResolver());
  }
  
  // 将 subMod1 和 subMod2 分别作为 "__torch__.OneInterface" 接口的实现注册到 parentMod
  parentMod.register_attribute(
      "subMod1",
      cu->get_interface("__torch__.OneInterface"),
      subMod1._ivalue());
  parentMod.register_attribute(
      "subMod2",
      cu->get_interface("__torch__.OneInterface"),
      subMod2._ivalue());
  
  // 定义 parentForward 方法并将其与 parentMod 关联
  parentMod.define(parentForward, nativeResolver());
  
  // 克隆 Module 对象 parentMod，创建新的 Module 对象 clonedMod
  Module clonedMod = parentMod.clone();
  
  // 断言：检查克隆后的 clonedMod 和原始 parentMod 的类型不相同
  ASSERT_NE(clonedMod.type(), parentMod.type());
}
TEST(ModuleAPITest, Copy) {
  // 创建一个共享指针指向 CompilationUnit 的实例
  auto cu = std::make_shared<CompilationUnit>();
  // 创建一个名为 "foo.bar" 的类，并与 CompilationUnit 相关联
  auto cls = ClassType::create("foo.bar", cu, true);
  // 定义一个属性名为 attr_name，并添加到 cls 类型中，类型为 IntType
  auto attr_name = "attr";
  cls->addAttribute(attr_name, IntType::get());
  // 使用 cu 和 cls 创建一个 Module 实例 m
  Module m(cu, cls);
  // 创建一个整数值为 2 的 IValue 实例 v
  auto v = IValue(2);
  // 将属性 attr_name 注册到 m 中，类型为 IntType，值为 v，非静态属性
  m.register_attribute(attr_name, IntType::get(), v, false);

  // 克隆 m 到 m2
  Module m2 = m.clone();
  // 拷贝 m 到 m3
  Module m3 = m.copy();

  // 确保克隆操作有效
  ASSERT_EQ(m2.attr(attr_name).toInt(), 2);
  ASSERT_EQ(m3.attr(attr_name).toInt(), 2);

  // clone 方法会复制类型和数据，因此类型不同
  ASSERT_NE(m.type(), m2.type());
  // copy 方法只复制数据，类型是共享的
  ASSERT_EQ(m.type(), m3.type());

  // 修改复制实例的属性值
  m3.register_attribute(attr_name, IntType::get(), IValue(3), false);
  // 验证原始实例的值未发生改变
  ASSERT_EQ(m2.attr(attr_name).toInt(), 2);
  ASSERT_EQ(m3.attr(attr_name).toInt(), 3);
}

TEST(ModuleAPITest, DeepCopy) {
  // 创建一个共享指针指向 CompilationUnit 的实例
  auto cu = std::make_shared<CompilationUnit>();
  // 创建一个名为 "foo.bar" 的类，并与 CompilationUnit 相关联
  auto cls = ClassType::create("foo.bar", cu, true);
  // 定义字符串属性名和整数属性名等
  auto str_attr = "str_attr";
  auto int_attr = "int_attr";
  auto tensor_attr = "tensor_attr";
  auto tensor_list_attr = "tensor_list_attr";
  // 向 cls 类型中添加不同类型的属性
  cls->addAttribute(int_attr, IntType::get());
  cls->addAttribute(str_attr, StringType::get());
  cls->addAttribute(tensor_attr, TensorType::get());
  cls->addAttribute(tensor_list_attr, ListType::ofTensors());
  // 使用 cu 和 cls 创建一个 Module 实例 m
  Module m(cu, cls);
  // 创建一个包含两个随机张量的列表
  c10::List<at::Tensor> list({at::rand(5), at::rand(5)});
  // 设置不同类型的属性值到 m 中
  m.setattr(int_attr, IValue(2));
  m.setattr(str_attr, IValue("str"));
  m.setattr(tensor_attr, at::randn(5));
  m.setattr(tensor_list_attr, list);

  // 对 m 进行深拷贝得到 m2
  Module m2 = m.deepcopy();
  // 对 m 进行浅拷贝得到 m3
  Module m3 = m.copy();
  
  // 确保拷贝操作有效
  ASSERT_EQ(m2.attr(int_attr).toInt(), 2);
  ASSERT_EQ(m3.attr(int_attr).toInt(), 2);

  // 测试重叠情况
  ASSERT_TRUE(!IValue(m2._ivalue()).overlaps(IValue(m._ivalue())));
  ASSERT_TRUE(IValue(m3._ivalue()).overlaps(IValue(m._ivalue())));

  // deepcopy 和 copy 方法都会保留类型
  ASSERT_EQ(m.type(), m2.type());
  ASSERT_EQ(m.type(), m3.type());

  // 修改复制实例的整数属性值
  m2.setattr(int_attr, IValue(3));
  m3.setattr(int_attr, IValue(4));

  // 验证原始实例的整数属性值未发生改变
  ASSERT_EQ(m.attr(int_attr).toInt(), 2);
  ASSERT_EQ(m2.attr(int_attr).toInt(), 3);
  ASSERT_EQ(m3.attr(int_attr).toInt(), 4);

  // 修改复制实例的张量属性值
  at::Tensor t1 = m.attr(tensor_attr).toTensor();
  at::Tensor t2 = m2.attr(tensor_attr).toTensor(); // deepcopy 会复制张量
  at::Tensor t3 = m3.attr(tensor_attr).toTensor(); // copy 不会复制张量

  // 验证拷贝操作有效
  ASSERT_TRUE(t1.equal(t2));
  ASSERT_TRUE(t1.equal(t3));

  // 将 t1 置零
  t1.zero_();
  // 验证 t2 未受影响，因为它是深拷贝
  ASSERT_TRUE(!t1.equal(t2));
  // 验证 t3 与 t1 相同，因为它是浅拷贝
  ASSERT_TRUE(t1.equal(t3));
}
// 定义测试函数，测试在模块 API 中深拷贝字符串属性的功能
TEST(ModuleAPITest, DeepCopyString) {
  // 创建共享的编译单元
  auto cu = std::make_shared<CompilationUnit>();
  // 创建一个名为 foo.bar 的类，并将其添加到编译单元中
  auto cls = ClassType::create("foo.bar", cu, true);
  // 定义一个名为 attr1 的字符串属性
  auto attr1 = "attr1";
  // 将 attr1 添加为 cls 类的属性，类型为 StringType
  cls->addAttribute(attr1, StringType::get());
  // 定义一个名为 str 的字符串变量，并赋值为 "str"
  std::string str = "str";
  // 创建一个 Module 对象 m，使用之前创建的编译单元 cu 和类 cls
  Module m(cu, cls);
  // 将 str 的值设置为 m 对象的属性 attr1 的值
  m.setattr(attr1, str);
  // 对 m 进行深拷贝，赋值给 copied
  auto copied = m.deepcopy();
  // 记录原始的 str 的值
  auto original_str = str;
  // 断言深拷贝后的 copied 对象的属性 attr1 的字符串表示与原始 str 相同
  ASSERT_EQ(copied.attr(attr1).toStringRef(), original_str);
  // 检查修改原始 str 的字符串内容不会反映在深拷贝的 copied 模块中
  str += "str";
  ASSERT_EQ(copied.attr(attr1).toStringRef(), original_str);
}

// 定义测试函数，测试在模块 API 中深拷贝枚举类型属性的功能
TEST(ModuleAPITest, DeepCopyEnum) {
  // 创建共享的编译单元
  auto cu = std::make_shared<CompilationUnit>();
  // 创建一个名为 foo.bar 的类，并将其添加到编译单元中
  auto cls = ClassType::create("foo.bar", cu, true);
  // 定义一个名为 enum_attr 的枚举属性
  auto enum_attr = "enum_attr";
  // 创建一个名为 int_enum_type 的整数类型枚举
  auto int_enum_type = EnumType::create(
      "enum_class",
      IntType::get(),
      {{"enum_name_1", 1}, {"enum_name_2", 2}},
      cu);
  // 将 enum_attr 添加为 cls 类的属性，类型为 int_enum_type
  cls->addAttribute(enum_attr, int_enum_type);
  // 创建一个 Module 对象 m，使用之前创建的编译单元 cu 和类 cls
  Module m(cu, cls);
  // 将 enum_attr 的值设置为 m 对象的属性，值为枚举类型 "enum_name_1"
  m.setattr(
      enum_attr,
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type, "enum_name_1", 1)));
  // 对 m 进行深拷贝，创建一个新的 Module 对象 m2
  Module m2 = m.deepcopy();

  // 确保深拷贝操作正确完成
  // 获取 m2 对象的 enum_attr 属性的 EnumHolder 指针
  c10::ivalue::EnumHolder* m2_holder = m2.attr(enum_attr).toEnumHolder().get();
  // 断言 m2_holder 的值为 1
  ASSERT_EQ(m2_holder->value().toInt(), 1);
  // 断言 m2_holder 的名称为 "enum_name_1"
  ASSERT_EQ(m2_holder->name(), "enum_name_1");
  // 断言 m2_holder 的类型与 int_enum_type 相同
  ASSERT_EQ(m2_holder->type(), int_enum_type);

  // 测试重叠情况，确保深拷贝不会影响原对象的值
  ASSERT_TRUE(!IValue(m2._ivalue()).overlaps(IValue(m._ivalue())));

  // 深拷贝会保留类型信息
  ASSERT_EQ(m.type(), m2.type());

  // 修改原始对象，验证不会影响深拷贝对象
  m.setattr(
      enum_attr,
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type, "enum_name_2", 2)));
  ASSERT_NE(
      m.attr(enum_attr).toEnumHolder().get()->value().toInt(),
      m2.attr(enum_attr).toEnumHolder().get()->value().toInt());
}
TEST(ModuleAPITest, DeepCopyPreservesAliasing) {
  // check deepcopy preserves aliasing
  auto cu = std::make_shared<CompilationUnit>();  // 创建共享的编译单元对象 cu
  auto cls = ClassType::create("foo.bar", cu, true);  // 在 cu 中创建名为 "foo.bar" 的类对象 cls
  auto attr1 = "attr1";  // 定义字符串常量 attr1
  auto attr2 = "attr2";  // 定义字符串常量 attr2
  auto attr3 = "attr3";  // 定义字符串常量 attr3
  auto attr4 = "attr4";  // 定义字符串常量 attr4
  cls->addAttribute(attr1, ListType::ofTensors());  // 在 cls 中添加名为 attr1 的张量列表属性
  cls->addAttribute(attr2, ListType::ofTensors());  // 在 cls 中添加名为 attr2 的张量列表属性
  cls->addAttribute(attr3, TensorType::get());  // 在 cls 中添加名为 attr3 的张量类型属性
  cls->addAttribute(attr4, TensorType::get());  // 在 cls 中添加名为 attr4 的张量类型属性
  Module m(cu, cls);  // 创建 Module 对象 m，关联 cu 和 cls

  auto t1 = at::rand(5);  // 创建一个形状为 [5] 的随机张量 t1
  auto t2 = at::rand(5);  // 创建一个形状为 [5] 的随机张量 t2
  auto t3 = at::rand(5);  // 创建一个形状为 [5] 的随机张量 t3
  auto t4 = at::rand({5, 2});  // 创建一个形状为 [5, 2] 的随机张量 t4
  c10::List<at::Tensor> list1({t1, t2});  // 创建包含 t1 和 t2 的张量列表 list1
  c10::List<at::Tensor> list2({t1, t3});  // 创建包含 t1 和 t3 的张量列表 list2

  // first element of attr1 and attr2 are aliased
  m.setattr(attr1, list1);  // 将 list1 设置为 m 的 attr1 属性
  m.setattr(attr2, list2);  // 将 list2 设置为 m 的 attr2 属性
  m.setattr(attr3, t4);  // 将 t4 设置为 m 的 attr3 属性
  m.setattr(attr4, t4.view(-1));  // 将 t4 的视图（按行展开）设置为 m 的 attr4 属性

  auto copied = m.deepcopy();  // 深拷贝 m 得到 copied 对象

  // test tensor aliasing
  auto copied_attr1_t1 = copied.attr(attr1).toList().get(0);  // 获取 copied 中 attr1 属性的列表，并取第一个张量
  auto copied_attr2_t1 = copied.attr(attr2).toList().get(0);  // 获取 copied 中 attr2 属性的列表，并取第一个张量
  ASSERT_TRUE(copied_attr1_t1.isAliasOf(copied_attr2_t1));  // 断言 copied_attr1_t1 和 copied_attr2_t1 是别名

  // test aliasing from view
  auto copied_attr3 = copied.attr(attr3);  // 获取 copied 中 attr3 属性
  auto copied_attr4 = copied.attr(attr3);  // 获取 copied 中 attr4 属性（实际是 attr3）
  ASSERT_TRUE(copied_attr3.isAliasOf(copied_attr4));  // 断言 copied_attr3 和 copied_attr4 是别名
}

TEST(ModuleAPITest, Constants) {
  auto cu = std::make_shared<CompilationUnit>();  // 创建共享的编译单元对象 cu
  auto cls = ClassType::create("foo.bar", cu, true);  // 在 cu 中创建名为 "foo.bar" 的类对象 cls
  auto attr_name = "attr";  // 定义字符串常量 attr_name
  auto const_name = "const";  // 定义字符串常量 const_name
  cls->addAttribute(attr_name, IntType::get());  // 在 cls 中添加名为 attr 的整数类型属性
  cls->addConstant(const_name, IValue(3));  // 在 cls 中添加名为 const 的常量属性，值为整数 3
  Module m(cu, cls);  // 创建 Module 对象 m，关联 cu 和 cls

  auto v = IValue(2);  // 创建整数值为 2 的 IValue 对象
  m.register_attribute(attr_name, IntType::get(), v, false);  // 在 m 中注册名为 attr_name 的整数属性，初始值为 v，非参数属性
  ASSERT_TRUE(m.hasattr(attr_name));  // 断言 m 中存在 attr_name 属性
  ASSERT_TRUE(m.hasattr(const_name));  // 断言 m 中存在 const_name 属性
  ASSERT_EQ(m.attr(attr_name).toInt(), 2);  // 断言 m 的 attr_name 属性值为 2
  ASSERT_EQ(m.attr(const_name).toInt(), 3);  // 断言 m 的 const_name 属性值为 3
}

TEST(ModuleAPITest, Parameters) {
  auto cu = std::make_shared<CompilationUnit>();  // 创建共享的编译单元对象 cu
  auto cls = ClassType::create("foo.bar", cu, true);  // 在 cu 中创建名为 "foo.bar" 的类对象 cls
  Module m(cu, cls);  // 创建 Module 对象 m，关联 cu 和 cls

  // Tensor parameter
  m.register_parameter(
      "tensor_param", at::empty({3}, at::kFloat), /* is_buffer */ false);  // 在 m 中注册名为 tensor_param 的浮点数张量参数，形状为 [3]

  // None parameter
  m.register_attribute(
      "none_param", NoneType::get(), IValue(), /* is_param */ true);  // 在 m 中注册名为 none_param 的空值参数属性
  m.register_attribute(
      "none_param2", NoneType::get(), IValue(), /* is_param */ true);  // 在 m 中注册名为 none_param2 的空值参数属性

  auto param_list = m.parameters();  // 获取 m 的所有参数列表
  ASSERT_EQ(param_list.size(), 1);  // 断言参数列表长度为 1
  ASSERT_TRUE(m.hasattr("tensor_param"));  // 断言 m 中存在 tensor_param 属性
  ASSERT_TRUE(m.hasattr("none_param"));  // 断言 m 中存在 none_param 属性
  ASSERT_TRUE(m.hasattr("none_param2"));  // 断言 m 中存在 none_param2 属性
}

TEST(ModuleAPITest, Define) {
  Module m("m");  // 创建名称为 "m" 的 Module 对象 m
  m.register_parameter("foo", torch::ones({}), false);  // 在 m 中注册名为 "foo" 的张量参数，形状为 [1]，非缓冲区参数
  m.define(R"(
    def add_it(self, x, b : int = 4):
      return self.foo + x + b
  )");  // 定义在 m 中添加一个名为 add_it 的方法，接受参数 self, x, b（b 的默认值为 4），返回 self.foo + x + b 的结果

  auto result = m.run_method("add_it", torch::ones({}));  // 在 m 上运行 add_it 方法，传入一个形状为 [1] 的张量作为参数
  AT_ASSERT(result.toTensor().item<float>() == 6);  // 断言 result 转换为张量后的浮点数值为 6
}

TEST(ModuleAPITest, Freezing) {
  Module m("m");  // 创建名称为 "m" 的 Module 对象 m
  m.register_parameter("foo", torch::ones({}), false);  // 在 m 中注册名为 "foo" 的张量参数，形状为 [1]，非缓冲区参数
  m.define(R"(
    // 定义一个成员函数 forward，接受参数 x 和可选参数 b，默认值为 4
    def forward(self, x, b : int = 4):
      // 返回 foo 属性值加上 x 和 b 的和
      return self.foo + x + b
  )");
  // 对模型进行评估
  m.eval();
  // 获取模型中的 forward 方法，并获取其计算图
  auto forward_g = m.get_method("forward").graph();
  // 使用测试工具 FileCheck 检查计算图，确保包含 "GetAttr"

  // GetAttr 的移除是通过冻结（freeze）实现的
  auto frozen_mod = torch::jit::freeze(m);
  // 获取冻结后模型中的 forward 方法，并获取其计算图
  forward_g = frozen_mod.get_method("forward").graph();
  // 使用 FileCheck 检查计算图，确保不包含 "GetAttr"

  // 如果没有设置训练模式，则 OFI 不会冻结模型
  auto frozen_mod2 = torch::jit::optimize_for_inference(m);
  // 获取优化后模型中的 forward 方法，并获取其计算图
  forward_g = frozen_mod2.get_method("forward").graph();
  // 使用 FileCheck 检查计算图，确保包含 "GetAttr"
TEST(ModuleAPITest, OfiFreezesTraining) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 向模块注册名为 "foo" 的参数，初始值为 torch::ones({})
  // 参数不需要梯度计算
  m.register_parameter("foo", torch::ones({}), false);
  // 定义模块的 forward 方法，接受参数 x 和可选参数 b（默认为 4）
  // 返回 self.foo + x + b 的计算结果
  m.define(R"(
    def forward(self, x, b : int = 4):
      return self.foo + x + b
  )");
  // 向模块注册名为 "training" 的属性，类型为 BoolType，初始值为 true
  m.register_attribute("training", BoolType::get(), true);
  // 将模块设置为评估模式（eval mode）
  m.eval();

  // 在冻结之前，检查 forward 方法的图形表示
  auto forward_g = m.get_method("forward").graph();
  testing::FileCheck().check("GetAttr")->run(*forward_g);

  // 演示 OFI 调用后的冻结效果
  // 冻结时会移除 GetAttr 操作，但仅当 training 属性设置为 true 时
  auto frozen_mod = torch::jit::optimize_for_inference(m);
  forward_g = frozen_mod.get_method("forward").graph();
  testing::FileCheck().check_not("GetAttr")->run(*forward_g);
}

TEST(ModuleAPITest, OfiFreezesNoForward) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 向模块注册名为 "foo" 的参数，初始值为 torch::ones({})
  // 参数不需要梯度计算
  m.register_parameter("foo", torch::ones({}), false);
  // 定义模块的 bar 方法，接受参数 x 和可选参数 b（默认为 4）
  // 返回 self.foo + x + b 的计算结果
  m.define(R"(
    def bar(self, x, b : int = 4):
      return self.foo + x + b
  )");
  // 将模块设置为评估模式（eval mode）
  m.eval();

  // 在不存在 forward 方法时调用 OFI
  auto frozen_mod =
      torch::jit::optimize_for_inference(m, std::vector<std::string>{"bar"});
  // 断言 bar 方法在原模块和冻结后的模块上运行结果一致
  ASSERT_EQ(
      m.run_method("bar", torch::ones({})).toTensor().item<float>(),
      frozen_mod.run_method("bar", torch::ones({})).toTensor().item<float>());
}

TEST(ModuleAPITest, To_CUDA) {
  // 创建名为 "test" 的模块对象
  Module m("test");
  {
    // 测试将参数和缓冲区从 CUDA 设备移动到 CPU 设备
    m.register_parameter("foo", torch::ones({}, at::kCUDA), false);
    m.register_buffer("bar", torch::ones({}, at::kCUDA));

    // 将模块移动到 CUDA 设备
    m.to(at::kCUDA);
    // 将模块移动回 CPU 设备后，断言 foo 和 bar 的设备为 CPU
    AT_ASSERT(m.attr("foo").toTensor().device().is_cpu());
    AT_ASSERT(m.attr("bar").toTensor().device().is_cpu());
  }
  {
    // 测试将参数和缓冲区从 CPU 设备移动到 CUDA 设备
    m.register_parameter("foo", torch::ones({}), false);
    m.register_buffer("bar", torch::ones({}));

    // 将模块移动到 CUDA 设备
    m.to(at::kCUDA);
    // 断言 foo 和 bar 的设备为 CUDA
    AT_ASSERT(m.attr("foo").toTensor().device().is_cuda());
    AT_ASSERT(m.attr("bar").toTensor().device().is_cuda());
  }
}
```