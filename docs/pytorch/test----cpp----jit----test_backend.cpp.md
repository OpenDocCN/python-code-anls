# `.\pytorch\test\cpp\jit\test_backend.cpp`

```py
// 包含 Google Test 的头文件，用于进行单元测试
#include <gtest/gtest.h>
// 包含 PyTorch JIT 编译测试所需的辅助函数和类
#include <test/cpp/jit/test_utils.h>
// 包含 PyTorch JIT 模块相关的头文件
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_detail.h>
// 包含 PyTorch 移动端模型导入相关的头文件
#include <torch/csrc/jit/mobile/import.h>
// 包含 PyTorch JIT 序列化导入相关的头文件
#include <torch/csrc/jit/serialization/import.h>
// 包含 PyTorch 核心头文件
#include <torch/torch.h>

// 测试代码应放在 torch::jit 命名空间中
namespace torch {
namespace jit {

// 定义 BackendTest 类，测试后端转换功能
TEST(BackendTest, ToBackend) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 定义模块的前向方法及其子方法
  m.define(R"(
    def forward(self, x, h):
        return self.accum(x, h), self.sub_accum(x, h)

    def accum(self, x, h):
        return x + h

    def sub_accum(self, x, h):
        return x - h
  )");

  // 创建输入向量
  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  // 调用模块的前向方法，并获取返回的元组对象的引用
  auto ref = m.forward(inputs).toTupleRef().elements().vec();

  // 创建编译规范的字典
  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  // 创建假字典并插入空键值对
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  // 将假字典插入到编译规范中
  compile_spec.insert("forward", fake_dict);
  // 创建字典类型，键为字符串类型，值为任意类型
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  
  // 生成降低后的模块
  auto lm = torch::jit::detail::codegen_backend_module(
      "test_backend", m, compile_spec, any_dict_ty);
  // 降低后的模块代码：
  /*
    class test_backendLoweredModule(Module):
      __parameters__ = []
      __buffers__ = []
      __processed_module : Any
      __method_compile_spec : Dict[str, Any]
      __backend : __torch__.torch.classes.__backends__.test_backend
      __handles : Dict[str, Any]
      def __create_backend(self: torch.jit.test_backendLoweredModule) -> None:
        _0 =
    __torch__.torch.classes.__backends__.test_backend.__new__(__torch__.torch.classes.__backends__.test_backend)
        _1 = (_0).__init__()
        self.__backend = _0
        return None
      def __getstate__(self: torch.jit.test_backendLoweredModule) ->
    Tuple[Dict[str, Any], Any]: _2 = (self.__method_compile_spec,
    self.__processed_module) return _2 def __setstate__(self:
    torch.jit.test_backendLoweredModule, state: Tuple[Dict[str, Any], Any]) ->
    None: self.__method_compile_spec = (state)[0] self.__processed_module =
    (state)[1] _3 = (self).__create_backend() _4 =
    (self.__backend).compile(self.__processed_module,
    self.__method_compile_spec, ) self.__handles = _4 return None def
    forward(self: torch.jit.test_backendLoweredModule, x: Tensor, h: Tensor) ->
    Tuple[Tensor, Tensor]: _5 = uninitialized(Tensor) typed_inputs =
    annotate(List[Any], [x, h]) _6 =
    (self.__backend).execute((self.__handles)["forward"], typed_inputs, ) _7,
  */
}
} // namespace jit
} // namespace torch
    # 解构赋值操作，将 _6 赋给 _8，同时将 isinstance(_7, Tensor) 的结果赋给 _9
    _8, = _6 _9 = isinstance(_7, Tensor) if _9: _10 = unchecked_cast(Tensor, _7)
        else:
          ops.prim.RaiseException("AssertionError: ")
          _10 = _5
        _11 = isinstance(_8, Tensor)
        if _11:
          _12 = unchecked_cast(Tensor, _8)
        else:
          ops.prim.RaiseException("AssertionError: ")
          _12 = _5
        return (_10, _12)

   """
  # 调用 lm 模型进行前向传播，并将结果转换为元组引用，再提取其中的元素为向量
  auto res = lm.forward(inputs).toTupleRef().elements().vec();
  # 断言第一个元素的张量是否与参考结果 ref 的第一个元素张量相等
  AT_ASSERT(res[0].toTensor().equal(ref[0].toTensor()));
  # 断言第二个元素的张量是否与参考结果 ref 的第二个元素张量相等
  AT_ASSERT(res[1].toTensor().equal(ref[1].toTensor()));
// 定义一个名为 `Module` 的对象 `m`，其名为 "m"
Module m("m");
// 定义一个 Python 脚本字符串，实现了 `forward`、`accum` 和 `sub_accum` 方法
m.define(R"(
  def forward(self, x, h):
      return self.accum(x, h), self.sub_accum(x, h)

  def accum(self, x, h):
      return x + h

  def sub_accum(self, x, h):
      return x - h
)");

// 创建一个空的输入向量 `inputs`
std::vector<IValue> inputs;
// 向 `inputs` 向量中添加一个包含全 2.0 的标量张量
inputs.emplace_back(2.0 * torch::ones({}));
// 向 `inputs` 向量中添加一个包含全 1.0 的标量张量
inputs.emplace_back(1.0 * torch::ones({}));

// 调用 `m` 的 `forward` 方法，并将返回值转换为元组引用
auto ref = m.forward(inputs).toTupleRef().elements().vec();

// 创建两个空的 `c10::Dict` 类型的字典，键和值的类型都是 `IValue`
c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());

// 向 `fake_dict` 中插入一个空键和一个空值
fake_dict.insert("", "");

// 向 `compile_spec` 中插入一个名为 "forward" 的键和 `fake_dict` 作为值
compile_spec.insert("forward", fake_dict);

// 创建一个 `DictType` 类型，键为 `StringType`，值为 `AnyType`
auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());

// 生成降级后的模块（后端不可用）
auto lm = torch::jit::detail::codegen_backend_module(
    "test_backend_unavailable", m, compile_spec, any_dict_ty);

// 验证在执行时如果后端不可用会抛出异常
ASSERT_THROWS_WITH_MESSAGE(
    lm.forward(inputs).toTupleRef().elements(), "Backend is not available.");
    // 定义一个成员函数 `forward`，接受两个参数 x 和 h，并返回它们的和
    def forward(self, x, h):
        return x + h
  )");

  // 创建一个空的输入向量 inputs
  std::vector<IValue> inputs;
  // 将 2.0 的张量添加到 inputs 向量中
  inputs.emplace_back(2.0 * torch::ones({}));
  // 将 1.0 的张量添加到 inputs 向量中
  inputs.emplace_back(1.0 * torch::ones({}));
  // 调用模型 m 的 forward 方法，得到结果 ref
  auto ref = m.forward(inputs);

  // 创建一个空的 compile_spec 字典，键和值的类型为 StringType 和 AnyType
  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  // 创建一个空的 fake_dict 字典，键和值的类型为 StringType 和 AnyType
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  // 向 fake_dict 中插入一个空键值对
  fake_dict.insert("", "");
  // 将 fake_dict 作为值插入 compile_spec 字典中的 "forward" 键
  compile_spec.insert("forward", fake_dict);
  // 创建一个类型为 DictType 的 any_dict_ty 变量，键类型为 StringType，值类型为 AnyType
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // 使用 codegen_backend_module 函数生成 lowered module lm
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m, compile_spec, any_dict_ty);
  // 调用 lm 的 forward 方法，得到结果 res
  auto res = lm.forward(inputs);
  // 断言 res 转换为张量后与 ref 转换为张量相等
  AT_ASSERT(res.toTensor().equal(ref.toTensor()));

  // 创建一个字符串流 ss
  std::stringstream ss;
  // 将 lm 保存为移动端模型到 ss 中
  lm._save_for_mobile(ss);
  // 从字符串流 ss 加载移动端模型，得到 mlm
  auto mlm = _load_for_mobile(ss);
  // 调用 mlm 的 forward 方法，得到 mres
  auto mres = mlm.forward(inputs);
  // 设置不使用带字符串表的格式化
  setShouldUseFormatWithStringTable(false);
  // 断言 mres 转换为张量后与 ref 转换为张量相等
  AT_ASSERT(mres.toTensor().equal(ref.toTensor()));
TEST(BackendTest, TestComposite) {
  // 创建一个字典，键和值的类型为 IValue
  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  // 创建另一个字典，键和值的类型为 IValue
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  // 向 fake_dict 中插入一个空键值对
  fake_dict.insert("", "");
  // 将 fake_dict 作为值插入 compile_spec 的 "forward" 键下
  compile_spec.insert("forward", fake_dict);
  // 创建一个字典类型，键为字符串类型，值为任意类型
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());

  // 创建名为 "m_add" 的 Module 对象
  Module m_add("m_add");
  // 定义 "m_add" 模块的前向计算方法，实现为返回输入 x 和 y 的和
  m_add.define(R"(
    def forward(self, x, y):
      return x + y
  )");
  // 使用 JIT 后端编码 m_add 模块，生成一个 LoweredModule
  auto lm_add = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m_add, compile_spec, any_dict_ty);

  // 创建名为 "m_sub" 的 Module 对象
  Module m_sub("m_sub");
  // 定义 "m_sub" 模块的前向计算方法，实现为返回输入 x 和 y 的差
  m_sub.define(R"(
    def forward(self, x, y):
      return x - y
  )");
  // 使用 JIT 后端编码 m_sub 模块，生成一个 LoweredModule
  auto lm_sub = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m_sub, compile_spec, any_dict_ty);

  // 创建名为 "C" 的 Module 对象
  Module c("C");
  // 向 "C" 模块注册 "Add" 子模块，使用 lm_add
  c.register_module("Add", lm_add);
  // 向 "C" 模块注册 "Sub" 子模块，使用 lm_sub
  c.register_module("Sub", lm_sub);
  // 定义 "C" 模块的前向计算方法，实现为调用 Add 和 Sub 子模块的前向方法并返回乘积
  c.define(R"(
    def forward(self, x, y):
      return self.Add.forward(x, y) * self.Sub.forward(x, y)
  )");

  // 创建输入向量
  std::vector<IValue> inputs;
  inputs.emplace_back(3.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  // 执行 "C" 模块的前向计算，得到 res_jit
  auto res_jit = c.forward(inputs);

  // 创建字符串流对象 ss
  std::stringstream ss;
  // 将 "C" 模块保存为移动端序列化格式到 ss 中
  c._save_for_mobile(ss);
  // 从 ss 中加载移动端序列化格式的模块，得到 mc
  auto mc = _load_for_mobile(ss);
  // 执行 mc 模块的前向计算，得到 res_mobile
  auto res_mobile = mc.forward(inputs);

  // 断言 res_jit 和 res_mobile 的张量是否相等
  AT_ASSERT(res_jit.toTensor().equal(res_mobile.toTensor()));
}
    # 定义一个方法 forward，接受三个参数：a，b 和 s（s 为整数类型）
    def forward(self, a, b, s:int):
      # 调用 self.Add 对象的 forward 方法，计算 a 和 b 的加法结果
      c = self.Add.forward(a, b)
      # 调用 self.Sub 对象的 forward 方法，计算 a 和 b 的减法结果
      d = self.Sub.forward(a, b)
      # 计算 y，等于 s 乘以 (c 乘以 d)
      y = s * (c * d)
      # 返回计算得到的 y 值
      return y
  )");

  # 返回变量 c 的值（注意：这行代码应当属于外部作用域，不在 forward 方法中）
  return c;
// 定义一个测试用例，测试带有设置状态的复合模块的行为
TEST(BackendTest, TestCompositeWithSetStates) {
  // 获取包含相同名称子模块的复合模块
  Module c = getCompositeModuleWithSameNameSubModules();

  // 准备输入数据
  std::vector<IValue> inputs;
  inputs.emplace_back(torch::ones({}));
  inputs.emplace_back(3.0 * torch::ones({}));
  inputs.emplace_back(3);

  // 使用输入数据进行前向推断
  auto res_jit = c.forward(inputs);

  // 创建一个字符串流对象，将复合模块保存为移动端可用格式
  std::stringstream ss;
  c._save_for_mobile(ss);

  // 从字符串流中加载移动端模块
  auto mc = _load_for_mobile(ss);

  // 使用加载的移动端模块进行前向推断
  auto res_mobile = mc.forward(inputs);

  // 断言两种推断结果是否一致
  AT_ASSERT(res_jit.toTensor().equal(res_mobile.toTensor()));
}

// 定义另一个测试用例，测试带有设置状态的复合模块的一致性
TEST(BackendTest, TestConsistencyOfCompositeWithSetStates) {
  // 获取包含相同名称子模块的复合模块
  Module c = getCompositeModuleWithSameNameSubModules();

  // 准备输入数据
  std::vector<IValue> inputs;
  inputs.emplace_back(torch::ones({}));
  inputs.emplace_back(3.0 * torch::ones({}));
  inputs.emplace_back(3);

  // 创建两个字符串流对象，一个用于保存模块，另一个用于重新保存模块
  std::stringstream ss, ss_resave;
  c._save_for_mobile(ss);

  // 从字符串流中加载移动端模块
  auto mc = _load_for_mobile(ss);

  // 使用加载的移动端模块进行前向推断
  auto res_mobile = mc.forward(inputs);

  // 重置字符串流的位置到开头
  ss.seekg(0, ss.beg);

  // 检查方法名是否始终相同，通过重新加载脚本模块并重新保存为移动端模块来确保
  auto script_module_load = torch::jit::load(ss);
  script_module_load._save_for_mobile(ss_resave);
  auto mc_reload = _load_for_mobile(ss_resave);

  // 使用重新加载的移动端模块进行前向推断
  auto res_mobile_reload = mc_reload.forward(inputs);

  // 断言重新加载的模块和原始模块的推断结果是否一致
  AT_ASSERT(res_mobile_reload.toTensor().equal(res_mobile.toTensor()));

  // 获取原始模块和重新加载模块的方法列表
  auto mc_methods = mc.get_methods();
  auto mc_reload_methods = mc_reload.get_methods();

  // 定义函数获取方法的完全限定名称
  auto get_qual_name = [](mobile::Method method) -> std::string {
    return method.function().qualname().qualifiedName();
  };

  // 获取原始模块和重新加载模块的方法完全限定名称列表
  std::vector<std::string> mc_method_qns, mc_reload_method_qns;
  std::transform(
      mc_methods.begin(),
      mc_methods.end(),
      std::back_inserter(mc_method_qns),
      get_qual_name);
  std::transform(
      mc_reload_methods.begin(),
      mc_reload_methods.end(),
      std::back_inserter(mc_reload_method_qns),
      get_qual_name);

  // 断言两个模块的方法完全限定名称列表是否完全相同
  AT_ASSERT(std::equal(
      mc_method_qns.begin(),
      mc_method_qns.end(),
      mc_reload_method_qns.begin()));
}

// 定义一个测试用例，测试编译器不支持的情况
TEST(BackendTest, TestCompilerNotSupport) {
  // 创建一个名为 "m" 的模块，并定义其 forward 方法
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return x * h
  )");

  // 创建用于编译的规范字典和假字典
  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");

  // 向编译规范字典中插入假的 forward 方法字典
  compile_spec.insert("forward", fake_dict);

  // 创建字符串类型到任意类型的字典类型
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());

  // 断言编译时抛出异常，且异常信息包含特定文本
  ASSERT_THROWS_WITH_MESSAGE(
      torch::jit::detail::codegen_backend_module(
          "backend_with_compiler_demo", m, compile_spec, any_dict_ty),
      "The node of aten::mul is not supported in this compiler. Source code:");
}

// 定义一个带有调试信息的测试用例，测试编译器功能
TEST(BackendTestDebugInfo, TestCompiler) {
  // 创建一个名为 "m" 的模块，并定义其 forward 方法
  Module m("m");
  m.define(R"(
    // 定义一个成员函数 `forward`，接受两个参数 x 和 h，并返回它们的和
    def forward(self, x, h):
        return x + h
  )");

  // 创建一个空的输入向量 inputs
  std::vector<IValue> inputs;
  // 向 inputs 中添加一个形状为 {2, 4} 的随机张量
  inputs.emplace_back(torch::rand({2, 4}));
  // 向 inputs 中添加一个形状为 {13, 9} 的随机张量
  inputs.emplace_back(torch::rand({13, 9}));

  // 创建一个空的编译规范字典 compile_spec，键和值类型均为任意类型
  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  // 创建一个空的任意类型字典 fake_dict
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  // 向 fake_dict 中插入空字符串键和空字符串值
  fake_dict.insert("", "");
  // 将 fake_dict 插入到 compile_spec 中，键为 "forward"
  compile_spec.insert("forward", fake_dict);

  // 创建一个键类型为字符串，值类型为任意类型的字典类型 any_dict_ty
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // 使用 codegen_backend_module 函数生成一个降阶的模块 lm
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m, compile_spec, any_dict_ty);

  // 创建一个字符串流 ss
  std::stringstream ss;
  // 将 lm 序列化为移动端可以加载的格式并写入 ss
  lm._save_for_mobile(ss, ExtraFilesMap(), true);
  // 从字符串流 ss 加载一个移动端模块 mlm
  auto mlm = _load_for_mobile(ss);
  // 定义一个错误匹配模式的字符串 error_pattern
  std::string error_pattern = R"(
  Module hierarchy:top(m)::<unknown>.__loweredModule__(m)::forward.aten::add
TEST(BackendTestDebugInfo, TestCompilerWithStringTable) {
  // 设置使用字符串表格式
  setShouldUseFormatWithStringTable(true);
  
  // 创建名为 "m" 的模块
  Module m("m");
  // 定义模块 "m" 的 forward 方法，接受两个参数 x 和 h，返回它们的和
  m.define(R"(
    def forward(self, x, h):
        return x + h
  )");

  // 创建输入向量 inputs，包含两个形状分别为 [2, 4] 和 [13, 9] 的随机张量
  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  // 创建 compile_spec 字典，键和值的类型都是 AnyType
  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  // 创建 fake_dict 字典，键和值的类型都是 AnyType，并插入一个空字符串键值对
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);

  // 创建 any_dict_ty 字典类型，键类型是 StringType，值类型是 AnyType
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());

  // 生成降阶模块 lm，通过调用代码生成后端模块函数
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m, compile_spec, any_dict_ty);

  // 创建字符串流 ss
  std::stringstream ss;
  // 将 lm 保存为移动模型格式到字符串流 ss
  lm._save_for_mobile(ss, ExtraFilesMap(), true);
  // 从字符串流 ss 加载移动模型到 mlm
  auto mlm = _load_for_mobile(ss);

  // 定义错误模式字符串
  std::string error_pattern = R"(
  Module hierarchy:top(m)::<unknown>.__loweredModule__(m)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

            def forward(self, x: Tensor, h: Tensor):
                return self.__loweredModule__.forward(x, h)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, h, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in <unknown>

    def forward(self, x, h):
        return x + h
               ~~~~~ <--- HERE
  )";

  // 设置不使用字符串表格式
  setShouldUseFormatWithStringTable(false);
  // 断言 mlm.forward(inputs) 抛出错误并且错误消息匹配 error_pattern
  ASSERT_THROWS_WITH_MESSAGE(mlm.forward(inputs), error_pattern);
}

TEST(BackendTestDebugInfo, TestExceptionStackForCompilerWithModuleHierarchy) {
  // 创建名为 "A" 的模块 a
  Module a("A");
  // 定义模块 "A" 的 forward 方法，接受两个参数 x 和 y，返回它们的和
  a.define(R"(
    def forward(self, x, y):
      return x + y
  )");

  // 创建名为 "B" 的模块 b
  Module b("B");
  // 定义模块 "B" 的 forward 方法，接受两个参数 x 和 y，返回它们的和
  b.define(R"(
    def forward(self, x, y):
      return x + y
  )");
}
    // 定义一个名为 forward 的方法，接受参数 x，返回 x 加 2 的结果
    def forward(self, x):
      return x + 2
  )");
  // 创建一个名为 c 的 Module 对象，名为 "C"
  Module c("C");
  // 将名为 a 的 Module 注册到 c 中，使用名字 "A0"
  c.register_module("A0", a);
  // 将名为 b 的 Module 注册到 c 中，使用名字 "B0"
  c.register_module("B0", b);
  // 定义一个新的方法 forward，接受参数 x 和 y，返回 self.A0.forward(x, y) + self.B0.forward(x) 的结果
  c.define(R"(
    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
  )");

  // 创建一个空的 IValue 向量 inputs
  std::vector<IValue> inputs;
  // 向 inputs 中添加一个大小为 {2, 4} 的随机张量
  inputs.emplace_back(torch::rand({2, 4}));
  // 向 inputs 中添加一个大小为 {13, 9} 的随机张量
  inputs.emplace_back(torch::rand({13, 9}));

  // 创建一个新的 c10::Dict，键和值的类型为 StringType 和 AnyType
  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  // 创建一个空的 c10::Dict，键和值的类型为 StringType 和 AnyType，并插入一个空的键值对
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", ""); // 插入一个空的键值对
  // 将 fake_dict 插入到 compile_spec 中，键为 "forward"
  compile_spec.insert("forward", fake_dict);
  // 创建一个新的字典类型，键类型为 StringType，值类型为 AnyType
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // 生成一个降低后的模块 lm，使用 codegen_backend_module 函数
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", c, compile_spec, any_dict_ty);

  // 创建一个 stringstream 对象 ss
  std::stringstream ss;
  // 将 lm 保存到 ss 中，包含额外文件信息，压缩
  lm._save_for_mobile(ss, ExtraFilesMap(), true);
  // 从 stringstream ss 中加载 lm，保存到 mlm 中
  auto mlm = _load_for_mobile(ss);
  // 定义一个字符串错误模式，用于错误匹配
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.__loweredModule__(C)::forward.A0(A)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

# TorchScript的错误回溯信息，显示错误发生在未知位置的第3行


            def forward(self, x: Tensor, y: Tensor):
                return self.__loweredModule__.forward(x, y)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

# 定义一个名为forward的方法，接受两个参数x和y，并调用__loweredModule__对象的forward方法进行处理


  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, y, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0

# 在forward方法中，将输入参数x和y封装为一个列表typed_inputs，然后检查self.__backend是否可用，如果可用则执行self.__handles["forward"]对应的操作，并期望返回一个Tensor对象


  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
             ~~~~~~~~~~~~~~~ <--- HERE

# 定义一个名为forward的方法，接受两个参数x和y，并调用self.A0.forward和self.B0.forward方法对它们进行处理


  File "<string>", line 3, in forward

    def forward(self, x, y):
      return x + y
             ~~~~~ <--- HERE

# 定义一个名为forward的方法，接受两个参数x和y，并返回它们的求和结果


  )";
  ASSERT_THROWS_WITH_MESSAGE(mlm.forward(inputs), error_pattern);
}

# 使用ASSERT_THROWS_WITH_MESSAGE宏来测试mlm.forward方法，期望其抛出特定的错误消息

TEST(
    BackendTestDebugInfo,
    TestExceptionStackForCompilerWithTwoLevelModuleHierarchy) {
  Module a("A");
  a.define(R"(
    def forward(self, x, y):
      return x + y
  )");
  Module b("B");
  b.register_module("A0", a);
  b.define(R"(
    def forward(self, x, y):
      return self.A0.forward(x, y) + 2
  )");
  Module c("C");
  c.register_module("B0", b);
  c.define(R"(

# 在BackendTestDebugInfo测试中，测试编译器在两级模块层次结构下的异常堆栈信息
    // 定义一个成员函数 forward，接受两个参数 x 和 y，返回 B0.forward(x, y) 加 3 的结果
    def forward(self, x, y):
      return self.B0.forward(x, y) + 3
  )");

  // 创建一个空的输入向量 inputs
  std::vector<IValue> inputs;
  // 将两个随机生成的张量 torch::rand({2, 4}) 和 torch::rand({13, 9}) 添加到 inputs 中
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  // 创建一个键值对字典 compile_spec，键和值类型为 StringType 和 AnyType
  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  // 创建一个空的键值对字典 fake_dict，键和值类型为 StringType 和 AnyType
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  // 向 fake_dict 中插入一个空键值对 ""
  fake_dict.insert("", "");
  // 将 fake_dict 插入到 compile_spec 中，键为 "forward"
  compile_spec.insert("forward", fake_dict);
  // 创建一个类型为 DictType 的 any_dict_ty，键和值类型为 StringType 和 AnyType
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  
  // 使用 codegen_backend_module 函数生成一个降低了的模块 lm
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", c, compile_spec, any_dict_ty);

  // 创建一个 stringstream 对象 ss
  std::stringstream ss;
  // 将 lm 保存到 ss 中，并额外指定空的 ExtraFilesMap 和 true 参数
  lm._save_for_mobile(ss, ExtraFilesMap(), true);
  // 从 stringstream ss 加载模块 mlm
  auto mlm = _load_for_mobile(ss);

  /*
   * 错误栈的抛出看起来像这样:
   * Module hierarchy:top(backend_with_compiler_demoLoweredModule).B0(B).A0(A)
   * TorchScript 的回溯（最近的调用在最前面）:
   * 文件 "<string>"，第 5 行，函数名 UNKNOWN
   *               typed_inputs: List[Any] = [x, y, ]
   *               if self.__backend.is_available() :
   *                 _0, = self.__backend.execute(self.__handles["forward"],
   * typed_inputs)
   *                       ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
   *                 assert isinstance(_0, Tensor)
   *                 return _0
   *  文件 "<string>"，第 3 行，函数名 UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.B0.forward(x, y) + 3
   *             ~~~~~~~~~~~~~~~ <--- HERE
   *
   *  文件 "<string>"，第 3 行，函数名 UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.A0.forward(x, y) + 2
   *             ~~~~~~~~~~~~~~~ <--- HERE
   *
   *  文件 "<string>"，第 3 行，函数名 UNKNOWN
   *
   *    def forward(self, x, y):
   *      return x + y
   *             ~~~~~ <--- HERE
   *
   */
  
  // 定义一个字符串 error_pattern，其内容是多行的错误模式
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.__loweredModule__(C)::forward.B0(B)::forward.A0(A)::forward.aten::add
# 定义一个名为 CompilationUnit 的共享编译单元对象
cu = std::make_shared<CompilationUnit>();
# 创建名为 a 的模块对象 "A"，并定义其 forward 方法
Module a("A");
a.define(R"(
  def forward(self, x, y):
    return x + y
)");
# 创建名为 b 的模块对象 "B"，并定义其 forward 方法
Module b("B");
b.define(R"(
  def forward(self, x):
    return x + 2
)");
# 创建名为 c 的模块对象 "C"，并注册模块 a 为 "A0"，模块 b 为 "B0"，然后定义 c 的 forward 方法
Module c("C");
c.register_module("A0", a);
c.register_module("B0", b);
c.define(R"(
  def forward(self, x, y):
    return self.A0.forward(x, y) + self.B0.forward(x)
)");

# 创建一个空的输入向量 inputs，并插入两个张量随机数值
std::vector<IValue> inputs;
inputs.emplace_back(torch::rand({2, 4}));
inputs.emplace_back(torch::rand({13, 9}));

# 创建两个空的字典，用于编译规格和假字典
c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());

# 向 fake_dict 中插入一个空字符串键值对
fake_dict.insert("", "");

# 向 compile_spec 中插入 "forward" 键，并赋予 fake_dict 的值
compile_spec.insert("forward", fake_dict);

# 获取 c 的 "A0" 属性，将其转换为模块 current_sm
IValue submodule = c.attr("A0");
Module current_sm = submodule.toModule();

# 创建一个键为字符串类型，值为任意类型的字典类型 any_dict_ty
auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());

# 使用代码生成后端模块函数，创建降级的子模块 lowered_submodule
auto lowered_submodule = torch::jit::detail::codegen_backend_module(
    "backend_with_compiler_demo", current_sm, compile_spec, any_dict_ty);

# 修改 c 的 "A0" 属性的类型为 lowered_submodule 的类型
c.type()->unsafeChangeAttributeType("A0", lowered_submodule.type());
# 设置 c 的 "A0" 属性为 lowered_submodule 的 _ivalue()
c.setattr("A0", lowered_submodule._ivalue());

# 创建类型重映射的无序映射 type_remap
std::unordered_map<TypePtr, TypePtr> type_remap;

# 定义类型重映射函数 type_remap_fn，根据 type_remap 修改方法的图形和模式
auto type_remap_fn = [&type_remap](TypePtr in) {
  auto it = type_remap.find(in);
  if (it == type_remap.end())
    return in;
  return it->second;
};

# 遍历 c 的方法，获取方法的图形，通过 type_remap_fn 重映方法的类型
for (auto& fn : c.type()->methods()) {
  auto method = c.get_method(fn->name());
  auto graph = method.graph();
  graph->remapTypes(type_remap_fn);
  auto new_schema = fn->getSchema().cloneWithRemappedTypes(type_remap_fn);
}
    fn->setSchema(new_schema);

设置 `fn` 对象的模式为 `new_schema`。


  std::stringstream ss;

创建一个 `std::stringstream` 对象 `ss`，用于操作字符串流。


  c._save_for_mobile(ss, ExtraFilesMap(), true);

调用对象 `c` 的 `_save_for_mobile` 方法，将其状态保存到字符串流 `ss` 中，同时传入空的附加文件映射和 `true` 作为参数。


  auto c_loaded = _load_for_mobile(ss);

从字符串流 `ss` 中加载移动设备所需的对象状态，并将结果存储在 `c_loaded` 变量中。


  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.A0(A)::forward.__loweredModule__(A)::forward.aten::add

创建一个名为 `error_pattern` 的 `std::string` 对象，用于存储多行原始字符串，内容描述了一个错误模式的层次结构。
// 定义 TorchScript 模块的 forward 方法，接受两个参数 x 和 y
def forward(self, x, y):
  // 调用 A0 模块的 forward 方法，并将结果与 B0 模块的 forward 方法的结果相加
  return self.A0.forward(x, y) + self.B0.forward(x)
         ~~~~~~~~~~~~~~~ <--- 这里指示 forward 方法的返回值是 A0 和 B0 模块方法的结果之和

// 定义 TorchScript 模块的 forward 方法，接受两个参数 x 和 y
def forward(self, x: Tensor, y: Tensor):
  // 调用 loweredModule 的 forward 方法，并返回其结果
  return self.__loweredModule__.forward(x, y)
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- 这里表明 forward 方法返回 loweredModule 的计算结果

// 定义 TorchScript 模块的 forward 方法，接受输入参数列表 typed_inputs
def forward(self, x, y):
  // 如果后端可用
  if self.__backend.is_available() :
    // 执行 forward 方法，使用输入参数 typed_inputs
    _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
          ~~~~~~~~~~~~~~~~~~~~~~ <--- 这里是调用 backend 的 execute 方法，获取执行结果 _0
    // 断言 _0 是 Tensor 类型
    assert isinstance(_0, Tensor)
    // 返回 _0 作为 forward 方法的结果
    return _0

// 定义 TorchScript 模块的 forward 方法，接受两个参数 x 和 y
def forward(self, x, y):
  // 返回 x 和 y 的和作为 forward 方法的结果
  return x + y
         ~~~~~ <--- 这里指示 forward 方法的返回值是 x 和 y 的和
    fn->setSchema(new_schema);
  }


// 设置 fn 对象的模式为 new_schema
fn->setSchema(new_schema);



  std::stringstream ss;
  c._save_for_mobile(ss, ExtraFilesMap(), true);
  auto c_loaded = _load_for_mobile(ss);
  /*
   * Erro stack trace will look like this:
   * Module hierarchy:top(C).A0(backend_with_compiler_demoLoweredModule).AA0(AA)
   * Traceback of TorchScript (most recent call last):
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.A0.forward(x, y) + self.B0.forward(x)
   *             ~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 5, in FunctionName_UNKNOWN
   *                typed_inputs: List[Any] = [x, y, ]
   *                if self.__backend.is_available() :
   *                  _0, = self.__backend.execute(self.__handles["forward"],
   * typed_inputs)
   *                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
   *                  assert isinstance(_0, Tensor)
   *                  return _0
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.AA0.forward(x, y) + 3
   *             ~~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return x + y
   *             ~~~~~ <--- HERE
   *
   *
   *  */
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.A0(A)::forward.__loweredModule__(A)::forward.AA0(AA)::forward.aten::add


// 创建一个用于存储字符串的流对象 ss
std::stringstream ss;
// 将 c 对象保存为移动端可用格式，并将其写入到 ss 流中
c._save_for_mobile(ss, ExtraFilesMap(), true);
// 从 ss 流中加载移动端数据，并将结果存储在 c_loaded 中
auto c_loaded = _load_for_mobile(ss);
/*
 * 下面是错误堆栈的示例：
 * Module hierarchy:top(C).A0(backend_with_compiler_demoLoweredModule).AA0(AA)
 * Traceback of TorchScript (most recent call last):
 *  File "<string>", line 3, in FunctionName_UNKNOWN
 *
 *    def forward(self, x, y):
 *      return self.A0.forward(x, y) + self.B0.forward(x)
 *             ~~~~~~~~~~~~~~~ <--- HERE
 *
 *  File "<string>", line 5, in FunctionName_UNKNOWN
 *                typed_inputs: List[Any] = [x, y, ]
 *                if self.__backend.is_available() :
 *                  _0, = self.__backend.execute(self.__handles["forward"],
 * typed_inputs)
 *                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
 *                  assert isinstance(_0, Tensor)
 *                  return _0
 *  File "<string>", line 3, in FunctionName_UNKNOWN
 *
 *    def forward(self, x, y):
 *      return self.AA0.forward(x, y) + 3
 *             ~~~~~~~~~~~~~~~~ <--- HERE
 *
 *  File "<string>", line 3, in FunctionName_UNKNOWN
 *
 *    def forward(self, x, y):
 *      return x + y
 *             ~~~~~ <--- HERE
 *
 *
 *  */
// 定义错误模式字符串，用于匹配错误堆栈中的模式
std::string error_pattern = R"(
Module hierarchy:top(C)::<unknown>.A0(A)::forward.__loweredModule__(A)::forward.AA0(AA)::forward.aten::add
";
# 定义一个类中的方法 `forward`，接受两个参数 x 和 y
def forward(self, x, y):
    # 调用对象 A0 的 forward 方法，传入参数 x 和 y，返回结果，并加上对象 B0 的 forward 方法对 x 的结果
    return self.A0.forward(x, y) + self.B0.forward(x)
                        ~~~~~~~~~~~~~~~ <--- 这里

# 在 forward 方法中定义一个新的 forward 方法，接受两个参数 x 和 y，并返回调用 __loweredModule__ 对象的 forward 方法的结果
def forward(self, x: Tensor, y: Tensor):
    return self.__loweredModule__.forward(x, y)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- 这里

# 在 forward 方法中定义一个新的 forward 方法，接受两个参数 x 和 y
def forward(self, x, y):
    # 创建一个列表 typed_inputs，包含 x 和 y 两个元素
    typed_inputs: List[Any] = [x, y, ]
    # 如果后端对象 __backend 可用
    if self.__backend.is_available() :
        # 调用 __backend 对象的 execute 方法，执行 forward 方法，传入 typed_inputs 参数，返回结果保存到 _0 变量中
        _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- 这里
        # 断言 _0 是 Tensor 类型
        assert isinstance(_0, Tensor)
        # 返回 _0 变量作为结果
        return _0

# 在 forward 方法中定义一个新的 forward 方法，接受两个参数 x 和 y
def forward(self, x, y):
    # 返回 x 和 y 的和
    return x + y
             ~~~~~ <--- 这里
```