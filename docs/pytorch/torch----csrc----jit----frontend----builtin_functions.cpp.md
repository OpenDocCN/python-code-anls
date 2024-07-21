# `.\pytorch\torch\csrc\jit\frontend\builtin_functions.cpp`

```py
// 引入头文件，包含 Torch JIT 前端内置函数的声明
#include <torch/csrc/jit/frontend/builtin_functions.h>

// 引入各种依赖的头文件
#include <ATen/code_template.h>
#include <caffe2/serialize/versions.h>
#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/jit/frontend/resolver.h>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 定义标量操作的源码模板
auto scalar_operators_source = at::jit::CodeTemplate(
    R"SCRIPT(
def mul(a : ${Scalar}, b : Tensor) -> Tensor:
  return b * a
def add(a : ${Scalar}, b : Tensor) -> Tensor:
  return b + a
def ne(a : ${Scalar}, b : Tensor) -> Tensor:
  return b != a
def eq(a : ${Scalar}, b : Tensor) -> Tensor:
  return b == a
def sub(a : ${Scalar}, b : Tensor) -> Tensor:
  return torch.neg(b) + a
def div(a : ${Scalar}, b : Tensor) -> Tensor:
  return torch.reciprocal(b) * a
)SCRIPT");

// 定义无复杂数的标量操作的源码模板
auto scalar_operators_no_complex_source = at::jit::CodeTemplate(
    R"SCRIPT(
def lt(a : ${Scalar}, b : Tensor) -> Tensor:
  return b > a
def le(a : ${Scalar}, b : Tensor) -> Tensor:
  return b >= a
def gt(a : ${Scalar}, b : Tensor) -> Tensor:
  return b < a
def ge(a : ${Scalar}, b : Tensor) -> Tensor:
  return b <= a
)SCRIPT");

// 定义 ntuple 操作的源码模板
auto _ntuple_ops = at::jit::CodeTemplate(
    R"SCRIPT(
def _${name}(x: BroadcastingList${Length}[${Scalar}]) -> List[${Scalar}]:
  return x
)SCRIPT");

// 定义 floor division 操作的源码模板
auto floordiv = at::jit::CodeTemplate(
    R"SCRIPT(
def floordiv(self : Tensor, other : ${Rhs_Type}) -> Tensor:
  return torch.floor_divide(self, other)
)SCRIPT");

// 定义一些张量属性的源码字符串
auto tensor_properties =
    R"SCRIPT(
def ndim(a : Tensor) -> int:
  return a.dim()
def T(a : Tensor) -> Tensor:
  return a.numpy_T()
def H(a : Tensor) -> Tensor:
  return a.matrix_H()
def mT(a : Tensor) -> Tensor:
  return a.mT
def mH(a : Tensor) -> Tensor:
  return a.mH
def shape(a : Tensor) -> List[int]:
  return a.size()
)SCRIPT";

// 定义 aten 操作的源码字符串
auto aten_ops =
    R"SCRIPT(
def _assert_int_or_pair(vals: List[int], name: str, message: str):
  pass
def list_with_default(out_size: List[int], defaults: List[int]):
  assert len(defaults) > len(out_size)
  return out_size
def _assert(condition : bool, message : str):
  assert condition, message
# existing device operator is registered with input name `a`, which prevents
# torch.device(type="cuda") from working. add shim-layer here
def device(type: str):
  return torch.device(type)
def type(self: Tensor, dtype: int, non_blocking: bool=False, copy: bool=False) -> Tensor:
  return self.to(dtype, non_blocking, copy)
)SCRIPT";

// 附加的 Tensor 变体的 aten 操作源码字符串
const auto aten_ops_additional =
    R"SCRIPT(
def _assert(condition : Tensor, message : str):
  assert bool(condition), message
def __contains__(self: str, key: str):
    return self.find(key, 0, len(self)) != -1
)SCRIPT";

// 定义 BuiltinFunctionRegistry 结构体
struct BuiltinFunctionRegistry {
  // 获取给定符号的所有内置函数的函数指针列表
  const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name) {
    const static std::vector<Function*> empty;
    // 定义一个静态常量空向量，用于返回空结果集
    
    // 进入函数时，使用递归互斥锁以避免死锁，并在初始化时报告未加载内建函数
    std::lock_guard<std::recursive_mutex> guard(mutex);
    if (state == INTIIALIZING) {
      // 如果状态为初始化中，则返回空向量
      return empty;
    } else if (state == UNINITIALIZED) {
      // 如果状态为未初始化，则将状态设置为初始化中
      state = INTIIALIZING;
      // 加载内建函数
      loadBuiltinFunctions();
      // 设置状态为已初始化
      state = INITIALIZED;
    }
    // 确保状态为已初始化
    AT_ASSERT(state == INITIALIZED);
    // 查找内建函数名是否存在于映射中
    auto it = builtins_by_name_.find(name);
    if (it == builtins_by_name_.end())
      // 如果未找到，则返回空向量
      return empty;
    // 返回找到的内建函数列表
    return it->second;
    }
    
    private:
    void loadSource(const std::string& source, const std::string& the_namespace) {
      // 创建一个共享指针编译单元
      std::shared_ptr<CompilationUnit> cu = std::make_shared<CompilationUnit>();
      // 将编译单元添加到模块列表中
      modules.emplace_back(cu);
      // 定义编译单元中的函数
      cu->define(c10::nullopt, source, nativeResolver(), /*self=*/nullptr);
      // 遍历编译单元中的函数，将内建函数按名称加入到内建函数映射中
      for (auto& method : cu->get_functions()) {
        builtins_by_name_[Symbol::fromQualString(
                              the_namespace + "::" + method->name())]
            .push_back(method);
      }
    }
    
    void loadBuiltinFunctions() {
      // 加载 float、int、complex 类型的操作源码
      for (auto scalar : {"float", "int", "complex"}) {
        at::jit::TemplateEnv env;
        env.s("Scalar", scalar);
        loadSource(scalar_operators_source.format(env), "aten");
      }
    
      // 加载 float、int 类型但不包含 complex 的操作源码
      for (auto scalar : {"float", "int"}) {
        at::jit::TemplateEnv env;
        env.s("Scalar", scalar);
        loadSource(scalar_operators_no_complex_source.format(env), "aten");
      }
    
      // 定义名称和长度对应关系，加载对应操作源码
      using str_pair = std::pair<std::string, std::string>;
      const std::vector<str_pair> name_len = {
          str_pair("single", "1"),
          str_pair("pair", "2"),
          str_pair("triple", "3"),
          str_pair("quadruple", "4"),
      };
      for (const auto scalar : {"float", "int"}) {
        for (const auto& pair : name_len) {
          at::jit::TemplateEnv env;
          env.s("Scalar", scalar);
          env.s("name", pair.first);
          env.s("Length", pair.second);
          loadSource(_ntuple_ops.format(env), "aten");
        }
      }
    
      // 加载 number 和 Tensor 类型的操作源码
      for (auto rhs : {"number", "Tensor"}) {
        at::jit::TemplateEnv env;
        env.s("Rhs_Type", rhs);
        loadSource(floordiv.format(env), "aten");
      }
    
      // 加载 aten_ops 和 aten_ops_additional 的操作源码
      loadSource(aten_ops, "aten");
      loadSource(aten_ops_additional, "aten");
    
      // 加载 tensor_properties 操作源码，放置在 prim 命名空间下
      loadSource(tensor_properties, "prim");
    }
    
    // 状态枚举，标识内建函数加载状态
    enum { UNINITIALIZED, INTIIALIZING, INITIALIZED } state = UNINITIALIZED;
    // 递归互斥锁，确保线程安全
    std::recursive_mutex mutex;
    // 模块列表，存储编译单元的共享指针
    std::vector<std::shared_ptr<CompilationUnit>> modules;
    // 内建函数名称到函数列表的映射
    std::unordered_map<Symbol, std::vector<Function*>> builtins_by_name_;
};

// 定义一个名为 getAllBuiltinFunctionsFor 的函数，返回一个指向 Function 对象的向量的引用，
// 该函数接受一个名为 name 的符号作为参数
const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name) {
  // 定义一个静态的 BuiltinFunctionRegistry 对象 registry
  static BuiltinFunctionRegistry registry;
  // 调用 registry 对象的 getAllBuiltinFunctionsFor 方法，传入参数 name，并返回结果
  return registry.getAllBuiltinFunctionsFor(name);
}

// 声明命名空间 torch::jit 结束
} // namespace torch::jit
```