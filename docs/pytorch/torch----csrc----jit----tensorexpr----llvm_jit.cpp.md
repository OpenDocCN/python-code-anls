# `.\pytorch\torch\csrc\jit\tensorexpr\llvm_jit.cpp`

```
#ifdef TORCH_ENABLE_LLVM
// 如果定义了 TORCH_ENABLE_LLVM，则编译以下代码块

#include <c10/macros/Macros.h>
// 包含 C10 库的宏定义

#include <torch/csrc/jit/tensorexpr/external_functions.h>
#include <torch/csrc/jit/tensorexpr/intrinsic_symbols.h>
#include <torch/csrc/jit/tensorexpr/llvm_jit.h>
// 包含 Torch 的 JIT（即时编译）相关头文件

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
// 忽略 "-Wsuggest-override" 警告，并将当前诊断状态推入堆栈

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
// 包含 LLVM 的 ExecutionEngine 和 JITSymbol 相关头文件

C10_DIAGNOSTIC_POP()
// 弹出先前推入的诊断状态

#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
// 包含 LLVM 的 Orc 模块相关头文件，用于即时编译

// llvm::SCEVPredicate has virtual function but non-virtual destructor
// https://github.com/llvm/llvm-project/blob/c1a0a213378a458fbea1a5c77b315c7dce08fd05/llvm/include/llvm/Analysis/ScalarEvolution.h#L198
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
// 忽略 "-Wnon-virtual-dtor" 警告，并将当前 GCC 诊断状态推入堆栈

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#pragma GCC diagnostic pop
// 弹出先前推入的 GCC 诊断状态，并包含 LLVM 的 LLJIT 头文件

#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/SymbolStringPool.h>
#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Mangler.h>
#include <llvm/Support/CFGUpdate.h>
#include <llvm/Support/DynamicLibrary.h>
#if LLVM_VERSION_MAJOR >= 18
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
// 包含 LLVM 和相关依赖库的头文件

#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
// 包含 Torch 的外部函数注册相关头文件

#include <c10/util/Half.h>
// 包含 C10 库的 Half 类定义

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
// 包含标准 C++ 库头文件

using namespace torch::jit::tensorexpr;
// 使用 Torch 的 JIT 与 TensorExpr 命名空间

template <typename T>
static llvm::JITTargetAddress toAddress(T* Ptr) {
  return static_cast<llvm::JITTargetAddress>(reinterpret_cast<uintptr_t>(Ptr));
}
// 定义模板函数 toAddress，将指针转换为 JITTargetAddress 类型

// 获取主机的子目标特性
static llvm::SubtargetFeatures getHostSubtargetFeatures() {
  llvm::SubtargetFeatures subtargetFeatures;
  llvm::StringMap<bool> featureMap;
  llvm::sys::getHostCPUFeatures(featureMap);
  for (auto& feature : featureMap) {
    subtargetFeatures.AddFeature(feature.first(), feature.second);
  }
  return subtargetFeatures;
}
// 定义函数 getHostSubtargetFeatures，获取主机的子目标特性

// 使用主机的三元组创建 JTMB。如果未提供 CPU 和 attrs，则默认使用主机的值。
static llvm::orc::JITTargetMachineBuilder makeJTMBFromHost(
    std::optional<std::string> cpu,
    std::optional<std::string> attrs) {
  llvm::orc::JITTargetMachineBuilder JTMB(
      (llvm::Triple(llvm::sys::getProcessTriple())));
  JTMB.setCPU(cpu.value_or(llvm::sys::getHostCPUName().str()));
  if (attrs) {
    std::vector<std::string> features;
    llvm::SubtargetFeatures::Split(features, *attrs);
    JTMB.addFeatures(features);
  } else {
    JTMB.addFeatures(getHostSubtargetFeatures().getFeatures());
  }
  return JTMB;
}
// 定义函数 makeJTMBFromHost，使用主机的三元组创建 JITTargetMachineBuilder

// 使用给定的三元组创建 JTMB。如果未提供 CPU 和 attrs，则不设置。


这段代码主要是关于使用 LLVM 进行即时编译的相关设置和功能定义，涉及到了 LLVM 的库和一些 Torch 的 JIT 相关的内容。
// 根据给定的三元组创建一个 JITTargetMachineBuilder 对象
static llvm::orc::JITTargetMachineBuilder makeJTMBFromTriple(
    const std::string& triple,
    std::optional<std::string> cpu,
    std::optional<std::string> attrs) {
  // 使用给定的三元组创建 JITTargetMachineBuilder 对象
  llvm::orc::JITTargetMachineBuilder JTMB((llvm::Triple(triple)));
  // 如果提供了 CPU 参数，则设置 JITTargetMachineBuilder 的 CPU
  if (cpu) {
    JTMB.setCPU(*cpu);
  }
  // 如果提供了属性参数，则将其拆分并添加到 JITTargetMachineBuilder 的特性列表中
  if (attrs) {
    std::vector<std::string> features;
    llvm::SubtargetFeatures::Split(features, *attrs);
    JTMB.addFeatures(features);
  }
  // 返回配置好的 JITTargetMachineBuilder 对象
  return JTMB;
}

// 根据给定的三元组或者主机信息创建 JITTargetMachineBuilder 对象
static llvm::orc::JITTargetMachineBuilder makeTargetMachineBuilder(
    std::optional<std::string> triple,
    std::optional<std::string> cpu,
    std::optional<std::string> attrs) {
  // 根据是否提供了三元组参数选择不同的创建方式
  auto JTMB = triple ? makeJTMBFromTriple(*triple, cpu, attrs)
                     : makeJTMBFromHost(cpu, attrs);
  
  // 设置代码生成优化级别为默认值
#if LLVM_VERSION_MAJOR >= 18
  JTMB.setCodeGenOptLevel(llvm::CodeGenOptLevel::Default);
#else
  JTMB.setCodeGenOptLevel(llvm::CodeGenOpt::Default);
#endif

  // 设置浮点运算融合策略为快速模式
  JTMB.getOptions().AllowFPOpFusion = llvm::FPOpFusion::Fast;

  // 返回配置好的 JITTargetMachineBuilder 对象
  return JTMB;
}

// 向给定的 JITDylib 中注册内置函数
static void registerIntrinsics(
    llvm::orc::JITDylib& JD,
    llvm::orc::MangleAndInterner& Mangle,
    std::unordered_set<std::string>& intrinsics) {
  using namespace llvm;
  using namespace llvm::orc;

  // 匿名函数，用于生成符号映射的条目
  auto entry = [&](const char* name, auto ptr) -> SymbolMap::value_type {
#if LLVM_VERSION_MAJOR >= 17
    return {Mangle(name), {ExecutorAddr(toAddress(ptr)), JITSymbolFlags::None}};
#else
    return {Mangle(name), {toAddress(ptr), JITSymbolFlags::None}};
#endif
  };

  SymbolMap symbols;
  
  // 遍历获取所有内置函数符号，并插入符号映射
  for (auto const& sym : getIntrinsicSymbols()) {
    symbols.insert(entry(sym.symbol, sym.address));
    intrinsics.insert(sym.symbol);
  }
  // 定义所有符号为绝对符号，并将其注册到 JITDylib 中
  assertSuccess(JD.define(absoluteSymbols(symbols)));

  // 遍历并注册 NNCF 函数注册表中的函数符号
  for (auto& kv : getNNCFunctionRegistry()) {
    assertSuccess(
        JD.define(absoluteSymbols({entry(kv.first.c_str(), kv.second)})));
  }
  
  // 注册特定的函数符号到 JITDylib 中
  assertSuccess(JD.define(
      absoluteSymbols({entry("DispatchParallel", DispatchParallel)})));
  assertSuccess(
      JD.define(absoluteSymbols({entry("nnc_aten_free", nnc_aten_free)})));
}

namespace llvm {
namespace orc {

// 基于 LLVM Kaleidoscope JIT 教程的轻微修改实现
// https://llvm.org/docs/tutorial/BuildingAJIT1.html
#if LLVM_VERSION_MAJOR >= 9
class TORCH_API PytorchLLVMJITImpl {
 private:
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<LLJIT> LLJ;
  std::unordered_set<std::string> intrinsics;

 public:
  // 构造函数，根据给定的三元组、CPU 和属性创建 PytorchLLVMJITImpl 对象
  PytorchLLVMJITImpl(
      std::optional<std::string> triple,
      std::optional<std::string> cpu,
      std::optional<std::string> attrs)
      : TM(assertSuccess(makeTargetMachineBuilder(triple, cpu, attrs)
                             .createTargetMachine())),
        LLJ(assertSuccess(
            LLJITBuilder()
                .setJITTargetMachineBuilder(
                    makeTargetMachineBuilder(triple, cpu, attrs))
#if LLVM_VERSION_MAJOR >= 17
                .setObjectLinkingLayerCreator([&](ExecutionSession& ES,
                                                  const Triple& TT) {
                  // 创建并返回一个 ObjectLinkingLayer 对象，用于对象链接
                  return std::make_unique<ObjectLinkingLayer>(
                      ES,
                      assertSuccess(jitlink::InProcessMemoryManager::Create()));
                })
#endif
                .create())) {
    auto ProcSymbolsGenerator =
        assertSuccess(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            LLJ->getDataLayout().getGlobalPrefix()));
    auto& JD = LLJ->getMainJITDylib();
#if LLVM_VERSION_MAJOR == 9
    // 设置当前进程符号生成器到主 JIT 动态库
    JD.setGenerator(std::move(ProcSymbolsGenerator));
#else
    // 添加当前进程符号生成器到主 JIT 动态库
    JD.addGenerator(std::move(ProcSymbolsGenerator));
#endif

    // 处理平台特定的符号重整
    MangleAndInterner Mangle(LLJ->getExecutionSession(), LLJ->getDataLayout());

    // 注册内置函数的实现
    registerIntrinsics(JD, Mangle, intrinsics);
  }

  // 将模块添加到编译层
  void addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C) {
    assertSuccess(
        LLJ->addIRModule(ThreadSafeModule(std::move(M), std::move(C))),
        "Failed to add module to compile layer");
  }

  // 查找指定名称的符号
  JITSymbol findSymbol(const std::string Name) {
#if LLVM_VERSION_MAJOR >= 15
    // 从 LLJIT 中查找指定名称的符号，并返回一个 JITSymbol 对象
    // llvm-15 之后返回的是地址而不是符号，这里将其封装在一个假的 JITSymbol 中返回
    auto result = assertSuccess(LLJ->lookup(Name));
    return JITSymbol(result.getValue(), JITSymbolFlags());
#else
    // 从 LLJIT 中查找指定名称的符号，并返回一个 JITSymbol 对象
    return assertSuccess(LLJ->lookup(Name));
#endif
  }

  // 判断是否存在指定名称的符号
  bool hasSymbol(const std::string& Name) {
    // 检查内置函数集合中是否存在指定名称的符号
    return intrinsics.find(Name) != intrinsics.end();
  }

  // 获取目标机器对象
  TargetMachine& getTargetMachine() {
    return *TM;
  }

  // 获取数据布局对象
  const DataLayout& getDataLayout() {
    return LLJ->getDataLayout();
  }
};
#elif LLVM_VERSION_MAJOR == 8 && LLVM_VERSION_PATCH == 20181009
class TORCH_API PytorchLLVMJITImpl {
 private:
  ExecutionSession ES;  // 执行会话对象，管理 JIT 的执行过程
  std::shared_ptr<SymbolResolver> Resolver;  // 共享的符号解析器指针，用于解析符号
  std::unique_ptr<TargetMachine> TM;  // 独特的目标机器指针，用于 JIT 的目标机器描述
  const DataLayout DL;  // 数据布局对象，描述数据的内存布局
  RTDyldObjectLinkingLayer ObjectLayer;  // 运行时动态链接对象层，处理目标代码的加载和链接
  IRCompileLayer<decltype(ObjectLayer), SimpleCompiler> CompileLayer;  // IR 编译层，将 IR 模块编译成目标代码
  std::unordered_set<std::string> intrinsics;  // 无序集合，存储内部函数名称

 public:
  PytorchLLVMJITImpl(
      std::optional<std::string> triple,
      std::optional<std::string> cpu,
      std::optional<std::string> attrs)
      : Resolver(createLegacyLookupResolver(
            ES,
            // 创建符号解析器，用于查找 JIT 符号
            [this](const std::string& Name) -> JITSymbol {
              if (auto Sym = CompileLayer.findSymbol(Name, false)) {
                return Sym;
              } else if (auto Err = Sym.takeError()) {
                return std::move(Err);
              }
              // 如果未在编译层找到符号，尝试从当前进程中获取符号地址
              if (auto SymAddr =
                      RTDyldMemoryManager::getSymbolAddressInProcess(Name)) {
                return JITSymbol(SymAddr, JITSymbolFlags::Exported);
              }
              MangleAndInterner Mangle(ES, DL);
              return assertSuccess(
                  lookup({&ES.getMainJITDylib()}, Mangle(Name)));
            },
            // 错误处理函数，确保符号查找成功
            [](Error Err) {
              assertSuccess(std::move(Err), "lookupFlags failed");
            })),
        // 创建目标机器对象，确保创建成功
        TM(assertSuccess(makeTargetMachineBuilder(triple, cpu, attrs)
                             .createTargetMachine())),
        // 创建数据布局对象，描述目标机器的数据布局
        DL(TM->createDataLayout()),
        // 创建对象层，用于运行时加载目标代码
        ObjectLayer(
            ES,
            [this](VModuleKey) {
              return RTDyldObjectLinkingLayer::Resources{
                  std::make_shared<SectionMemoryManager>(), Resolver};
            }),
        // 创建编译层，将 IR 模块编译成目标代码
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)) {
    auto& JD = ES.getMainJITDylib();
    MangleAndInterner Mangle(ES, DL);
    // 注册内部函数，确保注册成功
    registerIntrinsics(JD, Mangle, intrinsics);
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  // 获取目标机器对象的引用
  TargetMachine& getTargetMachine() {
    return *TM;
  }

  // 向 JIT 添加模块，并分配新的 VModuleKey
  void addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C) {
    auto K = ES.allocateVModule();
    // 将模块添加到编译层，确保添加成功
    assertSuccess(
        CompileLayer.addModule(K, std::move(M)),
        "Failed to add module to compile layer");
  }

  // 查找符号并返回 JITSymbol 对象
  JITSymbol findSymbol(const std::string Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CompileLayer.findSymbol(MangledNameStream.str(), true);
  }

  // 检查是否存在特定名称的内部函数
  bool hasSymbol(const std::string& Name) {
    return intrinsics.find(Name) != intrinsics.end();
  }

  // 获取特定符号的地址
  JITTargetAddress getSymbolAddress(const std::string Name) {
    return assertSuccess(findSymbol(Name).getAddress());
  }

  // 移除指定的模块
  void removeModule(VModuleKey K) {
    assertSuccess(CompileLayer.removeModule(K));
  }

  // 获取数据布局对象的引用
  const DataLayout& getDataLayout() {
    return DL;
  }
};
// 构造函数：根据给定的三个可选参数（目标三元组、CPU 类型、特性属性），创建 PytorchLLVMJIT 对象
PytorchLLVMJIT::PytorchLLVMJIT(
    std::optional<std::string> triple,
    std::optional<std::string> cpu,
    std::optional<std::string> attrs)
    : impl_(std::make_unique<PytorchLLVMJITImpl>(triple, cpu, attrs)) {}

// 析构函数：使用默认行为来销毁 PytorchLLVMJIT 对象
PytorchLLVMJIT::~PytorchLLVMJIT() = default;

// 向 JIT 实现对象中添加模块和上下文
void PytorchLLVMJIT::addModule(
    std::unique_ptr<Module> M,
    std::unique_ptr<LLVMContext> C) {
  impl_->addModule(std::move(M), std::move(C));
}

// 在 JIT 实现对象中查找给定名称的符号并返回其 JITSymbol
JITSymbol PytorchLLVMJIT::findSymbol(const std::string Name) {
  return impl_->findSymbol(std::move(Name));
}

// 检查 JIT 实现对象中是否存在给定名称的符号
bool PytorchLLVMJIT::hasSymbol(const std::string& Name) {
  return impl_->hasSymbol(Name);
}

// 获取 JIT 实现对象中的目标机器 TargetMachine 的引用
TargetMachine& PytorchLLVMJIT::getTargetMachine() {
  return impl_->getTargetMachine();
}

// 获取 JIT 实现对象中的数据布局 DataLayout 的引用
const DataLayout& PytorchLLVMJIT::getDataLayout() {
  return impl_->getDataLayout();
}

// 仅在非发布模式（NDEBUG 未定义）下定义的函数，用于打印 LLVM 控制流图的更新
#if !defined(NDEBUG)
void dumpCFG(const llvm::cfg::Update<llvm::BasicBlock*>& update) {
  // XXX: 此方法调用仅用于适应 gcov 构建。当 NDEBUG 未设置时，`dump` 方法有条件地定义，
  // 因此如果尝试将调试模式的 PyTorch 与优化模式的 LLVM 链接在一起，该符号将是未定义的。
  update.dump();
}
#endif

} // end namespace orc
} // end namespace llvm

#endif // TORCH_ENABLE_LLVM
```