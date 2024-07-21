# `.\pytorch\torch\csrc\jit\codegen\fuser\cpu\fused_kernel.cpp`

```
#ifdef _MSC_VER
// 定义获取临时路径的函数，用于在 Windows 下获取临时文件目录路径
static const std::string getTempPath() {
  wchar_t lpTempPathBuffer[MAX_PATH];

  // 调用 Windows API 获取临时文件目录路径
  DWORD dwRetVal = GetTempPathW(
      MAX_PATH, // 缓冲区长度
      lpTempPathBuffer); // 存放路径的缓冲区

  // 检查 API 调用是否成功
  TORCH_CHECK(dwRetVal < MAX_PATH && dwRetVal != 0, "GetTempPath failed.");

  // 将 wchar_t 类型的路径转换为 std::string 返回
  return std::string(c10::u16u8(lpTempPathBuffer));
}

// 在 Windows 下，定义全局变量保存临时路径
static const std::string temp_dir = getTempPath();

// 定义生成动态链接库文件名的模板路径，在临时路径下加上固定前缀和后缀
static const std::string so_template = temp_dir + "pytorch_fuserXXXXXX.dll";

// 定义生成 C++ 源文件名的模板路径，在临时路径下加上固定前缀和后缀
static const std::string cpp_template = temp_dir + "pytorch_fuserXXXXXX.cpp";

// 定义用于检查程序是否存在的命令字符串模板
static const std::string check_exists_string = "where ${program} > nul 2> nul";

// 定义存储环境变量的向量
static std::vector<std::wstring> env_list;

// 定义动态链接库文件后缀的长度
constexpr int so_suffix_len = 4;

// 定义 C++ 源文件后缀的长度
constexpr int cpp_suffix_len = 4;
#else
// 在非 Windows 系统下，定义固定的临时路径及文件名模板
static const std::string so_template = "/tmp/pytorch_fuserXXXXXX.so";
static const std::string cpp_template = "/tmp/pytorch_fuserXXXXXX.cpp";

// 定义用于检查程序是否存在的命令字符串模板
static const std::string check_exists_string = "which ${program} > /dev/null";

// 定义动态链接库文件后缀的长度
constexpr int so_suffix_len = 3;

// 定义 C++ 源文件后缀的长度
constexpr int cpp_suffix_len = 4;
#endif


这段代码定义了在不同操作系统下生成临时文件路径和文件名的模板，以及用于检查程序是否存在的命令字符串模板。
    return;
  }

  // 使用 `vswhere` 获取 Visual Studio 2017 的安装路径
  cmd = L"\"" + std::wstring(root) +
      L"\\Microsoft Visual Studio\\Installer\\vswhere.exe\""
      L" -latest -prerelease -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath";
  // 执行命令获取输出结果
  exec_out = exec(cmd);
  // 如果执行失败，则返回
  if (!exec_out) {
    return;
  }
  // 获取执行结果中的安装路径
  path = *exec_out;
  // 去除路径末尾的空白字符
  rtrim(path);

  // 检查是否存在激活脚本 `vcvarsall.bat`
  path += L"\\VC\\Auxiliary\\Build";
  struct _stati64 st;
  // 获取目录信息，并检查是否为目录
  if (_wstati64(path.c_str(), &st) == -1 || !(st.st_mode & _S_IFDIR)) {
    return;
  }
  // 组合完整的 `vcvarsall.bat` 文件路径
  path += L"\\vcvarsall.bat";
  // 检查文件是否存在
  if (_waccess(path.c_str(), 0) == -1) {
    return;
  }

  // 确定当前平台是 x64 还是 x86
  if (sizeof(void*) == 8) {
    vcruntime_plat = L"x64";
  } else {
    vcruntime_plat = L"x86";
  }

  // 激活 VS 开发环境后，获取环境变量
  cmd = L"\"" + path + L"\" " + vcruntime_plat + L">NUL && set";
  // 执行命令获取输出结果
  exec_out = exec(cmd);
  // 如果执行失败，则返回
  if (!exec_out) {
    return;
  }
  // 获取执行结果中的环境变量信息
  envvars = *exec_out;

  // 将获取的环境变量信息存入环境变量列表中
  std::wistringstream f(envvars);
  std::wstring envvar;
  // 逐行读取环境变量，并存入列表
  while (getline(f, envvar, L'\n')) {
    env_list.push_back(envvar);
  }
// intptr_t类型的函数run，接收一个const引用cmd作为参数
intptr_t run(const std::string& cmd) {
  // 获取环境变量COMSPEC的值，即cmd.exe的路径
  wchar_t* comspec = _wgetenv(L"COMSPEC");
  // 如果COMSPEC为空，则默认使用"C:\\Windows\\System32\\cmd.exe"
  if (!comspec) {
    comspec = L"C:\\Windows\\System32\\cmd.exe";
  }
  
  // 将cmd字符串转换为宽字符编码
  auto wCmd = c10::u8u16(cmd);
  // 构造命令行参数数组，以执行命令
  const wchar_t* a[] = {L"/c", wCmd.c_str(), nullptr};
  
  // 构造环境变量数组
  std::vector<const wchar_t*> e;
  if (!env_list.empty()) {  // 如果env_list不为空，则添加每个字符串的指针，并以nullptr结尾
    for (auto& s : env_list) {
      e.push_back(s.c_str());
    }
    e.push_back(nullptr);
  }
  
  // 调用_wspawnve函数执行命令，等待其完成，返回进程的退出码
  intptr_t r = _wspawnve(_P_WAIT, comspec, a, e.data());
  return r;
}
    # 如果当前系统支持 AVX512 指令集，返回编译器参数 "/arch:AVX512"
    return "/arch:AVX512";
  # 如果当前系统支持 AVX2 指令集，返回编译器参数 "/arch:AVX2"
  } else if (__isa_available >= 5) {
    return "/arch:AVX2";
  # 如果当前系统支持 AVX 指令集，返回编译器参数 "/arch:AVX"
  } else if (__isa_available >= 4) {
    return "/arch:AVX";
  # 如果系统不支持任何高级向量指令集，返回空字符串
  } else {
    return "";
  }
static const std::string arch_flags = getArchFlags();
// 获取与架构相关的编译标志

static const std::string compile_string = "cd /D \"" + temp_dir +
    "\" && "
    "${cxx} /nologo /MD /O2 " +
    arch_flags +
    " /LD /EHsc "
    "${fopenmp} \"${cpp_file}\" /link /out:\"${so_file}\"";
// 定义编译命令字符串，包括设置目录、编译器选项、OpenMP支持等

#else
static const std::string compile_string =
    "\"${cxx}\" -O3 -g "
#ifndef __PPC64__
//  "-march=native "
#endif
    "-std=c++17 -fPIC ${fopenmp} -shared \"${cpp_file}\" -o \"${so_file}\" -lm";
// 定义另一种编译命令字符串，针对不同平台和条件进行设置

#endif

static void runCompiler(
    const std::string& cpp_file,
    const std::string& so_file) {
  auto& config = getConfig();
  TORCH_CHECK(
      !config.cxx.empty(),
      "Failed to compile a fused CPU kernel: Compiler not found");
  at::jit::TemplateEnv env;
  env.s("cxx", config.cxx);
  env.s("fopenmp", config.openmp ? config.openmp_flags : "");
  env.s("cpp_file", cpp_file);
  env.s("so_file", so_file);
  std::string result = format(compile_string, env);
#ifdef _MSC_VER
  intptr_t r = run(result);
#else
  int r = system(result.c_str());
#endif
  if (config.openmp && r != 0) {
    std::cerr
        << "warning: pytorch jit fuser failed to compile with openmp, trying without it...\n";
    config.openmp = false; // disable for future compiles
    return runCompiler(cpp_file, so_file);
  }
  TORCH_CHECK(r == 0, "Failed to compile a fused CPU kernel");
}
// 编译器函数，根据指定的CPP文件和SO文件路径编译代码，并处理OpenMP编译失败的情况

#ifdef _MSC_VER
static const std::string disas_string =
    "dumpbin /DISASM:NOBYTES \"${so_file}\"";
#else
static const std::string disas_string = "objdump -M  intel -d \"${so_file}\"";
#endif

static void disas(const std::string& so_file) {
  at::jit::TemplateEnv env;
  env.s("so_file", so_file);
  std::string cmd = format(disas_string, env);
  int r = system(cmd.c_str());
  AT_ASSERT(r == 0);
}
// 反汇编函数，根据SO文件路径执行反汇编命令，输出汇编代码

FusedKernelCPU::FusedKernelCPU(
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random)
    : FusedKernel(
          std::move(name),
          std::move(code),
          std::move(input_desc),
          std::move(output_desc),
          std::move(chunk_desc),
          std::move(concat_desc),
          has_random) {
  TempFile so_file(so_template, so_suffix_len);
  TempFile cpp_file(cpp_template, cpp_suffix_len);
  cpp_file.write(code_);
  cpp_file.sync();
#ifdef _MSC_VER
  so_file.close();
  cpp_file.close();
#endif
  runCompiler(cpp_file.name(), so_file.name());
  if (debugFuser() >= 2)
    disas(so_file.name());
  so_lib = std::make_unique<at::DynamicLibrary>(so_file.name().c_str());
#pragma GCC diagnostic ignored "-Wpedantic"
  kernel =
      reinterpret_cast<void (*)(uint32_t, void**)>(so_lib->sym(name_.c_str()));
#pragma GCC diagnostic pop
}
// FusedKernelCPU构造函数，编译给定的代码并加载动态链接库，获取函数指针以执行融合内核

static std::shared_ptr<FusedKernel> createFusionKernel(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random) {
// 创建融合内核的函数，根据给定的设备、名称、代码和描述生成相应的内核对象
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random) {


# 创建一个名为 FusedKernelCPU 的对象，并返回其指针
return std::make_shared<FusedKernelCPU>(
    # 将参数 name 移动到 FusedKernelCPU 对象的构造函数中
    std::move(name),
    # 将参数 code 移动到 FusedKernelCPU 对象的构造函数中
    std::move(code),
    # 将参数 input_desc 移动到 FusedKernelCPU 对象的构造函数中
    std::move(input_desc),
    # 将参数 output_desc 移动到 FusedKernelCPU 对象的构造函数中
    std::move(output_desc),
    # 将参数 chunk_desc 移动到 FusedKernelCPU 对象的构造函数中
    std::move(chunk_desc),
    # 将参数 concat_desc 移动到 FusedKernelCPU 对象的构造函数中
    std::move(concat_desc),
    # 将参数 has_random 传递给 FusedKernelCPU 对象的构造函数
    has_random);
}
// 结束了名为torch的命名空间
} // namespace torch
// 结束了名为jit的命名空间
} // namespace jit
// 结束了名为fuser的命名空间
} // namespace fuser
// 结束了名为cpu的命名空间
} // namespace cpu
```