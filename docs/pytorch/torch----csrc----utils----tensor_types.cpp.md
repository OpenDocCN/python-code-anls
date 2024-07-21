# `.\pytorch\torch\csrc\utils\tensor_types.cpp`

```py
// 包含 Python.h 头文件，用于与 Python 解释器交互
#include <Python.h>

// 引入 PyTorch 的张量类型定义头文件
#include <torch/csrc/utils/tensor_types.h>

// 引入 ATen 库的上下文和格式化定义头文件
#include <ATen/Context.h>
#include <ATen/Formatting.h>

// 引入 PyTorch 异常处理定义头文件
#include <torch/csrc/Exceptions.h>

// 引入 PyTorch 自动微分生成的变量类型定义头文件
#include <torch/csrc/autograd/generated/VariableType.h>

// 引入 PyTorch Python 张量封装定义头文件
#include <torch/csrc/tensor/python_tensor.h>

// 引入 C10 实用工具中的 CallOnce 头文件
#include <c10/util/CallOnce.h>

// 引入 C++ 标准库头文件
#include <algorithm>
#include <sstream>
#include <unordered_map>

// 使用 at 命名空间
using namespace at;

// 定义在 torch::utils 命名空间中的静态函数
namespace torch::utils {

// 解析 PrivateUse1 后端名称的静态函数
static const char* parse_privateuseone_backend(bool is_sparse = false) {
  // 静态变量，包含 PrivateUse1 后端名称
  static std::string backend_name = "torch." + get_privateuse1_backend();
  static std::string sparse_backend_name = backend_name + ".sparse";
  // 根据是否稀疏选择返回相应的名称字符串指针
  return is_sparse == false ? backend_name.c_str() : sparse_backend_name.c_str();
}

// 将 Backend 枚举转换为对应字符串的函数
const char* backend_to_string(const at::Backend& backend) {
  switch (backend) {
    // 不同 Backend 对应的字符串表示
    case at::Backend::CPU:
      return "torch";
    case at::Backend::CUDA:
      return "torch.cuda";
    case at::Backend::XPU:
      return "torch.xpu";
    case at::Backend::IPU:
      return "torch.ipu";
    case at::Backend::SparseCPU:
      return "torch.sparse";
    case at::Backend::SparseCUDA:
      return "torch.cuda.sparse";
    case at::Backend::SparseXPU:
      return "torch.xpu.sparse";
    case at::Backend::QuantizedCPU:
      return "torch.quantized";
    case at::Backend::HPU:
      return "torch.hpu";
    case at::Backend::MPS:
      return "torch.mps";
    case at::Backend::MTIA:
      return "torch.mtia";
    case at::Backend::PrivateUse1:
      // 调用解析 PrivateUse1 后端名称的静态函数，返回相应字符串指针
      return parse_privateuseone_backend();
    case at::Backend::SparsePrivateUse1:
      // 调用解析 PrivateUse1 后端名称的静态函数（稀疏版本），返回相应字符串指针
      return parse_privateuseone_backend(true);
    case at::Backend::Lazy:
      return "torch.lazy";
    case at::Backend::XLA:
      return "torch.xla";
    case at::Backend::Meta:
      return "torch.meta";
    default:
      // 未实现的 Backend 引发错误
      AT_ERROR("Unimplemented backend ", backend);
  }
}

// 将 TensorOptions 转换为字符串表示的函数
std::string options_to_string(const at::TensorOptions& options) {
  std::ostringstream ss;
  // 拼接 Backend 字符串和数据类型的字符串表示，形成完整的 Tensor 类型字符串
  ss << backend_to_string(options.backend()) << "."
     << toString(at::typeMetaToScalarType(options.dtype())) << "Tensor";
  return ss.str();
}

// 将 DeprecatedTypeProperties 转换为字符串表示的函数
std::string type_to_string(const at::DeprecatedTypeProperties& type) {
  std::ostringstream ss;
  // 拼接 Backend 字符串、标量类型字符串表示，形成完整的 Tensor 类型字符串
  ss << backend_to_string(type.backend()) << "." << toString(type.scalarType())
     << "Tensor";
  return ss.str();
}

// 结束 torch::utils 命名空间
}
// 根据给定的字符串创建并返回一个包含 TensorOptions 的对象
at::TensorOptions options_from_string(const std::string& str) {
  // 静态变量，表示CUDA类型的前缀
  static std::string cuda_prefix("torch.cuda.");
  // 静态变量，表示XPU类型的前缀
  static std::string xpu_prefix("torch.xpu.");
  // 静态变量，表示私有用户类型的前缀
  static std::string privateUser_prefix(
      std::string(parse_privateuseone_backend()) + ".");
  // 静态变量，用于确保 CPU 相关操作只执行一次
  static c10::once_flag cpu_once;
  // 静态变量，用于确保 CUDA 相关操作只执行一次
  static c10::once_flag cuda_once;
  // 静态变量，用于确保 XPU 相关操作只执行一次
  static c10::once_flag xpu_once;
  // 静态变量，用于确保私有用户类型相关操作只执行一次
  static c10::once_flag privateUser1_once;
  // 静态变量，存储所有 CPU 类型的映射关系
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cpu_map;
  // 静态变量，存储所有 XPU 类型的映射关系
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> xpu_map;
  // 静态变量，存储所有 CUDA 类型的映射关系
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cuda_map;
  // 静态变量，存储所有私有用户类型的映射关系
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> privateUser1_map;

  // 映射指针，用于根据前缀选择不同的映射表
  const std::unordered_map<std::string, at::DeprecatedTypeProperties*>* map = nullptr;

  // 检查是否为默认的 torch.Tensor 类型
  if (str == "torch.Tensor") {
    auto backend = dispatchKeyToBackend(torch::tensors::get_default_dispatch_key());
    auto scalar_type = torch::tensors::get_default_scalar_type();
    // 返回默认 tensor 类型的 TensorOptions
    return getDeprecatedTypeProperties(backend, scalar_type).options();
  }

  // 检查是否以 torch.cuda. 开头
  if (std::mismatch(cuda_prefix.begin(), cuda_prefix.end(), str.begin()).first == cuda_prefix.end()) {
    // 如果是 CUDA 类型
    // 确保 CUDA 映射表只初始化一次
    c10::call_once(cuda_once, []() {
      for (auto type : autograd::VariableType::allCUDATypes()) {
        cuda_map.emplace(type_to_string(*type), type);
      }
    });
    // 使用 CUDA 映射表
    map = &cuda_map;
  } else if (std::mismatch(xpu_prefix.begin(), xpu_prefix.end(), str.begin()).first == xpu_prefix.end()) {
    // 如果是 XPU 类型
    // 确保 XPU 映射表只初始化一次
    c10::call_once(xpu_once, []() {
      for (auto type : autograd::VariableType::allXPUTypes()) {
        xpu_map.emplace(type_to_string(*type), type);
      }
    });
    // 使用 XPU 映射表
    map = &xpu_map;
  } else if (std::mismatch(privateUser_prefix.begin(), privateUser_prefix.end(), str.begin()).first == privateUser_prefix.end()) {
    // 如果是私有用户类型
    // 确保私有用户类型映射表只初始化一次
    c10::call_once(privateUser1_once, []() {
      for (auto type : autograd::VariableType::allPrivateUser1Types()) {
        privateUser1_map.emplace(type_to_string(*type), type);
      }
    });
    // 使用私有用户类型映射表
    map = &privateUser1_map;
  } else {
    // 默认为 CPU 类型
    // 确保 CPU 映射表只初始化一次
    c10::call_once(cpu_once, []() {
      for (auto type : autograd::VariableType::allCPUTypes()) {
        cpu_map.emplace(type_to_string(*type), type);
      }
    });
    // 使用 CPU 映射表
    map = &cpu_map;
  }

  // 在选择的映射表中查找给定类型的属性
  auto it = map->find(str);
  // 如果未找到，则报错
  TORCH_CHECK_VALUE(it != map->end(), "invalid type: '", str, "'");
  // 返回找到的类型的 TensorOptions
  return it->second->options();
}
std::vector<std::pair<Backend, ScalarType>> all_declared_types() {
  // 返回一个空的向量，用于存储所有声明的类型对（Backend, ScalarType）
  std::vector<std::pair<Backend, ScalarType>> ret;

  // 不要在这里添加更多的类型。这个列表控制创建旧版张量类型，
  // 例如 torch.cuda.FloatTensor，这些类型仅用于向后兼容。
  auto backends = {
      Backend::CPU, Backend::CUDA, Backend::SparseCPU, Backend::SparseCUDA};
  auto scalar_types = {
      ScalarType::Byte,
      ScalarType::Char,
      ScalarType::Double,
      ScalarType::Float,
      ScalarType::Int,
      ScalarType::Long,
      ScalarType::Short,
      ScalarType::Half,
      ScalarType::Bool,
      ScalarType::BFloat16};

  // 遍历每个 backend 和 scalar_type 的组合
  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      // 如果是稀疏 CUDA 或稀疏 CPU 并且 scalar_type 是 Bool，则跳过
      if (scalar_type == ScalarType::Bool &&
          (backend == Backend::SparseCUDA || backend == Backend::SparseCPU)) {
        continue;
      }
      // 将当前的 backend 和 scalar_type 添加到返回向量中
      ret.emplace_back(backend, scalar_type);
    }
  }

  // 返回填充后的向量，包含所有声明的（Backend, ScalarType）对
  return ret;
}

} // namespace torch::utils
```